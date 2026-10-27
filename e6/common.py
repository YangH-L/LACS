"""Shared definitions of E6 (standard library only: fetch.py must not need numpy or pandas).

Periods, joins and bill ids follow sections 2-3 of the plan (docs/PLANS.md (lacs_e6_plan_20261025)):
  * train 2021-01-03 .. 2022-02-08, val 2022-02-09 .. 2022-06-09, test 2022-06-10 .. 2023-01-03
    (both ends inclusive);
  * a member "joins" a bill as sponsor on its introducedDate and as cosponsor on the
    sponsorshipDate; withdrawn cosponsorships do not count;
  * bills: types hr, s, hjres, sjres of the 117th Congress; integer bill id = type code * 100000 +
    number (hr 1, s 2, hjres 3, sjres 4).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

E6_DIR = Path(__file__).resolve().parent
PKG_ROOT = E6_DIR.parent
DATA_DIR = E6_DIR / "data"  # shipped inputs: network JSON, crosswalk, MANIFEST.json

CONGRESS = 117
BILL_TYPES = ("hr", "s", "hjres", "sjres")
TYPE_CODE = {"hr": 1, "s": 2, "hjres": 3, "sjres": 4}
BILL_ID_MULT = 100_000

PERIOD_NAMES = ("train", "val", "test")
PERIODS = {"train": ("2021-01-03", "2022-02-08"),
           "val": ("2022-02-09", "2022-06-09"),
           "test": ("2022-06-10", "2023-01-03")}

MIN_TRAIN_COSPONSORSHIPS = 10   # plan section 3: members need >= 10 usable training cosponsorships
THETA_FLOOR = 0.02              # plan section 3: floor of the tolerances (DW-NOMINATE units)
THETA_QUANTILES = (0.5, 0.7, 0.9)
MIN_ELIGIBLE = 400              # plan section 2: data gate
SEED = 20261025
# d <= theta comparisons accept d <= theta * (1 + TIE_RTOL): DW-NOMINATE scores have three decimals and a
# tolerance is itself a distance (or the mean of two), so exact decimal ties d == theta are common and
# must not be decided by floating-point rounding (the same relative tolerance as the (iii) check of Q2)
TIE_RTOL = 1e-9
# The search engine and the P0 instance hold positions and tolerances in float32, where an exact decimal tie
# d == theta can round either way. The geo dataset (users.csv, read by the engine, P0 export, Q1-Q4) therefore
# stores theta * (1 + ENGINE_TIE_RTOL): exact ties are accepted everywhere, as d <= theta says. Distinct
# DW-NOMINATE distances differ by >= 5e-4 (3 decimals), so nothing but exact ties is affected.
ENGINE_TIE_RTOL = 1e-6
# plus an absolute margin: float32 rounding of positions (|x| <= 1) reaches ~6e-8, more than 1e-6 * theta at the
# floor (2e-8); distinct distances still differ by >= 1e-4 (q70/q90 interpolate in steps of 1e-4)
ENGINE_TIE_ATOL = 1e-6

# file names of BILLSTATUS XML: BILLSTATUS-<congress><type><number>.xml (all eight bill types, so
# that e.g. 'hres' files can never be taken for 'hr' files)
FILE_RE = re.compile(r"^BILLSTATUS-(\d+)(hr|hres|hjres|hconres|s|sres|sjres|sconres)(\d+)\.xml$")
DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})")

LOG = logging.getLogger("e6")


# --------------------------------------------------------------- ids, dates --

def bill_id(btype: str, number: int) -> int:
    return TYPE_CODE[btype] * BILL_ID_MULT + int(number)


def split_bill_id(bid: int) -> tuple[str, int]:
    code, number = divmod(int(bid), BILL_ID_MULT)
    inv = {v: k for k, v in TYPE_CODE.items()}
    return inv[code], number


def parse_filename(name: str):
    """(congress, type, number) of a BILLSTATUS file name, or None."""
    m = FILE_RE.match(name)
    return (int(m.group(1)), m.group(2), int(m.group(3))) if m else None


def norm_date(s) -> str | None:
    """'YYYY-MM-DD' from an ISO date or timestamp string; None if absent or malformed."""
    if s is None:
        return None
    m = DATE_RE.match(str(s).strip())
    return m.group(1) if m else None


def period_of(date: str | None) -> int:
    """0 train, 1 val, 2 test, -1 outside the three periods (or no date). ISO dates compare
    lexicographically in chronological order."""
    d = norm_date(date)
    if d is None:
        return -1
    for i, name in enumerate(PERIOD_NAMES):
        lo, hi = PERIODS[name]
        if lo <= d <= hi:
            return i
    return -1


# -------------------------------------------------------------- BILLSTATUS --

def _text(el, tag: str) -> str | None:
    t = el.findtext(tag)
    if t is None:
        return None
    t = t.strip()
    return t or None


def _bool(s) -> bool | None:
    if s is None:
        return None
    s = str(s).strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    return None


def parse_billstatus(src) -> dict:
    """Bill, sponsors and cosponsors of one BILLSTATUS XML file (path or bytes).

    Both formats: the older one (<billType>, <billNumber>, an empty <sponsorshipWithdrawnDate/>
    on every cosponsor) and the newer one (<type>, <number>, the withdrawn tag absent unless
    withdrawn). Only the bill's own <sponsors>/<cosponsors> are read (direct children of <bill>;
    amendments have sponsors and cosponsors of their own deeper in the tree)."""
    if isinstance(src, (bytes, bytearray)):
        root = ET.fromstring(src)
    else:
        root = ET.parse(src).getroot()
    if root.tag != "billStatus":
        raise ValueError(f"root element <{root.tag}>, expected <billStatus>")
    bill = root.find("bill")
    if bill is None:
        raise ValueError("no <bill> element")
    btype = _text(bill, "type") or _text(bill, "billType")
    number = _text(bill, "number") or _text(bill, "billNumber")
    congress = _text(bill, "congress")
    out = {"congress": int(congress) if congress and congress.isdigit() else None,
           "type": btype.lower() if btype else None,
           "number": int(number) if number and number.isdigit() else None,
           "introduced": norm_date(_text(bill, "introducedDate")),
           "sponsors": [], "cosponsors": [], "items_without_bioguide": 0}
    for it in bill.findall("sponsors/item"):
        b = _text(it, "bioguideId")
        if b:
            out["sponsors"].append(b)
        else:
            out["items_without_bioguide"] += 1
    for it in bill.findall("cosponsors/item"):
        b = _text(it, "bioguideId")
        if not b:
            out["items_without_bioguide"] += 1
            continue
        wd = _text(it, "sponsorshipWithdrawnDate")  # absent or empty -> not withdrawn
        out["cosponsors"].append({"bioguide": b, "date": norm_date(_text(it, "sponsorshipDate")),
                                  "original": _bool(_text(it, "isOriginalCosponsor")),
                                  "withdrawn": wd is not None, "withdrawn_date": norm_date(wd)})
    return out


def billstatus_wellformed(path) -> bool:
    """True if the file parses as XML with a <billStatus><bill> structure (detects truncated
    downloads and HTML error pages)."""
    try:
        root = ET.parse(str(path)).getroot()
    except (ET.ParseError, OSError, ValueError):
        return False
    return root.tag == "billStatus" and root.find("bill") is not None


# ------------------------------------------------------------------ files --

def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _clean(o):
    """JSON-safe copy: numpy scalars/arrays to Python, NaN/inf to None, paths to str."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if hasattr(o, "tolist") and not isinstance(o, (str, bytes)):  # numpy array or scalar
        return _clean(o.tolist())
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    return o


def write_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_clean(obj), f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def setup_logging(logfile=None, level: str = "INFO") -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S",
                        handlers=handlers, force=True)
