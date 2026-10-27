#!/usr/bin/env python3
"""E6 stage e6fetch: download Voteview members and GovInfo BILLSTATUS (plan sections 2 and 10).

  python -m e6.fetch --data data/e6                       # on the server
  python -m e6.fetch --data data/e6 --offline-dir DIR     # files placed by hand / tests

Writes into --data:
  voteview/HS117_members.csv      Voteview members of the 117th Congress (if only HSall_members.csv
                                  could be obtained, it is kept and filtered to congress == 117)
  billstatus/<type>/listing.xml   the GovInfo directory listing as received (.json if only the JSON
                                  listing answered)
  billstatus/<type>/*.zip         the bulk package, when the listing has one (kept as downloaded)
  billstatus/<type>/BILLSTATUS-117<type><n>.xml   every bill (extracted from the package or
                                  downloaded one by one)
  fetch_manifest.json             URLs, times, sizes, sha256, listing and download counts, failures
  fetch.log

Per bill type (hr, s, hjres, sjres): read the directory listing
https://www.govinfo.gov/bulkdata/xml/BILLSTATUS/117/<type> (JSON listing as fallback). If it has a
.zip entry, the package is downloaded and its XML files extracted; listed XML files that the package
does not contain, or all of them if the package cannot be obtained, are downloaded one by one
(thread pool, retries with backoff, browser-like User-Agent, timeouts). Resume: files already present
and well-formed are not downloaded again. The run is complete when every listed XML file was
obtained (and, for a listing without XML entries, its package); otherwise the manifest records the
failures and the exit status is 2 (plan section 10: stop and report). Exit 1 = unexpected error.

--offline-dir DIR uses files placed by hand instead of the network, with the same checks:
  DIR/HS117_members.csv (or HSall_members.csv; also accepted under DIR/voteview/)
  DIR/billstatus/<type>/ (or DIR/BILLSTATUS/117/<type>/) holding listing.xml or listing.json (the
      saved directory listing; required unless --no-listing-ok, which takes the folder's contents as
      the listing and records completeness as unverified), the package and/or the XML files.

Only the standard library and, when installed, requests are used (urllib otherwise).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import json
import re
import logging
import os
import random
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # also works under python -I
    sys.path.insert(0, str(ROOT))

from e6.common import (BILL_TYPES, CONGRESS, billstatus_wellformed, now_iso, parse_filename, sha256_file,  # noqa: E402
                       setup_logging, write_json)

try:  # optional
    import requests  # type: ignore
except Exception:  # pragma: no cover - depends on the environment
    requests = None

LOG = logging.getLogger("e6.fetch")

VOTEVIEW = "https://voteview.com"
VOTEVIEW_PATHS = ("/static/data/out/members/HS117_members.csv", "/static/data/out/members/HSall_members.csv")
GOVINFO = "https://www.govinfo.gov"
LISTING_PATHS = ("/bulkdata/xml/BILLSTATUS/{c}/{t}", "/bulkdata/json/BILLSTATUS/{c}/{t}")
FILE_PATH = "/bulkdata/BILLSTATUS/{c}/{t}/{name}"
USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/124.0.0.0 Safari/537.36")
VOTEVIEW_COLUMNS = ("congress", "chamber", "icpsr", "state_abbrev", "party_code", "bioname", "bioguide_id",
                    "nominate_dim1", "nominate_dim2", "nominate_number_of_votes")


_SOURCES: list = []  # sources of the current run, closed by main()


MAX_CONSECUTIVE_FAILURES = 50  # per bill type: stop and report instead of retrying thousands of files
MIN_NUMBER_COVERAGE = 0.9      # distinct bill numbers / largest number, below which a type is incomplete


class PermanentError(Exception):
    """Not worth retrying (HTTP 404 / 410, file missing in the offline directory)."""


class HTTPStatusError(Exception):
    def __init__(self, status: int, url: str, retry_after: float | None = None):
        super().__init__(f"HTTP {status} for {url}")
        self.status = status
        self.retry_after = retry_after


def _retry_after(value) -> float | None:
    try:
        return max(0.0, float(value)) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None  # an HTTP date: fall back to the exponential backoff


# ------------------------------------------------------------- listings --

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _entry(d: dict) -> dict | None:
    link = (d.get("link") or "").strip() or None
    name = (d.get("justFileName") or d.get("name") or "").strip()
    if not name and link:
        name = link.rstrip("/").rsplit("/", 1)[-1]
    if not name:
        return None
    folder = str(d.get("folder", "false")).strip().lower() == "true"
    ext = (str(d.get("fileExtension") or "").strip().lower() or (name.rsplit(".", 1)[-1].lower() if "." in name
                                                                  else ""))
    return {"name": os.path.basename(name), "link": link, "ext": ext.lstrip("."), "folder": folder,
            "size": d.get("size"), "modified": d.get("formattedLastModifiedTime") or d.get("lastModified")}


_NAME_RE = re.compile(rb"BILLSTATUS-\d+(?:hr|hres|hjres|hconres|s|sres|sjres|sconres)\d+\.xml|BILLSTATUS-\d+-[a-z]+\.zip")


def _regex_entries(raw: bytes) -> list[dict]:
    """Fallback when the listing's structure is not the expected one: every BILLSTATUS file name that
    occurs in the response (links are then built from the standard file path)."""
    names = sorted({m.group(0).decode() for m in _NAME_RE.finditer(raw)})
    return [{"name": n, "link": None, "ext": n.rsplit(".", 1)[-1], "folder": False, "size": None, "modified": None}
            for n in names]


def parse_listing(raw: bytes) -> list[dict]:
    """Entries of a GovInfo bulk-data directory listing: XML (<files><file><justFileName>, <link>,
    <fileExtension>, <folder> ...) or JSON ({"files": [{...}]}). Folders are dropped. If neither
    structure yields an entry, the BILLSTATUS file names found anywhere in the response are used."""
    s = raw.lstrip()
    entries = []
    try:
        if s[:1] in (b"{", b"["):
            obj = json.loads(raw.decode("utf-8-sig"))
            files = obj.get("files", []) if isinstance(obj, dict) else obj
            entries = [_entry({k: v for k, v in f.items()}) for f in files if isinstance(f, dict)]
        else:
            root = ET.fromstring(raw)
            for el in root.iter():
                if _local(el.tag) == "file" and len(el):
                    entries.append(_entry({_local(c.tag): (c.text or "") for c in el}))
    except (ValueError, ET.ParseError):
        entries = []
    out = [e for e in entries if e and not e["folder"]]
    return out if out else _regex_entries(raw)


# --------------------------------------------------------------- sources --

def _retry(fn, attempts: int, backoff: float, what: str):
    last = None
    for k in range(max(1, attempts)):
        try:
            return fn()
        except PermanentError:
            raise
        except Exception as e:  # network errors, timeouts, HTTP 5xx/429/403, malformed content
            last = e
            if k + 1 < attempts:
                wait = min(60.0, backoff * 2 ** k) * (0.5 + random.random())
                ra = getattr(e, "retry_after", None)
                if ra is not None:  # honour the server's Retry-After (429 / 503), at most 5 minutes
                    wait = max(wait, min(300.0, ra))
                LOG.debug("%s: attempt %d failed (%s); retrying in %.1fs", what, k + 1, e, wait)
                time.sleep(wait)
    raise last


class HttpSource:
    mode = "online"

    def __init__(self, retries: int = 5, timeout: float = 60.0, backoff: float = 2.0, user_agent: str = USER_AGENT,
                 govinfo: str = GOVINFO, voteview: str = VOTEVIEW):
        self.attempts, self.timeout, self.backoff, self.ua = retries + 1, timeout, backoff, user_agent
        self.govinfo, self.voteview_base = govinfo.rstrip("/"), voteview.rstrip("/")
        self._tls = threading.local()
        self._sessions: list = []
        self._lock = threading.Lock()
        self.client = "requests" if requests is not None else "urllib"

    def _session(self):
        s = getattr(self._tls, "session", None)
        if s is None:  # one session per thread (connection reuse)
            s = requests.Session()
            s.headers.update({"User-Agent": self.ua, "Accept": "*/*"})
            self._tls.session = s
            with self._lock:
                self._sessions.append(s)
        return s

    def close(self) -> None:
        with self._lock:
            for s in self._sessions:
                s.close()
            self._sessions.clear()

    def _once(self, url: str, dest: Path | None, accept: str = "*/*"):
        if requests is not None:
            r = self._session().get(url, timeout=(min(30.0, self.timeout), self.timeout), stream=dest is not None,
                                    headers={"Accept": accept})
            try:
                if r.status_code in (404, 410):
                    raise PermanentError(f"HTTP {r.status_code} for {url}")
                if r.status_code != 200:
                    raise HTTPStatusError(r.status_code, url, _retry_after(r.headers.get("Retry-After")))
                if dest is None:
                    return r.content
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        if chunk:
                            f.write(chunk)
                return None
            finally:
                r.close()
        req = urllib.request.Request(url, headers={"User-Agent": self.ua, "Accept": accept})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                if dest is None:
                    return r.read()
                with open(dest, "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 16)
                return None
        except urllib.error.HTTPError as e:
            code = e.code
            ra = _retry_after(e.headers.get("Retry-After") if e.headers else None)
            e.close()  # the error holds the response
            if code in (404, 410):
                raise PermanentError(f"HTTP {code} for {url}") from None
            raise HTTPStatusError(code, url, ra) from None

    def get_bytes(self, url: str) -> bytes:
        return _retry(lambda: self._once(url, None), self.attempts, self.backoff, url)

    def get_file(self, url: str, dest: Path, validate) -> None:
        """Download to dest (via dest.part); validate(path) must return True, else retry."""
        tmp = dest.with_name(dest.name + ".part")

        def go():
            self._once(url, tmp)
            if not validate(tmp):
                raise ValueError(f"malformed content from {url}")
            os.replace(tmp, dest)

        try:
            _retry(go, self.attempts, self.backoff, url)
        finally:
            if tmp.exists():
                tmp.unlink()

    # -- interface used by the fetch logic
    def listing(self, t: str) -> tuple[bytes, str, str]:
        """The XML listing (Accept: application/xml), then the JSON one (application/json); each also with
        a generic Accept. A response without any file entry (maintenance or error page) is retried like a
        network error. GovInfo asks for explicit Accept headers on the bulk-data listings (else HTTP 406)."""
        errors = []
        for tpl in LISTING_PATHS:
            url = self.govinfo + tpl.format(c=CONGRESS, t=t)
            fmt = "json" if "/json/" in url else "xml"
            for accept in (("application/json" if fmt == "json" else "application/xml"), "*/*"):
                def go(url=url, accept=accept):
                    raw = self._once(url, None, accept)
                    if not parse_listing(raw):
                        raise ValueError("listing without any file entry")
                    return raw
                try:
                    return _retry(go, self.attempts, self.backoff, url), url, fmt
                except Exception as e:
                    errors.append(f"{url} (Accept {accept}): {e}")
        raise RuntimeError("; ".join(errors))

    def url_of(self, t: str, e: dict) -> str:
        return e["link"] or self.govinfo + FILE_PATH.format(c=CONGRESS, t=t, name=e["name"])

    def fetch(self, t: str, e: dict, dest: Path, validate) -> str:
        url = self.url_of(t, e)
        self.get_file(url, dest, validate)
        return url

    def voteview(self, dest_dir: Path) -> list[tuple[str, Path]]:
        """Candidates (url, local file) in order of preference, downloaded lazily by fetch_voteview."""
        return [(self.voteview_base + u, dest_dir / u.rsplit("/", 1)[-1]) for u in VOTEVIEW_PATHS]

    def get_voteview(self, url: str, dest: Path, validate) -> None:
        self.get_file(url, dest, validate)


class LocalSource:
    mode = "offline"
    client = "local files"

    def __init__(self, root, no_listing_ok: bool = False):
        self.root = Path(root)
        self.no_listing_ok = no_listing_ok

    def type_dir(self, t: str) -> Path:
        for p in (self.root / "billstatus" / t, self.root / "BILLSTATUS" / str(CONGRESS) / t, self.root / t):
            if p.is_dir():
                return p
        raise PermanentError(f"no folder for type {t} under {self.root} (billstatus/{t}/)")

    def listing(self, t: str) -> tuple[bytes, str, str]:
        d = self.type_dir(t)
        for name, fmt in (("listing.xml", "xml"), ("listing.json", "json")):
            p = d / name
            if p.exists():
                return p.read_bytes(), str(p), fmt
        if not self.no_listing_ok:
            raise PermanentError(f"no listing.xml / listing.json in {d} (use --no-listing-ok to take the folder's "
                                 "contents as the listing; completeness is then unverified)")
        files = sorted(p.name for p in d.iterdir() if p.is_file() and p.suffix.lower() in (".xml", ".zip")
                       and not p.name.startswith("listing."))
        doc = {"files": [{"justFileName": n, "fileExtension": n.rsplit(".", 1)[-1], "folder": False} for n in files]}
        return json.dumps(doc).encode(), f"{d} (folder contents)", "directory"

    def url_of(self, t: str, e: dict) -> str:
        return str(self.type_dir(t) / e["name"])

    def close(self) -> None:
        pass

    def fetch(self, t: str, e: dict, dest: Path, validate) -> str:
        src = self.type_dir(t) / e["name"]
        if not src.exists():
            raise PermanentError(f"{src} missing")
        tmp = dest.with_name(dest.name + ".part")
        shutil.copyfile(src, tmp)
        if not validate(tmp):
            tmp.unlink()
            raise PermanentError(f"{src} is malformed")
        os.replace(tmp, dest)
        return str(src)

    def voteview(self, dest_dir: Path) -> list[tuple[str, Path]]:
        out = []
        for name in ("HS117_members.csv", "HSall_members.csv"):
            for d in (self.root, self.root / "voteview"):
                if (d / name).exists():
                    out.append((str(d / name), dest_dir / name))
                    break
        return out

    def get_voteview(self, url: str, dest: Path, validate) -> None:
        tmp = dest.with_name(dest.name + ".part")
        shutil.copyfile(url, tmp)
        if not validate(tmp):
            tmp.unlink()
            raise PermanentError(f"{url} is malformed")
        os.replace(tmp, dest)


# --------------------------------------------------------------- Voteview --

def voteview_check(path: Path) -> dict:
    """Header with the needed columns and at least one House/Senate row of congress 117."""
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            cols = rd.fieldnames or []
            missing = [c for c in VOTEVIEW_COLUMNS if c not in cols]
            if missing:
                return {"ok": False, "error": f"missing columns {missing}"}
            n = n117 = 0
            for row in rd:
                n += 1
                if row.get("congress") == str(CONGRESS) and row.get("chamber") in ("House", "Senate"):
                    n117 += 1
    except (OSError, UnicodeDecodeError, csv.Error) as e:
        return {"ok": False, "error": str(e)}
    return {"ok": n117 > 0, "rows": n, "rows_117_house_senate": n117,
            **({} if n117 else {"error": "no House/Senate rows of congress 117"})}


def _filter_117(src: Path, dest: Path) -> int:
    tmp = dest.with_name(dest.name + ".part")
    n = 0
    with open(src, newline="", encoding="utf-8") as fi, open(tmp, "w", newline="", encoding="utf-8") as fo:
        rd = csv.reader(fi)
        header = next(rd)
        ci = header.index("congress")
        w = csv.writer(fo, lineterminator="\n")
        w.writerow(header)
        for row in rd:
            if len(row) > ci and row[ci] == str(CONGRESS):
                w.writerow(row)
                n += 1
    os.replace(tmp, dest)
    return n


def fetch_voteview(src, out_dir: Path, old: dict | None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / "HS117_members.csv"
    rec = {"file": "voteview/HS117_members.csv", "attempts": []}
    if final.exists() and voteview_check(final)["ok"]:
        prev = (old or {}).get("voteview") or {}
        rec.update({k: v for k, v in prev.items() if k not in ("attempts", "resumed")})
        rec.update({"resumed": True, "sha256": sha256_file(final), "size": final.stat().st_size,
                    "check": voteview_check(final), "ok": True})
        LOG.info("Voteview: %s already present and valid (resume)", final)
        return rec
    for url, dest in src.voteview(out_dir):
        t0 = now_iso()
        try:
            src.get_voteview(url, dest, lambda p: voteview_check(p)["ok"])
        except Exception as e:
            rec["attempts"].append({"url": url, "time": t0, "error": str(e)})
            LOG.warning("Voteview: %s failed: %s", url, e)
            continue
        rec["attempts"].append({"url": url, "time": t0, "ok": True})
        rec.update({"url": url, "downloaded_at": t0, "source_file": dest.name, "source_sha256": sha256_file(dest),
                    "source_size": dest.stat().st_size})
        if dest.name != final.name:  # HSall: keep it and filter to the 117th Congress
            rec["filtered_rows"] = _filter_117(dest, final)
            rec["derived_from"] = dest.name
        rec.update({"sha256": sha256_file(final), "size": final.stat().st_size, "check": voteview_check(final),
                    "ok": True})
        LOG.info("Voteview: %s -> %s (%d bytes, sha256 %s)", url, final, rec["size"], rec["sha256"][:16])
        return rec
    rec["ok"] = False
    rec["error"] = "no Voteview members file could be obtained"
    return rec


# ------------------------------------------------------------ BILLSTATUS --

def _zip_ok(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None
    except (zipfile.BadZipFile, OSError, EOFError):
        return False


def _file_rec(path: Path, name: str, url: str, source: str) -> dict:
    return {"name": name, "url": url, "size": path.stat().st_size, "sha256": sha256_file(path), "source": source,
            "time": now_iso()}


def _extract_zip(zpath: Path, tdir: Path, zip_name: str) -> tuple[dict, list[str]]:
    """XML members of the package -> tdir (by base name; a well-formed file already there is kept)."""
    got, problems = {}, []
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            base = os.path.basename(info.filename)
            if not base.lower().endswith(".xml"):
                continue
            if base in got:
                problems.append(f"{zip_name}: duplicate member name {base} (first kept)")
                continue
            dest = tdir / base
            if dest.exists() and billstatus_wellformed(dest):
                got[base] = _file_rec(dest, base, f"{zip_name}:{info.filename}", "zip (already extracted)")
                continue
            tmp = dest.with_name(base + ".part")
            with z.open(info) as fi, open(tmp, "wb") as fo:
                shutil.copyfileobj(fi, fo, 1 << 16)
            if not billstatus_wellformed(tmp):
                tmp.unlink()
                problems.append(f"{zip_name}: member {info.filename} is malformed")
                continue
            os.replace(tmp, dest)
            got[base] = _file_rec(dest, base, f"{zip_name}:{info.filename}", "zip")
    return got, problems


def fetch_type(src, t: str, tdir: Path, threads: int, use_zip: bool) -> dict:
    tdir.mkdir(parents=True, exist_ok=True)
    rec = {"type": t, "listing": None, "zip": [], "listed": 0, "listed_xml": 0, "listed_zip": 0, "listed_other": [],
           "obtained_xml": 0, "downloaded": 0, "from_zip": 0, "resumed": 0, "failures": [], "warnings": [],
           "files": [], "complete": False}
    t0 = now_iso()
    try:
        raw, origin, fmt = src.listing(t)
        entries = parse_listing(raw)
    except Exception as e:
        rec["failures"].append({"name": "<listing>", "error": str(e)})
        LOG.error("%s: directory listing not obtained: %s", t, e)
        return rec
    n_raw, seen, uniq = len(entries), set(), []
    for e in entries:  # one entry per file name (the first); duplicates would race in the thread pool
        if e["name"] not in seen:
            seen.add(e["name"])
            uniq.append(e)
    entries = uniq
    if len(entries) != n_raw:
        rec["warnings"].append(f"listing names {n_raw - len(entries)} files twice (deduplicated)")
    lpath = tdir / ("listing.json" if fmt in ("json", "directory") else "listing.xml")
    lpath.write_bytes(raw)
    rec["listing"] = {"source": origin, "format": fmt, "file": lpath.name, "time": t0, "sha256": sha256_file(lpath),
                      "entries": len(entries), "verified": fmt != "directory"}
    def _is_bill_xml(name: str) -> bool:
        pf = parse_filename(name)
        return pf is not None and pf[1] == t
    xml_e = [e for e in entries if e["ext"] == "xml" and _is_bill_xml(e["name"])]
    zip_e = [e for e in entries if e["ext"] == "zip"]
    other = [e["name"] for e in entries if not (e["ext"] == "zip" or (e["ext"] == "xml" and _is_bill_xml(e["name"])))]
    rec["listing"]["parser"] = "structured" if any(e.get("link") for e in entries) else "file names in the page"
    rec.update({"listed": len(entries), "listed_xml": len(xml_e), "listed_zip": len(zip_e), "listed_other": other})
    if other:
        rec["warnings"].append(f"{len(other)} listed entries are neither XML nor ZIP and were not downloaded")
    if not entries:
        rec["failures"].append({"name": "<listing>", "error": "listing has no file entries"})
        return rec
    LOG.info("%s: listing %s: %d XML, %d ZIP entries", t, origin, len(xml_e), len(zip_e))

    obtained: dict[str, dict] = {}
    zip_ok_any = False
    if use_zip:
        for ze in zip_e:
            zpath = tdir / ze["name"]
            zrec = {"name": ze["name"], "url": src.url_of(t, ze)}
            try:
                if zpath.exists() and _zip_ok(zpath):
                    zrec["source"] = "resume"
                else:
                    zrec["url"] = src.fetch(t, ze, zpath, _zip_ok)
                    zrec["source"] = "download" if src.mode == "online" else "offline copy"
                zrec.update({"size": zpath.stat().st_size, "sha256": sha256_file(zpath), "time": now_iso()})
                got, problems = _extract_zip(zpath, tdir, ze["name"])
                zrec.update({"xml_members": len(got), "problems": problems})
                rec["warnings"] += problems
                for name, r in got.items():
                    obtained.setdefault(name, r)
                zip_ok_any = True
                LOG.info("%s: package %s: %d XML files", t, ze["name"], len(got))
            except Exception as e:
                zrec["error"] = str(e)
                rec["warnings"].append(f"package {ze['name']} not obtained ({e}); falling back to single files")
                LOG.warning("%s: package %s not obtained: %s", t, ze["name"], e)
            rec["zip"].append(zrec)

    todo = [e for e in xml_e if e["name"] not in obtained]
    lock = threading.Lock()
    done = [0]
    state = {"consecutive": 0, "abort": False}

    def one(e: dict):
        dest = tdir / e["name"]
        if state["abort"]:  # circuit breaker: the source is failing systematically, report instead of grinding on
            return ("fail", {"name": e["name"], "url": src.url_of(t, e), "error": "not attempted (circuit breaker)"})
        try:
            if dest.exists() and billstatus_wellformed(dest):
                r = _file_rec(dest, e["name"], src.url_of(t, e), "resume")
            else:
                url = src.fetch(t, e, dest, billstatus_wellformed)
                r = _file_rec(dest, e["name"], url, "download" if src.mode == "online" else "offline copy")
            out = ("ok", r)
        except Exception as ex:
            out = ("fail", {"name": e["name"], "url": src.url_of(t, e), "error": str(ex)})
        with lock:
            if out[0] == "ok":
                state["consecutive"] = 0
            else:
                state["consecutive"] += 1
                if state["consecutive"] >= MAX_CONSECUTIVE_FAILURES and not state["abort"]:
                    state["abort"] = True
                    LOG.error("%s: %d consecutive download failures: stopping this type (last error: %s)", t,
                              state["consecutive"], out[1]["error"])
            done[0] += 1
            if done[0] % 500 == 0:
                LOG.info("%s: %d / %d single files", t, done[0], len(todo))
        return out

    if todo:
        LOG.info("%s: %d XML files to obtain one by one (%d threads)", t, len(todo), threads)
        with cf.ThreadPoolExecutor(max_workers=max(1, threads)) as ex:
            for status, r in ex.map(one, todo):
                if status == "ok":
                    obtained[r["name"]] = r
                else:
                    rec["failures"].append(r)
    listed_names = {e["name"] for e in xml_e}
    # files from the package that the listing does not name are kept (and counted) too
    rec["files"] = [obtained[k] for k in sorted(obtained)]
    rec["obtained_xml"] = len(obtained)
    rec["unlisted_from_zip"] = len(set(obtained) - listed_names) if xml_e else 0
    rec["from_zip"] = sum(r["source"].startswith("zip") for r in obtained.values())
    rec["resumed"] = sum(r["source"] == "resume" for r in obtained.values())
    rec["downloaded"] = sum(r["source"] in ("download", "offline copy") for r in obtained.values())
    missing = sorted(listed_names - set(obtained))
    if state["abort"]:
        rec["failures"].append({"name": "<type>", "error": f"stopped after {MAX_CONSECUTIVE_FAILURES} consecutive "
                                                            "download failures"})
    # completeness that does not rest on the listing alone: bill numbers run 1..max with few gaps (reserved
    # numbers), so a truncated or paginated listing shows up as low coverage
    nums = sorted({pf[2] for pf in (parse_filename(k) for k in obtained) if pf is not None and pf[1] == t})
    cov = (len(nums) / max(nums)) if nums else 0.0
    rec["number_coverage"] = {"distinct": len(nums), "max_number": max(nums) if nums else 0, "coverage": cov}
    if xml_e:
        rec["complete"] = not missing and not state["abort"]
    else:  # the listing names only the package
        rec["complete"] = zip_ok_any
        if not zip_ok_any:
            rec["failures"].append({"name": "<package>", "error": "listing names only a package and it was not obtained"})
    if nums and cov < MIN_NUMBER_COVERAGE:
        rec["complete"] = False
        rec["failures"].append({"name": "<coverage>", "error": f"bill numbers cover {cov:.1%} of 1..{max(nums)} "
                                                                f"(< {MIN_NUMBER_COVERAGE:.0%}): listing truncated?"})
    elif nums and cov < 0.99:
        rec["warnings"].append(f"bill numbers cover {cov:.1%} of 1..{max(nums)}")
    rec["missing"] = missing
    LOG.info("%s: %d XML files (%d from package, %d downloaded, %d resumed), %d missing", t, len(obtained),
             rec["from_zip"], rec["downloaded"], rec["resumed"], len(missing))
    return rec


# ------------------------------------------------------------------- main --

def run(args) -> int:
    data = Path(args.data)
    data.mkdir(parents=True, exist_ok=True)
    setup_logging(data / "fetch.log")
    man_path = data / "fetch_manifest.json"
    old = None
    if man_path.exists():
        try:
            old = json.loads(man_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            old = None
    if args.offline_dir:
        src = LocalSource(args.offline_dir, no_listing_ok=args.no_listing_ok)
    else:
        src = HttpSource(retries=args.retries, timeout=args.timeout, backoff=args.backoff, govinfo=args.govinfo_url,
                         voteview=args.voteview_url)
    _SOURCES.append(src)
    man = {"tool": "e6.fetch", "started": now_iso(), "mode": src.mode, "client": src.client,
           "govinfo": getattr(src, "govinfo", None), "voteview_base": getattr(src, "voteview_base", None),
           "offline_dir": str(args.offline_dir) if args.offline_dir else None, "congress": CONGRESS,
           "types": list(args.types), "threads": args.threads, "use_zip": not args.no_zip,
           "user_agent": USER_AGENT if src.mode == "online" else None}
    LOG.info("e6 fetch (%s, %s) -> %s", src.mode, src.client, data)
    man["voteview"] = fetch_voteview(src, data / "voteview", old)
    man["billstatus"] = {}
    for t in args.types:
        man["billstatus"][t] = fetch_type(src, t, data / "billstatus" / t, args.threads, not args.no_zip)
        write_json(man, man_path)  # partial manifest after every type
    vv_ok = bool(man["voteview"].get("ok"))
    bs = man["billstatus"]
    unverified = [t for t, r in bs.items() if r.get("listing") and not r["listing"]["verified"]]
    man.update({"finished": now_iso(), "complete": vv_ok and all(r["complete"] for r in bs.values()),
                "listing_unverified_types": unverified,
                "totals": {"listed_xml": sum(r["listed_xml"] for r in bs.values()),
                           "obtained_xml": sum(r["obtained_xml"] for r in bs.values()),
                           "downloaded": sum(r["downloaded"] for r in bs.values()),
                           "from_zip": sum(r["from_zip"] for r in bs.values()),
                           "resumed": sum(r["resumed"] for r in bs.values()),
                           "failures": sum(len(r["failures"]) for r in bs.values()) + (0 if vv_ok else 1)}})
    write_json(man, man_path)
    for t, r in bs.items():
        LOG.info("%s: listed %d (xml %d, zip %d), obtained %d, failures %d", t, r["listed"], r["listed_xml"],
                 r["listed_zip"], r["obtained_xml"], len(r["failures"]))
    if unverified:
        LOG.warning("no saved listing for %s: completeness of these types is unverified", unverified)
    if not man["complete"]:
        parts = []
        if not vv_ok:
            parts.append("Voteview members file not obtained")
        for t, r in bs.items():
            if not r["complete"]:
                parts.append(f"{t}: {len(r.get('missing', []))} of {r['listed_xml']} listed XML files missing, "
                             f"{len(r['failures'])} failures")
        LOG.error("E6 FETCH INCOMPLETE (%s). Plan section 10: stop and report. Details in %s",
                  "; ".join(parts), man_path)
        return 2
    LOG.info("e6 fetch complete: %d BILLSTATUS files, Voteview sha256 %s", man["totals"]["obtained_xml"],
             man["voteview"]["sha256"])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m e6.fetch", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", "--out", dest="data", required=True, help="data directory (e.g. data/e6)")
    ap.add_argument("--offline-dir", default=None, help="use files placed by hand instead of the network")
    ap.add_argument("--no-listing-ok", action="store_true",
                    help="offline only: take a folder's contents as its listing when no listing file is present")
    ap.add_argument("--types", nargs="+", default=list(BILL_TYPES), choices=list(BILL_TYPES))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--retries", type=int, default=5)
    ap.add_argument("--timeout", type=float, default=60.0, help="seconds per request (read timeout)")
    ap.add_argument("--backoff", type=float, default=2.0, help="first retry wait in seconds (doubles)")
    ap.add_argument("--no-zip", action="store_true", help="ignore bulk packages, download single files")
    ap.add_argument("--govinfo-url", default=GOVINFO, help="base URL of GovInfo (a mirror; tests)")
    ap.add_argument("--voteview-url", default=VOTEVIEW, help="base URL of Voteview (a mirror; tests)")
    args = ap.parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130
    finally:
        for src in _SOURCES:
            src.close()
        _SOURCES.clear()


if __name__ == "__main__":
    sys.exit(main())
