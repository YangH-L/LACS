#!/usr/bin/env python3
"""E6 stage e6build: members, tolerances, cosponsorship events, the Twitter graph and the geo-format
dataset (plan sections 2-3).

  python -m e6.build --data data/e6 --out runs/e6

Inputs: --data as written by e6.fetch (voteview/HS117_members.csv, billstatus/<type>/*.xml), the
network JSON and crosswalk shipped in e6/data/ (--network, --crosswalk to override).

Definitions (plan section 3):
  * position: Voteview nominate_dim1 (1-D); (dim1, dim2) Euclidean for the 2-D sensitivity. One row per
    bioguide: President rows and rows without dim1 dropped, then the House/Senate row with the most
    nominate_number_of_votes kept. party_code 100 = D, 200 = R, 328 = I; independents caucus with the
    Democrats.
  * join: sponsor on introducedDate, cosponsor on sponsorshipDate; withdrawn cosponsorships (a non-
    empty sponsorshipWithdrawnDate) do not count anywhere; periods train / val / test as in
    e6.common.PERIODS.
  * tolerance theta_q50 (q70, q90): quantile (linear interpolation) over the member's training-period
    cosponsorships of the distance to the bill's sponsor, sponsors without a NOMINATE score skipped;
    floor 0.02. Members (nodes): matched to the network and to Voteview, >= 10 such cosponsorships.
  * graph: undirected union of the directed Twitter edges among members.
  * data gate: >= 400 members, else exit 3 (after writing the public tables and e6_build.json).
Exit 2 (stop and report, plan section 10): fetch_manifest.json says the download is incomplete, or
some BILLSTATUS file cannot be parsed / belongs to another congress or type (--allow-incomplete
continues anyway and records it).

Outputs (--out = the geo run directory, e.g. runs/e6):
  public/e6_members.csv        one row per network account: ids, Voteview fields, counts, tolerances
                               (1-D and 2-D, floored and raw), eligible flag and reason, member index
  public/e6_events.csv.gz      every sponsorship / cosponsorship of hr, s, hjres, sjres bills
  public/e6_twitter_edges.csv  directed edges among members: src, dst (member index), weight;
                               src -> dst means dst retweeted / quoted / replied to / mentioned src
  public/e6_build.json         counts, gate, crosswalk and fetch summary, sha256 of inputs
  geo/users.csv, positions.npy (dim1, 0, 0), positions_2d.npy (dim1, dim2, 0), covisits.npz,
  test_checkins.npz, test_checkins_notop.npz, twitter_weights.npz, meta.json; graph/graph.bin
                               the files lacs.apps.geo reads (same formats as geo.prepare); members are
                               indexed in sorted bioguide order; co-visit key = integer bill id
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # also works under python -I
    sys.path.insert(0, str(ROOT))

from e6.common import (ENGINE_TIE_ATOL, ENGINE_TIE_RTOL, BILL_ID_MULT, BILL_TYPES, CONGRESS, DATA_DIR, MIN_ELIGIBLE, MIN_TRAIN_COSPONSORSHIPS,  # noqa: E402
                       PERIOD_NAMES, PERIODS, THETA_FLOOR, THETA_QUANTILES, TYPE_CODE, bill_id, now_iso,
                       parse_billstatus, parse_filename, period_of, read_json, setup_logging, sha256_file,
                       write_json)

LOG = logging.getLogger("e6.build")
QNAMES = [f"q{int(round(q * 100))}" for q in THETA_QUANTILES]
PARTY = {100: "D", 200: "R", 328: "I"}
CAUCUS = {100: "D", 200: "R", 328: "D"}  # plan section 3: independents count with the Democrats
EVENT_COLS = ["bill_id", "bill_type", "bill_number", "introduced_date", "member", "role", "join_date", "original",
              "withdrawn", "withdrawn_date", "period"]


# ------------------------------------------------------------------ Voteview --

def load_voteview(path) -> tuple[pd.DataFrame, dict]:
    """One row per bioguide (index), see the module docstring. Returns (table, stats)."""
    df = pd.read_csv(path, dtype={"bioguide_id": str, "chamber": str, "state_abbrev": str, "bioname": str})
    if "congress" in df.columns:
        df = df[df["congress"] == CONGRESS]
    st = {"rows": int(len(df)), "rows_president": int((df["chamber"] == "President").sum())}
    df = df[df["bioguide_id"].notna() & (df["bioguide_id"].str.strip() != "")].copy()
    df["bioguide_id"] = df["bioguide_id"].str.strip()
    st["bioguides_any_row"] = int(df["bioguide_id"].nunique())
    hs = df[df["chamber"].isin(["House", "Senate"])]
    st["rows_house_senate"] = int(len(hs))
    has = hs[hs["nominate_dim1"].notna()].copy()
    st["rows_house_senate_without_dim1"] = int(len(hs) - len(has))
    has["_votes"] = has["nominate_number_of_votes"].fillna(-1)
    has = has.sort_values(["bioguide_id", "_votes", "chamber", "icpsr"], ascending=[True, False, True, True],
                          kind="mergesort")
    dup = has["bioguide_id"].duplicated(keep=False)
    st["bioguides_with_several_rows"] = int(has.loc[dup, "bioguide_id"].nunique())
    vv = has.drop_duplicates("bioguide_id", keep="first").set_index("bioguide_id")
    vv["dim1"] = vv["nominate_dim1"].astype(np.float64)
    vv["dim2"] = vv["nominate_dim2"].astype(np.float64)
    codes = vv["party_code"].astype("Int64")
    vv["party"] = [PARTY.get(int(c), f"O{int(c)}") if pd.notna(c) else "O" for c in codes]
    vv["caucus"] = [CAUCUS.get(int(c), "O") if pd.notna(c) else "O" for c in codes]
    st["bioguides_with_dim1"] = int(len(vv))
    st["party_codes"] = {str(k): int(v) for k, v in codes.value_counts().sort_index().items()}
    other = vv.index[vv["caucus"] == "O"].tolist()
    if other:
        LOG.warning("Voteview: %d members with a party code other than 100/200/328 (caucus 'O'): %s", len(other),
                    other[:10])
    st["bioguides_without_any_dim1_row"] = sorted(set(df["bioguide_id"]) - set(vv.index))
    return vv, st


# ---------------------------------------------------------------- BILLSTATUS --

def _parse_one(path: str):
    try:
        return path, parse_billstatus(path), None
    except Exception as e:  # malformed file: counted and reported
        return path, None, f"{type(e).__name__}: {e}"


def bill_files(data: Path, types) -> list[tuple[str, Path]]:
    out = []
    for t in types:
        d = data / "billstatus" / t
        if not d.is_dir():
            raise SystemExit(f"[e6 build] {d} missing; run e6.fetch first")
        for p in sorted(d.iterdir()):
            pf = parse_filename(p.name)
            if pf and pf[0] == CONGRESS and pf[1] == t:
                out.append((t, p))
    return out


def parse_bills(files, procs: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(bills, events, stats) from the BILLSTATUS files."""
    st = {"files": len(files), "files_by_type": {}, "parse_errors": [], "type_or_congress_mismatch": [],
          "number_mismatch": 0, "duplicate_bills": 0, "items_without_bioguide": 0}
    for t, _ in files:
        st["files_by_type"][t] = st["files_by_type"].get(t, 0) + 1
    paths = [str(p) for _, p in files]
    ftype = {str(p): t for t, p in files}
    if procs > 1 and len(paths) > 200:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=procs) as ex:
            results = list(ex.map(_parse_one, paths, chunksize=64))
    else:
        results = [_parse_one(p) for p in paths]
    bills, rows, seen = [], [], set()
    for path, d, err in results:
        name = os.path.basename(path)
        if err:
            st["parse_errors"].append({"file": name, "error": err})
            continue
        t_file = ftype[path]
        num_file = parse_filename(name)[2]
        if d["congress"] != CONGRESS or d["type"] != t_file:
            st["type_or_congress_mismatch"].append({"file": name, "congress": d["congress"], "type": d["type"]})
            continue
        if d["number"] is None:
            st["parse_errors"].append({"file": name, "error": "no bill number"})
            continue
        if d["number"] != num_file:
            st["number_mismatch"] += 1
        bid = bill_id(d["type"], d["number"])
        if bid in seen:
            st["duplicate_bills"] += 1
            continue
        seen.add(bid)
        st["items_without_bioguide"] += d["items_without_bioguide"]
        intro = d["introduced"]
        bills.append({"bill_id": bid, "bill_type": d["type"], "bill_number": d["number"], "introduced_date": intro,
                      "sponsor": d["sponsors"][0] if d["sponsors"] else None, "n_sponsors": len(d["sponsors"]),
                      "n_cosponsors": len(d["cosponsors"])})
        for s in d["sponsors"]:
            rows.append((bid, d["type"], d["number"], intro, s, "sponsor", intro, None, False, None))
        for c in d["cosponsors"]:
            rows.append((bid, d["type"], d["number"], intro, c["bioguide"], "cosponsor", c["date"], c["original"],
                         c["withdrawn"], c["withdrawn_date"]))
    bills = pd.DataFrame(bills, columns=["bill_id", "bill_type", "bill_number", "introduced_date", "sponsor",
                                         "n_sponsors", "n_cosponsors"]).sort_values("bill_id", kind="mergesort")
    ev = pd.DataFrame(rows, columns=EVENT_COLS[:-1])
    n_raw = len(ev)
    ev = ev.drop_duplicates().reset_index(drop=True)
    st["exact_duplicate_event_rows"] = int(n_raw - len(ev))
    per = {d: period_of(d) for d in pd.unique(ev["join_date"].dropna())}
    ev["period"] = ev["join_date"].map(per).fillna(-1).astype(np.int64)
    ev = ev.sort_values(["bill_id", "role", "join_date", "member"], ascending=[True, False, True, True],
                        kind="mergesort", na_position="last").reset_index(drop=True)
    st["bills"] = int(len(bills))
    st["bills_by_type"] = {t: int((bills["bill_type"] == t).sum()) for t in BILL_TYPES}
    st["bills_without_sponsor"] = int((bills["n_sponsors"] == 0).sum())
    st["bills_with_several_sponsors"] = int((bills["n_sponsors"] > 1).sum())
    st["events"] = int(len(ev))
    st["events_by_role"] = {r: int((ev["role"] == r).sum()) for r in ("sponsor", "cosponsor")}
    st["cosponsorships_withdrawn"] = int(ev["withdrawn"].sum())
    st["events_without_join_date"] = int(ev["join_date"].isna().sum())
    st["events_by_period"] = {n: int((ev["period"] == i).sum()) for i, n in enumerate(PERIOD_NAMES)}
    st["events_outside_periods"] = int((ev["period"] < 0).sum())
    if st["parse_errors"]:
        LOG.warning("%d BILLSTATUS files could not be parsed (first: %s)", len(st["parse_errors"]),
                    st["parse_errors"][0])
    return bills, ev, st


# ----------------------------------------------------------------- tolerances --

def joins_of(ev: pd.DataFrame) -> pd.DataFrame:
    """Joins that count: not withdrawn, with a date; one per (bill, member), the earliest."""
    j = ev[~ev["withdrawn"].astype(bool) & ev["join_date"].notna()]
    j = j.sort_values(["join_date", "role"], ascending=[True, False], kind="mergesort")
    return j.drop_duplicates(["bill_id", "member"], keep="first")


def train_cosponsorships(ev: pd.DataFrame, bills: pd.DataFrame) -> pd.DataFrame:
    """Training-period cosponsorships that count (not withdrawn), one per (bill, member), with the
    bill's sponsor."""
    c = ev[(ev["role"] == "cosponsor") & ~ev["withdrawn"].astype(bool) & ev["join_date"].notna()]
    c = c.sort_values("join_date", kind="mergesort").drop_duplicates(["bill_id", "member"], keep="first")
    c = c[c["period"] == 0].copy()
    c["sponsor"] = c["bill_id"].map(bills.set_index("bill_id")["sponsor"])
    return c


def tolerances(cos: pd.DataFrame, pos: pd.DataFrame, cols: list[str], floor: float = THETA_FLOOR,
               qs=THETA_QUANTILES) -> pd.DataFrame:
    """Per member: n (usable cosponsorships) and raw / floored quantiles of the Euclidean distance in
    `cols` (e.g. ['dim1'] or ['dim1', 'dim2']) between the member and the bill's sponsor. Members and
    sponsors need finite values in all `cols`."""
    P = pos[cols].dropna()
    ok = cos["member"].isin(P.index) & cos["sponsor"].isin(P.index)
    c = cos[ok]
    if len(c) == 0:
        return pd.DataFrame(columns=["n"] + [f"raw_{q}" for q in QNAMES] + QNAMES)
    a = P.loc[c["member"]].to_numpy(np.float64)
    b = P.loc[c["sponsor"]].to_numpy(np.float64)
    d = np.sqrt(((a - b) ** 2).sum(axis=1))
    g = pd.DataFrame({"member": c["member"].to_numpy(), "d": d}).groupby("member")["d"]
    out = pd.DataFrame({"n": g.size()})
    qv = g.quantile(list(qs)).unstack()  # linear interpolation (numpy's default)
    for q, name in zip(qs, QNAMES):
        out[f"raw_{name}"] = qv[q]
        out[name] = np.maximum(qv[q].to_numpy(), floor)
    return out


# --------------------------------------------------------------------- graph --

def read_network(path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict]:
    """Directed edges (src node, dst node, weight) from outList / outWeight; src -> dst means dst
    engaged with src. Checks that inList / inWeight hold the same edges."""
    with open(path, encoding="utf-8") as f:
        net = json.load(f)[0]
    names = net["usernameList"]
    s, d, w = [], [], []
    for i, (lst, wl) in enumerate(zip(net["outList"], net["outWeight"])):
        for j, x in zip(lst, wl):
            s.append(i)
            d.append(int(j))
            w.append(float(x))
    s, d, w = np.array(s, np.int64), np.array(d, np.int64), np.array(w, np.float64)
    inn = {(int(j), i, float(x)) for i, (lst, wl) in enumerate(zip(net["inList"], net["inWeight"]))
           for j, x in zip(lst, wl)}
    out = set(zip(s.tolist(), d.tolist(), w.tolist()))
    st = {"accounts": len(names), "directed_edges": int(len(s)), "in_out_consistent": inn == out,
          "self_loops": int((s == d).sum()), "weight_mean_all": float(w.mean()) if len(w) else None}
    if inn != out:
        LOG.warning("network: inList/inWeight and outList/outWeight differ (%d vs %d edges)", len(inn), len(out))
    return names, s, d, w, st


def notop_filter(user: np.ndarray, key: np.ndarray, all_member: np.ndarray, all_key: np.ndarray,
                 pct: float = 99.0) -> tuple[np.ndarray, dict]:
    """Mask of the test joins kept when the top 1% bills by number of test-window joiners are removed,
    with the rule of E5 (yelp_prepare): counts over everybody (here every member of Congress who joined a
    bill in the test window, not only the network members), threshold = 99th percentile (numpy's
    default linear interpolation) of the positive per-bill counts, bills with count >= threshold removed."""
    if len(all_key) == 0:
        return np.ones(len(key), bool), {"threshold": None, "bills_with_joins": 0, "bills_removed": 0,
                                         "joins_removed": 0}
    uk, cnt = np.unique(pd.DataFrame({"u": all_member, "k": all_key}).drop_duplicates()["k"].to_numpy(),
                        return_counts=True)
    thr = float(np.quantile(cnt, pct / 100.0))
    top = uk[cnt >= thr]
    keep = ~np.isin(key, top)
    return keep, {"threshold": thr, "percentile": pct, "method": "linear (as E5)", "counted_over": "all joiners",
                  "bills_with_joins": int(len(uk)), "bills_removed": int(len(top)),
                  "joins_removed": int((~keep).sum()), "removed_bill_ids": [int(x) for x in top]}


# ---------------------------------------------------------------------- main --

def run(args) -> int:
    from lacs.apps.geo import GEO_DEFAULTS, covisit_pairs
    from lacs.graph.io import edges_to_csr, write_graph

    data, out = Path(args.data), Path(args.out)
    pub, geo, gdir = out / "public", out / "geo", out / "graph"
    for d in (pub, geo, gdir):
        d.mkdir(parents=True, exist_ok=True)
    setup_logging(pub / "build.log")
    LOG.info("e6 build: data %s -> %s", data, out)
    info = {"tool": "e6.build", "started": now_iso(), "data_dir": str(data), "out_dir": str(out),
            "periods": PERIODS, "min_train_cosponsorships": MIN_TRAIN_COSPONSORSHIPS, "theta_floor": THETA_FLOOR,
            "theta_quantiles": list(THETA_QUANTILES), "min_eligible": args.min_eligible,
            "bill_types": list(args.types), "bill_id": f"type code * {BILL_ID_MULT} + number, codes {TYPE_CODE}"}

    # ---- fetch manifest (download complete?)
    fm = data / "fetch_manifest.json"
    if fm.exists():
        man = read_json(fm)
        info["fetch"] = {"complete": man.get("complete"), "mode": man.get("mode"), "finished": man.get("finished"),
                         "totals": man.get("totals"), "listing_unverified_types": man.get("listing_unverified_types"),
                         "voteview": {k: man.get("voteview", {}).get(k) for k in
                                      ("url", "downloaded_at", "sha256", "size", "derived_from", "source_sha256")},
                         "billstatus": {t: {k: r.get(k) for k in ("listed", "listed_xml", "listed_zip", "obtained_xml",
                                                                  "downloaded", "from_zip", "resumed", "complete")}
                                        | {"failures": len(r.get("failures", [])),
                                           "listing_sha256": (r.get("listing") or {}).get("sha256")}
                                        for t, r in (man.get("billstatus") or {}).items()}}
        if not man.get("complete") and not args.allow_incomplete:
            LOG.error("fetch_manifest.json says the download is incomplete; plan section 10: stop and report "
                      "(--allow-incomplete overrides)")
            write_json(info, pub / "e6_build.json")
            return 2
    else:
        LOG.warning("%s not found: files were not placed by e6.fetch; completeness unchecked", fm)
        info["fetch"] = None

    # ---- crosswalk and network
    cw = pd.read_csv(args.crosswalk, dtype={"handle": str, "bioguide": str, "name": str, "chamber": str,
                                            "state": str, "method": str, "note": str})
    names, s_node, d_node, w, net_st = read_network(args.network)
    if len(cw) != len(names) or sorted(cw["node"].tolist()) != list(range(len(names))) or \
            any(cw.set_index("node").loc[i, "handle"] != h for i, h in enumerate(names)):
        raise SystemExit("[e6 build] crosswalk does not match the network's usernameList")
    if cw["bioguide"].duplicated().any():
        raise SystemExit(f"[e6 build] bioguide twice in the crosswalk: "
                         f"{cw.loc[cw['bioguide'].duplicated(keep=False), ['handle', 'bioguide']].values.tolist()}")
    info["inputs"] = {"network": {"path": str(args.network), "sha256": sha256_file(args.network), **net_st},
                      "crosswalk": {"path": str(args.crosswalk), "sha256": sha256_file(args.crosswalk),
                                    "rows": int(len(cw)), "auto": int((cw["method"] == "auto").sum()),
                                    "manual": int((cw["method"] == "manual").sum()),
                                    "notes": cw.loc[cw["note"].fillna("").str.startswith("corrected"),
                                                    ["handle", "bioguide", "note"]].to_dict(orient="records")}}
    mpath = Path(args.network).parent / "MANIFEST.json"
    if mpath.exists():
        mf = read_json(mpath).get("files", {})
        for key, p in (("network", args.network), ("crosswalk", args.crosswalk)):
            want = (mf.get(Path(p).name) or {}).get("sha256")
            if want and want != info["inputs"][key]["sha256"]:
                LOG.warning("%s: sha256 differs from %s", p, mpath)
            info["inputs"][key]["manifest_sha256_match"] = (want == info["inputs"][key]["sha256"]) if want else None

    # ---- Voteview
    vvp = data / "voteview" / "HS117_members.csv"
    vv, vv_st = load_voteview(vvp)
    info["inputs"]["voteview"] = {"path": str(vvp), "sha256": sha256_file(vvp), **vv_st}
    LOG.info("Voteview: %d members with dim1 (%d rows)", len(vv), vv_st["rows"])

    # ---- BILLSTATUS
    files = bill_files(data, args.types)
    LOG.info("parsing %d BILLSTATUS files (%d processes)", len(files), args.procs)
    bills, ev, bs_st = parse_bills(files, args.procs)
    info["billstatus"] = {k: v for k, v in bs_st.items() if k not in ("parse_errors", "type_or_congress_mismatch")}
    info["billstatus"]["parse_errors"] = bs_st["parse_errors"][:50]
    info["billstatus"]["n_parse_errors"] = len(bs_st["parse_errors"])
    info["billstatus"]["type_or_congress_mismatch"] = bs_st["type_or_congress_mismatch"][:50]
    LOG.info("BILLSTATUS: %d bills, %d events (%d withdrawn cosponsorships)", bs_st["bills"], bs_st["events"],
             bs_st["cosponsorships_withdrawn"])
    sp_has = bills["sponsor"].isin(vv.index)
    info["billstatus"]["bills_sponsor_with_dim1"] = int(sp_has.sum())
    n_bad = len(bs_st["parse_errors"]) + len(bs_st["type_or_congress_mismatch"])
    if n_bad and not args.allow_incomplete:
        info["finished"] = now_iso()
        write_json(info, pub / "e6_build.json")
        LOG.error("%d BILLSTATUS files could not be used (parse errors or wrong congress/type, see e6_build.json): "
                  "the bill data are incomplete; plan section 10: stop and report (--allow-incomplete overrides)", n_bad)
        return 2

    # ---- tolerances
    cos = train_cosponsorships(ev, bills)
    info["train_cosponsorships"] = {"all": int(len(cos)), "sponsor_missing": int(cos["sponsor"].isna().sum()),
                                    "sponsor_without_dim1": int((cos["sponsor"].notna()
                                                                 & ~cos["sponsor"].isin(vv.index)).sum())}
    t1 = tolerances(cos, vv, ["dim1"])
    t2 = tolerances(cos, vv, ["dim1", "dim2"])

    # ---- members table
    m = cw.copy()
    m["in_voteview_any_row"] = m["bioguide"].isin(set(vv.index) | set(vv_st["bioguides_without_any_dim1_row"]))
    j = vv[["dim1", "dim2", "chamber", "state_abbrev", "party_code", "party", "caucus", "icpsr", "bioname",
            "nominate_number_of_votes"]].rename(columns={"chamber": "vv_chamber", "state_abbrev": "vv_state"})
    m = m.join(j, on="bioguide")
    m["n_train_cosponsorships"] = m["bioguide"].map(t1["n"]).fillna(0).astype(int)
    m["n_train_cosponsorships_2d"] = m["bioguide"].map(t2["n"]).fillna(0).astype(int)
    for name in QNAMES:
        m[f"theta_{name}_1d"] = m["bioguide"].map(t1[name])
        m[f"theta_{name}_2d"] = m["bioguide"].map(t2[name])
        m[f"theta_raw_{name}_1d"] = m["bioguide"].map(t1[f"raw_{name}"])
        m[f"theta_raw_{name}_2d"] = m["bioguide"].map(t2[f"raw_{name}"])
    has_dim1 = m["dim1"].notna()
    enough = m["n_train_cosponsorships"] >= MIN_TRAIN_COSPONSORSHIPS
    m["eligible"] = has_dim1 & enough
    m["reason"] = np.where(m["eligible"], "ok",
                           np.where(~m["in_voteview_any_row"], "not_in_voteview",
                                    np.where(~has_dim1, "no_nominate_dim1",
                                             f"lt{MIN_TRAIN_COSPONSORSHIPS}_train_cosponsorships")))
    # chamber / state for the analysis: Voteview's row (crosswalk's as fallback); mismatches logged
    mis = m[has_dim1 & ((m["vv_chamber"] != m["chamber"]) | (m["vv_state"] != m["state"]))]
    if len(mis):
        LOG.warning("%d members whose Voteview chamber/state differ from the crosswalk: %s", len(mis),
                    mis[["handle", "bioguide", "chamber", "vv_chamber", "state", "vv_state"]].values.tolist()[:10])
    m = m.rename(columns={"chamber": "cw_chamber", "state": "cw_state"})
    m["chamber"] = m["vv_chamber"].fillna(m["cw_chamber"])
    m["state"] = m["vv_state"].fillna(m["cw_state"])
    elig = m[m["eligible"]].sort_values("bioguide", kind="mergesort")
    idx_of = pd.Series(np.arange(len(elig)), index=elig["bioguide"].to_numpy())
    m["idx"] = m["bioguide"].map(idx_of).fillna(-1).astype(int)
    n = int(len(elig))
    cols = ["idx", "bioguide", "handle", "node", "name", "bioname", "chamber", "state", "party", "caucus",
            "party_code", "icpsr", "nominate_number_of_votes", "dim1", "dim2", "n_train_cosponsorships",
            "n_train_cosponsorships_2d"] + [f"theta_{q}_1d" for q in QNAMES] + [f"theta_{q}_2d" for q in QNAMES] + \
           [f"theta_raw_{q}_1d" for q in QNAMES] + [f"theta_raw_{q}_2d" for q in QNAMES] + \
           ["eligible", "reason", "method", "note", "cw_chamber", "cw_state"]
    m = m.sort_values("node", kind="mergesort")
    m[cols].to_csv(pub / "e6_members.csv", index=False)
    ev[EVENT_COLS].assign(period=ev["period"].map(dict(enumerate(PERIOD_NAMES)))).to_csv(
        pub / "e6_events.csv.gz", index=False, compression="gzip")
    info["members"] = {
        "network_accounts": int(len(m)), "matched_voteview_dim1": int(has_dim1.sum()), "eligible": n,
        "reasons": {k: int(v) for k, v in m["reason"].value_counts().items()},
        "eligible_by_caucus_chamber": {f"{a}-{b}": int(c) for (a, b), c in
                                       elig.groupby(["caucus", "chamber"]).size().items()},
        "eligible_by_party": {k: int(v) for k, v in elig["party"].value_counts().items()},
        "chamber_or_state_mismatch_crosswalk_voteview": int(len(mis)),
        "floor_hits": {f"{q}_{dim}": int((elig[f"theta_raw_{q}_{dim}"] < THETA_FLOOR).sum())
                       for dim in ("1d", "2d") for q in QNAMES},
        "theta_median": {f"{q}_{dim}": float(elig[f"theta_{q}_{dim}"].median()) if n else None
                         for dim in ("1d", "2d") for q in QNAMES},
        "n_train_cosponsorships_median": float(elig["n_train_cosponsorships"].median()) if n else None}
    gate = n >= args.min_eligible
    info["gate"] = {"eligible": n, "required": args.min_eligible, "pass": bool(gate),
                    "rule": "plan section 2: members matched to Voteview with an estimable tolerance >= 400"}
    if not gate:
        info["finished"] = now_iso()
        write_json(info, pub / "e6_build.json")
        LOG.error("E6 DATA GATE FAILED: %d eligible members < %d. Plan section 2: stop and report (%s).", n,
                  args.min_eligible, pub / "e6_build.json")
        return 3
    LOG.info("data gate passed: %d eligible members (reasons %s)", n, info["members"]["reasons"])

    # ---- Twitter graph among members
    b_of_node = cw.set_index("node")["bioguide"]
    si = idx_of.reindex(b_of_node.loc[s_node].to_numpy()).to_numpy()
    di = idx_of.reindex(b_of_node.loc[d_node].to_numpy()).to_numpy()
    keep = ~(np.isnan(si) | np.isnan(di))
    src, dst, wk = si[keep].astype(np.int64), di[keep].astype(np.int64), w[keep]
    A = edges_to_csr(n, src, dst)
    write_graph(gdir / "graph.bin", A)
    # norm_mean: the plan's normaliser of the observed influence weights (mean over all network edges)
    np.savez(geo / "twitter_weights.npz", src=src, dst=dst, w=wk.astype(np.float64),
             norm_mean=np.float64(w[w > 0].mean() if np.any(w > 0) else 1.0))
    bio = elig["bioguide"].to_numpy()
    pd.DataFrame({"src": src, "dst": dst, "weight": wk, "src_bioguide": bio[src], "dst_bioguide": bio[dst],
                  "src_node": s_node[keep], "dst_node": d_node[keep]}).to_csv(pub / "e6_twitter_edges.csv",
                                                                              index=False)

    # ---- geo dataset: users, positions
    users = pd.DataFrame({"idx": np.arange(n), "user": bio, "n_train": elig["n_train_cosponsorships"].to_numpy()})
    for q in QNAMES:  # ties accepted in float32 too (common.ENGINE_TIE_RTOL); raw values in e6_members.csv
        users[f"theta_{q}"] = elig[f"theta_{q}_1d"].to_numpy(np.float64) * (1.0 + ENGINE_TIE_RTOL) + ENGINE_TIE_ATOL
    for q in QNAMES:
        users[f"theta2d_{q}"] = (elig[f"theta_{q}_2d"].to_numpy(np.float64) * (1.0 + ENGINE_TIE_RTOL)
                                 + ENGINE_TIE_ATOL)
    users["n_train_2d"] = elig["n_train_cosponsorships_2d"].to_numpy()
    for c in ("dim1", "dim2", "party", "caucus", "chamber", "state", "handle", "node", "name"):
        users[c] = elig[c].to_numpy()
    users.to_csv(geo / "users.csv", index=False)
    d1 = elig["dim1"].to_numpy(np.float64)
    d2 = elig["dim2"].to_numpy(np.float64)
    np.save(geo / "positions.npy", np.column_stack([d1, np.zeros(n), np.zeros(n)]).astype(np.float64))
    np.save(geo / "positions_2d.npy", np.column_stack([d1, d2, np.zeros(n)]).astype(np.float64))

    # ---- joins of members: co-visits per period, test check-ins
    jn_all = joins_of(ev)
    jt_all = jn_all[jn_all["period"] == 2]  # every member's test-window joins (for the notop counts)
    jn = jn_all[jn_all["member"].isin(idx_of.index) & (jn_all["period"] >= 0)]
    ju = idx_of.loc[jn["member"]].to_numpy(np.int64)
    jk = jn["bill_id"].to_numpy(np.int64)
    jp = jn["period"].to_numpy()
    mb = int(GEO_DEFAULTS["max_bucket_users"]) if args.max_bucket_users is None else int(args.max_bucket_users)
    cov = {name: covisit_pairs(ju[jp == p], jk[jp == p], A.indptr, A.indices, mb)
           for p, name in enumerate(PERIOD_NAMES)}
    np.savez(geo / "covisits.npz", **cov)
    mt = jp == 2
    tu, tk = ju[mt].astype(np.int32), jk[mt].astype(np.int64)
    np.savez(geo / "test_checkins.npz", user=tu, key=tk)
    keep_nt, notop = notop_filter(tu, tk, jt_all["member"].astype(str).to_numpy(),
                                  jt_all["bill_id"].to_numpy(np.int64))
    np.savez(geo / "test_checkins_notop.npz", user=tu[keep_nt], key=tk[keep_nt])

    deg = np.diff(A.indptr)
    t_all = float(w.mean()) if len(w) else None
    meta = {
        "application": "e6_congress", "n_active": n, "n_edges": int(A.nnz // 2), "n_nonisolated": int((deg > 0).sum()),
        "n_directed_edges": int(len(src)), "tolerance_quantiles": list(THETA_QUANTILES), "tolerance_floor": THETA_FLOOR,
        "theta_tie_inflation": {"rtol": ENGINE_TIE_RTOL, "atol": ENGINE_TIE_ATOL,
                                "rule": "users.csv theta = theta * (1 + rtol) + atol (exact ties accepted in float32)"},
        "tolerance_median": {q: float(np.median(users[f"theta_{q}"])) for q in QNAMES},
        "tolerance2d_median": {q: float(np.median(users[f"theta2d_{q}"])) for q in QNAMES},
        "covisit_friend_pairs": {k: int(len(v)) for k, v in cov.items()},
        "checkins_by_period": {nm: int((jp == p).sum()) for p, nm in enumerate(PERIOD_NAMES)},
        "periods": PERIODS, "key": f"integer bill id = type code * {BILL_ID_MULT} + number, codes {TYPE_CODE}",
        "positions": "positions.npy = (nominate_dim1, 0, 0); positions_2d.npy = (dim1, dim2, 0); float64",
        "member_order": "sorted bioguide (users.csv 'user')", "max_bucket_users": mb,
        "test_notop": {k: v for k, v in notop.items() if k != "removed_bill_ids"},
        "twitter_weights": {"file": "twitter_weights.npz (src, dst, w): edge src -> dst means dst retweeted, quoted, "
                                    "replied to or mentioned src; w = count / number of tweets by src",
                            "mean_w_all_edges": t_all, "mean_w_member_edges": float(wk.mean()) if len(wk) else None},
        "joins": "sponsor at introducedDate, cosponsor at sponsorshipDate; withdrawn cosponsorships excluded"}
    write_json(meta, geo / "meta.json")
    info["graph"] = {"members": n, "undirected_edges": meta["n_edges"], "directed_edges": int(len(src)),
                     "nonisolated": meta["n_nonisolated"], "mean_degree": float(deg.mean()) if n else None}
    info["joins"] = {"member_joins_by_period": meta["checkins_by_period"],
                     "covisit_friend_pairs": meta["covisit_friend_pairs"], "test_notop": notop,
                     "test_active_members": int(len(np.unique(tu)))}
    info["outputs"] = {"public": sorted(p.name for p in pub.iterdir() if p.is_file()),
                       "geo": sorted(p.name for p in geo.iterdir() if p.is_file()), "graph": ["graph.bin"]}
    info["finished"] = now_iso()
    write_json(info, pub / "e6_build.json")
    LOG.info("e6 build done: %d members, %d graph edges, test joins %d (notop %d), covisit pairs %s", n,
             meta["n_edges"], len(tu), int(keep_nt.sum()), meta["covisit_friend_pairs"])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m e6.build", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="data directory written by e6.fetch")
    ap.add_argument("--out", required=True, help="geo run directory (e.g. runs/e6)")
    ap.add_argument("--network", default=str(DATA_DIR / "congress_network_data.json"))
    ap.add_argument("--crosswalk", default=str(DATA_DIR / "crosswalk.csv"))
    ap.add_argument("--types", nargs="+", default=list(BILL_TYPES), choices=list(BILL_TYPES))
    ap.add_argument("--min-eligible", type=int, default=MIN_ELIGIBLE, help="data gate (plan: 400)")
    ap.add_argument("--max-bucket-users", type=int, default=None, help="default: lacs.apps.geo GEO_DEFAULTS")
    ap.add_argument("--procs", type=int, default=min(8, os.cpu_count() or 1), help="processes for XML parsing")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="continue although fetch_manifest.json says the download is incomplete or some BILLSTATUS "
                         "files cannot be parsed (the plan says stop and report)")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
