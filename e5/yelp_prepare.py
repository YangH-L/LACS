"""E5 main experiment, step 1: the geo-format inputs of the Yelp geographic application.

  python e5/yelp_prepare.py --json-dir data/yelp/json --work runs/e5_feas/work \
      --feas runs/e5_feas/out/feasibility.json --out-dir runs/e5_main

Rebuilds exactly what the feasibility check chose (split `precovid`, definition G; pre-registered
plan docs/PLANS.md (lacs_e5_prereg_draft_20261025), section 2) from the feasibility check's parse cache,
with the feasibility code itself (e5/yelp_feasibility.py), and writes the files lacs/apps/geo.py
reads (geo-gates, geo-e2, geo-e2-sweep):

  <out-dir>/geo/users.csv            idx, user (= idx; no Yelp id), n_train, metro, theta_q50/q70/q90 (km)
  <out-dir>/geo/positions.npy        homes as 3-D Earth-centred coordinates (km)
  <out-dir>/graph/graph.bin          same-metro friendships among the active users
  <out-dir>/geo/covisits.npz         friend pairs co-reviewing a business, per period (train/val/test)
  <out-dir>/geo/test_checkins*.npz   test-period (2019) reviews: key = business (main outcome),
                                     business x calendar month (_month), business without the top 1%
                                     businesses by 2019 reviews (_notop)
  <out-dir>/geo/meta.json, <out-dir>/results/e5_prepare.json

Consistency (plan section 2): the vertex and edge counts, the eligible queries at k = 2..5 and the
quantiles of theta_q50 must equal feasibility.json; any difference stops the run (exit 3) before a
file is written. Everything under <out-dir>/geo, graph and work is user-level data and never leaves
the server; results/e5_prepare.json holds counts only.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e5 import yelp_feasibility as F  # noqa: E402  (the feasibility definitions, reused as is)
from lacs.apps import geo  # noqa: E402
from lacs.graph.io import write_graph  # noqa: E402

VERSION = "e5-main-0.2"
LOG = logging.getLogger("e5prep")


class Mismatch(RuntimeError):
    pass


def _close(a, b, rel=1e-6) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= rel * max(1.0, abs(float(b)))


def _cache_reusable(work: Path, inv: dict) -> bool:
    """Same test as yelp_feasibility.load_or_parse, without writing anything."""
    key = {"files": {k: next(f["sha256"] for f in inv["files"] if f["kind"] == k) for k in ("business", "user", "review")},
           "code": F._parse_code_hash()}
    try:
        with np.load(work / "parsed.npz") as z:
            return json.loads(str(z["meta_json"])).get("key") == key
    except Exception:
        return False


def build(json_dir: Path, work: Path, feas_path: Path, parse_dir: Path | None = None) -> dict:
    """Recompute definition G on split precovid. Returns arrays and the consistency report."""
    R = json.loads(feas_path.read_text())
    dec = R.get("decision") or {}
    if (dec.get("split") or {}).get("split") != "precovid" or (dec.get("spec") or {}).get("spec") != "G":
        raise Mismatch(f"feasibility decision is not (precovid, G): {dec.get('split', {}).get('split')}, "
                       f"{dec.get('spec', {}).get('spec')}")
    P = R["params"]
    ref = R["splits"]["precovid"]

    # the same JSON files as the feasibility check (sha256 of every file)
    inv = F.inventory_dir(json_dir)
    want = {f["kind"]: f["sha256"] for f in R["inventory"]["files"]}
    have = {f["kind"]: f["sha256"] for f in inv["files"]}
    if any(have.get(k) != v for k, v in want.items()):
        raise Mismatch(f"JSON files differ from the feasibility check: {sorted(k for k in want if have.get(k) != want[k])}")
    fpath = {f["kind"]: json_dir / f["file"] for f in inv["files"]}
    if not _cache_reusable(work, inv):
        # never rewrite the feasibility check's cache: parse into our own work directory instead
        work = parse_dir or work.parent / "e5_main_parse"
        LOG.warning("feasibility parse cache not reusable: parsing again into %s (about 30 minutes)", work)
    (blat, blon, bstate, bcity, bcats, bstats, fs, fd, deg_listed, ustats, U, B, D, rstats, NU, NB) = \
        F.load_or_parse(fpath, inv, work)
    reused = F.parsed_reused(work)
    LOG.info("parsed arrays: %d users, %d businesses, %d reviews (cache reused: %s)", NU, NB, len(U), reused)

    # friendship graph (union of both directions), metros: exactly as in the feasibility main()
    lo, hi = np.minimum(fs, fd), np.maximum(fs, fd)
    und = np.unique(lo * NU + hi)
    eu, ev = und // NU, und % NU
    del lo, hi, und, fs, fd
    bmetro, minfo = F.find_metros(blat, blon, bstate, bcity, P["metro_cell"], P["metro_min_business"])
    M = sum(1 for m in minfo if m["metro"] >= 0)

    # analyse_split("precovid"): active users with a home metro, taste-ready set V, same-metro graph
    per, bounds = F.split_periods(D, "precovid")
    tr = per == 0
    n_train = np.bincount(U[tr], minlength=NU)
    n_test = np.bincount(U[per == 2], minlength=NU)
    home, _ = F.plurality(U[tr], bmetro[B[tr]], NU, M)
    act_home = (n_train >= P["min_train"]) & (home >= 0)
    used = np.unique(B[tr & act_home[U]])
    Bv, _, _ = F.business_vectors(bcats, used, P["vocab_min_business"])
    has_vec = np.diff(Bv.indptr) > 0
    mt = tr & act_home[U] & has_vec[B]
    n_taste = np.bincount(U[mt], minlength=NU)
    ready = act_home & (n_taste >= P["min_taste"])
    V = np.flatnonzero(ready)
    nV = len(V)
    vi = np.full(NU, -1, np.int64)
    vi[V] = np.arange(nV)
    keep = (vi[eu] >= 0) & (vi[ev] >= 0) & (home[eu] == home[ev])
    A = F.edges_to_csr(nV, vi[eu[keep]], vi[ev[keep]])
    core = F.core_numbers(A)
    test_act = n_test[V] >= 1

    # definition G (the feasibility block, plus the 70% / 90% quantiles of the same distances)
    mg = tr & ready[U]
    gu = vi[U[mg]]
    glat, glon = blat[B[mg]], blon[B[mg]]
    okc = np.isfinite(glat) & np.isfinite(glon)
    gu, glat, glon = gu[okc], glat[okc], glon[okc]
    cell = np.floor(glat / 0.25).astype(np.int64) * 100000 + np.floor(glon / 0.25).astype(np.int64)
    o = np.lexsort((cell, gu))
    gu_s, cell_s = gu[o], cell[o]
    grp_start = np.flatnonzero(np.r_[True, (gu_s[1:] != gu_s[:-1]) | (cell_s[1:] != cell_s[:-1])])
    grp_cnt = np.diff(np.r_[grp_start, len(gu_s)])
    grp_u, grp_c = gu_s[grp_start], cell_s[grp_start]
    oo = np.lexsort((grp_c, -grp_cnt, grp_u))
    firsts = oo[np.r_[True, grp_u[oo][1:] != grp_u[oo][:-1]]]
    best_cell = np.full(nV, np.iinfo(np.int64).min)
    best_cell[grp_u[firsts]] = grp_c[firsts]
    inm = cell == best_cell[gu]
    hlat = F.group_quantile(gu[inm], glat[inm], 0.5, nV)
    hlon = F.group_quantile(gu[inm], glon[inm], 0.5, nV)
    miss = ~(np.isfinite(hlat) & np.isfinite(hlon))
    hlat[miss], hlon[miss] = 0.0, 0.0
    dist = F.haversine_km(hlat[gu], hlon[gu], glat, glon)
    theta = {}
    for q in (0.5, 0.7, 0.9):
        t = F.group_quantile(gu, dist, q, nV)
        theta[f"theta_q{int(round(q * 100))}"] = np.where(np.isfinite(t), np.maximum(t, 1.0), 0.0)
    pos = F.to_xyz(hlat, hlon)

    # consistency with feasibility.json (plan section 2)
    rep = {"vertices": [int(nV), int(ref["graph"]["vertices"])],
           "edges": [int(A.nnz // 2), int(ref["graph"]["edges"])],
           "taste_ready": [int(nV), int(ref["users"]["taste_ready"])],
           "home_missing": [int(miss.sum()), int(ref["taste"]["G_home_missing"])]}
    for k in (2, 3, 4, 5):
        rep[f"eligible_k{k}"] = [int(((core >= k) & test_act).sum()), int(ref["eligible_queries"][f"k{k}"])]
    bad = [k for k, (a, b) in rep.items() if a != b]
    qd = F.qdict(theta["theta_q50"])
    for k, v in ref["taste"]["G"]["theta"].items():
        rep[f"theta_q50_{k}"] = [qd.get(k), v]
        if not _close(qd.get(k), v):
            bad.append(f"theta_q50_{k}")
    if bad:
        raise Mismatch("rebuilt inputs differ from feasibility.json: "
                       + "; ".join(f"{k}: {rep[k][0]} vs {rep[k][1]}" for k in bad))
    return {"A": A, "pos": pos, "theta": theta, "V": V, "vi": vi, "U": U, "B": B, "D": D, "per": per,
            "NB": NB, "n_train": n_train[V], "metro": home[V], "bounds": bounds, "consistency": rep,
            "parse_cache_reused": bool(reused), "n_reviews": int(len(U))}


def write_outputs(o: dict, out_dir: Path, max_bucket: int) -> dict:
    A, vi, U, B, D, per = o["A"], o["vi"], o["U"], o["B"], o["D"], o["per"]
    N = A.shape[0]
    root = out_dir / "geo"
    for d in (root, out_dir / "graph", out_dir / "results", out_dir / "work"):
        d.mkdir(parents=True, exist_ok=True)
    users = pd.DataFrame({"idx": np.arange(N), "user": np.arange(N), "n_train": o["n_train"], "metro": o["metro"]})
    for c, t in o["theta"].items():
        users[c] = t
    sel = vi[U] >= 0  # reviews of active users
    au, ab, ad, ap = vi[U[sel]], B[sel], D[sel], per[sel]
    cov = {name: geo.covisit_pairs(au[ap == p], ab[ap == p], A.indptr, A.indices, max_bucket)
           for p, name in enumerate(("train", "val", "test"))}
    mt = ap == 2
    tu, tb, td = au[mt].astype(np.int32), ab[mt].astype(np.int64), ad[mt]
    month = td.astype("datetime64[D]").astype("datetime64[M]").astype(np.int64) % 12
    # top 1% businesses by 2019 reviews of everybody (the feasibility check's threshold)
    bc = np.bincount(B[per == 2], minlength=o["NB"])
    pc = bc[bc > 0]
    thr = float(np.quantile(pc, 0.99)) if len(pc) else np.inf
    nt = bc[tb] < thr
    write_graph(out_dir / "graph" / "graph.bin", A)
    np.save(root / "positions.npy", o["pos"].astype(np.float64))
    users.to_csv(root / "users.csv", index=False)
    np.savez(root / "covisits.npz", **cov)
    np.savez(root / "test_checkins.npz", user=tu, key=tb)
    np.savez(root / "test_checkins_month.npz", user=tu, key=tb * 12 + month)
    np.savez(root / "test_checkins_notop.npz", user=tu[nt], key=tb[nt])
    deg = np.diff(A.indptr)
    meta = {"version": VERSION, "source": "Yelp Open Dataset (feasibility parse cache), split precovid, definition G",
            "n_active": int(N), "n_edges": int(A.nnz // 2), "n_nonisolated": int((deg > 0).sum()),
            "boundaries": o["bounds"], "tolerance_quantiles": [0.5, 0.7, 0.9], "tolerance_floor_km": 1.0,
            "tolerance_km_median": {c.replace("theta_", ""): float(np.median(t)) for c, t in o["theta"].items()},
            "covisit_friend_pairs": {k: int(len(v)) for k, v in cov.items()},
            "checkins_by_period": {n: int((ap == p).sum()) for p, n in enumerate(("train", "val", "test"))},
            "test_reviews": int(mt.sum()), "test_reviews_notop": int(nt.sum()),
            "top1pct_threshold_reviews": thr, "top1pct_businesses": int((bc >= thr).sum()),
            "max_bucket_users": int(max_bucket),
            "outcome_keys": {"main": "business", "month": "business x calendar month", "notop": "business, without "
                             "the top 1% businesses by 2019 reviews"}}
    (root / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-dir", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("runs/e5_feas/work"), help="feasibility parse cache")
    ap.add_argument("--feas", type=Path, default=Path("runs/e5_feas/out/feasibility.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs/e5_main"))
    ap.add_argument("--max-bucket-users", type=int, default=geo.GEO_DEFAULTS["max_bucket_users"])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    t0 = time.time()
    (args.out_dir / "results").mkdir(parents=True, exist_ok=True)
    rep_path = args.out_dir / "results" / "e5_prepare.json"
    try:
        o = build(args.json_dir, args.work, args.feas, parse_dir=args.out_dir / "work" / "parse")
    except Mismatch as e:
        rep_path.write_text(json.dumps({"version": VERSION, "status": "MISMATCH", "error": str(e)}, indent=1))
        LOG.error("E5 PREPARE STOPPED: %s", e)
        print("E5 PREPARE MISMATCH")
        return 3
    meta = write_outputs(o, args.out_dir, args.max_bucket_users)
    out = {"version": VERSION, "status": "OK", "consistency": o["consistency"], "meta": meta,
           "parse_cache_reused": o["parse_cache_reused"], "seconds": round(time.time() - t0, 1)}
    rep_path.write_text(json.dumps(out, indent=1))
    LOG.info("E5 inputs written to %s: %d users, %d edges, eligible k3 %d (matches feasibility)", args.out_dir,
             meta["n_active"], meta["n_edges"], o["consistency"]["eligible_k3"][0])
    print("E5 PREPARE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
