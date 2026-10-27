#!/usr/bin/env python3
"""E6 structure check (plan section 4): does the tolerance filter anything on this graph?

  python -m e6.structure --run runs/e6 [--per-query runs/e6/results/geo_e2_per_query.csv] [--results DIR]

Reports
  * the share of graph edges (friend pairs) whose endpoints are mutually within tolerance:
    d_uv <= theta_u and d_uv <= theta_v (theta_q50, 1-D positions of RUN/geo; exact ties count as
    within, relative tolerance e6.common.TIE_RTOL);
  * from geo-e2's per-query table (rows of methods lcc_exact and kcore): the share of queries with a
    non-empty LCC (size >= 2, as in geo-e2) and, over those queries, the median of
    LCC size / size of the query's connected k-core component;
  * the rule: median >= 0.9 means the tolerance does not filter (as in E5's taste definition); Q2-Q4
    are then descriptive only and written up as a structural observation.
Missing per-query table: the LCC part is reported as missing. Output: RESULTS/e6_structure.json.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # also works under python -I
    sys.path.insert(0, str(ROOT))

from e6.common import TIE_RTOL, now_iso, setup_logging, write_json  # noqa: E402

LOG = logging.getLogger("e6.structure")
RATIO_RULE = 0.9


def mutual_share(A, pos: np.ndarray, theta: np.ndarray) -> dict:
    import scipy.sparse as sp

    up = sp.triu(A, k=1).tocoo()
    u, v = up.row.astype(np.int64), up.col.astype(np.int64)
    d = np.sqrt(((pos[u] - pos[v]) ** 2).sum(axis=1))
    au, av = d <= theta[u] * (1 + TIE_RTOL), d <= theta[v] * (1 + TIE_RTOL)  # ties count as within
    both = au & av
    one = au ^ av
    return {"edges": int(len(u)), "mutual_within": int(both.sum()),
            "share_mutual_within": float(both.mean()) if len(u) else None,
            "share_one_side_only": float(one.mean()) if len(u) else None,
            "share_neither": float((~both & ~one).mean()) if len(u) else None}


def lcc_vs_kcore(df: pd.DataFrame, k: int | None = None) -> dict:
    if k is not None and "k" in df.columns:
        df = df[df["k"] == k]
    lcc = df[df["method"] == "lcc_exact"].drop_duplicates("q").set_index("q")
    kc = df[df["method"] == "kcore"].drop_duplicates("q").set_index("q")
    if lcc.empty:
        return {"missing": "no lcc_exact rows in the per-query table"}
    nonempty = lcc["size"] >= 2
    out = {"queries": int(len(lcc)), "lcc_nonempty": int(nonempty.sum()),
           "share_lcc_nonempty": float(nonempty.mean()),
           "lcc_status_counts": {str(k_): int(v) for k_, v in lcc["status"].value_counts().items()}
           if "status" in lcc.columns else None}
    q = lcc.index[nonempty]
    if kc.empty:
        out["ratio"] = {"missing": "no kcore rows in the per-query table"}
        return out
    common = q.intersection(kc.index)
    ks = kc.loc[common, "size"].astype(float)
    ls = lcc.loc[common, "size"].astype(float)
    ok = ks > 0
    r = (ls[ok] / ks[ok]).to_numpy()
    out["ratio"] = {"queries": int(len(r)), "median": float(np.median(r)) if len(r) else None,
                    "mean": float(r.mean()) if len(r) else None,
                    "q25_q75": [float(np.percentile(r, 25)), float(np.percentile(r, 75))] if len(r) else None,
                    "lcc_size_median": float(ls.median()) if len(ls) else None,
                    "kcore_size_median": float(ks.median()) if len(ks) else None,
                    "nonempty_without_kcore_row": int(len(q) - len(common))}
    med = out["ratio"]["median"]
    out["no_filtering"] = None if med is None else bool(med >= RATIO_RULE)
    out["rule"] = (f"median(LCC size / connected k-core component size) over non-empty queries >= {RATIO_RULE}: "
                   "the tolerance does not filter; Q2-Q4 are descriptive only (structural observation)")
    return out


def run(args) -> int:
    from lacs.graph.io import read_graph

    run_dir = Path(args.run)
    res = Path(args.results) if args.results else run_dir / "results"
    res.mkdir(parents=True, exist_ok=True)
    setup_logging(res / "e6_structure.log")
    users = pd.read_csv(run_dir / "geo" / "users.csv", float_precision="round_trip")
    pos = np.load(run_dir / "geo" / "positions.npy").astype(np.float64)
    A = read_graph(run_dir / "graph" / "graph.bin")
    theta = users[args.tolerance].to_numpy(np.float64)
    out = {"tool": "e6.structure", "created": now_iso(), "run": str(run_dir), "tolerance": args.tolerance,
           "members": int(len(users)), "friend_pairs": mutual_share(A, pos, theta)}
    pq = Path(args.per_query) if args.per_query else res / "geo_e2_per_query.csv"
    if pq.exists():
        out["per_query_file"] = str(pq)
        out["lcc"] = lcc_vs_kcore(pd.read_csv(pq), args.k)
    else:
        out["per_query_file"] = None
        out["lcc"] = {"missing": f"{pq} not found (run geo-e2 first)"}
    write_json(out, res / "e6_structure.json")
    fp = out["friend_pairs"]
    LOG.info("structure: %d friend pairs, mutually within tolerance %.3f", fp["edges"], fp["share_mutual_within"] or 0)
    if "ratio" in out["lcc"] and isinstance(out["lcc"]["ratio"], dict) and "median" in out["lcc"]["ratio"]:
        LOG.info("LCC non-empty share %.3f, median LCC / k-core %.3f, no_filtering=%s",
                 out["lcc"]["share_lcc_nonempty"], out["lcc"]["ratio"]["median"] or float("nan"),
                 out["lcc"].get("no_filtering"))
    else:
        LOG.info("LCC part: %s", out["lcc"])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m e6.structure", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", "--geo", dest="run", required=True, help="geo run directory (e.g. runs/e6)")
    ap.add_argument("--per-query", default=None, help="geo-e2 per-query CSV (default RESULTS/geo_e2_per_query.csv)")
    ap.add_argument("--results", "--out", dest="results", default=None, help="default: RUN/results")
    ap.add_argument("--tolerance", default="theta_q50")
    ap.add_argument("--k", type=int, default=None, help="filter the per-query table to this k if it has a k column")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
