"""P2-5 analysis, plan docs/PLANS.md (lacs_p2_plan_20261026) section 5.3 (written before the v0.5 results were seen,
tested on the rehearsal outputs).

  python3 -I p25_analysis.py <root containing runs_p2/> <P0 runs_p0 root> <out dir>

Per dataset and setting (v0.4 = P0's run, v0.5 = this run; same instances and queries):
  * share of queries with an LCC: any method whose non-empty answer satisfies (iii) (exact, greedy, BallCore-F)
    found one, in either version (P0 results document section 7, item 1);
  * on the queries with an LCC: share proven optimal, share timed out, share too large (v0.4 -> v0.5), median
    exact time;
  * v0.5, timed-out queries with an LCC (plan 5.3): median and minimum of solution size / upper bound; the same
    for the too_large queries and for all unproven ones, reported separately (10-26, before any v0.5 result);
  * share of the queries with an LCC on which greedy finds one (v0.4 -> v0.5);
  * newly proven and no longer proven queries; proven-optimum mismatches (must be 0, the gate).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT, P0, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
WIT = ("exact", "greedy", "ballcore_f")
ORDER = {"e4_scale": 0, "e4_queries": 1, "e4_mix": 2}


def one(d, algo):
    return d[d["algo"] == algo].set_index("q")


rows = []
for pq in sorted((ROOT / "runs_p2" / "e4v05").glob("*/e4_*/*/per_query.csv")):
    ds, exp, st = pq.parts[-4], pq.parts[-3], pq.parts[-2]
    new = pd.read_csv(pq)
    old = pd.read_csv(P0 / ds / exp / st / "per_query.csv")
    x5, x4 = one(new, "exact"), one(old, "exact")
    assert set(x5.index) == set(x4.index), (ds, exp, st)
    qs = x5.index
    wit = pd.Series(False, index=qs)
    for d in (new, old):
        for a in WIT:
            if a in set(d["algo"]):
                wit |= one(d, a)["size"].reindex(qs).fillna(0) >= 2
    has = qs[wit.to_numpy()]
    r = {"dataset": ds, "experiment": exp, "setting": st, "queries": len(qs), "has_lcc": int(len(has)),
         "has_lcc_share": float(wit.mean())}
    for tag, x, d in (("v04", x4, old), ("v05", x5, new)):
        h = x.loc[has]
        r[f"{tag}_proven"] = float((h["status"] == "optimal").mean()) if len(has) else np.nan
        r[f"{tag}_timeout"] = float((h["status"] == "timeout").mean()) if len(has) else np.nan
        r[f"{tag}_too_large"] = float((h["status"] == "too_large").mean()) if len(has) else np.nan
        r[f"{tag}_time_median_ms"] = float(h["time_ms"].median()) if len(has) else np.nan
        g = one(d, "greedy").reindex(has)
        r[f"{tag}_greedy_finds"] = float((g["size"] >= 2).mean()) if len(has) else np.nan
        r[f"{tag}_exact_finds"] = float((h["size"] >= 2).mean()) if len(has) else np.nan
    hx = x5.loc[has]
    for lab, keep in (("timeout", hx["status"] == "timeout"), ("too_large", hx["status"] == "too_large"),
                      ("unproven", hx["status"] != "optimal")):
        un = hx[keep & (hx["ub"] > 0)]
        ratio = (un["size"] / un["ub"]).astype(float)
        r[f"v05_{lab}_with_ub"] = int(len(un))
        r[f"v05_{lab}_size_over_ub_median"] = float(ratio.median()) if len(un) else np.nan
        r[f"v05_{lab}_size_over_ub_min"] = float(ratio.min()) if len(un) else np.nan
    both = qs[((x5["status"] == "optimal") & (x4.loc[qs, "status"] == "optimal")).to_numpy()]
    r["both_proven"] = int(len(both))
    r["optimum_mismatches"] = int((x5.loc[both, "size"] != x4.loc[both, "size"]).sum())
    r["newly_proven"] = int(((x5["status"] == "optimal") & (x4.loc[qs, "status"] != "optimal")).sum())
    r["no_longer_proven"] = int(((x5["status"] != "optimal") & (x4.loc[qs, "status"] == "optimal")).sum())
    r["v05_timeouts_all_queries"] = float((x5["status"] != "optimal").mean())
    r["v04_timeouts_all_queries"] = float((x4["status"] != "optimal").mean())
    rows.append(r)
df = pd.DataFrame(rows)
df["_o"] = df["experiment"].map(ORDER)
df = df.sort_values(["_o", "dataset", "setting"]).drop(columns="_o")
df.to_csv(OUT / "p25_results.csv", index=False)
cols = ["dataset", "experiment", "setting", "has_lcc_share", "v04_proven", "v05_proven", "v04_timeout", "v05_timeout",
        "v04_too_large", "v05_too_large", "v04_time_median_ms", "v05_time_median_ms",
        "v05_timeout_size_over_ub_median", "v05_timeout_size_over_ub_min", "v05_too_large_with_ub", "v04_greedy_finds", "v05_greedy_finds", "newly_proven", "no_longer_proven",
        "optimum_mismatches"]
lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
for r in df[cols].itertuples(index=False):
    lines.append("| " + " | ".join(f"{x:.3g}" if isinstance(x, float) else str(x) for x in r) + " |")
lines.append(f"\nproven-optimum mismatches: {int(df['optimum_mismatches'].sum())} (must be 0)")
(OUT / "p25_results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
