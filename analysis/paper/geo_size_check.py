"""Size decomposition of the meetup co-visit comparisons (round 4, k = 3), added 2026-10-22
after the Section 6 fact-check.

Usage: python3 -I geo_size_check.py <runs_dir> <out.json>
For LCC (exact) against sac_sized, kr_tau and pairwise, on queries where both are non-empty:
  - share of queries on which the baseline is larger / of equal size / identical
    (identical = equal size and equal diameter, active members, both rates and co-visit pairs,
    the proxy used by make_figures.equal_size_rows);
  - mean paired difference (pp) of the pair rate and of the member meet rate (preregistered
    secondary measure) on larger-baseline and equal-size queries;
  - on equal-size, non-identical queries: both differences with query-bootstrap 95% CIs.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

runs, out_path = Path(sys.argv[1]), sys.argv[2]
rng = np.random.default_rng(20261022)
B = 2000
COLS = ["diam_km", "active_members", "covisit_pair_rate", "member_meet_rate", "n_covisit_pairs"]


def ci(x):
    if len(x) < 2:
        return None
    m = x[rng.integers(0, len(x), size=(B, len(x)))].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


out = {}
for ds in ["gowalla", "brightkite"]:
    pq = pd.read_csv(runs / ds / "results" / "r4" / "geo_e2_per_query.csv")
    lcc = pq[(pq.method == "lcc_exact") & (pq["size"] > 0)].set_index("q")
    rec = {}
    for base in ["sac_sized", "kr_tau", "pairwise"]:
        b = pq[(pq.method == base) & (pq["size"] > 0)].set_index("q")
        j = lcc.join(b, lsuffix="_l", rsuffix="_b", how="inner")
        larger = j.size_b > j.size_l
        equal = j.size_b == j.size_l
        ident = equal.copy()
        for c in COLS:
            ident &= np.isclose(j[f"{c}_l"].fillna(-1), j[f"{c}_b"].fillna(-1))
        r = {"n": int(len(j)), "larger": int(larger.sum()), "equal": int(equal.sum()), "identical": int(ident.sum())}
        for metric, tag in [("covisit_pair_rate", "pair"), ("member_meet_rate", "member")]:
            d = 100 * (j[f"{metric}_l"] - j[f"{metric}_b"])
            r[f"{tag}_diff_larger_pp"] = float(d[larger].mean())
            r[f"{tag}_diff_equal_pp"] = float(d[equal].mean())
            x = d[equal & ~ident].dropna().to_numpy()
            r[f"{tag}_diff_equal_nonidentical_pp"] = {"n": int(len(x)), "mean": float(x.mean()) if len(x) else None, "ci95": ci(x)}
        rec[base] = r
    out[ds] = rec

json.dump(out, open(out_path, "w", encoding="utf-8"), indent=1)
for ds, rec in out.items():
    for base, r in rec.items():
        p, m = r["pair_diff_equal_nonidentical_pp"], r["member_diff_equal_nonidentical_pp"]
        print(f"{ds:10s} {base:9s} n={r['n']} larger={r['larger']} equal={r['equal']} identical={r['identical']} "
              f"| pair: larger {r['pair_diff_larger_pp']:+.2f}, equal {r['pair_diff_equal_pp']:+.2f}, "
              f"equal&different {p['mean'] if p['mean'] is None else round(p['mean'], 2)} {p['ci95']} (n={p['n']}) "
              f"| member: larger {r['member_diff_larger_pp']:+.2f}, equal {r['member_diff_equal_pp']:+.2f}, "
              f"equal&different {m['mean'] if m['mean'] is None else round(m['mean'], 2)} {m['ci95']}")
