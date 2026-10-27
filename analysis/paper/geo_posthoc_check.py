"""Recompute the post hoc meetup analyses quoted in the handover (round 4, k = 3).

Usage: python3 -I geo_posthoc_check.py <runs_dir> <out.json>
1. Abstention: OLS of the SAC co-visit pair rate on [1, LCC exists, log size], query-bootstrap CI.
2. H1 size control: for each baseline, regress the paired difference (LCC - baseline, pp) on
   log(size_baseline / size_LCC); the intercept is the difference at equal size.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

runs, out_path = Path(sys.argv[1]), sys.argv[2]
rng = np.random.default_rng(20261022)
B = 2000
out = {}


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


for ds in ["gowalla", "brightkite"]:
    pq = pd.read_csv(runs / ds / "results" / "r4" / "geo_e2_per_query.csv")
    lcc = pq[pq.method == "lcc_exact"].set_index("q")
    exists = (lcc["size"] > 0)
    rec = {}
    # --- 1. abstention, SAC groups ---
    sac = pq[(pq.method == "sac") & (pq["size"] > 0)].set_index("q")
    sac = sac.join(exists.rename("lcc_exists"), how="inner")
    sac = sac[sac.covisit_pair_rate.notna()]
    y = 100 * sac.covisit_pair_rate.to_numpy()
    X = np.column_stack([np.ones(len(sac)), sac.lcc_exists.astype(float), np.log(sac["size"].astype(float))])
    b = ols(X, y)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        bs.append(ols(X[i], y[i])[1])
    rec["abstention_sac"] = {
        "n_exists": int(sac.lcc_exists.sum()), "n_empty": int((~sac.lcc_exists).sum()),
        "raw_diff_pp": float(y[sac.lcc_exists.to_numpy()].mean() - y[~sac.lcc_exists.to_numpy()].mean()),
        "size_median_exists": float(sac.loc[sac.lcc_exists, "size"].median()),
        "size_median_empty": float(sac.loc[~sac.lcc_exists, "size"].median()),
        "adj_diff_pp": float(b[1]), "adj_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    # --- 2. H1 size control ---
    base_l = lcc[lcc["size"] > 0]
    for m in ["sac_sized", "kr_tau", "pairwise"]:
        x = pq[(pq.method == m) & (pq["size"] > 0)].set_index("q")
        j = base_l[["size", "covisit_pair_rate"]].join(x[["size", "covisit_pair_rate"]], how="inner",
                                                        lsuffix="_l", rsuffix="_b").dropna()
        dpp = 100 * (j.covisit_pair_rate_l - j.covisit_pair_rate_b).to_numpy()
        lr = np.log(j.size_b / j.size_l).to_numpy()
        Xm = np.column_stack([np.ones(len(j)), lr])
        bm = ols(Xm, dpp)
        bs = []
        for _ in range(B):
            i = rng.integers(0, len(dpp), len(dpp))
            bs.append(ols(Xm[i], dpp[i])[0])
        rec[f"h1_size_control_{m}"] = {"n": int(len(j)), "mean_diff_pp": float(dpp.mean()),
                                       "intercept_pp": float(bm[0]),
                                       "intercept_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                       "share_baseline_larger": float((j.size_b > j.size_l).mean())}
    out[ds] = rec

json.dump(out, open(out_path, "w", encoding="utf-8"), indent=1)
for ds, rec in out.items():
    a = rec["abstention_sac"]
    print(f"{ds}: abstention raw {a['raw_diff_pp']:+.2f} pp, size-adjusted {a['adj_diff_pp']:+.2f} "
          f"[{a['adj_ci95'][0]:+.2f}, {a['adj_ci95'][1]:+.2f}]; SAC size median {a['size_median_exists']:.0f} (LCC exists) "
          f"vs {a['size_median_empty']:.0f} (LCC empty); n {a['n_exists']}/{a['n_empty']}")
    for m in ["sac_sized", "kr_tau", "pairwise"]:
        h = rec[f"h1_size_control_{m}"]
        print(f"   H1 {m:9s} n={h['n']} mean {h['mean_diff_pp']:+.2f} intercept {h['intercept_pp']:+.2f} "
              f"[{h['intercept_ci95'][0]:+.2f}, {h['intercept_ci95'][1]:+.2f}] baseline larger {100*h['share_baseline_larger']:.0f}%")
