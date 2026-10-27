"""E3 (exact solvers in one sweep framework), with the corrections of the P0 results doc, section 7.

- "LCC exists": some solver found a non-empty solution (not only proven ones).
- Speed ratios stratified by difficulty: hard = bnb >= 100 ms, both proven.
Usage: python3 -I p0_e3_numbers.py <runs_p0_dir> <out_json>
"""
import json
import sys

import numpy as np
import pandas as pd

root, out = sys.argv[1], sys.argv[2]
SOLV = ["bnb", "krmax_simple", "krmax", "ilp", "cpsat", "engine_now"]
res = {}
for ds in ["livejournal", "orkut", "gowalla", "brightkite"]:
    d = pd.read_csv(f"{root}/{ds}/e3/e3_per_query.csv")
    d = d[d.solver != "engine_r3"]
    piv_size = d.pivot_table(index="query", columns="solver", values="size", aggfunc="first")
    piv_prov = d.pivot_table(index="query", columns="solver", values="proven", aggfunc="first").astype(bool)
    piv_t = d.pivot_table(index="query", columns="solver", values="time_ms", aggfunc="first")
    n = len(piv_size)
    has = (piv_size[[s for s in SOLV if s in piv_size]].max(axis=1) > 0)
    rec = {"n": int(n), "lcc": int(has.sum())}
    for s in SOLV:
        if s not in piv_size:
            continue
        rec[s] = {"proven": float(piv_prov[s].mean()), "proven_lcc": float(piv_prov[s][has].mean())}
        if s != "bnb":
            both = piv_prov["bnb"] & piv_prov[s]
            ratio = piv_t[s] / piv_t["bnb"]
            hard = both & (piv_t["bnb"] >= 100)
            mid = both & (piv_t["bnb"] >= 1) & (piv_t["bnb"] < 100)
            small = both & (piv_t["bnb"] < 1)
            rec[s]["n_hard"] = int(hard.sum())
            rec[s]["ratio_hard_median"] = float(ratio[hard].median()) if hard.any() else None
            rec[s]["ratio_mid_median"] = float(ratio[mid].median()) if mid.any() else None
            rec[s]["ratio_small_median"] = float(ratio[small].median()) if small.any() else None
            rec[s]["proven_by_s_not_bnb"] = int((piv_prov[s] & ~piv_prov["bnb"]).sum())
            rec[s]["proven_by_bnb_not_s"] = int((~piv_prov[s] & piv_prov["bnb"]).sum())
    # agreement of proven optima
    mism = 0
    for q in piv_size.index:
        vals = {piv_size.loc[q, s] for s in SOLV if s in piv_size and piv_prov.loc[q, s]}
        mism += int(len(vals) > 1)
    rec["proven_optimum_mismatches"] = mism
    none = ~piv_prov[[s for s in SOLV if s in piv_prov]].any(axis=1)
    rec["proven_by_none"] = int(none.sum())
    if none.any():
        best_other = piv_size.loc[none, [s for s in SOLV if s not in ("bnb",) and s in piv_size]].max(axis=1)
        rec["unproven_bnb_ge_best_other"] = bool((piv_size.loc[none, "bnb"] >= best_other).all())
        rec["unproven_sizes"] = [int(x) for x in piv_size.loc[none, "bnb"]]
    res[ds] = rec
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
