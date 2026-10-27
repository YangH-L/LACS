"""Recompute the E1 numbers that go into the paper text, from the P0 per-group rows.

Usage: python3 -I p0_e1_checks.py <runs_p0_dir> <out_json>
"""
import gzip
import json
import sys
from collections import defaultdict

import numpy as np

root, out = sys.argv[1], sys.argv[2]
CLOSED = ["hk_eq_G", "hk_eq_K", "hk_rand_G", "hk_rand_K", "hk_rand_G_a0.1", "hk_rand_K_a0.1",
          "follow_G", "follow_K", "dw_G", "dw_K", "hk_eq_G_long"]
SJT = ["sjt_eq_G", "sjt_eq_K"]
res = {"closed_iii_pairs": 0, "closed_iii_broken": 0, "closed_iii_groups": set(),
       "sjt_iii_pairs": 0, "sjt_iii_newrej": 0}
pw = defaultdict(lambda: {"n": 0, "viol_iii": 0})
pw_break = defaultdict(list)   # (ds, variant) -> list of (mean_broken_frac, p_broken)
for ds in ["gowalla", "brightkite", "livejournal", "orkut"]:
    with gzip.open(f"{root}/{ds}/e1/default/e1_rows.jsonl.gz", "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("skipped"):
                continue
            v, m = r["variant"], r["method"]
            if v in CLOSED and r["sat_iii"]:
                res["closed_iii_pairs"] += 1
                res["closed_iii_groups"].add((ds, r["query"], m))
                if r["p_broken"] > 0:
                    res["closed_iii_broken"] += 1
            if v in SJT and r["sat_iii"]:
                res["sjt_iii_pairs"] += 1
                if r["p_new_rejection"] > 0:
                    res["sjt_iii_newrej"] += 1
            if m == "pairwise" and v == "hk_eq_G":
                pw[ds]["n"] += 1
                pw[ds]["viol_iii"] += int(not r["sat_iii"])
            if m == "pairwise" and v in ("dw_G", "hk_rand_G_a0.1", "follow_G") and r["p_broken"] > 0:
                pw_break[(ds, v)].append((r["mean_broken_frac"], r["p_broken"], r["p_consensus"]))
res["closed_iii_groups"] = len(res["closed_iii_groups"])
res["pairwise_violating_iii_share"] = {ds: pw[ds]["viol_iii"] / pw[ds]["n"] for ds in pw}
res["pairwise_breaking"] = {}
for (ds, v), lst in pw_break.items():
    a = np.array(lst)
    res["pairwise_breaking"][f"{ds}/{v}"] = {
        "n_groups": len(lst),
        "median_broken_frac": float(np.median(a[:, 0])),
        "median_p_broken": float(np.median(a[:, 1])),
        "mean_p_consensus": float(np.mean(a[:, 2])),
    }
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res, indent=1))
