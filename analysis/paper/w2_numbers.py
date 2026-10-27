"""Recompute the W2 efficiency numbers (engine v0.5) from the W2 bundle and the round-3 bundle.

Usage: python3 -I w2_numbers.py <w2_runs_dir> <round3_per_query_csv> <out_json>
Every number goes to the JSON, tables are written separately.
"""
import json
import sys

import numpy as np
import pandas as pd

w2, r3path, out = sys.argv[1], sys.argv[2], sys.argv[3]
A = pd.read_csv(f"{w2}/efficiency_w2/results/e3_per_query.csv")
B = pd.read_csv(f"{w2}/efficiency_w2_graphs/results/e3_per_query.csv")
R3 = pd.read_csv(r3path)
KEY = ["dataset", "sweep", "value", "q"]
res = {}


def med(x):
    x = np.asarray(x, dtype=float)
    return float(np.median(x)) if len(x) else None


def pct(x, p):
    x = np.asarray(x, dtype=float)
    return float(np.percentile(x, p)) if len(x) else None


# ---------- 1. v0.5 vs v0.4 on Run A (paired with round 3) ----------
ea = A[A.algo == "exact"].set_index(KEY)
e3 = R3[R3.algo == "exact"].set_index(KEY)
common = ea.index.intersection(e3.index)
pa, p3 = ea.loc[common], e3.loc[common]
to5, to4 = (pa.status == "timeout"), (p3.status == "timeout")
opt5 = pa.status.isin(["optimal", "empty"]) | ((pa.status == "ok"))
both = (~to5) & (~to4)
res["runA_paired"] = {
    "n_paired": int(len(common)),
    "timeouts_v04": int(to4.sum()),
    "timeouts_v05": int(to5.sum()),
    "newly_proven": int((to4 & ~to5).sum()),
    "v04_proven_v05_timeout": int((~to4 & to5).sum()),
    "both_proven": int(both.sum()),
    "both_proven_size_diff": int((pa["size"][both] != p3["size"][both]).sum()),
    "v05_smaller": int((pa["size"] < p3["size"]).sum()),
    "status_v05": pa.status.value_counts().to_dict(),
    "status_v04": p3.status.value_counts().to_dict(),
}
for ds in ["livejournal", "orkut", "gowalla", "brightkite"]:
    m = pa.index.get_level_values("dataset") == ds
    res["runA_paired"][f"total_speedup_{ds}"] = float(p3.time_ms[m].sum() / pa.time_ms[m].sum())
slow = p3.time_ms > 1000
sp = (p3.time_ms[slow] / pa.time_ms[slow])
res["runA_paired"]["v04_over_1s"] = int(slow.sum())
res["runA_paired"]["speedup_median_over_1s"] = float(sp.median())
res["runA_paired"]["speedup_max_over_1s"] = float(sp.max())
res["runA_paired"]["slower_than_2x_over_1s"] = int((sp < 0.5).sum())
# theta_scale = 2.0 proven rates
for ds in ["livejournal", "orkut"]:
    m = (pa.index.get_level_values("dataset") == ds) & (pa.index.get_level_values("sweep") == "theta_scale") & (pa.index.get_level_values("value") == 2.0)
    res["runA_paired"][f"proven_scale2_{ds}"] = [float((~to4[m]).mean()), float((~to5[m]).mean())]
# scale 3 (new): timeouts
s3 = A[(A.algo == "exact") & (A.sweep == "theta_scale") & (A.value == 3.0)]
res["runA_scale3_timeouts"] = int((s3.status == "timeout").sum())

# ---------- 2. Run B: exact vs exact_v04 on the same queries ----------
eb = B[(B.algo == "exact")].set_index(["dataset", "q"])
vb = B[(B.algo == "exact_v04")].set_index(["dataset", "q"])
cb = eb.index.intersection(vb.index)
res["runB"] = {}
for ds in ["dblp", "youtube", "pokec", "friendster"]:
    m = cb[cb.get_level_values("dataset") == ds]
    e, v = eb.loc[m], vb.loc[m]
    both_to = (e.status == "timeout") & (v.status == "timeout")
    bp = (e.status != "timeout") & (v.status != "timeout")
    rec = {
        "n": int(len(m)),
        "timeouts_v04": int((v.status == "timeout").sum()),
        "timeouts_v05": int((e.status == "timeout").sum()),
        "total_speedup": float(v.time_ms.sum() / e.time_ms.sum()),
        "both_proven_size_diff": int((e["size"][bp] != v["size"][bp]).sum()),
        "v05_smaller": int((e["size"] < v["size"]).sum()),
    }
    # incumbent of v0.4 when it times out but v0.5 proves: ratio to optimum
    tov4 = (v.status == "timeout") & (e.status != "timeout") & (e["size"] > 0)
    r = (v["size"][tov4] / e["size"][tov4])
    rec["v04_timeout_incumbent_over_opt_median"] = med(r)
    rec["v04_timeout_incumbent_over_opt_min"] = float(r.min()) if len(r) else None
    rec["v05_smaller_cases"] = [
        {"q": int(i[1]), "v05": int(e.loc[i, "size"]), "v04": int(v.loc[i, "size"]), "v05_status": e.loc[i, "status"], "v04_status": v.loc[i, "status"], "v05_ub": float(e.loc[i, "ub"]), "v04_ub": float(v.loc[i, "ub"])}
        for i in e.index[(e["size"] < v["size"]).values]
    ]
    res["runB"][ds] = rec

# ---------- 3. All 3,700 v0.5 Exact queries: timeouts and certificates ----------
allx = pd.concat([A[A.algo == "exact"], B[B.algo == "exact"]])
to = allx[allx.status == "timeout"]
res["all_exact"] = {
    "n": int(len(allx)),
    "timeouts": int(len(to)),
    "timeouts_runA": int((A[A.algo == "exact"].status == "timeout").sum()),
    "timeouts_runB": int((B[B.algo == "exact"].status == "timeout").sum()),
    "status": allx.status.value_counts().to_dict(),
    "timeout_rows": [
        {"dataset": r.dataset, "sweep": r.sweep, "value": float(r.value), "q": int(r.q), "size": int(r["size"]), "ub": float(r.ub), "ratio": float(r["size"] / r.ub) if r.ub > 0 else None}
        for _, r in to.iterrows()
    ],
}
rat = [x["ratio"] for x in res["all_exact"]["timeout_rows"] if x["ratio"] is not None]
res["all_exact"]["timeout_ratio_min"] = float(min(rat)) if rat else None
res["all_exact"]["timeout_ratio_max"] = float(max(rat)) if rat else None
# ok status check: what is 'ok' for exact?
res["all_exact"]["ok_status_count"] = int((allx.status == "ok").sum())


# ---------- 4. Default-setting strata per graph ----------
def default_rows(df, ds):
    d = df[df.dataset == ds]
    if ds in ("dblp", "youtube", "pokec", "friendster"):
        return d[(d.sweep == "k") & (d.value == 3)]
    m = ((d.sweep == "k") & (d.value == 3)) | ((d.sweep == "fraction") & (d.value == 1.0)) | ((d.sweep == "theta_scale") & (d.value == 1.0))
    if ds in ("livejournal", "orkut"):
        m = m | ((d.sweep == "dim") & (d.value == 16))
    return d[m]


res["default"] = {}
for ds, df in [("gowalla", A), ("brightkite", A), ("livejournal", A), ("orkut", A), ("dblp", B), ("youtube", B), ("pokec", B), ("friendster", B)]:
    d = default_rows(df, ds)
    piv = {}
    for algo in ["exact", "greedy", "ballcore_f", "ballcore_r", "kr_greedy", "kcore"]:
        x = d[d.algo == algo].set_index(["sweep", "value", "q"])
        piv[algo] = x
    ex = piv["exact"]
    n = len(ex)
    has = (ex["size"] > 0)
    for algo in ["greedy", "ballcore_f"]:
        has = has | (piv[algo]["size"].reindex(ex.index).fillna(0) > 0)
    no = ~has
    rec = {"n": int(n), "n_vertices": int(d.n.iloc[0]), "n_edges": int(d.m.iloc[0]), "no_lcc": int(no.sum()), "lcc": int(has.sum())}
    rec["exact_no_lcc_median_ms"] = med(ex.time_ms[no])
    rec["exact_no_lcc_max_ms"] = float(ex.time_ms[no].max()) if no.any() else None
    rec["no_lcc_zero_candidates"] = float((ex.n_candidates[no] == 0).mean()) if no.any() else None
    rec["no_lcc_region_zero"] = float((ex.s0_size[no] == 0).mean()) if no.any() else None
    rec["exact_lcc_median_ms"] = med(ex.time_ms[has])
    rec["exact_lcc_p90_ms"] = pct(ex.time_ms[has], 90)
    rec["exact_all_mean_ms"] = float(ex.time_ms.mean())
    rec["exact_all_median_ms"] = med(ex.time_ms)
    rec["exact_all_p90_ms"] = pct(ex.time_ms, 90)
    rec["exact_timeouts"] = int((ex.status == "timeout").sum())
    rec["exact_timeouts_lcc"] = int(((ex.status == "timeout") & has).sum())
    rec["exact_proven_rate"] = float((ex.status != "timeout").mean())
    rec["lcc_candidates_median"] = med(ex.n_candidates[has])
    g = piv["greedy"].reindex(ex.index)
    proven = has & (ex.status != "timeout")
    rec["greedy_found_rate_lcc"] = float((g["size"][has] > 0).mean()) if has.any() else None
    rr = (g["size"][proven] / ex["size"][proven])
    rec["greedy_ratio_mean_proven"] = float(rr.mean()) if proven.any() else None
    rec["greedy_ratio_median_proven"] = float(rr.median()) if proven.any() else None
    rr2 = rr[g["size"][proven] > 0]
    rec["greedy_ratio_mean_proven_found"] = float(rr2.mean()) if len(rr2) else None
    rec["greedy_lcc_median_ms"] = med(g.time_ms[has])
    rec["greedy_lcc_p90_ms"] = pct(g.time_ms[has], 90)
    f = piv["ballcore_f"].reindex(ex.index)
    rec["bcf_found_rate_lcc"] = float((f["size"][has] > 0).mean()) if has.any() else None
    rf = (f["size"][proven] / ex["size"][proven])
    rec["bcf_ratio_mean_proven"] = float(rf.mean()) if proven.any() else None
    rec["bcf_lcc_median_ms"] = med(f.time_ms[has])
    r_ = piv["ballcore_r"].reindex(ex.index)
    rec["bcr_lcc_median_ms"] = med(r_.time_ms[has])
    rec["bcr_lcc_p90_ms"] = pct(r_.time_ms[has], 90)
    rec["bcr_nonempty"] = int((r_["size"] > 0).sum())
    rec["bcr_feasible_rate_nonempty"] = float(r_.feasible[r_["size"] > 0].astype(str).str.lower().eq("true").mean()) if (r_["size"] > 0).any() else None
    rec["bcr_nonempty_when_no_lcc_proven"] = int(((r_["size"] > 0) & no & (ex.status != "timeout")).sum())
    rb = (r_["size"][proven & (ex["size"] > 0)] / ex["size"][proven & (ex["size"] > 0)])
    rec["bcr_ratio_median"] = float(rb.median()) if len(rb) else None
    kr = piv["kr_greedy"].reindex(ex.index)
    rec["kr_nonempty"] = int((kr["size"] > 0).sum())
    rec["kr_feasible_rate_nonempty"] = float(kr.feasible[kr["size"] > 0].astype(str).str.lower().eq("true").mean()) if (kr["size"] > 0).any() else None
    rec["kr_lcc_median_ms"] = med(kr.time_ms[has])
    if len(piv["kcore"]):
        kc = piv["kcore"].reindex(ex.index)
        rec["kcore_median_ms"] = med(kc.time_ms)
        rec["kcore_mean_ms"] = float(kc.time_ms.mean())
        rec["exact_slower_than_kcore"] = int((ex.time_ms > kc.time_ms).sum())
    else:
        rec["kcore_median_ms"] = None
    # index
    rec["index_ms_range"] = [float(ex.index_ms.min()), float(ex.index_ms.max())]
    rec["index_edges"] = int(ex.index_edges.iloc[0])
    rec["index_edge_frac"] = float(ex.index_edges.iloc[0] / d.m.iloc[0])
    # timeouts: size/ub
    t = ex[ex.status == "timeout"]
    rec["timeout_ratio"] = [float(a / b) for a, b in zip(t["size"], t.ub)]
    res["default"][ds] = rec

# ---------- 5. Friendster / Pokec non-empty stats (all 200 queries of the k=3 sample) ----------
for ds in ["friendster", "pokec", "livejournal", "orkut"]:
    df = B if ds in ("friendster", "pokec") else A
    d = default_rows(df, ds)
    ex = d[d.algo == "exact"]
    ne = ex[ex["size"] > 0]
    res.setdefault("scal", {})[ds] = {
        "mean_ms": float(ex.time_ms.mean()), "median_ms": med(ex.time_ms), "p90_ms": pct(ex.time_ms, 90),
        "proven": float((ex.status != "timeout").mean()),
        "nonempty_median_ms": med(ne.time_ms), "nonempty_p90_ms": pct(ne.time_ms, 90), "n_nonempty": int(len(ne)),
    }

# ---------- 6. Ablation (LJ, Orkut; 50 queries of the k=3 sample) ----------
ab = A[A.sweep == "ablation"]
k3 = A[(A.sweep == "k") & (A.value == 3) & (A.algo == "exact")]
res["ablation"] = {}
for ds in ["livejournal", "orkut", "gowalla", "brightkite"]:
    rec = {}
    full = k3[k3.dataset == ds]
    rec["exact"] = {"timeout_rate": float((full.status == "timeout").mean()), "median_ms": med(full.time_ms), "mean_ms": float(full.time_ms.mean()), "region_median": med(full.s0_size), "n": int(len(full))}
    for algo, g in ab[ab.dataset == ds].groupby("algo"):
        rec[algo] = {"timeout_rate": float((g.status == "timeout").mean()), "median_ms": med(g.time_ms), "mean_ms": float(g.time_ms.mean()), "region_median": med(g.s0_size), "n": int(len(g))}
    res["ablation"][ds] = rec

# ---------- 7. Baselines across all W2 queries at default ----------
json.dump(res, open(out, "w"), indent=1, default=float)
print("written", out)
