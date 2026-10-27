"""P2-4 (E11) analysis, plan docs/PLANS.md (lacs_p2_plan_20261026) section 4.3 (descriptive; written before any E11
result was seen, tested on the rehearsal outputs).

  python3 -I p24_analysis.py <root containing runs_p2/> <out dir>

Per dataset (gowalla, brightkite, yelp) and multiplier s:
  1. curve: share of queries with an LCC, share proven (status optimal), median LCC size when non-empty;
  2. s*: smallest s of the grid with a non-empty LCC (> 3 when none); its distribution over the queries
     without an LCC at s = 1; monotonicity check (an LCC at s but none at a larger s' is impossible unless an
     answer is unproven: proven violations must be 0);
  3. classes A/B/C/U (plan 1.2) of the LCC-empty queries at every s;
  4. quality: mean main metric (covisit_pair_rate) and secondary metric (member_meet_rate) of the LCC groups
     at every s; on the queries with an LCC at s = 1 (and at s), the paired difference s minus 1 of the main
     metric, 95% percentile bootstrap over queries (2,000 replicates, seed 20261027);
  5. conservative margin (s < 1): among the queries with an LCC at s = 1, share with an LCC at s and the median
     size ratio size(s) / size(1) (among those that keep one; and counting a lost LCC as 0).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
GRID = [0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0]
UNPROVEN = ("timeout", "too_large")
DATASETS = ["gowalla", "brightkite", "yelp"]


def label(s):
    return f"s{s:g}"


def classes(d):
    lcc0 = d["lcc_size"] == 0
    unp = (d["lcc_status"] != "optimal") | d["pairwise_status"].isin(UNPROVEN)
    U = lcc0 & unp
    rest = lcc0 & ~U
    A = rest & (d["edgeacc_size"] == 0)
    B = rest & (d["edgeacc_size"] > 0) & (d["pairwise_size"] == 0)
    C = rest & (d["pairwise_size"] > 0)
    n0 = int(lcc0.sum())
    out = {"lcc_empty": n0}
    for k, m in (("A", A), ("B", B), ("C", C), ("U", U)):
        out[k] = int(m.sum())
        out[k + "_share"] = int(m.sum()) / n0 if n0 else None
    return out


def boot_mean(x, seed=20261027, n=2000):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return {"n": int(len(x)), "mean": float(x.mean()) if len(x) else None, "ci95": None}
    rng = np.random.default_rng(seed)
    b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return {"n": int(len(x)), "mean": float(x.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}


res = {}
for ds in DATASETS:
    base = ROOT / "runs_p2" / "e11" / ds / "results" / "e11"
    if not base.exists():
        continue
    per = {}
    for s in GRID:
        f = base / label(s) / "e11_per_query.csv"
        if f.exists():
            per[s] = pd.read_csv(f).set_index("q").sort_index()
    if not per:
        continue
    qsets = {s: set(d.index) for s, d in per.items()}
    same = all(v == qsets[min(qsets)] for v in qsets.values())
    r = {"multipliers": sorted(per), "same_queries": same, "queries": {label(s): len(d) for s, d in per.items()},
         "curve": {}, "classes": {}, "quality": {}, "paired_vs_1": {}, "margin": {}}
    for s, d in sorted(per.items()):
        nz = d[d["lcc_size"] > 0]
        r["curve"][label(s)] = {"has_lcc": float((d["lcc_size"] > 0).mean()), "proven": float((d["lcc_status"] == "optimal").mean()),
                               "lcc_unproven_nonempty": int(((d["lcc_size"] > 0) & (d["lcc_status"] != "optimal")).sum()),
                               "median_size": float(nz["lcc_size"].median()) if len(nz) else None,
                               "edgeacc_nonempty": float((d["edgeacc_size"] > 0).mean()),
                               "pairwise_nonempty": float((d["pairwise_size"] > 0).mean())}
        r["classes"][label(s)] = classes(d)
        r["quality"][label(s)] = {"n": int(len(nz)), "covisit_pair_rate": float(nz["covisit_pair_rate"].mean()) if len(nz) else None,
                                 "member_meet_rate": float(nz["member_meet_rate"].mean()) if len(nz) else None}
    if 1.0 in per and same:
        one = per[1.0]
        has1 = one.index[one["lcc_size"] > 0]
        # s*: smallest multiplier with an LCC
        Q = sorted(qsets[1.0])
        star = pd.Series(np.inf, index=Q)
        for s in sorted(per, reverse=True):
            star[per[s].loc[Q, "lcc_size"].to_numpy() > 0] = s
        empty1 = one.index[one["lcc_size"] == 0]
        st = star.loc[empty1]
        dist = {label(s): int((st == s).sum()) for s in sorted(per) if s > 1.0}
        dist[">3"] = int(np.isinf(st).sum())
        cum = {label(s): float((st <= s).mean()) for s in sorted(per) if s > 1.0}
        r["s_star"] = {"queries_without_lcc_at_1": int(len(empty1)), "distribution": dist, "cumulative_share": cum}
        # monotonicity: LCC at s, none at a larger s'
        viol_proven = viol_unproven = 0
        ss = sorted(per)
        for i, s in enumerate(ss):
            for s2 in ss[i + 1:]:
                a, b = per[s].loc[Q], per[s2].loc[Q]
                bad = (a["lcc_size"] > 0) & (b["lcc_size"] == 0)
                viol_proven += int((bad & (b["lcc_status"] == "optimal")).sum())
                viol_unproven += int((bad & (b["lcc_status"] != "optimal")).sum())
        r["monotonicity"] = {"proven_violations": viol_proven, "unproven_cases": viol_unproven}
        for s, d in sorted(per.items()):
            if s == 1.0:
                continue
            both = [q for q in has1 if d.at[q, "lcc_size"] > 0]
            diff = d.loc[both, "covisit_pair_rate"].to_numpy(float) - one.loc[both, "covisit_pair_rate"].to_numpy(float)
            diff2 = d.loc[both, "member_meet_rate"].to_numpy(float) - one.loc[both, "member_meet_rate"].to_numpy(float)
            r["paired_vs_1"][label(s)] = {"main": boot_mean(diff), "secondary": boot_mean(diff2, seed=20261028)}
            if s < 1.0:
                keep = d.loc[has1, "lcc_size"] > 0
                ratio = d.loc[has1, "lcc_size"].to_numpy(float) / one.loc[has1, "lcc_size"].to_numpy(float)
                r["margin"][label(s)] = {"with_lcc_at_1": int(len(has1)), "keep_share": float(keep.mean()) if len(has1) else None,
                                         "median_size_ratio_kept": float(np.median(ratio[keep.to_numpy()])) if keep.any() else None,
                                         "median_size_ratio_all": float(np.median(ratio)) if len(has1) else None}
    res[ds] = r

(OUT / "p24_results.json").write_text(json.dumps(res, indent=1))
lines = []
for ds, r in res.items():
    lines += [f"## {ds}", "", f"queries per s: {r['queries']}; same query set at every s: {r['same_queries']}", "",
              "| s | has LCC | proven | median size | A | B | C | U | main metric | secondary |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in r["multipliers"]:
        c, k, q = r["curve"][label(s)], r["classes"][label(s)], r["quality"][label(s)]
        f = lambda x, p=1: "–" if x is None else f"{100 * x:.{p}f}%"
        lines.append(f"| {s:g} | {f(c['has_lcc'])} | {f(c['proven'])} | {c['median_size']} | {k['A']} | {k['B']} | {k['C']} | {k['U']} | "
                     f"{'–' if q['covisit_pair_rate'] is None else round(q['covisit_pair_rate'], 4)} | "
                     f"{'–' if q['member_meet_rate'] is None else round(q['member_meet_rate'], 4)} |")
    if "s_star" in r:
        lines += ["", f"s* among the {r['s_star']['queries_without_lcc_at_1']} queries without an LCC at s = 1: {r['s_star']['distribution']}",
                  f"cumulative share solved: {r['s_star']['cumulative_share']}", f"monotonicity: {r['monotonicity']}",
                  f"paired vs s = 1 (main): " + "; ".join(f"{k}: {v['main']}" for k, v in r["paired_vs_1"].items()),
                  f"margin (s < 1): {r['margin']}"]
    lines.append("")
(OUT / "p24_results.md").write_text("\n".join(lines))
print("\n".join(lines))
