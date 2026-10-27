"""P2-1: why queries have no LCC (plan docs/PLANS.md (lacs_p2_plan_20261026) section 1).

Nesting: LCC non-empty => pairwise non-empty => edgeacc non-empty. For queries whose LCC is empty:
  A  edgeacc empty
  B  edgeacc non-empty, pairwise empty
  C  pairwise non-empty, LCC empty
  U  a solver did not prove its answer (lcc_exact or pairwise status timeout / too_large)
Descriptive only. Usage: python3 -I p21_unsolvable.py <data root> <out dir>
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
DS = {
    "gowalla": (ROOT / "r4res/runs/gowalla/results/r4/geo_e2_per_query.csv",
                ROOT / "r4res/runs/gowalla/results/r4/geo_e2_sweep_per_query.csv"),
    "brightkite": (ROOT / "r4res/runs/brightkite/results/r4/geo_e2_per_query.csv",
                   ROOT / "r4res/runs/brightkite/results/r4/geo_e2_sweep_per_query.csv"),
    "yelp": (ROOT / "e5b/runs/e5_main/results/geo_e2_per_query.csv",
             ROOT / "e5b/runs/e5_main/results/geo_e2_sweep_per_query.csv"),
    "congress": (ROOT / "e5b/runs/e6/results/geo_e2_per_query.csv",
                 ROOT / "e5b/runs/e6/results/geo_e2_sweep_per_query.csv"),
}
METH = {"lcc_exact": "lcc", "pairwise": "pw", "edgeacc": "ea"}
UNPROVEN = {"timeout", "too_large"}


def wide(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["method"].isin(METH)].copy()
    d["m"] = d["method"].map(METH)
    st = d.pivot(index="q", columns="m", values="status")
    sz = d.pivot(index="q", columns="m", values="size").astype(int)
    w = pd.concat({"status": st, "size": sz}, axis=1)
    w.columns = [f"{b}_{a}" for a, b in w.columns]
    assert w.notna().all().all(), "missing method rows"
    return w


def classify(w: pd.DataFrame) -> dict:
    lcc0 = w["lcc_size"] == 0
    unproven_lcc = ~w["lcc_status"].eq("optimal")
    unproven_pw = w["pw_status"].isin(UNPROVEN)
    pw_proven_empty = (w["pw_size"] == 0) & w["pw_status"].isin({"empty", "optimal"})
    # nesting checks (witness based: a non-empty answer is a feasible group whatever the status)
    viol = {
        "lcc_nonempty_pw_proven_empty": int(((w["lcc_size"] > 0) & pw_proven_empty).sum()),
        "pw_nonempty_ea_empty": int(((w["pw_size"] > 0) & (w["ea_size"] == 0)).sum()),
        "lcc_nonempty_ea_empty": int(((w["lcc_size"] > 0) & (w["ea_size"] == 0)).sum()),
    }
    viol_u = {"lcc_nonempty_pw_unproven_size0": int(((w["lcc_size"] > 0) & unproven_pw & (w["pw_size"] == 0)).sum())}
    U = lcc0 & (unproven_lcc | unproven_pw)
    rest = lcc0 & ~U
    A = rest & (w["ea_size"] == 0)
    B = rest & (w["ea_size"] > 0) & (w["pw_size"] == 0)
    C = rest & (w["pw_size"] > 0)
    assert int(A.sum() + B.sum() + C.sum() + U.sum()) == int(lcc0.sum())
    n0 = int(lcc0.sum())
    out = {"queries": int(len(w)), "lcc_empty": n0, "lcc_empty_share": n0 / len(w),
           "lcc_empty_unproven": int((lcc0 & unproven_lcc).sum()),
           "pw_unproven_among_lcc_empty": int((lcc0 & unproven_pw).sum()),
           "nesting_violations": viol, "nesting_in_U": viol_u}
    for name, m in (("A", A), ("B", B), ("C", C), ("U", U)):
        out[name] = int(m.sum())
        out[name + "_share_of_empty"] = int(m.sum()) / n0 if n0 else float("nan")
    # descriptive: sizes in B and C
    out["B_median_edgeacc_size"] = float(w.loc[B, "ea_size"].median()) if B.any() else None
    out["C_median_edgeacc_size"] = float(w.loc[C, "ea_size"].median()) if C.any() else None
    out["C_median_pairwise_size"] = float(w.loc[C, "pw_size"].median()) if C.any() else None
    # the shares the 10-26 note reported (pairwise also empty / edgeacc also empty among LCC-empty)
    out["pw_size0_share_of_empty"] = float((w.loc[lcc0, "pw_size"] == 0).mean()) if n0 else None
    out["ea_size0_share_of_empty"] = float((w.loc[lcc0, "ea_size"] == 0).mean()) if n0 else None
    out["_masks"] = {"A": A, "B": B, "C": C, "U": U}
    return out


res = {}
for ds, (fmain, fsweep) in DS.items():
    main = pd.read_csv(fmain, usecols=["q", "method", "status", "size"], low_memory=False)
    wm = wide(main)
    r = {"main": classify(wm)}
    sw = pd.read_csv(fsweep, usecols=["sweep", "k", "tolerance", "q", "method", "status", "size"], low_memory=False)
    sw = sw[sw["sweep"] == "tolerance"]
    tw = {t: wide(sw[sw["tolerance"] == t]) for t in ("theta_q50", "theta_q70", "theta_q90")}
    qs = [set(x.index) for x in tw.values()]
    assert qs[0] == qs[1] == qs[2], "tolerance sweep: query sets differ"
    base = tw["theta_q50"]
    e50 = base.index[base["lcc_size"] == 0]
    relax = {"queries": int(len(base)), "lcc_empty_q50": int(len(e50)),
             "lcc_empty_q50_unproven": int((base.loc[e50, "lcc_status"] != "optimal").sum())}
    for t in ("theta_q70", "theta_q90"):
        s = tw[t].loc[e50, "lcc_size"]
        relax[f"has_lcc_at_{t}"] = int((s > 0).sum())
        relax[f"has_lcc_share_at_{t}"] = float((s > 0).mean()) if len(s) else None
        relax[f"median_lcc_size_at_{t}_when_found"] = float(s[s > 0].median()) if (s > 0).any() else None
        relax[f"has_lcc_unproven_status_at_{t}"] = int(((s > 0) & (tw[t].loc[e50, "lcc_status"] != "optimal")).sum())
    for t in ("theta_q50", "theta_q90"):
        c = classify(tw[t])
        c.pop("_masks")
        relax[f"classes_{t}"] = c
    # cross-check of the sample against the main analysis at theta_q50 (same k = 3, same tolerance)
    common = base.index.intersection(wm.index)
    relax["q50_sample_in_main"] = int(len(common))
    relax["q50_sample_lcc_size_equal_main"] = int((base.loc[common, "lcc_size"] == wm.loc[common, "lcc_size"]).sum())
    r["relax"] = relax
    r["main"].pop("_masks")
    res[ds] = r

(OUT / "p21_results.json").write_text(json.dumps(res, indent=1, default=float))

# small table (plan section 1.4)
lines = ["| 数据 | 查询 | 无 LCC | A | B | C | U | θ_q90 下有解（θ_q50 无解者） |", "|---|---|---|---|---|---|---|---|"]
for ds, r in res.items():
    m, x = r["main"], r["relax"]
    f = lambda k: f"{m[k]} ({100 * m[k + '_share_of_empty']:.1f}%)"
    lines.append(f"| {ds} | {m['queries']} | {m['lcc_empty']} ({100 * m['lcc_empty_share']:.1f}%) | {f('A')} | {f('B')} | {f('C')} | {f('U')} | "
                 f"{x['has_lcc_at_theta_q90']}/{x['lcc_empty_q50']} ({100 * x['has_lcc_share_at_theta_q90']:.1f}%) |")
(OUT / "p21_table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
for ds, r in res.items():
    print(ds, "nesting:", r["main"]["nesting_violations"], r["main"]["nesting_in_U"],
          "| sample==main at q50:", r["relax"]["q50_sample_lcc_size_equal_main"], "/", r["relax"]["q50_sample_in_main"])
