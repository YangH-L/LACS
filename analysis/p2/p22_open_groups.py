"""P2-2: group-level outcome of open groups (plan docs/PLANS.md (lacs_p2_plan_20261026) section 2).

Fields (checked in e1x/open_sweep.py, p0kit/dynamics.py:aggregate, p0/cpp/p0_dyn.cpp):
  mean_largest_frac   share of members in the largest single-linkage cluster (radius 1e-3 x initial
                      diameter) at the last step; one run per group here, so it is that run's value
  mean_broken_frac    broken / npairs, npairs = ALL directed in-group pairs of G[H] (also those refused at
                      t = 0). The plan's quantity "share of the initially accepted interactions that were
                      lost" is broken / (npairs - t0_refused) = mean_broken_frac / (1 - mean_t0_refused_frac)
                      (exact for single-run rows); both are reported
  p_consensus         1 if one cluster at the last step (single run: 0/1 per group)
  median_first_break  step of the first broken interaction (-1 if none)
Usage: python3 -I p22_open_groups.py <data root> <out dir>
"""
import gzip, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
RUNS = {"gowalla": ROOT / "e5b/runs/e1_open/gowalla/e1x_rows.jsonl.gz",
        "brightkite": ROOT / "e5b/runs/e1_open/brightkite/e1x_rows.jsonl.gz",
        "congress": ROOT / "e5b/runs/e6/e1x/e6/e1x_rows.jsonl.gz"}
METHODS = ["lcc_exact", "pairwise", "sac_sized", "kr_tau", "random_bfs"]
VARS = {"hk_eq_G": "0 (closed)", "open_w0.01": "0.01", "open_w0.1": "0.1", "open_w1": "1"}
VARS_E6 = {**VARS, "open_obs_G": "observed, open"}
SUMMARY = {"gowalla": ROOT / "e5b/runs/e1_open/report/e1x_summary.csv",
           "brightkite": ROOT / "e5b/runs/e1_open/report/e1x_summary.csv",
           "congress": ROOT / "e5b/runs/e6/results/e1x_summary.csv"}


def load(p):
    with gzip.open(p, "rt") as fh:
        df = pd.DataFrame([json.loads(l) for l in fh if l.strip()])
    if "skipped" in df.columns:
        df = df[df["skipped"].isna()]
    return df


def boot_ci(x, seed, n=2000):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(lo), float(hi)


cells, paired, checks = [], [], {}
for ds, path in RUNS.items():
    df = load(path)
    assert (df["runs"] == 1).all() or not df[df["variant"].isin(VARS_E6)]["runs"].ne(1).any()
    summ = pd.read_csv(SUMMARY[ds])
    summ = summ[summ["dataset"] == ds] if "dataset" in summ.columns else summ
    vars_ = VARS_E6 if ds == "congress" else VARS
    for var, wlab in vars_.items():
        for m in METHODS:
            d = df[(df["method"] == m) & (df["variant"] == var)]
            if d.empty:
                continue
            assert (d["runs"] == 1).all()
            br = d["broken_any"].astype(bool)
            lost_acc = d["mean_broken_frac"] / (1.0 - d["mean_t0_refused_frac"])
            rec = {"dataset": ds, "method": m, "variant": var, "w": wlab, "groups": int(len(d)),
                   "share_broken": float(br.mean()),
                   "mean_largest_frac_all": float(d["mean_largest_frac"].mean()),
                   "mean_largest_frac_broken": float(d.loc[br, "mean_largest_frac"].mean()) if br.any() else None,
                   "mean_largest_frac_unbroken": float(d.loc[~br, "mean_largest_frac"].mean()) if (~br).any() else None,
                   "median_lost_share_of_accepted_broken": float(lost_acc[br].median()) if br.any() else None,
                   "median_broken_frac_field_broken": float(d.loc[br, "mean_broken_frac"].median()) if br.any() else None,
                   "share_consensus_broken": float(d.loc[br, "p_consensus"].mean()) if br.any() else None,
                   "share_consensus_all": float(d["p_consensus"].mean()),
                   "median_first_break_broken": float(d.loc[br, "median_first_break"].median()) if br.any() else None,
                   "median_size": float(d["size"].median())}
            s = summ[(summ["method"] == m) & (summ["variant"] == var)]
            if len(s):
                rec["share_broken_summary_csv"] = float(s["share_broken"].iloc[0])
                rec["mean_largest_summary_csv"] = float(s["mean_largest_cluster"].iloc[0])
            cells.append(rec)
        # exploratory paired comparison: LCC minus sac_sized, largest-cluster share, same query
        a = df[(df["method"] == "lcc_exact") & (df["variant"] == var)].set_index("query")["mean_largest_frac"]
        b = df[(df["method"] == "sac_sized") & (df["variant"] == var)].set_index("query")["mean_largest_frac"]
        common = a.index.intersection(b.index)
        dd = (a.loc[common] - b.loc[common]).to_numpy(float)
        lo, hi = boot_ci(dd, seed=20261026)
        paired.append({"dataset": ds, "variant": var, "w": wlab, "n": int(len(common)),
                       "lcc_mean": float(a.loc[common].mean()), "sac_sized_mean": float(b.loc[common].mean()),
                       "diff": float(dd.mean()), "ci95": [lo, hi]})

C = pd.DataFrame(cells)
P = pd.DataFrame(paired)
C.to_csv(OUT / "p22_cells.csv", index=False)
P.to_csv(OUT / "p22_paired.csv", index=False)
chk = C.dropna(subset=["share_broken_summary_csv"])
print("share_broken vs e1x_summary.csv: max abs diff",
      float((chk["share_broken"] - chk["share_broken_summary_csv"]).abs().max()),
      "| mean largest vs csv:", float((chk["mean_largest_frac_all"] - chk["mean_largest_summary_csv"]).abs().max()),
      "| cells checked:", len(chk))
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 200)
cols = ["dataset", "method", "w", "groups", "share_broken", "mean_largest_frac_all", "mean_largest_frac_broken",
        "median_lost_share_of_accepted_broken", "median_broken_frac_field_broken", "share_consensus_broken",
        "median_first_break_broken"]
print(C[cols].round(4).to_string(index=False))
print(P.round(4).to_string(index=False))
