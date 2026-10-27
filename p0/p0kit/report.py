"""P0 report: tables, figures and REPORT_P0.md from everything under runs_p0/.

  PYTHONPATH=p0 python -m p0kit.report --root runs_p0 --e3-run runs/efficiency_round3 --out runs_p0/report

Sections (each one is skipped when its inputs are missing):
  E1  share of groups in which some interaction breaks (accepted at t = 0, refused later),
      any refusal including t = 0, SJT rejections; per dataset x method x dynamics variant,
      with bootstrap CIs over queries; static breakability (1 - share satisfying (iii), which by
      the characterisation means *some* interaction pattern and weights break the group);
      size-binned table; paired differences against lcc_exact
  E2  per sigma: LCC existence, size, Exact time (queries with / without an LCC separately),
      timeouts; tolerance violations of the tolerance-agnostic baselines; sigma = 0 check
  E3  per solver: share proven within the limit, time quantiles, speed ratio to the paper's
      B&B, CDFs (queries with / without an LCC); cross-check of proven sizes
  E4  the E2 table for the intrinsic scale, mixing and query-sampling settings
  R3  round 3's Table 1 recomputed in the reporting format of E4 (CDFs, timeouts, both groups)
Only aggregates are written; no member ids.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

LCC_METHODS = ("lcc_exact", "lcc_greedy", "lcc_ballcore_f", "exact", "greedy", "ballcore_f")
CLEAN_VARIANTS = ("hk_eq_G", "hk_eq_K", "hk_rand_G", "hk_rand_K", "hk_rand_G_a0.1", "hk_rand_K_a0.1", "follow_G",
                  "follow_K", "dw_G", "dw_K", "sjt_eq_G", "sjt_eq_K", "hk_eq_G_long")
SIZE_BINS = [2, 6, 11, 21, 51, 201, 10 ** 9]
SIZE_LABELS = ["2-5", "6-10", "11-20", "21-50", "51-200", "201+"]


def _boot_ci(x: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)))


def _fmt(v, d=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    if isinstance(v, (float, np.floating)):
        return f"{v:.{d}f}"
    return str(v)


def _md_table(df: pd.DataFrame, cols: List[str] = None, digits: int = 3) -> str:
    if df is None or df.empty:
        return "(none)\n"
    cols = cols or list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, r in df[cols].iterrows():
        lines.append("| " + " | ".join(_fmt(r[c], digits) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# ===================================================================== E1 ==
def load_e1(root: Path) -> pd.DataFrame:
    from .formats import recover_jsonl

    frames = []
    for p in sorted(glob.glob(str(root / "*" / "e1" / "*" / "e1_rows.jsonl.gz"))):
        rows, info = recover_jsonl(p)  # tolerant: an E1 run may have been interrupted
        if info["truncated"]:
            print(f"[report] {p}: interrupted run, using its {len(rows)} complete rows")
        if rows:
            df = pd.DataFrame(rows)
            df["e1_dir"] = str(Path(p).parent)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_e1(df: pd.DataFrame, out: Path) -> Dict:
    res = {}
    if df.empty:
        return res
    # coverage: which groups each variant could run on (size caps, members not exported)
    cov = df.copy()
    cov["skipped"] = cov["skipped"].fillna("")
    whole = cov[cov["variant"] == "*"]  # groups skipped for every variant
    per = cov[cov["variant"] != "*"]
    crow = []
    variants = sorted(set(per["variant"]))
    for (ds, st, m), g in cov.groupby(["dataset", "setting", "method"]):
        w = whole[(whole["dataset"] == ds) & (whole["setting"] == st) & (whole["method"] == m)]
        for v in variants:
            h = g[g["variant"] == v]
            total = len(h) + len(w)
            crow.append({"dataset": ds, "setting": st, "method": m, "variant": v, "groups": total,
                         "run": int((h["skipped"] == "").sum()),
                         "skipped_size_gt_max_n": int((h["skipped"] == "size>max_n").sum()),
                         "skipped_whole_group": len(w),
                         "share_run": float((h["skipped"] == "").sum() / total) if total else np.nan})
    pd.DataFrame(crow).to_csv(out / "e1_coverage.csv", index=False)
    d = df[df["skipped"].fillna("") == ""].copy()
    d["breakable"] = d["p_broken"] > 0
    d["any_refusal"] = d["p_any_refusal"] > 0
    d["new_rejection"] = d["p_new_rejection"] > 0
    keys = ["dataset", "setting", "method", "variant"]
    rows = []
    for key, g in d.groupby(keys):
        lo, hi = _boot_ci(g["breakable"].astype(float).to_numpy())
        rows.append({**dict(zip(keys, key)), "groups": len(g), "median_size": float(g["size"].median()),
                     "share_breakable": float(g["breakable"].mean()), "ci_lo": lo, "ci_hi": hi,
                     "mean_p_broken": float(g["p_broken"].mean()),
                     "share_any_refusal": float(g["any_refusal"].mean()),
                     "share_t0_refusal": float((g["p_t0_refusal"] > 0).mean()),
                     "share_new_rejection": float(g["new_rejection"].mean()),
                     "share_any_rejection": float((g["p_any_rejection"] > 0).mean()),
                     "mean_largest_cluster": float(g["mean_largest_frac"].mean()),
                     "mean_p_consensus": float(g["p_consensus"].mean()),
                     "share_violating_iii": float(1 - g["sat_iii"].astype(float).mean()),
                     "share_violating_pairwise": float(1 - g["sat_pairwise"].astype(float).mean()),
                     "mean_open_acc_share": float(g["mean_open_acc_share"].mean())
                     if "mean_open_acc_share" in g else np.nan})
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "e1_summary.csv", index=False)
    res["summary_rows"] = len(summ)
    # sanity: LCC-feasible groups never break (HK, follow, DW: Theorem 1 and its convexity
    # argument) and never reject (SJT: Theorem 2) when the theorems' assumptions hold. Under
    # SJT, leaving the (narrower) acceptance latitude is not a violation; only rejection is.
    lcc = summ[summ["method"].isin(LCC_METHODS) & summ["variant"].isin(CLEAN_VARIANTS)]
    is_sjt = lcc["variant"].str.startswith("sjt")
    viol = lcc[(is_sjt & (lcc["share_any_rejection"] > 0)) |
               (~is_sjt & ((lcc["share_breakable"] > 0) | (lcc["share_t0_refusal"] > 0)))]
    res["theorem_violations"] = viol[keys].to_dict(orient="records")
    # stopping rule: hk_eq_G (T_max 300, early stop) against hk_eq_G_long (3000 steps, no early stop),
    # same groups; both deterministic. late = broken only in the long run.
    hz = []
    a = d[d["variant"] == "hk_eq_G"].set_index(["dataset", "setting", "query", "method"])["p_broken"]
    b = d[d["variant"] == "hk_eq_G_long"].set_index(["dataset", "setting", "query", "method"])["p_broken"]
    common = a.index.intersection(b.index)
    if len(common):
        aa, bb = a.loc[common] > 0, b.loc[common] > 0
        for (ds, st), idx in pd.Series(range(len(common)), index=common).groupby(level=[0, 1]):
            sel = idx.index
            hz.append({"dataset": ds, "setting": st, "groups": len(sel),
                       "broken_short": int(aa.loc[sel].sum()), "broken_long": int(bb.loc[sel].sum()),
                       "late_only": int((bb.loc[sel] & ~aa.loc[sel]).sum())})
    res["horizon_check"] = hz
    # size bins
    d["size_bin"] = pd.cut(d["size"], SIZE_BINS, right=False, labels=SIZE_LABELS)
    sb = (d.groupby(["dataset", "setting", "variant", "method", "size_bin"], observed=True)["breakable"]
          .agg(["mean", "size"]).reset_index().rename(columns={"mean": "share_breakable", "size": "groups"}))
    sb.to_csv(out / "e1_size_bins.csv", index=False)
    # paired against lcc_exact on the same query
    pr = []
    for (ds, st, var), g in d.groupby(["dataset", "setting", "variant"]):
        base = g[g["method"] == "lcc_exact"].set_index("query")["p_broken"]
        for m, h in g.groupby("method"):
            if m == "lcc_exact":
                continue
            o = h.set_index("query")["p_broken"]
            common = base.index.intersection(o.index)
            if len(common) < 5:
                continue
            diff = (o.loc[common] - base.loc[common]).to_numpy()
            lo, hi = _boot_ci(diff)
            pr.append({"dataset": ds, "setting": st, "variant": var, "method": m, "queries": len(common),
                       "mean_diff_p_broken": float(diff.mean()), "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(pr).to_csv(out / "e1_paired_vs_lcc.csv", index=False)
    _plot_e1(summ, out)
    return res


def _plot_e1(summ: pd.DataFrame, out: Path) -> None:
    try:
        plt = _plt()
    except Exception:
        return
    main_vars = [v for v in ("hk_eq_G", "hk_eq_K", "hk_rand_K", "hk_rand_K_a0.1", "follow_K", "dw_G", "sjt_eq_G")
                 if v in set(summ["variant"])]
    for (ds, st), g in summ.groupby(["dataset", "setting"]):
        methods = sorted(set(g["method"]), key=lambda m: (m not in LCC_METHODS, m))
        fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(methods)), 3.4))
        w = 0.8 / max(1, len(main_vars))
        for i, v in enumerate(main_vars):
            h = g[g["variant"] == v].set_index("method").reindex(methods)
            col = "share_any_rejection" if v.startswith("sjt") else "share_breakable"
            ax.bar(np.arange(len(methods)) + i * w, h[col].fillna(0), width=w, label=v)
        ax.set_xticks(np.arange(len(methods)) + 0.4 - w / 2)
        ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("share of groups that break\n(SJT: that reject)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, ncol=2, frameon=False)
        ax.set_title(f"E1 {ds} ({st})")
        fig.tight_layout()
        fig.savefig(out / f"e1_{ds}_{st}.png", dpi=160)
        plt.close(fig)
        # robustness: stubborn outsiders (lambda), tolerance noise (nu), open group
        rob = g[g["variant"].str.contains("_ext|_noise|_open", regex=True)]
        if rob.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
        for ax, tag, lab in ((axes[0], "_ext", "lambda (stubborn outsiders)"),
                             (axes[1], "_noise", "nu (tolerance noise)")):
            r = rob[rob["variant"].str.contains(tag)].copy()
            if r.empty:
                continue
            r["x"] = r["variant"].str.extract(r"(\d+\.?\d*)$")[0].astype(float)
            base = g[g["variant"] == "hk_eq_G"].assign(x=0.0)
            r = pd.concat([base, r])
            col = "share_breakable" if tag == "_ext" else "share_any_refusal"
            for m, h in r.groupby("method"):
                h = h.sort_values("x")
                ax.plot(h["x"], h[col], marker="o", label=m)
            ax.set_xlabel(lab)
            ax.set_ylabel(col.replace("_", " "))
            ax.set_ylim(0, 1)
        pairs = [(c, o) for c, o in (("hk_eq_G", "hk_open_G"), ("hk_eq_K", "hk_open_K")) if o in set(g["variant"])]
        if pairs:
            methods = sorted(set(g["method"]), key=lambda m: (m not in LCC_METHODS, m))
            w = 0.8 / (2 * len(pairs))
            for i, (c, o) in enumerate(pairs):
                for j, v in enumerate((c, o)):
                    h = g[g["variant"] == v].set_index("method").reindex(methods)
                    axes[2].bar(np.arange(len(methods)) + (2 * i + j) * w, h["share_breakable"].fillna(0), width=w,
                                label=v, alpha=1.0 if j else 0.45)
            axes[2].set_xticks(np.arange(len(methods)) + 0.4 - w / 2)
            axes[2].set_xticklabels(methods, rotation=35, ha="right", fontsize=7)
            axes[2].set_ylabel("share breakable")
            axes[2].set_ylim(0, 1)
            axes[2].legend(fontsize=6, frameon=False)
            axes[2].set_title("closed vs open group", fontsize=9)
        axes[1].legend(fontsize=6, frameon=False, ncol=2)
        fig.suptitle(f"E1 robustness {ds} ({st})")
        fig.tight_layout()
        fig.savefig(out / f"e1_robust_{ds}_{st}.png", dpi=160)
        plt.close(fig)


# ================================================================= E2/E4 ==
def load_sweeps(root: Path, experiment: str) -> pd.DataFrame:
    frames = []
    for p in sorted(glob.glob(str(root / "*" / experiment / "*" / "per_query.csv"))):
        df = pd.read_csv(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_sweep(df: pd.DataFrame, out: Path, experiment: str, xcol: str) -> Dict:
    if df.empty:
        return {}
    df = df.copy()
    ex = df[df["algo"] == "exact"].set_index(["dataset", "setting", "q"])["size"]
    df["lcc_exists"] = [bool(ex.get((a, b, c), 0) >= 2) for a, b, c in zip(df["dataset"], df["setting"], df["q"])]
    rows = []
    for (ds, st, algo), g in df.groupby(["dataset", "setting", "algo"]):
        nz = g[g["size"] >= 2]
        t_yes = g[g["lcc_exists"]]["time_ms"]
        t_no = g[~g["lcc_exists"]]["time_ms"]
        row = {"dataset": ds, "setting": st, "algo": algo, xcol: g[xcol].iloc[0] if xcol in g else None,
               "queries": len(g), "nonempty": float((g["size"] >= 2).mean()),
               "median_size_nonempty": float(nz["size"].median()) if len(nz) else np.nan,
               "time_median_ms": float(g["time_ms"].median()), "time_p90_ms": float(g["time_ms"].quantile(0.9)),
               "time_median_ms_lcc": float(t_yes.median()) if len(t_yes) else np.nan,
               "time_p90_ms_lcc": float(t_yes.quantile(0.9)) if len(t_yes) else np.nan,
               "time_median_ms_nolcc": float(t_no.median()) if len(t_no) else np.nan,
               "timeout_rate": float((g["status"] == "timeout").mean()),
               "too_large_rate": float((g["status"] == "too_large").mean()),
               "share_iii_nonempty": float(nz["feasible"].astype(float).mean()) if len(nz) else np.nan}
        if "sat_pairwise" in g and len(nz):
            row["share_pairwise_nonempty"] = float(nz["sat_pairwise"].astype(float).mean())
            row["mean_member_viol"] = float(nz["member_viol_frac"].astype(float).mean())
        rows.append(row)
    summ = pd.DataFrame(rows).sort_values(["dataset", xcol if xcol in rows[0] else "setting", "algo"])
    summ.to_csv(out / f"{experiment}_summary.csv", index=False)
    res = {"settings": int(summ["setting"].nunique())}
    # size ratios against the LCC on queries where both are non-empty
    rat = []
    for (ds, st), g in df.groupby(["dataset", "setting"]):
        lcc = g[g["algo"] == "exact"].set_index("q")["size"]
        for a, h in g.groupby("algo"):
            if a == "exact":
                continue
            o = h.set_index("q")["size"]
            c = lcc.index.intersection(o.index)
            c = [q for q in c if lcc[q] >= 2 and o[q] >= 2]
            if len(c) >= 5:
                r = (o.loc[c] / lcc.loc[c]).to_numpy(float)
                rat.append({"dataset": ds, "setting": st, "algo": a, "queries": len(c),
                            "median_size_ratio_to_lcc": float(np.median(r)), "mean_size_ratio_to_lcc": float(r.mean())})
    pd.DataFrame(rat).to_csv(out / f"{experiment}_size_ratio_vs_lcc.csv", index=False)
    # sigma = 0: the LCC is the maximum (k, r)-core containing q
    if experiment == "e2":
        z = df[df["setting"] == "sigma0.0"]
        if not z.empty:
            e = z[z["algo"] == "exact"].set_index(["dataset", "q"])
            t = z[z["algo"] == "kr_tau"].set_index(["dataset", "q"])
            c = e.index.intersection(t.index)
            both = [(i) for i in c if e.loc[i, "status"] == "optimal" and t.loc[i, "status"] == "optimal"]
            res["sigma0_checked"] = len(both)
            res["sigma0_mismatch"] = int(sum(int(e.loc[i, "size"]) != int(t.loc[i, "size"]) for i in both))
    _plot_sweep(summ, out, experiment, xcol)
    return res


def _plot_sweep(summ: pd.DataFrame, out: Path, experiment: str, xcol: str) -> None:
    try:
        plt = _plt()
    except Exception:
        return
    if xcol not in summ:
        return
    for ds, g in summ.groupby("dataset"):
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
        lcc = g[g["algo"] == "exact"].sort_values(xcol)
        axes[0].plot(lcc[xcol], lcc["nonempty"], marker="o")
        axes[0].set_ylabel("share of queries with an LCC")
        axes[1].plot(lcc[xcol], lcc["median_size_nonempty"], marker="o", label="LCC")
        for a in ("pairwise", "kr_tau"):
            h = g[g["algo"] == a].sort_values(xcol)
            if len(h):
                axes[1].plot(h[xcol], h["median_size_nonempty"], marker=".", label=a)
        axes[1].set_ylabel("median size (non-empty)")
        axes[1].legend(fontsize=7, frameon=False)
        axes[2].plot(lcc[xcol], lcc["time_median_ms_lcc"], marker="o", label="median, LCC exists")
        axes[2].plot(lcc[xcol], lcc["time_p90_ms_lcc"], marker="o", label="p90, LCC exists")
        axes[2].plot(lcc[xcol], lcc["time_median_ms_nolcc"], marker="x", label="median, no LCC")
        axes[2].set_yscale("log")
        axes[2].set_ylabel("Exact time (ms)")
        axes[2].legend(fontsize=7, frameon=False)
        for a in ("kr_greedy", "kr_tau", "edgeacc", "ballcore_r"):
            h = g[g["algo"] == a].sort_values(xcol)
            if len(h):
                axes[3].plot(h[xcol], 1 - h["share_iii_nonempty"], marker=".", label=a)
        axes[3].set_ylabel("share violating (iii)")
        axes[3].set_ylim(0, 1)
        axes[3].legend(fontsize=7, frameon=False)
        for ax in axes:
            ax.set_xlabel(xcol)
        fig.suptitle(f"{experiment} {ds}")
        fig.tight_layout()
        fig.savefig(out / f"{experiment}_{ds}.png", dpi=160)
        plt.close(fig)


# ===================================================================== E3 ==
def load_e3(root: Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(glob.glob(str(root / "*" / "e3" / "e3_per_query.csv")))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_e3(df: pd.DataFrame, out: Path) -> Dict:
    if df.empty:
        return {}
    df = df.copy()
    df["proven"] = df["proven"].astype(bool)
    # optimum per query: any proven size (they agree, see cross-check)
    opt = df[df["proven"]].groupby(["dataset", "query"])["size"].max()
    df["lcc_exists"] = [bool(opt.get((a, b), 0) >= 2) for a, b in zip(df["dataset"], df["query"])]
    rows = []
    for (ds, s), g in df.groupby(["dataset", "solver"]):
        yes, no = g[g["lcc_exists"]], g[~g["lcc_exists"]]
        rows.append({"dataset": ds, "solver": s, "queries": len(g), "proven": float(g["proven"].mean()),
                     "proven_lcc": float(yes["proven"].mean()) if len(yes) else np.nan,
                     "median_ms": float(g["time_ms"].median()), "p90_ms": float(g["time_ms"].quantile(0.9)),
                     "median_ms_lcc": float(yes["time_ms"].median()) if len(yes) else np.nan,
                     "p90_ms_lcc": float(yes["time_ms"].quantile(0.9)) if len(yes) else np.nan,
                     "median_ms_nolcc": float(no["time_ms"].median()) if len(no) else np.nan,
                     "timeout_rate": float((g["status"] == "timeout").mean()),
                     "cap_rate": float(g["status"].isin(["candidate_cap", "too_large"]).mean()),
                     "error_rate": float((g["status"] == "error").mean())})
    summ = pd.DataFrame(rows)
    # per-query speed ratio against the harness B&B (same sweep), on queries both proved
    ratios = []
    for ds, g in df.groupby("dataset"):
        b = g[(g["solver"] == "bnb") & g["proven"]].set_index("query")["time_ms"]
        for s, h in g.groupby("solver"):
            if s == "bnb":
                continue
            o = h[h["proven"]].set_index("query")["time_ms"]
            c = b.index.intersection(o.index)
            if len(c):
                r = (o.loc[c] / b.loc[c].clip(lower=1e-3)).to_numpy()
                ratios.append({"dataset": ds, "solver": s, "both_proven": len(c),
                               "median_time_ratio_to_bnb": float(np.median(r)),
                               "geo_mean_time_ratio_to_bnb": float(np.exp(np.mean(np.log(np.maximum(r, 1e-9)))))})
    summ = summ.merge(pd.DataFrame(ratios), on=["dataset", "solver"], how="left") if ratios else summ
    summ.to_csv(out / "e3_summary.csv", index=False)
    _plot_e3(df, out)
    mism = 0
    for _, g in df[df["proven"]].groupby(["dataset", "query"]):
        mism += int(g["size"].nunique() > 1)
    return {"solvers": sorted(set(df["solver"])), "crosscheck_mismatches": mism}


def _plot_e3(df: pd.DataFrame, out: Path) -> None:
    try:
        plt = _plt()
    except Exception:
        return
    for ds, g in df.groupby("dataset"):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
        for ax, sel, title in ((axes[0], g["lcc_exists"], "queries with an LCC"),
                               (axes[1], ~g["lcc_exists"], "queries without an LCC")):
            for s, h in g[sel].groupby("solver"):
                t = np.sort(h["time_ms"].to_numpy(float))
                if len(t) == 0:
                    continue
                y = np.arange(1, len(t) + 1) / len(t)
                ax.step(np.maximum(t, 1e-3), y, where="post", label=s)
            ax.set_xscale("log")
            ax.set_xlabel("query time (ms)")
            ax.set_title(title, fontsize=9)
        axes[0].set_ylabel("share of queries")
        axes[1].legend(fontsize=7, frameon=False)
        fig.suptitle(f"E3 {ds}")
        fig.tight_layout()
        fig.savefig(out / f"e3_cdf_{ds}.png", dpi=160)
        plt.close(fig)


# ============================================================ round 3 ==
def summarize_round3(e3_run: Path, out: Path) -> Dict:
    p = e3_run / "results" / "e3_per_query.csv"
    if not p.exists():
        return {}
    per = pd.read_csv(p)
    keep = (((per["sweep"] == "k") & (per["value"].astype(float) == 3)) |
            ((per["sweep"] == "fraction") & (per["value"].astype(float) == 1.0)) |
            ((per["sweep"] == "theta_scale") & (per["value"].astype(float) == 1.0)) |
            ((per["sweep"] == "dim") & (per["value"].astype(float) == 16)))
    d = per[keep].copy()
    ex = d[d["algo"] == "exact"].set_index(["dataset", "sweep", "q"])["size"]
    d["lcc_exists"] = [bool(ex.get((a, b, c), 0) >= 2) for a, b, c in zip(d["dataset"], d["sweep"], d["q"])]
    rows = []
    for (ds, a), g in d.groupby(["dataset", "algo"]):
        yes, no = g[g["lcc_exists"]], g[~g["lcc_exists"]]
        rows.append({"dataset": ds, "algo": a, "queries": len(g), "with_lcc": len(yes),
                     "median_ms_lcc": float(yes["time_ms"].median()) if len(yes) else np.nan,
                     "p90_ms_lcc": float(yes["time_ms"].quantile(0.9)) if len(yes) else np.nan,
                     "median_ms_nolcc": float(no["time_ms"].median()) if len(no) else np.nan,
                     "p90_ms_nolcc": float(no["time_ms"].quantile(0.9)) if len(no) else np.nan,
                     "timeout_rate_lcc": float((yes["status"] == "timeout").mean()) if len(yes) else np.nan,
                     "too_large_rate_lcc": float((yes["status"] == "too_large").mean()) if len(yes) else np.nan})
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "r3_table1_reformatted.csv", index=False)
    try:
        plt = _plt()
        for ds, g in d.groupby("dataset"):
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
            for ax, sel, title in ((axes[0], g["lcc_exists"], "queries with an LCC"),
                                   (axes[1], ~g["lcc_exists"], "queries without an LCC")):
                for a, h in g[sel].groupby("algo"):
                    t = np.sort(h["time_ms"].to_numpy(float))
                    if len(t):
                        ax.step(np.maximum(t, 1e-3), np.arange(1, len(t) + 1) / len(t), where="post", label=a)
                ax.set_xscale("log")
                ax.set_xlabel("query time (ms)")
                ax.set_title(title, fontsize=9)
            axes[0].set_ylabel("share of queries")
            axes[1].legend(fontsize=7, frameon=False)
            fig.suptitle(f"round 3, default setting: {ds}")
            fig.tight_layout()
            fig.savefig(out / f"r3_cdf_{ds}.png", dpi=160)
            plt.close(fig)
    except Exception:
        pass
    return {"datasets": sorted(set(d["dataset"]))}


# ================================================================ main ==
def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m p0kit.report")
    ap.add_argument("--root", default="runs_p0")
    ap.add_argument("--e3-run", default="runs/efficiency_round3")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    root = Path(args.root)
    out = Path(args.out or root / "report")
    out.mkdir(parents=True, exist_ok=True)
    numbers = {}
    md = ["# P0 report\n", f"Inputs: `{root}` and `{args.e3_run}`. Only aggregates; no member ids.\n"]

    e1 = load_e1(root)
    if not e1.empty:
        numbers["E1"] = summarize_e1(e1, out)
        s = pd.read_csv(out / "e1_summary.csv")
        md.append("\n## E1 interaction stress test\n")
        md.append("`share_breakable`: share of groups in which an interaction accepted at t = 0 is refused later "
                  "(any run). `share_any_refusal` also counts members who refuse each other already at t = 0. "
                  "SJT variants: `share_new_rejection`. `share_violating_iii`: groups that *some* interaction "
                  "pattern and weights can break (characterisation).\n")
        if numbers["E1"].get("theorem_violations"):
            md.append(f"\n**WARNING: LCC-feasible groups broke under assumption-free variants: "
                      f"{numbers['E1']['theorem_violations'][:10]}**\n")
        hz = pd.DataFrame(numbers["E1"].get("horizon_check", []))
        if not hz.empty:
            md.append("\nStopping rule (hk_eq_G with T_max 300 and early stop vs hk_eq_G_long with 3000 steps and "
                      "no early stop, same groups; `late_only` = groups that break only in the long run):\n")
            md.append(_md_table(hz, digits=0))
        cov = pd.read_csv(out / "e1_coverage.csv")
        if not cov.empty:
            md.append("\nCoverage (share of groups each variant ran on; the rest exceeded its max_n, or the whole "
                      "group was skipped: members not exported, size < 2 or > max_group):\n")
            cv = cov.groupby(["dataset", "setting", "variant"])[["groups", "run"]].sum().reset_index()
            cv["share_run"] = cv["run"] / cv["groups"].clip(lower=1)
            md.append(_md_table(cv.pivot_table(index="variant", columns="dataset", values="share_run").reset_index(),
                                digits=2))
        for (ds, st), g in s.groupby(["dataset", "setting"]):
            md.append(f"\n### {ds} ({st})\n")
            piv = g.pivot_table(index="method", columns="variant", values="share_breakable").reset_index()
            md.append(_md_table(piv, digits=2))
            md.append(f"\n![e1](e1_{ds}_{st}.png)\n")
            md.append(f"\n![e1 robustness](e1_robust_{ds}_{st}.png)\n")
    for exp, xcol in (("e2", "sigma"), ("e4_scale", "c"), ("e4_mix", "mu"), ("e4_queries", "query_mode")):
        sw = load_sweeps(root, exp)
        if sw.empty:
            continue
        numbers[exp] = summarize_sweep(sw, out, exp, xcol)
        s = pd.read_csv(out / f"{exp}_summary.csv")
        md.append(f"\n## {exp}\n")
        cols = [c for c in ["dataset", "setting", "algo", "queries", "nonempty", "median_size_nonempty",
                            "time_median_ms_lcc", "time_p90_ms_lcc", "time_median_ms_nolcc", "timeout_rate",
                            "share_iii_nonempty", "share_pairwise_nonempty", "mean_member_viol"] if c in s]
        md.append(_md_table(s[s["algo"].isin(["exact", "pairwise", "kr_tau", "kr_greedy", "edgeacc"])], cols, 3))
        for ds in sorted(set(s["dataset"])):
            md.append(f"\n![{exp}]({exp}_{ds}.png)\n")
    e3 = load_e3(root)
    if not e3.empty:
        numbers["E3"] = summarize_e3(e3, out)
        md.append("\n## E3 exact baselines\n")
        md.append(_md_table(pd.read_csv(out / "e3_summary.csv"), digits=2))
        errs = pd.read_csv(out / "e3_summary.csv")
        errs = errs[errs["error_rate"] > 0] if "error_rate" in errs else errs.iloc[0:0]
        if len(errs):
            md.append(f"\n**WARNING: solver errors (see the `message` column of e3_per_query.csv): "
                      f"{errs[['dataset', 'solver', 'error_rate']].to_dict(orient='records')}**\n")
        if numbers["E3"].get("crosscheck_mismatches"):
            md.append(f"\n**WARNING: {numbers['E3']['crosscheck_mismatches']} queries with disagreeing proven sizes**\n")
        for ds in sorted(set(e3["dataset"])):
            md.append(f"\n![e3](e3_cdf_{ds}.png)\n")
    r3 = summarize_round3(Path(args.e3_run), out)
    if r3:
        numbers["R3"] = r3
        md.append("\n## Round 3, default setting, reformatted (queries with / without an LCC)\n")
        md.append(_md_table(pd.read_csv(out / "r3_table1_reformatted.csv"), digits=2))
    (out / "REPORT_P0.md").write_text("\n".join(md))
    with open(out / "p0_numbers.json", "w") as f:
        json.dump(numbers, f, indent=1, default=str)
    print(f"[report] {out / 'REPORT_P0.md'}")


if __name__ == "__main__":
    main()
