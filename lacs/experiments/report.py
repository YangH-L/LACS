"""report: collect gates and experiment outputs into results/REPORT.md."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("lacs.report")


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    if df is None or df.empty:
        return "_(none)_\n"
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)

    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _load(p: Path):
    if not p.exists():
        return None
    if p.suffix == ".json":
        return json.loads(p.read_text())
    return pd.read_csv(p)


def verdict(flag) -> str:
    return "n/a" if flag is None else ("PASS" if flag else "FAIL")


def run(cfg: dict, paths: dict) -> str:
    R = paths["results"]
    meta = _load(paths["data"] / "meta.json")
    g1, g2, g3 = _load(R / "g1.json"), _load(R / "g2.json"), _load(R / "g3.json")
    e1 = _load(R / "e1_metrics.csv")
    h3 = _load(R / "h3_latitude_traits.csv")
    e2s, e2j = _load(R / "e2_summary.csv"), _load(R / "e2.json")
    e3s = _load(R / "e3_summary.csv")
    out = [f"# LACS report: {cfg['name']}\n"]
    if meta:
        out.append(f"Data: {meta['n_pairs']} labelled pairs, {meta['n_comments']} messages, {meta['n_users']} users, "
                   f"{meta['n_classes']} classes; split {meta['split_counts']}.\n")
    out.append("## Gates (go / no-go)\n")
    rows = []
    if g1:
        rows.append({"gate": "G1 ordinal structure", "result": verdict(g1.get("pass")),
                     "evidence": f"rho={g1['raw_distance']['spearman_rho']:.3f}, "
                                 f"CI={np.round(g1['raw_distance']['ci95'], 3).tolist()}; learned-distance rho="
                                 f"{g1['learned_distance_val']['spearman_rho']:.3f}"})
    if g2 and "full_vs_global" in g2:
        d = g2["full_vs_global"]
        rows.append({"gate": "G2 personalised > global", "result": verdict(g2.get("pass")),
                     "evidence": f"dNLL={d['mean_diff']:.4f}, CI={np.round(d['ci95'], 4).tolist()}"})
    if g3:
        best = {k: max(v.get("C_median", 0), v.get("R_median", 0)) for k, v in g3["by_k"].items()}
        rows.append({"gate": "G3 community size", "result": verdict(g3.get("pass")),
                     "evidence": f"median size by k: {best}; recommended k={g3.get('recommended_k')}"})
    out.append(md_table(pd.DataFrame(rows)) if rows else "_gates not run yet_\n")
    if g1 and g1.get("inverted_u"):
        iu = g1["inverted_u"]
        out.append(f"Inverted-U check (binary task): quadratic term {iu['b_quadratic']:.3f} "
                   f"CI {np.round(iu['b_quadratic_ci95'], 3).tolist()}, supported={iu['supported']}.\n")
    if e1 is not None:
        out.append("## E1 acceptance prediction (test split)\n")
        cols = [c for c in ("Unnamed: 0", "nll", "macro_f1", "acc", "auc_reject_vs_accept", "auc_accept", "ece",
                            "ordinal_mae") if c in e1.columns]
        t = e1[cols].rename(columns={"Unnamed: 0": "model"})
        out.append(md_table(t, "{:.4f}"))
        out.append("`reference_reply_text` sees the reply text and is only an upper reference.\n")
    if g2 and g2.get("cold_start"):
        cs = {k: v for k, v in g2["cold_start"].items() if k != "n"}
        out.append(f"Cold-start receivers (n={g2['cold_start']['n']}), test NLL: " +
                   ", ".join(f"{k}={v:.3f}" for k, v in sorted(cs.items(), key=lambda t: t[1])) + "\n")
    if h3 is not None and len(h3):
        out.append("## H3 latitude-trait associations (full model)\n")
        h = h3.reindex(h3["w_acc"].abs().sort_values(ascending=False).index).head(12)
        out.append(md_table(h))
    if e2s is not None:
        out.append("## E2 community effectiveness\n")
        if e2j:
            out.append(f"k={e2j['k']}, {e2j['n_queries']} queries, latitudes from `{e2j['variant']}`, "
                       f"global r for (k,r)-core = {e2j['r_global']:.3f}.\n")
        keep = [c for c in ("method", "variant", "nonempty_rate", "size_median", "coverage_labeled", "iar_micro",
                            "irr_micro", "iar_macro", "density_mean", "conductance_mean", "feasible_c_rate",
                            "hk_consensus_rate", "hk_ext_consensus_rate", "sjt_rejection_free_rate",
                            "time_ms_mean") if c in e2s.columns]
        out.append(md_table(e2s[keep]))
        if e2j and e2j.get("paired_iar"):
            out.append("Paired per-query internal-accept-rate differences (lcc_c_exact minus baseline):\n")
            out.append(md_table(pd.DataFrame([{"baseline": c["b"], "queries": c["n_queries"],
                                               "mean_diff": c["mean_diff"], "ci95": np.round(c["ci95"], 3).tolist()}
                                              for c in e2j["paired_iar"]])))
    if e3s is not None:
        out.append("## E3 efficiency (mean ms per query)\n")
        piv = e3s.pivot_table(index=["dataset", "sweep", "value"], columns="algo", values="time_ms_mean")
        out.append(md_table(piv.reset_index(), "{:.2f}"))
    figs = sorted(p.name for p in paths["figs"].glob("*.png"))
    if figs:
        out.append("## Figures\n" + "\n".join(f"- figures/{f}" for f in figs) + "\n")
    text = "\n".join(out)
    (R / "REPORT.md").write_text(text, encoding="utf-8")
    LOG.info("report written to %s", R / "REPORT.md")
    return text
