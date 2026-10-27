"""E5 main experiment, last step: the pre-registered analyses on geo-e2's per-query results.

  python e5/analysis.py --results runs/e5_main/results [--out runs/e5_main/results/e5_analysis]

Plan: docs/PLANS.md (lacs_e5_prereg_draft_20261025), sections 3 and 5 (also used for E6 Q4, plan section 8).
Reads geo_e2_per_query.csv / geo_e2.json (and geo_e2_sweep_per_query.csv when present) and writes
e5_analysis.json and E5_ANALYSIS.md. Numbers only: the interpretation rules (section 6) are applied
by the reader once they are final; the tables flag whether each 95% CI lies above 0, below 0 or
contains 0, nothing more.

H1 (lcc_exact minus a baseline, queries where both groups have >= 2 members):
  raw         mean paired difference, bootstrap over queries (as geo-e2 / round 4)
  size (a)    queries where both groups have the same size but are different groups by the round-4
              criterion of the plan (size, diameter, active members, both metrics and co-review pairs
              all equal = same group); (a_hash) the same with exact member-set identity, a diagnostic
  size (b)    OLS of the paired difference on log(size_baseline / size_lcc) over all common queries;
              intercept = difference at equal size, bootstrap over queries
  bins        LCC size 4-10, 11-20, 21-50, 51-300, >300 (descriptive)
H3 (abstention, descriptive): baseline groups on queries with an empty vs non-empty LCC: raw
  difference, OLS coefficient of 'LCC non-empty' controlling log(group size), and by size bins.
H2 is computed inside geo-e2 (member_level) and copied here.
Metrics: covisit_pair_rate (primary), member_meet_rate (secondary) and, as robustness metrics that
are reported but not tested, every *__<name> column (E5: __month, __notop).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BINS = [(4, 10), (11, 20), (21, 50), (51, 300), (301, None)]
H1_PRIMARY = ["sac_sized", "kr_tau"]
H1_B_ONLY = ["pairwise"]
H3_METHODS = ["sac", "kr_global"]
ROUND4_ID = ["size", "diam_km", "active_members", "covisit_pair_rate", "member_meet_rate", "n_covisit_pairs"]
N_BOOT = 2000


def _ci_flag(ci) -> str:
    if ci is None or any(c is None or not np.isfinite(c) for c in ci):
        return "n/a"
    return "above 0" if ci[0] > 0 else ("below 0" if ci[1] < 0 else "contains 0")


def _weights(n: int, reps: int, rng, chunk: int = 200):
    """Bootstrap resampling as multinomial count weights, in chunks (reps x n)."""
    done = 0
    while done < reps:
        b = min(chunk, reps - done)
        yield rng.multinomial(n, np.full(n, 1.0 / n), size=b).astype(np.float64)
        done += b


def boot_mean(d: np.ndarray, seed: int, reps: int = N_BOOT) -> dict:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return {"n": int(n)}
    rng = np.random.default_rng(seed)
    bs = np.concatenate([(W @ d) / n for W in _weights(n, reps, rng)])
    ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return {"n": int(n), "mean_diff": float(d.mean()), "ci95": ci, "ci": _ci_flag(ci)}


def boot_ols(y: np.ndarray, X: np.ndarray, seed: int, reps: int = N_BOOT, names=None) -> dict:
    """OLS of y on [1, X] with a bootstrap over rows (rows = queries)."""
    y = np.asarray(y, float)
    X = np.asarray(X, float).reshape(len(y), -1)
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    y, X = y[ok], X[ok]
    n = len(y)
    Z = np.column_stack([np.ones(n), X])
    p = Z.shape[1]
    names = ["intercept"] + list(names or [f"x{i}" for i in range(1, p)])
    if n < 10 or np.linalg.matrix_rank(Z) < p:
        return {"n": int(n)}
    est = np.linalg.lstsq(Z, y, rcond=None)[0]
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(p)
    prod = np.column_stack([Z[:, i] * Z[:, j] for i, j in zip(*iu)])
    zy = Z * y[:, None]
    bs = []
    for W in _weights(n, reps, rng):
        G = np.zeros((W.shape[0], p, p))
        G[:, iu[0], iu[1]] = W @ prod
        G[:, iu[1], iu[0]] = G[:, iu[0], iu[1]]
        rhs = W @ zy
        for g_, r_ in zip(G, rhs):
            try:
                bs.append(np.linalg.solve(g_, r_))
            except np.linalg.LinAlgError:
                pass
    bs = np.array(bs)
    out = {"n": int(n)}
    for i, nm in enumerate(names):
        ci = [float(np.percentile(bs[:, i], 2.5)), float(np.percentile(bs[:, i], 97.5))] if len(bs) else None
        out[nm] = {"coef": float(est[i]), "ci95": ci, "ci": _ci_flag(ci)}
    return out


def _bin_of(size):
    for lo, hi in BINS:
        if size >= lo and (hi is None or size <= hi):
            return f"{lo}-{hi}" if hi else f">{lo - 1}"
    return f"<{BINS[0][0]}"


BIN_LABELS = [f"<{BINS[0][0]}"] + [_bin_of(lo) for lo, _ in BINS]


def metrics_of(df: pd.DataFrame) -> tuple[list, list]:
    main = [m for m in ("covisit_pair_rate", "member_meet_rate") if m in df.columns]
    rob = sorted(c for c in df.columns if "__" in c and c.split("__")[0] in ("covisit_pair_rate", "member_meet_rate"))
    return main, rob


def h1(df: pd.DataFrame, metrics: list, baselines: list, seed: int) -> dict:
    nz = df[df["size"] >= 2]
    lcc = nz[nz["method"] == "lcc_exact"].set_index("q")
    out = {}
    for m in baselines:
        oth = nz[nz["method"] == m].set_index("q")
        common = lcc.index.intersection(oth.index)
        if len(common) == 0:
            continue
        L, O = lcc.loc[common], oth.loc[common]
        sl, so = L["size"].to_numpy(float), O["size"].to_numpy(float)
        same = sl == so
        if "members_hash" in L.columns:
            ident = same & (L["members_hash"].astype(str).to_numpy() == O["members_hash"].astype(str).to_numpy())
        else:
            ident = np.zeros(len(common), bool)
        r4 = np.ones(len(common), bool)
        for c in ROUND4_ID:
            if c in L.columns:
                a, b = L[c].to_numpy(float), O[c].to_numpy(float)
                r4 &= (a == b) | (np.isnan(a) & np.isnan(b))
        res = {"n_common": int(len(common)), "n_same_size": int(same.sum()), "n_identical_members": int(ident.sum()),
               "n_identical_round4_criterion": int(r4.sum()), "n_same_size_different_members": int((same & ~r4).sum()),
               "n_same_size_different_members_hash": int((same & ~ident).sum()),
               "size_lcc_median": float(np.median(sl)), "size_baseline_median": float(np.median(so)),
               "metrics": {}}
        lr = np.log(so / sl)
        bins = np.array([_bin_of(s) for s in sl])
        for j, met in enumerate(metrics):
            d = L[met].to_numpy(float) - O[met].to_numpy(float)
            r = {"mean_lcc": float(np.nanmean(L[met])), "mean_baseline": float(np.nanmean(O[met])),
                 "raw": boot_mean(d, seed + 11 * j),
                 "size_a_same_size_different_members": boot_mean(d[same & ~r4], seed + 11 * j + 1),
                 "size_a_hash_diagnostic": boot_mean(d[same & ~ident], seed + 11 * j + 4),
                 "size_b_regression": boot_ols(d, lr, seed + 11 * j + 2, names=["log_size_ratio"]),
                 "by_lcc_size": {}}
            for b in BIN_LABELS:
                s_ = bins == b
                if s_.sum():
                    r["by_lcc_size"][b] = {"n": int(s_.sum()), "mean_lcc": float(np.nanmean(L[met].to_numpy(float)[s_])),
                                           "mean_baseline": float(np.nanmean(O[met].to_numpy(float)[s_])),
                                           "diff": boot_mean(d[s_], seed + 11 * j + 3)}
            res["metrics"][met] = r
        out[m] = res
    return out


UNKNOWN_STATUS = {"timeout", "too_large", "greedy_skipped", "error", "no_lcc"}  # no LCC found, emptiness not proven


def _h3_one(y, size, h, bins, seed):
    raw = {"mean_lcc_nonempty": float(np.nanmean(y[h])) if h.any() else None,
           "mean_lcc_empty": float(np.nanmean(y[~h])) if (~h).any() else None}
    if h.sum() >= 5 and (~h).sum() >= 5:
        rng = np.random.default_rng(seed)
        a, b = y[h][np.isfinite(y[h])], y[~h][np.isfinite(y[~h])]
        bs = np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(N_BOOT)])
        ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        raw.update({"diff": float(a.mean() - b.mean()), "ci95": ci, "ci": _ci_flag(ci)})
    r = {"raw": raw, "ols_controlling_log_size": boot_ols(y, np.column_stack([h.astype(float), np.log(size)]),
                                                          seed + 1, names=["lcc_nonempty", "log_size"]),
         "by_group_size": {}}
    for b in BIN_LABELS:
        s_ = bins == b
        if s_.sum():
            r["by_group_size"][b] = {"n_nonempty": int((s_ & h).sum()), "n_empty": int((s_ & ~h).sum()),
                                     "mean_lcc_nonempty": float(np.nanmean(y[s_ & h])) if (s_ & h).any() else None,
                                     "mean_lcc_empty": float(np.nanmean(y[s_ & ~h])) if (s_ & ~h).any() else None}
    return r


def h3(df: pd.DataFrame, metrics: list, methods: list, seed: int) -> dict:
    """'empty' = the LCC row has < 2 members (round-4 convention, includes queries where the exact search
    stopped without an answer); 'proven_empty_only' drops those unknown queries from the empty side."""
    L = df[df["method"] == "lcc_exact"].set_index("q")
    has = L["size"] >= 2
    unknown = (~has) & L["status"].astype(str).isin(UNKNOWN_STATUS)
    out = {}
    for m in methods:
        d = df[(df["method"] == m) & (df["size"] >= 2)].set_index("q")
        if d.empty:
            continue
        h = has.reindex(d.index).fillna(False).to_numpy(bool)
        unk = unknown.reindex(d.index).fillna(False).to_numpy(bool)
        size = d["size"].to_numpy(float)
        bins = np.array([_bin_of(s) for s in size])
        keep = ~unk
        res = {"n_lcc_nonempty": int(h.sum()), "n_lcc_empty": int((~h).sum()),
               "n_lcc_empty_unproven": int(unk.sum()), "metrics": {}, "metrics_proven_empty_only": {}}
        for j, met in enumerate(metrics):
            y = d[met].to_numpy(float)
            res["metrics"][met] = _h3_one(y, size, h, bins, seed + 7 * j)
            res["metrics_proven_empty_only"][met] = _h3_one(y[keep], size[keep], h[keep], bins[keep], seed + 7 * j + 3)
        out[m] = res
    return out


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _ci(r, flag: bool = True):
    if not r or "ci95" not in r:
        return f"n={r.get('n', 0) if r else 0}"
    v = r.get("mean_diff", r.get("coef", r.get("diff")))
    out = f"{_fmt(v)} [{_fmt(r['ci95'][0])}, {_fmt(r['ci95'][1])}]"
    return out + (f" ({r['ci']})" if flag else "")


def write_md(A: dict, path: Path) -> None:
    L = [f"# {A['title']}", "", f"Source: `{A['source']}`. Queries: {A['queries']}; k = {A.get('k')}; "
         f"tolerance {A.get('tolerance')}.", "",
         "Interpretation rules are not applied here (E5 plan section 6). Rows marked 'robustness' are reported "
         "only (plan section 3): their intervals carry no above/below flag.", ""]
    L += ["## H1: lcc_exact minus baseline", ""]
    for m, r in A["H1"].items():
        b_only = m in H1_B_ONLY
        L += [f"### vs {m} ({'primary' if m in H1_PRIMARY else ('size (b) only' if b_only else 'descriptive')})",
              "", f"common queries {r['n_common']}; same size {r['n_same_size']}, of which the same group by the "
              f"round-4 criterion {r['n_identical_round4_criterion']} (exact member identity {r['n_identical_members']}); "
              f"same size, different groups {r['n_same_size_different_members']} (by member hash "
              f"{r['n_same_size_different_members_hash']}); median size LCC {r['size_lcc_median']:.0f}, baseline "
              f"{r['size_baseline_median']:.0f}", "",
              "| metric | mean LCC | mean baseline | raw | (a) same size, different groups | (b) intercept | (b) slope |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for met, x in r["metrics"].items():
            rob = "__" in met
            b = x["size_b_regression"]
            fl = not rob
            name = f"{met} (robustness)" if rob else met
            raw = "—" if b_only else _ci(x["raw"], fl)
            a_ = "—" if b_only else _ci(x["size_a_same_size_different_members"], fl)
            L.append(f"| {name} | {_fmt(x['mean_lcc'])} | {_fmt(x['mean_baseline'])} | {raw} | {a_} | "
                     f"{_ci(b.get('intercept'), fl)} | {_ci(b.get('log_size_ratio'), False)} |")
        prim = r["metrics"].get("covisit_pair_rate", {}).get("by_lcc_size", {})
        if prim and not b_only:
            L += ["", "By LCC size (primary metric, descriptive):", "", "| LCC size | n | mean LCC | mean baseline | diff |",
                  "| --- | --- | --- | --- | --- |"]
            for bb, x in prim.items():
                L.append(f"| {bb} | {x['n']} | {_fmt(x['mean_lcc'])} | {_fmt(x['mean_baseline'])} | "
                         f"{_ci(x['diff'], False)} |")
        L.append("")
    L += ["## H3 (descriptive): baseline groups, LCC non-empty minus empty", ""]
    for m, r in A["H3"].items():
        L += [f"### {m} (non-empty {r['n_lcc_nonempty']}, empty {r['n_lcc_empty']}, of which not proven "
              f"{r['n_lcc_empty_unproven']})", "",
              "| metric | empty side | non-empty | empty | raw diff | coef controlling log size |",
              "| --- | --- | --- | --- | --- | --- |"]
        for key, lab in (("metrics", "all empty"), ("metrics_proven_empty_only", "proven empty only")):
            for met, x in r[key].items():
                fl = "__" not in met
                L.append(f"| {met} | {lab} | {_fmt(x['raw'].get('mean_lcc_nonempty'))} | "
                         f"{_fmt(x['raw'].get('mean_lcc_empty'))} | {_ci(x['raw'], fl)} | "
                         f"{_ci(x['ols_controlling_log_size'].get('lcc_nonempty'), fl)} |")
        L.append("")
    if A.get("sweeps"):
        L += ["## Parameter study (primary metric; raw and size (b) intercept)", "",
              "| sweep | k | tolerance | queries | baseline | common | raw | (b) intercept |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for sw in A["sweeps"]:
            for m, r in sw["H1"].items():
                x = r["metrics"].get("covisit_pair_rate")
                if x:
                    raw = "—" if m in H1_B_ONLY else _ci(x["raw"])
                    L.append(f"| {sw['sweep']} | {sw['k']} | {sw['tolerance']} | {sw['queries']} | {m} | "
                             f"{r['n_common']} | {raw} | {_ci(x['size_b_regression'].get('intercept'))} |")
        L.append("")
    path.write_text("\n".join(L))


def analyse(results: Path, seed: int = 20261025, title: str = "E5 main analysis") -> dict:
    df = pd.read_csv(results / "geo_e2_per_query.csv")
    js = json.loads((results / "geo_e2.json").read_text()) if (results / "geo_e2.json").exists() else {}
    main, rob = metrics_of(df)
    methods = [m for m in df["method"].unique() if m != "lcc_exact"]
    base = [m for m in H1_PRIMARY + H1_B_ONLY if m in methods] + [m for m in methods if m not in H1_PRIMARY + H1_B_ONLY]
    A = {"title": title, "source": str(results), "queries": int(df["q"].nunique()), "k": js.get("k"),
         "tolerance": js.get("tolerance"), "metrics_main": main, "metrics_robustness": rob,
         "H1": h1(df, main + rob, base, seed), "H3": h3(df, main + rob, [m for m in H3_METHODS if m in methods], seed),
         "H2_member_level": js.get("member_level"), "abstention_geo_e2": js.get("abstention")}
    sw = results / "geo_e2_sweep_per_query.csv"
    if sw.exists():
        S = pd.read_csv(sw)
        A["sweeps"] = []
        for (sweep, k, tol), d in S.groupby(["sweep", "k", "tolerance"], sort=False):
            A["sweeps"].append({"sweep": sweep, "k": int(k), "tolerance": tol, "queries": int(d["q"].nunique()),
                                "H1": h1(d, ["covisit_pair_rate", "member_meet_rate"],
                                         [m for m in H1_PRIMARY + H1_B_ONLY if m in set(d["method"])], seed)})
    return A


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True, help="directory with geo_e2_per_query.csv")
    ap.add_argument("--out", type=Path, default=None, help="output prefix (default: <results>/e5_analysis)")
    ap.add_argument("--title", default="E5 main analysis")
    ap.add_argument("--seed", type=int, default=20261025)
    args = ap.parse_args(argv)
    out = args.out or (args.results / "e5_analysis")
    A = analyse(args.results, args.seed, args.title)
    Path(str(out) + ".json").write_text(json.dumps(A, indent=1, default=float))
    write_md(A, Path(str(out) + ".md"))
    print(f"analysis written: {out}.json, {out}.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
