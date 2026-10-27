#!/usr/bin/env python3
"""E12 Q1, the premise test (docs/PLANS.md (lacs_e12_plan_20261029) section 5, finalised 2026-10-26): do tolerances
estimated from the training-period retweets predict who retweets whom in the test period, beyond ideological
distance and the past retweet relations?

  python -m e12.premise --run runs/e12 [--results runs/e12/results] [--n-perm 1000] [--procs 2]

Sample: ordered dyads (u, v), u != v, of the E12 nodes (RUN/geo/users.csv); y_uv = 1 when u retweeted v in
the test period (RUN/geo/e12_pairs.npz test_u -> test_v). The sender u is the member whose tolerance is used.
Models: E6's penalised fixed-effects logit (e6.premise.fit_fe_logit, unchanged):
  M0: logit P(y_uv) = c + alpha_u + beta_v + s(d_uv) + same group + same country + past u->v + past v->u
  M1: M0 + gamma * 1[d_uv <= theta_u]
s(d) is the piecewise-linear spline of e6.premise (knots at the 10/25/50/75/90% quantiles of d over all ordered
dyads); past u->v = u retweeted v in the training period; lambda = 1 on the fixed effects. Ties: d_uv <=
theta_u * (1 + 1e-9) on users.csv's theta_eng = theta_raw * (1 + 1e-6) + 1e-6, exactly like E6.
Primary statistic: gamma (1-D, theta_q50) with a one-sided permutation p-value (theta shuffled within political
groups, 1,000 permutations). Secondary (same permutations and out-of-fold comparison): the 5-fold out-of-fold
log loss M1 - M0 with a sender-cluster bootstrap CI; d_uv / theta_u; 2-D distances with the 2-D tolerances;
theta_q70, theta_q90; the position check (M0 + d*x_u + d*|x_u - median x of u's group|, permutations within
group x position tercile); the new-relations variant (only dyads without any training retweet between the two,
without the two past terms; folds = the main folds on that sample). Report only (not in the verdict): past
counts log(1 + c) instead of the two indicators; dyads whose two members were in office on 2015-07-01;
positions and theta_q50 from the training-period roll calls (RUN/geo/users_trainvotes.csv).
Member level: Spearman(theta_u, r_u), r_u = median distance to the nodes u retweeted in the test period (>= 5),
the partial rank correlation controlling the group and m_u; and, as a description, r_u^new over the test
targets u had not retweeted in the training period. Decision table of E6 section 5, then the readings of the
position check and of the new-relations variant (plan section 5).
A main fit (M0 or M1 of any variant) that does not converge stops the run (exit 3).
Outputs in --results: e12_premise.json, e12_premise_perm.csv, e12_premise_members.csv,
e12_premise_members_new.csv, e12_premise.log.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e6.common import now_iso, setup_logging, write_json  # noqa: E402
from e6.premise import (LAMBDA, N_FOLDS, fit_fe_logit, member_level, nll_grid, run_permutations,  # noqa: E402
                        spline_knots, term)

LOG = logging.getLogger("e12.premise")
SEED = 20261029
VERDICTS = {
    "predicts": "训练期转发估出的接受域，在距离与过去关系之外还能预测测试期的转发",
    "direction_only": "方向一致，预测增益不显著",
    "no_effect": "没有可测的作用",
}
NO_EFFECT_NOTE = "Q2–Q4 照常报告，但正文写明这一接受域估计在测试期没有得到验证"
# variant -> (distance, theta column, term, design, sample, grid); report-only variants are marked below
VARIANTS = {
    "primary_1d_q50_indicator": ("1d", "theta_q50", "indicator", "base", "all", "main"),
    "continuous_1d_q50_ratio": ("1d", "theta_q50", "ratio", "base", "all", "main"),
    "dist2d_q50_indicator": ("2d", "theta2d_q50", "indicator", "base", "all", "main"),
    "q70_1d_indicator": ("1d", "theta_q70", "indicator", "base", "all", "main"),
    "q90_1d_indicator": ("1d", "theta_q90", "indicator", "base", "all", "main"),
    "posadj_1d_q50_indicator": ("1d", "theta_q50", "indicator", "posadj", "all", "main"),
    "newpairs_1d_q50_indicator": ("1d", "theta_q50", "indicator", "nopast", "newpairs", "main"),
    "strength_1d_q50_indicator": ("1d", "theta_q50", "indicator", "strength", "all", "main"),
    "inoffice_1d_q50_indicator": ("1d", "theta_q50", "indicator", "base", "inoffice", "main"),
    "trainvotes_1d_q50_indicator": ("1d", "theta_q50", "indicator", "base", "all", "trainvotes"),
}
REPORT_ONLY = {"strength_1d_q50_indicator", "inoffice_1d_q50_indicator", "trainvotes_1d_q50_indicator"}
PRIMARY = "primary_1d_q50_indicator"


def load_run(run: Path) -> dict:
    geo = run / "geo"
    users = pd.read_csv(geo / "users.csv", float_precision="round_trip")
    if not (users["idx"].to_numpy() == np.arange(len(users))).all():
        raise ValueError("users.csv: idx must be 0..n-1 in order")
    z = np.load(geo / "e12_pairs.npz")
    D = {"users": users, "pos1": np.load(geo / "positions.npy"), "pos2": np.load(geo / "positions_2d.npy"),
         **{k: z[k].astype(np.int64) for k in ("train_u", "train_v", "train_n", "test_u", "test_v")}}
    tvp = geo / "users_trainvotes.csv"
    D["users_tv"] = pd.read_csv(tvp, float_precision="round_trip") if tvp.exists() else None
    return D


def make_dyads(users: pd.DataFrame, D: dict, keep: np.ndarray | None = None, pos1: np.ndarray | None = None,
               pos2: np.ndarray | None = None) -> dict:
    """Dyad data on the grid of the nodes `keep` (indices into users.csv; default all), in that order."""
    n0 = len(users)
    keep = np.arange(n0) if keep is None else np.asarray(keep, np.int64)
    n = len(keep)
    off = ~np.eye(n, dtype=bool)
    u, v = np.nonzero(off)
    Y0 = np.zeros((n0, n0))
    Y0[D["test_u"], D["test_v"]] = 1.0  # u retweeted v in the test period
    C0 = np.zeros((n0, n0))
    C0[D["train_u"], D["train_v"]] = D["train_n"]  # distinct original tweets of v retweeted by u in training
    Y, C = Y0[np.ix_(keep, keep)], C0[np.ix_(keep, keep)]
    np.fill_diagonal(Y, 0.0)
    np.fill_diagonal(C, 0.0)
    past = (C > 0).astype(np.float64)
    out = {"n": n, "keep": keep, "off": off, "M": off.astype(np.float64), "u": u.astype(np.int64),
           "v": v.astype(np.int64), "Y": Y, "y": Y[u, v], "past_uv": past, "past_vu": past.T.copy(),
           "logc_uv": np.log1p(C), "logc_vu": np.log1p(C.T)}
    p1 = D["pos1"][keep] if pos1 is None else pos1
    p2 = D["pos2"][keep] if pos2 is None else pos2
    for key, pos in (("d1d", p1), ("d2d", p2)):
        out[key] = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    sub = users.iloc[keep]
    for c in ("group", "country"):
        a = sub[c].astype(str).to_numpy()
        out[f"same_{c}"] = (a[:, None] == a[None, :]).astype(np.float64)
    out["strata"] = sub["group"].astype(str).to_numpy()
    return out


def design(dy: dict, dist: str, kind: str, x_u: np.ndarray | None = None, ext_u: np.ndarray | None = None):
    """M0 design (p, n, n): intercept, spline of d (knots over all ordered dyads of the grid), same group, same
    country, then the past terms of the variant (base / posadj: two indicators; strength: log(1 + c);
    nopast: none) and, for posadj, d*x_u and d*ext_u."""
    d = dy[f"d{dist}"]
    knots = spline_knots(d[dy["off"]])
    cols = [np.ones_like(d), d] + [np.maximum(d - k, 0.0) for k in knots]
    names = ["intercept", "d"] + [f"(d-{k:.4f})+" for k in knots]
    cols += [dy["same_group"], dy["same_country"]]
    names += ["same_group", "same_country"]
    if kind in ("base", "posadj"):
        cols += [dy["past_uv"], dy["past_vu"]]
        names += ["past_u_to_v", "past_v_to_u"]
    elif kind == "strength":
        cols += [dy["logc_uv"], dy["logc_vu"]]
        names += ["log1p_count_u_to_v", "log1p_count_v_to_u"]
    elif kind != "nopast":
        raise ValueError(kind)
    if kind == "posadj":
        cols += [d * x_u[:, None], d * ext_u[:, None]]
        names += ["d*x_u", "d*ext_u"]
    return np.stack(cols), names, knots


def oof_nll(Xg: np.ndarray, Y: np.ndarray, M: np.ndarray, us: np.ndarray, vs: np.ndarray, folds: np.ndarray,
            x0: np.ndarray) -> tuple[np.ndarray, int]:
    """Out-of-fold log loss of the sample dyads (us, vs): fit on the sample's other folds, predict the held-out
    one (with M = all off-diagonal dyads this is e6.premise.oof_nll). Returns (losses, fits not converged)."""
    out = np.empty(len(us))
    bad = 0
    for k in range(N_FOLDS):
        te = folds == k
        Mk = M.copy()
        Mk[us[te], vs[te]] = 0.0
        fk = fit_fe_logit(Xg, Y, Mk, x0=x0)
        out[te] = nll_grid(fk, Xg, Y)[us[te], vs[te]]
        if not fk["converged"]:
            bad += 1
            LOG.warning("out-of-fold fit %d did not converge: %s", k, fk["message"])
    return out, bad


def decide(gamma: float, p: float, ci) -> dict:
    if gamma > 0 and p < 0.05 and ci is not None and ci[1] < 0:
        code = "predicts"
    elif gamma > 0 and p < 0.05 and ci is not None and ci[0] <= 0 <= ci[1]:
        code = "direction_only"
    else:
        code = "no_effect"
    out = {"code": code, "text": VERDICTS[code], "gamma": gamma, "p_perm": p, "oof_diff_ci95": ci,
           "rule": "E12 plan section 5 (= E6 section 5): gamma > 0, permutation p < 0.05 and out-of-fold log-loss "
                   "difference CI < 0 -> 'predicts'; gamma > 0, p < 0.05, CI contains 0 -> 'direction_only'; "
                   "otherwise 'no_effect'"}
    if code == "no_effect":
        out["note"] = NO_EFFECT_NOTE
    return out


class _NotConverged(Exception):
    pass


def run(args) -> int:
    from lacs.model.metrics import cluster_bootstrap_diff

    run_dir = Path(args.run)
    res_dir = Path(args.results) if args.results else run_dir / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(res_dir / "e12_premise.log")
    t0 = time.time()
    D = load_run(run_dir)
    users = D["users"]
    dy = make_dyads(users, D)
    n, u, v, y = dy["n"], dy["u"], dy["v"], dy["y"]
    LOG.info("Q1: %d members, %d ordered dyads, %d positive (rate %.4f), past u->v %d", n, len(y), int(y.sum()),
             y.mean(), int(dy["past_uv"].sum()))
    out = {"tool": "e12.premise", "plan": "docs/PLANS.md (lacs_e12_plan_20261029) (finalised 2026-10-26)",
           "started": now_iso(), "run": str(run_dir), "n_members": n, "n_dyads": int(len(y)),
           "n_positive": int(y.sum()), "positive_rate": float(y.mean()),
           "n_past_uv": int(dy["past_uv"][dy["off"]].sum()), "lambda": LAMBDA, "seed": args.seed,
           "n_perm": args.n_perm, "folds": N_FOLDS, "n_boot": args.n_boot, "sender": "u (retweets; theta_u used)",
           "receiver": "v", "report_only": sorted(REPORT_ONLY), "variants": {}}
    folds = np.random.default_rng([args.seed, 99]).permutation(len(y)) % N_FOLDS
    out["fold_sizes"] = np.bincount(folds, minlength=N_FOLDS).tolist()
    # grids: the main one and the training-vote one (report only)
    grids = {"main": {"dy": dy, "users": users, "folds": folds}}
    if any(spec[5] == "trainvotes" for spec in VARIANTS.values()):
        utv = D["users_tv"]
        if utv is None:
            raise SystemExit("users_trainvotes.csv missing (e12.build writes it)")
        keep = utv["idx"].to_numpy(np.int64)
        p1 = np.column_stack([utv["dim1"].to_numpy(np.float64), np.zeros(len(utv)), np.zeros(len(utv))])
        dtv = make_dyads(users, D, keep=keep, pos1=p1, pos2=p1)
        ftv = np.random.default_rng([args.seed, 99]).permutation(len(dtv["y"])) % N_FOLDS
        utab = users.iloc[keep].reset_index(drop=True).copy()
        utab["theta_q50"] = utv["theta_q50"].to_numpy(np.float64)
        utab["dim1"] = utv["dim1"].to_numpy(np.float64)
        grids["trainvotes"] = {"dy": dtv, "users": utab, "folds": ftv}
        out["trainvotes_grid"] = {"nodes": int(len(keep)), "nodes_dropped": int(n - len(keep)),
                                  "dyads": int(len(dtv["y"])), "positive": int(dtv["y"].sum())}
    # position terms and terciles per grid
    for gname, G in grids.items():
        dyg, ug = G["dy"], G["users"]
        x_u = ug["dim1"].to_numpy(np.float64)
        ext_u = np.zeros(dyg["n"])
        strata_pos = dyg["strata"].astype(object).copy()
        for st in np.unique(dyg["strata"]):
            idx = np.flatnonzero(dyg["strata"] == st)
            ext_u[idx] = np.abs(x_u[idx] - np.median(x_u[idx]))
            r = np.argsort(np.argsort(x_u[idx], kind="stable"), kind="stable")
            strata_pos[idx] = [f"{st}-t{int(3 * k // max(len(idx), 1))}" for k in r]
        G.update(x_u=x_u, ext_u=ext_u, strata_pos=strata_pos.astype(str))
    inoff = users["in_office_20150701"].to_numpy(np.int64) > 0
    samples = {"all": dy["M"],
               "newpairs": dy["M"] * (1.0 - dy["past_uv"]) * (1.0 - dy["past_vu"]),
               "inoffice": dy["M"] * (inoff[:, None] & inoff[None, :]).astype(np.float64)}
    base, perm_rows = {}, []
    try:
        for name, (dist, tcol, kind, dkind, sample, gname) in VARIANTS.items():
            G = grids[gname]
            dyg, ug = G["dy"], G["users"]
            M = dyg["M"] if gname != "main" else samples[sample]
            sel = M[dyg["u"], dyg["v"]] > 0
            us, vs, fs = dyg["u"][sel], dyg["v"][sel], G["folds"][sel]
            bkey = (gname, dist, dkind, sample)
            if bkey not in base:
                X0, names0, knots = design(dyg, dist, dkind, G["x_u"], G["ext_u"])
                f0 = fit_fe_logit(X0, dyg["Y"], M)
                if not f0["converged"]:
                    raise _NotConverged(f"M0 of {name}: {f0['message']}")
                nll0, bad0 = oof_nll(X0, dyg["Y"], M, us, vs, fs, f0["x"])
                base[bkey] = (X0, names0, knots, f0, nll0, bad0)
                LOG.info("M0 (%s): %d iterations, converged %s", bkey, f0["nit"], f0["converged"])
            X0, names0, knots, f0, nll0, bad0 = base[bkey]
            strata_v = G["strata_pos"] if dkind == "posadj" else dyg["strata"]
            d = dyg[f"d{dist}"]
            theta = ug[tcol].to_numpy(np.float64)
            zg = term(kind, d, theta)
            X1 = np.concatenate([X0, zg[None]], axis=0)
            x0 = np.concatenate([f0["coef"], [0.0], f0["alpha"], f0["beta"]])
            f1 = fit_fe_logit(X1, dyg["Y"], M, x0=x0)
            if not f1["converged"]:
                raise _NotConverged(f"M1 of {name}: {f1['message']}")
            nll1, bad1 = oof_nll(X1, dyg["Y"], M, us, vs, fs, f1["x"])
            diff = cluster_bootstrap_diff(nll1, nll0, us, n_boot=args.n_boot, seed=args.seed)
            coef = float(f1["coef"][-1])
            payload = {"seed": args.seed, "theta": theta, "strata": strata_v, "kind": kind, "d": d, "Y": dyg["Y"],
                       "M": M, "lam": LAMBDA, "X0": X0, "x0": x0}
            tp = time.time()
            perms = run_permutations(payload, args.n_perm, args.procs)
            null = np.array([g for _, g, _ in perms], float)
            direction = "less" if kind == "ratio" else "greater"
            if len(null):
                hits = (null <= coef).sum() if direction == "less" else (null >= coef).sum()
                p = float((1 + hits) / (1 + len(null)))
                perm = {"n": int(len(null)), "p_one_sided": p, "direction": direction, "null_mean": float(null.mean()),
                        "null_sd": float(null.std(ddof=1)) if len(null) > 1 else None,
                        "null_q025_q975": [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
                        "not_converged": int(sum(not c for _, _, c in perms)), "seconds": round(time.time() - tp, 1),
                        "strata": "group x position tercile" if dkind == "posadj" else "group"}
            else:
                perm = {"n": 0, "p_one_sided": None, "direction": direction}
            perm_rows += [{"variant": name, "b": b, "coef": g, "converged": c} for b, g, c in perms]
            out["variants"][name] = {
                "distance": dist, "theta": tcol, "term": "1[d_uv <= theta_u]" if kind == "indicator" else "d_uv / theta_u",
                "design": dkind, "sample": sample, "grid": gname, "report_only": name in REPORT_ONLY,
                "n_dyads": int(sel.sum()), "n_positive": int(dyg["Y"][us, vs].sum()), "coef": coef, "perm": perm,
                "oof": {"nll_m0": float(nll0.mean()), "nll_m1": float(nll1.mean()), **diff,
                        "fits_not_converged": int(bad0 + bad1),
                        "note": "mean over the sample's dyads of out-of-fold log loss M1 - M0; CI: sender-cluster bootstrap"},
                "m0": {"coef": dict(zip(names0, f0["coef"].tolist())), "objective": f0["objective"],
                       "converged": f0["converged"], "nit": f0["nit"], "grad_max": f0["grad_max"]},
                "m1": {"coef": dict(zip(names0 + ["term"], f1["coef"].tolist())), "objective": f1["objective"],
                       "converged": f1["converged"], "nit": f1["nit"], "grad_max": f1["grad_max"]},
                "knots": knots.tolist(), "term_mean": float(zg[us, vs].mean())}
            LOG.info("Q1 %s: coef %.4f, permutation p %s (%d), OOF diff %.5f CI %s", name, coef, perm["p_one_sided"],
                     perm["n"], diff["mean_diff"], np.round(diff["ci95"], 5).tolist())
    except _NotConverged as e:
        out["stopped"] = f"a main fit did not converge: {e} (plan section 5: stop and look for the cause)"
        write_json(out, res_dir / "e12_premise.json")
        LOG.error(out["stopped"])
        return 3
    ml, tab = member_level(users, dy["Y"], D["pos1"], dy["strata"], args.n_boot, args.seed)
    ml["partial_controls"] = "political group dummies, rank of m_u (Pearson correlation of rank residuals)"
    out["member_level"] = ml
    Ynew = dy["Y"] * (1.0 - dy["past_uv"])  # test targets u had not retweeted in the training period
    mln, tabn = member_level(users, Ynew, D["pos1"], dy["strata"], args.n_boot, args.seed)
    mln["note"] = "description only: r_u^new over the test targets u had not retweeted in the training period"
    out["member_level_new_targets"] = mln
    pv = out["variants"].get(PRIMARY, {})
    out["verdict"] = decide(pv.get("coef", float("nan")), (pv.get("perm") or {}).get("p_one_sided") or 1.0,
                            (pv.get("oof") or {}).get("ci95"))
    if out["verdict"]["code"] != "no_effect":
        for vname, key, yes, no in (
                ("posadj_1d_q50_indicator", "position_check", "位置调整后仍成立",
                 "这一作用与随位置变化的距离衰减分不开"),
                ("newpairs_1d_q50_indicator", "new_relations", "在训练期没有转发关系的议员对上也成立",
                 "这一作用主要来自已有转发关系的延续")):
            vv = out["variants"].get(vname, {})
            pp = (vv.get("perm") or {}).get("p_one_sided")
            if pp is not None:
                ok = bool(vv.get("coef", 0) > 0 and pp < 0.05)
                out["verdict"][key] = {"variant": vname, "coef": vv.get("coef"), "p_one_sided": pp, "holds": ok,
                                       "text": yes if ok else no}
    out["finished"] = now_iso()
    out["seconds"] = round(time.time() - t0, 1)
    pd.DataFrame(perm_rows, columns=["variant", "b", "coef", "converged"]).to_csv(res_dir / "e12_premise_perm.csv",
                                                                                index=False)
    tab.to_csv(res_dir / "e12_premise_members.csv", index=False)
    tabn.to_csv(res_dir / "e12_premise_members_new.csv", index=False)
    write_json(out, res_dir / "e12_premise.json")
    LOG.info("Q1 verdict: %s (%s) in %.0fs", out["verdict"]["code"], out["verdict"]["text"], out["seconds"])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m e12.premise", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--results", default=None)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-boot", type=int, default=2000)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
