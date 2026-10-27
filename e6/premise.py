#!/usr/bin/env python3
"""E6 Q1, the premise test (plan section 5): do cosponsorship tolerances predict who engages with
whom on Twitter beyond ideological distance?

  python -m e6.premise --run runs/e6 [--results runs/e6/results] [--n-perm 1000] [--procs 8]

Sample: ordered dyads (u, v), u != v, of the E6 members (RUN/geo/users.csv); y_uv = 1 when u engaged
with v in the tweet period, i.e. the network has the edge v -> u (RUN/geo/twitter_weights.npz:
src = v, dst = u). The dyad's sender is u (the member who engages and whose tolerance is used),
the receiver is v.

Models (logistic regression, fitted here with a vectorised loss / gradient and scipy L-BFGS-B):
  M0: logit P(y_uv) = c + alpha_u + beta_v + s(d_uv) + b1 same caucus + b2 same state + b3 same chamber
  M1: M0 + gamma * 1[d_uv <= theta_u]
s(d) is a piecewise-linear spline (d and (d - k)_+ for knots k at the 10/25/50/75/90% quantiles of d
over the dyads). Objective: sum of the dyads' log losses + (lambda / 2) * (|alpha|^2 + |beta|^2) with
lambda = 1, i.e. the fixed effects (only they) get an L2 penalty / N(0, 1) prior; the intercept and
the other coefficients are unpenalised. "Same party" is the caucus (independents with the Democrats).
Exact ties d_uv == theta_u (common: scores have three decimals and theta_u is itself a distance) count
as accepted (relative tolerance e6.common.TIE_RTOL = 1e-9); users.csv is read with
float_precision='round_trip', so the tolerances are exactly the ones e6.build computed.

Primary statistic: gamma (1-D distance |dim1_u - dim1_v|, theta_q50), with a one-sided permutation
p-value (theta shuffled across members within caucus x chamber strata, M1 refitted, 1,000
permutations, p = (1 + #{gamma_b >= gamma}) / (1 + B)).
Secondary: 5-fold out-of-fold log loss M1 - M0 (dyads assigned to folds at random) with a 95% CI
from a sender-cluster bootstrap; the continuous term d_uv / theta_u instead of the indicator (its
coefficient is expected to be negative; one-sided p for "< 0"); 2-D distances (dim1, dim2) with
the 2-D tolerances; theta_q70 and theta_q90. Each variant gets the same permutation test and the
out-of-fold comparison against its own M0.
Member level: Spearman(theta_u, r_u), r_u = median distance from u to the members u engaged with (at
least 5), and the partial rank correlation controlling caucus x chamber and m_u (median distance
from u to all other members); 95% CIs from a member bootstrap.
The decision table of section 5 is applied to the primary variant; the verdict is in the output.

Outputs in --results: e6_premise.json, e6_premise_perm.csv (gamma of every permutation),
e6_premise_members.csv (theta_u, r_u, m_u per member), e6_premise.log.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):  # one BLAS thread per worker process
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import logging  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import expit  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # also works under python -I
    sys.path.insert(0, str(ROOT))

from e6.common import SEED, TIE_RTOL, now_iso, setup_logging, write_json  # noqa: E402

LOG = logging.getLogger("e6.premise")
LAMBDA = 1.0                       # plan: L2 penalty lambda = 1 on the fixed effects
KNOT_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
N_FOLDS = 5
MIN_PARTNERS = 5                   # member level: r_u needs >= 5 members u engaged with
VERDICTS = {
    "predicts": "联署估出的接受域，在距离之外还能预测 Twitter 互动",
    "direction_only": "方向一致，预测增益不显著",
    "no_effect": "没有可测的作用",
}
NO_EFFECT_NOTE = "Q2–Q4 照常报告，但正文写明这一接受域估计在 Twitter 上没有得到验证"
# variant name -> (distance, theta column of users.csv, term)
VARIANTS = {
    "primary_1d_q50_indicator": ("1d", "theta_q50", "indicator"),
    "continuous_1d_q50_ratio": ("1d", "theta_q50", "ratio"),
    "dist2d_q50_indicator": ("2d", "theta2d_q50", "indicator"),
    "q70_1d_indicator": ("1d", "theta_q70", "indicator"),
    "q90_1d_indicator": ("1d", "theta_q90", "indicator"),
    # pre-declared 10-25 (before any data): does theta add beyond a distance decay that varies with the
    # sender's own position? M0 also gets d*x_u and d*|x_u - median x of u's caucus x chamber|, and theta is
    # permuted within caucus x chamber x position tercile (so the null keeps the theta-position link)
    "posadj_1d_q50_indicator": ("1d", "theta_q50", "indicator", "posadj"),
}
PRIMARY = "primary_1d_q50_indicator"


# ------------------------------------------------------------------ data --

def load_run(run: Path) -> dict:
    geo = run / "geo"
    # exact float round trip: ties d_uv == theta_u decide the indicator
    users = pd.read_csv(geo / "users.csv", float_precision="round_trip")
    if not (users["idx"].to_numpy() == np.arange(len(users))).all():
        raise ValueError("users.csv: idx must be 0..n-1 in order")
    pos1 = np.load(geo / "positions.npy")
    p2 = geo / "positions_2d.npy"
    pos2 = np.load(p2) if p2.exists() else None
    tw = np.load(geo / "twitter_weights.npz")
    return {"users": users, "pos1": pos1, "pos2": pos2, "src": tw["src"].astype(np.int64),
            "dst": tw["dst"].astype(np.int64)}


def make_dyads(D: dict) -> dict:
    """Dyad data on the n x n grid (row = sender u, column = receiver v; the diagonal is masked out).
    The dyad order used for per-dyad outputs is np.nonzero(off-diagonal), i.e. row-major."""
    users = D["users"]
    n = len(users)
    off = ~np.eye(n, dtype=bool)
    u, v = np.nonzero(off)
    Y = np.zeros((n, n), dtype=np.float64)
    Y[D["dst"], D["src"]] = 1.0  # edge src -> dst: dst engaged with src, so y[dst, src] = 1
    np.fill_diagonal(Y, 0.0)
    out = {"n": n, "off": off, "M": off.astype(np.float64), "u": u.astype(np.int64), "v": v.astype(np.int64),
           "Y": Y, "y": Y[u, v]}
    for key, pos in (("d1d", D["pos1"]), ("d2d", D["pos2"])):
        if pos is not None:
            out[key] = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    for c in ("caucus", "state", "chamber"):
        a = users[c].astype(str).to_numpy()
        out[f"same_{c}"] = (a[:, None] == a[None, :]).astype(np.float64)
    out["strata"] = (users["caucus"].astype(str) + "-" + users["chamber"].astype(str)).to_numpy()
    return out


def spline_knots(d: np.ndarray) -> np.ndarray:
    k = np.quantile(d, KNOT_QUANTILES)
    return np.unique(k)  # coinciding knots would give identical columns


def base_design(dy: dict, dist: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    """M0 design as a (p, n, n) stack: intercept (always first), the spline of d, same caucus /
    state / chamber. Knots: quantiles of d over the ordered dyads (off-diagonal cells)."""
    d = dy[f"d{dist}"]
    knots = spline_knots(d[dy["off"]])
    cols = [np.ones_like(d), d] + [np.maximum(d - k, 0.0) for k in knots]
    names = ["intercept", "d"] + [f"(d-{k:.4f})+" for k in knots]
    cols += [dy["same_caucus"], dy["same_state"], dy["same_chamber"]]
    names += ["same_caucus", "same_state", "same_chamber"]
    return np.stack(cols), names, knots


def term(kind: str, d: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """The tolerance term on the grid: row u uses theta_u (the sender's tolerance)."""
    if kind == "indicator":  # ties d == theta_u count as accepted (common TIE_RTOL)
        return (d <= theta[:, None] * (1.0 + TIE_RTOL)).astype(np.float64)
    if kind == "ratio":
        return d / theta[:, None]
    raise ValueError(kind)


# ------------------------------------------------------------------- fit --

def _softplus(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))  # log(1 + e^x) without overflow


def fit_fe_logit(Xg: np.ndarray, Y: np.ndarray, M: np.ndarray, lam: float = LAMBDA, x0: np.ndarray | None = None,
                 maxiter: int = 20000) -> dict:
    """Penalised logit with sender (alpha_u, rows) and receiver (beta_v, columns) effects on the grid
    (see the module docstring). Xg: (p, n, n) covariates, Xg[0] = intercept; Y: outcomes; M: 0/1 mask
    of the dyads in the fit (the diagonal and held-out dyads are 0). x0: start (coefficients on the
    original scale, then alpha, beta).

    Objective sum_M [log(1 + e^eta) - y eta] + lam / 2 (|alpha|^2 + |beta|^2), minimised with scipy's
    L-BFGS-B on the exact gradient. For conditioning only (the optimum is unchanged): covariates are
    scaled to unit standard deviation over the masked dyads, and all parameters are rescaled by the
    square root of the diagonal of the Hessian at the start point."""
    p, n, _ = Xg.shape
    Xf = Xg.reshape(p, n * n)
    Mf = M.reshape(-1)
    Yf = Y.reshape(-1)
    msum = Mf.sum()
    mu = (Xf * Mf).sum(axis=1) / msum
    sd = np.sqrt((((Xf - mu[:, None]) ** 2) * Mf).sum(axis=1) / msum)
    sd[sd < 1e-12] = 1.0
    Xs = Xf / sd[:, None]
    if x0 is None:
        w0 = np.zeros(p + 2 * n)
        ybar = float(np.clip((Yf * Mf).sum() / msum, 1e-6, 1 - 1e-6))
        w0[0] = np.log(ybar / (1 - ybar))
    else:
        w0 = np.array(x0, dtype=np.float64, copy=True)
        w0[:p] *= sd

    def eta_of(w):
        e = (w[:p] @ Xs).reshape(n, n)
        e += w[p:p + n][:, None]
        e += w[p + n:][None, :]
        return e

    pi = expit(eta_of(w0))
    W = M * pi * (1.0 - pi)
    h = np.sqrt(np.concatenate([(Xs ** 2) @ W.reshape(-1), W.sum(axis=1) + lam, W.sum(axis=0) + lam]))
    h[h < 1e-8] = 1.0

    def fg(z):
        w = z / h
        a, c = w[p:p + n], w[p + n:]
        e = eta_of(w)
        loss = float((M * (_softplus(e) - Y * e)).sum() + 0.5 * lam * (a @ a + c @ c))
        R = M * (expit(e) - Y)
        g = np.empty(p + 2 * n)
        g[:p] = Xs @ R.reshape(-1)
        g[p:p + n] = R.sum(axis=1) + lam * a
        g[p + n:] = R.sum(axis=0) + lam * c
        return loss, g / h

    res = minimize(fg, w0 * h, jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter, "maxcor": 30, "ftol": 1e-14, "gtol": 1e-6})
    w = res.x / h
    _, gz = fg(res.x)
    coef = w[:p] / sd
    return {"coef": coef, "alpha": w[p:p + n], "beta": w[p + n:], "objective": float(res.fun), "nit": int(res.nit),
            "converged": bool(res.success), "message": str(res.message), "grad_max": float(np.abs(gz * h).max()),
            "x": np.concatenate([coef, w[p:]])}


def nll_grid(fit: dict, Xg: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Log loss of every cell of the grid under a fitted model."""
    p, n, _ = Xg.shape
    e = (fit["coef"] @ Xg.reshape(p, n * n)).reshape(n, n) + fit["alpha"][:, None] + fit["beta"][None, :]
    return _softplus(e) - Y * e


def oof_nll(Xg: np.ndarray, dy: dict, folds: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Out-of-fold log loss per dyad (dyad order): fit on the other folds, predict the held-out one."""
    u, v = dy["u"], dy["v"]
    n = dy["n"]
    out = np.empty(len(u))
    for k in range(N_FOLDS):
        te = folds == k
        Mk = dy["M"].copy()
        Mk[u[te], v[te]] = 0.0
        fk = fit_fe_logit(Xg, dy["Y"], Mk, x0=x0)
        out[te] = nll_grid(fk, Xg, dy["Y"])[u[te], v[te]]
        if not fk["converged"]:
            LOG.warning("out-of-fold fit %d did not converge: %s", k, fk["message"])
        del Mk
    assert out.shape[0] == n * (n - 1)
    return out


def permute_within(values: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = values.copy()
    for s in np.unique(strata):  # sorted: same order on every machine
        idx = np.flatnonzero(strata == s)
        out[idx] = values[idx[rng.permutation(len(idx))]]
    return out


# ------------------------------------------------------------ permutations --

_W: dict = {}


def _init_worker(payload: dict) -> None:
    _W.clear()
    _W.update(payload)


def _perm_one(b: int):
    W = _W
    rng = np.random.default_rng([W["seed"], int(b)])  # permutation b is the same in every variant
    th = permute_within(W["theta"], W["strata"], rng)
    Xg = np.concatenate([W["X0"], term(W["kind"], W["d"], th)[None]], axis=0)
    f = fit_fe_logit(Xg, W["Y"], W["M"], W["lam"], x0=W["x0"])
    return int(b), float(f["coef"][-1]), bool(f["converged"])


def run_permutations(payload: dict, n_perm: int, procs: int) -> list[tuple[int, float, bool]]:
    bs = list(range(1, n_perm + 1))
    if n_perm <= 0:
        return []
    if procs <= 1:
        _init_worker(payload)
        return [_perm_one(b) for b in bs]
    method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    with mp.get_context(method).Pool(processes=procs, initializer=_init_worker, initargs=(payload,)) as pool:
        return list(pool.imap(_perm_one, bs, chunksize=max(1, n_perm // (procs * 8))))


# ------------------------------------------------------------ member level --

def _partial_rank_corr(x, y, cat, z) -> float:
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    levels = np.unique(cat)
    Z = np.column_stack([np.ones(len(x)), rz] + [(cat == lv).astype(float) for lv in levels[1:]])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    den = np.sqrt((ex @ ex) * (ey @ ey))
    return float(ex @ ey / den) if den > 0 else float("nan")


def member_level(users: pd.DataFrame, Y: np.ndarray, pos: np.ndarray, strata: np.ndarray, n_boot: int,
                 seed: int) -> tuple[dict, pd.DataFrame]:
    n = len(users)
    Dm = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    theta = users["theta_q50"].to_numpy(np.float64)
    r = np.full(n, np.nan)
    k = Y.sum(axis=1).astype(int)  # members u engaged with
    m = np.empty(n)
    for i in range(n):
        others = np.arange(n) != i
        m[i] = np.median(Dm[i, others])
        if k[i] >= MIN_PARTNERS:
            r[i] = np.median(Dm[i, Y[i] > 0])
    ok = np.isfinite(r)
    tab = pd.DataFrame({"idx": np.arange(n), "user": users["user"], "theta_q50": theta, "n_engaged_with": k,
                        "r_u": r, "m_u": m, "stratum": strata})
    out = {"n_members": int(ok.sum()), "min_partners": MIN_PARTNERS, "theta": "theta_q50", "distance": "1d"}
    if ok.sum() < 5:
        out["note"] = "fewer than 5 members with r_u"
        return out, tab
    x, y, cat, z = theta[ok], r[ok], strata[ok], m[ok]
    rho = float(spearmanr(x, y).correlation)
    par = _partial_rank_corr(x, y, cat, z)
    rng = np.random.default_rng([seed, 7])
    bs_r, bs_p = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(x), len(x))
        if np.ptp(x[i]) == 0 or np.ptp(y[i]) == 0:
            continue
        bs_r.append(spearmanr(x[i], y[i]).correlation)
        bs_p.append(_partial_rank_corr(x[i], y[i], cat[i], z[i]))
    bs_r, bs_p = np.array(bs_r, float), np.array(bs_p, float)

    def ci(a):
        a = a[np.isfinite(a)]
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else None

    out.update({"spearman": rho, "spearman_ci95": ci(bs_r), "partial": par, "partial_ci95": ci(bs_p),
                "partial_controls": "caucus x chamber dummies, rank of m_u (Pearson correlation of rank residuals)",
                "n_boot": int(len(bs_r))})
    return out, tab


# --------------------------------------------------------------------- main --

def decide(gamma: float, p: float, ci) -> dict:
    if gamma > 0 and p < 0.05 and ci is not None and ci[1] < 0:
        code = "predicts"
    elif gamma > 0 and p < 0.05 and ci is not None and ci[0] <= 0 <= ci[1]:
        code = "direction_only"
    else:
        code = "no_effect"
    out = {"code": code, "text": VERDICTS[code], "gamma": gamma, "p_perm": p, "oof_diff_ci95": ci,
           "rule": "plan section 5: gamma > 0, permutation p < 0.05 and out-of-fold log-loss difference CI < 0 -> "
                   "'predicts'; gamma > 0, p < 0.05, CI contains 0 -> 'direction_only'; otherwise 'no_effect'"}
    if code == "no_effect":
        out["note"] = NO_EFFECT_NOTE
    return out


def run(args) -> int:
    from lacs.model.metrics import cluster_bootstrap_diff

    run_dir = Path(args.run)
    res_dir = Path(args.results) if args.results else run_dir / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(res_dir / "e6_premise.log")
    t0 = time.time()
    D = load_run(run_dir)
    dy = make_dyads(D)
    users, n, u, v, y = D["users"], dy["n"], dy["u"], dy["v"], dy["y"]
    LOG.info("Q1: %d members, %d ordered dyads, %d positive (rate %.4f)", n, len(y), int(y.sum()), y.mean())
    out = {"tool": "e6.premise", "started": now_iso(), "run": str(run_dir), "n_members": n, "n_dyads": int(len(y)),
           "n_positive": int(y.sum()), "positive_rate": float(y.mean()), "lambda": LAMBDA, "seed": args.seed,
           "n_perm_primary": args.n_perm, "n_perm_secondary": args.n_perm_secondary, "folds": N_FOLDS,
           "n_boot": args.n_boot, "sender": "u (engages; theta_u used)", "receiver": "v",
           "variants": {}}
    folds = np.random.default_rng([args.seed, 99]).permutation(len(y)) % N_FOLDS
    out["fold_sizes"] = np.bincount(folds, minlength=N_FOLDS).tolist()
    base = {}
    perm_rows = []
    x_u = users["dim1"].to_numpy(np.float64)
    ext_u = np.zeros(n)
    strata_pos = dy["strata"].astype(object).copy()
    for st in np.unique(dy["strata"]):  # extremity within the stratum, and position terciles for the permutation
        idx = np.flatnonzero(dy["strata"] == st)
        ext_u[idx] = np.abs(x_u[idx] - np.median(x_u[idx]))
        r = np.argsort(np.argsort(x_u[idx], kind="stable"), kind="stable")
        strata_pos[idx] = [f"{st}-t{int(3 * k // max(len(idx), 1))}" for k in r]
    strata_pos = strata_pos.astype(str)
    for name, spec in VARIANTS.items():
        dist, tcol, kind = spec[:3]
        adj = spec[3] if len(spec) > 3 else None
        if f"d{dist}" not in dy or tcol not in users.columns:
            out["variants"][name] = {"skipped": f"missing {'positions_2d.npy' if dist == '2d' else tcol}"}
            LOG.warning("Q1 %s skipped (missing inputs)", name)
            continue
        bad = int((~np.isfinite(users[tcol].to_numpy(np.float64))).sum() + (~np.isfinite(dy[f"d{dist}"])).any())
        if bad:
            out["variants"][name] = {"skipped": f"non-finite {tcol} or {dist} positions ({bad})"}
            LOG.warning("Q1 %s skipped (non-finite inputs)", name)
            continue
        bkey = (dist, adj)
        if bkey not in base:  # M0 of this distance: full fit and out-of-fold losses
            X0, names0, knots = base_design(dy, dist)
            if adj == "posadj":
                dd = dy[f"d{dist}"]
                X0 = np.concatenate([X0, (dd * x_u[:, None])[None], (dd * ext_u[:, None])[None]], axis=0)
                names0 = names0 + ["d*x_u", "d*ext_u"]
            f0 = fit_fe_logit(X0, dy["Y"], dy["M"])
            nll0 = oof_nll(X0, dy, folds, f0["x"])
            base[bkey] = (X0, names0, knots, f0, nll0)
            LOG.info("M0 (%s, %s): %d iterations, converged %s, knots %s", dist, adj, f0["nit"], f0["converged"],
                     np.round(knots, 4).tolist())
        X0, names0, knots, f0, nll0 = base[bkey]
        strata_v = strata_pos if adj == "posadj" else dy["strata"]
        d = dy[f"d{dist}"]
        theta = users[tcol].to_numpy(np.float64)
        zg = term(kind, d, theta)
        X1 = np.concatenate([X0, zg[None]], axis=0)
        x0 = np.concatenate([f0["coef"], [0.0], f0["alpha"], f0["beta"]])
        f1 = fit_fe_logit(X1, dy["Y"], dy["M"], x0=x0)
        nll1 = oof_nll(X1, dy, folds, f1["x"])
        diff = cluster_bootstrap_diff(nll1, nll0, u, n_boot=args.n_boot, seed=args.seed)
        coef = float(f1["coef"][-1])
        B = args.n_perm if name == PRIMARY else args.n_perm_secondary
        payload = {"seed": args.seed, "theta": theta, "strata": strata_v, "kind": kind, "d": d, "Y": dy["Y"],
                   "M": dy["M"], "lam": LAMBDA, "X0": X0, "x0": x0}
        tp = time.time()
        perms = run_permutations(payload, B, args.procs)
        null = np.array([g for _, g, _ in perms], float)
        direction = "less" if kind == "ratio" else "greater"
        if len(null):
            hits = (null <= coef).sum() if direction == "less" else (null >= coef).sum()
            p = float((1 + hits) / (1 + len(null)))
            perm = {"n": int(len(null)), "p_one_sided": p, "direction": direction, "null_mean": float(null.mean()),
                    "null_sd": float(null.std(ddof=1)) if len(null) > 1 else None,
                    "null_q025_q975": [float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))],
                    "not_converged": int(sum(not c for _, _, c in perms)), "seconds": round(time.time() - tp, 1),
                    "strata": "caucus x chamber x position tercile" if adj == "posadj" else "caucus x chamber"}
        else:
            perm = {"n": 0, "p_one_sided": None, "direction": direction}
        perm_rows += [{"variant": name, "b": b, "coef": g, "converged": c} for b, g, c in perms]
        out["variants"][name] = {
            "distance": dist, "theta": tcol, "term": "1[d_uv <= theta_u]" if kind == "indicator" else "d_uv / theta_u",
            "coef": coef, "perm": perm,
            "oof": {"nll_m0": float(nll0.mean()), "nll_m1": float(nll1.mean()), **diff,
                    "note": "mean over dyads of out-of-fold log loss M1 - M0; CI: sender-cluster bootstrap"},
            "m0": {"coef": dict(zip(names0, f0["coef"].tolist())), "objective": f0["objective"],
                   "converged": f0["converged"], "nit": f0["nit"], "grad_max": f0["grad_max"]},
            "m1": {"coef": dict(zip(names0 + ["term"], f1["coef"].tolist())), "objective": f1["objective"],
                   "converged": f1["converged"], "nit": f1["nit"], "grad_max": f1["grad_max"]},
            "knots": knots.tolist(), "term_mean": float(zg[dy["off"]].mean())}
        LOG.info("Q1 %s: coef %.4f, permutation p %s (%d), OOF diff %.5f CI %s", name, coef, perm["p_one_sided"],
                 perm["n"], diff["mean_diff"], np.round(diff["ci95"], 5).tolist())
    ml, tab = member_level(users, dy["Y"], D["pos1"], dy["strata"], args.n_boot, args.seed)
    out["member_level"] = ml
    pv = out["variants"].get(PRIMARY, {})
    out["verdict"] = decide(pv.get("coef", float("nan")), (pv.get("perm") or {}).get("p_one_sided") or 1.0,
                            (pv.get("oof") or {}).get("ci95"))
    pa = out["variants"].get("posadj_1d_q50_indicator", {})
    pa_p = (pa.get("perm") or {}).get("p_one_sided")
    if out["verdict"]["code"] != "no_effect" and pa_p is not None:  # pre-declared 10-25: the position check
        sep = bool(pa.get("coef", 0) > 0 and pa_p < 0.05)
        out["verdict"]["position_check"] = {
            "coef": pa.get("coef"), "p_one_sided": pa_p, "separable_from_position": sep,
            "text": ("位置调整后仍成立" if sep else "这一作用与随位置变化的距离衰减分不开（位置调整版本 p ≥ 0.05 或系数 ≤ 0）")}
    out["finished"] = now_iso()
    out["seconds"] = round(time.time() - t0, 1)
    pd.DataFrame(perm_rows, columns=["variant", "b", "coef", "converged"]).to_csv(res_dir / "e6_premise_perm.csv",
                                                                                index=False)
    tab.to_csv(res_dir / "e6_premise_members.csv", index=False)
    write_json(out, res_dir / "e6_premise.json")
    LOG.info("Q1 verdict: %s (%s) in %.0fs", out["verdict"]["code"], out["verdict"]["text"], out["seconds"])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m e6.premise", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", "--geo", dest="run", required=True,
                    help="geo run directory written by e6.build (e.g. runs/e6)")
    ap.add_argument("--results", "--out", dest="results", default=None, help="default: RUN/results")
    ap.add_argument("--n-perm", type=int, default=1000, help="permutations of the primary variant (plan: 1000)")
    ap.add_argument("--n-perm-secondary", type=int, default=None, help="permutations of each secondary variant "
                                                                       "(default: --n-perm)")
    ap.add_argument("--procs", "--processes", dest="procs", type=int, default=min(8, os.cpu_count() or 1),
                    help="worker processes for the permutations")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)
    if args.n_perm_secondary is None:
        args.n_perm_secondary = args.n_perm
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
