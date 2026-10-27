"""Static open-group conditions of the proposition in docs/PLANS.md (lacs_e1_open_groups_20261025).

Notation: H = members (positions X, tolerances theta), O_v = outside neighbours of v at their
fixed positions, Y0 = outside positions accepted at t = 0 by some member, K0 = conv(X_H u Y0).

  special   every outside neighbour is accepted at t = 0 by its member, and
            diam(X_H u Y) <= theta_min(H)
  refined   (a) diam(X_H u Y0) <= theta_min(H), and
            (b) dist(x_o, K0) > theta_v for every member v and o in O_v with x_o not in Y0

Both are sufficient for "no member ever refuses a neighbour it accepted at t = 0" under any
convex-combination dynamics with fixed outsiders (synchronous or not, any positive weights).

Numerics (pre-registered: relative margin 1e-6): a comparison counts only when it clears the
margin; otherwise the group is 'boundary' for that condition (neither true nor false) and is left
out of the theorem check. dist(x, K0) is bracketed: the upper bound is the distance to the point
of K0 found by NNLS on the convex-combination weights; the lower bound comes from the separating
direction through that point (u'x - max_p u'p), so (b) is certified true only when the lower
bound clears theta_v * (1 + margin) and certified false when the upper bound is below
theta_v * (1 - margin).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls

MARGIN = 1e-6
TRUE, FALSE, BOUNDARY = 1, 0, -1


def diameter(P: np.ndarray, block: int = 512, stop_above: float = np.inf) -> float:
    """Diameter of a point set; returns early (with a value > stop_above) once it exceeds stop_above."""
    P = np.asarray(P, np.float64)
    n = len(P)
    if n < 2:
        return 0.0
    sq = np.einsum("ij,ij->i", P, P)
    dmax = 0.0
    block = max(1, min(block, int(2e7 // max(n, 1))))  # at most ~160 MB per distance block
    for a in range(0, n, block):
        D2 = sq[a:a + block, None] + sq[None, :] - 2.0 * (P[a:a + block] @ P.T)
        dmax = max(dmax, float(np.sqrt(max(D2.max(), 0.0))))
        if dmax > stop_above:
            break
    return dmax


def dist_to_hull(x: np.ndarray, P: np.ndarray) -> tuple[float, float]:
    """(lower, upper) bounds on the Euclidean distance from x to conv(P)."""
    x = np.asarray(x, np.float64)
    P = np.asarray(P, np.float64)
    c = P.mean(0)
    scale = max(float(np.abs(P - c).max()), float(np.abs(x - c).max()), 1e-12)
    Pn, xn = (P - c) / scale, (x - c) / scale  # well-conditioned copy
    M = 1e3
    Aug = np.vstack([Pn.T, M * np.ones(len(P))])
    lam, _ = nnls(Aug, np.r_[xn, M], maxiter=50 * len(P) + 100)
    s = lam.sum()
    if s <= 0:
        lam = np.full(len(P), 1.0 / len(P))
    else:
        lam = lam / s
    y = lam @ Pn
    v = xn - y
    ub = float(np.linalg.norm(v))
    if ub <= 1e-15:
        return 0.0, 0.0
    u = v / ub
    lb = float(u @ xn - (Pn @ u).max())
    return max(lb, 0.0) * scale, ub * scale


def open_conditions(X: np.ndarray, theta: np.ndarray, optr: np.ndarray, oX: np.ndarray,
                    margin: float = MARGIN) -> dict:
    """Static conditions for one group. X [n x m], theta [n], outside neighbours of member i are
    oX[optr[i]:optr[i+1]] (the same lists the dynamics use)."""
    X = np.asarray(X, np.float64)
    th = np.asarray(theta, np.float64)
    n = len(th)
    out = {"n_outside_links": int(optr[-1]) if n else 0}
    tmin = float(th.min()) if n else 0.0
    # acceptance of each outside link at t = 0, with the margin
    owner = np.repeat(np.arange(n), np.diff(optr))
    d = np.linalg.norm(oX - X[owner], axis=1) if len(owner) else np.zeros(0)
    t_own = th[owner]
    acc = d <= t_own * (1 - margin)
    rej = d > t_own * (1 + margin)
    amb = ~(acc | rej)
    out["outside_links_accepted"] = int(acc.sum())
    out["outside_links_boundary"] = int(amb.sum())
    out["theta_min"] = tmin
    hi = tmin * (1 + margin)
    dX = diameter(X)
    out["diam_members"] = dX
    if dX > hi:  # X_H is part of K0: (a) fails already (most groups found without tolerances)
        out.update({"diam_K0": None, "cond_a": FALSE, "cond_b": None, "cond_refined": FALSE,
                    "cond_special": FALSE})
        return out
    Y0 = np.unique(oX[acc], axis=0) if acc.any() else np.zeros((0, X.shape[1]))
    P = np.vstack([X, Y0])
    dP = diameter(P, stop_above=hi)
    out["diam_K0"] = dP  # a lower bound of the diameter when it exceeds theta_min (early stop)
    a_ok = dP <= tmin * (1 - margin)
    a_bad = dP > tmin * (1 + margin)
    a = TRUE if a_ok else (FALSE if a_bad else BOUNDARY)
    if amb.any() and a != FALSE:
        a = BOUNDARY  # Y0 itself is uncertain
    out["cond_a"] = a
    # special case: every outside link accepted (no boundary links) and (a)
    if rej.any():
        out["cond_special"] = FALSE
    elif amb.any():
        out["cond_special"] = BOUNDARY
    else:
        out["cond_special"] = a
    # (b): only needed when (a) is not false
    if a == FALSE:
        out["cond_b"] = None
        out["cond_refined"] = FALSE
        return out
    if len(Y0):
        key = {tuple(r) for r in Y0}
        in_y0 = np.array([tuple(r) in key for r in oX], bool) if len(oX) else np.zeros(0, bool)
    else:
        in_y0 = np.zeros(len(oX), bool)
    # clearly rejected links whose position is not in Y0; boundary links make Y0 (hence K0) uncertain:
    # a larger K0 only shrinks distances, so a violation found with the clearly accepted Y0 stands,
    # but 'holds' cannot be certified
    todo = np.flatnonzero(rej & ~in_y0)
    b = TRUE
    c = P.mean(0)
    r = float(np.linalg.norm(P - c, axis=1).max())
    n_exact = 0
    for i in todo:
        t = t_own[i] * (1 + margin)
        if np.linalg.norm(oX[i] - c) - r > t:  # ball test: far from every point of K0
            continue
        lb, ub = dist_to_hull(oX[i], P)
        n_exact += 1
        if lb > t:
            continue
        if ub <= t_own[i] * (1 - margin):
            b = FALSE
            break
        b = BOUNDARY
    if b == TRUE and amb.any():
        b = BOUNDARY
    out["cond_b_exact_checks"] = n_exact
    out["cond_b"] = b
    out["cond_refined"] = FALSE if (b == FALSE) else (TRUE if (a == TRUE and b == TRUE) else BOUNDARY)
    return out
