"""Opinion dynamics used to check the propagation guarantees of a community.

hk():  heterogeneous Hegselmann-Krause dynamics on a graph. Agent v moves to the
       mean of itself and its graph neighbours whose position lies within its own
       latitude of acceptance theta_v. Theorem 1: if the community is connected and
       its diameter <= min theta, every interaction stays accepted and the group
       reaches consensus.
sjt():  Social-Judgment dynamics: assimilation inside theta_acc, no effect in the
       noncommitment band, repulsion beyond theta_rej. Theorem 2: if the diameter
       <= min theta_rej, no repulsive (rejection) interaction ever happens.

Each step reuses the CSR pattern of the graph (only the edge weights change),
and the run stops as soon as the bounding box of the focus agents shrinks
below the consensus threshold, the positions freeze, or (sjt) they diverge.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def _pairwise_d2(X: np.ndarray) -> np.ndarray:
    sq = (X * X).sum(1)
    return np.maximum(sq[:, None] + sq[None, :] - 2 * (X @ X.T), 0.0)


def _diam(X: np.ndarray) -> float:
    if len(X) < 2:
        return 0.0
    if len(X) > 3000:  # cheap upper bound for very large groups
        c = X.mean(0)
        return float(2 * np.linalg.norm(X - c, axis=1).max())
    return float(np.sqrt(_pairwise_d2(X).max()))


def _bbox(X: np.ndarray) -> float:
    return float(np.linalg.norm(X.max(0) - X.min(0))) if len(X) else 0.0


def _clusters(X: np.ndarray, eps: float) -> int:
    n = len(X)
    if n <= 1:
        return n
    if n > 3000:
        return -1
    M = sp.csr_matrix(_pairwise_d2(X) <= eps * eps)
    return int(connected_components(M, directed=False)[0])


def _pattern(A: sp.csr_matrix):
    A = sp.csr_matrix(A)
    A.sort_indices()
    rows = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
    return A.indptr, A.indices, rows


def hk(A: sp.csr_matrix, X0: np.ndarray, theta: np.ndarray, max_steps: int = 2000, tol: float = 1e-7,
       consensus_eps: float = 1e-3, focus: np.ndarray | None = None) -> dict:
    """focus: optional mask of the agents whose outcome is reported (e.g. the
    community members when their outside neighbours are simulated too)."""
    X = np.asarray(X0, dtype=np.float64).copy()
    n = len(X)
    focus = np.ones(n, bool) if focus is None else np.asarray(focus, bool)
    theta = np.asarray(theta, dtype=np.float64)
    d0 = _diam(X[focus])
    scale = max(d0, 1e-12)
    indptr, cols, rows = _pattern(A)
    in_focus = focus[rows] & focus[cols]
    rejected_any = 0
    steps = 0
    for steps in range(1, max_steps + 1):
        dist = np.sqrt(((X[rows] - X[cols]) ** 2).sum(1))
        keep = (dist <= theta[rows] + 1e-12).astype(float)
        rejected_any += int(((keep == 0) & in_focus).sum())
        W = sp.csr_matrix((keep, cols, indptr), shape=(n, n))
        den = np.asarray(W.sum(axis=1)).ravel() + 1.0
        Xn = (W @ X + X) / den[:, None]
        change = float(np.abs(Xn - X).max()) if n else 0.0
        X = Xn
        if change < tol or _bbox(X[focus]) <= consensus_eps * scale:
            break
    d1 = _diam(X[focus])
    return {"steps": steps, "diam0": d0, "diam_final": d1, "consensus": bool(d1 <= consensus_eps * scale),
            "n_clusters": _clusters(X[focus], consensus_eps * scale), "edge_rejections_seen": rejected_any}


def sjt(A: sp.csr_matrix, X0: np.ndarray, theta_acc: np.ndarray, theta_rej: np.ndarray, mu: float = 0.3,
        beta: float = 0.3, max_steps: int = 500, tol: float = 1e-7, focus: np.ndarray | None = None,
        diverge_ratio: float = 1e3) -> dict:
    X = np.asarray(X0, dtype=np.float64).copy()
    n = len(X)
    focus = np.ones(n, bool) if focus is None else np.asarray(focus, bool)
    theta_acc = np.asarray(theta_acc, dtype=np.float64)
    theta_rej = np.asarray(theta_rej, dtype=np.float64)
    d0 = _diam(X[focus])
    scale = max(d0, 1e-12)
    indptr, cols, rows = _pattern(A)
    deg = np.maximum(np.diff(indptr), 1).astype(float)
    in_focus = focus[rows] & focus[cols]
    rejections = 0
    max_ratio = 1.0
    steps = 0
    for steps in range(1, max_steps + 1):
        dist = np.sqrt(((X[rows] - X[cols]) ** 2).sum(1))
        acc = dist <= theta_acc[rows] + 1e-12
        rej = dist >= theta_rej[rows] - 1e-12
        rejections += int((rej & in_focus).sum())
        coef = mu * acc - beta * rej
        W = sp.csr_matrix((coef, cols, indptr), shape=(n, n))
        wsum = np.asarray(W.sum(axis=1)).ravel()
        Xn = X + (W @ X - wsum[:, None] * X) / deg[:, None]
        change = float(np.abs(Xn - X).max()) if n else 0.0
        X = Xn
        ratio = _bbox(X[focus]) / scale
        max_ratio = max(max_ratio, ratio)
        if change < tol or ratio > diverge_ratio:
            break
    return {"steps": steps, "rejection_events": rejections, "diam_final_ratio": _diam(X[focus]) / scale,
            "max_bbox_ratio": max_ratio}
