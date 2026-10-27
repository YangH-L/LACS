"""Evaluation metrics for ordinal acceptance prediction + cluster bootstrap."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def nll(P: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1.0))


def ece(P: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def evaluate(P: np.ndarray, y: np.ndarray) -> dict:
    K = P.shape[1]
    pred = P.argmax(axis=1)
    out = {
        "n": int(len(y)),
        "nll": float(nll(P, y).mean()),
        "acc": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(K)), zero_division=0)),
        "ece": ece(P, y),
        "brier": float(((P - np.eye(K)[y]) ** 2).sum(axis=1).mean()),
    }
    if K == 3:
        m = y != 1
        if len(np.unique(y[m])) == 2:
            out["auc_reject_vs_accept"] = float(roc_auc_score((y[m] == 2).astype(int), P[m, 2] - P[m, 0]))
        out["ordinal_mae"] = float(np.abs(pred - y).mean())
        out["expected_class_mae"] = float(np.abs(P @ np.arange(3) - y).mean())
    else:
        if len(np.unique(y)) == 2:
            out["auc_accept"] = float(roc_auc_score((y == 0).astype(int), P[:, 0]))
    return out


def cluster_bootstrap_diff(loss_a: np.ndarray, loss_b: np.ndarray, clusters: np.ndarray, n_boot: int = 2000,
                           seed: int = 0) -> dict:
    """Mean(loss_a - loss_b) with a bootstrap CI that resamples clusters
    (e.g. receivers), because judgements by the same person are dependent."""
    diff = np.asarray(loss_a) - np.asarray(loss_b)
    uniq, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=diff, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(uniq), size=len(uniq))
        boots[b] = sums[idx].sum() / max(cnts[idx].sum(), 1.0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_two = 2 * min((boots >= 0).mean(), (boots <= 0).mean())
    return {"mean_diff": float(diff.mean()), "ci95": [float(lo), float(hi)], "p_boot": float(min(p_two, 1.0)),
            "n_clusters": int(len(uniq))}
