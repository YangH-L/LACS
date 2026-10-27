"""Time-aware positions ("anchors") built from a user's earlier messages.

anchor(u, t) = recency-weighted mean of u's message embeddings with time < t
(weights exp(-rank / lambda), rank 0 = most recent, at most max_hist items),
shrunk towards a prior (forum mean) with strength kappa. Message times that
are only upper bounds make the "< t" test conservative, so no message written
at or after t can enter an anchor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class HistoryIndex:
    def __init__(self, comments: pd.DataFrame, emb: np.ndarray, valid: np.ndarray, lam: float = 3.0,
                 max_hist: int = 50, kappa: float = 0.5):
        self.emb = emb
        self.lam = float(lam)
        self.max_hist = int(max_hist)
        self.kappa = float(kappa)
        self.row_of = {c: i for i, c in enumerate(comments["cid"].astype(str))}
        df = pd.DataFrame({"author": comments["author"].astype(str).to_numpy(),
                           "time": comments["time"].to_numpy(dtype=float),
                           "row": np.arange(len(comments))})
        df = df[valid]
        df = df.sort_values(["author", "time"], kind="mergesort")
        self.hist: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for a, g in df.groupby("author", sort=False):
            self.hist[a] = (g["time"].to_numpy(), g["row"].to_numpy())
        self.weights = np.exp(-np.arange(self.max_hist + 1) / self.lam)

    def rows_before(self, author: str, t: float, inclusive: bool = False) -> np.ndarray:
        h = self.hist.get(author)
        if h is None:
            return np.empty(0, dtype=np.int64)
        times, rows = h
        k = np.searchsorted(times, t, side="right" if inclusive else "left")
        return rows[max(0, k - self.max_hist):k][::-1]  # most recent first

    def anchor(self, author: str, t: float, prior: np.ndarray, extra_row: int | None = None,
               exclude_row: int | None = None, inclusive: bool = False) -> tuple[np.ndarray, float]:
        rows = self.rows_before(author, t, inclusive)
        if exclude_row is not None and len(rows):
            rows = rows[rows != exclude_row]
        if extra_row is not None:
            rows = np.concatenate([[extra_row], rows])[: self.max_hist]
        if len(rows) == 0:
            return prior.copy(), 0.0
        w = self.weights[: len(rows)]
        mean = (w[:, None] * self.emb[rows]).sum(axis=0) / w.sum()
        n_eff = float(w.sum())
        vec = (n_eff * mean + self.kappa * prior) / (n_eff + self.kappa)
        return vec.astype(np.float32), float(len(rows))


def forum_priors(comments: pd.DataFrame, emb: np.ndarray, valid: np.ndarray, t_cut: float) -> dict[str, np.ndarray]:
    m = valid & (comments["time"].to_numpy() <= t_cut)
    glob = emb[m].mean(axis=0) if m.any() else emb.mean(axis=0)
    out = {"__global__": glob.astype(np.float32)}
    forums = comments["forum"].astype(str).to_numpy()
    for f in np.unique(forums):
        mf = m & (forums == f)
        out[f] = emb[mf].mean(axis=0).astype(np.float32) if mf.sum() >= 5 else out["__global__"]
    return out
