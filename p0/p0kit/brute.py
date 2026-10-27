"""Brute-force reference solvers for tests (small instances only)."""

from __future__ import annotations

from itertools import combinations
from typing import Optional, Tuple

import numpy as np


def _adj_lists(ptr, idx):
    n = len(ptr) - 1
    return [set(int(u) for u in idx[ptr[v]:ptr[v + 1]]) for v in range(n)]


def _connected(nodes, adj):
    nodes = set(nodes)
    start = next(iter(nodes))
    seen = {start}
    stack = [start]
    while stack:
        v = stack.pop()
        for u in adj[v]:
            if u in nodes and u not in seen:
                seen.add(u)
                stack.append(u)
    return len(seen) == len(nodes)


def _feasible(S, adj, k):
    if len(S) == 1:
        return k <= 0
    Sset = set(S)
    for v in S:
        if len(adj[v] & Sset) < k:
            return False
    return _connected(S, adj)


def brute_threshold(ptr, idx, X, tau: float, k: int, q: int, rel_eps: float = 1e-9) -> Tuple[int, Optional[tuple]]:
    """Largest H containing q, connected, min degree >= k, pairwise distance <= tau."""
    n = len(ptr) - 1
    if n > 20:
        raise ValueError("too large for brute force")
    adj = _adj_lists(ptr, idx)
    X = np.asarray(X, dtype=np.float64)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    t = tau * (1 + rel_eps)
    others = [v for v in range(n) if v != q and D[q, v] <= t]
    for size in range(len(others) + 1, 0, -1):
        for comb in combinations(others, size - 1):
            S = (q,) + comb
            if size > 1:
                sub = D[np.ix_(S, S)]
                if sub.max() > t:
                    continue
            if _feasible(S, adj, k):
                return size, tuple(sorted(S))
    return 0, None


def brute_pairwise(ptr, idx, X, theta, k: int, q: int, rel_eps: float = 1e-9) -> Tuple[int, Optional[tuple]]:
    """Largest connected H with q, min degree >= k and d(u,v) <= min(theta_u, theta_v)."""
    n = len(ptr) - 1
    if n > 18:
        raise ValueError("too large for brute force")
    adj = _adj_lists(ptr, idx)
    X = np.asarray(X, dtype=np.float64)
    th = np.asarray(theta, dtype=np.float64)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    L = np.minimum(th[:, None], th[None, :]) * (1 + rel_eps)
    others = [v for v in range(n) if v != q and D[q, v] <= L[q, v]]
    for size in range(len(others) + 1, 1, -1):
        for comb in combinations(others, size - 1):
            S = (q,) + comb
            if np.any(D[np.ix_(S, S)] > L[np.ix_(S, S)]):
                continue
            if _feasible(S, adj, k):
                return size, tuple(sorted(S))
    return 0, None


def brute_maxlcc(ptr, idx, X, theta, k: int, q: int, rel_eps: float = 1e-9) -> Tuple[int, Optional[tuple]]:
    """Largest LCC containing q (Definition 1) by enumeration."""
    n = len(ptr) - 1
    if n > 18:
        raise ValueError("too large for brute force")
    adj = _adj_lists(ptr, idx)
    X = np.asarray(X, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    others = [v for v in range(n) if v != q]
    for size in range(n, 1, -1):
        for comb in combinations(others, size - 1):
            S = (q,) + comb
            sub = D[np.ix_(S, S)]
            if sub.max() > theta[list(S)].min() * (1 + rel_eps):
                continue
            if _feasible(S, adj, k):
                return size, tuple(sorted(S))
    return 0, None
