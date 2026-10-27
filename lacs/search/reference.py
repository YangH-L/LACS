"""Pure-Python reference implementations used to test the C++ engine.

These are deliberately simple (and slow). brute_force() enumerates every
subset of the candidate set and is only meant for tiny graphs (<= ~18
candidates after filtering).
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np


def adjacency_sets(n: int, edges: Sequence[tuple[int, int]]) -> list[set[int]]:
    adj = [set() for _ in range(n)]
    for u, v in edges:
        if u != v:
            adj[u].add(v)
            adj[v].add(u)
    return adj


def kcore_component(adj: list[set[int]], alive: set[int], k: int, q: int) -> set[int]:
    alive = set(alive)
    changed = True
    while changed:
        changed = False
        for v in list(alive):
            if len(adj[v] & alive) < k:
                alive.discard(v)
                changed = True
    if q not in alive:
        return set()
    comp, stack = {q}, [q]
    while stack:
        v = stack.pop()
        for u in adj[v] & alive:
            if u not in comp:
                comp.add(u)
                stack.append(u)
    return comp


def dist(pos: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(pos[a].astype(np.float64) - pos[b].astype(np.float64)))


def diameter(pos: np.ndarray, H: set[int]) -> float:
    H = list(H)
    return max((dist(pos, a, b) for a, b in combinations(H, 2)), default=0.0)


def is_kcore_component(adj: list[set[int]], H: set[int], k: int, q: int) -> bool:
    if q not in H:
        return False
    if any(len(adj[v] & H) < k for v in H):
        return False
    seen, stack = {q}, [q]
    while stack:
        v = stack.pop()
        for u in adj[v] & H:
            if u not in seen:
                seen.add(u)
                stack.append(u)
    return seen == H


def compatible(pos, theta, a, b, mode: str, r: float | None) -> bool:
    d = dist(pos, a, b)
    if mode == "kr":
        return d <= r
    if mode == "pairwise":
        return d <= min(theta[a], theta[b])
    raise ValueError(mode)


def feasible(adj, pos, theta, H: set[int], k: int, q: int, mode: str = "diam", r: float | None = None,
             tol: float = 1e-6) -> bool:
    if not H or not is_kcore_component(adj, H, k, q):
        return False
    if mode == "diam":
        return diameter(pos, H) <= min(theta[v] for v in H) + tol
    return all(compatible(pos, theta, a, b, mode, r) or
               # numeric tolerance for float32 positions
               dist(pos, a, b) <= (r if mode == "kr" else min(theta[a], theta[b])) + tol
               for a, b in combinations(H, 2))


def brute_force(adj, pos, theta, k: int, q: int, mode: str = "diam", r: float | None = None,
                max_cand: int = 20) -> tuple[int, list[set[int]]]:
    """Maximum feasible size and all optimal sets."""
    n = len(adj)
    s0 = kcore_component(adj, set(range(n)), k, q)
    if not s0:
        return 0, []
    cand = sorted(v for v in s0 if v != q)
    if mode == "diam":
        cand = [v for v in cand if dist(pos, q, v) <= min(theta[q], theta[v])]
    elif mode == "pairwise":
        cand = [v for v in cand if dist(pos, q, v) <= min(theta[q], theta[v])]
    else:
        cand = [v for v in cand if dist(pos, q, v) <= r]
    if len(cand) > max_cand:
        raise ValueError(f"too many candidates for brute force: {len(cand)}")
    best, sets = 0, []
    for size in range(len(cand), -1, -1):
        if size + 1 < best:
            break
        for combo in combinations(cand, size):
            H = set(combo) | {q}
            if feasible(adj, pos, theta, H, k, q, mode, r, tol=0.0):
                if len(H) > best:
                    best, sets = len(H), [H]
                elif len(H) == best:
                    sets.append(H)
        if best == size + 1:
            break
    return best, sets


def ballcore(adj, pos, theta, k: int, q: int, half: bool = False) -> tuple[set[int], float]:
    """Reference sweep: max over tau of the k-core component of q among
    {v : scale*d(q,v) <= tau <= min(theta_v, theta_q)} (scale = 2 if half)."""
    n = len(adj)
    s0 = kcore_component(adj, set(range(n)), k, q)
    if not s0:
        return set(), float("nan")
    scale = 2.0 if half else 1.0
    iv = {}
    for v in s0:
        rr = min(theta[v], theta[q])
        ll = 0.0 if v == q else scale * dist(pos, q, v)
        if ll <= rr:
            iv[v] = (ll, rr)
    best, best_tau = set(), float("nan")
    for tau in sorted({rr for _, rr in iv.values()}):
        alive = {v for v, (ll, rr) in iv.items() if ll <= tau <= rr}
        comp = kcore_component(adj, alive, k, q)
        if len(comp) > len(best):
            best, best_tau = comp, tau
    return best, best_tau


def edgeacc(adj, pos, theta, k: int, q: int) -> set[int]:
    n = len(adj)
    s0 = kcore_component(adj, set(range(n)), k, q)
    if not s0:
        return set()
    fadj = [set() for _ in range(n)]
    for v in s0:
        for u in adj[v] & s0:
            if dist(pos, u, v) <= min(theta[u], theta[v]):
                fadj[v].add(u)
    return kcore_component(fadj, s0, k, q)
