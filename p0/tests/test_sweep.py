"""The stand-alone LatSweep harness against brute-force MaxLCC.

Run: python tests/test_sweep.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p0kit import sweep  # noqa: E402
from p0kit.brute import brute_maxlcc  # noqa: E402
from p0kit.graphs import group_stats, symmetric_csr  # noqa: E402


def _is_lcc(index, members, q, k):
    members = np.asarray(members, dtype=np.int64)
    if len(members) == 0:
        return True
    if q not in set(members.tolist()):
        return False
    st = group_stats(index.X[members], index.theta[members])
    if not st["sat_iii"]:
        return False
    S = set(members.tolist())
    adj = {int(v): [int(u) for u in index.idx[index.ptr[v]:index.ptr[v + 1]] if int(u) in S] for v in members}
    if min(len(a) for a in adj.values()) < k:
        return False
    seen, stack = {int(q)}, [int(q)]
    while stack:
        v = stack.pop()
        for u in adj[v]:
            if u not in seen:
                seen.add(u)
                stack.append(u)
    return len(seen) == len(S)


def _instance(rng, n, m):
    X = rng.normal(size=(n, m))
    th = np.exp(rng.normal(np.log(1.5), 0.4, size=n))
    u, v = np.nonzero(np.triu(rng.random((n, n)) < rng.uniform(0.3, 0.8), 1))
    ptr, idx = symmetric_csr(n, u, v)
    return ptr, idx, X, th


def test_sweep_exact_matches_brute_force():
    rng = np.random.default_rng(11)
    checked = 0
    for t in range(120):
        n = int(rng.integers(4, 13))
        m = int(rng.integers(1, 4))
        ptr, idx, X, th = _instance(rng, n, m)
        index = sweep.SweepIndex(ptr, idx, X, th)
        q = int(rng.integers(0, n))
        k = int(rng.integers(1, 4))
        opt, _ = brute_maxlcc(ptr, idx, X, th, k, q)
        for solver in ("bnb", "bnb_include_first", "krmax", "krmax_simple", "ilp"):
            res = sweep.exact(index, q, k, solver=solver, time_limit=30)
            assert res["proven"], (t, solver, res["status"])
            assert res["size"] == opt, (t, solver, res["size"], opt)
            assert _is_lcc(index, res["members"], q, k), (t, solver)
        f = sweep.ballcore_f(index, q, k)
        g = sweep.greedy(index, q, k)
        r = sweep.ballcore_r(index, q, k)
        assert _is_lcc(index, f, q, k) and len(f) <= opt
        assert _is_lcc(index, g, q, k) and len(g) <= opt
        assert len(r) >= opt
        if len(r):
            st = group_stats(index.X[r], index.theta[r])
            assert st["diam"] <= 2 * st["theta_min"] * (1 + 1e-9)
        checked += 1
    assert checked == 120


def _peel_reference(n, adj, alive, q, k):
    alive = set(i for i in range(n) if alive[i])
    changed = True
    while changed:
        changed = False
        for v in list(alive):
            if sum(1 for u in adj[v] if u in alive) < k:
                alive.discard(v)
                changed = True
    if q not in alive:
        return set()
    seen, stack = {q}, [q]
    while stack:
        v = stack.pop()
        for u in adj[v]:
            if u in alive and u not in seen:
                seen.add(u)
                stack.append(u)
    return seen


def test_peel_component_matches_reference():
    """Regression: removing vertices while still counting degrees over-peels."""
    from p0kit import ffi

    rng = np.random.default_rng(13)
    for t in range(400):
        n = int(rng.integers(3, 60))
        u, v = np.nonzero(np.triu(rng.random((n, n)) < rng.uniform(0.05, 0.4), 1))
        ptr, idx = symmetric_csr(n, u, v)
        adj = [set(idx[ptr[i]:ptr[i + 1]].tolist()) for i in range(n)]
        alive = rng.random(n) < rng.uniform(0.5, 1.0)
        q = int(rng.integers(0, n))
        alive[q] = True
        k = int(rng.integers(1, 5))
        got = set(np.flatnonzero(ffi.peel_component(ptr, idx, alive.astype(np.uint8), q, k)).tolist())
        assert got == _peel_reference(n, adj, alive, q, k), (t, n, k)


def run_all():
    for name in [k for k in globals() if k.startswith("test_")]:
        globals()[name]()
        print("ok", name)


if __name__ == "__main__":
    run_all()
