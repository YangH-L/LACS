"""Numerical checks of the dynamics kernels against the paper's statements.

Run: python tests/test_dynamics_kernels.py   (or: python -m pytest tests)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p0kit import ffi  # noqa: E402
from p0kit.graphs import complete_csr, reverse_index, symmetric_csr  # noqa: E402


def _random_lcc(rng, n, m, k, theta_lo=1.0, theta_hi=3.0):
    """A connected group with min degree >= k and diam <= theta_min."""
    th = rng.uniform(theta_lo, theta_hi, size=n)
    tmin = th.min()
    # points in a ball of radius tmin/2 -> diameter <= tmin
    Z = rng.normal(size=(n, m))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    r = rng.uniform(0, 1, size=(n, 1)) ** (1.0 / m)
    X = Z * r * (tmin / 2.0) * 0.999
    # ring + random chords -> connected, min degree >= k (k <= n-1)
    edges = set()
    for i in range(n):
        for d in range(1, k // 2 + 2):
            edges.add(tuple(sorted((i, (i + d) % n))))
    for _ in range(n):
        a, b = rng.integers(0, n, size=2)
        if a != b:
            edges.add(tuple(sorted((int(a), int(b)))))
    u = np.array([e[0] for e in edges])
    v = np.array([e[1] for e in edges])
    ptr, idx = symmetric_csr(n, u, v)
    assert np.min(np.diff(ptr)) >= k
    return X, th, ptr, idx


def test_theorem1_hk_never_breaks_on_lcc():
    rng = np.random.default_rng(1)
    for trial in range(40):
        n = int(rng.integers(4, 30))
        m = int(rng.integers(1, 6))
        k = int(rng.integers(1, min(4, n - 1) + 1))
        X, th, ptr, idx = _random_lcc(rng, n, m, k)
        for graph in ("G", "K"):
            if graph == "K":
                p, ix = complete_csr(n)
            else:
                p, ix = ptr, idx
            for weights in ("equal", "random"):
                W = rng.gamma(1.0, size=(n, n)) if weights == "random" else None
                res = ffi.sim_sync(X, th, th, p, ix, W=W, model="hk", T_max=500)
                assert res["t0_refused"] == 0, res
                assert res["broken"] == 0, (trial, graph, weights, res)
                assert res["diam_max"] <= res["diam0"] * (1 + 1e-12) + 1e-15
                assert res["n_clusters"] == 1, res


def test_theorem2_sjt_never_rejects_on_lcc_rej():
    rng = np.random.default_rng(2)
    for trial in range(40):
        n = int(rng.integers(4, 25))
        m = int(rng.integers(1, 5))
        k = int(rng.integers(1, min(3, n - 1) + 1))
        X, th, ptr, idx = _random_lcc(rng, n, m, k)
        th_rej = th
        th_acc = 0.5 * th
        for graph in ("G", "K"):
            p, ix = (complete_csr(n) if graph == "K" else (ptr, idx))
            W = rng.gamma(1.0, size=(n, n))
            res = ffi.sim_sync(X, th_acc, th_rej, p, ix, W=W, model="sjt", beta=0.2, T_max=300)
            assert res["t0_rejected"] == 0 and res["new_rejected"] == 0, (trial, graph, res)
            assert res["diam_max"] <= res["diam0"] * (1 + 1e-12) + 1e-15


def _prop1b(k):
    ell = max(k, 3)
    # H0: 0..ell-1 at 0 (theta 2); u = ell, v = ell+1 at 1 (theta 1); H2 at 2 (theta 2)
    n = 2 * ell + 2
    uu, vv = ell, ell + 1
    E = []
    for i in range(ell):
        for j in range(i + 1, ell):
            E.append((i, j))
            E.append((ell + 2 + i, ell + 2 + j))
        E.append((i, uu))
        E.append((ell + 2 + i, vv))
    E.append((uu, vv))
    X = np.array([0.0] * ell + [1.0, 1.0] + [2.0] * ell)[:, None]
    th = np.array([2.0] * ell + [1.0, 1.0] + [2.0] * ell)
    u = np.array([e[0] for e in E])
    v = np.array([e[1] for e in E])
    ptr, idx = symmetric_csr(n, u, v)
    return X, th, ptr, idx


def test_prop1b_two_exempt_members_break_acceptance_and_consensus():
    for k in (1, 2, 3, 5):
        X, th, ptr, idx = _prop1b(k)
        res = ffi.sim_sync(X, th, th, ptr, idx, model="hk", T_max=2000)
        assert res["t0_refused"] == 0
        assert res["broken"] > 0 and res["first_break"] == 1, res
        assert res["n_clusters"] == 2, res
        sj = ffi.sim_sync(X, th, th, ptr, idx, model="sjt", beta=0.05, T_max=50)
        assert sj["new_rejected"] > 0, sj


def _single_exempt(k):
    ell = max(k, 3)
    A = list(range(ell))
    w, u = ell, ell + 1
    B = list(range(ell + 2, 2 * ell + 2))
    n = 2 * ell + 2
    E = []
    for i in A:
        E.append((i, w))
        for j in A:
            if i < j:
                E.append((i, j))
    E.append((w, u))
    for i in B:
        E.append((i, u))
        for j in B:
            if i < j:
                E.append((i, j))
    X = np.array([-1.0] * ell + [0.0, 1.0] + [1.0] * ell)[:, None]
    th = np.array([2.0] * ell + [1.0, 2.0] + [2.0] * ell)
    ptr, idx = symmetric_csr(n, np.array([e[0] for e in E]), np.array([e[1] for e in E]))
    return X, th, ptr, idx


def test_single_exemption_breaks_acceptance_but_not_consensus():
    for k in (1, 2, 3, 4):
        X, th, ptr, idx = _single_exempt(k)
        res = ffi.sim_sync(X, th, th, ptr, idx, model="hk", T_max=5000, move_tol_rel=1e-13)
        assert res["t0_refused"] == 0
        assert res["broken"] > 0 and res["first_break"] == 1, res
        assert res["n_clusters"] == 1, res
        sj = ffi.sim_sync(X, th, th, ptr, idx, model="sjt", beta=0.05, T_max=50)
        assert sj["new_rejected"] > 0, sj


def test_prop1a_epsilon_slack():
    k = 2
    n = 2 * k + 2
    theta, eps = 1.0, 0.05
    X = np.array([0.0] * (n // 2) + [theta + eps] * (n // 2))[:, None]
    th = np.full(n, theta)
    ptr, idx = complete_csr(n)
    res = ffi.sim_sync(X, th, th, ptr, idx, model="hk", T_max=100)
    assert res["t0_refused"] > 0 and res["n_clusters"] == 2
    sj = ffi.sim_sync(X, 0.5 * th, th, ptr, idx, model="sjt", beta=0.1, T_max=10)
    assert sj["t0_rejected"] > 0 and sj["diam_max"] > sj["diam0"], sj


def test_characterisation_necessity_adversarial_weights():
    # pairwise compatible, diam > theta_min; w moves onto u, then refuses v.
    X = np.array([[0.0], [-0.6], [0.6]])
    th = np.array([1.0, 2.0, 2.0])
    ptr, idx = complete_csr(3)
    W = np.ones((3, 3))
    W[0, 1] = 1e6
    W[1, 1] = 1e6
    W[2, 2] = 1e6
    res = ffi.sim_sync(X, th, th, ptr, idx, W=W, model="hk", T_max=5)
    assert res["t0_refused"] == 0
    assert res["broken"] > 0 and res["first_break"] == 1, res
    # equal weights on the same group do not break (consensus at once)
    res_eq = ffi.sim_sync(X, th, th, ptr, idx, model="hk", T_max=50)
    assert res_eq["broken"] == 0


def test_dw_gossip_never_breaks_on_lcc_and_breaks_on_prop1b():
    rng = np.random.default_rng(3)
    for trial in range(20):
        n = int(rng.integers(4, 20))
        X, th, ptr, idx = _random_lcc(rng, n, 3, 2)
        rev = reverse_index(ptr, idx)
        res = ffi.sim_dw(X, th, th, ptr, idx, rev, mu=0.5, rounds=200, seed=trial)
        assert res["broken"] == 0 and res["t0_refused"] == 0, res
    X, th, ptr, idx = _prop1b(3)
    rev = reverse_index(ptr, idx)
    hits = 0
    for s in range(20):
        r = ffi.sim_dw(X, th, th, ptr, idx, rev, mu=0.5, rounds=300, seed=s)
        hits += r["broken"] > 0
    assert hits > 0


def test_external_pull_can_break_lcc():
    # the closed-group assumption matters: a stubborn outside neighbour pulls one member away
    X = np.array([[0.0], [0.5], [1.0]])
    th = np.array([1.0, 1.0, 1.0])
    ptr, idx = complete_csr(3)
    ext = np.array([[-5.0], [0.0], [0.0]])
    mask = np.array([1, 0, 0], dtype=np.uint8)
    res = ffi.sim_sync(X, th, th, ptr, idx, ext_mean=ext, ext_mask=mask, lambda_ext=0.3, model="hk", T_max=200)
    assert res["broken"] > 0


def test_edge_weights_equal_dense_weights():
    rng = np.random.default_rng(5)
    for trial in range(10):
        n = int(rng.integers(4, 20))
        X, th, ptr, idx = _random_lcc(rng, n, 2, 2)
        X = X * 3.0  # not an LCC any more: events happen
        W = rng.gamma(0.5, size=(n, n))
        src = np.repeat(np.arange(n), np.diff(ptr))
        a = ffi.sim_sync(X, th, th, ptr, idx, W=W, model="hk", T_max=200)
        b = ffi.sim_sync(X, th, th, ptr, idx, W_self=np.diag(W).copy(), W_edge=W[src, idx], model="hk", T_max=200)
        assert a == b, (a, b)


def test_open_group_outsiders():
    # two members at 0 and 1 (theta 1): an LCC. Outsiders they accept pull them apart.
    X = np.array([[0.0], [1.0]])
    th = np.array([1.0, 1.0])
    p, ix = complete_csr(2)
    closed = ffi.sim_sync(X, th, th, p, ix, model="hk", T_max=200)
    assert closed["broken"] == 0 and closed["open_acc0"] == 0
    # outsiders out of reach: identical to the closed group
    optr = np.array([0, 3, 6])
    far = np.array([[-1.5]] * 3 + [[2.5]] * 3)
    res = ffi.sim_sync(X, th, th, p, ix, model="hk", T_max=200, open_ptr=optr, open_X=far)
    assert res == closed, (res, closed)
    # accepted outsiders (three each side): equilibrium distance 1.68 > 1, so the pair breaks
    near = np.array([[-0.9]] * 3 + [[1.9]] * 3)
    res = ffi.sim_sync(X, th, th, p, ix, model="hk", T_max=200, open_ptr=optr, open_X=near)
    assert res["open_acc0"] == 2 and res["broken"] > 0, res
    # one accepted outsider each side: equilibrium distance 0.93 < 1, no break
    optr1 = np.array([0, 1, 2])
    res = ffi.sim_sync(X, th, th, p, ix, model="hk", T_max=500, open_ptr=optr1, open_X=np.array([[-0.9], [1.9]]))
    assert res["open_acc0"] == 2 and res["broken"] == 0, res
    # follow-one also uses accepted outsiders
    hits = sum(ffi.sim_sync(X, th, th, p, ix, model="follow", gamma=0.9, T_max=50, seed=s, open_ptr=optr,
                            open_X=near)["broken"] > 0 for s in range(20))
    assert hits > 0


def test_follow_one_never_breaks_on_lcc_but_can_break_pairwise():
    rng = np.random.default_rng(4)
    for trial in range(30):
        n = int(rng.integers(4, 25))
        X, th, ptr, idx = _random_lcc(rng, n, 3, 2)
        for graph in ("G", "K"):
            p, ix = (complete_csr(n) if graph == "K" else (ptr, idx))
            res = ffi.sim_sync(X, th, th, p, ix, model="follow", gamma=0.9, T_max=300, seed=trial)
            assert res["broken"] == 0 and res["t0_refused"] == 0, res
    # pairwise-compatible group with diam > theta_min (characterisation example)
    X = np.array([[0.0], [-0.6], [0.6]])
    th = np.array([1.0, 2.0, 2.0])
    p, ix = complete_csr(3)
    hits = sum(ffi.sim_sync(X, th, th, p, ix, model="follow", gamma=0.95, T_max=20, seed=s)["broken"] > 0
               for s in range(50))
    assert hits > 0


def run_all():
    names = [k for k in globals() if k.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)


if __name__ == "__main__":
    run_all()
