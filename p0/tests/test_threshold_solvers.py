"""Cross-check the per-threshold exact solvers against brute force.

Run: python tests/test_threshold_solvers.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p0kit.brute import brute_threshold  # noqa: E402
from p0kit.graphs import symmetric_csr  # noqa: E402
from p0kit.threshold_solvers import solve_ilp, solve_krmax, verify_solution  # noqa: E402


def random_instance(rng, n, m, p_edge, spread=1.0):
    X = rng.normal(scale=spread, size=(n, m))
    u, v = np.nonzero(np.triu(rng.random((n, n)) < p_edge, 1))
    ptr, idx = symmetric_csr(n, u, v)
    return ptr, idx, X


def check_solver(name, fn, trials=150, seed=7, nmax=14):
    rng = np.random.default_rng(seed)
    mismatches = 0
    for t in range(trials):
        n = int(rng.integers(3, nmax + 1))
        m = int(rng.integers(1, 4))
        ptr, idx, X = random_instance(rng, n, m, rng.uniform(0.25, 0.9))
        q = int(rng.integers(0, n))
        k = int(rng.integers(1, 4))
        D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
        tau = float(np.quantile(D[q], rng.uniform(0.3, 1.0))) + 1e-9
        opt, _ = brute_threshold(ptr, idx, X, tau, k, q)
        for lb in sorted({0, max(opt - 1, 0), opt}):
            res = fn(ptr, idx, X, tau, k, q, lb=lb, time_limit=30.0)
            if opt > lb:
                ok = res["status"] == "optimal_improved" and res["size"] == opt
                if ok:
                    reason = verify_solution(ptr, idx, X, tau, k, q, res["members"])
                    ok = reason is None
            else:
                ok = res["status"] == "optimal_none"
            if not ok:
                mismatches += 1
                print(f"[{name}] mismatch trial={t} n={n} k={k} lb={lb} opt={opt} got={res['status']},{res['size']}")
    assert mismatches == 0, f"{name}: {mismatches} mismatches"


def test_bnb_matches_brute_force():
    from p0kit.threshold_solvers import solve_bnb

    check_solver("bnb", lambda *a, **kw: solve_bnb(*a, branch_order=0, **kw), seed=5)
    check_solver("bnb_inc", lambda *a, **kw: solve_bnb(*a, branch_order=1, **kw), seed=6)


def test_krmax_matches_brute_force():
    check_solver("krmax", lambda *a, **kw: solve_krmax(*a, order_mode=0, **kw))


def test_krmax_simple_order_matches_brute_force():
    check_solver("krmax_simple", lambda *a, **kw: solve_krmax(*a, order_mode=1, **kw), seed=8)


def test_ilp_matches_brute_force():
    check_solver("ilp", solve_ilp, trials=80, seed=9, nmax=12)


def test_cpsat_matches_brute_force_if_available():
    try:
        import ortools  # noqa: F401
    except ImportError:
        print("skip cpsat (ortools not installed)")
        return
    from p0kit.threshold_solvers import solve_cpsat

    check_solver("cpsat", solve_cpsat, trials=60, seed=10, nmax=12)


def run_all():
    for name in [k for k in globals() if k.startswith("test_")]:
        globals()[name]()
        print("ok", name)


if __name__ == "__main__":
    run_all()
