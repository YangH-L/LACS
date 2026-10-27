"""Per-threshold exact solvers for the E3 baselines.

Each solver takes one fixed-threshold subproblem of LatSweep (Lemma 4): a
candidate set in local indices 0..n-1 with the query q, the structural CSR on
the set, positions X and the threshold tau, and returns a largest set H with
q in H, G[H] connected, min degree >= k and all pairwise distances <= tau,
of size strictly larger than lb, or reports that none exists.

Solvers
  krmax  : AdvMax of [35] (Zhang et al.), reimplemented in C++ (cpp/p0_krmax.cpp)
  ilp    : MILP with single-commodity flow connectivity, solved by HiGHS through
           scipy.optimize.milp (no extra dependency)
  cpsat  : the same model in OR-Tools CP-SAT (optional; pip install ortools)

All return a dict with keys: status in {optimal_improved, optimal_none,
timeout_improved, timeout_none, too_large, error}, members (local ids, sorted),
size, seconds, and solver-specific statistics.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from . import ffi
from .graphs import pairwise_sq


def conflict_pairs(X: np.ndarray, tau: float, rel_eps: float = 1e-9):
    X = np.asarray(X, dtype=np.float64)
    D2 = pairwise_sq(X)
    t2 = (tau * (1.0 + rel_eps)) ** 2
    iu, iv = np.nonzero(np.triu(D2 > t2, 1))
    return iu.astype(np.int64), iv.astype(np.int64), D2


def _similar_arcs(ptr, idx, D2, t2):
    n = len(ptr) - 1
    src = np.repeat(np.arange(n, dtype=np.int64), np.diff(ptr))
    dst = np.asarray(idx, dtype=np.int64)
    keep = (src != dst) & (D2[src, dst] <= t2)
    return src[keep], dst[keep]


def solve_krmax(ptr, idx, X, tau, k, q, lb=0, time_limit=60.0, order_mode=0, rel_eps=1e-9) -> dict:
    t0 = time.perf_counter()
    res = ffi.krmax_solve(ptr, idx, X, tau, k, q, lb=lb, time_limit=time_limit, order_mode=order_mode, rel_eps=rel_eps)
    res["wall_seconds"] = time.perf_counter() - t0
    res["solver"] = "krmax" if order_mode == 0 else "krmax_simple_order"
    return res


def solve_ilp(ptr, idx, X, tau, k, q, lb=0, time_limit=60.0, rel_eps=1e-9, max_conflicts=5_000_000) -> dict:
    """MILP solved by HiGHS (scipy.optimize.milp)."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    import scipy.sparse as sp

    t_start = time.perf_counter()
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    cu, cv, D2 = conflict_pairs(X, tau, rel_eps)
    t2 = (tau * (1.0 + rel_eps)) ** 2
    out = {"solver": "ilp_highs", "members": np.zeros(0, dtype=np.int32), "size": 0,
           "n_conflicts": int(len(cu))}
    if len(cu) > max_conflicts:
        out.update(status="too_large", seconds=time.perf_counter() - t_start)
        return out
    # candidates must be similar to q
    sim_q = D2[q] <= t2
    a_src, a_dst = _similar_arcs(ptr, idx, D2, t2)
    na = len(a_src)
    nx = n
    nv = nx + na  # variables: x (n), f (na)
    rows, cols, vals = [], [], []
    lo, hi = [], []
    r = 0
    # conflicts: x_u + x_v <= 1
    if len(cu):
        rr = np.arange(r, r + len(cu))
        rows += [rr, rr]
        cols += [cu, cv]
        vals += [np.ones(len(cu)), np.ones(len(cu))]
        lo.append(np.full(len(cu), -np.inf))
        hi.append(np.ones(len(cu)))
        r += len(cu)
    # degree: sum_{u in N'(v)} x_u - k x_v >= 0
    rr_deg = r + np.arange(n)
    rows += [r + a_src, rr_deg]
    cols += [a_dst, np.arange(n)]
    vals += [np.ones(na), np.full(n, -float(k))]
    lo.append(np.zeros(n))
    hi.append(np.full(n, np.inf))
    r += n
    big = float(max(n - 1, 1))
    if na:
        # capacity: f_e - big x_src <= 0 ; f_e - big x_dst <= 0
        e = np.arange(na)
        rows += [r + e, r + e, r + na + e, r + na + e]
        cols += [nx + e, a_src, nx + e, a_dst]
        vals += [np.ones(na), np.full(na, -big), np.ones(na), np.full(na, -big)]
        lo.append(np.full(2 * na, -np.inf))
        hi.append(np.zeros(2 * na))
        r += 2 * na
    # conservation for v != q: in - out - x_v = 0
    others = np.array([v for v in range(n) if v != q], dtype=np.int64)
    row_of = np.full(n, -1, dtype=np.int64)
    row_of[others] = r + np.arange(len(others))
    if na:
        e = np.arange(na)
        m_in = row_of[a_dst] >= 0
        m_out = row_of[a_src] >= 0
        rows += [row_of[a_dst][m_in], row_of[a_src][m_out]]
        cols += [nx + e[m_in], nx + e[m_out]]
        vals += [np.ones(int(m_in.sum())), -np.ones(int(m_out.sum()))]
    rows.append(row_of[others])
    cols.append(others)
    vals.append(-np.ones(len(others)))
    lo.append(np.zeros(len(others)))
    hi.append(np.zeros(len(others)))
    r += len(others)
    # improvement: sum x >= lb + 1
    rows.append(np.full(n, r))
    cols.append(np.arange(n))
    vals.append(np.ones(n))
    lo.append(np.array([lb + 1.0]))
    hi.append(np.array([np.inf]))
    r += 1
    A = sp.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, nv)
    )
    c = np.concatenate([-np.ones(nx), np.zeros(na)])
    lb_v = np.zeros(nv)
    ub_v = np.concatenate([np.where(sim_q, 1.0, 0.0), np.full(na, big)])
    lb_v[q] = 1.0
    ub_v[q] = 1.0
    integrality = np.concatenate([np.ones(nx), np.zeros(na)])
    t_build = time.perf_counter() - t_start
    remaining = max(time_limit - t_build, 1e-3)
    res = milp(
        c,
        integrality=integrality,
        bounds=Bounds(lb_v, ub_v),
        constraints=[LinearConstraint(A, np.concatenate(lo), np.concatenate(hi))],
        options={"time_limit": remaining, "disp": False, "mip_rel_gap": 0.0},
    )
    secs = time.perf_counter() - t_start
    out.update(seconds=secs, build_seconds=t_build, n_arcs=int(na), n_rows=int(r), milp_status=int(res.status),
               message=str(res.message))
    sol = None if res.x is None else np.flatnonzero(res.x[:nx] > 0.5).astype(np.int32)
    if res.status == 0 and sol is not None:
        out.update(status="optimal_improved", members=np.sort(sol), size=int(len(sol)))
    elif res.status == 2:
        out.update(status="optimal_none")
    elif res.status == 1:
        if sol is not None and len(sol) > lb:
            out.update(status="timeout_improved", members=np.sort(sol), size=int(len(sol)))
        else:
            out.update(status="timeout_none")
    else:
        out.update(status="error")
    return out


def _cp(obj, camel: str, snake: str):
    f = getattr(obj, snake, None)
    return f if f is not None else getattr(obj, camel)


def solve_cpsat(ptr, idx, X, tau, k, q, lb=0, time_limit=60.0, rel_eps=1e-9, workers=1,
                max_conflicts=5_000_000) -> dict:
    """Same model in OR-Tools CP-SAT. num_workers defaults to 1 (single thread)."""
    try:
        from ortools.sat.python import cp_model
    except ImportError:  # pragma: no cover - optional dependency
        return {"solver": "cpsat", "status": "error", "message": "ortools not installed",
                "members": np.zeros(0, dtype=np.int32), "size": 0, "seconds": 0.0}
    t_start = time.perf_counter()
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    cu, cv, D2 = conflict_pairs(X, tau, rel_eps)
    t2 = (tau * (1.0 + rel_eps)) ** 2
    out = {"solver": "cpsat", "members": np.zeros(0, dtype=np.int32), "size": 0,
           "n_conflicts": int(len(cu))}
    if len(cu) > max_conflicts:
        out.update(status="too_large", seconds=time.perf_counter() - t_start)
        return out
    a_src, a_dst = _similar_arcs(ptr, idx, D2, t2)
    model = cp_model.CpModel()
    new_bool = _cp(model, "NewBoolVar", "new_bool_var")
    new_int = _cp(model, "NewIntVar", "new_int_var")
    add = _cp(model, "Add", "add")
    add_bool_or = _cp(model, "AddBoolOr", "add_bool_or")
    x = [new_bool(f"x{v}") for v in range(n)]
    add(x[q] == 1)
    for v in range(n):
        if v != q and D2[q, v] > t2:
            add(x[v] == 0)
    for a, b in zip(cu.tolist(), cv.tolist()):
        add_bool_or([_cp(x[a], "Not", "negated")(), _cp(x[b], "Not", "negated")()])
    nbrs = [[] for _ in range(n)]
    for a, b in zip(a_src.tolist(), a_dst.tolist()):
        nbrs[a].append(b)
    for v in range(n):
        add(sum(x[u] for u in nbrs[v]) >= k * x[v])
    big = max(n - 1, 1)
    f = [new_int(0, big, f"f{e}") for e in range(len(a_src))]
    inflow = [[] for _ in range(n)]
    outflow = [[] for _ in range(n)]
    for e, (a, b) in enumerate(zip(a_src.tolist(), a_dst.tolist())):
        add(f[e] <= big * x[a])
        add(f[e] <= big * x[b])
        outflow[a].append(f[e])
        inflow[b].append(f[e])
    for v in range(n):
        if v == q:
            continue
        add(sum(inflow[v]) - sum(outflow[v]) == x[v])
    add(sum(x) >= lb + 1)
    _cp(model, "Maximize", "maximize")(sum(x))
    solver = cp_model.CpSolver()
    t_build = time.perf_counter() - t_start
    solver.parameters.max_time_in_seconds = max(time_limit - t_build, 1e-3)
    solver.parameters.num_workers = int(workers)
    st = _cp(solver, "Solve", "solve")(model)
    secs = time.perf_counter() - t_start
    out.update(seconds=secs, build_seconds=t_build, n_arcs=int(len(a_src)))
    value = _cp(solver, "Value", "value")
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sol = np.array([v for v in range(n) if value(x[v])], dtype=np.int32)
        if st == cp_model.OPTIMAL:
            out.update(status="optimal_improved", members=sol, size=int(len(sol)))
        else:
            out.update(status="timeout_improved", members=sol, size=int(len(sol)))
    elif st == cp_model.INFEASIBLE:
        out.update(status="optimal_none")
    elif st == cp_model.UNKNOWN:
        out.update(status="timeout_none")
    else:
        out.update(status="error", message=str(st))
    return out


def solve_bnb(ptr, idx, X, tau, k, q, lb=0, time_limit=60.0, branch_order=0, rel_eps=1e-9) -> dict:
    """Reference copy of the paper's Lemma 5 branch and bound (cpp/p0_bnb.cpp)."""
    t0 = time.perf_counter()
    res = ffi.bnb_solve(ptr, idx, X, tau, k, q, lb=lb, time_limit=time_limit, branch_order=branch_order, rel_eps=rel_eps)
    res["wall_seconds"] = time.perf_counter() - t0
    res["solver"] = "bnb" if branch_order == 0 else "bnb_include_first"
    return res


SOLVERS = {
    "bnb": lambda *a, **kw: solve_bnb(*a, branch_order=0, **kw),
    "bnb_include_first": lambda *a, **kw: solve_bnb(*a, branch_order=1, **kw),
    "krmax": lambda *a, **kw: solve_krmax(*a, order_mode=0, **kw),
    "krmax_simple": lambda *a, **kw: solve_krmax(*a, order_mode=1, **kw),
    "ilp": solve_ilp,
    "cpsat": solve_cpsat,
}


def verify_solution(ptr, idx, X, tau, k, q, members, rel_eps=1e-9) -> Optional[str]:
    """Returns None if `members` is feasible for the subproblem, else a reason."""
    members = np.asarray(members, dtype=np.int64)
    if len(members) == 0:
        return "empty"
    if q not in set(members.tolist()):
        return "q missing"
    X = np.asarray(X, dtype=np.float64)
    sub = X[members]
    D2 = pairwise_sq(sub)
    if D2.max() > (tau * (1 + rel_eps)) ** 2:
        return "diameter"
    Sset = set(members.tolist())
    adj = {int(v): [int(u) for u in idx[ptr[v]:ptr[v + 1]] if int(u) in Sset] for v in members}
    if min(len(a) for a in adj.values()) < k:
        return "degree"
    seen = {int(q)}
    stack = [int(q)]
    while stack:
        v = stack.pop()
        for u in adj[v]:
            if u not in seen:
                seen.add(u)
                stack.append(u)
    if len(seen) != len(Sset):
        return "disconnected"
    return None
