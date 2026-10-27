"""Stand-alone LatSweep harness (Algorithm 1 of the paper) with pluggable
per-threshold solvers, used by E3 so that the paper's branch and bound, the
reimplemented maximum (k,r)-core of [35], the MILP (HiGHS) and CP-SAT all run on
exactly the same query regions, thresholds, peeling and incumbents.

Semantics follow Sections 5.1-5.3:
  * survival interval I_v = [sigma * rho_v, min(theta_v, theta_q)], rho_v = d(x_v, x_q);
  * query region: traversal from q along G' visiting vertices with a non-empty
    interval and core number (in G') at least k;
  * candidate thresholds T_q = {min(theta_v, theta_q) : v in R}, processed in
    non-increasing |S(tau)|, stopping once |S(tau)| <= |B|;
  * C(tau) = C_k(S(tau), q) peeled in G';
  * BallCore-F: sigma = 2, G' = G_theta; Greedy: 20 thresholds with largest |S(tau)|,
    G' = G_theta; BallCore-R: sigma = 1, G' = G; Exact: sigma = 1, G' = G_theta,
    incumbent = larger of BallCore-F and Greedy, candidate cap and time limit as
    in the paper (answer marked unproven when either is hit).
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import numpy as np

from . import ffi
from .graphs import induced_csr
from .threshold_solvers import SOLVERS


class SweepIndex:
    """Per-dataset structures: G, positions, tolerances, G_theta and core numbers."""

    def __init__(self, ptr, idx, X, theta, rel_eps: float = 1e-9):
        t0 = time.perf_counter()
        self.ptr = np.ascontiguousarray(ptr, dtype=np.int64)
        self.idx = np.ascontiguousarray(idx, dtype=np.int32)
        self.X = np.ascontiguousarray(X, dtype=np.float64)
        self.theta = np.ascontiguousarray(theta, dtype=np.float64)
        self.n = len(self.ptr) - 1
        self.rel_eps = rel_eps
        mask = ffi.gtheta_mask(self.ptr, self.idx, self.X, self.theta, rel_eps)
        cum = np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])
        counts = cum[self.ptr[1:]] - cum[self.ptr[:-1]]
        self.tptr = np.zeros(self.n + 1, dtype=np.int64)
        self.tptr[1:] = np.cumsum(counts)
        self.tidx = np.ascontiguousarray(self.idx[mask.astype(bool)])
        self.kept_edge_frac = float(mask.mean()) if len(mask) else 0.0
        del mask, cum
        self.core_t = ffi.core_numbers(self.tptr, self.tidx)
        self.core_g = ffi.core_numbers(self.ptr, self.idx)
        self.build_seconds = time.perf_counter() - t0
        self._state = np.zeros(self.n, dtype=np.int8)

    # ------------------------------------------------------------------ region
    def region(self, q: int, k: int, sigma: float, graph: str, radius: Optional[float] = None,
               max_size: int = 0) -> np.ndarray:
        """Global ids of the query region, q first.

        Default: vertices with sigma * rho_v <= min(theta_v, theta_q) (Lemma 3).
        With `radius`, tolerances are ignored: sigma * rho_v <= radius (used by
        the tolerance-agnostic baselines). max_size > 0 stops the traversal
        once the region exceeds that many vertices (the result is then
        truncated and flagged by a size of max_size + 1 or more)."""
        if graph == "theta":
            ptr, idx, core = self.tptr, self.tidx, self.core_t
        else:
            ptr, idx, core = self.ptr, self.idx, self.core_g
        if core[q] < k:
            return np.zeros(0, dtype=np.int64)
        X, th, st = self.X, self.theta, self._state
        xq, tq = X[q], th[q]
        tol = 1.0 + self.rel_eps
        order = [int(q)]
        touched = [np.array([q], dtype=np.int64)]
        st[q] = 1
        head = 0
        while head < len(order):
            v = order[head]
            head += 1
            nb = np.asarray(idx[ptr[v]:ptr[v + 1]], dtype=np.int64)
            if len(nb) == 0:
                continue
            nb = nb[st[nb] == 0]
            if len(nb) == 0:
                continue
            ok = core[nb] >= k
            cand = nb[ok]
            bad = [nb[~ok]]
            if len(cand):
                d = np.sqrt(((X[cand] - xq) ** 2).sum(axis=1))
                if radius is None:
                    right = np.minimum(th[cand], tq)
                    good = sigma * d <= right * tol
                else:
                    good = sigma * d <= radius * tol
                acc = cand[good]
                bad.append(cand[~good])
                if len(acc):
                    st[acc] = 1
                    order.extend(acc.tolist())
                    touched.append(acc)
            for b in bad:
                if len(b):
                    st[b] = 2
                    touched.append(b)
            if max_size and len(order) > max_size:
                break
        for t in touched:
            st[t] = 0
        return np.asarray(order, dtype=np.int64)


class QueryContext:
    """Region-level data for one query and one (sigma, graph) choice."""

    def __init__(self, index: SweepIndex, q: int, k: int, sigma: float, graph: str):
        self.index, self.q, self.k, self.sigma, self.graph = index, q, k, sigma, graph
        t0 = time.perf_counter()
        R = index.region(q, k, sigma, graph)
        self.R = R
        self.nR = len(R)
        if self.nR == 0:
            self.left = self.right = self.T = self.cnt = np.zeros(0)
            self.order = np.zeros(0, dtype=np.int64)
            self.lptr = np.zeros(1, dtype=np.int64)
            self.lidx = np.zeros(0, dtype=np.int32)
            self.XR = np.zeros((0, index.X.shape[1]))
            self.seconds = time.perf_counter() - t0
            return
        ptr, idx = (index.tptr, index.tidx) if graph == "theta" else (index.ptr, index.idx)
        self.lptr, self.lidx = induced_csr(ptr, idx, R)
        self.XR = index.X[R]
        rho = np.sqrt(((self.XR - index.X[q]) ** 2).sum(axis=1))
        self.right = np.minimum(index.theta[R], index.theta[q])
        self.left = sigma * rho
        T = np.unique(self.right)
        ls = np.sort(self.left)
        rs = np.sort(self.right)
        tol = 1.0 + index.rel_eps
        cnt = np.searchsorted(ls, T * tol, side="right") - np.searchsorted(rs, T / tol, side="left")
        self.T = T
        self.cnt = cnt.astype(np.int64)
        self.order = np.lexsort((-T, -self.cnt))  # by count desc, then tau desc
        self.seconds = time.perf_counter() - t0

    def S_mask(self, tau: float) -> np.ndarray:
        tol = 1.0 + self.index.rel_eps
        return (self.left <= tau * tol) & (tau <= self.right * tol)

    def peel(self, mask: np.ndarray) -> np.ndarray:
        if not mask[0]:
            return np.zeros(self.nR, dtype=bool)
        return ffi.peel_component(self.lptr, self.lidx, mask.astype(np.uint8), 0, self.k)


def ballcore_f(index: SweepIndex, q: int, k: int) -> np.ndarray:
    ctx = QueryContext(index, q, k, 2.0, "theta")
    best = np.zeros(0, dtype=np.int64)
    for j in ctx.order:
        if ctx.cnt[j] <= len(best):
            break
        C = ctx.peel(ctx.S_mask(ctx.T[j]))
        if C.sum() > len(best):
            best = ctx.R[C]
    return best


def ballcore_r(index: SweepIndex, q: int, k: int) -> np.ndarray:
    ctx = QueryContext(index, q, k, 1.0, "G")
    best = np.zeros(0, dtype=np.int64)
    for j in ctx.order:
        if ctx.cnt[j] <= len(best):
            break
        C = ctx.peel(ctx.S_mask(ctx.T[j]))
        if C.sum() > len(best):
            best = ctx.R[C]
    return best


def greedy(index: SweepIndex, q: int, k: int, n_thresholds: int = 20, ctx: Optional[QueryContext] = None) -> np.ndarray:
    ctx = ctx or QueryContext(index, q, k, 1.0, "theta")
    best = np.zeros(0, dtype=np.int64)
    for j in ctx.order[:n_thresholds]:
        if ctx.cnt[j] <= len(best):
            continue
        tau = float(ctx.T[j])
        C = ctx.peel(ctx.S_mask(tau))
        if C.sum() <= len(best):
            continue
        G = ffi.greedy_conflict(ctx.lptr, ctx.lidx, ctx.XR, tau, k, 0, C.astype(np.uint8), index.rel_eps)
        if G.sum() > len(best):
            best = ctx.R[G]
    return best


def exact(
    index: SweepIndex,
    q: int,
    k: int,
    solver: str = "bnb",
    time_limit: float = 60.0,
    candidate_cap: int = 3000,
    incumbent: str = "paper",
    solver_kwargs: Optional[dict] = None,
) -> dict:
    """LatSweep-Exact with a pluggable per-threshold solver.

    Returns dict(members, size, proven, status, seconds, solver_seconds, n_thresholds,
    n_solver_calls, region_size, incumbent_size, max_candidates)."""
    t0 = time.perf_counter()
    c0 = time.process_time()
    fn: Callable = SOLVERS[solver]
    kw = dict(solver_kwargs or {})
    ctx = QueryContext(index, q, k, 1.0, "theta")
    out: Dict = {"solver": solver, "region_size": ctx.nR, "n_thresholds": int(len(ctx.T))}
    best = np.zeros(0, dtype=np.int64)
    if incumbent == "paper" and ctx.nR:
        f = ballcore_f(index, q, k)
        g = greedy(index, q, k, ctx=ctx)
        best = f if len(f) >= len(g) else g
    out["incumbent_size"] = int(len(best))
    proven = True
    status = "optimal"
    solver_seconds = 0.0
    calls = 0
    visited = 0
    max_c = 0
    for j in ctx.order:
        if ctx.cnt[j] <= len(best):
            break
        visited += 1
        tau = float(ctx.T[j])
        C = ctx.peel(ctx.S_mask(tau))
        csize = int(C.sum())
        max_c = max(max_c, csize)
        if csize <= len(best):
            continue
        if candidate_cap and csize > candidate_cap:
            proven = False
            status = "candidate_cap"
            continue
        left = time_limit - (time.perf_counter() - t0)
        if left <= 0:
            proven = False
            status = "timeout"
            break
        loc = np.flatnonzero(C)  # local ids within R; q is local 0 and C[0] is True
        cptr, cidx = induced_csr(ctx.lptr, ctx.lidx, loc)
        q_loc = int(np.searchsorted(loc, 0))
        res = fn(cptr, cidx, ctx.XR[loc], tau, k, q_loc, lb=len(best), time_limit=left, **kw)
        calls += 1
        solver_seconds += float(res.get("seconds", 0.0))
        if res["status"] in ("optimal_improved", "timeout_improved") and res["size"] > len(best):
            best = ctx.R[loc[np.asarray(res["members"], dtype=np.int64)]]
        if res["status"] in ("timeout_improved", "timeout_none"):
            proven = False
            status = "timeout"
            break
        if res["status"] in ("too_large", "error"):
            proven = False
            status = res["status"]
    out.update(
        members=np.sort(best),
        size=int(len(best)),
        proven=bool(proven),
        status=status,
        seconds=time.perf_counter() - t0,
        cpu_seconds=time.process_time() - c0,
        solver_seconds=solver_seconds,
        n_solver_calls=calls,
        n_thresholds_visited=visited,
        max_candidates=max_c,
    )
    return out
