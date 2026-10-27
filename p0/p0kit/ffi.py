"""ctypes bindings for p0kit/_lib/libp0.so (build it with ./build.sh)."""

from __future__ import annotations

import ctypes
import os
from typing import Optional

import numpy as np

METRICS = [
    "steps",
    "converged",
    "npairs",
    "t0_refused",
    "broken",
    "first_break",
    "t0_rejected",
    "new_rejected",
    "first_new_rej",
    "diam0",
    "diam_final",
    "diam_max",
    "n_clusters",
    "largest_frac",
    "t_consensus",
    "ever_rejected",
    "open_acc0",
]

KRMAX_STATUS = {
    0: "optimal_improved",
    1: "optimal_none",
    2: "timeout_improved",
    3: "timeout_none",
}

_LIB = None
_VP = ctypes.c_void_p
_I = ctypes.c_int
_I64 = ctypes.c_int64
_D = ctypes.c_double
_U64 = ctypes.c_uint64


def lib_path() -> str:
    env = os.environ.get("P0_LIB")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_lib", "libp0.so")


def lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    path = lib_path()
    if not os.path.exists(path):
        raise RuntimeError(f"{path} not found; run ./build.sh in the kit root first")
    L = ctypes.CDLL(path)
    L.p0_dyn_version.restype = _I
    L.p0_num_metrics.restype = _I
    L.p0_krmax_version.restype = _I
    L.p0_graph_version.restype = _I
    L.p0_sim_sync.restype = _I
    L.p0_sim_sync.argtypes = [
        _I, _I, _VP, _VP, _VP, _VP, _VP, _VP, _VP, _VP, _VP, _VP, _VP, _VP, _D, _I, _D, _I, _D, _D, _I, _D, _D, _D,
        _U64, _VP, _VP,
    ]
    L.p0_sim_dw.restype = _I
    L.p0_sim_dw.argtypes = [
        _I, _I, _VP, _VP, _VP, _VP, _VP, _VP, _D, _I, _U64, _D, _D, _I, _D, _D, _VP, _VP,
    ]
    L.p0_krmax_solve.restype = _I
    L.p0_krmax_solve.argtypes = [
        _I, _VP, _VP, _I, _VP, _D, _I, _I, _I, _D, _I, _D, _VP, _VP, _VP,
    ]
    L.p0_bnb_version.restype = _I
    L.p0_bnb_solve.restype = _I
    L.p0_bnb_solve.argtypes = [
        _I, _VP, _VP, _I, _VP, _D, _I, _I, _I, _D, _I, _D, _VP, _VP, _VP,
    ]
    L.p0_bnb_solve_radii.restype = _I
    L.p0_bnb_solve_radii.argtypes = [
        _I, _VP, _VP, _I, _VP, _VP, _I, _I, _I, _D, _I, _D, _VP, _VP, _VP,
    ]
    L.p0_gtheta_mask.restype = _I
    L.p0_gtheta_mask.argtypes = [_I64, _VP, _VP, _I, _VP, _VP, _D, _VP]
    L.p0_core_numbers.restype = _I
    L.p0_core_numbers.argtypes = [_I64, _VP, _VP, _VP, _VP]
    L.p0_peel_component.restype = _I
    L.p0_peel_component.argtypes = [_I, _VP, _VP, _VP, _I, _I, _VP]
    L.p0_greedy_conflict.restype = _I
    L.p0_greedy_conflict.argtypes = [_I, _VP, _VP, _I, _VP, _D, _I, _I, _VP, _D, _VP]
    L.p0_components.restype = _I64
    L.p0_components.argtypes = [_I64, _VP, _VP, _VP, _VP]
    if L.p0_num_metrics() != len(METRICS):
        raise RuntimeError("libp0.so and ffi.py disagree on the metric list; rebuild with ./build.sh")
    _LIB = L
    return L


def _arr(a, dtype, shape=None):
    """Contiguous copy/view with the given dtype; returns (array, pointer)."""
    if a is None:
        return None, None
    out = np.ascontiguousarray(a, dtype=dtype)
    if shape is not None and out.shape != shape:
        out = out.reshape(shape)
    return out, out.ctypes.data


def versions() -> dict:
    L = lib()
    return {
        "dyn": L.p0_dyn_version(),
        "krmax": L.p0_krmax_version(),
        "bnb": L.p0_bnb_version(),
        "graph": L.p0_graph_version(),
    }


def components(ptr, idx, node_mask=None):
    """Component id per vertex (-1 if excluded) and the number of components."""
    L = lib()
    ptr, pP = _arr(ptr, np.int64)
    idx, pI = _arr(idx, np.int32)
    node_mask, pM = _arr(node_mask, np.uint8)
    n = len(ptr) - 1
    comp = np.zeros(n, dtype=np.int32)
    nc = L.p0_components(n, pP, pI, pM, comp.ctypes.data)
    return comp, int(nc)


def sim_sync(
    X: np.ndarray,
    th_acc: np.ndarray,
    th_rej: np.ndarray,
    ptr: np.ndarray,
    idx: np.ndarray,
    W: Optional[np.ndarray] = None,
    ext_mean: Optional[np.ndarray] = None,
    ext_mask: Optional[np.ndarray] = None,
    lambda_ext: float = 0.0,
    model: str = "hk",
    beta: float = 0.05,
    T_max: int = 1000,
    move_tol_rel: float = 1e-10,
    rel_eps: float = 1e-9,
    diam_every: int = 10,
    cons_rel: float = 1e-3,
    cluster_rel: float = 1e-6,
    gamma: float = 0.5,
    seed: int = 0,
    return_X: bool = False,
    W_self: Optional[np.ndarray] = None,
    W_edge: Optional[np.ndarray] = None,
    open_ptr: Optional[np.ndarray] = None,
    open_X: Optional[np.ndarray] = None,
    open_W: Optional[np.ndarray] = None,
):
    """Synchronous dynamics: model 'hk' (eq. 1), 'sjt' (eq. 2) or 'follow'
    (each member moves gamma towards one accepted neighbour drawn per step).

    Weights: either a dense raw-affinity matrix W [n x n] (small groups, tests)
    or W_self [n] and W_edge [nnz] aligned with the CSR entries; None = equal.
    Open group: open_ptr [n+1] and open_X [*, m] give each member's outside
    neighbours (fixed positions); accepted ones join its update (weights open_W)."""
    L = lib()
    X, pX = _arr(X, np.float64)
    n, m = X.shape
    th_acc, pA = _arr(th_acc, np.float64)
    th_rej, pR = _arr(th_rej, np.float64)
    ptr, pP = _arr(ptr, np.int64)
    idx, pI = _arr(idx, np.int32)
    if W is not None:
        if W_self is not None or W_edge is not None:
            raise ValueError("pass either W or W_self/W_edge")
        W = np.asarray(W, dtype=np.float64)
        src = np.repeat(np.arange(n), np.diff(ptr))
        W_self = np.diag(W).copy()
        W_edge = W[src, idx.astype(np.int64)]
    if (W_self is None) != (W_edge is None):
        raise ValueError("W_self and W_edge go together")
    W_self, pWs = _arr(W_self, np.float64)
    W_edge, pWe = _arr(W_edge, np.float64)
    if W_self is not None and (len(W_self) != n or len(W_edge) != len(idx)):
        raise ValueError("W_self must have n entries and W_edge one per CSR entry")
    if (open_ptr is None) != (open_X is None):
        raise ValueError("open_ptr and open_X go together")
    open_ptr, pOp = _arr(open_ptr, np.int64)
    open_X, pOx = _arr(open_X, np.float64)
    open_W, pOw = _arr(open_W, np.float64)
    if open_ptr is not None:
        if len(open_ptr) != n + 1 or open_X.ndim != 2 or open_X.shape[1] != m or len(open_X) != open_ptr[-1]:
            raise ValueError("open_ptr must have n+1 entries and open_X open_ptr[-1] rows of m coordinates")
        if open_W is not None and len(open_W) != len(open_X):
            raise ValueError("open_W needs one weight per outside neighbour")
    ext_mean, pE = _arr(ext_mean, np.float64)
    ext_mask, pM = _arr(ext_mask, np.uint8)
    out = np.zeros(len(METRICS), dtype=np.float64)
    Xout = np.empty_like(X) if return_X else None
    code = {"hk": 0, "sjt": 1, "follow": 2}[model]
    rc = L.p0_sim_sync(
        n, m, pX, pA, pR, pP, pI, pWs, pWe, pOp, pOx, pOw, pE, pM, float(lambda_ext), code, float(beta),
        int(T_max), float(move_tol_rel), float(rel_eps), int(diam_every), float(cons_rel), float(cluster_rel),
        float(gamma), int(seed) & 0xFFFFFFFFFFFFFFFF,
        out.ctypes.data, Xout.ctypes.data if Xout is not None else None,
    )
    if rc != 0:
        raise RuntimeError(f"p0_sim_sync failed with code {rc}")
    res = dict(zip(METRICS, out.tolist()))
    if return_X:
        return res, Xout
    return res


def sim_dw(
    X: np.ndarray,
    th_acc: np.ndarray,
    th_rej: np.ndarray,
    ptr: np.ndarray,
    idx: np.ndarray,
    rev: np.ndarray,
    mu: float = 0.3,
    rounds: int = 200,
    seed: int = 0,
    move_tol_rel: float = 1e-10,
    rel_eps: float = 1e-9,
    diam_every: int = 5,
    cons_rel: float = 1e-3,
    cluster_rel: float = 1e-6,
    return_X: bool = False,
):
    L = lib()
    X, pX = _arr(X, np.float64)
    n, m = X.shape
    th_acc, pA = _arr(th_acc, np.float64)
    th_rej, pR = _arr(th_rej, np.float64)
    ptr, pP = _arr(ptr, np.int64)
    idx, pI = _arr(idx, np.int32)
    rev, pV = _arr(rev, np.int64)
    out = np.zeros(len(METRICS), dtype=np.float64)
    Xout = np.empty_like(X) if return_X else None
    rc = L.p0_sim_dw(
        n, m, pX, pA, pR, pP, pI, pV, float(mu), int(rounds), int(seed) & 0xFFFFFFFFFFFFFFFF,
        float(move_tol_rel), float(rel_eps), int(diam_every), float(cons_rel), float(cluster_rel),
        out.ctypes.data, Xout.ctypes.data if Xout is not None else None,
    )
    if rc != 0:
        raise RuntimeError(f"p0_sim_dw failed with code {rc}")
    res = dict(zip(METRICS, out.tolist()))
    if return_X:
        return res, Xout
    return res


def _threshold_call(fn_name, ptr, idx, X, tau, k, q_local, lb, time_limit, mode, rel_eps) -> dict:
    L = lib()
    ptr, pP = _arr(ptr, np.int64)
    idx, pI = _arr(idx, np.int32)
    X, pX = _arr(X, np.float64)
    n, m = X.shape
    members = np.zeros(max(n, 1), dtype=np.int32)
    size = ctypes.c_int(0)
    stats = np.zeros(8, dtype=np.float64)
    rc = getattr(L, fn_name)(
        n, pP, pI, m, pX, float(tau), int(k), int(q_local), int(lb), float(time_limit),
        int(mode), float(rel_eps), members.ctypes.data, ctypes.addressof(size),
        stats.ctypes.data,
    )
    if rc != 0:
        raise RuntimeError(f"{fn_name} failed with code {rc}")
    status = int(stats[2])
    return {
        "status": KRMAX_STATUS[status],
        "members": members[: size.value].copy(),
        "size": int(size.value),
        "nodes": int(stats[0]),
        "seconds": float(stats[1]),
        "ub_root": float(stats[3]),
        "pre_size": int(stats[4]),
    }


def bnb_solve(ptr, idx, X, tau, k, q_local, lb=0, time_limit=60.0, branch_order=0, rel_eps=1e-9) -> dict:
    """The paper's fixed-threshold branch and bound (Lemma 5), reference copy."""
    return _threshold_call("p0_bnb_solve", ptr, idx, X, tau, k, q_local, lb, time_limit, branch_order, rel_eps)


def bnb_solve_radii(ptr, idx, X, radius, k, q_local, lb=0, time_limit=60.0, branch_order=0, rel_eps=1e-9) -> dict:
    """Branch and bound with conflicts d(u,v) > min(r_u, r_v): exact pairwise model."""
    L = lib()
    ptr, pP = _arr(ptr, np.int64)
    idx, pI = _arr(idx, np.int32)
    X, pX = _arr(X, np.float64)
    radius, pR = _arr(radius, np.float64)
    n, m = X.shape
    members = np.zeros(max(n, 1), dtype=np.int32)
    size = ctypes.c_int(0)
    stats = np.zeros(8, dtype=np.float64)
    rc = L.p0_bnb_solve_radii(
        n, pP, pI, m, pX, pR, int(k), int(q_local), int(lb), float(time_limit), int(branch_order),
        float(rel_eps), members.ctypes.data, ctypes.addressof(size), stats.ctypes.data,
    )
    if rc != 0:
        raise RuntimeError(f"p0_bnb_solve_radii failed with code {rc}")
    return {
        "status": KRMAX_STATUS[int(stats[2])],
        "members": members[: size.value].copy(),
        "size": int(size.value),
        "nodes": int(stats[0]),
        "seconds": float(stats[1]),
        "ub_root": float(stats[3]),
        "pre_size": int(stats[4]),
    }


def peel_component(ptr, idx, in_mask, q_local: int, k: int) -> np.ndarray:
    """Boolean mask of C_k(S, q) on a local graph."""
    L = lib()
    ptr, pP = _arr(ptr, np.int64)
    idx, pI = _arr(idx, np.int32)
    in_mask, pM = _arr(in_mask, np.uint8)
    n = len(ptr) - 1
    out = np.zeros(n, dtype=np.uint8)
    L.p0_peel_component(n, pP, pI, pM, int(q_local), int(k), out.ctypes.data)
    return out.astype(bool)


def greedy_conflict(ptr, idx, X, tau: float, k: int, q_local: int, in_mask, rel_eps: float = 1e-9) -> np.ndarray:
    L = lib()
    ptr, pP = _arr(ptr, np.int64)
    idx, pI = _arr(idx, np.int32)
    X, pX = _arr(X, np.float64)
    in_mask, pM = _arr(in_mask, np.uint8)
    n, m = X.shape
    out = np.zeros(n, dtype=np.uint8)
    L.p0_greedy_conflict(n, pP, pI, m, pX, float(tau), int(k), int(q_local), pM, float(rel_eps), out.ctypes.data)
    return out.astype(bool)


def krmax_solve(
    ptr: np.ndarray,
    idx: np.ndarray,
    X: np.ndarray,
    tau: float,
    k: int,
    q_local: int,
    lb: int = 0,
    time_limit: float = 60.0,
    order_mode: int = 0,
    rel_eps: float = 1e-9,
) -> dict:
    """Maximum (k,tau)-core containing q on one candidate set ([35] AdvMax, adapted)."""
    return _threshold_call("p0_krmax_solve", ptr, idx, X, tau, k, q_local, lb, time_limit, order_mode, rel_eps)


def gtheta_mask(ptr, idx, X, theta, rel_eps: float = 1e-9) -> np.ndarray:
    L = lib()
    ptr, pP = _arr(ptr, np.int64)
    idx, pI = _arr(idx, np.int32)
    X, pX = _arr(X, np.float64)
    theta, pT = _arr(theta, np.float64)
    n, m = X.shape
    mask = np.zeros(len(idx), dtype=np.uint8)
    L.p0_gtheta_mask(n, pP, pI, m, pX, pT, float(rel_eps), mask.ctypes.data)
    return mask


def core_numbers(ptr, idx, mask=None) -> np.ndarray:
    L = lib()
    ptr, pP = _arr(ptr, np.int64)
    idx, pI = _arr(idx, np.int32)
    mask, pM = _arr(mask, np.uint8)
    n = len(ptr) - 1
    core = np.zeros(n, dtype=np.int32)
    L.p0_core_numbers(n, pP, pI, pM, core.ctypes.data)
    return core
