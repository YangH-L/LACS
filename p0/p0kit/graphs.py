"""Graph utilities: induced CSR of a member set, complete graphs, reverse
indices, outside-neighbour means, and group statistics."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def csr_from_edges(n: int, src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Directed CSR (both directions must already be present). Removes self loops
    and duplicate entries; rows are sorted."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    if len(src):
        key = np.unique(src * n + dst)
        src, dst = key // n, key % n
    counts = np.bincount(src, minlength=n) if len(src) else np.zeros(n, dtype=np.int64)
    ptr = np.zeros(n + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(counts)
    return ptr, dst.astype(np.int32)


def symmetric_csr(n: int, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """CSR of the undirected graph with edge list (u, v)."""
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    return csr_from_edges(n, np.concatenate([u, v]), np.concatenate([v, u]))


def induced_csr(
    gptr: np.ndarray,
    gidx: np.ndarray,
    members: np.ndarray,
    pos_buf: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Directed CSR of G[members] in local indices (order of `members`).

    pos_buf: optional int32 array of size |V| filled with -1, reused across
    calls to avoid allocations on large graphs; it is restored before return.
    """
    members = np.asarray(members, dtype=np.int64)
    n = len(members)
    own = pos_buf is None
    pos = np.full(len(gptr) - 1, -1, dtype=np.int32) if own else pos_buf
    pos[members] = np.arange(n, dtype=np.int32)
    try:
        starts = np.asarray(gptr[members], dtype=np.int64)
        ends = np.asarray(gptr[members + 1], dtype=np.int64)
        lens = ends - starts
        total = int(lens.sum())
        if total == 0:
            return np.zeros(n + 1, dtype=np.int64), np.zeros(0, dtype=np.int32)
        nb = np.concatenate([np.asarray(gidx[s:e]) for s, e in zip(starts, ends)])
        src = np.repeat(np.arange(n, dtype=np.int64), lens)
        loc = pos[nb.astype(np.int64)]
        keep = loc >= 0
        return csr_from_edges(n, src[keep], loc[keep].astype(np.int64))
    finally:
        if not own:
            pos[members] = -1


def complete_csr(n: int) -> Tuple[np.ndarray, np.ndarray]:
    ptr = np.arange(n + 1, dtype=np.int64) * max(n - 1, 0)
    if n <= 1:
        return np.zeros(n + 1, dtype=np.int64), np.zeros(0, dtype=np.int32)
    rows = np.repeat(np.arange(n), n - 1)
    cols = np.tile(np.arange(n - 1), n)
    cols = cols + (cols >= rows)
    return ptr, cols.astype(np.int32)


def reverse_index(ptr: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """rev[p] = position of the pair (u, v) for p = (v, u). Requires symmetry."""
    n = len(ptr) - 1
    src = np.repeat(np.arange(n, dtype=np.int64), np.diff(ptr))
    dst = idx.astype(np.int64)
    key = src * n + dst
    order = np.argsort(key, kind="stable")
    rkey = dst * n + src
    loc = np.searchsorted(key[order], rkey)
    if len(loc) and (np.any(loc >= len(key)) or np.any(key[order][np.minimum(loc, len(key) - 1)] != rkey)):
        raise ValueError("graph is not symmetric")
    return order[loc].astype(np.int64)


def outside_neighbors(
    gptr: np.ndarray,
    gidx: np.ndarray,
    members: np.ndarray,
    X: np.ndarray,
    pos_buf: Optional[np.ndarray] = None,
    max_neighbors: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each member's neighbours outside the group, at their initial positions.

    Returns (optr [n+1] int64, oX [optr[-1] x m] float64, n_outside [n] int64).
    With max_neighbors > 0, a member with more outside neighbours keeps a random
    sample of that many (keeps hubs cheap); n_outside counts all of them."""
    members = np.asarray(members, dtype=np.int64)
    n = len(members)
    m = X.shape[1]
    own = pos_buf is None
    pos = np.full(len(gptr) - 1, -1, dtype=np.int32) if own else pos_buf
    pos[members] = np.arange(n, dtype=np.int32)
    picks = []
    cnt = np.zeros(n, dtype=np.int64)
    try:
        for i, v in enumerate(members):
            nb = np.asarray(gidx[gptr[v]:gptr[v + 1]], dtype=np.int64)
            out = nb[pos[nb] < 0]
            cnt[i] = len(out)
            if max_neighbors and len(out) > max_neighbors:
                g = rng if rng is not None else np.random.default_rng(int(v))
                out = np.sort(g.choice(out, size=max_neighbors, replace=False))
            picks.append(out)
    finally:
        if not own:
            pos[members] = -1
    lens = np.array([len(p) for p in picks], dtype=np.int64)
    optr = np.zeros(n + 1, dtype=np.int64)
    optr[1:] = np.cumsum(lens)
    allp = np.concatenate(picks) if n and optr[-1] > 0 else np.zeros(0, dtype=np.int64)
    oX = np.asarray(X[allp], dtype=np.float64).reshape(len(allp), m)
    return optr, oX, cnt


def external_means(
    gptr: np.ndarray,
    gidx: np.ndarray,
    members: np.ndarray,
    X: np.ndarray,
    pos_buf: Optional[np.ndarray] = None,
    max_neighbors: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean initial position of each member's (sampled) outside neighbours.
    Returns (ext_mean [n x m], ext_mask [n] uint8, n_outside [n])."""
    optr, oX, cnt = outside_neighbors(gptr, gidx, members, X, pos_buf, max_neighbors, rng)
    return means_from_lists(optr, oX) + (cnt,)


def means_from_lists(optr: np.ndarray, oX: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(optr) - 1
    m = oX.shape[1]
    lens = np.diff(optr)
    ext = np.zeros((n, m), dtype=np.float64)
    if len(oX):
        rows = np.repeat(np.arange(n), lens)
        np.add.at(ext, rows, oX)
    nz = lens > 0
    ext[nz] /= lens[nz, None]
    return ext, nz.astype(np.uint8)


def pairwise_sq(Xh: np.ndarray) -> np.ndarray:
    G = Xh @ Xh.T
    sq = np.diag(G)
    D2 = sq[:, None] + sq[None, :] - 2.0 * G
    np.maximum(D2, 0.0, out=D2)
    np.fill_diagonal(D2, 0.0)
    return D2


def group_stats(Xh: np.ndarray, th: np.ndarray, rel_eps: float = 1e-9, block: int = 512) -> dict:
    """Static statistics of a group: diameter, tolerance conditions and the
    share of members whose tolerance does not cover everybody else.
    Distances are computed in row blocks (memory O(block * n))."""
    n = len(th)
    if n == 0:
        return {}
    Xh = np.asarray(Xh, dtype=np.float64)
    th = np.asarray(th, dtype=np.float64)
    tol = 1.0 + rel_eps
    sq = np.einsum("ij,ij->i", Xh, Xh)
    far = np.zeros(n)
    pair_ok = True
    for a in range(0, n, block):
        b = min(n, a + block)
        D2 = sq[a:b, None] + sq[None, :] - 2.0 * (Xh[a:b] @ Xh.T)
        np.maximum(D2, 0.0, out=D2)
        D2[np.arange(b - a), np.arange(a, b)] = 0.0
        D = np.sqrt(D2)
        far[a:b] = D.max(axis=1)
        if pair_ok:
            pair_ok = bool(np.all(D <= np.minimum(th[a:b, None], th[None, :]) * tol))
    diam = float(far.max()) if n > 1 else 0.0
    tmin = float(np.min(th))
    return {
        "size": n,
        "diam": diam,
        "theta_min": tmin,
        "theta_median": float(np.median(th)),
        "diam_over_theta_min": diam / tmin if tmin > 0 else float("inf"),
        "sat_iii": bool(diam <= tmin * tol),
        "sat_pairwise": bool(pair_ok),
        "member_viol_frac": float(np.mean(far > th * tol)),
    }
