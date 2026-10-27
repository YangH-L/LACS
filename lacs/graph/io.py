"""Binary formats shared with the C++ engine (cpp/src/lacs_core.hpp).

Graph  (LACSGR01): int64 n, int64 nnz, int64 off[n+1], int32 adj[nnz] (symmetric CSR)
Matrix (LACSMT01): int64 rows, int64 cols, float32 data[rows*cols] (row-major)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp

GRAPH_MAGIC = b"LACSGR01"
MATRIX_MAGIC = b"LACSMT01"


def edges_to_csr(n: int, u: np.ndarray, v: np.ndarray) -> sp.csr_matrix:
    """Undirected simple graph (no self loops, no duplicates) as a binary CSR matrix."""
    u = np.asarray(u, dtype=np.int64)
    v = np.asarray(v, dtype=np.int64)
    keep = u != v
    u, v = u[keep], v[keep]
    rows = np.concatenate([u, v])
    cols = np.concatenate([v, u])
    A = sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    A.sum_duplicates()
    A.data[:] = 1
    A.sort_indices()
    return A


def write_graph(path: str | Path, A: sp.csr_matrix) -> None:
    A = sp.csr_matrix(A)
    A.sort_indices()
    n = A.shape[0]
    off = A.indptr.astype(np.int64)
    adj = A.indices.astype(np.int32)
    with open(path, "wb") as f:
        f.write(GRAPH_MAGIC)
        f.write(np.int64(n).tobytes())
        f.write(np.int64(len(adj)).tobytes())
        f.write(off.tobytes())
        f.write(adj.tobytes())


def read_graph(path: str | Path) -> sp.csr_matrix:
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic != GRAPH_MAGIC:
            raise ValueError(f"bad graph magic in {path}")
        n = int(np.frombuffer(f.read(8), dtype=np.int64)[0])
        nnz = int(np.frombuffer(f.read(8), dtype=np.int64)[0])
        off = np.frombuffer(f.read(8 * (n + 1)), dtype=np.int64)
        adj = np.frombuffer(f.read(4 * nnz), dtype=np.int32)
    data = np.ones(nnz, dtype=np.int8)
    return sp.csr_matrix((data, adj.astype(np.int64), off), shape=(n, n))


def write_matrix(path: str | Path, M: np.ndarray) -> None:
    M = np.ascontiguousarray(np.asarray(M, dtype=np.float32))
    if M.ndim == 1:
        M = M[:, None]
    with open(path, "wb") as f:
        f.write(MATRIX_MAGIC)
        f.write(np.int64(M.shape[0]).tobytes())
        f.write(np.int64(M.shape[1]).tobytes())
        f.write(M.tobytes())


def read_matrix(path: str | Path) -> np.ndarray:
    with open(path, "rb") as f:
        if f.read(8) != MATRIX_MAGIC:
            raise ValueError(f"bad matrix magic in {path}")
        r = int(np.frombuffer(f.read(8), dtype=np.int64)[0])
        c = int(np.frombuffer(f.read(8), dtype=np.int64)[0])
        data = np.frombuffer(f.read(4 * r * c), dtype=np.float32)
    return data.reshape(r, c).copy()
