"""File formats of the kit (the contract with the LACS engine).

Instance, one per dataset and tolerance assignment, either
  (a) a single .npz with arrays  indptr int64 [n+1], indices int32 [nnz] (CSR of
      the undirected graph, both directions), X float [n, m] (positions; geo:
      Earth-centred coordinates in km, so Euclidean distance = chord length),
      theta float64 [n] (tolerances, same unit), optional meta (JSON string); or
  (b) an instance.json  {"indptr": path, "indices": path, "X": path,
      "theta": path, "meta": {...}}  pointing to .npy files (paths relative to
      the json file). (b) lets many settings share one copy of a large graph.

Groups (.jsonl or .jsonl.gz), one JSON object per line and per (query, method):
  {"dataset": "gowalla", "query": 123, "k": 3, "method": "lcc_exact",
   "members": [global ids ...], "proven": true, "time_ms": 0.42, ...}
  "members" may be null for groups that were too large to store; "size" then
  gives the size. Extra scalar keys are carried into outputs with prefix "g_".

Per-query results (.csv) for E2-E4: one row per (dataset, setting, query,
method); see e2_runs.py for the columns.
"""

from __future__ import annotations

import gzip
import json
import os
import zlib
from typing import Dict, Iterator, List, Tuple

import numpy as np


def save_npz_instance(path: str, indptr, indices, X, theta, meta: Dict = None) -> None:
    np.savez(
        path,
        indptr=np.asarray(indptr, dtype=np.int64),
        indices=np.asarray(indices, dtype=np.int32),
        X=np.asarray(X),
        theta=np.asarray(theta, dtype=np.float64),
        meta=np.array(json.dumps(meta or {})),
    )


def write_instance_json(path: str, indptr_path: str, indices_path: str, X_path: str, theta_path: str,
                        meta: Dict = None) -> None:
    base = os.path.dirname(os.path.abspath(path))

    def rel(p):
        return os.path.relpath(os.path.abspath(p), base)

    spec = {"indptr": rel(indptr_path), "indices": rel(indices_path), "X": rel(X_path),
            "theta": rel(theta_path), "meta": meta or {}}
    os.makedirs(base, exist_ok=True)
    with open(path, "w") as f:
        json.dump(spec, f, indent=1)


class Instance:
    def __init__(self, path: str, mmap: bool = True):
        self.path = path
        mode = "r" if mmap else None
        if path.endswith(".json"):
            with open(path) as f:
                spec = json.load(f)
            base = os.path.dirname(os.path.abspath(path))

            def ld(key):
                p = spec[key]
                p = p if os.path.isabs(p) else os.path.join(base, p)
                return np.load(p, mmap_mode=mode, allow_pickle=False)

            self.indptr, self.indices = ld("indptr"), ld("indices")
            self.X, self.theta = ld("X"), ld("theta")
            self.meta = spec.get("meta", {})
        else:
            z = np.load(path, mmap_mode=mode, allow_pickle=False)
            self.indptr = z["indptr"]
            self.indices = z["indices"]
            self.X = z["X"]
            self.theta = z["theta"]
            self.meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        self.n = len(self.indptr) - 1
        if self.X.shape[0] != self.n or self.theta.shape[0] != self.n:
            raise ValueError(f"{path}: X/theta do not match the graph size")


def _open(path: str, mode: str = "rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode)


def read_jsonl(path: str) -> Iterator[dict]:
    with _open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: {e}") from e


def read_groups(path: str) -> Iterator[dict]:
    for i, g in enumerate(read_jsonl(path), 1):
        for key in ("query", "method", "members"):
            if key not in g:
                raise ValueError(f"{path}: record {i}: missing key {key!r}")
        yield g


def recover_jsonl(path: str) -> Tuple[List[dict], Dict]:
    """All complete records of a JSONL(.gz) file, also after an interrupted run.

    A process killed while appending leaves a gzip member without its end marker
    and possibly a partial last line; gzip.open then raises at the end of the
    file. This reader decodes member by member, stops at a truncated member and
    keeps only complete, parseable lines. info: truncated, bad_lines, records."""
    with open(path, "rb") as fh:
        data = fh.read()
    truncated = False
    if path.endswith(".gz"):
        parts, pos = [], 0
        while pos < len(data):
            d = zlib.decompressobj(16 + zlib.MAX_WBITS)
            try:
                parts.append(d.decompress(data[pos:]))
            except zlib.error:
                truncated = True
                break
            if not d.eof:
                truncated = True
                break
            rest = d.unused_data
            pos = len(data) - len(rest)
            if not rest.strip(b"\x00"):
                break
        raw = b"".join(parts)
    else:
        raw = data
    lines = raw.split(b"\n")
    if lines and lines[-1].strip():
        truncated = True  # partial last line
    recs, bad = [], 0
    for line in lines[:-1]:
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            bad += 1
    return recs, {"truncated": truncated, "bad_lines": bad, "records": len(recs)}


def rewrite_jsonl(path: str, records: List[dict]) -> None:
    """Atomically replace path by a clean file holding records."""
    tmp = path + ".tmp" + (".gz" if path.endswith(".gz") else "")
    w = JsonlWriter(tmp, append=False)
    w.write_many(records)
    w.close()
    os.replace(tmp, path)


class JsonlWriter:
    """Append-only JSONL writer (gzip if the name ends with .gz); flushes per call."""

    def __init__(self, path: str, append: bool = True):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.path = path
        mode = "at" if append else "wt"
        self.f = gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)

    def write_many(self, rows: List[dict]) -> None:
        for r in rows:
            self.f.write(json.dumps(r, default=_json_default) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))
