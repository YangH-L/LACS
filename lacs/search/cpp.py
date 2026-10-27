"""Thin wrapper around the C++ search engine (cpp/build/lacs_search)."""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp

from ..graph.io import write_graph, write_matrix

LOG = logging.getLogger("lacs.search")

CPP_DIR = Path(__file__).resolve().parents[2] / "cpp"
BUILD_DIR = CPP_DIR / "build"
TOOLS = ("lacs_search", "lacs_convert", "lacs_project", "lacs_lpa")


def _sources() -> list[Path]:
    return sorted((CPP_DIR / "src").glob("*.[ch]pp")) + [CPP_DIR / "CMakeLists.txt"]


def _sources_digest() -> str:
    h = hashlib.sha256()
    for p in _sources():
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()


def ensure_built(force: bool = False) -> Path:
    """Build the C++ tools unless they exist and were built from exactly the current sources.

    Freshness is decided by a digest of the sources stored next to the binaries, not by file
    times: a code package unpacked over an older build can carry older (or, from a machine in
    another time zone, future) timestamps. Every build starts from an empty build directory, so
    objects of two versions are never mixed."""
    exes = [BUILD_DIR / t for t in TOOLS]
    stamp = BUILD_DIR / ".sources.sha256"
    digest = _sources_digest()
    if not force and all(e.exists() for e in exes) and stamp.exists() and stamp.read_text().strip() == digest:
        return BUILD_DIR
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    if shutil.which("cmake"):
        LOG.info("building C++ tools with cmake")
        subprocess.run(["cmake", "-S", str(CPP_DIR), "-B", str(BUILD_DIR), "-DCMAKE_BUILD_TYPE=Release"],
                       check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["cmake", "--build", str(BUILD_DIR), "-j"], check=True, stdout=subprocess.DEVNULL)
    else:
        cxx = shutil.which("g++") or shutil.which("clang++")
        if not cxx:
            raise RuntimeError("need cmake or g++/clang++ to build cpp/")
        LOG.info("building C++ tools with %s", cxx)
        for t in TOOLS:
            cmd = [cxx, "-std=c++17", "-O3", "-march=native", "-pthread", str(CPP_DIR / "src" / f"{t}.cpp"),
                   "-o", str(BUILD_DIR / t)]
            subprocess.run(cmd, check=True)
    missing = [e.name for e in exes if not e.exists()]
    if missing:
        raise RuntimeError(f"C++ build did not produce {missing}")
    stamp.write_text(digest + "\n")
    return BUILD_DIR


def _graph_header(path: Path) -> tuple[int, int] | None:
    try:
        with open(path, "rb") as f:
            if f.read(8) != b"LACSGR01":
                return None
            n = int(np.frombuffer(f.read(8), dtype=np.int64)[0])
            nnz = int(np.frombuffer(f.read(8), dtype=np.int64)[0])
        return n, nnz
    except (OSError, IndexError):
        return None


def _matrix_header(path: Path) -> tuple[int, int] | None:
    try:
        with open(path, "rb") as f:
            if f.read(8) != b"LACSMT01":
                return None
            rows = int(np.frombuffer(f.read(8), dtype=np.int64)[0])
            cols = int(np.frombuffer(f.read(8), dtype=np.int64)[0])
        return rows, cols
    except (OSError, IndexError):
        return None


class CppSearch:
    """Writes graph/positions/tolerances once, then runs batches of queries.

    graph_bin / pos_bin: an existing file that already holds exactly this graph (e.g. the file
    A was read from) or these positions; it is used as is instead of writing a copy, which
    matters for large graphs searched under many settings. Only the header (sizes) is checked,
    so the caller must make sure the contents match."""

    def __init__(self, A: sp.csr_matrix, pos: np.ndarray, thetas: dict[str, np.ndarray], workdir: str | Path,
                 graph_bin: str | Path | None = None, pos_bin: str | Path | None = None):
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.n = A.shape[0]
        if pos.shape[0] != self.n:
            raise ValueError("positions must have one row per vertex")
        ensure_built()
        self.exe = BUILD_DIR / "lacs_search"
        if graph_bin is not None and _graph_header(Path(graph_bin)) == (self.n, int(A.nnz)):
            self.graph_path = Path(graph_bin)
        else:
            self.graph_path = self.workdir / "graph.bin"
            write_graph(self.graph_path, A)
        if pos_bin is not None and _matrix_header(Path(pos_bin)) == (self.n, int(pos.shape[1])):
            self.pos_path = Path(pos_bin)
        else:
            self.pos_path = self.workdir / "pos.bin"
            write_matrix(self.pos_path, pos)
        self.theta_paths: dict[str, Path] = {}
        for name, th in thetas.items():
            th = np.asarray(th, dtype=np.float32).reshape(-1)
            if th.shape[0] != self.n:
                raise ValueError(f"theta '{name}' has wrong length")
            p = self.workdir / f"theta_{name}.bin"
            write_matrix(p, th)
            self.theta_paths[name] = p

    def info(self) -> dict:
        out = subprocess.run([str(self.exe), "--graph", str(self.graph_path), "--info"], check=True,
                             capture_output=True, text=True)
        return json.loads(out.stdout.strip().splitlines()[-1])

    def core_numbers(self) -> np.ndarray:
        path = self.workdir / "cores.bin"
        subprocess.run([str(self.exe), "--graph", str(self.graph_path), "--info", "--cores-out", str(path)],
                       check=True, capture_output=True, text=True)
        return np.fromfile(path, dtype=np.int32)


    def run(self, queries: Iterable[int], algos: Sequence[str], k: int, theta: str = "acc", *,
            time_limit: float = 600.0, members: bool = True, extra: Sequence[str] = (),
            r: float | None = None, tag: str = "run", n_proc: int = 1,
            r_per_query: Sequence[float] | None = None, size_per_query: Sequence[int] | None = None) -> list[dict]:
        """Run every algorithm on every query. n_proc > 1 splits the queries over
        that many engine processes (results are returned in query order).
        r_per_query: one threshold per query for the kr_* algorithms (overrides r);
        size_per_query: one minimum community size per query for sac_inc."""
        queries = [int(q) for q in queries]
        for name, vals in (("r_per_query", r_per_query), ("size_per_query", size_per_query)):
            if vals is not None and len(vals) != len(queries):
                raise ValueError(f"{name} needs one value per query")
        lines = []
        for i, q in enumerate(queries):
            parts = [str(q)]
            if r_per_query is not None or size_per_query is not None:
                parts.append(repr(float(r_per_query[i])) if r_per_query is not None else "-")
            if size_per_query is not None:
                parts.append(str(int(size_per_query[i])))
            lines.append(" ".join(parts))
        n_proc = max(1, min(int(n_proc), len(queries)))
        chunks = [lines[i::n_proc] for i in range(n_proc)] if queries else [[]]
        procs = []
        for ci, chunk in enumerate(chunks):
            suffix = f"_{ci}" if n_proc > 1 else ""
            qpath = self.workdir / f"queries_{tag}{suffix}.txt"
            qpath.write_text("\n".join(chunk) + "\n")
            out_path = self.workdir / f"out_{tag}{suffix}.jsonl"
            cmd = [str(self.exe), "--graph", str(self.graph_path), "--pos", str(self.pos_path), "--theta",
                   str(self.theta_paths[theta]), "--queries", str(qpath), "--algo", ",".join(algos), "--k",
                   str(k), "--time-limit", str(time_limit), "--out", str(out_path)]
            if r is not None:
                cmd += ["--r", repr(float(r))]
            if not members:
                cmd.append("--no-members")
            cmd += list(extra)
            procs.append((subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True),
                          out_path))
        rows = []
        for p, out_path in procs:
            _, err = p.communicate()
            if p.returncode != 0:
                raise RuntimeError(f"lacs_search failed ({p.returncode}): {err.strip()}")
            with open(out_path) as f:
                rows += [json.loads(line) for line in f if line.strip()]
        order = {q: i for i, q in enumerate(queries)}
        rows.sort(key=lambda row: order.get(row["q"], 0))
        return rows


def convert_edges(edges_txt: str | Path, out_bin: str | Path, relabel_ids: str | Path | None = None,
                  lowmem: bool = False) -> dict:
    """Edge list -> LACSGR01 graph. lowmem (v0.5): two passes over the file instead of holding all
    pairs in memory (about 16 GB instead of about 70 GB for 1.8 billion edges); same output."""
    ensure_built()
    cmd = [str(BUILD_DIR / "lacs_convert"), "--in", str(edges_txt), "--out", str(out_bin)]
    if relabel_ids:
        cmd += ["--relabel", str(relabel_ids)]
    if lowmem:
        cmd.append("--lowmem")
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def project_bipartite(bip_txt: str | Path, out_bin: str | Path, ids_txt: str | Path, max_right_degree: int = 0,
                      min_shared: int = 1) -> dict:
    ensure_built()
    cmd = [str(BUILD_DIR / "lacs_project"), "--in", str(bip_txt), "--out", str(out_bin), "--ids", str(ids_txt),
           "--max-right-degree", str(max_right_degree), "--min-shared", str(min_shared)]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


def lpa_labels(graph_bin: str | Path, out_bin: str | Path, max_size: int = 1000, iters: int = 20,
               seed: int = 0) -> tuple[np.ndarray, dict]:
    """Size-capped label propagation (cpp/src/lacs_lpa.cpp): int32 community label per vertex."""
    ensure_built()
    cmd = [str(BUILD_DIR / "lacs_lpa"), "--graph", str(graph_bin), "--out", str(out_bin), "--max-size",
           str(int(max_size)), "--iters", str(int(iters)), "--seed", str(int(seed))]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return np.fromfile(out_bin, dtype=np.int32), json.loads(out.stdout.strip().splitlines()[-1])
