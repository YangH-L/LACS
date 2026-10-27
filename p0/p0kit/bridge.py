"""Bridge to the LACS project (v0.4): instances, queries and engine runs.

Everything here reads what the earlier rounds left on disk, so that P0 runs on
exactly the instances and queries of the paper:

  round-3 E3 (runs/efficiency_round3/)
    work/e3_<name>/graph.bin           LACSGR01 graph
    work/e3_<name>/pos_default.bin     LACSMT01 default positions (n x 16)
    work/e3_<name>/base/theta_acc.bin  LACSMT01 default tolerances (s * unit)
    work/e3_<name>/lpa_labels.bin      size-capped LPA communities
    results/e3.json                    calibration (chosen / used scale)
    results/e3_per_query.csv           the queries of every setting
    config_resolved.json               seed and efficiency parameters
  meetup runs (runs/gowalla/, runs/brightkite/)
    graph/graph.bin, geo/positions.npy, geo/users.csv, results/geo_gates.json

Large files are memory-mapped, never copied. Run from the project root (the
directory that contains lacs/), with PYTHONPATH=p0.
"""

from __future__ import annotations

import json
import os
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

GRAPH_MAGIC = b"LACSGR01"
MATRIX_MAGIC = b"LACSMT01"


# ------------------------------------------------------------ binary files --
def mmap_graph(path) -> tuple:
    """(indptr int64 [n+1], indices int32 [nnz]) memory-mapped from a LACSGR01 file."""
    path = str(path)
    with open(path, "rb") as f:
        if f.read(8) != GRAPH_MAGIC:
            raise ValueError(f"bad graph magic in {path}")
        n = int(np.frombuffer(f.read(8), dtype=np.int64)[0])
        nnz = int(np.frombuffer(f.read(8), dtype=np.int64)[0])
    indptr = np.memmap(path, dtype=np.int64, mode="r", offset=24, shape=(n + 1,))
    indices = np.memmap(path, dtype=np.int32, mode="r", offset=24 + 8 * (n + 1), shape=(nnz,))
    if int(indptr[-1]) != nnz:
        raise ValueError(f"{path}: offsets inconsistent")
    return indptr, indices


def mmap_matrix(path) -> np.ndarray:
    path = str(path)
    with open(path, "rb") as f:
        if f.read(8) != MATRIX_MAGIC:
            raise ValueError(f"bad matrix magic in {path}")
        r = int(np.frombuffer(f.read(8), dtype=np.int64)[0])
        c = int(np.frombuffer(f.read(8), dtype=np.int64)[0])
    return np.memmap(path, dtype=np.float32, mode="r", offset=24, shape=(r, c))


class GraphShape:
    """Stands in for the scipy matrix CppSearch expects when an existing graph
    file is reused: only .shape and .nnz are read in that case."""

    def __init__(self, n: int, nnz: int):
        self.shape = (int(n), int(n))
        self.nnz = int(nnz)


# ------------------------------------------------------------- round 3 E3 --
def _ds_rng(seed, name: str) -> np.random.Generator:
    # identical to lacs.experiments.e3._ds_rng
    return np.random.default_rng([int(seed), zlib.crc32(str(name).encode("utf-8"))])


def load_round3(e3_run, name: str) -> Dict:
    """Instance of one synthetic-attribute dataset of round 3 (LiveJournal, Orkut)."""
    e3_run = Path(e3_run)
    work = e3_run / "work" / f"e3_{name}"
    cfg = json.loads((e3_run / "config_resolved.json").read_text())
    eff = cfg.get("efficiency", {})
    res = json.loads((e3_run / "results" / "e3.json").read_text())
    cal = (res.get("calibration") or {}).get(name) or {}
    s_def = cal.get("used", cal.get("chosen"))
    if s_def is None:
        raise ValueError(f"no calibrated scale for {name} in {e3_run}/results/e3.json")
    graph_bin = work / "graph.bin"
    pos_bin = work / "pos_default.bin"
    theta_bin = work / "base" / "theta_acc.bin"
    for p in (graph_bin, pos_bin, theta_bin):
        if not p.exists():
            raise FileNotFoundError(p)
    indptr, indices = mmap_graph(graph_bin)
    X0 = mmap_matrix(pos_bin)
    th0 = np.asarray(mmap_matrix(theta_bin)).reshape(-1)
    lab_path = work / "lpa_labels.bin"
    labels = np.fromfile(lab_path, dtype=np.int32) if lab_path.exists() else None
    return {
        "name": name, "kind": "round3", "e3_run": str(e3_run), "work": str(work),
        "graph_bin": str(graph_bin), "pos_bin": str(pos_bin), "theta_bin": str(theta_bin),
        "indptr": indptr, "indices": indices, "n": len(indptr) - 1, "nnz": int(indptr[-1]),
        "X0": X0, "theta0": th0, "labels": labels, "s_def": float(s_def),
        "sigma0": float(eff.get("theta_sigma", 0.4)), "dim": int(X0.shape[1]),
        "spacing": float(eff.get("city_spacing", 8.0)), "seed": int(cfg.get("seed", 0)),
        "k0": int(eff.get("default_k", 3)), "query_min_community": int(eff.get("query_min_community", 20)),
        "time_limit": float(eff.get("time_limit", 60)), "calibration": cal,
    }


def replay_synthetic(ds: Dict) -> Dict:
    """Replays the dataset's random stream of round 3 (lacs.experiments.e3: city_positions
    then the tolerance normals) to recover the community centres, the member noise and the
    standard normals Z exactly, and checks them against the stored files. If the check fails
    (another numpy version, say), falls back to Z recovered from the stored tolerances and to
    empirical community means; `exact` tells which happened."""
    n, dim = ds["n"], ds["dim"]
    labels = ds["labels"]
    X0 = np.asarray(ds["X0"])
    th0 = np.asarray(ds["theta0"], dtype=np.float32)
    s, sigma = ds["s_def"], ds["sigma0"]
    unit = np.sqrt(2.0 * dim)
    out = {"exact": False}
    if labels is not None and len(labels) == n:
        rng = _ds_rng(ds["seed"], ds["name"])
        C = int(labels.max()) + 1
        side = ds["spacing"] * unit * C ** (1.0 / dim)
        centres = rng.uniform(0.0, side, size=(C, dim))
        noise = rng.normal(size=(n, dim))
        Z = rng.normal(size=n)
        Xr = (centres[labels] + noise).astype(np.float32)
        th_unit = (unit * 1.0 * np.exp(sigma * Z - 0.5 * sigma ** 2)).astype(np.float32)
        thr = (th_unit * s).astype(np.float32)
        okX = bool(np.array_equal(Xr, X0))
        okT = bool(np.array_equal(thr, th0))
        out.update(check_positions=okX, check_theta=okT)
        if okX and okT:
            out.update(exact=True, centres=centres, noise=noise, Z=Z)
            return out
        del Xr, noise
    # fallback
    th_unit = th0.astype(np.float64) / s
    Z = (np.log(th_unit / unit) + 0.5 * sigma ** 2) / sigma
    out["Z"] = Z
    if labels is not None and len(labels) == n:
        C = int(labels.max()) + 1
        sums = np.zeros((C, dim))
        np.add.at(sums, labels, X0.astype(np.float64))
        cnt = np.bincount(labels, minlength=C).astype(float)
        out["centres_empirical"] = sums / np.maximum(cnt, 1)[:, None]
    return out


def default_queries(e3_run, name: str, k: int = 3, dim: int = 16) -> Dict:
    """Queries of the default setting in round 3, by batch. The paper pools four batches of
    50 (k sweep at k = 3, fraction 1.0, relative scale 1.0, dimension 16); the first three
    ran on the default instance, the dimension batch on a redrawn one of the same law."""
    import pandas as pd

    path = Path(e3_run) / "results" / "e3_per_query.csv"
    df = pd.read_csv(path, usecols=lambda c: c in {"dataset", "sweep", "value", "k", "q", "algo", "fraction", "dim",
                                                     "theta_scale"})
    d = df[(df["dataset"] == name) & (df["algo"] == "exact")]
    batches = {
        "k3": d[(d["sweep"] == "k") & (d["value"].astype(float) == float(k))],
        "fraction1": d[(d["sweep"] == "fraction") & (d["value"].astype(float) == 1.0)],
        "scale1": d[(d["sweep"] == "theta_scale") & (d["value"].astype(float) == 1.0)],
        "dim16": d[(d["sweep"] == "dim") & (d["value"].astype(float) == float(dim))],
    }
    out = {b: np.sort(g["q"].to_numpy(np.int64)) for b, g in batches.items() if len(g)}
    allq = np.concatenate(list(out.values())) if out else np.zeros(0, np.int64)
    batch_of = {}
    for b, qs in out.items():
        for q in qs.tolist():
            batch_of.setdefault(int(q), b)
    return {"batches": out, "all": np.unique(allq), "batch_of": batch_of, "n_pooled": int(len(allq))}


def paper_theta(Z, dim: int, s: float, sigma: float) -> np.ndarray:
    """theta_v = s * sqrt(2 dim) * exp(sigma Z_v - sigma^2 / 2), as float32 like the engine."""
    unit = np.sqrt(2.0 * dim)
    th_unit = (unit * np.exp(sigma * np.asarray(Z, dtype=np.float64) - 0.5 * sigma ** 2)).astype(np.float32)
    return (th_unit * float(s)).astype(np.float32)


def intrinsic_theta(Z, r_nbr: float, c: float, sigma: float) -> np.ndarray:
    """theta_v = c * r_nbr * exp(sigma Z_v - sigma^2 / 2): mean tolerance = c x median friend distance."""
    return (float(c) * float(r_nbr) * np.exp(sigma * np.asarray(Z, dtype=np.float64) - 0.5 * sigma ** 2)).astype(
        np.float32)


def median_edge_distance(indptr, indices, X, n_sample: int = 1_000_000, seed: int = 0) -> float:
    nnz = int(indptr[-1])
    rng = np.random.default_rng(seed)
    pos = np.sort(rng.integers(0, nnz, size=min(n_sample, nnz)))
    src = np.searchsorted(np.asarray(indptr), pos, side="right") - 1
    dst = np.asarray(indices[pos], dtype=np.int64)
    Xs = np.asarray(X[src], dtype=np.float64)
    Xd = np.asarray(X[dst], dtype=np.float64)
    return float(np.median(np.sqrt(((Xs - Xd) ** 2).sum(axis=1))))


def mixed_positions(ds: Dict, rep: Dict, mu: float, seed: int = 20261023) -> np.ndarray:
    """Positions with a share mu of vertices moved to the community of a random vertex
    (uniform U_v < mu; the same U and partners for every mu, so settings are nested)."""
    n = ds["n"]
    labels = ds["labels"]
    if labels is None:
        raise ValueError("mixing needs the LPA labels of round 3")
    rng = np.random.default_rng([int(seed), 77])
    U = rng.random(n)
    partner = rng.integers(0, n, size=n)
    moved = U < mu
    lab2 = labels.astype(np.int64).copy()
    lab2[moved] = labels[partner[moved]]
    if rep.get("exact"):
        X = (rep["centres"][lab2] + rep["noise"]).astype(np.float32)
    else:
        cen = rep["centres_empirical"]
        X = (np.asarray(ds["X0"], dtype=np.float64) - cen[labels] + cen[lab2]).astype(np.float32)
    return X


def query_pool(core: np.ndarray, k: int, labels: Optional[np.ndarray] = None, min_comm: int = 20,
               mode: str = "group20") -> np.ndarray:
    ok = np.asarray(core) >= k
    if mode == "group20":
        if labels is None:
            raise ValueError("group20 needs labels")
        sizes = np.bincount(labels)
        ok &= sizes[labels] >= min_comm
    elif mode != "kcore":
        raise ValueError(mode)
    return np.flatnonzero(ok)


# --------------------------------------------------------------- meetups --
def load_geo(run_dir, tolerance: Optional[str] = None) -> Dict:
    import pandas as pd

    run_dir = Path(run_dir)
    users = pd.read_csv(run_dir / "geo" / "users.csv")
    gates = run_dir / "results" / "geo_gates.json"
    tol = tolerance
    if tol is None and gates.exists():
        tol = json.loads(gates.read_text()).get("G2", {}).get("tolerance")
    tol = tol or [c for c in users.columns if c.startswith("theta_q")][0]
    graph_bin = run_dir / "graph" / "graph.bin"
    indptr, indices = mmap_graph(graph_bin)
    pos = np.load(run_dir / "geo" / "positions.npy")
    theta = users[tol].to_numpy(np.float32)
    return {"name": run_dir.name, "kind": "geo", "run": str(run_dir), "graph_bin": str(graph_bin),
            "indptr": indptr, "indices": indices, "n": len(indptr) - 1, "nnz": int(indptr[-1]),
            "X0": pos, "theta0": theta, "tolerance": tol, "dim": int(pos.shape[1])}


# --------------------------------------------------------------- engine --
def engine(ds: Dict, X, theta, workdir, pos_bin: Optional[str] = None):
    """CppSearch on the dataset's graph file (never copied); positions are written only
    when pos_bin is None or does not match."""
    from lacs.search.cpp import CppSearch

    return CppSearch(GraphShape(ds["n"], ds["nnz"]), X, {"acc": np.asarray(theta, dtype=np.float32)}, workdir,
                     graph_bin=ds["graph_bin"], pos_bin=pos_bin)


def engine_run(cs, queries, algo: str, k: int, time_limit: float, n_proc: int, members: bool, tag: str,
               r: Optional[float] = None, r_per_query: Optional[Sequence[float]] = None,
               size_per_query: Optional[Sequence[int]] = None, extra: Sequence[str] = ()) -> List[dict]:
    if len(queries) == 0:
        return []
    return cs.run([int(q) for q in queries], [algo], int(k), theta="acc", time_limit=float(time_limit),
                  members=members, extra=list(extra), r=r, tag=tag, n_proc=int(n_proc),
                  r_per_query=r_per_query, size_per_query=size_per_query)


def save_instance(out_dir, ds: Dict, X=None, theta=None, tag: str = "default", meta: Dict = None,
                  inst_dir=None) -> str:
    """instance.json for the P0 kernels (E1 and the E3 harness). The graph and the default
    positions are converted once per dataset into inst_dir (default: <out_dir>/../inst);
    a setting only adds its tolerances (and its positions if they differ)."""
    from .formats import write_instance_json

    out_dir = Path(out_dir)
    inst_dir = Path(inst_dir) if inst_dir is not None else out_dir.parent / "inst"
    inst_dir.mkdir(parents=True, exist_ok=True)
    ip, ix = inst_dir / "indptr.npy", inst_dir / "indices.npy"
    if not (ip.exists() and ix.exists()):
        np.save(ip, np.asarray(ds["indptr"], dtype=np.int64))
        np.save(ix, np.asarray(ds["indices"], dtype=np.int32))
    if X is None or X is ds["X0"]:
        Xp = inst_dir / "X_default.npy"
        if not Xp.exists():
            np.save(Xp, np.asarray(ds["X0"], dtype=np.float32))
    else:
        Xp = inst_dir / f"X_{tag}.npy"
        np.save(Xp, np.asarray(X, dtype=np.float32))
    Tp = inst_dir / f"theta_{tag}.npy"
    np.save(Tp, np.asarray(ds["theta0"] if theta is None else theta, dtype=np.float32))
    path = out_dir / "instance.json"
    write_instance_json(str(path), str(ip), str(ix), str(Xp), str(Tp),
                        meta={"dataset": ds["name"], "tag": tag, **(meta or {})})
    return str(path)


def kcore_components(indptr, indices, core, k: int):
    """Component id of every vertex in the k-core (-1 outside) and component sizes."""
    from . import ffi

    comp, _ = ffi.components(np.asarray(indptr), np.asarray(indices), (np.asarray(core) >= k).astype(np.uint8))
    sizes = np.bincount(comp[comp >= 0]) if np.any(comp >= 0) else np.zeros(0, np.int64)
    return comp, sizes


def env_info() -> Dict:
    return {"cwd": os.getcwd(), "lacs_importable": _lacs_ok()}


def _lacs_ok() -> bool:
    try:
        import lacs  # noqa: F401

        return True
    except ImportError:
        return False
