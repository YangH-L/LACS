"""E3: efficiency and scalability of the search algorithms (C++ engine).

Datasets are listed under `efficiency.datasets` in the config. Each entry is one of
  {name, from_run: runs/<name>}              graph + learned positions/latitudes of a finished run
  {name, graph_bin: path}                    LACSGR01 graph
  {name, edges: path.txt}                    edge list, converted with lacs_convert
  {name, bipartite: path.txt, max_right_degree: D}   meta-path projection with lacs_project
Graphs without learned attributes get synthetic positions (random vectors smoothed
over the graph, i.e. structure-aligned) and lognormal latitudes centred on the
median edge distance; `theta_sigma` can be copied from a run (std of log theta).

Sweeps (one parameter at a time around the defaults): k, vertex fraction,
position dimension (synthetic positions only) and latitude scale.

Synthetic attributes (`attr_model`), for graphs without learned or real attributes:
  communities (default) groups = size-capped label-propagation communities
           (cpp/src/lacs_lpa.cpp, at most `community_max_size` members), i.e. homophily:
           members of the same social circle hold nearby positions. Each group gets a
           centre (uniform in a box, about `city_spacing` x the intra-group scale apart)
           and its members unit Gaussian spread per coordinate around it. Tolerances are
           theta_v = s * sqrt(2 * dim) * LogNormal(sigma), sqrt(2 * dim) being the typical
           distance between two members of a group. Queries are drawn from members of
           groups with >= `query_min_community` members (smaller groups cannot hold a
           non-trivial community).
  voronoi  same, but groups are BFS-Voronoi cells around n / nodes_per_city random seeds.
           Only suitable for graphs whose edges are spatially local; on social graphs most
           edges cross cells and no k-core survives inside a cell.
  smoothed random vectors smoothed over the graph (3 steps); theta_v = s * m_e *
           LogNormal(sigma), m_e the median edge distance. No local regime on social graphs
           (s <= 1: almost every C-LCC is empty; s >= 1.5: the candidate region is most of
           the graph), as the round-2 pilot showed; kept for comparison only.
`theta_scale_default: auto` picks s on a short calibration run (see calibrate_scale);
`theta_scales` lists absolute values of s, or multiples of the default s when
`theta_scales_relative: true`.

Settings run in priority order: default k, ablations, other k, vertex fractions, theta
scales, dimensions. With `dataset_budget_h` > 0, settings that would start after a dataset
has used up its budget are skipped (listed in e3.json -> datasets_info -> skipped_settings).

A dataset entry may carry `theta_scale_fixed: s` to use scale s after the (still executed)
calibration, which keeps every random draw, and hence every query, of an earlier run.

v0.5 (W2). The engine reports `ub` (an upper bound on OPT; the size itself when optimal) and runs
LatSweep-Exact best-first over thresholds by default; the ablation variants below switch single
components off. `theta_scales_extra` lists further tolerance scales run after the dimension sweep, so
that the random draws of all earlier settings stay those of round 3. Dataset options for very large
graphs: `lowmem: true` converts the edge list in two passes (cpp/src/lacs_convert.cpp --lowmem), and
`light: true` never loads the graph into Python (only its header; the engine reads the file), which
rules out the vertex-fraction sweep for that dataset. A converted graph is reused when the edge list
has not changed since (work/e3_<name>/graph.src.json). Further dataset options: `override: {...}` replaces
keys of the `efficiency` block for this dataset only (e.g. its algorithms or calibration grid);
`batch_algos: true` runs the algorithms without extra flags in one engine process per setting, so a
very large graph is loaded and indexed once instead of once per algorithm; `expect_m: m` logs a
warning (and records it in e3.json) when the converted graph does not have m edges, e.g. after a
truncated download.

Every dataset has its own random stream (seed, dataset name). When a dataset finishes, its
rows are checkpointed in work/e3_<name>/; with `resume: true` a rerun of the same
configuration skips finished datasets (the checkpoint stores a fingerprint of the
configuration, E3_VERSION and the input files, and is ignored if any of them changed).
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..graph.io import read_graph, write_graph, write_matrix
from ..model.sjtlat import SJTLat
from ..search.cpp import CppSearch, convert_edges, lpa_labels, project_bipartite
from ..utils import dump_json

LOG = logging.getLogger("lacs.e3")

_PRUNE_OFF = ["--no-ub-order", "--no-color-bound", "--no-seed", "--no-compat-bound", "--no-edge-filter"]
ALGO_FLAGS = {
    # ablations of LatSweep-Exact (results must equal `exact` whenever both finish)
    "exact_noindex": ("exact", ["--no-index"]),            # without the compatibility-core index
    "exact_nolocal": ("exact", ["--no-local"]),            # without query-local candidates
    "exact_noreduce": ("exact", ["--no-index", "--no-local"]),
    "exact_basic": ("exact", _PRUNE_OFF + ["--no-index", "--no-local"]),  # no reduction, no pruning
    "ballcore_r_nolocal": ("ballcore_r", ["--no-local"]),
    # v0.5 (W2): one component at a time
    "exact_v04": ("exact", ["--tau-order", "count", "--color-order", "index", "--max-exact-size", "3000"]),  # v0.4
    "exact_count": ("exact", ["--tau-order", "count"]),        # thresholds by decreasing |S(tau)| (v0.4 order)
    "exact_indexcolor": ("exact", ["--color-order", "index"]),  # colouring in index order (v0.4)
    "exact_noorder": ("exact", ["--no-ub-order"]),             # main sweep in index order, no early stopping
    "exact_noseed": ("exact", ["--no-seed"]),                   # no BallCore-F / greedy incumbent
    "exact_nocolorbound": ("exact", ["--no-color-bound"]),     # without pruning rule (3)
    "exact_nocompat": ("exact", ["--no-compat-bound"]),        # without rule (2), compatibility degree
    "exact_noedgefilter": ("exact", ["--no-edge-filter"]),     # rule (1) peels G instead of G minus conflicts
    "exact_cap20000": ("exact", ["--max-exact-size", "20000"]),  # search candidate sets of up to 20,000 (default 3,000)
}


def smoothed_positions(A: sp.csr_matrix, dim: int, rng, steps: int = 3) -> np.ndarray:
    X = rng.normal(size=(A.shape[0], dim))
    deg = np.asarray(A.sum(axis=1)).ravel()
    Dinv = sp.diags(1.0 / np.maximum(deg, 1))
    P = Dinv @ A
    for _ in range(steps):
        X = 0.5 * X + 0.5 * (P @ X)
    X = (X - X.mean(0)) / np.maximum(X.std(0), 1e-12)
    return X.astype(np.float32)


def city_labels(A: sp.csr_matrix, rng, nodes_per_city: int = 1000) -> np.ndarray:
    """City of every vertex: the nearest of C random seeds by hop distance (multi-source BFS
    through a virtual source attached to all seeds; labels follow the BFS tree, so every
    city is connected). Vertices in components without a seed get a random city."""
    from scipy.sparse.csgraph import breadth_first_order

    n = A.shape[0]
    C = max(1, n // max(1, int(nodes_per_city)))
    seeds = rng.choice(n, size=C, replace=False)
    S = sp.csr_matrix((np.ones(C, dtype=A.dtype), (np.full(C, n), seeds)), shape=(n + 1, n + 1))
    Ab = sp.bmat([[A, None], [None, sp.csr_matrix((1, 1), dtype=A.dtype)]], format="csr")
    order, pred = breadth_first_order(Ab + S + S.T, n, directed=False, return_predecessors=True)
    label = np.full(n + 1, -1, dtype=np.int64)
    label[seeds] = np.arange(C)
    p = pred.astype(np.int64)
    p[p < 0] = n
    p[seeds] = seeds  # seeds are their own roots
    p[n] = n
    unknown = np.flatnonzero(label[:n] < 0)
    reached = np.zeros(n + 1, bool)
    reached[order] = True
    unknown = unknown[reached[unknown]]
    while len(unknown):  # pointer jumping along the BFS tree towards the seeds
        label[unknown] = label[p[unknown]]
        p[unknown] = p[p[unknown]]
        unknown = unknown[label[unknown] < 0]
    lab = label[:n]
    miss = lab < 0
    lab[miss] = rng.integers(0, C, size=int(miss.sum()))
    return lab


def community_labels(A: sp.csr_matrix, work: Path, max_size: int = 1000, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Size-capped label propagation communities (one int label per vertex)."""
    work.mkdir(parents=True, exist_ok=True)
    gpath = work / "graph.bin"
    if not gpath.exists():
        write_graph(gpath, A)
    return lpa_labels(gpath, work / "lpa_labels.bin", max_size=max_size, seed=seed)


def city_positions(labels: np.ndarray, dim: int, rng, spacing: float = 8.0, chunk: int = 0) -> np.ndarray:
    """Centres uniform in a box sized so that neighbouring centres are about `spacing` x the
    intra-city scale sqrt(2 * dim) apart; members get unit Gaussian spread per coordinate.
    chunk > 0 (light datasets) draws the spread in blocks of rows, which gives the same numbers
    with far less memory (numpy's Generator draws normals sequentially)."""
    C = int(labels.max()) + 1
    unit = np.sqrt(2.0 * dim)
    side = spacing * unit * C ** (1.0 / dim)
    centres = rng.uniform(0.0, side, size=(C, dim))
    if chunk <= 0:
        X = centres[labels] + rng.normal(size=(len(labels), dim))
        return X.astype(np.float32)
    X = np.empty((len(labels), dim), dtype=np.float32)
    for a in range(0, len(labels), chunk):
        b = min(len(labels), a + chunk)
        X[a:b] = centres[labels[a:b]] + rng.normal(size=(b - a, dim))
    return X


def edge_median_distance(A: sp.csr_matrix, X: np.ndarray, rng, n: int = 200000) -> float:
    r, c = sp.triu(A, 1).nonzero()
    if len(r) == 0:
        return 1.0
    idx = rng.choice(len(r), size=min(n, len(r)), replace=False)
    return float(np.median(np.linalg.norm(X[r[idx]] - X[c[idx]], axis=1)))


class GraphStub:
    """Header-only stand-in for a LACSGR01 graph too large to load into Python (light datasets):
    it has the shape and nnz the experiment needs; the engine reads the file itself."""

    def __init__(self, path: str | Path):
        with open(path, "rb") as f:
            if f.read(8) != b"LACSGR01":
                raise ValueError(f"bad graph magic in {path}")
            n = int(np.frombuffer(f.read(8), dtype=np.int64)[0])
            self.nnz = int(np.frombuffer(f.read(8), dtype=np.int64)[0])
        self.shape = (n, n)
        self.path = Path(path)


def _converted(ds: dict, work: Path) -> Path:
    """Convert the edge list once; reuse work/graph.bin while the edge list is unchanged."""
    src = Path(ds["edges"])
    st = src.stat()
    stamp = {"edges": str(src.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns,
             "lowmem": bool(ds.get("lowmem", False))}
    gbin, sfile = work / "graph.bin", work / "graph.src.json"
    if gbin.exists() and sfile.exists():
        try:
            if json.loads(sfile.read_text()) == stamp:
                LOG.info("%s: reusing converted graph %s", ds["name"], gbin)
                return gbin
        except ValueError:
            pass
    sfile.unlink(missing_ok=True)
    t0 = time.time()
    info = convert_edges(src, gbin, work / "ids.txt", lowmem=bool(ds.get("lowmem", False)))
    LOG.info("%s: converted edge list %s in %.0f s", ds["name"], info, time.time() - t0)
    sfile.write_text(json.dumps(stamp))
    return gbin


def load_dataset(ds: dict, work: Path, rng) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    out = {"name": ds["name"], "learned": False}
    if ds.get("from_run"):
        root = Path(ds["from_run"])
        variant = ds.get("variant", "full")
        A = read_graph(root / "graph" / "graph.bin")
        out["graph_bin"] = root / "graph" / "graph.bin"
        model = SJTLat.load(root / "models" / f"sjt_{variant}.npz")
        lat = pd.read_csv(root / "results" / "latitudes.csv", dtype={"user": str})
        out.update({"A": A, "pos": model.project(np.load(root / "features" / "positions_val.npy")).astype(np.float32),
                    "theta": lat[f"theta_acc_{variant}"].to_numpy(np.float32), "learned": True})
        return out
    if ds.get("geo_run"):  # real homes (km) and travel tolerances from the meetup application
        root = Path(ds["geo_run"])
        users = pd.read_csv(root / "geo" / "users.csv")
        col = ds.get("tolerance")
        gates = root / "results" / "geo_gates.json"
        if not col and gates.exists():
            col = json.loads(gates.read_text()).get("G2", {}).get("tolerance")
        col = col or [c for c in users.columns if c.startswith("theta_q")][0]
        out.update({"A": read_graph(root / "graph" / "graph.bin"), "pos": np.load(root / "geo" / "positions.npy"),
                    "theta": users[col].to_numpy(np.float32), "learned": True,
                    "graph_bin": root / "graph" / "graph.bin", "tolerance": col})
        return out
    if ds.get("graph_bin"):
        A = read_graph(ds["graph_bin"])
        out["graph_bin"] = Path(ds["graph_bin"])
    elif ds.get("edges"):
        gbin = _converted(ds, work)
        A = GraphStub(gbin) if ds.get("light") else read_graph(gbin)
        out["graph_bin"] = gbin
    elif ds.get("bipartite"):
        info = project_bipartite(ds["bipartite"], work / "graph.bin", work / "ids.txt",
                                 int(ds.get("max_right_degree", 0)), int(ds.get("min_shared", 1)))
        LOG.info("%s: projected bipartite graph %s", ds["name"], info)
        A = read_graph(work / "graph.bin")
        out["graph_bin"] = work / "graph.bin"
    else:
        raise ValueError(f"dataset {ds['name']}: need from_run, graph_bin, edges or bipartite")
    out["A"] = A
    return out


def synth_attrs(A, dim, scale, sigma, rng, model: str = "communities", labels: np.ndarray | None = None,
                spacing: float = 8.0, chunk: int = 0):
    if model in ("communities", "voronoi"):
        X = city_positions(labels, dim, rng, spacing, chunk)
        base = np.sqrt(2.0 * dim)
    elif model == "smoothed":
        X = smoothed_positions(A, dim, rng)
        base = edge_median_distance(A, X, rng)
    else:
        raise ValueError(f"unknown attr_model {model}")
    theta = base * scale * np.exp(sigma * rng.normal(size=A.shape[0]) - 0.5 * sigma ** 2)
    return X, theta.astype(np.float32)


def calibrate_scale(A, X, th_unit, k, scales, target, n_queries, rng, work: Path, n_proc: int = 1,
                    time_limit: float = 2.0, allowed: np.ndarray | None = None, graph_bin=None,
                    pos_bin=None, refine: int = 2) -> tuple[float, list]:
    """Smallest scale s whose median consensus-type community has >= target members, measured
    with LatSweep-Exact under a short time limit (its incumbent is feasible, so a lower bound on
    OPT). BallCore-F is cheaper but far too conservative for this (median 0 even when OPT is in
    the thousands on clustered graphs).

    The same queries are used at every scale (OPT can only grow with s, so the medians are
    comparable). In high dimension the median jumps steeply between two grid values (16-d:
    about 8 at s = 1.5 and 77 at s = 2 on a heavy-tailed test graph), so after the first
    passing grid value `refine` bisection steps between it and the last failing value narrow
    the choice."""
    table = []
    state = {"qs": None}

    def evaluate(sc: float):
        cs = CppSearch(A, X, {"acc": (th_unit * sc).astype(np.float32)},
                       work / f"calib{sc:.4g}".replace(".", "p"), graph_bin=graph_bin, pos_bin=pos_bin)
        if state["qs"] is None:
            ok = cs.core_numbers() >= k  # core numbers of the graph: the same at every scale
            if allowed is not None:
                ok &= allowed
            cand = np.flatnonzero(ok)
            if len(cand) == 0:
                return None
            state["qs"] = rng.choice(cand, size=min(n_queries, len(cand)), replace=False)
        rows = cs.run(state["qs"], ["exact"], k, theta="acc", time_limit=time_limit, members=False, tag="calib",
                      n_proc=n_proc)
        sizes = np.array([r["size"] for r in rows], float)
        table.append({"scale": float(sc), "exact_incumbent_median": float(np.median(sizes)),
                      "nonempty": float((sizes > 0).mean()),
                      "timeout_rate": float(np.mean([r["status"] == "timeout" for r in rows])),
                      "region_median": float(np.median([r["s0_size"] for r in rows]))})
        LOG.info("calibration: scale %.4g -> median community size %.1f", sc, table[-1]["exact_incumbent_median"])
        return table[-1]["exact_incumbent_median"]

    lo = hi = None
    for sc in scales:
        med = evaluate(float(sc))
        if med is None:
            break
        if med >= target:
            hi = float(sc)
            break
        lo = float(sc)
    if hi is None:
        return float(scales[-1]), table
    if lo is not None:
        for _ in range(int(refine)):
            mid = round((lo + hi) / 2, 4)
            med = evaluate(mid)
            if med is not None and med >= target:
                hi = mid
            else:
                lo = mid
    return hi, table


def run_setting(cs: CppSearch, core: np.ndarray, theta: np.ndarray, k: int, algos, n_queries, time_limit,
                tag: str, rng, n_proc: int = 1, qs_fixed=None, record=None, allowed=None,
                batch: bool = False) -> list[dict]:
    if qs_fixed is not None:
        qs = np.asarray(qs_fixed)
    else:
        if core is None:
            core = cs.core_numbers()
        ok = core >= k
        if allowed is not None:
            ok &= allowed
        cand = np.flatnonzero(ok)
        if len(cand) == 0:
            return []
        qs = rng.choice(cand, size=min(n_queries, len(cand)), replace=False)
    if record is not None:
        record[k] = qs
    rows = []
    if batch:  # one engine process for all algorithms without extra flags (same rows, one graph load)
        plain = [a for a in algos if not ALGO_FLAGS.get(a, (a, []))[1]]
        if plain:
            r = float(np.median(theta)) if any(a.startswith("kr_") for a in plain) else None
            for row in cs.run(qs, plain, k, theta="acc", time_limit=time_limit, members=False, r=r,
                              tag=f"{tag}_batch", n_proc=n_proc):
                rows.append(row)
        algos = [a for a in algos if a not in plain]
    for a in algos:
        algo, flags = ALGO_FLAGS.get(a, (a, []))
        r = float(np.median(theta)) if algo.startswith("kr_") else None
        for row in cs.run(qs, [algo], k, theta="acc", time_limit=time_limit, members=False, extra=flags, r=r,
                          tag=f"{tag}_{a}", n_proc=n_proc):
            row["algo"] = a
            rows.append(row)
    return rows


E3_VERSION = "0.5"  # bump whenever a dataset's results would change, so stale checkpoints are not resumed


def _ds_rng(seed, name: str) -> np.random.Generator:
    """Random stream of one dataset: its queries and synthetic attributes do not depend on which
    other datasets are listed or in what order (this is what makes resuming safe)."""
    return np.random.default_rng([int(seed), zlib.crc32(str(name).encode("utf-8"))])


def _input_stamp(ds: dict) -> list:
    """Size and modification time of the files a dataset is built from."""
    files = [ds.get(k) for k in ("edges", "graph_bin", "bipartite") if ds.get(k)]
    if ds.get("geo_run"):
        r = Path(ds["geo_run"])
        files += [r / "geo" / "users.csv", r / "geo" / "positions.npy", r / "graph" / "graph.bin",
                  r / "results" / "geo_gates.json"]
    if ds.get("from_run"):
        r = Path(ds["from_run"])
        files += [r / "graph" / "graph.bin", r / "results" / "latitudes.csv"]
    out = []
    for f in files:
        f = Path(f)
        st = f.stat() if f.exists() else None
        out.append([str(f), st.st_size if st else None, st.st_mtime_ns if st else None])
    return out


def _fingerprint(cfg: dict, e: dict, ds: dict) -> str:
    keep = {k: v for k, v in e.items() if k not in ("datasets", "resume")}
    blob = json.dumps({"version": E3_VERSION, "seed": cfg["seed"], "efficiency": keep, "dataset": ds,
                       "inputs": _input_stamp(ds)}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def run(cfg: dict, paths: dict) -> dict:
    e = cfg.get("efficiency") or {}
    if not e.get("datasets"):
        raise ValueError("efficiency.datasets is empty: see configs/efficiency.yaml")
    algos = e.get("algos", ["kcore", "ballcore_f", "ballcore_r", "greedy", "exact", "exact_basic"])
    k0, d0 = int(e.get("default_k", 3)), int(e.get("default_dim", 16))
    nq, tl = int(e.get("n_queries", 50)), float(e.get("time_limit", 60))
    sigma = float(e.get("theta_sigma", 0.4))
    s0_cfg = e.get("theta_scale_default", 1.0)
    model = e.get("attr_model", "communities")
    spacing = float(e.get("city_spacing", 8.0))
    resume = bool(e.get("resume", False))
    calibration, group_info, ds_info, failed = {}, {}, {}, {}
    frames = []
    e_global = e
    for ds in e_global["datasets"]:
        # v0.5: `override` replaces keys of the efficiency block for this dataset only
        e = {**e_global, **(ds.get("override") or {})}
        algos = e.get("algos", ["kcore", "ballcore_f", "ballcore_r", "greedy", "exact", "exact_basic"])
        k0, d0 = int(e.get("default_k", 3)), int(e.get("default_dim", 16))
        nq, tl = int(e.get("n_queries", 50)), float(e.get("time_limit", 60))
        sigma = float(e.get("theta_sigma", 0.4))
        s0_cfg = e.get("theta_scale_default", 1.0)
        model = e.get("attr_model", "communities")
        spacing = float(e.get("city_spacing", 8.0))
        resume = bool(e.get("resume", False))
        batch = bool(ds.get("batch_algos", False))
        name = ds["name"]
        work = paths["work"] / f"e3_{name}"
        work.mkdir(parents=True, exist_ok=True)
        ck_json, ck_rows = work / "checkpoint.json", work / "checkpoint_rows.csv"
        fp = _fingerprint(cfg, e, ds)
        if resume and ck_json.exists() and ck_rows.exists():
            ck = json.loads(ck_json.read_text())
            if ck.get("fingerprint") == fp:
                frames.append(pd.read_csv(ck_rows))
                if ck.get("calibration"):
                    calibration[name] = ck["calibration"]
                if ck.get("groups"):
                    group_info[name] = ck["groups"]
                ds_info[name] = {**ck.get("info", {}), "resumed": True}
                LOG.info("E3 %s: resumed from checkpoint (%d rows)", name, len(frames[-1]))
                continue
            LOG.warning("E3 %s: checkpoint belongs to another configuration or version; recomputing", name)
        try:
            rng = _ds_rng(cfg["seed"], name)
            t_ds = time.time()
            rows: list[dict] = []
            D = load_dataset(ds, work, rng)
            A = D["A"]
            LOG.info("E3 %s: n=%d m=%d learned=%s", name, A.shape[0], A.nnz // 2, D["learned"])
            m_expected = ds.get("expect_m")
            if m_expected is not None and int(m_expected) != A.nnz // 2:
                LOG.warning("E3 %s: graph has %d edges, expected %d (truncated or different edge list?)", name,
                            A.nnz // 2, int(m_expected))

            allowed = None
            labels = None
            gb = D.get("graph_bin")  # engine instances on the full graph share this file
            pos0 = work / "pos_default.bin"  # ... and the default positions
            if D["learned"]:
                X0, th_unit = D["pos"], D["theta"]
                write_matrix(pos0, X0)
                s_def = 1.0
            else:
                if model == "communities":
                    labels, info = community_labels(A, work, int(e.get("community_max_size", 1000)), int(cfg["seed"]))
                    labels = labels.astype(np.int64)
                    LOG.info("%s: label propagation %s", name, info)
                    group_info[name] = info
                elif model == "voronoi":
                    labels = city_labels(A, rng, int(e.get("nodes_per_city", 1000)))
                if labels is not None:
                    sizes = np.bincount(labels)
                    allowed = sizes[labels] >= int(e.get("query_min_community", 20))
                    LOG.info("%s: %d groups, median size %d, %.1f%% of vertices in groups >= %d", name, len(sizes),
                             int(np.median(sizes)), 100 * allowed.mean(), int(e.get("query_min_community", 20)))
                X0, th_unit = synth_attrs(A, d0, 1.0, sigma, rng, model, labels, spacing,
                                          chunk=int(ds.get("position_chunk", 1_000_000)) if ds.get("light") else 0)
                write_matrix(pos0, X0)
                if s0_cfg == "auto":
                    default_scales = ([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0] if model in ("communities", "voronoi")
                                      else [1.0, 1.25, 1.5, 2.0, 3.0])
                    s_def, table = calibrate_scale(A, X0, th_unit, k0, e.get("calibration_scales", default_scales),
                                                   float(e.get("calibration_target", 10)),
                                                   int(e.get("calibration_queries", 30)), rng, work,
                                                   int(e.get("n_proc", 1)), float(e.get("calibration_time_limit", 2.0)),
                                                   allowed, graph_bin=gb, pos_bin=pos0,
                                                   refine=int(e.get("calibration_refine", 2)))
                    calibration[name] = {"chosen": s_def, "table": table, "attr_model": model}
                else:
                    s_def = float(s0_cfg)
                if ds.get("theta_scale_fixed") is not None:
                    # calibration still runs (it draws its queries from the dataset's random stream, so every
                    # later draw stays the same), but the given scale is used, e.g. to re-time an earlier run
                    s_def = float(ds["theta_scale_fixed"])
                    calibration.setdefault(name, {})["used"] = s_def
            th0 = (th_unit * s_def).astype(np.float32)

            base_queries: dict = {}
            st = {"cs0": None, "core0": None}

            def emit(As, X, th, sweep, value, k, frac, scale, tag, cs=None, core=None, mask=None, gbin=None,
                     pbin=None):
                if cs is None:
                    cs = CppSearch(As, X, {"acc": th}, work / tag, graph_bin=gbin, pos_bin=pbin)
                    core = cs.core_numbers()
                rec = base_queries if (sweep == "k" and tag == "base") else None
                for row in run_setting(cs, core, th, k, algos, nq, tl, f"{tag}_k{k}", rng,
                                       int(e.get("n_proc", 1)), record=rec, allowed=allowed if mask is None else mask,
                                       batch=batch):
                    rows.append({"dataset": name, "sweep": sweep, "value": value, "k": k, "fraction": frac,
                                 "dim": int(X.shape[1]), "theta_scale": scale, "n": As.shape[0], "m": As.nnz // 2,
                                 **{c: v for c, v in row.items() if c != "members"}})
                LOG.info("E3 %s %s=%s done", name, sweep, value)
                return cs, core

            def job_k(kk):  # same graph and attributes for every k
                st["cs0"], st["core0"] = emit(A, X0, th0, "k", kk, kk, 1.0, s_def, "base", st["cs0"], st["core0"],
                                              gbin=gb, pbin=pos0)

            def job_ablation(abl):  # exact without index / local candidates / pruning, default setting only
                cs = st["cs0"] if st["cs0"] is not None else CppSearch(A, X0, {"acc": th0}, work / "base",
                                                                         graph_bin=gb, pos_bin=pos0)
                for row in run_setting(cs, st["core0"], th0, k0, abl, nq, tl, f"abl_k{k0}", rng,
                                       int(e.get("n_proc", 1)), qs_fixed=base_queries.get(k0), allowed=allowed):
                    rows.append({"dataset": name, "sweep": "ablation", "value": k0, "k": k0, "fraction": 1.0,
                                 "dim": int(X0.shape[1]), "theta_scale": s_def, "n": A.shape[0], "m": A.nnz // 2,
                                 **{c: v for c, v in row.items() if c != "members"}})
                LOG.info("E3 %s ablations done", name)

            def job_fraction(f):
                if f < 1.0 and ds.get("light"):
                    LOG.warning("E3 %s: fraction %s skipped (light dataset: graph not loaded in Python)", name, f)
                    return
                if f >= 1.0:
                    st["cs0"], st["core0"] = emit(A, X0, th0, "fraction", f, k0, 1.0, s_def, "base", st["cs0"],
                                                  st["core0"], gbin=gb, pbin=pos0)
                    return
                idx = np.sort(rng.choice(A.shape[0], size=int(f * A.shape[0]), replace=False))
                emit(A[idx][:, idx].tocsr(), X0[idx], th0[idx], "fraction", f, k0, f, s_def,
                     f"frac{f}".replace(".", "p"), mask=None if allowed is None else allowed[idx])

            def job_theta(sc):
                s_abs = sc * s_def if rel else sc
                emit(A, X0, (th_unit * s_abs).astype(np.float32), "theta_scale", sc, k0, 1.0, s_abs,
                     f"scale{s_abs:.4g}".replace(".", "p"), gbin=gb, pbin=pos0)

            def job_dim(dd):
                X, th = synth_attrs(A, dd, s_def, sigma, rng, model, labels, spacing)
                emit(A, X, th, "dim", dd, k0, 1.0, s_def, f"dim{dd}", gbin=gb)

            # priority order; once the dataset's time budget is used up, the remaining settings are skipped
            rel = bool(e.get("theta_scales_relative", False))
            ks = [int(x) for x in e.get("ks", [2, 3, 4, 5, 6])]
            abl = list(e.get("ablation_algos", []))
            jobs = [(f"k={k0}", lambda: job_k(k0))] if k0 in ks else []
            if abl:
                jobs.append(("ablation", lambda: job_ablation(abl)))
            jobs += [(f"k={kk}", (lambda kk=kk: job_k(kk))) for kk in ks if kk != k0]
            jobs += [(f"fraction={f}", (lambda f=f: job_fraction(f))) for f in e.get("fractions", [0.2, 0.4, 0.6, 0.8, 1.0])]
            jobs += [(f"theta_scale={sc}", (lambda sc=sc: job_theta(sc))) for sc in e.get("theta_scales", [0.5, 1.0, 2.0])]
            if not D["learned"]:
                jobs += [(f"dim={dd}", (lambda dd=dd: job_dim(dd))) for dd in e.get("dims", [4, 8, 16, 32, 64])]
            # v0.5: further scales after everything else, so that earlier random draws are unchanged
            jobs += [(f"theta_scale={sc}", (lambda sc=sc: job_theta(sc))) for sc in e.get("theta_scales_extra", [])]
            budget_s = float(e.get("dataset_budget_h", 0) or 0) * 3600.0
            skipped = []
            for label, job in jobs:
                if budget_s > 0 and time.time() - t_ds > budget_s:
                    skipped.append(label)
                    continue
                job()
            if skipped:
                LOG.warning("E3 %s: time budget of %.1f h used up, skipped %s", name, budget_s / 3600, skipped)
            df_ds = pd.DataFrame(rows)
            info = {"n": int(A.shape[0]), "m": int(A.nnz // 2), "m_expected": m_expected, "learned": bool(D["learned"]),
                    "theta_scale_default": float(s_def), "wall_s": round(time.time() - t_ds, 1),
                    "budget_h": budget_s / 3600.0, "skipped_settings": skipped}
            ds_info[name] = info
            df_ds.to_csv(ck_rows, index=False)  # checkpoint: rows first, then the marker that names them
            dump_json({"fingerprint": fp, "calibration": calibration.get(name), "groups": group_info.get(name),
                       "info": info}, ck_json)
            frames.append(df_ds)
            LOG.info("E3 %s finished in %.0f s (%d rows)", name, info["wall_s"], len(df_ds))
        except Exception as ex:  # keep going with the next dataset; a rerun retries this one
            LOG.exception("E3 %s failed", name)
            failed[name] = f"{type(ex).__name__}: {ex}"
    e = e_global
    algos = e.get("algos", ["kcore", "ballcore_f", "ballcore_r", "greedy", "exact", "exact_basic"])
    k0 = int(e.get("default_k", 3))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.to_csv(paths["results"] / "e3_per_query.csv", index=False)
    summ = summarize(df)
    summ.to_csv(paths["results"] / "e3_summary.csv", index=False)
    _plot(summ, paths)
    cons = consistency(df, k0)
    if cons["contradictions"]:
        LOG.warning("E3 consistency: %s", cons)
    if failed:
        LOG.error("E3 failed datasets: %s", failed)
    dump_json({"version": E3_VERSION, "datasets": [d["name"] for d in e["datasets"]], "algos": algos,
               "rows": len(df), "datasets_info": ds_info, "failed": failed, "calibration": calibration,
               "groups": group_info, "consistency": cons}, paths["results"] / "e3.json")
    return {"rows": len(df), "consistency": cons, "failed": failed}


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    for c, v in (("index_ms", 0.0), ("index_edges", -1), ("ub", -1), ("ub_ms", 0.0)):
        if c not in df:
            df[c] = v
    # v0.5: ratio of the returned size to the reported upper bound on OPT, for answers whose
    # optimality is not proven (timeout / too_large); 1 when the bound is 0
    open_ = df["status"].isin(["timeout", "too_large"]) & (df["ub"] >= 0)
    df["open_ratio"] = np.where(open_, df["size"] / df["ub"].clip(lower=1), np.nan)
    df.loc[open_ & (df["ub"] == 0), "open_ratio"] = 1.0
    summ = (df.groupby(["dataset", "sweep", "value", "algo"])
            .agg(queries=("q", "size"), time_ms_mean=("time_ms", "mean"), time_ms_median=("time_ms", "median"),
                 time_ms_p90=("time_ms", lambda s: float(np.percentile(s, 90))),
                 size_mean=("size", "mean"), nonempty_rate=("size", lambda s: (s > 0).mean()),
                 timeout_rate=("status", lambda s: (s == "timeout").mean()),
                 optimal_rate=("status", lambda s: (s == "optimal").mean()),
                 too_large_rate=("status", lambda s: (s == "too_large").mean()),
                 region_median=("s0_size", "median"), index_ms=("index_ms", "max"),
                 index_edges=("index_edges", "max"), m=("m", "max"),
                 tau_eval_mean=("n_tau_eval", "mean"), bb_nodes_mean=("bb_nodes", "mean"),
                 open_ratio_median=("open_ratio", "median"), open_ratio_min=("open_ratio", "min"),
                 ub_ms_max=("ub_ms", "max"))
            .reset_index())
    summ["index_edge_frac"] = np.where(summ["index_edges"] >= 0, summ["index_edges"] / summ["m"].clip(lower=1),
                                       np.nan)
    return summ


EXACT_VARIANTS = ("exact", "exact_noindex", "exact_nolocal", "exact_noreduce", "exact_basic", "exact_v04",
                  "exact_count", "exact_indexcolor", "exact_noorder", "exact_noseed", "exact_nocolorbound",
                  "exact_nocompat", "exact_noedgefilter", "exact_cap20000")
FEASIBLE_ALGOS = EXACT_VARIANTS + ("greedy", "ballcore_f")


def consistency(df: pd.DataFrame, k0: int) -> dict:
    """Per query: every exact variant that reports `optimal` must return the same size, and no
    always-feasible algorithm may find a larger feasible community than a proven optimum."""
    if df.empty or "status" not in df:
        return {"checked": 0, "contradictions": 0}
    d = df[df["algo"].isin(FEASIBLE_ALGOS)].copy()
    d["grp"] = d["sweep"].astype(str) + ":" + d["value"].astype(str)
    d.loc[d["sweep"].isin(["k", "ablation"]) & (d["k"] == k0), "grp"] = "base"
    checked = bad = 0
    examples = []
    for (ds, grp, q), g in d.groupby(["dataset", "grp", "q"]):
        opt = g[g["algo"].isin(EXACT_VARIANTS) & (g["status"] == "optimal")]
        if opt.empty:
            continue
        checked += 1
        lo, hi = int(opt["size"].min()), int(opt["size"].max())
        feas = g[g["feasible"].astype(bool)]
        best_feasible = int(feas["size"].max()) if len(feas) else 0
        if hi != lo or best_feasible > lo:
            bad += 1
            if len(examples) < 5:
                examples.append({"dataset": ds, "setting": grp, "optimal_sizes": sorted(set(opt["size"].astype(int))),
                                 "best_feasible": best_feasible})
    return {"checked": int(checked), "contradictions": int(bad), "examples": examples}


def _plot(summ: pd.DataFrame, paths) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for ds, g in summ.groupby("dataset"):
            sweeps = list(g["sweep"].unique())
            fig, axes = plt.subplots(1, len(sweeps), figsize=(4 * len(sweeps), 3.2), squeeze=False)
            for ax, sw in zip(axes[0], sweeps):
                h = g[g["sweep"] == sw]
                for algo, hh in h.groupby("algo"):
                    hh = hh.sort_values("value")
                    ax.plot(hh["value"], hh["time_ms_mean"], marker="o", label=algo)
                ax.set_yscale("log")
                ax.set_xlabel(sw)
                ax.set_ylabel("mean query time (ms)")
            handles, labels = axes[0][0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
            fig.suptitle(f"E3 efficiency: {ds}")
            fig.tight_layout()
            fig.savefig(paths["figs"] / f"e3_{ds}.png", dpi=160, bbox_inches="tight")
            plt.close(fig)
    except Exception as e:
        LOG.warning("E3 plot failed: %s", e)
