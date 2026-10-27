"""E2 (effectiveness with external ground truth), E4 (ablations via @variant
methods), E5 (case study) and gate G3 (community sizes over k).

Ground truth = labelled judgements from the TEST period whose receiver and
source are both inside the community; the graph, positions and latitudes only
use data up to the end of the validation period.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from ..eval import dynamics
from ..graph.io import read_graph
from ..model.sjtlat import SJTLat
from ..search.cpp import CppSearch
from ..utils import dump_json

LOG = logging.getLogger("lacs.e2")

CPP_METHODS = {
    "lcc_c_exact": ("exact", "acc"), "lcc_c_greedy": ("greedy", "acc"), "lcc_c_ballcore_f": ("ballcore_f", "acc"),
    "lcc_c_ballcore_r": ("ballcore_r", "acc"), "lcc_r_exact": ("exact", "rej"), "lcc_r_greedy": ("greedy", "rej"),
    "pairwise_c": ("pairwise_exact", "acc"), "kcore": ("kcore", "acc"), "kr_global": ("kr_exact", "acc"),
    "edgeacc": ("edgeacc", "acc"),
}
PY_METHODS = {"cluster_q", "random_bfs"}


# ----------------------------------------------------------------- inputs --

class Inputs:
    def __init__(self, cfg: dict, paths: dict, variant: str):
        self.variant = variant
        self.users = pd.read_csv(paths["feat"] / "users.csv", dtype={"user": str})
        self.A = read_graph(paths["graph"] / "graph.bin")
        lat = pd.read_csv(paths["results"] / "latitudes.csv", dtype={"user": str})
        model = SJTLat.load(paths["models"] / f"sjt_{variant}.npz")
        self.K = model.n_classes
        self.pos = model.project(np.load(paths["feat"] / "positions_val.npy")).astype(np.float32)
        self.th_acc = lat[f"theta_acc_{variant}"].to_numpy(dtype=np.float32)
        self.th_rej = lat[f"theta_rej_{variant}"].to_numpy(dtype=np.float32)
        thetas = {"acc": self.th_acc}
        if self.K == 3:
            thetas["rej"] = self.th_rej
        self.cpp = CppSearch(self.A, self.pos, thetas, paths["work"] / f"search_{variant}")
        G = nx.from_scipy_sparse_array(self.A)
        self.core = np.zeros(self.A.shape[0], dtype=int)
        for v, c in nx.core_number(G).items():
            self.core[v] = c
        deg = np.asarray(self.A.sum(axis=1)).ravel()
        self.deg = deg
        self.two_m = float(deg.sum())
        self.r_global = float(np.median(self.th_acc[deg > 0])) if (deg > 0).any() else 1.0


def test_pairs(paths) -> dict:
    F = np.load(paths["feat"] / "pairs.npz", allow_pickle=False)
    te = F["split"] == 2
    return {"recv": F["recv"][te], "src": F["src"][te], "y": F["y"][te]}


def sample_queries(inp: Inputs, k: int, n: int, pool: str, tp: dict, rng) -> np.ndarray:
    ok = inp.core >= k
    if pool == "test_active":
        act = np.zeros(len(ok), bool)
        both = ok[tp["recv"]] & ok[tp["src"]]
        act[tp["recv"][both]] = True
        act[tp["src"][both]] = True
        ok &= act
    cand = np.flatnonzero(ok)
    if len(cand) == 0:
        return cand
    return np.sort(rng.choice(cand, size=min(n, len(cand)), replace=False))


# ------------------------------------------------------- python baselines --

def kcore_component(A: sp.csr_matrix, members: np.ndarray, k: int, q: int) -> np.ndarray:
    S = np.unique(np.asarray(members, dtype=int))
    if q not in set(S.tolist()):
        return np.empty(0, int)
    sub = A[S][:, S].tocsr()
    alive = np.ones(len(S), bool)
    while True:
        deg = np.asarray(sub[alive][:, alive].sum(axis=1)).ravel() if alive.any() else np.empty(0)
        idx = np.flatnonzero(alive)
        drop = idx[deg < k]
        if len(drop) == 0:
            break
        alive[drop] = False
        if not alive.any():
            return np.empty(0, int)
    qi = int(np.searchsorted(S, q))
    if not alive[qi]:
        return np.empty(0, int)
    idx = np.flatnonzero(alive)
    _, lab = connected_components(sub[idx][:, idx], directed=False)
    qpos = int(np.searchsorted(idx, qi))
    return S[idx[lab == lab[qpos]]]


def random_connected(A: sp.csr_matrix, pool: np.ndarray, q: int, size: int, rng) -> np.ndarray:
    pool_set = set(pool.tolist())
    chosen = [q]
    in_set = {q}
    frontier = [u for u in A.indices[A.indptr[q]:A.indptr[q + 1]] if u in pool_set]
    while len(chosen) < size and frontier:
        i = int(rng.integers(len(frontier)))
        u = frontier.pop(i)
        if u in in_set:
            continue
        in_set.add(u)
        chosen.append(u)
        frontier += [w for w in A.indices[A.indptr[u]:A.indptr[u + 1]] if w in pool_set and w not in in_set]
    return np.array(sorted(chosen))


# ---------------------------------------------------------------- metrics --

def community_metrics(H: np.ndarray, inp: Inputs, tp: dict, sim_cfg: dict, k: int) -> dict:
    out = {"size": int(len(H))}
    if len(H) == 0:
        return out
    A = inp.A
    sub = A[H][:, H]
    e_in = sub.nnz / 2
    vol = float(inp.deg[H].sum())
    cut = vol - 2 * e_in
    out["density"] = float(2 * e_in / (len(H) * (len(H) - 1))) if len(H) > 1 else 0.0
    denom = min(vol, inp.two_m - vol)
    out["conductance"] = float(cut / denom) if denom > 0 else 0.0
    out["min_deg"] = int(np.asarray(sub.sum(axis=1)).min())
    X = inp.pos[H].astype(np.float64)
    ta, tr = inp.th_acc[H].astype(np.float64), inp.th_rej[H].astype(np.float64)
    out["min_theta_acc"] = float(ta.min())
    if len(H) <= 3000:
        G = X @ X.T
        sq = np.diag(G)
        D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0.0))
        iu = np.triu_indices(len(H), 1)
        out["diam"] = float(D.max())
        if len(iu[0]):
            out["frac_pairs_accept"] = float((D[iu] <= np.minimum(ta[iu[0]], ta[iu[1]])).mean())
            if inp.K == 3:
                out["frac_pairs_reject"] = float((D[iu] >= np.minimum(tr[iu[0]], tr[iu[1]])).mean())
        out["feasible_c"] = bool(out["diam"] <= ta.min() + 1e-6)
        if inp.K == 3:
            out["feasible_r"] = bool(out["diam"] <= tr.min() + 1e-6)
    mask = np.zeros(A.shape[0], bool)
    mask[H] = True
    inside = mask[tp["recv"]] & mask[tp["src"]]
    y = tp["y"][inside]
    out["n_lab"] = int(len(y))
    if len(y):
        out["n_acc"] = int((y == 0).sum())
        out["iar"] = float((y == 0).mean())
        if inp.K == 3:
            out["n_neu"] = int((y == 1).sum())
            out["n_rej"] = int((y == 2).sum())
            out["irr"] = float((y == 2).mean())
    if sim_cfg.get("enabled", True) and 1 < len(H) <= int(sim_cfg.get("max_size", 3000)):
        h = dynamics.hk(sub, X, ta, int(sim_cfg["max_steps"]), float(sim_cfg["tol"]))
        out.update({"hk_consensus": h["consensus"], "hk_steps": h["steps"], "hk_clusters": h["n_clusters"],
                    "hk_rejections_seen": h["edge_rejections_seen"]})
        if sim_cfg.get("external", True):
            nb = np.unique(np.concatenate([H, A[H].indices]))
            if len(nb) <= int(sim_cfg.get("max_external_size", 2000)):
                focus = np.isin(nb, H)
                he = dynamics.hk(A[nb][:, nb], inp.pos[nb].astype(np.float64), inp.th_acc[nb].astype(np.float64),
                                 int(sim_cfg.get("max_steps_external", 200)), float(sim_cfg["tol"]), focus=focus)
                out["hk_ext_consensus"] = he["consensus"]
        if inp.K == 3:
            s = dynamics.sjt(sub, X, ta, tr, float(sim_cfg["mu"]), float(sim_cfg["beta"]),
                             int(sim_cfg.get("max_steps_sjt", 300)), float(sim_cfg["tol"]))
            out.update({"sjt_rejections": s["rejection_events"], "sjt_diam_ratio": s["diam_final_ratio"]})
    return out


def _metrics_chunk(Hs, inp, tp, sim, k):
    return [community_metrics(H, inp, tp, sim, k) for H in Hs]


def _metrics_parallel(Hs: list, inp: "Inputs", tp: dict, sim: dict, k: int, n_jobs: int) -> list[dict]:
    if n_jobs <= 1 or len(Hs) < 2 * n_jobs:
        return _metrics_chunk(Hs, inp, tp, sim, k)
    from joblib import Parallel, delayed

    cpp, inp.cpp = inp.cpp, None  # the subprocess wrapper is not needed in workers
    try:
        order = np.argsort([-len(H) for H in Hs])  # spread the large communities
        chunks = [order[i::n_jobs] for i in range(n_jobs)]
        res = Parallel(n_jobs=n_jobs)(delayed(_metrics_chunk)([Hs[i] for i in c], inp, tp, sim, k) for c in chunks)
    finally:
        inp.cpp = cpp
    out = [None] * len(Hs)
    for c, part in zip(chunks, res):
        for i, m in zip(c, part):
            out[i] = m
    return out


def summarize(df: pd.DataFrame, K: int) -> pd.DataFrame:
    rows = []
    for (method, variant), g in df.groupby(["method", "variant"], sort=False):
        ne = g[g["size"] > 0]
        lab = g[g["n_lab"].fillna(0) > 0] if "n_lab" in g else g.iloc[0:0]
        row = {"method": method, "variant": variant, "queries": len(g), "nonempty_rate": len(ne) / max(len(g), 1),
               "size_mean": ne["size"].mean(), "size_median": ne["size"].median(),
               "coverage_labeled": len(lab), "time_ms_mean": g["time_ms"].mean() if "time_ms" in g else np.nan}
        if len(lab):
            row["iar_macro"] = lab["iar"].mean()
            row["iar_micro"] = lab["n_acc"].sum() / lab["n_lab"].sum()
            if K == 3:
                row["irr_macro"] = lab["irr"].mean()
                row["irr_micro"] = lab["n_rej"].sum() / lab["n_lab"].sum()
        for c in ("density", "conductance", "frac_pairs_accept", "frac_pairs_reject"):
            if c in ne:
                row[f"{c}_mean"] = ne[c].mean()
        for c in ("feasible_c", "feasible_r", "hk_consensus", "hk_ext_consensus"):
            if c in ne and ne[c].notna().any():
                row[f"{c}_rate"] = ne[c].dropna().astype(float).mean()
        if "sjt_rejections" in ne and ne["sjt_rejections"].notna().any():
            row["sjt_rejection_free_rate"] = (ne["sjt_rejections"].dropna() == 0).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def paired_iar(df: pd.DataFrame, a: str, b: str, seed: int = 0, n_boot: int = 2000) -> dict | None:
    pa = df[(df["method"] == a) & (df["n_lab"].fillna(0) > 0)].set_index("q")["iar"]
    pb = df[(df["method"] == b) & (df["n_lab"].fillna(0) > 0)].set_index("q")["iar"]
    common = pa.index.intersection(pb.index)
    if len(common) < 5:
        return None
    d = (pa.loc[common] - pb.loc[common]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return {"a": a, "b": b, "n_queries": int(len(d)), "mean_diff": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


# ------------------------------------------------------------------ runs --

def _run_methods(inp: Inputs, methods: list[str], queries: np.ndarray, k: int, scfg: dict, rng) -> list[dict]:
    out = []
    extra = ["--max-exact-size", str(scfg.get("max_exact_size", 3000))]
    results: dict[tuple[str, int], dict] = {}
    for m in methods:
        if m in CPP_METHODS:
            algo, th = CPP_METHODS[m]
            if th == "rej" and inp.K != 3:
                continue
            rows = inp.cpp.run(queries, [algo], k, theta=th, time_limit=float(scfg["time_limit"]),
                               r=inp.r_global if algo.startswith("kr_") else None, extra=extra,
                               tag=f"{m}_k{k}", n_proc=int(scfg.get("n_proc", 1)))
            for row in rows:
                results[(m, row["q"])] = row
    if "cluster_q" in methods:
        from sklearn.cluster import KMeans

        active = np.flatnonzero(inp.core >= 1)
        km = KMeans(n_clusters=min(int(scfg.get("cluster_k", 20)), len(active)), n_init=4,
                    random_state=0).fit(inp.pos[active])
        lab = np.full(inp.A.shape[0], -1)
        lab[active] = km.labels_
        for q in queries:
            H = kcore_component(inp.A, np.flatnonzero(lab == lab[q]), k, int(q))
            results[("cluster_q", int(q))] = {"members": H.tolist(), "status": "ok", "time_ms": np.nan}
    if "random_bfs" in methods:
        for q in queries:
            ref = results.get(("lcc_c_exact", int(q)))
            s0 = results.get(("kcore", int(q)))
            if ref is None or s0 is None or ref["size"] == 0:
                continue
            H = random_connected(inp.A, np.array(s0["members"]), int(q), int(ref["size"]), rng)
            results[("random_bfs", int(q))] = {"members": H.tolist(), "status": "ok", "time_ms": np.nan}
    for (m, q), row in results.items():
        out.append({"method": m, "q": q, "members": row.get("members", []), "status": row.get("status"),
                    "time_ms": row.get("time_ms")})
    return out


def run(cfg: dict, paths: dict) -> dict:
    scfg, sim = cfg["search"], cfg["simulation"]
    k = int(scfg["k"])
    rng = np.random.default_rng(cfg["seed"])
    tp = test_pairs(paths)
    base_variant = scfg.get("latitude_model", "full")
    if not (paths["models"] / f"sjt_{base_variant}.npz").exists():
        LOG.warning("model sjt_%s missing, falling back to sjt_global", base_variant)
        base_variant = "global"
    by_variant: dict[str, list[str]] = {}
    for m in scfg["methods"]:
        name, _, var = m.partition("@")
        by_variant.setdefault(var or base_variant, []).append(name)
    inputs = {v: Inputs(cfg, paths, v) for v in by_variant}
    base = inputs[base_variant]
    queries = sample_queries(base, k, int(scfg["n_queries"]), scfg.get("query_pool", "test_active"), tp, rng)
    LOG.info("E2: %d queries (k=%d, pool=%s), variants %s", len(queries), k, scfg.get("query_pool"),
             list(by_variant))
    rows = []
    n_jobs = int(sim.get("n_jobs", 1))
    for v, methods in by_variant.items():
        inp = inputs[v]
        if "random_bfs" in methods and "lcc_c_exact" not in methods:
            methods = methods + ["lcc_c_exact"]
        if "random_bfs" in methods and "kcore" not in methods:
            methods = methods + ["kcore"]
        found = _run_methods(inp, methods, queries, k, scfg, rng)
        LOG.info("E2[%s]: %d communities found, computing metrics and dynamics (n_jobs=%d)", v, len(found), n_jobs)
        mets = _metrics_parallel([np.array(r["members"], dtype=int) for r in found], inp, tp, sim, k, n_jobs)
        for r, met in zip(found, mets):
            r = {kk: vv for kk, vv in r.items() if kk != "members"}
            rows.append({**r, "variant": v, **met})
    df = pd.DataFrame(rows)
    df.to_csv(paths["results"] / "e2_per_query.csv", index=False)
    summ = summarize(df, base.K)
    summ.to_csv(paths["results"] / "e2_summary.csv", index=False)
    comps = [c for c in (paired_iar(df[df["variant"] == base_variant], "lcc_c_exact", b, cfg["seed"])
                         for b in ("kcore", "kr_global", "edgeacc", "cluster_q", "random_bfs", "pairwise_c"))
             if c is not None]
    case = case_study(df[df["variant"] == base_variant])
    out = {"k": k, "variant": base_variant, "n_queries": int(len(queries)), "r_global": base.r_global,
           "paired_iar": comps, "case": case}
    dump_json(out, paths["results"] / "e2.json")
    _plot(summ, base.K, paths)
    show = [c for c in ("method", "variant", "nonempty_rate", "size_median", "iar_micro", "irr_micro",
                        "density_mean", "feasible_c_rate", "hk_consensus_rate", "sjt_rejection_free_rate")
            if c in summ]
    LOG.info("E2 summary:\n%s", summ[show].round(3).to_string(index=False))
    return out


def case_study(df: pd.DataFrame) -> dict | None:
    a = df[(df["method"] == "lcc_c_exact") & (df["n_lab"].fillna(0) >= 3)].set_index("q")
    b = df[(df["method"] == "kcore") & (df["n_lab"].fillna(0) >= 3)].set_index("q")
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return None
    gap = (a.loc[common, "iar"] - b.loc[common, "iar"]).sort_values(ascending=False)
    q = int(gap.index[0])
    keep = ["method", "size", "density", "diam", "min_theta_acc", "n_lab", "iar", "irr", "hk_consensus"]
    rows = df[df["q"] == q][[c for c in keep if c in df]].to_dict(orient="records")
    return {"query_index": q, "iar_gap_vs_kcore": float(gap.iloc[0]), "methods": rows}


def _plot(summ: pd.DataFrame, K: int, paths) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        s = summ.dropna(subset=["iar_micro"]) if "iar_micro" in summ else summ.iloc[0:0]
        if s.empty:
            return
        lab = s["method"] + np.where(s["variant"] != s["variant"].iloc[0], "@" + s["variant"], "")
        x = np.arange(len(s))
        fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(s)), 3.6))
        ax.bar(x - 0.2, s["iar_micro"], 0.4, label="internal accept rate", color="#2a9d8f")
        if K == 3 and "irr_micro" in s:
            ax.bar(x + 0.2, s["irr_micro"], 0.4, label="internal reject rate", color="#e76f51")
        ax.set_xticks(x)
        ax.set_xticklabels(lab, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("share of test judgements inside community")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(paths["figs"] / "e2_iar_irr.png", dpi=160)
        plt.close(fig)
    except Exception as e:
        LOG.warning("E2 plot failed: %s", e)


def run_g3(cfg: dict, paths: dict) -> dict:
    g = cfg["gates"]
    rng = np.random.default_rng(cfg["seed"])
    tp = test_pairs(paths)
    variant = cfg["search"].get("latitude_model", "full")
    if not (paths["models"] / f"sjt_{variant}.npz").exists():
        variant = "global"
    inp = Inputs(cfg, paths, variant)
    res = {"variant": variant, "by_k": {}}
    ok_ks = []
    for k in g["g3_k"]:
        qs = sample_queries(inp, int(k), int(g["g3_queries"]), "kcore", tp, rng)
        if len(qs) == 0:
            res["by_k"][str(k)] = {"queries": 0}
            continue
        row = {"queries": int(len(qs))}
        kc = None
        for th in (["acc", "rej"] if inp.K == 3 else ["acc"]):
            algos = ["greedy", "ballcore_f"] + (["kcore"] if kc is None else [])
            rows = inp.cpp.run(qs, algos, int(k), theta=th, members=False, time_limit=60,
                               tag=f"g3_{th}_k{k}", n_proc=int(cfg["search"].get("n_proc", 1)))
            if kc is None:
                kc = np.array([r["size"] for r in rows if r["algo"] == "kcore"], dtype=float)
                row["kcore_median"] = float(np.median(kc))
            sizes = np.array([r["size"] for r in rows if r["algo"] == "greedy"], dtype=float)
            tag = "C" if th == "acc" else "R"
            row[f"{tag}_median"] = float(np.median(sizes))
            row[f"{tag}_nonempty"] = float((sizes > 0).mean())
            ok = kc > 0
            row[f"{tag}_over_kcore_median"] = float(np.median(sizes[ok] / kc[ok])) if ok.any() else None
        # Revised after round 1 (stricter): an R-LCC that is (almost) the whole k-core component only shows that
        # the rejection latitude does not bind, so it no longer passes the gate on its own.
        c_ok = row.get("C_median", 0) >= g["g3_min_median"]
        r_ratio = row.get("R_over_kcore_median")
        r_ok = (row.get("R_median", 0) >= g["g3_min_median"] and r_ratio is not None
                and r_ratio <= float(g.get("g3_max_r_over_kcore", 0.8)))
        row["pass_C"], row["pass_R"] = bool(c_ok), bool(r_ok)
        row["pass"] = bool(c_ok or r_ok)
        if row["pass"]:
            ok_ks.append(int(k))
        res["by_k"][str(k)] = row
    res["pass"] = bool(ok_ks)
    res["recommended_k"] = max(ok_ks) if ok_ks else None
    res["criterion"] = (f"for some k in {g['g3_k']}: median greedy C-LCC size >= {g['g3_min_median']}, or median "
                        f"R-LCC size >= {g['g3_min_median']} with median |R-LCC|/|k-core component| <= "
                        f"{g.get('g3_max_r_over_kcore', 0.8)}")
    dump_json(res, paths["results"] / "g3.json")
    LOG.info("G3: %s -> %s (recommended k=%s)", res["by_k"], "PASS" if res["pass"] else "FAIL",
             res["recommended_k"])
    return res
