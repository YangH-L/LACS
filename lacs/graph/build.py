"""graph: user-user graph induced by meta-paths, from messages up to the end of
the validation period (the test period is never used for structure).

meta-paths
  reply     U -writes-> C -replies_to-> C <-writes- U
  cothread  U -writes-> C -in-> Thread <-in- C <-writes- U  (threads with at most
            max_thread_users distinct authors, to avoid giant cliques)
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from ..utils import dump_json
from .io import edges_to_csr, write_graph

LOG = logging.getLogger("lacs.graph")


def build_edges(comments: pd.DataFrame, t_graph: float, metapaths: list[dict]) -> dict[str, np.ndarray]:
    C = comments[comments["time"] <= t_graph]
    author_of = dict(zip(C["cid"], C["author"]))
    out = {}
    for mp in metapaths:
        name = mp["name"]
        if name == "reply":
            ch = C.dropna(subset=["parent"])
            pa = ch["parent"].map(author_of)
            ok = pa.notna() & (pa != ch["author"])
            out[name] = np.stack([ch.loc[ok, "author"].to_numpy(), pa[ok].to_numpy()], axis=1)
        elif name == "cothread":
            cap = int(mp.get("max_thread_users", 50))
            rows = []
            skipped = 0
            for _, g in C.groupby("thread"):
                au = pd.unique(g["author"])
                if len(au) < 2:
                    continue
                if len(au) > cap:
                    skipped += 1
                    continue
                i, j = np.triu_indices(len(au), k=1)
                rows.append(np.stack([au[i], au[j]], axis=1))
            out[name] = np.concatenate(rows) if rows else np.empty((0, 2), dtype=object)
            if skipped:
                LOG.info("cothread: skipped %d threads with more than %d authors", skipped, cap)
        else:
            raise ValueError(f"unknown meta-path {name}")
    return out


def run(cfg: dict, paths: dict) -> dict:
    comments = pd.read_pickle(paths["data"] / "comments.pkl")
    meta = json.loads((paths["data"] / "meta.json").read_text())
    users = pd.read_csv(paths["feat"] / "users.csv", dtype={"user": str})
    uid = {u: i for i, u in zip(users["idx"], users["user"].astype(str))}
    has_pos = users["pos_hist"].to_numpy() > 0
    t_graph = meta["t_val_end"]
    per_mp = build_edges(comments, t_graph, cfg["graph"]["metapaths"])
    us, vs = [], []
    info = {"t_graph": t_graph, "metapaths": {}}
    for name, E in per_mp.items():
        if len(E) == 0:
            info["metapaths"][name] = 0
            continue
        a = pd.Series(E[:, 0]).map(uid)
        b = pd.Series(E[:, 1]).map(uid)
        ok = a.notna() & b.notna()
        a = a[ok].astype(int).to_numpy()
        b = b[ok].astype(int).to_numpy()
        ok2 = has_pos[a] & has_pos[b]
        us.append(a[ok2])
        vs.append(b[ok2])
        info["metapaths"][name] = int(ok2.sum())
    U = len(users)
    u = np.concatenate(us) if us else np.empty(0, int)
    v = np.concatenate(vs) if vs else np.empty(0, int)
    A = edges_to_csr(U, u, v)
    write_graph(paths["graph"] / "graph.bin", A)
    deg = np.asarray(A.sum(axis=1)).ravel()
    info.update({"n": U, "m": int(A.nnz // 2), "n_nonisolated": int((deg > 0).sum()),
                 "max_degree": int(deg.max()) if U else 0, "mean_degree_nonisolated": float(deg[deg > 0].mean())
                 if (deg > 0).any() else 0.0})
    dump_json(info, paths["graph"] / "info.json")
    LOG.info("graph: %d users with edges, %d edges (%s)", info["n_nonisolated"], info["m"], info["metapaths"])
    return info
