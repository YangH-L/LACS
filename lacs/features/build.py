"""features: anchors for every labelled pair, user feature matrix Z, and user
positions at the end of the validation period (inputs of community search)."""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from ..utils import dump_json
from . import ego as ego_mod
from . import psych as psych_mod
from .anchors import HistoryIndex, forum_priors

LOG = logging.getLogger("lacs.features")
SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


def _standardize(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df.to_numpy(dtype=float)
    miss = np.isnan(X).all(axis=1)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-9)] = 1.0
    mu[~np.isfinite(mu)] = 0.0
    Z = (X - mu) / sd
    Z = np.nan_to_num(Z, nan=0.0)
    return np.clip(Z, -5, 5), miss


def run(cfg: dict, paths: dict) -> dict:
    comments = pd.read_pickle(paths["data"] / "comments.pkl")
    pairs = pd.read_pickle(paths["data"] / "pairs.pkl")
    meta = json.loads((paths["data"] / "meta.json").read_text())
    emb = np.load(paths["emb"] / "emb_pca.npy")
    valid = np.load(paths["emb"] / "valid.npy")
    cids = (paths["emb"] / "cids.txt").read_text().split("\n")
    if cids != comments["cid"].astype(str).tolist():
        raise RuntimeError("embeddings are out of date: re-run `encode`")
    t1, t2 = meta["t_train_end"], meta["t_val_end"]
    an = cfg["anchors"]
    hist = HistoryIndex(comments, emb, valid, an["recency_lambda"], an["max_hist"], an["shrink_kappa"])
    priors = forum_priors(comments, emb, valid, t1)

    users = sorted(set(comments["author"].astype(str)) | set(pairs["src_user"]) | set(pairs["dst_user"]))
    uid = {u: i for i, u in enumerate(users)}
    row_of = hist.row_of

    N, D = len(pairs), emb.shape[1]
    X_recv = np.zeros((N, D), np.float32)
    X_src = np.zeros((N, D), np.float32)
    recv_hist = np.zeros(N)
    src_hist = np.zeros(N)
    is_cmv_op = an["receiver_repr"] == "op_post"
    if is_cmv_op and cfg["dataset"]["type"] != "cmv":
        raise ValueError("anchors.receiver_repr=op_post is only valid for CMV (the reply itself would leak)")
    for i, r in enumerate(pairs.itertuples(index=False)):
        prior = priors.get(str(r.forum), priors["__global__"])
        if is_cmv_op:
            X_recv[i] = emb[row_of[r.dst_cid]]
            recv_hist[i] = 1.0
        else:
            X_recv[i], recv_hist[i] = hist.anchor(r.dst_user, r.time, prior, exclude_row=row_of[r.dst_cid])
        src_row = row_of[r.src_cid]
        if an["source_repr"] == "message":
            X_src[i] = emb[src_row]
            src_hist[i] = 1.0
        else:
            X_src[i], src_hist[i] = hist.anchor(r.src_user, r.time, prior, extra_row=src_row, exclude_row=src_row)
    dX = X_recv - X_src
    d_raw = np.linalg.norm(dX, axis=1)

    # user features from messages up to the end of the TRAINING period only
    blocks, names, groups = [], [], {}
    if cfg["features"].get("ego", True):
        E = ego_mod.compute(comments, t1, cfg["features"].get("vader", "auto")).reindex(users)
        Zs, miss = _standardize(E)
        blocks += [Zs, miss[:, None].astype(float)]
        groups["ego"] = list(range(len(names), len(names) + Zs.shape[1]))
        names += [f"ego_{c}" for c in E.columns] + ["ego_missing"]
        groups["indicator"] = [len(names) - 1]
    P = psych_mod.load(cfg["features"].get("psych_csv"), users)
    if P is not None:
        Zp, missp = _standardize(P)
        start = len(names)
        blocks += [Zp, missp[:, None].astype(float)]
        names += [f"psych_{c}" for c in P.columns] + ["psych_missing"]
        groups["psych"] = list(range(start, start + Zp.shape[1]))
        groups.setdefault("indicator", []).append(len(names) - 1)
    else:
        LOG.warning("no psych profiles (features.psych_csv unset): latitudes use ego-involvement features only")
    Z = np.hstack(blocks) if blocks else np.zeros((len(users), 0))

    # positions at the end of the validation period (graph time)
    pos = np.zeros((len(users), D), np.float32)
    pos_hist = np.zeros(len(users))
    home = comments.groupby("author")["forum"].agg(lambda s: s.astype(str).mode().iloc[0]).to_dict()
    for u, i in uid.items():
        prior = priors.get(str(home.get(u)), priors["__global__"])
        pos[i], pos_hist[i] = hist.anchor(u, t2, prior, inclusive=True)

    n_train_recv = pairs[pairs["split"] == "train"]["dst_user"].value_counts()
    udf = pd.DataFrame({"user": users, "n_train_as_receiver": [int(n_train_recv.get(u, 0)) for u in users],
                        "pos_hist": pos_hist})
    udf.to_csv(paths["feat"] / "users.csv", index_label="idx")
    np.save(paths["feat"] / "Z.npy", Z.astype(np.float32))
    dump_json({"names": names, "groups": groups}, paths["feat"] / "Z_names.json")
    np.save(paths["feat"] / "positions_val.npy", pos)
    np.savez(paths["feat"] / "pairs.npz", dX=dX.astype(np.float32), d_raw=d_raw,
             recv=pairs["dst_user"].map(uid).to_numpy(), src=pairs["src_user"].map(uid).to_numpy(),
             y=pairs["label"].to_numpy(dtype=int), split=pairs["split"].map(SPLIT_CODE).to_numpy(),
             recv_hist=recv_hist, src_hist=src_hist, t=pairs["time"].to_numpy(),
             src_row=pairs["src_cid"].map(row_of).to_numpy(), dst_row=pairs["dst_cid"].map(row_of).to_numpy(),
             forum=np.asarray(pairs["forum"].astype(str).to_numpy(), dtype=str))
    info = {"n_pairs": N, "dim": D, "n_users": len(users), "n_features": Z.shape[1],
            "receiver_cold_share": float((recv_hist == 0).mean()), "feature_names": names}
    dump_json(info, paths["feat"] / "info.json")
    LOG.info("features: %d pairs, %d users, Z has %d columns, %.1f%% receivers without history",
             N, len(users), Z.shape[1], 100 * info["receiver_cold_share"])
    return info
