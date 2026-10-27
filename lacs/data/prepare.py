"""prepare: raw dataset -> canonical comments/pairs tables with a time split."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import run_paths
from ..utils import dump_json
from . import cmv, debagreement, synthetic
from .schema import validate

LOG = logging.getLogger("lacs.data")


def load_raw(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    t = cfg["dataset"]["type"]
    if t == "debagreement":
        return debagreement.load(cfg)
    if t == "cmv":
        return cmv.load(cfg)
    if t == "synthetic":
        raw = Path(cfg["dataset"]["raw"])
        if not (raw / "pairs.csv").exists():
            synthetic.generate(raw, **cfg["dataset"].get("synthetic", {}))
        c2 = dict(cfg)
        c2["dataset"] = dict(cfg["dataset"], raw=str(raw / "pairs.csv"),
                             label_map=cfg["dataset"].get("label_map") or synthetic.LABEL_MAP)
        return debagreement.load(c2)
    if t == "generic":
        raw = Path(cfg["dataset"]["raw"])
        comments = pd.read_csv(raw / "comments.csv", dtype={"cid": str, "parent": str, "thread": str})
        pairs = pd.read_csv(raw / "pairs.csv", dtype={"src_cid": str, "dst_cid": str})
        comments["time_is_upper"] = comments.get("time_is_upper", False)
        comments["parent"] = comments["parent"].where(comments["parent"].notna(), None)
        return comments, pairs, {"raw": str(raw)}
    raise ValueError(f"unknown dataset type {t}")


def assign_split(pairs: pd.DataFrame, ratios: list[float]) -> tuple[pd.DataFrame, float, float]:
    """Global time split: train < t1 <= val < t2 <= test."""
    tr, va, _ = ratios
    times = np.sort(pairs["time"].to_numpy())
    t1 = float(times[int(np.floor(tr * (len(times) - 1)))])
    t2 = float(times[int(np.floor((tr + va) * (len(times) - 1)))])
    split = np.where(pairs["time"] <= t1, "train", np.where(pairs["time"] <= t2, "val", "test"))
    pairs = pairs.assign(split=split)
    return pairs, t1, t2


def run(cfg: dict) -> dict:
    paths = run_paths(cfg)
    comments, pairs, info = load_raw(cfg)
    comments = comments.drop_duplicates("cid").reset_index(drop=True)
    pairs = pairs[pairs["src_cid"].isin(comments["cid"]) & pairs["dst_cid"].isin(comments["cid"])]
    pairs = pairs.sort_values("time", kind="mergesort").reset_index(drop=True)
    validate(comments, pairs)
    pairs, t1, t2 = assign_split(pairs, cfg["split"]["ratios"])
    n_classes = int(pairs["label"].max()) + 1
    meta = {
        "dataset": cfg["dataset"]["type"],
        "n_comments": len(comments),
        "n_pairs": len(pairs),
        "n_users": int(len(set(comments["author"]))),
        "n_classes": n_classes,
        "t_train_end": t1,
        "t_val_end": t2,
        "split_counts": pairs["split"].value_counts().to_dict(),
        "label_share": {s: pairs.loc[pairs["split"] == s, "label"].value_counts(normalize=True)
                        .sort_index().round(4).to_dict() for s in ("train", "val", "test")},
        "label_share_by_forum": pairs.groupby("forum")["label"].value_counts(normalize=True)
        .round(4).unstack().fillna(0).to_dict(orient="index"),
        "comments_time_upper_share": float(comments["time_is_upper"].mean()),
        "loader_info": info,
    }
    comments.to_pickle(paths["data"] / "comments.pkl")
    pairs.to_pickle(paths["data"] / "pairs.pkl")
    dump_json(meta, paths["data"] / "meta.json")
    LOG.info("prepared %d comments, %d pairs (%s), %d classes", len(comments), len(pairs),
             meta["split_counts"], n_classes)
    return meta
