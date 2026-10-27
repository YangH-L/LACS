"""Adapter for psychological profiles produced by an external pipeline
(e.g. the Big Five / Schwartz / BPN inference used in the CIKM'26 paper).

1. `python -m lacs export-texts -c CONFIG` writes, for every user, the texts
   written BEFORE the training cutoff (runs/<name>/features/user_texts.jsonl).
2. Run the external pipeline on that file only (no later text => no leakage).
3. Point features.psych_csv at its output: a CSV with a `user` column and one
   numeric column per dimension.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("lacs.features")


def export_user_texts(comments: pd.DataFrame, t_cut: float, out_path: Path, max_msgs: int = 200) -> int:
    df = comments[(comments["time"] <= t_cut) & (comments["text"].str.len() > 0)].sort_values("time")
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for user, g in df.groupby("author"):
            texts = g["text"].tolist()[-max_msgs:]
            f.write(json.dumps({"user": user, "n": len(texts), "texts": texts}, ensure_ascii=False) + "\n")
            n += 1
    LOG.info("wrote training-period texts for %d users to %s", n, out_path)
    return n


def load(path: str | None, users: list[str]) -> pd.DataFrame | None:
    if not path:
        return None
    df = pd.read_csv(path, dtype={"user": str})
    if "user" not in df.columns:
        raise KeyError(f"{path} needs a 'user' column")
    df["user"] = df["user"].astype(str)
    df = df.drop_duplicates("user").set_index("user")
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        raise ValueError(f"{path} has no numeric columns")
    cover = np.mean([u in num.index for u in users])
    LOG.info("psych profiles: %d dims, coverage %.1f%% of users", num.shape[1], 100 * cover)
    return num.reindex(users)
