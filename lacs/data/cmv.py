"""ChangeMyView 'Winning Arguments' corpus (ConvoKit: winning-args-corpus).

Labelled pairs: for every top-level reply r of a conversation whose thread is
marked success in {0, 1}, the receiver is the original poster (OP) and the
judged message is r. label 0 = delta awarded (accept), 1 = not accepted.

The corpus directory is the unzipped ConvoKit download and contains
utterances.jsonl and conversations.json. If dataset.raw is "convokit" the
corpus is fetched with convokit.download (needs network + `pip install convokit`).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .text import clean_text

LOG = logging.getLogger("lacs.data")


def _corpus_dir(raw: str) -> Path:
    if raw == "convokit":
        from convokit import download  # type: ignore

        return Path(download("winning-args-corpus"))
    p = Path(raw)
    if (p / "utterances.jsonl").exists():
        return p
    hits = list(p.rglob("utterances.jsonl"))
    if not hits:
        raise FileNotFoundError(f"utterances.jsonl not found under {raw}")
    return hits[0].parent


def load(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ds = cfg["dataset"]
    root = _corpus_dir(ds["raw"])
    rows = []
    with open(root / "utterances.jsonl", encoding="utf-8") as f:
        for line in f:
            u = json.loads(line)
            spk = u.get("speaker", u.get("user"))
            if isinstance(spk, dict):
                spk = spk.get("id", spk.get("name"))
            meta = u.get("meta") or {}
            rows.append({
                "cid": str(u["id"]),
                "author": str(spk),
                "text": u.get("text") or "",
                "time": float(u.get("timestamp") or np.nan),
                "thread": str(u.get("conversation_id", u.get("root"))),
                "parent": None if u.get("reply_to") in (None, "") else str(u.get("reply_to")),
                "success": meta.get("success"),
            })
    utt = pd.DataFrame(rows)
    convs_path = root / "conversations.json"
    op_of = {}
    if convs_path.exists():
        with open(convs_path, encoding="utf-8") as f:
            convs = json.load(f)
        for cid, obj in convs.items():
            meta = obj.get("meta", obj) if isinstance(obj, dict) else {}
            if meta.get("op-userID"):
                op_of[str(cid)] = str(meta["op-userID"])
    # fall back to the author of the root utterance
    roots = utt[utt["cid"] == utt["thread"]].set_index("cid")["author"].to_dict()
    for t, a in roots.items():
        op_of.setdefault(t, a)

    exclude = set(ds.get("exclude_authors", []))
    utt["forum"] = "cmv"
    utt["time_is_upper"] = False
    utt["text"] = utt["text"].map(clean_text)
    utt = utt[np.isfinite(utt["time"])]

    top = utt[(utt["parent"] == utt["thread"]) & utt["success"].isin([0, 1, 0.0, 1.0])].copy()
    top["op"] = top["thread"].map(op_of)
    top = top[top["op"].notna() & (top["author"] != top["op"]) & ~top["author"].isin(exclude)
              & ~top["op"].isin(exclude) & (top["text"].str.len() > 0)]
    pairs = pd.DataFrame({
        "src_cid": top["cid"].to_numpy(),          # the challenger's argument (judged message)
        "dst_cid": top["thread"].to_numpy(),       # the OP post (receiver's stated view)
        "src_user": top["author"].to_numpy(),
        "dst_user": top["op"].to_numpy(),
        "time": top["time"].to_numpy(),
        "forum": "cmv",
        "thread": top["thread"].to_numpy(),
        "label": np.where(top["success"].astype(float).to_numpy() == 1.0, 0, 1),
    })
    # one judgement per (OP, challenger, thread)
    pairs = pairs.drop_duplicates(["dst_user", "src_user", "thread"]).reset_index(drop=True)
    comments = utt[~utt["author"].isin(exclude)][
        ["cid", "author", "text", "time", "time_is_upper", "thread", "forum", "parent"]].reset_index(drop=True)
    # the OP post text: title + body, if available
    if convs_path.exists():
        titles = {}
        for cid, obj in convs.items():
            meta = obj.get("meta", obj) if isinstance(obj, dict) else {}
            if meta.get("op-title"):
                titles[str(cid)] = clean_text(meta["op-title"])
        has = comments["cid"].isin(titles.keys())
        comments.loc[has, "text"] = (comments.loc[has, "cid"].map(titles) + " " +
                                      comments.loc[has, "text"]).str.strip()
    pairs = pairs[pairs["src_cid"].isin(comments["cid"]) & pairs["dst_cid"].isin(comments["cid"])]
    LOG.info("CMV: %d utterances, %d labelled OP-challenger pairs (%.1f%% delta)", len(comments), len(pairs),
             100.0 * (pairs["label"] == 0).mean() if len(pairs) else 0.0)
    return comments, pairs.reset_index(drop=True), {"root": str(root)}
