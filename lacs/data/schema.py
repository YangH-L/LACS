"""Canonical tables produced by every loader.

comments: one row per message
  cid (str)         message id
  author (str)
  text (str)        raw text
  time (float)      unix seconds; for messages whose own time is unknown this is
                    an UPPER BOUND (time_is_upper=True), which keeps all
                    "strictly before t" tests conservative (no leakage)
  time_is_upper (bool)
  thread (str)      submission / conversation id
  forum (str)       subreddit / community
  parent (str|None) id of the message this one replies to, if known

pairs: one row per labelled interaction (receiver v reads message m of source u)
  src_cid, dst_cid  message being judged / the receiver's reply (or the post)
  src_user          author of the judged message (u)
  dst_user          receiver who judges it (v)
  time (float)      time of the judgement (reply time)
  forum, thread
  label (int)       ordinal class: 0 = accept (agree / delta), 1 = noncommit
                    (neutral), 2 = reject (disagree). Binary tasks use {0, 1}
                    with 1 = not accepted.
  split (str)       train / val / test (added by prepare)
"""
from __future__ import annotations

import pandas as pd

COMMENT_COLS = ["cid", "author", "text", "time", "time_is_upper", "thread", "forum", "parent"]
PAIR_COLS = ["src_cid", "dst_cid", "src_user", "dst_user", "time", "forum", "thread", "label"]

LABEL_NAMES_3 = ["accept", "noncommit", "reject"]
LABEL_NAMES_2 = ["accept", "not_accept"]


def validate(comments: pd.DataFrame, pairs: pd.DataFrame) -> None:
    missing_c = [c for c in COMMENT_COLS if c not in comments.columns]
    missing_p = [c for c in PAIR_COLS if c not in pairs.columns]
    if missing_c or missing_p:
        raise ValueError(f"missing columns: comments {missing_c}, pairs {missing_p}")
    if comments["cid"].duplicated().any():
        raise ValueError("duplicate comment ids")
    known = set(comments["cid"])
    bad = (~pairs["src_cid"].isin(known)) | (~pairs["dst_cid"].isin(known))
    if bad.any():
        raise ValueError(f"{int(bad.sum())} pairs reference unknown messages")
    if pairs["label"].isna().any():
        raise ValueError("pairs with missing labels")
