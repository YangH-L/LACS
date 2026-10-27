"""Ego-involvement proxies per user, from messages written before a cutoff.

Lexicon-free except for two short published word lists:
  absolutist words (Al-Mosaiwi & Johnstone, 2018) and first-person singular
  pronouns. VADER sentiment intensity is added when vaderSentiment is installed.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

LOG = logging.getLogger("lacs.features")

ABSOLUTIST = {"absolutely", "all", "always", "complete", "completely", "constant", "constantly", "definitely",
              "entire", "ever", "every", "everyone", "everything", "full", "must", "never", "nothing",
              "totally", "whole"}
FIRST_SG = {"i", "me", "my", "mine", "myself", "i'm", "i've", "i'd", "i'll"}
SECOND = {"you", "your", "yours", "yourself", "you're", "you've", "you'd", "you'll"}
NEGATION = {"not", "no", "never", "nobody", "nothing", "neither", "nor", "none"}
_TOK = re.compile(r"[A-Za-z']+")


def _msg_stats(text: str) -> tuple[float, ...]:
    toks = _TOK.findall(text)
    n = max(len(toks), 1)
    low = [t.lower() for t in toks]
    caps = sum(1 for t in toks if len(t) > 2 and t.isupper())
    neg = sum(1 for t in low if t in NEGATION or t.endswith("n't"))
    return (len(toks), sum(t in ABSOLUTIST for t in low) / n, sum(t in FIRST_SG for t in low) / n,
            sum(t in SECOND for t in low) / n, neg / n, caps / n, text.count("!") / n, text.count("?") / n)


def compute(comments: pd.DataFrame, t_cut: float, use_vader: str | bool = "auto") -> pd.DataFrame:
    df = comments[(comments["time"] <= t_cut) & (comments["text"].str.len() > 0)]
    stats = np.array([_msg_stats(t) for t in df["text"].tolist()], dtype=float) if len(df) else np.zeros((0, 8))
    cols = ["len", "absolutist", "first_sg", "second", "negation", "caps", "exclaim", "question"]
    S = pd.DataFrame(stats, columns=cols, index=df.index)
    vader = None
    if use_vader:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

            vader = SentimentIntensityAnalyzer()
        except ImportError:
            if use_vader is True:
                raise
            LOG.info("vaderSentiment not installed: skipping sentiment-intensity features")
    if vader is not None and len(df):
        comp = np.array([vader.polarity_scores(t)["compound"] for t in df["text"].tolist()])
        S["intensity"] = np.abs(comp)
        S["valence"] = comp
    S["author"] = df["author"].to_numpy()
    S["forum"] = df["forum"].to_numpy()
    S["thread"] = df["thread"].to_numpy()
    g = S.groupby("author")
    out = g[[c for c in S.columns if c not in ("author", "forum", "thread")]].mean()
    out["log_len"] = np.log1p(out.pop("len"))
    out["log_n_msgs"] = np.log1p(g.size())
    out["log_n_threads"] = np.log1p(g["thread"].nunique())
    out["n_forums"] = g["forum"].nunique().astype(float)
    return out
