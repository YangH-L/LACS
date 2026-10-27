"""DEBAGREEMENT loader (Pougue-Biyong et al., NeurIPS 2021 Datasets & Benchmarks).

The released CSV schema is not documented in the paper, so columns are detected
from common names and can be overridden in the config (dataset.columns). The
first run prints the detected mapping: CHECK IT.

Numeric labels: if dataset.label_map is not given, the mapping is inferred by
matching per-subreddit label shares to Table 1 of the paper and the fit is
printed; the run stops if the fit is poor.
"""
from __future__ import annotations

import glob
import itertools
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .text import clean_text

LOG = logging.getLogger("lacs.data")

CANDIDATES = {
    "parent_id": ["msg_id_parent", "parent_id", "id_parent", "comment_id_parent", "parent_msg_id"],
    "child_id": ["msg_id_child", "child_id", "id_child", "reply_id", "comment_id_child", "child_msg_id"],
    "thread": ["submission_id", "post_id", "link_id", "thread_id"],
    "parent_text": ["body_parent", "parent_body", "text_parent", "parent_text", "comment"],
    "child_text": ["body_child", "child_body", "text_child", "child_text", "reply"],
    "parent_author": ["author_parent", "parent_author", "user_parent", "parent_user"],
    "child_author": ["author_child", "child_author", "user_child", "child_user"],
    "forum": ["subreddit", "forum", "community"],
    "time": ["exact_time", "created_utc_child", "datetime", "created_utc", "timestamp", "time", "date"],
    "time_parent": ["created_utc_parent", "time_parent", "parent_time"],
    "label": ["label", "labels", "agreement", "gold_label", "y"],
}
REQUIRED = ["parent_id", "child_id", "parent_text", "parent_author", "child_author", "time"]

# Table 1 of the paper: share of (agree, neutral, disagree) labels per subreddit.
TABLE1 = {
    "brexit": (0.29, 0.29, 0.42),
    "climate": (0.32, 0.28, 0.40),
    "blacklivesmatter": (0.45, 0.22, 0.33),
    "republican": (0.34, 0.25, 0.41),
    "democrats": (0.42, 0.22, 0.36),
}
FORUM_ALIASES = {"blm": "blacklivesmatter", "r/blm": "blacklivesmatter"}

STRING_LABELS = {
    "agree": 0, "agreement": 0, "positive": 0, "pos": 0, "support": 0, "supported": 0,
    "neutral": 1, "none": 1, "neither": 1, "clarify": 1, "clarified": 1,
    "disagree": 2, "disagreement": 2, "negative": 2, "neg": 2, "dispute": 2, "disputed": 2,
}
NAME_TO_INT = {"agree": 0, "accept": 0, "neutral": 1, "noncommit": 1, "disagree": 2, "reject": 2}


def find_files(raw: str | list[str]) -> list[str]:
    if isinstance(raw, (list, tuple)):
        out = []
        for r in raw:
            out += find_files(r)
        return out
    p = Path(raw)
    if p.is_dir():
        files = sorted(str(x) for x in p.rglob("*") if x.suffix.lower() in (".csv", ".tsv"))
    elif any(ch in raw for ch in "*?["):
        files = sorted(glob.glob(raw, recursive=True))
    else:
        files = [raw]
    if not files:
        raise FileNotFoundError(f"no CSV files found at {raw}")
    return files


def read_tables(files: list[str]) -> pd.DataFrame:
    frames = []
    for f in files:
        sep = "\t" if f.lower().endswith(".tsv") else ","
        # everything as text: ids/usernames must never be re-typed as numbers
        df = pd.read_csv(f, sep=sep, dtype=str)
        df["__file__"] = os.path.basename(f)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def detect_columns(df: pd.DataFrame, overrides: dict | None) -> dict[str, str | None]:
    overrides = overrides or {}
    lower = {c.lower(): c for c in df.columns}
    out: dict[str, str | None] = {}
    for key, cands in CANDIDATES.items():
        if key in overrides and overrides[key]:
            if overrides[key] not in df.columns:
                raise KeyError(f"configured column {overrides[key]!r} for {key} not in {list(df.columns)}")
            out[key] = overrides[key]
            continue
        out[key] = next((lower[c] for c in cands if c in lower), None)
    missing = [k for k in REQUIRED if out.get(k) is None]
    if missing:
        raise KeyError(f"could not detect columns {missing}; available: {list(df.columns)}. "
                       f"Set dataset.columns in the config.")
    return out


def parse_time(s: pd.Series) -> np.ndarray:
    if not pd.api.types.is_numeric_dtype(s):
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().mean() > 0.99:  # epoch numbers stored as text
            s = num
    if pd.api.types.is_numeric_dtype(s):
        v = s.astype("float64").to_numpy()
        med = np.nanmedian(v)
        if med > 1e14:      # microseconds
            v = v / 1e6
        elif med > 1e11:    # milliseconds
            v = v / 1e3
        return v
    kw = {"format": "mixed"} if int(pd.__version__.split(".")[0]) >= 2 else {}
    t = pd.to_datetime(s, utc=True, errors="coerce", **kw)
    bad = float(t.isna().mean())
    if bad > 0.01:
        LOG.warning("%.1f%% of time values could not be parsed (column %s)", 100 * bad, s.name)
    return (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()


def _norm_forum(x: object) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"^r/", "", s)
    return FORUM_ALIASES.get(s, s)


def infer_label_map(raw: pd.Series, forum: pd.Series) -> tuple[dict, float, float]:
    """Choose the permutation raw-value -> (agree, neutral, disagree) whose
    per-forum shares best match Table 1. Returns (map, mean abs error, gap to
    the second-best permutation)."""
    vals = sorted(pd.unique(raw.dropna()))
    if len(vals) != 3:
        raise ValueError(f"expected 3 distinct labels, found {vals}")
    f = forum.map(_norm_forum)
    known = f.isin(TABLE1.keys())
    if known.sum() == 0:
        raise ValueError("no forum matches the subreddits of Table 1; set dataset.label_map")
    scores = []
    for perm in itertools.permutations(range(3)):
        m = {vals[i]: perm[i] for i in range(3)}
        err, wsum = 0.0, 0
        for name, grp in raw[known].groupby(f[known]):
            shares = np.bincount(grp.map(m).to_numpy(dtype=int), minlength=3) / len(grp)
            err += np.abs(shares - np.array(TABLE1[name])).mean() * len(grp)
            wsum += len(grp)
        scores.append((err / wsum, m))
    scores.sort(key=lambda t: t[0])
    return scores[0][1], scores[0][0], scores[1][0] - scores[0][0]


def _label_key(x: object) -> str:
    """'2', 2, 2.0 and '2.0' all map to the key '2'; strings are lower-cased."""
    try:
        f = float(x)
        if f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(x).strip().lower()


def to_canonical_labels(raw: pd.Series, forum: pd.Series, label_map: dict | None) -> pd.Series:
    if label_map:
        # keys may be raw values (as str or number); values names or ints
        m = {}
        for k, v in label_map.items():
            vi = NAME_TO_INT[v.lower()] if isinstance(v, str) else int(v)
            m[_label_key(k)] = vi
        out = raw.map(_label_key).map(m)
        if out.isna().any():
            bad = sorted(raw[out.isna()].astype(str).unique())[:10]
            raise ValueError(f"label_map does not cover raw labels {bad}")
        return out.astype(int)
    num = pd.to_numeric(raw, errors="coerce")
    if num.notna().all():
        raw = num
    if not pd.api.types.is_numeric_dtype(raw):
        out = raw.astype(str).str.strip().str.lower().map(STRING_LABELS)
        if out.isna().any():
            bad = sorted(raw[out.isna()].astype(str).unique())[:10]
            raise ValueError(f"unknown string labels {bad}; set dataset.label_map")
        return out.astype(int)
    m, err, gap = infer_label_map(raw, forum)
    LOG.warning("inferred numeric label map %s (mean abs share error %.3f, gap to next %.3f). "
                "Confirm it and set dataset.label_map to silence this.", m, err, gap)
    if err > 0.05 or gap < 0.02:
        raise ValueError(f"label-map inference is unreliable (error {err:.3f}, gap {gap:.3f}); "
                         "set dataset.label_map explicitly")
    return raw.map(m).astype(int)


def load(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ds = cfg["dataset"]
    files = find_files(ds["raw"])
    df = read_tables(files)
    cols = detect_columns(df, ds.get("columns"))
    LOG.info("DEBAGREEMENT files: %s", files)
    LOG.info("detected columns: %s", cols)
    if cols["label"] is None:
        raise KeyError("no label column detected; set dataset.columns.label")
    if cols["forum"] is None:
        df["__forum__"] = df["__file__"].str.replace(r"\.(csv|tsv)$", "", regex=True)
        cols["forum"] = "__forum__"
    if cols["thread"] is None:
        LOG.warning("no submission/thread column: co-thread meta-path will be unavailable")
        df["__thread__"] = "t_" + df[cols["parent_id"]].astype(str)
        cols["thread"] = "__thread__"

    t_pair = parse_time(df[cols["time"]])
    keep = np.isfinite(t_pair)
    if (~keep).any():
        LOG.warning("dropping %d rows with unparseable time", int((~keep).sum()))
    df = df.loc[keep].reset_index(drop=True)
    t_pair = t_pair[keep]
    t_parent = parse_time(df[cols["time_parent"]]) if cols.get("time_parent") else None

    exclude = set(ds.get("exclude_authors", []))
    rx = re.compile(ds["exclude_author_regex"]) if ds.get("exclude_author_regex") else None

    def bad_author(a: object) -> bool:
        a = str(a)
        return (a in exclude) or (a.lower() in ("nan", "none", "")) or bool(rx and rx.search(a))

    labels = to_canonical_labels(df[cols["label"]], df[cols["forum"]], ds.get("label_map"))
    pairs = pd.DataFrame({
        "src_cid": df[cols["parent_id"]].astype(str).to_numpy(),
        "dst_cid": df[cols["child_id"]].astype(str).to_numpy(),
        "src_user": df[cols["parent_author"]].astype(str).to_numpy(),
        "dst_user": df[cols["child_author"]].astype(str).to_numpy(),
        "time": t_pair,
        "forum": df[cols["forum"]].map(_norm_forum).to_numpy(),
        "thread": df[cols["thread"]].astype(str).to_numpy(),
        "label": labels.to_numpy(),
    })
    ptext = df[cols["parent_text"]].map(clean_text)
    ctext = df[cols["child_text"]].map(clean_text) if cols["child_text"] else pd.Series([""] * len(df))

    # messages: children have exact times; parents get exact times only if
    # they also appear as a child (or a parent-time column exists), otherwise
    # the earliest time of a reply to them is used as an upper bound.
    child_rows = pd.DataFrame({"cid": pairs["dst_cid"], "author": pairs["dst_user"], "text": ctext.to_numpy(),
                               "time": pairs["time"], "time_is_upper": False, "thread": pairs["thread"],
                               "forum": pairs["forum"], "parent": pairs["src_cid"]})
    parent_rows = pd.DataFrame({"cid": pairs["src_cid"], "author": pairs["src_user"], "text": ptext.to_numpy(),
                                "time": t_parent if t_parent is not None else pairs["time"],
                                "time_is_upper": t_parent is None, "thread": pairs["thread"],
                                "forum": pairs["forum"], "parent": None})
    allm = pd.concat([child_rows, parent_rows], ignore_index=True)
    allm = allm.sort_values(["cid", "time_is_upper", "time"]).reset_index(drop=True)
    first = allm.groupby("cid", sort=False).first()
    # an exact time beats an upper bound; among upper bounds take the minimum
    tmin = allm.groupby("cid")["time"].min()
    exact_any = allm.groupby("cid")["time_is_upper"].min() == False  # noqa: E712
    comments = first.reset_index()
    comments["time"] = np.where(exact_any.loc[comments["cid"]].to_numpy(),
                                comments["time"].to_numpy(), tmin.loc[comments["cid"]].to_numpy())
    comments["time_is_upper"] = ~exact_any.loc[comments["cid"]].to_numpy()
    par = allm.dropna(subset=["parent"]).groupby("cid")["parent"].first()
    comments["parent"] = comments["cid"].map(par)
    comments["text"] = comments["text"].fillna("")

    if ds.get("extra_raw"):
        # unlabelled comment-reply pairs (e.g. the "original Reddit interactions"
        # release): used for histories and graph edges, never as labels
        ex = read_tables(find_files(ds["extra_raw"]))
        xc = detect_columns(ex, ds.get("columns"))
        if xc["forum"] is None:
            ex["__forum__"] = ex["__file__"].str.replace(r"\.(csv|tsv)$", "", regex=True)
            xc["forum"] = "__forum__"
        if xc["thread"] is None:
            ex["__thread__"] = "t_" + ex[xc["parent_id"]].astype(str)
            xc["thread"] = "__thread__"
        xt = parse_time(ex[xc["time"]])
        ex = ex.loc[np.isfinite(xt)].reset_index(drop=True)
        xt = xt[np.isfinite(xt)]
        xch = pd.DataFrame({"cid": ex[xc["child_id"]].astype(str), "author": ex[xc["child_author"]].astype(str),
                            "text": ex[xc["child_text"]].map(clean_text) if xc["child_text"] else "",
                            "time": xt, "time_is_upper": False, "thread": ex[xc["thread"]].astype(str),
                            "forum": ex[xc["forum"]].map(_norm_forum), "parent": ex[xc["parent_id"]].astype(str)})
        xpa = pd.DataFrame({"cid": ex[xc["parent_id"]].astype(str), "author": ex[xc["parent_author"]].astype(str),
                            "text": ex[xc["parent_text"]].map(clean_text), "time": xt, "time_is_upper": True,
                            "thread": ex[xc["thread"]].astype(str), "forum": ex[xc["forum"]].map(_norm_forum),
                            "parent": None})
        extra = pd.concat([xch, xpa], ignore_index=True)
        extra = extra[~extra["cid"].isin(set(comments["cid"]))]
        extra = extra.sort_values(["cid", "time_is_upper", "time"])
        xf = extra.groupby("cid", sort=False).first().reset_index()
        xf["time"] = extra.groupby("cid")["time"].min().loc[xf["cid"]].to_numpy() \
            if len(xf) else xf["time"]
        xpar = extra.dropna(subset=["parent"]).groupby("cid")["parent"].first()
        xf["parent"] = xf["cid"].map(xpar)
        comments = pd.concat([comments, xf[comments.columns]], ignore_index=True)
        LOG.info("added %d unlabelled messages from extra_raw", len(xf))

    n0 = len(pairs)
    bad = pairs["src_user"].map(bad_author) | pairs["dst_user"].map(bad_author)
    selfrep = pairs["src_user"] == pairs["dst_user"]
    emptysrc = pairs["src_cid"].map(dict(zip(comments["cid"], comments["text"].str.len() == 0)))
    drop = bad | selfrep | emptysrc.fillna(True)
    pairs = pairs.loc[~drop].drop_duplicates(["src_cid", "dst_cid"]).reset_index(drop=True)
    LOG.info("pairs: %d -> %d after dropping deleted/bot authors, self-replies, empty sources", n0, len(pairs))
    comments = comments.loc[~comments["author"].map(bad_author)].reset_index(drop=True)
    info = {"files": files, "columns": cols, "n_pairs_raw": n0}
    return comments[["cid", "author", "text", "time", "time_is_upper", "thread", "forum", "parent"]], pairs, info
