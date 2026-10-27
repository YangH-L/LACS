"""Synthetic DEBAGREEMENT-like data with a known SJT ground truth.

Used to test the whole pipeline end-to-end without downloads: users hold
latent stances, write bag-of-words comments whose stance words reflect their
stance, and every reply is labelled agree/neutral/disagree by an ordinal SJT
model with PERSONALISED latitudes that depend on (hidden) traits:
  theta_acc = a0 * exp(0.35*O - 0.20*N + noise),  gap = g0 * exp(0.30*N + noise)
Observed traits (psych.csv) are the hidden traits plus noise, mimicking an
upstream personality-inference pipeline.

The CSV uses DEBAGREEMENT-style column names and numeric labels
(0 = disagree, 1 = neutral, 2 = agree), so the real loader is exercised.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("lacs.data")

LABEL_MAP = {0: "disagree", 1: "neutral", 2: "agree"}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(out_dir: str | Path, n_users: int = 500, n_forums: int = 2, n_threads: int = 260,
             replies: tuple[int, int] = (25, 60), temperature: float = 0.10, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    topic = [[f"topic{f}w{i}" for i in range(40)] for f in range(n_forums)]
    stance_words = [[[f"st{j}{pole}w{i}" for i in range(25)] for pole in ("neg", "pos")] for j in range(2)]
    filler = [f"fill{i}" for i in range(150)]

    users = [f"user{u:04d}" for u in range(n_users)]
    forum_of = rng.integers(0, n_forums, size=n_users)
    activity = rng.zipf(1.6, size=n_users).astype(float)
    activity = np.minimum(activity, 60.0)
    camp = rng.integers(0, 2, size=n_users)
    stance = np.stack([np.where(camp == 1, 0.8, -0.8) + 0.35 * rng.normal(size=n_users),
                       0.5 * rng.normal(size=n_users)], axis=1)
    traits = rng.normal(size=(n_users, 5))  # O C E A N
    O, N = traits[:, 0], traits[:, 4]
    theta_acc = 0.75 * np.exp(0.35 * O - 0.20 * N + 0.15 * rng.normal(size=n_users))
    gap = 0.45 * np.exp(0.30 * N + 0.10 * rng.normal(size=n_users))
    theta_rej = theta_acc + gap

    def make_text(ms: np.ndarray, f: int) -> str:
        toks = list(rng.choice(topic[f], size=6))
        for _ in range(14):
            j = int(rng.integers(0, 2))
            pole = int(rng.random() < _sigmoid(3.0 * ms[j]))
            toks.append(str(rng.choice(stance_words[j][pole])))
        toks += list(rng.choice(filler, size=8))
        rng.shuffle(toks)
        return " ".join(toks)

    by_forum = {f: np.flatnonzero(forum_of == f) for f in range(n_forums)}
    rows = []
    cid = 0
    for t in range(n_threads):
        f = int(rng.integers(0, n_forums))
        members = by_forum[f]
        w = activity[members] / activity[members].sum()
        t0 = float(rng.uniform(1.5e9, 1.5e9 + 3.0e7))
        thread_id = f"thr{t:04d}"
        msgs = []  # (cid, author_idx, msg_stance, text, time)
        a0 = int(rng.choice(members, p=w))
        ms0 = stance[a0] + 0.15 * rng.normal(size=2)
        msgs.append((f"m{cid:07d}", a0, ms0, make_text(ms0, f), t0))
        cid += 1
        n_rep = int(rng.integers(replies[0], replies[1] + 1))
        for r in range(n_rep):
            # recency-biased choice of the parent
            ages = np.arange(len(msgs))[::-1]
            pw = 0.85 ** ages
            pw /= pw.sum()
            p_idx = int(rng.choice(len(msgs), p=pw))
            parent = msgs[p_idx]
            for _ in range(10):
                a = int(rng.choice(members, p=w))
                if a != parent[1]:
                    break
            if a == parent[1]:
                continue
            ms = stance[a] + 0.15 * rng.normal(size=2)
            tm = msgs[-1][4] + float(rng.exponential(600.0)) + 1.0
            d = float(np.linalg.norm(stance[a] - parent[2]))
            s0 = (theta_acc[a] - d) / temperature
            s1 = (theta_rej[a] - d) / temperature
            p0, p01 = _sigmoid(s0), _sigmoid(s1)
            u = rng.random()
            y = 0 if u < p0 else (1 if u < p01 else 2)  # 0 accept, 1 neutral, 2 reject
            raw = {0: 2, 1: 1, 2: 0}[y]                  # DEBAGREEMENT-style numeric label
            new = (f"m{cid:07d}", a, ms, make_text(ms, f), tm)
            cid += 1
            msgs.append(new)
            rows.append({
                "msg_id_parent": parent[0], "msg_id_child": new[0], "submission_id": thread_id,
                "body_parent": parent[3], "body_child": new[3], "author_parent": users[parent[1]],
                "author_child": users[a], "subreddit": f"forum{f}", "exact_time": tm, "label": raw,
            })
    df = pd.DataFrame(rows)
    df.to_csv(out / "pairs.csv", index=False)
    obs = traits + 0.5 * rng.normal(size=traits.shape)
    pd.DataFrame(obs, columns=["O", "C", "E", "A", "N"]).assign(user=users)[
        ["user", "O", "C", "E", "A", "N"]].to_csv(out / "psych.csv", index=False)
    np.savez(out / "truth.npz", users=np.array(users), stance=stance, theta_acc=theta_acc,
             theta_rej=theta_rej, traits=traits, camp=camp)
    stats = {"pairs": len(df), "users": int(len(set(df["author_parent"]) | set(df["author_child"]))),
             "label_share": df["label"].value_counts(normalize=True).sort_index().round(3).to_dict()}
    LOG.info("synthetic data written to %s: %s", out, stats)
    return stats
