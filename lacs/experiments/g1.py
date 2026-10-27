"""G1: does stance distance carry ordinal acceptance structure?

Uses the training + validation pairs only (the test split is untouched).
Primary statistic: Spearman rho between the unlearned distance d_raw and the
ordinal label, with a receiver-cluster bootstrap CI. PASS if the CI is above 0
(larger distance -> more rejection). Secondary: binned label shares, the same
test on a cross-fitted learned distance, and (binary tasks) an inverted-U check.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from ..model.sjtlat import SJTLat
from ..utils import dump_json

LOG = logging.getLogger("lacs.g1")


def cluster_members(clusters):
    uniq, inv = np.unique(clusters, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    return uniq, np.split(order, np.cumsum(np.bincount(inv))[:-1])


def _rho_ci(d, y, clusters, n_boot=500, seed=0):
    rho = float(spearmanr(d, y).correlation)
    uniq, members = cluster_members(clusters)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        if len(np.unique(y[idx])) > 1:
            boots.append(spearmanr(d[idx], y[idx]).correlation)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return rho, [float(lo), float(hi)]


def binned(d, y, K, bins):
    q = np.quantile(d, np.linspace(0, 1, bins + 1))
    q[-1] += 1e-9
    b = np.clip(np.searchsorted(q, d, side="right") - 1, 0, bins - 1)
    rows = []
    for i in range(bins):
        m = b == i
        shares = np.bincount(y[m], minlength=K) / max(m.sum(), 1)
        rows.append({"bin": i, "d_mean": float(d[m].mean()) if m.any() else np.nan, "n": int(m.sum()),
                     **{f"share_{k}": float(shares[k]) for k in range(K)}})
    return pd.DataFrame(rows)


def run(cfg: dict, paths: dict) -> dict:
    F = np.load(paths["feat"] / "pairs.npz", allow_pickle=False)
    meta = json.loads((paths["data"] / "meta.json").read_text())
    K = int(meta["n_classes"])
    y, split, r = F["y"], F["split"], F["recv"]
    d = F["d_raw"]
    use = split <= 1
    alpha = cfg["gates"]["alpha"]
    out = {"n_pairs": int(use.sum()), "K": K}
    rho, ci = _rho_ci(d[use], y[use], r[use], seed=cfg["seed"])
    out["raw_distance"] = {"spearman_rho": rho, "ci95": ci}
    tab = binned(d[use], y[use], K, int(cfg["gates"]["g1_bins"]))
    tab.to_csv(paths["results"] / "g1_bins.csv", index=False)
    acc = tab["share_0"].to_numpy()
    rej = tab[f"share_{K - 1}"].to_numpy()
    out["bins"] = {"accept_share_first_last": [float(acc[0]), float(acc[-1])],
                   "reject_share_first_last": [float(rej[0]), float(rej[-1])],
                   "accept_monotone_violations": int((np.diff(acc) > 0).sum()),
                   "reject_monotone_violations": int((np.diff(rej) < 0).sum())}

    # cross-fitted learned distance: metric fitted on train, evaluated on val
    tr, va = split == 0, split == 1
    Z0 = np.zeros((int(r.max()) + 1, 0))
    m = SJTLat(n_classes=K, proj_dim=int(cfg["model"]["proj_dim"][-1]), variant="global",
               l2_W=float(cfg["model"]["l2_W"][0]), max_iter=int(cfg["model"]["max_iter"])).fit(
        F["dX"][tr], Z0, r[tr], y[tr], Z0.shape[0])
    dl = m.distances(F["dX"][va])
    rho2, ci2 = _rho_ci(dl, y[va], r[va], seed=cfg["seed"])
    out["learned_distance_val"] = {"spearman_rho": rho2, "ci95": ci2}

    if K == 2:  # inverted-U check for persuasion (accept = class 0)
        x = (d[use] - d[use].mean()) / d[use].std()
        X = np.column_stack([x, x ** 2])
        acc_y = (y[use] == 0).astype(int)
        lr = LogisticRegression(C=1e4, max_iter=2000).fit(X, acc_y)
        b1, b2 = lr.coef_[0]
        rng = np.random.default_rng(cfg["seed"])
        uniq, members = cluster_members(r[use])
        bs = []
        for _ in range(300):
            idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), size=len(uniq))])
            if acc_y[idx].min() != acc_y[idx].max():
                bs.append(LogisticRegression(C=1e4, max_iter=2000).fit(X[idx], acc_y[idx]).coef_[0][1])
        peak = -b1 / (2 * b2) if b2 != 0 else np.nan
        out["inverted_u"] = {"b_linear": float(b1), "b_quadratic": float(b2),
                             "b_quadratic_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                             "peak_in_std_units": float(peak),
                             "supported": bool(np.percentile(bs, 97.5) < 0 and x.min() < peak < x.max())}
    out["pass"] = bool(ci[0] > 0)
    out["criterion"] = "receiver-cluster bootstrap 95% CI of Spearman(d_raw, label) above 0 (train+val pairs)"
    dump_json(out, paths["results"] / "g1.json")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = ["accept", "noncommit", "reject"] if K == 3 else ["accept", "not accept"]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        bottom = np.zeros(len(tab))
        colors = ["#2a9d8f", "#bdbdbd", "#e76f51"] if K == 3 else ["#2a9d8f", "#bdbdbd"]
        for k in range(K):
            ax.bar(tab["bin"], tab[f"share_{k}"], bottom=bottom, color=colors[k], label=names[k])
            bottom += tab[f"share_{k}"].to_numpy()
        ax.set_xlabel("stance-distance decile (train + val)")
        ax.set_ylabel("label share")
        ax.set_title(f"G1: Spearman rho = {rho:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
        ax.legend(loc="upper center", ncol=K, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.18))
        fig.tight_layout()
        fig.savefig(paths["figs"] / "g1_bins.png", dpi=160)
        plt.close(fig)
    except Exception as e:  # plotting must never break the gate
        LOG.warning("G1 plot failed: %s", e)
    LOG.info("G1 rho=%.3f CI=%s learned-val rho=%.3f -> %s", rho, ci, rho2, "PASS" if out["pass"] else "FAIL")
    return out
