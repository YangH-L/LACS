"""E1 (acceptance prediction), G2 (personalised vs global latitudes) and H3
(latitude-trait associations). Everything is selected on the validation split
and reported on the test split."""
from __future__ import annotations

import itertools
import json
import logging

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

from ..utils import dump_json
from .baselines import BalanceFeatures, DistLogit, MFOrdinal, Prior, ReplyTextReference
from .metrics import cluster_bootstrap_diff, evaluate, nll
from .sjtlat import SJTLat

LOG = logging.getLogger("lacs.model")


def _load(paths):
    F = np.load(paths["feat"] / "pairs.npz", allow_pickle=False)
    Z = np.load(paths["feat"] / "Z.npy")
    zinfo = json.loads((paths["feat"] / "Z_names.json").read_text())
    meta = json.loads((paths["data"] / "meta.json").read_text())
    users = pd.read_csv(paths["feat"] / "users.csv", dtype={"user": str})
    return F, Z, zinfo, meta, users


def _fit_family(pd_, l2W, grid_w, grid_b, variants, data, mcfg, K, U, band, seed):
    """Fit the global model for (proj_dim, l2_W), then warm-start the others."""
    dX, Z, r, y, tr, va = data
    out = []
    g = SJTLat(n_classes=K, proj_dim=pd_, variant="global", l2_W=l2W, band=band, max_iter=mcfg["max_iter"],
               seed=seed).fit(dX[tr], Z, r[tr], y[tr], U)
    combos = {"global": [(None, None)], "feat": [(w, None) for w in grid_w], "resid": [(None, b) for b in grid_b],
              "full": [(w, b) for w in grid_w for b in grid_b]}
    for v in variants:
        for w, b in combos[v]:
            if v == "global":
                m = g
            else:
                m = SJTLat(n_classes=K, proj_dim=pd_, variant=v, l2_W=l2W, l2_w=w if w is not None else 1.0,
                           l2_b=b if b is not None else 1.0, band=band, max_iter=mcfg["max_iter"], seed=seed)
                m.fit(dX[tr], Z, r[tr], y[tr], U, warm=g)
            P = m.predict_proba(dX[va], Z, r[va])
            out.append({"variant": v, "proj_dim": pd_, "l2_W": l2W, "l2_w": w, "l2_b": b,
                        "val_nll": float(nll(P, y[va]).mean()), "model": m})
    return out


def fit_sjt_grid(dX, Z, r, y, tr, va, K, U, mcfg, variants, band=False, seed=0, n_jobs=1):
    data = (dX, Z, r, y, tr, va)
    fam = list(itertools.product(mcfg["proj_dim"], mcfg["l2_W"]))
    res = Parallel(n_jobs=n_jobs)(delayed(_fit_family)(p, l, mcfg["l2_w"], mcfg["l2_b"], variants, data, mcfg, K,
                                                       U, band, seed) for p, l in fam)
    rows = [x for fam_rows in res for x in fam_rows]
    best = {}
    for v in variants:
        cand = [x for x in rows if x["variant"] == v]
        best[v] = min(cand, key=lambda x: x["val_nll"])
    table = pd.DataFrame([{k: v for k, v in x.items() if k != "model"} for x in rows])
    return best, table


def run(cfg: dict, paths: dict) -> dict:
    F, Z, zinfo, meta, users = _load(paths)
    K = int(meta["n_classes"])
    y, split = F["y"], F["split"]
    r, s = F["recv"], F["src"]
    dX, d_raw = F["dX"].astype(np.float64), F["d_raw"]
    tr, va, te = split == 0, split == 1, split == 2
    U = Z.shape[0]
    mcfg = cfg["model"]
    band = bool(mcfg.get("band", False)) and K == 2
    n_jobs = int(mcfg.get("n_jobs", 1))
    res_dir, mdir = paths["results"], paths["models"]
    metrics, test_probs, test_loss = {}, {}, {}

    # ------------------------------------------------------------- SJT-Lat
    variants = [v for v in mcfg["variants"]]
    best, table = fit_sjt_grid(dX, Z, r, y, tr, va, K, U, mcfg, variants, band, cfg["seed"], n_jobs)
    table.to_csv(res_dir / "e1_grid.csv", index=False)
    lat_rows = {"user": users["user"]}
    for v, b in best.items():
        m = b["model"]
        P = m.predict_proba(dX[te], Z, r[te])
        name = f"sjt_{v}"
        metrics[name] = {**evaluate(P, y[te]), "val_nll": b["val_nll"], "proj_dim": b["proj_dim"],
                         "l2_W": b["l2_W"], "l2_w": b["l2_w"], "l2_b": b["l2_b"]}
        test_probs[name], test_loss[name] = P, nll(P, y[te])
        m.save(mdir / f"sjt_{v}.npz")
        acc, rej = m.latitudes(Z)
        lat_rows[f"theta_acc_{v}"] = acc
        lat_rows[f"theta_rej_{v}"] = rej
    pd.DataFrame(lat_rows).to_csv(res_dir / "latitudes.csv", index=False)

    # ablation: full model without psych features
    if "psych" in zinfo["groups"] and "full" in best:
        Zn = Z.copy()
        Zn[:, zinfo["groups"]["psych"]] = 0.0
        b = best["full"]
        mcfg_one = dict(mcfg, proj_dim=[b["proj_dim"]], l2_W=[b["l2_W"]], l2_w=[b["l2_w"]], l2_b=[b["l2_b"]])
        bn, _ = fit_sjt_grid(dX, Zn, r, y, tr, va, K, U, mcfg_one, ["full"], band, cfg["seed"], 1)
        P = bn["full"]["model"].predict_proba(dX[te], Zn, r[te])
        metrics["sjt_full_nopsych"] = {**evaluate(P, y[te]), "val_nll": bn["full"]["val_nll"]}
        test_probs["sjt_full_nopsych"], test_loss["sjt_full_nopsych"] = P, nll(P, y[te])

    # ------------------------------------------------------------ baselines
    bl = cfg.get("baselines", [])
    if "prior" in bl:
        P = Prior().fit(y[tr], K).predict_proba(te.sum())
        metrics["prior"] = evaluate(P, y[te]); test_loss["prior"] = nll(P, y[te])
    if "dist_logit" in bl:
        dl = DistLogit()
        X = dl.features(d_raw, Z[r], F["recv_hist"], F["src_hist"])
        bestC = min((1e-2, 1e-1, 1.0, 10.0), key=lambda C: nll(DistLogit().fit(X[tr], y[tr], C).predict_proba(
            X[va], K), y[va]).mean())
        P = dl.fit(X[tr], y[tr], bestC).predict_proba(X[te], K)
        metrics["dist_logit"] = {**evaluate(P, y[te]), "C": bestC}; test_loss["dist_logit"] = nll(P, y[te])
    if "mf" in bl:
        cands = []
        for rank, l2 in ((4, 1.0), (8, 1.0), (8, 10.0)):
            mf = MFOrdinal(rank=rank, l2=l2, seed=cfg["seed"]).fit(r[tr], s[tr], y[tr], U, K)
            cands.append((nll(mf.predict_proba(r[va], s[va]), y[va]).mean(), rank, l2, mf))
        _, rank, l2, mf = min(cands, key=lambda t: t[0])
        P = mf.predict_proba(r[te], s[te])
        metrics["mf"] = {**evaluate(P, y[te]), "rank": rank, "l2": l2}; test_loss["mf"] = nll(P, y[te])
    if "balance" in bl:
        bf = BalanceFeatures().fit_graph(r[tr], s[tr], y[tr], K)
        Xtr, Xva, Xte = bf.features(r[tr], s[tr]), bf.features(r[va], s[va]), bf.features(r[te], s[te])
        bestC = min((1e-2, 1e-1, 1.0), key=lambda C: nll(BalanceFeatures().fit(Xtr, y[tr], C).predict_proba(
            Xva, K), y[va]).mean())
        P = bf.fit(Xtr, y[tr], bestC).predict_proba(Xte, K)
        metrics["balance"] = {**evaluate(P, y[te]), "C": bestC}; test_loss["balance"] = nll(P, y[te])
    if "reply_text" in bl and cfg["dataset"]["type"] != "cmv":
        E = np.load(paths["emb"] / "emb_pca.npy")
        rt = ReplyTextReference()
        X = rt.features(E[F["src_row"]], E[F["dst_row"]])
        P = rt.fit(X[tr], y[tr], 1.0).predict_proba(X[te], K)
        metrics["reference_reply_text"] = evaluate(P, y[te]); test_loss["reference_reply_text"] = nll(P, y[te])

    # ------------------------------------------------- G2: personalisation
    g2 = {}
    if "full" in best and "global" in best:
        for v in ("full", "feat", "resid"):
            if f"sjt_{v}" in test_loss:
                g2[f"{v}_vs_global"] = cluster_bootstrap_diff(test_loss[f"sjt_{v}"], test_loss["sjt_global"], r[te],
                                                              seed=cfg["seed"])
        g2["pass"] = bool(g2["full_vs_global"]["ci95"][1] < 0)
        g2["criterion"] = "test NLL(full) - NLL(global) < 0 with the 95% receiver-cluster bootstrap CI below 0"
    # cold-start receivers (few training judgements)
    ntr = np.bincount(r[tr], minlength=U)
    cold = te & (ntr[r] <= int(mcfg.get("cold_start_max_train", 2)))
    cold_te = cold[te]
    if cold_te.sum() >= 30:
        g2["cold_start"] = {k: float(v[cold_te].mean()) for k, v in test_loss.items()}
        g2["cold_start"]["n"] = int(cold_te.sum())
    dump_json(g2, res_dir / "g2.json")

    # ------------------------------------------------- H3: latitude ~ traits
    h3 = []
    if "full" in best:
        m = best["full"]["model"]
        names = zinfo["names"]
        acc, rej = m.latitudes(Z)
        active = ntr > 0
        for j, nm in enumerate(names):
            row = {"feature": nm, "w_acc": float(m.params["w_a"][j])}
            if K == 3:
                row["w_noncommit_width"] = float(m.params["w_g"][j])
            rho, p = spearmanr(Z[active, j], acc[active]) if np.std(Z[active, j]) > 0 else (np.nan, np.nan)
            row.update({"spearman_theta_acc": float(rho), "p": float(p)})
            h3.append(row)
        h3 = pd.DataFrame(h3)
        ps = h3["p"].fillna(1.0).to_numpy()
        order = np.argsort(ps)
        holm = np.empty_like(ps)
        running = 0.0
        for rank_i, idx in enumerate(order):
            running = max(running, min(1.0, (len(ps) - rank_i) * ps[idx]))
            holm[idx] = running
        h3["p_holm"] = holm
        h3.to_csv(res_dir / "h3_latitude_traits.csv", index=False)

    table_out = pd.DataFrame(metrics).T
    table_out.to_csv(res_dir / "e1_metrics.csv")
    dump_json(metrics, res_dir / "e1_metrics.json")
    np.savez(res_dir / "e1_test_predictions.npz", y=y[te], recv=r[te], src=s[te],
             **{k: v for k, v in test_probs.items()})
    LOG.info("E1 test metrics:\n%s", table_out[[c for c in ("nll", "macro_f1", "acc", "auc_reject_vs_accept",
                                                            "auc_accept", "ece") if c in table_out]].round(4))
    if g2:
        LOG.info("G2 full-vs-global: %s -> %s", g2.get("full_vs_global"), "PASS" if g2.get("pass") else "FAIL")
    return {"metrics": metrics, "g2": g2}
