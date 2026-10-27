"""Diagnosis of a finished run (read-only; writes only to <out_dir>/diagnosis/).

Added after round 1 on DEBAGREEMENT (G1 and G2 failed, C-LCC empty at every k).
The rules below were fixed before any of these numbers were computed. Every
decision uses the train and validation splits only; the test split is touched
only by reading back results/e1_metrics.json, which already exists.

  R0  sanity    once the reply is visible, labels are predictable
                (existing test numbers: NLL(prior) - NLL(reply_text) >= 0.05)
  R1  signal    some pre-reply model beats the per-forum class prior on val by
                >= 0.01 nats, receiver-cluster 95% CI excluding 0
  R2  density   receivers with >= 10 training judgements make up >= 25% of val
                pairs and number >= 200
  R3  personal  among those receivers, NLL(resid) <= NLL(global) - 0.01 on val,
                CI below 0
  R4  ordinal   Spearman(raw distance, label) CI above 0 among receivers whose
                anchor uses >= 10 earlier messages (train + val, as in G1)

Verdict: AUDIT if R0 fails; SUBSET if R1, R2 and (R3 or R4) pass (the data may
serve as a secondary set restricted to active receivers); otherwise CLOSE (keep
the run as a documented negative result, no further tuning on it).

G3 geometry (explanatory, no verdict): share of graph edges that satisfy the
C and R constraints, how many vertices keep >= k such edges, and |R-LCC| over
|k-core component| on the G3 queries.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..graph.io import read_graph
from ..model.metrics import cluster_bootstrap_diff, nll
from ..model.sjtlat import SJTLat
from ..utils import dump_json
from .g1 import _rho_ci

LOG = logging.getLogger("lacs.diagnose")

RULES = {
    "R0_reply_text_gain": 0.05,
    "R1_min_gain": 0.01,
    "R2_active_min_train": 10,
    "R2_min_share": 0.25,
    "R2_min_receivers": 200,
    "R3_min_gain": 0.01,
    "R4_min_hist": 10,
    "G3_C_edge_min": 0.01,
    "G3_R_ratio_max": 0.8,
}
TRAIN_STRATA = [(0, 0, "0"), (1, 2, "1-2"), (3, 9, "3-9"), (10, np.inf, ">=10")]
HIST_STRATA = [(0, 0, "0"), (1, 4, "1-4"), (5, 9, "5-9"), (10, 19, "10-19"), (20, np.inf, ">=20")]


def _stratum(values: np.ndarray, strata) -> np.ndarray:
    out = np.empty(len(values), dtype=object)
    for lo, hi, name in strata:
        out[(values >= lo) & (values <= hi)] = name
    return out


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _proba_full(clf, X, K) -> np.ndarray:
    P = np.full((len(X), K), 1e-12)
    P[:, clf.classes_.astype(int)] = clf.predict_proba(X)
    return P / P.sum(axis=1, keepdims=True)


def _diff(loss_a, loss_b, clusters, seed) -> dict:
    if len(loss_a) == 0:
        return {"mean_diff": None, "ci95": [None, None], "n_clusters": 0, "n": 0}
    out = cluster_bootstrap_diff(loss_a, loss_b, clusters, seed=seed)
    out["n"] = int(len(loss_a))
    return out


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- sections --

def existing_test_metrics(R: Path) -> tuple[pd.DataFrame, dict]:
    em = json.loads((R / "e1_metrics.json").read_text())
    rows = [{"model": k, "test_nll": v.get("nll"), "macro_f1": v.get("macro_f1"), "acc": v.get("acc"),
             "auc_reject_vs_accept": v.get("auc_reject_vs_accept")} for k, v in em.items()]
    df = pd.DataFrame(rows)
    r0 = {"evaluable": "prior" in em and "reference_reply_text" in em}
    if "prior" in em:
        df["gain_over_prior"] = em["prior"]["nll"] - df["test_nll"]
    if r0["evaluable"]:
        gain = em["prior"]["nll"] - em["reference_reply_text"]["nll"]
        r0.update({"reply_text_gain": float(gain), "pass": bool(gain >= RULES["R0_reply_text_gain"])})
    else:
        r0["pass"] = None
    return df, r0


def val_models(cfg, paths, F, Z, K, models) -> tuple[dict, pd.DataFrame]:
    """Validation losses of the class priors, every SJT-Lat variant and two flexible pre-reply probes."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    y, split, r = F["y"], F["split"], F["recv"]
    tr, va = split == 0, split == 1
    forum = F["forum"].astype(str)
    loss = {}
    freq = np.bincount(y[tr], minlength=K) / tr.sum()
    loss["prior"] = nll(np.tile(freq, (va.sum(), 1)), y[va])
    Pf = np.empty((va.sum(), K))
    for f in np.unique(forum[va]):
        m_tr, m_va = tr & (forum == f), forum[va] == f
        fr = (np.bincount(y[m_tr], minlength=K) + 1.0) / (m_tr.sum() + K) if m_tr.any() else freq
        Pf[m_va] = fr
    loss["prior_forum"] = nll(Pf, y[va])
    dX = F["dX"].astype(np.float64)
    for v, m in models.items():
        loss[f"sjt_{v}"] = nll(m.predict_proba(dX[va], Z, r[va]), y[va])

    forums = np.unique(forum)
    onehot = (forum[:, None] == forums[None, :]).astype(float)
    X = np.column_stack([dX, np.abs(dX), F["d_raw"], np.log1p(F["recv_hist"]), np.log1p(F["src_hist"]),
                         Z[r], onehot])
    sc = StandardScaler().fit(X[tr])
    lr = LogisticRegression(C=0.05, max_iter=3000).fit(sc.transform(X[tr]), y[tr])
    loss["probe_logreg"] = nll(_proba_full(lr, sc.transform(X[va]), K), y[va])
    gb = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_leaf_nodes=31, min_samples_leaf=50,
                                        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                                        random_state=int(cfg["seed"])).fit(X[tr], y[tr])
    loss["probe_gbdt"] = nll(_proba_full(gb, X[va], K), y[va])

    rows = []
    for name, l in loss.items():
        d = _diff(l, loss["prior_forum"], r[va], int(cfg["seed"]))
        rows.append({"model": name, "val_nll": float(l.mean()), "diff_vs_prior_forum": d["mean_diff"],
                     "ci_lo": d["ci95"][0], "ci_hi": d["ci95"][1]})
    return loss, pd.DataFrame(rows)


def density(F, U) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    y, split, r = F["y"], F["split"], F["recv"]
    tr, va = split == 0, split == 1
    ntr = np.bincount(r[tr], minlength=U)
    nv = ntr[r[va]]
    st = _stratum(nv, TRAIN_STRATA)
    rows = []
    for _, _, name in TRAIN_STRATA:
        m = st == name
        rows.append({"train_judgements": name, "val_pairs": int(m.sum()), "val_share": float(m.mean()),
                     "val_receivers": int(len(np.unique(r[va][m])))})
    by_train = pd.DataFrame(rows)
    hv = F["recv_hist"][va]
    sth = _stratum(hv, HIST_STRATA)
    rows = []
    for _, _, name in HIST_STRATA:
        m = sth == name
        rows.append({"earlier_messages": name, "val_pairs": int(m.sum()), "val_share": float(m.mean())})
    by_hist = pd.DataFrame(rows)
    recv_val = np.unique(r[va])
    q = np.percentile(ntr[recv_val], [50, 75, 90, 99]) if len(recv_val) else [np.nan] * 4
    summary = {
        "n_receivers_train": int((ntr > 0).sum()),
        "train_pairs_per_receiver_mean": float(tr.sum() / max((ntr > 0).sum(), 1)),
        "val_receivers": int(len(recv_val)),
        "val_receiver_train_judgements_p50_p75_p90_p99": [float(x) for x in q],
        "val_share_cold_receiver_history": float((hv == 0).mean()),
    }
    return summary, by_train, by_hist


def personalisation_by_stratum(F, loss, U, seed) -> pd.DataFrame:
    split, r = F["split"], F["recv"]
    tr, va = split == 0, split == 1
    ntr = np.bincount(r[tr], minlength=U)
    st = _stratum(ntr[r[va]], TRAIN_STRATA)
    rv = r[va]
    rows = []
    for _, _, name in TRAIN_STRATA:
        m = st == name
        n_rec = len(np.unique(rv[m]))
        for a, b in (("sjt_resid", "sjt_global"), ("sjt_full", "sjt_global"), ("sjt_feat", "sjt_global"),
                     ("sjt_global", "prior_forum"), ("probe_gbdt", "prior_forum")):
            if a not in loss or b not in loss:
                continue
            d = _diff(loss[a][m], loss[b][m], rv[m], seed) if n_rec >= 30 else \
                {"mean_diff": None, "ci95": [None, None], "n": int(m.sum())}
            rows.append({"train_judgements": name, "comparison": f"{a} - {b}", "n_pairs": int(m.sum()),
                         "n_receivers": n_rec, "mean_diff": d["mean_diff"], "ci_lo": d["ci95"][0],
                         "ci_hi": d["ci95"][1]})
    return pd.DataFrame(rows)


def g1_by_stratum(F, models, seed) -> tuple[pd.DataFrame, dict]:
    y, split, r = F["y"], F["split"], F["recv"]
    use = split <= 1
    d, h = F["d_raw"][use], F["recv_hist"][use]
    yu, ru, fu = y[use], r[use], F["forum"].astype(str)[use]
    rows = []

    def add(label, m):
        if m.sum() < 200 or len(np.unique(yu[m])) < 2:
            rows.append({"subset": label, "n_pairs": int(m.sum()), "rho": None, "ci_lo": None, "ci_hi": None})
            return None
        rho, ci = _rho_ci(d[m], yu[m], ru[m], n_boot=300, seed=seed)
        rows.append({"subset": label, "n_pairs": int(m.sum()), "rho": rho, "ci_lo": ci[0], "ci_hi": ci[1]})
        return ci

    add("all (train+val)", np.ones(len(d), bool))
    st = _stratum(h, HIST_STRATA)
    for _, _, name in HIST_STRATA:
        add(f"earlier messages {name}", st == name)
    ci4 = add(f"earlier messages >= {RULES['R4_min_hist']}", h >= RULES["R4_min_hist"])
    for f in np.unique(fu):
        add(f"forum {f}", fu == f)
    if "global" in models:  # learned metric of the global model, validation pairs only
        va = split == 1
        dl = models["global"].distances(F["dX"][va].astype(np.float64))
        hv = F["recv_hist"][va]
        for label, m in (("learned distance, val, all", np.ones(va.sum(), bool)),
                         (f"learned distance, val, earlier messages >= {RULES['R4_min_hist']}",
                          hv >= RULES["R4_min_hist"])):
            if m.sum() >= 200 and len(np.unique(y[va][m])) > 1:
                rho, ci = _rho_ci(dl[m], y[va][m], r[va][m], n_boot=300, seed=seed)
                rows.append({"subset": label, "n_pairs": int(m.sum()), "rho": rho, "ci_lo": ci[0], "ci_hi": ci[1]})
    r4 = {"evaluable": ci4 is not None, "pass": bool(ci4 is not None and ci4[0] > 0)}
    return pd.DataFrame(rows), r4


def geometry(cfg, paths, models, diag_dir: Path) -> dict:
    """Why C-LCC is empty / how much the R constraint binds, on the G3 latitude model."""
    from ..search.cpp import CppSearch

    variant = cfg["search"].get("latitude_model", "full")
    if variant not in models:
        variant = "global"
    m = models[variant]
    A = read_graph(paths["graph"] / "graph.bin")
    lat = pd.read_csv(paths["results"] / "latitudes.csv", dtype={"user": str})
    pos = m.project(np.load(paths["feat"] / "positions_val.npy"))
    th_acc = lat[f"theta_acc_{variant}"].to_numpy(dtype=float)
    has_rej = f"theta_rej_{variant}" in lat and lat[f"theta_rej_{variant}"].notna().any()
    th_rej = lat[f"theta_rej_{variant}"].to_numpy(dtype=float) if has_rej else None
    up = sp.triu(A, k=1).tocoo()
    u, v = up.row, up.col
    de = np.linalg.norm(pos[u] - pos[v], axis=1)
    deg = A.getnnz(axis=1)
    act = deg > 0
    out = {"variant": variant, "n_edges": int(len(u)), "n_graph_users": int(act.sum()),
           "edge_distance_p10_p50_p90": [float(x) for x in np.percentile(de, [10, 50, 90])] if len(de) else [],
           "theta_acc_p10_p50_p90": [float(x) for x in np.percentile(th_acc[act], [10, 50, 90])]}
    okC = de <= np.minimum(th_acc[u], th_acc[v])
    out["edge_share_C"] = float(okC.mean()) if len(de) else None
    k_star = int(cfg["search"].get("k", 3))
    g3_path = paths["results"] / "g3.json"
    g3 = json.loads(g3_path.read_text()) if g3_path.exists() else {}
    if g3.get("recommended_k"):
        k_star = int(g3["recommended_k"])
    out["k"] = k_star

    def kdeg(ok):
        c = np.bincount(np.concatenate([u[ok], v[ok]]), minlength=A.shape[0])
        return int((c >= k_star).sum())

    out["vertices_with_k_C_edges"] = kdeg(okC)
    if th_rej is not None:
        okR = de <= np.minimum(th_rej[u], th_rej[v])
        out["theta_rej_p10_p50_p90"] = [float(x) for x in np.percentile(th_rej[act], [10, 50, 90])]
        out["edge_share_R"] = float(okR.mean()) if len(de) else None
        out["vertices_with_k_R_edges"] = kdeg(okR)

    # reproduce the G3 query sample (same RNG sequence as e2.run_g3) and compare with the k-core component
    G = nx.from_scipy_sparse_array(A)
    core = np.zeros(A.shape[0], dtype=int)
    for node, c in nx.core_number(G).items():
        core[node] = c
    rng = np.random.default_rng(cfg["seed"])
    qs = np.empty(0, int)
    for k in cfg["gates"]["g3_k"]:
        cand = np.flatnonzero(core >= int(k))
        if len(cand) == 0:
            continue
        pick = np.sort(rng.choice(cand, size=min(int(cfg["gates"]["g3_queries"]), len(cand)), replace=False))
        if int(k) == k_star:
            qs = pick
    if len(qs):
        thetas = {"acc": th_acc.astype(np.float32)}
        if th_rej is not None:
            thetas["rej"] = th_rej.astype(np.float32)
        cpp = CppSearch(A, pos.astype(np.float32), thetas, diag_dir / "search")
        n_proc = int(cfg["search"].get("n_proc", 1))
        base = cpp.run(qs, ["kcore", "greedy"], k_star, theta="acc", members=False, time_limit=60,
                       tag="diag_acc", n_proc=n_proc)
        kc = np.array([x["size"] for x in base if x["algo"] == "kcore"], dtype=float)
        cs = np.array([x["size"] for x in base if x["algo"] == "greedy"], dtype=float)
        out["g3_queries"] = int(len(qs))
        out["kcore_component_median"] = float(np.median(kc))
        out["C_greedy_median"] = float(np.median(cs))
        if th_rej is not None:
            rr = cpp.run(qs, ["greedy"], k_star, theta="rej", members=False, time_limit=60, tag="diag_rej",
                         n_proc=n_proc)
            rs = np.array([x["size"] for x in rr if x["algo"] == "greedy"], dtype=float)
            ok = kc > 0
            out["R_greedy_median"] = float(np.median(rs))
            out["R_over_kcore_median"] = float(np.median(rs[ok] / kc[ok])) if ok.any() else None
    notes = []
    if out.get("edge_share_C") is not None and out["edge_share_C"] < RULES["G3_C_edge_min"]:
        notes.append("C constraint holds on almost no edge: C-LCC is structurally impossible for nearly all queries")
    ratio = out.get("R_over_kcore_median")
    if ratio is not None and ratio >= RULES["G3_R_ratio_max"]:
        notes.append("R-LCC is nearly the whole k-core component: the R constraint barely binds")
    out["notes"] = notes
    return out


# -------------------------------------------------------------------- run --

def run(cfg: dict, paths: dict) -> dict:
    diag = paths["root"] / "diagnosis"
    diag.mkdir(parents=True, exist_ok=True)
    prev = paths["root"] / "config_resolved.json"
    if prev.exists():  # reproduce the run exactly: seed, gate and search settings as the run used them
        old = json.loads(prev.read_text())
        cfg = {**cfg, **{k: old[k] for k in ("seed", "gates", "search") if k in old}}
    seed = int(cfg["seed"])
    # runs made before 2026-10-17 stored `forum` as an object array; the file is this pipeline's own output
    F = dict(np.load(paths["feat"] / "pairs.npz", allow_pickle=True))
    F["forum"] = np.asarray(F["forum"]).astype(str)
    Z = np.load(paths["feat"] / "Z.npy").astype(np.float64)
    meta = json.loads((paths["data"] / "meta.json").read_text())
    K = int(meta["n_classes"])
    U = Z.shape[0]
    models = {}
    for v in ("global", "feat", "resid", "full"):
        p = paths["models"] / f"sjt_{v}.npz"
        if p.exists():
            models[v] = SJTLat.load(p)
    out = {"rules": RULES, "dataset": meta.get("dataset"), "n_pairs": int(len(F["y"])), "n_users": U}

    # class shares and the entropy floor
    y, split = F["y"], F["split"]
    out["class_share"] = {s: (np.bincount(y[split == c], minlength=K) / max((split == c).sum(), 1)).round(4).tolist()
                          for s, c in (("train", 0), ("val", 1), ("test", 2))}
    out["entropy_nats"] = {s: _entropy(np.array(p)) for s, p in out["class_share"].items()}

    test_tab, r0 = existing_test_metrics(paths["results"])
    out["R0"] = r0
    dens, by_train, by_hist = density(F, U)
    out["density"] = dens
    loss, val_tab = val_models(cfg, paths, F, Z, K, models)
    pers = personalisation_by_stratum(F, loss, U, seed)
    g1tab, r4 = g1_by_stratum(F, models, seed)
    geo = geometry(cfg, paths, models, diag)
    out["geometry"] = geo

    # rules
    pre = val_tab[val_tab["model"].str.startswith(("sjt_", "probe_"))]
    best = pre.sort_values("diff_vs_prior_forum").iloc[0]
    out["R1"] = {"best_model": best["model"], "gain": float(-best["diff_vs_prior_forum"]),
                 "ci_hi": float(best["ci_hi"]),
                 "pass": bool(-best["diff_vs_prior_forum"] >= RULES["R1_min_gain"] and best["ci_hi"] < 0)}
    act = by_train[by_train["train_judgements"] == ">=10"].iloc[0]
    out["R2"] = {"active_val_share": float(act["val_share"]), "active_val_receivers": int(act["val_receivers"]),
                 "pass": bool(act["val_share"] >= RULES["R2_min_share"]
                              and act["val_receivers"] >= RULES["R2_min_receivers"])}
    row = pers[(pers["train_judgements"] == ">=10") & (pers["comparison"] == "sjt_resid - sjt_global")]
    if len(row) and row.iloc[0]["mean_diff"] is not None and np.isfinite(row.iloc[0]["mean_diff"]):
        rr = row.iloc[0]
        out["R3"] = {"mean_diff": float(rr["mean_diff"]), "ci_hi": float(rr["ci_hi"]),
                     "n_receivers": int(rr["n_receivers"]),
                     "pass": bool(rr["mean_diff"] <= -RULES["R3_min_gain"] and rr["ci_hi"] < 0)}
    else:
        out["R3"] = {"pass": False, "note": "fewer than 30 active receivers in val"}
    out["R4"] = r4
    if r0.get("pass") is False:
        verdict = "AUDIT"
    elif out["R1"]["pass"] and out["R2"]["pass"] and (out["R3"]["pass"] or out["R4"]["pass"]):
        verdict = "SUBSET"
    else:
        verdict = "CLOSE"
    out["verdict"] = verdict

    # outputs
    test_tab.to_csv(diag / "existing_test_metrics.csv", index=False)
    val_tab.to_csv(diag / "val_models.csv", index=False)
    by_train.to_csv(diag / "density_by_train_judgements.csv", index=False)
    by_hist.to_csv(diag / "density_by_history.csv", index=False)
    pers.to_csv(diag / "personalisation_by_stratum.csv", index=False)
    g1tab.to_csv(diag / "g1_by_stratum.csv", index=False)
    dump_json(out, diag / "diagnosis.json")
    md = [f"# Diagnosis: {out['dataset']}", "",
          f"**Verdict: {verdict}** (rules fixed in lacs/experiments/diagnose.py before these numbers existed)", "",
          "| rule | result | evidence |", "|---|---|---|",
          f"| R0 sanity (reply visible) | {r0.get('pass')} | NLL gain of reference_reply_text over prior on test: "
          f"{_fmt(r0.get('reply_text_gain'))} |",
          f"| R1 pre-reply signal | {out['R1']['pass']} | best = {out['R1']['best_model']}, val gain over forum "
          f"prior {_fmt(out['R1']['gain'])}, CI hi of diff {_fmt(out['R1']['ci_hi'])} |",
          f"| R2 density | {out['R2']['pass']} | receivers with >= 10 training judgements: "
          f"{out['R2']['active_val_receivers']} in val, {100 * out['R2']['active_val_share']:.1f}% of val pairs |",
          f"| R3 personalisation (active) | {out['R3']['pass']} | resid - global = {_fmt(out['R3'].get('mean_diff'))}, "
          f"CI hi {_fmt(out['R3'].get('ci_hi'))} |",
          f"| R4 ordinal (history-rich) | {out['R4']['pass']} | see g1 table |", "",
          "## Class shares (agree, neutral, disagree) and entropy floor", "",
          *[f"- {s}: {out['class_share'][s]}, entropy {out['entropy_nats'][s]:.4f} nats" for s in ("train", "val", "test")],
          "", "## Existing test metrics (read back, not recomputed)", "", _md_table(test_tab), "",
          "## Validation: class priors, SJT-Lat variants and flexible pre-reply probes", "",
          "`diff_vs_prior_forum` < 0 means better than the per-forum class prior.", "", _md_table(val_tab), "",
          "## Data density", "", "```", json.dumps(dens, indent=2), "```", "", _md_table(by_train), "",
          _md_table(by_hist), "", "## Personalisation by receiver activity (val)", "", _md_table(pers), "",
          "## Ordinal structure of the raw distance by subset (G1 statistic)", "", _md_table(g1tab), "",
          "## G3 geometry", "", "```", json.dumps(geo, indent=2), "```", ""]
    (diag / "DIAGNOSIS.md").write_text("\n".join(md), encoding="utf-8")
    LOG.info("diagnosis verdict: %s (R0=%s R1=%s R2=%s R3=%s R4=%s); written to %s", verdict, r0.get("pass"),
             out["R1"]["pass"], out["R2"]["pass"], out["R3"]["pass"], out["R4"]["pass"], diag)
    return out
