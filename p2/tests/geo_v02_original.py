"""Second application: offline meetups in location-based social networks
(SNAP Gowalla / Brightkite check-ins and friendships).

Every user gets a home location and a personal travel tolerance theta_v (a quantile
of their training-period travel distances from home). A latitude-consistent
community is a connected k-core of friends whose homes all lie within the least
tolerant member's range of one another (diameter <= min theta), i.e. everyone can
reach everyone. External ground truth comes from the held-out test period: do the
members actually check in at the same venue on the same day ("co-visit")?

Steps (python -m lacs <step> -c configs/gowalla.yaml):
  geo-prepare  load, global time split, homes, tolerances, friendship graph, co-visits
  geo-gates    G1-geo: co-visits of friends decay with home distance (validation);
               G2-geo: distance in units of the pair's tighter tolerance predicts
                       validation co-visits better than raw distance (quantile chosen
                       on the training period);
               G3-geo: community sizes over k, same rule as G3
  geo-e2       community search + test-period co-visit metrics against baselines
  geo-e2-sweep the same at every k in geo.e2_sweep_k, and at every tolerance in
               geo.e2_sweep_tolerances (parameter study)
No leakage: homes, tolerances, the graph and every choice use train/val check-ins
only; test check-ins are read only by geo-e2's metrics.

Baselines (geo.methods): kcore; kr_global ((k,r)-core with r = median tolerance);
kr_tau ((k,r)-core at the LCC's own threshold tau: same distance scale, tolerances
removed); sac (spatial-aware community search, AppInc of Fang et al. 2017) and
sac_sized (its ball grown until the k-core reaches the LCC's size); pairwise and
edgeacc (tolerances on pairs / edges only); random_bfs (size-matched random group).
Re-evaluation (v0.4): geo.results_subdir writes to results/<subdir>/ without touching
earlier results; geo.queries = all evaluates every eligible query; geo.abstention and
geo.member_analysis add the analyses described in README section 12.

Distances: homes are mapped to 3-D Earth-centred coordinates in km, so the engine's
Euclidean distance is the chord length (within 0.1% of the great-circle distance
below 300 km). Tolerances are great-circle km.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

from ..graph.io import edges_to_csr, read_graph, write_graph
from ..model.metrics import cluster_bootstrap_diff
from ..utils import dump_json

LOG = logging.getLogger("lacs.geo")
EARTH_R = 6371.0088
DAY = 86400

GEO_DEFAULTS = {
    "edges": None,
    "checkins": None,
    "split": [0.7, 0.1, 0.2],
    "min_train_checkins": 10,
    "home_cell_deg": 0.25,
    "tolerance_quantiles": [0.5, 0.7, 0.9],
    "tolerance_floor_km": 1.0,
    "max_bucket_users": 500,
    "g3_k": [2, 3, 4, 5],
    "g3_queries": 200,
    "methods": ["lcc_exact", "lcc_greedy", "lcc_ballcore_f", "lcc_ballcore_r", "pairwise", "kcore", "kr_global",
                "edgeacc", "random_bfs"],
    "e2_sweep_k": None,             # geo-e2-sweep: values of k (default: g3_k)
    "e2_sweep_tolerances": [],      # geo-e2-sweep: theta columns evaluated at k = search.k
    "sweep_tolerance_queries": 2000,  # common query sample of the tolerance sweep
    "queries": "sample",            # geo-e2: 'sample' = search.n_queries random eligible queries; 'all' = every one
    "results_subdir": None,         # write geo-e2 / geo-e2-sweep results to results/<subdir>/ (later rounds)
    "member_analysis": [],          # methods whose groups get the member-level acceptance analysis
    "abstention": [],               # methods compared on queries where the LCC is empty vs non-empty
    "synthetic": None,
    # v0.6 (E5 main / E6), all off by default so earlier configs behave exactly as before:
    "tolerance_fixed": None,        # theta column used by geo-e2 / the k sweep instead of G2-geo's choice
    "alt_outcomes": [],             # extra test outcomes: geo/test_checkins_<name>.npz -> metric columns *__<name>
    "e2_sweep_k_queries": None,     # query mode of the k sweep ('all' / 'sample'; default: geo.queries)
    "e2_sweep_k_n": None,           # sample size of the k sweep (default: search.n_queries)
    "engine_block": 0,              # > 0: engine calls of geo-e2 run in blocks of this many queries, each block
                                    # cached under work/ so an interrupted run resumes (results are identical)
}
CPP = {"lcc_exact": "exact", "lcc_greedy": "greedy", "lcc_ballcore_f": "ballcore_f", "lcc_ballcore_r": "ballcore_r",
       "pairwise": "pairwise_exact", "kcore": "kcore", "kr_global": "kr_exact", "edgeacc": "edgeacc",
       "sac": "sac_inc"}
# methods computed outside the main engine pass: kcore (connected components in Python), kr_tau (kr_exact at
# the LCC's own threshold tau, per query), sac_sized (sac_inc grown to the LCC's size, per query) and
# random_bfs (size-matched random connected group); the last three exist only where the LCC is non-empty


def gcfg(cfg: dict) -> dict:
    out = dict(GEO_DEFAULTS)
    out.update(cfg.get("geo") or {})
    return out


# -------------------------------------------------------------- geometry --

def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def to_xyz(lat, lon) -> np.ndarray:
    phi, lam = np.radians(lat), np.radians(lon)
    return np.column_stack([EARTH_R * np.cos(phi) * np.cos(lam), EARTH_R * np.cos(phi) * np.sin(lam),
                            EARTH_R * np.sin(phi)]).astype(np.float32)


# ------------------------------------------------------------------ load --

def read_edges(path) -> np.ndarray:
    df = pd.read_csv(path, sep=r"\s+", header=None, usecols=[0, 1], names=["u", "v"], comment="#",
                     dtype={"u": np.int64, "v": np.int64})
    return df.to_numpy()


def read_checkins(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["user", "time", "lat", "lon", "loc"], comment="#",
                     dtype={"user": np.int64, "time": str, "lat": np.float64, "lon": np.float64, "loc": str})
    t = pd.to_datetime(df["time"], format="%Y-%m-%dT%H:%M:%SZ", utc=True, errors="coerce")
    df["t"] = (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    ok = (df["t"].notna() & df["lat"].between(-90, 90) & df["lon"].between(-180, 180)
          & ~((df["lat"] == 0) & (df["lon"] == 0)) & df["loc"].notna())
    dropped = int((~ok).sum())
    df = df[ok].copy()
    df["t"] = df["t"].astype(np.int64)
    df["loc"] = pd.factorize(df["loc"])[0].astype(np.int64)
    LOG.info("check-ins: %d rows kept, %d dropped (bad time or coordinates)", len(df), dropped)
    return df.drop(columns=["time"]).reset_index(drop=True)


# ------------------------------------------------------------- co-visits --

def covisit_pairs(user: np.ndarray, key: np.ndarray, indptr: np.ndarray, indices: np.ndarray,
                  max_bucket: int) -> np.ndarray:
    """Friend pairs (u < v, encoded u * N + v) with a check-in at the same venue on the
    same day. `key` = venue-day id. Buckets with more than max_bucket users are skipped."""
    if len(user) == 0:
        return np.empty(0, np.int64)
    N = len(indptr) - 1
    df = pd.DataFrame({"k": key, "u": user}).drop_duplicates().sort_values(["k", "u"], kind="mergesort")
    k_arr, u_arr = df["k"].to_numpy(), df["u"].to_numpy()
    starts = np.flatnonzero(np.r_[True, k_arr[1:] != k_arr[:-1]])
    ends = np.r_[starts[1:], len(k_arr)]
    sizes = ends - starts
    out = []
    for s, e in zip(starts[sizes >= 2], ends[sizes >= 2]):
        if e - s > max_bucket:
            continue
        members = u_arr[s:e]
        for u in members:
            nb = indices[indptr[u]:indptr[u + 1]]
            common = nb[np.isin(nb, members, assume_unique=False)]
            common = common[common > u]
            if len(common):
                out.append(u * N + common)
    return np.unique(np.concatenate(out)) if out else np.empty(0, np.int64)


def venue_day_key(loc: np.ndarray, t: np.ndarray) -> np.ndarray:
    return loc.astype(np.int64) * 1_000_000 + (t // DAY).astype(np.int64)


# --------------------------------------------------------------- prepare --

def _no_subdir(g: dict, step: str) -> None:
    if g.get("results_subdir"):
        raise ValueError(f"geo.results_subdir is for re-evaluating an existing run with geo-e2 / geo-e2-sweep; "
                         f"run {step} with the original configuration")


def prepare(cfg: dict, paths: dict) -> dict:
    g = gcfg(cfg)
    _no_subdir(g, "geo-prepare")
    if g.get("synthetic"):
        synth(Path(g["synthetic"]["dir"]), **{k: v for k, v in g["synthetic"].items() if k != "dir"})
    ck = read_checkins(g["checkins"])
    E = read_edges(g["edges"])
    tr_r, va_r, _ = g["split"]
    ts = np.sort(ck["t"].to_numpy())
    t1 = int(ts[int(np.floor(tr_r * (len(ts) - 1)))])
    t2 = int(ts[int(np.floor((tr_r + va_r) * (len(ts) - 1)))])
    period = np.where(ck["t"] <= t1, 0, np.where(ck["t"] <= t2, 1, 2))
    ck["period"] = period

    tr = ck[ck["period"] == 0]
    n_tr = tr.groupby("user").size()
    active = np.sort(n_tr.index[n_tr >= int(g["min_train_checkins"])].to_numpy())
    uid = pd.Series(np.arange(len(active)), index=active)
    U = len(active)
    tr = tr[tr["user"].isin(active)].copy()

    # home: median position inside the user's most frequent grid cell
    c = float(g["home_cell_deg"])
    tr["cell"] = np.floor(tr["lat"] / c).astype(np.int64) * 100000 + np.floor(tr["lon"] / c).astype(np.int64)
    cnt = tr.groupby(["user", "cell"]).size().reset_index(name="n")
    best = cnt.sort_values(["user", "n", "cell"], ascending=[True, False, True]).drop_duplicates("user")
    inmode = tr.merge(best[["user", "cell"]], on=["user", "cell"])
    home = inmode.groupby("user")[["lat", "lon"]].median().reindex(active)

    # tolerance: quantiles of training-period distances from home (km), floored
    tr = tr.join(home, on="user", rsuffix="_home")
    tr["dist"] = haversine_km(tr["lat_home"].to_numpy(), tr["lon_home"].to_numpy(), tr["lat"].to_numpy(),
                              tr["lon"].to_numpy())
    qs = [float(x) for x in g["tolerance_quantiles"]]
    tol = tr.groupby("user")["dist"].quantile(qs).unstack().reindex(active)
    tol = np.maximum(tol.to_numpy(), float(g["tolerance_floor_km"]))

    users = pd.DataFrame({"idx": np.arange(U), "user": active, "n_train": n_tr.reindex(active).to_numpy(),
                          "home_lat": home["lat"].to_numpy(), "home_lon": home["lon"].to_numpy()})
    for j, q in enumerate(qs):
        users[f"theta_q{int(round(q * 100))}"] = tol[:, j]
    pos = to_xyz(users["home_lat"].to_numpy(), users["home_lon"].to_numpy())

    # friendship graph among active users
    keep = np.isin(E[:, 0], active) & np.isin(E[:, 1], active)
    A = edges_to_csr(U, uid.loc[E[keep, 0]].to_numpy(), uid.loc[E[keep, 1]].to_numpy())
    root = paths["root"] / "geo"
    root.mkdir(parents=True, exist_ok=True)
    write_graph(paths["graph"] / "graph.bin", A)
    np.save(root / "positions.npy", pos)
    users.to_csv(root / "users.csv", index=False)

    # co-visits of friends per period, and the test check-ins of active users (for geo-e2 metrics)
    act = ck[ck["user"].isin(active)]
    au = uid.loc[act["user"]].to_numpy()
    key = venue_day_key(act["loc"].to_numpy(), act["t"].to_numpy())
    per = act["period"].to_numpy()
    cov = {}
    for p, name in enumerate(("train", "val", "test")):
        m = per == p
        cov[name] = covisit_pairs(au[m], key[m], A.indptr, A.indices, int(g["max_bucket_users"]))
    np.savez(root / "covisits.npz", **cov)
    m = per == 2
    np.savez(root / "test_checkins.npz", user=au[m].astype(np.int32), key=key[m])

    deg = np.diff(A.indptr)
    meta = {"n_checkins": int(len(ck)), "n_users_raw": int(ck["user"].nunique()), "n_active": U,
            "t_train_end": t1, "t_val_end": t2, "n_edges": int(A.nnz // 2),
            "n_nonisolated": int((deg > 0).sum()), "tolerance_quantiles": qs,
            "tolerance_km_median": {f"q{int(round(q * 100))}": float(np.median(tol[:, j])) for j, q in enumerate(qs)},
            "covisit_friend_pairs": {k: int(len(v)) for k, v in cov.items()},
            "checkins_by_period": {n: int((per == p).sum()) for p, n in enumerate(("train", "val", "test"))}}
    dump_json(meta, root / "meta.json")
    LOG.info("geo-prepare: %d active users, %d friend edges, co-visiting friend pairs %s", U, A.nnz // 2,
             meta["covisit_friend_pairs"])
    return meta


# ----------------------------------------------------------------- gates --

def _load(paths):
    root = paths["root"] / "geo"
    users = pd.read_csv(root / "users.csv")
    pos = np.load(root / "positions.npy")
    A = read_graph(paths["graph"] / "graph.bin")
    cov = dict(np.load(root / "covisits.npz"))
    meta = json.loads((root / "meta.json").read_text())
    return root, users, pos, A, cov, meta


def friend_pairs(A: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    up = sp.triu(A, k=1).tocoo()
    return up.row.astype(np.int64), up.col.astype(np.int64)


def _logit_nll(Xtr, ytr, Xev, yev) -> np.ndarray:
    m = LogisticRegression(C=1e4, max_iter=2000).fit(Xtr, ytr)
    p = np.clip(m.predict_proba(Xev)[:, 1], 1e-9, 1 - 1e-9)
    return -(yev * np.log(p) + (1 - yev) * np.log(1 - p))


def gates(cfg: dict, paths: dict) -> dict:
    from ..search.cpp import CppSearch

    g = gcfg(cfg)
    _no_subdir(g, "geo-gates")
    root, users, pos, A, cov, meta = _load(paths)
    N = A.shape[0]
    u, v = friend_pairs(A)
    code = u * N + v
    d = np.linalg.norm(pos[u] - pos[v], axis=1).astype(np.float64)
    y = {k: np.isin(code, cov[k]).astype(int) for k in ("train", "val")}
    out = {"n_friend_pairs": int(len(u)), "positive_rate": {k: float(y[k].mean()) for k in y}}

    # G1-geo: distance decay of validation co-visits
    from ..experiments.g1 import _rho_ci
    if y["val"].sum() == 0 or y["train"].sum() == 0:
        out["error"] = "no co-visiting friend pairs in the training or validation period"
        out["pass"] = False
        dump_json(out, paths["results"] / "geo_gates.json")
        LOG.warning("geo gates: %s", out["error"])
        return out
    rho, ci = _rho_ci(d, y["val"], u, n_boot=300, seed=int(cfg["seed"]))
    q = np.quantile(d, np.linspace(0, 1, 11))
    b = np.clip(np.searchsorted(q, d, side="right") - 1, 0, 9)
    bins = pd.DataFrame({"bin": np.arange(10), "d_km_mean": [float(d[b == i].mean()) for i in range(10)],
                         "covisit_rate_val": [float(y["val"][b == i].mean()) for i in range(10)]})
    bins.to_csv(paths["results"] / "geo_g1_bins.csv", index=False)
    out["G1"] = {"spearman_d_vs_covisit_val": rho, "ci95": ci, "pass": bool(ci[1] < 0),
                 "criterion": "receiver-cluster 95% CI of Spearman(home distance, validation co-visit) below 0"}

    # G2-geo: does the pair's tighter personal tolerance add information beyond raw distance?
    # global: logit(P) = a + b log d ; personal: + c log(min(theta_u, theta_v)).
    # Tolerances come from training check-ins, so training-period co-visits are not used as labels
    # (meeting someone adds a check-in away from home and inflates theta). Labels are validation
    # co-visits, evaluated out of fold with 5-fold cross-fitting grouped by user; the tolerance
    # quantile is chosen inside each training fold.
    qs = [c for c in users.columns if c.startswith("theta_q")]
    logd = np.log(d + 0.1)

    def feats(c):
        th = np.minimum(users[c].to_numpy()[u], users[c].to_numpy()[v])
        return np.column_stack([logd, np.log(th)])

    fold = (np.random.default_rng(int(cfg["seed"])).permutation(N)[u] % 5)
    yv = y["val"]
    nll_glob = np.empty(len(u))
    nll_pers = np.empty(len(u))
    picks = []
    for f in range(5):
        te, trn = fold == f, fold != f
        if yv[trn].min() == yv[trn].max():
            nll_glob[te] = nll_pers[te] = np.nan
            continue
        c = min(qs, key=lambda c_: _logit_nll(feats(c_)[trn], yv[trn], feats(c_)[trn], yv[trn]).mean())
        picks.append(c)
        nll_glob[te] = _logit_nll(logd[trn, None], yv[trn], logd[te, None], yv[te])
        Xp = feats(c)
        nll_pers[te] = _logit_nll(Xp[trn], yv[trn], Xp[te], yv[te])
    ok = np.isfinite(nll_glob) & np.isfinite(nll_pers)
    best_q = max(set(picks), key=picks.count) if picks else qs[0]
    diff = cluster_bootstrap_diff(nll_pers[ok], nll_glob[ok], u[ok], seed=int(cfg["seed"]))
    out["G2"] = {"tolerance": best_q, "fold_picks": picks, "val_nll_global": float(nll_glob[ok].mean()),
                 "val_nll_personal": float(nll_pers[ok].mean()), "diff": diff, "pass": bool(diff["ci95"][1] < 0),
                 "criterion": "out-of-fold validation NLL(log d + log tighter tolerance) - NLL(log d) < 0, "
                              "95% user-cluster CI; 5-fold by user, tolerance quantile chosen in each training fold"}

    # G3-geo: community sizes over k with the chosen tolerance (v0.6: geo.tolerance_fixed when set, the
    # tolerance the main run uses)
    g3_tol = g.get("tolerance_fixed") or best_q
    theta = users[g3_tol].to_numpy(dtype=np.float32)
    cs = CppSearch(A, pos, {"acc": theta}, paths["work"] / "geo_g3")
    core = cs.core_numbers()
    rng = np.random.default_rng(int(cfg["seed"]))
    res, ok_ks = {}, []
    for k in g["g3_k"]:
        cand = np.flatnonzero(core >= int(k))
        if len(cand) == 0:
            res[str(k)] = {"queries": 0}
            continue
        qq = np.sort(rng.choice(cand, size=min(int(g["g3_queries"]), len(cand)), replace=False))
        rows = cs.run(qq, ["greedy", "kcore"], int(k), theta="acc", members=False, time_limit=60,
                      tag=f"g3_k{k}", n_proc=int(cfg["search"].get("n_proc", 1)))
        gs = np.array([r["size"] for r in rows if r["algo"] == "greedy"], float)
        ks = np.array([r["size"] for r in rows if r["algo"] == "kcore"], float)
        row = {"queries": int(len(qq)), "C_median": float(np.median(gs)), "C_nonempty": float((gs > 0).mean()),
               "kcore_median": float(np.median(ks))}
        row["pass"] = bool(row["C_median"] >= cfg["gates"]["g3_min_median"])
        if row["pass"]:
            ok_ks.append(int(k))
        res[str(k)] = row
    out["G3"] = {"by_k": res, "pass": bool(ok_ks), "recommended_k": max(ok_ks) if ok_ks else None,
                 "tolerance": g3_tol,
                 "criterion": f"median greedy LCC size >= {cfg['gates']['g3_min_median']} for some k"}
    out["pass"] = bool(out["G1"]["pass"] and out["G2"]["pass"] and out["G3"]["pass"])
    dump_json(out, paths["results"] / "geo_gates.json")
    LOG.info("geo gates: G1 %s (rho=%.3f), G2 %s (%s, diff=%.4f), G3 %s (k=%s)", out["G1"]["pass"], rho,
             out["G2"]["pass"], best_q, diff["mean_diff"], out["G3"]["pass"], out["G3"]["recommended_k"])
    return out


# -------------------------------------------------------------------- E2 --

def community_covisits(H: np.ndarray, test_user: np.ndarray, test_key: np.ndarray, A: sp.csr_matrix) -> dict:
    """Test-period co-visit statistics of a member set H (reference implementation; geo-e2
    uses the indexed covisit_stats, which gives identical numbers)."""
    n = len(H)
    out = {"size": int(n)}
    if n < 2:
        return out
    m = np.isin(test_user, H)
    df = pd.DataFrame({"k": test_key[m], "u": test_user[m]}).drop_duplicates()
    out["active_members"] = int(df["u"].nunique())
    N = A.shape[0]
    df = df.sort_values(["k", "u"], kind="mergesort")
    k_arr, u_arr = df["k"].to_numpy(), df["u"].to_numpy().astype(np.int64)
    starts = np.flatnonzero(np.r_[True, k_arr[1:] != k_arr[:-1]]) if len(k_arr) else np.empty(0, int)
    ends = np.r_[starts[1:], len(k_arr)] if len(k_arr) else np.empty(0, int)
    codes = []
    for s_, e_ in zip(starts, ends):
        if e_ - s_ < 2:
            continue
        mem = u_arr[s_:e_]
        i, j = np.triu_indices(len(mem), 1)
        codes.append(mem[i] * N + mem[j])
    codes = np.unique(np.concatenate(codes)) if codes else np.empty(0, np.int64)
    pairs = int(len(codes))
    a, b = codes // N, codes % N
    friend_pairs_met = int(np.count_nonzero(np.asarray(A[a, b]).ravel())) if pairs else 0
    members_met = np.unique(np.concatenate([a, b])) if pairs else np.empty(0, np.int64)
    sub = A[H][:, H]
    n_friend_pairs = sub.nnz // 2
    out.update({
        "covisit_pair_rate": pairs / (n * (n - 1) / 2),
        "member_meet_rate": len(members_met) / n,
        "friend_pair_covisit_rate": friend_pairs_met / n_friend_pairs if n_friend_pairs else np.nan,
        "n_covisit_pairs": int(pairs),
    })
    return out


def checkin_index(test_user: np.ndarray, test_key: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """CSR index user -> distinct venue-day keys of the user's test check-ins."""
    u = np.asarray(test_user, np.int64)
    kk = np.asarray(test_key, np.int64)
    o = np.lexsort((kk, u))
    u, kk = u[o], kk[o]
    keep = np.r_[True, (u[1:] != u[:-1]) | (kk[1:] != kk[:-1])] if len(u) else np.empty(0, bool)
    u, kk = u[keep], kk[keep]
    indptr = np.zeros(n + 1, np.int64)
    np.add.at(indptr, u + 1, 1)
    return np.cumsum(indptr), kk


def covisit_stats(H: np.ndarray, idx: tuple[np.ndarray, np.ndarray], A: sp.csr_matrix,
                  members: bool = False) -> tuple[dict, np.ndarray | None]:
    """Same numbers as community_covisits, from the per-user index (cost grows with the
    members' check-ins, not with the whole test period). With members=True also returns,
    per member of H, whether they co-visited with at least one other member."""
    H = np.asarray(H, np.int64)
    n = len(H)
    out = {"size": int(n)}
    if n < 2:
        return out, (np.zeros(n, bool) if members else None)
    indptr, keys = idx
    N = A.shape[0]
    st = indptr[H]
    ln = indptr[H + 1] - st
    out["active_members"] = int((ln > 0).sum())
    tot = int(ln.sum())
    codes = np.empty(0, np.int64)
    if tot:
        pos = np.repeat(st - np.cumsum(np.r_[0, ln[:-1]]), ln) + np.arange(tot)
        kk = keys[pos]
        uu = np.repeat(H, ln)
        o = np.lexsort((uu, kk))
        kk, uu = kk[o], uu[o]
        # all member pairs within each venue-day bucket, vectorised: element p of a bucket of size s
        # starting at b pairs with the s - 1 - (p - b) elements after it
        b = np.flatnonzero(np.r_[True, kk[1:] != kk[:-1]])
        size = np.diff(np.r_[b, len(kk)])
        off = np.arange(len(kk)) - np.repeat(b, size)
        cnt = np.repeat(size, size) - 1 - off
        npairs = int(cnt.sum())
        if npairs:
            first = np.repeat(np.arange(len(kk)), cnt)
            run0 = np.repeat(np.cumsum(cnt) - cnt, cnt)
            second = first + 1 + (np.arange(npairs) - run0)
            codes = np.unique(uu[first] * N + uu[second])
    pairs = int(len(codes))
    a, b2 = codes // N, codes % N
    friend_pairs_met = int(np.count_nonzero(np.asarray(A[a, b2]).ravel())) if pairs else 0
    met_ids = np.unique(np.concatenate([a, b2])) if pairs else np.empty(0, np.int64)
    n_friend_pairs = A[H][:, H].nnz // 2
    out.update({
        "covisit_pair_rate": pairs / (n * (n - 1) / 2),
        "member_meet_rate": len(met_ids) / n,
        "friend_pair_covisit_rate": friend_pairs_met / n_friend_pairs if n_friend_pairs else np.nan,
        "n_covisit_pairs": int(pairs),
    })
    return out, (np.isin(H, met_ids) if members else None)


def _results_dir(cfg: dict, paths: dict) -> Path:
    sub = gcfg(cfg).get("results_subdir")
    d = paths["results"] / sub if sub else paths["results"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def _e2_context(cfg: dict, paths: dict, tag: str, tolerance: str | None = None) -> dict:
    """Everything geo-e2 needs, loaded once (shared by the sweeps). tolerance: a theta column
    of users.csv; default = the one G2-geo chose."""
    from ..search.cpp import CppSearch

    g = gcfg(cfg)
    root, users, pos, A, cov, meta = _load(paths)
    gates_path = paths["results"] / "geo_gates.json"
    gt = json.loads(gates_path.read_text()) if gates_path.exists() else {}
    tol_main = (g.get("tolerance_fixed") or gt.get("G2", {}).get("tolerance")
                or [c for c in users.columns if c.startswith("theta_q")][0])
    if tol_main not in users.columns:
        raise ValueError(f"geo.tolerance_fixed: no column {tol_main} in users.csv")
    tol = tolerance or tol_main
    theta = users[tol].to_numpy(dtype=np.float32)
    tz = np.load(root / "test_checkins.npz")
    cs = CppSearch(A, pos, {"acc": theta}, paths["work"] / tag, graph_bin=paths["graph"] / "graph.bin")
    deg = np.diff(A.indptr)
    test_active = np.zeros(A.shape[0], bool)
    test_active[np.unique(tz["user"])] = True  # eligibility uses the main outcome only
    idx_alt = {}
    for name in g.get("alt_outcomes") or []:
        za = np.load(root / f"test_checkins_{name}.npz")
        idx_alt[str(name)] = checkin_index(za["user"], za["key"], A.shape[0])
    return {"g": g, "A": A, "pos": pos, "theta": theta, "tol": tol, "tol_main": tol_main,
            "n_train": users["n_train"].to_numpy(), "idx": checkin_index(tz["user"], tz["key"], A.shape[0]),
            "idx_alt": idx_alt, "cs": cs, "core": cs.core_numbers(), "r_global": float(np.median(theta[deg > 0])),
            "test_active": test_active, "cache": {}, "kcomp": {}, "work": paths["work"] / tag}


def _diameter(P: np.ndarray) -> float:
    dmax = 0.0
    for i in range(0, len(P), 512):
        dd = np.sqrt(((P[i:i + 512, None, :] - P[None, :, :]) ** 2).sum(-1)).max()
        dmax = max(dmax, float(dd))
    return dmax


def _member_metrics(ctx: dict, H: np.ndarray, memo: bool = True) -> dict:
    """Test-period co-visit metrics, diameter and tightest tolerance of a member set. Memoised:
    baselines such as the k-core component return the same large set for many queries."""
    digest = hashlib.blake2b(np.sort(np.asarray(H, np.int64)).tobytes(), digest_size=16).digest()
    key = digest if memo else None
    if key is not None and key in ctx["cache"]:
        return ctx["cache"][key]
    met, _ = covisit_stats(H, ctx["idx"], ctx["A"])
    diam = _diameter(ctx["pos"][H].astype(np.float64)) if 2 <= len(H) <= 3000 else np.nan
    out = {"diam_km": diam, "min_theta_km": float(ctx["theta"][H].min()) if len(H) else np.nan, **met}
    for name, idx in (ctx.get("idx_alt") or {}).items():  # v0.6: alternative test outcomes, same groups
        alt, _ = covisit_stats(H, idx, ctx["A"])
        for c in ("covisit_pair_rate", "member_meet_rate", "n_covisit_pairs", "active_members"):
            out[f"{c}__{name}"] = alt.get(c, np.nan)
    out["members_hash"] = digest.hex()[:16]  # identity of the member set (internal indices only)
    if key is not None:
        ctx["cache"][key] = out
    return out


def _run_blocks(ctx: dict, queries, algos, k: int, tag: str, **kw) -> list[dict]:
    """ctx['cs'].run in blocks of geo.engine_block queries; each finished block is cached under the
    context's work directory together with a digest of everything that determines it, so a rerun after
    an interruption reuses finished blocks and recomputes the rest (block size 0: one plain call)."""
    cs, B = ctx["cs"], int(ctx["g"].get("engine_block") or 0)
    queries = [int(q) for q in queries]
    if B <= 0 or not queries:
        return cs.run(queries, algos, k, tag=tag, **kw)
    per_q = {n: kw.pop(n) for n in ("r_per_query", "size_per_query") if kw.get(n) is not None}
    th_path = cs.theta_paths[kw.get("theta", "acc")]
    base = {"algos": list(algos), "k": int(k), "tag": tag, "kw": {a: kw[a] for a in sorted(kw) if a != "n_proc"},
            "theta_sha": hashlib.sha256(Path(th_path).read_bytes()).hexdigest(),
            "inputs_sha": ctx.setdefault("_inputs_sha", hashlib.sha256(Path(cs.graph_path).read_bytes()
                                                                      + Path(cs.pos_path).read_bytes()
                                                                      + Path(cs.exe).read_bytes()).hexdigest())}
    cache_dir = Path(ctx["work"]) / "blocks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for b0 in range(0, len(queries), B):
        qb = queries[b0:b0 + B]
        pb = {n: [v[i] for i in range(b0, b0 + len(qb))] for n, v in per_q.items()}
        spec = {**base, "queries": qb, "per_query": pb}
        dig = hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:24]
        f = cache_dir / f"{tag}_{b0 // B:05d}_{dig}.jsonl"
        if f.exists():
            with open(f) as fh:
                got = [json.loads(line) for line in fh if line.strip()]
            if len(got) == len(qb) * len(algos):
                rows += got
                continue
        got = cs.run(qb, algos, k, tag=f"{tag}_b", **kw, **pb)
        tmp = f.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            for r in got:
                fh.write(json.dumps(r) + "\n")
        tmp.replace(f)
        rows += got
        LOG.info("%s: engine block %d/%d done", tag, b0 // B + 1, (len(queries) + B - 1) // B)
    return rows


def _kcore_members(ctx: dict, k: int):
    """Connected k-core component of every vertex (computed once per k): q -> members."""
    if k not in ctx["kcomp"]:
        from scipy.sparse.csgraph import connected_components

        n = ctx["A"].shape[0]
        lab = np.full(n, -1, np.int64)
        keep = np.flatnonzero(ctx["core"] >= k)
        groups = {}
        if len(keep):
            _, lk = connected_components(ctx["A"][keep][:, keep], directed=False)
            lab[keep] = lk
            o = np.argsort(lk, kind="stable")
            b = np.flatnonzero(np.r_[True, lk[o][1:] != lk[o][:-1]])
            for s_, e_ in zip(b, np.r_[b[1:], len(o)]):
                groups[int(lk[o[s_]])] = keep[o[s_:e_]]
        ctx["kcomp"][k] = (lab, groups)
    lab, groups = ctx["kcomp"][k]
    return lambda q: groups[int(lab[q])] if lab[q] >= 0 else np.empty(0, np.int64)


def _random_connected(A: sp.csr_matrix, ok: np.ndarray, q: int, size: int, rng) -> np.ndarray:
    """Random connected group of `size` vertices grown from q (same process as
    experiments.e2.random_connected, with a vertex mask instead of a pool set)."""
    chosen = [q]
    in_set = {q}
    frontier = [int(u) for u in A.indices[A.indptr[q]:A.indptr[q + 1]] if ok[u]]
    while len(chosen) < size and frontier:
        i = int(rng.integers(len(frontier)))
        u = frontier.pop(i)
        if u in in_set:
            continue
        in_set.add(u)
        chosen.append(u)
        frontier += [int(w) for w in A.indices[A.indptr[u]:A.indptr[u + 1]] if ok[w] and w not in in_set]
    return np.array(sorted(chosen))


def _select_queries(cfg: dict, ctx: dict, k: int, mode: str, n: int):
    """Eligible queries: test-active users in the k-core. mode 'all' takes every one,
    otherwise n sampled with the run's seed (the same draw as round 3's geo-e2)."""
    rng = np.random.default_rng(int(cfg["seed"]))
    cand = np.flatnonzero((ctx["core"] >= k) & ctx["test_active"])
    if mode == "all" or len(cand) == 0:
        return cand, rng, int(len(cand))
    return np.sort(rng.choice(cand, size=min(int(n), len(cand)), replace=False)), rng, int(len(cand))


def _boot_mean_diff(a: np.ndarray, b: np.ndarray, seed: int, n_boot: int = 2000) -> dict:
    """mean(a) - mean(b) for two independent samples, percentile bootstrap CI."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 5 or len(b) < 5:
        return {"n_a": int(len(a)), "n_b": int(len(b))}
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n_a": int(len(a)), "n_b": int(len(b)), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "diff": float(a.mean() - b.mean()), "ci95": [float(lo), float(hi)]}


def _abstention(df: pd.DataFrame, methods: list, seed: int) -> dict:
    """Secondary analysis: do a baseline's groups co-visit less on queries where the LCC is empty
    (LCC abstains) than on queries where it exists?"""
    if df.empty or "lcc_exact" not in set(df["method"]):
        return {}
    has = (df[df["method"] == "lcc_exact"].set_index("q")["size"] >= 2)
    out = {}
    for m in methods:
        d = df[(df["method"] == m) & (df["size"] >= 2)].set_index("q")
        if d.empty:
            continue
        h = has.reindex(d.index).fillna(False).to_numpy(bool)
        out[m] = {met: _boot_mean_diff(d.loc[h, met].to_numpy(), d.loc[~h, met].to_numpy(), seed)
                  for met in ("covisit_pair_rate", "member_meet_rate")}
        out[m]["note"] = "a = queries with a non-empty LCC, b = queries where the LCC is empty"
    return out


def _logit_first_coef(X: np.ndarray, y: np.ndarray, w: np.ndarray, iters: int = 30) -> float:
    """Weighted logistic regression with intercept by Newton's method; returns the coefficient
    of the first column of X (nan if the fit does not converge)."""
    X1 = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(X1.shape[1])
    ridge = 1e-6 * np.eye(X1.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X1 @ beta, -30, 30)))
        grad = X1.T @ (w * (y - p))
        hess = (X1 * (w * p * (1 - p))[:, None]).T @ X1 + ridge
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            return float(beta[1])
    return float(beta[1]) if np.all(np.isfinite(beta)) else float("nan")


def _member_level(ctx: dict, groups: dict, seed: int, n_boot: int = 200) -> dict:
    """Member-level test of the tolerance premise on baseline groups (which ignore tolerances):
    member u 'accepts' its group G if every other member lives within theta_u of u
    (eccentricity <= theta_u). Compare the share of members who co-visit with another member
    in the test period between accepting and non-accepting members: raw rates with a
    group-cluster bootstrap, and logistic models with controls (eccentricity, training activity,
    and in model b also log theta). Only aggregates are returned."""
    theta, pos, ntr = ctx["theta"], ctx["pos"], ctx["n_train"]
    out = {}
    for m, glist in groups.items():
        rows = []
        for gi, (q, H) in enumerate(glist):
            if len(H) < 3 or len(H) > 3000:
                continue
            P = pos[H].astype(np.float64)
            D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
            ecc = D.max(1)
            _, met = covisit_stats(H, ctx["idx"], ctx["A"], members=True)
            acc = ecc <= theta[H].astype(np.float64) + 1e-6
            rows.append(np.column_stack([np.full(len(H), gi), acc, met, np.log(ecc + 0.1),
                                         np.log(theta[H].astype(np.float64)), np.log(np.maximum(ntr[H], 1))]))
        if not rows:
            continue
        R = np.vstack(rows)
        g, acc, met = R[:, 0].astype(np.int64), R[:, 1] > 0, R[:, 2] > 0
        res = {"groups": int(len(np.unique(g))), "members": int(len(R)), "accept_share": float(acc.mean())}
        if acc.all() or (~acc).all():
            out[m] = res
            continue
        # raw rates, group-cluster bootstrap of the difference of ratios
        uniq, inv = np.unique(g, return_inverse=True)
        sa = np.bincount(inv, weights=(met & acc), minlength=len(uniq))
        ca = np.bincount(inv, weights=acc, minlength=len(uniq))
        sr = np.bincount(inv, weights=(met & ~acc), minlength=len(uniq))
        cr = np.bincount(inv, weights=~acc, minlength=len(uniq))
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(2000):
            i = rng.integers(0, len(uniq), len(uniq))
            if ca[i].sum() > 0 and cr[i].sum() > 0:
                boots.append(sa[i].sum() / ca[i].sum() - sr[i].sum() / cr[i].sum())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        res.update({"meet_rate_accept": float(met[acc].mean()), "meet_rate_reject": float(met[~acc].mean()),
                    "diff": float(met[acc].mean() - met[~acc].mean()), "ci95": [float(lo), float(hi)]})

        cells = [int((acc & met).sum()), int((acc & ~met).sum()), int((~acc & met).sum()), int((~acc & ~met).sum())]
        res["cells_accept_meet_nomeet_reject_meet_nomeet"] = cells
        if min(cells) < 20:  # too few events in some cell for a stable logistic fit
            out[m] = res
            continue
        # logistic models, coefficient of 'accept' with a group-cluster bootstrap (resampled groups
        # enter as integer weights, so every replicate is one weighted Newton fit)
        for name, cols in (("model_a", [1, 3, 5]), ("model_b", [1, 3, 4, 5])):
            X = R[:, cols]
            est = _logit_first_coef(X, met.astype(float), np.ones(len(R)))
            bs = []
            for _ in range(n_boot):
                cnt = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)).astype(float)
                bs.append(_logit_first_coef(X, met.astype(float), cnt[inv]))
            bs = np.array([b for b in bs if np.isfinite(b)])
            res[name] = {"coef_accept": est,
                         "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(bs) else None,
                         "controls": "log eccentricity, log training check-ins" + (", log theta" if 4 in cols else "")}
        out[m] = res
    return out


def _e2_eval(cfg: dict, ctx: dict, k: int, mode: str, n: int, tag: str, member_analysis: bool = True):
    """Community search for the selected queries and co-visit metrics per method.
    Returns (per-query rows, summary, paired differences vs lcc_exact, info, extras)."""
    g = ctx["g"]
    A, cs = ctx["A"], ctx["cs"]
    qs, rng, n_eligible = _select_queries(cfg, ctx, k, mode, n)
    info = {"k": int(k), "tolerance": ctx["tol"], "queries": int(len(qs)), "eligible": n_eligible,
            "query_mode": mode, "r_global_km": ctx["r_global"]}
    if len(qs) == 0:
        return pd.DataFrame(), pd.DataFrame(), {}, info, {}
    methods = list(g["methods"])
    tl, npr = float(cfg["search"]["time_limit"]), int(cfg["search"].get("n_proc", 1))
    needs_lcc = {"kr_tau", "sac_sized", "random_bfs"} & set(methods)
    algos = sorted({CPP[m] for m in methods if m in CPP and m != "kcore"} | ({"exact"} if needs_lcc else set()))
    rows = _run_blocks(ctx, qs, algos, k, f"{tag}_k{k}", theta="acc", time_limit=tl, r=ctx["r_global"], n_proc=npr)
    found = {(r["q"], r["algo"]): r for r in rows}
    lcc = [(int(q), found[(q, "exact")]) for q in qs if found[(q, "exact")]["size"] >= 2] if needs_lcc else []
    if "kr_tau" in methods and lcc:  # (k,r)-core at the LCC's own distance threshold: personal tolerances removed
        tq = [(q, float(r["tau"])) for q, r in lcc if r["tau"] is not None]
        for r in _run_blocks(ctx, [q for q, _ in tq], ["kr_exact"], k, f"{tag}_k{k}_krtau", theta="acc",
                             time_limit=tl, r_per_query=[t for _, t in tq], n_proc=npr):
            found[(r["q"], "kr_tau")] = r
    if "sac_sized" in methods and lcc:  # spatial community search grown to the LCC's size: tolerances removed
        for r in _run_blocks(ctx, [q for q, _ in lcc], ["sac_inc"], k, f"{tag}_k{k}_sacsized", theta="acc",
                             time_limit=tl, size_per_query=[int(r["size"]) for _, r in lcc], n_proc=npr):
            found[(r["q"], "sac_sized")] = r
    kcore_of = _kcore_members(ctx, k) if "kcore" in methods else None
    pool_ok = ctx["core"] >= 1
    wanted = set(g.get("member_analysis") or []) if member_analysis else set()
    groups = {m: [] for m in wanted}
    recs = []
    for q in qs:
        q = int(q)
        for m in methods:
            r = None
            if m == "random_bfs":
                ref = found.get((q, "exact"))
                size = int(ref["size"]) if ref else 0
                H = _random_connected(A, pool_ok, q, size, rng) if size >= 2 else np.empty(0, np.int64)
                status, memo = "ok", False
            elif m == "kcore":
                H, status, memo = kcore_of(q), "ok", True
            elif m in ("kr_tau", "sac_sized"):
                r = found.get((q, m))
                H = np.asarray(r["members"], dtype=np.int64) if r else np.empty(0, np.int64)
                status, memo = (r["status"] if r else "no_lcc"), True
            else:
                r = found[(q, CPP[m])]
                H = np.asarray(r["members"], dtype=np.int64)
                status, memo = r["status"], True
            rec = {"q": q, "method": m, "status": status, **_member_metrics(ctx, H, memo)}
            rec["radius_km"] = float(r["tau"]) if (r is not None and r.get("tau") is not None) else np.nan
            recs.append(rec)
            if m in groups and len(H) >= 3:
                groups[m].append((q, H))
    df = pd.DataFrame(recs)
    nz = df[df["size"] >= 2]
    summ = (nz.groupby("method").agg(nonempty=("q", "size"), size_median=("size", "median"),
                                     covisit_pair_rate=("covisit_pair_rate", "mean"),
                                     member_meet_rate=("member_meet_rate", "mean"),
                                     friend_pair_covisit_rate=("friend_pair_covisit_rate", "mean"),
                                     diam_km_median=("diam_km", "median"))
            .reindex(methods).reset_index())
    summ["coverage"] = summ["nonempty"].fillna(0) / len(qs)
    alt_names = list(ctx.get("idx_alt") or {})
    for name in alt_names:  # v0.6: means of the alternative outcomes (robustness metrics, reported only)
        for c in ("covisit_pair_rate", "member_meet_rate"):
            col = f"{c}__{name}"
            summ[col] = nz.groupby("method")[col].mean().reindex(methods).to_numpy()
    # paired differences lcc_exact minus each baseline, on queries where both are non-empty
    paired = {}
    for metric in ["covisit_pair_rate", "member_meet_rate"] + [f"{c}__{n}" for n in alt_names
                                                               for c in ("covisit_pair_rate", "member_meet_rate")]:
        base = nz[nz["method"] == "lcc_exact"].set_index("q")[metric]
        for m in methods:
            if m == "lcc_exact":
                continue
            other = nz[nz["method"] == m].set_index("q")[metric]
            common = base.index.intersection(other.index)
            if len(common) >= 10:
                dd = cluster_bootstrap_diff(base.loc[common].to_numpy(), other.loc[common].to_numpy(),
                                            np.asarray(common), seed=int(cfg["seed"]))
                paired.setdefault(metric, {})[m] = {"n": int(len(common)), **dd}
    extras = {"abstention": _abstention(df, list(g.get("abstention") or []), int(cfg["seed"]))}
    if groups:
        extras["member_level"] = _member_level(ctx, groups, int(cfg["seed"]))
    return df, summ, paired, info, extras


def e2(cfg: dict, paths: dict) -> dict:
    g = gcfg(cfg)
    R = _results_dir(cfg, paths)
    ctx = _e2_context(cfg, paths, "geo_e2" + (f"_{g['results_subdir']}" if g.get("results_subdir") else ""))
    k = int(cfg["search"]["k"])
    df, summ, paired, info, extras = _e2_eval(cfg, ctx, k, g.get("queries", "sample"),
                                              int(cfg["search"]["n_queries"]), "e2")
    df.to_csv(R / "geo_e2_per_query.csv", index=False)
    summ.to_csv(R / "geo_e2_summary.csv", index=False)
    out = {**info, "summary": summ.to_dict(orient="records"),
           "paired_covisit_pair_rate_vs_lcc_exact": paired.get("covisit_pair_rate", {}),
           "paired_member_meet_rate_vs_lcc_exact": paired.get("member_meet_rate", {}),
           "paired_alt_vs_lcc_exact": {m: v for m, v in paired.items() if "__" in m}, **extras}
    dump_json(out, R / "geo_e2.json")
    if not summ.empty:
        LOG.info("geo-e2 summary (k=%d, %s, %d queries):\n%s", k, ctx["tol"], info["queries"],
                 summ.round(4).to_string(index=False))
    return out


def e2_sweep(cfg: dict, paths: dict) -> dict:
    """Parameter study, rewritten after every setting so an interrupted sweep keeps what finished:
      * k in geo.e2_sweep_k (default geo.g3_k) with the G2-geo tolerance and geo.queries;
      * tolerance in geo.e2_sweep_tolerances (theta columns) at k = search.k, on a common sample
        of geo.sweep_tolerance_queries queries.
    Outputs geo_e2_sweep_summary.csv, geo_e2_sweep_per_query.csv, geo_e2_sweep.json."""
    g = gcfg(cfg)
    R = _results_dir(cfg, paths)
    sub = f"_{g['results_subdir']}" if g.get("results_subdir") else ""
    ks = [int(k) for k in (g.get("e2_sweep_k") or g["g3_k"])]
    tols = list(g.get("e2_sweep_tolerances") or [])
    k_mode = g.get("e2_sweep_k_queries") or g.get("queries", "sample")
    k_n = int(g.get("e2_sweep_k_n") or cfg["search"]["n_queries"])
    runs = [("k", k, None, k_mode, k_n) for k in ks]
    runs += [("tolerance", int(cfg["search"]["k"]), t, "sample", int(g.get("sweep_tolerance_queries", 2000)))
             for t in tols]
    out = {"runs": []}
    dfs, summs, ctxs = [], [], {}
    for sweep, k, tol, mode, n in runs:
        if tol not in ctxs:
            ctxs[tol] = _e2_context(cfg, paths, f"geo_e2_sweep{sub}" + (f"_{tol}" if tol else ""), tolerance=tol)
        ctx = ctxs[tol]
        df, summ, paired, info, extras = _e2_eval(cfg, ctx, k, mode, n, f"sw_{sweep}", member_analysis=False)
        out["runs"].append({"sweep": sweep, **info, "summary": summ.to_dict(orient="records"),
                            "paired_covisit_pair_rate_vs_lcc_exact": paired.get("covisit_pair_rate", {}),
                            "paired_member_meet_rate_vs_lcc_exact": paired.get("member_meet_rate", {}),
                            "paired_alt_vs_lcc_exact": {m: v for m, v in paired.items() if "__" in m}, **extras})
        if not df.empty:
            for d in (df, summ):
                d.insert(0, "tolerance", ctx["tol"])
                d.insert(0, "k", k)
                d.insert(0, "sweep", sweep)
            dfs.append(df)
            summs.append(summ)
            pd.concat(dfs, ignore_index=True).to_csv(R / "geo_e2_sweep_per_query.csv", index=False)
            pd.concat(summs, ignore_index=True).to_csv(R / "geo_e2_sweep_summary.csv", index=False)
        dump_json(out, R / "geo_e2_sweep.json")
        LOG.info("geo-e2 sweep: %s k=%d %s done (%d queries)", sweep, k, ctx["tol"], info["queries"])
    return out


# ------------------------------------------------------------- synthetic --

def synth(out_dir: Path, n_users: int = 600, n_days: int = 300, seed: int = 0, cities: int = 3) -> dict:
    """Small LBSN with a planted personal travel tolerance (tests and demos only)."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    centers = np.array([[40.0, -74.0], [37.8, -122.4], [41.9, -87.6], [34.0, -118.2]])[:cities]
    city = rng.integers(0, cities, size=n_users)
    home = centers[city] + rng.normal(scale=0.12, size=(n_users, 2))
    tol = np.exp(rng.normal(np.log(8.0), 0.6, size=n_users))  # km
    venues = np.concatenate([centers[c] + rng.normal(scale=0.18, size=(250, 2)) for c in range(cities)])
    vcity = np.repeat(np.arange(cities), 250)
    dist_uv = haversine_km(home[:, None, 0], home[:, None, 1], venues[None, :, 0], venues[None, :, 1])
    D = haversine_km(home[:, None, 0], home[:, None, 1], home[None, :, 0], home[None, :, 1])
    P = 0.25 * np.exp(-D / 12.0) * (city[:, None] == city[None, :]) + 0.0005
    iu = np.triu_indices(n_users, 1)
    e_mask = rng.random(len(iu[0])) < P[iu]
    eu, ev = iu[0][e_mask], iu[1][e_mask]
    t0 = 1_262_304_000  # 2010-01-01
    rows = []
    for u in range(n_users):
        w = np.exp(-dist_uv[u] / tol[u]) * (vcity == city[u])
        w = w / w.sum()
        n = rng.poisson(0.6 * n_days)
        days = rng.integers(0, n_days, size=n)
        vs = rng.choice(len(venues), size=n, p=w)
        for dday, vv in zip(days, vs):
            rows.append((u, t0 + int(dday) * DAY + int(rng.integers(8 * 3600, 22 * 3600)), int(vv)))
    tmin = np.minimum(tol[eu], tol[ev])
    p_meet = 0.08 * np.exp(-(D[eu, ev] / (2.0 * tmin)) ** 2)
    for dday in range(n_days):
        hit = rng.random(len(eu)) < p_meet
        for a, b in zip(eu[hit], ev[hit]):
            mid = (home[a] + home[b]) / 2
            vv = int(np.argmin(((venues - mid) ** 2).sum(1)))
            tt = t0 + dday * DAY + int(rng.integers(10 * 3600, 20 * 3600))
            rows.append((int(a), tt, vv))
            rows.append((int(b), tt + int(rng.integers(0, 1800)), vv))
    ck = pd.DataFrame(rows, columns=["user", "t", "v"])
    ck["time"] = pd.to_datetime(ck["t"], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    ck["lat"] = venues[ck["v"], 0] + rng.normal(scale=1e-4, size=len(ck))
    ck["lon"] = venues[ck["v"], 1] + rng.normal(scale=1e-4, size=len(ck))
    ck["loc"] = ck["v"].map(lambda x: f"v{x:05d}")
    ck = ck.sort_values(["user", "t"])
    ck[["user", "time", "lat", "lon", "loc"]].to_csv(out_dir / "checkins.txt", sep="\t", header=False, index=False,
                                                     float_format="%.6f")
    E = np.concatenate([np.column_stack([eu, ev]), np.column_stack([ev, eu])])
    np.savetxt(out_dir / "edges.txt", E, fmt="%d", delimiter="\t")
    np.savez(out_dir / "truth.npz", tolerance_km=tol, home=home)
    return {"users": n_users, "edges": int(len(eu)), "checkins": int(len(ck))}
