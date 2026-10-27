#!/usr/bin/env python3
"""E12 build (docs/PLANS.md (lacs_e12_plan_20261029) sections 2-3, as finalised 10-26):
positions, tolerances, nodes, the retweet graph and the test-period events of the European Parliament data
(MEP 1.0), written in the geo format that lacs.apps.geo, p0kit and e6 read.

  python -m e12.build --data DIR_WITH_MEP_FILES --out runs/e12 [--skip-consistency]

Definitions (plan section 3):
  * votes: FOR 0, AGAINST 1, NEUTRAL 2 are cast votes; UNACCOUNTED 5, EXTERNAL 6, SECRET 7 are missing.
    An MEP has a position with >= 20 FOR/AGAINST votes. No lopsided-vote filter.
  * agreement of i, j: share of the roll calls where both cast a vote with the same vote, over the roll calls
    where both cast one (>= 20 needed: observed distances); distance 1 - agreement. Pairs with < 20 common votes:
    min_k (d_ik + d_kj) over k observed with both (observed legs only, one pass, order free; may exceed 1);
    still missing: the largest observed distance (counted).
  * classical (Torgerson) MDS of all positioned MEPs with numpy.linalg.eigh, two dimensions; orientation before
    anything else: group means over all positioned MEPs, mean(S&D) < mean(EPP) on dim 1, mean(EFDD) > mean(EPP)
    on dim 2.
  * periods by retweetCreatedAt (UTC): train 2014-10-01 .. 2015-06-30, test 2015-07-01 .. 2016-03-01.
  * theta_raw of u: quantiles (linear interpolation; 0.5, 0.7, 0.9) of the distances from u to the distinct
    positioned MEPs u retweeted in the training period (>= 5), then the floor (1% of the dim-1 range, 1% of the
    largest 2-D distance among positioned MEPs; for every quantile). users.csv gets theta_eng = theta_raw *
    (1 + 1e-6) + 1e-6 (as E6), used by every accept/reject decision downstream.
  * nodes: positioned MEPs with >= 5 such training targets, indexed in increasing mepId; no mandate filter.
  * graph G: undirected union of the training retweets among nodes.
  * c_uv: distinct original tweets of v that u retweeted in the training period (a repeated retweet of the same
    original tweet counts once). Observed influence weight of u (listener) on v (speaker) = c_uv / mean c over
    all ordered MEP pairs with c > 0 (all MEPs); twitter_weights.npz src = speaker v, dst = listener u.
  * test events: every test retweet u -> v between nodes puts u and v at the key of the pair (no direction);
    notop: the original tweets whose number of distinct test-period retweeters (all MEPs) is >= the 99th
    percentile (linear) are removed; newpairs: only node pairs without any training retweet between them.
  * consistency (plan section 2): 783 positioned MEPs, 302 nodes, 2,273 edges of G, else stop
    (--skip-consistency only for synthetic test data).
  * report-only variant of Q1 (plan section 5): positions and theta_q50 from the roll calls up to 2015-06-30.
Outputs (--out): public/ (e12_members.csv, e12_retweets.csv.gz, e12_build.json, e12_positions.csv,
e12_positions_trainvotes.csv), geo/ (users.csv, users_trainvotes.csv, positions.npy, positions_2d.npy,
covisits.npz, test_checkins.npz, test_checkins_notop.npz, test_checkins_newpairs.npz, twitter_weights.npz,
e12_pairs.npz, meta.json), graph/graph.bin.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOG = logging.getLogger("e12.build")
CAST = (0, 1, 2)
FOR_AGAINST = (0, 1)
MIN_FOR_AGAINST = 20
MIN_COMMON = 20
MIN_TARGETS = 5
QUANTILES = (0.5, 0.7, 0.9)
QNAMES = ("q50", "q70", "q90")
FLOOR_SHARE = 0.01
ENGINE_TIE_RTOL = 1e-6
ENGINE_TIE_ATOL = 1e-6
TRAIN = ("2014-10-01", "2015-06-30")
TEST = ("2015-07-01", "2016-03-01")
NOTOP_PCT = 99.0
EXPECT = {"positioned": 783, "nodes": 302, "graph_edges": 2273}  # plan section 2 (gate and structure counts)
IN_OFFICE_DATE = "2015-07-01"
TRAIN_VOTES_END = "2015-06-30"
CMP_RTOL = 1e-9  # acceptance test downstream: d <= theta_eng * (1 + 1e-9) (e6.common.TIE_RTOL)
FILES = ["Readme.txt", "mep_info.json", "rcv_info.json", "action_info.csv", "mep_rcv_vote.csv", "retweets.csv"]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float))


# --------------------------------------------------------------- positions --
def agreement_distances(V: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """V: votes (MEPs x roll calls). Returns (D, common, info); D[i, j] = 1 - agreement, completed."""
    C = np.isin(V, CAST).astype(np.float64)
    same = np.zeros((V.shape[0], V.shape[0]))
    for a in CAST:
        X = (V == a).astype(np.float64)
        same += X @ X.T
    common = C @ C.T
    with np.errstate(invalid="ignore", divide="ignore"):
        D = 1.0 - same / common
    n = len(D)
    obs = common >= MIN_COMMON
    np.fill_diagonal(obs, True)
    D[~obs] = np.inf
    np.fill_diagonal(D, 0.0)
    miss = np.argwhere(np.triu(~obs, 1))
    filled, unfilled = 0, 0
    Dobs = D.copy()
    for i, j in miss:
        v = np.min(Dobs[i] + Dobs[:, j])
        if np.isfinite(v):
            D[i, j] = D[j, i] = v
            filled += 1
        else:
            unfilled += 1
    if unfilled:
        mx = float(Dobs[np.isfinite(Dobs)].max())
        D[~np.isfinite(D)] = mx
    comp = D[miss[:, 0], miss[:, 1]] if len(miss) else np.empty(0)
    info = {"pairs": int(n * (n - 1) // 2), "pairs_lt_min_common": int(len(miss)),
            "pairs_no_common_vote": int((np.triu(common, 1)[np.triu(~obs, 1)] == 0).sum()),
            "completed_two_step": filled, "completed_values_above_1": int((comp > 1).sum()),
            "set_to_max_observed": unfilled, "min_common": MIN_COMMON,
            "rule": "min_k (d_ik + d_kj) over k observed (>= 20 common votes) with both; one pass on observed distances"}
    return D, common, info


def classical_mds(D: np.ndarray, dims: int = 2) -> tuple[np.ndarray, dict]:
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    B = (B + B.T) / 2
    w, U = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    w, U = w[order], U[:, order]
    pos = w[w > 0].sum()
    X = U[:, :dims] * np.sqrt(np.maximum(w[:dims], 0.0))
    ratio23 = float(w[1] / w[2]) if len(w) > 2 and w[2] > 0 else None
    info = {"method": "numpy.linalg.eigh of B = -1/2 J D^2 J", "eigenvalues_top3": [float(x) for x in w[:3]],
            "share_of_positive": [float(x / pos) for x in w[:dims]],
            "share_of_positive_top2": float(w[:dims].sum() / pos),
            "negative_abs_share": float(np.abs(w[w < 0]).sum() / np.abs(w).sum()),
            "negative_eigenvalues": int((w < -1e-9 * abs(w[0])).sum()),
            "ratio_eig2_eig3": ratio23, "dim2_unstable": bool(ratio23 is not None and ratio23 < 1.05)}
    return X, info


def positions_from_votes(V: pd.DataFrame, mep: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Positioned MEPs (>= 20 FOR/AGAINST in V), their oriented 2-D MDS coordinates and the build info."""
    n_fa = V.isin(FOR_AGAINST).sum(axis=1)
    posd = np.sort(V.index[(n_fa >= MIN_FOR_AGAINST).to_numpy()].to_numpy().astype(np.int64))
    group = np.array([mep[str(int(m))]["politicalGroupShort"] for m in posd])
    D, _, dinfo = agreement_distances(V.loc[posd].to_numpy())
    X, minfo = classical_mds(D, 2)
    X, oinfo = orient(X, group)
    return posd, X, {"roll_calls": int(V.shape[1]), "meps_in_vote_matrix": int(len(V)), "positioned": int(len(posd)),
                     "min_for_against": MIN_FOR_AGAINST, "distances": dinfo, "mds": minfo, "orientation": oinfo}


def orient(X: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, dict]:
    X = X.copy()
    m = lambda g, j: float(X[groups == g, j].mean())  # noqa: E731
    flips = {}
    flips["dim1"] = bool(m("S&D", 0) > m("EPP", 0))
    if flips["dim1"]:
        X[:, 0] = -X[:, 0]
    flips["dim2"] = bool(m("EFDD", 1) < m("EPP", 1))
    if flips["dim2"]:
        X[:, 1] = -X[:, 1]
    return X, {"flipped": flips, "rule": "dim1: mean(S&D) < mean(EPP); dim2: mean(EFDD) > mean(EPP)",
               "group_means": {g: [float(X[groups == g, 0].mean()), float(X[groups == g, 1].mean())]
                               for g in sorted(set(groups))}}


# --------------------------------------------------------------- retweets --
def read_retweets(path: Path) -> pd.DataFrame:
    rt = pd.read_csv(path, dtype=str)
    t = pd.to_datetime(rt["retweetCreatedAt"], format="%a %b %d %H:%M:%S %z %Y", utc=True)
    day = t.dt.strftime("%Y-%m-%d")
    period = np.where((day >= TRAIN[0]) & (day <= TRAIN[1]), 0, np.where((day >= TEST[0]) & (day <= TEST[1]), 2, -1))
    return pd.DataFrame({"u": rt["retweetMepId"].astype(np.int64), "v": rt["origMepId"].astype(np.int64),
                         "time": t, "period": period, "orig_tweet": rt["origTweetId"].astype(str),
                         "tweet": rt["retweetTweetId"].astype(str)})


def tolerances(x: np.ndarray, targets: dict, idx_of: dict, floor: float, two_d: bool) -> pd.DataFrame:
    rows = {}
    for u, ts in targets.items():
        iu = idx_of[u]
        iv = np.array([idx_of[v] for v in ts], dtype=np.int64)
        d = np.sqrt(((x[iv] - x[iu]) ** 2).sum(axis=1)) if two_d else np.abs(x[iv] - x[iu])
        q = np.quantile(d, QUANTILES)  # numpy default: linear interpolation
        rows[u] = {"n": int(len(ts)), **{f"raw_{nm}": float(v) for nm, v in zip(QNAMES, q)},
                   **{nm: float(max(v, floor)) for nm, v in zip(QNAMES, q)}}
    return pd.DataFrame.from_dict(rows, orient="index")


# ------------------------------------------------------------------- main --
def in_office(mi: dict, day: str) -> bool:
    return (mi.get("from") or "0000") <= day and (mi.get("to") is None or mi["to"] >= day)


def tie_report(P: np.ndarray, th_raw: np.ndarray, th_eng: np.ndarray) -> dict:
    """Ordered node pairs with d == theta_raw (exact ties) and with theta_raw < d <= theta_eng * (1 + 1e-9)
    (accepted only because of the engine margin); d computed as in e12.premise from the stored positions."""
    d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    off = ~np.eye(len(P), dtype=bool)
    exact = (d == th_raw[:, None]) & off
    margin = (d > th_raw[:, None]) & (d <= th_eng[:, None] * (1 + CMP_RTOL)) & off
    return {"exact_ties": int(exact.sum()), "accepted_only_by_margin": int(margin.sum())}


def run(args) -> int:
    from lacs.graph.io import edges_to_csr, write_graph

    data, out = Path(args.data), Path(args.out)
    pub, geo, gdir = out / "public", out / "geo", out / "graph"
    for d in (pub, geo, gdir):
        d.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(pub / "build.log")], force=True)
    info = {"tool": "e12.build", "plan": "docs/PLANS.md (lacs_e12_plan_20261029) (finalised 2026-10-26)",
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "inputs": {f: {"sha256": sha256(data / f), "bytes": (data / f).stat().st_size} for f in FILES}}
    mep = json.loads((data / "mep_info.json").read_text())
    V = pd.read_csv(data / "mep_rcv_vote.csv", index_col=0)
    V.index = V.index.astype(np.int64)
    vals = set(np.unique(V.to_numpy()).tolist())
    if not vals <= {0, 1, 2, 5, 6, 7}:
        raise SystemExit(f"unexpected vote codes {sorted(vals)}")
    rcv = json.loads((data / "rcv_info.json").read_text())
    rdate = pd.Series({str(k): v["rcvDate"] for k, v in rcv.items()}).reindex(V.columns.astype(str))
    if rdate.isna().any():
        raise SystemExit(f"{int(rdate.isna().sum())} roll calls without rcvDate")
    n_fa = V.isin(FOR_AGAINST).sum(axis=1)
    posd, X, pinfo = positions_from_votes(V, mep)
    group = np.array([mep[str(int(m))]["politicalGroupShort"] for m in posd])
    pinfo["roll_call_dates"] = [str(rdate.min()), str(rdate.max())]
    pinfo["roll_calls_test_window"] = int((rdate >= TEST[0]).sum())
    info["positions"] = pinfo
    LOG.info("positions: %d MEPs, MDS top-2 share of positive %.4f, eigenvalues %s, completed pairs %s", len(posd),
             pinfo["mds"]["share_of_positive_top2"], np.round(pinfo["mds"]["eigenvalues_top3"], 4).tolist(),
             pinfo["distances"]["completed_two_step"])
    idx_pos = {int(m): i for i, m in enumerate(posd)}
    x1 = X[:, 0]
    floor1 = FLOOR_SHARE * float(x1.max() - x1.min())
    diam2 = max(float(np.sqrt(((X[i] - X) ** 2).sum(axis=1)).max()) for i in range(len(X)))
    floor2 = FLOOR_SHARE * diam2
    # retweets
    rt = read_retweets(data / "retweets.csv")
    info["retweets"] = {"rows": int(len(rt)), "self": int((rt["u"] == rt["v"]).sum()),
                        "by_period": {str(k): int(v) for k, v in rt["period"].value_counts().items()},
                        "last": str(rt["time"].max()), "first": str(rt["time"].min()),
                        "repeated_retweeter_original_rows": int(rt.duplicated(["u", "orig_tweet"]).sum())}
    rt = rt[rt["u"] != rt["v"]]
    both_pos = rt["u"].isin(idx_pos) & rt["v"].isin(idx_pos)
    tr = rt[(rt["period"] == 0) & both_pos]
    targets = {int(u): sorted(set(int(v) for v in g)) for u, g in tr.groupby("u")["v"]}
    t1 = tolerances(x1, targets, idx_pos, floor1, False)
    t2 = tolerances(X, targets, idx_pos, floor2, True)
    nodes = np.sort(np.array([u for u, ts in targets.items() if len(ts) >= MIN_TARGETS], dtype=np.int64))
    n = len(nodes)
    idx_of = {int(m): i for i, m in enumerate(nodes)}
    left_before = [int(m) for m in nodes if mep[str(int(m))].get("to") and mep[str(int(m))]["to"] < TEST[0]]
    left_test = [int(m) for m in nodes if mep[str(int(m))].get("to") and TEST[0] <= mep[str(int(m))]["to"] <= TEST[1]]
    info["nodes"] = {"n": n, "min_targets": MIN_TARGETS, "floor_1d": floor1, "floor_2d": floor2,
                     "floor_hits_1d": {q: int((t1.loc[nodes, f"raw_{q}"] < floor1).sum()) for q in QNAMES},
                     "floor_hits_2d": {q: int((t2.loc[nodes, f"raw_{q}"] < floor2).sum()) for q in QNAMES},
                     "left_office_before_test": len(left_before), "left_office_in_test": len(left_test),
                     "in_office_2015_07_01": int(sum(in_office(mep[str(int(m))], IN_OFFICE_DATE) for m in nodes)),
                     "groups": {g: int(c) for g, c in pd.Series([mep[str(int(m))]["politicalGroupShort"]
                                                              for m in nodes]).value_counts().items()}}
    LOG.info("nodes: %d MEPs with >= %d positioned training targets", n, MIN_TARGETS)
    # members table (all MEPs of the vote matrix)
    rows = []
    for m in V.index:
        mi = mep[str(int(m))]
        r = {"mepId": int(m), "fullName": mi.get("fullName"), "group": mi.get("politicalGroupShort"),
             "country": mi.get("countryShort"), "from": mi.get("from"), "to": mi.get("to"),
             "n_cast": int(V.loc[m].isin(CAST).sum()), "n_for_against": int(n_fa.loc[m]),
             "positioned": bool(int(m) in idx_pos), "node": bool(int(m) in idx_of), "idx": idx_of.get(int(m), -1)}
        if int(m) in idx_pos:
            r["dim1"], r["dim2"] = float(X[idx_pos[int(m)], 0]), float(X[idx_pos[int(m)], 1])
        if int(m) in t1.index:
            r["n_train_targets"] = int(t1.at[int(m), "n"])
            for q in QNAMES:
                r[f"theta_raw_{q}_1d"] = float(t1.at[int(m), f"raw_{q}"])
                r[f"theta_{q}_1d"] = float(t1.at[int(m), q])
                r[f"theta_raw_{q}_2d"] = float(t2.at[int(m), f"raw_{q}"])
                r[f"theta_{q}_2d"] = float(t2.at[int(m), q])
        rows.append(r)
    members = pd.DataFrame(rows).sort_values("mepId")
    members.to_csv(pub / "e12_members.csv", index=False)
    pd.DataFrame({"mepId": posd, "dim1": X[:, 0], "dim2": X[:, 1], "group": group}).to_csv(pub / "e12_positions.csv",
                                                                                          index=False)
    # node arrays; theta_eng = (floored) theta_raw * (1 + 1e-6) + 1e-6
    P = np.array([X[idx_pos[int(m)]] for m in nodes])
    users = pd.DataFrame({"idx": np.arange(n), "user": nodes, "n_train": [int(t1.at[int(m), "n"]) for m in nodes]})
    for q in QNAMES:
        users[f"theta_{q}"] = np.array([t1.at[int(m), q] for m in nodes]) * (1 + ENGINE_TIE_RTOL) + ENGINE_TIE_ATOL
    for q in QNAMES:
        users[f"theta2d_{q}"] = np.array([t2.at[int(m), q] for m in nodes]) * (1 + ENGINE_TIE_RTOL) + ENGINE_TIE_ATOL
    users["n_train_2d"] = users["n_train"]
    users["dim1"], users["dim2"] = P[:, 0], P[:, 1]
    users["group"] = [mep[str(int(m))]["politicalGroupShort"] for m in nodes]
    users["country"] = [mep[str(int(m))]["countryShort"] for m in nodes]
    users["name"] = [mep[str(int(m))].get("fullName") for m in nodes]
    users["office_from"] = [mep[str(int(m))].get("from") for m in nodes]
    users["office_to"] = [mep[str(int(m))].get("to") for m in nodes]
    users["in_office_20150701"] = [int(in_office(mep[str(int(m))], IN_OFFICE_DATE)) for m in nodes]
    users.to_csv(geo / "users.csv", index=False)
    pos1 = np.column_stack([P[:, 0], np.zeros(n), np.zeros(n)]).astype(np.float64)
    pos2 = np.column_stack([P[:, 0], P[:, 1], np.zeros(n)]).astype(np.float64)
    np.save(geo / "positions.npy", pos1)
    np.save(geo / "positions_2d.npy", pos2)
    # exact ties and acceptances due to the engine margin only (plan section 3.3), on the values written
    uu = pd.read_csv(geo / "users.csv", float_precision="round_trip")
    ties = {}
    for q in QNAMES:
        ties[f"1d_{q}"] = tie_report(pos1, np.array([t1.at[int(m), q] for m in nodes]), uu[f"theta_{q}"].to_numpy())
    ties["2d_q50"] = tie_report(pos2, np.array([t2.at[int(m), "q50"] for m in nodes]), uu["theta2d_q50"].to_numpy())
    info["ties_and_margin"] = {"rtol": ENGINE_TIE_RTOL, "atol": ENGINE_TIE_ATOL, "compare_rtol": CMP_RTOL,
                               "ordered_pairs": int(n * (n - 1)), **ties}
    # graph and events among nodes
    isn = rt["u"].isin(idx_of) & rt["v"].isin(idx_of)
    ev = rt[isn].copy()
    ev["ui"] = ev["u"].map(idx_of).astype(np.int64)
    ev["vi"] = ev["v"].map(idx_of).astype(np.int64)
    trn, tst = ev[ev["period"] == 0], ev[ev["period"] == 2]
    A = edges_to_csr(n, trn["ui"].to_numpy(), trn["vi"].to_numpy())
    n_edges = int(A.nnz // 2)
    # consistency with the gate (plan section 2)
    got = {"positioned": int(len(posd)), "nodes": n, "graph_edges": n_edges}
    info["consistency"] = {"expected": EXPECT, "got": got, "ok": got == EXPECT, "skipped": bool(args.skip_consistency)}
    if got != EXPECT and not args.skip_consistency:
        info["stopped"] = "consistency check failed: counts differ from the gate (plan section 2)"
        write_json(info, pub / "e12_build.json")
        LOG.error("consistency check failed: expected %s, got %s -- stop", EXPECT, got)
        return 3
    write_graph(gdir / "graph.bin", A)
    ev.assign(period=ev["period"].map({0: "train", 2: "test", -1: "other"}))[
        ["u", "v", "ui", "vi", "time", "period", "orig_tweet", "tweet"]].to_csv(pub / "e12_retweets.csv.gz", index=False,
                                                                             compression="gzip")
    # training counts c_uv: distinct original tweets (repeated retweets of the same original tweet count once)
    trn_dd = trn.drop_duplicates(["u", "orig_tweet"])
    cnt_tr = trn_dd.groupby(["ui", "vi"]).size()
    cnt_te = tst.groupby(["ui", "vi"]).size()
    np.savez(geo / "e12_pairs.npz", train_u=np.array([a for a, _ in cnt_tr.index], np.int64),
             train_v=np.array([b for _, b in cnt_tr.index], np.int64), train_n=cnt_tr.to_numpy(np.int64),
             test_u=np.array([a for a, _ in cnt_te.index], np.int64),
             test_v=np.array([b for _, b in cnt_te.index], np.int64), test_n=cnt_te.to_numpy(np.int64))
    # observed influence weights (speaker v -> listener u), normalised by the mean c over all MEPs' ordered pairs
    all_tr = rt[rt["period"] == 0].drop_duplicates(["u", "orig_tweet"]).groupby(["u", "v"]).size()
    norm_mean = float(all_tr.mean())
    np.savez(geo / "twitter_weights.npz", src=np.array([b for _, b in cnt_tr.index], np.int64),
             dst=np.array([a for a, _ in cnt_tr.index], np.int64), w=cnt_tr.to_numpy(np.float64),
             norm_mean=np.float64(norm_mean))
    # test events as co-visit keys (pair id, no direction)
    a, b = tst["ui"].to_numpy(), tst["vi"].to_numpy()
    key = np.minimum(a, b) * n + np.maximum(a, b)
    tu = np.concatenate([a, b]).astype(np.int32)
    tk = np.concatenate([key, key]).astype(np.int64)
    np.savez(geo / "test_checkins.npz", user=tu, key=tk)
    # notop (E6 10-25 / E5 rule): distinct test-period retweeters over all MEPs, remove counts >= 99th percentile
    tst_all = rt[rt["period"] == 2]
    per_orig = tst_all.groupby("orig_tweet")["u"].nunique()
    thr = float(np.quantile(per_orig.to_numpy(), NOTOP_PCT / 100.0)) if len(per_orig) else float("inf")
    top = set(per_orig.index[per_orig.to_numpy() >= thr])
    keep = ~tst["orig_tweet"].isin(top).to_numpy()
    np.savez(geo / "test_checkins_notop.npz", user=np.concatenate([a[keep], b[keep]]).astype(np.int32),
             key=np.concatenate([key[keep], key[keep]]).astype(np.int64))
    notop = {"rule": "distinct test-period retweeters per original tweet over all MEPs; threshold = 99th percentile "
                     "(numpy linear); original tweets with count >= threshold removed (E6 10-25, E5)",
             "threshold_retweeters": thr, "original_tweets_with_test_retweets": int(len(per_orig)),
             "original_tweets_removed": int(len(top)), "share_removed": float(len(top) / max(len(per_orig), 1)),
             "test_retweets_removed_all_meps": int(tst_all["orig_tweet"].isin(top).sum()),
             "test_retweets_removed_nodes": int((~keep).sum()), "test_retweets_kept_nodes": int(keep.sum())}
    # newpairs: test retweets of node pairs without any training retweet between them
    tr_keys = set((np.minimum(trn["ui"], trn["vi"]) * n + np.maximum(trn["ui"], trn["vi"])).tolist())
    keep_np = ~np.isin(key, np.array(sorted(tr_keys), dtype=np.int64))
    np.savez(geo / "test_checkins_newpairs.npz", user=np.concatenate([a[keep_np], b[keep_np]]).astype(np.int32),
             key=np.concatenate([key[keep_np], key[keep_np]]).astype(np.int64))
    newpairs = {"rule": "test retweets between node pairs with no training retweet in either direction",
                "test_retweets_kept": int(keep_np.sum()), "test_retweets_removed": int((~keep_np).sum()),
                "pairs_kept": int(len(np.unique(key[keep_np])))}
    # co-visits of friends per period (only read by geo's gates, which E12 does not run)
    up = np.triu(A.toarray() > 0, 1) if n <= 5000 else None
    fr = set(zip(*np.nonzero(up))) if up is not None else set()

    def friend_codes(df):
        p = {(min(x, y), max(x, y)) for x, y in zip(df["ui"], df["vi"])}
        return np.array(sorted(x * n + y for x, y in p if (x, y) in fr), dtype=np.int64)

    np.savez(geo / "covisits.npz", train=friend_codes(trn), val=np.empty(0, np.int64), test=friend_codes(tst))
    # report-only variant of Q1: positions and theta_q50 (1-D) from the roll calls up to 2015-06-30
    tv_cols = V.columns[(rdate <= TRAIN_VOTES_END).to_numpy()]
    posd_tv, X_tv, pinfo_tv = positions_from_votes(V[tv_cols], mep)
    idx_tv = {int(m): i for i, m in enumerate(posd_tv)}
    x1_tv = X_tv[:, 0]
    floor_tv = FLOOR_SHARE * float(x1_tv.max() - x1_tv.min())
    tr_tv = rt[(rt["period"] == 0) & rt["u"].isin(idx_tv) & rt["v"].isin(idx_tv)]
    targets_tv = {int(u): sorted(set(int(v) for v in g)) for u, g in tr_tv.groupby("u")["v"]}
    keep_tv = [int(m) for m in nodes if len(targets_tv.get(int(m), [])) >= MIN_TARGETS]
    t_tv = tolerances(x1_tv, {m: targets_tv[m] for m in keep_tv}, idx_tv, floor_tv, False)
    utv = pd.DataFrame({"idx": [idx_of[m] for m in keep_tv], "user": keep_tv,
                        "n_train": [int(t_tv.at[m, "n"]) for m in keep_tv],
                        "theta_raw_q50": [float(t_tv.at[m, "raw_q50"]) for m in keep_tv],
                        "theta_q50": [float(t_tv.at[m, "q50"]) * (1 + ENGINE_TIE_RTOL) + ENGINE_TIE_ATOL
                                      for m in keep_tv],
                        "dim1": [float(X_tv[idx_tv[m], 0]) for m in keep_tv],
                        "dim2": [float(X_tv[idx_tv[m], 1]) for m in keep_tv]})
    utv.to_csv(geo / "users_trainvotes.csv", index=False)
    pd.DataFrame({"mepId": posd_tv, "dim1": X_tv[:, 0], "dim2": X_tv[:, 1]}).to_csv(
        pub / "e12_positions_trainvotes.csv", index=False)
    no_pos = [int(m) for m in nodes if int(m) not in idx_tv]
    info["trainvotes_variant"] = {**pinfo_tv, "roll_calls_up_to": TRAIN_VOTES_END, "floor_1d": floor_tv,
                                  "nodes_kept": len(keep_tv), "nodes_without_position": len(no_pos),
                                  "nodes_lt_min_targets": int(n - len(keep_tv) - len(no_pos))}
    deg = np.diff(A.indptr)
    meta = {"application": "e12_european_parliament", "n_active": n, "n_edges": n_edges,
            "n_nonisolated": int((deg > 0).sum()), "tolerance_quantiles": list(QUANTILES),
            "tolerance_floor_1d": floor1, "tolerance_floor_2d": floor2,
            "theta_tie_inflation": {"rtol": ENGINE_TIE_RTOL, "atol": ENGINE_TIE_ATOL},
            "tolerance_median": {q: float(np.median(users[f"theta_{q}"])) for q in QNAMES},
            "periods": {"train": TRAIN, "test": TEST},
            "test_events": {"retweets": int(len(tst)), "pairs": int(len(np.unique(key))),
                            "active_nodes": int(len(np.unique(tu))), "notop": notop, "newpairs": newpairs},
            "train_retweets_nodes": int(len(trn)), "observed_weight_norm_mean": norm_mean,
            "positions": "positions.npy = (dim1, 0, 0); positions_2d.npy = (dim1, dim2, 0): classical MDS of vote "
                         "agreement distances", "member_order": "increasing mepId (users.csv 'user')"}
    write_json(meta, geo / "meta.json")
    info["graph"] = {"nodes": n, "edges": n_edges, "nonisolated": meta["n_nonisolated"],
                     "mean_degree": float(deg.mean()) if n else None}
    info["events"] = meta["test_events"]
    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(info, pub / "e12_build.json")
    LOG.info("e12 build done: %d nodes, %d edges, %d test retweets among nodes", n, n_edges, len(tst))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m e12.build", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-consistency", action="store_true",
                    help="synthetic test data only: do not stop when the counts differ from the gate")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
