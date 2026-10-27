#!/usr/bin/env python
"""Feasibility probe for the debate.org (DDO) corpus of Durmus & Cardie (ACL 2019).

Why: DEBAGREEMENT has very few judgements per receiver and only text-derived
positions. DDO has self-reported stances on 48 "big issues" (explicit
positions), a few hundred thousand votes (dense judgements) and user
interactions (a graph). Before any loader is written, this script checks four
criteria that were fixed in advance:

  P1 data      >= 2,000 active voters (>= 10 decisive "made more convincing
               arguments" votes whose two debaters differ in ideological
               distance to the voter, and >= 10 stated issues), covering >= 40%
               of such votes
  P2 G1-proxy  the convincing-arguments point goes to the ideologically closer
               debater more often than chance (voter-cluster 95% CI above 0.5)
  P3 G2-proxy  per-voter rates (shrunk towards the global rate) predict each
               active voter's later votes better than the global rate
               (voter-cluster 95% CI of the NLL difference below 0)
  P4 graph     among users with >= 10 stated issues, the 3-core of the
               interaction graph (votes, debates, comments) has >= 2,000 users,
               >= 500 of them active voters

Verdict: GO_FULL (P1-P4 pass), GO_GLOBAL (P1, P2, P4 pass: positions carry
signal but personal latitudes are not supported), NO_GO otherwise.

Only aggregates are written: no user names, no profile values. The corpus is
CC BY-NC-SA 3.0 according to its original release page (research use only; do
not redistribute it or anything user-level derived from it).

usage:
    python scripts/probe_ddo.py --data data/raw/ddo            # finds *users*.json and *debates*.json
    python scripts/probe_ddo.py --users U.json --debates D.json --out runs/ddo_probe
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RULES = {"min_shared_issues": 5, "P1_min_votes": 10, "P1_min_issues": 10, "P1_min_voters": 2000,
         "P1_min_cover": 0.40, "P2_chance": 0.5, "P3_shrink": 10.0, "P3_train_frac": 0.7, "P4_k": 3,
         "P4_min_core": 2000, "P4_min_active_in_core": 500}
STANCE = {"pro": 1, "con": -1}
STANCE_WORDS = {"pro", "con", "n/o", "n/s", "und", "undecided", "no opinion", "not saying", "n/a", ""}


class SchemaError(RuntimeError):
    pass


# ------------------------------------------------------------------ helpers --

def truthy(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return x != 0
    if isinstance(x, str):
        return x.strip().lower() in {"true", "yes", "y", "t", "1"}
    return False


def records(obj, name_keys=("user_name", "username", "name", "url", "id")):
    """[(key, record)] for a dict keyed by name/url or a list of records."""
    if isinstance(obj, dict):
        return [(str(k), v) for k, v in obj.items()]
    if isinstance(obj, list):
        out = []
        for i, rec in enumerate(obj):
            key = next((rec.get(k) for k in name_keys if isinstance(rec, dict) and rec.get(k) is not None), i)
            out.append((str(key), rec))
        return out
    raise SchemaError(f"top-level JSON is {type(obj).__name__}, expected dict or list")


def shape_of(x, depth=0, max_keys=60, redact=frozenset()):
    """Structure only: keys and types, never values. Keys that are user names are
    replaced by <user>, so no user data leaves the script."""
    if isinstance(x, dict):
        if depth >= 3:
            return f"dict[{len(x)}]"
        out = {}
        for k in list(x.keys())[:max_keys]:
            name = "<user>" if str(k).strip() in redact else str(k)
            out[name] = shape_of(x[k], depth + 1, max_keys, redact)
        return out
    if isinstance(x, list):
        return [shape_of(x[0], depth + 1, max_keys, redact)] if x else []
    return type(x).__name__


def cluster_ci(values, clusters, n_boot=2000, seed=0):
    """Mean of `values` with a 95% CI that resamples clusters."""
    uniq, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=values, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(uniq), size=len(uniq))
        boots[b] = sums[idx].sum() / max(cnts[idx].sum(), 1.0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(np.mean(values)), [float(lo), float(hi)], int(len(uniq))


def parse_times(raw) -> np.ndarray:
    s = pd.Series(raw, dtype="object")
    try:
        t = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
    except (TypeError, ValueError):
        t = pd.to_datetime(s, errors="coerce", utc=True)
    return (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy(dtype=float)


# ------------------------------------------------------------------ parsing --

def parse_users(users_obj, issue_key=None):
    recs = records(users_obj)
    if issue_key is None:
        for _, prof in recs[:500]:
            if not isinstance(prof, dict):
                continue
            for k, v in prof.items():
                if isinstance(v, dict) and len(v) >= 10:
                    vals = [str(x).strip().lower() for x in v.values()]
                    if any(x in STANCE for x in vals) and np.mean([x in STANCE_WORDS for x in vals]) >= 0.8:
                        issue_key = k
                        break
            if issue_key:
                break
    if issue_key is None:
        raise SchemaError("no big-issues field found in the user records (pass --issue-key)")
    issues = sorted({str(i) for _, p in recs if isinstance(p, dict) and isinstance(p.get(issue_key), dict)
                     for i in p[issue_key]})
    col = {i: j for j, i in enumerate(issues)}
    names = [n.strip() for n, _ in recs]
    X = np.zeros((len(names), len(issues)), np.int8)
    for r, (_, p) in enumerate(recs):
        d = p.get(issue_key) if isinstance(p, dict) else None
        if isinstance(d, dict):
            for i, val in d.items():
                s = STANCE.get(str(val).strip().lower())
                if s:
                    X[r, col[str(i)]] = s
    return names, X, issue_key, issues


def debate_keys(rec: dict) -> dict:
    keys = list(rec.keys())
    nk = sorted(k for k in keys if re.search(r"participant_?\d_?name", k, re.I))
    pk = sorted(k for k in keys if re.search(r"participant_?\d_?position", k, re.I))
    return {"names": nk[:2], "positions": pk[:2],
            "votes": next((k for k in keys if k.lower() in ("votes", "vote_list")), None),
            "comments": next((k for k in keys if k.lower() == "comments"), None),
            "date": next((k for k in keys if "date" in k.lower()), None)}


def vote_map(v: dict, names: list[str]):
    """The per-debater criteria map of one vote, keyed by stripped debater name."""
    cand = v.get("votes_map")
    if not isinstance(cand, dict):
        cand = next((x for x in v.values() if isinstance(x, dict) and any(str(k).strip() in names for k in x)), None)
    if not isinstance(cand, dict):
        return None
    return {str(k).strip(): val for k, val in cand.items() if isinstance(val, dict)}


def parse_debates(debates_obj, uid: dict, criterion: str):
    recs = records(debates_obj)
    first = next((r for _, r in recs if isinstance(r, dict)), None)
    if first is None:
        raise SchemaError("no debate records")
    K = debate_keys(first)
    if len(K["names"]) < 2 or K["votes"] is None:
        raise SchemaError(f"debate records lack participant names or votes; keys seen: {sorted(first.keys())[:40]}")
    crit_key = None
    votes, edges = [], []
    n_votes_raw = n_unparsed = 0
    for order, (_, d) in enumerate(recs):
        if not isinstance(d, dict):
            continue
        p = [str(d.get(k) or "").strip() for k in K["names"]]
        i1, i2 = uid.get(p[0]), uid.get(p[1])
        if i1 is not None and i2 is not None:
            edges.append((i1, i2))
        date = d.get(K["date"]) if K["date"] else None
        for v in d.get(K["votes"]) or []:
            n_votes_raw += 1
            if not isinstance(v, dict):
                n_unparsed += 1
                continue
            voter = str(v.get("user_name") or v.get("username") or v.get("voter") or "").strip()
            vm = vote_map(v, p)
            if not voter or vm is None or p[0] not in vm or p[1] not in vm:
                n_unparsed += 1
                continue
            if crit_key is None:
                crit_key = next((k for k in vm[p[0]] if criterion in str(k).lower()), None)
                if crit_key is None:
                    raise SchemaError(f"no vote criterion containing '{criterion}'; criteria seen: "
                                      f"{sorted(vm[p[0]].keys())}")
            c1, c2 = truthy(vm[p[0]].get(crit_key)), truthy(vm[p[1]].get(crit_key))
            award = 1 if (c1 and not c2) else 2 if (c2 and not c1) else 0
            iv = uid.get(voter)
            votes.append((iv, i1, i2, award, v.get("time"), date, order))
            if iv is not None:
                for ip in (i1, i2):
                    if ip is not None:
                        edges.append((iv, ip))
        if K["comments"]:
            for c in d.get(K["comments"]) or []:
                ic = uid.get(str(c.get("user_name") or "").strip()) if isinstance(c, dict) else None
                if ic is not None:
                    for ip in (i1, i2):
                        if ip is not None:
                            edges.append((ic, ip))
    cols = ["voter", "p1", "p2", "award", "time", "date", "order"]
    V = pd.DataFrame(votes, columns=cols)
    info = {"debate_keys": K, "criterion_key": crit_key, "n_debates": len(recs), "n_votes_raw": n_votes_raw,
            "n_votes_unparsed": n_unparsed, "n_votes_parsed": len(V)}
    return V, np.array(edges, dtype=np.int64).reshape(-1, 2), info


# -------------------------------------------------------------------- probe --

def distances(X, a, b, min_shared):
    S = (X[a] != 0) & (X[b] != 0)
    n = S.sum(axis=1)
    dis = ((X[a] != X[b]) & S).sum(axis=1)
    return np.where(n >= min_shared, dis / np.maximum(n, 1), np.nan)


def run(args) -> dict:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = Path(args.data) if args.data else None
    upath = Path(args.users) if args.users else next(iter(sorted(data.glob("*users*.json"))), None)
    dpath = Path(args.debates) if args.debates else next(iter(sorted(data.glob("*debates*.json"))), None)
    if upath is None or dpath is None:
        raise SystemExit("could not find users/debates JSON files; pass --users and --debates")
    t0 = time.time()
    users_obj = json.loads(Path(upath).read_text(encoding="utf-8"))
    debates_obj = json.loads(Path(dpath).read_text(encoding="utf-8"))
    ur, dr = records(users_obj), records(debates_obj)
    known = frozenset(n.strip() for n, _ in ur)
    schema = {"users": {"n": len(ur), "record": shape_of(ur[0][1], redact=known) if ur else None},
              "debates": {"n": len(dr), "record": shape_of(dr[0][1], redact=known) if dr else None}}
    for _, d in dr[:50]:  # one vote's criteria map, one level deeper than the record dump
        vs = d.get("votes") if isinstance(d, dict) else None
        if vs and isinstance(vs[0], dict):
            vm = vs[0].get("votes_map", vs[0])
            schema["first_vote_map"] = shape_of(vm, redact=known)
            break
    (out_dir / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    res = {"rules": RULES, "files": {"users": str(upath), "debates": str(dpath)}}
    try:
        names, X, issue_key, issues = parse_users(users_obj, args.issue_key)
        uid = {n: i for i, n in enumerate(names)}
        V, E, info = parse_debates(debates_obj, uid, args.criterion.lower())
    except SchemaError as e:
        res["schema_error"] = str(e)
        res["verdict"] = "SCHEMA"
        (out_dir / "probe.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"[probe] schema not recognised: {e}\n[probe] structure written to {out_dir / 'schema.json'}")
        return res
    del users_obj, debates_obj
    stated = (X != 0).sum(axis=1)
    res["users"] = {"n": len(names), "issue_key": issue_key, "n_issues": len(issues),
                    "with_10plus_stated": int((stated >= 10).sum()), "stated_mean": float(stated.mean())}
    res["votes"] = {**info, "award_share": {str(int(k)): float(v) for k, v in
                                            V["award"].value_counts(normalize=True).sort_index().items()}}

    # usable votes: decisive, all three users known, distances defined and different
    known = V[["voter", "p1", "p2"]].notna().all(axis=1).to_numpy()
    W = V[known & (V["award"] > 0).to_numpy()].copy()
    for c in ("voter", "p1", "p2"):
        W[c] = W[c].astype(int)
    d1 = distances(X, W["voter"].to_numpy(), W["p1"].to_numpy(), RULES["min_shared_issues"])
    d2 = distances(X, W["voter"].to_numpy(), W["p2"].to_numpy(), RULES["min_shared_issues"])
    keep = np.isfinite(d1) & np.isfinite(d2) & (d1 != d2)
    W = W[keep].copy()
    W["gap"] = np.abs(d1[keep] - d2[keep])
    W["closer"] = np.where(d1[keep] < d2[keep], 1, 2)
    W["y"] = (W["award"] == W["closer"]).astype(float)
    res["usable_votes"] = int(len(W))
    if len(W) == 0:
        res["verdict"] = "NO_GO"
        res["note"] = "no decisive vote with ideological distances to both debaters"
        (out_dir / "probe.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        return res

    # P1 density
    n_u = W["voter"].value_counts()
    active_ids = n_u[(n_u >= RULES["P1_min_votes"]) & (stated[n_u.index.to_numpy()] >= RULES["P1_min_issues"])].index
    cover = float(n_u[active_ids].sum() / len(W))
    res["P1"] = {"active_voters": int(len(active_ids)), "coverage": cover,
                 "votes_per_voter_p50_p90_p99": [float(x) for x in np.percentile(n_u.to_numpy(), [50, 90, 99])],
                 "pass": bool(len(active_ids) >= RULES["P1_min_voters"] and cover >= RULES["P1_min_cover"])}

    # P2 closer-debater rate
    rate, ci, n_cl = cluster_ci(W["y"].to_numpy(), W["voter"].to_numpy())
    res["P2"] = {"rate_award_closer": rate, "ci95": ci, "n_voters": n_cl, "pass": bool(ci[0] > RULES["P2_chance"])}
    bins = [0, 0.1, 0.2, 0.3, 0.5, 1.01]
    W["gap_bin"] = pd.cut(W["gap"], bins, right=False).astype(str)
    dose = W.groupby("gap_bin")["y"].agg(["mean", "size"]).reset_index()
    dose.to_csv(out_dir / "dose_response.csv", index=False)
    res["P2"]["dose_response"] = dose.to_dict(orient="records")

    # P3 personal rates vs global rate on each active voter's later votes
    A = W[W["voter"].isin(active_ids)].copy()
    t = parse_times(A["time"].to_numpy())
    t_d = parse_times(A["date"].to_numpy())
    A["t"] = np.where(np.isfinite(t), t, t_d)
    res["P3_time_parsed_share"] = float(np.isfinite(t).mean()) if len(A) else 0.0
    A = A.sort_values(["voter", "t", "order"], na_position="last", kind="mergesort")
    A["rank"] = A.groupby("voter").cumcount()
    A["n"] = A.groupby("voter")["y"].transform("size")
    A["is_train"] = A["rank"] < np.floor(RULES["P3_train_frac"] * A["n"])
    trn, tst = A[A["is_train"]], A[~A["is_train"]]
    if len(trn) and len(tst):
        g = float(np.clip(trn["y"].mean(), 1e-4, 1 - 1e-4))
        agg = trn.groupby("voter")["y"].agg(["sum", "size"])
        m = RULES["P3_shrink"]
        p_u = ((agg["sum"] + m * g) / (agg["size"] + m)).clip(1e-4, 1 - 1e-4)
        p = tst["voter"].map(p_u).fillna(g).to_numpy()
        yt = tst["y"].to_numpy()
        nll_p = -(yt * np.log(p) + (1 - yt) * np.log(1 - p))
        nll_g = -(yt * np.log(g) + (1 - yt) * np.log(1 - g))
        mean, ci3, n3 = cluster_ci(nll_p - nll_g, tst["voter"].to_numpy())
        res["P3"] = {"nll_diff_personal_minus_global": mean, "ci95": ci3, "n_voters": n3,
                     "test_votes": int(len(tst)), "pass": bool(ci3[1] < 0)}
    else:
        res["P3"] = {"pass": False, "note": "no active voters"}

    # P4 graph among users with >= 10 stated issues
    import networkx as nx

    pos_ok = stated >= RULES["P1_min_issues"]
    E2 = E[(E[:, 0] != E[:, 1]) & pos_ok[E[:, 0]] & pos_ok[E[:, 1]]] if len(E) else E
    G = nx.Graph()
    G.add_edges_from(map(tuple, E2.tolist()))
    core = nx.core_number(G) if G.number_of_nodes() else {}
    in_core = {n for n, c in core.items() if c >= RULES["P4_k"]}
    act_set = set(int(a) for a in active_ids)
    comps = sorted((len(c) for c in nx.connected_components(G)), reverse=True) if G.number_of_nodes() else [0]
    res["P4"] = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "largest_component": int(comps[0]),
                 f"{RULES['P4_k']}_core_size": len(in_core), "active_voters_in_core": len(in_core & act_set),
                 "pass": bool(len(in_core) >= RULES["P4_min_core"]
                              and len(in_core & act_set) >= RULES["P4_min_active_in_core"])}

    ok = [res[p]["pass"] for p in ("P1", "P2", "P3", "P4")]
    res["verdict"] = "GO_FULL" if all(ok) else "GO_GLOBAL" if (ok[0] and ok[1] and ok[3]) else "NO_GO"
    res["seconds"] = round(time.time() - t0, 1)
    (out_dir / "probe.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    md = [f"# DDO feasibility probe: **{res['verdict']}**", "",
          "| criterion | pass | evidence |", "|---|---|---|",
          f"| P1 data | {res['P1']['pass']} | {res['P1']['active_voters']} active voters, "
          f"{100 * res['P1']['coverage']:.1f}% of usable votes |",
          f"| P2 closer debater wins the convincing point | {res['P2']['pass']} | rate {rate:.3f}, "
          f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}] |",
          f"| P3 personal rates beat the global rate | {res['P3']['pass']} | NLL diff "
          f"{res['P3'].get('nll_diff_personal_minus_global', float('nan')):.4f}, CI {res['P3'].get('ci95')} |",
          f"| P4 graph | {res['P4']['pass']} | {RULES['P4_k']}-core {len(in_core)} users, "
          f"{len(in_core & act_set)} active voters |", "",
          f"Usable votes: {len(W)} of {info['n_votes_parsed']} parsed ({info['n_votes_unparsed']} unparsed). "
          f"Users: {len(names)}, {res['users']['with_10plus_stated']} with >= 10 stated issues.", "",
          "Dose-response (gap in ideological distance -> share awarded to the closer debater):", "",
          dose.to_markdown(index=False) if hasattr(dose, "to_markdown") and _has_tabulate() else dose.to_string(index=False),
          ""]
    (out_dir / "PROBE.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return res


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="folder with the DDO users/debates JSON files")
    ap.add_argument("--users")
    ap.add_argument("--debates")
    ap.add_argument("--out", default="runs/ddo_probe")
    ap.add_argument("--issue-key", default=None, help="name of the big-issues field (auto-detected)")
    ap.add_argument("--criterion", default="convinc", help="substring of the vote criterion used as the judgement")
    args = ap.parse_args(argv)
    if not (args.data or (args.users and args.debates)):
        ap.error("pass --data or both --users and --debates")
    res = run(args)
    return 0 if res.get("verdict") not in (None, "SCHEMA") else 2


if __name__ == "__main__":
    sys.exit(main())
