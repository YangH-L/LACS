#!/usr/bin/env python3
"""E6 Q2, static tolerance violations (plan section 6), from a P0-format export of the groups.

  python -m e6.static --groups runs/e6/e1_input --results runs/e6/results
  (equivalently: --instance runs/e6/e1_input/instance.json --groups runs/e6/e1_input/groups.jsonl.gz --out DIR)

--groups: a directory with instance.json (or instance.npz) and groups.jsonl.gz as written by
`p0kit.export_groups geo` (formats of p0kit/formats.py: positions X and tolerances theta of the
instance; one record per (query, method) with members, size, status, k, ...). Distances are
Euclidean on X, in double precision from the stored values, like the engine.

Per (dataset, setting, k, method), over groups with size >= 2 whose members are stored:
  * groups: number of such groups (records without members are counted separately);
  * share_iii: share of groups satisfying (iii), diameter <= min theta (relative tolerance --rtol);
  * share_members_ecc_gt_theta: share of members (pooled over all groups) whose eccentricity in the
    group exceeds their own theta;
  * share_pairs_rejected: share of member pairs (pooled) with d > min(theta_u, theta_v), i.e. at
    least one of the two rejects the other;
  * mean_group_share_members / mean_group_share_pairs: the same two shares averaged over groups
    (secondary);
reported over all queries (scope 'all') and over the queries where lcc_exact is non-empty
(scope 'lcc_nonempty'). The same relative tolerance is used for all three comparisons.
Note: the instance stores X and theta in float32 (as the engine uses them), so exact decimal ties
d == theta are decided by float32 rounding here exactly as in the engine; 1e-9 is below float32
resolution and does not change that.
Outputs: RESULTS/e6_static.json, RESULTS/e6_static.csv.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # also works under python -I
    sys.path.insert(0, str(ROOT))

from e6.common import now_iso, setup_logging, write_json  # noqa: E402

LOG = logging.getLogger("e6.static")
RTOL = 1e-9
METHOD_ORDER = ["lcc_exact", "lcc_greedy", "lcc_ballcore_f", "lcc_ballcore_r", "pairwise", "kcore", "kr_global",
                "kr_tau", "sac", "sac_sized", "edgeacc", "random_bfs"]


def load_instance(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """(X, theta, meta) of a P0 instance: instance.json pointing to .npy files, or an .npz."""
    path = Path(path)
    if path.is_dir():
        path = path / "instance.json" if (path / "instance.json").exists() else path / "instance.npz"
    if path.suffix == ".json":
        spec = json.loads(path.read_text())

        def ld(key):
            p = Path(spec[key])
            return np.load(p if p.is_absolute() else path.parent / p, allow_pickle=False)

        X, theta, meta = ld("X"), ld("theta"), spec.get("meta", {})
    else:
        z = np.load(path, allow_pickle=False)
        X, theta = z["X"], z["theta"]
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    if X.shape[0] != theta.shape[0]:
        raise ValueError(f"{path}: X and theta sizes differ")
    return np.asarray(X, np.float64), np.asarray(theta, np.float64), meta


def read_groups(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    out = []
    with opener(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for key in ("query", "method", "members"):
                if key not in r:
                    raise ValueError(f"{path}: record {i} has no {key!r}")
            out.append(r)
    return out


def group_stats(H: np.ndarray, X: np.ndarray, theta: np.ndarray, rtol: float = RTOL) -> dict:
    P = X[H]
    D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    th = theta[H]
    n = len(H)
    iu = np.triu_indices(n, 1)
    pair_lim = np.minimum(th[:, None], th[None, :])[iu]
    ecc = D.max(axis=1)
    diam = float(ecc.max())
    mt = float(th.min())
    return {"size": n, "diam": diam, "min_theta": mt, "iii": bool(diam <= mt * (1 + rtol)),
            "members_viol": int((ecc > th * (1 + rtol)).sum()), "pairs": int(len(pair_lim)),
            "pairs_viol": int((D[iu] > pair_lim * (1 + rtol)).sum())}


def summarise(rows: pd.DataFrame) -> dict:
    ok = rows[rows["has_members"]]
    out = {"groups": int(len(ok)), "groups_without_members": int((~rows["has_members"]).sum()),
           "queries": int(rows["query"].nunique())}
    if len(ok) == 0:
        return out
    out.update({
        "share_iii": float(ok["iii"].mean()),
        "members": int(ok["size"].sum()),
        "share_members_ecc_gt_theta": float(ok["members_viol"].sum() / ok["size"].sum()),
        "pairs": int(ok["pairs"].sum()),
        "share_pairs_rejected": float(ok["pairs_viol"].sum() / ok["pairs"].sum()),
        "mean_group_share_members": float((ok["members_viol"] / ok["size"]).mean()),
        "mean_group_share_pairs": float((ok["pairs_viol"] / ok["pairs"]).mean()),
        "size_median": float(ok["size"].median())})
    return out


def run(args) -> int:
    g = Path(args.groups)
    gdir = g if g.is_dir() else g.parent  # --groups: the export directory or its groups.jsonl(.gz)
    res = Path(args.results)
    res.mkdir(parents=True, exist_ok=True)
    setup_logging(res / "e6_static.log")
    inst = Path(args.instance) if args.instance else gdir
    X, theta, meta = load_instance(inst)
    if args.groups_file:
        gpath = Path(args.groups_file)
    elif not g.is_dir():
        gpath = g
    else:
        gpath = gdir / "groups.jsonl.gz"
        if not gpath.exists() and (gdir / "groups.jsonl").exists():
            gpath = gdir / "groups.jsonl"
    recs = read_groups(gpath)
    LOG.info("Q2: %d records from %s, instance with %d vertices", len(recs), gpath, len(theta))
    cache: dict[bytes, dict] = {}
    rows = []
    for r in recs:
        size = int(r.get("size") if r.get("size") is not None else len(r["members"] or []))
        base = {"dataset": r.get("dataset"), "setting": r.get("setting", "default"), "k": r.get("k"),
                "query": int(r["query"]), "method": r["method"], "status": r.get("status"), "size": size}
        if size < 2:
            rows.append({**base, "eligible": False, "has_members": False})
            continue
        if r["members"] is None:
            rows.append({**base, "eligible": True, "has_members": False})
            continue
        H = np.asarray(sorted(set(int(x) for x in r["members"])), dtype=np.int64)
        key = hashlib.blake2b(H.tobytes(), digest_size=16).digest()
        if key not in cache:
            cache[key] = group_stats(H, X, theta, args.rtol)
        rows.append({**base, "eligible": True, "has_members": True, **cache[key]})
    df = pd.DataFrame(rows)
    keys = ["dataset", "setting", "k"]
    for c in keys:
        df[c] = df[c].astype(object).where(df[c].notna(), "")
    lcc = df[(df["method"] == "lcc_exact") & (df["size"] >= 2)]
    lcc_q = {tuple(k_): set(g["query"]) for k_, g in lcc.groupby(keys)} if len(lcc) else {}
    methods = [m for m in METHOD_ORDER if m in set(df["method"])] + sorted(set(df["method"]) - set(METHOD_ORDER))
    table, out = [], {"tool": "e6.static", "created": now_iso(), "groups_file": str(gpath), "instance": str(inst),
                      "instance_meta": meta, "rtol": args.rtol, "records": int(len(df)), "results": []}
    for kv, g in df.groupby(keys, sort=True):
        qs_lcc = lcc_q.get(tuple(kv), set())
        block = {"dataset": kv[0], "setting": kv[1], "k": kv[2], "queries": int(g["query"].nunique()),
                 "queries_lcc_nonempty": len(qs_lcc), "methods": {}}
        for m in methods:
            gm = g[(g["method"] == m) & g["eligible"]]
            if not len(g[g["method"] == m]):
                continue
            block["methods"][m] = {}
            for scope, sub in (("all", gm), ("lcc_nonempty", gm[gm["query"].isin(qs_lcc)])):
                s = summarise(sub)
                block["methods"][m][scope] = s
                table.append({"dataset": kv[0], "setting": kv[1], "k": kv[2], "method": m, "scope": scope, **s})
        out["results"].append(block)
    if any(str(b["k"]) not in ("3", "") for b in out["results"]):
        LOG.warning("export has k other than 3: %s (plan section 6 uses k = 3)", [b["k"] for b in out["results"]])
    cols = ["dataset", "setting", "k", "method", "scope", "groups", "groups_without_members", "queries", "share_iii",
            "members", "share_members_ecc_gt_theta", "pairs", "share_pairs_rejected", "mean_group_share_members",
            "mean_group_share_pairs", "size_median"]
    pd.DataFrame(table).reindex(columns=cols).to_csv(res / "e6_static.csv", index=False)
    write_json(out, res / "e6_static.json")
    for t in table:
        if t["scope"] == "all":
            LOG.info("Q2 %-15s groups %4d  (iii) %s  members ecc>theta %s  pairs rejected %s", t["method"],
                     t["groups"], _fmt(t.get("share_iii")), _fmt(t.get("share_members_ecc_gt_theta")),
                     _fmt(t.get("share_pairs_rejected")))
    return 0


def _fmt(x):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m e6.static", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groups", required=True,
                    help="P0 export directory (instance.json + groups.jsonl.gz), or the groups.jsonl(.gz) file itself")
    ap.add_argument("--instance", default=None, help="instance.json / .npz (default: next to the groups)")
    ap.add_argument("--groups-file", default=None, help="groups.jsonl(.gz) (default: in --groups)")
    ap.add_argument("--results", "--out", dest="results", required=True)
    ap.add_argument("--rtol", type=float, default=RTOL)
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
