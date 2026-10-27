"""E1 inputs: the groups every method returns, with their members.

  # meetups: exactly the groups of round 4's geo-e2 (same queries, same engine calls,
  # same random groups), captured while lacs.apps.geo._e2_eval computes its metrics
  PYTHONPATH=p0 python -m p0kit.export_groups geo --config configs/gowalla_r4.yaml --out runs_p0/gowalla/e1_input

  # LiveJournal / Orkut: the round-3 default instance and its default queries, with the
  # same method set as the meetup application
  PYTHONPATH=p0 python -m p0kit.export_groups synth --e3-run runs/efficiency_round3 --dataset livejournal \
      --out runs_p0/livejournal/e1_input --n-proc 8

Outputs in --out: groups.jsonl.gz (one record per query and method; members are omitted,
size kept, for groups larger than --members-max), instance.json (+ ../inst/*.npy),
per_query.csv (the same rows without members) and manifest.json. Nothing under runs/ is
modified: the engine works in runs/<name>/work/p0_* (meetups) or under --out (synth).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from . import bridge
from .formats import JsonlWriter
from .runner_common import host_info, write_json

GEO_METRICS = ("covisit_pair_rate", "member_meet_rate", "friend_pair_covisit_rate", "diam_km", "min_theta_km",
               "active_members")


def _clean(v):
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


# ------------------------------------------------------------------ geo ----
def export_geo(config: str, out: str, k: int = None, members_max: int = 5000, mode: str = None) -> dict:
    from lacs.apps import geo
    from lacs.config import load_config, run_paths

    t0 = time.time()
    cfg = load_config(config)
    paths = run_paths(cfg)
    g = geo.gcfg(cfg)
    k = int(k or cfg["search"]["k"])
    mode = mode or g.get("queries", "sample")
    ctx = geo._e2_context(cfg, paths, "p0_e1_export")
    captured = []
    orig = geo._member_metrics

    def capture(ctx_, H, memo=True):
        captured.append(np.asarray(H, dtype=np.int64).copy())
        return orig(ctx_, H, memo)

    geo._member_metrics = capture
    try:
        df, summ, paired, info, extras = geo._e2_eval(cfg, ctx, k, mode, int(cfg["search"]["n_queries"]), "p0e1",
                                                      member_analysis=False)
    finally:
        geo._member_metrics = orig
    if len(captured) != len(df):
        raise RuntimeError(f"captured {len(captured)} groups for {len(df)} rows; geo._e2_eval changed?")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(cfg["out_dir"]).name
    w = JsonlWriter(str(out / "groups.jsonl.gz"), append=False)
    n_big = 0
    for rec, H in zip(df.to_dict(orient="records"), captured):
        if int(rec["size"]) != len(H):
            raise RuntimeError("row/group mismatch")
        big = len(H) > members_max
        n_big += int(big)
        g_rec = {"dataset": name, "setting": "default", "query": int(rec["q"]), "k": k, "method": rec["method"],
                 "members": None if big else H.tolist(), "size": int(len(H)), "status": rec.get("status"),
                 "radius_km": _clean(rec.get("radius_km"))}
        for m in GEO_METRICS:
            if m in rec:
                g_rec[m] = _clean(rec[m])
        w.write_many([g_rec])
    w.close()
    df.to_csv(out / "per_query.csv", index=False)
    ds = bridge.load_geo(cfg["out_dir"], tolerance=ctx["tol"])
    inst = bridge.save_instance(out, ds, tag="default", meta={"tolerance": ctx["tol"], "units": "km"})
    man = {"dataset": name, "config": os.path.abspath(config), "k": k, "query_mode": mode, "info": info,
           "rows": int(len(df)), "groups_without_members": n_big, "members_max": members_max,
           "instance": inst, "seconds": time.time() - t0, "host": host_info()}
    write_json(str(out / "manifest.json"), man)
    print(f"[export geo] {name}: {len(df)} groups ({n_big} stored by size only) in {man['seconds']:.0f}s")
    return man


# ---------------------------------------------------------------- synth ----
SYNTH_METHODS = ["lcc_exact", "lcc_greedy", "lcc_ballcore_f", "lcc_ballcore_r", "pairwise", "kr_global", "kr_tau",
                 "sac", "sac_sized", "edgeacc", "kcore", "random_bfs"]
CPP = {"lcc_exact": "exact", "lcc_greedy": "greedy", "lcc_ballcore_f": "ballcore_f",
       "lcc_ballcore_r": "ballcore_r", "pairwise": "pairwise_exact", "kr_global": "kr_exact",
       "edgeacc": "edgeacc", "sac": "sac_inc"}


def random_connected(indptr, indices, ok, q: int, size: int, rng) -> np.ndarray:
    """Same process as lacs.apps.geo._random_connected, on CSR arrays."""
    chosen = [q]
    in_set = {q}
    frontier = [int(u) for u in indices[indptr[q]:indptr[q + 1]] if ok[u]]
    while len(chosen) < size and frontier:
        i = int(rng.integers(len(frontier)))
        u = frontier.pop(i)
        if u in in_set:
            continue
        in_set.add(u)
        chosen.append(u)
        frontier += [int(w) for w in indices[indptr[u]:indptr[u + 1]] if ok[w] and w not in in_set]
    return np.array(sorted(chosen), dtype=np.int64)


def run_methods(ds, cs, core, queries, methods, k, time_limit, time_limit_baselines, n_proc, tag, seed,
                members_max=5000):
    """Runs the meetup method set on any instance; returns {(q, method): row} with members."""
    found = {}
    lcc_algos = [CPP[m] for m in ("lcc_exact", "lcc_greedy", "lcc_ballcore_f", "lcc_ballcore_r") if m in methods]
    needs_lcc = {"kr_tau", "sac_sized", "random_bfs"} & set(methods)
    if needs_lcc and "exact" not in lcc_algos:
        lcc_algos.append("exact")
    for a in lcc_algos:
        tl = time_limit if a == "exact" else time_limit_baselines
        for r in bridge.engine_run(cs, queries, a, k, tl, n_proc, True, f"{tag}_{a}"):
            found[(r["q"], a)] = r
    theta = np.asarray(ds["theta_eval"])
    deg = np.diff(np.asarray(ds["indptr"]))
    r_global = float(np.median(theta[deg > 0]))
    if "pairwise" in methods:
        for r in bridge.engine_run(cs, queries, "pairwise_exact", k, time_limit_baselines, n_proc, True,
                                   f"{tag}_pairwise"):
            found[(r["q"], "pairwise_exact")] = r
    if "kr_global" in methods:
        for r in bridge.engine_run(cs, queries, "kr_exact", k, time_limit_baselines, n_proc, True, f"{tag}_krg",
                                   r=r_global):
            found[(r["q"], "kr_exact")] = r
    if "sac" in methods:
        for r in bridge.engine_run(cs, queries, "sac_inc", k, time_limit_baselines, n_proc, True, f"{tag}_sac"):
            found[(r["q"], "sac_inc")] = r
    if "edgeacc" in methods:
        for r in bridge.engine_run(cs, queries, "edgeacc", k, time_limit_baselines, n_proc, False, f"{tag}_edgeacc"):
            found[(r["q"], "edgeacc")] = r
    lcc = [(int(q), found[(q, "exact")]) for q in queries if found.get((q, "exact"), {}).get("size", 0) >= 2]
    if "kr_tau" in methods and lcc:
        tq = [(q, float(r["tau"])) for q, r in lcc if r.get("tau") is not None]
        for r in bridge.engine_run(cs, [q for q, _ in tq], "kr_exact", k, time_limit_baselines, n_proc, True,
                                   f"{tag}_krtau", r_per_query=[t for _, t in tq]):
            found[(r["q"], "kr_tau")] = r
    if "sac_sized" in methods and lcc:
        for r in bridge.engine_run(cs, [q for q, _ in lcc], "sac_inc", k, time_limit_baselines, n_proc, True,
                                   f"{tag}_sacsized", size_per_query=[int(r["size"]) for _, r in lcc]):
            found[(r["q"], "sac_sized")] = r
    comp, sizes = (bridge.kcore_components(ds["indptr"], ds["indices"], core, k) if "kcore" in methods
                   else (None, None))
    rng = np.random.default_rng(int(seed))
    pool_ok = np.asarray(core) >= 1
    rows = []
    for q in queries:
        q = int(q)
        for m in methods:
            if m == "random_bfs":
                ref = found.get((q, "exact"))
                size = int(ref["size"]) if ref else 0
                H = (random_connected(ds["indptr"], ds["indices"], pool_ok, q, size, rng) if size >= 2
                     else np.empty(0, np.int64))
                r = {"status": "ok", "size": int(len(H)), "time_ms": None}
            elif m == "kcore":
                c = comp[q]
                size = int(sizes[c]) if c >= 0 else 0
                H = np.flatnonzero(comp == c) if (c >= 0 and size <= members_max) else None
                r = {"status": "ok", "size": size, "time_ms": None}
            elif m in ("kr_tau", "sac_sized"):
                r = found.get((q, m))
                if r is None:
                    continue
                H = np.asarray(r.get("members", []), dtype=np.int64)
            else:
                r = found.get((q, CPP[m]))
                if r is None:
                    continue
                H = np.asarray(r["members"], dtype=np.int64) if "members" in r else None
            size = int(r["size"])
            if H is not None and len(H) > members_max:
                H = None
            rows.append({"query": q, "method": m, "members": None if H is None else H.tolist(), "size": size,
                         "status": r.get("status"), "time_ms": r.get("time_ms"), "tau": r.get("tau"),
                         "diam": r.get("diam"), "min_theta": r.get("min_theta"), "feasible": r.get("feasible"),
                         "s0_size": r.get("s0_size"), "n_candidates": r.get("n_candidates"),
                         "bb_nodes": r.get("bb_nodes")})
    return rows, r_global


def export_synth(e3_run: str, name: str, out: str, n_proc: int = 8, time_limit: float = None,
                 time_limit_baselines: float = 60.0, members_max: int = 5000, methods=None,
                 n_queries: int = 0) -> dict:
    t0 = time.time()
    ds = bridge.load_round3(e3_run, name)
    ds["theta_eval"] = ds["theta0"]
    qinfo = bridge.default_queries(e3_run, name, k=ds["k0"], dim=ds["dim"])
    queries = qinfo["all"]
    if n_queries:
        queries = queries[:n_queries]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cs = bridge.engine(ds, ds["X0"], ds["theta0"], out / "work", pos_bin=ds["pos_bin"])
    core = cs.core_numbers()
    rows, r_global = run_methods(ds, cs, core, queries, methods or SYNTH_METHODS, ds["k0"],
                                 time_limit or ds["time_limit"], time_limit_baselines, n_proc, "e1x",
                                 seed=ds["seed"], members_max=members_max)
    w = JsonlWriter(str(out / "groups.jsonl.gz"), append=False)
    for r in rows:
        w.write_many([{"dataset": name, "setting": "default", "k": ds["k0"], "batch": qinfo["batch_of"].get(r["query"]),
                       **r}])
    w.close()
    import pandas as pd

    pd.DataFrame([{k: v for k, v in r.items() if k != "members"} for r in rows]).to_csv(out / "per_query.csv",
                                                                                      index=False)
    inst = bridge.save_instance(out, ds, tag="default", meta={"s_def": ds["s_def"], "sigma": ds["sigma0"]})
    man = {"dataset": name, "e3_run": os.path.abspath(e3_run), "queries": int(len(queries)),
           "batches": {b: int(len(v)) for b, v in qinfo["batches"].items()}, "r_global": r_global,
           "time_limit": time_limit or ds["time_limit"], "time_limit_baselines": time_limit_baselines,
           "n_proc": n_proc, "rows": len(rows), "instance": inst, "seconds": time.time() - t0, "host": host_info()}
    write_json(str(out / "manifest.json"), man)
    print(f"[export synth] {name}: {len(queries)} queries, {len(rows)} groups in {man['seconds']:.0f}s")
    return man


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m p0kit.export_groups")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("geo")
    g.add_argument("--config", required=True, help="e.g. configs/gowalla_r4.yaml")
    g.add_argument("--out", required=True)
    g.add_argument("--k", type=int, default=None)
    g.add_argument("--members-max", type=int, default=5000)
    g.add_argument("--queries", default=None, help="'all' or 'sample' (default: the config's geo.queries)")
    g.add_argument("--force", action="store_true", help="redo an export that already finished")
    s = sub.add_parser("synth")
    s.add_argument("--e3-run", default="runs/efficiency_round3")
    s.add_argument("--dataset", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n-proc", type=int, default=8)
    s.add_argument("--time-limit", type=float, default=None, help="exact (default: round 3's)")
    s.add_argument("--time-limit-baselines", type=float, default=60.0)
    s.add_argument("--members-max", type=int, default=5000)
    s.add_argument("--n-queries", type=int, default=0, help="first N default queries only (smoke tests)")
    s.add_argument("--force", action="store_true", help="redo an export that already finished")
    args = ap.parse_args(argv)
    from lacs.utils import setup_logging

    if (Path(args.out) / "manifest.json").exists() and not args.force:
        # the manifest is written last, so its presence means the export is complete
        print(f"[skip] {args.out}: export already finished (--force to redo)")
        return
    setup_logging("INFO")
    if args.cmd == "geo":
        export_geo(args.config, args.out, args.k, args.members_max, args.queries)
    else:
        export_synth(args.e3_run, args.dataset, args.out, args.n_proc, args.time_limit, args.time_limit_baselines,
                     args.members_max, n_queries=args.n_queries)


if __name__ == "__main__":
    main()
