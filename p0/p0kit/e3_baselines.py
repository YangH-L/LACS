"""E3: LatSweep-Exact against external exact baselines.

  PYTHONPATH=p0 python -m p0kit.e3_baselines --config p0/configs/p0.yaml --dataset livejournal --workers 1

Instances and queries are those of round 3's default setting (LiveJournal / Orkut:
runs/efficiency_round3/work/e3_<name>/; Gowalla / Brightkite: the meetup runs with the
G2-chosen tolerance). Every harness solver runs inside the same sweep (p0kit/sweep.py,
Algorithm 1 of the paper): the same query region, thresholds, peeling, incumbent
(BallCore-F and Greedy), candidate cap and per-query time limit; only the per-threshold
solver differs:
  bnb           the paper's branch and bound (Lemma 5), reference copy in C++
  krmax         maximum (k,r)-core of [35] (Zhang et al., AdvMax), reimplemented in C++
  krmax_simple  the same with a simple branching order (sensitivity of [35] to its order)
  ilp           MILP with single-commodity flow connectivity, HiGHS via scipy, one thread
  cpsat         the same model in OR-Tools CP-SAT, one worker (pip install ortools)
Two engine rows are added per query: engine_r3 (round 3's LatSweep-Exact, read from
results/e3_per_query.csv) and, with --engine-rerun, engine_now (the same engine rerun
now on this machine, one process) -- the production implementation the paper reports.

Outputs in runs_p0/<dataset>/e3/: e3_per_query.csv, e3_crosscheck.csv, manifest.json.
Proven answers of all solvers must have the same size; any disagreement is printed.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict

import numpy as np

from . import bridge, sweep
from .formats import JsonlWriter, recover_jsonl, rewrite_jsonl
from .runner_common import host_info, load_yaml, ortools_version, run_tasks_watchdog, write_json


def _solve_one(shared: Dict, q: int, solver: str):
    """One query, one solver: the whole LatSweep with that per-threshold solver (runs in a worker)."""
    index = shared["index"]
    kw = {"workers": 1} if solver == "cpsat" else {}
    base = {"dataset": shared["dataset"], "query": int(q), "solver": solver, "batch": shared["batch_of"].get(int(q))}
    try:
        res = sweep.exact(index, int(q), shared["k"], solver=solver, time_limit=shared["time_limit"],
                          candidate_cap=shared["cap"], solver_kwargs=kw)
    except Exception as exc:  # a failing solver must not stop the others; the row records it
        return [{**base, "time_ms": float("nan"), "cpu_ms": float("nan"), "solver_ms": float("nan"), "size": 0,
                 "proven": False, "status": "error", "message": repr(exc)[:300]}]
    res.pop("members")
    return [{**base, "time_ms": res.pop("seconds") * 1e3, "cpu_ms": res.pop("cpu_seconds") * 1e3,
             "solver_ms": res.pop("solver_seconds") * 1e3, **res}]


def _work(shared: Dict, item):
    q, solvers = item
    rows = []
    for solver in solvers:
        rows += _solve_one(shared, q, solver)
    return rows


def load_instance(cfg: Dict, name: str) -> Dict:
    geo_runs = cfg.get("geo_runs", {})
    if name in geo_runs:
        return bridge.load_geo(geo_runs[name])
    return bridge.load_round3(cfg.get("e3_run", "runs/efficiency_round3"), name)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m p0kit.e3_baselines")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--workers", type=int, default=1, help="parallel queries (timings: keep 1 or few)")
    ap.add_argument("--solvers", default=None, help="comma separated (default: config e3.solvers)")
    ap.add_argument("--n-queries", type=int, default=0)
    ap.add_argument("--engine-rerun", action="store_true", help="also rerun the engine's exact now (1 process)")
    ap.add_argument("--force", action="store_true", help="ignore the checkpoint of an earlier run")
    args = ap.parse_args(argv)
    from lacs.utils import setup_logging

    setup_logging("WARNING")
    cfg = load_yaml(args.config)
    e3 = cfg.get("e3", {})
    solvers = args.solvers.split(",") if args.solvers else list(e3.get("solvers", ["bnb", "krmax", "ilp", "cpsat"]))
    if "cpsat" in solvers and ortools_version() is None:
        # (checked without importing: the parent must not initialise OR-Tools before forking workers)
        print("[e3] ortools is not installed: cpsat dropped (pip install ortools to include it)")
        solvers = [s for s in solvers if s != "cpsat"]
    e3_run = cfg.get("e3_run", "runs/efficiency_round3")
    ds = load_instance(cfg, args.dataset)
    k = int(cfg.get("k", 3))
    time_limit = float(e3.get("time_limit", 60))
    cap = int(e3.get("cap", 3000))
    qinfo = bridge.default_queries(e3_run, args.dataset, k=k, dim=ds["dim"] if ds["kind"] == "round3" else 16)
    queries = qinfo["all"]
    if args.n_queries:
        queries = queries[: args.n_queries]
    out = Path(cfg.get("out_root", "runs_p0")) / args.dataset / "e3"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # checkpoint: one record per (query, solver), appended as queries finish; a rerun only
    # computes the missing pairs (and repairs the file if the earlier run was killed)
    ckpt = str(out / "e3_rows.jsonl.gz")
    have, old = {}, []
    if os.path.exists(ckpt) and not args.force:
        old, info = recover_jsonl(ckpt)
        if info["truncated"] or info["bad_lines"]:
            rewrite_jsonl(ckpt, old)
            print(f"[e3] repaired {ckpt} after an interrupted run ({len(old)} rows kept)")
        for r in old:
            have.setdefault(int(r["query"]), set()).add(r["solver"])
    elif os.path.exists(ckpt):
        os.remove(ckpt)
    todo = []
    for q in queries.tolist():
        miss = [s_ for s_ in solvers if s_ not in have.get(int(q), set())]
        if miss:
            todo.append((int(q), miss))
    print(f"[e3] {args.dataset}: {len(queries) - len(todo)} of {len(queries)} queries already done")
    index_info = {}
    if todo:
        index = sweep.SweepIndex(np.asarray(ds["indptr"]), np.asarray(ds["indices"]),
                                 np.asarray(ds["X0"], dtype=np.float64), np.asarray(ds["theta0"], dtype=np.float64))
        index_info = {"index_seconds": index.build_seconds, "kept_edge_frac": index.kept_edge_frac}
        print(f"[e3] {args.dataset}: index built in {index.build_seconds:.1f}s "
              f"(G_theta keeps {index.kept_edge_frac:.1%})")
        shared = {"index": index, "k": k, "time_limit": time_limit, "cap": cap,
                  "dataset": args.dataset, "batch_of": qinfo["batch_of"]}
        slack = float(e3.get("hang_slack", 180))

        def on_hang(q, solver, secs, detail):
            return [{"dataset": args.dataset, "query": int(q), "solver": solver,
                     "batch": qinfo["batch_of"].get(int(q)), "time_ms": float("nan"), "cpu_ms": float("nan"),
                     "solver_ms": float("nan"), "size": 0, "proven": False, "status": "error",
                     "message": f"killed after {secs:.0f}s ({detail})"[:2000]}]

        w = JsonlWriter(ckpt, append=True)
        wd = run_tasks_watchdog(shared, _solve_one, todo, args.workers, limit_of=lambda s_: time_limit + slack,
                                on_rows=w.write_many, on_hang=on_hang, label=f"{args.dataset}/e3", log_every=50,
                                hang_dir=str(out / "work" / "hang"))
        w.close()
        index_info["watchdog"] = wd
        if wd["hung"] or wd["died"]:
            print(f"[e3] WARNING: {wd['hung']} solver runs hung and {wd['died']} workers died; "
                  f"see the status/message columns of e3_per_query.csv")
    qset = set(int(q) for q in queries.tolist())
    rows, info = recover_jsonl(ckpt) if os.path.exists(ckpt) else ([], {})
    rows = [r for r in rows if int(r["query"]) in qset and r["solver"] in solvers]
    import pandas as pd

    # engine rows from round 3 (default instance batches only: the dim batch used a redrawn instance)
    per = pd.read_csv(Path(e3_run) / "results" / "e3_per_query.csv")
    d = per[(per["dataset"] == args.dataset) & (per["algo"] == "exact")]
    keep = (((d["sweep"] == "k") & (d["value"].astype(float) == k)) |
            ((d["sweep"] == "fraction") & (d["value"].astype(float) == 1.0)) |
            ((d["sweep"] == "theta_scale") & (d["value"].astype(float) == 1.0)))
    for r in d[keep].to_dict(orient="records"):
        if int(r["q"]) in set(queries.tolist()):
            rows.append({"dataset": args.dataset, "query": int(r["q"]), "solver": "engine_r3",
                         "batch": qinfo["batch_of"].get(int(r["q"])), "time_ms": r["time_ms"], "size": r["size"],
                         "status": r["status"], "proven": r["status"] == "optimal",
                         "region_size": r.get("s0_size"), "n_thresholds": r.get("n_tau"),
                         "n_thresholds_visited": r.get("n_tau_eval")})
    if args.engine_rerun:
        now_path = out / "e3_engine_now.csv"
        now = pd.read_csv(now_path) if now_path.exists() and not args.force else None
        if now is not None and qset <= set(int(q) for q in now["query"]):
            print(f"[e3] {args.dataset}: engine rerun taken from {now_path}")
        else:
            cs = bridge.engine(ds, ds["X0"], ds["theta0"], out / "work",
                               pos_bin=ds.get("pos_bin"))
            cs_rows = bridge.engine_run(cs, queries, "exact", k, time_limit, 1, False, "e3now")
            now = pd.DataFrame([{"dataset": args.dataset, "query": int(r["q"]), "solver": "engine_now",
                                 "batch": qinfo["batch_of"].get(int(r["q"])), "time_ms": r["time_ms"],
                                 "size": r["size"], "status": r["status"], "proven": r["status"] == "optimal",
                                 "region_size": r.get("s0_size"), "n_thresholds": r.get("n_tau"),
                                 "n_thresholds_visited": r.get("n_tau_eval")} for r in cs_rows])
            now.to_csv(now_path, index=False)
        rows += now[now["query"].isin(qset)].to_dict(orient="records")
    df = pd.DataFrame(rows).sort_values(["query", "solver"])
    df.to_csv(out / "e3_per_query.csv", index=False)
    bad = []
    for q, g in df.groupby("query"):
        pv = g[g["proven"].astype(bool)]
        if pv["size"].nunique() > 1:
            bad.append({"query": int(q), **{f"size_{s}": int(z) for s, z in zip(pv["solver"], pv["size"])}})
    cols = ["query"] + sorted({c for b in bad for c in b if c != "query"})
    pd.DataFrame(bad, columns=cols).to_csv(out / "e3_crosscheck.csv", index=False)
    if bad:
        print(f"[e3] WARNING: {len(bad)} queries where proven answers disagree, see {out / 'e3_crosscheck.csv'}")
    write_json(str(out / "manifest.json"), {
        "dataset": args.dataset, "kind": ds["kind"], "solvers": solvers, "queries": int(len(queries)),
        "time_limit": time_limit, "cap": cap, "workers": args.workers, "engine_rerun": bool(args.engine_rerun),
        "queries_from_checkpoint": int(len(queries) - len(todo)), **index_info,
        "crosscheck_mismatches": len(bad), "seconds": time.time() - t0, "host": host_info()})
    print(f"[done] {out}: {len(queries)} queries x {len(solvers)} solvers in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
