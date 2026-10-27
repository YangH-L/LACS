"""E2 (heterogeneity sigma) and E4 (data-intrinsic scale, position mixing, query sampling)
with the LACS engine on the round-3 instances of LiveJournal and Orkut.

  PYTHONPATH=p0 python -m p0kit.engine_sweeps --config p0/configs/p0.yaml --dataset livejournal --experiment e2
  ... --experiment e4_scale | e4_mix | e4_queries

Every setting keeps the round-3 graph, community labels and standard normals Z (replayed
from the dataset's random stream and checked bit for bit against the stored files), and
changes one thing:
  e2         theta = s * sqrt(2m) * exp(sigma Z - sigma^2/2) with the round-3 scale s, for
             every sigma in the grid (sigma = sigma_0 = 0.4 is the paper's default instance;
             sigma = 0 makes every tolerance equal, so the LCC is the maximum (k,r)-core
             containing q and must match kr_tau / kr_global)
  e4_scale   theta = c * r_nbr * exp(sigma_0 Z - sigma_0^2/2), r_nbr = median distance between
             friends: a data-intrinsic scale instead of the calibration against Exact
  e4_mix     a share mu of the vertices is moved to the community of a random vertex
             (nested across mu), same tolerances: weakens the alignment of positions and graph
  e4_queries the default instance with queries drawn uniformly from the k-core instead of
             only from communities with >= 20 members
The default queries are the 200 of round 3's default setting (four batches of 50).

Per setting (runs_p0/<dataset>/<experiment>/<setting>/): per_query.csv (one row per query
and algorithm: size, status, time_ms, tau, diam, min_theta, (iii) as reported by the engine,
plus the pairwise condition and the share of members whose own tolerance does not cover the
group, computed from the members), groups.jsonl.gz (members, input of E1 at this setting),
manifest.json, DONE. Settings with DONE are skipped on a rerun.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from . import bridge
from .formats import JsonlWriter
from .graphs import group_stats
from .runner_common import host_info, load_yaml, write_json

# algorithm name in the outputs -> (engine algo, members needed, time limit key)
ALGOS = {
    "exact": ("exact", True, "time_limit"),
    "greedy": ("greedy", True, "time_limit_baselines"),
    "ballcore_f": ("ballcore_f", True, "time_limit_baselines"),
    "ballcore_r": ("ballcore_r", True, "time_limit_baselines"),
    "kr_greedy": ("kr_greedy", True, "time_limit_baselines"),     # r = median tolerance (Table 1 reference)
    "kr_global": ("kr_exact", True, "time_limit_baselines"),      # r = median tolerance, exact
    "pairwise": ("pairwise_exact", True, "time_limit_baselines"),
    "kr_tau": ("kr_exact", True, "time_limit_baselines"),         # r = the LCC's own tau, per query
    "sac": ("sac_inc", True, "time_limit_baselines"),
    "edgeacc": ("edgeacc", False, "time_limit_baselines"),
}


# names of the stored groups (E1 input), aligned with the meetup application
GROUP_NAMES = {"exact": "lcc_exact", "greedy": "lcc_greedy", "ballcore_f": "lcc_ballcore_f",
               "ballcore_r": "lcc_ballcore_r", "kr_greedy": "kr_global_greedy", "kr_global": "kr_global",
               "pairwise": "pairwise", "kr_tau": "kr_tau", "sac": "sac", "edgeacc": "edgeacc"}


def _settings(cfg: Dict, experiment: str) -> List[Dict]:
    exp = cfg["experiments"][experiment]
    if experiment == "e4_queries":
        return [{"name": "kcore_uniform", "query_mode": "kcore", "n": int(exp.get("n", 200)),
                 "seed": int(exp.get("seed", 7))}]
    return list(exp["settings"])


def _instance(ds: Dict, rep: Dict, experiment: str, st: Dict, r_nbr: float):
    """(X, pos_bin or None, theta, description) for one setting."""
    X, pos_bin, theta = ds["X0"], ds["pos_bin"], ds["theta0"]
    desc = {}
    if experiment == "e2":
        sig = float(st["sigma"])
        if abs(sig - ds["sigma0"]) < 1e-12:
            theta = ds["theta0"]
        else:
            theta = bridge.paper_theta(rep["Z"], ds["dim"], ds["s_def"], sig)
        desc = {"sigma": sig, "s": ds["s_def"]}
    elif experiment == "e4_scale":
        c = float(st["c"])
        theta = bridge.intrinsic_theta(rep["Z"], r_nbr, c, ds["sigma0"])
        desc = {"c": c, "r_nbr": r_nbr, "s_equiv": c * r_nbr / np.sqrt(2.0 * ds["dim"]), "sigma": ds["sigma0"]}
    elif experiment == "e4_mix":
        mu = float(st["mu"])
        if mu > 0:
            X = bridge.mixed_positions(ds, rep, mu)
            pos_bin = None
        desc = {"mu": mu, "sigma": ds["sigma0"], "s": ds["s_def"]}
    elif experiment == "e4_queries":
        desc = {"query_mode": st["query_mode"], "sigma": ds["sigma0"], "s": ds["s_def"]}
    else:
        raise ValueError(experiment)
    return X, pos_bin, np.asarray(theta, dtype=np.float32), desc


def _member_stats(X, theta, members) -> Dict:
    H = np.asarray(members, dtype=np.int64)
    if len(H) < 2:
        return {}
    st = group_stats(np.asarray(X[H], dtype=np.float64), np.asarray(theta[H], dtype=np.float64))
    return {"sat_pairwise": st["sat_pairwise"], "member_viol_frac": st["member_viol_frac"],
            "sat_iii_py": st["sat_iii"]}


def run_setting(cfg: Dict, name: str, experiment: str, st: Dict, ds: Dict, rep: Dict, r_nbr: float,
                qinfo: Dict, n_proc: int, force: bool = False) -> str:
    out = Path(cfg.get("out_root", "runs_p0")) / name / experiment / st["name"]
    if (out / "DONE").exists() and not force:
        print(f"[skip] {out}")
        return str(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    X, pos_bin, theta, desc = _instance(ds, rep, experiment, st, r_nbr)
    cs = bridge.engine(ds, X, theta, out / "work", pos_bin=pos_bin)
    core = cs.core_numbers()
    k = int(cfg.get("k", ds["k0"]))
    if experiment == "e4_queries":
        pool = bridge.query_pool(core, k, mode=st["query_mode"])
        rng = np.random.default_rng(int(st["seed"]))
        queries = np.sort(rng.choice(pool, size=min(int(st["n"]), len(pool)), replace=False))
        batch_of = {}
    else:
        queries = qinfo["all"]
        batch_of = qinfo["batch_of"]
    if cfg.get("n_queries"):
        queries = queries[: int(cfg["n_queries"])]
    algos = list(cfg.get("algos_sweep", ["exact", "greedy", "ballcore_f", "ballcore_r", "kr_greedy", "pairwise",
                                         "kr_tau", "edgeacc"]))
    tl = {"time_limit": float(cfg.get("time_limit", ds["time_limit"])),
          "time_limit_baselines": float(cfg.get("time_limit_baselines", 10.0))}
    r_med = float(np.median(theta))
    found = {}
    for a in algos:
        if a == "kr_tau":
            continue
        eng, mem, tkey = ALGOS[a]
        r = r_med if a in ("kr_greedy", "kr_global") else None
        for row in bridge.engine_run(cs, queries, eng, k, tl[tkey], n_proc, mem, f"{experiment}_{a}", r=r):
            found[(int(row["q"]), a)] = row
    if "kr_tau" in algos and "exact" in algos:
        tq = [(int(q), float(found[(int(q), "exact")]["tau"])) for q in queries
              if found.get((int(q), "exact"), {}).get("size", 0) >= 2 and found[(int(q), "exact")].get("tau")]
        for row in bridge.engine_run(cs, [q for q, _ in tq], "kr_exact", k, tl["time_limit_baselines"], n_proc, True,
                                     f"{experiment}_krtau", r_per_query=[t for _, t in tq]):
            found[(int(row["q"]), "kr_tau")] = row
    members_max = int(cfg.get("members_max", 5000))
    rows = []
    gw = JsonlWriter(str(out / "groups.jsonl.gz"), append=False)
    for (q, a), row in sorted(found.items()):
        mem = row.pop("members", None)
        rec = {"dataset": name, "experiment": experiment, "setting": st["name"], "algo": a, **desc,
               "batch": batch_of.get(q), "theta_median": r_med, **row}
        rec["algo"] = a
        if mem is not None and 2 <= len(mem) <= members_max:
            rec.update(_member_stats(X, theta, mem))
        rows.append(rec)
        if mem is not None and len(mem) >= 2:
            gw.write_many([{"dataset": name, "setting": st["name"], "query": q, "k": k, "method": GROUP_NAMES[a],
                            "members": mem if len(mem) <= members_max else None, "size": len(mem),
                            "status": row.get("status")}])
    gw.close()
    import pandas as pd

    pd.DataFrame(rows).to_csv(out / "per_query.csv", index=False)
    inst = bridge.save_instance(out, ds, X=X, theta=theta, tag=f"{experiment}_{st['name']}", meta=desc,
                                inst_dir=Path(cfg.get("out_root", "runs_p0")) / name / "inst")
    write_json(str(out / "manifest.json"), {
        "dataset": name, "experiment": experiment, "setting": st, "desc": desc, "queries": int(len(queries)),
        "algos": algos, "time_limits": tl, "n_proc": n_proc, "replay_exact": bool(rep.get("exact")),
        "instance": inst, "seconds": time.time() - t0, "host": host_info()})
    open(out / "DONE", "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    print(f"[done] {out}: {len(queries)} queries x {len(algos)} algos in {time.time() - t0:.0f}s")
    return str(out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m p0kit.engine_sweeps")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--experiment", required=True, choices=["e2", "e4_scale", "e4_mix", "e4_queries"])
    ap.add_argument("--only", default=None, help="comma-separated setting names")
    ap.add_argument("--n-proc", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    from lacs.utils import setup_logging

    setup_logging("WARNING")
    cfg = load_yaml(args.config)
    e3_run = cfg.get("e3_run", "runs/efficiency_round3")
    ds = bridge.load_round3(e3_run, args.dataset)
    rep = bridge.replay_synthetic(ds)
    print(f"[{args.dataset}] replay of the round-3 random stream: exact={rep['exact']} "
          f"(positions {rep.get('check_positions')}, tolerances {rep.get('check_theta')})")
    r_nbr = bridge.median_edge_distance(ds["indptr"], ds["indices"], ds["X0"], seed=int(cfg.get("seed", 0)))
    qinfo = bridge.default_queries(e3_run, args.dataset, k=ds["k0"], dim=ds["dim"])
    n_proc = int(args.n_proc or cfg.get("n_proc", 1))
    only = set(args.only.split(",")) if args.only else None
    for st in _settings(cfg, args.experiment):
        if only and st["name"] not in only:
            continue
        run_setting(cfg, args.dataset, args.experiment, st, ds, rep, r_nbr, qinfo, n_proc, args.force)


if __name__ == "__main__":
    main()
