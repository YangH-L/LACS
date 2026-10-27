"""Self-test of the P0 kit on the server (run from the project root):

  PYTHONPATH=p0 python -m p0kit.selftest              # unit checks + integration (about 3-6 minutes)
  PYTHONPATH=p0 python -m p0kit.selftest --unit-only

Unit checks: the dynamics kernels against the paper's statements (Theorems 1-2,
Proposition 1(a)(b), single exemption, the characterisation's adversarial weights,
external pull); every per-threshold solver and the stand-alone sweep against brute force
(CP-SAT too when ortools is installed).

Integration (needs the lacs package, i.e. run from the project root): builds a small
round-3-style E3 run and a small meetup run with the project's own code in a temporary
directory, then runs every P0 driver on them with tiny settings and checks:
  * the round-3 random stream replays bit for bit (positions and tolerances);
  * the exported meetup groups line up one to one with geo-e2's per-query rows;
  * the sweep harness and the engine agree on every proven optimum;
  * LCC groups never break under the assumption-free dynamics variants;
  * the report builds.
Exit code 0 iff everything passed.
"""

from __future__ import annotations

import argparse
import gzip
import importlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KIT = HERE.parent


def _run_test_module(path: Path) -> list:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fails = []
    for name in sorted(n for n in dir(mod) if n.startswith("test_")):
        t0 = time.time()
        try:
            getattr(mod, name)()
            print(f"  ok   {path.stem}.{name} ({time.time() - t0:.1f}s)")
        except Exception:
            fails.append(f"{path.stem}.{name}")
            print(f"  FAIL {path.stem}.{name}")
            traceback.print_exc()
    return fails


def unit() -> list:
    print("== unit checks ==")
    fails = []
    for f in ("test_dynamics_kernels.py", "test_threshold_solvers.py", "test_sweep.py", "test_formats.py",
              "test_runner.py"):
        fails += _run_test_module(KIT / "tests" / f)
    return fails


def _check(cond, msg, fails):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        fails.append(msg)


def integration(keep: bool = False) -> list:
    print("== integration with lacs ==")
    fails = []
    try:
        import lacs  # noqa: F401
    except ImportError:
        print("  FAIL lacs is not importable: run from the project root")
        return ["lacs import"]
    from lacs.config import load_config, run_paths
    from lacs.experiments import e3 as lacs_e3
    from lacs.__main__ import dispatch
    from lacs.graph.io import edges_to_csr
    from lacs.utils import setup_logging

    from . import bridge, export_groups, report, engine_sweeps
    from .formats import read_jsonl
    from .runner_common import ortools_version

    setup_logging("WARNING")
    td = Path(tempfile.mkdtemp(prefix="p0_selftest_"))
    try:
        # ---- a small meetup run (synthetic LBSN of lacs.apps.geo)
        raw = td / "lbsn"
        geo_out = td / "geo_toy"
        gsets = [f"out_dir={geo_out}", f"geo.synthetic={{dir: {raw}, n_users: 450, n_days: 240, seed: 2}}",
                 f"geo.edges={raw}/edges.txt", f"geo.checkins={raw}/checkins.txt", "geo.g3_queries=40",
                 "search.k=3", "search.n_queries=40", "search.time_limit=10", "search.n_proc=1",
                 "geo.methods=[lcc_exact, lcc_greedy, lcc_ballcore_f, lcc_ballcore_r, pairwise, kcore, kr_global, "
                 "kr_tau, sac, sac_sized, edgeacc, random_bfs]"]
        gcfg = load_config(None, gsets)
        gpaths = run_paths(gcfg)
        for step in ("geo-prepare", "geo-gates", "geo-e2"):
            dispatch(step, gcfg, gpaths)
        r4 = td / "geo_toy_r4.yaml"
        import yaml

        g2 = dict(gcfg)
        g2 = {k: v for k, v in g2.items() if not k.startswith("_")}
        g2["geo"] = dict(g2["geo"], results_subdir="r4", synthetic=None)
        r4.write_text(yaml.safe_dump(g2))
        e1_geo = td / "p0" / "geo_toy" / "e1_input"
        export_groups.export_geo(str(r4), str(e1_geo), k=3)
        per = __import__("pandas").read_csv(gpaths["results"] / "geo_e2_per_query.csv")
        recs = list(read_jsonl(str(e1_geo / "groups.jsonl.gz")))
        same = len(recs) == len(per) and all(int(r["size"]) == int(s) for r, s in zip(recs, per["size"]))
        _check(same, f"exported meetup groups match geo-e2 row by row ({len(recs)} groups)", fails)
        lcc_ok = all(r["members"] is None or r["method"] != "lcc_exact" or len(r["members"]) < 2 or
                     r.get("diam_km") is None or r["diam_km"] <= r["min_theta_km"] * (1 + 1e-4) + 1e-3 for r in recs)
        _check(lcc_ok, "exported LCC groups satisfy (iii)", fails)

        # ---- a small round-3-style E3 run (planted partition graph, community attributes)
        rng = np.random.default_rng(3)
        nb, sz = 40, 50
        n = nb * sz
        block = np.repeat(np.arange(nb), sz)
        iu, ju = np.triu_indices(n, 1)
        p = np.where(block[iu] == block[ju], 0.22, 0.0015)
        keep_e = rng.random(len(iu)) < p
        E = np.column_stack([iu[keep_e], ju[keep_e]])
        np.savetxt(td / "edges.txt", E, fmt="%d")
        e3_out = td / "efficiency_toy"
        sets = [f"out_dir={e3_out}", "efficiency.datasets=[{name: toy, edges: " + f"{td}/edges.txt" + "}, "
                "{name: geo_toy, geo_run: " + f"{geo_out}" + "}]",
                "efficiency.algos=[kcore, ballcore_f, ballcore_r, greedy, exact, kr_greedy]",
                "efficiency.ablation_algos=[]", "efficiency.attr_model=communities",
                "efficiency.theta_scale_default=auto", "efficiency.calibration_queries=8",
                "efficiency.calibration_time_limit=2", "efficiency.ks=[3]", "efficiency.fractions=[1.0]",
                "efficiency.dims=[8]", "efficiency.theta_scales=[1.0]", "efficiency.theta_scales_relative=true",
                "efficiency.n_queries=12", "efficiency.time_limit=5", "efficiency.default_dim=8"]
        cfg = load_config(None, sets)
        from lacs.utils import dump_json

        paths = run_paths(cfg)
        dump_json(cfg, paths["root"] / "config_resolved.json")
        lacs_e3.run(cfg, paths)
        ds = bridge.load_round3(e3_out, "toy")
        rep = bridge.replay_synthetic(ds)
        _check(rep["exact"], f"round-3 random stream replays bit for bit ({rep.get('check_positions')}, "
                             f"{rep.get('check_theta')})", fails)
        th = bridge.paper_theta(rep["Z"], ds["dim"], ds["s_def"], ds["sigma0"])
        _check(np.array_equal(th, ds["theta0"]), "paper_theta(sigma0) reproduces the stored tolerances", fails)
        q = bridge.default_queries(e3_out, "toy", k=3, dim=ds["dim"])
        _check(len(q["batches"]) == 4 and q["n_pooled"] == 48, f"default queries: 4 batches, {q['n_pooled']} pooled",
               fails)

        # ---- P0 drivers
        p0cfg = {"out_root": str(td / "p0"), "seed": 0, "e3_run": str(e3_out), "geo_runs": {"geo_toy": str(geo_out)},
                 "k": 3, "time_limit": 5, "time_limit_baselines": 3, "n_proc": 2, "members_max": 5000,
                 "algos_sweep": ["exact", "greedy", "ballcore_f", "ballcore_r", "kr_greedy", "pairwise", "kr_tau",
                                 "edgeacc"],
                 "experiments": {"e2": {"settings": [{"name": "sigma0.0", "sigma": 0.0},
                                                     {"name": "sigma0.4", "sigma": 0.4}]},
                                 "e4_scale": {"settings": [{"name": "c1.0", "c": 1.0}]},
                                 "e4_mix": {"settings": [{"name": "mu0.2", "mu": 0.2}]},
                                 "e4_queries": {"n": 12, "seed": 7}},
                 "e3": {"solvers": ["bnb", "krmax", "ilp", "cpsat"], "time_limit": 10, "cap": 3000,
                        "hang_slack": 50}}
        cfg_path = td / "p0.yaml"
        cfg_path.write_text(yaml.safe_dump(p0cfg))
        e1_toy = td / "p0" / "toy" / "e1_input"
        export_groups.export_synth(str(e3_out), "toy", str(e1_toy), n_proc=2, time_limit=5, time_limit_baselines=3)
        from . import e1_stress, e3_baselines

        e1cfg = KIT / "configs" / "e1_quick.yaml"
        for name, src in (("toy", e1_toy), ("geo_toy", e1_geo)):
            outd = td / "p0" / name / "e1" / "default"
            e1_stress.main(["--config", str(e1cfg), "--instance", str(src / "instance.json"), "--groups",
                            str(src / "groups.jsonl.gz"), "--out", str(outd), "--workers", "2", "--lcc-only",
                            "--other-sample", "5"])
        for exp in ("e2", "e4_scale", "e4_mix", "e4_queries"):
            engine_sweeps.main(["--config", str(cfg_path), "--dataset", "toy", "--experiment", exp])
        pq = __import__("pandas").read_csv(td / "p0" / "toy" / "e2" / "sigma0.4" / "per_query.csv")
        r3 = __import__("pandas").read_csv(e3_out / "results" / "e3_per_query.csv")
        r3 = r3[(r3["algo"] == "exact") & (r3["sweep"] == "k")].set_index("q")["size"]
        mine = pq[pq["algo"] == "exact"].set_index("q")["size"]
        c = r3.index.intersection(mine.index)
        _check(len(c) > 0 and bool((r3.loc[c] == mine.loc[c]).all()),
               f"E2 at sigma0 reproduces round 3's exact sizes ({len(c)} queries)", fails)
        for name in ("toy", "geo_toy"):
            e3_baselines.main(["--config", str(cfg_path), "--dataset", name, "--workers", "2", "--engine-rerun"])
            x = __import__("pandas").read_csv(td / "p0" / name / "e3" / "e3_crosscheck.csv")
            _check(len(x) == 0, f"E3 {name}: proven sizes agree across solvers and the engine", fails)
            pq = __import__("pandas").read_csv(td / "p0" / name / "e3" / "e3_per_query.csv")
            err = pq[pq["status"] == "error"]
            msg = "; ".join(f"{r.solver} q{r.query}" for r in err.head(5).itertuples())
            _check(len(err) == 0, f"E3 {name}: no solver errors, hangs or crashes ({len(err)} {msg})", fails)
            for r in err.head(2).itertuples():
                print(f"    -- {r.solver} on query {r.query}:\n{r.message}")
            for solver in p0cfg["e3"]["solvers"]:
                if solver == "cpsat" and ortools_version() is None:
                    continue
                n_s = int((pq["solver"] == solver).sum())
                _check(n_s == len(set(pq["query"])), f"E3 {name}: {solver} ran on every query ({n_s})", fails)
        # ---- resuming after an interrupted run (killed while appending to the checkpoints)
        from .formats import read_jsonl as _rj

        e1_dir = td / "p0" / "toy" / "e1" / "default"
        rows_p = e1_dir / "e1_rows.jsonl.gz"
        before = [(r["query"], r["method"], r["variant"]) for r in _rj(str(rows_p))]
        raw = rows_p.read_bytes()
        rows_p.write_bytes(raw[: int(len(raw) * 0.6)])
        (e1_dir / "DONE").unlink()
        e1_stress.main(["--config", str(e1cfg), "--instance", str(e1_toy / "instance.json"), "--groups",
                        str(e1_toy / "groups.jsonl.gz"), "--out", str(e1_dir), "--workers", "2", "--lcc-only",
                        "--other-sample", "5"])
        after = [(r["query"], r["method"], r["variant"]) for r in _rj(str(rows_p))]
        _check(sorted(after) == sorted(before) and len(set(after)) == len(after),
               f"E1 resumes after a killed run: {len(after)} rows, none lost or doubled", fails)
        e3_dir = td / "p0" / "toy" / "e3"
        pq0 = __import__("pandas").read_csv(e3_dir / "e3_per_query.csv")
        ck = e3_dir / "e3_rows.jsonl.gz"
        raw = ck.read_bytes()
        ck.write_bytes(raw[: int(len(raw) * 0.5)])
        e3_baselines.main(["--config", str(cfg_path), "--dataset", "toy", "--workers", "2", "--engine-rerun"])
        pq1 = __import__("pandas").read_csv(e3_dir / "e3_per_query.csv")
        k0 = pq0.set_index(["query", "solver"])
        k1 = pq1.set_index(["query", "solver"])
        same_keys = sorted(k0.index) == sorted(k1.index)
        pv = k0[k0["proven"].astype(bool)].index
        same_sizes = same_keys and bool((k0.loc[pv, "size"] == k1.loc[pv, "size"]).all())
        x = __import__("pandas").read_csv(e3_dir / "e3_crosscheck.csv")
        _check(same_keys and same_sizes and len(x) == 0,
               f"E3 resumes after a killed run: {len(pq1)} rows, same proven sizes", fails)
        t_skip = time.time()
        export_groups.export_synth  # noqa: B018 (the CLI skips finished exports)
        export_groups.main(["synth", "--e3-run", str(e3_out), "--dataset", "toy", "--out", str(e1_toy)])
        _check(time.time() - t_skip < 5, "a finished export is skipped on a rerun", fails)
        report.main(["--root", str(td / "p0"), "--e3-run", str(e3_out)])
        nums = json.loads((td / "p0" / "report" / "p0_numbers.json").read_text())
        viol = nums.get("E1", {}).get("theorem_violations", [])
        _check(not viol, f"E1: LCC groups never break without assumption violations ({viol[:3]})", fails)
        _check((td / "p0" / "report" / "REPORT_P0.md").exists(), "report built", fails)
    except Exception:
        traceback.print_exc()
        fails.append("integration crashed")
    finally:
        if keep:
            print(f"  (kept {td})")
        else:
            shutil.rmtree(td, ignore_errors=True)
    return fails


def _unit_in_child() -> list:
    """The unit checks run HiGHS and CP-SAT; they do so in a child process so that this
    process stays free of solver threads before the integration part forks workers."""
    import subprocess

    env = dict(os.environ)
    env["PYTHONPATH"] = str(KIT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    rc = subprocess.run([sys.executable, "-m", "p0kit.selftest", "--unit-child"], env=env).returncode
    return [] if rc == 0 else ["unit checks"]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m p0kit.selftest")
    ap.add_argument("--unit-only", action="store_true")
    ap.add_argument("--unit-child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--keep", action="store_true", help="keep the integration directory")
    args = ap.parse_args(argv)
    if args.unit_child:
        fails = unit()
        print("unit checks:", "passed" if not fails else f"failed: {fails}")
        sys.exit(0 if not fails else 1)
    from . import ffi
    from .runner_common import ortools_version

    print("library:", ffi.lib_path(), ffi.versions())
    ov = ortools_version()
    print(f"ortools {ov}" if ov else "ortools not installed: CP-SAT checks skipped (pip install ortools)")
    fails = _unit_in_child()
    if not args.unit_only:
        fails += integration(args.keep)
    print("\nSELFTEST", "PASSED" if not fails else f"FAILED: {fails}")
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
