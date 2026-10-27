"""Self-test of the P2 package (run from the package root: PYTHONPATH=p0:. python -m unittest -v p2.tests.test_p2).

On small synthetic data built with the project's own code (a meetup run laid out like round 4, and a
round-3-style efficiency run like P0's self-test), checks:
  * geo.theta_mult: unset or 1 gives byte-identical geo-e2 outputs, also against the original v0.2 geo.py;
    s != 1 is exactly "every tolerance times s" (same output as scaling users.csv itself), in the engine's
    input file and in every method (LCC, edgeacc, pairwise);
  * p2.e11: s = 1 reproduces geo-e2 (the gate passes; an injected mismatch fails it with exit code 4); the
    query selection matches the reference sets (all / the parameter-study sample; a wrong seed -> exit 6);
    a rerun after an interruption reuses the cached engine blocks, --fresh (the pilot) removes them and
    records it; a finished run is not reused for other query options (exit 7); the pilot decision rule of
    plan 4.1, which refuses a pilot that was not fresh (exit 7);
  * p2.watchdog: identical rows to CppSearch.run; a hanging engine is killed and reported; a failing engine
    process is reported at once;
  * p2.e4v05: replay check, the v0.4/v0.5 gate (pass, and exit 4 on an injected mismatch), summary;
  * P2-4b chain: the LCC groups of the E11 runs at s = 0.8 and 1 read back from their cached engine blocks
    (no engine call; same members as E11), dynamics, summary (theorem check 0);
  * p2.bundle: only allowed files, refuses forbidden ones.
About 5-10 minutes on 2 cores.
"""
from __future__ import annotations

import contextlib
import gzip
import importlib.util
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "p0", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from lacs.__main__ import dispatch  # noqa: E402
from lacs.config import load_config, run_paths  # noqa: E402
from lacs.utils import setup_logging  # noqa: E402

from p2 import e11, e4v05, watchdog  # noqa: E402

TIMING = ["lcc_time_ms", "lcc_ub_ms", "pairwise_time_ms"]
ORIG_GEO = Path(__file__).resolve().parent / "geo_v02_original.py"
ORIG_GEO_SHA = "6de6aa696818fad93490a165869fe9611a545a29f1b255e87ffe4c025c38b45e"


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*a, **k)
    return rc, buf.getvalue()


def read_matrix(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        assert f.read(8) == b"LACSMT01"
        r = int(np.frombuffer(f.read(8), np.int64)[0])
        c = int(np.frombuffer(f.read(8), np.int64)[0])
        return np.frombuffer(f.read(4 * r * c), np.float32).reshape(r, c)


class P2Test(unittest.TestCase):
    td: Path

    @classmethod
    def setUpClass(cls):
        setup_logging("WARNING")
        cls.td = Path(tempfile.mkdtemp(prefix="p2_selftest_"))
        td = cls.td
        raw = td / "lbsn"
        cls.r4 = td / "r4"
        src = cls.r4 / "geo_toy"
        base = [f"out_dir={src}", f"geo.synthetic={{dir: {raw}, n_users: 450, n_days: 240, seed: 2}}",
                f"geo.edges={raw}/edges.txt", f"geo.checkins={raw}/checkins.txt", "geo.g3_queries=40",
                "search.k=3", "search.n_queries=40", "search.time_limit=10", "search.n_proc=2"]
        cfg = load_config(None, base)
        paths = run_paths(cfg)
        for step in ("geo-prepare", "geo-gates"):
            dispatch(step, cfg, paths)
        # round-4 layout: results/r4/ with the main run (all queries) and the tolerance study (sample of 40)
        r4sets = base + ["geo.synthetic=null", "geo.results_subdir=r4", "geo.queries=all",
                         "geo.tolerance_fixed=theta_q50", "geo.methods=[lcc_exact, pairwise, edgeacc, kcore]",
                         "geo.e2_sweep_k=[3]", "geo.e2_sweep_tolerances=[theta_q50, theta_q70, theta_q90]",
                         "geo.sweep_tolerance_queries=40"]
        cfg4 = load_config(None, r4sets)
        p4 = run_paths(cfg4)
        dispatch("geo-e2", cfg4, p4)
        dispatch("geo-e2-sweep", cfg4, p4)
        cls.ref_main = src / "results" / "r4" / "geo_e2_per_query.csv"
        cls.ref_sweep = src / "results" / "r4" / "geo_e2_sweep_per_query.csv"
        # the E11 configuration of the toy dataset (as p2/configs/e11_gowalla.yaml)
        cls.e11root = td / "runs_p2" / "e11" / "geo_toy"
        cls.cfg = td / "e11_geo_toy.yaml"
        cls.cfg.write_text(yaml.safe_dump({
            "name": "e11_geo_toy", "out_dir": str(cls.e11root), "seed": 0,
            "geo": {"tolerance_fixed": "theta_q50", "queries": "all", "engine_block": 25, "methods": ["lcc_exact"],
                    "member_analysis": [], "abstention": []},
            "search": {"k": 3, "n_queries": 40, "time_limit": 60, "n_proc": 2}}))
        rc, _ = quiet(e11.main, ["prepare-inputs", "--src", str(src), "--dst", str(cls.e11root)])
        assert rc == 0

    @classmethod
    def tearDownClass(cls):
        if os.environ.get("P2_SELFTEST_KEEP"):
            print(f"(kept {cls.td})")
        else:
            shutil.rmtree(cls.td, ignore_errors=True)

    def run_e11(self, *args, cfg=None):
        return quiet(e11.main, ["run", "--config", str(cfg or self.cfg), "--n-proc", "2", *args])

    def per_query(self, s, name="e11", root=None):
        return pd.read_csv((root or self.e11root) / "results" / name / e11.s_label(s) / "e11_per_query.csv")

    # ------------------------------------------------------------ theta_mult --
    def test_01_theta_mult_default_is_byte_identical(self):
        from lacs.apps import geo

        self.assertEqual(e11.sha256_file(ORIG_GEO), ORIG_GEO_SHA)
        spec = importlib.util.spec_from_file_location("lacs.apps.geo_v02_original", ORIG_GEO)
        orig = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(orig)
        outs = {}
        for name, mod, extra in (("orig", orig, []), ("new", geo, []), ("mult1", geo, ["geo.theta_mult=1.0"])):
            cfg = load_config(str(self.cfg), [f"geo.results_subdir=id_{name}", "geo.queries=all",
                                               "geo.methods=[lcc_exact, lcc_greedy, pairwise, edgeacc, kcore, kr_global, kr_tau, sac, sac_sized, random_bfs]",
                                               "geo.engine_block=0", "search.time_limit=10"] + extra)
            paths = run_paths(cfg)
            mod.e2(cfg, paths)
            d = paths["results"] / f"id_{name}"
            outs[name] = {f: (d / f).read_bytes() for f in ("geo_e2_per_query.csv", "geo_e2_summary.csv", "geo_e2.json")}
        for f in outs["orig"]:
            self.assertEqual(outs["orig"][f], outs["new"][f], f"{f}: new geo.py without theta_mult differs from v0.2")
            self.assertEqual(outs["orig"][f], outs["mult1"][f], f"{f}: theta_mult = 1 differs from v0.2")

    def test_02_theta_mult_scales_every_method(self):
        s = 2.0
        rc, out = self.run_e11("--s", str(s), "--mode", "all")
        self.assertEqual(rc, 0, out)
        a = self.per_query(s)
        # the same data with users.csv itself scaled, run at s = 1
        root2 = self.td / "runs_p2" / "e11" / "geo_toy_scaled"
        shutil.copytree(self.e11root / "geo", root2 / "geo")
        shutil.copytree(self.e11root / "graph", root2 / "graph")
        u = pd.read_csv(root2 / "geo" / "users.csv")
        u["theta_q50"] = u["theta_q50"].to_numpy(np.float64) * s
        u.to_csv(root2 / "geo" / "users.csv", index=False)
        cfg2 = self.td / "e11_geo_toy_scaled.yaml"
        c = yaml.safe_load(self.cfg.read_text())
        c["out_dir"] = str(root2)
        cfg2.write_text(yaml.safe_dump(c))
        rc, out = self.run_e11("--s", "1", "--mode", "all", cfg=cfg2)
        self.assertEqual(rc, 0, out)
        b = self.per_query(1.0, root=root2)
        cols = [x for x in a.columns if x not in TIMING + ["s"]]
        pd.testing.assert_frame_equal(a[cols], b[cols], check_dtype=False)
        # the engine's tolerance file holds float32(s * theta_q50) exactly
        th = read_matrix(self.e11root / "work" / "e11_s2" / "theta_acc.bin").ravel()
        u0 = pd.read_csv(self.e11root / "geo" / "users.csv")
        self.assertTrue(np.array_equal(th, (u0["theta_q50"].to_numpy(np.float64) * s).astype(np.float32)))
        # scaled LCCs satisfy (iii) for the scaled tolerances; more queries have one than at s = 1
        nz = a[a["lcc_size"] >= 2]
        self.assertTrue(len(nz) > 0)
        self.assertTrue((nz["diam_km"] <= nz["min_theta_km"] * (1 + 1e-6) + 1e-6).all())
        rc, out = self.run_e11("--s", "1", "--mode", "all", "--results-name", "e11_cmp")
        self.assertEqual(rc, 0, out)
        one = self.per_query(1.0, name="e11_cmp")
        self.assertGreaterEqual(int((a["lcc_size"] > 0).sum()), int((one["lcc_size"] > 0).sum()))
        self.assertGreaterEqual(int((a["edgeacc_size"] > 0).sum()), int((one["edgeacc_size"] > 0).sum()))

    # --------------------------------------------------------------- gate --
    def test_03_gate_and_query_selection(self):
        rc, out = self.run_e11("--s", "1", "--mode", "all", "--expect-queries-from", str(self.ref_main),
                               "--expect-kind", "r4main")
        self.assertEqual(rc, 0, out)
        run = self.e11root / "results" / "e11" / "s1" / "e11_per_query.csv"
        mine = pd.read_csv(run)
        ref = pd.read_csv(self.ref_main)
        ref = ref[ref["method"] == "lcc_exact"].set_index("q")
        m = mine.set_index("q")
        # s = 1 reproduces geo-e2's LCC, edgeacc and pairwise answers (sizes and statuses) on every query
        self.assertEqual(sorted(m.index), sorted(ref.index))
        self.assertTrue((m.loc[ref.index, "lcc_size"].to_numpy() == ref["size"].to_numpy()).all())
        full = pd.read_csv(self.ref_main)
        for meth, col in (("edgeacc", "edgeacc"), ("pairwise", "pairwise")):
            r = full[full["method"] == meth].set_index("q")
            self.assertTrue((m.loc[r.index, f"{col}_size"].to_numpy() == r["size"].to_numpy()).all(), meth)
            self.assertTrue((m.loc[r.index, f"{col}_status"].to_numpy() == r["status"].to_numpy()).all(), meth)
        for c in ("covisit_pair_rate", "member_meet_rate"):
            rr = full[full["method"] == "lcc_exact"].set_index("q")[c]
            np.testing.assert_allclose(m.loc[rr.index, c].to_numpy(float), rr.to_numpy(float), equal_nan=True)
        g = self.td / "gate.json"
        rc, out = quiet(e11.main, ["gate", "--run", str(run), "--ref", str(self.ref_main), "--ref-kind", "r4main",
                                   "--out", str(g)])
        self.assertEqual(rc, 0, out)
        self.assertTrue(json.loads(g.read_text())["pass"])
        # an injected mismatch on one proven query fails the gate
        bad = pd.read_csv(self.ref_main)
        i = bad.index[(bad["method"] == "lcc_exact") & (bad["status"] == "optimal") & (bad["size"] >= 4)][0]
        bad.loc[i, "size"] += 1
        badf = self.td / "ref_bad.csv"
        bad.to_csv(badf, index=False)
        rc, out = quiet(e11.main, ["gate", "--run", str(run), "--ref", str(badf), "--ref-kind", "r4main",
                                   "--out", str(self.td / "gate_bad.json")])
        self.assertEqual(rc, 4, out)
        # the sample mode draws the tolerance study's queries; another seed does not
        rc, out = self.run_e11("--s", "1", "--mode", "sample", "--n", "40", "--results-name", "e11_sample",
                               "--expect-queries-from", str(self.ref_sweep), "--expect-kind", "r4sweep")
        self.assertEqual(rc, 0, out)
        rc, out = self.run_e11("--s", "1", "--mode", "sample", "--n", "40", "--results-name", "e11_badseed",
                               "--expect-queries-from", str(self.ref_sweep), "--expect-kind", "r4sweep",
                               "--set", "seed=5")
        self.assertEqual(rc, 6, out)

    # ------------------------------------------------------------- resume --
    def test_04_resume_reuses_engine_blocks(self):
        rc, out = self.run_e11("--s", "1.5", "--mode", "all", "--results-name", "e11_resume")
        self.assertEqual(rc, 0, out)
        d = self.e11root / "results" / "e11_resume" / "s1.5"
        first = (d / "e11_per_query.csv").read_bytes()
        blocks = sorted((self.e11root / "work" / "e11_resume_s1.5" / "blocks").glob("*.jsonl"))
        self.assertGreater(len(blocks), 2)
        for f in ("DONE", "e11_per_query.csv", "e11_summary.json"):
            (d / f).unlink()
        calls = []
        from lacs.search.cpp import CppSearch

        real = CppSearch.run

        def counting(self_, *a, **k):
            calls.append(1)
            return real(self_, *a, **k)

        counting.__name__ = "run_with_watchdog"
        CppSearch.run = counting
        try:
            rc, out = self.run_e11("--s", "1.5", "--mode", "all", "--results-name", "e11_resume")
        finally:
            CppSearch.run = real
        self.assertEqual(rc, 0, out)
        self.assertEqual(len(calls), 0, "a rerun called the engine although every block was cached")
        self.assertEqual((d / "e11_per_query.csv").read_bytes(), first)
        rc, out = self.run_e11("--s", "1.5", "--mode", "all", "--results-name", "e11_resume")
        self.assertIn("[skip]", out)
        # a finished run is not reused for other query options
        rc, out = self.run_e11("--s", "1.5", "--mode", "sample", "--n", "40", "--results-name", "e11_resume")
        self.assertEqual(rc, 7, out)
        # --fresh (the pilot): cached blocks are removed first, the engine runs again, the summary says so
        for f in ("DONE", "e11_per_query.csv", "e11_summary.json"):
            (d / f).unlink()
        calls.clear()
        CppSearch.run = counting
        try:
            rc, out = self.run_e11("--s", "1.5", "--mode", "all", "--results-name", "e11_resume", "--fresh")
        finally:
            CppSearch.run = real
        self.assertEqual(rc, 0, out)
        self.assertGreater(len(calls), 0, "--fresh must not reuse cached blocks")
        js = json.loads((d / "e11_summary.json").read_text())
        self.assertTrue(js["fresh"])
        self.assertEqual(js["cached_blocks_found"], len(blocks))
        self.assertEqual(js["cached_blocks_used"], 0)
        a, b = pd.read_csv(d / "e11_per_query.csv"), pd.read_csv(io.BytesIO(first))
        cols = [c for c in a.columns if c not in TIMING]
        pd.testing.assert_frame_equal(a[cols], b[cols])

    # ------------------------------------------------------------- pilot --
    def test_05_pilot_decision(self):
        def pilot(name, secs, n_full):
            d = self.td / "pilot" / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "e11_summary.json").write_text(json.dumps({"selected": n_full, "subset": {"n": 200, "of": n_full},
                                                            "timing": {"total_s": secs + 5.0, "setup_s": 5.0},
                                                            "fresh": fresh, "cached_blocks_used": 0}))
            return f"{name}={d}"

        fresh = False  # a pilot that reused cached blocks (or predates --fresh) is refused
        specs = [pilot("gowalla", 10, 15829), pilot("brightkite", 10, 5260), pilot("yelp", 10, 2000)]
        rc, out = quiet(e11.main, ["decide", "--pilot", *specs, "--out", str(self.td / "pilot" / "refused.json")])
        self.assertEqual(rc, 7, out)
        self.assertFalse((self.td / "pilot" / "refused.json").exists())
        fresh = True

        # pilot seconds (Gowalla, Brightkite, Yelp): below the threshold, fallback, fallback and stop
        cases = [(10, 10, 10), (60, 30, 120), (120, 30, 120), (2000, 2000, 2000)]
        seen = set()
        for g, b, y in cases:
            specs = [pilot("gowalla", g, 15829), pilot("brightkite", b, 5260), pilot("yelp", y, 2000)]
            outf = self.td / "pilot" / "decision.json"
            rc, out = quiet(e11.main, ["decide", "--pilot", *specs, "--out", str(outf)])
            dec = json.loads(outf.read_text())
            h_all = (15 + g / 200 * 15829 + b / 200 * 5260 + y / 200 * 2000) * 8 / 3600   # 3 x 5 s set-up
            self.assertAlmostEqual(dec["projection_h_full"], h_all, places=6)
            exp_mode = "all" if h_all <= 12 else "r4_sample"
            self.assertEqual(dec["gowalla_mode"], exp_mode, out)
            if exp_mode == "r4_sample":
                h2 = (15 + g / 200 * 2000 + b / 200 * 5260 + y / 200 * 2000) * 8 / 3600
                self.assertAlmostEqual(dec["projection_h_fallback"], h2, places=6)
                self.assertEqual(dec["stop"], h2 > 24)
                self.assertEqual(rc, 5 if h2 > 24 else 0)
            else:
                self.assertEqual(rc, 0)
            seen.add((dec["gowalla_mode"], dec["stop"]))
        self.assertEqual(seen, {("all", False), ("r4_sample", False), ("r4_sample", True)})

    # ----------------------------------------------------------- watchdog --
    def test_06_watchdog(self):
        from lacs.apps import geo
        from lacs.search.cpp import CppSearch

        watchdog.install()
        cfg = load_config(str(self.cfg), [])
        paths = run_paths(cfg)
        ctx = geo._e2_context(cfg, paths, "wd_test")
        qs = [int(q) for q in geo._select_queries(cfg, ctx, 3, "all", 0)[0][:12]]
        cs = ctx["cs"]
        plain = CppSearch._run_plain(cs, qs, ["exact", "pairwise_exact", "edgeacc"], 3, time_limit=10, n_proc=2,
                                     tag="plain")
        wd = watchdog.run_with_watchdog(cs, qs, ["exact", "pairwise_exact", "edgeacc"], 3, time_limit=10, n_proc=2,
                                        tag="wd")
        strip = lambda rows: [{k: v for k, v in r.items() if k not in ("time_ms", "ub_ms", "index_ms")} for r in rows]
        self.assertEqual(strip(plain), strip(wd))
        # a hanging engine is killed after the stall limit and reported
        fake = self.td / "fake_engine.sh"
        fake.write_text("#!/bin/sh\nsleep 60\n")
        fake.chmod(0o755)
        real_exe = cs.exe
        cs.exe = fake
        old_poll = watchdog.POLL_SECONDS
        watchdog.POLL_SECONDS = 0.2
        t0 = time.time()
        try:
            with self.assertRaises(RuntimeError) as cm:
                watchdog.run_with_watchdog(cs, qs, ["exact"], 3, time_limit=10, n_proc=2, tag="hang", _stall=1.5)
        finally:
            cs.exe = real_exe
            watchdog.POLL_SECONDS = old_poll
        self.assertIn("stalled", str(cm.exception))
        self.assertLess(time.time() - t0, 20)
        # one process fails at once while the others would run for a minute: reported at once
        fake2 = self.td / "fake_engine_fail.sh"
        fake2.write_text('#!/bin/sh\ncase "$*" in *queries_fail_0.txt*) echo boom >&2; exit 3;; *) sleep 60;; esac\n')
        fake2.chmod(0o755)
        cs.exe = fake2
        watchdog.POLL_SECONDS = 0.2
        t0 = time.time()
        try:
            with self.assertRaises(RuntimeError) as cm:
                watchdog.run_with_watchdog(cs, qs, ["exact"], 3, time_limit=10, n_proc=3, tag="fail", _stall=120)
        finally:
            cs.exe = real_exe
            watchdog.POLL_SECONDS = old_poll
        self.assertIn("failed (3)", str(cm.exception))
        self.assertIn("boom", str(cm.exception))
        self.assertLess(time.time() - t0, 15)

    # -------------------------------------------------------------- P2-5 --
    def test_07_e4v05_gate(self):
        from lacs.experiments import e3 as lacs_e3
        from lacs.utils import dump_json

        td = self.td / "e4"
        rng = np.random.default_rng(3)
        nb, sz = 30, 40
        n = nb * sz
        block = np.repeat(np.arange(nb), sz)
        iu, ju = np.triu_indices(n, 1)
        p = np.where(block[iu] == block[ju], 0.25, 0.002)
        keep = rng.random(len(iu)) < p
        td.mkdir(parents=True, exist_ok=True)
        np.savetxt(td / "edges.txt", np.column_stack([iu[keep], ju[keep]]), fmt="%d")
        e3_out = td / "efficiency_toy"
        sets = [f"out_dir={e3_out}", "efficiency.datasets=[{name: toy, edges: " + f"{td}/edges.txt" + "}]",
                "efficiency.algos=[kcore, ballcore_f, ballcore_r, greedy, exact, kr_greedy]",
                "efficiency.ablation_algos=[]", "efficiency.attr_model=communities",
                "efficiency.theta_scale_default=auto", "efficiency.calibration_queries=8",
                "efficiency.calibration_time_limit=2", "efficiency.ks=[3]", "efficiency.fractions=[1.0]",
                "efficiency.dims=[8]", "efficiency.theta_scales=[1.0]", "efficiency.theta_scales_relative=true",
                "efficiency.n_queries=12", "efficiency.time_limit=5", "efficiency.default_dim=8"]
        cfg = load_config(None, sets)
        paths = run_paths(cfg)
        dump_json(cfg, paths["root"] / "config_resolved.json")
        lacs_e3.run(cfg, paths)
        p2cfg = td / "e4v05.yaml"
        p2cfg.write_text(yaml.safe_dump({"out_root": str(td / "unused"), "seed": 0, "e3_run": "/nonexistent", "k": 3,
                                         "time_limit": 5, "time_limit_baselines": 3, "n_proc": 2, "members_max": 5000,
                                         "algos_sweep": ["exact", "greedy"],
                                         "experiments": {"e4_scale": {"settings": [{"name": "c1.0", "c": 1.0}]},
                                                         "e4_queries": {"n": 12, "seed": 7}}}))
        ref = td / "ref"
        for exp in ("e4_scale", "e4_queries"):
            rc, out = quiet(e4v05.main, ["run", "--config", str(p2cfg), "--dataset", "toy", "--experiment", exp,
                                         "--e3-run", str(e3_out), "--out-root", str(ref)])
            self.assertEqual(rc, 0, out)
            self.assertIn("exact=True", out)
        self.assertFalse((ref / "toy" / "inst").exists(), "P2-5 must not export E1 instances")
        new = td / "new"
        for exp in ("e4_scale", "e4_queries"):
            rc, out = quiet(e4v05.main, ["run", "--config", str(p2cfg), "--dataset", "toy", "--experiment", exp,
                                         "--e3-run", str(e3_out), "--out-root", str(new), "--p0-root", str(ref)])
            self.assertEqual(rc, 0, out)
            self.assertIn("-> PASS", out)
        man = json.loads((new / "toy" / "e4_scale" / "c1.0" / "manifest.json").read_text())
        self.assertIn("engine", man)
        rc, out = quiet(e4v05.main, ["summary", "--root", str(new), "--p0-root", str(ref), "--out", str(td / "rep")])
        self.assertEqual(rc, 0, out)
        s = pd.read_csv(td / "rep" / "e4v05_summary.csv")
        self.assertEqual(int(s["optimum_mismatches"].sum()), 0)
        # an injected mismatch in the reference stops P2-5 with exit code 4
        pq = ref / "toy" / "e4_scale" / "c1.0" / "per_query.csv"
        d = pd.read_csv(pq)
        i = d.index[(d["algo"] == "exact") & (d["status"] == "optimal") & (d["size"] >= 4)][0]
        d.loc[i, "size"] += 1
        d.to_csv(pq, index=False)
        rc, out = quiet(e4v05.main, ["run", "--config", str(p2cfg), "--dataset", "toy", "--experiment", "e4_scale",
                                     "--e3-run", str(e3_out), "--out-root", str(new), "--p0-root", str(ref)])
        self.assertEqual(rc, 4, out)
        self.assertIn("FAIL", out)

    # -------------------------------------------------------------- P2-4b --
    def test_08_p24b_chain(self):
        import hashlib

        from e1x import open_sweep
        from lacs.search.cpp import CppSearch

        from p2 import p24b

        root = self.td / "runs_p2" / "p24b"
        dyn_cfg = ROOT / "p2" / "configs" / "p24b_dynamics.yaml"
        u = pd.read_csv(self.e11root / "geo" / "users.csv")
        # the export needs the finished E11 run of that s
        rc, out = quiet(e11.main, ["export-groups", "--config", str(self.cfg), "--s", "1.25", "--mode", "all",
                                   "--out", str(root / "geo_toy" / "s1.25" / "e1_input")])
        self.assertEqual(rc, 2, out)
        for s in ("1", "0.8"):
            rc, out = self.run_e11("--s", s, "--mode", "all")  # (skipped when an earlier test ran it)
            self.assertEqual(rc, 0, out)
            exp = root / "geo_toy" / f"s{s}" / "e1_input"
            calls = []
            real = CppSearch.run

            def counting(self_, *a, **k):
                calls.append(1)
                return real(self_, *a, **k)

            CppSearch.run = counting
            try:
                rc, out = quiet(e11.main, ["export-groups", "--config", str(self.cfg), "--s", s, "--mode", "all",
                                           "--out", str(exp)])
            finally:
                CppSearch.run = real
            self.assertEqual(rc, 0, out)
            self.assertEqual(len(calls), 0, "the export must not start the engine")
            # an existing export is not reused for other query options
            rc, out = quiet(e11.main, ["export-groups", "--config", str(self.cfg), "--s", s, "--mode", "sample",
                                       "--n", "40", "--out", str(exp)])
            self.assertEqual(rc, 7, out)
            inst = json.loads((exp / "instance.json").read_text())
            th = np.load(exp / inst["theta"] if not Path(inst["theta"]).is_absolute() else inst["theta"])
            self.assertTrue(np.array_equal(th, u["theta_q50"].to_numpy(np.float32)),
                            "the dynamics instance must hold the unscaled tolerances")
            # the same groups as the E11 run: sizes, and member sets through E11's members_hash
            pq = self.per_query(float(s)).set_index("q")
            with gzip.open(exp / "groups.jsonl.gz", "rt") as fh:
                grp = [json.loads(x) for x in fh]
            self.assertEqual(sorted(int(r["query"]) for r in grp), sorted(int(q) for q in pq.index))
            for r in grp:
                self.assertEqual(int(r["size"]), int(pq.at[r["query"], "lcc_size"]))
                if r["size"] >= 2:
                    hh = hashlib.blake2b(np.sort(np.asarray(r["members"], np.int64)).tobytes(),
                                         digest_size=16).hexdigest()[:16]
                    self.assertEqual(hh, pq.at[r["query"], "members_hash"])
            rc, out = quiet(open_sweep.main, ["--config", str(dyn_cfg), "--instance", str(exp / "instance.json"),
                                              "--groups", str(exp / "groups.jsonl.gz"), "--out",
                                              str(root / "geo_toy" / f"s{s}" / "dyn"), "--lcc-only", "--workers", "2",
                                              "--dataset", "geo_toy", "--setting", f"s{s}"])
            self.assertEqual(rc, 0, out)
        a = self.per_query(1.0)
        b = self.per_query(0.8)
        self.assertGreaterEqual(int((a["lcc_size"] > 0).sum()), int((b["lcc_size"] > 0).sum()))
        rc, out = quiet(p24b.main, ["--root", str(root), "--out", str(root / "report")])
        self.assertEqual(rc, 0, out)
        chk = json.loads((root / "report" / "p24b_check.json").read_text())
        self.assertEqual(chk["theorem_violations"], 0)

    # ------------------------------------------------------------- bundle --
    def test_09_bundle(self):
        from p2 import bundle

        r = self.td / "bundle_root"
        ok = ["runs_p2/logs/env.log", "runs_p2/e11/x/inputs.json", "runs_p2/e11/x/results/e11/gate_s1.json",
              "runs_p2/e11/x/results/e11/s1/e11_per_query.csv", "runs_p2/e11/x/results/e11/s1/e11_summary.json",
              "runs_p2/e4v05/livejournal/e4_scale/c1.0/per_query.csv",
              "runs_p2/p24b/gowalla/s1/dyn/e1x_rows.jsonl.gz"]
        never = ["runs_p2/e11/x/geo/users.csv", "runs_p2/e11/x/graph/graph.bin",
                 "runs_p2/e11/x/work/e11_s1/blocks/b.jsonl", "runs_p2/p24b/gowalla/s1/e1_input/groups.jsonl.gz",
                 "runs_p2/e4v05/livejournal/e4_scale/c1.0/groups.jsonl.gz"]
        for f in ok + never:
            (r / f).parent.mkdir(parents=True, exist_ok=True)
            (r / f).write_text("x")
        out = r / "b.tar.gz"
        rc, txt = quiet(bundle.main, ["--root", str(r), "--out", str(out)])
        self.assertEqual(rc, 0, txt)
        with tarfile.open(out) as tf:
            names = sorted(tf.getnames())
        self.assertEqual(names, sorted(ok))
        # a forbidden file that an include pattern would take: the bundle is refused
        (r / "runs_p2/e11/pilot").mkdir(parents=True, exist_ok=True)
        (r / "runs_p2/e11/pilot/positions.npy").write_text("x")
        rc, txt = quiet(bundle.main, ["--root", str(r), "--out", str(r / "c.tar.gz")])
        self.assertEqual(rc, 1, txt)
        self.assertFalse((r / "c.tar.gz").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
