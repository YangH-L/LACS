"""Self-test of the E5 main chain on synthetic Yelp-format data: feasibility parse (no engine) ->
e5/yelp_prepare.py -> geo-gates / geo-e2 / geo-e2-sweep (v0.6 options) -> e5/analysis.py."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e5 import analysis, make_fixture, yelp_prepare  # noqa: E402
from e5 import yelp_feasibility as yf  # noqa: E402

ARGS = ["--metro-min-business", "100", "--vocab-min-business", "5", "--pairs-random", "20000", "--pairs-fof",
        "20000", "--no-engine"]


class TestE5Main(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="e5main_"))
        make_fixture.make(cls.tmp / "fx", seed=7)
        z = cls.tmp / "fx" / "Yelp-JSON.zip"
        rc = yf.main(["--zip", str(z), "--data-dir", str(cls.tmp / "json"), "--work", str(cls.tmp / "work"),
                      "--out", str(cls.tmp / "feas"), "--expected-sha256", yf.sha256_file(z), *ARGS])
        assert rc == 0
        R = json.loads((cls.tmp / "feas" / "feasibility.json").read_text())
        R["decision"]["spec"] = {"spec": "G", "outcome": "test"}  # the fixture's decision is not what matters here
        R["decision"]["split"]["split"] = "precovid"
        cls.feas = cls.tmp / "feas" / "feasibility_G.json"
        cls.feas.write_text(json.dumps(R))
        cls.R = R
        cls.out = cls.tmp / "e5_main"
        cls.rc = yelp_prepare.main(["--json-dir", str(cls.tmp / "json"), "--work", str(cls.tmp / "work"),
                                    "--feas", str(cls.feas), "--out-dir", str(cls.out)])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_prepare_consistent_and_private(self):
        self.assertEqual(self.rc, 0)
        rep = json.loads((self.out / "results" / "e5_prepare.json").read_text())
        self.assertEqual(rep["status"], "OK")
        self.assertTrue(rep["parse_cache_reused"])
        for k, (a, b) in rep["consistency"].items():
            if k.startswith("theta"):
                self.assertAlmostEqual(a, b, places=6, msg=k)
            else:
                self.assertEqual(a, b, k)
        users = pd.read_csv(self.out / "geo" / "users.csv")
        self.assertEqual(list(users["user"]), list(range(len(users))))  # no Yelp ids anywhere
        self.assertTrue({"theta_q50", "theta_q70", "theta_q90", "n_train"} <= set(users.columns))
        self.assertTrue((users["theta_q50"] <= users["theta_q70"] + 1e-9).all())
        self.assertTrue((users["theta_q70"] <= users["theta_q90"] + 1e-9).all())
        self.assertGreaterEqual(users["theta_q50"].min(), 1.0)
        for f in ("positions.npy", "covisits.npz", "test_checkins.npz", "test_checkins_month.npz",
                  "test_checkins_notop.npz", "meta.json"):
            self.assertTrue((self.out / "geo" / f).exists(), f)
        self.assertTrue((self.out / "graph" / "graph.bin").exists())

    def test_outcome_keys(self):
        m = np.load(self.out / "geo" / "test_checkins.npz")
        mo = np.load(self.out / "geo" / "test_checkins_month.npz")
        nt = np.load(self.out / "geo" / "test_checkins_notop.npz")
        np.testing.assert_array_equal(m["user"], mo["user"])
        np.testing.assert_array_equal(mo["key"] // 12, m["key"])
        self.assertTrue(((mo["key"] % 12) >= 0).all() and ((mo["key"] % 12) < 12).all())
        # independent recomputation from the parsed arrays: 2019 reviews of active users per calendar month
        (_, _, _, _, _, _, _, _, _, _, U, B, D, _, NU, NB) = yf.load_or_parse(
            {f["kind"]: self.tmp / "json" / f["file"] for f in self.R["inventory"]["files"]},
            yf.inventory_dir(self.tmp / "json"), self.tmp / "work")
        per, _ = yf.split_periods(D, "precovid")
        meta = json.loads((self.out / "geo" / "meta.json").read_text())
        self.assertEqual(int(len(m["user"])), meta["test_reviews"])
        # notop removes exactly the businesses at or above the 99th percentile of 2019 review counts
        bc = np.bincount(B[per == 2], minlength=NB)
        thr = np.quantile(bc[bc > 0], 0.99)
        self.assertTrue((bc[nt["key"]] < thr).all())
        self.assertEqual(int((bc[m["key"]] < thr).sum()), int(len(nt["key"])))
        self.assertEqual(meta["top1pct_businesses"], int((bc >= thr).sum()))
        months = D[per == 2].astype("datetime64[D]").astype("datetime64[M]").astype(np.int64) % 12
        self.assertLessEqual(len(mo["key"]), int((per == 2).sum()))
        self.assertTrue(set(np.unique(mo["key"] % 12)) <= set(np.unique(months)))

    def test_mismatch_stops_before_writing(self):
        R = json.loads(self.feas.read_text())
        R["splits"]["precovid"]["eligible_queries"]["k3"] += 1
        bad = self.tmp / "feas" / "feasibility_bad.json"
        bad.write_text(json.dumps(R))
        out = self.tmp / "e5_bad"
        rc = yelp_prepare.main(["--json-dir", str(self.tmp / "json"), "--work", str(self.tmp / "work"),
                                "--feas", str(bad), "--out-dir", str(out)])
        self.assertEqual(rc, 3)
        self.assertFalse((out / "geo" / "users.csv").exists())
        self.assertEqual(json.loads((out / "results" / "e5_prepare.json").read_text())["status"], "MISMATCH")

    def test_stale_cache_is_not_overwritten(self):
        bad = self.tmp / "work_stale"
        bad.mkdir(exist_ok=True)
        (bad / "parsed.npz").write_bytes(b"not a cache")
        out = self.tmp / "e5_stale"
        rc = yelp_prepare.main(["--json-dir", str(self.tmp / "json"), "--work", str(bad),
                                "--feas", str(self.feas), "--out-dir", str(out)])
        self.assertEqual(rc, 0)
        self.assertEqual((bad / "parsed.npz").read_bytes(), b"not a cache")  # the feasibility cache is untouched
        self.assertTrue((out / "work" / "parse" / "parsed.npz").exists())
        a = pd.read_csv(out / "geo" / "users.csv")
        b = pd.read_csv(self.out / "geo" / "users.csv")
        pd.testing.assert_frame_equal(a, b)

    def test_geo_chain_and_analysis(self):
        from lacs.apps import geo
        from lacs.config import load_config, run_paths
        cfg_main = yaml.safe_load((ROOT / "configs" / "e5_yelp.yaml").read_text())
        cfg_main["out_dir"] = str(self.out)
        cfg_main["search"].update({"time_limit": 2, "n_proc": 2})
        cfg_main["geo"].update({"engine_block": 7, "e2_sweep_k": [2, 3], "e2_sweep_k_n": 15,
                                "sweep_tolerance_queries": 15, "g3_queries": 20})
        p = self.tmp / "e5_test.yaml"
        p.write_text(yaml.safe_dump(cfg_main))
        cfg = load_config(p)
        paths = run_paths(cfg)
        gt = geo.gates(cfg, paths)
        self.assertIn("G2", gt)
        out = geo.e2(cfg, paths)
        self.assertEqual(out["tolerance"], "theta_q50")
        rep = json.loads((self.out / "results" / "e5_prepare.json").read_text())
        self.assertEqual(out["queries"], rep["consistency"]["eligible_k3"][0])  # every eligible query
        df = pd.read_csv(self.out / "results" / "geo_e2_per_query.csv")
        for c in ("covisit_pair_rate__month", "covisit_pair_rate__notop", "members_hash"):
            self.assertIn(c, df.columns)
        ok = df["covisit_pair_rate"].notna()
        self.assertTrue((df.loc[ok, "covisit_pair_rate__month"] <= df.loc[ok, "covisit_pair_rate"] + 1e-12).all())
        self.assertTrue((df.loc[ok, "covisit_pair_rate__notop"] <= df.loc[ok, "covisit_pair_rate"] + 1e-12).all())
        sw = geo.e2_sweep(cfg, paths)
        got = [(r["sweep"], r["k"], r["tolerance"], r["queries"]) for r in sw["runs"]]
        self.assertEqual([g[:3] for g in got], [("k", 2, "theta_q50"), ("k", 3, "theta_q50"),
                                                ("tolerance", 3, "theta_q50"), ("tolerance", 3, "theta_q70"),
                                                ("tolerance", 3, "theta_q90")])
        self.assertTrue(all(g[3] <= 15 for g in got))
        rc = analysis.main(["--results", str(self.out / "results")])
        self.assertEqual(rc, 0)
        A = json.loads((self.out / "results" / "e5_analysis.json").read_text())
        self.assertIn("H1", A)
        self.assertIn("covisit_pair_rate__notop", A["metrics_robustness"])
        # rerun of geo-e2 reuses the cached engine blocks and gives identical rows
        out2 = geo.e2(cfg, paths)
        df2 = pd.read_csv(self.out / "results" / "geo_e2_per_query.csv")
        self.assertEqual(out2["queries"], out["queries"])
        pd.testing.assert_frame_equal(df.drop(columns=[]), df2)


if __name__ == "__main__":
    unittest.main()
