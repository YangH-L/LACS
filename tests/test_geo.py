"""Meetup application (lacs/apps/geo.py) end to end on a small synthetic LBSN with a
planted personal travel tolerance (CPU, about half a minute).

Checks: homes/tolerances are recovered; G1-geo, G2-geo and G3-geo pass; latitude-
consistent communities co-visit more in the test period than the k-core and than
size-matched random connected groups; every LCC is feasible; the k sweep writes one
block per k and repeats geo-e2 exactly at the same k.

Run: python -m unittest tests.test_geo -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lacs.__main__ import dispatch  # noqa: E402
from lacs.config import load_config, run_paths  # noqa: E402
from lacs.graph.io import edges_to_csr  # noqa: E402
from lacs.utils import setup_logging  # noqa: E402


class TestGeo(unittest.TestCase):
    def test_end_to_end(self):
        setup_logging("WARNING")
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw"
            sets = [f"out_dir={td}/run", f"geo.synthetic={{dir: {raw}, n_users: 450, n_days: 240, seed: 2}}",
                    f"geo.edges={raw}/edges.txt", f"geo.checkins={raw}/checkins.txt", "geo.g3_queries=40",
                    "search.k=3", "search.n_queries=40", "search.time_limit=10", "search.n_proc=1"]
            cfg = load_config(None, sets)
            paths = run_paths(cfg)
            for step in ("geo-prepare", "geo-gates", "geo-e2"):
                dispatch(step, cfg, paths)
            R = paths["results"]
            gates = json.loads((R / "geo_gates.json").read_text())
            self.assertTrue(gates["G1"]["pass"], gates["G1"])
            self.assertTrue(gates["G2"]["pass"], gates["G2"])
            self.assertTrue(gates["G3"]["pass"], gates["G3"])
            # estimated tolerances track the planted ones
            users = pd.read_csv(paths["root"] / "geo" / "users.csv")
            truth = np.load(raw / "truth.npz")
            rho = spearmanr(users[gates["G2"]["tolerance"]], truth["tolerance_km"][users["user"]]).correlation
            self.assertGreater(rho, 0.5)
            # effectiveness against external (test-period) ground truth
            per = pd.read_csv(R / "geo_e2_per_query.csv")
            s = pd.read_csv(R / "geo_e2_summary.csv").set_index("method")
            self.assertGreater(s.loc["lcc_exact", "covisit_pair_rate"], s.loc["kcore", "covisit_pair_rate"])
            self.assertGreater(s.loc["lcc_exact", "covisit_pair_rate"], s.loc["random_bfs", "covisit_pair_rate"])
            lcc = per[(per["method"] == "lcc_exact") & (per["size"] >= 2)]
            self.assertTrue((lcc["diam_km"] <= lcc["min_theta_km"] * (1 + 1e-4) + 1e-3).all())
            e2 = json.loads((R / "geo_e2.json").read_text())
            self.assertIn("kcore", e2["paired_covisit_pair_rate_vs_lcc_exact"])
            # k sweep: one block per k; at the main k it repeats geo-e2 (same queries, same answers)
            cfg_s = load_config(None, sets + ["geo.e2_sweep_k=[2, 3]"])
            dispatch("geo-e2-sweep", cfg_s, run_paths(cfg_s))
            sw = pd.read_csv(R / "geo_e2_sweep_per_query.csv")
            self.assertEqual(sorted(sw["k"].unique()), [2, 3])
            cols = ["q", "method", "size", "covisit_pair_rate"]
            a = sw[(sw["k"] == 3) & (sw["method"] != "random_bfs")][cols].reset_index(drop=True)
            b = per[per["method"] != "random_bfs"][cols].reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b, check_dtype=False)
            js = json.loads((R / "geo_e2_sweep.json").read_text())
            self.assertEqual(sorted(r["k"] for r in js["runs"]), [2, 3])
            self.assertTrue((pd.read_csv(R / "geo_e2_sweep_summary.csv").groupby("k").size() > 0).all())

            # round-4 style re-evaluation into results/r4: every eligible query, new baselines,
            # tolerance sweep, abstention and member-level analyses; round-3 files stay untouched
            before = {f.name: f.read_bytes() for f in R.iterdir() if f.is_file()}
            sets4 = sets + ["geo.results_subdir=r4", "geo.queries=all",
                            "geo.methods=[lcc_exact, lcc_greedy, pairwise, kcore, kr_global, kr_tau, sac, sac_sized, edgeacc, random_bfs]",
                            "geo.member_analysis=[sac, kr_global]", "geo.abstention=[sac, kr_global]",
                            "geo.e2_sweep_k=[3]", "geo.e2_sweep_tolerances=[theta_q50, theta_q90]",
                            "geo.sweep_tolerance_queries=60"]
            cfg4 = load_config(None, sets4)
            p4 = run_paths(cfg4)
            dispatch("geo-e2", cfg4, p4)
            dispatch("geo-e2-sweep", cfg4, p4)
            self.assertEqual(before, {f.name: f.read_bytes() for f in R.iterdir() if f.is_file()})
            with self.assertRaises(ValueError):  # data steps refuse a re-evaluation config
                dispatch("geo-gates", cfg4, p4)
            R4 = R / "r4"
            j4 = json.loads((R4 / "geo_e2.json").read_text())
            self.assertEqual(j4["query_mode"], "all")
            self.assertEqual(j4["queries"], j4["eligible"])
            self.assertGreater(j4["queries"], 40)
            per4 = pd.read_csv(R4 / "geo_e2_per_query.csv")
            # the sampled round-3 queries are a subset; their LCC answers are unchanged
            old = per[per["method"] == "lcc_exact"].set_index("q")["size"]
            new = per4[per4["method"] == "lcc_exact"].set_index("q")["size"]
            self.assertTrue((new.loc[old.index] == old).all())
            # kr_tau contains an LCC-sized feasible answer, so it is never smaller than the LCC
            kt = per4[per4["method"] == "kr_tau"].set_index("q")["size"]
            lc = new[new >= 2]
            self.assertTrue((kt.loc[lc.index] >= lc).all())
            self.assertTrue((per4[per4["method"] == "sac"]["size"] > 0).all())
            ss = per4[per4["method"] == "sac_sized"].set_index("q")["size"]
            self.assertTrue((ss.loc[lc.index] >= lc).all())  # grown to at least the LCC's size
            self.assertIn("sac", j4["abstention"])
            ml = j4["member_level"]["sac"]
            self.assertIn("meet_rate_accept", ml)
            if min(ml["cells_accept_meet_nomeet_reject_meet_nomeet"]) >= 20:
                self.assertIn("model_a", ml)
            sw4 = pd.read_csv(R4 / "geo_e2_sweep_summary.csv")
            self.assertEqual(set(sw4["sweep"]), {"k", "tolerance"})
            self.assertEqual(set(sw4.loc[sw4["sweep"] == "tolerance", "tolerance"]), {"theta_q50", "theta_q90"})

    def test_indexed_covisits_match_reference(self):
        from lacs.apps.geo import checkin_index, community_covisits, covisit_stats
        rng = np.random.default_rng(4)
        n, T = 300, 5000
        A = edges_to_csr(n, rng.integers(0, n, 900), rng.integers(0, n, 900))
        tu = rng.integers(0, n, T)
        tk = rng.integers(0, 400, T)
        idx = checkin_index(tu, tk, n)
        for _ in range(40):
            H = np.sort(rng.choice(n, size=int(rng.integers(0, 60)), replace=False))
            a = community_covisits(H, tu, tk, A)
            b, met = covisit_stats(H, idx, A, members=True)
            self.assertEqual(set(a), set(b))
            for key in a:
                if isinstance(a[key], float) and np.isnan(a[key]):
                    self.assertTrue(np.isnan(b[key]))
                else:
                    self.assertAlmostEqual(a[key], b[key])
            if len(H) >= 2:
                self.assertAlmostEqual(met.mean(), a["member_meet_rate"])


if __name__ == "__main__":
    unittest.main()
