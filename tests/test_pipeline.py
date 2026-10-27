"""End-to-end pipeline on a small synthetic dataset with a planted SJT ground
truth (CPU only, ~2-4 minutes).

Checks: every step runs; G1 and G2 pass; learned acceptance latitudes track the
planted ones; C-LCC communities are always feasible and reach consensus; and
their internal accept rate beats the structure-only k-core baseline.

Run: python -m unittest tests.test_pipeline -v
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

from lacs.__main__ import STEPS, dispatch  # noqa: E402
from lacs.config import load_config, run_paths  # noqa: E402
from lacs.utils import setup_logging  # noqa: E402


class TestPipeline(unittest.TestCase):
    def test_end_to_end(self):
        setup_logging("WARNING")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            raw = td / "raw"
            sets = [
                f"out_dir={td / 'run'}", "dataset.type=synthetic", f"dataset.raw={raw}",
                "dataset.synthetic={n_users: 250, n_threads: 140, seed: 1}",
                f"features.psych_csv={raw / 'psych.csv'}",
                "encoder.backend=tfidf", "encoder.tfidf_dim=64", "pca_dim=32",
                "model.proj_dim=[4]", "model.l2_W=[1.0]", "model.l2_w=[1.0]", "model.l2_b=[10.0]",
                "model.max_iter=200", "model.n_jobs=1",
                "baselines=[prior, dist_logit]",
                "gates.g3_queries=30", "gates.g3_k=[2, 3]",
                "search.k=3", "search.n_queries=25", "search.time_limit=10", "search.cluster_k=6",
                "search.methods=[lcc_c_exact, lcc_c_greedy, lcc_r_exact, kcore, random_bfs, lcc_c_exact@global]",
                "simulation.external=false", "simulation.max_steps=300", "simulation.n_jobs=1",
            ]
            cfg = load_config(None, sets)
            paths = run_paths(cfg)
            for step in STEPS:
                dispatch(step, cfg, paths)
            R = paths["results"]
            g1 = json.loads((R / "g1.json").read_text())
            g2 = json.loads((R / "g2.json").read_text())
            self.assertTrue(g1["pass"], g1)
            self.assertTrue(g2["pass"], g2)
            # learned acceptance latitudes track the planted ones
            truth = np.load(raw / "truth.npz")
            true_acc = dict(zip(truth["users"].tolist(), truth["theta_acc"]))
            lat = pd.read_csv(R / "latitudes.csv", dtype={"user": str})
            users = pd.read_csv(paths["feat"] / "users.csv", dtype={"user": str})
            act = users["n_train_as_receiver"].to_numpy() >= 5
            t = np.array([true_acc[u] for u in lat["user"][act]])
            rho = spearmanr(t, lat["theta_acc_full"][act]).correlation
            self.assertGreater(rho, 0.3, f"latitude recovery rho={rho:.3f}")
            # community guarantees and effectiveness
            s = pd.read_csv(R / "e2_summary.csv").set_index("method")
            lcc = s[s["variant"] == "full"].loc["lcc_c_exact"]
            self.assertEqual(lcc["feasible_c_rate"], 1.0)
            self.assertEqual(lcc["hk_consensus_rate"], 1.0)
            self.assertGreater(lcc["iar_micro"], s.loc["kcore", "iar_micro"])
            self.assertIn("global", set(s["variant"]))
            self.assertTrue((R / "REPORT.md").exists())
            # stricter G3 records the k-core comparison
            g3 = json.loads((R / "g3.json").read_text())
            some_k = next(iter(g3["by_k"].values()))
            self.assertIn("R_over_kcore_median", some_k)
            # read-only diagnosis reproduces the G3 query sample exactly
            d = dispatch("diagnose", cfg, paths)
            self.assertIn(d["verdict"], {"AUDIT", "SUBSET", "CLOSE"})
            self.assertTrue(d["R1"]["pass"])
            k = str(d["geometry"]["k"])
            if k in g3["by_k"] and "C_median" in g3["by_k"][k]:
                self.assertEqual(d["geometry"]["C_greedy_median"], g3["by_k"][k]["C_median"])
            self.assertTrue((paths["root"] / "diagnosis" / "DIAGNOSIS.md").exists())


if __name__ == "__main__":
    unittest.main()
