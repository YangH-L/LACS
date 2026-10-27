"""E3 support code: size-capped label propagation and the community attribute model.

Checks: label propagation recovers planted blocks and respects the size cap; on a
planted-partition graph with community-aligned attributes the calibrated setting gives
non-empty consensus-type communities, the ablations run on the same queries, and every
exact variant that proves optimality agrees with every feasible answer.

Run: python -m unittest tests.test_e3 -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lacs.config import load_config, run_paths  # noqa: E402
from lacs.experiments import e3  # noqa: E402
from lacs.graph.io import edges_to_csr, write_graph  # noqa: E402
from lacs.search.cpp import ensure_built, lpa_labels  # noqa: E402
from lacs.utils import setup_logging  # noqa: E402


def planted(n_blocks=20, size=50, p_in=0.3, p_out=0.002, seed=0):
    rng = np.random.default_rng(seed)
    n = n_blocks * size
    block = np.repeat(np.arange(n_blocks), size)
    iu, ju = np.triu_indices(n, 1)
    p = np.where(block[iu] == block[ju], p_in, p_out)
    keep = rng.random(len(iu)) < p
    return edges_to_csr(n, iu[keep], ju[keep]), block, np.column_stack([iu[keep], ju[keep]])


class TestLPA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_built()

    def test_recovers_blocks_and_respects_cap(self):
        A, block, _ = planted()
        with tempfile.TemporaryDirectory() as td:
            write_graph(Path(td) / "g.bin", A)
            lab, info = lpa_labels(Path(td) / "g.bin", Path(td) / "lab.bin", max_size=1000, seed=1)
            self.assertGreater(normalized_mutual_info_score(block, lab), 0.9, info)
            lab2, info2 = lpa_labels(Path(td) / "g.bin", Path(td) / "lab2.bin", max_size=30, seed=1)
            self.assertLessEqual(int(np.bincount(lab2).max()), 30, info2)


class TestE3Communities(unittest.TestCase):
    def test_small_run(self):
        setup_logging("WARNING")
        _, _, E = planted(n_blocks=30, size=60, p_in=0.25, p_out=0.001, seed=3)
        with tempfile.TemporaryDirectory() as td:
            np.savetxt(Path(td) / "edges.txt", E, fmt="%d")
            sets = [f"out_dir={td}/run", "efficiency.datasets=[{name: planted, edges: " + f"{td}/edges.txt" + "}]",
                    "efficiency.algos=[kcore, greedy, ballcore_f, exact]",
                    "efficiency.ablation_algos=[exact_noindex, exact_nolocal, exact_basic]",
                    "efficiency.attr_model=communities", "efficiency.theta_scale_default=auto",
                    "efficiency.calibration_queries=8", "efficiency.calibration_time_limit=2",
                    "efficiency.ks=[3]", "efficiency.fractions=[]", "efficiency.dims=[]",
                    "efficiency.theta_scales=[]", "efficiency.n_queries=8", "efficiency.time_limit=5",
                    "efficiency.default_dim=8"]
            cfg = load_config(None, sets)
            paths = run_paths(cfg)
            out = e3.run(cfg, paths)
            res = json.loads((paths["results"] / "e3.json").read_text())
            self.assertIn("planted", res["calibration"])
            self.assertEqual(out["consistency"]["contradictions"], 0, out["consistency"])
            self.assertGreater(out["consistency"]["checked"], 0)
            import pandas as pd

            # calibration stops at the first scale whose median community reaches the target; with 8
            # queries the share of non-empty answers at that scale is noisy, so only require some
            cal = res["calibration"]["planted"]
            self.assertGreaterEqual(cal["table"][-1]["exact_incumbent_median"], 10, cal)
            s = pd.read_csv(paths["results"] / "e3_summary.csv")
            ex = s[(s["sweep"] == "k") & (s["algo"] == "exact")].iloc[0]
            self.assertGreater(ex["nonempty_rate"], 0)
            self.assertIn("ablation", set(s["sweep"]))
            self.assertTrue((s.loc[s["algo"] == "exact", "index_edge_frac"].between(0, 1)).all())
            first = pd.read_csv(paths["results"] / "e3_per_query.csv")

            # rerun with resume and an extra dataset that cannot be loaded: the finished dataset is
            # taken from its checkpoint unchanged, the broken one is reported and skipped
            sets2 = [x for x in sets if not x.startswith("efficiency.datasets=")]
            sets2 += ["efficiency.resume=true",
                      "efficiency.datasets=[{name: planted, edges: " + f"{td}/edges.txt" + "}, "
                      "{name: missing, edges: " + f"{td}/no_such_file.txt" + "}]"]
            cfg2 = load_config(None, sets2)
            out2 = e3.run(cfg2, run_paths(cfg2))
            res2 = json.loads((paths["results"] / "e3.json").read_text())
            self.assertTrue(res2["datasets_info"]["planted"].get("resumed"))
            self.assertIn("missing", out2["failed"])
            again = pd.read_csv(paths["results"] / "e3_per_query.csv")
            pd.testing.assert_frame_equal(first, again)


if __name__ == "__main__":
    unittest.main()
