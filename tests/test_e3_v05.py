"""E3 driver, v0.5 (W2): ablation variants, upper bounds, extra scales, light/low-memory datasets.

Run: python -m unittest tests.test_e3_v05 -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lacs.config import load_config, run_paths  # noqa: E402
from lacs.experiments import e3  # noqa: E402
from lacs.utils import setup_logging  # noqa: E402
from tests.test_e3 import planted  # noqa: E402

V05_ABLATIONS = ["exact_v04", "exact_count", "exact_indexcolor", "exact_noorder", "exact_noseed",
                 "exact_nocolorbound", "exact_nocompat", "exact_noedgefilter", "exact_noindex", "exact_nolocal",
                 "exact_basic", "exact_cap20000"]


def _sets(td, ds, extra=()):
    return [f"out_dir={td}", f"efficiency.datasets=[{ds}]",
            "efficiency.algos=[kcore, greedy, ballcore_f, ballcore_r, exact]",
            "efficiency.attr_model=communities", "efficiency.theta_scale_default=auto",
            "efficiency.calibration_queries=8", "efficiency.calibration_time_limit=2",
            "efficiency.ks=[3]", "efficiency.fractions=[]", "efficiency.dims=[]",
            "efficiency.n_queries=8", "efficiency.time_limit=5", "efficiency.default_dim=8", *extra]


class TestE3V05(unittest.TestCase):
    def test_ablations_bounds_and_extra_scales(self):
        setup_logging("WARNING")
        _, _, E = planted(n_blocks=30, size=60, p_in=0.25, p_out=0.001, seed=3)
        with tempfile.TemporaryDirectory() as td:
            np.savetxt(Path(td) / "edges.txt", E, fmt="%d")
            cfg = load_config(None, _sets(f"{td}/run", "{name: planted, edges: " + f"{td}/edges.txt" + "}",
                                          ["efficiency.ablation_algos=[" + ", ".join(V05_ABLATIONS) + "]",
                                           "efficiency.theta_scales=[1.0]", "efficiency.theta_scales_relative=true",
                                           "efficiency.theta_scales_extra=[2.0]"]))
            paths = run_paths(cfg)
            out = e3.run(cfg, paths)
            self.assertEqual(out["failed"], {})
            self.assertEqual(out["consistency"]["contradictions"], 0, out["consistency"])
            df = pd.read_csv(paths["results"] / "e3_per_query.csv")
            self.assertEqual(set(df.loc[df["sweep"] == "ablation", "algo"]), set(V05_ABLATIONS))
            self.assertEqual(sorted(df.loc[df["sweep"] == "theta_scale", "value"].unique()), [1.0, 2.0])
            ex = df[df["algo"].str.startswith("exact")]
            opt = ex[ex["status"] == "optimal"]
            self.assertTrue((opt["ub"] == opt["size"]).all())
            rest = ex[ex["status"] != "optimal"]
            self.assertTrue((rest["ub"] >= rest["size"]).all())
            # every variant that proves optimality finds the same size as the default exact search
            base = ex[(ex["sweep"] == "k") & (ex["algo"] == "exact") & (ex["status"] == "optimal")].set_index("q")["size"]
            abl = opt[opt["sweep"] == "ablation"]
            for _, row in abl.iterrows():
                if row["q"] in base.index:
                    self.assertEqual(row["size"], base[row["q"]], msg=row["algo"])
            br = df[df["algo"] == "ballcore_r"]
            self.assertTrue((br.loc[br["status"] == "ok", "ub"] == br.loc[br["status"] == "ok", "size"]).all())
            summ = pd.read_csv(paths["results"] / "e3_summary.csv")
            self.assertIn("open_ratio_median", summ.columns)

    def test_light_lowmem_dataset_gives_the_same_results(self):
        setup_logging("WARNING")
        _, _, E = planted(n_blocks=30, size=60, p_in=0.25, p_out=0.001, seed=4)
        with tempfile.TemporaryDirectory() as td:
            np.savetxt(Path(td) / "edges.txt", E, fmt="%d")
            frames = []
            for tag, opts in (("plain", ""), ("light", ", light: true, lowmem: true, position_chunk: 97, batch_algos: true, "
                                              "expect_m: 1, override: {algos: [greedy, ballcore_f, ballcore_r, exact]}")):
                ds = "{name: planted, edges: " + f"{td}/edges.txt" + opts + "}"
                cfg = load_config(None, _sets(f"{td}/{tag}", ds, ["efficiency.theta_scales=[]"]))
                paths = run_paths(cfg)
                out = e3.run(cfg, paths)
                self.assertEqual(out["failed"], {})
                frames.append(pd.read_csv(paths["results"] / "e3_per_query.csv"))
                if tag == "light":  # a rerun reuses the converted graph; the wrong edge count is only recorded
                    work = paths["work"] / "e3_planted"
                    self.assertTrue((work / "graph.src.json").exists())
                    info = json.loads((paths["results"] / "e3.json").read_text())["datasets_info"]["planted"]
                    self.assertEqual(info["m_expected"], 1)
                    self.assertNotIn("kcore", set(frames[-1]["algo"]))
            keep = ["q", "algo", "size", "status", "s0_size", "n_candidates", "ub", "tau"]
            common = sorted(set(frames[0]["algo"]) & set(frames[1]["algo"]))
            self.assertEqual(common, ["ballcore_f", "ballcore_r", "exact", "greedy"])
            a, b = (f.loc[f["algo"].isin(common), keep].sort_values(["algo", "q"]).reset_index(drop=True) for f in frames)
            # identical queries, candidate regions and answers (times differ)
            pd.testing.assert_frame_equal(a[["q", "algo", "s0_size", "n_candidates"]],
                                          b[["q", "algo", "s0_size", "n_candidates"]])
            same = (a["size"] == b["size"]) | (a["status"] != "optimal") | (b["status"] != "optimal")
            self.assertTrue(same.all())


if __name__ == "__main__":
    unittest.main()
