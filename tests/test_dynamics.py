"""Theorem 1 / Theorem 2 as property tests.

C-LCC communities (diameter <= min theta_acc, connected k-core) must reach
consensus under heterogeneous HK dynamics with no rejected interaction; R-LCC
communities (diameter <= min theta_rej) must see no repulsive interaction under
SJT dynamics.

Run: python -m unittest tests.test_dynamics -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lacs.eval import dynamics  # noqa: E402
from lacs.graph.io import edges_to_csr  # noqa: E402
from lacs.search.cpp import CppSearch  # noqa: E402


def planted(rng, n=150, groups=5):
    g = rng.integers(0, groups, size=n)
    centers = rng.normal(scale=2.0, size=(groups, 3))
    pos = (centers[g] + 0.6 * rng.normal(size=(n, 3))).astype(np.float32)
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < (0.3 if g[i] == g[j] else 0.02)]
    acc = rng.gamma(6.0, 0.3, size=n).astype(np.float32)
    rej = (acc + rng.gamma(4.0, 0.25, size=n)).astype(np.float32)
    A = edges_to_csr(n, np.array([e[0] for e in edges]), np.array([e[1] for e in edges]))
    return A, pos, acc, rej


class TestGuarantees(unittest.TestCase):
    def test_theorem1_consensus(self):
        rng = np.random.default_rng(5)
        checked = 0
        with tempfile.TemporaryDirectory() as td:
            for t in range(4):
                A, pos, acc, rej = planted(rng)
                cs = CppSearch(A, pos, {"acc": acc, "rej": rej}, Path(td) / f"t{t}")
                for row in cs.run(range(0, A.shape[0], 5), ["exact", "greedy", "ballcore_f"], 3, theta="acc",
                                  time_limit=20):
                    H = np.array(row["members"], int)
                    if len(H) < 2:
                        continue
                    self.assertTrue(row["feasible"])
                    h = dynamics.hk(A[H][:, H], pos[H], acc[H], max_steps=3000, tol=1e-12)
                    self.assertTrue(h["consensus"], msg=f"{row['algo']} q={row['q']}: {h}")
                    self.assertEqual(h["edge_rejections_seen"], 0)
                    checked += 1
        self.assertGreater(checked, 50)

    def test_theorem2_rejection_free(self):
        rng = np.random.default_rng(6)
        checked = 0
        with tempfile.TemporaryDirectory() as td:
            for t in range(4):
                A, pos, acc, rej = planted(rng)
                cs = CppSearch(A, pos, {"acc": acc, "rej": rej}, Path(td) / f"t{t}")
                for row in cs.run(range(0, A.shape[0], 5), ["exact", "greedy"], 3, theta="rej", time_limit=20):
                    H = np.array(row["members"], int)
                    if len(H) < 2:
                        continue
                    s = dynamics.sjt(A[H][:, H], pos[H], acc[H], rej[H], mu=0.5, beta=0.5, max_steps=500)
                    self.assertEqual(s["rejection_events"], 0, msg=f"{row['algo']} q={row['q']}")
                    self.assertLessEqual(s["diam_final_ratio"], 1.0 + 1e-9)
                    checked += 1
        self.assertGreater(checked, 50)

    def test_violating_groups_can_fail(self):
        """Sanity check that the property is not vacuous: structure-only k-core
        communities frequently fragment or see rejections."""
        rng = np.random.default_rng(7)
        A, pos, acc, rej = planted(rng)
        with tempfile.TemporaryDirectory() as td:
            cs = CppSearch(A, pos, {"acc": acc}, Path(td))
            fails = 0
            for row in cs.run(range(0, A.shape[0], 10), ["kcore"], 3, theta="acc"):
                H = np.array(row["members"], int)
                if len(H) < 2:
                    continue
                h = dynamics.hk(A[H][:, H], pos[H], acc[H], max_steps=2000)
                fails += (not h["consensus"]) or h["edge_rejections_seen"] > 0
        self.assertGreater(fails, 0)


if __name__ == "__main__":
    unittest.main()
