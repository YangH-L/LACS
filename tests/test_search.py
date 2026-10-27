"""C++ engine vs. brute force / reference implementations on random small graphs.

Run: python -m unittest tests.test_search -v
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lacs.graph.io import edges_to_csr  # noqa: E402
from lacs.search import reference as ref  # noqa: E402
from lacs.search.cpp import CppSearch  # noqa: E402


def random_instance(rng, n, p, dim=2, theta_scale=1.0):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
    pos = rng.normal(size=(n, dim)).astype(np.float32)
    theta = (theta_scale * rng.gamma(4.0, 0.35, size=n)).astype(np.float32)
    return edges, pos, theta


class TestAgainstBruteForce(unittest.TestCase):
    """Exact algorithms must match exhaustive enumeration; approximations must
    satisfy their stated guarantees."""

    def test_random_small(self):
        rng = np.random.default_rng(7)
        checked = 0
        with tempfile.TemporaryDirectory() as td:
            for trial in range(160):
                n = int(rng.integers(8, 15))
                edges, pos, theta = random_instance(rng, n, p=float(rng.uniform(0.25, 0.6)),
                                                    dim=int(rng.integers(1, 4)),
                                                    theta_scale=float(rng.uniform(0.5, 2.0)))
                adj = ref.adjacency_sets(n, edges)
                k = int(rng.integers(1, 4))
                us = np.array([e[0] for e in edges], dtype=np.int64)
                vs = np.array([e[1] for e in edges], dtype=np.int64)
                A = edges_to_csr(n, us, vs)
                cs = CppSearch(A, pos, {"t": theta}, Path(td) / f"t{trial}")
                kr = float(np.median(theta))
                queries = [q for q in range(n) if ref.kcore_component(adj, set(range(n)), k, q)]
                if not queries:
                    continue
                queries = queries[:4]
                algos = ["exact", "ballcore_r", "ballcore_f", "greedy", "kcore", "edgeacc",
                         "pairwise_exact", "pairwise_greedy", "kr_exact", "kr_greedy"]
                rows = cs.run(queries, algos, k, theta="t", r=kr, time_limit=30)
                basic = cs.run(queries, ["exact"], k, theta="t", time_limit=30, tag="basic",
                               extra=["--no-ub-order", "--no-color-bound", "--no-seed", "--no-compat-bound", "--no-edge-filter"])
                res = {(row["q"], row["algo"]): row for row in rows}
                for row in basic:
                    res[(row["q"], "exact_basic")] = row
                for q in queries:
                    opt, _ = ref.brute_force(adj, pos, theta, k, q, "diam")
                    ex = res[(q, "exact")]
                    self.assertEqual(ex["status"], "optimal" if opt else ex["status"])
                    self.assertEqual(ex["size"], opt, msg=f"exact size trial={trial} q={q}")
                    self.assertEqual(res[(q, "exact_basic")]["size"], opt, msg="basic exact size")
                    if opt:
                        self.assertTrue(ref.feasible(adj, pos, theta, set(ex["members"]), k, q, "diam"))
                    # ballcore_r: size >= OPT, diameter <= 2 * min theta, structure valid
                    br = res[(q, "ballcore_r")]
                    self.assertGreaterEqual(br["size"], opt)
                    if br["size"]:
                        H = set(br["members"])
                        self.assertTrue(ref.is_kcore_component(adj, H, k, q))
                        self.assertLessEqual(ref.diameter(pos, H), 2 * min(theta[v] for v in H) + 1e-5)
                    ref_r, _ = ref.ballcore(adj, pos, theta, k, q, half=False)
                    self.assertEqual(br["size"], len(ref_r))
                    # ballcore_f and greedy: feasible, never larger than OPT
                    for algo in ("ballcore_f", "greedy"):
                        row = res[(q, algo)]
                        self.assertLessEqual(row["size"], opt)
                        if row["size"]:
                            self.assertTrue(ref.feasible(adj, pos, theta, set(row["members"]), k, q, "diam"),
                                            msg=f"{algo} infeasible")
                    ref_f, _ = ref.ballcore(adj, pos, theta, k, q, half=True)
                    self.assertEqual(res[(q, "ballcore_f")]["size"], len(ref_f))
                    # structure-only and edge-filter baselines
                    self.assertEqual(set(res[(q, "kcore")]["members"]),
                                     ref.kcore_component(adj, set(range(n)), k, q))
                    self.assertEqual(set(res[(q, "edgeacc")]["members"]), ref.edgeacc(adj, pos, theta, k, q))
                    # pairwise and global-threshold variants
                    opt_p, _ = ref.brute_force(adj, pos, theta, k, q, "pairwise")
                    self.assertEqual(res[(q, "pairwise_exact")]["size"], opt_p)
                    self.assertLessEqual(res[(q, "pairwise_greedy")]["size"], opt_p)
                    opt_k, _ = ref.brute_force(adj, pos, theta, k, q, "kr", r=kr)
                    self.assertEqual(res[(q, "kr_exact")]["size"], opt_k)
                    self.assertLessEqual(res[(q, "kr_greedy")]["size"], opt_k)
                    checked += 1
        self.assertGreater(checked, 100)


class TestLargerConsistency(unittest.TestCase):
    """On mid-size graphs, pruned and unpruned exact search must agree."""

    def test_pruning_does_not_change_optimum(self):
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as td:
            n = 120
            # planted groups so that non-trivial communities exist
            groups = rng.integers(0, 4, size=n)
            centers = rng.normal(scale=2.0, size=(4, 3))
            pos = (centers[groups] + 0.5 * rng.normal(size=(n, 3))).astype(np.float32)
            edges = [(i, j) for i in range(n) for j in range(i + 1, n)
                     if rng.random() < (0.25 if groups[i] == groups[j] else 0.03)]
            theta = rng.gamma(6.0, 0.35, size=n).astype(np.float32)
            us = np.array([e[0] for e in edges]); vs = np.array([e[1] for e in edges])
            A = edges_to_csr(n, us, vs)
            adj = ref.adjacency_sets(n, edges)
            cs = CppSearch(A, pos, {"t": theta}, Path(td))
            queries = list(range(0, n, 6))
            fast = {r["q"]: r for r in cs.run(queries, ["exact"], 3, theta="t", time_limit=60)}
            slow = {r["q"]: r for r in cs.run(queries, ["exact"], 3, theta="t", time_limit=60, tag="slow",
                                              extra=["--no-ub-order", "--no-color-bound", "--no-seed", "--no-compat-bound", "--no-edge-filter"])}
            for q in queries:
                a, b = fast[q], slow[q]
                if a["status"] == "optimal" and b["status"] == "optimal":
                    self.assertEqual(a["size"], b["size"], msg=f"q={q}")
                if a["size"]:
                    self.assertTrue(ref.feasible(adj, pos, theta, set(a["members"]), 3, q, "diam"))


MODES = {"default": [], "no_index": ["--no-index"], "no_local": ["--no-local"],
         "neither": ["--no-index", "--no-local"]}


class TestCandidateModes(unittest.TestCase):
    """The compatibility index and query-local candidates are reductions that
    must never change an exact answer (v0.3)."""

    def test_modes_match_brute_force(self):
        rng = np.random.default_rng(23)
        checked = 0
        with tempfile.TemporaryDirectory() as td:
            for trial in range(90):
                n = int(rng.integers(8, 15))
                edges, pos, theta = random_instance(rng, n, p=float(rng.uniform(0.25, 0.6)),
                                                    dim=int(rng.integers(1, 4)),
                                                    theta_scale=float(rng.uniform(0.5, 2.5)))
                adj = ref.adjacency_sets(n, edges)
                k = int(rng.integers(1, 4))
                A = edges_to_csr(n, np.array([e[0] for e in edges], dtype=np.int64),
                                 np.array([e[1] for e in edges], dtype=np.int64))
                cs = CppSearch(A, pos, {"t": theta}, Path(td) / f"m{trial}")
                kr = float(np.median(theta))
                queries = [q for q in range(n) if ref.kcore_component(adj, set(range(n)), k, q)][:4]
                if not queries:
                    continue
                algos = ["exact", "ballcore_r", "ballcore_f", "greedy", "edgeacc", "pairwise_exact",
                         "pairwise_greedy", "kr_exact", "kr_greedy"]
                for mode, flags in MODES.items():
                    rows = cs.run(queries, algos, k, theta="t", r=kr, time_limit=30, extra=flags, tag=mode)
                    res = {(row["q"], row["algo"]): row for row in rows}
                    for q in queries:
                        msg = f"mode={mode} trial={trial} q={q}"
                        opt, _ = ref.brute_force(adj, pos, theta, k, q, "diam")
                        self.assertEqual(res[(q, "exact")]["size"], opt, msg=msg)
                        self.assertEqual(res[(q, "ballcore_r")]["size"],
                                         len(ref.ballcore(adj, pos, theta, k, q, half=False)[0]), msg=msg)
                        self.assertEqual(res[(q, "ballcore_f")]["size"],
                                         len(ref.ballcore(adj, pos, theta, k, q, half=True)[0]), msg=msg)
                        self.assertEqual(set(res[(q, "edgeacc")]["members"]), ref.edgeacc(adj, pos, theta, k, q),
                                         msg=msg)
                        self.assertEqual(res[(q, "pairwise_exact")]["size"],
                                         ref.brute_force(adj, pos, theta, k, q, "pairwise")[0], msg=msg)
                        self.assertEqual(res[(q, "kr_exact")]["size"],
                                         ref.brute_force(adj, pos, theta, k, q, "kr", r=kr)[0], msg=msg)
                        g = res[(q, "greedy")]
                        self.assertLessEqual(g["size"], opt, msg=msg)
                        if g["size"]:
                            self.assertTrue(ref.feasible(adj, pos, theta, set(g["members"]), k, q, "diam"), msg=msg)
                        checked += 1
        self.assertGreater(checked, 200)

    def test_modes_agree_on_planted_graph(self):
        rng = np.random.default_rng(29)
        n = 400
        groups = rng.integers(0, 6, size=n)
        centers = rng.normal(scale=2.5, size=(6, 3))
        pos = (centers[groups] + 0.5 * rng.normal(size=(n, 3))).astype(np.float32)
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)
                 if rng.random() < (0.12 if groups[i] == groups[j] else 0.01)]
        theta = rng.gamma(6.0, 0.3, size=n).astype(np.float32)
        A = edges_to_csr(n, np.array([e[0] for e in edges]), np.array([e[1] for e in edges]))
        adj = ref.adjacency_sets(n, edges)
        with tempfile.TemporaryDirectory() as td:
            cs = CppSearch(A, pos, {"t": theta}, Path(td))
            queries = list(range(0, n, 9))
            out = {mode: {r["q"]: r for r in cs.run(queries, ["exact", "ballcore_r", "ballcore_f"], 3, theta="t",
                                                     time_limit=60, extra=flags, tag=mode)
                          if r["algo"] == "exact"}
                   for mode, flags in MODES.items()}
            nonempty = 0
            for q in queries:
                sizes = {m: out[m][q]["size"] for m in MODES if out[m][q]["status"] == "optimal"}
                self.assertLessEqual(len(set(sizes.values())), 1, msg=f"q={q} {sizes}")
                a = out["default"][q]
                if a["size"]:
                    nonempty += 1
                    self.assertTrue(ref.feasible(adj, pos, theta, set(a["members"]), 3, q, "diam"))
            self.assertGreater(nonempty, 5)


class TestParallelRuns(unittest.TestCase):
    def test_n_proc_gives_identical_results(self):
        rng = np.random.default_rng(13)
        edges, pos, theta = random_instance(rng, 60, 0.15, dim=3)
        A = edges_to_csr(60, np.array([e[0] for e in edges]), np.array([e[1] for e in edges]))
        with tempfile.TemporaryDirectory() as td:
            cs = CppSearch(A, pos, {"t": theta}, Path(td))
            qs = list(range(0, 60, 2))
            one = cs.run(qs, ["exact", "greedy"], 2, theta="t", tag="one")
            many = cs.run(qs, ["exact", "greedy"], 2, theta="t", tag="many", n_proc=3)
            strip = lambda rows: [(r["q"], r["algo"], r["size"], tuple(r["members"])) for r in rows]  # noqa: E731
            self.assertEqual(strip(one), strip(many))


class TestSpatialBaselines(unittest.TestCase):
    """sac_inc (SAC search, AppInc) against brute force, and per-query thresholds for kr_*."""

    def test_sac_inc_matches_brute_force(self):
        rng = np.random.default_rng(21)
        checked = 0
        with tempfile.TemporaryDirectory() as td:
            for trial in range(60):
                n = int(rng.integers(8, 30))
                edges, pos, theta = random_instance(rng, n, p=float(rng.uniform(0.15, 0.5)), dim=2)
                adj = ref.adjacency_sets(n, edges)
                k = int(rng.integers(1, 4))
                A = edges_to_csr(n, np.array([e[0] for e in edges], dtype=np.int64),
                                 np.array([e[1] for e in edges], dtype=np.int64))
                queries = [q for q in range(n) if ref.kcore_component(adj, set(range(n)), k, q)][:5]
                if not queries:
                    continue
                cs = CppSearch(A, pos, {"t": theta}, Path(td) / f"s{trial}")
                rows = {r["q"]: r for r in cs.run(queries, ["sac_inc"], k, theta="t", time_limit=30)}
                for q in queries:
                    d = sorted((ref.dist(pos, q, v), v) for v in range(n))
                    want, radius = set(), None
                    for rad, _ in d:  # smallest ball around q that holds a connected k-core with q
                        ball = {v for dv, v in d if dv <= rad}
                        H = ref.kcore_component(adj, ball, k, q)
                        if H:
                            want, radius = H, rad
                            break
                    got = rows[q]
                    self.assertEqual(set(got["members"]), want, msg=f"trial={trial} q={q}")
                    self.assertAlmostEqual(got["tau"], radius, places=4)
                    checked += 1
                # size-matched variant: smallest ball whose k-core with q has >= s members
                sizes = [int(rng.integers(0, n + 2)) for _ in queries]
                rows_s = {r["q"]: r for r in cs.run(queries, ["sac_inc"], k, theta="t", time_limit=30,
                                                     size_per_query=sizes, tag="sized")}
                for q, smin in zip(queries, sizes):
                    d = sorted((ref.dist(pos, q, v), v) for v in range(n))
                    want = None
                    for rad, _ in d:
                        H = ref.kcore_component(adj, {v for dv, v in d if dv <= rad}, k, q)
                        if H and len(H) >= smin:
                            want = H
                            break
                    if want is None:  # never large enough: the whole k-core component
                        want = ref.kcore_component(adj, set(range(n)), k, q)
                    self.assertEqual(set(rows_s[q]["members"]), want, msg=f"sized trial={trial} q={q} s={smin}")
        self.assertGreater(checked, 50)

    def test_per_query_r_equals_separate_runs(self):
        rng = np.random.default_rng(5)
        edges, pos, theta = random_instance(rng, 40, 0.2, dim=2)
        A = edges_to_csr(40, np.array([e[0] for e in edges]), np.array([e[1] for e in edges]))
        with tempfile.TemporaryDirectory() as td:
            cs = CppSearch(A, pos, {"t": theta}, Path(td))
            qs = list(range(0, 40, 3))
            rs = [float(x) for x in rng.uniform(0.5, 2.5, size=len(qs))]
            joint = cs.run(qs, ["kr_exact", "kr_greedy"], 2, theta="t", r_per_query=rs, tag="joint", n_proc=2)
            sep = []
            for i, (q, r) in enumerate(zip(qs, rs)):
                sep += cs.run([q], ["kr_exact", "kr_greedy"], 2, theta="t", r=r, tag=f"sep{i}")
            strip = lambda rows: [(r["q"], r["algo"], r["size"], tuple(r["members"])) for r in rows]  # noqa: E731
            self.assertEqual(strip(joint), strip(sep))


if __name__ == "__main__":
    unittest.main()
