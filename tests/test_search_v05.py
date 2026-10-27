"""Engine v0.5 (W2): best-first threshold order, colouring order, upper bound when optimality is
not proven, and the two-pass converter.

Run: python -m unittest tests.test_search_v05 -v
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lacs.graph.io import edges_to_csr  # noqa: E402
from lacs.search import reference as ref  # noqa: E402
from lacs.search.cpp import BUILD_DIR, CppSearch, ensure_built  # noqa: E402

ORDERS = {
    "bound+degree": [],
    "count+degree": ["--tau-order", "count"],
    "bound+index": ["--color-order", "index"],
    "count+index": ["--tau-order", "count", "--color-order", "index"],  # the v0.4 algorithm
}


def random_instance(rng, n, p, dim=2, theta_scale=1.0):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
    pos = rng.normal(size=(n, dim)).astype(np.float32)
    theta = (theta_scale * rng.gamma(4.0, 0.35, size=n)).astype(np.float32)
    return edges, pos, theta


def planted_instance(rng, n_comm=12, size=(40, 90), deg_in=14.0, deg_out=3.0, dim=8, scale=1.5, sigma=0.4):
    """Planted communities with community-aligned positions (as gen_synth / Section 6.1): exact
    search needs real branching here, so small time limits interrupt it."""
    sizes = rng.integers(size[0], size[1] + 1, size=n_comm)
    comm = np.repeat(np.arange(n_comm), sizes)
    n = len(comm)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    us, vs = [], []
    for c in range(n_comm):
        s0, sz = starts[c], sizes[c]
        iu = np.triu_indices(sz, 1)
        mask = rng.random(len(iu[0])) < min(1.0, deg_in / (sz - 1))
        us.append(iu[0][mask] + s0)
        vs.append(iu[1][mask] + s0)
    m_out = int(n * deg_out / 2)
    us.append(rng.integers(0, n, size=m_out))
    vs.append(rng.integers(0, n, size=m_out))
    u, v = np.concatenate(us), np.concatenate(vs)
    centres = rng.normal(scale=8.0, size=(n_comm, dim))
    X = (centres[comm] + rng.normal(size=(n, dim))).astype(np.float32)
    theta = (scale * np.sqrt(2 * dim) * np.exp(sigma * rng.normal(size=n) - sigma ** 2 / 2)).astype(np.float32)
    return n, u, v, X, theta


class TestOrdersAgainstBruteForce(unittest.TestCase):
    def test_all_orders_match_brute_force(self):
        rng = np.random.default_rng(2026)
        checked = 0
        with tempfile.TemporaryDirectory() as td:
            for trial in range(120):
                n = int(rng.integers(8, 15))
                edges, pos, theta = random_instance(rng, n, p=float(rng.uniform(0.25, 0.6)),
                                                    dim=int(rng.integers(1, 4)),
                                                    theta_scale=float(rng.uniform(0.5, 2.0)))
                adj = ref.adjacency_sets(n, edges)
                k = int(rng.integers(1, 4))
                A = edges_to_csr(n, np.array([e[0] for e in edges], dtype=np.int64),
                                 np.array([e[1] for e in edges], dtype=np.int64))
                cs = CppSearch(A, pos, {"t": theta}, Path(td) / f"t{trial}")
                queries = [q for q in range(n) if ref.kcore_component(adj, set(range(n)), k, q)][:4]
                if not queries:
                    continue
                br = {r["q"]: r for r in cs.run(queries, ["ballcore_r"], k, theta="t", time_limit=30, tag="br")}
                for name, flags in ORDERS.items():
                    rows = cs.run(queries, ["exact"], k, theta="t", time_limit=30, extra=flags, tag=name.replace("+", "_"))
                    for row in rows:
                        q = row["q"]
                        opt, _ = ref.brute_force(adj, pos, theta, k, q, "diam")
                        self.assertEqual(row["status"], "optimal", msg=f"{name} trial={trial} q={q}")
                        self.assertEqual(row["size"], opt, msg=f"{name} trial={trial} q={q}")
                        self.assertEqual(row["ub"], opt, msg=f"{name}: ub of an optimal answer")
                        if opt:
                            self.assertTrue(ref.feasible(adj, pos, theta, set(row["members"]), k, q, "diam"))
                        checked += 1
                for q, row in br.items():  # BallCore-R reports |B| as its bound, and |B| >= OPT
                    self.assertEqual(row["ub"], row["size"])
        self.assertGreater(checked, 300)


class TestUpperBoundUnderTimeouts(unittest.TestCase):
    """Interrupted searches: the answer is feasible and no larger than OPT, and ub >= OPT."""

    def test_bound_is_sound(self):
        rng = np.random.default_rng(11)
        with tempfile.TemporaryDirectory() as td:
            seen_open = seen_fixed = 0
            for inst in range(3):
                n, u, v, X, theta = planted_instance(rng)
                A = edges_to_csr(n, u, v)
                cs = CppSearch(A, X, {"t": theta}, Path(td) / f"i{inst}")
                core = cs.core_numbers()
                qs = rng.choice(np.flatnonzero(core >= 3), size=12, replace=False)
                full = {r["q"]: r for r in cs.run(qs, ["exact"], 3, theta="t", time_limit=120, members=False,
                                                  tag="full")}
                for r in full.values():
                    self.assertEqual(r["status"], "optimal")
                opt = {q: r["size"] for q, r in full.items()}
                for name, flags in ORDERS.items():
                    for tl in (0.00001, 0.0005, 0.003, 0.02):
                        tag = f"{name.replace('+', '_')}_{tl}"
                        for r in cs.run(qs, ["exact"], 3, theta="t", time_limit=tl, members=False, extra=flags,
                                        tag=tag):
                            q = r["q"]
                            self.assertLessEqual(r["size"], opt[q], msg=tag)
                            if r["status"] == "optimal":
                                self.assertEqual(r["size"], opt[q], msg=tag)
                                self.assertEqual(r["ub"], opt[q], msg=tag)
                            else:
                                self.assertEqual(r["status"], "timeout", msg=tag)
                                self.assertGreaterEqual(r["ub"], opt[q], msg=f"{tag} q={q}")
                                seen_open += 1
                    # candidate sets too large for branch and bound: status too_large, ub >= OPT
                    for r in cs.run(qs, ["exact"], 3, theta="t", time_limit=120, members=False,
                                    extra=flags + ["--max-exact-size", "20"], tag=f"{name}_cap"):
                        q = r["q"]
                        self.assertLessEqual(r["size"], opt[q])
                        if r["status"] != "optimal":
                            self.assertEqual(r["status"], "too_large")
                            self.assertGreaterEqual(r["ub"], opt[q])
                            seen_open += 1
                        else:
                            self.assertEqual(r["size"], opt[q])
                # the fixed-threshold solvers as well
                kr = float(np.median(theta))
                for algo in ("kr_exact", "pairwise_exact"):
                    full_t = {r["q"]: r for r in cs.run(qs, [algo], 3, theta="t", r=kr, time_limit=120,
                                                        members=False, tag=f"{algo}_full")}
                    for r in cs.run(qs, [algo], 3, theta="t", r=kr, time_limit=0.00001, members=False,
                                    tag=f"{algo}_short"):
                        o = full_t[r["q"]]
                        self.assertIn(o["status"], ("optimal", "empty"))  # empty: no community, OPT = 0
                        self.assertEqual(o["ub"], o["size"])
                        self.assertLessEqual(r["size"], o["size"])
                        self.assertGreaterEqual(r["ub"], o["size"])
                        seen_fixed += int(r["status"] == "timeout")
            print(f"\n  interrupted exact searches checked: {seen_open}; fixed-threshold: {seen_fixed}")
            self.assertGreater(seen_open, 0, "no interrupted search: time limits too generous for this machine")


class TestBestFirstAgreesWithCountOrder(unittest.TestCase):
    def test_same_optima_on_planted_graphs(self):
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as td:
            for inst in range(2):
                n, u, v, X, theta = planted_instance(rng, n_comm=10, scale=1.25)
                cs = CppSearch(edges_to_csr(n, u, v), X, {"t": theta}, Path(td) / f"i{inst}")
                qs = rng.choice(np.flatnonzero(cs.core_numbers() >= 3), size=15, replace=False)
                res = {}
                for name, flags in ORDERS.items():
                    res[name] = {r["q"]: r for r in cs.run(qs, ["exact"], 3, theta="t", time_limit=120,
                                                           members=False, extra=flags, tag=name.replace("+", "_"))}
                for q in qs:
                    sizes = {name: res[name][q]["size"] for name in ORDERS}
                    self.assertEqual(len(set(sizes.values())), 1, msg=f"q={q}: {sizes}")
                    for name in ORDERS:
                        self.assertEqual(res[name][q]["status"], "optimal")


class TestLowMemConverter(unittest.TestCase):
    def _convert(self, path, out, relabel=None, lowmem=False):
        cmd = [str(BUILD_DIR / "lacs_convert"), "--in", str(path), "--out", str(out)]
        if relabel:
            cmd += ["--relabel", str(relabel)]
        if lowmem:
            cmd.append("--lowmem")
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return json.loads(res.stdout.strip().splitlines()[-1])

    def test_identical_output(self):
        ensure_built()
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            for trial in range(25):
                n_ids = int(rng.integers(5, 400))
                ids = np.sort(rng.choice(5000, size=n_ids, replace=False)) if trial % 2 else np.arange(n_ids)
                m = int(rng.integers(0, 3000))
                a = rng.choice(ids, size=m)
                b = rng.choice(ids, size=m)
                lines = ["# comment line", "% another comment"]
                for x, y in zip(a, b):
                    sep = rng.choice([" ", "\t", ",", "  "])
                    extra = rng.choice(["", " 1", "\t0.5 x"])
                    lines.append(f"{x}{sep}{y}{extra}")
                    if rng.random() < 0.1:
                        lines.append(f"{y} {x}")  # reverse duplicate
                    if rng.random() < 0.05:
                        lines.append(f"{x} {x}")  # self loop
                path = td / f"e{trial}.txt"
                path.write_text("\n".join(lines) + ("\n" if rng.random() < 0.5 else ""))
                for relabel in (False, True):
                    o1, o2 = td / f"a{trial}{relabel}.bin", td / f"b{trial}{relabel}.bin"
                    r1, r2 = (td / f"a{trial}.ids", td / f"b{trial}.ids") if relabel else (None, None)
                    i1 = self._convert(path, o1, r1)
                    i2 = self._convert(path, o2, r2, lowmem=True)
                    self.assertEqual((i1["n"], i1["m"]), (i2["n"], i2["m"]))
                    self.assertEqual(o1.read_bytes(), o2.read_bytes(), msg=f"trial {trial} relabel={relabel}")
                    if relabel:
                        self.assertEqual(r1.read_text(), r2.read_text())


if __name__ == "__main__":
    unittest.main()
