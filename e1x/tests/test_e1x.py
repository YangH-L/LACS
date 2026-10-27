"""Tests of e1x: the open-group conditions on hand-made cases, the distance bracket against an exact
formula, and the sweep against P0's e1_stress on a random instance (closed hk_eq_G and open w = 1
must equal P0's hk_eq_G and hk_open_G group by group)."""
from __future__ import annotations

import gzip
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "p0", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from e1x.conditions import BOUNDARY, FALSE, TRUE, dist_to_hull, open_conditions  # noqa: E402


def _x(*vals):
    return np.array([[v, 0.0, 0.0] for v in vals])


class TestConditions(unittest.TestCase):
    def test_special_true(self):
        X, th = _x(0.0, 0.3), np.array([1.0, 1.0])
        optr, oX = np.array([0, 1, 2]), _x(0.5, 0.6)
        c = open_conditions(X, th, optr, oX)
        self.assertEqual(c["cond_special"], TRUE)
        self.assertEqual(c["cond_refined"], TRUE)

    def test_refined_true_with_far_rejected_outsider(self):
        X, th = _x(0.0, 0.5), np.array([1.0, 1.0])
        optr, oX = np.array([0, 1, 2]), _x(0.9, 2.5)  # 2.5 rejected by member 2, 1.6 from K0 = [0, 0.9]
        c = open_conditions(X, th, optr, oX)
        self.assertEqual(c["cond_special"], FALSE)
        self.assertEqual(c["cond_a"], TRUE)
        self.assertEqual(c["cond_b"], TRUE)
        self.assertEqual(c["cond_refined"], TRUE)

    def test_b_fails_when_rejected_outsider_is_near_k0(self):
        X, th = _x(0.0, 0.5), np.array([1.0, 1.0])
        optr, oX = np.array([0, 1, 2]), _x(0.9, 1.6)  # rejected (1.1 > 1) but 0.7 from K0
        c = open_conditions(X, th, optr, oX)
        self.assertEqual(c["cond_a"], TRUE)
        self.assertEqual(c["cond_b"], FALSE)
        self.assertEqual(c["cond_refined"], FALSE)

    def test_a_fails(self):
        X, th = _x(0.0, 0.5), np.array([1.0, 0.6])
        optr, oX = np.array([0, 1, 1]), _x(0.9)  # accepted by member 1; diam 0.9 > theta_min 0.6
        c = open_conditions(X, th, optr, oX)
        self.assertEqual(c["cond_a"], FALSE)
        self.assertEqual(c["cond_refined"], FALSE)

    def test_boundary(self):
        X, th = _x(0.0, 0.5), np.array([1.0, 1.0])
        optr, oX = np.array([0, 1, 1]), _x(1.0)  # exactly at the tolerance
        c = open_conditions(X, th, optr, oX)
        self.assertEqual(c["cond_special"], BOUNDARY)
        self.assertEqual(c["cond_refined"], BOUNDARY)

    def test_outsider_shared_by_two_members(self):
        # accepted by member 1, rejected by member 2 (x_o in Y0, so (b) does not apply), diam(K0) <= theta_min
        X, th = _x(0.0, 0.2), np.array([1.0, 0.75])
        optr, oX = np.array([0, 1, 2]), _x(0.6, 0.6)
        c = open_conditions(X, th, optr, oX)
        self.assertEqual(c["cond_a"], TRUE)
        self.assertEqual(c["cond_refined"], TRUE)

    def test_dist_bracket_triangle(self):
        rng = np.random.default_rng(3)
        T = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        def exact(p):  # distance to the triangle in the plane z = 0, lifted
            q = p[:2]
            inside = q[0] >= 0 and q[1] >= 0 and q[0] + q[1] <= 1
            if inside:
                d2 = 0.0
            else:
                d2 = np.inf
                for a, b in ((T[0], T[1]), (T[1], T[2]), (T[2], T[0])):
                    a, b = a[:2], b[:2]
                    t = np.clip(np.dot(q - a, b - a) / np.dot(b - a, b - a), 0, 1)
                    d2 = min(d2, float(np.sum((q - a - t * (b - a)) ** 2)))
            return float(np.sqrt(d2 + p[2] ** 2))

        for _ in range(300):
            p = rng.uniform(-2, 3, size=3)
            lb, ub = dist_to_hull(p, T)
            e = exact(p)
            self.assertLessEqual(lb, e + 1e-9)
            self.assertGreaterEqual(ub, e - 1e-9)
            self.assertLess(ub - lb, 1e-6 * max(1.0, e))


def _write_instance(d: Path, n=160, m=3, seed=0):
    from p0kit.formats import save_npz_instance
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, m)) * np.array([1.0, 1.0, 0.0])
    theta = np.exp(rng.normal(np.log(0.9), 0.3, size=n))
    E = set()
    for i in range(n):
        for j in rng.choice(n, size=6, replace=False):
            if i != j and np.linalg.norm(X[i] - X[j]) < 1.6:
                E.add((min(i, j), max(i, j)))
    E = np.array(sorted(E))
    src = np.r_[E[:, 0], E[:, 1]]
    dst = np.r_[E[:, 1], E[:, 0]]
    o = np.lexsort((dst, src))
    src, dst = src[o], dst[o]
    indptr = np.zeros(n + 1, np.int64)
    np.add.at(indptr, src + 1, 1)
    indptr = np.cumsum(indptr)
    save_npz_instance(str(d / "inst.npz"), indptr, dst.astype(np.int32), X, theta, {"units": "test"})
    # groups: balls around random queries, one "lcc_exact" whose diameter fits, one larger "sac"
    recs = []
    for q in range(0, n, 8):
        dq = np.linalg.norm(X - X[q], axis=1)
        small = np.flatnonzero(dq <= 0.25 * theta.min())
        big = np.flatnonzero(dq <= 0.8)
        for meth, H in (("lcc_exact", small), ("sac", big)):
            recs.append({"dataset": "toy", "setting": "default", "query": int(q), "k": 2, "method": meth,
                         "members": H.tolist() if len(H) else [], "size": int(len(H))})
    with gzip.open(d / "groups.jsonl.gz", "wt") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    # observed weights: random directed weights on the edges
    w = rng.gamma(1.0, size=len(src)) * (rng.random(len(src)) < 0.7)
    np.savez(d / "weights.npz", src=src, dst=dst, w=w)
    return X, theta


class TestSweepAgainstP0(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="e1x_"))
        _write_instance(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_matches_p0_and_theorem(self):
        from p0kit import e1_stress
        from e1x import open_sweep, summarize
        cfg_p0 = {"seed": 5, "max_group": 5000, "ext_max_neighbors": 3,
                  "defaults": {"T_max": 300, "move_tol_rel": 1e-6, "rel_eps": 1e-9, "diam_every": 10,
                               "cons_rel": 1e-3, "cluster_rel": 1e-3},
                  "variants": [{"name": "hk_eq_G", "model": "hk", "graph": "G", "weights": "equal"},
                               {"name": "hk_open_G", "model": "hk", "graph": "G", "weights": "equal", "open": True}]}
        cfg_x = {**{k: v for k, v in cfg_p0.items() if k != "variants"}, "margin": 1e-6,
                 "variants": [{"name": "hk_eq_G", "model": "hk", "graph": "G", "weights": "equal"},
                              {"name": "open_w0.1", "open_w": 0.1}, {"name": "open_w1", "open_w": 1.0},
                              {"name": "hk_obs_G", "observed": True},
                              {"name": "open_obs_G", "observed": True, "open": True}]}
        (self.tmp / "p0.yaml").write_text(yaml.safe_dump(cfg_p0))
        (self.tmp / "x.yaml").write_text(yaml.safe_dump(cfg_x))
        inst, grp = str(self.tmp / "inst.npz"), str(self.tmp / "groups.jsonl.gz")
        e1_stress.main(["--config", str(self.tmp / "p0.yaml"), "--instance", inst, "--groups", grp,
                        "--out", str(self.tmp / "p0"), "--lcc-only"])
        rc = open_sweep.main(["--config", str(self.tmp / "x.yaml"), "--instance", inst, "--groups", grp,
                              "--out", str(self.tmp / "x" / "toy"), "--lcc-only", "--weights",
                              str(self.tmp / "weights.npz"), "--workers", "2"])
        self.assertEqual(rc, 0)
        cmp = summarize.compare_p0(self.tmp / "x" / "toy", self.tmp / "p0" / "e1_rows.jsonl.gz")
        for k, v in cmp.items():
            self.assertGreater(v["common"], 10, k)
            self.assertEqual(v["same_p_broken"], v["common"], k)
            self.assertEqual(v["same_largest_cluster"], v["common"], k)
        S, Fw, C = summarize.summarize([self.tmp / "x" / "toy"])
        self.assertEqual(C["theorem_violations_total"], 0)
        lcc = S[(S["method"] == "lcc_exact") & (S["variant"] == "hk_eq_G")]
        self.assertEqual(float(lcc["share_broken"].iloc[0]), 0.0)  # every toy LCC satisfies (iii)
        # resume: a second run does nothing and keeps the rows
        rc = open_sweep.main(["--config", str(self.tmp / "x.yaml"), "--instance", inst, "--groups", grp,
                              "--out", str(self.tmp / "x" / "toy"), "--lcc-only", "--weights",
                              str(self.tmp / "weights.npz")])
        S2, _, _ = summarize.summarize([self.tmp / "x" / "toy"])
        self.assertEqual(len(S2), len(S))
        self.assertTrue((S2["groups"].to_numpy() == S["groups"].to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
