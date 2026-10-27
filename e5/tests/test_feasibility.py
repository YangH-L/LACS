"""Self-test of e5/yelp_feasibility.py on synthetic Yelp-format data (e5/make_fixture.py).

The main numbers are recomputed here by a separate, deliberately plain implementation that reads
the raw JSON lines directly (no shared code with the checker): taste-ready users, graph edges,
eligible queries, the friend-pair co-review rate, and the S1 / G acceptance shares. Also: unit
tests of the helpers, monotonicity of exact LCC existence in the tolerance (S3 <= S1), reuse of a
finished extraction, and that no user / business / review id appears in any output file.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e5 import make_fixture  # noqa: E402
from e5 import yelp_feasibility as yf  # noqa: E402

ARGS = ["--metro-min-business", "100", "--vocab-min-business", "5", "--pairs-random", "20000", "--pairs-fof",
        "20000", "--engine-queries", "120", "--exact-time", "2", "--n-proc", "2", "--budget-exact", "300",
        "--budget-greedy", "300"]


def _cats(c):
    return [s.strip() for s in c.split(",") if s.strip()] if isinstance(c, str) else []


def _friends(fr):
    if isinstance(fr, list):
        return [x.strip() for x in fr if x.strip() and x.strip() != "None"]
    if not isinstance(fr, str) or fr.strip() in ("", "None"):
        return []
    return [x.strip() for x in fr.split(",") if x.strip() and x.strip() != "None"]


def _day(s):
    try:
        return int(np.datetime64(s[:10], "D").astype(np.int64))
    except ValueError:
        return None


def reference(raw: Path, split: str, metro_id: dict, min_train=10, min_taste=5, vocab_min=5):
    """Plain re-implementation of the per-split quantities that the checker reports."""
    biz = {}
    for line in open(raw / "yelp_academic_dataset_business.json"):
        o = json.loads(line)
        biz[o["business_id"]] = (o["state"], _cats(o["categories"]), o["latitude"], o["longitude"])
    users, listed = [], {}
    for line in open(raw / "yelp_academic_dataset_user.json"):
        o = json.loads(line)
        users.append(o["user_id"])
        listed[o["user_id"]] = _friends(o["friends"])
    uset = set(users)
    revs = []
    for line in open(raw / "yelp_academic_dataset_review.json"):
        o = json.loads(line)
        d = _day(o["date"])
        if o["user_id"] in uset and o["business_id"] in biz and d is not None:
            revs.append((o["user_id"], o["business_id"], d))
    if split == "precovid":
        t1, t2, t3 = _day("2017-12-31"), _day("2018-12-31"), _day("2019-12-31")
    else:
        ds = sorted(r[2] for r in revs)
        t1, t2, t3 = ds[math.floor(0.7 * (len(ds) - 1))], ds[math.floor(0.8 * (len(ds) - 1))], ds[-1]

    def period(d):
        return 0 if d <= t1 else 1 if d <= t2 else 2 if d <= t3 else 3

    train = defaultdict(list)
    test = defaultdict(set)
    for u, b, d in revs:
        p = period(d)
        if p == 0:
            train[u].append(b)
        elif p == 2:
            test[u].add(b)
    M = len(metro_id)

    def mid(state):
        return metro_id.get(state, -1)

    home = {}
    for u, bs in train.items():
        c = Counter(mid(biz[b][0]) for b in bs)
        best = sorted(c.items(), key=lambda kv: (-kv[1], kv[0] if kv[0] >= 0 else M))[0][0]
        home[u] = best
    active = {u for u, bs in train.items() if len(bs) >= min_train and home[u] >= 0}
    used = {b for u in active for b in train[u]}
    cc = Counter()
    for b in used:
        cc.update(set(biz[b][1]))
    vocab = sorted(c for c, k in cc.items() if k >= vocab_min)
    col = {c: j for j, c in enumerate(vocab)}

    def vec(b):
        js = sorted({col[c] for c in biz[b][1] if c in col})
        v = np.zeros(len(vocab))
        if js:
            v[js] = 1 / math.sqrt(len(js))
        return v, bool(js)

    ready = []
    cen, th1 = {}, {}
    for u in sorted(active):
        vs = [vec(b) for b in train[u]]
        vs = [v for v, ok in vs if ok]
        if len(vs) >= min_taste:
            ready.append(u)
            c = np.mean(vs, axis=0)
            cen[u] = c
            th1[u] = float(np.median([np.linalg.norm(v - c) for v in vs]))
    rset = set(ready)
    edges = set()
    for u in users:
        for v in listed[u]:
            if v in uset and v != u and u in rset and v in rset and home[u] == home[v]:
                edges.add((min(u, v), max(u, v)))
    nb = defaultdict(set)
    for a, b in edges:
        nb[a].add(b)
        nb[b].add(a)
    core = {u: 0 for u in ready}  # naive peeling: core number = largest k with u in the k-core
    alive = set(ready)
    k = 0
    while alive:
        k += 1
        changed = True
        while changed:
            changed = False
            for u in list(alive):
                if len(nb[u] & alive) < k:
                    alive.remove(u)
                    core[u] = k - 1
                    changed = True
    tact = {u for u in ready if test[u]}
    elig3 = sum(1 for u in ready if core[u] >= 3 and u in tact)
    fp = [(a, b) for a, b in edges if a in tact and b in tact]
    hits = sum(1 for a, b in fp if test[a] & test[b])
    hits_all = sum(1 for a, b in edges if test[a] & test[b])  # inactive users have no test reviews
    within = [np.linalg.norm(cen[a] - cen[b]) <= min(th1[a], th1[b]) for a, b in edges]
    # G: home cell and median distance, as in geo.py
    hl, thg = {}, {}
    for u in ready:
        pts = [(biz[b][2], biz[b][3]) for b in train[u]]
        cells = Counter(math.floor(la / 0.25) * 100000 + math.floor(lo / 0.25) for la, lo in pts)
        best = sorted(cells.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        inm = [(la, lo) for la, lo in pts if math.floor(la / 0.25) * 100000 + math.floor(lo / 0.25) == best]
        h = (float(np.median([p[0] for p in inm])), float(np.median([p[1] for p in inm])))
        hl[u] = yf.to_xyz(np.array([h[0]]), np.array([h[1]]))[0]
        dist = [float(yf.haversine_km(h[0], h[1], la, lo)) for la, lo in pts]
        thg[u] = max(float(np.median(dist)), 1.0)
    within_g = [np.linalg.norm(hl[a] - hl[b]) <= min(thg[a], thg[b]) for a, b in edges]
    return {"taste_ready": len(ready), "edges": len(edges), "eligible_k3": elig3, "friend_pairs_n": len(fp),
            "friend_pairs_hits": hits, "friend_pairs_all_hits": hits_all, "S1_within": float(np.mean(within)), "G_within": float(np.mean(within_g)),
            "n_edges_for_share": len(edges)}


class TestHelpers(unittest.TestCase):
    def test_group_quantile_matches_numpy(self):
        rng = np.random.default_rng(1)
        g = rng.integers(0, 50, 2000)
        x = rng.random(2000)
        for q in (0.25, 0.5, 0.9):
            got = yf.group_quantile(g, x, q, 55)
            for i in range(55):
                if (g == i).any():
                    self.assertAlmostEqual(got[i], float(np.quantile(x[g == i], q)), places=12)
                else:
                    self.assertTrue(np.isnan(got[i]))

    def test_core_numbers_match_peeling(self):
        rng = np.random.default_rng(2)
        for _ in range(20):
            n = int(rng.integers(5, 60))
            m = int(rng.integers(0, 4 * n))
            A = yf.edges_to_csr(n, rng.integers(0, n, m), rng.integers(0, n, m))
            ref = np.zeros(n, int)
            alive = set(range(n))
            nbrs = [set(A.indices[A.indptr[i]:A.indptr[i + 1]].tolist()) for i in range(n)]
            k = 0
            while alive:
                k += 1
                ch = True
                while ch:
                    ch = False
                    for u in list(alive):
                        if len(nbrs[u] & alive) < k:
                            alive.discard(u)
                            ref[u] = k - 1
                            ch = True
            self.assertTrue(np.array_equal(yf.core_numbers(A), ref))

    def test_friend_lists_and_kinds(self):
        self.assertEqual(yf._friend_list("None"), [])
        self.assertEqual(yf._friend_list("a, b,c ,None"), ["a", "b", "c"])
        self.assertEqual(yf._friend_list(["a", " b"]), ["a", "b"])
        self.assertEqual(yf._friend_list(None), [])
        self.assertEqual(yf.kind_of("yelp_dataset/yelp_academic_dataset_review.json"), "review")
        self.assertIsNone(yf.kind_of("._yelp_academic_dataset_review.json.part"))

    def test_plurality_ties(self):
        u = np.array([0, 0, 1, 1, 2, 2, 2])
        m = np.array([-1, 1, 2, 1, -1, -1, 0])
        home, share = yf.plurality(u, m, 4, 3)
        self.assertEqual(home.tolist(), [1, 1, -1, -2])  # metro beats outside on ties; lower id on ties
        self.assertAlmostEqual(share[2], 2 / 3)


FAKE_ENGINE = r'''#!/usr/bin/env python3
import json, os, sys, time
a = sys.argv
q = a[a.index("--queries") + 1]; out = a[a.index("--out") + 1]
algos = a[a.index("--algo") + 1].split(",")
mode = os.environ.get("FAKE_MODE", "ok")
qs = [int(x) for x in open(q).read().split()]
if mode == "crash0_sleep1" and q.endswith("_0.txt"):
    sys.stderr.write("error: std::bad_alloc\n"); sys.exit(1)
with open(out, "w") as f:
    for i, x in enumerate(qs):
        if mode in ("sleep_all", "crash0_sleep1") and i == 1:
            f.write('{"q": %d, "algo": "kco' % x); f.flush(); time.sleep(60)
        for al in algos:
            f.write(json.dumps({"q": x, "algo": al, "size": 5, "status": "optimal" if al == "exact" else "ok",
                                "ub": 5, "time_ms": 1.0}) + "\n")
        f.flush()
'''


class TestEngineSummaries(unittest.TestCase):
    @staticmethod
    def _rows(spec):  # spec: list of (status, size, ub, kcore size); status 'budget' = query never reached
        rows = []
        for q, (st, size, ub, ks) in enumerate(spec):
            rows.append({"q": q, "algo": "kcore", "size": ks, "status": "ok", "ub": -1, "time_ms": 0.1})
            if st != "budget":
                rows.append({"q": q, "algo": "exact", "size": size, "status": st, "ub": ub, "time_ms": 1.0})
        return list(range(len(spec))), rows

    def test_unknown_queries_do_not_count_as_found(self):  # review finding B1
        qq, rows = self._rows([("optimal", 0, 0, 200)] * 255 + [("timeout", 0, 4, 200)] * 45)
        x, _ = yf.summarize_exact(qq, rows)
        self.assertEqual(x["share_found"], 0.0)
        self.assertAlmostEqual(x["share_not_proven_empty"], 0.15)
        self.assertAlmostEqual(x["share_proven"], 0.85)
        dec = yf.choose_spec({s: {"exact": x} for s in ("S1", "S2", "S3", "G")}, 4000)
        self.assertIsNone(dec["spec"])
        self.assertEqual(dec["outcome"], "infeasible as designed")
        self.assertFalse(dec["checks"]["S1"]["a"])
        self.assertFalse(dec["checks"]["S1"]["d"])

    def test_statuses(self):
        qq, rows = self._rows([("optimal", 10, 10, 100), ("timeout", 8, 8, 100), ("timeout", 5, 40, 100),
                               ("too_large", 0, 3000, 5000), ("budget", 0, -1, 100), ("optimal", 0, 0, 100)])
        x, (pf, xs) = yf.summarize_exact(qq, rows)
        self.assertAlmostEqual(x["share_proven"], 3 / 6)  # optimal, certified (ub == size), optimal empty
        self.assertAlmostEqual(x["share_certified_by_bound"], 1 / 6)
        self.assertAlmostEqual(x["share_found"], 3 / 6)
        self.assertAlmostEqual(x["share_not_proven_empty"], 5 / 6)
        self.assertAlmostEqual(x["share_unknown"], 2 / 6)
        # relative sizes of the open queries: 0.10, 0.08, 0.40 (bound), 0.60 (bound), 1 (none) -> median 0.40
        self.assertAlmostEqual(x["median_rel_size_open"], 0.40)
        self.assertEqual(pf.tolist(), [True, True, False, False, False, False])

    def test_rule_order_and_incomplete(self):
        good = {"share_found": 0.3, "share_not_proven_empty": 0.4, "median_rel_size_open": 0.1, "share_proven": 0.95}
        bad = {"share_found": 0.95, "share_not_proven_empty": 0.99, "median_rel_size_open": 1.0, "share_proven": 0.2}
        E = {"S1": {"exact": bad}, "S2": {"exact": good}, "S3": {"exact": good}, "G": {"exact": good}}
        self.assertEqual(yf.choose_spec(E, 10000)["spec"], "S2")
        d2 = yf.choose_spec({"S1": {"exact": bad}, "S2": {"error": "boom"}, "S3": {"exact": good},
                             "G": {"exact": good}}, 10000)
        self.assertIsNone(d2["spec"])
        self.assertTrue(d2["outcome"].startswith("incomplete"))
        E3 = {"S1": {"exact": bad}, "S2": {"exact": bad}, "S3": {"exact": bad}, "G": {"exact": good}}
        self.assertEqual(yf.choose_spec(E3, 10000)["spec"], "G")
        self.assertEqual(yf.choose_spec(E3, 1000)["outcome"], "infeasible as designed")  # (d): 1000 x 0.3 < 500
        edge = {"share_found": 0.10, "share_not_proven_empty": 0.90, "median_rel_size_open": 0.5, "share_proven": 0.80}
        self.assertEqual(yf.choose_spec({"S1": {"exact": edge}}, 5000)["spec"], "S1")  # bounds are inclusive

    def test_all_checks_are_recorded(self):
        good = {"share_found": 0.3, "share_not_proven_empty": 0.4, "median_rel_size_open": 0.1, "share_proven": 0.95}
        d = yf.choose_spec({s: {"exact": good} for s in ("S1", "S2", "S3", "G")}, 10000)
        self.assertEqual(d["spec"], "S1")
        self.assertEqual(sorted(d["checks"]), ["G", "S1", "S2", "S3"])
        self.assertTrue(all(c["passes"] for c in d["checks"].values()))

    def test_too_many_non_object_lines_stop(self):
        tmp = Path(tempfile.mkdtemp(prefix="e5lines_"))
        old = yf.MAX_BAD_LINES
        try:
            yf.MAX_BAD_LINES = 3
            p = tmp / "x.json"
            p.write_text("[1]\n" * 5 + '{"a": 1}\n')
            with self.assertRaises(RuntimeError):
                list(yf._Lines(p, Counter()))
        finally:
            yf.MAX_BAD_LINES = old
            shutil.rmtree(tmp, ignore_errors=True)

    def test_malformed_lines_are_skipped_and_counted(self):
        tmp = Path(tempfile.mkdtemp(prefix="e5lines_"))
        try:
            p = tmp / "x.json"
            p.write_text('{"a": 1}\n{"a": \n[1, 2]\n\n{"a": 2}\n')
            st = Counter()
            got = list(yf._Lines(p, st))
            self.assertEqual(got, [{"a": 1}, {"a": 2}])
            self.assertEqual(st["malformed_lines"], 2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestEngineBudget(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="e5eng_"))
        fake = self.tmp / "fake_engine.py"
        fake.write_text(FAKE_ENGINE)
        fake.chmod(0o755)

        class CS:
            pass

        self.cs = CS()
        self.cs.exe, self.cs.workdir = fake, self.tmp
        self.cs.graph_path, self.cs.pos_path, self.cs.theta_paths = self.tmp / "g", self.tmp / "p", {"acc": self.tmp / "t"}
        self.old = os.environ.get("FAKE_MODE")

    def tearDown(self):
        if self.old is None:
            os.environ.pop("FAKE_MODE", None)
        else:
            os.environ["FAKE_MODE"] = self.old
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_normal_batch(self):
        os.environ["FAKE_MODE"] = "ok"
        rows, meta = yf.run_engine_budget(self.cs, range(10), ["kcore", "exact"], 3, 1.0, 3, "t", 30.0)
        self.assertEqual(len(rows), 20)
        self.assertFalse(meta["killed_at_budget"])

    def test_budget_keeps_finished_queries(self):
        os.environ["FAKE_MODE"] = "sleep_all"
        rows, meta = yf.run_engine_budget(self.cs, range(10), ["kcore", "exact"], 3, 1.0, 2, "t", 2.0)
        self.assertTrue(meta["killed_at_budget"])
        self.assertEqual(sorted({r["q"] for r in rows}), [0, 1])  # first query of each process; cut line dropped
        self.assertEqual(len(rows), 4)

    def test_crash_is_reported_even_when_another_process_is_killed(self):  # review finding S2
        os.environ["FAKE_MODE"] = "crash0_sleep1"
        with self.assertRaises(RuntimeError) as cm:
            yf.run_engine_budget(self.cs, range(10), ["kcore", "exact"], 3, 1.0, 2, "t", 2.0)
        self.assertIn("bad_alloc", str(cm.exception))


class TestEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="e5feas_"))
        cls.truth = make_fixture.make(cls.tmp / "fx", seed=7)
        cls.zip = cls.tmp / "fx" / "Yelp-JSON.zip"
        cls.sha = yf.sha256_file(cls.zip)
        cls.out = cls.tmp / "out"
        rc = yf.main(["--zip", str(cls.zip), "--data-dir", str(cls.tmp / "json"), "--work", str(cls.tmp / "work"),
                      "--out", str(cls.out), "--expected-sha256", cls.sha, *ARGS])
        assert rc == 0, (cls.out / "feasibility.json").read_text()[:3000]
        cls.R = json.loads((cls.out / "feasibility.json").read_text())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_inventory_and_counts(self):
        R, T = self.R, self.truth
        inv = R["inventory"]
        self.assertTrue(inv["zip_sha256_ok"])
        self.assertEqual(inv["missing_kinds"], [])
        self.assertEqual(sorted(f["kind"] for f in inv["files"]), sorted(yf.KINDS))
        self.assertTrue(any(m["name"].endswith("Dataset_User_Agreement.pdf") for m in inv["tar_members"]))
        self.assertTrue(any(d["file"].endswith("ToS copy.pdf") for d in inv["docs"]))
        D = R["dataset"]
        self.assertEqual(D["n_business"], T["n_business"])
        self.assertEqual(D["n_users"], T["n_users"])
        self.assertEqual(D["n_reviews"], T["n_reviews_valid"])
        self.assertEqual(D["review_stats"], {"unknown_user": 2, "unknown_business": 1, "bad_date": 1})
        self.assertEqual(D["tip_lines"], T["tip_lines"])
        F = R["friends"]
        self.assertEqual(F["friend_refs"], T["friend_refs"])
        self.assertEqual(F["friend_refs_unknown"], T["friend_refs_unknown"])
        self.assertEqual(F["friend_refs_self"], T["friend_refs_self"])
        self.assertEqual(F["undirected_edges"], T["undirected_edges_known"])
        recip = 1 - T["one_directional_pairs"] / (F["friend_refs_resolved"])
        self.assertAlmostEqual(F["share_directed_reciprocated"], recip, places=9)
        metros = {m["state_top"][0][0]: m["businesses"] for m in R["metros"] if m["metro"] >= 0}
        self.assertEqual(metros, T["metros"])

    def test_splits_match_reference(self):
        metro_id = {m["state_top"][0][0]: m["metro"] for m in self.R["metros"] if m["metro"] >= 0}
        for split in ("precovid", "r4"):
            ref = reference(self.tmp / "fx" / "raw", split, metro_id)
            S = self.R["splits"][split]
            with self.subTest(split=split):
                self.assertEqual(S["users"]["taste_ready"], ref["taste_ready"])
                self.assertEqual(S["graph"]["edges"], ref["edges"])
                self.assertEqual(S["eligible_queries"]["k3"], ref["eligible_k3"])
                br = S["base_rates"]["test"]
                self.assertEqual(br["friend_pairs_both_active"]["n"], ref["friend_pairs_n"])
                self.assertEqual(br["friend_pairs_both_active"]["hits"], ref["friend_pairs_hits"])
                self.assertEqual(br["friend_pairs"]["n"], ref["edges"])  # decision-rule denominator: every edge
                self.assertEqual(br["friend_pairs"]["hits"], ref["friend_pairs_all_hits"])
                self.assertEqual(self.R["decision"]["split"]["checks"][split]["friend_rate"],
                                 ref["friend_pairs_all_hits"] / ref["edges"])
                tol = 1.0 / ref["n_edges_for_share"] + 1e-12  # one boundary pair may flip (float32 positions)
                self.assertLessEqual(abs(S["taste"]["S1"]["friend_pairs"]["share_mutually_within"] - ref["S1_within"]),
                                     tol)
                self.assertLessEqual(abs(S["taste"]["G"]["friend_pairs"]["share_mutually_within"] - ref["G_within"]),
                                     tol)

    def test_rates_are_consistent(self):
        for split, S in self.R["splits"].items():
            br = S["base_rates"]["test"]
            self.assertLessEqual(br["friend_pairs_within_30_days"]["hits"], br["friend_pairs"]["hits"])
            self.assertGreater(br["friend_pairs"]["rate"], br["random_same_metro_pairs"]["rate"])  # planted

    def test_engine_and_monotonicity(self):
        E = self.R["engine"]
        self.assertTrue(E["core_check"])
        for s in ("S1", "S2", "S3", "G"):
            self.assertNotIn("error", E[s])
            self.assertIn("exact", E[s])
            self.assertEqual(E[s]["exact"]["batch"]["killed_at_budget"], False)
            self.assertLessEqual(E[s]["exact"]["share_found"], E[s]["exact"]["share_not_proven_empty"])
        self.assertIn("--diam-cap", E["engine_settings"]["common"])
        if E["S1"]["exact"]["share_proven"] == 1.0 and E["S3"]["exact"]["share_proven"] == 1.0:
            # same positions, theta(S3) <= theta(S1): every LCC under S3 is one under S1
            self.assertLessEqual(E["S3"]["exact"]["share_found"], E["S1"]["exact"]["share_found"])
        self.assertIn("spec", self.R["decision"])
        self.assertIn("split", self.R["decision"]["split"])

    def test_no_ids_in_outputs(self):
        ids = set()
        raw = self.tmp / "fx" / "raw"
        for name, keys in (("yelp_academic_dataset_business.json", ["business_id"]),
                           ("yelp_academic_dataset_user.json", ["user_id"]),
                           ("yelp_academic_dataset_review.json", ["review_id", "user_id", "business_id"])):
            for line in open(raw / name):
                o = json.loads(line)
                ids.update(o[k] for k in keys)
        text = "".join(p.read_text(errors="ignore") for p in self.out.rglob("*") if p.is_file())
        leaked = [i for i in ids if i in text]
        self.assertEqual(leaked, [])

    def test_reuse_extraction(self):
        out2 = self.tmp / "out_again"
        rc = yf.main(["--zip", str(self.zip), "--data-dir", str(self.tmp / "json"), "--work", str(self.tmp / "work2"),
                      "--out", str(out2), "--expected-sha256", self.sha, "--no-engine", *ARGS])
        self.assertEqual(rc, 0)
        R2 = json.loads((out2 / "feasibility.json").read_text())
        self.assertTrue(R2["inventory"].get("reused_extraction"))
        self.assertFalse(R2["inventory"].get("reused_parse"))  # new work directory: parsed again
        out3 = self.tmp / "out_cached"
        rc = yf.main(["--zip", str(self.zip), "--data-dir", str(self.tmp / "json"), "--work", str(self.tmp / "work2"),
                      "--out", str(out3), "--expected-sha256", self.sha, "--no-engine", *ARGS])
        self.assertEqual(rc, 0)
        R3 = json.loads((out3 / "feasibility.json").read_text())
        self.assertTrue(R3["inventory"].get("reused_parse"))  # same work directory: cached arrays
        self.assertEqual(R3["friends"], R2["friends"])
        self.assertEqual(R3["splits"], R2["splits"])
        for split in ("precovid", "r4"):
            self.assertEqual(R2["splits"][split]["base_rates"]["test"]["friend_pairs"],
                             self.R["splits"][split]["base_rates"]["test"]["friend_pairs"])
            self.assertEqual(R2["splits"][split]["eligible_queries"], self.R["splits"][split]["eligible_queries"])

    def test_stale_or_broken_cache_is_ignored(self):
        work = self.tmp / "work_cache"
        work.mkdir()
        np.savez(work / "parsed.npz", meta_json=np.array(json.dumps({"key": {"files": {}, "code": "other"}})))
        out5 = self.tmp / "out_stale"
        rc = yf.main(["--zip", str(self.zip), "--data-dir", str(self.tmp / "json"), "--work", str(work),
                      "--out", str(out5), "--expected-sha256", self.sha, "--no-engine", *ARGS])
        self.assertEqual(rc, 0)
        R5 = json.loads((out5 / "feasibility.json").read_text())
        self.assertFalse(R5["inventory"]["reused_parse"])
        self.assertEqual(R5["splits"], self.R["splits"])
        (work / "parsed.npz").write_bytes(b"not a zip file")  # corrupted cache: ignored, parsed again
        rc = yf.main(["--zip", str(self.zip), "--data-dir", str(self.tmp / "json"), "--work", str(work),
                      "--out", str(self.tmp / "out_broken"), "--expected-sha256", self.sha, "--no-engine", *ARGS])
        self.assertEqual(rc, 0)
        self.assertFalse(json.loads((self.tmp / "out_broken" / "feasibility.json").read_text())["inventory"]["reused_parse"])

    def test_not_taste_ready_active_user_is_handled(self):  # review finding S4 (crashed before the fix)
        for split in ("precovid", "r4"):
            self.assertGreaterEqual(self.R["splits"][split]["users"]["active_without_taste"], 1)

    def test_small_conversion_chunks_give_the_same_results(self):
        old = yf.FLUSH_EVERY
        yf.FLUSH_EVERY = 1000
        try:
            out4 = self.tmp / "out_chunks"
            rc = yf.main(["--zip", str(self.zip), "--data-dir", str(self.tmp / "json"), "--work",
                          str(self.tmp / "work_chunks"), "--out", str(out4), "--expected-sha256", self.sha,
                          "--no-engine", *ARGS])
        finally:
            yf.FLUSH_EVERY = old
        self.assertEqual(rc, 0)
        R4 = json.loads((out4 / "feasibility.json").read_text())
        self.assertEqual(R4["dataset"]["n_reviews"], self.R["dataset"]["n_reviews"])
        self.assertEqual(R4["splits"], self.R["splits"])

    def test_wrong_checksum_stops(self):
        out3 = self.tmp / "out_bad"
        rc = yf.main(["--zip", str(self.zip), "--data-dir", str(self.tmp / "json_bad"), "--work", str(self.tmp / "w3"),
                      "--out", str(out3), "--expected-sha256", "0" * 64, "--no-engine", *ARGS])
        self.assertEqual(rc, 1)
        self.assertIn("sha256", json.loads((out3 / "feasibility.json").read_text())["error"])
        self.assertFalse((self.tmp / "json_bad" / "yelp_academic_dataset_review.json").exists())


if __name__ == "__main__":
    unittest.main()
