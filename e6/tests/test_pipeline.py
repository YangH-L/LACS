"""E6 chain on synthetic data in the real formats (e6/tests/fixture.py):
fetch --offline-dir -> build -> structure -> premise -> static -> report, with the build checked
against values the fixture computed on its own, an incomplete download, the data gate, and
hand-computed structure / Q2 numbers. About half a minute.

Run: python -m pytest e6/tests -q   (or: python -m unittest discover -s e6/tests -t .)
"""
from __future__ import annotations

import gzip
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e6 import build, fetch, premise, report, static, structure  # noqa: E402
from e6.common import sha256_file  # noqa: E402
from e6.tests import fixture  # noqa: E402
from lacs.graph.io import read_graph  # noqa: E402


class TestE6Chain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="e6_test_"))
        cls.truth = fixture.make(cls.tmp)
        cls.data, cls.run_dir = cls.tmp / "data", cls.tmp / "run"
        cls.res = cls.run_dir / "results"
        cls.rc_fetch = fetch.main(["--data", str(cls.data), "--offline-dir", str(cls.tmp / "offline"),
                                   "--threads", "4"])
        cls.build_args = ["--data", str(cls.data), "--network", str(cls.tmp / "network.json"),
                          "--crosswalk", str(cls.tmp / "crosswalk.csv"), "--procs", "1"]
        cls.rc_build = build.main(cls.build_args + ["--out", str(cls.run_dir), "--min-eligible", "100"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ------------------------------------------------------------- fetch --
    def test_fetch(self):
        self.assertEqual(self.rc_fetch, 0)
        man = json.loads((self.data / "fetch_manifest.json").read_text())
        self.assertTrue(man["complete"])
        bs = man["billstatus"]
        n_files = {t: len(list((self.tmp / "offline" / "billstatus" / t).glob("BILLSTATUS-117*.xml"))) for t in bs}
        self.assertEqual((bs["hr"]["listed_xml"], bs["hr"]["obtained_xml"], bs["hr"]["from_zip"]),
                         (n_files["hr"], n_files["hr"], 0))
        self.assertEqual(bs["hjres"]["listing"]["format"], "json")
        # s: package plus one listed file outside it
        self.assertEqual(bs["s"]["from_zip"], bs["s"]["listed_xml"] - 1)
        self.assertEqual(bs["s"]["downloaded"], 1)
        self.assertEqual(n_files["s"], 1)
        # sjres: the listing names only the package
        self.assertEqual((bs["sjres"]["listed_xml"], bs["sjres"]["listed_zip"]), (0, 1))
        self.assertGreater(bs["sjres"]["obtained_xml"], 10)
        self.assertTrue(all(len(f["sha256"]) == 64 for t in bs for f in bs[t]["files"]))
        self.assertEqual(man["voteview"]["sha256"], sha256_file(self.tmp / "offline" / "HS117_members.csv"))
        # resume: nothing is copied again
        rc = fetch.main(["--out", str(self.data), "--offline-dir", str(self.tmp / "offline")])  # --out = --data
        self.assertEqual(rc, 0)
        man2 = json.loads((self.data / "fetch_manifest.json").read_text())
        self.assertEqual(man2["totals"]["downloaded"], 0)
        self.assertTrue(man2["voteview"]["resumed"])

    def test_fetch_incomplete_stops(self):
        off = self.tmp / "offline_broken"
        shutil.copytree(self.tmp / "offline", off)
        victim = sorted((off / "billstatus" / "hr").glob("BILLSTATUS-117hr*.xml"))[3]
        victim.unlink()
        (off / "billstatus" / "hjres" / "BILLSTATUS-117hjres2.xml").write_text("<html>error</html>")
        data = self.tmp / "data_broken"
        rc = fetch.main(["--data", str(data), "--offline-dir", str(off)])
        self.assertEqual(rc, 2)
        man = json.loads((data / "fetch_manifest.json").read_text())
        self.assertFalse(man["complete"])
        self.assertEqual(man["billstatus"]["hr"]["missing"], [victim.name])
        self.assertEqual(man["billstatus"]["hjres"]["missing"], ["BILLSTATUS-117hjres2.xml"])
        # build refuses an incomplete download
        rc = build.main(["--data", str(data), "--out", str(self.tmp / "run_broken"), "--network",
                         str(self.tmp / "network.json"), "--crosswalk", str(self.tmp / "crosswalk.csv")])
        self.assertEqual(rc, 2)
        # without a saved listing the offline mode refuses, unless told otherwise
        (off / "billstatus" / "hr" / "listing.xml").unlink()
        self.assertEqual(fetch.main(["--data", str(self.tmp / "data_nolist"), "--offline-dir", str(off),
                                     "--types", "hr"]), 2)
        self.assertEqual(fetch.main(["--data", str(self.tmp / "data_nolist2"), "--offline-dir", str(off),
                                     "--types", "hr", "--no-listing-ok"]), 0)

    # ------------------------------------------------------------- build --
    def test_build_matches_truth(self):
        self.assertEqual(self.rc_build, 0)
        t = self.truth
        users = pd.read_csv(self.run_dir / "geo" / "users.csv")
        self.assertEqual(list(users.columns[:6]), ["idx", "user", "n_train", "theta_q50", "theta_q70", "theta_q90"])
        self.assertEqual(users["user"].tolist(), t["eligible"])  # sorted bioguide order
        self.assertEqual(users["idx"].tolist(), list(range(len(users))))
        u = users.set_index("user")
        from e6.common import ENGINE_TIE_ATOL, ENGINE_TIE_RTOL  # users.csv: theta * (1 + rtol) + atol (float32 ties)
        f, a0 = 1.0 + ENGINE_TIE_RTOL, ENGINE_TIE_ATOL
        for q in ("q50", "q70", "q90"):
            np.testing.assert_allclose(u.loc[t["eligible"], f"theta_{q}"],
                                       [t[f"theta_{q}"][b] * f + a0 for b in t["eligible"]], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(u.loc[t["eligible"], "theta2d_q50"],
                                   [t["theta2d_q50"][b] * f + a0 for b in t["eligible"]], rtol=1e-12, atol=1e-12)
        self.assertEqual(u.loc[t["eligible"], "n_train"].tolist(), [t["n_train"][b] for b in t["eligible"]])
        m = pd.read_csv(self.run_dir / "public" / "e6_members.csv").set_index("bioguide")
        self.assertEqual(len(m), len(t["bioguides"]))
        self.assertAlmostEqual(m.loc[t["dual_member"], "dim1"], t["dual_dim1"])  # the row with more votes
        self.assertTrue((m.loc[t["no_vv"], "reason"] == "not_in_voteview").all())
        self.assertTrue((m.loc[t["no_dim1"], "reason"] == "no_nominate_dim1").all())
        self.assertTrue((m.loc[t["low"], "reason"] == "lt10_train_cosponsorships").all())
        self.assertEqual(int(m["eligible"].sum()), len(t["eligible"]))
        self.assertTrue((m.loc[m["eligible"], "idx"].sort_values().to_numpy() == np.arange(len(users))).all())
        # independents caucus with the Democrats
        self.assertTrue((m.loc[m["party"] == "I", "caucus"] == "D").all())
        # test-period joins (withdrawn excluded, decoy amendment cosponsors ignored)
        idx = dict(zip(users["user"], users["idx"]))
        J = pd.DataFrame(t["joins"], columns=["key", "b", "p"])
        J = J[J["b"].isin(idx.keys()) & (J["p"] == 2)]
        tz = np.load(self.run_dir / "geo" / "test_checkins.npz")
        self.assertEqual(tz["user"].dtype, np.int32)
        self.assertEqual(tz["key"].dtype, np.int64)
        self.assertEqual(set(zip(tz["user"].tolist(), tz["key"].tolist())),
                         set(zip(J["b"].map(idx).tolist(), J["key"].tolist())))
        ev = pd.read_csv(self.run_dir / "public" / "e6_events.csv.gz")
        self.assertEqual(set(ev["role"]), {"sponsor", "cosponsor"})
        info = json.loads((self.run_dir / "public" / "e6_build.json").read_text())
        self.assertEqual(info["billstatus"]["bills"], t["n_bills"])  # bills without any (co)sponsor have no event
        self.assertLessEqual(ev["bill_id"].nunique(), t["n_bills"])
        self.assertEqual(int(ev["withdrawn"].sum()), info["billstatus"]["cosponsorships_withdrawn"])

    def test_geo_files(self):
        geo = self.run_dir / "geo"
        users = pd.read_csv(geo / "users.csv")
        n = len(users)
        pos = np.load(geo / "positions.npy")
        pos2 = np.load(geo / "positions_2d.npy")
        self.assertEqual((pos.shape, pos.dtype, pos2.shape), ((n, 3), np.float64, (n, 3)))
        np.testing.assert_array_equal(pos[:, 0], users["dim1"].to_numpy())
        self.assertTrue((pos[:, 1:] == 0).all() and (pos2[:, 2] == 0).all())
        np.testing.assert_array_equal(pos2[:, 1], users["dim2"].to_numpy())
        A = read_graph(self.run_dir / "graph" / "graph.bin")
        self.assertEqual(A.shape, (n, n))
        self.assertEqual((A != A.T).nnz, 0)
        # graph = undirected union of the network's edges among members; weights as in the JSON
        net = json.loads((self.tmp / "network.json").read_text())[0]
        cw = pd.read_csv(self.tmp / "crosswalk.csv")
        b_of = dict(zip(cw["node"], cw["bioguide"]))
        idx = dict(zip(users["user"], users["idx"]))
        und, dirw = set(), {}
        for i, (lst, wl) in enumerate(zip(net["outList"], net["outWeight"])):
            for j, w in zip(lst, wl):
                if b_of[i] in idx and b_of[j] in idx:
                    a, b = idx[b_of[i]], idx[b_of[j]]
                    und.add((min(a, b), max(a, b)))
                    dirw[(a, b)] = w
        up = sp.triu(A, 1).tocoo()
        self.assertEqual(set(zip(up.row.tolist(), up.col.tolist())), und)
        tw = np.load(geo / "twitter_weights.npz")
        self.assertEqual({(int(s), int(d)): float(w) for s, d, w in zip(tw["src"], tw["dst"], tw["w"])}, dirw)
        cov = np.load(geo / "covisits.npz")
        self.assertEqual(set(cov.files), {"train", "val", "test"})
        meta = json.loads((geo / "meta.json").read_text())
        self.assertEqual(meta["n_active"], n)
        self.assertEqual(meta["max_bucket_users"], 500)
        # notop: recompute the threshold from the public events table (test-window joins of all members,
        # withdrawn cosponsorships excluded; E5's linear 99th percentile)
        tz, nt = np.load(geo / "test_checkins.npz"), np.load(geo / "test_checkins_notop.npz")
        evp = pd.read_csv(self.run_dir / "public" / "e6_events.csv.gz")
        evp = evp[(evp["period"] == "test") & ~evp["withdrawn"].astype(bool)]
        cnt = evp[["member", "bill_id"]].drop_duplicates().groupby("bill_id").size()
        thr = np.quantile(cnt.to_numpy(), 0.99)
        top = set(cnt.index[cnt >= thr])
        self.assertEqual(meta["test_notop"]["threshold"], float(thr))
        keep = ~np.isin(tz["key"], list(top))
        np.testing.assert_array_equal(nt["key"], tz["key"][keep])
        np.testing.assert_array_equal(nt["user"], tz["user"][keep])
        self.assertGreater(len(top), 0)

    def test_gate(self):
        out = self.tmp / "run_gate"
        rc = build.main(self.build_args + ["--out", str(out)])  # default gate: 400 members
        self.assertEqual(rc, 3)
        info = json.loads((out / "public" / "e6_build.json").read_text())
        self.assertFalse(info["gate"]["pass"])
        self.assertTrue((out / "public" / "e6_members.csv").exists())
        self.assertFalse((out / "geo" / "users.csv").exists())

    def test_unparseable_file_stops_build(self):
        data = self.tmp / "data_corrupt"
        shutil.copytree(self.data, data)
        victim = sorted((data / "billstatus" / "s").glob("BILLSTATUS-117s*.xml"))[0]
        victim.write_text(victim.read_text()[:200])  # truncated after the download check
        args = ["--data", str(data), "--network", str(self.tmp / "network.json"), "--crosswalk",
                str(self.tmp / "crosswalk.csv"), "--procs", "1", "--min-eligible", "100"]
        self.assertEqual(build.main(args + ["--out", str(self.tmp / "run_corrupt")]), 2)
        info = json.loads((self.tmp / "run_corrupt" / "public" / "e6_build.json").read_text())
        self.assertEqual(info["billstatus"]["n_parse_errors"], 1)
        self.assertEqual(build.main(args + ["--out", str(self.tmp / "run_corrupt2"), "--allow-incomplete"]), 0)

    # ------------------------------------------------- structure, Q1, Q2, report --
    def test_downstream(self):
        users = pd.read_csv(self.run_dir / "geo" / "users.csv", float_precision="round_trip")
        pos = np.load(self.run_dir / "geo" / "positions.npy")
        th = users["theta_q50"].to_numpy()
        A = read_graph(self.run_dir / "graph" / "graph.bin")
        self.res.mkdir(parents=True, exist_ok=True)
        # structure: hand-made per-query table
        pq = pd.DataFrame([
            {"q": 0, "method": "lcc_exact", "status": "optimal", "size": 5},
            {"q": 0, "method": "kcore", "status": "ok", "size": 10},
            {"q": 1, "method": "lcc_exact", "status": "optimal", "size": 0},
            {"q": 1, "method": "kcore", "status": "ok", "size": 10},
            {"q": 2, "method": "lcc_exact", "status": "timeout", "size": 9},
            {"q": 2, "method": "kcore", "status": "ok", "size": 10},
            {"q": 3, "method": "lcc_exact", "status": "optimal", "size": 2},
            {"q": 3, "method": "kcore", "status": "ok", "size": 4},
            {"q": 3, "method": "pairwise", "status": "optimal", "size": 3}])
        pq.to_csv(self.res / "geo_e2_per_query.csv", index=False)
        self.assertEqual(structure.main(["--run", str(self.run_dir)]), 0)
        st = json.loads((self.res / "e6_structure.json").read_text())
        self.assertAlmostEqual(st["lcc"]["share_lcc_nonempty"], 0.75)
        self.assertAlmostEqual(st["lcc"]["ratio"]["median"], 0.5)  # ratios 0.5, 0.9, 0.5
        self.assertFalse(st["lcc"]["no_filtering"])
        cnt = tot = 0
        Ad = A.toarray()
        for a in range(len(th)):
            for b in range(a + 1, len(th)):
                if Ad[a, b]:
                    tot += 1
                    d = abs(pos[a, 0] - pos[b, 0])
                    cnt += (d <= th[a] * (1 + 1e-9)) and (d <= th[b] * (1 + 1e-9))
        self.assertEqual(st["friend_pairs"]["edges"], tot)
        self.assertAlmostEqual(st["friend_pairs"]["share_mutual_within"], cnt / tot)

        # Q1: planted acceptance effect, 20 permutations
        self.assertEqual(premise.main(["--run", str(self.run_dir), "--n-perm", "20", "--procs", "2",
                                       "--n-boot", "200"]), 0)
        pr = json.loads((self.res / "e6_premise.json").read_text())
        prim = pr["variants"]["primary_1d_q50_indicator"]
        self.assertGreater(prim["coef"], 0.5)
        self.assertLess(prim["perm"]["p_one_sided"], 0.05)
        self.assertLess(prim["oof"]["ci95"][1], 0)
        self.assertEqual(pr["verdict"]["code"], "predicts")
        self.assertEqual(set(pr["variants"]), set(premise.VARIANTS))
        self.assertTrue(all(v["m1"]["converged"] for v in pr["variants"].values()))
        self.assertIn("spearman", pr["member_level"])
        perm = pd.read_csv(self.res / "e6_premise_perm.csv")
        self.assertEqual(len(perm), 20 * len(premise.VARIANTS))
        # deterministic: one process gives the same permutation estimates
        res1 = self.tmp / "res_serial"
        self.assertEqual(premise.main(["--run", str(self.run_dir), "--results", str(res1), "--n-perm", "20",
                                       "--procs", "1", "--n-boot", "200"]), 0)
        np.testing.assert_allclose(pd.read_csv(res1 / "e6_premise_perm.csv")["coef"], perm["coef"], rtol=0, atol=1e-12)

        # Q2: hand-made groups in the P0 export format
        exp = self.tmp / "p0export"
        inst = self.tmp / "inst"
        exp.mkdir()
        inst.mkdir()
        np.save(inst / "indptr.npy", A.indptr.astype(np.int64))
        np.save(inst / "indices.npy", A.indices.astype(np.int32))
        np.save(inst / "X_default.npy", pos.astype(np.float32))
        np.save(inst / "theta_default.npy", th.astype(np.float32))
        (exp / "instance.json").write_text(json.dumps({"indptr": "../inst/indptr.npy", "indices": "../inst/indices.npy",
                                                       "X": "../inst/X_default.npy", "theta": "../inst/theta_default.npy",
                                                       "meta": {"dataset": "run", "tag": "default"}}))
        X32, t32 = pos.astype(np.float32).astype(np.float64), th.astype(np.float32).astype(np.float64)
        order = np.argsort(X32[:, 0], kind="stable")
        groups = [
            {"query": 0, "method": "lcc_exact", "members": [int(order[0]), int(order[1])], "size": 2},
            {"query": 0, "method": "kcore", "members": list(range(len(th))), "size": len(th)},
            {"query": 0, "method": "random_bfs", "members": [int(order[0]), int(order[-1]), int(order[len(th) // 2])],
             "size": 3},
            {"query": 1, "method": "lcc_exact", "members": [], "size": 0},
            {"query": 1, "method": "kcore", "members": list(range(len(th))), "size": len(th)},
            {"query": 1, "method": "random_bfs", "members": None, "size": 4}]
        with gzip.open(exp / "groups.jsonl.gz", "wt") as f:
            for g in groups:
                f.write(json.dumps({"dataset": "run", "setting": "default", "k": 3, "status": "ok", **g}) + "\n")
        self.assertEqual(static.main(["--groups", str(exp), "--results", str(self.res)]), 0)
        sres = json.loads((self.res / "e6_static.json").read_text())
        block = sres["results"][0]["methods"]
        # the same with the file form used by the server script
        self.assertEqual(static.main(["--instance", str(exp / "instance.json"), "--groups",
                                      str(exp / "groups.jsonl.gz"), "--out", str(self.tmp / "static2")]), 0)
        self.assertEqual(json.loads((self.tmp / "static2" / "e6_static.json").read_text())["results"], sres["results"])

        def brute(H):
            H = list(H)
            D = np.abs(X32[H][:, None, 0] - X32[H][None, :, 0])
            diam = D.max()
            mv = sum(D[i].max() > t32[H[i]] * (1 + 1e-9) for i in range(len(H)))
            pv = sum(D[i, j] > min(t32[H[i]], t32[H[j]]) * (1 + 1e-9) for i in range(len(H)) for j in range(i + 1, len(H)))
            return diam <= min(t32[H]) * (1 + 1e-9), mv, pv, len(H) * (len(H) - 1) // 2

        iii, mv, pv, npairs = brute(groups[0]["members"])
        self.assertEqual(block["lcc_exact"]["all"]["groups"], 1)
        self.assertEqual(block["lcc_exact"]["all"]["share_iii"], float(iii))
        self.assertAlmostEqual(block["lcc_exact"]["all"]["share_pairs_rejected"], pv / npairs)
        iii, mv, pv, npairs = brute(range(len(th)))
        k_all = block["kcore"]["all"]
        self.assertEqual(k_all["groups"], 2)
        self.assertAlmostEqual(k_all["share_members_ecc_gt_theta"], mv / len(th))
        self.assertAlmostEqual(k_all["share_pairs_rejected"], pv / npairs)
        self.assertEqual(k_all["share_iii"], float(iii))
        self.assertEqual(block["kcore"]["lcc_nonempty"]["groups"], 1)  # only query 0 has an LCC
        self.assertEqual(block["random_bfs"]["all"]["groups_without_members"], 1)
        self.assertEqual(block["random_bfs"]["all"]["groups"], 1)
        _, mv, _, _ = brute(groups[2]["members"])
        self.assertAlmostEqual(block["random_bfs"]["all"]["share_members_ecc_gt_theta"], mv / 3)

        # report without Q3 / Q4, then with them
        self.assertEqual(report.main(["--run", str(self.run_dir)]), 0)
        md = (self.res / "E6_REPORT.md").read_text()
        rj = json.loads((self.res / "e6_report.json").read_text())
        for h in ("## 1.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7."):
            self.assertIn(h, md)
        self.assertEqual(rj["status"]["q3"], "missing")
        self.assertEqual(rj["status"]["q4"], "missing")
        self.assertEqual(rj["flags"]["q1_verdict"], "predicts")
        self.assertIn("联署估出的接受域", md)
        pd.DataFrame([{"dataset": "e6", "method": "lcc_exact", "variant": "hk_equal", "w": None, "groups": 10,
                       "share_broken": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "share_cond_special": 1.0,
                       "share_cond_refined": 1.0, "theorem_violations": 0},
                      {"dataset": "e6", "method": "kcore", "variant": "open", "w": 0.1, "groups": 10,
                       "share_broken": 0.4, "ci_lo": 0.1, "ci_hi": 0.7, "share_cond_special": 0.0,
                       "share_cond_refined": 0.2, "theorem_violations": 0}]).to_csv(self.res / "e1x_summary.csv",
                                                                                  index=False)
        (self.res / "geo_e2.json").write_text(json.dumps({
            "k": 3, "tolerance": "theta_q50", "queries": 4, "eligible": 4, "query_mode": "all",
            "summary": [{"method": "lcc_exact", "nonempty": 3, "covisit_pair_rate": 0.2},
                        {"method": "kcore", "nonempty": 4, "covisit_pair_rate": 0.1}],
            "paired_covisit_pair_rate_vs_lcc_exact": {"kcore": {"n": 3, "mean_diff": 0.1, "ci95": [0.0, 0.2],
                                                                "p_boot": 0.04, "n_clusters": 3}}}))
        self.assertEqual(report.main(["--run", str(self.run_dir)]), 0)
        rj = json.loads((self.res / "e6_report.json").read_text())
        md = (self.res / "E6_REPORT.md").read_text()
        self.assertEqual(rj["status"]["q3"], "present")
        self.assertEqual(rj["status"]["q4"], "present")
        self.assertEqual(rj["flags"]["q3_theorem_violations"], 0)
        self.assertIn("paired_covisit_pair_rate_vs_lcc_exact", md)
        self.assertIn("hk_equal", md)


if __name__ == "__main__":
    unittest.main()
