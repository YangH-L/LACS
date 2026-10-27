"""Unit tests of the E6 package: BILLSTATUS parsing (real samples, both formats), file names, periods,
tolerances on a toy case computed by hand, the notop threshold, listings, the shipped crosswalk,
and the fixed-effects logit (dense reference solution with explicit dummy columns).

samples/: five BILLSTATUS files copied from github.com/usgpo/bill-status (commit 635c5dcf,
samples/formatChange/current, samples/formatChange/future and samples/legacy; US government works).
The whole samples tree is also parsed when data/e6/bill-status/samples exists.

Run: python -m pytest e6/tests -q   (or: python -m unittest discover -s e6/tests -t .)
"""
from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from e6 import build, fetch, premise  # noqa: E402
from e6.common import (DATA_DIR, bill_id, parse_billstatus, parse_filename, period_of, sha256_file,  # noqa: E402
                       split_bill_id)

SAMPLES = Path(__file__).resolve().parent / "samples"
FULL_SAMPLES = Path("data/e6/bill-status/samples")  # the usgpo/bill-status repository, if present
PLAN_MANUAL = {"CoryBooker", "SenGillibrand", "LindseyGrahamSC", "HawleyMO", "timkaine", "ChrisMurphyCT", "ossoff",
               "RandPaul", "marcorubio", "CongresswomanSC", "repdelgado", "Lancegooden", "claudiatenney",
               "michaelgwaltz"}


class TestBillstatus(unittest.TestCase):
    def test_real_samples_both_formats(self):
        cur = parse_billstatus(SAMPLES / "current" / "BILLSTATUS-117s35.xml")  # <billType>, <billNumber>
        new = parse_billstatus(SAMPLES / "future" / "BILLSTATUS-117s35.xml")   # <type>, <number>, no withdrawn tag
        for d in (cur, new):
            self.assertEqual((d["congress"], d["type"], d["number"], d["introduced"]), (117, "s", 35, "2021-01-22"))
            self.assertEqual(d["sponsors"], ["V000128"])
            self.assertEqual(len(d["cosponsors"]), 72)
            self.assertFalse(any(c["withdrawn"] for c in d["cosponsors"]))
        key = lambda d: sorted((c["bioguide"], c["date"], c["original"]) for c in d["cosponsors"])  # noqa: E731
        self.assertEqual(key(cur), key(new))
        a = parse_billstatus(SAMPLES / "current" / "BILLSTATUS-117hr6658.xml")
        b = parse_billstatus(SAMPLES / "future" / "BILLSTATUS-117hr6658.xml")
        self.assertEqual((a["type"], a["number"], a["sponsors"]), ("hr", 6658, ["D000619"]))
        extra = set(key(b)) - set(key(a))  # the newer snapshot has one later cosponsor
        self.assertEqual(extra, {("F000475", "2022-11-16", False)})
        self.assertEqual(len(a["cosponsors"]), 110)
        leg = parse_billstatus(SAMPLES / "legacy" / "BILLSTATUS-114hjres45.xml")
        self.assertEqual((leg["congress"], leg["type"], leg["number"], len(leg["cosponsors"])), (114, "hjres", 45, 6))

    @unittest.skipUnless(FULL_SAMPLES.exists(), "usgpo/bill-status samples not available")
    def test_all_repository_samples(self):
        files = sorted(FULL_SAMPLES.rglob("*.xml"))
        self.assertGreater(len(files), 40)
        for f in files:
            d = parse_billstatus(f)
            self.assertIsNotNone(d["type"], f)
            self.assertIsNotNone(d["number"], f)
        # amendment cosponsors (one withdrawn on 2021-05-25) are not the bill's
        for fmt in ("current", "future"):
            d = parse_billstatus(FULL_SAMPLES / "formatChange" / fmt / "BILLSTATUS-117s1260.xml")
            self.assertEqual(len(d["cosponsors"]), 13)
            self.assertFalse(any(c["withdrawn"] for c in d["cosponsors"]))
            self.assertEqual(d["sponsors"], ["S000148"])

    def test_withdrawn_tag_variants(self):
        base = ("<billStatus><bill><{n}>7</{n}><{t}>HR</{t}><introducedDate>2021-03-01</introducedDate>"
                "<congress>117</congress><sponsors><item><bioguideId>A000001</bioguideId></item></sponsors>"
                "<cosponsors>{items}</cosponsors></bill></billStatus>")
        items = ("<item><bioguideId>B1</bioguideId><sponsorshipDate>2021-03-01</sponsorshipDate>"
                 "<isOriginalCosponsor>True</isOriginalCosponsor><sponsorshipWithdrawnDate/></item>"
                 "<item><bioguideId>B2</bioguideId><sponsorshipDate>2021-03-05</sponsorshipDate>"
                 "<isOriginalCosponsor>False</isOriginalCosponsor>"
                 "<sponsorshipWithdrawnDate>2021-04-01</sponsorshipWithdrawnDate></item>"
                 "<item><bioguideId>B3</bioguideId><sponsorshipDate>2021-03-06</sponsorshipDate>"
                 "<isOriginalCosponsor>False</isOriginalCosponsor></item>")
        for n, t in (("billNumber", "billType"), ("number", "type")):
            d = parse_billstatus(base.format(n=n, t=t, items=items).encode())
            self.assertEqual((d["type"], d["number"]), ("hr", 7))
            self.assertEqual([c["withdrawn"] for c in d["cosponsors"]], [False, True, False])
            self.assertEqual(d["cosponsors"][1]["withdrawn_date"], "2021-04-01")
            self.assertEqual([c["original"] for c in d["cosponsors"]], [True, False, False])

    def test_file_names_and_ids(self):
        self.assertEqual(parse_filename("BILLSTATUS-117hr12.xml"), (117, "hr", 12))
        self.assertEqual(parse_filename("BILLSTATUS-117hres12.xml"), (117, "hres", 12))
        self.assertEqual(parse_filename("BILLSTATUS-117hjres3.xml"), (117, "hjres", 3))
        self.assertEqual(parse_filename("BILLSTATUS-117sconres3.xml"), (117, "sconres", 3))
        self.assertEqual(parse_filename("BILLSTATUS-117s3.xml"), (117, "s", 3))
        self.assertIsNone(parse_filename("BILLSTATUS-117-hr.zip"))
        self.assertEqual(bill_id("hr", 12), 100012)
        self.assertEqual(bill_id("sjres", 7), 400007)
        self.assertEqual(split_bill_id(bill_id("s", 5309)), ("s", 5309))

    def test_period_boundaries(self):
        cases = {"2021-01-02": -1, "2021-01-03": 0, "2022-02-08": 0, "2022-02-09": 1, "2022-06-09": 1,
                 "2022-06-10": 2, "2023-01-03": 2, "2023-01-04": -1, None: -1, "": -1, "2022-02-08T23:59:00Z": 0}
        for d, p in cases.items():
            self.assertEqual(period_of(d), p, d)


class TestTolerances(unittest.TestCase):
    def test_toy_case_by_hand(self):
        # sponsors S1..S5, members A (4 usable cosponsorships), B (distances below the floor), C (none usable)
        pos = pd.DataFrame({"dim1": {"A": 0.0, "B": 0.5, "C": 0.9, "S1": 0.1, "S2": -0.3, "S3": 0.2, "S4": 0.4,
                                     "S5": 0.505},
                            "dim2": {"A": 0.0, "B": 0.0, "C": 0.0, "S1": 0.0, "S2": 0.4, "S3": 0.0, "S4": 0.3,
                                     "S5": 0.0}})
        bills = pd.DataFrame({"bill_id": [1, 2, 3, 4, 5, 6, 7, 8],
                              "sponsor": ["S1", "S2", "S3", "S4", "S5", "S5", "S9", "S1"]})
        ev = pd.DataFrame({
            "bill_id": [1, 2, 3, 4, 5, 6, 7, 8, 5, 1, 2],
            "member": ["A", "A", "A", "A", "B", "B", "A", "A", "C", "C", "C"],
            "role": ["cosponsor"] * 11,
            "join_date": ["2021-02-01", "2021-05-01", "2022-02-08", "2021-07-01", "2021-03-01", "2021-04-01",
                          "2021-03-01", "2022-03-01", "2022-02-09", "2021-03-01", "2021-03-02"],
            "withdrawn": [False, False, False, False, False, False, False, False, False, True, True]})
        ev["period"] = ev["join_date"].map(period_of)
        cos = build.train_cosponsorships(ev, bills)
        # A: bills 1-4 (bill 7's sponsor S9 has no score, bill 8 is in the validation period);
        # C: bill 5 is in validation, bills 1 and 2 were withdrawn -> no usable cosponsorship
        t1 = build.tolerances(cos, pos, ["dim1"])
        self.assertEqual(int(t1.loc["A", "n"]), 4)
        # A's distances: 0.1, 0.3, 0.2, 0.4 -> sorted 0.1 0.2 0.3 0.4
        self.assertAlmostEqual(t1.loc["A", "q50"], 0.25)
        self.assertAlmostEqual(t1.loc["A", "q70"], 0.31)  # position 0.7 * 3 = 2.1: 0.3 + 0.1 * 0.1
        self.assertAlmostEqual(t1.loc["A", "q90"], 0.37)  # position 2.7: 0.3 + 0.7 * 0.1
        self.assertAlmostEqual(t1.loc["B", "raw_q50"], 0.005)
        self.assertAlmostEqual(t1.loc["B", "q50"], 0.02)  # floor
        self.assertNotIn("C", t1.index)
        t2 = build.tolerances(cos, pos, ["dim1", "dim2"])
        d = sorted([0.1, np.hypot(0.3, 0.4), 0.2, np.hypot(0.4, 0.3)])  # 0.1 0.2 0.5 0.5
        self.assertAlmostEqual(t2.loc["A", "q50"], (d[1] + d[2]) / 2)

    def test_notop_threshold(self):
        # E5's rule: numpy's default (linear) 99th percentile of the per-bill joiner counts over everybody
        user = np.concatenate([np.arange(c) for c in range(1, 101)])  # bill k has k joiners
        key = np.concatenate([np.full(c, 1000 + c) for c in range(1, 101)])
        keep, info = build.notop_filter(user, key, user, key)
        self.assertAlmostEqual(info["threshold"], float(np.quantile(np.arange(1, 101), 0.99)))  # 99.01
        self.assertEqual(info["bills_removed"], 1)
        self.assertEqual(set(np.unique(key[~keep])), {1100})
        # counts come from all members: bills that look small among network members can still be top bills
        sub = user < 3  # network members are 0, 1, 2 only
        keep2, info2 = build.notop_filter(user[sub], key[sub], user, key)
        self.assertEqual(set(np.unique(key[sub][~keep2])), {1100})
        # ties: 150 bills with 1 joiner, 3 with 50: the 99th percentile is 50, all three go
        user = np.concatenate([np.zeros(150, int)] + [np.arange(50)] * 3)
        key = np.concatenate([np.arange(150), np.full(50, 900), np.full(50, 901), np.full(50, 902)])
        keep, info = build.notop_filter(user, key, user, key)
        self.assertEqual((info["threshold"], info["bills_removed"]), (50.0, 3))
        # duplicate (user, bill) rows count once
        keep, info = build.notop_filter(np.array([1, 1, 2]), np.array([5, 5, 6]), np.array([1, 1, 2]),
                                        np.array([5, 5, 6]))
        self.assertEqual(info["bills_with_joins"], 2)


class TestListing(unittest.TestCase):
    def test_xml_and_json(self):
        xml = (b"<files><file><justFileName>BILLSTATUS-117-hr.zip</justFileName><fileExtension>zip</fileExtension>"
               b"<folder>false</folder><link>https://www.govinfo.gov/bulkdata/BILLSTATUS/117/hr/BILLSTATUS-117-hr.zip"
               b"</link></file><file><justFileName>BILLSTATUS-117hr1.xml</justFileName><fileExtension>xml"
               b"</fileExtension><folder>false</folder><link>https://x/BILLSTATUS-117hr1.xml</link></file>"
               b"<file><justFileName>resources</justFileName><folder>true</folder></file></files>")
        e = fetch.parse_listing(xml)
        self.assertEqual([(x["name"], x["ext"]) for x in e], [("BILLSTATUS-117-hr.zip", "zip"),
                                                               ("BILLSTATUS-117hr1.xml", "xml")])
        js = json.dumps({"files": [{"justFileName": "BILLSTATUS-117s1.xml", "fileExtension": "xml", "folder": False,
                                    "link": "https://x/BILLSTATUS-117s1.xml"},
                                   {"justFileName": "sub", "folder": True}]}).encode()
        e = fetch.parse_listing(js)
        self.assertEqual([(x["name"], x["ext"], x["link"]) for x in e],
                         [("BILLSTATUS-117s1.xml", "xml", "https://x/BILLSTATUS-117s1.xml")])


class TestShippedCrosswalk(unittest.TestCase):
    def test_crosswalk_consistency(self):
        with open(DATA_DIR / "crosswalk.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 475)
        self.assertEqual(sum(r["method"] == "auto" for r in rows), 461)
        self.assertEqual(sum(r["method"] == "manual" for r in rows), 14)
        self.assertEqual(len({r["bioguide"] for r in rows}), 475)
        self.assertEqual({r["handle"] for r in rows if r["method"] == "manual"}, PLAN_MANUAL)
        net = json.loads((DATA_DIR / "congress_network_data.json").read_text())[0]
        self.assertEqual([r["handle"] for r in sorted(rows, key=lambda r: int(r["node"]))], net["usernameList"])
        self.assertEqual(sorted(int(r["node"]) for r in rows), list(range(475)))
        by = {r["handle"]: r for r in rows}
        self.assertEqual(by["CongresswomanSC"]["bioguide"], "C001127")  # corrected appendix row
        self.assertEqual(by["RepNancyMace"]["bioguide"], "M000194")
        self.assertEqual((by["repdelgado"]["bioguide"], by["repdelgado"]["state"]), ("D000630", "NY"))
        self.assertTrue(all(r["chamber"] in ("House", "Senate") for r in rows))

    def test_manifest(self):
        man = json.loads((DATA_DIR / "MANIFEST.json").read_text())
        for name in ("congress_network_data.json", "crosswalk.csv"):
            self.assertEqual(man["files"][name]["sha256"], sha256_file(DATA_DIR / name))
        self.assertEqual(man["files"]["congress_network_data.json"]["sha256"],
                         "ee6d2fc1a2ac07e6481d41c0df463e2e7c5c0db4ae964980b43233e5c7253712")
        self.assertEqual(man["files"]["congress_network_data.json"]["commit"],
                         "205ca40d973e13944051e1e44240d005a9b1a1bc")
        self.assertTrue(man["files"]["crosswalk.csv"]["sources"]["commit"].startswith("b6c8c0bd"))
        self.assertEqual(man["plan_literal_duplicate_bioguides"], {"M000194": ["CongresswomanSC", "RepNancyMace"]})
        self.assertTrue(all(c["ok"] for c in man["manual_checks"]))
        self.assertTrue((DATA_DIR / "LICENSE_CongressionalTwitterNetwork.txt").exists())


class TestFit(unittest.TestCase):
    def _toy(self, n=12, seed=3):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=n)
        D = np.abs(x[:, None] - x[None, :])
        Y = (rng.random((n, n)) < 0.3).astype(float)
        np.fill_diagonal(Y, 0)
        M = 1.0 - np.eye(n)
        Xg = np.stack([np.ones((n, n)), D, (rng.random((n, n)) < 0.5).astype(float)])
        return Xg, Y, M

    def test_matches_dense_reference(self):
        from scipy.optimize import minimize

        Xg, Y, M = self._toy()
        p, n, _ = Xg.shape
        f = premise.fit_fe_logit(Xg, Y, M)
        u, v = np.nonzero(M)
        # dense reference: explicit dummy columns for the fixed effects, plain BFGS
        Z = np.column_stack([Xg[:, u, v].T, np.eye(n)[u], np.eye(n)[v]])
        y = Y[u, v]

        def obj(w):
            e = Z @ w
            pen = 0.5 * (w[p:] @ w[p:])
            return np.logaddexp(0, e).sum() - y @ e + pen, Z.T @ (1 / (1 + np.exp(-e)) - y) + np.r_[np.zeros(p), w[p:]]

        ref = minimize(obj, np.zeros(p + 2 * n), jac=True, method="BFGS", options={"gtol": 1e-10, "maxiter": 10000})
        np.testing.assert_allclose(f["coef"], ref.x[:p], atol=1e-5)
        np.testing.assert_allclose(f["alpha"], ref.x[p:p + n], atol=1e-5)
        np.testing.assert_allclose(f["beta"], ref.x[p + n:], atol=1e-5)
        self.assertTrue(f["converged"])
        # out-of-fold machinery: masking a dyad removes it from the fit
        M2 = M.copy()
        M2[0, 1] = 0
        f2 = premise.fit_fe_logit(Xg, Y, M2)
        keep = ~((u == 0) & (v == 1))
        Z2, y2 = Z[keep], y[keep]

        def obj2(w):
            e = Z2 @ w
            return (np.logaddexp(0, e).sum() - y2 @ e + 0.5 * (w[p:] @ w[p:]),
                    Z2.T @ (1 / (1 + np.exp(-e)) - y2) + np.r_[np.zeros(p), w[p:]])

        ref2 = minimize(obj2, np.zeros(p + 2 * n), jac=True, method="BFGS", options={"gtol": 1e-10, "maxiter": 10000})
        np.testing.assert_allclose(f2["coef"], ref2.x[:p], atol=1e-5)

    def test_permute_within_strata(self):
        rng = np.random.default_rng(0)
        vals = np.arange(10.0)
        strata = np.array(list("aaabbbbccc"))
        out = premise.permute_within(vals, strata, rng)
        for s in "abc":
            self.assertEqual(sorted(out[strata == s]), sorted(vals[strata == s]))
        again = premise.permute_within(vals, strata, np.random.default_rng(0))
        np.testing.assert_array_equal(out, again)

    def test_partial_rank_corr_and_decision(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=200)
        cat = np.where(rng.random(200) < 0.5, "a", "b")
        z = rng.normal(size=200)
        self.assertAlmostEqual(premise._partial_rank_corr(x, x, cat, z), 1.0)
        self.assertEqual(premise.decide(0.3, 0.01, [-0.01, -0.001])["code"], "predicts")
        self.assertEqual(premise.decide(0.3, 0.01, [-0.01, 0.002])["code"], "direction_only")
        self.assertEqual(premise.decide(0.3, 0.2, [-0.01, -0.001])["code"], "no_effect")
        self.assertEqual(premise.decide(-0.3, 0.01, [-0.01, -0.001])["code"], "no_effect")
        self.assertEqual(premise.decide(0.3, 0.01, [0.001, 0.01])["code"], "no_effect")


if __name__ == "__main__":
    unittest.main()
