"""Loaders: label-map inference, time parsing, text cleaning, DEBAGREEMENT and
ConvoKit-format CMV parsing on small fabricated files.

Run: python -m unittest tests.test_data -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lacs.config import load_config  # noqa: E402
from lacs.data import cmv, debagreement  # noqa: E402
from lacs.data.text import clean_text  # noqa: E402


class TestHelpers(unittest.TestCase):
    def test_label_map_inference_recovers_permutation(self):
        rng = np.random.default_rng(0)
        hidden = {0: 2, 1: 0, 2: 1}  # raw value -> canonical (0 agree, 1 neutral, 2 disagree)
        inv = {v: k for k, v in hidden.items()}
        raw, forum = [], []
        for name, shares in debagreement.TABLE1.items():
            n = 3000
            canon = rng.choice(3, size=n, p=shares)
            raw += [inv[c] for c in canon]
            forum += [name.capitalize()] * n
        m, err, gap = debagreement.infer_label_map(pd.Series(raw), pd.Series(forum))
        self.assertEqual(m, hidden)
        self.assertLess(err, 0.02)
        self.assertGreater(gap, 0.02)

    def test_string_labels(self):
        s = pd.Series(["Agree", "neutral", "DISAGREE", "positive", "negative"])
        out = debagreement.to_canonical_labels(s, pd.Series(["x"] * 5), None)
        self.assertEqual(out.tolist(), [0, 1, 2, 0, 2])

    def test_parse_time(self):
        np.testing.assert_allclose(debagreement.parse_time(pd.Series([1.6e9, 1.6e9 + 5])), [1.6e9, 1.6e9 + 5])
        np.testing.assert_allclose(debagreement.parse_time(pd.Series([1.6e12, 1.6e12 + 5000])), [1.6e9, 1.6e9 + 5])
        t = debagreement.parse_time(pd.Series(["2020-01-01 00:00:00", "2020-01-02"]))
        self.assertAlmostEqual(t[1] - t[0], 86400.0)

    def test_clean_text(self):
        s = "&gt; quoted line\nI disagree, see [link](http://x.com) and https://y.org &amp; more"
        self.assertEqual(clean_text(s), "I disagree, see link and & more")
        self.assertEqual(clean_text("[deleted]"), "")


class TestDebagreementLoader(unittest.TestCase):
    def test_load_small_file(self):
        rows = [
            # parent p1 (by alice) is only ever a parent -> upper-bound time = earliest reply time
            {"msg_id_parent": "p1", "msg_id_child": "c1", "submission_id": "s1", "body_parent": "brexit is good",
             "body_child": "no it is not", "author_parent": "alice", "author_child": "bob",
             "subreddit": "Brexit", "datetime": "2020-01-02", "label": "disagree"},
            {"msg_id_parent": "p1", "msg_id_child": "c2", "submission_id": "s1", "body_parent": "brexit is good",
             "body_child": "agreed", "author_parent": "alice", "author_child": "carol",
             "subreddit": "Brexit", "datetime": "2020-01-01", "label": "agree"},
            # c1 appears later as a parent: its exact time comes from its own reply row
            {"msg_id_parent": "c1", "msg_id_child": "c3", "submission_id": "s1", "body_parent": "no it is not",
             "body_child": "hmm", "author_parent": "bob", "author_child": "carol",
             "subreddit": "Brexit", "datetime": "2020-01-05", "label": "neutral"},
            # dropped: bot author, self reply
            {"msg_id_parent": "p9", "msg_id_child": "c9", "submission_id": "s2", "body_parent": "x",
             "body_child": "y", "author_parent": "AutoModerator", "author_child": "bob",
             "subreddit": "Brexit", "datetime": "2020-01-03", "label": "agree"},
            {"msg_id_parent": "c3", "msg_id_child": "c4", "submission_id": "s1", "body_parent": "hmm",
             "body_child": "me again", "author_parent": "carol", "author_child": "carol",
             "subreddit": "Brexit", "datetime": "2020-01-06", "label": "agree"},
        ]
        with tempfile.TemporaryDirectory() as td:
            pd.DataFrame(rows).to_csv(Path(td) / "brexit.csv", index=False)
            cfg = load_config(None, [f"dataset.raw={td}"])
            comments, pairs, info = debagreement.load(cfg)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(sorted(pairs["label"].tolist()), [0, 1, 2])
        c = comments.set_index("cid")
        day = 86400.0
        t0 = pd.Timestamp("2020-01-01", tz="UTC").timestamp()
        self.assertTrue(c.loc["p1", "time_is_upper"])
        self.assertAlmostEqual(c.loc["p1", "time"], t0)               # earliest reply to p1
        self.assertFalse(c.loc["c1", "time_is_upper"])
        self.assertAlmostEqual(c.loc["c1", "time"], t0 + day)          # its own (exact) time
        self.assertEqual(c.loc["c3", "parent"], "c1")
        self.assertEqual(set(pairs["forum"]), {"brexit"})


class TestCMVLoader(unittest.TestCase):
    def test_convokit_format(self):
        utts = [
            {"id": "t1", "speaker": "op", "conversation_id": "t1", "reply_to": None, "timestamp": 100,
             "text": "CMV: cats are better", "meta": {"success": None}},
            {"id": "a", "speaker": "u1", "conversation_id": "t1", "reply_to": "t1", "timestamp": 110,
             "text": "dogs are loyal", "meta": {"success": 1}},
            {"id": "b", "speaker": "u2", "conversation_id": "t1", "reply_to": "t1", "timestamp": 120,
             "text": "cats scratch", "meta": {"success": 0}},
            {"id": "c", "speaker": "op", "conversation_id": "t1", "reply_to": "a", "timestamp": 130,
             "text": "good point, delta", "meta": {"success": 1}},
            {"id": "d", "speaker": "u3", "conversation_id": "t1", "reply_to": "t1", "timestamp": 140,
             "text": "no opinion", "meta": {"success": None}},
        ]
        convs = {"t1": {"meta": {"op-userID": "op", "op-title": "CMV: cats are better", "op-text-body": "..."}}}
        with tempfile.TemporaryDirectory() as td:
            with open(Path(td) / "utterances.jsonl", "w") as f:
                for u in utts:
                    f.write(json.dumps(u) + "\n")
            (Path(td) / "conversations.json").write_text(json.dumps(convs))
            cfg = load_config(None, ["dataset.type=cmv", f"dataset.raw={td}"])
            comments, pairs, _ = cmv.load(cfg)
        self.assertEqual(len(pairs), 2)                     # only top-level replies with success in {0,1}
        p = pairs.set_index("src_user")
        self.assertEqual(p.loc["u1", "label"], 0)           # delta -> accept
        self.assertEqual(p.loc["u2", "label"], 1)
        self.assertTrue((pairs["dst_user"] == "op").all())
        self.assertTrue((pairs["dst_cid"] == "t1").all())
        self.assertEqual(len(comments), 5)


if __name__ == "__main__":
    unittest.main()
