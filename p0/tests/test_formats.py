"""Checkpoint files survive an interrupted run (recover_jsonl / rewrite_jsonl).

Run: python tests/test_formats.py
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p0kit.formats import JsonlWriter, read_jsonl, recover_jsonl, rewrite_jsonl  # noqa: E402


def test_recover_truncated_gzip_and_append():
    d = tempfile.mkdtemp(prefix="p0_fmt_")
    p = os.path.join(d, "rows.jsonl.gz")
    w = JsonlWriter(p, append=True)
    w.write_many([{"i": i} for i in range(5)])
    w.close()
    w = JsonlWriter(p, append=True)
    w.write_many([{"i": i} for i in range(5, 9)])
    w.close()
    w = JsonlWriter(p, append=True)  # a third session that is killed: never closed
    w.write_many([{"i": i, "pad": "y" * 50} for i in range(9, 30)])
    full = open(p, "rb").read()
    for cut in sorted({len(full), len(full) - 3, len(full) - 40, len(full) // 2, 25}, reverse=True):
        q = os.path.join(d, f"cut{cut}.jsonl.gz")
        with open(q, "wb") as fh:
            fh.write(full[:cut])
        recs, info = recover_jsonl(q)
        ids = [r["i"] for r in recs]
        assert ids == list(range(len(ids))), (cut, ids)
        assert info["truncated"]
        rewrite_jsonl(q, recs)
        assert [r["i"] for r in read_jsonl(q)] == ids
        w2 = JsonlWriter(q, append=True)
        w2.write_many([{"i": 1000}])
        w2.close()
        assert [r["i"] for r in read_jsonl(q)] == ids + [1000]
        assert not recover_jsonl(q)[1]["truncated"]
    w.close()


def test_recover_plain_partial_line():
    d = tempfile.mkdtemp(prefix="p0_fmt_")
    q = os.path.join(d, "rows.jsonl")
    with open(q, "w") as fh:
        fh.write('{"i": 0}\n{"i": 1}\n{"i": 2')
    recs, info = recover_jsonl(q)
    assert [r["i"] for r in recs] == [0, 1] and info["truncated"]


def run_all():
    for name in [k for k in globals() if k.startswith("test_")]:
        globals()[name]()
        print("ok", name)


if __name__ == "__main__":
    run_all()
