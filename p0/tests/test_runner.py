"""The E3 task runner survives solver calls that hang or crash.

Run: python tests/test_runner.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from p0kit.runner_common import run_tasks_watchdog  # noqa: E402


def _fake_solver(shared, key, solver):
    if solver == "hang":
        threading.Event().wait()  # never returns (stands for a deadlocked solver)
    if solver == "die" and key % 2 == 0:
        os._exit(3)  # stands for a segfault / OOM kill
    if solver == "slow":
        time.sleep(0.3)
    return [{"key": key, "solver": solver, "status": "ok", "base": shared["base"]}]


def test_watchdog_kills_hung_and_dead_workers_and_finishes_the_rest():
    tasks = [(i, ["ok", "hang", "slow", "die", "ok2"]) for i in range(6)]
    rows = []

    def on_hang(key, solver, secs, detail):
        return [{"key": key, "solver": solver, "status": "error", "detail": detail}]

    hang_dir = tempfile.mkdtemp(prefix="p0_hang_")
    t0 = time.time()
    stats = run_tasks_watchdog({"base": 7}, _fake_solver, tasks, workers=3, limit_of=lambda s: 2.5,
                               on_rows=rows.extend, on_hang=on_hang, label="test", log_every=0,
                               hang_dir=hang_dir, max_hangs_per_solver=100, verbose=False)
    took = time.time() - t0
    pairs = [(r["key"], r["solver"]) for r in rows]
    assert sorted(pairs) == sorted((k, s) for k, sv in tasks for s in sv), pairs
    assert len(set(pairs)) == len(pairs)
    by = {(r["key"], r["solver"]): r for r in rows}
    for k, _ in tasks:
        assert by[(k, "ok")]["status"] == "ok" and by[(k, "ok")]["base"] == 7
        assert by[(k, "ok2")]["status"] == "ok" and by[(k, "slow")]["status"] == "ok"
        h = by[(k, "hang")]
        assert h["status"] == "error" and h["detail"].startswith("hung"), h
        assert "_fake_solver" in h["detail"], h["detail"][:500]  # stack of the hung call
        d = by[(k, "die")]
        assert d["status"] == ("error" if k % 2 == 0 else "ok"), d
    assert stats["hung"] == 6 and stats["died"] == 3, stats
    assert took < 60, took


def test_watchdog_disables_a_solver_that_keeps_hanging():
    tasks = [(i, ["hang", "ok"]) for i in range(12)]
    rows = []

    def on_hang(key, solver, secs, detail):
        return [{"key": key, "solver": solver, "status": "error", "detail": detail}]

    t0 = time.time()
    stats = run_tasks_watchdog({"base": 1}, _fake_solver, tasks, workers=2, limit_of=lambda s: 1.5,
                               on_rows=rows.extend, on_hang=on_hang, label="test", log_every=0,
                               max_hangs_per_solver=2, verbose=False)
    pairs = sorted((r["key"], r["solver"]) for r in rows)
    assert pairs == sorted((k, s) for k, sv in tasks for s in sv), pairs
    assert all(r["status"] == "ok" for r in rows if r["solver"] == "ok")
    dis = [r for r in rows if r["solver"] == "hang" and r["detail"].startswith("disabled")]
    assert stats["disabled"] == ["hang"] and len(dis) >= 8 and stats["hung"] <= 3, (stats, len(dis))
    assert time.time() - t0 < 30


def run_all():
    for name in [k for k in globals() if k.startswith("test_")]:
        globals()[name]()
        print("ok", name)


if __name__ == "__main__":
    run_all()
