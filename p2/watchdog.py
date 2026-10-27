"""Engine runs with a stall watchdog.

`run_with_watchdog` is lacs.search.cpp.CppSearch.run with the same command lines, the same split of the
queries over processes and the same row order, so its rows are identical; the only differences:
  * each engine process is polled; the engine writes and flushes one line per query (all its algorithms),
    so a process whose output file has not grown for `stall_seconds(...)` is a hang: every process of the
    call is killed and RuntimeError is raised (nothing is cached for the block, a rerun redoes it);
  * a process that exits with an error is reported at once: the others are killed and RuntimeError is raised
    (instead of waiting for the rest of the block);
  * stderr goes to a file next to the output instead of a pipe (no pipe can fill up while polling).
`install()` puts it in place of CppSearch.run for everything in this process (geo._run_blocks,
p0kit.bridge.engine_run, ...).
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Iterable, Sequence

STARTUP_SECONDS = float(os.environ.get("P2_WATCHDOG_STARTUP", 1800))  # graph loading and index build (Orkut)
POLL_SECONDS = float(os.environ.get("P2_WATCHDOG_POLL", 2.0))


def stall_seconds(algos: Sequence[str], time_limit: float) -> float:
    """Longest time one query may take before the watchdog acts: per algorithm the search limit, the
    bound pass after a timeout (max(5 s, 10% of the limit), as in the engine) and a minute of slack for
    the diameter check and output; plus the start-up allowance."""
    tl = float(time_limit)
    per_algo = tl + max(5.0, 0.1 * tl) + 60.0
    return STARTUP_SECONDS + len(list(algos)) * per_algo


def run_with_watchdog(self, queries: Iterable[int], algos: Sequence[str], k: int, theta: str = "acc", *,
                      time_limit: float = 600.0, members: bool = True, extra: Sequence[str] = (),
                      r: float | None = None, tag: str = "run", n_proc: int = 1,
                      r_per_query: Sequence[float] | None = None, size_per_query: Sequence[int] | None = None,
                      _stall: float | None = None) -> list[dict]:
    import json

    queries = [int(q) for q in queries]
    for name, vals in (("r_per_query", r_per_query), ("size_per_query", size_per_query)):
        if vals is not None and len(vals) != len(queries):
            raise ValueError(f"{name} needs one value per query")
    lines = []
    for i, q in enumerate(queries):
        parts = [str(q)]
        if r_per_query is not None or size_per_query is not None:
            parts.append(repr(float(r_per_query[i])) if r_per_query is not None else "-")
        if size_per_query is not None:
            parts.append(str(int(size_per_query[i])))
        lines.append(" ".join(parts))
    n_proc = max(1, min(int(n_proc), len(queries)))
    chunks = [lines[i::n_proc] for i in range(n_proc)] if queries else [[]]
    limit = float(_stall) if _stall is not None else stall_seconds(algos, time_limit)
    procs = []
    try:
        for ci, chunk in enumerate(chunks):
            suffix = f"_{ci}" if n_proc > 1 else ""
            qpath = self.workdir / f"queries_{tag}{suffix}.txt"
            qpath.write_text("\n".join(chunk) + "\n")
            out_path = self.workdir / f"out_{tag}{suffix}.jsonl"
            err_path = self.workdir / f"err_{tag}{suffix}.txt"
            cmd = [str(self.exe), "--graph", str(self.graph_path), "--pos", str(self.pos_path), "--theta",
                   str(self.theta_paths[theta]), "--queries", str(qpath), "--algo", ",".join(algos), "--k",
                   str(k), "--time-limit", str(time_limit), "--out", str(out_path)]
            if r is not None:
                cmd += ["--r", repr(float(r))]
            if not members:
                cmd.append("--no-members")
            cmd += list(extra)
            if out_path.exists():
                out_path.unlink()
            errf = open(err_path, "w")
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=errf, text=True)
            procs.append({"p": p, "out": out_path, "err": err_path, "errf": errf, "size": -1,
                          "t_last": time.monotonic()})
        while True:
            alive = 0
            now = time.monotonic()
            for d in procs:
                rc = d["p"].poll()
                if rc is None:
                    alive += 1
                    sz = d["out"].stat().st_size if d["out"].exists() else 0
                    if sz != d["size"]:
                        d["size"], d["t_last"] = sz, now
                    elif now - d["t_last"] > limit:
                        raise RuntimeError(f"engine stalled: no output for {now - d['t_last']:.0f} s "
                                           f"(limit {limit:.0f} s) in {d['out']}; all processes of '{tag}' killed")
                elif rc != 0:  # fail fast: the block is lost anyway, do not wait for the other processes
                    d["errf"].flush()
                    err = Path(d["err"]).read_text(errors="replace").strip()
                    raise RuntimeError(f"lacs_search failed ({rc}) in '{tag}': {err[-2000:]}; "
                                       f"the other processes of '{tag}' were killed")
            if alive == 0:
                break
            time.sleep(POLL_SECONDS)
        rows = []
        for d in procs:
            d["errf"].close()
            if d["p"].returncode != 0:
                err = Path(d["err"]).read_text(errors="replace").strip()
                raise RuntimeError(f"lacs_search failed ({d['p'].returncode}): {err}")
            with open(d["out"]) as f:
                rows += [json.loads(line) for line in f if line.strip()]
    finally:
        for d in procs:
            if d["p"].poll() is None:
                d["p"].kill()
                try:
                    d["p"].wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
            if not d["errf"].closed:
                d["errf"].close()
    order = {q: i for i, q in enumerate(queries)}
    rows.sort(key=lambda row: order.get(row["q"], 0))
    return rows


def install() -> None:
    from lacs.search.cpp import CppSearch

    if getattr(CppSearch.run, "__name__", "") != "run_with_watchdog":
        CppSearch._run_plain = CppSearch.run
        CppSearch.run = run_with_watchdog
