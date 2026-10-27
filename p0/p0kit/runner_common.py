"""Shared plumbing for the experiment drivers: config loading, settings,
manifests, process pools that share the read-only index by fork."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import platform
import socket
import sys
import time
from typing import Callable, Dict, Iterable, List

import numpy as np

from . import __version__, ffi

KIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_yaml(path: str) -> Dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def kit_digest() -> str:
    """sha256 over the kit's source files (C++ and Python), for the ledger."""
    h = hashlib.sha256()
    for sub in ("cpp", "p0kit"):
        d = os.path.join(KIT_ROOT, sub)
        for name in sorted(os.listdir(d)):
            if name.endswith((".cpp", ".hpp", ".py")):
                with open(os.path.join(d, name), "rb") as f:
                    h.update(name.encode())
                    h.update(f.read())
    return h.hexdigest()


def host_info() -> Dict:
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
        "kit_version": __version__,
        "kit_sha256": kit_digest(),
        "lib_versions": ffi.versions(),
    }
    try:
        import scipy

        info["scipy"] = scipy.__version__
    except ImportError:
        pass
    info["ortools"] = ortools_version()  # metadata only: never initialise solvers in a parent that forks
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as f:
            info["mem_total_kb"] = int(f.readline().split()[1])
    except (OSError, ValueError, IndexError):
        pass
    try:
        info["loadavg_container"] = open("/proc/loadavg").read().split()[:3]
    except OSError:
        pass
    return info


def ortools_version():
    """Installed OR-Tools version or None, without importing the package.

    Parents that fork workers must not initialise OR-Tools (or run HiGHS): their
    worker threads do not survive fork() and a solver called in the child can
    then wait forever on them."""
    import importlib.util

    if importlib.util.find_spec("ortools") is None:
        return None
    try:
        from importlib.metadata import version

        return version("ortools")
    except Exception:
        return "unknown"


def write_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


# ---------------------------------------------------------------- process pool
_SHARED: Dict = {}


def _call(args):
    fn, item = args
    return fn(_SHARED, item)


def run_pool(shared: Dict, fn: Callable, items: Iterable, workers: int, log_every: int = 20,
             label: str = "", on_result: Callable = None) -> List:
    """Map fn(shared, item) over items. With workers > 1 the shared dict (e.g. a
    large read-only index) is inherited by fork, not pickled. If on_result is
    given, results are streamed to it (in completion order) and not kept."""
    items = list(items)
    out = []
    t0 = time.time()

    def emit(i, r):
        if on_result is not None:
            on_result(r)
        else:
            out.append(r)
        if log_every and i % log_every == 0:
            print(f"[{label}] {i}/{len(items)} done, {time.time() - t0:.0f}s", flush=True)

    if workers <= 1 or len(items) <= 1:
        for i, it in enumerate(items, 1):
            emit(i, fn(shared, it))
        return out
    global _SHARED
    _SHARED = shared
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_call, [(fn, it) for it in items], chunksize=1), 1):
            emit(i, r)
    _SHARED = {}
    return out


# ------------------------------------------------------- watchdog task runner
def _watch_worker(conn, fn, hang_dir):
    """Child side: run the solvers of each task one by one and report each row."""
    import faulthandler

    shared = _SHARED
    pid = os.getpid()
    trace = open(os.path.join(hang_dir, f"stack_{pid}.txt"), "w") if hang_dir else None
    try:
        while True:
            task = conn.recv()
            if task is None:
                break
            key, solvers, limit = task
            for s in solvers:
                conn.send(("start", key, s))
                if trace is not None:
                    trace.seek(0)
                    trace.truncate()
                    faulthandler.dump_traceback_later(max(1.0, limit - 2.0), repeat=False, file=trace)
                rows = fn(shared, key, s)
                if trace is not None:
                    faulthandler.cancel_dump_traceback_later()
                conn.send(("rows", key, s, rows))
            conn.send(("done", key))
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        if trace is not None:
            trace.close()
        conn.close()
        os._exit(0)


def run_tasks_watchdog(shared: Dict, fn: Callable, tasks: List, workers: int, limit_of: Callable,
                       on_rows: Callable, on_hang: Callable, label: str = "", log_every: int = 20,
                       hang_dir: str = None, max_hangs_per_solver: int = 3, verbose: bool = True) -> Dict:
    """Run tasks (key, [solvers]) on forked workers with a hard time limit per solver call.

    fn(shared, key, solver) -> list of rows (runs in the child). limit_of(solver) gives the
    hard limit in seconds. A worker that stays inside one solver call longer than that
    (a hang, e.g. a solver deadlocked on threads lost across fork) or that dies is killed
    and replaced; on_hang(key, solver, seconds, detail) must return the error rows for that
    call, and the rest of the task is re-queued. A solver that hung max_hangs_per_solver
    times is not started again (its remaining calls get error rows "disabled"), so a
    systematic hang costs minutes, not hours. The parent must stay free of solver state:
    workers are forked from it, also the replacements."""
    from collections import deque
    from multiprocessing.connection import wait

    global _SHARED
    _SHARED = shared
    ctx = mp.get_context("fork")
    if hang_dir:
        os.makedirs(hang_dir, exist_ok=True)
    queue = deque((k, list(sv)) for k, sv in tasks if sv)
    n_calls = sum(len(sv) for _, sv in queue)
    stats = {"calls": n_calls, "done": 0, "hung": 0, "died": 0}
    t0 = time.time()
    pool = {}  # conn -> state
    retries = {}
    hangs = {}
    disabled = set()

    def say(msg):
        if verbose:
            print(msg, flush=True)

    def spawn():
        a, b = ctx.Pipe(duplex=True)
        p = ctx.Process(target=_watch_worker, args=(b, fn, hang_dir), daemon=True)
        p.start()
        b.close()
        pool[a] = {"proc": p, "task": None, "solver": None, "since": None, "left": []}
        return a

    def dispatch(conn):
        st = pool[conn]
        while queue:
            key, sv = queue.popleft()
            for s_ in [x for x in sv if x in disabled]:
                on_rows(on_hang(key, s_, 0.0, f"disabled: {s_} hung {hangs.get(s_, 0)} times earlier in this run"))
                bump()
            sv = [x for x in sv if x not in disabled]
            if sv:
                st.update(task=key, left=list(sv), solver=None, since=None)
                conn.send((key, sv, max(limit_of(x) for x in sv)))
                return
        st["task"] = None

    def bump(k=1):
        before = stats["done"]
        stats["done"] += k
        if log_every and (stats["done"] // log_every) > (before // log_every):
            print(f"[{label}] {stats['done']}/{n_calls} solver runs done, {time.time() - t0:.0f}s", flush=True)

    def kill(conn, why):
        st = pool.pop(conn)
        try:
            st["proc"].kill()
        except Exception:
            pass
        st["proc"].join(5)
        try:
            conn.close()
        except Exception:
            pass
        detail = ""
        if hang_dir:
            p = os.path.join(hang_dir, f"stack_{st['proc'].pid}.txt")
            try:
                detail = open(p).read()[-4000:]
                os.remove(p)
            except OSError:
                pass
        if st["task"] is not None:
            left = list(st["left"])
            key = st["task"]
            stats["hung" if why == "hung" else "died"] += 1
            if left and st["solver"] is None:
                # died between two solver calls (e.g. killed from outside): retry the rest twice
                retries[key] = retries.get(key, 0) + 1
                say(f"[{label}] WARNING: worker {why} on {key} between solver calls (retry {retries[key]})")
                if retries[key] <= 2:
                    queue.appendleft((key, left))
                    return
                bad_list = left
            else:
                bad_list = [st["solver"]] if left else []
            secs = time.time() - st["since"] if st["since"] else 0.0
            for bad in bad_list:
                say(f"[{label}] WARNING: {why} worker killed during {bad} on {key} after {secs:.0f}s")
                on_rows(on_hang(key, bad, secs, (why + ": " + detail.strip())[:2000]))
                bump()
                if why == "hung":
                    hangs[bad] = hangs.get(bad, 0) + 1
                    if hangs[bad] >= max_hangs_per_solver and bad not in disabled:
                        disabled.add(bad)
                        say(f"[{label}] WARNING: {bad} hung {hangs[bad]} times: not started again in this run")
            rest = [x for x in left if x not in bad_list]
            if rest:
                queue.appendleft((key, rest))

    try:
        for _ in range(max(1, min(workers, len(queue)))):
            dispatch(spawn())
        while any(st["task"] is not None for st in pool.values()):
            ready = wait(list(pool), timeout=1.0)
            for conn in ready:
                st = pool.get(conn)
                if st is None:
                    continue
                try:
                    msg = conn.recv()
                except (EOFError, OSError):
                    kill(conn, "died")
                    if queue:
                        dispatch(spawn())
                    continue
                if msg[0] == "start":
                    st["solver"], st["since"] = msg[2], time.time()
                elif msg[0] == "rows":
                    on_rows(msg[3])
                    bump()
                    if msg[2] in st["left"]:
                        st["left"].remove(msg[2])
                    st["solver"], st["since"] = None, None
                elif msg[0] == "done":
                    dispatch(conn)
            now = time.time()
            for conn, st in list(pool.items()):
                if st["solver"] is not None and now - st["since"] > limit_of(st["solver"]):
                    kill(conn, "hung")
                    if queue:
                        dispatch(spawn())
                elif st["task"] is not None and not st["proc"].is_alive():
                    kill(conn, "died")
                    if queue:
                        dispatch(spawn())
            if queue and len(pool) < workers:
                dispatch(spawn())
    finally:
        for conn, st in list(pool.items()):
            try:
                conn.send(None)
            except Exception:
                pass
        for conn, st in list(pool.items()):
            st["proc"].join(10)
            if st["proc"].is_alive():
                st["proc"].kill()
            try:
                conn.close()
            except Exception:
                pass
        _SHARED = {}
    stats["seconds"] = time.time() - t0
    stats["disabled"] = sorted(disabled)
    return stats
