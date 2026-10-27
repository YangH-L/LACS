"""P2-X: the one missing cell of Table I -- median time to extract q's connected k-core component on Friendster.

W2 skipped this reference on Friendster (one traversal of a ~65M-vertex component per query). This script runs
only that algorithm ("kcore" of the W2 engine, sources checked byte for byte against W2) on the same 200
Friendster queries as W2 (k = 3), after a timing check on Pokec against W2's own k-core times.

  python p2x/fkcore.py launch      # start detached (no nohup needed); prints the PID and returns
  python p2x/fkcore.py run         # the same in the foreground
  python p2x/fkcore.py status      # one-line progress

Stages (all re-runnable; Friendster resumes from the queries already written):
  env        machine, memory, disk, load; no other lacs_search may be running
  build      copy the W2 engine sources, check their sha256, compile lacs_search (g++ -O3 -march=native)
  inputs     W2's converted graphs (header n / nnz); a missing Friendster graph is converted again (~15 min)
  pokec      timing check: W2's 200 Pokec queries, k-core sizes must equal W2's, median time within 0.67-1.5x
             of W2's (on another machine, P2X_SKIP_TIMING=1 keeps the size check and skips the time band)
  friendster the 200 W2 Friendster queries, one engine process, results flushed after every query
  summary    p2x_summary.json
  bundle     runs_p2/p2x_bundle_<time>.tar.gz (summary, per-query q/size/status/time, logs; no graph data)

Output only under OUT (default runs_p2/p2x). Inputs under W2_ROOT (default: this directory, where
scripts/run_w2.sh left runs/efficiency_w2_graphs) are only read. Python standard library only.
The log ends with "P2X END rc=<code> ..." when the run is over.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
W2 = Path(os.environ.get("W2_ROOT", "."))
OUT = Path(os.environ.get("OUT", "runs_p2/p2x"))
REF = Path(os.environ.get("P2X_REF", HERE / "w2_reference.json"))
LOCK = OUT.parent / ".run_p2.lock"  # the same lock as run_p2.sh: never two runs at once
LOG = OUT / "logs" / "p2x.log"
VERSION = "p2x v0.2"
MIN_MEM_GB = float(os.environ.get("P2X_MIN_MEM_GB", 30))      # Friendster: about 15 GB graph + 2 GB arrays
MIN_DISK_GB = float(os.environ.get("P2X_MIN_DISK_GB", 25))    # only when the graph must be converted again
TIMING_BAND = (0.67, 1.5)
SKIP_TIMING = os.environ.get("P2X_SKIP_TIMING") == "1"  # other machines: keep the size check, skip the timing band
WALL_LIMIT_H = float(os.environ.get("P2X_WALL_LIMIT_H", 9))   # watchdog for the Friendster engine process
MAX_LOAD = float(os.environ.get("P2X_MAX_LOAD", 2.0))
QDIR = Path(os.environ.get("P2X_QDIR", HERE))                  # query lists (overridden only in the rehearsal)
CURRENT: list = []                                             # the running engine process, if any


def say(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def stop(code: int, msg: str):
    say(f"!! {msg}")
    raise SystemExit(code)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def graph_header(p: Path):
    with open(p, "rb") as f:
        if f.read(8) != b"LACSGR01":
            return None
        n, nnz = struct.unpack("<qq", f.read(16))
    return n, nnz


def write_const_matrix(p: Path, n: int, value: float) -> None:
    """LACSMT01 n x 1 float32 matrix filled with one value (k-core ignores positions and tolerances,
    but the engine requires both files)."""
    if p.exists() and p.stat().st_size == 24 + 4 * n:
        return
    tmp = p.with_suffix(".tmp")
    block = struct.pack("<f", value) * (1 << 20)
    with open(tmp, "wb") as f:
        f.write(b"LACSMT01" + struct.pack("<qq", n, 1))
        left = n
        while left > 0:
            c = min(left, 1 << 20)
            f.write(block[: 4 * c])
            left -= c
    tmp.replace(p)


def mem_available_gb() -> float:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024 / 1024
    return 0.0


def engine_running() -> list[str]:
    out = subprocess.run(["ps", "-C", "lacs_search,lacs_convert", "-o", "pid=,stat=,etime=,args="], capture_output=True,
                         text=True).stdout
    return [l for l in out.splitlines() if l.strip() and "Z" not in l.split()[1]]


# ---------------------------------------------------------------------------------------------- stages
def st_env(ref):
    say(f"{VERSION}; W2_ROOT={W2}; OUT={OUT}; python {sys.version.split()[0]}")
    say(f"nproc={os.cpu_count()} loadavg={open('/proc/loadavg').read().split()[:3]} "
        f"MemAvailable={mem_available_gb():.1f} GB; disk free under OUT={shutil.disk_usage(OUT).free / 2**30:.0f} GB")
    busy = engine_running()
    if busy:
        stop(3, "another lacs_search is running; wait until it ends:\n  " + "\n  ".join(busy))
    if mem_available_gb() < MIN_MEM_GB:
        stop(3, f"MemAvailable {mem_available_gb():.1f} GB < {MIN_MEM_GB} GB needed for Friendster")
    load1 = float(open("/proc/loadavg").read().split()[0])
    if load1 > MAX_LOAD:
        stop(3, f"load average {load1} > {MAX_LOAD}: other heavy work is running, times would not be comparable")
    say("ENV OK")


def st_build(ref):
    src_dir, dst = W2 / "cpp" / "src", OUT / "engine"
    (dst / "src").mkdir(parents=True, exist_ok=True)
    bad = []
    for name, want in sorted(ref["engine_sources_sha256"].items()):
        got = sha256(src_dir / name) if (src_dir / name).exists() else "missing"
        if got != want:
            bad.append(f"{name}: {got}")
        shutil.copy2(src_dir / name, dst / "src" / name) if got != "missing" else None
    if bad:
        stop(4, "W2 engine sources differ from the W2 run:\n  " + "\n  ".join(bad))
    exe = dst / "lacs_search"
    cxx = shutil.which("g++") or stop(4, "g++ not found")
    cmd = [cxx, "-std=c++17", "-O3", "-march=native", "-pthread", str(dst / "src" / "lacs_search.cpp"), "-o", str(exe)]
    say("build: " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    say(f"BUILD OK (engine sources identical to W2; {subprocess.run([cxx, '--version'], capture_output=True, text=True).stdout.splitlines()[0]})")


def friendster_graph(ref) -> Path:
    n, m = ref["friendster"]["n"], ref["friendster"]["m"]
    w2g = W2 / "runs" / "efficiency_w2_graphs" / "work" / "e3_friendster" / "graph.bin"
    own = OUT / "friendster_graph.bin"
    for g in (w2g, own):
        if g.exists() and graph_header(g) == (n, 2 * m):
            return g
    return Path()


def st_inputs(ref):
    work = W2 / "runs" / "efficiency_w2_graphs" / "work"
    pg = work / "e3_pokec" / "graph.bin"
    pn, pm = ref["pokec"]["n"], ref["pokec"]["m"]
    if not pg.exists() or graph_header(pg) != (pn, 2 * pm):
        stop(5, f"Pokec graph of W2 missing or different: {pg} header {graph_header(pg) if pg.exists() else None}")
    say(f"pokec graph: {pg} (n={pn}, m={pm}) OK")
    g = friendster_graph(ref)
    if g == Path():
        n, m = ref["friendster"]["n"], ref["friendster"]["m"]
        edges = Path(json.loads((work / "e3_friendster" / "graph.src.json").read_text())["edges"]) \
            if (work / "e3_friendster" / "graph.src.json").exists() else W2 / "data/raw/snap/com-friendster.ungraph.txt"
        say(f"Friendster graph of W2 not found or different; converting {edges} again (W2 took 761 s)")
        if not edges.exists():
            stop(5, f"neither the converted Friendster graph nor the edge list {edges} exists")
        for stale in (OUT / "friendster_graph.tmp", OUT / "friendster_ids.txt"):  # left by an interrupted run
            stale.unlink(missing_ok=True)
        free = shutil.disk_usage(OUT).free / 2**30
        if free < MIN_DISK_GB:
            stop(5, f"only {free:.0f} GB free under {OUT}; converting needs {MIN_DISK_GB} GB")
        conv = OUT / "engine" / "lacs_convert"
        subprocess.run([shutil.which("g++"), "-std=c++17", "-O3", "-march=native", "-pthread",
                        str(OUT / "engine" / "src" / "lacs_convert.cpp"), "-o", str(conv)], check=True)
        own = OUT / "friendster_graph.bin"
        t0 = time.time()
        cp = subprocess.Popen([str(conv), "--in", str(edges), "--out", str(own.with_suffix(".tmp")), "--relabel",
                               str(OUT / "friendster_ids.txt"), "--lowmem"], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True)
        CURRENT[:] = [cp]
        so, se = cp.communicate()
        if cp.returncode != 0:
            stop(5, f"lacs_convert exit {cp.returncode}: {se.strip()[-2000:]}")
        own.with_suffix(".tmp").replace(own)
        say(f"converted in {time.time() - t0:.0f} s: {so.strip().splitlines()[-1]}")
        if graph_header(own) != (n, 2 * m):
            stop(5, f"converted Friendster graph has header {graph_header(own)}, expected {(n, 2 * m)}")
        g = own
    say(f"friendster graph: {g} (n={ref['friendster']['n']}, m={ref['friendster']['m']}) OK")


def run_engine(name: str, graph: Path, n: int, queries: list[int], out: Path, wall_h: float) -> float:
    """Run kcore for the queries not yet in `out` (appending), return the wall time in seconds."""
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    for left in sorted(d.glob("part_*.jsonl")):  # rows of an interrupted engine process
        merge(left, out)
    done = {r["q"] for r in rows_of(out)} if out.exists() else set()
    todo = [q for q in queries if q not in done]
    if not todo:
        say(f"{name}: all {len(queries)} queries already done")
        return 0.0
    pos, th = d / "pos_dummy.bin", d / "theta_dummy.bin"
    write_const_matrix(pos, n, 0.0)
    write_const_matrix(th, n, 1.0)
    part = d / f"part_{int(time.time())}.jsonl"
    qf = d / "queries_todo.txt"
    qf.write_text("\n".join(map(str, todo)) + "\n")
    cmd = [str(OUT / "engine" / "lacs_search"), "--graph", str(graph), "--pos", str(pos), "--theta", str(th),
           "--queries", str(qf), "--algo", "kcore", "--k", "3", "--time-limit", "60", "--no-members",
           "--out", str(part)]
    say(f"{name}: {len(todo)} queries to run ({len(done)} done before); " + " ".join(cmd))
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    CURRENT[:] = [p]
    last, rc = 0, None
    try:
        while rc is None:
            try:
                rc = p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                k = sum(1 for _ in open(part)) if part.exists() else 0
                if k != last and (k <= 5 or k // 10 > last // 10):
                    say(f"{name}: {k}/{len(todo)} queries after {time.time() - t0:.0f} s")
                    last = k
                if time.time() - t0 > wall_h * 3600:
                    p.kill()
                    p.wait()
                    stop(7, f"{name}: watchdog, {wall_h} h exceeded; rerun to resume")
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()
        if part.exists():
            merge(part, out)
    if rc != 0:
        stop(6, f"{name}: lacs_search exit {rc}: {p.stderr.read().strip()[-2000:]}")
    return time.time() - t0


def rows_of(path: Path) -> list[dict]:
    out = []
    for l in open(path):
        try:
            out.append(json.loads(l))
        except ValueError:
            pass  # a line cut off by an interruption
    return out


def merge(part: Path, out: Path) -> None:
    """Append the complete rows of an engine output file to `out`, then remove it."""
    with open(out, "a") as fo:
        for r in rows_of(part):
            fo.write(json.dumps(r) + "\n")
    part.unlink()


def qtile(xs, p):
    xs = sorted(xs)
    i = (len(xs) - 1) * p
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def st_pokec(ref):
    out = OUT / "pokec_kcore.jsonl"
    out.unlink(missing_ok=True)  # always timed afresh: the check is about the machine now
    for left in (OUT / "pokec").glob("part_*.jsonl"):
        left.unlink()
    qs = [int(x) for x in (QDIR / "queries_pokec.txt").read_text().split()]
    pg = W2 / "runs" / "efficiency_w2_graphs" / "work" / "e3_pokec" / "graph.bin"
    run_engine("pokec", pg, ref["pokec"]["n"], qs, out, 1.0)
    rows = {r["q"]: r for r in rows_of(out)}
    want = ref["pokec"]["kcore_size"]
    diff = [q for q in qs if rows.get(q, {}).get("size") != want[str(q)]]
    med = statistics.median(rows[q]["time_ms"] for q in qs if q in rows)
    ratio = med / ref["pokec"]["kcore_time_ms_median"]
    res = {"queries": len(qs), "size_mismatches": len(diff), "median_ms": med,
           "w2_median_ms": ref["pokec"]["kcore_time_ms_median"], "ratio": ratio}
    (OUT / "pokec_check.json").write_text(json.dumps(res, indent=1))
    say(f"[pokec check] sizes equal to W2 on {len(qs) - len(diff)}/{len(qs)}; median {med:.1f} ms vs W2 "
        f"{ref['pokec']['kcore_time_ms_median']:.1f} ms (ratio {ratio:.2f}, allowed {TIMING_BAND[0]}-{TIMING_BAND[1]})")
    if diff:
        stop(8, f"POKEC CHECK FAILED: k-core sizes differ from W2 on {len(diff)} queries (e.g. {diff[:5]})")
    if not TIMING_BAND[0] <= ratio <= TIMING_BAND[1] and not SKIP_TIMING:
        stop(8, "POKEC CHECK FAILED: timing not comparable with W2 (machine busy or different?)")
    say("POKEC CHECK PASS")


def st_friendster(ref):
    chk = json.loads((OUT / "pokec_check.json").read_text()) if (OUT / "pokec_check.json").exists() else None
    if not chk or chk["size_mismatches"] or (not TIMING_BAND[0] <= chk["ratio"] <= TIMING_BAND[1] and not SKIP_TIMING):
        stop(8, "Friendster not started: the Pokec check has not passed")
    if mem_available_gb() < MIN_MEM_GB:
        stop(3, f"MemAvailable {mem_available_gb():.1f} GB < {MIN_MEM_GB} GB")
    qs = [int(x) for x in (QDIR / "queries_friendster.txt").read_text().split()]
    if len(qs) != ref["friendster"]["n_queries"]:
        stop(5, "Friendster query list has the wrong length")
    g = friendster_graph(ref)
    if g == Path():
        stop(5, "Friendster graph not available; run the inputs stage first")
    wall = run_engine("friendster", g, ref["friendster"]["n"], qs, OUT / "friendster_kcore.jsonl", WALL_LIMIT_H)
    say(f"friendster: engine wall time {wall:.0f} s (includes loading the graph and its core numbers)")


def st_summary(ref):
    out = {"version": VERSION, "time": time.strftime("%F %T"), "pokec_check": None, "friendster": None}
    if (OUT / "pokec_check.json").exists():
        out["pokec_check"] = json.loads((OUT / "pokec_check.json").read_text())
    fq = [int(x) for x in (QDIR / "queries_friendster.txt").read_text().split()]
    f = OUT / "friendster_kcore.jsonl"
    if f.exists():
        rows = {r["q"]: r for r in rows_of(f)}
        got = [rows[q] for q in fq if q in rows]
        t = [r["time_ms"] for r in got if r["status"] == "ok"]
        sizes = {}
        for r in got:
            sizes[r["size"]] = sizes.get(r["size"], 0) + 1
        status = {}
        for r in got:
            status[r["status"]] = status.get(r["status"], 0) + 1
        out["friendster"] = {
            "queries_done": len(got), "queries_total": len(fq), "complete": len(got) == len(fq),
            "all_ok": status == {"ok": len(fq)}, "time_over": "status ok only",
            "status": status, "distinct_sizes": {str(k): v for k, v in sorted(sizes.items())},
            "time_ms": {"median": statistics.median(t), "p90": qtile(t, 0.9), "mean": statistics.fmean(t),
                        "min": min(t), "max": max(t)} if t else None}
        with open(OUT / "friendster_kcore_per_query.csv", "w") as fo:
            fo.write("q,size,status,time_ms\n")
            for r in got:
                fo.write(f"{r['q']},{r['size']},{r['status']},{r['time_ms']}\n")
    (OUT / "p2x_summary.json").write_text(json.dumps(out, indent=1))
    fs = out["friendster"]
    if fs and fs["time_ms"]:
        tag = "" if fs["complete"] and fs["all_ok"] else "PARTIAL OR NOT ALL OK, NOT A RESULT: "
        say(f"[friendster] {tag}{fs['queries_done']}/{fs['queries_total']} queries; k-core median "
            f"{fs['time_ms']['median'] / 1000:.1f} s, p90 {fs['time_ms']['p90'] / 1000:.1f} s; "
            f"statuses {fs['status']}; component sizes {fs['distinct_sizes']}")
    say("SUMMARY written: " + str(OUT / "p2x_summary.json"))


def st_bundle(ref):
    name = OUT.parent / f"p2x_bundle_{time.strftime('%Y%m%d_%H%M%S')}.tar.gz"
    files = [OUT / x for x in ("p2x_summary.json", "pokec_check.json", "friendster_kcore_per_query.csv",
                               "logs/p2x.log")]
    with tarfile.open(name, "w:gz") as tf:
        for p in files:
            if p.exists():
                tf.add(p, arcname=f"p2x/{p.relative_to(OUT)}")
        tf.add(REF, arcname="p2x/w2_reference.json")
    say(f"BUNDLE {name} {name.stat().st_size} bytes sha256 {sha256(name)}")


STAGES = {"env": st_env, "build": st_build, "inputs": st_inputs, "pokec": st_pokec, "friendster": st_friendster,
          "summary": st_summary, "bundle": st_bundle}


def run(stages: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    with open(LOCK, "a") as lf:
        try:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("P2X NOT STARTED: another run holds " + str(LOCK))
            return 1
        ref = json.loads(REF.read_text())
        rc, failed = 0, "none"

        def on_signal(sig, _):
            for p in CURRENT:
                if p.poll() is None:
                    p.kill()
            say(f"P2X END rc=130 INTERRUPTED by signal {sig}")
            os._exit(130)
        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)
        try:
            for s in stages:
                say(f"== {s}")
                STAGES[s](ref)
        except SystemExit as e:
            rc, failed = int(e.code or 1), s
        except Exception as e:  # noqa: BLE001
            say(f"!! {type(e).__name__}: {e}")
            rc, failed = 1, s
        if rc and "summary" in stages and failed not in ("summary", "bundle"):
            for s2 in ("summary", "bundle"):  # bring back what exists, even after a failure
                try:
                    STAGES[s2](ref)
                except Exception as e:  # noqa: BLE001
                    say(f"!! {s2}: {e}")
        say(f"P2X END rc={rc} failed: {failed}")
        return rc


def main(argv) -> int:
    cmd = argv[1] if len(argv) > 1 else "run"
    stages = os.environ.get("STAGES", "env build inputs pokec friendster summary bundle").split()
    if cmd == "run":
        return run(stages)
    if cmd == "launch":
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOCK, "a") as lf:  # refuse here, so that the pid file keeps naming the running run
            try:
                fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                print(f"P2X NOT STARTED: another run holds {LOCK}; it is still going (this is not its end)")
                return 1
            fcntl.flock(lf, fcntl.LOCK_UN)
        with open(LOG, "a") as fh:
            fh.write(f"\n### launch {time.strftime('%F %T')} STAGES=\"{' '.join(stages)}\"\n")
            fh.flush()
            p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "run"], stdout=fh,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                                 env=dict(os.environ, STAGES=" ".join(stages)))
        (LOG.parent / "p2x.pid").write_text(f"{p.pid}\n")
        print(f"started (pid {p.pid}); log {LOG}; finished when its last line starts with 'P2X END'")
        return 0
    if cmd == "status":
        files = [OUT / "friendster_kcore.jsonl"] + sorted((OUT / "friendster").glob("part_*.jsonl"))
        k = sum(len(rows_of(f)) for f in files if f.exists())
        last = LOG.read_text().strip().splitlines()[-1] if LOG.exists() else "(no log)"
        print(f"friendster queries done: {k}/200; last log line: {last}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
