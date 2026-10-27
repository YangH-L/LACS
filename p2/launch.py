"""Start scripts/run_p2.sh detached from the calling shell (the server has no nohup; a process started with
start_new_session=True keeps running when the shell that started it exits).

  python p2/launch.py --stages "env build selftest p24pilot p24gate bundle" --log runs_p2/logs/run_p2_batch1.log

Prints the PID and returns at once. Refuses to start (exit 1, nothing changed) while another run_p2.sh holds the
lock runs_p2/.run_p2.lock or while the PID recorded next to the log still belongs to a running run_p2.sh.
Follow the run with `tail -5 <log>`; it has finished when the log's last line starts with "RUN_P2 END rc=" (with
"INTERRUPTED" after rc when it was stopped by a signal). A run killed with SIGKILL or by a crash of the machine
prints nothing: then `kill -0 <pid>` failing is the sign. The PID is also the process group: `kill -- -<pid>`
stops the run with its engine processes. Extra environment variables (NPROC=8 ...) are passed through.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", required=True)
    ap.add_argument("--log", required=True)
    a = ap.parse_args(argv)
    log = Path(a.log)
    if not log.is_absolute():
        log = ROOT / log
    log.parent.mkdir(parents=True, exist_ok=True)
    lockp = ROOT / "runs_p2" / ".run_p2.lock"
    lockp.parent.mkdir(parents=True, exist_ok=True)
    with open(lockp, "a") as lf:  # the same flock(2) lock as run_p2.sh: free means no run is going
        try:
            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(f"!! not started: another run_p2.sh is running (lock {lockp}); wait until its log ends with "
                  f"'RUN_P2 END', or stop it first")
            return 1
        fcntl.flock(lf, fcntl.LOCK_UN)
    pidf = log.parent / (log.stem + ".pid")
    if pidf.exists():
        try:
            old = int(pidf.read_text().split()[0])
            os.kill(old, 0)
            cmd = Path(f"/proc/{old}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            if "run_p2.sh" in cmd:
                print(f"!! not started: the run of this log (pid {old}) is still going")
                return 1
        except (ValueError, IndexError, ProcessLookupError, FileNotFoundError):
            pass  # no such process any more: the earlier run has ended
        except PermissionError:
            print(f"!! not started: process {old} of this log exists and belongs to another user; check it first")
            return 1
    env = dict(os.environ, STAGES=a.stages, PY=os.environ.get("PY", sys.executable))
    with open(log, "a") as fh:
        fh.write(f"\n### launch {time.strftime('%Y-%m-%d %H:%M:%S')} STAGES=\"{a.stages}\" PY={env['PY']}\n")
        fh.flush()
        p = subprocess.Popen(["bash", str(ROOT / "scripts" / "run_p2.sh")], cwd=str(ROOT), env=env, stdout=fh,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    pidf.write_text(f"{p.pid}\n")
    print(f"started run_p2.sh (pid {p.pid}), stages: {a.stages}\nlog: {log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
