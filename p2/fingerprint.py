"""Read-only check of the inputs outside this directory: size and modification time of every file.

  python -I p2/fingerprint.py record --out runs_p2/logs/readonly_before.json DIR [DIR ...]
  python -I p2/fingerprint.py compare --before runs_p2/logs/readonly_before.json --out runs_p2/logs/readonly_check.json

record keeps the first fingerprint (a second `record` on the same --out leaves it alone), so the comparison
always refers to the state before the first stage. compare prints READ-ONLY INPUTS UNCHANGED or the changed
files (it never fails a stage: the result is reported).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def scan(dirs: list[str]) -> dict:
    out = {}
    for d in dirs:
        if not os.path.exists(d):
            out[d] = None
            continue
        for root, _, files in os.walk(d):
            for f in files:
                p = os.path.join(root, f)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                out[p] = [st.st_size, st.st_mtime_ns]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--out", required=True)
    r.add_argument("dirs", nargs="+")
    c = sub.add_parser("compare")
    c.add_argument("--before", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "record":
        if Path(a.out).exists():
            print(f"fingerprint kept: {a.out}")
            return 0
        fp = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "dirs": a.dirs, "files": scan(a.dirs)}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(fp))
        print(f"fingerprint of {len(fp['files'])} files under {a.dirs} -> {a.out}")
        return 0
    before = json.loads(Path(a.before).read_text())
    now = scan(before["dirs"])
    old = before["files"]
    changed = sorted(p for p in set(old) | set(now) if old.get(p) != now.get(p))
    res = {"before": before["created"], "checked": time.strftime("%Y-%m-%d %H:%M:%S"), "files": len(now),
           "changed": changed[:200], "n_changed": len(changed)}
    Path(a.out).write_text(json.dumps(res, indent=1))
    if changed:
        print(f"READ-ONLY INPUTS CHANGED: {len(changed)} files\n  " + "\n  ".join(changed[:30]))
    else:
        print(f"READ-ONLY INPUTS UNCHANGED ({len(now)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
