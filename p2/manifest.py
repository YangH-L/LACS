"""Source manifest of the P2 package: sha256 of every source file of the rehearsed tree.

  python -I p2/manifest.py check [--root .]     # every listed file present and identical, else exit code 1
  python -I p2/manifest.py write [--root .]     # (packaging only) regenerate p2/MANIFEST_SOURCES.sha256

The env stage runs `check`, so the server runs exactly the code that was rehearsed. Files that exist on the
server but are not listed (runs/, builds, caches, other packages) are ignored.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

NAME = "p2/MANIFEST_SOURCES.sha256"
TOP_SKIP = {"runs", "runs_p0", "runs_p2", "data"}          # outputs and data of the server directory
ANY_SKIP = {"__pycache__", ".git"}
PATH_SKIP = ("cpp/build/", "p0/p0kit/_lib/")               # build products
SKIP_SUFFIX = {".pyc", ".so", ".o", ".log", ".tar.gz", ".zip"}


def files(root: Path) -> list[Path]:
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        posix = rel.as_posix()
        if rel.parts[0] in TOP_SKIP or any(part in ANY_SKIP for part in rel.parts[:-1]) or posix == NAME:
            continue
        if any(posix.startswith(s) for s in PATH_SKIP):
            continue
        if any(rel.name.endswith(s) for s in SKIP_SUFFIX):
            continue
        out.append(rel)
    return out


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "write"])
    ap.add_argument("--root", default=".")
    a = ap.parse_args(argv)
    root = Path(a.root)
    man = root / NAME
    if a.cmd == "write":
        lines = [f"{sha(root / f)}  {f.as_posix()}" for f in files(root)]
        man.write_text("\n".join(lines) + "\n")
        print(f"wrote {man}: {len(lines)} files")
        return 0
    bad, n = [], 0
    for line in man.read_text().splitlines():
        if not line.strip():
            continue
        h, f = line.split("  ", 1)
        n += 1
        p = root / f
        if not p.is_file():
            bad.append(f"missing  {f}")
        elif sha(p) != h:
            bad.append(f"differs  {f}")
    if bad:
        print(f"SOURCE CHECK FAILED: {len(bad)} of {n} files\n  " + "\n  ".join(bad[:60]))
        return 1
    print(f"source check: all {n} files identical to the rehearsed package")
    return 0


if __name__ == "__main__":
    sys.exit(main())
