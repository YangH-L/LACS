"""Summary bundle of the P2 package: only summaries and per-query metrics (internal ids), never members,
positions, graphs, engine work files or Yelp-derived inputs.

  python -I p2/bundle.py --root . --out runs_p2/p2_bundle_<time>.tar.gz

Every file under runs_p2/ matching INCLUDE is taken; then every chosen path is checked against DENY and the
bundle is refused (exit code 1, no file left behind) if any matches.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import sys
import tarfile
from pathlib import Path

INCLUDE = [
    "runs_p2/logs/*",
    "runs_p2/e11/*/inputs.json",
    "runs_p2/e11/pilot/*",
    "runs_p2/e11/report/*",
    "runs_p2/e11/*/results/e11/gate_s1.json",
    "runs_p2/e11/*/results/e11*/s*/e11_per_query.csv",
    "runs_p2/e11/*/results/e11*/s*/e11_summary.json",
    "runs_p2/e11/*/results/e11*/s*/e11_selection_mismatch.json",
    "runs_p2/e11/*/results/e11*/s*/config_resolved.json",
    "runs_p2/e11/*/results/e11*/s*/e11.log",
    "runs_p2/e11/*/results/e11*/s*/DONE",
    "runs_p2/e4v05/report/*",
    "runs_p2/e4v05/*/e4_*/*/per_query.csv",
    "runs_p2/e4v05/*/e4_*/*/manifest.json",
    "runs_p2/e4v05/*/e4_*/*/gate_v04.json",
    "runs_p2/e4v05/*/e4_*/*/DONE",
    "runs_p2/p24b/report/*",
    "runs_p2/p24b/configs/*",
    "runs_p2/p24b/*/s*/e1_input/manifest.json",
    "runs_p2/p24b/*/s*/dyn/e1x_rows.jsonl.gz",
    "runs_p2/p24b/*/s*/dyn/e1x_groups.jsonl.gz",
    "runs_p2/p24b/*/s*/dyn/manifest.json",
    "runs_p2/p24b/*/s*/dyn/DONE",
]
DENY = ["*/geo/*", "*/graph/*", "*/work/*", "*/inst/*", "*/blocks/*", "*e1_input/groups.jsonl*",
        "*e4v05/*/groups.jsonl*", "*instance.json", "*.npy", "*.npz", "*.bin", "*users.csv", "*positions*",
        "*yelp_academic*", "*parsed*", "*e11/yelp/results/e11*/s*/e11_per_query.csv.tmp"]


def collect(root: Path) -> list[str]:
    base = root / "runs_p2"
    files = sorted(str(p.relative_to(root)) for p in base.rglob("*") if p.is_file())
    return [f for f in files if any(fnmatch.fnmatch(f, pat) for pat in INCLUDE)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    root, out = Path(args.root), Path(args.out)
    files = collect(root)
    bad = [f for f in files if any(fnmatch.fnmatch(f, pat) for pat in DENY)]
    if bad:
        print("BUNDLE WOULD CONTAIN FORBIDDEN FILES:\n  " + "\n  ".join(bad[:50]))
        return 1
    if not files:
        print("!! nothing to bundle")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with tarfile.open(tmp, "w:gz") as tf:
        for f in files:
            tf.add(str(root / f), arcname=f)
    tmp.replace(out)
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"{out}  {out.stat().st_size} bytes  sha256 {h}  ({len(files)} files)")
    for f in files:
        print("  " + f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
