#!/usr/bin/env bash
# ChangeMyView (binary: delta / no delta). Includes the inverted-U check in G1.
set -euo pipefail
python -m lacs all -c configs/cmv.yaml "$@"
echo "report: runs/cmv/results/REPORT.md"
