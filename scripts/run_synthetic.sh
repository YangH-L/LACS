#!/usr/bin/env bash
# End-to-end smoke test on synthetic data with a planted SJT ground truth (CPU only).
set -euo pipefail
python -m lacs all -c configs/synthetic.yaml "$@"
cat runs/synthetic/results/REPORT.md
