#!/usr/bin/env bash
# E3 efficiency (CPU). Needs runs/debagreement to exist for the from_run dataset.
set -euo pipefail
python -m lacs e3 -c configs/efficiency.yaml "$@"
python -m lacs report -c configs/efficiency.yaml
echo "results: runs/efficiency/results/e3_summary.csv, figures in runs/efficiency/figures/"
