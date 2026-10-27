#!/usr/bin/env bash
# Round 2 (decided 2026-10-17). CPU only, GPU 0. Budget: <= 2 h wall clock, <= 4 CPU core-hours, <= 3 GB disk.
#   1. read-only diagnosis of the round-1 DEBAGREEMENT run (rules R0-R4 fixed in lacs/experiments/diagnose.py)
#   2. feasibility probe of the debate.org (DDO) corpus (criteria P1-P4 fixed in scripts/probe_ddo.py)
# Nothing here retrains a model or touches the test split beyond reading results/e1_metrics.json.
set -euo pipefail
RUN_CFG=${RUN_CFG:-configs/debagreement.yaml}
DDO_DIR=${DDO_DIR:-data/raw/ddo}

echo "== 1/2 diagnosis of the round-1 run =="
python -m lacs diagnose -c "$RUN_CFG"
sed -n '1,12p' runs/debagreement/diagnosis/DIAGNOSIS.md 2>/dev/null || true

echo "== 2/2 DDO feasibility probe =="
if ls "$DDO_DIR"/*users*.json >/dev/null 2>&1 && ls "$DDO_DIR"/*debates*.json >/dev/null 2>&1; then
  python scripts/probe_ddo.py --data "$DDO_DIR" --out runs/ddo_probe
else
  echo "DDO files not found in $DDO_DIR (need *users*.json and *debates*.json)"
fi
