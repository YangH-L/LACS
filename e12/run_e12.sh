#!/usr/bin/env bash
# E12 (docs/PLANS.md (lacs_e12_plan_20261029)): every step in order, locally. Usage:
#   DATA=/path/to/MEP_files RUN=runs/e12 NPROC=2 NPERM=1000 bash e12/run_e12.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD/p0:$PWD"
DATA=${DATA:?MEP data directory}
RUN=${RUN:-runs/e12}
NPROC=${NPROC:-2}
NPERM=${NPERM:-1000}
PY=${PY:-python3}
mkdir -p "$RUN/logs"
CFG="$RUN/e12_config.yaml"
sed "s#out_dir: runs/e12#out_dir: $RUN#; s#n_proc: 2#n_proc: $NPROC#" configs/e12_ep.yaml > "$CFG"
step () { local name=$1; shift; echo "== $name ($(date '+%F %T'))"; "$@" > "$RUN/logs/$name.log" 2>&1; echo "== $name done ($(date '+%F %T'))"; }
BUILD_OPTS=${BUILD_OPTS:-}   # synthetic test data only: --skip-consistency
step build "$PY" -m e12.build --data "$DATA" --out "$RUN" $BUILD_OPTS
step geo_e2 "$PY" -m lacs geo-e2 -c "$CFG"
step structure "$PY" -m e6.structure --run "$RUN" --results "$RUN/results"
step export "$PY" -m p0kit.export_groups geo --config "$CFG" --out "$RUN/e1_input" --force
step static "$PY" -m e6.static --instance "$RUN/e1_input/instance.json" --groups "$RUN/e1_input/groups.jsonl.gz" --out "$RUN/results"
step sweep "$PY" -m lacs geo-e2-sweep -c "$CFG"
step q4 "$PY" e5/analysis.py --results "$RUN/results" --out "$RUN/results/e12_q4_analysis" --title "E12 Q4 (European Parliament)" --seed 20261029
step dynamics "$PY" -m e1x.open_sweep --config e1x/configs/e12_dynamics.yaml --instance "$RUN/e1_input/instance.json" \
  --groups "$RUN/e1_input/groups.jsonl.gz" --weights "$RUN/geo/twitter_weights.npz" --out "$RUN/e1x/e12" --lcc-only --workers "$NPROC"
step dyn_summary "$PY" -m e1x.summarize --runs "$RUN/e1x/e12" --out "$RUN/results"
step premise "$PY" -m e12.premise --run "$RUN" --results "$RUN/results" --n-perm "$NPERM" --procs "$NPROC"
echo "E12 done: $RUN/results "
