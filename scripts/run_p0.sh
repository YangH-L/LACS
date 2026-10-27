#!/usr/bin/env bash
# P0 (v0.5): E1-E4 of the revision plan, on top of the finished rounds 3 and 4.
# CPU only; run from the project root (the directory that contains lacs/ and p0/).
#
#   bash scripts/run_p0.sh                       # every stage, in priority order
#   STAGES="build selftest" bash scripts/run_p0.sh
#   STAGES="e1" DATASETS="gowalla" bash scripts/run_p0.sh
#
# Stages: build selftest export e1 e2 e3 e4 e1sigma report bundle   (e1sigma is optional, not in the default)
# Environment:
#   PY             python executable (default: python)
#   P0_NPROC       engine processes for export / E2 / E4 (default 8)
#   P0_WORKERS     python workers for E1 (default 8)
#   P0_E3_WORKERS  parallel queries in E3 (default 4; 1 gives the cleanest timings)
#   P0_E1_MAX_LCC  meetups, E1: at most this many queries with an LCC (random sample; default 1000)
#   P0_E1_OTHERS   meetups, E1: plus this many queries without an LCC (default 300)
#   P0_E1_CONFIG   E1 variants (default p0/configs/e1_stress.yaml; p0/configs/e1_quick.yaml for a smoke run)
#   P0_FORCE=1     redo finished exports, E2/E3/E4 settings instead of skipping them
#   DATASETS       subset of: gowalla brightkite livejournal orkut
# Inputs are read from runs/ and never modified; outputs go to runs_p0/. A failed step is
# reported and the independent steps after it still run; exit code 1 if any step failed.
# Rerunning the same command resumes after an interruption: finished exports and E2/E4
# settings are skipped, E1 and E3 continue from their checkpoints (repaired if a run was killed).
set -uo pipefail
PY=${PY:-python}
# single-threaded BLAS/OpenMP (parallelism is over processes; thread pools do not survive fork)
for v in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS; do
  export "$v=${!v:-1}"
done
export PYTHONPATH="p0${PYTHONPATH:+:$PYTHONPATH}"
NPROC=${P0_NPROC:-8}
WORKERS=${P0_WORKERS:-8}
E3W=${P0_E3_WORKERS:-4}
E1_MAX_LCC=${P0_E1_MAX_LCC:-1000}
E1_OTHERS=${P0_E1_OTHERS:-300}
FORCE=""
if [ "${P0_FORCE:-0}" = "1" ]; then FORCE="--force"; fi
STAGES=${STAGES:-"build selftest export e1 e2 e3 e4 report bundle"}
DATASETS=${DATASETS:-"gowalla brightkite livejournal orkut"}
CFG=p0/configs/p0.yaml
E1CFG=${P0_E1_CONFIG:-p0/configs/e1_stress.yaml}
LOG=runs_p0/logs
mkdir -p "$LOG"
FAILED=()

has () { [[ " $STAGES " == *" $1 "* ]]; }
want () { [[ " $DATASETS " == *" $1 "* ]]; }
step () {  # step <label> <command...>; output also goes to runs_p0/logs/<label>.log
  local label=$1
  shift
  local f="$LOG/$(echo "$label" | tr ' /' '__').log"
  echo "== $label ($(date '+%F %T')) =="
  if ! "$@" 2>&1 | tee -a "$f"; then   # pipefail: the command's status, not tee's
    echo "!! $label failed (log: $f)"
    FAILED+=("$label")
    return 1
  fi
}

if has build; then
  step "build p0 kernels" bash p0/build.sh
  step "build lacs engine" $PY -m lacs build-cpp
fi
if has selftest; then
  step "selftest" $PY -m p0kit.selftest || { echo "selftest failed: fix before running the experiments"; exit 1; }
fi

# ---- E1 inputs: groups with members (meetups: exactly round 4's geo-e2 groups)
if has export; then
  for ds in gowalla brightkite; do
    want "$ds" && step "export $ds" $PY -m p0kit.export_groups geo --config "configs/${ds}_r4.yaml" \
      --out "runs_p0/$ds/e1_input" $FORCE
  done
  for ds in livejournal orkut; do
    want "$ds" && step "export $ds" $PY -m p0kit.export_groups synth --e3-run runs/efficiency_round3 \
      --dataset "$ds" --out "runs_p0/$ds/e1_input" --n-proc "$NPROC" $FORCE
  done
fi

# ---- E1 interaction stress test
if has e1; then
  for ds in gowalla brightkite; do
    want "$ds" && step "E1 $ds" $PY -m p0kit.e1_stress --config "$E1CFG" \
      --instance "runs_p0/$ds/e1_input/instance.json" --groups "runs_p0/$ds/e1_input/groups.jsonl.gz" \
      --out "runs_p0/$ds/e1/default" --workers "$WORKERS" --lcc-only --max-lcc-queries "$E1_MAX_LCC" \
      --other-sample "$E1_OTHERS"
  done
  for ds in livejournal orkut; do
    want "$ds" && step "E1 $ds" $PY -m p0kit.e1_stress --config "$E1CFG" \
      --instance "runs_p0/$ds/e1_input/instance.json" --groups "runs_p0/$ds/e1_input/groups.jsonl.gz" \
      --out "runs_p0/$ds/e1/default" --workers "$WORKERS"
  done
fi

# ---- E2 heterogeneity sweep
if has e2; then
  for ds in livejournal orkut; do
    want "$ds" && step "E2 $ds" $PY -m p0kit.engine_sweeps --config "$CFG" --dataset "$ds" --experiment e2 \
      --n-proc "$NPROC" $FORCE
  done
fi

# ---- E3 exact baselines (same sweep for every solver; engine rerun for the paper's implementation)
if has e3; then
  for ds in gowalla brightkite livejournal orkut; do
    want "$ds" && step "E3 $ds" $PY -m p0kit.e3_baselines --config "$CFG" --dataset "$ds" --workers "$E3W" \
      --engine-rerun $FORCE
  done
fi

# ---- E4 intrinsic scale, mixing, query sampling
if has e4; then
  for ds in livejournal orkut; do
    for ex in e4_scale e4_mix e4_queries; do
      want "$ds" && step "$ex $ds" $PY -m p0kit.engine_sweeps --config "$CFG" --dataset "$ds" --experiment "$ex" \
        --n-proc "$NPROC" $FORCE
    done
  done
fi

# ---- optional: E1 on the groups of other heterogeneity levels (needs e2)
if has e1sigma; then
  for ds in livejournal orkut; do
    for st in sigma0.0 sigma0.2 sigma0.8; do
      d="runs_p0/$ds/e2/$st"
      want "$ds" && [ -f "$d/DONE" ] && step "E1 $ds $st" $PY -m p0kit.e1_stress --config "$E1CFG" \
        --instance "$d/instance.json" --groups "$d/groups.jsonl.gz" --out "runs_p0/$ds/e1/$st" --workers "$WORKERS"
    done
  done
fi

if has report; then
  step "report" $PY -m p0kit.report --root runs_p0 --e3-run runs/efficiency_round3 --out runs_p0/report
fi

# ---- what to bring back: aggregates and per-query rows with internal ids only
# (never: groups.jsonl.gz with member ids, inst/ with positions, work/ directories)
if has bundle; then
  B="runs_p0/p0_bundle_$(date +%Y%m%d_%H%M).tar.gz"
  step "bundle" bash -c "find runs_p0 \( -path '*/work' -o -path '*/inst' \) -prune -o -type f \
    \( -name '*.csv' -o -name 'manifest.json' -o -name 'e1_rows.jsonl.gz' -o -name 'variants.csv' \
       -o -name 'DONE' -o -path 'runs_p0/report/*' -o -path 'runs_p0/logs/*' \) -print \
    | grep -v 'p0_bundle_' | tar -czf '$B' -T - && ls -l '$B' && sha256sum '$B'"
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
  printf 'failed steps:\n'
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
echo "done: runs_p0/report/REPORT_P0.md"
