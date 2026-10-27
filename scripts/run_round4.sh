#!/usr/bin/env bash
# Round 4 (v0.4). CPU only; run from the project root. Steps in priority order:
#   1-2. meetup application, re-evaluation with the pre-declared plan of docs/PLANS.md (round-4 plan):
#        geo-e2 on every eligible query (results/r4/), then the k and tolerance sweeps
#   3.   optional (CLEAN_E3=1): E3 again on a quiet machine, same queries, for clean timings
# A failed step is reported and the independent steps after it still run; exit code 1 if any step failed.
set -uo pipefail
PY=${PY:-python}
FAILED=()

step () {  # step <label> <command...>
  local label=$1
  shift
  echo "== $label =="
  if ! "$@"; then
    echo "!! $label failed"
    FAILED+=("$label")
    return 1
  fi
}

for ds in gowalla brightkite; do
  step "$ds geo-e2 (r4)" $PY -m lacs geo-e2 -c "configs/${ds}_r4.yaml"
  step "$ds geo-e2-sweep (r4)" $PY -m lacs geo-e2-sweep -c "configs/${ds}_r4.yaml"
done
if [ "${CLEAN_E3:-0}" = "1" ]; then
  step "E3 clean timings" $PY -m lacs e3 -c configs/efficiency_round4_clean.yaml
else
  echo "== E3 clean timings: skipped (set CLEAN_E3=1 on a quiet machine) =="
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
  printf 'failed steps:\n'
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
echo "done: runs/gowalla/results/r4, runs/brightkite/results/r4"
