#!/usr/bin/env bash
# Round 3 (v0.3.1). CPU only; run from the project root. Steps in priority order:
#   1. DDO feasibility probe (only if users/debates JSON are in data/raw/ddo/)
#   2-3. meetup application on Gowalla and Brightkite: prepare -> gates -> E2 at the k recommended by G3-geo
#   4. E3 for the paper (configs/efficiency_round3.yaml; resumable: rerun the same command after an interruption)
#   5. parameter study of the meetup application: geo-e2 at every k (geo-e2-sweep)
# A failed step is reported and the independent steps after it still run; exit code 1 if any step failed.
set -uo pipefail
PY=${PY:-python}
DDO_DIR=${DDO_DIR:-data/raw/ddo}
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

recommended_k () {  # k recommended by G3-geo, else the config's search.k
  $PY -c '
import json, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
g = json.load(open(cfg["out_dir"] + "/results/geo_gates.json"))
print((g.get("G3") or {}).get("recommended_k") or cfg.get("search", {}).get("k", 3))
' "$1"
}

geo () {  # geo <config> <name>
  local cfg=$1 name=$2 k
  step "$name geo-prepare" $PY -m lacs geo-prepare -c "$cfg" || return 1
  step "$name geo-gates" $PY -m lacs geo-gates -c "$cfg" || return 1
  k=$(recommended_k "$cfg") || k=3
  step "$name geo-e2 (k=$k)" $PY -m lacs geo-e2 -c "$cfg" --set search.k="$k"
}

if ls "$DDO_DIR"/*users*.json >/dev/null 2>&1 && ls "$DDO_DIR"/*debates*.json >/dev/null 2>&1; then
  step "DDO probe" $PY scripts/probe_ddo.py --data "$DDO_DIR" --out runs/ddo_probe
else
  echo "== DDO probe: files not found in $DDO_DIR; skipped =="
fi
geo configs/gowalla.yaml gowalla
geo configs/brightkite.yaml brightkite
step "E3" $PY -m lacs e3 -c configs/efficiency_round3.yaml
step "gowalla geo-e2-sweep" $PY -m lacs geo-e2-sweep -c configs/gowalla.yaml
step "brightkite geo-e2-sweep" $PY -m lacs geo-e2-sweep -c configs/brightkite.yaml

if [ "${#FAILED[@]}" -gt 0 ]; then
  printf 'failed steps:\n'
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
echo "done: runs/gowalla/results, runs/brightkite/results, runs/efficiency_round3/results"
