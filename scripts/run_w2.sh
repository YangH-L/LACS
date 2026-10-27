#!/usr/bin/env bash
# W2: engine v0.5 (best-first LatSweep-Exact, colouring order, upper bound when not proven) on the
# round-3 datasets (run A) and on four more SNAP graphs (run B). CPU only; run from this directory.
#   bash scripts/run_w2.sh                      # all stages
#   STAGES="e3a bundle" bash scripts/run_w2.sh  # selected stages
# Stages: env build selftest lowmem fetch e3a e3b bundle. A failed stage is reported (line "!! ...")
# and the independent stages after it still run; the exit code is 1 if any stage failed.
# Interrupted runs: rerun the same command; E3 resumes finished datasets from their checkpoints.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PY
export LJ_EDGES=${LJ_EDGES:-data/raw/snap/com-lj.ungraph.txt}
STAGES=${STAGES:-"env build selftest lowmem fetch e3a e3b bundle"}
LOGS=runs/logs
mkdir -p "$LOGS"
FAILED=()

has () { [[ " $STAGES " == *" $1 "* ]]; }
stage () {  # stage <label> <command...>; output to $LOGS/<label>.log and the console
  local label=$1; shift
  echo "== $label ($(date '+%F %T')) =="
  echo "== $label ($(date '+%F %T')) ==" >> "$LOGS/$label.log"
  if ! "$@" > >(tee -a "$LOGS/$label.log") 2>&1; then
    echo "!! $label failed ($(date '+%F %T')); see $LOGS/$label.log"
    FAILED+=("$label")
    return 1
  fi
  echo "== $label done ($(date '+%F %T')) =="
}

if has env; then
  stage env bash -c '
    "$PY" -V; "$PY" -c "import numpy, scipy, pandas, sklearn; print(numpy.__version__, scipy.__version__, pandas.__version__, sklearn.__version__)"
    g++ --version | head -1; cmake --version | head -1; nproc; free -g; df -h .
    lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket"; cat /proc/loadavg
    sha256sum cpp/src/* | tee runs/logs/engine_sources.sha256'
fi
if has build; then
  stage build $PY -m lacs build-cpp
fi
if has selftest; then
  if stage selftest $PY -m unittest tests.test_search tests.test_search_v05 tests.test_e3 tests.test_e3_v05 -v; then
    echo "SELFTEST PASSED"
  else
    echo "SELFTEST FAILED: stopping (see $LOGS/selftest.log)"
    exit 1
  fi
fi
if has lowmem; then  # the two-pass converter must give the same file as the in-memory one on LiveJournal
  stage lowmem bash -c '
    set -e
    LJ="$LJ_EDGES"
    mkdir -p runs/lowmem_check
    TIME=""; [ -x /usr/bin/time ] && TIME="/usr/bin/time -v"
    $TIME cpp/build/lacs_convert --in "$LJ" --out runs/lowmem_check/a.bin --relabel runs/lowmem_check/a.ids 2> runs/lowmem_check/a.time
    $TIME cpp/build/lacs_convert --in "$LJ" --out runs/lowmem_check/b.bin --relabel runs/lowmem_check/b.ids --lowmem 2> runs/lowmem_check/b.time
    grep -E "Maximum resident|Elapsed" runs/lowmem_check/a.time runs/lowmem_check/b.time || true
    "$PY" -c "import filecmp, sys
d = \"runs/lowmem_check/\"
same = filecmp.cmp(d + \"a.bin\", d + \"b.bin\", shallow=False) and filecmp.cmp(d + \"a.ids\", d + \"b.ids\", shallow=False)
print(\"LOWMEM IDENTICAL\" if same else \"LOWMEM DIFFERENT\")
sys.exit(0 if same else 1)"
    rm -f runs/lowmem_check/*.bin runs/lowmem_check/*.ids'
fi
if has fetch; then
  stage fetch bash scripts/fetch_snap_w2.sh
fi
if has e3a; then
  stage e3a $PY -m lacs e3 -c configs/efficiency_w2.yaml
fi
if has e3b; then
  stage e3b $PY -m lacs e3 -c configs/efficiency_w2_graphs.yaml
fi
if has bundle; then
  stage bundle bash scripts/bundle_w2.sh
fi
if [ "${#FAILED[@]}" -gt 0 ]; then
  printf 'failed stages:\n'; printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
echo "done: runs/efficiency_w2/results, runs/efficiency_w2_graphs/results, bundle in runs/"
