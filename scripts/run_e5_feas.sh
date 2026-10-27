#!/usr/bin/env bash
# E5 feasibility check on the Yelp Open Dataset (see README.md). CPU only; run from this
# directory or anywhere (it cd's to the package root).
#   bash scripts/run_e5_feas.sh                        # env build selftest feas bundle
#   STAGES="feas bundle" bash scripts/run_e5_feas.sh   # selected stages
# A failed stage is reported (line "!! ...") and the independent stages after it still run; the exit
# code is 1 if any stage failed. Rerunning "feas" reuses a finished extraction and the parsed arrays.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PY
YELP_ZIP=${YELP_ZIP:-data/yelp/Yelp-JSON.zip}
YELP_JSON_DIR=${YELP_JSON_DIR:-data/yelp/json}
NPROC=${NPROC:-12}
STAGES=${STAGES:-"env build selftest feas bundle"}
LOGS=runs/logs
mkdir -p "$LOGS"
FAILED=()
# one run of this script at a time (a second run would mix outputs and bundle half-written files)
if command -v flock > /dev/null; then
  exec 9> runs/.run_e5_feas.lock
  if ! flock -n 9; then
    echo "!! another run_e5_feas.sh is running (lock runs/.run_e5_feas.lock): nothing done"
    exit 1
  fi
else
  echo "warning: flock not found; running without the single-run lock"
fi

has () { [[ " $STAGES " == *" $1 "* ]]; }
stage () {  # stage <label> <command...>; output appended to $LOGS/<label>.log and shown on the console
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
    "$PY" -V; "$PY" -c "import numpy, scipy, pandas; print(numpy.__version__, scipy.__version__, pandas.__version__)"
    "$PY" -c "import orjson; print(\"orjson\", orjson.__version__)" 2>/dev/null || echo "orjson not installed (fine, slower parsing)"
    g++ --version | head -1; nproc; free -g; df -h . 2>/dev/null
    lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket"; cat /proc/loadavg
    sha256sum e5/*.py e5/tests/*.py cpp/src/* | tee runs/logs/e5_sources.sha256'
fi
if has build; then
  stage build $PY -m lacs build-cpp
fi
if has selftest; then
  if stage selftest $PY -m unittest e5.tests.test_feasibility -v; then
    echo "SELFTEST PASSED"
  else
    echo "SELFTEST FAILED: stopping (see $LOGS/selftest.log)"
    exit 1
  fi
fi
if has feas; then
  if [ -f "$YELP_ZIP" ]; then
    # exit code 2 = FEASIBILITY INCOMPLETE (results written, decision not taken); reported as a failed stage
    stage feas $PY -I e5/yelp_feasibility.py --zip "$YELP_ZIP" --data-dir "$YELP_JSON_DIR" \
      --work runs/e5_feas/work --out runs/e5_feas/out --n-proc "$NPROC"
  else
    echo "!! feas: $YELP_ZIP not found (set YELP_ZIP)"; FAILED+=("feas")
  fi
fi
if has bundle; then
  stage bundle bash -c '
    set -e
    TS=$(date +%Y%m%d_%H%M%S)
    rm -rf runs/yelp_docs && mkdir -p runs/yelp_docs   # the documentation PDFs from Yelp, kept outside out/
    cp -f "'"$YELP_JSON_DIR"'"/docs/*.pdf runs/yelp_docs/ 2>/dev/null || echo "no documentation PDF found"
    B=runs/e5_feas_bundle_$TS.tar.gz
    tar czf "$B" runs/e5_feas/out runs/logs runs/yelp_docs
    ls -l "$B"; sha256sum "$B"
    tar tzf "$B" | sed "s/^/  /"
    if tar tzf "$B" | grep -E "work/|yelp_academic_dataset|\.npz$|\.bin$"; then echo "BUNDLE CONTAINS DATA FILES"; rm -f "$B"; exit 1; fi'
fi
if [ "${#FAILED[@]}" -gt 0 ]; then
  printf 'failed stages:\n'; printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
echo "done: runs/e5_feas/out, bundle in runs/"
