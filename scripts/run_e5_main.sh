#!/usr/bin/env bash
# E5 main experiment (Yelp geographic application), E1 open-group sweep and E6 (Congress) — see
# README.md. CPU only; run from this directory or anywhere (it cd's to the package root).
#   bash scripts/run_e5_main.sh                             # every stage, in the order below
#   STAGES="e6fetch e6build" bash scripts/run_e5_main.sh    # selected stages
# Stages: env build selftest | e5prep e5gates e5pilot e5main e5sweep e5analysis | e1sweep |
#         e6fetch e6build e6geo e6dyn e6premise e6report | bundle
# A failed stage is reported (line "!! ...") and the independent stages after it still run (a stage
# whose input is missing fails fast); the exit code is 1 if any stage failed. e5main / e5sweep /
# e1sweep / e6dyn resume where they stopped when rerun.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PY
export PYTHONPATH="$PWD/p0:$PWD${PYTHONPATH:+:$PYTHONPATH}"
YELP_JSON_DIR=${YELP_JSON_DIR:-data/yelp/json}
P0RUNS=${P0RUNS:-runs_p0}
E6DATA=${E6DATA:-data/e6}
NPROC=${NPROC:-12}
STAGES=${STAGES:-"env build selftest e5prep e5gates e5pilot e5main e5sweep e5analysis e1sweep e6fetch e6build e6geo e6dyn e6premise e6report bundle"}
LOGS=runs/logs
mkdir -p "$LOGS"
FAILED=()
if command -v flock > /dev/null; then
  exec 9> runs/.run_e5_main.lock
  if ! flock -n 9; then
    echo "!! another run_e5_main.sh is running (lock runs/.run_e5_main.lock): nothing done"
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
  local t0=$SECONDS
  if ! "$@" > >(tee -a "$LOGS/$label.log") 2>&1; then
    echo "!! $label failed ($(date '+%F %T'), $((SECONDS - t0)) s); see $LOGS/$label.log"
    FAILED+=("$label")
    return 1
  fi
  echo "== $label done ($(date '+%F %T'), $((SECONDS - t0)) s) =="
}
SET="--set search.n_proc=$NPROC"

if has env; then
  stage env bash -c '
    "$PY" -V; "$PY" -c "import numpy, scipy, pandas, sklearn, yaml; print(numpy.__version__, scipy.__version__, pandas.__version__, sklearn.__version__)"
    g++ --version | head -1; nproc; free -g; df -h . 2>/dev/null
    lscpu | grep -E "Model name|^CPU\(s\)|Thread|Socket"; cat /proc/loadavg
    "$PY" -c "from p0kit.runner_common import kit_digest; print(\"p0 kit\", kit_digest())"
    sha256sum lacs/apps/geo.py e5/*.py e5/tests/*.py e1x/*.py e1x/configs/*.yaml e1x/tests/*.py e6/*.py e6/data/* \
      configs/e5_yelp*.yaml configs/e6_congress.yaml cpp/src/* p0/cpp/* p0/p0kit/*.py scripts/run_e5_main.sh \
      | tee runs/logs/e5_main_sources.sha256'
fi
if has build; then
  stage build bash -c '"$PY" -m lacs build-cpp && bash p0/build.sh'
fi
if has selftest; then
  if stage selftest "$PY" -m unittest -v e1x.tests.test_e1x e5.tests.test_main e5.tests.test_feasibility \
       e6.tests.test_e6; then
    echo "SELFTEST PASSED"
  else
    echo "SELFTEST FAILED: stopping (see $LOGS/selftest.log)"
    exit 1
  fi
fi

# ---------------------------------------------------------------- E5 (Yelp) --
if has e5prep; then
  # exit code 3 = the rebuilt inputs differ from the feasibility check: nothing written, later E5 stages fail fast
  stage e5prep "$PY" -I e5/yelp_prepare.py --json-dir "$YELP_JSON_DIR" --work runs/e5_feas/work \
    --feas runs/e5_feas/out/feasibility.json --out-dir runs/e5_main
fi
e5_ready () { grep -q '"status": "OK"' runs/e5_main/results/e5_prepare.json 2>/dev/null; }
need_e5 () { if ! e5_ready; then echo "!! $1: E5 inputs not prepared (stage e5prep)"; FAILED+=("$1"); return 1; fi; }
if has e5gates && need_e5 e5gates; then
  stage e5gates "$PY" -m lacs geo-gates -c configs/e5_yelp.yaml $SET
fi
if has e5pilot && need_e5 e5pilot; then
  T0=$SECONDS
  if stage e5pilot "$PY" -m lacs geo-e2 -c configs/e5_yelp_pilot.yaml $SET; then
    DT=$((SECONDS - T0))
    awk -v dt="$DT" 'BEGIN { printf "pilot: 120 queries in %d s -> rough projection: main (10,363 queries) %.1f h, sweep (14,000 query-settings) %.1f h\n", dt, dt * 10363 / 120 / 3600, dt * 14000 / 120 / 3600 }' \
      | tee -a "$LOGS/e5pilot.log"
  fi
fi
if has e5main && need_e5 e5main; then
  stage e5main "$PY" -m lacs geo-e2 -c configs/e5_yelp.yaml $SET
fi
if has e5sweep && need_e5 e5sweep; then
  stage e5sweep "$PY" -m lacs geo-e2-sweep -c configs/e5_yelp.yaml $SET
fi
if has e5analysis; then
  if [ -f runs/e5_main/results/geo_e2_per_query.csv ]; then
    stage e5analysis "$PY" e5/analysis.py --results runs/e5_main/results --title "E5 main analysis (Yelp, G, precovid)"
  else
    echo "!! e5analysis: runs/e5_main/results/geo_e2_per_query.csv missing (stage e5main)"; FAILED+=("e5analysis")
  fi
fi

# ------------------------------------------------- E1 open-group sweep (P0) --
if has e1sweep; then
  stage e1sweep bash -c '
    ok=(); rc=0
    for ds in gowalla brightkite; do   # one dataset failing does not stop the other
      if "$PY" -m e1x.open_sweep --config e1x/configs/e1_open_meetup.yaml \
        --instance "'"$P0RUNS"'/$ds/e1_input/instance.json" --groups "'"$P0RUNS"'/$ds/e1_input/groups.jsonl.gz" \
        --out runs/e1_open/$ds --queries-from "'"$P0RUNS"'/$ds/e1/default/e1_rows.jsonl.gz" --workers "'"$NPROC"'"; then
        ok+=("$ds")
      else
        echo "!! e1sweep: $ds failed"; rc=1
      fi
    done
    [ "${#ok[@]}" -gt 0 ] || exit 1
    runs=(); p0=()
    for ds in "${ok[@]}"; do runs+=("runs/e1_open/$ds"); p0+=("$ds='"$P0RUNS"'/$ds/e1/default/e1_rows.jsonl.gz"); done
    "$PY" -m e1x.summarize --runs "${runs[@]}" --out runs/e1_open/report --p0-rows "${p0[@]}" || rc=1
    exit $rc'
fi

# ------------------------------------------------------------ E6 (Congress) --
if has e6fetch; then
  # E6_FETCH_ARGS: extra options, e.g. "--offline-dir DIR" when the files were placed by hand
  stage e6fetch "$PY" -m e6.fetch --out "$E6DATA" ${E6_FETCH_ARGS:-}
fi
if has e6build; then
  # a rebuild starts from a clean runs/e6 (an earlier one is kept as runs/e6_prev_<time>), so that no stage
  # can mix the new inputs with an old P0 instance, export or dynamics checkpoint
  [ -d runs/e6 ] && mv runs/e6 "runs/e6_prev_$(date +%Y%m%d_%H%M%S)"
  # exit code 3 = data gate (fewer than 400 eligible members), 2 = incomplete or unparseable data:
  # later E6 stages fail fast. E6_BUILD_ARGS: extra options (rehearsals only, e.g. a synthetic --network)
  stage e6build "$PY" -m e6.build --data "$E6DATA" --out runs/e6 ${E6_BUILD_ARGS:-}
fi
e6_ready () { [ -f runs/e6/geo/users.csv ] && [ -f runs/e6/graph/graph.bin ]; }
need_e6 () { if ! e6_ready; then echo "!! $1: E6 inputs not built (stage e6build)"; FAILED+=("$1"); return 1; fi; }
if has e6geo && need_e6 e6geo; then
  # order: what Q2/Q3 and the section 4 rule need comes first; the sweep and the Q4 analysis last, and a
  # failure of one step does not skip the independent ones
  stage e6geo bash -c '
    rc=0
    step () { echo "-- $*"; "$@" || { echo "!! step failed: $*"; rc=1; return 1; }; }
    step "$PY" -m lacs geo-gates -c configs/e6_congress.yaml '"$SET"'
    if step "$PY" -m lacs geo-e2 -c configs/e6_congress.yaml '"$SET"'; then
      step "$PY" -m e6.structure --geo runs/e6 --results runs/e6/results
      if step "$PY" -m p0kit.export_groups geo --config configs/e6_congress.yaml --out runs/e6/e1_input --force; then
        step "$PY" -m e6.static --instance runs/e6/e1_input/instance.json \
          --groups runs/e6/e1_input/groups.jsonl.gz --out runs/e6/results
      fi
      step "$PY" -m lacs geo-e2-sweep -c configs/e6_congress.yaml '"$SET"'
      step "$PY" e5/analysis.py --results runs/e6/results --out runs/e6/results/e6_q4_analysis --title "E6 Q4 (Congress)"
    fi
    exit $rc'
fi
if has e6dyn; then
  if [ -f runs/e6/e1_input/groups.jsonl.gz ]; then
    stage e6dyn bash -c '
      set -e
      "$PY" -m e1x.open_sweep --config e1x/configs/e6_dynamics.yaml --instance runs/e6/e1_input/instance.json \
        --groups runs/e6/e1_input/groups.jsonl.gz --weights runs/e6/geo/twitter_weights.npz \
        --out runs/e6/e1x/e6 --lcc-only --workers "'"$NPROC"'"
      "$PY" -m e1x.summarize --runs runs/e6/e1x/e6 --out runs/e6/results'
  else
    echo "!! e6dyn: runs/e6/e1_input/groups.jsonl.gz missing (stage e6geo)"; FAILED+=("e6dyn")
  fi
fi
if has e6premise && need_e6 e6premise; then
  stage e6premise "$PY" -m e6.premise --geo runs/e6 --out runs/e6/results --n-perm "${E6_NPERM:-1000}" \
    --processes "$NPROC"
fi
if has e6report; then
  stage e6report "$PY" -m e6.report --geo runs/e6 --results runs/e6/results --data "$E6DATA"
fi

# ------------------------------------------------------------------ bundle --
if has bundle; then
  stage bundle bash -c '
    set -e
    TS=$(date +%Y%m%d_%H%M%S)
    B=runs/e5_main_bundle_$TS.tar.gz
    rm -rf runs/e6_public && mkdir -p runs/e6_public/voteview runs/e6_public/billstatus
    # the downloaded public E6 inputs that matter for checking: manifest, log, Voteview table, directory listings
    cp -f "'"$E6DATA"'"/fetch_manifest.json "'"$E6DATA"'"/fetch.log runs/e6_public/ 2>/dev/null || true
    cp -f "'"$E6DATA"'"/voteview/*.csv runs/e6_public/voteview/ 2>/dev/null || true
    for t in hr s hjres sjres; do
      [ -d "'"$E6DATA"'/billstatus/$t" ] || continue
      mkdir -p runs/e6_public/billstatus/$t
      cp -f "'"$E6DATA"'"/billstatus/$t/listing.* runs/e6_public/billstatus/$t/ 2>/dev/null || true
    done
    ITEMS=""
    for p in runs/e5_main/results runs/e1_open runs/e6 runs/e6_public runs/logs; do [ -e "$p" ] && ITEMS="$ITEMS $p"; done
    # the Yelp-derived geo/, graph/ and work/ of runs/e5_main stay on the server; E6 is public data
    tar czf "$B" --exclude="runs/e6/work" $ITEMS
    ls -l "$B"; sha256sum "$B"
    tar tzf "$B" | sed "s/^/  /" | head -400
    if tar tzf "$B" | grep -E "e5_main/(geo|graph|work)/|e5_feas/work|yelp_academic_dataset|parsed\.npz"; then
      echo "BUNDLE CONTAINS YELP DATA FILES"; rm -f "$B"; exit 1; fi'
fi
if [ "${#FAILED[@]}" -gt 0 ]; then
  printf 'failed stages:\n'; printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
echo "done: runs/e5_main/results, runs/e1_open, runs/e6; bundle in runs/"
