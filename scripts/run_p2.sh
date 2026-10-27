#!/usr/bin/env bash
# P2 server package (lacs_p2 v0.3): E11 tolerance-multiplier curve (P2-4, with the optional P2-4b) and the E4
# rerun with the v0.5 engine (P2-5). See README.md. CPU only; run from anywhere (it cd's to the root).
#   STAGES="env build selftest p24pilot p24gate bundle" bash scripts/run_p2.sh      # batch 1
#   STAGES="p25 bundle" bash scripts/run_p2.sh                                     # batch 2
#   STAGES="p24 p24b p24sum bundle" bash scripts/run_p2.sh                         # batch 3
# Stages: env build selftest | p24pilot p24gate p24 p24b p24sum | p25 | bundle
# A failed stage prints "!! ..." and is listed at the end (exit code 1); stages that do not depend on it
# still run. Every stage resumes where it stopped when rerun (finished pieces have a DONE marker or a cached
# engine block). Inputs outside this directory are only read. A run that got the lock ends with the line
# "RUN_P2 END rc=<exit code> failed: <stages or none>" (with "INTERRUPTED" after rc when it was stopped by a
# signal); a run that did not get the lock prints "RUN_P2 NOT STARTED ..." and changes nothing.
set -uo pipefail
cd "$(dirname "$0")/.."
# single-threaded BLAS before any Python starts (P0: a BLAS thread pool and fork deadlocked)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=${PY:-python}
export PY
export PYTHONPATH="$PWD/p0:$PWD${PYTHONPATH:+:$PYTHONPATH}"
R4ROOT=${R4ROOT:-runs}             # round 4: gowalla/, brightkite/ (read only)
E5ROOT=${E5ROOT:-runs/e5_main}         # E5 Yelp inputs and results (read only)
E3RUN=${E3RUN:-runs/efficiency_round3}   # round-3 instances (read only)
P0ROOT=${P0ROOT:-runs_p0}         # P0 results, v0.4 reference of P2-5 (read only)
NPROC=${NPROC:-12}                                        # engine processes of E11 and P2-4b
E4_NPROC=${E4_NPROC:-8}                                   # engine processes of P2-5 (as in P0: timings comparable)
E4_CONFIG=${E4_CONFIG:-p2/configs/e4v05.yaml}             # (rehearsals only: another config)
PILOT_THRESHOLD_H=${PILOT_THRESHOLD_H:-12}                # plan 4.1: above -> Gowalla on 2,000 queries
PILOT_STOP_H=${PILOT_STOP_H:-24}                          # still above -> stop E11
STAGES=${STAGES:-"env build selftest p24pilot p24gate p25 p24 p24b p24sum bundle"}
DATASETS_E11=${DATASETS_E11:-"brightkite gowalla yelp"}
S_REST=${S_REST:-"0.5 0.67 0.8 1.25 1.5 2 3"}
E4_DATASETS=${E4_DATASETS:-"livejournal orkut"}
OUT=runs_p2
LOGS=$OUT/logs
mkdir -p "$LOGS"
FAILED=()
COMPLETED=0   # 1 once the run ends on purpose (normal end or a deliberate stop); signals leave it 0
if command -v flock > /dev/null; then
  exec 9> "$OUT/.run_p2.lock"
  if ! flock -n 9; then
    echo "RUN_P2 NOT STARTED: another run_p2.sh holds the lock $OUT/.run_p2.lock (nothing done; the other run is still going)"
    exit 1
  fi
else
  echo "warning: flock not found; running without the single-run lock"
fi
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
trap 'rc=$?; st=""; [ "$COMPLETED" = 1 ] || st=" INTERRUPTED"; printf "\nRUN_P2 END rc=%s%s failed: %s (%s)\n" "$rc" "$st" "${FAILED[*]:-none}" "$(date "+%F %T")"' EXIT

echo "run_p2 v0.3 ($(date '+%F %T')): STAGES=\"$STAGES\" PY=$PY"
echo "  R4ROOT=$R4ROOT E5ROOT=$E5ROOT E3RUN=$E3RUN P0ROOT=$P0ROOT"
echo "  NPROC=$NPROC E4_NPROC=$E4_NPROC E4_CONFIG=$E4_CONFIG PILOT_THRESHOLD_H=$PILOT_THRESHOLD_H PILOT_STOP_H=$PILOT_STOP_H"
echo "  DATASETS_E11=\"$DATASETS_E11\" S_REST=\"$S_REST\" E4_DATASETS=\"$E4_DATASETS\""

has () { [[ " $STAGES " == *" $1 "* ]]; }
stage () {  # stage <label> <command...>; output appended to $LOGS/<label>.log and shown on the console
  local label=$1; shift
  echo "== $label ($(date '+%F %T')) =="
  echo "== $label ($(date '+%F %T')) ==" >> "$LOGS/$label.log"
  local t0=$SECONDS
  "$@" > >(tee -a "$LOGS/$label.log") 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "!! $label failed (exit $rc, $(date '+%F %T'), $((SECONDS - t0)) s); see $LOGS/$label.log"
    FAILED+=("$label")
    return $rc
  fi
  echo "== $label done ($(date '+%F %T'), $((SECONDS - t0)) s) =="
}
export R4ROOT E5ROOT E3RUN P0ROOT NPROC E4_NPROC E4_CONFIG OUT

# per dataset: source run, gate reference and the expected query set of the E11 runs
e11_src () { case $1 in yelp) echo "$E5ROOT";; *) echo "$R4ROOT/$1";; esac; }
e11_ref () {  # <dataset> -> "<file> <kind>" of the s = 1 gate (plan 4.2)
  case $1 in
    yelp) echo "$E5ROOT/results/geo_e2_sweep_per_query.csv e5sweep";;
    *) echo "$R4ROOT/$1/results/r4/geo_e2_per_query.csv r4main";;
  esac
}
e11_mode () {  # <dataset> -> query options for p2.e11 run (decision of the pilot for Gowalla)
  local ds=$1
  case $ds in
    yelp) echo "--mode sample --n 2000 --expect-queries-from $E5ROOT/results/geo_e2_sweep_per_query.csv --expect-kind e5sweep";;
    gowalla)
      local m
      m=$("$PY" -c "import json; print(json.load(open('$OUT/e11/pilot/decision.json'))['gowalla_mode'])" 2>/dev/null)
      if [ "$m" = "r4_sample" ]; then
        echo "--mode sample --n 2000 --expect-queries-from $R4ROOT/gowalla/results/r4/geo_e2_sweep_per_query.csv --expect-kind r4sweep"
      elif [ "$m" = "all" ]; then
        echo "--mode all --expect-queries-from $R4ROOT/gowalla/results/r4/geo_e2_per_query.csv --expect-kind r4main"
      else
        echo "NO_DECISION"
      fi;;
    *) echo "--mode all --expect-queries-from $R4ROOT/$ds/results/r4/geo_e2_per_query.csv --expect-kind r4main";;
  esac
}
export -f e11_src e11_ref e11_mode
pilot_ok () {  # the pilot decided and did not ask to stop
  [ -f "$OUT/e11/pilot/decision.json" ] && \
    "$PY" -c "import json,sys; sys.exit(1 if json.load(open('$OUT/e11/pilot/decision.json'))['stop'] else 0)"
}
gates_ok () {  # every dataset's s = 1 gate passed
  for ds in $DATASETS_E11; do
    "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$OUT/e11/$ds/results/e11/gate_s1.json'))['pass'] else 1)" \
      2>/dev/null || return 1
  done
}

# ------------------------------------------------------------------ setup --
if has env; then
  # informational lines may fail; a broken Python, a missing input or a failed check stops the run
  stage env bash -c '
    "$PY" -V || { echo "!! no python ($PY): activate the lacs environment first"; exit 1; }
    "$PY" -c "import numpy, scipy, pandas, sklearn, yaml; print(numpy.__version__, scipy.__version__, pandas.__version__, sklearn.__version__)" \
      || { echo "!! python environment broken ($PY): activate the lacs environment"; exit 1; }
    g++ --version | head -1 || true; nproc --all || true; free -g || true; df -h . || true
    df -h . || true; cat /proc/loadavg || true
    miss=0
    for p in "$R4ROOT/gowalla/geo" "$R4ROOT/gowalla/graph/graph.bin" "$R4ROOT/gowalla/results/r4/geo_e2_per_query.csv" \
             "$R4ROOT/gowalla/results/r4/geo_e2_sweep_per_query.csv" \
             "$R4ROOT/brightkite/geo" "$R4ROOT/brightkite/graph/graph.bin" "$R4ROOT/brightkite/results/r4/geo_e2_per_query.csv" \
             "$E5ROOT/geo" "$E5ROOT/graph/graph.bin" "$E5ROOT/results/geo_e2_sweep_per_query.csv" \
             "$E3RUN/config_resolved.json" "$E3RUN/results/e3.json" "$E3RUN/work/e3_livejournal/graph.bin" \
             "$E3RUN/work/e3_orkut/graph.bin" "$P0ROOT/livejournal/e4_scale" "$P0ROOT/orkut/e4_scale"; do
      if [ -e "$p" ]; then echo "input ok: $p"; else echo "!! input missing: $p"; miss=1; fi
    done
    [ $miss -eq 0 ] || { echo "!! inputs missing (see above)"; exit 1; }
    "$PY" -I p2/fingerprint.py record --out "$OUT/logs/readonly_before.json" "$(dirname "$R4ROOT")" \
      "$(dirname "$E5ROOT")" "$E3RUN" "$P0ROOT" || exit 1
    "$PY" -I p2/manifest.py check || exit 1' || { echo "ENV CHECK FAILED: stopping (see $LOGS/env.log)"; COMPLETED=1; exit 1; }
fi
if has build; then
  # no forced rebuild: the engine is rebuilt only when its sources changed (digest stamp), so a failed
  # compiler never removes a working binary
  stage build bash -c '"$PY" -c "from lacs.search.cpp import ensure_built; print(ensure_built())" && bash p0/build.sh'
fi
if has selftest; then
  if stage selftest env PYTHONWARNINGS=ignore::ResourceWarning "$PY" -m unittest -v p2.tests.test_p2; then
    echo "SELFTEST PASSED"
  else
    echo "SELFTEST FAILED: stopping (see $LOGS/selftest.log)"
    COMPLETED=1
    exit 1
  fi
fi

# ----------------------------------------------------------- P2-4 (E11) --
if has p24pilot; then
  # inputs (copies with sha256 check), then 200 queries per dataset at s = 3, then the decision of plan 4.1
  stage p24pilot bash -c '
    set -e
    for ds in '"$DATASETS_E11"'; do
      "$PY" -m p2.e11 prepare-inputs --src "$(e11_src $ds)" --dst "$OUT/e11/$ds"
    done
    specs=()
    for ds in '"$DATASETS_E11"'; do
      case $ds in
        yelp) q="--mode sample --n 2000";;
        *) q="--mode all";;
      esac
      # --fresh: an interrupted earlier pilot leaves cached engine blocks; they are removed so the timing is real
      "$PY" -m p2.e11 run --config p2/configs/e11_$ds.yaml --s 3 $q --subset 200 --results-name e11_pilot \
        --n-proc "$NPROC" --fresh
      specs+=("$ds=$OUT/e11/$ds/results/e11_pilot/s3")
    done
    mkdir -p "$OUT/e11/pilot"
    "$PY" -m p2.e11 decide --pilot "${specs[@]}" --threshold-h "'"$PILOT_THRESHOLD_H"'" \
      --stop-h "'"$PILOT_STOP_H"'" --out "$OUT/e11/pilot/decision.json"'
fi
if has p24gate; then
  if pilot_ok; then
    # s = 1 for every dataset first; the gate (plan 4.2) decides whether the other multipliers run
    stage p24gate bash -c '
      rc=0; gate_failed=0
      for ds in '"$DATASETS_E11"'; do
        opts=$(e11_mode $ds)
        if [ "$opts" = "NO_DECISION" ]; then echo "!! $ds: no pilot decision"; rc=1; continue; fi
        if ! "$PY" -m p2.e11 run --config p2/configs/e11_$ds.yaml --s 1 $opts --n-proc "$NPROC"; then
          echo "!! $ds: s = 1 run failed"; rc=1; continue
        fi
        set -- $(e11_ref $ds)
        "$PY" -m p2.e11 gate --run "$OUT/e11/$ds/results/e11/s1/e11_per_query.csv" --ref "$1" --ref-kind "$2" \
          --out "$OUT/e11/$ds/results/e11/gate_s1.json" || gate_failed=1
      done
      if [ $gate_failed -eq 1 ]; then echo "E11 GATE FAILED: the other multipliers will not run (plan 4.2)"; exit 4; fi
      exit $rc'
  else
    echo "!! p24gate: no pilot decision, or the pilot asked to stop ($OUT/e11/pilot/decision.json)"; FAILED+=("p24gate")
  fi
fi

# ---------------------------------------------------------- P2-5 (E4 v0.5) --
if has p25; then
  # order of plan 5.1: e4_scale, then e4_queries, then e4_mix; the first failed gate stops P2-5 (plan 5.2)
  stage p25 bash -c '
    rc=0
    for exp in e4_scale e4_queries e4_mix; do
      for ds in '"$E4_DATASETS"'; do
        "$PY" -m p2.e4v05 run --config "$E4_CONFIG" --dataset $ds --experiment $exp --e3-run "$E3RUN" \
          --p0-root "$P0ROOT" --n-proc "$E4_NPROC"
        rc=$?
        if [ $rc -ne 0 ]; then echo "!! P2-5 stopped at $ds $exp (exit $rc): later settings not run"; break 2; fi
      done
    done
    "$PY" -m p2.e4v05 summary --root "$OUT/e4v05" --p0-root "$P0ROOT" --out "$OUT/e4v05/report" || rc=${rc/#0/1}
    exit $rc'
fi

# --------------------------------------------------- P2-4 (E11), the rest --
if has p24; then
  if pilot_ok && gates_ok; then
    stage p24 bash -c '
      rc=0
      for ds in '"$DATASETS_E11"'; do
        opts=$(e11_mode $ds)
        for s in '"$S_REST"'; do
          "$PY" -m p2.e11 run --config p2/configs/e11_$ds.yaml --s $s $opts --n-proc "$NPROC" || { echo "!! $ds s=$s failed"; rc=1; }
        done
      done
      exit $rc'
  else
    echo "!! p24: needs the pilot decision (no stop) and passed s = 1 gates for $DATASETS_E11"; FAILED+=("p24")
  fi
fi
if has p24b; then
  if pilot_ok && gates_ok; then
    # plan 4.4 (decided 10-26): LCC groups at s = 0.8 and 1, Gowalla and Brightkite
    # the groups are read back from E11's cached exact runs at s = 1 and 0.8 (no engine run, same groups as E11)
    stage p24b bash -c '
      set -e
      for ds in gowalla brightkite; do
        opts=$(e11_mode $ds)
        for s in 1 0.8; do
          "$PY" -m p2.e11 export-groups --config p2/configs/e11_$ds.yaml --s $s $opts --out "$OUT/p24b/$ds/s$s/e1_input"
          "$PY" -m e1x.open_sweep --config p2/configs/p24b_dynamics.yaml \
            --instance "$OUT/p24b/$ds/s$s/e1_input/instance.json" --groups "$OUT/p24b/$ds/s$s/e1_input/groups.jsonl.gz" \
            --out "$OUT/p24b/$ds/s$s/dyn" --lcc-only --workers "$NPROC" --dataset $ds --setting "s$s"
        done
      done
      "$PY" -m p2.p24b --root "$OUT/p24b" --out "$OUT/p24b/report"'
  else
    echo "!! p24b: needs the pilot decision (no stop) and passed s = 1 gates"; FAILED+=("p24b")
  fi
fi
if has p24sum; then
  stage p24sum "$PY" -m p2.e11 summary --root "$OUT/e11" --out "$OUT/e11/report"
fi

# ------------------------------------------------------------------ bundle --
if has bundle; then
  stage bundle bash -c '
    if [ -f "$OUT/logs/readonly_before.json" ]; then
      "$PY" -I p2/fingerprint.py compare --before "$OUT/logs/readonly_before.json" --out "$OUT/logs/readonly_check.json"
    else
      echo "READ-ONLY INPUTS: no fingerprint recorded (the env stage has not run)"
    fi
    "$PY" -I p2/bundle.py --root . --out "$OUT/p2_bundle_$(date +%Y%m%d_%H%M%S).tar.gz"'
fi
COMPLETED=1
if [ "${#FAILED[@]}" -gt 0 ]; then
  printf 'failed stages:\n'; printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
echo "done: $OUT (bundle in $OUT/)"
