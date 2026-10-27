#!/usr/bin/env bash
# Lay out the result directories of this repository the way the analysis scripts expect them (they were
# written against the result bundles of the run scripts). Only symbolic links are created.
#   bash analysis/make_workdir.sh [WORK=work]
# Afterwards, for example:
#   python3 -I analysis/paper/w2_numbers.py runs runs/efficiency_round3/results/e3_per_query.csv work/derived/w2_numbers.json
#   python3 -I analysis/paper/p0_e3_numbers.py runs_p0 work/derived/p0_e3_numbers.json
#   python3 -I analysis/paper/geo_size_check.py runs work/paper_base/notes/analysis/geo_size_check.json
#   python3 -I analysis/paper/geo_posthoc_check.py runs work/paper_base/notes/analysis/geo_posthoc.json
#   python3 -I analysis/paper/make_tables.py work <paper>/tables
#   python3 -I analysis/p2/p21_unsolvable.py work work/p21
#   python3 -I analysis/p2/p22_open_groups.py work work/p22
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
W=${1:-work}
mkdir -p "$W/derived" "$W/r4res" "$W/e5b" "$W/p0bundle" "$W/paper_base/notes/analysis"
link () { [ -e "$2" ] || [ -L "$2" ] || ln -s "$1" "$2"; }
link "$ROOT/runs" "$W/r4res/runs"            # round 4: runs/{gowalla,brightkite}/results/r4/
link "$ROOT/runs" "$W/e5b/runs"              # E5, open groups, Congress: runs/{e5_main,e1_open,e6}/
link "e5b" "$W/e5bundle"                     # make_tables.py names the same bundle e5bundle
link "$ROOT/runs_p0" "$W/p0bundle/runs_p0"   # P0: runs_p0/report/e1_summary.csv
link "$ROOT/runs_p2/p2x" "$W/p2x"            # Friendster k-core reference (Table I)
echo "work directory: $W"
