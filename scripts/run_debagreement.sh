#!/usr/bin/env bash
# Main dataset. Stops after the go/no-go gates so you can look at them first:
#   bash scripts/run_debagreement.sh gates   -> prepare, encode, features, g1, train (G2), graph, g3
#   bash scripts/run_debagreement.sh rest    -> e2, report
#   bash scripts/run_debagreement.sh all     -> everything
set -euo pipefail
CFG=configs/debagreement.yaml
STAGE=${1:-gates}
shift || true
if [[ "$STAGE" == "gates" || "$STAGE" == "all" ]]; then
  for s in prepare encode features g1 train graph g3; do python -m lacs "$s" -c "$CFG" "$@"; done
  echo "---- gates ----"
  python - <<'EOF'
import json
for g in ("g1", "g2", "g3"):
    try:
        d = json.load(open(f"runs/debagreement/results/{g}.json"))
        print(g.upper(), "PASS" if d.get("pass") else "FAIL", "|", d.get("criterion", ""))
    except FileNotFoundError:
        print(g.upper(), "missing")
d = json.load(open("runs/debagreement/results/g3.json"))
print("recommended k:", d.get("recommended_k"))
EOF
fi
if [[ "$STAGE" == "rest" || "$STAGE" == "all" ]]; then
  for s in e2 report; do python -m lacs "$s" -c "$CFG" "$@"; done
  echo "report: runs/debagreement/results/REPORT.md"
fi
