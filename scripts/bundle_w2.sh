#!/usr/bin/env bash
# Packs what W2 brings back: per-query results, summaries, run metadata and logs (no graphs, positions,
# tolerances or member lists). Output: runs/w2_bundle_<time>.tar.gz and its sha256.
set -euo pipefail
cd "$(dirname "$0")/.."
T=$(date +%Y%m%d_%H%M)
LIST=$(mktemp)
for r in runs/efficiency_w2 runs/efficiency_w2_graphs; do
  [ -d "$r" ] || continue
  find "$r/results" -maxdepth 1 -type f \( -name '*.csv' -o -name '*.json' \) >> "$LIST" 2>/dev/null || true
  [ -f "$r/config_resolved.json" ] && echo "$r/config_resolved.json" >> "$LIST"
  [ -f "$r/lacs.log" ] && echo "$r/lacs.log" >> "$LIST"
  find "$r/work" -maxdepth 2 -type f \( -name 'checkpoint.json' -o -name 'graph.src.json' \) >> "$LIST" 2>/dev/null || true
  # finished datasets of an interrupted run: their rows are only in the checkpoints
  find "$r/work" -maxdepth 2 -type f -name 'checkpoint_rows.csv' >> "$LIST" 2>/dev/null || true
done
find runs/logs -maxdepth 1 -type f >> "$LIST"
[ -f data/raw/snap/SHA256SUMS_w2 ] && echo data/raw/snap/SHA256SUMS_w2 >> "$LIST"
ls configs/efficiency_w2*.yaml >> "$LIST"
OUT=runs/w2_bundle_$T.tar.gz
tar -czf "$OUT" -T "$LIST"
rm -f "$LIST"
sha256sum "$OUT" | tee "$OUT.sha256"
ls -l "$OUT"
tar -tzf "$OUT" | wc -l
