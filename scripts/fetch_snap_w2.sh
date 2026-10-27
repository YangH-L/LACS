#!/usr/bin/env bash
# Downloads the SNAP graphs of W2 run B into data/raw/snap/ (skips files that are already there).
#   bash scripts/fetch_snap_w2.sh            # all four
#   FRIENDSTER=0 bash scripts/fetch_snap_w2.sh   # without Friendster (8.7 GB download, 31 GB unpacked)
# Records size and sha256 of every unpacked file in data/raw/snap/SHA256SUMS_w2.
set -uo pipefail
DIR=data/raw/snap
mkdir -p "$DIR"
URLS=(
  https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz
  https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz
  https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz
)
if [ "${FRIENDSTER:-1}" = "1" ]; then
  URLS+=(https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz)
fi
if [ -n "${W2_URLS:-}" ]; then  # testing: space-separated replacement list
  read -r -a URLS <<< "$W2_URLS"
fi
fail=0
for url in "${URLS[@]}"; do
  gz="$DIR/$(basename "$url")"
  txt="${gz%.gz}"
  if [ -s "$txt" ]; then
    echo "have $txt"
    continue
  fi
  if [ -s "$gz" ] && gunzip -t "$gz" 2>/dev/null; then
    echo "have $gz (downloaded earlier)"
  else
    rm -f "$gz"
    echo "downloading $url"
    if ! curl -fL --retry 5 --retry-delay 10 -C - -o "$gz.part" "$url"; then
      echo "!! download failed: $url (rerun to resume)"
      fail=1
      continue
    fi
    mv "$gz.part" "$gz"
  fi
  if ! gunzip -t "$gz"; then
    echo "!! corrupt archive: $gz (deleted; rerun to download again)"
    rm -f "$gz"
    fail=1
    continue
  fi
  # unpack to a temporary name: an interrupted unpack never leaves a truncated $txt behind
  if gunzip -c "$gz" > "$txt.part"; then
    mv "$txt.part" "$txt" && rm -f "$gz"
    echo "unpacked $txt ($(du -h "$txt" | cut -f1))"
  else
    echo "!! gunzip failed: $gz"
    rm -f "$txt.part"
    fail=1
  fi
done
( cd "$DIR" && for f in com-dblp.ungraph.txt com-youtube.ungraph.txt soc-pokec-relationships.txt com-friendster.ungraph.txt; do
    [ -s "$f" ] || continue
    # SNAP's header line "# Nodes: N Edges: M" (if any) and the last line, as a check against truncation
    echo "$(sha256sum "$f" | cut -d' ' -f1)  $(stat -c %s "$f")  $f  [$(head -5 "$f" | grep -m1 -o 'Nodes: [0-9]* Edges: [0-9]*' || true)] [last: $(tail -1 "$f" | tr '\t' ' ')]"
  done ) > "$DIR/SHA256SUMS_w2"
cat "$DIR/SHA256SUMS_w2"
exit $fail
