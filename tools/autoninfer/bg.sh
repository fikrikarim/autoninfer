#!/bin/bash
# autoninfer background-job ledger: liveness tracking for detached long-running
# work (downloads, pipeline runs) so a dead job is caught at the next iteration's
# step 0 instead of hours later.
#
#   bg.sh add <name> <progress-file> [note]   register a job. The progress file
#                                             must grow or change mtime while the
#                                             job is alive (the output file for a
#                                             download, the .log for a pipeline).
#   bg.sh check                                list all jobs; STALE = no progress
#                                             in 15 min (BG_STALE_S overrides);
#                                             exit 1 if any stale/missing
#   bg.sh forget <name>                        remove an entry (job done/abandoned)
#
# Ledger: /tmp/autoninfer-bg/ledger.tsv (name \t progress_file \t note)
set -uo pipefail

DIR=/tmp/autoninfer-bg
LEDGER="$DIR/ledger.tsv"
STALE_S=${BG_STALE_S:-900}
mkdir -p "$DIR"
touch "$LEDGER"

case "${1:-}" in
add)
  name="${2:?usage: bg.sh add <name> <progress-file> [note]}"
  path="${3:?usage: bg.sh add <name> <progress-file> [note]}"
  note="${4:-}"
  grep -v "^$name	" "$LEDGER" > "$LEDGER.tmp" || true
  printf '%s\t%s\t%s\n' "$name" "$path" "$note" >> "$LEDGER.tmp"
  mv "$LEDGER.tmp" "$LEDGER"
  echo "registered $name -> $path"
  ;;
check)
  stale=0
  now=$(date +%s)
  printf '%-22s %-9s %s\n' "name" "state" "progress-file (note)"
  while IFS=$'\t' read -r name path note; do
    [ -z "$name" ] && continue
    if [ ! -f "$path" ]; then
      printf '%-22s %-9s %s (%s) [FILE MISSING]\n' "$name" "STALE" "$path" "$note"
      stale=1
      continue
    fi
    mt=$(stat -c %Y "$path")
    sz=$(stat -c %s "$path")
    age=$(( now - mt ))
    if [ "$age" -gt "$STALE_S" ]; then
      printf '%-22s %-9s %s (%s) [no progress %ds, %s bytes]\n' "$name" "STALE" "$path" "$note" "$age" "$sz"
      stale=1
    else
      printf '%-22s %-9s %s (%s) [%ds ago, %s bytes]\n' "$name" "ok" "$path" "$note" "$age" "$sz"
    fi
  done < "$LEDGER"
  exit $stale
  ;;
forget)
  name="${2:?usage: bg.sh forget <name>}"
  grep -v "^$name	" "$LEDGER" > "$LEDGER.tmp" || true
  mv "$LEDGER.tmp" "$LEDGER"
  echo "forgotten $name"
  ;;
*)
  echo "usage: bg.sh add <name> <progress-file> [note] | check | forget <name>" >&2
  exit 2
  ;;
esac