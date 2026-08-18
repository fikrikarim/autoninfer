#!/bin/bash
# Poll a detached experiment pipeline (experiment_run.sh) until it finishes or
# the poll window expires. Prints the .result summary on completion.
#
#   tools/autoninfer/experiment_wait.sh <name> [poll_window_s, default 1800]
#
# Exit codes: 0 pipeline ok, 1 pipeline failed, 2 still running when the
# window expired (the agent should either keep waiting or narrow scope; the
# pipeline keeps running either way - check <name>.status for its stage).
set -uo pipefail

NAME="${1:?usage: experiment_wait.sh <name> [poll_window_s]}"
WINDOW="${2:-1800}"
DIR=/tmp/autoninfer-exps
STATUS="$DIR/$NAME.status"
RESULT="$DIR/$NAME.result"

[ -f "$DIR/$NAME.running" ] || { echo "no running pipeline for $NAME (last: $(cat "$STATUS" 2>/dev/null || echo none))" >&2; [ -f "$RESULT" ] && cat "$RESULT"; exit 1; }

deadline=$(( $(date +%s) + WINDOW ))
while :; do
  if [ -f "$RESULT" ]; then
    cat "$RESULT"
    [ "$(head -1 "$RESULT" | cut -d= -f2)" = "ok" ] && exit 0 || exit 1
  fi
  [ "$(date +%s)" -ge "$deadline" ] && { echo "STILL RUNNING: $(cat "$STATUS" 2>/dev/null)" >&2; exit 2; }
  sleep 15
done