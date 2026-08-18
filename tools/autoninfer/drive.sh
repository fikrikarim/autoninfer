#!/bin/bash
# Unattended autoninfer driver.
#
# Runs headless pi iterations (print mode) on a dedicated session id
# (autoninfer-driver) so the research loop keeps running while the user is away.
# Between iterations — when no pi process is running, so the GPU 0 serve can be
# touched without killing an in-flight agent turn — it applies pending serve
# ops written to $OPS_DIR by the agent.
#
#   start: supervisorctl start autoninfer-driver
#   stop:  supervisorctl stop autoninfer-driver     (or: touch /tmp/autoninfer-stop)
#   ops:   echo '{"action":"restart-primary"}' > /tmp/autoninfer-ops/pending.json
#   log:   /var/log/autoninfer-drive.log
#
# The iteration budget (MAX_ITER) is a hard stop; the service exits normally
# (autorestart=unexpected does not revive a clean exit).
set -uo pipefail

REPO=/workspace/autoninfer
OPS_DIR=/tmp/autoninfer-ops
STOP_FLAG=/tmp/autoninfer-stop
LOG=/var/log/autoninfer-drive.log
SESSION_ID=autoninfer-driver
MAX_ITER=${MAX_ITER:-24}
ITER=0

. /opt/nvm/nvm.sh >/dev/null 2>&1
mkdir -p "$OPS_DIR"
cd "$REPO"

say() { echo "$(date -Is) $*" | tee -a "$LOG"; }

PROMPT='Unattended autoninfer iteration. Follow docs/autoninfer/README.md (the autoninfer protocol) strictly.
1. If /tmp/autoninfer-ops/ contains *.done or *.failed files, read them first - the driver already applied those serve ops between iterations.
2. Run `bash tools/gpu_health.sh 1`. If it is not HEALTHY, do NOT run GPU experiments; record the finding in docs/autoninfer/results.tsv and end the iteration.
3. Pick the single highest-value next experiment from the hypothesis backlog and open threads (docs/autoninfer/README.md, docs/autoninfer/results.tsv).
4. Run it on GPU 1 only (CUDA_VISIBLE_DEVICES=1). NEVER bind, kill, restart, or reconfigure the GPU 0 serve - it powers you. If a serve flag change is needed, write {"action":"restart-primary"} to /tmp/autoninfer-ops/pending.json (the driver applies it between iterations) and continue with work that does not need the new flags.
5. Log every experiment to docs/autoninfer/results.tsv (one row; protocol: the README). Commit repo changes (Conventional Commit subject) and push to origin.
6. Stop the loop by writing a reason to /tmp/autoninfer-stop if: a decision requires the user, GPU 1 is unhealthy and no fix is available, or the backlog is exhausted. Then end the iteration.'

while [ "$ITER" -lt "$MAX_ITER" ]; do
  if [ -f "$STOP_FLAG" ]; then
    say "stop flag present - exiting"
    break
  fi

  # Apply pending serve ops. The driver itself runs ops between its own iterations
  # (no driver pi child alive), but an INTERACTIVE pi session may still be running
  # (ps args: "pi" or "pi -c"). Restarting the primary then would kill that session's
  # model mid-turn - so defer the op until no other pi process exists.
  if [ -f "$OPS_DIR/pending.json" ]; then
    other_pi=$(pgrep -x pi || true)
    if [ -n "$other_pi" ]; then
      say "op pending but pi session(s) running (pids: $(echo $other_pi | tr '\n' ' ')) - deferring until they exit"
    else
      action=$(sed -n 's/.*"action"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$OPS_DIR/pending.json")
    case "$action" in
      restart-primary)
        say "op: restart-primary"
        # Take over from a non-supervised (loose) serve process if one still holds the port:
        # pkill it and hand the port to the supervised service (one-shot migration hop).
        svc_state=$(supervisorctl status ninfer-serve 2>/dev/null | awk '{print $2}')
        if [ "$svc_state" != "RUNNING" ]; then
          if pgrep -f 'ninfer-serve models/' >/dev/null 2>&1; then
            say "loose serve process found - killing it, handing the port to supervisor"
            pkill -f 'ninfer-serve models/' || true
            sleep 3
          fi
          supervisorctl start ninfer-serve >>"$LOG" 2>&1 || true
        else
          supervisorctl restart ninfer-serve >>"$LOG" 2>&1 || true
        fi
        ok=0
        for _ in $(seq 1 90); do
          if curl -sf -m 2 http://127.0.0.1:8080/v1/models >/dev/null 2>&1; then ok=1; break; fi
          sleep 2
        done
        if [ "$ok" = 1 ]; then
          mv "$OPS_DIR/pending.json" "$OPS_DIR/restart-primary.done"
          say "op restart-primary applied"
        else
          mv "$OPS_DIR/pending.json" "$OPS_DIR/restart-primary.failed"
          say "op restart-primary FAILED: primary unhealthy after 180s - stopping (no model to iterate on)"
          break
        fi
        ;;
      *)
        say "unknown op action: $action (ignored)"
        rm -f "$OPS_DIR/pending.json"
        ;;
      esac
    fi
  fi

  ITER=$((ITER + 1))
  say "=== iteration $ITER/$MAX_ITER ==="
  pi -p "$PROMPT" --session-id "$SESSION_ID" --name autoninfer-driver -a >>"$LOG" 2>&1
  rc=$?
  say "iteration $ITER finished (exit=$rc)"
  [ -f "$STOP_FLAG" ] && break
  sleep 5
done

say "driver done (iterations=$ITER)"
exit 0