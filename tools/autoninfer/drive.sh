#!/bin/bash
# Unattended autoninfer driver.
#
# Runs headless pi iterations (print mode) on the dedicated autoninfer research
# loop. Each iteration is a FRESH pi session (autoninfer-driver-N): the repository
# is the persistent memory (docs/autoninfer/handover.md carries the state across
# iterations), so no session context grows unbounded over a day of iterations.
#
# Between iterations — when no driver pi child is alive — it:
#   - applies pending serve ops from $OPS_DIR (deferred while another interactive
#     pi session is alive: restarting the primary then would kill that session's model);
#   - self-heals the primary serve if it is down (the harness has no model otherwise);
#   - gates on GPU 1 health with 5-minute backoff WITHOUT consuming an iteration
#     (wedges can clear on their own; after 2 h it stops the loop and records a
#     blocker requiring an instance restart).
#
#   start: supervisorctl start autoninfer-driver
#   stop:  supervisorctl stop autoninfer-driver      (or: touch /tmp/autoninfer-stop)
#   ops:   echo '{"action":"restart-primary"}' > /tmp/autoninfer-ops/pending.json
#   log:   /var/log/autoninfer-drive.log
#
# Budget: runs for DURATION_HOURS (wall clock) up to MAX_ITER iterations.
# A clean exit (budget, stop flag, recorded blocker) is not restarted by
# supervisor (autorestart=unexpected + exitcodes=0).
set -uo pipefail

REPO=/workspace/autoninfer
OPS_DIR=/tmp/autoninfer-ops
STOP_FLAG=/tmp/autoninfer-stop
LOG=/var/log/autoninfer-drive.log
BLOCKERS="$REPO/docs/autoninfer/BLOCKERS.md"
SESSION_PREFIX=autoninfer-driver
MAX_ITER=${MAX_ITER:-120}                 # safety cap on iteration count
DURATION_HOURS=${DURATION_HOURS:-30}      # wall-clock budget
GPU1_UNHEALTHY_MAX_MIN=${GPU1_UNHEALTHY_MAX_MIN:-120}
ITER_TIMEOUT_S=${ITER_TIMEOUT_S:-3600}    # hard cap per iteration
START_EPOCH=$(date +%s)
ITER=0
GPU1_DOWN_SINCE=""

. /opt/nvm/nvm.sh >/dev/null 2>&1
mkdir -p "$OPS_DIR"
cd "$REPO"

say() { echo "$(date -Is) $*" | tee -a "$LOG"; }

serve_healthy() { curl -sf -m 2 http://127.0.0.1:8080/v1/models >/dev/null 2>&1; }

# Live (non-zombie) interactive pi processes. The driver's own child is never
# alive when ops run (sequential), so any match is a user session.
other_pi() {
  local p st
  for p in $(pgrep -x pi 2>/dev/null); do
    st=$(ps -o state= -p "$p" 2>/dev/null | tr -d ' ')
    [ -n "$st" ] && [ "${st:0:1}" != "Z" ] && echo "$p"
  done
}

# Bring the supervised primary serve up; return 0 once /v1/models answers.
restore_serve() {
  local svc_state pid
  svc_state=$(supervisorctl status ninfer-serve 2>/dev/null | awk '{print $2}')
  pid=$(supervisorctl pid ninfer-serve 2>/dev/null || echo 0)
  if [ "$svc_state" != "RUNNING" ] && [ "$pid" = "0" ]; then
    # No supervised process: take over from a loose serve process on port 8080
    # (one-shot migration from the manually launched pre-supervisor serve).
    if pgrep -f 'ninfer-serve models/qwen3_8_27b_nvfp4.ninfer .*--port 8080' >/dev/null 2>&1; then
      say "loose serve process on port 8080 - killing it, handing the port to supervisor"
      pkill -f 'ninfer-serve models/qwen3_8_27b_nvfp4.ninfer .*--port 8080' || true
      sleep 3
    fi
    supervisorctl start ninfer-serve >>"$LOG" 2>&1 || true
  else
    supervisorctl restart ninfer-serve >>"$LOG" 2>&1 || true
  fi
  local _
  for _ in $(seq 1 90); do
    if serve_healthy; then return 0; fi
    sleep 2
  done
  return 1
}

# Append a blocker row and push (best effort; the next healthy iteration tidies up).
record_blocker() {
  local what action
  what=$1; action=$2
  printf '\n| %s | %s | %s | loop stopped - needs you |\n' \
    "$(date -Is)" "$what" "$action" >> "$BLOCKERS"
  git add docs/autoninfer/BLOCKERS.md 2>/dev/null
  git commit -q -m "docs(autoninfer): blocker - ${what}" 2>/dev/null || true
  git push origin master 2>/dev/null || say "WARNING: blocker push failed - will retry next session"
  touch "$STOP_FLAG"
}

PROMPT='You are one iteration of the unattended autoninfer research loop (protocol: docs/autoninfer/README.md). You are the only agent running; the repository is the persistent memory.

Start by reading, in this order:
1. docs/autoninfer/handover.md - the previous iteration handed you the state.
2. docs/autoninfer/results.tsv (last 5 rows) and docs/autoninfer/BLOCKERS.md (active section).
3. docs/autoninfer/README.md - protocol, fixed measurement menu, hypothesis backlog.
4. AGENTS.md sections "Autoninfer research environment" and the governance rules.

Hard rules:
- GPU 1 is the research GPU. Run every GPU workload with CUDA_VISIBLE_DEVICES=1. Verify first: bash tools/gpu_health.sh 1. If it is not HEALTHY, do not run GPU experiments; note it in the handover and end the iteration (the driver backs off and retries).
- GPU 0 runs the primary serve that powers you. NEVER bind, kill, pkill, restart, or reconfigure it from this session - not even via supervisorctl. If a serve flag change is required, write {"action":"restart-primary"} to /tmp/autoninfer-ops/pending.json (the driver applies it between iterations after updating the wrapper in /opt/supervisor-scripts/ninfer-serve.sh AND its repo copy tools/autoninfer/supervisor/ninfer-serve.sh) and continue with work that does not need the new flags.
- Never start or leave the standby serve running.
- Do exactly ONE experiment per iteration: pick the highest-value hypothesis from the backlog (or the next step in the handover), implement it, build with `cmake --build build -j`, measure with the protocol menu (M1 at minimum, plus the op-level check the change touches), compare against the recorded baseline, decide keep/discard, and append one row to docs/autoninfer/results.tsv.
- Quality guardrail (north star: faster at EQUAL or better quality): changes that alter token output or sampling semantics must pass the reference/parity checks (tools/parity, docs/op-development) before being measured. For pure performance changes, verify a greedy (temperature 0) decode of a fixed prompt is token-identical before and after. A speedup that changes the output distribution is a discard regardless of the number.
- Commit (Conventional Commit subject, your git identity is configured) and push to origin before ending the iteration.
- End by REWRITING docs/autoninfer/handover.md for the next iteration: current HEAD; what was tried and the measured result; what is committed; the single next hypothesis with concrete first steps (exact commands where possible); anything the next iteration must not repeat. It is the only context the next iteration inherits - make it complete and concise.
- If you are blocked on something only the user can do (instance restart, artifact restore, a product decision): append a row under "## Active blockers" in docs/autoninfer/BLOCKERS.md (date, blocker, needed action, why), commit + push, write the same reason to /tmp/autoninfer-stop, and end the iteration.
- Keep the iteration self-contained: a single M1 bench takes ~2 minutes; if your experiment cannot finish in a reasonable time, narrow its scope, keep whatever result you have, and leave the rest as the next step in the handover. Do not start speculative multi-part changes you cannot finish and measure.'

while true; do
  # 1. Stop flag (also set by record_blocker / agent-side).
  if [ -f "$STOP_FLAG" ]; then
    say "stop flag present - exiting"
    break
  fi

  # 2. Budget.
  elapsed_min=$(( ( $(date +%s) - START_EPOCH ) / 60 ))
  if [ "$elapsed_min" -ge $(( DURATION_HOURS * 60 )) ]; then
    say "wall-clock budget ${DURATION_HOURS}h reached (iterations=$ITER) - exiting cleanly"
    break
  fi
  if [ "$ITER" -ge "$MAX_ITER" ]; then
    say "iteration cap ${MAX_ITER} reached (t+$((elapsed_min / 60))h) - exiting cleanly"
    break
  fi

  # 3. Pending serve ops. Deferred while another pi session is alive: restarting
  # the primary then would kill that session's model mid-turn.
  if [ -f "$OPS_DIR/pending.json" ]; then
    if [ -n "$(other_pi)" ]; then
      say "op pending but pi session(s) running (pids: $(other_pi | tr '\n' ' ')) - deferring until they exit"
    else
      action=$(sed -n 's/.*"action"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$OPS_DIR/pending.json")
      case "$action" in
        restart-primary)
          say "op: restart-primary"
          if restore_serve; then
            mv "$OPS_DIR/pending.json" "$OPS_DIR/restart-primary.done"
            say "op restart-primary applied"
          else
            mv "$OPS_DIR/pending.json" "$OPS_DIR/restart-primary.failed"
            record_blocker "primary serve could not be restored after a requested restart (see /var/log/portal/ninfer-serve.log)" "check the instance; relaunch via `supervisorctl start ninfer-serve` after fixing /var/log/portal/ninfer-serve.log"
            say "op restart-primary FAILED: primary unhealthy after 180s - stopping"
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

  # 4. Emergency serve recovery: the harness needs a model to think.
  if ! serve_healthy; then
    say "primary serve unhealthy - attempting self-heal before the next iteration"
    if restore_serve; then
      say "primary serve recovered"
    else
      record_blocker "primary serve down and self-heal failed (see /var/log/portal/ninfer-serve.log)" "relaunch the serve: `supervisorctl start ninfer-serve` (check the log first)"
      say "primary serve unrecoverable - stopping (no model to iterate on)"
      break
    fi
  fi

  # 5. GPU 1 gate. Back off WITHOUT consuming an iteration; wedges have cleared
  # on their own within the hour on this host. Give up after GPU1_UNHEALTHY_MAX_MIN.
  if ! bash tools/gpu_health.sh 1 >>"$LOG" 2>&1; then
    if [ -z "$GPU1_DOWN_SINCE" ]; then
      GPU1_DOWN_SINCE=$(date +%s)
      say "GPU 1 unhealthy (first seen) - backing off 5 min, no iteration consumed"
    else
      down_min=$(( ( $(date +%s) - GPU1_DOWN_SINCE ) / 60 ))
      if [ "$down_min" -ge "$GPU1_UNHEALTHY_MAX_MIN" ]; then
        record_blocker "GPU 1 unhealthy for ${down_min} min (probe output in /var/log/autoninfer-drive.log; userspace reset unavailable)" "instance RESTART (stop/start or reboot - safe, supervised services and this loop come back on their own)"
        say "GPU 1 unhealthy for ${down_min} min - blocker recorded, stopping"
        break
      fi
      say "GPU 1 unhealthy for ${down_min} min (max ${GPU1_UNHEALTHY_MAX_MIN}) - sleeping 300 s"
    fi
    sleep 300
    continue
  fi
  GPU1_DOWN_SINCE=""

  # 6. One iteration, fresh session, hard timeout. stdin MUST be /dev/null: the
  # supervisor child's stdin is an event pipe that never EOFs, and pi -p reads
  # piped stdin before starting - without this it blocks forever before the first
  # model call (observed 2026-08-18: iteration hung 10+ min with no session file).
  ITER=$((ITER + 1))
  say "=== iteration $ITER/$MAX_ITER (t+$((elapsed_min / 60))h) ==="
  sid="${SESSION_PREFIX}-${ITER}"
  timeout -k 60 "$ITER_TIMEOUT_S" pi -p "$PROMPT" --session-id "$sid" --name "autoninfer $sid" -a < /dev/null >>"$LOG" 2>&1
  rc=$?
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    say "iteration $ITER hit the ${ITER_TIMEOUT_S}s timeout (exit=$rc) - next iteration starts fresh from the handover"
  else
    say "iteration $ITER finished (exit=$rc)"
  fi
  [ -f "$STOP_FLAG" ] && break
  sleep 5
done

say "driver done (iterations=$ITER, wall=$(( ( $(date +%s) - START_EPOCH ) / 60 ))min)"
exit 0