#!/bin/bash
# autoninfer experiment pipeline: build -> quality gate (post) -> M1 menu bench.
#
# Designed to be launched DETACHED by the agent and polled via
# experiment_wait.sh, so the agent keeps doing LLM-side work while the
# GPU-1/CPU pipeline runs:
#
#   nohup bash tools/autoninfer/experiment_run.sh <name> >/dev/null 2>&1 & disown
#   bash tools/autoninfer/experiment_wait.sh <name> 1800
#
# Ownership: the runner serializes on a lock and owns build/ + GPU 1 while it
# runs. The agent must not edit sources or start other GPU-1 work until the
# marker says done/failed (single-writer repo, one research GPU).
#
# Status files under /tmp/autoninfer-exps/:
#   <name>.running  present while in flight (holds the pid)
#   <name>.status   one word per line: building | gating | measuring | done | failed:<stage>
#   <name>.log      full transcript of every stage
#   <name>.result   machine-readable summary, written only on completion:
#                    status=ok|failed
#                    stage=<last stage reached>
#                    gate=<path to /tmp/quality_gate_post-<name>.jsonl>
#                    m1=<tg128 decode out tok/s>,<acceptance rate> (JSON report)
#
# The M1 menu args can be overridden for A/B experiments (e.g. k=2). The override
# REPLACES the whole default, so always include the full menu line:
#   EXPERIMENT_M1_ARGS="--weights models/qwen3_8_27b_nvfp4.ninfer -n 128 -r 3 --warmup 1 --max-ctx 16384 --kv-dtype int8 --mtp-draft-tokens 2 --lm-head-draft" \
#     setsid bash tools/autoninfer/experiment_run.sh k2_ab < /dev/null & disown
# (change only what the experiment differs in; everything else stays menu-fixed).
set -uo pipefail

NAME="${1:?usage: experiment_run.sh <name> [m1 row label, default tg128]}"
M1_LABEL="${2:-tg128}"
REPO=/workspace/autoninfer
DIR=/tmp/autoninfer-exps
LOG="$DIR/$NAME.log"
STATUS="$DIR/$NAME.status"
RESULT="$DIR/$NAME.result"
RUNNING="$DIR/$NAME.running"
LOCK="$DIR/pipeline.lock"
M1_ARGS=${EXPERIMENT_M1_ARGS:---weights models/qwen3_8_27b_nvfp4.ninfer -n 128 -r 3 --warmup 1 --max-ctx 16384 --kv-dtype int8 --mtp-draft-tokens 3 --lm-head-draft}
M1_TIMEOUT=${EXPERIMENT_M1_TIMEOUT:-1200}
GATE_LABEL="post-$NAME"

mkdir -p "$DIR"
cd "$REPO" || exit 1

# One pipeline at a time (build/ + GPU 1 are single-owner).
exec 9>"$LOCK"
flock 9
rm -f "$RESULT"
echo $$ > "$RUNNING"

say() { echo "$(date -Is) $*" >> "$LOG"; }
stage() { echo "$1" > "$STATUS"; say "=== stage: $1"; }
finish() { # $1=ok|failed $2=stage $3=m1 summary (optional)
  echo "$([ "$1" = ok ] && echo done || echo "failed:$2")" > "$STATUS"
  {
    echo "status=$1"
    echo "stage=$2"
    echo "gate=/tmp/quality_gate_${GATE_LABEL}.jsonl"
    [ -n "${3:-}" ] && echo "m1=$3"
  } > "$RESULT"
  say "=== finished: $1 at stage $2"
  rm -f "$RUNNING"
}
fail() { finish failed "$1" ; exit 1; }

stage building
say "build: cmake --build build -j"
cmake --build build -j >> "$LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  say "build FAILED (rc=$rc)"
  fail build
fi

stage gating
say "quality gate: tools/autoninfer/quality_gate.sh $GATE_LABEL"
bash tools/autoninfer/quality_gate.sh "$GATE_LABEL" >> "$LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  say "quality gate FAILED (rc=$rc)"
  fail gate
fi

stage measuring
M1JSON="$DIR/$NAME.m1.json"
say "M1 ($M1_LABEL): $M1_ARGS"
timeout "$M1_TIMEOUT" env CUDA_VISIBLE_DEVICES=1 ./build/bench/ninfer_bench $M1_ARGS -o json --output-file "$M1JSON" >> "$LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  say "M1 FAILED or timed out (rc=$rc)"
  fail m1
fi

# Machine-readable metric from the JSON report: M1 = decode output tok/s
# (the menu's "decode out t/s"), acceptance from speculative.acceptance_rate.
M1_METRIC=$(python3 - "$M1JSON" "$M1_LABEL" <<'PYEOF'
import json, sys
rep = json.load(open(sys.argv[1]))
for t in rep["tests"]:
    if t["label"] == sys.argv[2]:
        print(f"{t['decode_output_tok_s_mean']:.2f} {t['speculative']['acceptance_rate']:.6f}")
        sys.exit(0)
sys.exit(1)
PYEOF
) || { say "M1 JSON row for $M1_LABEL not found in $M1JSON"; tail -15 "$LOG"; fail m1_extract; }
TOKS=$(echo "$M1_METRIC" | awk '{print $1}')
ACC=$(echo "$M1_METRIC"  | awk '{print $2}')
say "M1: $TOKS tok/s, $ACC accept"
finish ok measuring "$M1_LABEL $TOKS tok/s, $ACC accept"
exit 0