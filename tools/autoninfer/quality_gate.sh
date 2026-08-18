#!/bin/bash
# autoninfer quality gate: greedy decode of a fixed prompt set on the current binary.
#
# Usage: tools/autoninfer/quality_gate.sh <label>
#
# Boots a TEMPORARY greedy serve on the research GPU (127.0.0.1:8091, CUDA
# device 1, MTP path enabled so speculative decoding's losslessness is also
# exercised), decodes tools/autoninfer/quality_gate_prompts.json greedily
# (max_tokens 256 per prompt), writes one JSON line per prompt plus an overall
# hash, then always tears the serve down.
#
#   results file: /tmp/quality_gate_<label>.jsonl
#
# Use for a keep/discard decision: run `pre` before the change and `post` after
# (two runs, two labels), then diff the per-prompt token hashes. Zero diff =
# token-identical (required for pure perf changes); small diff = the measured
# quality delta to record in results.tsv.
#
# Fixed gate conditions (do not edit casually; a gate must be stable across
# runs): int8 KV, MTP3 + --lm-head-draft, max_context 8192, C=1, greedy.
# Alternative configurations (architectural experiments) may override the
# artifact and speculative flags via GATE_WEIGHTS / GATE_SPEC_ARGS; the diff
# for such a run is against the SAME configuration's pre-state, and the
# override must be recorded in results.tsv. The canonical MTP3 defaults below
# must not change.
set -euo pipefail

label="${1:?usage: quality_gate.sh <label>}"
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PORT=8091
WEIGHTS="${GATE_WEIGHTS:-models/qwen3_8_27b_nvfp4.ninfer}"
SPEC_ARGS="${GATE_SPEC_ARGS:---spec mtp --draft-tokens 3 --lm-head-draft}"
PROMPTS="$REPO/tools/autoninfer/quality_gate_prompts.json"
OUT="/tmp/quality_gate_${label}.jsonl"
LOG="/tmp/quality_gate_${label}.serve.log"

if [ -f /tmp/.quality_gate_serve.pid ] && kill -0 "$(cat /tmp/.quality_gate_serve.pid)" 2>/dev/null; then
  echo "a previous quality-gate serve is still up (pid $(cat /tmp/.quality_gate_serve.pid)) - refusing to start another" >&2
  exit 1
fi

say() { echo "$(date -Is) $*" ; }

# 1. Research GPU must be healthy.
bash "$REPO/tools/gpu_health.sh" 1 || { echo "GPU 1 unhealthy - quality gate aborted" >&2; exit 1; }

# 2. Boot the temporary greedy serve on GPU 1.
say "booting temporary greedy serve on GPU 1 (port $PORT)..."
CUDA_VISIBLE_DEVICES=1 "$REPO/build/apps/ninfer-serve" \
  "$WEIGHTS" \
  --host 127.0.0.1 --port "$PORT" \
  --max-context 8192 --kv-capacity auto --kv-dtype int8 \
  --max-concurrency 1 $SPEC_ARGS \
  --greedy >"$LOG" 2>&1 &
serve_pid=$!
echo "$serve_pid" > /tmp/.quality_gate_serve.pid
cleanup() {
  if [ -n "${serve_pid:-}" ] && kill -0 "$serve_pid" 2>/dev/null; then
    say "tearing down temporary serve (pid $serve_pid)"
    kill "$serve_pid" 2>/dev/null || true
    wait "$serve_pid" 2>/dev/null || true
  fi
  rm -f /tmp/.quality_gate_serve.pid
}
trap cleanup EXIT

ok=0
for _ in $(seq 1 90); do
  if curl -sf -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then ok=1; break; fi
  if ! kill -0 "$serve_pid" 2>/dev/null; then break; fi
  sleep 2
done
[ "$ok" = 1 ] || { echo "temporary serve failed to start; log: $LOG" >&2; tail -5 "$LOG" >&2; exit 1; }

# 3. Greedy decode every prompt; capture content + reasoning, hash both.
python3 - "$PORT" "$PROMPTS" "$OUT" <<'EOF'
import hashlib, json, sys, time, urllib.request

port, prompts_file, out_file = sys.argv[1], sys.argv[2], sys.argv[3]
prompts = json.load(open(prompts_file))
base = f"http://127.0.0.1:{port}/v1/chat/completions"
lines = []
for p in prompts:
    body = json.dumps({
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": p["text"]}],
        "max_tokens": 256,
    }).encode()
    req = urllib.request.Request(base, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as res:
        d = json.loads(res.read())
    m = d["choices"][0]["message"]
    content = m.get("content") or ""
    reasoning = m.get("reasoning_content") or ""
    blob = f"{reasoning}\n@@\n{content}"
    h = hashlib.sha256(blob.encode()).hexdigest()[:16]
    usage = d.get("usage", {})
    row = {
        "prompt": p["id"],
        "tokens": usage.get("completion_tokens"),
        "reasoning_chars": len(reasoning),
        "content_chars": len(content),
        "hash": h,
        "seconds": round(time.time() - t0, 2),
    }
    lines.append(json.dumps(row))
    print(f"  {p['id']:<22} {row['tokens']} tok, {row['reasoning_chars']}+{row['content_chars']} chars, {h}, {row['seconds']}s")

with open(out_file, "w") as f:
    for line in lines:
        f.write(line + "\n")
blob = "\n".join(lines)
print(f"OVERALL HASH {hashlib.sha256(blob.encode()).hexdigest()[:16]}  -> {out_file}")
EOF

say "done: $OUT (serve torn down)"