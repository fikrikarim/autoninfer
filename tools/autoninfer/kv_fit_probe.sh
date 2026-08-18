#!/bin/bash
# autoninfer KV fit probe: largest --max-ctx a .ninfer artifact fits on the
# research GPU (RTX 5090). This is the first-class constraint for architectural
# experiments: a candidate configuration (different checkpoint, speculator, KV
# dtype) must FIT FULL KV on the 5090 to be a real alternative.
#
# The bench runs with explicit kv capacity == --max-ctx, so a successful
# load+run IS the fit test. Ladder-scan down from the product max (262144);
# an OOM step fails fast right after the weight H2D (~15 s per step).
#
#   tools/autoninfer/kv_fit_probe.sh <weights>
# prints detail lines, then exactly one machine-readable line:
#   FIT ctx=<largest fitting> kv_gib=<x> total_gib=<y>
# or "FIT NONE" on stderr + exit 1 if it does not fit at the ladder floor (16384).
#
# FIT_ARGS overrides the default serving flags (menu: int8 KV, MTP3 + lm-head
# draft). For an alternative speculative configuration pass its own flags, e.g.
#   FIT_ARGS="--kv-dtype int8 --mtp-draft-tokens 7 --lm-head-draft"
set -uo pipefail

WEIGHTS="${1:?usage: kv_fit_probe.sh <weights>}"
REPO=/workspace/autoninfer
cd "$REPO"
FIT_ARGS=${FIT_ARGS:---kv-dtype int8 --mtp-draft-tokens 3 --lm-head-draft}
LADDER=(262144 196608 131072 98304 65536 49152 32768 24576 16384)
TMP=/tmp/autoninfer-fit-probe.json
TMPERR=/tmp/autoninfer-fit-probe.err

[ -f "$WEIGHTS" ] || { echo "weights artifact not found: $WEIGHTS" >&2; exit 1; }
bash tools/gpu_health.sh 1 || { echo "GPU 1 unhealthy - fit probe aborted" >&2; exit 1; }

say() { echo "$(date -Is) $*"; }
memline() { # $1=json -> "kv_gib=.. total_gib=.. headroom_gib=.."
  python3 - "$1" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))["memory"]
G = 2.0 ** 30
# device_total from the engine's own accounting (weights + what was left after
# weights). total_used is everything resident after startup; headroom is the
# free margin. These avoid double-counting (the KV payload lives inside the
# sequence arena, and the runtime reservation overlaps it).
device_total = m["weights"]["capacity_bytes"] + m["available_after_weights_bytes"]
total_used   = device_total - m["available_after_startup_bytes"]
headroom     = m["available_after_startup_bytes"]
kv           = m.get("kv_payload_bytes", 0)
print(f"kv_gib={kv / G:.2f} total_gib={total_used / G:.2f} headroom_gib={headroom / G:.2f}")
PYEOF
}

status=1
for ctx in "${LADDER[@]}"; do
  if CUDA_VISIBLE_DEVICES=1 timeout 300 ./build/bench/ninfer_bench --weights "$WEIGHTS" \
      -n 4 -r 1 --warmup 0 --max-ctx "$ctx" $FIT_ARGS -o json \
      --output-file "$TMP" >/dev/null 2>"$TMPERR"; then
    mem=$(memline "$TMP")
    say "ctx=$ctx FITS ($mem)"
    echo "FIT ctx=$ctx $mem"
    status=0
    break
  fi
  reason=$(tail -1 "$TMPERR" 2>/dev/null | head -c 140)
  if echo "$reason" | grep -qi "memory\|oom\|out of"; then
    say "ctx=$ctx does not fit (OOM): $reason"
  else
    # Non-OOM failure: the probe itself is invalid at this ctx (e.g. below the
    # decode-graph-prime floor) - not a fit verdict. Continue down the ladder.
    say "ctx=$ctx probe invalid (not a fit verdict): $reason"
  fi
done
if [ "$status" -ne 0 ]; then
  echo "FIT NONE (no ladder ctx succeeded; see details above)" >&2
  exit 1
fi
exit 0