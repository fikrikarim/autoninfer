# Autoninfer handover

**Last updated:** 2026-08-18 ~21:20 UTC by unattended driver iteration 14.
**Result this iteration: DSpark Op 2 (`dspark_ctx_commit`) LANDED — committed `1abe9a8f`, op test green on GPU 1. Tree clean.**

## HEAD & state
- **HEAD:** `1abe9a8f` (Op 2 landing) + this iteration's docs commit (results row + handover). Tree clean.
- **Canonical baseline M1 (k=3, unchanged):** `113.01 ± 0.05 tok/s, 30.15% accept` (82ef8337). Same-session A/B refs (valid): k=3 113.93 / k=4-lm-head 140.36 / k=5 138.33.
- **Two open user decisions (BLOCKERS, neither ratified yet — k=3 stays canonical, no config change):**
  1. 19:55 row: ratify MTP **k=4-lm-head** as canonical (+23.2% M1, 140.36; near-tie-class gate). If yes → flip serve wrappers (`--draft-tokens 4`, both copies `tools/autoninfer/supervisor/ninfer-serve.sh` + `/opt/supervisor-scripts/ninfer-serve.sh`), README M1 menu + baseline, `quality_gate.sh` canonical SPEC_ARGS (MTP3→MTP4), fresh M1+gate re-baseline, fresh `restart-primary` op (manual serve pid 21537 runs k=3 flags today).
  2. 17:45 row: ratify closing the MTP losslessness series (k>0 = accepted near-tie class).
- GPU 1 HEALTHY at 21:04 (triad 1467.1 GiB/s, FMA 111.25 TFLOP/s). All GPU work this iteration on GPU 1.
- Interactive session **pid 21662**: alive but idle (cputime frozen 1:17:40 since ~19:22, no child processes at 21:00). Its DSpark WIP is now **committed** (below) — nothing of it remains in the tree. Do not kill it; `restart-primary` stays deferred while it lives.
- `/tmp/autoninfer-ops/pending.json` still holds `{"action":"restart-primary"}`. Do NOT re-queue.
- No background jobs (bg.sh ledger empty).

## What was done (one experiment: DSpark Op 2 takeover + landing)
The previous driver iteration (13) had taken over the session's converged DSpark Op 2 WIP (protocol: 30+ min no writes, session idle) and was mid-debug (RoPE scatter mismatches vs oracle) when it hit the 3600s timeout at 20:56, leaving the tree dirty and unverified. This iteration continued and finished the takeover:
1. **Diagnosed where iter 13 died:** its /tmp evidence (`/tmp/it15_{lines,errs}.txt`, `/tmp/it14_*` probes/backups) showed it had (a) fixed the kernel's RoPE lane coverage, (b) re-criterion'd the test K/V vs the FP64 oracle under the kernel's BF16 materialization profile (measured max |err| 0.041 K / 0.031 V, ~2x margin in criteria), (c) added the missing stream sync for legacy-stream H2D input prep — but never re-ran the test.
2. **Rebuilt clean** (`cmake --build build -j`, green) and ran on GPU 1:
   - `ninfer_dspark_ctx_commit_test` **green** — YaRN table bit-exact vs transformers 5.12.1; commit correctness T=1..128 (small-T + MMA linear routes, base/append position layouts), K/V vs FP64 oracle under the documented profile.
   - `ninfer_linear_bf16_a16_test` + `ninfer_linear_add_bf16_a16_test` **green** (the WIP's 2 new BF16 linear shapes: 5120×25600 fc, 2048×5120 kv — also exercised end-to-end by the op test via `ops::linear`).
3. **Stripped** the leftover `#include <cstdio>` in `dspark_kv_rope_scatter.cu` (debug remnant; no other TEMP/debug in the WIP), rebuilt, re-ran the op test (green), **committed `1abe9a8f`** and pushed.

Op 2 = draft-context commit (lane doc §4): taps [25600,T] → fc GEMM → hidden RMSNorm(1e-6) → per-layer kv GEMM → fused per-head k_norm + split-half YaRN rotation (attention factor folded into cos/sin) + draft-KV arena scatter; V scatters bit-exactly, K takes one final BF16 rounding. **Engine-unwired** (no arena/allocation/backend yet) → no M1 impact.

## DSpark lane status (docs/maintainer/qwen3.8-27b-dspark-lane.md is the authority)
- Section artifact: done+verified (`out/dspark_27b.ninfer`, 47 objects, round-trip bit-exact; `tools/convert/qwen3_6_27b/dspark.py` @ 943713c0).
- **Op 2: LANDED (this iteration).** Op 1 (`dspark_tap_capture`, target-side transient store at layers 4/16/28/40/52), Op 3 (`dspark_block_decode`, the 5-layer draft-GQA decode — the big one; note its T=7 verify columns must be per-column T=1 bit-clones per the losslessness rule), Op 4 (`dspark_markov_logits`), the persistent draft-KV arena + round-graph engine wiring: all remaining. Adoption is a product-identity change → BLOCKERS ratification after E2E measurement.

## Next iteration (single next step)
1. **DSpark Op 1 — `dspark_tap_capture`** (chosen: smallest remaining op, unblocks everything downstream; owned by the target program per lane doc §4):
   - Read lane doc §4 Op 1 + `src/targets/qwen3_6/impl/runtime/text_context_impl.h` `run_layers`/`mlp_tail` (the PostMixer tails of target layers 0-based 4/16/28/40/52 are the store points).
   - Implement: conditional store of the post-residual-add layer output `[T,5120]` BF16 into the one-chunk transient tap buffer `[T,5,5120]` (slot order = layer order 4,16,28,40,52), active on prefill (per chunk) and verify (T=8) forwards; inert (no store) when the DSpark backend is not selected. A small dedicated store op under `src/ops` (or fused into the residual-add tail) + exact-store op-level test with the naive oracle.
   - Since no DSpark engine backend/flag exists yet, gate the store on a workspace/sink that is null unless a DSpark tap sink is bound (design the binding so Op 2 can consume it per chunk per the doc: "projected and freed per chunk — no persistent raw-tap arena"). Keep the store a single coalesced copy per layer.
   - Build, run the new op test + `ctest -R "qwen3_6"` (regression: the store must be zero-overhead when inert), commit.
2. Then Op 4 (markov logits — medium, uses the MTP lm-head route extended to T=7) or Op 3 (the big draft-GQA decode) — rank against E2E time-to-measurement at that point.
3. If the user ratified k=4 (check BLOCKERS first): the flip procedure in "state" above takes precedence.

## Do not repeat / do not touch
- Re-running the Op 2 test (green on committed source `1abe9a8f`) or the bf16 linear suites; re-measuring M1 k=3/4/5 lm-head (refs above); the k=4 full-head A/B (129.28, discarded).
- Flipping canonical config (k=4, losslessness closure) without an explicit user yes in BLOCKERS.
- The manual serve (pid 21537) and GPU 0 in any way; no standby serve; `restart-primary` already pending. Never commit `.env`, `models/`, `out/`.
- `stash@{0}` (iter-14 rx-dump tap, orphaned) and `stash@{1}` (267414's layer/logit tap) — don't drop either; the parked losslessness chain (BLOCKERS 17:45) can recover evidence from them if ever re-opened.
- `/tmp/wt-it13` (worktree @ 18198e78, committed-source binaries for config A/B / k=4 flip) — keep until the k=4 decision is settled; `/tmp/wt` (old gdn-fix worktree, discarded fix) — removable any time.
- GPU 1: `bash tools/gpu_health.sh 1` before any benchmark session.