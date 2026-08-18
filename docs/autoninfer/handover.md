# Autoninfer handover

**Last updated:** 2026-08-18 22:50 UTC by unattended driver iteration 20 — **RESULT: backlog #2 (prefill lag) audited and closed with a measurement row + a BLOCKERS request; no code change.**

## HEAD & state
- **HEAD:** `902ff970` (backlog #5 audit) + this iteration's docs commit on top. **Tree is DIRTY — owned by the live interactive session (facts below; decide fresh per the concurrent-session rule — a 3rd no-op backoff on this dirt is NOT allowed: act by taking over / stashing / BLOCKERS+stop).**
- **Canonical baseline M1 (k=3):** `113.01 ± 0.05 tok/s, 30.15% accept` (82ef8337). Unchanged this iteration (no code change).
- **Serve state (do not touch):** manual serve pid 21537 (k=3 flags) live on GPU 0, 200 on 8080; supervisor `ninfer-serve` FATAL (port collision — leave it); standby never started; `/tmp/autoninfer-ops/pending.json` empty.
- **GPU 1:** HEALTHY at 22:38 (triad 1467.7 GiB/s, FMA 111.27 TFLOP/s). All GPU work via `/tmp/it20-bin/` copies of the 22:24 WIP-build binaries (valid: the WIP diff is provably additive — 4 new BF16 shape keys only, zero existing-shape re-routing, no fp8/attention/GDN touches — verified by diff inspection).
- **bg.sh ledger:** empty; nothing registered (all work ran foreground).

## The dirty tree — facts (22:42 check)
- **Owner: interactive pi pid 21662** (alive, 14:19:17 etime, CPU 01:17:40 **flat = idle** at both 22:37 and 22:42 checks; ancestor `herdr server` = daemon-held tty). Working the **DSpark Op 3** line (dspark_block_attention + dspark_block_decode + new BF16 draft shapes 7168/5120, 5120/5120, 20480/5120, 5120/10240).
- **`git diff | sha256sum` = `e78b00608a84…074e1c`** (unchanged since iter 19's check). Last source write **22:21:51** (`dspark_block_attention.cu`); last build churn 22:24; no owner /tmp writes since (22:34 `nsys-min.*` are iter 19's audit scratch; 22:40 `it20-*` are this iteration's).
- **30-min convergence mark: 22:51:51** — the next iteration starts PAST it. Per the concurrent-session rule (a): if the fingerprint is still `e78b0060…`, owner still idle/gone → **take over** (see next step). If new writes appeared since 22:21:51 → owner active again → decide fresh (stash if the owner is gone, BLOCKERS+stop only if takeover is genuinely unsafe).
- **Pre-gate reference (reuse, do not re-run):** `/tmp/quality_gate_pre-dspark-op3.jsonl` (21:34, HEAD-equivalent, 8 per-prompt hashes) — valid until the next numerics change lands.

## What was done (backlog #2 — M2 prefill audit, measurement only)
1. **Local M2** (eager, INT8 KV, max-ctx 16384, chunk 1024, GPU 1): **pp2048 9,955.9 ± 43.95 tok/s; pp7680 9,186.1 ± 13.66 tok/s** (= 836 ms). Published qwen3.8 nvfp4 7,680 point = 8,340.4 tok/s @ TTFT 931.6 ms → **serve overhead ≈ 10%** (836 ms engine-level is the real number).
2. **A8 crossovers (the backlog's lever) — DO NOT EXIST:** at the 1024-token prefill chunk the engine already dispatches the A8 tensor-core path, at the FP8 roofline: linear_add N=5120 K=6144 **182.3 µs / 353 TFLOP/s (84.4% peak)**, K=17408 **489.5 µs / 373 (89.0%)**, fused swiglu **845.8 µs / 432 (103%)**; A16 is 10–34× SLOWER at T=1024 (2174/5509/28,846 µs). GEMM thresholds (A8 TC from T≥22/25) are far below prefill chunk size. No dispatch win available.
3. **The 1.34× lag is a weight-format effect, not an engine inefficiency:** qwen3.6-27b and qwen3.8-27b have **identical architectures** (64 layers, 16 full-attn/48 GDN, 24/4/256 heads, hidden 5120 — docs verified). qwen3.6-27b/nvfp4 text weights = `Q4G64/Q5G64/Q6G64` (4/5/6-bit, F16 scales); qwen3.8-27b/nvfp4 text weights = `FP8_E4M3FN_ROW_BF16S` (the converter "preserves source FP8 words" — the released checkpoint is FP8). GEMMs ≈ 70–80% of prefill time (GDN chunked pipeline is bandwidth-bound at 700–1,065 GB/s, ~10–20% of chunk time) → 4/5-bit TC rate + half the weight bytes explains the residual ~1.1× engine gap.
4. **The lever (BLOCKERS 22:45 row):** re-quantize qwen3.8 text weights from BF16 to the per-role Q4/Q5 recipe — a user-authorized checkpoint-content experiment that would also lift M1 (decode GEMMs → W4A4). Needs the **Qwen3.8-27B BF16 source checkpoint (NOT on disk)** + ~60–80 GiB transient disk (26 GiB free) + user ratification. If the FP8 checkpoint is the only source, this lever is a dead end — the BLOCKERS row asks for a yes/no either way.

## New engine bug found (M2 menu blocker — fix candidate)
`ninfer_bench` (and any k=0 / prefill-only Engine config) dies at graph prep: **"CUDA Graph preparation consumed 62914560 bytes, exceeding the planned allowance of 12582912 bytes"** — check at `src/targets/qwen3_6/impl/runtime/program_impl.h:1390`. The graph **allowance plan under-reserves** (12 MiB planned vs 60 MiB consumed), **context-size independent** (fails at max-ctx 2048 AND 16384, both -p 2048 and -p 7680); M1 k=3 menu works (iter 19 ran it), so the plan is only wrong for configs where the ordinary (non-MTP) graph family carries the capture. Workarounds: `--no-cuda-graph` or k≥3. Fix = small, token-identical, gate-verifiable — good next experiment once the tree is clean. (Supersedes the old "k=0 quirk" note.)

## Next iteration (single next step)
**DSpark Op 3 takeover (convergence mark passed 22:51:51):**
1. Check: `ps -o pid,etime,cputime -p 21662` (flat CPU = idle; gone = fine too), `git diff | sha256sum` (expect `e78b0060…`), `stat -c '%y' src/ops/dspark_block_attention/dspark_block_attention.cu` (expect still 22:21:51), `pgrep -a -x ninja cicc` (no build churn), `ls -lat /tmp | head -10`.
2. **Quiet since 22:21:51 + fingerprint matches → TAKE OVER (rule a):** `cmake --build build -j` (main tree; build/ already holds the WIP state), then on GPU 1: `./build/tests/ninfer_dspark_block_attention_test`, `ninfer_dspark_block_decode_test`, the dspark ctx_commit/tap suites, the bf16 linear suites (run the GPU health check first). Verify the per-column T=1-vs-T=7 bit-parity the lane doc requires; strip any TEMP instrumentation (grep `NINFER_DEBUG`, `TODO`, probe code); commit + push as one feat commit; then `tools/autoninfer/quality_gate.sh post-dspark-op3` vs `/tmp/quality_gate_pre-dspark-op3.jsonl` — engine-unwired → **must be 8/8 token-identical** — and the M1 menu (expect ≈ 113.0 baseline); results row; then Op 4 / backend wiring per `docs/maintainer/qwen3.8-27b-dspark-lane.md`.
3. **Owner active again (new writes) → do NOT idle:** if owner pid gone → `git stash push -m "DSpark Op 3 WIP (21662 abandoned)"` and proceed to the **M2 graph-allowance fix** (step above: small token-identical engine fix, M1 + gate pre/post vs `/tmp/quality_gate_pre-dspark-op3.jsonl`, then re-run the M2 menu command without `--no-cuda-graph` to close the loop); only if takeover is genuinely unsafe (owner actively compiling) → BLOCKERS.md row + `/tmp/autoninfer-stop` per the rule.

## Do not repeat / do not touch
- The prefill audit benches (M2 points, A8/A16 crossovers @ T=1024, GDN chunked breakdown) — numbers are in the results row + this handover; CSVs in `/tmp/it20-*.csv`, binaries in `/tmp/it20-bin/` (keep both until the next numerics change).
- nsys/ncu in-container (injection broken with host-injected libcuda — verified iter 19).
- The manual serve (pid 21537) and GPU 0 in any way; no standby serve; no `restart-primary` re-queue (pending.json empty).
- Never commit `.env`, `models/`, `out/`. `git stash @{0}` (rx-dump tap) and `@{1}` (layer/logit tap) — don't drop either. If you stash the Op 3 WIP, it becomes a new index — record it here.
- `/tmp/wt-it13` (worktree @ 18198e78, k=4 A/B binaries) — keep until the k=4 decision is settled; `/tmp/wt`, `/tmp/snap_pre` — removable.
- **Disk:** 26 GiB free; models 23 G + build 19 G — no second full build dir; use /tmp binary copies + incremental builds.
- **Test stream discipline:** host-side staging runs on the legacy default stream — `cuda_synchronize()` before the first op call in new test code.
- GPU 1: `bash tools/gpu_health.sh 1` before any benchmark session.
- Open user decisions (BLOCKERS): 17:45 (close MTP losslessness series), 19:55 (k=4 canonical, +23.2%), 22:45 (BF16 checkpoint + disk + NVFP4 re-quant ratify) — none ratified; k=3 stays canonical until then.