# Autoninfer handover

**Last updated:** 2026-08-18 ~19:57 UTC by unattended driver session.
**This iteration's result: MTP k=4 full-head draft A/B — full head = 129.28 tok/s, 8.0% SLOWER than lm-head 140.36; token-identical (gate 8/8). DISCARD full-head; k=4 stays lm-head (still ratify-pending).**

## HEAD & state
- **HEAD:** `d954ae42` + this iteration's docs commit (results row + BLOCKERS addendum + handover; no code). Tree otherwise clean except the session's DSpark WIP (below).
- **Canonical baseline M1 (k=3, unchanged):** `113.01 ± 0.05 tok/s, 30.15% accept` (82ef8337). Same-session re-measure: k=3 113.93 ± 0.26 (use same-session pairs for A/B deltas).
- **Pending candidate (user ratification OPEN, BLOCKERS 19:55 row):** k=4-lm-head = `140.36 ± 0.53 tok/s, 33.8% accept, 165 rounds` = +23.2% vs same-session k=3. This iteration confirmed **lm-head is the right head at k=4** (full-head loses, see below). k=3 stays canonical until the user says yes on k=4.
- GPU 1 HEALTHY at 19:51 (triad 1466.3 GiB/s, FMA 111.28 TFLOP/s). All GPU work on GPU 1.

## What was done (one experiment: k=4 full-head draft A/B, config-only)
**Why (handover step 2):** the only prior full-head A/B was at k=3 pre-bit-exact and stale (105.16 regression = ~1.5 ms/round of the 248K-row FP8 full-head GEMV at 3 draft steps). At k=4 (4 draft steps, pos2 accept 41.8%) the economics might invert — full head gives better one-step draft quality. Head choice is speed-only (target verify is exact) ⇒ **token-identical requirement**: gate diff vs the k=4-lm-head reference must be zero.
**Method (no build, no collision with session WIP):** reused the committed-source worktree binary `/tmp/wt-it13` @ `18198e78` (same binary that produced the 140.36 reference). M1 Run B = `CUDA_VISIBLE_DEVICES=1 /tmp/wt-it13/build/bench/ninfer_bench --weights models/qwen3_8_27b_nvfp4.ninfer -n 128 -r 3 --warmup 1 --max-ctx 16384 --kv-dtype int8 --mtp-draft-tokens 4` (NO `--lm-head-draft` = full head; bench default head is full, `--lm-head-draft` is opt-in). Gate = worktree copy of `tools/autoninfer/quality_gate.sh` (uses `/tmp/wt-it13/build/apps/ninfer-serve`, not the session-owned main build) with `GATE_SPEC_ARGS="--spec mtp --draft-tokens 4"`.
**Results:**

| config | M1 tok/s (±std) | rounds | accept | accept_len | per-position accept |
|---|---|---|---|---|---|
| k=4 **lm-head** (ref, 140.36) | 140.36 ± 0.53 | 165 | 33.80% | 2.327 | 111/69/24/15 |
| k=4 **full-head** (this run) | **129.28 ± 0.14** | 159 | 36.06% | 2.415 | 111/72/27/15 |

- **Full-head is 8.0% SLOWER** (129.28 vs 140.36) despite HIGHER acceptance (accept_len 2.415 vs 2.327, fewer rounds 159 vs 165). The 248K-row full-head GEMV ×4 draft steps costs more per round than the extra accepts save. **Hypothesis falsified — the k=3 inversion does not happen at k=4; lm-head shortlist stays the speed win.**
- **Token-identity (required check): PASSED.** Gate k=4 full-head per-prompt hashes are **8/8 IDENTICAL** to the k=4 lm-head reference (`post-k4ab`, overall 2f20d7d7): math-balls a53d700d, code-neighbor-sum 354c7843, translate-fr 75882ce5, logic-bulbs c96137af, math-modexp 459c4b77, summarize b834a21f, code-c-bug 34a5e319, haiku 33a65c45. (The gate OVERALL hash differs only because its blob includes the per-prompt `seconds` timing field — the decoded-text hashes, which is the quality signal, match exactly.) Confirms head choice is speed-only.
- **Decision: DISCARD full-head k=4.** k=4-lm-head (140.36) remains the pending candidate. No config change committed. BLOCKERS 19:55 row got a 19:56 addendum (if k=4 ratified → lm-head stays the head; no separate decision).

## /tmp evidence (this iteration)
- `/tmp/it13b_m1_k4.json` (M1 full-head report), `/tmp/it13b_run.log` (bench transcript).
- `/tmp/quality_gate_post-k4-fullhead.jsonl` + `/tmp/it13b_gate.log` + `/tmp/quality_gate_post-k4-fullhead.serve.log` (k=4 full-head gate; diff vs `/tmp/quality_gate_post-k4ab.jsonl` = 8/8 per-prompt identical).
- Reference (previous iteration): `/tmp/it13_m1_k4.json` (140.36), `/tmp/quality_gate_post-k4ab.jsonl` (lm-head k=4 gate).
- `/tmp/wt-it13` = worktree @ 18198e78 with the committed-source binary — **kept** for the next config A/B / k=4 flip; remove with `git worktree remove /tmp/wt-it13 --force` once done.

## Session WIP (live owner — do not touch the files)
- **pid 21662** (pts/6, herdr-daemon-held tty) ALIVE; CPU-idle (cputime frozen 1:17:40 since ~19:22). Last WIP write **18:55:35** (~59 min at check) → **converged (30+ min no writes)**; take-over is permitted per protocol but was NOT exercised this iteration (experiment completed).
- WIP = **DSpark Op 2 (`dspark_ctx_commit`) + BF16 linear support** (the `943713c0` handover's next step, which the session took over): `src/ops/dspark_ctx_commit/{dspark_ctx_commit.cpp, dspark_kv_rope_scatter.cu, dspark_launch.h, dspark_yarn.cpp}`, `include/ninfer/ops/dspark_ctx_commit.h`, `tests/ops/test_dspark_ctx_commit.cpp`, `src/ops/linear/bf16/{bf16_dispatch.cpp, bf16_gemm_mma.cu, bf16_gemv.cu, bf16_small_t.cu}`, `src/CMakeLists.txt`, `tests/CMakeLists.txt`.

## Next iteration (single next step)
1. **If the user ratified the k=4 row** (check BLOCKERS.md active section): flip canonical to k=4-lm-head — wrappers `--draft-tokens 4` (both copies: `tools/autoninfer/supervisor/ninfer-serve.sh` + `/opt/supervisor-scripts/ninfer-serve.sh`), README M1 menu row (`--mtp-draft-tokens 3`→`4`), README baseline section (113.01→140.36), `quality_gate.sh` canonical `SPEC_ARGS` (MTP3→MTP4), fresh M1 + gate re-baseline, fresh `restart-primary` op (the manual serve pid 21537 also needs the new flag). Do NOT flip without an explicit yes.
2. **Otherwise (expected): DSpark Op 2 take-over** (backlog #1, the session's lane). Check `ps -o pid,stat,cputime -p 21662` + WIP file mtimes first. If the session is gone OR the WIP still has 30+ min no writes (converged) → take it over per protocol: build in the **main tree** (`cmake --build build -j`), run `test_dspark_ctx_commit` (GPU 1), strip any TEMP debug code, commit + push. The DSpark lane doc §4 (see `943713c0`) is the Op 2 contract. If the session has resumed writing (mtimes advancing) → do NOT take over; instead run a small non-colliding config A/B (e.g. reuse `/tmp/wt-it13` for another head/window probe) or the k=2/k=3 re-confirm, and re-check DSpark Op 2 next iteration.
   - DSpark lane status: weights complete+verified (2.72 GiB, 47 objects, round-trip bit-exact; `out/dspark_27b.ninfer`). Op 2 = `dspark_ctx_commit` + draft-KV arena.

## Do not repeat / do not touch
- Re-measuring M1 at k=3/4/5 lm-head (valid: 113.93 / 140.36 / 138.33) or the k=4 full-head A/B (this iteration: 129.28, discard).
- The session's WIP file list above while pid 21662 is alive (and `src/CMakeLists.txt` + `tests/CMakeLists.txt`, which are session-dirty).
- The losslessness / rx-anomaly chain (parked: 17:45 BLOCKERS row + 5d4d6298; evidence in /tmp).
- The manual serve (pid 21537) and GPU 0 in any way; the FATAL supervisor `ninfer-serve` — no `supervisorctl` on it from research work. No standby serve. Never commit `.env`, `models/`, `out/`.
- GPU 1: `bash tools/gpu_health.sh 1` before any benchmark session (HEALTHY at 19:51).