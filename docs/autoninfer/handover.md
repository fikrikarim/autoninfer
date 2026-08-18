# Autoninfer handover

**Last updated:** 2026-08-18 ~20:15 UTC by unattended driver session (new protocol in force).
**This iteration's result: MTP draft-window A/B k=4/k=5 — k=4 = 140.36 tok/s (+23.2% M1). Rule-4 path: measurement kept, BLOCKERS ratify row filed, k=3 stays canonical pending user yes/no.**

## HEAD & state
- **HEAD:** `18198e78` + this iteration's docs commit (results row + BLOCKERS row + handover; no code). Tree otherwise clean except the session's DSpark WIP (below).
- **Baseline M1 (canonical, k=3):** `113.01 ± 0.05 tok/s, 30.15% accept` (82ef8337). This iteration's same-session re-measure: k=3 113.93 ± 0.26 (use same-session pairs for A/B deltas).
- **Pending candidate (user ratification open, BLOCKERS 19:55 row):** k=4 = `140.36 ± 0.53 tok/s, 33.8% accept, 165 rounds` = +23.2% vs same-session k=3; k=5 = 138.33 ± 0.17 / 28.96% = +21.4%.
- **The handover file was 0 bytes at `18198e78`** (the interactive session's amend committed it empty; last substantive version = `943713c0`). This rewrite restores the lost context. If you find it empty again, `git show 943713c0:docs/autoninfer/handover.md` is the recovery source.
- GPU 1 HEALTHY at 19:27 (triad 1466.3 GiB/s, FMA 111.22 TFLOP/s). All GPU work on GPU 1.
- DSpark checkpoint + section artifact: COMPLETE and verified (2.72 GiB, 47 objects, round-trip bit-exact; `out/dspark_27b.ninfer`, report `out/dspark_27b.ninfer.conversion.json`; `out/` + `models/` git-ignored).

## Session WIP (live owner — do not touch)
- **pid 21662** (pts/6, herdr-daemon-held tty) alive; CPU-idle since ~19:22 (cputime frozen 1:17:40); last WIP write 19:22:01.
- WIP = **DSpark Op 2 (`dspark_ctx_commit`) + BF16 linear support** = exactly the `943713c0` handover's next step (DSpark Op 2 + draft-KV arena), which the session took over:
  `src/ops/dspark_ctx_commit/{dspark_ctx_commit.cpp, dspark_kv_rope_scatter.cu, dspark_launch.h, dspark_yarn.cpp}`, `include/ninfer/ops/dspark_ctx_commit.h`, `tests/ops/test_dspark_ctx_commit.cpp`, `src/ops/linear/bf16/{bf16_dispatch.cpp, bf16_gemm_mma.cu, bf16_gemv.cu, bf16_small_t.cu}`, `src/CMakeLists.txt`, `tests/CMakeLists.txt`.
- Backoff ledger for this dirt: this iteration ran a fully non-colliding experiment (no WIP files touched, separate worktree build) — no backoff consumed.
- If the session is gone or the WIP has been without writes 30+ min and looks converged → the driver may take it over per protocol (build, run `test_dspark_ctx_commit`, strip TEMP, commit). Check `ps -o pid,stat,cputime -p 21662` + file mtimes first.

## What was done (one experiment: MTP draft-window A/B k=4/k=5)
**Why:** after the bit-exact-verify campaign, T=2..4 verify rounds run the T=1 GEMV bit-clone route (the expensive part — M1 130.13 → 113.0 was the cost of bit-exact verify), while T≥5 verify runs the fast wide routes. k=4/k=5 (verify widths 5/6) were never measured on this engine. Pure config A/B → zero collision with the session's WIP.
**Method:** `git worktree add /tmp/wt-it13 18198e78` (committed source, no WIP) + full build there (~4.5 min, 192 cores) → M1 menu (tg128, -r 3, `--lm-head-draft`) at k=3/4/5 on GPU 1 → quality gate via the **worktree copy** of `tools/autoninfer/quality_gate.sh` (main `build/` is owned by the session's WIP — the standard `experiment_run.sh` hardcodes the main tree/build and was therefore not usable; deviation recorded here).
**Results (same session, same binary, sequential runs):**

| k | M1 tok/s (±std) | accept | rounds | per-pos accept counts | fallbacks |
|---|---|---|---|---|---|
| 3 | 113.93 ± 0.26 | 30.15% | 201 | 108/48/24 | 3 |
| **4** | **140.36 ± 0.53** | **33.80%** | **165** | 111/69/24/15 | 0 |
| 5 | 138.33 ± 0.17 | 28.96% | 159 | 102/69/27/18/9 | 0 |

**Mechanism:** verify width 5 (k=4) leaves the GEMV-clone route for the fast wide route (round cost down) AND the wide route's column hidden states improve draft quality (pos2 accept 23.9% → 41.8%, pos1 53.7% → 67.3% on tg128; acceptance length 1.896 → 2.327). k=5 adds pos5 (5.7%) but T=6 verify is wider → slightly behind k=4. The T≤4 GEMV-clone hidden states are apparently *worse* for drafting than the wide-route hidden states — the bit-exact campaign optimized for k=0 canonical equality and made the k=3 draft path measurably worse (M1 130.13 pre-campaign at k=3 vs 113.93 post).
**Quality gate (8 fixed prompts, greedy, 256 tok):** k=3 overall hash `ce0b6244632d7783`, k=4 overall hash `2f20d7d7c3636f3a` — **8/8 prompts differ** (per-prompt hashes in `/tmp/quality_gate_{pre,post}-k4ab.jsonl`). Same near-tie class as the ratified k=3 status quo (k=3 vs k=0 = 5/8, ratified at 0ab130b8): token lengths match within ~5 tok (256/256, 224/225, 193/190, 244/256), no loops/cut-offs. **Spot check (haiku + code-c-bug, 256 tok, both k, `/tmp/it13_spot_*`):** reasoning text differs word-level but is semantically identical; code-c-bug: both identify the exact same bug (counts only e/E, not all vowels) and both produce correct corrected C. Clearly correct.
**Decision (quality policy rule 4: diff ≥3/8, speedup >20%):** measurement kept + logged (results row @ `18198e78`, status `discard` = k=4 not adopted without ratification), BLOCKERS ratify row filed 19:55. **k=3 stays canonical; no config change committed.**

## /tmp evidence (this iteration)
- `/tmp/it13_m1_k{3,4,5}.json` (M1 reports), `/tmp/it13_ab.log` (transcript), `/tmp/it13_build.log` (worktree build).
- `/tmp/quality_gate_{pre,post}-k4ab.jsonl` + `/tmp/it13_gate_{pre,post}.log`.
- `/tmp/it13_spot_{haiku,code-c-bug}_k{3,4}.{out,err}` (content spot check).
- `/tmp/wt-it13` = worktree at 18198e78 with current binary (`build/bench/ninfer_bench`, `build/apps/ninfer`, `build/apps/ninfer-serve`) — **kept** for the next config A/B (step 2 below); remove with `git worktree remove /tmp/wt-it13 --force` once done.
- Pre-existing: `/tmp/wt` (discarded gdn-gating chain), `/tmp/it9_*` (losslessness tap evidence), `/tmp/autoninfer-exps/draftcol.*` (k=3 re-baseline 113.34 @ 14:31).

## Serve (do not touch)
- Manual process **pid 21537** (since 08:19; flags: `--max-context 262144 --kv-capacity 262144 --kv-dtype int8 --max-concurrency 2 --spec mtp --draft-tokens 3 --lm-head-draft`, NO `--pending-timeout-ms`) still powers /v1 (verified 200 at 19:23).
- Supervisor `ninfer-serve` is **FATAL** (the applied `restart-primary` op raced the manual serve for 127.0.0.1:8080; marker `/tmp/autoninfer-ops/restart-primary.done`). `pending.json` is consumed. Resolving manual-vs-supervisor is a driver/user concern (943713c0 handover), not research work. **If k=4 gets ratified:** update both wrapper copies (`tools/autoninfer/supervisor/ninfer-serve.sh` + `/opt/supervisor-scripts/ninfer-serve.sh`: `--draft-tokens 4`) and queue a FRESH `restart-primary` op — the manual serve also needs the new flag to be restarted into, so the flip lands with the supervisor serve (see 943713c0 handover's serve note).

## Next iteration (single next step)
1. **If the user ratified the k=4 row** (check BLOCKERS.md active section): flip canonical to k=4 — wrappers `--draft-tokens 4` (both copies), README M1 menu row (`--mtp-draft-tokens 3` → `4`), README baseline section (113.01 → 140.36), `quality_gate.sh` canonical `SPEC_ARGS` (MTP3 → MTP4; the "canonical defaults must not change" rule applies to the *ratified* canonical), fresh M1 + gate re-baseline, fresh `restart-primary` op. Do NOT flip without an explicit yes.
2. **Otherwise: next experiment = k=4 + full-head draft A/B** (config only; reuse the `/tmp/wt-it13` binary — no rebuild):
   `CUDA_VISIBLE_DEVICES=1 /tmp/wt-it13/build/bench/ninfer_bench --weights models/qwen3_8_27b_nvfp4.ninfer -n 128 -r 3 --warmup 1 --max-ctx 16384 --kv-dtype int8 --mtp-draft-tokens 4` (NO `--lm-head-draft`) vs the k=4 lm-head 140.36 reference. Rationale: the only prior full-head A/B was at k=3 pre-bit-exact and stale (105.16 regression = ~1.5 ms/round of the 248K-row FP8 full-head GEMV at 3 draft steps); at k=4 (4 draft steps, pos2 accept 41.8%) the economics may invert — the draft path's binding constraint is one-step draft quality (full head > shortlist head), not just speed. Head choice is speed-only (exact target verify) → **token-IDENTICAL requirement**: gate diff vs this iteration's `post-k4ab` must be zero (run the gate with `GATE_SPEC_ARGS="--spec mtp --draft-tokens 4"`). If it beats 140.36 with gate-zero → BLOCKERS row for full-head+k=4 adoption (same ratify pattern). If it loses → discard, k=4-lm-head stays the pending candidate.
3. DSpark Op 2 (backlog #1) remains the session's lane: if the session is alive → do not touch; if gone/converged → take over per protocol (build in the main tree, run `test_dspark_ctx_commit`, lane doc §4 Op 2 is the contract).

## Do not repeat / do not touch
- Re-measuring M1 at k=3/4/5 (this iteration's same-session numbers are valid: 113.93 / 140.36 / 138.33).
- The session's WIP file list above while pid 21662 is alive; `src/CMakeLists.txt` + `tests/CMakeLists.txt` are session-dirty — don't edit them from a driver iteration.
- The losslessness / rx-anomaly chain (parked: 17:45 BLOCKERS row + 5d4d6298; evidence in /tmp).
- The manual serve (pid 21537) and GPU 0 in any way; the FATAL supervisor `ninfer-serve` — no `supervisorctl` on it from research work. No standby serve. Never commit `.env`, `models/`, `out/`.
- GPU 1: `bash tools/gpu_health.sh 1` before any benchmark session.