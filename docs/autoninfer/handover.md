# Autoninfer handover — 2026-07-15 ~16:58 (driver iteration, gdn-gating committed-column clone; state fixup 16:57)

## Outcome of this iteration: DISCARD (engine change not committed)

Experiment: **gdn_gating T=2..8 committed-column bit-clone** (part 2 of the MTP losslessness
series; the k=1 AIME flip fix left by the 15:17 attribution iteration). The T=2..8 small-T route
(Split10) produced verify-round gating columns that differed from the T=1 commit columns by ~1ulp,
the root cause of the k=1 near-tie flip at token 119. Fix (mirrors the A8-clone template already
used by the NVFP4/INT8 families): `k27Routes` T=2..8 -> `GemvPairedRows` (vpl=8, 4 interleaved
chains, zero workspace — bit-identical association to the T=1 route); the T=1 GEMV kernel gained a
runtime `token` param so T=2..8 can launch the same kernel (T=1 numerics bit-identical — verified).

Results (full row in results.tsv):
- **Op level GREEN**: new test `tests/ops/test_gdn_gating_proj_t_parity.cpp` — RED on base
  (T=2 col0 head0 g: 0xbea38a32 vs T=1 ref 0xbea38a33) -> GREEN (T=2..8, all 48 columns x g/beta
  bit-identical to the T=1 route; A8-parent columns identical across all T; route pinned + zero
  workspace asserted).
- **M1: 115.02 ± 0.15 tok/s, 30.65% accept, 201 rounds** (+1.8% vs 113.01/30.15% baseline, same
  operating point; both k=3 menu flags).
- **k=1 AIME**: flip MOVED 119 -> ~151 (still class (a), no reconvergence window checked; since the
  committed column is now bit-exact, the next suspect is T-dependence in argmax/lm_head, not gating).
- **k=0 AIME REGRESSED (the blocker)**: canonical greedy stream (no --spec) diverges at
  **token 137** — base emits `13` (newline), fixed emits `318` ("Let...") and the streams diverge
  persistently. Deterministic per binary (re-run + rebuilt-base both bit-reproduce), present in
  **eager AND graph** (`--no-cuda-graph`), T=1 gating op-level **bit-invariant across builds**
  (test ref hash `5a60bebb10800c96` on both), and codegen determinism verified (rebuilt clean base
  == original base_k0). So the flip is NOT the gemv kernel's T=1 numerics and not graph-specific.
  **Prime suspect: arena/workspace layout** — the fix removes the T=2..8 workspace (30,720 B at
  T=8 -> 0), which changes the load-time workspace composition if any registered range covers
  T=2..8 (e.g. the speculative [1,8]-ish range), moving later device allocations; an
  address-sensitive (stale-read) dependency would then flip a near-tie. Unverified — needs the
  engine's workspace-composition code + a targeted tap.

Decision rule applied: a losslessness-series fix that perturbs the canonical k=0 stream cannot be
kept until the k0 change is established as benign. **Discarded this iteration; nothing committed to
the engine.** All evidence preserved (below) for a quick re-land or re-route next iteration.

## Where the (uncommitted) fix lives

Scratch worktree **`/tmp/wt`** (detached at f2dbd156): modified
`src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu`,
`src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp`,
`tests/CMakeLists.txt` (+1 target), new `tests/ops/test_gdn_gating_proj_t_parity.cpp`
(contains a TEMP `TEMP_T1REF` stderr hash block for cross-build T=1 A/B — useful, remove before
any commit). `/tmp/wt/build/` currently holds the **BASE** binary (last build was the
rebuilt-base A/B check; `git stash list` there is empty — fix is in the working tree, just
rebuild: `cmake --build build -j --target ninfer`). `/tmp/wt/build/.../parity_test` is base-built
too.

## /tmp evidence (this session; also the user session's earlier files)

- `base_k0{,_r2,_r3}.{out,err}` — clean-HEAD k=0 AIME runs (all three bit-identical: determinism + codegen).
- `gdnfix_k0.{out,err}`, `gdnfix_k0_eager.{out,err}` — fixed k=0 (graph + eager; both flip @137).
- `m_new_{a,b,c}.*` — k=1 AIME runs (b/c: this session, flip @151); `m_old_*` base; `gr_k0.*` tap-build k0.
- `lh_{old,new}.err` — user's layer-hash tap runs (decode rounds 150-158: 4-9 diverging layers, class (a)); `opdump_{old,new}.txt`, `matrix.sh`, `ninfer_{old,new,new2}` — user's build artifacts.
- GPU 1 was HEALTHY at 16:01 (triad 1466.6 GiB/s, FMA 110.74 TFLOP/s).

## Next steps (ordered)

1. **Diagnose the k0@137 flip** (top priority; decides re-land vs re-route):
   a. Read the engine workspace composition: which (first,last) ranges does the 27B variant
      register for `gdn_norm_control_projection_workspace_capacity_bytes` when MTP backend is
      **None** (CLI k=0 run)? If a range covering T=2..8 is registered, the arena size differs
      between base and fixed builds (30,720 B) — then hunt for the address-sensitive read
      (uninitialized/stale workspace reads, grid-stride kernels sized by workspace, pool
      pre-allocation). `src/runtime` workspace + `src/targets/qwen3_6_27b/impl/variant.cpp`
      (~line 470) + the family program composition are the places.
   b. Fast path: rebuild the **user's tap** WIP (main tree, `NINFER_DEBUG_LAYERHASH=1` — note its
      15:23 edit had a compile error, `DType::F32` -> use `DType::BF16`) on base vs on
      `/tmp/wt`-fixed, run k=0 AIME both, diff per-layer hashes -> exact first-diverging
      layer/op. (The tap is inert to the stream: tap k0 == base k0 verified this iteration.)
   c. If the flip traces to a legitimate near-tie from an identified benign source (e.g. the
      layout moving an uninitialized-but-never-read region — i.e. NOT a numerical change),
      re-land the fix: copy the 4 files from /tmp/wt into the main tree (after the user session
      commits or the tree is clean), remove the TEMP_T1REF block, commit engine+test, re-verify
      M1 + k0 + k1, and close the row.
2. **Then the stale backlog (pre-fix numbers void)**: k=2 vs k=3 A/B (`--mtp-draft-tokens 2` vs `3`,
   AIME k=0..3 sweep); prefill FP8 crossovers (#2); host-side round overhead (#5).
3. **k=1 flip @151** (post-fix, if re-landed): check argmax/lm_head T-dependence (vocab-scan
   reduction at T=1 vs T=4) — same class-(a) attribution method (layer hashes / op dumps).
4. Baseline bookkeeping to reconcile next session: active comparison baseline **M1 113.01 ± 0.05,
   30.15% accept** (82ef8337 re-baseline, ratified 113.2); the older e84ddaef row says
   **130.13 ± 0.14, 38.4% accept at k=3** — the accept gap (30 vs 38%) implies a different
   operating point (check that row's flags in results.tsv before quoting either).

## Repo / serve state (as of ~16:57)

- **Main tree: NO engine change committed this iteration.** HEAD `912dd5cb` (this iteration's
  docs commit) on f2dbd156. **Tree state changed mid-iteration**: a SECOND interactive session
  (pid 267414, started 15:53; the first, pid 21662, still running since 08:22) was active during
  my experiment and, by 16:55: (a) reverted/abandoned the 8-file FP8-linear WIP + its temp tests
  (gone from status, no commit — user's call, do not resurrect), (b) locally ignored `.env` and
  `.pi/` (now invisible to git status — good hygiene, left alone), (c) left ONLY the 235-line
  layer-hash tap as WIP: `src/targets/qwen3_6/impl/runtime/text_context.h` +
  `text_context_impl.h` (uncommitted; note the tap's 15:23 edit had a compile error,
  `DType::F32` -> use `DType::BF16`). Both sessions may still be working — re-check
  `ps aux | grep " pi$"` + `git status` before touching the main tree; do NOT commit their WIP.
- `.env` never staged by me. Docs-only commits this iteration: 912dd5cb (handover + results row)
  + a follow-up fixup (this state section).
- **GPU 0 / serve: untouched.** Harness serve = manual process (pid 21537 at 15:17, flags
  `--max-context 262144 --kv-capacity 262144 --kv-dtype int8 --max-concurrency 2 --spec mtp
  --draft-tokens 3 --lm-head-draft` — note: NO `--pending-timeout-ms 300000`); supervisor
  `ninfer-serve` STOPPED; `/tmp/autoninfer-ops/pending.json` = `{"action":"restart-primary"}`
  (queued 08:45, still unapplied — the next driver iteration applies it between experiments;
  do NOT re-queue). Expect supervisor serve (wrapper flags incl. the pending-timeout) after that.
- Standby never started. GPU 1 free (idle at 16:57; HEALTHY at 16:01: triad 1466.6 GiB/s).
- OPEN BLOCKERS row (unchanged): user ratification of keeping MTP at k=3 after the losslessness
  series converges + quality gate. The k0@137 anomaly does NOT need the user (research question).

## Rules (restate)

Interactive session wins on any collision (GPU 1, handover.md, tree) — back off. GPU 0 = primary
serve, never bind/kill/restart from research work (restart via pending.json only). All GPU work on
GPU 1 with `CUDA_VISIBLE_DEVICES=1` after `bash tools/gpu_health.sh 1`. One experiment per
iteration; `cmake --build build -j`; M1 + touched op-level checks; log results.tsv; commit (Conventional) +
push; rewrite this file. Quality gate for numerics changes (tools/autoninfer/quality_gate.sh):
k0 canonical must hold.