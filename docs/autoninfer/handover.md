# Autoninfer handover

Rewritten at the end of every iteration; the next iteration's only inherited context.
Keep it complete and concise.

**Updated:** 2026-08-18 ~14:25 UTC by the unattended driver iteration (f58db502 E2E
ratification + M1 re-baseline + k=2/k=3 re-decision).

## State
- **HEAD:** `0ab130b8` (docs-only since `f58db502`, the FP8 committed-column bit-clone fix).
  Engine binary = `f58db502` (verified: `cmake --build build -j` → "no work to do"). This
  iteration's commit (results row + BLOCKERS + this file) is the new HEAD; no engine code
  changed this iteration.
- **Concurrent interactive session (21662, pts/6) is alive and was active during this
  iteration.** It left UNCOMMITTED WIP in the tree: `tools/autoninfer/drive.sh` (+1 line: the
  "Pipeline the build + measurement" prompt bullet), `tools/autoninfer/experiment_run.sh` +
  `experiment_wait.sh` (new, untracked). Do NOT edit or commit its files; if the tree is dirty
  at your start, identify ownership as before (it runs its own M1 probes on GPU 1 — check
  `nvidia-smi` first). Its 13:57 M1 probe completed (118.09 ± 0.14 — cross-confirmed this
  iteration's 118.16) and GPU 1 was clean at 14:20.
- **Baseline M1 (keep/discard reference):** `118.16 ± 0.16 tok/s, 32.3% accept, 195 rounds` at
  k=3 (tg128, menu command) measured this iteration; the interactive session's fresh 13:57 run
  gave 118.09 ± 0.14 / 32.3% — same. This REPLACES 130.13/38.4% (pre-f58db502): **−9.2% is the
  cost of bit-exact verify** — every T=2..4 MTP verify round now runs the T=1 GEMV bit-clone
  schedule (vpl=8, 4 chains, A8) instead of the measured one-chain vpl=16 schedules.
- **Canonical k=0 AIME reference (512 tok, `--seed 42`, ctx 16384, greedy, AIME fixture
  `long_decode_aime26_01`): `0cc41a24fbcd0df8`.** It MOVED from `434280f5` (the 5cbba58c..
  e84ddaef canonical) because f58db502 changed the FP8 small-T T=2..4 schedules, and the
  k=0 prefill processes T≤4 chunks that use them (same benign mechanism as the 5cbba58c move;
  eager ≡ graph verified). Any "k=0 path changed" check must compare against `0cc41a24`, not
  `434280f5`.
- **Serve:** still the loose manual serve pid 21537 on GPU 0 (supervisor `ninfer-serve`
  STOPPED; `/v1/models` 200). **Do not touch it.**
- **Pending op:** `/tmp/autoninfer-ops/pending.json` holds `restart-primary` — the driver
  applies it between iterations (picks up f58db502 + wrapper). **Do not re-queue it.**
- **GPU 1:** HEALTHY at 13:48 (triad 1465.8 GiB/s, FMA 110.73 TFLOP/s). All my temp serves
  torn down (ports 8091–8095). **Serve teardown gotcha: SIGTERM takes >30 s** (two of my
  serves orphaned; one needed SIGKILL) — after `kill <pid>`, poll `nvidia-smi
  --query-compute-apps` before the next GPU work; SIGKILL is fine for temp research serves,
  NEVER for pid 21537.

## This iteration — f58db502 end-to-end ratification (the BLOCKERS follow-up) + re-baseline
No code change. All GPU work on GPU 1 (health-gated), temp serves on 8091–8095, all torn down.

1. **AIME parity (1024 tok, greedy; harness `/tmp/rat_aime.sh`):** the idx-119 flip (280 vs 343)
   is **GONE** — k=0/k=1 agree through char 530 (≈token 233). Residual: each k still has its own
   stream (k=1 "p"→"v" at 530; k=3 vs k=0 at 629; k=1 vs k=3 at 530) — single-token near-tie
   flips at free-choice spots, reconvergent, outputs clearly correct.
2. **Quality gate k=0-vs-k=3 (8 prompts, 256 tok, gate conditions; `/tmp/qgate_diff.sh`):
   5/8 differ** — NOT below the pre-fix 4/8. Same single-token class (a blank line, a period,
   one word, phrasing variants): identical = math-balls, code-neighbor-sum (now exact — differed
   pre-fix), logic-bulbs; differ = translate-fr, math-modexp, summarize, code-c-bug, haiku.
   Interpretation: a near-tie flip is a chaotic function of the (margin, ε) pair — the FP8 fix
   changed ε (removed the committed-column defect), so the flip *set* moved (code-neighbor-sum
   fixed; summarize/code-c-bug newly hit), not a monotone noise cut. The residual class is
   unchanged. Per-prompt hashes (k0 / k3): math-balls 0a0ee172/0a0ee172, code-neighbor-sum
   06ba7f0e/06ba7f0e, translate-fr a40e7866/02f2e91c, logic-bulbs 43b50ca1/43b50ca1,
   math-modexp a0a7c0e6/e1213b87, summarize 455914cb/530db9b0, code-c-bug 72af03d8/38e2c6d1,
   haiku feaac7a8/b4395689.
3. **Root cause of the residual (code-level, certain): the GQA small-T DRAFT-column split
   partition.** `gqa_small_t_split_range(committed_window, full_window, split, ...)`
   (`src/ops/launcher/gqa_attention_decode.cu`) derives the split count/units from
   `committed_window` only (the T=1 policy). Correct for column 0 (draft keys fold into the
   split owning key `committed_window-1` as causally-masked no-ops ⇒ bit-exact, op-verified
   T=1..6). But **draft column j** (visible keys `[0, p0+j)`) gets that p0-derived layout, while
   the T=1 decode at position p0+j would use `gqa_small_t_default_splits(p0+j)` — different
   split count/boundaries ⇒ different per-split partials + split-ordered combine ⇒ NOT a
   bit-clone. Any ε at a draft column can flip a near-tie argmax at the accept decision
   (correct draft rejected → perturbed argmax emitted; or wrong draft accepted) ⇒ single-token
   divergent flips. Everything else in the k>0-vs-k=0 path is verified clean: committed columns
   of attention (op bit-parity, e84ddaef), NVFP4 linears (5cbba58c), FP8 linears (f58db502,
   `ninfer_fp8_linear_t_parity_test` T=1..4 per-column), GDN record+fold (bit-exact W=2..4,
   5cbba58c), KV INT8-G64 writes (absmax ⇒ order-independent ⇒ width-invariant), accept kernel
   (exact by inspection), `kFp8FirstSmallT = 2` (T=1 routes untouched by f58db502).
4. **M1 re-baseline (menu command):** k=3 `118.16 ± 0.16 tok/s, 32.3% accept, 195 rounds`.
   k=2 `117.35 ± 0.27, 42.3% accept, 207 rounds`. The pre-fix +4.2% k=2 advantage is **gone**
   (T=2 and T=4 verify now cost ~equally on the GEMV clone; dropping the weak pos3 draft no
   longer beats its −2.7% tok/round). **Decision: keep k=3 canonical** (within noise at +0.7%).

## Next iteration — make the GQA DRAFT columns a T=1 bit-clone (drive 5/8 → 0/8)
Single hypothesis: the entire residual losslessness gap is the draft-column partition
(root cause #3 above). Fix: per-column T=1 partition — column j uses split count/units derived
from its own window `p0+j` (`gqa_small_t_default_splits(p0+j)`), grid sized for the max over
columns (`f(p0+k)`), columns whose split index exceeds their own active count contribute
nothing. Generalizes e84ddaef (committed column only) to all T≤4 columns; T≥5/prefill
untouched.
Concrete first steps:
1. `bash tools/gpu_health.sh 1`, then check `nvidia-smi` for the interactive session's probes.
2. Pre-gate on the PRE-change binary (quality policy): `tools/autoninfer/quality_gate.sh
   pre-draftcol` (its k=3 stream + the k=0 reference above are the pre-state; this iteration's
   5/8 is the reference diff).
3. Use the pipeline (drive.sh prompt bullet): `setsid bash tools/autoninfer/experiment_run.sh
   draftcol < /dev/null & disown` for build → post-gate → M1, collect with
   `experiment_wait.sh draftcol 900`. (NOTE: those scripts are the interactive session's
   UNCOMMITTED WIP — if they are gone from the tree at your start, fall back to the manual
   sequence: build, `quality_gate.sh post-draftcol`, M1 menu command.)
4. Read the (token, split) work assignment in `src/ops/kernel/gqa_attention_decode.cuh`
   (+ `_bf16`/`_i8` variants) and the launcher: the per-token split-range change may require the
   split loop to become per-token (each token's visible range partitioned by ITS window).
   Implement for T≤4 only.
5. Extend the attention bit-parity op test (e84ddaef's T=1..6 committed-window test) to
   DRAFT columns: at T=2..4, column j's output must bit-equal the T=1 decode at window p0+j
   (per-column parity, INT8+BF16, windows across tile/tier boundaries).
6. E2E acceptance: AIME k=0/k=1/k=3 (1024 tok, greedy) all mutually identical (k=0 =
   `0cc41a24…` first 512); quality gate k0-vs-k3 → expect 0/8 (record the new count if not);
   M1 k=3 ≥ 118 (numerics-only change; schedule unchanged).
If the draft-column fix does NOT reach 0/8, the remaining suspects are the lm_head
(`Fp8Problem::Vocabulary`, A16 route) and GDN outputs at draft columns — both untested at draft
columns; attribute with a per-column target-logit dump (temporary tap) before touching any more
schedules.
Afterwards: backlog #2 (prefill FP8 crossovers; vLLM NVFP4 tile-sweep template in
inspiration.md) — and note the M1 −9.2% verify cost may be recoverable by a FASTER bit-exact
T=2..4 schedule (the T=1 GEMV clone is the current bit-exact schedule, not necessarily the
fastest one; a per-column-parallel schedule replicating the exact T=1 op association per column
is the search space).

## Do not repeat / do not touch
- Do NOT re-derive the ruled-out paths (committed columns of all three linear/attention
  families, GDN fold, KV write, accept kernel) — op-verified clean. The residual is the
  attention draft columns (code-confirmed).
- Do NOT commit the interactive session's WIP (`drive.sh`, `experiment_run.sh`,
  `experiment_wait.sh`) unless its own handover/commit lands first; the live on-disk `drive.sh`
  already carries the pipeline rule, so the next prompt has it regardless of commit state.
- The pre-fix k=2 +4.2% A/B is STALE — post-fix k=2 ≈ k=3; do not flip canonical to k=2.
- Cross-k tok/s A/B is only valid when the streams are identical (losslessness) — until the
  draft-column fix, cross-k tok/s comparisons are confounded (the 4f3bc6d9 discard reason).
- Never restart/touch the GPU 0 serve (pid 21537) — hard rule; do not re-queue
  `restart-primary` (already pending); never start/leave the standby serve; never commit `.env`.
- GPU 1: `bash tools/gpu_health.sh 1` before any benchmark; the interactive session may be
  running probes — check `nvidia-smi --query-compute-apps` first.
- 35B-A3B / 27B groupwise-int artifacts absent locally (`models/` has only
  `qwen3_8_27b_nvfp4.ninfer`).
- Scratch (lost on reboot): `/tmp/rat_aime.sh` (AIME k parity harness), `/tmp/qgate_diff.sh`
  (gate k0-vs-k3 harness), `/tmp/rat_aime_k{0,1,3}.json`, `/tmp/qgate_rows_k{0,3}.json`,
  `/tmp/m1_fresh_head.log` (interactive session's 118.09 run).