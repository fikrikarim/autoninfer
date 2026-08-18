# Autoninfer handover

Rewritten at the end of every iteration; the next iteration's only inherited context.
Keep it complete and concise.

**Updated:** 2026-08-18 ~15:30 UTC by the unattended driver iteration (a36570d3 E2E
ratification + M1 attribution).

## State
- **HEAD:** `28c3180a` (docs: DSpark architecture verified from model code — interactive
  session). Engine-code HEAD = `a36570d3` (GQA small-T **draft-column** T=1 bit-clone,
  committed by the interactive session at 14:30 — it is exactly the previous handover's
  "next iteration" step, landed mid-iteration). Build == HEAD (`cmake --build build -j`
  → "no work to do"; binary 14:28).
- **Concurrent interactive session (21662, pts/6) alive.** Active today on DSpark
  (28c3180a + the architectural-lane tooling d342a16c/6cdeeaee). Tree was CLEAN at this
  iteration's start (its WIP from 13:3x was converged + committed; its untracked pipeline
  scripts were committed in 6cdeeaee/d342a16c). Check `nvidia-smi --query-compute-apps`
  before GPU work (GPU 1 was clean ~15:10).
- **Baseline M1 (keep/discard reference):** `113.2 ± 0.2 tok/s, 30.15% accept, 201 rounds`
  at k=3 (tg128 menu, 3 samples this iteration: 113.03/113.28/113.43) on the a36570d3
  binary. REPLACES 118.16 ± 0.16 / 32.3% / 195 rounds (pre-draftcol, 0ab130b8).
- **Canonical k=0 AIME reference (512 tok, seed 42, ctx 16384, greedy):
  `0cc41a24fbcd0df8` — RE-VERIFIED EXACT this iteration on the HEAD binary** (k=0 path is
  bit-unchanged by a36570d3; the 13:54 qgate k0 rows remain valid for k0-vs-k3 diffs).
- **MTP losslessness status: STILL OPEN.** Attention (committed + draft columns), all
  linear families, GDN record+fold, KV write, accept kernel are op-verified bit-clean on
  HEAD, but E2E still diverges: AIME-1024 k0=`d49a2892c4ce7690`, k1=`b1e755d756ef7383`
  (first diff char 530 — free-choice variable-name spot "p" vs "v", PERSISTS from
  pre-draftcol — T=2 verify still not a full clone there), k3=`1ac93157f17527bd` (first
  diff char 761, MOVED from 629 — the earlier k=3 flip was fixed by the draft-column fix).
  Gate k0-vs-k3: **6/8 differ** (pre-draftcol 5/8 — math-balls regressed; code-neighbor-sum
  + logic-bulbs exact; all single-token near-tie class, outputs clearly correct).
- **Serve:** loose manual serve pid 21537 on GPU 0 (supervisor `ninfer-serve` STOPPED;
  `/v1/models` up). **Do not touch it.** `/tmp/autoninfer-ops/pending.json` holds
  `restart-primary` — the driver applies it between iterations. **Do not re-queue it.**
- **GPU 1:** HEALTHY ~15:10 (triad 1466.9, fma 110.72). All temp serves torn down.
  SIGTERM teardown gotcha stands: poll `nvidia-smi --query-compute-apps` after `kill`;
  SIGKILL fine for temp research serves, NEVER for pid 21537.

## This iteration — a36570d3 E2E ratification (driver made NO engine code change)
The handover's next step ("make the GQA draft columns a T=1 bit-clone") was implemented and
committed by the interactive session at 14:30 (a36570d3: per-column T=1 partition — each
column j uses `gqa_small_t_column_split_range` from its own window; CTA stages the union of
the columns' split ranges, rows masked per column; reducer per-column active count; grid
unchanged). This iteration measured it (all on GPU 1, temp serves 8093, torn down):
1. **Op-level on HEAD:** `ninfer_gqa_attention_parity_test` (per-column, committed + draft,
   T=2..6, I8+BF16, 10 windows across tile/tier boundaries) **bit-exact**;
   `ninfer_fp8_linear_t_parity_test` + `ninfer_gdn_decode_record_parity_test` PASS.
2. **AIME parity** (`/tmp/rat_aime.sh` rebuilt — original lost to scratch wipe; port 8093):
   k=0 512-tok = `0cc41a24fbcd0df8` exact (canonical held). 1024-tok pairwise: k1 first diff
   530 (unchanged), k3 first diff 761 (was 629).
3. **Quality gate:** k3 post row == interactive session's post-draftcol row 8/8 (same 14:28
   build). k0-vs-k3 diff **6/8** (was 5/8): exact = code-neighbor-sum, logic-bulbs;
   differ = math-balls (regressed), translate-fr, math-modexp, summarize, code-c-bug, haiku.
4. **M1 attribution (−4.2% vs 118.16):** two components — (a) per-round compute **+7.5%**
   (from rounds/tok/s: (201/113.2)/(195/118.16)); mechanism: per-column partitions give
   each column its own T=1 split units, and the CTA stages the UNION of the columns' split
   ranges — in the short-window regime where per-column units differ (e.g. 25/31/32-key
   units), the union has more, smaller intervals and non-participating rows are masked
   no-ops that still cost QK/PV work; the commit's "grid unchanged" holds, per-CTA work
   does not. (b) **accept 32.3% → 30.15%** (201 vs 195 rounds): draft columns are now true
   clones, so acceptance = the model's true draft quality; the pre-fix 32.3% included ~2pp
   of ε-inflated accepts (perturbed argmax accidentally matching drafts).
5. **Decision: keep** — the per-column clone is the model-doc 8 contract (a draft column's
   target argmax drives the accept decision); op-verified bit-exact; M1 delta is a
   correctness cost in the same class as the ratified f58db502 −9.2%. Total verify cost is
   now 130.13 (pre-fp8) → 118.16 (f58db502) → 113.2 (draftcol) = **−13% from peak**.

## Next iteration — attribute the residual (k=1 @ char 530 is the cleanest case)
Single hypothesis: with attention (committed + draft) + NVFP4/FP8 linears (committed) + GDN
record + KV write + accept kernel all op-verified clean, the remaining T=2 verify defect is
in **(a) lm_head (Fp8Problem::Vocabulary, A16 route) at verify widths** or **(b) GDN outputs
at draft columns** — both explicitly untested at draft columns (previous handover's
contingency).
Concrete first steps:
1. `bash tools/gpu_health.sh 1` + `nvidia-smi --query-compute-apps` (interactive session).
2. **Per-column target-logit dump** at the flip position: env-gated temporary tap (pattern:
   the `Tap` template in `src/targets/qwen3_6/impl/runtime/text_context.h`, or the
   NINFER_DEBUG_LAYERHASH approach from the f8b9880b era) dumping `flat_logits` columns
   0..1 of a T=2 verify round at the flip vs a T=1 decode at the same state. Flip spot:
   char 530 of the AIME stream ≈ token ~233 after prefill; AIME prefill = 228 tokens →
   absolute position ≈ 461 (window ≈ 462). Compare exact logit bits + argmax + top-3.
   Strip the tap before commit.
3. If logits bit-equal but the emitted token differs: re-check `ops::argmax` tie-breaking
   and the accept/emission op at T=2 (unlikely — exact by inspection, but the emission
   path `speculative_select_accepted_hidden` is untested at draft columns).
4. If logits differ: per-layer hidden dump (layerhash) at the same position → first
   diverging layer → op-level draft-column parity test of the suspect family (mirror
   `test_gqa_attention_parity`'s per-column pattern).
5. On resolution: re-run the full acceptance suite (`/tmp/rat_aime.sh 0/1/3 1024` + 512
   canonical check; gate k0-vs-k3 → expect 0/8; M1 menu) and record.
**M1 recovery lever (perf, backlog #2):** the −13% verify cost is now the top performance
item. Two prongs: (a) eliminate the union-staging overhead (a schedule whose per-CTA work
equals the per-column T=1 work — e.g. per-column split ranges without cross-column union
no-op rows, or grouping columns with equal units); (b) the "faster bit-exact T=2..4
schedule" search (the T=1 GEMV clone is the current bit-exact schedule, not necessarily the
fastest one that reproduces the exact T=1 op association per column — inspiration.md has
the vLLM NVFP4 tile-sweep template). Any schedule change must re-pass the per-column
parity tests (attention + fp8 linear) bit-exact and hold the k=0 canonical.

## Do not repeat / do not touch
- Do NOT re-derive ruled-out paths: attention committed + draft columns (per-column parity
  bit-exact on HEAD), NVFP4 + FP8 linear committed columns (op-verified), GDN record+fold
  (bit-exact W=2..4), KV INT8-G64 write (absmax ⇒ order-independent), accept kernel (exact
  by inspection).
- Cross-commit / cross-k tok/s A/B is confounded until losslessness holds — ε shifts move
  accept 1–2pp and M1 ±4% with zero compute change (this iteration is the example).
- The pre-fix k=2 +4.2% A/B is STALE; k=3 stays canonical.
- Do NOT commit or edit the interactive session's WIP if it reappears (it owns DSpark:
  `models/dspark/` mirror + the architectural lane; HF egress gap resolved via
  hf-mirror.com). Adopting a DSpark artifact identity needs user ratification via
  BLOCKERS.md (product-identity change).
- Architectural-lane pipeline exists now (committed): `kv_fit_probe.sh` (KV fit ladder on
  GPU 1), `GATE_WEIGHTS`/`GATE_SPEC_ARGS`/`EXPERIMENT_WEIGHTS` overrides, background
  `experiment_run.sh`/`experiment_wait.sh` (build → gate → M1, detached).
- 35B-A3B / 27B groupwise-int artifacts absent locally.
- Never restart/touch the GPU 0 serve (pid 21537); never re-queue `restart-primary`
  (already pending); never start/leave the standby serve; never commit `.env`.
- Scratch (lost on reboot): `/tmp/rat_aime.sh` (rebuilt this iteration, port 8093; usage
  `bash /tmp/rat_aime.sh K MAXTOK`), `/tmp/k0_k3_quality.sh` (port 8092),
  `/tmp/mtp_lossless.sh` (port 8091, STALE — pre-fp8 canonical), `/tmp/rat_resp_k{0,1,3}_1024.json`.