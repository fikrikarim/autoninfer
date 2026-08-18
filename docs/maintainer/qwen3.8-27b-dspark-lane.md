# DSpark draft lane (Qwen3.8-27B, experimental)

Status: **experimental lane, not a registered product identity.** This document is the
engine-side authority for the DSpark speculator on the `qwen3.8-27b/nvfp4` target: the
persistent draft-KV arena, the draft-GQA decode op contract, the block-verify round, and the
VRAM fit model. The draft model mathematics and the section-artifact layout are pinned to the
published checkpoint (below) and the converter (below). MTP remains the canonical speculator;
adoption of DSpark is a product-identity change requiring BLOCKERS ratification after
measurement.

Research context and the acceptance bet: [docs/autoninfer/inspiration.md](../autoninfer/inspiration.md),
H6 (DSpark speculator, architectural lane).

## 1. Checkpoint and section artifact

Source: `RadixArk/Qwen3.8-27B-DSpark` (mirrored at `models/dspark/`: `config.json`,
`dspark.py`, `dflash.py`, `model.safetensors` — 62 BF16 tensors, 1.359B parameters,
2,718,569,474 B data span, verified complete by safetensors header + size).

| Field | Value |
|---|---:|
| architecture | DFlash block-diffusion drafter (SpecForge `DFlashDraftModel`) + Markov head + AcceptRatePredictor confidence head |
| draft hidden layers | 5, all full-attention GQA |
| hidden size / intermediate | 5120 / 10240 |
| Q heads / KV heads / head dim | 40 / 8 / 128 (GQA group 5, no attention bias) |
| norm | RMSNorm, eps 1e-6 (input/post/final + per-head `q_norm`, `k_norm` on dim 128) |
| RoPE | YaRN: factor 32, original max 8192, theta 1e7, beta_fast 32 / beta_slow 1 |
| block size | 7 noise rows (one anchor + six proposals); verify width 8 |
| mask token | 248077 (drafted from the target's embedding; the drafter owns no embedding) |
| auxiliary taps | target decoder-layer outputs at 0-based layers (4, 16, 28, 40, 52), each 5120, concatenated in that order (25600) |
| context projection | `fc`: Linear(25600 -> 5120, no bias), then RMSNorm (`hidden_norm`) |
| Markov head | `w1`: Embedding(248320 -> 256); `w2`: Linear(256 -> 248320, no bias); bias = `w2(w1[prev])` added to draft logits |
| confidence head | one Linear(5120 + 256 -> 1) shared over draft positions (phase 2; fixed width 7 is canonical v1) |
| max position | 262144 |
| owned weights | none of the target's: no embedding, no lm_head — both come from the target |

Section artifact: the NInfer v2 object directory `out/dspark_27b.ninfer`, identity
(model = the 27B family `MODEL_ID`, weights id `dspark-bf16`, recipe
`qwen3_6_27b-dspark-bf16-v1`), produced by `tools/convert/qwen3_6_27b/dspark.py`
(identity BF16 ingestion, round-trip bit-exact by module contract; the file is the
authoritative tensor inventory — 47 objects under `dspark/`, fused
`attention/query_key_value` (7168 x 5120; k rows [5120:6144], v rows [6144:7168]) and fused
`mlp/gate_up` (20480 x 5120; gate rows then up rows), `dspark/feature_projection`
(5120 x 25600), `dspark/markov_w1` (248320 x 256), `dspark/markov_w2` (248320 x 256),
`dspark/confidence_weight` (1 x 5376), `dspark/confidence_bias` (1)). The section carries no
frontend resources and is not yet bound by the engine (binding is this lane's work).

## 2. Block algorithm (reference semantics, greedy)

Reference: `models/dspark/dflash.py` `DFlashDraftModel.spec_generate` +
`models/dspark/dspark.py` (heads). Positions are absolute. Let `start` be the anchor
position (the last committed token) and let the draft block cover positions `start..start+6`.

1. **Noise rows.** The 7 draft rows are the target-embedding of
   `[anchor, MASK, MASK, MASK, MASK, MASK, MASK]` (block rows 0..6, row `j` predicts the
   token at `start+1+j`).
2. **Draft context.** `target_hidden` = the auxiliary taps of the last committed window
   (`acc+1` rows from the previous verify, or the full prefill for the first block),
   projected once: `x = hidden_norm(fc(taps))`. The K/V of `x` are appended to the
   **persistent draft-KV arena** at those absolute positions (Op 2). The arena therefore
   holds, for every committed position `p < start`, the draft K/V derived from the taps at
   `p`. The 7 noise rows' K/V are **transient** (scratch, discarded after the round) —
   the reference's `past_key_values_draft.crop(start)` keeps exactly the committed
   positions and discards the noise rows.
3. **Draft forward** (Op 3), 5 layers, pre-norm residual:
   - attention per layer: `q` from the 7 noise rows (40 heads, `q_norm`, YaRN RoPE at
     positions `start..start+6`); K/V = arena rows `[0..start)` (persistent) + the 7 noise
     rows (`k_norm` + RoPE at the same positions). Each noise row attends to **all** arena
     rows (always strictly earlier — causal holds automatically) and to **all 7 noise rows
     non-causally** (block-diffusion inter-attention). Scale 1/sqrt(128).
   - `o_proj` + residual; post-norm; MLP `down(silu(gate) * up)` + residual.
   - after layer 5: `final_norm` -> 7 x 5120.
4. **Proposals** (Op 4): `base_j = target_lm_head(h_j)` (7 x 248320), then the sequential
   Markov chain: `d_0 = argmax(base_0 + w2*w1[anchor])`,
   `d_j = argmax(base_j + w2*w1[d_{j-1}])` for `j=1..6`. Exact argmax; ties per the
   standard argmax op contract. Proposals stay on device.
5. **Verify + accept.** The target verifies the 8 ids `[anchor, d_0..d_6]` in one
   batched forward (T=8, paged KV). Greedy acceptance: `acc` = the number of leading
   proposals equal to the target's argmax at positions `start+1..start+7`; commit
   `acc` proposals plus the **bonus** = the target's argmax at position `start+acc`.
   New tokens per round = `acc + 1` in [1, 8]. The committed window's taps (the `acc+1`
   rows, including the bonus position) feed the next round's Op 2.
6. **Prefill.** Target prefill forward; taps for all prefill positions; first token =
   target argmax; first block anchor = that token; Op 2 over the prefill taps in
   prefill-chunk order (the transient tap buffer is one chunk, projected and freed per
   chunk — there is no persistent raw-tap arena).

Losslessness note: under greedy sampling the committed stream is a function of the target's
own argmax decisions only (a proposal is committed only when it equals the target argmax;
the bonus always is). DSpark does not add a new numerical class beyond the **batched-verify
near-tie class** already accepted for MTP k>0 (a T=8 verify forward can flip a near-tie
target argmax exactly as T=2..4 does); the k=0 canonical stream (no speculator) is
unaffected. Feature-mismatch risk (the drafter was trained on an FP8 target's auxiliary
features; this target is NVFP4) is measured via acceptance, not assumed.

## 3. Persistent draft-KV arena

- **Content.** For each committed position `p`, per draft layer `l`: `x_p =
  hidden_norm(fc(taps_p))`; `K[l][p] = RoPE_YaRN(k_norm(k_proj_l(x_p)), p)`,
  `V[l][p] = v_proj_l(x_p)`. Computed once at commit (Op 2), never recomputed.
- **Geometry and size.** 5 layers x 8 KV heads x 128 dim x (K,V) x 2 B (BF16) =
  **20,480 B = 20 KiB per token**. Capacity is startup-fixed at `max_context` tokens:
  335 MiB at the menu ctx 16384, 2.50 GiB at 131072.
- **Layout.** Linear, slot-local, one arena per request slot:
  `offset(p, l) = p * 5 * 2 * 1024 * 2 B + l * 2 * 1024 * 2 B` (K block then V block,
  each [head][dim]). Monotonic append (positions commit in increasing order, never
  rewritten) — no crop, no zero-init, no page table.
- **CUDA Graph.** The arena address is capture-stable; the visible key count (`start`)
  is device state read inside the captured round graph, the same mechanism the target
  decode graph uses for the paged KV extent. Fixed T=7 draft shapes mean **one**
  draft-block graph capture per slot (MTP needs one per k).
- **Transient noise scratch.** 7 rows x 20,480 B = 143,360 B per round, outside the
  arena; freed at round end.
- **Prefix reuse.** v1: DSpark requests **do not participate in target prefix reuse** —
  the draft ctx K/V of a reused prefix cannot be rebuilt without that prefix's auxiliary
  taps, and the target does not recompute them on a reuse hit. A DSpark request bypasses
  the prefix cache entirely. (Lever for a later iteration: make the draft arena itself a
  reusable prefix section, adding 20 KiB/token to cached-prefix cost.)
- **INT8 lever (not v1).** Quantizing the draft arena to INT8-G64 halves it (10 KiB/token)
  and raises the fit ceiling (section 5); the published profile is BF16 and v1 is BF16.

## 4. Op contracts

New ops are admitted per [op-development.md](op-development.md). Each floating-point op
below has one independent naive FP32/FP64 oracle evaluated from the represented BF16
public inputs (the oracle does not copy the kernel's staging or reduction order);
integer state transitions (accept/commit, positions) are exact.

### Op 1 — `dspark_tap_capture` (target-side, transient)

At the five designated target decoder layers (0-based 4, 16, 28, 40, 52), store the
post-residual-add layer output `[T, 5120]` (BF16) into the transient tap buffer
`[T, 5, 5120]`, slot order = layer order (4, 16, 28, 40, 52). Active on prefill (per
chunk) and verify (T=8) forwards only; consumed immediately by Op 2. Inert (no store)
when the DSpark backend is not selected. Implementation: a conditional store fused into
the five layers' residual-add tails, or a small dedicated store op; owned by the target
program, not the draft package. Workspace: the one-chunk tap buffer
(4096 x 5 x 5120 x 2 B = 200 MiB at the default prefill chunk; sized by chunk).

### Op 2 — `dspark_ctx_commit` (draft-KV projection + append)

Inputs: taps `[T, 25600]` BF16, absolute positions `[T]` I32, the draft weights
(`feature_projection`, `context_norm`, per-layer QKV slices, `key_norm`, rope table).
Compute, per position: `x = rmsnorm_1e-6(fc(taps))`; per layer `l`:
`K = rope_yarn(rmsnorm_1e6_head128(k_proj_l(x)), pos)`, `V = v_proj_l(x)`; write both
into the arena at the given positions (BF16). `T` <= 8 in the verify round and
`T` = chunk on the prefill first build. Oracle: the exact formula above in FP64 from
the BF16 taps; the YaRN cos/sin table is precomputed once at load with the transformers
5.12.1 YaRN implementation (`rope_parameters` of `models/dspark/config.json`) as the
reference and validated bit-exactly at load. No workspace beyond the GEMM staging the
linear family already owns.

### Op 3 — `dspark_block_decode` (the draft-GQA decode)

Inputs: noise hidden `[7, 5120]` BF16 (target-embedding of the 7 block rows, via the
existing embedding op on the 7 ids), the draft arena (visible rows `[0..start)`), the 7
noise positions, the draft layer weights, the rope table.
Per layer: input norm; dual-source GQA attention — Q: 7 x 40 heads (q_norm, RoPE at the
noise positions); K/V: arena `[0..start)` (8 heads x 128) + the 7 noise rows
(k_proj/v_proj on the noise hidden, k_norm, RoPE); every noise query attends to all
visible keys and to all 7 noise rows without a causal mask among them; scale 1/sqrt(128);
BF16 out; `o_proj`; residual; post norm; `down(silu(gate)*up)`; residual. Output:
`[7, 5120]` after `final_norm`.

Oracle: the ideal-attention pattern of `gqa_attention.h` (A1) — BF16 Q and logical BF16
K/V, FP64 score dot products, stable softmax, FP64 value reduction — over the exact
visible-key set (arena rows + 7 noise rows), compared against the BF16 op output
promoted to FP64. This is a **new op family**, not a re-use of the target `gqa_attention`
A1/A2/A3: linear (non-paged) arena, non-causal 7-row noise block, YaRN table, 40/8 head
geometry, 20 KiB/token arena. The target small-T GQA decode (the family fixed by
`a36570d3`) is the implementation template for the per-column T=1-partition property,
which applies here identically: the T=7 verify columns must be bit-identical to the T=1
route per column (the losslessness campaign's hard-won rule, AGENTS.md "Numerical
correctness").

### Op 4 — `dspark_markov_logits` (proposal extraction)

Inputs: draft hidden `[7, 5120]` BF16, the target lm_head weights (NVFP4; the existing
MTP lm-head-draft route, extended to T=7), `markov_w1`/`markov_w2`, the anchor id, and
the standard argmax/sampling domain. Compute: `base_j` for `j=0..6`; then the sequential
chain `d_0 = argmax(base_0 + w2*w1[anchor])`, `d_j = argmax(base_j + w2*w1[d_{j-1}])`
(on device, no host round-trip; each step is a 256-d gather, a 256 -> 248320 GEMV, and an
argmax over 248320). Output: `drafts [7]` I32 on device — the same consumer the
`speculative_round` machinery takes from MTP, with the draft-count domain extended to 7.
The Markov bias is a natural-precision BF16/FP32 compute; the argmax is exact per the
argmax op contract.

### Op 5 (phase 2) — `dspark_confidence`

`sigmoid(confidence_weight * [h_j; w1[prev_j]] + confidence_bias)` per draft position ->
adaptive block length per the SGLang DSPARK serve rule. Fixed width 7 is canonical v1;
this op (and the width-bucketing it implies for the graph captures) is a later
optimization, not part of the v1 lane.

Acceptance/commit reuses the `speculative_round.h` contract
(`speculative_accept_greedy_drafts`), whose draft-count domain must be extended from
MTP's <= 5 to 7.

## 5. Fit model (RTX 5090, 32 GiB)

Anchors (measured, `tools/autoninfer/kv_fit_probe.sh`): base NVFP4 target
`total(ctx) ~= 20.21 + 8.77 * (ctx/262144)` GiB (262144 -> 28.98 GiB, 1.88 GiB headroom;
usable ceiling 30.5 GiB). DSpark additions:

| Term | Size |
|---|---:|
| draft weights (BF16, section artifact) | 2.53 GiB (constant) |
| draft-KV arena, 20 KiB/token BF16 | 5.00 GiB at 262144 (linear in ctx) |
| transient (tap chunk 200 MiB + noise scratch + draft activations) | ~0.3 GiB |

`total_dspark(ctx) ~= 23.04 + 13.77 * (ctx/262144)` GiB.

| ctx | total | fits? |
|---:|---:|---|
| 16384 (M1 menu) | ~23.9 GiB | yes (matched-context comparison valid) |
| 65536 | ~26.5 GiB | yes |
| 98304 | ~28.1 GiB | yes |
| 131072 | ~29.9 GiB | yes (largest power-of-two fit) |
| 147456 | ~30.8 GiB | over 30.5 GiB — probe to confirm |
| 262144 (MTP ceiling) | ~36.8 GiB | no |

The ladder (65536 / 98304 / 131072 / 147456) is measured with
`EXPERIMENT_FIT=1`, not assumed; the INT8-draft-arena lever (section 3) raises the
ceiling to ~169K if needed. The adoption comparison is at the **largest fitting
context** for both backends at **matched** context (ground rule 9): MTP holds the
262144 ceiling; DSpark must beat MTP at 131072 (or the measured ceiling) to be adopted
for long-context work, and the M1 menu comparison runs at the standard 16384 for both.

## 6. Round schedule and expected cost

One DSpark round (per slot, inside the existing decode-round machinery of
[concurrent-inference-architecture.md](concurrent-inference-architecture.md)):

1. target verify forward, T = 8 (paged INT8 KV) — ~2x the token work of MTP k=3's T=4;
2. greedy acceptance + commit (device);
3. Op 2 on the `acc+1` committed rows (trivial GEMM, <= 8 rows);
4. Op 3 + Op 4 for the next block: draft attention reads `start * 20 KiB` of arena
   (~320 MiB at 16K ctx, ~0.2 ms at measured HBM rate; draft GEMMs ~3.4 GFLOP total,
   negligible), lm_head T=7 (~18 GFLOP NVFP4, comparable to MTP's T=4), Markov chain
   (7 small steps, no host round-trip).

The bet (H6): the claimed DSpark acceptance 3.07-3.28 accepted tokens/step (AIME class,
FP8 target) vs the measured MTP k=3 1.97 tok/round at 113.01 tok/s — a +55-65%
round-level improvement **if** it holds under the NVFP4 target and greedy sampling.
Fixed width 7 removes the k-selection degree of freedom (no k=2-vs-k=3 re-decision
class). The confidence head (Op 5) is the later refinement for low-acceptance regimes.

## 7. Gate and experiment plan

Phase-1 pipeline (`tools/autoninfer/experiment_run.sh`): `EXPERIMENT_FIT=1` (ladder
above), `GATE_SPEC_ARGS` carrying the DSpark serve flags once the CLI binding exists
(`--spec dspark` plus block-size/lm-head flags per the CLI design), `EXPERIMENT_M1_ARGS`
with the DSpark flags; M1 compared at the matched menu ctx 16384 against the MTP
baseline (113.01 +/- 0.05 tok/s, 30.15% accept). Quality gate pre/post per the
2026-08-18 policy: DSpark's committed stream is the accepted batched-verify near-tie
class (section 2) — the gate diff against the MTP pre-state is expected in that class
and is recorded, not a pass/fail by itself; the adoption row in BLOCKERS.md carries the
measurement. Promotion to a registered product identity requires user ratification.

## 8. Do not confuse

- DSpark (this lane, 27B target): **standard causal full attention + persistent
  draft-KV arena** (section 3), YaRN RoPE, Markov + confidence heads, 5 layers.
- 35B-A3B DFlash (registered, [qwen3.6-35b-a3b-model.md](qwen3.6-35b-a3b-model.md) S9):
  non-causal **masked local** attention with temporary KV only, 6 layers, no Markov
  head, different mask/feature config. They share the block-verify skeleton only.
- MTP (registered canonical speculator): autoregressive draft chain, k = 1..5, its own
  section; the acceptance/commit consumer of both is the `speculative_round` machinery.