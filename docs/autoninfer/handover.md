# Autoninfer handover — 2026-08-22 ~19:45

**HEAD:** `82ef8337` (bench: AIME-512 gate + M1 menu on the ratified GQA
draft-column fix; engine `a36570d3`).
**Committed this iteration:** this handover + one `results.tsv` row
(attribution; **no engine change**).

## What changed this iteration (attribution, GPU 1, ~30 min)

Task = handover's next step: attribute the residual k=1-vs-k=0 greedy flip.
Used the live interactive session's uncommitted `NINFER_TAP_LOGITS` layer/logit
tap (15:17 build of `build/apps/ninfer`; tap is env-gated and numerically
inert — verified eager==graph bit-identical, so it does not perturb streams)
plus a `/tmp/snap_pre` build of `a36570d3^` (`28c3180a`) with the same tap
applied (the WIP tap has a compile error in the 15:23 edit — `DType::F32`
should be `DType::FP32` at `text_context_impl.h:1105`; fixed only in the
/tmp copy, NOT in the tree).

Findings (full text in the results.tsv row):

1. **The k=1 flip is at token idx 119 (280 vs 343)** on the committed source —
   a digit inside a repeated 7-token compute block (`...4478 283 280/343...`),
   deterministic per build, eager == graph. The ratified "char 530 / ~token
   233" is the same flip in text space: my CLI k0 `content` == the ratified
   serve k0 `content` exactly (`/tmp/rat_resp_k0_1024.json`), so CLI and
   serve conditions match and my runs reproduce the ratified streams.
2. **`a36570d3` (GQA draft-column) is E2E-neutral on this fixture**: the
   pre-draftcol snapshot (`28c3180a`, engine-identical to `f58db502` era)
   reproduces the current k0/k1 streams bit-identically (1024 tokens each).
3. **The 0ab130b8 ratification claim "idx-119 flip gone, k0/k1 agree to ~token
   233" is INVALIDATED** — the committed f58db502-era source reproduces the
   119 flip; that E2E was measured on a drifted/intermediate build.
4. **Localization (tap, window 228–360):** at the flip round (pos 346) the
   GDN layer-0 input hidden is bit-identical T=1 vs T=2, but the GDN layer-0
   **recurrent** state (rx) differs; the state **first diverges at pos 230 —
   the first verify round [228,229]** — with conv state (cx) clean there (later
   cx divergence is downstream of rx).
5. **Root cause:** the GDN state-transition kernel
   (`apply_gdn_transition`, `src/ops/linear_attention/gated_delta_net/recurrent.cuh`)
   is SHARED by decode/snapshot/record, so the divergence enters through the
   per-column inputs. `gdn_gating_proj` (BF16, produces `g`/`beta`) routes
   **T=1 → `GemvPairedRows` but T=2..8 → `SmallTSplit10`**
   (`src/ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp`, `k27Routes`) —
   a different kernel/association, NOT a committed-column bit-clone. Last-ulp
   `g`/`beta` difference → `alpha = expf(g)` and `delta = beta*(...)` differ →
   recurrent state diverges from round one. Same defect class as the FP8/NVFP4
   linear family (`5cbba58c`/`f58db502`) and GQA attention (`a36570d3`):
   **gdn_gating_proj is the remaining family with the T=2..8-vs-T=1
   committed-column non-clone.** (`test_gdn_decode_record_parity` passes
   because record+fold exercises the shared transition, not the gating inputs.)
6. M1 re-baselined on the committed-source binary: **113.01 ± 0.05 tok/s,
   30.15% accept, 201 rounds** (matches the ratified 113.2 baseline).
7. Secondary untested suspect (draft-column only): lm_head
   (`Fp8Problem::Vocabulary`) T≥2 still runs the A16 MMA route, not the T=1
   GEMV clone.

## Working-tree note (live session)

`git status` shows only `text_context.h` / `text_context_impl.h` modified —
the **live interactive pi session's** (`21662`) tap WIP. Do NOT commit, revert,
or "fix" those files; they are mid-flight (they add per-column GDN input
hashing — the exact measurement step (a) below — and the 15:23 edit has the
`DType::F32` typo above; the user will hit it on their next build). The
15:17 binary predates the 15:23 edit.

## Must not repeat (ruled out, cumulative)

- `--lm-head-draft` on/off: not the cause (flip in both; M1 −0.9%).
- `--no-cuda-graph`: eager == graph bit-identical this round (tap-verified).
- int8 KV: the flip is present with the canonical int8 serve; (earlier matrix:
  bf16 KV flips earlier — separate near-tie source, not this one).
- GQA attention draft-column (a36570d3): E2E-neutral on this fixture.
- GDN conv history: clean at first divergence (cx SAME at pos 230).
- GDN shared transition kernel: exonerated (single `apply_gdn_transition`).
- Build drift is real: ratified E2E claims must be re-verified against the
  committed source before trusting them (0ab130b8 was invalidated this round).

## Next hypothesis (loop-owned): gdn_gating_proj T=2..8 committed-column clone

Same fix class as 5cbba58c/f58db502/a36570d3. Steps:

a. **Confirm the suspect at op level:** `gdn_gating_proj` per-column output
   (`g`, `beta`) at T=1 vs T=2..4, production 27B shape (x: [5120, T],
   A/B weight rows → 64 heads × (1 g + 1 beta); A_log + dt_bias path). Expect
   the committed column (col 0) to differ at last ulp. (The live session's WIP
   GX tap measures the same per-column values in-engine — if the session lands
   a fix first, skip to (c).)
b. **Fix + parity test:** make T=2..8 committed columns a T=1 `GemvPairedRows`
   bit-clone (or route T=2..4 through the GEMV kernel if the N is small enough
   that the cost is negligible — gating proj is tiny, N≈128 out; check).
   Add `test_gdn_gating_proj_t_parity` (T=1 vs T=2..4 bit-identical per
   committed column + route pin).
c. **E2E verify:** AIME k=0/k=1/k=3 1024-tok greedy (expect k0==k1 — the 119
   flip should be gone; k=3 flip at 761 same class), gate 6/8→8/8 (k=3 hashes
   == k=0 hashes), M1 menu (expect ≥113.2; the gating kernel change may cost a
   hair at T=2..4 — measure, do not assume).
d. If (a) shows g/beta identical and the state still diverges, widen: hash the
   per-column k/v from `gdn_input_proj` (FP8 GdnInput — f58db502's clone; check
   the production verify path actually takes the cloned route) and the
   `active_valid_columns_`/snapshot-slot write pattern in
   `recurrent_snapshot_kernel` (`store_snapshot` per valid token; next round's
   live state = last snapshot).

## Scratch (in /tmp, lost on reboot)

- `tap*.log` — layer/state taps (windows 228–360 and 340–352, + logits),
  k0/k1, current + snapshot (`pre_*`).
- `gr_k{0,1}.err/.out` — 1024-token AIME streams (current tree); `pre_k{0,1}.err`
  — same on the `28c3180a` snapshot.
- `snap_pre/` — throwaway snapshot tree + build (includes the F32→FP32 tap fix).
- `rat_resp_k{0,1,3}_1024.json` — ratified serve responses (k0 content == my CLI k0 content).