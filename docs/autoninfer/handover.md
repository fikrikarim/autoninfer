# Autoninfer handover

Rewritten at the end of every iteration; the next iteration's only inherited context.
Keep it complete and concise.

**Updated:** 2026-08-18 by the unattended driver iteration (MTP head A/B experiment).

## State

- HEAD: `59f9e54d` is the commit under test (drive.sh stdin fix); the experiment-record commit
  (README changelog + results row + this file) is its child — confirm with `git log -2`.
  Working tree clean after the record commit.
- Baseline M1 (tg128, `--lm-head-draft`, INT8 KV, MTP3): **111.42 ± 0.05 tok/s, 27.14% accept,
  1.803 tok/round, per-position 105/51/15 of 213 rounds** — reproduced this iteration at
  `59f9e54d` (original baseline 111.82 ± 0.07 @ 27.1%). This is the keep/discard reference.
- Serve: still the loose process on GPU 0 (supervisor `ninfer-serve` STOPPED; `/v1/models` up).
  `{"action":"restart-primary"}` remains queued in `/tmp/autoninfer-ops/pending.json`; the driver
  applies it between iterations. Wrappers (live `/opt/supervisor-scripts/ninfer-serve*.sh` +
  repo copies `tools/autoninfer/supervisor/`) are back at the canonical flags
  (`--spec mtp --draft-tokens 3 --lm-head-draft`, pinned 262144 ctx/KV) — this iteration
  changed them and restored them; they stayed in sync throughout.
- GPU 1: HEALTHY this session (triad 1466.9 GiB/s, FMA 110.71 TFLOP/s).
- `results.tsv`: 2 experiment rows (baseline keep; head A/B discard).

## This iteration — hypothesis #1 (MTP acceptance): head A/B, DISCARD

Per-position acceptance (tg512, 756 rounds, optimized head): **447/237/93** → pos1 59.3%,
pos2 31.4%, pos3 12.3% (conditional 59→53→39%). Gradual decay along the AR chain, no position
collapses to zero → no positional KV/AR bug signature; the one-step draft quality itself is weak.

A/B `--lm-head-draft` (131,072-token shortlist head) vs full head (248,320), same corpus:

| menu | optimized | full | Δ |
|---|---|---|---|
| tg128 (M1) | 111.42 ± 0.05 tok/s, 27.14% accept, 213 rounds | 105.16 ± 0.09 tok/s, 28.78% accept, 207 rounds | **full −5.6% tok/s** |
| tg512 | 125.04 ± 0.25 tok/s, 34.35% accept, 756 rounds, 3 fallbacks | 128.13 ± 0.10 tok/s, 42.64% accept, 675 rounds, 0 fallbacks | full +2.5% tok/s |

Mechanism: the full head's 248,320-row FP8 output GEMV (~1.27 GB read per draft step) is
bandwidth-bound and costs ~1.5 ms/round more than the 131,072-row shortlist head (~672 MB).
tg128's low-predictability random-seed content gives only +2.9% tokens/round (1.855 vs 1.803) —
not enough to amortize; tg512's settled content gives +12% (2.276 vs 2.028) — enough. M1 is the
protocol metric, so the canonical config keeps `--lm-head-draft`. Output is unaffected by either
setting (exact target verification, `include/ninfer/ops/speculative_round.h`) — speed-only A/B,
no parity check needed. Committed: record only (README changelog, results row, this file);
wrappers and menu unchanged.

## Next experiment — MTP layer one-step quality (the binding constraint)

Even the full head reaches only pos1 58.0% (tg128) / 68.0% (tg512) vs the offline
teacher-forced single-step oracle of **0.835** on real text
(`tools/freq_corpus/fixtures/ranking/accept.heldout.manifest.json`; construction in
`tools/convert/qwen3_6_27b/draft_head.py`). No head choice fixes the draft layer's one-step
quality — that is the acceptance lever (each +10 pt pos1 ≈ +10% tok/s at fixed round cost).

Concrete first steps (audit before touching code):

1. Read the MTP layer forward: `src/targets/qwen3_6/impl/runtime/text_context_impl.h`
   (`mtp_forward_core`, `mtp_forward_stem`, `mtp_forward_ar_step`) and `mtp_impl.h`
   (`mtp_decode_batch_body` — note the `alignment_hidden` → `ar_hidden` handoff via
   `speculative_select_accepted_hidden`).
2. Check dtype/precision of each MTP-stage input against the target layer it approximates:
   - hidden-state handoff from the target's final layer (is it the pre-final-norm post-mixer
     output, BF16, selected from the *verified* column — a wrong-column or dropped-norm handoff
     would look exactly like weak one-step quality);
   - MTP `input_projection` + token embedding path (what does this artifact store — check
     `src/targets/qwen3_6_27b/impl/load/bindings.cpp` and `docs/maintainer/qwen3.8-27b-artifact.md`);
   - MTP KV cache dtype: `state.mtp_kv`/`mtp_cache` (INT8 like text KV? check layouts around
     `layouts_impl.h:312/499`);
   - MTP attention: single-layer cached GQA (`ops::gqa_attention_cached`) — envelope/visible
     positions correct for the AR steps?
3. Numeric spot check: for one fixed prefix, compare the MTP layer's one-step logits (full-head
   argmax distribution over a few positions) against the target model's own next-token
   distribution at the same positions (the draft should be a *weaker* but *aligned* predictor —
   rank correlation on top-32 tokens is the quick diagnostic; a large disagreement means a
   handoff/precision bug, a uniformly soft one means quantized draft weights are the ceiling).
4. Only if the path is at natural artifact precision: re-test the head swap on a real-text
   corpus via `tools/bench/run_serve_corpus.py` (long AIME-like streams) to decide the
   product-regime configuration (tg512 suggests full head wins there).

Keep/discard criterion: M1 `tg128` on the fixed menu vs 111.42 baseline; per-position acceptance
is the diagnostic. Do not change sampling semantics (the accept op is exact; keep it that way).

## Do not repeat / watch out

- Do NOT flip the canonical config to full head on tg512 evidence alone — M1 (tg128) is the
  protocol metric and full head regresses it 5.6%. Any future flip needs real-corpus evidence
  and a menu-update commit in the same change.
- If you edit a serve wrapper, keep the live `/opt/supervisor-scripts/` file and its repo copy
  in sync, and queue `{"action":"restart-primary"}`; never restart/touch the GPU 0 serve from an
  iteration (hard rule — driver owns it between iterations).
- Never start or leave the standby serve running.
- tg512 acceptance (34.4% optimized) > tg128 (27.1%): the first ~128 tokens after the one-token
  seed are low-predictability content. Use tg512 for acceptance diagnostics, tg128 (M1) for
  keep/discard.
- Do not compare generic-corpus M1 to the published AIME-stream numbers (corpus effect).
- 35B-A3B and 27B groupwise-int artifacts are not present locally (`models/` has only
  `qwen3_8_27b_nvfp4.ninfer`) — hypotheses needing them are out of reach.
- Bench JSON reports for this iteration: `/tmp/mtp_per_position.json` (tg512 optimized),
  `/tmp/mtp_per_position_fullhead.json` (tg512 full), `/tmp/m1_fullhead_tg128.json`,
  `/tmp/m1_lmhead_tg128_rerun.json`.