# Autoninfer handover

**Last updated:** 2026-07-09 by an unattended driver session (experiment: MTP losslessness root cause — DONE for the dominant causes, residual localized).

## State
- **HEAD:** `d24f598b` (fix(ops): make the MTP verify range (T<=4) bit-parity with the T=1 decode route), pushed.
- **Baseline M1 (keep/discard reference):** `114.35 ± 0.04 tok/s, 30.2% accept` at k=3 (tg128, `--lm-head-draft`, INT8 KV, menu command) — set this iteration (was 111.79 ± 0.07 / 27.1%); k=2 = `116.33 ± 0.43 tok/s, 39.7% accept`.
- **MTP losslessness status:** 3 of 4 root causes fixed and verified; the remaining one (attention small-t kernels) is precisely localized (below). k=2 ≡ k=3 end-to-end now; k=0/k=1 still diverge (first AIME char diff ~530, unchanged).
- **Reference streams MOVED:** k=0 AIME (greedy, 1024 tok) is now `434280f5` (2336 chars) — the old ref `c38d794e` is stale (prefill's final T=2..4 chunks use the new 4-chain association). k=1=`190da5d9`, k=2=k=3=`1ac93157`. Interactive-session quality-gate baselines must be refreshed against the new k=0 stream.

## This iteration's findings (the losslessness bug)

Contract (model-doc §8, replayssm-gdn §4.5): verify must be a finite-precision **clone** of the k=0 (T=1) path per committed column; a bad draft must not change emitted tokens. The T=2..4 verify widths were **not** clones — four defects found, three fixed:

1. **GDN conv history FP32 leak (fixed).** `GdnConvEpilogue::store` (gdn_conv.cuh) advanced the in-kernel multi-token conv history with the raw FP32 column instead of the BF16-represented value, so the multi-token conv snapshot fed the recurrent core a state the T=1 path can never produce. Fix: round the second-to-last column through BF16 (`s2 = bf16(p)`). T=1 publish path bit-unchanged.
2. **T=4 W4A4 route discontinuity (fixed).** At k=3 verify (T=4), `attn_input` and the GDN fused plan jumped to the W4A4 activation-quantized routes (z/q/k/v off 5–30% per element vs T=1) while T=1..3 use the A16 small-t/GEMV family. Fix: W4A4 boundaries moved T≥4 → T≥5 (`nvfp4_dispatch.cpp`, `nvfp4_attn_input_plan.cpp`, `nvfp4_gdn_snapshot_plan.cpp`). T=4 fused instantiations already existed — pure routing change.
3. **GEMM association mismatch (fixed).** T=1 NVFP4 GEMV accumulates 4 interleaved FP32 chains (pair → chain `(2*pair)&3`, total `((c0+c1)+c2)+c3`, then warp reduce) while the T≥2 small-t kernels used 1 chain. Same lane slices, same coefficient, same code loads — only the association differed, flipping rare last-ULP totals into BF16 diffs. Fix: `kAccumulatorChains = ActiveTokens <= 4 ? 4 : 1` in the small-t schedules (linear, GDN input, residual, attn_input, swiglu). T≥5 prefill arithmetic untouched; T=1 GEMV untouched.
4. **Attention small-t per-column T-dependence (NOT fixed — next).** After 1–3, `tests/ops/test_gdn_decode_record_parity` is **bit-exact** for W=2..4 (the GDN finite-precision clone holds for every verify width), and all linear families are T=1-parity for T≤4. The residual is in the GQA attention small-t kernels: per-TokenTile CTA warp shape (T=4 window≤1029 → 16-warp CTA; T=1..3 → 8-warp; I8 path) and per-T instantiation → different per-column partial/combine association (flash-decoding partials + combine). BF16 KV path has the same T-boundary (2 vs 4 warps). This matches the observed hashes (k=2≡k=3; k=0/k=1 each distinct) if the T=3 and T=4 attention column-0 outputs are bit-equal or differ only below the 1024-token argmax-flip threshold while T=1 differs above it — the parity test in step 1 confirms which.

## Verification evidence
- `tests/ops/test_gdn_decode_record_parity` (new, committed): drives both production chains (one-step T=1 snapshots vs one T=W record + per-commit folds) at NVFP4 27B geometry, W=2/3/4 + tail + A16Only isolation — **0 bit diffs** (was 4/48 layers per column pre-fix).
- ctest: 89/90 on GPU 1; the one failure (`ninfer_qwen3_6_frontend_test`) is a pre-existing environment limitation (needs a local BF16 HF checkpoint not present here), unrelated.
- M1 (menu command): k=3 114.35 ± 0.04 tok/s / 30.2% accept (+2.3% vs 111.79; accept +3.1pp — the verify target argmax is no longer perturbed by arithmetic); k=2 116.33 ± 0.43 / 39.7%.
- Quality (fixed 8-prompt gate set, 8192 ctx, greedy, 256 tok): k=0 vs k=3 exact on 3/8 (5 diverge within 256 tok); AIME: k=2≡k=3, k=0/k=1 first diff ~530. Harnesses in /tmp: `mtp_lossless.sh K`, `k0_k3_quality.sh` (recreate if /tmp was wiped).

## Next iteration: finish losslessness — fix the attention residual
1. **Op-level attention parity test** (mirror the GDN parity test pattern): same paged KV prefix, q widths T=1..4, compare **column 0** outputs bit-for-bit across widths — I8 and BF16, window ≤1029 and >1029 (the warp-shape branches). `tests/ops/test_gqa_attention.cpp` has the paged-cache/envelope setup to reuse.
2. **Make the committed-column arithmetic T-invariant** for T≤4: the per-split key partition, warp-parallel reduction shape, and partial/combine association for column 0 must not depend on TokenTile (the draft columns T>0 only feed acceptance — column 0 is the product contract). Concretely: the T=4 16-warp CTA at window≤1029 (`gqa_attention_decode.cu`, `gqa_small_t_active_splits`/TokenTile dispatch) and the BF16 2-vs-4-warp boundary; the cleanest option is one warp shape + one combine order across T=1..4 (measure the T≥5/prefill-untouched invariant holds — they use separate instantiations).
3. Re-verify: parity test bit-exact; end-to-end `mtp_lossless.sh 0..3` → **all four hashes equal** (acceptance criterion); M1 k=3 vs 114.35; k=0-vs-k=3 gate 8/8.
4. If attention turns out T-invariant in the test (i.e. the test passes before the fix), the residual is elsewhere — re-localize with a model-level probe (logits compare T=1 vs T=2 at fixed prefix; the runtime has a `Tap` template hook in `src/targets/qwen3_6/impl/runtime/text_context.h`, NullTap disabled by default).

## Backlog / notes
- After the attention fix lands, the quality-gate should gain a k=0-vs-k=3 mode (interactive-session-owned tool; the gap is recorded in results.tsv).
- k=2 draft window is now +1.6% vs k=3 (116.33 vs 114.35) — re-decide only after losslessness holds (the old A/B was invalidated by the bug).
- Do not commit `.env`. GPU 0 is reserved (live serve); GPU 1 for all tests/benchmarks (`CUDA_VISIBLE_DEVICES=1`), `bash tools/gpu_health.sh 1` before benchmarking. Serve flags: docs/autoninfer/README.md.