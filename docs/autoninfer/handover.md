# Autoninfer handover

Rewritten at the end of every iteration; the next iteration's only inherited context.
Keep it complete and concise.

**Updated:** 2026-08-18 ~09:00 UTC by the interactive session (pre-loop seed).

## State

- HEAD: `6c2796c8` — feat(autoninfer): autonomous serve management and unattended driver.
- Working tree: clean. Git identity configured (Fikri Karim). Pushing works.
- Baseline (row 1, `results.tsv`): M1 `tg128` = **111.82 ± 0.07 tok/s, 27.1% MTP acceptance**
  (generic `bench_corpus.ids`, C=1, max-ctx 16384, INT8 KV, MTP3 + `--lm-head-draft`).
  Generic-corpus point: acceptance is content-dependent (published sustained AIME stream:
  143.8 tok/s, 48.9% accept). Compare M1-to-M1 across commits on this corpus.
- Serve: primary on GPU 0 with the pinned flags (`--max-context 262144 --kv-capacity 262144`);
  a `restart-primary` op is queued in `/tmp/autoninfer-ops/pending.json` and the driver takes
  the serve under supervisor management the first time no interactive pi session is alive.
  Until then the serve is the loose process — treat it exactly the same (never touch it).
- GPU 1: healthy (probe 1,467 GiB/s / 110.7 TFLOP/s; `bash tools/gpu_health.sh 1`).
- Engine fix already in tree: worker-thread `cudaSetDevice` (makes `--device 1` work);
  rebuilt binary in `build/` includes it.

## Next experiment — hypothesis #1: MTP acceptance gap

qwen3.8-27b MTP acceptance (45.8–48.9% published; ~27% generic corpus) trails the other
profiles (67–71%). Acceptance is the main C=1 lever (each accepted draft position is a free
token), so top priority.

Concrete first steps (do step 1 first, it is one command):

1. Per-position acceptance on the bench corpus:
   ```
   CUDA_VISIBLE_DEVICES=1 ./build/bench/ninfer_bench --weights models/qwen3_8_27b_nvfp4.ninfer \
     -n 512 -r 3 --warmup 1 --max-ctx 16384 --kv-dtype int8 --mtp-draft-tokens 3 --lm-head-draft \
     -o json --output-file /tmp/mtp_per_position.json
   ```
   Read `speculative.accepted_per_position` (3 entries for draft window 3). Which positions
   collapse tells you where to look: pos 1 weak → draft head/embedding quality; pos 2–3 weak →
   the sequential draft path (error accumulation / KV handling in the draft rounds).
2. A/B `--lm-head-draft` vs full-head drafting (drop the flag) on the same command — the
   optimized proposal head may be trading accuracy for speed on this mixed FP8/NVFP4 artifact.
3. Draft-quality oracle exists offline: `tools/freq_corpus/fixtures/ranking/accept.heldout.manifest.json`
   (teacher-forced single-step P_accept per stratum, 0.835 heldout). The proposal head's
   construction is in `tools/convert/qwen3_6_27b/draft_head.py`.
4. Only if per-position data points there: audit the draft path numerics (draft KV cache dtype,
   draft attention, proposal head inputs) against the target-model path — the target is
   BF16-precision drafting quality, not a specific arithmetic path.

Keep/discard criterion: M1 `tg128` tok/s on the fixed menu (same corpus/settings); acceptance
is the diagnostic that explains it. Do not change sampling semantics.

## Do not repeat / watch out

- Do not restart or reconfigure the GPU 0 serve from an iteration (hard rule; the driver owns
  it via ops/self-heal between iterations).
- Do not compare the generic-corpus M1 to the published AIME-stream numbers — corpus effect.
- The 35B-A3B and 27B groupwise-int artifacts are not present locally (`models/` has only the
  qwen3.8 nvfp4 artifact) — hypotheses requiring them are out of reach this session.