# Autoninfer — web research log

Actionable findings from web searches (EXA API: `tools/autoninfer/exa_search.sh`,
`tools/autoninfer/exa_content.sh`). Every entry: source, date added, and what to try in
NInfer. Dedupe before appending. The loop may add ≤2 searches/iteration; the goal is
hypotheses, not browsing.

Seeded 2026-08-18 ~09:15 UTC by the interactive session.

## H1 — MTP acceptance (draft path; current target)

- **P-EAGLE — parallel speculative drafting** (vLLM blog, 2026-03):
  <https://vllm.ai/blog/2026-03-13-p-eagle> — EAGLE-style autoregressive drafting has a hidden
  bottleneck: drafting K tokens costs K serial steps. P-EAGLE parallelizes the draft (parallel
  tree/branch verification). **Idea:** our MTP module drafts 3 tokens autoregressively (3 serial
  draft rounds in the verify round). Check the round-time split (draft vs target verify); if draft
  rounds dominate, a parallel draft (one pass over a small draft tree, or shared-prefix drafting)
  is the structural win. Measure first: `ninfer_qwen3_6_27b_mtp_round_bench` or nsys on a decode
  round.
- **LK Losses — direct acceptance-rate optimization** (arXiv 2602.23881):
  <https://arxiv.org/html/2602.23881v2> — train the draft to maximize acceptance directly.
  **Idea:** our artifact is fixed (no retraining), but the *analysis* is usable: per-position and
  per-token-class acceptance diagnostics to find where the proposal head loses. Complements our
  `accepted_per_position` plan.
- **EAGLE-3** (SafeAILab, NeurIPS'25): <https://github.com/SafeAILab/EAGLE> — feature-level
  context feeding the draft (top-layer features, not just tokens) plus dynamic draft trees.
  **Idea:** does our MTP draft module see a useful context? Inspect the draft head's inputs
  (`tools/convert/qwen3_6_27b/draft_head.py`): if it conditions only on the last hidden state +
  token, feature-level conditioning is an upstream-quality lever (may require artifact change —
  rank accordingly).
- **Gemma 4 MTP architecture notes** (Google, 2026):
  <https://ai.google.dev/gemma/docs/mtp/overview> — production MTP verify/draft interplay,
  including how acceptance interacts with chunked verify. Useful as a second opinion on our
  verify-round design.

## H2 — prefill (1.34× behind the 3.6 counterpart)

- **FP4 fused attention on consumer Blackwell SM120** (F. Mattana, blog+kernel):
  <https://florianmattana.com/posts/fp4-fused-attention-kernel-sm120/> — FP4 attention for
  SM120-class parts; **Idea:** our artifact keeps FP8 on attention projections; check whether the
  prefill attention itself (the GQA prefill kernel) is bandwidth- or compute-bound at chunk 1024
  (nsys/ncu on the prefill phase) before chasing GEMM crossover changes.
- **NVFP4 GEMM tuned configs / SM120 patches**:
  <https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm> (SM120 patches for FlashInfer/CUTLASS
  on 5090-class parts), <https://github.com/vllm-project/vllm/pull/20646> (tuned NVFP4 CUTLASS
  dense-GEMM tile configs), <https://huggingface.co/blog/apsys/blackwell-nvfp4-comparison>
  (3.54× over BF16 via fusion on interactive inference). **Idea:** our FP8 prefill path's GEMM
  tile choices at chunk 1024 may be untuned for 170-SM sm_120a; the vLLM PR's tile sweep is a
  ready-made experiment template (tile configs → measure prefill tok/s at 7,680 tokens).

## H3 — GDN (linear attention) path — model-specific, likely underexplored

- **Atlas-Inference/gdn — hand-tuned Gated DeltaNet kernels for Qwen3.6 hybrid models, SM121**
  (GB10, near-identical arch to sm_120a): <https://huggingface.co/kernels/Atlas-Inference/gdn> —
  decode recurrence, prefill, and **MTP K=2/3 chunkwise verify** kernels. **Idea:** direct
  comparison target for our `gated_delta_net` decode kernel (their FP32-state recurrence shape
  matches ours). If their decode kernel is meaningfully faster per state step, that is a concrete
  kernel port/retune (single op, bounded blast radius, op bench exists).
- **flashrt/gated-delta-attention** (HF Kernel Hub):
  <https://huggingface.co/kernels/flashrt/gated-delta-attention> — "Qwen3.6-style
  linear-attention decode recurrence, prefill WY building blocks, native CUDA FLA-style MMA
  prefill". Same idea, second source.
- **ONNX Runtime decode-optimized GatedDeltaNet kernels** (PR #28985, merged 2026-06):
  <https://github.com/microsoft/onnxruntime/pull/28985> — decode-optimized GDN kernels; the PR
  description states the optimization approach (worth reading with `exa_content.sh`).

## H4 — attention decode kernel (full-attention layers, small batch)

- **Hand-rolled flash decoding on SM120 beating flashinfer** (WingEdge777, 2026):
  <https://www.wingedge777.com/en/article/2a8ffb697f7eb56e> — inline-PTX-tuned decode attention
  for sm_120 that beats flashinfer's single-decode kernel; full kernel + test code linked.
  **Idea:** our C=1 decode runs 16 full-attention layers per token; if nsys shows the GQA decode
  kernel as a top time-sink at batch 1–2, this is the reference implementation to beat/match.
  Caveat: their config (heads/dim) may differ from ours — adapt, don't port blindly.
- **CUTLASS PR #3030 — FA2 for SM120** (open): <https://github.com/NVIDIA/cutlass/pull/3030> —
  NVIDIA's own SM120 attention kernels; watch for merge + configs.

## H5 — KV cache (INT8 already in use; headroom?)

- **lmdeploy INT4/INT8 KV docs**: <https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html>
  — per-head/per-token asymmetric quantization; INT4 KV exists in production stacks.
  **Idea:** our pool is INT8-G64 (33,792 B/token). INT4 KV would halve the pool's B/token →
  same memory, double the KV ceiling (bigger contextWindow) — a capacity win, not a speed win;
  only relevant if long-context becomes the workload. Quality delta must be gated (quality gate
  tool + long-context prompts). Low priority for the speed north star; keep as an option.
- **vLLM vs TRT-LLM KV quantization comparison** (SqueezeBits, 2024-11):
  <https://blog.squeezebits.com/vllm-vs-tensorrtllm-8-kv-cache-quantization-35079> — accuracy
  data per scheme; useful priors if INT4 is ever attempted.

## H6 — DSpark speculator (architectural lane, 2026-08-18)

- **DSpark** (RadixArk, HF model card, read 2026-08-18): <https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark>
  A DSpark speculator for Qwen3.8-27B: "extends DFlash with target-model auxiliary features and a
  confidence head that dynamically chooses the number of draft tokens" (trained with SpecForge,
  served with SGLang `--speculative-algorithm DSPARK`). Facts: 1.36B params BF16 (2.72 GB),
  hidden 5120, **5 full-attention GQA layers** (40 Q / 8 KV heads), target auxiliary feature taps
  at target layers **4, 16, 28, 40, 52**, confidence head (vanilla Markov, rank 256), block size
  **7 draft tokens** (verify width 8), max position 262144. Claimed acceptance length (mean tokens
  accepted per verify step, incl. bonus; FP8 target, temp 0.6 / top-k 20 / top-p 0.95, thinking
  on, 2048 tokens): AIME 2026 **3.07**, AIME 2025 **3.28**, HumanEval 3.47, GSM8K 4.57,
  MT-Bench 3.10; macro mean 3.35 over 11 workloads (1,164 requests). SGLang serve line: TP1, FP8
  target, unquantized draft, block 7.
- **Why it matters (the on-paper case):** MTP k=3 today = 32.3% accept over 3 drafts (≈1.97
  expected tok/round incl. bonus, greedy tg128; ≈49% accept on the sustained AIME stream). DSpark
  claims 3.07–3.28 accepted/step on AIME-class streams — a claimed +55–65% round-level improvement
  IF it holds under NInfer conditions (nvfp4 target, greedy, our corpus). Draft window 7 > MTP's
  5, and the confidence head removes the fixed-k penalty (no more k=2-vs-k=3 re-decision class).
- **The 5090 fit constraint (user target, measured base):** the current nvfp4 config uses
  28.98 GiB at ctx 262144 (KV 8.77 GiB, headroom 1.88 GiB; `kv_fit_probe.sh` anchors: 262144 →
  28.98; 4096 → 20.34). total_used(ctx) ≈ 20.21 + 8.77·(ctx/262144) GiB. DSpark additions:
  draft weights 2.72 GiB (const); draft KV 20 KiB/tok BF16 (5 layers × 8 heads × 128 × 2 × 2B);
  auxiliary tap cache 5 taps × 5120 × 2B = **50 KiB/tok** if materialized over the whole window
  (conservative bound — NInfer discards per-layer hidden states today; SGLang's tap handling is
  the first thing to verify: cached vs recomputed per round). Estimated fit ceiling (≤30.5 GiB
  usable): **≈74K ctx with BF16 taps; ≈98K with INT8 taps; ≈112K with INT8 taps + INT8 draft
  KV.** vs the MTP baseline's 256K. The core trade the experiment must measure: DSpark's
  acceptance win at 64–96K ctx vs MTP's KV ceiling at 256K, at MATCHED context. (FP8 target is
  out of the question on the 5090: ~27 GiB weights leaves no KV headroom — nvfp4 target +
  DSpark is the only viable combination, even though DSpark was trained on the FP8 target's
  features; feature-ε mismatch is a measured risk, not an assumption.)
- **NInfer mapping (implementation sketch):** the 35B-A3B DFlash is the pattern — speculator
  tensors live in an artifact section (`dflash/`), `--spec dflash` selects it, block 1..15, and
  its persistent context already carries target-produced features (model doc §9). DSpark deltas:
  (1) **full-attention layers with persistent draft KV** (35B DFlash uses non-causal masked local
  attention with temporary KV only) — a draft KV arena + draft GQA decode op at verify width 8;
  (2) **5 auxiliary taps** (4/16/28/40/52 × 5120) — target runtime must capture those layers'
  hidden states into a cache arena (bf16 or int8) instead of discarding them;
  (3) **confidence head → dynamic draft length** — the CUDA-Graph decode path captures fixed
  shapes, so either capture one graph per verify width (2..8 = 7 captures) or bucket widths;
  (4) converter `tools/convert/qwen3_8_27b/dspark.py` ingesting the HF safetensors into a
  `dspark/` section of a new 27B artifact variant (we are not tied to the current checkpoint —
  the artifact is ours to extend). MTP stays the fallback/cross-check.
- **Experiment protocol (phase-1 pipeline, 2026-08-18):** `EXPERIMENT_FIT=1` (fit probe ladder,
  expect 65536–98304), `GATE_SPEC_ARGS` for the DSpark serve flags, `EXPERIMENT_M1_ARGS` with the
  dspark flags; M1 compares at the LARGEST FITTING ctx vs the MTP baseline at that same ctx
  (tok/s and accept); gate diff is vs the same configuration's pre-state. Promotion of a new
  spec backend / artifact identity for the 27B target is a product-identity change → BLOCKERS
  ratification after measurement.
- **Open practical question:** huggingface.co is not directly reachable from this instance
  (HTTP 000; github.com is 200) — the 2.72 GB draft download needs a working egress route
  (proxy env, mirror, or box-to-box copy). Check `env | grep -i proxy` first; if none, ask the
  user to drop the repo into `models/` (it is small enough for that).

## Standing method notes

- **Quality gate exists now** (2026-08-18): `tools/autoninfer/quality_gate.sh <label>` — greedy
  decode of 8 fixed prompts on a temporary GPU 1 serve (MTP path enabled), per-prompt token
  hashes; baseline at HEAD `fb49a02`: overall hash `d5d80c0211e7e33b`
  (`/tmp/quality_gate_pre-h1.jsonl` — re-capture if /tmp is cleared).
- **EXA search tooling**: `tools/autoninfer/exa_search.sh "<query>" [n]`,
  `tools/autoninfer/exa_content.sh <url>` (key in git-ignored `.env`).