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

## Standing method notes

- **Quality gate exists now** (2026-08-18): `tools/autoninfer/quality_gate.sh <label>` — greedy
  decode of 8 fixed prompts on a temporary GPU 1 serve (MTP path enabled), per-prompt token
  hashes; baseline at HEAD `fb49a02`: overall hash `d5d80c0211e7e33b`
  (`/tmp/quality_gate_pre-h1.jsonl` — re-capture if /tmp is cleared).
- **EXA search tooling**: `tools/autoninfer/exa_search.sh "<query>" [n]`,
  `tools/autoninfer/exa_content.sh <url>` (key in git-ignored `.env`).