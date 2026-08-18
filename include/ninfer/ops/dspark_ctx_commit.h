#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops {

// -----------------------------------------------------------------------------
// DSpark draft speculator (Qwen3.8-27B-DSpark, experimental lane).
//
// Geometry pinned to the published checkpoint (docs/maintainer/qwen3.8-27b-dspark-lane.md):
// 5 full-attention GQA draft layers (40 q / 8 kv heads, head dim 128, YaRN RoPE over the
// 32x-extended context), one fused auxiliary-tap projection from 5 target hidden states
// (layers 4/16/28/40/52 x 5120 = 25600 wide) to the draft hidden (5120), and per-layer
// fused qkv GEMMs (7168 = 6144 q + 2048 kv rows; this Op consumes the kv row slice).
// -----------------------------------------------------------------------------

namespace dspark {
inline constexpr std::int32_t kLayers                 = 5;
inline constexpr std::int32_t kHidden                 = 5120;
inline constexpr std::int32_t kTapWidth               = kLayers * kHidden;            // 25600
inline constexpr std::int32_t kKvHeadCount            = 8;
inline constexpr std::int32_t kHeadDim                = 128;
inline constexpr std::int32_t kHeadDimHalf            = kHeadDim / 2;                 // 64
inline constexpr std::int32_t kKvRows                 = 2 * kKvHeadCount * kHeadDim;  // 2048 (k then v)
inline constexpr std::int32_t kKvBlockOffset          = kKvHeadCount * kHeadDim;      // v rows start at 1024
inline constexpr std::int32_t kArenaElementsPerLayer  = 2 * kKvHeadCount * kHeadDim;  // 2048
inline constexpr std::int32_t kArenaElementsPerToken  = kLayers * kArenaElementsPerLayer; // 10240
} // namespace dspark

/**
 * Op: dspark_ctx_commit (DSpark draft-context commit: project the auxiliary target taps
 * into the persistent draft-KV arena for a contiguous position interval)
 *
 * Math / indexing (t in [0, T), l in [0, dspark::kLayers)):
 *   x[t]    = x2[t] / sqrt(mean_n(x2[t][n]^2) + 1e-6),  x2[t] = W_fc * taps[t]
 *   kv_l[t] = W_kv_l * x[t]                       (kv rows: k [0,1024), v [1024,2048))
 *   khat    = per-head RMSNorm(kv_l[t][h*128 + i]) with gain k_norm[i], eps 1e-6, no offset
 *   theta_i = inv_freq[i] * positions[t]          (i in [0, 64), FP32 angle)
 *   K_l[h][i]      = a * (khat[i]*cos(theta_i) - khat[i+64]*sin(theta_i))    (i <  64)
 *   K_l[h][i + 64] = a * (khat[i+64]*cos(theta_i) + khat[i]*sin(theta_i))    (i <  64)
 *   V_l[h][i]      = kv_l[t][1024 + h*128 + i]                                      (bit-exact passthrough)
 *   with a = attention_scaling (the YaRN attention factor).
 *
 * Logical shapes:
 *   taps BF16 [dspark::kTapWidth, T] (dim0 fastest); positions I32 [T] (device; positions
 *   are valid for all t, the kernel reads them on-device); W_fc BF16_CTRL weight n=5120,
 *   k=25600; W_kv_l BF16_CTRL n=2048, k=5120 (k/v row slice of the checkpoint's fused
 *   per-layer qkv weight [7168, 5120], rows [5120, 7168)); hidden_norm BF16 [5120]
 *   (gain-1 layout value 1.0 in production; the Op applies it as a general gain); k_norm
 *   BF16 [128]; inv_freq FP32 [64] (host-built by dspark_yarn_inverse_frequencies,
 *   device-resident); arena BF16 [dspark::kArenaElementsPerToken, capacity] (dim0 fastest,
 *   2-D so the token count is not bounded by the int32 element extents). K_l for position p
 *   lives at token p, elements l*2048 + h*128 + [0,128); V_l at the same token, offset + 1024.
 *
 * Supported domain:
 *   T >= 1 (MTP verify: T = accepted+1 <= 8; first-build prefill: T = prompt chunk).
 *   position_max_exclusive (host) is the execution envelope: the caller guarantees
 *   0 <= positions[t] < position_max_exclusive for all t and
 *   position_max_exclusive <= arena capacity. Commits are monotone appends: a position is
 *   committed exactly once, so earlier arena regions are never rewritten.
 *
 * Numeric:
 *   The two GEMM stages and the hidden RMSNorm follow the Linear and RMSNorm contracts
 *   (naive FP64 dot-product oracle; named BF16 criteria). The per-head RMSNorm, the
 *   rotation, and the V passthrough are evaluated by a fused kernel: the rotation runs in
 *   FP32 from the FP32 inv_freq table with library-precision cos/sin, the angles carry
 *   the attention factor, and each K output takes one final BF16 rounding; V elements
 *   copy bit-exactly. The FP32 table builder reproduces the transformers 5.12.1
 *   Qwen3RotaryEmbedding pipeline bit-for-bit for the published config.
 *
 * Effects:
 *   Writes K and V for the given positions of all 5 layers; all other arena regions are
 *   untouched. No persistent state beyond the arena contents.
 *
 * Workspace:
 *   Caller-owned, sized by dspark_ctx_commit_workspace_capacity_bytes(); holds the
 *   private x [5120, T], normalized x [5120, T], and per-layer kv [2048, T] buffers.
 *
 * Execution:
 *   Stream-ordered on the caller's stream; safe under CUDA Graph capture when the arena
 *   and all weight/table addresses are capture-stable (positions is device state).
 */

// Minimum caller-owned workspace capacity in bytes for token counts in [min_tokens,
// max_tokens]. The Op suballocates its private GEMM/normalization buffers from the
// passed DeviceArena cursor.
std::size_t dspark_ctx_commit_workspace_capacity_bytes(std::int32_t min_tokens, std::int32_t max_tokens);

// Fused commit (see contract above). Throws std::invalid_argument on any contract
// violation (shape, dtype, qtype, capacity, or envelope violation).
void dspark_ctx_commit(const Tensor& taps, const Tensor& positions, std::int32_t position_max_exclusive,
                       const Weight& fc_weight, const Tensor& hidden_norm_weight,
                       const std::span<const Weight>& kv_weights, const Tensor& k_norm_weight,
                       const Tensor& inv_freq, float attention_scaling, Tensor& arena,
                       WorkspaceArena& workspace, cudaStream_t stream);

// Host YaRN inv_freq table builder. Reference implementation: transformers 5.12.1
// Qwen3RotaryEmbedding._compute_yarn_parameters (bit-exact in FP32 for the published
// DSpark config: theta 1e7, original_max 8192, factor 32, beta_fast 32, beta_slow 1,
// head_dim 128 -> correction boundaries i in [14, 29]).
//
//   pos_freq_i    = theta^(i/(head_dim/2))                     (FP32 pow)
//   low/high      = floor/ceil of the beta_slow/beta_fast correction dims (FP64, 2*pi)
//   ramp_i        = clamp((i - low) / (high - low), 0, 1)      (FP32)
//   out_i         = (1/(factor*pos_freq_i)) * (1 - ext_i) + (1/pos_freq_i) * ext_i,
//                   ext_i = 1 - ramp_i                         (FP32)
//
// out must have head_dim/2 elements. Throws std::invalid_argument on bad parameters.
void dspark_yarn_inverse_frequencies(float rope_theta, std::int32_t original_max_position_embeddings,
                                     float factor, float beta_fast, float beta_slow,
                                     std::int32_t head_dim, std::span<float> out);

// YaRN attention scaling factor: (float)(0.1 * mscale * log(factor) + 1.0), mscale 1.0
// (transformers Qwen3RotaryEmbedding; FP64 arithmetic, single FP32 rounding).
[[nodiscard]] float dspark_yarn_attention_scaling(float factor);

} // namespace ninfer::ops