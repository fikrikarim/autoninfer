// Private launcher for the fused k_norm + YaRN-RoPE + draft-KV-arena-scatter kernel.
#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstdint>

namespace ninfer::ops::detail {

// One warp per (head, token): normalizes the 128-dim k head with the k_norm gain,
// applies the split-half (NeoX) YaRN rotation from the FP32 inv_freq table with the
// attention factor folded into the cos/sin terms, and scatters K (rotated) and V
// (bit-exact) into the draft-KV arena at the token's committed position.
void launch_dspark_kv_rope_scatter(const Tensor& kv, const Tensor& k_norm_weight, const Tensor& inv_freq,
                                   const Tensor& positions, Tensor& arena, int layer, std::int32_t T,
                                   float attention_scaling, cudaStream_t stream);

} // namespace ninfer::ops::detail