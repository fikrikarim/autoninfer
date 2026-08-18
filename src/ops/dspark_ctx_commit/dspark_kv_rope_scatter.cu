// Fused per-head k_norm RMSNorm + YaRN split-half rotation + draft-KV arena scatter
// (DSpark draft speculator). One warp handles one (head, token): it owns 4 K elements
// (two RoPE pairs: components (lane, lane+64) and (lane+32, lane+96)) and the matching
// 4 V elements. K outputs take one final BF16 rounding; V elements copy bit-exactly.
//
// Tensor convention: dim0 is stored fastest, so the [kKvRows, T] kv output keeps each
// token's k/v rows contiguous (element (r, t) at kv[t * kKvRows + r]).

#include "ops/dspark_ctx_commit/dspark_launch.h"
#include "core/device.h"
#include "ninfer/ops/dspark_ctx_commit.h"

#include <cuda_bf16.h>
#include <cstddef>

namespace ninfer::ops::detail {
namespace {

using namespace ninfer::ops::dspark;

// kv:      [kKvRows, T] BF16, dim0 fastest (k rows [0,1024), v rows [1024,2048))
// arena:   BF16 [kArenaElementsPerToken, capacity]; K_l[p] at token p, elements
//          l*kArenaElementsPerLayer + h*kHeadDim + [0,128), V_l at the same token,
//          offset + kKvHeadCount*kHeadDim.
__global__ void dspark_kv_rope_scatter_kernel(const __nv_bfloat16* __restrict__ kv,
                                              const __nv_bfloat16* __restrict__ k_norm,
                                              const float* __restrict__ inv_freq,
                                              const int* __restrict__ positions,
                                              __nv_bfloat16* __restrict__ arena, int layer, int T,
                                              float attention_scaling) {
    const int warp = (static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + threadIdx.x) >> 5;
    if (warp >= T * kKvHeadCount) return;
    const int t    = warp / kKvHeadCount;
    const int h    = warp % kKvHeadCount;
    const int lane = threadIdx.x & 31;
    const int p    = positions[t];

    // This token's k/v rows are contiguous: kv[t*kKvRows + r].
    const __nv_bfloat16* krow = kv + static_cast<std::size_t>(t) * kKvRows + h * kHeadDim;
    const __nv_bfloat16* vrow = krow + kKvHeadCount * kHeadDim;

    // Lane pairs: j0 = lane, j1 = lane + 32; pair j covers components (j, j + 64).
    // Together (j0, j0+64, j1, j1+64) cover [0, kHeadDim) disjointly.
    const int j0 = lane;
    const int j1 = lane + kHeadDimHalf / 2;
    const float k00 = __bfloat162float(krow[j0]);
    const float k01 = __bfloat162float(krow[j0 + kHeadDimHalf]);
    const float k10 = __bfloat162float(krow[j1]);
    const float k11 = __bfloat162float(krow[j1 + kHeadDimHalf]);

    // Per-head RMSNorm (gain k_norm, eps 1e-6, no offset). Warp-reduced sum of squares.
    float sq = k00 * k00 + k01 * k01 + k10 * k10 + k11 * k11;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sq += __shfl_xor_sync(0xffffffffu, sq, off);
    }
    const float inv = rsqrtf(sq * (1.0f / kHeadDim) + 1e-6f);

    const float x00 = k00 * __bfloat162float(k_norm[j0]) * inv;
    const float x01 = k01 * __bfloat162float(k_norm[j0 + kHeadDimHalf]) * inv;
    const float x10 = k10 * __bfloat162float(k_norm[j1]) * inv;
    const float x11 = k11 * __bfloat162float(k_norm[j1 + kHeadDimHalf]) * inv;

    const float pos = static_cast<float>(p);
    const float c0  = attention_scaling * cosf(inv_freq[j0] * pos);
    const float s0  = attention_scaling * sinf(inv_freq[j0] * pos);
    const float c1  = attention_scaling * cosf(inv_freq[j1] * pos);
    const float s1  = attention_scaling * sinf(inv_freq[j1] * pos);

    const std::size_t kbase =
        static_cast<std::size_t>(p) * kArenaElementsPerToken + layer * kArenaElementsPerLayer +
        h * kHeadDim;
    const std::size_t vbase = kbase + kKvHeadCount * kHeadDim;

    arena[kbase + j0]                = __float2bfloat16(x00 * c0 - x01 * s0);
    arena[kbase + j0 + kHeadDimHalf] = __float2bfloat16(x01 * c0 + x00 * s0);
    arena[kbase + j1]                = __float2bfloat16(x10 * c1 - x11 * s1);
    arena[kbase + j1 + kHeadDimHalf] = __float2bfloat16(x11 * c1 + x10 * s1);

    arena[vbase + j0]                = vrow[j0];
    arena[vbase + j0 + kHeadDimHalf] = vrow[j0 + kHeadDimHalf];
    arena[vbase + j1]                = vrow[j1];
    arena[vbase + j1 + kHeadDimHalf] = vrow[j1 + kHeadDimHalf];
}

} // namespace

void launch_dspark_kv_rope_scatter(const Tensor& kv, const Tensor& k_norm_weight, const Tensor& inv_freq,
                                   const Tensor& positions, Tensor& arena, int layer, std::int32_t T,
                                   float attention_scaling, cudaStream_t stream) {
    constexpr int kThreads = kKvHeadCount * 32;
    dim3 grid(T), block(kThreads);
    dspark_kv_rope_scatter_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(kv.data), static_cast<const __nv_bfloat16*>(k_norm_weight.data),
        static_cast<const float*>(inv_freq.data), static_cast<const int*>(positions.data),
        static_cast<__nv_bfloat16*>(arena.data), layer, T, attention_scaling);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail