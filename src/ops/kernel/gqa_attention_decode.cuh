#pragma once

// ninfer::ops - split-KV GQA small-T attention shared scaffolding. The bf16 and
// int8 partial kernels live in gqa_attention_decode_bf16.cuh and
// gqa_attention_decode_i8.cuh respectively; they are fully separate kernels (no
// shared body) so each KV format can be optimized independently. This header owns
// only what both share: layout constants, device helpers, and the split reducer.

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/gqa_attention_geometry.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaHeadDim = 256;

struct GqaAppendInput {
    static constexpr bool writes_cache = true;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
};

struct GqaCachedInput {
    static constexpr bool writes_cache = false;
};

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_cache_index(int physical_page, int kv_head, int d,
                                                        int page_offset) {
    return paged_kv_element_offset<kGqaHeadDim, Geometry::KVHeads>(physical_page, kv_head,
                                                                   page_offset, d);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_q_index(int q_head, int d, int token = 0) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kGqaHeadDim) *
                                              (static_cast<std::int64_t>(q_head) +
                                               static_cast<std::int64_t>(Geometry::QHeads) * token);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_new_index(int kv_head, int d, int token = 0) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaHeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_partial_acc_index(int q_head, int d, int token,
                                                              int split, int tokens) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaHeadDim) *
               (static_cast<std::int64_t>(q_head) +
                static_cast<std::int64_t>(Geometry::QHeads) *
                    (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split));
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_partial_stat_index(int q_head, int token, int split,
                                                               int tokens) {
    return static_cast<std::int64_t>(q_head) +
           static_cast<std::int64_t>(Geometry::QHeads) *
               (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split);
}

template <typename Geometry>
__device__ __forceinline__ bool gqa_valid_q_head(int kv_head, int q_head) {
    return kv_head >= 0 && kv_head < Geometry::KVHeads && q_head >= kv_head * Geometry::GroupSize &&
           q_head < (kv_head + 1) * Geometry::GroupSize && q_head < Geometry::QHeads;
}

template <typename Geometry>
__device__ __forceinline__ int gqa_small_t_default_splits(int window) {
    int target_keys_per_split = 480 / Geometry::DecodeSplitScale;
    if (window <= 4096) {
        target_keys_per_split = 64 / Geometry::DecodeSplitScale;
    } else if (window <= 8198) {
        target_keys_per_split = 128 / Geometry::DecodeSplitScale;
    } else if (window <= 16390) {
        target_keys_per_split = 256 / Geometry::DecodeSplitScale;
    }
    constexpr int kMinSplits = 4 * Geometry::DecodeSplitScale;
    int splits               = div_up(window, target_keys_per_split);
    splits                   = splits > kMinSplits ? splits : kMinSplits;
    return splits < Geometry::DecodeSplits ? splits : Geometry::DecodeSplits;
}

// Committed-column split count: the width-independent T=1 decode policy. The
// committed column must clone the width-1 route bit-for-bit, so the partition
// (active count and per-split units) is a pure function of the committed
// window; any tokens-dependent count would re-partition column 0, perturb the
// verify target argmax, and corrupt the emitted stream.
template <typename Geometry>
__device__ __forceinline__ std::int32_t gqa_small_t_active_splits(
    std::int32_t window, std::int32_t launch_capacity) {
    if (window <= 0) { return launch_capacity; }
    const std::int32_t splits = gqa_small_t_default_splits<Geometry>(window);
    return splits < launch_capacity ? splits : launch_capacity;
}

struct GqaSmallTSplitRange {
    std::int32_t start;
    std::int32_t end;
};

// Committed-column split partition (greedy MTP losslessness, model-doc 8). A
// verify round commits its first column, and that column must be a
// finite-precision clone of the ordinary T=1 decode route at the same state.
// The partition is therefore derived from the committed column's key window
// [0, committed_window) - the same active count and per-split units a width-1
// decode would use - and the draft columns' own keys [committed_window,
// full_window) are folded into the split that owns key committed_window - 1, so
// [0, full_window) is covered exactly once. For the committed column the causal
// mask makes that extended tail a numerical no-op (masked scores -inf -> p = 0,
// alpha = 1), so its per-split partials and the split-ordered combine are
// bit-identical to the T=1 route for every verify width.
// Returns the split's {start, end}; a negative start means the split exceeds the
// active count and must return without writing the neutral value (the reducer
// never reads it).
template <typename Geometry, int Bc>
__device__ __forceinline__ GqaSmallTSplitRange gqa_small_t_split_range(
    std::int32_t committed_window, std::int32_t full_window, std::int32_t split,
    std::int32_t launch_capacity) {
    const std::int32_t active_split_count =
        gqa_small_t_active_splits<Geometry>(committed_window, launch_capacity);
    if (split >= active_split_count) { return {-1, -1}; }
    const std::int32_t logical_tiles = div_up(committed_window, Bc);
    const bool tile_split            = logical_tiles >= active_split_count;
    const std::int32_t units_per_split =
        tile_split ? div_up(logical_tiles, active_split_count)
                   : div_up(committed_window, active_split_count);
    const std::int32_t unit_keys   = units_per_split * (tile_split ? Bc : 1);
    const std::int32_t split_start = split * unit_keys;
    const std::int32_t split_limit = split_start + unit_keys;
    const std::int32_t extended    = (committed_window - 1) / unit_keys;
    // The split owning committed_window - 1 is extended to the full verify window so the
    // draft columns' own keys are covered exactly once; every other split is capped at the
    // committed window (a later split whose start reaches the window is empty).
    const std::int32_t split_end = (split == extended)
                                       ? full_window
                                       : (split_limit < committed_window ? split_limit
                                                                         : committed_window);
    return {split_start, split_end};
}

__device__ __forceinline__ int gqa_small_t_tc_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __forceinline__ int gqa_small_t_tc_swz32(int row, int col) {
    return (((col >> 3) ^ (row & 3)) << 3) | (col & 7);
}

// Signed int8 QK MMA, k=32 contraction. A = 16x32 s8 (4 regs/thread, 4 s8 each),
// B = 8x32 s8 col-major (2 regs/thread), D = 16x8 s32 (4 regs/thread). The A/B
// register byte layout is identical to the m16n8k16 bf16 fragments loaded by
// ldmatrix_x4/x2 over a d-contiguous int8 tile reinterpreted as
// b16 (two packed int8 per 16-bit lane), so the same ldmatrix helpers and XOR
// swizzle feed this MMA. The s32 accumulator layout matches the bf16 f32
// accumulator (c0/c1 -> row groupID, c2/c3 -> row groupID+8), so score
// consumption is unchanged; only per-64-group scale rescale differs.
template <typename Geometry>
__device__ __forceinline__ void gqa_small_t_tc_row_to_qt(int row, int tokens, int kv_head,
                                                         int& q_head, int& token) {
    token             = row / Geometry::GroupSize;
    const int local_q = row - token * Geometry::GroupSize;
    q_head            = kv_head * Geometry::GroupSize + local_q;
}

template <typename Geometry, int DChunk, bool Int8, bool MultiBatch, bool Masked, bool Offset>
__launch_bounds__(256) __global__ void gqa_attention_small_t_reduce_output_kernel(
    const __nv_bfloat16* partial_acc, const float* partial_m, const float* partial_l,
    const std::int32_t* positions, const std::int32_t* valid_columns, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t batch_size,
    std::int32_t split_count, __nv_bfloat16* out) {
    static_assert(DChunk > 0 && DChunk <= kGqaHeadDim);

    const int q_head      = static_cast<int>(blockIdx.x);
    const int d_start     = static_cast<int>(blockIdx.y) * DChunk;
    const int flat_column = static_cast<int>(blockIdx.z);
    int batch             = 0;
    int token             = flat_column;
    if constexpr (MultiBatch) {
        batch = flat_column / tokens;
        token = flat_column - batch * tokens;
    }
    const int tid = threadIdx.x;
    if (q_head >= Geometry::QHeads || token >= tokens) { return; }
    if constexpr (MultiBatch) {
        if (batch >= batch_size) { return; }
    }

    if constexpr (Offset) { positions += column_begin; }
    if constexpr (MultiBatch) { positions += batch * full_width; }
    int output_column = token;
    if constexpr (Offset) { output_column += column_begin; }
    if constexpr (MultiBatch) { output_column += batch * full_width; }

    if constexpr (MultiBatch) {
        const std::int64_t partial_acc_row = static_cast<std::int64_t>(batch) * kGqaHeadDim *
                                             Geometry::QHeads * tokens * split_count;
        const std::int64_t partial_stat_row =
            static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_acc += partial_acc_row;
        partial_m += partial_stat_row;
        partial_l += partial_stat_row;
    }

    // The committed-column partition (gqa_small_t_split_range) keys the active split
    // count off the first column's window; match it exactly.
    const int committed_window = positions[0] + 1;
    const int active_split_count =
        gqa_small_t_active_splits<Geometry>(committed_window, split_count);

    __shared__ float reduce[256];

    float local_m = -CUDART_INF_F;
    for (int split = tid; split < active_split_count; split += blockDim.x) {
        local_m = fmaxf(local_m,
                        partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)]);
    }
    reduce[tid] = local_m;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]); }
        __syncthreads();
    }
    const float head_m = reduce[0];
    __syncthreads();

    if (head_m == -CUDART_INF_F) {
        const int d = d_start + tid;
        if (tid < DChunk && d < kGqaHeadDim) {
            out[gqa_q_index<Geometry>(q_head, d, output_column)] = __float2bfloat16(0.0f);
        }
        return;
    }

    float local_l = 0.0f;
    for (int split = tid; split < active_split_count; split += blockDim.x) {
        const float tile_l =
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)];
        if (tile_l > 0.0f) {
            local_l +=
                tile_l *
                expf(partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] -
                     head_m);
        }
    }
    reduce[tid] = local_l;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) { reduce[tid] += reduce[tid + stride]; }
        __syncthreads();
    }
    const float head_l = reduce[0];

    const int d = d_start + tid;
    if (tid >= DChunk || d >= kGqaHeadDim) { return; }

    float numerator = 0.0f;
    if (head_l > 0.0f) {
        for (int split = 0; split < active_split_count; ++split) {
            const float tile_l =
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)];
            if (tile_l <= 0.0f) { continue; }
            const float weight = expf(
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] - head_m);
            numerator +=
                __bfloat162float(
                    partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens)]) *
                weight;
        }
    }
    bool valid = true;
    if constexpr (Masked) {
        int absolute_column = token;
        if constexpr (Offset) { absolute_column += column_begin; }
        valid = absolute_column < valid_columns[batch];
    }
    const float value = (valid && head_l > 0.0f) ? numerator / head_l : 0.0f;
    out[gqa_q_index<Geometry>(q_head, d, output_column)] = __float2bfloat16(value);
}

} // namespace ninfer::ops
