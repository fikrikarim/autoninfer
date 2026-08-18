#pragma once

#include "ninfer/ops/dspark_ctx_commit.h" // ops::dspark lane geometry (kLayers, kHidden, kTapWidth)

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Op: dspark_tap_capture (DSpark auxiliary-tap store)
 *
 * Bit-exact store of one target decoder layer's post-residual-add output into
 * that layer's slot of the transient DSpark tap buffer. The DSpark speculator
 * (docs/maintainer/qwen3.8-27b-dspark-lane.md, Op 1) projects the five
 * auxiliary taps (target layers in slot order, e.g. 0-based 4/16/28/40/52 on
 * the 27B target) into the draft-KV arena (dspark_ctx_commit); this Op is the
 * producer side of that transient buffer.
 *
 * Math / indexing (t in [0, T), h in [0, dspark::kHidden)):
 *   dst[dspark::kHidden * slot + h][t] = src[h][t]      (bit-exact, no arithmetic)
 *
 * Logical shapes:
 *   src BF16 [dspark::kHidden, T] (dim0 fastest) - the post-residual-add output
 *   of one target decoder layer;
 *   slot host I32 in [0, dspark::kLayers); the caller's layer list (the
 *   checkpoint's auxiliary layers, in slot order) assigns the slot;
 *   dst BF16 [dspark::kTapWidth, T] (dim0 fastest) - the transient tap buffer
 *   window for this forward's token extent; only rows [slot*5120, (slot+1)*5120)
 *   are written.
 *
 * Supported domain:
 *   T >= 1 (verify: T = verify width, <= 8; prefill: T = prompt chunk). The dst
 *   buffer is transient and caller-sized per chunk (the DSpark lane projects
 *   each chunk and frees it; there is no persistent raw-tap arena). The caller
 *   passes the [dspark::kTapWidth, T] window of a larger chunk buffer by
 *   slicing it to this forward's token count.
 *
 * Numeric:
 *   No arithmetic, no cast: a bit-exact copy. Exact oracle (byte compare).
 *
 * Effects:
 *   Writes only the slot block's rows of dst; all other dst rows and src are
 *   untouched.
 *
 * Data representation:
 *   src.data and dst.data must be 16-byte aligned (workspace-arena allocations
 *   are 256-byte aligned).
 *
 * Execution:
 *   Stream-ordered on the caller's stream; safe under CUDA Graph capture when
 *   the buffer addresses are stable for the buffer's lifetime.
 */
void dspark_tap_capture(const Tensor& src, std::int32_t slot, Tensor& dst, cudaStream_t stream);

} // namespace ninfer::ops