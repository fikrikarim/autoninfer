// DSpark auxiliary-tap store (lane Op 1): bit-exact copy of one target layer's
// post-residual-add output into the slot block of the transient tap buffer.
//
// Tensor convention: dim0 is stored fastest, so element (r, t) of an [R, T]
// tensor lives at t * R + r. src [5120, T], dst [25600, T]; the slot block is
// dst rows [slot*5120, (slot+1)*5120). One 16-byte vector per thread (8 BF16),
// one block per token: consecutive threads read consecutive 16-byte vectors
// within the token, so every row is a fully coalesced copy.

#include "ops/dspark_tap_capture/dspark_launch.h"
#include "core/device.h"
#include "ninfer/ops/dspark_tap_capture.h"

#include <cstddef>

namespace ninfer::ops::detail {
namespace {

using namespace ninfer::ops::dspark;

constexpr std::int64_t kSrcRowVectors = kHidden / 8;        // 640
constexpr std::int64_t kDstRowVectors = kTapWidth / 8;      // 3200

__global__ void dspark_tap_capture_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst,
                                          std::int64_t slot_vectors) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x);
    const std::int64_t lane = static_cast<std::int64_t>(threadIdx.x);
    dst[row * kDstRowVectors + slot_vectors + lane] = src[row * kSrcRowVectors + lane];
}

} // namespace

void launch_dspark_tap_capture(const Tensor& src, std::int32_t slot, Tensor& dst, cudaStream_t stream) {
    const std::int32_t T = src.ne[1];
    dspark_tap_capture_kernel<<<T, kSrcRowVectors, 0, stream>>>(
        static_cast<const uint4*>(src.data), static_cast<uint4*>(dst.data),
        static_cast<std::int64_t>(slot) * kSrcRowVectors);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail