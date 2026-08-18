// Private launcher for the DSpark auxiliary-tap store kernel.
#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstdint>

namespace ninfer::ops::detail {

// Bit-exact copy of src [dspark::kHidden, T] into the slot block
// [slot*5120, (slot+1)*5120) of dst [dspark::kTapWidth, T]; both dim0-fastest
// BF16, 16-byte-aligned. One 16-byte vector per thread, one block per token.
void launch_dspark_tap_capture(const Tensor& src, std::int32_t slot, Tensor& dst, cudaStream_t stream);

} // namespace ninfer::ops::detail