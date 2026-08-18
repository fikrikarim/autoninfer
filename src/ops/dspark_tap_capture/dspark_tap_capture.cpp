// dspark_tap_capture wrapper: validates the DSpark auxiliary-tap store contract
// and issues the single stream-ordered copy of one layer's output into the
// transient tap buffer's slot block.

#include "ninfer/ops/dspark_tap_capture.h"
#include "ops/dspark_tap_capture/dspark_launch.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

using namespace ninfer::ops::dspark;

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("dspark_tap_capture: ") + what);
}

} // namespace

void dspark_tap_capture(const Tensor& src, std::int32_t slot, Tensor& dst, cudaStream_t stream) {
    const std::int32_t T = src.ne[1];
    require(src.dtype == DType::BF16 && src.ne[0] == kHidden && T >= 1 && src.ne[2] == 1 &&
                src.ne[3] == 1 && src.is_contiguous(),
            "src must be contiguous BF16 [5120, T, 1, 1] with T >= 1");
    require(slot >= 0 && slot < kLayers, "slot must be in [0, 5)");
    require(dst.dtype == DType::BF16 && dst.ne[0] == kTapWidth && dst.ne[1] == T && dst.ne[2] == 1 &&
                dst.ne[3] == 1 && dst.is_contiguous(),
            "dst must be the contiguous BF16 [25600, T, 1, 1] tap window matching src's token count");
    require(((reinterpret_cast<std::uintptr_t>(src.data) | reinterpret_cast<std::uintptr_t>(dst.data)) &
             0xfu) == 0,
            "src and dst must be 16-byte aligned");
    detail::launch_dspark_tap_capture(src, slot, dst, stream);
}

} // namespace ninfer::ops