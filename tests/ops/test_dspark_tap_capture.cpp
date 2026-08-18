// dspark_tap_capture (DSpark auxiliary-tap store) correctness suite.
//
// Oracle: exact - the Op is a bit-exact copy (no arithmetic, no cast), so the
// oracle is the input bytes at the slot block position and the untouched bytes
// elsewhere. Sources are raw 16-bit random patterns (no BF16 value restriction),
// distinct per slot so any cross-slot leakage is detected. Covers all five slots
// into one tap buffer, token extents T = 1 (decode), 8 (DSpark verify width),
// 129 (odd prefill chunk), 4096 (the default prefill chunk), guarded-device
// overflow detection, and contract validation.

#include "ninfer/ops/dspark_tap_capture.h"

#include "ops/op_tester.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::test;
namespace ds = ninfer::ops::dspark;

namespace {

constexpr std::int32_t kHidden   = ds::kHidden;
constexpr std::int32_t kTapWidth = ds::kTapWidth;
constexpr std::int32_t kLayers   = ds::kLayers;

std::vector<std::uint16_t> random_bits(std::size_t n, std::uint32_t seed) {
    std::mt19937 g(seed);
    std::uniform_int_distribution<std::uint16_t> d(0, 0xffff);
    std::vector<std::uint16_t> bits(n);
    for (auto& x : bits) x = d(g);
    return bits;
}

int run_store_case(cudaStream_t stream, int T, std::uint32_t seed_base) {
    int failures = 0;
    const std::size_t src_elems = static_cast<std::size_t>(kHidden) * T;
    const std::size_t dst_elems = static_cast<std::size_t>(kTapWidth) * T;

    std::vector<std::vector<std::uint16_t>> sources(kLayers);
    std::vector<DeviceBuffer> src_buffers(kLayers);
    std::vector<Tensor> src_tensors(kLayers);
    for (std::int32_t s = 0; s < kLayers; ++s) {
        sources[s] = random_bits(src_elems, seed_base + 1000u * static_cast<std::uint32_t>(s));
        src_buffers[s] = to_device(sources[s]);
        src_tensors[s] = Tensor(src_buffers[s].p, DType::BF16, {kHidden, T});
    }

    GuardedDeviceBuffer dst_guard(dst_elems * 2);
    dst_guard.fill(0);
    Tensor dst = Tensor(dst_guard.data(), DType::BF16, {kTapWidth, T});
    // The guard fill and the uploads above are legacy-default-stream work; the Op runs on
    // the caller's (non-blocking) stream, so order the prep before the captures.
    cuda_synchronize();

    for (std::int32_t s = 0; s < kLayers; ++s) {
        dspark_tap_capture(src_tensors[s], s, dst, stream);
    }
    cuda_synchronize(stream);
    failures += dst_guard.verify_guards("tap store T=" + std::to_string(T));

    const std::vector<std::uint16_t> got = from_device<std::uint16_t>(dst.data, dst_elems);
    // Exact oracle: row s*5120..(s+1)*5120 of dst must equal source s bit-for-bit.
    auto expected = [&](std::size_t idx) -> std::uint16_t {
        const std::int32_t t     = static_cast<std::int32_t>(idx / kTapWidth);
        const std::int32_t r     = static_cast<std::int32_t>(idx % kTapWidth);
        const std::int32_t s     = r / kHidden;
        const std::int32_t within = r % kHidden;
        return sources[static_cast<std::size_t>(s)][static_cast<std::size_t>(t) * kHidden + within];
    };
    const std::size_t first_mismatch = [&]() {
        for (std::size_t i = 0; i < dst_elems; ++i) {
            if (got[i] != expected(i)) return i;
        }
        return dst_elems;
    }();
    if (first_mismatch != dst_elems) {
        failures += 1;
        std::cerr << "tap store T=" << T << ": first mismatch at element " << first_mismatch
                  << " (slot " << (first_mismatch % kTapWidth) / kHidden << ", token "
                  << first_mismatch / kTapWidth << ")\n";
    }
    return failures;
}

bool throws_invalid_argument(const char* what, auto&& fn) {
    try {
        fn();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (const std::exception& e) {
        std::cerr << what << ": wrong exception type: " << e.what() << "\n";
        return false;
    }
    std::cerr << what << ": no exception thrown\n";
    return false;
}

int run_validation_check(cudaStream_t stream) {
    int failures = 0;
    constexpr int T = 8;
    std::vector<std::uint16_t> bits = random_bits(kHidden * T, 0x5e00);
    DeviceBuffer src = to_device(bits);
    DeviceBuffer dst = to_device(std::vector<std::uint16_t>(kTapWidth * T, 0));
    Tensor src_t = Tensor(src.p, DType::BF16, {kHidden, T});
    Tensor dst_t = Tensor(dst.p, DType::BF16, {kTapWidth, T});

    if (!throws_invalid_argument("slot range", [&] { dspark_tap_capture(src_t, kLayers, dst_t, stream); })) {
        failures += 1;
    }
    if (!throws_invalid_argument("negative slot", [&] { dspark_tap_capture(src_t, -1, dst_t, stream); })) {
        failures += 1;
    }
    {
        DeviceBuffer narrow = to_device(std::vector<std::uint16_t>(kHidden * T - 1, 0));
        Tensor bad_src      = Tensor(narrow.p, DType::BF16, {kHidden - 1, T});
        if (!throws_invalid_argument("src rows",
                                     [&] { dspark_tap_capture(bad_src, 0, dst_t, stream); })) {
            failures += 1;
        }
        DeviceBuffer i32 = to_device(std::vector<std::int32_t>(kHidden * T, 0));
        Tensor wrong_dtype = Tensor(i32.p, DType::I32, {kHidden, T});
        if (!throws_invalid_argument("src dtype",
                                     [&] { dspark_tap_capture(wrong_dtype, 0, dst_t, stream); })) {
            failures += 1;
        }
        // Non-contiguous src: a [5120, T] row slice of a [10240, T] buffer.
        DeviceBuffer wide = to_device(std::vector<std::uint16_t>(10240 * T, 0));
        Tensor strided    = Tensor(wide.p, DType::BF16, {10240, T}).slice(0, 0, kHidden);
        if (!throws_invalid_argument("src contiguity",
                                     [&] { dspark_tap_capture(strided, 0, dst_t, stream); })) {
            failures += 1;
        }
    }
    {
        DeviceBuffer short_dst = to_device(std::vector<std::uint16_t>(kTapWidth * (T - 1), 0));
        Tensor bad_dst         = Tensor(short_dst.p, DType::BF16, {kTapWidth, T - 1});
        if (!throws_invalid_argument("dst token count",
                                     [&] { dspark_tap_capture(src_t, 0, bad_dst, stream); })) {
            failures += 1;
        }
        DeviceBuffer narrow_dst = to_device(std::vector<std::uint16_t>((kTapWidth - 1) * T, 0));
        Tensor narrow_dst_t     = Tensor(narrow_dst.p, DType::BF16, {kTapWidth - 1, T});
        if (!throws_invalid_argument("dst rows",
                                     [&] { dspark_tap_capture(src_t, 0, narrow_dst_t, stream); })) {
            failures += 1;
        }
        // Unaligned src (2-byte offset inside an aligned allocation).
        DeviceBuffer aligned = to_device(bits);
        Tensor unaligned     = Tensor(static_cast<std::uint16_t*>(aligned.p) + 1, DType::BF16,
                                      {kHidden, T});
        if (!throws_invalid_argument("src alignment",
                                     [&] { dspark_tap_capture(unaligned, 0, dst_t, stream); })) {
            failures += 1;
        }
    }
    cuda_synchronize(stream);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    cudaStream_t stream;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");

    int failures = 0;
    failures += run_validation_check(stream);
    for (const int t : {1, 8, 129, 4096}) {
        failures += run_store_case(stream, t, static_cast<std::uint32_t>(0x1000 + t));
    }

    std::cout << (failures == 0 ? "OK" : "FAIL") << " dspark_tap_capture correctness\n";
    cudaStreamDestroy(stream);
    return failures == 0 ? 0 : 1;
}