// Bit-exact T=1 decode-route parity for the FP8 row-scaled linear family at MTP verify widths
// (T = draft_tokens+1 <= 4), per committed column (model-doc 8).
//
// Two contracts are pinned here:
//
// 1. A16 family (attn input, GDN input, residual output projections): the T=2..4 small-t
//    instantiations reproduce the T=1 GEMV decode association (vpl=8, four interleaved
//    accumulator chains, same pair-to-chain mapping and final chain sum). Every output column
//    of a T-width launch must be bit-identical to the T=1 launch on the same input column.
//
// 2. A8 family (FP8 MLP gate/up parent): the T=1 decode route is A8, so every token count uses
//    A8 (plain `linear`, `linear_swiglu`). The A8 MMA is T-independent per column (sequential K
//    accumulation in one warp per token-row block) and its activation quantization is
//    per-token, so the same bit-identity must hold across T=1..4.
//
// Any divergence here flips rare last-ULP BF16 values into committed-column differences,
// perturbs the target argmax, and corrupts the greedy MTP stream relative to the k=0 reference
// (the 2026-08-18 losslessness bug, FP8 residual: the NVFP4 family received the 4-chain clone
// in 5cbba58c while the FP8 family kept the one-chain small-t association and an A16 island
// over its A8 T=1 reference).

#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::quantized_weight;

namespace {

constexpr std::int32_t kHidden = 5120;
constexpr std::int32_t kMaxVerify = 4;
constexpr QType kQType = QType::FP8_E4M3FN_ROW_BF16S;

enum class OpKind : std::uint8_t {
    Linear,
    LinearAdd,
    LinearSwiGlu,
};

struct Case {
    const char* name;
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
    OpKind op;
    // The T=1 reference route for this parent. A8 parents pin the route change (verify widths
    // must stay on the T=1 A8 route); A16 parents pin the small-t association clone.
    bool a8_reference;
};

constexpr std::array kCases{
    Case{"AttnInput[14336,5120]", 14336, kHidden, 901U, OpKind::Linear, false},
    Case{"GdnInput[16384,5120]", 16384, kHidden, 907U, OpKind::Linear, false},
    Case{"MlpGateUp[34816,5120]", 34816, kHidden, 911U, OpKind::Linear, true},
    Case{"Residual6144[5120,6144]", 5120, 6144, 913U, OpKind::LinearAdd, false},
    Case{"Residual17408[5120,17408]", 5120, 17408, 917U, OpKind::LinearAdd, false},
    Case{"SwiGlu[34816,5120]", 34816, kHidden, 919U, OpKind::LinearSwiGlu, true},
};

// Deterministic BF16 activation [k, tokens] on a narrow grid: representable exactly by the
// quantized A8 activation route for the per-token scale while still exercising both A16 routes.
std::vector<std::uint16_t> make_activation(std::int32_t rows, std::int32_t tokens,
                                           std::uint32_t seed) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t row = 0; row < rows; ++row) {
            std::uint32_t value = seed ^ static_cast<std::uint32_t>(row) * 0x9e3779b9U ^
                                  static_cast<std::uint32_t>(token) * 0x85ebca6bU;
            value ^= value >> 16;
            value *= 0x7feb352dU;
            value ^= value >> 15;
            const float represented =
                static_cast<float>(static_cast<int>(value & 0xffU) - 128) * (1.0F / 256.0F);
            result[static_cast<std::size_t>(token) * rows + row] = f32_to_bf16(represented);
        }
    }
    return result;
}

std::size_t op_capacity(const Case& c, std::int32_t first, std::int32_t last) {
    switch (c.op) {
    case OpKind::Linear:
        return ops::linear_workspace_capacity_bytes(kQType, c.n, c.k,
                                                    ops::LinearPolicy::AllowA8, first, last);
    case OpKind::LinearAdd:
        return ops::linear_add_workspace_capacity_bytes(kQType, c.n, c.k,
                                                        ops::LinearPolicy::AllowA8, first, last);
    case OpKind::LinearSwiGlu:
        return ops::linear_swiglu_workspace_capacity_bytes(kQType, c.n, c.k,
                                                           ops::LinearPolicy::AllowA8, first, last);
    }
    return 0;
}

// Runs one op on the first `tokens` activation columns and returns the raw BF16 output bits
// ([out_rows, tokens]). out_rows is n/2 for the swiglu form.
std::vector<std::uint16_t> run_op(const Case& c, const Weight& weight,
                                  const void* activation, std::int32_t tokens,
                                  std::size_t out_rows, std::size_t capacity) {
    GuardedDeviceBuffer input(static_cast<std::size_t>(c.k) * tokens * sizeof(std::uint16_t));
    input.copy_from_host(activation, input.bytes());
    GuardedDeviceBuffer output(out_rows * static_cast<std::size_t>(tokens) * sizeof(std::uint16_t));
    Tensor x(input.data(), DType::BF16, {c.k, tokens});
    Tensor out(output.data(), DType::BF16, {static_cast<std::int32_t>(out_rows), tokens});
    WorkspaceArena workspace(std::max(capacity, std::size_t{256}));
    switch (c.op) {
    case OpKind::Linear:
        ops::linear(x, weight, out, ops::LinearPolicy::AllowA8, workspace, nullptr);
        break;
    case OpKind::LinearAdd:
        ops::linear_add(x, weight, out, ops::LinearPolicy::AllowA8, workspace, nullptr);
        break;
    case OpKind::LinearSwiGlu:
        ops::linear_swiglu(x, weight, out, ops::LinearPolicy::AllowA8, workspace, nullptr);
        break;
    }
    cuda_check(cudaDeviceSynchronize(), "synchronize FP8 T-parity op");
    std::vector<std::uint16_t> bits(out_rows * static_cast<std::size_t>(tokens));
    output.copy_to_host(bits.data(), output.bytes());
    return bits;
}

int run_case(const Case& c, int failures) {
    const PackedWeight host_weight = make_patterned_weight(kQType, c.n, c.k, c.seed);
    GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    const Weight weight = host_weight.device_weight(device_weight.data());

    const std::vector<std::uint16_t> activation = make_activation(c.k, kMaxVerify, c.seed + 1U);
    const std::size_t out_rows =
        c.op == OpKind::LinearSwiGlu ? static_cast<std::size_t>(c.n) / 2 : c.n;
    const std::size_t capacity = op_capacity(c, 1, kMaxVerify);

    // T=1 reference: one launch per input column.
    std::vector<std::vector<std::uint16_t>> reference(kMaxVerify);
    for (std::int32_t column = 0; column < kMaxVerify; ++column) {
        const void* column_input =
            reinterpret_cast<const std::uint8_t*>(activation.data()) +
            static_cast<std::size_t>(column) * c.k * sizeof(std::uint16_t);
        reference[static_cast<std::size_t>(column)] =
            run_op(c, weight, column_input, 1, out_rows, capacity);
    }

    // T=2..4 verify widths: every column must bit-clone its T=1 reference.
    for (std::int32_t tokens = 2; tokens <= kMaxVerify; ++tokens) {
        const std::vector<std::uint16_t> multi = run_op(c, weight, activation.data(), tokens,
                                                        out_rows, capacity);
        for (std::int32_t column = 0; column < tokens; ++column) {
            const std::uint16_t* actual = multi.data() + static_cast<std::size_t>(column) * out_rows;
            const std::vector<std::uint16_t>& expected = reference[static_cast<std::size_t>(column)];
            if (std::equal(expected.begin(), expected.end(), actual, actual + out_rows)) {
                continue;
            }
            std::cerr << c.name << ": T=" << tokens << " column " << column
                      << " diverges from the T=1 decode route\n";
            int first_diff = -1;
            for (std::size_t row = 0; row < out_rows; ++row) {
                if (actual[row] != expected[row]) {
                    first_diff = static_cast<int>(row);
                    break;
                }
            }
            std::cerr << "  first differing row " << first_diff << ": T=" << tokens << " 0x"
                      << std::hex << actual[first_diff] << " vs T=1 0x" << expected[first_diff]
                      << std::dec << '\n';
            ++failures;
        }
    }

    // Route pin: the A8-reference parents must report A8 activation scratch across the whole
    // verify interval (T=1..4); the A16 parents must report none.
    const std::size_t verify_interval = op_capacity(c, 1, kMaxVerify);
    const bool expected_a8 = c.a8_reference;
    if (expected_a8) {
        for (std::int32_t tokens = 1; tokens <= kMaxVerify; ++tokens) {
            if (op_capacity(c, tokens, tokens) == 0 || verify_interval == 0 ||
                verify_interval != op_capacity(c, kMaxVerify, kMaxVerify)) {
                std::cerr << c.name << ": A8 route not pinned for T=" << tokens << '\n';
                ++failures;
            }
        }
    } else if (verify_interval != 0) {
        std::cerr << c.name << ": A16 parent reported activation scratch\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    for (const Case& c : kCases) {
        failures = run_case(c, failures);
    }
    if (failures != 0) {
        std::cerr << failures << " FP8 linear T-parity failure(s)\n";
        return 1;
    }
    std::cout << "PASS: FP8 linear T=1..4 per-column bit parity\n";
    return 0;
}