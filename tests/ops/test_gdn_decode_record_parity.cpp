// Bit-exact parity between the two production GDN chains that must agree for greedy MTP
// losslessness.
//
// Path A (k=0 ordinary decode): `width` repeated width-1 in-place snapshot steps, exactly as
// TextContext::ordinary_decode_batch drives gdn_mix with Phase::Verify, width 1, dense
// (no valid_columns), snapshot base == initial slot.
//
// Path B (k>0 MTP verify): one width-W `gdn_input_proj_conv_record` +
// `gated_delta_net_replay_record` round (masked by valid_columns, as the MTP frame always
// passes a device valid tensor), then `gdn_replay_fold` committing the first `valid` columns.
//
// Greedy (temperature 0) MTP emission is lossless only if, for the same committed prefix and
// same initial state, both chains produce identical per-column BF16 q/k/v/z and recurrent
// outputs and bit-identical FP32 recurrent / BF16 convolution state at every commit. Any
// divergence here perturbs the recurrent state and can flip target argmaxes, corrupting the
// emitted stream relative to the k=0 reference (the 2026-08-18 losslessness bug).
//
// Registered 27B NVFP4 geometry: parent [16384,5120] q/k/value/z row order, conv [10240,4],
// 16 q/k heads, 48 value heads, state 128. Production text policy for NVFP4 is AllowA4; an
// A16Only case isolates the activation-quantized GEMM.

#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gdn_replay.h"

#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "ops/input_projection_test_common.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <bit>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

constexpr std::int32_t kHidden       = 5120;
constexpr std::int32_t kParentRows   = 16384;
constexpr std::int32_t kConvChannels = 10240;
constexpr std::int32_t kQkHeads      = 16;
constexpr std::int32_t kValueHeads   = 48;
constexpr std::int32_t kStateDim     = 128;
constexpr std::int32_t kQueryRows    = kStateDim * kQkHeads;
constexpr std::int32_t kValueRows    = kStateDim * kValueHeads;
constexpr std::int32_t kZRows        = kStateDim * kValueHeads;
constexpr std::int32_t kGdnLayers    = 48; // registered 27B fold geometry (48x48)
constexpr float kScale               = 1.0F / std::sqrt(128.0F);

std::uint32_t mix(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    return value ^ (value >> 16);
}

float f32_pattern(std::uint32_t key, float lo, float hi) {
    const float unit = static_cast<float>(mix(key) % 100000U) / 100000.0F;
    return lo + unit * (hi - lo);
}

std::uint16_t bf16_pattern(std::uint32_t key, float lo, float hi) {
    return f32_to_bf16(f32_pattern(key, lo, hi));
}

struct Case {
    std::int32_t width;
    std::int32_t valid;
    ops::LinearPolicy policy;
    std::uint32_t seed;
};

std::string policy_name(ops::LinearPolicy policy) {
    return policy == ops::LinearPolicy::AllowA4 ? "A4" : "A16";
}

struct Report {
    int failures = 0;

    template <class T>
    void compare_bits(const std::string& label, std::int32_t column,
                      const std::vector<T>& lhs, const std::vector<T>& rhs) {
        if (lhs.size() != rhs.size()) {
            std::cerr << label << " column=" << column << " size mismatch\n";
            ++failures;
            return;
        }
        int first = -1;
        int hard  = 0;
        int sign_zero = 0;
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            if (lhs[i] == rhs[i]) { continue; }
            if constexpr (std::is_same_v<T, std::uint16_t>) {
                const bool is_sign_zero =
                    (lhs[i] == 0x0000U && rhs[i] == 0x8000U) ||
                    (lhs[i] == 0x8000U && rhs[i] == 0x0000U);
                if (is_sign_zero) {
                    ++sign_zero;
                    continue;
                }
            }
            if (first < 0) { first = static_cast<int>(i); }
            ++hard;
            if (hard <= 8) {
                std::cerr << label << " column=" << column << " idx=" << i << " lhs=0x" << std::hex
                          << static_cast<std::uint32_t>(lhs[i]) << " rhs=0x"
                          << static_cast<std::uint32_t>(rhs[i]) << std::dec << "\n";
            }
        }
        if (hard == 0) {
            if (sign_zero > 0) {
                std::cerr << label << " column=" << column << ": " << sign_zero
                          << " sign-of-zero-only differences (soft)\n";
            }
            return;
        }
        std::cerr << label << " column=" << column << ": " << hard << " bit differences, first="
                  << first << " (" << sign_zero << " sign-of-zero)\n";
        ++failures;
    }

    void compare_f32(const std::string& label, std::int32_t column,
                     const std::vector<float>& lhs, const std::vector<float>& rhs) {
        if (lhs.size() != rhs.size()) {
            std::cerr << label << " column=" << column << " size mismatch\n";
            ++failures;
            return;
        }
        int first = -1;
        int count = 0;
        for (std::size_t i = 0; i < lhs.size(); ++i) {
            if (lhs[i] == rhs[i]) { continue; }
            if (first < 0) { first = static_cast<int>(i); }
            ++count;
            if (count <= 8) {
                std::cerr << label << " column=" << column << " idx=" << i << " lhs=0x" << std::hex
                          << std::bit_cast<std::uint32_t>(lhs[i]) << " rhs=0x"
                          << std::bit_cast<std::uint32_t>(rhs[i]) << std::dec << " ("
                          << lhs[i] << " vs " << rhs[i] << ")\n";
            }
        }
        if (count == 0) { return; }
        std::cerr << label << " column=" << column << ": " << count << " value differences, first="
                  << first << "\n";
        ++failures;
    }
};

struct Pool {
    DeviceBuffer storage;
    LinearAttentionStatePool pool;

    Pool(std::uint32_t layers, std::int32_t slot_count, const std::vector<std::uint16_t>& initial_conv,
         const std::vector<float>& initial_recurrent)
        : storage(plan_bytes(layers, slot_count)) {
        LayoutBuilder builder;
        const LinearAttentionStatePoolLayout layout =
            plan_linear_attention_state_pool(builder, {.layers         = layers,
                                                       .conv_channels  = kConvChannels,
                                                       .conv_width     = 3,
                                                       .value_heads    = kValueHeads,
                                                       .value_head_dim = kStateDim,
                                                       .key_head_dim   = kStateDim,
                                                       .slot_count     = slot_count,
                                                       .conv_dtype     = DType::BF16});
        pool = LinearAttentionStatePool({storage.p, storage.bytes}, layout);
        // Layer 0 slot 0 receives the driven initial state; other layers/slots stay untouched
        // (the fold rewrites every layer's selected slot; only layer 0 is inspected).
        cuda_check(cudaMemcpy(pool.conv[0].data, initial_conv.data(), initial_conv.size() * 2,
                              cudaMemcpyHostToDevice),
                   "upload initial conv state");
        cuda_check(cudaMemcpy(pool.recurrent[0].data, initial_recurrent.data(),
                              initial_recurrent.size() * 4, cudaMemcpyHostToDevice),
                   "upload initial recurrent state");
    }

    static std::size_t plan_bytes(std::uint32_t layers, std::int32_t slot_count) {
        LayoutBuilder builder;
        (void)plan_linear_attention_state_pool(builder, {.layers         = layers,
                                                         .conv_channels  = kConvChannels,
                                                         .conv_width     = 3,
                                                         .value_heads    = kValueHeads,
                                                         .value_head_dim = kStateDim,
                                                         .key_head_dim   = kStateDim,
                                                         .slot_count     = slot_count,
                                                         .conv_dtype     = DType::BF16});
        return builder.finish(256);
    }
};

int run_case(const Case& test_case, const DevicePackedWeight& parent, Report& report) {
    const std::int32_t W     = test_case.width;
    const std::int32_t valid = test_case.valid;
    const std::uint32_t seed = test_case.seed;
    const std::string tag    = "W=" + std::to_string(W) + " valid=" + std::to_string(valid) +
                               " " + policy_name(test_case.policy);

    // Host inputs (represented values; both paths consume these exactly).
    std::vector<std::uint16_t> x_bits(static_cast<std::size_t>(kHidden) * W);
    for (std::size_t i = 0; i < x_bits.size(); ++i) {
        x_bits[i] = bf16_pattern(seed + 1000003U + static_cast<std::uint32_t>(i), -0.02F, 0.02F);
    }
    std::vector<std::uint16_t> conv_weight_bits(static_cast<std::size_t>(kConvChannels) * 4);
    for (std::size_t i = 0; i < conv_weight_bits.size(); ++i) {
        conv_weight_bits[i] =
            bf16_pattern(seed + 2000003U + static_cast<std::uint32_t>(i), -0.03F, 0.03F);
    }
    std::vector<float> g_values(static_cast<std::size_t>(kValueHeads) * W);
    std::vector<float> beta_values(g_values.size());
    for (std::size_t i = 0; i < g_values.size(); ++i) {
        g_values[i]    = f32_pattern(seed + 3000007U + static_cast<std::uint32_t>(i), -0.35F,
                                     -0.01F);
        beta_values[i] = f32_pattern(seed + 4000009U + static_cast<std::uint32_t>(i), 0.05F,
                                     0.95F);
    }
    std::vector<std::uint16_t> initial_conv(static_cast<std::size_t>(kConvChannels) * 3);
    for (std::size_t i = 0; i < initial_conv.size(); ++i) {
        initial_conv[i] = bf16_pattern(seed + 5000011U + static_cast<std::uint32_t>(i), -0.05F,
                                       0.05F);
    }
    const std::size_t recurrent_elements =
        static_cast<std::size_t>(kStateDim) * kStateDim * kValueHeads;
    std::vector<float> initial_recurrent(recurrent_elements);
    for (std::size_t i = 0; i < initial_recurrent.size(); ++i) {
        initial_recurrent[i] =
            f32_pattern(seed + 6000013U + static_cast<std::uint32_t>(i), -0.02F, 0.02F);
    }

    DeviceBuffer x_device(x_bits.size() * 2);
    x_device.copy_from_host(x_bits.data(), x_bits.size() * 2);
    DeviceBuffer conv_weight_device(conv_weight_bits.size() * 2);
    conv_weight_device.copy_from_host(conv_weight_bits.data(), conv_weight_bits.size() * 2);
    DeviceBuffer g_device(g_values.size() * 4);
    g_device.copy_from_host(g_values.data(), g_values.size() * 4);
    DeviceBuffer beta_device(beta_values.size() * 4);
    beta_device.copy_from_host(beta_values.data(), beta_values.size() * 4);
    DeviceBuffer valid_device(sizeof(std::int32_t));
    const std::int32_t valid_value = valid;
    valid_device.copy_from_host(&valid_value, sizeof(valid_value));
    DeviceBuffer initial_slot_device(sizeof(std::int32_t));
    const std::int32_t initial_slot = 0;
    initial_slot_device.copy_from_host(&initial_slot, sizeof(initial_slot));
    DeviceBuffer step_initial_device(sizeof(std::int32_t));
    DeviceBuffer base_slot_device(sizeof(std::int32_t));

    const Tensor conv_weight(conv_weight_device.p, DType::BF16, {kConvChannels, 4});
    const Tensor valid_tensor(valid_device.p, DType::I32, {1});
    const Tensor initial_selector(initial_slot_device.p, DType::I32, {1});

    // ---------------- Path A: width repeated width-1 in-place snapshot steps. ----------------
    Pool state_a(1, W + 1, initial_conv, initial_recurrent);
    const std::size_t snapshot_ws_bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        QType::NVFP4, kParentRows, kHidden, test_case.policy, 1, 1, 1);
    WorkspaceArena snapshot_workspace(std::max<std::size_t>(256, snapshot_ws_bytes));

    const std::size_t q_elements = static_cast<std::size_t>(kQueryRows);
    const std::size_t v_elements = static_cast<std::size_t>(kValueRows);
    DeviceBuffer a_q(q_elements * 2);
    DeviceBuffer a_k(q_elements * 2);
    DeviceBuffer a_v(v_elements * 2);
    DeviceBuffer a_z(v_elements * 2);
    DeviceBuffer a_out(v_elements * 2);
    std::vector<std::vector<std::uint16_t>> a_q_per(W);
    std::vector<std::vector<std::uint16_t>> a_k_per(W);
    std::vector<std::vector<std::uint16_t>> a_v_per(W);
    std::vector<std::vector<std::uint16_t>> a_z_per(W);
    std::vector<std::vector<std::uint16_t>> a_out_per(W);
    std::vector<std::vector<std::uint16_t>> a_conv(W + 1);
    std::vector<std::vector<float>> a_recurrent(W + 1);

    for (std::int32_t t = 0; t < W; ++t) {
        Tensor x_t(static_cast<std::uint16_t*>(x_device.p) +
                       static_cast<std::size_t>(t) * kHidden,
                   DType::BF16, {kHidden, 1, 1});
        Tensor q_t_conv(a_q.p, DType::BF16, {kQueryRows, 1, 1});
        Tensor k_t_conv(a_k.p, DType::BF16, {kQueryRows, 1, 1});
        Tensor v_t_conv(a_v.p, DType::BF16, {kValueRows, 1, 1});
        Tensor z_t(a_z.p, DType::BF16, {kZRows, 1, 1});
        Tensor q_t(a_q.p, DType::BF16, {kStateDim, kQkHeads, 1, 1});
        Tensor k_t(a_k.p, DType::BF16, {kStateDim, kQkHeads, 1, 1});
        Tensor v_t(a_v.p, DType::BF16, {kStateDim, kValueHeads, 1, 1});
        Tensor out_t(a_out.p, DType::BF16, {kStateDim, kValueHeads, 1, 1});
        Tensor g_t(static_cast<float*>(g_device.p) + t * kValueHeads, DType::FP32,
                   {kValueHeads, 1, 1});
        Tensor beta_t(static_cast<float*>(beta_device.p) + t * kValueHeads, DType::FP32,
                      {kValueHeads, 1, 1});
        // Step t reads the history/state published by step t-1 (slot t) and publishes the new
        // one to slot t+1. Production k=0 chains this with initial == base per step (width 1);
        // distinct slots here only capture the per-step states.
        const std::int32_t step_initial  = t;
        const std::int32_t snapshot_base = t + 1;
        step_initial_device.copy_from_host(&step_initial, sizeof(step_initial));
        base_slot_device.copy_from_host(&snapshot_base, sizeof(snapshot_base));
        const Tensor initial_selector_t(step_initial_device.p, DType::I32, {1});
        const Tensor base_selector(base_slot_device.p, DType::I32, {1});

        ops::gdn_input_proj_conv_snapshot(x_t, parent.view(), conv_weight, state_a.pool.conv[0],
                                          Tensor{}, initial_selector_t, base_selector, q_t_conv,
                                          k_t_conv, v_t_conv, z_t, test_case.policy,
                                          snapshot_workspace, nullptr);
        ops::gated_delta_net_snapshot(q_t, k_t, v_t, g_t, beta_t, kScale, true,
                                      state_a.pool.recurrent[0], Tensor{}, initial_selector_t,
                                      base_selector, out_t, nullptr);
        cuda_synchronize();

        a_q_per[static_cast<std::size_t>(t)] =
            from_device<std::uint16_t>(a_q, q_elements);
        a_k_per[static_cast<std::size_t>(t)] = from_device<std::uint16_t>(a_k, q_elements);
        a_v_per[static_cast<std::size_t>(t)] = from_device<std::uint16_t>(a_v, v_elements);
        a_z_per[static_cast<std::size_t>(t)] = from_device<std::uint16_t>(a_z, v_elements);
        a_out_per[static_cast<std::size_t>(t)] =
            from_device<std::uint16_t>(a_out, v_elements);
        // State after c = t+1 columns lives in slot c.
        a_conv[static_cast<std::size_t>(t + 1)] = from_device<std::uint16_t>(
            state_a.pool.conv_slot(0, static_cast<std::int32_t>(snapshot_base)).data,
            static_cast<std::size_t>(kConvChannels) * 3);
        a_recurrent[static_cast<std::size_t>(t + 1)] = from_device<float>(
            state_a.pool.recurrent_slot(0, static_cast<std::uint32_t>(snapshot_base)).data,
            recurrent_elements);
    }

    // ---------------- Path B: width-W masked record round + fold per commit. ----------------
    Pool state_b(1, 1, initial_conv, initial_recurrent);

    LayoutBuilder record_builder;
    const GdnReplayRecordLayout record_layout = plan_gdn_replay_records(
        record_builder, {.layers          = kGdnLayers,
                         .record_capacity = 1,
                         .width           = W,
                         .conv_channels   = kConvChannels,
                         .qk_heads        = kQkHeads,
                         .value_heads     = kValueHeads,
                         .key_dim         = kStateDim,
                         .value_dim       = kStateDim});
    DeviceBuffer record_storage(record_builder.finish(256));
    record_storage.fill(0xcc);
    const GdnReplayRecords records({record_storage.p, record_storage.bytes}, record_layout);
    GdnReplayRecordLayer layer_records = records.layer(0, 1);

    const std::size_t record_ws_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
        QType::NVFP4, kParentRows, kHidden, test_case.policy, 1, W, W);
    WorkspaceArena record_workspace(std::max<std::size_t>(256, record_ws_bytes));

    DeviceBuffer b_q(static_cast<std::size_t>(kQueryRows) * W * 2);
    DeviceBuffer b_k(static_cast<std::size_t>(kQueryRows) * W * 2);
    DeviceBuffer b_v(static_cast<std::size_t>(kValueRows) * W * 2);
    DeviceBuffer b_z(static_cast<std::size_t>(kZRows) * W * 2);
    DeviceBuffer b_out(static_cast<std::size_t>(kValueRows) * W * 2);

    Tensor x_all(x_device.p, DType::BF16, {kHidden, W, 1});
    Tensor q_all_conv(b_q.p, DType::BF16, {kQueryRows, W, 1});
    Tensor k_all_conv(b_k.p, DType::BF16, {kQueryRows, W, 1});
    Tensor v_all_conv(b_v.p, DType::BF16, {kValueRows, W, 1});
    Tensor z_all(b_z.p, DType::BF16, {kZRows, W, 1});
    Tensor q_all(b_q.p, DType::BF16, {kStateDim, kQkHeads, W, 1});
    Tensor k_all(b_k.p, DType::BF16, {kStateDim, kQkHeads, W, 1});
    Tensor v_all(b_v.p, DType::BF16, {kStateDim, kValueHeads, W, 1});
    Tensor out_all(b_out.p, DType::BF16, {kStateDim, kValueHeads, W, 1});
    Tensor g_all(g_device.p, DType::FP32, {kValueHeads, W, 1});
    Tensor beta_all(beta_device.p, DType::FP32, {kValueHeads, W, 1});

    ops::gdn_input_proj_conv_record(x_all, parent.view(), conv_weight, state_b.pool.conv[0],
                                    valid_tensor, initial_selector, layer_records.conv,
                                    q_all_conv, k_all_conv, v_all_conv, z_all, test_case.policy,
                                    record_workspace, nullptr);
    ops::gated_delta_net_replay_record(q_all, k_all, v_all, g_all, beta_all, kScale,
                                       state_b.pool.recurrent[0], valid_tensor,
                                       initial_selector, layer_records.key, layer_records.value,
                                       layer_records.gate, out_all, nullptr);
    cuda_synchronize();

    const std::vector<std::uint16_t> b_q_flat =
        from_device<std::uint16_t>(b_q, q_elements * W);
    const std::vector<std::uint16_t> b_k_flat =
        from_device<std::uint16_t>(b_k, q_elements * W);
    const std::vector<std::uint16_t> b_v_flat =
        from_device<std::uint16_t>(b_v, v_elements * W);
    const std::vector<std::uint16_t> b_z_flat =
        from_device<std::uint16_t>(b_z, v_elements * W);
    const std::vector<std::uint16_t> b_out_flat =
        from_device<std::uint16_t>(b_out, v_elements * W);

    const auto column_view = [](const std::vector<std::uint16_t>& flat, std::size_t elements,
                                std::int32_t W_local, std::int32_t t) {
        return std::vector<std::uint16_t>(
            flat.begin() + static_cast<std::size_t>(t) * elements,
            flat.begin() + static_cast<std::size_t>(t + 1) * elements);
    };

    for (std::int32_t t = 0; t < valid; ++t) {
        report.compare_bits("parity(" + tag + ") query", t, a_q_per[static_cast<std::size_t>(t)],
                            column_view(b_q_flat, q_elements, W, t));
        report.compare_bits("parity(" + tag + ") key", t, a_k_per[static_cast<std::size_t>(t)],
                            column_view(b_k_flat, q_elements, W, t));
        report.compare_bits("parity(" + tag + ") value", t,
                            a_v_per[static_cast<std::size_t>(t)],
                            column_view(b_v_flat, v_elements, W, t));
        report.compare_bits("parity(" + tag + ") z", t, a_z_per[static_cast<std::size_t>(t)],
                            column_view(b_z_flat, v_elements, W, t));
        report.compare_bits("parity(" + tag + ") recurrent output", t,
                            a_out_per[static_cast<std::size_t>(t)],
                            column_view(b_out_flat, v_elements, W, t));
    }

    // Per-commit state parity: fold the record prefix onto a fresh initial-state pool and
    // compare against the Path A state after the same number of width-1 steps. The fold op
    // requires the registered all-layer geometry, so the record plane and fold pool span all
    // 48 GDN layers; only layer 0 carries driven records and is inspected.
    const std::vector<std::uint8_t> records_before =
        from_device<std::uint8_t>(record_storage, record_storage.bytes);
    for (std::int32_t commit = 1; commit <= valid; ++commit) {
        Pool state_c(kGdnLayers, 1, initial_conv, initial_recurrent);
        const std::array fold_row{ops::GdnReplayFoldRow{initial_slot, commit}};
        ops::gdn_replay_fold(records, state_c.pool.all_layers_view(), fold_row, nullptr);
        cuda_synchronize();

        const std::vector<float> c_recurrent = from_device<float>(
            state_c.pool.recurrent_slot(0, 0).data, recurrent_elements);
        const std::vector<std::uint16_t> c_conv =
            from_device<std::uint16_t>(state_c.pool.conv_slot(0, 0).data,
                                       static_cast<std::size_t>(kConvChannels) * 3);
        report.compare_f32("parity(" + tag + ") recurrent state", commit - 1,
                           a_recurrent[static_cast<std::size_t>(commit)], c_recurrent);
        report.compare_bits("parity(" + tag + ") conv state", commit - 1,
                            a_conv[static_cast<std::size_t>(commit)], c_conv);
    }

    const std::vector<std::uint8_t> records_after =
        from_device<std::uint8_t>(record_storage, record_storage.bytes);
    if (records_before != records_after) {
        std::cerr << "parity(" << tag << "): fold modified record storage\n";
        ++report.failures;
    }
    return 0;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    DevicePackedWeight parent(quantized_weight::make_patterned_weight(
        QType::NVFP4, kParentRows, kHidden, 7001U, options));

    Report report;
    const std::vector<Case> cases{
        {2, 2, ops::LinearPolicy::AllowA4, 7011U},
        {3, 3, ops::LinearPolicy::AllowA4, 7021U},
        {4, 4, ops::LinearPolicy::AllowA4, 7031U},
        {4, 2, ops::LinearPolicy::AllowA4, 7041U},
        {3, 1, ops::LinearPolicy::AllowA4, 7051U},
        {4, 4, ops::LinearPolicy::A16Only, 7061U},
    };
    for (const Case& test_case : cases) { run_case(test_case, parent, report); }
    report.failures += parent.verify_preserved("parity NVFP4 parent weight");

    std::cout << (report.failures == 0 ? "OK" : "FAIL") << " gdn decode/record parity\n";
    return report.failures == 0 ? 0 : 1;
}