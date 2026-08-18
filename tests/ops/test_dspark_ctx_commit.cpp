// dspark_ctx_commit (DSpark draft-context commit) correctness suite.
//
// Oracle: independent naive FP64 evaluation of the full logical formula from the
// represented public inputs (BF16-decoded weights/activations, FP32 inv_freq table,
// FP32 attention factor): fc GEMM -> hidden RMSNorm (eps 1e-6, gain applied, no offset)
// -> per-layer kv GEMM -> per-head k_norm RMSNorm -> split-half YaRN rotation with
// FP64 cos/sin of the FP32-table angle -> single BF16 rounding for K; V is the
// bit-exact passthrough of the kernel's own BF16 kv rows (verified numerically
// against the oracle under the kv materialization profile, see kCommitV below).
// Covers the MTP verify token range (T = 1..8 across all decode/small-T routes) and
// prefill-class T (32: small-T; 128: MMA full-token), position bases 0 / 137 / 262136
// (full YaRN angle range), a YaRN table bit-check against the published transformers
// 5.12.1 reference, monotone-append arena semantics, and contract validation.

#include "ninfer/ops/dspark_ctx_commit.h"

#include "ops/op_tester.h"

#include "core/device.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::test;
namespace ds = ninfer::ops::dspark;

namespace {

constexpr std::int32_t kCap       = ds::kLayers;
constexpr std::int32_t kHidden    = ds::kHidden;
constexpr std::int32_t kTapWidth  = ds::kTapWidth;
constexpr std::int32_t kKvRows    = ds::kKvRows;
constexpr std::int32_t kHeadDim   = ds::kHeadDim;
constexpr std::int32_t kHeadDimH  = ds::kHeadDimHalf;
constexpr std::int32_t kKvOffset  = ds::kKvBlockOffset;
constexpr std::int32_t kTok       = ds::kArenaElementsPerToken;
constexpr std::int32_t kTokLayer  = ds::kArenaElementsPerLayer;
constexpr int kHeads              = ds::kKvHeadCount;

// Suite criteria for the fused commit's BF16 K/V outputs. The GEMM/norm/rotation math
// accumulates in FP32, so the implementation profile is the BF16 materialization of the
// pipeline intermediates (x, normalized x, per-layer kv) plus the final K rounding:
// ~2^-8 relative per stage at 1 sigma. Measured against this oracle (seeds 0x1001-0x1900,
// T=1..128, RTX 5090): max |err| = 0.041 (K) and 0.031 (V). The criteria carry ~2x margin.
// V is the bit-exact passthrough of the kernel's own BF16 kv rows; against the FP64
// oracle it inherits the same kv materialization error, so it is verified numerically.
constexpr PointwiseCriterion kCommitK{8e-2, 4e-2};
constexpr PointwiseCriterion kCommitV{6e-2, 4e-2};

// transformers 5.12.1 Qwen3RotaryEmbedding inv_freq, RadixArk/Qwen3.8-27B-DSpark config
// (theta 1e7, original_max 8192, factor 32, beta_fast 32, beta_slow 1, head_dim 128),
// generated 2026-08-18 with /venv/main (CPU float32); YaRN correction i in [14, 29].
const std::uint32_t kReferenceInvFreq[64] = {
    0x3f800000, 0x3f470165, 0x3f1ab32b, 0x3ef0843c, 0x3ebaf81b, 0x3e9157e1, 0x3e61f835, 0x3e2fa92d,
    0x3e088d77, 0x3dd44d6c, 0x3da50956, 0x3d804b29, 0x3d47763f, 0x3d1b0e01, 0x3cf11177, 0x3caf4b93,
    0x3c7db85d, 0x3c369b18, 0x3c0294a3, 0x3bb956ce, 0x3b825561, 0x3b35446d, 0x3af89a68, 0x3aa76eb8,
    0x3a5c2a27, 0x3a0bf05b, 0x39a90b88, 0x393b6168, 0x38ae09fa, 0x37b077b3, 0x37892e02, 0x37554706,
    0x3725cb60, 0x3700e1fe, 0x36c860c1, 0x369bc44e, 0x36722ce2, 0x363c4236, 0x3612587e, 0x35e3872c,
    0x35b0df52, 0x35897e8f, 0x3555c441, 0x35262cb9, 0x35012dab, 0x34c8d669, 0x349c1fc4, 0x3472bb16,
    0x343cb0c1, 0x3412ae6d, 0x33e40cc6, 0x33b1472c, 0x3389cf4a, 0x335641c6, 0x33268e4c, 0x33017985,
    0x32c94c57, 0x329c7b70, 0x3273499d, 0x323d1f8c, 0x3213048e, 0x31e492ae, 0x31b1af44, 0x318a2036};

Weight bf16_weight(void* data, std::int32_t rows, std::int32_t hidden) {
    Weight weight{};
    weight.qtype           = QType::BF16_CTRL;
    weight.layout          = QuantLayout::Contiguous;
    weight.payload         = data;
    weight.payload_bytes   = static_cast<std::uint64_t>(rows) * hidden * sizeof(std::uint16_t);
    weight.qdata           = data;
    weight.ndim            = 2;
    weight.shape[0]        = rows;
    weight.shape[1]        = hidden;
    weight.padded_shape[0] = rows;
    weight.padded_shape[1] = hidden;
    weight.n               = rows;
    weight.k               = hidden;
    return weight;
}

// Naive FP64 oracle for one commit call. Weights are row-major [rows, cols] over the
// BF16 values; taps are [kTapWidth, T] dim0-fastest. Returns per-token K and V in
// arena order (layer, head, dim) for every committed position.
struct CommitReference {
    std::vector<double> k_ref; // [T, kCap, kHeads, kHeadDim]
    std::vector<double> v_ref; // [T, kCap, kHeads, kHeadDim]
};

CommitReference commit_reference(const std::vector<float>& taps, const std::vector<int>& positions,
                                 const std::vector<float>& fc, const std::vector<float>& hidden_norm,
                                 const std::vector<std::vector<float>>& kv_w, const std::vector<float>& k_norm,
                                 const std::vector<float>& inv_freq, float attention_scaling) {
    const int T = static_cast<int>(positions.size());
    CommitReference ref;
    ref.k_ref.assign(static_cast<std::size_t>(T) * kCap * kHeads * kHeadDim, 0.0);
    ref.v_ref = ref.k_ref;

    // x2[t][n] = hidden_norm[n] * (fc_row_n . taps_col_t) / sqrt(mean(fc . taps)^2 + 1e-6)
    // Layout: dim0 fastest = each token's K-block is contiguous (taps[t * K + k]).
    std::vector<double> x2(static_cast<std::size_t>(T) * kHidden);
    for (int t = 0; t < T; ++t) {
        double sq = 0.0;
        const float* taps_t = taps.data() + static_cast<std::size_t>(t) * kTapWidth;
        for (int n = 0; n < kHidden; ++n) {
            double acc = 0.0;
            const float* w = fc.data() + static_cast<std::size_t>(n) * kTapWidth;
            for (int kk = 0; kk < kTapWidth; ++kk) {
                acc += static_cast<double>(w[kk]) * static_cast<double>(taps_t[kk]);
            }
            x2[static_cast<std::size_t>(t) * kHidden + n] = acc;
            sq += acc * acc;
        }
        const double inv = 1.0 / std::sqrt(sq / kHidden + 1e-6);
        for (int n = 0; n < kHidden; ++n) {
            x2[static_cast<std::size_t>(t) * kHidden + n] =
                x2[static_cast<std::size_t>(t) * kHidden + n] * static_cast<double>(hidden_norm[n]) * inv;
        }
    }

    const double a = static_cast<double>(attention_scaling);
    for (int l = 0; l < kCap; ++l) {
        const std::vector<float>& w = kv_w[static_cast<std::size_t>(l)];
        for (int t = 0; t < T; ++t) {
            std::vector<double> kv(kKvRows);
            for (int r = 0; r < kKvRows; ++r) {
                double acc = 0.0;
                const float* row = w.data() + static_cast<std::size_t>(r) * kHidden;
                const double* xt = x2.data() + static_cast<std::size_t>(t) * kHidden;
                for (int kk = 0; kk < kHidden; ++kk) {
                    acc += static_cast<double>(row[kk]) * xt[kk];
                }
                kv[static_cast<std::size_t>(r)] = acc;
            }
            for (int h = 0; h < kHeads; ++h) {
                double sq = 0.0;
                double khat[kHeadDim];
                for (int i = 0; i < kHeadDim; ++i) {
                    khat[i] = kv[static_cast<std::size_t>(h * kHeadDim + i)];
                    sq += khat[i] * khat[i];
                }
                const double inv = 1.0 / std::sqrt(sq / kHeadDim + 1e-6);
                const double pos = positions[t];
                for (int j = 0; j < kHeadDimH; ++j) {
                    const double x0 = khat[j] * static_cast<double>(k_norm[j]) * inv;
                    const double x1 = khat[j + kHeadDimH] * static_cast<double>(k_norm[j + kHeadDimH]) * inv;
                    const double th = static_cast<double>(inv_freq[j]) * pos;
                    const double c  = a * std::cos(th);
                    const double s  = a * std::sin(th);
                    const std::size_t o =
                        (static_cast<std::size_t>(t) * kCap + l) * kHeads * kHeadDim + h * kHeadDim + j;
                    ref.k_ref[o]             = x0 * c - x1 * s;
                    ref.k_ref[o + kHeadDimH] = x1 * c + x0 * s;
                    ref.v_ref[o]             = kv[static_cast<std::size_t>(kKvOffset + h * kHeadDim + j)];
                    ref.v_ref[o + kHeadDimH] = kv[static_cast<std::size_t>(kKvOffset + h * kHeadDim + j + kHeadDimH)];
                }
            }
        }
    }
    return ref;
}

// Reads the committed K and V rows for position p, layer l from the device arena into
// doubles, in (head, dim) order.
void read_arena_row(const DeviceBuffer& arena, int p, int l, std::vector<double>& k_row,
                    std::vector<double>& v_row) {
    const std::size_t k_elem =
        static_cast<std::size_t>(p) * kTok + static_cast<std::size_t>(l) * kTokLayer;
    const std::uint16_t* base = static_cast<const std::uint16_t*>(arena.p);
    k_row = from_device_bf16(base + k_elem, static_cast<std::size_t>(kHeads) * kHeadDim);
    v_row = from_device_bf16(base + k_elem + kKvOffset, static_cast<std::size_t>(kHeads) * kHeadDim);
}

// BF16 sentinel 0x3C3C (about 1.47e-20, finite): both bytes are 0x3C, so one
// single-byte cudaMemset tiles it. Verification samples the row right after the
// committed interval plus the far end (and every row when the margin is small),
// strong because the kernel addresses whole position rows.
constexpr std::uint16_t kSentinel = 0x3c3c;

void fill_sentinel(DeviceBuffer& arena) {
    cuda_check(cudaMemset(arena.p, 0x3c, arena.bytes), "sentinel fill");
}

int check_sentinel_row(const DeviceBuffer& arena, int p) {
    std::vector<std::uint16_t> got = from_device<std::uint16_t>(
        static_cast<const std::uint16_t*>(arena.p) + static_cast<std::size_t>(p) * kTok, kTok);
    for (std::uint16_t v : got) {
        if (v != kSentinel) {
            std::cerr << "arena row p=" << p << " was rewritten outside its commit\n";
            return 1;
        }
    }
    return 0;
}

int check_sentinel_margin(const DeviceBuffer& arena, int first_free, int cap) {
    int failures = 0;
    if (cap - first_free <= 32) {
        for (int p = first_free; p < cap; ++p) failures += check_sentinel_row(arena, p);
    } else {
        failures += check_sentinel_row(arena, first_free);
        failures += check_sentinel_row(arena, first_free + (cap - first_free) / 2);
        failures += check_sentinel_row(arena, cap - 1);
    }
    return failures;
}

struct CaseData {
    std::int32_t T;
    int base;
    std::uint32_t seed;
};

int run_commit_case(cudaStream_t stream, const CaseData& c) {
    const int T   = c.T;
    const int cap = c.base + T + 8; // committed tail plus untouched margin
    char label[128];
    std::snprintf(label, sizeof(label), "dspark_ctx_commit T=%d base=%d", T, c.base);

    // Public inputs.
    std::vector<float> taps(static_cast<std::size_t>(kTapWidth) * T);
    fill_uniform(taps, c.seed, -8.0f, 8.0f);
    round_to_bf16(taps);
    std::vector<float> fc(static_cast<std::size_t>(kHidden) * kTapWidth);
    fill_uniform(fc, c.seed + 1, -0.04f, 0.04f);
    round_to_bf16(fc);
    std::vector<float> hidden_norm(kHidden);
    fill_uniform(hidden_norm, c.seed + 2, 0.5f, 1.5f);
    round_to_bf16(hidden_norm);
    std::vector<std::vector<float>> kv_w(kCap);
    for (int l = 0; l < kCap; ++l) {
        kv_w[static_cast<std::size_t>(l)].resize(static_cast<std::size_t>(kKvRows) * kHidden);
        fill_uniform(kv_w[static_cast<std::size_t>(l)], c.seed + 3 + static_cast<std::uint32_t>(l), -0.04f, 0.04f);
        round_to_bf16(kv_w[static_cast<std::size_t>(l)]);
    }
    std::vector<float> k_norm(kHeadDim);
    fill_uniform(k_norm, c.seed + 8, 0.5f, 1.5f);
    round_to_bf16(k_norm);
    std::vector<float> inv_freq(64);
    dspark_yarn_inverse_frequencies(1e7f, 8192, 32.0f, 32.0f, 1.0f, 128, inv_freq);
    const float attention_scaling = dspark_yarn_attention_scaling(32.0f);
    std::vector<int> positions(T);
    for (int t = 0; t < T; ++t) positions[static_cast<std::size_t>(t)] = c.base + t;

    // Device buffers.
    DeviceBuffer d_taps = to_device_bf16(taps);
    DeviceBuffer d_positions = to_device_i32(positions);
    DeviceBuffer d_fc = to_device_bf16(fc);
    DeviceBuffer d_hidden_norm = to_device_bf16(hidden_norm);
    std::vector<DeviceBuffer> d_kv_w(kCap);
    for (int l = 0; l < kCap; ++l) d_kv_w[static_cast<std::size_t>(l)] = to_device_bf16(kv_w[static_cast<std::size_t>(l)]);
    DeviceBuffer d_k_norm = to_device_bf16(k_norm);
    DeviceBuffer d_inv_freq = to_device_f32(inv_freq);
    DeviceBuffer d_arena(static_cast<std::size_t>(cap) * kTok * 2);
    fill_sentinel(d_arena);

    Tensor taps_t(d_taps.p, DType::BF16, {kTapWidth, T});
    Tensor positions_t(d_positions.p, DType::I32, {T});
    Tensor hidden_norm_t(d_hidden_norm.p, DType::BF16, {kHidden});
    Tensor k_norm_t(d_k_norm.p, DType::BF16, {kHeadDim});
    Tensor inv_freq_t(d_inv_freq.p, DType::FP32, {kHeadDimH});
    Tensor arena_t(d_arena.p, DType::BF16, {kTok, cap});
    std::vector<Weight> kv_weights(kCap);
    for (int l = 0; l < kCap; ++l) kv_weights[static_cast<std::size_t>(l)] = bf16_weight(d_kv_w[static_cast<std::size_t>(l)].p, kKvRows, kHidden);
    Weight fc_w = bf16_weight(d_fc.p, kHidden, kTapWidth);

    DeviceArena workspace(dspark_ctx_commit_workspace_capacity_bytes(1, T));
    // Input prep (H2D copies, sentinel fill) runs on the legacy default stream, while the
    // commit kernels run on the non-blocking `stream`; there is no implicit ordering between
    // the two, so make every input visible before the first op kernel is enqueued.
    cuda_synchronize();
    dspark_ctx_commit(taps_t, positions_t, c.base + T, fc_w, hidden_norm_t, kv_weights, k_norm_t, inv_freq_t,
                      attention_scaling, arena_t, workspace, stream);
    cuda_synchronize(stream);

    int failures = 0;
    const CommitReference ref =
        commit_reference(taps, positions, fc, hidden_norm, kv_w, k_norm, inv_freq, attention_scaling);
    for (int t = 0; t < T; ++t) {
        const int p = positions[static_cast<std::size_t>(t)];
        for (int l = 0; l < kCap; ++l) {
            std::vector<double> k_row, v_row;
            read_arena_row(d_arena, p, l, k_row, v_row);
            const std::size_t o =
                (static_cast<std::size_t>(t) * kCap + l) * kHeads * kHeadDim;
            std::vector<double> k_ref(ref.k_ref.begin() + o, ref.k_ref.begin() + o + kHeads * kHeadDim);
            std::vector<double> v_ref(ref.v_ref.begin() + o, ref.v_ref.begin() + o + kHeads * kHeadDim);
            char case_label[160];
            std::snprintf(case_label, sizeof(case_label), "%s l=%d t=%d K", label, l, t);
            failures += verify_pointwise(case_label, k_row, k_ref, kCommitK);
            // V against the FP64 oracle: same kv materialization profile as K.
            std::vector<double> v_ref_d(v_ref.begin(), v_ref.end());
            std::snprintf(case_label, sizeof(case_label), "%s l=%d t=%d V", label, l, t);
            failures += verify_pointwise(case_label, v_row, v_ref_d, kCommitV);
        }
    }

    // Untouched arena rows stay sentinel bit-exactly (monotone append, no rewrite).
    failures += check_sentinel_margin(d_arena, c.base + T, cap);
    return failures;
}

// Monotone append on one arena: commit [0, 64) with one tap set, then [64, 72) with a
// different tap set; the earlier region must stay bit-exact while the new tail matches
// the oracle and the margin stays sentinel.
int run_append_check(cudaStream_t stream) {
    char label[] = "dspark_ctx_commit append";
    const int cap = 72 + 8;
    auto build = [&](int T, int base, std::uint32_t seed) {
        struct In {
            std::vector<float> taps, fc, hidden_norm, k_norm, inv_freq;
            std::vector<std::vector<float>> kv_w;
            std::vector<int> positions;
            float a;
            DeviceBuffer d_taps, d_positions, d_fc, d_hidden_norm, d_k_norm, d_inv_freq;
            std::vector<DeviceBuffer> d_kv_w;
        } in;
        in.taps.resize(static_cast<std::size_t>(kTapWidth) * T);
        fill_uniform(in.taps, seed, -8.0f, 8.0f);
        round_to_bf16(in.taps);
        in.fc.resize(static_cast<std::size_t>(kHidden) * kTapWidth);
        fill_uniform(in.fc, seed + 1, -0.04f, 0.04f);
        round_to_bf16(in.fc);
        in.hidden_norm.resize(kHidden);
        fill_uniform(in.hidden_norm, seed + 2, 0.5f, 1.5f);
        round_to_bf16(in.hidden_norm);
        in.kv_w.resize(kCap);
        for (int l = 0; l < kCap; ++l) {
            in.kv_w[static_cast<std::size_t>(l)].resize(static_cast<std::size_t>(kKvRows) * kHidden);
            fill_uniform(in.kv_w[static_cast<std::size_t>(l)], seed + 3 + static_cast<std::uint32_t>(l), -0.04f,
                         0.04f);
            round_to_bf16(in.kv_w[static_cast<std::size_t>(l)]);
        }
        in.k_norm.resize(kHeadDim);
        fill_uniform(in.k_norm, seed + 8, 0.5f, 1.5f);
        round_to_bf16(in.k_norm);
        in.inv_freq.resize(kHeadDimH);
        dspark_yarn_inverse_frequencies(1e7f, 8192, 32.0f, 32.0f, 1.0f, 128, in.inv_freq);
        in.a = dspark_yarn_attention_scaling(32.0f);
        for (int t = 0; t < T; ++t) in.positions.push_back(base + t);
        in.d_taps = to_device_bf16(in.taps);
        in.d_positions = to_device_i32(in.positions);
        in.d_fc = to_device_bf16(in.fc);
        in.d_hidden_norm = to_device_bf16(in.hidden_norm);
        in.d_kv_w.resize(kCap);
        for (int l = 0; l < kCap; ++l) in.d_kv_w[static_cast<std::size_t>(l)] = to_device_bf16(in.kv_w[static_cast<std::size_t>(l)]);
        in.d_k_norm = to_device_bf16(in.k_norm);
        in.d_inv_freq = to_device_f32(in.inv_freq);
        return in;
    };
    auto commit = [&](const auto& in, DeviceBuffer& d_arena, int T, int envelope) {
        Tensor taps_t(in.d_taps.p, DType::BF16, {kTapWidth, T});
        Tensor positions_t(in.d_positions.p, DType::I32, {T});
        Tensor hidden_norm_t(in.d_hidden_norm.p, DType::BF16, {kHidden});
        Tensor k_norm_t(in.d_k_norm.p, DType::BF16, {kHeadDim});
        Tensor inv_freq_t(in.d_inv_freq.p, DType::FP32, {kHeadDimH});
        Tensor arena_t(d_arena.p, DType::BF16, {kTok, cap});
        std::vector<Weight> kv_weights(kCap);
        for (int l = 0; l < kCap; ++l) kv_weights[static_cast<std::size_t>(l)] =
            bf16_weight(in.d_kv_w[static_cast<std::size_t>(l)].p, kKvRows, kHidden);
        Weight fc_w = bf16_weight(in.d_fc.p, kHidden, kTapWidth);
        DeviceArena workspace(dspark_ctx_commit_workspace_capacity_bytes(1, T));
        // The inputs' H2D copies (legacy stream) must be complete before the non-blocking
        // `stream` kernels run; no implicit ordering exists between the two.
        cuda_synchronize();
        dspark_ctx_commit(taps_t, positions_t, envelope, fc_w, hidden_norm_t, kv_weights, k_norm_t, inv_freq_t,
                          in.a, arena_t, workspace, stream);
        cuda_synchronize(stream);
    };

    int failures = 0;
    DeviceBuffer d_arena(static_cast<std::size_t>(cap) * kTok * 2);
    fill_sentinel(d_arena);
    auto first = build(64, 0, 0x1C00);
    commit(first, d_arena, 64, 64);
    // Snapshot the committed region [0, 64) before the second call.
    std::vector<std::uint16_t> snapshot = from_device<std::uint16_t>(d_arena.p, 64 * kTok);
    auto second = build(8, 64, 0x1D00);
    commit(second, d_arena, 8, 72);

    // Earlier region bit-exact.
    std::vector<std::uint16_t> after = from_device<std::uint16_t>(d_arena.p, 64 * kTok);
    if (snapshot != after) {
        std::cerr << label << ": earlier arena region rewritten by the append\n";
        ++failures;
    }
    // New tail against the oracle.
    const CommitReference ref =
        commit_reference(second.taps, second.positions, second.fc, second.hidden_norm, second.kv_w,
                         second.k_norm, second.inv_freq, second.a);
    for (int t = 0; t < 8; ++t) {
        for (int l = 0; l < kCap; ++l) {
            std::vector<double> k_row, v_row;
            read_arena_row(d_arena, 64 + t, l, k_row, v_row);
            const std::size_t o = (static_cast<std::size_t>(t) * kCap + l) * kHeads * kHeadDim;
            std::vector<double> k_ref(ref.k_ref.begin() + o, ref.k_ref.begin() + o + kHeads * kHeadDim);
            std::vector<double> v_ref(ref.v_ref.begin() + o, ref.v_ref.begin() + o + kHeads * kHeadDim);
            char cl[96];
            std::snprintf(cl, sizeof(cl), "%s t=%d l=%d K", label, t, l);
            failures += verify_pointwise(cl, k_row, k_ref, kCommitK);
            // V against the FP64 oracle: same kv materialization profile as K.
            std::vector<double> v_ref_d(v_ref.begin(), v_ref.end());
            std::snprintf(cl, sizeof(cl), "%s t=%d l=%d V", label, t, l);
            failures += verify_pointwise(cl, v_row, v_ref_d, kCommitV);
        }
    }
    failures += check_sentinel_margin(d_arena, 72, cap);
    return failures;
}

int run_yarn_table_check() {
    int failures = 0;
    std::vector<float> table(64);
    dspark_yarn_inverse_frequencies(1e7f, 8192, 32.0f, 32.0f, 1.0f, 128, table);
    for (int i = 0; i < 64; ++i) {
        std::uint32_t bits;
        std::memcpy(&bits, &table[static_cast<std::size_t>(i)], 4);
        if (bits != kReferenceInvFreq[static_cast<std::size_t>(i)]) {
            std::cerr << "yarn inv_freq[" << i << "] = 0x" << std::hex << bits << " ref 0x"
                      << kReferenceInvFreq[static_cast<std::size_t>(i)] << " 0x\n";
            ++failures;
        }
    }
    const float scaling = dspark_yarn_attention_scaling(32.0f);
    std::uint32_t scaling_bits;
    std::memcpy(&scaling_bits, &scaling, 4);
    if (scaling_bits != 0x3fac5c86u) {
        std::cerr << "yarn attention_scaling = 0x" << std::hex << scaling_bits << " ref 0x3fac5c86\n";
        ++failures;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " yarn table vs transformers 5.12.1\n";
    return failures;
}

int run_validation_check(cudaStream_t stream) {
    int failures = 0;
    const std::uint32_t seed = 0xB000;
    std::vector<float> taps(static_cast<std::size_t>(kTapWidth), 0.5f);
    round_to_bf16(taps);
    std::vector<float> fc(static_cast<std::size_t>(kHidden) * kTapWidth, 0.01f);
    round_to_bf16(fc);
    std::vector<float> norm(5120, 1.0f);
    std::vector<float> kv_one(static_cast<std::size_t>(kKvRows) * kHidden, 0.01f);
    round_to_bf16(kv_one);
    std::vector<float> k_norm(128, 1.0f);
    std::vector<float> inv_freq(64);
    dspark_yarn_inverse_frequencies(1e7f, 8192, 32.0f, 32.0f, 1.0f, 128, inv_freq);
    std::vector<int> positions{0};
    DeviceBuffer d_taps = to_device_bf16(taps);
    DeviceBuffer d_positions = to_device_i32(positions);
    DeviceBuffer d_fc = to_device_bf16(fc);
    DeviceBuffer d_norm = to_device_bf16(norm);
    DeviceBuffer d_kv = to_device_bf16(kv_one);
    DeviceBuffer d_k_norm = to_device_bf16(k_norm);
    DeviceBuffer d_inv_freq = to_device_f32(inv_freq);
    DeviceBuffer d_arena(64 * kTok * 2);
    Tensor taps_t(d_taps.p, DType::BF16, {kTapWidth, 1});
    Tensor positions_t(d_positions.p, DType::I32, {1});
    Tensor norm_t(d_norm.p, DType::BF16, {kHidden});
    Tensor k_norm_t(d_k_norm.p, DType::BF16, {kHeadDim});
    Tensor inv_freq_t(d_inv_freq.p, DType::FP32, {kHeadDimH});
    Tensor arena_t(d_arena.p, DType::BF16, {kTok, 64});
    Weight fc_w = bf16_weight(d_fc.p, kHidden, kTapWidth);
    Weight kv_w = bf16_weight(d_kv.p, kKvRows, kHidden);
    std::vector<Weight> kv_weights(1, kv_w);
    DeviceArena workspace(dspark_ctx_commit_workspace_capacity_bytes(1, 1));
    const float a = dspark_yarn_attention_scaling(32.0f);

    auto expect_throw = [&](const char* what, auto&& call) {
        try {
            call();
        } catch (const std::invalid_argument&) {
            return;
        }
        std::cerr << "dspark_ctx_commit validation: no throw for " << what << "\n";
        ++failures;
    };

    expect_throw("too few kv weights", [&] {
        dspark_ctx_commit(taps_t, positions_t, 1, fc_w, norm_t, kv_weights, k_norm_t, inv_freq_t, a, arena_t,
                          workspace, stream);
    });
    kv_weights.resize(kCap, kv_w);
    expect_throw("arena too small for the envelope", [&] {
        Tensor small_arena(d_arena.p, DType::BF16, {kTok, 8});
        dspark_ctx_commit(taps_t, positions_t, 64, fc_w, norm_t, kv_weights, k_norm_t, inv_freq_t, a, small_arena,
                          workspace, stream);
    });
    expect_throw("wrong taps width", [&] {
        Tensor bad_taps(d_taps.p, DType::BF16, {kTapWidth - 1, 1});
        dspark_ctx_commit(bad_taps, positions_t, 1, fc_w, norm_t, kv_weights, k_norm_t, inv_freq_t, a, arena_t,
                          workspace, stream);
    });
    expect_throw("bad attention scaling", [&] {
        dspark_ctx_commit(taps_t, positions_t, 1, fc_w, norm_t, kv_weights, k_norm_t, inv_freq_t, -1.0f, arena_t,
                          workspace, stream);
    });
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
    failures += run_yarn_table_check();
    failures += run_validation_check(stream);

    // MTP verify range across every route: T=1 decode GEMV; T=2..8 small-T; three
    // position bases covering 0, the AIME flip neighborhood, and the full-context end.
    for (const int base : {0, 137, 262136}) {
        for (const int t : {1, 2, 3, 4, 6, 8}) {
            failures += run_commit_case(stream, {t, base, static_cast<std::uint32_t>(0x1000 + base + t)});
        }
    }
    // Prefill-class T: small-T route (32) and the MMA full-token variant (128).
    failures += run_commit_case(stream, {32, 0, 0x1800});
    failures += run_commit_case(stream, {128, 0, 0x1900});

    // Monotone append: one arena, two disjoint commits.
    failures += run_append_check(stream);

    std::cout << (failures == 0 ? "OK" : "FAIL") << " dspark_ctx_commit correctness\n";
    cudaStreamDestroy(stream);
    return failures == 0 ? 0 : 1;
}