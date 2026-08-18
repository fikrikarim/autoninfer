// dspark_ctx_commit wrapper: validates the DSpark draft-context commit contract,
// suballocates the private GEMM/normalization buffers, and issues the stream-ordered
// sequence: fc GEMM -> hidden RMSNorm -> per-layer (kv GEMM -> fused norm/rotate/scatter).

#include "ninfer/ops/dspark_ctx_commit.h"
#include "ops/dspark_ctx_commit/dspark_launch.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/rmsnorm.h"

#include <cmath>
#include <span>
#include <stdexcept>

namespace ninfer::ops {
namespace {

using namespace ninfer::ops::dspark;

void require(bool ok, const char* what) {
    if (!ok) throw std::invalid_argument(std::string("dspark_ctx_commit: ") + what);
}

} // namespace

std::size_t dspark_ctx_commit_workspace_capacity_bytes(std::int32_t min_tokens, std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("dspark_ctx_commit_workspace_capacity_bytes: bad token range");
    }
    // Three 256-byte-aligned suballocations (x, normalized x, per-layer kv); every size is
    // an exact multiple of 256 bytes for T >= 1, so from a 256-aligned cursor the
    // high-water mark is the exact sum.
    const std::size_t t = static_cast<std::size_t>(max_tokens);
    return (static_cast<std::size_t>(kHidden) * 2 + static_cast<std::size_t>(kKvRows)) * t * 2;
}

void dspark_ctx_commit(const Tensor& taps, const Tensor& positions, std::int32_t position_max_exclusive,
                       const Weight& fc_weight, const Tensor& hidden_norm_weight,
                       const std::span<const Weight>& kv_weights, const Tensor& k_norm_weight,
                       const Tensor& inv_freq, float attention_scaling, Tensor& arena,
                       WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t T = taps.ne[1];
    require(taps.dtype == DType::BF16 && taps.ne[0] == kTapWidth && T >= 1 && taps.ne[2] == 1 && taps.ne[3] == 1
                && taps.is_contiguous(),
            "taps must be contiguous BF16 [25600, T, 1, 1] with T >= 1");
    require(positions.dtype == DType::I32 && positions.ne[0] == T && positions.ne[1] == 1 && positions.is_contiguous(),
            "positions must be contiguous I32 [T, 1, 1, 1]");
    require(position_max_exclusive >= 1, "position_max_exclusive must be >= 1");
    require(fc_weight.qtype == QType::BF16_CTRL && fc_weight.n == kHidden && fc_weight.k == kTapWidth
                && fc_weight.qdata != nullptr,
            "fc_weight must be BF16_CTRL n=5120 k=25600");
    require(hidden_norm_weight.dtype == DType::BF16 && hidden_norm_weight.ne[0] == kHidden
                && hidden_norm_weight.is_contiguous(),
            "hidden_norm_weight must be BF16 [5120]");
    require(kv_weights.size() == static_cast<std::size_t>(kLayers), "kv_weights must span 5 draft layers");
    for (const Weight& w : kv_weights) {
        require(w.qtype == QType::BF16_CTRL && w.n == kKvRows && w.k == kHidden && w.qdata != nullptr,
                "kv weights must be BF16_CTRL n=2048 k=5120");
    }
    require(k_norm_weight.dtype == DType::BF16 && k_norm_weight.ne[0] == kHeadDim && k_norm_weight.is_contiguous(),
            "k_norm_weight must be BF16 [128]");
    require(inv_freq.dtype == DType::FP32 && inv_freq.ne[0] == kHeadDimHalf && inv_freq.is_contiguous(),
            "inv_freq must be FP32 [64]");
    require(std::isfinite(attention_scaling) && attention_scaling > 0.0f, "attention_scaling must be finite and > 0");
    require(arena.dtype == DType::BF16 && arena.ne[0] == kArenaElementsPerToken &&
                arena.ne[1] >= position_max_exclusive && arena.ne[2] == 1 && arena.is_contiguous(),
            "arena must be contiguous BF16 [10240, capacity] covering position_max_exclusive");

    Tensor x_buf = workspace.alloc(DType::BF16, {kHidden, T});
    Tensor x2_buf = workspace.alloc(DType::BF16, {kHidden, T});
    Tensor kv_buf = workspace.alloc(DType::BF16, {kKvRows, T});

    linear(taps, fc_weight, x_buf, stream);
    rmsnorm(x_buf, hidden_norm_weight, 1e-6f, false, x2_buf, stream);
    for (std::int32_t l = 0; l < kLayers; ++l) {
        linear(x2_buf, kv_weights[l], kv_buf, stream);
        detail::launch_dspark_kv_rope_scatter(kv_buf, k_norm_weight, inv_freq, positions, arena, l, T,
                                              attention_scaling, stream);
    }
}

} // namespace ninfer::ops