// Host YaRN inv_freq table builder and attention-scaling factor for the DSpark draft
// speculator. Bit-exact port of the transformers 5.12.1 Qwen3RotaryEmbedding
// _compute_yarn_parameters pipeline (FP32 pow / FP64 boundary arithmetic / 2*pi).

#include "ninfer/ops/dspark_ctx_commit.h"

#include <cmath>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr double kTwoPi = 6.2831853071795864769215;

// Correction dimension for one beta: find_correction_dim(theta, beta, original_max, dim)
// = (dim * ln(original_max / (beta * 2*pi))) / (2 * ln(theta)), FP64, beta*2*pi
// evaluated as (beta * 2.0) * pi in double... reference: num_rotations * 2 * math.pi,
// i.e. (beta * 2.0) * kPi; kTwoPi/2 == kPi is exact, use beta * kTwoPi directly
// ((beta * 2.0) * pi == beta * (2.0 * pi) only in FP64 when both associate identically;
// the reference multiplies left-to-right: (beta * 2.0) then * pi. beta * kTwoPi
// differs in general, so mirror the reference association exactly.
constexpr double kPi = kTwoPi / 2.0;

double correction_dim(std::int32_t head_dim, double omp, double beta, double theta) {
    const double dim = head_dim;
    const double rot = beta * 2.0 * kPi; // (beta * 2.0) * pi, left to right
    return dim * std::log(omp / rot) / (2.0 * std::log(theta));
}

} // namespace

void dspark_yarn_inverse_frequencies(float rope_theta, std::int32_t original_max_position_embeddings,
                                     float factor, float beta_fast, float beta_slow,
                                     std::int32_t head_dim, std::span<float> out) {
    if (head_dim < 2 || (head_dim % 2) != 0) {
        throw std::invalid_argument("dspark_yarn_inverse_frequencies: head_dim must be even and >= 2");
    }
    const int n = head_dim / 2;
    if (static_cast<std::int32_t>(out.size()) != n) {
        throw std::invalid_argument("dspark_yarn_inverse_frequencies: out must have head_dim/2 elements");
    }
    const double theta = rope_theta;
    const double omp   = original_max_position_embeddings;
    if (!(rope_theta > 0.0f) || !(factor > 1.0f) || original_max_position_embeddings <= 0) {
        throw std::invalid_argument("dspark_yarn_inverse_frequencies: bad theta/factor/original_max");
    }
    // Reference: truncate=False semantics (low = floor, high = ceil), clamped to range.
    int low  = static_cast<int>(std::floor(correction_dim(head_dim, omp, beta_fast, theta)));
    int high = static_cast<int>(std::ceil(correction_dim(head_dim, omp, beta_slow, theta)));
    if (low < 0) low = 0;
    if (high > head_dim - 1) high = head_dim - 1;
    float minf = static_cast<float>(low);
    float maxf = static_cast<float>(high);
    if (minf == maxf) {
        maxf += 0.001f; // reference singularity guard
    }
    const float denom = maxf - minf;
    const float factor_f = factor;
    for (int i = 0; i < n; ++i) {
        const float pf    = powf(rope_theta, static_cast<float>(i) / static_cast<float>(n));
        const float ext   = 1.0f / pf;
        const float inter = 1.0f / (factor_f * pf);
        float ramp        = (static_cast<float>(i) - minf) / denom;
        ramp              = ramp < 0.0f ? 0.0f : (ramp > 1.0f ? 1.0f : ramp);
        const float ef    = 1.0f - ramp;
        out[i]            = inter * (1.0f - ef) + ext * ef;
    }
}

float dspark_yarn_attention_scaling(float factor) {
    if (!(factor > 1.0f)) {
        throw std::invalid_argument("dspark_yarn_attention_scaling: factor must be > 1");
    }
    // Reference: 0.1 * mscale * math.log(scale) + 1.0, mscale = 1.0, FP64, then used as
    // an FP32 scalar in the rope cos/sin scaling.
    return static_cast<float>(0.1 * std::log(static_cast<double>(factor)) + 1.0);
}

} // namespace ninfer::ops