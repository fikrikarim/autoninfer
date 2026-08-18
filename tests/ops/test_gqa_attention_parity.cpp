// Bit-exact per-column parity across the GQA small-T verify widths (the MTP
// losslessness contract for attention, model-doc 8).
//
// The greedy MTP emitted stream is lossless only if every column of a verify
// round is a finite-precision clone of the ordinary T=1 decode route:
//   * the committed column drives the emitted token, and
//   * every draft column's target argmax drives the accept/reject decision, so
//     a wrong draft can be accepted or a correct one rejected, and the emitted
//     stream diverges from the k=0 reference.
// For each column of a T=2..6 verify round that means: same paged KV prefix,
// same new key at that column's position, and the column's output bit-identical
// to a width-1 decode at the same position - independent of the verify width
// (up to the five-draft maximum), the KV dtype (I8/BF16), the window (short
// <=1029 vs long, split/tier boundaries), and the execution envelope.
//
// Production fidelity: the T=1 runs use the ordinary-decode profile (no valid
// tensor, exact {W0, W0} envelope); the T=2..6 runs use the MTP-verify profile
// (device valid-columns tensor, {1, W_T} envelope), exactly as TextContext::
// target_verify_batch drives gqa_attention.

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHeadDim     = 256;
constexpr std::int32_t kQuantGroup  = 64;
constexpr std::int32_t kQuantGroups = kHeadDim / kQuantGroup;
constexpr float kAttentionScale     = 0.0625f;

struct Geometry {
    const char* name;
    std::int32_t q_heads;
    std::int32_t kv_heads;

    [[nodiscard]] std::int32_t query_group() const { return q_heads / kv_heads; }
};

// Registered 27B geometry (group of six): exercises RowTiles 1 (T=1,2) and 2
// (T=3,4) in the I8 producer/consumer split.
constexpr Geometry kGeometry{"qwen3_6_27b", 24, 4};

std::int32_t align_up_page(std::int32_t value) {
    constexpr std::int32_t kFixtureAlignment = 2 * kPagedKVPageSize;
    return ((value + kFixtureAlignment - 1) / kFixtureAlignment) * kFixtureAlignment;
}

std::size_t q_index(const Geometry& geometry, std::int32_t head, std::int32_t d,
                    std::int32_t token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(geometry.q_heads) * static_cast<std::int32_t>(token));
}

std::size_t kv_input_index(const Geometry& geometry, std::int32_t head, std::int32_t d,
                           std::int32_t token) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(geometry.kv_heads) * static_cast<std::int32_t>(token));
}

std::size_t cache_index(const Geometry& geometry, std::int32_t padded_context, std::int32_t head,
                        std::int32_t position, std::int32_t d) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kHeadDim) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::int32_t>(head));
}

std::size_t scale_index(const Geometry& geometry, std::int32_t padded_context, std::int32_t head,
                        std::int32_t position, std::int32_t group) {
    (void)geometry;
    return static_cast<std::size_t>(group) +
           static_cast<std::size_t>(kQuantGroups) *
               (static_cast<std::size_t>(position) +
                static_cast<std::size_t>(padded_context) * static_cast<std::int32_t>(head));
}

std::size_t cache_elements(const Geometry& geometry, std::int32_t padded_context) {
    return static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(padded_context) *
           static_cast<std::size_t>(geometry.kv_heads);
}

std::size_t scale_elements(const Geometry& geometry, std::int32_t padded_context) {
    return static_cast<std::size_t>(kQuantGroups) * static_cast<std::size_t>(padded_context) *
           static_cast<std::size_t>(geometry.kv_heads);
}

std::size_t paged_index(std::int32_t leading_extent, const Geometry& geometry,
                        std::int32_t physical_page, std::int32_t head, std::int32_t position,
                        std::int32_t leading) {
    return static_cast<std::size_t>(leading) +
           static_cast<std::size_t>(leading_extent) *
               (static_cast<std::size_t>(position % kPagedKVPageSize) +
                static_cast<std::size_t>(kPagedKVPageSize) *
                    (static_cast<std::size_t>(head) + static_cast<std::size_t>(geometry.kv_heads) *
                                                          static_cast<std::int32_t>(physical_page)));
}

template <typename T>
std::vector<T> scatter_paged(const std::vector<T>& logical, std::int32_t leading_extent,
                             const Geometry& geometry, std::int32_t logical_capacity) {
    const std::int32_t logical_pages = logical_capacity / kPagedKVPageSize;
    std::vector<T> physical(static_cast<std::size_t>(leading_extent) * kPagedKVPageSize *
                            static_cast<std::size_t>(geometry.kv_heads) *
                            static_cast<std::size_t>(logical_pages));
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            const std::int32_t page = position / kPagedKVPageSize; // identity mapping
            for (std::int32_t leading = 0; leading < leading_extent; ++leading) {
                const std::size_t source = static_cast<std::size_t>(leading) +
                                           static_cast<std::size_t>(leading_extent) *
                                               (static_cast<std::size_t>(position) +
                                                static_cast<std::size_t>(logical_capacity) *
                                                    static_cast<std::size_t>(head));
                physical[paged_index(leading_extent, geometry, page, head, position, leading)] =
                    logical[source];
            }
        }
    }
    return physical;
}

std::vector<float> make_bf16_values(std::size_t count, std::uint32_t seed, float lo, float hi) {
    std::vector<float> values(count);
    fill_uniform(values, seed, lo, hi);
    round_to_bf16(values);
    return values;
}

std::vector<std::uint16_t> to_bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::uint16_t f32_to_f16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign   = (bits >> 16) & 0x8000u;
    const std::uint32_t exp    = (bits >> 23) & 0xffu;
    std::uint32_t mantissa     = bits & 0x007fffffu;
    if (exp == 0xffu) {
        return static_cast<std::uint16_t>(sign | (mantissa == 0 ? 0x7c00u : 0x7e00u));
    }
    const int half_exp = static_cast<int>(exp) - 127 + 15;
    if (half_exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
    if (half_exp <= 0) {
        if (half_exp < -10) { return static_cast<std::uint16_t>(sign); }
        mantissa |= 0x00800000u;
        const int shift             = 14 - half_exp;
        std::uint32_t half_mantissa = mantissa >> shift;
        const std::uint32_t halfway = 1u << (shift - 1);
        const std::uint32_t tail    = mantissa & ((1u << shift) - 1u);
        if (tail > halfway || (tail == halfway && (half_mantissa & 1u) != 0u)) { ++half_mantissa; }
        return static_cast<std::uint16_t>(sign | half_mantissa);
    }
    std::uint32_t half_mantissa = mantissa >> 13;
    const std::uint32_t tail    = mantissa & 0x1fffu;
    std::uint32_t rounded_exp   = static_cast<std::uint32_t>(half_exp);
    if (tail > 0x1000u || (tail == 0x1000u && (half_mantissa & 1u) != 0u)) {
        ++half_mantissa;
        if (half_mantissa == 0x400u) {
            half_mantissa = 0;
            ++rounded_exp;
            if (rounded_exp >= 31) { return static_cast<std::uint16_t>(sign | 0x7c00u); }
        }
    }
    return static_cast<std::uint16_t>(sign | (rounded_exp << 10) | half_mantissa);
}

float f16_bits_to_f32(std::uint16_t bits) {
    const bool negative = (bits & 0x8000u) != 0;
    const int exp       = (bits >> 10) & 0x1f;
    const int mantissa  = bits & 0x03ff;
    float magnitude     = 0.0f;
    if (exp == 0) {
        magnitude = std::ldexp(static_cast<float>(mantissa), -24);
    } else if (exp == 31) {
        magnitude = mantissa == 0 ? std::numeric_limits<float>::infinity()
                                  : std::numeric_limits<float>::quiet_NaN();
    } else {
        magnitude = std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f, exp - 15);
    }
    return negative ? -magnitude : magnitude;
}

std::int32_t round_even_to_i32(float value) {
    const float lower_f  = std::floor(value);
    const float fraction = value - lower_f;
    std::int32_t lower   = static_cast<std::int32_t>(lower_f);
    if (fraction < 0.5f) return lower;
    if (fraction > 0.5f) return lower + 1;
    return (lower & 1) == 0 ? lower : lower + 1;
}

struct HostCache {
    Geometry geometry;
    DType dtype;
    std::int32_t max_context;
    std::int32_t logical_capacity;
    std::vector<std::uint16_t> k_bf16;
    std::vector<std::uint16_t> v_bf16;
    std::vector<std::int8_t> k_i8;
    std::vector<std::int8_t> v_i8;
    std::vector<std::uint16_t> k_scale;
    std::vector<std::uint16_t> v_scale;
};

void encode_group(const std::vector<float>& source, std::size_t source_base,
                  std::vector<std::int8_t>& codes, std::size_t code_base,
                  std::vector<std::uint16_t>& scales, std::size_t scale_offset) {
    float absmax = 0.0f;
    for (std::int32_t i = 0; i < kQuantGroup; ++i) {
        absmax = std::max(absmax, std::abs(source[source_base + static_cast<std::size_t>(i)]));
    }
    const float unrounded_scale    = absmax / 127.0f;
    const std::uint16_t scale_bits = f32_to_f16_bits(unrounded_scale);
    const float stored_scale       = f16_bits_to_f32(scale_bits);
    const float inverse_scale      = stored_scale == 0.0f ? 0.0f : 1.0f / stored_scale;
    scales[scale_offset]           = scale_bits;
    for (std::int32_t i = 0; i < kQuantGroup; ++i) {
        std::int32_t code = 0;
        if (stored_scale != 0.0f) {
            const float scaled = source[source_base + static_cast<std::size_t>(i)] * inverse_scale;
            code               = std::clamp(round_even_to_i32(scaled), -127, 127);
        }
        codes[code_base + static_cast<std::size_t>(i)] = static_cast<std::int8_t>(code);
    }
}

HostCache make_cache(const Geometry& geometry, DType dtype, std::int32_t max_context,
                     std::uint32_t seed) {
    const std::int32_t logical_capacity = align_up_page(max_context);
    const std::size_t elements          = cache_elements(geometry, logical_capacity);
    std::vector<float> logical_k        = make_bf16_values(elements, seed, -0.25f, 0.25f);
    std::vector<float> logical_v        = make_bf16_values(elements, seed + 1u, -1.0f, 1.0f);

    HostCache cache{geometry, dtype, max_context, logical_capacity};
    if (dtype == DType::BF16) {
        cache.k_bf16.assign(elements, 0);
        cache.v_bf16.assign(elements, 0);
        for (std::size_t i = 0; i < elements; ++i) {
            cache.k_bf16[i] = f32_to_bf16(logical_k[i]);
            cache.v_bf16[i] = f32_to_bf16(logical_v[i]);
        }
        return cache;
    }

    const std::size_t scales = scale_elements(geometry, logical_capacity);
    cache.k_i8.assign(elements, 0);
    cache.v_i8.assign(elements, 0);
    cache.k_scale.assign(scales, 0);
    cache.v_scale.assign(scales, 0);
    for (std::int32_t head = 0; head < geometry.kv_heads; ++head) {
        for (std::int32_t position = 0; position < logical_capacity; ++position) {
            for (std::int32_t group = 0; group < kQuantGroups; ++group) {
                const std::int32_t d   = group * kQuantGroup;
                const std::size_t code = cache_index(geometry, logical_capacity, head, position, d);
                const std::size_t scale =
                    scale_index(geometry, logical_capacity, head, position, group);
                encode_group(logical_k, code, cache.k_i8, code, cache.k_scale, scale);
                encode_group(logical_v, code, cache.v_i8, code, cache.v_scale, scale);
            }
        }
    }
    return cache;
}

class DeviceCache {
public:
    explicit DeviceCache(const HostCache& cache)
        : geometry_(cache.geometry), dtype_(cache.dtype),
          logical_capacity_(cache.logical_capacity),
          logical_pages_(logical_capacity_ / kPagedKVPageSize),
          code_elements_(static_cast<std::size_t>(kHeadDim) * kPagedKVPageSize *
                         geometry_.kv_heads * static_cast<std::size_t>(logical_pages_)),
          scale_elements_(static_cast<std::size_t>(kQuantGroups) * kPagedKVPageSize *
                          geometry_.kv_heads * static_cast<std::size_t>(logical_pages_)),
          k_(code_elements_ *
             (dtype_ == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::int8_t))),
          v_(code_elements_ *
             (dtype_ == DType::BF16 ? sizeof(std::uint16_t) : sizeof(std::int8_t))),
          k_scale_(dtype_ == DType::I8 ? scale_elements_ * sizeof(std::uint16_t) : 1),
          v_scale_(dtype_ == DType::I8 ? scale_elements_ * sizeof(std::uint16_t) : 1),
          block_table_(static_cast<std::size_t>(logical_pages_) * sizeof(std::int32_t)) {
        std::vector<std::int32_t> table(static_cast<std::size_t>(logical_pages_));
        for (std::size_t page = 0; page < table.size(); ++page) {
            table[page] = static_cast<std::int32_t>(page);
        }
        block_table_.copy_from_host(table.data(), table.size() * sizeof(std::int32_t));
        if (dtype_ == DType::BF16) {
            const auto k_physical =
                scatter_paged(cache.k_bf16, kHeadDim, geometry_, logical_capacity_);
            const auto v_physical =
                scatter_paged(cache.v_bf16, kHeadDim, geometry_, logical_capacity_);
            k_.copy_from_host(k_physical.data(), k_physical.size() * sizeof(std::uint16_t));
            v_.copy_from_host(v_physical.data(), v_physical.size() * sizeof(std::uint16_t));
        } else {
            const auto k_physical =
                scatter_paged(cache.k_i8, kHeadDim, geometry_, logical_capacity_);
            const auto v_physical =
                scatter_paged(cache.v_i8, kHeadDim, geometry_, logical_capacity_);
            const auto ks_physical =
                scatter_paged(cache.k_scale, kQuantGroups, geometry_, logical_capacity_);
            const auto vs_physical =
                scatter_paged(cache.v_scale, kQuantGroups, geometry_, logical_capacity_);
            k_.copy_from_host(k_physical.data(), k_physical.size() * sizeof(std::int8_t));
            v_.copy_from_host(v_physical.data(), v_physical.size() * sizeof(std::int8_t));
            k_scale_.copy_from_host(ks_physical.data(), ks_physical.size() * sizeof(std::uint16_t));
            v_scale_.copy_from_host(vs_physical.data(), vs_physical.size() * sizeof(std::uint16_t));
        }
    }

    PagedKVBatchLayerView view() {
        PagedKVBatchLayerView result;
        result.k_pages      = Tensor(k_.data(), dtype_,
                                     {kHeadDim, kPagedKVPageSize, geometry_.kv_heads,
                                      logical_pages_});
        result.v_pages      = Tensor(v_.data(), dtype_,
                                     {kHeadDim, kPagedKVPageSize, geometry_.kv_heads,
                                      logical_pages_});
        result.block_tables = Tensor(block_table_.data(), DType::I32,
                                     {logical_pages_, static_cast<std::int32_t>(1)});
        result.num_kv_heads = geometry_.kv_heads;
        result.head_dim     = kHeadDim;
        result.dtype        = dtype_;
        if (dtype_ == DType::I8) {
            result.k_scale_pages =
                Tensor(k_scale_.data(), DType::FP16,
                       {kQuantGroups, kPagedKVPageSize, geometry_.kv_heads, logical_pages_});
            result.v_scale_pages =
                Tensor(v_scale_.data(), DType::FP16,
                       {kQuantGroups, kPagedKVPageSize, geometry_.kv_heads, logical_pages_});
            result.quant_group = kQuantGroup;
        }
        return result;
    }

    int verify_guards(const std::string& label) const {
        int failures = 0;
        failures += k_.verify_guards((label + " cache-k").c_str());
        failures += v_.verify_guards((label + " cache-v").c_str());
        if (dtype_ == DType::I8) {
            failures += k_scale_.verify_guards((label + " cache-k-scale").c_str());
            failures += v_scale_.verify_guards((label + " cache-v-scale").c_str());
        }
        return failures;
    }

private:
    Geometry geometry_;
    DType dtype_;
    std::int32_t logical_capacity_;
    std::int32_t logical_pages_;
    std::size_t code_elements_;
    std::size_t scale_elements_;
    GuardedDeviceBuffer k_;
    GuardedDeviceBuffer v_;
    GuardedDeviceBuffer k_scale_;
    GuardedDeviceBuffer v_scale_;
    GuardedDeviceBuffer block_table_;
};

template <typename T>
std::vector<T> copy_from_guarded(const GuardedDeviceBuffer& buffer, std::size_t count) {
    std::vector<T> values(count);
    buffer.copy_to_host(values.data(), values.size() * sizeof(T));
    return values;
}

struct Report {
    int failures = 0;

    void compare_column(const std::string& label, const std::vector<std::uint16_t>& ref,
                        const std::vector<std::uint16_t>& other) {
        if (ref.size() != other.size()) {
            std::cerr << label << ": size mismatch (" << ref.size() << " vs " << other.size()
                      << ")\n";
            ++failures;
            return;
        }
        int hard      = 0;
        int sign_zero = 0;
        int first     = -1;
        for (std::size_t i = 0; i < ref.size(); ++i) {
            if (ref[i] == other[i]) { continue; }
            const bool is_sign_zero =
                (ref[i] == 0x0000U && other[i] == 0x8000U) ||
                (ref[i] == 0x8000U && other[i] == 0x0000U);
            if (is_sign_zero) {
                ++sign_zero;
                continue;
            }
            if (first < 0) { first = static_cast<int>(i); }
            ++hard;
            if (hard <= 8) {
                std::cerr << label << " idx=" << i << " ref=0x" << std::hex
                          << static_cast<std::uint32_t>(ref[i]) << " other=0x"
                          << static_cast<std::uint32_t>(other[i]) << std::dec << "\n";
            }
        }
        if (hard == 0) {
            if (sign_zero > 0) {
                std::cerr << label << ": " << sign_zero
                          << " sign-of-zero-only differences (soft)\n";
            }
            return;
        }
        std::cerr << label << ": " << hard << " bit differences, first=" << first << " ("
                  << sign_zero << " sign-of-zero)\n";
        ++failures;
    }
};

const char* cache_name(DType dtype) { return dtype == DType::BF16 ? "bf16" : "int8-g64"; }

// Runs one width against the shared cache/inputs. Input token t of the run is
// global token (token_offset + t) at position (position_base + t); returns all
// `width` output columns: q_heads x head_dim BF16 bits each.
std::vector<std::uint16_t> run_width(DType dtype, DeviceCache& cache, std::uint32_t seed,
                                     std::int32_t position_base, std::int32_t width,
                                     std::int32_t token_offset, bool masked) {
    const std::int32_t q_heads  = kGeometry.q_heads;
    const std::int32_t kv_heads = kGeometry.kv_heads;
    constexpr std::int32_t kMaxWidth = 6; // committed column + five draft columns

    // Shared represented inputs (kMaxWidth global tokens); the run consumes the
    // [token_offset, token_offset + width) slice, so every width of one case
    // shares the same prefix + token sequence.
    std::vector<std::uint16_t> q_bits = to_bf16_bits(
        make_bf16_values(static_cast<std::size_t>(kHeadDim) * q_heads * kMaxWidth, seed, -0.25f,
                         0.25f));
    std::vector<std::uint16_t> k_bits = to_bf16_bits(
        make_bf16_values(static_cast<std::size_t>(kHeadDim) * kv_heads * kMaxWidth, seed + 1u,
                         -0.25f, 0.25f));
    std::vector<std::uint16_t> v_bits = to_bf16_bits(
        make_bf16_values(static_cast<std::size_t>(kHeadDim) * kv_heads * kMaxWidth, seed + 2u,
                         -1.0f, 1.0f));

    const std::int32_t committed_window = position_base + 1;
    const std::int32_t full_window      = position_base + width;
    // Production profiles: T=1 decode is exact-envelope and unmasked; T>1 verify
    // is {1, W_T}-enveloped with a device valid-columns tensor.
    const ops::GqaExecutionEnvelope envelope =
        masked ? ops::GqaExecutionEnvelope{1U, static_cast<std::uint32_t>(full_window)}
               : ops::GqaExecutionEnvelope{static_cast<std::uint32_t>(committed_window),
                                           static_cast<std::uint32_t>(committed_window)};

    const std::size_t q_elems  =
        static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(q_heads);
    const std::size_t kv_elems =
        static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(kv_heads);
    const std::size_t q_off  = static_cast<std::size_t>(token_offset) * q_elems;
    const std::size_t kv_off = static_cast<std::size_t>(token_offset) * kv_elems;
    const std::size_t q_slice  = static_cast<std::size_t>(width) * q_elems;
    const std::size_t kv_slice = static_cast<std::size_t>(width) * kv_elems;

    GuardedDeviceBuffer dq(q_slice * sizeof(std::uint16_t));
    GuardedDeviceBuffer dk(kv_slice * sizeof(std::uint16_t));
    GuardedDeviceBuffer dv(kv_slice * sizeof(std::uint16_t));
    dq.copy_from_host(q_bits.data() + q_off, q_slice * sizeof(std::uint16_t));
    dk.copy_from_host(k_bits.data() + kv_off, kv_slice * sizeof(std::uint16_t));
    dv.copy_from_host(v_bits.data() + kv_off, kv_slice * sizeof(std::uint16_t));

    std::vector<std::int32_t> positions(static_cast<std::size_t>(width));
    for (std::int32_t t = 0; t < width; ++t) {
        positions[static_cast<std::size_t>(t)] = position_base + t;
    }
    GuardedDeviceBuffer dp(positions.size() * sizeof(std::int32_t));
    dp.copy_from_host(positions.data(), positions.size() * sizeof(std::int32_t));

    GuardedDeviceBuffer dtable_row(sizeof(std::int32_t));
    const std::int32_t table_row = 0;
    dtable_row.copy_from_host(&table_row, sizeof(table_row));

    GuardedDeviceBuffer dvalid(sizeof(std::int32_t));
    if (masked) {
        const std::int32_t valid = width;
        dvalid.copy_from_host(&valid, sizeof(valid));
    }

    const std::size_t out_elements = static_cast<std::size_t>(kHeadDim) *
                                     static_cast<std::size_t>(q_heads) *
                                     static_cast<std::size_t>(width);
    GuardedDeviceBuffer dout(out_elements * sizeof(std::uint16_t));
    std::vector<std::uint16_t> canary(out_elements, 0x7fc1u);
    dout.copy_from_host(canary.data(), canary.size() * sizeof(std::uint16_t));

    Tensor tq(dq.data(), DType::BF16, {kHeadDim, q_heads, width});
    Tensor tk(dk.data(), DType::BF16, {kHeadDim, kv_heads, width});
    Tensor tv(dv.data(), DType::BF16, {kHeadDim, kv_heads, width});
    Tensor tp(dp.data(), DType::I32, {width});
    Tensor ttable_row(dtable_row.data(), DType::I32, {1});
    const Tensor tvalid = masked ? Tensor(dvalid.data(), DType::I32, {1}) : Tensor{};
    Tensor tout(dout.data(), DType::BF16, {kHeadDim, q_heads, width});

    const std::size_t workspace_bytes =
        ops::gqa_attention_workspace_capacity_bytes(q_heads, dtype, envelope, 1, width, width);
    GuardedDeviceBuffer workspace_buffer(std::max<std::size_t>(workspace_bytes, 256));
    WorkspaceArena workspace(DeviceSpan{workspace_buffer.data(), workspace_buffer.bytes()});

    ops::gqa_attention(tq, tk, tv, tp, tvalid, ttable_row, kAttentionScale, cache.view(), envelope,
                       workspace, tout, nullptr);
    cuda_synchronize();

    return copy_from_guarded<std::uint16_t>(dout, out_elements);
}

struct ParityCase {
    std::int32_t base; // committed position; committed window W0 = base + 1
    std::uint32_t seed;
};

int run_dtype(DType dtype, const std::vector<ParityCase>& cases, Report& report) {
    int failures = 0;
    for (const ParityCase& test_case : cases) {
        // Covers the last draft position (base + 5) plus one key of slack.
        const std::int32_t max_context = align_up_page(test_case.base + 6);
        const HostCache initial =
            make_cache(kGeometry, dtype, max_context, test_case.seed + 10u);
        DeviceCache cache(initial);

        const std::string tag = std::string(cache_name(dtype)) + " base=" +
                                std::to_string(test_case.base);

        // T=1 decode references: global token j at position base + j, for every
        // column the verify widths may commit or draft.
        std::vector<std::vector<std::uint16_t>> refs(6);
        for (std::int32_t j = 0; j < 6; ++j) {
            refs[j] = run_width(dtype, cache, test_case.seed, test_case.base + j, 1, j,
                                /*masked=*/false);
        }
        // Every verify width, every column (committed and draft): the column's
        // output must bit-equal the T=1 decode at that column's position.
        for (std::int32_t width = 2; width <= 6; ++width) {
            const std::vector<std::uint16_t> got =
                run_width(dtype, cache, test_case.seed, test_case.base, width, 0,
                          /*masked=*/true);
            const std::size_t column_elements =
                static_cast<std::size_t>(kHeadDim) * static_cast<std::size_t>(kGeometry.q_heads);
            for (std::int32_t column = 0; column < width; ++column) {
                const std::vector<std::uint16_t> column_bits(
                    got.begin() + static_cast<std::size_t>(column) * column_elements,
                    got.begin() + static_cast<std::size_t>(column + 1) * column_elements);
                report.compare_column(
                    "parity(" + tag + " T=" + std::to_string(width) + " col=" +
                        std::to_string(column) + ")",
                    refs[column], column_bits);
            }
        }
        failures += cache.verify_guards("parity(" + tag + ")");
    }
    return failures;
}

} // namespace


int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    // Committed-window choices: inside a 64-key split, on split/tier boundaries
    // (W0 vs W_T crossing a div_up edge), at the I8 32/64-key tile boundary, at
    // the M1 context ceiling, and at a long window where the committed and full
    // windows cross a split tier.
    const std::vector<ParityCase> cases{
        {17, 9071U},    // W0=18: short window, five draft keys in the extended split
        {63, 9001U},    // W0=64: units 16 vs 17 across widths
        {95, 9011U},    // W0=96: units 24 vs 25
        {140, 9012U},   // W0=141: former T=6 short-window split-tuning range
        {200, 9013U},   // W0=201: former T=5 short-window split-tuning range
        {1003, 9021U},  // W0=1004: units 251 vs 252 (a=16)
        {1028, 9031U},  // W0=1029: 32/64-key tile boundary (same partition)
        {4095, 9041U},  // W0=4096: split tier boundary 64->33 splits
        {8197, 9051U},  // W0=8198: split tier boundary; former T=6 long-window range
        {16383, 9061U}, // W0=16384: M1 context ceiling
    };

    Report report;
    run_dtype(DType::I8, cases, report);
    run_dtype(DType::BF16, cases, report);

    std::cout << (report.failures == 0 ? "OK" : "FAIL")
              << " gqa attention per-column parity (committed + draft)\n";
    return report.failures == 0 ? 0 : 1;
}