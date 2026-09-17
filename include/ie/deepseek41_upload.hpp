// include/ie/deepseek41_upload.hpp — V4.1 device upload primitives.
//
// Split from deepseek41.hpp so the binder stays SYCL-free: Phase 3's load path is host-only and
// should not drag the device compiler into every translation unit that reads a config.
#pragma once

#include "ie/deepseek41.hpp"
#include "ie/deepseek4_experts.hpp"

#include <sycl/sycl.hpp>

#include <string>
#include <vector>

namespace ie {

// Device-side E4M3 decode. Deliberately NOT ie::e4m3_to_f32: that one uses std::nanf and
// std::memcpy, neither of which belongs in device code, and it builds values by multiplication.
// This builds the fp32 bit pattern directly, which is exact and — the point — cannot be
// affected by the fast float model the way arithmetic can. Checked against the host function
// on all 256 codes by tools/ds41_device_test.cpp.
inline float ds41_e4m3(uint8_t b) {
    // e == 0 subnormals: value = m * 2^-9, the seven fp32 patterns written out. Computing
    // these would be exact in fp32 too (2^-9 is normal), but a table keeps the kernel branchless
    // where it matters and matches the host bit-for-bit by construction.
    constexpr uint32_t kSub[8] = {0x00000000u, 0x3B000000u, 0x3B800000u, 0x3BC00000u,
                                  0x3C000000u, 0x3C200000u, 0x3C400000u, 0x3C600000u};
    const uint32_t s = (uint32_t(b) >> 7) & 1u, e = (uint32_t(b) >> 3) & 0xFu, m = uint32_t(b) & 7u;
    uint32_t mag;
    if (e == 0)                  mag = kSub[m];
    else if (e == 15 && m == 7)  mag = 0x7FC00000u;            // the one NaN encoding
    else                         mag = ((e + 120u) << 23) | (m << 20);  // 2^(e-7) * (1 + m/8)
    return sycl::bit_cast<float>((s << 31) | mag);
}

// Device-side E8M0: value = 2^(b-127), 255 is NaN. b == 0 is 2^-127, SUBNORMAL in fp32 — built
// from bits for the same reason the host version is (see include/ie/fp8.hpp): the arithmetic
// form is flushed to zero under icpx's default fast float model.
inline float ds41_e8m0(uint8_t b) {
    if (b == 255) return sycl::bit_cast<float>(0x7FC00000u);
    const int ex = int(b) - 127;
    const uint32_t bits = (ex >= -126) ? (uint32_t(ex + 127) << 23)
                                       : (uint32_t(1) << uint32_t(ex + 149));
    return sycl::bit_cast<float>(bits);
}


// Upload `E_take` of a layer's routed experts into a DS4ExpertBank.
//
// V4.1 ships each expert as an [N, K/2] FP4 plane plus an [N, K/32] E8M0 scale plane. The
// SHAPES are what DS4ExpertBank's mx_qs / mx_e already are, and the 16-code nibble table is
// bit-identical — but the element ORDER inside each 32-element block is NOT:
//
//   engine (gpt-oss interleaved) : byte j holds element j (low nibble) and j+16 (high)
//   V4.1   (OCP MX sequential)   : byte b holds elements 2b (low) and 2b+1 (high)
//
// Only elements 0 and 31 coincide, so the shipped bytes read straight give a PERMUTATION of the
// right weights — plausible magnitudes, no NaN, no alarm anywhere downstream. This function
// therefore permutes the FP4 planes at upload (the scale planes do cross unchanged), after
// which every existing tuned MXFP4 kernel runs unmodified. See
// docs/deepseek41/08_PHASE4_RESULT_2026-09-12.md.
//
// Errors (never silently falls back) on any expert whose shapes disagree with K and N.
std::string ds41_expert_bank_upload(sycl::queue& q, const std::vector<Ds41Tensor>& experts,
                                    uint32_t K, uint32_t N, uint32_t E_take, DS4ExpertBank& out);

// Dequantise a dense FP8 weight on device: out[n*K+k] = half(e4m3(w) * e8m0(scale)), with one
// E8M0 scale per `bn` x `bk` block (32x32 in this checkpoint). `w` and `scale` are DEVICE
// pointers; `out` holds N*K halves.
sycl::event ds41_dense_dequant_f16(sycl::queue& q, const uint8_t* w, const uint8_t* scale,
                                   uint32_t N, uint32_t K, uint32_t bn, uint32_t bk,
                                   sycl::half* out, const std::vector<sycl::event>& deps = {});

// The T = 1 GEMV over that same FP8 matrix without dequantising it (docs/deepseek41/43): y[n] =
// sum_k x[k] * e4m3(w[n, k]) * e8m0(scale[n/32, k/32]) in fp32, stored as fp32 (the oneDNN path's
// output type). K a multiple of 32; x [K] fp16, w [N, K] bytes, scale [N/32, K/32] bytes, y [N] fp32. Same weight values as the
// dequant, a different summation order from oneDNN's fp16 GEMM.
sycl::event gemv_fp8_e4m3_f16(sycl::queue& q, const sycl::half* x, const uint8_t* w, const uint8_t* scale,
                              float* y, uint32_t K, uint32_t N, const std::vector<sycl::event>& deps = {});
// The same GEMV with a per-call-site profiler name (Phase 31): the eight dense projections per layer
// share one kernel and lose efficiency by SHAPE, so a table keyed on the kernel name alone cannot say which.
sycl::event gemv_fp8_e4m3_f16_tagged(sycl::queue& q, const sycl::half* x, const uint8_t* w, const uint8_t* scale,
                                     float* y, uint32_t K, uint32_t N, const char* tag,
                                     const std::vector<sycl::event>& deps = {});
// Phase 32: block-diagonal forms (o_a). Column n reads the activation slice (n / cols_per_group) * K.
sycl::event gemv_fp8_e4m3_f16_grouped(sycl::queue& q, const sycl::half* x, const uint8_t* w, const uint8_t* scale,
                                      float* y, uint32_t K, uint32_t N, uint32_t cols_per_group, const char* tag,
                                      const std::vector<sycl::event>& deps = {});
sycl::event gemv_fp8_rows_grouped(sycl::queue& q, const sycl::half* x, uint32_t x_stride, const uint8_t* w, const uint8_t* scale,
                                  float* y, uint32_t M, uint32_t K, uint32_t N, uint32_t cols_per_group,
                                  const std::vector<sycl::event>& deps = {});
// The two decodes of docs/deepseek41/45 explicitly: the Phase 17 scalar E4M3 decode and the fp16 table
// decode ("packed" is the env's name; the pair decode it was written for is gone). gemv_fp8_e4m3_f16 picks
// the variant IE_DS41_FP8_PACKED names (0 scalar, 1-3 the tables; unset = the default in gemv_fp8.cpp).
sycl::event gemv_fp8_e4m3_f16_scalar(sycl::queue& q, const sycl::half* x, const uint8_t* w, const uint8_t* scale,
                                     float* y, uint32_t K, uint32_t N, const std::vector<sycl::event>& deps = {});
sycl::event gemv_fp8_e4m3_f16_packed(sycl::queue& q, const sycl::half* x, const uint8_t* w, const uint8_t* scale,
                                     float* y, uint32_t K, uint32_t N, const std::vector<sycl::event>& deps = {});
// DSpark P2 (docs/deepseek41/55): the ROWS kernels -- 1..8 activation rows [M, x_stride] against the same weights, every
// row's arithmetic the one-row kernel's term for term (bit-identical per row across M; the unit test checks), the
// weights read once. fp32 out [M, N]. The fp16 variant's column group: column n reads x + (n / cols_per_group) * K
// (0 = one group: a plain matrix); K a multiple of 32.
sycl::event gemv_fp8_rows(sycl::queue& q, const sycl::half* x, uint32_t x_stride, const uint8_t* w, const uint8_t* scale,
                          float* y, uint32_t M, uint32_t K, uint32_t N, const std::vector<sycl::event>& deps = {});
sycl::event gemv_f16_rows(sycl::queue& q, const sycl::half* x, uint32_t x_stride, const sycl::half* w, float* y,
                          uint32_t M, uint32_t K, uint32_t N, uint32_t cols_per_group = 0, const std::vector<sycl::event>& deps = {});
// Any decode variant explicitly (docs/deepseek41/46 term 3): 0 the scalar, 1 the fp16 table, 2 the fp32 table,
// 3 the fp32 table with the block's activation staged as fp32 -- all bit-identical (the unit test checks).
sycl::event gemv_fp8_e4m3_f16_variant(sycl::queue& q, const sycl::half* x, const uint8_t* w, const uint8_t* scale,
                                      float* y, uint32_t K, uint32_t N, int dec, const std::vector<sycl::event>& deps = {});

// V4.1's hyper-connection, split so `pre` can be consumed one sublayer later than it is
// produced — see src/ops/deepseek41_ops.cpp for why this is not ds4_hyper_connection.
//   streams [n_tokens, hc_mult, hidden]   fn [mix, hc_mult*hidden]   base [mix]   scale [3]
//   pre, post [n_tokens, hc_mult]         comb [n_tokens, hc_mult, hc_mult]
// `scratch` (device, >= n_tokens * ds41_hc_mixes_scratch_floats(hc_mult) floats): it selects the decode
// shape for every row of the step (DSpark P2: T rows, each chunked exactly as the 1-row shape) -- the mix dot products spread over ~200 work-groups and a
// one-work-group finish (two launches) instead of one work-group walking 2 MB serially
// (~700 us per call at T = 1, Phase 11). Without it, or at n_tokens > 1, the general kernel.
// The two launches carry no event dependency between them: the finish follows the partials
// because `q` is in-order, as every queue of this forward is (gate 11 finding 5).
sycl::event ds41_hc_mixes(sycl::queue& q,
                          const float* streams, const float* fn, const float* base,
                          const float* scale, float* pre, float* post, float* comb,
                          uint32_t n_tokens, uint32_t hidden, uint32_t hc_mult,
                          uint32_t sinkhorn_iters, float rms_eps, float hc_eps,
                          const std::vector<sycl::event>& deps = {}, float* scratch = nullptr);
constexpr uint32_t kDs41HcChunks = 8;
inline uint32_t ds41_hc_mixes_scratch_floats(uint32_t hc_mult) { return ((2u + hc_mult) * hc_mult + 1u) * kDs41HcChunks; }

// hc_pre: y[t, d] = sum_h pre[t, h] * streams[t, h, d].
sycl::event ds41_hc_collapse(sycl::queue& q, const float* streams, const float* pre, float* y,
                             uint32_t n_tokens, uint32_t hidden, uint32_t hc_mult,
                             const std::vector<sycl::event>& deps = {});

// Engram gate + residual: out = h + sigmoid(signed_sqrt(normalised dot(h, qk*key))) * value.
//   h, key, out [n_tokens, hc_mult, hidden]   value [n_tokens, hidden]   qk [hc_mult, hidden]
//   gate_out [n_tokens, hc_mult]   (the gate itself, so it can be checked on its own)
sycl::event ds41_engram_gate(sycl::queue& q, const float* h, const float* key, const float* value,
                             const float* qk, float* out, float* gate_out,
                             uint32_t n_tokens, uint32_t hidden, uint32_t hc_mult, float eps,
                             const std::vector<sycl::event>& deps = {});

}  // namespace ie
