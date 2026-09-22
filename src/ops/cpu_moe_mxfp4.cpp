// src/ops/cpu_moe_mxfp4.cpp — see include/ie/cpu_moe_mxfp4.hpp. Compiled with -mavx2 -mfma
// -fopenmp (src/CMakeLists.txt); the scalar reference needs none of them.
#include "ie/cpu_moe_mxfp4.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <omp.h>

namespace ie {

int cpu_mxfp4_nibble_int(uint32_t nb) {
    const int mag = int((0xC8643210u >> ((nb & 7u) * 4u)) & 0xFu);   // {0,1,2,3,4,6,8,12}
    return (nb & 8u) ? -mag : mag;
}

float cpu_mxfp4_e8m0_half(uint8_t e) {
    const uint32_t bits = (e < 2u) ? (0x00200000u << e) : (uint32_t(e - 1u) << 23);
    float f; std::memcpy(&f, &bits, 4); return f;
}

void cpu_dequant_mxfp4_ref(const uint8_t* qs, const uint8_t* e, float* w, uint32_t K, uint32_t N) {
    const uint32_t nb = K / 32;
    for (uint32_t n = 0; n < N; ++n)
        for (uint32_t b = 0; b < nb; ++b) {
            const uint8_t* blk = qs + (size_t(n) * nb + b) * 16;
            const float s = cpu_mxfp4_e8m0_half(e[size_t(n) * nb + b]);
            for (uint32_t j = 0; j < 16; ++j) {
                w[size_t(n) * K + b * 32 + j]      = float(cpu_mxfp4_nibble_int(blk[j] & 0xFu)) * s;
                w[size_t(n) * K + b * 32 + 16 + j] = float(cpu_mxfp4_nibble_int(blk[j] >> 4)) * s;
            }
        }
}

namespace {
// one row's dot product over its nb blocks
inline float row_dot(const float* x, const uint8_t* qs_row, const uint8_t* e_row, uint32_t nb) {
#if defined(__AVX2__)
    // the 16-entry signed table {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12} as int8, applied with
    // pshufb to 16 nibbles at once; the block's 32 weights are integers times one scale, so the
    // scale multiplies the block's dot once
    const __m128i tbl = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();
    for (uint32_t b = 0; b < nb; ++b) {
        const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs_row + size_t(b) * 16));
        const __m128i lo = _mm_shuffle_epi8(tbl, _mm_and_si128(raw, m4));                          // k = 0..15
        const __m128i hi = _mm_shuffle_epi8(tbl, _mm_and_si128(_mm_srli_epi16(raw, 4), m4));      // k = 16..31
        const float* xb = x + size_t(b) * 32;
        __m256 d = _mm256_setzero_ps();
        d = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), _mm256_loadu_ps(xb), d);
        d = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))), _mm256_loadu_ps(xb + 8), d);
        d = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), _mm256_loadu_ps(xb + 16), d);
        d = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))), _mm256_loadu_ps(xb + 24), d);
        acc = _mm256_fmadd_ps(d, _mm256_set1_ps(cpu_mxfp4_e8m0_half(e_row[b])), acc);
    }
    alignas(32) float tmp[8]; _mm256_store_ps(tmp, acc);
    return ((tmp[0] + tmp[4]) + (tmp[1] + tmp[5])) + ((tmp[2] + tmp[6]) + (tmp[3] + tmp[7]));
#else
    float acc = 0.f;
    for (uint32_t b = 0; b < nb; ++b) {
        const uint8_t* blk = qs_row + size_t(b) * 16; const float* xb = x + size_t(b) * 32;
        float d = 0.f;
        for (uint32_t j = 0; j < 16; ++j) { d += float(cpu_mxfp4_nibble_int(blk[j] & 0xFu)) * xb[j]; d += float(cpu_mxfp4_nibble_int(blk[j] >> 4)) * xb[16 + j]; }
        acc += d * cpu_mxfp4_e8m0_half(e_row[b]);
    }
    return acc;
#endif
}

// R dot products of one weight row with R activation vectors: per row exactly row_dot's sequence, the block's
// nibbles and scale decoded once for all R
inline void rows_dot(const float* const* xs, uint32_t R, const uint8_t* qs_row, const uint8_t* e_row, uint32_t nb, float* res) {
#if defined(__AVX2__)
    const __m128i tbl = _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    __m256 acc[kCpuExpertRows];
    for (uint32_t r = 0; r < R; ++r) acc[r] = _mm256_setzero_ps();
    for (uint32_t b = 0; b < nb; ++b) {
        const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs_row + size_t(b) * 16));
        const __m128i lo = _mm_shuffle_epi8(tbl, _mm_and_si128(raw, m4));
        const __m128i hi = _mm_shuffle_epi8(tbl, _mm_and_si128(_mm_srli_epi16(raw, 4), m4));
        const __m256 w0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), w1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
        const __m256 w2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), w3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
        const __m256 sc = _mm256_set1_ps(cpu_mxfp4_e8m0_half(e_row[b]));
        for (uint32_t r = 0; r < R; ++r) {
            const float* xb = xs[r] + size_t(b) * 32;
            __m256 d = _mm256_setzero_ps();
            d = _mm256_fmadd_ps(w0, _mm256_loadu_ps(xb), d);
            d = _mm256_fmadd_ps(w1, _mm256_loadu_ps(xb + 8), d);
            d = _mm256_fmadd_ps(w2, _mm256_loadu_ps(xb + 16), d);
            d = _mm256_fmadd_ps(w3, _mm256_loadu_ps(xb + 24), d);
            acc[r] = _mm256_fmadd_ps(d, sc, acc[r]);
        }
    }
    for (uint32_t r = 0; r < R; ++r) {
        alignas(32) float tmp[8]; _mm256_store_ps(tmp, acc[r]);
        res[r] = ((tmp[0] + tmp[4]) + (tmp[1] + tmp[5])) + ((tmp[2] + tmp[6]) + (tmp[3] + tmp[7]));
    }
#else
    for (uint32_t r = 0; r < R; ++r) res[r] = row_dot(xs[r], qs_row, e_row, nb);
#endif
}

void gemv_mxfp4_f32_rows(const float* const* xs, uint32_t R, const uint8_t* qs, const uint8_t* e, float* const* ys, uint32_t K, uint32_t N, int nthreads) {
    const uint32_t nb = K / 32;
    #pragma omp parallel for schedule(static) num_threads(nthreads > 0 ? nthreads : omp_get_max_threads())
    for (int64_t n = 0; n < int64_t(N); ++n) {
        float res[kCpuExpertRows];
        rows_dot(xs, R, qs + size_t(n) * nb * 16, e + size_t(n) * nb, nb, res);
        for (uint32_t r = 0; r < R; ++r) ys[r][n] = res[r];
    }
}
}  // namespace

void cpu_gemv_mxfp4_f32(const float* x, const uint8_t* qs, const uint8_t* e, float* y, uint32_t K, uint32_t N, int nthreads) {
    const uint32_t nb = K / 32;
    #pragma omp parallel for schedule(static) num_threads(nthreads > 0 ? nthreads : omp_get_max_threads())
    for (int64_t n = 0; n < int64_t(N); ++n)
        y[n] = row_dot(x, qs + size_t(n) * nb * 16, e + size_t(n) * nb, nb);
}

void cpu_expert_mxfp4(const void* slot, const Ds4SlotLayout& lay, const float* x, float* scratch, float* out, float swiglu_limit, int nthreads) {
    const auto* base = static_cast<const uint8_t*>(slot);
    const uint32_t H = lay.gate.K, EF = lay.gate.N;
    float* g = scratch; float* u = scratch + EF;
    cpu_gemv_mxfp4_f32(x, base + lay.gate.off0, base + lay.gate.off1, g, H, EF, nthreads);
    cpu_gemv_mxfp4_f32(x, base + lay.up.off0,   base + lay.up.off1,   u, H, EF, nthreads);
    for (uint32_t i = 0; i < EF; ++i) {                    // the GPU's ds4_swiglu_clamped_h, in fp32
        float gg = g[i], uu = u[i];
        if (swiglu_limit > 0.f) { gg = std::min(gg, swiglu_limit); uu = std::max(std::min(uu, swiglu_limit), -swiglu_limit); }
        g[i] = (gg / (1.0f + std::exp(-gg))) * uu;
    }
    cpu_gemv_mxfp4_f32(g, base + lay.down.off0, base + lay.down.off1, out, EF, H, nthreads);
}

void cpu_expert_mxfp4_rows(const void* slot, const Ds4SlotLayout& lay, const float* const* xs, uint32_t R, float* scratch, float* out, float swiglu_limit, int nthreads) {
    const auto* base = static_cast<const uint8_t*>(slot);
    const uint32_t H = lay.gate.K, EF = lay.gate.N;
    float* g[kCpuExpertRows]; float* u[kCpuExpertRows]; float* o[kCpuExpertRows]; const float* gc[kCpuExpertRows];
    for (uint32_t r = 0; r < R; ++r) { g[r] = scratch + size_t(r) * 2 * EF; u[r] = g[r] + EF; o[r] = out + size_t(r) * H; gc[r] = g[r]; }
    gemv_mxfp4_f32_rows(xs, R, base + lay.gate.off0, base + lay.gate.off1, g, H, EF, nthreads);
    gemv_mxfp4_f32_rows(xs, R, base + lay.up.off0,   base + lay.up.off1,   u, H, EF, nthreads);
    for (uint32_t r = 0; r < R; ++r)
        for (uint32_t i = 0; i < EF; ++i) {                // cpu_expert_mxfp4's SwiGLU, row by row
            float gg = g[r][i], uu = u[r][i];
            if (swiglu_limit > 0.f) { gg = std::min(gg, swiglu_limit); uu = std::max(std::min(uu, swiglu_limit), -swiglu_limit); }
            g[r][i] = (gg / (1.0f + std::exp(-gg))) * uu;
        }
    gemv_mxfp4_f32_rows(gc, R, base + lay.down.off0, base + lay.down.off1, o, EF, H, nthreads);
}

}  // namespace ie
