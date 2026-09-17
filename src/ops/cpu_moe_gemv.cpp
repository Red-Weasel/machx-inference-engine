// src/ops/cpu_moe_gemv.cpp — host Q4_K/Q5_K GEMV (AVX2 when available).
// Integer dots follow ggml vec_dot_q4_K_q8_K / q5_K_q8_K (AVX2 arm).

#include "ie/cpu_moe_gemv.hpp"
#include "ie/dequant_ref.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace ie {
namespace {

template <typename F>
void parallel_n_t(int n, int nt, F fn) {
    if (n < 32 || nt <= 1) {
        for (int i = 0; i < n; ++i) fn(i);
        return;
    }
    #pragma omp parallel for schedule(static) num_threads(nt)
    for (int i = 0; i < n; ++i) fn(i);
}
// The shipped entry points keep their 8-thread teams (hyv4, the bench, the
// tests); cpu_moe_expert_q8_nt lets a caller size the team to the cores it
// pinned the worker to (GLM q* on the 12 E-cores, 2026-09-02).
template <typename F>
void parallel_n(int n, F fn) {
    parallel_n_t(n, 8, fn);
}

#if defined(__AVX2__)
inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

inline __m256i scale_shuffle_k4(int i) {
    static const uint8_t sh[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sh) + i);
}

inline void unpack_k4_scales(const uint8_t scales[12], uint32_t utmp[4]) {
    constexpr uint32_t kmask1 = 0x3f3f3f3f;
    constexpr uint32_t kmask2 = 0x0f0f0f0f;
    constexpr uint32_t kmask3 = 0x03030303;
    std::memcpy(utmp, scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
}

float dot_q4k_q8k(const block_q4_K* x, const block_q8_K* y, int nblocks) {
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();
    uint32_t utmp[4];
    for (int i = 0; i < nblocks; ++i) {
        const float d    = y[i].d * fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * fp16_to_fp32(x[i].dmin);
        unpack_k4_scales(x[i].scales, utmp);
        const __m256i mins_and_scales =
            _mm256_cvtepu8_epi16(_mm_set_epi32(int(utmp[3]), int(utmp[2]),
                                               int(utmp[1]), int(utmp[0])));
        const __m256i q8sums = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[i].bsums));
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                           _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);
        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
        const uint8_t* q4 = x[i].qs;
        const int8_t*  q8 = y[i].qs;
        __m256i sumi = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
            const __m256i scale_l = _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j + 0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j + 1));
            const __m256i q4bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4)); q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
            const __m256i q8l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8)); q8 += 32;
            __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);
            const __m256i q8h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8)); q8 += 32;
            __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
    return hsum256(acc) + _mm_cvtss_f32(acc_m);
}

float dot_q5k_q8k(const block_q5_K* x, const block_q8_K* y, int nblocks) {
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m256i mone  = _mm256_set1_epi8(1);
    __m256 acc = _mm256_setzero_ps();
    float summs = 0.f;
    uint32_t utmp[4];
    for (int i = 0; i < nblocks; ++i) {
        const float d    = y[i].d * fp16_to_fp32(x[i].d);
        const float dmin = -y[i].d * fp16_to_fp32(x[i].dmin);
        unpack_k4_scales(x[i].scales, utmp);
        const __m256i mins_and_scales =
            _mm256_cvtepu8_epi16(_mm_set_epi32(int(utmp[3]), int(utmp[2]),
                                               int(utmp[1]), int(utmp[0])));
        const __m256i q8sums = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y[i].bsums));
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                           _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * float(_mm_extract_epi32(hsum, 0));
        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
        const __m256i hbits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[i].qh));
        __m256i hmask = mone;
        int bit = 0;
        const uint8_t* q5 = x[i].qs;
        const int8_t*  q8 = y[i].qs;
        __m256i sumi = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
            const __m256i scale_0 = _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j + 0));
            const __m256i scale_1 = _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j + 1));
            const __m256i q5bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q5)); q5 += 32;
            const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            const __m256i q5h_0 = _mm256_slli_epi16(
                _mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_0 = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);
            const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            const __m256i q5h_1 = _mm256_slli_epi16(
                _mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_1 = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);
            const __m256i q8_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8)); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8)); q8 += 32;
            __m256i p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);
            p16_0 = _mm256_madd_epi16(scale_0, p16_0);
            p16_1 = _mm256_madd_epi16(scale_1, p16_1);
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum256(acc) + summs;
}

// ---- per-32 activation scales (block_q8_Ks): the integer sub-block partials
// are the same as above; each is scaled by its own activation scale in float,
// and the min term is a scalar per-sub-block sum. --------------------------
float dot_q4k_q8s(const block_q4_K* x, const block_q8_Ks* y, int nblocks) {
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    float accm = 0.f;
    uint32_t utmp[4];
    for (int i = 0; i < nblocks; ++i) {
        const float dw   = fp16_to_fp32(x[i].d);
        const float dmin = fp16_to_fp32(x[i].dmin);
        unpack_k4_scales(x[i].scales, utmp);
        const uint8_t* sm = reinterpret_cast<const uint8_t*>(utmp);   // [0..7] scales, [8..15] mins
        const float* ad = y[i].d;
        for (int j = 0; j < 8; ++j)
            accm -= dmin * float(sm[8 + j]) * ad[j] *
                    float(int(y[i].bsums[2 * j]) + int(y[i].bsums[2 * j + 1]));
        const __m128i sc128 = _mm256_extracti128_si256(
            _mm256_cvtepu8_epi16(_mm_set_epi32(int(utmp[3]), int(utmp[2]), int(utmp[1]), int(utmp[0]))), 0);
        const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
        const uint8_t* q4 = x[i].qs;
        const int8_t*  q8 = y[i].qs;
        for (int j = 0; j < 4; ++j) {
            const __m256i scale_l = _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j + 0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j + 1));
            const __m256i q4bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4)); q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
            const __m256i q8l = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8)); q8 += 32;
            const __m256i q8h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8)); q8 += 32;
            const __m256i pl = _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, q8l));
            const __m256i ph = _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q4h, q8h));
            acc = _mm256_fmadd_ps(_mm256_set1_ps(dw * ad[2 * j]),     _mm256_cvtepi32_ps(pl), acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(dw * ad[2 * j + 1]), _mm256_cvtepi32_ps(ph), acc);
        }
    }
    return hsum256(acc) + accm;
}

float dot_q5k_q8s(const block_q5_K* x, const block_q8_Ks* y, int nblocks) {
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i mone = _mm256_set1_epi8(1);
    __m256 acc = _mm256_setzero_ps();
    float accm = 0.f;
    uint32_t utmp[4];
    for (int i = 0; i < nblocks; ++i) {
        const float dw   = fp16_to_fp32(x[i].d);
        const float dmin = fp16_to_fp32(x[i].dmin);
        unpack_k4_scales(x[i].scales, utmp);
        const uint8_t* sm = reinterpret_cast<const uint8_t*>(utmp);
        const float* ad = y[i].d;
        for (int j = 0; j < 8; ++j)
            accm -= dmin * float(sm[8 + j]) * ad[j] *
                    float(int(y[i].bsums[2 * j]) + int(y[i].bsums[2 * j + 1]));
        const __m128i sc128 = _mm256_extracti128_si256(
            _mm256_cvtepu8_epi16(_mm_set_epi32(int(utmp[3]), int(utmp[2]), int(utmp[1]), int(utmp[0]))), 0);
        const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
        const __m256i hbits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x[i].qh));
        __m256i hmask = mone;
        int bit = 0;
        const uint8_t* q5 = x[i].qs;
        const int8_t*  q8 = y[i].qs;
        for (int j = 0; j < 4; ++j) {
            const __m256i scale_0 = _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j + 0));
            const __m256i scale_1 = _mm256_shuffle_epi8(scales, scale_shuffle_k4(2 * j + 1));
            const __m256i q5bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q5)); q5 += 32;
            const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_0 = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);
            const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_1 = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);
            const __m256i q8_0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8)); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q8)); q8 += 32;
            const __m256i p0 = _mm256_madd_epi16(scale_0, _mm256_maddubs_epi16(q5_0, q8_0));
            const __m256i p1 = _mm256_madd_epi16(scale_1, _mm256_maddubs_epi16(q5_1, q8_1));
            acc = _mm256_fmadd_ps(_mm256_set1_ps(dw * ad[2 * j]),     _mm256_cvtepi32_ps(p0), acc);
            acc = _mm256_fmadd_ps(_mm256_set1_ps(dw * ad[2 * j + 1]), _mm256_cvtepi32_ps(p1), acc);
        }
    }
    return hsum256(acc) + accm;
}

float dot_q4k_f32(const float* x, const block_q4_K* col, int nblocks) {
    float acc = 0.f;
    for (int b = 0; b < nblocks; ++b) {
        const block_q4_K& blk = col[b];
        const float d  = fp16_to_fp32(blk.d);
        const float dm = fp16_to_fp32(blk.dmin);
        const uint8_t* q = blk.qs;
        const float* xb = x + b * 256;
        int is = 0;
        for (int j = 0; j < 256; j += 64, q += 32, is += 2) {
            uint8_t sc, m;
            ref::get_scale_min_k4(is + 0, blk.scales, sc, m);
            const float d1 = d * float(sc), m1 = dm * float(m);
            ref::get_scale_min_k4(is + 1, blk.scales, sc, m);
            const float d2 = d * float(sc), m2 = dm * float(m);
            __m256 sum_aq_l = _mm256_setzero_ps();
            __m256 sum_a_l  = _mm256_setzero_ps();
            __m256 sum_aq_h = _mm256_setzero_ps();
            __m256 sum_a_h  = _mm256_setzero_ps();
            for (int l = 0; l < 32; l += 8) {
                const __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(q + l));
                const __m256i q32 = _mm256_cvtepu8_epi32(q8);
                const __m256 qf_l = _mm256_cvtepi32_ps(_mm256_and_si256(q32, _mm256_set1_epi32(0x0F)));
                const __m256 qf_h = _mm256_cvtepi32_ps(_mm256_srli_epi32(q32, 4));
                const __m256 a_l = _mm256_loadu_ps(xb + j + l);
                const __m256 a_h = _mm256_loadu_ps(xb + j + 32 + l);
                sum_aq_l = _mm256_fmadd_ps(a_l, qf_l, sum_aq_l);
                sum_a_l  = _mm256_add_ps(sum_a_l, a_l);
                sum_aq_h = _mm256_fmadd_ps(a_h, qf_h, sum_aq_h);
                sum_a_h  = _mm256_add_ps(sum_a_h, a_h);
            }
            acc += d1 * hsum256(sum_aq_l) - m1 * hsum256(sum_a_l);
            acc += d2 * hsum256(sum_aq_h) - m2 * hsum256(sum_a_h);
        }
    }
    return acc;
}

float dot_q5k_f32(const float* x, const block_q5_K* col, int nblocks) {
    float acc = 0.f;
    for (int b = 0; b < nblocks; ++b) {
        const block_q5_K& blk = col[b];
        const float d  = fp16_to_fp32(blk.d);
        const float dm = fp16_to_fp32(blk.dmin);
        const uint8_t* ql = blk.qs;
        const uint8_t* qh = blk.qh;
        const float* xb = x + b * 256;
        uint8_t u1 = 1, u2 = 2;
        int is = 0;
        for (int j = 0; j < 256; j += 64, ql += 32, is += 2, u1 <<= 2, u2 <<= 2) {
            uint8_t sc, m;
            ref::get_scale_min_k4(is + 0, blk.scales, sc, m);
            const float d1 = d * float(sc), m1 = dm * float(m);
            ref::get_scale_min_k4(is + 1, blk.scales, sc, m);
            const float d2 = d * float(sc), m2 = dm * float(m);
            __m256 sum_aq_l = _mm256_setzero_ps();
            __m256 sum_a_l  = _mm256_setzero_ps();
            __m256 sum_aq_h = _mm256_setzero_ps();
            __m256 sum_a_h  = _mm256_setzero_ps();
            for (int l = 0; l < 32; l += 8) {
                const __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(ql + l));
                const __m256i q32 = _mm256_cvtepu8_epi32(q8);
                alignas(32) int bits_l[8], bits_h[8];
                for (int t = 0; t < 8; ++t) {
                    bits_l[t] = (qh[l + t] & u1) ? 16 : 0;
                    bits_h[t] = (qh[l + t] & u2) ? 16 : 0;
                }
                const __m256 qf_l = _mm256_cvtepi32_ps(
                    _mm256_add_epi32(_mm256_and_si256(q32, _mm256_set1_epi32(0x0F)),
                                     _mm256_load_si256(reinterpret_cast<const __m256i*>(bits_l))));
                const __m256 qf_h = _mm256_cvtepi32_ps(
                    _mm256_add_epi32(_mm256_srli_epi32(q32, 4),
                                     _mm256_load_si256(reinterpret_cast<const __m256i*>(bits_h))));
                const __m256 a_l = _mm256_loadu_ps(xb + j + l);
                const __m256 a_h = _mm256_loadu_ps(xb + j + 32 + l);
                sum_aq_l = _mm256_fmadd_ps(a_l, qf_l, sum_aq_l);
                sum_a_l  = _mm256_add_ps(sum_a_l, a_l);
                sum_aq_h = _mm256_fmadd_ps(a_h, qf_h, sum_aq_h);
                sum_a_h  = _mm256_add_ps(sum_a_h, a_h);
            }
            acc += d1 * hsum256(sum_aq_l) - m1 * hsum256(sum_a_l);
            acc += d2 * hsum256(sum_aq_h) - m2 * hsum256(sum_a_h);
        }
    }
    return acc;
}
#else
float dot_q4k_f32(const float* x, const block_q4_K* col, int nblocks) {
    float acc = 0.f, w[256];
    for (int b = 0; b < nblocks; ++b) {
        ref::dequant_q4_K(&col[b], w);
        for (int i = 0; i < 256; ++i) acc += x[b * 256 + i] * w[i];
    }
    return acc;
}
float dot_q5k_f32(const float* x, const block_q5_K* col, int nblocks) {
    float acc = 0.f, w[256];
    for (int b = 0; b < nblocks; ++b) {
        ref::dequant_q5_K(&col[b], w);
        for (int i = 0; i < 256; ++i) acc += x[b * 256 + i] * w[i];
    }
    return acc;
}
float dot_q4k_q8k(const block_q4_K* col, const block_q8_K* x, int nblocks) {
    float acc = 0.f, w[256];
    for (int b = 0; b < nblocks; ++b) {
        ref::dequant_q4_K(&col[b], w);
        const float xd = x[b].d;
        for (int i = 0; i < 256; ++i) acc += w[i] * xd * float(x[b].qs[i]);
    }
    return acc;
}
float dot_q5k_q8k(const block_q5_K* col, const block_q8_K* x, int nblocks) {
    float acc = 0.f, w[256];
    for (int b = 0; b < nblocks; ++b) {
        ref::dequant_q5_K(&col[b], w);
        const float xd = x[b].d;
        for (int i = 0; i < 256; ++i) acc += w[i] * xd * float(x[b].qs[i]);
    }
    return acc;
}
float dot_q4k_q8s(const block_q4_K* col, const block_q8_Ks* x, int nblocks) {
    float acc = 0.f, w[256];
    for (int b = 0; b < nblocks; ++b) {
        ref::dequant_q4_K(&col[b], w);
        for (int i = 0; i < 256; ++i) acc += w[i] * x[b].d[i >> 5] * float(x[b].qs[i]);
    }
    return acc;
}
float dot_q5k_q8s(const block_q5_K* col, const block_q8_Ks* x, int nblocks) {
    float acc = 0.f, w[256];
    for (int b = 0; b < nblocks; ++b) {
        ref::dequant_q5_K(&col[b], w);
        for (int i = 0; i < 256; ++i) acc += w[i] * x[b].d[i >> 5] * float(x[b].qs[i]);
    }
    return acc;
}
#endif

}  // namespace

void cpu_quantize_q8_K(const float* x, block_q8_K* y, int k) {
    const int nb = k / 256;
    for (int i = 0; i < nb; ++i) {
        float amax = 0.f;
        for (int j = 0; j < 256; ++j) amax = std::max(amax, std::fabs(x[j]));
        const float iscale = amax > 0.f ? 127.f / amax : 0.f;
        y[i].d = amax > 0.f ? amax / 127.f : 0.f;
        for (int j = 0; j < 256; ++j) {
            int iv = int(std::round(x[j] * iscale));
            if (iv > 127) iv = 127;
            if (iv < -127) iv = -127;
            y[i].qs[j] = int8_t(iv);
        }
        for (int j = 0; j < 16; ++j) {
            int s = 0;
            for (int ii = 0; ii < 16; ++ii) s += y[i].qs[j * 16 + ii];
            y[i].bsums[j] = int16_t(s);
        }
        x += 256;
    }
}

void cpu_gemv_q4k_f32(const float* x, const block_q4_K* W, float* y,
                      uint32_t K, uint32_t N) {
    const int bpc = int(K / 256);
    parallel_n(int(N), [&](int n) {
        y[n] = dot_q4k_f32(x, W + size_t(n) * bpc, bpc);
    });
}
void cpu_gemv_q4k_q8(const block_q8_K* x, const block_q4_K* W, float* y,
                     uint32_t K, uint32_t N) {
    const int bpc = int(K / 256);
    parallel_n(int(N), [&](int n) {
        y[n] = dot_q4k_q8k(W + size_t(n) * bpc, x, bpc);
    });
}
void cpu_gemv_q5k_f32(const float* x, const block_q5_K* W, float* y,
                      uint32_t K, uint32_t N) {
    const int bpc = int(K / 256);
    parallel_n(int(N), [&](int n) {
        y[n] = dot_q5k_f32(x, W + size_t(n) * bpc, bpc);
    });
}
void cpu_gemv_q5k_q8(const block_q8_K* x, const block_q5_K* W, float* y,
                     uint32_t K, uint32_t N) {
    const int bpc = int(K / 256);
    parallel_n(int(N), [&](int n) {
        y[n] = dot_q5k_q8k(W + size_t(n) * bpc, x, bpc);
    });
}

void cpu_moe_expert_q8_nt(const float* x,
                          const block_q4_K* gate, const block_q4_K* up,
                          const block_q5_K* down,
                          uint32_t H, uint32_t EF, float clamp, float* y, int nthreads) {
    std::vector<block_q8_K> xq(H / 256), mq(EF / 256);
    std::vector<float> yg(EF), yu(EF), mid(EF);
    cpu_quantize_q8_K(x, xq.data(), int(H));
    const int bpc = int(H / 256);
    parallel_n_t(int(EF), nthreads, [&](int n) {
        yg[n] = dot_q4k_q8k(gate + size_t(n) * bpc, xq.data(), bpc);
        yu[n] = dot_q4k_q8k(up + size_t(n) * bpc, xq.data(), bpc);
    });
    auto sigmoid = [](float v) { return 1.f / (1.f + std::exp(-v)); };
    for (uint32_t i = 0; i < EF; ++i) {
        const float g = std::min(yg[i], clamp);
        const float u = std::min(std::max(yu[i], -clamp), clamp);
        mid[i] = (g * sigmoid(g)) * u;
    }
    cpu_quantize_q8_K(mid.data(), mq.data(), int(EF));
    const int dbpc = int(EF / 256);
    parallel_n_t(int(H), nthreads, [&](int n) {
        y[n] = dot_q5k_q8k(down + size_t(n) * dbpc, mq.data(), dbpc);
    });
}
void cpu_moe_expert_q8(const float* x,
                       const block_q4_K* gate, const block_q4_K* up,
                       const block_q5_K* down,
                       uint32_t H, uint32_t EF, float clamp, float* y) {
    cpu_moe_expert_q8_nt(x, gate, up, down, H, EF, clamp, y, 8);
}

void cpu_quantize_q8_Ks(const float* x, block_q8_Ks* y, int k) {
    const int nb = k / 256;
    for (int i = 0; i < nb; ++i) {
        for (int j = 0; j < 8; ++j) {
            const float* xs = x + j * 32;
            float amax = 0.f;
            for (int t = 0; t < 32; ++t) amax = std::max(amax, std::fabs(xs[t]));
            const float iscale = amax > 0.f ? 127.f / amax : 0.f;
            y[i].d[j] = amax > 0.f ? amax / 127.f : 0.f;
            for (int t = 0; t < 32; ++t) {
                int iv = int(std::round(xs[t] * iscale));
                if (iv > 127) iv = 127;
                if (iv < -127) iv = -127;
                y[i].qs[j * 32 + t] = int8_t(iv);
            }
        }
        for (int j = 0; j < 16; ++j) {
            int s16 = 0;
            for (int t = 0; t < 16; ++t) s16 += y[i].qs[j * 16 + t];
            y[i].bsums[j] = int16_t(s16);
        }
        x += 256;
    }
}

void cpu_moe_expert_q8s_nt(const float* x,
                           const block_q4_K* gate, const block_q4_K* up,
                           const block_q5_K* down,
                           uint32_t H, uint32_t EF, float clamp, float* y, int nthreads) {
    std::vector<block_q8_Ks> xq(H / 256), mq(EF / 256);
    std::vector<float> yg(EF), yu(EF), mid(EF);
    cpu_quantize_q8_Ks(x, xq.data(), int(H));
    const int bpc = int(H / 256);
    parallel_n_t(int(EF), nthreads, [&](int n) {
        yg[n] = dot_q4k_q8s(gate + size_t(n) * bpc, xq.data(), bpc);
        yu[n] = dot_q4k_q8s(up + size_t(n) * bpc, xq.data(), bpc);
    });
    auto sigmoid = [](float v) { return 1.f / (1.f + std::exp(-v)); };
    for (uint32_t i = 0; i < EF; ++i) {
        const float g = std::min(yg[i], clamp);
        const float u = std::min(std::max(yu[i], -clamp), clamp);
        mid[i] = (g * sigmoid(g)) * u;
    }
    cpu_quantize_q8_Ks(mid.data(), mq.data(), int(EF));
    const int dbpc = int(EF / 256);
    parallel_n_t(int(H), nthreads, [&](int n) {
        y[n] = dot_q5k_q8s(down + size_t(n) * dbpc, mq.data(), dbpc);
    });
}

}  // namespace ie
