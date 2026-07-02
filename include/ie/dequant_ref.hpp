// include/ie/dequant_ref.hpp — host-side reference dequant.
//
// Transcribed *verbatim* from `ggml-quants.c` and research/03 §2 to be the
// numerical golden against which device kernels are validated. Written for
// clarity, not speed — runs once per unit test on the host.
//
// Rule: change ANYTHING here only if research/03 changes. This is a contract.

#pragma once

#include "ie/quant_blocks.hpp"

#include <cstddef>
#include <cstdint>

namespace ie::ref {

// Q4_K / Q5_K — unpack one of the 8 (scale, min) pairs from the 12-byte
// `scales[]` field. See research/03 §2.5.
inline void get_scale_min_k4(int j, const uint8_t* q,
                             uint8_t& sc_out, uint8_t& m_out) noexcept {
    if (j < 4) {
        sc_out = q[j]     & 0x3F;
        m_out  = q[j + 4] & 0x3F;
    } else {
        sc_out = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m_out  = (q[j + 4] >>   4) | ((q[j - 0] >> 6) << 4);
    }
}

// ---- Q8_0: 32 elements ----
inline void dequant_q8_0(const block_q8_0* b, float* y) noexcept {
    const float d = fp16_to_fp32(b->d);
    for (int i = 0; i < 32; ++i) y[i] = d * float(b->qs[i]);
}

// ---- Q4_K: 256 elements ----
inline void dequant_q4_K(const block_q4_K* b, float* y) noexcept {
    const float d   = fp16_to_fp32(b->d);
    const float dm  = fp16_to_fp32(b->dmin);
    const uint8_t* q = b->qs;
    int is = 0;
    for (int j = 0; j < 256; j += 64, q += 32, is += 2) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, b->scales, sc, m);
        const float d1 = d * sc, m1 = dm * m;
        for (int l = 0; l < 32; ++l) y[j + l]      = d1 * (q[l] & 0x0F) - m1;

        get_scale_min_k4(is + 1, b->scales, sc, m);
        const float d2 = d * sc, m2 = dm * m;
        for (int l = 0; l < 32; ++l) y[j + 32 + l] = d2 * (q[l] >>   4) - m2;
    }
}

// ---- Q5_K: 256 elements ----
inline void dequant_q5_K(const block_q5_K* b, float* y) noexcept {
    const float d  = fp16_to_fp32(b->d);
    const float dm = fp16_to_fp32(b->dmin);
    const uint8_t* ql = b->qs;
    const uint8_t* qh = b->qh;
    uint8_t u1 = 1, u2 = 2;
    int is = 0;
    for (int j = 0; j < 256; j += 64, ql += 32, is += 2, u1 <<= 2, u2 <<= 2) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, b->scales, sc, m);
        const float d1 = d * sc, m1 = dm * m;
        for (int l = 0; l < 32; ++l)
            y[j + l]      = d1 * ((ql[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1;

        get_scale_min_k4(is + 1, b->scales, sc, m);
        const float d2 = d * sc, m2 = dm * m;
        for (int l = 0; l < 32; ++l)
            y[j + 32 + l] = d2 * ((ql[l] >>   4) + ((qh[l] & u2) ? 16 : 0)) - m2;
    }
}

// ---- Q6_K: 256 elements ----
inline void dequant_q6_K(const block_q6_K* b, float* y) noexcept {
    const float d = fp16_to_fp32(b->d);
    const uint8_t* ql = b->ql;
    const uint8_t* qh = b->qh;
    const int8_t*  sc = b->scales;
    for (int n = 0; n < 256; n += 128, y += 128, ql += 64, qh += 32, sc += 8) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int8_t q1 = int8_t((ql[l]      & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = int8_t((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = int8_t((ql[l]      >>   4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = int8_t((ql[l + 32] >>   4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l +  0] = d * sc[is + 0] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
        }
    }
}

// Vectorized helpers: dequant a contiguous packed buffer of N elements into
// floats. The packed buffer must be a whole number of blocks.
inline void dequant_q8_0_buffer(const void* packed, size_t n, float* out) {
    auto* b = static_cast<const block_q8_0*>(packed);
    for (size_t i = 0; i < n / 32; ++i) dequant_q8_0(&b[i], out + i * 32);
}
inline void dequant_q4_K_buffer(const void* packed, size_t n, float* out) {
    auto* b = static_cast<const block_q4_K*>(packed);
    for (size_t i = 0; i < n / 256; ++i) dequant_q4_K(&b[i], out + i * 256);
}
inline void dequant_q5_K_buffer(const void* packed, size_t n, float* out) {
    auto* b = static_cast<const block_q5_K*>(packed);
    for (size_t i = 0; i < n / 256; ++i) dequant_q5_K(&b[i], out + i * 256);
}
inline void dequant_q6_K_buffer(const void* packed, size_t n, float* out) {
    auto* b = static_cast<const block_q6_K*>(packed);
    for (size_t i = 0; i < n / 256; ++i) dequant_q6_K(&b[i], out + i * 256);
}

}  // namespace ie::ref
