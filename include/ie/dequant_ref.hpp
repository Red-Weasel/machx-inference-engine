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

// ---- Q4_1: 32 elements (asymmetric 4-bit) ----
inline void dequant_q4_1(const block_q4_1* b, float* y) noexcept {
    const float d = fp16_to_fp32(b->d);
    const float m = fp16_to_fp32(b->m);
    for (int j = 0; j < 16; ++j) {
        y[j]      = float(b->qs[j] & 0x0F) * d + m;
        y[j + 16] = float(b->qs[j] >>   4) * d + m;
    }
}

// ---- Q2_0: 128 elements (Prism ternary 2-bit, single fp16 scale) ----
// Verbatim from ggml-quants.c dequantize_row_q2_0: code c = (qs[j/4] >> ((j%4)*2))
// & 3; y[j] = (c - 1) * d ∈ {-d, 0, +d, +2d}.
inline void dequant_q2_0(const block_q2_0* b, float* y) noexcept {
    const float d = fp16_to_fp32(b->d);
    for (int j = 0; j < 128; ++j) {
        const uint8_t c = (b->qs[j >> 2] >> ((j & 3) * 2)) & 0x03;
        y[j] = (int(c) - 1) * d;
    }
}

// ---- Q1_0: 128 elements (Prism 1-bit sign-only, single fp16 scale) ----
// Verbatim from ggml-quants.c dequantize_row_q1_0: bit b = (qs[j/8] >> (j%8)) & 1;
// y[j] = (2*b - 1) * d ∈ {-d, +d}. d = mean-abs over the block (set by the
// quantizer — NOT amax like Q2_0). No zero code.
inline void dequant_q1_0(const block_q1_0* b, float* y) noexcept {
    const float d = fp16_to_fp32(b->d);
    for (int j = 0; j < 128; ++j) {
        const uint8_t bit = (b->qs[j >> 3] >> (j & 7)) & 0x01;
        y[j] = (int(bit) * 2 - 1) * d;
    }
}

// ---- Q4_0: 32 elements (symmetric 4-bit, offset 8) ----
inline void dequant_q4_0(const block_q4_0* b, float* y) noexcept {
    const float d = fp16_to_fp32(b->d);
    for (int j = 0; j < 16; ++j) {
        y[j]      = float(int(b->qs[j] & 0x0F) - 8) * d;
        y[j + 16] = float(int(b->qs[j] >>   4) - 8) * d;
    }
}

// ---- Q5_0: 32 elements (symmetric 5-bit, offset 16) ----
inline void dequant_q5_0(const block_q5_0* b, float* y) noexcept {
    const float d = fp16_to_fp32(b->d);
    uint32_t qh;
    __builtin_memcpy(&qh, b->qh, sizeof(qh));   // little-endian 5th-bit field
    for (int j = 0; j < 16; ++j) {
        const uint8_t xh0 = uint8_t((qh >> (j +  0)) << 4) & 0x10;
        const uint8_t xh1 = uint8_t((qh >> (j + 12))     ) & 0x10;
        const int32_t x0  = ((b->qs[j] & 0x0F) | xh0) - 16;
        const int32_t x1  = ((b->qs[j] >>   4) | xh1) - 16;
        y[j]      = float(x0) * d;
        y[j + 16] = float(x1) * d;
    }
}

// ---- Q5_1: 32 elements (asymmetric 5-bit) ----
inline void dequant_q5_1(const block_q5_1* b, float* y) noexcept {
    const float d = fp16_to_fp32(b->d);
    const float m = fp16_to_fp32(b->m);
    uint32_t qh;
    __builtin_memcpy(&qh, b->qh, sizeof(qh));
    for (int j = 0; j < 16; ++j) {
        const uint8_t xh0 = uint8_t((qh >> (j +  0)) << 4) & 0x10;
        const uint8_t xh1 = uint8_t((qh >> (j + 12))     ) & 0x10;
        const int x0 = (b->qs[j] & 0x0F) | xh0;
        const int x1 = (b->qs[j] >>   4) | xh1;
        y[j]      = float(x0) * d + m;
        y[j + 16] = float(x1) * d + m;
    }
}

// ---- Q2_K: 256 elements ----
inline void dequant_q2_K(const block_q2_K* b, float* y) noexcept {
    const float d   = fp16_to_fp32(b->d);
    const float min = fp16_to_fp32(b->dmin);
    const uint8_t* q = b->qs;
    int is = 0;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            uint8_t sc = b->scales[is++];
            float dl = d * (sc & 0xF), ml = min * (sc >> 4);
            for (int l = 0; l < 16; ++l) *y++ = dl * float(int8_t((q[l]      >> shift) & 3)) - ml;
            sc = b->scales[is++];
            dl = d * (sc & 0xF); ml = min * (sc >> 4);
            for (int l = 0; l < 16; ++l) *y++ = dl * float(int8_t((q[l + 16] >> shift) & 3)) - ml;
            shift += 2;
        }
        q += 32;
    }
}

// ---- Q3_K: 256 elements (6-bit scales unpacked via the kmask shuffle) ----
inline void dequant_q3_K(const block_q3_K* b, float* y) noexcept {
    const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
    const float d_all = fp16_to_fp32(b->d);
    const uint8_t* q  = b->qs;
    const uint8_t* hm = b->hmask;
    uint8_t m = 1;
    __builtin_memcpy(aux, b->scales, 12);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2)        | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2)        | (((tmp >> 2) & kmask1) << 4);
    int is = 0;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            float dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l)
                *y++ = dl * float(int8_t((q[l +  0] >> shift) & 3) - ((hm[l +  0] & m) ? 0 : 4));
            dl = d_all * (scales[is++] - 32);
            for (int l = 0; l < 16; ++l)
                *y++ = dl * float(int8_t((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            shift += 2;
            m <<= 1;
        }
        q += 32;
    }
}

// ---- IQ4_NL: 32 elements (non-linear codebook) ----
inline void dequant_iq4_nl(const block_iq4_nl* b, float* y) noexcept {
    const float d = fp16_to_fp32(b->d);
    for (int j = 0; j < 16; ++j) {
        y[j]      = d * float(kIQ4NLValues[b->qs[j] & 0x0F]);
        y[j + 16] = d * float(kIQ4NLValues[b->qs[j] >>   4]);
    }
}

// ---- IQ4_XS: 256 elements (8 sub-blocks, 6-bit scales + non-linear codebook) ----
inline void dequant_iq4_xs(const block_iq4_xs* b, float* y) noexcept {
    const float d = fp16_to_fp32(b->d);
    const uint8_t* qs = b->qs;
    for (int ib = 0; ib < 8; ++ib) {   // QK_K/32 = 8 sub-blocks of 32
        const int ls = ((b->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) |
                       (((b->scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (ls - 32);
        for (int j = 0; j < 16; ++j) {
            y[j]      = dl * float(kIQ4NLValues[qs[j] & 0x0F]);
            y[j + 16] = dl * float(kIQ4NLValues[qs[j] >>   4]);
        }
        y  += 32;
        qs += 16;
    }
}

// ---- IQ2_XXS: 256 elements (8 sub-blocks, 4-bit sub-scale + signed u64 grid).
// Transcribed from ggml dequantize_row_iq2_xxs (llama.cpp ggml-quants.c). ----
inline void dequant_iq2_xxs(const block_iq2_xxs* b, float* y) noexcept {
    const float d = fp16_to_fp32(b->d);
    uint32_t aux32[2];
    for (int ib32 = 0; ib32 < 8; ++ib32) {       // kQK_K/32 sub-blocks
        __builtin_memcpy(aux32, b->qs + 4 * ib32, 2 * sizeof(uint32_t));
        const uint8_t* aux8 = reinterpret_cast<const uint8_t*>(aux32);
        const float db = d * (0.5f + float(aux32[1] >> 28)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
            const uint8_t* grid = reinterpret_cast<const uint8_t*>(&kIQ2XXSGrid[aux8[l]]);
            const uint8_t signs = kSignsIQ2XS[(aux32[1] >> (7 * l)) & 127];
            for (int j = 0; j < 8; ++j)
                y[j] = db * float(grid[j]) * ((signs & kMaskIQ2XS[j]) ? -1.f : 1.f);
            y += 8;
        }
    }
}

// ---- STQ1_0: 256 elements (64 stride-16 groups of 4 ternary weights).
// Transcribed from the hy4-preview patch dequantize_row_stq1_0. ----
inline void dequant_stq1_0(const block_stq1_0* b, float* y) noexcept {
    const float d = fp16_to_fp32(b->d);
    for (int g = 0; g < 64; ++g) {               // kQK_K/4 groups
        const uint8_t code  = uint8_t((b->qs[g / 2] >> (4 * (g & 1))) & 0x0F);
        const uint8_t sign  = uint8_t((b->sign[g / 8] >> (g % 8)) & 0x01);
        const uint8_t qpack = kSTQ1Codebook[(uint32_t(sign) << 4) | code];
        const int chunk = g / 16;
        const int gloc  = g % 16;
        for (int p = 0; p < 4; ++p) {
            const int qv = (qpack >> (2 * p)) & 0x3;
            y[chunk * 64 + gloc + p * 16] = float(qv - 1) * d;
        }
    }
}

// ---- IQ3_XXS: 256 elements (8 sub-blocks, 4-bit sub-scale + signed grid) ----
inline void dequant_iq3_xxs(const block_iq3_xxs* b, float* y) noexcept {
    const float d = fp16_to_fp32(b->d);
    const uint8_t* qs = b->qs;
    const uint8_t* scales_and_signs = qs + 64;   // QK_K/4 grid indices, then 8x u32
    for (int ib32 = 0; ib32 < 8; ++ib32) {       // QK_K/32 = 8 sub-blocks of 32
        uint32_t aux32;
        __builtin_memcpy(&aux32, scales_and_signs + 4 * ib32, sizeof(uint32_t));
        const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
        for (int l = 0; l < 4; ++l) {
            const uint8_t  signs = kSignsIQ2XS[(aux32 >> 7 * l) & 127];
            const uint32_t grid1 = kIQ3XXSGrid[qs[2 * l + 0]];
            const uint32_t grid2 = kIQ3XXSGrid[qs[2 * l + 1]];
            for (int j = 0; j < 4; ++j) {
                y[j    ] = db * float((grid1 >> (8 * j)) & 0xFF) *
                           ((signs & kMaskIQ2XS[j    ]) ? -1.f : 1.f);
                y[j + 4] = db * float((grid2 >> (8 * j)) & 0xFF) *
                           ((signs & kMaskIQ2XS[j + 4]) ? -1.f : 1.f);
            }
            y += 8;
        }
        qs += 8;
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
inline void dequant_q4_0_buffer(const void* packed, size_t n, float* out) {
    auto* b = static_cast<const block_q4_0*>(packed);
    for (size_t i = 0; i < n / 32; ++i) dequant_q4_0(&b[i], out + i * 32);
}
inline void dequant_q2_0_buffer(const void* packed, size_t n, float* out) {
    auto* b = static_cast<const block_q2_0*>(packed);
    for (size_t i = 0; i < n / 128; ++i) dequant_q2_0(&b[i], out + i * 128);
}
inline void dequant_q1_0_buffer(const void* packed, size_t n, float* out) {
    auto* b = static_cast<const block_q1_0*>(packed);
    for (size_t i = 0; i < n / 128; ++i) dequant_q1_0(&b[i], out + i * 128);
}
inline void dequant_q4_1_buffer(const void* packed, size_t n, float* out) {
    auto* b = static_cast<const block_q4_1*>(packed);
    for (size_t i = 0; i < n / 32; ++i) dequant_q4_1(&b[i], out + i * 32);
}
inline void dequant_q5_0_buffer(const void* packed, size_t n, float* out) {
    auto* b = static_cast<const block_q5_0*>(packed);
    for (size_t i = 0; i < n / 32; ++i) dequant_q5_0(&b[i], out + i * 32);
}
inline void dequant_q5_1_buffer(const void* packed, size_t n, float* out) {
    auto* b = static_cast<const block_q5_1*>(packed);
    for (size_t i = 0; i < n / 32; ++i) dequant_q5_1(&b[i], out + i * 32);
}
inline void dequant_q2_K_buffer(const void* packed, size_t n, float* out) {
    auto* b = static_cast<const block_q2_K*>(packed);
    for (size_t i = 0; i < n / 256; ++i) dequant_q2_K(&b[i], out + i * 256);
}
inline void dequant_q3_K_buffer(const void* packed, size_t n, float* out) {
    auto* b = static_cast<const block_q3_K*>(packed);
    for (size_t i = 0; i < n / 256; ++i) dequant_q3_K(&b[i], out + i * 256);
}
inline void dequant_iq4_nl_buffer(const void* packed, size_t n, float* out) {
    auto* b = static_cast<const block_iq4_nl*>(packed);
    for (size_t i = 0; i < n / 32; ++i) dequant_iq4_nl(&b[i], out + i * 32);
}
inline void dequant_iq4_xs_buffer(const void* packed, size_t n, float* out) {
    auto* b = static_cast<const block_iq4_xs*>(packed);
    for (size_t i = 0; i < n / 256; ++i) dequant_iq4_xs(&b[i], out + i * 256);
}
inline void dequant_iq2_xxs_buffer(const void* packed, size_t n, float* out) {
    auto* b = static_cast<const block_iq2_xxs*>(packed);
    for (size_t i = 0; i < n / 256; ++i) dequant_iq2_xxs(&b[i], out + i * 256);
}
inline void dequant_stq1_0_buffer(const void* packed, size_t n, float* out) {
    auto* b = static_cast<const block_stq1_0*>(packed);
    for (size_t i = 0; i < n / 256; ++i) dequant_stq1_0(&b[i], out + i * 256);
}
inline void dequant_iq3_xxs_buffer(const void* packed, size_t n, float* out) {
    auto* b = static_cast<const block_iq3_xxs*>(packed);
    for (size_t i = 0; i < n / 256; ++i) dequant_iq3_xxs(&b[i], out + i * 256);
}

}  // namespace ie::ref
