// include/ie/quant_blocks.hpp — packed-block layouts for GGUF k-quants.
//
// Verified byte-by-byte against research/03_quant_formats.md §2 and
// llama.cpp's ggml/src/ggml-common.h. These structs are read directly from
// the mmap'd GGUF tensor data — sizeof and layout MUST match the on-disk format.

#pragma once

#include <bit>
#include <cstdint>
#include <type_traits>

namespace ie {

inline constexpr int kQK_K = 256;       // K-quant super-block size
inline constexpr int kQK8_0 = 32;       // legacy Q8_0 block size
inline constexpr int kKScaleSize = 12;  // 6-bit packed scale/min table size

// fp16 stored on disk as little-endian uint16. We unpack to float at use site.
using gguf_half = uint16_t;

// ------------------------------------------------------------
// Q8_0 — 32 elements, 34 bytes, 8.5 bits/w
//   y[i] = d * qs[i]
// ------------------------------------------------------------
struct alignas(2) block_q8_0 {
    gguf_half d;       // FP16 scale
    int8_t    qs[32];  // signed 8-bit quants
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");

// ------------------------------------------------------------
// Q4_0 — 32 elements, 18 bytes, 4.5 bits/w. Symmetric 4-bit, one FP16 scale per
// block, no min. ggml nibble layout: qs[i] low nibble = weight i, high nibble =
// weight i+16 (i in 0..15).
//   y[i]    = d * ((qs[i] & 0x0F) - 8)
//   y[i+16] = d * ((qs[i] >>  4)  - 8)
// ------------------------------------------------------------
struct alignas(2) block_q4_0 {
    gguf_half d;        // FP16 scale
    uint8_t   qs[16];   // 32 4-bit quants
};
static_assert(sizeof(block_q4_0) == 18, "block_q4_0 must be 18 bytes");

// ------------------------------------------------------------
// MXFP4 — 32 elements, 17 bytes, ~4.25 bits/w. OpenAI gpt-oss MoE expert format.
// One E8M0 *half-scaled* shared exponent + 32 FP4 nibbles via a 16-value signed
// LUT. ggml nibble layout (== Q4_0): qs[j] low nibble = weight j, high = j+16.
//   d        = e8m0_half(e) = bit_cast<float>( e<2 ? 0x00200000u<<e : (e-1)<<23 )
//   LUT[16]  = {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12}   (signed)
//   y[j]     = LUT[qs[j] & 0x0F] * d ;   y[j+16] = LUT[qs[j] >> 4] * d
// Bit-exact with llama ggml_e8m0_to_fp32_half + kvalues_mxfp4 (no extra 0.5 —
// the "half" exponent already folds it). NEVER re-quantized in gpt-oss GGUFs.
// ------------------------------------------------------------
inline constexpr int kQK_MXFP4 = 32;
struct block_mxfp4 {
    uint8_t e;                       // E8M0 shared exponent (half-scaled)
    uint8_t qs[kQK_MXFP4 / 2];       // 16 bytes, 32 FP4 nibbles
};
static_assert(sizeof(block_mxfp4) == 17, "block_mxfp4 must be 17 bytes");

// ------------------------------------------------------------
// Q4_K — 256 elements, 144 bytes, 4.5 bits/w
//   super-scale `d`, super-min `dmin`, 8 sub-block (scale,min) packed into 12 bytes,
//   then 128 bytes of low/high-nibble paired 4-bit quants.
// ------------------------------------------------------------
struct alignas(2) block_q4_K {
    gguf_half d;                       // FP16 super-scale
    gguf_half dmin;                    // FP16 super-min
    uint8_t   scales[kKScaleSize];     // packed 6-bit scales+mins (see §2.5)
    uint8_t   qs[kQK_K / 2];           // 128 bytes, 4-bit packed
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K must be 144 bytes");

// ------------------------------------------------------------
// Q5_K — 256 elements, 176 bytes, 5.5 bits/w
//   Like Q4_K but with an extra 32-byte 'qh' that carries the 5th bit per element.
// ------------------------------------------------------------
struct alignas(2) block_q5_K {
    gguf_half d;
    gguf_half dmin;
    uint8_t   scales[kKScaleSize];
    uint8_t   qh[kQK_K / 8];           // 32 bytes, 5th bit per element
    uint8_t   qs[kQK_K / 2];           // 128 bytes, low 4 bits per element
};
static_assert(sizeof(block_q5_K) == 176, "block_q5_K must be 176 bytes");

// ------------------------------------------------------------
// Q6_K — 256 elements, 210 bytes, 6.5625 bits/w
//   Signed 6-bit (offset 32). Per-16-element int8 scales. Single super-scale.
// ------------------------------------------------------------
struct alignas(2) block_q6_K {
    uint8_t   ql[kQK_K / 2];   // 128 bytes, low 4 bits
    uint8_t   qh[kQK_K / 4];   //  64 bytes, high 2 bits
    int8_t    scales[kQK_K / 16]; // 16 bytes, signed per-16 scales
    gguf_half d;               //   2 bytes, FP16 super-scale
};
static_assert(sizeof(block_q6_K) == 210, "block_q6_K must be 210 bytes");

// ------------------------------------------------------------
// FP16 <-> FP32 conversion — IEEE 754 binary16 (sign,5-exp,10-mantissa).
// Pure host-side, no <cmath> dependency. constexpr-friendly.
// ------------------------------------------------------------
constexpr float fp16_to_fp32(uint16_t h) noexcept {
    // From "Float Toy" / Numpy's half.c. Handles subnormals, NaN, Inf.
    const uint32_t s = uint32_t(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu;
    uint32_t m =  h        & 0x3ffu;
    if (e == 0) {
        if (m == 0) return std::bit_cast<float>(s);                   // ±0
        // subnormal: normalize
        while ((m & 0x400u) == 0) { m <<= 1; e -= 1; }
        e += 1;
        m &= ~0x400u;
    } else if (e == 31) {
        const uint32_t r = s | 0x7f800000u | (m << 13);                // ±Inf or NaN
        return std::bit_cast<float>(r);
    }
    e += (127 - 15);
    const uint32_t r = s | (e << 23) | (m << 13);
    return std::bit_cast<float>(r);
}

// IEEE-754 binary16 round-to-nearest-even, the inverse of fp16_to_fp32.
inline uint16_t fp32_to_fp16(float f) {
    const uint32_t x    = std::bit_cast<uint32_t>(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t        exp  = int32_t((x >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mant = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFFu) == 0xFFu)                       // Inf / NaN
        return uint16_t(sign | 0x7C00u | (mant ? 0x200u : 0u));
    if (exp >= 0x1F) return uint16_t(sign | 0x7C00u);      // overflow → Inf
    if (exp <= 0) {                                         // subnormal / underflow
        if (exp < -10) return uint16_t(sign);
        const uint32_t m  = (mant | 0x800000u);
        const int      sh = 14 - exp;
        uint32_t       r  = m >> sh;
        const uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) ++r;
        return uint16_t(sign | r);
    }
    uint16_t       h   = uint16_t(sign | (uint32_t(exp) << 10) | (mant >> 13));
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1))) ++h;  // round-to-even
    return h;
}

// ------------------------------------------------------------
// Q8_1x — engine-internal ACTIVATION quantization block (P1a, 2026-06-09).
// 32 elements, 40 bytes.  Not a GGUF on-disk format: produced at runtime by
// quantize_q8_1() from fp16 activations and consumed by the integer-dot
// (dp4a) GEMV variants.  fp32 d/s (we generate them — no reason to round):
//   x[i] ≈ d * qs[i],   s = d * Σ qs[i]   (the Σq8 term the K-quant min/bias
//                                          correction needs per block)
// qs sits at offset 8 of a 40-byte block → every qs[] is 4-byte aligned for
// packed-uint32 dp4a loads.
// ------------------------------------------------------------
struct alignas(8) block_q8_1x {
    float  d;
    float  s;
    int8_t qs[32];
};
static_assert(sizeof(block_q8_1x) == 40, "block_q8_1x must be 40 bytes");

// ------------------------------------------------------------
// Q8_1s — MoE-prefill activation block (prefill-crown, 2026-06-10).
// 32 elements, 48 bytes.  Like block_q8_1x but the block sum is stored as
// TWO half sums so consumers with per-16-element weight scales (Q6_K) get
// exact min/offset corrections without re-deriving Σq8 via dp4a:
//   s0 = d · Σ qs[0..16),   s1 = d · Σ qs[16..32)     (s = s0 + s1)
// Header is one aligned float4 ({d, s0, s1, pad}); qs sits at offset 16 so
// 16/32 B vector loads of the quants stay aligned.
// ------------------------------------------------------------
struct alignas(16) block_q8_1s {
    float  d;
    float  s0;
    float  s1;
    float  pad;
    int8_t qs[32];
};
static_assert(sizeof(block_q8_1s) == 48, "block_q8_1s must be 48 bytes");

}  // namespace ie
