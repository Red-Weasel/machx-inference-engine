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
// Q2_0 — 128 elements, 34 bytes, 2.125 bits/w. Prism ternary 2-bit
// (llama.cpp-prism, ggml_type 42). One FP16 scale per block (d = amax over the
// block), 2 bits/weight in NATURAL order, LSB-first within each byte:
//   byte j/4 holds weight j at bit-offset (j%4)*2; code c ∈ {0,1,2,3}
//   y[j] = (c - 1) * d ∈ { -d, 0, +d, +2d }.
// (Byte 0 = w0,w1,w2,w3 ; byte 1 = w4..w7 ; ...) Verified vs ggml-common.h
// block_q2_0 + quantize_row_q2_0_ref / dequantize_row_q2_0.
// ------------------------------------------------------------
struct alignas(2) block_q2_0 {
    gguf_half d;        // FP16 scale (= amax over the 128-element block)
    uint8_t   qs[32];   // 128 2-bit quants, natural order, LSB-first per byte
};
static_assert(sizeof(block_q2_0) == 34, "block_q2_0 must be 34 bytes");

// ------------------------------------------------------------
// Q1_0 — 128 elements, 18 bytes, 1.125 bits/w. Prism 1-bit sign-only
// (llama.cpp-prism, ggml_type 41). One FP16 scale per block (d = mean-abs
// over the block — NOT amax), 1 bit/weight in NATURAL order, LSB-first:
//   byte j/8 holds weight j at bit (j%8); y[j] = bit ? +d : -d.
// No zero code. Verified vs ggml-common.h block_q1_0 +
// quantize_row_q1_0_ref / dequantize_row_q1_0.
// ------------------------------------------------------------
struct alignas(2) block_q1_0 {
    gguf_half d;        // FP16 scale (= mean-abs over the 128-element block)
    uint8_t   qs[16];   // 128 1-bit signs, natural order, LSB-first per byte
};
static_assert(sizeof(block_q1_0) == 18, "block_q1_0 must be 18 bytes");

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

// ============================================================
// Extended k-quant / legacy layouts (Increment 2 universal-loader coverage).
// Transcribed byte-for-byte from llama.cpp ggml-common.h. Each `dm` union in
// ggml is flattened to two ggml_half fields (d, m/dmin) — identical bytes.
// ============================================================

// ------------------------------------------------------------
// Q4_1 — 32 elements, 20 bytes, 5.0 bits/w. Asymmetric 4-bit: per-block scale
// d + min m; nibble layout == Q4_0 (low nibble = weight j, high = weight j+16).
//   y[j]    = (qs[j] & 0x0F) * d + m
//   y[j+16] = (qs[j] >>  4)  * d + m
// ------------------------------------------------------------
struct alignas(2) block_q4_1 {
    gguf_half d;        // FP16 scale
    gguf_half m;        // FP16 min
    uint8_t   qs[16];   // 32 4-bit quants
};
static_assert(sizeof(block_q4_1) == 20, "block_q4_1 must be 20 bytes");

// ------------------------------------------------------------
// Q5_0 — 32 elements, 22 bytes, 5.5 bits/w. Symmetric 5-bit (offset 16): FP16
// scale + 4-byte qh carrying the 5th bit of each quant. The 5th bit for element
// j is bit j of the little-endian uint32 qh; for element j+16 it is bit j+16.
//   xh0 = ((qh >> (j))    << 4) & 0x10 ;  x0 = ((qs[j] & 0x0F) | xh0) - 16
//   xh1 = ((qh >> (j+12))      ) & 0x10 ;  x1 = ((qs[j] >>  4)  | xh1) - 16
//   y[j] = x0*d ;  y[j+16] = x1*d
// ------------------------------------------------------------
struct alignas(2) block_q5_0 {
    gguf_half d;        // FP16 scale
    uint8_t   qh[4];    // 5th bit of each quant (little-endian uint32)
    uint8_t   qs[16];   // low 4 bits of 32 quants
};
static_assert(sizeof(block_q5_0) == 22, "block_q5_0 must be 22 bytes");

// ------------------------------------------------------------
// Q5_1 — 32 elements, 24 bytes, 6.0 bits/w. Asymmetric 5-bit: scale d + min m +
// 4-byte qh (5th bit). Same qh bit mapping as Q5_0, but unsigned (no -16) + min.
//   x0 = (qs[j] & 0x0F) | xh0 ;  x1 = (qs[j] >> 4) | xh1
//   y[j] = x0*d + m ;  y[j+16] = x1*d + m
// ------------------------------------------------------------
struct alignas(2) block_q5_1 {
    gguf_half d;        // FP16 scale
    gguf_half m;        // FP16 min
    uint8_t   qh[4];    // 5th bit of each quant
    uint8_t   qs[16];   // low 4 bits of 32 quants
};
static_assert(sizeof(block_q5_1) == 24, "block_q5_1 must be 24 bytes");

// ------------------------------------------------------------
// Q2_K — 256 elements, 84 bytes, 2.625 bits/w. 16 sub-blocks of 16 elements.
// Each sub-block has a 4-bit scale + 4-bit min packed in scales[]; the super
// d/dmin (FP16) scale those. NOTE: d/dmin sit AFTER the quants in this layout.
//   sc = scales[is] ;  dl = d*(sc & 0xF) ;  ml = dmin*(sc >> 4)
//   y = dl * ((q[l] >> shift) & 3) - ml
// ------------------------------------------------------------
struct alignas(2) block_q2_K {
    uint8_t   scales[kQK_K / 16];  // 16 bytes: 4-bit scale + 4-bit min per sub-block
    uint8_t   qs[kQK_K / 4];       // 64 bytes: 2-bit quants
    gguf_half d;                   // FP16 super-scale (for the scales)
    gguf_half dmin;                // FP16 super-scale (for the mins)
};
static_assert(sizeof(block_q2_K) == 84, "block_q2_K must be 84 bytes");

// ------------------------------------------------------------
// Q3_K — 256 elements, 110 bytes, 3.4375 bits/w. Signed 3-bit (2 low bits in
// qs[], 1 inverted high bit in hmask[]); 16 sub-blocks with 6-bit scales packed
// into 12 bytes (unpacked via the kmask shuffle, offset 32). Single super d.
//   q = ((qs[l] >> shift) & 3) - ((hmask[l] & m) ? 0 : 4)   // high bit INVERTED
//   y = d * (scale6[is] - 32) * q
// ------------------------------------------------------------
struct alignas(2) block_q3_K {
    uint8_t   hmask[kQK_K / 8];    // 32 bytes: inverted high bit per quant
    uint8_t   qs[kQK_K / 4];       // 64 bytes: low 2 bits per quant
    uint8_t   scales[12];          // 12 bytes: 6-bit scales (packed)
    gguf_half d;                   // FP16 super-scale
};
static_assert(sizeof(block_q3_K) == 110, "block_q3_K must be 110 bytes");

// ------------------------------------------------------------
// IQ4 non-linear codebook — 4-bit indices into a 16-value signed LUT (shared by
// IQ4_NL and IQ4_XS). Verbatim from llama.cpp ggml-common.h kvalues_iq4nl.
// ------------------------------------------------------------
inline constexpr int8_t kIQ4NLValues[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

// ------------------------------------------------------------
// IQ4_NL — 32 elements, 18 bytes, 4.5 bits/w. One FP16 scale + 32 4-bit LUT
// indices (nibble layout == Q4_0: qs[j] low = weight j, high = weight j+16).
//   y[j]    = d * kIQ4NLValues[qs[j] & 0x0F]
//   y[j+16] = d * kIQ4NLValues[qs[j] >>  4]
// ------------------------------------------------------------
struct alignas(2) block_iq4_nl {
    gguf_half d;        // FP16 scale
    uint8_t   qs[16];   // 32 4-bit LUT indices
};
static_assert(sizeof(block_iq4_nl) == 18, "block_iq4_nl must be 18 bytes");

// ------------------------------------------------------------
// IQ4_XS — 256 elements, 136 bytes, ~4.25 bits/w. FP16 super-scale + eight 6-bit
// sub-block scales (4 low bits in scales_l[4], 2 high bits packed in scales_h),
// then 128 bytes of 4-bit LUT indices. Per 32-elem sub-block ib:
//   ls = (scales_l[ib/2] >> 4*(ib%2) & 0xF) | ((scales_h >> 2*ib & 3) << 4)
//   dl = d * (ls - 32) ;  y = dl * kIQ4NLValues[nibble]
// ------------------------------------------------------------
struct alignas(2) block_iq4_xs {
    gguf_half d;                    // FP16 super-scale
    uint16_t  scales_h;             // 2 high bits of each of the 8 sub-scales
    uint8_t   scales_l[kQK_K / 64]; // 4 bytes: 4 low bits of each sub-scale (2/byte)
    uint8_t   qs[kQK_K / 2];        // 128 bytes: 4-bit LUT indices
};
static_assert(sizeof(block_iq4_xs) == 136, "block_iq4_xs must be 136 bytes");

// ------------------------------------------------------------
// IQ2_XXS — 256 elements, 66 bytes, 2.0625 bits/w (Hy4-preview expert gate/up
// on 48 layers). Per 32-elem sub-block: qs holds 2+2 uint16 = aux32[0] (four
// 8-bit indices into the 256-entry uint64 grid, 8 magnitudes per entry) and
// aux32[1] (four 7-bit sign selectors + 4-bit sub-scale in bits 31..28):
//   db  = d * (0.5 + (aux32[1] >> 28)) * 0.25
//   y   = db * grid_byte * (ksigns bit ? -1 : +1)
// Magnitudes are {8, 25, 43} — nonzero, <= 62, so the signed weight fits int8
// and the W2A8 int-dot form is exact in the integer domain (same argument as
// IQ3_XXS below). Grid verbatim from llama.cpp ggml-common.h iq2xxs_grid.
// ------------------------------------------------------------
struct alignas(2) block_iq2_xxs {
    gguf_half d;             // FP16 super-scale
    uint16_t  qs[kQK_K / 8]; // 8 sub-blocks x (2 grid-index u16 + 2 sign/scale u16)
};
static_assert(sizeof(block_iq2_xxs) == 66, "block_iq2_xxs must be 66 bytes");

inline constexpr uint64_t kIQ2XXSGrid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

// ------------------------------------------------------------
// STQ1_0 — Tencent Hy4-preview ternary, 256 elements, 42 bytes, 1.3125 bits/w
// (ggml type 43 from the hy4-preview llama.cpp patch; expert gate/up on 29
// layers). Groups of 4 weights are STRIDE-16 within each 64-weight chunk:
// group g (chunk c = g/16, gloc = g%16) holds weights {c*64 + gloc + p*16 :
// p in 0..3}. Each group stores a 4-bit code + 1 sign bit; the codebook maps
// (sign, code) to a packed byte of four 2-bit lanes q, and w = (q - 1) * d
// with exactly one zero lane per group. Codebook verbatim from the patch.
// ------------------------------------------------------------
struct alignas(2) block_stq1_0 {
    uint8_t   qs[kQK_K / 8];    // 4-bit code per group of 4 (64 groups)
    uint8_t   sign[kQK_K / 32]; // 1 sign bit per group
    gguf_half d;                // FP16 scale (= amax)
};
static_assert(sizeof(block_stq1_0) == 42, "block_stq1_0 must be 42 bytes");

inline constexpr uint8_t kSTQ1Codebook[32] = {
    // sign = 0 (first non-zero lane is +1)
    0xA9, 0x89, 0x29, 0x09, 0xA6, 0x86, 0x26, 0x06,
    0x9A, 0x92, 0x1A, 0x12, 0x6A, 0x62, 0x4A, 0x42,
    // sign = 1 (every non-zero lane negated)
    0x01, 0x21, 0x81, 0xA1, 0x04, 0x24, 0x84, 0xA4,
    0x10, 0x18, 0x90, 0x98, 0x40, 0x48, 0x60, 0x68,
};

// ------------------------------------------------------------
// IQ3_XXS codebook — 8-bit indices into a 256-entry grid; each entry packs four
// unsigned magnitudes in its 4 bytes (byte j = magnitude j). Sign bits come from
// kSignsIQ2XS (7 packed sign bits -> an 8-bit mask) applied via kMaskIQ2XS.
// Verbatim from llama.cpp ggml-common.h iq3xxs_grid / ksigns_iq2xs / kmask_iq2xs
// (script-extracted, element counts asserted against the GGML_TABLE_BEGIN decls).
// ------------------------------------------------------------
inline constexpr uint32_t kIQ3XXSGrid[256] = {
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
};

inline constexpr uint8_t kSignsIQ2XS[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

inline constexpr uint8_t kMaskIQ2XS[8] = {
      1,   2,   4,   8,  16,  32,  64, 128,
};

// ------------------------------------------------------------
// IQ3_XXS — 256 elements, 98 bytes, 3.0625 bits/w. FP16 super-scale then 96
// bytes of qs: the first 64 are grid indices (8 per 32-elem sub-block), the last
// 32 are eight uint32 words, one per sub-block, holding a 4-bit sub-scale in
// bits 28..31 and four 7-bit sign selectors in bits 0..27. Per sub-block ib32:
//   db = d * (0.5 + (aux32 >> 28)) * 0.5
//   y  = db * grid_byte * (signs & kMaskIQ2XS[j] ? -1 : +1)
// ------------------------------------------------------------
struct alignas(2) block_iq3_xxs {
    gguf_half d;                      // FP16 super-scale
    uint8_t   qs[3 * kQK_K / 8];      // 96 B: 64 grid indices + 32 B scales/signs
};
static_assert(sizeof(block_iq3_xxs) == 98, "block_iq3_xxs must be 98 bytes");

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
