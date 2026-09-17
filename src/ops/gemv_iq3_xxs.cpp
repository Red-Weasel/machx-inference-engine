// src/ops/gemv_iq3_xxs.cpp — device-side IQ3_XXS decode GEMV over a load-time
// SoA repack of the packed weights.  Phase 4 (DeepSeek-V4-Flash expert path).
//
// WHY: before this file the ONLY IQ3_XXS path in the engine was the host
// reference `ie::ref::dequant_iq3_xxs_buffer` (bit-exact vs ggml, but CPU and
// fp32-expanding).  DeepSeek-V4-Flash's routed experts are 120.4 GB of IQ3_XXS;
// expanding them to F16 would be ~2.6x that.  The weights must be consumed in
// their packed form on the device.
//
// THE LAYOUT TRAP (same one gemv_mxfp4.cpp hit): the native block_iq3_xxs is
// 98 bytes, so a bank of them is byte-packed and every cross-lane read is
// 98-strided and unaligned.  The proven fix (gemv_q6_soa / gemv_q4_soa /
// gemv_mxfp4 recipe) is a load-time SoA repack that de-interleaves the block
// into three aligned planes.  A block holds, in order:
//     d          fp16 super-scale                                     2 B
//     qs[0..63]  8 grid indices for each of the 8 32-elem sub-blocks  64 B
//     qs[64..95] 8 uint32 words, one per sub-block:                   32 B
//                  bits 31..28 = 4-bit sub-scale
//                  bits 27..0  = four 7-bit sign selectors (l = 0..3)
// so the split is exact — 2 + 64 + 32 == 98, ZERO storage overhead, 3.0625 bpw
// preserved:
//     gp[n*(K/4)   + sb*8 + i]  grid indices, 8 B per 32-elem sub-block sb
//     ap[n*(K/32)  + sb]        the scale/sign uint32 of sub-block sb
//     dp[n*(K/256) + sb/8]      the fp16 super-scale of sub-block sb's block
// Each lane reads ONE aligned 8-byte grid chunk + one aligned uint32 per
// sub-block; consecutive lanes stride whole sub-blocks, so a sub-group's grid
// reads cover 16*8 = 128 contiguous bytes (coalesced).
//
// NUMERICS.  The decode is transcribed from ie::ref::dequant_iq3_xxs
// (include/ie/dequant_ref.hpp), which is itself bit-exact vs ggml on 51.2M real
// weights:
//     db = d * (0.5 + (aux >> 28)) * 0.5
//     signs = kSignsIQ2XS[(aux >> 7l) & 127]
//     w = db * grid_byte * (signs & (1 << j) ? -1 : +1)
// The grid bytes are the fixed magnitudes {4,12,20,28,36,44,52,62}, so a signed
// weight fits an int8 exactly (|w| <= 62) and the W3A8 int-dot form is EXACT in
// the integer domain — the only approximation is the Q8_1 activation quantiser,
// exactly as for the Q4/Q6/MXFP4 SoA kernels.
//   gemv_iq3_xxs_soa_f16 : fp16 activation, value-faithful (same db*grid*sign
//     factors as the dequant path, fp32 accumulation).  Correctness reference.
//   gemv_iq3_xxs_soa_q8  : W3A8 int-dot (dp4a_ss) over a block_q8_1x activation
//     stream.  The ALU-efficient decode lever, same shape as gemv_mxfp4_soa_q8.
// Both write fp16, matching every other GEMV in this engine.
//
// The 256-entry grid is staged into SLM once per work-group (1 KB) so the eight
// lookups per sub-block never touch global memory.  The 128-entry ksigns table
// is not staged at all — it is DERIVED (see iq3_sign_nibbles).  No ESIMD, no
// block2d, no lsc_load — plain SLM + scalar/vector loads.

#include "ie/deepseek4_experts.hpp"

#include "ie/dp4a.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>

#include <cstring>

namespace ie {
namespace {

// Same fp16→fp32 helper as gemv_q*k.cpp / moe_fused.cpp / moe_qwen3.cpp
// (copy-not-hoist, per the repo convention).
inline float dev_fp16_to_fp32(uint16_t h) {
    const uint32_t s = uint32_t(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu;
    uint32_t m =  h        & 0x3ffu;
    if (e == 0) {
        if (m == 0) return sycl::bit_cast<float>(s);
        while ((m & 0x400u) == 0) { m <<= 1; e -= 1; }
        e += 1;
        m &= ~0x400u;
    } else if (e == 31) {
        const uint32_t r = s | 0x7f800000u | (m << 13);
        return sycl::bit_cast<float>(r);
    }
    e += (127 - 15);
    const uint32_t r = s | (e << 23) | (m << 13);
    return sycl::bit_cast<float>(r);
}

// ksigns_iq2xs[i], derived instead of looked up:  the table is exactly
// `i | (popcount(i) & 1) << 7` for all 128 entries — bits 0..6 are the index
// itself and bit 7 is the parity that makes the 8-bit mask even-weight.  The
// identity is asserted over the whole table in deepseek4_experts_test.cpp, so
// this cannot silently drift from quant_blocks.hpp.  Returns the 4 sign bits
// for grid1 (elements 0..3) in `s1` and for grid2 (elements 4..7) in `s2`.
inline void iq3_sign_nibbles(uint32_t i7, uint32_t& s1, uint32_t& s2) {
    s1 = i7 & 15u;
    s2 = ((i7 >> 4) & 7u) | ((uint32_t(sycl::popcount(i7)) & 1u) << 3);
}

// One grid word (4 unsigned magnitudes, one per byte) + 4 sign bits (bit j
// negates byte j) → 4 packed int8 ready for dp4a_ss.
//
// Per-byte two's complement done on the whole word at once:
//   bits = one 0x01 in each byte to be negated
//   mask = 0xFF in each byte to be negated
//   (g ^ mask) + bits   ==   per byte, b -> (~b + 1) = -b
// The carry-out of byte j would corrupt byte j+1 iff b == 0, and the IQ3_XXS
// grid magnitudes are exactly {4,12,20,28,36,44,52,62} — ZERO IS NOT AMONG
// THEM (asserted over all 256 grid entries in the test).  Magnitudes are <= 62
// so the negation also stays inside int8.  ~5 ALU ops per 4 weights instead of
// the ~20 of a per-byte select loop; the result is bit-identical.
inline uint32_t iq3_signed_word(uint32_t g, uint32_t sgn4) {
    const uint32_t bits = (sgn4 * 0x00204081u) & 0x01010101u;   // bit j → byte j LSB
    return (g ^ (bits * 0xFFu)) + bits;
}

constexpr int kSG    = 16;
constexpr int kNPerWG = 32;
constexpr int kWGItems = kNPerWG * kSG;   // 512

}  // namespace

// ---------------------------------------------------------------------------
// Host-side SoA repack.  `src` is a native IQ3_XXS bank of E*N*(K/256) blocks
// laid out (expert, column, super-block) — the GGUF order for a
// [K, N, E] expert tensor.  Plane strides (in ELEMENTS of each plane's type):
//   gp: E stride N*(K/4)   ; ap: E stride N*(K/32) ; dp: E stride N*(K/256)
// ---------------------------------------------------------------------------
void iq3_xxs_repack_soa(const void* src, uint32_t K, uint32_t N, uint32_t E,
                        uint8_t* gp, uint32_t* ap, uint16_t* dp) {
    const uint32_t spc = K / kQK_K;        // super-blocks per column
    const uint32_t bpc = K / 32;           // 32-elem sub-blocks per column
    const uint64_t g_stride = uint64_t(N) * (uint64_t(K) / 4);
    const uint64_t a_stride = uint64_t(N) * bpc;
    const uint64_t d_stride = uint64_t(N) * spc;
    const auto* blocks = static_cast<const block_iq3_xxs*>(src);

    for (uint64_t e = 0; e < E; ++e) {
        for (uint64_t n = 0; n < N; ++n) {
            const block_iq3_xxs* col = blocks + (e * N + n) * spc;
            uint8_t*  gcol = gp + e * g_stride + n * (uint64_t(K) / 4);
            uint32_t* acol = ap + e * a_stride + n * bpc;
            uint16_t* dcol = dp + e * d_stride + n * spc;
            for (uint32_t s = 0; s < spc; ++s) {
                dcol[s] = col[s].d;
                // 64 grid-index bytes: 8 per sub-block, contiguous → one copy.
                std::memcpy(gcol + uint64_t(s) * 64, col[s].qs, 64);
                // 8 scale/sign words.
                std::memcpy(acol + uint64_t(s) * 8, col[s].qs + 64, 32);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// W3A8 int-dot decode GEMV.  Mirrors gemv_mxfp4_soa_q8's occupancy exactly
// (SG_SIZE=16, N_PER_WG=32, activation staged once in SLM, split-K by whole
// 32-elem sub-blocks).  Per sub-block: idot = Σ dp4a_ss(signed_grid, q8); the
// per-sub-block scale db and the q8 block scale fold once at the end.
// ---------------------------------------------------------------------------
sycl::event gemv_iq3_xxs_soa_q8(sycl::queue& q, const void* x_q8,
                                const uint8_t* gp, const uint32_t* ap, const uint16_t* dp,
                                sycl::half* y, uint32_t K, uint32_t N,
                                const std::vector<sycl::event>& deps) {
    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t bpc = K / 32;
    const uint32_t spc = K / kQK_K;
    const uint32_t n_wgs = (N + kNPerWG - 1) / kNPerWG;

    return ie::ps(q, "gemv_iq3_xxs", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> grid_slm(256, h);
        sycl::local_accessor<uint32_t, 1> q8s(bpc * 8, h);
        sycl::local_accessor<float, 1>    q8d(bpc, h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * kWGItems, kWGItems),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / kSG;
            const uint32_t lane  = lid % kSG;
            const uint32_t n     = wgid * kNPerWG + sg_id;

            for (uint32_t i = lid; i < 256; i += kWGItems) grid_slm[i] = kIQ3XXSGrid[i];
            for (uint32_t i = lid; i < bpc * 8; i += kWGItems)
                q8s[i] = reinterpret_cast<const uint32_t*>(X8[i / 8].qs)[i % 8];
            for (uint32_t i = lid; i < bpc; i += kWGItems) q8d[i] = X8[i].d;
            sycl::group_barrier(it.get_group());
            if (n >= N) return;

            const uint8_t*  g_col = gp + uint64_t(n) * (uint64_t(K) / 4);
            const uint32_t* a_col = ap + uint64_t(n) * bpc;
            const uint16_t* d_col = dp + uint64_t(n) * spc;

            float acc = 0.f;
            // Split-K: each lane strides whole 32-elem sub-blocks of this column.
            for (uint32_t sb = lane; sb < bpc; sb += kSG) {
                const uint32_t aux = a_col[sb];
                const float d  = dev_fp16_to_fp32(d_col[sb >> 3]);
                const float db = d * (0.5f + float(aux >> 28)) * 0.5f;
                const uint8_t* gi = g_col + uint64_t(sb) * 8;   // 8-byte aligned
                int32_t idot = 0;
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    uint32_t s1, s2;
                    iq3_sign_nibbles((aux >> (7 * l)) & 127u, s1, s2);
                    const uint32_t w1 = iq3_signed_word(grid_slm[gi[2 * l + 0]], s1);
                    const uint32_t w2 = iq3_signed_word(grid_slm[gi[2 * l + 1]], s2);
                    idot = ie::dp4a_ss(int32_t(w1), int32_t(q8s[sb * 8 + 2 * l + 0]), idot);
                    idot = ie::dp4a_ss(int32_t(w2), int32_t(q8s[sb * 8 + 2 * l + 1]), idot);
                }
                acc += db * q8d[sb] * float(idot);
            }

            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// ---------------------------------------------------------------------------
// Value-faithful fp16-activation decode GEMV (same db*grid*sign factors as
// ie::ref::dequant_iq3_xxs, fp32 accumulation).  The correctness reference for
// the int-dot variant; A staged in SLM.
// ---------------------------------------------------------------------------
sycl::event gemv_iq3_xxs_soa_f16(sycl::queue& q, const sycl::half* A,
                                 const uint8_t* gp, const uint32_t* ap, const uint16_t* dp,
                                 sycl::half* y, uint32_t K, uint32_t N,
                                 const std::vector<sycl::event>& deps) {
    const uint32_t bpc = K / 32;
    const uint32_t spc = K / kQK_K;
    const uint32_t n_wgs = (N + kNPerWG - 1) / kNPerWG;

    return ie::ps(q, "gemv_iq3_xxs", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1>   grid_slm(256, h);
        sycl::local_accessor<sycl::half, 1> A_slm(sycl::range<1>(K), h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * kWGItems, kWGItems),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / kSG;
            const uint32_t lane  = lid % kSG;
            const uint32_t n     = wgid * kNPerWG + sg_id;

            for (uint32_t i = lid; i < 256; i += kWGItems) grid_slm[i] = kIQ3XXSGrid[i];
            for (uint32_t i = lid; i < K;   i += kWGItems) A_slm[i] = A[i];
            sycl::group_barrier(it.get_group());
            if (n >= N) return;

            const uint8_t*  g_col = gp + uint64_t(n) * (uint64_t(K) / 4);
            const uint32_t* a_col = ap + uint64_t(n) * bpc;
            const uint16_t* d_col = dp + uint64_t(n) * spc;

            float acc = 0.f;
            for (uint32_t sb = lane; sb < bpc; sb += kSG) {
                const uint32_t aux = a_col[sb];
                const float d  = dev_fp16_to_fp32(d_col[sb >> 3]);
                const float db = d * (0.5f + float(aux >> 28)) * 0.5f;
                const uint8_t* gi = g_col + uint64_t(sb) * 8;
                const sycl::half* a = &A_slm[sb * 32];
                float sum = 0.f;
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    uint32_t s1, s2;
                    iq3_sign_nibbles((aux >> (7 * l)) & 127u, s1, s2);
                    const uint32_t grid1 = grid_slm[gi[2 * l + 0]];
                    const uint32_t grid2 = grid_slm[gi[2 * l + 1]];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float v1 = float((grid1 >> (8 * j)) & 0xFFu);
                        const float v2 = float((grid2 >> (8 * j)) & 0xFFu);
                        sum += float(a[8 * l + j])     * ((s1 & (1u << j)) ? -v1 : v1);
                        sum += float(a[8 * l + j + 4]) * ((s2 & (1u << j)) ? -v2 : v2);
                    }
                }
                acc += db * sum;
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

// ---------------------------------------------------------------------------
// SoA → fp32 dequant of ONE column (used by the test's device/host cross-check
// and by anything that needs the expanded weights).  out[k] = w(n, k).
// Values are IDENTICAL to ie::ref::dequant_iq3_xxs (the repack is a pure
// split), so this is the device-side echo of the bit-exact host reference.
// ---------------------------------------------------------------------------
sycl::event dequant_iq3_xxs_soa_col(sycl::queue& q,
                                    const uint8_t* gp, const uint32_t* ap, const uint16_t* dp,
                                    float* out, uint32_t K, uint32_t N, uint32_t n,
                                    const std::vector<sycl::event>& deps) {
    const uint32_t bpc = K / 32;
    const uint32_t spc = K / kQK_K;
    return ie::ps(q, "dequant_iq3_xxs_soa_col", [&](sycl::handler& h) {
        h.depends_on(deps);
        (void)N;
        h.parallel_for(sycl::range<1>(bpc), [=](sycl::id<1> idx) {
            const uint32_t sb = uint32_t(idx[0]);
            const uint32_t aux = ap[uint64_t(n) * bpc + sb];
            const float d  = dev_fp16_to_fp32(dp[uint64_t(n) * spc + (sb >> 3)]);
            const float db = d * (0.5f + float(aux >> 28)) * 0.5f;
            const uint8_t* gi = gp + uint64_t(n) * (uint64_t(K) / 4) + uint64_t(sb) * 8;
            float* o = out + uint64_t(sb) * 32;
            for (int l = 0; l < 4; ++l) {
                const uint32_t sgn   = kSignsIQ2XS[(aux >> (7 * l)) & 127u];
                const uint32_t grid1 = kIQ3XXSGrid[gi[2 * l + 0]];
                const uint32_t grid2 = kIQ3XXSGrid[gi[2 * l + 1]];
                for (int j = 0; j < 4; ++j) {
                    o[8 * l + j]     = db * float((grid1 >> (8 * j)) & 0xFFu) *
                                       ((sgn & (1u << j)) ? -1.f : 1.f);
                    o[8 * l + j + 4] = db * float((grid2 >> (8 * j)) & 0xFFu) *
                                       ((sgn & (1u << (j + 4))) ? -1.f : 1.f);
                }
            }
        });
    });
}

}  // namespace ie
