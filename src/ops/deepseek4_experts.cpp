// src/ops/deepseek4_experts.cpp — mixed IQ3_XXS / MXFP4 expert dispatch and the
// DeepSeek-V4 routed-expert block.  Phase 4.
//
// The dispatch is the point of this file.  DeepSeek-V4-Flash UD-Q3_K_XL is
// mixed precision INSIDE one layer and ACROSS layers: gate/up are IQ3_XXS on 42
// layers and MXFP4 on blk.26, down is MXFP4 everywhere.  `ds4_expert_bank_upload`
// therefore reads each GGUF tensor's own `dtype` and picks the repack, and
// `ds4_expert_gemv` dispatches on the bank's recorded dtype — the caller (the
// block below) issues the same three calls for every layer with no branch.
//
// The MXFP4 side reuses the already-proven gpt-oss kernels verbatim
// (gemv_mxfp4_soa_q8 over the same qs/e plane split), which is what makes
// blk.26 a free correctness harness for the new IQ3_XXS kernel: identical
// dispatch, identical activation stream, one branch trusted since gpt-oss.
//
// Everything here is decode-shaped (one GEMV per routed slot) and assumes the
// banks are already device-resident.  No streaming — that is Phase 5.

#include "ie/deepseek4_experts.hpp"

#include "ie/deepseek4_ops.hpp"
#include "ie/dp4a.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <limits>
#include <type_traits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace ie {
namespace {

const char* dtype_tag(DType t) {
    switch (t) {
        case DType::kIQ3_XXS: return "IQ3_XXS";
        case DType::kMXFP4:   return "MXFP4";
        default:              return "unsupported";
    }
}

// ---------------------------------------------------------------------------
// Decode helpers, COPIED VERBATIM from gemv_iq3_xxs.cpp and gemv_mxfp4.cpp.
//
// Copy-not-hoist is this repo's convention for these (see the comment above
// `dev_fp16_to_fp32` in gemv_iq3_xxs.cpp, which is itself the fifth copy).  The
// drift guard is not discipline, it is deepseek4_experts_test §7: it asserts the
// batched kernels below are BIT-IDENTICAL to the GEMVs in those two files on
// real IQ3_XXS and MXFP4 weights, so any divergence in these helpers fails the
// gate immediately.
// ---------------------------------------------------------------------------

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

inline void iq3_sign_nibbles(uint32_t i7, uint32_t& s1, uint32_t& s2) {
    s1 = i7 & 15u;
    s2 = ((i7 >> 4) & 7u) | ((uint32_t(sycl::popcount(i7)) & 1u) << 3);
}

inline uint32_t iq3_signed_word(uint32_t g, uint32_t sgn4) {
    const uint32_t bits = (sgn4 * 0x00204081u) & 0x01010101u;
    return (g ^ (bits * 0xFFu)) + bits;
}

inline float mxfp4_e8m0_half(uint8_t e) {
    const uint32_t bits = (e < 2u) ? (0x00200000u << e) : (uint32_t(e - 1u) << 23);
    return sycl::bit_cast<float>(bits);
}

// THE MXFP4 NIBBLE TABLE, FOLDED INTO ONE WORD.  The reference form is
//
//   exp = (nb>>1)&3;  mant = nb&1;  mag = exp ? ((2+mant) << (exp-1)) : mant
//
// which over t = nb&7 produces {0,1,2,3,4,6,8,12} — eight values, every one
// below 16, so the whole magnitude table fits in eight 4-bit fields of the
// constant 0xC8643210 and one variable shift replaces the exponent branch.
// The INTEGERS ARE IDENTICAL, not approximately so; deepseek4_expertgemm_gate_test
// §1 checks all 16 nibbles against the reference formula exhaustively, and §2
// checks the resulting kernel against the untouched gemv_mxfp4_soa_q8.
//
// It matters because the decode, not the dot, is what this kernel spends its
// instruction issue on: at NC=4 the ISA dump has 256 dp4a in 5138 instructions.
// Measured on one real layer-chunk (grouped, NC=4): 11.32 ms -> 9.55 ms.
inline int mxfp4_nibble_int(uint32_t nb) {
    const int mag = int((0xC8643210u >> ((nb & 7u) * 4u)) & 0xFu);
    return (nb & 8u) ? -mag : mag;
}

inline void mxfp4_decode_word(uint32_t w, uint32_t& lo, uint32_t& hi) {
    lo = 0; hi = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t b = (w >> (i * 8)) & 0xFFu;
        const int vlo = mxfp4_nibble_int(b & 0x0Fu);
        const int vhi = mxfp4_nibble_int(b >> 4);
        lo |= (uint32_t(uint8_t(int8_t(vlo))) << (i * 8));
        hi |= (uint32_t(uint8_t(int8_t(vhi))) << (i * 8));
    }
}

// THE SAME DECODE, ONE STEP FURTHER: a whole BYTE (two nibbles) in one lookup.
// `mxfp4_decode_word` needs the 4-bit magnitude table plus a sign select for
// each of the 8 nibbles in a word; at M=1 (decode) that ALU work is what the
// kernel spends its issue slots on, because there is no second activation row
// to amortise it over.  A 256-entry SLM table indexed by the byte collapses the
// two nibble decodes into one load, and the two int8 results ride in the two
// halves of a uint16.
//
// IT CANNOT DRIFT FROM THE TABLE ABOVE: the entry is BUILT by calling
// `mxfp4_nibble_int` on both nibbles, so there is one magnitude table in this
// file, not two.  deepseek4_expertgemm_gate_test §6a additionally checks all
// 256 entries against the independently transcribed exponent-branch reference.
inline uint16_t mxfp4_byte_lut_entry(uint32_t b) {
    const int vlo = mxfp4_nibble_int(b & 0x0Fu);
    const int vhi = mxfp4_nibble_int(b >> 4);
    return uint16_t((uint32_t(uint8_t(int8_t(vhi))) << 8) | uint8_t(int8_t(vlo)));
}

// Byte-for-byte the same `lo`/`hi` words `mxfp4_decode_word` produces — the two
// are cross-checked exhaustively over all 2^32 words by §6a's 256-byte identity
// (a word is four independent bytes, so 256 cases cover the whole domain).
inline void mxfp4_decode_word_lut(uint32_t w, const uint16_t* tbl,
                                  uint32_t& lo, uint32_t& hi) {
    const uint32_t t0 = tbl[ w        & 0xFFu];
    const uint32_t t1 = tbl[(w >>  8) & 0xFFu];
    const uint32_t t2 = tbl[(w >> 16) & 0xFFu];
    const uint32_t t3 = tbl[(w >> 24) & 0xFFu];
    lo = (t0 & 0xFFu) | ((t1 & 0xFFu) << 8) | ((t2 & 0xFFu) << 16) | ((t3 & 0xFFu) << 24);
    hi = (t0 >> 8)    | ((t1 >> 8)    << 8) | ((t2 >> 8)    << 16) | ((t3 >> 8)    << 24);
}

// Occupancy constants.  `kSG` and the lane -> sub-block split-K assignment are
// identical to both decode GEMVs, so the fp32 summation order is unchanged.
constexpr int kSG      = 16;
constexpr int kSGPerWG = 32;
constexpr int kWGItems = kSGPerWG * kSG;   // 512, unchanged
// Rows a work-item accumulates before spilling to the sub-group reduce.  A
// register-blocking factor, not a numerical one — each row's accumulator chain
// is independent and visits its sub-blocks in exactly the decode order
// regardless of the tile width.
constexpr int kMTile   = 8;
// WEIGHT COLUMNS PER SUB-GROUP — the second register-blocking factor, and the
// one that mattered most.  At the routed-expert shape M is ~6, so a sub-group
// that owns ONE column issues 8 packed activation loads per row per block
// against 1 weight load: the activation traffic, not the weight traffic, set the
// pace.  Owning NC columns loads the activation block ONCE and dp4a's it against
// all of them.  Measured (grouped, one layer-chunk, card 1):
//   MXFP4    NC=1 14.03 ms   NC=2 12.16   NC=4 11.32   NC=8 12.90 (spills: the
//                                          ISA dump shows 177 spill / 116 fill)
//   IQ3_XXS  NC=1 14.39 ms   NC=2  9.25   NC=4  8.16
// 4 is the measured optimum for both dtypes and the largest that does not spill
// at kMTile=8 on 128-GRF Xe2.
constexpr int kNCols = 4;

// BUT IT IS NOT FREE, AND IT IS NOT UNCONDITIONAL.  Tiling by NC divides the
// launch's work-group count by NC.  That is nothing when the grid is already
// thousands wide (the grouped case) and a disaster when it is not: ONE expert's
// gate at N=1024 goes from 32 work-groups to 8 on a 32-Xe-core B70, i.e. from
// one work-group per core to one per four.  MEASURED, the same layer-chunk
// issued as 768 single-expert launches: 19.9 ms at NC=1, 41.1 ms at NC=4.
//
// So the tiling is CHOSEN from the resulting grid width, never assumed, and the
// threshold is the MEASURED crossover rather than an occupancy guess.  Grouped
// MXFP4 gate shape (K=4096, N=1024), NC=4 time / NC=1 time against the NC=4
// work-group count:
//     8 WGs 2.83x worse | 32 WGs 1.41x worse | 64 WGs parity
//   128 WGs 0.94x       | 256 WGs 0.80x      | 512+ WGs 0.72x
// 128 is the first width at which tiling is a clear win, and it is one step
// above the parity point, so this dispatch never picks the slower kernel.
// Consequence that matters: `ds4_expert_gemm_q8` on one expert can never regress
// relative to the kernel it replaces, even if no caller ever groups.
constexpr uint32_t kMinTiledWgs = 128;

inline bool tile_columns(uint32_t n_jobs, uint32_t N) {
    const uint32_t cols = uint32_t(kSGPerWG) * uint32_t(kNCols);
    return uint64_t(n_jobs) * ((N / cols + (N % cols != 0))) >= kMinTiledWgs;
}

// ===========================================================================
// M == 1 — THE DECODE REGIME, WHICH IS A THIRD REGIME
// ===========================================================================
//
// The two constants above were both measured at the PREFILL shape (M tiled at
// kMTile=8).  Single-token decode is neither of the regimes they describe:
// 6 routed experts x 1 token row, weights read exactly once with no reuse.
// Two things about the M>1 body are actively wrong there.
//
//   (1) `mt = min(kMTile, M - m0)` is a RUNTIME bound, so the `#pragma unroll`
//       row loop emits eight predicated bodies and holds acc[NC][8] live across
//       the whole sub-block loop.  At M=1 seven of the eight never execute and
//       7/8 of the accumulator registers are dead weight the scheduler still
//       has to allocate around.
//   (2) The NC choice.  `kMinTiledWgs=128` picks between NC=1 and NC=4 only,
//       and at the real decode shapes it picks the wrong one twice: gate/up
//       (12 jobs, N=1024) lands on NC=1 and down (6 jobs, N=4096) on NC=4,
//       when the measured optima are NC=2 and NC=4 respectively.
//
// MEASURED (B70 card 0, real decode shapes, weights rotated over a 408 MB pool
// so nothing survives in the 24 MB L2, clock pinned by a burn loop, min of 5
// interleaved passes — the ordering matters because this card idles at 400 MHz
// and ramps to 2800):
//
//   MXFP4  gate+up  12 jobs K=4096 N=1024   0.0826 ms -> 0.0486  (1.70x)
//   MXFP4  down      6 jobs K=1024 N=4096   0.0374 ms -> 0.0278  (1.35x)
//   IQ3    gate+up  12 jobs K=4096 N=1024   0.0711 ms -> 0.0488  (1.46x)
//
// A pure streaming read of the same MXFP4 arena runs at 619 GB/s; the tuned
// gate+up kernel reaches 550 GB/s, so after this change the decode expert GEMM
// is within 13% of the memory system and is no longer issue-bound.
//
// THE NC RULE.  Sweeping NC over 10 (n_jobs, K, N) combinations, the optimum is
// the largest NC whose launch still fills kM1TargetWgs work-groups, in 9 of 10
// cases exactly.  That is one number, measured, and it replaces a boolean that
// could only ever express two of the four useful widths.  The tenth case
// (4 jobs, N=1024) prefers NC=2 where the rule picks NC=1, costing 8% — the
// rule errs toward the WIDER grid, which is the safe side.
constexpr uint32_t kM1TargetWgs = 192;

// The M=1 column width for a launch of `n_jobs` at output width `N`.  Always
// one of 1/2/4/8, always >= 1, so there is no "no kernel" case to handle.
inline uint32_t pick_nc_m1(uint32_t n_jobs, uint32_t N) {
    uint32_t best = 1;
    for (uint32_t nc : {2u, 4u, 8u}) {
        const uint32_t cols = uint32_t(kSGPerWG) * nc;
        if (uint64_t(n_jobs) * ((N / cols + (N % cols != 0))) >= kM1TargetWgs) best = nc;
    }
    return best;
}

// ---------------------------------------------------------------------------
// The M == 1 kernel BODIES.  Same lane -> sub-block assignment, same operand
// order, same `db * q8d * float(idot)` fold, same 16-lane sub-group reduce as
// the bodies above — the row loop is simply gone, because there is one row.
// That is why the output is bit-identical and not merely close:  an output
// element's accumulation chain is a function of (lane, sub-block order, fold
// order) only, and none of the three changed.
// ---------------------------------------------------------------------------
template <int NC>
inline void iq3_xxs_tile_m1(const sycl::sub_group& sg, uint32_t lane,
                            const uint8_t* gp, const uint32_t* ap, const uint16_t* dp,
                            const block_q8_1x* X8, sycl::half* y,
                            uint32_t K, uint32_t N, uint32_t n0,
                            const uint32_t* grid_slm) {
    const uint32_t bpc = K / 32;
    const uint32_t spc = K / kQK_K;
    const uint32_t nc  = sycl::min(uint32_t(NC), N - n0);

    float acc[NC];
    #pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.f;

    for (uint32_t sb = lane; sb < bpc; sb += kSG) {
        // ONE activation block load, re-used by all NC columns.
        const block_q8_1x& B = X8[sb];
        const auto* qw = reinterpret_cast<const uint32_t*>(B.qs);
        uint32_t a[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) a[j] = qw[j];
        const float bd = B.d;
        #pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (uint32_t(c) >= nc) continue;
            const uint32_t aux = ap[uint64_t(n0 + c) * bpc + sb];
            const float d  = dev_fp16_to_fp32(dp[uint64_t(n0 + c) * spc + (sb >> 3)]);
            const float db = d * (0.5f + float(aux >> 28)) * 0.5f;
            const uint8_t* gi = gp + uint64_t(n0 + c) * (uint64_t(K) / 4) + uint64_t(sb) * 8;
            uint32_t w[8];
            #pragma unroll
            for (int l = 0; l < 4; ++l) {
                uint32_t s1, s2;
                iq3_sign_nibbles((aux >> (7 * l)) & 127u, s1, s2);
                w[2 * l + 0] = iq3_signed_word(grid_slm[gi[2 * l + 0]], s1);
                w[2 * l + 1] = iq3_signed_word(grid_slm[gi[2 * l + 1]], s2);
            }
            int32_t idot = 0;
            // SAME operand order as the GEMV: (w1,q0) (w2,q1) ... per l.
            #pragma unroll
            for (int j = 0; j < 8; ++j) idot = ie::dp4a_ss(int32_t(w[j]), int32_t(a[j]), idot);
            acc[c] += db * bd * float(idot);
        }
    }
    #pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (uint32_t(c) >= nc) continue;
        const float r = sycl::reduce_over_group(sg, acc[c], sycl::plus<float>());
        if (lane == 0) y[uint64_t(n0) + c] = sycl::half(r);
    }
}

template <int NC>
inline void mxfp4_tile_m1(const sycl::sub_group& sg, uint32_t lane,
                          const uint8_t* qs_plane, const uint8_t* e_plane,
                          const block_q8_1x* X8, sycl::half* y,
                          uint32_t K, uint32_t N, uint32_t n0,
                          const uint16_t* lut_slm) {
    const uint32_t bpc = K / kQK_MXFP4;
    const uint32_t nc  = sycl::min(uint32_t(NC), N - n0);

    float acc[NC];
    #pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.f;

    for (uint32_t b = lane; b < bpc; b += kSG) {
        // ONE activation block load, re-used by all NC columns.
        const block_q8_1x& B = X8[b];
        const auto* q8 = reinterpret_cast<const uint32_t*>(B.qs);
        uint32_t a[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) a[j] = q8[j];
        const float bd = B.d;
        #pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (uint32_t(c) >= nc) continue;
            const float d = mxfp4_e8m0_half(e_plane[uint64_t(n0 + c) * bpc + b]);
            const auto* qw = reinterpret_cast<const uint32_t*>(
                qs_plane + uint64_t(n0 + c) * (uint64_t(K) / 2) + uint64_t(b) * 16);
            int32_t idot = 0;
            // SAME operand order as the GEMV: lo(wi) then hi(wi).
            #pragma unroll
            for (int wi = 0; wi < 4; ++wi) {
                uint32_t lo, hi;
                mxfp4_decode_word_lut(qw[wi], lut_slm, lo, hi);
                idot = ie::dp4a_ss(int32_t(lo), int32_t(a[wi]),     idot);
                idot = ie::dp4a_ss(int32_t(hi), int32_t(a[wi + 4]), idot);
            }
            acc[c] += d * bd * float(idot);
        }
    }
    #pragma unroll
    for (int c = 0; c < NC; ++c) {
        if (uint32_t(c) >= nc) continue;
        const float r = sycl::reduce_over_group(sg, acc[c], sycl::plus<float>());
        if (lane == 0) y[uint64_t(n0) + c] = sycl::half(r);
    }
}

// ---------------------------------------------------------------------------
// The two kernel BODIES.  Each computes the NC columns starting at `n0` for
// every row of `M`, and is called from two places: the single-expert launcher
// (grid = column tiles) and the grouped launcher (grid = job x column tile).
// Factoring them this way is what makes "the grouped path is the same
// arithmetic" a property of the code rather than a claim about two copies.
//
// Bit-identity to the decode GEMVs survives the column tiling for one reason:
// nothing about an output element's accumulation changed.  Lane `lane` still
// visits sub-blocks lane, lane+16, lane+32, ... in that order for EVERY column
// it owns, still folds `db * q8d * float(idot)` in that order, and the result
// is still produced by a 16-lane sub-group reduce.  Only which columns share a
// sub-group changed, and columns never interact.
// ---------------------------------------------------------------------------
template <int NC>
inline void iq3_xxs_tile(const sycl::sub_group& sg, uint32_t lane,
                         const uint8_t* gp, const uint32_t* ap, const uint16_t* dp,
                         const block_q8_1x* X8, sycl::half* y,
                         uint32_t K, uint32_t N, uint32_t M, uint32_t n0,
                         const uint32_t* grid_slm) {
    const uint32_t bpc = K / 32;
    const uint32_t spc = K / kQK_K;
    const uint32_t nc  = sycl::min(uint32_t(NC), N - n0);

    // Constant stride preserves grouped-kernel codegen. Widen the induction
    // variable so the final increment cannot wrap when M is UINT32_MAX.
    for (uint64_t row = 0; row < M; row += kMTile) {
        const uint32_t m0 = uint32_t(row);
        const uint32_t mt = sycl::min(uint32_t(kMTile), M - m0);
        float acc[NC][kMTile];
        #pragma unroll
        for (int c = 0; c < NC; ++c)
            #pragma unroll
            for (int mm = 0; mm < kMTile; ++mm) acc[c][mm] = 0.f;

        for (uint32_t sb = lane; sb < bpc; sb += kSG) {
            float    db[NC];
            uint32_t w[NC][8];
            #pragma unroll
            for (int c = 0; c < NC; ++c) {
                if (uint32_t(c) >= nc) {
                    db[c] = 0.f;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) w[c][j] = 0u;
                    continue;
                }
                const uint32_t aux = ap[uint64_t(n0 + c) * bpc + sb];
                const float d  = dev_fp16_to_fp32(dp[uint64_t(n0 + c) * spc + (sb >> 3)]);
                db[c] = d * (0.5f + float(aux >> 28)) * 0.5f;
                const uint8_t* gi = gp + uint64_t(n0 + c) * (uint64_t(K) / 4)
                                       + uint64_t(sb) * 8;
                #pragma unroll
                for (int l = 0; l < 4; ++l) {
                    uint32_t s1, s2;
                    iq3_sign_nibbles((aux >> (7 * l)) & 127u, s1, s2);
                    w[c][2 * l + 0] = iq3_signed_word(grid_slm[gi[2 * l + 0]], s1);
                    w[c][2 * l + 1] = iq3_signed_word(grid_slm[gi[2 * l + 1]], s2);
                }
            }
            #pragma unroll
            for (int mm = 0; mm < kMTile; ++mm) {
                if (uint32_t(mm) >= mt) continue;
                const block_q8_1x& B = X8[uint64_t(m0 + mm) * bpc + sb];
                const auto* qw = reinterpret_cast<const uint32_t*>(B.qs);
                // ONE activation block load, re-used by all NC columns.
                uint32_t a[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j) a[j] = qw[j];
                #pragma unroll
                for (int c = 0; c < NC; ++c) {
                    int32_t idot = 0;
                    // SAME operand order as the GEMV: (w1,q0) (w2,q1) ... per l.
                    #pragma unroll
                    for (int j = 0; j < 8; ++j)
                        idot = ie::dp4a_ss(int32_t(w[c][j]), int32_t(a[j]), idot);
                    acc[c][mm] += db[c] * B.d * float(idot);
                }
            }
        }
        #pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (uint32_t(c) >= nc) continue;
            #pragma unroll
            for (int mm = 0; mm < kMTile; ++mm) {
                if (uint32_t(mm) >= mt) continue;
                const float r = sycl::reduce_over_group(sg, acc[c][mm], sycl::plus<float>());
                if (lane == 0) y[uint64_t(m0 + mm) * N + n0 + c] = sycl::half(r);
            }
        }
    }
}

template <int NC, bool LUT = false>
inline void mxfp4_tile(const sycl::sub_group& sg, uint32_t lane,
                       const uint8_t* qs_plane, const uint8_t* e_plane,
                       const block_q8_1x* X8, sycl::half* y,
                       uint32_t K, uint32_t N, uint32_t M, uint32_t n0,
                       const uint16_t* lut = nullptr) {
    const uint32_t bpc = K / kQK_MXFP4;
    const uint32_t nc  = sycl::min(uint32_t(NC), N - n0);

    // Constant stride preserves grouped-kernel codegen. Widen the induction
    // variable so the final increment cannot wrap when M is UINT32_MAX.
    for (uint64_t row = 0; row < M; row += kMTile) {
        const uint32_t m0 = uint32_t(row);
        const uint32_t mt = sycl::min(uint32_t(kMTile), M - m0);
        float acc[NC][kMTile];
        #pragma unroll
        for (int c = 0; c < NC; ++c)
            #pragma unroll
            for (int mm = 0; mm < kMTile; ++mm) acc[c][mm] = 0.f;

        for (uint32_t b = lane; b < bpc; b += kSG) {
            float    d[NC];
            uint32_t wlo[NC][4], whi[NC][4];
            #pragma unroll
            for (int c = 0; c < NC; ++c) {
                if (uint32_t(c) >= nc) {
                    d[c] = 0.f;
                    #pragma unroll
                    for (int wi = 0; wi < 4; ++wi) { wlo[c][wi] = 0u; whi[c][wi] = 0u; }
                    continue;
                }
                d[c] = mxfp4_e8m0_half(e_plane[uint64_t(n0 + c) * bpc + b]);
                // MEASURED AND LEFT ALONE (2026-08-09): replacing these four
                // scalar loads with one 16 B sycl::uint4 (the Q2 #1c-vec
                // recipe) was checksum-bit-identical and FLAT at every M in
                // the isolated bench — unlike the Q2 GEMV this kernel is NOT
                // load-issue-bound.  With decode amortized across kMTile rows
                // the dp4a chain dominates, and ~3 TFLOP/s at M>=8 is its
                // int-dot issue ceiling.  The remaining 2-3x on this 66%%-of-
                // prefill surface is XMX/DPAS only, which is gated behind the
                // small-M corruption controls (gptoss_tp.cpp:703).
                const auto* qw = reinterpret_cast<const uint32_t*>(
                    qs_plane + uint64_t(n0 + c) * (uint64_t(K) / 2) + uint64_t(b) * 16);
                #pragma unroll
                for (int wi = 0; wi < 4; ++wi) {
                    if constexpr (LUT)
                        mxfp4_decode_word_lut(qw[wi], lut, wlo[c][wi], whi[c][wi]);
                    else
                        mxfp4_decode_word(qw[wi], wlo[c][wi], whi[c][wi]);
                }
            }
            #pragma unroll
            for (int mm = 0; mm < kMTile; ++mm) {
                if (uint32_t(mm) >= mt) continue;
                const block_q8_1x& B = X8[uint64_t(m0 + mm) * bpc + b];
                const auto* q8 = reinterpret_cast<const uint32_t*>(B.qs);
                // ONE activation block load, re-used by all NC columns.
                uint32_t a[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j) a[j] = q8[j];
                #pragma unroll
                for (int c = 0; c < NC; ++c) {
                    int32_t idot = 0;
                    // SAME operand order as the GEMV: lo(wi) then hi(wi).
                    #pragma unroll
                    for (int wi = 0; wi < 4; ++wi) {
                        idot = ie::dp4a_ss(int32_t(wlo[c][wi]), int32_t(a[wi]),     idot);
                        idot = ie::dp4a_ss(int32_t(whi[c][wi]), int32_t(a[wi + 4]), idot);
                    }
                    acc[c][mm] += d[c] * B.d * float(idot);
                }
            }
        }
        #pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (uint32_t(c) >= nc) continue;
            #pragma unroll
            for (int mm = 0; mm < kMTile; ++mm) {
                if (uint32_t(mm) >= mt) continue;
                const float r = sycl::reduce_over_group(sg, acc[c][mm], sycl::plus<float>());
                if (lane == 0) y[uint64_t(m0 + mm) * N + n0 + c] = sycl::half(r);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Single-expert launchers (grid = column tiles).  Kept because
// `ds4_expert_gemm_q8` is a public entry point with existing callers and
// because a 1-job group would pay a descriptor copy for nothing.
// ---------------------------------------------------------------------------
template <int NC>
sycl::event gemm_iq3_xxs_soa_q8_nc(sycl::queue& q, const void* x_q8,
                                   const uint8_t* gp, const uint32_t* ap, const uint16_t* dp,
                                   sycl::half* y, uint32_t K, uint32_t N, uint32_t M,
                                   const std::vector<sycl::event>& deps) {
    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    constexpr uint32_t CPW = uint32_t(kSGPerWG) * uint32_t(NC);
    const uint32_t n_wgs = (N / CPW + (N % CPW != 0));

    return ie::ps(q, "ds4_gemm_iq3_xxs", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> grid_slm(256, h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * kWGItems, kWGItems),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0));
            for (uint32_t i = lid; i < 256; i += kWGItems) grid_slm[i] = kIQ3XXSGrid[i];
            sycl::group_barrier(it.get_group());
            const uint32_t n0 = uint32_t(it.get_group(0)) * CPW + (lid / kSG) * NC;
            if (n0 >= N) return;
            iq3_xxs_tile<NC>(it.get_sub_group(), lid % kSG, gp, ap, dp, X8, y, K, N, M, n0,
                             grid_slm.get_multi_ptr<sycl::access::decorated::no>().get());
        });
    });
}

template <int NC>
sycl::event gemm_mxfp4_soa_q8_nc(sycl::queue& q, const void* x_q8,
                                 const uint8_t* qs_plane, const uint8_t* e_plane,
                                 sycl::half* y, uint32_t K, uint32_t N, uint32_t M,
                                 const std::vector<sycl::event>& deps) {
    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    constexpr uint32_t CPW = uint32_t(kSGPerWG) * uint32_t(NC);
    const uint32_t n_wgs = (N / CPW + (N % CPW != 0));

    return ie::ps(q, "ds4_gemm_mxfp4", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * kWGItems, kWGItems),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t lid = uint32_t(it.get_local_id(0));
            const uint32_t n0 = uint32_t(it.get_group(0)) * CPW + (lid / kSG) * NC;
            if (n0 >= N) return;
            mxfp4_tile<NC>(it.get_sub_group(), lid % kSG, qs_plane, e_plane, X8, y, K, N, M, n0);
        });
    });
}

// ---------------------------------------------------------------------------
// Grouped launchers (grid dim 0 = job, dim 1 = column tile).
// ---------------------------------------------------------------------------
// ONE descriptor record for either dtype.  A union rather than two arrays so a
// ring slot has a fixed stride whatever the chunk turned out to contain — and
// so the kernel's `jobs[group]` indexing cannot silently use the wrong stride,
// which is the failure mode two differently-sized structs would invite.
struct Iq3JobDev {
    const uint8_t*     gp; const uint32_t* ap; const uint16_t* dp;
    const block_q8_1x* x;  sycl::half*     y;  uint32_t        M;
};
struct MxJobDev {
    const uint8_t*     qs; const uint8_t* ep;
    const block_q8_1x* x;  sycl::half*    y;  uint32_t M;
};
union JobDevSlot { Iq3JobDev iq3; MxJobDev mx; };

template <int NC>
sycl::event gemm_iq3_xxs_grouped(sycl::queue& q, const JobDevSlot* jobs, uint32_t n_jobs,
                                 uint32_t K, uint32_t N) {
    constexpr uint32_t CPW = uint32_t(kSGPerWG) * uint32_t(NC);
    const uint32_t n_wgs = (N / CPW + (N % CPW != 0));
    return ie::ps(q, "ds4_gemm_iq3_xxs", [&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> grid_slm(256, h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(n_jobs, uint64_t(n_wgs) * kWGItems),
                                         sycl::range<2>(1, kWGItems)),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t lid = uint32_t(it.get_local_id(1));
            for (uint32_t i = lid; i < 256; i += kWGItems) grid_slm[i] = kIQ3XXSGrid[i];
            sycl::group_barrier(it.get_group());
            const uint32_t n0 = uint32_t(it.get_group(1)) * CPW + (lid / kSG) * NC;
            if (n0 >= N) return;
            const Iq3JobDev J = jobs[it.get_group(0)].iq3;
            iq3_xxs_tile<NC>(it.get_sub_group(), lid % kSG, J.gp, J.ap, J.dp, J.x, J.y,
                             K, N, J.M, n0,
                             grid_slm.get_multi_ptr<sycl::access::decorated::no>().get());
        });
    });
}

template <int NC>
sycl::event gemm_mxfp4_grouped(sycl::queue& q, const JobDevSlot* jobs, uint32_t n_jobs,
                               uint32_t K, uint32_t N, uint32_t max_m) {
    constexpr uint32_t WG = 256;
    constexpr uint32_t CPW = (WG / kSG) * uint32_t(NC);
    const uint32_t n_wgs = (N / CPW + (N % CPW != 0));
    const size_t row_tiles = (size_t(max_m) + kMTile - 1) / kMTile;
    return ie::ps(q, "ds4_gemm_mxfp4", [&](sycl::handler& h) {
        sycl::local_accessor<uint16_t, 1> lut(256, h);
        h.parallel_for(sycl::nd_range<3>(sycl::range<3>(n_jobs, row_tiles, uint64_t(n_wgs) * WG),
                                         sycl::range<3>(1, 1, WG)),
                       [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t lid = uint32_t(it.get_local_id(2));
            lut[lid] = mxfp4_byte_lut_entry(lid);
            // Every work-item reaches the table barrier before any tail exits.
            sycl::group_barrier(it.get_group());
            const uint32_t n0 = uint32_t(it.get_group(2)) * CPW + (lid / kSG) * NC;
            if (n0 >= N) return;
            const MxJobDev J = jobs[it.get_group(0)].mx;
            const uint32_t m0 = uint32_t(it.get_group(1)) * kMTile;
            if (m0 >= J.M) return;
            // Independent eight-token tiles remove the serial row-tile loop
            // and spread uneven expert buckets over the grid. Each output
            // keeps the original block order, dp4a chain and subgroup fold.
            const uint32_t rows = sycl::min(uint32_t(kMTile), J.M - m0);
            mxfp4_tile<NC, true>(it.get_sub_group(), lid % kSG, J.qs, J.ep,
                           J.x + uint64_t(m0) * (K / kQK_MXFP4),
                           J.y + uint64_t(m0) * N, K, N, rows, n0,
                           lut.get_multi_ptr<sycl::access::decorated::no>().get());
        });
    });
}

// ---------------------------------------------------------------------------
// Grouped M == 1 launchers.  Same 2-D grid (dim 0 = job, dim 1 = column tile)
// and the same descriptor array as the launchers above; only the tile body and
// the column width differ.  The job's own `M` is not read — the caller has
// already established that every job in THIS launch has M == 1, and the kernel
// would silently drop rows if that were untrue, so the check is a hard gate on
// the host side rather than a per-job branch here.
// ---------------------------------------------------------------------------
template <int NC>
sycl::event gemm_iq3_xxs_grouped_m1(sycl::queue& q, const JobDevSlot* jobs, uint32_t n_jobs,
                                    uint32_t K, uint32_t N) {
    constexpr uint32_t CPW = uint32_t(kSGPerWG) * uint32_t(NC);
    const uint32_t n_wgs = (N / CPW + (N % CPW != 0));
    return ie::ps(q, "ds4_gemm_iq3_xxs", [&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> grid_slm(256, h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(n_jobs, uint64_t(n_wgs) * kWGItems),
                                         sycl::range<2>(1, kWGItems)),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t lid = uint32_t(it.get_local_id(1));
            for (uint32_t i = lid; i < 256; i += kWGItems) grid_slm[i] = kIQ3XXSGrid[i];
            sycl::group_barrier(it.get_group());
            const uint32_t n0 = uint32_t(it.get_group(1)) * CPW + (lid / kSG) * NC;
            if (n0 >= N) return;
            const Iq3JobDev J = jobs[it.get_group(0)].iq3;
            iq3_xxs_tile_m1<NC>(it.get_sub_group(), lid % kSG, J.gp, J.ap, J.dp, J.x, J.y,
                                K, N, n0,
                                grid_slm.get_multi_ptr<sycl::access::decorated::no>().get());
        });
    });
}

template <int NC>
sycl::event gemm_mxfp4_grouped_m1(sycl::queue& q, const JobDevSlot* jobs, uint32_t n_jobs,
                                  uint32_t K, uint32_t N) {
    constexpr uint32_t CPW = uint32_t(kSGPerWG) * uint32_t(NC);
    const uint32_t n_wgs = (N / CPW + (N % CPW != 0));
    return ie::ps(q, "ds4_gemm_mxfp4", [&](sycl::handler& h) {
        // 512 bytes.  BUILT, not uploaded: the entries come from
        // `mxfp4_nibble_int`, the same magnitude table the M>1 body uses, so
        // there is no second copy of the decode to keep in sync and no host
        // buffer to plumb through the workspace.
        sycl::local_accessor<uint16_t, 1> lut_slm(256, h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(n_jobs, uint64_t(n_wgs) * kWGItems),
                                         sycl::range<2>(1, kWGItems)),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(kSG)]] {
            const uint32_t lid = uint32_t(it.get_local_id(1));
            for (uint32_t i = lid; i < 256; i += kWGItems)
                lut_slm[i] = mxfp4_byte_lut_entry(i);
            sycl::group_barrier(it.get_group());
            const uint32_t n0 = uint32_t(it.get_group(1)) * CPW + (lid / kSG) * NC;
            if (n0 >= N) return;
            const MxJobDev J = jobs[it.get_group(0)].mx;
            mxfp4_tile_m1<NC>(it.get_sub_group(), lid % kSG, J.qs, J.ep, J.x, J.y, K, N, n0,
                              lut_slm.get_multi_ptr<sycl::access::decorated::no>().get());
        });
    });
}

// NC is a runtime choice and a template parameter, so the four widths need one
// switch each.  Kept here rather than at the call site so the two dtypes cannot
// drift apart in which widths they support.
inline void dispatch_iq3_m1(sycl::queue& q, const JobDevSlot* ds, uint32_t n,
                            uint32_t K, uint32_t N, uint32_t nc) {
    switch (nc) {
        case 8:  gemm_iq3_xxs_grouped_m1<8>(q, ds, n, K, N); break;
        case 4:  gemm_iq3_xxs_grouped_m1<4>(q, ds, n, K, N); break;
        case 2:  gemm_iq3_xxs_grouped_m1<2>(q, ds, n, K, N); break;
        default: gemm_iq3_xxs_grouped_m1<1>(q, ds, n, K, N); break;
    }
}
inline void dispatch_mxfp4_m1(sycl::queue& q, const JobDevSlot* ds, uint32_t n,
                              uint32_t K, uint32_t N, uint32_t nc) {
    switch (nc) {
        case 8:  gemm_mxfp4_grouped_m1<8>(q, ds, n, K, N); break;
        case 4:  gemm_mxfp4_grouped_m1<4>(q, ds, n, K, N); break;
        case 2:  gemm_mxfp4_grouped_m1<2>(q, ds, n, K, N); break;
        default: gemm_mxfp4_grouped_m1<1>(q, ds, n, K, N); break;
    }
}

// y[i] += w * float(x[i]).  Folds the fp16→fp32 widening of the per-slot down
// output into the fp32 accumulation, so the block's running sum never rounds
// through fp16 (the reference accumulates in the model dtype via index_add_;
// fp32 here is strictly tighter).
sycl::event ds4_expert_accum(sycl::queue& q, const sycl::half* x, float w,
                             float* y, uint32_t n,
                             const std::vector<sycl::event>& deps = {}) {
    constexpr uint32_t WG = 256;
    const uint64_t global = ((uint64_t(n) + WG - 1) / WG) * WG;
    return ie::ps(q, "ds4_expert_accum", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(global, WG), [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= n) return;
            y[i] += w * float(x[i]);
        });
    });
}

}  // namespace

// ---------------------------------------------------------------------------
// bank upload
// ---------------------------------------------------------------------------
std::string ds4_expert_bank_upload(sycl::queue& q, const GgufTensorInfo& ti,
                                   uint32_t K, uint32_t N, uint32_t E_take,
                                   DS4ExpertBank& out) {
    const std::string nm(ti.name);
    if (ti.n_dims != 3)
        return nm + ": expected a 3-D expert tensor, got n_dims=" + std::to_string(ti.n_dims);
    if (ti.shape[0] != K || ti.shape[1] != N)
        return nm + ": shape [" + std::to_string(ti.shape[0]) + "," +
               std::to_string(ti.shape[1]) + "] != expected [" + std::to_string(K) + "," +
               std::to_string(N) + "]";
    if (E_take == 0 || E_take > ti.shape[2])
        return nm + ": E_take=" + std::to_string(E_take) + " out of range (file has " +
               std::to_string(ti.shape[2]) + ")";

    out = DS4ExpertBank{};
    out.dtype = ti.dtype;
    out.K = K; out.N = N; out.E = E_take;

    if (ti.dtype == DType::kIQ3_XXS) {
        if (K % kQK_K != 0) return nm + ": IQ3_XXS requires K % 256 == 0";
        const uint64_t bpe = uint64_t(N) * (uint64_t(K) / kQK_K);       // blocks per expert
        if (ti.nbytes != ti.shape[2] * bpe * sizeof(block_iq3_xxs))
            return nm + ": IQ3_XXS byte size mismatch";
        out.gp_stride = uint64_t(N) * (uint64_t(K) / 4);
        out.ap_stride = uint64_t(N) * (uint64_t(K) / 32);
        out.dp_stride = uint64_t(N) * (uint64_t(K) / kQK_K);

        std::vector<uint8_t>  gh(uint64_t(E_take) * out.gp_stride);
        std::vector<uint32_t> ah(uint64_t(E_take) * out.ap_stride);
        std::vector<uint16_t> dh(uint64_t(E_take) * out.dp_stride);
        iq3_xxs_repack_soa(ti.data, K, N, E_take, gh.data(), ah.data(), dh.data());

        out.gp = sycl::malloc_device<uint8_t>(gh.size(), q);
        out.ap = sycl::malloc_device<uint32_t>(ah.size(), q);
        out.dp = sycl::malloc_device<uint16_t>(dh.size(), q);
        if (!out.gp || !out.ap || !out.dp) { ds4_expert_bank_free(q, out); return nm + ": device alloc failed"; }
        q.memcpy(out.gp, gh.data(), gh.size());
        q.memcpy(out.ap, ah.data(), ah.size() * sizeof(uint32_t));
        q.memcpy(out.dp, dh.data(), dh.size() * sizeof(uint16_t)).wait();
        return {};
    }

    if (ti.dtype == DType::kMXFP4) {
        if (K % kQK_MXFP4 != 0) return nm + ": MXFP4 requires K % 32 == 0";
        const uint32_t bpc = K / kQK_MXFP4;
        const uint64_t bpe = uint64_t(N) * bpc;
        if (ti.nbytes != ti.shape[2] * bpe * sizeof(block_mxfp4))
            return nm + ": MXFP4 byte size mismatch";
        out.mx_qs_stride = uint64_t(N) * (uint64_t(K) / 2);
        out.mx_e_stride  = bpe;

        std::vector<uint8_t> qs(uint64_t(E_take) * out.mx_qs_stride);
        std::vector<uint8_t> ep(uint64_t(E_take) * out.mx_e_stride);
        const auto* src = reinterpret_cast<const block_mxfp4*>(ti.data);
        for (uint64_t i = 0; i < uint64_t(E_take) * bpe; ++i) {
            const uint64_t ex = i / bpe, within = i % bpe;
            std::memcpy(qs.data() + ex * out.mx_qs_stride + within * 16, src[i].qs, 16);
            ep[ex * out.mx_e_stride + within] = src[i].e;
        }
        out.mx_qs = sycl::malloc_device<uint8_t>(qs.size(), q);
        out.mx_e  = sycl::malloc_device<uint8_t>(ep.size(), q);
        if (!out.mx_qs || !out.mx_e) { ds4_expert_bank_free(q, out); return nm + ": device alloc failed"; }
        q.memcpy(out.mx_qs, qs.data(), qs.size());
        q.memcpy(out.mx_e,  ep.data(), ep.size()).wait();
        return {};
    }

    return nm + ": expert dtype " + std::to_string(int(ti.dtype)) +
           " has no device expert path (implemented: IQ3_XXS, MXFP4)";
}

void ds4_expert_bank_free(sycl::queue& q, DS4ExpertBank& b) {
    if (b.gp)    sycl::free(b.gp, q);
    if (b.ap)    sycl::free(b.ap, q);
    if (b.dp)    sycl::free(b.dp, q);
    if (b.mx_qs) sycl::free(b.mx_qs, q);
    if (b.mx_e)  sycl::free(b.mx_e, q);
    b = DS4ExpertBank{};
}

// ---------------------------------------------------------------------------
// dtype dispatch — the single place in the engine that knows an expert bank can
// be either format.  Callers pass a bank and an expert id, nothing else.
// ---------------------------------------------------------------------------
sycl::event ds4_expert_gemv(sycl::queue& q, const DS4ExpertBank& b, uint32_t e,
                            const void* x_q8, const sycl::half* x_f16, sycl::half* y,
                            const std::vector<sycl::event>& deps) {
    switch (b.dtype) {
        case DType::kIQ3_XXS: {
            const uint8_t*  gp = b.gp + uint64_t(e) * b.gp_stride;
            const uint32_t* ap = b.ap + uint64_t(e) * b.ap_stride;
            const uint16_t* dp = b.dp + uint64_t(e) * b.dp_stride;
            return x_f16 ? gemv_iq3_xxs_soa_f16(q, x_f16, gp, ap, dp, y, b.K, b.N, deps)
                         : gemv_iq3_xxs_soa_q8 (q, x_q8,  gp, ap, dp, y, b.K, b.N, deps);
        }
        case DType::kMXFP4: {
            const uint8_t* qs = b.mx_qs + uint64_t(e) * b.mx_qs_stride;
            const uint8_t* ep = b.mx_e  + uint64_t(e) * b.mx_e_stride;
            return x_f16 ? gemv_mxfp4_soa_f16(q, x_f16, qs, ep, y, b.K, b.N, deps)
                         : gemv_mxfp4_soa_q8 (q, x_q8,  qs, ep, y, b.K, b.N, deps);
        }
        default:
            // Unreachable: ds4_expert_bank_upload rejects every other dtype at
            // load time.  Throw rather than emit zeros — a silent wrong answer
            // in an expert GEMV is unfindable downstream.
            throw std::runtime_error(std::string("ds4_expert_gemv: bank dtype ") +
                                     dtype_tag(b.dtype) + " has no kernel");
    }
}

// ---------------------------------------------------------------------------
// batched dtype dispatch — the expert-major twin of ds4_expert_gemv.  Same
// contract: the caller passes a bank and an expert id and has NO dtype branch.
// ---------------------------------------------------------------------------
sycl::event ds4_expert_gemm_q8(sycl::queue& q, const DS4ExpertBank& b, uint32_t e,
                               uint32_t M, const void* x_q8, sycl::half* y,
                               const std::vector<sycl::event>& deps) {
    switch (b.dtype) {
        case DType::kIQ3_XXS: {
            const uint8_t*  gp = b.gp + uint64_t(e) * b.gp_stride;
            const uint32_t* ap = b.ap + uint64_t(e) * b.ap_stride;
            const uint16_t* dp = b.dp + uint64_t(e) * b.dp_stride;
            return tile_columns(1, b.N)
                ? gemm_iq3_xxs_soa_q8_nc<kNCols>(q, x_q8, gp, ap, dp, y, b.K, b.N, M, deps)
                : gemm_iq3_xxs_soa_q8_nc<1>     (q, x_q8, gp, ap, dp, y, b.K, b.N, M, deps);
        }
        case DType::kMXFP4: {
            const uint8_t* qs = b.mx_qs + uint64_t(e) * b.mx_qs_stride;
            const uint8_t* ep = b.mx_e  + uint64_t(e) * b.mx_e_stride;
            return tile_columns(1, b.N)
                ? gemm_mxfp4_soa_q8_nc<kNCols>(q, x_q8, qs, ep, y, b.K, b.N, M, deps)
                : gemm_mxfp4_soa_q8_nc<1>     (q, x_q8, qs, ep, y, b.K, b.N, M, deps);
        }
        default:
            // Same reasoning as ds4_expert_gemv: a silent wrong answer here is
            // unfindable downstream, so refuse loudly.
            throw std::runtime_error(std::string("ds4_expert_gemm_q8: bank dtype ") +
                                     dtype_tag(b.dtype) + " has no kernel");
    }
}

// ---------------------------------------------------------------------------
// grouped dtype dispatch — every occupied expert of a chunk in ONE launch
// ---------------------------------------------------------------------------
bool ds4_expert_gemm_column_tiled(uint32_t n_jobs, uint32_t N) {
    return tile_columns(n_jobs, N);
}

uint32_t ds4_expert_gemm_m1_ncols(uint32_t n_jobs, uint32_t N) {
    return pick_nc_m1(n_jobs, N);
}

std::string ds4_gemm_group_ws_alloc(sycl::queue& q, uint32_t max_jobs,
                                    DS4GemmGroupWs& gws) {
    if (max_jobs == 0) return "ds4_gemm_group_ws_alloc: max_jobs must be non-zero";
    gws = DS4GemmGroupWs{};
    gws.cap = max_jobs;
    const size_t n = size_t(kDs4GemmRing) * max_jobs;
    gws.host = sycl::malloc_host<JobDevSlot>(n, q);
    gws.dev  = sycl::malloc_device<JobDevSlot>(n, q);
    if (!gws.host || !gws.dev) {
        ds4_gemm_group_ws_free(q, gws);
        return "ds4_gemm_group_ws_alloc: alloc failed";
    }
    gws.slot_ev.assign(kDs4GemmRing, sycl::event{});
    return {};
}

void ds4_gemm_group_ws_free(sycl::queue& q, DS4GemmGroupWs& gws) {
    if (gws.host) sycl::free(gws.host, q);
    if (gws.dev)  sycl::free(gws.dev, q);
    gws = DS4GemmGroupWs{};
}

std::string ds4_expert_gemm_q8_grouped(sycl::queue& q, const DS4GemmJob* jobs,
                                       uint32_t n_jobs, DS4GemmGroupWs& gws) {
    if (n_jobs == 0) return {};
    if (!q.is_in_order()) return "ds4_expert_gemm_q8_grouped: requires an in-order queue";
    if (!jobs) return "ds4_expert_gemm_q8_grouped: null job array";
    if (!gws.host || !gws.dev || !gws.cap || gws.slot_ev.size() < kDs4GemmRing || gws.cursor >= kDs4GemmRing)
        return "ds4_expert_gemm_q8_grouped: workspace not allocated or invalid";

    // Bucket by (dtype, K, N).  Nothing here reads a per-model constant: every
    // job's dtype comes from its own bank, so a layer whose gate/up are IQ3_XXS
    // and whose down is MXFP4 simply produces two buckets.
    struct Key { DType t; uint32_t K, N; };
    std::vector<Key>                  keys;
    std::vector<std::vector<uint32_t>> bucket;
    for (uint32_t i = 0; i < n_jobs; ++i) {
        const DS4ExpertBank& b = jobs[i].bank;
        if (b.dtype != DType::kIQ3_XXS && b.dtype != DType::kMXFP4)
            return std::string("ds4_expert_gemm_q8_grouped: bank dtype ") +
                   dtype_tag(b.dtype) + " has no kernel";
        if (jobs[i].M == 0) continue;
        const auto& J = jobs[i];
        constexpr uint64_t limit = std::numeric_limits<ptrdiff_t>::max();
        if (!b.K || !b.N || b.K % (b.dtype == DType::kIQ3_XXS ? 256 : 32) ||
            J.e >= b.E || !J.x_q8 || !J.y)
            return "ds4_expert_gemm_q8_grouped: invalid shape, expert index, or operand";
        if (uint64_t(b.K / 32) > limit / sizeof(block_q8_1x) / J.M ||
            uint64_t(b.N) > limit / sizeof(sycl::half) / J.M)
            return "ds4_expert_gemm_q8_grouped: tensor byte span overflow";
        // Worst-case NC=1 grid, bounded before descriptor submission. Each
        // bucket/chunk contains at most this many jobs and at most these rows.
        const uint64_t row_tiles = (uint64_t(J.M) + kMTile - 1) / kMTile;
        const uint64_t col_items = (uint64_t(b.N) / 16 + (b.N % 16 != 0)) * kWGItems;
        if (row_tiles > std::numeric_limits<size_t>::max() / col_items / std::min(n_jobs, gws.cap))
            return "ds4_expert_gemm_q8_grouped: grouped grid size overflow";
        size_t k = 0;
        for (; k < keys.size(); ++k)
            if (keys[k].t == b.dtype && keys[k].K == b.K && keys[k].N == b.N) break;
        if (k == keys.size()) { keys.push_back({b.dtype, b.K, b.N}); bucket.emplace_back(); }
        bucket[k].push_back(i);
    }

    auto* host_all = static_cast<JobDevSlot*>(gws.host);
    auto* dev_all  = static_cast<JobDevSlot*>(gws.dev);

    for (size_t k = 0; k < keys.size(); ++k) {
        const std::vector<uint32_t>& ids = bucket[k];
        for (size_t base = 0; base < ids.size(); base += gws.cap) {
            const uint32_t n = uint32_t(std::min<size_t>(gws.cap, ids.size() - base));
            const uint32_t slot = gws.cursor;
            gws.cursor = (gws.cursor + 1) % kDs4GemmRing;
            // The host is about to overwrite this slot's staging.  Its previous
            // H2D may still be queued behind kDs4GemmRing-1 other launches; wait
            // for that ONE copy, not for the queue.
            gws.slot_ev[slot].wait();

            JobDevSlot* hs = host_all + size_t(slot) * gws.cap;
            JobDevSlot* ds = dev_all  + size_t(slot) * gws.cap;
            uint32_t max_m = 0;
            for (uint32_t j = 0; j < n; ++j) {
                const DS4GemmJob&    J = jobs[ids[base + j]];
                max_m = std::max(max_m, J.M);
                const DS4ExpertBank& b = J.bank;
                const auto* X8 = static_cast<const block_q8_1x*>(J.x_q8);
                if (keys[k].t == DType::kIQ3_XXS)
                    hs[j].iq3 = {b.gp + uint64_t(J.e) * b.gp_stride,
                                 b.ap + uint64_t(J.e) * b.ap_stride,
                                 b.dp + uint64_t(J.e) * b.dp_stride, X8, J.y, J.M};
                else
                    hs[j].mx  = {b.mx_qs + uint64_t(J.e) * b.mx_qs_stride,
                                 b.mx_e  + uint64_t(J.e) * b.mx_e_stride, X8, J.y, J.M};
            }
            // A kernel, not a memcpy, for the reason stated at ds4_gemm_mxfp4_xmx:
            // the memcpy's SUBMISSION blocked the host until the queue's prior
            // work retired (measured 2026-09-02, docs/deepseek4/72 Phase F).
            {
                static_assert(sizeof(JobDevSlot) % sizeof(uint32_t) == 0, "slot must be word-copyable");
                const size_t nwords = (sizeof(JobDevSlot) / sizeof(uint32_t)) * size_t(n);
                const uint32_t* hsrc = reinterpret_cast<const uint32_t*>(hs);
                uint32_t*       ddst = reinterpret_cast<uint32_t*>(ds);
                gws.slot_ev[slot] = q.parallel_for(sycl::range<1>(nwords),
                                                   [=](sycl::id<1> i) { ddst[i] = hsrc[i]; });
            }
            // THE M=1 KERNEL IS USED ONLY WHEN EVERY JOB IN THIS LAUNCH HAS
            // M == 1.  It does not read the per-job M, so one M>1 job in the
            // bucket would have its extra rows silently dropped; that makes
            // this an all-or-nothing property of the launch, checked here.
            // Single-token decode satisfies it for every bucket; prefill never
            // does, and takes the untouched M>1 kernels below.
            bool all_m1 = true;
            for (uint32_t j = 0; j < n && all_m1; ++j)
                all_m1 = (jobs[ids[base + j]].M == 1);
            if (all_m1) {
                const uint32_t nc = pick_nc_m1(n, keys[k].N);
                if (keys[k].t == DType::kIQ3_XXS)
                    dispatch_iq3_m1(q, ds, n, keys[k].K, keys[k].N, nc);
                else
                    dispatch_mxfp4_m1(q, ds, n, keys[k].K, keys[k].N, nc);
                continue;
            }
            const bool tile = tile_columns(n, keys[k].N);
            if (keys[k].t == DType::kIQ3_XXS) {
                if (tile) gemm_iq3_xxs_grouped<kNCols>(q, ds, n, keys[k].K, keys[k].N);
                else      gemm_iq3_xxs_grouped<1>     (q, ds, n, keys[k].K, keys[k].N);
            } else {
                if (tile) gemm_mxfp4_grouped<kNCols>(q, ds, n, keys[k].K, keys[k].N, max_m);
                else      gemm_mxfp4_grouped<1>     (q, ds, n, keys[k].K, keys[k].N, max_m);
            }
        }
    }
    return {};
}

// ---------------------------------------------------------------------------
// pack / unpack
// ---------------------------------------------------------------------------
sycl::event ds4_expert_gather_cast(sycl::queue& q, const float* x, const int32_t* rows,
                                   sycl::half* xp, uint32_t M, uint32_t H,
                                   const std::vector<sycl::event>& deps) {
    // Two columns reuse the row index while preserving each conversion.
    // Limit this path to the measured cache-resident range.
    if (H >= 1024 && uint64_t(M) * H <= 2097152) {
        constexpr uint32_t WG = 256;
        const uint64_t global = (((uint64_t(H) + 1) / 2 + WG - 1) / WG) * WG;
        return ie::ps(q, "ds4_moe_gather", [&](sycl::handler& h) {
            h.depends_on(deps);
            h.parallel_for(sycl::nd_range<2>({M, global}, {1, WG}), [=](sycl::nd_item<2> it) {
                const uint32_t r = uint32_t(it.get_global_id(0));
                const uint64_t c = it.get_global_id(1) * 2;
                if (c >= H) return;
                const uint64_t i = uint64_t(r) * H + c;
                const uint64_t base = uint64_t(rows[r]) * H + c;
                xp[i] = sycl::half(x[base]);
                if (c + 1 < H) xp[i + 1] = sycl::half(x[base + 1]);
            });
        });
    }
    constexpr uint32_t WG = 256;
    const uint64_t total  = uint64_t(M) * H;
    const uint64_t global = ((total + WG - 1) / WG) * WG;
    return ie::ps(q, "ds4_moe_gather", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(global, WG), [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= total) return;
            const uint32_t r = uint32_t(i / H);
            const uint32_t c = uint32_t(i % H);
            xp[i] = sycl::half(x[uint64_t(rows[r]) * H + c]);
        });
    });
}

// f32-source scatter — identical arithmetic and identical ASCENDING-kslot
// accumulation order to the fp16 one below, reading an f32 packed buffer.
// Exists so the XMX `down` route can hand its oneDNN f32 result straight to the
// scatter: without it that result had to be cast BACK to fp16 (a whole R*H
// pass) AND the int-dot route's quantize_q8_1 stayed in the chain, which is
// exactly what made the first XMX-down attempt a net LOSS (pp2048 184.7 ->
// 170.8, measured 2026-08-09).  Bit-comparable to the fp16 path only up to the
// fp16 rounding the fp16 path applies to `yp`; this one skips that rounding, so
// it is strictly the MORE accurate of the two.
sycl::event ds4_expert_scatter_accum_f32(sycl::queue& q, const float* yp,
                                         const int32_t* tk2p, const float* w_packed,
                                         float* y, uint32_t T, uint32_t top_k, uint32_t H,
                                         const std::vector<sycl::event>& deps) {
    // Two columns share each packed index and weight. Each column retains
    // the ascending kslot FP32 accumulation order. Tiny launches and large
    // top-1 copies retain the original geometry.
    if (H >= 1024 && uint64_t(T) * H >= 131072 && (top_k > 1 || uint64_t(T) * H <= 2097152)) {
        constexpr uint32_t WG = 256;
        const uint64_t global = (((uint64_t(H) + 1) / 2 + WG - 1) / WG) * WG;
        return ie::ps(q, "ds4_moe_scatter", [&](sycl::handler& h) {
            h.depends_on(deps);
            h.parallel_for(sycl::nd_range<2>({T, global}, {1, WG}), [=](sycl::nd_item<2> it) {
                const uint32_t t = uint32_t(it.get_global_id(0));
                const uint64_t c = it.get_global_id(1) * 2;
                if (c >= H) return;
                const uint64_t i = uint64_t(t) * H + c;
                float v = y[i];
                float v1 = c + 1 < H ? y[i + 1] : 0.f;
                for (uint32_t k = 0; k < top_k; ++k) {
                    const uint32_t p = uint32_t(tk2p[uint64_t(t) * top_k + k]);
                    const float w = w_packed[p];
                    v += w * yp[uint64_t(p) * H + c];
                    if (c + 1 < H) v1 += w * yp[uint64_t(p) * H + c + 1];
                }
                y[i] = v;
                if (c + 1 < H) y[i + 1] = v1;
            });
        });
    }
    constexpr uint32_t WG = 256;
    const uint64_t total  = uint64_t(T) * H;
    const uint64_t global = ((total + WG - 1) / WG) * WG;
    return ie::ps(q, "ds4_moe_scatter", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(global, WG), [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= total) return;
            const uint32_t t = uint32_t(i / H);
            const uint32_t c = uint32_t(i % H);
            float v = y[i];
            for (uint32_t k = 0; k < top_k; ++k) {
                const uint32_t p = uint32_t(tk2p[uint64_t(t) * top_k + k]);
                v += w_packed[p] * yp[uint64_t(p) * H + c];
            }
            y[i] = v;
        });
    });
}

sycl::event ds4_expert_scatter_accum(sycl::queue& q, const sycl::half* yp,
                                     const int32_t* tk2p, const float* w_packed,
                                     float* y, uint32_t T, uint32_t top_k, uint32_t H,
                                     const std::vector<sycl::event>& deps) {
    // Two columns share each packed index and weight. Each column retains
    // the ascending kslot FP32 accumulation order. Tiny launches and large
    // top-1 copies retain the original geometry.
    if (H >= 1024 && uint64_t(T) * H >= 131072 && (top_k > 1 || uint64_t(T) * H <= 2097152)) {
        constexpr uint32_t WG = 256;
        const uint64_t global = (((uint64_t(H) + 1) / 2 + WG - 1) / WG) * WG;
        return ie::ps(q, "ds4_moe_scatter", [&](sycl::handler& h) {
            h.depends_on(deps);
            h.parallel_for(sycl::nd_range<2>({T, global}, {1, WG}), [=](sycl::nd_item<2> it) {
                const uint32_t t = uint32_t(it.get_global_id(0));
                const uint64_t c = it.get_global_id(1) * 2;
                if (c >= H) return;
                const uint64_t i = uint64_t(t) * H + c;
                // ASCENDING kslot: the same fp32 accumulation order the token-major
                // path's chain of `accum_f16` calls produced.  Read-modify-write of
                // one running float, not a tree reduction, for the same reason.
                float v = y[i];
                float v1 = c + 1 < H ? y[i + 1] : 0.f;
                for (uint32_t k = 0; k < top_k; ++k) {
                    const uint32_t p = uint32_t(tk2p[uint64_t(t) * top_k + k]);
                    const float w = w_packed[p];
                    v += w * float(yp[uint64_t(p) * H + c]);
                    if (c + 1 < H) v1 += w * float(yp[uint64_t(p) * H + c + 1]);
                }
                y[i] = v;
                if (c + 1 < H) y[i + 1] = v1;
            });
        });
    }
    constexpr uint32_t WG = 256;
    const uint64_t total  = uint64_t(T) * H;
    const uint64_t global = ((total + WG - 1) / WG) * WG;
    return ie::ps(q, "ds4_moe_scatter", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(global, WG), [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= total) return;
            const uint32_t t = uint32_t(i / H);
            const uint32_t c = uint32_t(i % H);
            // ASCENDING kslot: the same fp32 accumulation order the token-major
            // path's chain of `accum_f16` calls produced.  Read-modify-write of
            // one running float, not a tree reduction, for the same reason.
            float v = y[i];
            for (uint32_t k = 0; k < top_k; ++k) {
                const uint32_t p = uint32_t(tk2p[uint64_t(t) * top_k + k]);
                v += w_packed[p] * float(yp[uint64_t(p) * H + c]);
            }
            y[i] = v;
        });
    });
}

// ---------------------------------------------------------------------------
// batch workspace
// ---------------------------------------------------------------------------
// MXFP4 slot planes -> f16 W[N][K] (each output column's K values contiguous
// = the exact buffer gemm_nt_f16_onednn describes as {K,N}/{1,K}).  Bit-exact
// decode: value = e8m0_half(e) * mag[nib] with the SAME half-scaled exponent
// and +/-{0,1,2,3,4,6,8,12} magnitude table the int-dot kernels fold — every
// MXFP4 value is exactly representable in f16, so this materialization loses
// nothing.  Built 2026-08-09 for the XMX expert route (IE_DS4_EXPERT_XMX):
// the small-M corruption that banned XMX here (gptoss_tp.cpp:703) no longer
// reproduces on the current stack — probed 160 trials clean, M in [1,24].
sycl::event ds4_dequant_mxfp4_w16(sycl::queue& q,
                                  const uint8_t* qs_plane, const uint8_t* e_plane,
                                  sycl::half* w16, uint32_t K, uint32_t N,
                                  const std::vector<sycl::event>& deps) {
    const uint32_t bpc = K / 32u;
    return ie::ps(q, "ds4_dequant_mxfp4_w16", [&](sycl::handler& h) {
        h.depends_on(deps);
        const uint64_t total = uint64_t(N) * bpc;
        const uint64_t global = (total / 256 + (total % 256 != 0)) * 256;
        h.parallel_for(sycl::nd_range<1>(global, 256), [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= total) return;
            const uint32_t n = uint32_t(i / bpc), b = uint32_t(i % bpc);
            const float d = mxfp4_e8m0_half(e_plane[uint64_t(n) * bpc + b]);
            // ONE 16 B load of the block's nibble bytes.
            sycl::vec<uint8_t, 16> bytes;
            bytes.load(0, qs_plane + uint64_t(n) * (uint64_t(K) / 2) + uint64_t(b) * 16);
            const sycl::uint4 w4 = bytes.as<sycl::uint4>();
            const uint32_t wd[4] = {w4.x(), w4.y(), w4.z(), w4.w()};
            // BRANCHLESS decode — NO indexed magnitude table.  A `float mag[16]`
            // indexed by a runtime nibble is placed in PRIVATE memory on Xe and
            // it dominated: this kernel measured 3.66 s of a 7.85 s profiled
            // prefill chunk (47%) in that form.  `mxfp4_nibble_int` is the same
            // register-only ALU decode gemv_mxfp4.cpp uses for the same reason,
            // and it is THE table this file already owns, so the values cannot
            // drift.  `mxfp4_nibble_int` yields exactly {0,1,2,3,4,6,8,12} with
            // sign — the SAME magnitudes the indexed table held, so `d` is the
            // whole scale.  (A spurious 0.5 here was caught by the route gate:
            // max abs err 5.8e2 against int-dot's 9.8.  The gate earns its keep.)
            const float ds = d;
            sycl::half* out = w16 + uint64_t(n) * K + uint64_t(b) * 32;
            sycl::vec<sycl::half, 8> vlo[2], vhi[2];
            #pragma unroll
            for (int wi = 0; wi < 4; ++wi) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const uint32_t byte = (wd[wi] >> (j * 8)) & 0xFFu;
                    const int idx = wi * 4 + j;               // 0..15
                    vlo[idx / 8][idx % 8] =
                        sycl::half(ds * float(mxfp4_nibble_int(byte & 0x0Fu)));
                    vhi[idx / 8][idx % 8] =
                        sycl::half(ds * float(mxfp4_nibble_int(byte >> 4)));
                }
            }
            // Four 16 B vector stores instead of 32 scalar ones.
            vlo[0].store(0, out); vlo[1].store(1, out);
            vhi[0].store(2, out); vhi[1].store(3, out);
        });
    });
}

// DEFAULT ON as of 2026-08-09, on a real-model PPL gate that measures it
// BETTER than the route it replaces (int-dot quantises the ACTIVATION to int8;
// this does not).  wikitext batch-mode PPL 2.4159-2.4210 vs 2.4284; pp2048
// 162.6 -> 186.7.  Set IE_DS4_EXPERT_XMX=0 to revert.
bool ds4_expert_xmx_on() {
    static const bool on = [] {
        const char* e = std::getenv("IE_DS4_EXPERT_XMX");
        return !(e && *e && std::string(e) == "0") && ie::onednn_available();
    }();
    return on;
}


// ---------------------------------------------------------------------------
// Fused MXFP4 -> XMX GEMM (docs/deepseek4/72 Phase E)
//
// WG = 8 sub-groups (128 lanes) on a BM 32 x BN 128 tile of one job; the K
// step is 32 = one MXFP4 block, so lane n decodes column (n0 + n)'s block —
// one 16 B nibble load + one e8m0 byte — straight into B_smem[k][n] (the
// row-major [K][N] tile joint_matrix wants), with the SAME e8m0_half scale and
// {0,1,2,3,4,6,8,12} magnitudes ds4_dequant_mxfp4_w16 writes, so every B value
// is bit-for-bit the one the materialised route multiplies.  The A tile
// (32 rows x 32 k) comes from the packed fp16 activations; rows >= M are
// zero.  Each sub-group owns 16 columns and 4 accumulators down M.  The
// store goes through joint_matrix_apply with a per-row guard: the batch
// workspace rows are contiguous across experts, so a spilled 8-row tile would
// land in the NEXT job's rows.
//
// Measured 2026-09-02 (docs/deepseek4/72 Phase E): 544 ms per 512-token chunk
// (2-card sum) against the dequant pass's 528 ms + oneDNN — a small net win
// (+4% prefill).  A transposed variant with vector SLM stores and a 128-wide K
// step measured 4.6x SLOWER (SLM bank conflicts on the 256 B row stride);
// the fp16-dequant ALU alone bounds any variant of this route at roughly half
// of this kernel's time, so the remaining prefill lever is elsewhere (host
// gaps and the TP reductions, see the doc).
// ---------------------------------------------------------------------------
namespace xm {

namespace mat  = sycl::ext::oneapi::experimental::matrix;
namespace imat = sycl::ext::intel::experimental::matrix;

static_assert(std::is_trivially_copyable_v<sycl::vec<sycl::half, 8>>);
static_assert(std::is_trivially_copyable_v<sycl::vec<uint8_t, 16>>);

constexpr int SG  = 16;
constexpr int BM  = 32;
constexpr int BN  = 128;
constexpr int BK  = 32;
constexpr int TM  = 8;
constexpr int TN  = 16;
constexpr int TK  = 16;
constexpr int MR  = BM / TM;      // 4 accumulators down M per sub-group
constexpr int SGS = BN / TN;      // 8 sub-groups across N
constexpr int WG  = SGS * SG;     // 128 lanes
constexpr uint32_t kMaxJobs = 1024;
constexpr int      kRing    = 8;

struct Job {
    const uint8_t*    qs;
    const uint8_t*    e;
    const sycl::half* x;
    float*            y;
    uint32_t M, K, N;
    uint32_t wg0;     // first work-group of this job in the flattened grid
};

// Host staging remains live until its copy completes. Device staging remains
// live until its GEMM completes. Track both independently so correctness does
// not depend on caller drains or in-order queues. Cached for the queue lifetime
// as before; the per-queue mutex also serializes concurrent host submissions.
struct Ring {
    Job* host = nullptr;
    Job* dev = nullptr;
    int next = 0;
    std::array<sycl::event, kRing> copied, consumed;
    std::array<bool, kRing> used{};
    std::mutex submit_mutex;
};

// Heap ownership keeps references stable while other queues enter the cache.
Ring& ring_for(sycl::queue& q) {
    static std::mutex mu;
    static std::deque<std::pair<sycl::queue, std::unique_ptr<Ring>>> cache;
    std::lock_guard<std::mutex> lk(mu);
    for (auto& kv : cache)
        if (kv.first == q) return *kv.second;
    auto r = std::make_unique<Ring>();
    // Own tentative allocations until cache insertion has succeeded. In
    // particular, malloc_device and deque growth may throw after malloc_host.
    struct Pending {
        sycl::queue& q;
        Job* host = nullptr;
        Job* dev = nullptr;
        ~Pending() noexcept {
            try { if (host) sycl::free(host, q); } catch (...) {}
            try { if (dev) sycl::free(dev, q); } catch (...) {}
        }
    } pending{q};
    r->host = pending.host = sycl::malloc_host<Job>(size_t(kRing) * kMaxJobs, q);
    r->dev = pending.dev = sycl::malloc_device<Job>(size_t(kRing) * kMaxJobs, q);
    if (!r->host || !r->dev)
        throw std::runtime_error("ds4_gemm_mxfp4_xmx: job ring allocation failed");
    cache.emplace_back(q, std::move(r));
    pending.host = pending.dev = nullptr;
    return *cache.back().second;
}

}  // namespace xm

sycl::event ds4_gemm_mxfp4_xmx(sycl::queue& q, const Ds4MxXmxJob* jobs, uint32_t n_jobs,
                               const std::vector<sycl::event>& deps) {
    using namespace xm;
    auto empty = [&] {
        return q.submit([&](sycl::handler& h) {
            h.depends_on(deps);
            h.single_task([] {});
        });
    };
    if (!n_jobs) return empty();
    if (n_jobs > kMaxJobs)
        throw std::invalid_argument("ds4_gemm_mxfp4_xmx: " + std::to_string(n_jobs) + " jobs > " +
                                    std::to_string(kMaxJobs));
    if (!jobs) throw std::invalid_argument("ds4_gemm_mxfp4_xmx: null job array");
    // Validate the complete request before reserving or modifying any slot.
    std::vector<Job> validated;
    validated.reserve(n_jobs);
    uint64_t wg = 0;
    bool aligned_inputs = true;
    for (uint32_t j = 0; j < n_jobs; ++j) {
        const Ds4MxXmxJob& J = jobs[j];
        if (!J.qs || !J.e || !J.x || !J.y || J.K % BK != 0 || J.N % BN != 0 || J.K == 0 || J.N == 0)
            throw std::invalid_argument("ds4_gemm_mxfp4_xmx: job " + std::to_string(j) +
                                        ": null operand or K % 32 / N % 128 != 0 (K=" +
                                        std::to_string(J.K) + ", N=" + std::to_string(J.N) + ")");
        constexpr uint64_t max_bytes = std::numeric_limits<size_t>::max();
        if (J.M && (uint64_t(J.K) > max_bytes / sizeof(sycl::half) / J.M
                    || uint64_t(J.N) > max_bytes / sizeof(float) / J.M))
            throw std::invalid_argument("ds4_gemm_mxfp4_xmx: tensor byte span overflow");
        aligned_inputs &= (reinterpret_cast<uintptr_t>(J.x) % 16 == 0 &&
                           reinterpret_cast<uintptr_t>(J.qs) % 16 == 0);
        validated.push_back(Job{J.qs, J.e, J.x, J.y, J.M, J.K, J.N, uint32_t(wg)});
        wg += ((uint64_t(J.M) + BM - 1) / BM) * (J.N / BN);
        if (wg > std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument("ds4_gemm_mxfp4_xmx: workgroup index overflow");
    }
    if (!wg) return empty();
    const uint32_t total = uint32_t(wg);
    Ring& R = ring_for(q);
    std::lock_guard<std::mutex> submit_lock(R.submit_mutex);
    const int slot = R.next;
    if (R.used[slot]) R.copied[slot].wait_and_throw();
    Job* hj = R.host + size_t(slot) * kMaxJobs;
    Job* dj = R.dev  + size_t(slot) * kMaxJobs;
    std::copy(validated.begin(), validated.end(), hj);
    // IE_DS4_HOST_TRACE=1: host time of the two submissions, printed every 256 calls.
    static const bool trace = std::getenv("IE_DS4_HOST_TRACE") != nullptr;
    static thread_local double t_cp = 0, t_k = 0;
    static thread_local uint64_t n_tr = 0;
    auto now = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    const double t0 = trace ? now() : 0;
    // The job table goes device-side through a KERNEL, not a memcpy: measured
    // 2026-09-02 (docs/deepseek4/72 Phase F), `q.memcpy` of this 48 B x n table
    // from host USM blocked the HOST for 3.3 ms per launch — the length of the
    // previous group's GEMMs — while a kernel submission takes 10 us and reads
    // the pinned table over PCIe once (see ds4_tp_reduce_host for the same
    // finding at the reduction).  The GEMM kernel then reads the device copy.
    auto cp = q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        if (R.used[slot]) h.depends_on(R.consumed[slot]);
        h.parallel_for(sycl::range<1>(n_jobs), [=](sycl::id<1> i) { dj[i] = hj[i]; });
    });
    R.copied[slot] = R.consumed[slot] = cp;
    R.used[slot] = true;
    R.next = (slot + 1) % kRing;
    const double t1 = trace ? now() : 0;
    // Select a separately compiled aligned kernel only after every base was
    // checked. K is divisible by 32, so all A/B tile offsets preserve 16-byte
    // alignment. Copies target existing vector objects, never cast scalar
    // storage to vector-object pointers. Mixed/misaligned jobs use vec.load.
    auto launch = [&]<bool Aligned>() {
    return ie::ps(q, "ds4_gemm_mxfp4_xmx", [&](sycl::handler& h) {
        h.depends_on(cp);
        sycl::local_accessor<sycl::half, 2> A_smem({BM, BK}, h);
        sycl::local_accessor<sycl::half, 2> B_smem({BK, BN}, h);
        h.parallel_for(sycl::nd_range<1>(size_t(total) * WG, WG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t g   = uint32_t(it.get_group(0));
            const int      lid = int(it.get_local_id(0));
            // The job: last wg0 <= g.
            uint32_t lo = 0, hi = n_jobs - 1;
            while (lo < hi) {
                const uint32_t mid = (lo + hi + 1) / 2;
                if (dj[mid].wg0 <= g) lo = mid; else hi = mid - 1;
            }
            const Job J = dj[lo];
            const uint32_t local = g - J.wg0;
            const uint32_t nbs   = J.N / BN;
            const uint32_t m0    = (local / nbs) * BM;
            const uint32_t n0    = (local % nbs) * BN;
            const int sg_id = lid / SG;
            auto sg = it.get_sub_group();

            mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> acc[MR];
            #pragma unroll
            for (int mr = 0; mr < MR; ++mr) mat::joint_matrix_fill(sg, acc[mr], 0.0f);

            const uint32_t bpc  = J.K / 32u;
            const uint8_t* qcol = J.qs + uint64_t(n0 + lid) * (uint64_t(J.K) / 2);
            const uint8_t* ecol = J.e  + uint64_t(n0 + lid) * bpc;
            const int  ar   = lid / 4, ac = (lid % 4) * 8;   // A: row, first column of 8
            const bool a_ok = (m0 + uint32_t(ar)) < J.M;
            const sycl::half* arow = a_ok ? J.x + uint64_t(m0 + ar) * J.K : J.x;

            for (uint32_t kb = 0; kb < bpc; ++kb) {
                // ---- A tile: 128 lanes x 8 halves = 32 x 32 ----
                if (a_ok) {
                    sycl::vec<sycl::half, 8> v;
                    const sycl::half* ap = arow + uint64_t(kb) * BK + ac;
                    if constexpr (Aligned)
                        __builtin_memcpy(&v, __builtin_assume_aligned(ap, 16), sizeof(v));
                    else
                        v.load(0, ap);
                    #pragma unroll
                    for (int dc = 0; dc < 8; ++dc) A_smem[ar][ac + dc] = v[dc];
                } else {
                    #pragma unroll
                    for (int dc = 0; dc < 8; ++dc) A_smem[ar][ac + dc] = sycl::half(0);
                }
                // ---- B tile: this lane's column, one MXFP4 block -> 32 halves ----
                {
                    const float d = mxfp4_e8m0_half(ecol[kb]);
                    sycl::vec<uint8_t, 16> bytes;
                    const uint8_t* bp = qcol + uint64_t(kb) * 16;
                    if constexpr (Aligned)
                        __builtin_memcpy(&bytes, __builtin_assume_aligned(bp, 16), sizeof(bytes));
                    else
                        bytes.load(0, bp);
                    const sycl::uint4 w4 = bytes.as<sycl::uint4>();
                    const uint32_t wd[4] = {w4.x(), w4.y(), w4.z(), w4.w()};
                    #pragma unroll
                    for (int wi = 0; wi < 4; ++wi) {
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const uint32_t byte = (wd[wi] >> (j * 8)) & 0xFFu;
                            const int idx = wi * 4 + j;               // 0..15
                            B_smem[idx][lid]      = sycl::half(d * float(mxfp4_nibble_int(byte & 0x0Fu)));
                            B_smem[16 + idx][lid] = sycl::half(d * float(mxfp4_nibble_int(byte >> 4)));
                        }
                    }
                }
                sycl::group_barrier(it.get_group());

                #pragma unroll
                for (int kk = 0; kk < BK; kk += TK) {
                    mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b, TK, TN,
                                      mat::layout::row_major> b_tile;
                    mat::joint_matrix_load(sg, b_tile,
                        B_smem.template get_multi_ptr<sycl::access::decorated::no>() +
                            kk * BN + sg_id * TN,
                        /*stride=*/BN);
                    #pragma unroll
                    for (int mr = 0; mr < MR; ++mr) {
                        // The last M tile can have fewer than four live
                        // accumulator blocks. All lanes agree on this bound;
                        // skip only blocks that the store below also omits.
                        if (m0 + uint32_t(mr * TM) >= J.M) continue;
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a, TM, TK,
                                          mat::layout::row_major> a_tile;
                        mat::joint_matrix_load(sg, a_tile,
                            A_smem.template get_multi_ptr<sycl::access::decorated::no>() +
                                (mr * TM) * BK + kk,
                            /*stride=*/BK);
                        mat::joint_matrix_mad(sg, acc[mr], a_tile, b_tile, acc[mr]);
                    }
                }
                sycl::group_barrier(it.get_group());
            }

            // ---- store: only rows < M, element by element ----
            #pragma unroll
            for (int mr = 0; mr < MR; ++mr) {
                const uint32_t rbase = m0 + uint32_t(mr) * TM;
                if (rbase >= J.M) continue;                       // uniform across the sub-group
                float* yb = J.y + uint64_t(rbase) * J.N + n0 + uint32_t(sg_id) * TN;
                const uint32_t rows_left = J.M - rbase;
                const uint32_t N = J.N;
                imat::joint_matrix_apply(sg, acc[mr], [&](float& v, size_t r, size_t c) {
                    if (r < rows_left) yb[r * N + c] = v;
                });
            }
        });
    });
    };
    auto ev = aligned_inputs ? launch.template operator()<true>() : launch.template operator()<false>();
    R.consumed[slot] = ev;
    if (trace) {
        const double t2 = now();
        t_cp += t1 - t0; t_k += t2 - t1;
        if (++n_tr % 256 == 0)
            std::fprintf(stderr, "[ds4-xmx] launches=%llu avg ms: job-table memcpy submit %.3f  kernel submit %.3f\n",
                         (unsigned long long)n_tr, t_cp / n_tr * 1e3, t_k / n_tr * 1e3);
    }
    return ev;
}

// IE_DS4_EXPERT_XMX_FUSED=0 keeps the dequant + oneDNN route (the A/B control).
static bool ds4_expert_xmx_fused_on() {
    static const bool on = [] {
        const char* e = std::getenv("IE_DS4_EXPERT_XMX_FUSED");
        return !(e && *e == '0');
    }();
    return on;
}

std::string ds4_expert_gemm_xmx_grouped(sycl::queue& q, const DS4XmxGemmJob* jobs,
                                        uint32_t n_jobs, const sycl::half* x_f16,
                                        uint32_t H_in, sycl::half* w16_scratch,
                                        uint64_t w16_capacity) {
    const std::string prefix = "ds4_expert_gemm_xmx_grouped: ";
    if (!n_jobs) return {};
    if (!q.is_in_order()) return prefix + "requires an in-order queue";
    if (!jobs || n_jobs > xm::kMaxJobs) return prefix + "null job array or too many jobs";
    const bool fused = ds4_expert_xmx_fused_on();
    constexpr uint64_t limit = std::numeric_limits<ptrdiff_t>::max();
    // Check an expert slice without overflowing either stride multiplication
    // or its end. Strides are in bytes for both MXFP4 planes.
    auto plane_ok = [](uint64_t stride, uint64_t span, uint32_t e) {
        return stride >= span && span <= limit && (!e || stride <= (limit - span) / e);
    };
    uint64_t groups = 0;
    // Validate EVERY job before pointer arithmetic, staging, or submission.
    for (uint32_t j = 0; j < n_jobs; ++j) {
        const auto& J = jobs[j];
        if (!J.M) continue;
        const auto& B = J.bank;
        if (B.dtype != DType::kMXFP4 || !B.K || B.K % 32 || !B.N || B.N % 128 || B.K != H_in)
            return prefix + "job " + std::to_string(j) + ": requires MXFP4, K=H_in divisible by 32, N divisible by 128";
        if (J.e >= B.E || !B.mx_qs || !B.mx_e || !J.y || !x_f16)
            return prefix + "job " + std::to_string(j) + ": null operand or expert index outside bank";
        const uint64_t rows_end = uint64_t(J.row0) + J.M;
        if (B.K > limit / sizeof(sycl::half) / rows_end || B.N > limit / sizeof(float) / J.M ||
            !plane_ok(B.mx_qs_stride, uint64_t(B.K / 2) * B.N, J.e) ||
            !plane_ok(B.mx_e_stride, uint64_t(B.K / 32) * B.N, J.e))
            return prefix + "tensor row/plane byte span overflow or short expert stride";
        groups += ((uint64_t(J.M) + xm::BM - 1) / xm::BM) * (B.N / xm::BN);
        if (groups > std::numeric_limits<uint32_t>::max()) return prefix + "workgroup index overflow";
        if (!fused && (!w16_scratch || uint64_t(B.K) * B.N > w16_capacity ||
                       B.K > limit / sizeof(sycl::half) / B.N))
            return prefix + "null, short, or unrepresentable w16 scratch";
    }
    if (!groups) return {};
    try {
        static thread_local std::vector<Ds4MxXmxJob> xj;
        xj.clear();
        for (uint32_t j = 0; j < n_jobs; ++j) {
            const auto& J = jobs[j];
            if (!J.M) continue;
            const auto& B = J.bank;
            xj.push_back({B.mx_qs + J.e * B.mx_qs_stride, B.mx_e + J.e * B.mx_e_stride,
                          x_f16 + uint64_t(J.row0) * B.K, J.y, J.M, B.K, B.N});
        }
        if (fused) {
            ds4_gemm_mxfp4_xmx(q, xj.data(), uint32_t(xj.size()));
        } else {
            for (const auto& J : xj) {
                // The current oneDNN BMG kernel misreads half-aligned X.
                // Stage only operands lacking 16-byte alignment. The fused
                // kernel above uses scalar-aligned vec.load and needs no copy.
                struct AlignedOperands {
                    sycl::queue& q;
                    sycl::half* x = nullptr;
                    sycl::half* w = nullptr;
                    float* y = nullptr;
                    ~AlignedOperands() noexcept {
                        if (!x && !w && !y) return;
                        try { q.wait(); } catch (...) {}
                        try { if (x) sycl::free(x, q); } catch (...) {}
                        try { if (w) sycl::free(w, q); } catch (...) {}
                        try { if (y) sycl::free(y, q); } catch (...) {}
                    }
                } aligned{q};
                const sycl::half* ax = J.x;
                sycl::half* aw = w16_scratch;
                if (reinterpret_cast<uintptr_t>(ax) % 16) {
                    aligned.x = sycl::aligned_alloc_device<sycl::half>(64, uint64_t(J.M) * J.K, q);
                    if (!aligned.x) throw std::bad_alloc();
                    q.memcpy(aligned.x, ax, uint64_t(J.M) * J.K * sizeof(sycl::half));
                    ax = aligned.x;
                }
                if (reinterpret_cast<uintptr_t>(aw) % 16) {
                    aligned.w = sycl::aligned_alloc_device<sycl::half>(64, uint64_t(J.N) * J.K, q);
                    if (!aligned.w) throw std::bad_alloc();
                    aw = aligned.w;
                }
                float* ay = J.y;
                if (reinterpret_cast<uintptr_t>(ay) % 16) {
                    aligned.y = sycl::aligned_alloc_device<float>(64, uint64_t(J.M) * J.N, q);
                    if (!aligned.y) throw std::bad_alloc();
                    ay = aligned.y;
                }
                auto dq = ds4_dequant_mxfp4_w16(q, J.qs, J.e, aw, J.K, J.N);
                auto done = gemm_nt_f16_onednn(q, ax, aw, ay, J.M, J.N, J.K, {dq});
                if (aligned.y) done = q.memcpy(J.y, ay, uint64_t(J.M) * J.N * sizeof(float));
                if (aligned.x || aligned.w || aligned.y) done.wait_and_throw();
            }
        }
    } catch (const std::exception& ex) { return prefix + ex.what(); }
    return {};
}

uint64_t ds4_expert_batch_ws_bytes(uint32_t max_tokens, uint32_t top_k,
                                   uint32_t H, uint32_t EF) noexcept {
    // Mirrors the allocations below, buffer for buffer.
    const uint64_t R = uint64_t(max_tokens) * top_k;
    return R * H * sizeof(sycl::half)                    // xp
         + R * (H / 32) * sizeof(block_q8_1x)            // xp_q8
         + 2 * R * EF * sizeof(sycl::half)               // g_h, u_h
         + 3 * R * EF * sizeof(float)                    // g_f, u_f, h_f
         + R * EF * sizeof(sycl::half)                   // h_h
         + R * (EF / 32) * sizeof(block_q8_1x)           // h_q8
         + R * H * sizeof(sycl::half)                    // yp
         + 3 * R * sizeof(int32_t)                        // row_tok, tk2p, w_pk
         + (ds4_expert_xmx_on() ? uint64_t(H) * EF * sizeof(sycl::half)   // w16
                                + R * H * sizeof(float) : 0);            // y_f32
}

std::string ds4_expert_batch_ws_alloc(sycl::queue& q, uint32_t max_tokens, uint32_t top_k,
                                      uint32_t H, uint32_t EF, DS4ExpertBatchWs& ws,
                                      uint32_t max_jobs) {
    if (H % 32 != 0 || EF % 32 != 0)
        return "ds4_expert_batch_ws_alloc: H and EF must be % 32";
    if (max_tokens == 0 || top_k == 0)
        return "ds4_expert_batch_ws_alloc: max_tokens and top_k must be non-zero";
    // quantize_q8_1 takes a uint32 element count and the batch feeds it the
    // WHOLE packed buffer in one call, so the packed element count has to fit.
    if (uint64_t(max_tokens) * top_k * std::max(H, EF) > 0xFFFFFFFFull)
        return "ds4_expert_batch_ws_alloc: max_tokens*top_k*max(H,EF) overflows uint32";
    ws = DS4ExpertBatchWs{};
    ws.max_tokens = max_tokens; ws.top_k = top_k; ws.H = H; ws.EF = EF;
    ws.rows = max_tokens * top_k;
    const uint64_t R = ws.rows;
    ws.xp      = sycl::malloc_device<sycl::half>(R * H, q);
    ws.xp_q8   = sycl::malloc_device<block_q8_1x>(R * (H / 32), q);
    ws.g_h     = sycl::malloc_device<sycl::half>(R * EF, q);
    ws.u_h     = sycl::malloc_device<sycl::half>(R * EF, q);
    ws.g_f     = sycl::malloc_device<float>(R * EF, q);
    ws.u_f     = sycl::malloc_device<float>(R * EF, q);
    ws.h_f     = sycl::malloc_device<float>(R * EF, q);
    ws.h_h     = sycl::malloc_device<sycl::half>(R * EF, q);
    ws.h_q8    = sycl::malloc_device<block_q8_1x>(R * (EF / 32), q);
    ws.yp      = sycl::malloc_device<sycl::half>(R * H, q);
    ws.row_tok = sycl::malloc_device<int32_t>(R, q);
    ws.tk2p    = sycl::malloc_device<int32_t>(R, q);
    ws.w_pk    = sycl::malloc_device<float>(R, q);
    if (!ws.xp || !ws.xp_q8 || !ws.g_h || !ws.u_h || !ws.g_f || !ws.u_f || !ws.h_f ||
        !ws.h_h || !ws.h_q8 || !ws.yp || !ws.row_tok || !ws.tk2p || !ws.w_pk) {
        ds4_expert_batch_ws_free(q, ws);
        return "ds4_expert_batch_ws_alloc: device alloc failed";
    }
    if (std::string e = ds4_gemm_group_ws_alloc(q, max_jobs, ws.grp); !e.empty()) {
        ds4_expert_batch_ws_free(q, ws);
        return e;
    }
    // One dequantised weight buffer for the opt-in XMX route.  H*EF halves
    // covers both projection shapes ([H,EF] gate/up and [EF,H] down).
    if (ds4_expert_xmx_on()) {
        ws.w16_cap = uint64_t(H) * EF;
        ws.w16 = sycl::malloc_device<sycl::half>(ws.w16_cap, q);
        ws.y_f32 = sycl::malloc_device<float>(R * H, q);
        if (!ws.w16 || !ws.y_f32) { ds4_expert_batch_ws_free(q, ws); return
            "ds4_expert_batch_ws_alloc: XMX scratch alloc failed"; }
    }
    ws.jobs.reserve(max_jobs);
    ws.routes.resize(max_tokens);
    for (auto& r : ws.routes) r.reserve(top_k);
    return {};
}

void ds4_expert_batch_ws_free(sycl::queue& q, DS4ExpertBatchWs& ws) {
    for (void* p : {static_cast<void*>(ws.xp), ws.xp_q8, static_cast<void*>(ws.g_h),
                    static_cast<void*>(ws.u_h), static_cast<void*>(ws.g_f),
                    static_cast<void*>(ws.u_f), static_cast<void*>(ws.h_f),
                    static_cast<void*>(ws.h_h), ws.h_q8, static_cast<void*>(ws.yp),
                    static_cast<void*>(ws.row_tok), static_cast<void*>(ws.tk2p),
                    static_cast<void*>(ws.w_pk), static_cast<void*>(ws.w16),
                    static_cast<void*>(ws.y_f32)})
        if (p) sycl::free(p, q);
    ds4_gemm_group_ws_free(q, ws.grp);
    ws = DS4ExpertBatchWs{};
}

// ---------------------------------------------------------------------------
// the expert-major routed-expert block
// ---------------------------------------------------------------------------
std::string ds4_experts_forward_batched(sycl::queue& q, const Ds4ExpertBankFn& banks,
                                        const float* x, const int32_t* topk_idx,
                                        const float* topk_w, float* y,
                                        uint32_t T, uint32_t H, uint32_t EFc,
                                        uint32_t top_k, uint32_t n_experts,
                                        float swiglu_limit, DS4ExpertBatchWs& ws) {
    if (!q.is_in_order()) return "ds4_experts_forward_batched: requires an in-order queue";
    if (T == 0) return {};
    if (T > ws.max_tokens || top_k != ws.top_k || H != ws.H || EFc > ws.EF)
        return "ds4_experts_forward_batched: batch workspace too small (T=" +
               std::to_string(T) + "/" + std::to_string(ws.max_tokens) + ", top_k=" +
               std::to_string(top_k) + "/" + std::to_string(ws.top_k) + ", H=" +
               std::to_string(H) + "/" + std::to_string(ws.H) + ", EFc=" +
               std::to_string(EFc) + "/" + std::to_string(ws.EF) + ")";
    if (EFc % 32 != 0) return "ds4_experts_forward_batched: EFc must be % 32";

    // ---- host counting sort: tokens grouped by expert, kslot order preserved
    ws.routes.resize(T);
    for (uint32_t t = 0; t < T; ++t) {
        ws.routes[t].clear();
        ws.routes[t].reserve(top_k);
        for (uint32_t k = 0; k < top_k; ++k) {
            const int32_t e = topk_idx[uint64_t(t) * top_k + k];
            if (e < 0 || uint32_t(e) >= n_experts)
                return "ds4_experts_forward_batched: expert id " + std::to_string(e) +
                       " outside [0," + std::to_string(n_experts) + ")";
            ws.routes[t].push_back({uint32_t(e), topk_w[uint64_t(t) * top_k + k]});
        }
    }
    build_moe_packing(ws.routes, n_experts, top_k, ws.pk);
    const uint32_t TK = T * top_k;

    q.memcpy(ws.row_tok, ws.pk.sorted_idx.data(),     TK * sizeof(int32_t));
    q.memcpy(ws.tk2p,    ws.pk.tk_to_packed.data(),   TK * sizeof(int32_t));
    q.memcpy(ws.w_pk,    ws.pk.weights_packed.data(), TK * sizeof(float));

    // ---- one activation pass for the WHOLE batch -------------------------
    // Both stages are blockwise/element-wise and H is a multiple of 32, so
    // running them over the concatenated packed rows is bit-for-bit the same as
    // the token-major path's per-token calls (a Q8_1 block never straddles a row).
    ds4_expert_gather_cast(q, x, ws.row_tok, ws.xp, TK, H);
    quantize_q8_1(q, ws.xp, ws.xp_q8, TK * H);

    // ---- resolve every occupied expert ONCE ------------------------------
    struct Job { uint32_t off, n_e; DS4ExpertBank gate, up, down; };
    std::vector<Job> jobs;
    const auto& off = ws.pk.expert_offsets;
    for (uint32_t e = 0; e < n_experts; ++e) {
        const uint32_t o = off[e], n_e = off[e + 1] - o;
        if (n_e == 0) continue;
        Job j{o, n_e, {}, {}, {}};
        if (!banks(e, j.gate, j.up, j.down))
            return "ds4_experts_forward_batched: expert " + std::to_string(e) +
                   " has no resident bank — refusing to fabricate an expert output";
        if (j.gate.K != H || j.gate.N != EFc || j.up.K != H || j.up.N != EFc ||
            j.down.K != EFc || j.down.N != H)
            return "ds4_experts_forward_batched: expert " + std::to_string(e) +
                   " bank shape mismatch (gate [" + std::to_string(j.gate.K) + "," +
                   std::to_string(j.gate.N) + "], down [" + std::to_string(j.down.K) + "," +
                   std::to_string(j.down.N) + "] vs H=" + std::to_string(H) + " EFc=" +
                   std::to_string(EFc) + ")";
        jobs.push_back(j);
    }

    // ---- gate + up, ONE LAUNCH for every occupied expert -------------------
    // Gate and up go in the same group deliberately: they have the same (K, N),
    // read the same activation rows, and — when the layer is not blk.26-shaped —
    // the same dtype, so they collapse into a single kernel.  When they do NOT
    // (a layer with IQ3_XXS gate/up and MXFP4 down, or the reverse) the bucketing
    // in `ds4_expert_gemm_q8_grouped` splits them by their OWN dtypes and neither
    // this code nor the caller has a dtype branch.
    auto* xq8 = static_cast<block_q8_1x*>(ws.xp_q8);
    ws.jobs.clear();
    for (const Job& j : jobs) {
        const void* xe = xq8 + uint64_t(j.off) * (H / 32);
        ws.jobs.push_back({j.gate, 0, j.n_e, xe, ws.g_h + uint64_t(j.off) * EFc});
        ws.jobs.push_back({j.up,   0, j.n_e, xe, ws.u_h + uint64_t(j.off) * EFc});
    }
    if (std::string e = ds4_expert_gemm_q8_grouped(q, ws.jobs.data(),
                                                   uint32_t(ws.jobs.size()), ws.grp);
        !e.empty())
        return e;

    // ---- clamped SwiGLU for the whole batch in one pass -------------------
    // ONE launch, not four.  The chain was cast_f16_f32 x2 -> swiglu -> cast_f32_f16
    // purely because the SwiGLU was written fp32-in/fp32-out while its producer and
    // its consumer are both fp16; `ds4_swiglu_clamped_h` does the same arithmetic
    // with fp16 ends and is gated for EXACT BIT EQUALITY over all 63,490 non-NaN
    // fp16 patterns at four clamp limits (deepseek4_ops_gate_test).  Measured
    // 2.19 -> 0.73 us GPU and 8.2 -> 2.0 us wall per occurrence; at decode the
    // three removed launches matter more than the GPU time.
    const uint64_t HN = uint64_t(TK) * EFc;
    ds4_swiglu_clamped_h(q, ws.g_h, ws.u_h, ws.h_h, HN, swiglu_limit);
    quantize_q8_1(q, ws.h_h, ws.h_q8, uint32_t(HN));

    // ---- down, ONE LAUNCH ------------------------------------------------
    auto* hq8 = static_cast<block_q8_1x*>(ws.h_q8);
    ws.jobs.clear();
    for (const Job& j : jobs)
        ws.jobs.push_back({j.down, 0, j.n_e, hq8 + uint64_t(j.off) * (EFc / 32),
                           ws.yp + uint64_t(j.off) * H});
    if (std::string e = ds4_expert_gemm_q8_grouped(q, ws.jobs.data(),
                                                   uint32_t(ws.jobs.size()), ws.grp);
        !e.empty())
        return e;

    // ---- routing weight AFTER down, then the fp32 accumulate --------------
    ds4_expert_scatter_accum(q, ws.yp, ws.tk2p, ws.w_pk, y, T, top_k, H);
    q.wait();
    return {};
}

// ---------------------------------------------------------------------------
// workspace
// ---------------------------------------------------------------------------
std::string ds4_expert_ws_alloc(sycl::queue& q, uint32_t H, uint32_t EF,
                                DS4ExpertWorkspace& ws) {
    if (H % 32 != 0 || EF % 32 != 0) return "ds4_expert_ws_alloc: H and EF must be % 32";
    ws = DS4ExpertWorkspace{};
    ws.H = H; ws.EF = EF;
    ws.xh     = sycl::malloc_device<sycl::half>(H, q);
    ws.x_q8   = sycl::malloc_device<block_q8_1x>(H / 32, q);
    ws.gate_h = sycl::malloc_device<sycl::half>(EF, q);
    ws.up_h   = sycl::malloc_device<sycl::half>(EF, q);
    ws.gate_f = sycl::malloc_device<float>(EF, q);
    ws.up_f   = sycl::malloc_device<float>(EF, q);
    ws.h_f    = sycl::malloc_device<float>(EF, q);
    ws.h_h    = sycl::malloc_device<sycl::half>(EF, q);
    ws.h_q8   = sycl::malloc_device<block_q8_1x>(EF / 32, q);
    ws.y_h    = sycl::malloc_device<sycl::half>(H, q);
    if (!ws.xh || !ws.x_q8 || !ws.gate_h || !ws.up_h || !ws.gate_f || !ws.up_f ||
        !ws.h_f || !ws.h_h || !ws.h_q8 || !ws.y_h) {
        ds4_expert_ws_free(q, ws);
        return "ds4_expert_ws_alloc: device alloc failed";
    }
    return {};
}

void ds4_expert_ws_free(sycl::queue& q, DS4ExpertWorkspace& ws) {
    for (void* p : {static_cast<void*>(ws.xh), ws.x_q8, static_cast<void*>(ws.gate_h),
                    static_cast<void*>(ws.up_h), static_cast<void*>(ws.gate_f),
                    static_cast<void*>(ws.up_f), static_cast<void*>(ws.h_f),
                    static_cast<void*>(ws.h_h), ws.h_q8, static_cast<void*>(ws.y_h)})
        if (p) sycl::free(p, q);
    ws = DS4ExpertWorkspace{};
}

// ---------------------------------------------------------------------------
// routed-expert block
// ---------------------------------------------------------------------------
std::string ds4_experts_forward(sycl::queue& q,
                                const DS4ExpertBank& gate, const DS4ExpertBank& up,
                                const DS4ExpertBank& down,
                                const float* x, const int32_t* topk_idx,
                                const float* topk_w, float* y,
                                uint32_t T, uint32_t H, uint32_t EF, uint32_t top_k,
                                float swiglu_limit, DS4ExpertWorkspace& ws,
                                bool f16_activation) {
    if (!q.is_in_order()) return "ds4_experts_forward: requires an in-order queue";
    if (ws.H != H || ws.EF != EF) return "ds4_experts_forward: workspace shape mismatch";
    if (gate.K != H || gate.N != EF) return "ds4_experts_forward: gate bank shape mismatch";
    if (up.K != H   || up.N   != EF) return "ds4_experts_forward: up bank shape mismatch";
    if (down.K != EF || down.N != H) return "ds4_experts_forward: down bank shape mismatch";

    q.memset(y, 0, uint64_t(T) * H * sizeof(float));
    for (uint32_t t = 0; t < T; ++t) {
        // One Q8_1 quantisation of the token, shared by every routed expert's
        // gate and up GEMV (the activation is identical across k).
        cast_fp32_to_fp16(q, x + uint64_t(t) * H, ws.xh, H);
        if (!f16_activation) quantize_q8_1(q, ws.xh, ws.x_q8, H);
        const sycl::half* xa = f16_activation ? ws.xh : nullptr;
        for (uint32_t k = 0; k < top_k; ++k) {
            const int32_t e = topk_idx[uint64_t(t) * top_k + k];
            if (e < 0 || uint32_t(e) >= gate.E)
                return "ds4_experts_forward: expert id " + std::to_string(e) +
                       " outside the resident bank [0," + std::to_string(gate.E) + ")";
            // Same two calls regardless of dtype — blk.26 (MXFP4 gate/up) and
            // every other layer (IQ3_XXS gate/up) take this identical path.
            ds4_expert_gemv(q, gate, uint32_t(e), ws.x_q8, xa, ws.gate_h);
            ds4_expert_gemv(q, up,   uint32_t(e), ws.x_q8, xa, ws.up_h);
            // Phase-2, parity-gated clamped SwiGLU: gate clamped ABOVE only,
            // up clamped both sides, silu(gate)*up.  ONE launch — the fp16-ended
            // form is bit-identical to the cast/cast/swiglu/cast chain it replaces
            // (exhaustively gated over every non-NaN fp16 pattern), and this is the
            // per-(token, expert) loop, so it removes 3 launches per iteration.
            ds4_swiglu_clamped_h(q, ws.gate_h, ws.up_h, ws.h_h, EF, swiglu_limit);
            if (!f16_activation) quantize_q8_1(q, ws.h_h, ws.h_q8, EF);
            ds4_expert_gemv(q, down, uint32_t(e), ws.h_q8,
                            f16_activation ? ws.h_h : nullptr, ws.y_h);
            // Routing weight applied AFTER down (reference: `F.linear(...) *
            // top_k_weights`), then accumulated in fp32.
            ds4_expert_accum(q, ws.y_h, topk_w[uint64_t(t) * top_k + k],
                             y + uint64_t(t) * H, H);
        }
    }
    q.wait();
    return {};
}

// ---------------------------------------------------------------------------
// host reference — DeepseekV4Experts.forward over dequantised fp32 weights
// ---------------------------------------------------------------------------
void ds4_experts_forward_ref(const float* x, const float* gate_up, const float* down,
                             const int32_t* topk_idx, const float* topk_w, float* y,
                             uint32_t T, uint32_t H, uint32_t EF, uint32_t top_k,
                             uint32_t E, float swiglu_limit) {
    (void)E;
    std::vector<double> g(EF), u(EF), hv(EF);
    for (uint64_t i = 0; i < uint64_t(T) * H; ++i) y[i] = 0.f;
    for (uint32_t t = 0; t < T; ++t) {
        const float* xt = x + uint64_t(t) * H;
        for (uint32_t k = 0; k < top_k; ++k) {
            const uint32_t e = uint32_t(topk_idx[uint64_t(t) * top_k + k]);
            // gate_up_proj[e] is [2*EF, H]: rows [0,EF) = gate, [EF,2EF) = up.
            const float* GU = gate_up + uint64_t(e) * 2 * EF * H;
            for (uint32_t n = 0; n < EF; ++n) {
                double sg = 0.0, su = 0.0;
                const float* rg = GU + uint64_t(n) * H;
                const float* ru = GU + uint64_t(EF + n) * H;
                for (uint32_t h = 0; h < H; ++h) {
                    sg += double(rg[h]) * double(xt[h]);
                    su += double(ru[h]) * double(xt[h]);
                }
                g[n] = sg; u[n] = su;
            }
            // _apply_gate: gate.clamp(max=limit); up.clamp(-limit, limit);
            //              silu(gate) * up
            for (uint32_t n = 0; n < EF; ++n) {
                const double gc = std::min(g[n], double(swiglu_limit));
                const double uc = std::min(std::max(u[n], -double(swiglu_limit)),
                                           double(swiglu_limit));
                hv[n] = (gc / (1.0 + std::exp(-gc))) * uc;
            }
            const float* D = down + uint64_t(e) * H * EF;   // [H, EF]
            const double w = double(topk_w[uint64_t(t) * top_k + k]);
            for (uint32_t o = 0; o < H; ++o) {
                double s = 0.0;
                const float* rd = D + uint64_t(o) * EF;
                for (uint32_t n = 0; n < EF; ++n) s += double(rd[n]) * hv[n];
                y[uint64_t(t) * H + o] += float(w * s);
            }
        }
    }
}

}  // namespace ie
