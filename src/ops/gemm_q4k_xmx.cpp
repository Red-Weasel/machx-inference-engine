// src/ops/gemm_q4k_xmx.cpp — Q4_K weight GEMM via SYCL joint_matrix (XMX/DPAS).
//
// Same input/output contract as gemm_q4_K (the SLM-tiled scalar version):
//   y[M, N] = A[M, K] @ W[K, N]   with W = Q4_K-packed along K, super-blocks of 256.
//
// Layout (2026-05-05 update: M_TILE_MAX raised from 8 → 16):
//   WG tile    BM × BN = M × 64        (M ≤ 16, processed in M_GROUPS=2 of TM=8)
//   K tile     BK = 256 (one Q4_K super-block)
//   SGs/WG    4 (N axis only — BM is whole-WG, no M decomposition)
//   Per-SG    one TN=16 N-tile, M_GROUPS_MAX=2 stacked TM=8 sub-tiles
//   XMX inner TM=8, TN=16, TK=16 → 16 K-tiles × M_GROUPS mat_mads per SG per K-block
//
// SLM
//   A_smem[BM × BK] = 16 × 256 fp16 =  8 KiB
//   B_smem[BK × BN] = 256 × 64 fp16 = 32 KiB
//   C_scratch     = 4 × M_GROUPS × TM × TN fp32 = 4 KiB
//   Total per WG ≈ 44 KiB.  ≤ 192 KiB Xe-core SLM with comfortable occupancy.
//
// Each B-tile dequant is now reused across M_GROUPS_MAX=2 mat_mads per K-tile,
// halving the weight-read amortization cost relative to the M_TILE=8 baseline:
// gemv_q_T(T=512) drops from 64 launches/projection to 32, and every B_smem
// dequant pays for 2× as many output rows.

#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include "ie/kernel_profiler.hpp"

namespace ie {

namespace mat = sycl::ext::oneapi::experimental::matrix;
using fp16 = sycl::half;

namespace {

constexpr int SG_SIZE  = 16;
constexpr int BN       = 64;          // N tile per WG
constexpr int BK       = 256;         // K tile = 1 Q4_K block
constexpr int TM       = 8;           // XMX M (joint_matrix shape, fixed)
constexpr int TN       = 16;          // XMX N
constexpr int TK       = 16;          // XMX K
constexpr int M_GROUPS_MAX  = 2;            // 2 × TM = up to M=16 per WG
constexpr int M_TILE_MAX    = TM * M_GROUPS_MAX;   // = 16
constexpr int N_TILES_PER_WG  = BN / TN;     // = 4
constexpr int K_TILES_PER_BLK = BK / TK;    // = 16
constexpr int WG_ITEMS = N_TILES_PER_WG * SG_SIZE;   // 4 × 16 = 64

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

// Dequantize a single Q4_K super-block (256 elements) for ONE output column n
// into a 256-element fp16 strip starting at `dst`.  Caller arranges `dst` to
// be the column-n stripe of the B_smem [256, BN] tile (stride = BN per K).
inline void dequant_q4k_block_to_stripe(const block_q4_K& blk,
                                        fp16* dst, int stride) {
    // The standard Q4_K decode: for each of 8 sub-blocks of 32 elements,
    // unpack the (s, m) scale pair from blk.scales[12], reconstruct
    //   x[i] = d * s_raw * q4 - d * m_raw
    // where q4 ∈ [0, 15] is a 4-bit code from blk.qs.
    const float d_super = dev_fp16_to_fp32(blk.d);
    const float m_super = dev_fp16_to_fp32(blk.dmin);
    #pragma unroll
    for (int sub = 0; sub < 8; ++sub) {
        // get_scale_min_k4: extract 6-bit s and m for sub-block `sub`.
        uint8_t s_raw, m_raw;
        if (sub < 4) {
            s_raw = blk.scales[sub]     & 0x3F;
            m_raw = blk.scales[sub + 4] & 0x3F;
        } else {
            s_raw = (blk.scales[sub + 4] & 0x0F) | ((blk.scales[sub - 4] >> 6) << 4);
            m_raw = (blk.scales[sub + 4] >>   4) | ((blk.scales[sub    ] >> 6) << 4);
        }
        const float scale = d_super * float(s_raw);
        const float bias  = m_super * float(m_raw);

        // Each sub-block is 32 elements packed as 16 bytes in blk.qs[].
        // Layout: low nibble of byte i carries q4 for elem i within a 32-elem
        // group; high nibble carries the next group's q4 (super-blocks pair
        // sub 0&2, 1&3, 4&6, 5&7).  Mirror gemm_q4_K's lattice mapping:
        //   group g = sub >> 1, hi_nib = sub & 1.
        const int g       = sub >> 1;
        const int hi_nib  = sub & 1;
        const int qs_base = g * 32;       // 32-byte qs group
        const int dst_base = (g * 64) + (hi_nib * 32);

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            const uint8_t qb = blk.qs[qs_base + i];
            const int q4 = hi_nib ? (qb >> 4) : (qb & 0x0F);
            const float v = scale * float(q4) - bias;
            dst[(dst_base + i) * stride] = fp16(v);
        }
    }
}

}  // namespace

sycl::event gemm_q4_K_xmx(sycl::queue& q,
                          const fp16* A, const void* W_packed,
                          fp16* y,
                          uint32_t M, uint32_t K, uint32_t N,
                          const std::vector<sycl::event>& deps) {
    if (M == 0 || M > uint32_t(M_TILE_MAX)) return {};
    if (K % BK != 0 || N % BN != 0) return {};

    const auto* W = static_cast<const block_q4_K*>(W_packed);
    const uint32_t blocks_per_col = K / 256;          // = K / BK
    const uint32_t wgs_n = N / BN;
    // # of TM=8 row-groups actually needed for this M (1 or 2).
    const uint32_t m_groups = (M + TM - 1) / TM;

    return ie::ps(q, "gemm_q4k_xmx", [&](sycl::handler& h) {
        h.depends_on(deps);
        // A_smem[BM=16 × BK=256], B_smem[BK=256 × BN=64].
        sycl::local_accessor<fp16, 2> A_smem({M_TILE_MAX, BK}, h);
        sycl::local_accessor<fp16, 2> B_smem({BK, BN}, h);
        // C_scratch sized for M_GROUPS_MAX × N_TILES_PER_WG × TM × TN fp32.
        sycl::local_accessor<float, 1> C_scratch(
            M_GROUPS_MAX * N_TILES_PER_WG * TM * TN, h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(wgs_n) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid    = uint32_t(it.get_local_id(0));
            const uint32_t wgid   = uint32_t(it.get_group(0));
            const uint32_t sg_id  = lid / SG_SIZE;        // 0..3 — which N-tile
            const uint32_t lane   = lid % SG_SIZE;
            const uint32_t wg_n   = wgid * BN;
            auto sg = it.get_sub_group();

            // M_GROUPS_MAX accumulators per SG, each TM × TN = 8 × 16.  Group g
            // covers M-rows [g*TM, g*TM+TM).  Both groups share the same
            // B-tile per K-tile, so each B_smem dequant amortizes across them.
            mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator,
                              TM, TN> acc[M_GROUPS_MAX];
            #pragma unroll
            for (int g = 0; g < M_GROUPS_MAX; ++g) {
                mat::joint_matrix_fill(sg, acc[g], 0.0f);
            }

            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                // Cooperative A load: [M, BK=256] fp16 = M*256 elements.
                {
                    const uint32_t total = M * BK;
                    for (uint32_t i = lid; i < total; i += WG_ITEMS) {
                        const uint32_t m = i / BK;
                        const uint32_t kk = i % BK;
                        A_smem[m][kk] = A[uint64_t(m) * K + b * BK + kk];
                    }
                }
                // Cooperative B dequant: 64 lanes, BN=64 columns.
                {
                    const uint32_t n_local = lid;     // 0..63
                    if (n_local < BN) {
                        const uint32_t n_global = wg_n + n_local;
                        const block_q4_K& blk =
                            W[uint64_t(n_global) * blocks_per_col + b];
                        dequant_q4k_block_to_stripe(
                            blk, &B_smem[0][n_local], BN);
                    }
                }
                sycl::group_barrier(it.get_group());

                // K_TILES_PER_BLK × M_GROUPS_MAX mat_mads per SG.  The B-tile
                // is loaded once per kt and reused for both M-groups —
                // weight-read amortization is now 2× the M_TILE=8 baseline.
                #pragma unroll
                for (int kt = 0; kt < K_TILES_PER_BLK; ++kt) {
                    mat::joint_matrix<sycl::sub_group, fp16, mat::use::b,
                                      TK, TN, mat::layout::row_major> b_tile;
                    mat::joint_matrix_load(sg, b_tile,
                        B_smem.get_multi_ptr<sycl::access::decorated::no>() +
                        kt * TK * BN + sg_id * TN,
                        /*stride=*/BN);
                    #pragma unroll
                    for (int g = 0; g < M_GROUPS_MAX; ++g) {
                        mat::joint_matrix<sycl::sub_group, fp16, mat::use::a,
                                          TM, TK, mat::layout::row_major> a_tile;
                        mat::joint_matrix_load(sg, a_tile,
                            A_smem.get_multi_ptr<sycl::access::decorated::no>() +
                            uint64_t(g) * TM * BK + kt * TK,
                            /*stride=*/BK);
                        mat::joint_matrix_mad(sg, acc[g], a_tile, b_tile, acc[g]);
                    }
                }
                sycl::group_barrier(it.get_group());
            }

            // Store accumulators per group into C_scratch, then per-lane
            // convert+write to global y[M, N] masked to actual M.
            #pragma unroll
            for (int g = 0; g < M_GROUPS_MAX; ++g) {
                mat::joint_matrix_store(sg, acc[g],
                    C_scratch.get_multi_ptr<sycl::access::decorated::no>() +
                        uint64_t(g) * N_TILES_PER_WG * TM * TN +
                        sg_id * TM * TN,
                    /*stride=*/TN, mat::layout::row_major);
            }
            sycl::group_barrier(it.get_group());

            const uint32_t col0 = wg_n + sg_id * TN;
            const uint32_t lane_col = lane;             // 0..15 = TN lanes
            // Walk the actual M rows (≤ 16) and write the corresponding
            // (group, intra-group row) slot to global y.
            for (uint32_t r = 0; r < M; ++r) {
                const uint32_t g  = r / TM;
                if (g >= m_groups) break;
                const uint32_t rg = r - g * TM;
                const float v = C_scratch[
                    uint64_t(g)  * N_TILES_PER_WG * TM * TN +
                    sg_id * TM * TN +
                    rg * TN +
                    lane_col];
                y[uint64_t(r) * N + (col0 + lane_col)] = fp16(v);
            }
        });
    });
}

// =============================================================================
// gemm_q6_K_xmx — Q6_K version of gemm_q4_K_xmx.  Mirrors the Q4_K layout
// but the per-block dequant follows the Q6_K structure (ql[128] + qh[64]
// + int8 scales[16] + fp16 d, no dmin).
// =============================================================================
namespace {

// Dequantize one Q6_K super-block (256 elements) for ONE output column n
// into a 256-element fp16 strip starting at `dst` with given stride.
// Mirrors gemv_q6_K's lane lattice: each of 8 sub-blocks is 32 elements,
// addressed via (half ∈ {0,1}, sub ∈ {0..3}).
inline void dequant_q6k_block_to_stripe(const block_q6_K& blk,
                                        fp16* dst, int stride) {
    const float d_super = dev_fp16_to_fp32(blk.d);
    #pragma unroll
    for (int half_q = 0; half_q < 2; ++half_q) {
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            const int qh_shift = sub * 2;
            const bool high_nibble = (sub & 2) != 0;
            const int sub_idx = half_q * 4 + sub;     // 0..7
            // Two 16-element half-tiles per (half_q, sub) — l_half ∈ {0,1}.
            #pragma unroll
            for (int l_half = 0; l_half < 2; ++l_half) {
                const int l_start  = l_half * 16;
                const int ql_off   = half_q * 64 + (sub & 1) * 32 + l_start;
                const int qh_off   = half_q * 32 + l_start;
                const int scale_off = half_q * 8 + sub * 2 + l_half;
                const int out_off  = half_q * 128 + sub * 32 + l_start;
                const float scale = d_super * float(blk.scales[scale_off]);
                #pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const uint8_t ql_b = blk.ql[ql_off + i];
                    const uint8_t qh_b = blk.qh[qh_off + i];
                    const uint8_t lo_hi =
                        high_nibble ? uint8_t(ql_b >> 4) : uint8_t(ql_b & 0x0F);
                    const int8_t qsig =
                        int8_t(lo_hi | (((qh_b >> qh_shift) & 0x3) << 4)) - 32;
                    dst[(out_off + i) * stride] = fp16(scale * float(qsig));
                }
            }
            (void)sub_idx;
        }
    }
}

}  // namespace

sycl::event gemm_q6_K_xmx(sycl::queue& q,
                          const fp16* A, const void* W_packed,
                          fp16* y,
                          uint32_t M, uint32_t K, uint32_t N,
                          const std::vector<sycl::event>& deps) {
    if (M == 0 || M > uint32_t(M_TILE_MAX)) return {};
    if (K % BK != 0 || N % BN != 0) return {};

    const auto* W = static_cast<const block_q6_K*>(W_packed);
    const uint32_t blocks_per_col = K / 256;
    const uint32_t wgs_n = N / BN;
    const uint32_t m_groups = (M + TM - 1) / TM;

    return ie::ps(q, "gemm_q6k_xmx", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<fp16, 2> A_smem({M_TILE_MAX, BK}, h);
        sycl::local_accessor<fp16, 2> B_smem({BK, BN}, h);
        sycl::local_accessor<float, 1> C_scratch(
            M_GROUPS_MAX * N_TILES_PER_WG * TM * TN, h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(wgs_n) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid    = uint32_t(it.get_local_id(0));
            const uint32_t wgid   = uint32_t(it.get_group(0));
            const uint32_t sg_id  = lid / SG_SIZE;
            const uint32_t lane   = lid % SG_SIZE;
            const uint32_t wg_n   = wgid * BN;
            auto sg = it.get_sub_group();

            mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator,
                              TM, TN> acc[M_GROUPS_MAX];
            #pragma unroll
            for (int g = 0; g < M_GROUPS_MAX; ++g) {
                mat::joint_matrix_fill(sg, acc[g], 0.0f);
            }

            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                {
                    const uint32_t total = M * BK;
                    for (uint32_t i = lid; i < total; i += WG_ITEMS) {
                        const uint32_t mm = i / BK;
                        const uint32_t kk = i % BK;
                        A_smem[mm][kk] = A[uint64_t(mm) * K + b * BK + kk];
                    }
                }
                {
                    const uint32_t n_local = lid;
                    if (n_local < BN) {
                        const uint32_t n_global = wg_n + n_local;
                        const block_q6_K& blk =
                            W[uint64_t(n_global) * blocks_per_col + b];
                        dequant_q6k_block_to_stripe(blk, &B_smem[0][n_local], BN);
                    }
                }
                sycl::group_barrier(it.get_group());

                #pragma unroll
                for (int kt = 0; kt < K_TILES_PER_BLK; ++kt) {
                    mat::joint_matrix<sycl::sub_group, fp16, mat::use::b,
                                      TK, TN, mat::layout::row_major> b_tile;
                    mat::joint_matrix_load(sg, b_tile,
                        B_smem.get_multi_ptr<sycl::access::decorated::no>() +
                        kt * TK * BN + sg_id * TN,
                        /*stride=*/BN);
                    #pragma unroll
                    for (int g = 0; g < M_GROUPS_MAX; ++g) {
                        mat::joint_matrix<sycl::sub_group, fp16, mat::use::a,
                                          TM, TK, mat::layout::row_major> a_tile;
                        mat::joint_matrix_load(sg, a_tile,
                            A_smem.get_multi_ptr<sycl::access::decorated::no>() +
                            uint64_t(g) * TM * BK + kt * TK,
                            /*stride=*/BK);
                        mat::joint_matrix_mad(sg, acc[g], a_tile, b_tile, acc[g]);
                    }
                }
                sycl::group_barrier(it.get_group());
            }

            #pragma unroll
            for (int g = 0; g < M_GROUPS_MAX; ++g) {
                mat::joint_matrix_store(sg, acc[g],
                    C_scratch.get_multi_ptr<sycl::access::decorated::no>() +
                        uint64_t(g) * N_TILES_PER_WG * TM * TN +
                        sg_id * TM * TN,
                    /*stride=*/TN, mat::layout::row_major);
            }
            sycl::group_barrier(it.get_group());

            const uint32_t col0 = wg_n + sg_id * TN;
            const uint32_t lane_col = lane;
            for (uint32_t r = 0; r < M; ++r) {
                const uint32_t g  = r / TM;
                if (g >= m_groups) break;
                const uint32_t rg = r - g * TM;
                const float v = C_scratch[
                    uint64_t(g)  * N_TILES_PER_WG * TM * TN +
                    sg_id * TM * TN +
                    rg * TN +
                    lane_col];
                y[uint64_t(r) * N + (col0 + lane_col)] = fp16(v);
            }
        });
    });
}

// =============================================================================
// gemm_q4_0_xmx — Q4_0 (Gemma 4 QAT) version of gemm_q4_K_xmx.  Identical
// XMX/DPAS tiling and mat_mad structure; only the per-block dequant differs:
// Q4_0 is 32-elem blocks (one fp16 scale, symmetric w = d*(nib-8), no dmin),
// so a BK=256 K-tile is 8 consecutive Q4_0 blocks per column.
// =============================================================================
namespace {

// Dequant one 256-elem K-tile (8 Q4_0 blocks) for ONE output column into the
// column stripe `dst` (stride = BN).  `col_blocks` points at this column's
// first Q4_0 block for this K-tile (8 blocks contiguous along K).
inline void dequant_q4_0_tile_to_stripe(const block_q4_0* col_blocks,
                                        fp16* dst, int stride) {
    #pragma unroll
    for (int sb = 0; sb < 8; ++sb) {            // 8 blocks × 32 = 256 elems
        const block_q4_0& blk = col_blocks[sb];
        const float d = dev_fp16_to_fp32(blk.d);
        const int k0 = sb * 32;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint8_t qb = blk.qs[i];
            dst[uint64_t(k0 + i)      * stride] = fp16(d * float(int(qb & 0x0F) - 8));
            dst[uint64_t(k0 + i + 16) * stride] = fp16(d * float(int(qb >> 4)   - 8));
        }
    }
}

}  // namespace

sycl::event gemm_q4_0_xmx(sycl::queue& q,
                          const fp16* A, const void* W_packed,
                          fp16* y,
                          uint32_t M, uint32_t K, uint32_t N,
                          const std::vector<sycl::event>& deps) {
    if (M == 0 || M > uint32_t(M_TILE_MAX)) return {};
    if (K % BK != 0 || N % BN != 0) return {};

    const auto* W = static_cast<const block_q4_0*>(W_packed);
    const uint32_t q40_per_col = K / 32;              // Q4_0 blocks per column
    const uint32_t blocks_per_col = K / BK;           // BK=256 K-tiles per column
    const uint32_t wgs_n = N / BN;
    const uint32_t m_groups = (M + TM - 1) / TM;

    return ie::ps(q, "gemm_q4_0_xmx", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<fp16, 2> A_smem({M_TILE_MAX, BK}, h);
        sycl::local_accessor<fp16, 2> B_smem({BK, BN}, h);
        sycl::local_accessor<float, 1> C_scratch(
            M_GROUPS_MAX * N_TILES_PER_WG * TM * TN, h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(wgs_n) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid    = uint32_t(it.get_local_id(0));
            const uint32_t wgid   = uint32_t(it.get_group(0));
            const uint32_t sg_id  = lid / SG_SIZE;
            const uint32_t lane   = lid % SG_SIZE;
            const uint32_t wg_n   = wgid * BN;
            auto sg = it.get_sub_group();

            mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator,
                              TM, TN> acc[M_GROUPS_MAX];
            #pragma unroll
            for (int g = 0; g < M_GROUPS_MAX; ++g)
                mat::joint_matrix_fill(sg, acc[g], 0.0f);

            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                {
                    const uint32_t total = M * BK;
                    for (uint32_t i = lid; i < total; i += WG_ITEMS) {
                        const uint32_t m = i / BK;
                        const uint32_t kk = i % BK;
                        A_smem[m][kk] = A[uint64_t(m) * K + b * BK + kk];
                    }
                }
                {
                    const uint32_t n_local = lid;     // 0..63
                    if (n_local < BN) {
                        const uint32_t n_global = wg_n + n_local;
                        // This column's 8 Q4_0 blocks for K-tile b.
                        const block_q4_0* col_blocks =
                            W + uint64_t(n_global) * q40_per_col + uint64_t(b) * 8;
                        dequant_q4_0_tile_to_stripe(col_blocks, &B_smem[0][n_local], BN);
                    }
                }
                sycl::group_barrier(it.get_group());

                #pragma unroll
                for (int kt = 0; kt < K_TILES_PER_BLK; ++kt) {
                    mat::joint_matrix<sycl::sub_group, fp16, mat::use::b,
                                      TK, TN, mat::layout::row_major> b_tile;
                    mat::joint_matrix_load(sg, b_tile,
                        B_smem.get_multi_ptr<sycl::access::decorated::no>() +
                        kt * TK * BN + sg_id * TN,
                        /*stride=*/BN);
                    #pragma unroll
                    for (int g = 0; g < M_GROUPS_MAX; ++g) {
                        mat::joint_matrix<sycl::sub_group, fp16, mat::use::a,
                                          TM, TK, mat::layout::row_major> a_tile;
                        mat::joint_matrix_load(sg, a_tile,
                            A_smem.get_multi_ptr<sycl::access::decorated::no>() +
                            uint64_t(g) * TM * BK + kt * TK,
                            /*stride=*/BK);
                        mat::joint_matrix_mad(sg, acc[g], a_tile, b_tile, acc[g]);
                    }
                }
                sycl::group_barrier(it.get_group());
            }

            #pragma unroll
            for (int g = 0; g < M_GROUPS_MAX; ++g) {
                mat::joint_matrix_store(sg, acc[g],
                    C_scratch.get_multi_ptr<sycl::access::decorated::no>() +
                        uint64_t(g) * N_TILES_PER_WG * TM * TN +
                        sg_id * TM * TN,
                    /*stride=*/TN, mat::layout::row_major);
            }
            sycl::group_barrier(it.get_group());

            const uint32_t col0 = wg_n + sg_id * TN;
            const uint32_t lane_col = lane;
            for (uint32_t r = 0; r < M; ++r) {
                const uint32_t g  = r / TM;
                if (g >= m_groups) break;
                const uint32_t rg = r - g * TM;
                const float v = C_scratch[
                    uint64_t(g)  * N_TILES_PER_WG * TM * TN +
                    sg_id * TM * TN +
                    rg * TN +
                    lane_col];
                y[uint64_t(r) * N + (col0 + lane_col)] = fp16(v);
            }
        });
    });
}

// =============================================================================
// moe_prefill_proj_q4_0_xmx — XMX/DPAS version of moe_prefill_proj_q4_0_q8.
// y[TK,N] = x_fp16[TK,K] @ Q4_0 W[K,N] over all experts (expert_offsets). One
// WG per (expert, BN=64 n-chunk); the M-loop (tk_base) is inside, so a B-block
// dequant amortizes over up to 16 rows and is repeated only ~n_tok_e/16 ×.
// =============================================================================
sycl::event moe_prefill_proj_q4_0_xmx(sycl::queue& q,
                                      const fp16* x_fp16, const void* W_q4_0,
                                      const uint32_t* expert_offsets,
                                      fp16* out_packed,
                                      uint32_t E, uint32_t K, uint32_t N,
                                      uint64_t expert_stride_bytes,
                                      const std::vector<sycl::event>& deps) {
    if (K % BK != 0 || N % BN != 0) return {};
    constexpr int M_GROUPS   = M_GROUPS_MAX;     // 2 → M_TILE 16
    constexpr int M_TILE_MAX = TM * M_GROUPS;    // 16
    const uint32_t q40_per_col    = K / 32;
    const uint32_t blocks_per_col = K / BK;      // BK=256 K-tiles
    const uint32_t n_chunks = N / BN;
    const auto* WB = static_cast<const uint8_t*>(W_q4_0);

    return ie::ps(q, "moe_pfl_proj_q4_0_xmx", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<fp16, 2> A_smem({M_TILE_MAX, BK}, h);
        sycl::local_accessor<fp16, 2> B_smem({BK, BN}, h);
        sycl::local_accessor<float, 1> C_scratch(M_GROUPS * N_TILES_PER_WG * TM * TN, h);

        h.parallel_for(sycl::nd_range<2>({uint64_t(E) * n_chunks, WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t en = uint32_t(it.get_group(0));
            const uint32_t e  = en / n_chunks;
            const uint32_t nc = en % n_chunks;
            const uint32_t lid   = uint32_t(it.get_local_id(1));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t wg_n  = nc * BN;
            auto sg = it.get_sub_group();

            const uint32_t off_start = expert_offsets[e];
            const uint32_t n_tok     = expert_offsets[e + 1] - off_start;
            if (n_tok == 0) return;
            const auto* w_e = reinterpret_cast<const block_q4_0*>(
                WB + uint64_t(e) * expert_stride_bytes);

            for (uint32_t tk_base = 0; tk_base < n_tok; tk_base += M_TILE_MAX) {
                const uint32_t M = sycl::min(uint32_t(M_TILE_MAX), n_tok - tk_base);
                const uint64_t row0 = uint64_t(off_start) + tk_base;

                mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator,
                                  TM, TN> acc[M_GROUPS];
                #pragma unroll
                for (int g = 0; g < M_GROUPS; ++g)
                    mat::joint_matrix_fill(sg, acc[g], 0.0f);

                for (uint32_t b = 0; b < blocks_per_col; ++b) {
                    {
                        const uint32_t total = M * BK;
                        for (uint32_t i = lid; i < total; i += WG_ITEMS) {
                            const uint32_t m = i / BK;
                            const uint32_t kk = i % BK;
                            A_smem[m][kk] = x_fp16[(row0 + m) * K + b * BK + kk];
                        }
                    }
                    {
                        const uint32_t n_local = lid;     // 0..63
                        if (n_local < BN) {
                            const block_q4_0* col_blocks =
                                w_e + uint64_t(wg_n + n_local) * q40_per_col + uint64_t(b) * 8;
                            dequant_q4_0_tile_to_stripe(col_blocks, &B_smem[0][n_local], BN);
                        }
                    }
                    sycl::group_barrier(it.get_group());

                    #pragma unroll
                    for (int kt = 0; kt < K_TILES_PER_BLK; ++kt) {
                        mat::joint_matrix<sycl::sub_group, fp16, mat::use::b,
                                          TK, TN, mat::layout::row_major> b_tile;
                        mat::joint_matrix_load(sg, b_tile,
                            B_smem.get_multi_ptr<sycl::access::decorated::no>() +
                            kt * TK * BN + sg_id * TN, /*stride=*/BN);
                        #pragma unroll
                        for (int g = 0; g < M_GROUPS; ++g) {
                            mat::joint_matrix<sycl::sub_group, fp16, mat::use::a,
                                              TM, TK, mat::layout::row_major> a_tile;
                            mat::joint_matrix_load(sg, a_tile,
                                A_smem.get_multi_ptr<sycl::access::decorated::no>() +
                                uint64_t(g) * TM * BK + kt * TK, /*stride=*/BK);
                            mat::joint_matrix_mad(sg, acc[g], a_tile, b_tile, acc[g]);
                        }
                    }
                    sycl::group_barrier(it.get_group());
                }

                #pragma unroll
                for (int g = 0; g < M_GROUPS; ++g)
                    mat::joint_matrix_store(sg, acc[g],
                        C_scratch.get_multi_ptr<sycl::access::decorated::no>() +
                            uint64_t(g) * N_TILES_PER_WG * TM * TN + sg_id * TM * TN,
                        /*stride=*/TN, mat::layout::row_major);
                sycl::group_barrier(it.get_group());

                const uint32_t col0 = wg_n + sg_id * TN;
                for (uint32_t r = 0; r < M; ++r) {
                    const uint32_t g  = r / TM;
                    const uint32_t rg = r - g * TM;
                    const float v = C_scratch[
                        uint64_t(g) * N_TILES_PER_WG * TM * TN + sg_id * TM * TN +
                        rg * TN + lane];
                    out_packed[(row0 + r) * N + (col0 + lane)] = fp16(v);
                }
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

// =============================================================================
// XMX variant of moe_prefill_gate_up_silu_q4k — v2 (2026-06-09 rewrite).
// =============================================================================
//
// The 2026-05 attempt lost 10-15% to the scalar kernel.  Post-mortem with
// today's knowledge: (1) its dequant used per-byte scalar loads (pre-E2),
// (2) A rows were gathered through sorted_token_idx per K-block per M-tile
// (pre-E5), (3) M_TILE=8 re-dequanted every B block n_tok/8 times.
//
// v2: one WG per (expert, BN=64 n-chunk); 8 SGs (4 gate + 4 up, 128 lanes).
//   - A tiles load CONTIGUOUSLY from the E5 expert-sorted x_packed.
//   - B dequant per lane uses uint4 header + uint4 qs loads (E2/E4 style),
//     decode from registers, strided-but-lane-coalesced SLM stripe writes.
//   - M_TILE=16 via M_GROUPS=2 accumulators per SG: halves per-token A
//     traffic AND halves redundant B dequant vs M_TILE=8.
//   - SwiGLU fused on the way out via fp32 C scratch (writes h_packed).
// SLM: A 8 KiB + Bg/Bu 32 KiB each + C 2×4 KiB = 80 KiB → 1 WG/Xe-core.
// Numerics: B rounded to fp16 + fp32 mat_mad accumulate (same class as the
// E1 prefill GEMM path) — different summation order than the scalar kernel,
// so this is PPL-gated, not bit-compared.

sycl::event moe_prefill_gate_up_silu_q4k_xmx(sycl::queue& q,
                                             const fp16* x_packed,
                                             const void* gate_W, const void* up_W,
                                             const uint32_t* expert_offsets,
                                             fp16* h_packed,
                                             uint32_t E, uint32_t H, uint32_t E_ffn,
                                             uint64_t expert_stride_bytes,
                                             const std::vector<sycl::event>& deps) {
    if (H % 256 != 0 || E_ffn % 64 != 0) return {};
    constexpr int M_GROUPS   = 2;
    constexpr int M_TILE_MAX = TM * M_GROUPS;   // 16
    constexpr int BK_BLK = 256;           // one Q4_K block per K iteration
    constexpr int BN     = 64;            // each WG covers 64 N cols (gate AND up)
    constexpr int N_TILES_GATE = BN / TN; // 4 SGs handle gate
    constexpr int N_SGS_TOTAL  = 2 * N_TILES_GATE;            // 8 SGs
    constexpr int WG_ITEMS = N_SGS_TOTAL * SG_SIZE;           // 128 lanes
    const uint32_t blocks_per_col = H / BK_BLK;
    const uint32_t n_chunks = E_ffn / BN;                     // 12 for E_ffn=768

    return ie::ps(q, "moe_pfl_gate_xmx", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<fp16, 2> A_smem({M_TILE_MAX, BK_BLK}, h);
        sycl::local_accessor<fp16, 2> Bg_smem({BK_BLK, BN}, h);   // 32 KiB
        sycl::local_accessor<fp16, 2> Bu_smem({BK_BLK, BN}, h);   // 32 KiB
        sycl::local_accessor<float, 1> Cg_scratch(M_GROUPS * N_TILES_GATE * TM * TN, h);
        sycl::local_accessor<float, 1> Cu_scratch(M_GROUPS * N_TILES_GATE * TM * TN, h);

        h.parallel_for(sycl::nd_range<2>({uint64_t(E) * n_chunks, WG_ITEMS},
                                          {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t en = uint32_t(it.get_group(0));
            const uint32_t e  = en / n_chunks;
            const uint32_t nc = en % n_chunks;
            const uint32_t lid = uint32_t(it.get_local_id(1));
            const uint32_t lane = lid % SG_SIZE;
            const uint32_t sg_id = lid / SG_SIZE;          // 0..7
            auto sg = it.get_sub_group();

            const uint32_t off_start = expert_offsets[e];
            const uint32_t off_end   = expert_offsets[e + 1];
            const uint32_t n_tok     = off_end - off_start;
            if (n_tok == 0) return;

            const uint32_t n_base = nc * BN;
            const bool is_gate = sg_id < N_TILES_GATE;
            const uint32_t local_n_tile = is_gate ? sg_id : (sg_id - N_TILES_GATE);  // 0..3

            const auto* gate_e = reinterpret_cast<const block_q4_K*>(
                reinterpret_cast<const uint8_t*>(gate_W) + uint64_t(e) * expert_stride_bytes);
            const auto* up_e   = reinterpret_cast<const block_q4_K*>(
                reinterpret_cast<const uint8_t*>(up_W)   + uint64_t(e) * expert_stride_bytes);

            for (uint32_t tk_base = 0; tk_base < n_tok; tk_base += M_TILE_MAX) {
                const uint32_t M = sycl::min(uint32_t(M_TILE_MAX), n_tok - tk_base);

                mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator,
                                  TM, TN> acc[M_GROUPS];
                #pragma unroll
                for (int g = 0; g < M_GROUPS; ++g)
                    mat::joint_matrix_fill(sg, acc[g], 0.0f);

                for (uint32_t b = 0; b < blocks_per_col; ++b) {
                    // Cooperative A load — contiguous expert-sorted rows (E5).
                    {
                        const uint64_t src0 =
                            (uint64_t(off_start) + tk_base) * H + uint64_t(b) * BK_BLK;
                        const uint32_t total = M * BK_BLK;
                        for (uint32_t i = lid; i < total; i += WG_ITEMS) {
                            const uint32_t mm = i / BK_BLK;
                            const uint32_t kk = i % BK_BLK;
                            A_smem[mm][kk] = x_packed[src0 + uint64_t(mm) * H + kk];
                        }
                    }
                    // Cooperative B dequant: lane ↔ one column-block, uint4
                    // header + qs loads, decode from registers (E2/E4 style),
                    // stripe writes coalesce across lanes at each k.
                    {
                        const bool up_half = lid >= BN;
                        const uint32_t col = up_half ? (lid - BN) : lid;
                        const block_q4_K& blk = (up_half ? up_e : gate_e)
                            [uint64_t(n_base + col) * blocks_per_col + b];
                        fp16* dst = up_half ? &Bu_smem[0][col] : &Bg_smem[0][col];

                        const auto hdr = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(&blk);
                        uint32_t qsw[32];
                        #pragma unroll
                        for (int v = 0; v < 8; ++v) {
                            const auto qv =
                                reinterpret_cast<const sycl::vec<uint32_t, 4>*>(blk.qs)[v];
                            qsw[v * 4 + 0] = qv[0]; qsw[v * 4 + 1] = qv[1];
                            qsw[v * 4 + 2] = qv[2]; qsw[v * 4 + 3] = qv[3];
                        }
                        const float d_super = dev_fp16_to_fp32(uint16_t(hdr[0] & 0xFFFFu));
                        const float m_super = dev_fp16_to_fp32(uint16_t(hdr[0] >> 16));
                        const auto hsc = [&](int j) {
                            return uint8_t(hdr[1 + (j >> 2)] >> ((j & 3) * 8));
                        };
                        #pragma unroll
                        for (int sub = 0; sub < 8; ++sub) {
                            uint8_t s_raw, m_raw;
                            if (sub < 4) {
                                s_raw = hsc(sub)     & 0x3F;
                                m_raw = hsc(sub + 4) & 0x3F;
                            } else {
                                s_raw = (hsc(sub + 4) & 0x0F) | ((hsc(sub - 4) >> 6) << 4);
                                m_raw = (hsc(sub + 4) >>   4) | ((hsc(sub    ) >> 6) << 4);
                            }
                            const float scale = d_super * float(s_raw);
                            const float bias  = m_super * float(m_raw);
                            const int g       = sub >> 1;
                            const int hi_nib  = sub & 1;
                            const int qs_base = g * 32;
                            const int k_base  = g * 64 + hi_nib * 32;
                            #pragma unroll
                            for (int i = 0; i < 32; ++i) {
                                const uint8_t qb =
                                    uint8_t(qsw[(qs_base + i) >> 2] >> (((qs_base + i) & 3) * 8));
                                const int q4 = hi_nib ? (qb >> 4) : (qb & 0x0F);
                                dst[uint64_t(k_base + i) * BN] =
                                    fp16(scale * float(q4) - bias);
                            }
                        }
                    }
                    sycl::group_barrier(it.get_group());

                    // mat_mads: each SG → one (gate|up) TN-tile × M_GROUPS.
                    const auto& B_smem = is_gate ? Bg_smem : Bu_smem;
                    #pragma unroll
                    for (int kt = 0; kt < BK_BLK / TK; ++kt) {
                        mat::joint_matrix<sycl::sub_group, fp16, mat::use::b,
                                          TK, TN, mat::layout::row_major> b_tile;
                        mat::joint_matrix_load(sg, b_tile,
                            B_smem.get_multi_ptr<sycl::access::decorated::no>() +
                            kt * TK * BN + local_n_tile * TN,
                            /*stride=*/BN);
                        #pragma unroll
                        for (int g = 0; g < M_GROUPS; ++g) {
                            mat::joint_matrix<sycl::sub_group, fp16, mat::use::a,
                                              TM, TK, mat::layout::row_major> a_tile;
                            mat::joint_matrix_load(sg, a_tile,
                                A_smem.get_multi_ptr<sycl::access::decorated::no>() +
                                uint64_t(g) * TM * BK_BLK + kt * TK,
                                /*stride=*/BK_BLK);
                            mat::joint_matrix_mad(sg, acc[g], a_tile, b_tile, acc[g]);
                        }
                    }
                    sycl::group_barrier(it.get_group());
                }

                // Store accumulators into the matrix's scratch (per SG).
                auto& C_scratch = is_gate ? Cg_scratch : Cu_scratch;
                #pragma unroll
                for (int g = 0; g < M_GROUPS; ++g) {
                    mat::joint_matrix_store(sg, acc[g],
                        C_scratch.get_multi_ptr<sycl::access::decorated::no>() +
                            (uint64_t(g) * N_TILES_GATE + local_n_tile) * TM * TN,
                        /*stride=*/TN, mat::layout::row_major);
                }
                sycl::group_barrier(it.get_group());

                // SwiGLU + write to h_packed.  Gate SGs write their 16 cols
                // for both M-groups.
                if (is_gate) {
                    #pragma unroll
                    for (int g = 0; g < M_GROUPS; ++g) {
                        #pragma unroll
                        for (int r = 0; r < TM; ++r) {
                            const uint32_t row_local = uint32_t(g * TM + r);
                            if (row_local >= M) continue;
                            const uint32_t scratch_off =
                                (uint32_t(g) * N_TILES_GATE + local_n_tile) * TM * TN +
                                r * TN + lane;
                            const float gv = Cg_scratch[scratch_off];
                            const float uv = Cu_scratch[scratch_off];
                            const float silu_g = gv / (1.0f + sycl::native::exp(-gv));
                            const uint64_t row = uint64_t(off_start + tk_base + row_local);
                            h_packed[row * E_ffn + n_base + local_n_tile * TN + lane] =
                                fp16(silu_g * uv);
                        }
                    }
                }
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

}  // namespace ie
