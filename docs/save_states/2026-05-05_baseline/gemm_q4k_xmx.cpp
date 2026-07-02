// src/ops/gemm_q4k_xmx.cpp — Q4_K weight GEMM via SYCL joint_matrix (XMX/DPAS).
//
// Same input/output contract as gemm_q4_K (the SLM-tiled scalar version):
//   y[M, N] = A[M, K] @ W[K, N]   with W = Q4_K-packed along K, super-blocks of 256.
//
// Layout
//   WG tile    BM × BN = M × 64        (M comes from caller; BN fixed at 64)
//   K tile     BK = 256 (one Q4_K super-block)
//   SGs/WG    4 (N axis only — BM is whole-WG, no M decomposition)
//   Per-SG    one TN=16 N-tile, full TM=8 M-tile (dynamic M ≤ 8)
//   XMX inner TM=8, TN=16, TK=16 → 16 mat_mads per SG per K-block
//
// SLM
//   A_smem[BM × BK] = 8 × 256 fp16  =  4 KiB
//   B_smem[BK × BN] = 256 × 64 fp16 = 32 KiB
//   Total per WG ≈ 36 KiB.  ≤ 192 KiB Xe-core SLM with comfortable occupancy.
//
// The Q4_K → fp16 dequant runs cooperatively once per K-block, then is reused
// across the 16 inner K-tile mat_mads — the whole point of bundling BK=256.

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
constexpr int TM       = 8;           // XMX M
constexpr int TN       = 16;          // XMX N
constexpr int TK       = 16;          // XMX K
constexpr int N_TILES_PER_WG = BN / TN;     // = 4
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
    constexpr uint32_t M_TILE_MAX = 8;
    if (M == 0 || M > M_TILE_MAX) return {};
    if (K % BK != 0 || N % BN != 0) return {};

    const auto* W = static_cast<const block_q4_K*>(W_packed);
    const uint32_t blocks_per_col = K / 256;          // = K / BK
    const uint32_t wgs_n = N / BN;

    return ie::ps(q, "gemm_q4k_xmx", [&](sycl::handler& h) {
        h.depends_on(deps);
        // A_smem[BM=8 × BK=256], B_smem[BK=256 × BN=64].
        sycl::local_accessor<fp16, 2> A_smem({M_TILE_MAX, BK}, h);
        sycl::local_accessor<fp16, 2> B_smem({BK, BN}, h);
        // Output fp32 scratch — properly typed local memory so joint_matrix_store
        // gets a real local-space float pointer (reinterpret-cast over a half
        // local_accessor doesn't preserve address space).
        sycl::local_accessor<float, 1> C_scratch(N_TILES_PER_WG * TM * TN, h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(wgs_n) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid    = uint32_t(it.get_local_id(0));
            const uint32_t wgid   = uint32_t(it.get_group(0));
            const uint32_t sg_id  = lid / SG_SIZE;        // 0..3 — which N-tile
            const uint32_t lane   = lid % SG_SIZE;
            const uint32_t wg_n   = wgid * BN;
            auto sg = it.get_sub_group();

            // Accumulator for this SG's TM × TN = 8 × 16 output sub-tile.
            mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator,
                              TM, TN> acc;
            mat::joint_matrix_fill(sg, acc, 0.0f);

            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                // Cooperative A load: [M, BK=256] fp16 = M*256 elements.
                // 64 lanes split. Each lane loads M*256/64 = M*4 elements.
                {
                    const uint32_t total = M * BK;
                    for (uint32_t i = lid; i < total; i += WG_ITEMS) {
                        const uint32_t m = i / BK;
                        const uint32_t kk = i % BK;
                        A_smem[m][kk] = A[uint64_t(m) * K + b * BK + kk];
                    }
                }
                // Cooperative B dequant: 64 lanes, BN=64 columns.
                // Each lane handles ONE output column of one Q4_K block.
                {
                    const uint32_t n_local = lid;     // 0..63
                    if (n_local < BN) {
                        const uint32_t n_global = wg_n + n_local;
                        const block_q4_K& blk =
                            W[uint64_t(n_global) * blocks_per_col + b];
                        // dst points at B_smem[0][n_local]; stride = BN columns.
                        dequant_q4k_block_to_stripe(
                            blk, &B_smem[0][n_local], BN);
                    }
                }
                sycl::group_barrier(it.get_group());

                // 16 mat_mads per SG (one per inner K-tile within the BK=256
                // block).  TM=8 always, regardless of actual M ≤ 8 — extra
                // rows in A_smem are zero-padded by the loader (out-of-bounds
                // M rows aren't written; reads return whatever was last in
                // SLM, which is fine for correctness once we mask the output
                // store at the end).
                #pragma unroll
                for (int kt = 0; kt < K_TILES_PER_BLK; ++kt) {
                    mat::joint_matrix<sycl::sub_group, fp16, mat::use::a,
                                      TM, TK, mat::layout::row_major> a_tile;
                    mat::joint_matrix_load(sg, a_tile,
                        A_smem.get_multi_ptr<sycl::access::decorated::no>() +
                        kt * TK,
                        /*stride=*/BK);

                    mat::joint_matrix<sycl::sub_group, fp16, mat::use::b,
                                      TK, TN, mat::layout::row_major> b_tile;
                    mat::joint_matrix_load(sg, b_tile,
                        B_smem.get_multi_ptr<sycl::access::decorated::no>() +
                        kt * TK * BN + sg_id * TN,
                        /*stride=*/BN);

                    mat::joint_matrix_mad(sg, acc, a_tile, b_tile, acc);
                }
                sycl::group_barrier(it.get_group());
            }

            // Store FP32 accumulator → local C_scratch, then per-lane
            // convert+write to global y[M, N] in fp16, masked to actual M.
            mat::joint_matrix_store(sg, acc,
                C_scratch.get_multi_ptr<sycl::access::decorated::no>() +
                    sg_id * TM * TN,
                /*stride=*/TN, mat::layout::row_major);
            sycl::group_barrier(it.get_group());

            const uint32_t col0 = wg_n + sg_id * TN;
            const uint32_t lane_col = lane;             // 0..15 = TN lanes
            #pragma unroll
            for (int r = 0; r < TM; ++r) {
                if (uint32_t(r) >= M) continue;
                const float v = C_scratch[sg_id * TM * TN + r * TN + lane_col];
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
    constexpr uint32_t M_TILE_MAX = 8;
    if (M == 0 || M > M_TILE_MAX) return {};
    if (K % BK != 0 || N % BN != 0) return {};

    const auto* W = static_cast<const block_q6_K*>(W_packed);
    const uint32_t blocks_per_col = K / 256;
    const uint32_t wgs_n = N / BN;

    return ie::ps(q, "gemm_q6k_xmx", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<fp16, 2> A_smem({M_TILE_MAX, BK}, h);
        sycl::local_accessor<fp16, 2> B_smem({BK, BN}, h);
        sycl::local_accessor<float, 1> C_scratch(N_TILES_PER_WG * TM * TN, h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(wgs_n) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid    = uint32_t(it.get_local_id(0));
            const uint32_t wgid   = uint32_t(it.get_group(0));
            const uint32_t sg_id  = lid / SG_SIZE;
            const uint32_t lane   = lid % SG_SIZE;
            const uint32_t wg_n   = wgid * BN;
            auto sg = it.get_sub_group();

            mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator,
                              TM, TN> acc;
            mat::joint_matrix_fill(sg, acc, 0.0f);

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
                    mat::joint_matrix<sycl::sub_group, fp16, mat::use::a,
                                      TM, TK, mat::layout::row_major> a_tile;
                    mat::joint_matrix_load(sg, a_tile,
                        A_smem.get_multi_ptr<sycl::access::decorated::no>() + kt * TK,
                        /*stride=*/BK);
                    mat::joint_matrix<sycl::sub_group, fp16, mat::use::b,
                                      TK, TN, mat::layout::row_major> b_tile;
                    mat::joint_matrix_load(sg, b_tile,
                        B_smem.get_multi_ptr<sycl::access::decorated::no>() +
                        kt * TK * BN + sg_id * TN,
                        /*stride=*/BN);
                    mat::joint_matrix_mad(sg, acc, a_tile, b_tile, acc);
                }
                sycl::group_barrier(it.get_group());
            }

            mat::joint_matrix_store(sg, acc,
                C_scratch.get_multi_ptr<sycl::access::decorated::no>() +
                    sg_id * TM * TN,
                /*stride=*/TN, mat::layout::row_major);
            sycl::group_barrier(it.get_group());

            const uint32_t col0 = wg_n + sg_id * TN;
            const uint32_t lane_col = lane;
            #pragma unroll
            for (int r = 0; r < TM; ++r) {
                if (uint32_t(r) >= M) continue;
                const float v = C_scratch[sg_id * TM * TN + r * TN + lane_col];
                y[uint64_t(r) * N + (col0 + lane_col)] = fp16(v);
            }
        });
    });
}

// =============================================================================
// XMX variant of moe_prefill_gate_up_silu_q4k.
// =============================================================================
//
// One WG per (expert, n_chunk).  Per WG processes the expert's n_tok rows in
// M_TILE=8 chunks, gathering the corresponding x rows via sorted_token_idx
// and computing gate (Q4_K) + up (Q4_K) + swiglu via joint_matrix mat_mad.
// BN = N_PER_WG = 16 (one TN-tile per WG, since N_chunks = E_ffn/16 = 32).
// Output: h_packed[(expert_offsets[e] + i) * E_ffn + n].

sycl::event moe_prefill_gate_up_silu_q4k_xmx(sycl::queue& q,
                                             const fp16* x,
                                             const void* gate_W, const void* up_W,
                                             const uint32_t* expert_offsets,
                                             const int32_t* sorted_token_idx,
                                             fp16* h_packed,
                                             uint32_t E, uint32_t H, uint32_t E_ffn,
                                             uint64_t expert_stride_bytes,
                                             const std::vector<sycl::event>& deps) {
    if (H % 256 != 0) return {};
    constexpr int M_TILE_MAX = 8;
    constexpr int BK_BLK = 256;           // one Q4_K block per K iteration
    constexpr int BN     = 64;            // each WG covers 64 N cols (gate AND up)
    constexpr int N_TILES_GATE = BN / TN; // 4 SGs handle gate
    constexpr int N_TILES_UP   = BN / TN; // 4 SGs handle up
    constexpr int N_SGS_TOTAL  = N_TILES_GATE + N_TILES_UP;   // 8 SGs
    constexpr int WG_ITEMS = N_SGS_TOTAL * SG_SIZE;           // 128 lanes
    const uint32_t blocks_per_col = H / BK_BLK;
    const uint32_t n_chunks = (E_ffn + BN - 1) / BN;          // = 8 for E_ffn=512

    return ie::ps(q, "moe_pfl_gate_xmx", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<fp16, 2> A_smem({M_TILE_MAX, BK_BLK}, h);
        sycl::local_accessor<fp16, 2> Bg_smem({BK_BLK, BN}, h);   // 32 KiB
        sycl::local_accessor<fp16, 2> Bu_smem({BK_BLK, BN}, h);   // 32 KiB
        sycl::local_accessor<float, 1> Cg_scratch(N_TILES_GATE * TM * TN, h);
        sycl::local_accessor<float, 1> Cu_scratch(N_TILES_UP   * TM * TN, h);

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
                                  TM, TN> acc;
                mat::joint_matrix_fill(sg, acc, 0.0f);

                for (uint32_t b = 0; b < blocks_per_col; ++b) {
                    // Cooperative A gather (M × BK halfs).
                    {
                        const uint32_t total = M * BK_BLK;
                        for (uint32_t i = lid; i < total; i += WG_ITEMS) {
                            const uint32_t mm = i / BK_BLK;
                            const uint32_t kk = i % BK_BLK;
                            const int32_t tok = sorted_token_idx[off_start + tk_base + mm];
                            A_smem[mm][kk] = x[uint64_t(tok) * H + b * BK_BLK + kk];
                        }
                    }
                    // Cooperative B dequant: 128 lanes split BN=64 cols of
                    // gate and BN=64 cols of up.  First 64 lanes handle gate,
                    // next 64 lanes handle up.
                    if (lid < BN) {
                        const uint32_t n = n_base + lid;
                        const block_q4_K& blk =
                            gate_e[uint64_t(n) * blocks_per_col + b];
                        dequant_q4k_block_to_stripe(blk, &Bg_smem[0][lid], BN);
                    } else if (lid < 2 * BN) {
                        const uint32_t n = n_base + (lid - BN);
                        const block_q4_K& blk =
                            up_e[uint64_t(n) * blocks_per_col + b];
                        dequant_q4k_block_to_stripe(blk, &Bu_smem[0][lid - BN], BN);
                    }
                    sycl::group_barrier(it.get_group());

                    // mat_mads: each SG handles one (gate or up) TN-tile of N.
                    const auto& B_smem = is_gate ? Bg_smem : Bu_smem;
                    #pragma unroll
                    for (int kt = 0; kt < BK_BLK / TK; ++kt) {
                        mat::joint_matrix<sycl::sub_group, fp16, mat::use::a,
                                          TM, TK, mat::layout::row_major> a_tile;
                        mat::joint_matrix_load(sg, a_tile,
                            A_smem.get_multi_ptr<sycl::access::decorated::no>() +
                            kt * TK,
                            /*stride=*/BK_BLK);
                        mat::joint_matrix<sycl::sub_group, fp16, mat::use::b,
                                          TK, TN, mat::layout::row_major> b_tile;
                        mat::joint_matrix_load(sg, b_tile,
                            B_smem.get_multi_ptr<sycl::access::decorated::no>() +
                            kt * TK * BN + local_n_tile * TN,
                            /*stride=*/BN);
                        mat::joint_matrix_mad(sg, acc, a_tile, b_tile, acc);
                    }
                    sycl::group_barrier(it.get_group());
                }

                // Store accumulator into the right scratch (per SG).
                if (is_gate) {
                    mat::joint_matrix_store(sg, acc,
                        Cg_scratch.get_multi_ptr<sycl::access::decorated::no>() +
                            local_n_tile * TM * TN,
                        /*stride=*/TN, mat::layout::row_major);
                } else {
                    mat::joint_matrix_store(sg, acc,
                        Cu_scratch.get_multi_ptr<sycl::access::decorated::no>() +
                            local_n_tile * TM * TN,
                        /*stride=*/TN, mat::layout::row_major);
                }
                sycl::group_barrier(it.get_group());

                // SwiGLU + write to h_packed.  Each gate SG writes its 16 cols.
                if (is_gate) {
                    #pragma unroll
                    for (int r = 0; r < TM; ++r) {
                        if (uint32_t(r) >= M) continue;
                        const uint32_t scratch_off = local_n_tile * TM * TN + r * TN + lane;
                        const float gv = Cg_scratch[scratch_off];
                        const float uv = Cu_scratch[scratch_off];
                        const float silu_g = gv / (1.0f + sycl::native::exp(-gv));
                        const uint64_t row = uint64_t(off_start + tk_base + r);
                        h_packed[row * E_ffn + n_base + local_n_tile * TN + lane] =
                            fp16(silu_g * uv);
                    }
                }
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

}  // namespace ie
