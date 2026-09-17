// src/ops/gemm_q4k_esimd.cpp — Phase 1 (v2.0): SLM-staged tile load.
//
// lsc_load_block2d.ugm (__spirv_Subgroup2DBlockLoadINTEL) and
// lsc_load.ugm (M1_NM,1) d32x8t (__spirv_SubgroupBlockReadINTEL) both cause a
// GuC CT failure on BMG-G31 C0 (IP 20.2.0).  IGC also coalesces any
// stride-1 SIMD16 global (a64) load — including global_ptr[LocalInvocationId.x]
// — into the same broken scalar transposed form.
//
// Fix: SLM staging.  Lane 0 copies global→SLM with individual (non-unrolled)
// d16 point loads; all lanes then read from SLM via (M16,16) lsc_load.slm.d16,
// which is a completely separate instruction family unaffected by the UGM bug.
// The inner copy loop must NOT be #pragma unroll'd — unrolling lets IGC see 16
// consecutive loads and re-coalesce them into a block read.

#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <cstdint>
#include "ie/kernel_profiler.hpp"

namespace ie {

namespace mat = sycl::ext::oneapi::experimental::matrix;
using fp16 = sycl::half;

// =====================================================================
// Smoke: 8×32-u8 (= 256 byte) tile cooperative load + cross-lane checksum.
// =====================================================================
sycl::event esimd_block2d_smoke_tile256(sycl::queue& q,
                                        const uint8_t* src,
                                        uint32_t* out_sum,
                                        uint32_t width_bytes,
                                        uint32_t height_rows,
                                        uint32_t pitch_bytes,
                                        const std::vector<sycl::event>& deps) {
    return ie::ps(q, "esimd_smoke", [&](sycl::handler& h) {
        h.depends_on(deps);
        // SLM: 8 rows × 16 uint16 lanes = 256 bytes.
        sycl::local_accessor<uint16_t, 1> slm(8 * 16, h);
        h.parallel_for<class EsimdSmokeTile256Kernel>(
            sycl::nd_range<1>(16, 16),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
#ifdef __SYCL_DEVICE_ONLY__
                (void)width_bytes; (void)height_rows;
                const uint32_t lid = it.get_local_id(0);

                // Stage 1: lane 0 copies global→SLM via scalar point loads.
                // No #pragma unroll — IGC must not coalesce into a block read.
                if (lid == 0) {
                    for (uint32_t r = 0; r < 8; r++) {
                        const auto* row =
                            reinterpret_cast<const uint16_t*>(src + r * pitch_bytes);
                        for (uint32_t c = 0; c < 16; c++)
                            slm[r * 16 + c] = row[c];
                    }
                }

                sycl::group_barrier(it.get_group());

                // Stage 2: per-lane SLM reads → (M16,16) lsc_load.slm.d16
                uint16_t dst[8] = {};
                #pragma unroll
                for (int r = 0; r < 8; r++)
                    dst[r] = slm[r * 16 + lid];

                uint32_t lane_sum = 0;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    lane_sum += uint32_t(dst[i] & 0xFFu);
                    lane_sum += uint32_t((dst[i] >> 8) & 0xFFu);
                }
                const uint32_t total =
                    sycl::reduce_over_group(it.get_sub_group(), lane_sum,
                                            sycl::plus<uint32_t>());
                if (it.get_local_id(0) == 0) *out_sum = total;
#else
                (void)it;
                (void)src; (void)out_sum; (void)slm;
                (void)width_bytes; (void)height_rows; (void)pitch_bytes;
#endif
            });
    });
}

// =====================================================================
// gemm_q4_K_esimd — Q4_K weight GEMM, drop-in replacement for
// gemm_q4_K_xmx with cooperative SG block-read weight + activation
// loads.  Same input/output contract; same SLM layout; same
// joint_matrix mat_mad compute path.
// =====================================================================
//
// Differences vs gemm_q4_K_xmx:
//   1) A loads use sub_group::load<8> cooperative SG block reads
//      (verified working on BMG-G31 C0 via Step 0 smoke test
//      scripts/esimd_smoke/sgbr_smoke_multi.cpp).  Each call loads
//      16 lanes × 8 halfs = 128 contiguous halfs as one LSC.UGM
//      transaction.  Replaces the 64-lane scalar half-load loop —
//      forces the wide-load codegen IGC's auto-coalescer doesn't
//      always emit on the original.
//   2) Q4_K dequant reads blk.qs[] as sycl::vec<uint8_t,16>
//      (same trick that won on gemv_q4_K), replacing the byte-by-byte
//      #pragma unroll inner loop.  Two vec<16> loads cover the 32-byte
//      qs[qs_base..qs_base+31] needed per sub-block pair.
//   3) Compute path unchanged — joint_matrix mat_mad as the existing
//      gemm_q4_K_xmx.  This is intentional: bound the change to memory
//      patterns; ship-or-revert clears the gate based on memory
//      improvements alone.  Full ESIMD xmx::dpas is Step 2+ scope if
//      pp512 doesn't clear the intermediate gate.

namespace {

constexpr int ESIMD_SG_SIZE = 16;
constexpr int ESIMD_BN      = 64;     // N tile per WG
constexpr int ESIMD_BK      = 256;    // K tile = 1 Q4_K block
constexpr int ESIMD_TM      = 8;      // XMX M
constexpr int ESIMD_TN      = 16;     // XMX N
constexpr int ESIMD_TK      = 16;     // XMX K
constexpr int ESIMD_N_TILES_PER_WG  = ESIMD_BN / ESIMD_TN;   // 4
constexpr int ESIMD_K_TILES_PER_BLK = ESIMD_BK / ESIMD_TK;   // 16
constexpr int ESIMD_WG_ITEMS = ESIMD_N_TILES_PER_WG * ESIMD_SG_SIZE;  // 64

inline float esimd_fp16_to_fp32(uint16_t h) {
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

// Vector-load Q4_K sub-block decode.  Uses sycl::vec<uint8_t,16> to
// force IGC to emit the wide block-byte load that the byte-by-byte
// pattern in gemm_q4_K_xmx may or may not get.  Mirrors the algebra of
// dequant_q4k_block_to_stripe but with explicit vec ops.
inline void esimd_dequant_q4k_block(const block_q4_K& blk,
                                    fp16* dst, int stride) {
    const float d_super = esimd_fp16_to_fp32(blk.d);
    const float m_super = esimd_fp16_to_fp32(blk.dmin);

    #pragma unroll
    for (int sub = 0; sub < 8; ++sub) {
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

        const int g       = sub >> 1;
        const int hi_nib  = sub & 1;
        const int qs_base = g * 32;
        const int dst_base = (g * 64) + (hi_nib * 32);

        // Wide load: 32 packed bytes covering this sub-block pair (lo+hi).
        // The OUTER `sub` loop hits each byte twice (once for hi_nib=0,
        // once for hi_nib=1) — we explicitly reload to keep the inner
        // pattern uniform; IGC will CSE the duplicate load if profitable.
        const sycl::vec<uint8_t, 16> qs_lo16 =
            *reinterpret_cast<const sycl::vec<uint8_t, 16>*>(&blk.qs[qs_base]);
        const sycl::vec<uint8_t, 16> qs_hi16 =
            *reinterpret_cast<const sycl::vec<uint8_t, 16>*>(&blk.qs[qs_base + 16]);

        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint8_t qb = qs_lo16[i];
            const int q4 = hi_nib ? (qb >> 4) : (qb & 0x0F);
            dst[(dst_base + i) * stride] = fp16(scale * float(q4) - bias);
        }
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const uint8_t qb = qs_hi16[i];
            const int q4 = hi_nib ? (qb >> 4) : (qb & 0x0F);
            dst[(dst_base + 16 + i) * stride] = fp16(scale * float(q4) - bias);
        }
    }
}

}  // namespace

sycl::event gemm_q4_K_esimd(sycl::queue& q,
                             const fp16* A, const void* W_packed,
                             fp16* y,
                             uint32_t M, uint32_t K, uint32_t N,
                             const std::vector<sycl::event>& deps) {
    constexpr uint32_t M_TILE_MAX = 8;
    if (M == 0 || M > M_TILE_MAX) return {};
    if (K % ESIMD_BK != 0 || N % ESIMD_BN != 0) return {};

    const auto* W = static_cast<const block_q4_K*>(W_packed);
    const uint32_t blocks_per_col = K / 256;
    const uint32_t wgs_n = N / ESIMD_BN;

    return ie::ps(q, "gemm_q4k_esimd", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<fp16, 2> A_smem({M_TILE_MAX, ESIMD_BK}, h);
        sycl::local_accessor<fp16, 2> B_smem({ESIMD_BK, ESIMD_BN}, h);
        sycl::local_accessor<float, 1> C_scratch(
            ESIMD_N_TILES_PER_WG * ESIMD_TM * ESIMD_TN, h);

        h.parallel_for(sycl::nd_range<1>(uint64_t(wgs_n) * ESIMD_WG_ITEMS,
                                          ESIMD_WG_ITEMS),
                       [=](sycl::nd_item<1> it)
                       [[sycl::reqd_sub_group_size(ESIMD_SG_SIZE)]] {
            const uint32_t lid    = uint32_t(it.get_local_id(0));
            const uint32_t wgid   = uint32_t(it.get_group(0));
            const uint32_t sg_id  = lid / ESIMD_SG_SIZE;        // 0..3
            const uint32_t lane   = lid % ESIMD_SG_SIZE;
            const uint32_t wg_n   = wgid * ESIMD_BN;
            auto sg = it.get_sub_group();

            mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator,
                              ESIMD_TM, ESIMD_TN> acc;
            mat::joint_matrix_fill(sg, acc, 0.0f);

            for (uint32_t b = 0; b < blocks_per_col; ++b) {
                // ---- Cooperative A load via SG block reads ----
                // Layout: A_smem[M][BK].  Total M*BK halfs.  Each SG owns
                // a 512-half chunk = 2 rows of A_smem (since BK=256, two
                // rows = 512 halfs per SG).  Per row, 256 halfs = 2
                // sg.load<8> calls (each loads 128 halfs cooperatively).
                //
                // Lane semantics: sg.load<8>(ptr) returns vec<half,8>;
                // lane lid receives v[i] = ptr[lid + i*16] for i=0..7.
                // The loaded 128 halfs cover ptr[0..127] contiguously.
                {
                    constexpr int ROWS_PER_SG = M_TILE_MAX / ESIMD_N_TILES_PER_WG;  // 2
                    constexpr int CALLS_PER_ROW = ESIMD_BK / 128;                  // 2
                    #pragma unroll
                    for (int rr = 0; rr < ROWS_PER_SG; ++rr) {
                        const uint32_t m_row = sg_id * ROWS_PER_SG + rr;
                        if (m_row >= M) continue;  // mask out-of-bounds rows
                        #pragma unroll
                        for (int call = 0; call < CALLS_PER_ROW; ++call) {
                            const uint32_t k_base = call * 128;
                            auto a_ptr = sycl::address_space_cast<
                                            sycl::access::address_space::global_space,
                                            sycl::access::decorated::yes>(
                                const_cast<fp16*>(A + uint64_t(m_row) * K +
                                                    b * ESIMD_BK + k_base));
                            sycl::vec<fp16, 8> v = sg.load<8>(a_ptr);
                            // Lane lid contributes v[i] at position
                            // (k_base + lid + i*16) within row m_row.
                            #pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                A_smem[m_row][k_base + lane + i * ESIMD_SG_SIZE] = v[i];
                            }
                        }
                    }
                }

                // ---- Cooperative B dequant (per-lane, 1 column per lane) ----
                // Same lane-to-column mapping as gemm_q4_K_xmx, but the
                // qs[] reads inside esimd_dequant_q4k_block use explicit
                // sycl::vec<uint8_t,16> loads.
                {
                    const uint32_t n_local = lid;     // 0..63
                    if (n_local < ESIMD_BN) {
                        const uint32_t n_global = wg_n + n_local;
                        const block_q4_K& blk =
                            W[uint64_t(n_global) * blocks_per_col + b];
                        esimd_dequant_q4k_block(blk, &B_smem[0][n_local], ESIMD_BN);
                    }
                }
                sycl::group_barrier(it.get_group());

                // ---- Mat_mad (joint_matrix) — same as gemm_q4_K_xmx ----
                #pragma unroll
                for (int kt = 0; kt < ESIMD_K_TILES_PER_BLK; ++kt) {
                    mat::joint_matrix<sycl::sub_group, fp16, mat::use::a,
                                      ESIMD_TM, ESIMD_TK,
                                      mat::layout::row_major> a_tile;
                    mat::joint_matrix_load(sg, a_tile,
                        A_smem.get_multi_ptr<sycl::access::decorated::no>() +
                        kt * ESIMD_TK,
                        /*stride=*/ESIMD_BK);

                    mat::joint_matrix<sycl::sub_group, fp16, mat::use::b,
                                      ESIMD_TK, ESIMD_TN,
                                      mat::layout::row_major> b_tile;
                    mat::joint_matrix_load(sg, b_tile,
                        B_smem.get_multi_ptr<sycl::access::decorated::no>() +
                        kt * ESIMD_TK * ESIMD_BN + sg_id * ESIMD_TN,
                        /*stride=*/ESIMD_BN);

                    mat::joint_matrix_mad(sg, acc, a_tile, b_tile, acc);
                }
                sycl::group_barrier(it.get_group());
            }

            // ---- Store accumulator → fp32 scratch → fp16 global write ----
            mat::joint_matrix_store(sg, acc,
                C_scratch.get_multi_ptr<sycl::access::decorated::no>() +
                    sg_id * ESIMD_TM * ESIMD_TN,
                /*stride=*/ESIMD_TN, mat::layout::row_major);
            sycl::group_barrier(it.get_group());

            const uint32_t col0 = wg_n + sg_id * ESIMD_TN;
            const uint32_t lane_col = lane;
            #pragma unroll
            for (int r = 0; r < ESIMD_TM; ++r) {
                if (uint32_t(r) >= M) continue;
                const float v =
                    C_scratch[sg_id * ESIMD_TM * ESIMD_TN + r * ESIMD_TN + lane_col];
                y[uint64_t(r) * N + (col0 + lane_col)] = fp16(v);
            }
        });
    });
}

}  // namespace ie
