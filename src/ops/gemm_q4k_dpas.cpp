// src/ops/gemm_q4k_dpas.cpp — W4A8 grouped expert MoE via int8 DPAS (2026-08-27).
//
// The s-form int-dot grouped tiles measured 66 GB/s effective weight bandwidth
// L2-resident (ie-q4k-bench) — ~7x off roofline, ALU-bound in the T-loop: per
// weight word the SIMD kernel spends T x (dp4a + ~3 fp) lane-ops. DPAS moves
// the integer dot to the XMX array: one 8x16x32 s8 mad covers a whole 32-quant
// Q4_K sub-block for 8 rows x 16 columns (int32 accumulation is EXACT, so the
// int part is bit-free); the per-sub-block fp scale chain remains scalar and
// becomes the cost floor.
//
// fp ORDER (the new canonical dpas-form, per (row, col)):
//   facc += d4(col,sb) * (d8(row,sb) * idot32) - dm4(col,sb) * s(row,sb)
// accumulated over (superblock b, sub-block sb) ascending, one fp32 chain per
// output, where s = s0+s1 (the full 32-quant q8 block sum, fp-added once at
// stage time). NOTE this differs from the s-form's per-16-half split + lane
// tree reduce: migration to decode/verify requires moving ALL users to this
// expression together (the 2026-08-27 s-form precedent) and re-gating.
//
// v0 scope: gate+up dual from one expert slot per job tile (the grouped MoE
// call shape), M <= 16 via two M-groups, one 16-column N-tile per WG.

#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include "ie/kernel_profiler.hpp"

namespace ie {

namespace mat  = sycl::ext::oneapi::experimental::matrix;
namespace imat = sycl::ext::intel::experimental::matrix;   // coord apply

namespace {

constexpr int SG_SIZE = 16;
constexpr int TM = 8;      // dpas M
constexpr int TN = 16;     // dpas N
constexpr int TK = 32;     // dpas K for int8 (systolic depth 8 x opc 4)

inline float dpas_fp16_to_fp32(uint16_t h) {
    // Native half->float (exact => bit-identical; see gemv_q5_1.cpp note).
    return float(sycl::bit_cast<sycl::half>(h));
}

}  // namespace

// Grouped dual gate+up over a tile-job table (same contract as
// gemv_q4_K_q8s_grouped_tiles): jobs[4j] = {x row base, rows(<=16), slot id,
// y row base}; weights at slot_base + slot*slot_bytes + {gate_off, up_off};
// activations x = block_q8_1s rows (qs used as dpas A, d and s0+s1 as row
// scales); yG/yU = [rows, N] f16.
sycl::event gemv_q4_K_q8d_grouped_tiles(sycl::queue& q,
                        const void* x_q8s,
                        const uint8_t* slot_base, uint64_t slot_bytes,
                        uint64_t gate_off, uint64_t up_off,
                        const int32_t* jobs,
                        sycl::half* yG, sycl::half* yU,
                        uint32_t K, uint32_t N, uint32_t J,
                        const std::vector<sycl::event>& deps) {
    const auto* X = static_cast<const block_q8_1s*>(x_q8s);
    const uint32_t sblocks = K / 256;          // Q4_K superblocks per column
    const uint32_t n_tiles = N / TN;           // 16-column tiles
    const uint32_t q8_blocks = K / 32;

    return ie::ps(q, "gemv_q4k_q8d_grp", [&](sycl::handler& h) {
        h.depends_on(deps);
        // SLM: A stage (2 M-groups x 8 rows x 32 s8), B stage (VNNI dwords,
        // 2 mats x 8 x 16), row scales (16 rows x 2 f32 per mat-independent).
        sycl::local_accessor<int8_t, 1>   sA(TM * 2 * TK, h);
        sycl::local_accessor<uint32_t, 1> sB(2 * 8 * TN, h);
        sycl::local_accessor<float, 1>    sRow(16 * 2, h);   // d8, s per row
        h.parallel_for(
            sycl::nd_range<2>({uint64_t(J), uint64_t(n_tiles) * SG_SIZE},
                              {1, SG_SIZE}),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            auto sg = it.get_sub_group();
            const uint32_t j    = uint32_t(it.get_group(0));
            const uint32_t nt   = uint32_t(it.get_group(1));
            const uint32_t lane = uint32_t(it.get_local_id(1));
            const int32_t xbase = jobs[4 * j + 0];
            const int32_t rows  = jobs[4 * j + 1];
            const int32_t slot  = jobs[4 * j + 2];
            const int32_t ybase = jobs[4 * j + 3];
            const uint32_t n0 = nt * TN;
            const uint32_t mg = (uint32_t(rows) + TM - 1) / TM;  // M-groups

            const uint8_t* wg_ = slot_base + uint64_t(slot) * slot_bytes;
            const auto* WG4 = reinterpret_cast<const block_q4_K*>(wg_ + gate_off);
            const auto* WU4 = reinterpret_cast<const block_q4_K*>(wg_ + up_off);

            // Per-lane column for scale extraction and final writes.
            const uint32_t n = n0 + lane;
            const block_q4_K* colG = &WG4[uint64_t(n) * sblocks];
            const block_q4_K* colU = &WU4[uint64_t(n) * sblocks];

            float faccG[2][TM], faccU[2][TM];
            for (int g = 0; g < 2; ++g)
                for (int r = 0; r < TM; ++r) { faccG[g][r] = 0.f; faccU[g][r] = 0.f; }
            // Element coordinate maps, captured on the first apply.
            int rowmap[TM]; bool have_map = false;
            for (int r = 0; r < TM; ++r) rowmap[r] = r;

            for (uint32_t b = 0; b < sblocks; ++b) {
                const auto hdrG = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(colG + b);
                const auto hdrU = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(colU + b);
                for (uint32_t sb = 0; sb < 8; ++sb) {
                    const uint32_t q8b = b * 8 + sb;
                    // ---- stage A (all rows' 32 q8 bytes) + row scales -----
                    for (uint32_t i = lane; i < uint32_t(rows) * 8; i += SG_SIZE) {
                        const uint32_t r2 = i / 8, w = i % 8;
                        const block_q8_1s& xb =
                            X[uint64_t(xbase + r2) * q8_blocks + q8b];
                        reinterpret_cast<uint32_t*>(&sA[0])[r2 * 8 + w] =
                            reinterpret_cast<const uint32_t*>(xb.qs)[w];
                        if (w == 0) {
                            sRow[r2 * 2 + 0] = xb.d;
                            sRow[r2 * 2 + 1] = xb.s0 + xb.s1;
                        }
                    }
                    // ---- stage B: nibbles -> VNNI dwords (gate | up) ------
                    {
                        const uint32_t g4  = sb >> 1;
                        const uint32_t sh  = (sb & 1) ? 4 : 0;
                        const uint8_t* qsG = colG[b].qs + g4 * 32;
                        const uint8_t* qsU = colU[b].qs + g4 * 32;
                        #pragma unroll
                        for (int w = 0; w < 8; ++w) {
                            const uint32_t rawG =
                                reinterpret_cast<const uint32_t*>(qsG)[w];
                            const uint32_t rawU =
                                reinterpret_cast<const uint32_t*>(qsU)[w];
                            sB[uint32_t(w) * TN + lane]           = (rawG >> sh) & 0x0F0F0F0Fu;
                            sB[8 * TN + uint32_t(w) * TN + lane]  = (rawU >> sh) & 0x0F0F0F0Fu;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // ---- per-lane scales for THIS column, sub-block sb ----
                    auto scales = [&](const sycl::vec<uint32_t, 4>& hdr,
                                      float& d4, float& dm4) {
                        auto hsc = [&](uint32_t jj) {
                            return uint8_t(hdr[1 + (jj >> 2)] >> ((jj & 3) * 8));
                        };
                        uint8_t s_raw, m_raw;
                        if (sb < 4) {
                            s_raw = hsc(sb)     & 0x3F;
                            m_raw = hsc(sb + 4) & 0x3F;
                        } else {
                            s_raw = (hsc(sb + 4) & 0x0F) | ((hsc(sb - 4) >> 6) << 4);
                            m_raw = (hsc(sb + 4) >>   4) | ((hsc(sb    ) >> 6) << 4);
                        }
                        d4  = dpas_fp16_to_fp32(uint16_t(hdr[0] & 0xFFFFu)) * float(s_raw);
                        dm4 = dpas_fp16_to_fp32(uint16_t(hdr[0] >> 16))     * float(m_raw);
                    };
                    float d4G, dm4G, d4U, dm4U;
                    scales(hdrG, d4G, dm4G);
                    scales(hdrU, d4U, dm4U);

                    // ---- dpas per M-group, both mats ----------------------
                    for (uint32_t g = 0; g < mg; ++g) {
                        mat::joint_matrix<sycl::sub_group, int8_t, mat::use::a,
                                          TM, TK, mat::layout::row_major> A;
                        mat::joint_matrix<sycl::sub_group, uint8_t, mat::use::b,
                                          TK, TN, mat::layout::ext_intel_packed> Bg, Bu;
                        mat::joint_matrix<sycl::sub_group, int32_t,
                                          mat::use::accumulator, TM, TN> C;
                        mat::joint_matrix_load(sg, A,
                            sA.get_multi_ptr<sycl::access::decorated::no>() +
                                g * TM * TK, TK);
                        auto bptr = sycl::address_space_cast<
                            sycl::access::address_space::local_space,
                            sycl::access::decorated::no>(
                            reinterpret_cast<uint8_t*>(
                                &sB.get_multi_ptr<sycl::access::decorated::no>()[0]));
                        mat::joint_matrix_load(sg, Bg, bptr, TN * 4);
                        mat::joint_matrix_load(sg, Bu, bptr + 8 * TN * 4, TN * 4);

                        mat::joint_matrix_fill(sg, C, 0);
                        mat::joint_matrix_mad(sg, C, A, Bg, C);
                        {
                            int idx = 0;
                            imat::joint_matrix_apply(sg, C,
                                [&](int32_t& v, size_t r2, size_t c2) {
                                (void)c2;
                                if (!have_map && g == 0) rowmap[idx] = int(r2);
                                faccG[g][idx] +=
                                    d4G * (sRow[(r2 + g * TM) * 2] * float(v)) -
                                    dm4G * sRow[(r2 + g * TM) * 2 + 1];
                                ++idx;
                            });
                        }
                        mat::joint_matrix_fill(sg, C, 0);
                        mat::joint_matrix_mad(sg, C, A, Bu, C);
                        {
                            int idx = 0;
                            imat::joint_matrix_apply(sg, C,
                                [&](int32_t& v, size_t r2, size_t c2) {
                                (void)c2;
                                faccU[g][idx] +=
                                    d4U * (sRow[(r2 + g * TM) * 2] * float(v)) -
                                    dm4U * sRow[(r2 + g * TM) * 2 + 1];
                                ++idx;
                            });
                        }
                    }
                    have_map = true;
                    it.barrier(sycl::access::fence_space::local_space);
                }
            }
            // ---- write y (lane owns column n; rows via rowmap) ------------
            if (n < N) {
                for (uint32_t g = 0; g < mg; ++g)
                    for (int i = 0; i < TM; ++i) {
                        const int r2 = rowmap[i] + int(g) * TM;
                        if (r2 < rows) {
                            yG[uint64_t(ybase + r2) * N + n] = sycl::half(faccG[g][i]);
                            yU[uint64_t(ybase + r2) * N + n] = sycl::half(faccU[g][i]);
                        }
                    }
            }
        });
    });
}

// Q5_1 down-proj via fp16 DPAS over the same tile-job table, EXACT-INT form
// (v2). v1 dequanted B to f16 in SLM — the f16 rounding of q*d+m cost +2.3%
// PPL in-model (48 layers compound). v2 removes the rounding entirely:
//   sum_k A_k * (q_k * d_b + m_b) = d_b * (A . q)_b + m_b * rowsum_b
// B tiles hold f16(q) — 5-bit ints, EXACT in f16 — so the dpas (A . q)
// accumulates true products in fp32; d_b/m_b apply per 32-block in fp32
// scalars with rowsum_b = sum of the row's activations over the block,
// precomputed once per launch (q51_block_rowsums). Per-output fp chain runs
// (block)-ascending — row-exact across any M like the Q4_K kernel.
// Per-(mega row, 32-block) activation sums for the exact-int down form.
sycl::event q51_block_rowsums(sycl::queue& q, const sycl::half* x,
                              uint32_t rows_total, uint32_t K, float* out,
                              const std::vector<sycl::event>& deps) {
    const uint32_t blocks = K / 32;
    return q.submit([&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(uint64_t(rows_total) * blocks),
                       [=](sycl::id<1> i) {
            const uint32_t r = uint32_t(i / blocks), b = uint32_t(i % blocks);
            const sycl::half* xr = x + uint64_t(r) * K + uint64_t(b) * 32;
            float s = 0.f;
            #pragma unroll
            for (int k = 0; k < 32; ++k) s += float(xr[k]);
            out[i] = s;
        });
    });
}

sycl::event gemv_q5_1_f16d_grouped_tiles(sycl::queue& q,
                        const sycl::half* x,       // [*, K] rows (xbase-indexed)
                        const uint8_t* slot_base, uint64_t slot_bytes,
                        uint64_t down_off,
                        const int32_t* jobs,
                        sycl::half* y,             // [*, N] rows (ybase-indexed)
                        const float* rowsums,      // [*, K/32] from q51_block_rowsums
                        uint32_t K, uint32_t N, uint32_t J,
                        const std::vector<sycl::event>& deps) {
    constexpr int TKF = 16;                        // fp16 dpas K
    const uint32_t blocks = K / 32;
    const uint32_t n_tiles = N / TN;

    return ie::ps(q, "gemv_q51_f16d_grp", [&](sycl::handler& h) {
        h.depends_on(deps);
        // A tiles load from GLOBAL (L2-hot); B stages ONE block of f16(q)
        // (exact) per barrier pair; C is drained per block via the coord
        // apply so d_b/m_b can be applied in fp32.
        sycl::local_accessor<sycl::half, 1> sBt(32 * TN, h);
        h.parallel_for(
            sycl::nd_range<2>({uint64_t(J), uint64_t(n_tiles) * SG_SIZE},
                              {1, SG_SIZE}),
            [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            auto sg = it.get_sub_group();
            const uint32_t j    = uint32_t(it.get_group(0));
            const uint32_t nt   = uint32_t(it.get_group(1));
            const uint32_t lane = uint32_t(it.get_local_id(1));
            const int32_t xbase = jobs[4 * j + 0];
            const int32_t rows  = jobs[4 * j + 1];
            const int32_t slot  = jobs[4 * j + 2];
            const int32_t ybase = jobs[4 * j + 3];
            const uint32_t n0 = nt * TN;
            const uint32_t mg = (uint32_t(rows) + TM - 1) / TM;

            const auto* W5 = reinterpret_cast<const block_q5_1*>(
                slot_base + uint64_t(slot) * slot_bytes + down_off);
            const uint32_t n = n0 + lane;
            const block_q5_1* col = &W5[uint64_t(n) * blocks];
            const sycl::half* Ag = x + uint64_t(xbase) * K;
            const float* rs = rowsums + uint64_t(xbase) * blocks;

            float facc[2][TM];
            for (int g = 0; g < 2; ++g)
                for (int r = 0; r < TM; ++r) facc[g][r] = 0.f;
            int rowmap[TM]; bool have_map = false;
            for (int r = 0; r < TM; ++r) rowmap[r] = r;

            for (uint32_t b = 0; b < blocks; ++b) {
                {   // stage f16(q) — EXACT 5-bit ints, no d/m folded
                    const block_q5_1& blk = col[b];
                    const uint32_t qh = uint32_t(blk.qh[0])
                                      | (uint32_t(blk.qh[1]) << 8)
                                      | (uint32_t(blk.qh[2]) << 16)
                                      | (uint32_t(blk.qh[3]) << 24);
                    #pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const uint8_t qb = blk.qs[i];
                        const uint32_t xh0 = ((qh >> i)        << 4) & 0x10u;
                        const uint32_t xh1 = ((qh >> (i + 12))     ) & 0x10u;
                        sBt[uint32_t(i)      * TN + lane] =
                            sycl::half(float((qb & 0x0Fu) | xh0));
                        sBt[uint32_t(i + 16) * TN + lane] =
                            sycl::half(float((qb >>  4u ) | xh1));
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);
                const float d_b = dpas_fp16_to_fp32(col[b].d);
                const float m_b = dpas_fp16_to_fp32(col[b].m);
                for (uint32_t g = 0; g < mg; ++g) {
                    mat::joint_matrix<sycl::sub_group, float,
                                      mat::use::accumulator, TM, TN> C;
                    mat::joint_matrix_fill(sg, C, 0.f);
                    #pragma unroll
                    for (int kt = 0; kt < 2; ++kt) {
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b,
                                          TKF, TN, mat::layout::row_major> B;
                        mat::joint_matrix_load(sg, B,
                            sBt.get_multi_ptr<sycl::access::decorated::no>() +
                                kt * TKF * TN, TN);
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a,
                                          TM, TKF, mat::layout::row_major> A;
                        mat::joint_matrix_load(sg, A,
                            sycl::address_space_cast<
                                sycl::access::address_space::global_space,
                                sycl::access::decorated::no>(
                                const_cast<sycl::half*>(
                                    Ag + uint64_t(g) * TM * K +
                                    b * 32 + kt * TKF)), K);
                        mat::joint_matrix_mad(sg, C, A, B, C);
                    }
                    int idx = 0;
                    imat::joint_matrix_apply(sg, C,
                        [&](float& v, size_t r2, size_t c2) {
                        (void)c2;
                        if (!have_map && g == 0) rowmap[idx] = int(r2);
                        const uint32_t gr = uint32_t(r2) + g * TM;
                        facc[g][idx] += d_b * v + m_b * rs[uint64_t(gr) * blocks + b];
                        ++idx;
                    });
                }
                have_map = true;
                it.barrier(sycl::access::fence_space::local_space);
            }
            if (n < N)
                for (uint32_t g = 0; g < mg; ++g)
                    for (int i = 0; i < TM; ++i) {
                        const int r2 = rowmap[i] + int(g) * TM;
                        if (r2 < rows)
                            y[uint64_t(ybase + r2) * N + n] =
                                sycl::half(facc[g][i]);
                    }
        });
    });
}

}  // namespace ie
