// src/ops/qwen4_bank_dpas.cpp -- int8-DPAS grouped tiles (the prefill expert-major MoE) over qwen4exp Q8_0 / Q5_K / Q6_K
// gate/up banks (2026-11-07, Qwen 4 readiness item 6c). Why: the Q4_K path owes its ~500 pp tok/s to gemm_q4k_dpas.cpp's XMX
// tiles; the K-quant / Q8_0 banks ran SIMD tiles (A16, then the s-form int-dot ones) at 79-124 pp tok/s on the Q8_0 / Q6_K /
// Q5_K builds -- neither bandwidth- nor FLOP-bound, instruction-bound in the T-loop.
//
// gemm_q4k_dpas.cpp's shape: one subgroup per (tile job of <= 16 rows, 16-column N-tile); per 32-wide K span the rows' 32 q8
// bytes (block_q8_1s: the Q4_K path's activations) are staged as the dpas A operand and the 16 columns' 32 weight values as
// VNNI dwords (B), one 8x16x32 dpas per 8-row M-group and mat (the integer part exact), then the per-(row, column) fp32 scale
// chain in K-span-ascending order -- the dpas-form expression, NOT the s-form kernels' order (prefill numerics, gated by
// chunked PPL like the Q4_K path). The A / B stages are double-buffered by span parity, so one sub-group-wide barrier a span
// orders the stores before the loads (span s + 2 overwrites a buffer only after every lane passed span s + 1's barrier, i.e.
// after it finished reading span s).
//   Q8_0  B = the block's 32 int8 (s8 x s8);                            facc += d_w * (d8 * idot)
//   Q5_K  B = nibble | qh bit << 4 (u8, 0..31) per 32-sub-block;        facc += (d*sc) * (d8 * idot) - (dmin*m) * (s0 + s1)
//   Q6_K  B = nibble | 2 qh bits << 4 (u8, 0..63); its int8 scales are per 16 elements, so each 32-span runs TWO dpas with
//         the other half of B zeroed:  facc += (d*sc_lo) * (d8*idot_lo - 32*s0) + (d*sc_hi) * (d8*idot_hi - 32*s1)
// block_q8_0 (34 B) and block_q6_K (210 B) are 2-byte aligned: their words are built from halfwords; block_q5_K (176 B) has
// 16-aligned qs / qh: dword loads. N must be a multiple of 16; K a multiple of 32 (Q8_0) / 256 (K-quants).
// VERDICT of the first form (matrix ops inside nested mat / half loops, two barriers a span; ie-qwen4-bank-test IE_BANK_BENCH=1,
// 64 jobs x 16 rows, N 640, 64-expert bank, 2026-11-07): Q5_K A16 5.68 ms, s-form int-dot 2.38, DPAS 3.09; Q6_K 6.55 / 2.47 /
// 5.79; Q8_0 7.09 / - / 2.37; in-model on the UD-Q5_K_XL PPL 5.5195 vs the s-form's 5.5162 (the dpas-form numerics hold), pp
// 111 vs 131 tok/s. The s-form int-dot tiles (qwen4_bank.cpp) stay the default on the K-quants (second form in-model: pp 137
// vs 148); IE_Q4E_KQ_DPAS=1 takes this kernel for them (the A/B instrument). For Q8_0 banks it is the DEFAULT since the
// Unsloth Q8_0 build's gate (second form: PPL 5.4590 vs the A16 tiles' 5.4731, two-card pp 87.4 -> 100.3 tok/s;
// IE_Q4E_KQ_DPAS_Q8=0 = the A16 tiles). This file is the second form: straight-line matrix objects and one barrier a span.
#include "ie/qwen4_bank.hpp"
#include "ie/quant_blocks.hpp"
#include "ie/kernel_profiler.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <type_traits>

namespace ie {

namespace mat  = sycl::ext::oneapi::experimental::matrix;
namespace imat = sycl::ext::intel::experimental::matrix;

namespace {

constexpr int SG_SIZE = 16;
constexpr int TM = 8;      // dpas M
constexpr int TN = 16;     // dpas N
constexpr int TK = 32;     // dpas K for int8

inline float h2f(uint16_t h) { return float(sycl::bit_cast<sycl::half>(h)); }
inline uint32_t u16pair(const uint16_t* p) { return uint32_t(p[0]) | (uint32_t(p[1]) << 16); }
inline void scale_min_k4(int j, const uint8_t* q, float& sc, float& m) {
    uint32_t s, mm;
    if (j < 4) { s = q[j] & 0x3Fu;                                   mm = q[j + 4] & 0x3Fu; }
    else       { s = (q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4);   mm = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
    sc = float(s); m = float(mm);
}

template <BankQ Q> struct DpasBlk;
template <> struct DpasBlk<BankQ::kQ8_0> { using T = block_q8_0; using B = int8_t;  static constexpr uint32_t kElems = 32;  static constexpr uint32_t kSpans = 1; static constexpr int kHalves = 1; };
template <> struct DpasBlk<BankQ::kQ5_K> { using T = block_q5_K; using B = uint8_t; static constexpr uint32_t kElems = 256; static constexpr uint32_t kSpans = 8; static constexpr int kHalves = 1; };
template <> struct DpasBlk<BankQ::kQ6_K> { using T = block_q6_K; using B = uint8_t; static constexpr uint32_t kElems = 256; static constexpr uint32_t kSpans = 8; static constexpr int kHalves = 2; };

// This lane's column: the span's 32 weight values as 8 VNNI dwords and the span's scale(s) / min.
template <BankQ Q>
inline void span_words(const typename DpasBlk<Q>::T& blk, uint32_t sb, uint32_t* wv, float* sc, float& mn) {
    if constexpr (Q == BankQ::kQ8_0) {
        const auto* q16 = reinterpret_cast<const uint16_t*>(blk.qs);
        #pragma unroll
        for (int w = 0; w < 8; ++w) wv[w] = u16pair(q16 + 2 * w);
        sc[0] = h2f(blk.d); mn = 0.f;
    } else if constexpr (Q == BankQ::kQ5_K) {
        const uint32_t j4 = sb >> 1, sh = (sb & 1) ? 4 : 0;
        const auto* qs = reinterpret_cast<const uint32_t*>(blk.qs) + 8 * j4;
        const auto* qh = reinterpret_cast<const uint32_t*>(blk.qh);
        #pragma unroll
        for (int w = 0; w < 8; ++w) wv[w] = ((qs[w] >> sh) & 0x0F0F0F0Fu) | (((qh[w] >> sb) & 0x01010101u) << 4);
        float s5, m5;
        scale_min_k4(int(sb), blk.scales, s5, m5);
        sc[0] = h2f(blk.d) * s5; mn = h2f(blk.dmin) * m5;
    } else {   // Q6_K: span sb -> half hh = sb >> 2, group g = sb & 3 (32 elements = two 16-sub-blocks)
        const uint32_t hh = sb >> 2, g = sb & 3;
        const auto* ql = reinterpret_cast<const uint16_t*>(blk.ql + 64 * hh + 32 * (g & 1));
        const auto* qh = reinterpret_cast<const uint16_t*>(blk.qh + 32 * hh);
        const uint32_t sh = (g & 2) ? 4 : 0, qsh = 2 * g;
        #pragma unroll
        for (int w = 0; w < 8; ++w)
            wv[w] = ((u16pair(ql + 2 * w) >> sh) & 0x0F0F0F0Fu) | (((u16pair(qh + 2 * w) >> qsh) & 0x03030303u) << 4);
        const float d = h2f(blk.d);
        sc[0] = d * float(blk.scales[8 * hh + 2 * g]);
        sc[1] = d * float(blk.scales[8 * hh + 2 * g + 1]);
        mn = 0.f;
    }
}

template <BankQ Q>
sycl::event tiles_q8d_k(sycl::queue& q, const void* x_q8s, const uint8_t* slot_base, uint64_t slot_bytes,
                        uint64_t gate_off, uint64_t up_off, const int32_t* jobs, sycl::half* yG, sycl::half* yU,
                        uint32_t K, uint32_t N, uint32_t J, const std::vector<sycl::event>& deps) {
    using Blk = typename DpasBlk<Q>::T;
    using BT  = typename DpasBlk<Q>::B;
    constexpr int HALVES = DpasBlk<Q>::kHalves;
    constexpr int B_WORDS = 2 * HALVES * 8 * TN;              // one buffer: (gate | up) x halves x 8 words x 16 columns
    const auto* X = static_cast<const block_q8_1s*>(x_q8s);
    const uint32_t nblk = K / DpasBlk<Q>::kElems;             // weight blocks per column
    const uint32_t n_tiles = N / TN;
    const uint32_t q8_blocks = K / 32;

    return ie::ps(q, "q4e_bank_q8d_gtile", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<int8_t, 1>   sA(2 * 2 * TM * TK, h);          // 2 buffers x 16 rows x 32 s8
        sycl::local_accessor<uint32_t, 1> sB(2 * B_WORDS, h);              // 2 buffers
        sycl::local_accessor<float, 1>    sRow(2 * 16 * 3, h);             // 2 buffers x (d8, s0, s1) per row
        h.parallel_for(sycl::nd_range<2>({uint64_t(J), uint64_t(n_tiles) * SG_SIZE}, {1, SG_SIZE}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            auto sg = it.get_sub_group();
            const uint32_t j = uint32_t(it.get_group(0)), nt = uint32_t(it.get_group(1));
            const uint32_t lane = uint32_t(it.get_local_id(1));
            const int32_t xbase = jobs[4 * j + 0], rows = jobs[4 * j + 1], slot = jobs[4 * j + 2], ybase = jobs[4 * j + 3];
            const uint32_t n0 = nt * TN, n = n0 + lane;
            const uint32_t mg = (uint32_t(rows) + TM - 1) / TM;
            const uint8_t* wg_ = slot_base + uint64_t(slot) * slot_bytes;
            const Blk* colG = reinterpret_cast<const Blk*>(wg_ + gate_off) + uint64_t(n) * nblk;
            const Blk* colU = reinterpret_cast<const Blk*>(wg_ + up_off) + uint64_t(n) * nblk;

            float faccG[2][TM], faccU[2][TM];
            for (int g = 0; g < 2; ++g)
                for (int r = 0; r < TM; ++r) { faccG[g][r] = 0.f; faccU[g][r] = 0.f; }
            int rowmap[TM]; bool have_map = false;
            for (int r = 0; r < TM; ++r) rowmap[r] = r;

            int8_t*   sA0 = &sA.get_multi_ptr<sycl::access::decorated::no>()[0];
            uint32_t* sB0 = &sB.get_multi_ptr<sycl::access::decorated::no>()[0];
            float*    sR0 = &sRow.get_multi_ptr<sycl::access::decorated::no>()[0];
            uint32_t span = 0;
            for (uint32_t b = 0; b < nblk; ++b) {
                for (uint32_t sb = 0; sb < DpasBlk<Q>::kSpans; ++sb, ++span) {
                    const uint32_t buf = span & 1u;
                    int8_t*   sAb = sA0 + buf * (2 * TM * TK);
                    uint32_t* sBb = sB0 + buf * B_WORDS;
                    float*    sRb = sR0 + buf * (16 * 3);
                    // ---- stage A (every row's 32 q8 bytes) + the row scales ----
                    for (uint32_t i = lane; i < uint32_t(rows) * 8; i += SG_SIZE) {
                        const uint32_t r2 = i / 8, w = i % 8;
                        const block_q8_1s& xb = X[uint64_t(xbase + r2) * q8_blocks + span];
                        reinterpret_cast<uint32_t*>(sAb)[r2 * 8 + w] = reinterpret_cast<const uint32_t*>(xb.qs)[w];
                        if (w == 0) { sRb[r2 * 3 + 0] = xb.d; sRb[r2 * 3 + 1] = xb.s0; sRb[r2 * 3 + 2] = xb.s1; }
                    }
                    // ---- stage B: this lane's column, gate then up ----
                    float scG[2] = {0.f, 0.f}, scU[2] = {0.f, 0.f}, mnG = 0.f, mnU = 0.f;
                    uint32_t wvG[8], wvU[8];
                    span_words<Q>(colG[b], sb, wvG, scG, mnG);
                    span_words<Q>(colU[b], sb, wvU, scU, mnU);
                    if constexpr (HALVES == 1) {
                        #pragma unroll
                        for (int w = 0; w < 8; ++w) {
                            sBb[uint32_t(w) * TN + lane]          = wvG[w];
                            sBb[8 * TN + uint32_t(w) * TN + lane] = wvU[w];
                        }
                    } else {
                        #pragma unroll
                        for (int w = 0; w < 8; ++w) {
                            sBb[uint32_t(w) * TN + lane]               = w < 4 ? wvG[w] : 0u;   // gate, the low 16 elements
                            sBb[8 * TN + uint32_t(w) * TN + lane]      = w < 4 ? 0u : wvG[w];   // gate, the high 16
                            sBb[2 * 8 * TN + uint32_t(w) * TN + lane]  = w < 4 ? wvU[w] : 0u;   // up, low
                            sBb[3 * 8 * TN + uint32_t(w) * TN + lane]  = w < 4 ? 0u : wvU[w];   // up, high
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // ---- dpas per M-group, straight-line: gate, up (and their high halves for Q6_K), then the fp chains ----
                    auto bbase = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                          sycl::access::decorated::no>(reinterpret_cast<BT*>(sBb));
                    auto abase = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                          sycl::access::decorated::no>(sAb);
                    for (uint32_t g = 0; g < mg; ++g) {
                        mat::joint_matrix<sycl::sub_group, int8_t, mat::use::a, TM, TK, mat::layout::row_major> A;
                        mat::joint_matrix<sycl::sub_group, BT, mat::use::b, TK, TN, mat::layout::ext_intel_packed> Bg, Bu;
                        mat::joint_matrix<sycl::sub_group, int32_t, mat::use::accumulator, TM, TN> Cg, Cu;
                        mat::joint_matrix_load(sg, A, abase + g * TM * TK, TK);
                        mat::joint_matrix_load(sg, Bg, bbase, TN * 4);
                        mat::joint_matrix_load(sg, Bu, bbase + HALVES * 8 * TN * 4, TN * 4);
                        mat::joint_matrix_fill(sg, Cg, 0);
                        mat::joint_matrix_mad(sg, Cg, A, Bg, Cg);
                        mat::joint_matrix_fill(sg, Cu, 0);
                        mat::joint_matrix_mad(sg, Cu, A, Bu, Cu);
                        const float* rsb = sRb + g * TM * 3;
                        {
                            int idx = 0;
                            imat::joint_matrix_apply(sg, Cg, [&](int32_t& v, size_t r2, size_t c2) {
                                (void)c2;
                                if (!have_map && g == 0) rowmap[idx] = int(r2);
                                const float* rs = rsb + r2 * 3;
                                if constexpr (Q == BankQ::kQ8_0)      faccG[g][idx] += scG[0] * (rs[0] * float(v));
                                else if constexpr (Q == BankQ::kQ5_K) faccG[g][idx] += scG[0] * (rs[0] * float(v)) - mnG * (rs[1] + rs[2]);
                                else                                  faccG[g][idx] += scG[0] * (rs[0] * float(v) - 32.f * rs[1]);
                                ++idx;
                            });
                        }
                        {
                            int idx = 0;
                            imat::joint_matrix_apply(sg, Cu, [&](int32_t& v, size_t r2, size_t c2) {
                                (void)c2;
                                const float* rs = rsb + r2 * 3;
                                if constexpr (Q == BankQ::kQ8_0)      faccU[g][idx] += scU[0] * (rs[0] * float(v));
                                else if constexpr (Q == BankQ::kQ5_K) faccU[g][idx] += scU[0] * (rs[0] * float(v)) - mnU * (rs[1] + rs[2]);
                                else                                  faccU[g][idx] += scU[0] * (rs[0] * float(v) - 32.f * rs[1]);
                                ++idx;
                            });
                        }
                        if constexpr (HALVES == 2) {
                            mat::joint_matrix<sycl::sub_group, BT, mat::use::b, TK, TN, mat::layout::ext_intel_packed> Bg2, Bu2;
                            mat::joint_matrix<sycl::sub_group, int32_t, mat::use::accumulator, TM, TN> Cg2, Cu2;
                            mat::joint_matrix_load(sg, Bg2, bbase + 8 * TN * 4, TN * 4);
                            mat::joint_matrix_load(sg, Bu2, bbase + 3 * 8 * TN * 4, TN * 4);
                            mat::joint_matrix_fill(sg, Cg2, 0);
                            mat::joint_matrix_mad(sg, Cg2, A, Bg2, Cg2);
                            mat::joint_matrix_fill(sg, Cu2, 0);
                            mat::joint_matrix_mad(sg, Cu2, A, Bu2, Cu2);
                            {
                                int idx = 0;
                                imat::joint_matrix_apply(sg, Cg2, [&](int32_t& v, size_t r2, size_t c2) {
                                    (void)c2;
                                    const float* rs = rsb + r2 * 3;
                                    faccG[g][idx] += scG[1] * (rs[0] * float(v) - 32.f * rs[2]);
                                    ++idx;
                                });
                            }
                            {
                                int idx = 0;
                                imat::joint_matrix_apply(sg, Cu2, [&](int32_t& v, size_t r2, size_t c2) {
                                    (void)c2;
                                    const float* rs = rsb + r2 * 3;
                                    faccU[g][idx] += scU[1] * (rs[0] * float(v) - 32.f * rs[2]);
                                    ++idx;
                                });
                            }
                        }
                    }
                    have_map = true;
                }
            }
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

// ---- the DOWN tiles for Q8_0 banks (K = 640, N = 2560; every Unsloth build's down banks): the exact-int f16-DPAS form of
// gemv_q5_1_f16d_grouped_tiles without its min term. B tiles hold f16(q) (an int8 is exact in f16), A = the f16 activation
// rows straight from global (the swiglu output), C fp32:  sum_k x_k * (q_k * d_b) = d_b * (x . q)_b, d_b applied per
// 32-block in fp32, block-ascending per output (the dpas-form order). No activation quantisation, no row sums. The B stage
// is double-buffered by block parity: one barrier a block.
sycl::event tiles_down_f16d_q8_0(sycl::queue& q, const sycl::half* x, const uint8_t* slot_base, uint64_t slot_bytes,
                                 uint64_t down_off, const int32_t* jobs, sycl::half* y, uint32_t K, uint32_t N, uint32_t J,
                                 const std::vector<sycl::event>& deps) {
    constexpr int TKF = 16;                       // fp16 dpas K
    const uint32_t blocks = K / 32, n_tiles = N / TN;
    return ie::ps(q, "q4e_bank_q8_f16d_down", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> sBt(2 * 32 * TN, h);
        h.parallel_for(sycl::nd_range<2>({uint64_t(J), uint64_t(n_tiles) * SG_SIZE}, {1, SG_SIZE}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            auto sg = it.get_sub_group();
            const uint32_t j = uint32_t(it.get_group(0)), nt = uint32_t(it.get_group(1));
            const uint32_t lane = uint32_t(it.get_local_id(1));
            const int32_t xbase = jobs[4 * j + 0], rows = jobs[4 * j + 1], slot = jobs[4 * j + 2], ybase = jobs[4 * j + 3];
            const uint32_t n0 = nt * TN, n = n0 + lane;
            const uint32_t mg = (uint32_t(rows) + TM - 1) / TM;
            const auto* W8 = reinterpret_cast<const block_q8_0*>(slot_base + uint64_t(slot) * slot_bytes + down_off);
            const block_q8_0* col = &W8[uint64_t(n) * blocks];
            const sycl::half* Ag = x + uint64_t(xbase) * K;
            float facc[2][TM];
            for (int g = 0; g < 2; ++g)
                for (int r = 0; r < TM; ++r) facc[g][r] = 0.f;
            int rowmap[TM]; bool have_map = false;
            for (int r = 0; r < TM; ++r) rowmap[r] = r;
            sycl::half* sB0 = &sBt.get_multi_ptr<sycl::access::decorated::no>()[0];
            for (uint32_t b = 0; b < blocks; ++b) {
                sycl::half* sBb = sB0 + (b & 1u) * 32 * TN;
                {
                    const block_q8_0& blk = col[b];
                    #pragma unroll
                    for (int i = 0; i < 32; ++i) sBb[uint32_t(i) * TN + lane] = sycl::half(float(blk.qs[i]));
                }
                it.barrier(sycl::access::fence_space::local_space);
                const float d_b = h2f(col[b].d);
                auto bl = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sBb);
                for (uint32_t g = 0; g < mg; ++g) {
                    mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> C;
                    mat::joint_matrix_fill(sg, C, 0.f);
                    #pragma unroll
                    for (int kt = 0; kt < 2; ++kt) {
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::b, TKF, TN, mat::layout::row_major> B;
                        mat::joint_matrix_load(sg, B, bl + kt * TKF * TN, TN);
                        mat::joint_matrix<sycl::sub_group, sycl::half, mat::use::a, TM, TKF, mat::layout::row_major> A;
                        mat::joint_matrix_load(sg, A,
                            sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                                const_cast<sycl::half*>(Ag + uint64_t(g) * TM * K + b * 32 + kt * TKF)), K);
                        mat::joint_matrix_mad(sg, C, A, B, C);
                    }
                    int idx = 0;
                    imat::joint_matrix_apply(sg, C, [&](float& v, size_t r2, size_t c2) {
                        (void)c2;
                        if (!have_map && g == 0) rowmap[idx] = int(r2);
                        facc[g][idx] += d_b * v;
                        ++idx;
                    });
                }
                have_map = true;
            }
            if (n < N)
                for (uint32_t g = 0; g < mg; ++g)
                    for (int i = 0; i < TM; ++i) {
                        const int r2 = rowmap[i] + int(g) * TM;
                        if (r2 < rows) y[uint64_t(ybase + r2) * N + n] = sycl::half(facc[g][i]);
                    }
        });
    });
}

}  // namespace

bool qwen4_bank_has_down_f16d(BankQ kind) noexcept { return kind == BankQ::kQ8_0; }

sycl::event qwen4_bank_gemv_grouped_tiles_down_f16d(sycl::queue& q, BankQ kind, const sycl::half* x, const uint8_t* slot_base,
                                                    uint64_t slot_bytes, uint64_t down_off, const int32_t* tile_jobs,
                                                    sycl::half* y, uint32_t K, uint32_t N, uint32_t J,
                                                    const std::vector<sycl::event>& deps) {
    if (kind != BankQ::kQ8_0) return {};
    return tiles_down_f16d_q8_0(q, x, slot_base, slot_bytes, down_off, tile_jobs, y, K, N, J, deps);
}

bool qwen4_bank_has_q8dpas(BankQ kind) noexcept {
    return kind == BankQ::kQ8_0 || kind == BankQ::kQ5_K || kind == BankQ::kQ6_K;
}

sycl::event qwen4_bank_gemv_grouped_tiles_q8d(sycl::queue& q, BankQ kind, const void* x_q8s_mega, const uint8_t* slot_base,
                                              uint64_t slot_bytes, uint64_t gate_off, uint64_t up_off, const int32_t* tile_jobs,
                                              sycl::half* g_out, sycl::half* u_out, uint32_t K, uint32_t N, uint32_t J,
                                              const std::vector<sycl::event>& deps) {
    switch (kind) {
        case BankQ::kQ8_0: return tiles_q8d_k<BankQ::kQ8_0>(q, x_q8s_mega, slot_base, slot_bytes, gate_off, up_off, tile_jobs, g_out, u_out, K, N, J, deps);
        case BankQ::kQ5_K: return tiles_q8d_k<BankQ::kQ5_K>(q, x_q8s_mega, slot_base, slot_bytes, gate_off, up_off, tile_jobs, g_out, u_out, K, N, J, deps);
        case BankQ::kQ6_K: return tiles_q8d_k<BankQ::kQ6_K>(q, x_q8s_mega, slot_base, slot_bytes, gate_off, up_off, tile_jobs, g_out, u_out, K, N, J, deps);
        default: return {};
    }
}

}  // namespace ie
