// src/ops/moe_q8_gate_up_v2.cpp — P4 B32 (1): moe_prefill_gate_up_silu_q8 v2, the crown split's expert-batched Q8_0
// gate + up + silu (IE_Q8_MOE_GATEUP_V2=0 = the v1 kernel in moe_q8.cpp).
//
// Why: v1 is the crown split's second prefill kernel (19.6 % of card 0 at T = 2042, b30prof) and its bmg-g31 ISA (the
// installed IGC 2.36.5, offline) issues ~9.7 instructions per dp4a, two thirds of them byte movs: ie::dp4a_ss over
// vector-loaded words is lowered through <4 x i8> extract / insert, 12 movs per activation word on every row (96 per
// 16 dp4a) and 12 per weight word once per K-block (P4 B31, ~/ds41_work/p60/b31/BUILD.md). v2 = B31's recipe: the
// native integer dot (dp4a_native.hpp) and V2G_NB output columns per subgroup, so one SLM read of an activation block
// feeds V2G_NB gate + up column pairs (v1 reads the activation tile once per column: 21.5 GB of SLM per layer at T=2048).
//
// BIT-EXACT vs v1: per (row, column) every lane forms the same partials -- gacc / uacc = 0, then += gd*d8*float(gi) /
// ud*d8*float(ui) over its K-blocks j = lane, lane+16, ... in ascending order (the integer dots are exact) -- the same
// two reduce_over_group calls, and the column's silu(gate)*up is computed from the same two uniform sums by the same
// expression as v1, then moved to lane c for one store. tests/unit/moe_q8_gate_up_v2_test.cpp checks it bit for bit.
// ESIMD-safe: plain SLM + dp4a, no block2d / lsc paths.

#include "ie/moe_q8.hpp"

#include "ie/dp4a_native.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/quant_blocks.hpp"

namespace ie {
namespace {

// fp16→fp32 (copy-not-hoist, same as moe_q8.cpp).
inline float v2g_fp16_to_fp32(uint16_t h) {
    const uint32_t s = uint32_t(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1fu;
    uint32_t m =  h        & 0x3ffu;
    if (e == 0) {
        if (m == 0) return sycl::bit_cast<float>(s);
        while ((m & 0x400u) == 0) { m <<= 1; e -= 1; }
        e += 1; m &= ~0x400u;
    } else if (e == 31) {
        return sycl::bit_cast<float>(s | 0x7f800000u | (m << 13));
    }
    e += (127 - 15);
    return sycl::bit_cast<float>(s | (e << 23) | (m << 13));
}

// half(silu(gr) * ur) exactly as v1's ISA computes it (IGC 2.36.5, moe_prefill_gate_up_silu_q8): v1 writes
// gr / (1.f + sycl::exp(-gr)) * ur and IGC lowers sycl::exp(x) to exp2(n) * exp2(y), n = trunc(x * log2e) with a two-
// constant ln2 reduction, forms 1 + exp2(n) * exp2(y) as ONE mad, clamps at |gr| > 105 and divides as (gr * ur) * inv.
// Whether that 1 + e1 * e2 is fused is IGC's choice per context (in a first v2 it was mul + add: a different rounding),
// so v2 spells the lowered sequence out; tests/unit/moe_q8_gate_up_v2_test.cpp checks it bit for bit against v1.
inline sycl::half v2g_silu_mul(float gr, float ur) {
    const float log2e  = sycl::bit_cast<float>(0x3FB8AA3Bu);
    const float ln2_hi = sycl::bit_cast<float>(0xBF317200u);   // -ln2, high part
    const float ln2_lo = sycl::bit_cast<float>(0xB5BFBE8Eu);   // -ln2, low part
    const float n = sycl::trunc(gr * -log2e);                   // exponent of exp(-gr)
    float r = sycl::fma(n, ln2_hi, -gr);
    r = sycl::fma(n, ln2_lo, r);
    float den = sycl::fma(sycl::exp2(n), sycl::exp2(r * log2e), 1.f);   // 1 + exp(-gr), one rounding
    if (gr > 105.f) den = 1.f;
    if (gr < -105.f) den = sycl::bit_cast<float>(0x7F800000u);
    const float s_ = gr / den;                                  // v1's form: IGC makes it (gr * ur) * inv(den)
    return sycl::half(s_ * ur);
}

constexpr int      V2G_SG  = 16;                         // lanes: K-blocks j = lane, lane+16, ... (as v1)
constexpr int      V2G_SGS = 32;                         // subgroups per work-group
constexpr int      V2G_NB  = 2;                          // output columns (gate + up pairs) per subgroup
constexpr int      V2G_TM  = 8;                          // routed rows per staged tile
constexpr int      V2G_WG  = V2G_SG * V2G_SGS;           // 512
constexpr uint32_t V2G_BW  = sizeof(block_q8_1x) / 4;    // 10 words / activation block: word 0 = d, 2..9 = qs

}  // namespace

sycl::event moe_prefill_gate_up_silu_q8_v2(sycl::queue& q, const void* xq8_packed,
                                           const int8_t* g_qs, const uint16_t* g_d,
                                           const int8_t* u_qs, const uint16_t* u_d,
                                           uint64_t qs_stride, uint64_t d_stride,
                                           const uint32_t* expert_offsets,
                                           sycl::half* h_packed,
                                           uint32_t E, uint32_t H, uint32_t E_ffn,
                                           const std::vector<sycl::event>& deps) {
    const uint32_t q8_per_row     = H / 32;               // activation blocks / row == weight blocks / column
    const uint32_t n_blk_per_lane = (q8_per_row + V2G_SG - 1) / V2G_SG;
    const uint32_t cols_per_wg    = uint32_t(V2G_SGS * V2G_NB);
    const uint32_t n_chunks       = (E_ffn + cols_per_wg - 1) / cols_per_wg;
    const auto* X8 = static_cast<const block_q8_1x*>(xq8_packed);
    return ie::ps(q, "moe_pfl_gate_up_q8_v2", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> sx(uint64_t(V2G_TM) * q8_per_row * V2G_BW, h);
        h.parallel_for(sycl::nd_range<2>({uint64_t(E) * n_chunks, V2G_WG}, {1, V2G_WG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(V2G_SG)]] {
            const uint32_t en = uint32_t(it.get_group(0));
            const uint32_t e  = en / n_chunks;
            const uint32_t nc = en % n_chunks;
            const uint32_t lid  = uint32_t(it.get_local_id(1));
            const uint32_t sg = lid / V2G_SG, lane = lid % V2G_SG;
            const uint32_t n0 = nc * cols_per_wg + sg * V2G_NB;     // this subgroup's columns n0 .. n0+V2G_NB-1
            const uint32_t off_start = expert_offsets[e];
            const uint32_t n_tok     = expert_offsets[e + 1] - off_start;
            if (n_tok == 0) return;                                  // an empty expert: the whole work-group leaves

            const int8_t*   gcol = g_qs + e * qs_stride + uint64_t(n0) * H;
            const int8_t*   ucol = u_qs + e * qs_stride + uint64_t(n0) * H;
            const uint16_t* gsc  = g_d  + e * d_stride  + uint64_t(n0) * q8_per_row;
            const uint16_t* usc  = u_d  + e * d_stride  + uint64_t(n0) * q8_per_row;
            const uint32_t* base = sx.get_multi_ptr<sycl::access::decorated::no>().get();

            for (uint32_t tk = 0; tk < n_tok; tk += V2G_TM) {
                const uint32_t M = sycl::min(uint32_t(V2G_TM), n_tok - tk);
                const uint32_t* src = reinterpret_cast<const uint32_t*>(
                    X8 + uint64_t(off_start + tk) * q8_per_row);
                for (uint32_t i = lid; i < M * q8_per_row * V2G_BW; i += V2G_WG) sx[i] = src[i];
                sycl::group_barrier(it.get_group());
                if (n0 < E_ffn) {                                    // subgroup-uniform
                    float gacc[V2G_NB][V2G_TM], uacc[V2G_NB][V2G_TM];
                    #pragma unroll
                    for (int c = 0; c < V2G_NB; ++c) {
                        #pragma unroll
                        for (int mm = 0; mm < V2G_TM; ++mm) { gacc[c][mm] = 0.f; uacc[c][mm] = 0.f; }
                    }
                    for (uint32_t s = 0; s < n_blk_per_lane; ++s) {
                        const uint32_t j = lane + s * V2G_SG;
                        if (j >= q8_per_row) break;
                        uint32_t gw[V2G_NB][8], uw[V2G_NB][8];
                        float    gd[V2G_NB], ud[V2G_NB];
                        #pragma unroll
                        for (int c = 0; c < V2G_NB; ++c) {
                            // a column past E_ffn reads column n0 instead; it is never stored
                            const uint64_t cc = (n0 + uint32_t(c) < E_ffn) ? uint64_t(c) : 0u;
                            const uint32_t* gp = reinterpret_cast<const uint32_t*>(gcol + cc * H + uint64_t(j) * 32);
                            const uint32_t* up = reinterpret_cast<const uint32_t*>(ucol + cc * H + uint64_t(j) * 32);
                            #pragma unroll
                            for (int w = 0; w < 8; ++w) { gw[c][w] = gp[w]; uw[c][w] = up[w]; }
                            gd[c] = v2g_fp16_to_fp32(gsc[cc * q8_per_row + j]);
                            ud[c] = v2g_fp16_to_fp32(usc[cc * q8_per_row + j]);
                        }
                        #pragma unroll
                        for (int mm = 0; mm < V2G_TM; ++mm) {
                            if (uint32_t(mm) < M) {
                                const uint32_t* blk = base + (uint32_t(mm) * q8_per_row + j) * V2G_BW;
                                const float d8 = sycl::bit_cast<float>(blk[0]);
                                uint32_t xw[8];
                                #pragma unroll
                                for (int w = 0; w < 8; ++w) xw[w] = blk[2 + w];
                                #pragma unroll
                                for (int c = 0; c < V2G_NB; ++c) {
                                    int32_t gi = 0, ui = 0;
                                    #pragma unroll
                                    for (int w = 0; w < 8; ++w) {
                                        gi = ie::dp4a_ss_native(gw[c][w], xw[w], gi);
                                        ui = ie::dp4a_ss_native(uw[c][w], xw[w], ui);
                                    }
                                    gacc[c][mm] += gd[c] * d8 * float(gi);
                                    uacc[c][mm] += ud[c] * d8 * float(ui);
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int mm = 0; mm < V2G_TM; ++mm) {
                        if (uint32_t(mm) < M) {
                            sycl::half h_mine = sycl::half(0.f);    // lane c keeps column n0 + c's value
                            #pragma unroll
                            for (int c = 0; c < V2G_NB; ++c) {
                                const float gr = sycl::reduce_over_group(it.get_sub_group(), gacc[c][mm], sycl::plus<float>());
                                const float ur = sycl::reduce_over_group(it.get_sub_group(), uacc[c][mm], sycl::plus<float>());
                                const sycl::half hv = v2g_silu_mul(gr, ur);     // silu(gate) * up, as v1's ISA
                                if (lane == uint32_t(c)) h_mine = hv;
                            }
                            if (lane < uint32_t(V2G_NB) && n0 + lane < E_ffn)
                                h_packed[uint64_t(off_start + tk + mm) * E_ffn + n0 + lane] = h_mine;
                        }
                    }
                }
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

}  // namespace ie
