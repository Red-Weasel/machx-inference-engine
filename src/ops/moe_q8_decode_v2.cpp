// src/ops/moe_q8_decode_v2.cpp — P4 B32 (2): moe_gate_up_silu_q8 / moe_down_q8 v2, the crown split's per-slot Q8_0 MoE
// kernels (IE_Q8_MOE_DECODE_V2=0 = the v1 kernels in moe_q8.cpp). They serve T = 1 decode and the request lanes' rows
// step (forward_stage_rows: the same two calls with T = G rows, G = 2..16), one work-group per (token slot, column chunk).
//
// Why: their bmg-g31 ISA (installed IGC 2.36.5, offline) spends 20.4 (gate_up) and 27.4 (down) instructions per dp4a in
// the dot blocks, 18 and 24 of them byte movs: ie::dp4a_ss over vector-loaded words is lowered through <4 x i8> extract /
// insert, 12 movs per operand word (P4 B31). v2: the native integer dot (dp4a_native.hpp) and several output columns per
// subgroup -- V2D_NB_GU column pairs for gate_up, V2D_NB_DN columns for down -- so one SLM read of an activation word
// feeds them all and a row's columns go out in one store.
//
// BIT-EXACT vs v1: per (slot, column) every lane forms the same partials -- acc = 0, then += wscale*d8*float(idot) over
// its K-blocks b = lane, lane+16, ... in ascending order (the integer dots are exact) -- the same reduce_over_group per
// sum, then v1's epilogue: half(silu(gate)*up) as v1's ISA computes it (v2d_silu_mul, see moe_q8_gate_up_v2.cpp) and
// half(topk_w * sum). tests/unit/moe_q8_decode_v2_test.cpp checks it bit for bit for T = 1..16 rows of 8 slots.
// ESIMD-safe: plain SLM + dp4a, no block2d / lsc paths.

#include "ie/moe_q8.hpp"

#include "ie/dp4a_native.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/quant_blocks.hpp"

namespace ie {
namespace {

// fp16→fp32 (copy-not-hoist, same as moe_q8.cpp).
inline float v2d_fp16_to_fp32(uint16_t h) {
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

// half(silu(gr) * ur) exactly as v1's ISA computes it -- copy-not-hoist of moe_q8_gate_up_v2.cpp's v2g_silu_mul (the
// lowering of gr / (1.f + sycl::exp(-gr)) * ur that IGC 2.36.5 emits in moe_gate_up_silu_q8 too: exp2(n) * exp2(y),
// 1 + their product as ONE mad, the |gr| > 105 clamps, then (gr * ur) * inv).
inline sycl::half v2d_silu_mul(float gr, float ur) {
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

constexpr int V2D_SG    = 16;                            // lanes: K-blocks b = lane, lane+16, ... (as v1)
constexpr int V2D_SGS   = 32;                            // subgroups per work-group (as v1)
constexpr int V2D_WG    = V2D_SG * V2D_SGS;              // 512
constexpr int V2D_NB_GU = 2;                             // gate + up column pairs per subgroup
constexpr int V2D_NB_DN = 4;                             // down columns per subgroup

}  // namespace

sycl::event moe_gate_up_silu_q8_v2(sycl::queue& q, const void* x_q8,
                                   const int8_t* g_qs, const uint16_t* g_d,
                                   const int8_t* u_qs, const uint16_t* u_d,
                                   uint64_t qs_stride, uint64_t d_stride,
                                   const int32_t* topk_idx, sycl::half* h_out,
                                   uint32_t T, uint32_t K, uint32_t H, uint32_t E_ffn) {
    const uint32_t bpc     = H / 32;                              // act blocks per row == weight blocks per column
    const uint32_t cols    = uint32_t(V2D_SGS * V2D_NB_GU);       // columns per work-group
    const uint32_t col_wgs = (E_ffn + cols - 1) / cols;
    const uint64_t n_wgs   = uint64_t(T) * K * col_wgs;
    const auto* X8base = static_cast<const block_q8_1x*>(x_q8);
    return ie::ps(q, "moe_gate_up_silu_q8_v2", [&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> q8s(bpc * 8, h);
        sycl::local_accessor<float, 1>    q8d(bpc, h);
        h.parallel_for(sycl::nd_range<1>(n_wgs * V2D_WG, V2D_WG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(V2D_SG)]] {
            const uint32_t lid  = uint32_t(it.get_local_id(0));
            const uint32_t wgid = uint32_t(it.get_group(0));
            const uint32_t tk = uint32_t(wgid / col_wgs), cw = uint32_t(wgid % col_wgs);
            const uint32_t t  = tk / K;
            const uint32_t sg = lid / V2D_SG, lane = lid % V2D_SG;
            const uint32_t n0 = cw * cols + sg * V2D_NB_GU;           // this subgroup's columns n0 .. n0+NB-1
            const block_q8_1x* X8 = X8base + uint64_t(t) * bpc;
            for (uint32_t i = lid; i < bpc * 8; i += V2D_WG) {
                const uint32_t b = i / 8, w = i % 8;
                q8s[i] = reinterpret_cast<const uint32_t*>(X8[b].qs)[w];
            }
            for (uint32_t i = lid; i < bpc; i += V2D_WG) q8d[i] = X8[i].d;
            sycl::group_barrier(it.get_group());
            if (n0 >= E_ffn) return;
            const uint32_t e = uint32_t(topk_idx[tk]);
            const uint32_t* xs = q8s.get_multi_ptr<sycl::access::decorated::no>().get();
            float ga[V2D_NB_GU], ua[V2D_NB_GU];
            #pragma unroll
            for (int c = 0; c < V2D_NB_GU; ++c) { ga[c] = 0.f; ua[c] = 0.f; }
            for (uint32_t b = lane; b < bpc; b += V2D_SG) {
                uint32_t xw[8];
                #pragma unroll
                for (int w = 0; w < 8; ++w) xw[w] = xs[b * 8 + w];
                const float d8 = q8d[b];
                #pragma unroll
                for (int c = 0; c < V2D_NB_GU; ++c) {
                    // a column past E_ffn reads column n0 instead; it is never stored
                    const uint64_t n = n0 + ((n0 + uint32_t(c) < E_ffn) ? uint32_t(c) : 0u);
                    const uint32_t* gq = reinterpret_cast<const uint32_t*>(g_qs + e * qs_stride + n * H + uint64_t(b) * 32);
                    const uint32_t* uq = reinterpret_cast<const uint32_t*>(u_qs + e * qs_stride + n * H + uint64_t(b) * 32);
                    int32_t gi = 0, ui = 0;
                    #pragma unroll
                    for (int w = 0; w < 8; ++w) {
                        gi = ie::dp4a_ss_native(gq[w], xw[w], gi);
                        ui = ie::dp4a_ss_native(uq[w], xw[w], ui);
                    }
                    ga[c] += v2d_fp16_to_fp32(g_d[e * d_stride + n * bpc + b]) * d8 * float(gi);
                    ua[c] += v2d_fp16_to_fp32(u_d[e * d_stride + n * bpc + b]) * d8 * float(ui);
                }
            }
            sycl::half h_mine = sycl::half(0.f);                    // lane c keeps column n0 + c's value
            #pragma unroll
            for (int c = 0; c < V2D_NB_GU; ++c) {
                const float gr = sycl::reduce_over_group(it.get_sub_group(), ga[c], sycl::plus<float>());
                const float ur = sycl::reduce_over_group(it.get_sub_group(), ua[c], sycl::plus<float>());
                const sycl::half hv = v2d_silu_mul(gr, ur);           // silu(gate) * up, as v1's ISA
                if (lane == uint32_t(c)) h_mine = hv;
            }
            if (lane < uint32_t(V2D_NB_GU) && n0 + lane < E_ffn) h_out[uint64_t(tk) * E_ffn + n0 + lane] = h_mine;
        });
    });
}

sycl::event moe_down_q8_v2(sycl::queue& q, const void* h_q8,
                           const int8_t* d_qs, const uint16_t* d_d,
                           uint64_t qs_stride, uint64_t d_stride,
                           const int32_t* topk_idx, const sycl::half* topk_w,
                           sycl::half* y_packed, uint32_t T, uint32_t K,
                           uint32_t E_ffn, uint32_t H) {
    const uint32_t bpc     = E_ffn / 32;
    const uint32_t cols    = uint32_t(V2D_SGS * V2D_NB_DN);
    const uint32_t col_wgs = (H + cols - 1) / cols;
    const uint64_t n_wgs   = uint64_t(T) * K * col_wgs;
    const auto* X8base = static_cast<const block_q8_1x*>(h_q8);
    return ie::ps(q, "moe_down_q8_v2", [&](sycl::handler& h) {
        sycl::local_accessor<uint32_t, 1> q8s(bpc * 8, h);
        sycl::local_accessor<float, 1>    q8d(bpc, h);
        h.parallel_for(sycl::nd_range<1>(n_wgs * V2D_WG, V2D_WG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(V2D_SG)]] {
            const uint32_t lid  = uint32_t(it.get_local_id(0));
            const uint32_t wgid = uint32_t(it.get_group(0));
            const uint32_t tk = uint32_t(wgid / col_wgs), cw = uint32_t(wgid % col_wgs);
            const uint32_t sg = lid / V2D_SG, lane = lid % V2D_SG;
            const uint32_t n0 = cw * cols + sg * V2D_NB_DN;           // this subgroup's columns n0 .. n0+NB-1
            const block_q8_1x* X8 = X8base + uint64_t(tk) * bpc;
            for (uint32_t i = lid; i < bpc * 8; i += V2D_WG) {
                const uint32_t b = i / 8, w = i % 8;
                q8s[i] = reinterpret_cast<const uint32_t*>(X8[b].qs)[w];
            }
            for (uint32_t i = lid; i < bpc; i += V2D_WG) q8d[i] = X8[i].d;
            sycl::group_barrier(it.get_group());
            if (n0 >= H) return;
            const uint32_t e = uint32_t(topk_idx[tk]);
            const uint32_t* xs = q8s.get_multi_ptr<sycl::access::decorated::no>().get();
            float acc[V2D_NB_DN];
            #pragma unroll
            for (int c = 0; c < V2D_NB_DN; ++c) acc[c] = 0.f;
            for (uint32_t b = lane; b < bpc; b += V2D_SG) {
                uint32_t xw[8];
                #pragma unroll
                for (int w = 0; w < 8; ++w) xw[w] = xs[b * 8 + w];
                const float d8 = q8d[b];
                #pragma unroll
                for (int c = 0; c < V2D_NB_DN; ++c) {
                    // a column past H reads column n0 instead; it is never stored
                    const uint64_t n = n0 + ((n0 + uint32_t(c) < H) ? uint32_t(c) : 0u);
                    const uint32_t* wq = reinterpret_cast<const uint32_t*>(d_qs + e * qs_stride + n * E_ffn + uint64_t(b) * 32);
                    int32_t idot = 0;
                    #pragma unroll
                    for (int w = 0; w < 8; ++w) idot = ie::dp4a_ss_native(wq[w], xw[w], idot);
                    acc[c] += v2d_fp16_to_fp32(d_d[e * d_stride + n * bpc + b]) * d8 * float(idot);
                }
            }
            const float tw = float(topk_w[tk]);
            sycl::half y_mine = sycl::half(0.f);                    // lane c keeps column n0 + c's value
            #pragma unroll
            for (int c = 0; c < V2D_NB_DN; ++c) {
                const float r = sycl::reduce_over_group(it.get_sub_group(), acc[c], sycl::plus<float>());
                const sycl::half yv = sycl::half(tw * r);
                if (lane == uint32_t(c)) y_mine = yv;
            }
            if (lane < uint32_t(V2D_NB_DN) && n0 + lane < H) y_packed[uint64_t(tk) * H + n0 + lane] = y_mine;
        });
    });
}

}  // namespace ie
