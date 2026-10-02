// src/ops/gemv_q8_soa_batched_v2.cpp — P4 B37 (1): gemv_q8_0_soa_q8_batched / _batched_dual v2, the T-row (2..16)
// int-dot GEMVs over the Q8_0-SoA weights, reached through gemv_q8_soa_v2.hpp's _sel selectors (IE_Q8_SOA_BATCHED_V2=0
// = the v1 kernels in gemv_q8dot.cpp, untouched). They serve the request lanes' rows step (27B split stage_card_rows
// incl. the dual k+v / ffn gate+up launches, the 35B crown's srows, Flash-Next's run_block_rows), spec-decode verify
// and Flash-Next's T <= 16 projections.
//
// Why: their bmg-g31 ISA (installed IGC 2.36.5, offline) spends 14.8 (T bucket 16) to 21.1 (T bucket 2) instructions
// per dp4a in the dot blocks, 12.75-18 of them byte movs: ie::dp4a_ss over vector-loaded words is lowered through
// <4 x i8> extract / insert, 12 movs per operand word -- every activation word of every row (P4 B31,
// ~/ds41_work/p60/b31/BUILD.md §8). v2 changes ONLY the integer-dot lowering (dp4a_native.hpp): v1's geometry (one
// column per subgroup, 16 columns per work-group), T buckets, SLM rule and T > 16 chunking are kept, so occupancy and the
// launch shape are unchanged.
//
// BIT-EXACT vs v1 by construction: the same per-lane partials (K-blocks lane, lane+16, ... in order; mul dw*d8 then mad
// from a +0 accumulator, per row), the same reduce_over_group per row and the same half(r) store; the integer dots are
// exact. tests/unit/gemv_q8_soa_batched_v2_test.cpp checks it bit for bit. ESIMD-safe: plain SLM + dp4a.

#include "ie/gemv_q8_soa_v2.hpp"

#include "ie/dp4a_native.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/quant_blocks.hpp"

#include <algorithm>
#include <cstdlib>

namespace ie {
namespace {

// fp16→fp32 (copy-not-hoist of gemv_q8dot.cpp's dev_fp16_to_fp32: the native half->float conversion).
inline float b2_fp16_to_fp32(uint16_t h) {
    return float(sycl::bit_cast<sycl::half>(h));
}

// v1's SLM-vs-global rule, copied (gemv_q8dot.cpp q8bat_use_slm, the same IE_Q8BAT_SLM_KB override): stage the T
// activation streams in SLM when they fit the budget. Either path reads the same bytes in the same order.
inline bool b2_use_slm(uint32_t K, uint32_t T) {
    static const uint32_t kb = [] {
        const char* e = std::getenv("IE_Q8BAT_SLM_KB");
        return e ? uint32_t(std::atoi(e)) : 24u;
    }();
    return uint64_t(T) * (K / 32) * 36 <= uint64_t(kb) * 1024;
}

// One launch over NMAT (1 or 2) same-shape weights against one shared activation: the WG range's first n_wgs_one
// work-groups run matrix A, the next n_wgs_one matrix B (v1's batched_impl / batched_dual_impl, one body).
template <int T_MAX, bool SLM, int NMAT>
sycl::event b2_impl(sycl::queue& q, const char* name, const void* x_q8,
                    const int8_t* qsA, const uint16_t* dA, sycl::half* yA,
                    const int8_t* qsB, const uint16_t* dB, sycl::half* yB,
                    uint32_t K, uint32_t N, uint32_t T, const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;

    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs_one = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, name, [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> xs(SLM ? uint64_t(T) * blocks_per_col * 8 : 1, h);
        sycl::local_accessor<float, 1>    xd(SLM ? uint64_t(T) * blocks_per_col     : 1, h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs_one) * NMAT * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const bool     isB   = NMAT == 2 && wgid >= n_wgs_one;
            const uint32_t n     = (isB ? wgid - n_wgs_one : wgid) * N_PER_WG + sg_id;
            if constexpr (SLM) {
                const uint32_t nb = T * blocks_per_col;
                for (uint32_t i = lid; i < nb * 8; i += WG_ITEMS) {
                    const uint32_t blk = i / 8, w = i % 8;
                    xs[i] = reinterpret_cast<const uint32_t*>(X8[blk].qs)[w];
                }
                for (uint32_t i = lid; i < nb; i += WG_ITEMS) xd[i] = X8[i].d;
                sycl::group_barrier(it.get_group());
            }
            if (n >= N) return;
            const int8_t*   qs_W = isB ? qsB : qsA;
            const uint16_t* d_W  = isB ? dB  : dA;
            sycl::half*     y    = isB ? yB  : yA;

            const int8_t*    wcol = qs_W + uint64_t(n) * K;
            const gguf_half* dcol = d_W  + uint64_t(n) * blocks_per_col;

            float acc[T_MAX];
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) acc[t] = 0.f;

            for (uint32_t b = lane; b < blocks_per_col; b += SG_SIZE) {
                const uint32_t* wq = reinterpret_cast<const uint32_t*>(wcol + uint64_t(b) * 32);
                uint32_t w8[8];
                #pragma unroll
                for (int w = 0; w < 8; ++w) w8[w] = wq[w];
                const float dw = b2_fp16_to_fp32(dcol[b]);
                #pragma unroll
                for (int t = 0; t < T_MAX; ++t) {
                    if (uint32_t(t) >= T) break;
                    const uint64_t xi = uint64_t(t) * blocks_per_col + b;
                    int32_t idot = 0;
                    if constexpr (SLM) {
                        #pragma unroll
                        for (int w = 0; w < 8; ++w) idot = ie::dp4a_ss_native(w8[w], xs[xi * 8 + w], idot);
                        acc[t] += dw * xd[xi] * float(idot);
                    } else {
                        const block_q8_1x& xb = X8[xi];
                        const uint32_t* xqw = reinterpret_cast<const uint32_t*>(xb.qs);
                        #pragma unroll
                        for (int w = 0; w < 8; ++w) idot = ie::dp4a_ss_native(w8[w], xqw[w], idot);
                        acc[t] += dw * float(xb.d) * float(idot);
                    }
                }
            }
            #pragma unroll
            for (int t = 0; t < T_MAX; ++t) {
                if (uint32_t(t) >= T) break;
                const float r = sycl::reduce_over_group(it.get_sub_group(), acc[t], sycl::plus<float>());
                if (lane == 0) y[uint64_t(t) * N + n] = sycl::half(r);
            }
        });
    });
}

// v1's T-bucket dispatch (gemv_q8dot.cpp), for one or two matrices
template <int NMAT>
sycl::event b2_dispatch(sycl::queue& q, const void* x_q8,
                        const int8_t* qsA, const uint16_t* dA, sycl::half* yA,
                        const int8_t* qsB, const uint16_t* dB, sycl::half* yB,
                        uint32_t K, uint32_t N, uint32_t T, const std::vector<sycl::event>& deps) {
    const char* nm = NMAT == 2 ? "gemv_q8_soa_T2x_v2" : "gemv_q8_soa_T_v2";
    const bool slm = b2_use_slm(K, T);
    if (T == 2)  return slm ? b2_impl<2, true, NMAT>(q, nm, x_q8, qsA, dA, yA, qsB, dB, yB, K, N, T, deps)
                            : b2_impl<2, false, NMAT>(q, nm, x_q8, qsA, dA, yA, qsB, dB, yB, K, N, T, deps);
    if (T == 3)  return slm ? b2_impl<3, true, NMAT>(q, nm, x_q8, qsA, dA, yA, qsB, dB, yB, K, N, T, deps)
                            : b2_impl<3, false, NMAT>(q, nm, x_q8, qsA, dA, yA, qsB, dB, yB, K, N, T, deps);
    if (T <= 4)  return slm ? b2_impl<4, true, NMAT>(q, nm, x_q8, qsA, dA, yA, qsB, dB, yB, K, N, T, deps)
                            : b2_impl<4, false, NMAT>(q, nm, x_q8, qsA, dA, yA, qsB, dB, yB, K, N, T, deps);
    if (T <= 8)  return b2_impl<8, false, NMAT>(q, nm, x_q8, qsA, dA, yA, qsB, dB, yB, K, N, T, deps);
    return b2_impl<16, false, NMAT>(q, nm, x_q8, qsA, dA, yA, qsB, dB, yB, K, N, T, deps);
}

}  // namespace

sycl::event gemv_q8_0_soa_q8_batched_v2(sycl::queue& q,
                                        const void* x_q8, const int8_t* qs_W,
                                        const uint16_t* d_W, sycl::half* y,
                                        uint32_t K, uint32_t N, uint32_t T,
                                        const std::vector<sycl::event>& deps) {
    if (T > 16) {   // v1's 16-row chunks (acc[16] is the register cap)
        sycl::event e;
        const uint64_t row = uint64_t(K / 32) * sizeof(block_q8_1x);
        for (uint32_t t0 = 0; t0 < T; t0 += 16)
            e = gemv_q8_0_soa_q8_batched_v2(q, static_cast<const uint8_t*>(x_q8) + uint64_t(t0) * row,
                                            qs_W, d_W, y + uint64_t(t0) * N, K, N, std::min(16u, T - t0),
                                            t0 == 0 ? deps : std::vector<sycl::event>{});
        return e;
    }
    return b2_dispatch<1>(q, x_q8, qs_W, d_W, y, nullptr, nullptr, nullptr, K, N, T, deps);
}

sycl::event gemv_q8_0_soa_q8_batched_dual_v2(sycl::queue& q, const void* x_q8,
                                             const int8_t* qsA, const uint16_t* dA, sycl::half* yA,
                                             const int8_t* qsB, const uint16_t* dB, sycl::half* yB,
                                             uint32_t K, uint32_t N, uint32_t T,
                                             const std::vector<sycl::event>& deps) {
    if (T > 16) {
        sycl::event e;
        const uint64_t row = uint64_t(K / 32) * sizeof(block_q8_1x);
        for (uint32_t t0 = 0; t0 < T; t0 += 16)
            e = gemv_q8_0_soa_q8_batched_dual_v2(q, static_cast<const uint8_t*>(x_q8) + uint64_t(t0) * row,
                                                 qsA, dA, yA + uint64_t(t0) * N, qsB, dB, yB + uint64_t(t0) * N,
                                                 K, N, std::min(16u, T - t0),
                                                 t0 == 0 ? deps : std::vector<sycl::event>{});
        return e;
    }
    return b2_dispatch<2>(q, x_q8, qsA, dA, yA, qsB, dB, yB, K, N, T, deps);
}

}  // namespace ie
