// src/ops/gemv_q8_soa_nd.cpp — P4 B37 (2): gemv_q8_0_soa_q8_nd, the T = 1 decode int-dot GEMV over the Q8_0-SoA
// weights with the SLM-staged activation (the 27B split's and the 35B crown's T == 1 dense projections), reached through
// gemv_q8_soa_v2.hpp's gemv_q8_0_soa_q8_sel (IE_Q8_SOA_GEMV_ND=0 = v1 gemv_q8_0_soa_q8 in gemv_q8dot.cpp, untouched).
// "_nd" (native dot), not "_v2": gemv_q8_0_soa_q8_v2 already names the coalesced-load opt-in variant.
//
// Why: v1's bmg-g31 ISA (installed IGC 2.36.5, offline) spends 28 instructions per dp4a in the dot block, 24 of them
// byte movs (ie::dp4a_ss over vector-loaded words, 12 movs per operand word; P4 B31). At T = 1 the GEMV mostly streams
// weights, so the gain is bounded by bandwidth -- measured by the test's --bench, not assumed.
//
// BIT-EXACT vs v1 by construction: v1's geometry (one column per subgroup, 16 columns per work-group, the activation
// staged once per work-group in SLM), the same per-lane partials (K-blocks lane, lane+16, ... in order; mul dw*d8 then
// mad from a +0 accumulator), the same reduce_over_group and half(acc) store; only the integer dot is lowered natively.
// tests/unit/gemv_q8_soa_nd_test.cpp checks it bit for bit. ESIMD-safe: plain SLM + dp4a.

#include "ie/gemv_q8_soa_v2.hpp"

#include "ie/dp4a_native.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/quant_blocks.hpp"

namespace ie {
namespace {

// fp16→fp32 (copy-not-hoist of gemv_q8dot.cpp's dev_fp16_to_fp32: the native half->float conversion).
inline float nd_fp16_to_fp32(uint16_t h) {
    return float(sycl::bit_cast<sycl::half>(h));
}

}  // namespace

sycl::event gemv_q8_0_soa_q8_nd(sycl::queue& q,
                                const void* x_q8, const int8_t* qs_W, const uint16_t* d_W,
                                sycl::half* y,
                                uint32_t K, uint32_t N,
                                const std::vector<sycl::event>& deps) {
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 16;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;

    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q8_0_soa_nd", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> q8s(blocks_per_col * 8, h);   // act qs words
        sycl::local_accessor<float, 1>    q8d(blocks_per_col, h);       // act d

        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;

            for (uint32_t i = lid; i < blocks_per_col * 8; i += WG_ITEMS) {
                const uint32_t blk = i / 8, w = i % 8;
                q8s[i] = reinterpret_cast<const uint32_t*>(X8[blk].qs)[w];
            }
            for (uint32_t i = lid; i < blocks_per_col; i += WG_ITEMS) q8d[i] = X8[i].d;
            sycl::group_barrier(it.get_group());
            if (n >= N) return;

            const int8_t*     wcol = qs_W + uint64_t(n) * K;
            const gguf_half*  dcol = d_W  + uint64_t(n) * blocks_per_col;
            float acc = 0.f;
            for (uint32_t b = lane; b < blocks_per_col; b += SG_SIZE) {
                const uint32_t* wq = reinterpret_cast<const uint32_t*>(wcol + uint64_t(b) * 32);
                int32_t idot = 0;
                #pragma unroll
                for (int w = 0; w < 8; ++w) idot = ie::dp4a_ss_native(wq[w], q8s[b * 8 + w], idot);
                acc += nd_fp16_to_fp32(dcol[b]) * q8d[b] * float(idot);
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

}  // namespace ie
