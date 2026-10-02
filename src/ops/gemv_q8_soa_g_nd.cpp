// src/ops/gemv_q8_soa_g_nd.cpp — P4 B37 (3): gemv_q8_0_soa_q8_g_nd, the T = 1 int-dot GEMV over the Q8_0-SoA weights
// with the activation read from global (no SLM staging): Flash-Next's (qwen4exp) T == 1 projections, shared expert and
// LM head, and the 27B split's IE_QWEN35_SOA_GMEM opt-in. Reached through gemv_q8_soa_v2.hpp's gemv_q8_0_soa_q8_g_sel
// (IE_Q8_SOA_GEMV_G_ND=0 = v1 gemv_q8_0_soa_q8_g in gemv_q8dot.cpp, untouched).
//
// Why: v1's bmg-g31 ISA (installed IGC 2.36.5) spends 28.12 instructions per dp4a in the dot block, 24 of them byte
// movs (ie::dp4a_ss over vector-loaded words, 12 movs per operand word; P4 B31). Flash-Next's decode GEMVs mostly
// stream weights, so the gain is bounded by bandwidth -- measured by the test's --bench, not assumed.
//
// BIT-EXACT vs v1 by construction: v1's geometry (one column per subgroup, 32 columns per work-group, no SLM), the same
// per-lane partials (K-blocks lane, lane+16, ... in order; the scale through the native half->float conversion, mul
// dw*d8 then mad from a +0 accumulator), the same reduce_over_group and half(acc) store; only the integer dot is lowered
// natively. IE_GEMV_SMALLK's split-K kernel (v1's opt-in, not bit-exact vs _g) stays v1's: with it set, this entry
// hands the call to v1's public entry under v1's own rule. tests/unit/gemv_q8_soa_nd_test.cpp checks it bit for bit.

#include "ie/gemv_q8_soa_v2.hpp"

#include "ie/dp4a_native.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/quant_blocks.hpp"

#include <cstdlib>

namespace ie {
namespace {

// fp16→fp32 (copy-not-hoist of gemv_q8dot.cpp's dev_fp16_to_fp32: the native half->float conversion).
inline float gnd_fp16_to_fp32(uint16_t h) {
    return float(sycl::bit_cast<sycl::half>(h));
}

}  // namespace

sycl::event gemv_q8_0_soa_q8_g_nd(sycl::queue& q,
                                  const void* x_q8, const int8_t* qs_W, const uint16_t* d_W,
                                  sycl::half* y,
                                  uint32_t K, uint32_t N,
                                  const std::vector<sycl::event>& deps) {
    static const bool smallk = std::getenv("IE_GEMV_SMALLK") != nullptr;   // v1's opt-in and rule (gemv_q8dot.cpp)
    if (smallk && K <= 6144 && (K % 32) == 0)
        return gemv_q8_0_soa_q8_g(q, x_q8, qs_W, d_W, y, K, N, deps);
    constexpr int SG_SIZE  = 16;
    constexpr int N_PER_WG = 32;
    constexpr int WG_ITEMS = N_PER_WG * SG_SIZE;

    const auto* X8 = static_cast<const block_q8_1x*>(x_q8);
    const uint32_t blocks_per_col = K / 32;
    const uint32_t n_wgs = (N + N_PER_WG - 1) / N_PER_WG;

    return ie::ps(q, "gemv_q8_0_soa_g_nd", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(n_wgs) * WG_ITEMS, WG_ITEMS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const uint32_t lid   = uint32_t(it.get_local_id(0));
            const uint32_t wgid  = uint32_t(it.get_group(0));
            const uint32_t sg_id = lid / SG_SIZE;
            const uint32_t lane  = lid % SG_SIZE;
            const uint32_t n     = wgid * N_PER_WG + sg_id;
            if (n >= N) return;

            const int8_t*     wcol = qs_W + uint64_t(n) * K;
            const gguf_half*  dcol = d_W  + uint64_t(n) * blocks_per_col;
            float acc = 0.f;
            for (uint32_t b = lane; b < blocks_per_col; b += SG_SIZE) {
                const uint32_t* wq = reinterpret_cast<const uint32_t*>(wcol + uint64_t(b) * 32);
                const uint32_t* xq = reinterpret_cast<const uint32_t*>(X8[b].qs);
                int32_t idot = 0;
                #pragma unroll
                for (int w = 0; w < 8; ++w) idot = ie::dp4a_ss_native(wq[w], xq[w], idot);
                acc += gnd_fp16_to_fp32(dcol[b]) * float(X8[b].d) * float(idot);
            }
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[n] = sycl::half(acc);
        });
    });
}

}  // namespace ie
