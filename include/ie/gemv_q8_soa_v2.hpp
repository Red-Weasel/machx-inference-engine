// include/ie/gemv_q8_soa_v2.hpp — P4 B37: v2 of the Q8_0-SoA int-dot GEMVs and the selectors the model code calls.
//
// v2 = v1's kernels (gemv_q8dot.cpp) with ONLY the integer-dot lowering changed (dp4a_native.hpp): the same geometry,
// buckets and accumulation order, so the output is BIT-IDENTICAL. The v1 kernels stay untouched in gemv_q8dot.cpp: that
// file and ops.hpp key the DS4.1 prompt cache (src/model/deepseek41_numerics_inputs.txt), so the switch lives in these
// selectors, called by the models that run the kernels (qwen35_split, qwen35moe_split, qwen4exp).
#pragma once

#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

namespace ie {

// (1) the T-row batched GEMV (T = 2..16; T > 16 in 16-row chunks) and its two-matrix launch: gemv_q8_soa_batched_v2.cpp
sycl::event gemv_q8_0_soa_q8_batched_v2(sycl::queue& q,
                                        const void* x_q8, const int8_t* qs_W,
                                        const uint16_t* d_W, sycl::half* y,
                                        uint32_t K, uint32_t N, uint32_t T,
                                        const std::vector<sycl::event>& deps = {});
sycl::event gemv_q8_0_soa_q8_batched_dual_v2(sycl::queue& q, const void* x_q8,
                                             const int8_t* qsA, const uint16_t* dA, sycl::half* yA,
                                             const int8_t* qsB, const uint16_t* dB, sycl::half* yB,
                                             uint32_t K, uint32_t N, uint32_t T,
                                             const std::vector<sycl::event>& deps = {});

// IE_Q8_SOA_BATCHED_V2=0 = the v1 kernels. Read once.
inline bool gemv_q8_soa_batched_v2_on() {
    static const bool v = [] { const char* e = std::getenv("IE_Q8_SOA_BATCHED_V2"); return !(e && *e && std::string(e) == "0"); }();
    return v;
}

inline sycl::event gemv_q8_0_soa_q8_batched_sel(sycl::queue& q,
                                                const void* x_q8, const int8_t* qs_W,
                                                const uint16_t* d_W, sycl::half* y,
                                                uint32_t K, uint32_t N, uint32_t T,
                                                const std::vector<sycl::event>& deps = {}) {
    return gemv_q8_soa_batched_v2_on() ? gemv_q8_0_soa_q8_batched_v2(q, x_q8, qs_W, d_W, y, K, N, T, deps)
                                       : gemv_q8_0_soa_q8_batched(q, x_q8, qs_W, d_W, y, K, N, T, deps);
}

inline sycl::event gemv_q8_0_soa_q8_batched_dual_sel(sycl::queue& q, const void* x_q8,
                                                     const int8_t* qsA, const uint16_t* dA, sycl::half* yA,
                                                     const int8_t* qsB, const uint16_t* dB, sycl::half* yB,
                                                     uint32_t K, uint32_t N, uint32_t T,
                                                     const std::vector<sycl::event>& deps = {}) {
    return gemv_q8_soa_batched_v2_on()
               ? gemv_q8_0_soa_q8_batched_dual_v2(q, x_q8, qsA, dA, yA, qsB, dB, yB, K, N, T, deps)
               : gemv_q8_0_soa_q8_batched_dual(q, x_q8, qsA, dA, yA, qsB, dB, yB, K, N, T, deps);
}

// (2) the T = 1 GEMV with the SLM-staged activation (gemv_q8_soa_nd.cpp). "_nd" (native dot): gemv_q8_0_soa_q8_v2
// already names the coalesced-load opt-in variant (not bit-exact).
sycl::event gemv_q8_0_soa_q8_nd(sycl::queue& q,
                                const void* x_q8, const int8_t* qs_W, const uint16_t* d_W,
                                sycl::half* y,
                                uint32_t K, uint32_t N,
                                const std::vector<sycl::event>& deps = {});

// IE_Q8_SOA_GEMV_ND=0 = v1 gemv_q8_0_soa_q8. Read once.
inline bool gemv_q8_soa_nd_on() {
    static const bool v = [] { const char* e = std::getenv("IE_Q8_SOA_GEMV_ND"); return !(e && *e && std::string(e) == "0"); }();
    return v;
}

inline sycl::event gemv_q8_0_soa_q8_sel(sycl::queue& q,
                                        const void* x_q8, const int8_t* qs_W, const uint16_t* d_W,
                                        sycl::half* y,
                                        uint32_t K, uint32_t N,
                                        const std::vector<sycl::event>& deps = {}) {
    return gemv_q8_soa_nd_on() ? gemv_q8_0_soa_q8_nd(q, x_q8, qs_W, d_W, y, K, N, deps)
                               : gemv_q8_0_soa_q8(q, x_q8, qs_W, d_W, y, K, N, deps);
}

// (3) the T = 1 GEMV with the activation read from global, no SLM (gemv_q8_soa_g_nd.cpp); IE_GEMV_SMALLK keeps v1's
// split-K kernel either way.
sycl::event gemv_q8_0_soa_q8_g_nd(sycl::queue& q,
                                  const void* x_q8, const int8_t* qs_W, const uint16_t* d_W,
                                  sycl::half* y,
                                  uint32_t K, uint32_t N,
                                  const std::vector<sycl::event>& deps = {});

// IE_Q8_SOA_GEMV_G_ND=0 = v1 gemv_q8_0_soa_q8_g. Read once.
inline bool gemv_q8_soa_g_nd_on() {
    static const bool v = [] { const char* e = std::getenv("IE_Q8_SOA_GEMV_G_ND"); return !(e && *e && std::string(e) == "0"); }();
    return v;
}

inline sycl::event gemv_q8_0_soa_q8_g_sel(sycl::queue& q,
                                          const void* x_q8, const int8_t* qs_W, const uint16_t* d_W,
                                          sycl::half* y,
                                          uint32_t K, uint32_t N,
                                          const std::vector<sycl::event>& deps = {}) {
    return gemv_q8_soa_g_nd_on() ? gemv_q8_0_soa_q8_g_nd(q, x_q8, qs_W, d_W, y, K, N, deps)
                                 : gemv_q8_0_soa_q8_g(q, x_q8, qs_W, d_W, y, K, N, deps);
}

}  // namespace ie
