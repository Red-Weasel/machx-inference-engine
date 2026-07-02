// src/ops/gemm_onednn_stub.cpp — fallback for `gemm_fp16_onednn` when oneDNN
// (Intel DNNL) is compiled out (`-DIE_ENABLE_ONEDNN=OFF`, e.g. a CUDA/non-Intel
// build). Same signature as the real oneDNN path in gemm_onednn.cpp, so NO call
// site changes (the crown's qwen36.cpp and dense_dispatch.hpp keep calling
// `gemm_fp16_onednn` unchanged — they just link this stub instead).
//
// Correctness-only: routes through the in-house `gemm_fp16` (f16 in, f32
// accumulate) + `cast_fp32_to_fp16`. The oneDNN matmul is an Intel-XMX perf
// lever and is runtime-opt-in everywhere (default path is `gemm_fp16`), so a
// build without it is functionally identical — only the qwen35-27B prefill
// loses its ~1.65× oneDNN speedup and falls back to gemm_fp16.
#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>

namespace ie {

sycl::event gemm_fp16_onednn(sycl::queue& q,
                             const sycl::half* A, const sycl::half* B,
                             sycl::half* y,
                             uint32_t M, uint32_t N, uint32_t K,
                             const std::vector<sycl::event>& deps) {
    const uint64_t mn = uint64_t(M) * N;
    float* c_scr = sycl::malloc_device<float>(mn, q);
    auto e = gemm_fp16(q, A, B, c_scr, M, N, K, deps);
    e = cast_fp32_to_fp16(q, c_scr, y, mn, {e});
    e.wait();                       // fallback path: simplest correct lifetime
    sycl::free(c_scr, q);
    return e;
}

}  // namespace ie
