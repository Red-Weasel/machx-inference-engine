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
#include <cstdio>
#include <cstdlib>

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

// The DeepSeek-V4 dense route has its OWN kernel and treats oneDNN purely as a
// faster path, so it must ask before calling — and this build says no.
bool onednn_available() noexcept { return false; }

// Unreachable when callers honour onednn_available().  They abort rather than
// return an empty event: silently leaving `y` untouched would look like a
// working forward pass producing garbage logits, which is precisely the failure
// mode that is hardest to attribute.
namespace {
[[noreturn]] void no_onednn(const char* fn) {
    std::fprintf(stderr,
                 "ie: %s called in a build without oneDNN. Callers must branch on "
                 "ie::onednn_available(); this build has no implementation.\n", fn);
    std::abort();
}
}  // namespace

sycl::event gemm_nt_f16_onednn(sycl::queue&, const sycl::half*, const sycl::half*, float*,
                               uint32_t, uint32_t, uint32_t,
                               const std::vector<sycl::event>&) {
    no_onednn("gemm_nt_f16_onednn");
}

sycl::event gemm_bmm_nt_f16_onednn(sycl::queue&, const sycl::half*, const sycl::half*, float*,
                                   uint32_t, uint32_t, uint32_t, uint32_t,
                                   const std::vector<sycl::event>&) {
    no_onednn("gemm_bmm_nt_f16_onednn");
}

// The weight-decompression route reports "no primitive" rather than aborting:
// its contract is already "returns false and submits nothing when oneDNN cannot
// build this shape", and a build with no oneDNN at all is that same answer.
// Every caller therefore keeps its existing route without a second #if.
bool gemm_nt_s8_onednn(sycl::queue&, const sycl::half*, const int8_t*, const sycl::half*,
                       float*, uint32_t, uint32_t, uint32_t, uint32_t,
                       const std::vector<sycl::event>&, sycl::event*) {
    return false;
}

std::string onednn_nt_s8_impl(sycl::queue&, uint32_t, uint32_t, uint32_t, uint32_t) {
    return "unsupported: build has no oneDNN";
}

}  // namespace ie
