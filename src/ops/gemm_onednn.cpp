// src/ops/gemm_onednn.cpp — oneDNN matmul for the E1 prefill GEMM.  P1b.
//
// Why: llama.cpp SYCL master's prefill lead (1088 vs 955 pp512 on B70) comes
// from running its dequant-to-fp16 GEMMs through oneDNN's tuned kernels;
// our own gemm_fp16 sustains 33.5 TFLOPS (18% of peak).  oneDNN has been in
// this project's locked tech stack as the "production GEMM fallback" since
// Phase 0 — this wires it in where it counts.
//
// Bonus vs the gemm_fp16 pipeline: fp16 output directly, eliminating the
// fp32 C scratch round-trip AND the cast_fp32_to_fp16 launch per projection.
//
// Primitive + memory-desc creation is cached per (M, N, K) — decode/prefill
// shapes recur every layer/step.  The dnnl engine/stream wrap the caller's
// in-order SYCL queue, so ordering with surrounding kernels is automatic.

#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <dnnl.hpp>
#include <dnnl_sycl.hpp>

#include <map>
#include <mutex>
#include <tuple>
#include <unordered_map>

namespace ie {

namespace {

struct CachedMatmul {
    dnnl::matmul prim;
    dnnl::memory a_mem, b_mem, y_mem;   // handles swapped per call
};
struct OnednnCtx {
    dnnl::engine eng;
    dnnl::stream strm;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, CachedMatmul> prims;
};

// One oneDNN engine/stream (+ its per-(M,N,K) primitive cache) PER SYCL device,
// keyed by the caller's queue device. This is what makes oneDNN safe on the
// multi-card layer-split paths (27B/80B): each card builds and uses ITS OWN
// engine, so a card-1 GEMM never routes its USM through a card-0-bound stream —
// the old static singleton did exactly that → UR_RESULT_ERROR_DEVICE_LOST, which
// is why oneDNN was force-disabled on every multi-card path.
//
// Correctness rests on one invariant this codebase already upholds: each device
// owns exactly ONE persistent SYCL queue (DeviceAllocator::queue_), so a device
// maps to a single (context, queue) for the whole run. The engine is therefore
// built once from that device's context and the stream wraps that one queue;
// reusing them on every call for the same device is exact. unordered_map node
// storage keeps the returned reference (and the per-device prim cache) stable as
// new devices are inserted. Single-GPU is unchanged: one device → one entry →
// byte-identical to the prior singleton. Host is single-threaded (serial model
// loads), so the lazy insert needs no lock.
OnednnCtx& ctx_for(sycl::queue& q) {
    static std::unordered_map<sycl::device, OnednnCtx> ctxs;
    const sycl::device dev = q.get_device();
    auto it = ctxs.find(dev);
    if (it == ctxs.end()) {
        OnednnCtx c;
        c.eng  = dnnl::sycl_interop::make_engine(dev, q.get_context());
        c.strm = dnnl::sycl_interop::make_stream(c.eng, q);
        it = ctxs.emplace(dev, std::move(c)).first;
    }
    return it->second;
}

}  // namespace

// y[M, N] (fp16) = A[M, K] (fp16) @ B[K, N] (fp16), fp32 accumulate inside
// oneDNN (default for f16 matmul on XMX hardware).
sycl::event gemm_fp16_onednn(sycl::queue& q,
                             const sycl::half* A, const sycl::half* B,
                             sycl::half* y,
                             uint32_t M, uint32_t N, uint32_t K,
                             const std::vector<sycl::event>& deps) {
    auto& ctx = ctx_for(q);

    const auto key = std::make_tuple(M, N, K);
    auto it = ctx.prims.find(key);
    if (it == ctx.prims.end()) {
        using dt  = dnnl::memory::data_type;
        using tag = dnnl::memory::format_tag;
        dnnl::memory::desc a_md({M, K}, dt::f16, tag::ab);
        dnnl::memory::desc b_md({K, N}, dt::f16, tag::ab);
        dnnl::memory::desc y_md({M, N}, dt::f16, tag::ab);
        dnnl::matmul::primitive_desc pd(ctx.eng, a_md, b_md, y_md);
        CachedMatmul cm;
        cm.prim  = dnnl::matmul(pd);
        cm.a_mem = dnnl::memory(a_md, ctx.eng, nullptr);
        cm.b_mem = dnnl::memory(b_md, ctx.eng, nullptr);
        cm.y_mem = dnnl::memory(y_md, ctx.eng, nullptr);
        it = ctx.prims.emplace(key, std::move(cm)).first;
    }

    // Swap USM handles only — wrapper construction is host-side overhead
    // at 250 calls per prefill pass.
    auto& cm = it->second;
    cm.a_mem.set_data_handle(const_cast<sycl::half*>(A));
    cm.b_mem.set_data_handle(const_cast<sycl::half*>(B));
    cm.y_mem.set_data_handle(y);

    return dnnl::sycl_interop::execute(
        cm.prim, ctx.strm,
        {{DNNL_ARG_SRC, cm.a_mem}, {DNNL_ARG_WEIGHTS, cm.b_mem},
         {DNNL_ARG_DST, cm.y_mem}},
        deps);
}

}  // namespace ie
