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

#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>

namespace ie {

namespace {

// $IE_ONEDNN_DETERMINISTIC (default 1): oneDNN's GPU matmul may otherwise pick
// kernels whose reduction order is not fixed (split-K with atomics), which
// makes a prefill non-reproducible run to run — docs/deepseek4/72 (the
// batch-path nondeterminism hunt).  Set to 0 to allow those kernels.
static bool onednn_deterministic() {
    static const bool v = [] {
        const char* e = std::getenv("IE_ONEDNN_DETERMINISTIC");
        return !(e && *e && std::string(e) == "0");
    }();
    return v;
}

struct CachedMatmul {
    dnnl::matmul prim;
    dnnl::memory a_mem, b_mem, y_mem;   // handles swapped per call
};
struct CachedS8 {
    bool             ok = false;      // false => no primitive; `why` says so
    std::string      impl;            // impl_info_str(), or the failure reason
    dnnl::matmul     prim;
    dnnl::memory     a_mem, b_mem, y_mem, s_mem;
};

struct OnednnCtx {
    dnnl::engine eng;
    dnnl::stream strm;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, CachedS8> s8_prims;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, CachedMatmul> prims;
    // Separate caches for the N-transposed entry points: their B operand is the
    // SAME (M,N,K) with a different memory layout, so sharing one map by shape
    // would hand back a primitive built for the wrong strides.
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, CachedMatmul> nt_prims;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, CachedMatmul> bmm_prims;
};

// Bind every stream to the actual caller queue, including its context. A
// device may have multiple in-order queues; a device-keyed cache would execute
// later GEMMs on the first queue and lose surrounding copy/scratch ordering.
// Thread-local caches also keep mutable primitive memory handles independent
// when different host threads submit to the same queue.
OnednnCtx& ctx_for(sycl::queue& q) {
    static thread_local std::unordered_map<sycl::queue, OnednnCtx> ctxs;
    auto it = ctxs.find(q);
    if (it == ctxs.end()) {
        OnednnCtx c;
        c.eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        c.strm = dnnl::sycl_interop::make_stream(c.eng, q);
        it = ctxs.emplace(q, std::move(c)).first;
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
        dnnl::primitive_attr attr;
        attr.set_deterministic(onednn_deterministic());
        dnnl::matmul::primitive_desc pd(ctx.eng, a_md, b_md, y_md, attr);
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

bool onednn_available() noexcept { return true; }

// y[M, N] fp32 = A[M, K] fp16 @ W[N, K]^T fp16.
//
// The one line that matters is b_md: dims {K, N} with strides {1, K}.  Element
// (k, n) of the logical B therefore sits at k*1 + n*K — which is exactly where
// row-major W[n][k] already is.  oneDNN reads the weight where it lies.
sycl::event gemm_nt_f16_onednn(sycl::queue& q,
                               const sycl::half* A, const sycl::half* W,
                               float* y,
                               uint32_t M, uint32_t N, uint32_t K,
                               const std::vector<sycl::event>& deps) {
    auto& ctx = ctx_for(q);

    const auto key = std::make_tuple(M, N, K);
    auto it = ctx.nt_prims.find(key);
    if (it == ctx.nt_prims.end()) {
        using dt  = dnnl::memory::data_type;
        using tag = dnnl::memory::format_tag;
        dnnl::memory::desc a_md({M, K}, dt::f16, tag::ab);
        dnnl::memory::desc b_md({K, N}, dt::f16, dnnl::memory::dims{1, K});
        dnnl::memory::desc y_md({M, N}, dt::f32, tag::ab);
        dnnl::primitive_attr attr;
        attr.set_deterministic(onednn_deterministic());
        dnnl::matmul::primitive_desc pd(ctx.eng, a_md, b_md, y_md, attr);
        CachedMatmul cm;
        cm.prim  = dnnl::matmul(pd);
        cm.a_mem = dnnl::memory(a_md, ctx.eng, nullptr);
        cm.b_mem = dnnl::memory(b_md, ctx.eng, nullptr);
        cm.y_mem = dnnl::memory(y_md, ctx.eng, nullptr);
        it = ctx.nt_prims.emplace(key, std::move(cm)).first;
    }

    auto& cm = it->second;
    cm.a_mem.set_data_handle(const_cast<sycl::half*>(A));
    cm.b_mem.set_data_handle(const_cast<sycl::half*>(W));
    cm.y_mem.set_data_handle(y);

    return dnnl::sycl_interop::execute(
        cm.prim, ctx.strm,
        {{DNNL_ARG_SRC, cm.a_mem}, {DNNL_ARG_WEIGHTS, cm.b_mem},
         {DNNL_ARG_DST, cm.y_mem}},
        deps);
}

// ---------------------------------------------------------------------------
// WEIGHT DECOMPRESSION — s8 weights, f16 source, per-(K-group, n) f16 scales
// ---------------------------------------------------------------------------
//
// Both operand descriptions below are the SAME "describe the buffer where it
// lies" trick `gemm_nt_f16_onednn` already uses, applied twice:
//
//   weights  logical {K, N} s8   strides {1, K}       -> row-major qs[n][k]
//   scales   logical {K/g, N} f16 strides {1, K/g}    -> row-major  d[n][kb]
//
// so neither plane is copied, reordered or transposed.  `set_scales` with
// mask (1<<0)|(1<<1) and groups {g, 1} is what tells oneDNN that scale (kb, n)
// covers weights[kb*g .. kb*g+g-1][n] — i.e. exactly one Q8_0 block of one row.
//
// `set_fpmath_mode(f16, apply_to_int = true)` is the part that makes this a
// DECOMPRESSION rather than an integer quantised matmul: it permits oneDNN to
// up-convert the int8 weights and run the math in f16 on XMX, which is the only
// way this reaches a jitted kernel.  Without `apply_to_int` the primitive is
// still built and still correct and is measurably useless.
namespace {



// S8 primitives share the queue/thread lifetime of their engine and stream.
CachedS8& s8_entry(sycl::queue& q, uint32_t M, uint32_t N, uint32_t K, uint32_t g) {
    OnednnCtx& ctx = ctx_for(q);
    auto& cache = ctx.s8_prims;
    const auto key = std::make_tuple(M, N, K, g);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    CachedS8 cm;
    if (K == 0 || g == 0 || (K % g) != 0) {
        cm.impl = "unsupported: group " + std::to_string(g) + " does not divide K " +
                  std::to_string(K);
        return cache.emplace(key, std::move(cm)).first->second;
    }
    try {
        using dt  = dnnl::memory::data_type;
        using tag = dnnl::memory::format_tag;
        const int64_t nb = int64_t(K) / int64_t(g);
        dnnl::memory::desc a_md({M, K}, dt::f16, tag::ab);
        dnnl::memory::desc b_md({K, N}, dt::s8,  dnnl::memory::dims{1, K});
        dnnl::memory::desc y_md({M, N}, dt::f32, tag::ab);
        // PLAIN kb-major [K/g][N].  oneDNN consumes attr scales in plain layout
        // and IGNORES custom strides on this md (measured 2026-08-08: an
        // n-major buffer described with {1, nb} strides was read kb-major
        // anyway — 125x the error bound, while a kb-major buffer passed).  The
        // caller must hand the kb-major plane (`Ds4Dense::q8dt`); declaring
        // tag::ab here makes the md say what actually happens.
        dnnl::memory::desc s_md({nb, N}, dt::f16, tag::ab);

        dnnl::primitive_attr attr;
        attr.set_fpmath_mode(dnnl::fpmath_mode::f16, /*apply_to_int=*/true);
        attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1),
                        dnnl::memory::dims{int64_t(g), 1}, dt::f16);
        attr.set_deterministic(onednn_deterministic());

        dnnl::matmul::primitive_desc pd(ctx.eng, a_md, b_md, y_md, attr);
        cm.impl  = pd.impl_info_str();
        cm.prim  = dnnl::matmul(pd);
        cm.a_mem = dnnl::memory(a_md, ctx.eng, nullptr);
        cm.b_mem = dnnl::memory(b_md, ctx.eng, nullptr);
        cm.y_mem = dnnl::memory(y_md, ctx.eng, nullptr);
        cm.s_mem = dnnl::memory(s_md, ctx.eng, nullptr);
        cm.ok    = true;
    } catch (const dnnl::error& e) {
        cm.ok   = false;
        cm.impl = std::string("unsupported: ") + e.what();
    } catch (const std::exception& e) {
        cm.ok   = false;
        cm.impl = std::string("unsupported: ") + e.what();
    }
    return cache.emplace(key, std::move(cm)).first->second;
}

}  // namespace

bool gemm_nt_s8_onednn(sycl::queue& q,
                       const sycl::half* A, const int8_t* qs, const sycl::half* d,
                       float* y,
                       uint32_t M, uint32_t N, uint32_t K, uint32_t group,
                       const std::vector<sycl::event>& deps,
                       sycl::event* out) {
    CachedS8& cm = s8_entry(q, M, N, K, group);
    if (!cm.ok || !out) return false;
    // The handles are swapped per call exactly as the f16 routes do.  This is
    // safe across queues and host threads because both own distinct caches.
    cm.a_mem.set_data_handle(const_cast<sycl::half*>(A));
    cm.b_mem.set_data_handle(const_cast<int8_t*>(qs));
    cm.y_mem.set_data_handle(y);
    cm.s_mem.set_data_handle(const_cast<sycl::half*>(d));
    *out = dnnl::sycl_interop::execute(
        cm.prim, ctx_for(q).strm,
        {{DNNL_ARG_SRC, cm.a_mem}, {DNNL_ARG_WEIGHTS, cm.b_mem}, {DNNL_ARG_DST, cm.y_mem},
         {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, cm.s_mem}},
        deps);
    return true;
}

std::string onednn_nt_s8_impl(sycl::queue& q, uint32_t M, uint32_t N, uint32_t K,
                              uint32_t group) {
    return s8_entry(q, M, N, K, group).impl;
}

// Block-diagonal, as one batched matmul.  Batch dim = G; every operand keeps its
// existing contiguous buffer and is addressed by strides:
//   A [M, G*IPG]   -> {G, M, IPG} strides {IPG,     G*IPG, 1}
//   W [G*OPG, IPG] -> {G, IPG, OPG} strides {OPG*IPG, 1,   IPG}   (transposed, in place)
//   y [M, G*OPG]   -> {G, M, OPG} strides {OPG,     G*OPG, 1}
sycl::event gemm_bmm_nt_f16_onednn(sycl::queue& q,
                                   const sycl::half* A, const sycl::half* W,
                                   float* y,
                                   uint32_t M, uint32_t G, uint32_t IPG, uint32_t OPG,
                                   const std::vector<sycl::event>& deps) {
    auto& ctx = ctx_for(q);

    const auto key = std::make_tuple(M, G, IPG, OPG);
    auto it = ctx.bmm_prims.find(key);
    if (it == ctx.bmm_prims.end()) {
        using dt = dnnl::memory::data_type;
        dnnl::memory::desc a_md({G, M, IPG}, dt::f16,
                                dnnl::memory::dims{IPG, int64_t(G) * IPG, 1});
        dnnl::memory::desc b_md({G, IPG, OPG}, dt::f16,
                                dnnl::memory::dims{int64_t(OPG) * IPG, 1, IPG});
        dnnl::memory::desc y_md({G, M, OPG}, dt::f32,
                                dnnl::memory::dims{OPG, int64_t(G) * OPG, 1});
        dnnl::primitive_attr attr;
        attr.set_deterministic(onednn_deterministic());
        dnnl::matmul::primitive_desc pd(ctx.eng, a_md, b_md, y_md, attr);
        CachedMatmul cm;
        cm.prim  = dnnl::matmul(pd);
        cm.a_mem = dnnl::memory(a_md, ctx.eng, nullptr);
        cm.b_mem = dnnl::memory(b_md, ctx.eng, nullptr);
        cm.y_mem = dnnl::memory(y_md, ctx.eng, nullptr);
        it = ctx.bmm_prims.emplace(key, std::move(cm)).first;
    }

    auto& cm = it->second;
    cm.a_mem.set_data_handle(const_cast<sycl::half*>(A));
    cm.b_mem.set_data_handle(const_cast<sycl::half*>(W));
    cm.y_mem.set_data_handle(y);

    return dnnl::sycl_interop::execute(
        cm.prim, ctx.strm,
        {{DNNL_ARG_SRC, cm.a_mem}, {DNNL_ARG_WEIGHTS, cm.b_mem},
         {DNNL_ARG_DST, cm.y_mem}},
        deps);
}

}  // namespace ie
