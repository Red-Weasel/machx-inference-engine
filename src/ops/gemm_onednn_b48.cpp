// src/ops/gemm_onednn_b48.cpp — P4 B48: the oneDNN products of the 27B split's prefill (include/ie/prefill_gemm.hpp).
//
// A file of its own: src/ops/gemm_onednn.cpp is an input of DeepSeek-V4.1's numerics key (its disk prompt cache), and
// nothing here changes what that file computes. The engine / stream pair is therefore this file's own; it wraps the
// caller's in-order queue exactly as gemm_onednn.cpp's does, so ordering with the surrounding kernels is the queue's.

#include "ie/prefill_gemm.hpp"

#include <sycl/sycl.hpp>

#include <dnnl.hpp>
#include <dnnl_sycl.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>

namespace ie {

namespace {

// IE_ONEDNN_DETERMINISTIC (default 1), as gemm_onednn.cpp: a fixed reduction order.
bool b48_deterministic() {
    static const bool v = [] {
        const char* e = std::getenv("IE_ONEDNN_DETERMINISTIC");
        return !(e && *e && std::string(e) == "0");
    }();
    return v;
}

struct B48Matmul {
    bool         ok = false;
    std::string  impl;
    dnnl::matmul prim;
    dnnl::memory a_mem, b_mem, y_mem, s_mem;
};

constexpr size_t kAttnShapes = 1024;   // rounded shapes (ie/prefill_gemm.hpp): ~2 a 512-key step of context

// IE_B48_PRIM_PROF=1: how many primitives this process built and how long that took, printed at exit.
struct B48PrimProf {
    const bool on = std::getenv("IE_B48_PRIM_PROF") != nullptr;
    double ms = 0; unsigned n = 0;
    ~B48PrimProf() { if (on) std::fprintf(stderr, "[b48] oneDNN primitives built: %u in %.1f ms\n", n, ms); }
};
B48PrimProf g_b48_prim;
struct B48PrimTimer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    ~B48PrimTimer() {
        g_b48_prim.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ++g_b48_prim.n;
    }
};

struct B48Ctx {
    dnnl::engine eng;
    dnnl::stream strm;
    std::map<std::tuple<uint32_t, uint32_t, uint32_t>, B48Matmul> qk, pv;              // bounded: kAttnShapes
    std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, B48Matmul> s8;        // the model's weight shapes
};

// Per caller queue and per host thread (gemm_onednn.cpp's rule: the stream is bound to the queue it was made for, and
// the mutable memory handles stay one thread's).
B48Ctx& b48_ctx(sycl::queue& q) {
    static thread_local std::unordered_map<sycl::queue, B48Ctx> ctxs;
    auto it = ctxs.find(q);
    if (it == ctxs.end()) {
        B48Ctx c;
        c.eng  = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        c.strm = dnnl::sycl_interop::make_stream(c.eng, q);
        it = ctxs.emplace(q, std::move(c)).first;
    }
    return it->second;
}

// nt: the second operand is [N, K] read in place as its transpose (strides {1, K}); else a plain [K, N].
sycl::event b48_attn_matmul(sycl::queue& q, bool nt, const sycl::half* A, const sycl::half* B, float* y,
                            uint32_t M, uint32_t N, uint32_t K, const std::vector<sycl::event>& deps) {
    B48Ctx& ctx = b48_ctx(q);
    auto& cache = nt ? ctx.qk : ctx.pv;
    const auto key = std::make_tuple(M, N, K);
    auto it = cache.find(key);
    if (it == cache.end()) {
        if (cache.size() >= kAttnShapes) cache.clear();
        B48PrimTimer timer;
        using dt  = dnnl::memory::data_type;
        using tag = dnnl::memory::format_tag;
        dnnl::memory::desc a_md({M, K}, dt::f16, tag::ab);
        dnnl::memory::desc b_md = nt ? dnnl::memory::desc({K, N}, dt::f16, dnnl::memory::dims{1, K})
                                     : dnnl::memory::desc({K, N}, dt::f16, tag::ab);
        dnnl::memory::desc y_md({M, N}, dt::f32, tag::ab);
        dnnl::primitive_attr attr;
        attr.set_deterministic(b48_deterministic());
        dnnl::matmul::primitive_desc pd(ctx.eng, a_md, b_md, y_md, attr);
        B48Matmul cm;
        cm.prim  = dnnl::matmul(pd);
        cm.a_mem = dnnl::memory(a_md, ctx.eng, nullptr);
        cm.b_mem = dnnl::memory(b_md, ctx.eng, nullptr);
        cm.y_mem = dnnl::memory(y_md, ctx.eng, nullptr);
        cm.ok    = true;
        it = cache.emplace(key, std::move(cm)).first;
    }
    B48Matmul& cm = it->second;
    cm.a_mem.set_data_handle(const_cast<sycl::half*>(A));
    cm.b_mem.set_data_handle(const_cast<sycl::half*>(B));
    cm.y_mem.set_data_handle(y);
    return dnnl::sycl_interop::execute(
        cm.prim, ctx.strm,
        {{DNNL_ARG_SRC, cm.a_mem}, {DNNL_ARG_WEIGHTS, cm.b_mem}, {DNNL_ARG_DST, cm.y_mem}},
        deps);
}

// gemm_onednn.cpp's s8_entry with an fp16 result; `ok` only for a jitted kernel.
B48Matmul& b48_s8_entry(sycl::queue& q, uint32_t M, uint32_t N, uint32_t K, uint32_t g) {
    B48Ctx& ctx = b48_ctx(q);
    const auto key = std::make_tuple(M, N, K, g);
    auto it = ctx.s8.find(key);
    if (it != ctx.s8.end()) return it->second;

    B48Matmul cm;
    if (K == 0 || g == 0 || (K % g) != 0) {
        cm.impl = "unsupported: group " + std::to_string(g) + " does not divide K " + std::to_string(K);
        return ctx.s8.emplace(key, std::move(cm)).first->second;
    }
    B48PrimTimer timer;
    try {
        using dt  = dnnl::memory::data_type;
        using tag = dnnl::memory::format_tag;
        const int64_t nb = int64_t(K) / int64_t(g);
        dnnl::memory::desc a_md({M, K}, dt::f16, tag::ab);
        dnnl::memory::desc b_md({K, N}, dt::s8,  dnnl::memory::dims{1, K});
        dnnl::memory::desc y_md({M, N}, dt::f16, tag::ab);
        dnnl::memory::desc s_md({nb, N}, dt::f16, tag::ab);   // plain kb-major (see gemm_onednn.cpp)

        dnnl::primitive_attr attr;
        attr.set_fpmath_mode(dnnl::fpmath_mode::f16, /*apply_to_int=*/true);
        attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), dnnl::memory::dims{int64_t(g), 1}, dt::f16);
        attr.set_deterministic(b48_deterministic());

        dnnl::matmul::primitive_desc pd(ctx.eng, a_md, b_md, y_md, attr);
        cm.impl  = pd.impl_info_str();
        cm.ok    = cm.impl.find("jit") != std::string::npos;
        if (cm.ok) {
            cm.prim  = dnnl::matmul(pd);
            cm.a_mem = dnnl::memory(a_md, ctx.eng, nullptr);
            cm.b_mem = dnnl::memory(b_md, ctx.eng, nullptr);
            cm.y_mem = dnnl::memory(y_md, ctx.eng, nullptr);
            cm.s_mem = dnnl::memory(s_md, ctx.eng, nullptr);
        }
    } catch (const std::exception& e) {
        cm.ok   = false;
        cm.impl = std::string("unsupported: ") + e.what();
    }
    return ctx.s8.emplace(key, std::move(cm)).first->second;
}

}  // namespace

sycl::event gemm_attn_qk_onednn(sycl::queue& q, const sycl::half* Q, const sycl::half* Kc, float* S,
                                uint32_t M, uint32_t N, uint32_t K, const std::vector<sycl::event>& deps) {
    return b48_attn_matmul(q, true, Q, Kc, S, M, N, K, deps);
}

sycl::event gemm_attn_pv_onednn(sycl::queue& q, const sycl::half* P, const sycl::half* V, float* O,
                                uint32_t M, uint32_t N, uint32_t K, const std::vector<sycl::event>& deps) {
    return b48_attn_matmul(q, false, P, V, O, M, N, K, deps);
}

bool gemm_nt_s8_f16_onednn(sycl::queue& q, const sycl::half* A, const int8_t* qs, const sycl::half* dt,
                           sycl::half* y, uint32_t M, uint32_t N, uint32_t K, uint32_t group, sycl::event* out) {
    B48Matmul& cm = b48_s8_entry(q, M, N, K, group);
    if (!cm.ok || !out) return false;
    cm.a_mem.set_data_handle(const_cast<sycl::half*>(A));
    cm.b_mem.set_data_handle(const_cast<int8_t*>(qs));
    cm.y_mem.set_data_handle(y);
    cm.s_mem.set_data_handle(const_cast<sycl::half*>(dt));
    *out = dnnl::sycl_interop::execute(
        cm.prim, b48_ctx(q).strm,
        {{DNNL_ARG_SRC, cm.a_mem}, {DNNL_ARG_WEIGHTS, cm.b_mem}, {DNNL_ARG_DST, cm.y_mem},
         {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, cm.s_mem}});
    return true;
}

std::string onednn_nt_s8_f16_impl(sycl::queue& q, uint32_t M, uint32_t N, uint32_t K, uint32_t group) {
    return b48_s8_entry(q, M, N, K, group).impl;
}

}  // namespace ie
