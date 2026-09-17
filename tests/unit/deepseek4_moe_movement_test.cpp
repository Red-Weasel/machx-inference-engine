#include "ie/deepseek4_experts.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

// Frozen GPU oracle from the 2026-09-10 seven-more baseline. Arithmetic and
// flattened indexing are independent of the production row/column kernels.
namespace frozen {
sycl::event ds4_expert_gather_cast(sycl::queue& q, const float* x, const int32_t* rows,
                                   sycl::half* xp, uint32_t M, uint32_t H,
                                   const std::vector<sycl::event>& deps) {
    constexpr uint32_t WG = 256;
    const uint64_t total  = uint64_t(M) * H;
    const uint64_t global = ((total + WG - 1) / WG) * WG;
    return q.submit( [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(global, WG), [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= total) return;
            const uint32_t r = uint32_t(i / H);
            const uint32_t c = uint32_t(i % H);
            xp[i] = sycl::half(x[uint64_t(rows[r]) * H + c]);
        });
    });
}

// f32-source scatter — identical arithmetic and identical ASCENDING-kslot
// accumulation order to the fp16 one below, reading an f32 packed buffer.
// Exists so the XMX `down` route can hand its oneDNN f32 result straight to the
// scatter: without it that result had to be cast BACK to fp16 (a whole R*H
// pass) AND the int-dot route's quantize_q8_1 stayed in the chain, which is
// exactly what made the first XMX-down attempt a net LOSS (pp2048 184.7 ->
// 170.8, measured 2026-08-09).  Bit-comparable to the fp16 path only up to the
// fp16 rounding the fp16 path applies to `yp`; this one skips that rounding, so
// it is strictly the MORE accurate of the two.
sycl::event ds4_expert_scatter_accum_f32(sycl::queue& q, const float* yp,
                                         const int32_t* tk2p, const float* w_packed,
                                         float* y, uint32_t T, uint32_t top_k, uint32_t H,
                                         const std::vector<sycl::event>& deps) {
    constexpr uint32_t WG = 256;
    const uint64_t total  = uint64_t(T) * H;
    const uint64_t global = ((total + WG - 1) / WG) * WG;
    return q.submit( [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(global, WG), [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= total) return;
            const uint32_t t = uint32_t(i / H);
            const uint32_t c = uint32_t(i % H);
            float v = y[i];
            for (uint32_t k = 0; k < top_k; ++k) {
                const uint32_t p = uint32_t(tk2p[uint64_t(t) * top_k + k]);
                v += w_packed[p] * yp[uint64_t(p) * H + c];
            }
            y[i] = v;
        });
    });
}

sycl::event ds4_expert_scatter_accum(sycl::queue& q, const sycl::half* yp,
                                     const int32_t* tk2p, const float* w_packed,
                                     float* y, uint32_t T, uint32_t top_k, uint32_t H,
                                     const std::vector<sycl::event>& deps) {
    constexpr uint32_t WG = 256;
    const uint64_t total  = uint64_t(T) * H;
    const uint64_t global = ((total + WG - 1) / WG) * WG;
    return q.submit( [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(global, WG), [=](sycl::nd_item<1> it) {
            const uint64_t i = it.get_global_id(0);
            if (i >= total) return;
            const uint32_t t = uint32_t(i / H);
            const uint32_t c = uint32_t(i % H);
            // ASCENDING kslot: the same fp32 accumulation order the token-major
            // path's chain of `accum_f16` calls produced.  Read-modify-write of
            // one running float, not a tree reduction, for the same reason.
            float v = y[i];
            for (uint32_t k = 0; k < top_k; ++k) {
                const uint32_t p = uint32_t(tk2p[uint64_t(t) * top_k + k]);
                v += w_packed[p] * float(yp[uint64_t(p) * H + c]);
            }
            y[i] = v;
        });
    });
}


}
#ifdef MOE_PRIVATE
namespace candidate {
#ifndef MOE_CANDIDATE_FILE
#define MOE_CANDIDATE_FILE "candidate-2d.inc"
#endif
#include MOE_CANDIDATE_FILE
}
#else
namespace candidate = ie;
#endif

namespace {
struct DeviceMem {
    sycl::queue& q;
    std::vector<void*> ptrs;
    template<class T> T* get(size_t n) {
        auto p = sycl::malloc_device<T>(std::max(n, size_t(1)), q);
        if (!p) throw std::bad_alloc();
        ptrs.push_back(p);
        return p;
    }
    ~DeviceMem() { q.wait(); for (auto p : ptrs) sycl::free(p, q); }
};
template<class T> sycl::event copy_after(sycl::queue& q, T* dst, const T* src,
                                       size_t n, const std::vector<sycl::event>& deps) {
    return q.submit([&](sycl::handler& h) { h.depends_on(deps); h.memcpy(dst, src, n * sizeof(T)); });
}
void require(bool b, const char* message) { if (!b) throw std::runtime_error(message); }
float value(size_t i) { return float(int((i * 71 + 13) % 197) - 98) / 137.f; }

double duration(const sycl::event& e) {
    return (e.get_profiling_info<sycl::info::event_profiling::command_end>() -
            e.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000.;
}
template<class A, class B> void timing(const char* kind, unsigned rows, unsigned H,
                                      unsigned K, A baseline, B candidate) {
    std::vector<double> old, now;
    for (int i = 0; i < 72; ++i) {
        sycl::event a, b;
        if (i & 1) { b = candidate(); b.wait_and_throw(); a = baseline(); a.wait_and_throw(); }
        else { a = baseline(); a.wait_and_throw(); b = candidate(); b.wait_and_throw(); }
        if (i >= 12) { old.push_back(duration(a)); now.push_back(duration(b)); }
    }
    std::sort(old.begin(), old.end()); std::sort(now.begin(), now.end());
    std::printf("BENCH %s rows=%u H=%u K=%u frozen_us=%.3f candidate_us=%.3f speedup=%.4f\n",
                kind, rows, H, K, old[30], now[30], old[30]/now[30]);
}

void gather(sycl::queue& q, unsigned M, unsigned H, bool bench, bool special = false) {
    const unsigned R = std::max(1u, M / 2 + 3);
    const size_t n = size_t(M) * H, nx = size_t(R) * H, G = 19, shift = 3;
    std::vector<float> input(nx);
    for (size_t i = 0; i < nx; ++i) input[i] = value(i);
    if (special) {
        const float values[] = {0.f, -0.f, 0.00000001f, -0.00000001f, 65504.f, -65504.f, 1.00048828125f, 65536.f,
            std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::denorm_min()};
        for (size_t i = 0; i < nx; ++i) input[i] = values[i % std::size(values)];
    }
    std::vector<int32_t> rows(M);
    for (unsigned m = 0; m < M; ++m) rows[m] = (M - m + m / 3) % R;
    std::vector<sycl::half> got(n + 2 * G + shift), ref(got.size());
    DeviceMem mem{q, {}};
    auto xb = mem.get<float>(nx + 5), x = xb + 3;
    auto rb = mem.get<int32_t>(M + 4), r = rb + 1;
    auto out = mem.get<sycl::half>(got.size()), oracle = mem.get<sycl::half>(ref.size());
    std::vector<sycl::event> deps{q.memcpy(x, input.data(), nx * 4), q.memcpy(r, rows.data(), M * 4),
        q.fill(out, sycl::half(97), got.size()), q.fill(oracle, sycl::half(97), ref.size())};
    auto a = frozen::ds4_expert_gather_cast(q, x, r, oracle + G + shift, M, H, deps);
    auto b = candidate::ds4_expert_gather_cast(q, x, r, out + G + shift, M, H, deps);
    copy_after(q, ref.data(), oracle, ref.size(), {a}).wait_and_throw();
    copy_after(q, got.data(), out, got.size(), {b}).wait_and_throw();
    require(std::memcmp(got.data(), ref.data(), got.size() * 2) == 0, "gather differs from frozen GPU");
    for (size_t i = 0; i < got.size(); ++i) {
        if (i < G + shift || i >= G + shift + n) require(got[i] == sycl::half(97), "gather guard changed");
        else { size_t j = i - G - shift; sycl::half expected(input[size_t(rows[j / H]) * H + j % H]);
            require(std::isnan(float(expected)) ? std::isnan(float(got[i])) : std::memcmp(&expected, &got[i], 2) == 0,
                    "gather differs from CPU half conversion"); }
    }
    // Fresh dependencies on an out-of-order queue, with alternating producers
    // and reuse of the same input/output buffers. No host wait between launches.
    auto last = b;
    for (int iter = 0; iter < 128 && M && H; ++iter) {
        auto produce = q.submit([&](sycl::handler& h) { h.depends_on(last);
            h.parallel_for(sycl::range<1>(nx), [=](sycl::id<1> i) { x[i] = float(iter % 7) / 8.f; }); });
        last = candidate::ds4_expert_gather_cast(q, x, r, out + G + shift, M, H, {produce});
    }
    copy_after(q, got.data(), out, got.size(), {last}).wait_and_throw();
    if (M && H) for (size_t j = 0; j < n; ++j) require(got[G + shift + j] == sycl::half(1.f/8.f), "gather dependency/reuse mismatch");
    if (bench) timing("gather", M, H, 0,
        [&] { return frozen::ds4_expert_gather_cast(q, x, r, oracle + G + shift, M, H, {}); },
        [&] { return candidate::ds4_expert_gather_cast(q, x, r, out + G + shift, M, H, {}); });
}

template<class S> sycl::event scatter_call(bool baseline, sycl::queue& q, const S* src,
            const int32_t* map, const float* w, float* y, unsigned T, unsigned K,
            unsigned H, const std::vector<sycl::event>& deps) {
    if constexpr (std::is_same_v<S, float>) {
        if (baseline) return frozen::ds4_expert_scatter_accum_f32(q, src, map, w, y, T, K, H, deps);
        return candidate::ds4_expert_scatter_accum_f32(q, src, map, w, y, T, K, H, deps);
    } else {
        if (baseline) return frozen::ds4_expert_scatter_accum(q, src, map, w, y, T, K, H, deps);
        return candidate::ds4_expert_scatter_accum(q, src, map, w, y, T, K, H, deps);
    }
}
template<class S> void scatter(sycl::queue& q, unsigned T, unsigned H, unsigned K, bool bench) {
    const unsigned R = std::max(1u, T * std::max(K, 1u) + 7);
    const size_t n = size_t(T) * H, np = size_t(R) * H, G = 19, shift = 3;
    std::vector<S> src(np);
    std::vector<float> weights(R), initial(n), got(n + 2 * G + shift), ref(got.size());
    std::vector<int32_t> map(size_t(T) * K);
    for (size_t i = 0; i < np; ++i) src[i] = S(value(i + 97));
    for (unsigned r = 0; r < R; ++r) weights[r] = value(r + 173);
    for (size_t i = 0; i < map.size(); ++i) map[i] = (R - 1 - (i * 17 + i / 3) % R);
    for (size_t i = 0; i < n; ++i) initial[i] = value(i + 113);
    DeviceMem mem{q, {}};
    auto sb = mem.get<S>(np + 5), s = sb + 3;
    auto mb = mem.get<int32_t>(map.size() + 3), m = mb + 1;
    auto wb = mem.get<float>(R + 4), w = wb + 2;
    auto out = mem.get<float>(got.size()), oracle = mem.get<float>(ref.size());
    auto y = out + G + shift, oy = oracle + G + shift;
    auto fill = q.fill(out, 97.f, got.size()), ofill = q.fill(oracle, 97.f, ref.size());
    std::vector<sycl::event> deps{q.memcpy(s, src.data(), np * sizeof(S)),
        q.memcpy(m, map.data(), map.size() * 4), q.memcpy(w, weights.data(), R * 4),
        copy_after(q, y, initial.data(), n, {fill}), copy_after(q, oy, initial.data(), n, {ofill})};
    auto a = scatter_call(true, q, s, m, w, oy, T, K, H, deps);
    auto b = scatter_call(false, q, s, m, w, y, T, K, H, deps);
    copy_after(q, ref.data(), oracle, ref.size(), {a}).wait_and_throw();
    copy_after(q, got.data(), out, got.size(), {b}).wait_and_throw();
    require(std::memcmp(got.data(), ref.data(), got.size() * 4) == 0, "scatter differs from frozen GPU");
    for (size_t i = 0; i < got.size(); ++i) {
        if (i < G + shift || i >= G + shift + n) require(got[i] == 97.f, "scatter guard changed");
        else {
            size_t j = i - G - shift, t = j / H, c = j % H;
            double expected = initial[j], magnitude = std::fabs(expected);
            for (unsigned k = 0; k < K; ++k) { unsigned p = map[t * K + k];
                double term = double(weights[p]) * double(float(src[size_t(p) * H + c]));
                expected += term; magnitude += std::fabs(term); }
            require(std::fabs(double(got[i]) - expected) <= (K + 2) * 1.2e-7 * magnitude + 1e-7,
                    "scatter exceeds independent CPU numerical bound");
        }
    }
    // Dependency and read/modify/write stress with new source production each
    // iteration. Both implementations receive the same ordered chain.
    auto la = a, lb = b;
    for (int iter = 0; iter < 64 && T && H; ++iter) {
        auto produce = q.submit([&](sycl::handler& h) { h.depends_on({la, lb});
            h.parallel_for(sycl::range<1>(np), [=](sycl::id<1> i) { s[i] = S(float(iter % 7) / 8.f); }); });
        la = scatter_call(true, q, s, m, w, oy, T, K, H, {produce});
        lb = scatter_call(false, q, s, m, w, y, T, K, H, {produce});
    }
    copy_after(q, ref.data(), oracle, ref.size(), {la}).wait_and_throw();
    copy_after(q, got.data(), out, got.size(), {lb}).wait_and_throw();
    require(std::memcmp(got.data(), ref.data(), got.size() * 4) == 0, "scatter dependency/reuse mismatch");
    if (bench) timing(std::is_same_v<S,float> ? "scatter_f32" : "scatter_f16", T, H, K,
        [&] { return scatter_call(true, q, s, m, w, oy, T, K, H, {}); },
        [&] { return scatter_call(false, q, s, m, w, y, T, K, H, {}); });
}
}
// Empty launches must still return an event ordered after fresh dependencies.
// The marker has no connection to any kernel data pointer, so the only ordering
// edge between its producer and readback is the returned empty-work event.
void empty_dependencies(sycl::queue& q) {
    DeviceMem mem{q, {}};
    constexpr size_t N = 262144;
    auto marker = mem.get<uint32_t>(N);
    std::vector<uint32_t> got(N);
    for (unsigned variant = 0; variant < 6; ++variant) {
        auto producer = q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
            uint32_t v = uint32_t(i[0]) + variant;
            for (unsigned j = 0; j < 32; ++j) v = 1664525u * v + 1013904223u;
            marker[i] = v;
        });
        const unsigned rows = variant & 1 ? 3 : 0, width = variant & 1 ? 0 : 4096;
        sycl::event empty;
        if (variant < 2) empty = candidate::ds4_expert_gather_cast(q, nullptr, nullptr, nullptr, rows, width, {producer});
        else if (variant < 4) empty = candidate::ds4_expert_scatter_accum(q, nullptr, nullptr, nullptr, nullptr, rows, 8, width, {producer});
        else empty = candidate::ds4_expert_scatter_accum_f32(q, nullptr, nullptr, nullptr, nullptr, rows, 8, width, {producer});
        copy_after(q, got.data(), marker, N, {empty}).wait_and_throw();
        for (size_t i = 0; i < N; ++i) {
            uint32_t expected = uint32_t(i) + variant;
            for (unsigned j = 0; j < 32; ++j) expected = 1664525u * expected + 1013904223u;
            require(got[i] == expected, "empty movement event lost its dependency");
        }
    }
}
int main(int argc, char** argv) {
    try {
        const bool bench = argc > 1 && std::string(argv[1]) == "--bench";
        const std::string mode = argc > 2 ? argv[2] : "all";
        sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::enable_profiling{});
        std::printf("device=%s out_of_order=%d\n", q.get_device().get_info<sycl::info::device::name>().c_str(), !q.is_in_order());
        if (mode == "all" || mode == "gather") {
            for (unsigned M : {1u,16u,32u,64u,257u,512u,2048u}) for (unsigned H : {1024u,4096u}) gather(q,M,H,bench);
            for (unsigned H : {0u,1u,255u,257u,1025u,4097u}) gather(q,3,H,false);
            gather(q,3,1023,false); gather(q,512,4097,false); gather(q,0,4096,false); gather(q,33,257,false,true); gather(q,33,1025,false,true); gather(q,16384,4096,bench);
        }
        if (mode == "all" || mode == "scatter") {
            for (unsigned T : {1u,16u,32u,64u,257u,512u,2048u}) for (unsigned H : {1024u,4096u}) for (unsigned K : {1u,8u,16u}) {
                scatter<sycl::half>(q,T,H,K,bench); scatter<float>(q,T,H,K,bench);
            }
            for (unsigned H : {0u,1u,255u,257u,1025u,4097u}) {
                scatter<sycl::half>(q,3,H,8,false); scatter<float>(q,3,H,16,false);
            }
            scatter<sycl::half>(q,129,1025,8,false); scatter<float>(q,33,4097,16,false);
            scatter<sycl::half>(q,127,1024,8,false); scatter<float>(q,32,4095,16,false);
            scatter<sycl::half>(q,0,4096,8,false); scatter<float>(q,0,4096,8,false);
            scatter<sycl::half>(q,3,257,0,false); scatter<float>(q,3,257,0,false);
        }
        empty_dependencies(q);
        q.wait_and_throw();
        std::printf("PASS MoE movement %s frozen bits, CPU bounds, guards, shifts, empty work, OOO dependencies and repeated reuse\n", mode.c_str());
    } catch (const std::exception& e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}
