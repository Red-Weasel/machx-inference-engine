// src/ops/qwen4_route.cpp — see include/ie/qwen4_route.hpp. Built -fp-model=precise (src/CMakeLists.txt) and with
// contraction off below: the host twin and the kernel run the same float operations in the same order.
#include "ie/qwen4_route.hpp"

#include "ie/kernel_profiler.hpp"
#include "ie/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#pragma clang fp contract(off)

namespace ie {
namespace {

inline float q4r_fma(float a, float b, float c) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::fma(a, b, c);
#else
    return std::fma(a, b, c);
#endif
}
inline float q4r_rint(float a) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::rint(a);
#else
    return std::rint(a);
#endif
}

// exp(x) for x <= 0. x < -80 -> 0 (a weight below 1e-34 of the top pick's); otherwise n = rint(x / ln2) in [-116, 0],
// r = x - n ln2 (Cody-Waite, two fmas), p = Taylor degree 6 on |r| <= ln2/2 in Horner fma form, times 2^n built from the
// exponent bits (a normal number: no denormal handling can differ between host and device).
inline float q4r_exp(float x) {
    if (!(x >= -80.0f)) return 0.0f;
    const float n = q4r_rint(x * 1.44269504088896341f);
    float r = q4r_fma(-n, 0.693145751953125f, x);
    r = q4r_fma(-n, 1.42860682030941723212e-6f, r);
    float p = 1.0f / 720.0f;
    p = q4r_fma(p, r, 1.0f / 120.0f);
    p = q4r_fma(p, r, 1.0f / 24.0f);
    p = q4r_fma(p, r, 1.0f / 6.0f);
    p = q4r_fma(p, r, 0.5f);
    p = q4r_fma(p, r, 1.0f);
    p = q4r_fma(p, r, 1.0f);
    const int32_t ni = int32_t(n);
    const uint32_t sbits = uint32_t(ni + 127) << 23;
    float s;
#ifdef __SYCL_DEVICE_ONLY__
    s = sycl::bit_cast<float>(sbits);
#else
    std::memcpy(&s, &sbits, 4);
#endif
    return p * s;
}

// a / b for b in [1, k] (the weights' sum: the top pick's term is exactly 1): the device's float division is not correctly
// rounded, so neither side uses it -- a reciprocal by Newton from a fixed bit-trick seed (4 steps: ~4 % -> far below 1 ulp),
// then one fma correction of the quotient. fma and mul only: the same bits on host and device.
inline float q4r_div(float a, float b) {
    uint32_t bi;
#ifdef __SYCL_DEVICE_ONLY__
    bi = sycl::bit_cast<uint32_t>(b);
#else
    std::memcpy(&bi, &b, 4);
#endif
    const uint32_t yi = 0x7EF311C7u - bi;
    float y;
#ifdef __SYCL_DEVICE_ONLY__
    y = sycl::bit_cast<float>(yi);
#else
    std::memcpy(&y, &yi, 4);
#endif
    for (int i = 0; i < 4; ++i) y = q4r_fma(y, q4r_fma(-b, y, 1.0f), y);
    float qt = a * y;
    const float r = q4r_fma(-qt, b, a);
    qt = q4r_fma(r, y, qt);
    return qt;
}

// An order key: larger logit -> larger key; equal logits -> the lower id larger.
inline uint64_t q4r_key(float f, uint32_t e) {
    uint32_t u;
#ifdef __SYCL_DEVICE_ONLY__
    u = sycl::bit_cast<uint32_t>(f);
#else
    std::memcpy(&u, &f, 4);
#endif
    u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (uint64_t(u) << 32) | uint64_t(0xFFFFFFFFu - e);
}

// The weights of k picks already in ascending id: ids[], their logits lg[], the top logit lmax.
inline void q4r_weights(const float* lg, uint32_t k, float lmax, float* w) {
    float e[kQwen4RouteMaxK];
    float sum = 0.0f;
    for (uint32_t j = 0; j < k; ++j) {
        e[j] = q4r_exp(lg[j] - lmax);
        sum = sum + e[j];
    }
    for (uint32_t j = 0; j < k; ++j) w[j] = q4r_div(e[j], sum);
}

}  // namespace

void qwen4_route_host(const sycl::half* logits, uint32_t E, uint32_t k, int32_t* ids, float* w) {
    uint64_t keys[kQwen4RouteMaxE];
    for (uint32_t e = 0; e < E; ++e) keys[e] = q4r_key(float(logits[e]), e);
    std::partial_sort(keys, keys + k, keys + E, [](uint64_t a, uint64_t b) { return a > b; });
    const float lmax = float(logits[0xFFFFFFFFu - uint32_t(keys[0] & 0xFFFFFFFFu)]);
    uint32_t pick[kQwen4RouteMaxK];
    for (uint32_t j = 0; j < k; ++j) pick[j] = 0xFFFFFFFFu - uint32_t(keys[j] & 0xFFFFFFFFu);
    std::sort(pick, pick + k);
    float lg[kQwen4RouteMaxK];
    for (uint32_t j = 0; j < k; ++j) { ids[j] = int32_t(pick[j]); lg[j] = float(logits[pick[j]]); }
    q4r_weights(lg, k, lmax, w);
}

namespace {

// One work-group (WG lanes) a row: the k winners by repeated max-reduce over the order keys, then lane 0 sorts them by id,
// forms the weights and hands (row, ids[], w[]) to out -- the two entry points below differ only in what they write.
template <class Out>
sycl::event route_launch(sycl::queue& q, const char* tag, const sycl::half* logits, uint32_t rows, uint32_t E, uint32_t k,
                         Out out, const std::vector<sycl::event>& deps) {
    constexpr uint32_t WG = 256;
    return ie::ps(q, tag, [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<uint32_t, 1> pk(kQwen4RouteMaxK, h);
        h.parallel_for(sycl::nd_range<1>(uint64_t(rows) * WG, WG), [=](sycl::nd_item<1> it) {
            const uint32_t row = uint32_t(it.get_group(0));
            const uint32_t lid = uint32_t(it.get_local_id(0));
            auto grp = it.get_group();
            const sycl::half* lr = logits + uint64_t(row) * E;
            // each lane owns experts lid, lid + WG, ... (E <= 1024: up to 4); a winner's owner marks it dead
            uint64_t key[kQwen4RouteMaxE / WG];
            #pragma unroll
            for (uint32_t s = 0; s < kQwen4RouteMaxE / WG; ++s) {
                const uint32_t e = lid + s * WG;
                key[s] = e < E ? q4r_key(float(lr[e]), e) : 0ull;
            }
            for (uint32_t j = 0; j < k; ++j) {
                uint64_t best = 0ull;
                #pragma unroll
                for (uint32_t s = 0; s < kQwen4RouteMaxE / WG; ++s) best = key[s] > best ? key[s] : best;
                const uint64_t win = sycl::reduce_over_group(grp, best, sycl::maximum<uint64_t>());
                const uint32_t we = 0xFFFFFFFFu - uint32_t(win & 0xFFFFFFFFu);
                #pragma unroll
                for (uint32_t s = 0; s < kQwen4RouteMaxE / WG; ++s)
                    if (key[s] == win) key[s] = 0ull;
                if (lid == 0) pk[j] = we;
            }
            sycl::group_barrier(grp);
            if (lid != 0) return;
            uint32_t pick[kQwen4RouteMaxK];
            for (uint32_t j = 0; j < k; ++j) pick[j] = pk[j];
            const float lmax = float(lr[pick[0]]);   // the first winner is the largest logit
            for (uint32_t i = 1; i < k; ++i)          // insertion sort, ascending id
                for (uint32_t j = i; j > 0 && pick[j - 1] > pick[j]; --j) {
                    const uint32_t t = pick[j]; pick[j] = pick[j - 1]; pick[j - 1] = t;
                }
            float lg[kQwen4RouteMaxK], wv[kQwen4RouteMaxK];
            for (uint32_t j = 0; j < k; ++j) lg[j] = float(lr[pick[j]]);
            q4r_weights(lg, k, lmax, wv);
            out(row, pick, wv);
        });
    });
}

}  // namespace

sycl::event qwen4_route_dev(sycl::queue& q, const sycl::half* logits, uint32_t rows, uint32_t E, uint32_t k,
                            int32_t* ids, float* w, const std::vector<sycl::event>& deps) {
    return route_launch(q, "q4_route", logits, rows, E, k,
        [=](uint32_t row, const uint32_t* pick, const float* wv) {
            for (uint32_t j = 0; j < k; ++j) {
                ids[uint64_t(row) * k + j] = int32_t(pick[j]);
                w[uint64_t(row) * k + j] = wv[j];
            }
        }, deps);
}

sycl::event qwen4_route_jobs_dev(sycl::queue& q, const sycl::half* logits, uint32_t T, uint32_t E, uint32_t k,
                                 const int16_t* slot_of, uint32_t S, int32_t* jobs, uint32_t cnt_off, uint32_t w_off,
                                 int32_t* ids_out, const std::vector<sycl::event>& deps) {
    return route_launch(q, "q4_route_jobs", logits, T, E, k,
        [=](uint32_t vt, const uint32_t* pick, const float* wv) {
            float* jw = reinterpret_cast<float*>(jobs + w_off);
            for (uint32_t j = 0; j < k; ++j) {
                const uint32_t p = vt * k + j, e = pick[j];
                const int32_t sl = slot_of[e];
                jobs[3 * p + 0] = int32_t(vt);
                jobs[3 * p + 1] = sl >= 0 ? sl : int32_t(S + e);
                jobs[3 * p + 2] = int32_t(p);
                jw[p] = wv[j];
                ids_out[p] = int32_t(e);
            }
            jobs[cnt_off + vt] = int32_t(k);
        }, deps);
}

}  // namespace ie
