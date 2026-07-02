// tests/unit/attention_test.cpp — kernel-level correctness for full_attention.
//
// We can't (yet) diff against HF transformers — Phase 8 wires that up. For
// now, validate the kernel against algebraic identities and a scalar host
// reference run on the same inputs.

#include "ie/kv_cache.hpp"
#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

namespace {

constexpr const char* G = "\033[32m";
constexpr const char* R = "\033[31m";
constexpr const char* Z = "\033[0m";

bool report(const char* name, bool ok, const char* extra = "") {
    std::printf("  %s%-44s%s %s%s\n", ok ? G : R, name, Z, ok ? "OK" : "FAIL", extra);
    return ok;
}

bool close(float a, float b, float tol = 1e-2f) {
    const float scale = std::fmax(std::fabs(a), std::fabs(b));
    if (scale < 1e-4f) return std::fabs(a - b) <= tol;
    return std::fabs(a - b) / scale <= tol;
}

sycl::queue make_queue() {
    sycl::device dev;
    bool found = false;
    for (const auto& d : sycl::device::get_devices()) {
        if (d.is_gpu() && d.get_info<sycl::info::device::name>().find("0xe223") != std::string::npos) {
            dev = d; found = true; break;
        }
    }
    if (!found) for (const auto& d : sycl::device::get_devices()) if (d.is_gpu()) { dev = d; found = true; break; }
    if (!found) { std::fputs("no GPU\n", stderr); std::exit(2); }
    return sycl::queue(dev, sycl::property::queue::enable_profiling{});
}

// Scalar reference attention for comparison.
//   q [T, n_q, d]
//   k_full [n_kv, max_ctx, d] — caller fills [0..start_pos+T)
//   v_full [n_kv, max_ctx, d]
//   y [T, n_q, d]
void attention_ref(uint32_t T, uint32_t n_q, uint32_t n_kv, uint32_t d, uint32_t max_ctx,
                   uint32_t start_pos,
                   const std::vector<float>& q,
                   const std::vector<float>& k,
                   const std::vector<float>& v,
                   std::vector<float>& y)
{
    const float scale = 1.0f / std::sqrt(float(d));
    const uint32_t gqa = n_q / n_kv;
    for (uint32_t t = 0; t < T; ++t) {
        const uint32_t pos_t = start_pos + t;
        const uint32_t ctx_len = pos_t + 1;
        for (uint32_t h = 0; h < n_q; ++h) {
            const uint32_t kv_h = h / gqa;
            // scores
            std::vector<float> s(ctx_len);
            float m = -std::numeric_limits<float>::infinity();
            for (uint32_t i = 0; i < ctx_len; ++i) {
                float dot = 0.f;
                for (uint32_t dd = 0; dd < d; ++dd) {
                    const float qv = q[(t * n_q + h) * d + dd];
                    const float kv = k[(kv_h * max_ctx + i) * d + dd];
                    dot += qv * kv;
                }
                s[i] = dot * scale;
                m = std::fmax(m, s[i]);
            }
            float l = 0.f;
            for (auto& x : s) { x = std::exp(x - m); l += x; }
            for (auto& x : s) x /= l;
            for (uint32_t dd = 0; dd < d; ++dd) {
                float acc = 0.f;
                for (uint32_t i = 0; i < ctx_len; ++i) {
                    const float vv = v[(kv_h * max_ctx + i) * d + dd];
                    acc += s[i] * vv;
                }
                y[(t * n_q + h) * d + dd] = acc;
            }
        }
    }
}

// Helper: write a [n_kv, max_ctx, d] cache slot from a host vector.
void upload_cache(sycl::queue& q, sycl::half* dev_cache,
                  const std::vector<float>& host, uint32_t total) {
    std::vector<sycl::half> hv(total);
    for (uint32_t i = 0; i < total; ++i) hv[i] = sycl::half(host[i]);
    q.memcpy(dev_cache, hv.data(), total * sizeof(sycl::half)).wait();
}

// --------------------------------------------------------------
// Test 1: Constant V → output equals V's constant (regardless of Q, K).
// --------------------------------------------------------------
bool test_constant_v(sycl::queue& q) {
    const uint32_t T = 1, n_q = 16, n_kv = 2, d = 64, max_ctx = 32;
    const uint32_t start_pos = 5;     // ctx_len for the one new token = 6
    std::mt19937 rng(0xA1);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> hq(T * n_q * d);
    std::vector<float> hk(n_kv * max_ctx * d, 0.f);
    std::vector<float> hv(n_kv * max_ctx * d);
    std::vector<float> hk_in(T * n_kv * d, 0.f);
    std::vector<float> hv_in(T * n_kv * d);
    for (auto& x : hq) x = dist(rng);
    for (auto& x : hk) x = dist(rng);
    for (auto& x : hk_in) x = dist(rng);
    // V everywhere = constant 1.7
    const float kConst = 1.7f;
    std::fill(hv.begin(), hv.end(), kConst);
    std::fill(hv_in.begin(), hv_in.end(), kConst);

    auto* dq = sycl::malloc_device<sycl::half>(T * n_q * d, q);
    auto* dk_in = sycl::malloc_device<sycl::half>(T * n_kv * d, q);
    auto* dv_in = sycl::malloc_device<sycl::half>(T * n_kv * d, q);
    auto* dk = sycl::malloc_device<sycl::half>(n_kv * max_ctx * d, q);
    auto* dv = sycl::malloc_device<sycl::half>(n_kv * max_ctx * d, q);
    auto* dy = sycl::malloc_device<sycl::half>(T * n_q * d, q);

    auto upload = [&](sycl::half* p, const std::vector<float>& src) {
        std::vector<sycl::half> tmp(src.size());
        for (size_t i = 0; i < src.size(); ++i) tmp[i] = sycl::half(src[i]);
        q.memcpy(p, tmp.data(), tmp.size() * sizeof(sycl::half)).wait();
    };
    upload(dq, hq);
    upload(dk_in, hk_in);
    upload(dv_in, hv_in);
    upload(dk, hk);
    upload(dv, hv);

    ie::full_attention(q, dq, dk_in, dv_in, dk, dv, dy,
                       T, start_pos, n_q, n_kv, d, max_ctx).wait();

    std::vector<sycl::half> hy(T * n_q * d);
    q.memcpy(hy.data(), dy, hy.size() * sizeof(sycl::half)).wait();
    sycl::free(dq, q); sycl::free(dk_in, q); sycl::free(dv_in, q);
    sycl::free(dk, q); sycl::free(dv, q); sycl::free(dy, q);

    bool ok = true;
    char extra[128] = {};
    for (uint32_t i = 0; i < hy.size(); ++i) {
        if (!close(float(hy[i]), kConst, 1e-2f)) {
            ok = false;
            std::snprintf(extra, sizeof(extra), " (i=%u got=%.4f)", i, float(hy[i]));
            break;
        }
    }
    return report("constant-V → output equals const", ok, extra);
}

// --------------------------------------------------------------
// Test 2: Causality — garbage at cache positions > start_pos+T-1 must not
// affect the output for the new token.
// --------------------------------------------------------------
bool test_causality(sycl::queue& q) {
    const uint32_t T = 1, n_q = 8, n_kv = 2, d = 32, max_ctx = 64;
    const uint32_t start_pos = 7;
    std::mt19937 rng(0xC0);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> hq(T * n_q * d), hk_in(T * n_kv * d), hv_in(T * n_kv * d);
    std::vector<float> hk(n_kv * max_ctx * d, 0.f), hv(n_kv * max_ctx * d, 0.f);
    for (auto& x : hq) x = dist(rng);
    for (auto& x : hk_in) x = dist(rng);
    for (auto& x : hv_in) x = dist(rng);
    // Real KV at positions 0..start_pos-1
    for (uint32_t i = 0; i < start_pos; ++i)
        for (uint32_t kv = 0; kv < n_kv; ++kv)
            for (uint32_t dd = 0; dd < d; ++dd) {
                hk[(kv * max_ctx + i) * d + dd] = dist(rng);
                hv[(kv * max_ctx + i) * d + dd] = dist(rng);
            }
    // Tape A: zero garbage past start_pos
    auto hk_a = hk;
    auto hv_a = hv;
    // Tape B: random garbage past start_pos
    auto hk_b = hk;
    auto hv_b = hv;
    for (uint32_t i = start_pos + T; i < max_ctx; ++i)
        for (uint32_t kv = 0; kv < n_kv; ++kv)
            for (uint32_t dd = 0; dd < d; ++dd) {
                hk_b[(kv * max_ctx + i) * d + dd] = dist(rng) * 100.f;
                hv_b[(kv * max_ctx + i) * d + dd] = dist(rng) * 100.f;
            }

    auto run = [&](const std::vector<float>& kf, const std::vector<float>& vf) {
        auto* dq    = sycl::malloc_device<sycl::half>(T * n_q * d, q);
        auto* dk_in = sycl::malloc_device<sycl::half>(T * n_kv * d, q);
        auto* dv_in = sycl::malloc_device<sycl::half>(T * n_kv * d, q);
        auto* dk    = sycl::malloc_device<sycl::half>(n_kv * max_ctx * d, q);
        auto* dv    = sycl::malloc_device<sycl::half>(n_kv * max_ctx * d, q);
        auto* dy    = sycl::malloc_device<sycl::half>(T * n_q * d, q);
        auto upload = [&](sycl::half* p, const std::vector<float>& src) {
            std::vector<sycl::half> tmp(src.size());
            for (size_t i = 0; i < src.size(); ++i) tmp[i] = sycl::half(src[i]);
            q.memcpy(p, tmp.data(), tmp.size() * sizeof(sycl::half)).wait();
        };
        upload(dq, hq); upload(dk_in, hk_in); upload(dv_in, hv_in);
        upload(dk, kf); upload(dv, vf);
        ie::full_attention(q, dq, dk_in, dv_in, dk, dv, dy,
                           T, start_pos, n_q, n_kv, d, max_ctx).wait();
        std::vector<sycl::half> hy(T * n_q * d);
        q.memcpy(hy.data(), dy, hy.size() * sizeof(sycl::half)).wait();
        sycl::free(dq, q); sycl::free(dk_in, q); sycl::free(dv_in, q);
        sycl::free(dk, q); sycl::free(dv, q); sycl::free(dy, q);
        return hy;
    };

    auto y_a = run(hk_a, hv_a);
    auto y_b = run(hk_b, hv_b);
    bool ok = true;
    char extra[128] = {};
    for (size_t i = 0; i < y_a.size(); ++i) {
        if (float(y_a[i]) != float(y_b[i])) {
            ok = false;
            std::snprintf(extra, sizeof(extra),
                          " (i=%zu y_a=%.5f y_b=%.5f)", i, float(y_a[i]), float(y_b[i]));
            break;
        }
    }
    return report("causal mask: future garbage ignored", ok, extra);
}

// --------------------------------------------------------------
// Test 3: vs scalar reference. T=16 (prefill), n_q=8, n_kv=2, d=64, max_ctx=32.
// --------------------------------------------------------------
bool test_vs_reference(sycl::queue& q, uint32_t T, const char* label) {
    const uint32_t n_q = 8, n_kv = 2, d = 64, max_ctx = 256;
    const uint32_t start_pos = 0;       // simplest: clean prefill
    std::mt19937 rng(0xBE);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> hq(T * n_q * d), hk_in(T * n_kv * d), hv_in(T * n_kv * d);
    std::vector<float> hk(n_kv * max_ctx * d, 0.f), hv(n_kv * max_ctx * d, 0.f);
    for (auto& x : hq) x = dist(rng);
    for (auto& x : hk_in) x = dist(rng);
    for (auto& x : hv_in) x = dist(rng);

    // Pre-populate the cache with hk_in/hv_in so the reference and the kernel see
    // the same data after the kernel's append. (The reference uses k/v with the
    // appended values already in [start_pos, start_pos+T).)
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t kv = 0; kv < n_kv; ++kv)
            for (uint32_t dd = 0; dd < d; ++dd) {
                hk[(kv * max_ctx + (start_pos + t)) * d + dd] = hk_in[(t * n_kv + kv) * d + dd];
                hv[(kv * max_ctx + (start_pos + t)) * d + dd] = hv_in[(t * n_kv + kv) * d + dd];
            }

    // Reference
    std::vector<float> y_ref(T * n_q * d);
    attention_ref(T, n_q, n_kv, d, max_ctx, start_pos, hq, hk, hv, y_ref);

    // GPU: pre-fill cache with zeros (start_pos=0 means kernel writes everything).
    auto* dq    = sycl::malloc_device<sycl::half>(T * n_q * d, q);
    auto* dk_in = sycl::malloc_device<sycl::half>(T * n_kv * d, q);
    auto* dv_in = sycl::malloc_device<sycl::half>(T * n_kv * d, q);
    auto* dk    = sycl::malloc_device<sycl::half>(n_kv * max_ctx * d, q);
    auto* dv    = sycl::malloc_device<sycl::half>(n_kv * max_ctx * d, q);
    auto* dy    = sycl::malloc_device<sycl::half>(T * n_q * d, q);
    auto upload = [&](sycl::half* p, const std::vector<float>& src) {
        std::vector<sycl::half> tmp(src.size());
        for (size_t i = 0; i < src.size(); ++i) tmp[i] = sycl::half(src[i]);
        q.memcpy(p, tmp.data(), tmp.size() * sizeof(sycl::half)).wait();
    };
    upload(dq, hq); upload(dk_in, hk_in); upload(dv_in, hv_in);
    q.memset(dk, 0, n_kv * max_ctx * d * sizeof(sycl::half)).wait();
    q.memset(dv, 0, n_kv * max_ctx * d * sizeof(sycl::half)).wait();

    ie::full_attention(q, dq, dk_in, dv_in, dk, dv, dy,
                       T, start_pos, n_q, n_kv, d, max_ctx).wait();
    std::vector<sycl::half> hy(T * n_q * d);
    q.memcpy(hy.data(), dy, hy.size() * sizeof(sycl::half)).wait();
    sycl::free(dq, q); sycl::free(dk_in, q); sycl::free(dv_in, q);
    sycl::free(dk, q); sycl::free(dv, q); sycl::free(dy, q);

    float max_abs = 0.f, max_rel = 0.f;
    size_t worst = 0;
    for (size_t i = 0; i < hy.size(); ++i) {
        const float a = y_ref[i];
        const float b = float(hy[i]);
        const float ad = std::fabs(a - b);
        if (ad > max_abs) { max_abs = ad; worst = i; }
        const float scale = std::fmax(std::fabs(a), 1e-3f);
        const float r = ad / scale;
        if (r > max_rel) max_rel = r;
    }
    char extra[160];
    std::snprintf(extra, sizeof(extra), "  max_abs=%-8.4g max_rel=%-8.4g worst@%zu (exp=%g got=%g)",
                  max_abs, max_rel, worst, y_ref[worst], float(hy[worst]));
    // Phase 4 gate is max_abs ≤ 1e-2 vs scalar fp32 reference (spec: "within
    // 1e-2 max-abs, bf16/fp16 acceptable"). max_rel is only meaningful when
    // values aren't near zero, which they often are in random-init attention.
    return report(label, max_abs <= 1e-2f, extra);
}

// --------------------------------------------------------------
// Test 4: GQA mapping. With Q heads h and h+gqa_ratio mapped to different
// kv_heads but same Q, KV, expect different outputs (sanity for the gqa
// division logic).
// --------------------------------------------------------------
bool test_gqa(sycl::queue& q) {
    const uint32_t T = 1, n_q = 16, n_kv = 2, d = 32, max_ctx = 16;
    const uint32_t start_pos = 3;
    std::mt19937 rng(0x6A);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> hq(T * n_q * d, 1.0f);   // all Q heads identical
    std::vector<float> hk_in(T * n_kv * d), hv_in(T * n_kv * d);
    std::vector<float> hk(n_kv * max_ctx * d), hv(n_kv * max_ctx * d);
    for (auto& x : hk_in) x = dist(rng);
    for (auto& x : hv_in) x = dist(rng);
    for (auto& x : hk) x = dist(rng);
    for (auto& x : hv) x = dist(rng);

    auto* dq    = sycl::malloc_device<sycl::half>(T * n_q * d, q);
    auto* dk_in = sycl::malloc_device<sycl::half>(T * n_kv * d, q);
    auto* dv_in = sycl::malloc_device<sycl::half>(T * n_kv * d, q);
    auto* dk    = sycl::malloc_device<sycl::half>(n_kv * max_ctx * d, q);
    auto* dv    = sycl::malloc_device<sycl::half>(n_kv * max_ctx * d, q);
    auto* dy    = sycl::malloc_device<sycl::half>(T * n_q * d, q);
    auto upload = [&](sycl::half* p, const std::vector<float>& src) {
        std::vector<sycl::half> tmp(src.size());
        for (size_t i = 0; i < src.size(); ++i) tmp[i] = sycl::half(src[i]);
        q.memcpy(p, tmp.data(), tmp.size() * sizeof(sycl::half)).wait();
    };
    upload(dq, hq); upload(dk_in, hk_in); upload(dv_in, hv_in);
    upload(dk, hk); upload(dv, hv);
    ie::full_attention(q, dq, dk_in, dv_in, dk, dv, dy,
                       T, start_pos, n_q, n_kv, d, max_ctx).wait();
    std::vector<sycl::half> hy(T * n_q * d);
    q.memcpy(hy.data(), dy, hy.size() * sizeof(sycl::half)).wait();
    sycl::free(dq, q); sycl::free(dk_in, q); sycl::free(dv_in, q);
    sycl::free(dk, q); sycl::free(dv, q); sycl::free(dy, q);

    // All Q heads in [0, 8) share kv_head 0; heads in [8, 16) share kv_head 1.
    // Same kv group should give same output; different groups should differ.
    bool ok = true;
    char extra[128] = {};
    auto sample = [&](uint32_t h, uint32_t dd) { return float(hy[h * d + dd]); };
    for (uint32_t h = 1; h < 8; ++h) {
        if (!close(sample(0, 0), sample(h, 0), 1e-3f)) {
            ok = false;
            std::snprintf(extra, sizeof(extra), " (heads 0/%u differ in kv-grp 0)", h);
            break;
        }
    }
    if (ok) for (uint32_t h = 9; h < 16; ++h) {
        if (!close(sample(8, 0), sample(h, 0), 1e-3f)) {
            ok = false;
            std::snprintf(extra, sizeof(extra), " (heads 8/%u differ in kv-grp 1)", h);
            break;
        }
    }
    if (ok && close(sample(0, 0), sample(8, 0), 1e-2f)) {
        ok = false;
        std::snprintf(extra, sizeof(extra), " (kv-grps 0 and 1 produced same output: %.4f)", sample(0,0));
    }
    return report("GQA mapping (group invariance, group split)", ok, extra);
}

// --------------------------------------------------------------
// Test 5: KvCache integration smoke.
// --------------------------------------------------------------
bool test_kv_cache_smoke(sycl::queue& q) {
    ie::DeviceAllocator alloc;
    if (auto err = alloc.init(); !err.empty()) {
        std::printf("  %sFAIL%s alloc init: %s\n", R, Z, err.c_str());
        return false;
    }
    ie::KvCacheConfig cfg{ /*n_layers_full=*/2, /*n_kv_heads=*/2,
                           /*max_ctx=*/64, /*head_dim=*/32 };
    ie::KvCache cache;
    if (auto err = cache.init(alloc, cfg); !err.empty()) {
        std::printf("  %sFAIL%s kvcache init: %s\n", R, Z, err.c_str());
        return false;
    }
    // Append T=4 tokens to layer 1 and verify
    constexpr uint32_t T = 4;
    std::vector<sycl::half> hk(T * cfg.n_kv_heads * cfg.head_dim),
                            hv(T * cfg.n_kv_heads * cfg.head_dim);
    for (size_t i = 0; i < hk.size(); ++i) hk[i] = sycl::half(float(i) * 0.01f);
    for (size_t i = 0; i < hv.size(); ++i) hv[i] = sycl::half(-float(i) * 0.01f);
    auto* dk = alloc.malloc(hk.size() * sizeof(sycl::half));
    auto* dv = alloc.malloc(hv.size() * sizeof(sycl::half));
    alloc.queue().memcpy(dk, hk.data(), hk.size() * sizeof(sycl::half)).wait();
    alloc.queue().memcpy(dv, hv.data(), hv.size() * sizeof(sycl::half)).wait();
    cache.append(alloc.queue(), /*layer=*/1, /*start_pos=*/0, T,
                 static_cast<sycl::half*>(dk), static_cast<sycl::half*>(dv)).wait();
    cache.set_length(1, T);

    // Read back layer-1 cache and verify the first T positions for each kv head.
    const uint64_t per_layer_elems = uint64_t(cfg.n_kv_heads) * cfg.max_ctx * cfg.head_dim;
    std::vector<sycl::half> hk_back(per_layer_elems), hv_back(per_layer_elems);
    alloc.queue().memcpy(hk_back.data(),
                          cache.k_ptr() + per_layer_elems * 1,
                          per_layer_elems * sizeof(sycl::half)).wait();
    alloc.queue().memcpy(hv_back.data(),
                          cache.v_ptr() + per_layer_elems * 1,
                          per_layer_elems * sizeof(sycl::half)).wait();
    bool ok = true;
    for (uint32_t kv = 0; kv < cfg.n_kv_heads && ok; ++kv) {
        for (uint32_t t = 0; t < T && ok; ++t) {
            for (uint32_t d = 0; d < cfg.head_dim; ++d) {
                const uint64_t cache_idx = (uint64_t(kv) * cfg.max_ctx + t) * cfg.head_dim + d;
                const uint64_t in_idx    = (uint64_t(t) * cfg.n_kv_heads + kv) * cfg.head_dim + d;
                if (float(hk_back[cache_idx]) != float(hk[in_idx]) ||
                    float(hv_back[cache_idx]) != float(hv[in_idx])) {
                    ok = false; break;
                }
            }
        }
    }
    alloc.free(dk); alloc.free(dv);
    cache.free_storage();
    return report("KvCache append round-trip", ok);
}

// --------------------------------------------------------------
// Test 6: INT8 KV prefill→decode correctness.
// Calls kv.quantize_to_int8() (new API) after T>1 fp16 prefill,
// then verifies INT8 decode output is close to fp16 decode output.
// --------------------------------------------------------------
bool test_int8_kv_prefill_to_decode(sycl::queue& /*q_unused*/) {
    ie::DeviceAllocator alloc;
    if (auto err = alloc.init(); !err.empty()) {
        std::printf("  %sFAIL%s alloc: %s\n", R, Z, err.c_str());
        return false;
    }
    auto& q = alloc.queue();

    const uint32_t n_q = 16, n_kv = 2, d = 256, max_ctx = 256;
    const uint32_t T_pf = 64;   // T>1 prefill length
    std::mt19937 rng(0xB70);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    const size_t pf_elems = size_t(T_pf) * n_kv * d;
    std::vector<sycl::half> hk_pf(pf_elems), hv_pf(pf_elems);
    for (auto& x : hk_pf) x = sycl::half(dist(rng));
    for (auto& x : hv_pf) x = sycl::half(dist(rng));

    std::vector<sycl::half> hq_d(n_q * d), hk_d(n_kv * d), hv_d(n_kv * d);
    for (auto& x : hq_d) x = sycl::half(dist(rng));
    for (auto& x : hk_d) x = sycl::half(dist(rng));
    for (auto& x : hv_d) x = sycl::half(dist(rng));

    ie::KvCacheConfig kvcfg{/*n_layers_full=*/1, n_kv, max_ctx, d, /*use_int8=*/true};
    ie::KvCache kv;
    if (auto err = kv.init(alloc, kvcfg); !err.empty()) {
        std::printf("  %sFAIL%s kvcache: %s\n", R, Z, err.c_str());
        return false;
    }

    auto* dk_pf = static_cast<sycl::half*>(alloc.malloc(pf_elems * sizeof(sycl::half)));
    auto* dv_pf = static_cast<sycl::half*>(alloc.malloc(pf_elems * sizeof(sycl::half)));
    q.memcpy(dk_pf, hk_pf.data(), pf_elems * sizeof(sycl::half)).wait();
    q.memcpy(dv_pf, hv_pf.data(), pf_elems * sizeof(sycl::half)).wait();
    kv.append(q, /*layer=*/0, /*start_pos=*/0, T_pf, dk_pf, dv_pf).wait();
    kv.set_length(0, T_pf);
    alloc.free(dk_pf);
    alloc.free(dv_pf);

    // This is the new API — does not exist yet.
    kv.quantize_to_int8(q, /*layer=*/0, /*start_pos=*/0, T_pf).wait();

    const uint32_t n_chunks = (max_ctx + 63u) / 64u;
    auto* partials = static_cast<float*>(
        alloc.malloc(size_t(n_chunks) * n_q * (d + 2u) * sizeof(float)));

    auto* dq    = sycl::malloc_device<sycl::half>(n_q  * d, q);
    auto* dk_d  = sycl::malloc_device<sycl::half>(n_kv * d, q);
    auto* dv_d  = sycl::malloc_device<sycl::half>(n_kv * d, q);
    auto* y_i8  = sycl::malloc_device<sycl::half>(n_q  * d, q);
    auto* y_f16 = sycl::malloc_device<sycl::half>(n_q  * d, q);
    q.memcpy(dq,   hq_d.data(), n_q  * d * sizeof(sycl::half)).wait();
    q.memcpy(dk_d, hk_d.data(), n_kv * d * sizeof(sycl::half)).wait();
    q.memcpy(dv_d, hv_d.data(), n_kv * d * sizeof(sycl::half)).wait();

    ie::full_attention_fa2_decode_int8(
        q, dq, dk_d, dv_d,
        kv.k_int8_ptr(), kv.v_int8_ptr(),
        kv.k_scales_ptr(), kv.v_scales_ptr(),
        nullptr, nullptr,
        y_i8, partials, T_pf, n_q, n_kv, d, max_ctx).wait();

    ie::full_attention_fa2_decode(
        q, dq, dk_d, dv_d,
        kv.k_ptr(), kv.v_ptr(),
        y_f16, partials, T_pf, n_q, n_kv, d, max_ctx).wait();

    std::vector<sycl::half> h_i8(n_q * d), h_f16(n_q * d);
    q.memcpy(h_i8.data(),  y_i8,  n_q * d * sizeof(sycl::half)).wait();
    q.memcpy(h_f16.data(), y_f16, n_q * d * sizeof(sycl::half)).wait();

    double dot = 0.0, nrm_i = 0.0, nrm_f = 0.0;
    for (uint32_t i = 0; i < n_q * d; ++i) {
        const double a = float(h_i8[i]), b = float(h_f16[i]);
        dot += a * b; nrm_i += a * a; nrm_f += b * b;
    }
    const double cos_sim = dot / (std::sqrt(nrm_i * nrm_f) + 1e-12);

    sycl::free(dq, q); sycl::free(dk_d, q); sycl::free(dv_d, q);
    sycl::free(y_i8, q); sycl::free(y_f16, q);
    alloc.free(partials);
    kv.free_storage();

    char extra[64] = {};
    std::snprintf(extra, sizeof(extra), " (cos=%.4f)", cos_sim);
    return report("INT8 KV: prefill→decode cos-sim > 0.99", cos_sim > 0.99, extra);
}

}  // namespace

int main() {
    auto q = make_queue();
    std::printf("Device: %s\n\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    int fails = 0;
    if (!test_kv_cache_smoke(q)) ++fails;
    if (!test_int8_kv_prefill_to_decode(q)) ++fails;
    if (!test_constant_v(q))    ++fails;
    if (!test_causality(q))     ++fails;
    if (!test_gqa(q))           ++fails;
    if (!test_vs_reference(q, 1,   "vs scalar ref T=1"))   ++fails;
    if (!test_vs_reference(q, 16,  "vs scalar ref T=16"))  ++fails;
    if (!test_vs_reference(q, 256, "vs scalar ref T=256")) ++fails;

    std::printf("\n%s\n", fails == 0 ? "ALL PASS" : "FAILURES");
    return fails;
}
