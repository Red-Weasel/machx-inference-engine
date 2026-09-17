// tests/unit/deepseek4_attn_xmx_gate_test.cpp — docs/deepseek4/72 Phase K gate.
//
// ds4_attention_xmx (fp16 operands, fp32 softmax/accumulate, block skipping)
// against ds4_attention's fp32 kernel on the shapes the model runs: 32 heads
// per card, head_dim 512, a 128-token sliding causal band plus a compressed
// region with a top-k-style sparse bias, fully-masked tokens, and sinks.
//   1. max |y_xmx - y_fp32| within the DERIVED fp16-input bound (per row:
//      (exp(2B) - 1 + 2^-10) * max|k|_inf + 1e-5, B = scaling * 2^-10 * |q| * max|k|)
//   2. a token whose keys are all disallowed: both kernels give exactly 0
//   3. three launches bit-identical
//   4. negative control: one allowed block disallowed -> differs beyond the bound
//   5. eligibility: head_dim != 512 is refused by the kernel; the switch is read
//      once per process, so the T <= 16 boundary is not exercised here — the
//      stream-PPL bit-identity on the model proves it (T = 1 never enters).
#include "ie/deepseek4_attn.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

static int g_fail = 0;
static void ok(const char* what, bool cond, const std::string& detail = {}) {
    std::printf("  [%s] %s%s%s\n", cond ? "PASS" : "FAIL", what, detail.empty() ? "" : " -- ",
                detail.c_str());
    if (!cond) ++g_fail;
}

int main() {
    // The fp32 path must be what ds4_attention dispatches to in this process
    // (the reference): the switch is read once, before the first call.
    setenv("IE_DS4_ATTN_XMX", "0", 1);
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    const uint32_t H = 32, D = 512, WIN = 128;
    const float scaling = 1.f / std::sqrt(float(D));
    const float lowest = std::numeric_limits<float>::lowest();
    const float ninf   = -std::numeric_limits<float>::infinity();

    struct Shape { uint32_t T, n_sl, n_c; };
    const Shape shapes[] = {{17, 17 + WIN, 40}, {64, 64 + WIN, 300}, {1024, 1024 + WIN, 300}};
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> U(-0.5f, 0.5f);

    for (const Shape& sh : shapes) {
        const uint32_t T = sh.T, n_kv = sh.n_sl + sh.n_c;
        const size_t nq = size_t(T) * H * D, nk = size_t(n_kv) * D, nm = size_t(T) * n_kv;
        std::vector<float> hq(nq), hk(nk), hm(nm), hs(H);
        for (float& v : hq) v = U(rng);
        for (float& v : hk) v = U(rng);
        for (float& v : hs) v = U(rng) * 4.f;
        // mask: sliding band [t + WIN - WIN, t + WIN] over the first n_sl keys
        // (key index i corresponds to position i - WIN, the chunk's own tokens
        // occupy [WIN, WIN + T)), then the compressed region with a top-k
        // pattern: ~1/3 of the entries allowed with a small position bias.
        std::uniform_real_distribution<float> B(-2.f, 2.f);
        for (uint32_t t = 0; t < T; ++t) {
            float* row = hm.data() + size_t(t) * n_kv;
            for (uint32_t i = 0; i < sh.n_sl; ++i) {
                const int64_t pos = int64_t(i) - int64_t(WIN);
                row[i] = (pos <= int64_t(t) && pos > int64_t(t) - int64_t(WIN)) ? 0.f : lowest;
            }
            for (uint32_t i = sh.n_sl; i < n_kv; ++i)
                row[i] = ((i * 7u + t * 3u) % 3u == 0u) ? B(rng) : ninf;
            if (t == 3) for (uint32_t i = 0; i < n_kv; ++i) row[i] = ninf;   // fully masked token
        }
        float* dq = sycl::malloc_device<float>(nq, q);
        float* dk = sycl::malloc_device<float>(nk, q);
        float* dm = sycl::malloc_device<float>(nm, q);
        float* ds = sycl::malloc_device<float>(H, q);
        float* dy = sycl::malloc_device<float>(nq, q);
        float* dr = sycl::malloc_device<float>(nq, q);
        q.memcpy(dq, hq.data(), nq * 4); q.memcpy(dk, hk.data(), nk * 4);
        q.memcpy(dm, hm.data(), nm * 4); q.memcpy(ds, hs.data(), H * 4); q.wait();

        ie::ds4_attention(q, dq, dk, dm, ds, dr, T, H, D, n_kv, scaling).wait();
        ie::ds4_attention_xmx(q, dq, dk, dm, ds, dy, T, H, D, n_kv, scaling).wait();
        std::vector<float> yr(nq), yx(nq), y2(nq);
        q.memcpy(yr.data(), dr, nq * 4); q.memcpy(yx.data(), dy, nq * 4); q.wait();

        // ---- the bound ----
        float kmax = 0.f, kinf = 0.f;
        for (uint32_t i = 0; i < n_kv; ++i) {
            double s2 = 0; float mx = 0.f;
            for (uint32_t d = 0; d < D; ++d) { const float v = hk[size_t(i) * D + d]; s2 += double(v) * v; mx = std::max(mx, std::fabs(v)); }
            kmax = std::max(kmax, float(std::sqrt(s2))); kinf = std::max(kinf, mx);
        }
        double worst_ratio = 0, worst_err = 0, worst_bound = 0;
        uint64_t masked_exact = 0, masked_total = 0;
        for (uint32_t t = 0; t < T; ++t)
            for (uint32_t h = 0; h < H; ++h) {
                const float* qr = hq.data() + (size_t(t) * H + h) * D;
                double s2 = 0; for (uint32_t d = 0; d < D; ++d) s2 += double(qr[d]) * qr[d];
                const double Bs = double(scaling) * std::ldexp(1.0, -10) * std::sqrt(s2) * kmax;
                const double bound = (std::exp(2 * Bs) - 1 + std::ldexp(1.0, -10)) * kinf + 1e-5;
                double err = 0;
                for (uint32_t d = 0; d < D; ++d) {
                    const size_t o = (size_t(t) * H + h) * D + d;
                    err = std::max(err, double(std::fabs(yx[o] - yr[o])));
                    if (t == 3) { ++masked_total; masked_exact += (yx[o] == yr[o] && yr[o] == 0.f); }
                }
                if (err / bound > worst_ratio) { worst_ratio = err / bound; worst_err = err; worst_bound = bound; }
            }
        const std::string tag = "T=" + std::to_string(T) + " n_kv=" + std::to_string(n_kv);
        ok(("(1) within the derived fp16-input bound, " + tag).c_str(), worst_ratio <= 1.0,
           "worst err " + std::to_string(worst_err) + " / bound " + std::to_string(worst_bound) +
               " = " + std::to_string(worst_ratio));
        ok(("(2) fully-masked token is exactly 0 on both kernels, " + tag).c_str(),
           masked_total > 0 && masked_exact == masked_total,
           std::to_string(masked_exact) + " of " + std::to_string(masked_total));
        // (3) determinism
        bool same = true;
        for (int r = 0; r < 2; ++r) {
            ie::ds4_attention_xmx(q, dq, dk, dm, ds, dy, T, H, D, n_kv, scaling).wait();
            q.memcpy(y2.data(), dy, nq * 4).wait();
            same = same && std::memcmp(y2.data(), yx.data(), nq * 4) == 0;
        }
        ok(("(3) three launches bit-identical, " + tag).c_str(), same);
        // (4) negative control: disallow the keys [WIN, WIN + 64) for every token
        std::vector<float> hm2(hm);
        for (uint32_t t = 0; t < T; ++t)
            for (uint32_t i = WIN; i < std::min<uint32_t>(WIN + 64, n_kv); ++i) hm2[size_t(t) * n_kv + i] = ninf;
        q.memcpy(dm, hm2.data(), nm * 4).wait();
        ie::ds4_attention_xmx(q, dq, dk, dm, ds, dy, T, H, D, n_kv, scaling).wait();
        q.memcpy(y2.data(), dy, nq * 4).wait();
        double maxdiff = 0;
        for (size_t o = 0; o < nq; ++o) maxdiff = std::max(maxdiff, double(std::fabs(y2[o] - yr[o])));
        ok(("(4) NEGATIVE CONTROL: one allowed block disallowed -> beyond the bound, " + tag).c_str(),
           maxdiff > 10 * worst_bound, "max diff " + std::to_string(maxdiff));
        for (float* p : {dq, dk, dm, ds, dy, dr}) sycl::free(p, q);
    }
    // (5) eligibility (the env switch is "0" in this process, so it is false here
    // regardless; the shape refusals are checked through the kernel itself).
    {
        bool threw = false;
        try {
            float* dummy = sycl::malloc_device<float>(64, q);
            ie::ds4_attention_xmx(q, dummy, dummy, nullptr, nullptr, dummy, 20, 32, 256, 8, 1.f);
            sycl::free(dummy, q);
        } catch (const std::exception&) { threw = true; }
        ok("(5) head_dim 256 is refused by ds4_attention_xmx", threw);
        ok("(5) ds4_attention_xmx_eligible is false with IE_DS4_ATTN_XMX=0",
           !ie::ds4_attention_xmx_eligible(q.get_device(), 1024, 32, 512));
    }
    std::printf("\n%s: %d failure(s)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
