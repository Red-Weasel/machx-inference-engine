// tests/unit/ds41_attn_xmx_ht16_test.cpp -- the V4.1 XMX decode attention's 16-head tile (2026-10-20, tensor parallel
// over 4 cards: 16 of the 64 heads per card). The decode shapes: head_dim 512, a 128-slot window segment and a latent
// segment visited through ascending picks (negative sentinels after the valid ones), masked keys, sinks, T = 1 and 2.
//   1. 32 heads as two 16-head calls (the 16-head tile) == one 32-head call (the 32-head tile) within fp32 rounding of
//      l's summation order (the only arithmetic the tile changes)
//   2. both within 1e-2 x max|y| of the fp32 kernel (ds4_attention_segs) per row (fp16 operands)
//   3. the 16-head tile's three launches bit-identical
//   4. a row whose keys are all masked: 0 from both tiles
//   5. the prefill kernel (ds4_attention_xmx): two 16-head calls == one 32-head call
#include "ie/deepseek41_attn_xmx.hpp"
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
    std::printf("  [%s] %s%s%s\n", cond ? "PASS" : "FAIL", what, detail.empty() ? "" : " -- ", detail.c_str());
    if (!cond) ++g_fail;
}

int main() {
    setenv("IE_DS4_ATTN_XMX", "0", 1);   // ds4_attention_segs = the fp32 kernel (the reference)
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
    constexpr uint32_t D = 512, NH = 32, NA = 128, NB = 640, NP = 96;
    const float scaling = 1.0f / std::sqrt(float(D));
    std::mt19937 rng(20261020u);
    std::normal_distribution<float> nd(0.f, 1.f);

    for (uint32_t T : {1u, 2u}) {
        std::printf("T = %u\n", T);
        const uint32_t NKV = NA + NB;
        std::vector<float> hq(size_t(T) * NH * D), ha(size_t(NA) * D), hmask(size_t(T) * NKV, 0.f), hsinks(NH);
        std::vector<sycl::half> hb(size_t(NB) * D);
        for (auto& v : hq) v = nd(rng) * 2.0f;
        for (auto& v : ha) v = nd(rng);
        for (auto& v : hb) v = sycl::half(nd(rng));
        for (auto& v : hsinks) v = nd(rng);
        // the window: a few masked slots; the latent: every key's bias 0 where picked, -inf elsewhere
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t c = 0; c < NA; ++c) if ((c * 7 + t) % 11 == 0) hmask[size_t(t) * NKV + c] = -std::numeric_limits<float>::infinity();
            for (uint32_t c = NA; c < NKV; ++c) hmask[size_t(t) * NKV + c] = -std::numeric_limits<float>::infinity();
        }
        // picks: ascending, valid first, then negative sentinels; row 1 has fewer
        std::vector<int32_t> hp(size_t(T) * NP, -1);
        for (uint32_t t = 0; t < T; ++t) {
            std::vector<int32_t> all(NB); for (uint32_t i = 0; i < NB; ++i) all[i] = int32_t(i);
            std::shuffle(all.begin(), all.end(), rng);
            const uint32_t nv = t ? NP - 17 : NP;
            std::vector<int32_t> pick(all.begin(), all.begin() + nv); std::sort(pick.begin(), pick.end());
            for (uint32_t i = 0; i < nv; ++i) { hp[size_t(t) * NP + i] = pick[i]; hmask[size_t(t) * NKV + NA + uint32_t(pick[i])] = 0.f; }
        }
        float* dq = sycl::malloc_device<float>(hq.size(), q); float* da = sycl::malloc_device<float>(ha.size(), q);
        sycl::half* db = sycl::malloc_device<sycl::half>(hb.size(), q); float* dm = sycl::malloc_device<float>(hmask.size(), q);
        float* ds = sycl::malloc_device<float>(hsinks.size(), q); int32_t* dp = sycl::malloc_device<int32_t>(hp.size(), q);
        float* q16 = sycl::malloc_device<float>(size_t(T) * 16 * D, q);
        float* y32 = sycl::malloc_device<float>(hq.size(), q); float* yref = sycl::malloc_device<float>(hq.size(), q);
        float* y16 = sycl::malloc_device<float>(size_t(T) * 16 * D, q);
        q.memcpy(dq, hq.data(), hq.size() * 4); q.memcpy(da, ha.data(), ha.size() * 4); q.memcpy(db, hb.data(), hb.size() * 2);
        q.memcpy(dm, hmask.data(), hmask.size() * 4); q.memcpy(ds, hsinks.data(), hsinks.size() * 4); q.memcpy(dp, hp.data(), hp.size() * 4).wait();

        ie::Ds4KvSegs segs; segs.a = da; segs.n_a = NA; segs.b = db; segs.n_b = NB; segs.b_f16 = true; segs.b_picks = dp; segs.n_picks = NP;
        ok("eligible at 16 heads", ie::ds41_attention_decode_xmx_eligible(q.get_device(), T, 16, D));
        ok("refused at 8 heads", !ie::ds41_attention_decode_xmx_eligible(q.get_device(), T, 8, D));
        ie::ds41_attention_decode_xmx(q, dq, segs, dm, ds, y32, T, NH, D, scaling).wait();
        ie::ds4_attention_segs(q, dq, segs, dm, ds, yref, T, NH, D, scaling).wait();
        std::vector<float> h32(hq.size()), href(hq.size()), h16(hq.size());
        q.memcpy(h32.data(), y32, h32.size() * 4); q.memcpy(href.data(), yref, href.size() * 4).wait();
        std::vector<float> first16;
        for (uint32_t half = 0; half < 2; ++half) {
            for (uint32_t t = 0; t < T; ++t) q.memcpy(q16 + size_t(t) * 16 * D, dq + (size_t(t) * NH + half * 16) * D, size_t(16) * D * 4);
            ie::ds41_attention_decode_xmx(q, q16, segs, dm, ds + half * 16, y16, T, 16, D, scaling).wait();
            std::vector<float> tmp(size_t(T) * 16 * D);
            q.memcpy(tmp.data(), y16, tmp.size() * 4).wait();
            for (uint32_t t = 0; t < T; ++t) std::memcpy(h16.data() + (size_t(t) * NH + half * 16) * D, tmp.data() + size_t(t) * 16 * D, size_t(16) * D * 4);
            if (half == 0) {   // 3. the same call again: bit-identical
                ie::ds41_attention_decode_xmx(q, q16, segs, dm, ds, y16, T, 16, D, scaling).wait();
                std::vector<float> again(tmp.size()); q.memcpy(again.data(), y16, again.size() * 4).wait();
                ok("16-head tile: a repeat is bit-identical", std::memcmp(again.data(), tmp.data(), tmp.size() * 4) == 0);
            }
        }
        // 1 and 2, per row
        for (uint32_t t = 0; t < T; ++t) {
            double d1632 = 0, d32ref = 0, d16ref = 0, ymax = 0;
            for (size_t i = size_t(t) * NH * D; i < size_t(t + 1) * NH * D; ++i) {
                d1632 = std::max(d1632, double(std::fabs(h16[i] - h32[i])));
                d32ref = std::max(d32ref, double(std::fabs(h32[i] - href[i])));
                d16ref = std::max(d16ref, double(std::fabs(h16[i] - href[i])));
                ymax = std::max(ymax, double(std::fabs(href[i])));
            }
            // the fp16 operands' error, relative to the output's scale: ~1e-3 measured on the 32-head tile; 1e-2 the bar
            const double bound = 1e-2 * std::max(1.0, ymax);
            char buf[256];
            std::snprintf(buf, sizeof buf, "row %u: max |y16 - y32| %.3g (|y| max %.3g)", t, d1632, ymax);
            ok("16-head tile == 32-head tile within fp32 rounding", d1632 <= 1e-5 * std::max(1.0, ymax), buf);
            std::snprintf(buf, sizeof buf, "row %u: 32-tile %.3g, 16-tile %.3g vs fp32, bound %.3g", t, d32ref, d16ref, bound);
            ok("both tiles within 1e-2 x max|y| of the fp32 kernel", d32ref <= bound && d16ref <= bound, buf);
        }
        // 4. every key masked for every row: zeros from both tiles
        {
            std::vector<float> allm(hmask.size(), -std::numeric_limits<float>::infinity());
            q.memcpy(dm, allm.data(), allm.size() * 4).wait();
            ie::ds41_attention_decode_xmx(q, dq, segs, dm, nullptr, y32, T, NH, D, scaling).wait();
            ie::ds41_attention_decode_xmx(q, q16, segs, dm, nullptr, y16, T, 16, D, scaling).wait();
            std::vector<float> z32(hq.size()), z16(size_t(T) * 16 * D);
            q.memcpy(z32.data(), y32, z32.size() * 4); q.memcpy(z16.data(), y16, z16.size() * 4).wait();
            bool zero = true; for (float v : z32) zero &= v == 0.f; for (float v : z16) zero &= v == 0.f;
            ok("all keys masked, no sinks: both tiles give exactly 0", zero);
        }
        for (void* p : {(void*)dq, (void*)da, (void*)db, (void*)dm, (void*)ds, (void*)dp, (void*)q16, (void*)y32, (void*)yref, (void*)y16}) sycl::free(p, q);
    }
    // the PREFILL kernel (ds4_attention_xmx, T > 16 rows, one work-group per row and tile looping over the key blocks):
    // 32 heads as two 16-head calls == one 32-head call, within fp32 rounding of l's block sums
    {
        constexpr uint32_t T = 40, NKV = 300;
        std::printf("prefill, T = %u, %u keys\n", T, NKV);
        std::vector<float> hq(size_t(T) * NH * D), hk(size_t(NKV) * D), hm(size_t(T) * NKV, 0.f), hs(NH);
        for (auto& v : hq) v = nd(rng) * 2.0f; for (auto& v : hk) v = nd(rng); for (auto& v : hs) v = nd(rng);
        for (uint32_t t = 0; t < T; ++t)   // a causal band over the last 128 keys plus a sparse older region
            for (uint32_t c = 0; c < NKV; ++c) {
                const uint32_t pos = NKV - T + t;
                const bool ok_ = c <= pos && (pos - c < 128 || (c * 13 + t) % 5 == 0);
                if (!ok_) hm[size_t(t) * NKV + c] = -std::numeric_limits<float>::infinity();
            }
        float* dq = sycl::malloc_device<float>(hq.size(), q); float* dk = sycl::malloc_device<float>(hk.size(), q);
        float* dm = sycl::malloc_device<float>(hm.size(), q); float* ds = sycl::malloc_device<float>(hs.size(), q);
        float* q16 = sycl::malloc_device<float>(size_t(T) * 16 * D, q); float* y16 = sycl::malloc_device<float>(size_t(T) * 16 * D, q);
        float* y32 = sycl::malloc_device<float>(hq.size(), q);
        q.memcpy(dq, hq.data(), hq.size() * 4); q.memcpy(dk, hk.data(), hk.size() * 4); q.memcpy(dm, hm.data(), hm.size() * 4);
        q.memcpy(ds, hs.data(), hs.size() * 4).wait();
        // (ds4_attention_xmx_eligible is off in this process: IE_DS4_ATTN_XMX=0 above makes ds4_attention_segs the fp32
        // reference, and the switch is read once; the model run shows the 16-head prefill taking the XMX path)
        ie::ds4_attention_xmx(q, dq, dk, dm, ds, y32, T, NH, D, NKV, scaling).wait();
        std::vector<float> h32(hq.size()), h16(hq.size());
        q.memcpy(h32.data(), y32, h32.size() * 4).wait();
        for (uint32_t half = 0; half < 2; ++half) {
            for (uint32_t t = 0; t < T; ++t) q.memcpy(q16 + size_t(t) * 16 * D, dq + (size_t(t) * NH + half * 16) * D, size_t(16) * D * 4);
            ie::ds4_attention_xmx(q, q16, dk, dm, ds + half * 16, y16, T, 16, D, NKV, scaling).wait();
            std::vector<float> tmp(size_t(T) * 16 * D); q.memcpy(tmp.data(), y16, tmp.size() * 4).wait();
            for (uint32_t t = 0; t < T; ++t) std::memcpy(h16.data() + (size_t(t) * NH + half * 16) * D, tmp.data() + size_t(t) * 16 * D, size_t(16) * D * 4);
        }
        double d = 0, ymax = 0;
        for (size_t i = 0; i < h32.size(); ++i) { d = std::max(d, double(std::fabs(h16[i] - h32[i]))); ymax = std::max(ymax, double(std::fabs(h32[i]))); }
        char buf[160]; std::snprintf(buf, sizeof buf, "max |y16 - y32| %.3g over %u rows (|y| max %.3g)", d, T, ymax);
        ok("prefill: 16-head tile == 32-head tile within fp32 rounding", d <= 1e-5 * std::max(1.0, ymax), buf);
        for (void* p : {(void*)dq, (void*)dk, (void*)dm, (void*)ds, (void*)q16, (void*)y16, (void*)y32}) sycl::free(p, q);
    }
    if (g_fail) { std::printf("ds41_attn_xmx_ht16_test: %d FAILED\n", g_fail); return 1; }
    std::printf("ds41_attn_xmx_ht16_test: PASS\n");
    return 0;
}
