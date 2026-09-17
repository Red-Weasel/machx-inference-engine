// bench/ds41_attn_decode_probe.cpp -- FEASIBILITY PROBE (docs/deepseek41/74): is the block-sparse XMX attention
// kernel, called at the V4.1 DECODE shape with the 64 heads as its row tile, faster than the fp32 split kernel?
//
// Shape = the 2k prompt at decode: T rows, 64 heads, head_dim 512, segment a = the 128-slot window (fp32), segment
// b = NC fp32 latents with 512 picks live per row (the indexer's top-k), everything else masked. Synthetic data:
// this measures TIME and the fp16-operand error against the fp32 kernel, not model quality.
//   usage: ds41-attn-decode-probe [T=1] [NC=1024] [iters=200]
#include "ie/deepseek4_attn.hpp"
#include "ie/deepseek41_candidate.hpp"
#include "ie/deepseek41_attn_xmx.hpp"

#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    const uint32_t T = argc > 1 ? uint32_t(std::atoi(argv[1])) : 1u;
    const uint32_t NC = argc > 2 ? uint32_t(std::atoi(argv[2])) : 1024u;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 200;
    const uint32_t NH = 64, HD = 512, WIN = 128, K = 512;
    sycl::device dev;
    for (auto& d : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) { dev = d; break; }
    sycl::queue q(dev, sycl::property::queue::in_order{});
    std::printf("device %s, T %u, window %u, NC %u, %u picks/row\n", dev.get_info<sycl::info::device::name>().c_str(), T, WIN, NC, K);

    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    const uint32_t NKV = WIN + NC;
    std::vector<float> hq(size_t(T) * NH * HD), ha(size_t(WIN) * HD), hb(size_t(NC) * HD), hmask(size_t(T) * NKV, -INFINITY), hs(NH);
    for (auto& v : hq) v = nd(rng) * 0.05f;
    for (auto& v : ha) v = nd(rng);
    for (auto& v : hb) v = nd(rng);
    for (auto& v : hs) v = nd(rng);
    // picks: K distinct latents per row (ascending, sentinels -1 after), and the mask opened on the window + the picks
    std::vector<int32_t> hpk(size_t(T) * K), hnv(T);
    for (uint32_t t = 0; t < T; ++t) {
        std::vector<int32_t> all(NC); for (uint32_t i = 0; i < NC; ++i) all[i] = int32_t(i);
        std::shuffle(all.begin(), all.end(), rng);
        const uint32_t nv = std::min(K, NC); all.resize(nv); std::sort(all.begin(), all.end());
        for (uint32_t j = 0; j < K; ++j) hpk[size_t(t) * K + j] = j < nv ? all[j] : -1;
        hnv[t] = int32_t(nv);
        for (uint32_t i = 0; i < WIN; ++i) hmask[size_t(t) * NKV + i] = 0.f;
        for (uint32_t j = 0; j < nv; ++j) hmask[size_t(t) * NKV + WIN + uint32_t(all[j])] = 0.f;
    }
    auto up = [&](auto& h) { using V = typename std::decay_t<decltype(h)>::value_type; V* d = sycl::malloc_device<V>(h.size() + 64 * HD, q); q.memset(d, 0, (h.size() + 64 * HD) * sizeof(V)); q.memcpy(d, h.data(), h.size() * sizeof(V)).wait(); return d; };
    float* dq = up(hq); float* da = up(ha); float* db = up(hb); float* dm = up(hmask); float* ds = up(hs);
    int32_t* dpk = up(hpk); int32_t* dnv = up(hnv);
    float* y32 = sycl::malloc_device<float>(size_t(T) * NH * HD, q);
    float* y16 = sycl::malloc_device<float>(size_t(T) * NH * HD, q);
    const float scaling = 1.0f / std::sqrt(float(HD));

    ie::Ds4KvSegs segs; segs.a = da; segs.n_a = WIN; segs.b = db; segs.n_b = NC; segs.b_f16 = false;
    segs.b_picks = dpk; segs.n_picks = K; segs.b_pick_n = dnv;
    ie::g_ds4_attn_split_fixed64 = true;   // the V4.1 runtime's setting (P2)

    auto time = [&](const char* name, auto&& fn) {
        for (int i = 0; i < 5; ++i) fn(); q.wait();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) fn();
        q.wait();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
        std::printf("  %-44s %8.1f us/call  -> %6.2f ms/token over 40 layers\n", name, us, us * 40.0 / 1000.0);
        return us;
    };
    time("fp32 split kernel, gathered (the shipped path)", [&] { ie::ds4_attention_segs(q, dq, segs, dm, ds, y32, T, NH, HD, scaling); });
    ie::Ds4KvSegs dense = segs; dense.b_picks = nullptr; dense.n_picks = 0; dense.b_pick_n = nullptr;
    time("fp32 split kernel, dense mask scan", [&] { ie::ds4_attention_segs(q, dq, dense, dm, ds, y32, T, NH, HD, scaling); });
    if (auto e = ie::ds4_attention_xmx_reserve(q, T, NH, NKV); !e.empty()) { std::printf("xmx reserve: %s\n", e.c_str()); return 1; }
    time("XMX block-sparse, fp32 b converted per call", [&] { ie::ds4_attention_xmx_segs(q, dq, dense, dm, ds, y16, T, NH, HD, scaling, false); });
    time("XMX block-sparse, kv_prepared (kernel only)", [&] { ie::ds4_attention_xmx_segs(q, dq, dense, dm, ds, y16, T, NH, HD, scaling, true); });

    float* ydx = sycl::malloc_device<float>(size_t(T) * NH * HD, q);
    time("Phase 34: gather + split-K XMX + combine", [&] { ie::ds41_attention_decode_xmx(q, dq, segs, dm, ds, ydx, T, NH, HD, scaling); });
    {
        ie::ds4_attention_segs(q, dq, segs, dm, ds, y32, T, NH, HD, scaling).wait();
        ie::ds41_attention_decode_xmx(q, dq, segs, dm, ds, ydx, T, NH, HD, scaling).wait();
        std::vector<float> a(size_t(T) * NH * HD), b(a.size());
        q.memcpy(a.data(), y32, a.size() * 4).wait(); q.memcpy(b.data(), ydx, b.size() * 4).wait();
        double mx = 0, ref = 0, l2n = 0, l2d = 0;
        for (size_t i = 0; i < a.size(); ++i) { const double d = std::fabs(double(a[i]) - double(b[i])); mx = std::max(mx, d); ref = std::max(ref, std::fabs(double(a[i]))); l2n += d * d; l2d += double(a[i]) * a[i]; }
        std::printf("  Phase 34 vs fp32: max |diff| %.3e, max |ref| %.3e, rel %.3e, relative L2 %.3e\n", mx, ref, ref > 0 ? mx / ref : 0.0, l2d > 0 ? std::sqrt(l2n / l2d) : 0.0);
    }
    // the fp16-operand error against the fp32 kernel, on the same inputs
    ie::ds4_attention_segs(q, dq, segs, dm, ds, y32, T, NH, HD, scaling).wait();
    ie::ds4_attention_xmx_segs(q, dq, dense, dm, ds, y16, T, NH, HD, scaling, false).wait();
    std::vector<float> o32(size_t(T) * NH * HD), o16(o32.size());
    q.memcpy(o32.data(), y32, o32.size() * 4).wait(); q.memcpy(o16.data(), y16, o16.size() * 4).wait();
    double mx = 0, ref = 0, l2n = 0, l2d = 0;
    for (size_t i = 0; i < o32.size(); ++i) { const double d = std::fabs(double(o32[i]) - double(o16[i])); mx = std::max(mx, d); ref = std::max(ref, std::fabs(double(o32[i]))); l2n += d * d; l2d += double(o32[i]) * o32[i]; }
    std::printf("  XMX vs fp32: max |diff| %.3e, max |ref| %.3e, rel %.3e, relative L2 %.3e\n", mx, ref, ref > 0 ? mx / ref : 0.0, l2d > 0 ? std::sqrt(l2n / l2d) : 0.0);
    return 0;
}
