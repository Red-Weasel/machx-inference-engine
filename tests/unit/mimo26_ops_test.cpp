// tests/unit/mimo26_ops_test.cpp — MiMo-V2.6 P2 kernels vs host references (docs/mimo26/00_PORT_PLAN.md):
// the GQA attention with K head dim != V head dim, per-head sinks and a sliding window across two appends;
// the split-K decode and the XMX prefill attention at MiMo's geometry; the fp32 RMSNorm; the qkv split; the router logits; axpy and
// SwiGLU. Needs an Arc GPU: SKIPs (77) without.
#undef NDEBUG
#include "ie/mimo26.hpp"       // mimo26_fp8_rows_to_f32 (the host dequant)
#include "ie/mimo26_ops.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

using namespace ie;

namespace {

float h(float v) { return float(sycl::half(v)); }   // round through fp16

bool close(const char* tag, const std::vector<float>& ref, const std::vector<float>& got, double atol, double rtol) {
    assert(ref.size() == got.size());
    double worst = 0; size_t bad = 0, wi = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = std::fabs(double(ref[i]) - double(got[i])), tol = atol + rtol * std::fabs(double(ref[i]));
        // !(d <= tol): a NaN or inf result (d NaN) FAILS -- `d > tol` is false for NaN (P3a/P3b/P4 gate finding)
        if (!(d <= tol)) { ++bad; if (!(d - tol <= worst)) { worst = std::isfinite(d) ? d - tol : 1e300; wi = i; } }
    }
    if (bad) std::printf("  %-22s FAIL: %zu of %zu outside tolerance, worst at %zu: ref %.6g got %.6g\n", tag, bad, ref.size(), wi, double(ref[wi]), double(got[wi]));
    else     std::printf("  %-22s ok (%zu values)\n", tag, ref.size());
    return bad == 0;
}

// CPU attention over the whole sequence for the queries [pos0, pos0 + T): inputs already fp16-rounded.
void ref_attention(const std::vector<float>& Q, const std::vector<float>& Kall, const std::vector<float>& Vall, std::vector<float>& y,
                   uint32_t T, uint32_t pos0, uint32_t n_q, uint32_t n_kv, uint32_t hd, uint32_t hdv, uint32_t window, const float* sinks) {
    const uint32_t gqa = n_q / n_kv;
    y.assign(size_t(T) * n_q * hdv, 0.f);
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t hq = 0; hq < n_q; ++hq) {
            const uint32_t kvh = hq / gqa, p = pos0 + t, n_keys = p + 1, k0 = (window && n_keys > window) ? n_keys - window : 0;
            std::vector<double> s(n_keys, 0.0);
            double mx = -1e300;
            for (uint32_t i = k0; i < n_keys; ++i) {
                double dot = 0;
                for (uint32_t d = 0; d < hd; ++d) dot += double(Q[(size_t(t) * n_q + hq) * hd + d]) * double(Kall[(size_t(i) * n_kv + kvh) * hd + d]);
                s[i] = dot / std::sqrt(double(hd)); mx = std::max(mx, s[i]);
            }
            if (sinks) mx = std::max(mx, double(sinks[hq]));
            double den = sinks ? std::exp(double(sinks[hq]) - mx) : 0.0;
            for (uint32_t i = k0; i < n_keys; ++i) den += std::exp(s[i] - mx);
            for (uint32_t d = 0; d < hdv; ++d) {
                double acc = 0;
                for (uint32_t i = k0; i < n_keys; ++i) acc += std::exp(s[i] - mx) / den * double(Vall[(size_t(i) * n_kv + kvh) * hdv + d]);
                y[(size_t(t) * n_q + hq) * hdv + d] = float(acc);
            }
        }
}

}  // namespace

int main() {
    sycl::device dev;
    bool found = false;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos && !found) { dev = d; found = true; }
    }
    if (!found) { std::printf("SKIP: no Arc GPU\n"); return 77; }
    sycl::queue q(sycl::context(dev), dev, sycl::property_list{sycl::property::queue::in_order{}});
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    bool ok = true;

    // ---- rms_norm ----
    {
        const uint32_t T = 3, H = 4096; const float eps = 1e-6f;
        std::vector<float> x(size_t(T) * H), w(H), r32(size_t(T) * H);
        for (auto& v : x) v = nd(rng) * 3.f; for (auto& v : w) v = 1.f + 0.1f * nd(rng);
        for (uint32_t t = 0; t < T; ++t) {
            double ss = 0; for (uint32_t k = 0; k < H; ++k) ss += double(x[size_t(t) * H + k]) * x[size_t(t) * H + k];
            const double inv = 1.0 / std::sqrt(ss / H + eps);
            for (uint32_t k = 0; k < H; ++k) r32[size_t(t) * H + k] = float(x[size_t(t) * H + k] * inv * w[k]);
        }
        float* dx = sycl::malloc_device<float>(x.size(), q); float* dw = sycl::malloc_device<float>(H, q);
        float* dy32 = sycl::malloc_device<float>(x.size(), q); sycl::half* dy16 = sycl::malloc_device<sycl::half>(x.size(), q);
        q.memcpy(dx, x.data(), x.size() * 4); q.memcpy(dw, w.data(), H * 4);
        mimo26_rms_norm(q, dx, dw, dy16, dy32, T, H, eps);
        std::vector<float> g32(x.size()); std::vector<sycl::half> g16(x.size());
        q.memcpy(g32.data(), dy32, x.size() * 4); q.memcpy(g16.data(), dy16, x.size() * 2); q.wait();
        std::vector<float> g16f(x.size()); for (size_t i = 0; i < x.size(); ++i) g16f[i] = float(g16[i]);
        ok &= close("rms_norm fp32", r32, g32, 1e-5, 1e-5);
        ok &= close("rms_norm fp16", r32, g16f, 2e-3, 1e-3);
        sycl::free(dx, q); sycl::free(dw, q); sycl::free(dy32, q); sycl::free(dy16, q);
    }
    // ---- split_qkv ----
    {
        const uint32_t T = 2, n_q = 4, n_kv = 2, hd = 32, hdv = 16, N = n_q * hd + n_kv * hd + n_kv * hdv;
        std::vector<float> qkv(size_t(T) * N); for (size_t i = 0; i < qkv.size(); ++i) qkv[i] = float(i % 251) - 100.f;
        float* d = sycl::malloc_device<float>(qkv.size(), q);
        sycl::half* Q = sycl::malloc_device<sycl::half>(size_t(T) * n_q * hd, q); sycl::half* K = sycl::malloc_device<sycl::half>(size_t(T) * n_kv * hd, q); sycl::half* V = sycl::malloc_device<sycl::half>(size_t(T) * n_kv * hdv, q);
        q.memcpy(d, qkv.data(), qkv.size() * 4);
        mimo26_split_qkv(q, d, Q, K, V, T, n_q, n_kv, hd, hdv);
        std::vector<sycl::half> gq(size_t(T) * n_q * hd), gk(size_t(T) * n_kv * hd), gv(size_t(T) * n_kv * hdv);
        q.memcpy(gq.data(), Q, gq.size() * 2); q.memcpy(gk.data(), K, gk.size() * 2); q.memcpy(gv.data(), V, gv.size() * 2); q.wait();
        size_t bad = 0;
        for (uint32_t t = 0; t < T; ++t) {
            for (uint32_t c = 0; c < n_q * hd; ++c) bad += float(gq[size_t(t) * n_q * hd + c]) != qkv[size_t(t) * N + c];
            for (uint32_t c = 0; c < n_kv * hd; ++c) bad += float(gk[size_t(t) * n_kv * hd + c]) != qkv[size_t(t) * N + n_q * hd + c];
            for (uint32_t c = 0; c < n_kv * hdv; ++c) bad += float(gv[size_t(t) * n_kv * hdv + c]) != qkv[size_t(t) * N + n_q * hd + n_kv * hd + c];
        }
        std::printf("  %-22s %s\n", "split_qkv", bad ? "FAIL" : "ok"); ok &= bad == 0;
        sycl::free(d, q); sycl::free(Q, q); sycl::free(K, q); sycl::free(V, q);
    }
    // ---- attention: two appends, K 48 / V 32, GQA 8/2, window 5 with sinks, then window 0 without ----
    for (int variant = 0; variant < 2; ++variant) {
        const uint32_t n_q = 8, n_kv = 2, hd = 48, hdv = 32, max_ctx = 64, T1 = 6, T2 = 5, window = variant == 0 ? 5u : 0u;
        const uint32_t Tall = T1 + T2;
        std::vector<float> Q(size_t(Tall) * n_q * hd), Kf(size_t(Tall) * n_kv * hd), Vf(size_t(Tall) * n_kv * hdv), sinks(n_q);
        for (auto& v : Q) v = h(nd(rng)); for (auto& v : Kf) v = h(nd(rng)); for (auto& v : Vf) v = h(nd(rng)); for (auto& v : sinks) v = nd(rng);
        std::vector<sycl::half> Qh(Q.size()), Kh(Kf.size()), Vh(Vf.size());
        for (size_t i = 0; i < Q.size(); ++i) Qh[i] = sycl::half(Q[i]);
        for (size_t i = 0; i < Kf.size(); ++i) Kh[i] = sycl::half(Kf[i]);
        for (size_t i = 0; i < Vf.size(); ++i) Vh[i] = sycl::half(Vf[i]);
        sycl::half* dQ = sycl::malloc_device<sycl::half>(Qh.size(), q); sycl::half* dK = sycl::malloc_device<sycl::half>(Kh.size(), q); sycl::half* dV = sycl::malloc_device<sycl::half>(Vh.size(), q);
        sycl::half* kc = sycl::malloc_device<sycl::half>(size_t(n_kv) * max_ctx * hd, q); sycl::half* vc = sycl::malloc_device<sycl::half>(size_t(n_kv) * max_ctx * hdv, q);
        sycl::half* dy = sycl::malloc_device<sycl::half>(size_t(Tall) * n_q * hdv, q); float* ds = sycl::malloc_device<float>(n_q, q);
        q.memcpy(dQ, Qh.data(), Qh.size() * 2); q.memcpy(dK, Kh.data(), Kh.size() * 2); q.memcpy(dV, Vh.data(), Vh.size() * 2); q.memcpy(ds, sinks.data(), n_q * 4);
        const float* dsinks = variant == 0 ? ds : nullptr;
        mimo26_attention(q, dQ, dK, dV, kc, vc, dy, T1, 0, n_q, n_kv, hd, hdv, max_ctx, window, dsinks);
        mimo26_attention(q, dQ + size_t(T1) * n_q * hd, dK + size_t(T1) * n_kv * hd, dV + size_t(T1) * n_kv * hdv, kc, vc, dy + size_t(T1) * n_q * hdv, T2, T1, n_q, n_kv, hd, hdv, max_ctx, window, dsinks);
        std::vector<sycl::half> gy(size_t(Tall) * n_q * hdv); q.memcpy(gy.data(), dy, gy.size() * 2).wait();
        std::vector<float> got(gy.size()); for (size_t i = 0; i < gy.size(); ++i) got[i] = float(gy[i]);
        std::vector<float> ref;
        ref_attention(Q, Kf, Vf, ref, Tall, 0, n_q, n_kv, hd, hdv, window, variant == 0 ? sinks.data() : nullptr);
        ok &= close(variant == 0 ? "attention swa+sinks" : "attention full", ref, got, 4e-3, 4e-3);
        sycl::free(dQ, q); sycl::free(dK, q); sycl::free(dV, q); sycl::free(kc, q); sycl::free(vc, q); sycl::free(dy, q); sycl::free(ds, q);
    }
    // ---- split-K decode attention at MiMo's geometry: (0) the full layers -- GQA 64/4, K 192 / V 128 in 192-wide V rows,
    //      no window, NO sinks (as in production) -- after a 4,500-key prefill (mimo26_attention), then steps of T = 1 / 3 / 8
    //      (split counts 64 / 21 / 8);
    //      (1) the SWA layers -- GQA 64/8, window 128, sinks, a 136-slot ring wrapped twice -- decoded from position 0 in
    //      steps of 1..8. The caches and the V row padding start as NaN: a read of an unwritten slot or a pad column fails.
    for (int variant = 0; variant < 2; ++variant) {
        const bool full = variant == 0;
        const uint32_t n_q = 64, n_kv = full ? 4u : 8u, hd = 192, hdv = 128, vs = full ? 192u : 128u, window = full ? 0u : 128u;
        const uint32_t prefill = full ? 4500u : 0u, Tall = full ? prefill + 12u : 300u, ring = full ? 0u : window + 8u;
        const uint32_t max_ctx = Tall, slots = ring ? ring : max_ctx;
        std::vector<float> Q(size_t(Tall) * n_q * hd), Kf(size_t(Tall) * n_kv * hd), Vf(size_t(Tall) * n_kv * hdv), sinks(n_q);
        for (auto& v : Q) v = h(nd(rng) * 2.f); for (auto& v : Kf) v = h(nd(rng)); for (auto& v : Vf) v = h(nd(rng)); for (auto& v : sinks) v = nd(rng) * 2.f + 2.f;
        const sycl::half nan = sycl::half(std::numeric_limits<float>::quiet_NaN());
        std::vector<sycl::half> Qh(Q.size()), Kh(Kf.size()), Vh(size_t(Tall) * n_kv * vs, nan);
        for (size_t i = 0; i < Q.size(); ++i) Qh[i] = sycl::half(Q[i]);
        for (size_t i = 0; i < Kf.size(); ++i) Kh[i] = sycl::half(Kf[i]);
        for (size_t r = 0; r < size_t(Tall) * n_kv; ++r) for (uint32_t d = 0; d < hdv; ++d) Vh[r * vs + d] = sycl::half(Vf[r * hdv + d]);
        sycl::half* dQ = sycl::malloc_device<sycl::half>(Qh.size(), q); sycl::half* dK = sycl::malloc_device<sycl::half>(Kh.size(), q); sycl::half* dV = sycl::malloc_device<sycl::half>(Vh.size(), q);
        sycl::half* kc = sycl::malloc_device<sycl::half>(size_t(n_kv) * slots * hd, q); sycl::half* vc = sycl::malloc_device<sycl::half>(size_t(n_kv) * slots * vs, q);
        sycl::half* dy = sycl::malloc_device<sycl::half>(size_t(Tall) * n_q * hdv, q); float* ds = sycl::malloc_device<float>(n_q, q);
        float* dp = sycl::malloc_device<float>(size_t(8) * n_q * mimo26_decode_max_splits() * (hdv + 2), q);
        q.memcpy(dQ, Qh.data(), Qh.size() * 2); q.memcpy(dK, Kh.data(), Kh.size() * 2); q.memcpy(dV, Vh.data(), Vh.size() * 2); q.memcpy(ds, sinks.data(), n_q * 4);
        q.memset(kc, 0xFF, size_t(n_kv) * slots * hd * 2); q.memset(vc, 0xFF, size_t(n_kv) * slots * vs * 2);   // fp16 0xFFFF = NaN
        auto step = [&](uint32_t p0, uint32_t T, bool decode) {
            const sycl::half* Qp = dQ + size_t(p0) * n_q * hd; const sycl::half* Kp = dK + size_t(p0) * n_kv * hd; const sycl::half* Vp = dV + size_t(p0) * n_kv * vs;
            // the full layers run without sinks in production (the merge's no-sink branch; gate finding P3a-2)
            if (decode) mimo26_attention_decode(q, Qp, Kp, Vp, kc, vc, dy + size_t(p0) * n_q * hdv, dp, T, p0, n_q, n_kv, hd, hdv, max_ctx, window, full ? nullptr : ds, {}, ring, vs);
            else        mimo26_attention(q, Qp, Kp, Vp, kc, vc, dy + size_t(p0) * n_q * hdv, T, p0, n_q, n_kv, hd, hdv, max_ctx, window, full ? nullptr : ds, {}, ring, vs);
        };
        uint32_t p0 = 0;
        if (full) { step(0, prefill, false); p0 = prefill; }
        const uint32_t steps_full[] = {1, 3, 8}, steps_swa[] = {1, 8, 3, 1, 5, 2, 8, 7, 1, 4, 6};
        for (uint32_t k = 0; p0 < Tall; ++k) {
            const uint32_t T = std::min(full ? steps_full[k % 3] : steps_swa[k % 11], Tall - p0);
            step(p0, T, true); p0 += T;
        }
        const uint32_t r0 = full ? prefill : 0u, nr = Tall - r0;   // the decoded rows
        std::vector<sycl::half> gy(size_t(nr) * n_q * hdv); q.memcpy(gy.data(), dy + size_t(r0) * n_q * hdv, gy.size() * 2).wait();
        std::vector<float> got(gy.size()); for (size_t i = 0; i < gy.size(); ++i) got[i] = float(gy[i]);
        std::vector<float> Qr(Q.begin() + size_t(r0) * n_q * hd, Q.end()), ref;
        ref_attention(Qr, Kf, Vf, ref, nr, r0, n_q, n_kv, hd, hdv, window, full ? nullptr : sinks.data());
        ok &= close(full ? "decode attn full 4.5k" : "decode attn swa ring", ref, got, 2e-3, 4e-3);
        sycl::free(dQ, q); sycl::free(dK, q); sycl::free(dV, q); sycl::free(kc, q); sycl::free(vc, q); sycl::free(dy, q); sycl::free(ds, q); sycl::free(dp, q);
    }
    // ---- XMX prefill attention (K 192 / V 128, linear caches): (0) the full layers -- GQA 64/4, V in 192-wide rows, no
    //      window, no sinks -- in chunks of 37 / 1000 / 63 / 2048 / 9 rows (4-token work-groups cut mid-chunk, a 3,157-key
    //      context); (1) GQA 64/8, window 128, sinks, a 1152-slot RING (wrapped twice) in chunks of 37 / 1000 / 63 / 1000 /
    //      700. The caches, the K slack rows and the V pad columns start as NaN. Rows checked vs the host reference: every
    //      row of the chunks marked, the last 16 of the rest.
    for (int variant = 0; variant < 2; ++variant) {
        const bool full = variant == 0;
        const uint32_t n_q = 64, n_kv = full ? 4u : 8u, hd = 192, hdv = 128, vs = full ? 192u : 128u, window = full ? 0u : 128u;
        const std::vector<uint32_t> chunks = full ? std::vector<uint32_t>{37, 1000, 63, 2048, 9} : std::vector<uint32_t>{37, 1000, 63, 1000, 700};
        const std::vector<bool> check_all = full ? std::vector<bool>{true, false, true, false, true} : std::vector<bool>{true, true, true, true, true};
        uint32_t Tall = 0; for (uint32_t c : chunks) Tall += c;
        const uint32_t max_ctx = Tall, ring = full ? 0u : 1152u, slots = ring ? ring : max_ctx;
        std::vector<float> Q(size_t(Tall) * n_q * hd), Kf(size_t(Tall) * n_kv * hd), Vf(size_t(Tall) * n_kv * hdv), sinks(n_q);
        for (auto& v : Q) v = h(nd(rng) * 2.f); for (auto& v : Kf) v = h(nd(rng)); for (auto& v : Vf) v = h(nd(rng)); for (auto& v : sinks) v = nd(rng) * 2.f + 2.f;
        const sycl::half nan = sycl::half(std::numeric_limits<float>::quiet_NaN());
        std::vector<sycl::half> Qh(Q.size()), Kh(Kf.size()), Vh(size_t(Tall) * n_kv * vs, nan);
        for (size_t i = 0; i < Q.size(); ++i) Qh[i] = sycl::half(Q[i]);
        for (size_t i = 0; i < Kf.size(); ++i) Kh[i] = sycl::half(Kf[i]);
        for (size_t r = 0; r < size_t(Tall) * n_kv; ++r) for (uint32_t d = 0; d < hdv; ++d) Vh[r * vs + d] = sycl::half(Vf[r * hdv + d]);
        const size_t k_rows = size_t(n_kv) * slots + 64;
        sycl::half* dQ = sycl::malloc_device<sycl::half>(Qh.size(), q); sycl::half* dK = sycl::malloc_device<sycl::half>(Kh.size(), q); sycl::half* dV = sycl::malloc_device<sycl::half>(Vh.size(), q);
        sycl::half* kc = sycl::malloc_device<sycl::half>(k_rows * hd, q); sycl::half* vc = sycl::malloc_device<sycl::half>(size_t(n_kv) * slots * vs, q);
        sycl::half* dy = sycl::malloc_device<sycl::half>(size_t(Tall) * n_q * hdv, q); float* ds = sycl::malloc_device<float>(n_q, q);
        q.memcpy(dQ, Qh.data(), Qh.size() * 2); q.memcpy(dK, Kh.data(), Kh.size() * 2); q.memcpy(dV, Vh.data(), Vh.size() * 2); q.memcpy(ds, sinks.data(), n_q * 4);
        q.memset(kc, 0xFF, k_rows * hd * 2); q.memset(vc, 0xFF, size_t(n_kv) * slots * vs * 2);
        q.memset(dy, 0xFF, size_t(Tall) * n_q * hdv * 2);
        const float* dsinks = full ? nullptr : ds;
        uint32_t p0 = 0;
        std::vector<float> got, ref, g1, r1;
        for (size_t ci = 0; ci < chunks.size(); ++ci) {
            const uint32_t T = chunks[ci];
            mimo26_attention_prefill_xmx(q, dQ + size_t(p0) * n_q * hd, dK + size_t(p0) * n_kv * hd, dV + size_t(p0) * n_kv * vs, kc, vc,
                                         dy + size_t(p0) * n_q * hdv, T, p0, n_q, n_kv, max_ctx, window, dsinks, {}, vs, ring);
            const uint32_t nr = check_all[ci] ? T : std::min(T, 16u), r0 = p0 + T - nr;   // the rows checked
            std::vector<sycl::half> gy(size_t(nr) * n_q * hdv); q.memcpy(gy.data(), dy + size_t(r0) * n_q * hdv, gy.size() * 2).wait();
            for (auto v : gy) got.push_back(float(v));
            std::vector<float> Qr(Q.begin() + size_t(r0) * n_q * hd, Q.begin() + size_t(r0 + nr) * n_q * hd);
            ref_attention(Qr, Kf, Vf, r1, nr, r0, n_q, n_kv, hd, hdv, window, full ? nullptr : sinks.data());
            ref.insert(ref.end(), r1.begin(), r1.end());
            p0 += T;
        }
        ok &= close(full ? "prefill xmx full 3.2k" : "prefill xmx swa ring", ref, got, 2e-3, 4e-3);
        sycl::free(dQ, q); sycl::free(dK, q); sycl::free(dV, q); sycl::free(kc, q); sycl::free(vc, q); sycl::free(dy, q); sycl::free(ds, q);
    }
    // ---- FP8-resident dense (P4 lever 3): mimo26_gemv_fp8 at M = 1 / 3 / 8 vs a double host sum over the exact E4M3
    //      values and per-row 128-column F32 scales (N 1000: the 8-row work-group shape; 13568: the 16-row one), and
    //      mimo26_fp8_to_f16 BIT-exact vs the host dequant (mimo26_fp8_rows_to_f32, then fp16 rounded to nearest even) ----
    for (uint32_t N : {1000u, 13568u}) {
        const uint32_t K = 4096, SK = K / 128;
        std::uniform_int_distribution<int> bd(0, 255);
        std::vector<uint8_t> w8(size_t(N) * K); std::vector<float> sr(size_t(N) * SK);
        for (auto& b : w8) { int v = bd(rng); if ((v & 0x7F) == 0x7F) v &= 0xFE; b = uint8_t(v); }   // no NaN codes
        for (auto& v : sr) v = std::ldexp(0.5f + 0.5f * std::fabs(nd(rng)), -6 - int(bd(rng) % 12));   // arbitrary (not 2^k) scales
        std::vector<float> wf; mimo26_fp8_rows_to_f32(w8.data(), sr.data(), N, K, 128, wf);
        uint8_t* dw = sycl::malloc_device<uint8_t>(w8.size(), q); float* ds = sycl::malloc_device<float>(sr.size(), q);
        q.memcpy(dw, w8.data(), w8.size()); q.memcpy(ds, sr.data(), sr.size() * 4);
        for (uint32_t M : {1u, 3u, 8u}) {
            std::vector<float> x(size_t(M) * K); for (auto& v : x) v = h(nd(rng));
            std::vector<sycl::half> xh(x.size()); for (size_t i = 0; i < x.size(); ++i) xh[i] = sycl::half(x[i]);
            std::vector<float> ref(size_t(M) * N);
            for (uint32_t m = 0; m < M; ++m) for (uint32_t n = 0; n < N; ++n) {
                double acc = 0; for (uint32_t k = 0; k < K; ++k) acc += double(x[size_t(m) * K + k]) * double(wf[size_t(n) * K + k]);
                ref[size_t(m) * N + n] = float(acc);
            }
            sycl::half* dx = sycl::malloc_device<sycl::half>(xh.size(), q); float* dy = sycl::malloc_device<float>(ref.size(), q);
            q.memcpy(dx, xh.data(), xh.size() * 2);
            mimo26_gemv_fp8(q, dx, K, dw, ds, dy, M, K, N);
            std::vector<float> got(ref.size()); q.memcpy(got.data(), dy, got.size() * 4).wait();
            char tag[32]; std::snprintf(tag, sizeof tag, "gemv_fp8 N%u M%u", N, M);
            ok &= close(tag, ref, got, 5e-3, 1e-4);
            sycl::free(dx, q); sycl::free(dy, q);
        }
        sycl::half* d16 = sycl::malloc_device<sycl::half>(w8.size(), q);
        mimo26_fp8_to_f16(q, dw, ds, d16, N, K);
        std::vector<sycl::half> g16(w8.size()); q.memcpy(g16.data(), d16, g16.size() * 2).wait();
        size_t bad = 0;
        for (size_t i = 0; i < wf.size(); ++i) {
            // the host reference rounds to nearest even with the compiler's _Float16: the host sycl::half(float) conversion
            // TRUNCATES small magnitudes (-5.2090076e-05 = 873.93 subnormal ulps -> 873, not 874), the device's rounds
            const uint16_t hb = sycl::bit_cast<uint16_t>(static_cast<_Float16>(wf[i])), gb = sycl::bit_cast<uint16_t>(g16[i]);
            if (hb == gb) continue;
            if (bad < 4) std::printf("    mismatch at %zu: byte 0x%02x scale %.9g -> f32 %.9g host RNE 0x%04x device 0x%04x\n", i, w8[i], double(sr[(i / K) * SK + (i % K) / 128]),
                                     double(wf[i]), hb, gb);
            ++bad;
        }
        std::printf("  %-22s %s (%zu of %zu fp16 weights differ from the host dequant rounded to nearest even)\n", N == 1000 ? "fp8_to_f16 N1000" : "fp8_to_f16 N13568", bad ? "FAIL" : "ok", bad, wf.size());
        ok &= bad == 0;
        sycl::free(dw, q); sycl::free(ds, q); sycl::free(d16, q);
    }
    // ---- router logits ----
    {
        const uint32_t T = 3, H = 4096, E = 256;
        std::vector<float> x(size_t(T) * H), w(size_t(E) * H), ref(size_t(T) * E);
        for (auto& v : x) v = nd(rng); for (auto& v : w) v = nd(rng) * 0.02f;
        for (uint32_t t = 0; t < T; ++t) for (uint32_t e = 0; e < E; ++e) { double s = 0; for (uint32_t k = 0; k < H; ++k) s += double(x[size_t(t) * H + k]) * w[size_t(e) * H + k]; ref[size_t(t) * E + e] = float(s); }
        float* dx = sycl::malloc_device<float>(x.size(), q); float* dw = sycl::malloc_device<float>(w.size(), q); float* dl = sycl::malloc_device<float>(ref.size(), q);
        q.memcpy(dx, x.data(), x.size() * 4); q.memcpy(dw, w.data(), w.size() * 4);
        mimo26_router_logits(q, dx, dw, dl, T, H, E);
        std::vector<float> got(ref.size()); q.memcpy(got.data(), dl, got.size() * 4).wait();
        ok &= close("router_logits", ref, got, 1e-4, 1e-4);
        sycl::free(dx, q); sycl::free(dw, q); sycl::free(dl, q);
    }
    // ---- axpy + swiglu ----
    {
        const uint64_t n = 10007;
        std::vector<float> x(n), y(n), g(n), u(n), ra(n), rs(n);
        for (uint64_t i = 0; i < n; ++i) { x[i] = nd(rng); y[i] = nd(rng); g[i] = nd(rng) * 3; u[i] = nd(rng); ra[i] = y[i] + 0.707f * x[i]; rs[i] = g[i] / (1.f + std::exp(-g[i])) * u[i]; }
        float* dx = sycl::malloc_device<float>(n, q); float* dy = sycl::malloc_device<float>(n, q); float* dg = sycl::malloc_device<float>(n, q); float* du = sycl::malloc_device<float>(n, q); sycl::half* dh = sycl::malloc_device<sycl::half>(n, q);
        q.memcpy(dx, x.data(), n * 4); q.memcpy(dy, y.data(), n * 4); q.memcpy(dg, g.data(), n * 4); q.memcpy(du, u.data(), n * 4);
        mimo26_axpy(q, dx, 0.707f, dy, n); mimo26_swiglu_f32(q, dg, du, dh, n);
        std::vector<float> ga(n); std::vector<sycl::half> gh(n); q.memcpy(ga.data(), dy, n * 4); q.memcpy(gh.data(), dh, n * 2); q.wait();
        std::vector<float> gs(n); for (uint64_t i = 0; i < n; ++i) gs[i] = float(gh[i]);
        ok &= close("axpy", ra, ga, 1e-6, 1e-6);
        ok &= close("swiglu_f32", rs, gs, 4e-3, 2e-3);
        sycl::free(dx, q); sycl::free(dy, q); sycl::free(dg, q); sycl::free(du, q); sycl::free(dh, q);
    }
    // ---- fp16 subnormals on the device (informational: whether a small fp16 store flushes to zero) ----
    {
        const float in[4] = {1e-5f, 3e-6f, 6e-8f, 1e-4f};
        float* d = sycl::malloc_device<float>(4, q); sycl::half* hd = sycl::malloc_device<sycl::half>(4, q); float* back = sycl::malloc_device<float>(4, q);
        q.memcpy(d, in, 16);
        q.parallel_for(sycl::range<1>(4), [=](sycl::id<1> i) { hd[i] = sycl::half(d[i]); back[i] = float(hd[i]) * 2.f; });
        float got[4]; q.memcpy(got, back, 16).wait();
        std::printf("  fp16 subnormal store on device: 1e-5 -> %.3g, 3e-6 -> %.3g, 6e-8 -> %.3g, 1e-4 -> %.3g (x2; 0 = flushed)\n",
                    double(got[0]) / 2, double(got[1]) / 2, double(got[2]) / 2, double(got[3]) / 2);
        sycl::free(d, q); sycl::free(hd, q); sycl::free(back, q);
    }
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
