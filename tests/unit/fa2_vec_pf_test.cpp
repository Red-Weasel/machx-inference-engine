// tests/unit/fa2_vec_pf_test.cpp -- P4 B22: the FA2 decode partial passes with K/V prefetch must give the SAME bytes as
// without it: the vec kernel (full_attention_fa2_decode_vec, IE_FA2_VEC_PF = 8 / 16 / 32 positions ahead vs 0) and the
// SLM-tiled kernel (full_attention_fa2_decode at head_dim 256, IE_FA2_TILE_PF = 1 vs 0). Output y and every
// super-partial, for the 27B (24 q / 4 kv heads) and 35B (16 / 2) geometries at head_dim 256 and the hd-128 Qwen3
// geometries (32 / 8, 32 / 4), over context lengths that hit every tail (1, 2, 3, 63..65, 127, 1060, 4097, 33000). The
// settings are read once per process, so the test re-runs itself per setting:
//   fa2_vec_pf_test                 -- runs `self --dump FILE` under each prefetch setting, compares with the reference
//   fa2_vec_pf_test --combine       -- P4 B24: the same dump under IE_FA2_COMBINE_SPLIT=0 (the original combine kernels)
//                                      is the reference; SPLIT = 1 / 2 / 4 / 8 / 16 (the widened combine) must give the
//                                      same bytes (y and partials) for the vec, tiled and XMX kernels on every case
//   fa2_vec_pf_test --dump FILE     -- computes every case under the current environment, writes the raw bytes
//   fa2_vec_pf_test --bench         -- time per call (append + partial + combine) under the current environment, plus
//                                      the XMX decode kernel (not bit-identical to vec: max |y - y_vec| shown)
//   fa2_vec_pf_test --cbench        -- P4 B24: combine-pass kernel time per call (event profiling) of the vec, tiled
//                                      and XMX kernels at long context under the current environment
//   fa2_vec_pf_test --xmx           -- P4 B23: accuracy of the vec, tiled and XMX decode kernels against a host fp64
//                                      reference over many context lengths (both hd-256 geometries): max abs error,
//                                      relative error (to max |y_ref|) and max |y_xmx - y_vec|. PASS = XMX within 2e-3
//                                      relative (4 fp16 ulps of the output) on every case.
//   fa2_vec_pf_test --rows          -- P4 B26: full_attention_fa2_decode_rows (G rows in one launch per pass) against the
//                                      per-row single calls: groups of 2..16 rows of different context lengths (1..40950,
//                                      below and above 4096 in one group), each row on its own KV cache, the kernel kind
//                                      per row as the 27B (XMX / vec) and 35B (XMX / tiled) row steps pick it, plus
//                                      all-vec / all-tiled / all-XMX groups (one XMX row failing its shape gate). y, the
//                                      row's partials slice and the appended cache rows must be byte-identical.
//   fa2_vec_pf_test --rows-bench    -- P4 B26: the rows launch against the per-row loop, us per group.
#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

using namespace ie;

namespace {
struct Case { uint32_t nq, nkv, hd, ctx; };
// Buffers are sized for the largest geometry: nq * hd <= 24 * 256, nkv * hd <= 4 * 256, partials for 24 heads at hd 256.
const Case kCases[] = {{24, 4, 256, 1},    {24, 4, 256, 2},    {24, 4, 256, 3},     {24, 4, 256, 63},   {24, 4, 256, 64},
                       {24, 4, 256, 65},   {24, 4, 256, 127},  {24, 4, 256, 1060},  {24, 4, 256, 4097}, {24, 4, 256, 33000},
                       {16, 2, 256, 1},    {16, 2, 256, 65},   {16, 2, 256, 1060},  {16, 2, 256, 4097}, {16, 2, 256, 33000},
                       {32, 8, 128, 1},    {32, 8, 128, 65},   {32, 8, 128, 1060},  {32, 8, 128, 1600}, {32, 8, 128, 8192},
                       {32, 8, 128, 33000}, {32, 4, 128, 127}, {32, 4, 128, 4097},  {32, 4, 128, 33000}};
constexpr uint32_t HD = 256, MAXC = 40960;
constexpr size_t kPn = size_t(MAXC / 64 + 1) * 24 * (HD + 2);   // partial floats
constexpr int kNKer = 3;
const char* kKer[kNKer] = {"vec", "tiled", "xmx"};

// k = 0 vec, 1 the public dispatcher (SLM-tiled at head_dim 256; at head_dim 128 it takes XMX from 192 subgroups), 2 XMX.
sycl::event call(int k, sycl::queue& q, const sycl::half* qi, const sycl::half* ki, const sycl::half* vi, sycl::half* kc,
                 sycl::half* vc, sycl::half* y, float* part, const Case& c, AttnProfileData* prof = nullptr) {
    if (k == 2) return full_attention_fa2_decode_xmx(q, qi, ki, vi, kc, vc, y, part, c.ctx - 1, c.nq, c.nkv, c.hd, MAXC, {}, prof);
    return k == 0 ? full_attention_fa2_decode_vec(q, qi, ki, vi, kc, vc, y, part, c.ctx - 1, c.nq, c.nkv, c.hd, MAXC, {}, prof)
                  : full_attention_fa2_decode(q, qi, ki, vi, kc, vc, y, part, c.ctx - 1, c.nq, c.nkv, c.hd, MAXC, {}, prof);
}

struct Bufs {
    sycl::half *kc, *vc, *qi, *ki, *vi, *y;
    float* part;
    explicit Bufs(sycl::queue& q) {
        const size_t kvn = size_t(4) * MAXC * HD;
        kc = sycl::malloc_device<sycl::half>(kvn, q);
        vc = sycl::malloc_device<sycl::half>(kvn, q);
        qi = sycl::malloc_device<sycl::half>(24 * HD, q);
        ki = sycl::malloc_device<sycl::half>(4 * HD, q);
        vi = sycl::malloc_device<sycl::half>(4 * HD, q);
        y  = sycl::malloc_device<sycl::half>(24 * HD, q);
        part = sycl::malloc_device<float>(kPn, q);
    }
    void free(sycl::queue& q) {
        for (void* p : {static_cast<void*>(kc), static_cast<void*>(vc), static_cast<void*>(qi), static_cast<void*>(ki),
                        static_cast<void*>(vi), static_cast<void*>(y), static_cast<void*>(part)})
            sycl::free(p, q);
    }
};

int run(const char* dump, bool bench) {
    sycl::queue q{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
    std::mt19937 r(22);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto rh = [&](size_t n, float s) { std::vector<sycl::half> v(n); for (auto& x : v) x = sycl::half(nd(r) * s); return v; };
    const size_t kvn = size_t(4) * MAXC * HD;
    Bufs b(q);
    { auto hk = rh(kvn, 1.f), hv = rh(kvn, 1.f); q.memcpy(b.kc, hk.data(), kvn * 2).wait(); q.memcpy(b.vc, hv.data(), kvn * 2).wait(); }
    std::ofstream out;
    if (dump) out.open(dump, std::ios::binary);
    for (const Case& c : kCases) {
        auto hq = rh(c.nq * c.hd, 2.f), hk = rh(c.nkv * c.hd, 1.f), hv = rh(c.nkv * c.hd, 1.f);
        q.memcpy(b.qi, hq.data(), hq.size() * 2).wait();
        q.memcpy(b.ki, hk.data(), hk.size() * 2).wait();
        q.memcpy(b.vi, hv.data(), hv.size() * 2).wait();
        std::vector<sycl::half> yvec(c.nq * c.hd);
        for (int k = 0; k < kNKer; ++k) {
            q.memset(b.part, 0, kPn * sizeof(float)).wait();
            call(k, q, b.qi, b.ki, b.vi, b.kc, b.vc, b.y, b.part, c).wait();
            double maxd = 0, maxy = 0;
            if (bench) {
                std::vector<sycl::half> hy(c.nq * c.hd);
                q.memcpy(hy.data(), b.y, hy.size() * 2).wait();
                if (k == 0) yvec = hy;
                for (size_t i = 0; i < hy.size(); ++i) {
                    maxd = std::max(maxd, std::abs(double(float(hy[i])) - double(float(yvec[i]))));
                    maxy = std::max(maxy, std::abs(double(float(yvec[i]))));
                }
            }
            if (dump) {
                std::vector<sycl::half> hy(c.nq * c.hd);
                std::vector<float> hp(kPn);
                q.memcpy(hy.data(), b.y, hy.size() * 2).wait();
                q.memcpy(hp.data(), b.part, kPn * sizeof(float)).wait();
                out.write(reinterpret_cast<const char*>(hy.data()), hy.size() * 2);
                out.write(reinterpret_cast<const char*>(hp.data()), hp.size() * sizeof(float));
            }
            if (bench) {
                const int reps = 50;
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < reps; ++i) call(k, q, b.qi, b.ki, b.vi, b.kc, b.vc, b.y, b.part, c);
                q.wait();
                const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
                std::printf("  %-5s nq %2u nkv %u hd %u ctx %6u : %8.1f us / call   max|y - y_vec| %.3g (max|y_vec| %.3g)\n",
                            kKer[k], c.nq, c.nkv, c.hd, c.ctx, us, maxd, maxy);
            }
        }
    }
    b.free(q);
    return 0;
}

// --cbench: the combine pass alone (AttnProfileData, device event timestamps) per kernel at long context.
int run_cbench() {
    sycl::queue q{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order{},
                                                                sycl::property::queue::enable_profiling{}}};
    std::mt19937 r(22);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto rh = [&](size_t n, float s) { std::vector<sycl::half> v(n); for (auto& x : v) x = sycl::half(nd(r) * s); return v; };
    const size_t kvn = size_t(4) * MAXC * HD;
    Bufs b(q);
    { auto hk = rh(kvn, 1.f), hv = rh(kvn, 1.f); q.memcpy(b.kc, hk.data(), kvn * 2).wait(); q.memcpy(b.vc, hv.data(), kvn * 2).wait(); }
    const Case cases[] = {{24, 4, 256, 4097}, {24, 4, 256, 33000}, {16, 2, 256, 4097}, {16, 2, 256, 33000},
                          {32, 8, 128, 1600}, {32, 8, 128, 8192}, {32, 8, 128, 33000}};
    const char* split = std::getenv("IE_FA2_COMBINE_SPLIT");
    std::printf("IE_FA2_COMBINE_SPLIT=%s  (combine kernel busy time per call, mean of 50)\n", split ? split : "(default)");
    for (const Case& c : cases) {
        auto hq = rh(c.nq * c.hd, 2.f), hk = rh(c.nkv * c.hd, 1.f), hv = rh(c.nkv * c.hd, 1.f);
        q.memcpy(b.qi, hq.data(), hq.size() * 2).wait();
        q.memcpy(b.ki, hk.data(), hk.size() * 2).wait();
        q.memcpy(b.vi, hv.data(), hv.size() * 2).wait();
        for (int k = 0; k < kNKer; ++k) {
            AttnProfileData prof;
            call(k, q, b.qi, b.ki, b.vi, b.kc, b.vc, b.y, b.part, c, &prof).wait();   // warm (kernel compile / caches)
            prof.reset();
            const int reps = 50;
            for (int i = 0; i < reps; ++i) call(k, q, b.qi, b.ki, b.vi, b.kc, b.vc, b.y, b.part, c, &prof);
            q.wait();
            std::printf("  %-5s nq %2u nkv %u hd %u ctx %6u : combine %7.1f us   partial %8.1f us   append %5.1f us\n", kKer[k],
                        c.nq, c.nkv, c.hd, c.ctx, prof.combine_ms() * 1e3 / reps, prof.partial_ms() * 1e3 / reps,
                        prof.append_ms() * 1e3 / reps);
        }
    }
    b.free(q);
    return 0;
}

// --xmx: every kernel against a host fp64 reference (the same fp16 inputs; softmax and the P.V sum in double). The cache is
// re-uploaded per case because the XMX append zeroes the 15 rows after the token, which a later, longer case would read.
const uint32_t kXmxCtx[] = {1, 2, 15, 16, 17, 63, 64, 65, 127, 128, 1060, 2048, 4095, 4096, 4097, 8192, 16384, 33000, 40944};
constexpr double kXmxRelBound = 2e-3;

int run_xmx() {
    sycl::queue q{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
    std::mt19937 r(23);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto rh = [&](size_t n, float s) { std::vector<sycl::half> v(n); for (auto& x : v) x = sycl::half(nd(r) * s); return v; };
    const size_t kvn = size_t(4) * MAXC * HD;
    const auto hkc = rh(kvn, 1.f), hvc = rh(kvn, 1.f);
    Bufs b(q);
    int fail = 0;
    std::printf("%-6s %-4s %-6s | %-22s | %-22s | %-22s | %s\n", "ctx", "nq", "nkv", "vec abs / rel", "tiled abs / rel",
                "xmx abs / rel", "max|xmx - vec|");
    for (uint32_t nq : {24u, 16u}) {
        const uint32_t nkv = nq == 24 ? 4 : 2, gqa = nq / nkv;
        for (uint32_t ctx : kXmxCtx) {
            const Case c{nq, nkv, HD, ctx};
            const auto hq = rh(nq * HD, 2.f), hk = rh(nkv * HD, 1.f), hv = rh(nkv * HD, 1.f);
            q.memcpy(b.kc, hkc.data(), kvn * 2); q.memcpy(b.vc, hvc.data(), kvn * 2);
            q.memcpy(b.qi, hq.data(), hq.size() * 2); q.memcpy(b.ki, hk.data(), hk.size() * 2);
            q.memcpy(b.vi, hv.data(), hv.size() * 2).wait();
            // host fp64 reference: rows [0, ctx-1) from the cache, row ctx-1 = the appended token
            std::vector<double> ref(size_t(nq) * HD);
            std::vector<double> s(ctx);
            for (uint32_t h = 0; h < nq; ++h) {
                const uint32_t kv = h / gqa;
                const sycl::half* qh = hq.data() + size_t(h) * HD;
                auto krow = [&](uint32_t i) { return i + 1 < ctx ? hkc.data() + (size_t(kv) * MAXC + i) * HD : hk.data() + size_t(kv) * HD; };
                auto vrow = [&](uint32_t i) { return i + 1 < ctx ? hvc.data() + (size_t(kv) * MAXC + i) * HD : hv.data() + size_t(kv) * HD; };
                double m = -1e300;
                for (uint32_t i = 0; i < ctx; ++i) {
                    const sycl::half* kr = krow(i);
                    double d = 0;
                    for (uint32_t e = 0; e < HD; ++e) d += double(float(qh[e])) * double(float(kr[e]));
                    s[i] = d / 16.0;   // 1/sqrt(256)
                    m = std::max(m, s[i]);
                }
                double l = 0;
                std::vector<double> acc(HD, 0.0);
                for (uint32_t i = 0; i < ctx; ++i) {
                    const double p = std::exp(s[i] - m);
                    l += p;
                    const sycl::half* vr = vrow(i);
                    for (uint32_t e = 0; e < HD; ++e) acc[e] += p * double(float(vr[e]));
                }
                for (uint32_t e = 0; e < HD; ++e) ref[size_t(h) * HD + e] = acc[e] / l;
            }
            double maxref = 0;
            for (double v : ref) maxref = std::max(maxref, std::abs(v));
            double abs_err[3] = {0, 0, 0}, xv = 0;
            std::vector<sycl::half> hy[3];
            for (int k = 0; k < 3; ++k) {
                q.memset(b.part, 0, kPn * sizeof(float)).wait();
                call(k, q, b.qi, b.ki, b.vi, b.kc, b.vc, b.y, b.part, c).wait();
                hy[k].resize(size_t(nq) * HD);
                q.memcpy(hy[k].data(), b.y, hy[k].size() * 2).wait();
                for (size_t i = 0; i < ref.size(); ++i)
                    abs_err[k] = std::max(abs_err[k], std::abs(double(float(hy[k][i])) - ref[i]));
            }
            for (size_t i = 0; i < ref.size(); ++i)
                xv = std::max(xv, std::abs(double(float(hy[2][i])) - double(float(hy[0][i]))));
            const double rel[3] = {abs_err[0] / maxref, abs_err[1] / maxref, abs_err[2] / maxref};
            const bool ok = rel[2] <= kXmxRelBound;
            fail += !ok;
            std::printf("%-6u %-4u %-6u | %9.3g / %-10.3g | %9.3g / %-10.3g | %9.3g / %-10.3g | %.3g  (max|y_ref| %.3g)%s\n",
                        ctx, nq, nkv, abs_err[0], rel[0], abs_err[1], rel[1], abs_err[2], rel[2], xv, maxref,
                        ok ? "" : "  <-- FAIL");
        }
    }
    b.free(q);
    std::printf("%s (XMX relative error bound %.1e on every case)\n", fail ? "FAIL" : "PASS", kXmxRelBound);
    return fail ? 1 : 0;
}

// --rows / --rows-bench: P4 B26. A group = (geometry, the rows' context lengths, how their kernel kinds are chosen).
// kinds: 0 = as the model row steps: XMX from 4096 (and start_pos + 16 <= max_ctx), else vec (24/4, the 27B) or the
// dispatcher (16/2, the 35B); 1 = every row kVec; 2 = every row kAuto (the dispatcher = tiled at hd 256); 3 = every row
// kXmx (a row past MAXC - 16 fails the XMX shape gate and takes the tiled kernel inside the entry point).
struct RowsGroup { uint32_t nq, nkv, kinds; std::vector<uint32_t> ctx; };
const std::vector<RowsGroup> kRowsGroups = {
    {24, 4, 0, {1, 2, 63, 64, 65, 127, 1060, 4095, 4096, 4097, 8191, 8192, 16384, 24000, 33000, 40950}},
    {24, 4, 0, {4097, 33000}},
    {24, 4, 0, {100, 2000, 4000, 4095, 33000, 33000, 33000, 16384}},
    {24, 4, 1, {1, 4097, 33000, 8192}},
    {24, 4, 2, {1, 4097, 33000, 8192}},
    {24, 4, 3, {4096, 40950, 33000, 16, 15}},
    {16, 2, 0, {1, 2, 63, 64, 65, 127, 1060, 4095, 4096, 4097, 8191, 8192, 16384, 24000, 33000, 40950}},
    {16, 2, 0, {4097, 33000}},
    {16, 2, 0, {100, 2000, 4000, 4095, 33000, 33000, 33000, 16384}},
    {16, 2, 1, {1, 4097, 33000, 8192}},
    {16, 2, 2, {1, 4097, 33000, 8192}},
    {16, 2, 3, {4096, 40950, 33000, 16, 15}},
};
FaDecodeKind rows_kind(const RowsGroup& g, uint32_t ctx) {
    if (g.kinds == 1) return FaDecodeKind::kVec;
    if (g.kinds == 2) return FaDecodeKind::kAuto;
    if (g.kinds == 3) return FaDecodeKind::kXmx;
    const uint32_t start_pos = ctx - 1;
    if (ctx >= 4096 && start_pos + 16 <= MAXC) return FaDecodeKind::kXmx;   // q35::xmx_decode's rule (default MIN 4096)
    return g.nq == 24 ? FaDecodeKind::kVec : FaDecodeKind::kAuto;
}
const char* kind_name(FaDecodeKind k) { return k == FaDecodeKind::kXmx ? "xmx" : k == FaDecodeKind::kVec ? "vec" : "auto"; }

// The single-row entry point for a kind (what the row steps call per row today).
sycl::event single_call(sycl::queue& q, FaDecodeKind k, const sycl::half* qi, const sycl::half* ki, const sycl::half* vi,
                        sycl::half* kc, sycl::half* vc, sycl::half* y, float* part, uint32_t start_pos, uint32_t nq, uint32_t nkv) {
    if (k == FaDecodeKind::kXmx) return full_attention_fa2_decode_xmx(q, qi, ki, vi, kc, vc, y, part, start_pos, nq, nkv, HD, MAXC);
    if (k == FaDecodeKind::kVec) return full_attention_fa2_decode_vec(q, qi, ki, vi, kc, vc, y, part, start_pos, nq, nkv, HD, MAXC);
    return full_attention_fa2_decode(q, qi, ki, vi, kc, vc, y, part, start_pos, nq, nkv, HD, MAXC);
}

struct RowsBufs {   // kFaDecodeRowsMax rows, each on its own cache
    static constexpr uint32_t R = kFaDecodeRowsMax;
    size_t kvn;
    sycl::half *kc[R], *vc[R], *kmaster, *vmaster, *qi, *ki, *vi, *y_ref, *y_rows;
    float *part_single, *part_rows;
    void* table;
    uint64_t row_floats;
    RowsBufs(sycl::queue& q, uint64_t rf) : kvn(size_t(4) * MAXC * HD), row_floats(rf) {
        table = sycl::malloc_device<char>(kFaDecodeRowsTableBytes, q);
        for (uint32_t r = 0; r < R; ++r) { kc[r] = sycl::malloc_device<sycl::half>(kvn, q); vc[r] = sycl::malloc_device<sycl::half>(kvn, q); }
        kmaster = sycl::malloc_device<sycl::half>(kvn, q);
        vmaster = sycl::malloc_device<sycl::half>(kvn, q);
        qi = sycl::malloc_device<sycl::half>(size_t(R) * 24 * HD, q);
        ki = sycl::malloc_device<sycl::half>(size_t(R) * 4 * HD, q);
        vi = sycl::malloc_device<sycl::half>(size_t(R) * 4 * HD, q);
        y_ref  = sycl::malloc_device<sycl::half>(size_t(R) * 24 * HD, q);
        y_rows = sycl::malloc_device<sycl::half>(size_t(R) * 24 * HD, q);
        part_single = sycl::malloc_device<float>(kPn, q);
        part_rows   = sycl::malloc_device<float>(size_t(R) * row_floats, q);
    }
    void reset(sycl::queue& q, uint32_t n) {   // rows 0..n-1's caches back to the master content
        for (uint32_t r = 0; r < n; ++r) { q.memcpy(kc[r], kmaster, kvn * 2); q.memcpy(vc[r], vmaster, kvn * 2); }
        q.wait();
    }
    void free(sycl::queue& q) {
        for (uint32_t r = 0; r < R; ++r) { sycl::free(kc[r], q); sycl::free(vc[r], q); }
        for (void* p : {static_cast<void*>(kmaster), static_cast<void*>(vmaster), static_cast<void*>(qi), static_cast<void*>(ki),
                        static_cast<void*>(vi), static_cast<void*>(y_ref), static_cast<void*>(y_rows),
                        static_cast<void*>(part_single), static_cast<void*>(part_rows), table})
            sycl::free(p, q);
    }
    // the group's plan (single layer: layer_stride 0, slot 0); false = it would run the per-row calls (vacuous here)
    bool plan(sycl::queue& q, const std::vector<FaDecodeRow>& rows, uint32_t nq, uint32_t nkv, FaDecodeRowsPlan& p) {
        return fa2_decode_rows_plan(q, rows.data(), uint32_t(rows.size()), nq, nkv, HD, row_floats, table, p);
    }
};

// The cache rows an append may touch: [start_pos, start_pos + 16) of every kv head (XMX zeroes the 15 after the token).
std::vector<sycl::half> cache_window(sycl::queue& q, const sycl::half* c, uint32_t nkv, uint32_t start_pos) {
    const uint32_t n = std::min<uint32_t>(16, MAXC - start_pos);
    std::vector<sycl::half> w(size_t(nkv) * n * HD);
    for (uint32_t kv = 0; kv < nkv; ++kv)
        q.memcpy(w.data() + size_t(kv) * n * HD, c + (size_t(kv) * MAXC + start_pos) * HD, size_t(n) * HD * 2);
    q.wait();
    return w;
}

int run_rows(bool bench) {
    sycl::queue q{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
    std::mt19937 r(26);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto rh = [&](size_t n, float s) { std::vector<sycl::half> v(n); for (auto& x : v) x = sycl::half(nd(r) * s); return v; };
    const uint64_t row_floats = std::max(fa2_decode_rows_partials_floats(24, 4, HD, MAXC), fa2_decode_rows_partials_floats(16, 2, HD, MAXC));
    RowsBufs b(q, row_floats);
    { auto hk = rh(b.kvn, 1.f), hv = rh(b.kvn, 1.f); q.memcpy(b.kmaster, hk.data(), b.kvn * 2).wait(); q.memcpy(b.vmaster, hv.data(), b.kvn * 2).wait(); }
    std::printf("partials slice %llu floats per row (%.1f MB), %u rows\n", (unsigned long long)row_floats, row_floats * 4e-6, RowsBufs::R);
    int fail = 0;
    if (bench) {
        const RowsGroup groups[] = {{24, 4, 0, std::vector<uint32_t>(4, 16384)}, {24, 4, 0, std::vector<uint32_t>(4, 33000)},
                                    {24, 4, 0, std::vector<uint32_t>(8, 8192)},  {24, 4, 0, std::vector<uint32_t>(16, 4096)},
                                    {24, 4, 0, std::vector<uint32_t>(16, 2048)}, {24, 4, 0, std::vector<uint32_t>(4, 1024)},
                                    {16, 2, 0, std::vector<uint32_t>(4, 16384)}, {16, 2, 0, std::vector<uint32_t>(4, 33000)},
                                    {16, 2, 0, std::vector<uint32_t>(8, 8192)},  {16, 2, 0, std::vector<uint32_t>(16, 4096)},
                                    {16, 2, 0, std::vector<uint32_t>(16, 2048)}, {16, 2, 0, std::vector<uint32_t>(4, 1024)},
                                    // the tiled kernel (the 35B below 4096) at every group size, and vec on that shape
                                    {16, 2, 2, std::vector<uint32_t>(1, 2048)},  {16, 2, 2, std::vector<uint32_t>(4, 2048)},
                                    {16, 2, 2, std::vector<uint32_t>(8, 2048)},  {16, 2, 2, std::vector<uint32_t>(16, 1024)},
                                    {16, 2, 2, std::vector<uint32_t>(16, 3072)}, {16, 2, 2, std::vector<uint32_t>(16, 512)},
                                    {16, 2, 1, std::vector<uint32_t>(16, 2048)}, {24, 4, 2, std::vector<uint32_t>(16, 2048)},
                                    {24, 4, 2, std::vector<uint32_t>(4, 2048)}};
        std::printf("%-6s %-4s %-3s %-6s | %12s | %12s | %s\n", "geom", "kind", "G", "ctx", "per-row us", "rows us", "speedup");
        for (const RowsGroup& g : groups) {
            const uint32_t G = uint32_t(g.ctx.size()), N_q = g.nq * HD, N_kv = g.nkv * HD;
            std::vector<FaDecodeRow> rows(G);
            for (uint32_t i = 0; i < G; ++i)
                rows[i] = {b.kc[i], b.vc[i], 0, g.ctx[i] - 1, MAXC, rows_kind(g, g.ctx[i])};
            FaDecodeRowsPlan plan;
            if (!b.plan(q, rows, g.nq, g.nkv, plan)) { std::printf("plan not batched?\n"); return 1; }
            auto per_row = [&] {
                for (uint32_t i = 0; i < G; ++i)
                    single_call(q, rows[i].kind, b.qi + size_t(i) * N_q, b.ki + size_t(i) * N_kv, b.vi + size_t(i) * N_kv, b.kc[i], b.vc[i],
                                b.y_ref + size_t(i) * N_q, b.part_single, rows[i].start_pos, g.nq, g.nkv);
            };
            auto batched = [&] {
                full_attention_fa2_decode_rows(q, plan, b.qi, b.ki, b.vi, b.y_rows, b.part_rows, 0);
            };
            const int reps = 20;
            double us[2] = {0, 0};
            for (int which = 0; which < 2; ++which) {
                for (int i = 0; i < 3; ++i) { if (which) batched(); else per_row(); }
                q.wait();
                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < reps; ++i) { if (which) batched(); else per_row(); }
                q.wait();
                us[which] = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / reps;
            }
            std::printf("%2u/%-3u %-4s %-3u %-6u | %12.1f | %12.1f | %.2fx\n", g.nq, g.nkv, kind_name(rows[0].kind), G, g.ctx[0], us[0], us[1], us[0] / us[1]);
        }
        b.free(q);
        return 0;
    }
    for (const RowsGroup& g : kRowsGroups) {
        const uint32_t G = uint32_t(g.ctx.size()), N_q = g.nq * HD, N_kv = g.nkv * HD;
        auto hq = rh(size_t(G) * N_q, 2.f), hk = rh(size_t(G) * N_kv, 1.f), hv = rh(size_t(G) * N_kv, 1.f);
        q.memcpy(b.qi, hq.data(), hq.size() * 2); q.memcpy(b.ki, hk.data(), hk.size() * 2); q.memcpy(b.vi, hv.data(), hv.size() * 2).wait();
        std::vector<FaDecodeRow> rows(G);
        for (uint32_t i = 0; i < G; ++i) rows[i] = {b.kc[i], b.vc[i], 0, g.ctx[i] - 1, MAXC, rows_kind(g, g.ctx[i])};
        FaDecodeRowsPlan plan;
        const bool batched = b.plan(q, rows, g.nq, g.nkv, plan);
        // reference: every row alone through its single-row entry point
        std::vector<std::vector<float>> ref_part(G);
        std::vector<std::vector<sycl::half>> ref_k(G), ref_v(G);
        b.reset(q, G);
        for (uint32_t i = 0; i < G; ++i) {
            q.memset(b.part_single, 0, kPn * sizeof(float)).wait();
            single_call(q, rows[i].kind, b.qi + size_t(i) * N_q, b.ki + size_t(i) * N_kv, b.vi + size_t(i) * N_kv, b.kc[i], b.vc[i],
                        b.y_ref + size_t(i) * N_q, b.part_single, rows[i].start_pos, g.nq, g.nkv).wait();
            ref_part[i].resize(row_floats);
            q.memcpy(ref_part[i].data(), b.part_single, row_floats * sizeof(float)).wait();
            ref_k[i] = cache_window(q, b.kc[i], g.nkv, rows[i].start_pos);
            ref_v[i] = cache_window(q, b.vc[i], g.nkv, rows[i].start_pos);
        }
        std::vector<sycl::half> y_ref(size_t(G) * N_q), y_rows(size_t(G) * N_q);
        q.memcpy(y_ref.data(), b.y_ref, y_ref.size() * 2).wait();
        // the rows launch
        b.reset(q, G);
        q.memset(b.part_rows, 0, size_t(G) * row_floats * sizeof(float)).wait();
        full_attention_fa2_decode_rows(q, plan, b.qi, b.ki, b.vi, b.y_rows, b.part_rows, 0).wait();
        q.memcpy(y_rows.data(), b.y_rows, y_rows.size() * 2).wait();
        std::vector<float> part_rows(size_t(G) * row_floats);
        q.memcpy(part_rows.data(), b.part_rows, part_rows.size() * sizeof(float)).wait();
        size_t gdiff = 0;
        std::string kinds;
        for (uint32_t i = 0; i < G; ++i) {
            size_t dy = 0, dp = 0, dk = 0, dv = 0;
            const auto* a = reinterpret_cast<const char*>(y_ref.data() + size_t(i) * N_q);
            const auto* c = reinterpret_cast<const char*>(y_rows.data() + size_t(i) * N_q);
            for (size_t j = 0; j < size_t(N_q) * 2; ++j) dy += a[j] != c[j];
            const auto* pa = reinterpret_cast<const char*>(ref_part[i].data());
            const auto* pc = reinterpret_cast<const char*>(part_rows.data() + size_t(i) * row_floats);
            for (size_t j = 0; j < row_floats * sizeof(float); ++j) dp += pa[j] != pc[j];
            const auto wk = cache_window(q, b.kc[i], g.nkv, rows[i].start_pos), wv = cache_window(q, b.vc[i], g.nkv, rows[i].start_pos);
            const auto *ka = reinterpret_cast<const char*>(ref_k[i].data()), *kc = reinterpret_cast<const char*>(wk.data());
            const auto *va = reinterpret_cast<const char*>(ref_v[i].data()), *vc = reinterpret_cast<const char*>(wv.data());
            for (size_t j = 0; j < wk.size() * 2; ++j) { dk += ka[j] != kc[j]; dv += va[j] != vc[j]; }
            if (dy || dp || dk || dv)
                std::printf("   row %u ctx %u %s: y bytes differ %zu, partial bytes %zu, K cache %zu, V cache %zu\n", i, g.ctx[i],
                            kind_name(rows[i].kind), dy, dp, dk, dv);
            gdiff += dy + dp + dk + dv;
            kinds += (i ? "," : "") + std::string(kind_name(rows[i].kind));
        }
        size_t nz = 0;   // (a guard against a vacuous pass: the reference y must hold real values)
        for (const sycl::half& v : y_ref) nz += float(v) != 0.f;
        std::printf("[%s] %2u/%u G=%2u kinds %u (%s): %zu bytes differ (reference y: %zu of %zu values non-zero%s)\n",
                    (gdiff || nz == 0 || !batched) ? "FAIL" : " ok ", g.nq, g.nkv, G, g.kinds, kinds.c_str(), gdiff, nz, y_ref.size(),
                    batched ? "" : "; PLAN NOT BATCHED: the per-row calls ran");
        fail += gdiff != 0 || nz == 0 || !batched;
    }
    b.free(q);
    std::printf("%s (%zu groups)\n", fail ? "FAIL" : "PASS", kRowsGroups.size());
    return fail ? 1 : 0;
}

// Runs `self --dump` under each environment setting (sets[0] is the reference) and compares the bytes per case / kernel.
int compare_settings(const std::string& self, const char* const* sets, int nsets, const char* tag) {
    const char* tmp = std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp";
    int fail = 0;
    std::string ref;
    for (int s = 0; s < nsets; ++s) {
        const std::string file = std::string(tmp) + "/fa2_" + tag + "_test_" + std::to_string(s) + ".bin";
        const std::string cmd = std::string(sets[s]) + " \"" + self + "\" --dump \"" + file + "\"";
        if (std::system(cmd.c_str()) != 0) { std::printf("[FAIL] %s\n", cmd.c_str()); return 1; }
        std::ifstream f(file, std::ios::binary);
        std::string b((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (s == 0) { ref = b; std::printf("reference (%s): %zu bytes\n", sets[s], b.size()); continue; }
        size_t diff = 0, off = 0;
        for (const Case& c : kCases)
            for (int k = 0; k < kNKer; ++k) {   // per case and kernel: y bytes, then partial bytes
                const size_t ny = size_t(c.nq) * c.hd * 2, np = kPn * sizeof(float);
                size_t dy = 0, dp = 0;
                for (size_t i = off; i < off + ny && i < b.size() && i < ref.size(); ++i) dy += b[i] != ref[i];
                for (size_t i = off + ny; i < off + ny + np && i < b.size() && i < ref.size(); ++i) dp += b[i] != ref[i];
                if (dy || dp)
                    std::printf("   %s nq %u nkv %u hd %u ctx %u: y bytes differ %zu, partial bytes differ %zu\n", kKer[k], c.nq,
                                c.nkv, c.hd, c.ctx, dy, dp);
                diff += dy + dp;
                off += ny + np;
            }
        const bool ok = b.size() == ref.size() && diff == 0;
        std::printf("[%s] %s: %zu bytes, %zu differ from the reference\n", ok ? " ok " : "FAIL", sets[s], b.size(), diff);
        fail += !ok;
    }
    std::printf(fail ? "FAIL\n" : "PASS\n");
    return fail ? 1 : 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc > 2 && std::string(argv[1]) == "--dump") return run(argv[2], false);
    if (argc > 1 && std::string(argv[1]) == "--bench") return run(nullptr, true);
    if (argc > 1 && std::string(argv[1]) == "--cbench") return run_cbench();
    if (argc > 1 && std::string(argv[1]) == "--xmx") return run_xmx();
    if (argc > 1 && std::string(argv[1]) == "--rows") return run_rows(false);
    if (argc > 1 && std::string(argv[1]) == "--rows-bench") return run_rows(true);
    const std::string self = argv[0];
    if (argc > 1 && std::string(argv[1]) == "--combine") {
        const char* sets[] = {"IE_FA2_COMBINE_SPLIT=0", "IE_FA2_COMBINE_SPLIT=1", "IE_FA2_COMBINE_SPLIT=2",
                              "IE_FA2_COMBINE_SPLIT=4", "IE_FA2_COMBINE_SPLIT=8", "IE_FA2_COMBINE_SPLIT=16"};
        return compare_settings(self, sets, 6, "combine");
    }
    const char* sets[] = {"IE_FA2_VEC_PF=0 IE_FA2_TILE_PF=0", "IE_FA2_VEC_PF=8 IE_FA2_TILE_PF=1",
                          "IE_FA2_VEC_PF=16 IE_FA2_TILE_PF=1", "IE_FA2_VEC_PF=32 IE_FA2_TILE_PF=1"};
    return compare_settings(self, sets, 4, "pf");
}
