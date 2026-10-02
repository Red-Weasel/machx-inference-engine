// tests/unit/gemv_q8_soa_nd_test.cpp -- P4 B37 (2) / (3): the T = 1 int-dot GEMVs over Q8_0-SoA weights with the native
// integer dot must write the SAME bytes as their v1 kernels (gemv_q8dot.cpp), called directly:
//   gemv_q8_0_soa_q8_nd   (gemv_q8_soa_nd.cpp)    vs  gemv_q8_0_soa_q8    -- SLM-staged activation; the 27B split's and
//                                                                             the 35B crown's T == 1 dense projections
//   gemv_q8_0_soa_q8_g_nd (gemv_q8_soa_g_nd.cpp)  vs  gemv_q8_0_soa_q8_g  -- activation from global; Flash-Next's T == 1
//                                                                             projections, shared expert and LM head
// (with IE_GEMV_SMALLK set both _g entries take v1's split-K kernel for K <= 6144: the run says so, unset it to test
// _g_nd there). Random Q8_0-SoA weights (int8 qs over the full range; fp16 scales with subnormal, zero and negative ones) and a
// block_q8_1x activation (2 % all-zero blocks) over the T == 1 shapes of the three models (from the model code) plus edge
// shapes (one K-block, lanes past the last block, N not a multiple of the work-group's columns). Per (kernel, shape):
// v2 == v1 bit for bit, every output written, v1 and v2 within a host fp64 bound on every output.
//   gemv_q8_soa_nd_test           -- every case; PASS = all
//   gemv_q8_soa_nd_test --small   -- the edge shapes only (a CPU device: build the kernels with IE_DP4A_PORTABLE=1)
//   gemv_q8_soa_nd_test --bench   -- device time per launch (event profiling, median of 50 after 5 warm-ups) of v1 and
//                                    v2 per kernel on the model shapes, with the weight stream's GB/s
#include "ie/gemv_q8_soa_v2.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

namespace {

using Gemv = sycl::event (*)(sycl::queue&, const void*, const int8_t*, const uint16_t*, sycl::half*, uint32_t, uint32_t,
                             const std::vector<sycl::event>&);
struct Pair { const char* name; Gemv v1, v2; };

std::vector<Pair> pairs() {
    return {
        {"soa_q8 (SLM)", &ie::gemv_q8_0_soa_q8, &ie::gemv_q8_0_soa_q8_nd},
        {"soa_q8_g",     &ie::gemv_q8_0_soa_q8_g, &ie::gemv_q8_0_soa_q8_g_nd},
    };
}

struct Shape { const char* name; uint32_t K, N; bool small; };

std::vector<Shape> shapes() {
    return {
        {"27b-ffn-gu",   5120, 17408, false},
        {"27b-ffn-down", 17408, 5120, false},
        {"27b-attn-q",   5120, 12288, false},
        {"27b-kv",       5120, 1024,  false},
        {"35b-proj",     2048, 4096,  false},
        {"35b-o",        4096, 2048,  false},
        {"35b-shexp-up", 2048, 512,   false},
        {"35b-shexp-dn", 512,  2048,  false},
        {"fn-dn-qkv",    2560, 10240, false},
        {"fn-dn-gate",   2560, 6144,  false},
        {"fn-dn-out",    6144, 2560,  false},
        {"fn-shexp-gu",  2560, 640,   false},
        {"fn-shexp-dn",  640,  2560,  false},
        {"fn-lm-head",   2560, 248320, false},
        {"head-like",    2048, 32000, false},
        {"edge-k32",     32,   100,   true},
        {"edge-k544",    544,  1000,  true},
        {"edge-k96",     96,   37,    true},
        {"edge-k2048",   2048, 161,   true},
    };
}

uint16_t f16_bits(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }
double   f16_val(uint16_t h) { return double(float(sycl::bit_cast<sycl::half>(h))); }

struct Case {
    uint32_t K, N;
    std::vector<int8_t> qs; std::vector<uint16_t> d; std::vector<ie::block_q8_1x> x;
    int8_t* dqs = nullptr; uint16_t* dd = nullptr; ie::block_q8_1x* dx = nullptr;
    void make(sycl::queue& q, uint32_t K_, uint32_t N_, std::mt19937& rng) {
        K = K_; N = N_;
        std::uniform_int_distribution<int> q7(-127, 127), pct(0, 99), sub(1, 0x3ff);
        std::uniform_real_distribution<float> dw(2e-4f, 2e-2f), dxr(1e-3f, 5e-2f);
        qs.resize(uint64_t(N) * K);
        for (uint64_t i = 0; i < qs.size(); i += 4) { const uint32_t r = uint32_t(rng()); std::memcpy(qs.data() + i, &r, 4); }
        d.resize(uint64_t(N) * (K / 32));
        for (auto& v : d) {
            const int p = pct(rng);
            v = p < 2 ? uint16_t(sub(rng)) : p < 4 ? uint16_t(0) : p < 6 ? f16_bits(-dw(rng)) : f16_bits(dw(rng));
        }
        x.resize(K / 32);
        for (auto& b : x) {
            if (pct(rng) < 2) { b.d = 0.f; b.s = 0.f; std::memset(b.qs, 0, sizeof(b.qs)); continue; }
            b.d = dxr(rng);
            int sum = 0;
            for (auto& v : b.qs) { v = int8_t(q7(rng)); sum += v; }
            b.s = b.d * float(sum);
        }
        dqs = sycl::malloc_device<int8_t>(qs.size(), q);
        dd = sycl::malloc_device<uint16_t>(d.size(), q);
        dx = sycl::malloc_device<ie::block_q8_1x>(x.size(), q);
        q.memcpy(dqs, qs.data(), qs.size());
        q.memcpy(dd, d.data(), d.size() * 2);
        q.memcpy(dx, x.data(), x.size() * sizeof(ie::block_q8_1x)).wait();
    }
    void release(sycl::queue& q) { sycl::free(dqs, q); sycl::free(dd, q); sycl::free(dx, q); }
    double ref(uint32_t n, double& abs_sum) const {
        const uint32_t nb = K / 32;
        double s = 0, a = 0;
        for (uint32_t b = 0; b < nb; ++b) {
            long id = 0;
            for (int k = 0; k < 32; ++k) id += long(qs[uint64_t(n) * K + b * 32 + k]) * x[b].qs[k];
            const double p = f16_val(d[uint64_t(n) * nb + b]) * double(x[b].d) * double(id);
            s += p; a += std::fabs(p);
        }
        abs_sum = a;
        return s;
    }
};

void note_smallk() {
    if (std::getenv("IE_GEMV_SMALLK"))
        std::printf("note: IE_GEMV_SMALLK is set: for K <= 6144 both soa_q8_g entries run v1's split-K kernel "
                    "(unset it to test _g_nd there)\n");
}

bool check(sycl::queue& q, const Pair& p, const Shape& s, const Case& c) {
    auto* y1 = sycl::malloc_device<sycl::half>(s.N, q);
    auto* y2 = sycl::malloc_device<sycl::half>(s.N, q);
    q.memset(y1, 0xFF, s.N * 2);
    q.memset(y2, 0xFF, s.N * 2);
    p.v1(q, c.dx, c.dqs, c.dd, y1, s.K, s.N, {});
    p.v2(q, c.dx, c.dqs, c.dd, y2, s.K, s.N, {});
    q.wait();
    std::vector<uint16_t> h1(s.N), h2(s.N);
    q.memcpy(h1.data(), y1, s.N * 2);
    q.memcpy(h2.data(), y2, s.N * 2).wait();
    sycl::free(y1, q); sycl::free(y2, q);
    uint64_t diff = 0, unw = 0, bad = 0;
    for (uint32_t n = 0; n < s.N; ++n) {
        unw += (h1[n] == 0xFFFFu) + (h2[n] == 0xFFFFu);
        if (h1[n] != h2[n]) {
            if (diff < 3) std::printf("    differ at col %u: v1 0x%04x v2 0x%04x\n", n, h1[n], h2[n]);
            ++diff;
        }
        double asum;
        const double ref = c.ref(n, asum);
        const double tol = std::fabs(ref) / 1024.0 + asum * 1e-6 + 1e-6;
        for (uint16_t got : {h1[n], h2[n]}) {
            const double g = f16_val(got);
            const bool ok = (std::fabs(ref) + tol >= 65520.0) ? (std::isinf(g) || std::fabs(g - ref) <= tol)
                                                              : std::fabs(g - ref) <= tol;
            bad += !ok;
        }
    }
    const bool ok = diff == 0 && unw == 0 && bad == 0;
    std::printf("  %-14s %-13s K=%-5u N=%-6u  v2 vs v1: %llu of %u differ; unwritten %llu; host bound misses %llu  %s\n",
                p.name, s.name, s.K, s.N, (unsigned long long)diff, s.N, (unsigned long long)unw,
                (unsigned long long)bad, ok ? "PASS" : "FAIL");
    return ok;
}

int run_check(bool small_only) {
    sycl::queue q{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    note_smallk();
    std::mt19937 rng(3737);
    int fails = 0, n = 0;
    for (const Shape& s : shapes()) {
        if (small_only && !s.small) continue;
        Case c;
        c.make(q, s.K, s.N, rng);
        for (const Pair& p : pairs()) { fails += check(q, p, s, c) ? 0 : 1; ++n; }
        c.release(q);
    }
    if (fails) { std::printf("gemv_q8_soa_nd_test: FAIL (%d of %d cases)\n", fails, n); return 1; }
    std::printf("gemv_q8_soa_nd_test: ALL PASS (%d cases)\n", n);
    return 0;
}

double median_us(sycl::queue& q, const std::function<sycl::event()>& launch) {
    for (int i = 0; i < 5; ++i) launch();
    q.wait();
    std::vector<double> t;
    for (int i = 0; i < 50; ++i) {
        sycl::event ev = launch();
        ev.wait();
        t.push_back(double(ev.get_profiling_info<sycl::info::event_profiling::command_end>() -
                           ev.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-3);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

int run_bench() {
    sycl::queue q{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order{},
                                                                sycl::property::queue::enable_profiling{}}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    note_smallk();
    std::mt19937 rng(11);
    for (const Shape& s : shapes()) {
        if (s.small) continue;
        Case c;
        c.make(q, s.K, s.N, rng);
        auto* y = sycl::malloc_device<sycl::half>(s.N, q);
        const double wbytes = double(s.N) * s.K * (1.0 + 2.0 / 32);
        for (const Pair& p : pairs()) {
            const double t1 = median_us(q, [&] { return p.v1(q, c.dx, c.dqs, c.dd, y, s.K, s.N, {}); });
            const double t2 = median_us(q, [&] { return p.v2(q, c.dx, c.dqs, c.dd, y, s.K, s.N, {}); });
            std::printf("  %-14s %-13s K=%-5u N=%-6u  v1 %8.1f us (%5.0f GB/s)   v2 %8.1f us (%5.0f GB/s)   %.2fx\n",
                        p.name, s.name, s.K, s.N, t1, wbytes / (t1 * 1e3), t2, wbytes / (t2 * 1e3), t1 / t2);
        }
        sycl::free(y, q);
        c.release(q);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "--bench") return run_bench();
    return run_check(mode == "--small");
}
