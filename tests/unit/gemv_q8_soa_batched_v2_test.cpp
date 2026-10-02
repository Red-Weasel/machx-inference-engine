// tests/unit/gemv_q8_soa_batched_v2_test.cpp -- P4 B37 (1): gemv_q8_0_soa_q8_batched_v2 / _batched_dual_v2
// (src/ops/gemv_q8_soa_batched_v2.cpp) must write the SAME bytes as the v1 kernels gemv_q8_0_soa_q8_batched / _dual
// (gemv_q8dot.cpp), called directly. Random Q8_0-SoA weights (int8 qs over the full range; fp16 scales with subnormal,
// zero and negative ones) and block_q8_1x activation rows (2 % all-zero blocks), for T = 1..5, 7, 8, 9, 12, 15, 16, 17
// and 33 rows (every T bucket, SLM and global activation paths, the 16-row chunking) over the shapes the rows step runs
// (27B split, 35B crown, Flash-Next; from the model code) plus edge shapes (one K-block, lanes past the last block, N
// not a multiple of the 16-column work-group). Per case: v2 == v1 bit for bit (single and dual), every output written,
// v1 and v2 within a host fp64 bound on sampled outputs, and -- the rows contract the models rely on -- for T <= 4
// every row of v2 == the T = 1 leaf gemv_q8_0_soa_q8 on that row.
//   gemv_q8_soa_batched_v2_test           -- every case; PASS = all
//   gemv_q8_soa_batched_v2_test --small   -- the edge shapes only (a CPU device: build the kernels with IE_DP4A_PORTABLE=1)
//   gemv_q8_soa_batched_v2_test --bench   -- device time per launch (event profiling, median of 30 after 5 warm-ups) of v1
//                                            and v2 at T = 2, 4, 8, 16 on the model shapes, single and dual
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

struct Shape { const char* name; uint32_t K, N; bool small; };

std::vector<Shape> shapes() {
    return {
        {"27b-ffn-gu",   5120, 17408, false},   // ffn gate / up (the 27B rows step's dual launch)
        {"27b-ffn-down", 17408, 5120, false},   // ffn_down (global activation path at every T)
        {"27b-attn-q",   5120, 12288, false},   // attn q + gate
        {"27b-kv",       5120, 1024,  false},   // k / v (dual)
        {"35b-proj",     2048, 4096,  false},   // a crown projection at H = 2048
        {"35b-shexp-dn", 512,  2048,  false},   // shared expert down
        {"fn-hc",        2560, 10240, false},   // Flash-Next K = 2560 class
        {"fn-down",      10240, 2560, false},
        {"edge-k32",     32,   100,   true},    // one K-block: lanes 1..15 hold none; N % 16 != 0
        {"edge-k544",    544,  1000,  true},    // 17 K-blocks: lane 0 holds two
        {"edge-k96",     96,   37,    true},
        {"edge-k2048",   2048, 160,   true},
    };
}
const uint32_t kTs[] = {1, 2, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17, 33};

uint16_t f16_bits(float f) { return sycl::bit_cast<uint16_t>(sycl::half(f)); }
double   f16_val(uint16_t h) { return double(float(sycl::bit_cast<sycl::half>(h))); }

struct Weights {
    std::vector<int8_t> qs; std::vector<uint16_t> d;
    int8_t* dqs = nullptr; uint16_t* dd = nullptr;
    void make(sycl::queue& q, uint32_t K, uint32_t N, std::mt19937& rng) {
        std::uniform_int_distribution<int> pct(0, 99), sub(1, 0x3ff);
        std::uniform_real_distribution<float> dw(2e-4f, 2e-2f);
        qs.resize(uint64_t(N) * K);
        for (uint64_t i = 0; i < qs.size(); i += 4) { const uint32_t r = uint32_t(rng()); std::memcpy(qs.data() + i, &r, 4); }
        d.resize(uint64_t(N) * (K / 32));
        for (auto& v : d) {
            const int p = pct(rng);
            v = p < 2 ? uint16_t(sub(rng)) : p < 4 ? uint16_t(0) : p < 6 ? f16_bits(-dw(rng)) : f16_bits(dw(rng));
        }
        dqs = sycl::malloc_device<int8_t>(qs.size(), q);
        dd = sycl::malloc_device<uint16_t>(d.size(), q);
        q.memcpy(dqs, qs.data(), qs.size());
        q.memcpy(dd, d.data(), d.size() * 2).wait();
    }
    void release(sycl::queue& q) { sycl::free(dqs, q); sycl::free(dd, q); }
};

void make_act(std::vector<ie::block_q8_1x>& x, uint64_t n, std::mt19937& rng) {
    std::uniform_int_distribution<int> q7(-127, 127), pct(0, 99);
    std::uniform_real_distribution<float> dx(1e-3f, 5e-2f);
    x.resize(n);
    for (auto& b : x) {
        if (pct(rng) < 2) { b.d = 0.f; b.s = 0.f; std::memset(b.qs, 0, sizeof(b.qs)); continue; }
        b.d = dx(rng);
        int sum = 0;
        for (auto& v : b.qs) { v = int8_t(q7(rng)); sum += v; }
        b.s = b.d * float(sum);
    }
}

double reference(const Weights& w, const std::vector<ie::block_q8_1x>& x, uint32_t K, uint32_t t, uint32_t n,
                 double& abs_sum) {
    const uint32_t nb = K / 32;
    double s = 0, a = 0;
    for (uint32_t b = 0; b < nb; ++b) {
        const ie::block_q8_1x& xb = x[uint64_t(t) * nb + b];
        long id = 0;
        for (int k = 0; k < 32; ++k) id += long(w.qs[uint64_t(n) * K + b * 32 + k]) * xb.qs[k];
        const double p = f16_val(w.d[uint64_t(n) * nb + b]) * double(xb.d) * double(id);
        s += p; a += std::fabs(p);
    }
    abs_sum = a;
    return s;
}

struct Res { uint64_t diff = 0, unw = 0, bad = 0, leaf_diff = 0; };

// one (shape, T): single A, dual (A, B), v1 vs v2, the host bound, and the T = 1 leaf for T <= 4
bool run_case(sycl::queue& q, const Shape& s, uint32_t T, const Weights& A, const Weights& B, std::mt19937& rng) {
    const uint32_t K = s.K, N = s.N;
    std::vector<ie::block_q8_1x> x;
    make_act(x, uint64_t(T) * (K / 32), rng);
    auto* dx = sycl::malloc_device<ie::block_q8_1x>(x.size(), q);
    q.memcpy(dx, x.data(), x.size() * sizeof(ie::block_q8_1x)).wait();
    const uint64_t n_out = uint64_t(T) * N;
    sycl::half* y[6];   // v1 A, v2 A, v1 dual A, v1 dual B, v2 dual A, v2 dual B
    for (auto& p : y) { p = sycl::malloc_device<sycl::half>(n_out, q); q.memset(p, 0xFF, n_out * 2); }
    ie::gemv_q8_0_soa_q8_batched(q, dx, A.dqs, A.dd, y[0], K, N, T);
    ie::gemv_q8_0_soa_q8_batched_v2(q, dx, A.dqs, A.dd, y[1], K, N, T);
    ie::gemv_q8_0_soa_q8_batched_dual(q, dx, A.dqs, A.dd, y[2], B.dqs, B.dd, y[3], K, N, T);
    ie::gemv_q8_0_soa_q8_batched_dual_v2(q, dx, A.dqs, A.dd, y[4], B.dqs, B.dd, y[5], K, N, T);
    q.wait();
    std::vector<std::vector<uint16_t>> h(6, std::vector<uint16_t>(n_out));
    for (int i = 0; i < 6; ++i) q.memcpy(h[i].data(), y[i], n_out * 2);
    q.wait();
    Res r;
    for (uint64_t i = 0; i < n_out; ++i) {
        for (int k = 0; k < 6; ++k) r.unw += h[k][i] == 0xFFFFu;
        const bool d = h[0][i] != h[1][i] || h[2][i] != h[4][i] || h[3][i] != h[5][i];
        if (d && r.diff < 3)
            std::printf("    differ at row %llu col %llu: single v1 0x%04x v2 0x%04x  dual A v1 0x%04x v2 0x%04x  "
                        "dual B v1 0x%04x v2 0x%04x\n", (unsigned long long)(i / N), (unsigned long long)(i % N),
                        h[0][i], h[1][i], h[2][i], h[4][i], h[3][i], h[5][i]);
        r.diff += d;
    }
    std::uniform_int_distribution<uint64_t> pick(0, n_out - 1);
    for (int k = 0; k < 256; ++k) {
        const uint64_t i = pick(rng);
        const uint32_t t = uint32_t(i / N), n = uint32_t(i % N);
        double aa, ab;
        const double ra = reference(A, x, K, t, n, aa), rb = reference(B, x, K, t, n, ab);
        const auto off = [](uint16_t got, double ref, double asum) {
            const double tol = std::fabs(ref) / 1024.0 + asum * 1e-6 + 1e-6;
            if (std::fabs(ref) + tol >= 65520.0) return !(std::isinf(f16_val(got)) || std::fabs(f16_val(got) - ref) <= tol);
            return !(std::fabs(f16_val(got) - ref) <= tol);
        };
        r.bad += off(h[1][i], ra, aa) + off(h[4][i], ra, aa) + off(h[5][i], rb, ab) + off(h[0][i], ra, aa);
    }
    if (T <= 4) {   // the rows contract: row t of the batched GEMV == the T = 1 leaf on row t
        auto* yl = sycl::malloc_device<sycl::half>(N, q);
        std::vector<uint16_t> hl(N);
        for (uint32_t t = 0; t < T; ++t) {
            ie::gemv_q8_0_soa_q8(q, dx + uint64_t(t) * (K / 32), A.dqs, A.dd, yl, K, N);
            q.memcpy(hl.data(), yl, N * 2).wait();
            for (uint32_t n = 0; n < N; ++n) r.leaf_diff += hl[n] != h[1][uint64_t(t) * N + n];
        }
        sycl::free(yl, q);
    }
    for (auto& p : y) sycl::free(p, q);
    sycl::free(dx, q);
    const bool ok = r.diff == 0 && r.unw == 0 && r.bad == 0 && r.leaf_diff == 0;
    std::printf("  %-13s K=%-5u N=%-5u T=%-2u  v2 vs v1 (single + dual A/B): %llu of %llu differ; unwritten %llu; host "
                "bound misses %llu; rows vs T=1 leaf %s  %s\n", s.name, K, N, T, (unsigned long long)r.diff,
                (unsigned long long)n_out, (unsigned long long)r.unw, (unsigned long long)r.bad,
                T <= 4 ? (r.leaf_diff ? "DIFFER" : "equal") : "n/a", ok ? "PASS" : "FAIL");
    return ok;
}

int run_check(bool small_only) {
    sycl::queue q{sycl::default_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::mt19937 rng(2037);
    int fails = 0, n = 0;
    for (const Shape& s : shapes()) {
        if (small_only && !s.small) continue;
        Weights A, B;
        A.make(q, s.K, s.N, rng);
        B.make(q, s.K, s.N, rng);
        for (uint32_t T : kTs) { fails += run_case(q, s, T, A, B, rng) ? 0 : 1; ++n; }
        A.release(q); B.release(q);
    }
    if (fails) { std::printf("gemv_q8_soa_batched_v2_test: FAIL (%d of %d cases)\n", fails, n); return 1; }
    std::printf("gemv_q8_soa_batched_v2_test: ALL PASS (%d cases)\n", n);
    return 0;
}

double median_us(sycl::queue& q, const std::function<sycl::event()>& launch) {
    for (int i = 0; i < 5; ++i) launch();
    q.wait();
    std::vector<double> t;
    for (int i = 0; i < 30; ++i) {
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
    std::mt19937 rng(7);
    for (const Shape& s : shapes()) {
        if (s.small) continue;
        Weights A, B;
        A.make(q, s.K, s.N, rng);
        B.make(q, s.K, s.N, rng);
        const double wbytes = double(s.N) * s.K * (1.0 + 2.0 / 32);
        for (uint32_t T : {2u, 4u, 8u, 16u}) {
            std::vector<ie::block_q8_1x> x;
            make_act(x, uint64_t(T) * (s.K / 32), rng);
            auto* dx = sycl::malloc_device<ie::block_q8_1x>(x.size(), q);
            q.memcpy(dx, x.data(), x.size() * sizeof(ie::block_q8_1x)).wait();
            auto* ya = sycl::malloc_device<sycl::half>(uint64_t(T) * s.N, q);
            auto* yb = sycl::malloc_device<sycl::half>(uint64_t(T) * s.N, q);
            const double s1 = median_us(q, [&] { return ie::gemv_q8_0_soa_q8_batched(q, dx, A.dqs, A.dd, ya, s.K, s.N, T); });
            const double s2 = median_us(q, [&] { return ie::gemv_q8_0_soa_q8_batched_v2(q, dx, A.dqs, A.dd, ya, s.K, s.N, T); });
            const double d1 = median_us(q, [&] { return ie::gemv_q8_0_soa_q8_batched_dual(q, dx, A.dqs, A.dd, ya, B.dqs, B.dd, yb, s.K, s.N, T); });
            const double d2 = median_us(q, [&] { return ie::gemv_q8_0_soa_q8_batched_dual_v2(q, dx, A.dqs, A.dd, ya, B.dqs, B.dd, yb, s.K, s.N, T); });
            std::printf("  %-13s K=%-5u N=%-5u T=%-2u  single v1 %8.1f us v2 %8.1f us (%.2fx, v2 %5.0f GB/s)   "
                        "dual v1 %8.1f us v2 %8.1f us (%.2fx)\n", s.name, s.K, s.N, T, s1, s2, s1 / s2,
                        wbytes / (s2 * 1e3), d1, d2, d1 / d2);
            sycl::free(dx, q); sycl::free(ya, q); sycl::free(yb, q);
        }
        A.release(q); B.release(q);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "--bench") return run_bench();
    return run_check(mode == "--small");
}
