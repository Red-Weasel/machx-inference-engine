// deepseek4_mxfp4xmx_gate_test — numerical gate for ds4_gemm_mxfp4_xmx
// (docs/deepseek4/72 Phase E), the fused MXFP4 -> XMX prefill expert GEMM.
//
// Claims checked, per criterion 1 of the plan:
//   (a) max |y - y_ref| within a bound derived from fp16 inputs and fp32
//       accumulation, y_ref computed in double from the SAME dequantised values;
//   (b) rows >= M of every job are never written (canary rows between jobs);
//   (c) three launches produce bit-identical outputs;
//   (d) against the materialised route (ds4_dequant_mxfp4_w16 + oneDNN) the
//       difference stays within the same bound (reported, and gated when oneDNN
//       is available).
// Both real shapes: gate/up (K 4096, N 1024) and down (K 1024, N 4096), several
// jobs of different M in ONE launch, M ragged against the 32-row tile.
//
// Build: ninja -C build deepseek4_mxfp4xmx_gate_test && ./build/tests/deepseek4_mxfp4xmx_gate_test

#include "ie/deepseek4_experts.hpp"
#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

int g_fail = 0;

// The reference decode, written out independently of the kernel's helpers.
float e8m0_half(uint8_t e) {
    const uint32_t bits = (e < 2u) ? (0x00200000u << e) : (uint32_t(e - 1u) << 23);
    float f; std::memcpy(&f, &bits, 4); return f;
}
int nibble_val(uint32_t nb) {
    static const int mag[8] = {0, 1, 2, 3, 4, 6, 8, 12};
    const int m = mag[nb & 7u];
    return (nb & 8u) ? -m : m;
}

struct Case { uint32_t M, K, N; };

void run_shape(sycl::queue& q, const std::vector<Case>& jobs_spec, unsigned seed, const char* what) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> nib(0, 255), esc(120, 132);
    std::uniform_real_distribution<float> ux(-1.f, 1.f);
    const uint32_t CAN = 8;   // canary rows after every job

    struct Buf {
        std::vector<uint8_t> qs, e; std::vector<sycl::half> x;
        uint8_t* dqs; uint8_t* de; sycl::half* dx; float* dy; uint32_t M, K, N; size_t y_rows;
    };
    std::vector<Buf> B;
    for (const Case& c : jobs_spec) {
        Buf b; b.M = c.M; b.K = c.K; b.N = c.N;
        b.qs.resize(size_t(c.N) * c.K / 2); b.e.resize(size_t(c.N) * c.K / 32);
        b.x.resize(size_t(c.M) * c.K);
        for (auto& v : b.qs) v = uint8_t(nib(rng));
        for (auto& v : b.e)  v = uint8_t(esc(rng));
        for (auto& v : b.x)  v = sycl::half(ux(rng));
        b.dqs = sycl::malloc_device<uint8_t>(b.qs.size(), q);
        b.de  = sycl::malloc_device<uint8_t>(b.e.size(), q);
        b.dx  = sycl::malloc_device<sycl::half>(b.x.size(), q);
        b.y_rows = size_t(c.M) + CAN;
        b.dy  = sycl::malloc_device<float>(b.y_rows * c.N, q);
        q.memcpy(b.dqs, b.qs.data(), b.qs.size()).wait();
        q.memcpy(b.de,  b.e.data(),  b.e.size()).wait();
        q.memcpy(b.dx,  b.x.data(),  b.x.size() * sizeof(sycl::half)).wait();
        B.push_back(b);
    }
    auto fill_canary = [&] {
        for (auto& b : B) q.fill(b.dy, -12345.0f, b.y_rows * b.N).wait();
    };
    std::vector<ie::Ds4MxXmxJob> J;
    for (auto& b : B) J.push_back({b.dqs, b.de, b.dx, b.dy, b.M, b.K, b.N});

    fill_canary();
    ie::ds4_gemm_mxfp4_xmx(q, J.data(), uint32_t(J.size())).wait();
    std::vector<std::vector<float>> y1;
    for (auto& b : B) { std::vector<float> h(b.y_rows * b.N); q.memcpy(h.data(), b.dy, h.size() * 4).wait(); y1.push_back(h); }

    // (c) determinism: two more launches
    size_t nondet = 0;
    for (int rep = 0; rep < 2; ++rep) {
        fill_canary();
        ie::ds4_gemm_mxfp4_xmx(q, J.data(), uint32_t(J.size())).wait();
        for (size_t j = 0; j < B.size(); ++j) {
            std::vector<float> h(B[j].y_rows * B[j].N); q.memcpy(h.data(), B[j].dy, h.size() * 4).wait();
            for (size_t i = 0; i < h.size(); ++i)
                if (std::memcmp(&h[i], &y1[j][i], 4) != 0) ++nondet;
        }
    }

    // (a) reference in double + bound, (b) canary
    double worst_ratio = 0; size_t canary_bad = 0, nonfinite = 0;
    std::vector<std::vector<double>> bounds(B.size());
    for (size_t j = 0; j < B.size(); ++j) {
        const Buf& b = B[j];
        const uint32_t bpc = b.K / 32;
        std::vector<float> W(size_t(b.N) * b.K);
        for (uint32_t n = 0; n < b.N; ++n)
            for (uint32_t bl = 0; bl < bpc; ++bl) {
                const float d = e8m0_half(b.e[size_t(n) * bpc + bl]);
                for (uint32_t t = 0; t < 16; ++t) {
                    const uint8_t byte = b.qs[size_t(n) * (b.K / 2) + size_t(bl) * 16 + t];
                    W[size_t(n) * b.K + bl * 32 + t]      = float(sycl::half(d * float(nibble_val(byte & 0xF))));
                    W[size_t(n) * b.K + bl * 32 + 16 + t] = float(sycl::half(d * float(nibble_val(byte >> 4))));
                }
            }
        for (uint32_t m = 0; m < b.M; ++m)
            for (uint32_t n = 0; n < b.N; ++n) {
                double acc = 0, mag = 0;
                for (uint32_t k = 0; k < b.K; ++k) {
                    const double p = double(float(b.x[size_t(m) * b.K + k])) * double(W[size_t(n) * b.K + k]);
                    acc += p; mag += std::fabs(p);
                }
                // fp32 accumulation of K exact products: |err| <= K * u32 * sum|p| (+ output rounding);
                // XMX accumulates in fp32 with at most that many roundings.  Factor 2 of slack.
                const double bound = 2.0 * double(b.K) * 5.96e-8 * mag + 1e-6;
                bounds[j].push_back(bound);
                const float got = y1[j][size_t(m) * b.N + n];
                if (!std::isfinite(got)) ++nonfinite;
                const double ratio = std::fabs(double(got) - acc) / bound;
                if (ratio > worst_ratio) worst_ratio = ratio;
            }
        for (uint32_t m = b.M; m < b.M + CAN; ++m)
            for (uint32_t n = 0; n < b.N; ++n)
                if (y1[j][size_t(m) * b.N + n] != -12345.0f) ++canary_bad;
    }

    // (d) vs the materialised route
    double worst_vs_onednn = -1;
    if (ie::onednn_available()) {
        worst_vs_onednn = 0;
        for (size_t j = 0; j < B.size(); ++j) {
            const Buf& b = B[j];
            sycl::half* w16 = sycl::malloc_device<sycl::half>(size_t(b.K) * b.N, q);
            float* y2 = sycl::malloc_device<float>(size_t(b.M) * b.N, q);
            auto dq = ie::ds4_dequant_mxfp4_w16(q, b.dqs, b.de, w16, b.K, b.N);
            ie::gemm_nt_f16_onednn(q, b.dx, w16, y2, b.M, b.N, b.K, {dq}).wait();
            std::vector<float> h(size_t(b.M) * b.N); q.memcpy(h.data(), y2, h.size() * 4).wait();
            // Both routes accumulate the same exact products in fp32, so each is
            // within `bound` of the double reference and their difference within
            // 2 x bound.  (A relative metric is meaningless here: outputs near
            // zero after cancellation carry the absolute error of the whole sum.)
            for (size_t i = 0; i < h.size(); ++i) {
                const double diff = std::fabs(double(h[i]) - double(y1[j][i]));
                const double r = diff / (2.0 * bounds[j][i]);
                if (r > worst_vs_onednn) worst_vs_onednn = r;
            }
            sycl::free(w16, q); sycl::free(y2, q);
        }
    }

    const bool bad = worst_ratio > 1.0 || canary_bad || nondet || nonfinite ||
                     (worst_vs_onednn >= 0 && worst_vs_onednn > 1.0);
    std::printf("  %-10s jobs=%zu  err/bound %.3f  canary_bad %zu  nondet %zu  nonfinite %zu"
                "  |fused-oneDNN|/(2 bound) %s  %s\n",
                what, B.size(), worst_ratio, canary_bad, nondet, nonfinite,
                worst_vs_onednn < 0 ? "n/a" : std::to_string(worst_vs_onednn).c_str(),
                bad ? "*** FAIL ***" : "PASS");
    if (bad) ++g_fail;

    // Negative control: a wrong scale on one job must breach the bound.
    {
        std::vector<uint8_t> e_bad = B[0].e;
        for (auto& v : e_bad) v = uint8_t(v + 1);   // every scale doubled
        q.memcpy(B[0].de, e_bad.data(), e_bad.size()).wait();
        fill_canary();
        ie::ds4_gemm_mxfp4_xmx(q, J.data(), uint32_t(J.size())).wait();
        std::vector<float> h(B[0].y_rows * B[0].N); q.memcpy(h.data(), B[0].dy, h.size() * 4).wait();
        size_t differ = 0;
        for (size_t i = 0; i < size_t(B[0].M) * B[0].N; ++i)
            if (std::memcmp(&h[i], &y1[0][i], 4) != 0) ++differ;
        std::printf("    neg-control scales doubled on job 0: %zu differ -> %s\n",
                    differ, differ ? "CAUGHT" : "*** MISSED ***");
        if (!differ) ++g_fail;
    }
    for (auto& b : B) { sycl::free(b.dqs, q); sycl::free(b.de, q); sycl::free(b.dx, q); sycl::free(b.dy, q); }
}

}  // namespace

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("deepseek4_mxfp4xmx_gate_test — ds4_gemm_mxfp4_xmx vs double reference\n");
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    // gate/up shape: K = hidden 4096, N = per-card expert width 1024
    run_shape(q, {{1, 4096, 1024}, {5, 4096, 1024}, {12, 4096, 1024}, {33, 4096, 1024},
                  {64, 4096, 1024}, {130, 4096, 1024}}, 71, "gate/up");
    // down shape: K = 1024, N = 4096
    run_shape(q, {{1, 1024, 4096}, {5, 1024, 4096}, {12, 1024, 4096}, {33, 1024, 4096},
                  {64, 1024, 4096}, {130, 1024, 4096}}, 72, "down");
    // one job alone, tile-aligned M
    run_shape(q, {{32, 4096, 1024}}, 73, "M=32");
    // Every side of the eight-row accumulator boundaries, including the next
    // workgroup's tail. Compare to the independent double reference and guard
    // all eight rows beyond each output, with repeat and corruption controls.
    std::vector<Case> tails;
    for (uint32_t m : {1u, 7u, 8u, 9u, 15u, 16u, 17u, 23u, 24u, 25u,
                       31u, 32u, 33u, 39u, 40u, 41u, 63u, 64u, 65u})
        tails.push_back({m, 64, 128});
    run_shape(q, tails, 74, "8-row tails");
    std::printf("=======================================================\n");
    if (g_fail) { std::printf("GATE FAILED: %d check(s) wrong.\n", g_fail); return 1; }
    std::printf("GATE PASSED.\n");
    return 0;
}
