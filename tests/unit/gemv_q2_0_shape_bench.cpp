// tests/unit/gemv_q2_0_shape_bench.cpp — per-shape perf microbench of the Q2_0
// decode GEMV at the exact Ternary-Bonsai-27B production shapes (experiment
// campaign docs/q2_0_optimization/05, post-#1b probe). PERF ONLY — numeric
// correctness is gemv_q2_0_soa_test's job.
//
// Defeats L2 residency on small shapes by cycling REP independent weight
// replicas (working set > L2). Reports per-shape ms/call, weight-GB/s, and the
// predicted per-token contribution (× per-token call multiplicity) so the
// aggregate can be reconciled against the captured kprofile (29.7 ms/token).
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

using namespace ie;

int main() {
    // IE_BENCH_INORDER=1 makes the queue in-order, matching the app's real
    // per-call isolation (the default out-of-order queue overlaps ITERS calls
    // and reports optimistic pipelined throughput — see
    // docs/q2_0_optimization/05, the in-app-gap calibration).
    const bool inorder = std::getenv("IE_BENCH_INORDER") != nullptr;
    sycl::queue q = inorder
        ? sycl::queue{sycl::default_selector_v, sycl::property::queue::in_order{}}
        : sycl::queue{sycl::default_selector_v};
    std::printf("device: %s  queue: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str(),
                inorder ? "in_order" : "out_of_order");

    struct Shape { uint32_t K, N, mult; const char* name; };
    // mult = calls per token in the 64-layer Bonsai forward (48 DN + 16 FA).
    const Shape shapes[] = {
        {5120,  10240, 48,  "attn_qkv DN [5120,10240]"},
        {5120,  6144,  48,  "attn_gate  [5120,6144]"},
        {6144,  5120,  64,  "ssm/attn_o [6144,5120]"},
        {5120,  12288, 16,  "attn_q|g   [5120,12288]"},
        {5120,  1024,  32,  "attn_k+v   [5120,1024]"},
        {5120,  17408, 128, "ffn_gate+up[5120,17408]"},
        {17408, 5120,  64,  "ffn_down   [17408,5120]"},
        {5120,  248320, 1,  "lm_head    [5120,248320]"},
    };

    std::mt19937 rng(42);
    double tok_ms_total = 0.0;

    for (const auto& s : shapes) {
        const uint32_t K = s.K, N = s.N;
        const uint64_t qs_bytes = uint64_t(N) * (K / 4);
        const uint64_t d_elems  = uint64_t(N) * (K / 128);
        const uint64_t w_bytes  = qs_bytes + d_elems * 2;

        // Replicas so the cycled working set exceeds L2 (~24 MB) by a margin.
        const uint32_t REP = uint32_t(std::min<uint64_t>(64,
                                std::max<uint64_t>(1, (96ull << 20) / w_bytes + 1)));

        // Synthetic streams: random bytes are fine for perf (dp4a doesn't care).
        std::vector<uint8_t>  h_qs(qs_bytes);
        std::vector<uint16_t> h_d(d_elems, 0x3C00 /* fp16 1.0 */);
        for (auto& b : h_qs) b = uint8_t(rng());

        std::vector<uint8_t*>  d_qs(REP);
        std::vector<uint16_t*> d_dv(REP);
        for (uint32_t r = 0; r < REP; ++r) {
            d_qs[r] = sycl::malloc_device<uint8_t>(qs_bytes, q);
            d_dv[r] = sycl::malloc_device<uint16_t>(d_elems, q);
            q.memcpy(d_qs[r], h_qs.data(), qs_bytes).wait();
            q.memcpy(d_dv[r], h_d.data(), d_elems * 2).wait();
        }
        auto* d_A  = sycl::malloc_device<sycl::half>(K, q);
        auto* d_q8 = sycl::malloc_device<block_q8_1x>(K / 32, q);
        auto* d_y  = sycl::malloc_device<sycl::half>(N, q);
        std::vector<sycl::half> Ah(K, sycl::half(0.03125f));
        q.memcpy(d_A, Ah.data(), K * sizeof(sycl::half)).wait();
        quantize_q8_1(q, d_A, d_q8, K).wait();

        const uint32_t ITERS = uint32_t(std::min<uint64_t>(64,
                                 std::max<uint64_t>(12, (3ull << 30) / w_bytes)));
        for (uint32_t i = 0; i < 5; ++i)  // warmup (JIT + clocks)
            gemv_q2_0_soa_q8(q, d_q8, d_qs[i % REP], d_dv[i % REP], d_y, K, N, {});
        q.wait();

        const auto t0 = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < ITERS; ++i)
            gemv_q2_0_soa_q8(q, d_q8, d_qs[i % REP], d_dv[i % REP], d_y, K, N, {});
        q.wait();
        const auto t1 = std::chrono::steady_clock::now();

        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / ITERS;
        const double gbs = (double(w_bytes) / 1e9) / (ms / 1e3);
        const double tok_ms = ms * s.mult;
        tok_ms_total += tok_ms;
        std::printf("  %-26s w=%7.2f MB rep=%2u it=%2u  %8.4f ms/call  %6.1f GB/s  -> %6.2f ms/token (x%u)\n",
                    s.name, double(w_bytes) / 1e6, REP, ITERS, ms, gbs, tok_ms, s.mult);

        for (uint32_t r = 0; r < REP; ++r) { sycl::free(d_qs[r], q); sycl::free(d_dv[r], q); }
        sycl::free(d_A, q); sycl::free(d_q8, q); sycl::free(d_y, q);
    }
    std::printf("PREDICTED gemv_q2_soa total: %.2f ms/token (kprofile measured ~29.7)\n", tok_ms_total);
    return 0;
}
