// tools/qwen4exp_run2.cpp — Flash-Next 2-GPU layer-split pipeline.
//
// Layers [0, split) on B70 #0 (embedding + PLE), [split, 48) on B70 #1
// (final merge + lm_head). The wide residual [T, 4, 2560] fp32 bounces
// through host between stages (P2P direct is a later lever). Each card's
// expert cache covers ONLY its own layers' banks — the structural prefill
// lever (per-card residency ~2x).
//
// GATE: the split must be numerically INVISIBLE — same kernels, same order,
// exact fp32 handoff. Section 1 runs the same fixed prompt single-GPU and
// the pipeline must reproduce its logits BIT-EXACTLY.
//
// usage: ie-qwen4exp-run2 <model.gguf> [split=24]
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/qwen4exp.hpp"

#include <chrono>
#include <future>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace ie;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "\033[32mOK\033[0m" : "\033[31mFAIL\033[0m", what);
    if (!ok) ++g_fail;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.gguf> [split]\n", argv[0]); return 2; }
    const uint32_t split = argc > 2 ? uint32_t(std::atoi(argv[2])) : 24;

    GgufReader g;
    if (auto e = g.open(argv[1]); !e.empty()) { std::fprintf(stderr, "open: %s\n", e.c_str()); return 1; }
    Qwen4ExpConfig cfg;
    if (auto e = read_qwen4exp_config(g, cfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }

    const std::vector<int32_t> prompt = {3000, 415, 6023, 90, 12, 777, 4088, 22};
    std::vector<float> ref_logits;
    std::vector<int32_t> ref_decode;

    std::printf("\n\033[1mFlash-Next 2-GPU pipeline gate\033[0m  split at layer %u\n\n", split);
    std::printf("\033[1m1. single-GPU reference\033[0m\n");
    {
        DeviceAllocator a0;
        if (auto e = a0.init("B70", 0); !e.empty()) { std::fprintf(stderr, "gpu0: %s\n", e.c_str()); return 1; }
        Qwen4ExpModel ref;
        if (auto e = ref.load(a0, g, cfg); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
        if (auto e = ref.init_runtime(2051, 1024); !e.empty()) { std::fprintf(stderr, "rt: %s\n", e.c_str()); return 1; }
        ref.forward(prompt.data(), uint32_t(prompt.size()), 0, nullptr);
        std::vector<sycl::half> l16(cfg.vocab);
        a0.queue().memcpy(l16.data(), ref.logits(), cfg.vocab * 2).wait();
        ref_logits.resize(cfg.vocab);
        for (uint32_t i = 0; i < cfg.vocab; ++i) ref_logits[i] = float(l16[i]);
        int32_t tok = 0; float mx = -1e30f;
        for (uint32_t i = 0; i < cfg.vocab; ++i)
            if (ref_logits[i] > mx) { mx = ref_logits[i]; tok = int32_t(i); }
        uint32_t pos = uint32_t(prompt.size());
        for (int st = 0; st < 4; ++st) {
            ref_decode.push_back(tok);
            ref.forward(&tok, 1, pos, nullptr); ++pos;
            a0.queue().memcpy(l16.data(), ref.logits(), cfg.vocab * 2).wait();
            mx = -1e30f;
            for (uint32_t i = 0; i < cfg.vocab; ++i)
                if (float(l16[i]) > mx) { mx = float(l16[i]); tok = int32_t(i); }
        }
        std::printf("      reference decode:");
        for (int32_t t : ref_decode) std::printf(" %d", t);
        std::printf("\n");
    }   // reference model + its VRAM released here

    std::printf("\n\033[1m2. pipeline load (A: [0,%u) on GPU0, B: [%u,48) on GPU1)\033[0m\n", split, split);
    DeviceAllocator a0, a1;
    // IE_P2P=1: both allocators in ONE shared context with peer access —
    // the interstage wide handoff then PUSHES device-to-device instead of
    // bouncing through the host (bit-eq below is the transport gate).
    std::vector<sycl::device> p2p_gpus;
    std::unique_ptr<sycl::context> p2p_ctx;
    const bool p2p = std::getenv("IE_P2P") != nullptr &&
                     gpu_p2p_shared_context("B70", p2p_gpus, p2p_ctx);
    if (std::getenv("IE_P2P") && !p2p)
        std::fprintf(stderr, "IE_P2P requested but unavailable — host bounce\n");
    if (p2p) {
        if (auto e = a0.init_with(*p2p_ctx, p2p_gpus[0]); !e.empty()) { std::fprintf(stderr, "gpu0: %s\n", e.c_str()); return 1; }
        if (auto e = a1.init_with(*p2p_ctx, p2p_gpus[1]); !e.empty()) { std::fprintf(stderr, "gpu1: %s\n", e.c_str()); return 1; }
    } else {
    if (auto e = a0.init("B70", 0); !e.empty()) { std::fprintf(stderr, "gpu0: %s\n", e.c_str()); return 1; }
    if (auto e = a1.init("B70", 1); !e.empty()) { std::fprintf(stderr, "gpu1: %s\n", e.c_str()); return 1; }
    }
    Qwen4ExpModel A, B;
    const auto tl0 = std::chrono::steady_clock::now();
    if (auto e = A.load(a0, g, cfg, 0, 0, split); !e.empty()) { std::fprintf(stderr, "A: %s\n", e.c_str()); return 1; }
    if (auto e = B.load(a1, g, cfg, 0, split, cfg.n_layers); !e.empty()) { std::fprintf(stderr, "B: %s\n", e.c_str()); return 1; }
    // Expert-cache budget: adaptive (fill-to-reserve) inside init_runtime.
    // 8192: section 4 prefills to pos ~4.1K — the old 2051 sent forward_range
    // past the KV into unmapped VAs (the 2026-08-27 03:08 xe pagefault).
    if (auto e = A.init_runtime(8192, 1024); !e.empty()) { std::fprintf(stderr, "A rt: %s\n", e.c_str()); return 1; }
    if (auto e = B.init_runtime(8192, 1024); !e.empty()) { std::fprintf(stderr, "B rt: %s\n", e.c_str()); return 1; }
    std::printf("      loaded in %.0f s; device A %.2f GiB, B %.2f GiB\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - tl0).count(),
                A.device_bytes() / 1073741824.0, B.device_bytes() / 1073741824.0);

    std::vector<float> wide(uint64_t(1024) * cfg.hc_count * cfg.hidden);
    bool p2p_hand = p2p;              // handoff transport toggle (see below)
    if (p2p_hand) A.set_wide_peer(B.wide_device());
    auto pipe_fwd = [&](const int32_t* toks, uint32_t T, uint32_t pos) -> std::string {
        if (auto e = A.forward_range(toks, T, pos, nullptr, wide.data(), nullptr); !e.empty()) return e;
        return B.forward_range(toks, T, pos, p2p_hand ? nullptr : wide.data(),
                               nullptr, nullptr, /*wide_in_device=*/p2p_hand);
    };

    std::printf("\n\033[1m3. bit-equality vs single-GPU\033[0m\n");
    if (auto e = pipe_fwd(prompt.data(), uint32_t(prompt.size()), 0); !e.empty()) {
        std::fprintf(stderr, "pipeline: %s\n", e.c_str()); return 1;
    }
    std::vector<sycl::half> l16(cfg.vocab);
    a1.queue().memcpy(l16.data(), B.logits(), cfg.vocab * 2).wait();
    uint32_t ndiff = 0; float mdiff = 0; int32_t tok = 0; float mx = -1e30f;
    for (uint32_t i = 0; i < cfg.vocab; ++i) {
        const float v = float(l16[i]);
        if (v != ref_logits[i]) ++ndiff;
        mdiff = std::max(mdiff, std::abs(v - ref_logits[i]));
        if (v > mx) { mx = v; tok = int32_t(i); }
    }
    std::printf("      logits differing: %u  max |delta|: %g\n", ndiff, mdiff);
    check(ndiff == 0, "prefill logits BIT-IDENTICAL to single-GPU");
    std::vector<int32_t> pipe_decode;
    uint32_t pos = uint32_t(prompt.size());
    for (int st = 0; st < 4; ++st) {
        pipe_decode.push_back(tok);
        if (auto e = pipe_fwd(&tok, 1, pos); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
        ++pos;
        a1.queue().memcpy(l16.data(), B.logits(), cfg.vocab * 2).wait();
        mx = -1e30f;
        for (uint32_t i = 0; i < cfg.vocab; ++i)
            if (float(l16[i]) > mx) { mx = float(l16[i]); tok = int32_t(i); }
    }
    check(pipe_decode == ref_decode, "4-step greedy decode identical");

    std::printf("\n\033[1m4. pipelined prefill: host-bounce vs P2P double-bank\033[0m\n");
    {
        // Chunk-PIPELINED prefill (A runs chunk k+1 while B runs chunk k).
        // Variant 0: DOUBLE HOST buffers (the historical path). Variant 1:
        // P2P push into B's DOUBLE device banks (alloc_wide_bank2) — the
        // single-peer-target clobber is gone because A alternates
        // set_wide_peer(B.wide_bank(k & 1)) against B's use_wide_bank.
        // Both replay the identical token stream from a reset state; the
        // final-chunk logits must match BIT-EXACTLY.
        // MEASUREMENT CAVEAT (2026-08-31): variant 0 pays the expert-cache
        // warmup (reset_state empties it), variant 1 rides the warmed cache —
        // the throughput delta is order-biased in variant 1's favor. Honest
        // A/B on a warm second pass: banks ~539 vs bounce ~514 tok/s (~+5%).
        // The BIT-EQUALITY check is unaffected.
        std::vector<int32_t> chunk(4096);
        for (int i = 0; i < 4096; ++i) chunk[i] = 2000 + (i * 97) % 200000;
        std::vector<std::vector<float>> wbuf(2,
            std::vector<float>(uint64_t(1024) * cfg.hc_count * cfg.hidden));
        const bool have_banks = p2p && B.alloc_wide_bank2().empty();
        if (p2p && !have_banks)
            std::fprintf(stderr, "      (bank2 alloc failed — bank variant skipped)\n");
        std::vector<sycl::half> lg[2];
        double ms[2] = {0, 0};
        for (int variant = 0; variant < (have_banks ? 2 : 1); ++variant) {
            const bool banks = variant == 1;
            A.reset_state(); B.reset_state();
            A.set_wide_peer(nullptr);
            B.use_wide_bank(0);
            // context: the fixed 8-token prompt, serial host handoff
            if (auto e = A.forward_range(prompt.data(), 8, 0, nullptr,
                                         wbuf[0].data(), nullptr); !e.empty()) {
                std::fprintf(stderr, "A: %s\n", e.c_str()); return 1;
            }
            if (auto e = B.forward_range(prompt.data(), 8, 0, wbuf[0].data(),
                                         nullptr, nullptr); !e.empty()) {
                std::fprintf(stderr, "B: %s\n", e.c_str()); return 1;
            }
            const auto t0 = std::chrono::steady_clock::now();
            std::future<std::string> bfut;
            for (int k = 0; k < 4; ++k) {
                float* wb = wbuf[k & 1].data();
                if (banks) A.set_wide_peer(B.wide_bank(k & 1));
                if (auto e = A.forward_range(chunk.data() + k * 1024, 1024,
                                             8 + k * 1024, nullptr, wb, nullptr);
                    !e.empty()) {
                    std::fprintf(stderr, "A: %s\n", e.c_str()); return 1;
                }
                if (bfut.valid())
                    if (auto e = bfut.get(); !e.empty()) { std::fprintf(stderr, "B: %s\n", e.c_str()); return 1; }
                const int bank = k & 1;
                bfut = std::async(std::launch::async, [&B, wb, k, banks, bank]() {
                    if (banks) B.use_wide_bank(uint32_t(bank));
                    return B.forward_range(nullptr, 1024, 8 + uint32_t(k) * 1024,
                                           banks ? nullptr : wb, nullptr, nullptr,
                                           /*wide_in_device=*/banks);
                });
            }
            if (auto e = bfut.get(); !e.empty()) { std::fprintf(stderr, "B: %s\n", e.c_str()); return 1; }
            ms[variant] = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            A.set_wide_peer(nullptr);
            B.use_wide_bank(0);
            lg[variant].resize(cfg.vocab);
            a1.queue().memcpy(lg[variant].data(), B.logits(), cfg.vocab * 2).wait();
            std::printf("      prefill 4x1024 pipelined [%s]: %.0f ms (%.1f tok/s)\n",
                        banks ? "P2P banks" : "host bounce", ms[variant],
                        4096000.0 / ms[variant]);
        }
        if (have_banks) {
            uint32_t nd = 0;
            for (uint32_t i = 0; i < cfg.vocab; ++i)
                if (std::memcmp(&lg[0][i], &lg[1][i], 2) != 0) ++nd;
            check(nd == 0, "P2P-bank pipelined logits BIT-IDENTICAL to host bounce");
            std::printf("      pipeline speedup from P2P banks: %.2fx\n", ms[0] / ms[1]);
        }
        pos = 8 + 4096;
        A.ecache_hits = A.ecache_misses = 0;
        B.ecache_hits = B.ecache_misses = 0;
        if (p2p) { A.set_wide_peer(B.wide_device()); p2p_hand = true; }
        for (int w = 0; w < 24; ++w) { pipe_fwd(&tok, 1, pos); ++pos; }   // cache warm
        const auto t1 = std::chrono::steady_clock::now();
        for (int st = 0; st < 16; ++st) { pipe_fwd(&tok, 1, pos); ++pos; }
        const double dms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t1).count() / 16.0;
        std::printf("      decode steady: %.1f ms/token (%.1f tok/s)  [A hit %.0f%%, B hit %.0f%%]\n",
                    dms, 1000.0 / dms,
                    100.0 * A.ecache_hits / std::max<uint64_t>(1, A.ecache_hits + A.ecache_misses),
                    100.0 * B.ecache_hits / std::max<uint64_t>(1, B.ecache_hits + B.ecache_misses));
    }

    std::printf("\n\033[1m%s\033[0m  (%d failure%s)\n\n",
                g_fail ? "\033[31mGATE FAILED\033[0m" : "\033[32mGATE PASSED\033[0m",
                g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
