// tools/qwen4exp_fwd_test.cpp — Qwen3.8-Flash-Next first-forward smoke.
//
// Loads the real GGUF on one B70, runs a short prefill + a few greedy decode
// steps, and checks the machinery holds together: logits finite and
// non-degenerate, argmax stable across a repeated identical run
// (deterministic), decode step advances state without faulting. This is a
// SMOKE gate — numerical parity vs the HF oracle is the next gate.
//
// usage: ie-qwen4exp-fwd-test <model-00001-of-0000N.gguf> [gpu]
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/ops.hpp"
#include "ie/qwen4exp.hpp"

#include <chrono>
#include <cinttypes>
#include <cmath>
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
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.gguf> [gpu]\n", argv[0]); return 2; }
    const uint32_t ordinal = argc > 2 ? uint32_t(std::atoi(argv[2])) : 0;

    GgufReader g;
    if (auto e = g.open(argv[1]); !e.empty()) { std::fprintf(stderr, "open: %s\n", e.c_str()); return 1; }
    Qwen4ExpConfig cfg;
    if (auto e = read_qwen4exp_config(g, cfg); !e.empty()) {
        std::fprintf(stderr, "config: %s\n", e.c_str()); return 1;
    }
    DeviceAllocator alloc;
    if (auto e = alloc.init("B70", ordinal); !e.empty()) {
        std::fprintf(stderr, "gpu: %s\n", e.c_str()); return 1;
    }
    std::printf("\n\033[1mQwen3.8-Flash-Next forward smoke\033[0m  GPU %u\n\n", ordinal);

    Qwen4ExpModel m;
    if (auto e = m.load(alloc, g, cfg); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    if (auto e = m.init_runtime(/*max_ctx=*/2051, /*max_chunk=*/64); !e.empty()) {
        std::fprintf(stderr, "init_runtime: %s\n", e.c_str()); return 1;
    }
    std::printf("  loaded; device %.2f GiB\n\n", m.device_bytes() / 1073741824.0);

    // Fixed prompt ids (arbitrary but stable; well inside vocab, no EOS).
    const std::vector<int32_t> prompt = {3000, 415, 6023, 90, 12, 777, 4088, 22};

    auto logits_stats = [&](const char* tag, std::vector<float>& out) {
        std::vector<sycl::half> l16(cfg.vocab);
        alloc.queue().memcpy(l16.data(), m.logits(), cfg.vocab * 2).wait();
        out.resize(cfg.vocab);
        double s = 0, s2 = 0; uint32_t nonfinite = 0; float mx = -1e30f; uint32_t am = 0;
        for (uint32_t i = 0; i < cfg.vocab; ++i) {
            const float v = float(l16[i]);
            out[i] = v;
            if (!std::isfinite(v)) ++nonfinite;
            s += v; s2 += double(v) * v;
            if (v > mx) { mx = v; am = i; }
        }
        const double mean = s / cfg.vocab;
        const double var = s2 / cfg.vocab - mean * mean;
        std::printf("      %s: argmax %u  max %.3f  mean %.3f  std %.3f  nonfinite %u\n",
                    tag, am, mx, mean, std::sqrt(std::max(0.0, var)), nonfinite);
        return std::pair<uint32_t, uint32_t>(am, nonfinite);
    };

    std::printf("\033[1m1. prefill T=%zu\033[0m\n", prompt.size());
    const auto t0 = std::chrono::steady_clock::now();
    if (auto e = m.forward(prompt.data(), uint32_t(prompt.size()), 0, nullptr); !e.empty()) {
        std::fprintf(stderr, "forward: %s\n", e.c_str()); return 1;
    }
    const double ms0 = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    std::vector<float> l1, l2;
    auto [am1, nf1] = logits_stats("run 1", l1);
    check(nf1 == 0, "logits all finite");
    {
        double s2 = 0, mean = 0;
        for (float v : l1) mean += v;
        mean /= cfg.vocab;
        for (float v : l1) s2 += (v - mean) * (v - mean);
        check(std::sqrt(s2 / cfg.vocab) > 0.5, "logit distribution non-degenerate (std > 0.5)");
    }
    std::printf("      prefill %.0f ms (%.1f ms/token)\n", ms0, ms0 / prompt.size());

    std::printf("\n\033[1m2. determinism (identical rerun)\033[0m\n");
    m.reset_state();
    if (auto e = m.forward(prompt.data(), uint32_t(prompt.size()), 0, nullptr); !e.empty()) {
        std::fprintf(stderr, "forward2: %s\n", e.c_str()); return 1;
    }
    auto [am2, nf2] = logits_stats("run 2", l2);
    uint32_t ndiff = 0; float mdiff = 0;
    for (uint32_t i = 0; i < cfg.vocab; ++i) {
        if (l1[i] != l2[i]) ++ndiff;
        mdiff = std::max(mdiff, std::abs(l1[i] - l2[i]));
    }
    std::printf("      elems differing: %u  max |delta|: %g\n", ndiff, mdiff);
    check(am1 == am2, "argmax identical across identical runs");

    std::printf("\n\033[1m3. greedy decode x8\033[0m\n");
    int32_t tok = int32_t(am2);
    uint32_t pos = uint32_t(prompt.size());
    std::vector<int32_t> gen;
    const auto t1 = std::chrono::steady_clock::now();
    bool ok = true;
    for (int step = 0; step < 8 && ok; ++step) {
        if (auto e = m.forward(&tok, 1, pos, nullptr); !e.empty()) {
            std::fprintf(stderr, "decode: %s\n", e.c_str()); ok = false; break;
        }
        std::vector<sycl::half> l16(cfg.vocab);
        alloc.queue().memcpy(l16.data(), m.logits(), cfg.vocab * 2).wait();
        float mx = -1e30f; uint32_t am = 0; bool fin = true;
        for (uint32_t i = 0; i < cfg.vocab; ++i) {
            const float v = float(l16[i]);
            if (!std::isfinite(v)) fin = false;
            if (v > mx) { mx = v; am = i; }
        }
        if (!fin) { ok = false; break; }
        gen.push_back(int32_t(am));
        tok = int32_t(am);
        ++pos;
    }
    const double ms1 = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t1).count();
    std::printf("      tokens:");
    for (int32_t t : gen) std::printf(" %d", t);
    std::printf("\n      %.0f ms total, %.0f ms/token (v0 streamed experts, unoptimized)\n",
                ms1, gen.empty() ? 0.0 : ms1 / gen.size());
    check(ok && gen.size() == 8, "8 decode steps, all logits finite");

    std::printf("\n\033[1m%s\033[0m  (%d failure%s)\n\n",
                g_fail ? "\033[31mGATE FAILED\033[0m" : "\033[32mGATE PASSED\033[0m",
                g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
