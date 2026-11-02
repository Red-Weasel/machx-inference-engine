// tools/qwen4exp_run2.cpp — Flash-Next layer-split pipeline gate: 2 stages (the two-card split) or N stages (2026-11-02:
// --gpus N on an N-card box, proven here on fewer cards by mapping several stages to one card).
//
// Stage s owns layers [s*48/N, (s+1)*48/N): stage 0 the embedding + PLE, the last the final merge + lm_head. The wide
// residual [T, 4, 2560] fp32 crosses every stage boundary through the host (P2P direct stays a 2-stage lever). Each
// stage's expert cache covers ONLY its own layers' banks.
//
// GATE: the split must be numerically INVISIBLE — same kernels, same order, exact fp32 handoff. Section 1 runs the fixed
// prompt on one card and the chain must reproduce its logits BIT-EXACTLY; section 3 runs every stage on every forward
// (a chain that skips a stage fails: GitHub issue 3's first N-card cut ran 24 of 48 layers and passed its gates).
//
// usage: ie-qwen4exp-run2 <model.gguf> [stages=2] [cards=2]      stage s runs on card (s * cards) / stages
//   IE_Q4E_RUN2_SKIP_STAGE=s   the gate's self-check: the chain skips stage s, so the bit-identity checks MUST fail
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/qwen4exp.hpp"

#include <chrono>
#include <future>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <memory>
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
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.gguf> [stages=2] [cards=2]\n", argv[0]); return 2; }
    const uint32_t N = argc > 2 ? uint32_t(std::max(2, std::atoi(argv[2]))) : 2u;
    const uint32_t cards = argc > 3 ? uint32_t(std::max(1, std::atoi(argv[3]))) : 2u;
    const int skip = std::getenv("IE_Q4E_RUN2_SKIP_STAGE") ? std::atoi(std::getenv("IE_Q4E_RUN2_SKIP_STAGE")) : -1;

    GgufReader g;
    if (auto e = g.open(argv[1]); !e.empty()) { std::fprintf(stderr, "open: %s\n", e.c_str()); return 1; }
    Qwen4ExpConfig cfg;
    if (auto e = read_qwen4exp_config(g, cfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }
    if (N > cfg.n_layers / 2) { std::fprintf(stderr, "%u stages for %u layers\n", N, cfg.n_layers); return 2; }
    const auto lo_of = [&](uint32_t s) { return uint32_t(uint64_t(s) * cfg.n_layers / N); };
    const auto card_of = [&](uint32_t s) { return uint32_t(uint64_t(s) * cards / N); };

    const std::vector<int32_t> prompt = {3000, 415, 6023, 90, 12, 777, 4088, 22};
    std::vector<float> ref_logits;
    std::vector<int32_t> ref_decode;

    std::printf("\n\033[1mFlash-Next %u-stage pipeline gate\033[0m  on %u card(s)%s\n\n", N, cards,
                skip >= 0 ? "  [SELF-CHECK: a stage is skipped, the identity checks must FAIL]" : "");
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

    std::printf("\n\033[1m2. pipeline load\033[0m\n");
    // IE_P2P=1 (2 stages only): both allocators in ONE shared context with peer access — the interstage wide handoff then
    // PUSHES device-to-device instead of bouncing through the host (bit-eq below is the transport gate).
    std::vector<sycl::device> p2p_gpus;
    std::unique_ptr<sycl::context> p2p_ctx;
    const bool p2p = N == 2 && cards == 2 && std::getenv("IE_P2P") != nullptr &&
                     gpu_p2p_shared_context("B70", p2p_gpus, p2p_ctx);
    if (std::getenv("IE_P2P") && !p2p)
        std::fprintf(stderr, "IE_P2P requested but unavailable (or not a 2-stage run) — host bounce\n");
    std::vector<std::unique_ptr<DeviceAllocator>> al(N);
    std::vector<std::unique_ptr<Qwen4ExpModel>> S(N);
    const auto tl0 = std::chrono::steady_clock::now();
    for (uint32_t s = 0; s < N; ++s) {
        al[s] = std::make_unique<DeviceAllocator>();
        if (p2p) {
            if (auto e = al[s]->init_with(*p2p_ctx, p2p_gpus[s]); !e.empty()) { std::fprintf(stderr, "gpu%u: %s\n", s, e.c_str()); return 1; }
        } else if (auto e = al[s]->init("B70", card_of(s)); !e.empty()) { std::fprintf(stderr, "card %u: %s\n", card_of(s), e.c_str()); return 1; }
        S[s] = std::make_unique<Qwen4ExpModel>();
        if (auto e = S[s]->load(*al[s], g, cfg, 0, lo_of(s), lo_of(s + 1)); !e.empty()) { std::fprintf(stderr, "stage %u: %s\n", s, e.c_str()); return 1; }
        // Expert-cache budget: adaptive (fill-to-reserve) inside init_runtime. 8192: section 4 prefills to pos ~4.1K.
        // stages sharing a card: every context first (defer_cache), then the caches in order from what is really left
        if (auto e = S[s]->init_runtime(8192, 1024, 1, 0, 1, cards < N); !e.empty()) { std::fprintf(stderr, "stage %u rt: %s\n", s, e.c_str()); return 1; }
    }
    for (uint32_t s = 0; s < N; ++s) {
        uint32_t share = 0;   // this stage and the ones still to size on its card split the card's free VRAM
        for (uint32_t t = s; t < N; ++t) if (card_of(t) == card_of(s)) ++share;
        if (auto e = S[s]->init_ecache(share); !e.empty()) { std::fprintf(stderr, "stage %u cache: %s\n", s, e.c_str()); return 1; }
        std::printf("      stage %u: layers [%u, %u) on card %u, %.2f GiB, expert cache %u slots/layer\n", s, lo_of(s), lo_of(s + 1), card_of(s),
                    S[s]->device_bytes() / 1073741824.0, S[s]->ecache_slots());
    }
    std::printf("      loaded in %.0f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tl0).count());
    Qwen4ExpModel& A = *S[0];
    Qwen4ExpModel& B = *S[N - 1];
    sycl::queue& qB = al[N - 1]->queue();
    {   // the chain covers every layer exactly once
        bool cover = lo_of(0) == 0 && lo_of(N) == cfg.n_layers;
        for (uint32_t s = 0; s < N; ++s) cover = cover && lo_of(s) < lo_of(s + 1);
        check(cover, "the stages cover [0, n_layers) contiguously, every stage non-empty");
    }

    std::vector<float> wide(uint64_t(1024) * cfg.hc_count * cfg.hidden);
    bool p2p_hand = p2p;              // handoff transport toggle (see below)
    if (p2p_hand) A.set_wide_peer(B.wide_device());
    std::vector<uint64_t> calls(N, 0);   // forward_range calls per stage: every chain forward must run every stage
    uint64_t chains = 0;
    auto pipe_fwd = [&](const int32_t* toks, uint32_t T, uint32_t pos) -> std::string {
        ++chains;
        for (uint32_t s = 0; s < N; ++s) {
            if (int(s) == skip) continue;   // (the self-check)
            ++calls[s];
            const bool first = s == 0, last = s + 1 == N;   // (first: embeds here; last: the logits)
            std::string e;
            if (last) e = B.forward_range(toks, T, pos, p2p_hand ? nullptr : wide.data(), nullptr, nullptr, /*wide_in_device=*/p2p_hand);
            else      e = S[s]->forward_range(toks, T, pos, first ? nullptr : wide.data(), wide.data(), nullptr);
            if (!e.empty()) return "stage " + std::to_string(s) + ": " + e;
        }
        return {};
    };

    std::printf("\n\033[1m3. bit-equality vs single-GPU\033[0m\n");
    if (auto e = pipe_fwd(prompt.data(), uint32_t(prompt.size()), 0); !e.empty()) {
        std::fprintf(stderr, "pipeline: %s\n", e.c_str()); return 1;
    }
    std::vector<sycl::half> l16(cfg.vocab);
    qB.memcpy(l16.data(), B.logits(), cfg.vocab * 2).wait();
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
        qB.memcpy(l16.data(), B.logits(), cfg.vocab * 2).wait();
        mx = -1e30f;
        for (uint32_t i = 0; i < cfg.vocab; ++i)
            if (float(l16[i]) > mx) { mx = float(l16[i]); tok = int32_t(i); }
    }
    check(pipe_decode == ref_decode, "4-step greedy decode identical");
    {
        bool every = true;
        for (uint32_t s = 0; s < N; ++s) every = every && calls[s] == chains;
        std::printf("      chain forwards %llu; stage calls:", (unsigned long long)chains);
        for (uint32_t s = 0; s < N; ++s) std::printf(" %llu", (unsigned long long)calls[s]);
        std::printf("\n");
        check(every, "every stage ran on every forward of the chain");
    }

    std::printf("\n\033[1m4. pipelined prefill: stage 0 ahead, the rest chain behind%s\033[0m\n", p2p ? " (host bounce vs P2P double-bank)" : "");
    {
        // Chunk-PIPELINED prefill (stage 0 runs chunk k+1 while the rest of the chain runs chunk k, as the engine's
        // fwd_pipelined does). Variant 0: DOUBLE HOST buffers. Variant 1 (2 stages, IE_P2P): the push into B's double
        // device banks. Both replay the identical token stream from a reset state; the final-chunk logits must match the
        // SERIAL chain's BIT-EXACTLY (and each other's).
        std::vector<int32_t> chunk(4096);
        for (int i = 0; i < 4096; ++i) chunk[i] = 2000 + (i * 97) % 200000;
        std::vector<std::vector<float>> wbuf(2, std::vector<float>(uint64_t(1024) * cfg.hc_count * cfg.hidden));
        const bool have_banks = p2p && B.alloc_wide_bank2().empty();
        if (p2p && !have_banks) std::fprintf(stderr, "      (bank2 alloc failed — bank variant skipped)\n");
        std::vector<sycl::half> lg[3];
        double ms[3] = {0, 0, 0};
        // variant 2: the serial chain over the same chunks (the reference the pipelined variants are held to)
        for (int variant = 2; variant >= 0; --variant) {   // the serial chain first (the reference), then the pipelined variants
            if (variant == 1 && !have_banks) continue;
            const bool banks = variant == 1;
            for (auto& M : S) M->reset_state();
            A.set_wide_peer(nullptr); p2p_hand = false;
            B.use_wide_bank(0);
            // context: the fixed 8-token prompt, serial host handoff
            if (auto e = pipe_fwd(prompt.data(), 8, 0); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
            const auto t0 = std::chrono::steady_clock::now();
            if (variant == 2) {
                for (int k = 0; k < 4; ++k)
                    if (auto e = pipe_fwd(chunk.data() + k * 1024, 1024, 8 + uint32_t(k) * 1024); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
            } else {
                std::future<std::string> bfut;
                for (int k = 0; k < 4; ++k) {
                    float* wb = wbuf[k & 1].data();
                    if (banks) A.set_wide_peer(B.wide_bank(k & 1));
                    if (int(0) != skip)
                        if (auto e = A.forward_range(chunk.data() + k * 1024, 1024, 8 + k * 1024, nullptr, wb, nullptr); !e.empty()) {
                            std::fprintf(stderr, "A: %s\n", e.c_str()); return 1;
                        }
                    if (bfut.valid())
                        if (auto e = bfut.get(); !e.empty()) { std::fprintf(stderr, "rest: %s\n", e.c_str()); return 1; }
                    const int bank = k & 1;
                    const int32_t* bt = chunk.data() + k * 1024;
                    bfut = std::async(std::launch::async, [&S, &B, N, wb, k, banks, bank, bt, skip]() {
                        std::string e;
                        for (uint32_t s = 1; s < N && e.empty(); ++s) {
                            if (int(s) == skip) continue;
                            const bool last = s + 1 == N;
                            if (last && banks) B.use_wide_bank(uint32_t(bank));
                            e = S[s]->forward_range(last ? nullptr : bt, 1024, 8 + uint32_t(k) * 1024,
                                                    (last && banks) ? nullptr : wb, last ? nullptr : wb, nullptr,
                                                    /*wide_in_device=*/last && banks);
                        }
                        return e;
                    });
                }
                if (auto e = bfut.get(); !e.empty()) { std::fprintf(stderr, "rest: %s\n", e.c_str()); return 1; }
            }
            ms[variant] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            A.set_wide_peer(nullptr);
            B.use_wide_bank(0);
            lg[variant].resize(cfg.vocab);
            qB.memcpy(lg[variant].data(), B.logits(), cfg.vocab * 2).wait();
            std::printf("      prefill 4x1024 %s: %.0f ms (%.1f tok/s)\n",
                        variant == 2 ? "serial chain" : banks ? "pipelined [P2P banks]" : "pipelined [host bounce]", ms[variant], 4096000.0 / ms[variant]);
        }
        {
            uint32_t nd = 0;
            for (uint32_t i = 0; i < cfg.vocab; ++i) if (std::memcmp(&lg[0][i], &lg[2][i], 2) != 0) ++nd;
            check(nd == 0, "pipelined (host bounce) final-chunk logits BIT-IDENTICAL to the serial chain");
        }
        if (have_banks) {
            uint32_t nd = 0;
            for (uint32_t i = 0; i < cfg.vocab; ++i) if (std::memcmp(&lg[0][i], &lg[1][i], 2) != 0) ++nd;
            check(nd == 0, "P2P-bank pipelined logits BIT-IDENTICAL to host bounce");
            std::printf("      pipeline speedup from P2P banks: %.2fx\n", ms[0] / ms[1]);
        }
        pos = 8 + 4096;
        for (auto& M : S) { M->ecache_hits = M->ecache_misses = 0; }
        if (p2p) { A.set_wide_peer(B.wide_device()); p2p_hand = true; }
        for (int w = 0; w < 24; ++w) { pipe_fwd(&tok, 1, pos); ++pos; }   // cache warm
        const auto t1 = std::chrono::steady_clock::now();
        for (int st = 0; st < 16; ++st) { pipe_fwd(&tok, 1, pos); ++pos; }
        const double dms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count() / 16.0;
        std::printf("      decode steady: %.1f ms/token (%.1f tok/s)  [expert-cache hit per stage:", dms, 1000.0 / dms);
        for (auto& M : S) std::printf(" %.0f%%", 100.0 * M->ecache_hits / std::max<uint64_t>(1, M->ecache_hits + M->ecache_misses));
        std::printf("]\n");
    }

    std::printf("\n\033[1m%s\033[0m  (%d failure%s)\n\n",
                g_fail ? "\033[31mGATE FAILED\033[0m" : "\033[32mGATE PASSED\033[0m",
                g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
