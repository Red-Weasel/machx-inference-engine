// tools/laguna_run.cpp — ie-laguna-run: Laguna-S-2.1 v0 bring-up (docs/laguna/00_PORT_PLAN.md, P4).
// Loads the model on one Arc card or two (LagunaModel), checks every quant binding, runs token ids, prints per-layer
// residual stats of the last row (compare with llama.cpp's llama-eval-callback `l_out-N`) and the top-10 logits; optionally
// greedy-decodes N tokens.
//   usage: ie-laguna-run <shard-1.gguf> --ids 1,2,3 [--gen N] [--ctx 4096] [--chunk 256] [--card 0] [--cards 1|2] [--probe]
//          [--self-check]          (--cards 2: cards --card and --card + 1)
#include "ie/gguf.hpp"
#include "ie/laguna.hpp"
#include "ie/model_config.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: ie-laguna-run <shard-1.gguf> --ids a,b,c [--gen N] [--ctx C] [--chunk T] [--card N] [--cards 1|2] [--probe] [--self-check]\n"); return 2; }
    const std::string path = argv[1];
    std::vector<int32_t> ids;
    uint32_t gen = 0, ctx = 4096, chunk = 256, card = 0, cards = 1;
    bool probe = false, self_check = false;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--ids" && i + 1 < argc) {
            std::string s = argv[++i];
            for (size_t p = 0; p < s.size();) { size_t c = s.find(',', p); if (c == std::string::npos) c = s.size(); ids.push_back(std::atoi(s.substr(p, c - p).c_str())); p = c + 1; }
        } else if (a == "--gen" && i + 1 < argc) gen = uint32_t(std::atoi(argv[++i]));
        else if (a == "--ctx" && i + 1 < argc) ctx = uint32_t(std::atoi(argv[++i]));
        else if (a == "--chunk" && i + 1 < argc) chunk = uint32_t(std::atoi(argv[++i]));
        else if (a == "--card" && i + 1 < argc) card = uint32_t(std::atoi(argv[++i]));
        else if (a == "--cards" && i + 1 < argc) cards = uint32_t(std::atoi(argv[++i]));
        else if (a == "--probe") probe = true;
        else if (a == "--self-check") self_check = true;
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (ids.empty()) { std::fprintf(stderr, "--ids required\n"); return 2; }

    ie::GgufReader g;
    if (auto e = g.open(path); !e.empty()) { std::fprintf(stderr, "gguf: %s\n", e.c_str()); return 1; }
    ie::LagunaConfig cfg;
    if (auto e = ie::read_laguna_config(g, cfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }
    std::printf("laguna: %u layers, hidden %u, %u experts top-%u (+%u shared), vocab %u, window %u, dense layers %u\n",
                cfg.n_layers, cfg.hidden, cfg.n_experts, cfg.n_experts_used, cfg.shared_expert_ffn, cfg.vocab, cfg.sliding_window, cfg.dense_layers);

    std::vector<sycl::device> arcs;
    for (const auto& pl : sycl::platform::get_platforms()) {
        if (pl.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : pl.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) arcs.push_back(d);
    }
    if (cards < 1 || cards > 2 || card + cards > arcs.size()) { std::fprintf(stderr, "no Arc GPU #%u (+%u)\n", card, cards - 1); return 1; }
    // each card's queue in its own single-device context
    auto mk = [](const sycl::device& d) { return std::make_unique<sycl::queue>(sycl::context(d), d, sycl::property::queue::in_order{}); };
    std::unique_ptr<sycl::queue> q = mk(arcs[card]), q2 = cards == 2 ? mk(arcs[card + 1]) : nullptr;
    std::printf("device: %s%s\n", q->get_device().get_info<sycl::info::device::name>().c_str(), cards == 2 ? " x2" : "");

    ie::LagunaModel m;
    auto t0 = std::chrono::steady_clock::now();
    if (auto e = m.load(*q, g, cfg, q2.get()); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    std::printf("loaded in %.1f s, %.2f GiB on the device\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), m.device_bytes() / 1073741824.0);
    if (auto e = m.init_runtime(ctx, chunk); !e.empty()) { std::fprintf(stderr, "runtime: %s\n", e.c_str()); return 1; }
    std::printf("runtime: ctx %u, chunk %u, %.2f GiB on the device\n", ctx, chunk, m.device_bytes() / 1073741824.0);
    if (self_check) {
        if (auto e = m.self_check(); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
    }
    if (probe) {
        m.probe = [&](uint32_t L, sycl::queue& pq, const float* x, uint32_t T) {
            std::vector<float> h(uint64_t(T) * cfg.hidden);
            pq.memcpy(h.data(), x, h.size() * 4).wait();
            double s = 0; float mx = 0; uint64_t at = 0, nan = 0;
            for (uint64_t i = 0; i < h.size(); ++i) { const float v = h[i]; if (!std::isfinite(v)) { ++nan; continue; } s += v; if (std::fabs(v) > mx) { mx = std::fabs(v); at = i; } }
            const float* r = h.data() + uint64_t(T - 1) * cfg.hidden;
            std::printf("l_out-%-2u sum %+.6e | last row [0..2] %+.4f %+.4f %+.4f | max|x| %.1f (row %llu) | non-finite %llu\n", L, s, r[0], r[1], r[2],
                        mx, (unsigned long long)(at / cfg.hidden), (unsigned long long)nan);
        };
    }

    std::vector<float> logits(cfg.vocab);
    t0 = std::chrono::steady_clock::now();
    if (auto e = m.prefill(ids.data(), uint32_t(ids.size()), 0, logits.data()); !e.empty()) { std::fprintf(stderr, "forward: %s\n", e.c_str()); return 1; }
    const double pre_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("prompt %zu tokens in %.2f s (%.1f tok/s)\n", ids.size(), pre_s, ids.size() / pre_s);
    auto topk = [&](const char* tag) {
        std::vector<uint32_t> ix(cfg.vocab); std::iota(ix.begin(), ix.end(), 0u);
        std::partial_sort(ix.begin(), ix.begin() + 10, ix.end(), [&](uint32_t a, uint32_t b) { return logits[a] > logits[b]; });
        std::printf("%s top-10:", tag); for (int i = 0; i < 10; ++i) std::printf(" %u:%.3f", ix[i], logits[ix[i]]); std::printf("\n");
        return int32_t(ix[0]);
    };
    { double ls = 0; for (float v : logits) ls += v;
      std::printf("logits: sum %.4f | [0..2] %.4f %.4f %.4f | [V-3..V-1] %.4f %.4f %.4f\n", ls, logits[0], logits[1], logits[2],
                  logits[cfg.vocab - 3], logits[cfg.vocab - 2], logits[cfg.vocab - 1]); }
    int32_t next = topk("last prompt row");
    if (gen) {
        std::vector<int32_t> out;
        const uint64_t h0 = m.cache_hits(), m0 = m.cache_misses();
        t0 = std::chrono::steady_clock::now();
        uint32_t pos = uint32_t(ids.size());
        for (uint32_t i = 0; i < gen; ++i) {
            out.push_back(next);
            if (auto e = m.forward(&next, 1, pos++, logits.data()); !e.empty()) { std::fprintf(stderr, "decode: %s\n", e.c_str()); return 1; }
            next = int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
        }
        const double dec_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("greedy %u tokens in %.2f s (%.2f tok/s): ", gen, dec_s, gen / dec_s);
        for (int32_t t : out) std::printf("%d,", t);
        std::printf("\n");
        std::printf("decode expert lookups: %llu hits, %llu misses (%.1f misses/token)\n", (unsigned long long)(m.cache_hits() - h0),
                    (unsigned long long)(m.cache_misses() - m0), double(m.cache_misses() - m0) / gen);
    }
    std::printf("expert cache: %u slots/layer, %llu hits, %llu misses (%.1f%% hit rate); %.1f GiB pinned\n", m.cache_slots(),
                (unsigned long long)m.cache_hits(), (unsigned long long)m.cache_misses(),
                100.0 * m.cache_hits() / std::max<uint64_t>(1, m.cache_hits() + m.cache_misses()), m.pinned_bytes() / 1073741824.0);
    return 0;
}
