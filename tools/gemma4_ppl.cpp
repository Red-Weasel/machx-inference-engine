// tools/gemma4_ppl.cpp — Gemma 4 perplexity gate. Streaming T=1 NLL (mirrors
// ie-perplexity): forward(ids[i], start_pos=i) predicts ids[i+1];
// NLL += -log softmax(logits)[ids[i+1]]; PPL = exp(mean NLL). Validates the
// forward NUMERICALLY (beyond greedy argmax). Compare to llama-perplexity on the
// same text. Default 128 tokens (the correctness-first forward is slow).
#include "ie/gemma4.hpp"
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/tokenizer.hpp"
#include "ie/kv_cache.hpp"
#include "ie/allocator.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static const char* kSampleText =
    "I am by birth a Genevese; and my family is one of the most distinguished "
    "of that republic. My ancestors had been for many years counsellors and "
    "syndics; and my father had filled several public situations with honour "
    "and reputation. He was respected by all who knew him for his integrity "
    "and indefatigable attention to public business. He passed his younger "
    "days perpetually occupied by the affairs of his country; and it was not "
    "until the decline of life that he thought of marrying, and bestowing on "
    "the state sons who might carry his virtues and his name down to posterity.";

static double nll_of_target(const std::vector<float>& logits, int32_t target) {
    float m = logits[0];
    for (size_t i = 1; i < logits.size(); ++i) m = std::max(m, logits[i]);
    double sum_exp = 0.0;
    for (float v : logits) sum_exp += std::exp(double(v) - double(m));
    return -(double(logits[target]) - (double(m) + std::log(sum_exp)));
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <gguf> [--max-tokens N] [--text FILE]\n", argv[0]); return 2; }
    const std::string gguf = argv[1];
    uint32_t max_tokens = 128;
    std::string text_path;
    bool reprefill = false;   // diagnostic: re-prefill ids[0..i] each step (no KV-accum dependency)
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--max-tokens" && i + 1 < argc) max_tokens = std::atoi(argv[++i]);
        else if (a == "--text" && i + 1 < argc) text_path = argv[++i];
        else if (a == "--reprefill") reprefill = true;
    }

    ie::GgufReader g;
    if (auto e = g.open(gguf); !e.empty()) { std::fprintf(stderr, "gguf: %s\n", e.c_str()); return 1; }
    ie::GemmaConfig cfg;
    if (auto e = ie::read_gemma4_config(g, cfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tok: %s\n", e.c_str()); return 1; }
    ie::DeviceAllocator alloc;
    if (auto e = alloc.init(); !e.empty()) { std::fprintf(stderr, "alloc: %s\n", e.c_str()); return 1; }
    ie::Gemma4Model m;
    if (auto e = m.load(alloc, g, cfg); !e.empty()) { std::fprintf(stderr, "LOAD: %s\n", e.c_str()); return 1; }

    std::string text = kSampleText;
    if (!text_path.empty()) { std::ifstream f(text_path); std::stringstream ss; ss << f.rdbuf(); if (!ss.str().empty()) text = ss.str(); }
    std::vector<int32_t> ids = tok.encode(text, /*allow_special=*/true);
    if (ids.size() > max_tokens + 1) ids.resize(max_tokens + 1);
    const uint32_t N = uint32_t(ids.size());
    const uint32_t V = cfg.vocab;

    if (auto e = m.ensure_workspace(1); !e.empty()) { std::fprintf(stderr, "ws: %s\n", e.c_str()); return 1; }
    if (auto e = m.ensure_kv(N + 2); !e.empty()) { std::fprintf(stderr, "kv: %s\n", e.c_str()); return 1; }

    if (reprefill) { if (auto e = m.ensure_workspace(N + 2); !e.empty()) { std::fprintf(stderr, "ws: %s\n", e.c_str()); return 1; } }
    auto& q = alloc.queue();
    auto* d_ids = sycl::malloc_device<int32_t>(N + 2, q);
    auto* d_logits = sycl::malloc_device<sycl::half>(V, q);
    std::vector<sycl::half> hlog(V);
    std::vector<float> logf(V);

    std::printf("ie-gemma4-perplexity (%s)\n  tokens: %u\n", reprefill ? "reprefill" : "streaming-T1", N);
    double total_nll = 0.0; uint32_t counted = 0;
    ie::KvCache kv;
    for (uint32_t i = 0; i + 1 < N; ++i) {
        if (reprefill) {
            q.memcpy(d_ids, ids.data(), uint64_t(i + 1) * sizeof(int32_t)).wait();
            m.forward(q, d_ids, i + 1, /*start_pos=*/0, kv, d_logits).wait_and_throw();
        } else {
            q.memcpy(d_ids, &ids[i], sizeof(int32_t)).wait();
            m.forward(q, d_ids, 1, /*start_pos=*/i, kv, d_logits).wait_and_throw();
        }
        q.memcpy(hlog.data(), d_logits, V * sizeof(sycl::half)).wait();
        for (uint32_t v = 0; v < V; ++v) logf[v] = float(hlog[v]);
        total_nll += nll_of_target(logf, ids[i + 1]);
        ++counted;
        if ((i & 31) == 31) std::printf("  ... %u/%u  running NLL %.4f\n", i + 1, N, total_nll / counted);
    }
    const double avg = total_nll / std::max(1u, counted);
    std::printf("  avg NLL : %.6f\n  perplexity : %.4f\n", avg, std::exp(avg));
    sycl::free(d_ids, q); sycl::free(d_logits, q);
    return 0;
}
