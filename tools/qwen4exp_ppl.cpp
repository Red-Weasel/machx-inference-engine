// tools/qwen4exp_ppl.cpp — Flash-Next perplexity (quality gate for daily use
// + the certification instrument for quant-residency trades like
// IE_Q4E_DENSE_Q8). Mirrors qwen3next_ppl.cpp's streaming-T=1 NLL loop:
// forward(ids[i], 1, i) -> logits predict ids[i+1]; PPL = exp(mean NLL).
// Optional --chunk C runs one C-token prefill first (exercises the batched
// prefill path; a clean chunk tracks the streaming baseline).
//
// usage: ie-qwen4exp-ppl <model.gguf> [--tokens N] [--chunk C] [--text file] [--gpu G]
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/qwen4exp.hpp"
#include "ie/tokenizer.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace ie;

namespace {
// COPY of qwen3next_ppl.cpp's corpus opening (public-domain prose); repeated
// below to reach --tokens. Same-text comparability with the house PPL tools.
const char* kSampleText =
    "I am by birth a Genevese; and my family is one of the most distinguished "
    "of that republic. My ancestors had been for many years counsellors and "
    "syndics; and my father had filled several public situations with honour "
    "and reputation. He was respected by all who knew him for his integrity "
    "and indefatigable attention to public business. He passed his younger "
    "days perpetually occupied by the affairs of his country; and it was not "
    "until the decline of life that he thought of marrying, and bestowing on "
    "the state sons who might carry his virtues and his name down to "
    "posterity. As the circumstances of his marriage illustrate his "
    "character, I cannot refrain from relating them.\n\n"
    "During the years that followed, the household preserved a quiet discipline "
    "which gave dignity to ordinary labor. The library was small, but every "
    "volume had been chosen with care, and the books were read until their "
    "margins carried traces of many hands. At evening the shutters were closed, "
    "the lamp was trimmed, and the younger children listened while letters from "
    "distant friends were read aloud. These letters spoke of voyages, harvests, "
    "public debates, and the patient work by which families keep faith with one "
    "another across time and weather.\n\n";

double nll_of_target(const std::vector<float>& lg, int32_t target) {
    float mx = lg[0];
    for (float v : lg) mx = std::max(mx, v);
    double sum = 0.0;
    for (float v : lg) sum += std::exp(double(v) - mx);
    return -(double(lg[target]) - mx - std::log(sum));
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.gguf> [--tokens N] [--chunk C] [--text f] [--gpu G]\n", argv[0]); return 2; }
    uint32_t max_tokens = 1024, chunk = 0, gpu = 0;
    std::string text_path;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--tokens" && i + 1 < argc) max_tokens = uint32_t(std::atoi(argv[++i]));
        else if (a == "--chunk" && i + 1 < argc) chunk = uint32_t(std::atoi(argv[++i]));
        else if (a == "--text" && i + 1 < argc) text_path = argv[++i];
        else if (a == "--gpu" && i + 1 < argc) gpu = uint32_t(std::atoi(argv[++i]));
    }

    GgufReader g;
    if (auto e = g.open(argv[1]); !e.empty()) { std::fprintf(stderr, "open: %s\n", e.c_str()); return 1; }
    Qwen4ExpConfig cfg;
    if (auto e = read_qwen4exp_config(g, cfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }
    Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1; }
    DeviceAllocator alloc;
    if (auto e = alloc.init("B70", gpu); !e.empty()) { std::fprintf(stderr, "gpu: %s\n", e.c_str()); return 1; }
    Qwen4ExpModel m;
    if (auto e = m.load(alloc, g, cfg); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    const uint32_t max_ctx = std::max(4096u, max_tokens + 8);
    if (auto e = m.init_runtime(max_ctx, std::max(1024u, chunk)); !e.empty()) { std::fprintf(stderr, "rt: %s\n", e.c_str()); return 1; }

    std::string text = kSampleText;
    if (!text_path.empty()) {
        std::ifstream f(text_path);
        if (!f.good()) {
            // FAIL HARD: a silent fallback to the repeated sample text made
            // PPL collapse to ~1.1 and cost an hour of kernel-bug hunting
            // when /tmp lost the corpus (2026-08-27).
            std::fprintf(stderr, "ppl: cannot read --text %s\n", text_path.c_str());
            return 1;
        }
        std::stringstream ss; ss << f.rdbuf();
        if (!ss.str().empty()) text = ss.str();
    }
    std::vector<int32_t> ids = tok.encode(text, true);
    while (ids.size() < size_t(max_tokens) + 1 && !text.empty()) {
        text += "\n\n";
        text += (text_path.empty() ? kSampleText : text.substr(0, text.size() / 2));
        ids = tok.encode(text, true);
        if (text.size() > 1u << 20) break;
    }
    if (ids.size() > max_tokens + 1) ids.resize(max_tokens + 1);
    const bool chunked_score = std::getenv("IE_PPL_CHUNKED") != nullptr;
    const uint32_t N = uint32_t(ids.size());
    const uint32_t V = cfg.vocab;
    std::printf("ie-qwen4exp-ppl\n  tokens: %u  chunk: %u  dense_q8: %s\n",
                N, chunk, qwen4exp_dense_q8() ? "ON" : "off");
    if (N < 2) { std::fprintf(stderr, "need tokens\n"); return 1; }

    std::vector<sycl::half> l16(V);
    std::vector<float>      lg(V);
    double total_nll = 0.0;
    uint32_t counted = 0;
    auto score_last = [&](int32_t target) {
        alloc.queue().memcpy(l16.data(), m.logits(), V * 2).wait();
        for (uint32_t v = 0; v < V; ++v) lg[v] = float(l16[v]);
        total_nll += nll_of_target(lg, target);
        ++counted;
    };

    m.reset_state();
    if (chunked_score && chunk > 0) {
        // Chunk-prefill scoring: every forward is the PREFILL path (the
        // batched MoE/attention kernels), logits for all rows via
        // logits_rows — gates prefill-numerics changes (e.g. XMX experts)
        // directly instead of through the decode path.
        auto* d_rows = static_cast<sycl::half*>(
            alloc.malloc(uint64_t(chunk) * V * 2));
        std::vector<sycl::half> rows(uint64_t(chunk) * V);
        uint32_t pos = 0;
        while (pos + 1 < N) {
            const uint32_t n = std::min(chunk, N - pos);
            if (n < 2 && pos > 0) break;
            if (auto e = m.forward(&ids[pos], n, pos, nullptr); !e.empty()) {
                std::fprintf(stderr, "fwd @%u: %s\n", pos, e.c_str()); return 1;
            }
            if (auto e = m.logits_rows(n, d_rows); !e.empty()) {
                std::fprintf(stderr, "rows: %s\n", e.c_str()); return 1;
            }
            alloc.queue().memcpy(rows.data(), d_rows, uint64_t(n) * V * 2).wait();
            for (uint32_t r = 0; r + pos + 1 < N && r < n; ++r) {
                const sycl::half* lr = rows.data() + uint64_t(r) * V;
                for (uint32_t v = 0; v < V; ++v) lg[v] = float(lr[v]);
                total_nll += nll_of_target(lg, ids[pos + r + 1]);
                ++counted;
            }
            pos += n;
            std::printf("  ... %u/%u\n", pos, N); std::fflush(stdout);
        }
        alloc.free(d_rows);
        const double avg2 = counted ? total_nll / counted : 0.0;
        std::printf("  avg NLL: %.6f nats/token\n  PPL    : %.4f\n",
                    avg2, std::exp(avg2));
        return 0;
    }
    uint32_t i0 = 0;
    if (chunk > 0 && N > chunk + 1) {
        if (auto e = m.forward(&ids[0], chunk, 0, nullptr); !e.empty()) {
            std::fprintf(stderr, "prefill: %s\n", e.c_str()); return 1;
        }
        score_last(ids[chunk]);
        i0 = chunk;
    }
    for (uint32_t i = i0; i + 1 < N; ++i) {
        if (auto e = m.forward(&ids[i], 1, i, nullptr); !e.empty()) {
            std::fprintf(stderr, "fwd @%u: %s\n", i, e.c_str()); return 1;
        }
        score_last(ids[i + 1]);
        if ((i % 256) == 255) { std::printf("  ... %u/%u\n", i + 1, N); std::fflush(stdout); }
    }
    const double avg = counted ? total_nll / counted : 0.0;
    std::printf("  avg NLL: %.6f nats/token\n  PPL    : %.4f\n", avg, std::exp(avg));
    return 0;
}
