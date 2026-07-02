// tools/qwen3next_ppl.cpp — Qwen3-Next-80B perplexity (Step 2.5c).
// Mirrors ie_perplexity.cpp's streaming-T=1 NLL loop but on the split
// Qwen3NextModel (load path identical to qwen3next_gen.cpp). For each token i:
//   forward(ids[i], T=1, start_pos=i, reset_kv=(i==0)) -> logits predict ids[i+1]
//   NLL += -log softmax(logits)[ids[i+1]];  PPL = exp(mean NLL).
// The forward returns the last position's logits, so streaming T=1 gives each
// position's conditional distribution (KV/DeltaNet state carries the context).
// RUN ONLY with the cards free + serialized (loader GPU-safety rules).
#include "ie/qwen3next.hpp"
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {
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
    "another across time and weather.\n\n"
    "I learned early that knowledge is not gathered by haste alone. A page "
    "understood clearly was worth more than a chapter passed over in restless "
    "curiosity. My teachers encouraged questions, but they also required proof, "
    "comparison, and a willingness to revise an opinion when the evidence did "
    "not support it. In that habit I found a kind of freedom: the mind became "
    "less anxious when it could distinguish a bright guess from a settled fact.";

double nll_of_target(const std::vector<float>& logits, int32_t target_id) {
    float m = logits[0];
    for (size_t i = 1; i < logits.size(); ++i) m = std::max(m, logits[i]);
    double sum_exp = 0.0;
    for (float v : logits) sum_exp += std::exp(double(v) - double(m));
    const double lse = double(m) + std::log(sum_exp);
    return -(double(logits[target_id]) - lse);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <gguf> <n_gpus> [--max-tokens N] [--text FILE] "
                     "[--prefill-chunk N]\n", argv[0]);
        return 2;
    }
    const std::string gguf  = argv[1];
    const uint32_t    n_gpus = uint32_t(std::stoul(argv[2]));
    uint32_t max_tokens = 256;
    uint32_t prefill_chunk = 0;   // 0 = streaming T=1 (the validated baseline);
                                  // >0 = one big single-call prefill of N tokens
                                  // (stresses the DeltaNet recurrence over N
                                  // successive steps — the T<=256 cap test) then
                                  // streamed T=1 scoring for the remainder.
    std::string text_path;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--max-tokens"    && i + 1 < argc) max_tokens    = std::atoi(argv[++i]);
        else if (a == "--text"          && i + 1 < argc) text_path     = argv[++i];
        else if (a == "--prefill-chunk" && i + 1 < argc) prefill_chunk = std::atoi(argv[++i]);
    }

    ie::GgufReader g;
    std::string err = g.open(gguf);
    if (!err.empty()) { std::fprintf(stderr, "gguf open: %s\n", err.c_str()); return 1; }
    ie::Qwen3NextConfig cfg;
    err = ie::read_qwen3next_config(g, cfg);
    if (!err.empty()) { std::fprintf(stderr, "config: %s\n", err.c_str()); return 1; }
    ie::Tokenizer tok;
    err = tok.load_from_gguf(g);
    if (!err.empty()) { std::fprintf(stderr, "tok: %s\n", err.c_str()); return 1; }
    ie::DeviceFleet fleet;
    err = fleet.init(n_gpus, "B70");
    if (!err.empty()) { std::fprintf(stderr, "fleet: %s\n", err.c_str()); return 1; }
    const uint32_t n_layers = cfg.hybrid.n_transformer_layers();
    ie::LayerPlan plan = ie::LayerPlan::contiguous(n_layers, n_gpus);
    ie::Qwen3NextModel m;
    err = m.load(fleet, plan, g, cfg);
    if (!err.empty()) { std::fprintf(stderr, "LOAD FAILED: %s\n", err.c_str()); return 1; }
    const uint32_t V = cfg.hybrid.dense.vocab;

    std::string text = kSampleText;
    if (!text_path.empty()) {
        std::ifstream f(text_path); std::stringstream ss; ss << f.rdbuf();
        if (!ss.str().empty()) text = ss.str();
    }
    std::vector<int32_t> ids = tok.encode(text, /*allow_special=*/true);
    // --sweep needs >1k tokens to exercise prefill-chunk 1024; the built-in
    // corpus is short, so repeat it (re-encoding the concatenation, not the ids,
    // keeps tokenization honest). Repetition lowers the absolute PPL but collapse
    // detection only needs the relative jump (~4.7 -> 70+), which is unaffected.
    while (ids.size() < size_t(max_tokens) + 1 && !text.empty()) {
        text += "\n\n";
        text += (text_path.empty() ? kSampleText : text.substr(0, text.size() / 2));
        ids = tok.encode(text, /*allow_special=*/true);
        if (text.size() > 1u << 20) break;   // safety
    }
    if (ids.size() > max_tokens + 1) ids.resize(max_tokens + 1);
    const uint32_t N = uint32_t(ids.size());
    std::printf("ie-qwen3next-perplexity\n  tokens     : %u\n  gpus       : %u\n", N, n_gpus);
    if (N < 2) { std::fprintf(stderr, "need >=2 tokens\n"); return 1; }

    std::vector<sycl::half> logits(V);
    std::vector<float>      logf(V);

    // Run one full PPL pass at a given prefill-chunk and return {avg_nll, ppl}.
    // chunk==0 -> streaming T=1 (the validated baseline). chunk>0 -> one big
    // single-call prefill of `chunk` tokens (DeltaNet recurrence over `chunk`
    // successive steps — the T<=256 cap test) then streamed T=1 scoring for the
    // rest. A clean chunk tracks the baseline; if the §1 BMG recurrence bug
    // bites, the corrupted state makes every streamed token explode and PPL
    // collapses (cf. crown --prefill-chunk 1024 -> ~74).
    auto run_ppl = [&](uint32_t chunk) -> std::pair<double, double> {
        double total_nll = 0.0; uint32_t counted = 0;
        auto score_last = [&](int32_t target) {
            for (uint32_t v = 0; v < V; ++v) logf[v] = float(logits[v]);
            total_nll += nll_of_target(logf, target);
            ++counted;
        };
        uint32_t i0 = 0;
        if (chunk > 0 && N > chunk + 1) {
            std::string e = m.forward(&ids[0], chunk, /*start_pos=*/0, /*reset_kv=*/true, logits.data());
            if (!e.empty()) { std::fprintf(stderr, "PREFILL FAILED: %s\n", e.c_str()); return {-1, -1}; }
            score_last(ids[chunk]);
            i0 = chunk;
        }
        for (uint32_t i = i0; i + 1 < N; ++i) {
            std::string e = m.forward(&ids[i], 1, /*start_pos=*/i, /*reset_kv=*/(i == 0), logits.data());
            if (!e.empty()) { std::fprintf(stderr, "FWD FAILED @%u: %s\n", i, e.c_str()); return {-1, -1}; }
            score_last(ids[i + 1]);
        }
        const double avg = counted ? total_nll / counted : 0.0;
        return {avg, std::exp(avg)};
    };

    bool sweep = false;
    uint32_t repeats = 2;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--sweep") sweep = true;
        else if (a == "--repeats" && i + 1 < argc) repeats = std::max(1, std::atoi(argv[++i]));
    }

    if (!sweep) {
        std::printf("  mode       : %s\n",
                    prefill_chunk > 0 ? "prefill-chunk + streamed T=1" : "streaming T=1 (baseline)");
        auto [avg, ppl] = run_ppl(prefill_chunk);
        if (avg < 0) return 1;
        std::printf("  avg NLL    : %.6f nats / token\n", avg);
        std::printf("  perplexity : %.4f\n", ppl);
        std::printf("# TSV: tokens\tchunk\tavg_nll\tperplexity\n# %u\t%u\t%.6f\t%.4f\n",
                    N, prefill_chunk, avg, ppl);
        return 0;
    }

    // --sweep: one load, a battery of chunk sizes. Each over-cap chunk is run
    // `repeats` times — the §1 BMG divergence is STOCHASTIC (the doc: first
    // divergent chain typically appears between iter 7 and 500), so a single
    // clean pass is not proof. Two independent checks per chunk:
    //   (a) COLLAPSE — ppl >> the streaming baseline (corrupted DeltaNet state).
    //   (b) DIVERGENCE — any repeat's avg_nll differs from this chunk's first
    //       run (the bug fired on some iters but not others — non-determinism).
    std::printf("\n=== DeltaNet prefill-cap sweep (N=%u tokens, %u repeats/chunk) ===\n",
                N, repeats);
    std::printf("%-8s %-6s %-4s %12s %12s\n", "label", "chunk", "rep", "avg_nll", "perplexity");
    struct Cfg { const char* label; uint32_t chunk; };
    const Cfg battery[] = {
        {"stream", 0}, {"cap256", 256}, {"c512", 512}, {"c1024", 1024},
        {"full", N > 1 ? N - 1 : 0},
    };
    double base_ppl = -1;
    bool any_diverge = false, any_collapse = false;
    for (const auto& c : battery) {
        if (c.chunk > 0 && N <= c.chunk + 1) continue;   // not enough tokens
        const uint32_t R = (c.chunk == 0) ? 1u : repeats;
        double first_nll = -1;
        for (uint32_t r = 0; r < R; ++r) {
            auto [avg, ppl] = run_ppl(c.chunk);
            if (avg < 0) return 1;
            if (base_ppl < 0) base_ppl = ppl;
            if (first_nll < 0) first_nll = avg;
            const bool collapse = ppl > base_ppl * 3.0 + 5.0;
            const bool diverge  = std::abs(avg - first_nll) > 1e-9;
            any_collapse |= collapse; any_diverge |= diverge;
            const char* flag = collapse ? "  <== COLLAPSE" : diverge ? "  <== DIVERGED" : "";
            std::printf("%-8s %-6u %-4u %12.6f %12.4f%s\n", c.label, c.chunk, r, avg, ppl, flag);
        }
    }
    std::printf("\n# baseline ppl %.4f\n", base_ppl);
    std::printf("# VERDICT: %s\n",
                (any_collapse || any_diverge)
                    ? "DeltaNet bug BIT over-cap (collapse or divergence) -> keep T<=256"
                    : "CLEAN across all repeats -> over-cap prefill is safe for qwen3next");
    return 0;
}
