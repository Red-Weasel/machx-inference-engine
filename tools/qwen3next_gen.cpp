// tools/qwen3next_gen.cpp — Qwen3-Next-80B greedy-generation smoke (Step 2.5a).
// Loads the GGUF split across N GPUs (EXACT load path from qwen3next_fwd_test.cpp:
// GgufReader::open → read_qwen3next_config → DeviceFleet::init → LayerPlan::contiguous
// → Qwen3NextModel::load), tokenizes the prompt with the engine's GGUF-built
// tokenizer (gpt2/qwen family, same as the crown — encode() auto-prepends BOS when
// add_bos_token is set in the GGUF, exactly as qwen3moe_test.cpp relies on), runs the
// forward to greedy-decode n_new tokens, and prints the generated text + token ids.
// This is the correctness smoke for the completed forward — the lead runs it on
// "The capital of France is" expecting " Paris".
// RUN ONLY with the cards free + serialized (see loader GPU-safety rules).
#include "ie/qwen3next.hpp"
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
                     "usage: %s <gguf> <n_gpus> \"<prompt>\" [n_new=8]\n", argv[0]);
        return 2;
    }
    const std::string gguf  = argv[1];
    const uint32_t    n_gpus = uint32_t(std::stoul(argv[2]));
    const std::string prompt = argv[3];
    const int         n_new  = (argc > 4) ? std::atoi(argv[4]) : 8;

    // --- Load path: identical to tools/qwen3next_fwd_test.cpp -----------------
    ie::GgufReader g;
    std::string err = g.open(gguf);
    if (!err.empty()) { std::fprintf(stderr, "gguf open: %s\n", err.c_str()); return 1; }

    ie::Qwen3NextConfig cfg;
    err = ie::read_qwen3next_config(g, cfg);
    if (!err.empty()) { std::fprintf(stderr, "config: %s\n", err.c_str()); return 1; }

    // Tokenizer: built straight from the GGUF (gpt2/qwen-style byte-BPE; vocab +
    // merges + special ids + add_bos all read from tokenizer.ggml.* keys). Same
    // construction the crown/qwen3moe path uses (qwen3moe_test.cpp:26-27).
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

    // --- Tokenize ------------------------------------------------------------
    // encode() prepends BOS itself when the GGUF sets add_bos_token (mirrors the
    // crown/qwen3moe path in qwen3moe_test.cpp:62 — no manual BOS handling).
    std::vector<int32_t> ids = tok.encode(prompt, /*allow_special=*/true);
    if (ids.empty()) { std::fprintf(stderr, "encode produced 0 tokens\n"); return 1; }
    const uint32_t T = uint32_t(ids.size());
    std::printf("prompt: '%s'\n", prompt.c_str());
    std::printf("prompt ids (%u, add_bos=%d bos=%d):", T,
                int(tok.add_bos_token()), tok.bos_token_id());
    for (int32_t id : ids) std::printf(" %d", id);
    std::printf("\n");

    std::vector<sycl::half> logits(V);

    auto argmax = [&]() -> int32_t {
        int32_t best_i = 0; float best = -1e30f;
        for (uint32_t i = 0; i < V; ++i) {
            const float v = float(logits[i]);
            if (v > best) { best = v; best_i = int32_t(i); }
        }
        return best_i;
    };

    // --- Prefill -------------------------------------------------------------
    err = m.forward(ids.data(), T, /*start_pos=*/0, /*reset_kv=*/true, logits.data());
    if (!err.empty()) { std::fprintf(stderr, "PREFILL FWD FAILED: %s\n", err.c_str()); return 1; }

    // Prefill top-5 (id + logit) — lets the lead diff against the HF oracle.
    {
        std::vector<uint32_t> idx(V);
        for (uint32_t i = 0; i < V; ++i) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                          [&](uint32_t a, uint32_t b) {
                              return float(logits[a]) > float(logits[b]);
                          });
        std::printf("prefill top-5:");
        for (int k = 0; k < 5; ++k)
            std::printf(" [%u %.4f]", idx[k], float(logits[idx[k]]));
        std::printf("\n");
    }

    // --- Greedy decode -------------------------------------------------------
    std::vector<int32_t> gen;
    int32_t next = argmax();
    gen.push_back(next);
    uint32_t start_pos = T;  // KV already holds T prefill tokens

    for (int s = 1; s < n_new; ++s) {
        if (next == tok.eos_token_id()) break;
        err = m.forward(&next, 1, /*start_pos=*/start_pos, /*reset_kv=*/false,
                        logits.data());
        if (!err.empty()) { std::fprintf(stderr, "DECODE FWD FAILED (step %d): %s\n", s, err.c_str()); return 1; }
        next = argmax();
        gen.push_back(next);
        ++start_pos;
    }

    // --- Report --------------------------------------------------------------
    std::printf("generated ids (%zu):", gen.size());
    for (int32_t id : gen) std::printf(" %d", id);
    std::printf("\n");
    std::printf("generated text: '%s'\n",
                tok.decode(gen, /*skip_special=*/true, {}).c_str());
    return 0;
}
