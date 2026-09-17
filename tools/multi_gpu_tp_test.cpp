// tools/multi_gpu_tp_test.cpp — TP correctness proof. Loads a dense GGUF (default
// qwen3-8b — fits one GPU) single-GPU (DenseModel) and 2-GPU TENSOR-PARALLEL
// (DenseModelTP), and checks (TP is NOT bit-exact — reduction order differs):
//   (1) PREFILL last-token logits cosine >= 0.999 + same argmax.
//   (2) GREEDY GENERATION: argmax token sequences match (report mismatches).
//   (3) coherent TP-generated text.
// Both sides use the fp16 decode path (IE_NO_Q8_DECODE on) so the only difference
// is the TP split + all-reduce. Serialized loads (single frees before TP loads).
//
// usage: ie-multi-gpu-tp-test --gguf <dense.gguf> [-p "prompt"] [-n N]
#include "ie/allocator.hpp"
#include "ie/dense_tp.hpp"
#include "ie/dense_transformer.hpp"
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/model_config.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
int argmax_half(const std::vector<sycl::half>& v) {
    int best = 0; float bv = -1e30f;
    for (size_t i = 0; i < v.size(); ++i) { float f = float(v[i]); if (f > bv) { bv = f; best = int(i); } }
    return best;
}
double cosine(const std::vector<sycl::half>& a, const std::vector<sycl::half>& b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double x = float(a[i]), y = float(b[i]);
        dot += x * y; na += x * x; nb += y * y;
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}
}  // namespace

int main(int argc, char** argv) {
    setenv("IE_NO_Q8_DECODE", "1", 1);   // fp16 decode on the single-GPU side
    std::string gguf = std::string(getenv("HOME") ? getenv("HOME") : ".") +
                       "/.seal/models/Qwen3-8B-Q4_K_M.gguf";
    std::string prompt = "The capital of France is the city of";
    int n_gen = 16;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf" && i + 1 < argc) gguf   = argv[++i];
        else if (a == "-p"     && i + 1 < argc) prompt = argv[++i];
        else if (a == "-n"     && i + 1 < argc) n_gen  = std::atoi(argv[++i]);
    }

    ie::GgufReader g;
    if (auto e = g.open(gguf); !e.empty()) { std::fprintf(stderr, "gguf: %s\n", e.c_str()); return 1; }
    if (!ie::is_dense_arch(ie::detect_arch(g))) { std::fprintf(stderr, "not a dense GGUF\n"); return 1; }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tok: %s\n", e.c_str()); return 1; }
    ie::DenseConfig cfg;
    if (auto e = ie::read_dense_config_auto(g, cfg); !e.empty()) { std::fprintf(stderr, "cfg: %s\n", e.c_str()); return 1; }

    const auto ids = tok.encode(prompt, /*allow_special=*/true);
    const uint32_t T = uint32_t(ids.size());
    if (T < 2) { std::fprintf(stderr, "need a multi-token prompt (T>=2)\n"); return 1; }
    std::printf("prompt='%s' T=%u  vocab=%u  layers=%u  n_gen=%d\n", prompt.c_str(), T, cfg.vocab, cfg.n_layers, n_gen);

    ie::DeviceFleet fleet;
    if (auto e = fleet.init(2); !e.empty()) { std::fprintf(stderr, "fleet: %s\n", e.c_str()); return 1; }
    if (fleet.size() < 2) { std::fprintf(stderr, "need 2 GPUs (have %u)\n", fleet.size()); return 1; }

    const uint64_t VB = uint64_t(cfg.vocab);
    std::vector<sycl::half> logits_single(VB), logits_tp(VB);
    std::vector<int> toks_single, toks_tp;

    // ---- single-GPU (DenseModel on dev 0), then free before TP loads ----
    {
        auto& alloc = fleet.dev(0);
        auto& q = alloc.queue();
        ie::DenseModel m;
        if (auto e = m.load(alloc, g, cfg); !e.empty()) { std::fprintf(stderr, "single load: %s\n", e.c_str()); return 1; }
        ie::KvCache kv;
        if (auto e = kv.init(alloc, ie::KvCacheConfig{cfg.n_layers, cfg.n_kv_heads, 2048, cfg.head_dim}); !e.empty())
            { std::fprintf(stderr, "kv: %s\n", e.c_str()); return 1; }
        kv.reset();
        if (auto e = m.ensure_workspace(T); !e.empty()) { std::fprintf(stderr, "ws: %s\n", e.c_str()); return 1; }
        auto* d_ids = sycl::malloc_device<int32_t>(2048, q);
        auto* d_log = sycl::malloc_device<sycl::half>(VB, q);
        q.memcpy(d_ids, ids.data(), T * sizeof(int32_t)).wait();
        m.forward(q, d_ids, T, 0, kv, d_log).wait();
        q.memcpy(logits_single.data(), d_log, VB * sizeof(sycl::half)).wait();
        int tk = argmax_half(logits_single);
        toks_single.push_back(tk);
        uint32_t pos = T;
        for (int s = 1; s < n_gen; ++s) {
            q.memcpy(d_ids, &tk, sizeof(int32_t)).wait();
            m.forward(q, d_ids, 1, pos, kv, d_log).wait();
            std::vector<sycl::half> hl(VB);
            q.memcpy(hl.data(), d_log, VB * sizeof(sycl::half)).wait();
            tk = argmax_half(hl); toks_single.push_back(tk); ++pos;
        }
        sycl::free(d_ids, q); sycl::free(d_log, q);
        std::printf("single-GPU generation done\n");
    }

    // ---- 2-GPU tensor-parallel (DenseModelTP) ----
    {
        ie::DenseModelTP mt;
        if (auto e = mt.load(fleet, g, cfg); !e.empty()) { std::fprintf(stderr, "tp load: %s\n", e.c_str()); return 1; }
        auto db = mt.device_bytes();
        std::printf("tp: every layer split across 2 cards  (weights dev0=%.1fGB dev1=%.1fGB)\n",
                    db.size() > 0 ? db[0] / 1e9 : 0.0, db.size() > 1 ? db[1] / 1e9 : 0.0);
        if (auto e = mt.forward(ids.data(), T, 0, /*reset_kv=*/true, logits_tp.data()); !e.empty())
            { std::fprintf(stderr, "tp prefill: %s\n", e.c_str()); return 1; }
        int tk = argmax_half(logits_tp);
        toks_tp.push_back(tk);
        uint32_t pos = T;
        std::vector<sycl::half> hl(VB);
        for (int s = 1; s < n_gen; ++s) {
            if (auto e = mt.forward(&tk, 1, pos, /*reset_kv=*/false, hl.data()); !e.empty())
                { std::fprintf(stderr, "tp decode: %s\n", e.c_str()); return 1; }
            tk = argmax_half(hl); toks_tp.push_back(tk); ++pos;
        }
        std::printf("2-GPU TP generation done\n");
    }

    // ---- (1) prefill cosine + argmax ----
    const double cos = cosine(logits_single, logits_tp);
    const int am_s = argmax_half(logits_single), am_t = argmax_half(logits_tp);
    const bool cos_ok = cos >= 0.999, am_ok = (am_s == am_t);
    std::printf("\n(1) prefill last-token: cosine=%.6f %s, argmax %s (single=%d tp=%d)\n",
                cos, cos_ok ? "OK" : "LOW", am_ok ? "MATCH" : "DIFFER", am_s, am_t);

    // ---- (2) greedy token sequences ----
    int tok_mism = 0;
    for (int i = 0; i < n_gen; ++i) if (toks_single[i] != toks_tp[i]) ++tok_mism;
    std::printf("(2) greedy %d tokens: %s (%d differ)\n", n_gen,
                tok_mism == 0 ? "IDENTICAL" : "differ (fp16 ties possible)", tok_mism);
    std::printf("(3) TP text: '%s'\n", tok.decode(toks_tp, true, {}).c_str());
    std::printf("    single  : '%s'\n", tok.decode(toks_single, true, {}).c_str());

    const bool ok = cos_ok && am_ok && tok_mism == 0;
    if (ok) { std::printf("\n\033[32mPASS\033[0m: TP matches single-GPU (cosine %.5f, argmax match, greedy %d/%d)\n", cos, n_gen, n_gen); return 0; }
    std::printf("\n%s: cosine %.5f, argmax %s, %d/%d greedy match\n",
                (cos_ok && am_ok) ? "\033[33mPARTIAL\033[0m" : "\033[31mFAIL\033[0m",
                cos, am_ok ? "match" : "differ", n_gen - tok_mism, n_gen);
    return (cos_ok && am_ok) ? 0 : 1;
}
