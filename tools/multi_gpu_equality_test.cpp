// tools/multi_gpu_equality_test.cpp — P-B: the decisive multi-GPU correctness
// proof. Loads a dense GGUF (default qwen3-8b — RAM-safe, fits one GPU) BOTH
// single-GPU (DenseModel) and 2-GPU layer-split (DenseModelSplit), and checks:
//   (1) PREFILL: last-token logits BIT-IDENTICAL across the whole vocab.
//   (2) GREEDY GENERATION: the argmax token sequences are identical.
// Layer-split is bit-identical math (same ops/order, just placed across devices),
// so any difference is a placement/boundary bug.
//
// Both sides use the FP16 decode path (IE_NO_Q8_DECODE forced on; no FA2-decode
// partials) so the single-GPU decode matches the split's gemv_q path exactly.
//
// RAM/VRAM-safe: ~5GB across both 32GB cards, ~7GB host. NEVER point this at the
// 72B (won't fit the single-GPU baseline; not the right tool for it).
//
// usage: ie-multi-gpu-equality-test --gguf <dense.gguf> [-p "prompt"] [-n N]
#include "ie/allocator.hpp"
#include "ie/dense_split.hpp"
#include "ie/dense_transformer.hpp"
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/model_config.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
int argmax_half(const std::vector<sycl::half>& v) {
    int best = 0; float bv = -1e30f;
    for (size_t i = 0; i < v.size(); ++i) { float f = float(v[i]); if (f > bv) { bv = f; best = int(i); } }
    return best;
}
}  // namespace

int main(int argc, char** argv) {
    setenv("IE_NO_Q8_DECODE", "1", 1);   // fp16 decode on both sides → comparable
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
    std::vector<sycl::half> logits_single(VB), logits_split(VB);
    std::vector<int> toks_single, toks_split;

    // ---- single-GPU (DenseModel on dev 0; NO attn-partials → naive attn) ----
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

    // ---- 2-GPU layer-split (DenseModelSplit) -----------------------------
    {
        ie::LayerPlan plan = ie::LayerPlan::contiguous(cfg.n_layers, 2);
        ie::DenseModelSplit ms;
        if (auto e = ms.load(fleet, plan, g, cfg); !e.empty()) { std::fprintf(stderr, "split load: %s\n", e.c_str()); return 1; }
        auto db = ms.device_bytes();
        std::printf("split: layers 0..%u dev0, %u..%u dev1  (weights dev0=%.1fGB dev1=%.1fGB)\n",
                    cfg.n_layers/2 - 1, cfg.n_layers/2, cfg.n_layers - 1,
                    db.size()>0 ? db[0]/1e9 : 0.0, db.size()>1 ? db[1]/1e9 : 0.0);
        if (auto e = ms.forward(ids.data(), T, 0, /*reset_kv=*/true, logits_split.data()); !e.empty())
            { std::fprintf(stderr, "split prefill: %s\n", e.c_str()); return 1; }
        int tk = argmax_half(logits_split);
        toks_split.push_back(tk);
        uint32_t pos = T;
        std::vector<sycl::half> hl(VB);
        for (int s = 1; s < n_gen; ++s) {
            if (auto e = ms.forward(&tk, 1, pos, /*reset_kv=*/false, hl.data()); !e.empty())
                { std::fprintf(stderr, "split decode: %s\n", e.c_str()); return 1; }
            tk = argmax_half(hl); toks_split.push_back(tk); ++pos;
        }
        std::printf("2-GPU split generation done\n");
    }

    // ---- (1) prefill logits BIT-IDENTICAL --------------------------------
    size_t mism = 0;
    for (uint64_t i = 0; i < VB; ++i) {
        uint16_t a, b; std::memcpy(&a, &logits_single[i], 2); std::memcpy(&b, &logits_split[i], 2);
        if (a != b) ++mism;
    }
    bool ok = (mism == 0);
    std::printf("\n(1) prefill logits: %s (%zu/%llu differ)\n",
                mism == 0 ? "BIT-IDENTICAL" : "DIFFER", mism, (unsigned long long)VB);

    // ---- (2) greedy token sequences identical ----------------------------
    int tok_mism = 0;
    for (int i = 0; i < n_gen; ++i) if (toks_single[i] != toks_split[i]) ++tok_mism;
    ok = ok && (tok_mism == 0);
    std::string txt_s = tok.decode(toks_single, true, {});
    std::printf("(2) greedy %d tokens: %s (%d differ)\n", n_gen,
                tok_mism == 0 ? "IDENTICAL" : "DIFFER", tok_mism);
    std::printf("    split-generated text: '%s'\n", tok.decode(toks_split, true, {}).c_str());

    if (ok) { std::printf("\n\033[32mPASS\033[0m: single-GPU == 2-GPU-split (prefill bit-identical + greedy %d/%d)\n", n_gen, n_gen); return 0; }
    std::printf("\n\033[31mFAIL\033[0m\n");
    return 1;
}
