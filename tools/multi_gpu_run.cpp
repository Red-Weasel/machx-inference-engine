// tools/multi_gpu_run.cpp — load a DENSE GGUF split across N GPUs and generate.
// This is the tool for models too big for ONE GPU (e.g. Qwen2.5-72B across 2x B70):
// it only uses the layer-split path (DenseModelSplit), with NO single-GPU baseline,
// so it works when the model doesn't fit a single card. Greedy / fp16 decode.
// Validated bit-identical vs single-GPU on models that fit (ie-multi-gpu-equality-test).
//
// usage: ie-multi-gpu-run --gguf <dense.gguf> [--gpus N=2] [-p "prompt"] [-n 32]
#include "ie/allocator.hpp"
#include "ie/dense_split.hpp"
#include "ie/dense_tp.hpp"
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string gguf, prompt = "The capital of France is";
    uint32_t n_gpus = 2; int n_gen = 32; bool tp = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf" && i + 1 < argc) gguf   = argv[++i];
        else if (a == "--gpus" && i + 1 < argc) n_gpus = uint32_t(std::atoi(argv[++i]));
        else if (a == "-p"     && i + 1 < argc) prompt = argv[++i];
        else if (a == "-n"     && i + 1 < argc) n_gen  = std::atoi(argv[++i]);
        else if (a == "--tp")                   tp     = true;   // tensor-parallel (vs layer-split)
    }
    if (gguf.empty()) { std::fprintf(stderr, "usage: %s --gguf <dense.gguf> [--gpus N] [-p ..] [-n N]\n", argv[0]); return 2; }

    ie::GgufReader g;
    if (auto e = g.open(gguf); !e.empty()) { std::fprintf(stderr, "gguf: %s\n", e.c_str()); return 1; }
    if (!ie::is_dense_arch(ie::detect_arch(g))) { std::fprintf(stderr, "not a dense GGUF (qwen3/qwen2/llama)\n"); return 1; }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tok: %s\n", e.c_str()); return 1; }
    ie::DenseConfig cfg;
    if (auto e = ie::read_dense_config_auto(g, cfg); !e.empty()) { std::fprintf(stderr, "cfg: %s\n", e.c_str()); return 1; }
    std::printf("model: %u layers, hidden %u, vocab %u\n", cfg.n_layers, cfg.hidden, cfg.vocab);

    ie::DeviceFleet fleet;
    if (auto e = fleet.init(n_gpus); !e.empty()) { std::fprintf(stderr, "fleet: %s\n", e.c_str()); return 1; }
    std::printf("fleet: %u GPU(s)\n", fleet.size());

    ie::LayerPlan plan = ie::LayerPlan::contiguous(cfg.n_layers, fleet.size());
    ie::DenseModelSplit ms;
    ie::DenseModelTP    mt;
    std::function<std::string(const int32_t*, uint32_t, uint32_t, bool, sycl::half*)> fwd;
    const double t0 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    std::printf("loading %s across %u device(s)...\n", tp ? "tensor-parallel" : "layer-split", fleet.size());
    std::vector<uint64_t> db;
    if (tp) {
        if (auto e = mt.load(fleet, g, cfg); !e.empty()) { std::fprintf(stderr, "tp load: %s\n", e.c_str()); return 1; }
        db  = mt.device_bytes();
        fwd = [&](const int32_t* in, uint32_t TT, uint32_t sp, bool rk, sycl::half* lo) { return mt.forward(in, TT, sp, rk, lo); };
    } else {
        if (auto e = ms.load(fleet, plan, g, cfg); !e.empty()) { std::fprintf(stderr, "split load: %s\n", e.c_str()); return 1; }
        db  = ms.device_bytes();
        fwd = [&](const int32_t* in, uint32_t TT, uint32_t sp, bool rk, sycl::half* lo) { return ms.forward(in, TT, sp, rk, lo); };
    }
    for (uint32_t d = 0; d < fleet.size(); ++d) std::printf("  dev %u: %.1f GB weights\n", d, db[d] / 1e9);

    const auto ids = tok.encode(prompt, /*allow_special=*/true);
    const uint32_t T = uint32_t(ids.size());
    if (T < 1) { std::fprintf(stderr, "empty prompt\n"); return 1; }

    std::vector<sycl::half> logits(cfg.vocab);
    auto argmax = [&]{ int b = 0; float bv = -1e30f; for (uint32_t i = 0; i < cfg.vocab; ++i){ float f = float(logits[i]); if (f > bv){bv=f;b=int(i);} } return b; };

    // prefill
    if (auto e = fwd(ids.data(), T, 0, /*reset_kv=*/true, logits.data()); !e.empty()) { std::fprintf(stderr, "prefill: %s\n", e.c_str()); return 1; }
    std::vector<int> out; out.push_back(argmax());
    uint32_t pos = T;
    const double t1 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    for (int s = 1; s < n_gen; ++s) {
        int tk = out.back();
        if (tk == tok.eos_token_id()) break;
        if (auto e = fwd(&tk, 1, pos, /*reset_kv=*/false, logits.data()); !e.empty()) { std::fprintf(stderr, "decode: %s\n", e.c_str()); return 1; }
        out.push_back(argmax()); ++pos;
    }
    const double t2 = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();

    std::printf("\nprompt: '%s'\n", prompt.c_str());
    std::printf("output: '%s'\n", tok.decode(out, /*skip_special=*/true, {}).c_str());
    std::printf("[load %.1fs, %d tokens, decode %.1f tok/s]\n", t1 - t0, int(out.size()),
                out.size() > 1 ? (out.size() - 1) / (t2 - t1) : 0.0);
    return 0;
}
