// tools/qwen3next_fwd_test.cpp — Qwen3-Next-80B Step-2.1 (F1) FORWARD scaffold gate.
// Loads the 46 GB GGUF split across 2 B70s (same as the load test) then runs ONE
// forward pass over a fixed tiny token array and prints the argmax logit id. The
// per-layer body is STUBBED in F1 (attn + MoE zeroed → x passes through), so the
// top1 id is GARBAGE — this only proves the split pipeline runs end-to-end without
// crashing. RUN ONLY with both cards free + serialized (see loader GPU-safety rules).
#include "ie/qwen3next.hpp"
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <qwen3next.gguf> [n_gpus=2]\n", argv[0]); return 2; }
    const uint32_t n_gpus = (argc >= 3) ? uint32_t(std::stoul(argv[2])) : 2u;

    ie::GgufReader g;
    std::string err = g.open(argv[1]);
    if (!err.empty()) { std::fprintf(stderr, "gguf open: %s\n", err.c_str()); return 1; }

    ie::Qwen3NextConfig cfg;
    err = ie::read_qwen3next_config(g, cfg);
    if (!err.empty()) { std::fprintf(stderr, "config: %s\n", err.c_str()); return 1; }

    ie::DeviceFleet fleet;
    err = fleet.init(n_gpus, "B70");
    if (!err.empty()) { std::fprintf(stderr, "fleet: %s\n", err.c_str()); return 1; }

    const uint32_t n_layers = cfg.hybrid.n_transformer_layers();
    ie::LayerPlan plan = ie::LayerPlan::contiguous(n_layers, n_gpus);

    ie::Qwen3NextModel m;
    err = m.load(fleet, plan, g, cfg);
    if (!err.empty()) { std::fprintf(stderr, "LOAD FAILED: %s\n", err.c_str()); return 1; }

    // Fixed tiny token array (F1 just proves the pipeline runs; ids are arbitrary).
    const std::vector<int32_t> ids = {1, 2, 3, 4, 5};
    const uint32_t T = uint32_t(ids.size());
    const uint32_t V = cfg.hybrid.dense.vocab;

    std::vector<sycl::half> logits(V);
    err = m.forward(ids.data(), T, /*start_pos=*/0, /*reset_kv=*/true, logits.data());
    if (!err.empty()) { std::fprintf(stderr, "FWD FAILED: %s\n", err.c_str()); return 1; }

    uint32_t top1 = 0;
    float best = -1e30f;
    for (uint32_t i = 0; i < V; ++i) {
        const float v = float(logits[i]);
        if (v > best) { best = v; top1 = i; }
    }
    std::printf("FWD OK: top1=%u (garbage expected — layers stubbed in F1)\n", top1);
    return 0;
}
