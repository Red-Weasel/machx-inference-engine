// tools/qwen3next_load_test.cpp — Qwen3-Next-80B Step-1 LOAD gate.
// Loads the 46 GB GGUF split across 2 B70s via Qwen3NextModel and reports the
// per-card weight totals. No forward (Step 2). RUN ONLY with both cards free
// and serialized — see the loader's GPU-safety rules.
#include "ie/qwen3next.hpp"
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <qwen3next.gguf> [n_gpus=2]\n", argv[0]); return 2; }
    const uint32_t n_gpus = (argc >= 3) ? uint32_t(std::stoul(argv[2])) : 2u;

    ie::GgufReader g;
    std::string err = g.open(argv[1]);          // GgufReader::open(const std::string&) → "" or error
    if (!err.empty()) { std::fprintf(stderr, "gguf open: %s\n", err.c_str()); return 1; }

    ie::Qwen3NextConfig cfg;
    err = ie::read_qwen3next_config(g, cfg);
    if (!err.empty()) { std::fprintf(stderr, "config: %s\n", err.c_str()); return 1; }

    ie::DeviceFleet fleet;
    err = fleet.init(n_gpus, "B70");          // DeviceFleet::init(uint32_t n_request, name_filter) → "" or error
    if (!err.empty()) { std::fprintf(stderr, "fleet: %s\n", err.c_str()); return 1; }

    const uint32_t n_layers = cfg.hybrid.n_transformer_layers();
    ie::LayerPlan plan = ie::LayerPlan::contiguous(n_layers, n_gpus);

    ie::Qwen3NextModel m;
    err = m.load(fleet, plan, g, cfg);
    if (!err.empty()) { std::fprintf(stderr, "LOAD FAILED: %s\n", err.c_str()); return 1; }

    std::printf("LOAD OK across %ux B70  layers=%u  experts=%u/%u  E_ffn=%u\n",
                n_gpus, n_layers, cfg.n_experts_used, cfg.n_experts, cfg.expert_ffn);
    auto bytes = m.device_bytes();
    for (uint32_t dd = 0; dd < bytes.size(); ++dd)
        std::printf("  card %u: %.2f GB\n", dd, double(bytes[dd]) / (1u<<30));
    return 0;
}
