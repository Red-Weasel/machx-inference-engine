// tools/q4e_mini_logits.cpp -- the engine half of the mini qwen4exp end-to-end oracle (scripts/qwen4exp_oracle/
// mini_model.py, 2026-11-08). Loads a mini GGUF on one card, runs the token ids three ways and writes their logits as
// float32 rows [T + 2][vocab]:
//   rows 0..T-1  token-by-token decode (forward T = 1 at each position): every position's logits
//   row  T       one prefill of all T tokens (the last position)
//   row  T+1     the same prefill in two chunks (T/2, then the rest) after reset_state (the last position)
// mini_model.py compare scores them against the HF reference's fp32 logits.
//
// usage: ie-q4e-mini-logits <mini.gguf> <ids.i32> <out.f32> [gpu]
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/ops.hpp"
#include "ie/qwen4exp.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace ie;

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s <mini.gguf> <ids.i32> <out.f32> [gpu]\n", argv[0]); return 2; }
    const uint32_t ordinal = argc > 4 ? uint32_t(std::atoi(argv[4])) : 0;
    std::vector<int32_t> ids;
    {
        FILE* f = std::fopen(argv[2], "rb");
        if (!f) { std::fprintf(stderr, "ids: cannot open %s\n", argv[2]); return 2; }
        int32_t v;
        while (std::fread(&v, 4, 1, f) == 1) ids.push_back(v);
        std::fclose(f);
    }
    const uint32_t T = uint32_t(ids.size());
    if (T < 2) { std::fprintf(stderr, "ids: need >= 2 tokens\n"); return 2; }

    GgufReader g;
    if (auto e = g.open(argv[1]); !e.empty()) { std::fprintf(stderr, "open: %s\n", e.c_str()); return 1; }
    Qwen4ExpConfig cfg;
    if (auto e = read_qwen4exp_config(g, cfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }
    DeviceAllocator alloc;
    if (auto e = alloc.init("B70", ordinal); !e.empty()) { std::fprintf(stderr, "gpu: %s\n", e.c_str()); return 1; }
    Qwen4ExpModel m;
    if (auto e = m.load(alloc, g, cfg); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    if (auto e = m.init_runtime(/*max_ctx=*/256, /*max_chunk=*/64); !e.empty()) {
        std::fprintf(stderr, "init_runtime: %s\n", e.c_str()); return 1;
    }
    std::printf("mini: %u blocks, hidden %u, kv heads %u, %s, %s, PLE blocks %zu; %u tokens; device %.2f GiB\n",
                cfg.n_layers, cfg.hidden, cfg.n_kv_heads,
                cfg.n_experts ? "MoE" : "dense FFN", cfg.indexer_top_k ? "QSA" : "no QSA (dense attention)",
                cfg.ple_layers.size(), T, m.device_bytes() / 1073741824.0);

    const uint32_t V = cfg.vocab;
    std::vector<float> out(size_t(T + 2) * V);
    std::vector<sycl::half> l16(V);
    uint32_t nonfinite = 0;
    auto grab = [&](uint32_t row) {
        alloc.queue().memcpy(l16.data(), m.logits(), size_t(V) * 2).wait();
        for (uint32_t i = 0; i < V; ++i) {
            out[size_t(row) * V + i] = float(l16[i]);
            if (!std::isfinite(out[size_t(row) * V + i])) ++nonfinite;
        }
    };
    for (uint32_t t = 0; t < T; ++t) {
        if (auto e = m.forward(&ids[t], 1, t, nullptr); !e.empty()) { std::fprintf(stderr, "decode %u: %s\n", t, e.c_str()); return 1; }
        grab(t);
    }
    m.reset_state();
    if (auto e = m.forward(ids.data(), T, 0, nullptr); !e.empty()) { std::fprintf(stderr, "prefill: %s\n", e.c_str()); return 1; }
    grab(T);
    m.reset_state();
    const uint32_t h = T / 2;
    if (auto e = m.forward(ids.data(), h, 0, nullptr); !e.empty()) { std::fprintf(stderr, "prefill a: %s\n", e.c_str()); return 1; }
    if (auto e = m.forward(ids.data() + h, T - h, h, nullptr); !e.empty()) { std::fprintf(stderr, "prefill b: %s\n", e.c_str()); return 1; }
    grab(T + 1);

    FILE* f = std::fopen(argv[3], "wb");
    if (!f || std::fwrite(out.data(), 4, out.size(), f) != out.size()) { std::fprintf(stderr, "write %s failed\n", argv[3]); return 1; }
    std::fclose(f);
    std::printf("wrote %u rows x %u logits to %s; non-finite %u\n", T + 2, V, argv[3], nonfinite);
    return nonfinite ? 1 : 0;
}
