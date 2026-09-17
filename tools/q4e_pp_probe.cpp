// tools/q4e_pp_probe.cpp — single-GPU prefill chunk-size / machine-state
// re-baseline probe (2026-08-27). Compile standalone like tools/p2p_probe.cpp:
//   icpx -O3 -std=c++20 -fsycl -fsycl-targets=spir64 -I include <this> \
//     build/src/libie_core.a $ONEAPI/dnnl/.../libdnnl.so.3.11 -lpthread \
//     $ONEAPI/tbb/.../libtbb.so.12 -ldl
// Scratch probe: single-GPU prefill throughput vs chunk size (H2D
// amortization hypothesis: union ~512 experts/layer per chunk regardless of
// chunk size, so doubling the chunk halves expert-upload bytes per token).
// usage: q4e_pp_probe <model.gguf> [gpu]
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/qwen4exp.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace ie;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.gguf> [gpu]\n", argv[0]); return 2; }
    const uint32_t gpu = argc > 2 ? uint32_t(std::atoi(argv[2])) : 0;
    GgufReader g;
    if (auto e = g.open(argv[1]); !e.empty()) { std::fprintf(stderr, "open: %s\n", e.c_str()); return 1; }
    Qwen4ExpConfig cfg;
    if (auto e = read_qwen4exp_config(g, cfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }
    DeviceAllocator alloc;
    if (auto e = alloc.init("B70", gpu); !e.empty()) { std::fprintf(stderr, "gpu: %s\n", e.c_str()); return 1; }
    Qwen4ExpModel m;
    if (auto e = m.load(alloc, g, cfg); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    if (auto e = m.init_runtime(8192, 4096); !e.empty()) { std::fprintf(stderr, "rt: %s\n", e.c_str()); return 1; }

    const uint32_t TOTAL = 4096;
    std::vector<int32_t> toks(TOTAL);
    for (uint32_t i = 0; i < TOTAL; ++i) toks[i] = 2000 + (i * 97) % 200000;

    auto run_cfg = [&](uint32_t chunk) {
        m.reset_state();
        // warm (JIT already warm after first config; keep symmetric anyway)
        std::vector<int32_t> w8(8);
        for (int i = 0; i < 8; ++i) w8[i] = 3000 + i * 37;
        if (auto e = m.forward(w8.data(), 8, 0, nullptr); !e.empty()) {
            std::fprintf(stderr, "warm: %s\n", e.c_str()); std::exit(1);
        }
        double total_ms = 0;
        for (uint32_t pos = 0; pos < TOTAL; pos += chunk) {
            const auto t0 = std::chrono::steady_clock::now();
            if (auto e = m.forward(toks.data() + pos, chunk, 8 + pos, nullptr); !e.empty()) {
                std::fprintf(stderr, "fwd: %s\n", e.c_str()); std::exit(1);
            }
            const double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0).count();
            total_ms += ms;
            std::printf("  chunk=%u @pos %u: %.0f ms (%.1f tok/s)\n",
                        chunk, 8 + pos, ms, chunk * 1000.0 / ms);
        }
        std::printf("chunk=%u TOTAL: %.0f ms  (%.1f tok/s over %u toks)  "
                    "ecache hits %llu misses %llu\n",
                    chunk, total_ms, TOTAL * 1000.0 / total_ms, TOTAL,
                    (unsigned long long)m.ecache_hits,
                    (unsigned long long)m.ecache_misses);
        m.ecache_hits = m.ecache_misses = 0;
    };
    run_cfg(1024);   // first pass also absorbs JIT warmup
    run_cfg(4096);
    run_cfg(4096);
    return 0;
}
