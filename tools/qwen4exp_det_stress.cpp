// tools/qwen4exp_det_stress.cpp — decode determinism stress. Loads once,
// repeats an identical prefill+decode R times (reset_state between), and
// bit-compares logits at every step against run 0. Any mismatch = a race
// (in-order-queue kernels must be run-to-run deterministic).
//
// usage: ie-qwen4exp-det-stress <model.gguf> [-r R] [-n decode_steps] [gpu]

#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/qwen4exp.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace ie;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.gguf> [-r R] [-n N] [gpu]\n", argv[0]); return 2; }
    uint32_t R = 6, NG = 8, ordinal = 0, plen = 8;
    const char* dump = nullptr;
    for (int i = 2; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--dump") && i + 1 < argc) { dump = argv[++i]; continue; }
        if (!std::strcmp(argv[i], "-t") && i + 1 < argc) { plen = uint32_t(std::atoi(argv[++i])); continue; }
        if (!std::strcmp(argv[i], "-r") && i + 1 < argc) R = uint32_t(std::atoi(argv[++i]));
        else if (!std::strcmp(argv[i], "-n") && i + 1 < argc) NG = uint32_t(std::atoi(argv[++i]));
        else ordinal = uint32_t(std::atoi(argv[i]));
    }
    GgufReader g;
    if (auto e = g.open(argv[1]); !e.empty()) { std::fprintf(stderr, "open: %s\n", e.c_str()); return 1; }
    Qwen4ExpConfig cfg;
    if (auto e = read_qwen4exp_config(g, cfg); !e.empty()) { std::fprintf(stderr, "cfg: %s\n", e.c_str()); return 1; }
    DeviceAllocator a;
    if (auto e = a.init("B70", ordinal); !e.empty()) { std::fprintf(stderr, "gpu: %s\n", e.c_str()); return 1; }
    Qwen4ExpModel m;
    if (auto e = m.load(a, g, cfg); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    if (auto e = m.init_runtime(2051, 1024); !e.empty()) { std::fprintf(stderr, "rt: %s\n", e.c_str()); return 1; }

    std::vector<int32_t> prompt = {3000, 415, 6023, 90, 12, 777, 4088, 22};
    if (plen < prompt.size()) prompt.resize(plen);
    std::vector<std::vector<sycl::half>> ref;   // per step logits of run 0
    int bad = 0;
    for (uint32_t run = 0; run < R; ++run) {
        m.reset_state();
        if (auto e = m.forward(prompt.data(), uint32_t(prompt.size()), 0, nullptr);
            !e.empty()) { std::fprintf(stderr, "fwd: %s\n", e.c_str()); return 1; }
        uint32_t pos = uint32_t(prompt.size());
        for (uint32_t st = 0; st <= NG; ++st) {
            std::vector<sycl::half> l(cfg.vocab);
            a.queue().memcpy(l.data(), m.logits(), cfg.vocab * 2).wait();
            if (run == 0) {
                if (dump && st == 0) {   // cross-PROCESS determinism: prefill logits
                    FILE* f = std::fopen(dump, "wb");
                    if (f) { std::fwrite(l.data(), 2, l.size(), f); std::fclose(f); }
                }
                ref.push_back(std::move(l));
            } else {
                uint32_t nd = 0;
                for (uint32_t v = 0; v < cfg.vocab; ++v)
                    if (std::memcmp(&l[v], &ref[st][v], 2) != 0) ++nd;
                if (nd) {
                    std::printf("  run %u step %u: %u logits differ  <-- RACE\n",
                                run, st, nd);
                    ++bad;
                }
            }
            if (st == NG) break;
            // greedy next token from run-0's logits so every run walks the
            // same token path even if a race flips this run's argmax
            float mx = -1e30f; int32_t tok = 0;
            for (uint32_t v = 0; v < cfg.vocab; ++v)
                if (float(ref[st][v]) > mx) { mx = float(ref[st][v]); tok = int32_t(v); }
            if (auto e = m.forward(&tok, 1, pos, nullptr); !e.empty()) {
                std::fprintf(stderr, "fwd: %s\n", e.c_str()); return 1;
            }
            ++pos;
        }
        std::printf("run %u done%s\n", run, bad ? "" : " (clean so far)");
    }
    std::printf(bad ? "DETERMINISM FAILED (%d dirty steps)\n" : "DETERMINISM OK (%u runs)\n",
                bad ? bad : R);
    return bad ? 1 : 0;
}
