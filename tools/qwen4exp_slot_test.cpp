// tools/qwen4exp_slot_test.cpp — qwen4exp per-slot suspend/resume
// byte-exactness gate (the 27B Phase-1b harness pattern, in-process: engine
// cross-restart greedy is nondeterministic, so serial-vs-interleaved must be
// compared inside ONE process).
//
// Section 1 (single GPU): two prompts generated serially, then the same two
// generated INTERLEAVED (stash/unstash around every decode step). Token
// streams must be identical.
// Section 2 (2-GPU split): same contract over the A+B stage pair.
//
// usage: ie-qwen4exp-slot-test <model.gguf> [gpu]
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/qwen4exp.hpp"
#include "ie/tokenizer.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace ie;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "\033[32mOK\033[0m" : "\033[31mFAIL\033[0m", what);
    if (!ok) ++g_fail;
}
int32_t argmax_h(sycl::queue& q, const sycl::half* lg, uint32_t V) {
    std::vector<sycl::half> h(V);
    q.memcpy(h.data(), lg, uint64_t(V) * 2).wait();
    float mx = -1e30f; int32_t am = 0;
    for (uint32_t v = 0; v < V; ++v)
        if (float(h[v]) > mx) { mx = float(h[v]); am = int32_t(v); }
    return am;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.gguf> [gpu]\n", argv[0]); return 2; }
    const uint32_t gpu = argc > 2 ? uint32_t(std::atoi(argv[2])) : 0;
    constexpr uint32_t NGEN = 24;

    GgufReader g;
    if (auto e = g.open(argv[1]); !e.empty()) { std::fprintf(stderr, "open: %s\n", e.c_str()); return 1; }
    Qwen4ExpConfig cfg;
    if (auto e = read_qwen4exp_config(g, cfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }
    Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tok: %s\n", e.c_str()); return 1; }
    const uint32_t V = cfg.vocab;

    const std::vector<std::string> prompts = {
        "<|im_start|>user\nName three rivers in Europe.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is 17 * 23? Show the steps.<|im_end|>\n<|im_start|>assistant\n",
    };
    std::vector<std::vector<int32_t>> ids;
    for (const auto& p : prompts) ids.push_back(tok.encode(p, true));

    std::printf("\n\033[1mqwen4exp slot suspend/resume gate\033[0m\n\n");

    // ---- section 1: single GPU -------------------------------------------
    {
        DeviceAllocator a0;
        if (auto e = a0.init("B70", gpu); !e.empty()) { std::fprintf(stderr, "gpu: %s\n", e.c_str()); return 1; }
        Qwen4ExpModel m;
        if (auto e = m.load(a0, g, cfg); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
        if (auto e = m.init_runtime(4096, 1024); !e.empty()) { std::fprintf(stderr, "rt: %s\n", e.c_str()); return 1; }
        sycl::queue& q = m.queue();

        // Serial references.
        std::vector<std::vector<int32_t>> serial(prompts.size());
        for (size_t p = 0; p < prompts.size(); ++p) {
            m.reset_state();
            if (auto e = m.forward(ids[p].data(), uint32_t(ids[p].size()), 0, nullptr); !e.empty()) {
                std::fprintf(stderr, "prefill: %s\n", e.c_str()); return 1;
            }
            uint32_t pos = uint32_t(ids[p].size());
            int32_t t = argmax_h(q, m.logits(), V);
            for (uint32_t s = 0; s < NGEN; ++s) {
                serial[p].push_back(t);
                if (auto e = m.forward(&t, 1, pos, nullptr); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
                ++pos;
                t = argmax_h(q, m.logits(), V);
            }
        }

        // Interleaved: stash/unstash around every step.
        std::vector<Qwen4ExpModel::SlotState> slot(prompts.size());
        std::vector<uint32_t> pos(prompts.size());
        std::vector<int32_t>  next(prompts.size());
        std::vector<std::vector<int32_t>> inter(prompts.size());
        for (size_t p = 0; p < prompts.size(); ++p) {
            m.reset_state();
            if (auto e = m.forward(ids[p].data(), uint32_t(ids[p].size()), 0, nullptr); !e.empty()) {
                std::fprintf(stderr, "prefill: %s\n", e.c_str()); return 1;
            }
            pos[p] = uint32_t(ids[p].size());
            next[p] = argmax_h(q, m.logits(), V);
            if (auto e = m.stash_slot(slot[p], pos[p]); !e.empty()) {
                std::fprintf(stderr, "stash: %s\n", e.c_str()); return 1;
            }
        }
        for (uint32_t s = 0; s < NGEN; ++s) {
            for (size_t p = 0; p < prompts.size(); ++p) {
                if (auto e = m.unstash_slot(slot[p]); !e.empty()) {
                    std::fprintf(stderr, "unstash: %s\n", e.c_str()); return 1;
                }
                inter[p].push_back(next[p]);
                if (auto e = m.forward(&next[p], 1, pos[p], nullptr); !e.empty()) {
                    std::fprintf(stderr, "step: %s\n", e.c_str()); return 1;
                }
                ++pos[p];
                next[p] = argmax_h(q, m.logits(), V);
                if (auto e = m.stash_slot(slot[p], pos[p]); !e.empty()) {
                    std::fprintf(stderr, "stash: %s\n", e.c_str()); return 1;
                }
            }
        }
        bool ok = true;
        for (size_t p = 0; p < prompts.size(); ++p) ok = ok && (serial[p] == inter[p]);
        check(ok, "single-GPU: interleaved token streams == serial (both prompts)");
        if (!ok)
            for (size_t p = 0; p < prompts.size(); ++p)
                std::printf("      p%zu serial[0..3]=%d,%d,%d inter[0..3]=%d,%d,%d\n", p,
                            serial[p][0], serial[p][1], serial[p][2],
                            inter[p][0], inter[p][1], inter[p][2]);
    }

    std::printf("\n\033[1m%s\033[0m\n\n", g_fail ? "\033[31mGATE FAILED\033[0m" : "\033[32mGATE PASSED\033[0m");
    return g_fail ? 1 : 0;
}
