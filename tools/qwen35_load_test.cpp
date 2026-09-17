// tools/qwen35_load_test.cpp — P3d Task 3A load gate.
//
// Loads a `qwen35` GGUF through the Engine, exercising Qwen35DenseModel::load
// (every tensor for both layer kinds, the Q5_K/Q8_0 dequant-to-fp16 path),
// the hybrid cache init (KV for full-attn + DeltaNet state for linear), and
// the workspace allocation. Does NOT generate — the hybrid forward lands in
// Task 3B/3C. Success = the model ingests + allocates without error.
#include "ie/engine.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <qwen35.gguf>\n", argv[0]);
        return 2;
    }
    ie::EngineOptions opts;
    opts.max_ctx = 2048;   // small ctx → modest KV alloc; this is a load gate
    std::string err;
    auto eng = ie::Engine::load(argv[1], opts, err);
    if (!eng) {
        std::fprintf(stderr, "LOAD FAILED: %s\n", err.c_str());
        return 1;
    }
    std::printf("LOAD OK  arch=%d  vocab=%u  max_ctx=%u\n",
                int(eng->arch()), eng->vocab(), eng->max_ctx());

    // Optional forward smoke: generate a few tokens. With the DeltaNet layers
    // stubbed (Task 3B) the text is NOT correct — this only confirms the
    // forward pipeline runs on-device without kernel-shape crashes.
    if (argc >= 3) {
        ie::SamplingParams sp;
        sp.max_tokens = 48;
        sp.temperature = 0.0f;   // greedy
        std::printf("--- forward smoke (prompt=%.40s) ---\n[", argv[2]);
        auto r = eng->generate(argv[2], sp, [](std::string_view t) {
            std::fwrite(t.data(), 1, t.size(), stdout); return true;
        });
        std::printf("]\nforward ran: %u prompt + %u gen tokens (finish=%s)\n",
                    r.prompt_tokens, r.completion_tokens, r.finish_reason.c_str());

        // Chat-path diagnostic: greedy, thinking ON then OFF — isolates whether
        // the ie-run garbage is temp-0.7 sampling or the <think> template.
        ie::ChatTurn turn{"user", argv[2]};
        auto cb = [](std::string_view t){ std::fwrite(t.data(), 1, t.size(), stdout); return true; };
        // temp 0.7 + FIXED seed → reproducible sampler-bug investigation.
        // Sweep a few seeds (argv[3] overrides base seed) to find a failing draw.
        const uint64_t base_seed = (argc >= 4) ? std::strtoull(argv[3], nullptr, 10) : 1;

        // Sweep mode (argv[4] = seed count): thinking-OFF (the `ie run` config),
        // temp 0.7, many seeds — the chat-quality regression harness that RETIRED
        // the alleged "27B chat garbage" (see MASTER_DEV_PLAN §7: 114 gens clean;
        // the flagged 248068/248069 are <think>/</think>, legitimate markers, not
        // garbage). Prints every reply for visual scan; FLAGS a short reply that
        // STOPS early (the only plausible garbage signature).
        if (argc >= 5) {
            const uint64_t n_seeds = std::strtoull(argv[4], nullptr, 10);
            std::printf("\n=== thinking-OFF temp0.7 sweep: %llu seeds from %llu ===\n",
                        (unsigned long long)n_seeds, (unsigned long long)base_seed);
            uint32_t flagged = 0;
            for (uint64_t s = base_seed; s < base_seed + n_seeds; ++s) {
                ie::SamplingParams csp;
                csp.temperature = 0.7f; csp.max_tokens = 32; csp.seed = s;
                std::string out;
                auto r = eng->chat(std::span<const ie::ChatTurn>(&turn, 1), csp,
                    [&](std::string_view t){ out.append(t); return true; },
                    /*enable_thinking=*/false);
                // visible (non-whitespace) char count
                size_t vis = 0; for (char c : out) if (!std::isspace((unsigned char)c)) ++vis;
                // Garbage signature: very short reply that then STOPS (covers both
                // invisible special-token-then-stop AND tiny visible junk 'sd'/'etter').
                const bool suspect = (r.completion_tokens <= 4 && r.finish_reason == "stop");
                // Print EVERY generation (truncated) for visual garbage scanning.
                std::printf("[seed=%llu toks=%u fin=%s] %.60s\n",
                            (unsigned long long)s, r.completion_tokens,
                            r.finish_reason.c_str(), out.c_str());
                if (suspect) { ++flagged; std::printf("   ^^ SUSPECT (short+stop)\n"); }
            }
            std::printf("=== sweep done: %u/%llu flagged ===\n",
                        flagged, (unsigned long long)n_seeds);
        } else {
            for (uint64_t s = base_seed; s < base_seed + 6; ++s) {
                ie::SamplingParams csp;
                csp.temperature = 0.7f; csp.max_tokens = 24; csp.seed = s;
                std::printf("\n--- chat temp0.7 seed=%llu thinking ON ---\n[", (unsigned long long)s);
                eng->chat(std::span<const ie::ChatTurn>(&turn, 1), csp, cb, true);
                std::printf("]\n");
            }
        }
    }
    return 0;
}
