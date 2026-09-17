// Phase 1b-i harness: slot-stash suspend/resume losslessness, judged at the
// STATE level.
//
// Text-level byte-comparison cannot prove restore correctness on this stack:
// decode kernels have per-invocation nondeterminism (two plain greedy
// generations differ even within one process — docs/agent-serving-campaign.md),
// so near-tie argmaxes flip run to run with or without a yield. Copies,
// however, are deterministic: the proof is stash → wrong-token clobber →
// unstash → re-stash, then a bitwise device-state compare of the two
// snapshots (IE_TEST_STASH_VERIFY inside the engine's forced-yield block).
// Any mismatch or stash error aborts the generation with
// finish_reason="error: slot-stash: ...", which this tool checks for.
#include "ie/engine.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [n_gpus (default 2)]\n", argv[0]);
        return 2;
    }
    setenv("IE_NO_PROMPT_CACHE", "1", 1);
    unsetenv("IE_TEST_FORCE_YIELD");

    ie::EngineOptions opts;
    opts.max_ctx = 32768;
    opts.n_gpus  = argc > 2 ? uint32_t(std::atoi(argv[2])) : 2;
    std::string err;
    auto eng = ie::Engine::load(argv[1], opts, err);
    if (!eng) { std::fprintf(stderr, "load failed: %s\n", err.c_str()); return 1; }

    std::vector<ie::ChatTurn> turns{
        {"user", "Explain what a mutex is and when to prefer a spinlock, "
                 "in about four sentences."}};
    ie::SamplingParams sp;
    sp.temperature = 0.0f;
    sp.max_tokens  = 96;

    // Smoke: plain generation works on this path.
    ie::GenerateResult plain = eng->chat(turns, sp, {}, /*enable_thinking=*/false);
    if (plain.text.empty()) { std::printf("FAIL: plain generation empty\n"); return 1; }
    std::printf("plain generation: %u tokens OK\n", plain.completion_tokens);

    // Forced yields with state-level verification every 16 decoded tokens.
    setenv("IE_TEST_FORCE_YIELD", "16", 1);
    setenv("IE_TEST_STASH_VERIFY", "1", 1);
    ie::GenerateResult yielded = eng->chat(turns, sp, {}, /*enable_thinking=*/false);
    if (yielded.finish_reason.rfind("error:", 0) == 0) {
        std::printf("FAIL: %s\n", yielded.finish_reason.c_str());
        return 1;
    }
    if (yielded.text.empty()) { std::printf("FAIL: yielded generation empty\n"); return 1; }
    const uint32_t expected_yields = yielded.completion_tokens / 16;
    std::printf("PASS: %u tokens across %u verified stash round-trips "
                "(see [stash-verify] lines)\n",
                yielded.completion_tokens, expected_yields);
    return 0;
}
