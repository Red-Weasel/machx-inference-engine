// tools/gptoss_load_test.cpp — gpt-oss STEP 1 load gate.
//
// Loads a `gpt-oss` GGUF through the Engine, exercising GptOssModel::load:
// every per-layer tensor (Q8_0 attn q/k/v/o + F32 biases, F32 attn_sinks,
// post_attention_norm, F32 router + bias, MXFP4 gate/up/down experts + F32
// per-expert biases), plus the KV/workspace allocation. Does NOT generate —
// the forward lands in STEP 2. Success = the model ingests + allocates with
// no error (every tensor found with the expected dtype/shape).
#include "ie/engine.hpp"

#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <gpt-oss.gguf>\n", argv[0]);
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

    // Harmony chat smoke: render the Harmony prompt + greedy chat. A correct path
    // answers the user turn and stops on <|return|>; res.text is the FINAL channel.
    if (argc >= 3) {
        ie::SamplingParams sp;
        sp.max_tokens  = (argc >= 4) ? uint32_t(std::atoi(argv[3])) : 200;
        sp.temperature = 0.0f;   // greedy
        ie::ChatTurn turn{"user", argv[2]};
        std::printf("--- Harmony chat (user=%s) ---\n[", argv[2]);
        auto r = eng->chat(std::span<const ie::ChatTurn>(&turn, 1), sp,
            [](std::string_view t){ std::fwrite(t.data(), 1, t.size(), stdout); return true; },
            /*enable_thinking=*/false);
        std::printf("]\n--- res.text (final channel) ---\n%s\n", r.text.c_str());
        std::printf("ran: %u prompt + %u gen tokens (finish=%s)\n",
                    r.prompt_tokens, r.completion_tokens, r.finish_reason.c_str());
    }
    return 0;
}
