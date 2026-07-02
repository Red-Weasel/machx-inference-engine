// Requires the daily-driver GGUF + B70. Not registered with add_test
// (model-dependent); run manually: ./build/tests/engine_generate_test
#undef NDEBUG
#include "ie/engine.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
int main() {
    std::string err;
    ie::EngineOptions opts; opts.max_ctx = 2048;
    const char* mp = std::getenv("IE_MODEL");
    auto eng = ie::Engine::load(mp ? mp :
        "/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf", opts, err);
    if (!eng) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }

    std::vector<ie::ChatTurn> turns{
        {"system", "You are a concise assistant."},
        {"user",   "Reply with exactly the word: ping"}};
    ie::SamplingParams sp; sp.temperature = 0.f; sp.max_tokens = 64;
    std::string streamed;
    auto r = eng->chat(turns, sp,
        [&](std::string_view t) { streamed.append(t); return true; },
        /*enable_thinking=*/false);

    if (r.completion_tokens == 0)        { std::puts("FAIL: no tokens"); return 1; }
    if (r.finish_reason != "stop" && r.finish_reason != "length")
                                         { std::puts("FAIL: bad finish"); return 1; }
    if (streamed != r.text)              { std::puts("FAIL: stream != text"); return 1; }
    if (r.text.find("ping") == std::string::npos)
        std::puts("WARN: completion lacks 'ping' (model behavior, not API bug)");
    // max_tokens honored
    ie::SamplingParams sp2; sp2.temperature = 0.f; sp2.max_tokens = 3;
    auto r2 = eng->chat(turns, sp2, {}, false);
    if (r2.completion_tokens > 3)        { std::puts("FAIL: max_tokens"); return 1; }
    // abort path: callback returns false after first fragment
    int frags = 0;
    auto r3 = eng->chat(turns, sp, [&](std::string_view) { return ++frags < 2; }, false);
    if (r3.finish_reason != "abort" && r3.finish_reason != "stop")
                                         { std::puts("FAIL: abort path"); return 1; }
    std::printf("completion: %s\n", r.text.c_str());
    std::puts("engine_generate_test: all OK");
    return 0;
}
