#undef NDEBUG
#include "ie/serve_options.hpp"
#include <cassert>
#include <cstdio>
#include <stdexcept>
int main() {
    for (auto args : {std::vector<std::string>{"--ctx", "-1"},
            {"--ctx", "1"}, {"--ctx", "8"}, {"--slot-ctx", "8"},
            {"--ctx", "12abc"}, {"--temp", "nan"}, {"--top-p", "0"},
            {"--parallel", "5"}, {"--prefill-chunk", "0"}, {"--threads", "0"},
            {"--slot-ctx", "9000", "--ctx", "8000"}, {"--seed", "-1"},
            {"--stop", ""}, {"--thinking", "maybe"}, {"--unknown"}, {"--temp"},
            {"--reasoning-effort","ultra"}, {"--reasoning-effort"},
            {"--int8-kv", "--parallel", "2"}, {"--max-queue", "-1"},
            {"--max-queue", "2000"}, {"--max-queue"}}) {
        bool rejected = false;
        try { (void)ie::parse_launch_options(args); }
        catch (const std::exception&) { rejected = true; }
        assert(rejected);
    }
    auto minimal = ie::parse_launch_options({"--ctx", "9", "--slot-ctx", "9"});
    assert(minimal.engine.max_ctx == 9 && minimal.engine.slot_ctx == 9);
    assert(minimal.max_queue == 8);
    auto p = ie::parse_launch_options({"--ctx", "200000", "--gpus", "2",
        "--temp", "0", "--top-k", "0", "--top-p", "1", "--min-p", ".1",
        "--repeat-penalty", "1.2", "--repeat-last-n", "256",
        "--presence-penalty", ".5", "--frequency-penalty", "-.2",
        "--max-tokens", "0", "--seed", "18446744073709551615",
        "--stop", "$(touch /tmp/machx_should_not_exist)", "--stop", "line\nend",
        "--threads", "12", "--no-prompt-cache", "--thinking", "off",
        "--reasoning-effort", "high",
        "--prefill-chunk", "32", "--parallel", "2", "--slot-ctx", "8000",
        "--max-queue", "16"});
    assert(p.engine.max_ctx == 200000 && p.engine.n_gpus == 2);
    assert(p.max_queue == 16);
    assert(p.engine.cpu_threads == 12 && !p.engine.prompt_cache);
    assert(p.engine.prefill_chunk == 32 && p.engine.parallel == 2 && p.engine.slot_ctx == 8000);
    const auto& s = p.defaults.sampling;
    assert(s.temperature == 0 && s.top_k == 0 && s.top_p == 1 && s.min_p == .1f);
    assert(s.repeat_window == 256 && s.repeat_penalty == 1.2f);
    assert(s.presence_penalty == .5f && s.frequency_penalty == -.2f);
    assert(s.max_tokens == ie::kMaxTokensUnlimited && s.seed == UINT64_MAX);
    assert(p.defaults.stop.size() == 2 && p.defaults.stop[1] == "line\nend");
    assert(!p.defaults.enable_thinking);
    assert(p.defaults.reasoning_effort=="high");
    std::puts("launch controls validation and propagation passed");
}
