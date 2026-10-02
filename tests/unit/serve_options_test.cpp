#undef NDEBUG
#include "ie/serve_options.hpp"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
int main() {
    for (auto args : {std::vector<std::string>{"--ctx", "-1"},
            {"--ctx", "1"}, {"--ctx", "8"}, {"--slot-ctx", "8"},
            {"--ctx", "12abc"}, {"--temp", "nan"}, {"--top-p", "0"},
            {"--parallel", "17"}, {"--parallel", "0"}, {"--prefill-chunk", "0"}, {"--threads", "0"},
            {"--slot-ctx", "9000", "--ctx", "8000"}, {"--seed", "-1"},
            {"--stop", ""}, {"--thinking", "maybe"}, {"--unknown"}, {"--temp"},
            {"--reasoning-effort","ultra"}, {"--reasoning-effort"},
            {"--int8-kv", "--parallel", "2"}, {"--max-queue", "-1"},
            {"--max-queue", "2000"}, {"--max-queue"},
            {"--parallel"}, {"--parallel", "AUTO"}, {"--parallel", "auto4"}, {"--parallel", ""}}) {
        bool rejected = false;
        try { (void)ie::parse_launch_options(args); }
        catch (const std::exception&) { rejected = true; }
        assert(rejected);
    }
    // P4 B14: the cap is 16 (ie::kMaxParallel); the arch refuses a count that does not fit at load, not the parser
    for (const char* n : {"1", "4", "5", "8", "16"})
        assert(ie::parse_launch_options({"--parallel", n}).engine.parallel == uint32_t(std::atoi(n)));
    static_assert(ie::kMaxParallel == 16);
    auto minimal = ie::parse_launch_options({"--ctx", "9", "--slot-ctx", "9"});
    assert(minimal.engine.max_ctx == 9 && minimal.engine.slot_ctx == 9);
    assert(minimal.max_queue == 8 && !minimal.max_queue_set);
    // P4 B30: no --parallel = auto (the load picks N, ie/lanes_auto.hpp); --parallel auto = the same; the last flag wins;
    // an explicit N is passed through as given (the loop above); --int8-kv makes auto one lane (the lanes refuse it)
    assert(minimal.engine.parallel == ie::kLanesAuto && ie::parse_launch_options({}).engine.parallel == ie::kLanesAuto);
    assert(ie::parse_launch_options({"--parallel", "auto"}).engine.parallel == ie::kLanesAuto);
    assert(ie::parse_launch_options({"--parallel", "4", "--parallel", "auto"}).engine.parallel == ie::kLanesAuto);
    assert(ie::parse_launch_options({"--parallel", "auto", "--parallel", "4"}).engine.parallel == 4);
    assert(ie::parse_launch_options({"--parallel", "auto", "--ctx", "4096"}).engine.max_ctx == 4096);   // "auto" took its value
    assert(ie::parse_launch_options({"--int8-kv"}).engine.parallel == 1);
    assert(ie::parse_launch_options({"--parallel", "auto", "--int8-kv"}).engine.parallel == 1);
    assert(ie::EngineOptions{}.parallel == 1);   // tools that build EngineOptions themselves keep one lane
    // --max-queue: the flag when given; 8 beside an explicit --parallel (as before); N + 8 when the load picked N
    assert(ie::serve_max_queue(minimal, 16) == 24 && ie::serve_max_queue(minimal, 4) == 12 && ie::serve_max_queue(minimal, 1) == 9);
    for (uint32_t n : {1u, 4u, 16u}) {
        assert(ie::serve_max_queue(ie::parse_launch_options({"--parallel", "4"}), n) == 8);
        assert(ie::serve_max_queue(ie::parse_launch_options({"--parallel", "1"}), n) == 8);
        assert(ie::serve_max_queue(ie::parse_launch_options({"--max-queue", "3"}), n) == 3);
        assert(ie::serve_max_queue(ie::parse_launch_options({"--max-queue", "0", "--parallel", "auto"}), n) == 0);
        assert(ie::serve_max_queue(ie::parse_launch_options({"--parallel", "2", "--max-queue", "40"}), n) == 40);
    }
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
    // P4 B20: every CLI sampling flag marks its field explicit, so it outranks the recommended sampling
    assert(p.defaults.sampling_set == ie::oai::kSetAllRecommended);
    assert(minimal.defaults.sampling_set == 0);
    auto one = ie::parse_launch_options({"--top-p", "0.5"});
    assert(one.defaults.sampling_set == ie::oai::kSetTopP);
    ie::oai::ChatRequest r = one.defaults;
    const auto rec = ie::recommended_sampling(ie::ModelArch::kMimo26);
    assert(ie::oai::apply_recommended(r, &*rec) == (ie::oai::kSetAllRecommended & ~ie::oai::kSetTopP));
    assert(r.sampling.top_p == 0.5f && r.sampling.temperature == 1.0f && r.sampling.top_k == 0);
    std::puts("launch controls validation and propagation passed");
}
