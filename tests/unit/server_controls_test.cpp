#undef NDEBUG
#include "ie/openai_proto.hpp"
#include "nlohmann/json.hpp"
#include <cassert>
#include <cstdio>
using nlohmann::json;
int main() {
    // Reject malformed/overflowing settings before they reach a GPU allocation
    // or are narrowed into unsigned sampling fields.
    json base = {{"messages", {{{"role", "user"}, {"content", "hello"}}}}};
    for (const auto& fields : {
            json{{"temperature", -1}}, json{{"temperature", 2.1}},
            json{{"temperature", "nan"}}, json{{"top_k", -1}},
            json{{"top_k", 1.5}}, json{{"top_k", 1025}},
            json{{"top_p", 0}}, json{{"min_p", 1.1}},
            json{{"repeat_penalty", 0}}, json{{"presence_penalty", 3}},
            json{{"frequency_penalty", -3}}, json{{"repeat_last_n", 513}},
            json{{"max_tokens", 4294967296ULL}}, json{{"max_tokens", 1.2}},
            json{{"seed", -1}}, json{{"seed", 1.2}}, json{{"stream", "yes"}},
            json{{"enable_thinking", 1}}, json{{"reasoning_effort", 1}},
            json{{"reasoning_effort", "ultra"}}, json{{"reasoning_effort", ""}}}) {
        auto req = base; req.update(fields);
        auto parsed = ie::oai::parse_chat_request(req.dump());
        if (parsed.error.empty()) {
            std::fprintf(stderr, "accepted invalid setting: %s\n", fields.dump().c_str());
            return 1;
        }
    }
    auto defaults = ie::oai::server_defaults_from_environment();
    defaults.sampling.temperature = .25f;
    defaults.sampling.top_k = 8;
    defaults.sampling.top_p = .6f;
    defaults.sampling.min_p = .1f;
    defaults.sampling.repeat_penalty = 1.2f;
    defaults.sampling.repeat_window = 32;
    defaults.sampling.presence_penalty = .5f;
    defaults.sampling.frequency_penalty = -.25f;
    defaults.sampling.max_tokens = 79;
    defaults.sampling.seed = 123;
    defaults.enable_thinking = false;
    defaults.reasoning_effort = "max";
    defaults.stop = {"STOP", "line\nend"};
    ie::oai::configure_server_defaults(defaults);
    auto inherited = ie::oai::parse_chat_request(base.dump());
    assert(inherited.error.empty());
    assert(inherited.reasoning_effort=="max");
    const auto& d = inherited.sampling;
    assert(d.temperature == .25f && d.top_k == 8 && d.top_p == .6f && d.min_p == .1f);
    assert(d.repeat_penalty == 1.2f && d.repeat_window == 32);
    assert(d.presence_penalty == .5f && d.frequency_penalty == -.25f);
    assert(d.seed == 123 && d.max_tokens == 79);
    assert(!inherited.enable_thinking && inherited.stop == defaults.stop);
    auto overrides = base;
    overrides.update(json{{"temperature", 0}, {"top_k", 0}, {"top_p", 1}, {"min_p", 0},
        {"repeat_penalty", 1}, {"repeat_last_n", 0}, {"presence_penalty", 0},
        {"frequency_penalty", 0}, {"seed", UINT64_MAX}, {"max_tokens", 0},
        {"enable_thinking", true}, {"stop", json::array()}});
    auto requested = ie::oai::parse_chat_request(overrides.dump());
    assert(requested.error.empty());
    const auto& r = requested.sampling;
    assert(r.temperature == 0 && r.top_k == 0 && r.top_p == 1 && r.min_p == 0);
    assert(r.repeat_penalty == 1 && r.repeat_window == 0 && r.presence_penalty == 0 && r.frequency_penalty == 0);
    assert(r.seed == UINT64_MAX && r.max_tokens == 0 && requested.enable_thinking);
    assert(requested.stop.empty());
    overrides["reasoning_effort"]="low";
    assert(ie::oai::parse_chat_request(overrides.dump()).reasoning_effort=="low");
    overrides["reasoning_effort"]=nullptr;
    assert(ie::oai::parse_chat_request(overrides.dump()).reasoning_effort=="max");
    overrides["stop"] = "literal\nstop";
    overrides["max_tokens"] = nullptr;
    requested = ie::oai::parse_chat_request(overrides.dump());
    assert(requested.error.empty() && requested.sampling.max_tokens == 79);
    assert(requested.stop == std::vector<std::string>{"literal\nstop"});
    setenv("IE_SERVE_TEMP", "nan", 1);
    bool invalid_env = false;
    try { (void)ie::oai::server_defaults_from_environment(); }
    catch (const std::exception&) { invalid_env = true; }
    assert(invalid_env);
    unsetenv("IE_SERVE_TEMP");
    std::puts("server controls validation, defaults, and overrides passed");
}
