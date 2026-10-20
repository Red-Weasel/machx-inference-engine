// tests/unit/inkling_template_test.cpp — the engine's native Inkling chat template (inkling_render_prompt) against the GGUF's
// own Jinja template rendered by transformers (scripts/gen_inkling_template_golden.py -> tests/data/inkling_template_golden.json),
// byte for byte; and the reply parser (inkling_parse_reply) on hand-written completions. Host-only: no GPU, no weights.
#undef NDEBUG
#include "ie/inkling_engine.hpp"
#include "nlohmann/json.hpp"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using nlohmann::json;

int main() {
    std::ifstream f;
    for (const char* p : {"tests/data/inkling_template_golden.json", "../tests/data/inkling_template_golden.json",
                          "../../tests/data/inkling_template_golden.json"}) {
        f.open(p);
        if (f) break;
        f.clear();
    }
    if (!f) {   // assert() is compiled out in Release: say why instead of parsing an empty stream
        std::fputs("inkling_template_test: tests/data/inkling_template_golden.json not found (run from the source root)\n", stderr);
        return 1;
    }
    const json cases = json::parse(f);
    int fails = 0;
    for (const auto& c : cases) {
        // ChatTurns as the server builds them (openai_proto.cpp): an assistant turn with tool calls keeps its original content
        // in content_without_tool_calls and the calls verbatim in tool_calls_json
        std::vector<ie::ChatTurn> turns;
        for (const auto& m : c["messages"]) {
            ie::ChatTurn t;
            t.role = m["role"].get<std::string>();
            t.content = m.value("content", "");
            if (m.contains("reasoning_content")) t.reasoning_content = m["reasoning_content"].get<std::string>();
            if (m.contains("tool_calls")) { t.tool_calls_json = m["tool_calls"].dump(); t.content_without_tool_calls = t.content; }
            if (m.contains("tool_call_id")) t.tool_call_id = m["tool_call_id"].get<std::string>();
            turns.push_back(std::move(t));
        }
        const std::string tools = c.contains("tools") ? c["tools"].dump() : std::string();
        const std::string effort = c["effort"].is_null() ? std::string() : c["effort"].get<std::string>();
        std::string err;
        const std::string got = ie::inkling_render_prompt(turns, tools, c["thinking"].get<bool>(), effort, err);
        const std::string want = c["prompt"].get<std::string>();
        const bool ok = err.empty() && got == want;
        std::printf("  %-44s %s\n", c["name"].get<std::string>().c_str(), ok ? "OK" : "FAIL");
        if (!ok) {
            ++fails;
            size_t d = 0;
            while (d < got.size() && d < want.size() && got[d] == want[d]) ++d;
            std::printf("    err '%s'; first difference at byte %zu\n    got : %s\n    want: %s\n", err.c_str(), d,
                        got.substr(d > 40 ? d - 40 : 0, 160).c_str(), want.substr(d > 40 ? d - 40 : 0, 160).c_str());
        }
    }
    // the reply parser
    {
        const auto r = ie::inkling_parse_reply("<|content_thinking|>Think.<|end_message|><|message_model|><|content_text|>Paris.<|end_message|>"
                                               "<|content_model_end_sampling|>");
        assert(r.reasoning == "Think." && r.content == "Paris." && r.tool_calls_json == "[]");
        const auto t = ie::inkling_parse_reply("<|content_thinking|>t<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>"
                                               "{\"name\":\"get_weather\",\"args\":{\"city\":\"Paris\",\"days\":2}}<|end_message|>"
                                               "<|content_model_end_sampling|>");
        const json calls = json::parse(t.tool_calls_json);
        assert(calls.size() == 1 && calls[0]["function"]["name"] == "get_weather");
        assert(json::parse(calls[0]["function"]["arguments"].get<std::string>()) == json({{"city", "Paris"}, {"days", 2}}));
        assert(t.content.empty() && t.reasoning == "t");
        const auto bad = ie::inkling_parse_reply("<|content_text|>a<|end_message|><|message_model|>x<|content_invoke_tool_json|>{oops");
        assert(bad.tool_calls_json == "[]" && bad.content == "a{oops");   // an unparseable call stays visible
        std::printf("  reply parser: text/thinking, a tool call, a truncated call -- OK\n");
    }
    std::printf("inkling_template_test: %s (%zu template cases)\n", fails ? "FAILED" : "OK", cases.size());
    return fails ? 1 : 0;
}
