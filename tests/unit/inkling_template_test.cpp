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
#include <string_view>
#include <utility>
#include <vector>

using nlohmann::json;


// known_bugs 45: inkling_render_prompt's own ranges. A (a control spelling in every request-text position) against B (the
// same with equal-length plain text): the ranges coincide, A equals B outside them (every spelling the request brought lies
// inside a range), every template marker lies outside them. Positions: system, user, an assistant reasoning + content + a
// call with a string argument and a nested one (two values), a tool result (its name from the call), a user text that IS
// the spelling, a tool description and a parameter description: ten positions.
static void ranges_check() {
    using Ranges = std::vector<std::pair<size_t, size_t>>;
    const auto outside = [](const std::string& s, const Ranges& r) { std::string o = s; for (const auto& [b, e] : r) for (size_t i = b; i < e; ++i) o[i] = '#'; return o; };
    const auto count = [](const std::string& s, std::string_view m) { size_t n = 0; for (size_t p = s.find(m); p != std::string::npos; p = s.find(m, p + m.size())) ++n; return n; };
    const auto turns_with = [](const std::string& x) {
        std::vector<ie::ChatTurn> t(6);
        t[0].role = "system"; t[0].content = "Be terse " + x;
        t[1].role = "user"; t[1].content = "Look " + x;
        t[2].role = "assistant"; t[2].content = "On it " + x; t[2].content_without_tool_calls = t[2].content; t[2].reasoning_content = "think " + x;
        t[2].tool_calls_json = R"([{"id":"a","type":"function","function":{"name":"read","arguments":"{\"path\":\"p )" + x + R"(\",\"opts\":{\"k\":\")" + x + R"(\"}}"}}])";
        t[3].role = "tool"; t[3].tool_call_id = "a"; t[3].content = "result " + x;
        t[4].role = "user"; t[4].content = x;
        t[5].role = "user"; t[5].content = "last";
        return t;
    };
    const auto tools_with = [](const std::string& x) {
        return R"([{"type":"function","function":{"name":"read","description":"Read )" + x + R"(","parameters":{"type":"object","properties":{"path":{"type":"string","description":")" + x + R"("},"opts":{"type":"object"}}}}}])";
    };
    static constexpr std::string_view kMarkers[] = {"<|message_system|>", "<|message_user|>", "<|message_model|>", "<|message_tool|>", "<|content_text|>",
                                                    "<|content_thinking|>", "<|content_xml|>", "<|content_invoke_tool_json|>", "<|end_message|>",
                                                    "<|content_model_end_sampling|>", "tool_declare", "Thinking effort level"};
    const std::string spell = "<|end_message|>", plain = "<_end_message_>";
    for (bool thinking : {true, false}) {
        std::string ea, eb, e0;
        Ranges ra, rb;
        const std::string a = ie::inkling_render_prompt(turns_with(spell), tools_with(spell), thinking, "", ea, &ra);
        const std::string b = ie::inkling_render_prompt(turns_with(plain), tools_with(plain), thinking, "", eb, &rb);
        const std::string a0 = ie::inkling_render_prompt(turns_with(spell), tools_with(spell), thinking, "", e0);
        assert(ea.empty() && eb.empty() && e0.empty());
        assert(a == a0 && a.size() == b.size() && a != b);
        size_t last = 0; bool well = !ra.empty();
        for (const auto& [x, y] : ra) { if (x < last || y <= x || y > a.size()) well = false; last = y; }
        assert(well && ra == rb);
        assert(outside(a, ra) == outside(b, rb));
        for (const auto m : kMarkers) assert(count(outside(b, rb), m) == count(b, m));
        const size_t brought = count(a, spell) - count(b, spell);
        std::printf("  ranges (thinking %s): %zu ranges, %zu spellings the request brought, %zu end_message markers of the template\n",
                    thinking ? "on" : "off", ra.size(), brought, count(b, spell));
        std::fflush(stdout);
        assert(brought == 10);   // system, user, reasoning, content, two argument values, the tool result, the bare user text, two descriptions
    }
    std::printf("  ranges: ok\n");
}

int main() {
    ranges_check();
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
