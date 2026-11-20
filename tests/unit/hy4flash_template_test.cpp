// tests/unit/hy4flash_template_test.cpp — Hy4-flash: the engine's native render of the checkpoint's chat template against
// the template's own output (golden cases from scripts/hy4flash_oracle/template_cases.py, rendered with Jinja), and the
// reply translation the stream and the parser use. Host only.
#include "ie/glm5_server.hpp"
#include "ie/hy4flash_engine.hpp"
#include "nlohmann/json.hpp"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using nlohmann::ordered_json;

int main(int argc, char** argv) {
    const std::string path = argc > 1 ? argv[1] : "tests/unit/data/hy4flash_template_cases.json";
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); return 2; }
    const ordered_json cases = ordered_json::parse(f);
    int bad = 0;
    for (const auto& c : cases) {
        std::vector<ie::ChatTurn> turns;
        for (const auto& m : c["messages"]) {
            ie::ChatTurn t; t.role = m["role"].get<std::string>(); t.content = m.value("content", std::string());
            if (m.contains("reasoning_content")) t.reasoning_content = m["reasoning_content"].get<std::string>();
            if (m.contains("tool_call_id")) t.tool_call_id = m["tool_call_id"].get<std::string>();
            if (m.contains("tool_calls")) {
                t.tool_calls_json = m["tool_calls"].dump();
                for (const auto& tc : m["tool_calls"]) t.tool_call_ids.push_back(tc.value("id", std::string()));
            }
            turns.push_back(std::move(t));
        }
        const std::string tools = c["tools"].empty() ? std::string() : c["tools"].dump();
        std::string err;
        const std::string got = ie::hy4f_render_prompt(turns, tools, c["reasoning_effort"].get<std::string>(), err);
        const std::string want = c["expected"].get<std::string>();
        if (!err.empty() || got != want) {
            ++bad;
            size_t i = 0; while (i < got.size() && i < want.size() && got[i] == want[i]) ++i;
            std::fprintf(stderr, "FAIL %s%s%s\n  first difference at byte %zu\n  want: ...%s\n  got:  ...%s\n", c["name"].get<std::string>().c_str(), err.empty() ? "" : ": ", err.c_str(), i,
                         want.substr(i > 40 ? i - 40 : 0, 160).c_str(), got.substr(i > 40 ? i - 40 : 0, 160).c_str());
        }
    }
    // a reply in the model's tags -> GLM's spelling -> the engine's parse
    {
        const std::string raw = "The user wants two things.</think:opensource>On it.<tool_calls:opensource><tool_call:opensource>get_weather<arg_key:opensource>city</arg_key:opensource>"
                                "<arg_value:opensource>Oslo</arg_value:opensource><arg_key:opensource>days</arg_key:opensource><arg_value:opensource>2</arg_value:opensource></tool_call:opensource></tool_calls:opensource>";
        const std::string tr = ie::hy4f_translate_reply(raw);
        const std::string want = "The user wants two things.</think>On it.<tool_call>get_weather<arg_key>city</arg_key><arg_value>Oslo</arg_value><arg_key>days</arg_key><arg_value>2</arg_value></tool_call>";
        if (tr != want) { ++bad; std::fprintf(stderr, "FAIL reply translation\n  got: %s\n", tr.c_str()); }
        const auto pc = ie::parse_glm5_completion(tr, true, R"([{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}}])");
        const auto calls = ordered_json::parse(pc.tool_calls.empty() ? "[]" : pc.tool_calls, nullptr, false);
        const bool ok = pc.reasoning == "The user wants two things." && pc.content == "On it." && calls.is_array() && calls.size() == 1 &&
                        calls[0]["function"]["name"] == "get_weather";
        if (!ok) { ++bad; std::fprintf(stderr, "FAIL reply parse: reasoning '%s' content '%s' tool_calls %s\n", pc.reasoning.c_str(), pc.content.c_str(), pc.tool_calls.c_str()); }
        else std::printf("reply parse: reasoning, content and tool call recovered; arguments %s\n", calls[0]["function"]["arguments"].dump().c_str());
    }
    std::printf("%zu template cases, %d failures\n", cases.size(), bad);
    return bad ? 1 : 0;
}
