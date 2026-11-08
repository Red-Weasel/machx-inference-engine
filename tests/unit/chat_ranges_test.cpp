// tests/unit/chat_ranges_test.cpp -- known_bugs 45: the request-text ranges the plain-string chat renderers record as they
// render (tokenizer.hpp: build_llama3_prompt, build_mistral_prompt, build_deepseek_prompt (R1-distill), build_deepseek4_prompt,
// build_granite_prompt, build_gemma_prompt, build_harmony_prompt). Host-only, no GPU. For each family: render A, a request
// with a control spelling in every request-text position the family renders, and B, the same request with equal-length
// plain text instead; the ranges coincide, A equals B outside them (every spelling the request brought lies inside a
// range), every template marker lies outside them (its count outside the ranges equals its count in B, whose request text
// holds none), the render without the generation prompt is a byte prefix (where the engine takes a stable boundary), and
// the text is the same whether or not the ranges are asked for.
#include "ie/prompt_guard.hpp"
#include "ie/tokenizer.hpp"

#include <cstdio>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) { std::printf("[%s] %s\n", ok ? " ok " : "FAIL", what.c_str()); if (!ok) ++g_fail; }
using Ranges = std::vector<std::pair<size_t, size_t>>;
std::string outside_of(const std::string& s, const Ranges& r) { std::string o = s; for (const auto& [b, e] : r) for (size_t i = b; i < e; ++i) o[i] = '#'; return o; }
size_t count_of(const std::string& s, std::string_view m) { size_t n = 0; for (size_t p = s.find(m); p != std::string::npos; p = s.find(m, p + m.size())) ++n; return n; }
ie::ChatTurn T(const char* role, const std::string& content) { ie::ChatTurn t; t.role = role; t.content = content; return t; }

// One family: `render(turns, tools, gen, &ranges)` and its marker list.
struct Family {
    const char* name;
    std::function<std::string(std::span<const ie::ChatTurn>, const std::string&, bool, Ranges*)> render;
    std::vector<std::string_view> markers;
    bool stable_prefix;   // the engine takes the no-generation-prompt render as a byte prefix (cache boundary)
};

void run(const Family& f, const std::vector<ie::ChatTurn>& ta, const std::vector<ie::ChatTurn>& tb, const std::string& tools_a,
         const std::string& tools_b, std::string_view spell, size_t expect) {
    const std::string tag = std::string(f.name) + ": ";
    Ranges ra, rb;
    const std::string a = f.render(ta, tools_a, true, &ra), b = f.render(tb, tools_b, true, &rb), a0 = f.render(ta, tools_a, true, nullptr);
    check(a == a0 && a.size() == b.size() && a != b, tag + "the text is the same with the ranges asked for; A and B differ only by the spellings");
    size_t last = 0; bool well = !ra.empty();
    for (const auto& [x, y] : ra) { if (x < last || y <= x || y > a.size()) well = false; last = y; }
    check(well && ra == rb, tag + "the ranges are ordered, disjoint, inside the text, the same for A and B");
    check(outside_of(a, ra) == outside_of(b, rb), tag + "outside the ranges A equals B: every spelling the request brought lies inside a range");
    bool markers_outside = true;
    for (const auto m : f.markers) if (count_of(outside_of(b, rb), m) != count_of(b, m)) markers_outside = false;
    check(markers_outside, tag + "every template marker lies outside the ranges");
    const size_t brought = count_of(a, spell) - count_of(b, spell);
    std::printf("    %zu ranges, %zu spellings the request brought, %zu template markers of the same kind\n", ra.size(), brought, count_of(b, spell));
    check(brought == expect, tag + "every request-text position rendered its spelling");
    if (f.stable_prefix) {
        Ranges rs;
        const std::string stable = f.render(ta, tools_a, false, &rs);
        check(a.starts_with(stable) && ie::prompt_guard_clip(ra, stable.size()) == rs, tag + "the render without the generation prompt is a prefix with the clipped ranges");
    }
}
}  // namespace

int main() {
    // Plain conversations (no tools): a system text, a user text, an assistant text, a user text that IS the spelling.
    const auto plain_turns = [](const std::string& x) {
        return std::vector<ie::ChatTurn>{T("system", "Be terse " + x), T("user", "Look " + x), T("assistant", "On it " + x), T("user", x), T("user", "last")};
    };
    {
        const std::string spell = "<|eot_id|>", plain = "<_eot_id_>";
        Family f{"llama3", [](auto t, const std::string&, bool gen, Ranges* r) { return ie::build_llama3_prompt(t, gen, r); },
                 {"<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"}, false};
        run(f, plain_turns(spell), plain_turns(plain), "", "", spell, 4);
    }
    {
        const std::string spell = "[INST]", plain = "_INST_";
        for (bool block : {false, true}) {
            Family f{block ? "mistral, system block" : "mistral", [block](auto t, const std::string&, bool gen, Ranges* r) { return ie::build_mistral_prompt(t, gen, block, r); },
                     {"[INST] ", " [/INST]", "</s>", "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]"}, false};
            run(f, plain_turns(spell), plain_turns(plain), "", "", spell, 4);
        }
    }
    {
        const std::string spell = "<\xef\xbd\x9c" "User" "\xef\xbd\x9c>", plain10 = "<_User____>_";   // <｜User｜> is 12 bytes; so is the plain text
        check(spell.size() == 12 && plain10.size() == 12, "r1: the spelling and its plain stand-in have the same length");
        for (bool thinking : {true, false}) {
            Family f{thinking ? "r1-distill, thinking" : "r1-distill", [thinking](auto t, const std::string&, bool gen, Ranges* r) { return ie::build_deepseek_prompt(t, gen, thinking, r); },
                     {"<\xef\xbd\x9c" "User" "\xef\xbd\x9c>", "<\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c>", "<\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c>", "<think>", "</think>"}, false};
            run(f, plain_turns(spell), plain_turns(plain10), "", "", spell, 4);
        }
    }
    {
        const std::string spell = "<|end_of_text|>", plain = "<_end_of_text_>";
        Family f{"granite", [](auto t, const std::string&, bool gen, Ranges* r) { return ie::build_granite_prompt(t, gen, r); },
                 {"<|start_of_role|>", "<|end_of_role|>", "<|end_of_text|>"}, false};
        run(f, plain_turns(spell), plain_turns(plain), "", "", spell, 4);
    }
    {
        const std::string spell = "<turn|>", plain = "<turn_>";
        Family f{"gemma", [](auto t, const std::string&, bool gen, Ranges* r) { return ie::build_gemma_prompt(t, gen, r); },
                 {"<|turn>", "<turn|>", "<|channel>", "<channel|>", "model\n"}, true};
        run(f, plain_turns(spell), plain_turns(plain), "", "", spell, 4);
    }
    {   // gpt-oss Harmony: tools as the developer namespace, a prior call re-rendered on the commentary channel, a tool result
        const std::string spell = "<|end|>", plain = "<_end_>";
        const auto turns = [](const std::string& x) {
            return std::vector<ie::ChatTurn>{T("system", "Be terse " + x), T("user", "Look " + x),
                T("assistant", "<tool_call>{\"name\": \"read\", \"arguments\": {\"path\": \"p " + x + "\"}}</tool_call>"),
                T("tool", "result " + x), T("assistant", "On it " + x), T("user", x), T("user", "last")};
        };
        const auto tools = [](const std::string& x) {
            return R"([{"type":"function","function":{"name":"read","description":"Read )" + x + R"(","parameters":{"type":"object","properties":{"path":{"type":"string"},"mode":{"type":"string","enum":["a )" + x + R"(","b"]}},"required":["path"]}}}])";
        };
        Family f{"harmony", [](auto t, const std::string& tools, bool gen, Ranges* r) { return ie::build_harmony_prompt(t, gen, "high", tools, r); },
                 {"<|start|>", "<|end|>", "<|message|>", "<|channel|>", "<|constrain|>", "<|call|>", "system<|message|>", "developer<|message|>", "# Instructions",
                  "namespace functions", "commentary to=functions.", "to=assistant", "Knowledge cutoff"}, true};
        run(f, turns(spell), turns(plain), tools(spell), tools(plain), spell, 8);   // system, user, call args, result, content, bare user, description, enum
    }
    {   // DeepSeek-V4: the DSML tools block, a tool result merged into the user turn, a prior call transcoded into DSML
        const std::string spell = "<\xef\xbd\x9c" "User" "\xef\xbd\x9c>", plain10 = "<_User____>_";   // 12 bytes each
        const auto turns = [](const std::string& x) {
            return std::vector<ie::ChatTurn>{T("system", "Be terse " + x), T("user", "Look " + x),
                T("assistant", "Calling " + x + "\n<tool_call>{\"name\": \"read\", \"arguments\": {\"path\": \"p " + x + "\", \"n\": 2}}</tool_call>"),
                T("tool", "result " + x), T("user", "then " + x), T("assistant", "On it " + x), T("user", x), T("user", "last")};
        };
        const auto tools = [](const std::string& x) {
            return R"([{"type":"function","function":{"name":"read","description":"Read )" + x + R"(","parameters":{"type":"object","properties":{"path":{"type":"string","description":")" + x + R"("}}}}}])";
        };
        const std::string dsml = "\xef\xbd\x9c" "DSML" "\xef\xbd\x9c";
        for (bool thinking : {true, false}) {
            Family f{thinking ? "deepseek4, thinking" : "deepseek4", [thinking](auto t, const std::string& tools, bool gen, Ranges* r) {
                         ie::DeepSeek4ChatOptions o; o.thinking = thinking; o.tools_json = tools; o.reasoning_effort = "high";
                         return ie::build_deepseek4_prompt(t, gen, o, r); },
                     {"<\xef\xbd\x9c" "User" "\xef\xbd\x9c>", "<\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c>", "<think>", "</think>", "<tool_result>", "</tool_result>",
                      "<" + dsml + "tool_calls>", "</" + dsml + "tool_calls>", "<" + dsml + "invoke name=\"", "</" + dsml + "invoke>", "<" + dsml + "parameter name=\"",
                      "</" + dsml + "parameter>", "## Tools", "### Available Tool Schemas"}, true};
            // system, user, prose + path value, result, then-user, content, bare user, description, parameter description
            run(f, turns(spell), turns(plain10), tools(spell), tools(plain10), spell, 10);
        }
        {   // a turn with images: the engine's placeholders are the template's text, a typed one in an image-free turn is the request's
            std::vector<ie::ChatTurn> t = {T("user", "<image>see<image> this " + spell), T("assistant", "ok"), T("user", "typed <image> here")};
            t[0].images = {"bytes", "bytes"};
            ie::DeepSeek4ChatOptions o; o.image_placeholder = "<image>";
            Ranges r;
            const std::string s = ie::build_deepseek4_prompt(t, true, o, &r);
            const std::string out = outside_of(s, r);
            check(count_of(s, "<image>") == 3 && count_of(out, "<image>") == 2 && count_of(s, spell) == 3 && count_of(out, spell) == 2 &&
                  s.find("<image>see<image> this " + spell) != std::string::npos,
                  "deepseek4, images: the two placed placeholders are the template's, the typed one and the spelling are the request's");
        }
    }
    std::printf("\nCHAT RANGES: %s\n", g_fail ? "FAILURE(S)" : "PASS");
    return g_fail ? 1 : 0;
}
