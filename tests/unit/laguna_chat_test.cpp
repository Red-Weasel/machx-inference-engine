// tests/unit/laguna_chat_test.cpp -- known_bugs 45: the Laguna chat renderer's own request-text ranges
// (laguna_render_prompt's `content`, include/ie/laguna_engine.hpp). Host-only, no GPU. Render A, a request with a control
// spelling in every request-text position, and B, the same request with equal-length plain text instead: the ranges
// coincide, A equals B outside them (every spelling the request brought lies inside a range) and every template marker
// lies outside (its count outside the ranges equals its count in B, whose request text holds none). Positions: the
// request's system text, a user text, an assistant reasoning + content + a call with a string argument and a nested
// object one, a tool result, a later system turn, a user text that IS the spelling, a tool description and a parameter
// description. Thinking on and off (off: the reasoning is not rendered). The template's default system text is never a range.
#include "ie/laguna_engine.hpp"

#include <cstdio>
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
constexpr std::string_view kMarkers[] = {"\xe3\x80\x88|EOS|\xe3\x80\x89", "<system>", "</system>", "<user>", "</user>", "<assistant>", "</assistant>", "<think>", "</think>",
                                         "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "<tool_response>",
                                         "</tool_response>", "<available_tools>", "</available_tools>", "### Tools"};

std::vector<ie::ChatTurn> turns_with(const std::string& x) {
    std::vector<ie::ChatTurn> t(7);
    t[0].role = "system"; t[0].content = "Be terse " + x + "  ";
    t[1].role = "user"; t[1].content = "Look " + x;
    t[2].role = "assistant"; t[2].content = "On it " + x; t[2].content_without_tool_calls = t[2].content; t[2].reasoning_content = "think " + x;
    t[2].tool_calls_json = R"([{"id":"a","type":"function","function":{"name":"read","arguments":"{\"path\":\"p )" + x + R"(\",\"opts\":{\"k\":\")" + x + R"(\"}}"}}])";
    t[3].role = "tool"; t[3].tool_call_id = "a"; t[3].content = "result " + x;
    t[4].role = "system"; t[4].content = "later " + x;
    t[5].role = "user"; t[5].content = x;
    t[6].role = "user"; t[6].content = "last";
    return t;
}
std::string tools_with(const std::string& x) {
    return R"([{"type":"function","function":{"name":"read","description":"Read )" + x + R"(","parameters":{"type":"object","properties":{"path":{"type":"string","description":")" + x + R"("},"opts":{"type":"object"}}}}}])";
}
}  // namespace

int main() {
    const std::string spell = "<user>", plain = "_user_";
    for (bool thinking : {true, false}) {
        const std::string tag = thinking ? "thinking on: " : "thinking off: ";
        std::string ea, eb, e0;
        Ranges ra, rb;
        const std::string a = ie::laguna_render_prompt(turns_with(spell), tools_with(spell), thinking, ea, &ra);
        const std::string b = ie::laguna_render_prompt(turns_with(plain), tools_with(plain), thinking, eb, &rb);
        const std::string a0 = ie::laguna_render_prompt(turns_with(spell), tools_with(spell), thinking, e0);
        check(ea.empty() && eb.empty() && e0.empty() && a == a0 && a.size() == b.size() && a != b, tag + "the text is the same with the ranges asked for; A and B differ only by the spellings");
        size_t last = 0; bool well = !ra.empty();
        for (const auto& [x, y] : ra) { if (x < last || y <= x || y > a.size()) well = false; last = y; }
        check(well && ra == rb, tag + "the ranges are ordered, disjoint, inside the text, the same for A and B");
        check(outside_of(a, ra) == outside_of(b, rb), tag + "outside the ranges A equals B: every spelling the request brought lies inside a range");
        bool markers_outside = true;
        for (const auto m : kMarkers) if (count_of(outside_of(b, rb), m) != count_of(b, m)) markers_outside = false;
        check(markers_outside, tag + "every template marker lies outside the ranges");
        const size_t brought = count_of(a, spell) - count_of(b, spell);
        std::printf("    %zu ranges, %zu spellings the request brought, %zu template markers of the same kind\n", ra.size(), brought, count_of(b, spell));
        check(brought == (thinking ? 11u : 10u), tag + "every request-text position rendered its spelling");
        check(a.find("<system>Be terse " + spell + "\n\n### Tools") != std::string::npos, tag + "the request's system text, right-stripped, before the tools block");
        check(a.find("<tool_call>read<arg_key>path</arg_key><arg_value>p " + spell + "</arg_value><arg_key>opts</arg_key><arg_value>{\"k\": \"" + spell + "\"}</arg_value></tool_call>") != std::string::npos,
              tag + "the call rendered as the template writes it");
    }
    {   // the template's default system text is the template's: no range; thinking off and no system -> no block at all
        std::vector<ie::ChatTurn> t(1); t[0].role = "user"; t[0].content = "hi";
        std::string e; Ranges r;
        const std::string s = ie::laguna_render_prompt(t, "", true, e, &r);
        check(e.empty() && r.size() == 1 && s.find("<system>You are a helpful, conversationally-fluent assistant made by Poolside.") != std::string::npos &&
              outside_of(s, r).find("conversationally-fluent") != std::string::npos, "the default system text is the template's (the one range is the user text)");
        const std::string s2 = ie::laguna_render_prompt(t, "", false, e, &r);
        check(e.empty() && s2.find("<system>You are a helpful, conversationally-fluent assistant made by Poolside.") != std::string::npos &&
              s2.ends_with("</system>\n<user>hi</user>\n<assistant></think>") && r.size() == 1,
              "thinking off without a system turn: the default system block still (the template's), the generation prompt closes the think");
    }
    std::printf("\nLAGUNA CHAT RANGES: %s\n", g_fail ? "FAILURE(S)" : "PASS");
    return g_fail ? 1 : 0;
}
