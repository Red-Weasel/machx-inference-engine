// tests/unit/mimo26_chat_test.cpp -- known_bugs 45: the MiMo-V2.6 chat renderer's own request-text ranges
// (mimo26_render_chat's `content`, include/ie/mimo26_engine.hpp). Host-only, no GPU. Render A, a request with a control
// spelling in every request-text position, and B, the same request with equal-length plain text instead: the ranges
// coincide, A equals B outside them (every spelling the request brought lies inside a range) and every template marker
// lies outside (its count outside the ranges equals its count in B, whose request text holds none). Positions: a system
// text, a user text with two images placed (the placeholders stay the template's), an assistant content + reasoning +
// three calls (a string argument, a nested object argument, a custom call's input, an arguments string that is not JSON),
// a tool result, a user text that IS the spelling, a tool description and a parameter description. Thinking on and off.
#include "ie/mimo26_engine.hpp"
#include "ie/tokenizer.hpp"

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
constexpr std::string_view kMarkers[] = {"<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>", "<function=", "</function>",
                                         "<parameter=", "</parameter>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<tools>", "</tools>"};

std::vector<ie::Mimo26Message> messages(const std::string& x) {
    std::vector<ie::Mimo26Message> m(6);
    m[0].role = "system"; m[0].content = "Be terse " + x;
    m[1].role = "user"; m[1].content = ie::mimo26_place_images("see " + std::string(ie::kChatImageMarker) + " and " + std::string(ie::kChatImageMarker) + " then " + x, 2, &m[1].image_at);
    m[2].role = "assistant"; m[2].content = "On it " + x; m[2].reasoning = "think " + x;
    m[2].tool_calls_json = R"([{"id":"a","type":"function","function":{"name":"read","arguments":"{\"path\":\"p )" + x + R"(\",\"opts\":{\"k\":\")" + x + R"(\"}}"}},
                               {"id":"b","type":"custom","custom":{"name":"shell","input":"ls )" + x + R"("}},
                               {"id":"c","type":"function","function":{"name":"raw","arguments":"not json )" + x + R"("}}])";
    m[3].role = "tool"; m[3].content = "result " + x;
    m[4].role = "user"; m[4].content = x;
    m[5].role = "user"; m[5].content = "last";
    return m;
}
std::string tools(const std::string& x) {
    return R"([{"type":"function","function":{"name":"read","description":"Read )" + x + R"(","parameters":{"type":"object","properties":{"path":{"type":"string","description":")" + x + R"("},"opts":{"type":"object"}}}}},{"type":"function","function":{"name":"raw"}}])";
}
}  // namespace

int main() {
    const std::string spell = "<|im_end|>", plain = "<_im_end_>";
    for (bool thinking : {true, false}) {
        const std::string tag = thinking ? "thinking on: " : "thinking off: ";
        std::string ea, eb, e0;
        Ranges ra, rb;
        const std::string a = ie::mimo26_render_chat(messages(spell), tools(spell), true, thinking, ea, &ra);
        const std::string b = ie::mimo26_render_chat(messages(plain), tools(plain), true, thinking, eb, &rb);
        const std::string a0 = ie::mimo26_render_chat(messages(spell), tools(spell), true, thinking, e0);
        check(ea.empty() && eb.empty() && e0.empty() && a == a0 && a.size() == b.size() && a != b, tag + "the text is the same with the ranges asked for; A and B differ only by the spellings");
        size_t last = 0; bool well = !ra.empty();
        for (const auto& [x, y] : ra) { if (x < last || y <= x || y > a.size()) well = false; last = y; }
        check(well && ra == rb, tag + "the ranges are ordered, disjoint, inside the text, the same for A and B");
        check(outside_of(a, ra) == outside_of(b, rb), tag + "outside the ranges A equals B: every spelling the request brought lies inside a range");
        bool markers_outside = true;
        for (const auto m : kMarkers) if (count_of(outside_of(b, rb), m) != count_of(b, m)) markers_outside = false;
        check(markers_outside, tag + "every template marker lies outside the ranges (the two image placeholders included)");
        const size_t brought = count_of(a, spell) - count_of(b, spell);
        std::printf("    %zu ranges, %zu spellings the request brought, %zu template markers of the same kind\n", ra.size(), brought, count_of(b, spell));
        check(brought == 12, tag + "every request-text position rendered its spelling");
        check(a.find("<tool_call><function=read><parameter=path>p " + spell + "</parameter><parameter=opts>{\"k\": \"" + spell + "\"}</parameter></function></tool_call>"
                     "<tool_call><function=shell>ls " + spell + "</function></tool_call><tool_call><function=raw>not json " + spell + "</function></tool_call>") != std::string::npos,
              tag + "the three call forms rendered as the template writes them");
    }
    {   // a vision spelling typed into a message without images is the request's; the engine's placeholders never are
        std::vector<ie::Mimo26Message> m(2);
        m[0].role = "user"; m[0].content = "typed <|vision_start|><|image_pad|><|vision_end|> here";
        m[1].role = "user"; m[1].content = ie::mimo26_place_images("no marker", 1, &m[1].image_at);
        std::string e; Ranges r;
        const std::string s = ie::mimo26_render_chat(m, "", true, true, e, &r);
        const std::string o = outside_of(s, r);
        check(e.empty() && m[1].image_at == std::vector<size_t>{0} && count_of(s, "<|image_pad|>") == 2 && count_of(o, "<|image_pad|>") == 1 &&
              s.find("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>no marker<|im_end|>") != std::string::npos,
              "a typed vision spelling is the request's text; the placed one (in front when the markers do not match) is the template's");
    }
    {   // no messages, no tools: nothing but the generation prompt, no ranges; an empty tools array renders no block
        std::string e; Ranges r;
        const std::string s = ie::mimo26_render_chat({}, "[]", true, false, e, &r);
        check(e.empty() && r.empty() && s == "<|im_start|>assistant\n<think></think>", "an empty conversation: no ranges, the generation prompt only");
    }
    std::printf("\nMIMO26 CHAT RANGES: %s\n", g_fail ? "FAILURE(S)" : "PASS");
    return g_fail ? 1 : 0;
}
