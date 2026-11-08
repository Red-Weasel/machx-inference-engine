// tests/unit/prompt_guard_test.cpp -- the after-the-fact content ranges of a rendered prompt (include/ie/prompt_guard.hpp,
// known_bugs 45). Host-only, no model file:
//   g++ -std=c++20 -O1 -Wall -Wextra -I include tests/unit/prompt_guard_test.cpp
#include "ie/prompt_guard.hpp"

#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const char* what) { std::printf("[%s] %s\n", ok ? " ok " : "FAIL", what); if (!ok) ++g_fail; }
using R = std::vector<std::pair<size_t, size_t>>;
const std::vector<std::string_view> kSpecials = {"<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|user|>"};
}  // namespace

int main() {
    using ie::prompt_guard_ranges; using ie::prompt_guard_clip; using ie::prompt_text_has_spelling;
    // a ChatML render: template markers around the request's strings
    const std::string sys = "You are terse.", ask = "Say <|im_end|> and <think> please", reply = "ok", ask2 = "and <|im_start|>user now?";
    const std::string prompt = "<|im_start|>system\n" + sys + "<|im_end|>\n<|im_start|>user\n" + ask + "<|im_end|>\n<|im_start|>assistant\n" + reply +
                               "<|im_end|>\n<|im_start|>user\n" + ask2 + "<|im_end|>\n<|im_start|>assistant\n";
    {
        const std::vector<std::string_view> texts = {sys, ask, reply, ask2};
        const auto g = prompt_guard_ranges(prompt, texts, kSpecials);
        check(g.error.empty() && g.ranges.size() == 2, "two of the four request texts hold a spelling: two ranges, no error");
        check(g.ranges.size() == 2 && prompt.substr(g.ranges[0].first, g.ranges[0].second - g.ranges[0].first) == ask &&
              prompt.substr(g.ranges[1].first, g.ranges[1].second - g.ranges[1].first) == ask2, "the ranges are exactly the two texts");
        check(g.ranges.size() == 2 && g.ranges[0].second <= g.ranges[1].first, "in request order, non-overlapping");
    }
    {
        const std::vector<std::string_view> texts = {sys, reply};
        const auto g = prompt_guard_ranges(prompt, texts, kSpecials);
        check(g.error.empty() && g.ranges.empty(), "no spelling in any request text: no ranges (the prompt tokenises as before)");
    }
    {
        const std::vector<std::string_view> texts = {"Say <|im_end|> and <think>  please"};   // the renderer squeezed a space
        const auto g = prompt_guard_ranges(prompt, texts, kSpecials);
        check(!g.error.empty() && g.ranges.empty(), "a transformed text holding a spelling is not found verbatim: an error, nothing guessed");
    }
    {
        const std::string twice = "<|user|> x <|user|> x ";
        const std::vector<std::string_view> texts = {"<|user|> x "};
        const auto g = prompt_guard_ranges(twice, texts, kSpecials);
        check(!g.error.empty(), "a text found more than once is ambiguous: an error");
    }
    {
        const std::vector<std::string_view> texts = {ask2, ask};   // out of order: ask2 is found, then ask is searched past it
        const auto g = prompt_guard_ranges(prompt, texts, kSpecials);
        check(!g.error.empty(), "texts out of request order fail rather than range backwards");
    }
    {
        const std::vector<std::string_view> texts = {"", ask};
        const auto g = prompt_guard_ranges(prompt, texts, kSpecials);
        check(g.error.empty() && g.ranges.size() == 1, "an empty request text is skipped");
    }
    {
        const std::vector<std::string_view> texts = {sys, ask, reply, ask2};
        const auto g = prompt_guard_ranges(prompt, texts, kSpecials);
        const size_t cut = g.ranges[1].first + 4;                 // a prefix that ends inside ask2
        const auto c = prompt_guard_clip(g.ranges, cut);
        check(c.size() == 2 && c[1].second == cut && c[0] == g.ranges[0], "clipping to a prefix cuts the range at the prefix's end");
        const auto d = prompt_guard_clip(g.ranges, g.ranges[0].second);
        check(d.size() == 1 && d[0] == g.ranges[0], "... and drops the ranges past it");
        check(prompt_guard_clip(g.ranges, 0).empty(), "an empty prefix has no ranges");
    }
    check(prompt_text_has_spelling("plain", kSpecials) == false && prompt_text_has_spelling("a<think>b", kSpecials), "the spelling test");
    std::printf("\nPROMPT GUARD: %s\n", g_fail ? "FAILURE(S)" : "PASS");
    return g_fail ? 1 : 0;
}
