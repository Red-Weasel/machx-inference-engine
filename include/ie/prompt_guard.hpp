// include/ie/prompt_guard.hpp -- the content ranges of a rendered prompt, found after the fact (known_bugs 45, 2026-11-05).
//
// Every chat renderer in the engine builds its prompt by concatenating template markers with the request's own strings
// (message contents, tool results, tool-call arguments, the tool schemas) and the serve tokenises that string with
// special-token matching on -- so a control-token spelling inside the request's text became the control token: a page a
// tool fetched could forge a turn, end the reasoning or open a tool-call block. Since 2026-11-05 every chat renderer
// records its content ranges as it renders (deepseek41_prompt.cpp Piece; tokenizer.hpp PromptPiece for ChatML / Qwen,
// Llama-3, Mistral, R1-distill, DeepSeek-V4, Granite, Gemma, gpt-oss, GLM5, MiMo, Laguna, Inkling: tool-call arguments
// and tool schemas included, never a refusal), so no serve path uses this any more. Kept, host-tested, as the
// after-the-fact fallback for a renderer that only has a plain string -- it finds the ranges afterwards:
//   * only a request string that CONTAINS a special spelling needs a range (any other string tokenises the same either way);
//   * such a string is located verbatim in the prompt, in request order; found exactly once -> its range;
//   * found nowhere (the renderer transformed it: escaped, trimmed) or more than once (ambiguous) -> `error`, and the caller
//     refuses the request: the model never sees a forged marker, and a request that could not be guarded is not served
//     with one. A request without any spelling in its text gets an empty range list and tokenises byte for byte as before.
// Header-only, no SYCL: tests/unit/prompt_guard_test.cpp.
#pragma once

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ie {

struct PromptGuard {
    std::vector<std::pair<size_t, size_t>> ranges;   // byte ranges of `prompt` that are request text holding a spelling
    std::string error;                               // "" when every such string was isolated
};

inline bool prompt_text_has_spelling(std::string_view text, std::span<const std::string_view> specials) {
    for (const auto s : specials) if (!s.empty() && text.find(s) != std::string_view::npos) return true;
    return false;
}

inline PromptGuard prompt_guard_ranges(std::string_view prompt, std::span<const std::string_view> request_texts,
                                       std::span<const std::string_view> specials) {
    PromptGuard g;
    size_t cursor = 0;
    for (const auto t : request_texts) {
        if (t.empty() || !prompt_text_has_spelling(t, specials)) continue;
        const size_t at = prompt.find(t, cursor);
        if (at == std::string_view::npos) {
            g.error = "a request text holding a control-token spelling was not found verbatim in the rendered prompt";
            return g;
        }
        if (prompt.find(t, at + 1) != std::string_view::npos) {
            g.error = "a request text holding a control-token spelling occurs more than once in the rendered prompt";
            return g;
        }
        g.ranges.emplace_back(at, at + t.size());
        cursor = at + t.size();
    }
    return g;
}

// The ranges of a PREFIX of the prompt (the stable conversation boundary, the shared system prefix): the same ranges, cut
// at the prefix's end, dropping the ones past it.
inline std::vector<std::pair<size_t, size_t>> prompt_guard_clip(std::span<const std::pair<size_t, size_t>> ranges, size_t len) {
    std::vector<std::pair<size_t, size_t>> out;
    for (const auto& [b, e] : ranges) if (b < len) out.emplace_back(b, e < len ? e : len);
    return out;
}

}  // namespace ie

#include "ie/tokenizer.hpp"   // ChatTurn, Tokenizer::special_spellings (below the pure part so the host test stays small)

namespace ie {

// The guard of one chat request: the request texts are every turn's content (the tool-call-free content when the turn
// carries one: that is what the renderers insert) and reasoning, against the rendered prompt and this tokenizer's special
// spellings. `error` set = refuse the request with it.
inline PromptGuard chat_prompt_guard(std::string_view prompt, std::span<const ChatTurn> turns, const Tokenizer& tok) {
    std::vector<std::string_view> texts;
    for (const auto& t : turns) {
        texts.push_back(t.content_without_tool_calls ? std::string_view(*t.content_without_tool_calls) : std::string_view(t.content));
        if (t.reasoning_content) texts.push_back(*t.reasoning_content);
    }
    auto g = prompt_guard_ranges(prompt, texts, tok.special_spellings());
    if (!g.error.empty()) g.error = "request text holds a control-token spelling the renderer could not isolate (" + g.error + ")";
    return g;
}

}  // namespace ie
