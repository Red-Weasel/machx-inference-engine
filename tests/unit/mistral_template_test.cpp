// tests/unit/mistral_template_test.cpp — Wave-1 Task 3/4 host unit test.
// Byte-exact golden tests for the two NEW chat-template builders:
//   * build_mistral_prompt   — [INST] … [/INST] (+ [SYSTEM_PROMPT] variant)
//   * build_deepseek_prompt  — R1-Distill <｜User｜>/<｜Assistant｜> sentinels
// Both are pure string builders (no GGUF/SYCL); BOS is NOT a template literal
// (encode() prepends it). Golden strings mirror llama.cpp's rendered prompts.
//
// Host-only: no GPU, no GGUF. Built like tekken_tokenizer_test — compiles the
// SYCL-free tokenizer + gguf_reader + dtype TUs directly, does NOT link ie_core.
// -UNDEBUG keeps the asserts live under the Release -DNDEBUG.
#undef NDEBUG
#include "ie/tokenizer.hpp"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

using ie::ChatTurn;

// Fullwidth sentinels (U+FF5C bar, U+2581 underscore) as UTF-8 byte literals.
// Split each hex escape from a following hex-digit char ('U','A','e','o','s').
#define DS_USER "<\xef\xbd\x9c" "User" "\xef\xbd\x9c>"
#define DS_ASST "<\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c>"
#define DS_EOS  "<\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c>"

int main() {
    int checks = 0;

    // ======================= build_mistral_prompt =======================
    // (a) single user turn → "[INST] {u} [/INST]" (no BOS literal; no trailing).
    {
        std::vector<ChatTurn> t = {{"user", "Hello"}};
        auto s = ie::build_mistral_prompt(t, /*gen=*/true, /*sysblock=*/false);
        assert(s == "[INST] Hello [/INST]");
        checks++;
    }
    // (b) system + user (folded into the first [INST], default v1/v2 form).
    {
        std::vector<ChatTurn> t = {{"system", "You are terse."}, {"user", "Hi"}};
        auto s = ie::build_mistral_prompt(t, true, false);
        assert(s == "[INST] You are terse.\n\nHi [/INST]");
        checks++;
    }
    // (c) Devstral / v3+ [SYSTEM_PROMPT] block variant (mistral_sysprompt==1).
    {
        std::vector<ChatTurn> t = {{"system", "SYS"}, {"user", "Hi"}};
        auto s = ie::build_mistral_prompt(t, true, /*sysblock=*/true);
        assert(s == "[INST] [SYSTEM_PROMPT]SYS[/SYSTEM_PROMPT]\n\nHi [/INST]");
        checks++;
    }
    // (d) multi-turn: assistant reply closes with "</s>"; next user re-opens.
    {
        std::vector<ChatTurn> t = {
            {"user", "u1"}, {"assistant", "a1"}, {"user", "u2"}};
        auto s = ie::build_mistral_prompt(t, true, false);
        assert(s == "[INST] u1 [/INST] a1</s>[INST] u2 [/INST]");
        checks++;
    }
    // (e) system folds ONLY into the first user turn, not subsequent ones.
    {
        std::vector<ChatTurn> t = {
            {"system", "S"}, {"user", "u1"}, {"assistant", "a1"}, {"user", "u2"}};
        auto s = ie::build_mistral_prompt(t, true, false);
        assert(s == "[INST] S\n\nu1 [/INST] a1</s>[INST] u2 [/INST]");
        checks++;
    }

    // ======================= build_deepseek_prompt ======================
    // (f) single user, generation prompt opens assistant + <think> (R1 reasons).
    {
        std::vector<ChatTurn> t = {{"user", "Hi"}};
        auto s = ie::build_deepseek_prompt(t, /*gen=*/true, /*think=*/true);
        assert(s == DS_USER "Hi" DS_ASST "<think>\n");
        checks++;
    }
    // (g) enable_thinking=false → empty-think convention.
    {
        std::vector<ChatTurn> t = {{"user", "Hi"}};
        auto s = ie::build_deepseek_prompt(t, true, /*think=*/false);
        assert(s == DS_USER "Hi" DS_ASST "<think>\n\n</think>\n\n");
        checks++;
    }
    // (h) system prefix is emitted verbatim at the front (R1-Distill convention).
    {
        std::vector<ChatTurn> t = {{"system", "SYS"}, {"user", "Hi"}};
        auto s = ie::build_deepseek_prompt(t, true, true);
        assert(s == "SYS" DS_USER "Hi" DS_ASST "<think>\n");
        checks++;
    }
    // (i) multi-turn: assistant reply closes with the EOS sentinel.
    {
        std::vector<ChatTurn> t = {
            {"user", "u1"}, {"assistant", "a1"}, {"user", "u2"}};
        auto s = ie::build_deepseek_prompt(t, true, true);
        assert(s == DS_USER "u1" DS_ASST "a1" DS_EOS DS_USER "u2" DS_ASST "<think>\n");
        checks++;
    }
    // (j) no generation prompt → no trailing assistant/think.
    {
        std::vector<ChatTurn> t = {{"user", "Hi"}};
        auto s = ie::build_deepseek_prompt(t, /*gen=*/false, true);
        assert(s == DS_USER "Hi");
        checks++;
    }

    std::printf("mistral_template_test: all OK (%d checks)\n", checks);
    return 0;
}
