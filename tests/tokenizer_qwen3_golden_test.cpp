// tests/tokenizer_qwen3_golden_test.cpp — P2 Task 6: tokenizer goldens for
// the qwen3-8b dense vocab (gpt2 BPE, pre `qwen2`, vocab 151936).
//
// Reference IDs captured 2026-06-10 by running (CPU-only, vocab load):
//   /home/weezy/llama.cpp-vulkan/llama-b8902/llama-tokenize \
//     -m ~/.seal/models/Qwen3-8B-Q4_K_M.gguf -f <case-bytes> --ids \
//     [--no-parse-special when the engine encodes with allow_special=false]
// No BOS is prepended (qwen2 family: add_bos_token=false).
//
// Host-only (mmap reads, no GPU). Skips-with-warning (exit 0) if the GGUF
// is absent so CI on other boxes stays green.
//
// Each case asserts BOTH:
//   1. Tokenizer::encode(text, allow_special) == reference IDs, and
//   2. decode(encode(text)) == text  (lossless round-trip, specials literal).
#undef NDEBUG  // build is Release (-DNDEBUG); asserts must stay live here
#include "ie/gguf.hpp"
#include "ie/tokenizer.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

struct Golden {
    const char*          name;
    const char*          text;
    bool                 allow_special;  // mirrors llama-tokenize parse-special
    std::vector<int32_t> ids;
};

// 13 cases per the P2 plan: plain ASCII, leading/trailing spaces,
// numbers+punct, indented code, CJK, emoji, mixed RTL, ChatML message,
// <think> block, empty-adjacent specials — plus contractions (the f508c01
// qwen2 pre-split fix, re-verified on this vocab), whitespace edges, and a
// full build_chatml_prompt-shaped 2-turn prompt.
const std::vector<Golden> kGoldens = {
    {"plain ASCII pangram",
     "The quick brown fox jumps over the lazy dog.", false,
     {785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13}},

    {"leading/trailing spaces",
     "  leading and trailing spaces  ", false,
     {220, 6388, 323, 27748, 12621, 256}},

    {"numbers + punctuation",
     "3.14159, x<=42!", false,
     {18, 13, 16, 19, 16, 20, 24, 11, 856, 8203, 19, 17, 0}},

    {"indented code snippet",
     "def fib(n):\n    if n < 2:\n        return n\n    return fib(n-1) + fib(n-2)\n",
     false,
     {750, 15801, 1445, 982, 262, 421, 308, 366, 220, 17, 510, 286, 470,
      308, 198, 262, 470, 15801, 1445, 12, 16, 8, 488, 15801, 1445, 12,
      17, 340}},

    {"CJK (zh + ja)",
     "你好,世界!東京タワーは高い。", false,
     {108386, 11, 99489, 0, 102356, 46553, 46207, 125980, 15322, 126721,
      1773}},

    {"emoji",
     "Emoji test \xf0\x9f\x9a\x80\xf0\x9f\x94\xa5 done \xe2\x9c\x85", false,
     {92731, 1273, 11162, 248, 222, 144670, 2814, 25521, 227}},

    {"mixed RTL (ar + he)",
     "Hello \xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7 world "
     "\xd7\xa9\xd7\x9c\xd7\x95\xd7\x9d!", false,
     {9707, 23364, 126860, 124671, 1879, 124756, 123881, 0}},

    {"contractions (f508c01 fix, qwen3 vocab)",
     "React's state isn't done, they're sure we've won.", false,
     {14799, 594, 1584, 4436, 944, 2814, 11, 807, 2299, 2704, 582, 3003,
      2765, 13}},

    {"whitespace edges (tabs/newlines/runs)",
     "Tabs\tand\nnewlines\n\n  mixed   spaces", false,
     {36985, 52477, 198, 931, 7969, 271, 220, 9519, 256, 12621}},

    {"ChatML message (specials as IDs)",
     "<|im_start|>user\nhi<|im_end|>\n", true,
     {151644, 872, 198, 6023, 151645, 198}},

    {"<think> block",
     "<think>\nstep one\n</think>\nThe answer is 4.", true,
     {151667, 198, 9520, 825, 198, 151668, 198, 785, 4226, 374, 220, 19, 13}},

    {"empty-adjacent specials",
     "<|im_start|><|im_end|>", true,
     {151644, 151645}},

    {"full 2-turn ChatML prompt (thinking open)",
     "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
     "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
     "<|im_start|>assistant\n<think>\n", true,
     {151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198,
      151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30, 151645, 198,
      151644, 77091, 198, 151667, 198}},
};

std::string home_path(const char* rel) {
    const char* home = std::getenv("HOME");
    return std::string(home ? home : "") + "/" + rel;
}

void dump_ids(const char* label, const std::vector<int32_t>& ids) {
    std::fprintf(stderr, "    %s: [", label);
    for (size_t i = 0; i < ids.size(); ++i)
        std::fprintf(stderr, "%s%d", i ? ", " : "", ids[i]);
    std::fprintf(stderr, "]\n");
}

}  // namespace

int main() {
    const std::string path = home_path(".seal/models/Qwen3-8B-Q4_K_M.gguf");

    ie::GgufReader g;
    if (const std::string err = g.open(path); !err.empty()) {
        std::fprintf(stderr,
                     "tokenizer_qwen3_golden_test: SKIP (cannot open %s: %s)\n",
                     path.c_str(), err.c_str());
        return 0;
    }

    ie::Tokenizer tok;
    if (const std::string err = tok.load_from_gguf(g); !err.empty()) {
        std::fprintf(stderr, "load_from_gguf failed: %s\n", err.c_str());
        return 1;
    }
    assert(tok.vocab_size() == 151936);
    assert(!tok.add_bos_token());  // qwen2 family: reference adds no BOS

    int fails = 0;
    for (const auto& gc : kGoldens) {
        const std::string text(gc.text);
        const auto got = tok.encode(text, gc.allow_special);
        if (got != gc.ids) {
            ++fails;
            std::fprintf(stderr, "  ENCODE MISMATCH '%s'\n", gc.name);
            dump_ids("got     ", got);
            dump_ids("expected", gc.ids);
        }
        // Lossless round-trip: specials decode literally (skip_special=false).
        const std::string back = tok.decode(got, /*skip_special=*/false);
        if (back != text) {
            ++fails;
            std::fprintf(stderr,
                         "  ROUND-TRIP MISMATCH '%s'\n    got:      '%s'\n"
                         "    expected: '%s'\n",
                         gc.name, back.c_str(), text.c_str());
        }
    }

    if (fails) {
        std::fprintf(stderr, "tokenizer_qwen3_golden_test: %d FAILURE(S)\n", fails);
        return 1;
    }
    std::printf("tokenizer_qwen3_golden_test: all OK (%zu goldens, encode + round-trip)\n",
                kGoldens.size());
    return 0;
}
