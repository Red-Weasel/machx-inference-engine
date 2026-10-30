// tokenizer_bos_literal_test (2026-10-28): a GGUF that sets add_bos_token with bos = <|im_start|> (Nanbeige4.3-33B: its
// HF post-processor opens every plain encode with it) must NOT get a second BOS when the text already begins with the
// BOS literal -- a ChatML render starts with <|im_start|>, and apply_chat_template / llama.cpp add none. The engine
// rendered a doubled <|im_start|> before Tokenizer::encode learned llama.cpp's rule.
// CPU only (the tokenizer reads the GGUF header). Skips (77) without the file: IE_BOS_TEST_GGUF names another.
#include "ie/gguf.hpp"
#include "ie/tokenizer.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
    const char* env = std::getenv("IE_BOS_TEST_GGUF");
    const char* home = std::getenv("HOME");
    const std::string path = env && *env ? env
        : std::string(home ? home : "") + "/models/Nanbeige4.3-33B-abliterated-GGUF/Nanbeige4.3-33B-abliterated-Q6_K.gguf";
    ie::GgufReader g;
    if (auto err = g.open(path); !err.empty()) { std::fprintf(stderr, "SKIP: %s (%s)\n", path.c_str(), err.c_str()); return 77; }
    ie::Tokenizer tok;
    if (auto err = tok.load_from_gguf(g); !err.empty()) { std::fprintf(stderr, "FAIL: tokenizer load: %s\n", err.c_str()); return 1; }
    const int32_t bos = tok.bos_token_id();
    const std::string_view bos_str = tok.token_str(bos);
    std::printf("bos id %d = '%.*s'\n", bos, int(bos_str.size()), bos_str.data());
    if (bos_str != "<|im_start|>") { std::fprintf(stderr, "SKIP: this GGUF's bos is not <|im_start|>; the test wants Nanbeige4.3\n"); return 77; }
    int fails = 0;
    auto count = [&](const std::vector<int32_t>& ids, int32_t id) { int n = 0; for (int32_t t : ids) if (t == id) ++n; return n; };

    // 1. a plain text still gets its BOS (add_bos_token = true)
    const auto plain = tok.encode("Hello world");
    if (plain.empty() || plain[0] != bos || count(plain, bos) != 1) { std::fprintf(stderr, "FAIL: plain text did not get exactly one leading BOS\n"); ++fails; }

    // 2. a ChatML render that begins with <|im_start|> keeps that one and gets no second
    const std::string q = "What is the capital of France? Answer in one sentence.";
    const std::string on  = "<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n<think>\n";
    const std::string off = "<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    const auto ids_on = tok.encode(on), ids_off = tok.encode(off);
    if (ids_on.size() < 2 || ids_on[0] != bos || ids_on[1] == bos) { std::fprintf(stderr, "FAIL: the ON render does not start with exactly one <|im_start|>\n"); ++fails; }
    if (ids_off.size() < 2 || ids_off[0] != bos || ids_off[1] == bos) { std::fprintf(stderr, "FAIL: the OFF render does not start with exactly one <|im_start|>\n"); ++fails; }
    // llama.cpp (-no-cnv, BOS off) tokenizes the same renders to 24 / 29 tokens (results/nanbeige-2026-10-28)
    std::printf("tokens: on %zu (llama.cpp 24), off %zu (llama.cpp 29)\n", ids_on.size(), ids_off.size());
    if (ids_on.size() != 24 || ids_off.size() != 29) { std::fprintf(stderr, "FAIL: token counts differ from llama.cpp's\n"); ++fails; }

    // 3. the rule does not fire on a text that merely contains the literal later
    const auto mid = tok.encode("x <|im_start|>user");
    if (mid.size() < 2 || mid[0] != bos || count(mid, bos) != 2) { std::fprintf(stderr, "FAIL: a mid-text literal changed the BOS rule\n"); ++fails; }

    std::printf(fails ? "FAILED (%d)\n" : "OK\n", fails);
    return fails ? 1 : 0;
}
