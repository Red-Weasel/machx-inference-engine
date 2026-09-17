// tools/gemma4_tok_test.cpp — host-only encode check for the Gemma SPM tokenizer.
// Loads the GGUF tokenizer (mmap, no GPU) and prints token ids for each arg, to
// diff against `llama-tokenize -m <gguf> -p <str>`.
#include "ie/gguf.hpp"
#include "ie/tokenizer.hpp"
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <gguf> <str> [str...]\n", argv[0]); return 2; }
    ie::GgufReader g;
    if (auto e = g.open(argv[1]); !e.empty()) { std::fprintf(stderr, "gguf: %s\n", e.c_str()); return 1; }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tok: %s\n", e.c_str()); return 1; }
    for (int i = 2; i < argc; ++i) {
        auto ids = tok.encode(argv[i], /*allow_special=*/true);
        std::printf("'%s' ->", argv[i]);
        for (int32_t id : ids) std::printf(" %d", id);
        std::printf("\n");
    }
    return 0;
}
