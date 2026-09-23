// tools/mimo26_tokenize.cpp — the engine's tokenizer.json path on a MiMo-V2.6 checkpoint, printed in
// llama-tokenize's `--ids` shape so the P1 gate can diff the two line by line
// (docs/mimo26/00_PORT_PLAN.md, P1 (c)). One input line per output line; special tokens parsed.
//   ie-mimo26-tokenize <model_dir> < strings.txt
#include "ie/tokenizer.hpp"

#include <cstdio>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s <model_dir> < strings.txt\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    ie::Tokenizer tok;
    if (auto e = tok.load_from_hf_json(dir + "/tokenizer.json", dir + "/tokenizer_config.json"); !e.empty()) {
        std::fprintf(stderr, "tokenizer: %s\n", e.c_str());
        return 1;
    }
    std::fprintf(stderr, "vocab %u pre '%s'\n", tok.vocab_size(), std::string(tok.pre()).c_str());
    std::string line;
    while (std::getline(std::cin, line)) {
        // the gate strings carry newlines as the two characters '\' 'n'
        std::string s;
        for (size_t i = 0; i < line.size(); ++i) {
            if (line[i] == '\\' && i + 1 < line.size() && line[i + 1] == 'n') { s += '\n'; ++i; }
            else if (line[i] == '\\' && i + 1 < line.size() && line[i + 1] == 't') { s += '\t'; ++i; }
            else s += line[i];
        }
        const auto ids = tok.encode(s, /*allow_special=*/true);
        std::printf("[");
        for (size_t i = 0; i < ids.size(); ++i) std::printf("%s%d", i ? ", " : "", ids[i]);
        std::printf("]\n");
    }
    return 0;
}
