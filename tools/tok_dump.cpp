// tools/tok_dump.cpp — ie-tok-dump: the engine tokenizer's ids for each line of a file (escapes \n, \r, \t, \\ decoded),
// one comma-separated line out per line in, for byte-exact comparison against llama.cpp's llama-tokenize.
//   usage: ie-tok-dump <model.gguf> <lines.txt> [--bos]      (no BOS by default, like llama-tokenize --no-bos)
#include "ie/gguf.hpp"
#include "ie/tokenizer.hpp"

#include <cstdio>
#include <fstream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: ie-tok-dump <model.gguf> <lines.txt> [--bos]\n"); return 2; }
    const bool bos = argc > 3 && std::string(argv[3]) == "--bos";
    ie::GgufReader g;
    if (auto e = g.open(argv[1]); !e.empty()) { std::fprintf(stderr, "gguf: %s\n", e.c_str()); return 1; }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1; }
    std::ifstream in(argv[2]);
    std::string line;
    while (std::getline(in, line)) {
        std::string s;
        for (size_t i = 0; i < line.size(); ++i) {
            if (line[i] == '\\' && i + 1 < line.size()) {
                const char c = line[++i];
                s += c == 'n' ? '\n' : c == 'r' ? '\r' : c == 't' ? '\t' : c;
            } else s += line[i];
        }
        auto ids = tok.encode(s, /*allow_special=*/true);
        size_t from = (!bos && tok.add_bos_token() && !ids.empty() && ids[0] == tok.bos_token_id()) ? 1 : 0;
        for (size_t i = from; i < ids.size(); ++i) std::printf("%s%d", i > from ? "," : "", ids[i]);
        std::printf("\n");
    }
    return 0;
}
