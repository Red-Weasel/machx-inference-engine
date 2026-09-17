// tools/ds41_text_ids.cpp — ie-ds41-text-ids: tokenise text files with the engine's V4.1 tokenizer and write the
// first N ids as int32 -- the pp_ids_2048 / pp_ids_512 / corpus_ids_65536 inputs of the ds41 tests (the originals
// lived in a /tmp scratchpad and were lost to the 2026-09-13 reboot; this makes them reproducible).
//   usage: ie-ds41-text-ids <model> <out.i32> <N> <text file>...   (the files are concatenated in order, "\n\n" between)
#include "ie/tokenizer.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
int main(int argc, char** argv) {
    if (argc < 5) { std::fprintf(stderr, "usage: ie-ds41-text-ids <model> <out.i32> <N> <text file>...\n"); return 2; }
    const std::string model = argv[1], out = argv[2]; const size_t n = size_t(std::atol(argv[3]));
    std::string text;
    for (int i = 4; i < argc; ++i) { std::ifstream f(argv[i]); if (!f) { std::fprintf(stderr, "cannot read %s\n", argv[i]); return 1; } std::stringstream ss; ss << f.rdbuf(); if (!text.empty()) text += "\n\n"; text += ss.str(); }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_hf_json(model + "/tokenizer.json", model + "/tokenizer_config.json"); !e.empty()) { std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1; }
    const auto ids = tok.encode(text, /*allow_special=*/false);
    if (ids.size() < n) { std::fprintf(stderr, "only %zu tokens in the text, %zu wanted\n", ids.size(), n); return 1; }
    std::vector<int32_t> v(ids.begin(), ids.begin() + std::ptrdiff_t(n));
    std::ofstream o(out, std::ios::binary); o.write(reinterpret_cast<const char*>(v.data()), std::streamsize(n * 4));
    std::printf("%s: %zu of %zu tokens from %d file(s), %zu bytes of text\n", out.c_str(), n, ids.size(), argc - 4, text.size());
    return 0;
}
