// tests/unit/tokenizer_qwen35_golden_test.cpp -- the "qwen35" pre-tokenizer (Qwen3.5 / 3.6 / 3.8, Flash-Next = the Qwen 4
// preview) against the official tokenizer: tests/data/qwen35_pretok_lines.txt tokenised by HF `tokenizers` with
// Qwen/Qwen3.8-Flash-Next's tokenizer.json (add_special_tokens=False) -> tests/data/qwen35_pretok_ids.txt, one comma-separated
// line per input line (the lines decode \n \r \t \\ as ie-tok-dump does). The engine tokenises the same lines from a Flash-Next
// GGUF (vocab + merges + pre = "qwen35") and every line must match id for id. The corpus leans on what the qwen35 regex does
// differently from qwen2: combining marks inside letter runs (decomposed accents, Devanagari matras, Thai marks, Arabic
// harakat, Hebrew niqqud), marks after symbols, zero-width format characters, emoji sequences, CJK punctuation. Known,
// deliberate difference kept OUT of the corpus: HF normalises to NFC first ("o" + U+0304 becomes U+022B before BPE);
// neither llama.cpp nor this engine normalises, so decomposed sequences WITH a precomposed form tokenise differently.
// usage: tokenizer_qwen35_golden_test [flash-next.gguf]   (or IE_QWEN35_GGUF); exit 77 = skipped (no GGUF)
#include "ie/gguf.hpp"
#include "ie/tokenizer.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {
std::string unescape(const std::string& s) {
    std::string o; o.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            const char c = s[i + 1];
            if (c == 'n') { o += '\n'; ++i; continue; }
            if (c == 'r') { o += '\r'; ++i; continue; }
            if (c == 't') { o += '\t'; ++i; continue; }
            if (c == '\\') { o += '\\'; ++i; continue; }
        }
        o += s[i];
    }
    return o;
}
std::vector<std::string> read_lines(const char* rel) {
    std::ifstream f;
    for (const char* pre : {"", "../", "../../"}) { f.open(std::string(pre) + rel); if (f) break; f.clear(); }
    std::vector<std::string> out; std::string line;
    if (!f) return out;
    while (std::getline(f, line)) out.push_back(line);
    return out;
}
std::string join(const std::vector<int32_t>& ids) {
    std::string s;
    for (size_t i = 0; i < ids.size(); ++i) { if (i) s += ','; s += std::to_string(ids[i]); }
    return s;
}
}  // namespace

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : std::getenv("IE_QWEN35_GGUF");
    if (!path || !*path) { std::printf("tokenizer_qwen35_golden_test: SKIP (no Flash-Next GGUF given; pass the path or set IE_QWEN35_GGUF)\n"); return 77; }
    ie::GgufReader g;
    if (auto e = g.open(path); !e.empty()) { std::printf("tokenizer_qwen35_golden_test: SKIP (cannot open %s: %s)\n", path, e.c_str()); return 77; }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::printf("tokenizer: %s\n", e.c_str()); return 1; }
    const auto lines = read_lines("tests/data/qwen35_pretok_lines.txt");
    const auto want = read_lines("tests/data/qwen35_pretok_ids.txt");
    if (lines.empty() || want.size() != lines.size()) { std::printf("corpus / ids missing or mismatched (%zu lines, %zu id lines)\n", lines.size(), want.size()); return 1; }
    int fails = 0;
    if (!tok.pre_warning().empty()) { std::printf("  [FAIL] the loader still warns about the pre-tokenizer: %s\n", std::string(tok.pre_warning()).c_str()); ++fails; }
    for (size_t i = 0; i < lines.size(); ++i) {
        const std::string got = join(tok.encode(unescape(lines[i]), /*allow_special=*/true));
        const bool ok = got == want[i];
        if (!ok) { ++fails; std::printf("  [FAIL] line %zu: %s\n         got  %s\n         want %s\n", i + 1, lines[i].substr(0, 70).c_str(), got.substr(0, 160).c_str(), want[i].substr(0, 160).c_str()); }
    }
    std::printf("tokenizer_qwen35_golden_test: %s (%zu lines, %d mismatches)\n", fails ? "FAILED" : "OK", lines.size(), fails);
    return fails ? 1 : 0;
}
