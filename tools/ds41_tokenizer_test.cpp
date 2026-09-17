// tools/ds41_tokenizer_test.cpp — V4.1 Phase 12 step 1: the tokenizer.json loader vs the Python
// reference (tools/ds41_reference/tokenizer_golden.py), token for token (docs/deepseek41/31).
//   1. encode(text) == the reference's ids on every golden text (docs corpus, stress mix, pp2048)
//   2. decode(ids) == text, byte for byte; decode(encode(text)) == text
//   3. the added tokens (chat roles, bos/eos) resolve to their ids and encode as ONE token each;
//      bos/eos ids and add_bos as configured (the roles are flagged non-special in the JSON)
//   usage: ie-ds41-tokenizer-test <model_dir> <golden_dir>
#include "ie/tokenizer.hpp"

#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const std::string& w, const std::string& d = "") {
    std::printf("%s%s%s\n", ok ? "[ ok ] " : "[FAIL] ", w.c_str(), d.empty() ? "" : ("  (" + d + ")").c_str()); if (!ok) ++g_fail; }
bool read_text(const std::string& p, std::string& out) {
    std::ifstream f(p, std::ios::binary); if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()); return true; }
bool read_ids(const std::string& p, std::vector<int32_t>& out) {
    std::ifstream f(p, std::ios::binary | std::ios::ate); if (!f) return false;
    const auto n = size_t(f.tellg()) / 4; out.resize(n); f.seekg(0);
    f.read(reinterpret_cast<char*>(out.data()), std::streamsize(n * 4)); return true; }
std::string show(const ie::Tokenizer& t, const std::vector<int32_t>& ids, size_t from, size_t to) {
    std::string s;
    for (size_t i = from; i < to && i < ids.size(); ++i) { s += "[" + std::to_string(ids[i]) + ":" + std::string(t.token_str(ids[i])) + "]"; }
    return s; }
}  // namespace

int main(int argc, char** argv) {
    const std::string model = argc > 1 ? argv[1] : std::string(std::getenv("HOME")) + "/models/DeepSeek-V4.1-Flash";
    const std::string gd    = argc > 2 ? argv[2] : "/tmp/ds41_golden";
    ie::Tokenizer tok;
    if (auto e = tok.load_from_hf_json(model + "/tokenizer.json", model + "/tokenizer_config.json"); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    std::printf("tokenizer.json loaded: vocab %u, pre %s, bos %d, eos %d, pad %d, add_bos %d\n", tok.vocab_size(), std::string(tok.pre()).c_str(),
                tok.bos_token_id(), tok.eos_token_id(), tok.pad_token_id(), int(tok.add_bos_token()));
    check(tok.vocab_size() == 129280, "vocab size is the model's 129,280", std::to_string(tok.vocab_size()));
    check(tok.bos_token_id() == 0 && tok.eos_token_id() == 1, "bos = 0 (<｜begin▁of▁sentence｜>), eos = 1 (<｜end▁of▁sentence｜>)", std::to_string(tok.bos_token_id()) + "/" + std::to_string(tok.eos_token_id()));
    check(!tok.add_bos_token(), "add_bos_token is false, as configured");
    for (const char* sp : {"<｜User｜>", "<｜Assistant｜>", "<｜System｜>", "<｜end▁of▁sentence｜>", "<｜begin▁of▁sentence｜>"}) {
        const int32_t id = tok.find_token(sp);
        const auto one = tok.encode(sp, true);
        check(id >= 0 && one.size() == 1 && one[0] == id, std::string("added token ") + sp + " resolves and encodes as ONE token", "id " + std::to_string(id) + ", encoded to " + std::to_string(one.size()) + (id >= 0 && tok.is_special(id) ? ", special" : ", not flagged special"));
    }

    std::printf("\n=== parity with the Python reference (HF tokenizers over the checkpoint's tokenizer.json) ===\n");
    for (const char* name : {"docs", "stress", "pp2048"}) {
        std::string text; std::vector<int32_t> ids;
        if (!read_text(gd + "/tok_" + name + ".txt", text) || !read_ids(gd + "/tok_" + name + ".i32", ids)) { std::printf("       (no golden %s -- skipped)\n", name); continue; }
        const auto enc = tok.encode(text, /*allow_special=*/true);
        size_t first = 0; while (first < enc.size() && first < ids.size() && enc[first] == ids[first]) ++first;
        const bool same = enc == ids;
        std::string detail = std::to_string(enc.size()) + " vs " + std::to_string(ids.size()) + " ids";
        if (!same) detail += "; first difference at " + std::to_string(first) + ": engine " + show(tok, enc, first > 2 ? first - 2 : 0, first + 4) + " | reference " + show(tok, ids, first > 2 ? first - 2 : 0, first + 4);
        check(same, std::string("encode(") + name + ") == the reference's ids, token for token", detail);
        const std::string dec = tok.decode(ids, /*skip_special=*/false);
        check(dec == text, std::string("decode(reference ids of ") + name + ") == the text, byte for byte", dec == text ? std::to_string(text.size()) + " bytes" : "differs (" + std::to_string(dec.size()) + " vs " + std::to_string(text.size()) + " bytes)");
        const std::string rt = tok.decode(enc, false);
        check(rt == text, std::string("decode(encode(") + name + ")) == the text");
    }
    std::printf("\n%s\n", g_fail ? "TOKENIZER TEST: FAILURE(S)" : "TOKENIZER TEST: PASS");
    return g_fail ? 1 : 0;
}
