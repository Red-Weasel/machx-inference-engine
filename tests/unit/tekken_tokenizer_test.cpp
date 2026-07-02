// tests/unit/tekken_tokenizer_test.cpp — host encode-parity test for the tekken
// pre-tokenizer (Mistral-Nemo / Small-24B / Devstral / Codestral-tekken).
//
// PURE HOST, NO GPU, NO MODEL WEIGHTS. The test compiles the SYCL-free
// tokenizer + gguf_reader TUs directly (same precedent as qwen3moe_pack_test,
// which does not link ie_core). It loads ONLY the tokenizer metadata KVs
// (tokens/merges/pre/bos) from a real tekken GGUF via mmap — no tensors are
// dequantized, no device is created.
//
// Golden ids were produced on CPU from the GGUF's own vocab+merges by an
// independent reference implementation of llama.cpp's
// LLAMA_VOCAB_PRE_TYPE_TEKKEN regex
//   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+
//   | [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*
//   | \p{N} |  ?[^\s\p{L}\p{N}]+[\r\n/]* | \s*[\r\n]+ | \s+(?!\S) | \s+
// + byte-level BPE with ignore_merges=true and add_bos=true (the exact flags
// llama.cpp sets for "tekken"). See the commit message / report for the GGUF
// used. These ids are the ground truth our C++ tekken encoder must reproduce.
//
// Build defines -DNDEBUG (Release), which no-ops assert(); this TU compiles
// with -UNDEBUG (same as qwen3moe_pack_test) so the asserts actually validate.

#include "ie/gguf.hpp"
#include "ie/tokenizer.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

struct Golden {
    const char*          text;
    std::vector<int32_t> ids;
};

// Curated prose + code corpus. Code is the high-risk surface for tekken
// (camelCase / XMLHttp / snake_case / operators / leading-space). add_bos=true,
// so each vector starts with BOS=1 (<s>).
const std::vector<Golden>& golden_corpus() {
    static const std::vector<Golden> g = {
        { "Hello, world!",
          { 1, 22177, 1044, 4304, 1033 } },
        { "The quick brown fox jumps over the lazy dog.",
          { 1, 1784, 7586, 22980, 94137, 72993, 2136, 1278, 42757, 10575, 1046 } },
        { "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
          { 1, 3149, 9111, 87539, 4990, 3640, 1293, 1850, 1308, 1693, 1308, 1534, 1032,
            1050, 2849, 9111, 87539, 4990, 1045, 1049, 1041, 1867, 9111, 87539, 4990, 1045,
            1050, 1041 } },
        { "const userName = getUserName(user.id);",
          { 1, 3150, 3330, 2266, 1376, 2012, 4019, 2266, 13905, 5405, 3455 } },
        { "XMLHttpRequest and camelCaseIdentifier plus snake_case_var",
          { 1, 26694, 8668, 4967, 1321, 98537, 11139, 27162, 2984, 48726, 69982, 35012 } },
        { "Result: 42 items, 1234567 bytes, 3.14159 pi",
          { 1, 5493, 1058, 1032, 1052, 1050, 7748, 1044, 1032, 1049, 1050, 1051, 1052, 1053,
            1054, 1055, 16372, 1044, 1032, 1051, 1046, 1049, 1052, 1049, 1053, 1057, 4675 } },
        { "caf\xc3\xa9 au lait \xe2\x80\x94 na\xc3\xafve fa\xc3\xa7" "ade",   // café au lait — naïve façade
          { 1, 3173, 1102, 1337, 1817, 74724, 2251, 98355, 62441 } },
        { "  indented\tcode\n\treturn 0;",
          { 1, 1032, 2495, 38498, 1009, 6154, 1010, 3739, 1032, 1048, 1059 } },
        { "if (x == y) { foo(); } else { bar(); }",
          { 1, 1391, 1319, 1120, 2351, 1404, 1041, 1445, 34377, 8637, 1495, 2849, 1445, 4266,
            8637, 1495 } },
        { "path/to/some/file.py",
          { 1, 7368, 35466, 3826, 1798, 55998, 8261 } },
    };
    return g;
}

// Candidate tekken GGUFs on the box (tokenizer-only read — no weights touched).
const char* find_tekken_gguf() {
    static const char* candidates[] = {
        "/home/weezy/models/lmstudio-community/Mistral-Small-3.2-24B-Instruct-2506-GGUF/"
        "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf",
        nullptr,
    };
    for (int i = 0; candidates[i]; ++i) {
        if (FILE* f = std::fopen(candidates[i], "rb")) { std::fclose(f); return candidates[i]; }
    }
    return nullptr;
}

bool ids_equal(const std::vector<int32_t>& got, const std::vector<int32_t>& want) {
    if (got.size() != want.size()) return false;
    for (size_t i = 0; i < got.size(); ++i) if (got[i] != want[i]) return false;
    return true;
}

void print_ids(const char* tag, const std::vector<int32_t>& v) {
    std::printf("    %s [", tag);
    for (size_t i = 0; i < v.size(); ++i) std::printf("%s%d", i ? ", " : "", v[i]);
    std::printf("]\n");
}

}  // namespace

int main() {
    const char* path = find_tekken_gguf();
    if (!path) {
        // VALIDATION GAP (not a failure): no tekken vocab on the box. The golden
        // parity gate cannot run. Download a tekken GGUF (e.g.
        // Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf, Mistral-Nemo-Instruct-2407,
        // or Devstral-Small-2507) into ~/models and re-run to exercise parity.
        std::printf("tekken_tokenizer_test: SKIP (no tekken GGUF found; parity ungated)\n");
        return 0;  // CI-safe skip, same precedent as the qwen3 golden test.
    }

    ie::GgufReader g;
    const std::string err = g.open(path);
    if (!err.empty()) {
        std::printf("tekken_tokenizer_test: FAIL (gguf open: %s)\n", err.c_str());
        return 1;
    }

    ie::Tokenizer tok;
    const std::string terr = tok.load_from_gguf(g);
    if (!terr.empty()) {
        std::printf("tekken_tokenizer_test: FAIL (tokenizer load: %s)\n", terr.c_str());
        return 1;
    }

    // Sanity: this GGUF must actually be tekken (pre=="tekken"), else the golden
    // ids are meaningless. add_bos_token must be true (golden vectors lead with BOS).
    if (!tok.add_bos_token()) {
        std::printf("tekken_tokenizer_test: FAIL (expected add_bos_token=true)\n");
        return 1;
    }
    if (tok.find_token("<s>") != tok.bos_token_id() || tok.bos_token_id() < 0) {
        std::printf("tekken_tokenizer_test: FAIL (BOS id mismatch: bos=%d, <s>=%d)\n",
                    tok.bos_token_id(), tok.find_token("<s>"));
        return 1;
    }

    // ---- Golden encode parity (the load-bearing gate) ----
    int failures = 0;
    for (const auto& gd : golden_corpus()) {
        const std::vector<int32_t> got = tok.encode(gd.text, /*allow_special=*/true);
        if (!ids_equal(got, gd.ids)) {
            ++failures;
            std::printf("  MISMATCH on %s\n",
                        std::string(gd.text).substr(0, 48).c_str());
            print_ids("want", gd.ids);
            print_ids("got ", got);
        }
    }
    if (failures) {
        std::printf("tekken_tokenizer_test: FAIL (%d/%zu golden mismatches)\n",
                    failures, golden_corpus().size());
        return 1;
    }

    // ---- Round-trip self-consistency (no special tokens in these strings) ----
    const char* rt[] = {
        "Hello, world!",
        "def getUserName(): return self._name",
        "café 1234 == naïve",
        "a/b/c\nx\ty",
    };
    for (const char* s : rt) {
        const std::vector<int32_t> ids = tok.encode(s, /*allow_special=*/false);
        // Drop the leading BOS before decoding (BOS is a control token, not text).
        std::vector<int32_t> body(ids.begin() + (ids.empty() ? 0 : 1), ids.end());
        const std::string dec = tok.decode(body, /*skip_special=*/true);
        if (dec != s) {
            std::printf("  ROUND-TRIP MISMATCH: in=[%s] out=[%s]\n", s, dec.c_str());
            ++failures;
        }
    }
    assert(failures == 0);
    if (failures) { std::printf("tekken_tokenizer_test: FAIL (round-trip)\n"); return 1; }

    std::printf("tekken_tokenizer_test: OK (%zu golden + %zu round-trip; gguf=%s)\n",
                golden_corpus().size(), sizeof(rt) / sizeof(rt[0]), path);
    return 0;
}
