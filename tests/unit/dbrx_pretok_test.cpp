// tests/unit/dbrx_pretok_test.cpp — host encode-parity test for the dbrx
// (cl100k / gpt2-tiktoken) pre-tokenizer reported by Phi-4 (pre="dbrx").
//
// PURE HOST, NO GPU, NO MODEL WEIGHTS. Compiles the SYCL-free tokenizer +
// gguf_reader + dtype TUs directly (same precedent as tekken_tokenizer_test —
// does NOT link ie_core). It loads ONLY the tokenizer metadata KVs
// (tokens/merges/pre/bos) from a real dbrx GGUF (Phi-4) via mmap — no tensors
// are dequantized, no device is created.
//
// Key finding (plan §Task 3): dbrx pretokenizes IDENTICALLY to llama-bpe — the
// only split delta vs the qwen2 default is digit-runs-≤3 (digits_1to3_) plus
// ignore_merges. It differs only in vocab/merges (cl100k vs llama3 BPE), which
// come from the GGUF. So the test guards the digit-run case (1234567 → 123 456 7,
// ids 4513 10961 22) as the load-bearing assertion, plus prose / code / multibyte.
//
// Golden ids produced on CPU by `llama-tokenize -m phi-4-Q4_K_M.gguf -p <text>
// --ids` (llama.cpp build-vk, phi3 first-class arch). Phi-4 carries NO
// add_bos_token KV and llama-tokenize prepends no BOS (e.g. "Hello" -> [9906]),
// so the golden vectors have NO leading BOS — matching add_bos_token_=false.
//
// Build defines -DNDEBUG (Release); this TU compiles with -UNDEBUG (same as
// tekken_tokenizer_test) so the asserts actually validate.

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

// Curated prose + code + digits + multibyte corpus. add_bos=false for Phi-4,
// so NO vector leads with a BOS id. The digit-run case (1234567 -> 4513 10961 22
// = "123" "456" "7") is the load-bearing assertion that proves digits_1to3_ is
// on; a single-\p{N} (qwen2-default) split would mis-encode it.
const std::vector<Golden>& golden_corpus() {
    static const std::vector<Golden> g = {
        { "Hello world. def f(x): return x*2  # 1234567",
          { 9906, 1917, 13, 711, 282, 2120, 1680, 471, 865, 9, 17, 220, 674, 220, 4513, 10961, 22 } },
        { "Hello, world!",
          { 9906, 11, 1917, 0 } },
        { "The quick brown fox jumps over the lazy dog.",
          { 791, 4062, 14198, 39935, 35308, 927, 279, 16053, 5679, 13 } },
        { "const userName = getUserName(user.id);",
          { 1040, 20446, 284, 636, 19387, 4374, 1801, 1237 } },
        { "XMLHttpRequest and camelCaseIdentifier plus snake_case_var",
          { 10833, 27459, 323, 50252, 4301, 8887, 5636, 26332, 19640, 4715 } },
        { "Result: 42 items, 1234567 bytes, 3.14159 pi",
          { 2122, 25, 220, 2983, 3673, 11, 220, 4513, 10961, 22, 5943, 11, 220, 18, 13, 9335, 2946, 9115 } },
        { "  indented\tcode\n\treturn 0;",
          { 220, 1280, 16243, 44443, 198, 862, 220, 15, 26 } },
        { "if (x == y) { foo(); } else { bar(); }",
          { 333, 320, 87, 624, 379, 8, 314, 15586, 2178, 335, 775, 314, 3703, 2178, 335 } },
        { "path/to/some/file.py",
          { 2398, 33529, 2754, 638, 24849, 7345 } },
        // café au lait — naïve façade (multibyte / accents)
        { "caf\xc3\xa9 au lait \xe2\x80\x94 na\xc3\xafve fa\xc3\xa7" "ade",
          { 936, 59958, 8065, 1208, 275, 2001, 95980, 588, 95972, 1037 } },
    };
    return g;
}

// Candidate dbrx GGUFs on the box (tokenizer-only read — no weights touched).
const char* find_dbrx_gguf() {
    static const char* candidates[] = {
        "/home/weezy/models/phi-4-GGUF/phi-4-Q4_K_M.gguf",
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
    const char* path = find_dbrx_gguf();
    if (!path) {
        // VALIDATION GAP (not a failure): no dbrx vocab on the box. The golden
        // parity gate cannot run. Place phi-4-Q4_K_M.gguf in ~/models/phi-4-GGUF
        // and re-run to exercise parity.
        std::printf("dbrx_pretok_test: SKIP (no dbrx GGUF found; parity ungated)\n");
        return 0;  // CI-safe skip, same precedent as the tekken golden test.
    }

    ie::GgufReader g;
    const std::string err = g.open(path);
    if (!err.empty()) {
        std::printf("dbrx_pretok_test: FAIL (gguf open: %s)\n", err.c_str());
        return 1;
    }

    ie::Tokenizer tok;
    const std::string terr = tok.load_from_gguf(g);
    if (!terr.empty()) {
        std::printf("dbrx_pretok_test: FAIL (tokenizer load: %s)\n", terr.c_str());
        return 1;
    }

    // Sanity: Phi-4 carries no add_bos_token KV → must NOT default-prepend BOS
    // (the dbrx branch deliberately does not join the llama-bpe/tekken BOS
    // default). Golden vectors have no leading BOS, so add_bos_token must be false.
    if (tok.add_bos_token()) {
        std::printf("dbrx_pretok_test: FAIL (expected add_bos_token=false for Phi-4/dbrx)\n");
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
        std::printf("dbrx_pretok_test: FAIL (%d/%zu golden mismatches)\n",
                    failures, golden_corpus().size());
        return 1;
    }

    // ---- Round-trip self-consistency (no special tokens in these strings) ----
    const char* rt[] = {
        "Hello, world!",
        "def getUserName(): return self._name",
        "café 1234567 == naïve",
        "a/b/c\nx\ty",
    };
    for (const char* s : rt) {
        const std::vector<int32_t> ids = tok.encode(s, /*allow_special=*/false);
        const std::string dec = tok.decode(ids, /*skip_special=*/true);
        if (dec != s) {
            std::printf("  ROUND-TRIP MISMATCH: in=[%s] out=[%s]\n", s, dec.c_str());
            ++failures;
        }
    }
    assert(failures == 0);
    if (failures) { std::printf("dbrx_pretok_test: FAIL (round-trip)\n"); return 1; }

    std::printf("dbrx_pretok_test: OK (%zu golden + %zu round-trip; gguf=%s)\n",
                golden_corpus().size(), sizeof(rt) / sizeof(rt[0]), path);
    return 0;
}
