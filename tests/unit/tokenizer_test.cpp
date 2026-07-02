// tests/unit/tokenizer_test.cpp — Phase 7 tokenizer + sampler.
//
// Phase 7 gate: round-trip a corpus, match HF for canonical tokenizations,
// sampler picks reproducibly. Without HF on this box, "match HF" is reduced
// to "match the known reference IDs published in research/04 §6.1 for
// `Hello, world` → [9707, 11, 1879]". Round-trip is the load-bearing
// correctness check.

#include "ie/gguf.hpp"
#include "ie/ops.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr const char* G = "\033[32m";
constexpr const char* R = "\033[31m";
constexpr const char* Y = "\033[33m";
constexpr const char* Z = "\033[0m";

bool report(const char* name, bool ok, const char* extra = "") {
    std::printf("  %s%-46s%s %s%s\n", ok ? G : R, name, Z, ok ? "OK" : "FAIL", extra);
    return ok;
}

sycl::queue make_queue() {
    sycl::device dev;
    bool found = false;
    for (const auto& d : sycl::device::get_devices()) {
        if (d.is_gpu() && d.get_info<sycl::info::device::name>().find("0xe223") != std::string::npos) {
            dev = d; found = true; break;
        }
    }
    if (!found) for (const auto& d : sycl::device::get_devices()) if (d.is_gpu()) { dev = d; found = true; break; }
    if (!found) { std::fputs("no GPU\n", stderr); std::exit(2); }
    return sycl::queue(dev, sycl::property::queue::enable_profiling{});
}

// ============================================================
//                     Tokenizer tests
// ============================================================

bool test_tok_load(ie::Tokenizer& tok, ie::GgufReader& g) {
    auto err = tok.load_from_gguf(g);
    if (!err.empty()) {
        char extra[160];
        std::snprintf(extra, sizeof(extra), " (%s)", err.c_str());
        return report("Tokenizer::load_from_gguf", false, extra);
    }
    char extra[160];
    std::snprintf(extra, sizeof(extra),
                  " vocab=%u bos=%d eos=%d pad=%d add_bos=%d",
                  tok.vocab_size(), tok.bos_token_id(), tok.eos_token_id(),
                  tok.pad_token_id(), int(tok.add_bos_token()));
    return report("Tokenizer::load_from_gguf", tok.vocab_size() > 100000, extra);
}

bool test_special_tokens(const ie::Tokenizer& tok) {
    // Per research/04 §4.1, IDs:
    //   248044 = <|endoftext|>
    //   248045 = <|im_start|>
    //   248046 = <|im_end|>
    bool ok = true;
    char extra[256] = {};
    auto check = [&](const char* s, int32_t expected) {
        const int32_t id = tok.find_token(s);
        if (id != expected) {
            ok = false;
            std::snprintf(extra, sizeof(extra), " (%s expected=%d got=%d)", s, expected, id);
            return false;
        }
        if (!tok.is_special(id)) {
            ok = false;
            std::snprintf(extra, sizeof(extra), " (%s id=%d not flagged special)", s, id);
            return false;
        }
        return true;
    };
    check("<|endoftext|>", 248044);
    if (ok) check("<|im_start|>", 248045);
    if (ok) check("<|im_end|>",   248046);
    return report("special token IDs match research/04", ok, extra);
}

// Encode then decode and compare. Skip-special so the round-trip is on raw text.
bool test_round_trip(const ie::Tokenizer& tok, const std::string& text, const char* label) {
    const auto ids = tok.encode(text, /*allow_special=*/true);
    const std::string back = tok.decode(ids, /*skip_special=*/false);
    char extra[256];
    if (back == text) {
        std::snprintf(extra, sizeof(extra), " (%zu tokens for %zu bytes)", ids.size(), text.size());
        return report(label, true, extra);
    } else {
        std::snprintf(extra, sizeof(extra), " input='%s' -> %zu ids -> '%s'",
                      text.size() > 30 ? "..." : text.c_str(),
                      ids.size(), back.c_str());
        return report(label, false, extra);
    }
}

bool test_known_ids(const ie::Tokenizer& tok) {
    // Encoded "Hello, world" should produce 3 pretokens decoding to
    // {"Hello", ",", " world"}. Research/04 §6.1 claimed the IDs are
    // [9707, 11, 1879] (the Qwen2.5 IDs), but Qwen3.6's vocab reassigns
    // those positions — the actually-correct Qwen3.6 IDs are [9419, 11, 1814]
    // per the GGUF in `models/lmstudio-community/Qwen3.6-35B-A3B-GGUF`.
    // The semantic correctness check is "do the 3 tokens decode to the
    // canonical 3 strings", which is what HF cares about.
    const auto ids = tok.encode("Hello, world", /*allow_special=*/false);
    if (ids.size() != 3) {
        char extra[80]; std::snprintf(extra, sizeof(extra), " got %zu tokens, expected 3", ids.size());
        return report("encode('Hello, world') gives 3 tokens", false, extra);
    }
    const std::string s0 = tok.decode(std::vector<int32_t>{ids[0]}, false);
    const std::string s1 = tok.decode(std::vector<int32_t>{ids[1]}, false);
    const std::string s2 = tok.decode(std::vector<int32_t>{ids[2]}, false);
    char extra[256];
    std::snprintf(extra, sizeof(extra), " ids=[%d,%d,%d] decode=['%s','%s','%s']",
                  ids[0], ids[1], ids[2], s0.c_str(), s1.c_str(), s2.c_str());
    const bool ok = (s0 == "Hello" && s1 == "," && s2 == " world");
    return report("encode('Hello, world') = [Hello, , world]", ok, extra);
}

// Golden set vs the reference tokenizer. Reference IDs captured by running
//   /home/weezy/llama.cpp-vulkan/llama-b8902/llama-tokenize \
//     -m /home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf -p "<text>"
// (b8902, 2026-06-10). Focus: the qwen2 contraction alternative
// `(?i:'s|'t|'re|'ve|'m|'ll|'d)` and the alt-2 optional non-letter prefix
// `[^\r\n\p{L}\p{N}]?\p{L}+` — both dropped by the pre-2026-06-10 splitter
// (Seal dogfooding found `React's` -> `React`+`'`+`s`, a sequence the model
// never saw in training).
bool test_golden_vs_reference(const ie::Tokenizer& tok) {
    struct Golden { const char* text; std::vector<int32_t> ids; };
    const std::vector<Golden> cases = {
        {"React's state",    {14370, 579, 1528}},            // React|'s| state
        {"don't",            {14572, 914}},                   // don|'t
        {"it's",             {275, 579}},                     // it|'s
        {"I'll",             {40, 3172}},                     // I|'ll
        {"they're",          {19458, 2224}},                  // they|'re
        {"we've",            {868, 2908}},                    // we|'ve
        {"I'm",              {40, 2688}},                     // I|'m
        {"you'd",            {9053, 4035}},                   // you|'d
        {"IT'S",             {922, 12887}},                   // IT|'S (case-insensitive)
        {"O'Brien",          {46, 59387}},                    // O|'Brien (alt-2 ' prefix)
        {"the users' files", {1719, 3719, 6, 3425}},          // the| users|'| files
        {"console.log",      {5186, 1607}},                   // console|.log (alt-2 . prefix)
        {"src/app.js",       {3431, 10325, 2764}},            // src|/app|.js
        // No word boundary in the contraction alternative: 'sup -> 's|up.
        {"'sup 'twas 'Tis the season",
         {579, 446, 359, 15103, 299, 359, 51, 284, 279, 3098}},
        {"REACT'S STATE ISN'T DONE",
         {762, 6609, 12887, 21726, 3314, 45, 16813, 52847}},
        // Contraction-heavy sentence exercising every suffix + punctuation.
        {"She said it's done, and they're sure we've won — I'll bet you'd "
         "agree, don't you?",
         {7691, 1018, 424, 579, 2725, 11, 321, 781, 2224, 2617, 567, 2908,
          2677, 1892, 353, 3172, 1229, 488, 4035, 7268, 11, 1459, 914, 488,
          30}},
    };
    bool all_ok = true;
    int pass = 0;
    char extra[160] = {};
    for (const auto& gc : cases) {
        const auto got = tok.encode(gc.text, /*allow_special=*/false);
        if (got == gc.ids) { ++pass; continue; }
        all_ok = false;
        std::fprintf(stderr, "  golden mismatch '%s'\n    got:      [", gc.text);
        for (auto id : got) std::fprintf(stderr, "%d,", id);
        std::fprintf(stderr, "]\n    expected: [");
        for (auto id : gc.ids) std::fprintf(stderr, "%d,", id);
        std::fprintf(stderr, "]\n");
    }
    std::snprintf(extra, sizeof(extra), " %d/%zu vs llama-tokenize", pass, cases.size());
    return report("golden encode == reference tokenizer", all_ok, extra);
}

bool test_chatml_template(const ie::Tokenizer& tok) {
    std::vector<ie::ChatTurn> turns = {
        {"system", "You are a helpful assistant."},
        {"user",   "What is 2+2?"},
    };
    const std::string p = ie::build_chatml_prompt(turns, true, true);
    // Golden string: enable_thinking=true opens a bare <think> block.
    const std::string expected =
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n";
    const bool ok = (p == expected);
    if (!ok) {
        std::fprintf(stderr, "  chatml_thinking=true golden mismatch:\n"
                             "    got:      '%s'\n"
                             "    expected: '%s'\n", p.c_str(), expected.c_str());
    }

    // Encoding should pick up the special tokens as single IDs, not raw bytes.
    const auto ids = tok.encode(p, /*allow_special=*/true);
    bool has_specials = false;
    for (auto id : ids) if (tok.is_special(id)) { has_specials = true; break; }

    char extra[160];
    std::snprintf(extra, sizeof(extra), " prompt=%zu bytes, %zu tokens, has_specials=%d",
                  p.size(), ids.size(), int(has_specials));
    return report("ChatML thinking=true golden + encode specials", ok && has_specials, extra);
}

bool test_chatml_no_thinking(const ie::Tokenizer& tok) {
    std::vector<ie::ChatTurn> turns = {
        {"system", "You are a concise assistant."},
        {"user",   "Reply with exactly the word: ping"},
    };
    // enable_thinking=false must emit the empty-think-block suppression prefix.
    const std::string p = ie::build_chatml_prompt(turns, true, false);
    const std::string expected =
        "<|im_start|>system\nYou are a concise assistant.<|im_end|>\n"
        "<|im_start|>user\nReply with exactly the word: ping<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n";
    const bool ok = (p == expected);
    if (!ok) {
        std::fprintf(stderr, "  chatml_thinking=false golden mismatch:\n"
                             "    got:      '%s'\n"
                             "    expected: '%s'\n", p.c_str(), expected.c_str());
    }
    // The empty think block must NOT appear when thinking is enabled.
    const std::string p_think = ie::build_chatml_prompt(turns, true, true);
    const bool no_bleed = (p_think.find("</think>") == std::string::npos);

    char extra[160];
    std::snprintf(extra, sizeof(extra), " prompt=%zu bytes, no_bleed=%d", p.size(), int(no_bleed));
    return report("ChatML thinking=false empty-think golden", ok && no_bleed, extra);
}

// Shared golden fragment: the canonical Qwen3 tools preamble emitted by
// build_chatml_prompt for kToolsJson (JSON tool-call convention — chosen so
// OpenAI-compat clients recover `{"name":...,"arguments":...}` from text;
// see docs/seal-integration.md gap 1).
constexpr const char* kToolsJson =
    R"([{"type":"function","function":{"name":"write_file","description":"Write a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}])";

const std::string kToolsPreamble =
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    R"({"type":"function","function":{"name":"write_file","description":"Write a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}})" "\n"
    "</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>";

bool test_chatml_tools_golden() {
    std::vector<ie::ChatTurn> turns = {
        {"system", "You are a helpful assistant."},
        {"user",   "Create hello.txt"},
    };
    const std::string p = ie::build_chatml_prompt(turns, true, false, kToolsJson);
    const std::string expected =
        "<|im_start|>system\n"
        "You are a helpful assistant.\n\n" + kToolsPreamble + "<|im_end|>\n"
        "<|im_start|>user\nCreate hello.txt<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n";
    bool ok = (p == expected);
    if (!ok) {
        std::fprintf(stderr, "  chatml tools golden mismatch:\n    got:      '%s'\n"
                             "    expected: '%s'\n", p.c_str(), expected.c_str());
    }
    // Empty tools_json must stay byte-identical to the historical 3-arg call.
    const std::string p3 = ie::build_chatml_prompt(turns, true, false);
    const std::string p4 = ie::build_chatml_prompt(turns, true, false, {});
    if (p3 != p4) { ok = false; std::fputs("  empty-tools not byte-identical\n", stderr); }
    return report("ChatML tools preamble golden", ok);
}

bool test_chatml_tool_sequence_golden() {
    // 5-turn agent trace: user -> assistant(tool_call) -> tool -> assistant -> user.
    // No system turn in input: the tools preamble must create one.
    std::vector<ie::ChatTurn> turns = {
        {"user",      "Create hello.txt with content hi"},
        {"assistant", "<tool_call>\n{\"name\": \"write_file\", \"arguments\": "
                      "{\"path\":\"hello.txt\",\"content\":\"hi\"}}\n</tool_call>"},
        {"tool",      "ok"},
        {"assistant", "Created hello.txt."},
        {"user",      "thanks"},
    };
    const std::string p = ie::build_chatml_prompt(turns, true, false, kToolsJson);
    const std::string expected =
        "<|im_start|>system\n" + kToolsPreamble + "<|im_end|>\n"
        "<|im_start|>user\nCreate hello.txt with content hi<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<tool_call>\n{\"name\": \"write_file\", \"arguments\": "
        "{\"path\":\"hello.txt\",\"content\":\"hi\"}}\n</tool_call><|im_end|>\n"
        "<|im_start|>user\n<tool_response>\nok\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\nCreated hello.txt.<|im_end|>\n"
        "<|im_start|>user\nthanks<|im_end|>\n"
        "<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n";
    bool ok = (p == expected);
    if (!ok) {
        std::fprintf(stderr, "  chatml tool-sequence golden mismatch:\n    got:      '%s'\n"
                             "    expected: '%s'\n", p.c_str(), expected.c_str());
    }
    // Consecutive tool turns merge into ONE user turn (Qwen3 template rule).
    std::vector<ie::ChatTurn> two_tools = {
        {"user", "go"}, {"tool", "r1"}, {"tool", "r2"},
    };
    const std::string p2 = ie::build_chatml_prompt(two_tools, false, false, {});
    const std::string expected2 =
        "<|im_start|>user\ngo<|im_end|>\n"
        "<|im_start|>user\n<tool_response>\nr1\n</tool_response>\n"
        "<tool_response>\nr2\n</tool_response><|im_end|>\n";
    if (p2 != expected2) {
        ok = false;
        std::fprintf(stderr, "  consecutive-tool merge mismatch:\n    got:      '%s'\n"
                             "    expected: '%s'\n", p2.c_str(), expected2.c_str());
    }
    return report("ChatML 5-turn tool sequence golden", ok);
}

// ============================================================
//                       Sampler tests
// ============================================================

bool test_argmax(sycl::queue& q) {
    constexpr uint32_t V = 1024;
    std::vector<sycl::half> hl(V);
    std::mt19937 rng(0x52L);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    for (auto& v : hl) v = sycl::half(dist(rng));
    const uint32_t expected = 313;
    hl[expected] = sycl::half(10.0f);  // unambiguous max

    auto* dl = sycl::malloc_device<sycl::half>(V, q);
    auto* dx = sycl::malloc_device<int32_t>(1, q);
    q.memcpy(dl, hl.data(), V * sizeof(sycl::half)).wait();
    ie::sample_argmax(q, dl, dx, V).wait();
    int32_t got;
    q.memcpy(&got, dx, sizeof(int32_t)).wait();
    sycl::free(dl, q); sycl::free(dx, q);
    char extra[80];
    std::snprintf(extra, sizeof(extra), " expected=%u got=%d", expected, got);
    return report("sample_argmax", got == int32_t(expected), extra);
}

bool test_sample_temp_zero(sycl::queue& q) {
    // temperature <= 0 should fall back to argmax.
    constexpr uint32_t V = 256;
    std::vector<sycl::half> hl(V);
    for (uint32_t i = 0; i < V; ++i) hl[i] = sycl::half(float(i) * 0.01f);
    hl[42] = sycl::half(99.0f);
    auto* dl = sycl::malloc_device<sycl::half>(V, q);
    auto* dx = sycl::malloc_device<int32_t>(1, q);
    q.memcpy(dl, hl.data(), V * sizeof(sycl::half)).wait();
    ie::sample_softmax_topk_topp(q, dl, dx, V, /*temp=*/0.f, /*top_k=*/0, /*top_p=*/1.f, /*min_p=*/0.f, /*rng=*/1ull).wait();
    int32_t got;
    q.memcpy(&got, dx, sizeof(int32_t)).wait();
    sycl::free(dl, q); sycl::free(dx, q);
    char extra[80];
    std::snprintf(extra, sizeof(extra), " expected=42 got=%d", got);
    return report("sample temp=0 falls back to argmax", got == 42, extra);
}

bool test_sample_topk1(sycl::queue& q) {
    // top_k=1 should always pick argmax regardless of seed/temp.
    constexpr uint32_t V = 512;
    std::vector<sycl::half> hl(V);
    for (uint32_t i = 0; i < V; ++i) hl[i] = sycl::half(float(i % 5) * 0.5f);
    const uint32_t target = 200;
    hl[target] = sycl::half(20.0f);
    auto* dx = sycl::malloc_device<int32_t>(1, q);
    bool ok = true;
    char extra[160] = {};
    for (uint64_t seed = 1; seed <= 5 && ok; ++seed) {
        auto* dl = sycl::malloc_device<sycl::half>(V, q);
        q.memcpy(dl, hl.data(), V * sizeof(sycl::half)).wait();
        ie::sample_softmax_topk_topp(q, dl, dx, V, /*temp=*/1.f, /*top_k=*/1,
                                     /*top_p=*/1.f, /*min_p=*/0.f, seed).wait();
        int32_t got;
        q.memcpy(&got, dx, sizeof(int32_t)).wait();
        sycl::free(dl, q);
        if (got != int32_t(target)) {
            ok = false;
            std::snprintf(extra, sizeof(extra), " seed=%lu got=%d expected=%u",
                          (unsigned long)seed, got, target);
        }
    }
    sycl::free(dx, q);
    return report("sample top_k=1 always picks argmax", ok, extra);
}

bool test_sample_reproducible(sycl::queue& q) {
    // Same seed + same logits → same token across runs.
    constexpr uint32_t V = 256;
    std::vector<sycl::half> hl(V);
    std::mt19937 rng(0xBEEFL);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : hl) v = sycl::half(dist(rng));

    auto run = [&](uint64_t seed) -> int32_t {
        auto* dl = sycl::malloc_device<sycl::half>(V, q);
        auto* dx = sycl::malloc_device<int32_t>(1, q);
        q.memcpy(dl, hl.data(), V * sizeof(sycl::half)).wait();
        ie::sample_softmax_topk_topp(q, dl, dx, V, 1.0f, 0, 0.95f, 0.f, seed).wait();
        int32_t got;
        q.memcpy(&got, dx, sizeof(int32_t)).wait();
        sycl::free(dl, q); sycl::free(dx, q);
        return got;
    };
    const auto a = run(0xCAFEull);
    const auto b = run(0xCAFEull);
    const auto c = run(0xCAFFull);  // different seed
    char extra[120];
    std::snprintf(extra, sizeof(extra), " seed1=%d, seed1_again=%d, seed2=%d", a, b, c);
    return report("sampling reproducible per seed", a == b, extra);
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf_path = "/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) gguf_path = argv[++i];
    }
    auto q = make_queue();
    std::printf("Device: %s\n\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    int fails = 0;
    ie::GgufReader g;
    if (auto err = g.open(gguf_path); !err.empty()) {
        std::fprintf(stderr, "gguf open: %s\n", err.c_str());
        return 3;
    }
    ie::Tokenizer tok;
    if (!test_tok_load(tok, g)) ++fails;
    if (tok.ready()) {
        if (!test_special_tokens(tok)) ++fails;
        if (!test_known_ids(tok)) ++fails;
        if (!test_golden_vs_reference(tok)) ++fails;
        if (!test_round_trip(tok, "Hello, world",                              "round-trip ASCII 'Hello, world'")) ++fails;
        if (!test_round_trip(tok, "The quick brown fox jumps over the lazy dog.", "round-trip ASCII pangram")) ++fails;
        if (!test_round_trip(tok, "你好,世界",                                  "round-trip CJK '你好,世界'")) ++fails;
        if (!test_round_trip(tok, "🚀 Llama \xf0\x9f\xa6\x99 emoji \xe2\x9c\x85", "round-trip mixed emoji+text")) ++fails;
        if (!test_round_trip(tok, "Hello\n\tworld\n\n  spaces",                  "round-trip whitespace edge cases")) ++fails;
        if (!test_chatml_template(tok)) ++fails;
        if (!test_chatml_no_thinking(tok)) ++fails;
    }
    if (!test_chatml_tools_golden()) ++fails;
    if (!test_chatml_tool_sequence_golden()) ++fails;

    if (!test_argmax(q))             ++fails;
    if (!test_sample_temp_zero(q))   ++fails;
    if (!test_sample_topk1(q))       ++fails;
    if (!test_sample_reproducible(q))++fails;

    std::printf("\n%s\n", fails == 0 ? "ALL PASS" : "FAILURES");
    return fails;
}
