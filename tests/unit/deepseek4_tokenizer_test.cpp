// tests/unit/deepseek4_tokenizer_test.cpp — Phase 6 gate for DeepSeek-V4-Flash:
// the `joyai-llm` pre-tokenizer, the unknown-`pre` diagnostic, the
// encoding_dsv4.py prompt encoder and its matching output parser.
//
// PURE HOST, NO GPU, NO MODEL WEIGHTS. Compiles the SYCL-free tokenizer +
// gguf_reader + dtype TUs directly (same precedent as tekken_tokenizer_test /
// dbrx_pretok_test — does NOT link ie_core). It reads ONLY the tokenizer
// metadata KVs from shard 1 of the real GGUF, which is metadata-only (0 tensors,
// 5 MB), so this is a fast mmap header parse. Skips with exit 0 if absent.
//
// GOLDEN PROVENANCE — every expected value here is a captured REFERENCE output,
// not a re-derivation of the engine's own logic:
//   * §2 token ids: llama.cpp `llama_tokenize(add_special=false,
//     parse_special=true)` at commit fdc3db9b (LLAMA_VOCAB_PRE_TYPE_JOYAI_LLM),
//     run against this same GGUF's vocab. llama.cpp has no `deepseek4` arch, so
//     the reference run used a copy of shard 1 with general.architecture
//     rewritten to "llama" and the split.* keys dropped. That rewrite cannot
//     change the ids: load_hparams early-returns for vocab_only, and
//     llama-vocab.cpp reads general.architecture in exactly one place
//     (llama-vocab.cpp:2963) for a nomic-bert-moe / jina-bert-v3 <mask>-token
//     LSTRIP check that neither "deepseek4" nor "llama" matches.
//   * §4/§5 prompts: the official `encoding/encoding_dsv4.py`
//     encode_messages() / parse_message_from_completion_text() from
//     deepseek-ai/DeepSeek-V4-Flash-0731, run under python 3 + transformers 5.12.
//
// Build defines -DNDEBUG (Release); this TU compiles with -UNDEBUG so the
// asserts actually validate.

#include "ie/gguf.hpp"
#include "ie/tokenizer.hpp"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr const char* kDs4Shard1 =
    "${IE_MODELS_DIR}/DeepSeek-V4-Flash-0731-GGUF/UD-Q3_K_XL/"
    "DeepSeek-V4-Flash-0731-UD-Q3_K_XL-00001-of-00004.gguf";

struct Golden {
    const char*          text;
    std::vector<int32_t> ids;
};

// §2 corpus: multi-digit numbers, CJK/Kana/Hangul, code, whitespace edge cases,
// accents/combining marks, emoji, non-ASCII digits, and the '~' case (which is
// in NEITHER \p{P} nor \p{S} in llama.cpp's collapsed maps).
//
// The first four entries are MUTATION DISCRIMINATORS, each minimal and each
// pinning one property of pretokenize_joyai() that nothing else here detects.
// They exist because a gate review mutated tokenizer.cpp and found these four
// changes shipped green against the rest of this corpus:
//   "  1"          — cascade ORDER. Pass 1 (\p{N}{1,3}) must run BEFORE pass 3,
//                    so the digit is carved out first and the two spaces are left
//                    as one chunk (262 19). Run pass 1 last and alt-5 claims the
//                    spaces separately -> 223 223 19.
//   "~~<U+4E2D>"   — pass 2 (CJK) must exist. Drop it and the ideograph no longer
//                    terminates the '~' run -> 96 96 525 instead of 5906 525.
//   NBSP + space   — non-ASCII whitespace collapses to \v, NOT \n (llama.cpp maps
//                    it to 0x0B). As \n it would match alt-4 `\s*[\r\n]+` and
//                    split -> 2162 223 instead of the single token 84238. This is
//                    the direct minimal probe for that quirk; the U+2028 entry
//                    below also happens to catch it, but only as a side effect of
//                    testing something else, so keep this one.
//   <U+3072><U+1F600><SP><U+2028><U+9FFF>
//                  — the CJK upper bound is U+9FA5, not U+9FFF. Widen it and
//                    U+9FFF becomes a pass-2 match, moving the chunk boundary so
//                    BPE merges the space into <U+2028> -> ...225 663 104...
//                    rather than ...225 223 300 104...
// Do not delete these without replacing the coverage.
const std::vector<Golden>& token_goldens() {
    static const std::vector<Golden> g = {
    { "  1",
      { 262, 19 } },
    { "~~\xe4\xb8\xad",
      { 5906, 525 } },
    { "\xc2\xa0 ",
      { 84238 } },
    { "\xe3\x81\xb2\xf0\x9f\x98\x80 \xe2\x80\xa8\xe9\xbf\xbf",
      { 40259, 28927, 225, 223, 300, 104, 168, 126, 126 } },
    { "Hello, world!",
      { 19923, 14, 2058, 3 } },
    { "Result: 42 items, 1234567 bytes, 3.14159 pi",
      { 10610, 28, 223, 3180, 7316, 14, 223, 6895, 18009, 25, 20711, 14, 223, 21, 16, 9926, 3318, 7323 } },
    { "1 12 123 1234 12345 123456789",
      { 19, 223, 736, 223, 6895, 223, 6895, 22, 223, 6895, 1883, 223, 6895, 18009, 25744 } },
    { "\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c\xef\xbc\x81",
      { 30594, 303, 3427, 1175 } },
    { "\xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf\xe4\xb8\x96\xe7\x95\x8c"
      "\xe3\x80\x82\xe3\x82\xab\xe3\x82\xbf\xe3\x82\xab\xe3\x83\x8a",
      { 4549, 7245, 2298, 12457, 2841, 3427, 320, 15961, 11767, 15961, 27071 } },
    { "\xe4\xb8\xad\xe6\x96\x87\xe6\xb7\xb7\xe5\x90\x88" "English\xe5\x92\x8c" "123\xe6\x95"
      "\xb0\xe5\xad\x97\xe3\x80\x82",
      { 21134, 14769, 17530, 548, 6895, 8283, 320 } },
    { "\xed\x95\x9c\xea\xb5\xad\xec\x96\xb4 \xed\x85\x8d\xec\x8a\xa4\xed\x8a\xb8\xec\x9e\x85"
      "\xeb\x8b\x88\xeb\x8b\xa4.",
      { 5634, 17564, 9227, 61291, 238, 45652, 34941, 16 } },
    { "def f(x): return x*2",
      { 3465, 285, 4042, 2605, 1354, 1527, 12, 20 } },
    { "const userName = getUserName(user.id);",
      { 3949, 116326, 438, 93411, 3240, 28616, 12831, 3171 } },
    { "if (x == y) { foo(); } else { bar(); }",
      { 394, 343, 90, 2606, 383, 11, 680, 52735, 12184, 837, 3006, 680, 4758, 12184, 837 } },
    { "path/to/some/file.py",
      { 9860, 89215, 2283, 661, 82511, 23042 } },
    { "a = b ~ c; d = ~e; x ~= y",
      { 67, 438, 291, 223, 96, 274, 29, 283, 438, 223, 96, 71, 29, 1527, 223, 96, 31, 383 } },
    { "caf\xc3\xa9 au lait \xe2\x80\x94 na\xc3\xafve fa\xc3\xa7" "ade",
      { 69, 2797, 619, 6377, 124504, 2136, 112752, 112054 } },
    { "Combining: e\xcc\x81 a\xcc\x80 n\xcc\x83",
      { 30580, 2367, 28, 312, 17793, 260, 45345, 313, 59436 } },
    { "  two leading and trailing  ",
      { 223, 1234, 6646, 305, 52964, 262 } },
    { "tab\tmiddle\tend\tend",
      { 14278, 21840, 4920, 55655, 55655 } },
    { "a\n\n\nb",
      { 67, 6328, 68 } },
    { "\r\ncrlf\r\nlines\r\n",
      { 204, 201, 18977, 39724, 204, 201, 12678, 204, 201 } },
    { "   \n   \n   ",
      { 14320, 14320, 361 } },
    { " nbsp \xc2\xa0 thin \xe2\x80\x89 \xe3\x80\x80ideo\xe3\x80\x80",
      { 313, 12997, 13714, 12220, 663, 234, 223, 18524, 4778, 18524 } },
    { "!@#$%^&*()_+-=[]{}|;':\",./<>?`~",
      { 3, 34, 125463, 7, 64, 8, 12, 1393, 65, 36557, 31, 5071, 25902, 94, 29, 8201, 1760, 6984, 47357, 33, 66, 96 } },
    { "emoji \xf0\x9f\x98\x80 \xe2\x9d\xa4\xef\xb8\x8f math \xe2\x88\x80x\xe2\x88\x88\xe2\x84"
      "\x9d",
      { 18872, 7063, 53769, 225, 53341, 100, 10759, 7704, 76547, 90, 20954, 119659 } },
    { "\xef\xbc\x91\xef\xbc\x92\xef\xbc\x93 \xd9\xa0\xd9\xa1\xd9\xa2 \xc2\xb2\xc2\xb3 \xe2"
      "\x91\xa0\xe2\x91\xa1",
      { 127167, 25081, 223, 82025, 63468, 76647, 223, 1628, 5826, 223, 104894 } },
    { "https://example.com/a/b?c=1&d=2#frag",
      { 5395, 2272, 30357, 2193, 20922, 9928, 33, 69, 31, 19, 8, 70, 31, 20, 5, 72, 3174 } },
    { "XMLHttpRequest camelCase snake_case_var",
      { 52390, 15718, 8546, 82389, 15434, 34951, 78392, 72321 } },
    };
    return g;
}

struct ChatTurnLit { const char* role; const char* content; };
struct ChatGolden {
    const char*                     name;
    std::vector<ChatTurnLit>        turns;
    bool                            thinking;
    const char*                     effort;
    bool                            add_bos;
    const char*                     tools_json;
    const char*                     prompt;
};

const std::vector<ChatGolden>& chat_goldens() {
    static const std::vector<ChatGolden> g = {
    { "plain_thinking",
      { {"user", "Hello"} },
      true, "low", true,
      "",
      "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c><\xef\xbd\x9cUser"
      "\xef\xbd\x9c>Hello<\xef\xbd\x9c" "Assistant\xef\xbd\x9c><think>" },
    { "plain_chat",
      { {"user", "Hello"} },
      false, "low", true,
      "",
      "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c><\xef\xbd\x9cUser"
      "\xef\xbd\x9c>Hello<\xef\xbd\x9c" "Assistant\xef\xbd\x9c></think>" },
    { "effort_high",
      { {"user", "Solve x^2=4"} },
      true, "high", true,
      "",
      "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>Reasoning Effort: "
      "Absolute maximum with no shortcuts permitted.\nYou MUST be very thorough in your think"
      "ing and comprehensively decompose the problem to resolve the root cause, rigorously st"
      "ress-testing your logic against all potential paths, edge cases, and adversarial scena"
      "rios.\nExplicitly write out your entire deliberation process, documenting every interm"
      "ediate step, considered alternative, and rejected hypothesis to ensure absolutely no a"
      "ssumption is left unchecked.\n\n<\xef\xbd\x9cUser\xef\xbd\x9c>Solve x^2=4<\xef\xbd\x9c"
      "" "Assistant\xef\xbd\x9c><think>" },
    { "effort_max",
      { {"user", "Solve x^2=4"} },
      true, "max", true,
      "",
      "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>Reasoning Effort: "
      "Beyond maximum \xe2\x80\x94 exhaustive, relentless, and uncompromising.\nYou MUST reas"
      "on with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively"
      " decompose the problem into its most fundamental components, trace every causal chain "
      "to its root, and resolve the underlying cause rather than any surface symptom.\nDo not"
      " stop reasoning until you have independently verified the solution from multiple angle"
      "s and are certain that no assumption remains unchecked and no error remains undiscover"
      "ed.\n\n<\xef\xbd\x9cUser\xef\xbd\x9c>Solve x^2=4<\xef\xbd\x9c" "Assistant\xef\xbd\x9c>"
      "<think>" },
    { "effort_high_chat",
      { {"user", "Solve x^2=4"} },
      false, "high", true,
      "",
      "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c><\xef\xbd\x9cUser"
      "\xef\xbd\x9c>Solve x^2=4<\xef\xbd\x9c" "Assistant\xef\xbd\x9c></think>" },
    { "multi_turn",
      { {"user", "Hi"}, {"assistant", "Hello!"}, {"user", "How are you?"} },
      true, "low", true,
      "",
      "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c><\xef\xbd\x9cUser"
      "\xef\xbd\x9c>Hi<\xef\xbd\x9c" "Assistant\xef\xbd\x9c></think>Hello!<\xef\xbd\x9c" "end"
      "\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c><\xef\xbd\x9cUser\xef\xbd\x9c>How are y"
      "ou?<\xef\xbd\x9c" "Assistant\xef\xbd\x9c><think>" },
    { "multi_turn_chat",
      { {"user", "Hi"}, {"assistant", "Hello!"}, {"user", "How are you?"} },
      false, "low", true,
      "",
      "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c><\xef\xbd\x9cUser"
      "\xef\xbd\x9c>Hi<\xef\xbd\x9c" "Assistant\xef\xbd\x9c></think>Hello!<\xef\xbd\x9c" "end"
      "\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c><\xef\xbd\x9cUser\xef\xbd\x9c>How are y"
      "ou?<\xef\xbd\x9c" "Assistant\xef\xbd\x9c></think>" },
    { "multi_turn_3",
      { {"user", "a"}, {"assistant", "b"}, {"user", "c"}, {"assistant", "d"}, {"user", "e"} },
      true, "low", true,
      "",
      "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c><\xef\xbd\x9cUser"
      "\xef\xbd\x9c>a<\xef\xbd\x9c" "Assistant\xef\xbd\x9c></think>b<\xef\xbd\x9c" "end\xe2"
      "\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c><\xef\xbd\x9cUser\xef\xbd\x9c>c<\xef\xbd"
      "\x9c" "Assistant\xef\xbd\x9c></think>d<\xef\xbd\x9c" "end\xe2\x96\x81of\xe2\x96\x81sen"
      "tence\xef\xbd\x9c><\xef\xbd\x9cUser\xef\xbd\x9c>e<\xef\xbd\x9c" "Assistant\xef\xbd\x9c"
      "><think>" },
    { "system_user",
      { {"system", "You are terse."}, {"user", "Hi"} },
      true, "low", true,
      "",
      "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>You are terse.<"
      "\xef\xbd\x9cUser\xef\xbd\x9c>Hi<\xef\xbd\x9c" "Assistant\xef\xbd\x9c><think>" },
    { "consecutive_users",
      { {"user", "first"}, {"user", "second"} },
      true, "low", true,
      "",
      "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c><\xef\xbd\x9cUser"
      "\xef\xbd\x9c>first\n\nsecond<\xef\xbd\x9c" "Assistant\xef\xbd\x9c><think>" },
    { "no_bos",
      { {"user", "Hi"} },
      true, "low", false,
      "",
      "<\xef\xbd\x9cUser\xef\xbd\x9c>Hi<\xef\xbd\x9c" "Assistant\xef\xbd\x9c><think>" },
    { "assistant_last",
      { {"user", "Hi"}, {"assistant", "There"} },
      true, "low", true,
      "",
      "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c><\xef\xbd\x9cUser"
      "\xef\xbd\x9c>Hi<\xef\xbd\x9c" "Assistant\xef\xbd\x9c><think></think>There<\xef\xbd\x9c"
      "" "end\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>" },
    { "unicode",
      { {"user", "\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c\xef\xbc\x81 12345"} },
      true, "low", true,
      "",
      "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c><\xef\xbd\x9cUser"
      "\xef\xbd\x9c>\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe4\xb8\x96\xe7\x95\x8c\xef\xbc\x81 "
      "12345<\xef\xbd\x9c" "Assistant\xef\xbd\x9c><think>" },
    { "tool_result",
      { {"user", "Weather in Paris?"}, {"assistant", "Checking.\n\n<\xef\xbd\x9c" "DSML\xef\xbd\x9ctool_calls>\n<\xef\xbd\x9c" "DSML\xef\xbd" "\x9cinvoke name=\"get_weather\">\n<\xef\xbd\x9c" "DSML\xef\xbd\x9cparameter name=\"cit" "y\" string=\"true\">Paris</\xef\xbd\x9c" "DSML\xef\xbd\x9cparameter>\n</\xef\xbd\x9c" "" "DSML\xef\xbd\x9cinvoke>\n</\xef\xbd\x9c" "DSML\xef\xbd\x9ctool_calls>"}, {"tool", "sunny"} },
      true, "low", true,
      "",
      "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c><\xef\xbd\x9cUser"
      "\xef\xbd\x9c>Weather in Paris?<\xef\xbd\x9c" "Assistant\xef\xbd\x9c></think>Checking."
      "\n\n<\xef\xbd\x9c" "DSML\xef\xbd\x9ctool_calls>\n<\xef\xbd\x9c" "DSML\xef\xbd\x9cinvok"
      "e name=\"get_weather\">\n<\xef\xbd\x9c" "DSML\xef\xbd\x9cparameter name=\"city\" strin"
      "g=\"true\">Paris</\xef\xbd\x9c" "DSML\xef\xbd\x9cparameter>\n</\xef\xbd\x9c" "DSML\xef"
      "\xbd\x9cinvoke>\n</\xef\xbd\x9c" "DSML\xef\xbd\x9ctool_calls><\xef\xbd\x9c" "end\xe2"
      "\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c><\xef\xbd\x9cUser\xef\xbd\x9c><tool_result>"
      "sunny</tool_result><\xef\xbd\x9c" "Assistant\xef\xbd\x9c><think>" },
    // Phase L: the server renders a prior structured tool call as the canonical
    // <tool_call> block; the builder must transcode it to the DSML the model emits
    // (byte-identical to the "tool_result" case above, which echoes the DSML text).
    { "tool_calls_canonical",
      { {"user", "Weather in Paris?"}, {"assistant", "Checking.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"}, {"tool", "sunny"} },
      true, "low", true,
      "",
      "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c><\xef\xbd\x9cUser"
      "\xef\xbd\x9c>Weather in Paris?<\xef\xbd\x9c" "Assistant\xef\xbd\x9c></think>Checking."
      "\n\n<\xef\xbd\x9c" "DSML\xef\xbd\x9ctool_calls>\n<\xef\xbd\x9c" "DSML\xef\xbd\x9cinvok"
      "e name=\"get_weather\">\n<\xef\xbd\x9c" "DSML\xef\xbd\x9cparameter name=\"city\" strin"
      "g=\"true\">Paris</\xef\xbd\x9c" "DSML\xef\xbd\x9cparameter>\n</\xef\xbd\x9c" "DSML\xef"
      "\xbd\x9cinvoke>\n</\xef\xbd\x9c" "DSML\xef\xbd\x9ctool_calls><\xef\xbd\x9c" "end\xe2"
      "\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c><\xef\xbd\x9cUser\xef\xbd\x9c><tool_result>"
      "sunny</tool_result><\xef\xbd\x9c" "Assistant\xef\xbd\x9c><think>" },
    { "tool_calls_canonical_nonstring",
      { {"user", "Weather in Paris?"}, {"assistant", "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 3}}\n</tool_call>"}, {"tool", "sunny"} },
      true, "low", true,
      "",
      "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c><\xef\xbd\x9cUser"
      "\xef\xbd\x9c>Weather in Paris?<\xef\xbd\x9c" "Assistant\xef\xbd\x9c></think>"
      "\n\n<\xef\xbd\x9c" "DSML\xef\xbd\x9ctool_calls>\n<\xef\xbd\x9c" "DSML\xef\xbd\x9cinvok"
      "e name=\"get_weather\">\n<\xef\xbd\x9c" "DSML\xef\xbd\x9cparameter name=\"city\" strin"
      "g=\"true\">Paris</\xef\xbd\x9c" "DSML\xef\xbd\x9cparameter>\n<\xef\xbd\x9c" "DSML\xef\xbd\x9cparameter name=\"days\" string=\"false\">3</\xef\xbd\x9c" "DSML\xef\xbd\x9cparameter>\n</\xef\xbd\x9c" "DSML\xef"
      "\xbd\x9cinvoke>\n</\xef\xbd\x9c" "DSML\xef\xbd\x9ctool_calls><\xef\xbd\x9c" "end\xe2"
      "\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c><\xef\xbd\x9cUser\xef\xbd\x9c><tool_result>"
      "sunny</tool_result><\xef\xbd\x9c" "Assistant\xef\xbd\x9c><think>" },
    { "tools_multi",
      { {"system", "SYS"}, {"user", "u1"}, {"assistant", "a1"}, {"user", "u2"} },
      true, "low", true,
      R"TOOLS([{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}])TOOLS",
      "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>SYS\n\n## Tools\n"
      "\nYou have access to a set of tools to help answer the user's question. You can invoke"
      " tools by writing a \"<\xef\xbd\x9c" "DSML\xef\xbd\x9ctool_calls>\" block like the fol"
      "lowing:\n\n<\xef\xbd\x9c" "DSML\xef\xbd\x9ctool_calls>\n<\xef\xbd\x9c" "DSML\xef\xbd"
      "\x9cinvoke name=\"$TOOL_NAME\">\n<\xef\xbd\x9c" "DSML\xef\xbd\x9cparameter name=\"$PAR"
      "AMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</\xef\xbd\x9c" "DSML\xef\xbd\x9cp"
      "arameter>\n...\n</\xef\xbd\x9c" "DSML\xef\xbd\x9cinvoke>\n<\xef\xbd\x9c" "DSML\xef\xbd"
      "\x9cinvoke name=\"$TOOL_NAME2\">\n...\n</\xef\xbd\x9c" "DSML\xef\xbd\x9cinvoke>\n</"
      "\xef\xbd\x9c" "DSML\xef\xbd\x9ctool_calls>\n\nString parameters should be specified as"
      " is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects"
      "), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is en"
      "abled (triggered by <think>), you MUST output your complete reasoning inside <think>.."
      ".</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after"
      " </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n{\"name"
      "\": \"get_weather\", \"description\": \"Get weather\", \"parameters\": {\"type\": \"ob"
      "ject\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}"
      "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invo"
      "ke tool calls.\n<\xef\xbd\x9cUser\xef\xbd\x9c>u1<\xef\xbd\x9c" "Assistant\xef\xbd\x9c>"
      "<think></think>a1<\xef\xbd\x9c" "end\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c><"
      "\xef\xbd\x9cUser\xef\xbd\x9c>u2<\xef\xbd\x9c" "Assistant\xef\xbd\x9c><think>" },
    { "tools_multi_chat",
      { {"system", "SYS"}, {"user", "u1"}, {"assistant", "a1"}, {"user", "u2"} },
      false, "low", true,
      R"TOOLS([{"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}])TOOLS",
      "<\xef\xbd\x9c" "begin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c>SYS\n\n## Tools\n"
      "\nYou have access to a set of tools to help answer the user's question. You can invoke"
      " tools by writing a \"<\xef\xbd\x9c" "DSML\xef\xbd\x9ctool_calls>\" block like the fol"
      "lowing:\n\n<\xef\xbd\x9c" "DSML\xef\xbd\x9ctool_calls>\n<\xef\xbd\x9c" "DSML\xef\xbd"
      "\x9cinvoke name=\"$TOOL_NAME\">\n<\xef\xbd\x9c" "DSML\xef\xbd\x9cparameter name=\"$PAR"
      "AMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</\xef\xbd\x9c" "DSML\xef\xbd\x9cp"
      "arameter>\n...\n</\xef\xbd\x9c" "DSML\xef\xbd\x9cinvoke>\n<\xef\xbd\x9c" "DSML\xef\xbd"
      "\x9cinvoke name=\"$TOOL_NAME2\">\n...\n</\xef\xbd\x9c" "DSML\xef\xbd\x9cinvoke>\n</"
      "\xef\xbd\x9c" "DSML\xef\xbd\x9ctool_calls>\n\nString parameters should be specified as"
      " is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects"
      "), pass the value in JSON format and set `string=\"false\"`.\n\nIf thinking_mode is en"
      "abled (triggered by <think>), you MUST output your complete reasoning inside <think>.."
      ".</think> BEFORE any tool calls or final response.\n\nOtherwise, output directly after"
      " </think> with tool calls or final response.\n\n### Available Tool Schemas\n\n{\"name"
      "\": \"get_weather\", \"description\": \"Get weather\", \"parameters\": {\"type\": \"ob"
      "ject\", \"properties\": {\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}"
      "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invo"
      "ke tool calls.\n<\xef\xbd\x9cUser\xef\xbd\x9c>u1<\xef\xbd\x9c" "Assistant\xef\xbd\x9c>"
      "</think>a1<\xef\xbd\x9c" "end\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c><\xef\xbd"
      "\x9cUser\xef\xbd\x9c>u2<\xef\xbd\x9c" "Assistant\xef\xbd\x9c></think>" },
    };
    return g;
}

bool ids_equal(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return false;
    return true;
}

void print_ids(const char* tag, const std::vector<int32_t>& v) {
    std::printf("    %s [", tag);
    for (size_t i = 0; i < v.size(); ++i) std::printf("%s%d", i ? ", " : "", v[i]);
    std::printf("]\n");
}

// A GGUF whose tokenizer.ggml.pre this engine has NO dedicated split for
// (Qwen3.5/3.6 report "qwen35"), used to prove the diagnostic actually fires.
const char* find_unimplemented_pre_gguf() {
    static const char* candidates[] = {
        "$HOME/models/quant-lab-35b/Qwen3.6-35B-A3B-Q1_0.gguf",
        "$HOME/models/lmstudio-community/Bonsai-27B-GGUF/Bonsai-27B-Q1_0.gguf",
        "$HOME/models/unsloth-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
        nullptr,
    };
    for (int i = 0; candidates[i]; ++i) {
        if (FILE* f = std::fopen(candidates[i], "rb")) { std::fclose(f); return candidates[i]; }
    }
    return nullptr;
}

int failures = 0;

}  // namespace

int main() {
    // The token goldens (§1-§3, §6) need the 0731 GGUF; the prompt builder (§4,
    // string half) and the completion parser (§5) are pure string code and run
    // regardless — so a box without that file still gates the chat encoding.
    ie::GgufReader g;
    ie::Tokenizer tok;
    bool have_tok = true;
    const std::string err = g.open(kDs4Shard1);
    if (!err.empty()) {
        std::fprintf(stderr, "deepseek4_tokenizer_test: SKIP token sections (cannot open %s: %s)"
                             " — §4 prompt strings and §5 parsing still run\n",
                     kDs4Shard1, err.c_str());
        have_tok = false;
    } else {
        const std::string terr = tok.load_from_gguf(g);
        if (!terr.empty()) {
            std::printf("deepseek4_tokenizer_test: FAIL (tokenizer load: %s)\n", terr.c_str());
            return 1;
        }
    }

    if (have_tok) {
    // ---- §1 pre-type dispatch ----
    assert(tok.pre() == "joyai-llm");
    // joyai-llm IS implemented, so it must NOT warn.
    assert(tok.pre_warning().empty());
    assert(tok.vocab_size() == 129280);
    assert(tok.bos_token_id() == 0);
    assert(tok.eos_token_id() == 1);
    assert(tok.pad_token_id() == 2);
    // add_bos_token=false in the GGUF: the BOS sentinel is a TEMPLATE literal for
    // this model, so encode() must not prepend a second one.
    assert(!tok.add_bos_token());
    if (tok.pre() != "joyai-llm" || !tok.pre_warning().empty() ||
        tok.vocab_size() != 129280 || tok.add_bos_token()) {
        std::printf("deepseek4_tokenizer_test: FAIL (§1 pre-type dispatch)\n");
        return 1;
    }

    // ---- §2 token-for-token parity vs llama.cpp ----
    for (const auto& gd : token_goldens()) {
        const std::vector<int32_t> got = tok.encode(gd.text, /*allow_special=*/true);
        if (!ids_equal(got, gd.ids)) {
            ++failures;
            std::printf("  §2 MISMATCH on [%s]\n", std::string(gd.text).substr(0, 60).c_str());
            print_ids("want", gd.ids);
            print_ids("got ", got);
        }
    }

    // ---- §3 lossless round-trip (no special-token literals in these) ----
    for (const auto& gd : token_goldens()) {
        const auto ids = tok.encode(gd.text, /*allow_special=*/false);
        const std::string dec = tok.decode(ids, /*skip_special=*/true);
        if (dec != gd.text) {
            ++failures;
            std::printf("  §3 ROUND-TRIP MISMATCH: in=[%s] out=[%s]\n", gd.text, dec.c_str());
        }
    }

    }   // have_tok: §1-§3

    // ---- §4 chat encoding vs encoding_dsv4.py encode_messages() ----
    for (const auto& cg : chat_goldens()) {
        std::vector<ie::ChatTurn> turns;
        for (const auto& t : cg.turns) turns.push_back({t.role, t.content});
        ie::DeepSeek4ChatOptions opt;
        opt.thinking         = cg.thinking;
        opt.reasoning_effort = cg.effort;
        opt.add_bos          = cg.add_bos;
        opt.tools_json       = cg.tools_json;
        const std::string got = ie::build_deepseek4_prompt(turns, /*add_generation_prompt=*/true, opt);
        if (got != cg.prompt) {
            ++failures;
            std::printf("  §4 CHAT MISMATCH [%s]\n    want: %s\n    got : %s\n",
                        cg.name, cg.prompt, got.c_str());
        }
        // The rendered prompt must also tokenize — the sentinels are CONTROL
        // tokens, so each one collapses to a single id.
        if (have_tok) {
            const auto ids = tok.encode(got, /*allow_special=*/true);
            if (ids.empty()) { ++failures; std::printf("  §4 EMPTY ENCODE [%s]\n", cg.name); }
        }
    }

    // ---- §5 output parsing vs parse_message_from_completion_text() ----
    {
        const std::string dsml = "\xef\xbd\x9c" "DSML" "\xef\xbd\x9c";
        const std::string eos =
            "<\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c>";
        struct PCase {
            std::string text;
            bool        thinking;
            bool        want_ok;
            const char* content;
            const char* reasoning;
            const char* tool_calls;   // engine JSON (compact); "" when want_ok=false
        };
        const std::vector<PCase> pc = {
            { "Some reasoning here.</think>The answer is 4." + eos, true, true,
              "The answer is 4.", "Some reasoning here.", "[]" },
            { "The answer is 4." + eos, false, true, "The answer is 4.", "", "[]" },
            { "</think>hi" + eos, true, true, "hi", "", "[]" },
            { "reason</think>Calling.\n\n<" + dsml + "tool_calls>\n<" + dsml +
              "invoke name=\"get_weather\">\n<" + dsml +
              "parameter name=\"city\" string=\"true\">Paris</" + dsml + "parameter>\n<" + dsml +
              "parameter name=\"days\" string=\"false\">3</" + dsml + "parameter>\n</" + dsml +
              "invoke>\n</" + dsml + "tool_calls>", true, true,
              "Calling.", "reason",
              R"([{"type":"function","function":{"name":"get_weather",)"
              R"("arguments":"{\"city\": \"Paris\", \"days\": 3}"}}])" },
            { "r</think>c\n\n<" + dsml + "tool_calls>\n<" + dsml + "invoke name=\"ping\">\n\n</" +
              dsml + "invoke>\n</" + dsml + "tool_calls>" + eos, true, true,
              "c", "r",
              R"([{"type":"function","function":{"name":"ping","arguments":"{}"}}])" },
            { "c\n\n<" + dsml + "tool_calls>\n<" + dsml + "invoke name=\"f\">\n<" + dsml +
              "parameter name=\"a\" string=\"false\">[1, 2]</" + dsml + "parameter>\n<" + dsml +
              "parameter name=\"b\" string=\"true\">x\"y</" + dsml + "parameter>\n</" + dsml +
              "invoke>\n</" + dsml + "tool_calls>", false, true,
              "c", "",
              R"([{"type":"function","function":{"name":"f",)"
              R"("arguments":"{\"a\": [1, 2], \"b\": \"x\\\"y\"}"}}])" },
            { "r</think>c\n\n<" + dsml + "tool_calls>\n<" + dsml + "invoke name=\"f\">\n<" + dsml +
              "parameter name=\"code\" string=\"true\">line1\nline2</" + dsml + "parameter>\n</" +
              dsml + "invoke>\n</" + dsml + "tool_calls>", true, true,
              "c", "r",
              R"([{"type":"function","function":{"name":"f",)"
              R"("arguments":"{\"code\": \"line1\\nline2\"}"}}])" },
            // Malformed: upstream RAISES on each of these, so the port must report an
            // error rather than invent a plausible parse.
            { "no think marker" + eos, true, false, "", "", "" },
            { "content with no terminator", false, false, "", "", "" },
            { "x" + eos + "trailing", false, false, "", "", "" },
        };
        for (size_t i = 0; i < pc.size(); ++i) {
            const auto r = ie::parse_deepseek4_completion(pc[i].text, pc[i].thinking);
            if (pc[i].want_ok) {
                if (!r.error.empty() || r.content != pc[i].content ||
                    r.reasoning_content != pc[i].reasoning ||
                    r.tool_calls_json != pc[i].tool_calls) {
                    ++failures;
                    std::printf("  §5 PARSE MISMATCH #%zu\n    err=[%s]\n"
                                "    content want=[%s] got=[%s]\n"
                                "    reason  want=[%s] got=[%s]\n"
                                "    calls   want=[%s] got=[%s]\n",
                                i, r.error.c_str(), pc[i].content, r.content.c_str(),
                                pc[i].reasoning, r.reasoning_content.c_str(),
                                pc[i].tool_calls, r.tool_calls_json.c_str());
                }
            } else if (r.error.empty()) {
                ++failures;
                std::printf("  §5 PARSE #%zu accepted a MALFORMED completion\n", i);
            }
        }
    }

    if (have_tok) {
    // ---- §6 the unknown-`pre` diagnostic actually fires ----
    if (const char* other = find_unimplemented_pre_gguf()) {
        ie::GgufReader g2;
        const std::string e2 = g2.open(other);
        if (e2.empty()) {
            ie::Tokenizer t2;
            const std::string te2 = t2.load_from_gguf(g2);
            if (!te2.empty()) {
                std::printf("  §6 tokenizer load failed on %s: %s\n", other, te2.c_str());
                ++failures;
            } else if (t2.pre() == "qwen35") {
                // The engine has no dedicated qwen35 split (llama.cpp's differs
                // from qwen2 by \p{M} handling) — it MUST say so, not fall
                // through in silence, and it must still LOAD.
                if (t2.pre_warning().empty()) {
                    ++failures;
                    std::printf("  §6 SILENT FALLBACK: pre=\"qwen35\" produced no diagnostic\n");
                }
                if (t2.pre_warning().find("qwen35") == std::string::npos) {
                    ++failures;
                    std::printf("  §6 diagnostic does not name the pre value: [%s]\n",
                                std::string(t2.pre_warning()).c_str());
                }
                if (!t2.ready()) {
                    ++failures;
                    std::printf("  §6 REGRESSION: an unimplemented pre must WARN, not refuse\n");
                }
            }
        }
    } else {
        // VALIDATION GAP (not a failure): no GGUF with an unimplemented `pre` on
        // this box, so the negative half of §6 could not run.
        std::printf("  §6 VALIDATION GAP: no qwen35 GGUF found; the warning path is ungated here\n");
    }

    }   // have_tok: §6

    if (failures) {
        std::printf("deepseek4_tokenizer_test: FAIL (%d checks failed)\n", failures);
        return 1;
    }
    if (!have_tok)
        std::printf("deepseek4_tokenizer_test: OK on the string sections only (%zu chat goldens, "
                    "parse cases); token sections SKIPPED (no GGUF)\n", chat_goldens().size());
    else
        std::printf("deepseek4_tokenizer_test: OK (%zu token goldens, %zu round-trips, "
                    "%zu chat goldens, 10 parse cases)\n",
                    token_goldens().size(), token_goldens().size(), chat_goldens().size());
    return 0;
}
