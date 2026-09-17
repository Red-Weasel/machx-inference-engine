// include/ie/tokenizer.hpp — byte-level BPE tokenizer for Qwen3.6.
//
// Design notes:
//   * Vocab + merges live in the GGUF (`tokenizer.ggml.tokens`,
//     `tokenizer.ggml.merges`).
//   * Encoding uses GPT-2-style byte mapping: each input byte → a single
//     printable UTF-8 codepoint, then standard BPE merging using the merge
//     priority table.
//   * Pre-tokenization is a simplified GPT-2 regex (splits on whitespace and
//     punctuation boundaries, attaches a leading space to the next pretoken).
//     The full Unicode `\p{L}\p{N}\p{M}` regex is in the Phase 9 backlog —
//     v1 tokenization may diverge from HF on edge cases (CJK script transitions,
//     numbers attached to letters, etc.) but round-trips losslessly and
//     matches HF on standard ASCII / common-prose inputs.

#pragma once

#include "ie/gguf.hpp"

#include <cstdint>
#include <span>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ie {

class Tokenizer {
public:
    Tokenizer() = default;

    // Load vocab + merges + special-token ids from the GGUF. Returns empty
    // string on success, error message otherwise.
    std::string load_from_gguf(const GgufReader& g);
    // HF `tokenizers` JSON (DeepSeek-V4.1-Flash, docs/deepseek41/31): byte-level BPE vocab + merges
    // from tokenizer.json, the added tokens as specials, bos/eos from tokenizer_config.json; the
    // pre-tokenizer is the DeepSeek-V3/V4 split the GGUF path knows as "joyai-llm" (the JSON's
    // Split sequence: \p{N}{1,3} runs, CJK runs, the DeepSeek pattern). `""` on success.
    std::string load_from_hf_json(const std::string& tokenizer_json, const std::string& tokenizer_config_json);

    bool        ready() const noexcept { return !vocab_.empty(); }
    uint32_t    vocab_size() const noexcept { return uint32_t(vocab_.size()); }

    // tokenizer.ggml.pre as read from the GGUF (empty when the key is absent).
    std::string_view pre() const noexcept { return pre_; }
    // Non-empty when `pre` named a pre-tokenizer this engine has NO dedicated
    // implementation for; the load then falls back to the qwen2-family split and
    // tokenization may diverge from the reference. load_from_gguf() also prints
    // this to stderr — an unknown `pre` must never fall through silently.
    std::string_view pre_warning() const noexcept { return pre_warning_; }

    int32_t     bos_token_id() const noexcept { return bos_id_; }
    int32_t     eos_token_id() const noexcept { return eos_id_; }
    int32_t     pad_token_id() const noexcept { return pad_id_; }
    bool        add_bos_token() const noexcept { return add_bos_token_; }
    bool        is_special(int32_t id) const noexcept;
    std::string_view token_str(int32_t id) const noexcept;
    int32_t     find_token(std::string_view s) const noexcept;

    // Encode text → token IDs.
    //   `allow_special`: if true, recognize special-token literals like
    //   `<|im_start|>` in the input and emit them as their special IDs.
    //   If false, encode them as raw bytes (safer when input is user content).
    std::vector<int32_t> encode(std::string_view text, bool allow_special = true) const;

    // Decode token IDs → text.
    //   `skip_special`: if true, omit special tokens from the output.
    //   `keep_special`: ids emitted literally even when skip_special=true
    //   (used to preserve <tool_call>/</tool_call> markers in generated text
    //   so OpenAI clients can recover text-embedded tool calls).
    std::string decode(std::span<const int32_t> ids, bool skip_special = false,
                       std::span<const int32_t> keep_special = {}) const;

private:
    void               build_byte_maps();
    int32_t            bpe_lookup_or_neg(std::string_view s) const noexcept;
    std::vector<int32_t> bpe_merge_word(std::string_view encoded) const;
    // Gemma SPM-style BPE: raw UTF-8 (no byte-encode), spaces→U+2581 done by the
    // caller, <0xXX> byte fallback. Appends ids to `out`.
    void               bpe_merge_gemma(std::string_view seg, std::vector<int32_t>& out) const;
    // Classic SentencePiece (Llama-1/2, Mistral/Codestral): raw UTF-8 symbols
    // merged by TOKEN SCORE (highest first, leftmost on tie — llama.cpp
    // llm_tokenizer_spm), <0xXX> byte fallback. Caller does add_space_prefix +
    // spaces→U+2581. Appends ids to `out`.
    void               bpe_merge_spm(std::string_view seg, std::vector<int32_t>& out) const;

    // Vocab: owned strings; lookup map references vocab_[i].
    std::vector<std::string>                          vocab_;
    std::unordered_map<std::string_view, int32_t>     vocab_lookup_;

    // Merges: (a_id << 32 | b_id) → rank (lower = higher priority).
    std::unordered_map<uint64_t, int32_t>             merge_rank_;

    // SentencePiece per-token scores (tokenizer.ggml.scores), one per vocab id.
    // Non-empty only for the SPM path (model=="llama"); the merge picks the
    // adjacent pair whose merged token has the highest score.
    std::vector<float>                                scores_;

    // Special tokens (token_type == CONTROL or > USER_DEFINED in HF parlance).
    std::unordered_set<int32_t>                       special_ids_;

    // Sorted special-token strings for greedy match in encode().
    std::vector<std::pair<std::string, int32_t>>      special_text_;

    int32_t bos_id_     = -1;
    int32_t eos_id_     = -1;
    int32_t pad_id_     = -1;
    bool    add_bos_token_ = false;

    // P3a: tokenizer.ggml.pre dispatch. "llama-bpe" groups digits in runs of
    // 1-3 (\p{N}{1,3}) and does ignore_merges (whole pretoken looked up in the
    // vocab before BPE). Empty/"qwen2" → single-digit split, no ignore_merges.
    std::string pre_;
    bool        digits_1to3_   = false;
    bool        ignore_merges_ = false;
    bool        gemma_         = false;   // pre=="gemma4": SPM-style raw-UTF-8 BPE

    // Wave-1: classic SentencePiece (tokenizer.ggml.model=="llama") — Llama-1/2,
    // Mistral, Codestral. Score-based merge (scores_), no merges list, <0xXX>
    // byte fallback, and a leading dummy ▁ when add_space_prefix_. Gated so every
    // BPE/Gemma path is byte-identical.
    bool        spm_              = false;
    bool        add_space_prefix_ = true;  // tokenizer.ggml.add_space_prefix (SPM)

    // Wave-1: "tekken" pre-tokenizer (Mistral-Nemo / Small-24B / Devstral /
    // Codestral-tekken). tiktoken-style split with CASE-AWARE letter words
    // (camelCase / XMLHttp split on case boundaries — critical for code) and
    // single-digit runs. Distinct from the qwen2/llama-bpe `\p{L}+` split.
    // Gated so the crown/qwen3/llama default paths are byte-identical.
    bool        tekken_        = false;

    // Phase 6: "joyai-llm" pre-tokenizer (DeepSeek-V4-Flash; llama.cpp
    // LLAMA_VOCAB_PRE_TYPE_JOYAI_LLM). A THREE-PASS cascade (\p{N}{1,3}, then a
    // CJK/Kana pre-split, then the DeepSeek-V3/V4 split) over real Unicode
    // categories — see pretokenize_joyai(). Gated so every other pre-type keeps
    // the byte-identical simple/tekken path.
    bool        joyai_         = false;
    // "hyv4" (Tencent Hy4-preview): the SAME three-pass cascade, except '~'
    // (0x7E) is a real \p{S} there — upstream's hand-coded hyv4 splitter uses
    // unicode_cpt_flags (where '~' is a symbol), not the k_ucat_map replica
    // that deliberately omits it. P2 gate 2026-09-01: 15/62 '~'+punct pairs
    // split differently through the joyai table.
    bool        hyv4_          = false;

    // Diagnostic for an unimplemented tokenizer.ggml.pre (see pre_warning()).
    std::string pre_warning_;

    // GPT-2 byte ↔ codepoint maps.
    std::array<std::string, 256>                      byte_encoder_;
    std::unordered_map<std::string, uint8_t>          byte_decoder_;
};

// ChatML chat template builder for Qwen3.6.
//   Produces the prompt string corresponding to a list of (role, content)
//   pairs, following the structure:
//     <|im_start|>system\n{sys}<|im_end|>\n
//     <|im_start|>user\n{u1}<|im_end|>\n
//     <|im_start|>assistant\n{a1}<|im_end|>\n
//     ...
//     <|im_start|>assistant\n   (open turn, with thinking prefix)
//
// Caller passes `add_generation_prompt=true` to open the assistant turn for
// generation.
//   enable_thinking=true  — appends `<think>\n` so the model's first
//                           generated tokens go inside the reasoning trace.
//   enable_thinking=false — appends `<think>\n\n</think>\n\n` (an empty
//                           think block), which is the Qwen3-family convention
//                           for signalling "reasoning already done" and
//                           suppressing any further thinking trace.
// Where an image sat inside multi-part chat content: the OpenAI parser drops
// this marker into `content` at the image's position (one per image, in
// order) so an arch that has a native image token can honour the placement;
// an arch that only prepends strips it. Not a vocab token anywhere.
inline constexpr std::string_view kChatImageMarker = "<<ie-image>>";

struct ChatTurn {
    std::string role;          // "system", "user", "assistant", "tool"
    std::string content;
    // Raw encoded image bytes (JPEG/PNG/BMP) attached to this turn, in order.
    // Only consumed by vision-capable arches (qwen4exp); empty everywhere else.
    std::vector<std::string> images;
    // Structured history metadata. Appended to preserve three-field callers.
    std::optional<std::string> reasoning_content;
    std::vector<std::string> tool_call_ids; // assistant call order; empty ids retain fallback order
    std::string tool_call_id;              // tool response association
    // Original assistant data for native protocols; legacy content retains Qwen rendering.
    std::string tool_calls_json;
    std::optional<std::string> content_without_tool_calls;
};

// `tools_json`: raw OpenAI `tools` array (JSON text). Empty → byte-identical
// to the historical 3-arg behavior. Non-empty → the canonical Qwen3 tools
// preamble ("# Tools ... <tools>{schema per line}</tools> ... emit
// <tool_call>{\"name\":..., \"arguments\":...}</tool_call>") is appended to
// the system turn (a system turn is created if the conversation has none).
// role:"tool" turns render inside a user turn as
// <tool_response>\n{content}\n</tool_response>; consecutive tool turns are
// merged into one user turn (official Qwen3 chat-template behavior).
// `model_has_think`: true ONLY for reasoning models whose CHAT TEMPLATE uses
// <think> (crown, 27B, Qwen3-dense). FALSE for non-reasoning models (e.g.
// Qwen3-Coder) → NO <think> block. NOTE: gate on the template, NOT on <think>
// token presence — Qwen3-Coder HAS <think> in its vocab but never uses it, and
// injecting the empty-think convention makes it free-continue ("Human:" turns).
// `reasoning_preamble`: pre-resolved reasoning-effort sentence (Qwen3.8-style
// templates) rendered at the TOP of the system block — before the tools
// preamble / system content — exactly where the vendor template puts it. A
// standalone system turn is created if the conversation has none. Empty =
// no injection (Qwen3.6-and-earlier templates, effort "medium", thinking off).
std::string build_chatml_prompt(std::span<const ChatTurn> turns,
                                bool add_generation_prompt = true,
                                bool enable_thinking      = true,
                                std::string_view tools_json = {},
                                bool model_has_think      = true,
                                std::string_view reasoning_preamble = {});

// P3a: Llama 3.x Instruct chat template (llama.cpp llama-chat.cpp LLAMA_3):
//   <|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>
// per turn; add_generation_prompt opens the assistant turn with
//   <|start_header_id|>assistant<|end_header_id|>\n\n
// BOS (<|begin_of_text|>) is NOT a template literal — add_bos_token=true
// prepends it at encode(). No <think> convention; tools are NOT supported on
// the llama path in v1 (Engine::chat errors if tools_json is non-empty).
std::string build_llama3_prompt(std::span<const ChatTurn> turns,
                                bool add_generation_prompt = true);

// Wave-1: Mistral / Nemo / Small + Devstral / Codestral chat template
// (llama.cpp LLM_CHAT_TEMPLATE_MISTRAL_V*):
//   <s>[INST] {user} [/INST] {assistant}</s>[INST] {user2} [/INST] ...
// BOS (<s>) is prepended by encode() (add_bos_token), NOT a template literal
// (avoids a double-BOS). A leading "system" turn folds into the first [INST]:
//   * default (v1/v2)            : "{system}\n\n{user}"
//   * system_prompt_block (v3+)  : "[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT]\n\n{user}"
//     (Devstral / Mistral-v3+; gated on DenseConfig::mistral_sysprompt).
// No trailing generation token: the model continues right after [/INST].
std::string build_mistral_prompt(std::span<const ChatTurn> turns,
                                 bool add_generation_prompt = true,
                                 bool system_prompt_block   = false);

// Wave-1: DeepSeek-R1-Distill (Qwen2 & Llama) reasoning chat template. Uses the
// fullwidth sentinels (U+FF5C bar) directly rather than ChatML:
//   {system}<｜User｜>{u1}<｜Assistant｜>{a1}<｜end▁of▁sentence｜><｜User｜>{u2}...
// A leading "system" turn is emitted verbatim (no wrapper) at the very front,
// matching the R1-Distill template. <｜begin▁of▁sentence｜> BOS is prepended by
// encode() (add_bos_token), NOT a template literal. add_generation_prompt opens
// the assistant turn with <｜Assistant｜>; enable_thinking appends <think>\n
// (R1 emits a reasoning trace), else the empty-think convention <think>\n\n</think>\n\n.
std::string build_deepseek_prompt(std::span<const ChatTurn> turns,
                                  bool add_generation_prompt = true,
                                  bool enable_thinking       = true);

// Phase 6: DeepSeek-V4-Flash (arch `deepseek4`). There is NO official jinja
// template for this model — the authoritative encoder is the repo's
// `encoding/encoding_dsv4.py`, and build_deepseek4_prompt() is a port of its
// `encode_messages()`. It is NOT the R1-Distill template above: the sentinels
// are the same but the thinking convention, the reasoning-effort preamble, the
// tools block and the tool-result merging are all different.
//
// Ported: BOS, reasoning-effort preamble, system/user/assistant rendering,
// consecutive-user merging, role:"tool" → <tool_result> merging into the
// preceding user turn, the DSML tools block, and the per-turn
// <｜Assistant｜>+<think>/</think> transition (drop_thinking = true, the
// upstream default).
// NOT ported (see the .cpp header comment for the full list): drop_thinking=false
// (needs per-turn reasoning_content), `latest_reminder`, DS_TASK_SP_TOKENS,
// `developer`, `response_format`, `context`, `wo_eos`, and re-encoding a prior
// assistant turn's STRUCTURED tool_calls (its content is emitted verbatim).
struct DeepSeek4ChatOptions {
    bool             thinking         = true;   // thinking_mode "thinking" vs "chat"
    std::string_view reasoning_effort = "low";  // "low" | "high" | "max"; thinking only
    bool             add_bos          = true;   // encode_messages(add_default_bos_token)
    std::string_view tools_json;                // OpenAI `tools` array; empty → no tools
};
std::string build_deepseek4_prompt(std::span<const ChatTurn> turns,
                                   bool add_generation_prompt = true,
                                   const DeepSeek4ChatOptions& opt = {});

// Phase 6: the matching output parser — a port of
// `parse_message_from_completion_text()` / `parse_tool_calls()`. `text` is one
// raw assistant completion (including its trailing <｜end▁of▁sentence｜>).
// A malformed completion sets `error` and leaves the other fields undefined:
// the upstream parser raises, so this must NOT invent a plausible parse.
struct DeepSeek4Completion {
    std::string content;
    std::string reasoning_content;
    std::string tool_calls_json;   // OpenAI-format JSON array; "[]" when none
    std::string error;
};
DeepSeek4Completion parse_deepseek4_completion(std::string_view text, bool thinking);

// IBM Granite-3.x: <|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>
std::string build_granite_prompt(std::span<const ChatTurn> turns,
                                 bool add_generation_prompt = true);

// Gemma 4: <|turn>{role}\n{content}<turn|>\n per turn (assistant→model);
// <|turn>model\n to generate. BOS added by encode(); stop on <turn|>.
std::string build_gemma_prompt(std::span<const ChatTurn> turns,
                               bool add_generation_prompt = true);

// gpt-oss (OpenAI Harmony). Emits the canned Harmony SYSTEM message (identity +
// knowledge cutoff + current date + reasoning effort + valid channels); a caller
// role=="system" turn maps to the Harmony DEVELOPER message; user/assistant turns
// follow (prior assistant turns render only their final channel). add_generation_prompt
// opens "<|start|>assistant" (the model emits its own <|channel|>…<|message|>… and
// closes with <|return|>). No BOS. reasoning_effort ∈ {low,medium,high}.
// `tools_json` (raw OpenAI `tools` array): empty → byte-identical to the old
// 3-arg behavior. Non-empty → a Harmony `namespace functions { … }` block is
// appended to the DEVELOPER message; the model emits its call on the commentary
// channel (`to=functions.NAME … <|message|>{json}<|call|>`), which Engine::chat
// translates back into a canonical <tool_call> block.
std::string build_harmony_prompt(std::span<const ChatTurn> turns,
                                 bool add_generation_prompt = true,
                                 std::string_view reasoning_effort = "low",
                                 std::string_view tools_json = {});

}  // namespace ie
