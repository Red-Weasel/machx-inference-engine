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

    bool        ready() const noexcept { return !vocab_.empty(); }
    uint32_t    vocab_size() const noexcept { return uint32_t(vocab_.size()); }

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
struct ChatTurn {
    std::string role;          // "system", "user", "assistant", "tool"
    std::string content;
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
std::string build_chatml_prompt(std::span<const ChatTurn> turns,
                                bool add_generation_prompt = true,
                                bool enable_thinking      = true,
                                std::string_view tools_json = {},
                                bool model_has_think      = true);

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
