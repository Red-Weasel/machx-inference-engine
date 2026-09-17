// tests/unit/arch_template_test.cpp — Wave-1 Task 1 host unit test.
// Covers the PURE, GGUF-free logic added in this sprint:
//   * classify_template_family  — chat_template → TemplateFamily
//   * sliding_window_unsupported — SWA load-guard predicate
// Plus the arch-detect EXPECTATIONS that this sprint relies on (documented as
// static asserts on the enum so a future enum re-order can't silently break the
// kLlama3 routing the dense forward keys off of).
//
// Host-only: no GPU, no GGUF, no SYCL. Built like qwen3moe_pack_test —
// compiles model_config.cpp + gguf_reader.cpp + dtype.cpp directly and does NOT
// link ie_core. -UNDEBUG keeps the asserts live under the Release -DNDEBUG.
#undef NDEBUG
#include "ie/model_config.hpp"

#include <cassert>
#include <cstdio>
#include <string_view>

using ie::DenseConfig;
using TF = ie::DenseConfig::TemplateFamily;

int main() {
    int checks = 0;

    // ---- classify_template_family --------------------------------------
    // Mistral: the [INST] marker is decisive (the model detects as kLlama3 but
    // must NOT use the llama-3 header template).
    assert(ie::classify_template_family("{{ '[INST] ' + m + ' [/INST]' }}") == TF::kMistral);
    // Devstral / Mistral-v3+ also carry [SYSTEM_PROMPT] but still classify
    // kMistral (the sysprompt VARIANT flag is read separately in read_dense_config).
    assert(ie::classify_template_family("[SYSTEM_PROMPT]{{s}}[/SYSTEM_PROMPT][INST]{{m}}[/INST]") == TF::kMistral);
    // Llama-3 header template.
    assert(ie::classify_template_family("<|start_header_id|>user<|end_header_id|>") == TF::kLlama3);
    // ChatML (Qwen / Yi / InternLM2 / many distills).
    assert(ie::classify_template_family("<|im_start|>user\n{{m}}<|im_end|>") == TF::kChatML);
    // Unrecognised / empty → kAuto (engine keeps its existing arch dispatch).
    assert(ie::classify_template_family("") == TF::kAuto);
    assert(ie::classify_template_family("{{ bos_token }}{{ messages }}") == TF::kAuto);
    // Precedence: [INST] beats a stray im_start mention (most-specific wins).
    assert(ie::classify_template_family("<|im_start|> ... [INST] ...") == TF::kMistral);
    // DeepSeek-R1-Distill (Qwen2 & Llama): the <｜Assistant｜> fullwidth-bar
    // sentinel is decisive (its template uses these directly, not <|im_start|>).
    assert(ie::classify_template_family(
               "{{bos}}<\xef\xbd\x9c" "User" "\xef\xbd\x9c>{{m}}<\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c>")
           == TF::kDeepSeek);
    // A DeepSeek template that also stamps a stray im_start string still wins
    // DeepSeek (the sentinel is checked before ChatML).
    assert(ie::classify_template_family(
               "<|im_start|> ... <\xef\xbd\x9c" "Assistant" "\xef\xbd\x9c> ...")
           == TF::kDeepSeek);
    checks += 9;

    // ---- sliding_window_unsupported (SWA load-guard) -------------------
    // window strictly below the trained context → unsupported (must hard-error).
    assert(ie::sliding_window_unsupported(4096, 32768) == true);   // Mistral-7B-v0.1
    // window == ctx_train or above → effectively full attention → OK.
    assert(ie::sliding_window_unsupported(32768, 32768) == false);
    assert(ie::sliding_window_unsupported(131072, 32768) == false);
    // window 0 / absent → full attention → OK.
    assert(ie::sliding_window_unsupported(0, 32768) == false);
    // ctx_train unknown (0) → cannot prove the window bites → do NOT block.
    assert(ie::sliding_window_unsupported(4096, 0) == false);
    checks += 5;

    // ---- DenseConfig no-op defaults (Granite scalars must be identity) ---
    {
        DenseConfig c;
        assert(c.template_family == TF::kAuto);
        assert(c.mistral_sysprompt == 0);
        assert(c.embedding_multiplier == 1.0f);
        assert(c.residual_multiplier  == 1.0f);
        assert(c.attention_multiplier == 0.0f);   // 0 → "use 1/sqrt(head_dim)"
        assert(c.logits_scaling       == 1.0f);
        checks += 6;
    }

    // ---- arch-detect routing expectations (enum-stability guard) --------
    // The dense forward keys its llama-specific Q/K un-permute + RoPE path off
    // ModelArch::kLlama3 and treats phi3/granite as kLlama3. is_dense_arch must
    // include kLlama3 (Mistral/Yi/InternLM2/phi3/granite all route through it).
    static_assert(ie::is_dense_arch(ie::ModelArch::kLlama3), "kLlama3 must be a dense arch");
    static_assert(ie::is_dense_arch(ie::ModelArch::kQwen3Dense), "kQwen3Dense must be a dense arch");
    static_assert(!ie::is_dense_arch(ie::ModelArch::kQwen35Moe), "crown is not the dense path");
    checks += 1;

    std::printf("arch_template_test: all OK (%d checks)\n", checks);
    return 0;
}
