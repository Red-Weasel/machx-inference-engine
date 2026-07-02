// include/ie/model_config.hpp — arch-keyed GGUF metadata → model config.
// P2: reads `<arch>.*` keys the way the deleted Rust prototype did; one
// reader serves every standard dense transformer (qwen3 now, llama later).
#pragma once
#include "ie/gguf.hpp"
#include <cstdint>
#include <string>
#include <string_view>

namespace ie {

enum class ModelArch { kQwen35Moe, kQwen3Dense, kQwen35Dense, kLlama3, kQwen3Moe, kQwen3Next, kGemma4, kGptOss, kUnknown };

// Identify the forward implementation from general.architecture.
// "qwen35moe" → crown (DeltaNet+full-attn+MoE); "qwen3"/"qwen2" → plain dense
// transformer; "qwen35" → Qwen3.6 dense-hybrid (DeltaNet+full-attn+dense MLP,
// e.g. Qwen3.6-27B); "llama" → Llama-3.x on the same dense path.
ModelArch detect_arch(const GgufReader& g);

// Metadata-key prefix for the dense reader ("qwen3" / "llama"); nullptr for
// non-dense arches. (qwen2 GGUFs report kQwen3Dense and read "qwen2.*" via
// read_dense_config_auto, so this helper is only used by callers that compose
// keys from a known ModelArch.)
const char* arch_key(ModelArch a);

// Is this arch served by DenseModel (src/model/dense_transformer.cpp)?
constexpr bool is_dense_arch(ModelArch a) {
    return a == ModelArch::kQwen3Dense || a == ModelArch::kLlama3;
}

struct DenseConfig {
    // Which CHAT template a dense GGUF wants. Several distinct families all
    // convert to GGUF arch `llama` and so all detect as kLlama3 (Mistral, Yi,
    // InternLM2, Nemotron, distill-Llama, …) — but they need DIFFERENT chat
    // templates. This field is read from the GGUF's own `tokenizer.chat_template`
    // string (the authoritative signal the <think> gate already trusts) so chat
    // dispatch can pick [INST] vs ChatML vs Llama-3 independent of arch_.
    // kAuto = "no chat_template recognised" → fall back to the existing
    // arch-based dispatch (ZERO behaviour change for every current model).
    // kDeepSeek = R1-Distill (Qwen2 & Llama) reasoning models: their template
    // uses the <｜User｜>/<｜Assistant｜> sentinels directly (NOT <|im_start|>),
    // so they would otherwise classify as kAuto. Wave-1 adds a dedicated builder.
    // kGranite = IBM Granite-3.x: <|start_of_role|>{role}<|end_of_role|>{content}
    // <|end_of_text|> turns (NOT ChatML); arch `granite`→kLlama3, so it needs its
    // own builder + stop token (<|end_of_text|>=eos id 0).
    enum class TemplateFamily { kAuto, kChatML, kLlama3, kMistral, kDeepSeek, kGranite };

    uint32_t n_layers   = 0;     // <arch>.block_count
    uint32_t hidden     = 0;     // <arch>.embedding_length
    uint32_t n_q_heads  = 0;     // <arch>.attention.head_count
    uint32_t n_kv_heads = 0;     // <arch>.attention.head_count_kv
    uint32_t head_dim   = 0;     // <arch>.attention.key_length, fallback hidden/n_q_heads
    uint32_t ffn        = 0;     // <arch>.feed_forward_length
    uint32_t vocab      = 0;     // tokenizer.ggml.tokens array length
    uint32_t rope_dim   = 0;     // <arch>.rope.dimension_count, fallback head_dim (full)
    float    rope_theta = 10000.f;  // <arch>.rope.freq_base
    float    rms_eps    = 1e-6f;    // <arch>.attention.layer_norm_rms_epsilon
    uint32_t ctx_train  = 0;     // <arch>.context_length (informational)
    // Which dense family this config is for — set by read_dense_config from the
    // arch string. The loader/forward/template key llama-specific deltas off it
    // (Q/K un-permute, rope freq factors, chat template). Default qwen3.
    ModelArch arch = ModelArch::kQwen3Dense;

    // Chat-template family (read from tokenizer.chat_template). kAuto → existing
    // arch-based dispatch (no current model carries a string that trips the
    // [INST]/llama-3/im_start matchers into a NEW branch — verified bit-exact).
    TemplateFamily template_family = TemplateFamily::kAuto;
    // 1 → Devstral/Mistral-v3+ [SYSTEM_PROMPT]…[/SYSTEM_PROMPT] variant.
    uint32_t mistral_sysprompt = 0;

    // Granite-3.x scalar multipliers (granite.*_scale). Defaults are no-ops so
    // every non-Granite model is byte-identical: 1.0 multiplies are identity and
    // attention_multiplier == 0 means "use the standard 1/sqrt(head_dim)".
    // NOTE: the dense forward CONSUMES these in a later (GPU-validated) Wave-1
    // task; Task 1 only plumbs the config fields. Absent keys → defaults.
    float embedding_multiplier = 1.0f;  // granite.embedding_scale
    float residual_multiplier  = 1.0f;  // granite.residual_scale
    float attention_multiplier = 0.0f;  // granite.attention_scale (0 → 1/sqrt(hd))
    float logits_scaling       = 1.0f;  // granite.logit_scale
};

// Pure classifier for the chat-template family from a GGUF's
// tokenizer.chat_template string. Host-testable (no GgufReader / no I/O).
// Order matters: [INST] is the most specific Mistral marker; a template can
// contain multiple markers (rare) so the most-discriminating wins. An empty or
// unrecognised string → kAuto (caller keeps its existing arch-based dispatch).
DenseConfig::TemplateFamily classify_template_family(std::string_view chat_template);

// Pure SWA-guard predicate. Sliding-window attention is UNSUPPORTED (no window
// arg in any attn op; the KV cache is full-context). A finite window strictly
// below the trained context means the GGUF expects windowed attention we cannot
// honour → returns true ("must hard-error"). window == 0 / absent, or
// window >= ctx_train (ctx_train==0 → unknown), means full attention → false.
bool sliding_window_unsupported(uint64_t window, uint32_t ctx_train);

// Fills `out` from `<arch>.*` keys. Returns "" on success, error text on
// any missing REQUIRED key (block_count, embedding_length, head_count,
// head_count_kv, feed_forward_length, tokens). Optional keys keep defaults.
std::string read_dense_config(const GgufReader& g, std::string_view arch,
                              DenseConfig& out);

// Same, but composes keys from the GGUF's own general.architecture (so both
// "qwen3" and "qwen2" dense GGUFs read correctly). Prefer this in callers.
std::string read_dense_config_auto(const GgufReader& g, DenseConfig& out);

// Qwen3.6 dense-hybrid (`qwen35`, e.g. Qwen3.6-27B): the full-attn/dense-MLP
// dims come from the standard `<arch>.*` keys (DenseConfig `dense`, where
// head_count/head_count_kv = the FULL-ATTENTION head counts) plus the
// gated-DeltaNet ("ssm") and layer-pattern keys below. Field→GGUF-key map and
// llama.cpp semantics (src/models/qwen35.cpp build_layer_attn_linear) noted
// inline. Linear (DeltaNet) layers are those where recurrent_layer(il) is true.
struct Qwen35Config {
    DenseConfig dense;                  // full-attn + dense-MLP dims, vocab, rope, eps

    uint32_t full_attn_interval = 4;    // <arch>.full_attention_interval
    uint32_t ssm_inner          = 0;    // <arch>.ssm.inner_size      (d_inner)
    uint32_t ssm_state          = 0;    // <arch>.ssm.state_size      (k head dim)
    uint32_t ssm_n_k_heads      = 0;    // <arch>.ssm.group_count     (num_k_heads)
    uint32_t ssm_n_v_heads      = 0;    // <arch>.ssm.time_step_rank  (num_v_heads)
    uint32_t ssm_conv_kernel    = 0;    // <arch>.ssm.conv_kernel
    uint32_t ssm_v_head_dim     = 0;    // derived: ssm_inner / ssm_n_v_heads
    int32_t  rope_sections[4]   = {0, 0, 0, 0};  // <arch>.rope.dimension_sections

    // NextN / Multi-Token-Prediction. `dense.n_layers` (= <arch>.block_count)
    // COUNTS these trailing MTP layers (Qwen3.6-27B: block_count 65 = 64
    // transformer + 1 NextN). They are used only for speculative decoding;
    // standard text-only decode runs n_transformer_layers().
    uint32_t nextn_predict_layers = 0;  // <arch>.nextn_predict_layers (optional)

    // Real transformer layers for a standard forward (excludes NextN/MTP).
    uint32_t n_transformer_layers() const {
        return dense.n_layers - nextn_predict_layers;
    }

    // Over the transformer layers [0, n_transformer_layers): layer il is a
    // linear-attention (gated DeltaNet) layer unless it is the every-interval-th
    // layer, which is full attention. Matches llama.cpp:
    // recurrent_layer_arr[i] = ((i+1) % full_attn_interval) != 0.
    bool recurrent_layer(uint32_t il) const {
        return ((il + 1) % full_attn_interval) != 0;
    }
};

// Fills `out` for the `qwen35` dense-hybrid. Reads the dense block via
// read_dense_config, then the REQUIRED ssm keys (inner_size, state_size,
// group_count, time_step_rank, conv_kernel). Returns "" on success or error
// text on any missing required key. rope.dimension_sections is optional.
std::string read_qwen35_config(const GgufReader& g, Qwen35Config& out);

// Qwen3 standard MoE (`qwen3moe`, e.g. Qwen3-Coder-30B-A3B): a plain dense
// transformer block (QK-norm attention + RoPE + GQA — same as Qwen3-dense)
// where the FFN is a top-k Mixture-of-Experts (NO DeltaNet, NO shared expert).
// `dense` carries the attention dims/vocab/rope/eps; the MoE fields below come
// from `qwen3moe.expert_*`. dense.ffn is unused (experts replace the dense FFN).
struct Qwen3MoeConfig {
    DenseConfig dense;            // attention dims, vocab, rope, eps
    uint32_t n_experts      = 0;  // qwen3moe.expert_count          (128)
    uint32_t n_experts_used = 0;  // qwen3moe.expert_used_count     (8)
    uint32_t expert_ffn     = 0;  // qwen3moe.expert_feed_forward_length (768)
};

// Fills `out` for `qwen3moe`. Reads the dense attention block, then the REQUIRED
// expert keys. feed_forward_length is optional for this arch (pure MoE → no
// dense FFN; absent or 0 is fine). Returns "" on success or error text.
std::string read_qwen3moe_config(const GgufReader& g, Qwen3MoeConfig& out);

// gpt-oss (OpenAI MoE, `gpt-oss`): top-k MoE + dense GQA attention with three
// non-standard pieces — per-head ATTENTION SINKS, ALTERNATING sliding-window
// (even layers, il%2==0) / full-dense attention, and top-4 SOFTMAX_WEIGHT
// gating. `dense` carries attn dims/vocab/rope/eps (note: q-dim = n_q_heads*
// head_dim = 4096 ≠ hidden 2880; head_dim 64; rope_freq_base 150000; YaRN above
// ctx_train 4096). Experts are MXFP4; non-experts Q8_0; biases on q/k/v/o,
// router, and all three expert GEMMs. Full arch spec: docs/gptoss_arch_spec_2026-06-26.md.
struct GptOssConfig {
    DenseConfig dense;            // attn dims, vocab, rope, eps
    uint32_t n_experts      = 0;  // gpt-oss.expert_count (32 @20b, 128 @120b)
    uint32_t n_experts_used = 0;  // gpt-oss.expert_used_count (4)
    uint32_t expert_ffn     = 0;  // gpt-oss.expert_feed_forward_length (2880)
    uint32_t sliding_window = 0;  // gpt-oss.attention.sliding_window (128, even layers)
    // YaRN RoPE (gpt-oss ships with YaRN as its deployed rope). The magnitude
    // scale mscale = 1+0.1*ln(1/freq_scale) is applied to cos/sin at ALL positions
    // (ggml rope_yarn), so plain rope is wrong even at pos 0 (q·k 1.81x too small).
    float    rope_freq_scale = 1.0f;  // 1 / gpt-oss.rope.scaling.factor (1/32). 1 = none.
    uint32_t rope_orig_ctx   = 0;     // gpt-oss.rope.scaling.original_context_length (4096)
    float    rope_ext_factor = 0.0f;  // 1.0 when scaling.type=="yarn" (else plain rope)
};
std::string read_gptoss_config(const GgufReader& g, GptOssConfig& out);

// Qwen3-Next-80B-A3B (`qwen3next`): the crown family at 80B scale — gated
// DeltaNet + gated full-attn (NEOX partial RoPE) + top-k MoE WITH a shared
// expert. Its config is the UNION of Qwen35Config (hybrid attn + DeltaNet ssm +
// full_attn_interval) and the MoE fields, plus the shared expert. Verified vs
// the bartowski Q4_K_M GGUF (2026-06-12): 48 layers, hidden 2048, head_dim 256,
// 16q/2kv, NEOX rope n_rot 64 θ1e7, full_attn_interval 4, ssm{inner 4096, state
// 128, group_count 16, time_step_rank 32, conv 4}, 512 experts / 10 used,
// E_ffn 512, shared-expert E_ffn 512. Unlike `qwen35` (27B), the per-layer FFN
// is MoE (not dense) and the attention rope is NEOX (not interleaved M-RoPE).
struct Qwen3NextConfig {
    Qwen35Config hybrid;             // dense attn dims + DeltaNet ssm + interval
    uint32_t n_experts         = 0;  // qwen3next.expert_count                    (512)
    uint32_t n_experts_used    = 0;  // qwen3next.expert_used_count               (10)
    uint32_t expert_ffn        = 0;  // qwen3next.expert_feed_forward_length      (512)
    uint32_t shared_expert_ffn = 0;  // qwen3next.expert_shared_feed_forward_length (512; 0 = none)
};

// Fills `out` for `qwen3next`. Reads the dense attention block + DeltaNet ssm
// keys (mirrors read_qwen35_config with the qwen3next prefix — copy-not-hoist to
// keep the validated 27B path byte-identical) + the REQUIRED expert keys + the
// optional shared-expert width. Returns "" on success or error text.
std::string read_qwen3next_config(const GgufReader& g, Qwen3NextConfig& out);

// Gemma 4 (`gemma4`): dense 31B + MoE 26B-A4B. Per-layer head geometry (sliding
// vs global) is the defining trait — see project_gemma4_arch.md / the plan doc.
// Verified vs gemma-4-26B-A4B-it-qat-q4_0 (2026-06-13). All quirks captured here
// so the model/forward can index per-layer without re-reading the GGUF.
struct GemmaConfig {
    uint32_t n_layers   = 0;   // gemma4.block_count
    uint32_t hidden     = 0;   // gemma4.embedding_length
    uint32_t n_q_heads  = 0;   // gemma4.attention.head_count (query heads, constant)
    uint32_t vocab      = 0;   // tokenizer.ggml.tokens length
    float    rms_eps    = 1e-6f;
    uint32_t ctx_train  = 0;
    uint32_t sliding_window = 0;        // gemma4.attention.sliding_window (1024)
    float    final_logit_softcap = 0.f; // gemma4.final_logit_softcapping (30)

    // Two head geometries, selected per-layer by is_swa.
    uint32_t head_dim_swa    = 0;       // attention.key_length_swa (256)
    uint32_t head_dim_global = 0;       // attention.key_length     (512)
    uint32_t n_rot_swa       = 0;       // rope.dimension_count_swa (256)
    uint32_t n_rot_global    = 0;       // rope.dimension_count     (512)
    float    rope_theta_swa    = 10000.f;   // rope.freq_base_swa
    float    rope_theta_global = 1000000.f; // rope.freq_base

    // Per-layer arrays (length n_layers). is_swa from sliding_window_pattern
    // (true=sliding/local, false=global/full); n_kv_heads from head_count_kv.
    std::vector<uint8_t>  is_swa;       // 1 = sliding-window layer
    std::vector<uint32_t> n_kv_heads;   // per-layer KV head count (8 swa / 2 global)

    // FFN: every layer has a shared dense GeGLU FFN (ffn) + (if MoE) a routed path.
    uint32_t ffn = 0;                   // gemma4.feed_forward_length (2112; shared/dense FFN)
    bool     is_moe = false;            // any layer carries ffn_gate_inp
    uint32_t n_experts = 0;             // gemma4.expert_count (128)
    uint32_t n_experts_used = 0;        // gemma4.expert_used_count (8)
    uint32_t expert_ffn = 0;            // gemma4.expert_feed_forward_length (704)

    // Convenience accessors.
    uint32_t head_dim(uint32_t L) const { return is_swa[L] ? head_dim_swa : head_dim_global; }
    uint32_t n_rot(uint32_t L)    const { return is_swa[L] ? n_rot_swa    : n_rot_global; }
    float    rope_theta(uint32_t L) const { return is_swa[L] ? rope_theta_swa : rope_theta_global; }
};

// Fills `out` for `gemma4`. Reads scalars + the two per-layer arrays
// (sliding_window_pattern, head_count_kv). Returns "" on success or error text.
std::string read_gemma4_config(const GgufReader& g, GemmaConfig& out);

}  // namespace ie
