// include/ie/model_config.hpp — arch-keyed GGUF metadata → model config.
// P2: reads `<arch>.*` keys the way the deleted Rust prototype did; one
// reader serves every standard dense transformer (qwen3 now, llama later).
#pragma once
#include "ie/gguf.hpp"
#include <cstdint>
#include <string>
#include <string_view>

namespace ie {

enum class ModelArch { kQwen35Moe, kQwen3Dense, kQwen35Dense, kLlama3, kQwen3Moe, kQwen3Next, kGemma4, kGptOss, kDeepSeek4, kInkling, kLaguna, kGlmDsa, kQwen4Exp, kGlm5Next, kHyv4, kDeepSeek41, kMimo26, kUnknown };

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

// DeepSeek-V4-Flash (`deepseek4`): MLA-style compressed attention (q_lora /
// kv_lora + a LoRA'd output projection) + a "lightning indexer" that selects
// top_k KV positions per query + per-layer HYPER-CONNECTIONS (n-way residual
// mixing with a Sinkhorn-normalised routing matrix) + a 256-expert MoE with one
// shared expert. NOTHING in this engine can run it yet — this reader exists so
// the model is INSPECTABLE (preflight / tooling) and so the GGUF's own numbers
// are recorded verbatim instead of being re-derived later from a blog post.
//
// Key names verified 2026-08-01 against the real header of
// DeepSeek-V4-Flash-0731-UD-Q3_K_XL-00001-of-00004.gguf (74 KVs, metadata-only
// shard). The GGUF is a THIRD-PARTY (Unsloth) conversion: these key names are
// the converter's invention, not an upstream llama.cpp arch — upstream has no
// `deepseek4` at all. Where the KV set and the HF config.json disagree the GGUF
// wins here, because the GGUF is what we load.
//
// Deliberately NOT read (absent from this GGUF, present in HF config.json):
// feed_forward_length (there is no dense FFN — hence no read_dense_config
// reuse), num_nextn_predict_layers (MTP head), and every dspark_* key.
struct DeepSeek4Config {
    // -- dims ------------------------------------------------------------
    uint32_t n_layers   = 0;      // deepseek4.block_count                    (43)
    uint32_t hidden     = 0;      // deepseek4.embedding_length               (4096)
    uint32_t n_q_heads  = 0;      // deepseek4.attention.head_count           (64)
    uint32_t n_kv_heads = 0;      // deepseek4.attention.head_count_kv        (1 — MLA latent)
    uint32_t head_dim   = 0;      // deepseek4.attention.key_length           (512)
    uint32_t v_head_dim = 0;      // deepseek4.attention.value_length         (512)
    uint32_t vocab      = 0;      // tokenizer.ggml.tokens array length       (129280)
    uint32_t ctx_train  = 0;      // deepseek4.context_length                 (1048576)
    float    rms_eps    = 1e-6f;  // deepseek4.attention.layer_norm_rms_epsilon

    // -- MLA low-rank projections ----------------------------------------
    uint32_t q_lora_rank    = 0;  // deepseek4.attention.q_lora_rank          (1024)
    uint32_t o_lora_rank    = 0;  // deepseek4.attention.output_lora_rank     (1024)
    uint32_t o_groups       = 0;  // deepseek4.attention.output_group_count   (8)
    uint32_t sliding_window = 0;  // deepseek4.attention.sliding_window       (128)

    // -- lightning indexer (top-k KV selection) ---------------------------
    uint32_t indexer_n_heads = 0; // deepseek4.attention.indexer.head_count   (64)
    uint32_t indexer_head_dim = 0;// deepseek4.attention.indexer.key_length   (128)
    uint32_t indexer_top_k   = 0; // deepseek4.attention.indexer.top_k        (512)

    // Per-layer KV compression schedule. VERIFIED length 46 in the real file
    // while block_count is 43 — a 3-entry surplus this reader does NOT try to
    // explain. Observed values: [0,0, then 4,128 alternating, …, 4, 0,0,0].
    // Stored VERBATIM (no reindexing, no truncation): mapping entry→layer is
    // UNKNOWN and must be settled against the reference module before any
    // forward consumes it.
    std::vector<int32_t> compress_ratios;   // deepseek4.attention.compress_ratios
    float compress_rope_theta = 0.f;        // deepseek4.attention.compress_rope_freq_base (160000)

    // -- RoPE (YaRN) ------------------------------------------------------
    uint32_t rope_dim      = 0;       // deepseek4.rope.dimension_count       (64 = qk_rope_head_dim)
    float    rope_theta    = 10000.f; // deepseek4.rope.freq_base
    float    rope_freq_scale = 1.0f;  // 1 / deepseek4.rope.scaling.factor    (1/16)
    uint32_t rope_orig_ctx = 0;       // deepseek4.rope.scaling.original_context_length (65536)
    float    rope_ext_factor = 0.0f;  // 1.0 when scaling.type == "yarn" (mirrors GptOssConfig)
    float    yarn_beta_fast = 0.f;    // deepseek4.rope.scaling.yarn_beta_fast (32)
    float    yarn_beta_slow = 0.f;    // deepseek4.rope.scaling.yarn_beta_slow (1)

    // -- MoE --------------------------------------------------------------
    uint32_t n_experts        = 0;    // deepseek4.expert_count               (256)
    uint32_t n_experts_used   = 0;    // deepseek4.expert_used_count          (6)
    uint32_t n_shared_experts = 0;    // deepseek4.expert_shared_count        (1)
    uint32_t expert_ffn       = 0;    // deepseek4.expert_feed_forward_length (2048)
    // deepseek4.expert_gating_func — the real file carries 4, which is OUTSIDE
    // llama.cpp's llama_expert_gating_func_type (0 none/1 softmax/2 sigmoid/
    // 3 softmax_weight). ASSUMPTION (not verified against converter source):
    // 4 == the "sqrtsoftplus" scoring_func in the HF config.json. Kept raw.
    uint32_t expert_gating_func  = 0;
    float    expert_weights_scale = 1.0f;  // deepseek4.expert_weights_scale  (1.5)
    bool     expert_weights_norm  = false; // deepseek4.expert_weights_norm   (true)

    // Per-layer SwiGLU clamp limits (length n_layers; 10.0 everywhere in the
    // real file, matching HF swiglu_limit). Separate schedules for the routed
    // experts and the shared expert.
    std::vector<float> swiglu_clamp_exp;    // deepseek4.swiglu_clamp_exp   (43 x f32)
    std::vector<float> swiglu_clamp_shexp;  // deepseek4.swiglu_clamp_shexp (43 x f32)

    // -- hyper-connections + hash routing ---------------------------------
    uint32_t hc_count            = 0;      // deepseek4.hyper_connection.count               (4)
    uint32_t hc_sinkhorn_iters   = 0;      // deepseek4.hyper_connection.sinkhorn_iterations (20)
    float    hc_eps              = 0.f;    // deepseek4.hyper_connection.epsilon             (1e-6)
    uint32_t hash_layer_count    = 0;      // deepseek4.hash_layer_count                     (3)
};

// GLM-5.2 (`glm-dsa`, 256x22B, 751 B params @ 2.244 bpw). DeepSeek-style MLA
// (q/kv LoRA + absorb-form k_b/v_b) + a DSA "indexer" that selects the top-2048
// KV positions + 256-expert top-8 MoE with one shared expert + a trailing
// NextN/MTP block. Every field below was read out of the real GGUF header; see
// docs/glm52/GLM-5.2_ARCH_SPEC.md.
//
// Deliberately NOT modelled on DeepSeek4Config's extras: this arch has no
// hyper-connections, no hash routing, no compress_ratios, no sliding window, no
// SwiGLU clamp schedule, and no YaRN keys. Reading those would invent facts.
struct GlmDsaConfig {
    // -- dims --------------------------------------------------------------
    uint32_t n_layers   = 0;      // glm-dsa.block_count                       (79)
    uint32_t hidden     = 0;      // glm-dsa.embedding_length                  (6144)
    uint32_t n_q_heads  = 0;      // glm-dsa.attention.head_count              (64)
    uint32_t n_kv_heads = 0;      // glm-dsa.attention.head_count_kv           (1 — MLA latent)
    uint32_t head_dim   = 0;      // glm-dsa.attention.key_length              (576 = 512+64)
    uint32_t v_head_dim = 0;      // glm-dsa.attention.value_length            (512)
    uint32_t ffn        = 0;      // glm-dsa.feed_forward_length               (12288, dense blocks)
    uint32_t vocab      = 0;      // tokenizer.ggml.tokens array length        (154880)
    uint32_t ctx_train  = 0;      // glm-dsa.context_length                    (1048576)
    float    rms_eps    = 1e-5f;  // glm-dsa.attention.layer_norm_rms_epsilon

    // -- MLA ---------------------------------------------------------------
    // key_length_mla = qk_nope(192) + qk_rope(64); value_length_mla = v_head(256).
    // attn_k_b is [qk_nope, kv_lora, n_head]; attn_v_b is [kv_lora, v_head, n_head].
    uint32_t q_lora_rank   = 0;   // glm-dsa.attention.q_lora_rank             (2048)
    uint32_t kv_lora_rank  = 0;   // glm-dsa.attention.kv_lora_rank            (512)
    uint32_t key_len_mla   = 0;   // glm-dsa.attention.key_length_mla          (256)
    uint32_t value_len_mla = 0;   // glm-dsa.attention.value_length_mla        (256)

    // -- DSA indexer (sparse attention) ------------------------------------
    // Present on only 22 of 79 blocks; the index is shared by the blocks that
    // follow one that has it ("IndexShare", ik's -fidx).
    uint32_t indexer_n_heads  = 0;  // glm-dsa.attention.indexer.head_count    (32)
    uint32_t indexer_head_dim = 0;  // glm-dsa.attention.indexer.key_length    (128)
    uint32_t indexer_top_k    = 0;  // glm-dsa.attention.indexer.top_k         (2048)

    // -- RoPE (plain; no scaling keys present in the file) -----------------
    uint32_t rope_dim   = 0;        // glm-dsa.rope.dimension_count            (64)
    float    rope_theta = 10000.f;  // glm-dsa.rope.freq_base                  (8e6)

    // -- MoE ---------------------------------------------------------------
    uint32_t leading_dense = 0;     // glm-dsa.leading_dense_block_count       (3)
    uint32_t n_experts        = 0;  // glm-dsa.expert_count                    (256)
    uint32_t n_experts_used   = 0;  // glm-dsa.expert_used_count               (8)
    uint32_t n_shared_experts = 0;  // glm-dsa.expert_shared_count             (1)
    uint32_t expert_ffn       = 0;  // glm-dsa.expert_feed_forward_length      (2048)
    uint32_t expert_gating_func   = 0;     // glm-dsa.expert_gating_func       (2 = sigmoid)
    float    expert_weights_scale = 1.0f;  // glm-dsa.expert_weights_scale     (2.5)
    bool     expert_weights_norm  = false; // glm-dsa.expert_weights_norm      (true)
    // Flat top-k: both are 1, i.e. NO group-limited routing (unlike DeepSeek).
    uint32_t expert_group_count = 1;       // glm-dsa.expert_group_count       (1)
    uint32_t expert_group_used  = 1;       // glm-dsa.expert_group_used_count  (1)

    // -- NextN / MTP -------------------------------------------------------
    // block_count COUNTS the MTP block (79 = 78 transformer + 1 NextN). The MTP
    // block carries a COMPLETE MLA stack and its OWN 256-expert bank, and reuses
    // token_embd / output.weight (nextn.embed_tokens and nextn.shared_head_head
    // are absent from the file).
    uint32_t nextn_predict_layers = 0;     // glm-dsa.nextn_predict_layers     (1)

    // Real transformer blocks for a standard forward (excludes NextN/MTP).
    uint32_t n_transformer_layers() const {
        return n_layers - nextn_predict_layers;
    }
    // Blocks [0, leading_dense) use a dense FFN of width `ffn`; the rest are MoE.
    bool is_dense_layer(uint32_t il) const { return il < leading_dense; }

    // Which blocks carry DSA indexer tensors. Derived from the real file's
    // tensor table: {0,1,2} then every 4th from 6 to 74, plus the MTP block 78 —
    // which collapses to (il < 3 || il % 4 == 2). This is a DERIVATION, and the
    // loader still binds indexer tensors by PRESENCE (they are optional), so a
    // future GLM revision with a different schedule cannot silently mis-bind.
    bool has_indexer(uint32_t il) const { return il < 3 || (il % 4) == 2; }
};

// Fills `out` for `glm-dsa`. Bespoke, like the deepseek4 reader: read_dense_config
// would "succeed" and silently set head_dim = attention.key_length = 576, the MLA
// latent width, which is not a head dimension at all. REQUIRED: block_count,
// embedding_length, attention.head_count, attention.head_count_kv,
// attention.key_length, attention.kv_lora_rank, expert_count/used_count/
// feed_forward_length, tokens.
std::string read_glmdsa_config(const GgufReader& g, GlmDsaConfig& out);

// -- GLM-5.3-Flash (glm5next) — hybrid KDA + DSA/MLA, Sinkhorn HC, 288-MoE --
// Cousin of GlmDsaConfig with: a PER-LAYER attention map (head_count_kv is a
// 46-entry array; 1 = DSA/MLA full-attn layer, 0 = KDA linear layer), KDA
// hparams, Sinkhorn hyper-connections, per-layer swiglu clamps, NOPE-only
// attention (rope.dimension_count = 0 — no rope anywhere), and a gated
// compressor on the indexer (kpool).
struct Glm5NextConfig {
    uint32_t n_layers   = 0;      // glm5next.block_count                      (46)
    uint32_t hidden     = 0;      // glm5next.embedding_length                 (4096)
    uint32_t n_q_heads  = 0;      // glm5next.attention.head_count             (64)
    uint32_t ffn        = 0;      // glm5next.feed_forward_length              (12288)
    uint32_t vocab      = 0;      // glm5next.vocab_size                       (154880)
    uint32_t ctx_train  = 0;      // glm5next.context_length                   (1048576)
    float    rms_eps    = 1e-5f;  // glm5next.attention.layer_norm_rms_epsilon
    float    ln_eps     = 1e-6f;  // glm5next.attention.layer_norm_epsilon (indexer k LayerNorm)

    // Per-layer attention type: attn_kind[il] != 0 => DSA/MLA full attention,
    // 0 => KDA linear layer. From the head_count_kv array.
    std::vector<int32_t> attn_kind;
    bool is_full_attn(uint32_t il) const {
        return il < attn_kind.size() && attn_kind[il] != 0;
    }

    // -- MLA (full-attn layers; NOPE-only, no rope) ------------------------
    uint32_t q_lora_rank   = 0;   // glm5next.attention.q_lora_rank            (1536)
    uint32_t kv_lora_rank  = 0;   // glm5next.attention.kv_lora_rank           (512)
    uint32_t key_len       = 0;   // glm5next.attention.key_length             (512)
    uint32_t value_len     = 0;   // glm5next.attention.value_length           (512)
    uint32_t key_len_mla   = 0;   // glm5next.attention.key_length_mla         (256)
    uint32_t value_len_mla = 0;   // glm5next.attention.value_length_mla       (256)

    // -- KDA (linear layers) -----------------------------------------------
    uint32_t kda_head_dim  = 0;   // glm5next.kda.head_dim                     (128)
    float    kda_gate_lower_bound = 0.f;  // glm5next.kda.gate_lower_bound     (-5.0)
    uint32_t conv_kernel   = 0;   // glm5next.ssm.conv_kernel                  (4)

    // -- DSA indexer + gated compressor ------------------------------------
    uint32_t indexer_n_heads  = 0;  // glm5next.attention.indexer.head_count   (32)
    uint32_t indexer_head_dim = 0;  // glm5next.attention.indexer.key_length   (128)
    uint32_t indexer_top_k    = 0;  // glm5next.attention.indexer.top_k        (2048)
    uint32_t indexer_kpool    = 0;  // glm5next.attention.indexer.kpool        (4)

    // -- Hyper-connections (Sinkhorn — the deepseek4 flavor) ---------------
    uint32_t hc_count = 0;          // glm5next.hyper_connection.count         (4)
    uint32_t hc_sinkhorn_iters = 0; // glm5next.hyper_connection.sinkhorn_iterations (20)
    float    hc_eps = 1e-7f;        // glm5next.hyper_connection.epsilon

    // -- MoE ----------------------------------------------------------------
    uint32_t leading_dense    = 0;  // glm5next.leading_dense_block_count      (3)
    uint32_t n_experts        = 0;  // glm5next.expert_count                   (288)
    uint32_t n_experts_used   = 0;  // glm5next.expert_used_count              (8)
    uint32_t n_shared_experts = 0;  // glm5next.expert_shared_count            (1)
    uint32_t expert_ffn       = 0;  // glm5next.expert_feed_forward_length     (2048)
    uint32_t shexp_ffn        = 0;  // glm5next.expert_shared_feed_forward_length (2048)
    uint32_t expert_gating_func   = 0;     // (2 = sigmoid)
    float    expert_weights_scale = 1.0f;  // (2.5)
    bool     expert_weights_norm  = false; // (true)
    std::vector<float> swiglu_clamp_exp;   // per layer (10.0)
    std::vector<float> swiglu_clamp_shexp; // per layer (10.0)

    // -- NextN / MTP ---------------------------------------------------------
    uint32_t nextn_predict_layers = 0;     // (1; block_count counts it)
    uint32_t n_transformer_layers() const {
        return n_layers - nextn_predict_layers;
    }
    bool is_dense_layer(uint32_t il) const { return il < leading_dense; }
};

std::string read_glm5next_config(const GgufReader& g, Glm5NextConfig& out);

// Tencent Hy4-preview (`hyv4`, 256x29B): DeepSeek-V3-style MLA-as-MQA with a
// ROPED trailing 64 (interleaved / LLAMA_ROPE_TYPE_NORM — key 576 = kv_lora
// 512 + rope 64), attention sinks + a sigmoid attention gate, iHC
// hyper-connections (pre/post only, NO comb/Sinkhorn, magnitude-scaled post,
// learned output_hc head), DSA indexer (32x128 top-k 2048, LayerNorm+bias k,
// per-layer is_full sharing), 256-expert top-8 sigmoid MoE + 1 unclamped
// shared expert, leading dense block 0, no MTP. Field names deliberately
// mirror Glm5NextConfig so the forked model code reads identically.
struct Hyv4Config {
    uint32_t n_layers   = 0;      // hyv4.block_count                          (78)
    uint32_t hidden     = 0;      // hyv4.embedding_length                     (6144)
    uint32_t n_q_heads  = 0;      // hyv4.attention.head_count                 (64)
    uint32_t ffn        = 0;      // hyv4.feed_forward_length                  (18432)
    uint32_t vocab      = 0;      // hyv4.vocab_size                           (120832)
    uint32_t ctx_train  = 0;      // hyv4.context_length                       (1048576)
    float    rms_eps    = 1e-5f;  // hyv4.attention.layer_norm_rms_epsilon (1e-5; also the
                                  // indexer k LayerNorm eps — the reference plumbs rms_eps there)
    float    ln_eps     = 1e-5f;

    bool is_full_attn(uint32_t il) const { return il < n_layers; }   // every block is MLA

    // -- MLA ----------------------------------------------------------------
    uint32_t q_lora_rank   = 0;   // hyv4.attention.q_lora_rank                (2048)
    uint32_t kv_lora_rank  = 0;   // hyv4.attention.kv_lora_rank               (512)
    uint32_t key_len       = 0;   // hyv4.attention.key_length                 (576 = 512 + rope 64)
    uint32_t value_len     = 0;   // hyv4.attention.value_length               (512)
    uint32_t key_len_mla   = 0;   // hyv4.attention.key_length_mla             (256)
    uint32_t value_len_mla = 0;   // hyv4.attention.value_length_mla           (256)
    // interleaved (NORM) rope on the trailing rope_dim of q heads and kv rows
    uint32_t rope_dim   = 0;      // hyv4.rope.dimension_count                 (64)
    float    rope_theta = 1e7f;   // hyv4.rope.freq_base

    // -- DSA indexer --------------------------------------------------------
    uint32_t indexer_n_heads  = 0;  // hyv4.attention.indexer.head_count       (32)
    uint32_t indexer_head_dim = 0;  // hyv4.attention.indexer.key_length       (128)
    uint32_t indexer_top_k    = 0;  // hyv4.attention.indexer.top_k            (2048)
    uint32_t indexer_kpool    = 1;  // no compressor pooling on hyv4 (kept so the
                                    // n_sel = top_k + kpool - 1 identity reads the same)
    std::vector<uint8_t> indexer_is_full;   // hyv4.attention.indexer.is_full [n_layers]

    // -- iHC hyper-connections ---------------------------------------------
    uint32_t hc_count     = 0;    // hyv4.hyper_connection.count               (4)
    float    hc_eps       = 1e-6f;// hyv4.hyper_connection.epsilon
    float    hc_magnitude = 2.f;  // hyv4.hyper_connection.magnitude

    // -- MoE ----------------------------------------------------------------
    uint32_t leading_dense    = 0;  // hyv4.leading_dense_block_count          (1)
    uint32_t n_experts        = 0;  // hyv4.expert_count                       (256)
    uint32_t n_experts_used   = 0;  // hyv4.expert_used_count                  (8)
    uint32_t n_shared_experts = 0;  // hyv4.expert_shared_count                (1)
    uint32_t expert_ffn       = 0;  // hyv4.expert_feed_forward_length         (2048)
    uint32_t shexp_ffn        = 0;  // = expert_ffn * n_shared_experts
    uint32_t expert_gating_func   = 0;     // (2 = sigmoid)
    float    expert_weights_scale = 1.0f;  // (2.827)
    bool     expert_weights_norm  = false; // (true)
    std::vector<float> swiglu_clamp_exp;   // per layer (10.0, routed experts only)
    std::vector<float> swiglu_clamp_shexp; // filled +INF by the reader (unclamped)

    uint32_t nextn_predict_layers = 0;     // hyv4 ships no MTP block
    bool is_dense_layer(uint32_t il) const { return il < leading_dense; }
    uint32_t n_transformer_layers() const { return n_layers - nextn_predict_layers; }

    // Dead-compat fields: the model file is forked from glm5next.cpp and its
    // KDA/Sinkhorn branches still compile against these. They are UNREACHABLE
    // at runtime (is_full_attn() is always true; the iHC kernels ignore
    // sinkhorn_iters) — do not read them in new code.
    std::vector<int32_t> attn_kind;        // unused (every block is MLA)
    uint32_t kda_head_dim = 128;           // unused
    uint32_t conv_kernel  = 4;             // unused
    float    kda_gate_lower_bound = 0.f;   // unused
    uint32_t hc_sinkhorn_iters = 0;        // unused (iHC has no Sinkhorn)
};

std::string read_hyv4_config(const GgufReader& g, Hyv4Config& out);

// Qwen3.8-Flash-Next (`qwen4exp`, "A Preview of the Qwen4 Architecture",
// size_label 512x56B). Every value in the comments below was read out of the
// real UD-Q4_K_XL GGUF (2026-08-26) and cross-checked against the HF
// config.json; see docs/qwen4-flash-next-port-campaign.md.
//
// The structure this config must carry faithfully:
//  - 48 blocks, 3:1 Gated-DeltaNet : full-attention hybrid — full attention
//    ONLY on blocks il % 4 == 3 (full_attention_interval 4);
//  - full-attn blocks: 24 Q / 2 KV heads, head_dim 256, sigmoid output gate
//    FUSED into attn_q (out width = n_q_heads*head_dim*2 = 12288), partial
//    rotary 64/256 with INTERLEAVED mrope sections [11,11,10,0], plus a QSA
//    lightning indexer (4 heads, dim 128, top-k 2048) marked per-block by
//    attention.compress_ratios[il] == 4;
//  - DeltaNet blocks: dims IDENTICAL to Qwen3.8-27B (16 K-heads, 48 V-heads,
//    head_dim 128 == ssm.state_size, conv 4, fused attn_qkv width 10240) —
//    the qwen35 kernels reuse as-is;
//  - hyper-connections residual: 4 streams, low-rank-320 mix; per-block
//    hc_attn_*/hc_ffn_* tensors REPLACE attn_norm/ffn_norm, and a final
//    output_hc_{up,down,norm} replaces output_norm;
//  - MoE on EVERY block: 512 experts top-10 + 1 gated shared expert, both
//    ffn 640 (~6B active/token);
//  - PLE n-gram embeddings: bigram+trigram hash tables, 8 heads each (16
//    heads x 160 = ple_embed_dim 2560), 16 prime-sized vocabs summing
//    320,001,446 rows, injected at ONE block — ple.layers is 0-BASED ([1] →
//    blk.1; the HF ple_layer_ids [2] is 1-based, converter writes i-1);
//  - untied embeddings; NO MTP tensors in the GGUF (the convert script sets
//    no_mtp=True), so native-MTP spec decode cannot come from this file.
struct Qwen4ExpConfig {
    // -- dims --------------------------------------------------------------
    uint32_t n_layers  = 0;      // qwen4exp.block_count                      (48)
    uint32_t hidden    = 0;      // qwen4exp.embedding_length                 (2560)
    uint32_t vocab     = 0;      // tokenizer.ggml.tokens array length        (248320)
    uint32_t ctx_train = 0;      // qwen4exp.context_length                   (262144)
    float    rms_eps   = 1e-6f;  // qwen4exp.attention.layer_norm_rms_epsilon

    // -- full attention (blocks il % interval == interval-1) ---------------
    uint32_t n_q_heads  = 0;     // qwen4exp.attention.head_count             (24)
    uint32_t n_kv_heads = 0;     // qwen4exp.attention.head_count_kv          (2)
    uint32_t head_dim   = 0;     // qwen4exp.attention.key_length             (256; == value_length, enforced)
    uint32_t full_attention_interval = 0;  // qwen4exp.full_attention_interval (4)
    uint32_t rope_dim   = 0;         // qwen4exp.rope.dimension_count         (64 of 256: partial 0.25)
    float    rope_theta = 10000.f;   // qwen4exp.rope.freq_base               (1e7)
    // Interleaved mrope; entries are HALF-dims (2*sum == rope_dim), trailing
    // zero-pad is legitimate — the real file is [11,11,10,0].
    std::vector<int32_t> rope_sections;  // qwen4exp.rope.dimension_sections

    // -- QSA indexer (full-attn blocks only) -------------------------------
    uint32_t indexer_n_heads  = 0;  // qwen4exp.attention.indexer.head_count  (4)
    uint32_t indexer_head_dim = 0;  // qwen4exp.attention.indexer.key_length  (128)
    uint32_t indexer_top_k    = 0;  // qwen4exp.attention.indexer.top_k       (2048)
    std::vector<int32_t> compress_ratios;  // qwen4exp.attention.compress_ratios (48 entries: 4 on full blocks, 0 else)

    // -- Gated DeltaNet (all non-full blocks); dims == Qwen3.8-27B ---------
    uint32_t ssm_state       = 0;  // qwen4exp.ssm.state_size       (128 = linear head_dim)
    uint32_t ssm_conv_kernel = 0;  // qwen4exp.ssm.conv_kernel      (4)
    uint32_t ssm_k_heads     = 0;  // qwen4exp.ssm.group_count      (16 = linear_num_key_heads)
    uint32_t ssm_v_heads     = 0;  // qwen4exp.ssm.time_step_rank   (48 = linear_num_value_heads)
    uint32_t ssm_inner       = 0;  // qwen4exp.ssm.inner_size       (6144 = 48*128)

    // -- hyper-connections residual ----------------------------------------
    uint32_t hc_count    = 0;  // qwen4exp.hyper_connection.count             (4)
    uint32_t hc_low_rank = 0;  // qwen4exp.hyper_connection.low_rank          (320)

    // -- MoE (every block) -------------------------------------------------
    uint32_t n_experts         = 0;  // qwen4exp.expert_count                 (512)
    uint32_t n_experts_used    = 0;  // qwen4exp.expert_used_count            (10)
    uint32_t expert_ffn        = 0;  // qwen4exp.expert_feed_forward_length   (640)
    uint32_t shared_expert_ffn = 0;  // qwen4exp.expert_shared_feed_forward_length (640; 0 = none)

    // -- PLE n-gram embeddings (absent group = no PLE) ---------------------
    std::vector<int32_t> ple_layers;   // qwen4exp.ple.layers                 ([1], 0-based)
    uint32_t ple_ngram           = 0;  // qwen4exp.ple.ngram_size             (3)
    uint32_t ple_heads_per_ngram = 0;  // qwen4exp.ple.heads_per_ngram        (8)
    uint32_t ple_conv_kernel     = 0;  // qwen4exp.ple.conv_kernel            (4)
    uint32_t ple_embed_per_input = 0;  // qwen4exp.embedding_length_per_layer_input (160)
    uint32_t ple_eos_token       = 0;  // qwen4exp.ple.eos_token_id           (248044)
    uint32_t ple_image_token     = 0;  // qwen4exp.ple.image_token_id         (248056)
    std::vector<uint64_t> ple_layer_multipliers;  // qwen4exp.ple.layer_multipliers (ngram_size hash multipliers)
    std::vector<uint64_t> ple_head_offsets;       // qwen4exp.ple.head_offsets      (16, cumulative sums)
    std::vector<uint64_t> ple_head_vocab_sizes;   // qwen4exp.ple.head_vocab_sizes  (16 primes ~= 20M each)

    bool has_ple() const { return !ple_layers.empty(); }
    bool is_ple_layer(uint32_t il) const {
        for (int32_t l : ple_layers) if (uint32_t(l) == il) return true;
        return false;
    }
    bool is_full_attn(uint32_t il) const {
        return full_attention_interval && (il % full_attention_interval) == full_attention_interval - 1;
    }
    // 12 on the real file (48/4; exact because interval divides n_layers —
    // the reader rejects files where it does not).
    uint32_t n_full_attn_layers() const {
        return full_attention_interval ? n_layers / full_attention_interval : 0;
    }
    uint32_t n_linear_layers() const { return n_layers - n_full_attn_layers(); }
    // The n-gram lookup width: per-head 160 x 8 heads x (3-1) tables = 2560
    // (== HF ple_embed_dim, which has no GGUF key of its own).
    uint32_t ple_embed_dim() const {
        return ple_embed_per_input * ple_heads_per_ngram * (ple_ngram ? ple_ngram - 1 : 0);
    }
    // Total hash rows across all heads. The on-disk per_layer_token_embd pads
    // this up to a multiple of 128 (HF make_ngram_vocab_size_divisible_by —
    // the key is NOT in the GGUF; the config gate verifies the identity on the
    // real file: 320,001,446 → 320,001,536).
    uint64_t ple_total_vocab() const {
        return ple_head_offsets.empty() ? 0
             : ple_head_offsets.back() + ple_head_vocab_sizes.back();
    }
};

// Fills `out` for `qwen4exp` (Qwen3.8-Flash-Next). Bespoke: the arch is a
// hybrid, and a dense binding would take the full-attention geometry and
// silently drop the DeltaNet/HC/MoE/PLE structure carrying ~99% of the
// weights. REQUIRED: block_count, embedding_length, attention.head_count/
// head_count_kv/key_length, full_attention_interval, rope.dimension_count,
// all five ssm.* keys, hyper_connection.count/low_rank, expert_count/
// used_count/feed_forward_length, tokens. The indexer keys and the PLE group
// are each all-or-none. Returns "" on success or error text.
std::string read_qwen4exp_config(const GgufReader& g, Qwen4ExpConfig& out);

// Fills `out` for `deepseek4`. Bespoke (no read_dense_config reuse) because the
// arch has no feed_forward_length and no ordinary GQA head geometry. REQUIRED:
// block_count, embedding_length, attention.head_count, attention.head_count_kv,
// attention.key_length, expert_count/used_count/feed_forward_length, tokens.
// Everything else keeps its default when the key is absent. Returns "" on
// success or error text.
std::string read_deepseek4_config(const GgufReader& g, DeepSeek4Config& out);

// Inkling-Small (arch "inkling") — RECOGNITION + CONFIG ONLY, no runtime yet
// (docs/inkling/00_PORT_PLAN.md; reference at ~/models/inkling-reference/).
// Geometry ground-truthed against the real GGUF 2026-08-09.
struct InklingConfig {
    uint32_t n_layers = 0, hidden = 0, ffn_dense = 0;
    uint32_t n_q_heads = 0, head_dim = 0, v_head_dim = 0;
    std::vector<int32_t> n_kv_heads;       // PER LAYER (42 entries on the real file)
    uint32_t n_experts = 0, n_experts_used = 0, n_shared_experts = 0;
    uint32_t expert_ffn = 0;
    float    expert_weights_scale = 1.f;
    uint32_t expert_gating_func = 0;       // 2 = sigmoid on the real file
    uint32_t dense_layers = 0;             // leading dense blocks (2)
    uint32_t sliding_window = 0;           // 512
    std::vector<bool> swa_pattern;         // per layer: true = sliding
    // Relative attention.  `rel_extent` / `rel_extent_swa` are the two config
    // values; the one a LAYER uses depends on whether it is a sliding-window
    // layer — see `rel_extent_for(layer)`.  Verified on the real GGUF
    // 2026-08-09: `attn_rel_proj` has ne[0] = 512 on 35 layers and 1024 on
    // layers 5, 11, 17, 23, 29, 35, 41, matching `swa_pattern` exactly.  A
    // loader that applies one extent everywhere is wrong on 7 of 42 layers.
    uint32_t d_rel = 0, rel_extent = 0, rel_extent_swa = 0;
    uint32_t shortconv_kernel = 0;         // 4-tap Canon layers
    uint32_t vocab = 0, vocab_unpadded = 0, ctx_train = 0;
    float    rms_eps = 1e-6f, logit_scale_denom = 0.f;
    uint32_t log_scaling_n_floor = 0;
    float    log_scaling_alpha = 0.f;

    // The `attn_rel_proj` row length for layer L: the SWA extent on sliding
    // layers, the full extent otherwise.  Falls back to `rel_extent` when the
    // pattern is absent or short rather than guessing per-layer.
    uint32_t rel_extent_for(uint32_t layer) const noexcept {
        if (layer >= swa_pattern.size()) return rel_extent;
        return swa_pattern[layer] ? rel_extent_swa : rel_extent;
    }
    bool is_sliding(uint32_t layer) const noexcept {
        return layer < swa_pattern.size() && swa_pattern[layer];
    }
};
std::string read_inkling_config(const GgufReader& g, InklingConfig& out);

// Laguna S 2.1 (arch "laguna") — RECOGNITION + CONFIG ONLY, no runtime yet
// (docs/laguna/00_PORT_PLAN.md).  poolside 118B-A8B MoE: 48 layers (layer 0
// dense), 256 experts top-10 + 1 shared, GQA 48/8 at head_dim 128, alternating
// full/sliding attention 1:3, per-head attention output gating, sinks on the
// sliding layers.
struct LagunaConfig {
    uint32_t n_layers = 0, hidden = 0, ffn_dense = 0;
    uint32_t n_kv_heads = 0, head_dim = 0, v_head_dim = 0;
    uint32_t n_experts = 0, n_experts_used = 0, expert_ffn = 0, shared_expert_ffn = 0;
    uint32_t vocab = 0, ctx_train = 0, sliding_window = 0, dense_layers = 0;
    uint32_t expert_gating_func = 0;
    float    rms_eps = 1e-6f, expert_weights_scale = 1.f;
    bool     expert_weights_norm = false;

    // PER-LAYER QUERY HEAD COUNT — and it is also the LAYER-TYPE DISCRIMINATOR.
    // Verified on the real GGUF 2026-08-09: `laguna.attention.head_count` is an
    // ARRAY of 48 entries, not a scalar, and it takes exactly two values —
    // 48 on full-attention layers, 72 on sliding ones, in the pattern
    // [48,72,72,72] x 12.  **The file carries NO sliding_window_pattern key**,
    // so this array IS how a loader knows a layer's type.  (The HF config's
    // `layer_types` says the same thing; the GGUF encodes it here instead.)
    std::vector<int32_t> n_q_heads;

    // TWO ROPE CONFIGURATIONS, one per layer type — the trap this arch carries.
    // Full-attention layers: YaRN, theta 500000, and `rope.dimension_count` 64
    // of head_dim 128 rotate (the HF `partial_rotary_factor: 0.5`).
    // Sliding layers: plain rope, theta 10000, `rope.dimension_count_swa` 128
    // i.e. ALL dims rotate.  One table for the model is wrong on 3/4 of layers.
    float    rope_theta_full = 0.f, rope_theta_swa = 0.f;
    uint32_t rope_dim_full = 0, rope_dim_swa = 0;
    float    yarn_factor = 0.f, yarn_attn_factor = 1.f;
    float    yarn_beta_fast = 0.f, yarn_beta_slow = 0.f;
    uint32_t yarn_orig_ctx = 0;

    // A layer is SLIDING iff its query-head count is not the modal minimum.
    // Derived from the head_count array because the file has no pattern key.
    bool is_sliding(uint32_t layer) const noexcept {
        if (layer >= n_q_heads.size() || n_q_heads.empty()) return false;
        int32_t lo = n_q_heads[0];
        for (int32_t v : n_q_heads) lo = v < lo ? v : lo;
        return n_q_heads[layer] != lo;
    }
    uint32_t heads_for(uint32_t layer) const noexcept {
        return layer < n_q_heads.size() ? uint32_t(n_q_heads[layer]) : 0u;
    }
    uint32_t rope_dim_for(uint32_t layer) const noexcept {
        return is_sliding(layer) ? rope_dim_swa : rope_dim_full;
    }
    float rope_theta_for(uint32_t layer) const noexcept {
        return is_sliding(layer) ? rope_theta_swa : rope_theta_full;
    }
};
std::string read_laguna_config(const GgufReader& g, LagunaConfig& out);

}  // namespace ie
