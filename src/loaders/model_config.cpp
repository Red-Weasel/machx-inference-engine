// src/loaders/model_config.cpp — arch detection + dense metadata reader.
// P2 Task 1: the only novel bit is `<arch>.` key composition; everything
// else is straight KV reads off the already-parsed GgufReader table.
#include "ie/model_config.hpp"

#include <cstdlib>
#include <limits>

namespace ie {

ModelArch detect_arch(const GgufReader& g) {
    const auto* kv = g.find_kv("general.architecture");
    if (!kv) return ModelArch::kUnknown;
    const auto a = kv->as_string();
    if (a == "qwen35moe") return ModelArch::kQwen35Moe;
    if (a == "qwen3moe")  return ModelArch::kQwen3Moe;   // standard MoE (Qwen3-Coder-30B)
    if (a == "qwen3next") return ModelArch::kQwen3Next;   // hybrid DeltaNet+attn+MoE (Qwen3-Next-80B)
    if (a == "gemma4")    return ModelArch::kGemma4;      // Gemma 4 (dense 31B + MoE 26B-A4B); per-layer head geom, sandwich norms, dual-path MoE
    if (a == "gpt-oss")   return ModelArch::kGptOss;      // OpenAI MoE (20b/120b): attention sinks + alternating SWA + top-4 MoE + MXFP4 experts
    if (a == "deepseek4") return ModelArch::kDeepSeek4;
    // GLM-5.2: MLA (q/kv LoRA + absorb-form k_b/v_b) + DSA top-k indexer on a
    // SUBSET of blocks + 256-expert top-8 MoE w/ shared expert + NextN/MTP block.
    // Routed experts are IQ2_KT, everything else Q6_0 (both ik_llama.cpp types).
    if (a == "glm-dsa")   return ModelArch::kGlmDsa;
    // Inkling-Small: qwen3moe-class batched MoE (256 experts top-6, 2 gate-routed
    // shared experts) + shortconv layers + attn_r/attn_rel_proj. RECOGNIZED so
    // tools refuse honestly instead of misdetecting (the ie-perplexity-on-ds4
    // failure class); NO runtime yet — docs/inkling/00_PORT_PLAN.md.
    if (a == "inkling")   return ModelArch::kInkling;
    // Laguna S 2.1: alternating full/sliding attention with PER-LAYER-TYPE rope
    // (YaRN + partial-rotary on full layers, plain on sliding), per-head attn
    // output gating, sinks, 256-expert top-10 MoE.  RECOGNIZED so tools refuse
    // honestly instead of misdetecting; NO runtime yet —
    // docs/laguna/00_PORT_PLAN.md.
    if (a == "laguna")    return ModelArch::kLaguna;   // DeepSeek-V4-Flash: MLA (q/kv/o LoRA) + lightning indexer top-k KV + hyper-connections + 256-expert MoE w/ shared expert
    // Qwen3.8-Flash-Next (Qwen4 preview): 48-layer 3:1 DeltaNet:QSA hybrid,
    // hyper-connections residual (4 streams, low-rank 320), 512-expert top-10
    // MoE + shared, PLE trigram embeddings at blk.1 (0-based), MTP absent
    // from GGUF exports (converter no_mtp=True). RECOGNIZED
    // so tools report honestly; runtime port in progress —
    // docs/qwen4-flash-next-port-campaign.md.
    if (a == "qwen4exp")  return ModelArch::kQwen4Exp;
    // GLM-5.3-Flash: hybrid KDA (per-channel gated delta recurrence) + DSA/MLA
    // full attn (NOPE-only, lightning indexer w/ gated compressor) on a
    // per-layer head_count_kv MAP, Sinkhorn hyper-connections, 288-expert
    // sigmoid MoE + shared, NextN/MTP block IN the GGUF. RECOGNIZED so tools
    // report honestly; runtime port in progress — docs/glm53/PORT_PLAN.md.
    if (a == "glm5next")  return ModelArch::kGlm5Next;
    if (a == "hyv4")      return ModelArch::kHyv4;      // Tencent Hy4-preview: MLA(roped 64)+sinks+gated attn, iHC, DSA indexer, 256-expert MoE

    if (a == "qwen35")    return ModelArch::kQwen35Dense;
    if (a == "qwen3")     return ModelArch::kQwen3Dense;
    // qwen2 rides the same DenseModel path; the forward gates QK-norm (Qwen3)
    // vs attention bias (Qwen2) on tensor presence, so one path serves both.
    if (a == "qwen2")     return ModelArch::kQwen3Dense;
    // Llama-3.x also rides DenseModel: GQA + SwiGLU + RMS, no QK-norm, with a
    // load-time Q/K un-permute + rope frequency factors (see dense_transformer).
    if (a == "llama")     return ModelArch::kLlama3;
    // Wave-1 free riders on the kLlama3 forward (un-permute + RoPE):
    //   phi3    — Phi-3/3.5/Phi-4: NEOX-permuted dense; GGUF pre-splits qkv and
    //             gate_up at convert; partial RoPE via phi3.rope.dimension_count.
    //   granite — Granite-3.x dense: llama-shape + 4 scalar multipliers
    //             (read into DenseConfig; consumed by the forward in a later task).
    if (a == "phi3")      return ModelArch::kLlama3;
    if (a == "granite")   return ModelArch::kLlama3;
    // NOTE (zero-change families): Mistral / Yi / InternLM2 / Nemotron(Llama) /
    // Baichuan-7B / DeepSeek-R1-Distill-Llama all convert to GGUF arch `llama`
    // → already kLlama3 above; DeepSeek-R1-Distill-Qwen converts to `qwen2` →
    // already kQwen3Dense. They route with NO detect change (scoping docs
    // 2026-06-12-arch-scoping-mistral / -tier1-sweep). Do NOT add redundant lines.
    return ModelArch::kUnknown;
}

DenseConfig::TemplateFamily classify_template_family(std::string_view ct) {
    using TF = DenseConfig::TemplateFamily;
    // [INST] is the decisive Mistral marker — it lets a Mistral GGUF (arch
    // `llama`, so detected kLlama3) pick the right prompt while Yi/distill-Llama
    // (also `llama` but llama-3/ChatML templates) keep theirs.
    if (ct.find("[INST]") != std::string_view::npos)              return TF::kMistral;
    if (ct.find("<|start_header_id|>") != std::string_view::npos) return TF::kLlama3;
    // DeepSeek-R1-Distill (Qwen2 & Llama) — the <｜Assistant｜> sentinel (note the
    // U+FF5C fullwidth bar, distinct from the ASCII '|' in ChatML/Llama-3 markers)
    // is decisive: its template uses these directly, not <|im_start|>. Checked
    // before ChatML because a DeepSeek template never carries <|im_start|> but is
    // unambiguous on its own sentinel.
    if (ct.find("<\xef\xbd\x9c" "Assistant\xef\xbd\x9c>") != std::string_view::npos) return TF::kDeepSeek;
    // Granite-3.x uses <|start_of_role|>...<|end_of_role|> role markers (ASCII),
    // never <|im_start|>; decisive on its own marker.
    if (ct.find("<|start_of_role|>") != std::string_view::npos)   return TF::kGranite;
    if (ct.find("<|im_start|>") != std::string_view::npos)        return TF::kChatML;
    return TF::kAuto;
}

bool sliding_window_unsupported(uint64_t window, uint32_t ctx_train) {
    // 0/absent → full attention. A window at/above the trained context is also
    // effectively full attention. ctx_train == 0 (unknown) → cannot prove the
    // window bites, so do not block on it.
    if (window == 0 || ctx_train == 0) return false;
    return window < uint64_t(ctx_train);
}

const char* arch_key(ModelArch a) {
    switch (a) {
        case ModelArch::kQwen3Dense: return "qwen3";
        case ModelArch::kLlama3:     return "llama";
        default:                     return nullptr;
    }
}

std::string read_dense_config(const GgufReader& g, std::string_view arch,
                              DenseConfig& out) {
    auto key = [&](const char* suffix) {
        return std::string(arch) + "." + suffix;
    };
    auto need_u32 = [&](const char* suffix, uint32_t& dst) -> bool {
        const auto* kv = g.find_kv(key(suffix));
        if (!kv) return false;
        dst = uint32_t(kv->as_uint());
        return true;
    };
    if (!need_u32("block_count", out.n_layers))            return "missing " + key("block_count");
    if (!need_u32("embedding_length", out.hidden))         return "missing " + key("embedding_length");
    if (!need_u32("attention.head_count", out.n_q_heads))  return "missing " + key("attention.head_count");
    if (!need_u32("attention.head_count_kv", out.n_kv_heads)) return "missing " + key("attention.head_count_kv");
    if (!need_u32("feed_forward_length", out.ffn))         return "missing " + key("feed_forward_length");
    if (!need_u32("attention.key_length", out.head_dim))   out.head_dim = out.hidden / out.n_q_heads;
    if (!need_u32("rope.dimension_count", out.rope_dim))   out.rope_dim = out.head_dim;  // full rotary
    need_u32("context_length", out.ctx_train);
    if (const auto* kv = g.find_kv(key("rope.freq_base")))  out.rope_theta = float(kv->as_float());
    if (const auto* kv = g.find_kv(key("attention.layer_norm_rms_epsilon"))) out.rms_eps = float(kv->as_float());

    // Sliding-window attention LOAD GUARD. We have no window arg in any attn op
    // (ops.hpp) and the KV cache is full-context, so a windowed checkpoint
    // (Mistral-7B-v0.1 only; v0.2+/Nemo/Small/Devstral/Codestral dropped SWA)
    // would silently full-attend past the window and diverge from the oracle.
    // Hard-error rather than lie. window 0/absent or >= ctx_train → no error.
    if (const auto* kv = g.find_kv(key("attention.sliding_window"))) {
        const uint64_t w = kv->as_uint();
        // gpt-oss has REAL per-layer sliding-window support (even layers windowed,
        // applied inside the FA softmax via the gemma-window mechanism), so it is
        // exempt from this guard — unlike the dense arches (Mistral-7B-v0.1) that
        // would silently full-attend past the window and diverge from the oracle.
        if (arch != "gpt-oss" && sliding_window_unsupported(w, out.ctx_train))
            return "sliding-window attention unsupported (window=" + std::to_string(w) +
                   " < ctx_train=" + std::to_string(out.ctx_train) +
                   "); use a v0.2+ / full-attention checkpoint";
    }

    // Chat-template family from the GGUF's own chat_template (authoritative; the
    // same signal the <think> gate trusts). kAuto if unrecognised → the engine
    // keeps its existing arch-based dispatch (no current model trips a branch).
    if (const auto* kv = g.find_kv("tokenizer.chat_template");
        kv && kv->type == GgufValueType::kString) {
        const auto ct = kv->as_string();
        out.template_family = classify_template_family(ct);
        if (out.template_family == DenseConfig::TemplateFamily::kMistral)
            out.mistral_sysprompt =
                (ct.find("[SYSTEM_PROMPT]") != std::string_view::npos) ? 1u : 0u;
    }

    // Granite-3.x scalar multipliers. Absent → no-op defaults (1.0 / attn 0.0),
    // so every non-Granite model is byte-identical. Key names VERIFIED against
    // granite-3.3-8b-instruct (2026-06-13): embedding_scale=12.0, residual_scale=
    // 0.22, attention.scale=0.0078125 (note the DOTTED namespace — llama.cpp's
    // LLM_KV_ATTENTION_SCALE), logit_scale=16.0.
    if (const auto* kv = g.find_kv(key("embedding_scale"))) out.embedding_multiplier = float(kv->as_float());
    if (const auto* kv = g.find_kv(key("residual_scale")))  out.residual_multiplier  = float(kv->as_float());
    if (const auto* kv = g.find_kv(key("attention.scale"))) out.attention_multiplier = float(kv->as_float());
    if (const auto* kv = g.find_kv(key("logit_scale")))     out.logits_scaling       = float(kv->as_float());

    const auto* toks = g.find_kv("tokenizer.ggml.tokens");
    if (!toks || toks->type != GgufValueType::kArray) return "missing tokenizer.ggml.tokens";
    out.vocab = uint32_t(toks->n_array);
    // Record the family so the loader/forward/template can branch on it. phi3
    // and granite ride the kLlama3 forward (un-permute + partial RoPE), same as
    // arch `llama`.
    if      (arch == "llama" || arch == "phi3" || arch == "granite") out.arch = ModelArch::kLlama3;
    else if (arch == "qwen35")                      out.arch = ModelArch::kQwen35Dense;
    else                                            out.arch = ModelArch::kQwen3Dense;  // qwen3/qwen2
    return {};
}

std::string read_dense_config_auto(const GgufReader& g, DenseConfig& out) {
    const auto* kv = g.find_kv("general.architecture");
    if (!kv) return "missing general.architecture";
    return read_dense_config(g, kv->as_string(), out);
}

std::string read_qwen35_config(const GgufReader& g, Qwen35Config& out) {
    // Dense block (full-attn head counts, ffn, vocab, rope, eps) via the
    // shared reader; for `qwen35` attention.head_count(_kv) are the FULL-
    // ATTENTION counts and feed_forward_length is the dense MLP width.
    if (auto m = read_dense_config(g, "qwen35", out.dense); !m.empty()) return m;

    auto key = [](const char* suffix) {
        return std::string("qwen35.") + suffix;
    };
    auto need_u32 = [&](const char* suffix, uint32_t& dst) -> bool {
        const auto* kv = g.find_kv(key(suffix));
        if (!kv) return false;
        dst = uint32_t(kv->as_uint());
        return true;
    };

    // full_attention_interval is OPTIONAL (llama.cpp defaults it to 4).
    need_u32("full_attention_interval", out.full_attn_interval);
    if (out.full_attn_interval == 0) return key("full_attention_interval") + " is zero";

    // Gated-DeltaNet ("ssm") params are REQUIRED for the hybrid forward.
    if (!need_u32("ssm.inner_size",     out.ssm_inner))     return "missing " + key("ssm.inner_size");
    if (!need_u32("ssm.state_size",     out.ssm_state))     return "missing " + key("ssm.state_size");
    if (!need_u32("ssm.group_count",    out.ssm_n_k_heads)) return "missing " + key("ssm.group_count");
    if (!need_u32("ssm.time_step_rank", out.ssm_n_v_heads)) return "missing " + key("ssm.time_step_rank");
    if (!need_u32("ssm.conv_kernel",    out.ssm_conv_kernel)) return "missing " + key("ssm.conv_kernel");
    if (out.ssm_n_v_heads == 0) return key("ssm.time_step_rank") + " is zero";
    out.ssm_v_head_dim = out.ssm_inner / out.ssm_n_v_heads;

    // NextN/MTP layers are counted in block_count but only used for speculative
    // decoding; optional (absent → 0). Standard decode runs n_transformer_layers().
    need_u32("nextn_predict_layers", out.nextn_predict_layers);

    // M-RoPE section split (optional; only load-bearing with vision tokens —
    // we run text-only where M-RoPE collapses to standard partial RoPE).
    if (const auto* kv = g.find_kv(key("rope.dimension_sections"))) {
        const auto arr = kv->as_pod_array<int32_t>();
        for (size_t i = 0; i < arr.size() && i < 4; ++i) out.rope_sections[i] = arr[i];
    }
    return {};
}

std::string read_qwen3moe_config(const GgufReader& g, Qwen3MoeConfig& out) {
    // Attention block reuses the dense reader (qwen3moe attention == qwen3-dense:
    // QK-norm + partial RoPE + GQA). feed_forward_length is present in qwen3moe
    // GGUFs (llama.cpp writes it) but unused here — the experts replace the FFN.
    if (auto m = read_dense_config(g, "qwen3moe", out.dense); !m.empty()) return m;
    out.dense.arch = ModelArch::kQwen3Moe;

    auto need_u32 = [&](const char* suffix, uint32_t& dst) -> bool {
        const auto* kv = g.find_kv(std::string("qwen3moe.") + suffix);
        if (!kv) return false;
        dst = uint32_t(kv->as_uint());
        return true;
    };
    if (!need_u32("expert_count", out.n_experts))
        return "missing qwen3moe.expert_count";
    if (!need_u32("expert_used_count", out.n_experts_used))
        return "missing qwen3moe.expert_used_count";
    if (!need_u32("expert_feed_forward_length", out.expert_ffn))
        return "missing qwen3moe.expert_feed_forward_length";
    if (out.n_experts == 0 || out.n_experts_used == 0 || out.expert_ffn == 0)
        return "qwen3moe: zero expert_count/used_count/ffn";
    if (out.n_experts_used > out.n_experts)
        return "qwen3moe: expert_used_count > expert_count";
    return {};
}

std::string read_gptoss_config(const GgufReader& g, GptOssConfig& out) {
    // Attention reuses the dense reader (GQA + RoPE); the dense path derives
    // q-dim = n_q_heads*head_dim (4096), distinct from hidden (2880). The arch
    // prefix is "gpt-oss" (hyphenated), matching the GGUF KV keys.
    if (auto m = read_dense_config(g, "gpt-oss", out.dense); !m.empty()) return m;
    out.dense.arch = ModelArch::kGptOss;

    auto u32 = [&](const char* suffix, uint32_t& dst) -> bool {
        const auto* kv = g.find_kv(std::string("gpt-oss.") + suffix);
        if (!kv) return false;
        dst = uint32_t(kv->as_uint());
        return true;
    };
    if (!u32("expert_count", out.n_experts))
        return "missing gpt-oss.expert_count";
    if (!u32("expert_used_count", out.n_experts_used))
        return "missing gpt-oss.expert_used_count";
    if (!u32("expert_feed_forward_length", out.expert_ffn))
        return "missing gpt-oss.expert_feed_forward_length";
    u32("attention.sliding_window", out.sliding_window);   // optional (even layers windowed)
    if (out.n_experts == 0 || out.n_experts_used == 0 || out.expert_ffn == 0)
        return "gpt-oss: zero expert_count/used_count/ffn";
    if (out.n_experts_used > out.n_experts)
        return "gpt-oss: expert_used_count > expert_count";
    // YaRN rope: freq_scale = 1/factor; ext_factor=1 only when scaling.type=="yarn".
    // The mscale (1+0.1*ln(1/freq_scale)=1.3466) folds into cos/sin at ALL positions.
    if (const auto* kv = g.find_kv("gpt-oss.rope.scaling.factor"))
        out.rope_freq_scale = 1.0f / float(kv->as_float());                  // 32 -> 0.03125
    if (const auto* kv = g.find_kv("gpt-oss.rope.scaling.original_context_length"))
        out.rope_orig_ctx = uint32_t(kv->as_uint());                         // 4096
    if (const auto* kv = g.find_kv("gpt-oss.rope.scaling.type"))
        if (kv->as_string() == "yarn") out.rope_ext_factor = 1.0f;
    return {};
}

std::string read_gemma4_config(const GgufReader& g, GemmaConfig& out) {
    auto u32 = [&](const char* suffix, uint32_t& dst) -> bool {
        const auto* kv = g.find_kv(std::string("gemma4.") + suffix);
        if (!kv) return false;
        dst = uint32_t(kv->as_uint());
        return true;
    };
    auto f32 = [&](const char* suffix, float& dst) {
        if (const auto* kv = g.find_kv(std::string("gemma4.") + suffix))
            dst = float(kv->as_float());
    };
    if (!u32("block_count", out.n_layers))           return "missing gemma4.block_count";
    if (!u32("embedding_length", out.hidden))        return "missing gemma4.embedding_length";
    if (!u32("attention.head_count", out.n_q_heads)) return "missing gemma4.attention.head_count";
    f32("attention.layer_norm_rms_epsilon", out.rms_eps);
    u32("context_length", out.ctx_train);
    u32("attention.sliding_window", out.sliding_window);
    f32("final_logit_softcapping", out.final_logit_softcap);
    if (!u32("attention.key_length_swa", out.head_dim_swa))    return "missing gemma4.attention.key_length_swa";
    if (!u32("attention.key_length",     out.head_dim_global)) return "missing gemma4.attention.key_length";
    out.n_rot_swa = out.head_dim_swa;       u32("rope.dimension_count_swa", out.n_rot_swa);
    out.n_rot_global = out.head_dim_global;  u32("rope.dimension_count",     out.n_rot_global);
    f32("rope.freq_base_swa", out.rope_theta_swa);
    f32("rope.freq_base",     out.rope_theta_global);
    u32("feed_forward_length", out.ffn);
    if (u32("expert_count", out.n_experts) && out.n_experts > 0) {
        out.is_moe = true;
        u32("expert_used_count", out.n_experts_used);
        u32("expert_feed_forward_length", out.expert_ffn);
        if (out.n_experts_used == 0 || out.expert_ffn == 0)
            return "gemma4: zero expert_used_count/expert_ffn";
    }
    const auto* toks = g.find_kv("tokenizer.ggml.tokens");
    if (!toks || toks->type != GgufValueType::kArray) return "missing tokenizer.ggml.tokens";
    out.vocab = uint32_t(toks->n_array);
    if (out.n_layers == 0) return "gemma4: zero block_count";

    // Per-layer sliding/global pattern (BOOL array): true = sliding-window layer.
    out.is_swa.assign(out.n_layers, 1);
    if (const auto* kv = g.find_kv("gemma4.attention.sliding_window_pattern");
        kv && kv->type == GgufValueType::kArray && kv->n_array == out.n_layers) {
        auto a = kv->as_pod_array<uint8_t>();
        for (uint32_t i = 0; i < out.n_layers; ++i) out.is_swa[i] = a[i] ? 1 : 0;
    } else {
        // Fallback: documented 5 sliding : 1 global (global at idx % 6 == 5).
        for (uint32_t i = 0; i < out.n_layers; ++i) out.is_swa[i] = ((i % 6) == 5) ? 0 : 1;
    }
    // Per-layer KV head count (INT32 array; or scalar fallback).
    out.n_kv_heads.assign(out.n_layers, 0);
    const auto* hk = g.find_kv("gemma4.attention.head_count_kv");
    if (!hk) return "missing gemma4.attention.head_count_kv";
    if (hk->type == GgufValueType::kArray && hk->n_array == out.n_layers) {
        auto a = hk->as_pod_array<int32_t>();
        for (uint32_t i = 0; i < out.n_layers; ++i) out.n_kv_heads[i] = uint32_t(a[i]);
    } else {
        const uint32_t v = uint32_t(hk->as_uint());
        out.n_kv_heads.assign(out.n_layers, v);
    }
    return {};
}

std::string read_laguna_config(const GgufReader& g, LagunaConfig& out) {
    // KEY NAMES VERIFIED AGAINST THE REAL GGUF 2026-08-09 — an earlier version
    // of this reader guessed several of them wrong (`attn_factor` vs
    // `yarn_attn_factor`, `dense_block_count` vs `leading_dense_block_count`,
    // and it read `attention.head_count` as a SCALAR when the file stores a
    // 48-entry ARRAY).  Guessing GGUF keys is how a loader silently reads
    // zeros.
    auto u32 = [&](const char* suffix, uint32_t& dst) -> bool {
        const auto* kv = g.find_kv(std::string("laguna.") + suffix);
        if (!kv) return false;
        dst = uint32_t(kv->as_uint());
        return true;
    };
    auto f32 = [&](const char* suffix, float& dst) {
        if (const auto* kv = g.find_kv(std::string("laguna.") + suffix))
            dst = float(kv->as_float());
    };
    if (!u32("block_count", out.n_layers))    return "missing laguna.block_count";
    if (!u32("embedding_length", out.hidden)) return "missing laguna.embedding_length";
    u32("feed_forward_length", out.ffn_dense);
    if (!u32("attention.head_count_kv", out.n_kv_heads))
        return "missing laguna.attention.head_count_kv";
    if (!u32("attention.key_length", out.head_dim))
        return "missing laguna.attention.key_length";
    u32("attention.value_length", out.v_head_dim);
    u32("vocab_size", out.vocab);
    u32("context_length", out.ctx_train);
    u32("attention.sliding_window", out.sliding_window);
    u32("leading_dense_block_count", out.dense_layers);
    f32("attention.layer_norm_rms_epsilon", out.rms_eps);
    u32("expert_count", out.n_experts);
    u32("expert_used_count", out.n_experts_used);
    u32("expert_feed_forward_length", out.expert_ffn);
    u32("expert_shared_feed_forward_length", out.shared_expert_ffn);
    u32("expert_gating_func", out.expert_gating_func);
    f32("expert_weights_scale", out.expert_weights_scale);
    if (const auto* kv = g.find_kv("laguna.expert_weights_norm"))
        out.expert_weights_norm = kv->as_uint() != 0;
    // Two rope configs, one per layer type.
    f32("rope.freq_base", out.rope_theta_full);
    f32("rope.freq_base_swa", out.rope_theta_swa);
    u32("rope.dimension_count", out.rope_dim_full);
    u32("rope.dimension_count_swa", out.rope_dim_swa);
    f32("rope.scaling.factor", out.yarn_factor);
    f32("rope.scaling.yarn_attn_factor", out.yarn_attn_factor);
    f32("rope.scaling.yarn_beta_fast", out.yarn_beta_fast);
    f32("rope.scaling.yarn_beta_slow", out.yarn_beta_slow);
    u32("rope.scaling.original_context_length", out.yarn_orig_ctx);
    // The per-layer query-head array, which is ALSO the layer-type map.
    if (const auto* kv = g.find_kv("laguna.attention.head_count");
        kv && kv->type == GgufValueType::kArray) {
        const auto a = kv->as_pod_array<int32_t>();
        out.n_q_heads.assign(a.begin(), a.end());
    }
    if (out.n_q_heads.size() != out.n_layers)
        return "laguna.attention.head_count has " + std::to_string(out.n_q_heads.size()) +
               " entries for " + std::to_string(out.n_layers) +
               " layers — it is the per-layer head count AND the full/sliding map, so a "
               "mismatch means the layer types are unknown";
    if (out.n_experts_used > out.n_experts)
        return "laguna: expert_used_count > expert_count";
    return {};
}

std::string read_inkling_config(const GgufReader& g, InklingConfig& out) {
    // Same bespoke shape as the deepseek4 reader below: the arch has per-layer
    // KV-head and sliding-window ARRAYS and relative-attention keys no dense
    // reader knows.  Every REQUIRED key was present on the real file
    // (Inkling-Small-UD-Q4_K_XL, dumped 2026-08-09).
    auto u32 = [&](const char* suffix, uint32_t& dst) -> bool {
        const auto* kv = g.find_kv(std::string("inkling.") + suffix);
        if (!kv) return false;
        dst = uint32_t(kv->as_uint());
        return true;
    };
    auto f32 = [&](const char* suffix, float& dst) {
        if (const auto* kv = g.find_kv(std::string("inkling.") + suffix))
            dst = float(kv->as_float());
    };
    if (!u32("block_count", out.n_layers))           return "missing inkling.block_count";
    if (!u32("embedding_length", out.hidden))        return "missing inkling.embedding_length";
    if (!u32("attention.head_count", out.n_q_heads)) return "missing inkling.attention.head_count";
    if (!u32("attention.key_length", out.head_dim))  return "missing inkling.attention.key_length";
    u32("attention.value_length", out.v_head_dim);
    u32("feed_forward_length", out.ffn_dense);
    u32("context_length", out.ctx_train);
    f32("attention.layer_norm_rms_epsilon", out.rms_eps);
    if (!u32("expert_count", out.n_experts))             return "missing inkling.expert_count";
    if (!u32("expert_used_count", out.n_experts_used))   return "missing inkling.expert_used_count";
    if (!u32("expert_feed_forward_length", out.expert_ffn))
        return "missing inkling.expert_feed_forward_length";
    u32("expert_shared_count", out.n_shared_experts);
    f32("expert_weights_scale", out.expert_weights_scale);
    u32("expert_gating_func", out.expert_gating_func);
    u32("dense_block_count", out.dense_layers);
    u32("attention.sliding_window", out.sliding_window);
    u32("d_rel", out.d_rel);
    u32("rel_extent", out.rel_extent);
    u32("rel_extent_swa", out.rel_extent_swa);
    u32("shortconv_kernel", out.shortconv_kernel);
    u32("vocab_size", out.vocab);
    u32("unpadded_vocab_size", out.vocab_unpadded);
    f32("logit_scale_denom", out.logit_scale_denom);
    u32("log_scaling_n_floor", out.log_scaling_n_floor);
    f32("log_scaling_alpha", out.log_scaling_alpha);
    if (const auto* kv = g.find_kv("inkling.attention.head_count_kv");
        kv && kv->type == GgufValueType::kArray) {
        const auto a = kv->as_pod_array<int32_t>();
        out.n_kv_heads.assign(a.begin(), a.end());
    }
    if (const auto* kv = g.find_kv("inkling.attention.sliding_window_pattern");
        kv && kv->type == GgufValueType::kArray) {
        const auto a = kv->as_pod_array<uint8_t>();
        out.swa_pattern.assign(a.begin(), a.end());
    }
    if (out.n_kv_heads.size() != out.n_layers)
        return "inkling.attention.head_count_kv has " + std::to_string(out.n_kv_heads.size()) +
               " entries for " + std::to_string(out.n_layers) + " layers";
    if (out.n_experts_used > out.n_experts)
        return "inkling: expert_used_count > expert_count";
    return {};
}

std::string read_glmdsa_config(const GgufReader& g, GlmDsaConfig& out) {
    // Bespoke reader. read_dense_config would not fail on this file — it would
    // succeed and set head_dim = attention.key_length = 576, which is the MLA
    // latent width (kv_lora 512 + rope 64), not a per-head dimension. Silently
    // wrong geometry is worse than an honest refusal, so nothing is shared.
    auto u32 = [&](const char* suffix, uint32_t& dst) -> bool {
        const auto* kv = g.find_kv(std::string("glm-dsa.") + suffix);
        if (!kv) return false;
        dst = uint32_t(kv->as_uint());
        return true;
    };
    auto f32 = [&](const char* suffix, float& dst) {
        if (const auto* kv = g.find_kv(std::string("glm-dsa.") + suffix))
            dst = float(kv->as_float());
    };

    if (!u32("block_count", out.n_layers))               return "missing glm-dsa.block_count";
    if (!u32("embedding_length", out.hidden))            return "missing glm-dsa.embedding_length";
    if (!u32("attention.head_count", out.n_q_heads))     return "missing glm-dsa.attention.head_count";
    if (!u32("attention.head_count_kv", out.n_kv_heads)) return "missing glm-dsa.attention.head_count_kv";
    if (!u32("attention.key_length", out.head_dim))      return "missing glm-dsa.attention.key_length";
    u32("attention.value_length", out.v_head_dim);
    u32("feed_forward_length", out.ffn);
    u32("context_length", out.ctx_train);
    f32("attention.layer_norm_rms_epsilon", out.rms_eps);

    // MLA ranks. kv_lora_rank is REQUIRED: without it the absorb-form k_b/v_b
    // shapes cannot be checked and a mis-shaped bind would go unnoticed.
    u32("attention.q_lora_rank", out.q_lora_rank);
    if (!u32("attention.kv_lora_rank", out.kv_lora_rank))
        return "missing glm-dsa.attention.kv_lora_rank";
    u32("attention.key_length_mla", out.key_len_mla);
    u32("attention.value_length_mla", out.value_len_mla);

    // DSA indexer.
    u32("attention.indexer.head_count", out.indexer_n_heads);
    u32("attention.indexer.key_length", out.indexer_head_dim);
    u32("attention.indexer.top_k", out.indexer_top_k);

    // RoPE. The file carries no rope.scaling.* keys, so this is plain RoPE over
    // rope.dimension_count (64) of each 256-wide MLA head — do NOT add YaRN
    // handling on the assumption that a 1M-context model must have it.
    if (!u32("rope.dimension_count", out.rope_dim)) out.rope_dim = out.head_dim;
    f32("rope.freq_base", out.rope_theta);

    // MoE.
    u32("leading_dense_block_count", out.leading_dense);
    if (!u32("expert_count", out.n_experts))                return "missing glm-dsa.expert_count";
    if (!u32("expert_used_count", out.n_experts_used))      return "missing glm-dsa.expert_used_count";
    if (!u32("expert_feed_forward_length", out.expert_ffn)) return "missing glm-dsa.expert_feed_forward_length";
    u32("expert_shared_count", out.n_shared_experts);
    u32("expert_gating_func", out.expert_gating_func);
    f32("expert_weights_scale", out.expert_weights_scale);
    if (const auto* kv = g.find_kv("glm-dsa.expert_weights_norm"))
        out.expert_weights_norm = kv->as_bool();
    u32("expert_group_count", out.expert_group_count);
    u32("expert_group_used_count", out.expert_group_used);

    // NextN / MTP.
    u32("nextn_predict_layers", out.nextn_predict_layers);

    const auto* toks = g.find_kv("tokenizer.ggml.tokens");
    if (!toks || toks->type != GgufValueType::kArray) return "missing tokenizer.ggml.tokens";
    out.vocab = uint32_t(toks->n_array);

    // -- consistency, so a wrong file fails here rather than in a kernel -----
    if (out.n_layers == 0)          return "glm-dsa: zero block_count";
    if (out.hidden == 0)            return "glm-dsa: zero embedding_length";
    if (out.n_experts == 0 || out.n_experts_used == 0 || out.expert_ffn == 0)
        return "glm-dsa: zero expert_count/used_count/ffn";
    if (out.n_experts_used > out.n_experts)
        return "glm-dsa: expert_used_count > expert_count";
    if (out.nextn_predict_layers >= out.n_layers)
        return "glm-dsa: nextn_predict_layers >= block_count";
    if (out.leading_dense >= out.n_transformer_layers())
        return "glm-dsa: leading_dense_block_count covers every transformer block";
    // key_length must be kv_lora_rank + rope_dim — that identity is what makes
    // the MLA cache layout (one 512-d latent + one 64-d rope key per position)
    // correct. If it ever fails, the cache would be silently mis-sized.
    if (out.key_len_mla && out.kv_lora_rank && out.rope_dim &&
        out.head_dim != out.kv_lora_rank + out.rope_dim)
        return "glm-dsa: attention.key_length != kv_lora_rank + rope.dimension_count";
    return {};
}

std::string read_glm5next_config(const GgufReader& g, Glm5NextConfig& out) {
    // Bespoke reader (see the header comment): head_count_kv here is a
    // 46-entry per-layer array — any scalar binding would misread the whole
    // hybrid layout. Helper idiom as read_glmdsa_config; f32 returns presence
    // because two of the floats (gate_lower_bound, weights_scale) change the
    // math silently if a missing key fell through to a default.
    auto u32 = [&](const char* suffix, uint32_t& dst) -> bool {
        const auto* kv = g.find_kv(std::string("glm5next.") + suffix);
        if (!kv) return false;
        dst = uint32_t(kv->as_uint());
        return true;
    };
    auto f32 = [&](const char* suffix, float& dst) -> bool {
        const auto* kv = g.find_kv(std::string("glm5next.") + suffix);
        if (!kv) return false;
        dst = float(kv->as_float());
        return true;
    };
    auto arr_i32 = [&](const char* suffix, std::vector<int32_t>& dst) -> bool {
        const auto* kv = g.find_kv(std::string("glm5next.") + suffix);
        if (!kv || kv->type != GgufValueType::kArray) return false;
        const auto a = kv->as_pod_array<int32_t>();
        dst.assign(a.begin(), a.end());
        return true;
    };
    auto arr_f32 = [&](const char* suffix, std::vector<float>& dst) -> bool {
        const auto* kv = g.find_kv(std::string("glm5next.") + suffix);
        if (!kv || kv->type != GgufValueType::kArray) return false;
        const auto a = kv->as_pod_array<float>();
        dst.assign(a.begin(), a.end());
        return true;
    };

    if (!u32("block_count", out.n_layers))           return "missing glm5next.block_count";
    if (!u32("embedding_length", out.hidden))        return "missing glm5next.embedding_length";
    if (!u32("attention.head_count", out.n_q_heads)) return "missing glm5next.attention.head_count";
    if (!u32("feed_forward_length", out.ffn))        return "missing glm5next.feed_forward_length";
    u32("context_length", out.ctx_train);
    f32("attention.layer_norm_rms_epsilon", out.rms_eps);
    f32("attention.layer_norm_epsilon", out.ln_eps);   // indexer k LayerNorm

    // The per-layer attention map — THE key that makes this arch hybrid.
    if (!arr_i32("attention.head_count_kv", out.attn_kind))
        return "missing glm5next.attention.head_count_kv (per-layer array)";

    // MLA (full-attn layers). All four widths are load-bearing for the cache
    // layout and the absorbed k_b/v_b binds.
    if (!u32("attention.q_lora_rank", out.q_lora_rank))   return "missing glm5next.attention.q_lora_rank";
    if (!u32("attention.kv_lora_rank", out.kv_lora_rank)) return "missing glm5next.attention.kv_lora_rank";
    if (!u32("attention.key_length", out.key_len))        return "missing glm5next.attention.key_length";
    if (!u32("attention.value_length", out.value_len))    return "missing glm5next.attention.value_length";
    if (!u32("attention.key_length_mla", out.key_len_mla))     return "missing glm5next.attention.key_length_mla";
    if (!u32("attention.value_length_mla", out.value_len_mla)) return "missing glm5next.attention.value_length_mla";

    // NOPE-only: the file says rope.dimension_count = 0 and the forward will
    // apply no rope anywhere. A nonzero value would make that silently wrong.
    {
        uint32_t rope_dim = 0;
        if (u32("rope.dimension_count", rope_dim) && rope_dim != 0)
            return "glm5next: rope.dimension_count != 0 (this port is NOPE-only)";
    }

    // KDA (linear layers).
    if (!u32("kda.head_dim", out.kda_head_dim))   return "missing glm5next.kda.head_dim";
    if (!u32("ssm.conv_kernel", out.conv_kernel)) return "missing glm5next.ssm.conv_kernel";
    if (!f32("kda.gate_lower_bound", out.kda_gate_lower_bound))
        return "missing glm5next.kda.gate_lower_bound";

    // DSA indexer + gated compressor — all four required (every full-attn
    // layer routes through it).
    if (!u32("attention.indexer.head_count", out.indexer_n_heads)) return "missing glm5next.attention.indexer.head_count";
    if (!u32("attention.indexer.key_length", out.indexer_head_dim)) return "missing glm5next.attention.indexer.key_length";
    if (!u32("attention.indexer.top_k", out.indexer_top_k))         return "missing glm5next.attention.indexer.top_k";
    if (!u32("attention.indexer.kpool", out.indexer_kpool))         return "missing glm5next.attention.indexer.kpool";

    // Hyper-connections (Sinkhorn).
    if (!u32("hyper_connection.count", out.hc_count)) return "missing glm5next.hyper_connection.count";
    if (!u32("hyper_connection.sinkhorn_iterations", out.hc_sinkhorn_iters))
        return "missing glm5next.hyper_connection.sinkhorn_iterations";
    f32("hyper_connection.epsilon", out.hc_eps);

    // MoE.
    u32("leading_dense_block_count", out.leading_dense);
    if (!u32("expert_count", out.n_experts))                return "missing glm5next.expert_count";
    if (!u32("expert_used_count", out.n_experts_used))      return "missing glm5next.expert_used_count";
    if (!u32("expert_feed_forward_length", out.expert_ffn)) return "missing glm5next.expert_feed_forward_length";
    u32("expert_shared_feed_forward_length", out.shexp_ffn);
    u32("expert_shared_count", out.n_shared_experts);
    u32("expert_gating_func", out.expert_gating_func);
    if (!f32("expert_weights_scale", out.expert_weights_scale))
        return "missing glm5next.expert_weights_scale";
    if (const auto* kv = g.find_kv("glm5next.expert_weights_norm"))
        out.expert_weights_norm = kv->as_bool();
    // Grouped routing is NOT implemented; the file ships group_count 1 (a
    // no-op). Refuse anything else instead of silently ignoring it.
    {
        uint32_t groups = 1;
        if (u32("expert_group_count", groups) && groups != 1)
            return "glm5next: expert_group_count != 1 not supported";
    }
    // Per-layer swiglu clamps: a missing array would silently drop the clamp.
    if (!arr_f32("swiglu_clamp_exp", out.swiglu_clamp_exp))     return "missing glm5next.swiglu_clamp_exp";
    if (!arr_f32("swiglu_clamp_shexp", out.swiglu_clamp_shexp)) return "missing glm5next.swiglu_clamp_shexp";

    // NextN / MTP.
    u32("nextn_predict_layers", out.nextn_predict_layers);

    const auto* toks = g.find_kv("tokenizer.ggml.tokens");
    if (!toks || toks->type != GgufValueType::kArray) return "missing tokenizer.ggml.tokens";
    out.vocab = uint32_t(toks->n_array);

    // -- consistency, so a wrong file fails here rather than in a kernel -----
    if (out.n_layers == 0 || out.hidden == 0)
        return "glm5next: zero block_count/embedding_length";
    if (out.attn_kind.size() != out.n_layers)
        return "glm5next: attention.head_count_kv array length != block_count";
    if (out.swiglu_clamp_exp.size() != out.n_layers ||
        out.swiglu_clamp_shexp.size() != out.n_layers)
        return "glm5next: swiglu_clamp array length != block_count";
    if (out.n_experts == 0 || out.n_experts_used == 0 || out.expert_ffn == 0)
        return "glm5next: zero expert_count/used_count/ffn";
    if (out.n_experts_used > out.n_experts)
        return "glm5next: expert_used_count > expert_count";
    if (out.expert_gating_func != 2)
        return "glm5next: expert_gating_func != 2 (only sigmoid gating is implemented)";
    if (out.nextn_predict_layers >= out.n_layers)
        return "glm5next: nextn_predict_layers >= block_count";
    if (out.leading_dense >= out.n_transformer_layers())
        return "glm5next: leading_dense_block_count covers every transformer block";
    // NOPE identity: with no rope dims, the full-attn cache key IS the MLA
    // latent — key_length must equal kv_lora_rank or the cache is mis-sized.
    if (out.key_len != out.kv_lora_rank)
        return "glm5next: attention.key_length != kv_lora_rank (rope dims expected 0)";
    if (out.value_len != out.key_len)
        return "glm5next: attention.value_length != attention.key_length";
    if (out.key_len_mla == 0 || out.key_len_mla != out.value_len_mla)
        return "glm5next: bad key/value_length_mla";
    // KDA geometry: q/k/v are n_q_heads x kda_head_dim wide; the fused conv
    // and the [d,d] recurrence state both hang off this product.
    if (out.kda_head_dim == 0 || out.n_q_heads == 0)
        return "glm5next: zero kda.head_dim/attention.head_count";
    if (const auto* kv = g.find_kv("glm5next.vocab_size");
        kv && uint32_t(kv->as_uint()) != out.vocab)
        return "glm5next: vocab_size != tokenizer.ggml.tokens length";
    return {};
}


std::string read_hyv4_config(const GgufReader& g, Hyv4Config& out) {
    // Bespoke reader (Tencent Hy4-preview). Helper idiom as read_glm5next_config;
    // required keys hard-fail so a wrong file refuses instead of mis-running.
    auto u32 = [&](const char* suffix, uint32_t& dst) -> bool {
        const auto* kv = g.find_kv(std::string("hyv4.") + suffix);
        if (!kv) return false;
        dst = uint32_t(kv->as_uint());
        return true;
    };
    auto f32 = [&](const char* suffix, float& dst) -> bool {
        const auto* kv = g.find_kv(std::string("hyv4.") + suffix);
        if (!kv) return false;
        dst = float(kv->as_float());
        return true;
    };
    auto arr_f32 = [&](const char* suffix, std::vector<float>& dst) -> bool {
        const auto* kv = g.find_kv(std::string("hyv4.") + suffix);
        if (!kv || kv->type != GgufValueType::kArray) return false;
        const auto a = kv->as_pod_array<float>();
        dst.assign(a.begin(), a.end());
        return true;
    };

    if (!u32("block_count", out.n_layers))           return "missing hyv4.block_count";
    if (!u32("embedding_length", out.hidden))        return "missing hyv4.embedding_length";
    if (!u32("attention.head_count", out.n_q_heads)) return "missing hyv4.attention.head_count";
    if (!u32("feed_forward_length", out.ffn))        return "missing hyv4.feed_forward_length";
    u32("context_length", out.ctx_train);
    f32("attention.layer_norm_rms_epsilon", out.rms_eps);
    // the reference plumbs rms_eps into the indexer k LayerNorm
    out.ln_eps = out.rms_eps;

    if (!u32("attention.q_lora_rank", out.q_lora_rank))   return "missing hyv4.attention.q_lora_rank";
    if (!u32("attention.kv_lora_rank", out.kv_lora_rank)) return "missing hyv4.attention.kv_lora_rank";
    if (!u32("attention.key_length", out.key_len))        return "missing hyv4.attention.key_length";
    if (!u32("attention.value_length", out.value_len))    return "missing hyv4.attention.value_length";
    if (!u32("attention.key_length_mla", out.key_len_mla))     return "missing hyv4.attention.key_length_mla";
    if (!u32("attention.value_length_mla", out.value_len_mla)) return "missing hyv4.attention.value_length_mla";

    // Interleaved (NORM) rope on the trailing rope_dim — REQUIRED nonzero
    // (the NOPE assumption of the glm5next donor does not hold here).
    if (!u32("rope.dimension_count", out.rope_dim) || out.rope_dim == 0)
        return "missing/zero hyv4.rope.dimension_count";
    f32("rope.freq_base", out.rope_theta);

    if (!u32("attention.indexer.head_count", out.indexer_n_heads)) return "missing hyv4.attention.indexer.head_count";
    if (!u32("attention.indexer.key_length", out.indexer_head_dim)) return "missing hyv4.attention.indexer.key_length";
    if (!u32("attention.indexer.top_k", out.indexer_top_k))         return "missing hyv4.attention.indexer.top_k";
    out.indexer_kpool = 1;   // no compressor pooling on hyv4
    {
        const auto* kv = g.find_kv("hyv4.attention.indexer.is_full");
        if (!kv || kv->type != GgufValueType::kArray)
            return "missing hyv4.attention.indexer.is_full (per-layer array)";
        if (kv->inner_type == GgufValueType::kI32 || kv->inner_type == GgufValueType::kU32) {
            const auto a = kv->as_pod_array<int32_t>();
            out.indexer_is_full.assign(a.begin(), a.end());
        } else {
            const auto a = kv->as_pod_array<uint8_t>();
            out.indexer_is_full.assign(a.begin(), a.end());
        }
    }

    if (!u32("hyper_connection.count", out.hc_count)) return "missing hyv4.hyper_connection.count";
    f32("hyper_connection.epsilon", out.hc_eps);
    if (!f32("hyper_connection.magnitude", out.hc_magnitude))
        return "missing hyv4.hyper_connection.magnitude";

    u32("leading_dense_block_count", out.leading_dense);
    if (!u32("expert_count", out.n_experts))                return "missing hyv4.expert_count";
    if (!u32("expert_used_count", out.n_experts_used))      return "missing hyv4.expert_used_count";
    if (!u32("expert_feed_forward_length", out.expert_ffn)) return "missing hyv4.expert_feed_forward_length";
    u32("expert_shared_count", out.n_shared_experts);
    u32("expert_gating_func", out.expert_gating_func);
    if (!f32("expert_weights_scale", out.expert_weights_scale))
        return "missing hyv4.expert_weights_scale";
    if (const auto* kv = g.find_kv("hyv4.expert_weights_norm"))
        out.expert_weights_norm = kv->as_bool();
    out.shexp_ffn = out.expert_ffn * std::max(1u, out.n_shared_experts);

    // Routed-expert swiglu clamp only; shared/dense are UNCLAMPED (the GGUF
    // carries no swiglu_clamp_shexp key) — fill +INF so the shared/dense
    // clamp calls become identities.
    if (!arr_f32("swiglu_clamp_exp", out.swiglu_clamp_exp)) return "missing hyv4.swiglu_clamp_exp";
    out.swiglu_clamp_shexp.assign(out.n_layers, std::numeric_limits<float>::infinity());

    const auto* toks = g.find_kv("tokenizer.ggml.tokens");
    if (!toks || toks->type != GgufValueType::kArray) return "missing tokenizer.ggml.tokens";
    out.vocab = uint32_t(toks->n_array);

    // -- consistency ---------------------------------------------------------
    if (out.n_layers == 0 || out.hidden == 0)
        return "hyv4: zero block_count/embedding_length";
    if (out.swiglu_clamp_exp.size() != out.n_layers)
        return "hyv4: swiglu_clamp_exp length != block_count";
    if (out.indexer_is_full.size() != out.n_layers)
        return "hyv4: indexer.is_full length != block_count";
    if (!out.indexer_is_full[0])
        return "hyv4: layer 0 must own an indexer";
    if (out.n_experts == 0 || out.n_experts_used == 0 || out.expert_ffn == 0)
        return "hyv4: zero expert_count/used_count/ffn";
    if (out.n_experts_used > out.n_experts)
        return "hyv4: expert_used_count > expert_count";
    if (out.expert_gating_func != 2)
        return "hyv4: expert_gating_func != 2 (only sigmoid gating is implemented)";
    if (out.leading_dense >= out.n_layers)
        return "hyv4: leading_dense_block_count covers every block";
    if (out.key_len != out.kv_lora_rank + out.rope_dim)
        return "hyv4: attention.key_length != kv_lora_rank + rope_dim";
    if (out.value_len != out.kv_lora_rank)
        return "hyv4: attention.value_length != kv_lora_rank";
    if (out.key_len_mla == 0 || out.key_len_mla != out.value_len_mla)
        return "hyv4: bad key/value_length_mla";
    if (const auto* kv = g.find_kv("hyv4.vocab_size");
        kv && uint32_t(kv->as_uint()) != out.vocab)
        return "hyv4: vocab_size != tokenizer.ggml.tokens length";
    return {};
}

std::string read_qwen4exp_config(const GgufReader& g, Qwen4ExpConfig& out) {
    // Bespoke reader (see the header comment): a dense binding would be
    // silently wrong on 36 of the 48 blocks. Helper idiom as read_glmdsa_config.
    auto u32 = [&](const char* suffix, uint32_t& dst) -> bool {
        const auto* kv = g.find_kv(std::string("qwen4exp.") + suffix);
        if (!kv) return false;
        dst = uint32_t(kv->as_uint());
        return true;
    };
    auto f32 = [&](const char* suffix, float& dst) {
        if (const auto* kv = g.find_kv(std::string("qwen4exp.") + suffix))
            dst = float(kv->as_float());
    };
    auto arr_i32 = [&](const char* suffix, std::vector<int32_t>& dst) -> bool {
        const auto* kv = g.find_kv(std::string("qwen4exp.") + suffix);
        if (!kv || kv->type != GgufValueType::kArray) return false;
        const auto a = kv->as_pod_array<int32_t>();
        dst.assign(a.begin(), a.end());
        return true;
    };
    auto arr_u64 = [&](const char* suffix, std::vector<uint64_t>& dst) -> bool {
        const auto* kv = g.find_kv(std::string("qwen4exp.") + suffix);
        if (!kv || kv->type != GgufValueType::kArray) return false;
        const auto a = kv->as_pod_array<uint64_t>();
        dst.assign(a.begin(), a.end());
        return true;
    };

    if (!u32("block_count", out.n_layers))               return "missing qwen4exp.block_count";
    if (!u32("embedding_length", out.hidden))            return "missing qwen4exp.embedding_length";
    if (!u32("attention.head_count", out.n_q_heads))     return "missing qwen4exp.attention.head_count";
    if (!u32("attention.head_count_kv", out.n_kv_heads)) return "missing qwen4exp.attention.head_count_kv";
    if (!u32("attention.key_length", out.head_dim))      return "missing qwen4exp.attention.key_length";
    // value_length exists on the real file and must EQUAL key_length — the KV
    // cache and attn_output binding assume one head width (256 == 256 today).
    {
        uint32_t v_len = out.head_dim;
        u32("attention.value_length", v_len);
        if (v_len != out.head_dim)
            return "qwen4exp: attention.value_length != attention.key_length";
    }
    if (!u32("full_attention_interval", out.full_attention_interval))
        return "missing qwen4exp.full_attention_interval";
    u32("context_length", out.ctx_train);
    f32("attention.layer_norm_rms_epsilon", out.rms_eps);

    // Partial rotary (64 of 256) is architectural: a missing key would
    // silently become full rotary, so unlike most rope keys it is REQUIRED.
    if (!u32("rope.dimension_count", out.rope_dim)) return "missing qwen4exp.rope.dimension_count";
    f32("rope.freq_base", out.rope_theta);
    arr_i32("rope.dimension_sections", out.rope_sections);

    // DeltaNet geometry — all five required: the fused-QKV split (K,K,V) and
    // the conv/state/norm shapes all hang off these.
    if (!u32("ssm.state_size", out.ssm_state))        return "missing qwen4exp.ssm.state_size";
    if (!u32("ssm.conv_kernel", out.ssm_conv_kernel)) return "missing qwen4exp.ssm.conv_kernel";
    if (!u32("ssm.group_count", out.ssm_k_heads))     return "missing qwen4exp.ssm.group_count";
    if (!u32("ssm.time_step_rank", out.ssm_v_heads))  return "missing qwen4exp.ssm.time_step_rank";
    if (!u32("ssm.inner_size", out.ssm_inner))        return "missing qwen4exp.ssm.inner_size";

    // Hyper-connections replace the plain pre-norms; without both keys the
    // residual stream cannot be shaped at all.
    if (!u32("hyper_connection.count", out.hc_count))       return "missing qwen4exp.hyper_connection.count";
    if (!u32("hyper_connection.low_rank", out.hc_low_rank)) return "missing qwen4exp.hyper_connection.low_rank";

    // QSA indexer — all-or-none.
    {
        const bool h = u32("attention.indexer.head_count", out.indexer_n_heads);
        const bool d = u32("attention.indexer.key_length", out.indexer_head_dim);
        const bool t = u32("attention.indexer.top_k", out.indexer_top_k);
        if (h != d || h != t) return "qwen4exp: partial attention.indexer.* key group";
    }
    arr_i32("attention.compress_ratios", out.compress_ratios);

    // MoE.
    if (!u32("expert_count", out.n_experts))                return "missing qwen4exp.expert_count";
    if (!u32("expert_used_count", out.n_experts_used))      return "missing qwen4exp.expert_used_count";
    if (!u32("expert_feed_forward_length", out.expert_ffn)) return "missing qwen4exp.expert_feed_forward_length";
    u32("expert_shared_feed_forward_length", out.shared_expert_ffn);

    // PLE — all-or-none, keyed on ple.layers (the convert script omits the
    // whole group when HF ple_layer_ids is empty).
    if (arr_i32("ple.layers", out.ple_layers)) {
        if (!u32("ple.ngram_size", out.ple_ngram))                return "missing qwen4exp.ple.ngram_size";
        if (!u32("ple.heads_per_ngram", out.ple_heads_per_ngram)) return "missing qwen4exp.ple.heads_per_ngram";
        if (!u32("ple.conv_kernel", out.ple_conv_kernel))         return "missing qwen4exp.ple.conv_kernel";
        if (!u32("embedding_length_per_layer_input", out.ple_embed_per_input))
            return "missing qwen4exp.embedding_length_per_layer_input";
        u32("ple.eos_token_id", out.ple_eos_token);
        u32("ple.image_token_id", out.ple_image_token);
        if (!arr_u64("ple.layer_multipliers", out.ple_layer_multipliers))
            return "missing qwen4exp.ple.layer_multipliers";
        if (!arr_u64("ple.head_offsets", out.ple_head_offsets))
            return "missing qwen4exp.ple.head_offsets";
        if (!arr_u64("ple.head_vocab_sizes", out.ple_head_vocab_sizes))
            return "missing qwen4exp.ple.head_vocab_sizes";
    }

    const auto* toks = g.find_kv("tokenizer.ggml.tokens");
    if (!toks || toks->type != GgufValueType::kArray) return "missing tokenizer.ggml.tokens";
    out.vocab = uint32_t(toks->n_array);

    // -- consistency, so a wrong file fails here rather than in a kernel -----
    if (out.n_layers == 0 || out.hidden == 0)
        return "qwen4exp: zero block_count/embedding_length";
    if (out.full_attention_interval == 0 || out.n_layers % out.full_attention_interval)
        return "qwen4exp: block_count not a multiple of full_attention_interval";
    if (out.n_kv_heads == 0 || out.n_q_heads % out.n_kv_heads)
        return "qwen4exp: head_count not a multiple of head_count_kv";
    if (out.rope_dim > out.head_dim)
        return "qwen4exp: rope.dimension_count > attention.key_length";
    // Interleaved mrope sections are half-dims; trailing zero-pad entries are
    // legitimate (the real file ends [.., 10, 0]).
    if (!out.rope_sections.empty()) {
        int64_t s = 0;
        for (int32_t v : out.rope_sections) {
            if (v < 0) return "qwen4exp: negative rope.dimension_sections entry";
            s += v;
        }
        if (2 * s != int64_t(out.rope_dim))
            return "qwen4exp: 2*sum(rope.dimension_sections) != rope.dimension_count";
    }
    // The DeltaNet identity everything downstream leans on (6144 = 48*128),
    // and the 16→48 head-repeat convention the qwen35 kernels assume.
    if (out.ssm_inner != out.ssm_v_heads * out.ssm_state)
        return "qwen4exp: ssm.inner_size != time_step_rank * state_size";
    if (out.ssm_k_heads == 0 || out.ssm_v_heads % out.ssm_k_heads)
        return "qwen4exp: ssm.time_step_rank not a multiple of ssm.group_count";
    if (out.n_experts == 0 || out.n_experts_used == 0 || out.expert_ffn == 0)
        return "qwen4exp: zero expert_count/used_count/ffn";
    if (out.n_experts_used > out.n_experts)
        return "qwen4exp: expert_used_count > expert_count";
    if (out.hc_count == 0 || out.hc_low_rank == 0)
        return "qwen4exp: zero hyper_connection.count/low_rank";
    // The compress-ratio schedule and the interval must agree block-by-block
    // (nonzero exactly on full-attention blocks). Disagreement means the layer
    // schedule is ambiguous and NOTHING may be bound.
    if (!out.compress_ratios.empty()) {
        if (out.compress_ratios.size() != out.n_layers)
            return "qwen4exp: attention.compress_ratios length != block_count";
        for (uint32_t il = 0; il < out.n_layers; ++il)
            if ((out.compress_ratios[il] != 0) != out.is_full_attn(il))
                return "qwen4exp: compress_ratios schedule disagrees with full_attention_interval";
    }
    if (out.has_ple()) {
        for (int32_t l : out.ple_layers)
            if (l < 0 || uint32_t(l) >= out.n_layers)
                return "qwen4exp: ple.layers entry out of range";
        if (out.ple_ngram < 2)
            return "qwen4exp: ple.ngram_size < 2";
        if (out.ple_layer_multipliers.size() != out.ple_ngram)
            return "qwen4exp: ple.layer_multipliers size != ngram_size";
        // (ngram_size-1) tables (bigram..ngram), heads_per_ngram heads each.
        const size_t heads = size_t(out.ple_heads_per_ngram) * (out.ple_ngram - 1);
        if (out.ple_head_offsets.size() != heads || out.ple_head_vocab_sizes.size() != heads)
            return "qwen4exp: ple head offset/vocab arrays != heads_per_ngram*(ngram_size-1)";
        for (size_t i = 1; i < heads; ++i)
            if (out.ple_head_offsets[i] != out.ple_head_offsets[i - 1] + out.ple_head_vocab_sizes[i - 1])
                return "qwen4exp: ple.head_offsets not the cumulative sum of head_vocab_sizes";
    }
    return {};
}

std::string read_deepseek4_config(const GgufReader& g, DeepSeek4Config& out) {
    // Bespoke reader (mirrors read_gemma4_config's shape): `deepseek4` has no
    // feed_forward_length and no ordinary GQA geometry, so read_dense_config
    // would hard-error on a key that legitimately does not exist.
    auto u32 = [&](const char* suffix, uint32_t& dst) -> bool {
        const auto* kv = g.find_kv(std::string("deepseek4.") + suffix);
        if (!kv) return false;
        dst = uint32_t(kv->as_uint());
        return true;
    };
    auto f32 = [&](const char* suffix, float& dst) {
        if (const auto* kv = g.find_kv(std::string("deepseek4.") + suffix))
            dst = float(kv->as_float());
    };
    if (!u32("block_count", out.n_layers))              return "missing deepseek4.block_count";
    if (!u32("embedding_length", out.hidden))           return "missing deepseek4.embedding_length";
    if (!u32("attention.head_count", out.n_q_heads))    return "missing deepseek4.attention.head_count";
    if (!u32("attention.head_count_kv", out.n_kv_heads)) return "missing deepseek4.attention.head_count_kv";
    if (!u32("attention.key_length", out.head_dim))     return "missing deepseek4.attention.key_length";
    u32("attention.value_length", out.v_head_dim);
    u32("context_length", out.ctx_train);
    f32("attention.layer_norm_rms_epsilon", out.rms_eps);
    // DeepSeek-V4-Flash-Vision-Exp trains with rms_norm_eps = 1e-20 (config.json),
    // but the GGUF converter wrote the 0731 value 1e-6 into the file. Override
    // seam until the vision config is plumbed: IE_DS4_RMS_EPS=<float>.
    if (const char* ev = std::getenv("IE_DS4_RMS_EPS"); ev && *ev) out.rms_eps = std::strtof(ev, nullptr);

    // MLA low-rank projections + the SWA window. NOTE: sliding_window (128) is
    // NOT run through sliding_window_unsupported — that guard exists to stop a
    // dense GGUF silently full-attending, and this arch has no forward at all
    // yet. The window is recorded so whoever writes the forward cannot miss it.
    u32("attention.q_lora_rank", out.q_lora_rank);
    u32("attention.output_lora_rank", out.o_lora_rank);
    u32("attention.output_group_count", out.o_groups);
    u32("attention.sliding_window", out.sliding_window);

    // Lightning indexer.
    u32("attention.indexer.head_count", out.indexer_n_heads);
    u32("attention.indexer.key_length", out.indexer_head_dim);
    u32("attention.indexer.top_k", out.indexer_top_k);

    // Per-layer compression schedule — copied VERBATIM, length not validated
    // against n_layers (the real file has 46 entries for 43 layers; see header).
    if (const auto* kv = g.find_kv("deepseek4.attention.compress_ratios");
        kv && kv->type == GgufValueType::kArray) {
        const auto a = kv->as_pod_array<int32_t>();
        out.compress_ratios.assign(a.begin(), a.end());
    }
    f32("attention.compress_rope_freq_base", out.compress_rope_theta);

    // YaRN RoPE — same field shape as GptOssConfig (freq_scale = 1/factor,
    // ext_factor 1 only when scaling.type == "yarn") plus the two beta terms.
    if (!u32("rope.dimension_count", out.rope_dim)) out.rope_dim = out.head_dim;
    f32("rope.freq_base", out.rope_theta);
    if (const auto* kv = g.find_kv("deepseek4.rope.scaling.factor"))
        if (const double f = kv->as_float(); f != 0.0) out.rope_freq_scale = 1.0f / float(f);
    u32("rope.scaling.original_context_length", out.rope_orig_ctx);
    if (const auto* kv = g.find_kv("deepseek4.rope.scaling.type"))
        if (kv->as_string() == "yarn") out.rope_ext_factor = 1.0f;
    f32("rope.scaling.yarn_beta_fast", out.yarn_beta_fast);
    f32("rope.scaling.yarn_beta_slow", out.yarn_beta_slow);

    // MoE.
    if (!u32("expert_count", out.n_experts))                   return "missing deepseek4.expert_count";
    if (!u32("expert_used_count", out.n_experts_used))         return "missing deepseek4.expert_used_count";
    if (!u32("expert_feed_forward_length", out.expert_ffn))    return "missing deepseek4.expert_feed_forward_length";
    u32("expert_shared_count", out.n_shared_experts);
    u32("expert_gating_func", out.expert_gating_func);
    f32("expert_weights_scale", out.expert_weights_scale);
    if (const auto* kv = g.find_kv("deepseek4.expert_weights_norm"))
        out.expert_weights_norm = kv->as_bool();
    if (out.n_experts == 0 || out.n_experts_used == 0 || out.expert_ffn == 0)
        return "deepseek4: zero expert_count/used_count/ffn";
    if (out.n_experts_used > out.n_experts)
        return "deepseek4: expert_used_count > expert_count";

    // Per-layer SwiGLU clamp schedules (f32 arrays, length n_layers).
    auto f32_array = [&](const char* suffix, std::vector<float>& dst) {
        if (const auto* kv = g.find_kv(std::string("deepseek4.") + suffix);
            kv && kv->type == GgufValueType::kArray) {
            const auto a = kv->as_pod_array<float>();
            dst.assign(a.begin(), a.end());
        }
    };
    f32_array("swiglu_clamp_exp", out.swiglu_clamp_exp);
    f32_array("swiglu_clamp_shexp", out.swiglu_clamp_shexp);

    // Hyper-connections + hash routing.
    u32("hyper_connection.count", out.hc_count);
    u32("hyper_connection.sinkhorn_iterations", out.hc_sinkhorn_iters);
    f32("hyper_connection.epsilon", out.hc_eps);
    u32("hash_layer_count", out.hash_layer_count);

    const auto* toks = g.find_kv("tokenizer.ggml.tokens");
    if (!toks || toks->type != GgufValueType::kArray) return "missing tokenizer.ggml.tokens";
    out.vocab = uint32_t(toks->n_array);
    if (out.n_layers == 0) return "deepseek4: zero block_count";
    return {};
}

std::string read_qwen3next_config(const GgufReader& g, Qwen3NextConfig& out) {
    auto& h = out.hybrid;
    // Dense attention block (FULL-ATTN head counts via attention.key_length=256,
    // vocab, NEOX rope n_rot 64 / θ, eps). feed_forward_length is present but
    // unused (experts replace the dense FFN).
    if (auto m = read_dense_config(g, "qwen3next", h.dense); !m.empty()) return m;
    h.dense.arch = ModelArch::kQwen3Next;

    auto key = [](const char* suffix) { return std::string("qwen3next.") + suffix; };
    auto need_u32 = [&](const char* suffix, uint32_t& dst) -> bool {
        const auto* kv = g.find_kv(key(suffix));
        if (!kv) return false;
        dst = uint32_t(kv->as_uint());
        return true;
    };

    // full_attention_interval optional (llama.cpp defaults 4).
    need_u32("full_attention_interval", h.full_attn_interval);
    if (h.full_attn_interval == 0) h.full_attn_interval = 4;

    // Gated-DeltaNet ("ssm") params REQUIRED (mirrors read_qwen35_config body).
    if (!need_u32("ssm.inner_size",     h.ssm_inner))     return "missing " + key("ssm.inner_size");
    if (!need_u32("ssm.state_size",     h.ssm_state))     return "missing " + key("ssm.state_size");
    if (!need_u32("ssm.group_count",    h.ssm_n_k_heads)) return "missing " + key("ssm.group_count");
    if (!need_u32("ssm.time_step_rank", h.ssm_n_v_heads)) return "missing " + key("ssm.time_step_rank");
    if (!need_u32("ssm.conv_kernel",    h.ssm_conv_kernel)) return "missing " + key("ssm.conv_kernel");
    if (h.ssm_n_v_heads == 0) return key("ssm.time_step_rank") + " is zero";
    h.ssm_v_head_dim = h.ssm_inner / h.ssm_n_v_heads;
    need_u32("nextn_predict_layers", h.nextn_predict_layers);

    // MoE (parameterized like qwen3moe) + the shared expert (crown pattern).
    if (!need_u32("expert_count",               out.n_experts))      return "missing " + key("expert_count");
    if (!need_u32("expert_used_count",          out.n_experts_used)) return "missing " + key("expert_used_count");
    if (!need_u32("expert_feed_forward_length", out.expert_ffn))     return "missing " + key("expert_feed_forward_length");
    need_u32("expert_shared_feed_forward_length", out.shared_expert_ffn);  // optional (0 = no shared expert)
    if (out.n_experts == 0 || out.n_experts_used == 0 || out.expert_ffn == 0)
        return "qwen3next: zero expert_count/used_count/ffn";
    if (out.n_experts_used > out.n_experts)
        return "qwen3next: expert_used_count > expert_count";
    return {};
}

}  // namespace ie
