// src/loaders/model_config.cpp — arch detection + dense metadata reader.
// P2 Task 1: the only novel bit is `<arch>.` key composition; everything
// else is straight KV reads off the already-parsed GgufReader table.
#include "ie/model_config.hpp"

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
