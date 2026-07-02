// src/model/gemma4_assistant.cpp — gemma4-assistant MTP draft head loader.
// See include/ie/gemma4_assistant.hpp. Forward (draft) lands in a follow-up.
#include "ie/gemma4_assistant.hpp"
#include "dense_dispatch.hpp"   // dense::upload<T>

#include <cstdio>
#include <cstring>

namespace ie {

std::string read_gemma4_assistant_config(const GgufReader& g, GemmaAssistantConfig& out) {
    auto key = [](const char* s) { return std::string("gemma4-assistant.") + s; };
    auto u32 = [&](const char* s, uint32_t& dst) -> bool {
        const auto* kv = g.find_kv(key(s)); if (!kv) return false;
        dst = uint32_t(kv->as_uint()); return true;
    };
    auto f32 = [&](const char* s, float& dst) {
        if (const auto* kv = g.find_kv(key(s))) dst = float(kv->as_float());
    };

    if (!u32("block_count", out.n_layers))      return "missing gemma4-assistant.block_count";
    if (!u32("embedding_length", out.hidden))   return "missing gemma4-assistant.embedding_length";
    if (!u32("feed_forward_length", out.ffn))   return "missing gemma4-assistant.feed_forward_length";
    if (!u32("attention.head_count", out.n_q_heads)) return "missing gemma4-assistant.attention.head_count";
    f32("attention.layer_norm_rms_epsilon", out.rms_eps);
    u32("attention.sliding_window", out.sliding_window);

    // Global vs SWA head/rope geometry.
    uint32_t hd_g = 512, hd_s = 256, nr_g = 512, nr_s = 256;
    u32("attention.key_length",       hd_g);
    u32("attention.key_length_swa",   hd_s);
    u32("rope.dimension_count",       nr_g);
    u32("rope.dimension_count_swa",   nr_s);
    float theta_g = 1e6f, theta_s = 1e4f;
    f32("rope.freq_base",     theta_g);
    f32("rope.freq_base_swa", theta_s);

    // vocab from the head's tok_embd (Q8_0 [hidden, vocab]); backbone from
    // nextn.post_projection [hidden, backbone].
    const auto* te = g.find_tensor("token_embd.weight");
    if (!te) return "missing token_embd.weight";
    out.vocab = uint32_t(te->shape[1]);
    const auto* pp = g.find_tensor("nextn.post_projection.weight");
    if (!pp) return "missing nextn.post_projection.weight";
    out.n_embd_backbone = uint32_t(pp->shape[1]);

    // Per-layer SWA pattern + KV head counts.
    const uint32_t L = out.n_layers;
    out.is_swa.assign(L, 1);
    if (const auto* kv = g.find_kv(key("attention.sliding_window_pattern"));
        kv && kv->type == GgufValueType::kArray && kv->n_array == L) {
        auto a = kv->as_pod_array<uint8_t>();
        for (uint32_t i = 0; i < L; ++i) out.is_swa[i] = a[i] ? 1 : 0;
    } else {
        for (uint32_t i = 0; i < L; ++i) out.is_swa[i] = (i + 1 == L) ? 0 : 1;  // last global
    }
    out.n_kv_heads.assign(L, 0);
    if (const auto* kv = g.find_kv(key("attention.head_count_kv"));
        kv && kv->type == GgufValueType::kArray && kv->n_array == L) {
        auto a = kv->as_pod_array<int32_t>();
        for (uint32_t i = 0; i < L; ++i) out.n_kv_heads[i] = uint32_t(a[i]);
    } else if (const auto* kv = g.find_kv(key("attention.head_count_kv"))) {
        const uint32_t v = uint32_t(kv->as_uint());
        out.n_kv_heads.assign(L, v);
    } else {
        return "missing gemma4-assistant.attention.head_count_kv";
    }
    out.head_dim.assign(L, 0); out.n_rot.assign(L, 0); out.rope_theta.assign(L, 0);
    for (uint32_t i = 0; i < L; ++i) {
        const bool swa = out.is_swa[i];
        out.head_dim[i]   = swa ? hd_s : hd_g;
        out.n_rot[i]      = swa ? nr_s : nr_g;
        out.rope_theta[i] = swa ? theta_s : theta_g;
    }
    return {};
}

Gemma4AssistantHead::~Gemma4AssistantHead() {
    if (alloc_) for (void* p : owned_) if (p) alloc_->free(p);
}

std::string Gemma4AssistantHead::load(DeviceAllocator& alloc, const GgufReader& g,
                                      const GemmaAssistantConfig& cfg) {
    cfg_ = cfg;
    alloc_ = &alloc;
    std::string err;
    char buf[128];

    auto Tt = [&](const char* n) { return g.find_tensor(n); };
    auto Tl = [&](uint32_t l, const char* n) {
        std::snprintf(buf, sizeof(buf), "blk.%u.%s", l, n);
        return g.find_tensor(buf);
    };
    auto up_f32 = [&](const GgufTensorInfo* t, const char* what) -> float* {
        if (!t) { err = std::string(what) + ": not found"; return nullptr; }
        auto* p = dense::upload<float>(alloc, t, owned_, err, DType::kF32);
        if (!err.empty()) err = std::string(what) + ": " + err;
        return p;
    };
    auto up_q = [&](const GgufTensorInfo* t, const char* what) -> DenseQuantPtr {
        if (!t) { err = std::string(what) + ": not found"; return {}; }
        DenseQuantPtr dq{nullptr, t->dtype};
        dq.p = dense::upload<void>(alloc, t, owned_, err, t->dtype);
        if (!err.empty()) { err = std::string(what) + ": " + err; return {}; }
        return dq;
    };

    // top-level: tok_embd (Q8_0, head's output/logits), output_norm, nextn projs.
    if (const auto* te = Tt("token_embd.weight")) {
        tok_embd_dt_ = te->dtype;
        tok_embd_ = dense::upload<void>(alloc, te, owned_, err, te->dtype);
        if (!err.empty()) return "token_embd: " + err;
    } else return "token_embd: not found";
    output_norm_ = up_f32(Tt("output_norm.weight"), "output_norm"); if (!err.empty()) return err;
    pre_proj_  = up_q(Tt("nextn.pre_projection.weight"),  "nextn.pre_projection");  if (!err.empty()) return err;
    post_proj_ = up_q(Tt("nextn.post_projection.weight"), "nextn.post_projection"); if (!err.empty()) return err;
    // rope_freqs (freq_factors for the partial NEOX rope on GLOBAL layers) is a
    // single top-level [n_rot_global/2] tensor, shared across global layers.
    float* rope_freqs_shared = nullptr;
    if (const auto* rf = Tt("rope_freqs.weight")) {
        rope_freqs_shared = up_f32(rf, "rope_freqs"); if (!err.empty()) return err;
    }

    layers_.assign(cfg.n_layers, {});
    for (uint32_t l = 0; l < cfg.n_layers; ++l) {
        Layer& w = layers_[l];
        w.is_swa     = cfg.is_swa[l] != 0;
        w.head_dim   = cfg.head_dim[l];
        w.n_kv       = cfg.n_kv_heads[l];
        w.n_rot      = cfg.n_rot[l];
        w.rope_theta = cfg.rope_theta[l];

        w.attn_norm      = up_f32(Tl(l, "attn_norm.weight"), "attn_norm");           if (!err.empty()) return err;
        w.attn_q_norm    = up_f32(Tl(l, "attn_q_norm.weight"), "attn_q_norm");       if (!err.empty()) return err;
        w.post_attn_norm = up_f32(Tl(l, "post_attention_norm.weight"), "post_attn"); if (!err.empty()) return err;
        w.ffn_norm       = up_f32(Tl(l, "ffn_norm.weight"), "ffn_norm");             if (!err.empty()) return err;
        w.post_ffw_norm  = up_f32(Tl(l, "post_ffw_norm.weight"), "post_ffw_norm");   if (!err.empty()) return err;
        w.out_scale      = up_f32(Tl(l, "layer_output_scale.weight"), "out_scale");  if (!err.empty()) return err;
        if (!w.is_swa) w.rope_freqs = rope_freqs_shared;  // global layers: shared top-level freq_factors
        w.wq       = up_q(Tl(l, "attn_q.weight"),     "attn_q");     if (!err.empty()) return err;
        w.wo       = up_q(Tl(l, "attn_output.weight"), "attn_output"); if (!err.empty()) return err;
        w.ffn_gate = up_q(Tl(l, "ffn_gate.weight"),   "ffn_gate");   if (!err.empty()) return err;
        w.ffn_up   = up_q(Tl(l, "ffn_up.weight"),     "ffn_up");     if (!err.empty()) return err;
        w.ffn_down = up_q(Tl(l, "ffn_down.weight"),   "ffn_down");   if (!err.empty()) return err;
    }

    loaded_ = true;
    std::printf("[mtp-head] loaded gemma4-assistant: %u layers, hidden=%u, backbone=%u, vocab=%u\n",
                cfg.n_layers, cfg.hidden, cfg.n_embd_backbone, cfg.vocab);
    return {};
}

}  // namespace ie
