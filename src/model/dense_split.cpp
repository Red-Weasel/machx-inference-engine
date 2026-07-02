// src/model/dense_split.cpp — P-B: multi-GPU layer-split dense forward (prefill).
// ADDITIVE — never touches DenseModel. Re-uses dense_dispatch.hpp upload + GEMV
// helpers and the src/ops leaf functions, threading a per-device queue/workspace.
// The per-layer body is a faithful copy of DenseModel::forward's T>1 prefill path
// (so single-GPU and 2-GPU-split logits are BIT-IDENTICAL).

#include "ie/dense_split.hpp"

#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include "dense_dispatch.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace ie {

// ---- LayerPlan --------------------------------------------------------
LayerPlan LayerPlan::contiguous(uint32_t n_layers, uint32_t n_dev) {
    LayerPlan p;
    p.dev_of_layer.resize(n_layers);
    if (n_dev == 0) n_dev = 1;
    for (uint32_t L = 0; L < n_layers; ++L)
        p.dev_of_layer[L] = uint32_t(uint64_t(L) * n_dev / n_layers);
    p.embed_dev = 0;
    p.head_dev  = n_dev - 1;
    return p;
}
uint32_t LayerPlan::n_dev() const {
    uint32_t m = std::max(embed_dev, head_dev);
    for (uint32_t d : dev_of_layer) m = std::max(m, d);
    return m + 1;
}

DenseModelSplit::~DenseModelSplit() { free_all(); }

void DenseModelSplit::free_all() {
    if (!fleet_) return;
    for (uint32_t d = 0; d < owned_.size(); ++d)
        for (void* p : owned_[d]) fleet_->dev(d).free(p);
    owned_.clear();
}

std::string DenseModelSplit::ensure_ws(uint32_t dev, uint32_t max_T) {
    Workspace& w = ws_[dev];
    if (max_T <= w.T) return {};
    auto& alloc = fleet_->dev(dev);
    auto ah = [&](size_t n) {
        auto* p = static_cast<sycl::half*>(alloc.malloc(n * sizeof(sycl::half)));
        if (p) owned_[dev].push_back(p);
        return p;
    };
    const uint32_t H = cfg_.hidden, HD = cfg_.head_dim;
    const uint32_t N_q = cfg_.n_q_heads * HD, N_kv = cfg_.n_kv_heads * HD, F = cfg_.ffn;
    const uint64_t T = max_T;
    w.x          = ah(T * H);
    w.x_normed   = ah(T * H);
    w.q          = ah(T * N_q);
    w.k          = ah(T * N_kv);
    w.v          = ah(T * N_kv);
    w.attn_out   = ah(T * N_q);
    w.attn_block = ah(T * H);
    w.gate       = ah(T * F);
    w.up         = ah(T * F);
    w.h          = ah(T * F);
    w.positions  = static_cast<int32_t*>(alloc.malloc(T * sizeof(int32_t)));
    if (w.positions) owned_[dev].push_back(w.positions);
    if (!w.x || !w.x_normed || !w.q || !w.k || !w.v || !w.attn_out ||
        !w.attn_block || !w.gate || !w.up || !w.h || !w.positions)
        return "split workspace alloc failed on dev " + std::to_string(dev);
    w.T = max_T;
    return {};
}

std::string DenseModelSplit::load(DeviceFleet& fleet, const LayerPlan& plan,
                                  const GgufReader& g, const DenseConfig& cfg) {
    fleet_ = &fleet;
    plan_  = plan;
    cfg_   = cfg;
    n_dev_ = plan.n_dev();
    if (n_dev_ > fleet.size())
        return "LayerPlan uses " + std::to_string(n_dev_) + " devices but fleet has " +
               std::to_string(fleet.size());
    if (cfg.n_layers == 0 || cfg.head_dim < 16 || (cfg.head_dim & (cfg.head_dim - 1)))
        return "split: bad config";
    if (plan.dev_of_layer.size() != cfg.n_layers)
        return "split: plan layer count mismatch";

    owned_.assign(n_dev_, {});
    ws_.assign(n_dev_, {});
    kv_.resize(n_dev_);
    rope_freqs_per_dev_.assign(n_dev_, nullptr);
    dev_bytes_.assign(n_dev_, 0);
    layers_.assign(cfg.n_layers, {});
    local_idx_.assign(cfg.n_layers, 0);

    char buf[64];
    auto T = [&](const char* name) { return g.find_tensor(name); };
    auto Tlayer = [&](uint32_t L, const char* name) {
        std::snprintf(buf, sizeof(buf), "blk.%u.%s", L, name);
        return g.find_tensor(buf);
    };
    std::string err;

    // top-level: token_embd → embed_dev; output_norm + output → head_dev.
    {
        const auto* ti = T("token_embd.weight");
        if (!ti) return "token_embd: not found";
        if (ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ6_K)
            return "token_embd: unsupported dtype";
        token_embd_dtype_ = ti->dtype;
        token_embd_ = dense::upload<void>(fleet.dev(plan.embed_dev), ti,
                                          owned_[plan.embed_dev], err, ti->dtype);
        if (!err.empty()) return "token_embd: " + err;
        dev_bytes_[plan.embed_dev] += ti->nbytes;
    }
    output_norm_ = dense::upload<float>(fleet.dev(plan.head_dev), T("output_norm.weight"),
                                        owned_[plan.head_dev], err, DType::kF32);
    if (!err.empty()) return "output_norm: " + err;
    {
        const auto* ti = T("output.weight");
        if (!ti) { output_ = token_embd_; output_dtype_ = token_embd_dtype_;
                   if (plan.head_dev != plan.embed_dev)
                       return "split: tied embeddings need embed_dev == head_dev (single-device head)"; }
        else {
            if (ti->dtype != DType::kQ4_K && ti->dtype != DType::kQ6_K)
                return "output: unsupported dtype";
            output_dtype_ = ti->dtype;
            output_ = dense::upload<void>(fleet.dev(plan.head_dev), ti,
                                          owned_[plan.head_dev], err, ti->dtype);
            if (!err.empty()) return "output: " + err;
            dev_bytes_[plan.head_dev] += ti->nbytes;
        }
    }
    // rope_freqs → every device (used by rope_partial_ff on each layer's device).
    if (const auto* ti = T("rope_freqs.weight")) {
        const uint64_t n = ti->nbytes / sizeof(float);
        if (n != cfg.rope_dim / 2u) return "rope_freqs: count mismatch";
        for (uint32_t d = 0; d < n_dev_; ++d) {
            rope_freqs_per_dev_[d] = dense::upload<float>(fleet.dev(d), ti, owned_[d], err, DType::kF32);
            if (!err.empty()) return "rope_freqs dev " + std::to_string(d) + ": " + err;
        }
    }

    // llama Q/K un-permute (same as DenseModel).
    const bool unpermute = (cfg.arch == ModelArch::kLlama3);
    const std::vector<uint32_t> perm_q = unpermute ? llama_qk_unpermute_rows(cfg.n_q_heads,  cfg.head_dim) : std::vector<uint32_t>{};
    const std::vector<uint32_t> perm_k = unpermute ? llama_qk_unpermute_rows(cfg.n_kv_heads, cfg.head_dim) : std::vector<uint32_t>{};

    // per-device local layer counters (for KV sizing + local index).
    std::vector<uint32_t> per_dev_count(n_dev_, 0);

    for (uint32_t L = 0; L < cfg.n_layers; ++L) {
        const uint32_t d = plan.dev_of_layer[L];
        local_idx_[L] = per_dev_count[d]++;
        auto& alloc = fleet.dev(d);
        auto& own   = owned_[d];
        auto& w = layers_[L];
        auto LQ = [&](const char* name) { return dense::upload_quant_dense(alloc, Tlayer(L, name), own, err); };
        auto optf = [&](const char* name, float*& dst) -> std::string {
            if (const auto* ti = Tlayer(L, name)) {
                dst = dense::upload<float>(alloc, ti, own, err, DType::kF32);
                if (!err.empty()) return std::string("layer ") + std::to_string(L) + " " + name + ": " + err;
            }
            return {};
        };
        w.attn_norm = dense::upload<float>(alloc, Tlayer(L, "attn_norm.weight"), own, err, DType::kF32);
        if (!err.empty()) return "attn_norm: " + err;
        w.ffn_norm  = dense::upload<float>(alloc, Tlayer(L, "ffn_norm.weight"),  own, err, DType::kF32);
        if (!err.empty()) return "ffn_norm: " + err;
        w.attn_q = unpermute ? dense::upload_quant_dense_permuted(alloc, Tlayer(L, "attn_q.weight"), own, err, perm_q)
                             : LQ("attn_q.weight");   if (!err.empty()) return "attn_q: " + err;
        w.attn_k = unpermute ? dense::upload_quant_dense_permuted(alloc, Tlayer(L, "attn_k.weight"), own, err, perm_k)
                             : LQ("attn_k.weight");   if (!err.empty()) return "attn_k: " + err;
        w.attn_v      = LQ("attn_v.weight");      if (!err.empty()) return "attn_v: " + err;
        w.attn_output = LQ("attn_output.weight"); if (!err.empty()) return "attn_output: " + err;
        if (auto m = optf("attn_q_norm.weight", w.attn_q_norm); !m.empty()) return m;
        if (auto m = optf("attn_k_norm.weight", w.attn_k_norm); !m.empty()) return m;
        if (auto m = optf("attn_q.bias", w.attn_q_bias); !m.empty()) return m;
        if (auto m = optf("attn_k.bias", w.attn_k_bias); !m.empty()) return m;
        if (auto m = optf("attn_v.bias", w.attn_v_bias); !m.empty()) return m;
        w.ffn_gate = LQ("ffn_gate.weight"); if (!err.empty()) return "ffn_gate: " + err;
        w.ffn_up   = LQ("ffn_up.weight");   if (!err.empty()) return "ffn_up: " + err;
        w.ffn_down = LQ("ffn_down.weight"); if (!err.empty()) return "ffn_down: " + err;
        // rough weight-byte tally for reporting (the 7 matmul tensors dominate)
        for (const char* nm : {"attn_q.weight","attn_k.weight","attn_v.weight",
                               "attn_output.weight","ffn_gate.weight","ffn_up.weight","ffn_down.weight"})
            if (const auto* ti = Tlayer(L, nm)) dev_bytes_[d] += ti->nbytes;
    }

    // KV per device (dense: every layer is full-attn → n_layers_full = per-dev count).
    for (uint32_t d = 0; d < n_dev_; ++d) {
        if (per_dev_count[d] == 0) continue;
        KvCacheConfig kc{};
        kc.n_layers_full = per_dev_count[d];
        kc.n_kv_heads    = cfg.n_kv_heads;
        kc.max_ctx       = 2048;       // bring-up ctx; enough for the equality test
        kc.head_dim      = cfg.head_dim;
        if (auto m = kv_[d].init(fleet.dev(d), kc); !m.empty())
            return "kv dev " + std::to_string(d) + ": " + m;
    }
    return {};
}

std::string DenseModelSplit::forward(const int32_t* input_ids, uint32_t T,
                                     uint32_t start_pos, bool reset_kv,
                                     sycl::half* out_logits_host) {
    if (T == 0) return "T == 0";
    const uint32_t H = cfg_.hidden, HD = cfg_.head_dim;
    const uint32_t N_q = cfg_.n_q_heads * HD, N_kv = cfg_.n_kv_heads * HD, F = cfg_.ffn;
    const uint32_t rope_n = cfg_.rope_dim;
    const float eps = cfg_.rms_eps;

    for (uint32_t d = 0; d < n_dev_; ++d)
        if (auto m = ensure_ws(d, T); !m.empty()) return m;

    // positions [start_pos..] on every device + reset KV.
    std::vector<int32_t> pos(T);
    for (uint32_t t = 0; t < T; ++t) pos[t] = int32_t(start_pos + t);
    for (uint32_t d = 0; d < n_dev_; ++d) {
        fleet_->dev(d).queue().memcpy(ws_[d].positions, pos.data(), T * sizeof(int32_t)).wait();
        if (reset_kv) kv_[d].reset();
    }

    // embedding → ws_[embed_dev].x
    {
        auto& q = fleet_->dev(plan_.embed_dev).queue();
        // input_ids must be on the embed device.
        int32_t* d_ids = static_cast<int32_t*>(fleet_->dev(plan_.embed_dev).malloc(T * sizeof(int32_t)));
        if (!d_ids) return "d_ids alloc failed";
        q.memcpy(d_ids, input_ids, T * sizeof(int32_t)).wait();
        if (token_embd_dtype_ == DType::kQ4_K) embedding_lookup_q4k(q, d_ids, token_embd_, ws_[plan_.embed_dev].x, T, H);
        else                                    embedding_lookup_q6k(q, d_ids, token_embd_, ws_[plan_.embed_dev].x, T, H);
        q.wait();
        fleet_->dev(plan_.embed_dev).free(d_ids);
    }

    // Optional per-phase profiler (IE_SPLIT_PROFILE=1): times each device's layer
    // block + the inter-device copy, with a host sync at each boundary. Decode is
    // expected to be serial-pipeline + memory-bound (each card idle while the other
    // runs); this confirms the breakdown. Adds syncs → for measurement only.
    static const bool prof = std::getenv("IE_SPLIT_PROFILE") != nullptr;
    auto now_ms = [] { return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count(); };
    double t_copy = 0, t_dev[8] = {0};

    // device-by-device: copy residual in, run this device's layers.
    for (uint32_t d = 0; d < n_dev_; ++d) {
        Workspace& w = ws_[d];
        auto& q = fleet_->dev(d).queue();
        if (d > 0) {  // residual from device d-1 → device d
            const double tc = prof ? now_ms() : 0;
            fleet_->copy_across(d - 1, w.x, d, ws_[d - 1].x, uint64_t(T) * H * sizeof(sycl::half));
            if (prof) t_copy += now_ms() - tc;
        }
        const double td = prof ? now_ms() : 0;
        const float* ff = rope_freqs_per_dev_[d];
        for (uint32_t L = 0; L < cfg_.n_layers; ++L) {
            if (plan_.dev_of_layer[L] != d) continue;
            const auto& lw = layers_[L];
            const uint32_t li = local_idx_[L];
            // pre-attn norm
            rms_norm_f32w(q, w.x, lw.attn_norm, w.x_normed, T, H, eps);
            // QKV
            dense::gemv_q_T(q, w.x_normed, lw.attn_q, w.q, H, N_q,  T);
            dense::gemv_q_T(q, w.x_normed, lw.attn_k, w.k, H, N_kv, T);
            dense::gemv_q_T(q, w.x_normed, lw.attn_v, w.v, H, N_kv, T);
            if (lw.attn_q_bias) add_bias(q, w.q, lw.attn_q_bias, T, N_q);
            if (lw.attn_k_bias) add_bias(q, w.k, lw.attn_k_bias, T, N_kv);
            if (lw.attn_v_bias) add_bias(q, w.v, lw.attn_v_bias, T, N_kv);
            // QK-norm (qwen3) — skipped when null (llama/qwen2)
            if (lw.attn_q_norm) rms_norm_f32w(q, w.q, lw.attn_q_norm, w.q, T * cfg_.n_q_heads,  HD, eps);
            if (lw.attn_k_norm) rms_norm_f32w(q, w.k, lw.attn_k_norm, w.k, T * cfg_.n_kv_heads, HD, eps);
            // RoPE (ff variant when rope_freqs present)
            if (ff) {
                rope_partial_ff(q, w.q, w.positions, w.q, T, cfg_.n_q_heads,  HD, rope_n, cfg_.rope_theta, ff);
                rope_partial_ff(q, w.k, w.positions, w.k, T, cfg_.n_kv_heads, HD, rope_n, cfg_.rope_theta, ff);
            } else {
                rope_partial(q, w.q, w.positions, w.q, T, cfg_.n_q_heads,  HD, rope_n, cfg_.rope_theta);
                rope_partial(q, w.k, w.positions, w.k, T, cfg_.n_kv_heads, HD, rope_n, cfg_.rope_theta);
            }
            // attention (T>1 naive path) — KV slice = local layer index on this dev
            const uint64_t per_layer = uint64_t(cfg_.n_kv_heads) * kv_[d].config().max_ctx * HD;
            sycl::half* kc = kv_[d].k_ptr() + per_layer * li;
            sycl::half* vc = kv_[d].v_ptr() + per_layer * li;
            full_attention(q, w.q, w.k, w.v, kc, vc, w.attn_out, T, start_pos,
                           cfg_.n_q_heads, cfg_.n_kv_heads, HD, kv_[d].config().max_ctx);
            kv_[d].set_length(li, start_pos + T);
            // o-proj
            dense::gemv_q_T(q, w.attn_out, lw.attn_output, w.attn_block, N_q, H, T);
            // residual + ffn norm
            residual_add_rms_norm_fused(q, w.x, w.attn_block, lw.ffn_norm, w.x_normed, T, H, eps);
            // SwiGLU FFN
            dense::gemv_q_T(q, w.x_normed, lw.ffn_gate, w.gate, H, F, T);
            dense::gemv_q_T(q, w.x_normed, lw.ffn_up,   w.up,   H, F, T);
            swiglu(q, w.gate, w.up, w.h, uint64_t(T) * F);
            dense::gemv_q_T(q, w.h, lw.ffn_down, w.attn_block, F, H, T);
            residual_add(q, w.x, w.attn_block, w.x, uint64_t(T) * H);
        }
        q.wait();   // finish this device before the boundary copy reads its ws.x
        if (prof) t_dev[d] = now_ms() - td;
    }
    if (prof) {
        std::fprintf(stderr, "[split-prof T=%u] copy=%.2fms", T, t_copy);
        for (uint32_t d = 0; d < n_dev_; ++d) std::fprintf(stderr, " dev%u=%.2fms", d, t_dev[d]);
        std::fprintf(stderr, "\n");
    }

    // final norm + lm_head on head_dev → logits → host.
    {
        const uint32_t hd = plan_.head_dev;
        Workspace& w = ws_[hd];
        auto& alloc = fleet_->dev(hd);
        auto& q = alloc.queue();
        rms_norm_f32w(q, w.x, output_norm_, w.x_normed, T, H, eps);
        const sycl::half* last = w.x_normed + uint64_t(T - 1) * H;
        sycl::half* d_logits = static_cast<sycl::half*>(alloc.malloc(uint64_t(cfg_.vocab) * sizeof(sycl::half)));
        if (!d_logits) return "logits alloc failed";
        DenseQuantPtr lm{output_, output_dtype_};
        dense::gemv_q(q, last, lm, d_logits, H, cfg_.vocab).wait();
        q.memcpy(out_logits_host, d_logits, uint64_t(cfg_.vocab) * sizeof(sycl::half)).wait();
        alloc.free(d_logits);
    }
    return {};
}

}  // namespace ie
