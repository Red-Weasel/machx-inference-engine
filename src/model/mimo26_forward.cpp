// src/model/mimo26_forward.cpp — MiMo-V2.6 forward (P2 bring-up). See include/ie/mimo26_forward.hpp.
#include "ie/mimo26_forward.hpp"

#include "ie/kernel_profiler.hpp"
#include "ie/mimo26_ops.hpp"
#include "ie/ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>

namespace ie {

namespace {

double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// A safetensors F32 tensor into a host vector (the data region has no alignment guarantee).
std::vector<float> f32_vec(const Ds41Tensor& t) {
    std::vector<float> v(size_t(t.w->numel()));
    std::memcpy(v.data(), t.w->data, v.size() * 4);
    return v;
}
std::vector<float> bf16_vec(const Ds41Tensor& t) {
    std::vector<float> v(size_t(t.w->numel()));
    mimo26_bf16_to_f32(t.w->data, uint32_t(v.size()), v.data());
    return v;
}
std::vector<sycl::half> to_half(const std::vector<float>& v) {
    std::vector<sycl::half> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = sycl::half(v[i]);
    return h;
}

}  // namespace

Mimo26Forward::~Mimo26Forward() { free_all(); }

template <class T> T* Mimo26Forward::dev(Card& c, size_t n) {
    T* p = sycl::malloc_device<T>(n, *c.q);
    if (p) { c.owned.push_back(p); c.bytes += n * sizeof(T); }
    return p;
}

uint64_t Mimo26Forward::vram_bytes(size_t card) const {
    if (card >= cards_.size()) return 0;
    return cards_[card]->bytes + (cards_[card]->tier_on ? cards_[card]->tier.vram_bytes() : 0);
}
uint64_t Mimo26Forward::pinned_bytes() const {
    uint64_t b = 0;
    for (const auto& c : cards_) if (c->tier_on) b += c->tier.pinned_bytes();
    return b;
}

void Mimo26Forward::free_all() {
    for (auto& c : cards_) {
        if (!c->q) continue;
        c->q->wait();
        if (c->tier_on) { c->tier.free_storage(*c->q); c->tier_on = false; }
        for (void* p : c->owned) sycl::free(p, *c->q);
        c->owned.clear(); c->bytes = 0;
    }
    cards_.clear();
}

std::string Mimo26Forward::upload_card(Card& c) {
    const auto& m = *m_; const auto& cfg = m.config();
    const uint32_t H = cfg.dim, E = cfg.n_routed_experts, FI = cfg.inter_dim;
    sycl::queue& q = *c.q;
    auto up32 = [&](const std::vector<float>& v) -> float* {
        float* p = dev<float>(c, v.size()); if (p) q.memcpy(p, v.data(), v.size() * 4).wait(); return p; };
    auto up16 = [&](const std::vector<float>& v) -> sycl::half* {
        const auto h = to_half(v);
        sycl::half* p = dev<sycl::half>(c, h.size()); if (p) q.memcpy(p, h.data(), h.size() * 2).wait(); return p; };
    auto up8 = [&](const std::vector<uint8_t>& v) -> uint8_t* {
        uint8_t* p = dev<uint8_t>(c, v.size()); if (p) q.memcpy(p, v.data(), v.size()).wait(); return p; };
    // P4 lever 3: the FP8 weights stay FP8 on the card (a chunk dequantises one at a time into wscratch)
    static const bool fp8_dense = [] { const char* v = std::getenv("IE_MIMO26_FP8_DENSE"); return !(v && *v == '0'); }();
    if (fp8_dense && cfg.fp8_block_k != 128) return "FP8-resident dense weights need 128-column scale blocks (IE_MIMO26_FP8_DENSE=0)";
    uint64_t scratch_elems = 0;

    c.dense.assign(c.L1 - c.L0, Dense{});
    std::vector<float> f;
    for (uint32_t L = c.L0; L < c.L1; ++L) {
        const auto& Lw = m.layers()[L]; Dense& d = c.dense[L - c.L0];
        d.attn_norm = up32(bf16_vec(Lw.attn_norm));
        d.ffn_norm  = up32(bf16_vec(Lw.ffn_norm));
        if (Lw.has_sink) d.sink = up32(bf16_vec(Lw.sink));
        uint32_t tp = 0;
        // IE_MIMO26_QKV_TP_SWA (diagnostic): the shard count to read the SWA layers' fused qkv rows as (0 = detect: 4)
        static const uint32_t tp_swa = [] { const char* v = std::getenv("IE_MIMO26_QKV_TP_SWA"); return v && *v ? uint32_t(std::atoi(v)) : 0u; }();
        if (fp8_dense) {
            std::vector<uint8_t> w8; std::vector<float> sr;
            if (auto e = mimo26_qkv_fp8_rows(Lw.qkv, cfg.q_rows(), cfg.k_rows(L), cfg.v_rows(L), cfg.fp8_block_n, w8, sr, &tp, cfg.is_swa(L) ? tp_swa : 0u); !e.empty())
                return "layer " + std::to_string(L) + ": " + e;
            d.qkv8 = up8(w8); d.qkv_s = up32(sr);
            scratch_elems = std::max<uint64_t>(scratch_elems, w8.size());
        } else {
            if (auto e = mimo26_qkv_dequant_f32(Lw.qkv, cfg.q_rows(), cfg.k_rows(L), cfg.v_rows(L), cfg.fp8_block_n, f, &tp, cfg.is_swa(L) ? tp_swa : 0u); !e.empty())
                return "layer " + std::to_string(L) + ": " + e;
            d.qkv = up16(f);
        }
        d.o   = up16(bf16_vec(Lw.o_proj));
        if (Lw.moe) {
            d.router_w = up32(bf16_vec(Lw.gate_w));
            d.router_b = f32_vec(Lw.gate_bias);
            if (d.router_b.size() != E) return "layer " + std::to_string(L) + ": router bias size";
        } else {
            if (fp8_dense) {
                for (auto [t, w8d, srd] : {std::tuple{&Lw.mlp_gate, &d.gate8, &d.gate_s}, std::tuple{&Lw.mlp_up, &d.up8, &d.up_s}, std::tuple{&Lw.mlp_down, &d.down8, &d.down_s}}) {
                    std::vector<uint8_t> w8; std::vector<float> sr;
                    if (auto e = mimo26_fp8_rows(*t, cfg.fp8_block_n, cfg.fp8_block_k, w8, sr); !e.empty()) return "layer " + std::to_string(L) + ": " + e;
                    *w8d = up8(w8); *srd = up32(sr);
                    scratch_elems = std::max<uint64_t>(scratch_elems, w8.size());
                }
            } else {
                for (auto [t, dst] : {std::pair{&Lw.mlp_gate, &d.gate}, std::pair{&Lw.mlp_up, &d.up}, std::pair{&Lw.mlp_down, &d.down}}) {
                    if (auto e = mimo26_fp8_dequant_f32(*t, cfg.fp8_block_n, cfg.fp8_block_k, f); !e.empty()) return "layer " + std::to_string(L) + ": " + e;
                    *dst = up16(f);
                }
            }
        }
        const uint32_t n_kv = cfg.n_kv(L);
        // SWA: a ring of ring_ slots (or linear); full: linear to max_ctx, V rows padded to head_dim for the XMX FA-2
        const bool swa = cfg.is_swa(L);
        const size_t slots = swa && ring_ ? ring_ : opt_.max_ctx;
        const uint32_t v_row = swa ? cfg.v_head_dim : cfg.head_dim;
        const size_t k_rows = size_t(n_kv) * slots + 64;   // mimo26_attention_prefill_xmx's 64-row slack
        d.k = dev<sycl::half>(c, k_rows * cfg.head_dim);
        d.v = dev<sycl::half>(c, size_t(n_kv) * slots * v_row);
        if (d.k && d.v) { q.memset(d.k, 0, k_rows * cfg.head_dim * 2); q.memset(d.v, 0, size_t(n_kv) * slots * v_row * 2).wait(); }
        if (!d.attn_norm || !d.ffn_norm || !(d.qkv || (d.qkv8 && d.qkv_s)) || !d.o || !d.k || !d.v || (Lw.has_sink && !d.sink) ||
            (Lw.moe ? !d.router_w : !((d.gate && d.up && d.down) || (d.gate8 && d.gate_s && d.up8 && d.up_s && d.down8 && d.down_s))))
            return "layer " + std::to_string(L) + ": device allocation failed";
    }
    // workspaces
    const uint32_t T = opt_.max_tokens, NQ = cfg.q_rows(), NK = cfg.n_kv_heads_swa * cfg.head_dim, NV = cfg.n_kv_heads_swa * cfg.v_head_dim;
    uint32_t Nqkv = 0; for (uint32_t L = c.L0; L < c.L1; ++L) Nqkv = std::max(Nqkv, cfg.qkv_rows(L));
    c.x = dev<float>(c, size_t(T) * H); c.xn32 = dev<float>(c, size_t(T) * H); c.qkv32 = dev<float>(c, size_t(T) * Nqkv);
    c.o32 = dev<float>(c, size_t(T) * H); c.moe = dev<float>(c, size_t(T) * H); c.rlogits = dev<float>(c, size_t(T) * E);
    c.xn16 = dev<sycl::half>(c, size_t(T) * H); c.Q = dev<sycl::half>(c, size_t(T) * NQ); c.K = dev<sycl::half>(c, size_t(T) * NK);
    c.V = dev<sycl::half>(c, size_t(T) * NV); c.attn = dev<sycl::half>(c, size_t(T) * cfg.n_heads * cfg.v_head_dim);
    c.pos = dev<int32_t>(c, T);
    bool has_full = false; for (uint32_t L = c.L0; L < c.L1; ++L) has_full |= !cfg.is_swa(L);
    if (has_full) {
        c.Vp = dev<sycl::half>(c, size_t(T) * cfg.n_kv_heads * cfg.head_dim);
        c.attn_p = dev<sycl::half>(c, size_t(T) * cfg.n_heads * cfg.head_dim);
        if (!c.Vp || !c.attn_p) return "workspace allocation failed (full-layer attention)";
    }
    c.partials = dev<float>(c, size_t(kDecodeRows) * cfg.n_heads * mimo26_decode_max_splits() * (cfg.v_head_dim + 2));
    if (scratch_elems && T > kDecodeRows) {
        c.wscratch = dev<sycl::half>(c, scratch_elems);
        if (!c.wscratch) return "workspace allocation failed (FP8 dequant scratch)";
    }
    bool has_dense = false; for (uint32_t L = c.L0; L < c.L1; ++L) has_dense |= !m.layers()[L].moe;
    if (has_dense) { c.gate32 = dev<float>(c, size_t(T) * FI); c.up32 = dev<float>(c, size_t(T) * FI); c.h16 = dev<sycl::half>(c, size_t(T) * FI); }
    if (&c == cards_.back().get()) {
        c.fnorm = up32(bf16_vec(m.final_norm));
        c.head = up16(bf16_vec(m.lm_head));
        c.head_out = dev<float>(c, size_t(kHeadRows) * cfg.vocab_size);
        if (!c.fnorm || !c.head || !c.head_out) return "head: device allocation failed";
    }
    if (!c.x || !c.xn32 || !c.qkv32 || !c.o32 || !c.moe || !c.rlogits || !c.xn16 || !c.Q || !c.K || !c.V || !c.attn || !c.pos || !c.partials ||
        (has_dense && (!c.gate32 || !c.up32 || !c.h16)))
        return "workspace allocation failed";

    // the expert tier over this card's MoE layers
    uint32_t first = c.L0; while (first < c.L1 && !m.layers()[first].moe) ++first;
    if (first < c.L1) {
        Ds41ExpertSource s;
        s.store = &m.store(); s.H = H; s.EF = cfg.moe_inter_dim; s.n_experts = E; s.top_k = cfg.n_activated_experts;
        s.n_text_layers = cfg.n_layers;
        s.layers.resize(m.layers().size());
        for (uint32_t L = 0; L < m.layers().size(); ++L) {
            const auto& Lw = m.layers()[L];
            if (Lw.moe) s.layers[L] = Ds41ExpertLayer{Lw.exp_w1.data(), Lw.exp_w3.data(), Lw.exp_w2.data()};
        }
        if (const char* fe = std::getenv("IE_MIMO26_EXPERT_FILE"); fe && *fe) s.expert_file = fe;
        s.fp32_out = true;   // no SwiGLU clamp: expert outputs can exceed fp16 (see Ds41ExpertSource::fp32_out)
        std::vector<std::vector<uint32_t>> ranking = opt_.ranking;
        if (ranking.empty()) {
            ranking.assign(cfg.n_layers, std::vector<uint32_t>(E));
            for (auto& r : ranking) std::iota(r.begin(), r.end(), 0u);
        }
        for (uint32_t L = first; L < c.L1; ++L) if (!m.layers()[L].moe) return "layer " + std::to_string(L) + ": a dense layer after the first MoE layer is not supported by the tier range";
        const uint32_t n_moe = c.L1 - first;
        uint32_t n_static = opt_.n_static;
        if (n_static == 0) {
            // AUTO (P4 lever 2): what this card has free now, less the tier's batch workspace (allocated inside tier.init)
            // and the reserve -- the runtime grows ~0.4 GiB during a forward (xpu-smi peak vs steady, ctx 131072)
            const auto& dev = q.get_device();
            if (!dev.has(sycl::aspect::ext_intel_free_memory)) return "auto static tier: the device does not report free memory; pass a static count";
            const uint64_t free_now = dev.get_info<sycl::ext::intel::info::device::free_memory>();
            double reserve_gib = 1.5;
            if (const char* v = std::getenv("IE_MIMO26_VRAM_RESERVE_GIB"); v && *v) reserve_gib = std::max(0.0, std::atof(v));
            const uint64_t reserve = uint64_t(reserve_gib * 1073741824.0) + ds4_expert_batch_ws_bytes(T, cfg.n_activated_experts, H, cfg.moe_inter_dim);
            Ds4SlotLayout lay;
            if (auto e = ds41_slot_layout(H, cfg.moe_inter_dim, lay); !e.empty()) return e;
            const uint64_t slots = free_now > reserve ? (free_now - reserve) / (uint64_t(n_moe) * lay.bytes) : 0;
            if (slots <= opt_.stream_slots) return "auto static tier: " + std::to_string(free_now >> 20) + " MiB free leaves no static slots";
            n_static = uint32_t(std::min<uint64_t>(E, slots - opt_.stream_slots));
        }
        // the claim can overshoot what the driver hands out (allocator padding): an AUTO count gives back a slot per layer
        // at a time -- those experts fall to the pinned tier, a speed cost, not a correctness one
        std::string te;
        for (uint32_t give = 0; ; ++give) {
            const uint32_t n_pinned = std::min(opt_.n_pinned, E - std::min(E, n_static));
            try { te = c.tier.init(q, s, first, n_moe, ranking, n_static, n_pinned, opt_.stream_slots, T); }
            catch (const sycl::exception& ex) { te = std::string("threw: ") + ex.what(); }
            if (te.empty() || opt_.n_static != 0 || give == 8 || n_static <= 1 ||
                (te.find("OUT_OF_DEVICE_MEMORY") == std::string::npos && te.find("malloc_device failed") == std::string::npos)) break;
            c.tier.free_storage(q);
            --n_static;
        }
        if (!te.empty()) return "tier: " + te;
        c.tier_on = true; c.tier_L0 = first;
    }
    return {};
}

std::string Mimo26Forward::set_feature_layers(std::vector<uint32_t> layers, uint32_t max_rows) {
    if (cards_.empty()) return "set_feature_layers: not initialised";
    feat_layers_ = std::move(layers); feat_max_ = feat_layers_.empty() ? 0 : std::min(max_rows, opt_.max_tokens); feat_rows_ = 0;
    for (uint32_t L : feat_layers_) if (L >= m_->config().n_layers) return "set_feature_layers: layer " + std::to_string(L) + " out of range";
    if (feat_layers_.empty()) return {};
    h_feat_.assign(feat_layers_.size() * size_t(feat_max_) * m_->config().dim, 0.f);
    return {};
}

std::string Mimo26Forward::init(const std::vector<sycl::queue*>& qs, const Mimo26Model& m, const Mimo26Options& o) {
    free_all();
    if (qs.empty()) return "no queues";
    if (!onednn_available()) return "this build has no oneDNN matmul (the dense GEMMs need it)";
    m_ = &m; opt_ = o;
    const auto& cfg = m.config();
    if (cfg.head_dim % 16 || cfg.v_head_dim % 16 || cfg.head_dim > 256 || cfg.v_head_dim > 256) return "head dims outside the attention kernel's range";
    // mimo26_attention_decode (every layer) and mimo26_attention_prefill_xmx (full layers) are built for MiMo's geometry
    if (cfg.head_dim != 192 || cfg.v_head_dim != 128 || cfg.n_heads > 16 * std::min(cfg.n_kv_heads, cfg.n_kv_heads_swa) ||
        64 % (cfg.n_heads / cfg.n_kv_heads))
        return "head geometry outside the attention kernels' (head_dim 192, v_head_dim 128, GQA <= 16, full-layer GQA dividing 64)";
    const uint32_t nL = cfg.n_layers;   // the MTP layers are not part of the forward
    // The pinned host tier is capped at MemAvailable - 40 GiB (V4.1's rule, which Dream's preflight assumes): the
    // experts past the cap go to the mmap tier -- slower, never an allocation failure. IE_MIMO26_PIN_RESERVE_GIB moves it.
    {
        uint64_t avail_kb = 0;
        if (std::FILE* mf = std::fopen("/proc/meminfo", "r")) {
            char line[256];
            while (std::fgets(line, sizeof line, mf)) if (std::sscanf(line, "MemAvailable: %lu kB", &avail_kb) == 1) break;
            std::fclose(mf);
        }
        Ds4SlotLayout lay;
        if (auto e = ds41_slot_layout(cfg.dim, cfg.moe_inter_dim, lay); !e.empty()) return e;
        uint32_t moe_layers = 0; for (uint32_t L = 0; L < nL; ++L) moe_layers += cfg.is_moe(L);
        double reserve_gib = 40.0;
        if (const char* v = std::getenv("IE_MIMO26_PIN_RESERVE_GIB"); v && *v) reserve_gib = std::max(0.0, std::atof(v));
        const double budget = double(avail_kb) * 1024.0 - reserve_gib * 1073741824.0;
        const uint32_t cap = budget <= 0 || !moe_layers ? 0u : uint32_t(budget / (double(lay.bytes) * moe_layers));
        const uint32_t room = cfg.n_routed_experts > opt_.n_static ? cfg.n_routed_experts - opt_.n_static : 0u;
        const uint32_t want = std::min(opt_.n_pinned, room);
        if (cap < want) {
            std::fprintf(stderr, "[mimo26] pinned tier capped at %u experts/layer (MemAvailable %.1f GiB - %.0f GiB reserve); the other %u go to the mmap tier\n",
                         cap, double(avail_kb) / 1048576.0, reserve_gib, want - cap);
            opt_.n_pinned = cap;
        } else opt_.n_pinned = want;
    }
    // the SWA ring: window + max_tokens slots holds every key a chunk of up to max_tokens rows needs (mimo26_attention)
    ring_ = (cfg.window + opt_.max_tokens + 63) / 64 * 64;   // a multiple of 64: mimo26_attention_prefill_xmx's K tiles
    if (const char* v = std::getenv("IE_MIMO26_SWA_LINEAR"); (v && *v == '1') || ring_ >= opt_.max_ctx) ring_ = 0;
    stats_.assign(nL, {});
    const size_t nc = qs.size();
    for (size_t i = 0; i < nc; ++i) {
        auto c = std::make_unique<Card>();
        c->q = qs[i];
        c->L0 = uint32_t(nL * i / nc); c->L1 = uint32_t(nL * (i + 1) / nc);
        cards_.push_back(std::move(c));
    }
    for (auto& c : cards_) {
        try { if (auto e = upload_card(*c); !e.empty()) return e; }
        catch (const sycl::exception& e) { return std::string("card ") + std::to_string(c->L0) + ": sycl: " + e.what(); }
    }
    h_x_.resize(size_t(opt_.max_tokens) * cfg.dim);
    h_rlogits_.resize(size_t(opt_.max_tokens) * cfg.n_routed_experts);
    h_ridx_.resize(size_t(opt_.max_tokens) * cfg.n_activated_experts);
    h_rw_.resize(h_ridx_.size());
    h_pos_.resize(opt_.max_tokens);
    n_pos_ = 0; hi_end_ = 0;
    return {};
}

std::string Mimo26Forward::run_card(Card& c, uint32_t T, uint32_t pos0, std::vector<float>& logits, bool all_rows) {
    const auto& m = *m_; const auto& cfg = m.config();
    const uint32_t H = cfg.dim, E = cfg.n_routed_experts, TK = cfg.n_activated_experts, FI = cfg.inter_dim;
    const uint32_t n_q = cfg.n_heads, hd = cfg.head_dim, hdv = cfg.v_head_dim, RD = cfg.rope_dim();
    sycl::queue& q = *c.q;
    const float inf = std::numeric_limits<float>::infinity();   // the tier's SwiGLU clamp: none
    auto prof = [&](const char* nm, sycl::event e) { if (g_profiler) [[unlikely]] g_profiler->push(nm, e, q); };

    q.memcpy(c.x, h_x_.data(), size_t(T) * H * 4);
    q.memcpy(c.pos, h_pos_.data(), size_t(T) * 4);
    for (uint32_t L = c.L0; L < c.L1; ++L) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto& Lw = m.layers()[L]; const Dense& d = c.dense[L - c.L0];
        const uint32_t n_kv = cfg.n_kv(L), Nqkv = cfg.qkv_rows(L);
        const bool swa = cfg.is_swa(L);   // (sinks only on these; the full layers have none -- add_full_attention_sink_bias false)
        const float theta = swa ? cfg.rope_theta_swa : cfg.rope_theta;
        const uint32_t window = swa ? cfg.window : 0u;
        static const bool split_decode = [] { const char* v = std::getenv("IE_MIMO26_SPLIT_DECODE"); return !(v && *v == '0'); }();

        // a dense projection: FP8-resident -> the FP8 GEMV for a step (<= kDecodeRows rows; profiled under its kernel name),
        // a chunk dequantises the weight into wscratch for oneDNN; fp16 copies (IE_MIMO26_FP8_DENSE=0) -> oneDNN directly
        auto dense = [&](const char* nm, const sycl::half* act, const uint8_t* w8, const float* sr, const sycl::half* w16, float* y, uint32_t N_, uint32_t K_) {
            if (!w8) { prof(nm, gemm_nt_f16_onednn(q, act, w16, y, T, N_, K_)); return; }
            if (T <= kDecodeRows) { mimo26_gemv_fp8(q, act, K_, w8, sr, y, T, K_, N_); return; }
            mimo26_fp8_to_f16(q, w8, sr, c.wscratch, N_, K_);
            prof(nm, gemm_nt_f16_onednn(q, act, c.wscratch, y, T, N_, K_));
        };
        mimo26_rms_norm(q, c.x, d.attn_norm, c.xn16, nullptr, T, H, cfg.norm_eps);
        dense("mimo26.gemm.qkv", c.xn16, d.qkv8, d.qkv_s, d.qkv, c.qkv32, Nqkv, H);
        if (swa) {
            mimo26_split_qkv(q, c.qkv32, c.Q, c.K, c.V, T, n_q, n_kv, hd, hdv);
            rope_partial(q, c.Q, c.pos, c.Q, T, n_q, hd, RD, theta);
            rope_partial(q, c.K, c.pos, c.K, T, n_kv, hd, RD, theta);
            // a step through the split-K decode attention, a chunk through mimo26_attention_prefill_xmx (ring or linear);
            // IE_MIMO26_SPLIT_DECODE=0 / IE_MIMO26_SWA_PREFILL=naive: mimo26_attention
            static const bool swa_xmx = [] { const char* v = std::getenv("IE_MIMO26_SWA_PREFILL"); return !(v && std::string(v) == "naive"); }();
            if (T <= kDecodeRows && split_decode)
                mimo26_attention_decode(q, c.Q, c.K, c.V, d.k, d.v, c.attn, c.partials, T, pos0, n_q, n_kv, hd, hdv, opt_.max_ctx, window, d.sink, {}, ring_);
            else if (T > kDecodeRows && swa_xmx)
                mimo26_attention_prefill_xmx(q, c.Q, c.K, c.V, d.k, d.v, c.attn, T, pos0, n_q, n_kv, opt_.max_ctx, window, d.sink, {}, 0, ring_);
            else
                mimo26_attention(q, c.Q, c.K, c.V, d.k, d.v, c.attn, T, pos0, n_q, n_kv, hd, hdv, opt_.max_ctx, window, d.sink, {}, ring_);
        } else {
            // full layer: V padded to hd rows (the cache's layout); a chunk (T > 8) through mimo26_attention_prefill_xmx
            // (IE_MIMO26_FULL_PREFILL=fa2: the generic XMX FA-2 on the padded V; =naive: mimo26_attention), a step through
            // the split-K decode attention (IE_MIMO26_SPLIT_DECODE=0: mimo26_attention, the serial-key bring-up kernel)
            mimo26_split_qkv(q, c.qkv32, c.Q, c.K, c.Vp, T, n_q, n_kv, hd, hdv, {}, hd);
            rope_partial(q, c.Q, c.pos, c.Q, T, n_q, hd, RD, theta);
            rope_partial(q, c.K, c.pos, c.K, T, n_kv, hd, RD, theta);
            static const int full_prefill = [] { const char* v = std::getenv("IE_MIMO26_FULL_PREFILL");
                                                 const std::string m = v ? v : ""; return m == "naive" ? 0 : m == "fa2" ? 1 : 2; }();
            if (T > kDecodeRows && full_prefill == 2) {
                mimo26_attention_prefill_xmx(q, c.Q, c.K, c.Vp, d.k, d.v, c.attn, T, pos0, n_q, n_kv, opt_.max_ctx, 0, nullptr, {}, hd);
            } else if (T > kDecodeRows && full_prefill == 1) {
                full_attention_fa2_prefill_xmx(q, c.Q, c.K, c.Vp, d.k, d.v, c.attn_p, T, pos0, n_q, n_kv, hd, opt_.max_ctx);
                mimo26_compact_heads(q, c.attn_p, c.attn, T, n_q, hd, hdv);
            } else if (T <= kDecodeRows && split_decode) {
                mimo26_attention_decode(q, c.Q, c.K, c.Vp, d.k, d.v, c.attn, c.partials, T, pos0, n_q, n_kv, hd, hdv, opt_.max_ctx, 0, nullptr, {}, 0, hd);
            } else {
                mimo26_attention(q, c.Q, c.K, c.Vp, d.k, d.v, c.attn, T, pos0, n_q, n_kv, hd, hdv, opt_.max_ctx, 0, nullptr, {}, 0, hd);
            }
        }
        prof("mimo26.gemm.o", gemm_nt_f16_onednn(q, c.attn, d.o, c.o32, T, H, n_q * hdv));
        mimo26_axpy(q, c.o32, cfg.value_scale, c.x, size_t(T) * H);
        mimo26_rms_norm(q, c.x, d.ffn_norm, c.xn16, c.xn32, T, H, cfg.norm_eps);
        if (!Lw.moe) {
            dense("mimo26.gemm.mlp_gate", c.xn16, d.gate8, d.gate_s, d.gate, c.gate32, FI, H);
            dense("mimo26.gemm.mlp_up", c.xn16, d.up8, d.up_s, d.up, c.up32, FI, H);
            mimo26_swiglu_f32(q, c.gate32, c.up32, c.h16, size_t(T) * FI);
            dense("mimo26.gemm.mlp_down", c.h16, d.down8, d.down_s, d.down, c.o32, H, FI);
            mimo26_axpy(q, c.o32, 1.f, c.x, size_t(T) * H);
        } else {
            // noaux_tc: sigmoid scores, select the top-k of (score + bias), weight by the unbiased scores normalised to 1
            // a chunk's router logits through oneDNN's fp32 GEMM (the per-(row, expert) dot kernel took 9 ms at 2048 rows)
            if (T > kDecodeRows) prof("mimo26.gemm.router", gemm_nt_f32_onednn(q, c.xn32, d.router_w, c.rlogits, T, E, H));
            else mimo26_router_logits(q, c.xn32, d.router_w, c.rlogits, T, H, E);
            q.memcpy(h_rlogits_.data(), c.rlogits, size_t(T) * E * 4).wait();
            std::vector<uint32_t> order(E);
            for (uint32_t t = 0; t < T; ++t) {
                const float* lg = h_rlogits_.data() + size_t(t) * E;
                float s[512];   // E <= 512 (forward() refuses more): Flash 256, Pro 384
                for (uint32_t e = 0; e < E; ++e) s[e] = 1.f / (1.f + std::exp(-lg[e]));
                std::iota(order.begin(), order.end(), 0u);
                std::partial_sort(order.begin(), order.begin() + TK, order.end(), [&](uint32_t a, uint32_t b) {
                    const float sa = s[a] + d.router_b[a], sb = s[b] + d.router_b[b];
                    return sa > sb || (sa == sb && a < b); });
                float sum = 0.f;
                for (uint32_t k = 0; k < TK; ++k) sum += s[order[k]];
                for (uint32_t k = 0; k < TK; ++k) {
                    h_ridx_[size_t(t) * TK + k] = int32_t(order[k]);
                    h_rw_[size_t(t) * TK + k] = cfg.norm_topk_prob ? s[order[k]] / sum * cfg.route_scale : s[order[k]] * cfg.route_scale;
                }
            }
            if (profiling_)
                for (uint32_t t = 0; t < T; ++t)
                    if (!profile_rows_ || profile_rows_[t])
                        for (uint32_t k = 0; k < TK; ++k) ++profile_[L][uint32_t(h_ridx_[size_t(t) * TK + k])];
            const auto tm = std::chrono::steady_clock::now();
            if (auto e = c.tier.moe(q, L, c.xn32, h_ridx_.data(), h_rw_.data(), T, c.moe, inf); !e.empty()) return "layer " + std::to_string(L) + " tier: " + e;
            stats_[L].moe_ms = ms_since(tm);
            const auto& ts = c.tier.last();
            stats_[L].experts_static = ts.experts_static; stats_[L].experts_pinned = ts.experts_pinned; stats_[L].experts_mmap = ts.experts_mmap;
            mimo26_axpy(q, c.moe, 1.f, c.x, size_t(T) * H);
        }
        if (!feat_layers_.empty()) {   // P5: this layer's residual rows for the drafter, the call's last rows
            const auto it = std::find(feat_layers_.begin(), feat_layers_.end(), L);
            if (it != feat_layers_.end()) {
                const uint32_t rows = std::min(T, feat_max_), fi = uint32_t(it - feat_layers_.begin());
                q.memcpy(h_feat_.data() + (size_t(fi) * rows) * H, c.x + size_t(T - rows) * H, size_t(rows) * H * 4);   // compact [n][rows][dim]
            }
        }
        q.wait();
        stats_[L].ms = ms_since(t0);
        // IE_MIMO26_CHECK=1 (diagnostic): the residual stream after every layer -- non-finite count and max |x| per row,
        // and for the dense FFN / the routed MoE output the largest |value| entering the residual. Prints only rows
        // that are non-finite or exceed IE_MIMO26_CHECK_MAX (default 30000: an fp16 GEMM operand would be close to overflow).
        // IE_MIMO26_DUMP=<dir> (diagnostic): the residual stream after every layer, x_L{L}_s{call}.f32 [T, H] -- `call`
        // counts forward() calls since init, so ie-mimo26-score's sequence k is s{k} (tools/mimo26/ref_forward.py --dump-dir)
        static const char* dump_dir = std::getenv("IE_MIMO26_DUMP");
        // IE_MIMO26_DUMP_LAYERS=a,b,c (diagnostic): only those layers (P5's DFlash study reads 5 of 48)
        static const std::vector<uint32_t> dump_layers = [] {
            std::vector<uint32_t> v; const char* e = std::getenv("IE_MIMO26_DUMP_LAYERS");
            for (const char* p = e; p && *p; ) { v.push_back(uint32_t(std::strtoul(p, const_cast<char**>(&p), 10))); if (*p == ',') ++p; else break; }
            return v; }();
        if (dump_dir && *dump_dir && (dump_layers.empty() || std::find(dump_layers.begin(), dump_layers.end(), L) != dump_layers.end())) {
            std::vector<float> hx(size_t(T) * H);
            q.memcpy(hx.data(), c.x, hx.size() * 4).wait();
            const std::string fn = std::string(dump_dir) + "/x_L" + std::to_string(L) + "_s" + std::to_string(n_calls_) + ".f32";
            if (std::FILE* df = std::fopen(fn.c_str(), "wb")) { std::fwrite(hx.data(), 4, hx.size(), df); std::fclose(df); }
        }
        static const bool check = [] { const char* v = std::getenv("IE_MIMO26_CHECK"); return v && *v == '1'; }();
        if (check) {
            static const float lim = [] { const char* v = std::getenv("IE_MIMO26_CHECK_MAX"); return v && *v ? float(std::atof(v)) : 30000.f; }();
            std::vector<float> hx(size_t(T) * H), hf(size_t(T) * H);
            q.memcpy(hx.data(), c.x, hx.size() * 4);
            q.memcpy(hf.data(), Lw.moe ? c.moe : c.o32, hf.size() * 4).wait();
            for (uint32_t t = 0; t < T; ++t) {
                uint32_t bad = 0; float mx = 0.f, mf = 0.f; uint32_t ai = 0;
                for (uint32_t k = 0; k < H; ++k) {
                    const float v = hx[size_t(t) * H + k], f = hf[size_t(t) * H + k];
                    if (!std::isfinite(v)) ++bad; else if (std::fabs(v) > mx) { mx = std::fabs(v); ai = k; }
                    if (std::isfinite(f)) mf = std::max(mf, std::fabs(f)); else ++bad;
                }
                if (bad || mx > lim || mf > lim) {
                    std::fprintf(stderr, "[mimo26 check] layer %u row %u (pos %u): non-finite %u, max|x| %.1f at dim %u, max|ffn out| %.1f\n",
                                 L, t, pos0 + t, bad, mx, ai, mf);
                    // IE_MIMO26_CHECK_DUMP=<dir>: the MoE input row and its routing, for an fp32 host recomputation
                    static const char* dump = std::getenv("IE_MIMO26_CHECK_DUMP");
                    if (dump && *dump && Lw.moe) {
                        std::vector<float> xr(H); q.memcpy(xr.data(), c.xn32 + size_t(t) * H, size_t(H) * 4).wait();
                        const std::string fn = std::string(dump) + "/moe_L" + std::to_string(L) + "_pos" + std::to_string(pos0 + t) + ".bin";
                        if (std::FILE* df = std::fopen(fn.c_str(), "wb")) {
                            std::fwrite(xr.data(), 4, H, df);
                            std::fwrite(h_ridx_.data() + size_t(t) * TK, 4, TK, df);
                            std::fwrite(h_rw_.data() + size_t(t) * TK, 4, TK, df);
                            std::fclose(df);
                        }
                    }
                }
            }
        }
    }
    if (&c == cards_.back().get()) {
        mimo26_rms_norm(q, c.x, c.fnorm, c.xn16, nullptr, T, H, cfg.norm_eps);
        const uint32_t rows = all_rows ? T : 1, r0 = all_rows ? 0 : T - 1;
        logits.resize(size_t(rows) * cfg.vocab_size);
        for (uint32_t r = 0; r < rows; r += kHeadRows) {
            const uint32_t n = std::min(kHeadRows, rows - r);
            prof("mimo26.gemm.head", gemm_nt_f16_onednn(q, c.xn16 + size_t(r0 + r) * H, c.head, c.head_out, n, cfg.vocab_size, H));
            q.memcpy(logits.data() + size_t(r) * cfg.vocab_size, c.head_out, size_t(n) * cfg.vocab_size * 4).wait();
        }
    } else {
        q.memcpy(h_x_.data(), c.x, size_t(T) * H * 4).wait();   // the residual stream crosses to the next card through the host
    }
    return {};
}

std::string Mimo26Forward::forward(const int32_t* ids, uint32_t T, uint32_t pos0, std::vector<float>& logits, bool all_rows) {
    if (cards_.empty()) return "not initialised";
    const auto& cfg = m_->config();
    if (T == 0 || T > opt_.max_tokens) return "T out of range";
    if (pos0 != n_pos_) return "pos0 " + std::to_string(pos0) + " != n_pos " + std::to_string(n_pos_) + " (call reset() to start over)";
    if (pos0 + T > opt_.max_ctx) return "context capacity " + std::to_string(opt_.max_ctx) + " exceeded";
    if (cfg.n_routed_experts > 512) return "more than 512 routed experts";
    const uint32_t H = cfg.dim;
    // embeddings: a host gather from the mmap (BF16 rows)
    const uint8_t* emb = m_->embed.w->data;
    for (uint32_t t = 0; t < T; ++t) {
        const int32_t id = ids[t];
        if (id < 0 || uint32_t(id) >= cfg.vocab_size) return "token id " + std::to_string(id) + " out of range";
        mimo26_bf16_to_f32(emb + size_t(id) * H * 2, H, h_x_.data() + size_t(t) * H);
        h_pos_[t] = int32_t(pos0 + t);
    }
    try {
        for (auto& c : cards_) if (auto e = run_card(*c, T, pos0, logits, all_rows); !e.empty()) return e;
    } catch (const sycl::exception& e) {
        return std::string("sycl: ") + e.what();
    }
    n_pos_ = pos0 + T;
    hi_end_ = std::max(hi_end_, n_pos_);
    feat_rows_ = feat_layers_.empty() ? 0 : std::min(T, feat_max_);
    ++n_calls_;
    profile_rows_ = nullptr;   // the mask covered this call only
    return {};
}

}  // namespace ie
