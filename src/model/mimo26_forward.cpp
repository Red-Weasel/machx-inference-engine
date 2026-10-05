// src/model/mimo26_forward.cpp — MiMo-V2.6 forward (P2 bring-up). See include/ie/mimo26_forward.hpp.
#include "ie/mimo26_forward.hpp"

#include "ie/kernel_profiler.hpp"
#include "ie/mimo26_host_rules.hpp"
#include "ie/mimo26_lanes.hpp"
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

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

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
    for (const auto& c : cards_) { if (c->tier_on) b += c->tier.pinned_bytes(); b += c->h_stage_bytes; }   // (+ B13's staging)
    return b;
}

void Mimo26Forward::free_all() {
    pipe_stop();   // (a no-op without stage threads; a paused pipe is stopped too)
    // every card's queue drained BEFORE anything is freed: a stage exception leaves the other card's work in flight, and freeing
    // one card's memory while another card's queue still runs wedged a card before (B2 gate note 6)
    for (auto& c : cards_) if (c->q) c->q->wait();
    for (auto& c : cards_) {
        if (!c->q) continue;
        if (c->tier_on) { c->tier.free_storage(*c->q); c->tier_on = false; }
        for (void* p : c->owned) sycl::free(p, *c->q);
        c->owned.clear(); c->bytes = 0;
    }
    cards_.clear();
    lanes_.clear(); lane_bytes_.clear(); lane_ = 0;
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
    const size_t ci = size_t(std::find_if(cards_.begin(), cards_.end(), [&](const auto& p) { return p.get() == &c; }) - cards_.begin());
    for (uint32_t L = c.L0; L < c.L1; ++L) lanes_[0].kv[ci].push_back({c.dense[L - c.L0].k, c.dense[L - c.L0].v});
    // P4 B1: the extra lanes' caches (lane 0's layout at lane_ctx positions), before the tier sizes -- refused with the numbers
    // unless they leave the reserve, the tier's batch workspace and its smallest static tier free
    if (lanes_.size() > 1) {
        const uint32_t lctx = lanes_[1].cap;
        std::vector<Mimo26LaneLayer> ll;
        for (uint32_t L = c.L0; L < c.L1; ++L) ll.push_back({cfg.is_swa(L), cfg.n_kv(L)});
        const uint64_t per = mimo26_lane_kv_bytes(ll, cfg.head_dim, cfg.v_head_dim, ring_, lctx);
        const auto& dv = q.get_device();
        if (!dv.has(sycl::aspect::ext_intel_free_memory)) return "lanes: the device does not report free memory";
        const uint64_t free_now = dv.get_info<sycl::ext::intel::info::device::free_memory>();
        double reserve_gib = 1.5;
        if (const char* v = std::getenv("IE_MIMO26_VRAM_RESERVE_GIB"); v && *v) reserve_gib = std::max(0.0, std::atof(v));
        Ds4SlotLayout lay;
        if (auto e = ds41_slot_layout(H, cfg.moe_inter_dim, lay); !e.empty()) return e;
        const uint64_t tier_min = first < c.L1 ? uint64_t(c.L1 - first) * lay.bytes * (opt_.n_static ? opt_.n_static + opt_.stream_slots : opt_.stream_slots + 1) +
                                                 ds4_expert_batch_ws_bytes(T, cfg.n_activated_experts, H, cfg.moe_inter_dim) : 0;
        const uint64_t reserve = uint64_t(reserve_gib * 1073741824.0) + tier_min + (ci == 0 ? opt_.reserve_card0 : 0);
        if (auto e = mimo26_lanes_fit(free_now, per, uint32_t(lanes_.size() - 1), reserve, uint32_t(ci), lctx); !e.empty()) {
            // P4 B61: the free-memory reserve is advice (a warning, the load goes on); what stays a limit is the experts'
            // minimum tier, without which a step cannot run
            const uint64_t hard = tier_min + (ci == 0 ? opt_.reserve_card0 : 0);
            if (auto e2 = mimo26_lanes_fit(free_now, per, uint32_t(lanes_.size() - 1), hard, uint32_t(ci), lctx); !e2.empty()) return e2;
            std::fprintf(stderr, "[mimo26] WARNING: %s -- the load goes on as asked\n", e.c_str());
        }
        const uint64_t b0 = c.bytes;
        for (size_t l = 1; l < lanes_.size(); ++l)
            for (uint32_t L = c.L0; L < c.L1; ++L) {
                const bool swa = cfg.is_swa(L);
                const size_t slots = swa && ring_ ? ring_ : lctx, n_kv = cfg.n_kv(L);
                const size_t k_n = (n_kv * slots + 64) * cfg.head_dim, v_n = n_kv * slots * (swa ? cfg.v_head_dim : cfg.head_dim);
                sycl::half* k = dev<sycl::half>(c, k_n);
                sycl::half* v = dev<sycl::half>(c, v_n);
                if (!k || !v) return "lanes: lane " + std::to_string(l) + " layer " + std::to_string(L) + ": device allocation failed";
                q.memset(k, 0, k_n * 2); q.memset(v, 0, v_n * 2);
                lanes_[l].kv[ci].push_back({k, v});
            }
        q.wait();
        lane_bytes_[ci] = (c.bytes - b0) / (lanes_.size() - 1);
        if (lane_bytes_[ci] != per) return "lanes: internal: allocated " + std::to_string(lane_bytes_[ci]) + " bytes per lane, the arithmetic says " + std::to_string(per);
    }
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
        s.f16_rescale = true;   // ... and so can the SwiGLU products the XMX route stores fp16 (#74, Ds41ExpertSource::f16_rescale)
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
            const uint64_t reserve = uint64_t(reserve_gib * 1073741824.0) + ds4_expert_batch_ws_bytes(T, cfg.n_activated_experts, H, cfg.moe_inter_dim)
                                   + (&c == cards_.front().get() ? opt_.reserve_card0 : 0);   // the vision tower's encode block (P6.2)
            Ds4SlotLayout lay;
            if (auto e = ds41_slot_layout(H, cfg.moe_inter_dim, lay); !e.empty()) return e;
            const uint64_t slots = free_now > reserve ? (free_now - reserve) / (uint64_t(n_moe) * lay.bytes) : 0;
            if (slots <= opt_.stream_slots) return "auto static tier: " + std::to_string(free_now >> 20) + " MiB free leaves no static slots";
            n_static = uint32_t(std::min<uint64_t>(E, slots - opt_.stream_slots));
            if (lanes_.size() > 1)   // P4 B14: what the lanes cost this card's static tier, never silent
                std::fprintf(stderr, "[mimo26 forward] card %zu: the %zu extra lane(s) take %.0f MiB = %llu static slots/layer (now %u)\n",
                             ci, lanes_.size() - 1, double(lane_bytes_[ci] * (lanes_.size() - 1)) / 1048576.0,
                             (unsigned long long)(lane_bytes_[ci] * (lanes_.size() - 1) / (uint64_t(n_moe) * lay.bytes)), n_static);
        }
        // P4 B2: pipelined lanes run both cards' CPU legs at once -- each on its own half of the E-cores (V4.1's split)
        if (opt_.split_cpu_cores && cards_.size() == 2 && !std::getenv("IE_DS41_CPU_CORES")) c.tier.set_cpu_cores(ci == 0 ? "8-13" : "14-19");
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
        c.tier_on = true; c.tier_L0 = first; c.qstar_multi0 = c.tier.qstar_multi();
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

void Mimo26Forward::state_spans(size_t ci, uint32_t n, uint32_t hi, std::vector<std::pair<void*, uint64_t>>& out, bool rings_only) const {
    out.clear();
    if (ci >= cards_.size() || !m_ || n == 0) return;
    const auto& cfg = m_->config();
    const Card& c = *cards_[ci];
    for (uint32_t L = c.L0; L < c.L1; ++L) {
        const auto [dk, dv] = lanes_[lane_].kv[ci][L - c.L0];   // the active lane's caches
        // the caches' layout (upload_card): k [n_kv, slots, head_dim], v [n_kv, slots, v_row]; a linear cache holds position
        // p at slot p, an SWA ring at p % ring_ (and holds only [written_end - ring_, n_pos) of them)
        const bool in_ring = cfg.is_swa(L) && ring_;
        if (rings_only && !in_ring) continue;
        const uint64_t n_kv = cfg.n_kv(L), slots = in_ring ? ring_ : cap_;
        const uint64_t k_row = cfg.head_dim, v_row = cfg.is_swa(L) ? cfg.v_head_dim : cfg.head_dim;
        uint32_t runs[4] = {0, n, 0, 0}, n_runs = 1;
        if (in_ring) n_runs = mimo26_ring_runs(std::min(hi > ring_ ? hi - ring_ : 0u, n), n, ring_, runs);
        for (const auto& [base, row] : {std::pair<sycl::half*, uint64_t>{dk, k_row}, std::pair<sycl::half*, uint64_t>{dv, v_row}})
            for (uint64_t h = 0; h < n_kv; ++h)
                for (uint32_t r = 0; r < n_runs; ++r)
                    out.push_back({base + (h * slots + runs[2 * r]) * row, uint64_t(runs[2 * r + 1] - runs[2 * r]) * row * sizeof(sycl::half)});
    }
}

std::string Mimo26Forward::set_state(uint32_t n, uint32_t hi) {
    if (cards_.empty()) return "set_state: not initialised";
    if (n > hi || hi > cap_)
        return "set_state: " + std::to_string(n) + " positions written to " + std::to_string(hi) + " do not fit the capacity " + std::to_string(cap_);
    n_pos_ = n; hi_end_ = hi; feat_rows_ = 0;
    return {};
}

void Mimo26Forward::probe_op(sycl::queue& q, const char* op, const void* dev, bool f16, uint32_t rows, uint32_t cols) {
    q.wait();
    const size_t n = size_t(rows) * cols;
    const bool watch = probe_row_ >= 0 && uint32_t(probe_row_) < rows;
    ProbeOp p; p.op = op; p.rows = rows; p.cols = cols;
    if (f16) {
        std::vector<uint16_t> h(n);
        q.memcpy(h.data(), dev, n * 2).wait();
        p.all = mimo26_scan_f16(h.data(), n);
        if (watch) p.row = mimo26_scan_f16(h.data() + size_t(probe_row_) * cols, cols);
    } else {
        std::vector<float> f(n);
        q.memcpy(f.data(), dev, n * 4).wait();
        p.all = mimo26_scan_f32(f.data(), n);
        if (watch) p.row = mimo26_scan_f32(f.data() + size_t(probe_row_) * cols, cols);
        if (watch && p.op == "moe")                            // the row itself, for an fp64 comparison (tools/mimo26/moe_agree.py)
            probe_moe_out_.assign(f.begin() + std::ptrdiff_t(size_t(probe_row_) * cols), f.begin() + std::ptrdiff_t(size_t(probe_row_ + 1) * cols));
    }
    probe_ops_.push_back(std::move(p));
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
    if (opt_.lanes == 0) return "lanes: at least one";
    lanes_.assign(opt_.lanes, LaneState{});
    for (size_t l = 0; l < lanes_.size(); ++l) {
        lanes_[l].cap = l == 0 || opt_.lane_ctx == 0 ? opt_.max_ctx : opt_.lane_ctx;
        lanes_[l].kv.assign(nc, {});
    }
    lane_bytes_.assign(nc, 0);
    lane_ = 0; cap_ = opt_.max_ctx;
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
    for (auto& c : cards_) {
        c->h_stage = sycl::malloc_host<float>(size_t(opt_.max_tokens) * std::max(cfg.dim, cfg.n_routed_experts), *c->q);
        if (!c->h_stage) return "card " + std::to_string(c->L0) + ": pinned host staging alloc failed";
        c->owned.push_back(c->h_stage);   // (host memory: not counted in the card's VRAM bytes)
        c->h_stage_bytes = uint64_t(opt_.max_tokens) * std::max(cfg.dim, cfg.n_routed_experts) * 4;
        c->h_ridx.resize(size_t(opt_.max_tokens) * cfg.n_activated_experts);
        c->h_rw.resize(c->h_ridx.size());
    }
    h_pos_.resize(opt_.max_tokens);
    lane_tier_.assign(lanes_.size(), std::vector<LaneTierStats>(nc));
    group_tier_.assign(nc, {});
    n_pos_ = 0; hi_end_ = 0;
    if (lanes_.size() > 1) {
        std::fprintf(stderr, "[mimo26] lanes: %zu (lane 0 at ctx %u, lanes 1-%zu at ctx %u), one extra lane's caches", lanes_.size(), opt_.max_ctx,
                     lanes_.size() - 1, lanes_[1].cap);
        for (size_t c = 0; c < lane_bytes_.size(); ++c) std::fprintf(stderr, " %.2f GB", double(lane_bytes_[c]) / 1e9);
        std::fprintf(stderr, " (per card)\n");
    }
    return {};
}

std::string Mimo26Forward::select_lane(uint32_t l) {
    if (l >= lanes_.size()) return "select_lane: lane " + std::to_string(l) + " of " + std::to_string(lanes_.size());
    if (piping()) return "select_lane: the lane pipe is running";
    if (l == lane_) return {};
    lanes_[lane_].n_pos = n_pos_; lanes_[lane_].hi_end = hi_end_;
    const LaneState& nx = lanes_[l];
    n_pos_ = nx.n_pos; hi_end_ = nx.hi_end; cap_ = nx.cap; feat_rows_ = 0;
    lane_ = l;
    return {};
}

std::string Mimo26Forward::run_card(Card& c, size_t ci, const Call& k) {
    const auto& m = *m_; const auto& cfg = m.config();
    // the call's rows: T over its segments, each a lane's rows (its caches and capacity, P4 B1/B2/B3); pos0 labels the
    // diagnostics' rows (the serial forward is one segment)
    const uint32_t T = k.T, pos0 = k.segs.front().pos0;
    std::vector<int32_t>& h_ridx_ = c.h_ridx; std::vector<float>& h_rw_ = c.h_rw;
    // IE_MIMO26_ROWS_INVARIANT=1 (B3): the o-proj and the LM head per segment (the serial launches; oneDNN's kernel follows M)
    static const bool rows_invariant = [] { const char* v = std::getenv("IE_MIMO26_ROWS_INVARIANT"); return v && *v == '1'; }();
    // B3: a step of several lanes' rows splits its pinned misses at IE_MIMO26_QSTAR_GROUP (default 0.35: swept 0.20 / 0.30 / 0.35 /
    // 0.50 on 4 lanes as two 2-row groups, docs/mimo26/P4_B3_ROWS.md) -- the rows share almost no experts, so the tier's multi-row
    // share (0.20, tuned on one sequence's verify rows) sends too much to the CPU leg; a one-lane step keeps the tier's own share
    static const float qstar_group = [] { const char* v = std::getenv("IE_MIMO26_QSTAR_GROUP"); return v && *v ? float(std::atof(v)) : 0.35f; }();
    if (c.tier_on) c.tier.set_qstar_multi(k.segs.size() > 1 ? qstar_group : c.qstar_multi0);
    const uint32_t H = cfg.dim, E = cfg.n_routed_experts, TK = cfg.n_activated_experts, FI = cfg.inter_dim;
    const uint32_t n_q = cfg.n_heads, hd = cfg.head_dim, hdv = cfg.v_head_dim, RD = cfg.rope_dim();
    sycl::queue& q = *c.q;
    const float inf = std::numeric_limits<float>::infinity();   // the tier's SwiGLU clamp: none
    auto prof = [&](const char* nm, sycl::event e) { if (g_profiler) [[unlikely]] g_profiler->push(nm, e, q); };

    q.memcpy(c.x, k.x, size_t(T) * H * 4);
    q.memcpy(c.pos, k.pos, size_t(T) * 4);
    struct FeatOut { float* dst; size_t off, n; };
    std::vector<FeatOut> feat_out;
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
        // (diagnostic) the NaN probe (set_probe): this layer's ops, each scanned by bits once it has run
        const bool probe = int(L) == probe_layer_;
        auto pr = [&](const char* op, const void* p, bool f16, uint32_t cols) { if (probe) [[unlikely]] probe_op(q, op, p, f16, T, cols); };
        pr("in", c.x, false, H);
        mimo26_rms_norm(q, c.x, d.attn_norm, c.xn16, nullptr, T, H, cfg.norm_eps);
        pr("attn_norm", c.xn16, true, H);
        dense("mimo26.gemm.qkv", c.xn16, d.qkv8, d.qkv_s, d.qkv, c.qkv32, Nqkv, H);
        pr("qkv", c.qkv32, false, Nqkv);
        // the split and the RoPE are per row (positions per row); attention runs PER SEGMENT (B3) -- each lane's rows against
        // its own caches (positions, capacity), the group's other rows untouched; one segment at r0 = 0 is the pre-group launch
        // SWA: a step through the split-K decode attention, a chunk through mimo26_attention_prefill_xmx (ring or linear);
        // IE_MIMO26_SPLIT_DECODE=0 / IE_MIMO26_SWA_PREFILL=naive: mimo26_attention
        static const bool swa_xmx = [] { const char* v = std::getenv("IE_MIMO26_SWA_PREFILL"); return !(v && std::string(v) == "naive"); }();
        // full layer: V padded to hd rows (the cache's layout); a chunk (T > 8) through mimo26_attention_prefill_xmx
        // (IE_MIMO26_FULL_PREFILL=fa2: the generic XMX FA-2 on the padded V; =naive: mimo26_attention), a step through
        // the split-K decode attention (IE_MIMO26_SPLIT_DECODE=0: mimo26_attention, the serial-key bring-up kernel)
        static const int full_prefill = [] { const char* v = std::getenv("IE_MIMO26_FULL_PREFILL");
                                             const std::string m = v ? v : ""; return m == "naive" ? 0 : m == "fa2" ? 1 : 2; }();
        sycl::half* const Vw = swa ? c.V : c.Vp;   // the V workspace: plain rows, or padded to hd
        const uint32_t vst = swa ? hdv : hd;
        mimo26_split_qkv(q, c.qkv32, c.Q, c.K, Vw, T, n_q, n_kv, hd, hdv, {}, swa ? 0u : hd);
        rope_partial(q, c.Q, c.pos, c.Q, T, n_q, hd, RD, theta);
        rope_partial(q, c.K, c.pos, c.K, T, n_kv, hd, RD, theta);
        pr("q", c.Q, true, n_q * hd); pr("k", c.K, true, n_kv * hd); pr("v", Vw, true, n_kv * vst);
        for (const Seg& s : k.segs) {
            sycl::half* const kc = lanes_[s.lane].kv[ci][L - c.L0].first; sycl::half* const vc = lanes_[s.lane].kv[ci][L - c.L0].second;
            const uint32_t cap = lanes_[s.lane].cap, Ts = s.T, p0 = s.pos0;
            const sycl::half* Qs = c.Q + size_t(s.r0) * n_q * hd; const sycl::half* Ks = c.K + size_t(s.r0) * n_kv * hd;
            const sycl::half* Vs = Vw + size_t(s.r0) * n_kv * vst; sycl::half* ys = c.attn + size_t(s.r0) * n_q * hdv;
            if (swa) {
                if (Ts <= kDecodeRows && split_decode)
                    mimo26_attention_decode(q, Qs, Ks, Vs, kc, vc, ys, c.partials, Ts, p0, n_q, n_kv, hd, hdv, cap, window, d.sink, {}, ring_);
                else if (Ts > kDecodeRows && swa_xmx)
                    mimo26_attention_prefill_xmx(q, Qs, Ks, Vs, kc, vc, ys, Ts, p0, n_q, n_kv, cap, window, d.sink, {}, 0, ring_);
                else
                    mimo26_attention(q, Qs, Ks, Vs, kc, vc, ys, Ts, p0, n_q, n_kv, hd, hdv, cap, window, d.sink, {}, ring_);
            } else if (Ts > kDecodeRows && full_prefill == 2) {
                mimo26_attention_prefill_xmx(q, Qs, Ks, Vs, kc, vc, ys, Ts, p0, n_q, n_kv, cap, 0, nullptr, {}, hd);
            } else if (Ts > kDecodeRows && full_prefill == 1) {
                sycl::half* const yp = c.attn_p + size_t(s.r0) * n_q * hd;
                full_attention_fa2_prefill_xmx(q, Qs, Ks, Vs, kc, vc, yp, Ts, p0, n_q, n_kv, hd, cap);
                mimo26_compact_heads(q, yp, ys, Ts, n_q, hd, hdv);
            } else if (Ts <= kDecodeRows && split_decode) {
                mimo26_attention_decode(q, Qs, Ks, Vs, kc, vc, ys, c.partials, Ts, p0, n_q, n_kv, hd, hdv, cap, 0, nullptr, {}, 0, hd);
            } else {
                mimo26_attention(q, Qs, Ks, Vs, kc, vc, ys, Ts, p0, n_q, n_kv, hd, hdv, cap, 0, nullptr, {}, 0, hd);
            }
        }
        pr("attn", c.attn, true, n_q * hdv);
        // the o-proj through oneDNN, whose kernel follows M: rows_invariant runs it per segment (the serial launch, M = that
        // segment's rows); a one-segment call is the same launch either way
        if (rows_invariant && k.segs.size() > 1)
            for (const Seg& s : k.segs) prof("mimo26.gemm.o", gemm_nt_f16_onednn(q, c.attn + size_t(s.r0) * n_q * hdv, d.o, c.o32 + size_t(s.r0) * H, s.T, H, n_q * hdv));
        else
            prof("mimo26.gemm.o", gemm_nt_f16_onednn(q, c.attn, d.o, c.o32, T, H, n_q * hdv));
        pr("o_proj", c.o32, false, H);
        mimo26_axpy(q, c.o32, cfg.value_scale, c.x, size_t(T) * H);
        pr("x+attn", c.x, false, H);
        mimo26_rms_norm(q, c.x, d.ffn_norm, c.xn16, c.xn32, T, H, cfg.norm_eps);
        pr("ffn_norm", c.xn32, false, H);
        if (!Lw.moe) {
            dense("mimo26.gemm.mlp_gate", c.xn16, d.gate8, d.gate_s, d.gate, c.gate32, FI, H);
            pr("gate", c.gate32, false, FI);
            dense("mimo26.gemm.mlp_up", c.xn16, d.up8, d.up_s, d.up, c.up32, FI, H);
            pr("up", c.up32, false, FI);
            mimo26_swiglu_f32(q, c.gate32, c.up32, c.h16, size_t(T) * FI);
            pr("swiglu", c.h16, true, FI);
            dense("mimo26.gemm.mlp_down", c.h16, d.down8, d.down_s, d.down, c.o32, H, FI);
            pr("down", c.o32, false, H);
            mimo26_axpy(q, c.o32, 1.f, c.x, size_t(T) * H);
        } else {
            // noaux_tc: sigmoid scores, select the top-k of (score + bias), weight by the unbiased scores normalised to 1
            // a chunk's router logits through oneDNN's fp32 GEMM (the per-(row, expert) dot kernel took 9 ms at 2048 rows)
            if (T > kDecodeRows) prof("mimo26.gemm.router", gemm_nt_f32_onednn(q, c.xn32, d.router_w, c.rlogits, T, E, H));
            else mimo26_router_logits(q, c.xn32, d.router_w, c.rlogits, T, H, E);
            {   // B13: a kernel into the pinned staging, not q.memcpy (T x 1 KiB is the driver's slow band; see Card::h_stage)
                const uint32_t* src = reinterpret_cast<const uint32_t*>(c.rlogits); uint32_t* dst = reinterpret_cast<uint32_t*>(c.h_stage);
                q.parallel_for<class Mimo26RouterReadbackK>(sycl::range<1>(size_t(T) * E), [=](sycl::id<1> i) { dst[i] = src[i]; }).wait();
            }
            std::vector<uint32_t> order(E);
            for (uint32_t t = 0; t < T; ++t) {
                const float* lg = c.h_stage + size_t(t) * E;
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
            pr("router", c.rlogits, false, E);
            if (probe && probe_row_ >= 0 && uint32_t(probe_row_) < T) {   // (diagnostic) the watched row's MoE input and routing
                probe_moe_.assign(size_t(H) * 4 + size_t(TK) * 8, 0);
                q.memcpy(probe_moe_.data(), c.xn32 + size_t(probe_row_) * H, size_t(H) * 4).wait();
                std::memcpy(probe_moe_.data() + size_t(H) * 4, h_ridx_.data() + size_t(probe_row_) * TK, size_t(TK) * 4);
                std::memcpy(probe_moe_.data() + size_t(H) * 4 + size_t(TK) * 4, h_rw_.data() + size_t(probe_row_) * TK, size_t(TK) * 4);
            }
            const auto tm = std::chrono::steady_clock::now();
            if (auto e = c.tier.moe(q, L, c.xn32, h_ridx_.data(), h_rw_.data(), T, c.moe, inf); !e.empty()) return "layer " + std::to_string(L) + " tier: " + e;
            stats_[L].moe_ms = ms_since(tm);
            const auto& ts = c.tier.last();
            stats_[L].experts_static = ts.experts_static; stats_[L].experts_pinned = ts.experts_pinned; stats_[L].experts_mmap = ts.experts_mmap;
            if (T <= kDecodeRows) {   // a lane's step to its counters; a group's (several lanes' rows) to the card's group counters
                LaneTierStats& ls = k.segs.size() == 1 ? lane_tier_[k.segs[0].lane][ci] : group_tier_[ci];
                ls.experts_static += ts.experts_static; ls.experts_pinned += ts.experts_pinned; ls.stream_hits += ts.stream_hits;
                ls.experts_cpu += ts.experts_cpu; ls.moe_ms += stats_[L].moe_ms;
            }
            pr("moe", c.moe, false, H);
            mimo26_axpy(q, c.moe, 1.f, c.x, size_t(T) * H);
        }
        if (!feat_layers_.empty()) {   // P5: this layer's residual rows for the drafter, each segment's last rows into its lane's buffer
            const auto it = std::find(feat_layers_.begin(), feat_layers_.end(), L);
            if (it != feat_layers_.end()) {
                const uint32_t fi = uint32_t(it - feat_layers_.begin());
                size_t o = 0;   // B13: through the pinned staging (see Card::h_stage), copied out after the layer's wait
                for (const Seg& s : k.segs) {
                    const uint32_t rows = std::min(s.T, feat_max_);
                    q.memcpy(c.h_stage + o, c.x + size_t(s.r0 + s.T - rows) * H, size_t(rows) * H * 4);
                    feat_out.push_back({s.feat + (size_t(fi) * rows) * H, o, size_t(rows) * H});   // compact [n][rows][dim]
                    o += size_t(rows) * H;
                }
            }
        }
        q.wait();
        for (const auto& f : feat_out) std::memcpy(f.dst, c.h_stage + f.off, f.n * 4);
        feat_out.clear();
        pr("out", c.x, false, H);
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
    if (T <= kDecodeRows) { LaneTierStats& ls = k.segs.size() == 1 ? lane_tier_[k.segs[0].lane][ci] : group_tier_[ci]; ++ls.steps; ls.rows += T; }
    if (&c == cards_.back().get()) {
        mimo26_rms_norm(q, c.x, c.fnorm, c.xn16, nullptr, T, H, cfg.norm_eps);
        // the LM head over the rows the segments want (a segment's last row, or all of them): one segment = the pre-group
        // launches; a group (B3) = ONE GEMM over the span from the first wanted row to the last (a group has at most
        // kDecodeRows rows, the head weight is read once for any M <= kHeadRows) and each lane's rows copied out of it;
        // rows_invariant: per segment, M = that segment's rows (the serial launch)
        auto want = [](const Seg& s) { return std::pair<uint32_t, uint32_t>{s.r0 + (s.all_rows ? 0u : s.T - 1), s.all_rows ? s.T : 1u}; };   // (first row, rows)
        auto head_rows = [&](uint32_t r0, uint32_t rows, float* dst) {   // dst [rows, vocab] on the host
            for (uint32_t r = 0; r < rows; r += kHeadRows) {
                const uint32_t n = std::min(kHeadRows, rows - r);
                prof("mimo26.gemm.head", gemm_nt_f16_onednn(q, c.xn16 + size_t(r0 + r) * H, c.head, c.head_out, n, cfg.vocab_size, H));
                q.memcpy(dst + size_t(r) * cfg.vocab_size, c.head_out, size_t(n) * cfg.vocab_size * 4).wait();
            }
        };
        for (const Seg& s : k.segs) s.logits->resize(size_t(want(s).second) * cfg.vocab_size);
        uint32_t lo = T, hi = 0;
        for (const Seg& s : k.segs) { const auto [r0, rows] = want(s); lo = std::min(lo, r0); hi = std::max(hi, r0 + rows); }
        if (k.segs.size() == 1 || rows_invariant || hi - lo > kHeadRows) {
            for (const Seg& s : k.segs) { const auto [r0, rows] = want(s); head_rows(r0, rows, s.logits->data()); }
        } else {
            prof("mimo26.gemm.head", gemm_nt_f16_onednn(q, c.xn16 + size_t(lo) * H, c.head, c.head_out, hi - lo, cfg.vocab_size, H));
            for (const Seg& s : k.segs) { const auto [r0, rows] = want(s); q.memcpy(s.logits->data(), c.head_out + size_t(r0 - lo) * cfg.vocab_size, size_t(rows) * cfg.vocab_size * 4); }
            q.wait();
        }
    } else {
        q.memcpy(c.h_stage, c.x, size_t(T) * H * 4).wait();   // the residual stream crosses to the next card through the host
        std::memcpy(k.x, c.h_stage, size_t(T) * H * 4);         // (B13: via the pinned staging, see Card::h_stage)
    }
    return {};
}

std::string Mimo26Forward::forward(const int32_t* ids, uint32_t T, uint32_t pos0, std::vector<float>& logits, bool all_rows) {
    if (cards_.empty()) return "not initialised";
    if (piping()) return "forward: the lane pipe is running (pipe_submit)";
    const auto& cfg = m_->config();
    if (T == 0 || T > opt_.max_tokens) return "T out of range";
    if (pos0 != n_pos_) return "pos0 " + std::to_string(pos0) + " != n_pos " + std::to_string(n_pos_) + " (call reset() to start over)";
    if (pos0 + T > cap_) return "context capacity " + std::to_string(cap_) + " exceeded";
    if (cfg.n_routed_experts > 512) return "more than 512 routed experts";
    const uint32_t H = cfg.dim;
    // embeddings: a host gather from the mmap (BF16 rows)
    const uint8_t* emb = m_->embed.w->data;
    for (uint32_t t = 0; t < T; ++t) {
        const int32_t id = ids[t];
        if (id < 0) {                                        // an image position (P6.2): its row comes from the provider
            const VisionProvider& vis_provider_ = lanes_[lane_].vis;
            if (!vis_provider_) return "token id " + std::to_string(id) + " is an image position but no vision provider is set";
            const float* row = nullptr;
            try {                                            // the provider encodes on the device: a SYCL fault is an error string here
                if (auto e = vis_provider_(pos0 + t, row); !e.empty()) return e.starts_with("vision:") ? e : "vision: " + e;
            } catch (const std::exception& e) {
                return std::string("vision: ") + e.what();
            }
            if (!row) return "vision: the provider returned no row for position " + std::to_string(pos0 + t);
            std::memcpy(h_x_.data() + size_t(t) * H, row, size_t(H) * 4);
        } else {
            if (uint32_t(id) >= cfg.vocab_size) return "token id " + std::to_string(id) + " out of range";
            mimo26_bf16_to_f32(emb + size_t(id) * H * 2, H, h_x_.data() + size_t(t) * H);
        }
        h_pos_[t] = int32_t(pos0 + t);
    }
    try {
        Call k; k.T = T; k.x = h_x_.data(); k.pos = h_pos_.data();   // one segment: the active lane
        k.segs.push_back({lane_, T, pos0, 0, all_rows, h_feat_.data(), &logits});
        for (size_t ci = 0; ci < cards_.size(); ++ci) if (auto e = run_card(*cards_[ci], ci, k); !e.empty()) return e;
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

// ---- P4 B2: card-pipelined lanes -----------------------------------------------------------------------------------------

std::string Mimo26Forward::pipe_start(PipeDone done, uint32_t group_lanes, uint32_t group_rows) {
    if (cards_.empty()) return "pipe_start: not initialised";
    if (group_rows > kDecodeRows) return "pipe_start: group_rows " + std::to_string(group_rows) + " > " + std::to_string(kDecodeRows);
    if (!stage_th_.empty()) return ppaused_ ? "pipe_start: the pipe is paused (pipe_resume)" : "pipe_start: already running";
    static const char* dump_dir = std::getenv("IE_MIMO26_DUMP");
    if (probe_layer_ >= 0 || profiling_ || (dump_dir && *dump_dir)) return "pipe_start: the diagnostics (probe, routing profile, IE_MIMO26_DUMP) are serial-only";
    const auto& cfg = m_->config();
    lanes_[lane_].n_pos = n_pos_; lanes_[lane_].hi_end = hi_end_;   // the active lane's positions live in lanes_ while the pipe runs
    for (auto& ln : lanes_) {
        if (ln.h_x.empty()) {
            ln.h_x.resize(size_t(opt_.max_tokens) * cfg.dim);
            ln.h_pos.resize(opt_.max_tokens);
        }
        ln.h_feat.resize(h_feat_.size());
        ln.in_flight = false; ln.in_cb = false; ln.resub = false;
    }
    // B3: the group pool -- a lane is in at most one group, so lanes + 1 groups always leave one for stage 0 to form; a
    // multi-lane group holds at most kDecodeRows rows (a one-lane group reads the lane's own buffers)
    groups_.clear(); gfree_.clear(); lq_.clear();
    for (size_t g = 0; g < lanes_.size() + 1; ++g) {
        groups_.push_back(std::make_unique<Group>());
        groups_.back()->h_x.resize(size_t(kDecodeRows) * cfg.dim); groups_.back()->h_pos.resize(kDecodeRows);
        gfree_.push_back(groups_.back().get());
    }
    pq_.assign(cards_.size(), {});
    pbusy_ = 0; pstop_ = false; pgate_ = false; perr_.clear(); pdone_ = std::move(done); group_lanes_ = group_lanes;
    group_rows_ = group_rows ? group_rows : kDecodeRows; cb_lanes_.clear();
    // with the split CPU legs, the stage threads keep off both legs' cores (the E-cores; IE_DS41_CPU_CORES when set)
    std::vector<int> off;
    if (opt_.split_cpu_cores) {
        const char* ev = std::getenv("IE_DS41_CPU_CORES");
        const std::string cores = ev && *ev ? ev : "8-19";
        for (size_t i = 0; i < cores.size();) {
            size_t j = cores.find(',', i); if (j == std::string::npos) j = cores.size();
            const std::string tok = cores.substr(i, j - i); const size_t d = tok.find('-');
            if (d == std::string::npos) off.push_back(std::atoi(tok.c_str()));
            else for (int c = std::atoi(tok.substr(0, d).c_str()); c <= std::atoi(tok.substr(d + 1).c_str()); ++c) off.push_back(c);
            i = j + 1;
        }
    }
    for (size_t s = 0; s < cards_.size(); ++s)
        stage_th_.emplace_back([this, s, off] {
            if (!off.empty()) {
                cpu_set_t set; CPU_ZERO(&set);
                for (int c = 0; c < int(sysconf(_SC_NPROCESSORS_ONLN)); ++c) if (std::find(off.begin(), off.end(), c) == off.end()) CPU_SET(c, &set);
                if (CPU_COUNT(&set)) pthread_setaffinity_np(pthread_self(), sizeof set, &set);
            }
            stage_loop(s);
        });
    return {};
}

std::string Mimo26Forward::pipe_submit(uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0, bool all_rows) {
    if (!piping()) return "pipe_submit: the pipe is not running";
    if (lane >= lanes_.size()) return "pipe_submit: lane " + std::to_string(lane) + " of " + std::to_string(lanes_.size());
    LaneState& ln = lanes_[lane];
    // claim the lane (check-and-set under the lock): an idle lane from any thread; a lane inside its done callback only from that
    // callback's thread, once (B2 gate note 7)
    bool fresh = false;
    {
        std::lock_guard<std::mutex> lk(pmu_);
        if (!perr_.empty()) return "pipe_submit: a stage failed: " + perr_;
        // (piping() above is read unlocked) a submit racing pipe_stop or pipe_pause: a step queued now would find a stage thread
        // already gone and strand the lane half-stepped, or start while the serial API owns the cards
        if (pstop_) return "pipe_submit: the pipe is stopping";
        if (ppaused_) return "pipe_submit: the pipe is paused";
        if (ln.in_flight) {
            if (!ln.in_cb || ln.cb_tid != std::this_thread::get_id()) return "pipe_submit: lane " + std::to_string(lane) + " has a step in flight";
            if (ln.resub) return "pipe_submit: lane " + std::to_string(lane) + " was already resubmitted from its callback";
        } else { ln.in_flight = true; ++pbusy_; fresh = true; }
    }
    auto unclaim = [&](std::string e) {   // a refused step gives a fresh claim back (a callback's lane stays in flight until it returns)
        if (fresh) { std::lock_guard<std::mutex> lk(pmu_); ln.in_flight = false; --pbusy_; }
        return e;
    };
    const auto& cfg = m_->config();
    if (T == 0 || T > opt_.max_tokens) return unclaim("pipe_submit: T out of range");
    if (pos0 != ln.n_pos) return unclaim("pipe_submit: lane " + std::to_string(lane) + ": pos0 " + std::to_string(pos0) + " != n_pos " + std::to_string(ln.n_pos));
    if (pos0 + T > ln.cap) return unclaim("pipe_submit: lane " + std::to_string(lane) + ": context capacity " + std::to_string(ln.cap) + " exceeded");
    const uint32_t H = cfg.dim;
    const uint8_t* emb = m_->embed.w->data;
    for (uint32_t t = 0; t < T; ++t) {   // the embeddings: forward()'s host gather, into the lane's own buffer
        const int32_t id = ids[t];
        if (id < 0) return unclaim("pipe_submit: image positions are serial-only (forward)");
        if (uint32_t(id) >= cfg.vocab_size) return unclaim("pipe_submit: token id " + std::to_string(id) + " out of range");
        mimo26_bf16_to_f32(emb + size_t(id) * H * 2, H, ln.h_x.data() + size_t(t) * H);
        ln.h_pos[t] = int32_t(pos0 + t);
    }
    ln.T = T; ln.pos0 = pos0; ln.all_rows = all_rows;
    {
        std::lock_guard<std::mutex> lk(pmu_);
        if (!fresh) ln.resub = true;   // the callback's resubmit: the lane's in-flight count carries over
        lq_.push_back(lane);            // stage 0 groups the waiting lanes in this order
    }
    pcv_.notify_all();
    return {};
}

std::string Mimo26Forward::pipe_reset_lane(uint32_t lane) {
    if (lane >= lanes_.size()) return "pipe_reset_lane: lane " + std::to_string(lane) + " of " + std::to_string(lanes_.size());
    std::lock_guard<std::mutex> lk(pmu_);
    LaneState& ln = lanes_[lane];   // an idle lane, or -- from its own done callback -- the lane whose sequence just ended
    if (ln.in_flight && !(ln.in_cb && ln.cb_tid == std::this_thread::get_id() && !ln.resub))
        return "pipe_reset_lane: lane " + std::to_string(lane) + " has a step in flight";
    ln.n_pos = 0; ln.hi_end = 0;
    return {};
}

std::string Mimo26Forward::pipe_rewind_lane(uint32_t lane, uint32_t n) {
    if (lane >= lanes_.size()) return "pipe_rewind_lane: lane " + std::to_string(lane) + " of " + std::to_string(lanes_.size());
    std::lock_guard<std::mutex> lk(pmu_);
    LaneState& ln = lanes_[lane];   // (the pipe_reset_lane rule: an idle lane, or the lane whose callback is running, from that thread)
    if (ln.in_flight && !(ln.in_cb && ln.cb_tid == std::this_thread::get_id() && !ln.resub))
        return "pipe_rewind_lane: lane " + std::to_string(lane) + " has a step in flight";
    if (n < ln.n_pos) ln.n_pos = n;
    return {};
}

std::string Mimo26Forward::pipe_error() const {
    std::lock_guard<std::mutex> lk(pmu_);
    return perr_;
}

const std::vector<uint32_t>& Mimo26Forward::pipe_cb_lanes() const {
    static const std::vector<uint32_t> none;
    std::lock_guard<std::mutex> lk(pmu_);   // (the tid is set and cleared under the lock; only its own thread gets the list)
    return cb_lanes_tid_ == std::this_thread::get_id() ? cb_lanes_ : none;
}

std::pair<double, double> Mimo26Forward::pipe_gate_ms() const {
    std::lock_guard<std::mutex> lk(pmu_);
    return {double(gate_ns_one_) * 1e-6, double(gate_ns_multi_) * 1e-6};
}

std::string Mimo26Forward::pipe_pause() {
    if (stage_th_.empty()) return "pipe_pause: the pipe is not running";
    if (ppaused_) return {};
    std::string e;
    {
        std::unique_lock<std::mutex> lk(pmu_);
        pcv_.wait(lk, [&] { return pbusy_ == 0; });   // (the caller's callbacks park their lanes instead of resubmitting)
        ppaused_ = true;
        e = perr_;
    }
    n_pos_ = lanes_[lane_].n_pos; hi_end_ = lanes_[lane_].hi_end; feat_rows_ = 0;   // the serial path's active lane, as after pipe_stop
    return e;
}

std::string Mimo26Forward::pipe_resume() {
    if (stage_th_.empty()) return "pipe_resume: the pipe is not running";
    if (!ppaused_) return {};
    lanes_[lane_].n_pos = n_pos_; lanes_[lane_].hi_end = hi_end_;   // the active lane's positions live in lanes_ while steps run (pipe_start)
    std::lock_guard<std::mutex> lk(pmu_);
    ppaused_ = false;
    return {};
}

void Mimo26Forward::stage_loop(size_t s) {
    const bool last = s + 1 == cards_.size();
    const uint32_t H = m_->config().dim, nc = uint32_t(cards_.size());
    for (;;) {
        Group* g = nullptr;
        {
            std::unique_lock<std::mutex> lk(pmu_);
            if (s == 0) {
                // B3: form a group from the lanes waiting, in submit order -- at most `cap` lanes and kDecodeRows rows (a step of
                // more rows, a prefill chunk, runs alone) -- once the last stage's callbacks are done (pgate_), so the lanes a
                // finishing group resubmits meet again. AUTO cap: the lanes in flight spread over the cards, ceil(busy / cards)
                while (!(pstop_ || (!lq_.empty() && !pgate_ && !gfree_.empty()))) {
                    // (diagnostic, pipe_gate_ms) a lane waits and a group is free, but a finished group's callbacks hold the gate
                    const bool gated = pgate_ && !lq_.empty() && !gfree_.empty();
                    const auto tw = gated ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
                    const bool multi = pgate_lanes_ > 1;
                    pcv_.wait(lk);
                    if (gated) (multi ? gate_ns_multi_ : gate_ns_one_) += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - tw).count());
                }
                if (lq_.empty()) return;                    // stopped (pipe_stop waits for no step in flight first)
                g = gfree_.front(); gfree_.pop_front();
                const uint32_t cap = group_lanes_ ? group_lanes_ : std::max<uint32_t>(1, (pbusy_ + nc - 1) / nc);
                g->segs.clear(); g->T = 0;
                while (!lq_.empty() && g->segs.size() < cap) {   // (B5: group_rows_ = the draft budget, <= kDecodeRows)
                    LaneState& ln = lanes_[lq_.front()];
                    if (!g->segs.empty() && g->T + ln.T > group_rows_) break;
                    g->segs.push_back({lq_.front(), ln.T, ln.pos0, g->T, ln.all_rows, ln.h_feat.data(), &ln.logits});
                    g->T += ln.T; lq_.pop_front();
                    if (g->T > group_rows_) break;          // a lone step of more rows
                }
            } else {
                pcv_.wait(lk, [&] { return pstop_ || !pq_[s].empty(); });
                if (pq_[s].empty()) return;
                g = pq_[s].front(); pq_[s].pop_front();
            }
        }
        if (s == 0) {   // the group's rows: a one-lane group reads the lane's buffers, a multi-lane group gathers them into its own
            if (g->segs.size() == 1) { LaneState& ln = lanes_[g->segs[0].lane]; g->x = ln.h_x.data(); g->pos = ln.h_pos.data(); }
            else {
                g->x = g->h_x.data(); g->pos = g->h_pos.data();
                for (const Seg& sg : g->segs) {
                    const LaneState& ln = lanes_[sg.lane];
                    std::copy_n(ln.h_x.data(), size_t(sg.T) * H, g->x + size_t(sg.r0) * H);
                    std::copy_n(ln.h_pos.data(), sg.T, g->pos + sg.r0);
                }
            }
        }
        Call k; k.T = g->T; k.x = g->x; k.pos = g->pos; k.segs = g->segs;
        std::string e;
        try { e = run_card(*cards_[s], s, k); }
        catch (const std::exception& ex) { e = std::string("threw: ") + ex.what(); }   // a stage thread must not terminate the process
        catch (...) { e = "threw a non-std exception"; }
        if (!e.empty()) {
            {
                std::lock_guard<std::mutex> lk(pmu_);
                if (perr_.empty()) {
                    perr_ = "lane " + std::to_string(g->segs[0].lane) + " at " + std::to_string(g->segs[0].pos0) + " (" + std::to_string(g->T) + " rows";
                    if (g->segs.size() > 1) perr_ += " of " + std::to_string(g->segs.size()) + " lanes";
                    perr_ += "), card " + std::to_string(s) + ": " + e;
                }
                for (const Seg& sg : g->segs) { lanes_[sg.lane].in_flight = false; --pbusy_; }
                gfree_.push_back(g);
            }
            pcv_.notify_all();
            continue;
        }
        if (!last) {
            { std::lock_guard<std::mutex> lk(pmu_); pq_[s + 1].push_back(g); }
            pcv_.notify_all();
            continue;
        }
        {
            std::lock_guard<std::mutex> lk(pmu_);   // every lane's positions, committed when the group's last stage is done
            for (const Seg& sg : g->segs) {
                LaneState& ln = lanes_[sg.lane];
                ln.n_pos = sg.pos0 + sg.T; ln.hi_end = std::max(ln.hi_end, ln.n_pos); ++n_calls_;
            }
            pgate_ = true; pgate_lanes_ = uint32_t(g->segs.size());   // stage 0 waits until the group's callbacks have all run
            cb_lanes_.clear(); for (const Seg& sg : g->segs) cb_lanes_.push_back(sg.lane);   // (B5: pipe_cb_lanes)
            cb_lanes_tid_ = std::this_thread::get_id();
        }
        for (const Seg& sg : g->segs) {
            // the lane stays in flight through its callback: only this thread may submit it again meanwhile (pipe_submit), so no
            // other thread's step rewrites the buffers the callback reads (its logits, its features)
            LaneState& ln = lanes_[sg.lane];
            const uint32_t rows = feat_layers_.empty() ? 0 : std::min(sg.T, feat_max_);
            { std::lock_guard<std::mutex> lk(pmu_); ln.in_cb = true; ln.resub = false; ln.cb_tid = std::this_thread::get_id(); }
            std::string ce;
            if (pdone_) {
                try { pdone_(sg.lane, ln.logits, ln.h_feat.data(), rows); }
                catch (const std::exception& ex) { ce = std::string("done callback threw: ") + ex.what(); }
                catch (...) { ce = "done callback threw a non-std exception"; }
            }
            std::lock_guard<std::mutex> lk(pmu_);
            if (!ce.empty() && perr_.empty()) perr_ = "lane " + std::to_string(sg.lane) + ": " + ce;
            ln.in_cb = false;
            if (!ln.resub) { ln.in_flight = false; --pbusy_; }   // resubmitted: its in-flight count carries over to the next step
            ln.resub = false;
        }
        {
            std::lock_guard<std::mutex> lk(pmu_);
            pgate_ = false; gfree_.push_back(g);
            cb_lanes_.clear(); cb_lanes_tid_ = std::thread::id();
        }
        pcv_.notify_all();
    }
}

std::string Mimo26Forward::pipe_stop() {
    if (stage_th_.empty()) return {};
    const bool was_paused = ppaused_;   // (paused: the serial members already hold the active lane's positions)
    {
        std::unique_lock<std::mutex> lk(pmu_);
        pcv_.wait(lk, [&] { return pbusy_ == 0; });
        pstop_ = true;
    }
    pcv_.notify_all();
    for (auto& t : stage_th_) t.join();
    stage_th_.clear(); pq_.clear(); lq_.clear(); gfree_.clear(); groups_.clear(); pdone_ = nullptr; ppaused_ = false;
    if (!was_paused) { n_pos_ = lanes_[lane_].n_pos; hi_end_ = lanes_[lane_].hi_end; feat_rows_ = 0; }   // back to the serial path's active lane
    std::string e; std::swap(e, perr_);
    return e;
}

}  // namespace ie
