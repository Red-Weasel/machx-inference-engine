// src/model/deepseek41_dspark.cpp — see include/ie/deepseek41_dspark.hpp (DSpark P1, docs/deepseek41/57).
#include "ie/deepseek41_dspark.hpp"

#include "ie/deepseek41_forward.hpp"
#include "ie/deepseek41_upload.hpp"
#include "ie/deepseek4.hpp"
#include "ie/deepseek4_attn.hpp"
#include "ie/deepseek4_ops.hpp"
#include "ie/kernel_profiler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

namespace ie {
namespace {
inline float bf16f(uint16_t h) { const uint32_t b = uint32_t(h) << 16; float f; std::memcpy(&f, &b, 4); return f; }
constexpr float kLowest = -std::numeric_limits<float>::max();
}  // namespace

void* Ds41Drafter::take(sycl::queue& q, size_t bytes) { void* p = sycl::malloc_device<uint8_t>(bytes, q); if (p) scratch_.push_back(p); return p; }

std::string Ds41Drafter::init(sycl::queue& q, const DeepSeek41Model& m, Ds41DenseCache& cache,
                              const std::vector<std::vector<uint32_t>>& ranking, const Options& opt) {
    const auto& c = m.config();
    if (!m.dspark || !c.dspark_block_size || !c.n_mtp_layers) return "dspark: the checkpoint has no drafter";
    m_ = &m; cache_ = &cache; L0_ = c.n_layers; nS_ = c.n_mtp_layers; B_ = c.dspark_block_size;
    TK_ = c.n_activated(L0_); E_ = c.n_routed(L0_); NT_ = uint32_t(c.dspark_target_layer_ids.size());
    const uint32_t H = c.dim, HD = c.head_dim, RD = c.rope_head_dim, WIN = c.window_size, V = c.vocab_size, R = c.dspark_markov_rank;
    // the three stages' dense sets, like backbone layers
    for (uint32_t s = 0; s < nS_; ++s) if (auto e = cache.upload_layer(q, m, L0_ + s); !e.empty()) return "dspark stage " + std::to_string(s) + ": " + e;
    for (uint32_t s = 0; s < nS_; ++s) dense_bytes_ += cache.layer(L0_ + s)->bytes;
    // the extras
    {
        const auto& t = m.dspark;
        const uint32_t N = uint32_t(t.main_proj.w->shape[0]), K = uint32_t(t.main_proj.w->shape[1]);
        main_proj_.N = N; main_proj_.K = K;
        main_proj_.w = static_cast<uint8_t*>(take(q, size_t(N) * K)); main_proj_.s = static_cast<uint8_t*>(take(q, t.main_proj.s->nbytes));
        if (!main_proj_.w || !main_proj_.s) return "dspark: main_proj alloc failed";
        q.memcpy(main_proj_.w, t.main_proj.w->data, size_t(N) * K); q.memcpy(main_proj_.s, t.main_proj.s->data, t.main_proj.s->nbytes);
        dense_bytes_ += size_t(N) * K + t.main_proj.s->nbytes;
        auto up32 = [&](const Ds41Tensor& tt, float*& dst) -> std::string {
            const size_t n = size_t(tt.w->numel()); std::vector<float> h(n); const auto* s16 = reinterpret_cast<const uint16_t*>(tt.w->data);
            for (size_t i = 0; i < n; ++i) h[i] = bf16f(s16[i]);
            dst = static_cast<float*>(take(q, n * 4)); if (!dst) return "alloc"; q.memcpy(dst, h.data(), n * 4).wait(); dense_bytes_ += n * 4; return {};
        };
        if (auto e = up32(t.main_norm, main_norm_); !e.empty()) return "dspark main_norm: " + e;
        if (auto e = up32(t.norm, norm_); !e.empty()) return "dspark norm: " + e;
        { const size_t n = size_t(V) * R; std::vector<sycl::half> h(n); const auto* s16 = reinterpret_cast<const uint16_t*>(t.markov_head.w->data);
          for (size_t i = 0; i < n; ++i) h[i] = sycl::half(bf16f(s16[i]));
          markov_head_ = static_cast<sycl::half*>(take(q, n * 2)); if (!markov_head_) return "dspark markov_head alloc failed";
          q.memcpy(markov_head_, h.data(), n * 2).wait(); dense_bytes_ += n * 2; }
        { const size_t n = size_t(V) * R; markov_embed_bf16_.resize(n); std::memcpy(markov_embed_bf16_.data(), t.markov_embed.w->data, n * 2); }
        { const size_t n = size_t(t.confidence_proj.w->numel()); conf_proj_.resize(n); const auto* s16 = reinterpret_cast<const uint16_t*>(t.confidence_proj.w->data);
          for (size_t i = 0; i < n; ++i) conf_proj_[i] = bf16f(s16[i]); }
    }
    // the rings, the RoPE tables (main θ, no YaRN: the stages are ratio 0), the pass's scratch
    for (uint32_t s = 0; s < nS_; ++s) { float* r = static_cast<float*>(take(q, size_t(WIN) * HD * 4)); if (!r) return "dspark ring alloc failed"; q.memset(r, 0, size_t(WIN) * HD * 4); rings_.push_back(r); }
    {
        Ds4RopeConfig rmain; rmain.theta = c.rope_theta; rmain.yarn = false;
        const auto inv = ds4_rope_inv_freq(rmain, RD); const uint32_t HR = RD / 2, NP = 1 + B_;
        rope_inv_ = static_cast<float*>(take(q, inv.size() * 4)); rope_pos_ = static_cast<int32_t*>(take(q, size_t(NP) * 4));
        rope_cs_ = static_cast<float*>(take(q, size_t(NP) * HR * 4)); rope_sn_ = static_cast<float*>(take(q, size_t(NP) * HR * 4));
        if (!rope_inv_ || !rope_pos_ || !rope_cs_ || !rope_sn_) return "dspark rope alloc failed";
        q.memcpy(rope_inv_, inv.data(), inv.size() * 4).wait();
    }
    // the expert tier over the stages: the budget from the card's free VRAM after the reserve, the pinned bytes given
    {
        Ds4SlotLayout lay; if (auto e = ds41_slot_layout(H, c.moe_inter_dim, lay); !e.empty()) return e;
        uint32_t n_static = opt.n_static, n_pinned = opt.n_pinned;
        if (n_static + n_pinned == 0) {
            const auto& dev = q.get_device();
            const uint64_t free_now = dev.has(sycl::aspect::ext_intel_free_memory) ? dev.get_info<sycl::ext::intel::info::device::free_memory>() : 0;
            const uint64_t for_experts = free_now > opt.vram_reserve ? free_now - opt.vram_reserve : 0;
            const uint64_t per_slot_all = uint64_t(nS_) * lay.bytes;
            const uint32_t slots = uint32_t(std::min<uint64_t>(E_, for_experts / per_slot_all));
            n_static = slots > opt.stream_slots ? slots - opt.stream_slots : 0;
            n_pinned = uint32_t(std::min<uint64_t>(E_ - n_static, opt.host_pin_bytes / per_slot_all));
        }
        // the tier wants a full permutation per layer: the given ranking's, else index order for the stages
        std::vector<std::vector<uint32_t>> rk(size_t(L0_) + nS_);
        for (uint32_t s = 0; s < nS_; ++s) {
            if (ranking.size() > size_t(L0_) + s && ranking[L0_ + s].size() == E_) rk[L0_ + s] = ranking[L0_ + s];
            else { rk[L0_ + s].resize(E_); std::iota(rk[L0_ + s].begin(), rk[L0_ + s].end(), 0u); }
        }
        if (auto e = tier_.init(q, m, L0_, nS_, rk, n_static, n_pinned, opt.stream_slots, 8, 0, 1); !e.empty()) return "dspark tier: " + e;
    }
    last_idx_.assign(nS_, {});
    q.wait_and_throw();
    ready_ = true;
    return {};
}

void Ds41Drafter::free(sycl::queue& q) {
    if (!ready_) return;
    tier_.free_storage(q);
    for (void* p : scratch_) sycl::free(p, q);
    scratch_.clear(); rings_.clear(); ready_ = false; bufs_ready_ = false; bb_ = Bufs{};
}

// main_x rows from main_hidden [n, dim * n_targets] (host): main_norm(main_proj(.)) -> [n, dim] fp32 on the device
static std::string ds41_dspark_main_x(sycl::queue& q, const float* main_hidden, uint32_t n, uint32_t Kin, uint32_t H, const Ds41Fp8Mat& mp,
                                      const float* main_norm, float eps, sycl::half* in16, float* out) {
    std::vector<sycl::half> h16(size_t(n) * Kin); for (size_t i = 0; i < h16.size(); ++i) h16[i] = sycl::half(main_hidden[i]);
    q.memcpy(in16, h16.data(), h16.size() * 2).wait();
    if (n <= kDs41MaxDecodeRows) gemv_fp8_rows(q, in16, Kin, mp.w, mp.s, out, n, Kin, H);
    else for (uint32_t r0 = 0; r0 < n; r0 += kDs41MaxDecodeRows) { const uint32_t nr = std::min(kDs41MaxDecodeRows, n - r0); gemv_fp8_rows(q, in16 + size_t(r0) * Kin, Kin, mp.w, mp.s, out + size_t(r0) * H, nr, Kin, H); }
    ds4_rms_norm(q, out, main_norm, out, n, H, eps);
    return {};
}

std::string Ds41Drafter::seed(sycl::queue& q, const float* main_hidden, uint32_t n, uint32_t pos0) {
    if (!ready_) return "dspark: not initialised";
    const auto& c = m_->config(); const uint32_t H = c.dim, HD = c.head_dim, RD = c.rope_head_dim, WIN = c.window_size, Kin = H * NT_;
    const uint32_t nw = std::min(n, WIN), first = n - nw;                       // the last min(n, window) positions land (only they are projected)
    sycl::half* in16 = static_cast<sycl::half*>(take(q, size_t(nw) * Kin * 2)); float* mx = static_cast<float*>(take(q, size_t(nw) * H * 4));
    sycl::half* mx16 = static_cast<sycl::half*>(take(q, size_t(nw) * H * 2)); float* kv = static_cast<float*>(take(q, size_t(nw) * HD * 4));
    int32_t* dpos = static_cast<int32_t*>(take(q, size_t(nw) * 4)); float* cs = static_cast<float*>(take(q, size_t(nw) * (RD / 2) * 4)); float* sn = static_cast<float*>(take(q, size_t(nw) * (RD / 2) * 4));
    if (!in16 || !mx || !mx16 || !kv || !dpos || !cs || !sn) return "dspark seed: alloc failed";
    if (auto e = ds41_dspark_main_x(q, main_hidden + size_t(first) * Kin, nw, Kin, H, main_proj_, main_norm_, c.norm_eps, in16, mx); !e.empty()) return e;
    last_main_x_.resize(size_t(nw) * H); q.memcpy(last_main_x_.data(), mx, last_main_x_.size() * 4);    // lands before the wait below
    std::vector<int32_t> pos(nw); std::iota(pos.begin(), pos.end(), int32_t(pos0 + first));
    q.memcpy(dpos, pos.data(), nw * 4); ds4_rope_cos_sin(q, rope_inv_, dpos, cs, sn, nw, RD / 2, 1.0f);
    q.parallel_for(sycl::range<1>(size_t(nw) * H), [=](sycl::id<1> i) { mx16[i] = sycl::half(mx[i]); });
    for (uint32_t s = 0; s < nS_; ++s) {
        const Ds41LayerDense* rd = cache_->layer(L0_ + s); const sycl::half* x16 = mx16;
        for (uint32_t r0 = 0; r0 < nw; r0 += kDs41MaxDecodeRows) {                  // the rows kernels take at most 8 rows
            const uint32_t nr = std::min(kDs41MaxDecodeRows, nw - r0);
            if (rd->f8_wkv.w) gemv_fp8_rows(q, x16 + size_t(r0) * H, H, rd->f8_wkv.w, rd->f8_wkv.s, kv + size_t(r0) * HD, nr, H, HD);
            else gemv_f16_rows(q, x16 + size_t(r0) * H, H, rd->wkv, kv + size_t(r0) * HD, nr, H, HD);
        }
        ds4_rms_norm(q, kv, rd->n_kv, kv, nw, HD, c.norm_eps);
        ds4_rope_apply(q, kv, cs, sn, kv, nw, 1, HD, RD, +1.0f);
        float* ring = rings_[s]; const uint32_t p0 = pos0 + first;
        q.parallel_for(sycl::range<1>(size_t(nw) * HD), [=](sycl::id<1> i) { const uint32_t r = uint32_t(i[0] / HD), d = uint32_t(i[0] % HD); ring[size_t((p0 + r) % WIN) * HD + d] = kv[size_t(r) * HD + d]; });
    }
    q.wait_and_throw();
    for (void* p : {static_cast<void*>(in16), static_cast<void*>(mx), static_cast<void*>(mx16), static_cast<void*>(kv), static_cast<void*>(dpos), static_cast<void*>(cs), static_cast<void*>(sn)}) { sycl::free(p, q); scratch_.erase(std::find(scratch_.begin(), scratch_.end(), p)); }
    return {};
}

std::string Ds41Drafter::draft(sycl::queue& q, const float* main_hidden, int32_t token, uint32_t pos,
                               std::vector<int32_t>& ids, std::vector<float>& logits, std::vector<float>& confidence,
                               const Probe& probe, const std::vector<const int32_t*>* forced_idx, const std::vector<const float*>* forced_w) {
    if (!ready_) return "dspark: not initialised";
    const auto t0 = std::chrono::steady_clock::now();
    const auto& c = m_->config();
    const uint32_t H = c.dim, HC = c.hc_mult, FLAT = HC * H, HD = c.head_dim, RD = c.rope_head_dim, WIN = c.window_size, V = c.vocab_size, R = c.dspark_markov_rank;
    const uint32_t NH = c.n_heads, QH = NH * HD, QR = c.q_lora_rank, G = c.o_groups, OPG = c.o_lora_rank, IPG = QH / G, OR = G * OPG, EF = c.moe_inter_dim, HR = RD / 2;
    const uint32_t B = B_, TK = TK_, E = E_, Kin = H * NT_;
    if (main_hidden) if (auto e = seed(q, main_hidden, 1, pos); !e.empty()) return e;   // the ring gets this position's main_kv first (model.py:1065)
    // ---- scratch (persistent across passes: the first pass allocates)
    auto S32 = [&](size_t n) { return static_cast<float*>(take(q, n * 4)); };
    auto S16 = [&](size_t n) { return static_cast<sycl::half*>(take(q, n * 2)); };
    Bufs& bb = bb_;
    if (!bufs_ready_) {
        bb.hA = S32(size_t(B) * FLAT); bb.hB = S32(size_t(B) * FLAT); bb.pre_mix = S32(size_t(B) * HC); bb.xa = S32(size_t(B) * H); bb.xn = S32(size_t(B) * H);
        bb.qr = S32(size_t(B) * QR); bb.qrn = S32(size_t(B) * QR); bb.qq = S32(size_t(B) * QH); bb.kvw = S32(size_t(B) * HD); bb.oo = S32(size_t(B) * QH); bb.oa = S32(size_t(B) * OR); bb.ao = S32(size_t(B) * H);
        bb.xf = S32(size_t(B) * H); bb.xfn = S32(size_t(B) * H); bb.moe = S32(size_t(B) * H); bb.shg = S32(size_t(B) * EF); bb.shu = S32(size_t(B) * EF); bb.shh = S32(size_t(B) * H);
        bb.mask = S32(size_t(B) * (WIN + B)); bb.r_logits = S32(size_t(B) * E); bb.r_w = S32(size_t(B) * TK); bb.r_i = static_cast<int32_t*>(take(q, size_t(B) * TK * 4));
        bb.a_pre = S32(size_t(B) * HC); bb.a_post = S32(size_t(B) * HC); bb.a_comb = S32(size_t(B) * HC * HC); bb.f_pre = S32(size_t(B) * HC); bb.f_post = S32(size_t(B) * HC); bb.f_comb = S32(size_t(B) * HC * HC);
        bb.hcs = S32(size_t(B) * ds41_hc_mixes_scratch_floats(HC)); bb.lg = S32(size_t(B) * V); bb.bias = S32(V); bb.xhead = S32(size_t(B) * H);
        bb.x16 = S16(size_t(B) * H); bb.t16 = S16(size_t(B) * std::max({QR, QH, OR, EF})); bb.e16 = S16(R);
        bufs_ready_ = true;
    }
    // ---- the five rows: embed(token), embed(noise) x 4, broadcast over hc; the identity mix
    {
        std::vector<float> hh(size_t(B) * FLAT); const auto* EW = reinterpret_cast<const uint16_t*>(m_->embed.w->data);
        for (uint32_t t = 0; t < B; ++t) { const int32_t id = t == 0 ? token : int32_t(c.dspark_noise_token_id);
            for (uint32_t d = 0; d < H; ++d) { const float v = bf16f(EW[size_t(id) * H + d]); for (uint32_t cp = 0; cp < HC; ++cp) hh[(size_t(t) * HC + cp) * H + d] = v; } }
        std::vector<float> pm(size_t(B) * HC, 0.f); for (uint32_t t = 0; t < B; ++t) pm[size_t(t) * HC] = 1.f;
        q.memcpy(bb.hA, hh.data(), hh.size() * 4); q.memcpy(bb.pre_mix, pm.data(), pm.size() * 4);
        if (probe) probe("embed", 0, bb.hA, hh.size(), q);
    }
    // ---- the draft rows' positions pos + 1 .. pos + B (the diagnostic shifts them)
    { std::vector<int32_t> p(B); for (uint32_t t = 0; t < B; ++t) p[t] = std::max(0, int32_t(pos + 1 + t) + rope_off_);
      q.memcpy(rope_pos_, p.data(), B * 4); ds4_rope_cos_sin(q, rope_inv_, rope_pos_, rope_cs_, rope_sn_, B, HR, 1.0f); }
    // ---- the mask [B, WIN + B]: every filled ring slot (<= pos), every draft column (no causal mask among the drafts)
    { float* mask = bb.mask; const uint32_t NKV = WIN + B, lim = std::min(WIN, pos + 1); const bool causal = causal_diag_;
      q.parallel_for(sycl::range<1>(size_t(B) * NKV), [=](sycl::id<1> i) { const uint32_t t = uint32_t(i[0] / NKV), kk = uint32_t(i[0] % NKV);
          mask[i] = kk < WIN ? (kk < lim ? 0.f : kLowest) : ((!causal || kk - WIN <= t) ? 0.f : kLowest); });
      if (probe) probe("mask", 0, bb.mask, size_t(B) * NKV, q); }
    float* h = bb.hA; float* h_next = bb.hB; float* pre_mix = bb.pre_mix;
    for (uint32_t s = 0; s < nS_; ++s) {
        const uint32_t L = L0_ + s; const Ds41LayerDense* rd = cache_->layer(L);
        // attention site
        ds41_hc_mixes(q, h, rd->a_fn, rd->a_bs, rd->a_sc, bb.a_pre, bb.a_post, bb.a_comb, B, H, HC, c.hc_sinkhorn_iters, c.norm_eps, c.hc_eps, {}, bb.hcs);
        ds41_hc_collapse(q, h, pre_mix, bb.xa, B, H, HC);
        ds4_rms_norm(q, bb.xa, rd->n_at, bb.xn, B, H, c.norm_eps);
        if (probe) probe("attn_in", s, bb.xn, size_t(B) * H, q);
        auto proj = [&](const sycl::half* act, uint32_t K_, const Ds41Fp8Mat& f8, sycl::half* w16, float* out, uint32_t N_) {
            if (f8.w) gemv_fp8_rows(q, act, K_, f8.w, f8.s, out, B, K_, N_); else gemv_f16_rows(q, act, K_, w16, out, B, K_, N_); };
        q.parallel_for(sycl::range<1>(size_t(B) * H), [=, x16 = bb.x16, xn = bb.xn](sycl::id<1> i) { x16[i] = sycl::half(xn[i]); });
        proj(bb.x16, H, rd->f8_wq_a, rd->wq_a, bb.qr, QR);
        ds4_rms_norm(q, bb.qr, rd->n_q, bb.qrn, B, QR, c.norm_eps);
        q.parallel_for(sycl::range<1>(size_t(B) * QR), [=, t16 = bb.t16, qrn = bb.qrn](sycl::id<1> i) { t16[i] = sycl::half(qrn[i]); });
        proj(bb.t16, QR, rd->f8_wq_b, rd->wq_b, bb.qq, QH);
        ds4_rope_apply(q, bb.qq, rope_cs_, rope_sn_, bb.qq, B, NH, HD, RD, +1.0f);
        proj(bb.x16, H, rd->f8_wkv, rd->wkv, bb.kvw, HD);
        ds4_rms_norm(q, bb.kvw, rd->n_kv, bb.kvw, B, HD, c.norm_eps);
        ds4_rope_apply(q, bb.kvw, rope_cs_, rope_sn_, bb.kvw, B, 1, HD, RD, +1.0f);
        Ds4KvSegs segs; segs.a = rings_[s]; segs.n_a = WIN; segs.b = bb.kvw; segs.n_b = B; segs.b_f16 = false;
        ds4_attention_segs(q, bb.qq, segs, bb.mask, rd->sinks, bb.oo, B, NH, HD, 1.0f / std::sqrt(float(HD)));
        ds4_rope_apply(q, bb.oo, rope_cs_, rope_sn_, bb.oo, B, NH, HD, RD, -1.0f);
        q.parallel_for(sycl::range<1>(size_t(B) * QH), [=, t16 = bb.t16, oo = bb.oo](sycl::id<1> i) { t16[i] = sycl::half(oo[i]); });
        if (rd->f8_wo_a.w) gemv_fp8_rows_grouped(q, bb.t16, QH, rd->f8_wo_a.w, rd->f8_wo_a.s, bb.oa, B, IPG, OR, OPG);   // Phase 32
        else gemv_f16_rows(q, bb.t16, QH, rd->wo_a, bb.oa, B, IPG, OR, OPG);
        q.parallel_for(sycl::range<1>(size_t(B) * OR), [=, t16 = bb.t16, oa = bb.oa](sycl::id<1> i) { t16[i] = sycl::half(oa[i]); });
        proj(bb.t16, OR, rd->f8_wo_b, rd->wo_b, bb.ao, H);
        if (probe) probe("attn_out", s, bb.ao, size_t(B) * H, q);
        ds4_hc_mix(q, h, bb.a_post, bb.a_comb, bb.ao, h_next, B, H, HC).wait();
        std::swap(h, h_next);
        // ffn site
        ds41_hc_mixes(q, h, rd->f_fn, rd->f_bs, rd->f_sc, bb.f_pre, bb.f_post, bb.f_comb, B, H, HC, c.hc_sinkhorn_iters, c.norm_eps, c.hc_eps, {}, bb.hcs);
        ds41_hc_collapse(q, h, bb.a_pre, bb.xf, B, H, HC);
        ds4_rms_norm(q, bb.xf, rd->n_ff, bb.xfn, B, H, c.norm_eps).wait();
        if (probe) probe("ffn_in", s, bb.xfn, size_t(B) * H, q);
        std::vector<int32_t> h_idx(size_t(B) * TK); std::vector<float> h_w(size_t(B) * TK);
        ds4_router_topk(q, bb.xfn, rd->g_w, rd->g_b, bb.r_logits, bb.r_w, bb.r_i, B, H, E, TK, c.route_scale).wait();
        q.memcpy(h_idx.data(), bb.r_i, h_idx.size() * 4); q.memcpy(h_w.data(), bb.r_w, h_w.size() * 4); q.wait();
        last_idx_[s] = h_idx;
        if (forced_idx && forced_w) { std::memcpy(h_idx.data(), (*forced_idx)[s], h_idx.size() * 4); std::memcpy(h_w.data(), (*forced_w)[s], h_w.size() * 4); }
        if (auto e = tier_.moe(q, L, bb.xfn, h_idx.data(), h_w.data(), B, bb.moe, c.swiglu_limit); !e.empty()) return "dspark stage " + std::to_string(s) + " moe: " + e;
        q.parallel_for(sycl::range<1>(size_t(B) * H), [=, x16 = bb.x16, xfn = bb.xfn](sycl::id<1> i) { x16[i] = sycl::half(xfn[i]); });
        proj(bb.x16, H, rd->f8_sh_w1, rd->sh_w1, bb.shg, EF);
        proj(bb.x16, H, rd->f8_sh_w3, rd->sh_w3, bb.shu, EF);
        { const float Lim = c.swiglu_limit; float* shg = bb.shg; float* shu = bb.shu;
          q.parallel_for(sycl::range<1>(size_t(B) * EF), [=](sycl::id<1> i) { float gg = shg[i], uu = shu[i]; if (Lim > 0.f) { gg = sycl::fmin(gg, Lim); uu = sycl::fmax(sycl::fmin(uu, Lim), -Lim); } shg[i] = (gg / (1.0f + sycl::exp(-gg))) * uu; }); }
        q.parallel_for(sycl::range<1>(size_t(B) * EF), [=, t16 = bb.t16, shg = bb.shg](sycl::id<1> i) { t16[i] = sycl::half(shg[i]); });
        proj(bb.t16, EF, rd->f8_sh_w2, rd->sh_w2, bb.shh, H);
        { float* moe = bb.moe; float* shh = bb.shh; q.parallel_for(sycl::range<1>(size_t(B) * H), [=](sycl::id<1> i) { moe[i] += shh[i]; }); }
        if (probe) probe("moe_out", s, bb.moe, size_t(B) * H, q);
        ds4_hc_mix(q, h, bb.f_post, bb.f_comb, bb.moe, h_next, B, H, HC).wait();
        std::swap(h, h_next);
        if (probe) { probe("layer_out", s, h, size_t(B) * FLAT, q); probe("ffn_pre", s, bb.f_pre, size_t(B) * HC, q); }
        std::swap(pre_mix, bb.f_pre);          // the stage's ffn_pre feeds the next collapse (bb.f_pre now the old pre_mix, free to write)
    }
    // ---- the head: collapse, the last stage's norm, the tied head over the B rows
    ds41_hc_collapse(q, h, pre_mix, bb.xhead, B, H, HC);
    if (probe) probe("head_in", 0, bb.xhead, size_t(B) * H, q);
    ds4_rms_norm(q, bb.xhead, norm_, bb.xn, B, H, c.norm_eps);
    q.parallel_for(sycl::range<1>(size_t(B) * H), [=, x16 = bb.x16, xn = bb.xn](sycl::id<1> i) { x16[i] = sycl::half(xn[i]); });
    gemv_f16_rows(q, bb.x16, H, const_cast<sycl::half*>(cache_->head()), bb.lg, B, H, V).wait();
    if (probe) probe("logits_base", 0, bb.lg, size_t(B) * V, q);
    logits.resize(size_t(B) * V); q.memcpy(logits.data(), bb.lg, logits.size() * 4).wait();
    std::vector<float> head_in(size_t(B) * H); q.memcpy(head_in.data(), bb.xhead, head_in.size() * 4).wait();
    // ---- the Markov chain: row i's bias from the previous id's embedding, argmax -> the next id; the confidence
    ids.assign(B + 1, 0); ids[0] = token; confidence.assign(B, 0.f);
    last_embed_.resize(size_t(B) * R); last_bias_.resize(size_t(B) * V); std::vector<sycl::half> emb16(R);
    for (uint32_t i = 0; i < B; ++i) {
        const int32_t id = ids[i]; const uint16_t* row = markov_embed_bf16_.data() + size_t(id) * R;
        float* emb = last_embed_.data() + size_t(i) * R; float* bias = last_bias_.data() + size_t(i) * V;
        for (uint32_t r = 0; r < R; ++r) { emb[r] = bf16f(row[r]); emb16[r] = sycl::half(emb[r]); }
        q.memcpy(bb.e16, emb16.data(), R * 2);
        gemv_f16_rows(q, bb.e16, R, markov_head_, bb.bias, 1, R, V).wait();
        q.memcpy(bias, bb.bias, V * 4).wait();
        float* lr = logits.data() + size_t(i) * V; float best = kLowest; int32_t arg = 0;
        for (uint32_t v = 0; v < V; ++v) { lr[v] += bias[v]; if (lr[v] > best) { best = lr[v]; arg = int32_t(v); } }
        ids[i + 1] = arg;
        double cf = 0; const float* hi = head_in.data() + size_t(i) * H;
        for (uint32_t d = 0; d < H; ++d) cf += double(conf_proj_[d]) * hi[d];
        for (uint32_t r = 0; r < R; ++r) cf += double(conf_proj_[H + r]) * emb[r];
        confidence[i] = float(cf);
    }
    last_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return {};
}

}  // namespace ie
