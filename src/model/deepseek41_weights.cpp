// src/model/deepseek41_weights.cpp — see the header.
#include "ie/deepseek41_weights.hpp"
#include "ie/deepseek41_upload.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

#include "ie/fp8.hpp"

namespace ie {

namespace {
inline float bf16f(uint16_t h) { const uint32_t b = uint32_t(h) << 16; float f; std::memcpy(&f, &b, 4); return f; }

struct Up {
    sycl::queue& q; Ds41LayerDense& d;
    std::vector<sycl::half> h16; std::vector<float> h32;
    bool fp8 = false;
    // FP8 + 32x32 scale -> fp16 on the device; or (fp8 mode, `f8` given) the pair itself stays
    // resident and the fp16 pointer is null (Phase 17, docs/deepseek41/43)
    sycl::half* dense(const Ds41Tensor& t, Ds41Fp8Mat* f8 = nullptr) {
        const uint32_t N = uint32_t(t.w->shape[0]), K = uint32_t(t.w->shape[1]);
        const uint32_t bn = N / uint32_t(t.s->shape[0]), bk = K / uint32_t(t.s->shape[1]);
        uint8_t* dw = sycl::malloc_device<uint8_t>(size_t(N) * K, q);
        uint8_t* ds = sycl::malloc_device<uint8_t>(t.s->nbytes, q);
        if (fp8 && f8) {
            if (bn != 32 || bk != 32 || K % 32) { sycl::free(dw, q); sycl::free(ds, q); return nullptr; }   // the GEMV's block shape only
            q.memcpy(dw, t.w->data, size_t(N) * K); q.memcpy(ds, t.s->data, t.s->nbytes); q.wait();
            f8->w = dw; f8->s = ds; f8->N = N; f8->K = K;
            d.owned.push_back(dw); d.owned.push_back(ds); d.bytes += size_t(N) * K + t.s->nbytes; return nullptr;
        }
        sycl::half* o = sycl::malloc_device<sycl::half>(size_t(N) * K, q);
        q.memcpy(dw, t.w->data, size_t(N) * K); q.memcpy(ds, t.s->data, t.s->nbytes);
        ds41_dense_dequant_f16(q, dw, ds, N, K, bn, bk, o).wait();
        sycl::free(dw, q); sycl::free(ds, q);
        d.owned.push_back(o); d.bytes += size_t(N) * K * 2; return o;
    }
    // tensor parallel: rows [r0, r0 + nr) x columns [k0, k0 + nk) of an FP8 matrix (whole 32 x 32 blocks), resident as its own pair
    std::string dense_slice(const Ds41Tensor& t, Ds41Fp8Mat& f8, uint32_t r0, uint32_t nr, uint32_t k0, uint32_t nk) {
        const uint32_t N = uint32_t(t.w->shape[0]), K = uint32_t(t.w->shape[1]);
        if (N / uint32_t(t.s->shape[0]) != 32 || K / uint32_t(t.s->shape[1]) != 32) return "not 32 x 32 FP8 blocks";
        if ((r0 | nr | k0 | nk) % 32 || r0 + nr > N || k0 + nk > K || !nr || !nk) return "slice off the 32-blocks or outside the matrix";
        std::vector<uint8_t> w(size_t(nr) * nk), sc(size_t(nr / 32) * (nk / 32));
        for (uint32_t r = 0; r < nr; ++r) std::memcpy(w.data() + size_t(r) * nk, t.w->data + size_t(r0 + r) * K + k0, nk);
        for (uint32_t b = 0; b < nr / 32; ++b) std::memcpy(sc.data() + size_t(b) * (nk / 32), t.s->data + size_t(r0 / 32 + b) * (K / 32) + k0 / 32, nk / 32);
        uint8_t* dw = sycl::malloc_device<uint8_t>(w.size(), q); uint8_t* ds = sycl::malloc_device<uint8_t>(sc.size(), q);
        if (!dw || !ds) { if (dw) sycl::free(dw, q); if (ds) sycl::free(ds, q); return "alloc failed"; }
        q.memcpy(dw, w.data(), w.size()); q.memcpy(ds, sc.data(), sc.size()); q.wait();
        f8.w = dw; f8.s = ds; f8.N = nr; f8.K = nk;
        d.owned.push_back(dw); d.owned.push_back(ds); d.bytes += w.size() + sc.size(); return {};
    }
    sycl::half* bf16_16(const Ds41Tensor& t) {
        const size_t n = size_t(t.w->numel()); const auto* s = reinterpret_cast<const uint16_t*>(t.w->data);
        h16.resize(n); for (size_t i = 0; i < n; ++i) h16[i] = sycl::half(bf16f(s[i]));
        sycl::half* o = sycl::malloc_device<sycl::half>(n, q); q.memcpy(o, h16.data(), n * 2).wait();
        d.owned.push_back(o); d.bytes += n * 2; return o;
    }
    float* bf16_32(const Ds41Tensor& t) {
        const size_t n = size_t(t.w->numel()); const auto* s = reinterpret_cast<const uint16_t*>(t.w->data);
        h32.resize(n); for (size_t i = 0; i < n; ++i) h32[i] = bf16f(s[i]);
        float* o = sycl::malloc_device<float>(n, q); q.memcpy(o, h32.data(), n * 4).wait();
        d.owned.push_back(o); d.bytes += n * 4; return o;
    }
    float* f32(const Ds41Tensor& t) {
        const size_t n = size_t(t.w->numel()); float* o = sycl::malloc_device<float>(n, q);
        q.memcpy(o, t.w->data, n * 4).wait(); d.owned.push_back(o); d.bytes += n * 4; return o;
    }
};
}  // namespace

std::string Ds41DenseCache::upload_layer(sycl::queue& q, const DeepSeek41Model& m, uint32_t L) {
    const auto& c = m.config();
    if (L >= m.layers().size()) return "upload_layer: layer out of range";
    if (layers_.size() < m.layers().size()) layers_.resize(m.layers().size());
    if (layers_[L].bytes) return {};                      // already resident
    const auto& Lw = m.layers()[L]; const auto& k = Lw.kind;
    Ds41LayerDense d; Up u{q, d, {}, {}};
    if (layers_.empty() || !std::any_of(layers_.begin(), layers_.end(), [](const Ds41LayerDense& x) { return x.bytes != 0; })) {
        const char* v = std::getenv("IE_DS41_DENSE_FP8"); fp8_ = !(v && *v && std::string(v) == "0");   /* ON since gate 17 (docs/47); =0 the fp16 dense */   // decided at the first layer
    }
    u.fp8 = fp8_;
    if (tp_n_ > 1) {
        // tensor parallel: this card's query heads [h0, h0 + nh) = o_groups [g0, g0 + ng) (heads per group = n_heads / o_groups),
        // and the shared expert's units [f0, f0 + nf)
        const uint32_t n = tp_n_, NH = c.n_heads, HD = c.head_dim, G = c.o_groups, OPG = c.o_lora_rank, EF = c.moe_inter_dim;
        if (!fp8_) return "upload_layer: tensor parallel needs the FP8-resident dense (IE_DS41_DENSE_FP8 is 0)";
        if (NH % n || G % n || (NH / G) * (G / n) != NH / n || EF % (32 * n)) return "upload_layer: " + std::to_string(n) + " tensor-parallel parts do not split the heads, o_groups and shared units";
        const uint32_t nh = NH / n, h0 = tp_part_ * nh, ng = G / n, g0 = tp_part_ * ng, nf = EF / n, f0 = tp_part_ * nf;
        auto chk = [&](const char* nm, const Ds41Tensor& t, uint64_t N, uint64_t K) -> std::string {
            return (uint64_t(t.w->shape[0]) == N && uint64_t(t.w->shape[1]) == K) ? std::string() : std::string(nm) + " is not [" + std::to_string(N) + ", " + std::to_string(K) + "]"; };
        const uint32_t IPG = NH * HD / G;
        std::string e = chk("wq_b", Lw.wq_b, uint64_t(NH) * HD, c.q_lora_rank);
        if (e.empty()) e = chk("wo_a", Lw.wo_a, uint64_t(G) * OPG, IPG);
        if (e.empty()) e = chk("wo_b", Lw.wo_b, c.dim, uint64_t(G) * OPG);
        if (e.empty()) e = chk("sh_w1", Lw.sh_w1, EF, c.dim);
        if (e.empty()) e = chk("sh_w3", Lw.sh_w3, EF, c.dim);
        if (e.empty()) e = chk("sh_w2", Lw.sh_w2, c.dim, EF);
        if (e.empty()) e = u.dense_slice(Lw.wq_b, d.f8_wq_b, h0 * HD, nh * HD, 0, c.q_lora_rank);
        if (e.empty()) e = u.dense_slice(Lw.wo_a, d.f8_wo_a, g0 * OPG, ng * OPG, 0, IPG);
        if (e.empty()) e = u.dense_slice(Lw.wo_b, d.f8_wo_b, 0, c.dim, g0 * OPG, ng * OPG);
        if (e.empty()) e = u.dense_slice(Lw.sh_w1, d.f8_sh_w1, f0, nf, 0, c.dim);
        if (e.empty()) e = u.dense_slice(Lw.sh_w3, d.f8_sh_w3, f0, nf, 0, c.dim);
        if (e.empty()) e = u.dense_slice(Lw.sh_w2, d.f8_sh_w2, 0, c.dim, f0, nf);
        if (!e.empty()) return "upload_layer: tensor-parallel slice (layer " + std::to_string(L) + "): " + e;
        if (uint64_t(Lw.attn_sink.w->numel()) != NH) return "upload_layer: attn_sink is not one value per head";
        d.sinks = sycl::malloc_device<float>(nh, q);
        q.memcpy(d.sinks, reinterpret_cast<const float*>(Lw.attn_sink.w->data) + h0, size_t(nh) * 4).wait();
        d.owned.push_back(d.sinks); d.bytes += size_t(nh) * 4;
        d.wq_a = u.dense(Lw.wq_a, &d.f8_wq_a); d.wkv = u.dense(Lw.wkv, &d.f8_wkv);
    } else {
    d.wq_a = u.dense(Lw.wq_a, &d.f8_wq_a); d.wq_b = u.dense(Lw.wq_b, &d.f8_wq_b); d.wkv = u.dense(Lw.wkv, &d.f8_wkv);
    d.wo_a = u.dense(Lw.wo_a, &d.f8_wo_a); d.wo_b = u.dense(Lw.wo_b, &d.f8_wo_b);
    d.sh_w1 = u.dense(Lw.sh_w1, &d.f8_sh_w1); d.sh_w3 = u.dense(Lw.sh_w3, &d.f8_sh_w3); d.sh_w2 = u.dense(Lw.sh_w2, &d.f8_sh_w2);
    d.sinks = u.f32(Lw.attn_sink);
    }
    if (fp8_ && !(d.f8_wq_a.w && d.f8_wq_b.w && d.f8_wkv.w && d.f8_wo_a.w && d.f8_wo_b.w && d.f8_sh_w1.w && d.f8_sh_w3.w && d.f8_sh_w2.w))
        return "upload_layer: IE_DS41_DENSE_FP8 needs 32x32 FP8 blocks and K % 32 == 0 on every dense matrix (layer " + std::to_string(L) + ")";
    d.n_q = u.bf16_32(Lw.q_norm); d.n_kv = u.bf16_32(Lw.kv_norm); d.n_at = u.bf16_32(Lw.attn_norm); d.n_ff = u.bf16_32(Lw.ffn_norm);
    d.a_fn = u.f32(Lw.hc_attn_fn); d.a_bs = u.f32(Lw.hc_attn_base); d.a_sc = u.f32(Lw.hc_attn_scale);
    d.f_fn = u.f32(Lw.hc_ffn_fn);  d.f_bs = u.f32(Lw.hc_ffn_base);  d.f_sc = u.f32(Lw.hc_ffn_scale);
    d.g_w = u.bf16_32(Lw.gate_w);  d.g_b = u.f32(Lw.gate_bias);
    if (Lw.gate_bias_vl.w) d.g_b_vl = u.f32(Lw.gate_bias_vl);
    if (k.is_kv_source) {
        d.comp_wkv = u.bf16_16(Lw.comp_wkv); d.n_c = u.bf16_32(Lw.comp_norm);
        if (k.has_compressor_gate) d.comp_wgate = u.bf16_16(Lw.comp_wgate);
        d.idx_wk = u.bf16_16(Lw.idx_wk); d.n_ik = u.bf16_32(Lw.idx_k_norm);
    }
    if (k.is_index_source) { d.idx_wq_b = u.dense(Lw.idx_wq_b, &d.f8_idx_wq_b); d.idx_weights = u.bf16_16(Lw.idx_weights); }
    if (k.has_engram) {
        d.engram_wkv = u.dense(Lw.engram_wkv, &d.f8_engram_wkv);
        const size_t n = size_t(c.hc_mult) * c.dim;
        const auto* qw = reinterpret_cast<const uint16_t*>(Lw.engram_q.w->data);
        const auto* kw = reinterpret_cast<const uint16_t*>(Lw.engram_k.w->data);
        std::vector<float> qk(n); for (size_t i = 0; i < n; ++i) qk[i] = bf16f(qw[i]) * bf16f(kw[i]);
        d.engram_qk = sycl::malloc_device<float>(n, q); q.memcpy(d.engram_qk, qk.data(), n * 4).wait();
        d.owned.push_back(d.engram_qk); d.bytes += n * 4;
    }
    layers_[L] = std::move(d);
    return {};
}

const sycl::half* Ds41DenseCache::f16(sycl::queue& q, const Ds41Fp8Mat& m) {
    const size_t need = size_t(m.N) * m.K;
    if (need > scratch_halves_ || scratch_q_ != &q) {
        q.wait();
        if (scratch_ && scratch_q_) sycl::free(scratch_, *scratch_q_);
        scratch_ = sycl::malloc_device<sycl::half>(need, q); scratch_halves_ = scratch_ ? need : 0; scratch_q_ = &q;
        if (!scratch_) return nullptr;
    }
    ds41_dense_dequant_f16(q, m.w, m.s, m.N, m.K, 32, 32, scratch_);
    return scratch_;
}

std::string Ds41DenseCache::upload_head(sycl::queue& q, const DeepSeek41Model& m, bool fp8_head) {
    if (head_ || head8_.w) return {};
    Ds41LayerDense tmp; Up u{q, tmp, {}, {}};
    final_norm_ = u.bf16_32(m.final_norm);
    if (!fp8_head) {
        head_       = u.bf16_16(m.lm_head);
        head_bytes_ = tmp.bytes;
        return {};
    }
    // Phase 54: BF16 -> E4M3 + E8M0 per 32x32 block, round to nearest representable code. The block exponent is the
    // smallest power of two that keeps the block's largest magnitude within E4M3's 448, so nothing saturates.
    const uint32_t N = uint32_t(m.lm_head.w->shape[0]), K = uint32_t(m.lm_head.w->shape[1]);
    if (N % 32 || K % 32) return "upload_head: the FP8 head needs both dimensions a multiple of 32";
    const auto* src = reinterpret_cast<const uint16_t*>(m.lm_head.w->data);
    const uint32_t BN = N / 32, BK = K / 32;
    std::vector<uint8_t> w8(size_t(N) * K), s8(size_t(BN) * BK);
    // the 127 non-negative finite codes by value (0x00..0x7E are ascending; 0x7F is NaN)
    float code_val[127]; for (int c = 0; c < 127; ++c) code_val[c] = e4m3_to_f32(uint8_t(c));
    auto encode = [&](float x) -> uint8_t {
        const float a = std::fabs(x);
        const float* it = std::lower_bound(code_val, code_val + 127, a);
        int c = int(it - code_val);
        if (c >= 127) c = 126;
        else if (c > 0 && (a - code_val[c - 1]) <= (code_val[c] - a)) c = c - 1;
        return uint8_t(c) | ((x < 0.f && c) ? uint8_t(0x80) : uint8_t(0));
    };
    const unsigned nth = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (unsigned t = 0; t < nth; ++t) th.emplace_back([&, t] {
        for (uint32_t bn = BN * t / nth; bn < BN * (t + 1) / nth; ++bn)
            for (uint32_t bk = 0; bk < BK; ++bk) {
                float amax = 0.f;
                for (uint32_t r = bn * 32; r < (bn + 1) * 32; ++r)
                    for (uint32_t k = bk * 32; k < (bk + 1) * 32; ++k) amax = std::max(amax, std::fabs(bf16f(src[size_t(r) * K + k])));
                int e = amax > 0.f ? int(std::ceil(std::log2(double(amax) / 448.0))) : -127;
                e = std::max(-127, std::min(127, e));
                s8[size_t(bn) * BK + bk] = uint8_t(e + 127);
                const float inv = 1.0f / e8m0_to_f32(uint8_t(e + 127));
                for (uint32_t r = bn * 32; r < (bn + 1) * 32; ++r)
                    for (uint32_t k = bk * 32; k < (bk + 1) * 32; ++k) w8[size_t(r) * K + k] = encode(bf16f(src[size_t(r) * K + k]) * inv);
            }
    });
    for (auto& x : th) x.join();
    uint8_t* dw = sycl::malloc_device<uint8_t>(w8.size(), q);
    uint8_t* ds = sycl::malloc_device<uint8_t>(s8.size(), q);
    if (!dw || !ds) { if (dw) sycl::free(dw, q); if (ds) sycl::free(ds, q); return "upload_head: FP8 head alloc failed"; }
    q.memcpy(dw, w8.data(), w8.size()); q.memcpy(ds, s8.data(), s8.size()); q.wait_and_throw();
    head8_.w = dw; head8_.s = ds; head8_.N = N; head8_.K = K;
    head_bytes_ = tmp.bytes + w8.size() + s8.size();
    return {};
}

uint64_t Ds41DenseCache::bytes() const {
    uint64_t b = head_bytes_;
    for (const auto& l : layers_) b += l.bytes;
    return b;
}

void Ds41DenseCache::free_all(sycl::queue& q) {
    if (scratch_ && scratch_q_) { scratch_q_->wait(); sycl::free(scratch_, *scratch_q_); scratch_ = nullptr; scratch_halves_ = 0; scratch_q_ = nullptr; }
    for (auto& l : layers_) { for (void* p : l.owned) sycl::free(p, q); l = Ds41LayerDense{}; }
    if (final_norm_) sycl::free(final_norm_, q); if (head_) sycl::free(head_, q);
    if (head8_.w) sycl::free(head8_.w, q); if (head8_.s) sycl::free(head8_.s, q);
    final_norm_ = nullptr; head_ = nullptr; head8_ = Ds41Fp8Mat{}; head_bytes_ = 0;
}

}  // namespace ie
