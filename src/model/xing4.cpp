// src/model/xing4.cpp -- Xing4.0-29B-A4B: load a Q8_0 GGUF onto the cards and run the forward (include/ie/xing4.hpp).
#include "ie/xing4.hpp"

#include "ie/xing4_ops.hpp"
#include "ie/dequant_ref.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ie {

namespace {

inline float xg4h_f16(uint16_t b) { _Float16 h; std::memcpy(&h, &b, 2); return float(h); }
inline float xg4h_bf16(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }

// rows of a tensor as fp32 on the host
bool xg4h_dequant(const GgufTensorInfo* t, size_t row0, size_t n_rows, size_t K, float* out) {
    switch (t->dtype) {
    case DType::kF32: std::memcpy(out, reinterpret_cast<const float*>(t->data) + row0 * K, n_rows * K * 4); return true;
    case DType::kF16: { const uint16_t* p = reinterpret_cast<const uint16_t*>(t->data) + row0 * K;
                        for (size_t i = 0; i < n_rows * K; ++i) out[i] = xg4h_f16(p[i]); return true; }
    case DType::kBF16: { const uint16_t* p = reinterpret_cast<const uint16_t*>(t->data) + row0 * K;
                         for (size_t i = 0; i < n_rows * K; ++i) out[i] = xg4h_bf16(p[i]); return true; }
    case DType::kQ8_0: {
        const size_t nb = K / 32; const uint8_t* p = t->data + row0 * nb * 34;
        for (size_t b = 0; b < n_rows * nb; ++b) {
            uint16_t db; std::memcpy(&db, p + b * 34, 2); const float dd = xg4h_f16(db);
            const int8_t* qv = reinterpret_cast<const int8_t*>(p + b * 34 + 2);
            for (int i = 0; i < 32; ++i) out[b * 32 + i] = dd * float(qv[i]);
        }
        return true;
    }
    default: break;
    }
    // block formats: dequantize whole rows through the reference code (ggml's dequantize_row_*)
    using Fn = void (*)(const void*, size_t, float*);
    Fn fn = nullptr; size_t bs = 0, be = 0;
    switch (t->dtype) {
    case DType::kQ6_K: fn = ref::dequant_q6_K_buffer; bs = 210; be = 256; break;
    case DType::kQ5_K: fn = ref::dequant_q5_K_buffer; bs = 176; be = 256; break;
    case DType::kQ4_K: fn = ref::dequant_q4_K_buffer; bs = 144; be = 256; break;
    case DType::kQ5_0: fn = ref::dequant_q5_0_buffer; bs = 22; be = 32; break;
    case DType::kQ5_1: fn = ref::dequant_q5_1_buffer; bs = 24; be = 32; break;
    case DType::kQ4_0: fn = ref::dequant_q4_0_buffer; bs = 18; be = 32; break;
    case DType::kQ4_1: fn = ref::dequant_q4_1_buffer; bs = 20; be = 32; break;
    default: return false;
    }
    if (K % be) return false;
    fn(t->data + row0 * (K / be) * bs, n_rows * K, out);
    return true;
}

uint32_t xg4h_fmt(DType t, uint32_t k) {
    if (k % 256 == 0) {
        if (t == DType::kQ6_K) return kXg4Q6K;
        if (t == DType::kQ5_K) return kXg4Q5K;
        if (t == DType::kQ4_K) return kXg4Q4K;
    }
    return kXg4Q8;
}

}  // namespace

Xg4W W(const Xg4Mat& m) { Xg4W w; w.qs = m.qs; w.d = m.d; w.qh = m.qh; w.sc = m.sc; w.fmt = m.fmt; return w; }

// ------------------------------------------------------------------------------------------------------------- config
std::string xing4_read_config(const GgufReader& g, Xing4Config& c) {
    const KvRef* a = g.find_kv("general.architecture");
    if (!a || a->as_string() != "xing4_0") return "not a xing4_0 GGUF";
    auto u32 = [&](const char* k, uint32_t& v) { if (const KvRef* r = g.find_kv(std::string("xing4_0.") + k)) v = uint32_t(r->as_uint()); };
    auto f32 = [&](const char* k, float& v) { if (const KvRef* r = g.find_kv(std::string("xing4_0.") + k)) v = float(r->as_float()); };
    uint32_t blocks = 0, nextn = 0;
    u32("block_count", blocks); u32("nextn_predict_layers", nextn);
    c.n_layers = blocks - nextn;
    u32("embedding_length", c.dim); u32("attention.head_count", c.n_heads); u32("attention.q_lora_rank", c.q_lora);
    u32("attention.kv_lora_rank", c.kv_lora); u32("rope.dimension_count", c.rope); u32("feed_forward_length", c.ffn);
    u32("expert_feed_forward_length", c.moe_ffn); u32("expert_count", c.n_experts); u32("expert_used_count", c.top_k);
    u32("expert_shared_count", c.n_shared); u32("leading_dense_block_count", c.n_dense); u32("vocab_size", c.vocab);
    u32("hyper_connection.count", c.hc); u32("hyper_connection.sinkhorn_iterations", c.sinkhorn_iters);
    f32("hyper_connection.epsilon", c.hc_eps); f32("attention.layer_norm_rms_epsilon", c.norm_eps);
    f32("expert_weights_scale", c.routed_scale); f32("rope.freq_base", c.rope_theta); f32("rope.scaling.factor", c.yarn_factor);
    u32("rope.scaling.original_context_length", c.yarn_orig_ctx);
    if (const KvRef* r = g.find_kv("xing4_0.expert_weights_norm")) c.norm_topk = r->as_bool();
    uint32_t kmla = 0, vmla = 0; u32("attention.key_length_mla", kmla); u32("attention.value_length_mla", vmla);
    if (kmla) c.nope = kmla - c.rope;
    if (vmla) c.v_dim = vmla;
    if (const GgufTensorInfo* t = g.find_tensor("token_embd.weight")) c.vocab = uint32_t(t->shape[1]);
    if (c.hc != 4) return "xing4: only 4 residual streams are supported";
    if (c.kv_lora != 512 || c.rope != 64 || c.nope != 128 || c.v_dim != 128 || c.n_heads % 32)
        return "xing4: the MLA kernels need latent 512, rope 64, nope 128, v 128, heads a multiple of 32";
    if (c.dim % 256 || c.n_dense > c.n_layers) return "xing4: unexpected hidden size / dense layer count";
    return {};
}

// --------------------------------------------------------------------------------------------------------- the model
struct Xing4Model::Layer {
    float *attn_norm = nullptr, *ffn_norm = nullptr, *q_a_norm = nullptr, *kv_a_norm = nullptr;
    float *hca_fn = nullptr, *hca_base = nullptr, *hca_scale = nullptr, *hcf_fn = nullptr, *hcf_base = nullptr, *hcf_scale = nullptr;
    Xg4Mat qkv_a, kv_a, q_b, k_b, v_b, o;      // qkv_a = [q_a (768 rows) ; kv_a (576 rows)], or q_a alone and kv_a apart
    bool moe = false;
    Xg4Mat gate, up, down;          // dense FFN, or the shared expert
    Xg4Mat eg, eu, ed;              // the routed experts, [E][N][K]
    float *router = nullptr, *bias = nullptr;
    sycl::half* kv = nullptr;       // the paged pool [pages * 64, 576]
};

struct Xing4Model::Card {
    sycl::queue* q = nullptr;
    std::vector<void*> allocs;
    uint64_t bytes = 0;
    uint8_t* stage = nullptr; size_t stage_cap = 0;             // pinned host staging (no pageable copies: playbook trap)
    float *S = nullptr, *part = nullptr, *post = nullptr, *comb = nullptr, *x32 = nullptr, *qa32 = nullptr, *q32 = nullptr;
    float *kv32 = nullptr, *y = nullptr, *job_w = nullptr, *cs = nullptr, *logits = nullptr;
    sycl::half *x16 = nullptr, *qa16 = nullptr, *qn16 = nullptr, *a16 = nullptr, *ctx16 = nullptr, *o16 = nullptr, *h16 = nullptr;
    float* h32 = nullptr; float* hs = nullptr;   // the expert hidden in fp32 and each row's fp16 scale (xg4_h_pack)
    int32_t *job_e = nullptr, *rinfo = nullptr, *tab = nullptr;    // rinfo: [lane T][pos T][token T][logit rows]
    float *rlog = nullptr, *inv_freq = nullptr, *kv32b = nullptr, *yj = nullptr;
    int32_t *goff = nullptr, *glist = nullptr, *mtiles = nullptr;   // mtiles: the XMX prompt kernels' tile list
    Xg4A ax, aqa, ao, ah;                                       // int8 activations: the normed x, q_a out, attention out, FFN hidden
    Xg4Mat embd;                                                // card 0: token_embd
    float* out_norm = nullptr; Xg4Mat head;                     // the last card
    template <class V> V* dev(size_t n) {
        V* p = sycl::malloc_device<V>(std::max<size_t>(n, 1), *q);
        if (p) { allocs.push_back(p); bytes += n * sizeof(V); }
        return p;
    }
    void put(void* dst, const void* src, size_t n) {            // host -> card through the pinned stage
        for (size_t o = 0; o < n; o += stage_cap) {
            const size_t k = std::min(stage_cap, n - o);
            std::memcpy(stage, static_cast<const uint8_t*>(src) + o, k);
            q->memcpy(static_cast<uint8_t*>(dst) + o, stage, k).wait();
        }
    }
    void get(void* dst, const void* src, size_t n) {            // card -> host through the pinned stage
        for (size_t o = 0; o < n; o += stage_cap) {
            const size_t k = std::min(stage_cap, n - o);
            q->memcpy(stage, static_cast<const uint8_t*>(src) + o, k).wait();
            std::memcpy(static_cast<uint8_t*>(dst) + o, stage, k);
        }
    }
};

Xing4Model::Xing4Model() = default;
Xing4Model::~Xing4Model() {
    for (auto& c : cards_) {
        if (!c || !c->q) continue;
        try { c->q->wait(); } catch (...) {}
        for (void* p : c->allocs) sycl::free(p, *c->q);
        if (c->stage) sycl::free(c->stage, *c->q);
    }
}

void Xing4Model::prof(const char* name, const sycl::event& e) { if (prof_on_) prof_ev_.emplace_back(name, e); }

std::string Xing4Model::profile_report() {
    if (!prof_on_) return "profiling is off (IE_XG4_PROFILE=1 and profiling queues)\n";
    std::vector<std::pair<std::string, std::pair<double, uint32_t>>> agg;
    double total = 0.0;
    for (auto& pe : prof_ev_) {
        pe.second.wait();
        const double us = double(pe.second.get_profiling_info<sycl::info::event_profiling::command_end>() -
                                 pe.second.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1000.0;
        total += us;
        auto it = std::find_if(agg.begin(), agg.end(), [&](auto& x) { return x.first == pe.first; });
        if (it == agg.end()) agg.push_back({pe.first, {us, 1u}}); else { it->second.first += us; ++it->second.second; }
    }
    prof_ev_.clear();
    std::sort(agg.begin(), agg.end(), [](auto& x, auto& y) { return x.second.first > y.second.first; });
    std::string r; char buf[160];
    std::snprintf(buf, sizeof buf, "kernels: %.1f us in total\n", total); r += buf;
    for (auto& [n, v] : agg) { std::snprintf(buf, sizeof buf, "  %-18s %9.1f us  %5.1f %%  x%u\n", n.c_str(), v.first, 100.0 * v.first / total, v.second); r += buf; }
    return r;
}
void Xing4Model::profile_reset() { prof_ev_.clear(); }

uint64_t Xing4Model::vram_bytes(size_t card) const { return card < cards_.size() ? cards_[card]->bytes : 0; }

Xg4Mat Xing4Model::q8_mat(Card& c, const GgufTensorInfo* t, uint32_t n_rows, uint32_t k, std::string& err) {
    Xg4Mat m;
    if (!err.empty()) return m;
    if (!t) { err = "missing tensor"; return m; }
    if (!alloc_mat(c, m, xg4h_fmt(t->dtype, k), n_rows, k)) { err = "out of device memory at " + std::string(t->name); return m; }
    q8_fill(c, t, m, 0, n_rows, err);
    return m;
}

bool Xing4Model::alloc_mat(Card& c, Xg4Mat& m, uint32_t fmt, uint32_t n_rows, uint32_t k) {
    const size_t N = n_rows;
    m.N = n_rows; m.K = k; m.fmt = fmt;
    switch (fmt) {
    default:      m.qs = c.dev<int8_t>(N * k); m.d = c.dev<uint16_t>(N * (k / 32)); return m.qs && m.d;
    case kXg4Q6K: m.qs = c.dev<int8_t>(N * k / 2); m.qh = c.dev<uint8_t>(N * k / 4); m.sc = c.dev<uint8_t>(N * k / 16);
                  m.d = c.dev<uint16_t>(N * (k / 256)); return m.qs && m.qh && m.sc && m.d;
    case kXg4Q5K: m.qh = c.dev<uint8_t>(N * k / 8); [[fallthrough]];
    case kXg4Q4K: m.qs = c.dev<int8_t>(N * k / 2); m.sc = c.dev<uint8_t>(N * (k / 32) * 2);
                  m.d = c.dev<uint16_t>(N * (k / 256) * 2); return m.qs && m.sc && m.d && (fmt != kXg4Q5K || m.qh);
    }
}

void Xing4Model::q8_fill(Card& c, const GgufTensorInfo* t, Xg4Mat& dst, uint32_t row0, uint32_t n_rows, std::string& err) {
    if (!err.empty()) return;
    if (!t) { err = "missing tensor"; return; }
    const uint32_t k = dst.K;
    uint64_t elems = 1; for (uint32_t i = 0; i < t->n_dims; ++i) elems *= t->shape[i];
    if (t->shape[0] != k || elems != uint64_t(n_rows) * k) { err = std::string("unexpected shape of ") + std::string(t->name); return; }
    const size_t nb = k / 32, N = n_rows;
    if (dst.fmt != kXg4Q8) {                                    // K-quant blocks -> the planes (rows row0 ..)
        if (xg4h_fmt(t->dtype, k) != dst.fmt) { err = std::string("mixed formats in one matrix at ") + std::string(t->name); return; }
        const uint32_t f = dst.fmt, nbk = k / 256;
        const size_t bs = f == kXg4Q6K ? 210 : f == kXg4Q5K ? 176 : 144;
        const size_t qs_r = k / 2, qh_r = f == kXg4Q6K ? k / 4 : f == kXg4Q5K ? k / 8 : 0, sc_r = f == kXg4Q6K ? k / 16 : (k / 32) * 2,
                     d_r = f == kXg4Q6K ? nbk : nbk * 2;
        const size_t row_bytes = qs_r + qh_r + sc_r + d_r * 2;
        const size_t rows_per = std::max<size_t>(1, c.stage_cap / row_bytes);
        std::vector<uint8_t> pq, ph, ps; std::vector<uint16_t> pd;
        for (size_t r0 = 0; r0 < N; r0 += rows_per) {
            const size_t nr = std::min(rows_per, N - r0);
            pq.resize(nr * qs_r); ph.resize(nr * qh_r); ps.resize(nr * sc_r); pd.resize(nr * d_r);
#pragma omp parallel for schedule(static)
            for (long long rb = 0; rb < (long long)(nr * nbk); ++rb) {
                const size_t r = size_t(rb) / nbk, b = size_t(rb) % nbk;
                const uint8_t* src = t->data + ((r0 + r) * nbk + b) * bs;
                if (f == kXg4Q6K) {
                    std::memcpy(&pq[r * qs_r + b * 128], src, 128);
                    std::memcpy(&ph[r * qh_r + b * 64], src + 128, 64);
                    std::memcpy(&ps[r * sc_r + b * 16], src + 192, 16);
                    std::memcpy(&pd[r * d_r + b], src + 208, 2);
                } else {
                    std::memcpy(&pd[r * d_r + b * 2], src, 4);
                    for (int j = 0; j < 8; ++j) ref::get_scale_min_k4(j, src + 4, ps[r * sc_r + (b * 8 + j) * 2], ps[r * sc_r + (b * 8 + j) * 2 + 1]);
                    if (f == kXg4Q5K) { std::memcpy(&ph[r * qh_r + b * 32], src + 16, 32); std::memcpy(&pq[r * qs_r + b * 128], src + 48, 128); }
                    else std::memcpy(&pq[r * qs_r + b * 128], src + 16, 128);
                }
            }
            const size_t R = row0 + r0;
            c.put(reinterpret_cast<uint8_t*>(dst.qs) + R * qs_r, pq.data(), pq.size());
            if (qh_r && dst.qh) c.put(dst.qh + R * qh_r, ph.data(), ph.size());
            c.put(dst.sc + R * sc_r, ps.data(), ps.size());
            c.put(dst.d + R * d_r, pd.data(), pd.size() * 2);
        }
        return;
    }
    Xg4Mat m = dst; m.qs += size_t(row0) * k; m.d += size_t(row0) * nb;
    if (t->dtype == DType::kQ8_0) {                             // repack the 34-byte blocks into the two planes, in pieces
        const size_t rows_per = std::max<size_t>(1, c.stage_cap / (k + nb * 2));
        std::vector<int8_t> qs; std::vector<uint16_t> dd;
        for (size_t r0 = 0; r0 < N; r0 += rows_per) {
            const size_t nr = std::min(rows_per, N - r0);
            qs.resize(nr * k); dd.resize(nr * nb);
            const uint8_t* src = t->data + r0 * nb * 34;
#pragma omp parallel for schedule(static)
            for (long long b = 0; b < (long long)(nr * nb); ++b) {
                std::memcpy(&dd[size_t(b)], src + size_t(b) * 34, 2);
                std::memcpy(&qs[size_t(b) * 32], src + size_t(b) * 34 + 2, 32);
            }
            c.put(m.qs + r0 * k, qs.data(), nr * k);
            c.put(m.d + r0 * nb, dd.data(), nr * nb * 2);
        }
        return;
    }
    // other types: dequantize and round to Q8_0 here (amax / 127 per 32)
    std::vector<float> f(static_cast<size_t>(k));
    std::vector<int8_t> qs(N * k); std::vector<uint16_t> dd(N * nb);
    for (size_t r = 0; r < N; ++r) {
        if (!xg4h_dequant(t, r, 1, k, f.data())) { err = "unsupported dtype of " + std::string(t->name); return; }
        for (size_t b = 0; b < nb; ++b) {
            float amax = 0.f; for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(f[b * 32 + i]));
            const _Float16 dh = _Float16(amax / 127.f); const float d = float(dh), id = d > 0 ? 1.f / d : 0.f;
            std::memcpy(&dd[r * nb + b], &dh, 2);
            for (int i = 0; i < 32; ++i) qs[r * k + b * 32 + i] = int8_t(std::lround(std::clamp(f[b * 32 + i] * id, -127.f, 127.f)));
        }
    }
    c.put(m.qs, qs.data(), qs.size()); c.put(m.d, dd.data(), dd.size() * 2);
}

float* Xing4Model::f32_vec(Card& c, const GgufTensorInfo* t, size_t n, std::string& err) {
    if (!err.empty()) return nullptr;
    if (!t) { err = "missing tensor"; return nullptr; }
    uint64_t elems = 1; for (uint32_t i = 0; i < t->n_dims; ++i) elems *= t->shape[i];
    if (elems != n) { err = std::string("unexpected size of ") + std::string(t->name); return nullptr; }
    const size_t K = t->shape[0];
    std::vector<float> h(n);
    if (!xg4h_dequant(t, 0, n / K, K, h.data())) { err = "unsupported dtype of " + std::string(t->name); return nullptr; }
    float* p = c.dev<float>(n);
    if (!p) { err = "out of device memory"; return nullptr; }
    c.put(p, h.data(), n * 4);
    return p;
}

std::string Xing4Model::upload_layer(Card& c, uint32_t L) {
    const Xing4Config& k = cfg_;
    auto t = [&](const char* s) { return g_.find_tensor("blk." + std::to_string(L) + "." + s); };
    Layer& d = *layers_[L];
    std::string err;
    const uint32_t H = k.dim, HH = k.hc * H, NM = (2 + k.hc) * k.hc;
    d.attn_norm = f32_vec(c, t("attn_norm.weight"), H, err);
    d.ffn_norm  = f32_vec(c, t("ffn_norm.weight"), H, err);
    d.q_a_norm  = f32_vec(c, t("attn_q_a_norm.weight"), k.q_lora, err);
    d.kv_a_norm = f32_vec(c, t("attn_kv_a_norm.weight"), k.kv_lora, err);
    d.hca_fn = f32_vec(c, t("hc_attn_fn.weight"), size_t(NM) * HH, err);
    d.hca_base = f32_vec(c, t("hc_attn_base.weight"), NM, err);
    d.hca_scale = f32_vec(c, t("hc_attn_scale.weight"), 3, err);
    d.hcf_fn = f32_vec(c, t("hc_ffn_fn.weight"), size_t(NM) * HH, err);
    d.hcf_base = f32_vec(c, t("hc_ffn_base.weight"), NM, err);
    d.hcf_scale = f32_vec(c, t("hc_ffn_scale.weight"), 3, err);
    {   // q_a and kv_a as one matrix when they share a format, else two
        const GgufTensorInfo *ta = t("attn_q_a.weight"), *tk = t("attn_kv_a_mqa.weight");
        if (!ta || !tk) err = "missing attn_q_a / attn_kv_a_mqa";
        else if (xg4h_fmt(ta->dtype, H) == xg4h_fmt(tk->dtype, H)) {
            if (!alloc_mat(c, d.qkv_a, xg4h_fmt(ta->dtype, H), k.q_lora + k.lat_w(), H)) err = "out of device memory";
            q8_fill(c, ta, d.qkv_a, 0, k.q_lora, err);
            q8_fill(c, tk, d.qkv_a, k.q_lora, k.lat_w(), err);
        } else {
            d.qkv_a = q8_mat(c, ta, k.q_lora, H, err);
            d.kv_a = q8_mat(c, tk, k.lat_w(), H, err);
        }
    }
    d.q_b  = q8_mat(c, t("attn_q_b.weight"), k.n_heads * k.qk_dim(), k.q_lora, err);
    d.k_b  = q8_mat(c, t("attn_k_b.weight"), k.n_heads * k.kv_lora, k.nope, err);
    d.v_b  = q8_mat(c, t("attn_v_b.weight"), k.n_heads * k.v_dim, k.kv_lora, err);
    d.o    = q8_mat(c, t("attn_output.weight"), H, k.n_heads * k.v_dim, err);
    d.moe = L >= k.n_dense;
    if (!d.moe) {
        d.gate = q8_mat(c, t("ffn_gate.weight"), k.ffn, H, err);
        d.up   = q8_mat(c, t("ffn_up.weight"), k.ffn, H, err);
        d.down = q8_mat(c, t("ffn_down.weight"), H, k.ffn, err);
    } else {
        const uint32_t S = k.moe_ffn * k.n_shared;
        d.router = f32_vec(c, t("ffn_gate_inp.weight"), size_t(k.n_experts) * H, err);
        if (const GgufTensorInfo* b = t("exp_probs_b.bias")) d.bias = f32_vec(c, b, k.n_experts, err);
        d.gate = q8_mat(c, t("ffn_gate_shexp.weight"), S, H, err);
        d.up   = q8_mat(c, t("ffn_up_shexp.weight"), S, H, err);
        d.down = q8_mat(c, t("ffn_down_shexp.weight"), H, S, err);
        d.eg = q8_mat(c, t("ffn_gate_exps.weight"), k.n_experts * k.moe_ffn, H, err);
        d.eu = q8_mat(c, t("ffn_up_exps.weight"), k.n_experts * k.moe_ffn, H, err);
        d.ed = q8_mat(c, t("ffn_down_exps.weight"), k.n_experts * H, k.moe_ffn, err);
        if (k.n_shared != 1) err = err.empty() ? "xing4: one shared expert expected" : err;
    }
    if (!err.empty()) return "layer " + std::to_string(L) + ": " + err;
    d.kv = c.dev<sycl::half>(size_t(n_pages_) * 64 * k.lat_w());
    if (!d.kv) return "layer " + std::to_string(L) + ": no room for the KV cache";
    c.q->memset(d.kv, 0, size_t(n_pages_) * 64 * k.lat_w() * 2).wait();   // (masked keys of a page must be finite)
    return {};
}

std::string Xing4Model::load(const std::string& path, const std::vector<sycl::queue*>& qs, const Xing4Options& o) {
    if (qs.empty()) return "no queues";
    opt_ = o;
    { const char* v = std::getenv("IE_XG4_PROFILE"); prof_on_ = v && *v == '1'; }
    { const char* v = std::getenv("IE_XG4_INTDOT"); intdot_ = !(v && *v == '0'); }
    if (auto e = g_.open(path); !e.empty()) return e;
    if (auto e = xing4_read_config(g_, cfg_); !e.empty()) return e;
    const Xing4Config& k = cfg_;
    const uint32_t nc = uint32_t(std::min<size_t>(qs.size(), 2));
    const uint32_t split = nc == 1 ? k.n_layers : (o.split ? o.split : k.n_layers / 2);
    for (uint32_t i = 0; i < nc; ++i) {
        auto c = std::make_unique<Card>(); c->q = qs[i];
        c->stage_cap = size_t(64) << 20;
        c->stage = static_cast<uint8_t*>(sycl::malloc_host(c->stage_cap, *c->q));
        if (!c->stage) return "no pinned staging buffer";
        cards_.push_back(std::move(c));
    }
    layers_.resize(k.n_layers);
    for (auto& l : layers_) l = std::make_unique<Layer>();
    layer_card_.resize(k.n_layers);
    for (uint32_t L = 0; L < k.n_layers; ++L) layer_card_[L] = L < split ? 0 : nc - 1;
    // the queues share one context (the caller enabled peer access): card 0 PUSHES the streams into card 1 (a write to the
    // peer; never a read across, which this bridge corrupts -- docs P2P notes) and card 1's queue waits on that copy
    // OFF unless IE_XG4_PUSH=1 -- BROKEN as of 2026-11-20: ordered by a barrier or a depends_on kernel on card 1 the logits are
    // garbage (KL 10.2 vs 0.0029, decode "116 tok/s" = card 1 not waiting); with host waits it is right and no faster than the
    // host copy (66.7 tok/s). The shared context alone is correct (KL 0.00288).
    p2p_ = nc == 2 && qs[0]->get_context() == qs[1]->get_context() && std::getenv("IE_XG4_PUSH") && *std::getenv("IE_XG4_PUSH") == '1';
    if (p2p_) std::fprintf(stderr, "[xing4] the streams move card to card as a push (no host round trip)\n");
    // the paged cache
    maxp_ = (o.max_ctx + 63) / 64; n_pages_ = (o.kv_tokens + 63) / 64;
    lane_len_.assign(o.max_lanes, 0); lane_pages_.assign(o.max_lanes, {});
    free_pages_.clear(); for (int32_t p = int32_t(n_pages_) - 1; p >= 0; --p) free_pages_.push_back(p);
    page_ref_.assign(n_pages_, 0);
    tab_host_.assign(size_t(o.max_lanes) * maxp_, 0); tab_dirty_ = true;
    // rows of one call
    const size_t T = o.max_tokens, H = k.dim, NH = k.n_heads, J = size_t(k.top_k + 1);
    for (auto& cp : cards_) {
        Card& c = *cp;
        c.S = c.dev<float>(T * k.hc * H); c.part = c.dev<float>(T * 25 * kXg4HcParts); c.post = c.dev<float>(T * 4); c.comb = c.dev<float>(T * 16);
        c.x16 = c.dev<sycl::half>(T * H); c.x32 = c.dev<float>(T * H); c.qa32 = c.dev<float>(T * k.q_lora); c.qa16 = c.dev<sycl::half>(T * k.q_lora);
        c.q32 = c.dev<float>(T * NH * k.qk_dim()); c.qn16 = c.dev<sycl::half>(T * NH * k.nope); c.a16 = c.dev<sycl::half>(T * NH * k.lat_w());
        c.kv32 = c.dev<float>(T * k.lat_w()); c.ctx16 = c.dev<sycl::half>(T * NH * k.lat_w()); c.o16 = c.dev<sycl::half>(T * NH * k.v_dim);
        c.y = c.dev<float>(T * H); c.job_e = c.dev<int32_t>(T * J); c.job_w = c.dev<float>(T * J); c.rlog = c.dev<float>(T * k.n_experts);
        c.h16 = c.dev<sycl::half>(T * std::max<size_t>(J * k.moe_ffn, k.ffn));
        c.h32 = c.dev<float>(T * std::max<size_t>(J * k.moe_ffn, k.ffn)); c.hs = c.dev<float>(T * std::max<size_t>(J, 1)); c.cs = c.dev<float>(T * k.rope);
        c.rinfo = c.dev<int32_t>(T * 6 + 2); c.tab = c.dev<int32_t>(size_t(o.max_lanes) * maxp_); c.inv_freq = c.dev<float>(k.rope / 2);
        c.kv32b = c.dev<float>(T * (k.q_lora + k.lat_w()));
        c.yj = c.dev<float>(T * J * H); c.goff = c.dev<int32_t>(k.n_experts + 2); c.glist = c.dev<int32_t>(T * J);
        c.mtiles = c.dev<int32_t>(1 + 2 * (T * J / 32 + k.n_experts + 2));
        auto mk = [&](size_t rows, size_t K) { Xg4A a; a.q = c.dev<int8_t>(rows * K); a.d = c.dev<float>(rows * K / 32); a.s = c.dev<float>(rows * K / 32); return a; };
        if (intdot_) {
            c.ax = mk(T, H); c.aqa = mk(T, k.q_lora); c.ao = mk(T, size_t(NH) * k.v_dim);
            c.ah = mk(T * std::max<size_t>(J, 1), std::max<size_t>(k.moe_ffn, k.ffn));
            if (!c.ax.q || !c.aqa.q || !c.ao.q || !c.ah.q) return "no room for the int8 activations";
        }
        if (!c.S || !c.part || !c.x16 || !c.a16 || !c.ctx16 || !c.h16 || !c.h32 || !c.hs || !c.rinfo || !c.tab || !c.kv32b || !c.yj || !c.goff || !c.glist || !c.mtiles) return "no room for the activations";
    }
    for (uint32_t L = 0; L < k.n_layers; ++L) {
        if (auto e = upload_layer(*cards_[layer_card_[L]], L); !e.empty()) return e;
        if (L % 10 == 9) std::fprintf(stderr, "[xing4] %u / %u layers on the cards\n", L + 1, k.n_layers);
    }
    Card& last = *cards_.back();
    std::string err;
    last.out_norm = f32_vec(last, g_.find_tensor("output_norm.weight"), H, err);
    const GgufTensorInfo* out = g_.find_tensor("output.weight");
    if (!out) out = g_.find_tensor("token_embd.weight");
    last.head = q8_mat(last, out, k.vocab, k.dim, err);
    last.logits = last.dev<float>(T * k.vocab);
    if (!err.empty() || !last.logits) return "head: " + (err.empty() ? std::string("no room for the logits") : err);
    const GgufTensorInfo* emb = g_.find_tensor("token_embd.weight");
    if (!emb || emb->shape[0] != k.dim) return "token_embd missing or of the wrong width";
    cards_[0]->embd = q8_mat(*cards_[0], emb, k.vocab, k.dim, err);
    if (!err.empty()) return "token_embd: " + err;
    // YaRN (transformers _compute_yarn_parameters, truncate = true): the frequencies and the softmax scale
    const double base = k.rope_theta, dim = k.rope, f = k.yarn_factor, omax = k.yarn_orig_ctx;
    auto corr = [&](double rot) { return (dim * std::log(omax / (rot * 2.0 * M_PI))) / (2.0 * std::log(base)); };
    double low = std::floor(corr(k.yarn_beta_fast)), high = std::ceil(corr(k.yarn_beta_slow));
    low = std::max(low, 0.0); high = std::min(high, dim - 1);
    if (low == high) high += 0.001;
    rope_inv_freq_.resize(k.rope / 2);
    for (uint32_t i = 0; i < k.rope / 2; ++i) {
        const float pos_freq = std::pow(float(base), float(2 * i) / float(dim));
        const float extra = 1.0f / pos_freq, inter = 1.0f / (float(f) * pos_freq);
        const float ramp = std::clamp(float((double(i) - low) / (high - low)), 0.f, 1.f);
        const float ext_factor = f > 1.0 ? 1.0f - ramp : 1.0f;
        rope_inv_freq_[i] = inter * (1.0f - ext_factor) + extra * ext_factor;
    }
    const float mscale = f > 1.0 ? float(0.1 * k.yarn_mscale_all * std::log(f) + 1.0) : 1.0f;
    attn_scale_ = mscale * mscale / std::sqrt(float(k.qk_dim()));
    for (auto& c : cards_) c->put(c->inv_freq, rope_inv_freq_.data(), rope_inv_freq_.size() * 4);
    for (size_t i = 0; i < cards_.size(); ++i)
        std::fprintf(stderr, "[xing4] card %zu: %.2f GiB (layers %u..%u)\n", i, double(cards_[i]->bytes) / 1073741824.0,
                     i == 0 ? 0u : split, i == 0 ? split - 1 : k.n_layers - 1);
    std::fprintf(stderr, "[xing4] paged cache: %u keys (%u pages of 64), %u lanes, up to %u keys a lane\n", n_pages_ * 64, n_pages_, o.max_lanes, maxp_ * 64);
    return {};
}

int32_t Xing4Model::page_take() {
    if (free_pages_.empty()) return -1;
    const int32_t p = free_pages_.back(); free_pages_.pop_back();
    page_ref_[size_t(p)] = 1;
    return p;
}
void Xing4Model::page_drop(int32_t p) {
    if (page_ref_[size_t(p)] > 0 && --page_ref_[size_t(p)] == 0) free_pages_.push_back(p);
}
void Xing4Model::page_copy(int32_t to, int32_t from, uint32_t rows) {
    const size_t W = cfg_.lat_w();
    for (uint32_t L = 0; L < cfg_.n_layers; ++L) {
        sycl::half* kv = layers_[L]->kv;
        cards_[layer_card_[L]]->q->memcpy(kv + size_t(to) * 64 * W, kv + size_t(from) * 64 * W, size_t(rows) * W * sizeof(sycl::half));
    }
}

void Xing4Model::lane_truncate(uint32_t lane, uint32_t n) {
    if (lane >= lane_len_.size()) return;
    n = std::min(n, lane_len_[lane]);
    auto& pg = lane_pages_[lane];
    const size_t keep = (size_t(n) + 63) / 64;
    while (pg.size() > keep) { page_drop(pg.back()); pg.pop_back(); }
    lane_len_[lane] = n;
    // the lane writes on into its last page: a page another lane still holds is copied first (copy on write)
    if (n % 64 && !pg.empty() && page_ref_[size_t(pg.back())] > 1) {
        const int32_t np = page_take();
        if (np < 0) { page_drop(pg.back()); pg.pop_back(); lane_len_[lane] = uint32_t(pg.size()) * 64; return; }   // (no room: drop to the page edge)
        page_copy(np, pg.back(), n % 64);
        page_drop(pg.back());
        pg.back() = np;
        tab_host_[size_t(lane) * maxp_ + pg.size() - 1] = np; tab_dirty_ = true;
    }
}

std::string Xing4Model::lane_share(uint32_t dst, uint32_t src, uint32_t n) {
    if (dst >= lane_len_.size() || src >= lane_len_.size() || dst == src) return "lane_share: bad lanes";
    n = std::min(n, lane_len_[src]);
    lane_truncate(dst, 0);
    auto& d = lane_pages_[dst];
    const auto& s = lane_pages_[src];
    const uint32_t full = n / 64, rem = n % 64;
    for (uint32_t i = 0; i < full; ++i) {
        d.push_back(s[i]); ++page_ref_[size_t(s[i])];
        tab_host_[size_t(dst) * maxp_ + i] = s[i];
    }
    if (rem) {
        const int32_t np = page_take();
        if (np < 0) { lane_len_[dst] = full * 64; tab_dirty_ = true; return {}; }
        page_copy(np, s[full], rem);
        d.push_back(np);
        tab_host_[size_t(dst) * maxp_ + full] = np;
    }
    lane_len_[dst] = n;
    tab_dirty_ = true;
    return {};
}

std::string Xing4Model::run_layer(Card& c, uint32_t L, uint32_t T, uint32_t max_keys) {
    const Xing4Config& k = cfg_;
    const Layer& d = *layers_[L];
    sycl::queue& q = *c.q;
    const uint32_t H = k.dim, HH = k.hc * H, NH = k.n_heads, LW = k.lat_w(), QKV = k.q_lora + LW;
    const int32_t *rlane = c.rinfo, *rpos = c.rinfo + T;
    const bool I = intdot_;
    // int8 activations only where a consumer is a K-quant matrix (Q8_0 keeps the fp16 activations: no faster with int8)
    auto kq = [](const Xg4Mat& m) { return m.qs && m.fmt != kXg4Q8; };
    const bool attn_k = kq(d.qkv_a) || kq(d.kv_a), ffn_k = kq(d.eg) || kq(d.eu) || kq(d.gate) || kq(d.up),
               dn_k = kq(d.ed) || kq(d.down);
    const Xg4A axa = I && attn_k ? c.ax : Xg4A{}, axf = I && ffn_k ? c.ax : Xg4A{}, aqa = I && kq(d.q_b) ? c.aqa : Xg4A{},
               ao = I && kq(d.o) ? c.ao : Xg4A{}, ah = I && dn_k ? c.ah : Xg4A{};
    // IE_XG4_NANCHECK=1 (debug): after each step, the first row of a buffer that is not finite, named
    static const bool nanchk = [] { const char* v = std::getenv("IE_XG4_NANCHECK"); return v && *v == '1'; }();
    auto chk = [&](const char* what, const void* buf, size_t n, bool half) {
        if (!nanchk) return;
        q.wait();
        std::vector<float> h(n);
        if (half) { std::vector<sycl::half> t(n); q.memcpy(t.data(), buf, n * 2).wait(); for (size_t i = 0; i < n; ++i) h[i] = float(t[i]); }
        else q.memcpy(h.data(), buf, n * 4).wait();
        float mx = 0; size_t bad = n;
        for (size_t i = 0; i < n; ++i) { if (!std::isfinite(h[i])) { bad = i; break; } mx = std::max(mx, std::fabs(h[i])); }
        if (bad < n) std::fprintf(stderr, "[nancheck] layer %u %s: element %zu not finite (absmax before %.3g)\n", L, what, bad, mx);
        else if (mx > 3e4f) std::fprintf(stderr, "[nancheck] layer %u %s: absmax %.3g\n", L, what, mx);
    };
    // attention half
    prof("hc_parts", xg4_hc_parts(q, c.S, d.hca_fn, c.part, T, HH));
    prof("hc_gate", xg4_hc_gate(q, c.S, c.part, d.hca_base, d.hca_scale, d.attn_norm, c.post, c.comb, c.x16, nullptr, T, H, k.norm_eps, k.hc_eps,
                k.clamp_lo, k.clamp_hi, k.sinkhorn_iters, axa));
    chk("attn x16", c.x16, size_t(T) * H, true);
    if (!d.kv_a) prof("gemv qkv_a", xg4_gemv(q, c.x16, H, W(d.qkv_a), c.kv32b, nullptr, QKV, QKV, H, T, 1, 0, 0, axa));
    else {
        prof("gemv qkv_a", xg4_gemv(q, c.x16, H, W(d.qkv_a), c.kv32b, nullptr, QKV, k.q_lora, H, T, 1, 0, 0, axa));
        prof("gemv qkv_a", xg4_gemv(q, c.x16, H, W(d.kv_a), c.kv32b + k.q_lora, nullptr, QKV, LW, H, T, 1, 0, 0, axa));
    }
    prof("rmsnorm", xg4_rmsnorm16(q, c.kv32b, QKV, d.q_a_norm, c.qa16, k.q_lora, T, k.q_lora, k.norm_eps, aqa));
    prof("gemv q_b", xg4_gemv(q, c.qa16, k.q_lora, W(d.q_b), c.q32, nullptr, NH * k.qk_dim(), NH * k.qk_dim(), k.q_lora, T, 1, 0, 0, aqa));
    prof("kv_append", xg4_kv_append(q, c.kv32b + k.q_lora, QKV, d.kv_a_norm, c.cs, d.kv, c.tab, maxp_, rlane, rpos, T, k.norm_eps));
    prof("q_split", xg4_q_split(q, c.q32, c.cs, c.qn16, c.a16, T, NH));
    chk("q32", c.q32, size_t(T) * NH * k.qk_dim(), false);
    // q_lat[t, h] = K_b[h] (512 x 128) . q_nope[t, h] -> a16 columns 0..511
    prof("gemv k_b absorb", xg4_gemv(q, c.qn16, NH * k.nope, W(d.k_b), nullptr, c.a16, NH * LW, k.kv_lora, k.nope, T, NH, k.nope, LW));
    if (T <= 32) prof("attn", xg4_attn_decode(q, c.a16, d.kv, c.tab, maxp_, rlane, rpos, c.ctx16, T, NH, attn_scale_, max_keys));
    else {
        static const bool old = [] { const char* v = std::getenv("IE_XG4_ATTN_OLD"); return v && *v == '1'; }();
        if (old) prof("attn", xg4_attn_rows(q, c.a16, d.kv, c.tab, maxp_, rlane, rpos, c.ctx16, T, NH, attn_scale_));
        else prof("attn", xg4_attn_rows4(q, c.a16, d.kv, c.tab, maxp_, rlane, rpos, c.rinfo + agrp_off_, n_agrp_, c.ctx16, NH, attn_scale_));
    }
    chk("a16", c.a16, size_t(T) * NH * LW, true); chk("ctx16", c.ctx16, size_t(T) * NH * LW, true);
    // out[t, h] = V_b[h] (128 x 512) . ctx[t, h]
    prof("gemv v_b absorb", xg4_gemv(q, c.ctx16, NH * LW, W(d.v_b), nullptr, c.o16, NH * k.v_dim, k.v_dim, k.kv_lora, T, NH, LW, k.v_dim));
    if (ao.q) prof("act_q8", xg4_act_q8(q, c.o16, NH * k.v_dim, NH * k.v_dim, T, ao));
    prof("gemv o", xg4_gemv(q, c.o16, NH * k.v_dim, W(d.o), c.y, nullptr, H, H, NH * k.v_dim, T, 1, 0, 0, ao));
    chk("o16", c.o16, size_t(T) * NH * k.v_dim, true); chk("attn y", c.y, size_t(T) * H, false);
    prof("hc_post", xg4_hc_post(q, c.S, c.post, c.comb, c.y, T, H));
    chk("S after attn", c.S, size_t(T) * HH, false);
    // FFN half
    prof("hc_parts", xg4_hc_parts(q, c.S, d.hcf_fn, c.part, T, HH));
    prof("hc_gate", xg4_hc_gate(q, c.S, c.part, d.hcf_base, d.hcf_scale, d.ffn_norm, c.post, c.comb, c.x16, c.x32, T, H, k.norm_eps, k.hc_eps,
                k.clamp_lo, k.clamp_hi, k.sinkhorn_iters, axf));
    if (T >= 2) {                                               // batched rows: expert-grouped
        const uint32_t J = d.moe ? T * (k.top_k + 1) : T, per = d.moe ? k.top_k + 1 : 1;
        if (d.moe) {
            prof("router", xg4_router(q, c.x32, d.router, d.bias, c.job_e, c.job_w, T, H, k.n_experts, k.top_k, k.norm_topk, k.routed_scale, c.rlog));
            prof("moe_group", xg4_moe_group(q, c.job_e, J, k.n_experts, c.goff, c.glist));
        }
        const uint32_t NF = d.moe ? k.moe_ffn : k.ffn;
        // prompt chunks with Q8_0 weights: the XMX tile kernels (32 jobs x 64 outputs a work-group, weights dequantized once a
        // tile into local memory); IE_XG4_XMX_MOE=0: the dot-product kernels
        static const bool xmx_on = !(std::getenv("IE_XG4_XMX_MOE") && *std::getenv("IE_XG4_XMX_MOE") == '0');
        auto q8 = [](const Xg4Mat& m) { return !m.qs || m.fmt == kXg4Q8; };
        const bool xmx = xmx_on && T >= 64 && q8(d.eg) && q8(d.eu) && q8(d.ed) && d.gate.fmt == kXg4Q8 && d.up.fmt == kXg4Q8 &&
                         d.down.fmt == kXg4Q8 && NF % 64 == 0 && H % 128 == 0;
        const uint32_t max_tiles = J / 32 + k.n_experts + 2;
        if (xmx) {
            if (d.moe) prof("moe_tiles", xg4_moe_tiles(q, c.goff, k.n_experts, c.mtiles, max_tiles));
            prof("moe_gate_up", xg4_moe_gate_up_xmx(q, c.x16, d.eg.qs, d.eg.d, d.eu.qs, d.eu.d, d.gate.qs, d.gate.d, d.up.qs, d.up.d,
                                                    c.goff, c.glist, c.mtiles, max_tiles, c.h32, J, per, NF, H, k.n_experts, !d.moe));
        } else
            prof("moe_gate_up", xg4_moe_gate_up_grp(q, c.x16, W(d.eg), W(d.eu), W(d.gate), W(d.up), c.goff, c.glist,
                                                    c.h32, J, per, NF, H, k.n_experts, !d.moe, axf));
        prof("h_pack", xg4_h_pack(q, c.h32, c.h16, c.hs, J, NF));
        if (ah.q) prof("act_q8", xg4_act_q8(q, c.h16, NF, NF, J, ah));
        if (xmx) prof("moe_down", xg4_moe_down_xmx(q, c.h16, c.hs, d.ed.qs, d.ed.d, d.down.qs, d.down.d, c.goff, c.glist, c.mtiles, max_tiles,
                                                   c.job_w, d.moe ? c.yj : c.y, J, H, NF, k.n_experts, !d.moe));
        else prof("moe_down", xg4_moe_down_grp(q, c.h16, c.hs, W(d.ed), W(d.down), c.goff, c.glist, c.job_w, d.moe ? c.yj : c.y, J, H,
                                          NF, k.n_experts, !d.moe, ah));
        if (d.moe) prof("moe_sum", xg4_moe_sum(q, c.yj, c.y, T, per, H));
    } else if (d.moe) {
        const uint32_t J = k.top_k + 1;
        prof("router", xg4_router(q, c.x32, d.router, d.bias, c.job_e, c.job_w, T, H, k.n_experts, k.top_k, k.norm_topk, k.routed_scale, c.rlog));
        prof("moe_gate_up", xg4_moe_gate_up(q, c.x16, W(d.eg), W(d.eu), W(d.gate), W(d.up), c.job_e, c.h32, T, J, k.moe_ffn, H, axf));
        prof("h_pack", xg4_h_pack(q, c.h32, c.h16, c.hs, T * J, k.moe_ffn));
        if (ah.q) prof("act_q8", xg4_act_q8(q, c.h16, k.moe_ffn, k.moe_ffn, T * J, ah));
        prof("moe_down", xg4_moe_down(q, c.h16, c.hs, W(d.ed), W(d.down), c.job_e, c.job_w, c.y, T, J, H, k.moe_ffn, ah));
    } else {
        prof("moe_gate_up", xg4_moe_gate_up(q, c.x16, Xg4W{}, Xg4W{}, W(d.gate), W(d.up), nullptr, c.h32, T, 1, k.ffn, H, axf));
        prof("h_pack", xg4_h_pack(q, c.h32, c.h16, c.hs, T, k.ffn));
        if (ah.q) prof("act_q8", xg4_act_q8(q, c.h16, k.ffn, k.ffn, T, ah));
        prof("moe_down", xg4_moe_down(q, c.h16, c.hs, Xg4W{}, W(d.down), nullptr, nullptr, c.y, T, 1, H, k.ffn, ah));
    }
    prof("hc_post", xg4_hc_post(q, c.S, c.post, c.comb, c.y, T, H));
    chk("ffn y", c.y, size_t(T) * H, false); chk("S after ffn", c.S, size_t(T) * HH, false);
    return {};
}

std::string Xing4Model::forward(const int32_t* ids, uint32_t T, float* logits, uint32_t n_logits) {
    std::vector<Xing4Row> rows(T);
    for (uint32_t t = 0; t < T; ++t) rows[t] = {ids[t], 0u};
    const uint32_t nl = std::min(n_logits, T);
    std::vector<uint32_t> lr(nl);
    for (uint32_t i = 0; i < nl; ++i) lr[i] = T - nl + i;
    return forward_rows(rows.data(), T, logits, lr.data(), nl);
}

std::string Xing4Model::forward_rows(const Xing4Row* rows, uint32_t T, float* logits, const uint32_t* logit_rows, uint32_t n_logit_rows) {
    const Xing4Config& k = cfg_;
    if (T == 0) return {};
    if (!logits) n_logit_rows = 0;
    if (T > opt_.max_tokens) {                                  // in chunks, in order
        uint32_t li = 0;
        for (uint32_t t0 = 0; t0 < T; t0 += opt_.max_tokens) {
            const uint32_t n = std::min(opt_.max_tokens, T - t0);
            std::vector<uint32_t> lr; const uint32_t l0 = li;
            while (li < n_logit_rows && logit_rows[li] < t0 + n) lr.push_back(logit_rows[li++] - t0);
            if (auto e = forward_rows(rows + t0, n, lr.empty() ? nullptr : logits + size_t(l0) * k.vocab, lr.data(), uint32_t(lr.size())); !e.empty()) return e;
        }
        return {};
    }
    const uint32_t H = k.dim, HH = k.hc * H;
    // positions and pages
    std::vector<uint32_t> len = lane_len_;
    rinfo_.assign(size_t(T) * 3 + n_logit_rows, 0);
    uint32_t max_keys = 0;
    for (uint32_t t = 0; t < T; ++t) {
        const uint32_t l = rows[t].lane;
        if (l >= len.size()) return "lane out of range";
        if (rows[t].token < 0 || uint32_t(rows[t].token) >= k.vocab) return "token id out of range";
        const uint32_t p = len[l]++;
        if (len[l] > maxp_ * 64) return "the lane's context is full";
        rinfo_[t] = int32_t(l); rinfo_[T + t] = int32_t(p); rinfo_[2 * T + t] = rows[t].token;
        max_keys = std::max(max_keys, p + 1);
    }
    for (uint32_t i = 0; i < n_logit_rows; ++i) rinfo_[3 * T + i] = int32_t(logit_rows[i]);
    // the prompt attention's groups: up to 4 consecutive rows of one lane
    agrp_off_ = 3 * T + n_logit_rows; n_agrp_ = 0;
    for (uint32_t t = 0; t < T;) {
        uint32_t n = 1;
        while (n < 4 && t + n < T && rows[t + n].lane == rows[t].lane) ++n;
        rinfo_.push_back(int32_t(t)); rinfo_.push_back(int32_t(n)); ++n_agrp_;
        t += n;
    }
    for (uint32_t l = 0; l < len.size(); ++l) {
        auto& pg = lane_pages_[l];
        while (pg.size() * 64 < len[l]) {
            if (free_pages_.empty()) { lane_truncate(l, lane_len_[l]); return "the KV cache is full"; }
            pg.push_back(page_take());
            tab_host_[size_t(l) * maxp_ + pg.size() - 1] = pg.back(); tab_dirty_ = true;
        }
    }
    static const bool hostprof = [] { const char* v = std::getenv("IE_XG4_HOSTPROF"); return v && *v == '1'; }();
    using clk = std::chrono::steady_clock;
    const auto h0 = clk::now(); double h_put = 0, h_sub = 0, h_hand = 0, h_tail = 0; auto hm = h0;
    auto lap = [&](double& acc) { const auto n = clk::now(); acc += std::chrono::duration<double, std::micro>(n - hm).count(); hm = n; };
    try {
        for (auto& c : cards_) {
            c->put(c->rinfo, rinfo_.data(), rinfo_.size() * 4);
            if (tab_dirty_) c->put(c->tab, tab_host_.data(), tab_host_.size() * 4);
            prof("rope", xg4_rope_rows(*c->q, c->rinfo + T, c->inv_freq, c->cs, T));
        }
        tab_dirty_ = false;
        lap(h_put);
        Card* cur = cards_[layer_card_[0]].get();
        prof("embed", xg4_embed(*cur->q, cur->rinfo + 2 * T, W(cards_[0]->embd), cur->S, T, H));
        for (uint32_t L = 0; L < k.n_layers; ++L) {
            Card* c = cards_[layer_card_[L]].get();
            if (c != cur) {                                     // hand the streams to the next card
                lap(h_sub);
                if (p2p_) {
                    static const bool hostwait = std::getenv("IE_XG4_PUSH_WAIT") && *std::getenv("IE_XG4_PUSH_WAIT") == '1';
                    if (hostwait) { c->q->wait(); }
                    sycl::event ev = cur->q->memcpy(c->S, cur->S, size_t(T) * HH * 4);
                    if (hostwait) ev.wait();
                    else c->q->submit([&](sycl::handler& h) { h.depends_on(ev); h.single_task<class xg4_push_wait>([] {}); });
                } else {
                    host_x_.resize(size_t(T) * HH);
                    cur->get(host_x_.data(), cur->S, size_t(T) * HH * 4);
                    c->put(c->S, host_x_.data(), size_t(T) * HH * 4);
                }
                cur = c;
                lap(h_hand);
            }
            if (auto e = run_layer(*c, L, T, max_keys); !e.empty()) return e;
        }
        lap(h_sub);
        if (n_logit_rows) {
            sycl::queue& q = *cur->q;
            prof("final_norm", xg4_stream_mean_norm(q, cur->S, cur->rinfo + 3 * T, cur->out_norm, cur->x16, n_logit_rows, H, k.norm_eps));
            const Xg4A ah = intdot_ && cur->head.fmt != kXg4Q8 ? cur->ax : Xg4A{};
            if (ah.q) prof("act_q8", xg4_act_q8(q, cur->x16, H, H, n_logit_rows, ah));
            prof("gemv head", xg4_gemv(q, cur->x16, H, W(cur->head), cur->logits, nullptr, k.vocab, k.vocab, H, n_logit_rows, 1, 0, 0, ah));
            cur->get(logits, cur->logits, size_t(n_logit_rows) * k.vocab * 4);
        } else {
            cur->q->wait();
        }
        lap(h_tail);
        if (hostprof) {
            static double a_put = 0, a_sub = 0, a_hand = 0, a_tail = 0; static uint32_t n = 0;
            a_put += h_put; a_sub += h_sub; a_hand += h_hand; a_tail += h_tail;
            if (++n % 32 == 0) {
                std::fprintf(stderr, "[xing4 host] per call (us): row data %.0f, submitting %.0f, card handoff (waits for card 0) %.0f, tail (waits for card 1 + logits) %.0f\n",
                             a_put / 32, a_sub / 32, a_hand / 32, a_tail / 32);
                a_put = a_sub = a_hand = a_tail = 0;
            }
        }
    } catch (const sycl::exception& e) {
        return std::string("sycl: ") + e.what();
    } catch (const std::exception& e) {
        return e.what();
    }
    lane_len_ = len;
    return {};
}

}  // namespace ie
