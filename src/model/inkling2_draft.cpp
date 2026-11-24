// src/model/inkling2_draft.cpp — Inkling 2 Mini's draft head (include/ie/inkling2_draft.hpp).
#include "ie/inkling2_draft.hpp"

#include "ie/inkling2_ops.hpp"
#include "ie/safetensors.hpp"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <cmath>
#include <cstring>
#include <fstream>

namespace ie {

namespace {
constexpr uint32_t kD = 128;            // head_dim
constexpr uint32_t kChunk = 1024;       // attention keys per work-group
constexpr uint32_t kRewind = 64;        // how far back a truncate may go
constexpr uint32_t kArgWG = 256;        // argmax partial groups

float bf16f(uint16_t b) { uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
uint16_t f2bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return uint16_t(u >> 16); }

struct DLayer {
    const uint16_t *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr, *wkc = nullptr, *wvc = nullptr;
    const uint16_t *wg = nullptr, *wu = nullptr, *wd = nullptr;
    const float *in_norm = nullptr, *post_norm = nullptr, *dflare = nullptr, *qn = nullptr, *kn = nullptr, *knc = nullptr;
    sycl::half *Kc = nullptr, *Vc = nullptr;   // [slots][nkv][128]
    uint32_t slots = 0;                        // ring length (local) or max positions (global)
    bool local = false;
};
}  // namespace

struct Inkling2Draft::Impl {
    sycl::queue* q = nullptr;
    bool ok = false;
    uint32_t H = 0, NH = 0, NKV = 0, L = 0, B = 0, R = 0, I = 0, win = 0, V = 0, max_ctx = 0, taps = 0;
    float eps = 1e-6f, theta = 1e6f, softcap = 30.f;
    std::vector<uint32_t> visits;
    std::vector<DLayer> lay;
    const uint16_t* unembed = nullptr;
    const float *final_norm = nullptr, *alpha = nullptr, *mk_dflare = nullptr, *rp_b = nullptr, *ho_b = nullptr;
    const uint16_t *w1 = nullptr, *rp_a = nullptr, *rp_c = nullptr, *fold = nullptr;   // fold [V][R] = unembed @ hidden_out
    float* fold_b = nullptr;                    // [V] = unembed @ hidden_out.bias
    std::vector<float> mask_embed, embed_norm;  // host
    SafetensorsReader mtp;
    SafetensorsModel main;
    const SafeTensorInfo* embed = nullptr;      // host mmap bf16 [vocab, H]
    // scratch
    float *feat = nullptr, *mixd = nullptr, *kv = nullptr, *frnn = nullptr;   // frnn ring [kRewind][H]
    uint32_t feat_rows = 0;
    float *x = nullptr, *xn = nullptr, *qb = nullptr, *kb = nullptr, *vb = nullptr, *att = nullptr, *tmp = nullptr, *g = nullptr,
          *u = nullptr, *h = nullptr, *base = nullptr, *zc = nullptr, *ctxt = nullptr, *state = nullptr, *lat = nullptr,
          *fl = nullptr, *part = nullptr, *argv = nullptr;
    int32_t *argi = nullptr, *prev = nullptr, *outd = nullptr;
    std::vector<int32_t> pos_host;
    int32_t* pos = nullptr;
    uint32_t n = 0, hw = 0;
    uint64_t bytes = 0;
    std::vector<void*> owned;

    template <class T> T* dev(size_t count) {
        T* p = sycl::malloc_device<T>(count, *q);
        if (!p) throw std::runtime_error("draft: device allocation failed");
        owned.push_back(p); bytes += count * sizeof(T);
        return p;
    }
    void free_all() {
        if (q) { q->wait(); for (void* p : owned) sycl::free(p, *q); }
        owned.clear(); bytes = 0; ok = false;
    }
    std::string up16(const std::string& name, size_t numel, const uint16_t*& out) {
        const SafeTensorInfo* t = mtp.find(name);
        if (!t) return "draft: missing " + name;
        if (t->dtype_str != "BF16" || t->numel() != numel) return "draft: unexpected dtype / shape of " + name;
        uint16_t* d = dev<uint16_t>(numel);
        q->memcpy(d, t->data, numel * 2).wait();
        out = d;
        return {};
    }
    std::string host32(const std::string& name, size_t numel, std::vector<float>& out) {
        const SafeTensorInfo* t = mtp.find(name);
        if (!t) return "draft: missing " + name;
        if (t->dtype_str != "BF16" || t->numel() != numel) return "draft: unexpected dtype / shape of " + name;
        out.resize(numel);
        for (size_t i = 0; i < numel; ++i) { uint16_t b; std::memcpy(&b, t->data + 2 * i, 2); out[i] = bf16f(b); }
        return {};
    }
    std::string up32(const std::string& name, size_t numel, const float*& out) {
        std::vector<float> h;
        if (auto e = host32(name, numel, h); !e.empty()) return e;
        float* d = dev<float>(numel);
        q->memcpy(d, h.data(), numel * 4).wait();
        out = d;
        return {};
    }

    // ---------------------------------------------------------------------------------------------------- kernels
    // y[t][n] = x[t] . W[n] for T <= 8 rows (bf16 W [N][K], K % 128 == 0): one sub-group of 16 per output row, the
    // weight read once for all rows (the generic i2_linear_bf16 re-reads it per row).
    void lin(const float* x, uint32_t T, uint32_t K, const uint16_t* W, uint32_t N, float* y) {
        if (T > 8 || K % 128) { i2_linear_bf16(*q, x, T, K, W, N, y, false, false); return; }
        constexpr uint32_t SG = 16, RPW = 8;
        const uint32_t groups = (N + RPW - 1) / RPW;
        q->parallel_for(sycl::nd_range<1>(size_t(groups) * RPW * SG, RPW * SG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const auto sg = it.get_sub_group();
            const uint32_t n = it.get_group(0) * RPW + sg.get_group_linear_id(), l = sg.get_local_linear_id();
            if (n >= N) return;
            const uint16_t* w = W + size_t(n) * K;
            float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (uint32_t k = l * 8; k < K; k += SG * 8) {
                const sycl::vec<uint32_t, 4> wv = *reinterpret_cast<const sycl::vec<uint32_t, 4>*>(w + k);
                float wf[8];
                for (int j = 0; j < 4; ++j) {
                    const uint32_t u = wv[j];
                    wf[2 * j] = sycl::bit_cast<float>(u << 16); wf[2 * j + 1] = sycl::bit_cast<float>(u & 0xffff0000u);
                }
#pragma unroll
                for (uint32_t t = 0; t < 8; ++t) {
                    if (t < T) {
                        const float* xr = x + size_t(t) * K + k;
                        float a = 0.f;
#pragma unroll
                        for (int j = 0; j < 8; ++j) a += xr[j] * wf[j];
                        acc[t] += a;
                    }
                }
            }
#pragma unroll
            for (uint32_t t = 0; t < 8; ++t)
                if (t < T) {
                    const float s = sycl::reduce_over_group(sg, acc[t], sycl::plus<float>());
                    if (l == 0) y[size_t(t) * N + n] = s;
                }
        });
    }
    // out[r] = rmsnorm(sum_j alpha[j] * f[j][first + r]) * w   (f [taps][T][H])
    void mix_norm(const float* f, uint32_t T, uint32_t first, uint32_t rows, const float* al, const float* w, float* out) {
        const uint32_t H_ = H, tp = taps; const float e = eps;
        q->submit([&](sycl::handler& cg) {
            sycl::local_accessor<float, 1> red(sycl::range<1>(256), cg);
            cg.parallel_for(sycl::nd_range<1>(rows * 256, 256), [=](sycl::nd_item<1> it) {
                const uint32_t r = it.get_group(0), t = it.get_local_id(0);
                float ss = 0.f;
                for (uint32_t i = t; i < H_; i += 256) {
                    float m = 0.f;
                    for (uint32_t j = 0; j < tp; ++j) m += al[j] * f[(uint64_t(j) * T + first + r) * H_ + i];
                    out[uint64_t(r) * H_ + i] = m; ss += m * m;
                }
                ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
                const float inv = sycl::rsqrt(ss / H_ + e);
                for (uint32_t i = t; i < H_; i += 256) out[uint64_t(r) * H_ + i] *= inv * w[i];
            });
        });
    }
    // Per (row, head) of x [rows][heads][128]: optional RMSNorm (w), neox RoPE at pos[row]; result to y (fp32,
    // same layout) or to a half cache (slot = slot0 + row mod slots).
    void norm_rope(float* x, uint32_t rows, uint32_t heads, const float* w, const int32_t* ps, sycl::half* cache,
                   uint32_t slots, uint32_t slot0) {
        const float e = eps, th = theta;
        q->parallel_for(sycl::nd_range<1>(rows * heads * kD, kD), [=](sycl::nd_item<1> it) {
            const uint32_t g = it.get_group(0), d = it.get_local_id(0), r = g / heads, hh = g % heads;
            float* v = x + uint64_t(g) * kD;
            const float xv = v[d];
            float inv = 1.f;
            if (w) inv = sycl::rsqrt(sycl::reduce_over_group(it.get_group(), xv * xv, sycl::plus<float>()) / kD + e);
            const float a = w ? xv * inv * w[d] : xv;
            const uint32_t j = d % (kD / 2);
            const float other_raw = v[d < kD / 2 ? d + kD / 2 : d - kD / 2];
            const float o = w ? other_raw * inv * w[d < kD / 2 ? d + kD / 2 : d - kD / 2] : other_raw;
            const float f = float(ps[r]) * sycl::pow(th, -float(2 * j) / float(kD));
            const float c = sycl::cos(f), s = sycl::sin(f);
            const float y = d < kD / 2 ? a * c - o * s : a * c + o * s;
            sycl::group_barrier(it.get_group());
            if (cache) cache[(uint64_t((slot0 + r) % slots) * heads + hh) * kD + d] = sycl::half(y);
            else v[d] = y;
        });
    }
    // Attention of the B block rows: q [B][NH][128], context keys [lo, p) from the layer cache, block keys kb/vb
    // [B][NKV][128]; out [B][NH * 128]. Split over chunks of kChunk keys, then combined.
    void attention(const DLayer& l, uint32_t p, uint32_t lo, float* out) {
        const uint32_t S = p - lo + B, nch = (S + kChunk - 1) / kChunk, nh = NH, nkv = NKV, Bb = B, slots = l.slots;
        const sycl::half *Kc = l.Kc, *Vc = l.Vc;
        const float *Q = qb, *Kb = kb, *Vb = vb;
        float* P = part;   // [B * NH][nch][2 + 128]
        const float sc = 1.f / std::sqrt(float(kD));
        q->submit([&](sycl::handler& cg) {
            sycl::local_accessor<float, 1> sp(sycl::range<1>(kChunk), cg);
            sycl::local_accessor<float, 1> qs(sycl::range<1>(kD), cg);
            sycl::local_accessor<float, 1> acc(sycl::range<1>(2 * kD), cg);
            cg.parallel_for(sycl::nd_range<2>({size_t(Bb * nh), size_t(nch) * 256}, {1, 256}), [=](sycl::nd_item<2> it) {
                const uint32_t rh = it.get_global_id(0), ch = it.get_group(1), t = it.get_local_id(1);
                const uint32_t r = rh / nh, hh = rh % nh, kh = hh / (nh / nkv);
                if (t < kD) qs[t] = Q[uint64_t(rh) * kD + t] * sc;
                sycl::group_barrier(it.get_group());
                const uint32_t s0 = ch * kChunk, s1 = sycl::min(S, s0 + kChunk);
                float mx = -INFINITY;
                for (uint32_t s = s0 + t; s < s1; s += 256) {
                    const uint32_t key = lo + s;
                    float dot = 0.f;
                    if (key < p) {
                        const sycl::half* k = Kc + (uint64_t(key % slots) * nkv + kh) * kD;
                        for (uint32_t d = 0; d < kD; ++d) dot += qs[d] * float(k[d]);
                    } else {
                        const float* k = Kb + (uint64_t(key - p) * nkv + kh) * kD;
                        for (uint32_t d = 0; d < kD; ++d) dot += qs[d] * k[d];
                    }
                    sp[s - s0] = dot; mx = sycl::max(mx, dot);
                }
                mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
                float sum = 0.f;
                for (uint32_t s = s0 + t; s < s1; s += 256) { const float e = sycl::exp(sp[s - s0] - mx); sp[s - s0] = e; sum += e; }
                sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
                // out[d]: threads 0..127 the even keys, 128..255 the odd
                const uint32_t d = t % kD, half = t / kD;
                float o = 0.f;
                for (uint32_t s = s0 + half; s < s1; s += 2) {
                    const uint32_t key = lo + s;
                    const float vv = key < p ? float(Vc[(uint64_t(key % slots) * nkv + kh) * kD + d])
                                             : Vb[(uint64_t(key - p) * nkv + kh) * kD + d];
                    o += sp[s - s0] * vv;
                }
                acc[t] = o;
                sycl::group_barrier(it.get_group());
                float* dst = P + (uint64_t(rh) * nch + ch) * (2 + kD);
                if (t < kD) dst[2 + t] = acc[t] + acc[t + kD];
                if (t == 0) { dst[0] = mx; dst[1] = sum; }
            });
        });
        q->parallel_for(sycl::nd_range<1>(Bb * nh * kD, kD), [=](sycl::nd_item<1> it) {
            const uint32_t rh = it.get_group(0), d = it.get_local_id(0);
            const float* src = P + uint64_t(rh) * nch * (2 + kD);
            float m = -INFINITY;
            for (uint32_t c = 0; c < nch; ++c) m = sycl::max(m, src[c * (2 + kD)]);
            float l = 0.f, o = 0.f;
            for (uint32_t c = 0; c < nch; ++c) {
                const float w = sycl::exp(src[c * (2 + kD)] - m);
                l += w * src[c * (2 + kD) + 1]; o += w * src[c * (2 + kD) + 2 + d];
            }
            out[uint64_t(rh) * kD + d] = o / l;
        });
    }
};

Inkling2Draft::Inkling2Draft() : p_(std::make_unique<Impl>()) {}
Inkling2Draft::~Inkling2Draft() { p_->free_all(); }
bool Inkling2Draft::ready() const { return p_->ok; }
uint32_t Inkling2Draft::block() const { return p_->B; }
const std::vector<uint32_t>& Inkling2Draft::feature_visits() const { return p_->visits; }
uint64_t Inkling2Draft::device_bytes() const { return p_->bytes; }
uint32_t Inkling2Draft::n_ctx() const { return p_->n; }
void Inkling2Draft::reset() { p_->n = 0; p_->hw = 0; }

std::string Inkling2Draft::init(sycl::queue& q, const std::string& dir, const uint16_t* unembed, uint32_t vocab, uint32_t max_ctx) {
    Impl& m = *p_;
    m.free_all();
    m.q = &q;
    nlohmann::json cfg;
    try { cfg = nlohmann::json::parse(std::ifstream(dir + "/config.json")); } catch (const std::exception& e) { return std::string("draft: config.json: ") + e.what(); }
    if (!cfg.contains("mtp_config")) return "draft: no mtp_config";
    const auto& mc = cfg["mtp_config"];
    const auto& tc = cfg.contains("text_config") ? cfg["text_config"] : cfg;
    try {
        m.H = mc.at("hidden_size"); m.NH = mc.at("num_attention_heads"); m.NKV = mc.at("num_key_value_heads");
        m.L = mc.at("num_hidden_layers"); m.B = mc.at("block_size"); m.R = mc.at("markov_rank"); m.I = mc.at("intermediate_size");
        m.win = mc.at("sliding_window_size"); m.eps = mc.at("rms_norm_eps"); m.theta = mc.at("rope_theta");
        m.visits = mc.at("target_layer_ids").get<std::vector<uint32_t>>();
        m.softcap = tc.value("final_logit_softcapping", 30.f);
        if (uint32_t(mc.at("head_dim")) != kD || mc.value("markov_head_type", "") != "rnn_hidden" || mc.value("q_bias", false) ||
            mc.value("o_bias", false) || m.NH % m.NKV || m.R != 256 || m.B < 2)
            return "draft: unsupported mtp_config";
    } catch (const std::exception& e) { return std::string("draft: mtp_config: ") + e.what(); }
    m.taps = uint32_t(m.visits.size());
    m.V = vocab; m.unembed = unembed; m.max_ctx = max_ctx;
    if (auto e = m.mtp.open(dir + "/mtp.safetensors"); !e.empty()) return "draft: " + e;
    if (auto e = m.main.open(dir); !e.empty()) return "draft: " + e;
    m.embed = m.main.find("model.llm.embed.weight");
    const SafeTensorInfo* en = m.main.find("model.llm.embed_norm.weight");
    if (!m.embed || !en || en->numel() != m.H) return "draft: the target's embed / embed_norm not found";
    m.embed_norm.resize(m.H);
    for (uint32_t i = 0; i < m.H; ++i) { uint16_t b; std::memcpy(&b, en->data + 2 * i, 2); m.embed_norm[i] = bf16f(b); }
    std::string e;
    auto ok = [&](std::string r) { if (e.empty() && !r.empty()) e = r; return e.empty(); };
    try {
        const uint32_t H = m.H, QD = m.NH * kD, KD = m.NKV * kD, R = m.R;
        std::vector<int32_t> local = mc.value("local_layer_ids", std::vector<int32_t>{});
        m.lay.resize(m.L);
        for (uint32_t i = 0; i < m.L && e.empty(); ++i) {
            DLayer& l = m.lay[i];
            const std::string p = "layers." + std::to_string(i) + ".";
            ok(m.up16(p + "self_attn.q_proj.weight", size_t(QD) * H, l.wq)); ok(m.up16(p + "self_attn.k_proj.weight", size_t(KD) * H, l.wk));
            ok(m.up16(p + "self_attn.v_proj.weight", size_t(KD) * H, l.wv)); ok(m.up16(p + "self_attn.o_proj.weight", size_t(H) * QD, l.wo));
            ok(m.up16(p + "self_attn.k_proj_ctx.weight", size_t(KD) * H, l.wkc)); ok(m.up16(p + "self_attn.v_proj_ctx.weight", size_t(KD) * H, l.wvc));
            ok(m.up16(p + "mlp.gate_proj.weight", size_t(m.I) * H, l.wg)); ok(m.up16(p + "mlp.up_proj.weight", size_t(m.I) * H, l.wu));
            ok(m.up16(p + "mlp.down_proj.weight", size_t(H) * m.I, l.wd));
            ok(m.up32(p + "input_layernorm.weight", H, l.in_norm)); ok(m.up32(p + "post_attention_layernorm.weight", H, l.post_norm));
            ok(m.up32(p + "dflare_norm.weight", H, l.dflare)); ok(m.up32(p + "self_attn.q_norm.weight", kD, l.qn));
            ok(m.up32(p + "self_attn.k_norm.weight", kD, l.kn)); ok(m.up32(p + "self_attn.k_norm_ctx.weight", kD, l.knc));
            l.local = std::find(local.begin(), local.end(), int32_t(i)) != local.end();
            l.slots = l.local ? m.win + kRewind : max_ctx + kRewind;
            if (e.empty()) { l.Kc = m.dev<sycl::half>(size_t(l.slots) * KD); l.Vc = m.dev<sycl::half>(size_t(l.slots) * KD); }
        }
        ok(m.up32("norm.weight", H, m.final_norm));
        ok(m.up32("markov_head.dflare_norm.weight", H, m.mk_dflare));
        ok(m.host32("mask_embed", H, m.mask_embed));
        if (!e.empty()) { m.free_all(); return e; }
        {   // markov_w1 [padded vocab, R]: its row count from the file
            const SafeTensorInfo* t = m.mtp.find("markov_head.markov_w1.weight");
            if (!t || t->shape.size() != 2 || uint64_t(t->shape[1]) != R || uint64_t(t->shape[0]) < m.V) { m.free_all(); return "draft: markov_w1 shape"; }
            ok(m.up16("markov_head.markov_w1.weight", t->numel(), m.w1));
        }
        {   // fuse_weight [L + 1, taps] -> softmax rows
            std::vector<float> fw;
            ok(m.host32("fuse_weight", size_t(m.L + 1) * m.taps, fw));
            if (e.empty()) {
                for (uint32_t r = 0; r <= m.L; ++r) {
                    float mx = -INFINITY, s = 0.f;
                    for (uint32_t j = 0; j < m.taps; ++j) mx = std::max(mx, fw[r * m.taps + j]);
                    for (uint32_t j = 0; j < m.taps; ++j) { fw[r * m.taps + j] = std::exp(fw[r * m.taps + j] - mx); s += fw[r * m.taps + j]; }
                    for (uint32_t j = 0; j < m.taps; ++j) fw[r * m.taps + j] /= s;
                }
                float* d = m.dev<float>(fw.size());
                q.memcpy(d, fw.data(), fw.size() * 4).wait();
                m.alpha = d;
            }
        }
        {   // rnn_proj [3R, 2R + 2H] split into the recurrent columns [state, prev] and the context columns [h, f_rnn]
            std::vector<float> w, b;
            ok(m.host32("markov_head.rnn_proj.weight", size_t(3 * R) * (2 * R + 2 * H), w));
            ok(m.host32("markov_head.rnn_proj.bias", 3 * R, b));
            if (e.empty()) {
                const size_t K = 2 * R + 2 * H;
                std::vector<uint16_t> a(size_t(3 * R) * 2 * R), c(size_t(3 * R) * 2 * H);
                for (size_t o = 0; o < 3 * R; ++o) {
                    for (size_t k = 0; k < 2 * R; ++k) a[o * 2 * R + k] = f2bf16(w[o * K + k]);
                    for (size_t k = 0; k < 2 * H; ++k) c[o * 2 * H + k] = f2bf16(w[o * K + 2 * R + k]);
                }
                uint16_t* da = m.dev<uint16_t>(a.size()); uint16_t* dc = m.dev<uint16_t>(c.size()); float* db = m.dev<float>(b.size());
                q.memcpy(da, a.data(), a.size() * 2); q.memcpy(dc, c.data(), c.size() * 2); q.memcpy(db, b.data(), b.size() * 4).wait();
                m.rp_a = da; m.rp_c = dc; m.rp_b = db;
            }
        }
        {   // fold = unembed @ hidden_out [V, R] (bf16), fold_b = unembed @ hidden_out.bias: on the card
            std::vector<float> hw, hb;
            ok(m.host32("markov_head.hidden_out.weight", size_t(H) * R, hw));
            ok(m.host32("markov_head.hidden_out.bias", H, hb));
            if (!e.empty()) { m.free_all(); return e; }
            std::vector<float> hwT(size_t(R) * H);
            for (uint32_t i = 0; i < H; ++i) for (uint32_t r = 0; r < R; ++r) hwT[size_t(r) * H + i] = hw[size_t(i) * R + r];
            float* x = sycl::malloc_device<float>(hwT.size() + H, q);
            float* y = sycl::malloc_device<float>(size_t(R) * m.V, q);
            if (!x || !y) { if (x) sycl::free(x, q); if (y) sycl::free(y, q); m.free_all(); return "draft: device allocation failed (fold)"; }
            q.memcpy(x, hwT.data(), hwT.size() * 4); q.memcpy(x + hwT.size(), hb.data(), H * 4);
            i2_linear_bf16(q, x, R, H, m.unembed, m.V, y, false, false);
            m.fold_b = m.dev<float>(m.V);
            i2_linear_bf16(q, x + hwT.size(), 1, H, m.unembed, m.V, m.fold_b, false, false);
            uint16_t* f = m.dev<uint16_t>(size_t(m.V) * R);
            const uint32_t Vv = m.V;
            q.parallel_for(sycl::range<2>(Vv, R), [=](sycl::id<2> id) {
                const uint32_t v = id[0], r = id[1];
                const uint32_t u = sycl::bit_cast<uint32_t>(y[size_t(r) * Vv + v]);
                f[size_t(v) * R + r] = uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16);
            }).wait();
            sycl::free(x, q); sycl::free(y, q);
            m.fold = f;
        }
        // scratch
        const uint32_t B = m.B;
        m.feat_rows = 0;
        m.mixd = nullptr; m.feat = nullptr; m.kv = nullptr;
        m.frnn = m.dev<float>(size_t(kRewind) * H);
        m.x = m.dev<float>(size_t(B) * H); m.xn = m.dev<float>(size_t(B) * H); m.tmp = m.dev<float>(size_t(B) * H);
        m.qb = m.dev<float>(size_t(B) * QD); m.kb = m.dev<float>(size_t(B) * KD); m.vb = m.dev<float>(size_t(B) * KD);
        m.att = m.dev<float>(size_t(B) * QD); m.g = m.dev<float>(size_t(B) * m.I); m.u = m.dev<float>(size_t(B) * m.I);
        m.h = m.dev<float>(size_t(B) * H); m.base = m.dev<float>(size_t(B) * m.V);
        m.zc = m.dev<float>(size_t(B) * 2 * H); m.ctxt = m.dev<float>(size_t(B) * 3 * R);
        m.state = m.dev<float>(R); m.lat = m.dev<float>(R); m.fl = m.dev<float>(m.V);
        m.argv = m.dev<float>(kArgWG); m.argi = m.dev<int32_t>(kArgWG); m.prev = m.dev<int32_t>(1); m.outd = m.dev<int32_t>(B);
        m.pos = m.dev<int32_t>(std::max<uint32_t>(B, 4096));
        const uint32_t nch_max = (max_ctx + B + kChunk - 1) / kChunk;
        m.part = m.dev<float>(size_t(B) * m.NH * nch_max * (2 + kD));
    } catch (const std::exception& ex) { m.free_all(); return std::string("draft: ") + ex.what(); }
    if (!e.empty()) { m.free_all(); return e; }
    m.n = 0; m.hw = 0; m.ok = true;
    return {};
}

std::string Inkling2Draft::add_context(const std::vector<float>& features, uint32_t T, uint32_t first, uint32_t rows) {
    Impl& m = *p_;
    if (!m.ok) return "draft: not loaded";
    if (rows == 0) return {};
    if (features.size() != size_t(m.taps) * T * m.H || first + rows > T) return "draft: features shape";
    if (m.n + rows > m.max_ctx) return "draft: context full";
    const uint32_t H = m.H, KD = m.NKV * kD;
    try {
        if (m.feat_rows < T) {   // device copies of a forward's features (grown to the largest forward)
            m.q->wait();
            auto drop = [&](void* p) { if (!p) return; sycl::free(p, *m.q); m.owned.erase(std::find(m.owned.begin(), m.owned.end(), p)); };
            drop(m.feat); drop(m.mixd); drop(m.kv);
            m.feat_rows = std::max<uint32_t>(T, 64);
            m.feat = m.dev<float>(size_t(m.taps) * m.feat_rows * H);
            m.mixd = m.dev<float>(size_t(m.feat_rows) * H);
            m.kv = m.dev<float>(size_t(m.feat_rows) * KD);
        }
        m.q->memcpy(m.feat, features.data(), features.size() * 4);
        m.pos_host.resize(rows);
        for (uint32_t r = 0; r < rows; ++r) m.pos_host[r] = int32_t(m.n + r);
        int32_t* ps = sycl::malloc_device<int32_t>(rows, *m.q);
        m.q->memcpy(ps, m.pos_host.data(), rows * 4);
        for (uint32_t i = 0; i < m.L; ++i) {
            DLayer& l = m.lay[i];
            m.mix_norm(m.feat, T, first, rows, m.alpha + i * m.taps, l.dflare, m.mixd);
            i2_linear_bf16(*m.q, m.mixd, rows, H, l.wkc, KD, m.kv, false, false);
            m.norm_rope(m.kv, rows, m.NKV, l.knc, ps, l.Kc, l.slots, m.n);
            i2_linear_bf16(*m.q, m.mixd, rows, H, l.wvc, KD, m.kv, false, false);
            sycl::half* Vc = l.Vc; const float* kvp = m.kv; const uint32_t slots = l.slots, n0 = m.n;
            m.q->parallel_for(sycl::range<2>(rows, KD), [=](sycl::id<2> id) {
                Vc[uint64_t((n0 + id[0]) % slots) * KD + id[1]] = sycl::half(kvp[uint64_t(id[0]) * KD + id[1]]);
            });
        }
        // the Markov context of each row, into the ring (the draft reads the one at anchor - 1)
        const uint32_t keep = std::min<uint32_t>(rows, kRewind);
        m.mix_norm(m.feat, T, first + rows - keep, keep, m.alpha + m.L * m.taps, m.mk_dflare, m.mixd);
        for (uint32_t r = 0; r < keep; ++r)
            m.q->memcpy(m.frnn + size_t((m.n + rows - keep + r) % kRewind) * H, m.mixd + size_t(r) * H, H * 4);
        m.q->wait();
        sycl::free(ps, *m.q);
    } catch (const std::exception& ex) { return std::string("draft: ") + ex.what(); }
    m.n += rows;
    m.hw = std::max(m.hw, m.n);
    return {};
}

std::string Inkling2Draft::truncate(uint32_t n) {
    Impl& m = *p_;
    if (n > m.n) return "draft: truncate beyond the context";
    if (m.hw - n > kRewind) return "draft: truncate too far back (reset and refill)";
    m.n = n;
    return {};
}

std::string Inkling2Draft::draft(int32_t anchor, std::vector<int32_t>& out) {
    Impl& m = *p_;
    if (!m.ok) return "draft: not loaded";
    if (m.n == 0) return "draft: no context";
    if (anchor < 0 || !m.embed || uint64_t(anchor) >= uint64_t(m.embed->shape[0])) return "draft: anchor out of range";
    const uint32_t H = m.H, B = m.B, QD = m.NH * kD, KD = m.NKV * kD, R = m.R, p = m.n, V = m.V;
    static const bool prof = [] { const char* e = std::getenv("IE_INKLING2_DRAFT_PROFILE"); return e && *e == '1'; }();
    std::vector<std::pair<const char*, double>> marks;
    auto tick = std::chrono::steady_clock::now();
    auto mark = [&](const char* what) {
        if (!prof) return;
        m.q->wait();
        const auto now = std::chrono::steady_clock::now();
        marks.emplace_back(what, std::chrono::duration<double, std::milli>(now - tick).count());
        tick = now;
    };
    try {
        sycl::queue& q = *m.q;
        std::vector<float> x0(size_t(B) * H);
        {   // row 0: embed_norm(embed[anchor]); rows 1..: mask_embed
            const uint8_t* row = m.embed->data + size_t(anchor) * H * 2;
            double ss = 0;
            for (uint32_t i = 0; i < H; ++i) { uint16_t b; std::memcpy(&b, row + 2 * i, 2); x0[i] = bf16f(b); ss += double(x0[i]) * x0[i]; }
            const float inv = 1.f / std::sqrt(float(ss / H) + m.eps);
            for (uint32_t i = 0; i < H; ++i) x0[i] *= inv * m.embed_norm[i];
            for (uint32_t r = 1; r < B; ++r) std::memcpy(x0.data() + size_t(r) * H, m.mask_embed.data(), H * 4);
        }
        std::vector<int32_t> ph(B);
        for (uint32_t r = 0; r < B; ++r) ph[r] = int32_t(p + r);
        q.memcpy(m.x, x0.data(), x0.size() * 4);
        q.memcpy(m.pos, ph.data(), B * 4);
        q.memcpy(m.prev, &anchor, 4);
        for (uint32_t i = 0; i < m.L; ++i) {
            const DLayer& l = m.lay[i];
            i2_rmsnorm(q, m.x, B, H, l.in_norm, m.eps, m.xn, H, false);
            m.lin(m.xn, B, H, l.wq, QD, m.qb);
            m.lin(m.xn, B, H, l.wk, KD, m.kb);
            m.lin(m.xn, B, H, l.wv, KD, m.vb);
            m.norm_rope(m.qb, B, m.NH, l.qn, m.pos, nullptr, 0, 0);
            m.norm_rope(m.kb, B, m.NKV, l.kn, m.pos, nullptr, 0, 0);
            mark("qkv");
            const uint32_t lo = l.local ? (p > m.win ? p - m.win : 0) : 0;
            m.attention(l, p, lo, m.att);
            mark("attention");
            m.lin(m.att, B, QD, l.wo, H, m.tmp);
            i2_add(q, m.x, m.tmp, uint64_t(B) * H, m.x, false);
            i2_rmsnorm(q, m.x, B, H, l.post_norm, m.eps, m.xn, H, false);
            m.lin(m.xn, B, H, l.wg, m.I, m.g);
            m.lin(m.xn, B, H, l.wu, m.I, m.u);
            float *g = m.g; const float* u = m.u;
            q.parallel_for(sycl::range<1>(size_t(B) * m.I), [=](sycl::id<1> j) { const float a = g[j]; g[j] = a / (1.f + sycl::exp(-a)) * u[j]; });
            m.lin(m.g, B, m.I, l.wd, H, m.tmp);
            i2_add(q, m.x, m.tmp, uint64_t(B) * H, m.x, false);
            mark("o + mlp");
        }
        i2_rmsnorm(q, m.x, B, H, m.final_norm, m.eps, m.h, H, false);
        const uint32_t D = B - 1;
        m.lin(m.h + H, D, H, m.unembed, V, m.base);              // base logits of rows 1..
        {   // the Markov context term [h_k, f_rnn] for every step at once
            float* zc = m.zc; const float* hh = m.h + H; const float* fr = m.frnn + size_t((p - 1) % kRewind) * H;
            q.parallel_for(sycl::range<2>(D, 2 * H), [=](sycl::id<2> id) {
                const uint32_t r = id[0], c = id[1];
                zc[size_t(r) * 2 * H + c] = c < H ? hh[size_t(r) * H + c] : fr[c - H];
            });
            m.lin(m.zc, D, 2 * H, m.rp_c, 3 * R, m.ctxt);
        }
        mark("base logits + markov ctx");
        q.memset(m.state, 0, R * 4);
        for (uint32_t s = 0; s < D; ++s) {
            {   // raw = ctxt[s] + bias + Wa [state, w1[prev]]; gate / cand / out
                const float *ct = m.ctxt + size_t(s) * 3 * R, *bb = m.rp_b;
                const uint16_t *wa = m.rp_a, *w1 = m.w1;
                float *st = m.state, *lat = m.lat; const int32_t* pv = m.prev;
                q.submit([&](sycl::handler& cg) {
                    sycl::local_accessor<float, 1> in(sycl::range<1>(2 * R), cg);
                    sycl::local_accessor<float, 1> raw(sycl::range<1>(3 * R), cg);
                    cg.parallel_for(sycl::nd_range<1>(R, R), [=](sycl::nd_item<1> it) {
                        const uint32_t t = it.get_local_id(0);
                        in[t] = st[t];
                        in[R + t] = sycl::bit_cast<float>(uint32_t(w1[size_t(*pv) * R + t]) << 16);
                        sycl::group_barrier(it.get_group());
                        for (uint32_t o = t; o < 3 * R; o += R) {
                            float a = ct[o] + bb[o];
                            for (uint32_t k = 0; k < 2 * R; ++k) a += sycl::bit_cast<float>(uint32_t(wa[size_t(o) * 2 * R + k]) << 16) * in[k];
                            raw[o] = a;
                        }
                        sycl::group_barrier(it.get_group());
                        const float gt = 1.f / (1.f + sycl::exp(-raw[t]));
                        st[t] = gt * in[t] + (1.f - gt) * sycl::tanh(raw[R + t]);
                        lat[t] = sycl::tanh(raw[2 * R + t]);
                    });
                });
            }
            mark("markov gate");
            m.lin(m.lat, 1, R, m.fold, V, m.fl);
            mark("fold");
            {   // logits = softcap(base + fold + fold_b); argmax (lower id on ties)
                const float *bs = m.base + size_t(s) * V, *fl = m.fl, *fb = m.fold_b; const float cap = m.softcap;
                float* av = m.argv; int32_t* ai = m.argi;
                q.parallel_for(sycl::nd_range<1>(kArgWG * 256, 256), [=](sycl::nd_item<1> it) {
                    float best = -INFINITY; int32_t bi = 0x7fffffff;
                    for (uint32_t v = it.get_global_id(0); v < V; v += kArgWG * 256) {
                        const float z = cap * sycl::tanh((bs[v] + fl[v] + fb[v]) / cap);
                        if (z > best) { best = z; bi = int32_t(v); }
                    }
                    const float gm = sycl::reduce_over_group(it.get_group(), best, sycl::maximum<float>());
                    const int32_t gi = sycl::reduce_over_group(it.get_group(), best == gm ? bi : 0x7fffffff, sycl::minimum<int32_t>());
                    if (it.get_local_id(0) == 0) { av[it.get_group(0)] = gm; ai[it.get_group(0)] = gi; }
                });
                int32_t *pv = m.prev, *od = m.outd;
                q.parallel_for(sycl::nd_range<1>(kArgWG, kArgWG), [=](sycl::nd_item<1> it) {
                    const uint32_t t = it.get_local_id(0);
                    const float v = av[t];
                    const float gm = sycl::reduce_over_group(it.get_group(), v, sycl::maximum<float>());
                    const int32_t gi = sycl::reduce_over_group(it.get_group(), v == gm ? ai[t] : 0x7fffffff, sycl::minimum<int32_t>());
                    if (t == 0) { *pv = gi; od[s] = gi; }
                });
            }
            mark("argmax");
        }
        out.resize(D);
        q.memcpy(out.data(), m.outd, D * 4).wait();
        if (prof) {
            std::map<std::string, double> agg;
            for (auto& [k, v] : marks) agg[k] += v;
            std::string line;
            for (auto& [k, v] : agg) line += " " + k + " " + std::to_string(v).substr(0, 5);
            std::fprintf(stderr, "[draft profile ms]%s\n", line.c_str());
        }
    } catch (const std::exception& ex) { return std::string("draft: ") + ex.what(); }
    return {};
}

}  // namespace ie
