// src/model/mimo26_dflash.cpp — MiMo-V2.6's DFlash drafter (P5). See include/ie/mimo26_dflash.hpp.
#include "ie/mimo26_dflash.hpp"

#include "ie/kernel_profiler.hpp"
#include "ie/mimo26.hpp"       // mimo26_bf16_to_f32
#include "ie/mimo26_ops.hpp"
#include "ie/ops.hpp"

#include "../../third_party/nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

namespace ie {

namespace {

// A zip entry whose name ends in `suffix`, through the central directory: torch.save writes STORED entries with data
// descriptors (flag 0x08), so the local headers carry zero sizes.
std::string zip_entry(const std::vector<uint8_t>& z, const std::string& suffix, std::vector<uint8_t>& out) {
    auto u16 = [&](size_t o) { return uint32_t(z[o]) | uint32_t(z[o + 1]) << 8; };
    auto u32 = [&](size_t o) { return u16(o) | u16(o + 2) << 16; };
    if (z.size() < 22) return "not a zip archive";
    const size_t lo = z.size() > 65557 ? z.size() - 65557 : 0;   // the end record sits within 22 + a 64 KiB comment
    size_t eocd = z.size() - 22;
    while (eocd > lo && u32(eocd) != 0x06054b50u) --eocd;
    if (u32(eocd) != 0x06054b50u) return "no zip end-of-central-directory record";
    size_t o = u32(eocd + 16);
    for (uint32_t i = 0, n = u16(eocd + 10); i < n; ++i) {
        if (o + 46 > z.size() || u32(o) != 0x02014b50u) return "bad zip central directory";
        const uint32_t method = u16(o + 10), csize = u32(o + 20), nlen = u16(o + 28), loc = u32(o + 42);
        if (o + 46 + nlen > z.size()) return "bad zip central directory";
        const std::string name(reinterpret_cast<const char*>(z.data() + o + 46), nlen);
        o += 46 + nlen + u16(o + 30) + u16(o + 32);
        if (name.size() < suffix.size() || name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
        if (method != 0) return "entry " + name + " is compressed (method " + std::to_string(method) + ")";
        if (csize == 0xFFFFFFFFu || size_t(loc) + 30 > z.size() || u32(loc) != 0x04034b50u) return "entry " + name + ": zip64 or a bad local header";
        const size_t data = size_t(loc) + 30 + u16(loc + 26) + u16(loc + 28);
        if (data + csize > z.size()) return "entry " + name + " is truncated";
        out.assign(z.begin() + std::ptrdiff_t(data), z.begin() + std::ptrdiff_t(data + csize));
        return {};
    }
    return "no entry *" + suffix;
}

// the raw storage of a torch.save'd bf16 tensor: the entry "<dir>/data/0", its pickle naming BFloat16Storage
std::string read_pt_storage(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "cannot open " + path;
    const std::vector<uint8_t> z((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    std::vector<uint8_t> pkl;
    if (auto e = zip_entry(z, "/data.pkl", pkl); !e.empty()) return path + ": " + e;
    if (std::string(pkl.begin(), pkl.end()).find("BFloat16Storage") == std::string::npos) return path + ": the stored tensor is not bf16";
    std::vector<uint8_t> bo;   // torch writes the storage byte order ("little"); older archives have no entry
    if (zip_entry(z, "/byteorder", bo).empty() && std::string(bo.begin(), bo.end()) != "little") return path + ": the storage is not little-endian";
    if (auto e = zip_entry(z, "/data/0", out); !e.empty()) return path + ": " + e;
    return {};
}

std::vector<float> bf16_rows(const SafeTensorInfo& t) {
    std::vector<float> v(size_t(t.numel()));
    mimo26_bf16_to_f32(t.data, uint32_t(v.size()), v.data());
    return v;
}

}  // namespace

Mimo26DFlash::~Mimo26DFlash() { free_all(); }

void Mimo26DFlash::free_all() {
    if (q_) { q_->wait(); for (void* p : owned_) sycl::free(p, *q_); }
    owned_.clear(); bytes_ = 0; L_.clear();
}

template <class T> T* Mimo26DFlash::dev(size_t n) {
    T* p = sycl::malloc_device<T>(n, *q_);
    if (p) { owned_.push_back(p); bytes_ += n * sizeof(T); }
    return p;
}

std::string Mimo26DFlash::load(const std::string& model_dir) {
    const std::string d = model_dir + "/dflash";
    std::ifstream cf(d + "/config.json");
    if (!cf) return "dflash: no " + d + "/config.json";
    const auto j = nlohmann::json::parse(cf, nullptr, false);
    if (j.is_discarded()) return "dflash: config.json does not parse";
    try {
        const auto& dc = j.at("dflash_config");
        cfg_.hidden = j.at("hidden_size"); cfg_.inter = j.at("intermediate_size"); cfg_.n_layers = j.at("num_hidden_layers");
        cfg_.n_q = j.at("num_attention_heads"); cfg_.n_kv = j.at("num_key_value_heads"); cfg_.head_dim = j.at("head_dim");
        cfg_.rope_dim = uint32_t(float(cfg_.head_dim) * j.at("partial_rotary_factor").get<float>());
        cfg_.block = dc.at("block_size"); cfg_.window = j.at("sliding_window"); cfg_.mask_id = dc.at("mask_token_id");
        cfg_.rope_theta = j.at("rope_theta"); cfg_.eps = j.at("rms_norm_eps"); cfg_.value_scale = dc.value("attention_value_scale", 1.0f);
        cfg_.target_layers = dc.at("target_layer_ids").get<std::vector<uint32_t>>();
    } catch (const std::exception& e) { return std::string("dflash config: ") + e.what(); }
    if (cfg_.head_dim != 128 || cfg_.n_q % cfg_.n_kv || cfg_.block < 2 || cfg_.block > 8)
        return "dflash: the draft kernels are built for head_dim 128 and a block of 2..8";
    if (auto e = file_.open(d + "/dflash_draft_model.safetensors"); !e.empty()) return "dflash: " + e;
    std::vector<uint8_t> raw;
    if (auto e = read_pt_storage(d + "/mask_embedding.pt", raw); !e.empty()) return "dflash: " + e;
    if (raw.size() != size_t(cfg_.hidden) * 2) return "dflash: mask_embedding.pt holds " + std::to_string(raw.size()) + " bytes, expected hidden x bf16";
    mask_emb_.resize(cfg_.hidden);
    mimo26_bf16_to_f32(raw.data(), cfg_.hidden, mask_emb_.data());   // (torch.save stores bf16 as its raw 2-byte words)
    return {};
}

std::string Mimo26DFlash::init(sycl::queue& q, const uint8_t* embed_bf16, uint32_t vocab, uint32_t max_ctx_rows) {
    free_all();
    q_ = &q; embed_ = embed_bf16; head_ = nullptr; vocab_ = vocab; max_rows_ = std::max<uint32_t>(max_ctx_rows, cfg_.block);
    R_ = (cfg_.window + max_rows_ + 63) / 64 * 64;
    const uint32_t H = cfg_.hidden, HD = cfg_.head_dim, NQ = cfg_.n_q * HD, NKV = cfg_.n_kv * HD, F = uint32_t(cfg_.target_layers.size()) * H;
    auto find = [&](const std::string& n) -> const SafeTensorInfo* { return file_.find(n); };
    auto up16 = [&](const std::string& n, size_t expect) -> sycl::half* {
        const SafeTensorInfo* t = find(n);
        if (!t || t->numel() != expect) return nullptr;
        const auto f = bf16_rows(*t);
        std::vector<sycl::half> h(f.size()); for (size_t i = 0; i < f.size(); ++i) h[i] = sycl::half(f[i]);
        sycl::half* p = dev<sycl::half>(h.size()); if (p) q.memcpy(p, h.data(), h.size() * 2).wait(); return p;
    };
    auto up32 = [&](const std::string& n, size_t expect) -> float* {
        const SafeTensorInfo* t = find(n);
        if (!t || t->numel() != expect) return nullptr;
        const auto f = bf16_rows(*t);
        float* p = dev<float>(f.size()); if (p) q.memcpy(p, f.data(), f.size() * 4).wait(); return p;
    };
    fc_ = up16("fc.weight", size_t(H) * F); hnorm_ = up32("hidden_norm.weight", H); fnorm_ = up32("norm.weight", H);
    if (!fc_ || !hnorm_ || !fnorm_) return "dflash: fc / hidden_norm / norm missing or misshapen";
    L_.resize(cfg_.n_layers);
    for (uint32_t l = 0; l < cfg_.n_layers; ++l) {
        const std::string p = "layers." + std::to_string(l) + ".";
        Layer& y = L_[l];
        y.q = up16(p + "self_attn.q_proj.weight", size_t(NQ) * H);   y.k = up16(p + "self_attn.k_proj.weight", size_t(NKV) * H);
        y.v = up16(p + "self_attn.v_proj.weight", size_t(NKV) * H);  y.o = up16(p + "self_attn.o_proj.weight", size_t(H) * NQ);
        y.gate = up16(p + "mlp.gate_proj.weight", size_t(cfg_.inter) * H); y.up = up16(p + "mlp.up_proj.weight", size_t(cfg_.inter) * H);
        y.down = up16(p + "mlp.down_proj.weight", size_t(H) * cfg_.inter);
        y.in_norm = up32(p + "input_layernorm.weight", H); y.post_norm = up32(p + "post_attention_layernorm.weight", H);
        y.q_norm = up32(p + "self_attn.q_norm.weight", HD); y.k_norm = up32(p + "self_attn.k_norm.weight", HD);
        y.sink = up32(p + "self_attn.attention_sink_bias", cfg_.n_q);
        y.kc = dev<sycl::half>(size_t(cfg_.n_kv) * R_ * HD); y.vc = dev<sycl::half>(size_t(cfg_.n_kv) * R_ * HD);
        if (!y.q || !y.k || !y.v || !y.o || !y.gate || !y.up || !y.down || !y.in_norm || !y.post_norm || !y.q_norm || !y.k_norm || !y.sink || !y.kc || !y.vc)
            return "dflash: layer " + std::to_string(l) + " weights missing, misshapen or not allocated";
        q.memset(y.kc, 0, size_t(cfg_.n_kv) * R_ * HD * 2); q.memset(y.vc, 0, size_t(cfg_.n_kv) * R_ * HD * 2);
    }
    const size_t M = max_rows_, B = cfg_.block;
    feat32_ = dev<float>(M * F); feat16_ = dev<sycl::half>(M * F); ctx32_ = dev<float>(M * H); ctx16_ = dev<sycl::half>(M * H);
    kv32_ = dev<float>(std::max(M, B) * NKV); kh_ = dev<sycl::half>(std::max(M, B) * NKV); vh_ = dev<sycl::half>(std::max(M, B) * NKV);
    x32_ = dev<float>(B * H); t32_ = dev<float>(B * std::max(H, NQ)); x16_ = dev<sycl::half>(B * H); q32_ = dev<float>(B * NQ); qh_ = dev<sycl::half>(B * NQ);
    att_ = dev<sycl::half>(B * NQ); g32_ = dev<float>(B * cfg_.inter); u32_ = dev<float>(B * cfg_.inter); h16_ = dev<sycl::half>(B * cfg_.inter);
    lg32_ = dev<float>(B * size_t(vocab)); pos_ = dev<int32_t>(std::max(M, B)); top_id_ = dev<int32_t>(B); top_p_ = dev<float>(B);
    if (!feat32_ || !feat16_ || !ctx32_ || !ctx16_ || !kv32_ || !kh_ || !vh_ || !x32_ || !t32_ || !x16_ || !q32_ || !qh_ || !att_ || !g32_ || !u32_ || !h16_ || !lg32_ || !pos_ || !top_id_ || !top_p_)
        return "dflash: workspace allocation failed";
    q.wait();
    reset();
    return {};
}

void Mimo26DFlash::rewind(uint32_t L) {
    if (L >= ctx_end_) return;
    ctx_end_ = L;
    if (hi_ > R_) ctx_lo_ = std::max(ctx_lo_, hi_ - R_);   // the slots of positions below hi_ - R_ hold newer rows
    ctx_lo_ = std::min(ctx_lo_, L);                        // (nothing left before L: the context restarts at L)
}

std::string Mimo26DFlash::add_context(const float* feats, uint32_t T, uint32_t pos0, uint32_t stride) {
    if (!q_ || L_.empty()) return "dflash: not initialised";
    if (T == 0) return {};
    if (stride == 0) stride = T;
    if (T > stride || stride > max_rows_) return "dflash: " + std::to_string(stride) + " context rows exceed " + std::to_string(max_rows_);
    if (pos0 < ctx_end_) rewind(pos0);
    if (pos0 > ctx_end_) ctx_lo_ = pos0;   // a gap: the context restarts here
    sycl::queue& q = *q_;
    const uint32_t H = cfg_.hidden, HD = cfg_.head_dim, NKV = cfg_.n_kv * HD, F = uint32_t(cfg_.target_layers.size()) * H;
    q.memcpy(feat32_, feats, size_t(stride) * F * 4);   // [n_target_layers][stride][H]
    { std::vector<int32_t> ps(T); for (uint32_t t = 0; t < T; ++t) ps[t] = int32_t(pos0 + t); q.memcpy(pos_, ps.data(), T * 4).wait(); }
    { const float* s = feat32_; sycl::half* d = feat16_; const uint32_t nf = uint32_t(cfg_.target_layers.size());
      q.parallel_for(sycl::range<3>(T, nf, H), [=](sycl::id<3> id) {   // -> rows [T][nf * H], fp16 for the fc GEMM
          const size_t t = id[0], f = id[1], c = id[2];
          d[(t * nf + f) * H + c] = sycl::half(s[(f * stride + t) * H + c]);
      }); }
    gemm_nt_f16_onednn(q, feat16_, fc_, ctx32_, T, H, F);
    mimo26_rms_norm(q, ctx32_, hnorm_, ctx16_, nullptr, T, H, cfg_.eps);
    const uint32_t R = R_, nkv = cfg_.n_kv;
    for (auto& y : L_) {
        gemm_nt_f16_onednn(q, ctx16_, y.k, kv32_, T, NKV, H);
        mimo26_rms_norm(q, kv32_, y.k_norm, kh_, nullptr, T * nkv, HD, cfg_.eps);   // k_norm per head
        rope_partial(q, kh_, pos_, kh_, T, nkv, HD, cfg_.rope_dim, cfg_.rope_theta);
        gemm_nt_f16_onednn(q, ctx16_, y.v, kv32_, T, NKV, H);
        { float* s = kv32_; sycl::half* d = vh_; q.parallel_for(sycl::range<1>(uint64_t(T) * NKV), [=](sycl::id<1> i) { d[i] = sycl::half(s[i]); }); }
        sycl::half *kc = y.kc, *vc = y.vc; const sycl::half *kh = kh_, *vh = vh_;
        q.parallel_for(sycl::range<2>(T, NKV), [=](sycl::id<2> id) {   // row t -> ring slot (pos0 + t) mod R, per kv head
            const uint32_t t = uint32_t(id[0]), c = uint32_t(id[1]), kvh = c / HD, dd = c % HD, slot = (pos0 + t) % R;
            kc[(size_t(kvh) * R + slot) * HD + dd] = kh[size_t(t) * NKV + c];
            vc[(size_t(kvh) * R + slot) * HD + dd] = vh[size_t(t) * NKV + c];
        });
    }
    q.wait();
    ctx_end_ = pos0 + T;
    hi_ = std::max(hi_, ctx_end_);
    if (hi_ > R_) ctx_lo_ = std::max(ctx_lo_, hi_ - R_);
    return {};
}

std::string Mimo26DFlash::draft(int32_t anchor, uint32_t k, std::vector<int32_t>& out, float min_p) {
    out.clear(); probs_.clear();
    if (!q_ || L_.empty()) return "dflash: not initialised";
    if (!head_) return "dflash: no lm_head attached (set_head)";
    const uint32_t B = cfg_.block;
    k = std::min(k, B - 1);
    if (k == 0) return {};
    if (anchor < 0 || uint32_t(anchor) >= vocab_) return "dflash: anchor id out of range";
    sycl::queue& q = *q_;
    const uint32_t H = cfg_.hidden, HD = cfg_.head_dim, NQ = cfg_.n_q * HD, NKV = cfg_.n_kv * HD, p = ctx_end_;
    // the block: the anchor's embedding, then the learned mask vector
    std::vector<float> x(size_t(B) * H);
    mimo26_bf16_to_f32(embed_ + size_t(anchor) * H * 2, H, x.data());
    for (uint32_t r = 1; r < B; ++r) std::copy(mask_emb_.begin(), mask_emb_.end(), x.begin() + std::ptrdiff_t(size_t(r) * H));
    q.memcpy(x32_, x.data(), x.size() * 4);
    { std::vector<int32_t> ps(B); for (uint32_t t = 0; t < B; ++t) ps[t] = int32_t(p + t); q.memcpy(pos_, ps.data(), B * 4).wait(); }
    const uint32_t nq = cfg_.n_q, nkv = cfg_.n_kv, gqa = nq / nkv, R = R_, c0 = std::max(ctx_lo_, p > cfg_.window ? p - cfg_.window : 0u);
    const float scale = 1.0f / std::sqrt(float(HD));
    for (auto& y : L_) {
        mimo26_rms_norm(q, x32_, y.in_norm, x16_, nullptr, B, H, cfg_.eps);
        gemm_nt_f16_onednn(q, x16_, y.q, q32_, B, NQ, H);
        mimo26_rms_norm(q, q32_, y.q_norm, qh_, nullptr, B * nq, HD, cfg_.eps);
        rope_partial(q, qh_, pos_, qh_, B, nq, HD, cfg_.rope_dim, cfg_.rope_theta);
        gemm_nt_f16_onednn(q, x16_, y.k, kv32_, B, NKV, H);
        mimo26_rms_norm(q, kv32_, y.k_norm, kh_, nullptr, B * nkv, HD, cfg_.eps);
        rope_partial(q, kh_, pos_, kh_, B, nkv, HD, cfg_.rope_dim, cfg_.rope_theta);
        gemm_nt_f16_onednn(q, x16_, y.v, kv32_, B, NKV, H);
        { float* s = kv32_; sycl::half* d = vh_; q.parallel_for(sycl::range<1>(uint64_t(B) * NKV), [=](sycl::id<1> i) { d[i] = sycl::half(s[i]); }); }
        // attention: a sub-group per (block row, q head) over the ring's context [c0, p) then the whole block (bidirectional),
        // online softmax, the sink folded into the denominator
        const sycl::half *qh = qh_, *kh = kh_, *vh = vh_, *kc = y.kc, *vc = y.vc; const float* sink = y.sink; sycl::half* att = att_; const bool use_sink = !ref_form_;
        ie::ps(q, "mimo26_dflash_attn", [&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<2>({B, size_t(nq) * 16}, {1, 16}), [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
                const uint32_t row = uint32_t(it.get_group(0)), hq = uint32_t(it.get_group(1)), lane = uint32_t(it.get_local_id(1)), kvh = hq / gqa;
                auto sg = it.get_sub_group();
                float qv[8], acc[8], m = -std::numeric_limits<float>::infinity(), l = 0.f;
                #pragma unroll
                for (int d = 0; d < 8; ++d) { qv[d] = float(qh[(size_t(row) * nq + hq) * 128 + lane * 8 + d]); acc[d] = 0.f; }
                auto step = [&](const sycl::half* kr, const sycl::half* vr) {
                    float part = 0.f;
                    #pragma unroll
                    for (int d = 0; d < 8; ++d) part += qv[d] * float(kr[d]);
                    const float s = sycl::reduce_over_group(sg, part, sycl::plus<float>()) * scale;
                    const float mn = sycl::fmax(m, s), a = sycl::native::exp(m - mn), e = sycl::native::exp(s - mn);
                    #pragma unroll
                    for (int d = 0; d < 8; ++d) acc[d] = acc[d] * a + e * float(vr[d]);
                    l = l * a + e; m = mn;
                };
                for (uint32_t ps = c0; ps < p; ++ps) {
                    const size_t off = (size_t(kvh) * R + ps % R) * 128 + lane * 8;
                    step(kc + off, vc + off);
                }
                for (uint32_t j = 0; j < B; ++j) {
                    const size_t off = (size_t(j) * nkv + kvh) * 128 + lane * 8;
                    step(kh + off, vh + off);
                }
                const float sv = use_sink ? sink[hq] : -std::numeric_limits<float>::infinity(), mf = sycl::fmax(m, sv);
                const float corr = sycl::native::exp(m - mf), ll = l * corr + sycl::native::exp(sv - mf);
                #pragma unroll
                for (int d = 0; d < 8; ++d) att[(size_t(row) * nq + hq) * 128 + lane * 8 + d] = sycl::half(acc[d] * corr / ll);
            });
        });
        gemm_nt_f16_onednn(q, att_, y.o, t32_, B, H, NQ);
        mimo26_axpy(q, t32_, ref_form_ ? 1.f : cfg_.value_scale, x32_, size_t(B) * H);
        mimo26_rms_norm(q, x32_, y.post_norm, x16_, nullptr, B, H, cfg_.eps);
        gemm_nt_f16_onednn(q, x16_, y.gate, g32_, B, cfg_.inter, H);
        gemm_nt_f16_onednn(q, x16_, y.up, u32_, B, cfg_.inter, H);
        mimo26_swiglu_f32(q, g32_, u32_, h16_, size_t(B) * cfg_.inter);
        gemm_nt_f16_onednn(q, h16_, y.down, t32_, B, H, cfg_.inter);
        mimo26_axpy(q, t32_, 1.f, x32_, size_t(B) * H);
    }
    mimo26_rms_norm(q, x32_, fnorm_, x16_, nullptr, B, H, cfg_.eps);
    gemm_nt_f16_onednn(q, x16_ + H, head_, lg32_, k, vocab_, H);   // rows 1..k: the drafts
    // per row on the device: the argmax (the first index of the maximum, as std::max_element) and its softmax probability
    {
        constexpr uint32_t WG = 1024;
        const float* lg = lg32_; int32_t* tid = top_id_; float* tp = top_p_; const uint32_t V = vocab_;
        ie::ps(q, "mimo26_dflash_argmax", [&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(size_t(k) * WG, WG), [=](sycl::nd_item<1> it) {
                const uint32_t r = uint32_t(it.get_group(0)), l = uint32_t(it.get_local_id(0));
                const float* row = lg + size_t(r) * V;
                float m = -std::numeric_limits<float>::infinity(); uint32_t mi = 0xFFFFFFFFu;
                for (uint32_t i = l; i < V; i += WG) if (row[i] > m) { m = row[i]; mi = i; }
                const float gm = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
                const uint32_t gi = sycl::reduce_over_group(it.get_group(), m == gm ? mi : 0xFFFFFFFFu, sycl::minimum<uint32_t>());
                float s = 0.f;
                for (uint32_t i = l; i < V; i += WG) s += sycl::exp(row[i] - gm);
                const float gs = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
                if (l == 0) { tid[r] = int32_t(gi); tp[r] = 1.f / gs; }
            });
        });
    }
    std::vector<int32_t> ids(k); probs_.resize(k);
    q.memcpy(ids.data(), top_id_, k * 4);
    q.memcpy(probs_.data(), top_p_, k * 4).wait();
    for (uint32_t r = 0; r < k; ++r) {
        if (ids[r] < 0 || uint32_t(ids[r]) >= vocab_) return "dflash: draft row " + std::to_string(r) + " has no finite maximum";
        if (probs_[r] < min_p) break;
        out.push_back(ids[r]);
    }
    return {};
}

}  // namespace ie
