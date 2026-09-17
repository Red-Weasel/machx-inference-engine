// tools/ds41_block_test.cpp — V4.1 Phase 5c: a WHOLE layer against DeepSeek's own Block(0).
//
// Assembles the pieces the earlier phases verified separately — mHC (Phase 5b addendum),
// attention (5b), routed experts (5a) — plus the shared expert, and compares against the
// reference at every mHC boundary as well as at the block output.
//
// Stated plainly: the MoE gate runs on the HOST here. It is 8 tokens x [384, 5120] and a device
// routing kernel is not what this phase is testing; Phase 5a already proved the engine
// reproduces the reference's routing exactly, so the risk it covers is already retired.
#include "ie/deepseek4.hpp"
#include "ie/deepseek4_attn.hpp"
#include "ie/deepseek4_experts.hpp"
#include "ie/deepseek41_upload.hpp"
#include "ie/ops.hpp"

#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
double g_worst = 0;

template <class T> std::vector<T> rd(const std::string& p, size_t n) {
    std::vector<T> v(n);
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p.c_str()); std::exit(1); }
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(T)));
    return v;
}
inline float bf16(uint16_t h) { const uint32_t b = uint32_t(h) << 16; float f; std::memcpy(&f, &b, 4); return f; }

void stage(sycl::queue& q, const char* name, const float* d, size_t n,
           const std::vector<float>& ref, double tol) {
    std::vector<float> got(n);
    q.memcpy(got.data(), d, n * 4).wait();
    double m = 0, scale = 0;
    for (size_t i = 0; i < n; ++i) {
        m = std::max(m, std::fabs(double(got[i]) - double(ref[i])));
        scale = std::max(scale, std::fabs(double(ref[i])));
    }
    const double rel = scale > 0 ? m / scale : 0.0;
    g_worst = std::max(g_worst, rel);
    const bool ok = rel <= tol;
    std::printf("%s %-16s rel %.3e  (max|diff| %.3e, ref scale %.4f, tol %.0e)\n",
                ok ? "[ ok ]" : "[FAIL]", name, rel, m, scale, tol);
    if (!ok) ++g_fail;
}
}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : std::string(std::getenv("HOME")) + "/models/DeepSeek-V4.1-Flash";
    const std::string gd  = argc > 2 ? argv[2] : "/tmp/ds41_golden";

    ie::DeepSeek41Model m;
    if (const auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    const auto& c = m.config();
    const auto& L0 = m.layers()[0];
    const uint32_t T = 8, H = c.dim, HC = c.hc_mult, QR = c.q_lora_rank, HD = c.head_dim;
    const uint32_t NH = c.n_heads, QH = NH * HD, RD = c.rope_head_dim;
    const uint32_t G = c.o_groups, OPG = c.o_lora_rank, IPG = QH / G, OR = G * OPG;
    const uint32_t EF = c.moe_inter_dim, E = c.n_routed_experts, TK = c.n_activated_experts;
    const uint32_t MIX = (2 + HC) * HC, FLAT = HC * H;

    sycl::device dev;
    for (const auto& p : sycl::platform::get_platforms())
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) { dev = d; goto found; }
    std::fprintf(stderr, "no Arc GPU\n"); return 1;
found:
    sycl::queue q(dev, sycl::property::queue::in_order{});
    std::printf("%s — layer 0 FULL BLOCK, T=%u, hc_mult=%u\n",
                dev.get_info<sycl::info::device::name>().c_str(), T, HC);

    auto f32 = [&](size_t n) { return sycl::malloc_device<float>(n, q); };
    auto f16 = [&](size_t n) { return sycl::malloc_device<sycl::half>(n, q); };
    auto to16 = [&](const float* s, sycl::half* d, size_t n) {
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { d[i] = sycl::half(s[i]); }).wait();
    };
    auto dense = [&](const ie::Ds41Tensor& t) {
        const uint32_t N = uint32_t(t.w->shape[0]), K = uint32_t(t.w->shape[1]);
        const uint32_t bn = N / uint32_t(t.s->shape[0]), bk = K / uint32_t(t.s->shape[1]);
        uint8_t* dw = sycl::malloc_device<uint8_t>(size_t(N) * K, q);
        uint8_t* ds = sycl::malloc_device<uint8_t>(t.s->nbytes, q);
        sycl::half* o = f16(size_t(N) * K);
        q.memcpy(dw, t.w->data, size_t(N) * K);
        q.memcpy(ds, t.s->data, t.s->nbytes).wait();
        ie::ds41_dense_dequant_f16(q, dw, ds, N, K, bn, bk, o).wait();
        sycl::free(dw, q); sycl::free(ds, q);
        return o;
    };
    auto upf32 = [&](const std::vector<float>& v) {
        float* d = f32(v.size()); q.memcpy(d, v.data(), v.size() * 4).wait(); return d;
    };
    auto bf16_vec = [&](const ie::Ds41Tensor& t, size_t n) {
        const auto* s = reinterpret_cast<const uint16_t*>(t.w->data);
        std::vector<float> h(n);
        for (size_t i = 0; i < n; ++i) h[i] = bf16(s[i]);
        return h;
    };

    // ---- weights -------------------------------------------------------------------------
    sycl::half *w_qa = dense(L0.wq_a), *w_qb = dense(L0.wq_b), *w_kv = dense(L0.wkv);
    sycl::half *w_oa = dense(L0.wo_a), *w_ob = dense(L0.wo_b);
    sycl::half *s_w1 = dense(L0.sh_w1), *s_w3 = dense(L0.sh_w3), *s_w2 = dense(L0.sh_w2);
    float* n_q  = upf32(bf16_vec(L0.q_norm,  QR));
    float* n_kv = upf32(bf16_vec(L0.kv_norm, HD));
    float* n_at = upf32(bf16_vec(L0.attn_norm, H));
    float* n_ff = upf32(bf16_vec(L0.ffn_norm, H));
    float* sinks = f32(NH);
    q.memcpy(sinks, L0.attn_sink.w->data, NH * 4).wait();
    auto hc_t = [&](const ie::Ds41Tensor& t, size_t n) {
        float* d = f32(n); q.memcpy(d, t.w->data, n * 4).wait(); return d;
    };
    float* a_fn = hc_t(L0.hc_attn_fn, size_t(MIX) * FLAT);
    float* a_bs = hc_t(L0.hc_attn_base, MIX);
    float* a_sc = hc_t(L0.hc_attn_scale, 3);
    float* f_fn = hc_t(L0.hc_ffn_fn, size_t(MIX) * FLAT);
    float* f_bs = hc_t(L0.hc_ffn_base, MIX);
    float* f_sc = hc_t(L0.hc_ffn_scale, 3);

    ie::DS4ExpertBank bg, bu, bd;
    if (auto e = ie::ds41_expert_bank_upload(q, L0.exp_w1, H,  EF, E, bg); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
    if (auto e = ie::ds41_expert_bank_upload(q, L0.exp_w3, H,  EF, E, bu); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
    if (auto e = ie::ds41_expert_bank_upload(q, L0.exp_w2, EF, H,  E, bd); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
    ie::DS4ExpertWorkspace ws;
    if (auto e = ie::ds4_expert_ws_alloc(q, H, EF, ws); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }

    // ---- buffers -------------------------------------------------------------------------
    float *h0 = f32(size_t(T) * FLAT), *st1 = f32(size_t(T) * FLAT), *st2 = f32(size_t(T) * FLAT);
    float *a_pre = f32(size_t(T) * HC), *a_post = f32(size_t(T) * HC), *a_comb = f32(size_t(T) * HC * HC);
    float *fp_ = f32(size_t(T) * HC), *fq_ = f32(size_t(T) * HC), *fc_ = f32(size_t(T) * HC * HC);
    float *xa = f32(size_t(T) * H), *xn = f32(size_t(T) * H), *xf = f32(size_t(T) * H), *xfn = f32(size_t(T) * H);
    float *qr = f32(size_t(T) * QR), *qrn = f32(size_t(T) * QR), *qq = f32(size_t(T) * QH);
    float *kv = f32(size_t(T) * HD), *kvn = f32(size_t(T) * HD), *oo = f32(size_t(T) * QH);
    float *oa = f32(size_t(T) * OR), *ao = f32(size_t(T) * H);
    float *moe = f32(size_t(T) * H), *shg = f32(size_t(T) * EF), *shu = f32(size_t(T) * EF), *shh = f32(size_t(T) * H);
    sycl::half *x16 = f16(size_t(T) * H), *t16 = f16(size_t(T) * std::max({QR, QH, OR, EF}));
    float* pre_mix = f32(size_t(T) * HC);
    float* mask = f32(size_t(T) * T);

    auto g_in = rd<float>(gd + "/blk_in.f32", size_t(T) * FLAT);
    q.memcpy(h0, g_in.data(), g_in.size() * 4).wait();
    auto g_pm = rd<float>(gd + "/blk_pre_mix.f32", size_t(T) * HC);
    q.memcpy(pre_mix, g_pm.data(), g_pm.size() * 4).wait();

    ie::Ds4RopeConfig rope; rope.theta = c.rope_theta; rope.yarn = false;
    const auto inv = ie::ds4_rope_inv_freq(rope, RD);
    std::vector<int32_t> pos(T); std::iota(pos.begin(), pos.end(), 0);
    float* d_inv = upf32(inv);
    int32_t* d_pos = sycl::malloc_device<int32_t>(T, q);
    q.memcpy(d_pos, pos.data(), T * 4).wait();
    float *d_cos = f32(size_t(T) * (RD / 2)), *d_sin = f32(size_t(T) * (RD / 2));
    ie::ds4_rope_cos_sin(q, d_inv, d_pos, d_cos, d_sin, T, RD / 2, 1.0f).wait();

    // ---- the block -----------------------------------------------------------------------
    ie::ds41_hc_mixes(q, h0, a_fn, a_bs, a_sc, a_pre, a_post, a_comb, T, H, HC,
                      c.hc_sinkhorn_iters, c.norm_eps, c.hc_eps).wait();
    stage(q, "hc attn_pre",  a_pre,  size_t(T) * HC, rd<float>(gd + "/blk_attn_pre.f32", size_t(T) * HC), 1e-5);
    stage(q, "hc attn_post", a_post, size_t(T) * HC, rd<float>(gd + "/blk_attn_post.f32", size_t(T) * HC), 1e-5);
    stage(q, "hc attn_comb", a_comb, size_t(T) * HC * HC, rd<float>(gd + "/blk_attn_comb.f32", size_t(T) * HC * HC), 1e-5);

    ie::ds41_hc_collapse(q, h0, pre_mix, xa, T, H, HC).wait();
    ie::ds4_rms_norm(q, xa, n_at, xn, T, H, c.norm_eps).wait();
    stage(q, "attn_norm in", xn, size_t(T) * H, rd<float>(gd + "/blk_attn_in.f32", size_t(T) * H), 1e-5);

    to16(xn, x16, size_t(T) * H);
    ie::gemm_nt_f16_onednn(q, x16, w_qa, qr, T, QR, H).wait();
    ie::ds4_rms_norm(q, qr, n_q, qrn, T, QR, c.norm_eps).wait();
    to16(qrn, t16, size_t(T) * QR);
    ie::gemm_nt_f16_onednn(q, t16, w_qb, qq, T, QH, QR).wait();
    ie::ds4_rope_apply(q, qq, d_cos, d_sin, qq, T, NH, HD, RD, +1.0f).wait();
    ie::gemm_nt_f16_onednn(q, x16, w_kv, kv, T, HD, H).wait();
    ie::ds4_rms_norm(q, kv, n_kv, kvn, T, HD, c.norm_eps).wait();
    ie::ds4_rope_apply(q, kvn, d_cos, d_sin, kvn, T, 1, HD, RD, +1.0f).wait();
    ie::ds4_sliding_causal_mask(q, d_pos, mask, T, T, c.window_size).wait();
    ie::ds4_attention(q, qq, kvn, mask, sinks, oo, T, NH, HD, T, 1.0f / std::sqrt(float(HD))).wait();
    ie::ds4_rope_apply(q, oo, d_cos, d_sin, oo, T, NH, HD, RD, -1.0f).wait();
    to16(oo, t16, size_t(T) * QH);
    ie::gemm_bmm_nt_f16_onednn(q, t16, w_oa, oa, T, G, IPG, OPG).wait();
    to16(oa, t16, size_t(T) * OR);
    ie::gemm_nt_f16_onednn(q, t16, w_ob, ao, T, H, OR).wait();
    stage(q, "attn out", ao, size_t(T) * H, rd<float>(gd + "/blk_attn_out.f32", size_t(T) * H), 5e-3);

    ie::ds4_hc_mix(q, h0, a_post, a_comb, ao, st1, T, H, HC).wait();

    ie::ds41_hc_mixes(q, st1, f_fn, f_bs, f_sc, fp_, fq_, fc_, T, H, HC,
                      c.hc_sinkhorn_iters, c.norm_eps, c.hc_eps).wait();
    stage(q, "hc ffn_pre", fp_, size_t(T) * HC, rd<float>(gd + "/blk_ffn_pre.f32", size_t(T) * HC), 5e-3);
    stage(q, "hc ffn_post", fq_, size_t(T) * HC, rd<float>(gd + "/blk_ffn_post.f32", size_t(T) * HC), 5e-3);
    stage(q, "hc ffn_comb", fc_, size_t(T) * HC * HC, rd<float>(gd + "/blk_ffn_comb.f32", size_t(T) * HC * HC), 5e-3);

    ie::ds41_hc_collapse(q, st1, a_pre, xf, T, H, HC).wait();
    ie::ds4_rms_norm(q, xf, n_ff, xfn, T, H, c.norm_eps).wait();
    stage(q, "ffn_norm in", xfn, size_t(T) * H, rd<float>(gd + "/blk_ffn_in.f32", size_t(T) * H), 5e-3);

    // routing on the host (see the header note), then the engine's routed-expert block
    std::vector<float> hx(size_t(T) * H);
    q.memcpy(hx.data(), xfn, hx.size() * 4).wait();
    const auto* GW = reinterpret_cast<const uint16_t*>(L0.gate_w.w->data);
    const auto* GB = reinterpret_cast<const float*>(L0.gate_bias.w->data);
    std::vector<int32_t> idx(size_t(T) * TK);
    std::vector<float>   wgt(size_t(T) * TK);
    for (uint32_t t = 0; t < T; ++t) {
        std::vector<float> sc(E);
        for (uint32_t e = 0; e < E; ++e) {
            double acc = 0;
            for (uint32_t k = 0; k < H; ++k) acc += double(hx[size_t(t) * H + k]) * bf16(GW[size_t(e) * H + k]);
            sc[e] = std::sqrt(std::log1p(std::exp(-std::fabs(acc))) + std::max(acc, 0.0));
        }
        std::vector<uint32_t> ord(E); std::iota(ord.begin(), ord.end(), 0u);
        std::partial_sort(ord.begin(), ord.begin() + TK, ord.end(),
                          [&](uint32_t a, uint32_t b) { return sc[a] + GB[a] > sc[b] + GB[b]; });
        double sum = 0;
        for (uint32_t k = 0; k < TK; ++k) sum += sc[ord[k]];
        for (uint32_t k = 0; k < TK; ++k) {
            idx[size_t(t) * TK + k] = int32_t(ord[k]);
            wgt[size_t(t) * TK + k] = float(sc[ord[k]] / (sum + 1e-20) * c.route_scale);
        }
    }
    if (auto e = ie::ds4_experts_forward(q, bg, bu, bd, xfn, idx.data(), wgt.data(), moe,
                                         T, H, EF, TK, c.swiglu_limit, ws, true); !e.empty()) {
        std::fprintf(stderr, "experts: %s\n", e.c_str()); return 1;
    }

    // shared expert: SwiGLU with the same asymmetric clamp, then accumulate
    to16(xfn, x16, size_t(T) * H);
    ie::gemm_nt_f16_onednn(q, x16, s_w1, shg, T, EF, H).wait();
    ie::gemm_nt_f16_onednn(q, x16, s_w3, shu, T, EF, H).wait();
    {
        const float L = c.swiglu_limit;
        float* g_ = shg; float* u_ = shu;
        q.parallel_for(sycl::range<1>(size_t(T) * EF), [=](sycl::id<1> i) {
            float gg = g_[i], uu = u_[i];
            if (L > 0.f) { gg = sycl::fmin(gg, L); uu = sycl::fmax(sycl::fmin(uu, L), -L); }
            g_[i] = (gg / (1.0f + sycl::exp(-gg))) * uu;
        }).wait();
    }
    to16(shg, t16, size_t(T) * EF);
    ie::gemm_nt_f16_onednn(q, t16, s_w2, shh, T, H, EF).wait();
    {
        float* a_ = moe; const float* b_ = shh;
        q.parallel_for(sycl::range<1>(size_t(T) * H), [=](sycl::id<1> i) { a_[i] += b_[i]; }).wait();
    }
    stage(q, "moe out", moe, size_t(T) * H, rd<float>(gd + "/blk_moe_out.f32", size_t(T) * H), 5e-3);

    ie::ds4_hc_mix(q, st1, fq_, fc_, moe, st2, T, H, HC).wait();
    stage(q, "BLOCK OUT", st2, size_t(T) * FLAT, rd<float>(gd + "/blk_out.f32", size_t(T) * FLAT), 5e-3);

    // teardown: everything this test allocated (5b's criterion 7, which 5c had not carried)
    ie::ds4_expert_ws_free(q, ws);
    ie::ds4_expert_bank_free(q, bg); ie::ds4_expert_bank_free(q, bu); ie::ds4_expert_bank_free(q, bd);
    for (void* p : {(void*)w_qa,(void*)w_qb,(void*)w_kv,(void*)w_oa,(void*)w_ob,(void*)s_w1,(void*)s_w3,(void*)s_w2,
                    (void*)n_q,(void*)n_kv,(void*)n_at,(void*)n_ff,(void*)sinks,(void*)a_fn,(void*)a_bs,(void*)a_sc,
                    (void*)f_fn,(void*)f_bs,(void*)f_sc,(void*)h0,(void*)st1,(void*)st2,(void*)a_pre,(void*)a_post,
                    (void*)a_comb,(void*)fp_,(void*)fq_,(void*)fc_,(void*)xa,(void*)xn,(void*)xf,(void*)xfn,(void*)qr,
                    (void*)qrn,(void*)qq,(void*)kv,(void*)kvn,(void*)oo,(void*)oa,(void*)ao,(void*)moe,(void*)shg,
                    (void*)shu,(void*)shh,(void*)x16,(void*)t16,(void*)pre_mix,(void*)mask,(void*)d_inv,(void*)d_pos,
                    (void*)d_cos,(void*)d_sin})
        sycl::free(p, q);

    std::printf("\nworst stage %.3e — %s\n", g_worst,
                g_fail ? (std::to_string(g_fail) + " FAILURE(S)").c_str() : "BLOCK TEST: PASS");
    return g_fail ? 1 : 0;
}
