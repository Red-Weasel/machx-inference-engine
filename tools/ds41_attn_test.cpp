// tools/ds41_attn_test.cpp — V4.1 Phase 5b: the attention block on device vs the reference.
//
// Golden from tools/ds41_reference/golden_block.py, which runs DeepSeek's own Attention(0) and
// dumps every stage boundary. Each stage is compared separately: a single end-to-end number
// would say something is wrong without saying where, which is the lesson Phase 4 paid for.
//
// The point of this phase is how little is new. q/kv projections, RMSNorm, RoPE (including the
// conjugate rotation on the output), the shared-KV attention with sinks, and even the
// block-diagonal wo_a are all kernels the V4 port already shipped.
#include "ie/deepseek4_attn.hpp"
#include "ie/deepseek41_upload.hpp"
#include "ie/ops.hpp"

#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
double g_worst = 0;

template <class T>
std::vector<T> read_bin(const std::string& p, size_t n) {
    std::vector<T> v(n);
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p.c_str()); std::exit(1); }
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(T)));
    if (size_t(f.gcount()) != n * sizeof(T)) { std::fprintf(stderr, "%s: short\n", p.c_str()); std::exit(1); }
    return v;
}

inline float bf16(uint16_t h) { const uint32_t b = uint32_t(h) << 16; float f; std::memcpy(&f, &b, 4); return f; }

// Compare a device fp32 buffer against a golden, reporting relative error against the
// golden's own scale. Tolerances are per-stage because error compounds down the chain.
void stage(sycl::queue& q, const char* name, const float* d, size_t n,
           const std::vector<float>& ref, double tol) {
    std::vector<float> got(n);
    q.memcpy(got.data(), d, n * 4).wait();
    double m = 0, scale = 0;
    size_t worst = 0;
    for (size_t i = 0; i < n; ++i) {
        const double e = std::fabs(double(got[i]) - double(ref[i]));
        if (e > m) { m = e; worst = i; }
        scale = std::max(scale, std::fabs(double(ref[i])));
    }
    const double rel = scale > 0 ? m / scale : 0.0;
    g_worst = std::max(g_worst, rel);
    const bool ok = rel <= tol;
    std::printf("%s %-16s rel %.3e  (max|diff| %.3e at %zu, ref scale %.4f, tol %.0e)\n",
                ok ? "[ ok ]" : "[FAIL]", name, rel, m, worst, scale, tol);
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
    const uint32_t T = 8, H = c.dim, QR = c.q_lora_rank, HD = c.head_dim;
    const uint32_t NH = c.n_heads, QH = NH * HD, RD = c.rope_head_dim;
    const uint32_t G = c.o_groups, OPG = c.o_lora_rank, IPG = QH / G, OR = G * OPG;

    sycl::device dev;
    for (const auto& p : sycl::platform::get_platforms())
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) { dev = d; goto found; }
    std::fprintf(stderr, "no Arc GPU\n"); return 1;
found:
    sycl::queue q(dev, sycl::property::queue::in_order{});
    std::printf("%s — layer 0 attention, T=%u, %u heads x %u (rope %u), o_groups %u x %u\n",
                dev.get_info<sycl::info::device::name>().c_str(), T, NH, HD, RD, G, OPG);

    // ---- dense FP8 weights -> fp16 on device ---------------------------------------------
    auto dense = [&](const ie::Ds41Tensor& t) {
        const uint32_t N = uint32_t(t.w->shape[0]), K = uint32_t(t.w->shape[1]);
        const uint32_t bn = N / uint32_t(t.s->shape[0]), bk = K / uint32_t(t.s->shape[1]);
        uint8_t* dw = sycl::malloc_device<uint8_t>(size_t(N) * K, q);
        uint8_t* ds = sycl::malloc_device<uint8_t>(t.s->nbytes, q);
        sycl::half* out = sycl::malloc_device<sycl::half>(size_t(N) * K, q);
        q.memcpy(dw, t.w->data, size_t(N) * K);
        q.memcpy(ds, t.s->data, t.s->nbytes).wait();
        ie::ds41_dense_dequant_f16(q, dw, ds, N, K, bn, bk, out).wait();
        sycl::free(dw, q); sycl::free(ds, q);
        return out;
    };
    sycl::half* w_qa = dense(L0.wq_a);
    sycl::half* w_qb = dense(L0.wq_b);
    sycl::half* w_kv = dense(L0.wkv);
    sycl::half* w_oa = dense(L0.wo_a);
    sycl::half* w_ob = dense(L0.wo_b);

    auto norm_w = [&](const ie::Ds41Tensor& t, uint32_t n) {
        const auto* s = reinterpret_cast<const uint16_t*>(t.w->data);
        std::vector<float> h(n);
        for (uint32_t i = 0; i < n; ++i) h[i] = bf16(s[i]);
        float* d = sycl::malloc_device<float>(n, q);
        q.memcpy(d, h.data(), n * 4).wait();
        return d;
    };
    float* n_q  = norm_w(L0.q_norm,  QR);
    float* n_kv = norm_w(L0.kv_norm, HD);
    float* sinks = sycl::malloc_device<float>(NH, q);
    q.memcpy(sinks, L0.attn_sink.w->data, NH * 4).wait();

    // ---- buffers --------------------------------------------------------------------------
    auto dev_f32 = [&](size_t n) { return sycl::malloc_device<float>(n, q); };
    auto dev_f16 = [&](size_t n) { return sycl::malloc_device<sycl::half>(n, q); };
    float* x   = dev_f32(size_t(T) * H);
    float* qr  = dev_f32(size_t(T) * QR);
    float* qrn = dev_f32(size_t(T) * QR);
    float* qq  = dev_f32(size_t(T) * QH);
    float* kv  = dev_f32(size_t(T) * HD);
    float* kvn = dev_f32(size_t(T) * HD);
    float* oo  = dev_f32(size_t(T) * QH);
    float* oa  = dev_f32(size_t(T) * OR);
    float* ob  = dev_f32(size_t(T) * H);
    sycl::half* x16 = dev_f16(size_t(T) * H);
    sycl::half* t16 = dev_f16(size_t(T) * std::max(std::max(QR, QH), OR));

    auto to_f16 = [&](const float* src, sycl::half* dst, size_t n) {
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { dst[i] = sycl::half(src[i]); }).wait();
    };

    auto g_in = read_bin<float>(gd + "/blk_attn_in.f32", size_t(T) * H);
    q.memcpy(x, g_in.data(), g_in.size() * 4).wait();
    to_f16(x, x16, size_t(T) * H);

    // ---- RoPE tables (layer 0 is compress_ratio 0 -> the "main" table: theta 10000, no YaRN)
    ie::Ds4RopeConfig rope;
    rope.theta = c.rope_theta;
    rope.yarn  = false;
    const auto inv = ie::ds4_rope_inv_freq(rope, RD);
    std::vector<int32_t> pos(T);
    for (uint32_t i = 0; i < T; ++i) pos[i] = int32_t(i);
    float* d_inv = dev_f32(inv.size());
    int32_t* d_pos = sycl::malloc_device<int32_t>(T, q);
    float* d_cos = dev_f32(size_t(T) * (RD / 2));
    float* d_sin = dev_f32(size_t(T) * (RD / 2));
    q.memcpy(d_inv, inv.data(), inv.size() * 4);
    q.memcpy(d_pos, pos.data(), T * 4).wait();
    ie::ds4_rope_cos_sin(q, d_inv, d_pos, d_cos, d_sin, T, RD / 2, 1.0f).wait();

    // ---- the block ------------------------------------------------------------------------
    ie::gemm_nt_f16_onednn(q, x16, w_qa, qr, T, QR, H).wait();
    ie::ds4_rms_norm(q, qr, n_q, qrn, T, QR, c.norm_eps).wait();
    stage(q, "qr (q_norm)", qrn, size_t(T) * QR, read_bin<float>(gd + "/attn_qr.f32", size_t(T) * QR), 3e-3);

    to_f16(qrn, t16, size_t(T) * QR);
    ie::gemm_nt_f16_onednn(q, t16, w_qb, qq, T, QH, QR).wait();
    ie::ds4_rope_apply(q, qq, d_cos, d_sin, qq, T, NH, HD, RD, +1.0f).wait();
    stage(q, "q (roped)", qq, size_t(T) * QH, read_bin<float>(gd + "/attn_q_roped.f32", size_t(T) * QH), 3e-3);

    ie::gemm_nt_f16_onednn(q, x16, w_kv, kv, T, HD, H).wait();
    ie::ds4_rms_norm(q, kv, n_kv, kvn, T, HD, c.norm_eps).wait();
    ie::ds4_rope_apply(q, kvn, d_cos, d_sin, kvn, T, 1, HD, RD, +1.0f).wait();
    stage(q, "kv (roped)", kvn, size_t(T) * HD, read_bin<float>(gd + "/attn_kv_roped.f32", size_t(T) * HD), 3e-3);

    float* mask = dev_f32(size_t(T) * T);
    ie::ds4_sliding_causal_mask(q, d_pos, mask, T, T, c.window_size).wait();
    ie::ds4_attention(q, qq, kvn, mask, sinks, oo, T, NH, HD, T, 1.0f / std::sqrt(float(HD))).wait();
    stage(q, "o (attention)", oo, size_t(T) * QH, read_bin<float>(gd + "/attn_o_raw.f32", size_t(T) * QH), 3e-3);

    // the conjugate rotation that removes the query's rotation from the output
    ie::ds4_rope_apply(q, oo, d_cos, d_sin, oo, T, NH, HD, RD, -1.0f).wait();
    stage(q, "o (unroped)", oo, size_t(T) * QH, read_bin<float>(gd + "/attn_o_unroped.f32", size_t(T) * QH), 3e-3);

    to_f16(oo, t16, size_t(T) * QH);
    ie::gemm_bmm_nt_f16_onednn(q, t16, w_oa, oa, T, G, IPG, OPG).wait();
    stage(q, "wo_a (grouped)", oa, size_t(T) * OR, read_bin<float>(gd + "/attn_wo_a_out.f32", size_t(T) * OR), 5e-3);

    to_f16(oa, t16, size_t(T) * OR);
    ie::gemm_nt_f16_onednn(q, t16, w_ob, ob, T, H, OR).wait();
    stage(q, "attn out", ob, size_t(T) * H, read_bin<float>(gd + "/blk_attn_out.f32", size_t(T) * H), 5e-3);

    for (void* p : {(void*)w_qa,(void*)w_qb,(void*)w_kv,(void*)w_oa,(void*)w_ob,(void*)n_q,(void*)n_kv,
                    (void*)sinks,(void*)x,(void*)qr,(void*)qrn,(void*)qq,(void*)kv,(void*)kvn,(void*)oo,
                    (void*)oa,(void*)ob,(void*)x16,(void*)t16,(void*)d_inv,(void*)d_pos,(void*)d_cos,
                    (void*)d_sin,(void*)mask})
        sycl::free(p, q);

    std::printf("\nworst stage %.3e — %s\n", g_worst,
                g_fail ? (std::to_string(g_fail) + " FAILURE(S)").c_str() : "ATTENTION TEST: PASS");
    return g_fail ? 1 : 0;
}
