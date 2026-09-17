// P2 gate: a WHOLE Inkling decoder layer (dense variant = blk.0/1), composed
// from the gated operators, against golden vectors from the reference layer
// (`InklingDecoderLayer.forward` over `InklingAttention.forward`,
// `InklingShortConvolution`, `InklingRMSNorm`, `InklingMLP` — transcribed
// verbatim in scripts/gen_inkling_layer_golden.py).
//
// WHY A LAYER-LEVEL GATE WHEN EVERY OP ALREADY PASSES.  Correct operators in
// the wrong ORDER produce a model that runs and is wrong.  This layer has four
// shortconv sites (k and v INSIDE attention; the attention output and the MLP
// output at the layer) and two residuals, and each residual attaches BEFORE its
// norm — attaching after, or putting a shortconv on the wrong side of a
// residual, is invisible to per-op tests.
#undef NDEBUG
#include "ie/inkling_ops.hpp"

#include <sycl/sycl.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

namespace {
bool rd(std::ifstream& f, void* p, size_t n) {
    f.read(reinterpret_cast<char*>(p), std::streamsize(n));
    return bool(f);
}
struct Dev {
    sycl::queue& q;
    float* alloc(size_t n) { return sycl::malloc_device<float>(n, q); }
    float* up(const std::vector<float>& v) {
        float* d = alloc(v.size());
        q.memcpy(d, v.data(), v.size() * 4).wait();
        return d;
    }
};
}  // namespace

int main() {
    std::ifstream f;
    for (const char* p : {"tests/data/inkling_layer_golden.bin",
                          "../tests/data/inkling_layer_golden.bin",
                          "../../tests/data/inkling_layer_golden.bin"}) {
        f.open(p, std::ios::binary);
        if (f) break;
        f.clear();
    }
    if (!f) { std::puts("inkling_layer_test: SKIPPED (golden blob not found)"); return 0; }
    uint32_t nc = 0;
    assert(rd(f, &nc, 4) && nc && nc <= 32);
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    Dev dev{q};
    int fails = 0;
    std::puts("Inkling decoder layer (dense) — composed from the gated ops");
    for (uint32_t ci = 0; ci < nc; ++ci) {
        uint32_t T, H, NH, KVH, HD, DR, K, EXT, FF;
        uint32_t hdr[9];
        assert(rd(f, hdr, sizeof(hdr)));
        T = hdr[0]; H = hdr[1]; NH = hdr[2]; KVH = hdr[3]; HD = hdr[4];
        DR = hdr[5]; K = hdr[6]; EXT = hdr[7]; FF = hdr[8];
        auto rdv = [&](size_t n) {
            std::vector<float> v(n);
            assert(rd(f, v.data(), n * 4));
            return v;
        };
        const auto x = rdv(size_t(T) * H);
        const auto w_in = rdv(H), w_post = rdv(H);
        const auto wq = rdv(size_t(NH) * HD * H), wk = rdv(size_t(KVH) * HD * H),
                   wv = rdv(size_t(KVH) * HD * H), wo = rdv(size_t(H) * NH * HD);
        const auto wr = rdv(size_t(NH) * DR * H);
        const auto qn = rdv(HD), kn = rdv(HD);
        const auto sc_k = rdv(size_t(KVH) * HD * K), sc_v = rdv(size_t(KVH) * HD * K),
                   sc_a = rdv(size_t(H) * K), sc_m = rdv(size_t(H) * K);
        const auto proj = rdv(size_t(DR) * EXT);
        const auto wg = rdv(size_t(FF) * H), wu = rdv(size_t(FF) * H), wd = rdv(size_t(H) * FF);
        const auto ref = rdv(size_t(T) * H);

        float *dx = dev.up(x), *dwin = dev.up(w_in), *dwpost = dev.up(w_post);
        float *dwq = dev.up(wq), *dwk = dev.up(wk), *dwv = dev.up(wv), *dwo = dev.up(wo);
        float *dwr = dev.up(wr), *dqn = dev.up(qn), *dkn = dev.up(kn);
        float *dsck = dev.up(sc_k), *dscv = dev.up(sc_v), *dsca = dev.up(sc_a),
              *dscm = dev.up(sc_m);
        float *dproj = dev.up(proj), *dwg = dev.up(wg), *dwu = dev.up(wu), *dwd = dev.up(wd);
        float *hn = dev.alloc(size_t(T) * H), *qbuf = dev.alloc(size_t(T) * NH * HD),
              *kbuf = dev.alloc(size_t(T) * KVH * HD), *vbuf = dev.alloc(size_t(T) * KVH * HD),
              *ktmp = dev.alloc(size_t(T) * KVH * HD), *vtmp = dev.alloc(size_t(T) * KVH * HD),
              *rel = dev.alloc(size_t(T) * NH * DR), *bias = dev.alloc(size_t(NH) * T * T),
              *att = dev.alloc(size_t(T) * NH * HD), *proj_o = dev.alloc(size_t(T) * H),
              *sc_out = dev.alloc(size_t(T) * H), *h1 = dev.alloc(size_t(T) * H),
              *hn2 = dev.alloc(size_t(T) * H), *gbuf = dev.alloc(size_t(T) * FF),
              *ubuf = dev.alloc(size_t(T) * FF), *sw = dev.alloc(size_t(T) * FF),
              *mlp = dev.alloc(size_t(T) * H), *mlp_sc = dev.alloc(size_t(T) * H),
              *yout = dev.alloc(size_t(T) * H);
        // causal mask [T,T] and positions
        std::vector<float> hmask(size_t(T) * T, 0.f);
        for (uint32_t i = 0; i < T; ++i)
            for (uint32_t j = i + 1; j < T; ++j)
                hmask[size_t(i) * T + j] = -std::numeric_limits<float>::infinity();
        std::vector<int32_t> hpos(T);
        for (uint32_t i = 0; i < T; ++i) hpos[i] = int32_t(i);
        float* dmask = dev.up(hmask);
        int32_t* dpos = sycl::malloc_device<int32_t>(T, q);
        q.memcpy(dpos, hpos.data(), T * 4).wait();

        // ---- the layer, in the reference's order ----
        ie::ink_rms_norm(q, dx, dwin, hn, T, H, 1e-6f);
        ie::ink_matmul_nt(q, hn, dwq, qbuf, T, H, NH * HD);
        ie::ink_matmul_nt(q, hn, dwk, ktmp, T, H, KVH * HD);
        ie::ink_matmul_nt(q, hn, dwv, vtmp, T, H, KVH * HD);
        ie::ink_shortconv(q, ktmp, dsck, kbuf, T, KVH * HD, K);   // k_sconv
        ie::ink_shortconv(q, vtmp, dscv, vbuf, T, KVH * HD, K);   // v_sconv
        ie::ink_rms_norm(q, qbuf, dqn, qbuf, T * NH, HD, 1e-6f);  // per-head q norm
        ie::ink_rms_norm(q, kbuf, dkn, kbuf, T * KVH, HD, 1e-6f); // per-head k norm
        ie::ink_matmul_nt(q, hn, dwr, rel, T, H, NH * DR);
        ie::ink_rel_logits(q, rel, dproj, dpos, dpos, bias, T, NH, DR, EXT, T);
        ie::ink_attention(q, qbuf, kbuf, vbuf, bias, dmask, att, T, NH, KVH, HD, T,
                          1.0f / std::sqrt(float(HD)));
        ie::ink_matmul_nt(q, att, dwo, proj_o, T, NH * HD, H);
        ie::ink_shortconv(q, proj_o, dsca, sc_out, T, H, K);      // attn_sconv
        ie::ink_add(q, dx, sc_out, h1, size_t(T) * H);            // residual #1
        ie::ink_rms_norm(q, h1, dwpost, hn2, T, H, 1e-6f);
        ie::ink_matmul_nt(q, hn2, dwg, gbuf, T, H, FF);
        ie::ink_matmul_nt(q, hn2, dwu, ubuf, T, H, FF);
        ie::ink_swiglu(q, gbuf, ubuf, sw, size_t(T) * FF);
        ie::ink_matmul_nt(q, sw, dwd, mlp, T, FF, H);
        ie::ink_shortconv(q, mlp, dscm, mlp_sc, T, H, K);         // mlp_sconv
        ie::ink_add(q, h1, mlp_sc, yout, size_t(T) * H);          // residual #2
        q.wait();

        std::vector<float> got(size_t(T) * H);
        q.memcpy(got.data(), yout, got.size() * 4).wait();
        double me = 0.0, mag = 0.0;
        for (size_t i = 0; i < ref.size(); ++i) {
            me = std::max(me, double(std::fabs(got[i] - ref[i])));
            mag = std::max(mag, double(std::fabs(ref[i])));
        }
        const bool ok = me <= 2e-5;
        std::printf("  T=%-3u H=%-4u heads=%u/%u head_dim=%u ffn=%-4u  max abs err %.3e "
                    "(|y|max %.2e)  %s\n", T, H, NH, KVH, HD, FF, me, mag, ok ? "OK" : "FAIL");
        if (!ok) ++fails;
        sycl::free(dpos, q);
    }
    // ---- §2 the MoE BLOCK (layers 2-41) -------------------------------------
    // `InklingMoE.forward`: routed_sum(x) + shared_sum(x, gammas), where BOTH
    // weight sets come from ONE router call whose log-softmax spans the k
    // routed experts AND the shared experts together.  Gating the block (not
    // just the router) is what proves the shared term is applied with the
    // ROUTER'S gammas rather than a fixed weight — the difference between this
    // architecture and every other MoE in the engine.
    std::ifstream mf;
    for (const char* p : {"tests/data/inkling_moe_golden.bin",
                          "../tests/data/inkling_moe_golden.bin",
                          "../../tests/data/inkling_moe_golden.bin"}) {
        mf.open(p, std::ios::binary);
        if (mf) break;
        mf.clear();
    }
    if (!mf) { std::puts("  §2 SKIPPED (moe blob not found)"); }
    else {
        uint32_t nm = 0;
        assert(rd(mf, &nm, 4) && nm && nm <= 32);
        std::puts("\nInkling MoE block (routed + gamma-weighted shared)");
        for (uint32_t ci = 0; ci < nm; ++ci) {
            uint32_t hd2[6];
            float rscale = 0.f, gscale = 0.f;
            assert(rd(mf, hd2, sizeof(hd2)) && rd(mf, &rscale, 4) && rd(mf, &gscale, 4));
            const uint32_t T = hd2[0], H = hd2[1], E = hd2[2], S = hd2[3], K = hd2[4],
                           FF = hd2[5];
            auto rdv2 = [&](size_t n) {
                std::vector<float> v(n);
                assert(rd(mf, v.data(), n * 4));
                return v;
            };
            const auto x = rdv2(size_t(T) * H);
            const auto wrt = rdv2(size_t(E + S) * H);
            const auto bi = rdv2(E);
            const auto gu = rdv2(size_t(E) * 2 * FF * H);
            const auto dn = rdv2(size_t(E) * H * FF);
            const auto shg = rdv2(size_t(S) * FF * H), shu = rdv2(size_t(S) * FF * H),
                       shd = rdv2(size_t(S) * H * FF);
            const auto ref = rdv2(size_t(T) * H);

            float* dx = dev.up(x);
            float* dwrt = dev.up(wrt);
            float* dbi = dev.up(bi);
            float* dlog = dev.alloc(size_t(T) * (E + S));
            int32_t* didx = sycl::malloc_device<int32_t>(size_t(T) * K, q);
            float* dw2 = dev.alloc(size_t(T) * K);
            float* dsh = dev.alloc(size_t(T) * S);
            ie::ink_matmul_nt(q, dx, dwrt, dlog, T, H, E + S);
            ie::ink_moe_router(q, dlog, dbi, didx, dw2, dsh, T, E, S, K, rscale, gscale);
            q.wait();
            std::vector<int32_t> idx(size_t(T) * K);
            std::vector<float> tw(size_t(T) * K), gam(size_t(T) * S);
            q.memcpy(idx.data(), didx, idx.size() * 4).wait();
            q.memcpy(tw.data(), dw2, tw.size() * 4).wait();
            q.memcpy(gam.data(), dsh, gam.size() * 4).wait();

            // Expert compute, one (token, slot) at a time — the reference's own
            // shape.  This is a CORRECTNESS harness, not the batched path.
            std::vector<float> acc(size_t(T) * H, 0.f);
            float* dg = dev.alloc(FF);
            float* du = dev.alloc(FF);
            float* dh = dev.alloc(FF);
            float* dxo = dev.alloc(H);
            for (uint32_t t = 0; t < T; ++t) {
                for (uint32_t r = 0; r < K; ++r) {
                    const uint32_t e = uint32_t(idx[size_t(t) * K + r]);
                    float* dgu = dev.up(std::vector<float>(
                        gu.begin() + size_t(e) * 2 * FF * H,
                        gu.begin() + size_t(e + 1) * 2 * FF * H));
                    float* ddn = dev.up(std::vector<float>(
                        dn.begin() + size_t(e) * H * FF, dn.begin() + size_t(e + 1) * H * FF));
                    ie::ink_matmul_nt(q, dx + size_t(t) * H, dgu, dg, 1, H, FF);
                    ie::ink_matmul_nt(q, dx + size_t(t) * H, dgu + size_t(FF) * H, du, 1, H, FF);
                    ie::ink_swiglu(q, dg, du, dh, FF);
                    ie::ink_matmul_nt(q, dh, ddn, dxo, 1, FF, H);
                    q.wait();
                    std::vector<float> o(H);
                    q.memcpy(o.data(), dxo, H * 4).wait();
                    for (uint32_t c = 0; c < H; ++c)
                        acc[size_t(t) * H + c] += o[c] * tw[size_t(t) * K + r];
                    sycl::free(dgu, q); sycl::free(ddn, q);
                }
            }
            // Shared experts, gamma-weighted INSIDE the activation.
            for (uint32_t sIdx = 0; sIdx < S; ++sIdx) {
                float* dsg = dev.up(std::vector<float>(
                    shg.begin() + size_t(sIdx) * FF * H, shg.begin() + size_t(sIdx + 1) * FF * H));
                float* dsu = dev.up(std::vector<float>(
                    shu.begin() + size_t(sIdx) * FF * H, shu.begin() + size_t(sIdx + 1) * FF * H));
                float* dsd = dev.up(std::vector<float>(
                    shd.begin() + size_t(sIdx) * H * FF, shd.begin() + size_t(sIdx + 1) * H * FF));
                for (uint32_t t = 0; t < T; ++t) {
                    ie::ink_matmul_nt(q, dx + size_t(t) * H, dsg, dg, 1, H, FF);
                    ie::ink_matmul_nt(q, dx + size_t(t) * H, dsu, du, 1, H, FF);
                    ie::ink_swiglu(q, dg, du, dh, FF);
                    q.wait();
                    std::vector<float> hv(FF);
                    q.memcpy(hv.data(), dh, FF * 4).wait();
                    const float gm = gam[size_t(t) * S + sIdx];
                    for (auto& z : hv) z *= gm;
                    q.memcpy(dh, hv.data(), FF * 4).wait();
                    ie::ink_matmul_nt(q, dh, dsd, dxo, 1, FF, H);
                    q.wait();
                    std::vector<float> o(H);
                    q.memcpy(o.data(), dxo, H * 4).wait();
                    for (uint32_t c = 0; c < H; ++c) acc[size_t(t) * H + c] += o[c];
                }
                sycl::free(dsg, q); sycl::free(dsu, q); sycl::free(dsd, q);
            }
            double me = 0.0;
            for (size_t i = 0; i < ref.size(); ++i)
                me = std::max(me, double(std::fabs(acc[i] - ref[i])));
            const bool ok = me <= 5e-5;
            std::printf("  T=%-3u H=%-4u routed=%-3u shared=%u k=%u ffn=%-4u  max abs err %.3e  %s\n",
                        T, H, E, S, K, FF, me, ok ? "OK" : "FAIL");
            if (!ok) ++fails;
            for (void* p : {(void*)dx,(void*)dwrt,(void*)dbi,(void*)dlog,(void*)didx,
                            (void*)dw2,(void*)dsh,(void*)dg,(void*)du,(void*)dh,(void*)dxo})
                sycl::free(p, q);
        }
    }

    if (fails) { std::printf("inkling_layer_test: FAILED (%d)\n", fails); return 1; }
    std::puts("inkling_layer_test: OK — the whole layer matches the reference, so the "
              "op ORDER, both residuals and all four shortconv sites are right");
    return 0;
}
