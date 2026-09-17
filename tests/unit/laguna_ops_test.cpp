// Gate for lag_attn_gate against golden vectors produced by the REFERENCE
// gating block (upstream modeling_laguna.py), not a re-derivation.
// Generator: scripts/gen_laguna_gate_golden.py.
//
// The cases deliberately include gate logits of +120, -120 and 88.5: naive
// softplus (`log(1+exp(x))`) returns inf past x ~ 88 in fp32, and an inf gate
// silently zeroes-or-destroys a whole head.  A negative control at the end
// proves the bound discriminates.
#undef NDEBUG
#include "ie/laguna_ops.hpp"
#include "ie/inkling_ops.hpp"
#include "ie/deepseek4_attn.hpp"

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
}  // namespace

int main() {
    std::ifstream f;
    for (const char* p : {"tests/data/laguna_gate_golden.bin",
                          "../tests/data/laguna_gate_golden.bin",
                          "../../tests/data/laguna_gate_golden.bin"}) {
        f.open(p, std::ios::binary);
        if (f) break;
        f.clear();
    }
    if (!f) { std::puts("laguna_ops_test: SKIPPED (golden blob not found)"); return 0; }
    uint32_t nc = 0;
    assert(rd(f, &nc, 4) && nc && nc <= 64);
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    int fails = 0;
    std::puts("lag_attn_gate vs LagunaAttention.forward gating block");
    for (uint32_t ci = 0; ci < nc; ++ci) {
        uint32_t T = 0, H = 0, D = 0;
        assert(rd(f, &T, 4) && rd(f, &H, 4) && rd(f, &D, 4));
        std::vector<float> a(size_t(T) * H * D), gl(size_t(T) * H), ref(a.size());
        assert(rd(f, a.data(), a.size() * 4));
        assert(rd(f, gl.data(), gl.size() * 4));
        assert(rd(f, ref.data(), ref.size() * 4));
        float* da = sycl::malloc_device<float>(a.size(), q);
        float* dg = sycl::malloc_device<float>(gl.size(), q);
        q.memcpy(da, a.data(), a.size() * 4).wait();
        q.memcpy(dg, gl.data(), gl.size() * 4).wait();
        ie::lag_attn_gate(q, da, dg, T, H, D).wait();
        std::vector<float> got(a.size());
        q.memcpy(got.data(), da, got.size() * 4).wait();
        double me = 0.0;
        bool nonfinite = false;
        for (size_t i = 0; i < ref.size(); ++i) {
            if (!std::isfinite(got[i])) nonfinite = true;
            me = std::max(me, double(std::fabs(got[i] - ref[i])));
        }
        const bool ok = !nonfinite && me <= 1e-4;
        std::printf("  T=%-3u heads=%-3u head_dim=%-4u  max abs err %.3e%s  %s\n",
                    T, H, D, me, nonfinite ? "  NON-FINITE OUTPUT" : "", ok ? "OK" : "FAIL");
        if (!ok) ++fails;
        // NEGATIVE CONTROL on the last case: apply the gate of the WRONG head
        // (rotate by one).  Per-head broadcasting is the whole contract, so the
        // bound must reject a head-index slip.
        if (ci + 1 == nc && H >= 2) {
            std::vector<float> rot(gl.size());
            for (uint32_t t = 0; t < T; ++t)
                for (uint32_t hh = 0; hh < H; ++hh)
                    rot[size_t(t) * H + (hh + 1) % H] = gl[size_t(t) * H + hh];
            q.memcpy(da, a.data(), a.size() * 4).wait();
            q.memcpy(dg, rot.data(), rot.size() * 4).wait();
            ie::lag_attn_gate(q, da, dg, T, H, D).wait();
            q.memcpy(got.data(), da, got.size() * 4).wait();
            double e2 = 0.0;
            for (size_t i = 0; i < ref.size(); ++i)
                e2 = std::max(e2, double(std::fabs(got[i] - ref[i])));
            std::printf("  control: gates rotated by one head -> max abs err %.3e  %s\n",
                        e2, e2 > 1e-4 ? "REJECTED (bound discriminates)"
                                      : "NOT REJECTED (bound is useless)");
            if (!(e2 > 1e-4)) ++fails;
        }
        sycl::free(da, q); sycl::free(dg, q);
    }
    // ---- §2 REUSE CLAIM: does `ink_attention` serve Laguna unchanged? -------
    // docs/laguna/00_PORT_PLAN.md asserts the Inkling attention kernel is
    // directly reusable here (same signature, bias argument null, Laguna's own
    // GQA shapes).  That claim is load-bearing for the port estimate, so it is
    // MEASURED rather than assumed: the goldens below use Laguna's real
    // per-layer geometry — 48 and 72 query heads over 8 KV heads (n_rep 6 and
    // 9), head_dim 128 — with a causal 512-wide sliding-window mask.
    std::ifstream af;
    for (const char* p : {"tests/data/laguna_attn_golden.bin",
                          "../tests/data/laguna_attn_golden.bin",
                          "../../tests/data/laguna_attn_golden.bin"}) {
        af.open(p, std::ios::binary);
        if (af) break;
        af.clear();
    }
    if (!af) { std::puts("  §2 SKIPPED (laguna attn blob not found)"); }
    else {
        uint32_t nc2 = 0;
        assert(rd(af, &nc2, 4) && nc2 && nc2 <= 64);
        std::puts("\n§2 ink_attention REUSED at Laguna's shapes (GQA 48/8 and 72/8, SWA mask)");
        for (uint32_t ci = 0; ci < nc2; ++ci) {
            uint32_t T = 0, H = 0, KVH = 0, D = 0, KV = 0;
            assert(rd(af, &T, 4) && rd(af, &H, 4) && rd(af, &KVH, 4) && rd(af, &D, 4) &&
                   rd(af, &KV, 4));
            std::vector<float> hq(size_t(T)*H*D), hk(size_t(KV)*KVH*D), hv(hk.size()),
                               hm(size_t(T)*KV), ref(hq.size());
            assert(rd(af, hq.data(), hq.size()*4));
            assert(rd(af, hk.data(), hk.size()*4));
            assert(rd(af, hv.data(), hv.size()*4));
            assert(rd(af, hm.data(), hm.size()*4));
            assert(rd(af, ref.data(), ref.size()*4));
            float* dq = sycl::malloc_device<float>(hq.size(), q);
            float* dk = sycl::malloc_device<float>(hk.size(), q);
            float* dv = sycl::malloc_device<float>(hv.size(), q);
            float* dm = sycl::malloc_device<float>(hm.size(), q);
            float* dout = sycl::malloc_device<float>(ref.size(), q);
            q.memcpy(dq, hq.data(), hq.size()*4).wait();
            q.memcpy(dk, hk.data(), hk.size()*4).wait();
            q.memcpy(dv, hv.data(), hv.size()*4).wait();
            q.memcpy(dm, hm.data(), hm.size()*4).wait();
            // bias = nullptr: Laguna has no relative-bias term (and no sinks).
            ie::ink_attention(q, dq, dk, dv, nullptr, dm, dout, T, H, KVH, D, KV,
                              1.0f / std::sqrt(float(D))).wait();
            std::vector<float> got(ref.size());
            q.memcpy(got.data(), dout, got.size()*4).wait();
            double me = 0.0;
            for (size_t i = 0; i < ref.size(); ++i)
                me = std::max(me, double(std::fabs(got[i] - ref[i])));
            const bool ok = me <= 1e-5;
            std::printf("  T=%-3u heads=%-3u kv=%u (n_rep %u) D=%u KV=%-4u  max abs err %.3e  %s\n",
                        T, H, KVH, H/KVH, D, KV, me, ok ? "OK — reusable" : "FAIL");
            if (!ok) ++fails;
            for (void* p : {(void*)dq,(void*)dk,(void*)dv,(void*)dm,(void*)dout})
                sycl::free(p, q);
        }
    }

    // ---- §3 lag_rope_apply vs apply_rotary_pos_emb + rotate_half ------------
    // The point of this gate: Laguna's rope is NOT DeepSeek-V4's.  It rotates
    // the LEADING rope_dim dims with non-interleaved half-split pairing; DS4
    // rotates the TRAILING slice with adjacent pairing.  Reusing the wrong one
    // is silent.  A control at the end feeds DS4-style ADJACENT pairing and
    // must be rejected.
    std::ifstream rf2;
    for (const char* p : {"tests/data/laguna_rope_golden.bin",
                          "../tests/data/laguna_rope_golden.bin",
                          "../../tests/data/laguna_rope_golden.bin"}) {
        rf2.open(p, std::ios::binary);
        if (rf2) break;
        rf2.clear();
    }
    if (!rf2) { std::puts("  §3 SKIPPED (rope blob not found)"); }
    else {
        uint32_t nc3 = 0;
        assert(rd(rf2, &nc3, 4) && nc3 && nc3 <= 64);
        std::puts("\n§3 lag_rope_apply (leading dims, non-interleaved half-split)");
        for (uint32_t ci = 0; ci < nc3; ++ci) {
            uint32_t T = 0, H = 0, D = 0, RD = 0;
            assert(rd(rf2, &T, 4) && rd(rf2, &H, 4) && rd(rf2, &D, 4) && rd(rf2, &RD, 4));
            std::vector<float> x(size_t(T)*H*D), co(size_t(T)*RD), si(co.size()), ref(x.size());
            assert(rd(rf2, x.data(), x.size()*4));
            assert(rd(rf2, co.data(), co.size()*4));
            assert(rd(rf2, si.data(), si.size()*4));
            assert(rd(rf2, ref.data(), ref.size()*4));
            float* dx = sycl::malloc_device<float>(x.size(), q);
            float* dc = sycl::malloc_device<float>(co.size(), q);
            float* ds = sycl::malloc_device<float>(si.size(), q);
            float* dy = sycl::malloc_device<float>(ref.size(), q);
            q.memcpy(dx, x.data(), x.size()*4).wait();
            q.memcpy(dc, co.data(), co.size()*4).wait();
            q.memcpy(ds, si.data(), si.size()*4).wait();
            ie::lag_rope_apply(q, dx, dc, ds, dy, T, H, D, RD).wait();
            std::vector<float> got(ref.size());
            q.memcpy(got.data(), dy, got.size()*4).wait();
            double me = 0.0;
            for (size_t i = 0; i < ref.size(); ++i)
                me = std::max(me, double(std::fabs(got[i] - ref[i])));
            const bool ok = me <= 1e-6;
            std::printf("  T=%-3u heads=%-3u head_dim=%-4u rope_dim=%-4u (%s)  max abs err %.3e  %s\n",
                        T, H, D, RD, RD == D ? "all dims" : "partial", me, ok ? "OK" : "FAIL");
            if (!ok) ++fails;
            // CONTROL (last case): DS4-style ADJACENT pairing instead of the
            // half-split.  Emulated by permuting cos/sin into the interleaved
            // order; if the bound cannot see the difference it is measuring
            // nothing, because that is exactly the wrong-kernel-reuse mistake.
            if (ci + 1 == nc3 && RD >= 4) {
                std::vector<float> ci2(co.size()), si2(si.size());
                for (uint32_t t = 0; t < T; ++t)
                    for (uint32_t d = 0; d < RD; ++d) {
                        const uint32_t src = (d % 2 == 0) ? d / 2 : RD / 2 + d / 2;
                        ci2[size_t(t)*RD + d] = co[size_t(t)*RD + src];
                        si2[size_t(t)*RD + d] = si[size_t(t)*RD + src];
                    }
                q.memcpy(dc, ci2.data(), ci2.size()*4).wait();
                q.memcpy(ds, si2.data(), si2.size()*4).wait();
                ie::lag_rope_apply(q, dx, dc, ds, dy, T, H, D, RD).wait();
                q.memcpy(got.data(), dy, got.size()*4).wait();
                double e2 = 0.0;
                for (size_t i = 0; i < ref.size(); ++i)
                    e2 = std::max(e2, double(std::fabs(got[i] - ref[i])));
                std::printf("  control: interleaved (DS4-style) cos/sin -> max abs err %.3e  %s\n",
                            e2, e2 > 1e-6 ? "REJECTED (bound discriminates)"
                                          : "NOT REJECTED (bound is useless)");
                if (!(e2 > 1e-6)) ++fails;
            }
            for (void* p : {(void*)dx,(void*)dc,(void*)ds,(void*)dy}) sycl::free(p, q);
        }
    }

    // ---- §4 lag_moe_router vs LagunaTopKRouter.forward -----------------------
    // Compared as a DENSE per-expert weight row (slot order is unspecified and
    // the expert sum is order-independent).  A control at the end folds the
    // selection bias into the WEIGHT — the classic misreading of aux-loss-free
    // balancing — and must be rejected.
    std::ifstream mf;
    for (const char* p : {"tests/data/laguna_router_golden.bin",
                          "../tests/data/laguna_router_golden.bin",
                          "../../tests/data/laguna_router_golden.bin"}) {
        mf.open(p, std::ios::binary);
        if (mf) break;
        mf.clear();
    }
    if (!mf) { std::puts("  §4 SKIPPED (router blob not found)"); }
    else {
        uint32_t nc4 = 0;
        assert(rd(mf, &nc4, 4) && nc4 && nc4 <= 64);
        std::puts("\n§4 lag_moe_router (sigmoid scoring; bias shifts SELECTION only)");
        for (uint32_t ci = 0; ci < nc4; ++ci) {
            uint32_t T = 0, E = 0, K = 0, norm = 0;
            assert(rd(mf, &T, 4) && rd(mf, &E, 4) && rd(mf, &K, 4) && rd(mf, &norm, 4));
            std::vector<float> lg(size_t(T)*E), bi(E), dref(size_t(T)*E);
            assert(rd(mf, lg.data(), lg.size()*4));
            assert(rd(mf, bi.data(), bi.size()*4));
            assert(rd(mf, dref.data(), dref.size()*4));
            float* dl = sycl::malloc_device<float>(lg.size(), q);
            float* db = sycl::malloc_device<float>(bi.size(), q);
            int32_t* di = sycl::malloc_device<int32_t>(size_t(T)*K, q);
            float* dw = sycl::malloc_device<float>(size_t(T)*K, q);
            q.memcpy(dl, lg.data(), lg.size()*4).wait();
            q.memcpy(db, bi.data(), bi.size()*4).wait();
            ie::lag_moe_router(q, dl, db, di, dw, T, E, K, norm != 0, 0.f).wait();
            std::vector<int32_t> idx(size_t(T)*K);
            std::vector<float> w(size_t(T)*K);
            q.memcpy(idx.data(), di, idx.size()*4).wait();
            q.memcpy(w.data(), dw, w.size()*4).wait();
            std::vector<float> dense(size_t(T)*E, 0.f);
            bool bad = false;
            for (uint32_t t = 0; t < T; ++t)
                for (uint32_t r = 0; r < K; ++r) {
                    const int32_t e2 = idx[size_t(t)*K + r];
                    if (e2 < 0 || uint32_t(e2) >= E) { bad = true; continue; }
                    dense[size_t(t)*E + uint32_t(e2)] = w[size_t(t)*K + r];
                }
            double me = 0.0;
            for (size_t i = 0; i < dense.size(); ++i)
                me = std::max(me, double(std::fabs(dense[i] - dref[i])));
            const bool ok = !bad && me <= 2e-6;
            std::printf("  T=%-3u experts=%-4u k=%-3u norm=%u  max abs err %.3e  %s\n",
                        T, E, K, norm, me, ok ? "OK" : "FAIL");
            if (!ok) ++fails;
            // CONTROL: fold the selection bias into the WEIGHT (bias=0 for the
            // engine call means the weights come from unbiased scores; here we
            // deliberately pass biased logits so the weights shift).  Must differ.
            if (ci + 1 == nc4) {
                std::vector<float> lg2 = lg;
                for (uint32_t t = 0; t < T; ++t)
                    for (uint32_t e2 = 0; e2 < E; ++e2) lg2[size_t(t)*E + e2] += bi[e2];
                q.memcpy(dl, lg2.data(), lg2.size()*4).wait();
                std::vector<float> zero(E, 0.f);
                q.memcpy(db, zero.data(), zero.size()*4).wait();
                ie::lag_moe_router(q, dl, db, di, dw, T, E, K, norm != 0, 0.f).wait();
                q.memcpy(idx.data(), di, idx.size()*4).wait();
                q.memcpy(w.data(), dw, w.size()*4).wait();
                std::vector<float> d2(size_t(T)*E, 0.f);
                for (uint32_t t = 0; t < T; ++t)
                    for (uint32_t r = 0; r < K; ++r) {
                        const int32_t e2 = idx[size_t(t)*K + r];
                        if (e2 >= 0 && uint32_t(e2) < E) d2[size_t(t)*E + uint32_t(e2)] = w[size_t(t)*K + r];
                    }
                double e3 = 0.0;
                for (size_t i = 0; i < d2.size(); ++i)
                    e3 = std::max(e3, double(std::fabs(d2[i] - dref[i])));
                std::printf("  control: bias folded into the WEIGHT -> max abs err %.3e  %s\n",
                            e3, e3 > 2e-6 ? "REJECTED (bound discriminates)"
                                          : "NOT REJECTED (bound is useless)");
                if (!(e3 > 2e-6)) ++fails;
            }
            for (void* p : {(void*)dl,(void*)db,(void*)di,(void*)dw}) sycl::free(p, q);
        }
    }

    // ---- §5 THE OTHER HALF OF THE ROPE CLAIM --------------------------------
    // docs/laguna/00_PORT_PLAN.md says: reuse DeepSeek-V4's YaRN TABLE, not its
    // apply.  §3 proved the apply differs; this proves the TABLE does not.
    // Goldens come from transformers' OWN `_compute_yarn_parameters` (the
    // function Laguna's rope_init_fn resolves to), at Laguna's real
    // full-attention parameters — theta 500000, factor 32, original ctx 8192,
    // beta 32/1, partial_rotary 0.5 (so 64 of 128 dims, i.e. 32 frequencies).
    std::ifstream yf;
    for (const char* p : {"tests/data/yarn_golden.bin", "../tests/data/yarn_golden.bin",
                          "../../tests/data/yarn_golden.bin"}) {
        yf.open(p, std::ios::binary);
        if (yf) break;
        yf.clear();
    }
    if (!yf) { std::puts("  §5 SKIPPED (yarn blob not found)"); }
    else {
        uint32_t nc5 = 0;
        assert(rd(yf, &nc5, 4) && nc5 && nc5 <= 64);
        std::puts("\n§5 ds4_rope_inv_freq REUSED as Laguna's YaRN table");
        for (uint32_t ci = 0; ci < nc5; ++ci) {
            uint32_t hd = 0, orig = 0, n = 0;
            float partial = 0, base = 0, factor = 0, bf = 0, bs = 0;
            assert(rd(yf, &hd, 4) && rd(yf, &partial, 4) && rd(yf, &base, 4) &&
                   rd(yf, &factor, 4) && rd(yf, &orig, 4) && rd(yf, &bf, 4) && rd(yf, &bs, 4));
            assert(rd(yf, &n, 4));
            std::vector<float> ref(n);
            assert(rd(yf, ref.data(), ref.size() * 4));
            ie::Ds4RopeConfig rc;
            rc.yarn = true;
            rc.theta = base;
            rc.factor = factor;
            rc.original_max_pos = orig;
            rc.beta_fast = bf;
            rc.beta_slow = bs;
            const uint32_t rope_dim = uint32_t(float(hd) * partial);
            const std::vector<float> got = ie::ds4_rope_inv_freq(rc, rope_dim);
            double me = 0.0;
            bool sized = got.size() == ref.size();
            if (sized)
                for (size_t i = 0; i < ref.size(); ++i)
                    me = std::max(me, double(std::fabs(got[i] - ref[i])) /
                                      (double(std::fabs(ref[i])) + 1e-12));
            const bool ok = sized && me <= 1e-5;
            std::printf("  head_dim=%-4u partial=%.2f theta=%-8.0f factor=%-5.1f  "
                        "%zu freqs  max rel err %.3e  %s\n",
                        hd, partial, base, factor, got.size(), me,
                        !sized ? "FAIL (size)" : (ok ? "OK — table reusable" : "FAIL"));
            if (!ok) ++fails;
        }
    }

    if (fails) { std::printf("laguna_ops_test: FAILED (%d)\n", fails); return 1; }
    std::puts("laguna_ops_test: OK — matches the upstream reference");
    return 0;
}
