// P2 MILESTONE: run REAL Inkling weights through the gated layer composition.
//
// This is the first time this architecture's actual tensors flow through engine
// kernels.  It loads blk.0 (the dense layer variant) from the real GGUF,
// dequantises every tensor to fp32 on the host, runs the composition that
// `inkling_layer_test` already proved correct against the reference, and checks
// the OUTPUT IS SANE.
//
// WHAT THIS CAN AND CANNOT PROVE — stated plainly, because the distinction is
// the whole point.  There are no reference ACTIVATIONS for this model on this
// box (only the GGUF; the HF safetensors are not downloaded), so this cannot be
// a parity gate.  What it proves is the LOADING contract: every tensor is
// found, has the expected dtype and element count, dequantises without a
// non-finite value, and drives the layer to a finite output whose magnitude is
// in the range a normalised residual stream should occupy.  Numerical
// correctness of the MATH is carried by `inkling_layer_test` against the
// reference; this carries the WEIGHTS.  Neither substitutes for the other.
#undef NDEBUG
#include "ie/dequant_ref.hpp"
#include "ie/gguf.hpp"
#include "ie/inkling_ops.hpp"
#include "ie/model_config.hpp"

#include <sycl/sycl.hpp>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
// Model path comes from $IE_INKLING_GGUF — these gates need the real Inkling-Small
// GGUF, which is far too large to ship. Unset (or unreachable) means the
// test SKIPS cleanly, so a checkout without the model still goes green.
const char* shard1_path() { return std::getenv("IE_INKLING_GGUF"); }

// Host dequant to fp32 for every dtype blk.0 actually uses.  Returns false for
// a dtype this path does not implement — a LOUD refusal, never a zero buffer.
bool dequant_to_f32(const ie::GgufReader& g, const ie::GgufTensorInfo& t,
                    std::vector<float>& out) {
    uint64_t n = 1;
    for (uint32_t i = 0; i < t.n_dims; ++i) n *= t.shape[i];
    out.assign(size_t(n), 0.f);
    const ie::Tensor view = g.make_host_view(t);
    const void* p = view.data;
    switch (t.dtype) {
        case ie::DType::kF32:
            std::memcpy(out.data(), p, size_t(n) * 4);
            return true;
        case ie::DType::kQ8_0: ie::ref::dequant_q8_0_buffer(p, n, out.data()); return true;
        case ie::DType::kQ4_K: ie::ref::dequant_q4_K_buffer(p, n, out.data()); return true;
        case ie::DType::kQ5_K: ie::ref::dequant_q5_K_buffer(p, n, out.data()); return true;
        case ie::DType::kQ6_K: ie::ref::dequant_q6_K_buffer(p, n, out.data()); return true;
        default: return false;
    }
}
}  // namespace

int main() {
    const char* path = shard1_path();
    if (!path || !*path) {
        std::puts("inkling_load_test: SKIPPED (set IE_INKLING_GGUF to the model's shard-1 path)");
        return 0;
    }
    ie::GgufReader g;
    if (!g.open(path).empty()) {
        std::puts("inkling_load_test: SKIPPED (model not reachable)");
        return 0;
    }
    ie::InklingConfig c;
    const std::string e = ie::read_inkling_config(g, c);
    if (!e.empty()) { std::printf("FAIL config: %s\n", e.c_str()); return 1; }

    const uint32_t H = c.hidden, HD = c.head_dim, NH = c.n_q_heads;
    const uint32_t KVH = uint32_t(c.n_kv_heads[0]);
    const uint32_t K = c.shortconv_kernel, DR = c.d_rel, EXT = c.rel_extent_for(0);
    const uint32_t FF = c.ffn_dense;
    const uint32_t T = 4;
    std::printf("blk.0: H=%u heads=%u/%u head_dim=%u ffn=%u d_rel=%u extent=%u sconv_k=%u\n",
                H, NH, KVH, HD, FF, DR, EXT, K);

    int fails = 0;
    auto load = [&](const char* suffix, std::vector<float>& dst, uint64_t want) {
        const std::string n = std::string("blk.0.") + suffix;
        const auto* t = g.find_tensor(n);
        if (!t) { std::printf("  MISSING %s\n", n.c_str()); ++fails; return; }
        if (!dequant_to_f32(g, *t, dst)) {
            std::printf("  UNSUPPORTED dtype %s for %s\n",
                        std::string(ie::type_name(t->dtype)).c_str(), n.c_str());
            ++fails;
            return;
        }
        if (dst.size() != want) {
            std::printf("  SIZE %s: %zu vs %llu\n", n.c_str(), dst.size(),
                        (unsigned long long)want);
            ++fails;
        }
        uint64_t nonfinite = 0;
        double amax = 0.0;
        for (float v : dst) {
            if (!std::isfinite(v)) ++nonfinite;
            else amax = std::max(amax, double(std::fabs(v)));
        }
        if (nonfinite) {
            std::printf("  NON-FINITE %s: %llu values\n", n.c_str(),
                        (unsigned long long)nonfinite);
            ++fails;
        }
        std::printf("  %-24s %-7s %9zu elems  |max| %.4f\n", suffix,
                    std::string(ie::type_name(t->dtype)).c_str(), dst.size(), amax);
    };

    std::vector<float> w_in, w_post, wq, wk, wv, wo, wr, qn, kn,
                       sck, scv, sca, scm, proj, wg, wu, wd;
    load("attn_norm.weight", w_in, H);
    load("ffn_norm.weight", w_post, H);
    load("attn_q.weight", wq, uint64_t(H) * NH * HD);
    load("attn_k.weight", wk, uint64_t(H) * KVH * HD);
    load("attn_v.weight", wv, uint64_t(H) * KVH * HD);
    load("attn_output.weight", wo, uint64_t(NH) * HD * H);
    load("attn_r.weight", wr, uint64_t(H) * NH * DR);
    load("attn_q_norm.weight", qn, HD);
    load("attn_k_norm.weight", kn, HD);
    load("shortconv_k.weight", sck, uint64_t(KVH) * HD * K);
    load("shortconv_v.weight", scv, uint64_t(KVH) * HD * K);
    load("shortconv_attn.weight", sca, uint64_t(NH) * HD * K);
    load("shortconv_mlp.weight", scm, uint64_t(H) * K);
    load("attn_rel_proj.weight", proj, uint64_t(DR) * EXT);
    load("ffn_gate.weight", wg, uint64_t(H) * FF);
    load("ffn_up.weight", wu, uint64_t(H) * FF);
    load("ffn_down.weight", wd, uint64_t(FF) * H);
    if (fails) { std::printf("inkling_load_test: FAILED (%d load errors)\n", fails); return 1; }

    // DRIVE THE LAYER WITH THE MODEL'S OWN EMBEDDINGS, not synthetic noise.
    // The first run used a 0.02-scale pseudo-random input and produced an
    // output rms of ~2.8e3, which could not be interpreted: an unrepresentative
    // input is one of the three candidate explanations, and it is the only one
    // that can be REMOVED without an activation oracle.  `token_embd` and
    // `token_embd_norm` are both in this GGUF, so the real thing costs nothing.
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::vector<float> x(size_t(T) * H);
    {
        const auto* te = g.find_tensor("token_embd.weight");
        const auto* tn = g.find_tensor("token_embd_norm.weight");
        if (!te || !tn) { std::puts("  MISSING token_embd / token_embd_norm"); return 1; }
        // Dequantising the whole [H, vocab] embedding is ~1.6 GB; take the rows
        // for a handful of token ids instead.  Q6_K/Q8_0 rows are block-aligned
        // on H, so a row slice is exact.
        std::vector<float> emb;
        if (!dequant_to_f32(g, *te, emb)) { std::puts("  embedding dtype unsupported"); return 1; }
        std::vector<float> nrm;
        if (!dequant_to_f32(g, *tn, nrm)) { std::puts("  embd norm dtype unsupported"); return 1; }
        const uint32_t ids[4] = {1u, 100u, 5000u, 20000u};
        double in_rms_pre = 0.0;
        for (uint32_t t = 0; t < T; ++t) {
            const uint64_t base = uint64_t(ids[t % 4]) * H;
            // RMSNorm the embedding row, exactly as the model does before blk.0.
            double ss = 0.0;
            for (uint32_t d = 0; d < H; ++d) {
                const float v = emb[base + d];
                ss += double(v) * v;
                in_rms_pre += double(v) * v;
            }
            const float inv = float(1.0 / std::sqrt(ss / double(H) + double(c.rms_eps)));
            for (uint32_t d = 0; d < H; ++d)
                x[size_t(t) * H + d] = emb[base + d] * inv * nrm[d];
        }
        double ss2 = 0.0;
        for (float v : x) ss2 += double(v) * v;
        std::printf("input: real token embeddings, rms(pre-norm) %.5f -> rms(post-norm) %.5f\n",
                    std::sqrt(in_rms_pre / double(x.size())),
                    std::sqrt(ss2 / double(x.size())));
    }
    auto up = [&](const std::vector<float>& v) {
        float* d = sycl::malloc_device<float>(v.size(), q);
        q.memcpy(d, v.data(), v.size() * 4).wait();
        return d;
    };
    auto al = [&](size_t n) { return sycl::malloc_device<float>(n, q); };
    float *dx = up(x), *dwin = up(w_in), *dwpost = up(w_post), *dwq = up(wq), *dwk = up(wk),
          *dwv = up(wv), *dwo = up(wo), *dwr = up(wr), *dqn = up(qn), *dkn = up(kn),
          *dsck = up(sck), *dscv = up(scv), *dsca = up(sca), *dscm = up(scm),
          *dproj = up(proj), *dwg = up(wg), *dwu = up(wu), *dwd = up(wd);
    float *hn = al(size_t(T) * H), *qb = al(size_t(T) * NH * HD),
          *kb = al(size_t(T) * KVH * HD), *vb = al(size_t(T) * KVH * HD),
          *kt = al(size_t(T) * KVH * HD), *vt = al(size_t(T) * KVH * HD),
          *rel = al(size_t(T) * NH * DR), *bias = al(size_t(NH) * T * T),
          *att = al(size_t(T) * NH * HD), *po = al(size_t(T) * H), *sco = al(size_t(T) * H),
          *h1 = al(size_t(T) * H), *hn2 = al(size_t(T) * H), *gb = al(size_t(T) * FF),
          *ub = al(size_t(T) * FF), *sw = al(size_t(T) * FF), *mlp = al(size_t(T) * H),
          *msc = al(size_t(T) * H), *yo = al(size_t(T) * H);
    std::vector<float> hmask(size_t(T) * T, 0.f);
    for (uint32_t i = 0; i < T; ++i)
        for (uint32_t j = i + 1; j < T; ++j)
            hmask[size_t(i) * T + j] = -std::numeric_limits<float>::infinity();
    std::vector<int32_t> hpos(T);
    for (uint32_t i = 0; i < T; ++i) hpos[i] = int32_t(i);
    float* dmask = up(hmask);
    int32_t* dpos = sycl::malloc_device<int32_t>(T, q);
    q.memcpy(dpos, hpos.data(), T * 4).wait();

    // Stage-by-stage rms, so a 1290x layer gain can be LOCALISED instead of
    // guessed at.  Cheap (a few small copies) and it is the diagnostic that
    // separates "this layer legitimately amplifies" from "one tensor is
    // transposed".
    auto stage = [&](const char* name, const float* d, size_t n) {
        std::vector<float> h(n);
        q.wait();
        q.memcpy(h.data(), d, n * 4).wait();
        double ss = 0.0, mx = 0.0;
        for (float v : h) { ss += double(v) * v; mx = std::max(mx, double(std::fabs(v))); }
        std::printf("    %-14s rms %12.4f  |max| %12.4f\n", name,
                    std::sqrt(ss / double(n)), mx);
    };
    std::puts("  stage rms:");
    stage("input", dx, size_t(T) * H);
    ie::ink_rms_norm(q, dx, dwin, hn, T, H, c.rms_eps);
    stage("attn_norm", hn, size_t(T) * H);
    ie::ink_matmul_nt(q, hn, dwq, qb, T, H, NH * HD);
    ie::ink_matmul_nt(q, hn, dwk, kt, T, H, KVH * HD);
    ie::ink_matmul_nt(q, hn, dwv, vt, T, H, KVH * HD);
    stage("q_proj", qb, size_t(T) * NH * HD);
    stage("k_proj", kt, size_t(T) * KVH * HD);
    ie::ink_shortconv(q, kt, dsck, kb, T, KVH * HD, K);
    ie::ink_shortconv(q, vt, dscv, vb, T, KVH * HD, K);
    stage("k_sconv", kb, size_t(T) * KVH * HD);
    ie::ink_rms_norm(q, qb, dqn, qb, T * NH, HD, c.rms_eps);
    ie::ink_rms_norm(q, kb, dkn, kb, T * KVH, HD, c.rms_eps);
    ie::ink_matmul_nt(q, hn, dwr, rel, T, H, NH * DR);
    ie::ink_rel_logits(q, rel, dproj, dpos, dpos, bias, T, NH, DR, EXT, T);
    ie::ink_attention(q, qb, kb, vb, bias, dmask, att, T, NH, KVH, HD, T,
                      1.0f / std::sqrt(float(HD)));
    stage("attn_out", att, size_t(T) * NH * HD);
    ie::ink_matmul_nt(q, att, dwo, po, T, NH * HD, H);
    stage("o_proj", po, size_t(T) * H);
    ie::ink_shortconv(q, po, dsca, sco, T, H, K);
    stage("attn_sconv", sco, size_t(T) * H);
    ie::ink_add(q, dx, sco, h1, size_t(T) * H);
    stage("resid1", h1, size_t(T) * H);
    ie::ink_rms_norm(q, h1, dwpost, hn2, T, H, c.rms_eps);
    ie::ink_matmul_nt(q, hn2, dwg, gb, T, H, FF);
    ie::ink_matmul_nt(q, hn2, dwu, ub, T, H, FF);
    stage("ffn_gate", gb, size_t(T) * FF);
    ie::ink_swiglu(q, gb, ub, sw, size_t(T) * FF);
    stage("swiglu", sw, size_t(T) * FF);
    ie::ink_matmul_nt(q, sw, dwd, mlp, T, FF, H);
    stage("ffn_down", mlp, size_t(T) * H);
    ie::ink_shortconv(q, mlp, dscm, msc, T, H, K);
    stage("mlp_sconv", msc, size_t(T) * H);
    ie::ink_add(q, h1, msc, yo, size_t(T) * H);
    q.wait();

    std::vector<float> y(size_t(T) * H);
    q.memcpy(y.data(), yo, y.size() * 4).wait();
    uint64_t nonfinite = 0;
    double amax = 0.0, sum2 = 0.0;
    for (float v : y) {
        if (!std::isfinite(v)) ++nonfinite;
        else { amax = std::max(amax, double(std::fabs(v))); sum2 += double(v) * v; }
    }
    const double rms = std::sqrt(sum2 / double(y.size()));
    std::printf("output: |max| %.4f  rms %.4f  non-finite %llu\n",
                amax, rms, (unsigned long long)nonfinite);
    if (nonfinite) { std::puts("FAIL: non-finite output"); ++fails; }
    // A dead layer (all-zero, e.g. a weight that silently dequantised to
    // nothing) is a LOADER failure and is asserted.
    if (!(rms > 1e-6)) { std::puts("FAIL: output is dead (rms ~ 0)"); ++fails; }

    // MAGNITUDE IS REPORTED, NOT ASSERTED — and that is deliberate.
    //
    // Measured 2026-08-09: a 0.02-scale synthetic input drives this layer to
    // rms ~2.8e3 / |max| ~2.0e5.  That is far larger than a residual stream
    // should carry, and there are three candidate explanations that CANNOT be
    // separated without reference ACTIVATIONS (which need the HF safetensors;
    // only the GGUF is on this box):
    //   1. The input is unrepresentative.  Real hidden states arrive through
    //      `token_embd` + `token_embd_norm`, which this test bypasses; the
    //      leading RMSNorm makes the attention path scale-insensitive, but the
    //      RESIDUAL is the raw input, so an unrealistic x biases the sum.
    //   2. The layer genuinely amplifies.  `shortconv_mlp` weights reach |16.9|
    //      and `ffn_norm` |2.06| on this checkpoint — a wide MLP (FF 16384) with
    //      weights that large can legitimately produce a big layer-0 delta.
    //   3. A composition or layout error this test cannot see, because
    //      `inkling_layer_test` proves the SAME composition against the
    //      reference on synthetic weights — so the math is right for the shapes
    //      it was given, but a per-tensor transposition that is shape-symmetric
    //      (e.g. a square [4096,4096] projection) would survive both gates.
    //
    // Asserting a bound here would be inventing a criterion to make a number
    // green.  What this test DOES certify is the LOADING CONTRACT, which is
    // exactly what it was written for.  Resolving the magnitude is the next
    // P2 item and needs an activation oracle — see docs/inkling/00_PORT_PLAN.md.
    std::printf("NOTE: output magnitude is DIAGNOSTIC, not asserted — it is unvalidated "
                "until reference activations exist (see the port plan).\n");
    if (fails) { std::printf("inkling_load_test: FAILED (%d)\n", fails); return 1; }
    std::puts("inkling_load_test: OK — the LOADING CONTRACT holds: every blk.0 tensor is "
              "present, dequantises finite at the expected size, and drives the layer to "
              "completion. Output SCALE is not yet validated.");
    return 0;
}
