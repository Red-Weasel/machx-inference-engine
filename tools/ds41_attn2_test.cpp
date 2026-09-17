// tools/ds41_attn2_test.cpp — V4.1 Phase 6b: compressed attention at layer 2 vs the reference.
//
// Layer 2 is a compress_ratio-2 SOURCE layer: it runs the compressor, owns the index key, runs
// the indexer, and attends over [window ; compressed]. T = 1536 so the indexer's top-512 of 768
// is a real selection (below 1,024 tokens it selects everything and this test would be vacuous;
// the test asserts the golden made a real selection before trusting itself).
// Criteria: docs/deepseek41/16_PHASE6B_CRITERIA_2026-09-12.md
#include "ie/deepseek4_attn.hpp"
#include "ie/deepseek41_upload.hpp"
#include "ie/ops.hpp"

#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what, const std::string& detail = "") {
    std::printf("%s%s%s\n", ok ? "[ ok ] " : "[FAIL] ", what.c_str(),
                detail.empty() ? "" : ("  (" + detail + ")").c_str());
    if (!ok) ++g_fail;
}
template <class T> std::vector<T> rd(const std::string& p, size_t n) {
    std::vector<T> v(n);
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p.c_str()); std::exit(1); }
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(T)));
    if (size_t(f.gcount()) != n * sizeof(T)) { std::fprintf(stderr, "%s: short\n", p.c_str()); std::exit(1); }
    return v;
}
inline float bf16(uint16_t h) { const uint32_t b = uint32_t(h) << 16; float f; std::memcpy(&f, &b, 4); return f; }
struct Cmp { double rel, max_abs; };
Cmp cmp(const std::vector<float>& a, const std::vector<float>& b, bool finite_only = false) {
    double m = 0, s = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (finite_only && !std::isfinite(b[i])) continue;
        m = std::max(m, std::fabs(double(a[i]) - double(b[i])));
        s = std::max(s, std::fabs(double(b[i])));
    }
    return {s > 0 ? m / s : 0.0, m};
}
}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : std::string(std::getenv("HOME")) + "/models/DeepSeek-V4.1-Flash";
    const std::string gd  = argc > 2 ? argv[2] : "/tmp/ds41_golden";

    ie::DeepSeek41Model m;
    if (const auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    const auto& c = m.config();
    const uint32_t LAYER = 2;
    const auto& Lw = m.layers()[LAYER];
    check(Lw.kind.compress_ratio == 2 && Lw.kind.is_kv_source && Lw.kind.is_index_source &&
          Lw.kind.has_compressor_gate, "layer 2 is a ratio-2 source layer with a gate");

    const uint32_t T = 1536, R = 2, NC = T / R, H = c.dim, QR = c.q_lora_rank, HD = c.head_dim;
    const uint32_t NH = c.n_heads, QH = NH * HD, RD = c.rope_head_dim, WIN = c.window_size;
    const uint32_t G = c.o_groups, OPG = c.o_lora_rank, IPG = QH / G, OR = G * OPG;
    const uint32_t IH = c.index_n_heads, IHD = c.index_head_dim, IQ = IH * IHD, TOPK = std::min(c.index_topk, NC);
    const uint32_t NKV = T + NC;

    // ---- the golden must have made a real selection, or this test proves nothing ------------
    const auto g_topk = rd<int32_t>(gd + "/a2_topk.i32", size_t(T) * TOPK);
    {
        size_t kept = 0;
        for (uint32_t j = 0; j < TOPK; ++j) kept += g_topk[size_t(T - 1) * TOPK + j] >= 0;
        const uint32_t reach = T / R;
        check(kept < reach, "golden's last query selected fewer than it could reach (not vacuous)",
              std::to_string(kept) + " of " + std::to_string(reach));
        if (g_fail) return 1;
    }

    sycl::device dev;
    for (const auto& p : sycl::platform::get_platforms())
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) { dev = d; goto found; }
    std::fprintf(stderr, "no Arc GPU\n"); return 1;
found:
    sycl::queue q(dev, sycl::property::queue::in_order{});
    std::printf("%s — layer %u compressed attention, T=%u, %u compressed, top-%u\n",
                dev.get_info<sycl::info::device::name>().c_str(), LAYER, T, NC, TOPK);

    auto f32 = [&](size_t n) { return sycl::malloc_device<float>(n, q); };
    auto f16 = [&](size_t n) { return sycl::malloc_device<sycl::half>(n, q); };
    auto to16 = [&](const float* s, sycl::half* d, size_t n) {
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { d[i] = sycl::half(s[i]); }).wait();
    };
    auto dense = [&](const ie::Ds41Tensor& t) {          // FP8 + 32x32 scale -> fp16
        const uint32_t N = uint32_t(t.w->shape[0]), K = uint32_t(t.w->shape[1]);
        const uint32_t bn = N / uint32_t(t.s->shape[0]), bk = K / uint32_t(t.s->shape[1]);
        uint8_t* dw = sycl::malloc_device<uint8_t>(size_t(N) * K, q);
        uint8_t* ds = sycl::malloc_device<uint8_t>(t.s->nbytes, q);
        sycl::half* o = f16(size_t(N) * K);
        q.memcpy(dw, t.w->data, size_t(N) * K); q.memcpy(ds, t.s->data, t.s->nbytes).wait();
        ie::ds41_dense_dequant_f16(q, dw, ds, N, K, bn, bk, o).wait();
        sycl::free(dw, q); sycl::free(ds, q);
        return o;
    };
    auto bf16_dense = [&](const ie::Ds41Tensor& t) {     // BF16 -> fp16 (both ~3 digits; exact for most values)
        const size_t n = size_t(t.w->numel());
        const auto* s = reinterpret_cast<const uint16_t*>(t.w->data);
        std::vector<sycl::half> h(n);
        for (size_t i = 0; i < n; ++i) h[i] = sycl::half(bf16(s[i]));
        sycl::half* d = f16(n); q.memcpy(d, h.data(), n * 2).wait(); return d;
    };
    auto bf16_f32 = [&](const ie::Ds41Tensor& t) {
        const size_t n = size_t(t.w->numel());
        const auto* s = reinterpret_cast<const uint16_t*>(t.w->data);
        std::vector<float> h(n);
        for (size_t i = 0; i < n; ++i) h[i] = bf16(s[i]);
        float* d = f32(n); q.memcpy(d, h.data(), n * 4).wait(); return d;
    };
    auto upf = [&](const std::vector<float>& v) { float* d = f32(v.size()); q.memcpy(d, v.data(), v.size() * 4).wait(); return d; };

    // ---- weights --------------------------------------------------------------------------
    sycl::half *w_qa = dense(Lw.wq_a), *w_qb = dense(Lw.wq_b), *w_kv = dense(Lw.wkv);
    sycl::half *w_oa = dense(Lw.wo_a), *w_ob = dense(Lw.wo_b);
    sycl::half *w_ckv = bf16_dense(Lw.comp_wkv), *w_cg = bf16_dense(Lw.comp_wgate);
    sycl::half *w_iqb = dense(Lw.idx_wq_b), *w_iwp = bf16_dense(Lw.idx_weights), *w_ik = bf16_dense(Lw.idx_wk);
    float *n_q = bf16_f32(Lw.q_norm), *n_kv = bf16_f32(Lw.kv_norm), *n_c = bf16_f32(Lw.comp_norm), *n_ik = bf16_f32(Lw.idx_k_norm);
    float* sinks = f32(NH); q.memcpy(sinks, Lw.attn_sink.w->data, NH * 4).wait();

    // ---- RoPE: layer 2 uses the COMPRESS table for everything (theta 160000, YaRN) ---------
    ie::Ds4RopeConfig rc; rc.theta = c.compress_rope_theta; rc.yarn = true; rc.factor = c.rope_factor;
    rc.beta_fast = float(c.beta_fast); rc.beta_slow = float(c.beta_slow); rc.original_max_pos = c.original_seq_len;
    ie::Ds4RopeConfig rmain; rmain.theta = c.rope_theta; rmain.yarn = false;
    auto tables = [&](const ie::Ds4RopeConfig& cfg, const std::vector<int32_t>& pos, float*& cs, float*& sn) {
        const auto inv = ie::ds4_rope_inv_freq(cfg, RD);
        float* d_inv = upf(inv);
        int32_t* d_pos = sycl::malloc_device<int32_t>(pos.size(), q);
        q.memcpy(d_pos, pos.data(), pos.size() * 4).wait();
        cs = f32(pos.size() * (RD / 2)); sn = f32(pos.size() * (RD / 2));
        ie::ds4_rope_cos_sin(q, d_inv, d_pos, cs, sn, uint32_t(pos.size()), RD / 2, 1.0f).wait();
        sycl::free(d_inv, q); return d_pos;
    };
    std::vector<int32_t> pos_t(T); std::iota(pos_t.begin(), pos_t.end(), 0);
    const auto pos_g = ie::ds4_compress_positions(NC, R, 0);          // 0, 2, 4, ...
    float *cos_t, *sin_t, *cos_g, *sin_g, *cos_main, *sin_main;
    int32_t* d_pos_t = tables(rc, pos_t, cos_t, sin_t);
    tables(rc, pos_g, cos_g, sin_g);
    tables(rmain, pos_g, cos_main, sin_main);

    // ---- input + q path (shared with the indexer) -----------------------------------------
    const auto g_x = rd<float>(gd + "/a2_x.f32", size_t(T) * H);
    float* x = upf(g_x);
    sycl::half* x16 = f16(size_t(T) * H); to16(x, x16, size_t(T) * H);
    sycl::half* t16 = f16(size_t(T) * std::max({QR, QH, OR, IQ}));
    float *qr = f32(size_t(T) * QR), *qrn = f32(size_t(T) * QR);
    ie::gemm_nt_f16_onednn(q, x16, w_qa, qr, T, QR, H).wait();
    ie::ds4_rms_norm(q, qr, n_q, qrn, T, QR, c.norm_eps).wait();
    {
        std::vector<float> got(size_t(T) * QR); q.memcpy(got.data(), qrn, got.size() * 4).wait();
        const Cmp cq = cmp(got, rd<float>(gd + "/a2_qr.f32", got.size()));
        check(cq.rel < 3e-3, "qr (q_norm)", "rel " + std::to_string(cq.rel));
    }

    // ---- C1 compressor: pool consecutive pairs with a per-channel softmax, then norm ---------
    float *ckv = f32(size_t(T) * HD), *cgate = f32(size_t(T) * HD), *pooled = f32(size_t(NC) * HD), *latent = f32(size_t(NC) * HD);
    ie::gemm_nt_f16_onednn(q, x16, w_ckv, ckv, T, HD, H).wait();
    ie::gemm_nt_f16_onednn(q, x16, w_cg, cgate, T, HD, H).wait();
    float* zero_bias = f32(size_t(R) * HD); q.memset(zero_bias, 0, size_t(R) * HD * 4).wait();
    // [T, HD] contiguous IS [NC, R, HD]: consecutive tokens are consecutive slots.
    ie::ds4_compress_pool(q, ckv, cgate, zero_bias, nullptr, nullptr, pooled, NC, R, HD, false).wait();
    ie::ds4_rms_norm(q, pooled, n_c, latent, NC, HD, c.norm_eps).wait();
    {
        std::vector<float> got(size_t(NC) * HD); q.memcpy(got.data(), latent, got.size() * 4).wait();
        const Cmp cl = cmp(got, rd<float>(gd + "/a2_latent.f32", got.size()));
        check(cl.rel < 3e-3, "C1 compressor latent (RoPE-free)", "rel " + std::to_string(cl.rel));
    }

    // ---- C2 index keys: k_norm(wk(latent)), RoPE at group positions with the compress table --
    sycl::half* lat16 = f16(size_t(NC) * HD); to16(latent, lat16, size_t(NC) * HD);
    float *ik = f32(size_t(NC) * IHD), *ikn = f32(size_t(NC) * IHD), *ik_main = f32(size_t(NC) * IHD);
    ie::gemm_nt_f16_onednn(q, lat16, w_ik, ik, NC, IHD, HD).wait();
    ie::ds4_rms_norm(q, ik, n_ik, ikn, NC, IHD, c.norm_eps).wait();
    q.memcpy(ik_main, ikn, size_t(NC) * IHD * 4).wait();
    ie::ds4_rope_apply(q, ikn, cos_g, sin_g, ikn, NC, 1, IHD, RD, +1.0f).wait();
    ie::ds4_rope_apply(q, ik_main, cos_main, sin_main, ik_main, NC, 1, IHD, RD, +1.0f).wait();   // C6 control
    const auto g_ik = rd<float>(gd + "/a2_index_k.f32", size_t(NC) * IHD);
    {
        std::vector<float> got(size_t(NC) * IHD), bad(size_t(NC) * IHD);
        q.memcpy(got.data(), ikn, got.size() * 4); q.memcpy(bad.data(), ik_main, bad.size() * 4).wait();
        const Cmp ck = cmp(got, g_ik), cb = cmp(bad, g_ik);
        check(ck.rel < 3e-3, "C2 index keys (compress table)", "rel " + std::to_string(ck.rel));
        check(cb.rel > 10 * ck.rel && cb.rel > 3e-3,
              "C6 control: the MAIN rope table would FAIL here, so the table choice is load-bearing",
              "main-table rel " + std::to_string(cb.rel) + " vs compress-table " + std::to_string(ck.rel));
    }

    std::vector<float> g_sc_all;      // golden masked scores, kept for the C4 diagnostic
    double g_c3_rel = 0;              // measured score error, which sets C4's tie band
    std::vector<char>  g_q_differs;   // queries whose top-k set differed, for the C5 diagnostic
    // ---- C3 index scores ---------------------------------------------------------------------
    float *iq = f32(size_t(T) * IQ), *wp = f32(size_t(T) * IH), *scores = f32(size_t(T) * NC);
    to16(qrn, t16, size_t(T) * QR);
    ie::gemm_nt_f16_onednn(q, t16, w_iqb, iq, T, IQ, QR).wait();
    ie::ds4_rope_apply(q, iq, cos_t, sin_t, iq, T, IH, IHD, RD, +1.0f).wait();
    ie::gemm_nt_f16_onednn(q, x16, w_iwp, wp, T, IH, H).wait();
    ie::ds4_indexer_score(q, iq, ikn, wp, scores, T, IH, IHD, NC,
                          1.0f / std::sqrt(float(IHD)), 1.0f / std::sqrt(float(IH))).wait();
    {
        std::vector<float> got(size_t(T) * NC); q.memcpy(got.data(), scores, got.size() * 4).wait();
        const auto g_sc = rd<float>(gd + "/a2_score.f32", got.size());
        g_sc_all = g_sc;
        // the reference masks e >= (t+1)//ratio to -inf; the engine applies that inside topk,
        // so apply it here and compare the PATTERN exactly and the finite values numerically
        size_t pattern_bad = 0;
        for (uint32_t t = 0; t < T; ++t)
            for (uint32_t e = 0; e < NC; ++e) {
                const bool eng_masked = e >= (t + 1) / R, gold_masked = !std::isfinite(g_sc[size_t(t) * NC + e]);
                if (eng_masked != gold_masked) ++pattern_bad;
                if (eng_masked) got[size_t(t) * NC + e] = -std::numeric_limits<float>::infinity();
            }
        check(pattern_bad == 0, "C3 causal mask pattern identical", std::to_string(pattern_bad) + " cells differ");
        const Cmp cs = cmp(got, g_sc, true);
        g_c3_rel = cs.rel;
        check(cs.rel < 3e-3, "C3 index scores (finite entries)", "rel " + std::to_string(cs.rel));
    }

    // ---- C4 top-k as sets ----------------------------------------------------------------------
    int32_t* topk = sycl::malloc_device<int32_t>(size_t(T) * TOPK, q);
    ie::ds4_indexer_topk(q, scores, d_pos_t, topk, T, NC, c.index_topk, R).wait();
    {
        std::vector<int32_t> got(size_t(T) * TOPK); q.memcpy(got.data(), topk, got.size() * 4).wait();
        std::vector<float> sc(size_t(T) * NC); q.memcpy(sc.data(), scores, sc.size() * 4).wait();
        size_t bad_q = 0, ties = 0; uint32_t first_bad = 0;
        float worst_gap = 0.f, worst_gap_rel = 0.f;
        std::vector<char> q_differs(T, 0);
        for (uint32_t t = 0; t < T; ++t) {
            std::vector<int32_t> a, b;
            for (uint32_t j = 0; j < TOPK; ++j) {
                if (got[size_t(t) * TOPK + j] >= 0) a.push_back(got[size_t(t) * TOPK + j]);
                if (g_topk[size_t(t) * TOPK + j] >= 0) b.push_back(g_topk[size_t(t) * TOPK + j] - int32_t(T));
            }
            std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
            if (a == b) continue;
            // excuse a mismatch only when the swapped-in and swapped-out entries tie in score
            std::vector<int32_t> only_a, only_b;
            std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(only_a));
            std::set_difference(b.begin(), b.end(), a.begin(), a.end(), std::back_inserter(only_b));
            // DIAGNOSTIC: how far apart in GOLDEN score are the entries the two selections
            // disagree on? A boundary flip between near-tied entries is the expected result of
            // comparing a discrete top-k over scores that differ by ~4e-4; a large gap is a bug.
            q_differs[t] = 1;
            float qmax = 0.f;
            for (uint32_t e = 0; e < NC; ++e) if (std::isfinite(g_sc_all[size_t(t) * NC + e])) qmax = std::max(qmax, std::fabs(g_sc_all[size_t(t) * NC + e]));
            // Pair the swapped-in and swapped-out entries by GOLDEN SCORE, not by index: with
            // more than one difference per query, index-order pairing can mate a near-tie with
            // a near-tie while the true correspondence is a large gap (gate finding 3).
            auto by_score = [&](int32_t a, int32_t b) { return g_sc_all[size_t(t) * NC + a] > g_sc_all[size_t(t) * NC + b]; };
            std::sort(only_a.begin(), only_a.end(), by_score); std::sort(only_b.begin(), only_b.end(), by_score);
            // AMENDED (disclosed in 16_PHASE6B_CRITERIA): a discrete top-k over scores that
            // differ by the measured score error cannot be expected to agree on entries closer
            // than that. The band is ABSOLUTE -- 3 x the 3.85e-4 measured when the criterion
            // was amended -- so a regression that degrades the scores does not widen its own
            // excuse (gate finding 2), and it is void when C3 itself failed (finding 4).
            const float band = (g_c3_rel <= 3e-3) ? 3.0f * 3.85e-4f * qmax : 0.f;
            bool tie = only_a.size() == only_b.size();
            for (size_t i = 0; tie && i < only_a.size(); ++i) {
                const float ga = g_sc_all[size_t(t) * NC + only_a[i]], gb = g_sc_all[size_t(t) * NC + only_b[i]];
                const float gap = std::fabs(ga - gb);
                worst_gap = std::max(worst_gap, gap); worst_gap_rel = std::max(worst_gap_rel, qmax > 0 ? gap / qmax : 0.f);
                tie = gap < band;
            }
            if (tie) ++ties; else { if (!bad_q) first_bad = t; ++bad_q; }
        }
        std::printf("       C4 diagnostic: %zu queries differ; worst golden-score gap between a swapped-in and "
                    "swapped-out entry = %.3e (%.3e of that query's score scale)\n", bad_q + ties, worst_gap, worst_gap_rel);
        check(bad_q == 0, "C4 selected indices equal the reference's as SETS, up to swaps inside the score-error band",
              std::to_string(bad_q) + " queries differ beyond the band (first t=" + std::to_string(first_bad) + "), " +
              std::to_string(ties) + " boundary-tie queries excused (absolute band 3 x 3.85e-4 of scale)");
        check(ties < T / 20, "fewer than 5% of queries needed a boundary-tie excuse",
              std::to_string(ties) + " of " + std::to_string(T));
        {   // for tools/ds41_reference/prove_ties.py: the engine's own selection, so the
            // reference can be re-run with it and the tie queries turned from excused into proved.
            // Written under <golden>/eng/, NOT next to the reference goldens (gate finding: an
            // experimental run must not overwrite the reference set), and with the measured tie
            // error alongside so the proof script can check it is looking at THIS run's dumps.
            const std::string ed = gd + "/eng"; std::filesystem::create_directories(ed);
            std::ofstream f(ed + "/a2_engine_topk.i32", std::ios::binary | std::ios::trunc);
            f.write(reinterpret_cast<const char*>(got.data()), std::streamsize(got.size() * 4));
            std::ofstream g(ed + "/a2_engine_tie_queries.i32", std::ios::binary | std::ios::trunc);
            // q_differs is the local set; g_q_differs is only assigned after this block
            for (uint32_t t = 0; t < T; ++t) { const int32_t v = q_differs[t] ? int32_t(t) : -1; g.write(reinterpret_cast<const char*>(&v), 4); }
        }
        g_q_differs = q_differs;
    }

    // ---- C5 attention over [window ; compressed] ----------------------------------------------
    float *qq = f32(size_t(T) * QH), *kvw = f32(size_t(T) * HD), *kvwn = f32(size_t(T) * HD), *ckv_r = f32(size_t(NC) * HD);
    ie::gemm_nt_f16_onednn(q, t16, w_qb, qq, T, QH, QR).wait();          // t16 still holds qrn
    ie::ds4_rope_apply(q, qq, cos_t, sin_t, qq, T, NH, HD, RD, +1.0f).wait();
    ie::gemm_nt_f16_onednn(q, x16, w_kv, kvw, T, HD, H).wait();
    ie::ds4_rms_norm(q, kvw, n_kv, kvwn, T, HD, c.norm_eps).wait();
    ie::ds4_rope_apply(q, kvwn, cos_t, sin_t, kvwn, T, 1, HD, RD, +1.0f).wait();
    q.memcpy(ckv_r, latent, size_t(NC) * HD * 4).wait();
    ie::ds4_rope_apply(q, ckv_r, cos_g, sin_g, ckv_r, NC, 1, HD, RD, +1.0f).wait();
    {
        std::vector<float> got(size_t(NC) * HD); q.memcpy(got.data(), ckv_r, got.size() * 4).wait();
        const Cmp cc = cmp(got, rd<float>(gd + "/a2_ckv_roped.f32", got.size()));
        check(cc.rel < 3e-3, "C5 compressed KV (RoPE'd at group positions)", "rel " + std::to_string(cc.rel));
    }
    float *mwin = f32(size_t(T) * T), *mcomp = f32(size_t(T) * NC), *mask = f32(size_t(T) * NKV);
    ie::ds4_sliding_causal_mask(q, d_pos_t, mwin, T, T, WIN).wait();
    ie::ds4_block_bias_topk(q, topk, mcomp, T, NC, TOPK).wait();
    q.parallel_for(sycl::range<1>(size_t(T) * NKV), [=](sycl::id<1> i) {
        const uint32_t t = uint32_t(i[0] / NKV), k = uint32_t(i[0] % NKV);
        mask[i] = k < T ? mwin[size_t(t) * T + k] : mcomp[size_t(t) * NC + (k - T)];
    }).wait();
    float* oo = f32(size_t(T) * QH);
    ie::Ds4KvSegs segs; segs.a = kvwn; segs.n_a = T; segs.b = ckv_r; segs.n_b = NC; segs.b_f16 = false;
    ie::ds4_attention_segs(q, qq, segs, mask, sinks, oo, T, NH, HD, 1.0f / std::sqrt(float(HD))).wait();
    ie::ds4_rope_apply(q, oo, cos_t, sin_t, oo, T, NH, HD, RD, -1.0f).wait();
    float *oa = f32(size_t(T) * OR), *out = f32(size_t(T) * H);
    to16(oo, t16, size_t(T) * QH);
    ie::gemm_bmm_nt_f16_onednn(q, t16, w_oa, oa, T, G, IPG, OPG).wait();
    to16(oa, t16, size_t(T) * OR);
    ie::gemm_nt_f16_onednn(q, t16, w_ob, out, T, H, OR).wait();
    {
        std::vector<float> got(size_t(T) * H); q.memcpy(got.data(), out, got.size() * 4).wait();
        const auto g_out = rd<float>(gd + "/a2_out.f32", got.size());
        const Cmp co = cmp(got, g_out);
        // DIAGNOSTIC: is the error concentrated in the queries whose top-k differed?
        double scale = 0; for (float v : g_out) scale = std::max(scale, std::fabs(double(v)));
        double m_same = 0, m_diff = 0;
        for (uint32_t t = 0; t < T; ++t)
            for (uint32_t d = 0; d < H; ++d) {
                const double e = std::fabs(double(got[size_t(t) * H + d]) - double(g_out[size_t(t) * H + d]));
                if (!g_q_differs.empty() && g_q_differs[t]) m_diff = std::max(m_diff, e); else m_same = std::max(m_same, e);
            }
        std::printf("       C5 diagnostic: rel error over queries with the SAME top-k set = %.3e, over queries "
                    "whose set DIFFERED = %.3e\n", m_same / scale, m_diff / scale);
        {   // the engine's block output and its measured tie/same errors, for prove_ties.py
            const std::string ed = gd + "/eng";
            std::ofstream f(ed + "/a2_engine_out.f32", std::ios::binary | std::ios::trunc);
            f.write(reinterpret_cast<const char*>(got.data()), std::streamsize(got.size() * 4));
            std::ofstream j(ed + "/a2_engine_meta.json", std::ios::trunc);
            j << "{\"tie_rel\": " << m_diff / scale << ", \"same_rel\": " << m_same / scale
              << ", \"n_tie\": " << (g_q_differs.empty() ? 0 : std::count(g_q_differs.begin(), g_q_differs.end(), char(1)))
              << ", \"T\": " << T << "}\n";
        }
        // AMENDED (disclosed): the fp16 bar applies to queries whose selection matched; a
        // query that attended to a swapped near-tied entry is bounded separately.
        check(m_same / scale < 5e-3, "C5 attention output over queries with the same top-k set",
              "rel " + std::to_string(m_same / scale));
        check(m_diff / scale < 2e-2, "C5 attention output over boundary-tie queries (swapped a near-tied entry)",
              "rel " + std::to_string(m_diff / scale));
    }

    std::printf("\n%s\n", g_fail ? ("ATTN2 TEST: " + std::to_string(g_fail) + " FAILURE(S)").c_str() : "ATTN2 TEST: PASS");
    return g_fail ? 1 : 0;
}
