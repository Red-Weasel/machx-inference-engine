// tools/ds41_engram_test.cpp — V4.1 Phase 6a: engram vs the reference, stage by stage.
// Criteria: docs/deepseek41/15_PHASE6A_CRITERIA_2026-09-12.md
//   usage: ie-ds41-engram-test <model_dir> <golden_dir> [host]     ("host" skips the GPU stages)
#include "ie/deepseek41_engram.hpp"
#include "ie/deepseek41_upload.hpp"
#include "ie/ops.hpp"

#include "../third_party/nlohmann/json.hpp"

#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/resource.h>
#include <algorithm>
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
struct Cmp { double rel, max_abs; size_t worst; };
Cmp cmp(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0, s = 0; size_t w = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(double(a[i]) - double(b[i]));
        if (d > m) { m = d; w = i; }
        s = std::max(s, std::fabs(double(b[i])));
    }
    return {s > 0 ? m / s : 0.0, m, w};
}
inline float bf16(uint16_t h) { const uint32_t b = uint32_t(h) << 16; float f; std::memcpy(&f, &b, 4); return f; }
double rss_gib() { struct rusage u{}; getrusage(RUSAGE_SELF, &u); return double(u.ru_maxrss) / (1024.0 * 1024.0); }
}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : std::string(std::getenv("HOME")) + "/models/DeepSeek-V4.1-Flash";
    const std::string gd  = argc > 2 ? argv[2] : "/tmp/ds41_golden";
    const bool host_only  = argc > 3 && std::string(argv[3]) == "host";

    ie::DeepSeek41Model m;
    if (const auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    ie::Ds41EngramTables tb;
    if (const auto e = tb.load(gd); !e.empty()) { std::fprintf(stderr, "tables: %s\n", e.c_str()); return 1; }
    const auto& c = m.config();

    // golden meta: L tokens, model layer 1 == engram layer index 0
    nlohmann::json emeta; { std::ifstream f(gd + "/eng_meta.json"); f >> emeta; }
    const uint32_t L = emeta["L"], LAYER = emeta["layer"], LI = emeta["layer_hash_index"];
    const uint32_t NC = tb.n_hash_cols, HDm = tb.head_dim;
    const uint32_t H = c.dim, HC = c.hc_mult, IN = NC * HDm, OUTW = H * (HC + 1);
    const auto& Lw = m.layers()[LAYER];
    check(Lw.kind.has_engram && Lw.engram_embed.w && Lw.engram_wkv.w, "layer 1 has engram bound");
    std::printf("       tables: %zu layers, %u-gram, %u heads x %u, %u hash cols, pad %lld, "
                "token_map %zu\n", tb.layer_ids.size(), tb.max_ngram_size, tb.n_heads, HDm, NC,
                (long long)tb.pad_id, tb.token_map.size());

    // ---- C2/C3: hash, integer-exact, including positions 0..2 -------------------------------
    std::printf("\n=== C2/C3 hash ===\n");
    const auto ids = rd<int32_t>(gd + "/eng_ids.i32", L);
    const auto g_hash = rd<int64_t>(gd + "/eng_hashes.i64", size_t(L) * tb.layer_ids.size() * NC);
    std::vector<int64_t> hash(size_t(L) * NC);
    ds41_engram_hash(tb, ids.data(), L, LI, hash.data());
    size_t bad = 0, bad_head = 0;
    for (uint32_t t = 0; t < L; ++t)
        for (uint32_t j = 0; j < NC; ++j) {
            // golden layout [L, n_layers, NC]
            const int64_t g = g_hash[(size_t(t) * tb.layer_ids.size() + LI) * NC + j];
            if (hash[size_t(t) * NC + j] != g) { ++bad; if (t < 3) ++bad_head; }
        }
    check(bad == 0, "all " + std::to_string(size_t(L) * NC) + " hash ids equal the reference as integers",
          std::to_string(bad) + " differ");
    check(bad_head == 0, "positions 0, 1, 2 (look-back crosses the start) exact",
          std::to_string(bad_head) + " differ in the first 3 positions");
    std::printf("       t=0: %lld %lld ... | t=5: %lld %lld ...\n",
                (long long)hash[0], (long long)hash[1], (long long)hash[5 * NC], (long long)hash[5 * NC + 1]);

    // ---- C4: gather + dequant, exact --------------------------------------------------------
    std::printf("\n=== C4 gather ===\n");
    std::vector<float> emb(size_t(L) * IN);
    ds41_engram_gather(Lw.engram_embed, hash.data(), L, NC, HDm, emb.data());
    const auto g_emb = rd<float>(gd + "/eng_emb_fp32.f32", size_t(L) * IN);
    const Cmp ce = cmp(emb, g_emb);
    check(ce.max_abs == 0.0, "24 dequantised rows per position == reference, EXACTLY",
          "max|diff| " + std::to_string(ce.max_abs) + " over " + std::to_string(emb.size()) + " values");
    check(rss_gib() < 4.0, "peak RSS under 4 GiB against the 91.5 GB table",
          std::to_string(rss_gib()).substr(0, 5) + " GiB");
    if (host_only) {
        std::printf("\n%s\n", g_fail ? "ENGRAM HOST STAGES: FAILURE(S)" : "ENGRAM HOST STAGES: PASS (GPU stages skipped)");
        return g_fail ? 1 : 0;
    }

    // ---- C5/C6: wkv, gate, residual on device ----------------------------------------------
    std::printf("\n=== C5/C6 wkv + gate + residual (device) ===\n");
    sycl::device dev;
    for (const auto& p : sycl::platform::get_platforms())
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) { dev = d; goto found; }
    std::fprintf(stderr, "no Arc GPU\n"); return 1;
found:
    sycl::queue q(dev, sycl::property::queue::in_order{});

    // wkv: FP8 [25600, 6144] with a 32x32 grid -> fp16
    const auto& wk = Lw.engram_wkv;
    const uint32_t N = uint32_t(wk.w->shape[0]), K = uint32_t(wk.w->shape[1]);
    check(N == OUTW && K == IN, "wkv is [dim*(hc+1), n_hash_cols*head_dim]",
          std::to_string(N) + "x" + std::to_string(K));
    const uint32_t bn = N / uint32_t(wk.s->shape[0]), bk = K / uint32_t(wk.s->shape[1]);
    uint8_t* dw = sycl::malloc_device<uint8_t>(size_t(N) * K, q);
    uint8_t* ds = sycl::malloc_device<uint8_t>(wk.s->nbytes, q);
    sycl::half* w16 = sycl::malloc_device<sycl::half>(size_t(N) * K, q);
    q.memcpy(dw, wk.w->data, size_t(N) * K); q.memcpy(ds, wk.s->data, wk.s->nbytes).wait();
    ie::ds41_dense_dequant_f16(q, dw, ds, N, K, bn, bk, w16).wait();
    sycl::free(dw, q); sycl::free(ds, q);

    sycl::half* x16 = sycl::malloc_device<sycl::half>(size_t(L) * IN, q);
    {
        // the reference rounds the gathered rows to bf16 before wkv; the engine feeds them to
        // the fp16 GEMM, so the input rounding differs (bf16 vs fp16) -- both ~3 significant
        // decimal digits, and inside the 3e-3 the criteria set for this stage.
        std::vector<sycl::half> h16(emb.size());
        for (size_t i = 0; i < emb.size(); ++i) h16[i] = sycl::half(emb[i]);
        q.memcpy(x16, h16.data(), h16.size() * 2).wait();
    }
    float* kv = sycl::malloc_device<float>(size_t(L) * OUTW, q);
    ie::gemm_nt_f16_onednn(q, x16, w16, kv, L, OUTW, IN).wait();
    {
        std::vector<float> got(size_t(L) * OUTW);
        q.memcpy(got.data(), kv, got.size() * 4).wait();
        const Cmp ck = cmp(got, rd<float>(gd + "/eng_kv.f32", got.size()));
        check(ck.rel < 3e-3, "wkv output (keys + value) tracks the reference", "rel " + std::to_string(ck.rel));
    }

    // key = kv[:, :hc*dim] viewed [L, hc, dim]; value = kv[:, hc*dim:]. Both are already laid
    // out that way in the GEMM output row, so pass offsets rather than copying.
    const float* key = kv;                       // row stride OUTW, first HC*H entries
    const float* val = kv + size_t(HC) * H;      // row stride OUTW, last H entries
    // ds41_engram_gate wants contiguous [L, hc, H] and [L, H]; repack once.
    float* keyc = sycl::malloc_device<float>(size_t(L) * HC * H, q);
    float* valc = sycl::malloc_device<float>(size_t(L) * H, q);
    q.parallel_for(sycl::range<1>(size_t(L) * HC * H), [=](sycl::id<1> i) {
        const uint32_t t = uint32_t(i[0] / (size_t(HC) * H)), r = uint32_t(i[0] % (size_t(HC) * H));
        keyc[i] = key[size_t(t) * OUTW + r];
    });
    q.parallel_for(sycl::range<1>(size_t(L) * H), [=](sycl::id<1> i) {
        const uint32_t t = uint32_t(i[0] / H), d = uint32_t(i[0] % H);
        valc[i] = val[size_t(t) * OUTW + d];
    }).wait();

    std::vector<float> qk(size_t(HC) * H);
    {
        const auto* qw = reinterpret_cast<const uint16_t*>(Lw.engram_q.w->data);
        const auto* kw = reinterpret_cast<const uint16_t*>(Lw.engram_k.w->data);
        for (size_t i = 0; i < qk.size(); ++i) qk[i] = bf16(qw[i]) * bf16(kw[i]);
    }
    float* dqk = sycl::malloc_device<float>(qk.size(), q);
    q.memcpy(dqk, qk.data(), qk.size() * 4).wait();
    const auto g_h = rd<float>(gd + "/eng_h_in.f32", size_t(L) * HC * H);
    float* dh = sycl::malloc_device<float>(g_h.size(), q);
    q.memcpy(dh, g_h.data(), g_h.size() * 4).wait();
    float* dout = sycl::malloc_device<float>(g_h.size(), q);
    float* dgate = sycl::malloc_device<float>(size_t(L) * HC, q);
    ie::ds41_engram_gate(q, dh, keyc, valc, dqk, dout, dgate, L, H, HC, c.norm_eps).wait();

    {
        std::vector<float> gg(size_t(L) * HC), oo(g_h.size());
        q.memcpy(gg.data(), dgate, gg.size() * 4);
        q.memcpy(oo.data(), dout, oo.size() * 4).wait();
        const auto g_gate = rd<float>(gd + "/eng_gate.f32", gg.size());
        const auto g_dot  = rd<float>(gd + "/eng_dot.f32", gg.size());
        size_t neg = 0, big = 0;
        for (float d : g_dot) { if (d < 0) ++neg; if (std::fabs(d) > 1e-6f) ++big; }
        check(neg > 0 && neg < g_dot.size() && big == g_dot.size(),
              "the signed sqrt is exercised: dots of both signs, all |dot| > clamp",
              std::to_string(neg) + " negative of " + std::to_string(g_dot.size()));
        const Cmp cg = cmp(gg, g_gate);
        check(cg.rel < 3e-3, "gate (on its own) tracks the reference", "rel " + std::to_string(cg.rel) +
              ", gate range " + std::to_string(*std::min_element(gg.begin(), gg.end())).substr(0, 6) + ".." +
              std::to_string(*std::max_element(gg.begin(), gg.end())).substr(0, 6));
        const Cmp co = cmp(oo, rd<float>(gd + "/eng_out.f32", oo.size()));
        check(co.rel < 3e-3, "engram output stream tracks the reference", "rel " + std::to_string(co.rel));
    }
    for (void* p : {(void*)w16,(void*)x16,(void*)kv,(void*)keyc,(void*)valc,(void*)dqk,(void*)dh,(void*)dout,(void*)dgate})
        sycl::free(p, q);

    std::printf("\n%s\n", g_fail ? ("ENGRAM TEST: " + std::to_string(g_fail) + " FAILURE(S)").c_str() : "ENGRAM TEST: PASS");
    return g_fail ? 1 : 0;
}
