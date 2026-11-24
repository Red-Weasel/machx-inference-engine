// Inkling 2 Mini phase-1 ops (include/ie/inkling2_ops.hpp) against values produced by the numpy reference's own
// functions (scripts/inkling2/gen_ops_fixtures.py -> tests/unit/inkling2_ops_fixtures.inc), plus the host-side config
// reader, call schedule and cache-entry map (include/ie/inkling2.hpp). Small synthetic tensors: no model file.
//
// Covers what the 5-token golden prompt cannot (SPEC 16.7): the sliding window mask, an exact top-8 tie, and the
// conv / attention state carried across two forward chunks (the incremental path the KV-cached decode will use).
// Needs one GPU (a few MB). Bounds are fp32-accumulation-order bounds; a negative control at the end shows the
// window check discriminates.
#undef NDEBUG
#include "ie/fp8.hpp"
#include "ie/inkling2.hpp"
#include "ie/inkling2_ops.hpp"

#include <sycl/sycl.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace {

struct Fix { const char* name; std::vector<float> v; };
const Fix kFix[] = {
#include "inkling2_ops_fixtures.inc"
};
const std::vector<float>& F(const std::string& n) {
    for (const auto& f : kFix) if (n == f.name) return f.v;
    std::fprintf(stderr, "missing fixture %s\n", n.c_str());
    std::exit(2);
}

int g_fail = 0;
bool check(const char* what, const std::vector<float>& got, const std::vector<float>& want, float atol, float rtol,
           size_t off = 0, size_t n = 0) {
    if (!n) n = want.size() - off;
    double worst = 0;
    size_t bad = 0;
    for (size_t i = 0; i < n; ++i) {
        const float g = got[i], w = want[off + i];
        const double d = std::fabs(double(g) - double(w));
        worst = std::max(worst, d);
        if (!(d <= atol + rtol * std::fabs(double(w)))) ++bad;
    }
    std::printf("%-44s %s  max|d| %.3g  (%zu values)\n", what, bad ? "FAIL" : "ok  ", worst, n);
    if (bad) ++g_fail;
    return !bad;
}
std::vector<uint8_t> bytes(const std::vector<float>& v) {
    std::vector<uint8_t> b(v.size());
    for (size_t i = 0; i < v.size(); ++i) b[i] = uint8_t(v[i]);
    return b;
}
std::vector<int32_t> ints(const std::vector<float>& v) {
    std::vector<int32_t> b(v.size());
    for (size_t i = 0; i < v.size(); ++i) b[i] = int32_t(v[i]);
    return b;
}

struct Dev {
    sycl::queue& q;
    std::vector<void*> all;
    template <class T> T* up(const std::vector<T>& h) {
        T* d = sycl::malloc_device<T>(h.size() ? h.size() : 1, q);
        all.push_back(d);
        if (!h.empty()) q.memcpy(d, h.data(), h.size() * sizeof(T)).wait();
        return d;
    }
    float* zeros(size_t n) { return up(std::vector<float>(n, 0.f)); }
    template <class T> std::vector<T> down(const T* d, size_t n) {
        std::vector<T> h(n);
        q.memcpy(h.data(), d, n * sizeof(T)).wait();
        return h;
    }
    ~Dev() { for (void* p : all) sycl::free(p, q); }
};

const char* kCfg = R"({"text_config": {
  "hidden_size": 2304, "num_hidden_layers": 24, "vocab_size": 201024, "unpadded_vocab_size": 200058,
  "num_attention_heads": 24, "num_key_value_heads": 8, "head_dim": 128, "d_rel": 16, "rel_extent": 1024,
  "sliding_window_size": 1024, "sconv_kernel_size": 4, "n_layers_per_attn_res_block": 6, "n_routed_experts": 128,
  "num_experts_per_tok": 8, "n_shared_experts": 0, "intermediate_size": 1152, "log_scaling_n_floor": 8192,
  "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-06, "qk_l2_clip_threshold": 22.627416997969522,
  "rel_l2_clip_threshold": 8.0, "route_scale": 8.0, "mlp_gate_softcap": 4.0, "mlp_up_softcap": 25.0,
  "final_logit_softcapping": 30.0, "use_value_norm_no_gamma": true, "attn_res_normalize_values": true,
  "local_layer_ids": [0,1,2,3,4,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22],
  "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64, "is_varloop": false}}})";

void host_tests() {
    // bf16 rounding (ties to even) and the FP8 / FP4 decoders
    {
        const auto& x = F("bf16.x");
        std::vector<float> y(x.size());
        for (size_t i = 0; i < x.size(); ++i) y[i] = ie::i2_bf16_round(x[i]);
        check("bf16 round (ties to even)", y, F("bf16.y"), 0, 0);
    }
    {
        const auto& t = F("e4m3");
        size_t bad = 0;
        for (int b = 0; b < 256; ++b) {
            const float a = ie::i2_e4m3(uint8_t(b)), r = t[b], h = ie::e4m3_to_f32(uint8_t(b));
            const bool same = (std::isnan(r) && std::isnan(a) && std::isnan(h)) ||
                              (std::memcmp(&a, &r, 4) == 0 && std::memcmp(&a, &h, 4) == 0);
            if (!same) ++bad;
        }
        std::printf("%-44s %s  (256 codes vs numpy table and fp8.hpp)\n", "E4M3 decode", bad ? "FAIL" : "ok  ");
        if (bad) ++g_fail;
        const float e2m1[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -0.5f, -1, -1.5f, -2, -3, -4, -6};
        bad = 0;
        for (uint32_t c = 0; c < 16; ++c) { const float v = ie::i2_e2m1(c); if (std::memcmp(&e2m1[c], &v, 4) != 0) ++bad; }
        bad += ie::i2_ue8m0(127) != 1.f || ie::i2_ue8m0(118) != 0.001953125f;
        std::printf("%-44s %s\n", "E2M1 / UE8M0 decode", bad ? "FAIL" : "ok  ");
        if (bad) ++g_fail;
    }
    // config, schedule, cache entries (SPEC 0, 5.1, 6)
    {
        ie::Inkling2Config c;
        int bad = 0;
        if (auto e = ie::inkling2_parse_config(kCfg, c); !e.empty()) { std::printf("config: %s\n", e.c_str()); ++bad; }
        bad += c.n_calls() != 48;
        for (uint32_t L : {5u, 11u, 17u, 23u}) bad += !c.is_global(L);
        bad += c.is_global(0) || c.is_global(6) || c.is_global(18);
        bad += c.entry(3, 1) != 3 || c.entry(6, 1) != 6 || c.entry(17, 1) != 17 || c.entry(20, 1) != 20;
        bad += c.entry(6, 2) != 24 || c.entry(17, 2) != 35 || c.entry(6, 3) != 36 || c.entry(17, 3) != 47;
        const auto s = ie::inkling2_schedule(c);
        bad += s.size() != 48 || s[5].layer != 5 || s[6].layer != 6 || s[6].pass != 1 || s[18].layer != 6 ||
               s[18].pass != 2 || s[41].layer != 17 || s[41].pass != 3 || s[42].layer != 18 || s[47].layer != 23;
        std::vector<bool> used(48, false);                  // every (layer, pass) call has its own entry
        for (const auto& k : s) { bad += used[c.entry(k.layer, k.pass)]; used[c.entry(k.layer, k.pass)] = true; }
        std::string j = kCfg;
        ie::Inkling2Config c2;
        bad += ie::inkling2_parse_config(std::string(j).replace(j.find("\"n_shared_experts\": 0"), 21, "\"n_shared_experts\": 2"), c2).empty();
        bad += ie::inkling2_parse_config(std::string(j).replace(j.find("[0,1,2"), 6, "[0,1,3"), c2).empty();
        std::printf("%-44s %s\n", "config / schedule / entry map / refusals", bad ? "FAIL" : "ok  ");
        if (bad) ++g_fail;
    }
}

void device_tests(sycl::queue& q) {
    Dev d{q, {}};
    const float tol = 2e-5f;
    // bf16 linear: every kernel variant (T = 1, 2..4 sub-group path; T = 5 and K % 8 != 0 the generic one), against
    // a host fp64 sum, with a row count that is not a multiple of the 8 rows per work-group
    for (const auto [T, K, N] : {std::array<uint32_t, 3>{1, 2304, 37}, {3, 64, 21}, {4, 136, 9}, {5, 64, 7}, {2, 36, 5}, {1, 4608, 64}, {3, 2304, 600}}) {
        std::vector<float> x(size_t(T) * K), want(size_t(T) * N);
        std::vector<uint16_t> w(size_t(N) * K);
        uint32_t s = 12345u + K;
        auto rnd = [&] { s = s * 1664525u + 1013904223u; return float(int32_t(s >> 8) - (1 << 23)) / float(1 << 23); };
        for (auto& v : x) v = rnd();
        for (auto& v : w) { const float f = rnd(); uint32_t u; std::memcpy(&u, &f, 4); v = uint16_t(u >> 16); }
        for (uint32_t t = 0; t < T; ++t)
            for (uint32_t n = 0; n < N; ++n) {
                double a = 0;
                for (uint32_t k = 0; k < K; ++k) {
                    const uint32_t u = uint32_t(w[size_t(n) * K + k]) << 16;
                    float f; std::memcpy(&f, &u, 4);
                    a += double(x[size_t(t) * K + k]) * f;
                }
                want[size_t(t) * N + n] = float(a);
            }
        float* dx = d.up(x); uint16_t* dw = d.up(w); float* y = d.zeros(want.size());
        ie::i2_linear_bf16(q, dx, T, K, dw, N, y, false);
        const std::string name = "bf16 linear T=" + std::to_string(T) + " K=" + std::to_string(K) + " N=" + std::to_string(N);
        check(name.c_str(), d.down(y, want.size()), want, 1e-5f, 1e-5f);
    }
    // rmsnorm
    {
        const auto& x = F("rms.x");
        float* dx = d.up(x); float* dw = d.up(F("rms.w")); float* y = d.zeros(x.size());
        ie::i2_rmsnorm(q, dx, 3, 40, dw, 1e-6f, y, 40, false);
        check("rmsnorm, weighted", d.down(y, x.size()), F("rms.yw"), tol, tol);
        ie::i2_rmsnorm(q, dx, 3, 40, nullptr, 1e-6f, y, 40, false);
        check("rmsnorm, unweighted", d.down(y, x.size()), F("rms.y"), tol, tol);
    }
    // scale + L2 clip (in place)
    {
        const auto& x = F("clip.x");
        float* dx = d.up(x);
        ie::i2_scale_l2clip(q, dx, 6, 8, d.up(F("clip.scale")), 3, 2.0f, dx, false);
        check("scale + L2 clip", d.down(dx, x.size()), F("clip.y"), tol, tol);
    }
    // static conv over a position-indexed input ring: whole sequence; 4 + 3 tokens; and a rewind: feed 7, pretend
    // positions 5.. never happened (truncate to 5), feed 5 and 6 again with garbage written in between
    {
        const auto& x = F("sconv.x");
        const uint32_t T = 7, C = 5, rows = 8;                  // rows >= rewind depth + 3
        float* dx = d.up(x); float* w = d.up(F("sconv.w")); float* y = d.zeros(T * C);
        float* h1 = d.zeros(rows * C);
        ie::i2_sconv_static(q, dx, T, C, w, h1, rows, 0, y, false);
        check("static conv (whole)", d.down(y, T * C), F("sconv.y"), tol, tol);
        float* h2 = d.up(std::vector<float>(rows * C, 123.f));  // stale contents must never be read before position 0
        ie::i2_sconv_static(q, dx, 4, C, w, h2, rows, 0, y, false);
        ie::i2_sconv_static(q, dx + 4 * C, 3, C, w, h2, rows, 4, y + 4 * C, false);
        check("static conv (4 + 3, input ring)", d.down(y, T * C), F("sconv.y"), tol, tol);
        float* junk = d.up(std::vector<float>(3 * C, -7.f));
        ie::i2_sconv_static(q, junk, 3, C, w, h2, rows, 7, y, false);     // positions 7..9 (rewound below)
        ie::i2_sconv_static(q, dx + 5 * C, 2, C, w, h2, rows, 5, y + 5 * C, false);
        check("static conv (rewind to 5, refeed 5..6)", d.down(y + 5 * C, 2 * C), F("sconv.y"), tol, tol, 5 * C, 2 * C);
    }
    // dynamic conv: whole, then 2 + 5 (fewer than 3 rows in the first chunk), input ring of 4 rows (the minimum + 1)
    {
        const auto& x = F("dconv.x");
        const uint32_t T = 7, C = 8, rows = 8;
        float* dx = d.up(x); float* dyn = d.up(F("dconv.dyn")); float* sw = d.up(F("dconv.sw")); float* y = d.zeros(T * C);
        float* h1 = d.zeros(rows * C);
        ie::i2_sconv_dynamic(q, dx, T, C, dyn, sw, 2, h1, rows, 0, y, false);
        check("dynamic conv (whole)", d.down(y, T * C), F("dconv.y"), tol, tol);
        float* h2 = d.zeros(rows * C);
        ie::i2_sconv_dynamic(q, dx, 2, C, dyn, sw, 2, h2, rows, 0, y, false);
        ie::i2_sconv_dynamic(q, dx + 2 * C, 5, C, dyn + 2 * 8, sw, 2, h2, rows, 2, y + 2 * C, false);
        check("dynamic conv (2 + 5, input ring)", d.down(y, T * C), F("dconv.y"), tol, tol);
    }
    // attn_res
    {
        ie::I2Sources s{};
        for (int i = 0; i < 3; ++i) s.p[i] = d.up(F("ares.s" + std::to_string(i)));
        float* y = d.zeros(3 * 4 * 32);
        ie::i2_attn_res(q, s, 3, 4, 32, d.up(F("ares.q")), 3, 1e-6f, y);
        check("attn_res (3 sources, 3 queries)", d.down(y, 3 * 4 * 32), F("ares.y"), tol, tol);
        ie::i2_attn_res(q, s, 3, 4, 32, d.up(F("ares.q")), 3, 1e-6f, y, true);
        check("attn_res fast (3 sources, 3 queries)", d.down(y, 3 * 4 * 32), F("ares.y"), tol, tol);
        ie::i2_attn_res(q, s, 3, 4, 32, d.up(F("ares.q")), 3, 1e-6f, y, true, d.zeros(4 * 16 * 40));
        check("attn_res fast, sliced (3 sources, 3 queries)", d.down(y, 3 * 4 * 32), F("ares.y"), tol, tol);
    }
    // attention core (relative bias formed in the kernel, extent 4 < T): global and window 3; whole, then the last 2
    // queries against cached keys (pos0 = 4), then window 3 through ring entries fed in chunks of 2 and one at a time
    {
        const uint32_t T = 6, nh = 4, nkv = 2, hd = 8, dr = 4, ext = 4, KD = nkv * hd;
        float* Q = d.up(F("att.q")); float* K = d.up(F("att.k")); float* V = d.up(F("att.v"));
        float* rel = d.up(F("att.rel")); float* proj = d.up(F("att.proj")); float* sink = d.up(F("att.sink"));
        float* o = d.zeros(T * nh * hd);
        for (auto [name, win] : {std::pair<const char*, uint32_t>{"glob", 0}, {"win3", 3}}) {
            const std::string key = std::string("att.o_") + name;
            ie::i2_attention(q, Q, K, V, 0, rel, proj, dr, ext, sink, o, T, 0, nh, nkv, hd, win, false);
            check(("attention " + std::string(name) + " (whole)").c_str(), d.down(o, T * nh * hd), F(key), 1e-5f, 1e-4f);
            ie::i2_attention(q, Q + 4 * nh * hd, K, V, 0, rel + 4 * nh * dr, proj, dr, ext, sink + 4 * nh, o, 2, 4, nh, nkv, hd,
                             win, false);
            check(("attention " + std::string(name) + " (2 queries at pos 4, cached)").c_str(), d.down(o, 2 * nh * hd), F(key),
                  1e-5f, 1e-4f, 4 * nh * hd, 2 * nh * hd);
        }
        for (uint32_t step : {2u, 1u}) {                        // ring rows = window + chunk: the smallest the runtime allows
            const uint32_t ring = 3 + step;
            float* Kr = d.zeros(ring * KD); float* Vr = d.zeros(ring * KD);
            std::vector<float> got(T * nh * hd);
            for (uint32_t p0 = 0; p0 < T; p0 += step) {
                ie::i2_store_rows(q, K + p0 * KD, step, KD, Kr, p0, ring);
                ie::i2_store_rows(q, V + p0 * KD, step, KD, Vr, p0, ring);
                ie::i2_attention(q, Q + p0 * nh * hd, Kr, Vr, ring, rel + p0 * nh * dr, proj, dr, ext, sink + p0 * nh, o, step, p0,
                                 nh, nkv, hd, 3, false);
                const auto part = d.down(o, step * nh * hd);
                std::copy(part.begin(), part.end(), got.begin() + p0 * nh * hd);
            }
            check(("attention win3 through a " + std::to_string(ring) + "-row ring, chunks of " + std::to_string(step)).c_str(),
                  got, F("att.o_win3"), 1e-5f, 1e-4f);
        }
        // split-K attention (fp32 decode path): whole, the cached 2 queries, a ring fed one at a time, and a part buffer
        // small enough to force one split per query (the merge then sees a single partial)
        for (uint64_t pf : {uint64_t(1) << 16, uint64_t(T * nh * (hd + 2))}) {
            float* part = d.zeros(pf);
            for (auto [name, win] : {std::pair<const char*, uint32_t>{"glob", 0}, {"win3", 3}}) {
                const std::string key = std::string("att.o_") + name, tag = " (split, " + std::to_string(ie::i2_attention_splits(T, 0, nh, hd, win, pf, 2)) + " max)";
                ie::i2_attention_split(q, Q, K, V, 0, rel, proj, dr, ext, sink, o, T, 0, nh, nkv, hd, win, part, pf, 2);
                check(("attention " + std::string(name) + tag).c_str(), d.down(o, T * nh * hd), F(key), 1e-5f, 1e-4f);
            }
            const uint32_t ring = 4;
            float* Kr = d.zeros(ring * KD); float* Vr = d.zeros(ring * KD);
            std::vector<float> got(T * nh * hd);
            for (uint32_t p0 = 0; p0 < T; ++p0) {
                ie::i2_store_rows(q, K + p0 * KD, 1, KD, Kr, p0, ring);
                ie::i2_store_rows(q, V + p0 * KD, 1, KD, Vr, p0, ring);
                ie::i2_attention_split(q, Q + p0 * nh * hd, Kr, Vr, ring, rel + p0 * nh * dr, proj, dr, ext, sink + p0 * nh, o, 1,
                                       p0, nh, nkv, hd, 3, part, pf, 1);
                const auto part_o = d.down(o, nh * hd);
                std::copy(part_o.begin(), part_o.end(), got.begin() + p0 * nh * hd);
            }
            check("attention win3 split, 4-row ring, one at a time", got, F("att.o_win3"), 1e-5f, 1e-4f);
        }
        // the model's head shape (hd 128, 3 query heads per KV head, d_rel 16): split vs the three-pass kernel for one
        // query at position 700 (global, window 300 through a 556-row ring) and two queries at 64..65 (prefill-like)
        {
            const uint32_t nh2 = 6, nkv2 = 2, hd2 = 128, dr2 = 16, ext2 = 200, P2 = 702, KD2 = nkv2 * hd2;
            uint32_t st = 99u;
            auto rf = [&] { st = st * 1664525u + 1013904223u; return float(int32_t(st >> 8) - (1 << 23)) / float(1 << 23); };
            std::vector<float> hQ(2 * nh2 * hd2), hK(size_t(P2) * KD2), hV(size_t(P2) * KD2), hR(2 * nh2 * dr2), hP(dr2 * ext2), hS(2 * nh2);
            for (auto* v : {&hQ, &hK, &hV, &hR, &hP, &hS}) for (auto& x : *v) x = rf();
            for (auto& x : hQ) x *= 0.3f;
            float *Q2 = d.up(hQ), *K2 = d.up(hK), *V2 = d.up(hV), *R2 = d.up(hR), *P2d = d.up(hP), *S2 = d.up(hS);
            float *o1 = d.zeros(2 * nh2 * hd2), *o2 = d.zeros(2 * nh2 * hd2), *part = d.zeros(1u << 18);
            const uint32_t ring2 = 556;
            float *Kr = d.zeros(size_t(ring2) * KD2), *Vr = d.zeros(size_t(ring2) * KD2);
            ie::i2_store_rows(q, K2, P2, KD2, Kr, 0, ring2);                 // positions 146.. survive in the ring
            ie::i2_store_rows(q, V2, P2, KD2, Vr, 0, ring2);
            for (auto [name, win, T2, p0, rg] : {std::tuple<const char*, uint32_t, uint32_t, uint32_t, uint32_t>{"glob, 1 query at 700", 0, 1, 700, 0},
                                                 {"win300 ring, 1 query at 700", 300, 1, 700, ring2},
                                                 {"glob, 2 queries at 64", 0, 2, 64, 0}}) {
                const float* Kc = rg ? Kr : K2; const float* Vc = rg ? Vr : V2;
                ie::i2_attention(q, Q2, Kc, Vc, rg, R2, P2d, dr2, ext2, S2, o1, T2, p0, nh2, nkv2, hd2, win, false);
                ie::i2_attention_split(q, Q2, Kc, Vc, rg, R2, P2d, dr2, ext2, S2, o2, T2, p0, nh2, nkv2, hd2, win, part, 1u << 18);
                check((std::string("attention hd 128 split vs three-pass, ") + name).c_str(), d.down(o2, T2 * nh2 * hd2),
                      d.down(o1, T2 * nh2 * hd2), 1e-5f, 1e-4f);
            }
        }
        // negative control: the window must change the answer by far more than the bound
        const auto& g = F("att.o_glob"); const auto& w = F("att.o_win3");
        double mx = 0;
        for (size_t i = 0; i < g.size(); ++i) mx = std::max(mx, std::fabs(double(g[i]) - double(w[i])));
        std::printf("%-44s %s  max|glob - win3| %.3g\n", "negative control: window discriminates", mx > 1e-2 ? "ok  " : "FAIL", mx);
        if (!(mx > 1e-2)) ++g_fail;
    }
    // router: own picks with an exact tie, forced picks, margins
    {
        const uint32_t T = 3, E = 16, K = 8;
        float* lg = d.up(F("rt.logits")); float* b = d.up(F("rt.bias"));
        int32_t* used = d.up(std::vector<int32_t>(T * K)); int32_t* own = d.up(std::vector<int32_t>(T * K));
        float* w = d.zeros(T * K); float* mg = d.zeros(T);
        ie::i2_router(q, lg, b, T, E, K, 8.f, nullptr, used, w, own, mg);
        auto ow = d.down(own, T * K), us = d.down(used, T * K);
        std::vector<float> owf(ow.begin(), ow.end()), usf(us.begin(), us.end());
        check("router own top-8 (exact tie -> lower id)", owf, F("rt.own"), 0, 0);
        check("router used == own when not forced", usf, F("rt.own"), 0, 0);
        check("router weights", d.down(w, T * K), F("rt.w_own"), 1e-6f, 1e-5f);
        check("router 8th-9th margin", d.down(mg, T), F("rt.margin"), 1e-6f, 1e-5f);
        int32_t* forced = d.up(ints(F("rt.forced")));
        ie::i2_router(q, lg, b, T, E, K, 8.f, forced, used, w, own, mg);
        us = d.down(used, T * K);
        check("router forced picks (sorted)", std::vector<float>(us.begin(), us.end()), F("rt.forced_sorted"), 0, 0);
        check("router forced weights", d.down(w, T * K), F("rt.w_forced"), 1e-6f, 1e-5f);
    }
    // expert banks
    {
        const uint32_t E = 3, rows = 4;
        const auto& amax = F("nv.amax");
        std::vector<float> fac(E);
        for (uint32_t e = 0; e < E; ++e) fac[e] = amax[e] / 2688.0f;
        ie::I2ExpertBank nv;
        nv.fmt = 0; nv.rows = rows; nv.cols = 32;
        nv.w = d.up(bytes(F("nv.codes"))); nv.s = d.up(bytes(F("nv.scale"))); nv.fac = d.up(fac);
        float* y = d.zeros(2 * 2 * rows);
        ie::i2_expert_matvec(q, nv, d.up(F("nv.x")), false, d.up(ints(F("nv.idx"))), 2, 2, y, false, false);
        check("NVFP4 expert matvec (per-token input)", d.down(y, 16), F("nv.y"), 1e-5f, 1e-5f);
        ie::I2ExpertBank mx;
        mx.fmt = 1; mx.rows = rows; mx.cols = 64;
        mx.w = d.up(bytes(F("mx.vals"))); mx.s = d.up(bytes(F("mx.exp")));
        ie::i2_expert_matvec(q, mx, d.up(F("mx.x")), true, d.up(ints(F("mx.idx"))), 2, 2, y, false, false);
        check("MXFP8 expert matvec (per-slot input)", d.down(y, 16), F("mx.y"), 1e-5f, 1e-5f);
        // full-width rows (every lane loads several 16-byte chunks), 9 rows (not a multiple of 8), against a host sum of
        // the same per-element weights
        for (int fmt = 0; fmt < 2; ++fmt) {
            const uint32_t R2 = 9, C2 = 1152, E2 = 3, T2 = 2, K2 = 2;
            uint32_t st = 777u + fmt;
            auto rb = [&] { st = st * 1664525u + 1013904223u; return uint8_t(st >> 24); };
            std::vector<uint8_t> wv(size_t(E2) * R2 * (fmt == 0 ? C2 / 2 : C2)), sv(size_t(E2) * R2 * (fmt == 0 ? C2 / 16 : C2 / 32));
            for (auto& b : wv) { b = rb(); if (fmt == 1 && (b & 0x7f) == 0x7f) b ^= 1; }      // no E4M3 NaN
            for (auto& b : sv) b = fmt == 0 ? uint8_t(0x30 + (rb() & 15)) : uint8_t(118 + (rb() & 7));
            std::vector<float> fc = {0.5f, 0.25f, 0.75f}, xv(size_t(T2) * C2);
            for (auto& v : xv) v = float(int(rb()) - 128) / 64.f;
            std::vector<int32_t> ix = {2, 0, 1, 2};
            std::vector<float> want(size_t(T2) * K2 * R2);
            for (uint32_t sl = 0; sl < T2 * K2; ++sl)
                for (uint32_t r = 0; r < R2; ++r) {
                    const uint64_t e = uint64_t(ix[sl]);
                    double a = 0;
                    for (uint32_t c = 0; c < C2; ++c) {
                        float wgt;
                        if (fmt == 0) {
                            const uint8_t byte = wv[(e * R2 + r) * (C2 / 2) + c / 2];
                            wgt = (ie::i2_e2m1(c & 1 ? byte >> 4 : byte & 15u) * ie::i2_e4m3(sv[(e * R2 + r) * (C2 / 16) + c / 16])) * fc[e];
                        } else {
                            wgt = ie::i2_e4m3(wv[(e * R2 + r) * C2 + c]) * ie::i2_ue8m0(sv[(e * R2 + r) * (C2 / 32) + c / 32]);
                        }
                        a += double(wgt) * xv[size_t(sl / K2) * C2 + c];
                    }
                    want[size_t(sl) * R2 + r] = float(a);
                }
            ie::I2ExpertBank bk;
            bk.fmt = fmt; bk.rows = R2; bk.cols = C2;
            bk.w = d.up(wv); bk.s = d.up(sv); bk.fac = d.up(fc);
            float* y2 = d.zeros(want.size());
            ie::i2_expert_matvec(q, bk, d.up(xv), false, d.up(ix), T2, K2, y2, false, false);
            check(fmt == 0 ? "NVFP4 expert matvec, 1152 cols, 9 rows" : "MXFP8 expert matvec, 1152 cols, 9 rows",
                  d.down(y2, want.size()), want, 2e-3f, 1e-5f);
            // prefill grouping: 21 tokens x 2 picks over 3 experts (several tiles per expert, a partial last tile),
            // grouped kernel vs the few-token kernel, per-token input and per-slot input
            {
                const uint32_t T3 = 21, K3 = 2, TK = T3 * K3;
                std::vector<int32_t> ix3(TK);
                for (uint32_t i = 0; i < TK; ++i) ix3[i] = int32_t(rb() % E2);
                std::vector<float> x3(size_t(TK) * C2);
                for (auto& v : x3) v = float(int(rb()) - 128) / 64.f;
                int32_t* dix = d.up(ix3);
                int32_t* order = d.up(std::vector<int32_t>(TK, -1));
                int32_t* tiles = d.up(std::vector<int32_t>(3 * (TK / ie::kI2MoeTile + E2 + 1), -1));
                int32_t* nt = d.up(std::vector<int32_t>(1, 0));
                ie::i2_moe_group(q, dix, T3, K3, E2, ie::kI2MoeTile, order, tiles, nt);
                const uint32_t maxt = (TK + ie::kI2MoeTile - 1) / ie::kI2MoeTile + E2;
                float* dx3 = d.up(x3);
                for (bool per_slot : {false, true}) {
                    float *ya = d.zeros(size_t(TK) * R2), *yb = d.zeros(size_t(TK) * R2);
                    ie::i2_expert_matvec(q, bk, dx3, per_slot, dix, T3, K3, ya, false, false);
                    ie::i2_expert_grouped(q, bk, dx3, per_slot, K3, order, tiles, nt, maxt, yb, false);
                    check((std::string(fmt == 0 ? "NVFP4" : "MXFP8") + " grouped (21 x 2 slots, 3 experts" +
                           (per_slot ? ", per-slot input)" : ")")).c_str(),
                          d.down(yb, size_t(TK) * R2), d.down(ya, size_t(TK) * R2), 1e-6f, 1e-6f);
                }
            }
        }
    }
    // softcapped SwiGLU
    {
        float* gu = d.up(F("act.gu")); float* a = d.zeros(12);
        ie::i2_expert_act(q, gu, 2, 6, 4.f, 25.f, a, false);
        check("softcapped SwiGLU (interleaved rows)", d.down(a, 12), F("act.y"), 1e-5f, 1e-4f);
    }
}

}  // namespace


// Phase-4 fused kernels against the separate kernels they replace (random data; fp32 reduction order only).
void fused_tests(sycl::queue& q) {
    Dev d{q, {}};
    uint32_t st = 4242u;
    auto rf = [&] { st = st * 1664525u + 1013904223u; return float(int32_t(st >> 8) - (1 << 23)) / float(1 << 23); };
    auto rv = [&](size_t n, float sc = 1.f) { std::vector<float> v(n); for (auto& x : v) x = rf() * sc; return v; };
    const float tol = 2e-5f;
    // attention prep: static (window layer, ring) and dynamic (global, tau), T = 3 at pos0 = 5 with history
    for (int glob = 0; glob < 2; ++glob) {
        const uint32_t T = 3, pos0 = 5, NH = 6, NKV = 2, HD = 128, DR = 16, KD = NKV * HD, crow = 9, ring = glob ? 0 : 12;
        const uint32_t cache_rows = ring ? ring : 16;
        auto hq = rv(T * NH * HD), hr = rv(T * NH * DR), hk = rv(T * KD), hv = rv(T * KD), hqs = rv(NH * HD), hrs = rv(NH * DR);
        auto hck = rv(crow * KD), hcv = rv(crow * KD), hkc = rv(glob ? 4 * KD : KD * 4, 0.3f), hvc = rv(glob ? 4 * KD : KD * 4, 0.3f);
        auto hdk = rv(T * NKV * 4, 0.3f), hdv = rv(T * NKV * 4, 0.3f), htau = std::vector<float>{1.1f, 1.2f, 1.3f};
        for (auto& x : hq) x *= 3.f;                                    // some heads above the clip
        // reference: the separate kernels
        float *qa = d.up(hq), *ra = d.up(hr), *cka = d.up(hck), *cva = d.up(hcv), *kca = d.zeros(T * KD), *vca = d.zeros(T * KD);
        float *Ka = d.zeros(cache_rows * KD), *Va = d.zeros(cache_rows * KD);
        float *kp = d.up(hk), *vp = d.up(hv), *qs = d.up(hqs), *rs = d.up(hrs), *kcw = d.up(hkc), *vcw = d.up(hvc);
        float *dk = d.up(hdk), *dv = d.up(hdv), *tau = d.up(htau);
        const float clip = 22.6f, rclip = 8.f, eps = 1e-6f;
        ie::i2_scale_l2clip(q, qa, T * NH, HD, qs, NH, clip, qa, false);
        if (glob) ie::i2_mul_token(q, qa, T, NH * HD, tau, qa, false);
        if (glob) ie::i2_sconv_dynamic(q, kp, T, KD, dk, kcw, NKV, cka, crow, pos0, kca, false);
        else ie::i2_sconv_static(q, kp, T, KD, kcw, cka, crow, pos0, kca, false);
        ie::i2_scale_l2clip(q, kca, T * NKV, HD, nullptr, NKV, clip, kca, false);
        ie::i2_store_rows(q, kca, T, KD, Ka, pos0, ring);
        if (glob) ie::i2_sconv_dynamic(q, vp, T, KD, dv, vcw, NKV, cva, crow, pos0, vca, false);
        else ie::i2_sconv_static(q, vp, T, KD, vcw, cva, crow, pos0, vca, false);
        ie::i2_rmsnorm(q, vca, T * NKV, HD, nullptr, eps, vca, HD, false);
        ie::i2_store_rows(q, vca, T, KD, Va, pos0, ring);
        ie::i2_scale_l2clip(q, ra, T * NH, DR, rs, NH, rclip, ra, false);
        if (glob) ie::i2_mul_token(q, ra, T, NH * DR, tau, ra, false);
        // fused
        ie::I2AttnPrep p;
        p.T = T; p.pos0 = pos0; p.nh = NH; p.nkv = NKV; p.hd = HD; p.d_rel = DR; p.crow = crow; p.ring = ring; p.glob = glob;
        p.qk_clip = clip; p.rel_clip = rclip; p.eps = eps;
        p.qb = d.up(hq); p.rb = d.up(hr); p.q_scale = qs; p.rel_scale = rs; p.tau = tau;
        p.kp = kp; p.vp = vp; p.dyn_k = dk; p.dyn_v = dv; p.k_conv = kcw; p.v_conv = vcw;
        p.ck = d.up(hck); p.cv = d.up(hcv); p.Kc = d.zeros(cache_rows * KD); p.Vc = d.zeros(cache_rows * KD);
        ie::i2_attn_prep(q, p);
        const std::string tag = glob ? " (dynamic conv, tau)" : " (static conv, ring)";
        check(("attn prep: q" + tag).c_str(), d.down(p.qb, T * NH * HD), d.down(qa, T * NH * HD), tol, tol);
        check(("attn prep: rel" + tag).c_str(), d.down(p.rb, T * NH * DR), d.down(ra, T * NH * DR), tol, tol);
        check(("attn prep: K cache" + tag).c_str(), d.down(p.Kc, cache_rows * KD), d.down(Ka, cache_rows * KD), tol, tol);
        check(("attn prep: V cache" + tag).c_str(), d.down(p.Vc, cache_rows * KD), d.down(Va, cache_rows * KD), tol, tol);
        check(("attn prep: k / v histories" + tag).c_str(), d.down(p.ck, crow * KD), d.down(cka, crow * KD), 0, 0);
        check(("attn prep: v history" + tag).c_str(), d.down(p.cv, crow * KD), d.down(cva, crow * KD), 0, 0);
    }
    // conv + add, MoE tail, folded ablation (k 1 and 2, keep_norm off / on), T = 3 at pos0 = 2
    {
        const uint32_t T = 3, pos0 = 2, H = 2304, K = 8, rows = 9;
        auto hx = rv(T * H), ha = rv(T * H), hw = rv(H * 4, 0.4f), hh = rv(rows * H), hey = rv(T * K * H), hwt = rv(T * K, 0.5f);
        std::vector<float> dirs = rv(2 * H);
        auto norm = [&](float* v) { double n = 0; for (uint32_t i = 0; i < H; ++i) n += double(v[i]) * v[i]; for (uint32_t i = 0; i < H; ++i) v[i] = float(v[i] / std::sqrt(n)); };
        norm(dirs.data());
        { double dd = 0; for (uint32_t i = 0; i < H; ++i) dd += double(dirs[i]) * dirs[H + i];
          for (uint32_t i = 0; i < H; ++i) dirs[H + i] -= float(dd) * dirs[i]; norm(dirs.data() + H); }
        float *x = d.up(hx), *a = d.up(ha), *w = d.up(hw), *ey = d.up(hey), *wt = d.up(hwt), *dd = d.up(dirs);
        // attention write
        float *h1 = d.up(hh), *y1 = d.zeros(T * H), *r1 = d.zeros(T * H), *h2 = d.up(hh), *y2 = d.zeros(T * H), *r2 = d.zeros(T * H);
        ie::i2_sconv_static(q, x, T, H, w, h1, rows, pos0, y1, false);
        ie::i2_add(q, a, y1, uint64_t(T) * H, r1, false);
        ie::i2_conv_add(q, x, T, H, w, h2, rows, pos0, y2, a, r2);
        check("conv + add: write", d.down(y2, T * H), d.down(y1, T * H), 1e-6f, 1e-6f);
        check("conv + add: residual", d.down(r2, T * H), d.down(r1, T * H), 1e-6f, 1e-6f);
        check("conv + add: history", d.down(h2, rows * H), d.down(h1, rows * H), 0, 0);
        // MoE tail
        float *m3 = d.zeros(T * H), *h3 = d.up(hh), *y3 = d.zeros(T * H), *o3 = d.zeros(T * H);
        float *h4 = d.up(hh), *y4 = d.zeros(T * H), *o4 = d.zeros(T * H);
        ie::i2_expert_combine(q, ey, wt, T, K, H, m3, false);
        ie::i2_sconv_static(q, m3, T, H, w, h3, rows, pos0, y3, false);
        ie::i2_add(q, a, y3, uint64_t(T) * H, o3, false);
        ie::i2_moe_out(q, ey, wt, T, K, H, w, h4, rows, pos0, y4, a, o4);
        check("MoE tail: write", d.down(y4, T * H), d.down(y3, T * H), 1e-6f, 1e-6f);
        check("MoE tail: output", d.down(o4, T * H), d.down(o3, T * H), 1e-6f, 1e-6f);
        check("MoE tail: history", d.down(h4, rows * H), d.down(h3, rows * H), 0, 0);
        float* part = d.zeros(T * 9 * 5);
        int32_t* cnt = d.up(std::vector<int32_t>(T, 0));
        for (int moe = 0; moe < 2; ++moe)
            for (uint32_t kd : {1u, 2u})
                for (int kn = 0; kn < 2; ++kn)
                    for (int with_a = 0; with_a < 2; ++with_a) {
                        float *hr = d.up(hh), *yr = d.zeros(T * H), *rr = d.zeros(T * H), *mm = d.zeros(T * H);
                        if (moe) { ie::i2_expert_combine(q, ey, wt, T, K, H, mm, false); ie::i2_sconv_static(q, mm, T, H, w, hr, rows, pos0, yr, false); }
                        else ie::i2_sconv_static(q, x, T, H, w, hr, rows, pos0, yr, false);
                        ie::i2_ablate_add(q, with_a ? a : nullptr, yr, T, H, dd, kd, 0.9f, rr, kn);
                        ie::I2WriteAbl p;
                        p.moe = moe; p.keep_norm = kn; p.T = T; p.H = H; p.k_top = K; p.hist_rows = rows; p.pos0 = pos0; p.kd = kd;
                        p.x = moe ? ey : x; p.wts = wt; p.w = w; p.a = with_a ? a : nullptr; p.dirs = dd; p.strength = 0.9f;
                        p.hist = d.up(hh); p.y = d.zeros(T * H); p.res = d.zeros(T * H); p.part = part; p.cnt = reinterpret_cast<uint32_t*>(cnt);
                        ie::i2_write_ablate(q, p);
                        const std::string nm = std::string("folded ablation: ") + (moe ? "MoE" : "attn") + ", k " + std::to_string(kd) +
                                               (kn ? ", keep_norm" : "") + (with_a ? ", + residual" : "");
                        check((nm + ", write").c_str(), d.down(p.y, T * H), d.down(yr, T * H), 1e-6f, 1e-6f);
                        check((nm + ", output").c_str(), d.down(p.res, T * H), d.down(rr, T * H), 2e-6f, 1e-5f);
                        check((nm + ", history").c_str(), d.down(p.hist, rows * H), d.down(hr, rows * H), 0, 0);
                    }
        check("folded ablation: tickets back at zero", [&] { auto c2 = d.down(cnt, T); return std::vector<float>(c2.begin(), c2.end()); }(),
              std::vector<float>(T, 0.f), 0, 0);
    }
    // gate combine (3 at once), gated merge
    {
        const uint64_t n = 3000;
        std::vector<ie::I2GateSeg> sg(3);
        std::vector<float*> ref(3);
        for (int i = 0; i < 3; ++i) {
            float *h = d.up(rv(n)), *m = d.up(rv(n)), *f = d.up(rv(n)), *g = d.up(rv(n));
            ref[i] = d.zeros(n);
            ie::i2_gate_combine(q, h, m, f, g, n, ref[i]);
            sg[i] = {h, m, f, g, d.zeros(n)};
        }
        ie::i2_gate_combine_multi(q, 3, sg.data(), n);
        for (int i = 0; i < 3; ++i) check(("gate combine x3, gate " + std::to_string(i)).c_str(), d.down(sg[i].out, n), d.down(ref[i], n), 1e-6f, 1e-6f);
    }
    {
        const uint32_t nh2 = 6, nkv2 = 2, hd2 = 128, dr2 = 16, ext2 = 200, P2 = 300, KD2 = nkv2 * hd2;
        float *Q2 = d.up(rv(nh2 * hd2, 0.3f)), *K2 = d.up(rv(size_t(P2) * KD2)), *V2 = d.up(rv(size_t(P2) * KD2));
        float *R2 = d.up(rv(nh2 * dr2)), *Pj = d.up(rv(dr2 * ext2)), *S2 = d.up(rv(nh2)), *gt = d.up(rv(nh2 * hd2));
        float *o1 = d.zeros(nh2 * hd2), *o2 = d.zeros(nh2 * hd2), *part = d.zeros(1u << 18);
        ie::i2_attention_split(q, Q2, K2, V2, 0, R2, Pj, dr2, ext2, S2, o1, 1, P2 - 1, nh2, nkv2, hd2, 0, part, 1u << 18, 128);
        ie::i2_mul_sigmoid(q, o1, gt, nh2 * hd2, o1, false);
        ie::i2_attention_split(q, Q2, K2, V2, 0, R2, Pj, dr2, ext2, S2, o2, 1, P2 - 1, nh2, nkv2, hd2, 0, part, 1u << 18, 128, gt);
        check("attention merge with the output gate", d.down(o2, nh2 * hd2), d.down(o1, nh2 * hd2), 1e-6f, 1e-6f);
    }
    // w13 straight to the activation (few-token and grouped), NVFP4 and MXFP8
    for (int fmt = 0; fmt < 2; ++fmt) {
        const uint32_t I = 40, R2 = 2 * I, C2 = 1152, E2 = 3, T2 = 21, K2 = 2, TK = T2 * K2;
        uint32_t s8 = 77u + fmt;
        auto rb = [&] { s8 = s8 * 1664525u + 1013904223u; return uint8_t(s8 >> 24); };
        std::vector<uint8_t> wv(size_t(E2) * R2 * (fmt == 0 ? C2 / 2 : C2)), sv(size_t(E2) * R2 * (fmt == 0 ? C2 / 16 : C2 / 32));
        for (auto& b : wv) { b = rb(); if (fmt == 1 && (b & 0x7f) == 0x7f) b ^= 1; }
        for (auto& b : sv) b = fmt == 0 ? uint8_t(0x28 + (rb() & 7)) : uint8_t(112 + (rb() & 3));
        ie::I2ExpertBank bk;
        bk.fmt = fmt; bk.rows = R2; bk.cols = C2; bk.w = d.up(wv); bk.s = d.up(sv); bk.fac = d.up(std::vector<float>{0.01f, 0.02f, 0.015f});
        std::vector<int32_t> ix(TK);
        for (auto& v : ix) v = int32_t(rb() % E2);
        int32_t* dix = d.up(ix);
        float* x = d.up(rv(size_t(T2) * C2, 0.5f));
        for (uint32_t T : {1u, T2}) {
            float *gu = d.zeros(size_t(T) * K2 * R2), *ar = d.zeros(size_t(T) * K2 * I), *af = d.zeros(size_t(T) * K2 * I);
            ie::i2_expert_matvec(q, bk, x, false, dix, T, K2, gu, false, false);
            ie::i2_expert_act(q, gu, T * K2, I, 4.f, 25.f, ar, false);
            ie::i2_expert_w13_act(q, bk, x, dix, T, K2, af, 4.f, 25.f);
            check((std::string(fmt ? "MXFP8" : "NVFP4") + " w13 -> act, T " + std::to_string(T)).c_str(), d.down(af, T * K2 * I),
                  d.down(ar, T * K2 * I), 1e-6f, 1e-6f);
        }
        int32_t* order = d.up(std::vector<int32_t>(TK, -1));
        int32_t* tiles = d.up(std::vector<int32_t>(3 * (TK / ie::kI2MoeTile + E2 + 1), -1));
        int32_t* nt = d.up(std::vector<int32_t>(1, 0));
        ie::i2_moe_group(q, dix, T2, K2, E2, ie::kI2MoeTile, order, tiles, nt);
        const uint32_t maxt = (TK + ie::kI2MoeTile - 1) / ie::kI2MoeTile + E2;
        float *gu = d.zeros(size_t(TK) * R2), *ar = d.zeros(size_t(TK) * I), *af = d.zeros(size_t(TK) * I);
        ie::i2_expert_grouped(q, bk, x, false, K2, order, tiles, nt, maxt, gu, false);
        ie::i2_expert_act(q, gu, TK, I, 4.f, 25.f, ar, false);
        ie::i2_expert_grouped_w13_act(q, bk, x, K2, order, tiles, nt, maxt, af, 4.f, 25.f);
        check((std::string(fmt ? "MXFP8" : "NVFP4") + " grouped w13 -> act").c_str(), d.down(af, TK * I), d.down(ar, TK * I), 1e-6f, 1e-6f);
    }
}


// Batched decoding: row t of a T-row call (rowwise / per-row state) == the same token alone (T = 1), bit for bit.
void batch_tests(sycl::queue& q) {
    Dev d{q, {}};
    uint32_t st = 777u;
    auto rf = [&] { st = st * 1664525u + 1013904223u; return float(int32_t(st >> 8) - (1 << 23)) / float(1 << 23); };
    auto rv = [&](size_t n, float sc = 1.f) { std::vector<float> v(n); for (auto& x : v) x = rf() * sc; return v; };
    auto rbf = [&](size_t n) { std::vector<uint16_t> v(n); for (auto& x : v) { const float f = rf() * 0.05f; uint32_t u; std::memcpy(&u, &f, 4); x = uint16_t(u >> 16); } return v; };
    // linears: the router / gate-down shape (work-group per row), the projections, wo (long K), forget / gain (short K)
    const uint32_t shapes[5][2] = {{128, 2304}, {1024, 2304}, {2304, 3072}, {2304, 64}, {9000, 2304}};
    for (const auto& sh : shapes) {
        const uint32_t N = sh[0], K = sh[1];
        const uint16_t* W = d.up(rbf(size_t(N) * K));
        for (uint32_t T : {2u, 3u, 7u, 13u}) {
            const auto hx = rv(size_t(T) * K);
            float *x = d.up(hx), *yb = d.zeros(size_t(T) * N), *y1 = d.zeros(size_t(T) * N);
            ie::i2_linear_bf16(q, x, T, K, W, N, yb, false, true, true);
            for (uint32_t t = 0; t < T; ++t) ie::i2_linear_bf16(q, x + size_t(t) * K, 1, K, W, N, y1 + size_t(t) * N, false, true);
            check(("rowwise linear N " + std::to_string(N) + " K " + std::to_string(K) + " T " + std::to_string(T)).c_str(),
                  d.down(yb, size_t(T) * N), d.down(y1, size_t(T) * N), 0, 0);
        }
    }
    // attn_res (2 sources x 4 queries, 5 sources x 1 query), T = 20 > 16
    {
        const uint32_t H = 2304, T = 20;
        for (uint32_t ns : {2u, 5u}) {
            const uint32_t nq = ns == 2 ? 4 : 1;
            ie::I2Sources src{};
            std::vector<float*> sp;
            for (uint32_t i = 0; i < ns; ++i) { sp.push_back(d.up(rv(size_t(T) * H))); src.p[i] = sp.back(); }
            float *Q = d.up(rv(size_t(nq) * H)), *ob = d.zeros(size_t(nq) * T * H), *o1 = d.zeros(size_t(nq) * H);
            float *scr = d.zeros(size_t(T) * 16 * 40);
            ie::i2_attn_res(q, src, ns, T, H, Q, nq, 1e-6f, ob, true, scr, true);
            auto hb = d.down(ob, size_t(nq) * T * H);
            std::vector<float> want(size_t(nq) * T * H), got(hb);
            for (uint32_t t = 0; t < T; ++t) {
                ie::I2Sources s1{};
                for (uint32_t i = 0; i < ns; ++i) s1.p[i] = sp[i] + size_t(t) * H;
                ie::i2_attn_res(q, s1, ns, 1, H, Q, nq, 1e-6f, o1, true, scr);
                auto h1 = d.down(o1, size_t(nq) * H);
                for (uint32_t qi = 0; qi < nq; ++qi)
                    std::memcpy(&want[(size_t(qi) * T + t) * H], &h1[size_t(qi) * H], H * 4);
            }
            check(("rowwise attn_res, " + std::to_string(ns) + " sources").c_str(), got, want, 0, 0);
        }
    }
    // experts: grouped (> 4 tokens) == per-slot (decode), w13 -> act and w2, NVFP4 and MXFP8
    for (int fmt = 0; fmt < 2; ++fmt) {
        const uint32_t I = 64, C = 2304, E = 5, T = 11, K = 3, TK = T * K;
        uint32_t s8 = 91u + fmt;
        auto rb = [&] { s8 = s8 * 1664525u + 1013904223u; return uint8_t(s8 >> 24); };
        auto bank = [&](uint32_t rows, uint32_t cols) {
            std::vector<uint8_t> wv(size_t(E) * rows * (fmt == 0 ? cols / 2 : cols)), sv(size_t(E) * rows * (fmt == 0 ? cols / 16 : cols / 32));
            for (auto& b : wv) { b = rb(); if (fmt == 1 && (b & 0x7f) == 0x7f) b ^= 1; }
            for (auto& b : sv) b = fmt == 0 ? uint8_t(0x28 + (rb() & 7)) : uint8_t(112 + (rb() & 3));
            ie::I2ExpertBank bk;
            bk.fmt = fmt; bk.rows = rows; bk.cols = cols; bk.w = d.up(wv); bk.s = d.up(sv);
            bk.fac = d.up(std::vector<float>{0.01f, 0.02f, 0.015f, 0.012f, 0.03f});
            return bk;
        };
        const ie::I2ExpertBank w13 = bank(2 * I, C), w2 = bank(C, I);
        std::vector<int32_t> ix(TK);
        for (auto& v : ix) v = int32_t(rb() % E);
        int32_t* dix = d.up(ix);
        float* x = d.up(rv(size_t(T) * C, 0.5f));
        float *a1 = d.zeros(size_t(TK) * I), *ag = d.zeros(size_t(TK) * I), *y1 = d.zeros(size_t(TK) * C), *yg = d.zeros(size_t(TK) * C);
        ie::i2_expert_w13_act(q, w13, x, dix, T, K, a1, 4.f, 25.f);
        ie::i2_expert_matvec(q, w2, a1, true, dix, T, K, y1, false, false, true);
        int32_t* order = d.up(std::vector<int32_t>(TK, -1));
        int32_t* tiles = d.up(std::vector<int32_t>(3 * (TK / ie::kI2MoeTile + E + 1), -1));
        int32_t* nt = d.up(std::vector<int32_t>(1, 0));
        ie::i2_moe_group(q, dix, T, K, E, ie::kI2MoeTile, order, tiles, nt);
        const uint32_t maxt = (TK + ie::kI2MoeTile - 1) / ie::kI2MoeTile + E;
        ie::i2_expert_grouped_w13_act(q, w13, x, K, order, tiles, nt, maxt, ag, 4.f, 25.f);
        ie::i2_expert_grouped(q, w2, a1, true, K, order, tiles, nt, maxt, yg, false);
        const std::string f = fmt ? "MXFP8" : "NVFP4";
        check((f + " grouped w13 -> act == per-slot").c_str(), d.down(ag, size_t(TK) * I), d.down(a1, size_t(TK) * I), 0, 0);
        check((f + " grouped w2 == per-slot").c_str(), d.down(yg, size_t(TK) * C), d.down(y1, size_t(TK) * C), 0, 0);
    }
}

int main() {
    host_tests();
    std::vector<sycl::device> gpus;
    for (const auto& pl : sycl::platform::get_platforms())
        if (pl.get_backend() == sycl::backend::ext_oneapi_level_zero)
            for (const auto& dv : pl.get_devices(sycl::info::device_type::gpu))
                if (dv.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) gpus.push_back(dv);
    if (gpus.empty()) { std::printf("no Arc GPU: device tests skipped\n"); return g_fail ? 1 : 0; }
    const char* ce = std::getenv("IE_TEST_CARD");                       // which Arc card (default 0)
    const size_t card = ce ? size_t(std::atoi(ce)) : 0;
    if (card >= gpus.size()) { std::printf("IE_TEST_CARD %zu: only %zu Arc GPUs\n", card, gpus.size()); return 1; }
    sycl::queue q(sycl::context(gpus[card]), gpus[card], sycl::property::queue::in_order{});
    device_tests(q);
    fused_tests(q);
    batch_tests(q);
    std::printf("%s (%d failed)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
