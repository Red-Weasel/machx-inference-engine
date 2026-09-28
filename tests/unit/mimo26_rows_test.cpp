// tests/unit/mimo26_rows_test.cpp — P4 B3 (docs/mimo26/P4_B3_ROWS.md): batch invariance of the ops a row-batched MiMo-V2.6
// decode step shares between lanes. For each op, the M-row call (M = 2, 3, 5, 8) is compared ROW BY ROW against the 1-row call
// on that row's input, at MiMo-V2.6-Flash's shapes (dim 4096, 64 q heads x 128 = 8192 o-proj K, vocab 152,576, 256 experts,
// expert inter 2,048): the fp32 RMSNorm, the FP8 GEMV (the three work-group widths), the router logits, the expert path as
// the tier runs a 2..8-row step (per-row M = 1 int-dot jobs in ONE grouped launch, fp32 rows, the SwiGLU, the Q8_1 requant and
// the fp32 scatter), and the two oneDNN GEMMs -- the o-proj and the LM head -- whose kernel choice follows M. The row-independent
// kernels must be bit-identical (a FAIL otherwise); the oneDNN GEMMs are REPORTED (identical, or the max abs diff and how many
// rows' argmax changed) and never fail: the forward's IE_MIMO26_ROWS_INVARIANT=1 runs them per lane. Needs an Arc GPU (SKIP 77).
#include "ie/deepseek4_experts.hpp"
#include "ie/deepseek4_ops.hpp"
#include "ie/mimo26_ops.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

using namespace ie;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what, const std::string& detail = "") {
    std::printf("[%s] %s%s%s\n", ok ? " ok " : "FAIL", what.c_str(), detail.empty() ? "" : "  ", detail.c_str());
    if (!ok) ++g_fail;
}
void info(const std::string& what, const std::string& detail) { std::printf("[info] %s  %s\n", what.c_str(), detail.c_str()); }

constexpr uint32_t kRows = 8;                           // Mimo26Forward::kDecodeRows
const uint32_t kMs[] = {2, 3, 5, 8};

// Row r of `batched` (an M-row result, row stride `n`) against `solo` (the 1-row result on row r): bytes, or the difference.
struct RowDiff { uint32_t rows_differ = 0; double max_abs = 0, max_rel = 0; uint32_t argmax_changes = 0; };
template <class T>
void row_diff(const std::vector<T>& solo, const std::vector<T>& batched, uint32_t r, uint32_t n, RowDiff& d, bool argmax) {
    bool differs = false; double mx = 0, mr = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const T a = solo[i], b = batched[size_t(r) * n + i];
        if (std::memcmp(&a, &b, sizeof(T)) != 0) {
            differs = true;
            const double fa = double(float(a)), fb = double(float(b)), ad = std::fabs(fa - fb);
            mx = std::max(mx, ad); mr = std::max(mr, ad / std::max(1e-30, std::fabs(fa)));
        }
    }
    if (differs) { ++d.rows_differ; d.max_abs = std::max(d.max_abs, mx); d.max_rel = std::max(d.max_rel, mr); }
    if (argmax && differs) {
        auto am = [&](const T* p) { uint32_t best = 0; for (uint32_t i = 1; i < n; ++i) if (float(p[i]) > float(p[best])) best = i; return best; };
        if (am(solo.data()) != am(batched.data() + size_t(r) * n)) ++d.argmax_changes;
    }
}
std::string fmt(const RowDiff& d, uint32_t rows, bool argmax) {
    if (!d.rows_differ) return "bit-identical";
    char b[160];
    std::snprintf(b, sizeof b, "%u of %u rows differ: max abs %.3g, max rel %.3g%s%s", d.rows_differ, rows, d.max_abs, d.max_rel,
                  argmax ? ", argmax changes " : "", argmax ? std::to_string(d.argmax_changes).c_str() : "");
    return b;
}

// a device-side pseudo-random fill (the head weight is 1.25 GB: too slow from the host): values in about (-a, a)
template <class T>
void fill_dev(sycl::queue& q, T* p, size_t n, uint32_t seed, float a) {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        uint32_t h = uint32_t(i[0]) * 2654435761u ^ seed; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
        p[i] = T((float(h & 0xFFFFFFu) / float(0xFFFFFF) * 2.f - 1.f) * a);
    }).wait();
}

}  // namespace

int main() {
    sycl::device dev; bool found = false;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos && !found) { dev = d; found = true; }
    }
    if (!found) { std::printf("SKIP: no Arc GPU\n"); return 77; }
    sycl::queue q(sycl::context(dev), dev, sycl::property_list{sycl::property::queue::in_order{}});
    std::mt19937 rng(23);
    std::normal_distribution<float> nd(0.f, 1.f);
    const uint32_t H = 4096, E = 256, EF = 2048, TK = 8, VOCAB = 152576, KO = 64 * 128;
    auto up = [&](const auto& v) { using T = typename std::decay_t<decltype(v)>::value_type; T* p = sycl::malloc_device<T>(v.size(), q); q.memcpy(p, v.data(), v.size() * sizeof(T)).wait(); return p; };
    auto down = [&](auto* p, size_t n) { std::vector<std::remove_pointer_t<decltype(p)>> v(n); q.memcpy(v.data(), p, n * sizeof(v[0])).wait(); return v; };

    // the 8 activation rows every op takes (fp32 and their fp16 rounding)
    std::vector<float> x32(size_t(kRows) * H); for (auto& v : x32) v = nd(rng);
    std::vector<sycl::half> x16(x32.size()); for (size_t i = 0; i < x32.size(); ++i) x16[i] = sycl::half(x32[i]);
    float* dx32 = up(x32); sycl::half* dx16 = up(x16);

    // ---- RMSNorm (fp16 and fp32 outputs) ----
    {
        std::vector<float> w(H); for (auto& v : w) v = 1.f + 0.1f * nd(rng);
        float* dw = up(w);
        sycl::half* y16 = sycl::malloc_device<sycl::half>(size_t(kRows) * H, q); float* y32 = sycl::malloc_device<float>(size_t(kRows) * H, q);
        std::vector<std::vector<sycl::half>> s16(kRows); std::vector<std::vector<float>> s32(kRows);
        for (uint32_t r = 0; r < kRows; ++r) {
            mimo26_rms_norm(q, dx32 + size_t(r) * H, dw, y16, y32, 1, H, 1e-6f).wait();
            s16[r] = down(y16, H); s32[r] = down(y32, H);
        }
        for (uint32_t M : kMs) {
            mimo26_rms_norm(q, dx32, dw, y16, y32, M, H, 1e-6f).wait();
            const auto b16 = down(y16, size_t(M) * H); const auto b32 = down(y32, size_t(M) * H);
            RowDiff d16, d32;
            for (uint32_t r = 0; r < M; ++r) { row_diff(s16[r], b16, r, H, d16, false); row_diff(s32[r], b32, r, H, d32, false); }
            check(!d16.rows_differ && !d32.rows_differ, "rms_norm M = " + std::to_string(M) + " rows == 1-row calls (fp16 and fp32 out)", fmt(d16, M, false) + " / " + fmt(d32, M, false));
        }
        sycl::free(dw, q); sycl::free(y16, q); sycl::free(y32, q);
    }

    // ---- the FP8 GEMV: qkv (N % 32 == 0 -> 16 columns per work-group), the dense MLP's gate and down, N <= 256 (4), N % 32 != 0 (8) ----
    {
        struct Shape { uint32_t N, K; const char* name; };
        const Shape shapes[] = {{13568, 4096, "qkv (16 cols/wg)"}, {16384, 4096, "mlp gate"}, {4096, 16384, "mlp down"}, {200, 4096, "N <= 256 (4 cols/wg)"}, {300, 4096, "N % 32 != 0 (8 cols/wg)"}};
        for (const Shape& s : shapes) {
            const uint32_t N = s.N, K = s.K, SK = K / 128;
            std::vector<uint8_t> w8(size_t(N) * K); for (auto& b : w8) { uint8_t v; do { v = uint8_t(rng() & 0xFF); } while ((v & 0x7F) == 0x7F); b = v; }
            std::vector<float> sr(size_t(N) * SK); for (auto& v : sr) v = std::ldexp(1.f, -int(rng() % 8) - 2);
            std::vector<sycl::half> xa(size_t(kRows) * K); for (auto& v : xa) v = sycl::half(nd(rng));
            uint8_t* dw8 = up(w8); float* dsr = up(sr); sycl::half* dxa = up(xa);
            float* dy = sycl::malloc_device<float>(size_t(kRows) * N, q);
            std::vector<std::vector<float>> solo(kRows);
            for (uint32_t r = 0; r < kRows; ++r) { mimo26_gemv_fp8(q, dxa + size_t(r) * K, K, dw8, dsr, dy, 1, K, N).wait(); solo[r] = down(dy, N); }
            for (uint32_t M : kMs) {
                mimo26_gemv_fp8(q, dxa, K, dw8, dsr, dy, M, K, N).wait();
                const auto b = down(dy, size_t(M) * N);
                RowDiff d; for (uint32_t r = 0; r < M; ++r) row_diff(solo[r], b, r, N, d, false);
                check(!d.rows_differ, "gemv_fp8 " + std::string(s.name) + " N " + std::to_string(N) + " K " + std::to_string(K) + ": M = " + std::to_string(M) + " == 1-row calls", fmt(d, M, false));
            }
            sycl::free(dw8, q); sycl::free(dsr, q); sycl::free(dxa, q); sycl::free(dy, q);
        }
    }

    // ---- the router logits (fp32) ----
    {
        std::vector<float> w(size_t(E) * H); for (auto& v : w) v = 0.02f * nd(rng);
        float* dw = up(w); float* dl = sycl::malloc_device<float>(size_t(kRows) * E, q);
        std::vector<std::vector<float>> solo(kRows);
        for (uint32_t r = 0; r < kRows; ++r) { mimo26_router_logits(q, dx32 + size_t(r) * H, dw, dl, 1, H, E).wait(); solo[r] = down(dl, E); }
        for (uint32_t M : kMs) {
            mimo26_router_logits(q, dx32, dw, dl, M, H, E).wait();
            const auto b = down(dl, size_t(M) * E);
            RowDiff d; for (uint32_t r = 0; r < M; ++r) row_diff(solo[r], b, r, E, d, true);
            check(!d.rows_differ, "router_logits M = " + std::to_string(M) + " == 1-row calls", fmt(d, M, true));
        }
        sycl::free(dw, q); sycl::free(dl, q);
    }

    // ---- the expert path as the tier runs a 2..8-row step (Ds41ExpertTier::moe, row_jobs): the gather + cast, the Q8_1 quant of
    //      the packed rows as one stream, per-row M = 1 gate/up jobs in one grouped launch (fp32 rows), the SwiGLU, the requant,
    //      per-row down jobs, the fp32 scatter -- against one row through the same kernels alone (the T = 1 step's shape)
    {
        // one random MXFP4 expert: gate/up [EF, H], down [H, EF] (nibbles random, E8M0 scales 2^-5..2^0 as the DS4 gate test)
        auto bank = [&](uint32_t K, uint32_t N) {
            DS4ExpertBank b{}; b.dtype = DType::kMXFP4; b.K = K; b.N = N; b.E = 1;
            const uint64_t qs = uint64_t(N) * (K / 2), ep = uint64_t(N) * (K / 32);
            b.mx_qs_stride = qs; b.mx_e_stride = ep;
            std::vector<uint8_t> hq(qs), he(ep); for (auto& v : hq) v = uint8_t(rng()); for (auto& v : he) v = uint8_t(123 + rng() % 6);
            b.mx_qs = up(hq); b.mx_e = up(he);
            return b;
        };
        const DS4ExpertBank gate = bank(H, EF), upb = bank(H, EF), downb = bank(EF, H);
        DS4GemmGroupWs gws{};
        check(ds4_gemm_group_ws_alloc(q, 64, gws).empty(), "expert group workspace");
        // the packed rows: 8 rows of x, each routed to THIS expert (as 8 rows of a group hitting one expert -> 8 per-row jobs)
        std::vector<int32_t> rows(kRows); for (uint32_t r = 0; r < kRows; ++r) rows[r] = int32_t(r);
        int32_t* drows = up(rows);
        sycl::half* xp = sycl::malloc_device<sycl::half>(size_t(kRows) * H, q);
        block_q8_1x* xq = sycl::malloc_device<block_q8_1x>(size_t(kRows) * (H / 32), q);
        float* g32 = sycl::malloc_device<float>(size_t(kRows) * EF, q); float* u32 = sycl::malloc_device<float>(size_t(kRows) * EF, q);
        sycl::half* g16 = sycl::malloc_device<sycl::half>(size_t(kRows) * EF, q); sycl::half* u16 = sycl::malloc_device<sycl::half>(size_t(kRows) * EF, q);
        sycl::half* h16 = sycl::malloc_device<sycl::half>(size_t(kRows) * EF, q);
        block_q8_1x* hq = sycl::malloc_device<block_q8_1x>(size_t(kRows) * (EF / 32), q);
        float* y32 = sycl::malloc_device<float>(size_t(kRows) * H, q);
        const float inf = std::numeric_limits<float>::infinity();
        // the tier's int-dot route lands gate / up fp16 (g_h / u_h) at T <= 8 -- and with fp32_out the down rows fp32 (y32)
        auto run = [&](uint32_t M, uint32_t r_first) -> std::string {   // rows [r_first, r_first + M) as one step; "" or the error
            ds4_expert_gather_cast(q, dx32, drows + r_first, xp, M, H);
            quantize_q8_1(q, xp, xq, M * H);
            std::vector<DS4GemmJob> jobs;
            for (uint32_t r = 0; r < M; ++r) {
                jobs.push_back({gate, 0, 1, xq + size_t(r) * (H / 32), g16 + size_t(r) * EF, r, nullptr});
                jobs.push_back({upb,  0, 1, xq + size_t(r) * (H / 32), u16 + size_t(r) * EF, r, nullptr});
            }
            if (auto e = ds4_expert_gemm_q8_grouped(q, jobs.data(), uint32_t(jobs.size()), gws); !e.empty()) return e;
            ds4_swiglu_clamped_h(q, g16, u16, h16, size_t(M) * EF, inf);
            quantize_q8_1(q, h16, hq, M * EF);
            jobs.clear();
            for (uint32_t r = 0; r < M; ++r) jobs.push_back({downb, 0, 1, hq + size_t(r) * (EF / 32), nullptr, r, y32 + size_t(r) * H});
            if (auto e = ds4_expert_gemm_q8_grouped(q, jobs.data(), uint32_t(jobs.size()), gws); !e.empty()) return e;
            q.wait();
            return {};
        };
        std::vector<std::vector<sycl::half>> solo_g(kRows), solo_h(kRows); std::vector<std::vector<float>> solo_y(kRows);
        bool ok = true;
        for (uint32_t r = 0; r < kRows; ++r) {
            if (auto e = run(1, r); !e.empty()) { check(false, "expert path 1-row step", e); ok = false; break; }
            solo_g[r] = down(g16, EF); solo_h[r] = down(h16, EF); solo_y[r] = down(y32, H);
        }
        for (uint32_t M : kMs) {
            if (!ok) break;
            if (auto e = run(M, 0); !e.empty()) { check(false, "expert path M-row step", e); break; }
            const auto bg = down(g16, size_t(M) * EF); const auto bh = down(h16, size_t(M) * EF); const auto by = down(y32, size_t(M) * H);
            RowDiff dg, dh, dy;
            for (uint32_t r = 0; r < M; ++r) { row_diff(solo_g[r], bg, r, EF, dg, false); row_diff(solo_h[r], bh, r, EF, dh, false); row_diff(solo_y[r], by, r, H, dy, false); }
            check(!dg.rows_differ && !dh.rows_differ && !dy.rows_differ,
                  "expert path M = " + std::to_string(M) + " rows (per-row jobs in one launch: gate fp16, SwiGLU fp16, down fp32) == 1-row steps",
                  fmt(dg, M, false) + " / " + fmt(dh, M, false) + " / " + fmt(dy, M, false));
        }
        // (informational) ONE job of M = 8 rows -- the M-tiled kernel the tier does NOT use at T <= 8 (row_jobs splits it)
        if (ok) {
            ds4_expert_gather_cast(q, dx32, drows, xp, kRows, H); quantize_q8_1(q, xp, xq, kRows * H);
            DS4GemmJob j{gate, 0, kRows, xq, g16, 0, nullptr};
            if (auto e = ds4_expert_gemm_q8_grouped(q, &j, 1, gws); e.empty()) {
                q.wait(); const auto bg = down(g16, size_t(kRows) * EF);
                RowDiff dg; for (uint32_t r = 0; r < kRows; ++r) row_diff(solo_g[r], bg, r, EF, dg, false);
                info("expert gate as ONE M = 8 job (the M-tiled kernel, unused by a decode step)", fmt(dg, kRows, false));
            }
        }
        // the fp32 scatter: token t's rows are its own -- T = 1 (its 8 packed rows) against T = 8 tokens x 8 experts
        {
            std::vector<float> yp(size_t(kRows) * TK * H); for (auto& v : yp) v = nd(rng);
            std::vector<float> w(size_t(kRows) * TK); for (auto& v : w) v = std::fabs(nd(rng)) * 0.3f;
            std::vector<int32_t> tk2p(size_t(kRows) * TK);
            std::vector<uint32_t> perm(size_t(kRows) * TK); std::iota(perm.begin(), perm.end(), 0u); std::shuffle(perm.begin(), perm.end(), rng);
            for (size_t i = 0; i < tk2p.size(); ++i) tk2p[i] = int32_t(perm[i]);   // a packed row per (t, k), permuted like a counting sort would
            float* dyp = up(yp); float* dw = up(w); int32_t* dt = up(tk2p);
            float* y = sycl::malloc_device<float>(size_t(kRows) * H, q);
            std::vector<std::vector<float>> solo(kRows);
            for (uint32_t t = 0; t < kRows; ++t) {   // the 1-token scatter of token t: its (k -> packed row) map alone, the same rows
                q.memset(y, 0, size_t(H) * 4);
                ds4_expert_scatter_accum_f32(q, dyp, dt + size_t(t) * TK, dw, y, 1, TK, H).wait();
                solo[t] = down(y, H);
            }
            q.memset(y, 0, size_t(kRows) * H * 4);
            ds4_expert_scatter_accum_f32(q, dyp, dt, dw, y, kRows, TK, H).wait();
            const auto b = down(y, size_t(kRows) * H);
            RowDiff d; for (uint32_t t = 0; t < kRows; ++t) row_diff(solo[t], b, t, H, d, false);
            check(!d.rows_differ, "expert scatter (fp32) T = 8 tokens == 1-token scatters", fmt(d, kRows, false));
            sycl::free(dyp, q); sycl::free(dw, q); sycl::free(dt, q); sycl::free(y, q);
        }
        ds4_gemm_group_ws_free(q, gws);
        for (const auto& b : {gate, upb, downb}) { sycl::free(b.mx_qs, q); sycl::free(b.mx_e, q); }
        sycl::free(drows, q); sycl::free(xp, q); sycl::free(xq, q); sycl::free(g32, q); sycl::free(u32, q); sycl::free(g16, q); sycl::free(u16, q);
        sycl::free(h16, q); sycl::free(hq, q); sycl::free(y32, q);
    }

    // ---- the oneDNN GEMMs (reported, not gated): the o-proj [H, 64 x 128] and the LM head [vocab, H], M = 1 vs M rows ----
    {
        struct Shape { uint32_t N, K; const char* name; bool argmax; };
        const Shape shapes[] = {{H, KO, "o-proj", false}, {VOCAB, H, "lm_head", true}};
        for (const Shape& s : shapes) {
            sycl::half* w = sycl::malloc_device<sycl::half>(size_t(s.N) * s.K, q);
            if (!w) { info(std::string(s.name) + " weight allocation failed", "skipped"); continue; }
            fill_dev(q, w, size_t(s.N) * s.K, 77u + s.N, s.name[0] == 'l' ? 0.02f : 0.05f);
            sycl::half* a = sycl::malloc_device<sycl::half>(size_t(kRows) * s.K, q);
            fill_dev(q, a, size_t(kRows) * s.K, 91u, 1.f);
            float* y = sycl::malloc_device<float>(size_t(kRows) * s.N, q);
            std::vector<std::vector<float>> solo(kRows);
            for (uint32_t r = 0; r < kRows; ++r) { gemm_nt_f16_onednn(q, a + size_t(r) * s.K, w, y, 1, s.N, s.K).wait(); solo[r] = down(y, s.N); }
            for (uint32_t M : kMs) {
                gemm_nt_f16_onednn(q, a, w, y, M, s.N, s.K).wait();
                const auto b = down(y, size_t(M) * s.N);
                RowDiff d; for (uint32_t r = 0; r < M; ++r) row_diff(solo[r], b, r, s.N, d, s.argmax);
                info("oneDNN " + std::string(s.name) + " N " + std::to_string(s.N) + " K " + std::to_string(s.K) + ": M = " + std::to_string(M) + " vs 1-row calls", fmt(d, M, s.argmax));
            }
            // and M = 1 against itself twice (the same call is deterministic)
            { gemm_nt_f16_onednn(q, a, w, y, 1, s.N, s.K).wait(); const auto b = down(y, s.N); RowDiff d; row_diff(solo[0], b, 0, s.N, d, false);
              check(!d.rows_differ, "oneDNN " + std::string(s.name) + " M = 1 repeated is bit-identical", fmt(d, 1, false)); }
            sycl::free(w, q); sycl::free(a, q); sycl::free(y, q);
        }
    }

    sycl::free(dx32, q); sycl::free(dx16, q);
    std::printf("\nMIMO26 ROWS TEST: %s\n", g_fail ? "FAILURE(S)" : "PASS");
    return g_fail ? 1 : 0;
}
