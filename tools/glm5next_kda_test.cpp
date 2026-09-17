// tools/glm5next_kda_test.cpp — KDA kernel gate (glm5next P2, no model file).
//
// Adversarial CPU fp64 references for the three new KDA ops (kda_l2norm,
// kda_gate, kda_recurrence — src/ops/deltanet.cpp), on random tensors shaped
// like the real model (head_dim 128). Section 4 is the state-carry contract:
// one T=32 scan must be BIT-IDENTICAL to 4 chained T=8 scans, because chunked
// prefill and decode both rely on carrying the recurrent state across calls.
//
// usage: ie-glm5next-kda-test [gpu_ordinal]
#include "ie/allocator.hpp"
#include "ie/ops.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace ie;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "\033[32mOK\033[0m" : "\033[31mFAIL\033[0m", what);
    if (!ok) ++g_fail;
}

// CPU fp64 reference of the full KDA recurrence (build_delta_net, KDA branch).
void ref_kda(const std::vector<float>& qv, const std::vector<float>& kv,
             const std::vector<float>& vv, const std::vector<float>& gv,
             const std::vector<float>& bv, std::vector<double>& S,
             std::vector<double>& out, uint32_t T, uint32_t H, uint32_t D) {
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t h = 0; h < H; ++h) {
            const uint64_t row = (uint64_t(t) * H + h) * D;
            double* Sh = S.data() + uint64_t(h) * D * D;   // S[key][value], value-fastest
            // decay per key channel
            for (uint32_t k = 0; k < D; ++k) {
                const double a = std::exp(double(gv[row + k]));
                for (uint32_t v = 0; v < D; ++v) Sh[uint64_t(k) * D + v] *= a;
            }
            // kv_mem[v] = sum_k S[k][v] * k[k];  delta = (v - kv_mem) * beta
            const double beta = double(bv[uint64_t(t) * H + h]);
            for (uint32_t v = 0; v < D; ++v) {
                double m = 0.0;
                for (uint32_t k = 0; k < D; ++k) m += Sh[uint64_t(k) * D + v] * double(kv[row + k]);
                const double delta = (double(vv[row + v]) - m) * beta;
                for (uint32_t k = 0; k < D; ++k) Sh[uint64_t(k) * D + v] += double(kv[row + k]) * delta;
                double o = 0.0;
                for (uint32_t k = 0; k < D; ++k) o += Sh[uint64_t(k) * D + v] * double(qv[row + k]);
                out[row + v] = o;
            }
        }
}
}  // namespace

int main(int argc, char** argv) {
    const uint32_t ordinal = argc > 1 ? uint32_t(std::atoi(argv[1])) : 0;
    DeviceAllocator alloc;
    if (auto e = alloc.init("B70", ordinal); !e.empty()) {
        std::fprintf(stderr, "gpu init: %s\n", e.c_str()); return 1;
    }
    sycl::queue& q = alloc.queue();
    std::mt19937 rng(20260828);
    std::normal_distribution<float> nd(0.f, 1.f);

    constexpr uint32_t D = 128, H = 4, T = 32;
    std::printf("\n\033[1mglm5next KDA kernel gate\033[0m  (H=%u, D=%u, T=%u)\n\n", H, D, T);

    // ---- 1. kda_l2norm vs fp64 (floor semantics) ---------------------------
    {
        const uint32_t rows = T * H;
        std::vector<float> x(rows * D);
        for (auto& v : x) v = nd(rng);
        x[3] = 1e-9f;   // one near-zero row region exercises nothing special;
        float* dx = static_cast<float*>(alloc.malloc(x.size() * 4));
        float* dy = static_cast<float*>(alloc.malloc(x.size() * 4));
        q.memcpy(dx, x.data(), x.size() * 4).wait();
        const float scale = 1.f / std::sqrt(float(D));
        kda_l2norm(q, dx, dy, rows, D, scale, 1e-6f, {}).wait();
        std::vector<float> y(x.size());
        q.memcpy(y.data(), dy, y.size() * 4).wait();
        double mx = 0;
        for (uint32_t r = 0; r < rows; ++r) {
            double s = 0;
            for (uint32_t i = 0; i < D; ++i) s += double(x[r * D + i]) * x[r * D + i];
            const double rr = double(scale) / std::max(std::sqrt(s), 1e-6);
            for (uint32_t i = 0; i < D; ++i)
                mx = std::max(mx, std::abs(double(y[r * D + i]) - double(x[r * D + i]) * rr));
        }
        std::printf("      l2norm max abs err vs fp64: %.3e\n", mx);
        check(mx < 2e-6, "kda_l2norm matches fp64 floor-semantics reference");
        alloc.free(dx); alloc.free(dy);
    }

    // ---- 2. kda_gate vs fp64 ----------------------------------------------
    {
        const uint32_t rows = T;
        std::vector<float> pre(rows * H * D), bias(H * D), a(H);
        for (auto& v : pre)  v = nd(rng) * 2.f;
        for (auto& v : bias) v = nd(rng);
        for (auto& v : a)    v = -std::exp(nd(rng));   // ssm_a = -exp(A_log)
        float* dp = static_cast<float*>(alloc.malloc(pre.size() * 4));
        float* db = static_cast<float*>(alloc.malloc(bias.size() * 4));
        float* da = static_cast<float*>(alloc.malloc(a.size() * 4));
        float* dg = static_cast<float*>(alloc.malloc(pre.size() * 4));
        q.memcpy(dp, pre.data(), pre.size() * 4).wait();
        q.memcpy(db, bias.data(), bias.size() * 4).wait();
        q.memcpy(da, a.data(), a.size() * 4).wait();
        kda_gate(q, dp, db, da, dg, rows, H, D, -5.0f, {}).wait();
        std::vector<float> g(pre.size());
        q.memcpy(g.data(), dg, g.size() * 4).wait();
        double mx = 0;
        bool in_range = true;
        for (uint64_t i = 0; i < pre.size(); ++i) {
            const uint32_t c = uint32_t(i % (H * D)), hh = c / D;
            const double x = -(double(pre[i]) + bias[c]) * a[hh];
            const double want = -5.0 / (1.0 + std::exp(-x));
            mx = std::max(mx, std::abs(double(g[i]) - want));
            in_range = in_range && g[i] <= 0.f && g[i] >= -5.f;
        }
        std::printf("      gate max abs err vs fp64: %.3e\n", mx);
        check(mx < 1e-5, "kda_gate matches fp64 reference");
        check(in_range, "log decay in [-5, 0] (lower-bound scale, not clamp)");
        alloc.free(dp); alloc.free(db); alloc.free(da); alloc.free(dg);
    }

    // ---- 3. kda_recurrence vs fp64 -----------------------------------------
    const uint64_t n = uint64_t(T) * H * D;
    std::vector<float> qv(n), kv(n), vv(n), gv(n), bv(uint64_t(T) * H);
    // q/k L2-normed rows (as the real graph feeds them), q pre-scaled.
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t h = 0; h < H; ++h) {
            const uint64_t row = (uint64_t(t) * H + h) * D;
            double sq = 0, sk = 0;
            for (uint32_t i = 0; i < D; ++i) {
                qv[row + i] = nd(rng); kv[row + i] = nd(rng);
                sq += double(qv[row + i]) * qv[row + i];
                sk += double(kv[row + i]) * kv[row + i];
            }
            for (uint32_t i = 0; i < D; ++i) {
                qv[row + i] = float(qv[row + i] / std::sqrt(sq) / std::sqrt(double(D)));
                kv[row + i] = float(kv[row + i] / std::sqrt(sk));
                vv[row + i] = nd(rng);
                gv[row + i] = -5.f / (1.f + std::exp(-nd(rng)));   // realistic log decay
            }
            bv[uint64_t(t) * H + h] = 1.f / (1.f + std::exp(-nd(rng)));
        }
    float *dq = static_cast<float*>(alloc.malloc(n * 4)), *dk = static_cast<float*>(alloc.malloc(n * 4));
    float *dv = static_cast<float*>(alloc.malloc(n * 4)), *dg = static_cast<float*>(alloc.malloc(n * 4));
    float *db = static_cast<float*>(alloc.malloc(bv.size() * 4));
    float *ds = static_cast<float*>(alloc.malloc(uint64_t(H) * D * D * 4));
    float *dout = static_cast<float*>(alloc.malloc(n * 4));
    q.memcpy(dq, qv.data(), n * 4).wait(); q.memcpy(dk, kv.data(), n * 4).wait();
    q.memcpy(dv, vv.data(), n * 4).wait(); q.memcpy(dg, gv.data(), n * 4).wait();
    q.memcpy(db, bv.data(), bv.size() * 4).wait();
    q.memset(ds, 0, uint64_t(H) * D * D * 4).wait();
    kda_recurrence(q, dq, dk, dv, dg, db, ds, dout, 1, T, H, D, D, {}).wait();
    std::vector<float> out(n), state_one(uint64_t(H) * D * D);
    q.memcpy(out.data(), dout, n * 4).wait();
    q.memcpy(state_one.data(), ds, state_one.size() * 4).wait();
    {
        std::vector<double> S(uint64_t(H) * D * D, 0.0), ref(n);
        ref_kda(qv, kv, vv, gv, bv, S, ref, T, H, D);
        double mx = 0, mag = 0;
        for (uint64_t i = 0; i < n; ++i) {
            mx  = std::max(mx, std::abs(double(out[i]) - ref[i]));
            mag = std::max(mag, std::abs(ref[i]));
        }
        double smx = 0;
        for (uint64_t i = 0; i < state_one.size(); ++i)
            smx = std::max(smx, std::abs(double(state_one[i]) - S[i]));
        std::printf("      recurrence: out max abs err %.3e (max |ref| %.3f), state max err %.3e\n",
                    mx, mag, smx);
        check(mx < 5e-5 * std::max(1.0, mag), "kda_recurrence out matches fp64 (T=32 scan)");
        check(smx < 5e-5, "final state matches fp64");
    }

    // ---- 4. state-carry contract: 4 x T=8 chained == T=32, bit-exact -------
    {
        float* ds2 = static_cast<float*>(alloc.malloc(uint64_t(H) * D * D * 4));
        float* dout2 = static_cast<float*>(alloc.malloc(n * 4));
        q.memset(ds2, 0, uint64_t(H) * D * D * 4).wait();
        for (uint32_t c = 0; c < 4; ++c) {
            const uint64_t off = uint64_t(c) * 8 * H * D;
            kda_recurrence(q, dq + off, dk + off, dv + off, dg + off,
                           db + uint64_t(c) * 8 * H, ds2, dout2 + off, 1, 8, H, D, D, {}).wait();
        }
        std::vector<float> out2(n), state2(uint64_t(H) * D * D);
        q.memcpy(out2.data(), dout2, n * 4).wait();
        q.memcpy(state2.data(), ds2, state2.size() * 4).wait();
        check(std::memcmp(out2.data(), out.data(), n * 4) == 0,
              "chunked scan output BIT-IDENTICAL to one-shot");
        check(std::memcmp(state2.data(), state_one.data(), state2.size() * 4) == 0,
              "chunked final state BIT-IDENTICAL to one-shot");
        alloc.free(ds2); alloc.free(dout2);
    }
    alloc.free(dq); alloc.free(dk); alloc.free(dv); alloc.free(dg);
    alloc.free(db); alloc.free(ds); alloc.free(dout);

    std::printf("\n\033[1m%s\033[0m  (%d failure%s)\n\n",
                g_fail ? "\033[31mGATE FAILED\033[0m" : "\033[32mGATE PASSED\033[0m",
                g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
