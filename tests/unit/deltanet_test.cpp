// tests/unit/deltanet_test.cpp — Gated DeltaNet correctness.
//
// Tests:
//   1. compute_g_beta closed-form
//   2. l2_norm_scale closed-form
//   3. gated_rms_norm closed-form
//   4. Single-step sanity (T=1, S=0): closed-form per the brief
//   5. CPU recurrent reference vs GPU at T ∈ {1, 16, 256, 4096}
//   6. State persistence: prefill T=8 then decode T=1 == prefill T=9

#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

constexpr const char* G = "\033[32m";
constexpr const char* R = "\033[31m";
constexpr const char* Z = "\033[0m";

bool report(const char* name, bool ok, const char* extra = "") {
    std::printf("  %s%-48s%s %s%s\n", ok ? G : R, name, Z, ok ? "OK" : "FAIL", extra);
    return ok;
}

bool close(float a, float b, float tol = 1e-2f) {
    return std::fabs(a - b) <= tol;
}

bool close_rel(float a, float b, float tol_abs = 1e-3f, float tol_rel = 1e-2f) {
    const float d = std::fabs(a - b);
    if (d <= tol_abs) return true;
    const float scale = std::fmax(std::fabs(a), std::fabs(b));
    return scale > 0 && d / scale <= tol_rel;
}

sycl::queue make_queue() {
    sycl::device dev;
    bool found = false;
    for (const auto& d : sycl::device::get_devices()) {
        if (d.is_gpu() && d.get_info<sycl::info::device::name>().find("0xe223") != std::string::npos) {
            dev = d; found = true; break;
        }
    }
    if (!found) for (const auto& d : sycl::device::get_devices()) if (d.is_gpu()) { dev = d; found = true; break; }
    if (!found) { std::fputs("no GPU\n", stderr); std::exit(2); }
    return sycl::queue(dev, sycl::property::queue::enable_profiling{});
}

template <typename T>
T* upload(sycl::queue& q, const std::vector<T>& host) {
    T* d = sycl::malloc_device<T>(host.size(), q);
    q.memcpy(d, host.data(), host.size() * sizeof(T)).wait();
    return d;
}

// ---- CPU reference for the recurrence (research/05 §7 verbatim) ----
void recurrence_ref(uint32_t B, uint32_t T, uint32_t H, uint32_t K, uint32_t V,
                    const std::vector<float>& q,
                    const std::vector<float>& k,
                    const std::vector<float>& v,
                    const std::vector<float>& g,
                    const std::vector<float>& beta,
                    std::vector<float>& state,
                    std::vector<float>& out)
{
    // Shapes:
    //   q, k: [B, T, H, K]  (post-l2norm, post-Q-scale on q)
    //   v:    [B, T, H, V]
    //   g, beta: [B, T, H]
    //   state: [B, H, K, V] in/out
    //   out:   [B, T, H, V]
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t h = 0; h < H; ++h) {
            float* S = state.data() + ((b * H + h) * K * V);
            for (uint32_t t = 0; t < T; ++t) {
                const float alpha = std::exp(g[(b * T + t) * H + h]);
                const float bt    = beta[(b * T + t) * H + h];
                const float* qt = q.data() + ((b * T + t) * H + h) * K;
                const float* kt = k.data() + ((b * T + t) * H + h) * K;
                const float* vt = v.data() + ((b * T + t) * H + h) * V;

                // 9a: decay
                for (uint32_t kk = 0; kk < K; ++kk)
                    for (uint32_t vv = 0; vv < V; ++vv)
                        S[kk * V + vv] *= alpha;

                // 9b: kv_mem[vv] = sum_kk S[kk, vv] * kt[kk]
                std::vector<float> kv_mem(V, 0.f);
                for (uint32_t kk = 0; kk < K; ++kk)
                    for (uint32_t vv = 0; vv < V; ++vv)
                        kv_mem[vv] += S[kk * V + vv] * kt[kk];

                // 9c: δ
                std::vector<float> delta(V);
                for (uint32_t vv = 0; vv < V; ++vv)
                    delta[vv] = (vt[vv] - kv_mem[vv]) * bt;

                // 9d: rank-1 update
                for (uint32_t kk = 0; kk < K; ++kk)
                    for (uint32_t vv = 0; vv < V; ++vv)
                        S[kk * V + vv] += kt[kk] * delta[vv];

                // 9e: out = q^T S
                float* outt = out.data() + ((b * T + t) * H + h) * V;
                for (uint32_t vv = 0; vv < V; ++vv) {
                    float acc = 0.f;
                    for (uint32_t kk = 0; kk < K; ++kk) acc += S[kk * V + vv] * qt[kk];
                    outt[vv] = acc;
                }
            }
        }
    }
}

// ---- Test 1: compute_g_beta closed-form ----
bool test_compute_g_beta(sycl::queue& q) {
    constexpr uint32_t N = 4, H = 8;
    std::vector<float> ha(N * H, 0.f), hb(N * H, 0.f);
    // 2026-06-09 contract update: compute_g_beta takes A as stored in the
    // GGUF (= -exp(A_log), pre-baked by the converter) and computes
    // g = A * softplus(a + dt_bias) directly.  For A_log = 0, pass A = -1.
    std::vector<float> hAlog(H, -1.0f), hdt(H, 0.f);
    // a = b = 0, A = -exp(0) = -1, dt_bias = 0:
    //   g = -1 * softplus(0) = -ln(2) = -0.6931
    //   β = sigmoid(0) = 0.5
    auto* da = upload(q, ha);
    auto* db = upload(q, hb);
    auto* dA = upload(q, hAlog);
    auto* ddt = upload(q, hdt);
    auto* dg = sycl::malloc_device<float>(N * H, q);
    auto* dbeta = sycl::malloc_device<float>(N * H, q);
    ie::compute_g_beta(q, da, db, dA, ddt, dg, dbeta, N, H).wait();
    std::vector<float> hg(N * H), hbeta(N * H);
    q.memcpy(hg.data(), dg, hg.size() * sizeof(float)).wait();
    q.memcpy(hbeta.data(), dbeta, hbeta.size() * sizeof(float)).wait();
    sycl::free(da, q); sycl::free(db, q); sycl::free(dA, q); sycl::free(ddt, q);
    sycl::free(dg, q); sycl::free(dbeta, q);
    bool ok = true;
    char extra[128] = {};
    for (uint32_t i = 0; i < N * H; ++i) {
        if (!close(hg[i], -std::log(2.f), 1e-4f) || !close(hbeta[i], 0.5f, 1e-4f)) {
            ok = false;
            std::snprintf(extra, sizeof(extra), " (i=%u g=%.4f β=%.4f)",
                          i, hg[i], hbeta[i]);
            break;
        }
    }
    return report("compute_g_beta(0,0,0,0)→(-ln2, 0.5)", ok, extra);
}

// ---- Test 2: l2_norm_scale closed-form ----
bool test_l2norm(sycl::queue& q) {
    constexpr uint32_t N = 1, D = 16;
    std::vector<float> hx(D);
    for (uint32_t i = 0; i < D; ++i) hx[i] = float(i + 1);   // norm² = 1+4+...+256 = 1496
    auto* dx = upload(q, hx);
    auto* dy = sycl::malloc_device<float>(D, q);
    ie::l2_norm_scale(q, dx, dy, N, D, /*scale=*/1.0f / std::sqrt(float(D)), 1e-6f).wait();
    std::vector<float> hy(D);
    q.memcpy(hy.data(), dy, hy.size() * sizeof(float)).wait();
    sycl::free(dx, q); sycl::free(dy, q);

    // Expected: y[i] = x[i] * rsqrt(sum(x²) + eps) * scale
    const float sum_sq = 1496.f;
    const float r = 1.f / std::sqrt(sum_sq + 1e-6f) * (1.f / std::sqrt(float(D)));
    bool ok = true;
    char extra[128] = {};
    for (uint32_t i = 0; i < D; ++i) {
        const float exp_v = float(i + 1) * r;
        if (!close(exp_v, hy[i], 1e-4f)) {
            ok = false;
            std::snprintf(extra, sizeof(extra), " (i=%u exp=%.5f got=%.5f)", i, exp_v, hy[i]);
            break;
        }
    }
    return report("l2_norm_scale 1×16", ok, extra);
}

// ---- Test 3: gated_rms_norm closed-form ----
bool test_gated_rms_norm(sycl::queue& q) {
    constexpr uint32_t N = 1, H = 8;
    std::vector<float> hx(H);
    std::vector<sycl::half> hz(H), hw(H, sycl::half(1.0f));
    for (uint32_t i = 0; i < H; ++i) {
        hx[i] = float(i + 1);
        hz[i] = sycl::half(0.5f);
    }
    auto* dx = upload(q, hx);
    auto* dz = upload(q, hz);
    auto* dw = upload(q, hw);
    auto* dy = sycl::malloc_device<sycl::half>(H, q);
    ie::gated_rms_norm(q, dx, dz, dw, dy, N, H, 1e-6f).wait();
    std::vector<sycl::half> hy(H);
    q.memcpy(hy.data(), dy, hy.size() * sizeof(sycl::half)).wait();
    sycl::free(dx, q); sycl::free(dz, q); sycl::free(dw, q); sycl::free(dy, q);

    // sum_sq=204, mean=204/8=25.5, rsqrt=1/√25.5 ≈ 0.198
    // x_norm[i] = (i+1)*0.198
    // silu(0.5) = 0.5 * sigmoid(0.5) = 0.5 * 0.6225 = 0.3112
    // y[i] = 1 * x_norm[i] * silu(0.5)
    const float r = 1.f / std::sqrt(204.f / 8.f + 1e-6f);
    const float silu_z = 0.5f / (1.f + std::exp(-0.5f));
    bool ok = true;
    char extra[128] = {};
    for (uint32_t i = 0; i < H; ++i) {
        const float exp_v = (float(i + 1) * r) * silu_z;
        if (!close(exp_v, float(hy[i]), 1e-2f)) {
            ok = false;
            std::snprintf(extra, sizeof(extra), " (i=%u exp=%.4f got=%.4f)", i, exp_v, float(hy[i]));
            break;
        }
    }
    return report("gated_rms_norm closed-form 1×8", ok, extra);
}

// ---- Test 4: Single-step recurrence sanity ----
//
// With S=0, T=1, the recurrence collapses to:
//   S = α · 0 = 0
//   kv_mem = 0
//   δ = (v − 0) · β = β v
//   S = 0 + k ⊗ (β v) = β · k ⊗ v
//   out = qᵀ · S = β · (qᵀk) · v
bool test_single_step(sycl::queue& q) {
    constexpr uint32_t B = 1, T = 1, H = 1, K = 128, V = 128;
    std::mt19937 rng(0xD1);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> hq(K), hk(K), hv(V);
    for (auto& x : hq) x = dist(rng);
    for (auto& x : hk) x = dist(rng);
    for (auto& x : hv) x = dist(rng);
    const float beta = 0.42f;
    const float g    = 0.0f;   // α = 1 (irrelevant since S=0)

    std::vector<float> hg(1, g), hbeta(1, beta);
    std::vector<float> hstate(K * V, 0.f);
    std::vector<float> hout(V, 0.f);

    auto* dq = upload(q, hq);
    auto* dk = upload(q, hk);
    auto* dv = upload(q, hv);
    auto* dg = upload(q, hg);
    auto* dbeta = upload(q, hbeta);
    auto* dstate = upload(q, hstate);
    auto* dout = sycl::malloc_device<float>(V, q);
    ie::deltanet_recurrence(q, dq, dk, dv, dg, dbeta, dstate, dout, B, T, H, K, V).wait();
    q.memcpy(hout.data(), dout, V * sizeof(float)).wait();
    sycl::free(dq, q); sycl::free(dk, q); sycl::free(dv, q); sycl::free(dg, q);
    sycl::free(dbeta, q); sycl::free(dstate, q); sycl::free(dout, q);

    // Reference: out[vv] = β · (qᵀk) · v[vv]
    float qk = 0.f;
    for (uint32_t i = 0; i < K; ++i) qk += hq[i] * hk[i];
    bool ok = true;
    char extra[128] = {};
    for (uint32_t vv = 0; vv < V; ++vv) {
        const float exp_v = beta * qk * hv[vv];
        if (!close_rel(exp_v, hout[vv], 1e-3f, 5e-3f)) {
            ok = false;
            std::snprintf(extra, sizeof(extra),
                          " (vv=%u exp=%.5f got=%.5f β·qᵀk=%.5f)",
                          vv, exp_v, hout[vv], beta * qk);
            break;
        }
    }
    return report("single-step S=0 → β·(qᵀk)·v", ok, extra);
}

// ---- Test 5: CPU reference vs GPU recurrence ----
bool test_vs_reference(sycl::queue& q, uint32_t T, const char* label) {
    constexpr uint32_t B = 1, H = 4, K = 128, V = 128;
    std::mt19937 rng(0xE2 + T);
    std::uniform_real_distribution<float> qkv_dist(-0.2f, 0.2f);
    std::uniform_real_distribution<float> g_dist(-0.5f, 0.0f);
    std::uniform_real_distribution<float> b_dist(0.1f, 0.9f);

    std::vector<float> hq(B * T * H * K), hk(B * T * H * K), hv(B * T * H * V);
    std::vector<float> hg(B * T * H), hbeta(B * T * H);
    std::vector<float> hstate_init(B * H * K * V, 0.f);
    for (auto& x : hq) x = qkv_dist(rng);
    for (auto& x : hk) x = qkv_dist(rng);
    for (auto& x : hv) x = qkv_dist(rng);
    for (auto& x : hg) x = g_dist(rng);
    for (auto& x : hbeta) x = b_dist(rng);

    // Reference
    std::vector<float> hstate_ref = hstate_init;
    std::vector<float> hout_ref(B * T * H * V);
    recurrence_ref(B, T, H, K, V, hq, hk, hv, hg, hbeta, hstate_ref, hout_ref);

    // GPU
    auto* dq = upload(q, hq);
    auto* dk = upload(q, hk);
    auto* dv = upload(q, hv);
    auto* dg = upload(q, hg);
    auto* dbeta = upload(q, hbeta);
    auto* dstate = upload(q, hstate_init);
    auto* dout = sycl::malloc_device<float>(B * T * H * V, q);
    ie::deltanet_recurrence(q, dq, dk, dv, dg, dbeta, dstate, dout, B, T, H, K, V).wait();
    std::vector<float> hout_gpu(B * T * H * V);
    std::vector<float> hstate_gpu(B * H * K * V);
    q.memcpy(hout_gpu.data(), dout, hout_gpu.size() * sizeof(float)).wait();
    q.memcpy(hstate_gpu.data(), dstate, hstate_gpu.size() * sizeof(float)).wait();
    sycl::free(dq, q); sycl::free(dk, q); sycl::free(dv, q); sycl::free(dg, q);
    sycl::free(dbeta, q); sycl::free(dstate, q); sycl::free(dout, q);

    auto diff = [&](const std::vector<float>& a, const std::vector<float>& b) {
        float max_abs = 0.f;
        for (size_t i = 0; i < a.size(); ++i) {
            const float d = std::fabs(a[i] - b[i]);
            if (d > max_abs) max_abs = d;
        }
        return max_abs;
    };
    const float dout_max = diff(hout_ref, hout_gpu);
    const float dstate_max = diff(hstate_ref, hstate_gpu);
    char extra[160];
    std::snprintf(extra, sizeof(extra),
                  "  out_max_abs=%.4g  state_max_abs=%.4g (gates: out≤1e-2, state≤5e-3)",
                  dout_max, dstate_max);
    return report(label, dout_max <= 1e-2f && dstate_max <= 5e-3f, extra);
}

// ---- Test 6: prefill + decode = prefill ----
bool test_state_persistence(sycl::queue& q) {
    constexpr uint32_t B = 1, H = 2, K = 128, V = 128;
    constexpr uint32_t T_pre = 8, T_dec = 1, T_full = T_pre + T_dec;
    std::mt19937 rng(0xF3);
    std::uniform_real_distribution<float> qkv(-0.2f, 0.2f);
    std::uniform_real_distribution<float> gd(-0.5f, 0.0f);
    std::uniform_real_distribution<float> bd(0.1f, 0.9f);

    std::vector<float> hq(B * T_full * H * K), hk(B * T_full * H * K), hv(B * T_full * H * V);
    std::vector<float> hg(B * T_full * H), hbeta(B * T_full * H);
    for (auto& x : hq) x = qkv(rng);
    for (auto& x : hk) x = qkv(rng);
    for (auto& x : hv) x = qkv(rng);
    for (auto& x : hg) x = gd(rng);
    for (auto& x : hbeta) x = bd(rng);

    auto run_chunk = [&](uint32_t t0, uint32_t T_chunk, std::vector<float>& state_io) {
        auto* dq    = sycl::malloc_device<float>(B * T_chunk * H * K, q);
        auto* dk    = sycl::malloc_device<float>(B * T_chunk * H * K, q);
        auto* dv    = sycl::malloc_device<float>(B * T_chunk * H * V, q);
        auto* dg    = sycl::malloc_device<float>(B * T_chunk * H, q);
        auto* dbeta = sycl::malloc_device<float>(B * T_chunk * H, q);
        auto* dstate = sycl::malloc_device<float>(state_io.size(), q);
        auto* dout  = sycl::malloc_device<float>(B * T_chunk * H * V, q);
        q.memcpy(dq,    hq.data() + t0 * H * K, B * T_chunk * H * K * sizeof(float)).wait();
        q.memcpy(dk,    hk.data() + t0 * H * K, B * T_chunk * H * K * sizeof(float)).wait();
        q.memcpy(dv,    hv.data() + t0 * H * V, B * T_chunk * H * V * sizeof(float)).wait();
        q.memcpy(dg,    hg.data() + t0 * H,    B * T_chunk * H * sizeof(float)).wait();
        q.memcpy(dbeta, hbeta.data() + t0 * H, B * T_chunk * H * sizeof(float)).wait();
        q.memcpy(dstate, state_io.data(),      state_io.size() * sizeof(float)).wait();
        ie::deltanet_recurrence(q, dq, dk, dv, dg, dbeta, dstate, dout, B, T_chunk, H, K, V).wait();
        std::vector<float> out_chunk(B * T_chunk * H * V);
        q.memcpy(out_chunk.data(), dout, out_chunk.size() * sizeof(float)).wait();
        q.memcpy(state_io.data(), dstate, state_io.size() * sizeof(float)).wait();
        sycl::free(dq, q); sycl::free(dk, q); sycl::free(dv, q); sycl::free(dg, q);
        sycl::free(dbeta, q); sycl::free(dstate, q); sycl::free(dout, q);
        return out_chunk;
    };

    // Path A: full prefill
    std::vector<float> stateA(B * H * K * V, 0.f);
    auto outA = run_chunk(0, T_full, stateA);
    // Path B: prefill T_pre, then decode T_dec (state carries)
    std::vector<float> stateB(B * H * K * V, 0.f);
    auto out_pre = run_chunk(0, T_pre, stateB);
    auto out_dec = run_chunk(T_pre, T_dec, stateB);

    // Compare last-token output
    float max_abs = 0.f;
    for (uint32_t hh = 0; hh < H; ++hh)
        for (uint32_t vv = 0; vv < V; ++vv) {
            const float a = outA[((T_pre) * H + hh) * V + vv];
            const float b = out_dec[(0 * H + hh) * V + vv];
            const float d = std::fabs(a - b);
            if (d > max_abs) max_abs = d;
        }
    // Also compare end-of-sequence states
    float state_max = 0.f;
    for (size_t i = 0; i < stateA.size(); ++i)
        state_max = std::fmax(state_max, std::fabs(stateA[i] - stateB[i]));
    char extra[160];
    std::snprintf(extra, sizeof(extra),
                  "  out_diff=%.4g state_diff=%.4g", max_abs, state_max);
    return report("prefill T=8 + decode T=1 == prefill T=9", max_abs <= 1e-3f && state_max <= 1e-3f, extra);
}

}  // namespace

int main() {
    auto q = make_queue();
    std::printf("Device: %s\n\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    int fails = 0;
    if (!test_compute_g_beta(q))    ++fails;
    if (!test_l2norm(q))            ++fails;
    if (!test_gated_rms_norm(q))    ++fails;
    if (!test_single_step(q))       ++fails;
    if (!test_vs_reference(q, 1,    "vs CPU ref T=1"))    ++fails;
    if (!test_vs_reference(q, 16,   "vs CPU ref T=16"))   ++fails;
    if (!test_vs_reference(q, 256,  "vs CPU ref T=256"))  ++fails;
    if (!test_vs_reference(q, 4096, "vs CPU ref T=4096")) ++fails;
    if (!test_state_persistence(q)) ++fails;

    // Conv1d smoke test: kernel=4, channels=2, T=8.
    // 2026-06-09 contract update (kernel changed in c8c00ea + GGUF layout):
    //   - weight (k, c) lives at offset k + c*K (kernel dim contiguous);
    //   - W[K-1] multiplies the CURRENT tap, W[0] the oldest (PyTorch conv);
    //   - SiLU is fused into the output.
    // x[t, c=0] = 1, w[c=0] all-ones        → y[t, 0] = silu(min(t+1, 4))
    // x[t, c=1] = t+1, w[c=1] current-tap-1 → y[t, 1] = silu(t+1)
    {
        constexpr uint32_t T = 8, C = 2, K = 4;
        std::vector<sycl::half> hx(T * C), hw(K * C), hy(T * C);
        for (uint32_t t = 0; t < T; ++t) {
            hx[t * C + 0] = sycl::half(1.0f);
            hx[t * C + 1] = sycl::half(float(t + 1));
        }
        for (uint32_t k = 0; k < K; ++k) {
            hw[0 * K + k] = sycl::half(1.0f);
            hw[1 * K + k] = sycl::half(k == K - 1 ? 1.0f : 0.0f);
        }
        auto* dx = upload(q, hx);
        auto* dw = upload(q, hw);
        auto* dy = sycl::malloc_device<sycl::half>(T * C, q);
        ie::depthwise_conv1d_causal(q, dx, dw, /*conv_state=*/nullptr, dy, T, C, K).wait();
        q.memcpy(hy.data(), dy, hy.size() * sizeof(sycl::half)).wait();
        sycl::free(dx, q); sycl::free(dw, q); sycl::free(dy, q);
        bool ok = true;
        char extra[128] = {};
        auto silu = [](float v) { return v / (1.0f + std::exp(-v)); };
        for (uint32_t t = 0; t < T; ++t) {
            const float exp_c0 = silu(float(std::min<uint32_t>(t + 1, K)));
            const float exp_c1 = silu(float(t + 1));
            if (!close(exp_c0, float(hy[t * C + 0]), 5e-3f) ||
                !close(exp_c1, float(hy[t * C + 1]), 5e-3f)) {
                ok = false;
                std::snprintf(extra, sizeof(extra),
                              " (t=%u c0 exp=%.0f got=%.2f, c1 exp=%.0f got=%.2f)",
                              t, exp_c0, float(hy[t * C + 0]), exp_c1, float(hy[t * C + 1]));
                break;
            }
        }
        if (!report("depthwise_conv1d_causal smoke", ok, extra)) ++fails;
    }

    std::printf("\n%s\n", fails == 0 ? "ALL PASS" : "FAILURES");
    return fails;
}
