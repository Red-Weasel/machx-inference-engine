// tests/unit/elementwise_test.cpp — Phase 3 element-wise correctness.
//
// Each op is tested with:
//   1. A small hand-computable input where the expected output is closed-form.
//   2. A larger random input where the expected is computed by a scalar
//      reference and compared with 1e-3 relative tolerance.

#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

constexpr const char* G = "\033[32m";
constexpr const char* R = "\033[31m";
constexpr const char* Z = "\033[0m";

template <typename T> T h2f(sycl::half h) { return T(float(h)); }

bool close(float a, float b, float tol = 1e-3f) {
    const float scale = std::fmax(std::fabs(a), std::fabs(b));
    if (scale < 1e-6f) return std::fabs(a - b) <= tol;
    return std::fabs(a - b) / scale <= tol;
}

bool report(const char* name, bool ok, const char* extra = "") {
    std::printf("  %s%-26s%s %s%s\n", ok ? G : R, name, Z, ok ? "OK" : "FAIL", extra);
    return ok;
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

// ===== RMSNorm =====
bool test_rmsnorm(sycl::queue& q) {
    constexpr uint32_t H = 8;
    sycl::half hx[H], hw[H], hy[H];
    for (uint32_t i = 0; i < H; ++i) { hx[i] = sycl::half(float(i + 1)); hw[i] = sycl::half(1.0f); }

    auto* dx = sycl::malloc_device<sycl::half>(H, q);
    auto* dw = sycl::malloc_device<sycl::half>(H, q);
    auto* dy = sycl::malloc_device<sycl::half>(H, q);
    q.memcpy(dx, hx, H * sizeof(sycl::half)).wait();
    q.memcpy(dw, hw, H * sizeof(sycl::half)).wait();

    ie::rms_norm(q, dx, dw, dy, /*n_rows=*/1, H, /*eps=*/1e-6f).wait();
    q.memcpy(hy, dy, H * sizeof(sycl::half)).wait();
    sycl::free(dx, q); sycl::free(dw, q); sycl::free(dy, q);

    // Reference: mean(x²) = (1+4+9+16+25+36+49+64)/8 = 204/8 = 25.5
    // rsqrt(25.5 + 1e-6) ≈ 0.198
    float sum_sq = 0.f;
    for (uint32_t i = 0; i < H; ++i) sum_sq += float(i + 1) * float(i + 1);
    const float r = 1.f / std::sqrt(sum_sq / float(H) + 1e-6f);
    bool ok = true;
    char extra[128] = {};
    for (uint32_t i = 0; i < H; ++i) {
        const float exp_v = float(i + 1) * r;
        const float got_v = float(hy[i]);
        if (!close(exp_v, got_v)) {
            ok = false;
            std::snprintf(extra, sizeof(extra), " (i=%u exp=%.4f got=%.4f)", i, exp_v, got_v);
            break;
        }
    }
    return report("RMSNorm 1x8", ok, extra);
}

// ===== SiLU =====
bool test_silu(sycl::queue& q) {
    constexpr uint64_t N = 1024;
    std::vector<sycl::half> hx(N), hy(N);
    std::mt19937 rng(0x51L);
    std::uniform_real_distribution<float> dist(-3.f, 3.f);
    for (auto& v : hx) v = sycl::half(dist(rng));

    auto* dx = sycl::malloc_device<sycl::half>(N, q);
    auto* dy = sycl::malloc_device<sycl::half>(N, q);
    q.memcpy(dx, hx.data(), N * sizeof(sycl::half)).wait();
    ie::silu(q, dx, dy, N).wait();
    q.memcpy(hy.data(), dy, N * sizeof(sycl::half)).wait();
    sycl::free(dx, q); sycl::free(dy, q);

    bool ok = true;
    char extra[128] = {};
    for (uint64_t i = 0; i < N; ++i) {
        const float v = float(hx[i]);
        const float exp_v = v / (1.f + std::exp(-v));
        const float got_v = float(hy[i]);
        if (!close(exp_v, got_v, 5e-3f)) {  // 5e-3 to absorb expf precision differences
            ok = false;
            std::snprintf(extra, sizeof(extra), " (i=%llu x=%.3f exp=%.4f got=%.4f)",
                          (unsigned long long)i, v, exp_v, got_v);
            break;
        }
    }
    return report("SiLU 1024 random", ok, extra);
}

// ===== SwiGLU =====
bool test_swiglu(sycl::queue& q) {
    constexpr uint64_t N = 1024;
    std::vector<sycl::half> hg(N), hu(N), hy(N);
    std::mt19937 rng(0x52L);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    for (auto& v : hg) v = sycl::half(dist(rng));
    for (auto& v : hu) v = sycl::half(dist(rng));

    auto* dg = sycl::malloc_device<sycl::half>(N, q);
    auto* du = sycl::malloc_device<sycl::half>(N, q);
    auto* dy = sycl::malloc_device<sycl::half>(N, q);
    q.memcpy(dg, hg.data(), N * sizeof(sycl::half)).wait();
    q.memcpy(du, hu.data(), N * sizeof(sycl::half)).wait();
    ie::swiglu(q, dg, du, dy, N).wait();
    q.memcpy(hy.data(), dy, N * sizeof(sycl::half)).wait();
    sycl::free(dg, q); sycl::free(du, q); sycl::free(dy, q);

    bool ok = true;
    char extra[128] = {};
    for (uint64_t i = 0; i < N; ++i) {
        const float g = float(hg[i]), u = float(hu[i]);
        const float exp_v = g / (1.f + std::exp(-g)) * u;
        const float got_v = float(hy[i]);
        if (!close(exp_v, got_v, 5e-3f)) {
            ok = false;
            std::snprintf(extra, sizeof(extra), " (i=%llu exp=%.4f got=%.4f)",
                          (unsigned long long)i, exp_v, got_v);
            break;
        }
    }
    return report("SwiGLU 1024 random", ok, extra);
}

// ===== Residual add =====
bool test_residual(sycl::queue& q) {
    constexpr uint64_t N = 4096;
    std::vector<sycl::half> ha(N), hb(N), hy(N);
    std::mt19937 rng(0x53L);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (uint64_t i = 0; i < N; ++i) { ha[i] = sycl::half(dist(rng)); hb[i] = sycl::half(dist(rng)); }
    auto* da = sycl::malloc_device<sycl::half>(N, q);
    auto* db = sycl::malloc_device<sycl::half>(N, q);
    auto* dy = sycl::malloc_device<sycl::half>(N, q);
    q.memcpy(da, ha.data(), N * sizeof(sycl::half)).wait();
    q.memcpy(db, hb.data(), N * sizeof(sycl::half)).wait();
    ie::residual_add(q, da, db, dy, N).wait();
    q.memcpy(hy.data(), dy, N * sizeof(sycl::half)).wait();
    sycl::free(da, q); sycl::free(db, q); sycl::free(dy, q);
    bool ok = true;
    for (uint64_t i = 0; i < N; ++i) {
        const float exp_v = float(ha[i]) + float(hb[i]);
        const float got_v = float(hy[i]);
        if (!close(exp_v, got_v, 1e-3f)) { ok = false; break; }
    }
    return report("residual_add 4096", ok);
}

// ===== Partial RoPE =====
//   Constraint #1 (high-confidence): at pos=0, output equals input bit-exactly.
//   Constraint #2: dims past `n_rotary` pass through unchanged at any pos.
//   Constraint #3: rotated norm is preserved (||(a,b)|| == ||(a',b')||).
bool test_rope(sycl::queue& q) {
    const uint32_t n_tokens = 4;
    const uint32_t n_heads  = 2;
    const uint32_t head_dim = 64;
    const uint32_t n_rotary = 32;            // factor 0.5 for the test
    const uint32_t total = n_tokens * n_heads * head_dim;

    std::vector<sycl::half> hx(total), hy(total);
    std::mt19937 rng(0x54L);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : hx) v = sycl::half(dist(rng));

    std::vector<int32_t> hpos = {0, 1, 5, 100};

    auto* dx = sycl::malloc_device<sycl::half>(total, q);
    auto* dy = sycl::malloc_device<sycl::half>(total, q);
    auto* dp = sycl::malloc_device<int32_t>(n_tokens, q);
    q.memcpy(dx, hx.data(), total * sizeof(sycl::half)).wait();
    q.memcpy(dp, hpos.data(), n_tokens * sizeof(int32_t)).wait();
    ie::rope_partial(q, dx, dp, dy, n_tokens, n_heads, head_dim, n_rotary, 1e7f).wait();
    q.memcpy(hy.data(), dy, total * sizeof(sycl::half)).wait();
    sycl::free(dx, q); sycl::free(dy, q); sycl::free(dp, q);

    bool ok = true;
    char extra[128] = {};

    // Constraint #1: pos=0 -> identity for token 0
    for (uint32_t h = 0; h < n_heads && ok; ++h) {
        for (uint32_t d = 0; d < head_dim; ++d) {
            const uint64_t idx = h * head_dim + d;
            if (float(hx[idx]) != float(hy[idx])) {
                ok = false;
                std::snprintf(extra, sizeof(extra),
                              " (token0 h=%u d=%u: in=%.4f out=%.4f)",
                              h, d, float(hx[idx]), float(hy[idx]));
                break;
            }
        }
    }
    if (!ok) return report("RoPE partial", ok, extra);

    // Constraint #2: dims past n_rotary unchanged at any token
    for (uint32_t t = 0; t < n_tokens && ok; ++t) {
        for (uint32_t h = 0; h < n_heads && ok; ++h) {
            for (uint32_t d = n_rotary; d < head_dim; ++d) {
                const uint64_t idx = (t * n_heads + h) * head_dim + d;
                if (float(hx[idx]) != float(hy[idx])) {
                    ok = false;
                    std::snprintf(extra, sizeof(extra),
                                  " (passthrough t=%u h=%u d=%u: in=%.4f out=%.4f)",
                                  t, h, d, float(hx[idx]), float(hy[idx]));
                    break;
                }
            }
        }
    }
    if (!ok) return report("RoPE partial", ok, extra);

    // Constraint #3: norm-preserving for each rotated pair (within fp16 noise)
    for (uint32_t t = 0; t < n_tokens && ok; ++t) {
        for (uint32_t h = 0; h < n_heads && ok; ++h) {
            for (uint32_t r = 0; r < n_rotary / 2 && ok; ++r) {
                const uint64_t base = (t * n_heads + h) * head_dim;
                const float a  = float(hx[base + r]);
                const float b  = float(hx[base + r + n_rotary / 2]);
                const float ap = float(hy[base + r]);
                const float bp = float(hy[base + r + n_rotary / 2]);
                const float in_norm  = a * a + b * b;
                const float out_norm = ap * ap + bp * bp;
                if (!close(in_norm, out_norm, 5e-3f)) {
                    ok = false;
                    std::snprintf(extra, sizeof(extra),
                                  " (norm t=%u h=%u r=%u: in=%.4f out=%.4f)",
                                  t, h, r, in_norm, out_norm);
                    break;
                }
            }
        }
    }

    return report("RoPE partial", ok, extra);
}

}  // namespace

int main() {
    auto q = make_queue();
    std::printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    int fails = 0;
    if (!test_residual(q)) ++fails;
    if (!test_silu    (q)) ++fails;
    if (!test_swiglu  (q)) ++fails;
    if (!test_rmsnorm (q)) ++fails;
    if (!test_rope    (q)) ++fails;
    {
        constexpr uint64_t N = 1024;
        std::vector<sycl::half> hx(N), hg(N), hy(N);
        std::mt19937 rng(0x55L);
        std::uniform_real_distribution<float> dist(-2.f, 2.f);
        for (auto& v : hx) v = sycl::half(dist(rng));
        for (auto& v : hg) v = sycl::half(dist(rng));
        auto* dx = sycl::malloc_device<sycl::half>(N, q);
        auto* dg = sycl::malloc_device<sycl::half>(N, q);
        auto* dy = sycl::malloc_device<sycl::half>(N, q);
        q.memcpy(dx, hx.data(), N*sizeof(sycl::half)).wait();
        q.memcpy(dg, hg.data(), N*sizeof(sycl::half)).wait();
        ie::sigmoid_gate(q, dx, dg, dy, N).wait();
        q.memcpy(hy.data(), dy, N*sizeof(sycl::half)).wait();
        sycl::free(dx, q); sycl::free(dg, q); sycl::free(dy, q);
        bool ok = true;
        char extra[128] = {};
        for (uint64_t i = 0; i < N; ++i) {
            const float x = float(hx[i]), g = float(hg[i]);
            const float exp_v = x * (1.f / (1.f + std::exp(-g)));
            if (!close(exp_v, float(hy[i]), 5e-3f)) {
                ok = false;
                std::snprintf(extra, sizeof(extra), " (i=%llu exp=%.4f got=%.4f)",
                              (unsigned long long)i, exp_v, float(hy[i]));
                break;
            }
        }
        if (!report("sigmoid_gate 1024 random", ok, extra)) ++fails;
    }

    std::printf("\n%s\n", fails == 0 ? "ALL PASS" : "FAILURES");
    return fails;
}
