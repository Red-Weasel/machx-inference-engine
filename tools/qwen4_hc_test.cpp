// tools/qwen4_hc_test.cpp — qwen4exp hyper-connections correctness gate.
//
// Self-contained (NO model file): random tensors with fixed seeds, CPU fp64
// reference implementing docs/qwen4/11_hyper_connections.md directly, GPU
// kernels from src/ops/qwen4_hc.cpp compared against it.
//   * MIX (f16-weight GEMV chain): max rel error <= 2e-3
//   * COMBINE (no quantized weights): max rel error <= 1e-5
//   * Degenerate: inj_logits = 0  => combine adds block_out EXACTLY once per
//     stream (bit-equal fp32); gamma = 1 + identical streams => mixed equals
//     the single-stream normed * mean(gate) sanity relation.
//   * FINAL MERGE = mix with w_inject == nullptr (mixed only).
//   * expand_streams: wide[t,s,:] == float(emb[t,:]) exactly.
//
// usage: ie-qwen4-hc-test   (exit 0 on GATE PASSED, 1 on GATE FAILED)

#include "ie/ops.hpp"
#include "ie/qwen4_hc.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace ie;

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "\033[32mOK\033[0m" : "\033[31mFAIL\033[0m", what);
    if (!ok) ++g_fail;
}

void check_tol(double err, double tol, const char* what) {
    const bool ok = (err <= tol) && std::isfinite(err);
    std::printf("  [%s] %-52s max_rel %.3e (tol %.0e)\n",
                ok ? "\033[32mOK\033[0m" : "\033[31mFAIL\033[0m", what, err, tol);
    if (!ok) ++g_fail;
}

// max |got-ref| / max(|ref|, floor)
template <class G>
double max_rel(const G* got, const double* ref, size_t n, double floor_ = 1e-2) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double g = double(float(got[i]));
        const double e = std::fabs(g - ref[i]) / std::max(std::fabs(ref[i]), floor_);
        if (e > m) m = e;
    }
    return m;
}

template <class G>
double max_abs(const G* got, const double* ref, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i)
        m = std::max(m, std::fabs(double(float(got[i])) - ref[i]));
    return m;
}

double d_silu(double a)    { return a / (1.0 + std::exp(-a)); }
double d_sigmoid(double a) { return 1.0 / (1.0 + std::exp(-a)); }

// Fixed dims (qwen4exp): H=2560, hc=4, r=320, D=10240, eps=1e-6.
constexpr uint32_t H  = 2560;
constexpr uint32_t HC = 4;
constexpr uint32_t R  = 320;
constexpr uint32_t D  = HC * H;
constexpr float    EPS = 1e-6f;

struct RefMix {
    std::vector<double> xn;     // [T, D]
    std::vector<double> mixed;  // [T, H]
    std::vector<double> inj;    // [T, HC]
};

// CPU fp64 reference of the MIX spec math.  Weights are passed as the SAME
// f16/f32 values the GPU consumes (converted to double), so the only error
// left is precision/accumulation-order.
RefMix ref_mix(uint32_t T, const std::vector<float>& x, const std::vector<float>& gamma,
               const std::vector<sycl::half>& w_down, const std::vector<sycl::half>& w_up,
               const std::vector<float>& w_inject) {
    RefMix o;
    o.xn.assign(size_t(T) * D, 0.0);
    o.mixed.assign(size_t(T) * H, 0.0);
    o.inj.assign(size_t(T) * HC, 0.0);
    std::vector<double> lo(R), gate(D);
    for (uint32_t t = 0; t < T; ++t) {
        const float* xt = x.data() + size_t(t) * D;
        double* xnt = o.xn.data() + size_t(t) * D;
        // Grouped RMS norm: RMS per 2560-stream group, gamma spans all D.
        for (uint32_t s = 0; s < HC; ++s) {
            double ss = 0.0;
            for (uint32_t i = 0; i < H; ++i) {
                const double v = double(xt[s * H + i]);
                ss += v * v;
            }
            const double inv = 1.0 / std::sqrt(ss / double(H) + double(EPS));
            for (uint32_t i = 0; i < H; ++i) {
                const uint32_t j = s * H + i;
                xnt[j] = double(xt[j]) * inv * double(gamma[j]);
            }
        }
        // lo = silu((W_down . xn) / hc)
        for (uint32_t i = 0; i < R; ++i) {
            double acc = 0.0;
            const sycl::half* wr = w_down.data() + size_t(i) * D;
            for (uint32_t j = 0; j < D; ++j) acc += double(float(wr[j])) * xnt[j];
            lo[i] = d_silu(acc / double(HC));
        }
        // gate = sigmoid(W_up . lo)
        for (uint32_t d = 0; d < D; ++d) {
            double acc = 0.0;
            const sycl::half* wr = w_up.data() + size_t(d) * R;
            for (uint32_t k = 0; k < R; ++k) acc += double(float(wr[k])) * lo[k];
            gate[d] = d_sigmoid(acc);
        }
        // mixed = mean_s(xn . gate)
        for (uint32_t ch = 0; ch < H; ++ch) {
            double m = 0.0;
            for (uint32_t s = 0; s < HC; ++s) m += xnt[s * H + ch] * gate[s * H + ch];
            o.mixed[size_t(t) * H + ch] = m / double(HC);
        }
        // inj = W_inject . xn (PRE-block xn)
        for (uint32_t s = 0; s < HC; ++s) {
            double acc = 0.0;
            for (uint32_t j = 0; j < D; ++j)
                acc += double(w_inject[size_t(s) * D + j]) * xnt[j];
            o.inj[size_t(t) * HC + s] = acc;
        }
    }
    return o;
}

template <class T2>
T2* dcopy(sycl::queue& q, const std::vector<T2>& v) {
    T2* p = sycl::malloc_device<T2>(v.size(), q);
    q.memcpy(p, v.data(), v.size() * sizeof(T2)).wait();
    return p;
}

}  // namespace

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("\n\033[1mqwen4exp hyper-connections gate\033[0m  device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("  H=%u hc=%u r=%u D=%u eps=%g\n", H, HC, R, D, double(EPS));

    // ---- shared weights (fixed seed, f16 values used by ref AND gpu) -------
    std::mt19937 rng(12345);
    auto rndf = [&](float lo, float hi) {
        return std::uniform_real_distribution<float>(lo, hi)(rng);
    };
    std::vector<float>      gamma(D);
    std::vector<sycl::half> w_down(size_t(R) * D), w_up(size_t(D) * R);
    std::vector<float>      w_inject(size_t(HC) * D);
    for (auto& v : gamma)    v = rndf(0.5f, 1.5f);
    for (auto& v : w_down)   v = sycl::half(rndf(-0.05f, 0.05f));
    for (auto& v : w_up)     v = sycl::half(rndf(-0.05f, 0.05f));
    for (auto& v : w_inject) v = rndf(-0.02f, 0.02f);

    float*      d_gamma  = dcopy(q, gamma);
    sycl::half* d_wdown  = dcopy(q, w_down);
    sycl::half* d_wup    = dcopy(q, w_up);
    float*      d_winj   = dcopy(q, w_inject);
    // [K, N]-transposed copies for the prefill mix (mirrors the model's
    // load-time transposes; same f16 values, so the fp64 ref is unchanged).
    std::vector<sycl::half> w_down_t(size_t(D) * R), w_up_t(size_t(R) * D);
    for (uint32_t i = 0; i < R; ++i)
        for (uint32_t j = 0; j < D; ++j)
            w_down_t[size_t(j) * R + i] = w_down[size_t(i) * D + j];
    for (uint32_t d2 = 0; d2 < D; ++d2)
        for (uint32_t k2 = 0; k2 < R; ++k2)
            w_up_t[size_t(k2) * D + d2] = w_up[size_t(d2) * R + k2];
    sycl::half* d_wdown_t = dcopy(q, w_down_t);
    sycl::half* d_wup_t   = dcopy(q, w_up_t);

    const uint32_t Ts[3] = {1, 7, 256};
    for (uint32_t T : Ts) {
        std::printf("\n\033[1mT = %u\033[0m\n", T);
        std::mt19937 rt(777 + T);
        auto rf = [&](float lo, float hi) {
            return std::uniform_real_distribution<float>(lo, hi)(rt);
        };

        // ---- MIX -----------------------------------------------------------
        std::vector<float> x(size_t(T) * D);
        for (auto& v : x) v = rf(-1.f, 1.f);
        const RefMix ref = ref_mix(T, x, gamma, w_down, w_up, w_inject);

        float*      d_x     = dcopy(q, x);
        float*      d_xn    = sycl::malloc_device<float>(size_t(T) * D, q);
        float*      d_lo    = sycl::malloc_device<float>(size_t(T) * R, q);
        sycl::half* d_mixed = sycl::malloc_device<sycl::half>(size_t(T) * H, q);
        float*      d_inj   = sycl::malloc_device<float>(size_t(T) * HC, q);

        qwen4_hc_mix(q, d_x, d_gamma, d_wdown, d_wup, d_winj,
                     d_xn, d_lo, d_mixed, d_inj, T, H, HC, R, EPS).wait();

        std::vector<sycl::half> mixed(size_t(T) * H);
        std::vector<float>      inj(size_t(T) * HC);
        q.memcpy(mixed.data(), d_mixed, mixed.size() * sizeof(sycl::half)).wait();
        q.memcpy(inj.data(), d_inj, inj.size() * sizeof(float)).wait();
        check_tol(max_rel(mixed.data(), ref.mixed.data(), mixed.size()), 2e-3,
                  "mix: mixed vs fp64 ref");
        check_tol(max_rel(inj.data(), ref.inj.data(), inj.size()), 2e-3,
                  "mix: inj_logits vs fp64 ref");

        // ---- FINAL MERGE (w_inject == nullptr) -----------------------------
        sycl::half* d_mixed2 = sycl::malloc_device<sycl::half>(size_t(T) * H, q);
        qwen4_hc_mix(q, d_x, d_gamma, d_wdown, d_wup, /*w_inject=*/nullptr,
                     d_xn, d_lo, d_mixed2, /*inj_logits=*/nullptr,
                     T, H, HC, R, EPS).wait();
        std::vector<sycl::half> mixed2(size_t(T) * H);
        q.memcpy(mixed2.data(), d_mixed2, mixed2.size() * sizeof(sycl::half)).wait();
        check_tol(max_rel(mixed2.data(), ref.mixed.data(), mixed2.size()), 2e-3,
                  "final merge (no inject): mixed vs fp64 ref");

        // ---- MIX v2 (device-wide decomposition; identical math) ------------
        qwen4_hc_mix_v2(q, d_x, d_gamma, d_wdown, d_wup, d_winj,
                        d_xn, d_lo, d_mixed, d_inj, T, H, HC, R, EPS).wait();
        q.memcpy(mixed.data(), d_mixed, mixed.size() * sizeof(sycl::half)).wait();
        q.memcpy(inj.data(), d_inj, inj.size() * sizeof(float)).wait();
        check_tol(max_rel(mixed.data(), ref.mixed.data(), mixed.size()), 2e-3,
                  "mix v2: mixed vs fp64 ref");
        check_tol(max_rel(inj.data(), ref.inj.data(), inj.size()), 2e-3,
                  "mix v2: inj_logits vs fp64 ref");
        qwen4_hc_mix_v2(q, d_x, d_gamma, d_wdown, d_wup, /*w_inject=*/nullptr,
                        d_xn, d_lo, d_mixed2, /*inj_logits=*/nullptr,
                        T, H, HC, R, EPS).wait();
        q.memcpy(mixed2.data(), d_mixed2, mixed2.size() * sizeof(sycl::half)).wait();
        check_tol(max_rel(mixed2.data(), ref.mixed.data(), mixed2.size()), 2e-3,
                  "final merge v2: mixed vs fp64 ref");

        // ---- MIX prefill (T > 16: oneDNN NT gemm path, W16A16 down/up) -----
        // Wider mixed tolerance (8e-3): the A operands are deliberately
        // rounded to F16 before each GEMM (the dense-prefill convention) and
        // the fp64 reference chain is unrounded — measured max_rel 6.0e-3
        // over T*H = 650K elements (worst element at a sigmoid near-tie);
        // layout/scale bugs measure O(0.1-20) here, so the band still bites.
        // inj keeps v2's fp32 dot and holds the v2 tolerance. Quality at
        // model scale is adjudicated by the chunked-PPL gate, as for A16
        // dense.
        if (T > 16) {
            sycl::half* d_xn16 = sycl::malloc_device<sycl::half>(size_t(T) * D, q);
            sycl::half* d_lo16 = sycl::malloc_device<sycl::half>(size_t(T) * R, q);
            float*      d_gws  = sycl::malloc_device<float>(size_t(T) * D, q);
            qwen4_hc_mix_prefill(q, d_x, d_gamma, d_wdown_t, d_wup_t, d_winj,
                                 d_xn, d_xn16, d_lo16, d_gws, d_mixed, d_inj,
                                 T, H, HC, R, EPS).wait();
            q.memcpy(mixed.data(), d_mixed, mixed.size() * sizeof(sycl::half)).wait();
            q.memcpy(inj.data(), d_inj, inj.size() * sizeof(float)).wait();
            check_tol(max_rel(mixed.data(), ref.mixed.data(), mixed.size()), 8e-3,
                      "mix prefill: mixed vs fp64 ref");
            check_tol(max_rel(inj.data(), ref.inj.data(), inj.size()), 2e-3,
                      "mix prefill: inj_logits vs fp64 ref");
            qwen4_hc_mix_prefill(q, d_x, d_gamma, d_wdown_t, d_wup_t, nullptr,
                                 d_xn, d_xn16, d_lo16, d_gws, d_mixed2, nullptr,
                                 T, H, HC, R, EPS).wait();
            q.memcpy(mixed2.data(), d_mixed2,
                     mixed2.size() * sizeof(sycl::half)).wait();
            check_tol(max_rel(mixed2.data(), ref.mixed.data(), mixed2.size()), 8e-3,
                      "final merge prefill: mixed vs fp64 ref");
            sycl::free(d_xn16, q); sycl::free(d_lo16, q); sycl::free(d_gws, q);
        }

        // ---- COMBINE -------------------------------------------------------
        std::vector<float>      res(size_t(T) * HC * H);
        std::vector<sycl::half> blk(size_t(T) * H);
        std::vector<float>      injc(size_t(T) * HC);
        for (auto& v : res)  v = rf(-1.f, 1.f);
        for (auto& v : blk)  v = sycl::half(rf(-1.f, 1.f));
        for (auto& v : injc) v = rf(-2.f, 2.f);
        std::vector<double> res_ref(res.size());
        for (uint32_t t = 0; t < T; ++t)
            for (uint32_t s = 0; s < HC; ++s) {
                const double w = 2.0 * d_sigmoid(double(injc[size_t(t) * HC + s]) / double(HC));
                for (uint32_t ch = 0; ch < H; ++ch) {
                    const size_t i = (size_t(t) * HC + s) * H + ch;
                    res_ref[i] = double(res[i]) + w * double(float(blk[size_t(t) * H + ch]));
                }
            }
        float*      d_res  = dcopy(q, res);
        sycl::half* d_blk  = dcopy(q, blk);
        float*      d_injc = dcopy(q, injc);
        qwen4_hc_combine(q, d_res, d_blk, d_injc, T, H, HC).wait();
        std::vector<float> res_got(res.size());
        q.memcpy(res_got.data(), d_res, res_got.size() * sizeof(float)).wait();
        // Combine outputs are O(1..3) by construction (res U(-1,1) + w*b,
        // w in (0,2), b in (-1,1)).  Denominator floor 1.0 makes this a true
        // <=1e-5 relative-at-scale bound; elements that cancel to ~0 carry the
        // absolute uncertainty of their O(1) addends, so a tiny floor would
        // demand <=1e-7 ABS — beyond fp32 exp's 2-3 ulp guarantee and thus
        // unmeetable by ANY correct fp32 kernel.  The abs check below pins
        // those cancelled elements to fp32-exact class (~8 ulp at scale 1).
        check_tol(max_rel(res_got.data(), res_ref.data(), res_got.size(), 1.0), 1e-5,
                  "combine vs fp64 ref (rel at scale)");
        check_tol(max_abs(res_got.data(), res_ref.data(), res_got.size()), 5e-7,
                  "combine vs fp64 ref (abs)");

        // ---- degenerate: inj = 0 => res += block_out exactly (w == 1) ------
        std::vector<float> inj0(size_t(T) * HC, 0.f);
        float* d_inj0 = dcopy(q, inj0);
        q.memcpy(d_res, res.data(), res.size() * sizeof(float)).wait();  // reset
        qwen4_hc_combine(q, d_res, d_blk, d_inj0, T, H, HC).wait();
        q.memcpy(res_got.data(), d_res, res_got.size() * sizeof(float)).wait();
        {
            size_t bad = 0;
            for (uint32_t t = 0; t < T && bad == 0; ++t)
                for (uint32_t s = 0; s < HC && bad == 0; ++s)
                    for (uint32_t ch = 0; ch < H; ++ch) {
                        const size_t i = (size_t(t) * HC + s) * H + ch;
                        const float expect = res[i] + float(blk[size_t(t) * H + ch]);
                        if (res_got[i] != expect) { ++bad; break; }
                    }
            check(bad == 0, "combine inj=0: res + block_out bit-exact (w=1)");
        }

        // ---- expand_streams ------------------------------------------------
        std::vector<sycl::half> emb(size_t(T) * H);
        for (auto& v : emb) v = sycl::half(rf(-1.f, 1.f));
        sycl::half* d_emb  = dcopy(q, emb);
        float*      d_wide = sycl::malloc_device<float>(size_t(T) * HC * H, q);
        qwen4_hc_expand_streams(q, d_emb, d_wide, T, H, HC).wait();
        std::vector<float> wide(size_t(T) * HC * H);
        q.memcpy(wide.data(), d_wide, wide.size() * sizeof(float)).wait();
        {
            size_t bad = 0;
            for (uint32_t t = 0; t < T && bad == 0; ++t)
                for (uint32_t s = 0; s < HC && bad == 0; ++s)
                    for (uint32_t ch = 0; ch < H; ++ch)
                        if (wide[(size_t(t) * HC + s) * H + ch] !=
                            float(emb[size_t(t) * H + ch])) { ++bad; break; }
            check(bad == 0, "expand_streams: wide[t,s,:] == emb[t,:] exactly");
        }

        for (auto* p : {(void*)d_x, (void*)d_xn, (void*)d_lo, (void*)d_mixed,
                        (void*)d_inj, (void*)d_mixed2, (void*)d_res, (void*)d_blk,
                        (void*)d_injc, (void*)d_inj0, (void*)d_emb, (void*)d_wide})
            sycl::free(p, q);
    }

    // ---- degenerate: gamma = 1, identical streams (T = 7) ------------------
    // mixed[h] must equal xn0[h] * mean_s(gate[s*H+h]) where xn0 is the
    // SINGLE-stream norm — validates grouping + mean collapse independently.
    {
        std::printf("\n\033[1mdegenerate: gamma=1, all streams identical (T=7)\033[0m\n");
        const uint32_t T = 7;
        std::mt19937 rt(4242);
        auto rf = [&](float lo, float hi) {
            return std::uniform_real_distribution<float>(lo, hi)(rt);
        };
        std::vector<float> x0(size_t(T) * H), x(size_t(T) * D);
        for (auto& v : x0) v = rf(-1.f, 1.f);
        for (uint32_t t = 0; t < T; ++t)
            for (uint32_t s = 0; s < HC; ++s)
                for (uint32_t ch = 0; ch < H; ++ch)
                    x[(size_t(t) * HC + s) * H + ch] = x0[size_t(t) * H + ch];
        std::vector<float> gamma1(D, 1.f);

        // CPU sanity relation in fp64.
        std::vector<double> want(size_t(T) * H);
        std::vector<double> xn0(H), lo(R);
        for (uint32_t t = 0; t < T; ++t) {
            double ss = 0.0;
            for (uint32_t ch = 0; ch < H; ++ch) {
                const double v = double(x0[size_t(t) * H + ch]);
                ss += v * v;
            }
            const double inv = 1.0 / std::sqrt(ss / double(H) + double(EPS));
            for (uint32_t ch = 0; ch < H; ++ch)
                xn0[ch] = double(x0[size_t(t) * H + ch]) * inv;
            for (uint32_t i = 0; i < R; ++i) {
                double acc = 0.0;
                const sycl::half* wr = w_down.data() + size_t(i) * D;
                for (uint32_t s = 0; s < HC; ++s)
                    for (uint32_t ch = 0; ch < H; ++ch)
                        acc += double(float(wr[s * H + ch])) * xn0[ch];
                lo[i] = d_silu(acc / double(HC));
            }
            for (uint32_t ch = 0; ch < H; ++ch) {
                double mg = 0.0;
                for (uint32_t s = 0; s < HC; ++s) {
                    double g = 0.0;
                    const sycl::half* wr = w_up.data() + size_t(s * H + ch) * R;
                    for (uint32_t k = 0; k < R; ++k)
                        g += double(float(wr[k])) * lo[k];
                    mg += d_sigmoid(g);
                }
                want[size_t(t) * H + ch] = xn0[ch] * (mg / double(HC));
            }
        }

        float*      d_x     = dcopy(q, x);
        float*      d_g1    = dcopy(q, gamma1);
        float*      d_xn    = sycl::malloc_device<float>(size_t(T) * D, q);
        float*      d_lo    = sycl::malloc_device<float>(size_t(T) * R, q);
        sycl::half* d_mixed = sycl::malloc_device<sycl::half>(size_t(T) * H, q);
        float*      d_inj   = sycl::malloc_device<float>(size_t(T) * HC, q);
        qwen4_hc_mix(q, d_x, d_g1, d_wdown, d_wup, d_winj,
                     d_xn, d_lo, d_mixed, d_inj, T, H, HC, R, EPS).wait();
        std::vector<sycl::half> mixed(size_t(T) * H);
        q.memcpy(mixed.data(), d_mixed, mixed.size() * sizeof(sycl::half)).wait();
        check_tol(max_rel(mixed.data(), want.data(), mixed.size()), 2e-3,
                  "identical streams: mixed == xn0 * mean(gate)");
        for (auto* p : {(void*)d_x, (void*)d_g1, (void*)d_xn, (void*)d_lo,
                        (void*)d_mixed, (void*)d_inj})
            sycl::free(p, q);
    }

    sycl::free(d_gamma, q);
    sycl::free(d_wdown, q);
    sycl::free(d_wup, q);
    sycl::free(d_winj, q);

    std::printf("\n\033[1m%s\033[0m  (%d failure%s)\n\n",
                g_fail ? "\033[31mGATE FAILED\033[0m" : "\033[32mGATE PASSED\033[0m",
                g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
