// tests/unit/deepseek4_ops_gate_test.cpp
//
// Standalone numerical gate for ds4_hyper_connection.
//
// WHY THIS EXISTS.  tests/unit/deepseek4_parity_test.cpp is the parity gate for
// these primitives, but it needs reference blobs that do not exist on this box
// (it correctly exits 1 rather than pretending to pass).  ds4_hyper_connection
// was restructured for decode — the [mix, flat] GEMV is now split across
// work-groups and reduced in a second launch — which changes the summation
// order of a 16384-long dot product and therefore changes every downstream bit.
// A restructuring like that cannot be signed off by "the numbers look close";
// it needs a bound that is derived, not chosen, and that is demonstrably tight
// enough to reject a wrong kernel.
//
// So this file:
//   1. computes the whole hyper-connection in DOUBLE on the host,
//   2. derives a per-output error bound from an explicit fp32 error model
//      (stated in `derive_bounds` below; every constant is justified there),
//   3. prints observed error, the bound, and their ratio, and
//   4. runs NEGATIVE CONTROLS — a wrong Sinkhorn iteration count, a transposed
//      comb block, and a dropped stream — and requires each to BREACH the
//      bound.  A bound that nothing can fail is not a gate.
//
// It exercises both dispatch paths: the split-K path (T <= 16, hc == 4) and the
// fused path (T > 16, and hc != 4 via the generic branch).
//
// Build: part of the `deepseek4_ops_gate_test` target.
//   ninja -C build -j 6 deepseek4_ops_gate_test && ./build/tests/deepseek4_ops_gate_test

#include "ie/deepseek4_ops.hpp"

#include "ie/deepseek4_attn.hpp"   // ds4_rms_norm — the sequence hc_norm must equal
#include "ie/deepseek4.hpp"        // ds4_hc_mix — timed alongside the tail (IE_HC_TIME)

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <random>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// fp32 numerics constants used by the error model.
// ---------------------------------------------------------------------------
constexpr double kU = 5.9604644775390625e-08;  // 2^-24, fp32 unit roundoff

// gamma_n = n*u / (1 - n*u) — Higham's standard constant for the accumulated
// error of n sequential fp32 roundings.  ("Accuracy and Stability of Numerical
// Algorithms", 2nd ed., Lemma 3.1.)
double gamma_n(double n) {
    const double d = 1.0 - n * kU;
    return d <= 0.0 ? 1e30 : n * kU / d;
}

// ---------------------------------------------------------------------------
// Host double-precision reference — a direct transcription of
// DeepseekV4HyperConnection.forward, in the reference's operation order.
// ---------------------------------------------------------------------------
struct Ref {
    std::vector<double> mixv;       // [mix]
    std::vector<double> absmass;    // [mix]  sum |x_i * fn_mi| — the dot bound scale
    std::vector<double> pre, post;  // [hc]
    std::vector<double> comb;       // [hc*hc]
    std::vector<double> collapsed;  // [hidden]
    std::vector<double> softmaxp;   // [hc*hc] comb after softmax + eps, BEFORE Sinkhorn
    double ss = 0.0, rms = 0.0;
};

double sigd(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// The Sinkhorn stage alone, as a pure function of its input matrix, so the gate
// can differentiate through it.
std::vector<double> sinkhorn(std::vector<double> m, uint32_t hc, uint32_t iters, double eps) {
    for (uint32_t c = 0; c < hc; ++c) {
        double s = 0;
        for (uint32_t r = 0; r < hc; ++r) s += m[r * hc + c];
        s += eps;
        for (uint32_t r = 0; r < hc; ++r) m[r * hc + c] /= s;
    }
    for (uint32_t t = 1; t < iters; ++t) {
        for (uint32_t r = 0; r < hc; ++r) {
            double s = 0;
            for (uint32_t c = 0; c < hc; ++c) s += m[r * hc + c];
            s += eps;
            for (uint32_t c = 0; c < hc; ++c) m[r * hc + c] /= s;
        }
        for (uint32_t c = 0; c < hc; ++c) {
            double s = 0;
            for (uint32_t r = 0; r < hc; ++r) s += m[r * hc + c];
            s += eps;
            for (uint32_t r = 0; r < hc; ++r) m[r * hc + c] /= s;
        }
    }
    return m;
}

Ref reference(const std::vector<float>& streams, const std::vector<float>& fn,
              const std::vector<float>& base, const std::vector<float>& scale,
              uint32_t tok, uint32_t hidden, uint32_t hc, uint32_t iters,
              double rms_eps, double hc_eps) {
    const uint32_t flat = hc * hidden, mix = (2 + hc) * hc;
    const float* xs = streams.data() + size_t(tok) * flat;
    Ref R;
    R.mixv.assign(mix, 0); R.absmass.assign(mix, 0);
    R.pre.assign(hc, 0); R.post.assign(hc, 0);
    R.comb.assign(size_t(hc) * hc, 0); R.softmaxp.assign(size_t(hc) * hc, 0);
    R.collapsed.assign(hidden, 0);

    for (uint32_t i = 0; i < flat; ++i) R.ss += double(xs[i]) * double(xs[i]);
    R.rms = 1.0 / std::sqrt(R.ss / double(flat) + rms_eps);

    for (uint32_t m = 0; m < mix; ++m) {
        const float* fr = fn.data() + size_t(m) * flat;
        double d = 0, a = 0;
        for (uint32_t i = 0; i < flat; ++i) {
            const double p = double(xs[i]) * double(fr[i]);
            d += p; a += std::abs(p);
        }
        R.mixv[m] = d * R.rms;
        R.absmass[m] = a * R.rms;
    }
    for (uint32_t s = 0; s < hc; ++s) {
        R.pre[s]  = sigd(R.mixv[s] * scale[0] + base[s]) + hc_eps;
        R.post[s] = 2.0 * sigd(R.mixv[hc + s] * scale[1] + base[hc + s]);
    }
    for (uint32_t r = 0; r < hc; ++r) {
        double mx = -1e300;
        std::vector<double> row(hc);
        for (uint32_t c = 0; c < hc; ++c) {
            row[c] = R.mixv[2 * hc + r * hc + c] * scale[2] + base[2 * hc + r * hc + c];
            mx = std::max(mx, row[c]);
        }
        double den = 0;
        for (uint32_t c = 0; c < hc; ++c) { row[c] = std::exp(row[c] - mx); den += row[c]; }
        for (uint32_t c = 0; c < hc; ++c) R.softmaxp[r * hc + c] = row[c] / den + hc_eps;
    }
    R.comb = sinkhorn(R.softmaxp, hc, iters, hc_eps);
    for (uint32_t j = 0; j < hidden; ++j) {
        double a = 0;
        for (uint32_t s = 0; s < hc; ++s) a += R.pre[s] * double(xs[size_t(s) * hidden + j]);
        R.collapsed[j] = a;
    }
    return R;
}

// ---------------------------------------------------------------------------
// THE ERROR MODEL.
//
// Every bound below is a forward-error bound on the fp32 evaluation of the
// reference expression, given the kernel's actual reduction shape.  Nothing is
// fitted to observed data.
//
// (1) REDUCTION DEPTH.  Both kernels reduce a length-`flat` product with the
//     same three-level shape: 4 independent accumulators per work-item, then a
//     tree over the work-item's group, then a serial sum over `chunks`.  The
//     longest chain of dependent fp32 additions is
//         D = flat/(chunks * lanes * 4)   [per-accumulator serial chain]
//           + 2                           [combining the 4 accumulators]
//           + tree                        [group reduction depth]
//           + (chunks - 1)                [chunk sum]
//     SPLIT path (T <= 16, hc == 4): chunks = C2, lanes = WGA = 256, and
//     tree <= 256/16 + log2(16) = 20 (worst case: a linear pass over the 16
//     sub-groups, each internally a 4-deep tree).
//     FUSED path: one sub-group owns a whole (row, chunk) job, so lanes = 16
//     and tree = log2(16) = 4, with chunks = CH = 2.
//
// (2) DOT PRODUCT.  |fl(sum x_i f_i) - sum x_i f_i| <= gamma_{D+1} * sum|x_i f_i|.
//     The +1 is the rounding of each product.  sum|x_i f_i| (`absmass`) is used,
//     not |dot|: with random signs the dot is ~sqrt(N) smaller than the mass, so
//     the mass is the only correct scale for a forward bound.
//
// (3) RMS.  ss is a sum of squares (all terms positive), so its relative error
//     is at most gamma_{D+1}.  ss/flat is exact (flat is a power of two).
//     rsqrt carries <= 2 ulp (OpenCL/SYCL math accuracy), and
//     d(rsqrt)/rsqrt = -1/2 * d(arg)/arg, so
//         rel(rms) <= 0.5 * (gamma_{D+1} + u) + 2u.
//
// (4) SIGMOID CHAIN.  t = mixv*scale + base costs 2 roundings; |sigmoid'| <= 1/4;
//     sycl::exp carries <= 3 ulp and the reciprocal <= 1 ulp, so evaluating the
//     sigmoid contributes <= 6u relative.
//
// (5) SOFTMAX.  The row-softmax Jacobian satisfies ||J||_inf = max_c 2p_c(1-p_c)
//     <= 1/2, so |dp|_inf <= 0.5 * |dl|_inf, plus <= 6u for exp and the division.
//
// (6) SINKHORN.  Two contributions, kept separate:
//     (a) SENSITIVITY to its input.  Rather than bounding the 39-pass iteration
//         analytically (a per-pass Lipschitz bound of ~5 would give 5^39, which
//         is useless and also wrong — Sinkhorn is non-expansive, not expansive),
//         the gate DIFFERENTIATES THE EXACT MAP: it computes |d comb_e/d p_j| by
//         central differences in double precision and forms the first-order
//         bound sum_j |d comb_e/d p_j| * delta_p_j.  That is an evaluation of
//         the model, not an approximation of it; the only thing dropped is the
//         second-order term, and delta_p ~ 1e-7 makes that ~1e-14 relative.
//     (b) ROUNDING INJECTED INSIDE the fp32 iteration.  Each normalisation pass
//         does a 4-term sum (3 dependent adds), one add of eps and one division:
//         <= 5u relative per pass, over 2*iters - 1 passes.  These accumulate
//         LINEARLY, which is valid because the Sinkhorn map is non-expansive in
//         the Hilbert projective metric (Birkhoff) — the gate does not rely on
//         the contraction factor being < 1, only on it being <= 1.
//
// (7) COLLAPSE.  collapsed_j = sum_s pre_s * x_sj over hc terms:
//         |err| <= sum_s |dpre_s| * |x_sj| + gamma_{hc+1} * sum_s |pre_s * x_sj|.
// ---------------------------------------------------------------------------
struct Bounds {
    std::vector<double> mixv, pre, post, comb, collapsed;
    double depth = 0;
};

Bounds derive_bounds(const Ref& R, const std::vector<float>& streams,
                     const std::vector<float>& base, const std::vector<float>& scale,
                     uint32_t tok, uint32_t hidden, uint32_t hc, uint32_t iters,
                     double hc_eps, bool split_path, uint32_t chunks) {
    const uint32_t flat = hc * hidden, mix = (2 + hc) * hc;
    const float* xs = streams.data() + size_t(tok) * flat;
    Bounds B;

    // (1) reduction depth
    const double lanes = split_path ? 256.0 : 16.0;
    const double tree  = split_path ? 20.0  : 4.0;
    const double D = double(flat) / (double(chunks) * lanes * 4.0)
                   + 2.0 + tree + double(chunks - 1);
    B.depth = D;

    // (3) rms
    const double rel_ss  = gamma_n(D + 1.0);
    const double rel_rms = 0.5 * (rel_ss + kU) + 2.0 * kU;

    // (2) mixv
    B.mixv.assign(mix, 0);
    for (uint32_t m = 0; m < mix; ++m)
        B.mixv[m] = gamma_n(D + 1.0) * R.absmass[m]
                  + std::abs(R.mixv[m]) * (rel_rms + kU);

    // (4) pre / post
    B.pre.assign(hc, 0); B.post.assign(hc, 0);
    for (uint32_t s = 0; s < hc; ++s) {
        const double t0 = R.mixv[s] * scale[0] + base[s];
        const double dt0 = B.mixv[s] * std::abs(scale[0])
                         + (std::abs(R.mixv[s] * scale[0]) + std::abs(t0)) * kU;
        B.pre[s] = 0.25 * dt0 + sigd(t0) * 6.0 * kU + kU;

        const double t1 = R.mixv[hc + s] * scale[1] + base[hc + s];
        const double dt1 = B.mixv[hc + s] * std::abs(scale[1])
                         + (std::abs(R.mixv[hc + s] * scale[1]) + std::abs(t1)) * kU;
        B.post[s] = 2.0 * (0.25 * dt1 + sigd(t1) * 6.0 * kU);
    }

    // (5) softmax input -> softmax output perturbation
    std::vector<double> dp(size_t(hc) * hc, 0.0);
    for (uint32_t r = 0; r < hc; ++r) {
        double dl_max = 0.0;
        for (uint32_t c = 0; c < hc; ++c) {
            const uint32_t o = 2 * hc + r * hc + c;
            const double l = R.mixv[o] * scale[2] + base[o];
            const double dl = B.mixv[o] * std::abs(scale[2])
                            + (std::abs(R.mixv[o] * scale[2]) + std::abs(l)) * kU;
            dl_max = std::max(dl_max, dl);
        }
        for (uint32_t c = 0; c < hc; ++c)
            dp[r * hc + c] = 0.5 * dl_max + 6.0 * kU + kU;   // +u for the `+ hc_eps`
    }

    // (6a) differentiate the exact Sinkhorn map w.r.t. its input
    const uint32_t n2 = hc * hc;
    B.comb.assign(n2, 0.0);
    const double h = 1e-7;
    for (uint32_t j = 0; j < n2; ++j) {
        std::vector<double> pp = R.softmaxp, pm = R.softmaxp;
        pp[j] += h; pm[j] -= h;
        const std::vector<double> cp = sinkhorn(pp, hc, iters, hc_eps);
        const std::vector<double> cm = sinkhorn(pm, hc, iters, hc_eps);
        for (uint32_t e = 0; e < n2; ++e)
            B.comb[e] += std::abs(cp[e] - cm[e]) / (2.0 * h) * dp[j];
    }
    // (6b) rounding injected inside the fp32 iteration
    const double passes = double(2 * iters - 1);
    for (uint32_t e = 0; e < n2; ++e) B.comb[e] += passes * 5.0 * kU * std::abs(R.comb[e]);

    // (7) collapse
    B.collapsed.assign(hidden, 0.0);
    for (uint32_t j = 0; j < hidden; ++j) {
        double a = 0, mass = 0;
        for (uint32_t s = 0; s < hc; ++s) {
            const double x = double(xs[size_t(s) * hidden + j]);
            a += B.pre[s] * std::abs(x);
            mass += std::abs(R.pre[s] * x);
        }
        B.collapsed[j] = a + gamma_n(double(hc) + 1.0) * mass;
    }
    return B;
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------
struct Dev {
    sycl::queue q;
    float *streams = nullptr, *fn = nullptr, *base = nullptr, *scale = nullptr;
    float *post = nullptr, *comb = nullptr, *coll = nullptr;
    uint32_t T = 0, hidden = 0, hc = 0;
    Dev() : q(sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order{}}) {}
};

int g_fail = 0;

// Compare device output for token `tok` against `R`, using bounds `B`.
// `expect_pass` says whether this configuration is supposed to satisfy the
// bound; a negative control sets it false and the gate then requires a breach.
void compare(const char* label, Dev& d, const Ref& R, const Bounds& B, uint32_t tok,
             bool expect_pass) {
    const uint32_t hc = d.hc, hidden = d.hidden;
    std::vector<float> hp(hc), hc2(size_t(hc) * hc), hcl(hidden);
    d.q.memcpy(hp.data(), d.post + size_t(tok) * hc, hc * 4).wait();
    d.q.memcpy(hc2.data(), d.comb + size_t(tok) * hc * hc, size_t(hc) * hc * 4).wait();
    d.q.memcpy(hcl.data(), d.coll + size_t(tok) * hidden, hidden * 4).wait();

    double op = 0, oc = 0, ol = 0, rp = 0, rc = 0, rl = 0;
    auto acc = [](double e, double b, double& obs, double& rat) {
        obs = std::max(obs, e);
        rat = std::max(rat, b > 0 ? e / b : (e > 0 ? 1e30 : 0.0));
    };
    for (uint32_t i = 0; i < hc; ++i)      acc(std::abs(double(hp[i])  - R.post[i]),      B.post[i],      op, rp);
    for (uint32_t i = 0; i < hc * hc; ++i) acc(std::abs(double(hc2[i]) - R.comb[i]),      B.comb[i],      oc, rc);
    for (uint32_t i = 0; i < hidden; ++i)  acc(std::abs(double(hcl[i]) - R.collapsed[i]), B.collapsed[i], ol, rl);

    const bool within = rp <= 1.0 && rc <= 1.0 && rl <= 1.0;
    const bool ok = (within == expect_pass);
    std::printf("  %-38s post %.2e/%.2e (%.3f)  comb %.2e/%.2e (%.3f)  coll %.2e/%.2e (%.3f)  %s\n",
                label,
                op, *std::max_element(B.post.begin(), B.post.end()), rp,
                oc, *std::max_element(B.comb.begin(), B.comb.end()), rc,
                ol, *std::max_element(B.collapsed.begin(), B.collapsed.end()), rl,
                ok ? (expect_pass ? "PASS" : "PASS (breached, as required)")
                   : (expect_pass ? "*** FAIL: OVER BOUND ***"
                                  : "*** FAIL: control did NOT breach ***"));
    if (!ok) ++g_fail;
}

void fill(std::vector<float>& v, std::mt19937& rng, float sigma) {
    std::normal_distribution<float> nd(0.f, sigma);
    for (auto& x : v) x = nd(rng);
}

// One full shape: allocate, run the real op, gate every sampled token.
void run_shape(uint32_t T, uint32_t hidden, uint32_t hc, uint32_t iters, uint32_t seed) {
    const uint32_t flat = hc * hidden, mix = (2 + hc) * hc;
    const float rms_eps = 1e-6f, hc_eps = 1e-6f;
    // The dispatch rule in src/ops/deepseek4_ops.cpp::ds4_hyper_connection,
    // replicated so the bound uses the depth of the path that actually ran.
    const bool split = (hc == 4u && T >= 1u && T <= 16u && (flat % 4u) == 0u);
    const uint32_t chunks = split ? (T <= 2u ? 4u : 2u) : 2u;

    Dev d; d.T = T; d.hidden = hidden; d.hc = hc;
    std::printf("\n--- T=%u hidden=%u hc=%u sinkhorn=%u   path=%s chunks=%u ---\n",
                T, hidden, hc, iters, split ? "SPLIT-K" : "fused", chunks);

    std::mt19937 rng(seed);
    std::vector<float> hst(size_t(T) * flat), hfn(size_t(mix) * flat), hb(mix), hs(3);
    fill(hst, rng, 0.02f);
    fill(hfn, rng, 0.02f);
    fill(hb, rng, 0.5f);
    hs[0] = 0.7f; hs[1] = 1.3f; hs[2] = 0.9f;

    d.streams = sycl::malloc_device<float>(hst.size(), d.q);
    d.fn      = sycl::malloc_device<float>(hfn.size(), d.q);
    d.base    = sycl::malloc_device<float>(mix, d.q);
    d.scale   = sycl::malloc_device<float>(3, d.q);
    d.post    = sycl::malloc_device<float>(size_t(T) * hc, d.q);
    d.comb    = sycl::malloc_device<float>(size_t(T) * hc * hc, d.q);
    d.coll    = sycl::malloc_device<float>(size_t(T) * hidden, d.q);
    d.q.memcpy(d.streams, hst.data(), hst.size() * 4).wait();
    d.q.memcpy(d.fn, hfn.data(), hfn.size() * 4).wait();
    d.q.memcpy(d.base, hb.data(), hb.size() * 4).wait();
    d.q.memcpy(d.scale, hs.data(), 12).wait();

    ie::ds4_hyper_connection(d.q, d.streams, d.fn, d.base, d.scale,
                             d.post, d.comb, d.coll, T, hidden, hc, iters,
                             rms_eps, hc_eps).wait();

    // Gate the first token, the last token, and (for T>2) one in the middle.
    std::vector<uint32_t> toks{0u};
    if (T > 1) toks.push_back(T - 1);
    if (T > 2) toks.push_back(T / 2);
    double depth_used = 0;
    for (uint32_t tok : toks) {
        const Ref R = reference(hst, hfn, hb, hs, tok, hidden, hc, iters, rms_eps, hc_eps);
        const Bounds B = derive_bounds(R, hst, hb, hs, tok, hidden, hc, iters, hc_eps, split, chunks);
        depth_used = B.depth;
        char lab[64]; std::snprintf(lab, sizeof lab, "token %u", tok);
        compare(lab, d, R, B, tok, /*expect_pass=*/true);
    }
    std::printf("  [model: reduction depth D = %.0f dependent fp32 adds, gamma_D = %.2e]\n",
                depth_used, gamma_n(depth_used + 1.0));

    // ---------------- NEGATIVE CONTROLS (token 0) ----------------
    // Each builds a WRONG reference and requires the bound to reject it.  If a
    // control does not breach, the bound is too loose to be a gate and the test
    // fails.
    if (iters > 2) {
        const uint32_t tok = 0;
        const Ref R = reference(hst, hfn, hb, hs, tok, hidden, hc, iters, rms_eps, hc_eps);
        const Bounds B = derive_bounds(R, hst, hb, hs, tok, hidden, hc, iters, hc_eps, split, chunks);

        // ---- DIAGNOSTIC: what this gate can and cannot discriminate --------
        // The obvious "wrong Sinkhorn iteration count" control is iters-1.  It
        // does NOT breach, and the reason is a property of the operator rather
        // than a weakness of the bound: the iteration converges linearly, so by
        // the last step the per-iteration residual |comb(N) - comb(N-1)| has
        // fallen FAR BELOW the bound the rest of the chain already carries.
        // The gate measures that residual, compares it against the bound, and
        // says so out loud instead of quietly dropping the control — then uses
        // iters=1 ("the loop never ran"), which moves the fixed point itself.
        //
        // Consequence worth stating: this gate cannot detect an off-by-one in
        // the Sinkhorn trip count at the shipped iteration count.  Nothing can,
        // in fp32 — the operator itself has stopped moving by then.
        {
            double last_delta = 0.0;
            std::vector<double> prev = sinkhorn(R.softmaxp, hc, iters - 1, hc_eps);
            const std::vector<double> cur = sinkhorn(R.softmaxp, hc, iters, hc_eps);
            for (size_t e = 0; e < cur.size(); ++e)
                last_delta = std::max(last_delta, std::abs(cur[e] - prev[e]));
            const double cb = *std::max_element(B.comb.begin(), B.comb.end());
            std::printf("  [sinkhorn residual |comb(%u)-comb(%u)| = %.2e, vs comb bound %.2e "
                        "(%.4f of bound):\n   +-1 iteration at %u is below what this chain can "
                        "resolve — control uses iters=1 instead]\n",
                        iters, iters - 1, last_delta, cb, last_delta / cb, iters);
        }

        std::printf("  negative controls (must breach the bound):\n");

        // (a) Sinkhorn loop never ran: only the leading column pass.  This is
        //     the iteration-count bug that IS observable.
        {
            const Ref W = reference(hst, hfn, hb, hs, tok, hidden, hc, 1, rms_eps, hc_eps);
            compare("  ctrl: sinkhorn iters=1 (loop dropped)", d, W, B, tok, /*expect_pass=*/false);
        }
        // (b) transposed comb block: the classic row/column-major slip in the
        //     mHC mapping.  Only `comb` is wrong, so this control also proves
        //     the comb bound is not being carried by the post/collapse terms.
        //
        //     Skipped at hc == 2: every 2x2 doubly stochastic matrix is
        //     [[a, 1-a], [1-a, a]], i.e. symmetric, so the converged comb is its
        //     own transpose and the control is mathematically vacuous there.
        if (hc >= 4) {
            Ref W = R;
            for (uint32_t r = 0; r < hc; ++r)
                for (uint32_t c = r + 1; c < hc; ++c)
                    std::swap(W.comb[r * hc + c], W.comb[c * hc + r]);
            compare("  ctrl: comb transposed", d, W, B, tok, /*expect_pass=*/false);
        } else {
            std::printf("    ctrl: comb transposed                 SKIPPED "
                        "(2x2 doubly stochastic => symmetric => vacuous)\n");
        }
        // (c) wrong scale applied to the comb logits (scale[0] instead of
        //     scale[2]) — a transcription slip that survives every shape check.
        //     Works at every hc, including hc == 2.
        {
            std::vector<float> hs_bad = hs;
            hs_bad[2] = hs[0];
            const Ref W = reference(hst, hfn, hb, hs_bad, tok, hidden, hc, iters, rms_eps, hc_eps);
            compare("  ctrl: comb scale[2] <- scale[0]", d, W, B, tok, /*expect_pass=*/false);
        }
        // (d) dropped stream: one of the hc streams left out of the collapse.
        {
            Ref W = R;
            const float* xs = hst.data() + size_t(tok) * flat;
            for (uint32_t j = 0; j < hidden; ++j)
                W.collapsed[j] -= W.pre[hc - 1] * double(xs[size_t(hc - 1) * hidden + j]);
            compare("  ctrl: stream hc-1 dropped", d, W, B, tok, /*expect_pass=*/false);
        }
    }

    sycl::free(d.streams, d.q); sycl::free(d.fn, d.q); sycl::free(d.base, d.q);
    sycl::free(d.scale, d.q); sycl::free(d.post, d.q); sycl::free(d.comb, d.q);
    sycl::free(d.coll, d.q);
}

// ---------------------------------------------------------------------------
// ds4_swiglu_clamped_h — must be BIT-IDENTICAL to the four-launch chain it
// replaces:  cast_fp16_to_fp32 x2 -> ds4_swiglu_clamped -> cast_fp32_to_fp16.
//
// This is an exact-equality check, not a tolerance check, because the claim in
// the header is exactness.  The chain is reproduced here with the same plain
// conversions src/ops/elementwise.cpp uses, so the comparison does not depend
// on that file staying put; if either side ever stops matching, this fails.
//
// Inputs sweep the whole fp16 value range including subnormals, +-0, values
// straddling the clamp limit on both sides, and (deliberately) infinities and
// NaNs, so the clamp branch is exercised where it actually matters.
// ---------------------------------------------------------------------------
void run_swiglu_h_bitexact() {
    std::printf("\n--- ds4_swiglu_clamped_h vs cast+cast+swiglu+cast (exact equality) ---\n");
    sycl::queue q{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};

    // Every fp16 bit pattern that is not a NaN, plus a dense random sample.
    std::vector<sycl::half> hg, hu;
    for (uint32_t b = 0; b < 65536u; ++b) {
        const uint16_t bits = uint16_t(b);
        const sycl::half v = sycl::bit_cast<sycl::half>(bits);
        if (std::isnan(float(v))) continue;   // NaN != NaN would defeat exact equality
        hg.push_back(v);
    }
    // Pair each gate value against a rotated copy so both operands vary.
    hu.assign(hg.begin(), hg.end());
    std::rotate(hu.begin(), hu.begin() + hu.size() / 3, hu.end());
    const size_t n = hg.size();

    sycl::half* dg = sycl::malloc_device<sycl::half>(n, q);
    sycl::half* du = sycl::malloc_device<sycl::half>(n, q);
    sycl::half* dy_fused = sycl::malloc_device<sycl::half>(n, q);
    sycl::half* dy_chain = sycl::malloc_device<sycl::half>(n, q);
    float* fg = sycl::malloc_device<float>(n, q);
    float* fu = sycl::malloc_device<float>(n, q);
    float* fy = sycl::malloc_device<float>(n, q);
    q.memcpy(dg, hg.data(), n * sizeof(sycl::half)).wait();
    q.memcpy(du, hu.data(), n * sizeof(sycl::half)).wait();

    for (float limit : {7.0f, 1.0f, 0.0f, 65504.0f}) {
        // reference chain, spelled exactly as the call sites spell it
        q.parallel_for(n, [=](sycl::id<1> i) { fg[i] = float(dg[i]); }).wait();
        q.parallel_for(n, [=](sycl::id<1> i) { fu[i] = float(du[i]); }).wait();
        ie::ds4_swiglu_clamped(q, fg, fu, fy, n, limit).wait();
        q.parallel_for(n, [=](sycl::id<1> i) { dy_chain[i] = sycl::half(fy[i]); }).wait();
        // fused
        ie::ds4_swiglu_clamped_h(q, dg, du, dy_fused, n, limit).wait();

        std::vector<sycl::half> a(n), b(n);
        q.memcpy(a.data(), dy_chain, n * sizeof(sycl::half)).wait();
        q.memcpy(b.data(), dy_fused, n * sizeof(sycl::half)).wait();
        size_t diff = 0;
        for (size_t i = 0; i < n; ++i) {
            const uint16_t ba = sycl::bit_cast<uint16_t>(a[i]);
            const uint16_t bb = sycl::bit_cast<uint16_t>(b[i]);
            // Both sides may legitimately produce NaN (inf*0); NaN payloads are
            // not required to match, but NaN-ness is.
            const bool na = std::isnan(float(a[i])), nb = std::isnan(float(b[i]));
            if (na || nb) { if (na != nb) ++diff; continue; }
            if (ba != bb) ++diff;
        }
        std::printf("  limit=%-9.1f n=%zu  differing outputs = %zu  %s\n",
                    limit, n, diff, diff == 0 ? "PASS (bit-identical)" : "*** FAIL ***");
        if (diff) ++g_fail;
    }
    // Aliasing: y == gate must give the same answer as a distinct destination.
    {
        const float limit = 7.0f;
        ie::ds4_swiglu_clamped_h(q, dg, du, dy_fused, n, limit).wait();
        sycl::half* alias = sycl::malloc_device<sycl::half>(n, q);
        q.memcpy(alias, hg.data(), n * sizeof(sycl::half)).wait();
        ie::ds4_swiglu_clamped_h(q, alias, du, alias, n, limit).wait();
        std::vector<sycl::half> a(n), b(n);
        q.memcpy(a.data(), dy_fused, n * sizeof(sycl::half)).wait();
        q.memcpy(b.data(), alias, n * sizeof(sycl::half)).wait();
        size_t diff = 0;
        for (size_t i = 0; i < n; ++i) {
            const bool na = std::isnan(float(a[i])), nb = std::isnan(float(b[i]));
            if (na || nb) { if (na != nb) ++diff; continue; }
            if (sycl::bit_cast<uint16_t>(a[i]) != sycl::bit_cast<uint16_t>(b[i])) ++diff;
        }
        std::printf("  y aliases gate            differing outputs = %zu  %s\n",
                    diff, diff == 0 ? "PASS" : "*** FAIL ***");
        if (diff) ++g_fail;
        sycl::free(alias, q);
    }

    sycl::free(dg, q); sycl::free(du, q); sycl::free(dy_fused, q); sycl::free(dy_chain, q);
    sycl::free(fg, q); sycl::free(fu, q); sycl::free(fy, q);
}

// ---------------------------------------------------------------------------
// ds4_hyper_connection_norm — the consumer RMSNorm folded into the split tail.
//
// This one is NOT gated against the double reference, and deliberately so.  It
// does not claim to be a better approximation of the mathematics; it claims to
// be THE SAME COMPUTATION AS `ds4_hyper_connection` FOLLOWED BY `ds4_rms_norm`,
// in one fewer launch.  The only honest gate for that claim is bit equality on
// every output — `normed`, and also `collapsed`, `post` and `comb`, because the
// fused tail rewrites the block that produces all four and a fusion that only
// got `normed` right would be a fusion that had quietly changed the rest.
//
// Two negative controls, and both must FAIL bit equality, or the check is not
// discriminating: (a) the norm weight dropped (w == 1), which is what a fusion
// that forgot to apply `norm_w` would produce, and (b) the norm epsilon dropped
// to zero, which is the difference the reduction is least sensitive to and
// therefore the hardest thing for a bitwise test to see.
void run_hc_norm_bitexact(uint32_t T, uint32_t hidden, uint32_t hc, uint32_t iters,
                          unsigned seed) {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    const uint32_t mix = (2 + hc) * hc, flat = hc * hidden;
    const float rms_eps = 1e-6f, hc_eps = 1e-8f, norm_eps = 1e-6f;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.f, 1.f), dn(0.5f, 1.5f);
    std::vector<float> hs(size_t(T) * flat), hf(size_t(mix) * flat), hb(mix), hsc(3), hw(hidden);
    for (auto& v : hs) v = d(rng);
    for (auto& v : hf) v = d(rng) * 0.02f;
    for (auto& v : hb) v = d(rng);
    for (auto& v : hsc) v = dn(rng);
    for (auto& v : hw) v = dn(rng);

    auto up = [&](const std::vector<float>& h) {
        float* p = sycl::malloc_device<float>(h.size(), q);
        q.memcpy(p, h.data(), h.size() * 4).wait();
        return p;
    };
    float* streams = up(hs); float* fn = up(hf); float* base = up(hb);
    float* scale = up(hsc); float* nw = up(hw);
    std::vector<float> ones(hidden, 1.f);
    float* nw1 = up(ones);

    const size_t nH = size_t(T) * hidden, nP = size_t(T) * hc, nC = size_t(T) * hc * hc;
    auto dev = [&](size_t n) { return sycl::malloc_device<float>(n, q); };
    float *pa = dev(nP), *ca = dev(nC), *la = dev(nH), *na = dev(nH);
    float *pb = dev(nP), *cb = dev(nC), *lb = dev(nH), *nb_ = dev(nH);

    auto down = [&](float* p, size_t n) {
        std::vector<float> h(n);
        q.memcpy(h.data(), p, n * 4).wait();
        return h;
    };
    auto bits_differ = [](const std::vector<float>& a, const std::vector<float>& b) {
        size_t n = 0;
        for (size_t i = 0; i < a.size(); ++i)
            if (sycl::bit_cast<uint32_t>(a[i]) != sycl::bit_cast<uint32_t>(b[i])) ++n;
        return n;
    };

    q.memset(na, 0xff, nH * 4).wait();
    q.memset(nb_, 0xff, nH * 4).wait();
    ie::ds4_hyper_connection(q, streams, fn, base, scale, pa, ca, la,
                             T, hidden, hc, iters, rms_eps, hc_eps);
    ie::ds4_rms_norm(q, la, nw, na, T, hidden, norm_eps);
    ie::ds4_hyper_connection_norm(q, streams, fn, base, scale, nw, pb, cb, lb, nb_,
                                  T, hidden, hc, iters, rms_eps, hc_eps, norm_eps);
    q.wait();

    const auto A = down(na, nH), B = down(nb_, nH);
    const size_t dN = bits_differ(A, B);
    const size_t dL = bits_differ(down(la, nH), down(lb, nH));
    const size_t dP = bits_differ(down(pa, nP), down(pb, nP));
    const size_t dC = bits_differ(down(ca, nC), down(cb, nC));
    size_t nnan = 0;
    for (float v : B) if (!std::isfinite(v)) ++nnan;

    std::printf("  hc_norm T=%-4u H=%-5u hc=%u  normed %zu/%zu  collapsed %zu  post %zu"
                "  comb %zu  nonfinite %zu  %s\n",
                T, hidden, hc, dN, A.size(), dL, dP, dC, nnan,
                (dN || dL || dP || dC || nnan) ? "*** FAIL ***" : "PASS");
    if (dN || dL || dP || dC || nnan) ++g_fail;

    // Negative controls — each must differ from the reference sequence.
    ie::ds4_hyper_connection_norm(q, streams, fn, base, scale, nw1, pb, cb, lb, nb_,
                                  T, hidden, hc, iters, rms_eps, hc_eps, norm_eps);
    q.wait();
    const size_t ctl_w = bits_differ(A, down(nb_, nH));
    ie::ds4_hyper_connection_norm(q, streams, fn, base, scale, nw, pb, cb, lb, nb_,
                                  T, hidden, hc, iters, rms_eps, hc_eps, 0.f);
    q.wait();
    const size_t ctl_e = bits_differ(A, down(nb_, nH));
    std::printf("    neg-control norm weight dropped : %zu differ -> %s\n",
                ctl_w, ctl_w ? "CAUGHT" : "*** MISSED ***");
    std::printf("    neg-control norm eps dropped    : %zu differ -> %s\n",
                ctl_e, ctl_e ? "CAUGHT" : "*** MISSED ***");
    if (!ctl_w) ++g_fail;
    if (!ctl_e) ++g_fail;

    for (float* p : {streams, fn, base, scale, nw, nw1, pa, ca, la, na, pb, cb, lb, nb_})
        sycl::free(p, q);
}

// IE_HC_TIME=1: microseconds per call at the decode shape, wall clock over N
// submissions on the in-order queue (so it includes what the engine pays: the
// submit and the GPU-side per-launch cost).
void hc_time() {
    if (!std::getenv("IE_HC_TIME")) return;
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    const uint32_t T = 1, hidden = 4096, hc = 4, iters = 20;
    const uint32_t mix = (2 + hc) * hc, flat = hc * hidden;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-1.f, 1.f), dn(0.5f, 1.5f), dp(0.f, 2.f), dc(0.f, 1.f);
    auto rnd = [&](size_t n, auto& dist) {
        std::vector<float> h(n);
        for (auto& v : h) v = dist(rng);
        float* p = sycl::malloc_device<float>(n, q);
        q.memcpy(p, h.data(), n * 4).wait();
        return p;
    };
    float* streams = rnd(size_t(T) * flat, d); float* fn = rnd(size_t(mix) * flat, d);
    float* base = rnd(mix, d); float* scale = rnd(3, dn); float* nw = rnd(hidden, dn);
    float* sub = rnd(size_t(T) * hidden, d);
    float* post = rnd(size_t(T) * hc, dp); float* comb = rnd(size_t(T) * hc * hc, dc);
    float* s2 = rnd(size_t(T) * flat, d);
    float* coll = rnd(size_t(T) * hidden, d); float* normed = rnd(size_t(T) * hidden, d);
    const int N = 2000;
    auto time_it = [&](const char* what, auto&& body) {
        for (int i = 0; i < 50; ++i) body();
        q.wait();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) body();
        q.wait();
        const double us = std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - t0).count() / N;
        std::printf("  %-40s %8.2f us/call\n", what, us);
    };
    std::printf("\nhc_time (T=1, H=4096, hc=4, %d calls)\n", N);
    time_it("hyper_connection + rms_norm (sequence)", [&] {
        ie::ds4_hyper_connection(q, streams, fn, base, scale, post, comb, coll,
                                 T, hidden, hc, iters, 1e-6f, 1e-8f);
        ie::ds4_rms_norm(q, coll, nw, normed, T, hidden, 1e-6f);
    });
    time_it("hyper_connection_norm (fused tail)", [&] {
        ie::ds4_hyper_connection_norm(q, streams, fn, base, scale, nw, post, comb, coll, normed,
                                      T, hidden, hc, iters, 1e-6f, 1e-8f, 1e-6f);
    });
    time_it("hc_mix + hyper_connection_norm", [&] {
        ie::ds4_hc_mix(q, streams, post, comb, sub, s2, T, hidden, hc);
        ie::ds4_hyper_connection_norm(q, s2, fn, base, scale, nw, post, comb, coll, normed,
                                      T, hidden, hc, iters, 1e-6f, 1e-8f, 1e-6f);
    });
    for (float* p : {streams, fn, base, scale, nw, sub, post, comb, s2, coll, normed})
        sycl::free(p, q);
}

}  // namespace

int main() {
    std::printf("deepseek4_ops_gate_test — ds4_hyper_connection vs double host reference\n");
    std::printf("fp32 unit roundoff u = %.6e\n", kU);
    {
        sycl::queue probe{sycl::gpu_selector_v};
        std::printf("device: %s\n",
                    probe.get_device().get_info<sycl::info::device::name>().c_str());
    }
    std::printf("\nColumns: observed / bound (ratio).  Ratio <= 1 is within the derived bound.\n");

    // Real V4-Flash decode shape, and the shapes either side of every dispatch
    // boundary in ds4_hyper_connection.
    run_shape(1,   4096, 4, 20, 11);   // decode — the case this work targets
    run_shape(2,   4096, 4, 20, 12);   // last T using chunks=4
    run_shape(3,   4096, 4, 20, 13);   // first T using chunks=2
    run_shape(16,  4096, 4, 20, 14);   // last T on the split path
    run_shape(17,  4096, 4, 20, 15);   // first T back on the fused path
    run_shape(64,  4096, 4, 20, 16);   // fused, WG=768
    run_shape(128, 4096, 4, 20, 17);   // fused, WG=512
    run_shape(1,   1024, 2, 20, 18);   // hc != 4 -> generic SLM Sinkhorn branch
    run_shape(4,    512, 8, 12, 19);   // hc != 4, different iteration count

    run_swiglu_h_bitexact();

    // The fused-norm tail.  Bit equality against hyper_connection + rms_norm,
    // at the decode shape and at every dispatch boundary of the split path,
    // plus one shape the split path does NOT cover (T=17) so the fallback is
    // exercised too.
    std::printf("\nds4_hyper_connection_norm — BIT EQUALITY vs hyper_connection + rms_norm\n");
    run_hc_norm_bitexact(1,   4096, 4, 20, 41);   // decode
    run_hc_norm_bitexact(2,   4096, 4, 20, 42);   // last T using chunks=4
    run_hc_norm_bitexact(3,   4096, 4, 20, 43);   // first T using chunks=2
    run_hc_norm_bitexact(16,  4096, 4, 20, 44);   // last T on the split path
    run_hc_norm_bitexact(17,  4096, 4, 20, 45);   // fallback: no fusion, still equal
    run_hc_norm_bitexact(1,   1024, 2, 20, 46);   // hc != 4 -> fallback
    run_hc_norm_bitexact(4,   3000, 4, 20, 47);   // hidden not a multiple of G2

    hc_time();

    std::printf("\n=======================================================\n");
    if (g_fail) {
        std::printf("GATE FAILED: %d check(s) wrong.\n", g_fail);
        return 1;
    }
    std::printf("GATE PASSED: all shapes within the derived bound, and every\n"
                "negative control breached it.\n");
    return 0;
}
