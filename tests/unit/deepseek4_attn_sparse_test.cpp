// tests/unit/deepseek4_attn_sparse_test.cpp — gate for the mask-first early skip
// in ds4_attention (src/ops/deepseek4_attn.cpp).
//
// WHAT CHANGED, AND WHAT IS CLAIMED
// ---------------------------------
// ds4_attention used to compute the full head_dim Q.K dot product and its
// sub-group reduction for EVERY key column, then throw the result away if the
// mask said the column was disallowed.  That discarded the entire benefit of the
// lightning indexer, which exists to select top_k = 512 columns so attention does
// not have to look at the rest.  The kernel now tests the mask first and skips
// the dot product outright.
//
// The claim is not "close enough".  The claim is BIT-IDENTICAL output whenever at
// least one column is unmasked (which causality guarantees: a query token always
// sees itself inside the 128-wide sliding window).  A masked column contributed
// exp(s - m_new) with s at or below -1e30, which underflows to exactly 0.0, and
// left the online-softmax state untouched.
//
// HOW THAT IS TESTED WITHOUT A REFERENCE MODEL
// --------------------------------------------
// deepseek4_attn_test — the blob-based gate that covers this kernel — cannot run
// on this machine: it needs $DS4_PARITY2_DIR and refuses (correctly) to exit 0
// without it.  So this test does not need one.  It drives the SAME kernel down
// both paths and compares them to each other:
//
//   SKIP path:  masked columns carry -inf (the indexer's block bias) or
//               float::lowest() (the sliding causal mask).  Both are at or below
//               the kMaskedCut sentinel threshold, so the kernel skips them.
//   WALK path:  the identical mask with every sentinel replaced by -1e20, which
//               is ABOVE the sentinel threshold and so is not skipped — the
//               kernel walks every column exactly as it did before the change —
//               but is still far enough negative that exp() underflows to
//               exactly 0.0, so the mathematics is unchanged.
//
// Same kernel, same inputs, same arithmetic; one path skips and one does not.
// Bit-identity between them is precisely the property the change asserts, and a
// mis-set threshold or an early `continue` that skipped a LIVE column would break
// it immediately.
//
// §2 additionally checks both paths against a double-precision host softmax, so
// this file does not merely prove the two paths agree with each other — it proves
// they agree with the definition, sinks included.
#undef NDEBUG  // build is Release (-DNDEBUG); asserts must stay live here
#include "ie/deepseek4_attn.hpp"
#include "ie/kernel_profiler.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::printf("  FAIL: %s\n", what); ++g_fail; }
}

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// A mask shaped like the real one: a 128-wide sliding causal band over the local
// rows (float::lowest() outside it) concatenated with an indexer block bias that
// is -inf everywhere except `top_k` randomly chosen compressed columns.
std::vector<float> make_mask(uint32_t T, uint32_t n_sl, uint32_t n_c, uint32_t window,
                             uint32_t top_k, uint64_t seed) {
    const uint32_t n_kv = n_sl + n_c;
    std::vector<float> m(size_t(T) * n_kv, kNegInf);
    std::mt19937_64 rng(seed);
    const float lowest = std::numeric_limits<float>::lowest();
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t i = 0; i < n_sl; ++i) {
            const bool ok = (i <= t) && (t - i < window);
            m[size_t(t) * n_kv + i] = ok ? 0.f : lowest;
        }
        std::vector<uint32_t> idx(n_c);
        for (uint32_t i = 0; i < n_c; ++i) idx[i] = i;
        std::shuffle(idx.begin(), idx.end(), rng);
        for (uint32_t j = 0; j < std::min(top_k, n_c); ++j)
            m[size_t(t) * n_kv + n_sl + idx[j]] = 0.f;
    }
    return m;
}

// The same mask with every sentinel lifted above the skip threshold.
std::vector<float> soften(const std::vector<float>& m) {
    std::vector<float> s(m.size());
    for (size_t i = 0; i < m.size(); ++i) s[i] = (m[i] > -1e30f) ? m[i] : -1e20f;
    return s;
}

double run(sycl::queue& q, const std::vector<float>& hq, const std::vector<float>& hkv,
           const std::vector<float>& hmask, const std::vector<float>& hsinks,
           std::vector<float>& out, uint32_t T, uint32_t H, uint32_t D, uint32_t n_kv,
           int reps) {
    float* dq = sycl::malloc_device<float>(hq.size(), q);
    float* dkv = sycl::malloc_device<float>(hkv.size(), q);
    float* dm = sycl::malloc_device<float>(hmask.size(), q);
    float* ds = sycl::malloc_device<float>(hsinks.size(), q);
    float* dy = sycl::malloc_device<float>(out.size(), q);
    q.memcpy(dq, hq.data(), hq.size() * 4).wait();
    q.memcpy(dkv, hkv.data(), hkv.size() * 4).wait();
    q.memcpy(dm, hmask.data(), hmask.size() * 4).wait();
    q.memcpy(ds, hsinks.data(), hsinks.size() * 4).wait();

    const float scaling = 1.f / std::sqrt(float(D));
    ie::ds4_attention(q, dq, dkv, dm, ds, dy, T, H, D, n_kv, scaling).wait();
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r)
        ie::ds4_attention(q, dq, dkv, dm, ds, dy, T, H, D, n_kv, scaling);
    q.wait();
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() /
        reps;

    q.memcpy(out.data(), dy, out.size() * 4).wait();
    sycl::free(dq, q); sycl::free(dkv, q); sycl::free(dm, q);
    sycl::free(ds, q); sycl::free(dy, q);
    return ms;
}

void gate(sycl::queue& q, const char* label, uint32_t T, uint32_t H, uint32_t D,
          uint32_t n_sl, uint32_t n_c, uint32_t top_k, uint64_t seed) {
    const uint32_t n_kv = n_sl + n_c;
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);

    std::vector<float> hq(size_t(T) * H * D), hkv(size_t(n_kv) * D), hs(H);
    for (auto& v : hq) v = nd(rng) * 0.3f;
    for (auto& v : hkv) v = nd(rng) * 0.3f;
    for (auto& v : hs) v = nd(rng) * 0.5f;

    const auto mask_sharp = make_mask(T, n_sl, n_c, 128, top_k, seed ^ 0xbeef);
    const auto mask_soft  = soften(mask_sharp);

    std::vector<float> y_skip(size_t(T) * H * D), y_walk(y_skip.size());
    const double ms_skip = run(q, hq, hkv, mask_sharp, hs, y_skip, T, H, D, n_kv, 20);
    const double ms_walk = run(q, hq, hkv, mask_soft,  hs, y_walk, T, H, D, n_kv, 20);

    size_t mism = 0;
    double worst = 0.0;
    for (size_t i = 0; i < y_skip.size(); ++i) {
        if (std::memcmp(&y_skip[i], &y_walk[i], 4) != 0) {
            ++mism;
            worst = std::max(worst, std::fabs(double(y_skip[i]) - double(y_walk[i])));
        }
    }

    // Fraction of columns the mask actually admits, so the timing has context.
    size_t live = 0;
    for (float v : mask_sharp) if (v > -1e30f) ++live;
    const double dens = double(live) / double(mask_sharp.size());

    std::printf("  %-26s T=%-4u n_kv=%-5u live=%4.1f%%   skip %7.3f ms   walk %7.3f ms   "
                "%5.2fx   bitwise mismatches %zu\n",
                label, T, n_kv, dens * 100.0, ms_skip, ms_walk, ms_walk / ms_skip, mism);
    if (mism) std::printf("      worst |delta| = %.3e\n", worst);
    check(mism == 0, "skip and walk paths are not bit-identical");
}

// ---- §2 host reference ----------------------------------------------------
// Online softmax has no bearing here: accumulate in double, straight from the
// definition, including the sink logit that enters only the denominator.
void gate_vs_host(sycl::queue& q, uint32_t T, uint32_t H, uint32_t D,
                  uint32_t n_sl, uint32_t n_c, uint32_t top_k, uint64_t seed) {
    const uint32_t n_kv = n_sl + n_c;
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hq(size_t(T) * H * D), hkv(size_t(n_kv) * D), hs(H);
    for (auto& v : hq) v = nd(rng) * 0.3f;
    for (auto& v : hkv) v = nd(rng) * 0.3f;
    for (auto& v : hs) v = nd(rng) * 0.5f;
    const auto mask = make_mask(T, n_sl, n_c, 128, top_k, seed ^ 0xf00d);

    std::vector<float> y(size_t(T) * H * D);
    run(q, hq, hkv, mask, hs, y, T, H, D, n_kv, 1);

    const double scaling = 1.0 / std::sqrt(double(D));
    // Deviation is scaled by the OUTPUT TENSOR's own magnitude, not per element.
    // An attention output is a convex combination of value rows, so individual
    // components pass through zero and a per-element relative error there is
    // unbounded no matter how correct the kernel is.
    double worst_abs = 0.0, out_scale = 0.0, sum_qk = 0.0;
    for (uint32_t t = 0; t < T; ++t) {
        for (uint32_t h = 0; h < H; ++h) {
            std::vector<double> logit(n_kv);
            double mx = -std::numeric_limits<double>::infinity();
            for (uint32_t i = 0; i < n_kv; ++i) {
                const float mv = mask[size_t(t) * n_kv + i];
                if (!(mv > -1e30f)) { logit[i] = -std::numeric_limits<double>::infinity(); continue; }
                double dot = 0.0, mag = 0.0;
                for (uint32_t d = 0; d < D; ++d) {
                    const double p = double(hq[(size_t(t) * H + h) * D + d]) *
                                     double(hkv[size_t(i) * D + d]);
                    dot += p;
                    mag += std::fabs(p);
                }
                sum_qk = std::max(sum_qk, mag);
                logit[i] = dot * scaling + double(mv);
                mx = std::max(mx, logit[i]);
            }
            mx = std::max(mx, double(hs[h]));
            double den = std::exp(double(hs[h]) - mx);
            std::vector<double> w(n_kv, 0.0);
            for (uint32_t i = 0; i < n_kv; ++i)
                if (logit[i] > -1e30) { w[i] = std::exp(logit[i] - mx); den += w[i]; }
            for (uint32_t d = 0; d < D; ++d) {
                double acc = 0.0;
                for (uint32_t i = 0; i < n_kv; ++i)
                    if (w[i] != 0.0) acc += w[i] * double(hkv[size_t(i) * D + d]);
                const double ref = acc / den;
                const double got = double(y[(size_t(t) * H + h) * D + d]);
                worst_abs = std::max(worst_abs, std::fabs(got - ref));
                out_scale = std::max(out_scale, std::fabs(ref));
            }
        }
    }
    // DERIVED bound, not a round number.  Each logit is an fp32 dot of length D
    // spread over kSG=16 sub-group lanes, so the longest serial chain is D/16 and
    // the reduce adds log2(16)=4 levels:
    //     dLogit <= (D/16 + 4) * u * max_i SUM|q_d k_d| * scaling
    // A relative perturbation of that size on every softmax weight moves the
    // convex combination by at most ~2*dLogit of the value magnitude, and the
    // fp32 accumulation over n_kv terms adds sqrt(n_kv)*u.  C=4 covers the
    // difference between this and the exact worst case.
    const double u = 5.9604644775390625e-08;
    const double d_logit = (double(D) / 16.0 + 4.0) * u * sum_qk * scaling;
    const double bound = 4.0 * (2.0 * d_logit + std::sqrt(double(n_kv)) * u);
    const double rel = out_scale > 0 ? worst_abs / out_scale : 0.0;
    std::printf("  %-26s T=%-4u n_kv=%-5u   deviation %.3e vs derived bound %.3e "
                "(%.1f%% of bound)\n",
                "vs host reference", T, n_kv, rel, bound, 100.0 * rel / bound);
    check(rel <= bound, "ds4_attention deviates from the double-precision host reference");
}

}  // namespace

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    std::printf("device: %s\n\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    std::puts("§1 mask-first skip is bit-identical to walking every column");
    // head_dim 512 and 64 heads are the real V4-Flash attention shape; the
    // n_sl/n_c splits below are the real ones at ctx 1024 / 4096 for a ratio-4
    // (indexer, top_k 512) layer, plus a decode step.
    gate(q, "decode ctx4096 ratio4",  1, 8, 512, 128, 1024, 512, 11);
    gate(q, "prefill T=64 ctx4096",  64, 8, 512, 128, 1024, 512, 12);
    gate(q, "decode ctx16k ratio4",   1, 8, 512, 128, 4096, 512, 13);
    gate(q, "indexer head_dim 128",   4, 8, 128, 128, 1024, 512, 14);
    gate(q, "dense mask (no sparsity)", 8, 4, 512, 128,  0,   0, 15);

    std::puts("\n§2 both paths against a double-precision host softmax");
    gate_vs_host(q, 4, 4, 512, 128, 256, 128, 21);
    gate_vs_host(q, 1, 4, 128, 128, 512, 256, 22);

    if (g_fail) { std::printf("\nFAILED: %d check(s)\n", g_fail); return 1; }
    std::puts("\nOK — the early skip is bit-identical to the full walk, and both "
              "match the host reference.");
    return 0;
}
