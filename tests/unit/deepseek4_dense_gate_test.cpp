// tests/unit/deepseek4_dense_gate_test.cpp — numerical gate for the two dense
// projection kernels in src/model/deepseek4.cpp: `dense_f16` and `grouped_f16`.
//
// WHY THIS EXISTS
// ---------------
// These two kernels carry ~66% of DeepSeek-V4 prefill GPU time (`dense_f16@attn`
// 38.8%, `grouped_f16@attn` 17.6%, `dense_f16@shexp` 9.5%, measured with
// --kprofile-prefill on the real model), and until this file they had **no
// numerical coverage of any kind**.  deepseek4_parity_test covers the Phase-2
// primitives in deepseek4_ops.cpp; deepseek4_experts_test covers the packed
// expert GEMVs; `ds4_dense_packed_gemv` covers the Q8_0/Q6_K branch.  The fp16
// branch — the one EVERY BF16 weight in the Q8_K_XL model takes, because BF16 is
// deliberately not packed — was covered by nothing.  A mutant that transposed
// the activation index would have shipped.
//
// That gap has to close before the kernels are replaced, not after.
//
// GROUND TRUTH
// ------------
// The exact dot product, accumulated in `double` on the host from the SAME fp16
// weight bytes and fp32 activations the device is given.  This is not a second
// implementation that could share a bug with the first: it is the definition of
// the operation, evaluated at ~2^29 times the precision of the thing under test.
// Full coverage of every output element at small shapes; a random sample of
// output elements at the real per-card shapes, where a full host reference would
// be 17 GFLOP of scalar double.
//
// TOLERANCES — derived, then printed next to the observation
// ----------------------------------------------------------
// fp32-accumulate path (`IE_DS4_ONEDNN` off).  Lane `l` of a WG=128 work-group
// accumulates the terms k = l, l+128, ... — a serial fp32 chain of length K/128
// — and `reduce_over_group` then adds ~7 tree levels.  Worst case:
//
//     |err| <= (K/128 + 7) * u * SUM|w_k * x_k|,   u = 2^-24 = 5.96e-8
//
// evaluated per output element from that element's own SUM|w*x|.  This is the
// deterministic worst case, not an RMS estimate, so it is a bound the kernel
// must not merely usually satisfy.
//
// fp16-activation path (`IE_DS4_ONEDNN` on).  Rounding x to fp16 before the
// matmul is a REAL loss of precision and this gate states it rather than hiding
// it: each activation carries relative error <= 2^-11, so
//
//     |err| <= (2^-11 + (K/128 + 7) * u) * SUM|w_k * x_k|
//
// The 2^-11 term dominates by ~4 orders of magnitude.  The test asserts the
// oneDNN path meets THIS bound and additionally reports whether it also met the
// fp32 bound, so nobody can later mistake the fp16 path for a bit-parity one.
//
// NEGATIVE CONTROLS
// -----------------
// A tolerance is worthless unless a plausible bug exceeds it.  Four mis-ports are
// evaluated on the host against the same inputs — transposed activation,
// transposed weight, grouped group-offset dropped, grouped group index off by one
// — each over 64 sampled output elements.  These are the exact mistakes this
// layout invites: the weight is stored [N, K] while a matmul wants [K, N], and the
// grouped kernel indexes the activation by GROUP while indexing the weight by
// OUTPUT ROW.  The pass criterion and the reason it is not 100% are at
// `kMinCatchRate` below.
//
// §4/§5 — THE REQUANTISED (Q8_0-SoA) BRANCH
// -----------------------------------------
// Added when `upload_dense` gained the option to requantise a BF16 dense weight
// to Q8_0 instead of expanding it to fp16 (1.0625 B/element instead of 2, which
// is the only thing that moves fewer bytes at M=1).  That is the FIRST LOSSY
// step applied to the always-resident set, so it is gated by TWO independent
// assertions rather than one, because it makes two separate claims:
//
//   (a) THE KERNEL IS RIGHT.  Against the exact double dot of the SAME
//       int8-and-scale bytes the device is given, the only admissible error is
//       fp32 reassociation: gamma_p * SUM|w_hat*x| with p from
//       `ds4_decode_gemv_q8_rounding_steps`, which reports the Q8 kernel's OWN
//       reduction shape (32 serial fma inside a block, one per block into the
//       row accumulator, then the sub-group tree).  This bound is ~1e-7
//       relative, it is what the mis-port controls are scored against, and a
//       transposed index misses it by four orders of magnitude.
//
//   (b) THE QUANTISATION COSTS WHAT THE MODEL SAYS.  Against the exact double
//       dot of the ORIGINAL BF16 weights — the thing that was actually replaced
//       — the admissible error additionally carries the round-off:
//
//           |w_hat - w| <= d16/2   per element, d16 = fp16(amax/127) per 32
//        => |y_hat - y| <= SUM_b (d16_b/2) * SUM_{k in b} |x_k|
//
//       evaluated per output element from that element's OWN scales.  There is
//       no second fp16 term because `ds4_requantise_q8_soa` takes its reciprocal
//       from the already-rounded d16, so the decoded weight is exactly
//       d16*round(w/d16).  Both the bound and the observed error are printed as
//       a fraction of SUM|w*x|, because the number that matters to a reader is
//       "how much did 8 bits cost", not "did an inequality hold".
//
// Checking only (b) would let a genuinely broken kernel hide inside a bound made
// loose by quantisation; checking only (a) would prove the kernel faithfully
// computes over bytes nobody showed were close to the weights.  Both, or
// neither.
#undef NDEBUG  // build is Release (-DNDEBUG); asserts must stay live here
#include "ie/deepseek4.hpp"
#include "ie/ds4_decode_gemv.hpp"
#include "ie/kernel_profiler.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr double kU = 5.9604644775390625e-08;  // 2^-24, fp32 unit roundoff
constexpr double kU16 = 4.8828125e-04;         // 2^-11, fp16 unit roundoff
constexpr uint32_t kWG = 128;                  // the kernels' work-group size

int g_fail = 0;
bool g_onednn = false;   // whether the dispatcher is expected to route to oneDNN

void check(bool ok, const char* what) {
    if (!ok) { std::printf("  FAIL: %s\n", what); ++g_fail; }
}

// double-precision reference for one output element of dense_f16, returning both
// the value and SUM|w*x| (the tolerance is proportional to it).
struct Ref { double y, mag; };

Ref ref_dense(const std::vector<sycl::half>& w, const std::vector<float>& x,
              uint32_t t, uint32_t n, uint32_t K) {
    Ref r{0.0, 0.0};
    const sycl::half* wr = w.data() + uint64_t(n) * K;
    const float* xr = x.data() + uint64_t(t) * K;
    for (uint32_t k = 0; k < K; ++k) {
        const double p = double(float(wr[k])) * double(xr[k]);
        r.y += p;
        r.mag += std::fabs(p);
    }
    return r;
}

Ref ref_grouped(const std::vector<sycl::half>& w, const std::vector<float>& x,
                uint32_t t, uint32_t n, uint32_t G, uint32_t IPG, uint32_t OPG) {
    const uint32_t g = n / OPG;
    Ref r{0.0, 0.0};
    const sycl::half* wr = w.data() + uint64_t(n) * IPG;
    const float* xr = x.data() + uint64_t(t) * G * IPG + uint64_t(g) * IPG;
    for (uint32_t k = 0; k < IPG; ++k) {
        const double p = double(float(wr[k])) * double(xr[k]);
        r.y += p;
        r.mag += std::fabs(p);
    }
    return r;
}

double bound_for(double mag, uint32_t K) {
    const double acc = (double(K) / kWG + 7.0) * kU;
    return (g_onednn ? kU16 + acc : acc) * mag;
}

// Evaluate each mis-port over `S` sampled output elements and report the
// fraction at which the gate would catch it, plus the median separation.
//
// WHY THE BAR IS NOT 100%.  At an output element where the true dot product and
// the mutated one happen to nearly coincide — dot products of length K cancel,
// and two different sums of K random products agree closely every so often — no
// per-element tolerance can tell them apart.  That is a property of the
// arithmetic, not of the kernel or the bound, and it is only VISIBLE under the
// fp16 route, whose bound is ~200x looser (2^-11 vs 2^-24) than the fp32 one.
//
// What the real assertion needs is that a mutant fails SOMEWHERE among the 2048
// elements each shape checks.  A mutant caught at >=90% of elements escapes 2048
// independent draws with probability ~1e-2048.  So the criterion is a high catch
// RATE plus an order-of-magnitude median separation, and both are printed so any
// future erosion of the margin is legible rather than silent.
constexpr double kMinCatchRate = 0.90;
constexpr double kMinMedianSep = 10.0;
struct Mutant {
    const char* name;
    std::function<double(uint32_t, uint32_t)> eval;   // -> mutated value at (t, n)
};

// `rel` >= 0 overrides `bound_for` with `rel * mag`.  §4/§5 need it: the
// requantised route's admissible arithmetic error comes from the Q8 kernel's own
// reduction shape, not from the fp16 kernels' K/128+7 model, and a control
// scored against the wrong bound measures the wrong gate.
void run_controls(uint32_t K, uint32_t S, uint64_t seed,
                  const std::function<Ref(uint32_t, uint32_t)>& truth,
                  uint32_t T, uint32_t N, std::vector<Mutant> mutants,
                  double rel = -1.0) {
    for (const auto& m : mutants) {
        std::mt19937_64 pick(seed);
        std::vector<double> seps;
        uint32_t caught = 0;
        for (uint32_t i = 0; i < S; ++i) {
            const uint32_t t = uint32_t(pick() % T), n = uint32_t(pick() % N);
            const Ref r = truth(t, n);
            const double b = rel >= 0.0 ? rel * r.mag : bound_for(r.mag, K);
            if (b <= 0) continue;
            const double mv = m.eval(t, n);
            if (std::isnan(mv)) continue;   // element where this mutant is a no-op
            const double sep = std::fabs(mv - r.y) / b;
            seps.push_back(sep);
            if (sep > 1.0) ++caught;
        }
        std::sort(seps.begin(), seps.end());
        const double med = seps.empty() ? 0.0 : seps[seps.size() / 2];
        const double worst = seps.empty() ? 0.0 : seps.front();
        const double rate = seps.empty() ? 0.0 : double(caught) / double(seps.size());
        std::printf("      neg ctrl %-24s caught %u/%zu (%.0f%%), separation median %.0fx "
                    "min %.1fx\n", m.name, caught, seps.size(), rate * 100.0, med, worst);
        check(rate >= kMinCatchRate,
              (std::string("mutant '") + m.name + "' catch rate below " +
               std::to_string(int(kMinCatchRate * 100)) + "%").c_str());
        check(med >= kMinMedianSep,
              (std::string("mutant '") + m.name + "' median separation below " +
               std::to_string(int(kMinMedianSep)) + "x").c_str());
    }
}

// ---------------------------------------------------------------------------
// dense_f16
// ---------------------------------------------------------------------------
// `sample` == 0 means check every output element.
void gate_dense(sycl::queue& q, const char* label, uint32_t T, uint32_t K, uint32_t N,
                uint32_t sample, uint64_t seed) {
    const uint64_t xn = uint64_t(T) * K, wn = uint64_t(N) * K, yn = uint64_t(T) * N;

    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hx(xn);
    std::vector<sycl::half> hw(wn);
    for (auto& v : hx) v = nd(rng) * 0.5f;
    for (auto& v : hw) v = sycl::half(nd(rng) * 0.05f);

    float* x = sycl::malloc_device<float>(xn, q);
    sycl::half* w = sycl::malloc_device<sycl::half>(wn, q);
    float* y = sycl::malloc_device<float>(yn, q);
    q.memcpy(x, hx.data(), xn * sizeof(float)).wait();
    q.memcpy(w, hw.data(), wn * sizeof(sycl::half)).wait();

    const std::string err = ie::ds4_dense_f16_gemv(q, x, w, y, T, K, N);
    check(err.empty(), err.empty() ? "" : err.c_str());

    std::vector<float> hy(yn);
    q.memcpy(hy.data(), y, yn * sizeof(float)).wait();

    // ---- the check -------------------------------------------------------
    double worst_ratio = 0.0, worst_abs = 0.0, worst_bound = 0.0;
    uint32_t checked = 0;
    std::mt19937_64 pick(seed ^ 0x9e3779b97f4a7c15ull);
    const uint32_t n_checks = sample ? sample : uint32_t(yn);
    for (uint32_t i = 0; i < n_checks; ++i) {
        uint32_t t, n;
        if (sample) { t = uint32_t(pick() % T); n = uint32_t(pick() % N); }
        else        { t = uint32_t(i / N);      n = uint32_t(i % N); }
        const Ref r = ref_dense(hw, hx, t, n, K);
        const double b = bound_for(r.mag, K);
        const double d = std::fabs(double(hy[uint64_t(t) * N + n]) - r.y);
        if (b > 0 && d / b > worst_ratio) { worst_ratio = d / b; worst_abs = d; worst_bound = b; }
        ++checked;
    }
    std::printf("  %-30s T=%-4u K=%-5u N=%-6u  %u elems  worst |d|=%.3e  bound=%.3e  (%.2f%% of bound)\n",
                label, T, K, N, checked, worst_abs, worst_bound, worst_ratio * 100.0);
    check(worst_ratio <= 1.0, "dense_f16 exceeded its error bound");

    // ---- negative controls ----------------------------------------------
    // A mutant is CAUGHT at an output element when its deviation exceeds that
    // element's own bound — that is literally the assertion above, so this
    // measures the gate's real discriminating power rather than a proxy.
    //
    // Sampled over many elements on purpose.  A single element is not a stable
    // control: how far a mis-port lands from the truth depends on how much
    // cancellation that particular dot product happened to have, and an element
    // where the two nearly agree proves nothing about the kernel.
    if (T > 1 && N > 1) {
        run_controls(K, 64, seed ^ 0xa5a5,
                     [&](uint32_t t, uint32_t n) { return ref_dense(hw, hx, t, n, K); },
                     T, N,
                     {{"transposed activation",
                       [&](uint32_t t, uint32_t n) {
                           double m = 0.0;
                           for (uint32_t k = 0; k < K; ++k)
                               m += double(float(hw[uint64_t(n) * K + k])) *
                                    double(hx[uint64_t(k) * T + t]);
                           return m;
                       }},
                      {"transposed weight",
                       [&](uint32_t t, uint32_t n) {
                           double m = 0.0;
                           for (uint32_t k = 0; k < K; ++k)
                               m += double(float(hw[uint64_t(k) * N + n])) *
                                    double(hx[uint64_t(t) * K + k]);
                           return m;
                       }}});
    }

    sycl::free(x, q); sycl::free(w, q); sycl::free(y, q);
}

// ---------------------------------------------------------------------------
// grouped_f16
// ---------------------------------------------------------------------------
void gate_grouped(sycl::queue& q, const char* label, uint32_t T, uint32_t G,
                  uint32_t IPG, uint32_t OPG, uint32_t sample, uint64_t seed) {
    const uint32_t N = G * OPG;
    const uint64_t xn = uint64_t(T) * G * IPG, wn = uint64_t(N) * IPG, yn = uint64_t(T) * N;

    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hx(xn);
    std::vector<sycl::half> hw(wn);
    for (auto& v : hx) v = nd(rng) * 0.5f;
    for (auto& v : hw) v = sycl::half(nd(rng) * 0.05f);

    float* x = sycl::malloc_device<float>(xn, q);
    sycl::half* w = sycl::malloc_device<sycl::half>(wn, q);
    float* y = sycl::malloc_device<float>(yn, q);
    q.memcpy(x, hx.data(), xn * sizeof(float)).wait();
    q.memcpy(w, hw.data(), wn * sizeof(sycl::half)).wait();

    const std::string err = ie::ds4_grouped_f16_gemv(q, x, w, y, T, G, IPG, OPG);
    check(err.empty(), err.empty() ? "" : err.c_str());

    std::vector<float> hy(yn);
    q.memcpy(hy.data(), y, yn * sizeof(float)).wait();

    double worst_ratio = 0.0, worst_abs = 0.0, worst_bound = 0.0;
    uint32_t checked = 0;
    std::mt19937_64 pick(seed ^ 0xd1b54a32d192ed03ull);
    const uint32_t n_checks = sample ? sample : uint32_t(yn);
    for (uint32_t i = 0; i < n_checks; ++i) {
        uint32_t t, n;
        if (sample) { t = uint32_t(pick() % T); n = uint32_t(pick() % N); }
        else        { t = uint32_t(i / N);      n = uint32_t(i % N); }
        const Ref r = ref_grouped(hw, hx, t, n, G, IPG, OPG);
        const double b = bound_for(r.mag, IPG);
        const double d = std::fabs(double(hy[uint64_t(t) * N + n]) - r.y);
        if (b > 0 && d / b > worst_ratio) { worst_ratio = d / b; worst_abs = d; worst_bound = b; }
        ++checked;
    }
    std::printf("  %-30s T=%-4u G=%-3u IPG=%-5u OPG=%-5u  %u elems  worst |d|=%.3e  bound=%.3e  (%.2f%% of bound)\n",
                label, T, G, IPG, OPG, checked, worst_abs, worst_bound, worst_ratio * 100.0);
    check(worst_ratio <= 1.0, "grouped_f16 exceeded its error bound");

    // ---- negative controls ----------------------------------------------
    // The block-diagonal structure is the whole point of this kernel: group g
    // must never see group g'!=g's activations.  Both mutants break exactly
    // that, and neither changes the output SHAPE, so nothing else would notice.
    // Elements in group 0 are skipped for both: reading "group 0" or shifting a
    // group index is a no-op there, so scoring them would dilute the control
    // with elements where the mutant is not actually a mutation.
    if (G > 1) {
        run_controls(IPG, 64, seed ^ 0x5a5a,
                     [&](uint32_t t, uint32_t n) { return ref_grouped(hw, hx, t, n, G, IPG, OPG); },
                     T, N,
                     {{"group offset dropped",
                       [&](uint32_t t, uint32_t n) {
                           const uint32_t g = n / OPG;
                           if (g == 0) return std::numeric_limits<double>::quiet_NaN();
                           double m = 0.0;
                           for (uint32_t k = 0; k < IPG; ++k)
                               m += double(float(hw[uint64_t(n) * IPG + k])) *
                                    double(hx[uint64_t(t) * G * IPG + k]);
                           return m;
                       }},
                      {"group index off by one",
                       [&](uint32_t t, uint32_t n) {
                           const uint32_t bad_g = (n / OPG + 1) % G;
                           double m = 0.0;
                           for (uint32_t k = 0; k < IPG; ++k)
                               m += double(float(hw[uint64_t(n) * IPG + k])) *
                                    double(hx[uint64_t(t) * G * IPG + uint64_t(bad_g) * IPG + k]);
                           return m;
                       }}});
    }

    sycl::free(x, q); sycl::free(w, q); sycl::free(y, q);
}

// ---------------------------------------------------------------------------
// §4/§5 helpers — the requantised branch
// ---------------------------------------------------------------------------

// Round-to-nearest-even fp32 -> BF16, returned as the fp32 it decodes to.  The
// gate's "original" weight has to be a REAL BF16 value, because that is what the
// file holds and what `dequant_rows` hands the requantiser; comparing against an
// arbitrary fp32 would charge Q8_0 for a rounding BF16 had already done.
float to_bf16(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    b += 0x7fffu + ((b >> 16) & 1u);
    b &= 0xffff0000u;
    float o;
    std::memcpy(&o, &b, 4);
    return o;
}

// One output element, three ways: the exact BF16 dot, the exact dot over the
// quantised bytes, and the two magnitudes the two bounds scale by.
struct RefQ8 {
    double y_bf;    // exact dot of the ORIGINAL bf16 weights
    double y_q8;    // exact dot of the dequantised q8 weights
    double amag;    // SUM |w_hat * x|      — scales the arithmetic bound
    double qmag;    // SUM_b (d16_b/2) * SUM_{k in b} |x_k|  — the quantisation bound
    double wmag;    // SUM |w * x|          — the denominator the report uses
};

RefQ8 ref_q8(const std::vector<float>& wbf, const std::vector<int8_t>& qs,
             const std::vector<sycl::half>& dsc, const std::vector<float>& x,
             uint32_t n, uint32_t K, uint64_t woff, uint64_t xoff) {
    const uint32_t nb = K / 32;
    RefQ8 r{0, 0, 0, 0, 0};
    for (uint32_t b = 0; b < nb; ++b) {
        const double d = double(float(dsc[uint64_t(n) * nb + b]));
        double sabs = 0.0;
        for (uint32_t j = 0; j < 32; ++j) {
            const uint32_t k = b * 32 + j;
            const double xv = double(x[xoff + k]);
            const double wv = double(wbf[woff + k]);
            const double wh = d * double(qs[woff + k]);
            r.y_bf += wv * xv;
            r.y_q8 += wh * xv;
            r.amag += std::fabs(wh * xv);
            r.wmag += std::fabs(wv * xv);
            sabs += std::fabs(xv);
        }
        r.qmag += 0.5 * d * sabs;
    }
    return r;
}

// gamma_p = p*u / (1 - p*u) — the standard bound for p sequential fp32 roundings.
double gamma_p(uint32_t p) {
    const double pu = double(p) * kU;
    return pu < 1.0 ? pu / (1.0 - pu) : 1.0;
}

// The ARITHMETIC-only relative bound for whichever route `dense_w` takes.
//
// T == 1 is the Q8 decode GEMV and its rounding count comes from the kernel
// itself.  T > 1 materialises the weight as fp16 (`ds4_q8_soa_to_f16`) and
// rejoins the fp16 route, so it additionally carries ONE fp16 rounding of each
// dequantised weight — d16*q is up to 19 significant bits and fp16 holds 11 —
// plus, under IE_DS4_ONEDNN, the fp16 rounding of the activation that route has
// always cost.
double arith_rel(uint32_t T, uint32_t K, uint32_t N) {
    if (T == 1) return gamma_p(ie::ds4_decode_gemv_q8_rounding_steps(K, N));
    const double acc = (double(K) / kWG + 7.0) * kU;
    return kU16 + (g_onednn ? kU16 : 0.0) + acc;
}

// Quantises `nrows` rows of K floats with the LOADER's own packer, so the gate
// cannot be testing a second quantiser that agrees with nothing.
void pack_q8(const std::vector<float>& w, uint32_t K, uint32_t nrows,
             std::vector<int8_t>& qs, std::vector<sycl::half>& dsc) {
    qs.resize(size_t(nrows) * K);
    dsc.resize(size_t(nrows) * (K / 32));
    ie::ds4_requantise_q8_soa(w.data(), K, nrows, qs.data(), dsc.data());
}

// `OPG == 0` selects the dense form; OPG > 0 the block-diagonal one.
void gate_q8(sycl::queue& q, const char* label, uint32_t T, uint32_t K, uint32_t N,
             uint32_t OPG, uint32_t sample, uint64_t seed) {
    const uint32_t G = OPG ? N / OPG : 0;
    const uint64_t xn = OPG ? uint64_t(T) * N / OPG * K : uint64_t(T) * K;
    const uint64_t wn = uint64_t(N) * K, yn = uint64_t(T) * N;
    const uint32_t nb = K / 32;

    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hx(xn), hw(wn);
    for (auto& v : hx) v = nd(rng) * 0.5f;
    // The ORIGINAL weight, as BF16-exact fp32 values.
    for (auto& v : hw) v = to_bf16(nd(rng) * 0.05f);

    std::vector<int8_t> hqs;
    std::vector<sycl::half> hd;
    pack_q8(hw, K, N, hqs, hd);

    float* x = sycl::malloc_device<float>(xn, q);
    int8_t* qs = sycl::malloc_device<int8_t>(wn, q);
    sycl::half* dsc = sycl::malloc_device<sycl::half>(uint64_t(N) * nb, q);
    float* y = sycl::malloc_device<float>(yn, q);
    q.memcpy(x, hx.data(), xn * sizeof(float)).wait();
    q.memcpy(qs, hqs.data(), wn).wait();
    q.memcpy(dsc, hd.data(), uint64_t(N) * nb * sizeof(sycl::half)).wait();

    const std::string err =
        OPG ? ie::ds4_grouped_q8_gemv(q, x, qs, dsc, y, T, G, K, OPG)
            : ie::ds4_dense_q8_gemv(q, x, qs, dsc, y, T, K, N);
    check(err.empty(), err.empty() ? "" : err.c_str());

    std::vector<float> hy(yn);
    q.memcpy(hy.data(), y, yn * sizeof(float)).wait();

    double worst_arith = 0.0, worst_total = 0.0;
    double worst_arith_abs = 0.0, worst_arith_b = 0.0;
    double worst_total_abs = 0.0, worst_total_b = 0.0;
    double sum_qrel = 0.0, worst_qrel = 0.0, worst_qtight = 0.0;
    uint32_t checked = 0;
    std::mt19937_64 pick(seed ^ 0x2545f4914f6cdd1dull);
    const uint32_t n_checks = sample ? sample : uint32_t(yn);
    for (uint32_t i = 0; i < n_checks; ++i) {
        uint32_t t, n;
        if (sample) { t = uint32_t(pick() % T); n = uint32_t(pick() % N); }
        else        { t = uint32_t(i / N);      n = uint32_t(i % N); }
        const uint64_t xoff = OPG ? uint64_t(t) * G * K + uint64_t(n / OPG) * K
                                  : uint64_t(t) * K;
        const RefQ8 r = ref_q8(hw, hqs, hd, hx, n, K, uint64_t(n) * K, xoff);
        const double dev = double(hy[uint64_t(t) * N + n]);

        // (a) the kernel, against its own bytes
        const double ba = arith_rel(T, K, N) * r.amag;
        const double da = std::fabs(dev - r.y_q8);
        if (ba > 0 && da / ba > worst_arith) {
            worst_arith = da / ba; worst_arith_abs = da; worst_arith_b = ba;
        }
        // (b) the whole thing, against the BF16 original
        const double bt = r.qmag + ba;
        const double dt = std::fabs(dev - r.y_bf);
        if (bt > 0 && dt / bt > worst_total) {
            worst_total = dt / bt; worst_total_abs = dt; worst_total_b = bt;
        }
        // what 8 bits actually cost, as a fraction of SUM|w*x|
        if (r.wmag > 0) {
            const double qrel = std::fabs(r.y_q8 - r.y_bf) / r.wmag;
            sum_qrel += qrel;
            worst_qrel = std::max(worst_qrel, qrel);
            worst_qtight = std::max(worst_qtight, r.qmag / r.wmag);
        }
        ++checked;
    }

    std::printf("  %-28s T=%-4u K=%-5u N=%-6u %5u elems\n", label, T, K, N, checked);
    std::printf("      (a) vs its own q8 bytes : |d|=%.3e  bound=%.3e  (%.2f%% of bound)\n",
                worst_arith_abs, worst_arith_b, worst_arith * 100.0);
    std::printf("      (b) vs the BF16 original: |d|=%.3e  bound=%.3e  (%.2f%% of bound)\n",
                worst_total_abs, worst_total_b, worst_total * 100.0);
    std::printf("      quantisation cost as a fraction of SUM|w*x|: mean %.3e, worst %.3e"
                "  (bound %.3e)\n",
                checked ? sum_qrel / checked : 0.0, worst_qrel, worst_qtight);
    check(worst_arith <= 1.0, "q8 kernel exceeded its ARITHMETIC bound over its own bytes");
    check(worst_total <= 1.0, "q8 result exceeded the derived quantisation+arithmetic bound "
                              "against the BF16 original");

    // ---- negative controls ----------------------------------------------
    // Scored against bound (a), the tight one, for the reason spelled out at the
    // top of this file: a mis-port is a claim about INDEXING, and the
    // quantisation term in bound (b) is a property of the data that the correct
    // kernel and every mutant share equally, so including it would only dilute
    // the control.
    //
    // ONLY AT T == 1, and that is a scoping decision with a reason, not an
    // omission.  T == 1 is where this route's NEW kernel is; at T > 1 `dense_w`
    // materialises fp16 and hands over to the fp16 route these controls already
    // cover in §1-§3, and the handover legitimately widens bound (a) by the
    // 2^-11 rounding of `ds4_q8_soa_to_f16`'s output — which is ~28x, enough to
    // pull the weakest mutant's median separation from ~300x down to ~10x.
    // Scoring a control against a bound that is loose for an honest reason
    // measures the reason, not the control.  The dequantiser that widens it is
    // instead checked DIRECTLY and bit-exactly in §5b, which is a stronger
    // statement than any statistical control could make.
    //
    // The first two mutants are the SoA layout's own hazards and exist nowhere
    // else in this file: the weight now lives in TWO planes that have to stay in
    // step, so a scale read for the wrong block, or not read at all, is the
    // mistake this layout invites.
    if (T == 1 && N > 1) {
        auto truth = [&](uint32_t t, uint32_t n) {
            const uint64_t xoff = OPG ? uint64_t(t) * G * K + uint64_t(n / OPG) * K
                                      : uint64_t(t) * K;
            const RefQ8 r = ref_q8(hw, hqs, hd, hx, n, K, uint64_t(n) * K, xoff);
            return Ref{r.y_q8, r.amag};
        };
        auto xo = [&](uint32_t t, uint32_t n) {
            return OPG ? uint64_t(t) * G * K + uint64_t(n / OPG) * K : uint64_t(t) * K;
        };
        std::vector<Mutant> mut = {
            {"scale plane ignored",
             [&](uint32_t t, uint32_t n) {
                 double m = 0.0;
                 for (uint32_t k = 0; k < K; ++k)
                     m += double(hqs[uint64_t(n) * K + k]) * double(hx[xo(t, n) + k]);
                 return m;
             }},
            {"scale block index +1",
             [&](uint32_t t, uint32_t n) {
                 double m = 0.0;
                 for (uint32_t k = 0; k < K; ++k) {
                     const uint32_t b = std::min(nb - 1, (k >> 5) + 1);
                     m += double(float(hd[uint64_t(n) * nb + b])) *
                          double(hqs[uint64_t(n) * K + k]) * double(hx[xo(t, n) + k]);
                 }
                 return m;
             }},
            {"transposed int8 plane",
             [&](uint32_t t, uint32_t n) {
                 double m = 0.0;
                 for (uint32_t k = 0; k < K; ++k)
                     m += double(float(hd[uint64_t(n) * nb + (k >> 5)])) *
                          double(hqs[uint64_t(k) * N + n]) * double(hx[xo(t, n) + k]);
                 return m;
             }}};
        // The block-diagonal form's own hazards, and the reason group 0 is
        // skipped, are §3's verbatim: reading "group 0" or shifting a group
        // index is a no-op there, so scoring those elements would dilute the
        // control with elements where the mutant is not a mutation.
        if (OPG && G > 1) {
            mut.push_back({"group offset dropped",
                           [&](uint32_t t, uint32_t n) {
                               if (n / OPG == 0) return std::numeric_limits<double>::quiet_NaN();
                               double m = 0.0;
                               for (uint32_t k = 0; k < K; ++k)
                                   m += double(float(hd[uint64_t(n) * nb + (k >> 5)])) *
                                        double(hqs[uint64_t(n) * K + k]) *
                                        double(hx[uint64_t(t) * G * K + k]);
                               return m;
                           }});
            mut.push_back({"group index off by one",
                           [&](uint32_t t, uint32_t n) {
                               const uint32_t bad = (n / OPG + 1) % G;
                               double m = 0.0;
                               for (uint32_t k = 0; k < K; ++k)
                                   m += double(float(hd[uint64_t(n) * nb + (k >> 5)])) *
                                        double(hqs[uint64_t(n) * K + k]) *
                                        double(hx[uint64_t(t) * G * K + uint64_t(bad) * K + k]);
                               return m;
                           }});
        }
        run_controls(K, 64, seed ^ 0xc3c3, truth, T, N, mut, arith_rel(T, K, N));
    }

    sycl::free(x, q); sycl::free(qs, q); sycl::free(dsc, q); sycl::free(y, q);
}

}  // namespace

int main() {
    // §4/§5/§8 exercise the Q8_0-SoA requantisation path, which became OPT-IN on
    // 2026-08-03: it is faster in isolation but regressed the real model on both
    // prefill (116.85 -> 55.67 tok/s) and decode (16.70 -> 14.52), so
    // `ds4_dense_requantises` now returns false unless IE_DS4_DENSE_Q8 is set.
    //
    // The gate turns it on for ITSELF rather than deleting those sections. The
    // path is still in the build, still reachable by one environment variable,
    // and whoever explains the regression needs its correctness proof intact —
    // an untested disabled path is how a lever comes back broken.
    setenv("IE_DS4_DENSE_Q8", "1", /*overwrite=*/0);

    // MUST mirror ds4_onednn_on() in src/model/deepseek4.cpp exactly, including
    // its default.  That routing is DEFAULT ON, so an unset variable means the
    // fp16 route is live; defaulting this to the fp32 bound instead made the gate
    // apply a ~200x tighter tolerance than the path it was actually measuring and
    // fail 15 checks.  The CTest entries set the variable explicitly, so this only
    // bites someone running the binary by hand — which is exactly when a
    // misleading failure costs the most time.
    const char* e = std::getenv("IE_DS4_ONEDNN");
    g_onednn = !(e && *e) || std::string(e) != "0";

    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    std::printf("device: %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());
    std::printf("IE_DS4_ONEDNN=%s -> tolerance model: %s\n\n",
                g_onednn ? "on" : "off",
                g_onednn ? "fp16 activations (2^-11 dominates)" : "fp32 accumulate (2^-24)");

    // -- §1 dense_f16, exhaustive at small shapes --------------------------
    // T=1 and T=4 take the kTTSmall=4 tile (the decode path); T=17 takes
    // kTTLarge=16 AND leaves a 1-token partial tile, which is where an
    // off-by-one in the `t < T` guard would live.
    std::puts("§1 dense_f16 — every output element checked");
    gate_dense(q, "tiny/decode T=1",        1, 256,  64, 0, 1001);
    gate_dense(q, "tiny/decode T=4",        4, 256,  64, 0, 1002);
    gate_dense(q, "tiny/prefill T=5",       5, 256,  64, 0, 1003);
    gate_dense(q, "tiny/prefill T=17",     17, 256,  64, 0, 1004);
    gate_dense(q, "ragged K T=17",         17, 300,  48, 0, 1005);

    // -- §2 dense_f16 at the real per-card shapes --------------------------
    std::puts("\n§2 dense_f16 — real DeepSeek-V4 per-card shapes, sampled");
    gate_dense(q, "q_a  4096->1024",      512, 4096, 1024, 2048, 2001);
    gate_dense(q, "q_b  1024->16384",     512, 1024, 16384, 2048, 2002);
    gate_dense(q, "kv   4096->512",       512, 4096,  512, 2048, 2003);
    gate_dense(q, "o_b  8192->4096",      512, 8192, 4096, 2048, 2004);
    gate_dense(q, "shexp_d 2048->4096",   512, 2048, 4096, 2048, 2005);
    gate_dense(q, "decode q_a T=1",         1, 4096, 1024,    0, 2006);

    // -- §3 grouped_f16 ----------------------------------------------------
    std::puts("\n§3 grouped_f16 — block-diagonal output projection");
    gate_grouped(q, "tiny G=3 T=9",         9, 3,  128,  32, 0, 3001);
    gate_grouped(q, "tiny G=4 T=17",       17, 4,  256,  64, 0, 3002);
    gate_grouped(q, "o_a G=4 4096->1024", 512, 4, 4096, 1024, 2048, 3003);
    gate_grouped(q, "o_a decode T=1",       1, 4, 4096, 1024,    0, 3004);

    // -- §4 the requantised branch, dense ----------------------------------
    // T=1 is the Q8 decode GEMV; T=17 is the fp16 materialisation plus the
    // ordinary prefill route, so BOTH halves of `dense_w`'s q8 branch run, and
    // under IE_DS4_ONEDNN=1 the T=17 rows additionally cross oneDNN.
    // K=288 is a whole number of Q8_0 blocks but NOT of 256, which is where a
    // decoder that assumed a Q_K superblock would go wrong.
    std::puts("\n§4 requantised Q8_0-SoA dense — every output element checked");
    gate_q8(q, "tiny/decode T=1",       1, 256,  64, 0, 0, 4001);
    gate_q8(q, "tiny/decode T=4",       4, 256,  64, 0, 0, 4002);
    gate_q8(q, "tiny/prefill T=17",    17, 256,  64, 0, 0, 4003);
    gate_q8(q, "block-ragged K=288",   17, 288,  48, 0, 0, 4004);

    std::puts("\n§4b requantised Q8_0-SoA dense — real per-card shapes, sampled");
    gate_q8(q, "decode q_a  4096->1024",  1, 4096,  1024, 0,    0, 4101);
    gate_q8(q, "decode q_b  1024->16384", 1, 1024, 16384, 0, 2048, 4102);
    gate_q8(q, "decode kv   4096->512",   1, 4096,   512, 0,    0, 4103);
    gate_q8(q, "decode o_b  4096->4096",  1, 4096,  4096, 0, 2048, 4104);
    gate_q8(q, "decode sh_d 1024->4096",  1, 1024,  4096, 0, 2048, 4105);
    gate_q8(q, "prefill q_a 4096->1024", 17, 4096,  1024, 0, 2048, 4106);

    // -- §5 the requantised branch, grouped --------------------------------
    std::puts("\n§5 requantised Q8_0-SoA grouped");
    gate_q8(q, "tiny G=3 T=9",            9,  128,  96,  32, 0, 5001);
    gate_q8(q, "tiny G=4 T=17",          17,  256, 256,  64, 0, 5002);
    gate_q8(q, "o_a decode T=1",          1, 4096, 4096, 1024, 2048, 5003);
    gate_q8(q, "o_a prefill T=17",       17, 4096, 4096, 1024, 2048, 5004);

    // -- §5b the dequantiser the T>1 route stands on -----------------------
    // `ds4_q8_soa_to_f16` is what lets a requantised weight keep the oneDNN
    // prefill route, and it is the one piece of the q8 branch the T=1 controls
    // above cannot reach.  It is elementwise, so it does not need a statistical
    // control: the exact expected value is computable and the check is for BIT
    // EQUALITY over every element.  A scale read for the wrong block, the wrong
    // row, or a plane confused for the other fails this on the first element.
    std::puts("\n§5b ds4_q8_soa_to_f16 — bit-exact over every element");
    for (auto shape : {std::pair<uint32_t, uint32_t>{256, 64},
                       std::pair<uint32_t, uint32_t>{288, 48},
                       std::pair<uint32_t, uint32_t>{4096, 512}}) {
        const uint32_t K = shape.first, N = shape.second, nb = K / 32;
        const uint64_t wn = uint64_t(K) * N;
        std::mt19937_64 rng(6001 + K);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> hw(wn);
        for (auto& v : hw) v = to_bf16(nd(rng) * 0.05f);
        std::vector<int8_t> hqs;
        std::vector<sycl::half> hd;
        pack_q8(hw, K, N, hqs, hd);

        int8_t* qs = sycl::malloc_device<int8_t>(wn, q);
        sycl::half* dsc = sycl::malloc_device<sycl::half>(uint64_t(N) * nb, q);
        sycl::half* out = sycl::malloc_device<sycl::half>(wn, q);
        q.memcpy(qs, hqs.data(), wn).wait();
        q.memcpy(dsc, hd.data(), uint64_t(N) * nb * sizeof(sycl::half)).wait();
        ie::ds4_q8_soa_to_f16(q, qs, dsc, out, K, N).wait();
        std::vector<sycl::half> hy(wn);
        q.memcpy(hy.data(), out, wn * sizeof(sycl::half)).wait();

        uint64_t bad = 0;
        double worst = 0.0;
        for (uint64_t i = 0; i < wn; ++i) {
            const uint64_t row = i / K;
            const uint32_t k = uint32_t(i - row * K);
            const sycl::half want =
                sycl::half(float(hd[row * nb + (k >> 5)]) * float(hqs[i]));
            if (float(hy[i]) != float(want)) ++bad;
            worst = std::max(worst, std::fabs(double(float(hy[i])) - double(hw[i])));
        }
        std::printf("  K=%-5u N=%-5u  %llu elements, %llu mismatched;"
                    "  worst |fp16(d*q) - bf16 original| = %.3e\n",
                    K, N, (unsigned long long)wn, (unsigned long long)bad, worst);
        check(bad == 0, "ds4_q8_soa_to_f16 is not bit-exact against fp16(d*q)");
        sycl::free(qs, q); sycl::free(dsc, q); sycl::free(out, q);
    }

    // -- §6 the seam refuses a K that is not a whole number of blocks -------
    // The decoder addresses `k` inside a 32-element block; a ragged final block
    // is not addressable, and the alternative to refusing is reading past the
    // scale plane.  This is the same contract `ds4_dense_packed_gemv` enforces.
    std::puts("\n§6 the requantised seam refuses a ragged K");
    {
        int8_t* qs = sycl::malloc_device<int8_t>(64 * 40, q);
        sycl::half* d = sycl::malloc_device<sycl::half>(64, q);
        float* x = sycl::malloc_device<float>(40, q);
        float* y = sycl::malloc_device<float>(64, q);
        const std::string e1 = ie::ds4_dense_q8_gemv(q, x, qs, d, y, 1, 40, 64);
        const std::string e2 = ie::ds4_grouped_q8_gemv(q, x, qs, d, y, 1, 2, 40, 32);
        const std::string e3 = ie::ds4_dense_q8_gemv(q, x, nullptr, d, y, 1, 64, 64);
        std::printf("  dense K=40   -> \"%s\"\n", e1.c_str());
        std::printf("  grouped IPG=40 -> \"%s\"\n", e2.c_str());
        std::printf("  null int8 plane -> \"%s\"\n", e3.c_str());
        check(!e1.empty(), "ds4_dense_q8_gemv accepted a K that is not a multiple of 32");
        check(!e2.empty(), "ds4_grouped_q8_gemv accepted an IPG that is not a multiple of 32");
        check(!e3.empty(), "ds4_dense_q8_gemv accepted a null weight plane");
        sycl::free(qs, q); sycl::free(d, q); sycl::free(x, q); sycl::free(y, q);
    }

    // -- §7 the policy itself ----------------------------------------------
    // The predicate and the byte count are what the loader and every residency
    // spreadsheet consult; pinning them here keeps "1.0625 B/element" a checked
    // fact rather than a comment.
    std::puts("\n§7 the requantisation policy");
    check(ie::ds4_dense_requantises(ie::DType::kBF16, 4096),
          "BF16 with K=4096 must be requantisable");
    check(!ie::ds4_dense_requantises(ie::DType::kBF16, 48),
          "BF16 with a K that is not a whole number of Q8_0 blocks must NOT be");
    check(!ie::ds4_dense_requantises(ie::DType::kBF16, 0), "K=0 must not be requantisable");
    check(ie::ds4_dense_requantises(ie::DType::kF16, 4096),
          "F16 source-preserving weights with K=4096 must be requantisable");
    check(ie::ds4_dense_requantises(ie::DType::kF32, 4096),
          "F32 source-preserving weights with K=4096 must be requantisable");
    check(ie::ds4_dense_requantises(ie::DType::kQ8_0, 4096),
          "Q8_0 sources take the SoA route ahead of their packed path (2026-09-11: "
          "the detour is exact and the packed kernels are ~10x slower at prefill)");
    check(!ie::ds4_dense_requantises(ie::DType::kQ8_0, 48),
          "Q8_0 with a ragged K stays packed (the SoA seam needs whole blocks)");
    check(!ie::ds4_dense_requantises(ie::DType::kQ6_K, 4096),
          "Q6_K stays packed — a re-quantisation to Q8_0 would be lossy");
    check(ie::ds4_dense_requant_bytes(4096, 1024) == 4096ull * 1024 + 2 * (1024ull * 128 * 2),
          "requantised bytes are the int8 plane plus the fp16 scale plane twice "
          "(n-major for the GEMV, kb-major for oneDNN's s8 route)");
    check(ie::ds4_dense_requant_bytes(4096, 1024) * 32 == 18ull * 4096 * 1024 * 2,
          "requantised bytes are exactly 1.125 B/element, i.e. 18/32 of fp16");
    // `ds4_dense_device_bytes` must NOT have learned about requantisation: it
    // answers the packed-or-expanded question, and deepseek4_residency_test's
    // §4/§8c arithmetic is built on that meaning.
    check(ie::ds4_dense_device_bytes(ie::DType::kBF16, 4096, 512) == 4096ull * 512 * 2,
          "ds4_dense_device_bytes must still report the fp16 expansion for BF16");
    std::printf("  policy: BF16/F16/F32/Q8_0 K%%32==0 -> Q8_0-SoA at %llu B/1024 elems (fp16 would be %llu)\n",
                (unsigned long long)ie::ds4_dense_requant_bytes(1024, 1),
                (unsigned long long)(1024ull * 2));

    // -- §8 the UPLOAD, not just the kernel --------------------------------
    // `ds4_build_q8_soa` is the plumbing `upload_dense` runs for every
    // requantised weight: the chunked row loop, the column gather for a kCols
    // split, and the two plane offsets inside one allocation.  NO GGUF ON THIS
    // BOX CARRIES A BF16 DENSE TENSOR — the miniature fixtures in
    // deepseek4_residency_test and deepseek4_forward_test store their dense
    // weights as Q8_0, mirroring UD-Q3_K_XL — so this section is the only thing
    // that executes it.  Without it the branch would ship with its arithmetic
    // checked by nothing, and a wrong scale-plane offset is a SILENT wrong
    // answer, not a crash.
    //
    // The check is exact: every element read back off the device must equal the
    // value the host quantiser produced for that element, and the row source
    // deliberately makes each row distinct so a row-offset error cannot pass.
    std::puts("\n§8 ds4_build_q8_soa — the chunked, column-sliced upload");
    {
        struct Case { const char* name; uint32_t K, K0, Ks, N0, Ns; };
        // The row chunk is (16 MB / K) rows, so K=8192 gives 2048 and Ns=5000
        // makes THREE chunks with a ragged last one.  That case is the reason
        // this section exists: `r0` offsets into BOTH planes are dead code at one
        // chunk, and a wrong one is silent.
        const Case cases[] = {
            {"mirrored small",        256,    0,  256,  0,   40},
            {"mirrored, 3 chunks",   8192,    0, 8192,  0, 5000},
            {"row slice (kRows)",     256,    0,  256, 24,   40},
            {"col slice (kCols) lo", 8192,    0, 4096,  0,   64},
            {"col slice (kCols) hi", 8192, 4096, 4096,  0,   64},
        };
        for (const Case& c : cases) {
            // The full [K, N] source, BF16-exact.  Element (r, k) is hashed from
            // BOTH indices so any row or column mis-offset changes the value.
            // A cheap splitmix rather than a per-element engine: the 3-chunk case
            // is 41 M elements and constructing an mt19937 for each would
            // dominate the whole test's runtime.
            const uint32_t Nfull = c.N0 + c.Ns;
            std::vector<float> full(size_t(Nfull) * c.K);
            for (uint32_t r = 0; r < Nfull; ++r)
                for (uint32_t k = 0; k < c.K; ++k) {
                    uint64_t z = uint64_t(r) * 0x9e3779b97f4a7c15ull + uint64_t(k) + 1;
                    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
                    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
                    z ^= z >> 31;
                    full[size_t(r) * c.K + k] =
                        to_bf16(float(int64_t(z % 20001) - 10000) * 1e-5f);
                }

            ie::Ds4Dense d;
            uint64_t bytes = 0;
            uint32_t chunks = 0;
            const std::string e = ie::ds4_build_q8_soa(
                q, c.K, c.K0, c.Ks, c.Ns, d, bytes,
                [&](uint32_t r0, uint32_t nr, std::vector<float>& out) -> std::string {
                    ++chunks;
                    out.resize(size_t(nr) * c.K);
                    std::memcpy(out.data(), full.data() + size_t(c.N0 + r0) * c.K,
                                size_t(nr) * c.K * sizeof(float));
                    return {};
                });
            check(e.empty(), e.empty() ? "" : e.c_str());
            if (!e.empty()) continue;

            // What the host says the answer is, computed from this card's slice.
            std::vector<float> mine(size_t(c.Ns) * c.Ks);
            for (uint32_t r = 0; r < c.Ns; ++r)
                std::memcpy(mine.data() + size_t(r) * c.Ks,
                            full.data() + size_t(c.N0 + r) * c.K + c.K0,
                            size_t(c.Ks) * sizeof(float));
            std::vector<int8_t> wqs;
            std::vector<sycl::half> wd;
            pack_q8(mine, c.Ks, c.Ns, wqs, wd);

            std::vector<int8_t> gqs(wqs.size());
            std::vector<sycl::half> gd(wd.size());
            q.memcpy(gqs.data(), d.q8, gqs.size()).wait();
            q.memcpy(gd.data(), d.q8d, gd.size() * sizeof(sycl::half)).wait();

            uint64_t bad_q = 0, bad_d = 0;
            for (size_t i = 0; i < wqs.size(); ++i) if (gqs[i] != wqs[i]) ++bad_q;
            for (size_t i = 0; i < wd.size(); ++i)
                if (float(gd[i]) != float(wd[i])) ++bad_d;

            std::printf("  %-22s K=%-5u[%u,+%u) rows[%u,+%u)  %u chunk(s)  %llu B"
                        "  int8 bad %llu / scales bad %llu\n",
                        c.name, c.K, c.K0, c.Ks, c.N0, c.Ns, chunks,
                        (unsigned long long)bytes, (unsigned long long)bad_q,
                        (unsigned long long)bad_d);
            check(bad_q == 0, "ds4_build_q8_soa uploaded the wrong int8 plane");
            check(bad_d == 0, "ds4_build_q8_soa uploaded the wrong scale plane");
            check(bytes == ie::ds4_dense_requant_bytes(c.Ks, c.Ns),
                  "ds4_build_q8_soa charged the wrong byte count");
            check(d.K == c.Ks && d.N == c.Ns && d.q8 && d.q8d && !d.w && !d.p,
                  "ds4_build_q8_soa left the Ds4Dense in the wrong shape");
            // The scale plane must sit exactly after the int8 plane in the SAME
            // allocation — that identity is what makes freeing only `q8` correct.
            check(reinterpret_cast<const int8_t*>(d.q8d) ==
                      d.q8 + uint64_t(c.Ks) * c.Ns,
                  "the scale plane is not at the end of the int8 plane");
            // The kb-major plane sits after the n-major one, and is its exact
            // transpose — orientation is what the oneDNN s8 route eats, so it
            // is gated element-for-element, not just for presence.
            const uint32_t nbs8 = c.Ks / 32;
            check(d.q8dt == d.q8d + uint64_t(c.Ns) * nbs8,
                  "the kb-major scale plane is not at the end of the n-major one");
            std::vector<sycl::half> gdt(size_t(c.Ns) * nbs8);
            q.memcpy(gdt.data(), d.q8dt, gdt.size() * sizeof(sycl::half)).wait();
            uint64_t bad_t = 0;
            for (uint32_t n = 0; n < c.Ns; ++n)
                for (uint32_t b = 0; b < nbs8; ++b)
                    if (float(gdt[size_t(b) * c.Ns + n]) != float(gd[size_t(n) * nbs8 + b]))
                        ++bad_t;
            check(bad_t == 0, "the kb-major scale plane is not the transpose of the n-major one");
            sycl::free(d.q8, q);
        }
        // A K slice that is not a whole number of blocks must be refused, not
        // rounded — the kCols guard in upload_dense depends on this.
        ie::Ds4Dense bad;
        uint64_t bb = 0;
        const std::string er = ie::ds4_build_q8_soa(
            q, 80, 0, 40, 4, bad, bb,
            [&](uint32_t, uint32_t nr, std::vector<float>& out) -> std::string {
                out.assign(size_t(nr) * 80, 0.f);
                return {};
            });
        std::printf("  ragged K slice 40 -> \"%s\"\n", er.c_str());
        check(!er.empty(), "ds4_build_q8_soa accepted a K slice that is not a whole block count");
        check(bad.q8 == nullptr, "ds4_build_q8_soa allocated before refusing");
    }

    if (g_fail) {
        std::printf("\nFAILED: %d check(s)\n", g_fail);
        return 1;
    }
    std::puts("\nOK — dense_f16, grouped_f16 and the requantised Q8_0-SoA route are within "
              "their derived bounds, and every mis-port control is caught at >=90% of sampled "
              "elements with >=10x median separation.");
    return 0;
}
