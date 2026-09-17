// tests/unit/deepseek4_decgemv_gate_test.cpp — numerical gate for the decode
// (M=1) GEMV in src/ops/ds4_decode_gemv.cpp.
//
// WHAT IS BEING GATED
// -------------------
// `ds4_decode_gemv_f16` / `ds4_decode_gemv_grouped_f16` replace the oneDNN
// matmul on the T=1 dense projections.  They are a NEW summation order (per-lane
// serial fma chain -> 16-lane sub-group tree -> optional cross-sub-group fixup),
// so they are NOT bit-identical to `dense_f16` and cannot be gated by equality
// against it.  They are gated against the definition of the operation instead.
//
// GROUND TRUTH
// ------------
// The exact dot product accumulated in `double` on the host from the SAME fp16
// weight bytes and the SAME fp32 activations handed to the device.  Not a second
// implementation that could share a bug: the definition, evaluated at ~2^29
// times the precision of the thing under test.
//
// THE BOUND, AND WHERE IT COMES FROM
// ----------------------------------
// Every term enters through one `sycl::fma`, so one fp32 rounding each; the
// 16-lane `reduce_over_group` adds 4 tree levels; SGS>1 adds SGS-1 serial adds.
// The longest chain of roundings reaching an output element is therefore
//
//     p = ceil(K / (SGS*16)) + 4 + (SGS - 1)
//
// and the standard fp32 summation bound gives
//
//     |y_kernel - y_exact| <= gamma_p * SUM_k |w[n,k] * x[k]|,
//     gamma_p = p*u / (1 - p*u),   u = 2^-24 = 5.96e-8.
//
// p is NOT written down here.  It is read from
// `ie::ds4_decode_gemv_rounding_steps(K, N)`, which is defined next to the
// kernel from the same `sgs_for(N)` the kernel dispatches on.  A future change
// to the sub-groups-per-row rule therefore moves the bound with it and cannot
// silently invalidate this gate — and the bound cannot be widened here without
// changing the kernel's actual reduction shape.
//
// NEGATIVE CONTROLS
// -----------------
// A bound is worth nothing unless a plausible bug exceeds it.  Four are checked,
// and every one must FAIL the bound above:
//   1. transposed weight — reading W[k*N + n] instead of W[n*K + k].  This is
//      the specific hazard of this repository: `gemv_fp16` (every OTHER model)
//      consumes fp16 [K, N] because its loader transposes at upload, while the
//      DS4 path (`dense_f16`, `gemm_nt_f16_onednn`, and this kernel) consumes
//      fp16 [N, K] untransposed.  Checked on a SQUARE shape, where a transpose
//      stays in bounds and would otherwise look plausible.
//   2. fp16-rounded activation — exactly what the oneDNN route this kernel
//      replaces does to the activation before multiplying.  It must exceed the
//      fp32 bound, which is what makes the claim "the GEMV is strictly more
//      accurate than the route it replaces" a measurement rather than an
//      assertion.
//   3. one dropped weight element.
//   4. grouped: the group offset dropped, so every group reads block 0 — the
//      mistake the block-diagonal indexing invites.
#undef NDEBUG  // build is Release (-DNDEBUG); asserts must stay live here
#include "ie/ds4_decode_gemv.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

constexpr double kU = 5.9604644775390625e-08;  // 2^-24, fp32 unit roundoff

int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::printf("  FAIL: %s\n", what); ++g_fail; }
}

double gamma_of(uint32_t p) {
    const double pu = double(p) * kU;
    return pu / (1.0 - pu);
}

// Exact value and SUM|w*x| for one output row of the dense form.
struct Ref { double y, mag; };

Ref ref_dense(const std::vector<sycl::half>& w, const std::vector<float>& x,
              uint32_t n, uint32_t K) {
    Ref r{0.0, 0.0};
    const sycl::half* wr = w.data() + uint64_t(n) * K;
    for (uint32_t k = 0; k < K; ++k) {
        const double p = double(float(wr[k])) * double(x[k]);
        r.y += p; r.mag += std::fabs(p);
    }
    return r;
}

// Block-diagonal form: row n reads activation block (n / OPG).
Ref ref_grouped(const std::vector<sycl::half>& w, const std::vector<float>& x,
                uint32_t n, uint32_t IPG, uint32_t OPG) {
    Ref r{0.0, 0.0};
    const sycl::half* wr = w.data() + uint64_t(n) * IPG;
    const float* xr = x.data() + uint64_t(n / OPG) * IPG;
    for (uint32_t k = 0; k < IPG; ++k) {
        const double p = double(float(wr[k])) * double(xr[k]);
        r.y += p; r.mag += std::fabs(p);
    }
    return r;
}

// Boundary rows first (row 0, last row, first row of the final work-group),
// then a random sample.
std::vector<uint32_t> sample_rows(uint32_t N, uint32_t want, std::mt19937& rng) {
    std::vector<uint32_t> rows;
    if (N <= want) { for (uint32_t n = 0; n < N; ++n) rows.push_back(n); return rows; }
    rows = {0u, N - 1u, (N / 16u) * 16u};
    if (rows.back() >= N) rows.back() = N - 1u;
    std::uniform_int_distribution<uint32_t> d(0, N - 1);
    while (rows.size() < want) rows.push_back(d(rng));
    return rows;
}

struct Case { const char* name; uint32_t K, N; };

// ---------------------------------------------------------------- dense gate
void gate_dense(sycl::queue& q, const Case& c, std::mt19937& rng, bool full) {
    const uint64_t we = uint64_t(c.K) * c.N;
    std::vector<sycl::half> hw(we);
    std::vector<float> hx(c.K);
    std::uniform_real_distribution<float> dw(-0.09f, 0.09f), dx(-1.5f, 1.5f);
    for (auto& v : hw) v = sycl::half(dw(rng));
    for (auto& v : hx) v = dx(rng);

    sycl::half* w = sycl::malloc_device<sycl::half>(we, q);
    float* x = sycl::malloc_device<float>(c.K, q);
    float* y = sycl::malloc_device<float>(c.N, q);
    assert(w && x && y);
    q.memcpy(w, hw.data(), we * 2).wait();
    q.memcpy(x, hx.data(), c.K * 4).wait();
    q.memset(y, 0xff, c.N * 4).wait();   // poison: an unwritten row shows as NaN

    ie::ds4_decode_gemv_f16(q, "gate", x, w, y, c.K, c.N).wait();
    std::vector<float> hy(c.N);
    q.memcpy(hy.data(), y, c.N * 4).wait();

    const uint32_t p = ie::ds4_decode_gemv_rounding_steps(c.K, c.N);
    const double gam = gamma_of(p);

    const auto rows = full ? sample_rows(c.N, c.N, rng) : sample_rows(c.N, 96, rng);
    double worst_ratio = 0.0;
    uint32_t nbad = 0, nnan = 0;
    for (uint32_t n : rows) {
        const Ref r = ref_dense(hw, hx, n, c.K);
        if (!std::isfinite(hy[n])) { ++nnan; continue; }
        const double err = std::fabs(double(hy[n]) - r.y);
        if (!(err <= gam * r.mag)) ++nbad;
        if (r.mag > 0) worst_ratio = std::max(worst_ratio, err / r.mag);
    }
    std::printf("  %-14s K=%-5u N=%-6u rows=%-5zu p=%-5u observed %.3e   bound %.3e"
                "   headroom %5.1fx  %s\n",
                c.name, c.K, c.N, rows.size(), p, worst_ratio, gam,
                worst_ratio > 0 ? gam / worst_ratio : 0.0,
                (nbad || nnan) ? "<-- OVER" : "ok");
    check(nbad == 0, c.name);
    check(nnan == 0, "no output row left unwritten");

    sycl::free(w, q); sycl::free(x, q); sycl::free(y, q);
}

// -------------------------------------------------------------- grouped gate
void gate_grouped(sycl::queue& q, uint32_t G, uint32_t IPG, uint32_t OPG, std::mt19937& rng) {
    const uint32_t N = G * OPG;
    const uint64_t we = uint64_t(N) * IPG;
    std::vector<sycl::half> hw(we);
    std::vector<float> hx(uint64_t(G) * IPG);
    std::uniform_real_distribution<float> dw(-0.09f, 0.09f), dx(-1.5f, 1.5f);
    for (auto& v : hw) v = sycl::half(dw(rng));
    for (auto& v : hx) v = dx(rng);

    sycl::half* w = sycl::malloc_device<sycl::half>(we, q);
    float* x = sycl::malloc_device<float>(hx.size(), q);
    float* y = sycl::malloc_device<float>(N, q);
    assert(w && x && y);
    q.memcpy(w, hw.data(), we * 2).wait();
    q.memcpy(x, hx.data(), hx.size() * 4).wait();
    q.memset(y, 0xff, N * 4).wait();

    ie::ds4_decode_gemv_grouped_f16(q, "gate", x, w, y, G, IPG, OPG).wait();
    std::vector<float> hy(N);
    q.memcpy(hy.data(), y, N * 4).wait();

    const uint32_t p = ie::ds4_decode_gemv_rounding_steps(IPG, N);
    const double gam = gamma_of(p);
    const auto rows = sample_rows(N, 96, rng);
    double worst = 0.0, worst_drop = 0.0;
    uint32_t nbad = 0;
    for (uint32_t n : rows) {
        const Ref r = ref_grouped(hw, hx, n, IPG, OPG);
        const double err = std::fabs(double(hy[n]) - r.y);
        if (!(err <= gam * r.mag)) ++nbad;
        if (r.mag > 0) worst = std::max(worst, err / r.mag);
        // negative control, same rows: group offset dropped -> every group would
        // read activation block 0.  Rows in group 0 are unaffected, so skip them.
        if (n >= OPG && r.mag > 0) {
            double y0 = 0.0;
            const sycl::half* wr = hw.data() + uint64_t(n) * IPG;
            for (uint32_t k = 0; k < IPG; ++k) y0 += double(float(wr[k])) * double(hx[k]);
            worst_drop = std::max(worst_drop, std::fabs(y0 - r.y) / r.mag);
        }
    }
    std::printf("  %-14s G=%u IPG=%-5u OPG=%-5u rows=%-4zu p=%-5u observed %.3e   bound %.3e"
                "   headroom %5.1fx  %s\n",
                "grouped", G, IPG, OPG, rows.size(), p, worst, gam,
                worst > 0 ? gam / worst : 0.0, nbad ? "<-- OVER" : "ok");
    check(nbad == 0, "grouped");
    std::printf("    neg-control group-offset dropped: %.3e vs bound %.3e -> %s\n",
                worst_drop, gam, worst_drop > gam ? "CAUGHT" : "MISSED");
    check(worst_drop > gam, "neg-control: grouped group offset dropped must exceed the bound");

    sycl::free(w, q); sycl::free(x, q); sycl::free(y, q);
}

// ----------------------------------------------------------- negative controls
// Square shape so a weight transpose stays in bounds and looks plausible.
void neg_controls(sycl::queue& q, std::mt19937& rng) {
    const uint32_t K = 1024, N = 1024;
    const uint64_t we = uint64_t(K) * N;
    std::vector<sycl::half> hw(we);
    std::vector<float> hx(K);
    std::uniform_real_distribution<float> dw(-0.09f, 0.09f), dx(-1.5f, 1.5f);
    for (auto& v : hw) v = sycl::half(dw(rng));
    for (auto& v : hx) v = dx(rng);

    sycl::half* w = sycl::malloc_device<sycl::half>(we, q);
    float* x = sycl::malloc_device<float>(K, q);
    float* y = sycl::malloc_device<float>(N, q);
    assert(w && x && y);
    q.memcpy(w, hw.data(), we * 2).wait();
    q.memcpy(x, hx.data(), K * 4).wait();
    ie::ds4_decode_gemv_f16(q, "gate", x, w, y, K, N).wait();
    std::vector<float> hy(N);
    q.memcpy(hy.data(), y, N * 4).wait();

    const uint32_t p = ie::ds4_decode_gemv_rounding_steps(K, N);
    const double gam = gamma_of(p);
    const auto rows = sample_rows(N, 64, rng);

    double worst_T = 0.0, worst_f16 = 0.0, worst_drop = 0.0, worst_real = 0.0;
    for (uint32_t n : rows) {
        const Ref r = ref_dense(hw, hx, n, K);
        if (r.mag <= 0) continue;
        double yT = 0.0, y16 = 0.0;
        for (uint32_t k = 0; k < K; ++k) {
            yT  += double(float(hw[uint64_t(k) * N + n])) * double(hx[k]);
            y16 += double(float(hw[uint64_t(n) * K + k])) * double(float(sycl::half(hx[k])));
        }
        worst_T    = std::max(worst_T,    std::fabs(yT  - r.y) / r.mag);
        worst_f16  = std::max(worst_f16,  std::fabs(y16 - r.y) / r.mag);
        worst_drop = std::max(worst_drop,
            std::fabs(double(float(hw[uint64_t(n) * K + K / 2])) * double(hx[K / 2])) / r.mag);
        worst_real = std::max(worst_real, std::fabs(double(hy[n]) - r.y) / r.mag);
    }
    std::printf("\nNEGATIVE CONTROLS at K=N=%u  (bound %.3e, p=%u)\n", K, gam, p);
    std::printf("  transposed weight W[k*N+n]      : %.3e -> %s\n",
                worst_T, worst_T > gam ? "CAUGHT" : "MISSED");
    std::printf("  fp16-rounded activation (oneDNN): %.3e -> %s   (%.0fx the fp32 bound)\n",
                worst_f16, worst_f16 > gam ? "CAUGHT" : "MISSED", worst_f16 / gam);
    std::printf("  one weight element dropped      : %.3e -> %s\n",
                worst_drop, worst_drop > gam ? "CAUGHT" : "MISSED");
    check(worst_T    > gam, "neg-control: transposed weight must exceed the bound");
    check(worst_f16  > gam, "neg-control: fp16 activation rounding must exceed the fp32 bound");
    check(worst_drop > gam, "neg-control: a dropped term must exceed the bound");

    // The accuracy claim the routing change rests on, measured rather than
    // asserted: this kernel keeps the activation in fp32, the route it replaces
    // does not.
    std::printf("  this kernel (fp32 activation)   : %.3e -> %.0fx more accurate than the\n"
                "                                                 fp16-activation route it replaces\n",
                worst_real, worst_real > 0 ? worst_f16 / worst_real : 0.0);
    check(worst_real < worst_f16, "kernel must be more accurate than the fp16-activation route");

    sycl::free(w, q); sycl::free(x, q); sycl::free(y, q);
}

// ------------------------------------------------------- multi-projection gate
//
// `ds4_decode_gemv_multi` makes a DIFFERENT kind of claim from the kernels
// above.  They claim to approximate a dot product within a derived bound; this
// one claims to be THE SAME COMPUTATION as the per-projection calls, issued in
// fewer launches.  A tolerance would not gate that claim at all — a fused
// kernel that quietly changed the sub-groups-per-row of one projection would
// still sit inside the same gamma_p envelope while producing different bits and
// a different reduction shape from the one `ds4_decode_gemv_rounding_steps`
// reports.  So this is EQUALITY, on every output element of every projection.
//
// It also gates the PACKING, which is the part with room for an off-by-one: the
// op concatenates each projection's work-group range into one nd_range and every
// work-group recovers its projection from `wg0`.  Getting that wrong writes one
// projection's rows into another's buffer, so the outputs are poisoned first and
// any unwritten element is reported.
//
// Negative controls: the packer must be shown to be doing something a naive
// implementation would get wrong, so the same list is also submitted with the
// projections REORDERED (which changes every work-group's projection assignment
// but must not change any output) and with one output buffer swapped (which
// must change them).
void gate_multi(sycl::queue& q, const char* label, uint32_t K,
                const std::vector<uint32_t>& Ns, std::mt19937& rng) {
    const uint32_t n = uint32_t(Ns.size());
    std::vector<std::vector<sycl::half>> hw(n);
    std::vector<sycl::half*> w(n);
    std::vector<float*> ysep(n), ymul(n);
    std::uniform_real_distribution<float> dw(-0.09f, 0.09f), dx(-1.5f, 1.5f);
    for (uint32_t j = 0; j < n; ++j) {
        const uint64_t ne = uint64_t(K) * Ns[j];
        hw[j].resize(ne);
        for (auto& v : hw[j]) v = sycl::half(dw(rng));
        w[j] = sycl::malloc_device<sycl::half>(ne, q);
        assert(w[j]);
        q.memcpy(w[j], hw[j].data(), ne * 2).wait();
        ysep[j] = sycl::malloc_device<float>(Ns[j], q);
        ymul[j] = sycl::malloc_device<float>(Ns[j], q);
        assert(ysep[j] && ymul[j]);
        q.memset(ysep[j], 0xff, Ns[j] * 4).wait();   // poison
        q.memset(ymul[j], 0xff, Ns[j] * 4).wait();
    }
    std::vector<float> hx(K);
    for (auto& v : hx) v = dx(rng);
    float* x = sycl::malloc_device<float>(K, q);
    assert(x);
    q.memcpy(x, hx.data(), K * 4).wait();

    for (uint32_t j = 0; j < n; ++j)
        ie::ds4_decode_gemv_f16(q, "sep", x, w[j], ysep[j], K, Ns[j]);
    std::vector<ie::Ds4GemvProj> pr(n);
    for (uint32_t j = 0; j < n; ++j) { pr[j] = {}; pr[j].w = w[j]; pr[j].y = ymul[j]; pr[j].N = Ns[j]; }
    ie::ds4_decode_gemv_multi(q, "multi", x, pr.data(), n, K);
    q.wait();

    auto readback = [&](const std::vector<float*>& d) {
        std::vector<std::vector<float>> h(n);
        for (uint32_t j = 0; j < n; ++j) {
            h[j].resize(Ns[j]);
            q.memcpy(h[j].data(), d[j], Ns[j] * 4).wait();
        }
        return h;
    };
    auto count_diff = [&](const std::vector<std::vector<float>>& a,
                          const std::vector<std::vector<float>>& b) {
        uint64_t nd = 0;
        for (uint32_t j = 0; j < n; ++j)
            for (uint32_t i = 0; i < Ns[j]; ++i)
                if (sycl::bit_cast<uint32_t>(a[j][i]) != sycl::bit_cast<uint32_t>(b[j][i])) ++nd;
        return nd;
    };

    const auto A = readback(ysep), B = readback(ymul);
    uint64_t ntot = 0, nnan = 0;
    for (uint32_t j = 0; j < n; ++j) {
        ntot += Ns[j];
        for (uint32_t i = 0; i < Ns[j]; ++i) if (!std::isfinite(B[j][i])) ++nnan;
    }
    const uint64_t nd = count_diff(A, B);
    std::printf("  %-16s K=%-5u n=%u  %llu/%llu elements differ  unwritten %llu  %s\n",
                label, K, n, (unsigned long long)nd, (unsigned long long)ntot,
                (unsigned long long)nnan, (nd || nnan) ? "<-- NOT EQUAL" : "ok");
    check(nd == 0, label);
    check(nnan == 0, "multi: no output element left unwritten");

    // Reorder: the packing changes completely, the arithmetic must not.
    if (n > 1) {
        for (uint32_t j = 0; j < n; ++j) q.memset(ymul[j], 0xff, Ns[j] * 4).wait();
        std::vector<ie::Ds4GemvProj> rev(pr.rbegin(), pr.rend());
        ie::ds4_decode_gemv_multi(q, "multi-rev", x, rev.data(), n, K);
        q.wait();
        const uint64_t nd2 = count_diff(A, readback(ymul));
        std::printf("    reordered list           : %llu differ -> %s\n",
                    (unsigned long long)nd2, nd2 == 0 ? "ok" : "<-- NOT EQUAL");
        check(nd2 == 0, "multi: reordering the list must not change any output");

        // Negative control on the same comparison: shift one projection's weight
        // base by a row.  That is exactly what a packing off-by-one produces —
        // every row of that projection reads its neighbour's weights — and it
        // has to be visible to the equality check above, or the check is not a
        // gate.  `N` is reduced with the base so the read stays IN BOUNDS: a
        // control that walks off the end of the allocation would be testing the
        // driver, not the packer.
        if (Ns[0] >= 2) {
            for (uint32_t j = 0; j < n; ++j) q.memset(ymul[j], 0xff, Ns[j] * 4).wait();
            std::vector<ie::Ds4GemvProj> bad(pr);
            bad[0].w = w[0] + uint64_t(K);
            bad[0].N = Ns[0] - 1;
            ie::ds4_decode_gemv_multi(q, "multi-bad", x, bad.data(), n, K);
            q.wait();
            const uint64_t nd3 = count_diff(A, readback(ymul));
            std::printf("    neg-control weight base off by one row: %llu differ -> %s\n",
                        (unsigned long long)nd3, nd3 ? "CAUGHT" : "MISSED");
            check(nd3 > 0, "neg-control: a shifted weight base must change the output");
        }
    }

    for (uint32_t j = 0; j < n; ++j) { sycl::free(w[j], q); sycl::free(ysep[j], q); sycl::free(ymul[j], q); }
    sycl::free(x, q);
}

}  // namespace

int main() {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::printf("deepseek4_decgemv_gate_test — device: %s\n\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());
    std::mt19937 rng(20260803);

    // The real per-card DeepSeek-V4-Flash decode shapes (K = gguf shape[0],
    // N = gguf shape[1], with the 2-card row/column split applied).
    std::printf("REAL PER-CARD DECODE SHAPES\n");
    const Case real_shapes[] = {
        {"q_a",           4096, 1024},
        {"q_b",           1024, 16384},
        {"kv",            4096, 512},
        {"o_b",           4096, 4096},    // column-split, K = 8192/2
        {"comp_kv/gate",  4096, 1024},
        {"comp_512",      4096, 512},
        {"idx_q_b",       1024, 8192},
        {"idx_proj",      4096, 64},      // N < 128  -> SGS=8
        {"idx_comp",      4096, 256},     // N < 512  -> SGS=4
        {"shexp_gate/up", 4096, 1024},
        {"shexp_down",    1024, 4096},    // column-split, K = 2048/2
        {"lm_head",       4096, 129280},
    };
    for (const auto& c : real_shapes) gate_dense(q, c, rng, /*full=*/c.N <= 256);

    std::printf("\nEDGE SHAPES (tail, scalar and partial-work-group paths)\n");
    const Case edges[] = {
        {"K%16!=0",       1000, 512},   // -> scalar V=1 path
        {"K tail",         272, 512},   // K%16==0 but not a whole step (SGS=1: 256)
        {"K<step",         128, 1024},  // no vector iteration at all
        {"N%16!=0",       4096, 1000},  // partial final work-group
        {"N=1",           4096, 1},     // SGS=8, single row
        {"N=17",          4096, 17},
        {"K=16",            16, 4096},
        {"tiny",            32, 33},
    };
    for (const auto& c : edges) gate_dense(q, c, rng, /*full=*/true);

    std::printf("\nGROUPED (block-diagonal)\n");
    gate_grouped(q, 4, 4096, 1024, rng);   // real per-card attn_output_a
    gate_grouped(q, 2, 128, 64, rng);      // small: exercises SGS>1 with grouping

    // ---------------------------------------------------------------- multi
    // The real per-card sets that share one activation and one K, plus edges
    // that exercise the packer: a single element (must degenerate to the plain
    // entry point), a run past the descriptor-table bound (must chunk), mixed
    // sub-groups-per-row, and a run that is not a whole number of waves.
    std::printf("\nMULTI-PROJECTION (one activation, one launch) — BIT EQUALITY\n");
    gate_multi(q, "attn CSA",   4096, {1024, 512, 1024, 1024, 64}, rng);
    gate_multi(q, "attn HCA",   4096, {1024, 512, 512, 512}, rng);
    gate_multi(q, "indexer",    4096, {256, 256}, rng);
    gate_multi(q, "shexp",      4096, {1024, 1024}, rng);
    gate_multi(q, "qres pair",  1024, {16384, 8192}, rng);
    gate_multi(q, "n=1",        4096, {512}, rng);
    gate_multi(q, "over-max",   4096, {64, 64, 64, 64, 64, 64, 64, 64, 64, 64}, rng);
    gate_multi(q, "mixed sgs",  4096, {64, 256, 512, 1024}, rng);
    gate_multi(q, "odd N",      4096, {1000, 17, 1}, rng);
    gate_multi(q, "scalar K",   1000, {512, 256}, rng);
    gate_multi(q, "K<step",      128, {1024, 64}, rng);

    neg_controls(q, rng);

    std::printf("\n%s\n", g_fail ? "FAILED" : "PASS");
    return g_fail ? 1 : 0;
}
