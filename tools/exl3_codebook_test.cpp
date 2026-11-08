// tools/exl3_codebook_test.cpp — the EXL3 codebooks (0 3INST, 1 mcg, 2 mul1) gate. No model file.
//
//  [F] every 16-bit code (65536) through each codebook: the device decode (exl3_decode_codes = the exl3_decode_dev the
//      GEMVs inline) == exl3_decode_code_host (exact double + one RNE fp16 rounding, exllamav3 codebook.cuh) bit for bit;
//  [G] per codebook, random trellis tiles at bits 2 / 4 / 6: gemv_exl3 and gemv_exl3_moe (one expert) against
//      y = A . W_rot in double, W_rot from exl3_decode_host (whose cb 0 branch is the shipped decode_cb0): max error
//      <= 2e-3 of max |y| (fp32 accumulation over K); moe == gemv bit for bit;
//  [H] per codebook, the whole linear (Hadamard + suh / svh around the trellis MAC): gemm_exl3 at T = 5 (the prefill path)
//      and gemv_exl3_forward row by row against the same wrap with the MAC done in double on the host: max error <= 4e-3
//      of max |y|, no non-finite value.
// Why: turboderp's Qwen3.8-Flash-Next EXL3 quants are codebook mul1 (config "codebook": "mul1"); the engine read cb 0 only.
//
// usage: ie-exl3-codebook-test   (exit 0 on GATE PASSED)
#include "ie/exl3.hpp"
#include "ie/ops.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "OK" : "FAIL", what);
    if (!ok) ++g_fail;
}
double h2d(uint16_t b) { sycl::half h; std::memcpy(&h, &b, 2); return double(float(h)); }
}  // namespace

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("\nEXL3 codebooks gate  device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    const char* cbn[3] = {"3INST", "mcg", "mul1"};

    // ---- [F] exhaustive decode ----
    std::printf("\n[F] every 16-bit code, device vs the exact host reference\n");
    std::vector<uint16_t> codes(65536);
    for (uint32_t i = 0; i < 65536; ++i) codes[i] = uint16_t(i);
    uint16_t* d_codes = sycl::malloc_device<uint16_t>(65536, q);
    sycl::half* d_out = sycl::malloc_device<sycl::half>(65536, q);
    q.memcpy(d_codes, codes.data(), 65536 * 2).wait();
    for (uint32_t cb = 0; cb < 3; ++cb) {
        ie::exl3_decode_codes(q, d_codes, 65536, cb, d_out).wait();
        std::vector<uint16_t> dev(65536);
        q.memcpy(dev.data(), d_out, 65536 * 2).wait();
        uint32_t bad = 0, first = 0;
        double lo = 1e9, hi = -1e9;
        for (uint32_t i = 0; i < 65536; ++i) {
            const uint16_t ref = ie::exl3_decode_code_host(i, cb);
            if (dev[i] != ref) { if (!bad) first = i; ++bad; }
            lo = std::min(lo, h2d(ref)); hi = std::max(hi, h2d(ref));
        }
        char what[128];
        std::snprintf(what, sizeof what, "cb %u (%s): device == host on all 65536 codes (%u differ%s; values in [%.3f, %.3f])",
                      cb, cbn[cb], bad, bad ? ", first shown below" : "", lo, hi);
        check(bad == 0, what);
        if (bad) std::printf("        code %u: device 0x%04x host 0x%04x\n", first, dev[first], ie::exl3_decode_code_host(first, cb));
    }

    // ---- [G] the GEMVs per codebook ----
    std::printf("\n[G] gemv_exl3 / gemv_exl3_moe vs y = A . W_rot (W_rot from exl3_decode_host)\n");
    const uint32_t K = 512, N = 256;
    std::mt19937 rng(2026);
    for (uint32_t cb = 0; cb < 3; ++cb) {
        for (uint32_t bits : {2u, 4u, 6u}) {
            ie::Exl3Tensor t;
            t.K = K; t.N = N; t.bits = bits; t.cb = cb; t.tile_k = K / 16; t.tile_n = N / 16;
            t.trellis.resize(size_t(t.tile_k) * t.tile_n * 16 * bits);
            for (auto& v : t.trellis) v = uint16_t(rng() & 0xffff);
            std::vector<float> W(size_t(K) * N);
            if (auto e = ie::exl3_decode_host(t, W.data()); !e.empty()) { std::printf("  host decode: %s\n", e.c_str()); ++g_fail; continue; }
            std::vector<sycl::half> A(K);
            for (auto& v : A) v = sycl::half(std::uniform_real_distribution<float>(-1.f, 1.f)(rng));
            std::vector<double> ref(N, 0.0);
            for (uint32_t k = 0; k < K; ++k)
                for (uint32_t n = 0; n < N; ++n) ref[n] += double(float(A[k])) * double(W[size_t(k) * N + n]);
            void* d_tr = sycl::malloc_device(t.trellis.size() * 2, q);
            sycl::half* d_a = sycl::malloc_device<sycl::half>(K, q);
            sycl::half* d_y = sycl::malloc_device<sycl::half>(N, q);
            int32_t* d_e = sycl::malloc_device<int32_t>(1, q);
            q.memcpy(d_tr, t.trellis.data(), t.trellis.size() * 2).wait();
            q.memcpy(d_a, A.data(), K * 2).wait();
            q.memset(d_e, 0, 4).wait();
            auto err_of = [&](const std::vector<sycl::half>& y) {
                double m = 0, scale = 0;
                for (uint32_t n = 0; n < N; ++n) scale = std::max(scale, std::fabs(ref[n]));
                for (uint32_t n = 0; n < N; ++n) m = std::max(m, std::fabs(double(float(y[n])) - ref[n]) / std::max(scale, 1e-6));
                return m;
            };
            std::vector<sycl::half> y(N), ym(N);
            ie::gemv_exl3(q, d_a, d_tr, d_y, K, N, bits, {}, cb).wait();
            q.memcpy(y.data(), d_y, N * 2).wait();
            ie::gemv_exl3_moe(q, d_a, d_tr, 0, d_e, d_y, K, N, 1, bits, {}, cb).wait();
            q.memcpy(ym.data(), d_y, N * 2).wait();
            const double eg = err_of(y), em = err_of(ym);
            char what[160];
            std::snprintf(what, sizeof what, "cb %u (%s) bits %u: gemv max err %.2e, moe max err %.2e (of max |y|), moe == gemv: %s",
                          cb, cbn[cb], bits, eg, em, std::memcmp(y.data(), ym.data(), N * 2) == 0 ? "bit-identical" : "differs");
            check(eg <= 2e-3 && em <= 2e-3, what);
            sycl::free(d_tr, q); sycl::free(d_a, q); sycl::free(d_y, q); sycl::free(d_e, q);
        }
    }
    // ---- [H] the full linear forward per codebook: gemm_exl3 (prefill, T rows) and gemv_exl3_forward (one row) ----
    // reference per row: xh = had128(A (.) suh) and y = had128(acc) (.) svh on the device (hadamard_transform, codebook-free),
    // acc = xh . W_rot in double on the host from exl3_decode_host -- so a kernel decoding with the wrong codebook fails here.
    std::printf("\n[H] gemm_exl3 (T=5) / gemv_exl3_forward vs had128(had128(A.suh) . W_rot).svh, W_rot from exl3_decode_host\n");
    {
        const uint32_t T = 5;
        for (uint32_t cb = 0; cb < 3; ++cb) {
            const uint32_t bits = 4;
            ie::Exl3Tensor t;
            t.K = K; t.N = N; t.bits = bits; t.cb = cb; t.tile_k = K / 16; t.tile_n = N / 16;
            t.trellis.resize(size_t(t.tile_k) * t.tile_n * 16 * bits);
            for (auto& v : t.trellis) v = uint16_t(rng() & 0xffff);
            std::vector<float> W(size_t(K) * N);
            if (auto e = ie::exl3_decode_host(t, W.data()); !e.empty()) { std::printf("  host decode: %s\n", e.c_str()); ++g_fail; continue; }
            std::uniform_real_distribution<float> sc(0.5f, 1.5f), ad(-1.f, 1.f);
            std::vector<sycl::half> A(size_t(T) * K), suh(K), svh(N);
            for (auto& v : A) v = sycl::half(ad(rng));
            for (auto& v : suh) v = sycl::half((rng() & 1) ? sc(rng) : -sc(rng));
            for (auto& v : svh) v = sycl::half((rng() & 1) ? sc(rng) : -sc(rng));
            void* d_tr = sycl::malloc_device(t.trellis.size() * 2, q);
            sycl::half* d_a = sycl::malloc_device<sycl::half>(size_t(T) * K, q);
            sycl::half* d_suh = sycl::malloc_device<sycl::half>(K, q);
            sycl::half* d_svh = sycl::malloc_device<sycl::half>(N, q);
            sycl::half* d_xh = sycl::malloc_device<sycl::half>(size_t(T) * K, q);
            sycl::half* d_acc = sycl::malloc_device<sycl::half>(size_t(T) * N, q);
            sycl::half* d_ref = sycl::malloc_device<sycl::half>(size_t(T) * N, q);
            sycl::half* d_y = sycl::malloc_device<sycl::half>(size_t(T) * N, q);
            sycl::half* d_y1 = sycl::malloc_device<sycl::half>(size_t(T) * N, q);
            q.memcpy(d_tr, t.trellis.data(), t.trellis.size() * 2).wait();
            q.memcpy(d_a, A.data(), A.size() * 2).wait();
            q.memcpy(d_suh, suh.data(), K * 2).wait();
            q.memcpy(d_svh, svh.data(), N * 2).wait();
            for (uint32_t r = 0; r < T; ++r) ie::hadamard_transform(q, d_a + size_t(r) * K, d_xh + size_t(r) * K, K, d_suh, nullptr).wait();
            std::vector<sycl::half> xh(size_t(T) * K), acc(size_t(T) * N);
            q.memcpy(xh.data(), d_xh, xh.size() * 2).wait();
            for (uint32_t r = 0; r < T; ++r)
                for (uint32_t n = 0; n < N; ++n) {
                    double s = 0.0;
                    for (uint32_t k = 0; k < K; ++k) s += double(float(xh[size_t(r) * K + k])) * double(W[size_t(k) * N + n]);
                    const _Float16 h16 = static_cast<_Float16>(s);   // RNE (host sycl::half(float) truncates tiny values)
                    std::memcpy(&acc[size_t(r) * N + n], &h16, 2);
                }
            q.memcpy(d_acc, acc.data(), acc.size() * 2).wait();
            for (uint32_t r = 0; r < T; ++r) ie::hadamard_transform(q, d_acc + size_t(r) * N, d_ref + size_t(r) * N, N, nullptr, d_svh).wait();
            ie::gemm_exl3(q, d_a, d_tr, d_suh, d_svh, d_y, T, K, N, bits, {}, cb).wait();
            for (uint32_t r = 0; r < T; ++r)
                ie::gemv_exl3_forward(q, d_a + size_t(r) * K, d_tr, d_suh, d_svh, d_y1 + size_t(r) * N, K, N, bits, {}, cb).wait();
            std::vector<sycl::half> ref(size_t(T) * N), yg(size_t(T) * N), yf(size_t(T) * N);
            q.memcpy(ref.data(), d_ref, ref.size() * 2).wait();
            q.memcpy(yg.data(), d_y, yg.size() * 2).wait();
            q.memcpy(yf.data(), d_y1, yf.size() * 2).wait();
            // a NaN would vanish in std::max: count the non-finite values and fail on any (and on an all-zero reference)
            double scale = 0, eg = 0, ef = 0;
            uint32_t nonfinite = 0;
            for (size_t i = 0; i < ref.size(); ++i) {
                const double r = double(float(ref[i])), g = double(float(yg[i])), f = double(float(yf[i]));
                if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(f)) { ++nonfinite; continue; }
                scale = std::max(scale, std::fabs(r));
            }
            for (size_t i = 0; i < ref.size(); ++i) {
                const double r = double(float(ref[i])), g = double(float(yg[i])), f = double(float(yf[i]));
                if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(f)) continue;
                eg = std::max(eg, std::fabs(g - r) / std::max(scale, 1e-6));
                ef = std::max(ef, std::fabs(f - r) / std::max(scale, 1e-6));
            }
            char what[200];
            std::snprintf(what, sizeof what, "cb %u (%s) bits %u: gemm_exl3 max err %.2e, gemv_exl3_forward max err %.2e (of max |y| %.3g), "
                          "non-finite %u", cb, cbn[cb], bits, eg, ef, scale, nonfinite);
            check(nonfinite == 0 && scale > 0 && eg <= 4e-3 && ef <= 4e-3, what);
            for (void* p : {d_tr, (void*)d_a, (void*)d_suh, (void*)d_svh, (void*)d_xh, (void*)d_acc, (void*)d_ref, (void*)d_y, (void*)d_y1})
                sycl::free(p, q);
        }
    }

    sycl::free(d_codes, q); sycl::free(d_out, q);
    std::printf("\n%s  (%d failure%s)\n\n", g_fail ? "GATE FAILED" : "GATE PASSED", g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
