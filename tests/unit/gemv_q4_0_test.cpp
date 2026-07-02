// tests/unit/gemv_q4_0_test.cpp — correctness gate for the Q4_0 W4A16 GEMV/GEMM.
//
// Uses REAL Q4_0 weight tensors from the Gemma 4 26B-A4B QAT GGUF, dequants on
// the host with an inline reference (validates the byte/nibble layout too), then
// computes y_ref = A @ W in fp32. Compares to the GPU gemv_q4_0 (T=1) and
// gemm_q4_0 (T>1). This is the first Q4_0 brick — everything in the Gemma loader
// rides it. SKIPs cleanly if the GGUF isn't present.

#include "ie/gguf.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {
constexpr const char* G = "\033[32m";
constexpr const char* R = "\033[31m";
constexpr const char* Z = "\033[0m";

sycl::queue make_queue() {
    for (const auto& d : sycl::device::get_devices())
        if (d.is_gpu()) return sycl::queue(d);
    std::fputs("gemv_q4_0_test: no GPU — SKIP\n", stderr);
    std::exit(0);
}

// Inline Q4_0 host dequant of a [N, K]-packed buffer (each column = K/32 blocks
// of 18 bytes contiguous). w[n*K + b*32 + i]    = d*((qs[i]&0xF)-8);
//                          w[n*K + b*32 + i+16] = d*((qs[i]>>4) -8).
void dequant_q4_0_ref(const uint8_t* buf, uint32_t N, uint32_t K, std::vector<float>& out) {
    const uint32_t bpc = K / 32;
    out.resize(size_t(N) * K);
    for (uint32_t n = 0; n < N; ++n) {
        for (uint32_t b = 0; b < bpc; ++b) {
            const uint8_t* blk = buf + (size_t(n) * bpc + b) * 18;
            uint16_t dh; std::memcpy(&dh, blk, 2);
            const float d = ie::fp16_to_fp32(dh);
            const uint8_t* qs = blk + 2;
            for (uint32_t i = 0; i < 16; ++i) {
                out[size_t(n) * K + b * 32 + i]      = d * float(int(qs[i] & 0x0F) - 8);
                out[size_t(n) * K + b * 32 + i + 16] = d * float(int(qs[i] >> 4)   - 8);
            }
        }
    }
}

bool test_tensor(sycl::queue& q, ie::GgufReader& g, const char* name) {
    auto* t = g.find_tensor(name);
    if (!t) { std::printf("  %sSKIP%s  %s — not found\n", "\033[33m", Z, name); return true; }
    if (t->dtype != ie::DType::kQ4_0) { std::printf("  %sSKIP%s  %s — not Q4_0\n", "\033[33m", Z, name); return true; }
    const uint32_t K = uint32_t(t->shape[0]);   // inner/block dim
    const uint32_t N = uint32_t(t->shape[1]);   // output dim
    if (K % 32 != 0) { std::printf("  %sSKIP%s  %s — K%%32\n", "\033[33m", Z, name); return true; }

    std::vector<float> Wf;
    dequant_q4_0_ref(static_cast<const uint8_t*>(t->data), N, K, Wf);

    std::vector<sycl::half> hA(size_t(K) * 3);  // up to 3 rows for the gemm path
    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto& v : hA) v = sycl::half(dist(rng));

    auto* dA = sycl::malloc_device<sycl::half>(size_t(K) * 3, q);
    auto* dW = sycl::malloc_device(t->nbytes, q);
    auto* dY = sycl::malloc_device<sycl::half>(size_t(N) * 3, q);
    q.memcpy(dA, hA.data(), size_t(K) * 3 * sizeof(sycl::half)).wait();
    q.memcpy(dW, t->data, t->nbytes).wait();

    auto rel_check = [&](const char* tag, const sycl::half* hY, uint32_t row) -> bool {
        float max_rel = 0.f;
        for (uint32_t n = 0; n < N; ++n) {
            float acc = 0.f;
            const float* wcol = Wf.data() + size_t(n) * K;
            for (uint32_t k = 0; k < K; ++k) acc += float(hA[size_t(row) * K + k]) * wcol[k];
            const float got = float(hY[n]);
            const float r = std::fabs(acc - got) / std::fmax(std::fabs(acc), 1e-3f);
            if (r > max_rel) max_rel = r;
        }
        const bool ok = max_rel <= 0.02f;
        std::printf("  %s%-26s%s %-22s K=%u N=%u max_rel=%.5g  %s\n",
                    ok ? G : R, name, Z, tag, K, N, max_rel, ok ? "OK" : "FAIL");
        return ok;
    };

    bool ok = true;
    // gemv (T=1) on row 0
    ie::gemv_q4_0(q, dA, dW, dY, K, N).wait_and_throw();
    std::vector<sycl::half> hY(size_t(N) * 3);
    q.memcpy(hY.data(), dY, N * sizeof(sycl::half)).wait();
    ok &= rel_check("gemv", hY.data(), 0);

    // gemm (M=3): y[m,n] row-major
    ie::gemm_q4_0(q, dA, dW, dY, 3, K, N).wait_and_throw();
    q.memcpy(hY.data(), dY, size_t(N) * 3 * sizeof(sycl::half)).wait();
    for (uint32_t m = 0; m < 3; ++m) ok &= rel_check("gemm", hY.data() + size_t(m) * N, m);

    sycl::free(dA, q); sycl::free(dW, q); sycl::free(dY, q);
    return ok;
}
}  // namespace

int main(int argc, char** argv) {
    std::string gguf = "/home/weezy/models/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/gemma-4-26B_q4_0-it.gguf";
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--gguf" && i + 1 < argc) gguf = argv[++i];

    auto q = make_queue();
    std::printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    ie::GgufReader g;
    if (auto err = g.open(gguf); !err.empty()) {
        std::fprintf(stderr, "gemv_q4_0_test: gguf open failed (%s) — SKIP\n", err.c_str());
        return 0;  // SKIP if model absent
    }
    int fails = 0;
    if (!test_tensor(q, g, "blk.0.attn_q.weight"))      ++fails;  // [2816, 4096] swa
    if (!test_tensor(q, g, "blk.5.attn_q.weight"))      ++fails;  // [2816, 8192] global
    if (!test_tensor(q, g, "blk.0.ffn_down.weight"))    ++fails;  // [2112, 2816] shared FFN
    std::printf("\n%s\n", fails == 0 ? "ALL PASS" : "FAILURES");
    return fails;
}
