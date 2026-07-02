// tests/mxfp4_dequant_test.cpp — correctness gate for the MXFP4 dequant kernels
// (gpt-oss MoE expert format). Validates dequant_mxfp4 (contiguous) and
// dequant_mxfp4_to_Bt (transposed [K,N]) against a host reference that uses the
// SAME E8M0-half + 16-value FP4 LUT math as llama's ggml_e8m0_to_fp32_half +
// kvalues_mxfp4. No GGUF: synthetic blocks covering all 16 LUT codes and a
// spread of exponents (incl. the e<2 denormal patterns). Both kernels run on the
// caller's GPU queue; the only allowed divergence is the final fp16 store, which
// is identical between kernel and reference (same float → fp16 round).

#include "ie/dequant.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr const char* G = "\033[32m";
constexpr const char* R = "\033[31m";
constexpr const char* Z = "\033[0m";

sycl::queue make_queue() {
    sycl::device dev;
    bool found = false;
    for (const auto& d : sycl::device::get_devices())
        if (d.is_gpu()) { dev = d; found = true; break; }
    if (!found) { std::fputs("no GPU\n", stderr); std::exit(2); }
    return sycl::queue(dev);
}

// Host mirror of dev_e8m0_half / kvalues_mxfp4 (the llama reference).
float host_e8m0_half(uint8_t e) {
    uint32_t bits = (e < 2u) ? (0x00200000u << e) : (uint32_t(e - 1u) << 23);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
const int kLUT[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

ie::block_mxfp4 make_block(uint8_t e, uint32_t seed) {
    ie::block_mxfp4 b;
    b.e = e;
    for (int j = 0; j < 16; ++j) {
        // pack two LUT codes per byte; sweep all 16 codes across blocks
        uint8_t lo = uint8_t((seed + j) & 0x0F);
        uint8_t hi = uint8_t((seed + j + 7) & 0x0F);
        b.qs[j] = uint8_t(lo | (hi << 4));
    }
    return b;
}

// host dequant of one block → 32 floats (then fp16-rounded to match the kernel)
void ref_block(const ie::block_mxfp4& b, sycl::half out[32]) {
    const float d = host_e8m0_half(b.e);
    for (int j = 0; j < 16; ++j) {
        out[j]      = sycl::half(d * float(kLUT[b.qs[j] & 0x0F]));
        out[j + 16] = sycl::half(d * float(kLUT[b.qs[j] >> 4]));
    }
}

}  // namespace

int main() {
    sycl::queue q = make_queue();
    int fails = 0;

    // Exponents spanning the denormal (e<2) patterns + a normalized spread.
    const std::vector<uint8_t> exps = {0, 1, 2, 3, 64, 126, 127, 128, 200, 254, 255};

    // ---- Test 1: dequant_mxfp4 (contiguous [n_elements] fp16) ----
    {
        const size_t NB = exps.size();
        std::vector<ie::block_mxfp4> blocks(NB);
        for (size_t i = 0; i < NB; ++i) blocks[i] = make_block(exps[i], uint32_t(i * 3 + 1));
        const size_t n_elem = NB * 32;

        auto* d_in  = sycl::malloc_device<ie::block_mxfp4>(NB, q);
        auto* d_out = sycl::malloc_device<sycl::half>(n_elem, q);
        q.memcpy(d_in, blocks.data(), NB * sizeof(ie::block_mxfp4)).wait();
        ie::dequant_mxfp4(q, d_in, d_out, n_elem, {}).wait();
        std::vector<sycl::half> got(n_elem);
        q.memcpy(got.data(), d_out, n_elem * sizeof(sycl::half)).wait();

        size_t bad = 0;
        for (size_t i = 0; i < NB; ++i) {
            sycl::half ref[32];
            ref_block(blocks[i], ref);
            for (int k = 0; k < 32; ++k)
                if (float(got[i * 32 + k]) != float(ref[k])) ++bad;
        }
        sycl::free(d_in, q);
        sycl::free(d_out, q);
        if (bad) { std::printf("%sFAIL%s dequant_mxfp4: %zu/%zu elems differ\n", R, Z, bad, n_elem); ++fails; }
        else     { std::printf("%sOK%s   dequant_mxfp4: %zu elems bit-exact vs ref\n", G, Z, n_elem); }
    }

    // ---- Test 2: dequant_mxfp4_to_Bt (transposed [K,N]) ----
    {
        const uint32_t K = 128;   // %32==0
        const uint32_t N = 64;    // %64==0
        const uint32_t bpc = K / 32;
        // Input: N columns, each K/32 blocks along K → W[n*bpc + b].
        std::vector<ie::block_mxfp4> W(size_t(N) * bpc);
        for (uint32_t n = 0; n < N; ++n)
            for (uint32_t b = 0; b < bpc; ++b)
                W[n * bpc + b] = make_block(exps[(n + b) % exps.size()], n * 5 + b * 2 + 1);

        auto* d_in  = sycl::malloc_device<ie::block_mxfp4>(W.size(), q);
        auto* d_out = sycl::malloc_device<sycl::half>(size_t(K) * N, q);
        q.memcpy(d_in, W.data(), W.size() * sizeof(ie::block_mxfp4)).wait();
        ie::dequant_mxfp4_to_Bt(q, d_in, d_out, K, N, {}).wait();
        std::vector<sycl::half> got(size_t(K) * N);
        q.memcpy(got.data(), d_out, got.size() * sizeof(sycl::half)).wait();

        size_t bad = 0;
        for (uint32_t n = 0; n < N; ++n)
            for (uint32_t b = 0; b < bpc; ++b) {
                sycl::half ref[32];
                ref_block(W[n * bpc + b], ref);
                for (int k = 0; k < 32; ++k) {
                    const uint32_t row = b * 32 + k;
                    if (float(got[size_t(row) * N + n]) != float(ref[k])) ++bad;
                }
            }
        sycl::free(d_in, q);
        sycl::free(d_out, q);
        if (bad) { std::printf("%sFAIL%s dequant_mxfp4_to_Bt: %zu/%u elems differ\n", R, Z, bad, K * N); ++fails; }
        else     { std::printf("%sOK%s   dequant_mxfp4_to_Bt: %ux%u Bt bit-exact vs ref\n", G, Z, K, N); }
    }

    if (fails) { std::printf("%sMXFP4 dequant: %d test(s) FAILED%s\n", R, fails, Z); return 1; }
    std::printf("%sMXFP4 dequant: all tests passed%s\n", G, Z);
    return 0;
}
