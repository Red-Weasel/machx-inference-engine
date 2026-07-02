// tests/unit/gemv_q4k_test.cpp — correctness gate for the W4A16 GEMV.
//
// Uses real Q4_K weight tensors from the Qwen3.6-35B-A3B GGUF, dequants on the
// host (proven golden in Phase 2), then computes y_ref = A @ W_dequanted in
// fp32. Compares to the GPU GEMV's fp16 output.

#include "ie/dequant_ref.hpp"
#include "ie/gguf.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"
#include "ie/tensor.hpp"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr const char* G = "\033[32m";
constexpr const char* R = "\033[31m";
constexpr const char* Z = "\033[0m";

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

bool test_one(sycl::queue& q, ie::GgufReader& g,
              const char* tensor_name, ie::DType expected_dtype, float tol_rel) {
    auto* t = g.find_tensor(tensor_name);
    if (!t) {
        std::printf("  %sSKIP%s   %s — tensor not found\n", "\033[33m", Z, tensor_name);
        return true;
    }
    if (t->dtype != expected_dtype) {
        std::printf("  %sSKIP%s   %s — dtype %.*s, expected %.*s\n", "\033[33m", Z, tensor_name,
                    int(ie::type_name(t->dtype).size()), ie::type_name(t->dtype).data(),
                    int(ie::type_name(expected_dtype).size()), ie::type_name(expected_dtype).data());
        return true;
    }
    if (t->n_dims < 2) {
        std::printf("  %sSKIP%s   %s — needs >=2 dims\n", "\033[33m", Z, tensor_name);
        return true;
    }

    // Use [K, N] = [shape[0], shape[1]] as the 2D matmul. Higher dims (e.g. for
    // expert tensors with [in, out, n_experts]) get a dedicated test below.
    if (t->n_dims != 2) {
        std::printf("  %sSKIP%s   %s — multi-dim (use single 2D test)\n", "\033[33m", Z, tensor_name);
        return true;
    }
    const uint32_t K = uint32_t(t->shape[0]);
    const uint32_t N = uint32_t(t->shape[1]);

    // Random A, fixed seed
    std::vector<sycl::half> hA(K);
    std::vector<sycl::half> hY(N);
    {
        std::mt19937 rng(0xA53);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        for (auto& v : hA) v = sycl::half(dist(rng));
    }

    // Host reference: dequant W -> [K, N] fp32, then y = A @ W
    std::vector<float> W_full(size_t(K) * N);
    if (expected_dtype == ie::DType::kQ4_K) {
        ie::ref::dequant_q4_K_buffer(t->data, size_t(K) * N, W_full.data());
    } else if (expected_dtype == ie::DType::kQ6_K) {
        ie::ref::dequant_q6_K_buffer(t->data, size_t(K) * N, W_full.data());
    } else {
        std::fprintf(stderr, "unsupported dtype in test_one\n");
        return false;
    }
    std::vector<float> y_ref(N, 0.f);
    for (uint32_t k = 0; k < K; ++k) {
        const float a = float(hA[k]);
        for (uint32_t n = 0; n < N; ++n) y_ref[n] += a * W_full[size_t(k) * N + n];
    }

    // Wait — the GGUF layout we assume in gemv_q4_K is "column-major super-blocks":
    // each output column n's K elements are 8 consecutive super-blocks, and column
    // n+1's super-blocks immediately follow. That corresponds to W_full being
    // shaped [N, K] in row-major (i.e. W[n][k] = element n,k along the contiguous
    // axis-0). So if shape[0] is K and shape[1] is N, the stored layout is row-major
    // [K, N] in matmul terms? Let's just match the actual byte layout:
    //
    // The dequant_q4_K_buffer call decodes the buffer in linear order, i.e. the
    // 256 elements of super-block 0 come first, then super-block 1, etc. That
    // means W_full[i] for i in [0, 256) is super-block 0 of the FIRST column
    // (whichever interpretation). The kernel iterates `col_blocks = &W[n * blocks_per_col]`
    // — so super-block 0 of column n=0 is at file offset 0. Therefore col 0's
    // K=shape[0] elements are linear offsets [0, K), col 1's are [K, 2K), etc.
    //
    // So in matmul terms: shape is [N, K] not [K, N]. Let's reinterpret:
    //   W_logical[n, k] = W_full[n * K + k]
    //   y[n] = sum_k A[k] * W_logical[n, k]
    //
    // That means N = shape[0] in my call, K = shape[1].
    // Let's redo:
    const uint32_t real_N = uint32_t(t->shape[0]);
    const uint32_t real_K = uint32_t(t->shape[1]);
    if (real_K % 256 != 0) {
        std::printf("  %sSKIP%s   %s — real_K=%u not multiple of 256\n", "\033[33m", Z, tensor_name, real_K);
        return true;
    }
    if (real_K != K || real_N != N) {
        // Recompute with the correct interpretation
        // Actually the correct math: shape[0]=K, shape[1]=N, contiguous axis is shape[0].
        // For each "N column" in matmul, the K=shape[0] elements are stored linearly?
        // Let's just trust that shape[0] is the inner dim in GGUF (the block dim),
        // and shape[1] is the outer.
    }

    // Use the original interpretation: shape[0]=K (block dim), shape[1]=N.
    // Then `W[k, n]` element is at byte offset = (n * blocks_per_col + k/256) * 144 + ...
    // i.e. all of column 0's K elements come first, then column 1's K elements, etc.
    // That's the layout `gemv_q4_K` assumes. So:
    //   W_full[i] is element (k = i % K, n = i / K)? -> for k_blk = 0, i in [0, K),
    //   that's all K elements of column 0. So W_full[i] for i in [0, K) is column 0.
    //   In W_logical[k, n] terms: W_logical[k, 0] = W_full[k]. ✓
    // So y[n] = sum_k A[k] * W_logical[k, n] = sum_k A[k] * W_full[n*K + k]. Reset:
    {
        std::fill(y_ref.begin(), y_ref.end(), 0.f);
        for (uint32_t n = 0; n < N; ++n) {
            float acc = 0.f;
            const float* wcol = W_full.data() + size_t(n) * K;
            for (uint32_t k = 0; k < K; ++k) acc += float(hA[k]) * wcol[k];
            y_ref[n] = acc;
        }
    }

    // GPU
    auto* dA = sycl::malloc_device<sycl::half>(K, q);
    auto* dW = sycl::malloc_device(t->nbytes, q);
    auto* dY = sycl::malloc_device<sycl::half>(N, q);
    q.memcpy(dA, hA.data(), K * sizeof(sycl::half)).wait();
    q.memcpy(dW, t->data, t->nbytes).wait();
    if (expected_dtype == ie::DType::kQ4_K) {
        ie::gemv_q4_K(q, dA, dW, dY, K, N).wait_and_throw();
    } else {
        ie::gemv_q6_K(q, dA, dW, dY, K, N).wait_and_throw();
    }
    q.memcpy(hY.data(), dY, N * sizeof(sycl::half)).wait();
    sycl::free(dA, q); sycl::free(dW, q); sycl::free(dY, q);

    // Diff
    float max_abs = 0.f;
    float max_rel = 0.f;
    uint32_t worst = 0;
    for (uint32_t n = 0; n < N; ++n) {
        const float a = y_ref[n];
        const float b = float(hY[n]);
        const float d = std::fabs(a - b);
        const float scale = std::fmax(std::fabs(a), 1e-3f);
        const float r = d / scale;
        if (r > max_rel) { max_rel = r; worst = n; }
        if (d > max_abs) max_abs = d;
    }
    const bool ok = max_rel <= tol_rel;
    std::printf("  %s%-46s%s K=%4u N=%4u  max_abs=%-7.4f max_rel=%-8.5g worst@%u (exp=%g got=%g)  %s\n",
                ok ? G : R, tensor_name, Z, K, N,
                max_abs, max_rel, worst, y_ref[worst], float(hY[worst]),
                ok ? "OK" : "FAIL");
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf_path = "/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) gguf_path = argv[++i];
    }
    auto q = make_queue();
    std::printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    ie::GgufReader g;
    if (auto err = g.open(gguf_path); !err.empty()) {
        std::fprintf(stderr, "gguf open: %s\n", err.c_str());
        return 3;
    }

    int fails = 0;
    // Q4_K shapes
    if (!test_one(q, g, "blk.0.attn_gate.weight",   ie::DType::kQ4_K, 0.02f)) ++fails;
    if (!test_one(q, g, "blk.3.attn_q.weight",      ie::DType::kQ4_K, 0.02f)) ++fails;
    if (!test_one(q, g, "blk.3.attn_output.weight", ie::DType::kQ4_K, 0.02f)) ++fails;
    if (!test_one(q, g, "blk.0.ssm_out.weight",     ie::DType::kQ4_K, 0.02f)) ++fails;
    // Q6_K shapes
    if (!test_one(q, g, "blk.3.attn_v.weight",      ie::DType::kQ6_K, 0.02f)) ++fails;  // [2048, 512]
    if (!test_one(q, g, "blk.0.attn_qkv.weight",    ie::DType::kQ6_K, 0.02f)) ++fails;  // [2048, 8192]

    std::printf("\n%s\n", fails == 0 ? "ALL PASS" : "FAILURES");
    return fails;
}
