// bench/bench_gemm_fp16.cpp — Phase 0 sanity GEMM via SYCL joint_matrix.
//
// What this is:
//   - Validates the joint_matrix XMX path on the B70.
//   - Measures effective TFLOPS on a 4096^3 FP16 GEMM (C = A @ B), accumulator FP32.
//   - Tile shape per subgroup: M=8, N=16, K=16 (the canonical Xe2 fp16 tile).
//   - One subgroup per work-group, M_REPEAT accumulators stacked along M.
//
// What this is NOT:
//   - SLM-tiled / 2D-block-load / double-buffered. That's Phase 3.
//   - The "real" GEMM. This will land somewhere around 10-40% of peak.
//
// Peak FP16 on B70 (XMX, dense): ~183 TFLOPS (research/01_hardware.md §2).
// Phase 0 gate goal: ≥50% peak; if we miss, document and defer to Phase 3.

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace mat = sycl::ext::oneapi::experimental::matrix;
using fp16 = sycl::half;

namespace {

// ===== Tunables =====
constexpr int TM = 8;            // joint_matrix M
constexpr int TN = 16;           // joint_matrix N (== subgroup size)
constexpr int TK = 16;           // joint_matrix K (fp16)
constexpr int SG = 16;           // subgroup size
constexpr int M_REPEAT = 8;      // accumulators stacked along M per subgroup
constexpr int SG_TILE_M = TM * M_REPEAT;   // 64
constexpr int SG_TILE_N = TN;              // 16

// =====================================================================
//  Naive joint_matrix kernel: 1 subgroup per workgroup.
//   Each subgroup computes a SG_TILE_M (=64) x SG_TILE_N (=16) tile of C.
//   K loop is straight global-memory reads via joint_matrix_load.
// =====================================================================
void gemm_jm_naive(sycl::queue& q,
                   const fp16* A,   // [M, K] row-major
                   const fp16* B,   // [K, N] row-major
                   float*      C,   // [M, N] row-major fp32
                   int M, int N, int K)
{
    sycl::range<2> global((M + SG_TILE_M - 1) / SG_TILE_M,
                          ((N + SG_TILE_N - 1) / SG_TILE_N) * SG);
    sycl::range<2> local(1, SG);

    q.parallel_for(sycl::nd_range<2>(global, local),
                   [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
        const int gm = it.get_group(0);                          // row-tile
        const int gn = it.get_group(1);                          // col-tile (subgroup grid)
        const int row0 = gm * SG_TILE_M;
        const int col0 = gn * SG_TILE_N;
        if (row0 >= M || col0 >= N) return;

        auto sg = it.get_sub_group();

        // M_REPEAT accumulators along M
        mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator,
                          TM, TN> acc[M_REPEAT];
        for (int i = 0; i < M_REPEAT; ++i) {
            mat::joint_matrix_fill(sg, acc[i], 0.0f);
        }

        for (int k = 0; k < K; k += TK) {
            mat::joint_matrix<sycl::sub_group, fp16, mat::use::a,
                              TM, TK, mat::layout::row_major> a_tile;
            mat::joint_matrix<sycl::sub_group, fp16, mat::use::b,
                              TK, TN, mat::layout::row_major> b_tile;

            // Load B tile once, reuse across all M_REPEAT accumulators.
            mat::joint_matrix_load(sg, b_tile,
                sycl::address_space_cast<sycl::access::address_space::global_space,
                                         sycl::access::decorated::no>(B + k * N + col0),
                /*stride=*/N);

            for (int i = 0; i < M_REPEAT; ++i) {
                mat::joint_matrix_load(sg, a_tile,
                    sycl::address_space_cast<sycl::access::address_space::global_space,
                                             sycl::access::decorated::no>(
                        A + (row0 + i * TM) * K + k),
                    /*stride=*/K);
                mat::joint_matrix_mad(sg, acc[i], a_tile, b_tile, acc[i]);
            }
        }

        for (int i = 0; i < M_REPEAT; ++i) {
            mat::joint_matrix_store(sg, acc[i],
                sycl::address_space_cast<sycl::access::address_space::global_space,
                                         sycl::access::decorated::no>(
                    C + (row0 + i * TM) * N + col0),
                /*stride=*/N, mat::layout::row_major);
        }
    });
}

// =====================================================================
//  Reference (slow, on host) for a small block, just to verify correctness.
// =====================================================================
// Computes the top-left rows x cols block of A @ B into C (row-major, leading dim = cols).
void gemm_ref_block(const fp16* A, const fp16* B, float* C,
                    int N, int K, int rows, int cols)
{
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                acc += float(A[i*K + k]) * float(B[k*N + j]);
            }
            C[i*cols + j] = acc;
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096, iters = 10, warmup = 3;
    bool verify = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--m" && i+1 < argc) M = std::atoi(argv[++i]);
        else if (a == "--n" && i+1 < argc) N = std::atoi(argv[++i]);
        else if (a == "--k" && i+1 < argc) K = std::atoi(argv[++i]);
        else if (a == "--iters" && i+1 < argc) iters = std::atoi(argv[++i]);
        else if (a == "--verify") verify = true;
    }
    if (M % SG_TILE_M || N % SG_TILE_N || K % TK) {
        std::printf("M,N must be multiples of %d,%d and K of %d\n", SG_TILE_M, SG_TILE_N, TK);
        return 1;
    }

    // Pick first GPU whose name has "0xe223", else first GPU.
    sycl::device dev;
    bool found = false;
    for (const auto& d : sycl::device::get_devices()) {
        if (d.is_gpu() && d.get_info<sycl::info::device::name>().find("0xe223") != std::string::npos) {
            dev = d; found = true; break;
        }
    }
    if (!found) for (const auto& d : sycl::device::get_devices()) {
        if (d.is_gpu()) { dev = d; found = true; break; }
    }
    if (!found) { std::fputs("no GPU\n", stderr); return 2; }

    sycl::queue q(dev, sycl::property::queue::enable_profiling{});
    std::printf("Device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    std::printf("Shape : M=%d N=%d K=%d  (FP16, FP32 acc)\n", M, N, K);
    std::printf("Tile  : SG=%d  per-SG=%dx%dxK  M_REPEAT=%d  SG_TILE=%dx%d\n",
                SG, TM, TN, M_REPEAT, SG_TILE_M, SG_TILE_N);

    // Init
    std::vector<fp16> hA(size_t(M)*K), hB(size_t(K)*N);
    std::mt19937 rng(0xC0FFEE);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : hA) x = fp16(dist(rng));
    for (auto& x : hB) x = fp16(dist(rng));

    fp16* dA = sycl::malloc_device<fp16>(size_t(M)*K, q);
    fp16* dB = sycl::malloc_device<fp16>(size_t(K)*N, q);
    float* dC = sycl::malloc_device<float>(size_t(M)*N, q);
    if (!dA || !dB || !dC) { std::fputs("device alloc failed\n", stderr); return 3; }

    q.memcpy(dA, hA.data(), hA.size()*sizeof(fp16)).wait();
    q.memcpy(dB, hB.data(), hB.size()*sizeof(fp16)).wait();

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        gemm_jm_naive(q, dA, dB, dC, M, N, K);
    }
    q.wait_and_throw();

    // Time
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    for (int i = 0; i < iters; ++i) gemm_jm_naive(q, dA, dB, dC, M, N, K);
    q.wait_and_throw();
    auto t1 = clock::now();

    const double sec = std::chrono::duration<double>(t1 - t0).count() / iters;
    const double flops = 2.0 * double(M) * double(N) * double(K);
    const double tflops = flops / sec / 1e12;
    const double peak_tflops = 183.0;          // Arc Pro B70 dense FP16 XMX
    const double pct_peak = 100.0 * tflops / peak_tflops;

    std::printf("\nResult\n  iters     : %d\n  time/iter : %.3f ms\n",
                iters, sec * 1e3);
    std::printf("  TFLOPS    : %.2f\n", tflops);
    std::printf("  %% peak    : %.1f%% of %.0f TFLOPS\n", pct_peak, peak_tflops);
    std::printf("  gate (50%% peak): %s\n", pct_peak >= 50.0 ? "PASS" : "MISS (Phase 3 will fix)");

    int rc = 0;
    if (verify) {
        std::vector<float> hC(size_t(M)*N);
        q.memcpy(hC.data(), dC, hC.size()*sizeof(float)).wait();
        std::vector<float> hRef(size_t(SG_TILE_M)*SG_TILE_N);
        gemm_ref_block(hA.data(), hB.data(), hRef.data(), N, K, SG_TILE_M, SG_TILE_N);
        float max_abs = 0.f;
        for (int i = 0; i < SG_TILE_M; ++i)
            for (int j = 0; j < SG_TILE_N; ++j)
                max_abs = std::max(max_abs, std::abs(hC[i*N + j] - hRef[i*SG_TILE_N + j]));
        std::printf("  verify    : max-abs (top-left %dx%d block) = %.4f  ", SG_TILE_M, SG_TILE_N, max_abs);
        // FP16 mul + FP32 acc, K=4096 -> rough error budget ~K*eps_fp16 ≈ 1-4
        if (max_abs > 8.0f) { std::puts("FAIL"); rc = 4; } else std::puts("OK");
    }

    sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);
    return rc;
}
