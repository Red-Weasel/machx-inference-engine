// src/ops/gemm_fp16.cpp — FP16 GEMM via SYCL joint_matrix with SLM tiling.
//
// Reference: research/02_programming_stack.md §6.1.
//
// Layout
//   WG tile      BM × BN = 128 × 128
//   SG layout    4 × 4 subgroups (16 SGs/WG)
//   Per-SG tile  SGM × SGN = 32 × 32, decomposed into M_REPEAT=4 × N_REPEAT=2
//                XMX accumulators of shape TM=8 × TN=16
//   K tile       BK = 16 (one fp16 dpas worth)
//
// SLM
//   A_smem[BM × BK]   = 128 × 16 fp16 = 4 KiB
//   B_smem[BK × BN]   = 16 × 128 fp16 = 4 KiB
//   Single-buffered (no double-buffer in v1) — barrier between K iters.

#include "ie/ops.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include "ie/kernel_profiler.hpp"

namespace ie {

namespace mat = sycl::ext::oneapi::experimental::matrix;
using fp16 = sycl::half;

namespace {

constexpr int SG_SIZE  = 16;
constexpr int BM       = 128;
constexpr int BN       = 128;
constexpr int BK       = 16;
constexpr int SGM      = 32;
constexpr int SGN      = 32;
constexpr int TM       = 8;
constexpr int TN       = 16;
constexpr int TK       = 16;
constexpr int M_REPEAT = SGM / TM;        // 4
constexpr int N_REPEAT = SGN / TN;        // 2
constexpr int SGS_PER_WG = (BM / SGM) * (BN / SGN);  // 16
constexpr int WG_ITEMS = SGS_PER_WG * SG_SIZE;       // 256

}  // namespace

sycl::event gemm_fp16(sycl::queue& q,
                      const fp16* A, const fp16* B, float* C,
                      uint32_t M, uint32_t N, uint32_t K,
                      const std::vector<sycl::event>& deps) {
    // Caller's responsibility to ensure M, N are multiples of 128 and K of 16.
    const uint32_t wgs_m = (M + BM - 1) / BM;
    const uint32_t wgs_n = (N + BN - 1) / BN;

    return ie::ps(q, "gemm_fp16", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<fp16, 2> A_smem({BM, BK}, h);
        sycl::local_accessor<fp16, 2> B_smem({BK, BN}, h);

        h.parallel_for(sycl::nd_range<2>({wgs_m * 1, wgs_n * WG_ITEMS},
                                         {1, WG_ITEMS}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG_SIZE)]] {
            const int lid     = int(it.get_local_id(1));
            const int sg_id   = lid / SG_SIZE;            // 0..15
            const int sg_row  = sg_id / 4;                // 0..3
            const int sg_col  = sg_id % 4;                // 0..3
            const int wg_m    = int(it.get_group(0)) * BM;
            const int wg_n    = int(it.get_group(1)) * BN;
            const int sg_m0   = wg_m + sg_row * SGM;
            const int sg_n0   = wg_n + sg_col * SGN;
            auto sg = it.get_sub_group();

            // Accumulators: M_REPEAT × N_REPEAT
            mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator,
                              TM, TN> acc[M_REPEAT][N_REPEAT];
            #pragma unroll
            for (int mr = 0; mr < M_REPEAT; ++mr)
                #pragma unroll
                for (int nr = 0; nr < N_REPEAT; ++nr)
                    mat::joint_matrix_fill(sg, acc[mr][nr], 0.0f);

            const int sg_lane = lid % SG_SIZE;        // 0..15
            for (int k = 0; k < int(K); k += BK) {
                // A load: 256 items × 8 fp16 each → 2048 = BM*BK.
                //   Map (lid) -> (row = lid/2, c_start = (lid%2)*8). Each item
                //   reads 8 contiguous fp16 from one row of A; subsequent lanes
                //   in the subgroup read consecutive 8-element strips.
                {
                    const int r       = lid / 2;
                    const int c_start = (lid & 1) * 8;
                    const uint64_t base = uint64_t(wg_m + r) * K + (k + c_start);
                    if (wg_m + r < int(M)) {
                        #pragma unroll
                        for (int dc = 0; dc < 8; ++dc) A_smem[r][c_start + dc] = A[base + dc];
                    } else {
                        #pragma unroll
                        for (int dc = 0; dc < 8; ++dc) A_smem[r][c_start + dc] = fp16(0);
                    }
                }
                // B load: 16 SGs × 16 lanes × 8 fp16 = 2048 = BK*BN.
                //   Map (sg_id) -> row, (sg_lane*8) -> c_start. Coalesced 16 B/lane.
                {
                    const int r       = sg_id;
                    const int c_start = sg_lane * 8;
                    const uint64_t base = uint64_t(k + r) * N + (wg_n + c_start);
                    if (wg_n + c_start + 7 < int(N)) {
                        #pragma unroll
                        for (int dc = 0; dc < 8; ++dc) B_smem[r][c_start + dc] = B[base + dc];
                    } else {
                        #pragma unroll
                        for (int dc = 0; dc < 8; ++dc) {
                            B_smem[r][c_start + dc] =
                                (wg_n + c_start + dc < int(N)) ? B[base + dc] : fp16(0);
                        }
                    }
                }
                sycl::group_barrier(it.get_group());

                // 4 × 2 = 8 mat_mads per SG.
                #pragma unroll
                for (int mr = 0; mr < M_REPEAT; ++mr) {
                    mat::joint_matrix<sycl::sub_group, fp16, mat::use::a,
                                      TM, TK, mat::layout::row_major> a_tile;
                    mat::joint_matrix_load(sg, a_tile,
                        A_smem.get_multi_ptr<sycl::access::decorated::no>() +
                        (sg_row * SGM + mr * TM) * BK,
                        /*stride=*/BK);
                    #pragma unroll
                    for (int nr = 0; nr < N_REPEAT; ++nr) {
                        mat::joint_matrix<sycl::sub_group, fp16, mat::use::b,
                                          TK, TN, mat::layout::row_major> b_tile;
                        mat::joint_matrix_load(sg, b_tile,
                            B_smem.get_multi_ptr<sycl::access::decorated::no>() +
                            sg_col * SGN + nr * TN,
                            /*stride=*/BN);
                        mat::joint_matrix_mad(sg, acc[mr][nr], a_tile, b_tile, acc[mr][nr]);
                    }
                }
                sycl::group_barrier(it.get_group());
            }

            // Store accumulators to global C.
            #pragma unroll
            for (int mr = 0; mr < M_REPEAT; ++mr) {
                #pragma unroll
                for (int nr = 0; nr < N_REPEAT; ++nr) {
                    const int row = sg_m0 + mr * TM;
                    const int col = sg_n0 + nr * TN;
                    if (row < int(M) && col < int(N)) {
                        mat::joint_matrix_store(sg, acc[mr][nr],
                            sycl::address_space_cast<sycl::access::address_space::global_space,
                                                     sycl::access::decorated::no>(
                                C + uint64_t(row) * N + col),
                            /*stride=*/N, mat::layout::row_major);
                    }
                }
            }
        });
    });
}

}  // namespace ie
