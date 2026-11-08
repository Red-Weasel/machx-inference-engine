// include/ie/qwen4_route.hpp — Flash-Next (qwen4exp) MoE routing that the host and the device compute bit for bit alike
// (2026-11-08; option 2 phase 1, docs/research/2026-11-08-flashnext-device-routing-design.md).
//
// From one row of E F16 router logits:
//   picks   = the k largest logits; equal logits -> the lower expert id first. This is the softmax's top-k (the softmax is
//             monotonic), without the probabilities' rounding deciding the order.
//   weights = w_j = exp(l_j - l_max) / sum_i exp(l_i - l_max) over the k picks -- the softmax's full denominator cancels in
//             the renormalisation, so it is never formed.
//   order   = the picks in ascending expert id; the sum runs in that order.
// The exp is a fixed Cody-Waite + degree-6 fma polynomial (about 1 ulp), and every step is an IEEE float operation with no
// contraction left to the compiler (the .cpp is built -fp-model=precise with contraction off), so qwen4_route_host and
// qwen4_route_dev return the same bits. route_from_logits (qwen4exp.cpp today) differs in the last bits of the weights
// (libm expf, a double sum, a float sum in partial_sort order) and in how exactly-equal probabilities are ordered;
// ie-qwen4-route-test measures both against it.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <vector>

namespace ie {

constexpr uint32_t kQwen4RouteMaxK = 16;   // k <= 16
constexpr uint32_t kQwen4RouteMaxE = 1024; // E <= 1024

// One row: ids[k] ascending, w[k].
void qwen4_route_host(const sycl::half* logits, uint32_t E, uint32_t k, int32_t* ids, float* w);

// rows rows of E logits (row-major) -> ids [rows][k] int32, w [rows][k] float, on the device. One work-group a row.
sycl::event qwen4_route_dev(sycl::queue& q, const sycl::half* logits, uint32_t rows, uint32_t E, uint32_t k,
                            int32_t* ids, float* w, const std::vector<sycl::event>& deps = {});

// The decode / verify MoE job table of one layer, routed on the device (option 2 phase 3): T rows of logits -> the table the
// grouped pick kernels read, in qwen4exp's host layout (every pick a job; the expert-parallel masking is not done here):
//   pick p = vt*k + j (j-th pick of row vt, ascending id):  jobs[3p] = vt, jobs[3p+1] = row, jobs[3p+2] = p
//   row     = slot_of[e] >= 0 ? slot_of[e] : S + e  -- the slot address table's VRAM row, or the expert's pinned host row
//             (S = the layer's cache slots; include/ie/qwen4_bank.hpp's slot_tab)
//   jobs[cnt_off + vt] = k;  float weights at reinterpret_cast<float*>(jobs + w_off)[p];  ids_out[p] = e (for the host's
//   lagging cache bookkeeping). Picks and weights are qwen4_route_dev's, bit for bit.
sycl::event qwen4_route_jobs_dev(sycl::queue& q, const sycl::half* logits, uint32_t T, uint32_t E, uint32_t k,
                                 const int16_t* slot_of, uint32_t S, int32_t* jobs, uint32_t cnt_off, uint32_t w_off,
                                 int32_t* ids_out, const std::vector<sycl::event>& deps = {});

}  // namespace ie
