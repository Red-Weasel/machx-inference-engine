// include/ie/deepseek41_candidate.hpp — V4.1's two-level candidate top-k, level one.
// Semantics, the tie-break and the no-SLM-atomics constraint are documented at the top of
// src/ops/deepseek41_candidate.cpp; read that before changing anything here.
#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <vector>

namespace ie {

// A block's score: the MAX over its `block_size` latents, latents at or past the query's causal
// threshold ((positions[t] + 1) / compress_rate) counted as -inf, and the block holding the query's
// newest reachable latent pinned to +inf because it is only partly filled.
//   scores    [T, n_keys]   the indexer's raw scores
//   positions [T] int32
//   bscore    [T, ceil(n_keys / block_size)]
sycl::event ds41_candidate_block_scores(sycl::queue& q,
                                        const float* scores, const int32_t* positions,
                                        float* bscore,
                                        uint32_t T, uint32_t n_keys, uint32_t block_size,
                                        uint32_t compress_rate,
                                        const std::vector<sycl::event>& deps = {});

// Keep the `min(topk_blocks, n_blocks)` highest-scoring blocks per query, dropping any pick whose
// score is -inf (unreachable).  `keep` is [T, n_blocks] with 1 = this block's latents stay candidates.
sycl::event ds41_candidate_select(sycl::queue& q,
                                  const float* bscore, uint8_t* keep,
                                  uint32_t T, uint32_t n_keys, uint32_t block_size,
                                  uint32_t topk_blocks,
                                  const std::vector<sycl::event>& deps = {});

// Level two: -inf every score whose block was not kept, in place, before the fine top-k.
sycl::event ds41_candidate_apply(sycl::queue& q,
                                 const uint8_t* keep, float* scores,
                                 uint32_t T, uint32_t n_keys, uint32_t block_size,
                                 const std::vector<sycl::event>& deps = {});

// Sort each query row's top-k picks into ASCENDING index order, -1 sentinels first.
//
// WHY IT EXISTS. `ds4_indexer_topk` emits its picks in DESCENDING SCORE order. The dense attention path never
// cared, because it walked the KV axis in ascending column order and consulted a mask. An attention that visits
// the picks DIRECTLY must visit them in that same ascending order, or the online softmax accumulates the same
// columns in a different sequence and the result moves in the last bits. Sorting here is what lets the gathered
// path be bit-identical to the dense one rather than merely close.
// `k` is the width ds4_indexer_topk EMITTED -- min(index_topk, n_keys), not index_topk -- and it is the
// INPUT row pitch too: the topk packs its rows at that width. `stride` is the OUTPUT pitch (index_topk),
// and the whole of it is written: the valid picks ascending in [0, n_valid), then -1 to `stride`, so the
// attention kernel is handed `stride` as n_picks and a row's layout does not depend on k (which differs
// between a T-row step and the one-row steps it must equal -- P2 row identity).
//   picks [T, k] int32 in, out [T, stride] int32: ascending valid picks, then -1 (sentinels at the END);
//   n_valid [T] int32 out (nullable): each row's count of valid picks, for the attention kernel's partition
sycl::event ds41_sort_picks_asc(sycl::queue& q,
                                const int32_t* picks, int32_t* out, int32_t* n_valid,
                                uint32_t T, uint32_t k, uint32_t stride,
                                const std::vector<sycl::event>& deps = {});

}  // namespace ie
