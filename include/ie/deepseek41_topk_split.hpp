// include/ie/deepseek41_topk_split.hpp -- Phase 39 (docs/deepseek41/79): the indexer top-k split across work-groups.
// Two-level EXACT top-k: chunks of kChunk keys sorted side by side, then one sort over the chunk winners. Emits the
// row exactly as ds4_indexer_topk does (descending score, pitch top_k, the causal sentinel -1), bit for bit.
#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
#include <vector>

namespace ie {

// True where the split is worth it and applicable: n_keys > 8,192 and T <= 8 (decode rows), unless IE_DS41_TOPK_SPLIT=0.
bool ds41_topk_split_wanted(uint32_t T, uint32_t n_keys, uint32_t top_k);
// scores [T, n_keys], positions [T] (thr = (pos + 1) / compress_rate), out [T, top_k]; top_k <= 512. Scratch is per queue.
sycl::event ds41_indexer_topk_split(sycl::queue& q, const float* scores, const int32_t* positions, int32_t* out,
                                    uint32_t T, uint32_t n_keys, uint32_t top_k, uint32_t compress_rate,
                                    const std::vector<sycl::event>& deps = {});

}  // namespace ie
