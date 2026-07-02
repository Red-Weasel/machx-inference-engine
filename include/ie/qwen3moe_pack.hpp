// include/ie/qwen3moe_pack.hpp — host MoE counting-sort packer (no SYCL).
#pragma once
#include <cstdint>
#include <utility>
#include <vector>

namespace ie {
struct MoePacking {
    std::vector<uint32_t> expert_offsets;   // [E+1] prefix sums of per-expert row counts
    std::vector<int32_t>  sorted_idx;       // [T*K] token id per expert-sorted packed row
    std::vector<float>    weights_packed;   // [T*K] router weight per packed row
    std::vector<int32_t>  tk_to_packed;     // [T*K] (t*K+kslot) -> packed row index
};

// routes[t] = ascending-by-expert list of (expert_id, weight) for token t.
void build_moe_packing(
    const std::vector<std::vector<std::pair<uint32_t, float>>>& routes,
    uint32_t E, uint32_t K, MoePacking& out);
}  // namespace ie
