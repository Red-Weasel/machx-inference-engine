// src/model/qwen3moe_pack.cpp — host MoE counting-sort packer.
#include "ie/qwen3moe_pack.hpp"

namespace ie {
void build_moe_packing(
    const std::vector<std::vector<std::pair<uint32_t, float>>>& routes,
    uint32_t E, uint32_t K, MoePacking& out) {
    const uint32_t T = uint32_t(routes.size());
    const uint32_t TK = T * K;
    out.expert_offsets.assign(E + 1, 0);
    out.sorted_idx.assign(TK, 0);
    out.weights_packed.assign(TK, 0.f);
    out.tk_to_packed.assign(TK, 0);

    std::vector<uint32_t> count(E, 0);
    for (uint32_t t = 0; t < T; ++t)
        for (const auto& [e, w] : routes[t]) ++count[e];
    for (uint32_t e = 0; e < E; ++e) out.expert_offsets[e + 1] = out.expert_offsets[e] + count[e];
    std::vector<uint32_t> cursor(out.expert_offsets.begin(), out.expert_offsets.end() - 1);
    for (uint32_t t = 0; t < T; ++t) {
        const auto& r = routes[t];
        for (uint32_t kslot = 0; kslot < uint32_t(r.size()); ++kslot) {
            const uint32_t e   = r[kslot].first;
            const float    w   = r[kslot].second;
            const uint32_t pos = cursor[e]++;
            out.sorted_idx[pos]            = int32_t(t);
            out.weights_packed[pos]        = w;
            out.tk_to_packed[t * K + kslot] = int32_t(pos);
        }
    }
}
}  // namespace ie
