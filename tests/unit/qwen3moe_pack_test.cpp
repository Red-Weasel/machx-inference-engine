// tests/unit/qwen3moe_pack_test.cpp — host test for the MoE counting-sort packer.
#include "ie/qwen3moe_pack.hpp"
#include <cassert>
#include <cstdio>
#include <vector>

int main() {
    using ie::MoePacking;
    const uint32_t T = 2, E = 4, K = 2;
    std::vector<std::vector<std::pair<uint32_t,float>>> routes = {
        {{0, 0.6f}, {2, 0.4f}},
        {{2, 0.7f}, {3, 0.3f}},
    };
    MoePacking p;
    ie::build_moe_packing(routes, E, K, p);

    assert(p.sorted_idx.size() == T * K);
    assert(p.tk_to_packed.size() == T * K);
    assert(p.weights_packed.size() == T * K);
    assert(p.expert_offsets.size() == E + 1);

    const uint32_t exp_off[5] = {0, 1, 1, 3, 4};
    for (uint32_t e = 0; e <= E; ++e) assert(p.expert_offsets[e] == exp_off[e]);
    assert(p.sorted_idx[0] == 0 && p.sorted_idx[1] == 0 &&
           p.sorted_idx[2] == 1 && p.sorted_idx[3] == 1);
    assert(p.weights_packed[0] == 0.6f && p.weights_packed[1] == 0.4f &&
           p.weights_packed[2] == 0.7f && p.weights_packed[3] == 0.3f);
    assert(p.tk_to_packed[0] == 0 && p.tk_to_packed[1] == 1 &&
           p.tk_to_packed[2] == 2 && p.tk_to_packed[3] == 3);
    std::printf("qwen3moe_pack_test: OK\n");
    return 0;
}
