// tests/dense_unpermute_test.cpp — P3a Task 2: the Llama Q/K un-permute.
// Host-only (no GPU). Proves llama_qk_unpermute_rows INVERTS
// convert_hf_to_gguf.py LlamaModel.permute, so our NEOX rope_partial(_ff) sees
// HF rotate_half layout. A formula typo here would blow up layer-0 parity.
#include "ie/dense_transformer.hpp"

#include <cassert>
#include <cstdio>
#include <vector>

int main() {
    // Tiny distinguishable case: 2 heads × head_dim 8 (half = 4).
    const uint32_t n_heads = 2, hd = 8, half = hd / 2u, N = n_heads * hd;
    const auto perm = ie::llama_qk_unpermute_rows(n_heads, hd);
    assert(perm.size() == N);

    // 1. Exact formula: perm[h*hd + b*half + k] == h*hd + 2k + b.
    for (uint32_t h = 0; h < n_heads; ++h)
        for (uint32_t k = 0; k < half; ++k)
            for (uint32_t b = 0; b < 2u; ++b)
                assert(perm[h * hd + b * half + k] == h * hd + 2u * k + b);

    // 2. Round-trip identity: GGUF-permute(HF) then our un-permute == HF.
    // Converter places HF row (h*hd + b*half + k) at GGUF row (h*hd + 2k + b).
    std::vector<uint32_t> hf(N), gguf(N), back(N);
    for (uint32_t i = 0; i < N; ++i) hf[i] = i;     // HF row i carries value i
    for (uint32_t h = 0; h < n_heads; ++h)
        for (uint32_t k = 0; k < half; ++k)
            for (uint32_t b = 0; b < 2u; ++b)
                gguf[h * hd + 2u * k + b] = hf[h * hd + b * half + k];
    for (uint32_t i = 0; i < N; ++i) back[i] = gguf[perm[i]];   // dest[i] = src[perm[i]]
    for (uint32_t i = 0; i < N; ++i) assert(back[i] == hf[i]);

    // 3. Bijection: every source row used exactly once (no drop/dup).
    std::vector<int> seen(N, 0);
    for (uint32_t i = 0; i < N; ++i) { assert(perm[i] < N); ++seen[perm[i]]; }
    for (uint32_t i = 0; i < N; ++i) assert(seen[i] == 1);

    // 4. Identity guard for the non-square GQA head count (kv heads).
    const auto perm_kv = ie::llama_qk_unpermute_rows(8, 128);
    assert(perm_kv.size() == 8u * 128u);

    std::printf("dense_unpermute_test: OK (formula + round-trip + bijection, N=%u)\n", N);
    return 0;
}
