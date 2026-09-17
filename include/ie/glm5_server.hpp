#pragma once
#include "ie/model_config.hpp"
#include "ie/tokenizer.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
namespace ie {
// Conservative runtime bound: latent KV/pool keys and two copies of KDA
// state are exact. Per-token scratch includes all prefill projections and
// routing buffers; the 1 GiB margin covers fixed scratch and the sampler.
inline uint64_t glm5_runtime_reserve(const Glm5NextConfig& c, uint32_t ctx,
                                    uint32_t chunk, uint32_t lo, uint32_t hi) {
    uint64_t full=0, linear=0;
    for (uint32_t l=lo;l<hi;++l) (c.is_full_attn(l)?full:linear)++;
    uint64_t pools=(uint64_t(ctx)+c.indexer_kpool-1)/std::max(1u,c.indexer_kpool);
    uint64_t state=std::max(uint64_t(1),linear)*c.n_q_heads*c.kda_head_dim*c.kda_head_dim*8;
    uint64_t latent=full*ctx*c.kv_lora_rank*2;
    uint64_t index=full*pools*c.indexer_head_dim*2+std::min(64u,chunk)*pools*4;
    uint64_t mt=(uint64_t(chunk)+7)&~uint64_t(7);
    uint64_t per_row=uint64_t(c.hidden)*(8*c.hc_count+64)+
        uint64_t(c.n_q_heads)*(48*c.kda_head_dim+8*c.kv_lora_rank+
                                  4*c.key_len_mla+2*c.value_len_mla)+
        16ull*std::max(c.ffn,c.expert_ffn)+16ull*c.indexer_top_k+
        8ull*c.indexer_n_heads*c.indexer_head_dim+32ull*c.n_experts_used;
    return (1ull<<30)+state+latent+index+mt*per_row;
}
inline bool glm5_residency_fits(uint64_t weights,uint64_t runtime,uint64_t cache,uint64_t available) {
    return weights<=available && runtime<=available-weights && cache<=available-weights-runtime;
}
std::string build_glm5_prompt(std::span<const ChatTurn> turns, bool thinking,
                             std::string_view tools_json, std::string& error,
                             std::string_view reasoning_effort = {});
struct Glm5Completion { std::string content, reasoning, tool_calls; };
Glm5Completion parse_glm5_completion(std::string_view text, bool thinking, std::string_view tools_json = {});
}
