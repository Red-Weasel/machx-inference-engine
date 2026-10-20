#pragma once
#include "ie/model_config.hpp"
#include "ie/tokenizer.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
namespace ie {
// `ie serve`'s GLM prefill chunk when --prefill-chunk is not given (Engine load; IE_G5_PREFILL_CHUNK overrides).
// Prefill streams the host expert bank once per chunk: 256 -> 1024 doubled it (74 -> 148 tok/s on a 9.6K-token
// prompt). With the GEMV kernels it was then per-token compute bound and 2048 gained nothing (147, while each chunk
// row reserves ~1.4 MB per card from the expert cache: decode 13.0-13.9 vs 13.9-14.2); with the prefill MoE and Q8
// dense projections on XMX (Glm5NextModel xmx_rows_min_ / q8_gemm_rows_) 2048 runs 278 tok/s vs 214 at 1024
// (pinned runner, 10K tokens). results/glm53-2026-10-19 (chunk_sweep.log, pp_pin3.log).
inline constexpr uint32_t kGlm5PrefillChunk = 2048;
// The largest GLM prefill chunk the load accepts: at 4096 a forward overflows a kernel's 32-bit SYCL range
// ("Provided range and/or offset does not fit in int"; the sparse attention's rows x 64 heads x 2051 selected x 4
// passes 2^31) and every request fails; 2048 is the largest measured.
inline constexpr uint32_t kGlm5MaxPrefillChunk = 2048;
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
    // The prefill Q8 dense XMX route's fp16 weight scratch (Glm5NextModel::q8_w16_): the largest dense projection,
    // the MLA output [heads x value_len_mla, hidden] (134 MB on GLM-5.3), the dense FFN or the MTP eh_proj.
    const uint64_t q8_w16=2ull*c.hidden*std::max({uint64_t(c.n_q_heads)*c.value_len_mla,
                                                   uint64_t(c.n_q_heads)*c.kda_head_dim,
                                                   uint64_t(c.ffn),uint64_t(2)*c.hidden});
    return (1ull<<30)+state+latent+index+mt*per_row+q8_w16;
}
// The MTP draft kit's runtime buffers on the stage that hosts it (Glm5NextModel::init_runtime with the kit loaded): its
// own latent cache [ctx, kv_lora] f16 and the few [H] rows, with 64 MiB of slack; its weights come through
// plan_device_weights and its expert slots out of that stage's cache budget.
inline uint64_t glm5_mtp_reserve(const Glm5NextConfig& c, uint32_t ctx) {
    return uint64_t(ctx)*c.kv_lora_rank*2+uint64_t(c.hidden)*16+(64ull<<20);
}
// P4 B7: the device bytes ONE extra request lane takes on stage [lo, hi) at `ctx` positions -- what
// Glm5NextModel::init_runtime allocates per lane: the KDA scan (fp32) + conv (f16) state, the MLA latents,
// and when the stage runs the DSA indexer (`sparse`: its context is above the selection width) the pooled
// keys and the open pool's key/gate rolls.
inline uint64_t glm5_lane_bytes(const Glm5NextConfig& c, uint32_t ctx, uint32_t lo, uint32_t hi, bool sparse) {
    uint64_t full=0, linear=0;
    for (uint32_t l=lo;l<hi;++l) (c.is_full_attn(l)?full:linear)++;
    const uint64_t di=uint64_t(c.n_q_heads)*c.kda_head_dim;
    const uint64_t kda=std::max<uint64_t>(1,linear)*(di*c.kda_head_dim*4+3*di*(c.conv_kernel-1)*2);
    const uint64_t kp=std::max(1u,c.indexer_kpool);
    const uint64_t idx=sparse?full*((uint64_t(ctx)+kp-1)/kp)*c.indexer_head_dim*2+2*full*kp*c.indexer_head_dim*2:0;
    return kda+full*uint64_t(ctx)*c.kv_lora_rank*2+idx;
}
inline bool glm5_residency_fits(uint64_t weights,uint64_t runtime,uint64_t cache,uint64_t available) {
    return weights<=available && runtime<=available-weights && cache<=available-weights-runtime;
}
std::string build_glm5_prompt(std::span<const ChatTurn> turns, bool thinking,
                             std::string_view tools_json, std::string& error,
                             std::string_view reasoning_effort = {});
// The stable conversation boundary of a build_glm5_prompt render: the prompt without its generation suffix
// (<|assistant|><think>, and </think> after it when thinking is off) -- the text the next turn's prompt starts
// with, where the prompt cache takes its copy. Empty when the prompt does not end with that suffix.
std::string_view glm5_stable_prefix(std::string_view prompt, bool thinking);
struct Glm5Completion { std::string content, reasoning, tool_calls; };
Glm5Completion parse_glm5_completion(std::string_view text, bool thinking, std::string_view tools_json = {});
}
