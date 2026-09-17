// include/ie/deepseek41.hpp — DeepSeek-V4.1-Flash (`deepseek_v41`) config + weight binding.
//
// Phase 3 territory only: host-side parse of `config.json`, host-side binding and shape
// validation over the safetensors mmap. No device memory, no compute.
//
// Every pointer is a non-owning view into SafetensorsModel's mmaps. The model owns nothing;
// it is invalidated when the reader closes.
//
// Shapes below are safetensors order, [out, in], verified against the shipped checkpoint on
// 2026-09-12. Structure read from the reference `inference/model.py` that ships with the
// weights — see docs/deepseek41/03_ARCH_FROM_THE_CODE_2026-09-12.md.
#pragma once

#include "ie/safetensors.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace ie {

// Fields are named for the reference's ModelArgs, whose names are the config JSON keys.
// Nothing here has a default that could silently stand in for a missing key: `load` errors.
struct Ds41Config {
    uint32_t vocab_size = 0, dim = 0, moe_inter_dim = 0;
    uint32_t n_layers = 0, n_mtp_layers = 0, n_heads = 0;
    uint32_t n_routed_experts = 0, n_shared_experts = 0, n_activated_experts = 0;
    std::string score_func;            // "sqrtsoftplus"
    bool     norm_topk_prob = false;
    float    route_scale = 0.f, swiglu_limit = 0.f;
    uint32_t q_lora_rank = 0, head_dim = 0, rope_head_dim = 0;
    uint32_t o_groups = 0, o_lora_rank = 0, window_size = 0;
    float    norm_eps = 0.f;
    // One entry per layer, MTP included: 0 = sliding window only, r = KV compressed r-to-1.
    std::vector<uint32_t> compress_ratios, kv_source_layers, index_source_layers;
    float    rope_theta = 0.f, compress_rope_theta = 0.f, rope_factor = 0.f;
    uint32_t original_seq_len = 0, beta_fast = 0, beta_slow = 0;
    uint32_t index_n_heads = 0, index_head_dim = 0, index_topk = 0;
    int32_t  candidate_source_layer = -1;
    uint32_t candidate_topk_blocks = 0, candidate_block_size = 0;
    uint32_t hc_mult = 0, hc_sinkhorn_iters = 0;
    float    hc_eps = 0.f;
    std::vector<uint32_t> engram_layer_ids;
    std::vector<uint64_t> engram_num_embeddings;
    uint32_t engram_max_ngram_size = 0, engram_n_heads = 0, engram_head_dim = 0;
    uint64_t engram_vocab_size = 0;
    uint32_t engram_compressed_vocab_size = 0, engram_pad_id = 0;
    uint32_t vision_n_layers = 0, vision_dim = 0, image_token_id = 0;
    uint32_t dspark_block_size = 0, dspark_n_routed_experts = 0, dspark_n_activated_experts = 0;
    uint32_t dspark_markov_rank = 0, dspark_noise_token_id = 0;   // the Markov head's rank, the draft placeholder token (DSpark P1)
    std::vector<uint32_t> dspark_target_layer_ids;

    uint32_t nope_head_dim() const { return head_dim - rope_head_dim; }
    // Routed/activated expert counts for a layer: MTP layers have their own, smaller pool.
    uint32_t n_routed(uint32_t L) const {
        return L < n_layers ? n_routed_experts
                            : (dspark_n_routed_experts ? dspark_n_routed_experts : n_routed_experts);
    }
    uint32_t n_activated(uint32_t L) const {
        return L < n_layers ? n_activated_experts
                            : (dspark_n_activated_experts ? dspark_n_activated_experts
                                                          : n_activated_experts);
    }
};

// Per-layer variation, computed ONCE from the config so the binder never branches on a bare
// layer index. Mirrors `Attention.__init__` and `Compressor.__init__` in the reference.
struct Ds41LayerKind {
    uint32_t compress_ratio = 0;
    bool     is_kv_source    = false;  // owns a Compressor + the compressed KV cache
    bool     is_index_source = false;  // owns an Indexer
    // Compressor.__init__ builds a gate only when it actually pools, i.e. ratio > 1. Layer 20
    // is a kv_source at ratio 1, so it has wkv and norm but NO wgate.
    bool     has_compressor_gate = false;
    bool     has_engram      = false;
    uint32_t n_routed = 0, n_activated = 0;
};

// A quantised weight and its scale plane. Every FP8/FP4 tensor in this checkpoint stores its
// scales as a sibling tensor `<name>.scale` rather than interleaved in blocks, so the pair is
// the unit the binder deals in. `scale` is null for BF16/F32 weights, which carry none.
struct Ds41Tensor {
    const SafeTensorInfo* w = nullptr;
    const SafeTensorInfo* s = nullptr;
    explicit operator bool() const { return w != nullptr; }
};

struct Ds41Layer {
    Ds41LayerKind kind;

    // -- attention (every layer) ------------------------------------------
    Ds41Tensor attn_norm;      // BF16 [dim]
    Ds41Tensor wq_a;           // FP8  [q_lora_rank, dim]
    Ds41Tensor q_norm;         // BF16 [q_lora_rank]
    Ds41Tensor wq_b;           // FP8  [n_heads*head_dim, q_lora_rank]
    Ds41Tensor wkv;            // FP8  [head_dim, dim]          (ONE latent KV head)
    Ds41Tensor kv_norm;        // BF16 [head_dim]
    Ds41Tensor attn_sink;      // F32  [n_heads]
    Ds41Tensor wo_a;           // FP8  [o_groups*o_lora_rank, n_heads*head_dim/o_groups]
    Ds41Tensor wo_b;           // FP8  [dim, o_groups*o_lora_rank]

    // -- compressor (kind.is_kv_source) -----------------------------------
    Ds41Tensor comp_wkv;       // BF16 [head_dim, dim]
    Ds41Tensor comp_wgate;     // BF16 [head_dim, dim]   (kind.has_compressor_gate)
    Ds41Tensor comp_norm;      // BF16 [head_dim]

    // -- indexer (kind.is_index_source) -----------------------------------
    Ds41Tensor idx_wq_b;       // FP8  [index_n_heads*index_head_dim, q_lora_rank]
    Ds41Tensor idx_weights;    // BF16 [index_n_heads, dim]
    // wk/k_norm live only on the KV-source layers: the indexer K is derived from the shared
    // compressed latent, so only a layer that produces one also produces its key.
    Ds41Tensor idx_wk;         // BF16 [index_head_dim, head_dim]
    Ds41Tensor idx_k_norm;     // BF16 [index_head_dim]

    // -- MoE (every layer) -------------------------------------------------
    Ds41Tensor ffn_norm;       // BF16 [dim]
    Ds41Tensor gate_w;         // BF16 [n_routed, dim]
    Ds41Tensor gate_bias;      // F32  [n_routed]
    Ds41Tensor gate_bias_vl;   // F32  [n_routed]        (routing bias inside image spans)
    // Routed experts, one tensor each rather than V4's stacked [dim, ffn, n_experts]: 384 per
    // layer x 3 matrices. FP4 planes are [N, K/2] I8 with a [N, K/32] E8M0 scale.
    std::vector<Ds41Tensor> exp_w1, exp_w3, exp_w2;
    Ds41Tensor sh_w1, sh_w3, sh_w2;   // FP8 shared expert

    // -- hyper-connections (every layer) -----------------------------------
    Ds41Tensor hc_attn_fn, hc_attn_base, hc_attn_scale;   // F32 [(2+hc)*hc, hc*dim] / [..] / [3]
    Ds41Tensor hc_ffn_fn,  hc_ffn_base,  hc_ffn_scale;

    // -- engram (kind.has_engram) ------------------------------------------
    Ds41Tensor engram_embed;   // FP8 [num_embeddings, engram_head_dim]  -- 91.5 GiB each
    Ds41Tensor engram_wkv;     // FP8 [engram_n_heads*..., dim*...]
    Ds41Tensor engram_q, engram_k;  // BF16 [max_ngram_size, dim]
};

// Byte accounting by group, so a residency plan can be costed before anything is uploaded.
struct Ds41Budget {
    uint64_t routed_experts = 0, engram = 0, attention = 0, shared_experts = 0;
    uint64_t embed = 0, lm_head = 0, norms_gates_hc = 0, indexer = 0, compressor = 0;
    uint64_t vision = 0, mtp = 0;
    uint64_t total() const {
        return routed_experts + engram + attention + shared_experts + embed + lm_head +
               norms_gates_hc + indexer + compressor + vision + mtp;
    }
};

// DSpark (docs/deepseek41/48, 57): the drafter's tensors outside the three MTP layers' own dense sets --
// bound like everything else (non-owning views); null when the checkpoint has no drafter.
struct Ds41DSpark {
    Ds41Tensor main_proj;        // FP8 [dim, dim * n_targets] + 32x32 scale
    Ds41Tensor main_norm;        // BF16 [dim]
    Ds41Tensor norm;             // BF16 [dim]   (the last stage's, before the tied head)
    Ds41Tensor markov_embed;     // BF16 [vocab, markov_rank]
    Ds41Tensor markov_head;      // BF16 [vocab, markov_rank]
    Ds41Tensor confidence_proj;  // BF16 [1, dim + markov_rank]
    explicit operator bool() const { return bool(main_proj); }
};

class DeepSeek41Model {
public:
    // Opens `dir` (config.json + the safetensors shards) and binds every tensor.
    // Returns "" on success, human-readable error text otherwise.
    std::string load(const std::string& dir);

    const Ds41Config&              config() const { return cfg_; }
    const std::vector<Ds41Layer>&  layers() const { return layers_; }   // n_layers + n_mtp_layers
    const Ds41Budget&              budget() const { return budget_; }
    const SafetensorsModel&        store()  const { return store_; }
    // Names present in the checkpoint that no role claimed. Empty on a clean bind.
    const std::vector<std::string>& unclaimed() const { return unclaimed_; }

    Ds41Tensor embed, lm_head, final_norm;
    Ds41DSpark dspark;                       // the drafter's extras (DSpark P1)

private:
    SafetensorsModel         store_;
    Ds41Config               cfg_;
    std::vector<Ds41Layer>   layers_;
    Ds41Budget               budget_;
    std::vector<std::string> unclaimed_;
};

// Parse just the config (exposed so a test can check it without a 476 GiB checkpoint).
// Returns "" on success. `json_text` is the contents of config.json.
std::string ds41_parse_config(const std::string& json_text, Ds41Config& out);

}  // namespace ie
