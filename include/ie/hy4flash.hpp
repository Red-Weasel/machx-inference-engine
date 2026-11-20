// include/ie/hy4flash.hpp — Tencent Hy4-flash (HF `hy_v4_flash` / `HYV4FlashForConditionalGeneration`): config + weight
// binding. Plan and architecture notes: docs/hy4flash/00_PORT_AND_ABLITERATION_PLAN.md. Reference: transformers PR 50903.
//
// Host only: no device memory, no compute. Every pointer is a non-owning view into a safetensors mmap (the V4.1 / MiMo
// convention: the model owns nothing and is invalidated when the readers close).
//
// The model directory is the ENGINE's directory, written by scripts/hy4flash_oracle/pack.py:
//   hy4flash_ie.json           the source directory (Tencent's BF16 shards, read in place), the n-gram hash multipliers
//   config.json                copied from the source
//   experts-LNN.safetensors    every routed expert as MXFP4 planes (the layout Ds41ExpertTier reads) + an index file
// Dense tensors, the n-gram tables, vision and the drafter stay in the source directory.
//   overlay.safetensors        optional: tensors under the SOURCE's names (same shape) that replace the source's -- how an
//                              edited model (our abliteration: ~130 small matrices) is kept beside the untouched release
//
// Shapes are safetensors order, [out, in]. Names are the RELEASED checkpoint's (the reference module's differ in five
// places, see hy4flash_common.py).
#pragma once
#include "ie/deepseek41.hpp"   // Ds41Tensor: the (weight, scale) pair the expert tier deals in
#include "ie/safetensors.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace ie {

struct Hy4fConfig {
    uint32_t vocab_size = 0, dim = 0, inter_dim = 0, moe_inter_dim = 0, latent_dim = 0, n_layers = 0;
    uint32_t n_heads = 0, n_heads_swa = 0;            // indexed layers 32, sliding layers 64
    uint32_t q_lora_rank = 0, kv_lora_rank = 0;       // 1024, 512
    uint32_t nope_dim = 0, rope_dim = 0, v_dim = 0;   // 192, 64, 256 (the same on both layer kinds; load refuses otherwise)
    uint32_t window = 0;                              // sliding_window: a sliding layer sees the last `window` keys, itself included
    uint32_t n_experts = 0, top_k = 0, n_shared = 0;
    float    route_scale = 0.f, swiglu_limit = 0.f;
    bool     norm_topk_prob = false;
    uint32_t hc_mult = 0;
    float    hc_magnitude = 0.f, hc_eps = 0.f, norm_eps = 0.f;
    // nn.RMSNorm modules built without an eps (the n-gram layer's) use the epsilon of the dtype the model runs in:
    // BF16 -> 2^-7. Taken from the top-level "dtype"; the reference in bf16 is the behaviour reproduced.
    float    torch_default_eps = 0.f;
    uint32_t ix_ratio = 0, ix_heads = 0, ix_head_dim = 0, ix_group_topk = 0;   // 4, 16, 128, 512
    std::vector<uint8_t> indexed;                     // per layer: 1 = indexed (sparse) attention, 0 = sliding window
    std::vector<uint8_t> moe;                         // per layer: 1 = routed MoE, 0 = dense FFN
    float    theta_indexed = 0.f, theta_swa = 0.f;
    uint32_t mrope_indexed[3] = {0, 0, 0}, mrope_swa[3] = {0, 0, 0};
    std::vector<uint32_t> engram_layer_ids;           // 1-based: id n applies before layer n - 1's attention
    uint32_t engram_max_ngram = 0, engram_heads = 0, engram_embed = 0, engram_pad_id = 0;   // 3, 8, 768
    bool     engram_fp8 = false;
    int32_t  engram_seed = 0;
    int32_t  bos_id = -1, eos_id = -1, pad_id = -1;
    int32_t  image_token_id = -1, video_token_id = -1, image_start_id = -1, image_end_id = -1;   // top level of config.json (-1 = none)
    // The drafter the checkpoint carries (model.dfly_layers.*; top level of config.json): its layers, the mask token of the
    // block, the target layers whose keys / indexer choices each draft layer reads, the drafts a block proposes.
    uint32_t dfly_layers = 0, dfly_drafts = 0; int32_t dfly_mask_id = -1; std::vector<uint32_t> dfly_kv_layers;

    uint32_t heads(uint32_t L) const { return indexed[L] ? n_heads : n_heads_swa; }
    uint32_t qk_dim() const { return nope_dim + rope_dim; }
    uint32_t ix_budget() const { return ix_group_topk * ix_ratio; }       // candidate_topk: keys an indexed query keeps
    uint32_t engram_head_dim() const { return engram_embed / engram_heads; }                 // 96
    uint32_t engram_n_heads() const { return (engram_max_ngram - 1) * engram_heads; }        // 16
    uint32_t engram_width() const { return (engram_max_ngram - 1) * engram_embed; }          // 1536
};

struct Hy4fEngram {
    uint32_t id = 0;                         // the config's layer id (model layer + 1)
    std::vector<uint64_t> head_sizes;        // [(max_ngram - 1) * heads] prime moduli, n-gram order 2 first
    std::vector<uint64_t> head_offsets;      // row at which each head's slice of the flat table starts
    std::vector<int64_t>  multipliers;       // [max_ngram] odd hash multipliers (the sidecar's; numpy's generator)
    uint32_t table_size = 0;                 // lookup_table entries (token id -> compressed id)
    Ds41Tensor table;                        // BF16 [sum(head_sizes), head_dim]      -- 31.7 GB each in the release; w == nullptr when fp8 is set
    // The same table as the exact FP8 values the forward rounds every row to (<ie_dir>/engram-<id>.fp8, written by
    // ie-hy4flash-run --pack-tables: half the size, no BF16 source needed): rows of [fp32 scale | head_dim E4M3 codes].
    const uint8_t* fp8 = nullptr; uint64_t fp8_rows = 0;
    uint64_t rows() const { uint64_t r = 0; for (uint64_t s : head_sizes) r += s; return r; }
    Ds41Tensor lookup;                       // I32  [table_size]
    std::vector<Ds41Tensor> head_norms;      // [n_heads] x [head_dim]
    Ds41Tensor value_w, value_b;             // [dim, width], [dim]
    std::vector<Ds41Tensor> key_w, key_b;    // [hc] x [dim, width], [dim]
    std::vector<Ds41Tensor> norm_key, norm_query, hc_scales;   // [hc] x [dim]
    Ds41Tensor pre_gamma;                    // [dim]
};

struct Hy4fLayer {
    bool     indexed = false, moe = false;
    int32_t  engram = -1;                    // index into Hy4FlashModel::engrams(), -1 = none
    uint32_t n_heads = 0;
    // -- hyper-connections: fn [2 * hc, hc * dim], base [2 * hc], scale [2] --------------------------
    Ds41Tensor hc_attn_fn, hc_attn_base, hc_attn_scale, hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
    // -- attention (gated MLA) ----------------------------------------------------------------------
    Ds41Tensor attn_norm;                    // [dim]                       input_layernorm
    Ds41Tensor q_a, q_a_norm, q_b;           // [q_lora, dim], [q_lora], [heads * (nope + rope), q_lora]
    Ds41Tensor kv_a, kv_a_norm, kv_b;        // [kv_lora + rope, dim], [kv_lora], [heads * (nope + v), kv_lora]
    Ds41Tensor gate, o, post_norm, sinks;    // [heads * v, dim], [dim, heads * v], [dim], [heads]
    // -- compressed indexer (indexed layers) ----------------------------------------------------------
    Ds41Tensor ix_q;                         // [ix_heads * ix_head_dim, q_lora]
    Ds41Tensor ix_head_w;                    // [ix_heads, dim]
    Ds41Tensor ix_tok_k, ix_tok_norm;        // [ix_head_dim, dim], [ix_head_dim]
    Ds41Tensor ix_kv, ix_kg, ix_key_norm;    // [ix_head_dim, ix_head_dim] x 2, [ix_head_dim]
    Ds41Tensor ix_ape;                       // [ix_ratio, ix_head_dim]
    // -- FFN ----------------------------------------------------------------------------------------
    Ds41Tensor ffn_norm;                     // [dim]                       post_attention_layernorm
    Ds41Tensor mlp_gate, mlp_up, mlp_down, mlp_post_norm;     // dense: [inter, dim] x 2, [dim, inter], [dim]
    Ds41Tensor router_w, router_bias;        // [E, dim], [E]
    Ds41Tensor lat_down, lat_up;             // [latent, dim], [dim, latent]   (shared by the layer's routed experts)
    Ds41Tensor sh_gate, sh_up, sh_down;      // [moe_inter, dim] x 2, [dim, moe_inter]
    Ds41Tensor moe_norm;                     // [dim]                       combine_output_norm
    Ds41Tensor exp_gate_up, exp_down;        // the source's BF16 stacks [E, 2 * moe_inter, latent], [E, latent, moe_inter]
    // The pack (empty when the directory has none for this layer): MXFP4 planes [N, K / 2] + exponents [N, K / 32].
    std::vector<Ds41Tensor> exp_w1, exp_w3, exp_w2;           // gate, up: N = moe_inter, K = latent; down: N = latent
};

class Hy4FlashModel {
public:
    // Opens the engine directory (see the file comment). "" on success, human-readable error text otherwise.
    std::string load(const std::string& ie_dir);

    const Hy4fConfig&              config()  const { return cfg_; }
    const std::vector<Hy4fLayer>&  layers()  const { return layers_; }
    const std::vector<Hy4fEngram>& engrams() const { return engrams_; }
    const SafetensorsModel&        source()  const { return src_; }
    const SafetensorsModel&        pack()    const { return pack_; }
    bool                           has_pack() const { return has_pack_; }     // every MoE layer has its MXFP4 planes
    bool                           has_source_experts() const { return has_src_experts_; }   // the BF16 expert stacks are in the source directory
    ~Hy4FlashModel();
    const std::string&             source_dir() const { return source_dir_; }
    const std::string&             dir() const { return dir_; }                // the engine directory load() was given
    // Text-path names of the source no role claimed (vision `vit.*` and the drafter `model.dfly_layers.*` are claimed by
    // prefix until their phases bind them). Empty on a clean bind.
    const std::vector<std::string>& unclaimed() const { return unclaimed_; }
    size_t   overlay_tensors() const { return n_overlay_; }  // source tensors the directory's overlay.safetensors replaced
    uint64_t dense_bytes_f16() const { return dense_f16_; }   // what the text path's non-expert matrices take as fp16

    Ds41Tensor embed, lm_head, final_norm;                    // [vocab, dim], [vocab, dim], [dim]
    Ds41Tensor head_fn, head_base, head_scale;                // [hc, hc * dim], [hc], [1]

private:
    SafetensorsModel          src_, pack_;
    SafetensorsReader         ovl_;
    size_t                    n_overlay_ = 0;
    bool                      has_pack_ = false, has_src_experts_ = true;
    std::vector<std::pair<void*, size_t>> maps_;        // the FP8 n-gram files
    std::string               source_dir_, dir_;
    Hy4fConfig                cfg_;
    std::vector<Hy4fLayer>    layers_;
    std::vector<Hy4fEngram>   engrams_;
    std::vector<std::string>  unclaimed_;
    uint64_t                  dense_f16_ = 0;
};

// Parse just the config (the release's config.json: text fields under "text_config"). "" on success.
std::string hy4f_parse_config(const std::string& json_text, Hy4fConfig& out,
                              std::vector<std::vector<uint64_t>>* engram_head_sizes = nullptr);

// A float tensor of the checkpoint (BF16, F32 or F16) as fp32, exactly.
std::string hy4f_to_f32(const Ds41Tensor& t, std::vector<float>& out);

// The n-gram hash of one position: `ngram` holds the token at the position first, then the tokens before it (pad id past
// the start of the sequence), as HYV4FlashTextModel.build_ngram_tokens stacks them. Writes one table ROW per head
// (the head's offset included) into rows[n_heads].
void hy4f_engram_rows(const Hy4fEngram& eg, const Hy4fConfig& cfg, const int32_t* lookup, const int32_t* ngram, uint64_t* rows);

}  // namespace ie
