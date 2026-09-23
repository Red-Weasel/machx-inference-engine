// include/ie/mimo26.hpp — MiMo-V2.6 (Xiaomi, HF arch `mimo_v2` / `MiMoV2ForCausalLM`) config + weight binding.
//
// P1 territory only (docs/mimo26/00_PORT_PLAN.md): host-side parse of `config.json`, host-side binding
// and shape validation over the safetensors mmap, and the host dequant references the P1 gate compares
// against llama.cpp's converter. No device memory, no compute.
//
// Every pointer is a non-owning view into SafetensorsModel's mmaps (the V4.1 convention: the model
// owns nothing and is invalidated when the reader closes). Shapes are safetensors order, [out, in],
// verified against the shipped checkpoint on 2026-09-22 (plan section "Model facts").
#pragma once

#include "ie/deepseek41.hpp"   // Ds41Tensor: the (weight, scale) pair the expert tier deals in
#include "ie/safetensors.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace ie {

// Fields are named for the HF config keys. Nothing here has a default that could silently stand in
// for a missing key: `load` errors. The two keys the checkpoint ships as JSON null
// (`routed_scaling_factor`, `bos_token_id`) are the only ones allowed to be null.
struct Mimo26Config {
    uint32_t vocab_size = 0, dim = 0, inter_dim = 0, moe_inter_dim = 0;
    uint32_t n_layers = 0, n_mtp_layers = 0;
    uint32_t n_heads = 0, n_kv_heads = 0, n_kv_heads_swa = 0, head_dim = 0, v_head_dim = 0;
    uint32_t window = 0;                     // sliding_window (128): SWA layers see keys within it
    float    value_scale = 0.f;              // attention_value_scale (0.707), applied to the attention output
    float    norm_eps = 0.f;                 // layernorm_epsilon
    float    rope_theta = 0.f, rope_theta_swa = 0.f, partial_rotary = 0.f;
    std::vector<uint32_t> hybrid;            // per layer: 1 = sliding-window attention, 0 = full
    std::vector<uint32_t> moe_layer;         // per layer: 1 = routed MoE, 0 = dense FFN
    uint32_t n_routed_experts = 0, n_activated_experts = 0, n_group = 0, topk_group = 0;
    std::string topk_method, scoring_func;   // "noaux_tc", "sigmoid"
    bool     norm_topk_prob = false;
    float    route_scale = 1.f;              // routed_scaling_factor; null in the file = 1
    bool     sink_full = false, sink_swa = false;   // add_{full,swa}_attention_sink_bias
    uint32_t fp8_block_n = 0, fp8_block_k = 0;      // quantization_config.weight_block_size (128 x 128)
    uint32_t mxfp4_block = 0;                       // quantization_config.mxfp4_block_size (32)
    int32_t  eos_id = -1, pad_id = -1;
    uint32_t image_token_id = 0;

    // The rotated slice of a head: int(head_dim * partial_rotary_factor) = 64 (the converter's formula).
    uint32_t rope_dim() const { return uint32_t(double(head_dim) * double(partial_rotary)); }
    // Layer kinds, MTP layers included (they are SWA layers with a dense FFN, per the checkpoint).
    bool     is_mtp(uint32_t L) const { return L >= n_layers; }
    bool     is_swa(uint32_t L) const { return is_mtp(L) || hybrid[L] == 1; }
    bool     is_moe(uint32_t L) const { return !is_mtp(L) && moe_layer[L] == 1; }
    uint32_t n_kv(uint32_t L) const { return is_swa(L) ? n_kv_heads_swa : n_kv_heads; }
    bool     has_sink(uint32_t L) const { return is_swa(L) ? sink_swa : sink_full; }
    // Fused qkv row counts, unsharded [Q | K | V] order.
    uint32_t q_rows() const { return n_heads * head_dim; }
    uint32_t k_rows(uint32_t L) const { return n_kv(L) * head_dim; }
    uint32_t v_rows(uint32_t L) const { return n_kv(L) * v_head_dim; }
    uint32_t qkv_rows(uint32_t L) const { return q_rows() + k_rows(L) + v_rows(L); }
};

struct Mimo26Layer {
    bool     swa = false, moe = false, mtp = false, has_sink = false;
    uint32_t n_kv = 0, qkv_tp = 0;   // qkv_tp: the tensor-parallel shard count the fused qkv rows were saved in (from the scale plane)

    // -- attention (every layer) --------------------------------------------------------------
    Ds41Tensor attn_norm;   // BF16 [dim]                              (input_layernorm)
    Ds41Tensor qkv;         // FP8  [qkv_rows, dim] + `_scale_inv` F32 [tp * ceil(rows/tp/128), dim/128]
                            //      rows are TP-sharded: rank r holds [Q_r | K_r | V_r]; see mimo26_qkv_dequant_f32
    Ds41Tensor o_proj;      // BF16 [dim, n_heads * v_head_dim]
    Ds41Tensor sink;        // BF16 [n_heads]                          (SWA layers and MTP layers only)

    // -- FFN ------------------------------------------------------------------------------------
    Ds41Tensor ffn_norm;    // BF16 [dim]                              (post_attention_layernorm / MTP pre_mlp_layernorm)
    // dense (layer 0 and the MTP layers): FP8 with a 128x128 F32 `_scale_inv` grid
    Ds41Tensor mlp_gate, mlp_up;   // FP8 [inter_dim, dim] + [inter_dim/128, dim/128]
    Ds41Tensor mlp_down;           // FP8 [dim, inter_dim] + [dim/128, inter_dim/128]
    // routed (layers 1..47)
    Ds41Tensor gate_w;      // BF16 [E, dim]                           (mlp.gate.weight)
    Ds41Tensor gate_bias;   // F32  [E]                                (mlp.gate.e_score_correction_bias)
    // MXFP4 planes, compressed-tensors layout: nibbles [N, K/2] (U8, bound as kI8 -- element 2i in
    // the low nibble), E8M0 `_scale` [N, K/32]. gate/up: N = moe_inter_dim, K = dim; down: N = dim.
    std::vector<Ds41Tensor> exp_w1, exp_w3, exp_w2;   // gate_proj, up_proj, down_proj

    // -- MTP extras (mtp) -----------------------------------------------------------------------
    Ds41Tensor eh_proj;     // BF16 [dim, 2 * dim]
    Ds41Tensor enorm, hnorm, final_norm;   // BF16 [dim]
};

// Byte accounting by group, so a residency plan can be costed before anything is uploaded.
struct Mimo26Budget {
    uint64_t routed_experts = 0, attention = 0, dense_ffn = 0, routers = 0, norms = 0;
    uint64_t embed = 0, lm_head = 0, mtp = 0, vision = 0, audio = 0;
    uint64_t total() const {
        return routed_experts + attention + dense_ffn + routers + norms + embed + lm_head + mtp + vision + audio;
    }
};

class Mimo26Model {
public:
    // Opens `dir` (config.json + the safetensors shards) and binds every text-path tensor.
    // Returns "" on success, human-readable error text otherwise.
    std::string load(const std::string& dir);

    const Mimo26Config&             config() const { return cfg_; }
    const std::vector<Mimo26Layer>& layers() const { return layers_; }   // n_layers + n_mtp_layers
    const Mimo26Budget&             budget() const { return budget_; }
    const SafetensorsModel&         store()  const { return store_; }
    // Names present in the checkpoint that no role claimed (the vision/audio encoders and the speech
    // embeddings are claimed by prefix). Empty on a clean bind.
    const std::vector<std::string>& unclaimed() const { return unclaimed_; }

    Ds41Tensor embed, lm_head, final_norm;   // BF16 [vocab, dim], BF16 [vocab, dim], BF16 [dim]

private:
    SafetensorsModel          store_;
    Mimo26Config              cfg_;
    std::vector<Mimo26Layer>  layers_;
    Mimo26Budget              budget_;
    std::vector<std::string>  unclaimed_;
};

// Parse just the config (exposed so a test can check it without the 178 GB checkpoint).
std::string mimo26_parse_config(const std::string& json_text, Mimo26Config& out);

// ---- host dequant references (src/model/mimo26_dequant.cpp) --------------------------------------
// FP8 E4M3 [N, K] times an F32 `scale_inv` grid of block_n x block_k blocks -> fp32 [N, K], row-major.
// The scale plane must be [ceil(N/block_n), ceil(K/block_k)]. "" on success.
std::string mimo26_fp8_dequant_f32(const Ds41Tensor& t, uint32_t block_n, uint32_t block_k, std::vector<float>& out);

// The same weight kept in FP8 (P4 lever 3: the device GEMV reads the checkpoint's bytes): E4M3 [N, K] bytes as stored and
// srow [N, SK] -- each row's F32 scale for every block_k columns (the block_n x block_k plane expanded per row).
// mimo26_fp8_rows_to_f32 is the dequant both f32 forms are: out[n][k] = e4m3(w8[n][k]) * srow[n][k / block_k].
std::string mimo26_fp8_rows(const Ds41Tensor& t, uint32_t block_n, uint32_t block_k, std::vector<uint8_t>& w8, std::vector<float>& srow);
void mimo26_fp8_rows_to_f32(const uint8_t* w8, const float* srow, uint32_t N, uint32_t K, uint32_t block_k, std::vector<float>& out);

// The fused qkv projection: FP8 rows saved TP-sharded -- rank r holds [Q_r | K_r | V_r] with q_rows/tp,
// k_rows/tp, v_rows/tp rows, and the scale plane has ceil(rows_per_rank / block) block-rows PER RANK
// (the last one covering fewer rows when rows_per_rank is not a multiple of block). Dequantises with the
// per-rank scale rows and regroups into the unsharded [Q | K | V] order -> fp32 [q+k+v rows, K].
// `tp` is detected from the scale plane's row count, candidates 8, 4, then 1 (the llama.cpp converter's
// `_tp_aware_qkv_dequant`, conversion/mimo.py), and returned in `tp_out`. "" on success.
// `force_tp` (diagnostic, 0 = detect) takes that shard count instead, if the scale plane fits it too.
std::string mimo26_qkv_dequant_f32(const Ds41Tensor& t, uint32_t q_rows, uint32_t k_rows, uint32_t v_rows,
                                   uint32_t block, std::vector<float>& out, uint32_t* tp_out, uint32_t force_tp = 0);
// The regrouped rows in FP8 (the same TP detection and [Q | K | V] order; per-row scales as mimo26_fp8_rows).
std::string mimo26_qkv_fp8_rows(const Ds41Tensor& t, uint32_t q_rows, uint32_t k_rows, uint32_t v_rows, uint32_t block,
                                std::vector<uint8_t>& w8, std::vector<float>& srow, uint32_t* tp_out, uint32_t force_tp = 0);

// MXFP4 in the checkpoint's layout: nibbles [N, K/2] with element 2i in the LOW nibble of byte i and
// 2i+1 in the high one, E8M0 [N, K/32]. A weight is e2m1(nibble) * 2^(e - 127). Scalar reference.
void mimo26_mxfp4_dequant_ref(const uint8_t* qs, const uint8_t* e, uint32_t N, uint32_t K, float* out);

// BF16 [n] -> fp32 [n] (a shift; exact).
void mimo26_bf16_to_f32(const uint8_t* src, uint32_t n, float* out);

}  // namespace ie
