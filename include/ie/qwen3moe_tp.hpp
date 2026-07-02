// include/ie/qwen3moe_tp.hpp — Qwen3 standard MoE (`kQwen3Moe`, e.g.
// Qwen3-Coder-30B-A3B / Tongyi-DeepResearch) TENSOR-PARALLEL decode/prefill.
//
// The engine's SECOND MoE tensor-parallel path (after GptOssTpModel), and the
// first that ALSO head-shards attention as the primary lever. ADDITIVE glue over
// the single-GPU `Qwen3MoeModel` math (never edited) + `Qwen3MoeSplitModel`'s
// per-card MoE machinery (int-dot W4A8/W6A8) + `GptOssTpModel`'s host-bounce
// skeleton + `DeviceFleet::all_reduce_sum_fp16`.
//
// WHY (measured): the qwen3moe LAYER-SPLIT's long-context decode is bound by
// ATTENTION-AT-DEPTH — the FA compute over the growing KV runs at single-card
// parity (each card owns whole layers, so its attention is NOT parallelized;
// 2-card Tongyi decode ~46 tok/s short → ~21 at 17k, halving with ctx). TP
// head-shards the attention so BOTH cards compute it concurrently over half the
// heads + half the KV → the long-ctx decode lever. It ALSO shards the ~24 GB
// experts across cards → the high-ctx memory fit (Tongyi Q6_K ~25 GB, Coder
// Q4_K_M ~17 GB on 2×32 GB B70).
//
// DESIGN (2 all-reduces/layer, x replicated):
//   ATTENTION — HEAD-SHARD. Card c owns q-heads [c·nq, (c+1)·nq) (nq=n_q/N) and
//     kv-heads [c·nkv, (c+1)·nkv) (nkv=n_kv/N). q/k/v projections COLUMN-sliced to
//     the card's heads (quantized byte-slice, kept Q4_K/Q6_K), per-head QK-norm +
//     rope over the card's heads, per-card KV holds ONLY the card's kv-heads
//     (HALF KV/card — the memory win). Standard GQA — NO sinks/SWA/biases. o-proj
//     ROW-parallel (contraction split → dequant-to-fp16 Bt row-slice) → PARTIAL
//     [T,H] → all-reduce #1.
//   MoE — EXPERT-SHARD. Card c owns experts [c·E/N, (c+1)·E/N). Router REPLICATED
//     (each card gemv's the full router weight → identical host top-K → identical
//     FULL packing). Each card gathers the full expert-sorted rows but runs the
//     qwen3moe int-dot kernels over ONLY its experts (E_param = local count,
//     expert_offsets base = full_offsets + ef0_e, weight banks = the card's expert
//     slice); out_packed is memset so off-card packed rows stay 0 → the weighted
//     reduce yields a PARTIAL [T,H] → all-reduce #2. Full E_ffn=768/expert
//     (kernels UNCHANGED). Shards ~24 GB experts → ~12 GB/card.
//   Residual / RMSNorm / embedding / output_norm / lm_head REPLICATED (x stays in
//     lock-step via the two all-reduces). NOT bit-exact vs single-GPU (reduction
//     order); validate via coherent generation + the untouched single-GPU PPL gate.
//
// `--gpus 1` keeps `Qwen3MoeModel`; `kQwen3Moe && --gpus>1` DEFAULTS to the
// layer-split `Qwen3MoeSplitModel` (UNCHANGED); IE_QWEN3MOE_TP=1 + `--gpus>1`
// selects THIS path (mirrors the crown `IE_QWEN35_TP` opt-in).
#pragma once

#include "ie/allocator.hpp"          // DeviceFleet, DeviceAllocator
#include "ie/dense_transformer.hpp"  // DenseQuantPtr
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/model_config.hpp"       // Qwen3MoeConfig
#include "ie/qwen3moe_pack.hpp"      // MoePacking + build_moe_packing

#include <sycl/sycl.hpp>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ie {

class Qwen3MoeTpModel {
public:
    Qwen3MoeTpModel() = default;
    ~Qwen3MoeTpModel();
    Qwen3MoeTpModel(const Qwen3MoeTpModel&) = delete;
    Qwen3MoeTpModel& operator=(const Qwen3MoeTpModel&) = delete;

    // Load with experts expert-sharded + attention head-sharded across `fleet`.
    // max_ctx sizes each card's (halved) KV. `fleet` and `g` must outlive this. ""
    // on success.
    std::string load(DeviceFleet& fleet, const GgufReader& g,
                     const Qwen3MoeConfig& cfg, uint32_t max_ctx);
    const Qwen3MoeConfig& config() const noexcept { return cfg_; }
    std::vector<uint64_t> device_bytes() const { return dev_bytes_; }
    uint32_t n_devices() const noexcept { return n_dev_; }

    // Host ids in, last token's logits → host fp16 [vocab] (mirrors the other TP
    // paths; the engine bounces logits into d_logits_ so the GPU sampler is unchanged).
    std::string forward(const int32_t* input_ids, uint32_t T, uint32_t start_pos,
                        bool reset_kv, sycl::half* out_logits_host);

private:
    // Per-card shard geometry: expert slice [ef0_e, ef0_e+efn) + attention head
    // slice (q-heads [q_head0, q_head0+nq), kv-heads [kv_head0, kv_head0+nkv)).
    struct Shard {
        uint32_t ef0_e = 0, efn = 0;                 // expert range (count of experts)
        uint32_t q_head0 = 0, nq = 0, kv_head0 = 0, nkv = 0;
        uint32_t Nq = 0, Nkv = 0;                    // nq*head_dim, nkv*head_dim
    };

    // Per-card layer weights. Norms + QK-norm + router REPLICATED; attn q/k/v/o
    // head-sliced; experts EXPERT-sliced (the card's [ef0_e, ef0_e+efn) banks).
    struct LayerW {
        float* attn_norm   = nullptr;                // [H] replicated
        float* ffn_norm    = nullptr;                // [H] replicated
        float* attn_q_norm = nullptr;                // [head_dim] replicated (per-head QK-norm)
        float* attn_k_norm = nullptr;                // [head_dim] replicated
        // attention projections — q/k/v COLUMN-sliced (quantized byte-slice, kept
        // dtype); attn_output ROW-sliced (dequant-to-fp16 Bt [Nq_card, H]).
        DenseQuantPtr attn_q, attn_k, attn_v, attn_output;
        // MoE router (replicated full): device F16 [H, E] transposed for gemv_q_T.
        sycl::half* router_w_dev = nullptr;
        // MoE experts — the card's [ef0_e, ef0_e+efn) slice (stacked banks; per-expert
        // stride = full_nbytes/E unchanged). SoA (intra-expert reorder) only for Q6_K.
        void* gate_exps = nullptr; DType gate_dt = DType::kCount; uint64_t gate_stride = 0; bool gate_soa = false;
        void* up_exps   = nullptr; DType up_dt   = DType::kCount; uint64_t up_stride   = 0; bool up_soa   = false;
        void* down_exps = nullptr; DType down_dt = DType::kCount; uint64_t down_stride = 0; bool down_soa = false;
    };

    // Per-card forward scratch. Mirrors Qwen3MoeSplitModel::Workspace, but q/k/v/
    // attn_out sized to the card's sharded heads (Nq/Nkv), MoE packing FULL.
    struct Workspace {
        uint32_t T = 0;
        sycl::half *x = nullptr, *x_normed = nullptr;
        sycl::half *q = nullptr, *k = nullptr, *v = nullptr;  // sharded-head sizes
        sycl::half *attn_out = nullptr, *attn_block = nullptr;
        int32_t* positions = nullptr;
        float*   attn_partials = nullptr; uint32_t partials_ctx = 0;
        sycl::half* router_logits = nullptr;   // [T, E]  (full router, replicated)
        // fused MoE staging (FULL expert-sorted packing; the card runs only its experts).
        sycl::half* xp_packed  = nullptr;   // [T*K, H]
        void*       xp_q8      = nullptr;   // [T*K, H/32]   block_q8_1s
        sycl::half* h_packed   = nullptr;   // [T*K, E_ffn]
        void*       h_q8       = nullptr;   // [T*K, E_ffn/32] block_q8_1s
        sycl::half* out_packed = nullptr;   // [T*K, H]  memset each layer → PARTIAL rows
        sycl::half* ffn_y      = nullptr;   // [T, H]    PARTIAL MoE → all-reduce #2
        uint32_t*   expert_offsets = nullptr; // [E+1]  FULL packing offsets
        int32_t*    sorted_idx     = nullptr; // [T*K]
        int32_t*    tk_to_packed   = nullptr; // [T*K]
        sycl::half* weights_packed = nullptr; // [T*K]
    };

    // Per-card sharded MoE FFN for one layer → ws_[dev].ffn_y (PARTIAL). Routing +
    // FULL packing computed on THIS card (replicated router → identical top-K).
    void moe_ffn_card(uint32_t dev, const LayerW& w, uint32_t T);

    std::string ensure_ws(uint32_t dev, uint32_t max_T);
    void free_ws(uint32_t dev);
    void free_all();

    DeviceFleet*   fleet_ = nullptr;
    Qwen3MoeConfig cfg_;
    uint32_t       n_dev_ = 0;

    std::vector<Shard>                 shard_;    // [n_dev]
    std::vector<std::vector<LayerW>>   dlayers_;  // [n_dev][n_layers]
    std::vector<std::vector<void*>>    owned_;    // [n_dev]
    std::vector<Workspace>             ws_;       // [n_dev]
    std::vector<KvCache>               kv_;       // [n_dev] (halved: nkv kv-heads/card)
    std::vector<uint64_t>              dev_bytes_;// [n_dev] weight bytes (reporting)

    // globals on card 0 (embedding + final norm + lm_head — x replicated).
    void*  token_embd_ = nullptr;  DType token_embd_dtype_ = DType::kCount;
    float* output_norm_ = nullptr;
    void*  output_ = nullptr;      DType output_dtype_ = DType::kCount;

    std::vector<float> host_logits_;   // [T*E] router-logit host scratch (reused)
};

}  // namespace ie
