// include/ie/gptoss_tp.hpp — gpt-oss (OpenAI MoE) TENSOR-PARALLEL decode/prefill.
//
// The first MoE tensor-parallel path in the engine. ADDITIVE glue over the
// single-GPU GptOssModel math (lifted verbatim) + Qwen35TpModel's host-bounce
// skeleton + DenseModelTP's all-reduce. The single-GPU GptOssModel is NEVER
// edited; the crown/Coder/gemma MoE paths are untouched.
//
// PHASE 1 (this file): MoE expert-TP only. Attention runs REPLICATED (full
// weights, full heads) on every card from the replicated residual x → the
// attn_block is bit-identical across cards, so x stays in lock-step with NO
// attention all-reduce (exactly the Qwen35TpModel Phase-0 pattern, but with
// gpt-oss attn — sinks + alternating SWA — and gpt-oss MoE). Only the MXFP4
// experts are sharded:
//   within-expert TP — router REPLICATED (deterministic route, computed once on
//     card 0 → identical packing broadcast to every card). Each expert's gate/up
//     column-parallel (intermediate EF → EF/N), down row-parallel (the
//     contraction EF → EF/N, landing on MXFP4 32-element block boundaries) →
//     PARTIAL [T,H] → ONE all-reduce/layer. down_bias applied on RANK 0 ONLY
//     (row-parallel bias rule: it would double across the all-reduce).
//   Residual / norms / embedding / attention REPLICATED (RMSNorm needs full H;
//     the all-reduce scatters the sum back to every card so x stays in sync).
//
// 20b and 120b differ ONLY in n_layers and n_experts → one class serves both.
// NOT bit-exact vs single-GPU (reduction order differs); validate via cosine
// >=0.999 / greedy-match (the crown PPL gate stays green via the untouched
// single-GPU path). P2P is hw-blocked on 2xB70 → the all-reduce host-bounces.
//
// PHASE 2 (later): attention head-shard (q/k/v column, o-proj row, sinks + KV
// sliced to the head range) → halves KV/card + attention compute, the 120b@99k
// fit lever. PHASE 3: multi-shard GGUF loader + host-RAM spill → full 120b.
#pragma once

#include "ie/allocator.hpp"
#include "ie/dense_transformer.hpp"   // DenseQuantPtr
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/model_config.hpp"
#include "ie/qwen3moe_pack.hpp"       // MoePacking + build_moe_packing

#include <sycl/sycl.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ie {

class GptOssTpModel {
public:
    GptOssTpModel() = default;
    ~GptOssTpModel();
    GptOssTpModel(const GptOssTpModel&) = delete;
    GptOssTpModel& operator=(const GptOssTpModel&) = delete;

    // Load gpt-oss with the MXFP4 experts sharded across `fleet` (attn replicated).
    // max_ctx sizes each card's KvCache. "" on success.
    std::string load(DeviceFleet& fleet, const GgufReader& g,
                     const GptOssConfig& cfg, uint32_t max_ctx);
    const GptOssConfig& config() const noexcept { return cfg_; }

    // IE_GPTOSS_TP_TIMING attribution (Step-0 decode breakdown). When the env flag
    // is set, forward() accumulates per-section wall-time (attn / all-reduces /
    // router / MoE-gemv / lm_head) into internal buckets. reset before a timed run,
    // report after. PPL-neutral: with the flag unset the forward path is unchanged
    // (no extra barriers, no member writes).
    void reset_tp_timing() noexcept;
    void report_tp_timing(uint32_t steps) const;

    // Host ids in, host logits out (mirrors Qwen35TpModel) — the engine bounces
    // logits into d_logits_ so the GPU sampler is unchanged.
    std::string forward(const int32_t* input_ids, uint32_t T, uint32_t start_pos,
                        bool reset_kv, sycl::half* out_logits_host);

private:
    // Per-card shard geometry: expert-ffn slice [ef0,ef0+efc) (Phase 1) + attention
    // head slice (Phase 2 — q-heads [q_head0,q_head0+nq), kv-heads [kv_head0,+nkv)).
    struct Shard {
        uint32_t ef0 = 0, efc = 0;
        uint32_t q_head0 = 0, nq = 0, kv_head0 = 0, nkv = 0;
        uint32_t Nq = 0, Nkv = 0;          // nq*head_dim, nkv*head_dim
    };

    // Per-card layer weights. Attention + router REPLICATED full; experts EF-sliced.
    struct LayerW {
        float* attn_norm = nullptr;        // [H]
        float* post_attn_norm = nullptr;   // [H]
        DenseQuantPtr attn_q, attn_k, attn_v, attn_output;  // F16 (full, replicated)
        float* attn_q_bias = nullptr;      // [N_q]
        float* attn_k_bias = nullptr;      // [N_kv]
        float* attn_v_bias = nullptr;      // [N_kv]
        float* attn_o_bias = nullptr;      // [H]
        float* attn_sinks  = nullptr;      // [n_q_heads]
        bool   is_swa = false;

        // router (replicated full on every card; routing computed once on card 0).
        sycl::half* router_w_dev = nullptr;  // F16 [H,E]
        float*      router_bias  = nullptr;  // F32 [E]

        // experts (MXFP4 SoA, EF-sliced): gate/up column [ef0,ef0+efc) → K=H, N=efc.
        //   down row [ef0,ef0+efc) → K=efc, N=H (PARTIAL → all-reduce).
        uint8_t* gate_qs = nullptr; uint8_t* gate_e = nullptr;
        uint8_t* up_qs   = nullptr; uint8_t* up_e   = nullptr;
        uint8_t* down_qs = nullptr; uint8_t* down_e = nullptr;
        uint64_t gate_qs_stride = 0, gate_e_stride = 0;
        uint64_t up_qs_stride   = 0, up_e_stride   = 0;
        uint64_t down_qs_stride = 0, down_e_stride = 0;
        float* gate_bias = nullptr;        // F32 [E*efc] (EF-sliced)
        float* up_bias   = nullptr;        // F32 [E*efc] (EF-sliced)
        float* down_bias = nullptr;        // F32 [E*H] — RANK 0 ONLY (else doubled)
    };

    // Per-card workspace.
    struct Workspace {
        uint32_t T = 0;
        sycl::half *x = nullptr, *xn = nullptr, *block = nullptr;   // [T,H]
        sycl::half *q = nullptr, *attn_out = nullptr;               // [T,N_q]
        sycl::half *k = nullptr, *v = nullptr;                      // [T,N_kv]
        int32_t* positions = nullptr;
        // MoE scratch (EF→efc).
        sycl::half *moe_logits = nullptr;  // [T,E]  (card 0)
        sycl::half *moe_xp = nullptr;      // [T*K,H]
        sycl::half *moe_h  = nullptr;      // [T*K,efc]
        sycl::half *moe_h2 = nullptr;      // [T*K,efc]
        sycl::half *moe_out = nullptr;     // [T*K,H]
        sycl::half *moe_y  = nullptr;      // [T,H]  PARTIAL → all-reduce
        sycl::half *moe_wpk = nullptr;     // [T*K]
        sycl::half *moe_btf16 = nullptr;   // [H*efc] prefill dequant Bt
        sycl::half *moe_pout = nullptr;    // [32*H] small-M pad-GEMM output scratch
        int32_t *moe_sorted = nullptr;     // [T*K]
        int32_t *moe_tk2pk  = nullptr;     // [T*K]
        void *q8_x = nullptr;              // [H/32]   block_q8_1x (decode gate/up act)
        void *q8_h = nullptr;              // [efc/32] block_q8_1x (decode down act)
        float* attn_partials = nullptr;    // FA-2 decode split-K partials [n_chunks,n_q,HD+2]
        uint32_t partials_ctx = 0;
    };

    // Per-card top-4 MoE (efc-sliced): gather → gate/up+swiglu_oai → down → PARTIAL
    // weighted-reduce into ws.moe_y [T,H]. Routing (pk_) computed once on card 0.
    void moe_ffn_card(uint32_t c, uint32_t L, uint32_t T);

    std::string ensure_ws(uint32_t dev, uint32_t max_T);
    void free_ws(uint32_t dev);
    void free_all();

    DeviceFleet* fleet_ = nullptr;
    uint32_t n_dev_ = 0;
    bool attn_shard_ = true;   // Phase 2: shard attention heads (both cards compute attn
                               // + halved KV/card). opt-out IE_GPTOSS_TP_REPLICATE_ATTN → Phase 1.
    GptOssConfig cfg_;
    std::vector<Shard> shard_;                         // [n_dev]
    std::vector<std::vector<LayerW>> dlayers_;          // [n_dev][n_layers]
    std::vector<std::vector<void*>> owned_;             // [n_dev]
    std::vector<uint64_t> dev_bytes_;
    std::vector<Workspace> ws_;                         // [n_dev]
    std::vector<KvCache> kv_;                           // [n_dev]

    // globals on card 0 (embedding + final norm + lm_head — x replicated).
    void*  token_embd_ = nullptr;  DType token_embd_dtype_ = DType::kCount;
    float* output_norm_ = nullptr;
    int8_t*  lmhead_qs_ = nullptr; uint16_t* lmhead_d_ = nullptr;
    uint32_t lmhead_K_ = 0, lmhead_N_ = 0;

    // shared host routing scratch (computed once per layer on card 0).
    std::vector<float> host_logits_;
    std::vector<std::vector<std::pair<uint32_t,float>>> host_routes_;
    MoePacking pk_;

    // IE_GPTOSS_TP_TIMING buckets (ms, accumulated across forward() calls).
    double tt_attn_ = 0, tt_ar_attn_ = 0, tt_router_ = 0,
           tt_moe_ = 0, tt_ar_moe_ = 0, tt_lmhead_ = 0, tt_misc_ = 0;
};

}  // namespace ie
