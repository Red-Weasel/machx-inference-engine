// include/ie/laguna.hpp — Laguna-S-2.1 (`laguna`) runtime (docs/laguna/00_PORT_PLAN.md, P4).
//
// The forward mirrors upstream llama.cpp src/models/laguna.cpp op for op:
//   embed -> 48 x [ RMSNorm -> q/k/v + g_proj(same input) -> per-head QK RMSNorm -> rope (YaRN over 64 dims on full
//   layers, plain over 128 on sliding) -> attention (1/sqrt(128); a 512-key window on sliding layers) -> softplus
//   per-head gate -> o_proj -> +residual -> RMSNorm -> FFN (layer 0 dense; else sigmoid router with the selection
//   bias, top-10, renormalised, x2.5, plus the always-on shared expert) -> +residual ] -> RMSNorm -> head.
// Verified against the llama.cpp oracle (llama-eval-callback on the same ids): per-layer residuals track to layer 46,
// greedy text identical (ie-laguna-run).
// Placement (one card):
//  - DEVICE Q8_0, repacked to SoA at load (gemv_q8_0_soa_f16_*'s layout): every Q8_0 projection. Decode reads the
//    int8 weights (half the bytes of F16); prefill dequantizes each matrix into one F16 scratch for gemm_fp16.
//  - DEVICE F32: the norms, the router [256 x hidden] and its selection bias.  DEVICE raw Q6_K: the output head.
//  - Routed experts (gate/up Q4_K, down Q5_K or Q6_K): a PINNED host copy (68.8 GB; the GGUF mapping when the RAM
//    guard refuses) feeding a per-layer LRU cache of VRAM slots sized from the card's free memory. The MoE runs
//    EXPERT-MAJOR: each selected expert is acquired once per layer and runs over every row that picked it.
//  - Prefill (T > 1): the projections are batched FP16 GEMMs with fp32 outputs (gemm_fp16); decode: Q8_0 GEMVs.
//  - The residual stream is fp32, as in llama.cpp's graph. A SwiGLU output is scaled per row by a power of two
//    before it feeds an f16 GEMV (the BOS row's massive activations exceed f16 at the last layers).
// Two cards: the layers split between them (a second LagunaModel runs the second card's layers), the residual handed over
// through pinned host buffers; each card caches its own layers' experts, so the two caches together hold twice the experts
// one card does.
#pragma once

#include "ie/gguf.hpp"
#include "ie/model_config.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ie {

// A Q8_0 projection [K in, N out] on the card in gemv_q8_0_soa_f16_*'s SoA layout: qs [N][K] int8, d [N][K/32] fp16.
struct LagQ8 { int8_t* qs = nullptr; uint16_t* d = nullptr; };

struct LagunaLayer {
    uint32_t n_heads = 0, rope_dim = 0;
    float    rope_theta = 0.f;
    bool     sliding = false, moe = false;
    LagQ8 wq, wk, wv, wo, wg;
    LagQ8 ffn_gate, ffn_up, ffn_down;     // dense FFN (leading layers)
    LagQ8 sh_gate, sh_up, sh_down;        // shared expert
    // device F32
    float *attn_norm = nullptr, *ffn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    float *router = nullptr, *router_bias = nullptr;                                  // [E][hidden], [E]
    // routed-expert banks: the GGUF mapping, and the pinned copy when there is one
    const GgufTensorInfo *gate_exps = nullptr, *up_exps = nullptr, *down_exps = nullptr;
    const uint8_t *gate_src = nullptr, *up_src = nullptr, *down_src = nullptr;      // where a miss copies from
    uint64_t gate_slice = 0, up_slice = 0, down_slice = 0;                           // bytes per expert
    bool down_q5 = true;                                                              // else Q6_K
};

// One MoE layer's VRAM expert slots (LRU). A slot holds one expert's [gate | up | down] raw slices.
struct LagunaECache {
    uint8_t* base = nullptr;
    std::vector<int16_t>  slot_of;     // [E] -> slot, -1 = not resident
    std::vector<int16_t>  expert_of;   // [S] -> expert, -1 = empty
    std::vector<uint64_t> stamp;       // [S] last use
};

class LagunaModel {
public:
    ~LagunaModel();
    // `g` must outlive the model: the embeddings (and the experts when they are not pinned) are read from its mapping.
    // With q2, two cards: q's card runs the embeddings, the dense layers and half the MoE layers; q2's card the other half
    // and the head. Each queue must have its own single-device context (a two-device context mirrors VRAM into RAM).
    std::string load(sycl::queue& q, const GgufReader& g, const LagunaConfig& cfg, sycl::queue* q2 = nullptr);
    // Device KV and workspaces for sequences up to max_ctx positions, forward() calls up to max_chunk tokens; then the
    // expert cache takes the card's free memory minus IE_LAGUNA_RESERVE_GB (default 1.5).
    std::string init_runtime(uint32_t max_ctx, uint32_t max_chunk);
    // Run T tokens at absolute position pos0 (pos0 == 0 starts a new sequence; otherwise pos0 must equal pos()).
    // logits_host (optional): the LAST row's logits [vocab] as fp32.
    std::string forward(const int32_t* ids, uint32_t T, uint32_t pos0, float* logits_host);
    // A prompt's rows [pos0, pos0 + n) in chunks of max_chunk (pos0 == 0 or pos()); logits_host gets the last row's. On
    // two cards the cards overlap: card 1 runs chunk k+1 while card 2 runs chunk k. after_chunk(end), when given, is
    // called as each chunk is in on both cards; false stops the prefill -- on two cards after the chunk card 1 already
    // holds -- and pos() says where the sequence ends. IE_LAGUNA_OVERLAP=0: one chunk at a time on both cards.
    std::string prefill(const int32_t* ids, uint32_t n, uint32_t pos0, float* logits_host,
                        const std::function<bool(uint32_t end)>& after_chunk = {});
    // Prefix reuse: continue the sequence from `pos` (<= pos()). Attention has no recurrent state, so the full layers' KV
    // below pos stays valid as is; the sliding layers' ring must still hold the window before pos (the slots a longer
    // previous sequence overwrote are not). False = the window is gone: start again from 0.
    bool rewind(uint32_t pos);
    // Called after every layer with the residual stream [T, hidden] (device fp32 on q's card), for oracle comparisons.
    std::function<void(uint32_t layer, sycl::queue& q, const float* x_dev, uint32_t T)> probe;
    // Every quant binding (single- and multi-row) against a host dequant + matvec.
    std::string self_check();

    const LagunaConfig& config() const noexcept { return cfg_; }
    uint32_t cards() const noexcept { return next_ ? 2 : 1; }
    uint64_t device_bytes() const noexcept { return dev_bytes_ + (next_ ? next_->dev_bytes_ : 0); }   // both cards
    uint64_t pinned_bytes() const noexcept { return pinned_bytes_ + (next_ ? next_->pinned_bytes_ : 0); }
    uint32_t cache_slots() const noexcept { return next_ ? std::min(ec_slots_, next_->ec_slots_) : ec_slots_; }   // per layer, the fewer
    uint64_t cache_hits() const noexcept { return ec_hits_ + (next_ ? next_->ec_hits_ : 0); }
    uint64_t cache_misses() const noexcept { return ec_misses_ + (next_ ? next_->ec_misses_ : 0); }
    uint32_t pos() const noexcept { return pos_; }
    uint32_t max_ctx() const noexcept { return max_ctx_; }
    sycl::queue& queue() { return *q_; }

private:
    std::string load_stage(sycl::queue& q, const GgufReader& g, const LagunaConfig& cfg, uint32_t l0, uint32_t l1);
    std::string stage_forward(const int32_t* ids, uint32_t T, uint32_t pos0, float* logits_host);
    std::string stage_tag() const;
    template <class T> T* dev_alloc(uint64_t n);
    // Host <-> device copies of pageable memory (the GGUF mapping, std::vectors) go through hstage_, a pinned buffer: the
    // driver's own mapping of a pageable copy source was not present yet on a fresh context's first copies (the engine
    // read unmapped pages, DEVICE_LOST at ie serve load, 2026-10-16/18). Both wait for the copy.
    void to_device(void* dst, const void* src, uint64_t n);
    void from_device(void* dst, const void* src, uint64_t n);
    std::string upload_q8_soa(const GgufTensorInfo* t, uint32_t K, uint32_t N, LagQ8& out);
    void gemm_q8(const sycl::half* A, const LagQ8& W, float* C, uint32_t T, uint32_t K, uint32_t N);   // prefill: W via wscratch_
    std::string upload_f32(const GgufTensorInfo* t, uint64_t n, float*& out);
    std::string pin_experts();
    const uint8_t* acquire(const LagunaLayer& w, uint32_t L, int32_t e);          // the slot's device pointer
    std::string ffn_dense(const LagQ8& gate, const LagQ8& up, const LagQ8& down, uint32_t F, uint32_t T, float* dst);
    std::string moe_experts(const LagunaLayer& w, uint32_t L, uint32_t T);

    sycl::queue*        q_ = nullptr;
    const GgufReader*   g_ = nullptr;
    LagunaConfig        cfg_;
    uint32_t            l0_ = 0, l1_ = 0;        // this card's layers [l0_, l1_): l0_ == 0 embeds, l1_ == n_layers runs the head
    std::unique_ptr<LagunaModel> next_;          // two cards: the second card's layers
    float*              hx_ = nullptr;           // two cards: pinned host [chunk][hidden], the residual's hand-off
    uint8_t*            hstage_ = nullptr;       // pinned host staging for to_device / from_device
    uint64_t            hstage_n_ = 0;
    std::vector<LagunaLayer> layers_;
    std::vector<void*>  allocs_;
    uint64_t            dev_bytes_ = 0, pinned_bytes_ = 0;
    const GgufTensorInfo* tok_embd_ = nullptr;
    float*              out_norm_ = nullptr;
    uint8_t*            head_q6_ = nullptr;      // raw Q6_K [vocab rows of hidden]
    sycl::half*         wscratch_ = nullptr;     // prefill: one Q8_0 projection dequantized to F16 [K, N]
    uint64_t            wscratch_n_ = 0;         // its size: the largest K * N
    // expert cache
    std::vector<LagunaECache> ec_;
    uint64_t slot_bytes_ = 0, off_up_ = 0, off_down_ = 0, ec_clock_ = 0, ec_hits_ = 0, ec_misses_ = 0;
    uint32_t ec_slots_ = 0;
    bool     fused_moe_ = true;                  // decode MoE in three launches (IE_LAGUNA_FUSED_MOE=0: the expert-major loop)
    // runtime
    uint32_t max_ctx_ = 0, max_chunk_ = 0, pos_ = 0, max_heads_ = 0;
    std::vector<float*> kc_, vc_;                // per layer: full [max_ctx][n_kv][head_dim], sliding a ring [ring_][n_kv][head_dim] fp32
    // Sliding layers keep a RING of ring_ = window + max_chunk positions (a chunk's T new rows never overwrite a key one
    // of its queries still needs); ring_pos_[slot] = the position the slot holds (-1 none), shared by every sliding layer.
    uint32_t ring_ = 0;
    std::vector<int64_t> ring_pos_;
    float *x_ = nullptr, *xn_ = nullptr, *qf_ = nullptr, *kf_ = nullptr, *vf_ = nullptr, *gf_ = nullptr;
    float *attn_ = nullptr, *acc_ = nullptr, *rlog_ = nullptr, *rw_ = nullptr, *fy_ = nullptr, *fg_ = nullptr, *fu_ = nullptr;
    float *cos_full_ = nullptr, *sin_full_ = nullptr, *cos_swa_ = nullptr, *sin_swa_ = nullptr;
    float *mask_full_ = nullptr, *mask_swa_ = nullptr, *sc_ = nullptr, *gs_ = nullptr, *d_rw_ = nullptr;
    float *attn_part_ = nullptr;                 // decode attention: per (head, key split) partials [heads][splits][hd + 2]
    int32_t *ridx_ = nullptr, *d_rows_ = nullptr;
    sycl::half *xn16_ = nullptr, *q16_ = nullptr, *k16_ = nullptr, *v16_ = nullptr, *g16_ = nullptr, *fh16_ = nullptr;
    sycl::half *a16_ = nullptr, *o16_ = nullptr, *f1_ = nullptr, *f2_ = nullptr, *f3_ = nullptr, *logits16_ = nullptr;
    sycl::half *gx_ = nullptr, *gg_ = nullptr, *gu_ = nullptr, *gh_ = nullptr, *gy_ = nullptr;
    std::vector<float>   h_emb_, h_cos_, h_sin_, h_mask_, h_w_, h_rw_;
    std::vector<int32_t> h_idx_, h_rows_;
};

}  // namespace ie
