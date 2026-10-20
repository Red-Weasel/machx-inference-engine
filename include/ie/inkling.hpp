// include/ie/inkling.hpp — Inkling-Small (`inkling`) runtime (docs/inkling/00_PORT_PLAN.md, P4).
//
// The forward mirrors llama.cpp's Inkling graph (PR #25731, src/models/inkling.cpp) op for op:
//   embed -> RMSNorm(token_embd_norm) -> 42 x [ RMSNorm -> q/k/v/r projections -> shortconv on k and on v -> per-head
//   q/k RMSNorm -> attention: q.k / head_dim + the relative bias (head h's 16 r values dotted with rel_proj's column at
//   distance pos_q - pos_k, 0 at or past rel_extent; a 512-key window on sliding layers; on full layers q AND the bias
//   are scaled by tau(pos) = 1 + alpha ln max((pos+1)/n_floor, 1), 1 below 128K) -> o_proj -> shortconv -> +residual ->
//   RMSNorm -> FFN (layers 0-1 dense x gscale; else a sigmoid router with the selection bias, top-6, the WEIGHTS a softmax
//   over logsigmoid of the 6 picked raw logits AND the 2 shared-expert logits, x expert_weights_scale x gscale; a shared
//   expert multiplies its SwiGLU by its weight before down) -> shortconv -> +residual ] -> RMSNorm -> x 1/16 -> head
//   (ids past the unpadded vocabulary -inf). No rope: position is the relative bias and the four shortconvs.
// shortconv(x) = x + causal depthwise 4-tap conv; each of the 4 per layer keeps the last 3 inputs (the rolling state).
// Placement, as LagunaModel: the Q8_0 projections repacked to SoA on the card (decode: gemv_q8_0_soa_f16_*; prefill:
// dequantized into one F16 scratch for gemm_fp16); routed experts pinned in host memory feeding a per-layer LRU of VRAM
// slots; decode MoE fused over the picks; decode attention split over the keys. Two cards: the layers split between them
// (a second InklingModel runs the second card's layers), the residual handed over through pinned host buffers; each card
// caches its own layers' experts, so the two caches together hold twice the experts one card does.
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
struct InkQ8 { int8_t* qs = nullptr; uint16_t* d = nullptr; };

struct InklingLayer {
    bool     sliding = false, moe = false;
    uint32_t n_kv_heads = 0, rel_extent = 0;
    float    gscale = 1.f;                                 // ffn_gscale (F32 [1])
    InkQ8 wq, wk, wv, wr, wo;
    InkQ8 ffn_gate, ffn_up, ffn_down;                      // dense FFN (leading layers)
    InkQ8 sh_gate[2], sh_up[2], sh_down[2];                // the two shared experts
    // device F32
    float *attn_norm = nullptr, *ffn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    float *rel_proj = nullptr;                             // [d_rel][rel_extent]
    float *sc_k = nullptr, *sc_v = nullptr, *sc_attn = nullptr, *sc_mlp = nullptr;   // [C][4]
    float *router = nullptr, *router_bias = nullptr;       // [routed + shared][hidden], [routed]
    // routed-expert banks: the GGUF mapping, and the pinned copy when there is one
    const GgufTensorInfo *gate_exps = nullptr, *up_exps = nullptr, *down_exps = nullptr;
    const uint8_t *gate_src = nullptr, *up_src = nullptr, *down_src = nullptr;
    uint64_t gate_slice = 0, up_slice = 0, down_slice = 0;
    int      gu_q = 4, down_q = 5;                         // gate/up Q4_K|Q5_K, down Q5_K|Q6_K
};

struct InklingECache {
    uint8_t* base = nullptr;
    uint64_t slot_bytes = 0, off_up = 0, off_down = 0;   // this layer's [gate | up | down] slot layout (its own quants)
    std::vector<int16_t>  slot_of;     // [E] -> slot, -1 = not resident
    std::vector<int16_t>  expert_of;   // [S] -> expert, -1 = empty
    std::vector<uint64_t> stamp;       // [S] last use
};

class InklingModel {
public:
    ~InklingModel();
    // `g` must outlive the model: the embeddings (and the experts when they are not pinned) are read from its mapping.
    // With q2, two cards: q's card runs the embeddings, the dense layers and half the MoE layers; q2's card the other half
    // and the head. Each queue must have its own single-device context (a two-device context mirrors VRAM into RAM).
    std::string load(sycl::queue& q, const GgufReader& g, const InklingConfig& cfg, sycl::queue* q2 = nullptr);
    // Device KV, conv states and workspaces for sequences up to max_ctx positions, forward() calls up to max_chunk
    // tokens; then the expert cache takes the card's free memory minus IE_INKLING_RESERVE_GB (default 1.5).
    std::string init_runtime(uint32_t max_ctx, uint32_t max_chunk);
    // Run T tokens at absolute position pos0 (pos0 == 0 starts a new sequence; otherwise pos0 must equal pos()).
    // logits_host (optional): the LAST row's logits [vocab] as fp32 (padded ids -inf).
    std::string forward(const int32_t* ids, uint32_t T, uint32_t pos0, float* logits_host);
    // A prompt's rows [pos0, pos0 + n) in chunks of max_chunk (pos0 == 0 or pos()); logits_host gets the last row's. On
    // two cards the cards overlap: card 1 runs chunk k+1 while card 2 runs chunk k. after_chunk(end), when given, is
    // called as each chunk is in on both cards; false stops the prefill -- on two cards after the chunk card 1 already
    // holds -- and pos() says where the sequence ends. IE_INKLING_OVERLAP=0: one chunk at a time on both cards.
    std::string prefill(const int32_t* ids, uint32_t n, uint32_t pos0, float* logits_host,
                        const std::function<bool(uint32_t end)>& after_chunk = {});
    // Prefix reuse: the largest position <= want the sequence can continue from, the state set there. The shortconv
    // state exists at the live end and at the end of the last multi-token forward (a prompt's end); the sliding layers'
    // ring must still hold that point's window. 0 = start again.
    uint32_t restart_at(uint32_t want);
    // Called after every layer with the residual stream [T, hidden] (device fp32 on q's card), for oracle comparisons.
    std::function<void(uint32_t layer, sycl::queue& q, const float* x_dev, uint32_t T)> probe;
    std::string self_check();

    const InklingConfig& config() const noexcept { return cfg_; }
    uint32_t cards() const noexcept { return next_ ? 2 : 1; }
    uint64_t device_bytes() const noexcept { return dev_bytes_ + (next_ ? next_->dev_bytes_ : 0); }   // both cards
    uint64_t pinned_bytes() const noexcept { return pinned_bytes_ + (next_ ? next_->pinned_bytes_ : 0); }
    uint32_t cache_slots() const noexcept { return next_ ? std::min(ec_slots_, next_->ec_slots_) : ec_slots_; }   // per layer, the fewer
    uint64_t cache_hits() const noexcept { return ec_hits_ + (next_ ? next_->ec_hits_ : 0); }
    uint64_t cache_misses() const noexcept { return ec_misses_ + (next_ ? next_->ec_misses_ : 0); }
    uint32_t pos() const noexcept { return pos_; }
    uint32_t max_ctx() const noexcept { return max_ctx_; }

private:
    std::string load_stage(sycl::queue& q, const GgufReader& g, const InklingConfig& cfg, uint32_t l0, uint32_t l1);
    std::string stage_forward(const int32_t* ids, uint32_t T, uint32_t pos0, float* logits_host);
    std::string stage_tag() const;
    template <class T> T* dev_alloc(uint64_t n);
    // Host <-> device copies of pageable memory (the GGUF mapping, std::vectors) go through hstage_, a pinned buffer: the
    // driver's own mapping of a pageable copy source was not present yet on a fresh context's first copies (the engine
    // read unmapped pages, DEVICE_LOST at Laguna's ie serve load, 2026-10-16/18). Both wait for the copy.
    void to_device(void* dst, const void* src, uint64_t n);
    void from_device(void* dst, const void* src, uint64_t n);
    std::string upload_f32(const GgufTensorInfo* t, uint64_t n, float*& out);
    std::string upload_q8(const GgufTensorInfo* t, uint32_t K, uint32_t N, InkQ8& out, uint32_t slice = 0, uint32_t n_slices = 1);
    void gemm_q8(const sycl::half* A, const InkQ8& W, float* C, uint32_t T, uint32_t K, uint32_t N);   // prefill: W via wscratch_
    std::string pin_experts();
    const uint8_t* acquire(const InklingLayer& w, uint32_t L, int32_t e);
    void swiglu_ffn(const InkQ8& gate, const InkQ8& up, const InkQ8& down, uint32_t F, uint32_t T, float* dst, bool accumulate,
                    float scale, const float* gamma, uint32_t gamma_stride);
    std::string moe_experts(const InklingLayer& w, uint32_t L, uint32_t T);
    void sconv(float* x, float* y, float* state, const float* w, uint32_t T, uint32_t C);

    sycl::queue*        q_ = nullptr;
    const GgufReader*   g_ = nullptr;
    InklingConfig       cfg_;
    uint32_t            l0_ = 0, l1_ = 0;        // this card's layers [l0_, l1_): l0_ == 0 embeds, l1_ == n_layers runs the head
    std::unique_ptr<InklingModel> next_;         // two cards: the second card's layers
    float*              hx_ = nullptr;           // two cards: pinned host [chunk][hidden], the residual's hand-off
    uint8_t*            hstage_ = nullptr;       // pinned host staging for to_device / from_device
    uint64_t            hstage_n_ = 0;
    std::vector<InklingLayer> layers_;
    std::vector<void*>  allocs_;
    uint64_t            dev_bytes_ = 0, pinned_bytes_ = 0;
    const GgufTensorInfo* tok_embd_ = nullptr;
    float              *embd_norm_ = nullptr, *out_norm_ = nullptr;
    InkQ8               head_;
    sycl::half*         wscratch_ = nullptr;     // prefill: one Q8_0 projection dequantized to F16 [K, N]
    uint64_t            wscratch_n_ = 0;
    uint32_t            kvd_max_ = 0;            // widest per-layer KV projection
    bool                fused_moe_ = true;
    // expert cache
    std::vector<InklingECache> ec_;
    uint64_t ec_clock_ = 0, ec_hits_ = 0, ec_misses_ = 0;
    uint32_t ec_slots_ = 0;
    // runtime
    uint32_t max_ctx_ = 0, max_chunk_ = 0, pos_ = 0;
    std::vector<float*> kc_, vc_;                // per layer: full [max_ctx][n_kv][hd], sliding a ring [ring_][n_kv][hd] fp32
    uint32_t ring_ = 0;                          // sliding layers' ring: window + max_chunk positions
    std::vector<int64_t> ring_pos_;              // [ring_] the position each slot holds (-1 none), shared by every sliding layer
    int32_t* d_ring_pos_ = nullptr;              // its device copy (int32)
    // shortconv rolling states: per layer [k | v | attn | mlp] x 3 rows; and their copy at the last multi-token forward's end
    float*   cstate_ = nullptr;
    float*   csnap_ = nullptr;
    uint64_t cstate_layer_ = 0;                  // floats per layer
    uint32_t snap_pos_ = 0;                      // the position csnap_ belongs to (0 = none)
    float *x_ = nullptr, *xn_ = nullptr, *acc_ = nullptr, *fy_ = nullptr, *ffo_ = nullptr;
    float *qf_ = nullptr, *kf_ = nullptr, *vf_ = nullptr, *rf_ = nullptr, *ka_ = nullptr, *va_ = nullptr;
    float *attn_ = nullptr, *ao_ = nullptr, *rlog_ = nullptr, *rw_ = nullptr, *fg_ = nullptr, *fu_ = nullptr;
    float *sc_ = nullptr, *gs_ = nullptr, *d_rw_ = nullptr, *attn_part_ = nullptr;
    int32_t *ridx_ = nullptr, *d_rows_ = nullptr;
    sycl::half *xn16_ = nullptr, *fh16_ = nullptr, *a16_ = nullptr, *logits16_ = nullptr;
    sycl::half *gx_ = nullptr, *gg_ = nullptr, *gu_ = nullptr, *gh_ = nullptr, *gy_ = nullptr;
    std::vector<float>   h_emb_, h_w_;
    std::vector<int32_t> h_idx_, h_rows_, h_rpos_;
    std::vector<float>   h_rw_;
};

}  // namespace ie
