// include/ie/mimo26_forward.hpp — MiMo-V2.6 forward on 1-2 cards (P2 bring-up, docs/mimo26/00_PORT_PLAN.md).
//
// Layers are split evenly across the queues in order (layer 0's dense FFN and the embeddings on the
// first card, the head on the last); the residual stream crosses cards through the host. Dense weights
// live on their card as fp16 (the FP8 x scale product and the BF16 tensors rounded once at load); the
// routed experts run on the V4.1 expert tier (VRAM static / stream slots / pinned host / mmap) through
// Ds41ExpertSource. Attention is mimo26_attention (K 192 / V 128, per-head sinks on the SWA layers,
// window 128); the router is sigmoid + bias top-k on the host from fp32 device logits.
//
// Everything is the plain in-order path: one call = T rows at positions [pos0, pos0 + T) appended to the
// KV caches. Chunked prefill is the caller's loop; `reset()` forgets the caches.
// P3a (docs/mimo26/00_PORT_PLAN.md): the 39 SWA layers keep a ring of window + max_tokens K/V slots; the 9 full
// layers keep linear caches to max_ctx with V padded to head_dim (192) rows, so a T > 8 chunk runs the XMX FA-2
// prefill kernel (full_attention_fa2_prefill_xmx: one head dim for K and V, no sinks -- the full layers have none)
// and its 192-wide output is compacted to 128 before o_proj; T <= 8 steps stay on mimo26_attention.
#pragma once

#include "ie/deepseek41_experts.hpp"
#include "ie/mimo26.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ie {

struct Mimo26Options {
    uint32_t max_ctx = 4096;        // KV capacity per layer, in positions
    uint32_t max_tokens = 2048;     // rows per forward() call (the workspaces' size)
    // per layer and card: VRAM-static experts, pinned-host experts, evictable VRAM stream slots; the rest is
    // the mmap tier (256 - n_static - n_pinned; 0 keeps every expert in VRAM or pinned RAM). n_static = 0: AUTO --
    // each card's static tier fills the VRAM it has free after its dense set, caches and workspaces, less the tier's
    // batch workspace and IE_MIMO26_VRAM_RESERVE_GIB (default 1.5); its pinned count is then min(n_pinned, 256 - static)
    uint32_t n_static = 48, n_pinned = 208, stream_slots = 8;
    std::vector<std::vector<uint32_t>> ranking;   // [n_layers][E] most-important-first; empty = identity
};

struct Mimo26LayerStats {
    double   ms = 0, moe_ms = 0;
    uint32_t experts_static = 0, experts_pinned = 0, experts_mmap = 0;
};

class Mimo26Forward {
public:
    Mimo26Forward() = default;
    ~Mimo26Forward();
    Mimo26Forward(const Mimo26Forward&) = delete;
    Mimo26Forward& operator=(const Mimo26Forward&) = delete;

    // Uploads the dense weights, builds the expert tiers and the KV caches. "" on success.
    std::string init(const std::vector<sycl::queue*>& qs, const Mimo26Model& m, const Mimo26Options& o);
    // T <= max_tokens rows at positions [pos0, pos0 + T); pos0 must equal n_pos() (the caches hold [0, pos0)).
    // logits: all_rows ? [T, vocab] : [1, vocab] (the last row), fp32 on the host.
    std::string forward(const int32_t* ids, uint32_t T, uint32_t pos0, std::vector<float>& logits, bool all_rows);
    void     reset() { n_pos_ = 0; hi_end_ = 0; }
    uint32_t n_pos() const { return n_pos_; }
    // Keep only positions [0, n) (n <= n_pos()): the full layers' caches truncate for free; an SWA ring still holds
    // the window before n only while written_end() - n <= ring() - window() -- the caller checks (P3b prefix reuse).
    void     rewind(uint32_t n) { if (n < n_pos_) n_pos_ = n; }
    // One past the highest position ever written since reset(). A rewind leaves the rows it dropped in the caches
    // (a lookup verify's rejected drafts), and in an SWA ring those rows overwrote the slots of the positions R before
    // them: the ring holds position p only for p >= written_end() - ring() (P3a/P3b/P4 gate finding 3).
    uint32_t written_end() const { return hi_end_; }
    uint32_t ring() const { return ring_; }                           // 0 = linear SWA caches
    uint32_t window() const { return m_ ? m_->config().window : 0; }
    uint32_t capacity() const { return opt_.max_ctx; }
    uint32_t max_tokens() const { return opt_.max_tokens; }
    const std::vector<Mimo26LayerStats>& stats() const { return stats_; }
    // Routing profile (P4, docs/mimo26/00_PORT_PLAN.md): with profiling on, every routed selection of every counted row is
    // added to profile()[layer][expert]; `rows` (T entries, 1 = count; null = all rows) is read by the NEXT forward() only.
    void set_profile(bool on) { profiling_ = on; if (on && m_) profile_.assign(m_->config().n_layers, std::vector<uint64_t>(m_->config().n_routed_experts, 0)); }
    void set_profile_rows(const uint8_t* rows) { profile_rows_ = rows; }
    const std::vector<std::vector<uint64_t>>& profile() const { return profile_; }
    static constexpr uint32_t kDecodeRows = 8;   // a forward of at most this many rows runs the split-K decode attention
    // P5 (DFlash): the residual stream after `layers` (their order = the features' order), the LAST min(T, max_rows) rows
    // of every forward() call, copied to host memory (a plain vector: each card has its own SYCL context, so no one card's
    // pinned allocation serves both), compact layer-major [layers.size()][feat_rows()][dim]. Empty = off.
    std::string set_feature_layers(std::vector<uint32_t> layers, uint32_t max_rows);
    const float* features() const { return h_feat_.data(); }
    uint32_t     feat_rows() const { return feat_rows_; }
    // The last card's fp16 lm_head [vocab, dim] and its queue (the drafter shares them).
    const sycl::half* head_weights() const { return cards_.empty() ? nullptr : cards_.back()->head; }
    sycl::queue*      last_queue() const { return cards_.empty() ? nullptr : cards_.back()->q; }
    uint64_t vram_bytes(size_t card) const;
    uint32_t card_static(size_t card) const { return card < cards_.size() && cards_[card]->tier_on ? cards_[card]->tier.n_static() : 0; }
    uint32_t card_pinned(size_t card) const { return card < cards_.size() && cards_[card]->tier_on ? cards_[card]->tier.n_pinned() : 0; }
    uint64_t pinned_bytes() const;
    void     free_all();

private:
    struct Dense {
        float *attn_norm = nullptr, *ffn_norm = nullptr, *sink = nullptr, *router_w = nullptr;   // fp32 device
        std::vector<float> router_b;                                                              // host [E]
        sycl::half *qkv = nullptr, *o = nullptr, *gate = nullptr, *up = nullptr, *down = nullptr; // fp16 [N, K] device
        // FP8-resident (P4 lever 3, the default; IE_MIMO26_FP8_DENSE=0 keeps the fp16 copies above): E4M3 [N, K] bytes and
        // per-row F32 scales [N, K / 128] (mimo26_fp8_rows) for the qkv and the dense MLP
        uint8_t *qkv8 = nullptr, *gate8 = nullptr, *up8 = nullptr, *down8 = nullptr;
        float   *qkv_s = nullptr, *gate_s = nullptr, *up_s = nullptr, *down_s = nullptr;
        sycl::half *k = nullptr, *v = nullptr;                                                    // KV caches
    };
    struct Card {
        sycl::queue* q = nullptr;
        uint32_t L0 = 0, L1 = 0;                  // model layers [L0, L1)
        bool tier_on = false; uint32_t tier_L0 = 0;
        Ds41ExpertTier tier;
        std::vector<Dense> dense;                 // [L1 - L0]
        // workspaces, max_tokens rows
        float *x = nullptr, *xn32 = nullptr, *qkv32 = nullptr, *o32 = nullptr, *gate32 = nullptr, *up32 = nullptr, *moe = nullptr, *rlogits = nullptr;
        sycl::half *xn16 = nullptr, *Q = nullptr, *K = nullptr, *V = nullptr, *attn = nullptr, *h16 = nullptr;
        sycl::half *Vp = nullptr, *attn_p = nullptr;   // full layers: V padded to head_dim rows, the XMX FA-2 output [T, n_q, head_dim]
        float* partials = nullptr;                // mimo26_attention_decode's split partials (forwards of <= kDecodeRows rows)
        sycl::half* wscratch = nullptr;           // a chunk's fp16 copy of one FP8-resident weight (mimo26_fp8_to_f16)
        int32_t* pos = nullptr;
        float* fnorm = nullptr; sycl::half* head = nullptr; float* head_out = nullptr;   // last card
        std::vector<void*> owned;
        uint64_t bytes = 0;
    };
    template <class T> T* dev(Card& c, size_t n);
    std::string upload_card(Card& c);
    std::string run_card(Card& c, uint32_t T, uint32_t pos0, std::vector<float>& logits, bool all_rows);

    std::vector<std::unique_ptr<Card>> cards_;
    const Mimo26Model* m_ = nullptr;
    Mimo26Options opt_;
    uint32_t n_pos_ = 0;
    uint32_t hi_end_ = 0;   // written_end()
    std::vector<uint32_t> feat_layers_; uint32_t feat_max_ = 0, feat_rows_ = 0;
    std::vector<float> h_feat_;   // [feat_layers_.size()][feat_rows_][dim] (compact per call)
    uint32_t ring_ = 0;   // the SWA layers' K/V ring slots (window + max_tokens; 0 = linear, IE_MIMO26_SWA_LINEAR=1)
    uint32_t n_calls_ = 0;
    bool profiling_ = false; const uint8_t* profile_rows_ = nullptr;
    std::vector<std::vector<uint64_t>> profile_;   // forward() calls since init (the IE_MIMO26_DUMP file index)
    std::vector<Mimo26LayerStats> stats_;
    std::vector<float>   h_x_, h_rlogits_;
    std::vector<int32_t> h_ridx_;
    std::vector<float>   h_rw_;
    std::vector<int32_t> h_pos_;
    static constexpr uint32_t kHeadRows = 256;   // the LM head runs in row blocks of this size
};

}  // namespace ie
