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
#include "ie/mimo26_host_rules.hpp"   // Mimo26Scan (the NaN probe)

#include <sycl/sycl.hpp>

#include <cstdint>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <string>
#include <utility>
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
    uint64_t reserve_card0 = 0;     // bytes the AUTO tier on the first card leaves free besides the reserve (the vision tower's encode block, P6.2)
    std::vector<std::vector<uint32_t>> ranking;   // [n_layers][E] most-important-first; empty = identity
    // P4 B1 lanes (include/ie/mimo26_lanes.hpp): sequences with their own caches and positions, stepped one at a time
    // (select_lane). Lane 0 is the conversation above (max_ctx positions, the buffers a one-lane forward allocates); lanes
    // 1..lanes-1 get lane_ctx positions each (0 = max_ctx) in the same layout. They are allocated before the auto static tier
    // sizes, so their VRAM comes out of it, and init refuses with the numbers when they do not fit. 1 = the pre-lane forward.
    uint32_t lanes = 1, lane_ctx = 0;
    // P4 B2: each card's CPU expert leg on its own half of the E-cores (card 0 "8-13", card 1 "14-19", V4.1's split) unless
    // IE_DS41_CPU_CORES names the set -- for card-pipelined lanes, where both cards' legs run at once. false = today's (both on
    // 8-19). Measured with 2 pipelined lanes (docs/mimo26/P4_B2_PIPELINE.md): the split was SLOWER (25.3-26.7 vs 28.2 tokens/s).
    bool split_cpu_cores = false;
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
    // Vision (P6.2, docs/mimo26/00_PORT_PLAN.md; V4.1's Phase 56 shape). An image position carries a NEGATIVE id, unique
    // per image and slot: the prefix cache's match reads it off the id. Its stream row comes from the provider, not
    // from the embedding table: `row` [dim] f32 for absolute position `pos`, "" on success. Positions stay 1-D.
    using VisionProvider = std::function<std::string(uint32_t pos, const float*& row)>;
    // P4 B2: the provider belongs to the ACTIVE lane (each lane's request brings its own images); set after init
    void set_vision_provider(VisionProvider p) { if (!lanes_.empty()) lanes_[lane_].vis = std::move(p); }
    void clear_vision() { set_vision_provider(nullptr); }
    void     reset() { n_pos_ = 0; hi_end_ = 0; }   // (the active lane)
    // P4 B1: the lane every later call reads and writes -- forward, reset, rewind, n_pos, written_end, capacity, state_spans,
    // set_state -- until the next select_lane. Each lane keeps its own positions; features() describe the last forward only
    // (cleared by a switch). One lane (the default) never switches: lane 0 is the pre-lane forward, byte for byte.
    std::string select_lane(uint32_t lane);
    uint32_t lane() const { return lane_; }
    uint32_t n_lanes() const { return uint32_t(lanes_.size()); }
    uint64_t lane_bytes(size_t card) const { return card < lane_bytes_.size() ? lane_bytes_[card] : 0; }   // ONE extra lane's caches
    uint32_t n_pos() const { return n_pos_; }
    // P4 B2 -- card-pipelined lanes (docs/mimo26/P4_B2_PIPELINE.md). One host thread per card ("stage"): while card 0 runs a
    // step of lane B, card 1 runs the rest of lane A's. Each lane has at most one step in flight; its step runs exactly the
    // launches forward() would issue for it (on that lane's caches, per-stage host buffers), so with the CPU expert leg off a
    // lane's logits are the serial ones, bit for bit. pipe_start(done) starts the stages; pipe_submit queues a step for an idle
    // lane (T <= max_tokens rows at the lane's n_pos, from any thread, image ids refused); when its last stage finishes, the
    // lane's positions are committed and done(lane, logits, features, feat_rows) runs ON THE LAST CARD'S STAGE THREAD (the
    // drafter shares that card's queue, so drafter work belongs there) -- it may submit the lane's next step. pipe_stop()
    // waits until no step is in flight, stops the stages and returns the first stage error. While the pipe runs, forward(),
    // select_lane() and the state calls must not be used; the diagnostics (probe, profile, IE_MIMO26_DUMP) are refused.
    // P4 B3 (docs/mimo26/P4_B3_ROWS.md) -- row-batched groups: stage 0 takes the lanes waiting for it, in submit order, as ONE
    // forward of up to kDecodeRows rows (a step of more rows -- a prefill chunk -- runs alone): the norms, the FP8 GEMVs, the
    // router, the expert tier and the LM head run over all the rows, attention per lane against its own caches. `group_lanes`
    // caps a group's lanes: 1 = B2 (one lane per group), 0 = AUTO -- the lanes in flight spread over the cards, ceil(lanes /
    // cards), so 4 lanes on 2 cards pipeline as two 2-row groups (1 or 2 lanes: one lane per group, B2's shape). A group's
    // callbacks run one after another with stage 0 held back, so the lanes they resubmit meet in one group again. The FP8
    // GEMV, the router, the norms, the tier's per-row expert jobs and the scatter are row-independent (tests/unit/
    // mimo26_rows_test.cpp); the o-proj and the head go through oneDNN, whose kernel follows M -- IE_MIMO26_ROWS_INVARIANT=1
    // runs those per segment (the serial launches, slower), the default runs them over the group's rows.
    // Rules (B2 gate notes 7-9): a lane stays in flight through its done callback, and only the callback's thread may submit
    // that lane again -- any other thread's pipe_submit of it is refused until the callback returns (so nothing rewrites the
    // lane's buffers while the callback reads `feats`). A callback that keeps resubmitting blocks pipe_stop forever (it waits
    // for no step in flight); pipe_stop called from inside a callback deadlocks (the callback's stage thread is the one it
    // joins). After any stage error, every pipe_submit is refused with that error until pipe_stop (B4 decides per-lane recovery);
    // B4: a submit is also refused once pipe_stop has begun (a racing step would strand its lane) and while the pipe is paused.
    // P4 B5 (docs/mimo26/P4_B5_DRAFT_BUDGET.md): `group_rows` caps a multi-lane group's rows (0 = kDecodeRows, the B3 rule; the
    // serving layer's draft budget); a lone lane's step of more rows still runs alone.
    using PipeDone = std::function<void(uint32_t lane, const std::vector<float>& logits, const float* feats, uint32_t feat_rows)>;
    std::string pipe_start(PipeDone done, uint32_t group_lanes = 0, uint32_t group_rows = 0);
    // P4 B5: from inside a done callback (the last card's stage thread) only -- the lanes of the group whose callbacks are running,
    // in callback order (the lanes a finishing group resubmits meet in one group again, B3's pgate_). Empty on any other thread.
    const std::vector<uint32_t>& pipe_cb_lanes() const;
    std::string pipe_submit(uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0, bool all_rows);
    std::string pipe_reset_lane(uint32_t lane);   // an idle lane forgets its positions (a new sequence) while the pipe runs
    // P4 B4: the lane keeps positions [0, n) (n <= its n_pos; the rows it drops stay written, as rewind()) -- an idle lane, or
    // the lane whose done callback is running, from that thread (a speculative verify's rejected rows, docs/mimo26/P4_B4_SERVE.md)
    std::string pipe_rewind_lane(uint32_t lane, uint32_t n);
    std::string pipe_error() const;               // P4 B4: the first stage error while the pipe runs ("" = none); pipe_stop returns and clears it
    // P4 B4: a PAUSED pipe keeps its stage threads (and their oneDNN contexts: a restart's first steps cost ~300 ms) but takes no
    // steps, and the serial API -- forward, select_lane, reset, rewind, state_spans, set_state -- works again on the active lane,
    // as after pipe_stop. pipe_pause waits until no step is in flight (the caller's callbacks must have stopped resubmitting) and
    // returns a stage error without clearing it (then pipe_stop is the way out); pipe_resume takes steps again. piping() is false
    // while paused; pipe_stop works from either state.
    std::string pipe_pause();
    std::string pipe_resume();
    bool        pipe_paused() const { return !stage_th_.empty() && ppaused_; }
    // (diagnostic, P4 B4) what the group gate (pgate_, B3) costs: the time stage 0 sat with a lane waiting and a group free while
    // a finished group's callbacks held it back, split by that group's lanes (one lane / several). Cumulative since init, ms.
    std::pair<double, double> pipe_gate_ms() const;
    std::string pipe_stop();
    bool        piping() const { return !stage_th_.empty() && !ppaused_; }
    uint32_t    lane_pos(uint32_t lane) const { return lane == lane_ && !piping() ? n_pos_ : lanes_[lane].n_pos; }
    // Per-lane, per-card counters of the decode steps (T <= kDecodeRows) since the last reset: what the tier did with the pinned
    // experts -- how many a stream slot already held (the B1 finding: lanes interleaving on one card share its 8 stream slots
    // per layer). Filled by the serial forward and the pipe alike. B3: a step of SEVERAL lanes' rows (a group) is counted per
    // card in group_tier_stats instead (the tier's counters are per call, not per row); `rows` sums the steps' rows.
    struct LaneTierStats { uint64_t steps = 0, rows = 0, experts_static = 0, experts_pinned = 0, stream_hits = 0, experts_cpu = 0; double moe_ms = 0; };
    const LaneTierStats& lane_tier_stats(uint32_t lane, size_t card) const { return lane_tier_[lane][card]; }
    const LaneTierStats& group_tier_stats(size_t card) const { return group_tier_[card]; }
    void reset_lane_tier_stats() { for (auto& v : lane_tier_) for (auto& s : v) s = {}; for (auto& s : group_tier_) s = {}; }
    // Keep only positions [0, n) (n <= n_pos()): the full layers' caches truncate for free; an SWA ring still holds
    // the window before n only while written_end() - n <= ring() - window() -- the caller checks (P3b prefix reuse).
    void     rewind(uint32_t n) { if (n < n_pos_) n_pos_ = n; }
    // One past the highest position ever written since reset(). A rewind leaves the rows it dropped in the caches
    // (a lookup verify's rejected drafts), and in an SWA ring those rows overwrote the slots of the positions R before
    // them: the ring holds position p only for p >= written_end() - ring() (P3a/P3b/P4 gate finding 3).
    uint32_t written_end() const { return hi_end_; }
    uint32_t ring() const { return ring_; }                           // 0 = linear SWA caches
    uint32_t window() const { return m_ ? m_->config().window : 0; }
    // P7 (#70, docs/mimo26/P7_FIX64_FIX70.md): the conversation state as device byte spans, for the host slots. Card
    // `card`'s share of the state for positions [0, n) written up to `hi` (n <= hi): every full layer's K / V rows [0, n)
    // per kv head, every SWA ring's slots of the positions it still holds, [max(0, hi - ring()), n) -- at most two runs
    // per head -- in a fixed order (layer; K heads; V heads). Nothing else is state: rows past n are rewritten before a
    // forward reads them, and no forward reads a ring slot outside that range. Saved at (n_pos(), written_end()) and
    // written back before set_state(n, hi), the caches read exactly what they read before. `rings_only` (#87, the
    // prompt-end snapshot): the SWA rings' spans alone -- the full layers' rows [0, n) stay on the card.
    void         state_spans(size_t card, uint32_t n, uint32_t hi, std::vector<std::pair<void*, uint64_t>>& out, bool rings_only = false) const;
    std::string  set_state(uint32_t n, uint32_t hi);   // n_pos() = n, written_end() = hi -- the spans for (n, hi) written first
    size_t       n_cards() const { return cards_.size(); }
    sycl::queue* card_queue(size_t card) const { return card < cards_.size() ? cards_[card]->q : nullptr; }
    uint32_t capacity() const { return cap_; }   // the active lane's positions (lane 0: max_ctx)
    uint32_t max_tokens() const { return opt_.max_tokens; }
    const std::vector<Mimo26LayerStats>& stats() const { return stats_; }
    // Routing profile (P4, docs/mimo26/00_PORT_PLAN.md): with profiling on, every routed selection of every counted row is
    // added to profile()[layer][expert]; `rows` (T entries, 1 = count; null = all rows) is read by the NEXT forward() only.
    void set_profile(bool on) { profiling_ = on; if (on && m_) profile_.assign(m_->config().n_layers, std::vector<uint64_t>(m_->config().n_routed_experts, 0)); }
    void set_profile_rows(const uint8_t* rows) { profile_rows_ = rows; }
    const std::vector<std::vector<uint64_t>>& profile() const { return profile_; }
    // (diagnostic, docs/mimo26/P7_FIX64_FIX70.md section 6: the NaN probe) set_probe(L, row): every forward() then checks
    // layer L's intermediates after each op -- the input, the attention norm, qkv, q/k/v, the attention, o_proj, the
    // residual, the ffn norm, the router / the dense FFN's gate, up and SwiGLU, the MoE / down output, the layer's output
    // -- each copied to the host and scanned by bits (mimo26_scan_*), the whole block and row `row` of the call alone, and
    // keeps that row's MoE input and routing in the IE_MIMO26_CHECK_DUMP layout (H f32, top-k i32 ids, top-k f32
    // weights) for tools/mimo26/expert_recompute.py. A wait and a copy per op; L = -1 turns it off.
    struct ProbeOp { std::string op; uint32_t rows = 0, cols = 0; Mimo26Scan all, row; };
    void set_probe(int layer, int row) { probe_layer_ = layer; probe_row_ = row; probe_ops_.clear(); probe_moe_.clear(); probe_moe_out_.clear(); }
    const std::vector<ProbeOp>& probe_ops() const { return probe_ops_; }
    const std::vector<uint8_t>& probe_moe_row() const { return probe_moe_; }
    const std::vector<float>&   probe_moe_out() const { return probe_moe_out_; }   // the watched row of the MoE output [H]
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
        std::vector<float> h_rw; std::vector<int32_t> h_ridx;   // the router's host side (per card: per pipe stage)
        // P4 B13: pinned host staging on this card's context, max_tokens x max(dim, n_routed_experts) floats: the router's
        // logits (read in place), the drafter's feature rows and the residual crossing to the next card land here before
        // any pageable vector. NEO 26.35 serves a small device-to-host memcpy by CPU reads through the BAR (~22 us per KiB):
        // into pageable memory up to 64 KiB, into pinned memory at 0.5-4 KiB; the router's 1-8 KiB therefore goes by a kernel
        float* h_stage = nullptr;
        uint64_t h_stage_bytes = 0;   // (pinned_bytes counts it)
        float qstar_multi0 = 0.f;                 // the tier's own multi-row PCIe share (B3: a group of several lanes' rows uses IE_MIMO26_QSTAR_GROUP)
        std::vector<void*> owned;
        uint64_t bytes = 0;
    };
    template <class T> T* dev(Card& c, size_t n);
    std::string upload_card(Card& c);
    // One call's rows for run_card: a GROUP of segments, each one lane's rows (that lane's caches, capacity and positions) laid
    // one after another in the call's host buffers -- the serial forward's members (one segment: the active lane), or in the
    // pipe the lanes stage 0 grouped (B3). Attention runs per segment; everything else over the call's T rows. One segment at
    // r0 = 0 issues the launches the pre-group forward issued, on the same pointers.
    struct Seg { uint32_t lane = 0, T = 0, pos0 = 0, r0 = 0; bool all_rows = false; float* feat = nullptr; std::vector<float>* logits = nullptr; };
    struct Call {
        uint32_t T = 0;                                     // rows over the segments
        float* x = nullptr; const int32_t* pos = nullptr;   // [T, dim] the residual in / out, [T] positions (host)
        std::vector<Seg> segs;
    };
    std::string run_card(Card& c, size_t ci, const Call& k);
    void        probe_op(sycl::queue& q, const char* op, const void* dev, bool f16, uint32_t rows, uint32_t cols);
    int probe_layer_ = -1, probe_row_ = -1;
    std::vector<ProbeOp> probe_ops_;
    std::vector<uint8_t> probe_moe_;
    std::vector<float>   probe_moe_out_;

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
    std::vector<float>   h_x_;
    std::vector<int32_t> h_pos_;
    // P4 B1 lanes: each lane's capacity, positions and cache pointers ([card][layer - L0] = {k, v}); the active lane's live in
    // Dense::k/v, n_pos_, hi_end_ and cap_ (select_lane swaps them in; lane 0's pointers are the ones upload_card allocated)
    // B2: the pipe's per-lane host buffers (the residual crossing the cards, positions, features, logits) and its in-flight step
    struct LaneState {
        uint32_t cap = 0, n_pos = 0, hi_end = 0; std::vector<std::vector<std::pair<sycl::half*, sycl::half*>>> kv;
        std::vector<float> h_x, h_feat, logits; std::vector<int32_t> h_pos;
        // in_flight: claimed by pipe_submit (check-and-set under pmu_) and held through the step AND its done callback -- only the
        // callback's own thread may submit the lane again meanwhile (in_cb, cb_tid; resub = it did), so no other thread rewrites
        // the lane's buffers while the callback reads its features (B2 gate note 7)
        bool in_flight = false, in_cb = false, resub = false; std::thread::id cb_tid; uint32_t T = 0, pos0 = 0; bool all_rows = false;
        VisionProvider vis;   // the lane's image rows (forward only: the pipe refuses image positions)
    };
    std::vector<std::vector<LaneTierStats>> lane_tier_;   // [lane][card]
    std::vector<LaneTierStats> group_tier_;               // [card]: the steps of several lanes' rows (B3)
    // B2 pipe: one mutex for the queues, the in-flight count and the first error. B3: lanes wait in lq_ for stage 0, which forms
    // a Group from them (its segments; a one-lane group reads the lane's own buffers, a multi-lane group gathers the rows into
    // its h_x / h_pos); stage s >= 1 pops groups from pq_[s]. The last stage runs a group's callbacks with pgate_ set: stage 0
    // forms no group until they are all done, so the lanes they resubmit group together.
    struct Group {
        std::vector<Seg> segs; uint32_t T = 0;
        std::vector<float> h_x; std::vector<int32_t> h_pos;
        float* x = nullptr; int32_t* pos = nullptr;
    };
    std::vector<std::thread> stage_th_;
    std::vector<std::unique_ptr<Group>> groups_;   // the pool (one per lane + one)
    std::deque<Group*> gfree_;
    std::deque<uint32_t> lq_;
    std::vector<std::deque<Group*>> pq_;
    mutable std::mutex pmu_; std::condition_variable pcv_;
    uint32_t pbusy_ = 0; bool pstop_ = false, pgate_ = false, ppaused_ = false; std::string perr_;
    uint32_t pgate_lanes_ = 0; uint64_t gate_ns_one_ = 0, gate_ns_multi_ = 0;   // (diagnostic) pipe_gate_ms
    uint32_t group_lanes_ = 0, group_rows_ = kDecodeRows;
    // P4 B5: the group whose callbacks the last stage runs and that stage's thread (both set and cleared under pmu_), for pipe_cb_lanes
    std::vector<uint32_t> cb_lanes_; std::thread::id cb_lanes_tid_;
    PipeDone pdone_;
    void stage_loop(size_t s);
    std::vector<LaneState> lanes_;
    uint32_t lane_ = 0, cap_ = 0;
    std::vector<uint64_t> lane_bytes_;   // per card: one extra lane's cache bytes (0 with one lane)
    static constexpr uint32_t kHeadRows = 256;   // the LM head runs in row blocks of this size
};

}  // namespace ie
