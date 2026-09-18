// include/ie/deepseek41_forward.hpp — DeepSeek-V4.1-Flash text forward: prefill and decode.
//
// Phase 7: correctness only. Transformer.forward (model.py:1242) unrolled on the engine's
// kernels: embed -> hc expand -> 40 Blocks (engram at 1 and 14, attention by per-layer kind,
// shared caches between source and consumer layers, MoE with the engine's own router) ->
// collapse -> norm -> head. Every layer's dense weights are uploaded, used and freed; the
// routed experts a chunk actually selects are uploaded per layer.
//
// Phase 8: the same loop over CARDS. Each card owns a contiguous layer range with its dense
// weights and expert tiers resident (pipeline-by-layer, single-device contexts -- the 83554c2
// mirror lesson); the layer body is unchanged and runs on whichever card owns the layer. What
// crosses a card boundary is the hyper-connection stream, the previous layer's ffn_pre mix and
// the shared attention caches, bounced through the host (a few MiB at pp2048).
//
// Phase 9 (docs/deepseek41/24): `forward(ids, T, pos0)` with the state the reference keeps in
// its modules carried across calls -- per layer the window ring, the compressed latents and
// index keys of kv-source layers, the ratio-2 compressor's partial group; plus the token
// sequence for the engram hashes. Two regimes: pos0 == 0 with T a multiple of every ratio
// (prefill, which fills the state) and T == 1 at pos0 == n_pos() (decode). `prefill` is the
// former after a reset.
#pragma once

#include "ie/deepseek41.hpp"
#include "ie/deepseek41_engram.hpp"
#include "ie/deepseek41_experts.hpp"
#include "ie/deepseek41_weights.hpp"

#include <sycl/sycl.hpp>

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ie {

struct Ds41LayerStats {
    uint32_t experts_uploaded = 0;    // union of the chunk's routed sets at this layer
    uint64_t expert_bytes     = 0;    // FP4 planes + scales moved for them
    uint64_t dense_bytes      = 0;    // attention / shared-expert / hc weights moved
    double   ms               = 0;    // wall time for the layer
    uint32_t routing_diff     = 0;    // tokens whose routed SET differs from `expected` (if given)
    // resident mode: the tier's accounting for this layer's MoE
    uint32_t experts_static = 0, experts_pinned = 0, experts_mmap = 0;
    uint64_t bytes_mmap = 0;          // mmap -> transient bank this forward
    uint64_t bytes_pinned = 0;        // arena -> stream slot this forward: the cache's misses, actual H2D bytes (Phase 11)
    uint32_t experts_stream_hit = 0;  // of experts_pinned: found in a stream slot, no transfer (the decode hit rate's second term)
    uint32_t experts_cpu = 0;         // pinned misses computed on the CPU (Phase 13, IE_DS41_CPU_MISS)
    double   moe_cpu_ms = 0;          // the CPU leg's wall inside the MoE call (overlaps the groups)
    double   moe_cpu_work_ms = 0;     // the worker's compute time within it
    double   moe_ms = 0;
    double   moe_prep_ms = 0, moe_mmap_ms = 0, moe_mmap_pack_ms = 0, moe_groups_ms = 0;   // Ds41TierStats' breakdown
    // expert parallel: the other cards' tier walls (max), the staging + add time, and the split
    // of this layer's served experts / link misses between the owner and the other cards
    double   moe_remote_ms = 0, ep_xfer_ms = 0; uint32_t experts_owner = 0, experts_remote = 0, miss_owner = 0, miss_remote = 0;
    // Phase 19 step 0 (docs/deepseek41/50): the expert-parallel timeline of this layer, ms from the MoE call's start --
    // the staging D2H done, the helper thread's entry, the remote tier's moe() entry / return, its rows landed on the
    // host, the owner's tier entry, the import hook's entry / exit (the max over remote cards where there are several)
    double   ep_t_stage = 0, ep_t_spawn = 0, ep_t_remote_in = 0, ep_t_remote_out = 0, ep_t_rows = 0, ep_t_owner_in = 0, ep_t_import_in = 0, ep_t_import_out = 0;
    uint32_t experts_mmap_file = 0;          // of experts_mmap: filled by one direct pread from the expert file (Phase 15 C2)
    double   moe_mmap_read_ms = 0, moe_mmap_permute_ms = 0, moe_mmap_group_ms = 0, moe_tail_ms = 0, moe_spawn_ms = 0, moe_join_ms = 0;
    // the dense body by stage, wall between the waits that already bound them: engram (layers 1, 14),
    // attention (hc mixes + collapse + norm + projections + attention + o), ffn_pre (hc mixes + collapse
    // + norm + router), shared (shared expert + hc mix). ms - moe_ms - these = the layer's untimed rest.
    double   engram_ms = 0, attn_ms = 0, ffn_pre_ms = 0, shared_ms = 0, moe_call_ms = 0;   // moe_call: the tier call's wall from here
    // the attention kernel's GPU timestamps, when the queue has profiling: queue = start - submit, exec = end - start (ms)
    double   attn_kernel_queue_ms = -1, attn_kernel_exec_ms = -1;
    float    flip_gap_max     = 0;    // the golden's 6th-vs-7th score margin, max over flipped tokens
                                      // (needs `expected_gap`); a flip at a large margin is a defect
};

// DSpark P2 (docs/deepseek41/48): a decode step may carry up to this many rows (1 + the draft block) with
// IE_DS41_DECODE_MULTI=1; the RoPE slots hold 4T + T/2 positions for the largest step
constexpr uint32_t kDs41MaxDecodeRows = 8;
constexpr uint32_t kDs41RopeSlots = 40;

class Ds41Forward {
public:
    ~Ds41Forward() { if (pc_writer_.joinable()) pc_writer_.join(); for (auto& cp : cards_) { ep_thread_stop(*cp); free_state(*cp); } }   // the decode state outlives no owner
    // `tables` must be loaded (engram). "" on success. STREAMING mode: every layer's weights are
    // uploaded and freed per forward (Phase 7's verified loop), all on one card.
    std::string init(sycl::queue& q, const DeepSeek41Model& m, const Ds41EngramTables& tables);

    // RESIDENT mode (Phase 8): the dense text path stays on the devices across forwards
    // (Ds41DenseCache) and the routed experts live in three tiers (Ds41ExpertTier) by
    // `ranking` -- per model layer, most-important-first; index order when empty. The MoE runs
    // V4's Q8 int-dot grouped prefill path, so its error is that path's floor (~3e-3), not the
    // fp16 token-major path's (~3e-4) that streaming mode uses. Nothing else in the layer
    // changes: the same body serves both modes with the weight source swapped.
    //
    // `qs` is one queue per card, each on its OWN single-device context. Card c owns layers
    // [n_layers*c/n, n_layers*(c+1)/n); the head lives on the last card. Budgets are per card:
    // static slots from that card's free VRAM after its dense set and `vram_reserve`, pinned
    // slots from `host_pin_cap` split in proportion to its layer count.
    struct ResidentOptions {
        uint32_t n_static = 0, n_pinned = 0, stream_slots = 8;   // per layer; 0 static+pinned = derive from budgets
        // Headroom kept free per card AFTER the dense set, for the forward's scratch (~1.1 GiB at
        // 2048 tokens: the two hc streams, masks, scores, engram rows), the persistent mmap expert
        // bank (grow-only, ~1.5 GiB at pp2048 on the held-out profile) and oneDNN's workspaces. At
        // 3 GiB the load ended ~1.9 GiB free and a pp2048 pass ran the driver into eviction: the
        // pass took 22 s with ~10 s of stages floating between passes; at 6 GiB it took 12.5 s and
        // the stages agree (docs/deepseek41/23, runs O-T2). Sized by measurement, not derived yet.
        uint64_t vram_reserve = 6ull << 30;
        // Phase 54 (docs/deepseek41/93): the FP8 LM head (lossy; off for the reference tests, on in ie serve / ie-ds41-run).
        // IE_DS41_HEAD_FP8=1/0 overrides whatever the caller set.
        bool head_fp8 = false;
        // pinned host RAM for the expert arenas, all cards: 0 = the live rule of docs/deepseek4/40
        // (`ds4_host_pin_cap_live`: MemAvailable and MemTotal less their reserves -- ~184 GiB on this
        // box; the 160 GiB constant it replaces is docs/deepseek41/40's lever C1)
        uint64_t host_pin_cap = 0;
        uint32_t max_tokens = 2048;                              // also the position capacity of the decode state
        // Phase 24 (docs/deepseek41/64): the largest T ONE forward will be handed. The expert batch
        // workspace and the expert-parallel staging are scratch for a single call, so they are sized
        // by THIS, not by the context -- at max_tokens 200,000 the batch workspace alone would be
        // 101 GiB (and its uint32 element count overflows), while the KV it is confused with is only
        // ~12.5 KB per position. 0 = follow max_tokens (the old behaviour); IE_DS41_MAX_FWD overrides.
        uint32_t max_forward_tokens = 2048;
        // Expert parallel (docs/deepseek41/38): 1 = every card's tier over all layers with its parity
        // share of each layer's experts, the owner staging x out and adding the partials back;
        // 2 = the control arm (the same machinery, the other cards' tiers empty). 0 = off; the env
        // IE_DS41_EP=1|control sets it when the option is 0. Needs two cards; refused with the CPU split.
        uint32_t expert_parallel = 0;
    };
    std::string init_resident(const std::vector<sycl::queue*>& qs, const DeepSeek41Model& m, const Ds41EngramTables& tables,
                              const std::vector<std::vector<uint32_t>>& ranking, const ResidentOptions& opt);
    void free_resident();
    bool resident() const { return resident_; }
    uint64_t dense_bytes() const;            // summed over cards
    uint64_t expert_vram_bytes() const;
    uint64_t expert_pinned_bytes() const;

    struct CardInfo {
        uint32_t first_layer = 0, n_layers = 0, n_static = 0, n_stream = 0, n_pinned = 0;
        uint64_t dense_bytes = 0, expert_vram_bytes = 0, pinned_bytes = 0;
    };
    uint32_t n_cards() const { return uint32_t(cards_.size()); }
    uint32_t expert_parallel() const { return ep_; }   // 0 off, 1 on, 2 the control arm (docs/deepseek41/38)
    CardInfo card_info(uint32_t c) const;
    Ds41DenseCache& dense_cache(uint32_t c) { return cards_.at(c)->cache; }   // the card's resident dense set (the drafter uploads its stages through it)
    // Where expert e of model layer L lives: 0 static VRAM, 1 pinned host, 2 mmap (2 when not resident).
    uint32_t expert_tier(uint32_t L, uint32_t e) const;

    // Routing profile: per (layer, expert) selection counts accumulated over every own-router
    // forward since init (or the last reset_profile). write_profile stores the per-layer ranking
    // (most-selected first) in V4's `ds4-expert-priority 1` per-layer format, readable by
    // ds4_expert_priority_read_layers.
    const std::vector<std::vector<uint64_t>>& profile() const { return profile_; }
    void reset_profile();
    std::string write_profile(const std::string& path, const std::string& note) const;

    //   probe(name, L, data, n, q) -- called with device pointers at stage boundaries so a test
    //                              can compare: "engram" after engram at L, "layer" after L.
    //                              `q` is the queue of the card that owns the pointer.
    //   expected_routing         -- optional [n_layers][T*top_k] golden indices; when given,
    //                              per-layer routing_diff counts tokens whose set differs.
    using Probe = std::function<void(const char* name, uint32_t L, const float* dev, size_t n, sycl::queue& q)>;
    //   force_routing            -- DIAGNOSTIC: when true, the experts run with `expected_routing`'s
    //                              indices and `expected_weights` instead of the engine's router,
    //                              isolating the expert arithmetic from discrete routing flips.

    // One forward over `T` tokens at positions [pos0, pos0 + T), with the state carried. logits
    // comes back [T, vocab] fp32 on the host. Admitted: pos0 == 0 (a prefill; T a multiple of
    // every compress ratio; resets and fills the state) and T == 1 at pos0 == n_pos() (a decode
    // step). Anything else is refused.
    std::string forward(const int32_t* ids, uint32_t T, uint32_t pos0, std::vector<float>& logits,
                        const Probe& probe = {}, const std::vector<std::vector<int32_t>>* expected_routing = nullptr,
                        const std::vector<std::vector<float>>* expected_weights = nullptr, bool force_routing = false,
                        const std::vector<std::vector<float>>* expected_gap = nullptr);
    // Prefill `T` tokens from position 0: forward(ids, T, 0, ...).
    std::string prefill(const int32_t* ids, uint32_t T, std::vector<float>& logits,
                        const Probe& probe = {}, const std::vector<std::vector<int32_t>>* expected_routing = nullptr,
                        const std::vector<std::vector<float>>* expected_weights = nullptr, bool force_routing = false,
                        const std::vector<std::vector<float>>* expected_gap = nullptr) {
        return forward(ids, T, 0, logits, probe, expected_routing, expected_weights, force_routing, expected_gap);
    }
    // Phase 43 (docs/deepseek41/83): a chunked prefill with the cards as PIPELINE STAGES. Card c runs chunk k on its own
    // host thread while card c-1 runs chunk k+1, so the two PCIe links stream expert bytes at once instead of taking
    // turns (a 2,048-token chunk moves ~180 GiB of expert weights to VRAM, which binds the chunk, not its kernels). Each
    // card executes exactly the launches, in exactly the order, that forward() would issue for it: bit-identical by
    // construction. `chunks` are (pos0, T) in order and contiguous, the first at 0 or at n_pos(); every T is a prefill
    // (pos0 == 0) or a continuation (T > kDs41MaxDecodeRows; IE_DS41_CONT=0 refuses those). Refused -- the caller runs forward() per
    // chunk instead -- with expert parallel, the drafter capture, or a single card. `logits` is the last chunk's.
    std::string forward_pipelined(const int32_t* ids, const std::vector<std::pair<uint32_t, uint32_t>>& chunks, std::vector<float>& logits);
    bool pipelined_admissible() const { return cards_.size() >= 2 && !ep_ && !capture_main_hidden_ && resident_; }

    // Phase 46 (docs/deepseek41/86): the PREFIX CACHE. The state at a position P is the latent / index-key caches'
    // first nc(P) rows (append-only: later positions never rewrite them), each layer's 128-slot window ring and the
    // ratio-2 layers' held half. So a CHECKPOINT is just the rings + halves + the nc counts (~5 MB per card, pinned
    // host), taken at every prefill chunk boundary and at the end of a prompt; restoring one truncates nc and copies
    // the rings back -- the continuation then runs the same arithmetic it would have run from that state. Other
    // conversations are kept in HOST SLOTS (their latents + checkpoints, pageable, under a byte budget) and swapped in
    // when a prompt matches one better than the live state. Resident mode only; the drafter's rings are not covered
    // (the generator keeps the cache off under speculation).
    struct PrefixCacheOptions {
        uint32_t checkpoints = 24;                 // ring checkpoints kept for the live state (pinned, ~5 MB per card each)
        uint64_t host_budget = 4ull << 30;         // bytes of host slots (other conversations), LRU-evicted
        uint64_t keep_free = 24ull << 30;          // never grow the host slots below this much MemAvailable
        uint32_t min_slot_tokens = 1024;           // a state shorter than this is not worth keeping in a host slot
        // Phase 47 (docs/deepseek41/87): DISK entries -- a prompt prefix's state (typically a system prompt + tools) written
        // once and loaded by later processes. Empty dir = off. Keyed by the token ids, the model's shape and this
        // executable's size + mtime (a rebuild may change the arithmetic, so its entries are not trusted).
        std::string disk_dir;
        uint64_t disk_budget = 8ull << 30;         // bytes of entries kept, least recently used removed first
        uint64_t disk_keep_free = 32ull << 30;     // never write when the filesystem would fall under this
        uint32_t min_disk_tokens = 4096;           // a shorter prefix is cheaper to recompute than to keep
    };
    std::string set_prefix_cache(bool on, const PrefixCacheOptions& o);
    std::string set_prefix_cache(bool on) { return set_prefix_cache(on, PrefixCacheOptions{}); }
    bool prefix_cache() const { return pc_on_; }
    // Positions of `ids` served from cached state: the state is left holding exactly ids[0, reused) (n_pos() == reused),
    // reused <= ids.size() - 1 because the caller needs the last token's logits. reused == 0: nothing matched and the
    // state is untouched (the caller's pos0 = 0 prefill resets it). `source` says where it came from ("live",
    // "checkpoint", "host slot", "none").
    std::string prefix_prepare(const std::vector<int32_t>& ids, uint32_t& reused, std::string* source = nullptr);
    // A checkpoint at the current n_pos() on every card (synchronous) -- the generator takes one when a prompt is done.
    std::string prefix_checkpoint();
    struct PrefixCacheStats { uint32_t checkpoints = 0, host_slots = 0; uint64_t host_bytes = 0; double last_restore_ms = 0, last_save_ms = 0; };
    PrefixCacheStats prefix_cache_stats() const;
    // Phase 47: write the state at checkpoint `pos` (which must exist) as a disk entry, unless one for ids[0, pos) exists.
    // The latents are read from the devices now (they are only valid until the state rolls below pos); the file is
    // written on a background thread. "" also when disk entries are off or pos is too short.
    std::string prefix_persist(uint32_t pos);
    uint32_t n_pos() const { return n_pos_; }
    uint32_t capacity() const { return cap_pos_; }   // the position capacity (ResidentOptions::max_tokens)
    // Phase 24: the largest T one forward call may carry. A caller with a longer prompt feeds it in pieces of
    // this size (the generator does) -- it is NOT the context, which is capacity().
    uint32_t forward_capacity() const { return fwd_cap_; }
    void     reset_state();                  // forget the sequence; the next forward must be a prefill
    // DSpark P3 (docs/deepseek41/56): after a T-row step (T >= 2) at pos0, return the state to `n_pos` in
    // [pos0, pos0 + T] -- exactly the state L = n_pos - pos0 one-row steps would have left (bit-identical: the
    // evicted ring slots restored, nc and the ratio-2 half recomputed from the step's own rows, the sequence cut).
    // Valid once per multi-row step. "" on success.
    std::string rollback_to(uint32_t n_pos);
    // P3 criterion 3 (negative control): a rollback that skips the ring restore -- must break criterion 1 past the wrap
    void set_rollback_noring_diagnostic(bool on) { noring_diag_ = on; }
    // Phase 9 criterion 1: host copies of layer L's state -- the window ring [window, head_dim]
    // (every slot, written or not), the latents [nc, head_dim] and index keys [nc, index_head_dim]
    // (empty for layers that are not kv sources).
    std::string read_state(uint32_t L, std::vector<float>& win, std::vector<float>& comp, std::vector<float>& idxk, uint32_t& nc) const;
    // Phase 9 criterion 6: the engram hashes the last forward used, [engram layer][T * n_hash_cols].
    const std::vector<std::vector<int64_t>>& last_hashes() const { return last_hashes_; }
    // Phase 9 criterion 5 (negative control): add `off` to the RoPE position of q/kv at DECODE
    // steps only. 0 restores the real bookkeeping.
    void set_rope_offset_diagnostic(int32_t off) { rope_off_ = off; }
    // DSpark P2 criterion 2 (negative control): a T-row step's ring mask WITHOUT the per-row causality (every
    // written slot open to every row) -- must make the T-row step differ from T one-row steps. false restores it.
    void set_multi_nocausal_diagnostic(bool on) { nocausal_diag_ = on; }
    // When set, a prefill computes and returns the logits of the LAST position only ([1, vocab]):
    // at pp2048 the full [T, vocab] fp32 block is 1 GiB of VRAM and a 1 GiB copy to the host, and
    // generation needs one row. Off by default so the goldens' full-logits comparisons still work.
    void set_logits_last_only(bool on) { logits_last_only_ = on; }
    // Phase 58: admit 2..kDs41MaxDecodeRows-row decode steps (a speculative verify) without IE_DS41_DECODE_MULTI --
    // the prompt-lookup loop turns it on for itself; the multi-row path is gated (docs/deepseek41/55-56)
    void set_multi_row_decode(bool on) { multi_rows_ = on; }
    // Decoder SWA Bounded Replay (docs/deepseek41/27; the report's §2.2/§3.2.2), ON by default as
    // deployed: at a prefill longer than the window, the decoder half (layers n_layers/2..) runs
    // over the last `window_size` prompt tokens only, after its first layer's compressor has
    // produced the global latents and index keys for EVERY prompt token from the encoder output.
    // Prompts that fit the window replay everything and are bit-identical to the exact path.
    // Off = the exact path, all layers over all tokens (what the goldens are). With the mode active
    // the head runs over the 128-row segment, so forward() requires set_logits_last_only(true).
    void set_bounded_replay(bool on) { bounded_replay_ = on; }
    bool bounded_replay() const { return bounded_replay_; }

    const std::vector<Ds41LayerStats>& stats() const { return stats_; }
    // DSpark P1 (docs/deepseek41/57): the last forward's `main_hidden` -- per row, the mean over the hc copies of the
    // stream ENTERING each dspark target layer (37, 38, 39), concatenated: [T, n_targets * dim] on the host. Empty
    // when the model has no drafter. The drafter reads it (its seed and its draft).
    const std::vector<float>& main_hidden() const { return main_hidden_; }
    // the capture costs a synchronous D2H + a host mean at the three target layers on every forward (gate P1 finding 7):
    // on only when a drafter consumes it (the generator sets it with a drafter attached; the drafter tests set it)
    void set_capture_main_hidden(bool on) { capture_main_hidden_ = on; }
    const std::vector<int32_t>& all_ids() const { return all_ids_; }        // the sequence the caches hold (gate P3 finding 3)
    // Vision (Phase 56, docs/deepseek41/95). An image position carries a NEGATIVE id, unique per image and slot: the
    // prefix cache's match, the engram's dead-token rule, the image router bias and the splice all read it off the id.
    // Its stream row comes from a span given here, not from the embedding table: rows [n, hidden] f32 for the absolute
    // positions [pos0, pos0 + n). Spans live until clear_vision(); a position already in the caches needs none.
    void set_vision_span(uint32_t pos0, std::vector<float> rows);
    // An image position no span covers asks the provider: `row` [hidden] for absolute position `pos`, "" on success.
    // The engine encodes an image on its first row, so an image the prefix cache already holds is never encoded.
    using VisionProvider = std::function<std::string(uint32_t pos, const float*& row)>;
    void set_vision_provider(VisionProvider p) { vis_provider_ = std::move(p); }
    void clear_vision() { vis_spans_.clear(); vis_provider_ = nullptr; }
    // the last forward's time outside the layer loop: host prep (engram hashes + table gathers +
    // embed) and the tail (collapse, norm, head, logits to the host), ms
    double prep_ms() const { return prep_ms_; }
    double head_ms() const { return head_ms_; }
    uint64_t head_bytes() const { return head_bytes_; }       // lm_head + final norm, moved after the loop
    // The candidate top-k is a no-op below candidate_topk_blocks * candidate_block_size compressed
    // positions and the runtime REFUSES above it (the error return in prefill), so this is
    // information, not a checkable outcome -- the gate pointed out a check on it could not fail.
    bool candidate_topk_was_noop() const { return cand_noop_; }

private:
    // Per layer, on the owning card: what the reference keeps in its Attention module
    // (docs/deepseek41/24). Allocated on the first forward, position capacity = max_tokens.
    struct LayerState {
        float* win_kv = nullptr;                  // [window, HD]  ring, RoPE'd rows, slot = pos % window
        float* comp_kv = nullptr;                 // [cap, HD]     kv-source layers: latents (RoPE'd)
        float* idx_k = nullptr;                   // [cap, IHD]    kv-source layers: index keys (RoPE'd)
        float* part_kv = nullptr; float* part_gate = nullptr;   // [ratio, HD]  ratio-2 sources: the open group
        uint32_t nc = 0; bool part_valid = false;
        // DSpark P3 (docs/deepseek41/56): what a T-row step overwrites, kept for rollback_to -- the T evicted ring
        // slots [kDs41MaxDecodeRows, HD]; for a ratio-2 kv source its per-row compressor outputs [rows, HD] x 2 and
        // the pre-step half; the pre-step nc and part_valid
        float* snap_ring = nullptr; float* snap_ckv = nullptr; float* snap_cg = nullptr; float* snap_part_kv = nullptr; float* snap_part_gate = nullptr;
        uint32_t snap_nc = 0; bool snap_part_valid = false;
    };
    // One card: its queue, its layer range, and (resident mode) its dense cache and expert tiers.
    // Ds41ExpertTier's arena/cache are non-copyable, hence the unique_ptr.
    struct Card {
        sycl::queue*   q = nullptr;
        uint32_t       L0 = 0, L1 = 0;
        bool           snaps_ready = false;       // P3's rollback snapshots exist (allocated at the first T >= 2 decode step)
        Ds41DenseCache cache;
        Ds41ExpertTier tier;
        Ds41ExpertTier ctl_tier;                  // the control arm's empty tier over every layer (expert_parallel == 2)
        // expert parallel: x staged in from an owner (device + its pinned host half); this tier's
        // packed expert rows gathered for export / another tier's rows landing here (fp16, device +
        // pinned host), and their packed row indices (device)
        float* ep_x = nullptr; float* ep_hx = nullptr;
        sycl::half* ep_ypc = nullptr; sycl::half* ep_hyc = nullptr; int32_t* ep_rows = nullptr;
        double ep_ms = 0;                         // the last remote moe wall on this card
        std::vector<LayerState> state;            // [n_layers], only [L0, L1) allocated
        // the caches a consumer layer at the top of this card reads when the source sits on an
        // earlier card: the previous card's source state, imported through the host
        float* imp_ckv = nullptr; float* imp_ik = nullptr;
        bool   state_ready = false;
        // Phase 18 (docs/deepseek41/46 term 2): the decode step's persistent pieces -- the T = 1
        // scratch, taken in the same order every step and freed with the state; the two RoPE
        // inv_freq tables resident; the step's positions and their cos / sin (one copy, no waits)
        std::vector<std::vector<std::pair<void*, size_t>>> dec_scratch; std::vector<size_t> dec_cursor;   // per step length T (DSpark P2)
        float* rope_inv = nullptr;                                  // [2, RD/2]: main, compress
        int32_t* rope_pos = nullptr; int32_t* rope_hpos = nullptr;  // [kDs41RopeSlots] device / pinned host
        float* rope_cs = nullptr; float* rope_sn = nullptr;         // [kDs41RopeSlots, RD/2]
        // Phase 19 (docs/deepseek41/50): a persistent helper thread that runs this card's remote share of
        // the owner's layers (docs/51: bit-identical; -0.2 to -4 ms/token depending on the cache state -- the join
        // was NOT the term; the per-layer E[max] is)
        std::thread ep_thread; std::mutex ep_mu; std::condition_variable ep_cv;
        std::function<void()> ep_job; bool ep_pending = false, ep_done = true, ep_stop = false;
    };
    // Phase 43: what a card hands the next at its boundary (forward()'s host carry), and the card range one pipeline
    // stage runs. A stage that does not start at card 0 skips the host prep (the engram layers must sit before it) and
    // takes the stream from `in`; a stage that does not end at the last card exports it into `out` and commits no n_pos.
    struct StageCarry {
        std::vector<float> hh, hpm, imp_ckv, imp_ik; std::vector<int32_t> sh_topk;
        uint32_t T_carry = 0, off_carry = 0, bounce_nc = 0; bool bounce_topk = false;
    };
    struct Stage { size_t card_first = 0, card_last = 0; StageCarry* in = nullptr; StageCarry* out = nullptr; };
    std::string forward_impl(const int32_t* ids, uint32_t T, uint32_t pos0, std::vector<float>& logits,
                             const Probe& probe, const std::vector<std::vector<int32_t>>* expected_routing,
                             const std::vector<std::vector<float>>* expected_weights, bool force_routing,
                             const std::vector<std::vector<float>>* expected_gap, const Stage* stage);
    void        ep_thread_stop(Card& card);
    std::string ensure_state(Card& card);
    // P3's rollback snapshots, allocated on the first multi-row step only (gate P4: they are ~0.7 MiB of permanent
    // per-card state otherwise, which the forward test's "state returns to start" bar caught)
    std::string ensure_snapshots(Card& card);
    void        free_state(Card& card);

    std::vector<std::unique_ptr<Card>> cards_;
    uint32_t                  ep_ = 0;             // ResidentOptions::expert_parallel as resolved at init
    const DeepSeek41Model*    m_ = nullptr;
    const Ds41EngramTables*   tb_ = nullptr;
    std::vector<Ds41LayerStats> stats_;
    uint64_t                  head_bytes_ = 0;
    bool                      cand_noop_ = true;
    bool                      resident_ = false;
    uint32_t                  cap_pos_ = 2048;    // position capacity of the state (max_tokens)
    uint32_t                  fwd_cap_ = 2048;    // Phase 24: the largest T one forward may carry (max_forward_tokens)
    uint32_t                  n_pos_ = 0;
    double                    prep_ms_ = 0, head_ms_ = 0;
    int32_t                   rope_off_ = 0;
    bool                      nocausal_diag_ = false;
    bool                      noring_diag_ = false;
    uint32_t                  snap_pos0_ = 0, snap_T_ = 0; bool snap_valid_ = false;   // P3: the last multi-row step's snapshot
    bool                      logits_last_only_ = false;
    bool                      multi_rows_ = false;
    bool                      bounded_replay_ = true;
    std::vector<int32_t>      all_ids_;           // the sequence so far (engram look-back)
    struct VisSpan { uint32_t pos0 = 0, n = 0; std::vector<float> rows; };
    std::vector<VisSpan>      vis_spans_;         // image rows for positions whose id is negative
    VisionProvider            vis_provider_;
    bool capture_main_hidden_ = false;
    std::vector<std::vector<int64_t>> last_hashes_;
    std::vector<std::vector<uint64_t>> profile_;   // [layer][expert] selection counts
    std::vector<float>        main_hidden_;       // [T, n_targets * dim] (DSpark P1)

    // ---- Phase 46: the prefix cache (deepseek41_prefix_cache.cpp) ----
    // One checkpoint: the position, every layer's nc / part_valid (indexed by model layer), and which cards finished
    // writing their pinned block (index `block` in each card's pool; card c's block holds its layers' rings, then the
    // ratio-2 halves, at pc_off_). A pipelined prefill captures card 0 while card 1 is a chunk behind, hence the mask.
    struct PcCkpt { bool used = false; uint32_t pos = 0; uint32_t cards_done = 0; uint64_t tick = 0;
                    std::vector<uint32_t> nc; std::vector<uint8_t> part_valid; };
    // A host slot: a whole conversation's state, pageable. `ring[k][card]` is checkpoint k's block for that card;
    // `lat[card][layer]` the latents then the index keys for nc at the slot's LAST checkpoint (its live position).
    struct PcSlot { std::vector<int32_t> ids; std::vector<PcCkpt> ckpts; std::vector<std::vector<std::vector<float>>> ring;
                    std::vector<std::vector<std::vector<float>>> lat; uint64_t bytes = 0, tick = 0; };
    void        pc_capture(size_t ci, uint32_t pos);            // enqueue card ci's rings into the checkpoint at pos
    std::string pc_restore_live(const PcCkpt& c, size_t index); // rings back from pool block `index`, nc truncated
    std::string pc_save_live_to_slot(bool& saved);              // the live conversation -> a host slot
    std::string pc_load_slot(PcSlot& slot, size_t ckpt);        // a host slot -> the live state, at that checkpoint
    void        pc_clear_live();                                // the live state's checkpoints dropped (a reset)
    bool                      pc_on_ = false;
    PrefixCacheOptions        pc_opt_;
    std::mutex                pc_mu_;
    std::vector<PcCkpt>       pc_ckpts_;                   // pool-indexed, pc_opt_.checkpoints entries
    std::vector<std::vector<float*>> pc_pool_;             // [card][block] pinned host, pc_block_[card] floats each
    std::vector<uint64_t>     pc_block_;                   // floats per block, per card
    std::vector<std::vector<uint64_t>> pc_off_;            // [card][model layer] offset of the layer's ring in the block
    std::vector<float*>       pc_bounce_; uint64_t pc_bounce_n_ = 0;   // [card] pinned bounce for slot latents
    std::vector<PcSlot>       pc_slots_;
    uint64_t                  pc_tick_ = 0, pc_slot_bytes_ = 0;
    double                    pc_restore_ms_ = 0, pc_save_ms_ = 0;
    // Phase 47: the disk entries' index (ids read from each file's header) and the writer
    struct PcDisk { std::string path; std::vector<int32_t> ids; uint64_t bytes = 0; };
    std::vector<PcDisk>       pc_disk_;
    std::string               pc_disk_key_;         // model shape + executable identity
    std::thread               pc_writer_;
    void        pc_disk_scan();
    std::string pc_load_disk(const PcDisk& d);      // a disk entry -> the live state at its position
    void        pc_join_writer() { if (pc_writer_.joinable()) pc_writer_.join(); }
};

}  // namespace ie
