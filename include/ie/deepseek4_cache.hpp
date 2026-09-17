// include/ie/deepseek4_cache.hpp — DeepSeek-V4 per-layer attention cache (Phase 3).
//
// WHY A NEW CACHE TYPE (and not include/ie/kv_cache.hpp)
// -----------------------------------------------------
// `ie::KvCache` is a flat `[L, n_kv_heads, max_ctx, head_dim]` fp16 slab with a
// per-layer scalar length.  It CANNOT express DeepSeek-V4's cache, for five
// independent reasons — every one of them is a hard blocker, not a preference:
//
//   1. V4's sliding branch is a RING that only retains `sliding_window - 1`
//      positions (reference `DeepseekV4HCACache.update`,
//      modeling_deepseek_v4.py:214 `self.keys = full[:, :, -self.sliding_window + 1 :, :]`).
//      `KvCache` never evicts; `length()` is monotone and indexes an absolute
//      position.  A ring cannot be described by one scalar length.
//   2. `update()` RETURNS the full pre-truncation concatenation (retained ++ new)
//      — the attention attends over more rows than the cache keeps.  `KvCache`
//      has no such "returns more than it stores" contract.
//   3. K and V are literally the SAME storage (`self.values = self.keys`,
//      modeling:211; `past_key_values.update(kv, kv, ...)`, modeling:825).
//      `KvCache` allocates two disjoint slabs and copies into both.
//   4. Each CSA/HCA layer additionally owns a compressor state machine —
//      a partially-filled window buffer (kv AND gate), a monotone list of
//      emitted compressed entries, an emitted-entry counter, and (CSA only)
//      the previous window's Ca overlap slice.  None of that fits a
//      `[kv_heads, ctx, head_dim]` slab.
//   5. The compressed list GROWS at `1 / compress_rate` of the token rate and
//      is unbounded in `max_ctx` terms, whereas `KvCache` is preallocated at a
//      fixed `max_ctx`.  Here the growing arrays are geometrically reallocated.
//
// So this type diverges from the engine's existing cache contract in exactly
// these ways, and shares nothing with it.  `KvCache` is untouched.
//
// SEMANTICS — a 1:1 transcription of the reference cache classes
// --------------------------------------------------------------
//   DeepseekV4HCACache (modeling:171) extends transformers' DynamicSlidingWindowLayer:
//       update()                     -> Ds4LayerCache::update_sliding
//       store_compression_weights()  -> Ds4LayerCache::store_compression_weights
//       update_compressor_states()   -> Ds4LayerCache::update_compressor_states
//   DeepseekV4CSACache (modeling:255) adds the "indexer" entry to the same dicts
//   plus the per-name overlap slices:
//       update_overlap_state()       -> Ds4LayerCache::update_overlap_state
//
// The reference's `dict[str, ...]` keyed by `"compressor"` / `"indexer"` is a
// two-element array indexed by `Ds4CacheEntry` here.  An HCA layer only ever
// touches `Compressor`; a CSA layer touches both; a sliding layer touches
// neither.
//
// STORAGE / PRECISION
// -------------------
// fp32 device USM, matching every other Phase 2/3 DeepSeek-V4 op
// (see include/ie/deepseek4_ops.hpp — the reference keeps these tensors in
// fp32 too).  Production sizing in docs/deepseek4/11_compressors_and_cache.md
// is quoted at 2 bytes/element (bf16 reference / fp16 engine), so
// `state_bytes(elem_bytes)` takes the element size as a parameter and the
// 6 880 B/token figure is `state_bytes(2)` growth.  The fp32 allocation this
// class actually makes is exactly twice that; nothing is hidden.
//
// Allocation uses `sycl::malloc_device` / `sycl::free` directly rather than
// `ie::DeviceAllocator`, because the compressed arrays need geometric growth
// (alloc → copy → free) and `DeviceAllocator` is a thin malloc/free wrapper
// with no realloc.  Nothing is gained by routing through it.
//
// PERSISTENT STATE vs TRANSIENT SCRATCH  (added for multi-sequence serving)
// ------------------------------------------------------------------------
// Serving N users concurrently needs N copies of this object's PER-SEQUENCE
// state, and the obvious "just make N of them" is unaffordable — because most
// of what this class allocates is not per-sequence state at all, it is working
// memory for ONE forward call:
//
//   PERSISTENT (per sequence, must be duplicated)   TRANSIENT (one copy serves all)
//     sliding_        (sliding_window - 1) * HD       sliding_full_  (sw-1 + T) * HD
//     buf_kv_/gate_   (rate - 1) * source_width       chunk_kv_      (rate-1 + T) * sw
//     overlap_kv_/gate_  2 * rate * width             chunk_gate_    (rate-1 + T) * sw
//     compressed_     entry_count * width  <-- grows
//
// The transient three are sized by the PREFILL CHUNK `T`, not by the sliding
// window or the context, so at a 2048-token chunk they are two orders of
// magnitude larger than the persistent state on a CSA layer.  Duplicating them
// per sequence would make batch size 2 cost more VRAM than the whole rest of
// the cache.  They do not need duplicating: every pointer this class hands out
// of them (`update_sliding`'s return, `Ds4CompressChunk::kv/gate`) is consumed
// inside the SAME `layer_forward_pre` call that produced it, and the runtime
// drains its queue between layers and between forwards.  So they live in a
// separate `Ds4CacheScratch`, one per layer, shared by every sequence slot.
//
// A cache constructed without an explicit scratch keeps a private one and
// behaves exactly as it did before this split — same allocations, same copies,
// same numbers.  That is what makes the one-sequence path unchanged.

#pragma once

#include <sycl/sycl.hpp>

#include <utility>

#include <cstdint>
#include <string>

namespace ie {

// The reference's `dict` key.  HCA layers use Compressor only.
enum class Ds4CacheEntry : uint32_t { Compressor = 0, Indexer = 1 };
inline constexpr uint32_t kDs4CacheEntries = 2;

struct Ds4CacheConfig {
    // compress_ratios[L] verbatim: 0 = sliding_attention (no compressor),
    // 4 = compressed_sparse_attention (compressor + indexer, Ca/Cb overlap),
    // 128 = heavily_compressed_attention (compressor only, no overlap).
    uint32_t compress_ratio = 0;
    uint32_t head_dim       = 512;  // compressor entry width
    uint32_t index_head_dim = 128;  // indexer entry width (ratio 4 only)
    uint32_t sliding_window = 128;  // retention is sliding_window - 1
    // Element width of the two compressed entry arrays — the only context-
    // growing state (docs/deepseek4/73 Phase 2 gate).  fp16 halves their VRAM.
    // VALUES (the K=V rows) fp16 hold exactly the numbers the XMX prefill kernel
    // already converted to on every call; KEYS (the lightning indexer's) fp16
    // flip near-tie top-k picks against a fp32-key run, so the default keeps
    // them fp32 and prefill stays bit-identical to the all-fp32 cache.  The
    // runtime sets these from $IE_DS4_KV16 (values | 1 | 0).
    bool f16_values = true;
    bool f16_keys   = false;
};

// One chunk of window-aligned source tokens peeled off by
// store_compression_weights().  Pointers are device USM owned by the cache and
// stay valid until the next store_compression_weights() call for the same entry.
struct Ds4CompressChunk {
    const float* kv    = nullptr;  // [n_tokens, entry_width]
    const float* gate  = nullptr;  // [n_tokens, entry_width]
    uint32_t     n_tokens = 0;     // always a multiple of compress_ratio
    uint32_t     first_window_position = 0;  // entry_count * compress_ratio
};

// The previous forward call's Ca slice for the CSA overlap scheme.
// `valid == false` on the very first call (reference returns None), which the
// compressor turns into zero-kv / -inf-gate slots.
struct Ds4OverlapSlice {
    const float* kv    = nullptr;  // [compress_ratio, head_dim]
    const float* gate  = nullptr;  // [compress_ratio, head_dim]
    bool         valid = false;
};

// A geometrically-grown device array.  Shared by the cache and its scratch so
// that `grow` is written once.
struct Ds4Growable {
    float*   p   = nullptr;  // device USM
    uint64_t cap = 0;        // elements
};

// The compressed entry arrays — the ONLY context-growing state — are stored at
// a per-entry width, fp16 or fp32 (Ds4CacheConfig::f16_values / f16_keys;
// docs/deepseek4/73 Phase 2).  fp16 halves their VRAM (13.76 -> 6.88 kB per
// token per card across the layers) and, for the K=V rows, holds exactly the
// values the XMX prefill attention already computed on (it converted these
// rows with the same saturating cast on every call).  Rows are converted ONCE,
// when appended.  Every allocation carries kDs4CompressedPadRows ZEROED rows
// past the last live entry, so an attention kernel that reads whole 64-row key
// blocks straight out of the cache never reads past it or into non-finite
// memory.
struct Ds4GrowableW {
    void*    p   = nullptr;  // device USM, elements of compressed_elem_bytes(e)
    uint64_t cap = 0;        // elements
};
inline constexpr uint32_t kDs4CompressedPadRows = 64;

// The transient working buffers of `Ds4LayerCache` — see the PERSISTENT vs
// TRANSIENT note at the top of this file.  ONE of these serves every sequence
// slot at a given layer.
//
// THE SAFETY CONDITION, STATED SO IT CAN BE CHECKED: a pointer obtained from
// this scratch by sequence A must be dead before sequence B touches the same
// layer.  It is, for two independent reasons, and both are properties of the
// runtime rather than of this class:
//   * within one forward, `layer_forward_post` drains the compute queue at the
//     end of every layer, which is the same drain that already makes the
//     persistent buffers safe to reallocate (see the comment at
//     `layer_forward_pre`'s cache call);
//   * between forwards, `forward_epilogue` reads logits back to the host, so
//     the queue is empty before the next sequence's forward is submitted.
// A batched forward that ran two sequences' cache updates for one layer WITHOUT
// a drain between them would violate this; that is why continuous batching
// needs a per-row scratch and this class is not it.
class Ds4CacheScratch {
public:
    Ds4CacheScratch() = default;
    ~Ds4CacheScratch() noexcept { free_storage(); }
    Ds4CacheScratch(const Ds4CacheScratch&)            = delete;
    Ds4CacheScratch& operator=(const Ds4CacheScratch&) = delete;
    Ds4CacheScratch(Ds4CacheScratch&&) noexcept;
    Ds4CacheScratch& operator=(Ds4CacheScratch&&) noexcept;

    void bind(sycl::queue& q) noexcept { q_ = &q; }
    void free_storage() noexcept;
    bool ready() const noexcept { return q_ != nullptr; }
    // Device bytes this scratch currently holds (capacity, fp32).  Reported
    // apart from the per-slot figure precisely because it is NOT charged per
    // sequence: adding a slot does not add any of it.
    uint64_t allocated_bytes() const noexcept;

private:
    friend class Ds4LayerCache;
    void grow(Ds4Growable& g, uint64_t need);
    void release(Ds4Growable& g) noexcept;

    sycl::queue* q_ = nullptr;
    Ds4Growable  sliding_full_{};                    // returned concatenation
    Ds4Growable  chunk_kv_[kDs4CacheEntries]{};      // window-aligned peel-off
    Ds4Growable  chunk_gate_[kDs4CacheEntries]{};
};

// A layer's cache state at one depth, in PINNED HOST memory (docs/deepseek4/72
// Phase L3: the prompt cache keeps several conversations' states on the host,
// 6.88 kB per token per card across the layers (fp16 compressed entries), instead of one snapshot in
// VRAM).  Filled by Ds4LayerCache::export_host, consumed by import_host.
// Buffers are allocated on `q`'s context and freed there; moves transfer them.
struct Ds4LayerHostSnap {
    sycl::queue* q = nullptr;
    float*   sliding = nullptr;                 uint64_t sliding_cap = 0;   // elements
    float*   buf_kv[kDs4CacheEntries]{};        uint64_t buf_cap[kDs4CacheEntries]{};
    float*   buf_gate[kDs4CacheEntries]{};      uint64_t buf_gate_cap[kDs4CacheEntries]{};
    uint8_t* compressed[kDs4CacheEntries]{};    uint64_t comp_cap[kDs4CacheEntries]{};   // BYTES
    uint32_t comp_elem_bytes[kDs4CacheEntries]{};   // 2 or 4, the width it was exported at
    float*   overlap_kv[kDs4CacheEntries]{};    uint64_t overlap_cap[kDs4CacheEntries]{};
    float*   overlap_gate[kDs4CacheEntries]{};  uint64_t overlap_gate_cap[kDs4CacheEntries]{};
    uint32_t sliding_len = 0, sliding_full_len = 0;
    uint64_t cumulative_length = 0;
    uint32_t buf_len[kDs4CacheEntries]{}, entry_count[kDs4CacheEntries]{};
    bool     overlap_valid[kDs4CacheEntries]{};
    uint32_t head_dim = 0, compress_ratio = 0;   // the config it was taken under

    Ds4LayerHostSnap() = default;
    ~Ds4LayerHostSnap() noexcept { free(); }
    Ds4LayerHostSnap(const Ds4LayerHostSnap&)            = delete;
    Ds4LayerHostSnap& operator=(const Ds4LayerHostSnap&) = delete;
    Ds4LayerHostSnap(Ds4LayerHostSnap&& o) noexcept { *this = std::move(o); }
    Ds4LayerHostSnap& operator=(Ds4LayerHostSnap&& o) noexcept;
    void     free() noexcept;
    uint64_t bytes() const noexcept;
};

class Ds4LayerCache {
public:
    Ds4LayerCache() = default;
    ~Ds4LayerCache() noexcept;
    Ds4LayerCache(const Ds4LayerCache&)            = delete;
    Ds4LayerCache& operator=(const Ds4LayerCache&) = delete;
    Ds4LayerCache(Ds4LayerCache&&) noexcept;
    Ds4LayerCache& operator=(Ds4LayerCache&&) noexcept;

    // Binds the queue; allocates nothing yet (the reference lazily initialises
    // too).  Returns a non-empty error string on an invalid config.
    //
    // `scratch`, when non-null, is the SHARED transient working memory for this
    // layer (see Ds4CacheScratch).  It must outlive this object and must be
    // bound to the same queue.  Null keeps a private scratch, which is exactly
    // what this class did before the split — so a single-sequence caller that
    // passes nothing is byte-for-byte unchanged.
    std::string init(sycl::queue& q, const Ds4CacheConfig& cfg,
                     Ds4CacheScratch* scratch = nullptr);
    void        reset() noexcept;          // drop all state, keep capacity
    void        free_storage() noexcept;

    const Ds4CacheConfig& config() const noexcept { return cfg_; }
    bool                  ready() const noexcept { return q_ != nullptr; }
    bool                  has_compressor() const noexcept { return cfg_.compress_ratio != 0; }
    bool                  has_indexer() const noexcept { return cfg_.compress_ratio == 4; }
    // Width of one *source-token* projection row for the given entry, i.e. the
    // width of the reference's kv_proj / gate_proj output.  Ratio 4 doubles it
    // (the Ca|Cb pair); ratio 128 does not.  Returns 0 for a sliding layer.
    uint32_t entry_source_width(Ds4CacheEntry e) const noexcept;
    // Width of one *compressed* entry (the post-kv_norm row).
    uint32_t entry_width(Ds4CacheEntry e) const noexcept;

    // ---- sliding K=V branch (reference DeepseekV4HCACache.update, modeling:204)
    // Appends `T` rows of [head_dim] and returns the FULL concatenation
    // (retained ++ new, length written to *out_len) exactly as the reference
    // returns `full`.  Internally retains only the last `sliding_window - 1`
    // rows.  The returned pointer is valid until the next update_sliding().
    const float* update_sliding(const float* kv_in, uint32_t T, uint32_t* out_len);

    uint32_t sliding_len() const noexcept { return sliding_len_; }
    uint64_t cumulative_length() const noexcept { return cumulative_length_; }

    // ---- compressor window machinery (modeling:218, 241, 286) ----------------
    Ds4CompressChunk store_compression_weights(Ds4CacheEntry e,
                                               const float* kv, const float* gate,
                                               uint32_t T);
    // Appends `n` freshly emitted compressed rows (fp32 in; stored at
    // compressed_elem_bytes(e), fp16 via the saturating cast); returns the
    // running array and writes the running count to *out_count.
    const void*  update_compressor_states(Ds4CacheEntry e, const float* compressed,
                                          uint32_t n, uint32_t* out_count);
    uint32_t     entry_count(Ds4CacheEntry e) const noexcept;
    const void*  compressed(Ds4CacheEntry e) const noexcept;
    // 2 (fp16) or 4 (fp32): the element width `compressed(e)` is stored at.
    uint32_t     compressed_elem_bytes(Ds4CacheEntry e) const noexcept;
    bool         compressed_f16(Ds4CacheEntry e) const noexcept { return compressed_elem_bytes(e) == 2; }

    // Reads the prior call's Ca slice and persists this call's last-window Ca
    // slice.  `chunk_kv` / `chunk_gate` are the [n_win, compress_ratio, width]
    // chunk (width == entry_source_width(e)); only `[..., :entry_width]` of the
    // last window is stored, matching modeling:298.
    Ds4OverlapSlice update_overlap_state(Ds4CacheEntry e,
                                         const float* chunk_kv, const float* chunk_gate,
                                         uint32_t n_win);

    // ---- prompt-cache snapshot / restore ------------------------------------
    // Deep device-to-device copy of the PERSISTENT half only.  `dst` must be
    // `init`-ed on the SAME queue with an identical config; it needs no scratch
    // and never grows one, because neither of these methods calls anything that
    // touches the transient half.
    //
    // Only the LIVE prefix of each array is copied (logical length, not
    // capacity), so a snapshot costs 22.55 MiB + 6.88 kB (fp16 compressed entries) per
    // token across all 43 layers — and not `allocated_bytes()`.
    //
    // WHY THIS IS A VALID PROMPT CACHE.  The state after consuming tokens
    // [0..N) is a pure function of those tokens, so restoring it and continuing
    // the prefill at pos0 = N is exactly equivalent to having prefilled all N
    // + suffix in one go.  That is the same contract the engine's other prompt
    // caches use; nothing here rewinds, and nothing needs to.
    //
    // Both return a non-empty error string on a config mismatch or a failed
    // device allocation.  On allocation failure the destination is left RESET
    // (empty, valid, usable) rather than half-written — a partially restored
    // cache would generate fluent text conditioned on nothing.
    std::string snapshot_to(Ds4LayerCache& dst) const;
    std::string restore_from(const Ds4LayerCache& src);
    // Host-resident form of the pair above (Phase L3).  export_host ISSUES the
    // D2H copies on this cache's queue and returns; the caller waits the queue
    // once for all layers.  import_host drains the queue, resets, and issues the
    // H2D copies (in-order: the next kernel on the queue sees them).
    std::string export_host(Ds4LayerHostSnap& dst) const;
    std::string import_host(const Ds4LayerHostSnap& src);

    // Pre-grow every growable to the capacity it will reach after consuming
    // `ctx` tokens, so the forward path NEVER reallocates.  The geometric
    // growth in `grow`/`append_compressed` is malloc + blocking copy + free,
    // per entry per layer — measured 2026-08-08: the accumulated churn (not
    // any kernel) collapsed decode from 37 ms/token at depth 8192 to a
    // UNIFORM 5.06 s/token at 12288 (GPU kernels still 46 ms in the same
    // step), and on the live server one over-long job then blew the xe
    // watchdog -> Engine reset -> DEVICE_LOST.  Pre-sizing converts all of
    // that into one up-front allocation the residency planner can account.
    // Returns non-empty on allocation failure; the cache is left valid and
    // lazy growth still works (the caller may treat failure as fatal or not).
    std::string reserve(uint64_t ctx);

    // ---- accounting (Phase 3 pass criterion 5) ------------------------------
    // Elements that grow with context: the compressed entry arrays only.
    uint64_t growing_elements() const noexcept;
    // Every live element of the cache state (logical, not allocated capacity):
    // sliding ring + window buffers + overlap slices + compressed arrays.
    uint64_t state_elements() const noexcept;
    uint64_t state_bytes(uint32_t elem_bytes) const noexcept {
        return state_elements() * elem_bytes;
    }
    // Bytes this object actually holds on the device right now (capacity,
    // fp32).  Reported separately so the 2-byte production figure is never
    // confused with the fp32 footprint.
    //
    // PER-SEQUENCE ONLY as of the transient/persistent split: the shared
    // scratch is NOT counted here, because it is not what a second sequence
    // costs.  Ask `Ds4CacheScratch::allocated_bytes()` for that half.  A cache
    // holding its own private scratch counts it, since then it really is part
    // of what that one cache holds.
    uint64_t allocated_bytes() const noexcept;
    // The shared scratch this cache was bound to, or its private one.
    const Ds4CacheScratch& scratch() const noexcept { return *sc_; }

private:
    using Growable = Ds4Growable;
    void  grow(Growable& g, uint64_t need);
    // As `grow`, but reports a failed `malloc_device` instead of storing null.
    // Used only by the snapshot path, which must degrade to "no cache" rather
    // than crash when VRAM is short (the same rule FleetPrefixCache follows).
    bool  grow_checked(Growable& g, uint64_t need);
    void  release(Growable& g) noexcept;
    // The compressed arrays (`eb` = element bytes): a fresh allocation is zeroed
    // (pad rows stay finite); the live prefix is NOT preserved —
    // update_compressor_states copies it itself, every other caller resets first.
    bool  growW_checked(Ds4GrowableW& g, uint64_t need, uint32_t eb, bool exact = false);
    void  releaseW(Ds4GrowableW& g) noexcept;

    sycl::queue*   q_ = nullptr;
    Ds4CacheConfig cfg_{};
    // Transient working memory.  Points at `own_scratch_` unless `init` was
    // given a shared one; never null after a successful `init`.
    Ds4CacheScratch* sc_ = nullptr;
    Ds4CacheScratch  own_scratch_{};

    // sliding K=V (K and V alias — one array, exactly as the reference)
    Growable sliding_{};        // [sliding_len_, head_dim]
    uint32_t sliding_len_      = 0;
    uint32_t sliding_full_len_ = 0;
    uint64_t cumulative_length_ = 0;

    // per-entry compressor state
    Growable buf_kv_[kDs4CacheEntries]{};
    Growable buf_gate_[kDs4CacheEntries]{};
    Ds4GrowableW compressed_[kDs4CacheEntries]{};   // per-entry width, see Ds4GrowableW
    // Ping-pong so update_overlap_state can hand back the prior slice while
    // writing the new one.
    Growable overlap_kv_[kDs4CacheEntries][2]{};
    Growable overlap_gate_[kDs4CacheEntries][2]{};
    uint32_t buf_len_[kDs4CacheEntries]{};
    uint32_t entry_count_[kDs4CacheEntries]{};
    uint32_t overlap_slot_[kDs4CacheEntries]{};
    bool     overlap_valid_[kDs4CacheEntries]{};
};

}  // namespace ie
