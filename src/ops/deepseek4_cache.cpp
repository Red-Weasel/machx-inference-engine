// src/ops/deepseek4_cache.cpp — DeepSeek-V4 per-layer attention cache (Phase 3).
//
// Every method is a transcription of the correspondingly named method on
// DeepseekV4HCACache / DeepseekV4CSACache in
// transformers/models/deepseek_v4/modeling_deepseek_v4.py (lines 171-300).
// Line references are given at each site.
//
// This is a STATE-MANAGEMENT path, not a hot kernel: the copies are plain queue
// memcpys, and every pointer a method returns is safe to hand straight to a
// kernel ON THE SAME QUEUE, which is in-order.  Not safe to READ ON THE HOST —
// no caller does, and buying that guarantee cost a host round trip per call per
// layer per card, which the two-card forward pays 61 times a token per card.
//
// The waits that REMAIN are the ones the geometric-growth realloc needs: `grow`
// does alloc → copy → free of the OLD pointer, and `sycl::free` is not ordered
// against anything the queue is doing, so a copy whose SOURCE is about to be
// freed must have completed first.  Each is marked LOAD-BEARING at its site.
// The wider invariant they sit inside — that no command reading layer L's cache
// buffers is ever in flight when layer L reallocates them — is `layer_forward_post`'s
// per-layer drain, which is kept for exactly this reason.

#include "ie/deepseek4_cache.hpp"

#include <algorithm>
#include <cstdio>
#include <new>
#include <utility>

namespace ie {

namespace {
// Geometric growth so a per-token decode loop does not realloc every step.
uint64_t next_capacity(uint64_t have, uint64_t need) {
    uint64_t cap = have ? have : 1024;
    while (cap < need) cap *= 2;
    return cap;
}

// A growable buffer that could not be grown ends the run, here, with a number.
//
// WHY THIS THROWS INSTEAD OF RETURNING.  These caches allocate lazily during
// forward, out of whatever VRAM the expert arena left behind, and a doubling
// lands mid-decode.  The old code took `malloc_device`'s null, freed the
// previous buffer anyway, and committed the new capacity — so the entry was
// left with p == nullptr and cap == what-was-asked-for.  Every later call then
// short-circuited on `cap >= need` and handed kernels `nullptr + offset`: a low,
// unmapped device address, dereferenced on the COMPUTE engine.  The driver
// cannot fix that fault up (-ENOENT), recovers the context, and the run limps
// at a fraction of its speed until a job fails to recover and takes the device
// with it (`CAT error class=ccs` -> UR_RESULT_ERROR_DEVICE_LOST).
//
// An out-of-memory that says so is strictly better than a GPU that dies twenty
// minutes later somewhere else, so this is fatal and legible.  Every caller
// sits in a non-noexcept forward path.
[[noreturn]] void cache_oom(const char* what, uint64_t cap) {
    std::fprintf(stderr,
                 "[ds4-cache] out of device memory growing %s to %llu elements "
                 "(%.3f GB). The KV/compressor caches grow during decode out of "
                 "the VRAM left over after the expert arena is sized; lower --ctx "
                 "or leave the arena more headroom.\n",
                 what, static_cast<unsigned long long>(cap),
                 double(cap * sizeof(float)) / (1024.0 * 1024.0 * 1024.0));
    throw std::bad_alloc();
}
}  // namespace

// ---------------------------------------------------------------------------
// Ds4CacheScratch — the transient half.  Nothing here is per-sequence state;
// see the header's PERSISTENT vs TRANSIENT note.
// ---------------------------------------------------------------------------
Ds4CacheScratch::Ds4CacheScratch(Ds4CacheScratch&& o) noexcept { *this = std::move(o); }

Ds4CacheScratch& Ds4CacheScratch::operator=(Ds4CacheScratch&& o) noexcept {
    if (this == &o) return *this;
    free_storage();
    q_            = o.q_;
    sliding_full_ = o.sliding_full_;
    for (uint32_t i = 0; i < kDs4CacheEntries; ++i) {
        chunk_kv_[i]   = o.chunk_kv_[i];
        chunk_gate_[i] = o.chunk_gate_[i];
    }
    o.q_ = nullptr;
    o.sliding_full_ = {};
    for (uint32_t i = 0; i < kDs4CacheEntries; ++i) {
        o.chunk_kv_[i] = {}; o.chunk_gate_[i] = {};
    }
    return *this;
}

void Ds4CacheScratch::release(Ds4Growable& g) noexcept {
    if (g.p && q_) sycl::free(g.p, *q_);
    g.p = nullptr;
    g.cap = 0;
}

void Ds4CacheScratch::free_storage() noexcept {
    release(sliding_full_);
    for (uint32_t i = 0; i < kDs4CacheEntries; ++i) {
        release(chunk_kv_[i]); release(chunk_gate_[i]);
    }
}

void Ds4CacheScratch::grow(Ds4Growable& g, uint64_t need) {
    if (g.cap >= need) return;
    const uint64_t cap = next_capacity(g.cap, need);
    // Allocate BEFORE releasing the old buffer, and commit nothing until the new
    // one exists: a failure here must not leave a null pointer behind a capacity
    // that says it is fine. See cache_oom.
    float* p = sycl::malloc_device<float>(cap, *q_);
    if (!p) cache_oom("scratch buffer", cap);
    if (g.p) sycl::free(g.p, *q_);
    g.p = p;
    g.cap = cap;
}

uint64_t Ds4CacheScratch::allocated_bytes() const noexcept {
    uint64_t n = sliding_full_.cap;
    for (uint32_t i = 0; i < kDs4CacheEntries; ++i)
        n += chunk_kv_[i].cap + chunk_gate_[i].cap;
    return n * sizeof(float);
}

// ---------------------------------------------------------------------------
// Ds4LayerCache — the persistent half.
// ---------------------------------------------------------------------------
Ds4LayerCache::~Ds4LayerCache() noexcept { free_storage(); }

Ds4LayerCache::Ds4LayerCache(Ds4LayerCache&& o) noexcept { *this = std::move(o); }

Ds4LayerCache& Ds4LayerCache::operator=(Ds4LayerCache&& o) noexcept {
    if (this == &o) return *this;
    free_storage();
    q_   = o.q_;
    cfg_ = o.cfg_;
    // A PRIVATE scratch moves with the object; a SHARED one is owned elsewhere
    // and only the pointer moves.  Getting this backwards would either free
    // another slot's scratch or leak this one, so the distinction is explicit.
    const bool private_scratch = (o.sc_ == &o.own_scratch_);
    own_scratch_ = std::move(o.own_scratch_);
    sc_          = private_scratch ? &own_scratch_ : o.sc_;
    sliding_          = o.sliding_;
    sliding_len_      = o.sliding_len_;
    sliding_full_len_ = o.sliding_full_len_;
    cumulative_length_ = o.cumulative_length_;
    for (uint32_t i = 0; i < kDs4CacheEntries; ++i) {
        buf_kv_[i] = o.buf_kv_[i];
        buf_gate_[i] = o.buf_gate_[i];
        compressed_[i] = o.compressed_[i];
        overlap_kv_[i][0] = o.overlap_kv_[i][0];
        overlap_kv_[i][1] = o.overlap_kv_[i][1];
        overlap_gate_[i][0] = o.overlap_gate_[i][0];
        overlap_gate_[i][1] = o.overlap_gate_[i][1];
        buf_len_[i] = o.buf_len_[i];
        entry_count_[i] = o.entry_count_[i];
        overlap_slot_[i] = o.overlap_slot_[i];
        overlap_valid_[i] = o.overlap_valid_[i];
    }
    o.q_ = nullptr;
    o.sc_ = nullptr;
    o.sliding_ = {};
    for (uint32_t i = 0; i < kDs4CacheEntries; ++i) {
        o.buf_kv_[i] = {}; o.buf_gate_[i] = {};
        o.compressed_[i] = {};
        o.overlap_kv_[i][0] = {}; o.overlap_kv_[i][1] = {};
        o.overlap_gate_[i][0] = {}; o.overlap_gate_[i][1] = {};
    }
    return *this;
}

std::string Ds4LayerCache::init(sycl::queue& q, const Ds4CacheConfig& cfg,
                                Ds4CacheScratch* scratch) {
    if (cfg.compress_ratio != 0 && cfg.compress_ratio != 4 && cfg.compress_ratio != 128)
        return "Ds4LayerCache: compress_ratio must be 0, 4 or 128, got "
               + std::to_string(cfg.compress_ratio);
    if (cfg.head_dim == 0) return "Ds4LayerCache: head_dim must be non-zero";
    if (cfg.sliding_window < 2) return "Ds4LayerCache: sliding_window must be >= 2";
    if (cfg.compress_ratio == 4 && cfg.index_head_dim == 0)
        return "Ds4LayerCache: index_head_dim must be non-zero on a CSA layer";
    free_storage();
    q_   = &q;
    cfg_ = cfg;
    if (scratch) {
        if (!scratch->ready()) scratch->bind(q);
        sc_ = scratch;
    } else {
        own_scratch_.bind(q);
        sc_ = &own_scratch_;
    }
    return {};
}

std::string Ds4LayerCache::reserve(uint64_t ctx) {
    if (!q_) return "Ds4LayerCache::reserve: init first";
    // Mirrors every forward-path growth site exactly (the need expressions are
    // copied from them, not re-derived): the sliding ring's fixed retention,
    // the per-entry window buffers' fixed (rate-1)*source, the ping-pong
    // overlap slices' fixed rate*width, and the compressed entry arrays at
    // ctx/rate entries — the only truly growing term, reserved EXACTLY (plus
    // the pad rows), since nothing grows it again while the context stays
    // within `ctx`; the fixed-size buffers keep grow_checked's rounding.  A
    // forward pass after this finds every `cap >= need` test true and never
    // reallocates or drains.
    const uint64_t d = cfg_.head_dim;
    if (!grow_checked(sliding_, uint64_t(cfg_.sliding_window - 1u) * d))
        return "Ds4LayerCache::reserve: sliding ring";
    if (cfg_.compress_ratio) {
        const uint32_t rate = cfg_.compress_ratio;
        for (uint32_t i = 0; i < kDs4CacheEntries; ++i) {
            const auto e = Ds4CacheEntry(i);
            if (e == Ds4CacheEntry::Indexer && !has_indexer()) continue;
            const uint32_t w  = entry_width(e);
            const uint32_t sw = entry_source_width(e);
            const uint64_t buf_need = uint64_t(rate - 1u) * sw;
            if (!grow_checked(buf_kv_[i], buf_need) || !grow_checked(buf_gate_[i], buf_need))
                return "Ds4LayerCache::reserve: window buffer";
            for (uint32_t s = 0; s < 2; ++s)
                if (!grow_checked(overlap_kv_[i][s], uint64_t(rate) * w) ||
                    !grow_checked(overlap_gate_[i][s], uint64_t(rate) * w))
                    return "Ds4LayerCache::reserve: overlap slice";
            const uint64_t entries = ctx / rate + 1u + kDs4CompressedPadRows;
            const uint32_t eb      = compressed_elem_bytes(e);
            if (!growW_checked(compressed_[i], entries * w, eb, /*exact=*/true))
                return "Ds4LayerCache::reserve: compressed entries (" +
                       std::to_string(entries * w * eb) + " B)";
        }
    }
    return {};
}

void Ds4LayerCache::reset() noexcept {
    sliding_len_ = 0;
    sliding_full_len_ = 0;
    cumulative_length_ = 0;
    for (uint32_t i = 0; i < kDs4CacheEntries; ++i) {
        buf_len_[i] = 0;
        entry_count_[i] = 0;
        overlap_slot_[i] = 0;
        overlap_valid_[i] = false;
    }
}

void Ds4LayerCache::release(Growable& g) noexcept {
    if (g.p && q_) sycl::free(g.p, *q_);
    g.p = nullptr;
    g.cap = 0;
}

// Frees this SEQUENCE's storage.  A shared scratch is deliberately left alone —
// other slots are still using it, and the runtime that owns it frees it.  A
// private one is ours and goes.
void Ds4LayerCache::free_storage() noexcept {
    release(sliding_);
    for (uint32_t i = 0; i < kDs4CacheEntries; ++i) {
        release(buf_kv_[i]); release(buf_gate_[i]);
        releaseW(compressed_[i]);
        release(overlap_kv_[i][0]); release(overlap_kv_[i][1]);
        release(overlap_gate_[i][0]); release(overlap_gate_[i][1]);
    }
    own_scratch_.free_storage();
    reset();
}

// Grows to at least `need` elements, preserving nothing (callers that need the
// old contents copy them out first — only `compressed_` does, and it copies
// through the new pointer explicitly below).
void Ds4LayerCache::grow(Growable& g, uint64_t need) {
    if (g.cap >= need) return;
    const uint64_t cap = next_capacity(g.cap, need);
    // Same ordering rule as Ds4CacheScratch::grow — see cache_oom for what the
    // old free-then-assign cost. `grow_checked` below is the variant callers use
    // when a failure is recoverable; this one is on the forward path, where it
    // is not.
    float* p = sycl::malloc_device<float>(cap, *q_);
    if (!p) cache_oom("layer cache buffer", cap);
    if (g.p) sycl::free(g.p, *q_);
    g.p = p;
    g.cap = cap;
}

// As `grow`, but a failed allocation leaves the existing buffer untouched and
// says so.  `grow` cannot: it frees the old pointer before it would find out.
bool Ds4LayerCache::grow_checked(Growable& g, uint64_t need) {
    if (g.cap >= need) return true;
    const uint64_t cap = next_capacity(g.cap, need);
    float* p = sycl::malloc_device<float>(cap, *q_);
    if (!p) return false;
    if (g.p) sycl::free(g.p, *q_);
    g.p = p;
    g.cap = cap;
    return true;
}

bool Ds4LayerCache::growW_checked(Ds4GrowableW& g, uint64_t need, uint32_t eb, bool exact) {
    if (g.cap >= need) return true;
    // `exact` is the pre-size path: the capacity is the context's, not a
    // power of two above it.  The geometric rounding exists for the lazy
    // per-token growth; applied to a pre-size it allocated up to 2x the
    // context (3.65 GB for the 1.83 GB a 131072-token fp32 cache needs).
    const uint64_t cap = exact ? need : next_capacity(g.cap, need);
    void* p = sycl::malloc_device<uint8_t>(cap * eb, *q_);
    if (!p) return false;
    // Zeroed: the pad rows past the last live entry are read (masked) by the
    // XMX attention's whole-block loads and must be finite.
    q_->memset(p, 0, cap * eb);
    if (g.p) sycl::free(g.p, *q_);
    g.p = p;
    g.cap = cap;
    return true;
}

void Ds4LayerCache::releaseW(Ds4GrowableW& g) noexcept {
    if (g.p && q_) sycl::free(g.p, *q_);
    g.p = nullptr;
    g.cap = 0;
}

uint32_t Ds4LayerCache::entry_width(Ds4CacheEntry e) const noexcept {
    if (cfg_.compress_ratio == 0) return 0;
    return e == Ds4CacheEntry::Compressor ? cfg_.head_dim : cfg_.index_head_dim;
}

uint32_t Ds4LayerCache::entry_source_width(Ds4CacheEntry e) const noexcept {
    const uint32_t w = entry_width(e);
    // Ratio 4 (CSA) projects to 2*width — the Ca|Cb pair (modeling:616, 499).
    // Ratio 128 (HCA) projects to width (modeling:388).
    return cfg_.compress_ratio == 4 ? 2u * w : w;
}

uint32_t Ds4LayerCache::entry_count(Ds4CacheEntry e) const noexcept {
    return entry_count_[uint32_t(e)];
}

const void* Ds4LayerCache::compressed(Ds4CacheEntry e) const noexcept {
    return compressed_[uint32_t(e)].p;
}

uint32_t Ds4LayerCache::compressed_elem_bytes(Ds4CacheEntry e) const noexcept {
    return (e == Ds4CacheEntry::Compressor ? cfg_.f16_values : cfg_.f16_keys) ? 2u : 4u;
}

// ---------------------------------------------------------------------------
// Sliding K=V branch — DeepseekV4HCACache.update (modeling:204-216).
//
//   full = torch.cat([self.keys, key_states], dim=-2)
//   self.keys = full[:, :, -self.sliding_window + 1 :, :]
//   self.values = self.keys                 <-- alias, not a second array
//   return full, full
// ---------------------------------------------------------------------------
const float* Ds4LayerCache::update_sliding(const float* kv_in, uint32_t T, uint32_t* out_len) {
    const uint32_t d = cfg_.head_dim;
    const uint64_t total = uint64_t(sliding_len_) + T;
    Ds4Growable& full = sc_->sliding_full_;
    sc_->grow(full, total * d);
    if (sliding_len_)
        q_->memcpy(full.p, sliding_.p, uint64_t(sliding_len_) * d * sizeof(float));
    if (T)
        q_->memcpy(full.p + uint64_t(sliding_len_) * d, kv_in,
                   uint64_t(T) * d * sizeof(float));
    // LOAD-BEARING, BUT ONLY WHEN THE GROW BELOW ACTUALLY REALLOCATES.
    //
    // The hazard is real: `grow(sliding_)` may `sycl::free(sliding_.p)`, and the
    // first memcpy above READS `sliding_.p`. Freeing it while that copy is still
    // queued is a use-after-free on the device.
    //
    // But that grow is to a FIXED size — `(sliding_window - 1) * d` — so it
    // reallocates at most once per layer and never again. In the decode steady
    // state this drained the whole compute queue every layer to protect a free
    // that cannot happen. `grow` is a no-op exactly when `cap >= need`
    // (deepseek4_cache.cpp:129), so asking that question first is precise, not a
    // heuristic: when it is false nothing is freed and nothing needs ordering.
    //
    // Submission cost is 1.43 us and a submit+wait round trip is 5.11 us on this
    // box, and `submit` is 70-87% of card-busy during decode — so removing an
    // unconditional per-layer drain is directly in the decode budget.
    const uint64_t sliding_need = uint64_t(cfg_.sliding_window - 1u) * d;
    if (sliding_.cap < sliding_need) q_->wait();

    const uint32_t keep = uint32_t(std::min<uint64_t>(total, cfg_.sliding_window - 1u));
    grow(sliding_, sliding_need);
    if (keep)
        q_->memcpy(sliding_.p, full.p + (total - keep) * d,
                   uint64_t(keep) * d * sizeof(float));
    sliding_len_ = keep;
    sliding_full_len_ = uint32_t(total);
    cumulative_length_ += T;
    if (out_len) *out_len = uint32_t(total);
    return full.p;
}

// ---------------------------------------------------------------------------
// store_compression_weights — modeling:218-239.
//
//   first_window_position = entry_count[name] * compress_rate
//   kv   = cat([buffer_kv[name], kv], dim=1)     (when the buffer is non-empty)
//   gate = cat([buffer_gate[name], gate], dim=1)
//   usable = (kv.shape[1] // compress_rate) * compress_rate
//   buffer_kv[name], buffer_gate[name] = kv[:, usable:], gate[:, usable:]
//   return kv[:, :usable], gate[:, :usable], first_window_position
// ---------------------------------------------------------------------------
Ds4CompressChunk Ds4LayerCache::store_compression_weights(Ds4CacheEntry e,
                                                          const float* kv, const float* gate,
                                                          uint32_t T) {
    const uint32_t i = uint32_t(e);
    const uint32_t w = entry_source_width(e);
    const uint32_t rate = cfg_.compress_ratio;
    const uint64_t total = uint64_t(buf_len_[i]) + T;

    Ds4Growable& ck = sc_->chunk_kv_[i];
    Ds4Growable& cg = sc_->chunk_gate_[i];
    sc_->grow(ck, total * w);
    sc_->grow(cg, total * w);
    if (buf_len_[i]) {
        const uint64_t nb = uint64_t(buf_len_[i]) * w * sizeof(float);
        q_->memcpy(ck.p, buf_kv_[i].p, nb);
        q_->memcpy(cg.p, buf_gate_[i].p, nb);
    }
    if (T) {
        const uint64_t off = uint64_t(buf_len_[i]) * w;
        q_->memcpy(ck.p + off, kv, uint64_t(T) * w * sizeof(float));
        q_->memcpy(cg.p + off, gate, uint64_t(T) * w * sizeof(float));
    }
    // LOAD-BEARING, BUT ONLY WHEN A GROW BELOW ACTUALLY REALLOCATES — same
    // reasoning as update_sliding above. The two grows are to the FIXED size
    // `(rate - 1) * w`, so after the first few tokens `cap >= need` holds forever
    // and neither buffer is ever freed again; the drain was protecting nothing.
    // `grow` no-ops exactly when `cap >= need`, so testing that is exact.
    //
    // This one is worth more than the sliding drain: it runs once per compressor
    // entry, i.e. TWICE per CSA layer (Compressor + Indexer), so it was up to
    // ~129 device round-trips per token per card on its own.
    const uint64_t buf_need = uint64_t(rate - 1u) * w;
    if (buf_kv_[i].cap < buf_need || buf_gate_[i].cap < buf_need) q_->wait();

    const uint64_t usable   = (total / rate) * rate;
    const uint64_t leftover = total - usable;
    grow(buf_kv_[i], buf_need);
    grow(buf_gate_[i], buf_need);
    if (leftover) {
        const uint64_t nb = leftover * w * sizeof(float);
        q_->memcpy(buf_kv_[i].p, ck.p + usable * w, nb);
        q_->memcpy(buf_gate_[i].p, cg.p + usable * w, nb);
    }
    buf_len_[i] = uint32_t(leftover);

    Ds4CompressChunk c;
    c.kv       = ck.p;
    c.gate     = cg.p;
    c.n_tokens = uint32_t(usable);
    c.first_window_position = entry_count_[i] * rate;
    return c;
}

// ---------------------------------------------------------------------------
// update_compressor_states — modeling:241-252.
//   compressed_kv[name] = cat([compressed_kv[name], compressed], dim=1)
//   entry_count[name] += compressed.shape[1]
// ---------------------------------------------------------------------------
namespace {
// fp32 -> fp16 with saturation: the XMX attention's own cast
// (deepseek4_attn_xmx.cpp `sat16`), so a stored row is exactly what that kernel
// used to compute from the fp32 row on every call.  (Prefixed: the SYCL
// kernel-name collision trap.)
sycl::event ds4c_f32_to_f16_sat(sycl::queue& q, const float* src, sycl::half* dst, uint64_t n) {
    return q.parallel_for(sycl::range<1>(size_t(n)), [=](sycl::id<1> i) {
        const float v = sycl::fmin(sycl::fmax(src[i], -65504.0f), 65504.0f);
        dst[i] = sycl::half(v);
    });
}
}  // namespace

const void* Ds4LayerCache::update_compressor_states(Ds4CacheEntry e, const float* compressed,
                                                    uint32_t n, uint32_t* out_count) {
    const uint32_t i  = uint32_t(e);
    const uint32_t w  = entry_width(e);
    const uint32_t eb = compressed_elem_bytes(e);
    if (n) {
        const uint64_t need = (uint64_t(entry_count_[i]) + n + kDs4CompressedPadRows) * w;
        if (compressed_[i].cap < need) {
            const uint64_t cap = next_capacity(compressed_[i].cap, need);
            void* p = sycl::malloc_device<uint8_t>(cap * eb, *q_);
            // Checked BEFORE the copy below, which writes THROUGH `p` on the
            // device: a null here is not a pointer the run can survive handing
            // to a kernel, and this buffer grows with cumulative context, so it
            // is the one that reaches the VRAM ceiling first. See cache_oom.
            if (!p) cache_oom("compressor state", cap);
            // Zero the whole allocation (the pad rows must stay finite), then
            // the live prefix, both on the in-order queue.
            // LOAD-BEARING: the wait precedes the free of the copy's SOURCE.
            q_->memset(p, 0, cap * eb);
            if (entry_count_[i])
                q_->memcpy(p, compressed_[i].p, uint64_t(entry_count_[i]) * w * eb);
            q_->wait();
            if (compressed_[i].p) sycl::free(compressed_[i].p, *q_);
            compressed_[i].p = p;
            compressed_[i].cap = cap;
        }
        uint8_t* dst = static_cast<uint8_t*>(compressed_[i].p) + uint64_t(entry_count_[i]) * w * eb;
        if (eb == 2)
            ds4c_f32_to_f16_sat(*q_, compressed, reinterpret_cast<sycl::half*>(dst), uint64_t(n) * w);
        else
            q_->memcpy(dst, compressed, uint64_t(n) * w * sizeof(float));
        entry_count_[i] += n;
    }
    if (out_count) *out_count = entry_count_[i];
    return compressed_[i].p;
}

// ---------------------------------------------------------------------------
// update_overlap_state — modeling:286-300.
//   prior = overlap[name]
//   overlap_kv[name]   = chunk_kv[:, -1, :, :head_dim].clone()
//   overlap_gate[name] = chunk_gate[:, -1, :, :head_dim].clone()
//   return prior
//
// `chunk_*` is [n_win, compress_ratio, source_width]; only the first
// `entry_width` columns (the Ca half) of the LAST window are kept.
// ---------------------------------------------------------------------------
Ds4OverlapSlice Ds4LayerCache::update_overlap_state(Ds4CacheEntry e,
                                                    const float* chunk_kv, const float* chunk_gate,
                                                    uint32_t n_win) {
    const uint32_t i    = uint32_t(e);
    const uint32_t w    = entry_width(e);
    const uint32_t sw   = entry_source_width(e);
    const uint32_t rate = cfg_.compress_ratio;
    const uint32_t cur  = overlap_slot_[i];
    const uint32_t nxt  = 1u - cur;

    Ds4OverlapSlice prior;
    prior.valid = overlap_valid_[i];
    prior.kv    = overlap_valid_[i] ? overlap_kv_[i][cur].p : nullptr;
    prior.gate  = overlap_valid_[i] ? overlap_gate_[i][cur].p : nullptr;

    if (n_win) {
        grow(overlap_kv_[i][nxt], uint64_t(rate) * w);
        grow(overlap_gate_[i][nxt], uint64_t(rate) * w);
        const uint64_t base = (uint64_t(n_win) - 1u) * rate * sw;
        for (uint32_t j = 0; j < rate; ++j) {
            q_->memcpy(overlap_kv_[i][nxt].p + uint64_t(j) * w,
                       chunk_kv + base + uint64_t(j) * sw, uint64_t(w) * sizeof(float));
            q_->memcpy(overlap_gate_[i][nxt].p + uint64_t(j) * w,
                       chunk_gate + base + uint64_t(j) * sw, uint64_t(w) * sizeof(float));
        }
        // No drain: these fill the slot that becomes `prior` on the NEXT call,
        // which reads it on this same queue.  `prior` as returned NOW is the
        // OTHER slot and is untouched here.
        overlap_slot_[i]  = nxt;
        overlap_valid_[i] = true;
    }
    return prior;
}

// ---------------------------------------------------------------------------
// Prompt-cache snapshot / restore — the PERSISTENT half only.
//
// Both are written as "size every allocation first, then copy" so that a failed
// allocation can bail with the destination merely EMPTY rather than half-filled.
// A half-restored cache is the worst failure this subsystem has: it does not
// crash, it answers a question with somebody else's context.
// ---------------------------------------------------------------------------
namespace {
bool same_cfg(const Ds4CacheConfig& a, const Ds4CacheConfig& b) noexcept {
    return a.compress_ratio == b.compress_ratio && a.head_dim == b.head_dim &&
           a.index_head_dim == b.index_head_dim && a.sliding_window == b.sliding_window &&
           a.f16_values == b.f16_values && a.f16_keys == b.f16_keys;
}
constexpr const char* kOom = "device allocation failed (snapshot skipped)";
}  // namespace

std::string Ds4LayerCache::snapshot_to(Ds4LayerCache& dst) const {
    if (!q_ || !dst.q_)   return "Ds4LayerCache::snapshot_to: cache not initialised";
    if (dst.q_ != q_)     return "Ds4LayerCache::snapshot_to: destination is on another queue";
    if (!same_cfg(cfg_, dst.cfg_))
        return "Ds4LayerCache::snapshot_to: config mismatch";
    if (&dst == this)     return {};

    dst.reset();
    const uint32_t d = cfg_.head_dim;
    const uint64_t n_slide = uint64_t(sliding_len_) * d;

    // --- size everything ---
    if (n_slide && !dst.grow_checked(dst.sliding_, n_slide))
        return std::string("Ds4LayerCache::snapshot_to: ") + kOom;
    for (uint32_t i = 0; i < kDs4CacheEntries; ++i) {
        const auto     e  = Ds4CacheEntry(i);
        const uint64_t sw = entry_source_width(e), w = entry_width(e);
        if (buf_len_[i] && (!dst.grow_checked(dst.buf_kv_[i],   uint64_t(buf_len_[i]) * sw) ||
                            !dst.grow_checked(dst.buf_gate_[i], uint64_t(buf_len_[i]) * sw)))
            return std::string("Ds4LayerCache::snapshot_to: ") + kOom;
        if (entry_count_[i] &&
            !dst.growW_checked(dst.compressed_[i],
                               (uint64_t(entry_count_[i]) + kDs4CompressedPadRows) * w,
                               compressed_elem_bytes(e)))
            return std::string("Ds4LayerCache::snapshot_to: ") + kOom;
        // The live overlap slice always lands in the destination's slot 0 — only
        // `[overlap_slot_]` is ever read, so which physical slot holds it is not
        // observable, and pinning it to 0 keeps the snapshot's second slot unallocated.
        if (overlap_valid_[i] &&
            (!dst.grow_checked(dst.overlap_kv_[i][0],   uint64_t(cfg_.compress_ratio) * w) ||
             !dst.grow_checked(dst.overlap_gate_[i][0], uint64_t(cfg_.compress_ratio) * w)))
            return std::string("Ds4LayerCache::snapshot_to: ") + kOom;
    }

    // --- copy ---
    if (n_slide) q_->memcpy(dst.sliding_.p, sliding_.p, n_slide * sizeof(float));
    for (uint32_t i = 0; i < kDs4CacheEntries; ++i) {
        const auto     e  = Ds4CacheEntry(i);
        const uint64_t sw = entry_source_width(e), w = entry_width(e);
        if (buf_len_[i]) {
            const uint64_t nb = uint64_t(buf_len_[i]) * sw * sizeof(float);
            q_->memcpy(dst.buf_kv_[i].p,   buf_kv_[i].p,   nb);
            q_->memcpy(dst.buf_gate_[i].p, buf_gate_[i].p, nb);
        }
        if (entry_count_[i])
            q_->memcpy(dst.compressed_[i].p, compressed_[i].p,
                       uint64_t(entry_count_[i]) * w * compressed_elem_bytes(e));
        if (overlap_valid_[i]) {
            const uint32_t cur = overlap_slot_[i];
            const uint64_t nb  = uint64_t(cfg_.compress_ratio) * w * sizeof(float);
            q_->memcpy(dst.overlap_kv_[i][0].p,   overlap_kv_[i][cur].p,   nb);
            q_->memcpy(dst.overlap_gate_[i][0].p, overlap_gate_[i][cur].p, nb);
        }
        dst.buf_len_[i]       = buf_len_[i];
        dst.entry_count_[i]   = entry_count_[i];
        dst.overlap_slot_[i]  = 0;
        dst.overlap_valid_[i] = overlap_valid_[i];
    }
    dst.sliding_len_       = sliding_len_;
    dst.sliding_full_len_  = sliding_full_len_;
    dst.cumulative_length_ = cumulative_length_;
    // LOAD-BEARING: the caller resumes generating on this queue the moment this
    // returns, which overwrites the very buffers these copies READ.
    q_->wait();
    return {};
}

// ---- host-resident snapshots (Phase L3) ----
Ds4LayerHostSnap& Ds4LayerHostSnap::operator=(Ds4LayerHostSnap&& o) noexcept {
    if (this == &o) return *this;
    free();
    q = o.q; sliding = o.sliding; sliding_cap = o.sliding_cap;
    for (uint32_t i = 0; i < kDs4CacheEntries; ++i) {
        buf_kv[i] = o.buf_kv[i]; buf_gate[i] = o.buf_gate[i]; buf_cap[i] = o.buf_cap[i]; buf_gate_cap[i] = o.buf_gate_cap[i];
        compressed[i] = o.compressed[i]; comp_cap[i] = o.comp_cap[i];
        overlap_kv[i] = o.overlap_kv[i]; overlap_gate[i] = o.overlap_gate[i]; overlap_cap[i] = o.overlap_cap[i]; overlap_gate_cap[i] = o.overlap_gate_cap[i];
        buf_len[i] = o.buf_len[i]; entry_count[i] = o.entry_count[i]; overlap_valid[i] = o.overlap_valid[i];
    }
    sliding_len = o.sliding_len; sliding_full_len = o.sliding_full_len;
    cumulative_length = o.cumulative_length; head_dim = o.head_dim; compress_ratio = o.compress_ratio;
    o.q = nullptr; o.sliding = nullptr; o.sliding_cap = 0;
    for (uint32_t i = 0; i < kDs4CacheEntries; ++i) {
        o.buf_kv[i] = o.buf_gate[i] = o.overlap_kv[i] = o.overlap_gate[i] = nullptr;
        o.compressed[i] = nullptr;
        o.buf_cap[i] = o.buf_gate_cap[i] = o.comp_cap[i] = o.overlap_cap[i] = o.overlap_gate_cap[i] = 0;
    }
    return *this;
}

void Ds4LayerHostSnap::free() noexcept {
    if (q) {
        auto rel = [&](auto*& p) { if (p) { sycl::free(p, *q); p = nullptr; } };
        rel(sliding);
        for (uint32_t i = 0; i < kDs4CacheEntries; ++i) {
            rel(buf_kv[i]); rel(buf_gate[i]); rel(compressed[i]); rel(overlap_kv[i]); rel(overlap_gate[i]);
        }
    }
    sliding_cap = 0;
    for (uint32_t i = 0; i < kDs4CacheEntries; ++i) {
        buf_cap[i] = buf_gate_cap[i] = comp_cap[i] = overlap_cap[i] = overlap_gate_cap[i] = 0;
        buf_len[i] = entry_count[i] = 0; overlap_valid[i] = false;
    }
    sliding_len = sliding_full_len = 0; cumulative_length = 0;
}

uint64_t Ds4LayerHostSnap::bytes() const noexcept {
    uint64_t n32 = sliding_cap, nb = 0;
    for (uint32_t i = 0; i < kDs4CacheEntries; ++i) {
        n32 += buf_cap[i] + buf_gate_cap[i] + overlap_cap[i] + overlap_gate_cap[i];
        nb  += comp_cap[i];   // bytes
    }
    return n32 * sizeof(float) + nb;
}

namespace {
// Pinned host growth for a snapshot buffer: exact size (a conversation's
// snapshot only ever grows with it), freed and reallocated when short.
template <typename T>
bool ds4c_grow_host(sycl::queue& q, T*& p, uint64_t& cap, uint64_t need) {
    if (cap >= need) return true;
    if (p) { q.wait(); sycl::free(p, q); p = nullptr; cap = 0; }
    p = sycl::malloc_host<T>(size_t(need), q);
    if (!p) return false;
    cap = need;
    return true;
}
}  // namespace

std::string Ds4LayerCache::export_host(Ds4LayerHostSnap& dst) const {
    if (!q_) return "Ds4LayerCache::export_host: cache not initialised";
    if (dst.q && dst.q != q_) dst.free();   // pinned on another context: start over
    dst.q = q_;
    const uint32_t d = cfg_.head_dim;
    const uint64_t n_slide = uint64_t(sliding_len_) * d;
    static const char* kOomH = "pinned host allocation failed";
    if (n_slide && !ds4c_grow_host(*q_, dst.sliding, dst.sliding_cap, n_slide))
        return std::string("Ds4LayerCache::export_host: ") + kOomH;
    for (uint32_t i = 0; i < kDs4CacheEntries; ++i) {
        const auto     e  = Ds4CacheEntry(i);
        const uint64_t sw = entry_source_width(e), w = entry_width(e);
        if (buf_len_[i]) {
            const uint64_t n = uint64_t(buf_len_[i]) * sw;
            if (!ds4c_grow_host(*q_, dst.buf_kv[i],   dst.buf_cap[i],      n)) return std::string("Ds4LayerCache::export_host: ") + kOomH;
            if (!ds4c_grow_host(*q_, dst.buf_gate[i], dst.buf_gate_cap[i], n)) return std::string("Ds4LayerCache::export_host: ") + kOomH;
        }
        if (entry_count_[i] &&
            !ds4c_grow_host(*q_, dst.compressed[i], dst.comp_cap[i],
                            uint64_t(entry_count_[i]) * w * compressed_elem_bytes(e)))
            return std::string("Ds4LayerCache::export_host: ") + kOomH;
        dst.comp_elem_bytes[i] = compressed_elem_bytes(e);
        if (overlap_valid_[i]) {
            const uint64_t n = uint64_t(cfg_.compress_ratio) * w;
            if (!ds4c_grow_host(*q_, dst.overlap_kv[i],   dst.overlap_cap[i],      n)) return std::string("Ds4LayerCache::export_host: ") + kOomH;
            if (!ds4c_grow_host(*q_, dst.overlap_gate[i], dst.overlap_gate_cap[i], n)) return std::string("Ds4LayerCache::export_host: ") + kOomH;
        }
    }
    if (n_slide) q_->memcpy(dst.sliding, sliding_.p, n_slide * sizeof(float));
    for (uint32_t i = 0; i < kDs4CacheEntries; ++i) {
        const auto     e  = Ds4CacheEntry(i);
        const uint64_t sw = entry_source_width(e), w = entry_width(e);
        if (buf_len_[i]) {
            const uint64_t nb = uint64_t(buf_len_[i]) * sw * sizeof(float);
            q_->memcpy(dst.buf_kv[i],   buf_kv_[i].p,   nb);
            q_->memcpy(dst.buf_gate[i], buf_gate_[i].p, nb);
        }
        if (entry_count_[i])
            q_->memcpy(dst.compressed[i], compressed_[i].p,
                       uint64_t(entry_count_[i]) * w * compressed_elem_bytes(e));
        if (overlap_valid_[i]) {
            const uint32_t cur = overlap_slot_[i];
            const uint64_t nb  = uint64_t(cfg_.compress_ratio) * w * sizeof(float);
            q_->memcpy(dst.overlap_kv[i],   overlap_kv_[i][cur].p,   nb);
            q_->memcpy(dst.overlap_gate[i], overlap_gate_[i][cur].p, nb);
        }
        dst.buf_len[i]       = buf_len_[i];
        dst.entry_count[i]   = entry_count_[i];
        dst.overlap_valid[i] = overlap_valid_[i];
    }
    dst.sliding_len       = sliding_len_;
    dst.sliding_full_len  = sliding_full_len_;
    dst.cumulative_length = cumulative_length_;
    dst.head_dim          = cfg_.head_dim;
    dst.compress_ratio    = cfg_.compress_ratio;
    return {};
}

std::string Ds4LayerCache::import_host(const Ds4LayerHostSnap& src) {
    if (!q_) return "Ds4LayerCache::import_host: cache not initialised";
    if (src.q != q_) return "Ds4LayerCache::import_host: snapshot pinned on another queue";
    if (src.head_dim != cfg_.head_dim || src.compress_ratio != cfg_.compress_ratio)
        return "Ds4LayerCache::import_host: config mismatch";
    for (uint32_t i = 0; i < kDs4CacheEntries; ++i)
        if (src.entry_count[i] && src.comp_elem_bytes[i] != compressed_elem_bytes(Ds4CacheEntry(i)))
            return "Ds4LayerCache::import_host: snapshot was taken at another cache width (IE_DS4_KV16)";
    q_->wait();
    reset();
    const uint32_t d = cfg_.head_dim;
    const uint64_t n_slide = uint64_t(src.sliding_len) * d;
    if (n_slide && !grow_checked(sliding_, n_slide))
        return std::string("Ds4LayerCache::import_host: ") + kOom;
    for (uint32_t i = 0; i < kDs4CacheEntries; ++i) {
        const auto     e  = Ds4CacheEntry(i);
        const uint64_t sw = entry_source_width(e), w = entry_width(e);
        if (src.buf_len[i] && (!grow_checked(buf_kv_[i],   uint64_t(src.buf_len[i]) * sw) ||
                               !grow_checked(buf_gate_[i], uint64_t(src.buf_len[i]) * sw)))
            return std::string("Ds4LayerCache::import_host: ") + kOom;
        if (src.entry_count[i] &&
            !growW_checked(compressed_[i], (uint64_t(src.entry_count[i]) + kDs4CompressedPadRows) * w,
                           compressed_elem_bytes(e)))
            return std::string("Ds4LayerCache::import_host: ") + kOom;
        if (src.overlap_valid[i] &&
            (!grow_checked(overlap_kv_[i][0],   uint64_t(cfg_.compress_ratio) * w) ||
             !grow_checked(overlap_gate_[i][0], uint64_t(cfg_.compress_ratio) * w)))
            return std::string("Ds4LayerCache::import_host: ") + kOom;
    }
    if (n_slide) q_->memcpy(sliding_.p, src.sliding, n_slide * sizeof(float));
    for (uint32_t i = 0; i < kDs4CacheEntries; ++i) {
        const auto     e  = Ds4CacheEntry(i);
        const uint64_t sw = entry_source_width(e), w = entry_width(e);
        if (src.buf_len[i]) {
            const uint64_t nb = uint64_t(src.buf_len[i]) * sw * sizeof(float);
            q_->memcpy(buf_kv_[i].p,   src.buf_kv[i],   nb);
            q_->memcpy(buf_gate_[i].p, src.buf_gate[i], nb);
        }
        if (src.entry_count[i])
            q_->memcpy(compressed_[i].p, src.compressed[i],
                       uint64_t(src.entry_count[i]) * w * compressed_elem_bytes(e));
        if (src.overlap_valid[i]) {
            const uint64_t nb = uint64_t(cfg_.compress_ratio) * w * sizeof(float);
            q_->memcpy(overlap_kv_[i][0].p,   src.overlap_kv[i],   nb);
            q_->memcpy(overlap_gate_[i][0].p, src.overlap_gate[i], nb);
        }
        buf_len_[i]       = src.buf_len[i];
        entry_count_[i]   = src.entry_count[i];
        overlap_slot_[i]  = 0;
        overlap_valid_[i] = src.overlap_valid[i];
    }
    sliding_len_       = src.sliding_len;
    sliding_full_len_  = src.sliding_full_len;
    cumulative_length_ = src.cumulative_length;
    return {};
}

std::string Ds4LayerCache::restore_from(const Ds4LayerCache& src) {
    if (!q_ || !src.q_)   return "Ds4LayerCache::restore_from: cache not initialised";
    if (src.q_ != q_)     return "Ds4LayerCache::restore_from: source is on another queue";
    if (!same_cfg(cfg_, src.cfg_))
        return "Ds4LayerCache::restore_from: config mismatch";
    if (&src == this)     return {};

    // LOAD-BEARING: `grow_checked` may free a buffer that the previous forward's
    // kernels are still reading.  Every other realloc site in this file is
    // covered by `layer_forward_post`'s per-layer drain; this one runs from the
    // engine between forwards, so it drains for itself.
    q_->wait();
    reset();
    const uint32_t d = cfg_.head_dim;
    const uint64_t n_slide = uint64_t(src.sliding_len_) * d;

    if (n_slide && !grow_checked(sliding_, n_slide))
        return std::string("Ds4LayerCache::restore_from: ") + kOom;
    for (uint32_t i = 0; i < kDs4CacheEntries; ++i) {
        const auto     e  = Ds4CacheEntry(i);
        const uint64_t sw = entry_source_width(e), w = entry_width(e);
        if (src.buf_len_[i] && (!grow_checked(buf_kv_[i],   uint64_t(src.buf_len_[i]) * sw) ||
                                !grow_checked(buf_gate_[i], uint64_t(src.buf_len_[i]) * sw)))
            return std::string("Ds4LayerCache::restore_from: ") + kOom;
        if (src.entry_count_[i] &&
            !growW_checked(compressed_[i], (uint64_t(src.entry_count_[i]) + kDs4CompressedPadRows) * w,
                           compressed_elem_bytes(e)))
            return std::string("Ds4LayerCache::restore_from: ") + kOom;
        if (src.overlap_valid_[i] &&
            (!grow_checked(overlap_kv_[i][0],   uint64_t(cfg_.compress_ratio) * w) ||
             !grow_checked(overlap_gate_[i][0], uint64_t(cfg_.compress_ratio) * w)))
            return std::string("Ds4LayerCache::restore_from: ") + kOom;
    }

    if (n_slide) q_->memcpy(sliding_.p, src.sliding_.p, n_slide * sizeof(float));
    for (uint32_t i = 0; i < kDs4CacheEntries; ++i) {
        const auto     e  = Ds4CacheEntry(i);
        const uint64_t sw = entry_source_width(e), w = entry_width(e);
        if (src.buf_len_[i]) {
            const uint64_t nb = uint64_t(src.buf_len_[i]) * sw * sizeof(float);
            q_->memcpy(buf_kv_[i].p,   src.buf_kv_[i].p,   nb);
            q_->memcpy(buf_gate_[i].p, src.buf_gate_[i].p, nb);
        }
        if (src.entry_count_[i])
            q_->memcpy(compressed_[i].p, src.compressed_[i].p,
                       uint64_t(src.entry_count_[i]) * w * compressed_elem_bytes(e));
        if (src.overlap_valid_[i]) {
            const uint32_t scur = src.overlap_slot_[i];
            const uint64_t nb   = uint64_t(cfg_.compress_ratio) * w * sizeof(float);
            q_->memcpy(overlap_kv_[i][0].p,   src.overlap_kv_[i][scur].p,   nb);
            q_->memcpy(overlap_gate_[i][0].p, src.overlap_gate_[i][scur].p, nb);
        }
        buf_len_[i]       = src.buf_len_[i];
        entry_count_[i]   = src.entry_count_[i];
        overlap_slot_[i]  = 0;
        overlap_valid_[i] = src.overlap_valid_[i];
    }
    sliding_len_       = src.sliding_len_;
    sliding_full_len_  = src.sliding_full_len_;
    cumulative_length_ = src.cumulative_length_;
    // LOAD-BEARING: the restored state must be on the device before the caller
    // submits the suffix prefill that reads it.
    q_->wait();
    return {};
}

// ---------------------------------------------------------------------------
// Accounting.
// ---------------------------------------------------------------------------
uint64_t Ds4LayerCache::growing_elements() const noexcept {
    uint64_t n = 0;
    for (uint32_t i = 0; i < kDs4CacheEntries; ++i)
        n += uint64_t(entry_count_[i]) * entry_width(Ds4CacheEntry(i));
    return n;
}

uint64_t Ds4LayerCache::state_elements() const noexcept {
    // K and V alias, so the sliding ring is counted ONCE (modeling:211).
    uint64_t n = uint64_t(sliding_len_) * cfg_.head_dim;
    for (uint32_t i = 0; i < kDs4CacheEntries; ++i) {
        const auto e = Ds4CacheEntry(i);
        n += uint64_t(buf_len_[i]) * entry_source_width(e) * 2u;  // kv + gate
        if (overlap_valid_[i]) n += uint64_t(cfg_.compress_ratio) * entry_width(e) * 2u;
    }
    return n + growing_elements();
}

uint64_t Ds4LayerCache::allocated_bytes() const noexcept {
    uint64_t n = sliding_.cap;
    uint64_t nb = 0;   // the compressed arrays, counted at their own width (bytes)
    for (uint32_t i = 0; i < kDs4CacheEntries; ++i) {
        n += buf_kv_[i].cap + buf_gate_[i].cap;
        nb += compressed_[i].cap * compressed_elem_bytes(Ds4CacheEntry(i));
        n += overlap_kv_[i][0].cap + overlap_kv_[i][1].cap;
        n += overlap_gate_[i][0].cap + overlap_gate_[i][1].cap;
    }
    // A PRIVATE scratch is part of what this one cache holds, so it is counted;
    // a SHARED one belongs to the runtime and is not this sequence's cost.
    // Under the single-sequence default the scratch IS private, so this total
    // is the same number the pre-split class reported.
    if (sc_ == &own_scratch_) n += own_scratch_.allocated_bytes() / sizeof(float);
    return n * sizeof(float) + nb;
}

}  // namespace ie
