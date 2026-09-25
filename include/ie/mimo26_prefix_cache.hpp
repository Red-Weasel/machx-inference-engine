// include/ie/mimo26_prefix_cache.hpp — MiMo-V2.6 host slots (P7, fix-list #70; docs/mimo26/P7_FIX64_FIX70.md).
//
// The live conversation's state stays on the cards as before (P3b: the full layers' caches truncate for free, the SWA rings
// serve a prefix while they still hold the window before it). OTHER conversations are kept in HOST SLOTS -- their full-layer
// K/V rows, the SWA ring slots of the positions the rings hold, the DFlash drafter's context ring, and the bookkeeping of
// all three -- in pageable host memory, under a byte budget and a MemAvailable floor, and swapped back in when a prompt
// matches one clearly better than the live state. A restored slot is the saved state byte for byte
// (Mimo26Forward::state_spans, Mimo26DFlash::state_spans): the next forward reads exactly what it would have read had the
// conversation stayed live. Eviction: one-shot states first, then conversations that were continued, least recently used
// first within each, one-shot states cheapest to rebuild first (mimo26_plan_admit, #77). #87: at each prompt's end the SWA
// rings' slots of the window before it (plus a margin) and the drafter's context are copied to a host snapshot, so a next
// prompt that discards the reply -- a length cut, an interrupt, an error, a regenerate -- is served from there once the
// reply's decode has overwritten the rings (the full layers' rows of the prompt are still on the cards). The snapshot is
// ~70-90 MiB of pageable RAM whatever the context, reused request to request, and OUTSIDE the byte budget, like the
// 2 x 64 MiB pinned bounces. Every transfer goes through a pinned bounce allocated in the queue's own context
// (a pageable or foreign-context pointer turns a SYCL memcpy into a minutes-long stall -- deepseek41_prefix_cache.cpp).
// One request at a time (the engine's parallel is 1 for this arch): no locking.
#pragma once

#include "ie/mimo26_dflash.hpp"
#include "ie/mimo26_forward.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ie {

class Mimo26PrefixCache {
public:
    struct Options {
        uint64_t budget = 16ull << 30;      // host slot bytes (IE_MIMO26_PROMPT_CACHE_GIB); 0 = the live conversation only
        uint64_t keep_free = 24ull << 30;   // never hold a new slot below this much MemAvailable (IE_MIMO26_CACHE_KEEP_FREE_GIB)
        uint32_t min_tokens = 1024;         // a shorter state is not kept; also the continuation slack (mimo26_continues)
        uint32_t swap_margin = 256;         // a slot must serve this many positions more than the live state to be swapped in
        uint32_t snap_margin = 128;         // #87: the prompt-end snapshot serves a divergence up to this far before the prompt's end
    };
    Mimo26PrefixCache() = default;
    ~Mimo26PrefixCache() { free_all(); }
    Mimo26PrefixCache(const Mimo26PrefixCache&) = delete;
    Mimo26PrefixCache& operator=(const Mimo26PrefixCache&) = delete;

    // Attach to an initialised forward and its drafter (or null). Host slots are on when o.budget > 0: a pinned bounce per
    // queue. Without host slots prepare() serves from the live state only (the P3b rule).
    std::string init(Mimo26Forward& fwd, Mimo26DFlash* df, const Options& o);
    void        free_all();   // the slots and the bounces; the queues must still be alive

    // The prefix step before a prompt runs. `live` = the ids the cards hold (positions [0, fwd.n_pos())). Serves `ids` from
    // the live state, or from a host slot that serves at least swap_margin positions more (the live state is kept in a slot
    // first, when worth it); before truncating away another conversation's prompt positions the live state is kept too.
    // On return the cards and the drafter hold a state whose first `reused` positions are ids[0, reused), and `live` holds
    // its ids: the caller rewinds / resets to `reused` exactly as for live reuse. `source`: "live", "slot K" or "none".
    // An error leaves the device state undefined: the caller resets the forward and the drafter and calls live_lost().
    std::string prepare(const std::vector<int32_t>& ids, std::vector<int32_t>& live, uint32_t& reused, std::string& source);
    // The prompt has run: the live state's positions [0, prompt_len) are prompt. A slot the request continued is dropped
    // (the live state supersedes it). #87: the prompt-end snapshot is taken (host slots on, prompt_len >= min_tokens,
    // mimo26_prompt_snapshot says one is needed). An error is a failed device-to-host copy or host allocation: the caller fails the request
    // and clears the live state as for a failed keep.
    std::string prompt_done(uint32_t prompt_len);
    // The live state was cleared (a failed forward): its bookkeeping and its snapshot are forgotten; every slot stays.
    void live_lost();

    struct Stats {
        uint32_t slots = 0; uint64_t bytes = 0; double last_save_ms = 0, last_load_ms = 0;
        uint64_t snap_bytes = 0; double last_snap_ms = 0, last_snap_load_ms = 0;   // #87: the valid snapshot's bytes (0 = none)
    };
    Stats stats() const { return {uint32_t(slots_.size()), held_, save_ms_, load_ms_, snap_.valid ? snap_.bytes : 0, snap_ms_, snap_load_ms_}; }
    bool  slots_on() const { return on_; }
    const Options& options() const { return opt_; }
    // Conversations of `max_ctx` tokens the byte budget holds at once, counting the live one: 1 when host slots are
    // off, else 1 + budget / one full-length state (KV spans, ids, a 64 MiB margin for the drafter ring). Shorter
    // conversations pack more; the MemAvailable floor and the one-shot rule can keep fewer. Served at /props as
    // "prompt_cache_slots".
    uint32_t guaranteed_slots(uint32_t max_ctx, bool with_drafter) const;

private:
    struct Slot {
        uint64_t id = 0, tick = 0, bytes = 0, state_bytes = 0;   // bytes = state_bytes + the ids
        uint32_t uses = 0, prompt_end = 0, written_end = 0;
        uint32_t df_lo = 0, df_end = 0, df_hi = 0;
        std::vector<int32_t> ids;
        std::unique_ptr<uint8_t[]> data;                          // the spans' bytes back to back: the cards', then the drafter's
    };
    struct Spans { sycl::queue* q = nullptr; std::vector<std::pair<void*, uint64_t>> v; };
    uint64_t    spans(uint32_t n, uint32_t hi, uint32_t df_lo, uint32_t df_end, std::vector<Spans>& out, bool rings_only = false) const;
    std::string copy(const std::vector<Spans>& groups, uint8_t* host, bool to_host);
    std::string keep_live(const std::vector<int32_t>& live, uint64_t keep_id, bool keep_leaves);
    std::string load(Slot& s, std::vector<int32_t>& live);
    std::string load_snapshot(std::vector<int32_t>& live);
    uint8_t*    bounce_for(const sycl::queue* q) const;

    Mimo26Forward* fwd_ = nullptr;
    Mimo26DFlash*  df_ = nullptr;
    Options        opt_;
    bool           on_ = false;
    std::vector<std::pair<sycl::queue*, uint8_t*>> bounce_;   // one pinned bounce per queue, in that queue's context
    std::vector<Slot> slots_;
    uint64_t serial_ = 0, tick_ = 0, held_ = 0;
    uint32_t live_prompt_ = 0, live_uses_ = 0;   // the live conversation: its last prompt's length, how often it was continued
    uint64_t drop_id_ = 0;                       // the slot this request continued (dropped at prompt_done)
    double   save_ms_ = 0, load_ms_ = 0;
    // #87: the live conversation's prompt-end snapshot -- valid while the cards still hold that prompt's rows [0, n), i.e.
    // until a cut below n, a swap, a lost live state or the next prompt's end (which replaces it)
    struct Snap {
        bool     valid = false;
        uint32_t n = 0, hi = 0;                       // the prompt's end; the written_end that describes the restored rings
        uint32_t df_lo = 0, df_end = 0, df_hi = 0;    // the drafter's bookkeeping at the prompt's end
        uint64_t bytes = 0;
        std::vector<uint8_t> data;                    // the rings' spans, then the drafter's; grows to the largest, then reused
    };
    Snap     snap_;
    double   snap_ms_ = 0, snap_load_ms_ = 0;
};

}  // namespace ie
