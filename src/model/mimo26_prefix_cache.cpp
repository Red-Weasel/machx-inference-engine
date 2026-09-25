// src/model/mimo26_prefix_cache.cpp — MiMo-V2.6 host slots (P7, fix-list #70). See include/ie/mimo26_prefix_cache.hpp and
// docs/mimo26/P7_FIX64_FIX70.md.
#include "ie/mimo26_prefix_cache.hpp"

#include "ie/mimo26_host_rules.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <new>
#include <tuple>

namespace ie {

namespace {

constexpr uint64_t kMimo26pcBounce = 64ull << 20;   // pinned bytes per queue for the slot copies (V4.1's bounce size)

uint64_t mimo26pc_mem_available() {   // 0 when unreadable: the floor then refuses every new slot
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) return 0;
    char line[256]; unsigned long long kb = 0; uint64_t out = 0;
    while (std::fgets(line, sizeof line, f))
        if (std::sscanf(line, "MemAvailable: %llu kB", &kb) == 1) { out = uint64_t(kb) * 1024ull; break; }
    std::fclose(f);
    return out;
}
double mimo26pc_gib(uint64_t b) { return double(b) / 1073741824.0; }
double mimo26pc_ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

std::string Mimo26PrefixCache::init(Mimo26Forward& fwd, Mimo26DFlash* df, const Options& o) {
    free_all();
    fwd_ = &fwd; df_ = df; opt_ = o;
    live_prompt_ = live_uses_ = 0;
    if (!o.budget) return {};
    auto add = [&](sycl::queue* q) -> std::string {
        if (!q) return "a card has no queue";
        if (bounce_for(q)) return {};
        uint8_t* b = sycl::malloc_host<uint8_t>(kMimo26pcBounce, *q);
        if (!b) return "the " + std::to_string(kMimo26pcBounce >> 20) + " MiB pinned bounce could not be allocated";
        bounce_.push_back({q, b});
        return {};
    };
    for (size_t c = 0; c < fwd.n_cards(); ++c)
        if (auto e = add(fwd.card_queue(c)); !e.empty()) { free_all(); return e; }
    if (df) if (auto e = add(df->queue()); !e.empty()) { free_all(); return e; }
    on_ = true;
    return {};
}

void Mimo26PrefixCache::free_all() {
    for (auto& [q, b] : bounce_) {
        if (!q || !b) continue;
        try { q->wait(); } catch (const std::exception& e) { std::fprintf(stderr, "[mimo26 cache] teardown drain: %s\n", e.what()); }
        sycl::free(b, *q);
    }
    bounce_.clear();
    slots_.clear(); held_ = 0; on_ = false; drop_id_ = 0;
    snap_ = Snap{};
}

uint8_t* Mimo26PrefixCache::bounce_for(const sycl::queue* q) const {
    for (const auto& [qq, b] : bounce_) if (qq == q) return b;
    return nullptr;
}

uint64_t Mimo26PrefixCache::spans(uint32_t n, uint32_t hi, uint32_t df_lo, uint32_t df_end, std::vector<Spans>& out, bool rings_only) const {
    out.clear();
    uint64_t total = 0;
    for (size_t c = 0; c < fwd_->n_cards(); ++c) {
        Spans g; g.q = fwd_->card_queue(c);
        fwd_->state_spans(c, n, hi, g.v, rings_only);
        for (const auto& s : g.v) total += s.second;
        out.push_back(std::move(g));
    }
    if (df_) {
        Spans g; g.q = df_->queue();
        df_->state_spans(df_lo, df_end, g.v);
        for (const auto& s : g.v) total += s.second;
        out.push_back(std::move(g));
    }
    return total;
}

// The groups' spans <-> `host` (their bytes back to back, in order), through each queue's pinned bounce: a bounce fill is
// consecutive pieces of the spans, so it maps to one contiguous range of `host` -- one host memcpy and one wait per fill.
std::string Mimo26PrefixCache::copy(const std::vector<Spans>& groups, uint8_t* host, bool to_host) {
    uint64_t at = 0;                                                  // host offset of the current fill
    std::vector<std::tuple<uint8_t*, uint64_t, uint64_t>> pieces;     // (device address, bounce offset, bytes)
    for (const auto& g : groups) {
        uint8_t* bb = bounce_for(g.q);
        if (!g.q || !bb) return "no pinned bounce for a queue";
        sycl::queue& q = *g.q;
        size_t i = 0; uint64_t in = 0;                                // the span, and the offset in it, the next piece starts at
        while (i < g.v.size()) {
            pieces.clear();
            uint64_t fill = 0;
            while (i < g.v.size() && fill < kMimo26pcBounce) {
                const uint64_t take = std::min(g.v[i].second - in, kMimo26pcBounce - fill);
                pieces.emplace_back(static_cast<uint8_t*>(g.v[i].first) + in, fill, take);
                fill += take; in += take;
                if (in == g.v[i].second) { ++i; in = 0; }
            }
            if (to_host) {
                for (const auto& [d, o, n] : pieces) q.memcpy(bb + o, d, n);
                q.wait_and_throw();
                std::memcpy(host + at, bb, fill);
            } else {
                std::memcpy(bb, host + at, fill);
                for (const auto& [d, o, n] : pieces) q.memcpy(d, bb + o, n);
                q.wait_and_throw();
            }
            at += fill;
        }
    }
    return {};
}

// The live conversation into a new slot -- when host slots are on, it is at least min_tokens long, and the admission
// policy finds room (evicting what it names). Refusals are logged with their reason and are not errors. Slot `keep_id` is
// the one about to be restored: never evicted, and with `keep_leaves` (it is dropped after the prompt) not counted against
// the budget (#78).
std::string Mimo26PrefixCache::keep_live(const std::vector<int32_t>& live, uint64_t keep_id, bool keep_leaves) {
    const uint32_t n = fwd_->n_pos();
    if (!on_ || n < opt_.min_tokens || live.size() != n) return {};
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t hi = fwd_->written_end();
    const uint32_t dlo = df_ ? df_->ctx_lo() : 0u, dend = df_ ? df_->ctx_end() : 0u, dhi = df_ ? df_->ctx_hi() : 0u;
    std::vector<Spans> sp;
    const uint64_t state = spans(n, hi, dlo, dend, sp);
    const uint64_t bytes = state + uint64_t(n) * sizeof(int32_t);
    const char* kind = live_uses_ ? "continued" : "one-shot";
    std::vector<Mimo26SlotCost> costs;
    int keep = -1;
    for (size_t i = 0; i < slots_.size(); ++i) {
        costs.push_back({slots_[i].bytes, slots_[i].tick, slots_[i].uses, uint32_t(slots_[i].ids.size())});
        if (slots_[i].id == keep_id) keep = int(i);
    }
    const uint64_t avail = mimo26pc_mem_available();
    std::vector<size_t> victims;
    const Mimo26Admit adm = mimo26_plan_admit(costs, bytes, live_uses_, opt_.budget, avail, opt_.keep_free, keep, victims, keep_leaves);
    if (adm != Mimo26Admit::kOk) {
        char why[160];
        if (adm == Mimo26Admit::kTooLarge) std::snprintf(why, sizeof why, "larger than the whole %.0f GiB budget", mimo26pc_gib(opt_.budget));
        else if (adm == Mimo26Admit::kBudget) std::snprintf(why, sizeof why, "the %.0f GiB budget is held by conversations a %s state may not evict", mimo26pc_gib(opt_.budget), kind);
        else std::snprintf(why, sizeof why, "MemAvailable %.1f GiB would fall under the %.0f GiB floor", mimo26pc_gib(avail), mimo26pc_gib(opt_.keep_free));
        std::fprintf(stderr, "[mimo26 cache] not keeping the live conversation (%u tokens, %s, %.2f GiB): %s -- it is re-read if it "
                             "returns (IE_MIMO26_PROMPT_CACHE_GIB, IE_MIMO26_CACHE_KEEP_FREE_GIB)\n", n, kind, mimo26pc_gib(bytes), why);
        return {};
    }
    // the new slot's memory first: a failed allocation then evicts nothing. It is an mmap reservation until the copy
    // touches it, so it takes no MemAvailable before the victims below have given theirs back
    Slot s;
    s.data.reset(new (std::nothrow) uint8_t[state]);
    if (!s.data) {
        std::fprintf(stderr, "[mimo26 cache] not keeping the live conversation (%u tokens): %.2f GiB of host memory could not be allocated\n",
                     n, mimo26pc_gib(bytes));
        return {};
    }
    std::sort(victims.begin(), victims.end(), std::greater<size_t>());   // erase from the back: the other indices stay put
    for (size_t v : victims) {
        const Slot& ev = slots_[v];
        std::fprintf(stderr, "[mimo26 cache] evicted slot %llu (%zu tokens, %s, %.2f GiB) for the live conversation\n",
                     (unsigned long long)ev.id, ev.ids.size(), ev.uses ? "continued" : "one-shot", mimo26pc_gib(ev.bytes));
        held_ -= ev.bytes;
        slots_.erase(slots_.begin() + std::ptrdiff_t(v));
    }
    if (auto e = copy(sp, s.data.get(), true); !e.empty()) return e;
    s.id = ++serial_; s.tick = ++tick_; s.bytes = bytes; s.state_bytes = state;
    s.uses = live_uses_; s.prompt_end = std::min(live_prompt_, n); s.written_end = hi;
    s.df_lo = dlo; s.df_end = dend; s.df_hi = dhi;
    s.ids = live;
    held_ += bytes;
    save_ms_ = mimo26pc_ms(t0);
    std::fprintf(stderr, "[mimo26 cache] kept the live conversation as slot %llu: %u tokens (%s), %.2f GiB in %.0f ms; %zu slot(s), %.2f of %.0f GiB\n",
                 (unsigned long long)s.id, n, kind, mimo26pc_gib(bytes), save_ms_, slots_.size() + 1, mimo26pc_gib(held_), mimo26pc_gib(opt_.budget));
    slots_.push_back(std::move(s));
    return {};
}

// A slot's bytes back onto the cards and into the drafter; `live` = its ids. The slot stays in the store.
std::string Mimo26PrefixCache::load(Slot& s, std::vector<int32_t>& live) {
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t n = uint32_t(s.ids.size());
    std::vector<Spans> sp;
    const uint64_t state = spans(n, s.written_end, s.df_lo, s.df_end, sp);
    if (state != s.state_bytes)
        return "slot " + std::to_string(s.id) + " holds " + std::to_string(s.state_bytes) + " bytes but its spans take " + std::to_string(state);
    if (auto e = copy(sp, s.data.get(), false); !e.empty()) return e;
    if (auto e = fwd_->set_state(n, s.written_end); !e.empty()) return e;
    if (df_) if (auto e = df_->set_state(s.df_lo, s.df_end, s.df_hi); !e.empty()) return e;
    live = s.ids;
    load_ms_ = mimo26pc_ms(t0);
    return {};
}

// #87: the prompt-end snapshot back onto the cards and into the drafter; `live` = the snapshot's prompt. The full layers'
// rows [0, snap_.n) never left the cards (the reply's decode wrote only past them); the rings' slots of [hi - ring, n) and
// the drafter's context are the saved bytes, so every later forward and draft reads what it read at the prompt's end.
std::string Mimo26PrefixCache::load_snapshot(std::vector<int32_t>& live) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<Spans> sp;
    const uint64_t bytes = spans(snap_.n, snap_.hi, snap_.df_lo, snap_.df_end, sp, true);
    if (bytes != snap_.bytes)
        return "the prompt-end snapshot holds " + std::to_string(snap_.bytes) + " bytes but its spans take " + std::to_string(bytes);
    if (auto e = copy(sp, snap_.data.data(), false); !e.empty()) return e;
    if (auto e = fwd_->set_state(snap_.n, snap_.hi); !e.empty()) return e;
    if (df_) if (auto e = df_->set_state(snap_.df_lo, snap_.df_end, snap_.df_hi); !e.empty()) return e;
    live.resize(snap_.n);
    snap_load_ms_ = mimo26pc_ms(t0);
    return {};
}

std::string Mimo26PrefixCache::prepare(const std::vector<int32_t>& ids, std::vector<int32_t>& live, uint32_t& reused, std::string& source) {
    reused = 0; source = "none"; drop_id_ = 0;
    if (!fwd_) return "the prefix cache is not attached to a forward";
    const uint32_t ring = fwd_->ring(), win = fwd_->window();
    const uint32_t L_live = mimo26_servable(live, fwd_->written_end(), ring, win, ids);
    // #87: what the live conversation's prompt-end snapshot serves -- its prompt, under the rings it restores
    uint32_t L_snap = 0;
    if (snap_.valid && live.size() >= snap_.n) {
        const std::vector<int32_t> have(live.begin(), live.begin() + std::ptrdiff_t(snap_.n));
        L_snap = mimo26_servable(have, snap_.hi, ring, win, ids);
    }
    const uint32_t L_here = std::max(L_live, L_snap);
    int best = -1; uint32_t L_slot = 0;
    for (size_t i = 0; i < slots_.size(); ++i) {
        const uint32_t l = mimo26_servable(slots_[i].ids, slots_[i].written_end, ring, win, ids);
        if (l > L_slot) { L_slot = l; best = int(i); }
    }
    try {
        if (best >= 0 && L_slot >= L_here + opt_.swap_margin) {
            // another conversation's slot serves clearly more: keep the live one first (never evicting the chosen slot; #78:
            // a slot restored for a continuation is dropped after the prompt, so it does not count against the budget)
            const uint64_t sid = slots_[size_t(best)].id;
            const bool cont = mimo26_continues(L_slot, slots_[size_t(best)].prompt_end, opt_.min_tokens);
            if (auto e = keep_live(live, sid, cont); !e.empty()) return e;
            snap_.valid = false;   // the live conversation leaves the cards
            Slot* s = nullptr;
            for (auto& x : slots_) if (x.id == sid) s = &x;
            if (!s) return "internal: slot " + std::to_string(sid) + " vanished while the live conversation was kept";
            if (auto e = load(*s, live); !e.empty()) return e;
            live_uses_ = cont ? s->uses + 1 : 0;
            live_prompt_ = s->prompt_end;
            s->tick = ++tick_;
            if (cont) drop_id_ = sid;   // dropped once the prompt has run: the live state then holds all it served
            reused = L_slot; source = "slot " + std::to_string(sid);
            std::fprintf(stderr, "[mimo26 cache] restored slot %llu: %zu tokens, %.2f GiB in %.0f ms -- it serves %u of the prompt's %zu tokens "
                                 "(the live state %u)%s\n", (unsigned long long)sid, s->ids.size(), mimo26pc_gib(s->bytes), load_ms_, L_slot, ids.size(), L_live,
                         cont ? "" : "; a branch of it: the slot stays");
            return {};
        }
        if (L_here > 0) {
            // a request that re-reads more than min_tokens of the live conversation's prompt is another conversation
            // branching off it: keep the live one before those prompt positions are cut
            const bool cont = mimo26_continues(L_here, live_prompt_, opt_.min_tokens);
            if (!cont) if (auto e = keep_live(live, 0, false); !e.empty()) return e;
            source = "live";
            if (L_snap > L_live) {
                // #87: the reply overwrote the rings the prompt's end needs; the snapshot has them
                const uint32_t n_was = fwd_->n_pos(), hi_was = fwd_->written_end();
                if (auto e = load_snapshot(live); !e.empty()) return e;
                source = "prompt end";
                std::fprintf(stderr, "[mimo26 cache] restored the prompt-end snapshot: %.1f MiB in %.0f ms -- it serves %u of the prompt's %zu tokens "
                                     "(the live state %u: %u positions written to %u)\n", double(snap_.bytes) / 1048576.0, snap_load_ms_, L_snap, ids.size(),
                             L_live, n_was, hi_was);
            }
            if (L_here < snap_.n) snap_.valid = false;   // the cut drops rows of the snapshot's prompt
            live_uses_ = cont ? live_uses_ + 1 : 0;
            reused = L_here;
            return {};
        }
        // nothing serves the prompt: keep the live conversation when it is worth it, the prefill starts over
        if (auto e = keep_live(live, 0, false); !e.empty()) return e;
        snap_.valid = false;
        live_uses_ = 0;
        return {};
    } catch (const std::exception& e) {
        return std::string("sycl: ") + e.what();
    }
}

std::string Mimo26PrefixCache::prompt_done(uint32_t prompt_len) {
    live_prompt_ = prompt_len;
    if (drop_id_) {
        for (size_t i = 0; i < slots_.size(); ++i)
            if (slots_[i].id == drop_id_) {
                std::fprintf(stderr, "[mimo26 cache] slot %llu is live again: dropped (%.2f GiB freed)\n", (unsigned long long)drop_id_, mimo26pc_gib(slots_[i].bytes));
                held_ -= slots_[i].bytes;
                slots_.erase(slots_.begin() + std::ptrdiff_t(i));
                break;
            }
        drop_id_ = 0;
    }
    // #87: the prompt-end snapshot -- the rings' slots of [s0, prompt_len) and the drafter's context, before any decode
    snap_.valid = false;
    uint32_t s0 = 0, hs = 0;
    if (!on_ || prompt_len < opt_.min_tokens || fwd_->n_pos() != prompt_len ||
        !mimo26_prompt_snapshot(prompt_len, fwd_->written_end(), fwd_->ring(), fwd_->window(), opt_.snap_margin, fwd_->capacity(), s0, hs))
        return {};
    const auto t0 = std::chrono::steady_clock::now();
    const uint32_t dlo = df_ ? df_->ctx_lo() : 0u, dend = df_ ? df_->ctx_end() : 0u, dhi = df_ ? df_->ctx_hi() : 0u;
    std::vector<Spans> sp;
    const uint64_t bytes = spans(prompt_len, hs, dlo, dend, sp, true);
    try {
        if (snap_.data.size() < bytes) snap_.data.resize(bytes);   // a bad_alloc takes the failed-copy path below
        if (auto e = copy(sp, snap_.data.data(), true); !e.empty()) return e;
    } catch (const std::bad_alloc&) {
        return "the prompt-end snapshot's " + std::to_string(bytes >> 20) + " MiB of host memory could not be allocated";
    } catch (const std::exception& e) {
        return std::string("sycl: ") + e.what();
    }
    snap_.n = prompt_len; snap_.hi = hs; snap_.df_lo = dlo; snap_.df_end = dend; snap_.df_hi = dhi; snap_.bytes = bytes;
    snap_.valid = true;
    snap_ms_ = mimo26pc_ms(t0);
    return {};
}

void Mimo26PrefixCache::live_lost() {
    live_prompt_ = live_uses_ = 0;
    drop_id_ = 0;
    snap_.valid = false;
}

uint32_t Mimo26PrefixCache::guaranteed_slots(uint32_t max_ctx, bool with_drafter) const {
    if (!on_ || !fwd_ || max_ctx == 0) return 1;
    std::vector<Spans> tmp;
    // the KV spans, the token ids (4 B each) and a 64 MiB margin for the drafter ring
    const uint64_t per = spans(max_ctx, max_ctx, 0, 0, tmp) + uint64_t(max_ctx) * 4 + (with_drafter ? (64ull << 20) : 0);
    return per ? 1 + uint32_t(std::min<uint64_t>(opt_.budget / per, 1u << 20)) : 1;
}

}  // namespace ie
