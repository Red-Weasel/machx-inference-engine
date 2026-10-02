// tests/unit/q35m_lanes_test.cpp -- host-only (no GPU, no SYCL): the crown split's request lanes (P4 B10,
// include/ie/q35m_lanes.hpp, docs/lanes/LANES_SERVE.md) -- the per-lane VRAM arithmetic, the prefill chunk rule, the
// plan against --parallel 1's prefill loop, and the crown's hook rules (a prompt cache SHARED by every lane, restored into the
// chosen lane) served by the shared lanes module (ie/lanes_serve.hpp) with a fake two-stage model.
//
// The fake: each lane keeps, per stage, the ids it has "forwarded" (a step must start at the stage's own depth), and stage 1's
// "logits" hash every id AND every step's start position -- so a reply depends on how its prompt was cut into pieces, like the
// real model's rounding does. A serial reference (Engine::generate's crown-split path: restore, prefill_to(snap_at), insert,
// prefill_to(T), decode one row a step) run over the same request sequence predicts every reply; concurrent lanes, the serial
// turn, pieces through the pipe beside decoding lanes, the shared cache and a cancel must all leave each request == it.
// Does NOT link ie_core.
#undef NDEBUG
#include "ie/q35m_lanes.hpp"
#include "ie/q4e_lanes.hpp"   // q4e_lane_ctx (the --slot-ctx rule the crown shares)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("[%s] %s\n", ok ? " ok " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

constexpr int32_t kStop = 7;
constexpr uint32_t kChunk = 5;   // the fake's prefill chunk (pf_chunk)

uint64_t mix(uint64_t h, uint64_t v) { return (h ^ v) * 0x100000001B3ull; }
constexpr uint64_t kSeed = 0xcbf29ce484222325ull;
uint32_t g_stop_at = 0;   // the depth at which the "model" emits kStop (0 = never)
int32_t  g_stop_first = -1;   // (P4 B39) ... only for a sequence whose first id is this (-1 = any)
bool stops_at(const std::vector<int32_t>& seq) { return g_stop_at && seq.size() == g_stop_at && (g_stop_first < 0 || (!seq.empty() && seq[0] == g_stop_first)); }
int32_t id_of(uint64_t h, uint64_t seed) { return int32_t(10 + ((h ^ (seed * 0x9E3779B97F4A7C15ull)) >> 11) % 990); }
// P4 B39: the crown's IE_Q35MOE_TURN_DRAIN=1 in every FakeCrown made while this is set -- every serial turn drains the pipe and
// the plan's tail runs in prompt_end (B10-B38); off = the default: the turns run beside the pipe, the tail through it
bool g_turn_drain = false;
// P4 B42: the crown's decode regroup (IE_Q35MOE_GROUP_WAIT_US) in every FakeCrown made while this is set; 0 = IE_Q35MOE_REGROUP=0
uint32_t g_regroup_us = 20000;

// Stage 1's state after one step: a marker for the step's start, then its ids.
uint64_t step_hash(uint64_t h, const int32_t* ids, uint32_t T, uint32_t pos0) {
    if (pos0 == 0) h = kSeed;
    h = mix(h, 0xC0000000ull | pos0);
    for (uint32_t i = 0; i < T; ++i) h = mix(h, uint64_t(uint32_t(ids[i])));
    return h;
}

// The shared prompt cache (FleetPrefixCache + fleet_cache_restore's rule): endpoints (tokens, state); the longest endpoint
// that is a prefix of the prompt, used only when it leaves >= 1 row (M == match_len <= T - 1; the DeltaNet rule).
// P4 B29: with cap > 0, FleetPrefixCache's entry cap and LRU (a lookup refreshes the deepest endpoint along the prompt; an
// insert at capacity evicts the least recently used), and with supersede its rule (an insert that is not a shared prefix
// drops the non-shared endpoints that are strict prefixes of it: that conversation's earlier turns).
struct Cache {
    struct Ep { std::vector<int32_t> toks; uint64_t h; uint64_t tick = 0; bool shared = false, anchor = false; };
    mutable std::vector<Ep> eps;
    mutable uint64_t tick = 0;
    uint32_t cap = 0;          // 0 = unbounded (the references)
    bool supersede = false;
    // P4 B36 (B): a lookup also refreshes the anchors along the prompt (FleetPrefixCache::find_longest_match), shallow first
    uint32_t restore(const std::vector<int32_t>& ids, uint64_t& h, bool touch = false) const {
        Ep* best = nullptr;
        std::vector<Ep*> on_path;
        for (auto& e : eps)
            if (e.toks.size() <= ids.size() && std::equal(e.toks.begin(), e.toks.end(), ids.begin())) {
                on_path.push_back(&e);
                if (!best || e.toks.size() > best->toks.size()) best = &e;
            }
        if (touch) {
            std::sort(on_path.begin(), on_path.end(), [](const Ep* a, const Ep* b) { return a->toks.size() < b->toks.size(); });
            for (Ep* e : on_path) if (e->anchor) e->tick = ++tick;
        }
        if (best && touch) best->tick = ++tick;
        if (!best || best->toks.size() > ids.size() - 1) return 0;
        h = best->h;
        return uint32_t(best->toks.size());
    }
    // P4 B36 (B): an anchor is kept through a regular insert's supersede, and a new anchor supersedes the older ones on its path
    // (one anchor per conversation: its last query's end)
    void insert(const std::vector<int32_t>& toks, uint64_t h, bool shared = false, bool anchor = false) {
        for (auto& e : eps) if (e.toks == toks) { e.tick = ++tick; e.shared = e.shared || shared; e.anchor = e.anchor || anchor; return; }
        if (supersede && !shared)
            for (size_t i = 0; i < eps.size();)
                if (!eps[i].shared && (anchor || !eps[i].anchor) && eps[i].toks.size() < toks.size() &&
                    std::equal(eps[i].toks.begin(), eps[i].toks.end(), toks.begin()))
                    eps.erase(eps.begin() + long(i));
                else ++i;
        while (cap && eps.size() >= cap) {
            size_t v = 0;
            for (size_t i = 1; i < eps.size(); ++i) if (eps[i].tick < eps[v].tick) v = i;
            eps.erase(eps.begin() + long(v));
        }
        eps.push_back({toks, h, ++tick, shared, anchor});
    }
};

// --parallel 1 on the crown split (Engine::generate): what a request gives, given the cache state before it.
// reply_cache: its gen-cache insert at prompt ++ reply after a "stop"/"length" reply (every sampled id was forwarded).
std::vector<int32_t> serial_ref(Cache& c, const std::vector<int32_t>& ids, uint32_t snap_at, uint32_t n, uint64_t rng,
                                uint32_t* restored_out = nullptr, bool reply_cache = false, uint32_t share_at = 0, bool anchor = false) {
    uint64_t h = kSeed;
    const uint32_t T = uint32_t(ids.size());
    uint32_t pos = c.restore(ids, h);
    if (restored_out) *restored_out = pos;
    const uint32_t restored = pos;
    auto prefill_to = [&](uint32_t end) {
        while (pos < end) { const uint32_t k = std::min(kChunk, end - pos); h = step_hash(h, ids.data() + pos, k, pos); pos += k; }
    };
    if (share_at > restored && share_at < snap_at) {   // P4 B15: --parallel 1's split + insert at the shared prefix
        prefill_to(share_at);
        c.insert(std::vector<int32_t>(ids.begin(), ids.begin() + share_at), h);
    }
    prefill_to(snap_at);
    if (snap_at > restored) c.insert(std::vector<int32_t>(ids.begin(), ids.begin() + snap_at), h, false, anchor);
    prefill_to(T);
    std::vector<int32_t> out;
    for (uint32_t step = 0; step < n; ++step) {
        const int32_t id = g_stop_at && pos == g_stop_at && (g_stop_first < 0 || ids[0] == g_stop_first) ? kStop : id_of(h, rng + step);
        if (id == kStop) break;
        out.push_back(id);
        h = step_hash(h, &id, 1, pos); ++pos;
    }
    if (reply_cache && !out.empty()) {
        std::vector<int32_t> full(ids);
        full.insert(full.end(), out.begin(), out.end());
        c.insert(full, h);
    }
    return out;
}

std::string text_of(const std::vector<int32_t>& ids) {
    std::string s;
    for (int32_t id : ids) s += std::to_string(id) + ",";
    return s;
}

// P4 B34: the lane pipeline in every FakeCrown made while this is set (Q35mLanesModel's IE_Q35MOE_LANE_PIPELINE), and the
// lookaheads their pipes took
bool g_pipeline = false;
std::atomic<uint64_t> g_lookaheads{0};

// The crown's hooks (Q35mLanesModel in src/engine/engine.cpp) over fake state.
struct FakeCrown final : ie::LanesModel {
    struct St { std::vector<int32_t> seq[2]; uint64_t h = kSeed, kept = 0; };
    std::vector<St> st;
    std::vector<uint32_t> cap;
    Cache cache;                   // touched only in the serial turn (prefix_prepare, prompt_end)
    std::atomic<uint32_t> bad{0};
    std::atomic<int> active{0}, max_active{0};
    ie::CardPipe pipe;
    // P4 B14 rows mode (Q35mLanesModel with rows on): the pipe groups the decoding lanes' 1-row steps
    bool rows_on = false;
    std::atomic<uint32_t> rows_calls{0}, rows_ids{0}, group_steps{0}, kept_in_group{0};
    // P4 B29 Fix B (Q35mLanesModel's sticky lanes): the shared policy + the fake's "DeltaNet checkpoint" (the hash at it)
    ie::Q35mSticky sticky;
    std::vector<uint64_t> ckh;
    std::atomic<uint32_t> own_restores{0}, cache_restores{0};
    std::atomic<bool> in_group_cb{false};

    explicit FakeCrown(std::vector<uint32_t> caps, bool rows = false)
        : st(caps.size()), cap(caps), ckh(caps.size(), 0),
          pipe(uint32_t(caps.size()), kChunk, 1, 2, [this](uint32_t s, const ie::Glm5LanePipe::Step& x) { return stage(s, x); },
               [this](uint32_t s, std::span<const ie::Glm5LanePipe::Step> x, float*) { return stage_rows(s, x); }, 16, 0, g_pipeline),
          rows_on(rows) { sticky.resize(uint32_t(caps.size())); pipe.set_regroup(g_regroup_us); }
    // P4 B42: while record_groups, every decode group stage 0 ran, by size, in order (a lone 1-row step counts as a group of 1)
    std::atomic<bool> record_groups{false};
    std::vector<uint32_t> gsz;
    void rec_group(uint32_t n) {
        if (!record_groups) return;
        std::lock_guard<std::mutex> g(rec_mu);
        gsz.push_back(n);
    }
    // P4 B34 (Q35mLanesModel::pipe_lookahead / pipe_submit_ahead)
    bool pipe_lookahead(std::function<void()> idle) override {
        if (!g_pipeline) return false;
        pipe.set_idle(std::move(idle));
        return true;
    }
    std::string pipe_submit_ahead(uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0, bool& taken) override {
        std::string e = pipe.submit_ahead(lane, ids, T, pos0, taken);
        if (taken) { ++g_lookaheads; rec('S', lane, pos0, T); }   // (P4 B36 (C): still under the serve's lock, where it decided)
        return e;
    }

    // P4 B39: the serial hooks take turn_ms (a restore's / snapshot's device copies), and while record_dec every decode row's
    // stage-0 start time per lane (a drained turn shows as a gap in a decoder's steps)
    std::atomic<int> turn_ms{0};
    std::atomic<bool> record_dec{false};
    std::vector<std::vector<double>> dec_t0 = std::vector<std::vector<double>>(64);
    const std::chrono::steady_clock::time_point t_base = std::chrono::steady_clock::now();
    double now_ms() const { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_base).count(); }
    void rec_dec(uint32_t lane) {
        if (!record_dec || lane >= dec_t0.size()) return;
        std::lock_guard<std::mutex> g(rec_mu);
        dec_t0[lane].push_back(now_ms());
    }
    void turn_sleep() { if (const int t = turn_ms.load()) std::this_thread::sleep_for(std::chrono::milliseconds(t)); }
    // (B39 follow-up) the serial hooks' order in time while record_turns: 'P' prefix_prepare begins, 'M' mark, 'S' snapshot, and
    // 'L' = stage 1 forwarded a lane's piece ending at land_at (the piece whose landing makes the lane wait for its mark turn)
    std::atomic<bool> record_turns{false};
    std::atomic<uint32_t> land_at{0};
    std::vector<std::tuple<char, uint32_t, double>> turns;
    void rec_turn(char k, uint32_t lane) {
        if (!record_turns) return;
        std::lock_guard<std::mutex> g(rec_mu);
        turns.emplace_back(k, lane, now_ms());
    }

    std::string stage_rows(uint32_t s, std::span<const ie::Glm5LanePipe::Step> x) {
        if (s == 0) { for (const auto& step : x) rec_dec(step.lane); rec_group(uint32_t(x.size())); }
        const int a = ++active;
        for (int m = max_active.load(); a > m && !max_active.compare_exchange_weak(m, a);) {}
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (!x.empty()) dec_sleep(x[0].lane, x[0].pos0);
        --active;
        if (s == 0) for (const auto& step : x) rec0(step.lane, step.pos0, 1);
        if (s == 1) ++group_steps;
        for (const auto& step : x)
            if (auto e = forward(step.lane, s, step.ids, 1, step.pos0); !e.empty()) return e;
        return {};
    }
    bool rows() const override { return rows_on; }
    std::string pipe_start_rows(DoneFn d, RowsDoneFn rd) override {
        return pipe.start([this, d](uint32_t lane) { d(lane); rec('L', lane, 0, 0); },
                          [this, rd](std::span<const uint32_t> ls) {
                              in_group_cb = true; rd(ls); in_group_cb = false;
                              for (uint32_t lane : ls) rec('L', lane, 0, 0);
                          });
    }
    std::string sample_rows(std::span<const uint32_t> lanes, std::span<const ie::LanesSampling* const> sp,
                            std::span<const std::span<const int32_t>> windows, std::span<const uint64_t> seeds,
                            std::span<int32_t> picks) override {
        ++rows_calls; rows_ids += uint32_t(lanes.size());
        return ie::LanesModel::sample_rows(lanes, sp, windows, seeds, picks);
    }

    // P4 B36: while `record`, every step stage 1 commits per lane, (pos0, T) in order (the pieces that made its hash), and
    // stage 0's run order (lane, pos0, T) -- a decode row is T 1 past the lane's prompt
    std::atomic<bool> record{false};
    std::mutex rec_mu;
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> pieces;
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> s0_log;
    // P4 B36 (C): and in ONE order (kind, lane, pos0, T): stage 0's starts '0', stage 1's commits '1', the serve's submits 'S'
    // (made under its lock, where it decides) and the done callbacks' ends 'L' (on stage 1's thread, as its commits)
    std::vector<std::tuple<char, uint32_t, uint32_t, uint32_t>> ev;
    void rec(char k, uint32_t lane, uint32_t pos0, uint32_t T) {
        if (!record) return;
        std::lock_guard<std::mutex> g(rec_mu);
        ev.emplace_back(k, lane, pos0, T);
    }
    void rec0(uint32_t lane, uint32_t pos0, uint32_t T) {
        if (!record) return;
        std::lock_guard<std::mutex> g(rec_mu);
        s0_log.emplace_back(lane, pos0, T);
        ev.emplace_back('0', lane, pos0, T);
    }
    // (P4 B36 (C)) a decode row's step takes dec_ms more a stage (stalled decoders); plen = each lane's prompt length
    std::atomic<int> dec_ms{0};
    std::vector<uint32_t> plen = std::vector<uint32_t>(64, UINT32_MAX);
    void dec_sleep(uint32_t lane, uint32_t pos0) {
        if (const int dm = dec_ms.load(); dm && lane < plen.size() && pos0 >= plen[lane]) std::this_thread::sleep_for(std::chrono::milliseconds(dm));
    }
    std::string forward(uint32_t lane, uint32_t s, const int32_t* ids, uint32_t T, uint32_t pos0) {
        St& L = st[lane];
        if (pos0 == 0) L.seq[s].clear();
        if (L.seq[s].size() != pos0) { ++bad; return "stage " + std::to_string(s) + " at " + std::to_string(L.seq[s].size()) + ", step at " + std::to_string(pos0); }
        L.seq[s].insert(L.seq[s].end(), ids, ids + T);
        if (s == 1) L.h = step_hash(L.h, ids, T, pos0);
        if (s == 1 && land_at && pos0 + T == land_at) rec_turn('L', lane);
        if (s == 1 && record) {
            std::lock_guard<std::mutex> g(rec_mu);
            if (pieces.size() < st.size()) pieces.resize(st.size());
            pieces[lane].emplace_back(pos0, T);
            ev.emplace_back('1', lane, pos0, T);
        }
        return {};
    }
    std::atomic<uint32_t> max_piece{0};   // P4 B29: the largest pipe prefill piece while `watch` (test_mix_chunk)
    std::atomic<bool> watch{false};
    // P4 B34 (3): a prefill piece at depth >= deep_from takes deep_ms more 1-ms "layers" a stage (a deep lead's pieces), each
    // checking abort(): the crown's request_abort stops a running piece at a layer boundary
    std::atomic<uint32_t> deep_from{UINT32_MAX};
    std::atomic<int> deep_ms{0};
    std::atomic<int> deep_row_ms{0};   // (P4 B36) and deep_row_ms more a row: a piece's time grows with its rows
    std::atomic<bool> aborted{false};
    std::atomic<uint32_t> calls_after_abort{0};
    uint32_t abort() override { const uint32_t h = pipe.cancel(); aborted = true; return h; }
    int deep_cost(uint32_t T, uint32_t pos0) const { return T > 1 && pos0 >= deep_from.load() ? deep_ms.load() + deep_row_ms.load() * int(T) : 0; }
    std::atomic<int> slow_lane{-1}, slow_ms{0};   // (B39 (5)) this lane's stage-0 steps take slow_ms more (a step that holds stage 0)
    std::string stage(uint32_t s, const ie::Glm5LanePipe::Step& x) {
        if (aborted) ++calls_after_abort;
        if (s == 0 && x.T == 1) { rec_dec(x.lane); if (x.pos0) rec_group(1); }
        if (s == 0) rec0(x.lane, x.pos0, x.T);
        if (s == 0 && int(x.lane) == slow_lane.load()) std::this_thread::sleep_for(std::chrono::milliseconds(slow_ms.load()));
        if (watch && x.T > 1) for (uint32_t v = max_piece; x.T > v && !max_piece.compare_exchange_weak(v, x.T);) {}
        const int a = ++active;
        for (int m = max_active.load(); a > m && !max_active.compare_exchange_weak(m, a);) {}
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (x.T == 1) dec_sleep(x.lane, x.pos0);
        --active;
        for (int i = 0, n = deep_cost(x.T, x.pos0); i < n; ++i) {
            if (aborted) return "stopped at a layer boundary";
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return forward(x.lane, s, x.ids, x.T, x.pos0);
    }
    const char* tag() const override { return "fake crown lanes"; }
    uint32_t n_lanes() const override { return uint32_t(st.size()); }
    uint32_t lane_cap(uint32_t lane) const override { return cap[lane]; }
    uint32_t own_match(uint32_t lane, std::span<const int32_t> ids) const override { return sticky.match(lane, ids); }
    bool occupied(uint32_t lane) const override { return sticky.end(lane) > 0; }
    uint32_t last_end(uint32_t lane) const override { return sticky.end(lane); }
    std::string prefix_prepare(uint32_t lane, const ie::LanesRequest& rq, uint32_t& reused, std::string& source) override {
        St& L = st[lane];
        source.clear();
        rec_turn('P', lane);
        turn_sleep();
        if (lane < plen.size()) plen[lane] = uint32_t(rq.ids->size());   // (P4 B36 (C): dec_sleep's decode rows)
        // Fix B (Q35mLanesModel::prefix_prepare): the lane's own checkpoint when it serves at least as much as the cache
        const uint32_t own = sticky.match(lane, *rq.ids);
        uint64_t hc = kSeed;
        if (ie::Q35mSticky::own_wins(own, cache.restore(*rq.ids, hc))) {
            for (int s = 0; s < 2; ++s)   // the lane's KV rows below the depth must be this prompt's (no one wrote them since)
                if (L.seq[s].size() < own || !std::equal(L.seq[s].begin(), L.seq[s].begin() + own, rq.ids->begin())) ++bad;
            L.seq[0].resize(own); L.seq[1].resize(own);   // every layer's KV length = the depth
            L.h = ckh[lane];                               // the DeltaNet + conv checkpoint copied back
            reused = own; ++own_restores;
            return {};
        }
        sticky.drop(lane);   // the lane's rows are about to be overwritten
        uint64_t h = kSeed;
        reused = cache.restore(*rq.ids, h, /*touch=*/true);
        cache_restores += reused > 0;
        L.seq[0].assign(rq.ids->begin(), rq.ids->begin() + reused);
        L.seq[1] = L.seq[0];
        L.h = reused ? h : kSeed;
        return {};
    }
    // P4 B39 (Q35mLanesModel::plan): the tail through the pipe behind the snapshot boundary unless the drain switch is on
    ie::LanesPlan plan(const ie::LanesRequest& rq, uint32_t reused) override {
        return ie::q35m_plan(uint32_t(rq.ids->size()), rq.snap_at, reused, kChunk, rq.share_at, /*pipe_tail=*/!g_turn_drain);
    }
    // P4 B39 (Q35mLanesModel::serial_drains): with the switch every turn drains; else only a "length" finish (it forwards the
    // last id: a full forward) and a lost lane's reset
    bool serial_drains(Turn k) const override { return g_turn_drain || k == Turn::kFinishLength || k == Turn::kReset; }
    std::string pipe_wait_lane(uint32_t lane) override { return pipe.wait_lane(lane); }
    // P4 B39 (Q35mLanesModel::snapshot): the conversation snapshot at `pos` -- prompt_end's insert + the lane's checkpoint
    std::atomic<int> snap_turns{0};
    std::string conv_snapshot(uint32_t lane, const ie::LanesRequest& rq, uint32_t pos) {
        St& L = st[lane];
        if (L.seq[0].size() != pos || L.seq[1].size() != pos) { ++bad; return "snapshot: the lane is not at " + std::to_string(pos); }
        cache.insert(std::vector<int32_t>(rq.ids->begin(), rq.ids->begin() + pos), L.h, false, anchor_on && rq.anchor);
        ++snaps;
        if (sticky.on) { ckh[lane] = L.h; sticky.set(lane, *rq.ids, pos); }   // Fix B
        return {};
    }
    std::string snapshot(uint32_t lane, const ie::LanesRequest& rq, uint32_t pos) override {
        ++snap_turns;
        rec_turn('S', lane);
        turn_sleep();
        return conv_snapshot(lane, rq, pos);
    }
    // P4 B15 (Q35mLanesModel::mark / cache_peek): the lane's state at the shared prefix into the shared cache
    std::atomic<int> marks{0};
    std::string mark(uint32_t lane, const ie::LanesRequest& rq, uint32_t pos) override {
        ++marks;
        rec_turn('M', lane);
        turn_sleep();
        St& L = st[lane];
        if (L.seq[0].size() != pos || L.seq[1].size() != pos) { ++bad; return "mark: the lane is not at " + std::to_string(pos); }
        cache.insert(std::vector<int32_t>(rq.ids->begin(), rq.ids->begin() + pos), L.h, /*shared=*/true);
        return {};
    }
    uint32_t cache_peek(const ie::LanesRequest& rq) const override { uint64_t h = 0; return cache.restore(*rq.ids, h); }
    std::string prefill_serial(uint32_t lane, const ie::LanesRequest& rq, std::span<const ie::LanesChunk> ch,
                               const std::function<bool()>& stop, size_t& done) override {
        done = 0;
        for (size_t k = 0; k < ch.size(); ++k) {
            if (k > 0 && stop && stop()) break;
            for (uint32_t s = 0; s < 2; ++s) {
                if (deep_row_ms) std::this_thread::sleep_for(std::chrono::milliseconds(deep_cost(ch[k].second, ch[k].first)));   // (P4 B36)
                if (auto e = forward(lane, s, rq.ids->data() + ch[k].first, ch[k].second, ch[k].first); !e.empty()) return e;
            }
            ++done;
        }
        return {};
    }
    bool anchor_on = true;   // P4 B36 (B) (Q35mLanesModel's IE_Q35MOE_ANCHOR)
    std::string prompt_end(uint32_t lane, const ie::LanesRequest& rq, uint32_t Tp, uint32_t reused, bool kept) override {
        St& L = st[lane];
        const uint32_t T = uint32_t(rq.ids->size());
        turn_sleep();
        if (rq.snap_at > reused && Tp == rq.snap_at)
            if (auto e = conv_snapshot(lane, rq, Tp); !e.empty()) return e;
        if (Tp < T) {
            std::vector<ie::LanesChunk> rest;
            ie::q35m_chunks(Tp, T, kChunk, rest);
            for (const auto& [p0, n] : rest)
                for (uint32_t s = 0; s < 2; ++s)
                    if (auto e = forward(lane, s, rq.ids->data() + p0, n, p0); !e.empty()) return e;
            kept = false;
        }
        if (!kept) L.kept = L.h;
        return {};
    }
    std::string reset_lane(uint32_t lane) override { st[lane] = St{}; sticky.drop(lane); return pipe.reset_lane(lane); }
    int32_t sample(uint32_t lane, bool first, const ie::LanesSampling&, std::span<const int32_t>, uint64_t seed, std::string&) override {
        if (stops_at(st[lane].seq[1])) return kStop;
        return id_of(first ? st[lane].kept : st[lane].h, seed);
    }
    // the crown's finish (Q35mLanesModel::finish): forward a "length" reply's last id, then the reply snapshot
    std::atomic<int> finishes{0};
    std::string finish(uint32_t lane, const ie::LanesRequest& rq, std::span<const int32_t> out, const std::string&, uint32_t pos) override {
        ++finishes;
        turn_sleep();
        std::vector<int32_t> full(rq.ids->begin(), rq.ids->end());
        full.insert(full.end(), out.begin(), out.end());
        if (pos + 1 == full.size()) {
            for (uint32_t s = 0; s < 2; ++s)
                if (auto e = forward(lane, s, full.data() + pos, 1, pos); !e.empty()) return e;
            ++pos;
        }
        if (pos != full.size() || st[lane].seq[1] != full) { ++bad; return "the lane does not hold prompt ++ reply"; }
        cache.insert(full, st[lane].h);
        ++snaps;
        return {};
    }
    std::atomic<uint64_t> snaps{0};
    uint64_t snapshots() const override { return snaps.load(); }
    void keep_logits(uint32_t lane) override { if (in_group_cb) ++kept_in_group; st[lane].kept = st[lane].h; }
    bool is_stop(int32_t id) const override { return id == kStop; }
    std::string detok(std::span<const int32_t> out) const override {
        std::string s;
        for (int32_t id : out) s += std::to_string(id) + ",";
        return s;
    }
    std::string pipe_start(DoneFn d) override { return pipe.start([this, d](uint32_t lane) { d(lane); rec('L', lane, 0, 0); }); }
    std::string pipe_submit(uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0) override {
        rec('S', lane, pos0, T);
        return pipe.submit(lane, ids, T, pos0);
    }
    // (B39 (5): Q35mLanesModel::pipe_submit_front -- the tail piece to the front of stage 0's queue)
    std::atomic<uint32_t> fronts{0};
    std::string pipe_submit_front(uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0) override {
        ++fronts;
        rec('F', lane, pos0, T);
        return pipe.submit(lane, ids, T, pos0, /*front=*/true);
    }
    std::string pipe_pause() override { return pipe.pause(); }
    std::string pipe_resume() override { return pipe.resume(); }
    bool pipe_paused() const override { return pipe.paused(); }
    std::string pipe_stop() override { return pipe.stop(); }
    std::string pipe_error() const override { return pipe.error(); }
    uint32_t pipe_max_rows() const override { return pipe.max_rows(); }
};

std::vector<int32_t> conv(uint32_t n, int32_t base) {
    std::vector<int32_t> p(n);
    for (uint32_t i = 0; i < n; ++i) p[i] = base + int32_t(i % 89) + 1;
    return p;
}

struct Req {
    std::vector<int32_t> ids;
    ie::LanesRequest rq;
    ie::LanesResult res;
    int cancel_after = -1;
    int pieces = 0;
    const std::atomic<bool>* quit = nullptr;   // the client leaves once this is set
    std::chrono::steady_clock::time_point t_first{};   // (P4 B34 (3)) its first token reached the client
    Req(std::vector<int32_t> p, uint32_t n, uint64_t rng, uint32_t snap_at, uint32_t share_at = 0) : ids(std::move(p)) {
        rq.ids = &ids; rq.sp.max_tokens = n; rq.rng = rng; rq.snap_at = snap_at; rq.share_at = share_at;
    }
    void run(ie::LanesServe& s) {
        rq.ids = &ids;
        res = s.run(rq, [this](std::string_view piece) {
            if (piece.empty()) return true;
            if (++pieces == 1) t_first = std::chrono::steady_clock::now();
            return !(cancel_after >= 0 && pieces >= cancel_after) && !(quit && quit->load());
        });
    }
};

void run_all(std::vector<Req*> rs, ie::LanesServe& s) {
    std::vector<std::thread> th;
    for (Req* r : rs) th.emplace_back([r, &s] { r->run(s); });
    for (auto& t : th) t.join();
}

void test_rules() {
    ie::Q35mLaneShape sh;   // one card of the crown split: layers [0, 20) = 5 full attention + 15 DeltaNet
    sh.n_full = 5; sh.n_kv_heads = 2; sh.head_dim = 256;
    sh.n_lin = 15; sh.v_heads = 32; sh.k_head_dim = 128; sh.v_head_dim = 128; sh.conv_channels = 8192; sh.conv_kernel = 4;
    const uint64_t b = ie::q35m_lane_bytes(sh, 32768);
    const uint64_t want = 335544320ull + 15ull * (2097152ull + 49152ull);   // KV 320 MiB + DeltaNet 30.7 MiB
    check(b == want, "q35m_lane_bytes at 32K = " + std::to_string(b) + " B (" + std::to_string(b >> 20) + " MiB per card)");
    ie::Q35mLaneShape kv_only = sh; kv_only.n_lin = 0;
    check(ie::q35m_lane_bytes(kv_only, 1024) == 2ull * 5 * 2 * 1024 * 256 * 2, "q35m_lane_bytes: a card with no DeltaNet layer");
    {   // P4 B14: the admission of --parallel N (q35m_lanes_fit): N - 1 extra lanes + the reserve within the card's free VRAM
        const uint64_t G = 1ull << 30, res = 1536ull << 20;
        const uint64_t exact = 15 * b + res;   // --parallel 16 at 32K: 15 x ~350 MiB + 1.5 GiB
        check(ie::q35m_lanes_fit(exact, b, 16, 32768, res, 1).empty(), "q35m_lanes_fit: 16 lanes fit exactly 15 lanes + the reserve");
        const std::string m = ie::q35m_lanes_fit(exact - 1, b, 16, 32768, res, 1);
        check(m.find("card 1 has 6.64 GiB free; 15 extra lane(s) at ctx 32768 need 5.14 GiB (0.342 GiB each) + a 1.50 GiB reserve") !=
                  std::string::npos && m.find("Lower --parallel or --slot-ctx") != std::string::npos,
              "q35m_lanes_fit: one byte short refuses with the numbers: " + m);
        check(ie::q35m_lanes_fit(res, b, 1, 32768, res, 0).empty() && !ie::q35m_lanes_fit(res - 1, b, 1, 32768, res, 0).empty(),
              "q35m_lanes_fit: one lane needs only the reserve");
        bool mono = true;   // a card that fits N lanes fits N - 1
        for (uint32_t n = 2; n <= 16; ++n)
            if (ie::q35m_lanes_fit(4 * G, b, n, 32768, res, 0).empty() && !ie::q35m_lanes_fit(4 * G, b, n - 1, 32768, res, 0).empty()) mono = false;
        check(mono, "q35m_lanes_fit: monotone in N");
        check(!ie::q35m_lanes_fit(6 * G, b, 16, 32768, res, 0).empty() &&
                  ie::q35m_lanes_fit(6 * G, ie::q35m_lane_bytes(sh, 16384), 16, 16384, res, 0).empty(),
              "q35m_lanes_fit: 16 x 32K refused on 6 GiB free, 16 x 16K fits (--slot-ctx sizes the lanes)");
    }
    check(ie::q35m_prefill_chunk(32768, nullptr, nullptr, nullptr) == 8192, "prefill chunk: ctx >= 8192 -> 8192");
    check(ie::q35m_prefill_chunk(4096, nullptr, nullptr, nullptr) == 512, "prefill chunk: short ctx -> 512");
    check(ie::q35m_prefill_chunk(4096, nullptr, "1", nullptr) == 4096, "prefill chunk: IE_QWEN36_MOE_ONEDNN below 8192 -> min(8192, ctx)");
    check(ie::q35m_prefill_chunk(32768, "1", nullptr, nullptr) == 512, "prefill chunk: IE_QWEN36_NO_MOE_ONEDNN -> 512");
    check(ie::q35m_prefill_chunk(32768, nullptr, nullptr, "1024") == 1024, "prefill chunk: IE_QWEN35_PREFILL_CHUNK overrides");
    check(ie::q35m_prefill_chunk(4096, nullptr, nullptr, "99999") == 512, "prefill chunk: an override past the ctx is ignored");
    check(ie::q35m_prefill_chunk(300, nullptr, nullptr, nullptr) == 300, "prefill chunk: never above the ctx");
    check(ie::q4e_lane_ctx(0, 262144) == 32768 && ie::q4e_lane_ctx(4096, 32768) == 4096, "lane ctx: --slot-ctx 0 = 32K, capped");
    // the plan + the prompt end's rest == --parallel 1's prefill_to(snap_at), prefill_to(T), for every shape
    bool all = true;
    for (uint32_t T = 1; T <= 40 && all; ++T)
        for (uint32_t snap = 1; snap <= T && all; ++snap)
            for (uint32_t reused = 0; reused < T && all; ++reused)
                for (uint32_t pf : {1u, 3u, 8u, 64u}) {
                    std::vector<ie::LanesChunk> ref;
                    uint32_t pos = reused;
                    auto to = [&](uint32_t end) { while (pos < end) { const uint32_t k = std::min(pf, end - pos); ref.emplace_back(pos, k); pos += k; } };
                    to(snap); const size_t at = ref.size(); to(T);
                    ie::LanesPlan p = ie::q35m_plan(T, snap, reused, pf);
                    std::vector<ie::LanesChunk> got = p.chunks;
                    const bool boundary = snap > reused ? (p.Tp == snap && got.size() == at) : p.Tp == T;
                    ie::q35m_chunks(p.Tp, T, pf, got);
                    if (got != ref || !boundary) {
                        all = false;
                        std::printf("  plan mismatch at T %u snap %u reused %u pf %u\n", T, snap, reused, pf);
                    }
                }
    check(all, "q35m_plan + the rest == --parallel 1's chunks for every (T <= 40, snap, reused, pf); the pipe's part ends at the snapshot");
}

void test_serve(bool rows) {
    FakeCrown m({300, 120, 120}, rows);
    ie::LanesServe s(m, {});
    Cache ref;   // --parallel 1's cache, driven by the same request sequence
    auto expect = [&](const Req& r, uint32_t* restored = nullptr) {
        return text_of(serial_ref(ref, r.ids, r.rq.snap_at, r.rq.sp.max_tokens, r.rq.rng, restored));
    };
    // three conversations, each with a stable prefix (snap_at) and a volatile tail
    auto mk = [&](int k, uint32_t n) { std::vector<int32_t> p = conv(23 + 7 * k, 100 * (k + 1)); p.push_back(3); p.push_back(4); return Req(p, n, 40 + k, uint32_t(p.size() - 2)); };

    // warm: each once, alone; the reference in the same order
    bool warm_ok = true;
    for (int k = 0; k < 3; ++k) {
        Req r = mk(k, 8);
        r.run(s);
        uint32_t rs = 0;
        const std::string want = expect(r, &rs);
        warm_ok = warm_ok && r.res.text == want && r.res.cached_tokens == rs;
    }
    check(warm_ok, "warm: 3 requests alone == --parallel 1's (cold prefills, snapshots inserted)");
    // solo: each again (restored from the shared cache)
    std::vector<std::string> solo(3);
    bool solo_ok = true, hit = true;
    for (int k = 0; k < 3; ++k) {
        Req r = mk(k, 30);
        r.run(s);
        uint32_t rs = 0;
        const std::string want = expect(r, &rs);
        solo[k] = r.res.text;
        solo_ok = solo_ok && r.res.text == want && r.res.cached_tokens == rs;
        hit = hit && r.res.cached_tokens == r.rq.snap_at;
    }
    check(solo_ok && hit, "solo: 3 requests alone == --parallel 1's, each restored from the shared cache at its snapshot");
    // batch: all at once == solo
    {
        Req a = mk(0, 30), b = mk(1, 30), c = mk(2, 30);
        run_all({&a, &b, &c}, s);
        check(a.res.text == solo[0] && b.res.text == solo[1] && c.res.text == solo[2], "batch: 3 at once == solo");
        check(m.max_active.load() >= 2, "the two stages overlapped (max " + std::to_string(m.max_active.load()) + " at once)");
    }
    // cancel: request 1 closed after 3 pieces, the others == solo; then it answers again
    {
        Req a = mk(0, 30), b = mk(1, 30), c = mk(2, 30);
        b.cancel_after = 3;
        run_all({&a, &b, &c}, s);
        check(b.res.finish_reason == "abort" && a.res.text == solo[0] && c.res.text == solo[2], "cancel: the survivors == solo");
        Req again = mk(1, 30);
        again.run(s);
        check(again.res.text == solo[1], "the cancelled conversation answers == solo afterwards");
    }
    // cold: three NEW long prompts at once (several pieces each: prefilled in a turn and through the pipe beside decoding
    // lanes) == --parallel 1 running them one after another
    {
        Req a(conv(61, 5000), 20, 1, 61), b(conv(47, 6000), 20, 2, 40), c(conv(33, 7000), 20, 3, 33);
        run_all({&a, &b, &c}, s);
        const bool ok = a.res.text == expect(a) && b.res.text == expect(b) && c.res.text == expect(c);
        check(ok && a.res.cached_tokens == 0 && b.res.cached_tokens == 0, "cold: 3 new prompts at once == --parallel 1's");
    }
    // a follow-up turn extends a conversation past its snapshot: restored at the old boundary, == --parallel 1's
    {
        Req f = mk(2, 12);
        std::vector<int32_t> p(f.ids.begin(), f.ids.begin() + f.rq.snap_at);
        for (int i = 0; i < 9; ++i) p.push_back(900 + i);
        Req g(p, 12, 77, uint32_t(p.size() - 2));
        uint32_t rs = 0;
        const std::string want = expect(g, &rs);
        g.run(s);
        check(g.res.text == want && g.res.cached_tokens == rs && rs == f.rq.snap_at,
              "a follow-up turn restores the conversation's snapshot (" + std::to_string(g.res.cached_tokens) + " cached) == --parallel 1's");
    }
    // an exact repeat of a whole-prompt snapshot (snap_at == T): the DeltaNet rule refuses the restore (full prefill)
    {
        std::vector<int32_t> p = conv(19, 8000);
        Req x(p, 6, 5, uint32_t(p.size())), y(p, 6, 5, uint32_t(p.size()));
        x.run(s); y.run(s);
        const std::string wx = expect(x), wy = expect(y);
        check(x.res.text == wx && y.res.text == wy && y.res.cached_tokens == 0, "an exact repeat of a full-depth snapshot prefills in full");
    }
    check(m.bad == 0, "no stage saw a wrong position (no cross-lane mix-up)");
    check(m.finishes.load() == 0, "no finish hook without reply_cache");
    const std::string js = s.status_json();
    check(js.find("\"lanes\":3") != std::string::npos, "status_json: " + js);
    check(m.snaps.load() > 0 && js.find("\"snapshots\":" + std::to_string(m.snaps.load())) != std::string::npos,
          "status_json: \"snapshots\" = the model's conversation snapshots (" + std::to_string(m.snaps.load()) + ")");
    s.shutdown();
    if (rows) {
        const auto gs = m.pipe.group_sizes();   // (since the first start: every pipe of this server)
        uint64_t multi = 0;
        for (size_t g = 2; g < gs.size(); ++g) multi += gs[g];
        const std::string seen = "rows: groups of >= 2 lanes ran (" + std::to_string(multi) + "), sample_rows " + std::to_string(m.rows_calls.load()) +
                                 " calls for " + std::to_string(m.rows_ids.load()) + " ids";
        // P4 B39: with equal stage times, singleton decode groups rotate over the two stages and stage 0 finds two lanes queued
        // only after a BURST -- several lanes resubmitted together -- which the paused turns' releases gave at every turn; the
        // turns beside the pipe give none, so a merge is a matter of the lanes' alignment (0 or dozens from run to run). P4 B42:
        // the regroup wait merges them again (required with it, as under the drain switch); reported only with both off.
        if (g_turn_drain || g_regroup_us) check(multi > 0 && m.group_steps.load() > 0 && m.rows_calls.load() > 0 && m.rows_calls.load() < m.rows_ids.load(), seen);
        else std::printf("  (%s; the merged groups need a burst the drain-free turns do not give)\n", seen.c_str());
        check(m.rows_calls.load() <= m.rows_ids.load(), "rows: every sample_rows call covered >= 1 id");
    } else {
        check(m.rows_calls.load() == 0 && m.group_steps.load() == 0, "rows off: the per-lane pipe (no group step, no sample_rows)");
    }
}

// P4 B14 rows: a 1-row prefill piece lands inside a group (its logits kept from the group's callback), beside decoding lanes;
// every reply == --parallel 1's.
void test_rows_prefill_piece() {
    // three decoding lanes + the prompt: more units in flight than stages, so a queued decode lane and the piece meet
    // (with two units the AUTO groups alternate on the two cards and never merge)
    FakeCrown m({4000, 4000, 4000, 4000, 4000}, true);
    ie::LanesServe s(m, {});
    Cache ref;
    Req d1(conv(12, 9000), 3000, 91, 12), d2(conv(13, 9100), 3000, 92, 13), d3(conv(14, 9200), 3000, 93, 14);
    std::atomic<bool> quit{false};
    d1.quit = d2.quit = d3.quit = &quit;   // (they leave once the piece test is done: aborted replies, prefixes of --parallel 1's)
    std::thread t1([&] { d1.run(s); }), t2([&] { d2.run(s); }), t3([&] { d3.run(s); });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    bool ok = true;
    int tries = 0;
    for (; tries < 300 && m.kept_in_group.load() == 0; ++tries) {   // T = 26 = 5 x 5 + 1: the last piece is 1 row
        std::vector<int32_t> p = conv(26, 20000 + 100 * tries);
        Req r(p, 4, 100 + tries, uint32_t(p.size()));
        r.run(s);
        ok = ok && r.res.text == text_of(serial_ref(ref, r.ids, r.rq.snap_at, r.rq.sp.max_tokens, r.rq.rng));
    }
    quit = true;
    t1.join(); t2.join(); t3.join();
    auto prefix_of = [](const std::string& got, const std::string& want) { return want.rfind(got, 0) == 0; };
    ok = ok && prefix_of(d1.res.text, text_of(serial_ref(ref, d1.ids, 12, 3000, 91))) &&
         prefix_of(d2.res.text, text_of(serial_ref(ref, d2.ids, 13, 3000, 92))) &&
         prefix_of(d3.res.text, text_of(serial_ref(ref, d3.ids, 14, 3000, 93)));
    check(ok, "rows: prompts beside decoding lanes == --parallel 1's (" + std::to_string(tries) + " prompts)");
    // (P4 B34: a lookahead enters only an idle stage 0, so a 1-row last piece sent that way never meets a queued decode lane;
    // with the lane pipeline the group landing is reported, not required -- the block without it requires it. P4 B39: the
    // landing needs a burst at stage 0, which the paused turns' releases gave; with the turns beside the pipe (no bursts) it is
    // reported, and required in the IE_Q35MOE_TURN_DRAIN=1 run)
    if (!g_pipeline && g_turn_drain) check(m.kept_in_group.load() > 0, "rows: a prompt's last (1-row) piece landed inside a group and kept its logits");
    else std::printf("  (%s: a last 1-row piece landed inside a group %u time(s))\n", g_pipeline ? "lane pipeline" : "turns beside the pipe", m.kept_in_group.load());
    check(m.bad == 0, "rows: no stage saw a wrong position");
    s.shutdown();
}

// Follow-up turns with --parallel 1's reply snapshot (the gen-cache): turn 1 ends on its length (A: the last id is forwarded
// in the finish turn) or on a stop id (B); turn 2 of both at once restores prompt ++ reply and == --parallel 1's.
void test_reply_snapshot(bool rows) {
    FakeCrown m({300, 300}, rows);
    ie::LanesServe s(m, {});
    Cache ref;
    auto run_ref = [&](Req& r, uint32_t* rs) {
        return text_of(serial_ref(ref, r.ids, r.rq.snap_at, r.rq.sp.max_tokens, r.rq.rng, rs, r.rq.reply_cache));
    };
    std::vector<int32_t> pa = conv(29, 3000), pb = conv(36, 4000);
    Req a1(pa, 12, 21, 27), b1(pb, 40, 22, 34);
    a1.rq.reply_cache = b1.rq.reply_cache = true;
    uint32_t ra = 0, rb = 0;
    a1.run(s);
    const std::string wa1 = run_ref(a1, &ra);
    g_stop_at = uint32_t(pb.size()) + 6;          // B's reply stops after 6 ids
    b1.run(s);
    const std::string wb1 = run_ref(b1, &rb);
    g_stop_at = 0;
    check(a1.res.finish_reason == "length" && a1.res.text == wa1, "turn 1 of A (length) == --parallel 1's");
    check(b1.res.finish_reason == "stop" && b1.res.completion_tokens == 6 && b1.res.text == wb1, "turn 1 of B (a stop id after 6) == --parallel 1's");
    check(m.finishes.load() == 2 && m.bad == 0, "the finish hook ran once per reply, on a lane holding prompt ++ reply");
    auto turn2 = [](const std::vector<int32_t>& p1, const std::string& reply, int32_t base) {
        std::vector<int32_t> p = p1;
        for (size_t i = 0; i < reply.size();) { const size_t j = reply.find(',', i); p.push_back(std::stoi(reply.substr(i, j - i))); i = j + 1; }
        for (int k = 0; k < 7; ++k) p.push_back(base + k);
        return p;
    };
    std::vector<int32_t> pa2 = turn2(pa, a1.res.text, 700), pb2 = turn2(pb, b1.res.text, 800);
    Req a2(pa2, 15, 31, uint32_t(pa2.size() - 2)), b2(pb2, 15, 32, uint32_t(pb2.size() - 2));
    a2.rq.reply_cache = b2.rq.reply_cache = true;
    run_all({&a2, &b2}, s);
    uint32_t ra2 = 0, rb2 = 0;
    const std::string wa2 = run_ref(a2, &ra2), wb2 = run_ref(b2, &rb2);
    const uint32_t da = uint32_t(pa.size() + a1.res.completion_tokens), db = uint32_t(pb.size() + b1.res.completion_tokens);
    check(ra2 == da && rb2 == db, "reference: turn 2 restores prompt ++ reply (" + std::to_string(ra2) + ", " + std::to_string(rb2) + ")");
    check(a2.res.cached_tokens == da && b2.res.cached_tokens == db,
          "lanes: turn 2 restores the reply snapshots (" + std::to_string(a2.res.cached_tokens) + ", " + std::to_string(b2.res.cached_tokens) + " cached)");
    check(a2.res.text == wa2 && b2.res.text == wb2, "turn 2 of A and B at once == --parallel 1's");
    check(m.bad == 0, "no stage saw a wrong position");
    s.shutdown();
}

// P4 B15: the shared-prefix rules (include/ie/q35m_lanes.hpp, q4e_lanes.hpp)
void test_share_rules() {
    check(ie::shared_prefix_enabled(nullptr) && ie::shared_prefix_enabled("1") && !ie::shared_prefix_enabled("0") &&
              ie::shared_prefix_enabled("01"), "IE_SHARED_PREFIX: on unless \"0\"");
    check(ie::shared_prefix_min(nullptr) == 1024 && ie::shared_prefix_min("64") == 64 && ie::shared_prefix_min("0") == 1024 &&
              ie::shared_prefix_min("x") == 1024, "IE_SHARED_PREFIX_MIN: default 1024, >= 1");
    const std::vector<int32_t> ids = {1, 2, 3, 4, 5, 6};
    check(ie::shared_prefix_accept({1, 2, 3}, ids, 3) == 3 && ie::shared_prefix_accept({1, 2, 3}, ids, 4) == 0 &&
              ie::shared_prefix_accept({1, 2, 9}, ids, 1) == 0 && ie::shared_prefix_accept(ids, ids, 1) == 0 &&
              ie::shared_prefix_accept({}, ids, 0) == 0,
          "shared_prefix_accept: a strict prefix of at least min tokens, else 0");
    check(ie::shared_prefix_boundary(10, 20, true) == 10 && ie::shared_prefix_boundary(20, 20, true) == 0 &&
              ie::shared_prefix_boundary(10, 20, false) == 0 && ie::shared_prefix_boundary(0, 20, true) == 0,
          "shared_prefix_boundary: below the snapshot boundary, the cache on");
    // q35m_plan with a share boundary == --parallel 1's prefill_to(share), prefill_to(snap), prefill_to(T); a chunk ends at the
    // mark; the pieces after it are those of the plan restored at the mark
    bool all = true, same_after = true, crown_mark = true;
    for (uint32_t T = 2; T <= 34 && all; ++T)
        for (uint32_t snap = 1; snap <= T && all; ++snap)
            for (uint32_t share = 0; share < snap && all; ++share)
                for (uint32_t reused = 0; reused < T && all; ++reused)
                    for (uint32_t pf : {1u, 3u, 8u}) {
                        std::vector<ie::LanesChunk> ref;
                        uint32_t pos = reused;
                        auto to = [&](uint32_t end) { while (pos < end) { const uint32_t k = std::min(pf, end - pos); ref.emplace_back(pos, k); pos += k; } };
                        if (share > reused) to(share);
                        to(snap); to(T);
                        ie::LanesPlan p = ie::q35m_plan(T, snap, reused, pf, share);
                        std::vector<ie::LanesChunk> got = p.chunks;
                        ie::q35m_chunks(p.Tp, T, pf, got);
                        if (got != ref) { all = false; std::printf("  plan mismatch T %u snap %u share %u reused %u pf %u\n", T, snap, share, reused, pf); }
                        const bool due = share > reused && share < p.Tp;
                        if (due != (p.mark == share && share > 0)) crown_mark = false;
                        if (due) {
                            bool ends = false;
                            size_t k = 0;
                            for (; k < p.chunks.size(); ++k) if (p.chunks[k].first + p.chunks[k].second == share) { ends = true; break; }
                            const ie::LanesPlan r = ie::q35m_plan(T, snap, share, pf, share);
                            const std::vector<ie::LanesChunk> after(p.chunks.begin() + long(k) + 1, p.chunks.end());
                            if (!ends || after != r.chunks || r.mark != 0) same_after = false;
                        }
                        // Flash-Next's q4e_plan_split: the same rule over its ranges
                        std::vector<std::pair<uint32_t, uint32_t>> q, qr;
                        ie::q4e_plan_split(reused, snap > reused ? snap : T, share, pf, true, q);
                        uint32_t qp = reused;
                        auto qto = [&](uint32_t end) { ie::q4e_plan_range(qp, end, pf, true, qr); if (end > qp) qp = end; };
                        const uint32_t qTp = snap > reused ? snap : T;
                        if (share > reused && share < qTp) qto(share);
                        qto(qTp);
                        if (q != qr) { all = false; std::printf("  q4e split mismatch T %u snap %u share %u reused %u pf %u\n", T, snap, share, reused, pf); }
                    }
    check(all, "q35m_plan / q4e_plan_split with a shared-prefix boundary == --parallel 1's split prefill, every shape (T <= 34)");
    check(crown_mark, "q35m_plan: mark == share exactly when reused < share < Tp");
    check(same_after, "q35m_plan: a chunk ends at the mark, and the pieces after it == the plan restored at the mark");
}

// P4 B15: agent traffic on the crown -- one system prefix, a new user turn per request. The first request prefills the prefix
// and marks it; later requests (concurrent, other lanes) restore it into their own lane (cached == share_at), and every reply
// == --parallel 1's over the same sequence AND == a cold server's reply to the same request (the restore changes no byte).
void test_agent_traffic() {
    constexpr uint32_t kSys = 23;   // the shared prefix (system + tools)
    const std::vector<int32_t> sys = conv(kSys, 400);
    auto agent_req = [&](int k, uint32_t n) {
        std::vector<int32_t> ids = sys;
        for (int32_t v : conv(6 + uint32_t(k) % 5, 700 + 17 * k)) ids.push_back(v);   // the user turn
        const uint32_t snap = uint32_t(ids.size());
        ids.push_back(3); ids.push_back(5);                                         // the generation prompt
        auto r = std::make_unique<Req>(ids, n, 900 + uint64_t(k), snap, kSys);
        return r;
    };
    FakeCrown m({300, 120, 120, 120}, false);
    ie::LanesServe s(m, {});
    Cache ref;
    std::vector<std::unique_ptr<Req>> rs;
    rs.push_back(agent_req(0, 12));
    rs[0]->run(s);
    bool ok = rs[0]->res.finish_reason == "length";
    uint32_t r0 = 0;
    ok = ok && text_of(serial_ref(ref, rs[0]->ids, rs[0]->rq.snap_at, 12, rs[0]->rq.rng, &r0, false, kSys)) == rs[0]->res.text;
    check(ok && m.marks.load() >= 1 && rs[0]->res.cached_tokens == 0,
          "agent traffic: request 1 cold (0 cached), == --parallel 1, its shared prefix marked (" + std::to_string(m.marks.load()) + " mark(s))");
    for (int k = 1; k <= 3; ++k) rs.push_back(agent_req(k, 12));
    run_all({rs[1].get(), rs[2].get(), rs[3].get()}, s);
    bool all_share = true, all_ref = true, all_cold = true;
    for (int k = 1; k <= 3; ++k) {
        Req& r = *rs[size_t(k)];
        all_share = all_share && r.res.cached_tokens == kSys;
        Cache mine = ref;   // --parallel 1 given the cache state after request 1 (requests 2-4 each hit only the shared prefix)
        all_ref = all_ref && text_of(serial_ref(mine, r.ids, r.rq.snap_at, 12, r.rq.rng, nullptr, false, kSys)) == r.res.text;
        Cache none;         // a cold server: the whole prompt prefilled (split at the shared prefix all the same)
        all_cold = all_cold && text_of(serial_ref(none, r.ids, r.rq.snap_at, 12, r.rq.rng, nullptr, false, kSys)) == r.res.text;
    }
    check(all_share, "agent traffic: requests 2-4 (concurrent) each restored the shared prefix into their own lane (cached == 23)");
    check(all_ref, "agent traffic: requests 2-4 == --parallel 1");
    check(all_cold, "agent traffic: requests 2-4 == a cold server's reply (restore == cold prefill)");
    // with the share boundary off, nothing is shared: a new user turn caches 0 (today's behaviour)
    FakeCrown m2({300, 120}, false);
    ie::LanesServe s2(m2, {});
    auto a = agent_req(0, 4); a->rq.share_at = 0; a->run(s2);
    auto b = agent_req(1, 4); b->rq.share_at = 0; b->run(s2);
    check(b->res.cached_tokens == 0 && m2.marks.load() == 0, "agent traffic, share_at 0: no mark, 0 cached (HEAD behaviour)");
    check(m.bad.load() == 0 && m2.bad.load() == 0, "agent traffic: no step at a wrong position, no mark off its boundary");
    s.shutdown(); s2.shutdown();
}

// P4 B29: Options::mix_chunk (the crown's default-on cap, IE_Q35MOE_MIX_CHUNK): a prompt that arrives while other lanes decode
// goes through the pipe in pieces of at most mix_chunk rows, each plan chunk cut in order; its reply == --parallel 1's prefill
// over those pieces (the fake's hash sees every piece boundary, so the cut is checked exactly). A prompt alone is untouched.
void test_mix_chunk(bool rows, uint32_t cap) {
    FakeCrown m({4000, 4000, 4000}, rows);
    ie::LanesServe::Options o;
    o.mix_chunk = cap;
    ie::LanesServe s(m, o);
    Cache ref;
    // alone: the serial turn's plan pieces, == --parallel 1
    std::vector<int32_t> pa = conv(37, 31000);
    Req a(pa, 6, 11, uint32_t(pa.size()));
    a.run(s);
    check(a.res.text == text_of(serial_ref(ref, a.ids, a.rq.snap_at, 6, a.rq.rng)), "mix_chunk " + std::to_string(cap) + ": a prompt alone == --parallel 1 (no cap)");
    // beside two decoding lanes: the pipe's pieces are cut to the cap
    Req d1(conv(12, 32000), 3000, 91, 12), d2(conv(13, 33000), 3000, 92, 13);
    std::atomic<bool> quit{false};
    d1.quit = d2.quit = &quit;
    std::thread t1([&] { d1.run(s); }), t2([&] { d2.run(s); });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    m.watch = true;
    std::vector<int32_t> pb = conv(43, 34000);
    Req b(pb, 6, 12, uint32_t(pb.size()));
    b.run(s);
    m.watch = false;
    quit = true;
    t1.join(); t2.join();
    const uint32_t eff = cap ? std::min(cap, kChunk) : kChunk;
    uint64_t h = kSeed;   // --parallel 1 over the cut plan: no restore (a new prompt), snap_at == T
    const ie::LanesPlan p = ie::q35m_plan(uint32_t(pb.size()), b.rq.snap_at, 0, kChunk);
    for (const auto& [p0, t] : p.chunks)
        for (uint32_t q = 0; q < t; q += eff) h = step_hash(h, pb.data() + p0 + q, std::min(eff, t - q), p0 + q);
    std::vector<ie::LanesChunk> rest;
    ie::q35m_chunks(p.Tp, uint32_t(pb.size()), kChunk, rest);
    for (const auto& [p0, t] : rest) h = step_hash(h, pb.data() + p0, t, p0);
    std::vector<int32_t> want;
    uint32_t pos = uint32_t(pb.size());
    for (uint32_t k = 0; k < 6; ++k) { const int32_t id = id_of(h, b.rq.rng + k); want.push_back(id); h = step_hash(h, &id, 1, pos++); }
    check(b.res.text == text_of(want), "mix_chunk " + std::to_string(cap) + (rows ? " rows" : "") +
                                           ": a prompt beside decoding lanes == --parallel 1 over the cut pieces");
    check(b.res.cached_tokens == 0 && m.max_piece.load() >= 2 && m.max_piece.load() <= eff,
          "mix_chunk " + std::to_string(cap) + ": its pipe pieces <= " + std::to_string(eff) + " rows (largest " + std::to_string(m.max_piece.load()) + ")");
    check(m.bad == 0, "mix_chunk: no stage saw a wrong position");
    s.shutdown();
}

// P4 B29: the 2026-09-30 wave -- N new prompts with ONE shared system prefix arrive at once on a fresh server. With the prefill
// FIFO (the crown's default now) the shared prefix is prefilled once: the first prompt marks it, the others restore it
// (cached == the prefix); off (B10-B28), every prompt prefills it. Every reply == a cold server's (restore == cold prefill).
void test_wave(bool fifo) {
    constexpr uint32_t kSys = 23;
    const std::vector<int32_t> sys = conv(kSys, 500);
    FakeCrown m({300, 120, 120, 120, 120}, true);
    ie::LanesServe::Options o;
    o.prefill_fifo = fifo;
    ie::LanesServe s(m, o);
    std::vector<std::unique_ptr<Req>> rs;
    for (int k = 0; k < 4; ++k) {
        std::vector<int32_t> ids = sys;
        for (int32_t v : conv(6 + uint32_t(k), 800 + 13 * k)) ids.push_back(v);
        const uint32_t snap = uint32_t(ids.size());
        ids.push_back(3); ids.push_back(5);
        rs.push_back(std::make_unique<Req>(ids, 10, 950 + uint64_t(k), snap, kSys));
    }
    run_all({rs[0].get(), rs[1].get(), rs[2].get(), rs[3].get()}, s);
    int restored = 0;
    bool cold = true;
    for (auto& r : rs) {
        restored += r->res.cached_tokens == kSys;
        Cache none;
        cold = cold && text_of(serial_ref(none, r->ids, r->rq.snap_at, 10, r->rq.rng, nullptr, false, kSys)) == r->res.text;
    }
    check(cold, std::string("wave, FIFO ") + (fifo ? "on" : "off") + ": 4 prompts at once == a cold server's replies");
    if (fifo) check(restored >= 3, "wave, FIFO on: " + std::to_string(restored) + " of 4 restored the shared prefix (>= 3: prefilled once)");
    else      std::printf("  (wave, FIFO off: %d of 4 restored the shared prefix)\n", restored);
    check(m.bad == 0, "wave: no stage saw a wrong position");
    s.shutdown();
}

// P4 B29 Fix A: a swarm's multi-turn workers on a bounded cache (24 entries = the lanes' max(12, lanes + 8) at --parallel 16).
// A turn's prompt = the previous prompt's stable part ++ the reply ++ a new user turn, so it should restore the previous turn's
// snapshot. The order is the swarm's (conversations move at their own pace): all 15 start at once; 14 of them run turns 1 and
// 2 (14 at a time); then the slow one runs its turns 1 and 2. Without the supersede rule every turn leaves its older snapshot
// behind (43 entries for 15 conversations), and the LRU evicts the slow conversation's live one (the gate's misses); with it
// each conversation holds one entry, every follow-up restores its previous turn, and every reply == --parallel 1's (a
// per-conversation reference over the same turns).
void test_multiturn_cap(bool supersede, bool sticky = false) {
    constexpr int kConv = 15;
    std::vector<uint32_t> caps(16, 4000);
    FakeCrown m(caps, true);
    m.cache.cap = 24; m.cache.supersede = supersede; m.sticky.on = sticky;   // (Fix B: the lanes' own checkpoints)
    ie::LanesServe s(m, {});
    std::vector<Cache> ref(kConv);
    std::vector<std::vector<int32_t>> stable(kConv);   // each conversation's history without the generation prompt
    for (int k = 0; k < kConv; ++k) stable[k] = conv(30 + uint32_t(k), 1000 + 100 * k);
    int misses = 0, wrong = 0, fu = 0;
    auto round = [&](const std::vector<int>& ks, int t) {
        std::vector<std::unique_ptr<Req>> rs;
        for (int k : ks) {
            std::vector<int32_t> ids = stable[size_t(k)];
            const uint32_t snap = uint32_t(ids.size());
            ids.push_back(3); ids.push_back(4);   // the generation prompt
            rs.push_back(std::make_unique<Req>(ids, 6, 500 + 10 * uint64_t(k) + uint64_t(t), snap));
        }
        std::vector<Req*> all;
        for (auto& r : rs) all.push_back(r.get());
        run_all(all, s);
        for (size_t i = 0; i < ks.size(); ++i) {
            const int k = ks[i];
            Req& r = *rs[i];
            uint32_t rcached = 0;
            const std::string want = text_of(serial_ref(ref[size_t(k)], r.ids, r.rq.snap_at, 6, r.rq.rng, &rcached));
            if (t > 0) { ++fu; if (r.res.cached_tokens != rcached) ++misses; }
            if (r.res.cached_tokens == rcached && r.res.text != want) ++wrong;
            for (size_t q = 0; q < r.res.text.size();) {   // the next turn: this turn's stable part ++ the reply ++ a user turn
                const size_t j = r.res.text.find(',', q);
                stable[size_t(k)].push_back(std::stoi(r.res.text.substr(q, j - q)));
                q = j + 1;
            }
            for (int q = 0; q < 9; ++q) stable[size_t(k)].push_back(7000 + 50 * k + 9 * t + q);
        }
    };
    std::vector<int> every, rest;
    for (int k = 0; k < kConv; ++k) { every.push_back(k); if (k) rest.push_back(k); }
    round(every, 0);
    round(rest, 1); round(rest, 2);
    round({0}, 1); round({0}, 2);
    if (supersede || sticky) check(misses == 0 && wrong == 0, std::string("cap 24, ") + (supersede ? "supersede on" : "supersede off, sticky lanes on") + ": " + std::to_string(fu - misses) + " of " + std::to_string(fu) +
                                                         " follow-ups restored the previous turn; replies == --parallel 1 (" + std::to_string(wrong) + " differ)");
    else check(misses > 0 && wrong == 0, "cap 24, supersede off: " + std::to_string(misses) + " of " + std::to_string(fu) +
                                             " follow-ups lost the previous turn to the LRU (the gate's failure)");
    check(m.bad == 0, "multi-turn: no stage saw a wrong position");
    s.shutdown();
}

// P4 B29 Fix B: sticky lanes. Every case's replies == --parallel 1's (a per-conversation reference over the same turns).
//  (a) own-lane landing: 3 conversations x 3 turns on 4 lanes, a 1-entry cache (it cannot hold them): every follow-up lands on
//      its own lane and restores its previous turn in place;
//  (b) taken lane: 2 lanes, A then B then C -- C takes A's lane (the least recently used), so A's next turn finds no
//      checkpoint and restores its previous turn from the shared cache instead;
//  (c) diverged history: A's next prompt keeps its checkpoint's tokens but differs right after them (an edited reply) --
//      restored in place at the checkpoint; a prompt that leaves A's history BEFORE the checkpoint gets no in-place restore.
std::vector<int32_t> with_reply(std::vector<int32_t> stable, const std::string& reply, int32_t base) {
    for (size_t q = 0; q < reply.size();) { const size_t j = reply.find(',', q); stable.push_back(std::stoi(reply.substr(q, j - q))); q = j + 1; }
    for (int i = 0; i < 9; ++i) stable.push_back(base + i);
    return stable;
}
Req turn_req(const std::vector<int32_t>& stable, uint64_t rng) {
    std::vector<int32_t> ids = stable;
    ids.push_back(3); ids.push_back(4);
    return Req(ids, 6, rng, uint32_t(stable.size()));
}
void test_sticky() {
    {   // (a)
        FakeCrown m({4000, 4000, 4000, 4000}, true);
        m.sticky.on = true; m.cache.cap = 1;
        ie::LanesServe s(m, {});
        std::vector<Cache> ref(3);
        std::vector<std::vector<int32_t>> st = {conv(31, 100), conv(37, 200), conv(41, 300)};
        std::vector<uint32_t> lane(3, 99);
        bool same_lane = true, restored = true, eq = true;
        for (int t = 0; t < 3; ++t) {
            std::vector<Req> rs;
            for (int k = 0; k < 3; ++k) rs.push_back(turn_req(st[size_t(k)], 600 + 10 * uint64_t(k) + uint64_t(t)));
            run_all({&rs[0], &rs[1], &rs[2]}, s);
            for (int k = 0; k < 3; ++k) {
                Req& r = rs[size_t(k)];
                uint32_t rc = 0;
                eq = eq && r.res.text == text_of(serial_ref(ref[size_t(k)], r.ids, r.rq.snap_at, 6, r.rq.rng, &rc));
                if (t > 0) { same_lane = same_lane && r.res.lane == lane[size_t(k)]; restored = restored && r.res.cached_tokens == rc && rc > 0; }
                lane[size_t(k)] = r.res.lane;
                st[size_t(k)] = with_reply(st[size_t(k)], r.res.text, 9000 + 20 * k + 3 * t);
            }
        }
        check(same_lane && restored && m.own_restores.load() == 6 && m.cache_restores.load() == 0,
              "sticky (a): 6 of 6 follow-ups landed on their own lane and restored the previous turn in place (" +
                  std::to_string(m.own_restores.load()) + " in place, " + std::to_string(m.cache_restores.load()) + " from the 1-entry cache)");
        check(eq && m.bad == 0, "sticky (a): replies == --parallel 1; the lane's rows below the depth were the prompt's");
        s.shutdown();
    }
    {   // (b)
        FakeCrown m({4000, 4000}, true);
        m.sticky.on = true;
        ie::LanesServe s(m, {});
        Cache refA, refB, refC;
        std::vector<int32_t> a = conv(33, 100), b = conv(35, 200), c = conv(39, 300);
        Req a0 = turn_req(a, 1); a0.run(s);
        Req b0 = turn_req(b, 2); b0.run(s);
        Req c0 = turn_req(c, 3); c0.run(s);
        (void)serial_ref(refA, a0.ids, a0.rq.snap_at, 6, 1); (void)serial_ref(refB, b0.ids, b0.rq.snap_at, 6, 2); (void)serial_ref(refC, c0.ids, c0.rq.snap_at, 6, 3);
        const uint32_t own0 = m.own_restores.load();
        Req a1 = turn_req(with_reply(a, a0.res.text, 9100), 4);
        a1.run(s);
        uint32_t rc = 0;
        const std::string want = text_of(serial_ref(refA, a1.ids, a1.rq.snap_at, 6, 4, &rc));
        check(c0.res.lane == a0.res.lane && b0.res.lane != a0.res.lane,
              "sticky (b): C took A's lane (" + std::to_string(c0.res.lane) + "), the least recently used; B kept its own");
        check(m.own_restores.load() == own0 && a1.res.cached_tokens == rc && rc == a0.rq.snap_at && a1.res.text == want && m.bad == 0,
              "sticky (b): A's next turn found no checkpoint and restored its previous turn from the cache (" +
                  std::to_string(a1.res.cached_tokens) + " cached) == --parallel 1");
        s.shutdown();
    }
    {   // (c)
        FakeCrown m({4000, 4000}, true);
        m.sticky.on = true; m.cache.cap = 1;
        ie::LanesServe s(m, {});
        std::vector<int32_t> a = conv(45, 100);
        Req a0 = turn_req(a, 1); a0.run(s);
        Cache ref;
        (void)serial_ref(ref, a0.ids, a0.rq.snap_at, 6, 1);
        // an edited reply: A's checkpoint tokens, then a different assistant turn
        std::vector<int32_t> ed = a;
        for (int i = 0; i < 5; ++i) ed.push_back(8000 + i);
        Req a1 = turn_req(ed, 5); a1.run(s);
        uint32_t rc = 0;
        Cache ref2 = ref;
        const std::string want = text_of(serial_ref(ref2, a1.ids, a1.rq.snap_at, 6, 5, &rc));
        check(a1.res.lane == a0.res.lane && a1.res.cached_tokens == a0.rq.snap_at && m.own_restores.load() == 1 && a1.res.text == want,
              "sticky (c): a prompt diverging right after the checkpoint restores in place at it (" + std::to_string(a1.res.cached_tokens) + ") == --parallel 1");
        // a prompt that leaves the lane's history before its checkpoint: no in-place restore (a cold prefill here)
        std::vector<int32_t> early(ed.begin(), ed.begin() + 20);
        for (int i = 0; i < 30; ++i) early.push_back(8500 + i);
        Req e = turn_req(early, 6); e.run(s);
        Cache none;
        check(m.sticky.match(a0.res.lane, e.ids) == 0 && m.own_restores.load() == 1 && e.res.cached_tokens == 0 && e.res.text == text_of(serial_ref(none, e.ids, e.rq.snap_at, 6, 6)) && m.bad == 0,
              "sticky (c): a prompt diverging before the checkpoint gets no in-place restore (" + std::to_string(e.res.cached_tokens) + " cached) == --parallel 1");
        s.shutdown();
    }
    {   // (d) sticky OFF as Q35mLanesModel has it: resize() is never called (IE_Q35MOE_STICKY_LANES=0, or the load-time
        // fallback when the reserve is not kept), so the policy holds no per-lane rows. The B29 gate's segfault at 8 in occupied().
        ie::Q35mSticky off;
        const std::vector<int32_t> ids = conv(51, 100);
        off.drop(0); off.drop(15);
        check(off.end(0) == 0 && off.end(15) == 0 && off.match(0, ids) == 0,
              "sticky (d): an unsized policy (sticky off) answers end 0 / match 0 and drop is a no-op");
        FakeCrown m({4000, 4000, 4000}, true);
        m.sticky = ie::Q35mSticky{};   // exactly the engine's sticky-off state
        ie::LanesServe s(m, {});
        Cache ref;
        std::vector<int32_t> a = conv(53, 100);
        Req a0 = turn_req(a, 7); a0.run(s);
        Req a1 = turn_req(with_reply(a, a0.res.text, 9200), 8); a1.run(s);
        (void)serial_ref(ref, a0.ids, a0.rq.snap_at, 6, 7);
        const std::string want = text_of(serial_ref(ref, a1.ids, a1.rq.snap_at, 6, 8));
        check(m.own_restores.load() == 0 && a1.res.text == want && m.bad == 0,
              "sticky (d): with sticky off two turns serve without a checkpoint (0 in place) == --parallel 1");
        s.shutdown();
    }
}

// ---- P4 B34 (3): short-first ----------------------------------------------------------------------------------------------

uint64_t health_field(const ie::LanesServe& s, const std::string& k) {
    const std::string js = s.status_json();
    const size_t p = js.find("\"" + k + "\":");
    return p == std::string::npos ? UINT64_MAX : std::stoull(js.substr(p + k.size() + 3));
}

// Dream's shape (the B33 gate's b2): a decoding lane, a deep lead prefilling through the pipe (40 pieces of 5 rows; past depth
// 40 each takes deep_ms more a stage), then 6 short prompts (2-3 pieces) arriving behind it. Short = at most max(2 pieces,
// short_rows 20) rows left. ON: every short prompt's first token comes before the lead's (its pieces done), the slots guard
// lets the lead through, and every reply == --parallel 1's (each lane's pieces and their order are the plan's). OFF: the FIFO
// window and arrival order (B15-B34 (2)), reported for comparison; replies == --parallel 1's too.
// slots / wait_ms: the guard (wait test: slots never, a 15 ms wait, every piece slow). quota (P4 B36 (C)): the decode quota on
// too -- the lead also waits for the decoder's steps; the short prompts (short over their whole prompt) never do.
void test_short_first(bool on, uint32_t slots = 2, uint32_t wait_ms = 60000, int deep_from = 40, uint32_t quota = 0) {
    using Clock = std::chrono::steady_clock;
    const std::string tag = std::string("short-first ") + (on ? "ON" : "OFF") + (g_pipeline ? ", lane pipeline" : "") +
                            (wait_ms < 60000 ? ", wait guard" : "") + (quota ? ", decode quota " + std::to_string(quota) : "");
    FakeCrown m({4000, 400, 400, 400, 400, 400, 400, 400, 400}, true);
    m.deep_from = uint32_t(deep_from); m.deep_ms = 8;
    ie::LanesServe::Options o;
    o.prefill_fifo = true; o.short_first = on; o.short_rows = 20; o.short_slots = slots; o.short_wait_ms = wait_ms;
    o.decode_quota = quota; o.quota_max_ms = 60000;
    ie::LanesServe s(m, o);
    std::atomic<bool> quit{false};
    Req d(conv(12, 52000), 3000, 501, 12);
    d.quit = &quit;
    std::thread td([&] { d.run(s); });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    Req lead(conv(200, 53000), 4, 502, 200);   // through the pipe (the decoder runs): 40 pieces
    std::thread tl([&] { lead.run(s); });
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    std::vector<std::unique_ptr<Req>> sh;
    for (int k = 0; k < 6; ++k) {
        std::vector<int32_t> p = conv(9 + uint32_t(k % 3), 54000 + 1000 * k);
        sh.push_back(std::make_unique<Req>(p, 4, 510 + uint64_t(k), uint32_t(p.size())));
    }
    const auto t_sent = Clock::now();
    std::vector<std::thread> ts;
    for (auto& r : sh) ts.emplace_back([&r, &s] { r->run(s); });
    for (auto& t : ts) t.join();
    tl.join();
    quit = true;
    td.join();
    int before = 0;
    double worst = 0;
    bool same = true;
    for (auto& r : sh) {
        before += r->t_first.time_since_epoch().count() && r->t_first < lead.t_first;
        worst = std::max(worst, std::chrono::duration<double, std::milli>(r->t_first - t_sent).count());
        Cache none;
        same = same && r->res.text == text_of(serial_ref(none, r->ids, r->rq.snap_at, 4, r->rq.rng));
    }
    Cache c_lead, c_dec;
    same = same && lead.res.text == text_of(serial_ref(c_lead, lead.ids, 200, 4, 502));
    const std::string dref = text_of(serial_ref(c_dec, d.ids, 12, 3000, 501));
    same = same && dref.rfind(d.res.text, 0) == 0;
    const uint64_t bypass = health_field(s, "short_bypass"), guard = health_field(s, "short_guard"),
                   gwait = health_field(s, "short_guard_wait");
    const double lead_ms = std::chrono::duration<double, std::milli>(lead.t_first - t_sent).count();
    check(same && m.bad == 0, tag + ": the lead, the 6 short prompts and the decoder == --parallel 1's");
    if (quota) {
        const uint64_t holds = health_field(s, "quota_holds");
        check(before == 6 && holds != UINT64_MAX && holds > 0,
              tag + ": all 6 short prompts had their first token before the lead's last piece landed (" + std::to_string(before) +
                  " of 6; the slowest " + std::to_string(int(worst)) + " ms, the lead " + std::to_string(int(lead_ms)) + " ms); the quota held the lead " +
                  std::to_string(holds) + " time(s); short_bypass " + std::to_string(bypass) + ", short_guard " + std::to_string(guard));
    } else if (on && wait_ms >= 60000)
        check(before == 6 && bypass > 0 && guard > 0, tag + ": all 6 short prompts had their first token before the lead's last piece landed (" +
                                                          std::to_string(before) + " of 6; the slowest " + std::to_string(int(worst)) + " ms, the lead " +
                                                          std::to_string(int(lead_ms)) + " ms); " + std::to_string(bypass) + " short pieces went ahead of it, the guard let it through " +
                                                          std::to_string(guard) + " time(s)");
    else if (on)
        check(gwait > 0, tag + ": the guard's wait let the held lead through " + std::to_string(gwait) + " time(s) (" + std::to_string(guard) +
                             " firings, slots " + std::to_string(slots) + ", wait " + std::to_string(wait_ms) + " ms)");
    else
        check(bypass == 0 && guard == 0, tag + ": no bypass, no guard (" + std::to_string(before) + " of 6 short prompts before the lead; the slowest " +
                                             std::to_string(int(worst)) + " ms, the lead " + std::to_string(int(lead_ms)) + " ms)");
    s.shutdown();
}

// abort_all (B33) with short-first: the lead held for the short prompts, short pieces in the pipe, a decoder -- every request
// ends "abort" at once, nothing new starts on a stage, shutdown waits only for the held steps' layer boundary.
void test_short_first_abort() {
    using Clock = std::chrono::steady_clock;
    FakeCrown m({4000, 400, 400, 400, 400, 400, 400, 400, 400}, true);
    m.deep_from = 40; m.deep_ms = 40;
    ie::LanesServe::Options o;
    o.prefill_fifo = true; o.short_first = true; o.short_rows = 20; o.short_slots = 2; o.short_wait_ms = 60000;
    ie::LanesServe s(m, o);
    Req d(conv(12, 56000), 3000, 601, 12);
    std::thread td([&] { d.run(s); });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    Req lead(conv(200, 57000), 4, 602, 200);
    std::thread tl([&] { lead.run(s); });
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    std::vector<std::unique_ptr<Req>> sh;
    for (int k = 0; k < 6; ++k) { std::vector<int32_t> p = conv(9, 58000 + 1000 * k); sh.push_back(std::make_unique<Req>(p, 4, 610 + uint64_t(k), 9)); }
    std::vector<std::thread> ts;
    for (auto& r : sh) ts.emplace_back([&r, &s] { r->run(s); });
    // (P4 B39: the lead's piece in flight is no longer drained for each short prompt's turns, so a hold -- short_bypass > 0 -- is
    // not guaranteed before the fake's short prompts finish; the stop comes once the short prompts are admitted beside the lead)
    for (int i = 0; i < 3000 && health_field(s, "short_bypass") == 0 && health_field(s, "lanes_active") < 7; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const auto t0 = Clock::now();
    const uint32_t held = s.abort_all();
    const double abort_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    for (auto& t : ts) t.join();
    tl.join(); td.join();
    const double join_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    s.shutdown();
    const double down_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    bool all_abort = d.res.finish_reason == "abort" && lead.res.finish_reason == "abort";
    std::string fins = "decoder " + d.res.finish_reason + ", lead " + lead.res.finish_reason + ", short:";
    for (auto& r : sh) { all_abort = all_abort && (r->res.finish_reason == "abort" || r->res.finish_reason == "length"); fins += " " + r->res.finish_reason; }
    check(abort_ms < 50 && join_ms < 300, "short-first abort: abort_all does not block (" + std::to_string(int(abort_ms)) + " ms), every request returned (" +
                                              std::to_string(int(join_ms)) + " ms)");
    check(all_abort, "short-first abort: the decoder and the held lead end abort; each short prompt ends abort or had finished (" + fins + ")");
    check(held <= 2 && m.calls_after_abort.load() <= held, "short-first abort: nothing started on a stage after the stop beyond the " + std::to_string(held) +
                                                             " held step(s) (" + std::to_string(m.calls_after_abort.load()) + " calls after)");
    check(down_ms < 400 && m.bad == 0, "short-first abort: shutdown waited only for the held steps (" + std::to_string(int(down_ms)) + " ms); no wrong position");
}

// ---- P4 B36 (A): re-cut ---------------------------------------------------------------------------------------------------

// The reply of a lane whose prompt went through exactly `pieces` (stage 1's commits in order: the fake's hash sees every piece
// boundary), then the greedy decode (serial_ref's tail). Rows past the prompt (decode steps) are skipped. {} when the pieces do
// not cover the prompt contiguously.
std::vector<int32_t> ref_over(const std::vector<int32_t>& ids, const std::vector<std::pair<uint32_t, uint32_t>>& pieces, uint32_t n,
                              uint64_t rng) {
    uint64_t h = kSeed;
    uint32_t pos = 0;
    for (const auto& [p0, t] : pieces) {
        if (p0 >= ids.size()) continue;
        if (p0 != pos) return {};
        h = step_hash(h, ids.data() + p0, t, p0);
        pos = p0 + t;
    }
    if (pos != ids.size()) return {};
    std::vector<int32_t> out;
    for (uint32_t step = 0; step < n; ++step) {
        const int32_t id = id_of(h, rng + step);
        out.push_back(id);
        h = step_hash(h, &id, 1, pos); ++pos;
    }
    return out;
}

// Dream's shape: a lead that started ALONE (the serial turn: its plan in kChunk-row pieces, every row 2 ms a stage) is joined by
// a worker that prefills 9 rows (mix_chunk 2 pieces: it started beside the lead) and then decodes. Re-cut ON: once the worker is
// active, the lead's pieces not yet in the pipe become mix_chunk-row pieces, each plan chunk cut in order, so between two of the
// worker's decode steps card 0 runs a cut piece or two of the lead; OFF (B29-B34): a whole plan piece. Both replies == a run
// over the same pieces: the lead's over its recorded cut sequence (self-consistent), the worker's over its prep-time cut.
struct RecutRun { uint32_t max_rows_between = 0, worker_steps = 0, lead_rows = 0; };
RecutRun test_recut(bool on) {
    const std::string tag = std::string("re-cut ") + (on ? "ON" : "OFF");
    FakeCrown m({4000, 400, 400}, true);
    m.deep_from = 0; m.deep_row_ms = 2;
    m.record = true;
    ie::LanesServe::Options o;
    o.prefill_fifo = true; o.short_first = true; o.short_rows = 20; o.mix_chunk = 2; o.recut = on;
    ie::LanesServe s(m, o);
    Req lead(conv(200, 61000), 4, 701, 200);   // 40 plan pieces of 5 rows
    std::thread tl([&] { lead.run(s); });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));   // ~2 of its pieces in its serial turn (20 ms each)
    std::atomic<bool> quit{false};
    Req w(conv(9, 62000), 1000, 702, 9);
    w.quit = &quit;
    std::thread tw([&] { w.run(s); });
    tl.join();
    quit = true;
    tw.join();
    std::vector<std::pair<uint32_t, uint32_t>> lp, wp;
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> log0;
    {
        std::lock_guard<std::mutex> g(m.rec_mu);
        lp = m.pieces[lead.res.lane]; wp = m.pieces[w.res.lane]; log0 = m.s0_log;
    }
    bool cut = false, shape = !lp.empty() && lp[0].second == kChunk;   // (its serial turn ran plan pieces first)
    for (const auto& [p0, t] : lp) {
        if (p0 >= 200) continue;
        if (t != kChunk) cut = true;
        if (cut && (t > o.mix_chunk || p0 / kChunk != (p0 + t - 1) / kChunk)) shape = false;   // mix_chunk pieces inside plan chunks
    }
    RecutRun r;
    const uint32_t ll = lead.res.lane, wl = w.res.lane;
    size_t last_lead = 0;
    for (size_t i = 0; i < log0.size(); ++i) if (std::get<0>(log0[i]) == ll && std::get<1>(log0[i]) < 200) last_lead = i;
    bool stepped = false;
    uint32_t rows = 0;
    for (size_t i = 0; i < log0.size(); ++i) {
        const auto [ln, p0, t] = log0[i];
        if (ln == ll && p0 < 200) { rows += t; r.lead_rows += t; }
        else if (ln == wl && p0 >= 9) {   // a decode row of the worker
            if (stepped) r.max_rows_between = std::max(r.max_rows_between, rows);
            stepped = true; rows = 0;
            if (i < last_lead) ++r.worker_steps;
        }
    }
    const std::string lref = text_of(ref_over(lead.ids, lp, 4, 701)), wref = text_of(ref_over(w.ids, wp, 1000, 702));
    check(lead.res.text == lref && !w.res.text.empty() && wref.rfind(w.res.text, 0) == 0 && m.bad == 0,
          tag + ": the lead's reply == a run over its own piece sequence, the worker's == a run over its pieces");
    if (on)
        check(cut && shape && r.max_rows_between <= 2 * o.mix_chunk && health_field(s, "recuts") >= 1,
              tag + ": once the worker was active the lead's pieces were cut to " + std::to_string(o.mix_chunk) + " rows inside its plan chunks; at most " +
                  std::to_string(r.max_rows_between) + " lead rows between two decode steps on card 0 (" + std::to_string(r.worker_steps) +
                  " steps beside " + std::to_string(r.lead_rows) + " lead rows)");
    else
        check(!cut && health_field(s, "recuts") == 0, tag + ": the lead kept its plan pieces; up to " + std::to_string(r.max_rows_between) +
                                                         " lead rows between two decode steps on card 0 (" + std::to_string(r.worker_steps) +
                                                         " steps beside " + std::to_string(r.lead_rows) + " lead rows)");
    s.shutdown();
    return r;
}

// abort_all (B33) in the middle of a re-cut lead beside a decoding worker: both end abort at once, nothing new starts on a
// stage, shutdown waits only for the held steps' layer boundary.
void test_recut_abort() {
    using Clock = std::chrono::steady_clock;
    FakeCrown m({4000, 400, 400}, true);
    m.deep_from = 0; m.deep_row_ms = 4;
    ie::LanesServe::Options o;
    o.prefill_fifo = true; o.short_first = true; o.short_rows = 20; o.mix_chunk = 2; o.recut = true;
    ie::LanesServe s(m, o);
    Req lead(conv(200, 63000), 4, 711, 200);
    std::thread tl([&] { lead.run(s); });
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    Req w(conv(9, 64000), 3000, 712, 9);
    std::thread tw([&] { w.run(s); });
    for (int i = 0; i < 3000 && !(health_field(s, "recuts") >= 1 && health_field(s, "decoding") >= 1); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const auto t0 = Clock::now();
    const uint32_t held = s.abort_all();
    const double abort_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    tl.join(); tw.join();
    const double join_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    s.shutdown();
    const double down_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    check(lead.res.finish_reason == "abort" && w.res.finish_reason == "abort" && abort_ms < 50 && join_ms < 300,
          "re-cut abort: the lead (re-cut) and the decoding worker end abort at once (abort_all " + std::to_string(int(abort_ms)) +
              " ms, returned " + std::to_string(int(join_ms)) + " ms)");
    check(held <= 2 && m.calls_after_abort.load() <= held && down_ms < 400 && m.bad == 0,
          "re-cut abort: nothing new on a stage after the stop beyond the " + std::to_string(held) + " held step(s); shutdown " +
              std::to_string(int(down_ms)) + " ms");
}

// ---- P4 B36 (C): the decode quota -----------------------------------------------------------------------------------------

using Ev = std::tuple<char, uint32_t, uint32_t, uint32_t>;
// From FakeCrown::ev: for every lead prefill submit ('S', p0 < lead_T) after one of the lead's landings ('L', its done callback's
// end), the decode rows each decoder (lane, prompt length) committed ('1') between that landing -- the lead's latest before the
// submit -- and the submit: what the serve counted when it decided (a commit whose callback had not run yet counts here too, so
// the check can only be lenient). Returns the minimum over those submits and the decoders (UINT32_MAX: none); n = the submits
// checked; dec_rows = the decoders' rows from the lead's first piece's start to its last commit. K > 0: the events behind the
// first submit short of K are printed.
uint32_t quota_min(const std::vector<Ev>& ev, uint32_t lead, uint32_t lead_T, const std::vector<std::pair<uint32_t, uint32_t>>& decs,
                   uint32_t& n, uint32_t& dec_rows, uint32_t K = 0) {
    uint32_t mn = UINT32_MAX;
    n = 0; dec_rows = 0;
    size_t first0 = SIZE_MAX, last1 = 0, land_at = 0;
    for (size_t i = 0; i < ev.size(); ++i) {
        const auto [k, ln, p0, t] = ev[i];
        if (ln != lead || p0 >= lead_T) continue;
        if (k == '0' && first0 == SIZE_MAX) first0 = i;
        if (k == '1') last1 = i;
    }
    bool landed = false, shown = false;
    std::vector<uint32_t> since(decs.size(), 0);
    for (size_t i = 0; i < ev.size(); ++i) {
        const auto [k, ln, p0, t] = ev[i];
        if (ln == lead) {
            if (k == 'L') { landed = true; land_at = i; std::fill(since.begin(), since.end(), 0u); }
            if (k == 'S' && p0 < lead_T && landed) {
                ++n;
                const uint32_t m = *std::min_element(since.begin(), since.end());
                mn = std::min(mn, m);
                if (K && m < K && !shown) {
                    shown = true;
                    std::printf("  first lead submit short of the quota (lead lane %u): events %zu..%zu:", lead, land_at, i);
                    for (size_t j = land_at; j <= i && j < land_at + 60; ++j)
                        std::printf(" %c:%u@%u+%u", std::get<0>(ev[j]), std::get<1>(ev[j]), std::get<2>(ev[j]), std::get<3>(ev[j]));
                    std::printf("\n");
                }
            }
            continue;
        }
        for (size_t d = 0; d < decs.size(); ++d)
            if (k == '1' && ln == decs[d].first && p0 >= decs[d].second) { ++since[d]; dec_rows += i > first0 && i < last1; }
    }
    return mn;
}

// A deep lead (200 rows, every row 2 ms a stage) that starts beside 2 decoding lanes (its prep cuts it to mix_chunk 2 pieces),
// with the production defaults (rows or not, the lane pipeline, short-first, re-cut). Quota K: before each lead piece starts on
// card 0 -- a lookahead too -- every decoder committed >= K steps since the lead's latest piece landed; OFF (K 0, B36 (A)):
// about one a piece. The lead's reply == a run over its own pieces; each decoder's == a run over its pieces.
struct QuotaRun { uint32_t min_steps = 0, pieces = 0; double per_piece = 0, lead_ms = 0; std::string lead_text; };
QuotaRun test_quota(uint32_t K, bool rows) {
    using Clock = std::chrono::steady_clock;
    const std::string tag = "decode quota " + std::to_string(K) + (rows ? " (rows)" : " (one lane a step)");
    FakeCrown m({4000, 4000, 4000}, rows);
    m.deep_from = 0; m.deep_row_ms = 2;
    m.record = true;
    ie::LanesServe::Options o;
    o.prefill_fifo = true; o.short_first = true; o.short_rows = 20; o.mix_chunk = 2; o.recut = true;
    o.decode_quota = K; o.quota_max_ms = 60000;
    ie::LanesServe s(m, o);
    std::atomic<bool> quit{false};
    Req d1(conv(9, 65000), 3000, 721, 9), d2(conv(11, 66000), 3000, 722, 11);
    d1.quit = &quit; d2.quit = &quit;
    std::thread t1([&] { d1.run(s); }), t2([&] { d2.run(s); });
    for (int i = 0; i < 3000 && health_field(s, "decoding") < 2; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    Req lead(conv(200, 67000), 4, 723, 200);
    const auto t0 = Clock::now();
    std::thread tl([&] { lead.run(s); });
    tl.join();
    QuotaRun r;
    r.lead_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    quit = true;
    t1.join(); t2.join();
    std::vector<Ev> ev;
    std::vector<std::pair<uint32_t, uint32_t>> lp, p1, p2;
    {
        std::lock_guard<std::mutex> g(m.rec_mu);
        ev = m.ev; lp = m.pieces[lead.res.lane]; p1 = m.pieces[d1.res.lane]; p2 = m.pieces[d2.res.lane];
    }
    uint32_t dec_rows = 0, lead_pieces = 0;
    r.min_steps = quota_min(ev, lead.res.lane, 200, {{d1.res.lane, 9u}, {d2.res.lane, 11u}}, r.pieces, dec_rows, K);
    for (const auto& pc : lp) lead_pieces += pc.first < 200;
    r.per_piece = lead_pieces ? double(dec_rows) / 2.0 / lead_pieces : 0;
    r.lead_text = lead.res.text;
    const std::string lref = text_of(ref_over(lead.ids, lp, 4, 723));
    const std::string r1 = text_of(ref_over(d1.ids, p1, 3000, 721)), r2 = text_of(ref_over(d2.ids, p2, 3000, 722));
    check(lead.res.text == lref && !d1.res.text.empty() && r1.rfind(d1.res.text, 0) == 0 && !d2.res.text.empty() &&
              r2.rfind(d2.res.text, 0) == 0 && m.bad == 0,
          tag + ": the lead's reply == a run over its own pieces, each decoder's == a run over its pieces");
    const uint64_t holds = health_field(s, "quota_holds"), guard = health_field(s, "quota_guard");
    const std::string seen = "fewest decoder steps between a lead landing and its next piece " +
                             (r.pieces ? std::to_string(r.min_steps) : std::string("-")) + " over " + std::to_string(r.pieces) + " pieces; " +
                             std::to_string(r.per_piece).substr(0, 4) + " steps a decoder per lead piece; the lead " + std::to_string(int(r.lead_ms)) + " ms";
    if (K)
        check(r.pieces > 0 && r.min_steps >= K && holds != UINT64_MAX && holds > 0 && guard == 0,
              tag + ": " + seen + "; held " + std::to_string(holds) + " time(s), guard " + std::to_string(guard));
    else
        check(holds == 0 && guard == 0, tag + " (off): " + seen);
    s.shutdown();
    return r;
}

// The guard: the decoder stalls (each decode step takes 60 ms more a stage: 4 steps ~0.5 s), quota_max_ms 20 -- the held lead
// takes its next piece once the guard's wait passed, without the decoder's 4 steps: some submits went out with fewer than 4 decode
// steps since the lead's latest landing (with the quota alone none can), the guard counted them, and nothing hangs.
void test_quota_guard() {
    using Clock = std::chrono::steady_clock;
    FakeCrown m({4000, 4000, 4000}, true);
    m.deep_from = 0; m.deep_row_ms = 2; m.dec_ms = 60;
    m.record = true;
    ie::LanesServe::Options o;
    o.prefill_fifo = true; o.short_first = true; o.short_rows = 20; o.mix_chunk = 2; o.recut = true;
    o.decode_quota = 4; o.quota_max_ms = 20;
    ie::LanesServe s(m, o);
    std::atomic<bool> quit{false};
    Req d(conv(9, 68000), 3000, 731, 9);
    d.quit = &quit;
    std::thread td([&] { d.run(s); });
    for (int i = 0; i < 3000 && health_field(s, "decoding") < 1; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    Req lead(conv(40, 69000), 4, 732, 40);
    const auto t0 = Clock::now();
    std::thread tl([&] { lead.run(s); });
    tl.join();
    const double lead_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    quit = true;
    td.join();
    std::vector<std::pair<uint32_t, uint32_t>> lp;
    std::vector<Ev> ev;
    { std::lock_guard<std::mutex> g(m.rec_mu); lp = m.pieces[lead.res.lane]; ev = m.ev; }
    uint32_t n = 0, dec_rows = 0;
    const uint32_t mn = quota_min(ev, lead.res.lane, 40, {{d.res.lane, 9u}}, n, dec_rows);
    const uint64_t holds = health_field(s, "quota_holds"), guard = health_field(s, "quota_guard");
    check(holds != UINT64_MAX && holds > 0 && guard != UINT64_MAX && guard > 0 && n > 0 && mn < 4 && lead_ms < 8000 &&
              lead.res.text == text_of(ref_over(lead.ids, lp, 4, 732)) && m.bad == 0,
          "decode quota guard: the stalled decoder's steps were not waited for -- the guard let the lead through " + std::to_string(guard) +
              " of " + std::to_string(holds) + " hold(s); fewest decode steps before a lead submit " + std::to_string(mn) + " of 4 (" +
              std::to_string(n) + " submits); the lead took " + std::to_string(int(lead_ms)) + " ms; its reply == a run over its pieces (the decoder: " +
              d.res.finish_reason + ", " + std::to_string(d.res.completion_tokens) + " tokens)");
    s.shutdown();
}

// abort_all (B33) with the lead held for the decoders' quota: every request ends abort at once, nothing new starts on a stage,
// shutdown waits only for the held steps' layer boundary.
void test_quota_abort() {
    using Clock = std::chrono::steady_clock;
    FakeCrown m({4000, 4000, 4000}, true);
    m.deep_from = 0; m.deep_row_ms = 4;
    ie::LanesServe::Options o;
    o.prefill_fifo = true; o.short_first = true; o.short_rows = 20; o.mix_chunk = 2; o.recut = true;
    o.decode_quota = 4; o.quota_max_ms = 60000;
    ie::LanesServe s(m, o);
    Req d1(conv(9, 70000), 3000, 741, 9), d2(conv(11, 71000), 3000, 742, 11);
    std::thread t1([&] { d1.run(s); }), t2([&] { d2.run(s); });
    for (int i = 0; i < 3000 && health_field(s, "decoding") < 2; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    Req lead(conv(200, 72000), 4, 743, 200);
    std::thread tl([&] { lead.run(s); });
    for (int i = 0; i < 3000; ++i) {
        const uint64_t h = health_field(s, "quota_holds");
        if (h != UINT64_MAX && h >= 3) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const auto t0 = Clock::now();
    const uint32_t held = s.abort_all();
    const double abort_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    tl.join(); t1.join(); t2.join();
    const double join_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    s.shutdown();
    const double down_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    check(lead.res.finish_reason == "abort" && d1.res.finish_reason == "abort" && d2.res.finish_reason == "abort" && abort_ms < 50 && join_ms < 300,
          "decode quota abort: the held lead and both decoders end abort at once (abort_all " + std::to_string(int(abort_ms)) + " ms, returned " +
              std::to_string(int(join_ms)) + " ms)");
    check(held <= 2 && m.calls_after_abort.load() <= held && down_ms < 400 && m.bad == 0,
          "decode quota abort: nothing new on a stage after the stop beyond the " + std::to_string(held) + " held step(s); shutdown " +
              std::to_string(int(down_ms)) + " ms");
}

// ---- P4 B36 (B): the anchor at the last user query's end ------------------------------------------------------------------

// A Dream task's prompt as the Qwen thinking templates render it: the system prompt S, the user's query Q1, then per tool round
// an assistant turn (its reasoning R + its call C) and the tool result TR, then the generation prompt. While Q1 is the last query
// every assistant turn keeps its reasoning; once a new query Q2 exists the template drops the reasoning of the turns before it,
// so the new prompt leaves the old one right after Q1 -- the re-render behind the B35 gate's round 5 restoring nothing.
struct Task {
    std::vector<int32_t> S, Q1, Q2;
    std::vector<std::vector<int32_t>> R, C, TR;
    explicit Task(int32_t base) : S(conv(20, base)), Q1(conv(10, base + 1000)), Q2(conv(8, base + 2000)) {
        for (int k = 0; k < 4; ++k) {
            R.push_back(conv(6, base + 3000 + 100 * k)); C.push_back(conv(4, base + 4000 + 100 * k)); TR.push_back(conv(8, base + 5000 + 100 * k));
        }
    }
    // rounds 1-4: under Q1 (round - 1 tool rounds before it); round 5: the new query Q2 after the 4 rounds
    std::vector<int32_t> prompt(int round) const {
        std::vector<int32_t> p = S;
        p.insert(p.end(), Q1.begin(), Q1.end());
        for (int k = 0; k < (round <= 4 ? round - 1 : 4); ++k) {
            if (round <= 4) p.insert(p.end(), R[size_t(k)].begin(), R[size_t(k)].end());
            p.insert(p.end(), C[size_t(k)].begin(), C[size_t(k)].end());
            p.insert(p.end(), TR[size_t(k)].begin(), TR[size_t(k)].end());
        }
        if (round == 5) p.insert(p.end(), Q2.begin(), Q2.end());
        p.push_back(3); p.push_back(4);
        return p;
    }
    static bool ends_with_query(int round) { return round == 1 || round == 5; }   // (Engine::chat's rule: the last turn is the user's)
    uint32_t anchor() const { return uint32_t(S.size() + Q1.size()); }
};

std::unique_ptr<Req> task_round(ie::LanesServe& s, const Task& t, int round, uint64_t rng) {
    const std::vector<int32_t> ids = t.prompt(round);
    auto r = std::make_unique<Req>(ids, 4, rng, uint32_t(ids.size() - 2));
    r->rq.anchor = Task::ends_with_query(round);
    r->run(s);
    return r;
}

// One task on a cache with B29's supersede. ON: round 1's snapshot (its prompt ends with Q1: the anchor) outlives rounds 2-4's
// supersedes, so the new query restores the old query's end; OFF: round 2 superseded it and round 5 restores nothing. Rounds 2-4
// restore the round before either way, and every reply == --parallel 1's given the same cache rules.
void test_anchor(bool on) {
    const std::string tag = std::string("anchor ") + (on ? "ON" : "OFF");
    FakeCrown m({400, 400, 400}, true);
    m.cache.supersede = true; m.anchor_on = on;
    ie::LanesServe s(m, {});
    Cache ref;
    ref.supersede = true;
    const Task task(80000);
    bool same = true, prev_ok = true;
    uint32_t prev_snap = 0, r5 = 0;
    for (int rnd = 1; rnd <= 5; ++rnd) {
        const auto r = task_round(s, task, rnd, 800 + uint64_t(rnd));
        const uint32_t cached = r->res.cached_tokens;
        uint32_t rc = 0;
        same = same && r->res.text == text_of(serial_ref(ref, r->ids, r->rq.snap_at, 4, r->rq.rng, &rc, false, 0, on && r->rq.anchor)) &&
               cached == rc;
        if (rnd >= 2 && rnd <= 4) prev_ok = prev_ok && cached == prev_snap;
        if (rnd == 5) r5 = cached;
        prev_snap = r->rq.snap_at;
    }
    check(same && m.bad == 0, tag + ": every round == --parallel 1's given the same cache rules");
    check(prev_ok, tag + ": tool rounds 2-4 each restored the round before");
    if (on) check(r5 == task.anchor(), tag + ": the new query restored " + std::to_string(r5) + " tokens = the old query's end (" +
                                           std::to_string(task.anchor()) + ")");
    else check(r5 == 0, tag + ": the new query restored " + std::to_string(r5) + " tokens (round 2 superseded round 1's snapshot)");
    s.shutdown();
}

// Three tasks on a 5-entry cache: A and B run 4 tool rounds and a new query, interleaved; C stops after 2 rounds. A restore also
// refreshes the anchors on its path (FleetPrefixCache::find_longest_match), so an active task's anchor outlives an idle task's
// entries under the cap, and A's and B's new queries restore their old queries' ends.
void test_anchor_lru() {
    FakeCrown m({400, 400, 400}, true);
    m.cache.supersede = true; m.cache.cap = 5;
    ie::LanesServe s(m, {});
    const Task A(81000), B(82000), C(83000);
    task_round(s, A, 1, 1); task_round(s, B, 1, 2); task_round(s, C, 1, 3); task_round(s, C, 2, 4);
    for (int rnd = 2; rnd <= 4; ++rnd) { task_round(s, A, rnd, 10 + uint64_t(rnd)); task_round(s, B, rnd, 20 + uint64_t(rnd)); }
    const uint32_t a5 = task_round(s, A, 5, 15)->res.cached_tokens, b5 = task_round(s, B, 5, 25)->res.cached_tokens;
    check(a5 == A.anchor() && b5 == B.anchor() && m.bad == 0,
          "anchor LRU: the active tasks' new queries restored " + std::to_string(a5) + " / " + std::to_string(b5) + " tokens (their old queries' "
              "ends) on a 5-entry cache beside an idle task");
    s.shutdown();
}

// ---- P4 B39: the serial turns beside the running pipe ---------------------------------------------------------------------

// The plan with the tail through the pipe (q35m_plan pipe_tail): the pieces are --parallel 1's prefill_to(share), prefill_to(snap),
// prefill_to(T) exactly (the tail's pieces = prompt_end's pf_chunk-cut rest), Tp = T, snap = snap_at exactly when reused < snap_at
// < T, mark as before, and a chunk ends at the snap.
void test_pipe_tail_plan() {
    bool all = true, bounds = true, ends = true;
    for (uint32_t T = 2; T <= 34 && all; ++T)
        for (uint32_t snap = 1; snap <= T && all; ++snap)
            for (uint32_t share = 0; share < snap && all; ++share)
                for (uint32_t reused = 0; reused < T && all; ++reused)
                    for (uint32_t pf : {1u, 3u, 8u}) {
                        std::vector<ie::LanesChunk> ref;
                        uint32_t pos = reused;
                        auto to = [&](uint32_t end) { while (pos < end) { const uint32_t k = std::min(pf, end - pos); ref.emplace_back(pos, k); pos += k; } };
                        if (share > reused) to(share);
                        to(snap); to(T);
                        const ie::LanesPlan old = ie::q35m_plan(T, snap, reused, pf, share);
                        const ie::LanesPlan p = ie::q35m_plan(T, snap, reused, pf, share, /*pipe_tail=*/true);
                        if (p.chunks != ref || p.Tp != T) { all = false; std::printf("  pipe-tail plan mismatch T %u snap %u share %u reused %u pf %u\n", T, snap, share, reused, pf); }
                        const bool due = snap > reused && snap < T;
                        if ((p.snap != 0) != due || (due && p.snap != snap) || p.mark != old.mark) bounds = false;
                        // the old plan's pieces + prompt_end's rest == the new plan's pieces (the same cuts in the same order)
                        std::vector<ie::LanesChunk> joined = old.chunks;
                        ie::q35m_chunks(old.Tp, T, pf, joined);
                        if (joined != p.chunks) all = false;
                        if (due) {
                            bool e = false;
                            for (const auto& c : p.chunks) e = e || c.first + c.second == snap;
                            if (!e) ends = false;
                        }
                    }
    check(all, "pipe-tail plan: pieces == --parallel 1's prefill_to(share), prefill_to(snap), prefill_to(T) == the old plan + prompt_end's rest, Tp = T (T <= 34)");
    check(bounds && ends, "pipe-tail plan: snap == snap_at exactly when reused < snap_at < T (a chunk ends there), mark as before");
}

// The decoders keep stepping through other lanes' serial turns. Production shape (rows, the lane pipeline, short-first, re-cut,
// the quota, sticky lanes): conversation C's turn 1 alone (its checkpoint on its lane), then two decoders; then, beside them,
// request A ends with a reply snapshot (its finish turn) while C's turn 2 restores in place (prefix_prepare), prefills through
// the pipe with its snapshot turn at snap_at and its prompt-end turn, and decodes. Every serial hook takes 150 ms. Drain OFF
// (the default): the decoders' longest gap between two steps stays well under a turn (nothing drained: /health drains 0,
// turns_nodrain > 0); ON (IE_Q35MOE_TURN_DRAIN=1): each turn drains the pipe and the decoders stall for it (a gap >= 150 ms,
// drains > 0). Both: every reply == --parallel 1's, A's snapshot and C's restore happened, no stage saw a wrong position.
void test_turn_nodrain(bool drain) {
    const bool saved = g_turn_drain;
    g_turn_drain = drain;
    const std::string tag = std::string("turns beside the pipe, drain ") + (drain ? "ON (IE_Q35MOE_TURN_DRAIN=1)" : "OFF");
    FakeCrown m({4000, 4000, 4000, 4000, 4000}, true);
    m.sticky.on = true; m.cache.supersede = true;
    ie::LanesServe::Options o;   // (no mix cut: the plan's pieces, so --parallel 1's reference applies; the cut is test_mix_chunk's)
    o.prefill_fifo = true; o.short_first = true; o.short_rows = 20; o.decode_quota = 4; o.quota_max_ms = 60000;
    ie::LanesServe s(m, o);
    Cache refC, refA;
    std::vector<int32_t> c = conv(31, 90000);
    Req c1 = turn_req(c, 901); c1.run(s);
    check(c1.res.text == text_of(serial_ref(refC, c1.ids, c1.rq.snap_at, 6, 901)), tag + ": C's turn 1 alone == --parallel 1");
    std::atomic<bool> quit{false};
    Req d1(conv(12, 91000), 3000, 902, 12), d2(conv(13, 92000), 3000, 903, 13);
    d1.quit = &quit; d2.quit = &quit;
    std::thread t1([&] { d1.run(s); }), t2([&] { d2.run(s); });
    for (int i = 0; i < 3000 && health_field(s, "decoding") < 2; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    m.turn_ms = 150; m.record_dec = true;
    const uint64_t drains0 = health_field(s, "drains");
    // A: a new prompt whose reply stops after 3 ids and is snapshotted (its finish turn); C2: C's turn 2 (restored in place)
    std::vector<int32_t> pa = conv(40, 93000);
    Req a(pa, 20, 904, 38);
    a.rq.reply_cache = true;
    g_stop_first = pa[0]; g_stop_at = uint32_t(pa.size()) + 3;
    Req c2 = turn_req(with_reply(c, c1.res.text, 9300), 905);
    std::thread ta([&] { a.run(s); }), tc([&] { c2.run(s); });
    ta.join(); tc.join();
    const std::string wa = text_of(serial_ref(refA, a.ids, a.rq.snap_at, 20, 904, nullptr, true));   // (under A's stop rule)
    g_stop_at = 0; g_stop_first = -1;
    m.record_dec = false;
    quit = true;
    t1.join(); t2.join();
    const uint64_t drains = health_field(s, "drains") - drains0, nodrain = health_field(s, "turns_nodrain");
    // the decoders' longest gap between two consecutive decode steps while A and C2 ran
    double gap = 0;
    {
        std::lock_guard<std::mutex> g(m.rec_mu);
        for (uint32_t ln : {d1.res.lane, d2.res.lane}) {
            const auto& ts = m.dec_t0[ln];
            for (size_t i = 1; i < ts.size(); ++i) gap = std::max(gap, ts[i] - ts[i - 1]);
        }
    }
    uint32_t rc2 = 0;
    const std::string wc2 = text_of(serial_ref(refC, c2.ids, c2.rq.snap_at, 6, 905, &rc2));
    Cache n1, n2;
    const bool decs = text_of(serial_ref(n1, d1.ids, 12, 3000, 902)).rfind(d1.res.text, 0) == 0 &&
                      text_of(serial_ref(n2, d2.ids, 13, 3000, 903)).rfind(d2.res.text, 0) == 0 && !d1.res.text.empty() && !d2.res.text.empty();
    std::printf("  (%s: A %s %u ids '%s' want '%s'; C2 lane %u cached %u '%s' want '%s' (ref %u); d1 %s %u ids, d2 %s %u ids; bad %u, own restores %u, finishes %d)\n",
                tag.c_str(), a.res.finish_reason.c_str(), a.res.completion_tokens, a.res.text.c_str(), wa.c_str(), c2.res.lane, c2.res.cached_tokens,
                c2.res.text.c_str(), wc2.c_str(), rc2, d1.res.finish_reason.c_str(), d1.res.completion_tokens, d2.res.finish_reason.c_str(),
                d2.res.completion_tokens, m.bad.load(), m.own_restores.load(), m.finishes.load());
    check(a.res.finish_reason == "stop" && a.res.completion_tokens == 3 && a.res.text == wa && m.finishes.load() == 1,
          tag + ": A stopped after 3 ids == --parallel 1, its reply snapshot taken in its finish turn");
    check(c2.res.text == wc2 && c2.res.cached_tokens == rc2 && rc2 == c1.rq.snap_at && m.own_restores.load() == 1,
          tag + ": C's turn 2 restored in place at " + std::to_string(c2.res.cached_tokens) + " beside the decoders == --parallel 1");
    check(decs && m.bad == 0, tag + ": the decoders' replies are prefixes of --parallel 1's; no stage saw a wrong position");
    const std::string seen = "longest decoder gap " + std::to_string(int(gap)) + " ms; drains " + std::to_string(drains) + ", turns beside the pipe " +
                             std::to_string(nodrain) + ", snapshot turns " + std::to_string(m.snap_turns.load());
    // (snapshot turns: C1's in its lone prefill, then A's and C2's at their snap boundaries)
    if (!drain)
        check(gap < 100 && drains == 0 && nodrain >= 4 && m.snap_turns.load() == 3,
              tag + ": the decoders kept stepping through A's finish and C2's prepare / snapshot / prompt-end turns (" + seen + ")");
    else
        check(gap >= 150 && drains >= 2 && m.snap_turns.load() == 0,
              tag + ": every turn drained the pipe and the decoders waited for it (" + seen + ")");
    s.shutdown();
    g_turn_drain = saved;
}

// The wave on a running pipe (the B39 gate's finding): a leader and N followers with one shared prefix arrive together while the
// pipe runs (a request finished before them). The leader prepares alone and prefills its first piece in its turn; when the
// followers wait for the turn it stops, and its remaining pieces go through the pipe while the followers' prepare turns (each a
// cache MISS -- the leader's mark is not inserted yet -- then parked in the FIFO on extends_mark) chain, every one turn_ms long.
// The leader's piece ending at the mark lands early in that chain and the leader waits for ITS mark turn. Without priority the mark
// turn queues behind every follower prepare still waiting (the followers cannot restore anything before it, so every turn they
// hold delays the whole wave); with it the mark runs right after the prepare holding the turn. Checked: follower prepares that
// BEGIN after the leader's mark piece landed and before its mark ran <= 1 (the one that already held the turn, or one that took
// it in the gap before the leader registered), the followers all restore the shared prefix (their re-prepare after the mark),
// every reply == a cold server's (the restore changes no byte), /health turns_boundary_first >= 1.
void test_wave_priority() {
    constexpr uint32_t kSys = 23;
    constexpr int kN = 8;
    const std::vector<int32_t> sys = conv(kSys, 95000);
    auto wave_req = [&](int k, uint64_t rng) {
        std::vector<int32_t> ids = sys;
        for (int32_t v : conv(10, 96000 + 100 * k)) ids.push_back(v);
        const uint32_t snap = uint32_t(ids.size());
        ids.push_back(3); ids.push_back(5);
        return std::make_unique<Req>(ids, 4, rng, snap, kSys);
    };
    std::vector<uint32_t> caps(kN + 2, 4000);
    FakeCrown m(caps, true);
    m.sticky.on = true; m.cache.supersede = true;
    ie::LanesServe::Options o;
    o.prefill_fifo = true; o.short_first = true; o.short_rows = 20; o.decode_quota = 4; o.quota_max_ms = 60000;
    ie::LanesServe s(m, o);
    Req warm(conv(9, 94000), 3, 940, 9);   // the pipe runs (idle) when the wave arrives, as after the replay's workers phase
    warm.run(s);
    m.turn_ms = 20; m.land_at = kSys; m.record_turns = true;
    auto leader = wave_req(0, 950);
    std::thread tl([&] { leader->run(s); });
    std::this_thread::sleep_for(std::chrono::milliseconds(5));   // (inside the leader's prepare: its `alone` was decided already)
    std::vector<std::unique_ptr<Req>> fs;
    for (int k = 1; k <= kN; ++k) fs.push_back(wave_req(k, 950 + uint64_t(k)));
    std::vector<std::thread> ts;
    for (auto& f : fs) ts.emplace_back([&f, &s] { f->run(s); });
    tl.join();
    for (auto& t : ts) t.join();
    m.record_turns = false; m.land_at = 0;
    std::vector<std::tuple<char, uint32_t, double>> ev;
    { std::lock_guard<std::mutex> g(m.rec_mu); ev = m.turns; }
    const uint32_t ll = leader->res.lane;
    double t_land = -1, t_mark = -1;
    for (const auto& [k, ln, t] : ev) {
        if (ln != ll) continue;
        if (k == 'L' && t_land < 0) t_land = t;
        if (k == 'M' && t_mark < 0) t_mark = t;
    }
    int between = 0, before = 0;
    for (const auto& [k, ln, t] : ev)
        if (k == 'P' && ln != ll) { if (t > t_land && t < t_mark) ++between; else if (t < t_land) ++before; }
    int restored = 0;
    bool cold = true;
    for (auto& f : fs) {
        restored += f->res.cached_tokens == kSys;
        Cache none;
        cold = cold && f->res.text == text_of(serial_ref(none, f->ids, f->rq.snap_at, 4, f->rq.rng, nullptr, false, kSys));
    }
    Cache none;
    cold = cold && leader->res.text == text_of(serial_ref(none, leader->ids, leader->rq.snap_at, 4, leader->rq.rng, nullptr, false, kSys));
    const uint64_t pri = health_field(s, "turns_boundary_first");
    const std::string seen = std::to_string(between) + " follower prepare(s) began between the leader's mark piece landing and its mark (" +
                             std::to_string(before) + " before the landing); " + std::to_string(restored) + " of " + std::to_string(kN) +
                             " followers restored the shared prefix; turns_boundary_first " + std::to_string(pri);
    check(t_land >= 0 && t_mark > t_land && m.marks.load() >= 1, "wave priority: the leader's mark piece landed, then its mark ran (" + seen + ")");
    check(between <= 1 && pri != UINT64_MAX && pri >= 1, "wave priority: the leader's mark turn went ahead of the follower prepares waiting with it (" + seen + ")");
    check(restored == kN && cold && m.bad == 0, "wave priority: every follower restored the shared prefix after the mark; every reply == a cold server's; no wrong position");
    s.shutdown();
}

// The server's wave sequence (the B39 gate, b39gate2: 15 conversations with one shared prefix arriving while the pipe runs): the
// leader prepares ALONE (no follower has registered yet) and prefills its first plan piece in its turn; the followers register
// during it, so it stops there and its remainder is parked; its release submits the remainder into the running pipe -- before
// any follower is a busy lane; the followers' miss-prepares chain, park on the leader's mark, re-prepare after it and send their
// mark-extension pieces; the leader's snapshot turn then sends its tail. Two B39 effects the gate measured: (i) the remainder
// went as ONE plan piece (3,275 rows, unpipelined: maybe_recut saw no busy lane; +1.2 s), (ii) the 7-row tail landed behind the
// 14 followers' first pieces (+1.3 s). B39 (5): (i) a request waiting for its prepare turn counts as an arriving lane, so the
// release re-cuts the remainder into mix pieces (the documented kind; a lead that stays alone keeps its plan); (ii) the tail goes
// to the front of stage 0's queue. Checked here with the fake (kChunk 5, mix 2, 2 ms a row a stage): the leader's pieces after its
// first are the mix cut (<= 2 rows) and /health recuts >= 1; the tail was submitted to the front and ran before any follower
// piece that was queued when it was sent; every follower restored the shared prefix; the followers' replies == a cold server's;
// the leader's == a run over its own pieces; a leader that stays alone keeps its 5-row pieces and == --parallel 1.
void test_wave_sequence() {
    constexpr uint32_t kSys = 23;
    constexpr int kN = 8;
    const std::vector<int32_t> sys = conv(kSys, 98000);
    auto wave_req = [&](int k, uint64_t rng) {
        std::vector<int32_t> ids = sys;
        for (int32_t v : conv(10, 98500 + 100 * k)) ids.push_back(v);
        const uint32_t snap = uint32_t(ids.size());
        ids.push_back(3); ids.push_back(5);
        return std::make_unique<Req>(ids, 4, rng, snap, kSys);
    };
    // a leader that stays alone: its plan's 5-row pieces, == --parallel 1
    {
        FakeCrown m(std::vector<uint32_t>(kN + 2, 4000), true);
        ie::LanesServe::Options o;
        o.prefill_fifo = true; o.short_first = true; o.short_rows = 20; o.mix_chunk = 2; o.recut = true; o.decode_quota = 4; o.quota_max_ms = 60000;
        ie::LanesServe s(m, o);
        m.deep_from = 0; m.deep_row_ms = 2; m.record = true;
        auto a = wave_req(0, 990);
        a->run(s);
        std::vector<std::pair<uint32_t, uint32_t>> lp;
        { std::lock_guard<std::mutex> g(m.rec_mu); lp = m.pieces[a->res.lane]; }
        bool plan5 = !lp.empty();
        for (const auto& [p0, t] : lp) if (p0 < kSys + 10) plan5 = plan5 && (t == kChunk || p0 + t == kSys || p0 + t == kSys + 10);
        Cache none;
        check(plan5 && a->res.text == text_of(serial_ref(none, a->ids, a->rq.snap_at, 4, 990, nullptr, false, kSys)) && health_field(s, "recuts") == 0,
              "wave sequence: a leader that stays alone keeps its plan pieces (no re-cut) and == --parallel 1");
        s.shutdown();
    }
    FakeCrown m(std::vector<uint32_t>(kN + 2, 4000), true);
    ie::LanesServe::Options o;
    o.prefill_fifo = true; o.short_first = true; o.short_rows = 20; o.mix_chunk = 2; o.recut = true; o.decode_quota = 4; o.quota_max_ms = 60000;
    ie::LanesServe s(m, o);
    Req warm(conv(9, 97900), 3, 989, 9);   // the pipe runs when the wave arrives (the replay's workers phase before it)
    warm.run(s);
    m.deep_from = 0; m.deep_row_ms = 2;   // a 5-row piece: 10 ms a stage; the leader's first piece runs 20 ms in its turn
    m.record = true;
    auto leader = wave_req(0, 991);
    std::thread tl([&] { leader->run(s); });
    std::this_thread::sleep_for(std::chrono::milliseconds(6));   // (the leader decided `alone` and is in its first piece)
    std::vector<std::unique_ptr<Req>> fs;
    for (int k = 1; k <= kN; ++k) fs.push_back(wave_req(k, 991 + uint64_t(k)));
    std::vector<std::thread> ts;
    for (auto& f : fs) ts.emplace_back([&f, &s] { f->run(s); });
    tl.join();
    for (auto& t : ts) t.join();
    m.record = false;
    std::vector<std::pair<uint32_t, uint32_t>> lp;
    std::vector<Ev> ev;
    { std::lock_guard<std::mutex> g(m.rec_mu); lp = m.pieces[leader->res.lane]; ev = m.ev; }
    const uint32_t ll = leader->res.lane, T = uint32_t(leader->ids.size()), snap = leader->rq.snap_at;
    // (i) the leader's pieces: its first 5 rows in the turn, then the mix cut (<= 2 rows) up to its snapshot boundary
    bool first5 = !lp.empty() && lp[0] == std::make_pair(0u, kChunk), cut = true;
    uint32_t cut_pieces = 0;
    for (const auto& [p0, t] : lp)
        if (p0 >= kChunk && p0 < snap) { cut = cut && t <= o.mix_chunk; ++cut_pieces; }
    const uint64_t recuts = health_field(s, "recuts");
    check(first5 && cut && cut_pieces >= 9 && recuts >= 1,
          "wave sequence (i): the alone leader's parked remainder was re-cut at its release for the arriving followers (" + std::to_string(cut_pieces) +
              " pieces of <= " + std::to_string(o.mix_chunk) + " rows after its first " + std::to_string(kChunk) + "; recuts " + std::to_string(recuts) + ")");
    // (ii) the tail: submitted to the front, and no other lane's prefill piece that was queued when it was sent ran before it
    size_t i_front = SIZE_MAX, i_run = SIZE_MAX;
    for (size_t i = 0; i < ev.size(); ++i) {
        const auto [k, ln, p0, t] = ev[i];
        if (ln != ll || p0 != snap) continue;
        if (k == 'F' && i_front == SIZE_MAX) i_front = i;
        if (k == '0' && i_run == SIZE_MAX && i > i_front) i_run = i;
    }
    int ahead = 0, queued = 0;
    if (i_front != SIZE_MAX && i_run != SIZE_MAX) {
        for (size_t i = i_front + 1; i < i_run; ++i) { const auto [k, ln, p0, t] = ev[i]; if (k == '0' && ln != ll && t > 1) ++ahead; }
        for (size_t i = 0; i < i_front; ++i) {   // followers' pieces submitted before the tail and not yet run by then
            const auto [k, ln, p0, t] = ev[i];
            if (k != 'S' || ln == ll || t <= 1) continue;
            bool ran = false;
            for (size_t j = i + 1; j < i_front && !ran; ++j) { const auto [k2, ln2, p02, t2] = ev[j]; ran = k2 == '0' && ln2 == ln && p02 == p0; }
            if (!ran) ++queued;
        }
    }
    if (queued == 0 && i_front != SIZE_MAX) {   // (diagnostic: the events around the tail's submit)
        std::string trail;
        for (size_t i = i_front > 12 ? i_front - 12 : 0; i < std::min(ev.size(), i_run + 4); ++i) {
            const auto [k, ln, p0, t] = ev[i];
            trail += std::string(1, k) + ":" + std::to_string(ln) + "@" + std::to_string(p0) + "+" + std::to_string(t) + " ";
        }
        std::printf("  (wave sequence: events around the tail's submit: %s)\n", trail.c_str());
    }
    check(m.fronts.load() >= 1 && i_front != SIZE_MAX && i_run != SIZE_MAX && ahead == 0,
          "wave sequence (ii): the leader's tail [" + std::to_string(snap) + ", " + std::to_string(T) + ") went to the front of the queue and ran before the " +
              std::to_string(queued) + " follower piece(s) queued when it was sent (" + std::to_string(ahead) + " ran ahead of it; fronts " +
              std::to_string(m.fronts.load()) + ")");
    // the followers: each restored the leader's state at the mark (the hash over the leader's pieces up to kSys) and ran its own
    // pieces from there (mix-cut, as a cold prompt arriving beside busy lanes would be): its reply == that run
    int restored = 0, same = 0;
    for (auto& f : fs) {
        restored += f->res.cached_tokens == kSys;
        std::vector<std::pair<uint32_t, uint32_t>> fp;
        { std::lock_guard<std::mutex> g(m.rec_mu); fp = m.pieces[f->res.lane]; }
        uint64_t h = kSeed;
        uint32_t pos = 0;
        for (const auto& [p0, t] : lp) if (p0 < kSys) { h = step_hash(h, leader->ids.data() + p0, t, p0); pos = p0 + t; }
        bool contiguous = pos == kSys;
        for (const auto& [p0, t] : fp) {
            if (p0 >= f->ids.size()) continue;
            if (p0 != pos) { contiguous = false; break; }
            h = step_hash(h, f->ids.data() + p0, t, p0); pos = p0 + t;
        }
        std::vector<int32_t> want;
        if (contiguous && pos == f->ids.size())
            for (uint32_t k = 0; k < 4; ++k) { const int32_t id = id_of(h, f->rq.rng + k); want.push_back(id); h = step_hash(h, &id, 1, pos++); }
        same += contiguous && f->res.text == text_of(want);
        if (&f == &fs.front()) {   // (the first follower's pieces: the mix cut from the mark, then its tail)
            std::string ps;
            for (const auto& [p0, t] : fp) if (p0 < f->ids.size()) ps += std::to_string(p0) + "+" + std::to_string(t) + " ";
            std::printf("  (wave sequence: follower 1 on lane %u, cached %u, pieces %s)\n", f->res.lane, f->res.cached_tokens, ps.c_str());
        }
    }
    const std::string lref = text_of(ref_over(leader->ids, lp, 4, 991));
    check(restored == kN && same == kN && leader->res.text == lref && !leader->res.text.empty() && m.bad == 0,
          "wave sequence: " + std::to_string(restored) + " of " + std::to_string(kN) + " followers restored the shared prefix, " + std::to_string(same) +
              " replies == the restored state ++ their own pieces; the leader's == a run over its own pieces (" +
              (leader->res.text == lref ? "yes" : "NO") + "); wrong positions " + std::to_string(m.bad.load()));
    s.shutdown();
}

// The pipe's front flag itself (Glm5LanePipe::submit(front)): lane 1's piece holds stage 0 (40 ms), lanes 2 and 3 queue their
// pieces behind it, lane 4's piece is submitted to the FRONT; stage 0 runs 1, then 4, then 2, 3. Direct, no serve module.
void test_front_submit() {
    FakeCrown m({100, 100, 100, 100, 100}, true);
    m.record = true; m.slow_lane = 1; m.slow_ms = 40;
    std::mutex dm; std::condition_variable dcv; int done = 0;
    check(m.pipe_start_rows([&](uint32_t) { std::lock_guard<std::mutex> g(dm); ++done; dcv.notify_all(); },
                            [&](std::span<const uint32_t> ls) { std::lock_guard<std::mutex> g(dm); done += int(ls.size()); dcv.notify_all(); }).empty(),
          "front submit: the rows pipe started");
    const std::vector<int32_t> ids = {11, 12, 13};
    check(m.pipe.submit(1, ids.data(), 3, 0).empty(), "front submit: lane 1's piece in (it holds stage 0 for 40 ms)");
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    check(m.pipe.submit(2, ids.data(), 3, 0).empty() && m.pipe.submit(3, ids.data(), 3, 0).empty(), "front submit: lanes 2 and 3 queued behind it");
    check(m.pipe.submit(4, ids.data(), 3, 0, /*front=*/true).empty(), "front submit: lane 4's piece to the front");
    { std::unique_lock<std::mutex> g(dm); dcv.wait_for(g, std::chrono::seconds(5), [&] { return done >= 4; }); }
    std::vector<uint32_t> order;
    { std::lock_guard<std::mutex> g(m.rec_mu); for (const auto& [ln, p0, t] : m.s0_log) order.push_back(ln); }
    check(order == std::vector<uint32_t>{1, 4, 2, 3}, "front submit: stage 0 ran lane 1, then the FRONT piece (4), then 2 and 3 (" +
                                                       [&] { std::string s; for (uint32_t x : order) s += std::to_string(x) + " "; return s; }() + ")");
    check(m.pipe.stop().empty() && m.bad == 0, "front submit: the pipe stopped clean, every step at its stage's depth");
}

// ---- P4 B42: the decode regroup ------------------------------------------------------------------------------------------

// D decoders (prompts of 9-14 rows, 3000-token replies, cut by `quit`) started one by one as their prompt ends release them --
// the way the swarm's lanes arrive -- with nothing else in the pipe; rows + the lane pipeline. Returns the decode groups stage 0
// formed (by size, in order) after every decoder ran for `settle_ms`, the regroup's stats, and whether every reply was a prefix of
// --parallel 1's.
struct RegroupRun { std::vector<uint32_t> groups; ie::Glm5LanePipe::RegroupStats rg; bool same = true; double step_ms = 0; };
RegroupRun run_decoders(int D, int settle_ms, int window_ms, bool with_lead = false) {
    std::vector<uint32_t> caps(size_t(D) + 2, 4000);
    FakeCrown m(caps, true);
    ie::LanesServe::Options o;
    o.prefill_fifo = true; o.short_first = true; o.short_rows = 20; o.decode_quota = 4; o.quota_max_ms = 60000;
    if (with_lead) { o.mix_chunk = 2; o.recut = true; m.deep_from = 0; m.deep_row_ms = 2; }
    ie::LanesServe s(m, o);
    std::atomic<bool> quit{false};
    std::vector<std::unique_ptr<Req>> ds;
    std::vector<std::thread> ts;
    for (int k = 0; k < D; ++k) {
        ds.push_back(std::make_unique<Req>(conv(9 + uint32_t(k % 6), 97000 + 1000 * k), 3000, 970 + uint64_t(k), 9 + uint32_t(k % 6)));
        ds.back()->quit = &quit;
        ts.emplace_back([&r = ds.back(), &s] { r->run(s); });
        std::this_thread::sleep_for(std::chrono::milliseconds(3));   // (staggered first decode steps, as the prompt ends release them)
    }
    for (int i = 0; i < 3000 && health_field(s, "decoding") < uint64_t(D); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    std::unique_ptr<Req> lead;
    std::thread tl;
    if (with_lead) {   // a deep lead prefilling through the pipe beside the decoders (200 rows, 2 ms a row a stage)
        lead = std::make_unique<Req>(conv(200, 99000), 4, 981, 200);
        tl = std::thread([&] { lead->run(s); });
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(settle_ms));
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t steps0 = health_field(s, "tokens");
    m.record_groups = true;
    std::this_thread::sleep_for(std::chrono::milliseconds(window_ms));
    m.record_groups = false;
    const uint64_t steps1 = health_field(s, "tokens");
    RegroupRun r;
    r.step_ms = steps1 > steps0 ? std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / double(steps1 - steps0) * D : 0;
    if (with_lead) tl.join();
    quit = true;
    for (auto& t : ts) t.join();
    { std::lock_guard<std::mutex> g(m.rec_mu); r.groups = m.gsz; }
    r.rg = m.pipe.regroup_stats();
    for (auto& d : ds) {
        Cache none;
        r.same = r.same && !d->res.text.empty() && text_of(serial_ref(none, d->ids, d->rq.snap_at, 3000, d->rq.rng)).rfind(d->res.text, 0) == 0;
    }
    if (with_lead) { Cache none; r.same = r.same && lead->res.text == text_of(serial_ref(none, lead->ids, 200, 4, 981)) || lead->res.finish_reason == "length"; }
    r.same = r.same && m.bad == 0;
    s.shutdown();
    return r;
}

std::string hist_of(const std::vector<uint32_t>& g) {
    std::map<uint32_t, int> h;
    for (uint32_t x : g) ++h[x];
    std::string s;
    for (const auto& [k, v] : h) s += std::to_string(k) + ":" + std::to_string(v) + " ";
    return s.empty() ? "none" : s;
}

// The contract (Glm5LanePipe regroup, the AUTO cap = ceil(busy / stages)): with D >= 3 decoders and nothing else in the pipe, once
// settled every decode group stage 0 forms has >= floor(D / 2) rows -- the decoders rotate in exactly two groups of ceil(D / 2)
// and floor(D / 2) rows (6 -> 3 + 3, 5 -> 3 + 2), the state the paused turns' bursts used to leave and the drain-free turns lost
// (singletons rotating, B39 gate). Rows == solo for every lane. One or two decoders never wait (regroup waits 0: their cadence is
// untouched). With the regroup OFF (IE_Q35MOE_REGROUP=0) the same start gives the fragmented rotation (reported, not required: it
// is an alignment coin toss). Under the lane pipeline beside a deep prefilling lane, the decoders' groups stay merged: the mean
// decode group >= 1.8 rows with 4 decoders (two groups), where the fragmented rotation gives ~1.
void test_regroup() {
    for (int D : {6, 5}) {
        const RegroupRun r = run_decoders(D, 150, 250);
        uint32_t mn = UINT32_MAX;
        for (uint32_t g : r.groups) mn = std::min(mn, g);
        const uint32_t want = uint32_t(D / 2);
        check(r.same && !r.groups.empty() && mn >= want,
              "regroup: " + std::to_string(D) + " decoders settle into two groups (every group >= " + std::to_string(want) + " rows: sizes " +
                  hist_of(r.groups) + "); waits " + std::to_string(r.rg.waits) + " merged " + std::to_string(r.rg.merges) + " timed out " +
                  std::to_string(r.rg.timeouts) + "; replies prefixes of --parallel 1's");
    }
    for (int D : {1, 2}) {
        const RegroupRun r = run_decoders(D, 60, 150);
        check(r.same && r.rg.waits == 0 && !r.groups.empty(),
              "regroup: " + std::to_string(D) + " decoder(s) never wait (regroup waits " + std::to_string(r.rg.waits) + "; " +
                  std::to_string(r.groups.size()) + " steps, sizes " + hist_of(r.groups) + "; replies prefixes of --parallel 1's)");
    }
    {
        const RegroupRun r = run_decoders(4, 150, 300, /*with_lead=*/true);
        double mean = 0;
        for (uint32_t g : r.groups) mean += g;
        mean = r.groups.empty() ? 0 : mean / double(r.groups.size());
        check(r.same && mean >= 1.8,
              "regroup beside a deep prefilling lead (lane pipeline, re-cut, quota): 4 decoders' groups stay merged (mean " +
                  std::to_string(mean).substr(0, 4) + " rows, sizes " + hist_of(r.groups) + "; waits " + std::to_string(r.rg.waits) +
                  ", merged " + std::to_string(r.rg.merges) + "); replies prefixes of --parallel 1's, the lead == --parallel 1");
    }
    {
        const uint32_t saved = g_regroup_us;
        g_regroup_us = 0;
        const RegroupRun r = run_decoders(6, 150, 250);
        g_regroup_us = saved;
        check(r.same && r.rg.waits == 0, "regroup OFF (IE_Q35MOE_REGROUP=0): no wait; 6 decoders' groups as the alignment left them (sizes " + hist_of(r.groups) + ")");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) {   // one section alone (the gate's quick runs): b39 = the P4 B39 tests; abort = the B33 abort tests with the production defaults
        const std::string which = argv[1];
        g_pipeline = true;
        if (which == "b39") { test_pipe_tail_plan(); test_turn_nodrain(false); test_turn_nodrain(true); test_wave_priority(); test_front_submit(); test_wave_sequence(); }
        else if (which == "abort") { test_short_first_abort(); test_recut_abort(); test_quota_abort(); }
        else if (which == "b42") { test_regroup(); for (bool rows : {true}) test_serve(rows); }
        else { std::printf("unknown section %s (b39 | abort | b42)\n", which.c_str()); return 2; }
        std::printf("%s\n", g_fail ? "Q35M LANES TEST (section): FAIL" : "Q35M LANES TEST (section): PASS");
        return g_fail ? 1 : 0;
    }
    test_rules();
    for (bool rows : {false, true}) {
        std::printf("---- %s\n", rows ? "rows ON (P4 B14: grouped decode steps)" : "rows off (the per-lane pipe)");
        test_serve(rows);
        test_reply_snapshot(rows);
    }
    test_rows_prefill_piece();
    test_share_rules();
    test_agent_traffic();
    for (bool rows : {false, true})
        for (uint32_t cap : {0u, 2u, 3u}) test_mix_chunk(rows, cap);
    for (bool fifo : {false, true}) test_wave(fifo);
    for (bool sup : {false, true}) test_multiturn_cap(sup);
    test_multiturn_cap(false, true);
    test_sticky();
    for (bool on : {true, false}) test_short_first(on);   // (P4 B34 (3), without the lane pipeline)
    // P4 B34: the same scenarios with the lane pipeline: every reply == --parallel 1's (the fake hashes every piece boundary
    // and refuses a step off its stage's depth, so a piece out of order on a card or a cut moved would show)
    std::printf("---- lane pipeline ON (P4 B34: a lane's next piece on stage 0 beside the piece before on stage 1)\n");
    g_pipeline = true;
    for (bool rows : {false, true}) {
        test_serve(rows);
        test_reply_snapshot(rows);
    }
    test_rows_prefill_piece();
    test_agent_traffic();
    for (bool rows : {false, true})
        for (uint32_t cap : {0u, 2u, 3u}) test_mix_chunk(rows, cap);
    for (bool fifo : {false, true}) test_wave(fifo);
    test_multiturn_cap(true);
    test_sticky();
    check(g_lookaheads.load() > 0, "lane pipeline ON: the pipes took " + std::to_string(g_lookaheads.load()) + " lookahead pieces");
    std::printf("---- short-first (P4 B34 (3)) with the lane pipeline, the production defaults\n");
    for (bool on : {true, false}) test_short_first(on);
    test_short_first(true, /*slots=*/1000, /*wait_ms=*/15, /*deep_from=*/0);
    test_short_first_abort();
    std::printf("---- re-cut (P4 B36 (A)) with the lane pipeline and short-first (the production defaults)\n");
    {
        const RecutRun off = test_recut(false), on = test_recut(true);
        const double per_off = off.lead_rows ? double(off.worker_steps) / off.lead_rows : 0, per_on = on.lead_rows ? double(on.worker_steps) / on.lead_rows : 0;
        check(on.max_rows_between < off.max_rows_between && per_on > per_off,
              "re-cut: the worker's decode cadence beside the lead -- at most " + std::to_string(on.max_rows_between) + " vs " +
                  std::to_string(off.max_rows_between) + " lead rows between two steps; " + std::to_string(per_on).substr(0, 4) + " vs " +
                  std::to_string(per_off).substr(0, 4) + " steps a lead row");
    }
    test_recut_abort();
    std::printf("---- the decode quota (P4 B36 (C)) with the lane pipeline, short-first and re-cut (the production defaults)\n");
    for (bool rows : {true, false}) {
        const QuotaRun off = test_quota(0, rows), on = test_quota(4, rows);
        check(!on.lead_text.empty() && on.lead_text == off.lead_text && on.per_piece > off.per_piece,
              std::string("decode quota") + (rows ? " (rows)" : " (one lane a step)") + ": the lead's reply ON == OFF (it orders whole pieces); " +
                  std::to_string(on.per_piece).substr(0, 4) + " vs " + std::to_string(off.per_piece).substr(0, 4) +
                  " steps a decoder per lead piece; the lead " + std::to_string(int(on.lead_ms)) + " vs " + std::to_string(int(off.lead_ms)) + " ms");
    }
    test_quota_guard();
    test_quota_abort();
    test_short_first(true, 2, 60000, 40, /*quota=*/4);
    std::printf("---- the anchor (P4 B36 (B))\n");
    for (bool on : {true, false}) test_anchor(on);
    test_anchor_lru();
    std::printf("---- the serial turns beside the pipe (P4 B39): the plan's tail through the pipe, drain-free turns\n");
    test_pipe_tail_plan();
    test_turn_nodrain(false);
    test_turn_nodrain(true);
    test_wave_priority();
    test_front_submit();
    test_wave_sequence();
    std::printf("---- the decode regroup (P4 B42): two groups without the drains' bursts\n");
    test_regroup();
    // the drain switch (IE_Q35MOE_TURN_DRAIN=1) = B10-B38's protocol for the whole set: every reply == --parallel 1's there too
    std::printf("---- IE_Q35MOE_TURN_DRAIN=1 (every turn drains, the tail in prompt_end): the serving set again\n");
    g_turn_drain = true;
    for (bool rows : {false, true}) {
        test_serve(rows);
        test_reply_snapshot(rows);
    }
    test_rows_prefill_piece();
    test_agent_traffic();
    for (bool fifo : {false, true}) test_wave(fifo);
    test_multiturn_cap(true);
    test_sticky();
    for (bool on : {true, false}) test_anchor(on);
    g_turn_drain = false;
    std::printf("%s\n", g_fail ? "Q35M LANES TEST: FAIL" : "Q35M LANES TEST: PASS");
    return g_fail ? 1 : 0;
}
