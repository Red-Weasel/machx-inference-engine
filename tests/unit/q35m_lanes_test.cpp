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
int32_t id_of(uint64_t h, uint64_t seed) { return int32_t(10 + ((h ^ (seed * 0x9E3779B97F4A7C15ull)) >> 11) % 990); }

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
        const int32_t id = g_stop_at && pos == g_stop_at ? kStop : id_of(h, rng + step);
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
          rows_on(rows) { sticky.resize(uint32_t(caps.size())); }
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

    std::string stage_rows(uint32_t s, std::span<const ie::Glm5LanePipe::Step> x) {
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
    std::string stage(uint32_t s, const ie::Glm5LanePipe::Step& x) {
        if (aborted) ++calls_after_abort;
        if (s == 0) rec0(x.lane, x.pos0, x.T);
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
    ie::LanesPlan plan(const ie::LanesRequest& rq, uint32_t reused) override {
        return ie::q35m_plan(uint32_t(rq.ids->size()), rq.snap_at, reused, kChunk, rq.share_at);
    }
    // P4 B15 (Q35mLanesModel::mark / cache_peek): the lane's state at the shared prefix into the shared cache
    std::atomic<int> marks{0};
    std::string mark(uint32_t lane, const ie::LanesRequest& rq, uint32_t pos) override {
        ++marks;
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
        if (rq.snap_at > reused && Tp == rq.snap_at) {
            cache.insert(std::vector<int32_t>(rq.ids->begin(), rq.ids->begin() + Tp), L.h, false, anchor_on && rq.anchor);
            ++snaps;
        }
        if (sticky.on && rq.snap_at > reused && Tp == rq.snap_at) { ckh[lane] = L.h; sticky.set(lane, *rq.ids, Tp); }   // Fix B
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
        if (g_stop_at && st[lane].seq[1].size() == g_stop_at) return kStop;
        return id_of(first ? st[lane].kept : st[lane].h, seed);
    }
    // the crown's finish (Q35mLanesModel::finish): forward a "length" reply's last id, then the reply snapshot
    std::atomic<int> finishes{0};
    std::string finish(uint32_t lane, const ie::LanesRequest& rq, std::span<const int32_t> out, const std::string&, uint32_t pos) override {
        ++finishes;
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
        check(multi > 0 && m.group_steps.load() > 0 && m.rows_calls.load() > 0 && m.rows_calls.load() < m.rows_ids.load(),
              "rows: groups of >= 2 lanes ran (" + std::to_string(multi) + "), sample_rows " + std::to_string(m.rows_calls.load()) +
                  " calls for " + std::to_string(m.rows_ids.load()) + " ids");
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
    // with the lane pipeline the group landing is reported, not required -- the block without it requires it)
    if (!g_pipeline) check(m.kept_in_group.load() > 0, "rows: a prompt's last (1-row) piece landed inside a group and kept its logits");
    else std::printf("  (lane pipeline: a last 1-row piece landed inside a group %u time(s))\n", m.kept_in_group.load());
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
    for (int i = 0; i < 3000 && health_field(s, "short_bypass") == 0; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const auto t0 = Clock::now();
    const uint32_t held = s.abort_all();
    const double abort_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    for (auto& t : ts) t.join();
    tl.join(); td.join();
    const double join_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    s.shutdown();
    const double down_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    bool all_abort = d.res.finish_reason == "abort" && lead.res.finish_reason == "abort";
    for (auto& r : sh) all_abort = all_abort && (r->res.finish_reason == "abort" || r->res.finish_reason == "length");
    check(abort_ms < 50 && join_ms < 300, "short-first abort: abort_all does not block (" + std::to_string(int(abort_ms)) + " ms), every request returned (" +
                                              std::to_string(int(join_ms)) + " ms)");
    check(all_abort, "short-first abort: the decoder and the held lead end abort; each short prompt ends abort or had finished");
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

}  // namespace

int main() {
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
    std::printf("%s\n", g_fail ? "Q35M LANES TEST: FAIL" : "Q35M LANES TEST: PASS");
    return g_fail ? 1 : 0;
}
