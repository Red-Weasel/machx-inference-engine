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
struct Cache {
    struct Ep { std::vector<int32_t> toks; uint64_t h; };
    std::vector<Ep> eps;
    uint32_t restore(const std::vector<int32_t>& ids, uint64_t& h) const {
        const Ep* best = nullptr;
        for (const auto& e : eps)
            if (e.toks.size() <= ids.size() && std::equal(e.toks.begin(), e.toks.end(), ids.begin()) &&
                (!best || e.toks.size() > best->toks.size())) best = &e;
        if (!best || best->toks.size() > ids.size() - 1) return 0;
        h = best->h;
        return uint32_t(best->toks.size());
    }
    void insert(const std::vector<int32_t>& toks, uint64_t h) {
        for (const auto& e : eps) if (e.toks == toks) return;
        eps.push_back({toks, h});
    }
};

// --parallel 1 on the crown split (Engine::generate): what a request gives, given the cache state before it.
// reply_cache: its gen-cache insert at prompt ++ reply after a "stop"/"length" reply (every sampled id was forwarded).
std::vector<int32_t> serial_ref(Cache& c, const std::vector<int32_t>& ids, uint32_t snap_at, uint32_t n, uint64_t rng,
                                uint32_t* restored_out = nullptr, bool reply_cache = false, uint32_t share_at = 0) {
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
    if (snap_at > restored) c.insert(std::vector<int32_t>(ids.begin(), ids.begin() + snap_at), h);
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
    std::atomic<bool> in_group_cb{false};

    explicit FakeCrown(std::vector<uint32_t> caps, bool rows = false)
        : st(caps.size()), cap(caps),
          pipe(uint32_t(caps.size()), kChunk, 1, 2, [this](uint32_t s, const ie::Glm5LanePipe::Step& x) { return stage(s, x); },
               [this](uint32_t s, std::span<const ie::Glm5LanePipe::Step> x, float*) { return stage_rows(s, x); }, 16, 0),
          rows_on(rows) {}

    std::string stage_rows(uint32_t s, std::span<const ie::Glm5LanePipe::Step> x) {
        const int a = ++active;
        for (int m = max_active.load(); a > m && !max_active.compare_exchange_weak(m, a);) {}
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        --active;
        if (s == 1) ++group_steps;
        for (const auto& step : x)
            if (auto e = forward(step.lane, s, step.ids, 1, step.pos0); !e.empty()) return e;
        return {};
    }
    bool rows() const override { return rows_on; }
    std::string pipe_start_rows(DoneFn d, RowsDoneFn rd) override {
        return pipe.start(std::move(d), [this, rd](std::span<const uint32_t> ls) { in_group_cb = true; rd(ls); in_group_cb = false; });
    }
    std::string sample_rows(std::span<const uint32_t> lanes, std::span<const ie::LanesSampling* const> sp,
                            std::span<const std::span<const int32_t>> windows, std::span<const uint64_t> seeds,
                            std::span<int32_t> picks) override {
        ++rows_calls; rows_ids += uint32_t(lanes.size());
        return ie::LanesModel::sample_rows(lanes, sp, windows, seeds, picks);
    }

    std::string forward(uint32_t lane, uint32_t s, const int32_t* ids, uint32_t T, uint32_t pos0) {
        St& L = st[lane];
        if (pos0 == 0) L.seq[s].clear();
        if (L.seq[s].size() != pos0) { ++bad; return "stage " + std::to_string(s) + " at " + std::to_string(L.seq[s].size()) + ", step at " + std::to_string(pos0); }
        L.seq[s].insert(L.seq[s].end(), ids, ids + T);
        if (s == 1) L.h = step_hash(L.h, ids, T, pos0);
        return {};
    }
    std::string stage(uint32_t s, const ie::Glm5LanePipe::Step& x) {
        const int a = ++active;
        for (int m = max_active.load(); a > m && !max_active.compare_exchange_weak(m, a);) {}
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        --active;
        return forward(x.lane, s, x.ids, x.T, x.pos0);
    }
    const char* tag() const override { return "fake crown lanes"; }
    uint32_t n_lanes() const override { return uint32_t(st.size()); }
    uint32_t lane_cap(uint32_t lane) const override { return cap[lane]; }
    uint32_t own_match(uint32_t, std::span<const int32_t>) const override { return 0; }
    bool occupied(uint32_t) const override { return false; }
    uint32_t last_end(uint32_t) const override { return 0; }
    std::string prefix_prepare(uint32_t lane, const ie::LanesRequest& rq, uint32_t& reused, std::string& source) override {
        St& L = st[lane];
        source.clear();
        uint64_t h = kSeed;
        reused = cache.restore(*rq.ids, h);
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
        cache.insert(std::vector<int32_t>(rq.ids->begin(), rq.ids->begin() + pos), L.h);
        return {};
    }
    uint32_t cache_peek(const ie::LanesRequest& rq) const override { uint64_t h = 0; return cache.restore(*rq.ids, h); }
    std::string prefill_serial(uint32_t lane, const ie::LanesRequest& rq, std::span<const ie::LanesChunk> ch,
                               const std::function<bool()>& stop, size_t& done) override {
        done = 0;
        for (size_t k = 0; k < ch.size(); ++k) {
            if (k > 0 && stop && stop()) break;
            for (uint32_t s = 0; s < 2; ++s)
                if (auto e = forward(lane, s, rq.ids->data() + ch[k].first, ch[k].second, ch[k].first); !e.empty()) return e;
            ++done;
        }
        return {};
    }
    std::string prompt_end(uint32_t lane, const ie::LanesRequest& rq, uint32_t Tp, uint32_t reused, bool kept) override {
        St& L = st[lane];
        const uint32_t T = uint32_t(rq.ids->size());
        if (rq.snap_at > reused && Tp == rq.snap_at) cache.insert(std::vector<int32_t>(rq.ids->begin(), rq.ids->begin() + Tp), L.h);
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
    std::string reset_lane(uint32_t lane) override { st[lane] = St{}; return pipe.reset_lane(lane); }
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
        return {};
    }
    void keep_logits(uint32_t lane) override { if (in_group_cb) ++kept_in_group; st[lane].kept = st[lane].h; }
    bool is_stop(int32_t id) const override { return id == kStop; }
    std::string detok(std::span<const int32_t> out) const override {
        std::string s;
        for (int32_t id : out) s += std::to_string(id) + ",";
        return s;
    }
    std::string pipe_start(DoneFn d) override { return pipe.start(std::move(d)); }
    std::string pipe_submit(uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0) override { return pipe.submit(lane, ids, T, pos0); }
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
    Req(std::vector<int32_t> p, uint32_t n, uint64_t rng, uint32_t snap_at, uint32_t share_at = 0) : ids(std::move(p)) {
        rq.ids = &ids; rq.sp.max_tokens = n; rq.rng = rng; rq.snap_at = snap_at; rq.share_at = share_at;
    }
    void run(ie::LanesServe& s) {
        rq.ids = &ids;
        res = s.run(rq, [this](std::string_view piece) {
            if (piece.empty()) return true;
            ++pieces;
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
    check(m.kept_in_group.load() > 0, "rows: a prompt's last (1-row) piece landed inside a group and kept its logits");
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
    std::printf("%s\n", g_fail ? "Q35M LANES TEST: FAIL" : "Q35M LANES TEST: PASS");
    return g_fail ? 1 : 0;
}
