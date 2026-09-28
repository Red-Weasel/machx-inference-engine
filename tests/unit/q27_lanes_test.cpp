// tests/unit/q27_lanes_test.cpp -- host-only (no GPU, no SYCL): the Qwen3.8-27B split's request lanes (P4 B18,
// include/ie/q27_lanes.hpp, docs/lanes/LANES_SERVE.md) -- the prefill chunk and lane context rules, the prefill ops against
// --parallel 1's prefill loop (the pipelined ranges and the kind of every 1-row piece), the plan / prompt-end split, and the
// 27B's hooks served by the shared lanes module with a fake two-stage model.
//
// The fake: each lane keeps, per stage, the ids it has "forwarded" (a step must start at the stage's own depth); stage 1's
// "logits" hash every id, every step's start position AND, for a 1-row step, whether it ran the prefill kernels (a piece of a
// pipelined range) or the decode kernels -- so a reply depends on how its prompt was cut AND on each 1-row piece's kernels,
// like the real model's rounding does. A serial reference (Engine::generate's 27B-split path: restore, prefill_to(share_at)
// + insert when the shared prefix is on, prefill_to(snap_at), insert, prefill_to(T) with the pipelined rule, decode one row a
// step) run over the same request sequence predicts every reply; concurrent lanes, rows on and off, the serial turn, pieces
// through the pipe beside decoding lanes, the shared cache, a cancel and the reply snapshot must all leave each request == it.
// Does NOT link ie_core.
#undef NDEBUG
#include "ie/q27_lanes.hpp"
#include "ie/q35m_lanes.hpp"   // q35m_lane_bytes / q35m_lanes_fit (the 27B's lanes are the same KV + DeltaNet shape)

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
int32_t id_of(uint64_t h, uint64_t seed) { return int32_t(10 + ((h ^ (seed * 0x9E3779B97F4A7C15ull)) >> 11) % 990); }

// Stage 1's state after one step: a marker for the step's start (and a 1-row step's kernels), then its ids.
uint64_t step_hash(uint64_t h, const int32_t* ids, uint32_t T, uint32_t pos0, bool pk) {
    if (pos0 == 0) h = kSeed;
    h = mix(h, 0xC0000000ull | pos0);
    if (T == 1) h = mix(h, pk ? 0xA1u : 0xD1u);
    for (uint32_t i = 0; i < T; ++i) h = mix(h, uint64_t(uint32_t(ids[i])));
    return h;
}

// The shared prompt cache (FleetPrefixCache + fleet_cache_restore's rule): the longest endpoint that is a prefix of the
// prompt, used only when it leaves >= 1 row.
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

bool g_pipeline = true;   // IE_QWEN35_NO_PIPELINE unset

// --parallel 1 on the 27B split (Engine::generate), written from its prefill_to loop, not from q27_lanes.hpp: a range longer
// than one chunk runs pipelined (every chunk the prefill kernels, a 1-row last one too); a shorter range runs forward() chunk by
// chunk (a 1-row chunk the decode kernels); decode steps are 1-row decode-kernel steps.
std::vector<int32_t> serial_ref(Cache& c, const std::vector<int32_t>& ids, uint32_t snap_at, uint32_t n, uint64_t rng,
                                uint32_t* restored_out = nullptr, bool reply_cache = false, uint32_t share_at = 0,
                                bool cache_on = true) {
    uint64_t h = kSeed;
    const uint32_t T = uint32_t(ids.size());
    uint32_t pos = cache_on ? c.restore(ids, h) : 0;
    if (restored_out) *restored_out = pos;
    const uint32_t restored = pos;
    auto prefill_to = [&](uint32_t end) {
        if (g_pipeline && end > pos && end - pos > kChunk) {
            while (pos < end) { const uint32_t k = std::min(kChunk, end - pos); h = step_hash(h, ids.data() + pos, k, pos, true); pos += k; }
            return;
        }
        while (pos < end) { const uint32_t k = std::min(kChunk, end - pos); h = step_hash(h, ids.data() + pos, k, pos, false); pos += k; }
    };
    if (share_at > restored) {
        prefill_to(share_at);
        c.insert(std::vector<int32_t>(ids.begin(), ids.begin() + share_at), h);
    }
    prefill_to(snap_at);
    if (cache_on && snap_at > restored) c.insert(std::vector<int32_t>(ids.begin(), ids.begin() + snap_at), h);
    prefill_to(T);
    std::vector<int32_t> out;
    for (uint32_t step = 0; step < n; ++step) {
        const int32_t id = id_of(h, rng + step);
        if (id == kStop) break;
        out.push_back(id);
        h = step_hash(h, &id, 1, pos, false); ++pos;
    }
    if (reply_cache && cache_on && !out.empty()) {
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

// The 27B's hooks (Q27LanesModel in src/engine/engine.cpp) over fake state: the same q27_lanes.hpp rules drive the plan and
// prompt_end; a pipe step of 1 row runs the decode kernels (the real stage: forward_stage with pk false), a serial piece its
// plan kind.
struct FakeQ27 final : ie::LanesModel {
    struct St { std::vector<int32_t> seq[2]; uint64_t h = kSeed, kept = 0; };
    std::vector<St> st;
    std::vector<uint32_t> cap;
    Cache cache;
    bool cache_on = true;
    std::atomic<uint32_t> bad{0}, pk1{0};
    std::atomic<int> active{0}, max_active{0};
    ie::CardPipe pipe;
    bool rows_on = false;
    std::atomic<uint32_t> rows_calls{0}, rows_ids{0}, group_steps{0};
    std::atomic<bool> in_group_cb{false};

    explicit FakeQ27(std::vector<uint32_t> caps, bool rows = false)
        : st(caps.size()), cap(caps),
          pipe(uint32_t(caps.size()), kChunk, 1, 2, [this](uint32_t s, const ie::Glm5LanePipe::Step& x) { return stage(s, x); },
               [this](uint32_t s, std::span<const ie::Glm5LanePipe::Step> x, float*) { return stage_rows(s, x); }, 16, 0),
          rows_on(rows) {}

    std::string forward(uint32_t lane, uint32_t s, const int32_t* ids, uint32_t T, uint32_t pos0, bool pk) {
        St& L = st[lane];
        if (pos0 == 0) L.seq[s].clear();
        if (L.seq[s].size() != pos0) { ++bad; return "stage " + std::to_string(s) + " at " + std::to_string(L.seq[s].size()) + ", step at " + std::to_string(pos0); }
        L.seq[s].insert(L.seq[s].end(), ids, ids + T);
        if (s == 1) { L.h = step_hash(L.h, ids, T, pos0, pk); if (T == 1 && pk) ++pk1; }
        return {};
    }
    void busy() {
        const int a = ++active;
        for (int m = max_active.load(); a > m && !max_active.compare_exchange_weak(m, a);) {}
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        --active;
    }
    std::string stage(uint32_t s, const ie::Glm5LanePipe::Step& x) { busy(); return forward(x.lane, s, x.ids, x.T, x.pos0, x.T > 1); }
    std::string stage_rows(uint32_t s, std::span<const ie::Glm5LanePipe::Step> x) {
        busy();
        if (s == 1) ++group_steps;
        for (const auto& step : x)
            if (auto e = forward(step.lane, s, step.ids, 1, step.pos0, false); !e.empty()) return e;
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
    const char* tag() const override { return "fake q27 lanes"; }
    uint32_t n_lanes() const override { return uint32_t(st.size()); }
    uint32_t lane_cap(uint32_t lane) const override { return cap[lane]; }
    uint32_t own_match(uint32_t, std::span<const int32_t>) const override { return 0; }
    bool occupied(uint32_t) const override { return false; }
    uint32_t last_end(uint32_t) const override { return 0; }
    std::string prefix_prepare(uint32_t lane, const ie::LanesRequest& rq, uint32_t& reused, std::string& source) override {
        St& L = st[lane];
        source.clear();
        uint64_t h = kSeed;
        reused = cache_on ? cache.restore(*rq.ids, h) : 0;
        L.seq[0].assign(rq.ids->begin(), rq.ids->begin() + reused);
        L.seq[1] = L.seq[0];
        L.h = reused ? h : kSeed;
        return {};
    }
    std::vector<ie::Q27Op> ops(const ie::LanesRequest& rq, uint32_t reused) const {
        return ie::q27_prefill_ops(uint32_t(rq.ids->size()), reused, rq.snap_at, rq.share_at, cache_on, kChunk, g_pipeline);
    }
    ie::LanesPlan plan(const ie::LanesRequest& rq, uint32_t reused) override {
        return ie::q27_plan(ops(rq, reused), reused, uint32_t(rq.ids->size()));
    }
    std::atomic<int> marks{0};
    std::string mark(uint32_t lane, const ie::LanesRequest& rq, uint32_t pos) override {
        ++marks;
        St& L = st[lane];
        if (L.seq[0].size() != pos || L.seq[1].size() != pos) { ++bad; return "mark: the lane is not at " + std::to_string(pos); }
        cache.insert(std::vector<int32_t>(rq.ids->begin(), rq.ids->begin() + pos), L.h);
        return {};
    }
    uint32_t cache_peek(const ie::LanesRequest& rq) const override {
        if (!cache_on) return 0;
        uint64_t h = 0;
        return cache.restore(*rq.ids, h);
    }
    std::string run(uint32_t lane, const int32_t* ids, std::span<const ie::Q27Piece> ps) {
        for (const ie::Q27Piece& p : ps)
            for (uint32_t s = 0; s < 2; ++s)
                if (auto e = forward(lane, s, ids + p.pos0, p.rows, p.pos0, p.rows > 1 || p.pk); !e.empty()) return e;
        return {};
    }
    std::string prefill_serial(uint32_t lane, const ie::LanesRequest& rq, std::span<const ie::LanesChunk> ch,
                               const std::function<bool()>& stop, size_t& done) override {
        done = 0;
        for (size_t k = 0; k < ch.size(); ++k) {
            if (k > 0 && stop && stop()) break;
            const ie::Q27Piece p{ch[k].first, ch[k].second, false};   // (a plan never holds a 1-row pk piece)
            if (auto e = run(lane, rq.ids->data(), std::span<const ie::Q27Piece>(&p, 1)); !e.empty()) return e;
            ++done;
        }
        return {};
    }
    std::string prompt_end(uint32_t lane, const ie::LanesRequest& rq, uint32_t Tp, uint32_t reused, bool kept) override {
        St& L = st[lane];
        bool ran = false;
        const std::string e = ie::q27_walk_rest(
            ie::q27_rest(ops(rq, reused), Tp),
            [&](std::span<const ie::Q27Piece> ps) { return run(lane, rq.ids->data(), ps); },
            [&](const ie::Q27Op& o) {
                if (L.seq[0].size() != o.at || L.seq[1].size() != o.at) ++bad;
                cache.insert(std::vector<int32_t>(rq.ids->begin(), rq.ids->begin() + o.at), L.h);
            },
            ran);
        if (!e.empty()) return e;
        if (ran) kept = false;
        if (!kept) L.kept = L.h;
        return {};
    }
    std::string reset_lane(uint32_t lane) override { st[lane] = St{}; return pipe.reset_lane(lane); }
    int32_t sample(uint32_t lane, bool first, const ie::LanesSampling&, std::span<const int32_t>, uint64_t seed, std::string&) override {
        return id_of(first ? st[lane].kept : st[lane].h, seed);
    }
    std::atomic<int> finishes{0};
    std::string finish(uint32_t lane, const ie::LanesRequest& rq, std::span<const int32_t> out, const std::string&, uint32_t pos) override {
        if (!cache_on) return {};
        ++finishes;
        std::vector<int32_t> full(rq.ids->begin(), rq.ids->end());
        full.insert(full.end(), out.begin(), out.end());
        if (pos + 1 == full.size()) {   // --parallel 1 forwarded the last id of a "length" reply as a decode step
            for (uint32_t s = 0; s < 2; ++s)
                if (auto e = forward(lane, s, full.data() + pos, 1, pos, false); !e.empty()) return e;
            ++pos;
        }
        if (pos != full.size() || st[lane].seq[1] != full) { ++bad; return "the lane does not hold prompt ++ reply"; }
        cache.insert(full, st[lane].h);
        return {};
    }
    void keep_logits(uint32_t lane) override { st[lane].kept = st[lane].h; }
    bool is_stop(int32_t id) const override { return id == kStop; }
    std::string detok(std::span<const int32_t> out) const override { return text_of(std::vector<int32_t>(out.begin(), out.end())); }
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
    Req(std::vector<int32_t> p, uint32_t n, uint64_t rng, uint32_t snap_at, uint32_t share_at = 0) : ids(std::move(p)) {
        rq.ids = &ids; rq.sp.max_tokens = n; rq.rng = rng; rq.snap_at = snap_at; rq.share_at = share_at;
    }
    void run(ie::LanesServe& s) {
        rq.ids = &ids;
        res = s.run(rq, [this](std::string_view piece) {
            if (piece.empty()) return true;
            ++pieces;
            return !(cancel_after >= 0 && pieces >= cancel_after);
        });
    }
};

void run_all(std::vector<Req*> rs, ie::LanesServe& s) {
    std::vector<std::thread> th;
    for (Req* r : rs) th.emplace_back([r, &s] { r->run(s); });
    for (auto& t : th) t.join();
}

// --parallel 1's op sequence, written from Engine::generate independently of q27_prefill_ops: (piece | insert) in order.
std::vector<ie::Q27Op> ref_ops(uint32_t T, uint32_t reused, uint32_t snap, uint32_t share, bool cache_on, uint32_t pf, bool pipe) {
    std::vector<ie::Q27Op> r;
    uint32_t pos = reused;
    auto prefill_to = [&](uint32_t end) {
        const bool piped = pipe && end > pos && (end - pos) > pf;
        while (pos < end) {
            const uint32_t n = std::min(pf, end - pos);
            ie::Q27Op o; o.piece = ie::Q27Piece{pos, n, piped};
            r.push_back(o);
            pos += n;
        }
    };
    if (share > reused) { prefill_to(share); ie::Q27Op o; o.kind = ie::Q27Op::kShared; o.at = share; r.push_back(o); }
    prefill_to(snap);
    if (cache_on && snap > reused) { ie::Q27Op o; o.kind = ie::Q27Op::kSnap; o.at = snap; r.push_back(o); }
    prefill_to(T);
    return r;
}
bool same_op(const ie::Q27Op& a, const ie::Q27Op& b) {
    if (a.kind != b.kind) return false;
    return a.kind == ie::Q27Op::kPiece ? (a.piece.pos0 == b.piece.pos0 && a.piece.rows == b.piece.rows &&
                                          (a.piece.rows > 1 || a.piece.pk == b.piece.pk))
                                       : a.at == b.at;
}

void test_rules() {
    check(ie::q27_prefill_chunk(65536, nullptr) == 512 && ie::q27_prefill_chunk(300, nullptr) == 300 &&
              ie::q27_prefill_chunk(65536, "256") == 256 && ie::q27_prefill_chunk(4096, "99999") == 512 &&
              ie::q27_prefill_chunk(4096, "0") == 512,
          "prefill chunk: min(512, ctx); IE_QWEN35_PREFILL_CHUNK 1..ctx overrides (no 8192 rule: the 27B is dense)");
    check(ie::q27_lane_ctx(0, 262144) == 65536 && ie::q27_lane_ctx(0, 16384) == 16384 && ie::q27_lane_ctx(8192, 65536) == 8192,
          "lane ctx: --slot-ctx 0 = 65,536 (the joint-step banks' default), capped at --ctx");
    // the 27B's per-lane bytes, one card of the split: 32 layers = 8 full attention (4 KV heads x 256) + 24 DeltaNet
    // (48 v heads x 128 x 128 f32, conv 10240 channels x 3 fp16) -- q35m_lane_bytes' shape (Q35mLaneShape)
    ie::Q35mLaneShape sh;
    sh.n_full = 8; sh.n_kv_heads = 4; sh.head_dim = 256;
    sh.n_lin = 24; sh.v_heads = 48; sh.k_head_dim = 128; sh.v_head_dim = 128; sh.conv_channels = 10240; sh.conv_kernel = 4;
    const uint64_t b = ie::q35m_lane_bytes(sh, 65536);
    const uint64_t want = 2ull * 8 * 4 * 65536 * 256 * 2 + 24ull * (48ull * 128 * 128 * 4 + 10240ull * 3 * 2);
    check(b == want, "27B lane bytes per card at 64K = " + std::to_string(b >> 20) + " MiB (KV " +
                         std::to_string((2ull * 8 * 4 * 65536 * 256 * 2) >> 20) + " MiB + DeltaNet)");
    const uint64_t res = 1536ull << 20;
    check(ie::q35m_lanes_fit(15 * b + res, b, 16, 65536, res, 0).empty() && !ie::q35m_lanes_fit(15 * b + res - 1, b, 16, 65536, res, 0).empty(),
          "admission: 16 lanes need exactly 15 extra lanes + the reserve free");

    // the ops == --parallel 1's prefill loop, every shape; the plan + the rest cover it in order; no 1-row pk piece in the pipe
    bool ops_ok = true, split_ok = true, no_pk1 = true, mark_ok = true, cut_ok = true;
    uint64_t shapes = 0, with_pk1 = 0;
    for (bool pipe : {true, false})
        for (bool cache_on : {true, false})
            for (uint32_t T = 1; T <= 30; ++T)
                for (uint32_t snap = 1; snap <= T; ++snap)
                    for (uint32_t share = 0; share < snap; ++share) {
                        if (share && !cache_on) continue;   // (shared_prefix_boundary: the cache on)
                        for (uint32_t reused = 0; reused < T; ++reused)
                            for (uint32_t pf : {1u, 2u, 3u, 5u, 64u}) {
                                ++shapes;
                                const auto ref = ref_ops(T, reused, cache_on ? snap : T, share, cache_on, pf, pipe);
                                const auto ops = ie::q27_prefill_ops(T, reused, cache_on ? snap : T, share, cache_on, pf, pipe);
                                bool same = ref.size() == ops.size();
                                for (size_t i = 0; same && i < ops.size(); ++i) same = same_op(ref[i], ops[i]);
                                if (!same) { if (ops_ok) std::printf("  ops mismatch T %u snap %u share %u reused %u pf %u pipe %d\n", T, snap, share, reused, pf, pipe); ops_ok = false; }
                                const ie::LanesPlan p = ie::q27_plan(ops, reused, T);
                                std::vector<ie::Q27Op> got;
                                for (const auto& [p0, n] : p.chunks) { ie::Q27Op o; o.piece = ie::Q27Piece{p0, n, false}; got.push_back(o); }
                                // the module's mark, then prompt_end's rest
                                std::vector<ie::Q27Op> all;
                                size_t k = 0;
                                for (const auto& o : ref) {
                                    if (o.kind == ie::Q27Op::kPiece) { if (k < got.size()) { all.push_back(got[k]); ++k; } }
                                    else if (o.kind == ie::Q27Op::kShared && p.mark == o.at && o.at < p.Tp) all.push_back(o);
                                }
                                for (const auto& o : ie::q27_rest(ops, p.Tp)) all.push_back(o);
                                bool sp = all.size() == ref.size();
                                for (size_t i = 0; sp && i < all.size(); ++i) sp = same_op(ref[i], all[i]);
                                if (!sp) { if (split_ok) std::printf("  split mismatch T %u snap %u share %u reused %u pf %u pipe %d\n", T, snap, share, reused, pf, pipe); split_ok = false; }
                                for (const auto& [p0, n] : p.chunks) {
                                    if (n > pf) cut_ok = false;
                                    for (const auto& o : ref)
                                        if (o.kind == ie::Q27Op::kPiece && o.piece.pos0 == p0 && o.piece.rows == 1 && o.piece.pk) no_pk1 = false;
                                }
                                bool has_pk1 = false;
                                for (const auto& o : ref) has_pk1 = has_pk1 || (o.kind == ie::Q27Op::kPiece && o.piece.rows == 1 && o.piece.pk);
                                with_pk1 += has_pk1;
                                const bool due = share > reused && share < p.Tp;
                                if (due != (p.mark == share && share > 0)) mark_ok = false;
                                if (p.Tp < reused || p.Tp > T) cut_ok = false;
                                if (cache_on && snap > reused && p.Tp > snap) cut_ok = false;   // the pipe's part ends by the snapshot
                            }
                    }
    check(ops_ok, "q27_prefill_ops == --parallel 1's prefill_to loop (pieces, the kind of every 1-row piece, the inserts), " +
                      std::to_string(shapes) + " shapes (T <= 30, pipeline on/off, cache on/off)");
    check(split_ok, "the pipe's pieces + the module's mark + prompt_end's rest == --parallel 1's ops in order");
    check(no_pk1 && with_pk1 > 0, "no 1-row pipelined piece ever goes through the pipe (" + std::to_string(with_pk1) + " shapes have one)");
    check(mark_ok, "q27_plan: mark == share exactly when reused < share < Tp");
    check(cut_ok, "q27_plan: pieces <= pf_chunk rows, reused <= Tp <= T, Tp <= the snapshot boundary when one is due");
    {
        // T 13 with pf 5 cold, no cache: one pipelined range 5+5+3 (no 1-row piece) -> the pipe runs all of it
        const auto p = ie::q27_plan(ie::q27_prefill_ops(13, 0, 13, 0, false, 5, true), 0, 13);
        check(p.Tp == 13 && p.chunks.size() == 3, "T 13 / pf 5: the pipe runs the whole prompt");
        // T 11: 5+5+1 pipelined -> the pipe stops at 10, prompt_end runs the 1-row prefill-kernel piece
        const auto ops = ie::q27_prefill_ops(11, 0, 11, 0, false, 5, true);
        const auto q = ie::q27_plan(ops, 0, 11);
        const auto rest = ie::q27_rest(ops, q.Tp);
        check(q.Tp == 10 && q.chunks.size() == 2 && rest.size() == 1 && rest[0].piece.pk && rest[0].piece.rows == 1,
              "T 11 / pf 5: the pipe stops at 10; the last row runs in the turn through the prefill kernels");
        // a 1-row range (the exact-repeat restore of T - 1): forward(T=1) = the decode kernels, through the pipe
        const auto r = ie::q27_plan(ie::q27_prefill_ops(11, 10, 11, 0, false, 5, true), 10, 11);
        check(r.Tp == 11 && r.chunks.size() == 1, "a 1-row range goes through the pipe (the decode kernels, like forward(T = 1))");
    }
}

void test_serve(bool rows) {
    FakeQ27 m({300, 120, 120}, rows);
    ie::LanesServe s(m, {});
    Cache ref;
    auto expect = [&](const Req& r, uint32_t* restored = nullptr) {
        return text_of(serial_ref(ref, r.ids, r.rq.snap_at, r.rq.sp.max_tokens, r.rq.rng, restored, false, r.rq.share_at));
    };
    // three conversations, each with a stable prefix (snap_at) and a volatile tail
    auto mk = [&](int k, uint32_t n) { std::vector<int32_t> p = conv(23 + 7 * k, 100 * (k + 1)); p.push_back(3); p.push_back(4); return Req(p, n, 40 + k, uint32_t(p.size() - 2)); };
    bool warm_ok = true;
    for (int k = 0; k < 3; ++k) {
        Req r = mk(k, 8);
        r.run(s);
        uint32_t rs = 0;
        const std::string want = expect(r, &rs);
        warm_ok = warm_ok && r.res.text == want && r.res.cached_tokens == rs;
    }
    check(warm_ok, "warm: 3 requests alone == --parallel 1's (cold prefills, snapshots inserted)");
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
    {
        Req a = mk(0, 30), b = mk(1, 30), c = mk(2, 30);
        run_all({&a, &b, &c}, s);
        check(a.res.text == solo[0] && b.res.text == solo[1] && c.res.text == solo[2], "batch: 3 at once == solo");
        check(m.max_active.load() >= 2, "the two stages overlapped (max " + std::to_string(m.max_active.load()) + " at once)");
    }
    {
        Req a = mk(0, 30), b = mk(1, 30), c = mk(2, 30);
        b.cancel_after = 3;
        run_all({&a, &b, &c}, s);
        check(b.res.finish_reason == "abort" && a.res.text == solo[0] && c.res.text == solo[2], "cancel: the survivors == solo");
        Req again = mk(1, 30);
        again.run(s);
        check(again.res.text == solo[1], "the cancelled conversation answers == solo afterwards");
    }
    // cold: NEW long prompts at once, whose ranges end on 1-row pipelined pieces (snap 41 = 8 x 5 + 1; T - snap 11 = 5 + 5 + 1;
    // T 36 = 7 x 5 + 1) or not, == --parallel 1 running them one after another
    const uint32_t pk_before = m.pk1.load();
    {
        std::vector<int32_t> pa = conv(52, 5000), pb = conv(36, 6000), pc = conv(33, 7000);
        Req a(pa, 20, 1, 41), b(pb, 20, 2, 36), c(pc, 20, 3, 33);
        run_all({&a, &b, &c}, s);
        const bool ok = a.res.text == expect(a) && b.res.text == expect(b) && c.res.text == expect(c);
        check(ok && a.res.cached_tokens == 0 && b.res.cached_tokens == 0, "cold: 3 new prompts at once (1-row pipelined pieces) == --parallel 1's");
    }
    check(m.pk1.load() > pk_before, "the 1-row pipelined pieces ran through the prefill kernels (" +
                                        std::to_string(m.pk1.load() - pk_before) + ")");
    {   // a follow-up turn extends a conversation past its snapshot
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
    {   // an exact repeat of a whole-prompt snapshot: the DeltaNet rule refuses the restore (full prefill)
        std::vector<int32_t> p = conv(19, 8000);
        Req x(p, 6, 5, uint32_t(p.size())), y(p, 6, 5, uint32_t(p.size()));
        x.run(s); y.run(s);
        const std::string wx = expect(x), wy = expect(y);
        check(x.res.text == wx && y.res.text == wy && y.res.cached_tokens == 0, "an exact repeat of a full-depth snapshot prefills in full");
    }
    check(m.bad == 0, "no stage saw a wrong position, no insert off its boundary");
    s.shutdown();
    if (rows) {
        const auto gs = m.pipe.group_sizes();
        uint64_t multi = 0;
        for (size_t g = 2; g < gs.size(); ++g) multi += gs[g];
        check(multi > 0 && m.rows_calls.load() > 0 && m.rows_calls.load() < m.rows_ids.load(),
              "rows: groups of >= 2 lanes ran (" + std::to_string(multi) + "), sample_rows " + std::to_string(m.rows_calls.load()) +
                  " calls for " + std::to_string(m.rows_ids.load()) + " ids");
    } else {
        check(m.rows_calls.load() == 0 && m.group_steps.load() == 0, "rows off: the per-lane pipe (no group step, no sample_rows)");
    }
}

// Cache off (--no-prompt-cache) and the pipeline off (IE_QWEN35_NO_PIPELINE): every 1-row piece takes the decode kernels.
void test_cache_and_pipeline_off() {
    for (bool pipe : {true, false}) {
        g_pipeline = pipe;
        FakeQ27 m({300, 120, 120}, true);
        m.cache_on = false;
        ie::LanesServe s(m, {});
        Cache ref;
        Req a(conv(36, 100), 25, 1, 36), b(conv(41, 200), 25, 2, 41), c(conv(16, 300), 25, 3, 16);
        run_all({&a, &b, &c}, s);
        auto want = [&](const Req& r) { return text_of(serial_ref(ref, r.ids, r.rq.snap_at, 25, r.rq.rng, nullptr, false, 0, false)); };
        check(a.res.text == want(a) && b.res.text == want(b) && c.res.text == want(c) && m.bad == 0 && a.res.cached_tokens == 0,
              std::string("cache off, pipeline ") + (pipe ? "on" : "off") + ": 3 cold prompts at once == --parallel 1's");
        s.shutdown();
    }
    g_pipeline = true;
}

// Follow-up turns with the reply snapshot (the gen-cache): turn 2 of two conversations at once restores prompt ++ reply.
void test_reply_snapshot(bool rows) {
    FakeQ27 m({300, 300}, rows);
    ie::LanesServe s(m, {});
    Cache ref;
    auto run_ref = [&](Req& r, uint32_t* rs) {
        return text_of(serial_ref(ref, r.ids, r.rq.snap_at, r.rq.sp.max_tokens, r.rq.rng, rs, r.rq.reply_cache));
    };
    std::vector<int32_t> pa = conv(29, 3000), pb = conv(36, 4000);
    Req a1(pa, 12, 21, 27), b1(pb, 17, 22, 34);
    a1.rq.reply_cache = b1.rq.reply_cache = true;
    uint32_t ra = 0, rb = 0;
    a1.run(s);
    const std::string wa1 = run_ref(a1, &ra);
    b1.run(s);
    const std::string wb1 = run_ref(b1, &rb);
    check(a1.res.text == wa1 && b1.res.text == wb1 && m.finishes.load() == 2, "turn 1 of A and B == --parallel 1's; the finish hook ran twice");
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
    check(ra2 == da && rb2 == db && a2.res.cached_tokens == da && b2.res.cached_tokens == db,
          "turn 2 restores the reply snapshots (" + std::to_string(a2.res.cached_tokens) + ", " + std::to_string(b2.res.cached_tokens) + " cached)");
    check(a2.res.text == wa2 && b2.res.text == wb2 && m.bad == 0, "turn 2 of A and B at once == --parallel 1's");
    s.shutdown();
}

// The shared prefix (IE_QWEN35_SHARED_PREFIX=1 turns it on for --parallel 1 AND the lanes): one system prefix, a new user turn
// per request; the first request marks it, later concurrent ones restore it into their own lane; every reply == --parallel 1's.
void test_agent_traffic(bool rows) {
    constexpr uint32_t kSys = 21;   // 4 x 5 + 1: the shared range ends on a 1-row pipelined piece (prompt_end inserts it)
    for (uint32_t sys_len : {kSys, 25u}) {
        const std::vector<int32_t> sys = conv(sys_len, 400);
        auto agent_req = [&](int k, uint32_t n) {
            std::vector<int32_t> ids = sys;
            for (int32_t v : conv(6 + uint32_t(k) % 5, 700 + 17 * k)) ids.push_back(v);
            const uint32_t snap = uint32_t(ids.size());
            ids.push_back(3); ids.push_back(5);
            return std::make_unique<Req>(ids, n, 900 + uint64_t(k), snap, sys_len);
        };
        FakeQ27 m({300, 120, 120, 120}, rows);
        ie::LanesServe s(m, {});
        Cache ref;
        std::vector<std::unique_ptr<Req>> rs;
        rs.push_back(agent_req(0, 12));
        rs[0]->run(s);
        uint32_t r0 = 0;
        const bool ok0 = text_of(serial_ref(ref, rs[0]->ids, rs[0]->rq.snap_at, 12, rs[0]->rq.rng, &r0, false, sys_len)) == rs[0]->res.text;
        for (int k = 1; k <= 3; ++k) rs.push_back(agent_req(k, 12));
        run_all({rs[1].get(), rs[2].get(), rs[3].get()}, s);
        bool all_share = true, all_ref = true;
        for (int k = 1; k <= 3; ++k) {
            Req& r = *rs[size_t(k)];
            all_share = all_share && r.res.cached_tokens == sys_len;
            Cache mine = ref;
            all_ref = all_ref && text_of(serial_ref(mine, r.ids, r.rq.snap_at, 12, r.rq.rng, nullptr, false, sys_len)) == r.res.text;
        }
        check(ok0 && all_share && all_ref && m.bad == 0,
              "shared prefix of " + std::to_string(sys_len) + " rows (" + (sys_len == kSys ? "inserted in prompt_end" : "the module's mark") +
                  "): request 1 cold, 2-4 at once restore it (cached == " + std::to_string(sys_len) + ") and == --parallel 1's");
        s.shutdown();
    }
}

}  // namespace

int main() {
    test_rules();
    for (bool rows : {false, true}) {
        std::printf("---- %s\n", rows ? "rows ON (grouped decode steps)" : "rows off (the per-lane pipe)");
        test_serve(rows);
        test_reply_snapshot(rows);
        test_agent_traffic(rows);
    }
    test_cache_and_pipeline_off();
    std::printf("%s\n", g_fail ? "Q27 LANES TEST: FAIL" : "Q27 LANES TEST: PASS");
    return g_fail ? 1 : 0;
}
