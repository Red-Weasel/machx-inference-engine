// tests/unit/lanes_serve_test.cpp -- host-only (no GPU, no SYCL): the shared lanes module (P4 B8, include/ie/lanes_serve.hpp,
// docs/lanes/LANES_SERVE.md) with a fake two-stage model, plus Flash-Next's host rules (include/ie/q4e_lanes.hpp) and GLM's
// pipe additions (Glm5LanePipe::set_lane_pos / error).
//
// The fake model: each lane keeps, per stage, the ids it has "forwarded"; a step at pos0 must start at the stage's own depth
// (a cross-lane mix-up or a lost step fails it), and a lane's "logits" are a hash of every id stage 1 has seen. So a request's
// ids are a pure function of its own prompt and a solo reference (ref_ids) predicts them: concurrent lanes, the serial turn,
// prefill through the pipe beside decoding lanes, the prompt cache and cancels must all leave each request == solo.
// Does NOT link ie_core.
#undef NDEBUG
#include "ie/lanes_serve.hpp"
#include "ie/q4e_lanes.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("[%s] %s\n", ok ? " ok " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}

constexpr int32_t kStop = 7;

uint64_t mix(uint64_t h, int32_t id) { return (h ^ uint64_t(uint32_t(id))) * 0x100000001B3ull; }
int32_t id_of(uint64_t h, uint64_t seed, uint32_t stop_at, size_t depth) {
    if (stop_at && depth == stop_at) return kStop;
    return int32_t(10 + ((h ^ seed) >> 7) % 990);
}

// What one request gives alone: greedy-like ids from the prompt's hash (seed = rng + k), up to n or the stop id.
std::vector<int32_t> ref_ids(const std::vector<int32_t>& prompt, uint32_t n, uint64_t rng, uint32_t stop_at) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (int32_t id : prompt) h = mix(h, id);
    std::vector<int32_t> out;
    size_t depth = prompt.size();
    for (uint32_t k = 0; k < n; ++k) {
        const int32_t id = id_of(h, rng + k, stop_at, depth);
        if (id == kStop) break;
        out.push_back(id);
        h = mix(h, id); ++depth;
    }
    return out;
}

std::string text_of(const std::vector<int32_t>& ids) {
    std::string s;
    for (int32_t id : ids) s += std::to_string(id) + ",";
    return s;
}

struct FakeModel final : ie::LanesModel {
    struct St { std::vector<int32_t> seq[2]; uint64_t h = 0xcbf29ce484222325ull, kept = 0; std::vector<int32_t> snap; uint32_t snap_d = 0; };
    std::vector<St> st;
    std::vector<uint32_t> cap;
    uint32_t chunk = 5;
    uint32_t stop_at = 0;                    // the depth at which the "model" emits kStop (0 = never)
    std::atomic<int> fail_stage{-1};         // inject: stage s fails its next step
    std::atomic<uint32_t> bad{0};
    std::atomic<int> active{0}, max_active{0};
    std::mutex mu;
    ie::CardPipe pipe;
    uint32_t active_lane = 0;
    // P4 B15: a shared-prefix store (LanesModel::mark / cache_peek), the pipe's prefill pieces in order, a slow prefix step
    bool shared = false;
    std::vector<std::vector<int32_t>> store;
    std::vector<std::pair<uint32_t, uint32_t>> marks;   // (lane, pos)
    std::atomic<uint32_t> mark_bad{0}, peek_in_mark{0};
    std::atomic<bool> in_mark{false};
    std::vector<uint32_t> pf_log;                       // the lanes of the pipe's prefill pieces (stage 0), in order
    int prep_sleep_ms = 0;
    // P4 B33: a prefill piece takes pf_ms 1-ms "layers" per stage (the pipe's and the serial turn's); abort() drops the pipe's
    // waiting steps and stops a running piece at its next layer boundary, as the crown's Q35mLanesModel::abort does
    std::atomic<int> pf_ms{0};
    std::atomic<bool> aborted{false};
    std::atomic<uint32_t> calls_after_abort{0}, preps{0}, serial_chunks{0};
    uint32_t abort() override { const uint32_t h = pipe.cancel(); aborted = true; return h; }
    std::string layers(uint32_t T) {
        for (int i = 0; T > 1 && i < pf_ms.load(); ++i) {
            if (aborted.load()) return "stopped at a layer boundary";
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return {};
    }

    // P4 B34: the lane pipeline (as the crown's Q35mLanesModel has it: the pipe with lookahead slots, its idle hook); `tl` =
    // every prefill piece's stage run (stage, lane, pos0, t0, t1 ms) while `record`; a piece of lane hold_lane at hold_pos
    // holds stage 1, the one after it holds stage 0, both until abort()
    bool pipeline_on = false;
    std::atomic<bool> record{false};
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t, double, double>> tl;
    std::atomic<int> hold_lane{-1};
    std::atomic<uint32_t> hold_pos{0};
    std::atomic<bool> held_at[2] = {false, false};
    const std::chrono::steady_clock::time_point t_base = std::chrono::steady_clock::now();
    double now_ms() const { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_base).count(); }
    bool pipe_lookahead(std::function<void()> idle) override {
        if (!pipeline_on) return false;
        pipe.set_idle(std::move(idle));
        return true;
    }
    std::string pipe_submit_ahead(uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0, bool& taken) override {
        return pipe.submit_ahead(lane, ids, T, pos0, taken);
    }

    explicit FakeModel(std::vector<uint32_t> caps, bool pipeline = false)
        : st(caps.size()), cap(caps),
          pipe(uint32_t(caps.size()), 8, 1, 2, [this](uint32_t s, const ie::Glm5LanePipe::Step& x) { return stage(s, x); }, pipeline) {
        pipeline_on = pipeline;
    }

    std::string forward(uint32_t lane, uint32_t s, const int32_t* ids, uint32_t T, uint32_t pos0) {
        St& L = st[lane];
        if (pos0 == 0) L.seq[s].clear();
        if (L.seq[s].size() != pos0) { ++bad; return "stage " + std::to_string(s) + " at " + std::to_string(L.seq[s].size()) + ", step at " + std::to_string(pos0); }
        L.seq[s].insert(L.seq[s].end(), ids, ids + T);
        if (s == 1) {
            L.h = 0xcbf29ce484222325ull;
            for (int32_t id : L.seq[1]) L.h = mix(L.h, id);
        }
        return {};
    }
    std::string stage(uint32_t s, const ie::Glm5LanePipe::Step& x) {
        if (aborted.load()) ++calls_after_abort;
        const double t0 = now_ms();
        const int a = ++active;
        for (int m = max_active.load(); a > m && !max_active.compare_exchange_weak(m, a);) {}
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        --active;
        if (fail_stage.load() == int(s)) { fail_stage = -1; return "injected failure"; }
        if (s == 0 && x.T > 1) { std::lock_guard<std::mutex> g(mu); pf_log.push_back(x.lane); }
        if (int(x.lane) == hold_lane.load() && x.T > 1 && x.pos0 == hold_pos.load() + (s == 0 ? chunk : 0)) {   // (P4 B34)
            held_at[s] = true;
            while (!aborted.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            return "stopped at a layer boundary";
        }
        if (auto e = layers(x.T); !e.empty()) return e;
        std::string e = forward(x.lane, s, x.ids, x.T, x.pos0);
        if (record.load() && x.T > 1) { std::lock_guard<std::mutex> g(mu); tl.emplace_back(s, x.lane, x.pos0, t0, now_ms()); }
        return e;
    }
    const char* tag() const override { return "fake lanes"; }
    uint32_t n_lanes() const override { return uint32_t(st.size()); }
    uint32_t lane_cap(uint32_t lane) const override { return cap[lane]; }
    uint32_t own_match(uint32_t lane, std::span<const int32_t> ids) const override { return ie::q4e_snap_match(st[lane].snap, ids); }
    bool occupied(uint32_t lane) const override { return !st[lane].snap.empty(); }
    uint32_t last_end(uint32_t lane) const override { return uint32_t(st[lane].snap.size()); }
    uint32_t shared_match(const std::vector<int32_t>& ids) const {
        uint32_t best = 0;
        for (const auto& e : store)
            if (e.size() < ids.size() && e.size() > best && std::equal(e.begin(), e.end(), ids.begin())) best = uint32_t(e.size());
        return best;
    }
    std::string mark(uint32_t lane, const ie::LanesRequest& rq, uint32_t pos) override {
        in_mark = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        const St& L = st[lane];
        if (L.seq[0].size() != pos || L.seq[1].size() != pos || pos != rq.share_at) ++mark_bad;
        std::vector<int32_t> t(rq.ids->begin(), rq.ids->begin() + pos);
        bool have = false;
        for (const auto& e : store) have = have || e == t;
        if (!have) store.push_back(t);
        marks.emplace_back(lane, pos);
        in_mark = false;
        return {};
    }
    uint32_t cache_peek(const ie::LanesRequest& rq) const override {
        if (in_mark.load()) const_cast<FakeModel*>(this)->peek_in_mark++;
        return shared ? shared_match(*rq.ids) : 0;
    }
    std::string prefix_prepare(uint32_t lane, const ie::LanesRequest& rq, uint32_t& reused, std::string& source) override {
        ++preps;
        if (prep_sleep_ms) std::this_thread::sleep_for(std::chrono::milliseconds(prep_sleep_ms));
        St& L = st[lane];
        const uint32_t D = ie::q4e_snap_match(L.snap, *rq.ids);
        if (shared) {
            const uint32_t S = shared_match(*rq.ids);
            if (S > D) {   // the shared entry's state: the prefix's ids on both stages
                L.seq[0].assign(rq.ids->begin(), rq.ids->begin() + S); L.seq[1] = L.seq[0];
                L.snap.clear(); L.snap_d = 0;
                reused = S; source = "shared";
                return {};
            }
        }
        if (D && L.snap_d == D && L.seq[0].size() >= D && L.seq[1].size() >= D) {
            L.seq[0].resize(D); L.seq[1].resize(D);
            reused = D; source = "snapshot";
            return {};
        }
        L.seq[0].clear(); L.seq[1].clear(); L.snap.clear(); L.snap_d = 0;
        reused = 0; source = "none";
        return {};
    }
    ie::LanesPlan plan(const ie::LanesRequest& rq, uint32_t reused) override {
        ie::LanesPlan p;
        const uint32_t T = uint32_t(rq.ids->size());
        p.Tp = rq.snap_at > reused ? rq.snap_at : T;
        if (shared && rq.share_at > reused && rq.share_at < p.Tp) {   // P4 B15: split at the shared prefix, marked
            ie::q4e_plan_range(reused, rq.share_at, chunk, false, p.chunks);
            ie::q4e_plan_range(rq.share_at, p.Tp, chunk, false, p.chunks);
            p.mark = rq.share_at;
        } else {
            ie::q4e_plan_range(reused, p.Tp, chunk, false, p.chunks);
        }
        return p;
    }
    std::string prefill_serial(uint32_t lane, const ie::LanesRequest& rq, std::span<const ie::LanesChunk> ch,
                               const std::function<bool()>& stop, size_t& done) override {
        done = 0;
        for (size_t k = 0; k < ch.size(); ++k) {
            if (k > 0 && stop && stop()) break;
            ++serial_chunks;
            for (uint32_t s = 0; s < 2; ++s) {
                if (auto e = layers(ch[k].second); !e.empty()) return e;
                if (auto e = forward(lane, s, rq.ids->data() + ch[k].first, ch[k].second, ch[k].first); !e.empty()) return e;
            }
            ++done;
        }
        return {};
    }
    std::string prompt_end(uint32_t lane, const ie::LanesRequest& rq, uint32_t Tp, uint32_t reused, bool kept) override {
        St& L = st[lane];
        const uint32_t T = uint32_t(rq.ids->size());
        if (rq.snap_at > reused && Tp == rq.snap_at) { L.snap.assign(rq.ids->begin(), rq.ids->begin() + Tp); L.snap_d = Tp; }
        if (Tp < T) {
            for (uint32_t s = 0; s < 2; ++s)
                if (auto e = forward(lane, s, rq.ids->data() + Tp, T - Tp, Tp); !e.empty()) return e;
            kept = false;
        }
        if (!kept) L.kept = L.h;
        return {};
    }
    std::string reset_lane(uint32_t lane) override { st[lane] = St{}; return pipe.reset_lane(lane); }   // as Flash-Next's
    int32_t sample(uint32_t lane, bool first, const ie::LanesSampling&, std::span<const int32_t>, uint64_t seed, std::string&) override {
        const St& L = st[lane];
        return id_of(first ? L.kept : L.h, seed, stop_at, L.seq[1].size());
    }
    void keep_logits(uint32_t lane) override { st[lane].kept = st[lane].h; }
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

std::vector<int32_t> prompt_of(uint32_t n, int32_t base) {
    std::vector<int32_t> p(n);
    for (uint32_t i = 0; i < n; ++i) p[i] = base + int32_t(i % 97);
    return p;
}

struct Req {
    std::vector<int32_t> ids;
    ie::LanesRequest rq;
    ie::LanesResult res;
    std::string streamed;
    int cancel_after = -1;   // decline the callback after this many non-empty pieces
    int pieces = 0;
    Req(std::vector<int32_t> p, uint32_t n, uint64_t rng, uint32_t snap_at = 0) : ids(std::move(p)) {
        rq.ids = &ids; rq.sp.max_tokens = n; rq.rng = rng; rq.snap_at = snap_at ? snap_at : uint32_t(ids.size());
    }
    void run(ie::LanesServe& s) {
        res = s.run(rq, [this](std::string_view piece) {
            if (piece.empty()) return true;
            streamed += piece;
            ++pieces;
            return !(cancel_after >= 0 && pieces >= cancel_after);
        });
    }
};

void test_q4e_rules() {
    ie::Q4eLaneShape sh;   // Flash-Next stage 0 ([0, 24): 6 full + 18 DeltaNet, PLE on blk.1)
    sh.n_full = 6; sh.n_lin = 18; sh.n_kv_heads = 2; sh.head_dim = 256; sh.idx_head_dim = 128;
    sh.v_heads = 48; sh.state = 128; sh.conv_channels = 10240; sh.conv_kernel = 4; sh.ple_conv_floats = 9 * 10240;
    const uint64_t b32 = ie::q4e_lane_bytes(sh, 32768);
    // KV 384 MiB + indexer 48 + 24 MiB + DeltaNet (state 54 + conv 1.05 MiB) x2 (live + snapshot) + PLE 0.35 MiB x2
    const uint64_t want = 402653184ull + 50331648ull + 25165824ull + 2ull * (18ull * (3145728ull + 61440ull) + 368640ull);
    check(b32 == want, "q4e_lane_bytes at 32K = " + std::to_string(b32) + " B (" + std::to_string(b32 >> 20) + " MiB; design ~0.5 GiB/stage)");
    {   // P4 B14: the expert-cache floor of the extra lanes (q4e_lanes_cache_fit); slot = one expert slot over the stage's layers
        const uint64_t slot = 75ull << 20, budget = 298 * slot;   // synthetic: ~B8's 298 slots/layer (not the model's bytes)
        uint64_t left = 0;
        check(ie::q4e_lanes_cache_fit(budget, b32, 15, 32768, slot, left).empty() && left == budget - 15 * b32 && left / slot >= 64,
              "q4e_lanes_cache_fit: 15 extra lanes at 32K leave " + std::to_string(left / slot) + " slots/layer (>= 64)");
        const uint64_t at_floor = 64 * slot + 3 * b32;   // exactly 64 slots after 3 extra lanes
        check(ie::q4e_lanes_cache_fit(at_floor, b32, 3, 32768, slot, left).empty() && left == 64 * slot,
              "q4e_lanes_cache_fit: 64 slots/layer is admitted");
        const std::string m = ie::q4e_lanes_cache_fit(at_floor - 1, b32, 3, 32768, slot, left);
        check(m == "qwen4exp lanes: 3 extra lane(s) of " + std::to_string(b32 >> 20) + " MiB at ctx 32768 leave " +
                   std::to_string(left >> 20) + " MiB of the " + std::to_string((at_floor - 1) >> 20) +
                   " MiB expert-cache budget = 63 slots/layer (< 64); lower --parallel or --slot-ctx",
              "q4e_lanes_cache_fit: 63 slots/layer refused with the numbers: " + m);
        check(!ie::q4e_lanes_cache_fit(b32, b32, 15, 32768, slot, left).empty() && left == 0,
              "q4e_lanes_cache_fit: lanes above the budget leave 0, refused");
    }
    check(ie::q4e_lane_ctx(0, 262144) == 32768 && ie::q4e_lane_ctx(0, 16384) == 16384 && ie::q4e_lane_ctx(8192, 65536) == 8192,
          "q4e_lane_ctx: --slot-ctx 0 = 32K, capped at the context");
    const std::vector<int32_t> snap{1, 2, 3}, ids{1, 2, 3, 4}, same{1, 2, 3}, other{1, 9, 3, 4};
    check(ie::q4e_snap_match(snap, ids) == 3 && ie::q4e_snap_match(snap, same) == 0 && ie::q4e_snap_match(snap, other) == 0 &&
          ie::q4e_snap_match({}, ids) == 0, "q4e_snap_match: strict prefix only");
    std::vector<ie::LanesChunk> c;
    ie::q4e_plan_range(0, 5000, 4096, true, c);
    check(c.size() == 3 && c[0] == ie::LanesChunk(0, 2048) && c[2] == ie::LanesChunk(4096, 904),
          "plan: > 2048 rows = fwd_pipelined's 2048-row chunks");
    c.clear(); ie::q4e_plan_range(100, 2148, 4096, true, c);
    check(c.size() == 1 && c[0] == ie::LanesChunk(100, 2048), "plan: <= 2048 rows = one serial chunk (pf_chunk 4096)");
    c.clear(); ie::q4e_plan_range(0, 5000, 4096, false, c);
    check(c.size() == 2 && c[0].second == 4096, "plan: IE_Q4E_NO_PIPELINE = pf_chunk chunks");
    check(ie::q4e_plan_max_rows(4096, true) == 2048 && ie::q4e_plan_max_rows(4096, false) == 4096, "plan max rows");
    {   // P4 B17 rows: the group cap (the expert cache holds the union of the group's top-k picks) and the route rule
        check(ie::q4e_rows_max_group(298, 10) == 16 && ie::q4e_rows_max_group(160, 10) == 16 && ie::q4e_rows_max_group(159, 10) == 15 &&
              ie::q4e_rows_max_group(103, 10) == 10 && ie::q4e_rows_max_group(64, 10) == 6 && ie::q4e_rows_max_group(19, 10) == 1 &&
              ie::q4e_rows_max_group(100, 0) == 0,
              "q4e_rows_max_group: min(16, slots / top_k) (298 -> 16, 159 -> 15, the 64-slot floor -> 6)");
        ie::Q4eRowsRoute r;
        r.n_lanes = 4; r.slots_per_layer = 190; r.top_k = 10;
        check(ie::q4e_rows_route_refusal(r).empty(), "q4e_rows_route_refusal: the default route with 4 lanes serves rows");
        auto refused = [&](auto set, const char* what) {
            ie::Q4eRowsRoute x = r;
            set(x);
            const std::string m = ie::q4e_rows_route_refusal(x);
            check(!m.empty(), std::string("q4e_rows_route_refusal: ") + what + " -> \"" + m + "\"");
        };
        refused([](ie::Q4eRowsRoute& x) { x.n_lanes = 1; }, "one lane");
        refused([](ie::Q4eRowsRoute& x) { x.dense_q8 = false; }, "IE_Q4E_DENSE_Q8=0");
        refused([](ie::Q4eRowsRoute& x) { x.a16 = true; }, "IE_Q4E_A16=1");
        refused([](ie::Q4eRowsRoute& x) { x.smallk = true; }, "IE_GEMV_SMALLK");
        refused([](ie::Q4eRowsRoute& x) { x.decode_grouped = false; }, "IE_Q4E_DECODE_GROUPED=0");
        refused([](ie::Q4eRowsRoute& x) { x.moe_alt = true; }, "a non-default MoE route");
        refused([](ie::Q4eRowsRoute& x) { x.tp = true; }, "verify expert parallelism");
        refused([](ie::Q4eRowsRoute& x) { x.dn_mixed = true; }, "a mixed-dtype DeltaNet layer");
        refused([](ie::Q4eRowsRoute& x) { x.slots_per_layer = 19; }, "19 slots/layer (< 2 x top-10)");
        ie::Q4eRowsRoute y = r;
        y.slots_per_layer = 20;
        check(ie::q4e_rows_route_refusal(y).empty(), "q4e_rows_route_refusal: 20 slots/layer = groups of 2");
    }
    check(ie::q4e_vision_refusal(1).empty() && ie::q4e_vision_refusal(0).empty() &&
          ie::q4e_vision_refusal(2).find("--parallel 1") != std::string::npos,
          "q4e_vision_refusal: images at --parallel 1 only (/props and Engine::chat)");
    check(ie::ds4_vision_refusal(true, 1).empty() && ie::ds4_vision_refusal(true, 0).empty(),
          "ds4_vision_refusal: sidecar loaded, --parallel 1: images taken");
    check(ie::ds4_vision_refusal(true, 2) == ie::q4e_vision_refusal(2) && ie::ds4_vision_refusal(true, 4) == ie::q4e_vision_refusal(4),
          "ds4_vision_refusal: sidecar loaded, --parallel > 1: refused, engine-global staging");
    check(ie::ds4_vision_refusal(false, 1).find("no vision sidecar") != std::string::npos &&
          ie::ds4_vision_refusal(false, 2).find("no vision sidecar") != std::string::npos,
          "ds4_vision_refusal: no sidecar: the sidecar reason first, whatever --parallel");
}

void test_choose() {
    using V = ie::LanesLaneView;
    std::vector<V> l(3);
    l[0] = {true, false, 1000, 0, 0, 1}; l[1] = {true, false, 200, 0, 0, 2}; l[2] = {true, true, 200, 0, 0, 0};
    check(ie::lanes_choose_lane(l, 50, 40, 1024) == 1, "choose: the smallest empty lane with reply room");
    check(ie::lanes_choose_lane(l, 170, 400, 1024) == 0, "choose: reply room first (170 + a quarter of a 200 lane does not fit)");
    l[2].match = 30; l[2].last_end = 30;
    check(ie::lanes_choose_lane(l, 50, 40, 1024) == 2, "choose: the lane whose own snapshot continues the prompt");
    l[0].idle = false;
    check(ie::lanes_choose_lane(l, 500, 40, 1024) == -1, "choose: none fits -> -1");
    check(ie::lanes_reply_budget(10, 5, 100) == 5 && ie::lanes_reply_budget(90, 50, 100) == 10 && ie::lanes_reply_budget(10, 0, 100) == 90,
          "reply budget");
    const std::vector<int32_t> h{1, 2, 3, 4, 5};
    const auto w = ie::lanes_penalty_window(h, 3);
    check(w.size() == 3 && w[0] == 3 && w[2] == 5, "penalty window = the last repeat_window ids");
}

void test_glm_pipe_additions() {
    ie::Glm5LanePipe p(2, 4, 1);
    std::atomic<int> fail{0};
    std::promise<void> got;
    std::atomic<bool> once{false};
    p.start(1, [&](uint32_t, const ie::Glm5LanePipe::Step&) { return fail.load() ? std::string("boom") : std::string(); },
            [&](uint32_t, uint32_t, uint32_t) { if (!once.exchange(true)) got.set_value(); });
    check(p.set_lane_pos(0, 17).empty() && p.lane_pos(0) == 17, "set_lane_pos: an idle lane");
    const int32_t id = 1;
    check(p.submit(0, &id, 1, 17).empty(), "submit at the synced position");
    got.get_future().wait();
    check(p.lane_pos(0) == 18, "the step committed 18");
    fail = 1;
    check(p.submit(1, &id, 1, 0).empty(), "a failing step");
    for (int i = 0; i < 200 && p.error().empty(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    check(!p.error().empty(), "error() shows the latched stage error: " + p.error());
    check(!p.stop().empty() && p.error().empty(), "stop() returns it and clears it");
    check(!p.set_lane_pos(5, 1).empty(), "set_lane_pos refuses an unknown lane");
    // P4 B9: lane 1's step failed part-way; its position may not move until it is reset
    const std::string e1 = p.set_lane_pos(1, 3);
    check(!e1.empty() && e1.find("failed part-way") != std::string::npos, "set_lane_pos refuses a failed lane: " + e1);
    check(p.reset_lane(1).empty() && p.lane_pos(1) == 0, "reset_lane: position 0");
    check(p.set_lane_pos(1, 3).empty() && p.lane_pos(1) == 3, "set_lane_pos after reset_lane");
    check(!p.reset_lane(5).empty(), "reset_lane refuses an unknown lane");
}

// P4 B9 (B8 gate finding 3): CardPipe::submit honours the position sync. A fresh submit at a moved position syncs (the happy
// path); on a lane whose last step failed part-way it is refused -- visibly, the lane not left claimed -- until reset_lane.
void test_card_pipe_sync() {
    std::atomic<int> fail{0}, done{0};
    ie::CardPipe cp(2, 4, 1, 1, [&](uint32_t, const ie::Glm5LanePipe::Step&) { return fail.load() ? std::string("boom") : std::string(); });
    auto wait_done = [&](int n) { for (int i = 0; i < 500 && done.load() < n; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(2)); return done.load() >= n; };
    check(cp.start([&](uint32_t) { ++done; }).empty(), "card pipe: start");
    const int32_t id = 1;
    check(cp.submit(0, &id, 1, 5).empty() && wait_done(1), "card pipe: a fresh submit at a moved position syncs and runs");
    check(cp.submit(0, &id, 1, 6).empty() && wait_done(2), "card pipe: the next step at the committed position");
    fail = 1;
    check(cp.submit(1, &id, 1, 0).empty(), "card pipe: a failing step");
    for (int i = 0; i < 500 && cp.error().empty(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(2));
    check(!cp.stop().empty(), "card pipe: stop returns the stage error");
    fail = 0;
    check(cp.start([&](uint32_t) { ++done; }).empty(), "card pipe: a new pipe (LanesServe's recovery)");
    const std::string e = cp.submit(1, &id, 1, 4);
    check(!e.empty() && e.find("failed part-way") != std::string::npos, "card pipe: a fresh submit on the failed lane is refused: " + e);
    check(cp.pause().empty() && cp.resume().empty(), "card pipe: the refused submit left no step in flight (pause returns)");
    check(cp.reset_lane(1).empty(), "card pipe: reset_lane");
    check(cp.submit(1, &id, 1, 4).empty() && wait_done(3), "card pipe: after reset_lane the lane syncs and runs");
    check(cp.stop().empty(), "card pipe: stop");
}

// P4 B14: CardPipe's rows mode -- a rows stage alone does not change the pipe (start without a rows callback = per lane);
// start(done, rows_done) groups; pause/resume keeps the mode.
void test_card_pipe_rows() {
    std::atomic<int> plain{0}, grouped{0}, done_n{0}, rows_cb{0};
    std::atomic<bool> hold{true};
    ie::CardPipe cp(3, 4, 1, 1,
                    [&](uint32_t, const ie::Glm5LanePipe::Step&) {
                        ++plain;
                        while (hold.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        return std::string();
                    },
                    [&](uint32_t, std::span<const ie::Glm5LanePipe::Step> st, float*) { grouped += int(st.size()); return std::string(); },
                    16, 0);
    auto wait = [&](const std::atomic<int>& a, int n) { for (int i = 0; i < 1000 && a.load() < n; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(2)); return a.load() >= n; };
    const int32_t id = 1;
    {   // B14 gate #5: the rows constructor makes the group pool's host buffers at load (4 groups x 16 rows x 1 float)
        ie::CardPipe plain_cp(3, 4, 1, 1, [](uint32_t, const ie::Glm5LanePipe::Step&) { return std::string(); });
        check(cp.host_bytes() == plain_cp.host_bytes() + 4 * 16 * sizeof(float),
              "card pipe rows: the group buffers are counted from construction (" + std::to_string(cp.host_bytes()) + " vs " +
              std::to_string(plain_cp.host_bytes()) + " B)");
    }
    // per lane: no rows callback given
    hold = false;
    check(cp.start([&](uint32_t) { ++done_n; }).empty() && !cp.rows_mode(), "card pipe rows: start(done) alone = the per-lane pipe");
    for (uint32_t l = 0; l < 3; ++l) (void)cp.submit(l, &id, 1, 0);
    check(wait(done_n, 3) && grouped.load() == 0, "card pipe rows: per-lane pipe never calls the rows stage");
    check(cp.stop().empty(), "card pipe rows: stop");
    // rows: lane 0 holds the stage (a lone group) while lanes 1, 2 queue up -> they run as one group of 2 (AUTO cap 2 at 3 busy)
    hold = true; plain = 0; done_n = 0;
    check(cp.start([&](uint32_t) { ++done_n; }, [&](std::span<const uint32_t> ls) { rows_cb += int(ls.size()); }).empty() && cp.rows_mode(),
          "card pipe rows: start(done, rows_done) = rows mode");
    (void)cp.submit(0, &id, 1, 1);
    check(wait(plain, 1), "card pipe rows: lane 0 runs alone");
    (void)cp.submit(1, &id, 1, 1); (void)cp.submit(2, &id, 1, 1);
    hold = false;
    check(wait(rows_cb, 2) && grouped.load() == 2 && done_n.load() == 1, "card pipe rows: the two queued lanes ran as one group, one rows callback");
    check(cp.pause().empty() && cp.resume().empty() && cp.rows_mode(), "card pipe rows: pause/resume keeps rows mode");
    hold = true; plain = 0;
    (void)cp.submit(0, &id, 1, 2);
    check(wait(plain, 1), "card pipe rows (resumed): lane 0 alone");
    (void)cp.submit(1, &id, 1, 2); (void)cp.submit(2, &id, 1, 2);
    hold = false;
    check(wait(rows_cb, 4) && grouped.load() == 4, "card pipe rows (resumed): the queued lanes grouped again");
    check(cp.stop().empty(), "card pipe rows: stop");
    const auto gs = cp.group_sizes();
    check(gs.size() == 17 && gs[1] == 2 && gs[2] == 2, "card pipe rows: group sizes over both pipes: two lone, two pairs (" + std::to_string(gs.size() > 2 ? gs[2] : 0) + " of 2)");
}

void test_serve() {
    // three lanes: lane 0 cap 400, lanes 1-2 cap 120
    {
        FakeModel m({400, 120, 120});
        ie::LanesServe s(m, {});
        Req a(prompt_of(23, 100), 30, 11);
        a.run(s);
        check(a.res.finish_reason == "length" && a.res.text == text_of(ref_ids(a.ids, 30, 11, 0)) && a.streamed == a.res.text,
              "solo request == its reference (" + a.res.finish_reason + ", " + std::to_string(a.res.completion_tokens) + " ids)");
        check(m.bad == 0, "no stage saw a wrong position");

        // three at once: prefill through the pipe beside decoding lanes; each == its own reference
        std::vector<Req> rs;
        rs.reserve(3);   // (a Req points at its own ids: no reallocation)
        rs.emplace_back(prompt_of(37, 200), 40, 5);
        rs.emplace_back(prompt_of(12, 300), 25, 6);
        rs.emplace_back(prompt_of(58, 400), 33, 7);
        std::vector<std::thread> th;
        for (auto& r : rs) th.emplace_back([&r, &s] { r.run(s); });
        for (auto& t : th) t.join();
        bool all = true;
        for (auto& r : rs) all = all && r.res.text == text_of(ref_ids(r.ids, r.rq.sp.max_tokens, r.rq.rng, 0)) && r.res.finish_reason == "length";
        check(all, "3 concurrent requests: each == its solo reference");
        check(m.max_active.load() >= 2, "the two stages overlapped (max " + std::to_string(m.max_active.load()) + " stages at once)");
        check(m.bad == 0, "no stage saw a wrong position (no cross-lane mix-up)");

        // a follow-up turn continues from its lane's snapshot; the reply equals the reference
        Req f(prompt_of(23, 1234), 10, 11, 20);
        f.run(s);
        Req g(prompt_of(23, 1234), 10, 11, 0);
        g.ids.push_back(5); g.ids.push_back(6); g.rq.snap_at = uint32_t(g.ids.size());
        g.run(s);
        check(g.res.cached_tokens == 20 && g.res.text == text_of(ref_ids(g.ids, 10, 11, 0)),
              "a follow-up restores its lane's snapshot (" + std::to_string(g.res.cached_tokens) + " cached) and == reference");

        // a cancel mid-batch: the survivor == its reference
        Req c1(prompt_of(30, 500), 60, 3), c2(prompt_of(31, 600), 60, 4, 25);   // c2: the pipe's part ends at 25, the rest in a turn
        c1.cancel_after = 5;
        std::thread t1([&] { c1.run(s); }), t2([&] { c2.run(s); });
        t1.join(); t2.join();
        check(c1.res.finish_reason == "abort", "the cancelled request ends with abort");
        check(c2.res.text == text_of(ref_ids(c2.ids, 60, 4, 0)) && c2.res.finish_reason == "length", "the survivor == its reference");

        // a stop id ends the reply with "stop", not delivered
        m.stop_at = 29;
        Req st(prompt_of(25, 700), 20, 9);
        st.run(s);
        check(st.res.finish_reason == "stop" && st.res.text == text_of(ref_ids(st.ids, 20, 9, 29)), "a stop id -> \"stop\", not delivered");
        m.stop_at = 0;

        // a prompt only lane 0 holds waits for it while another request decodes there
        Req big1(prompt_of(200, 800), 40, 1), big2(prompt_of(210, 900), 20, 2);
        std::thread b1([&] { big1.run(s); });
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        std::thread b2([&] { big2.run(s); });
        b1.join(); b2.join();
        check(big1.res.lane == 0 && big2.res.lane == 0 && big2.res.text == text_of(ref_ids(big2.ids, 20, 2, 0)),
              "two lane-0-only prompts run one after the other on lane 0");

        // a stage error: the running requests fail, the lanes are reset, the next request works
        Req e1(prompt_of(20, 1000), 200, 8);
        std::thread te([&] { e1.run(s); });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        m.fail_stage = 1;
        te.join();
        check(e1.res.finish_reason.rfind("error:", 0) == 0, "a stage error fails the running request: " + e1.res.finish_reason);
        Req after(prompt_of(21, 1100), 15, 12);
        after.run(s);
        check(after.res.text == text_of(ref_ids(after.ids, 15, 12, 0)), "after the error a new request == its reference");
        const std::string js = s.status_json();
        check(js.front() == '{' && js.find("\"lanes\":3") != std::string::npos && js.find("\"tokens\":") != std::string::npos,
              "status_json: " + js);
        s.shutdown();
        s.shutdown();   // idempotent
    }
    check(true, "teardown with the pipe and the serial worker joined");
}

}  // namespace

namespace {
// The B9 gate's extra test (~/ds41_work/p60/b9gate/gate_extra_test.cpp): a stage error on a ONE-lane server, three rounds
// (stage 1, 0, 1), so the next request MUST reuse the failed lane -- which works only when the model's reset_lane resets the
// pipe's record of it (CardPipe::reset_lane).
void test_failed_lane_reuse() {
    FakeModel m({400});
    ie::LanesServe s(m, {});
    for (int round = 0; round < 3; ++round) {
        Req e1(prompt_of(20, 1000 + round), 200, 8);
        std::thread te([&] { e1.run(s); });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        m.fail_stage = round & 1 ? 0 : 1;
        te.join();
        check(e1.res.finish_reason.rfind("error:", 0) == 0, "one lane, round " + std::to_string(round) + ": the stage error fails the request");
        Req after(prompt_of(21, 1100 + round), 15, 12);
        after.run(s);
        check(after.res.lane == 0 && after.res.text == text_of(ref_ids(after.ids, 15, 12, 0)),
              "one lane, round " + std::to_string(round) + ": the failed lane 0 is reused and == reference (" + after.res.finish_reason + ")");
    }
    check(m.bad == 0, "one lane: no stage saw a wrong position");
    s.shutdown();
}
}  // namespace


namespace {
std::string status_field(ie::LanesServe& s, const std::string& k) {
    const std::string js = s.status_json();
    const size_t p = js.find("\"" + k + "\":");
    if (p == std::string::npos) return "";
    const size_t a = p + k.size() + 3, b = js.find_first_of(",}", a);
    return js.substr(a, b - a);
}
std::vector<int32_t> with_tail(const std::vector<int32_t>& sys, int32_t base, uint32_t n) {
    std::vector<int32_t> v = sys;
    for (uint32_t i = 0; i < n; ++i) v.push_back(base + int32_t(i));
    return v;
}

// P4 B15: the shared-prefix mark and restore, in the serial turn and through the pipe; agent traffic (one system prefix, new
// user turns) arriving at once: one snapshot, the others restore it (their FIFO re-prepare), every reply == its reference.
void test_shared_prefix() {
    FakeModel m({400, 400, 400, 400, 400, 400});
    m.shared = true;
    ie::LanesServe::Options so;
    so.prefill_fifo = true;   // (the re-prepare onto a new shared prefix needs the FIFO; default off)
    ie::LanesServe s(m, so);
    const std::vector<int32_t> sys = prompt_of(40, 3000);
    Req a(with_tail(sys, 5000, 7), 20, 21); a.rq.share_at = 40;
    a.run(s);
    check(a.res.cached_tokens == 0 && a.res.text == text_of(ref_ids(a.ids, 20, 21, 0)) && m.marks.size() == 1 &&
          m.marks[0].second == 40 && m.mark_bad == 0, "shared prefix, serial turn: one mark at 40, reply == reference");
    Req b(with_tail(sys, 6000, 9), 20, 22); b.rq.share_at = 40;
    b.run(s);
    check(b.res.cached_tokens == 40 && b.res.cache_source == "shared" && b.res.text == text_of(ref_ids(b.ids, 20, 22, 0)) &&
          m.marks.size() == 1, "shared prefix: a new user turn restores the 40-token prefix (" + std::to_string(b.res.cached_tokens) +
          " cached) and == its cold reference");

    // through the pipe: another lane decodes, so the prompt's pieces go through the pipe and the mark takes a turn
    const std::vector<int32_t> sys2 = prompt_of(40, 4000);
    Req d(prompt_of(15, 9000), 250, 30);
    std::thread td([&] { d.run(s); });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    Req c(with_tail(sys2, 7000, 8), 20, 23); c.rq.share_at = 40;
    c.run(s);
    check(c.res.text == text_of(ref_ids(c.ids, 20, 23, 0)) && m.marks.size() == 2 && m.marks[1].second == 40 && m.mark_bad == 0,
          "shared prefix through the pipe: the mark ran in a turn at exactly 40, reply == reference");

    // four agent requests at once behind a decoding lane: one mark; the three others restore (re-prepared at the FIFO head)
    const std::vector<int32_t> sys3 = prompt_of(60, 8000);
    std::vector<Req> rs;
    rs.reserve(4);
    for (int k = 0; k < 4; ++k) { rs.emplace_back(with_tail(sys3, 10000 + 100 * k, 6 + k), 20, 40 + k); rs.back().rq.share_at = 60; }
    const size_t marks0 = m.marks.size();
    std::vector<std::thread> th;
    for (auto& r : rs) th.emplace_back([&r, &s] { r.run(s); });
    for (auto& t : th) t.join();
    int restored = 0;
    bool all = true;
    for (auto& r : rs) {
        all = all && r.res.text == text_of(ref_ids(r.ids, 20, r.rq.rng, 0));
        restored += r.res.cached_tokens == 60;
    }
    check(all, "4 agent requests at once: each == its cold reference");
    check(m.marks.size() - marks0 == 1 && restored == 3,
          "4 agent requests at once: 1 snapshot, " + std::to_string(restored) + " of 4 restored it (repreps " + status_field(s, "repreps") + ")");
    td.join();
    check(d.res.text == text_of(ref_ids(d.ids, 250, 30, 0)), "the decoding lane == its reference through the marks and turns");
    check(m.bad == 0 && m.mark_bad == 0 && m.peek_in_mark == 0, "no wrong position, every mark at its depth, cache_peek never beside a mark");
    s.shutdown();
}

// P4 B15: the prefill FIFO -- prompts arriving together behind a decoding lane go through the pipe one at a time, in order
// (their pieces never interleave), and each reply == its reference.
void test_prefill_fifo() {
    for (int fifo = 1; fifo >= 0; --fifo) {
        FakeModel m({400, 400, 400, 400});
        ie::LanesServe::Options o;
        o.prefill_fifo = fifo != 0;
        ie::LanesServe s(m, o);
        Req d(prompt_of(10, 9100), 300, 31);
        std::thread td([&] { d.run(s); });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        std::vector<Req> rs;
        rs.reserve(3);
        for (int k = 0; k < 3; ++k) rs.emplace_back(prompt_of(60, 11000 + 500 * k), 15, 50 + k);
        std::vector<std::thread> th;
        for (auto& r : rs) th.emplace_back([&r, &s] { r.run(s); });
        for (auto& t : th) t.join();
        td.join();
        bool all = d.res.text == text_of(ref_ids(d.ids, 300, 31, 0));
        for (auto& r : rs) all = all && r.res.text == text_of(ref_ids(r.ids, 15, r.rq.rng, 0));
        uint32_t switches = 0;
        std::vector<uint32_t> seenl;
        for (size_t i = 0; i < m.pf_log.size(); ++i) {
            if (i && m.pf_log[i] != m.pf_log[i - 1]) ++switches;
            if (std::find(seenl.begin(), seenl.end(), m.pf_log[i]) == seenl.end()) seenl.push_back(m.pf_log[i]);
        }
        size_t overlap = 0;   // the most prompts whose pieces were running at once (first..last piece intervals)
        for (size_t i = 0; i < m.pf_log.size(); ++i) {
            size_t live = 0;
            for (uint32_t ln : seenl) {
                size_t f = m.pf_log.size(), la = 0;
                for (size_t k = 0; k < m.pf_log.size(); ++k) if (m.pf_log[k] == ln) { f = std::min(f, k); la = k; }
                live += f <= i && i <= la;
            }
            overlap = std::max(overlap, live);
        }
        if (fifo) check(all && overlap <= 2 && seenl.size() == 3 && m.bad == 0,
                        "prefill FIFO on: " + std::to_string(m.pf_log.size()) + " pipe pieces of " + std::to_string(seenl.size()) +
                        " prompts, at most " + std::to_string(overlap) + " prompts' pieces at once (window 2); replies == references");
        else check(all && m.bad == 0, "prefill FIFO off (B14): replies == references (" + std::to_string(switches) + " switches, " +
                                      std::to_string(overlap) + " prompts at once)");
        s.shutdown();
    }
}

// P4 B15: the handover budget -- consecutive slow turns (a slow prefix step) with a decoding lane parked: past the budget a
// turn is released normally (the decoder steps) instead of handed over; handover_ms 0 = always hand over (B8-B14).
void test_handover_budget() {
    for (uint32_t budget : {5u, 0u}) {
        FakeModel m({400, 400, 400, 400, 400, 400, 400});
        m.prep_sleep_ms = 15;
        ie::LanesServe::Options o;
        o.handover_ms = budget;
        ie::LanesServe s(m, o);
        Req d(prompt_of(10, 9200), 300, 32);
        std::thread td([&] { d.run(s); });
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        std::vector<Req> rs;
        rs.reserve(6);
        for (int k = 0; k < 6; ++k) rs.emplace_back(prompt_of(12, 12000 + 50 * k), 5, 60 + k);
        std::vector<std::thread> th;
        for (auto& r : rs) th.emplace_back([&r, &s] { r.run(s); });
        for (auto& t : th) t.join();
        td.join();
        bool all = d.res.text == text_of(ref_ids(d.ids, 300, 32, 0));
        if (!all) std::printf("  decoder: %s, %u ids\n", d.res.finish_reason.c_str(), d.res.completion_tokens);
        for (auto& r : rs) {
            const bool ok = r.res.text == text_of(ref_ids(r.ids, 5, r.rq.rng, 0));
            if (!ok) std::printf("  request: %s, %u ids\n", r.res.finish_reason.c_str(), r.res.completion_tokens);
            all = all && ok;
        }
        const std::string br = status_field(s, "budget_releases");
        if (budget) check(all && br != "0" && !br.empty(), "handover budget 5 ms: " + br + " turn(s) released to the parked decoder; replies == references");
        else        check(all && br == "0", "handover budget 0: always handed over (budget_releases " + br + "); replies == references");
        s.shutdown();
    }
}
// P4 B33: LanesServe::abort_all (the server's stop) -- every request ends with "abort" at once, nothing new starts on a stage,
// the pipe's waiting pieces are dropped, a running piece ends at its next layer boundary and shutdown() waits only for it.
void test_abort_all() {
    using Clock = std::chrono::steady_clock;
    auto ms_since = [](Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); };
    auto pieces = [](FakeModel& m) { std::lock_guard<std::mutex> g(m.mu); return m.pf_log.size(); };
    // (a) mid-prefill through the pipe: a decoding lane, two new 60-token prompts in 5-row pieces of 20 ms a stage (24 pieces,
    // ~0.5 s each prompt) beside it; the stop 120 ms in
    {
        FakeModel m({4000, 4000, 4000});
        ie::LanesServe s(m, {});
        Req d(prompt_of(10, 9300), 3000, 70);
        std::thread td([&] { d.run(s); });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        m.pf_ms = 20;
        Req p1(prompt_of(60, 9400), 20, 71), p2(prompt_of(60, 9500), 20, 72);
        std::thread t1([&] { p1.run(s); }), t2([&] { p2.run(s); });
        std::this_thread::sleep_for(std::chrono::milliseconds(120));
        const size_t before = pieces(m);
        const auto t0 = Clock::now();
        const uint32_t held = s.abort_all();
        const double abort_ms = ms_since(t0);
        td.join(); t1.join(); t2.join();
        const double join_ms = ms_since(t0);
        const uint32_t preps = m.preps.load();
        Req late(prompt_of(12, 9800), 5, 75);
        late.run(s);
        s.shutdown();
        const double down_ms = ms_since(t0);
        const size_t after = pieces(m);
        check(abort_ms < 50, "stop mid-prefill: abort_all does not block (" + std::to_string(int(abort_ms)) + " ms)");
        check(d.res.finish_reason == "abort" && p1.res.finish_reason == "abort" && p2.res.finish_reason == "abort",
              "stop mid-prefill: the decoding request and both prefilling ones end with abort (" + d.res.finish_reason + ", " +
                  p1.res.finish_reason + ", " + p2.res.finish_reason + ")");
        const std::string dref = text_of(ref_ids(d.ids, 3000, 70, 0));
        check(!d.res.text.empty() && dref.compare(0, d.res.text.size(), d.res.text) == 0,
              "stop mid-prefill: the decoding request's ids before the stop == its reference's (" + std::to_string(d.res.completion_tokens) + " ids)");
        check(join_ms < 200, "stop mid-prefill: every request returned at once, not after the pieces (" + std::to_string(int(join_ms)) + " ms)");
        check(held <= 2 && m.calls_after_abort.load() <= held,
              "stop mid-prefill: no stage started a step after the stop beyond the " + std::to_string(held) + " a stage held (" +
                  std::to_string(m.calls_after_abort.load()) + " stage calls after)");
        check(before < 24 && after <= before + held + 1 && after < 24,
              "stop mid-prefill: the waiting pieces were dropped (" + std::to_string(before) + " pieces before the stop, " +
                  std::to_string(after) + " in all, of 24)");
        check(down_ms < 400, "stop mid-prefill: shutdown waited only for the held step's next layer boundary (" + std::to_string(int(down_ms)) + " ms)");
        check(late.res.finish_reason == "abort" && m.preps.load() == preps, "a request after the stop is refused (abort) before any device work");
        check(m.bad == 0, "stop mid-prefill: no stage saw a wrong position");
    }
    // (b) mid-serial-turn: a lone 80-token prompt prefills in the serial turn (16 chunks x 2 stages x 20 ms); the stop 100 ms in
    // ends its chunk at a layer boundary and no next chunk starts
    {
        FakeModel m({400});
        m.pf_ms = 20;
        ie::LanesServe s(m, {});
        Req a(prompt_of(80, 9600), 20, 73);
        std::thread ta([&] { a.run(s); });
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const auto t0 = Clock::now();
        const uint32_t held = s.abort_all();
        ta.join();
        const double join_ms = ms_since(t0);
        s.shutdown();
        check(a.res.finish_reason == "abort" && held == 1 && m.serial_chunks.load() < 16 && join_ms < 100,
              "stop in the serial turn: abort (" + a.res.finish_reason + "), the turn's work counted (" + std::to_string(held) + "), " +
                  std::to_string(m.serial_chunks.load()) + " of 16 chunks, returned in " + std::to_string(int(join_ms)) + " ms");
    }
    // (c) a request waiting for the turn while the holder's (uninterruptible) prefix step runs: refused at once, never prepared
    {
        FakeModel m({400, 400});
        m.prep_sleep_ms = 300;
        ie::LanesServe s(m, {});
        Req a(prompt_of(20, 9900), 20, 76), b(prompt_of(20, 9950), 20, 77);
        std::thread ta([&] { a.run(s); });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::thread tb([&] { b.run(s); });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const auto t0 = Clock::now();
        s.abort_all();
        tb.join();
        const double b_ms = ms_since(t0);
        ta.join();
        s.shutdown();
        check(b.res.finish_reason == "abort" && b_ms < 100 && m.preps.load() == 1,
              "stop: a request waiting for the turn is refused at once (" + std::to_string(int(b_ms)) + " ms), never prepared");
        check(a.res.finish_reason == "abort", "stop: the turn holder ends with abort after its prefix step (" + a.res.finish_reason + ")");
    }
    // (d) nothing in flight: 0 steps; the server refuses later requests
    {
        FakeModel m({400});
        ie::LanesServe s(m, {});
        Req a(prompt_of(15, 9990), 5, 78);
        a.run(s);
        const uint32_t held = s.abort_all();
        check(held == 0 && s.abort_all() == 0, "stop with nothing in flight: 0 steps; abort_all again: 0");
        Req b(prompt_of(15, 9991), 5, 79);
        b.run(s);
        s.shutdown();
        check(a.res.finish_reason == "length" && b.res.finish_reason == "abort", "stop: a finished request is untouched, a later one refused");
    }
}

// P4 B34: the lane pipeline -- a prompt prefilling through the pipe beside a decoding lane. On: its pieces overlap across the
// two stages (piece k + 1 on stage 0 while piece k is on stage 1); off (B10-B33): they never do. Replies == references either
// way, every stage sees every lane's pieces in order (the fake refuses a step off its stage's depth), /health counts them.
void test_lane_pipeline() {
    for (const bool on : {true, false}) {
        const std::string tag = on ? "lane pipeline ON" : "lane pipeline OFF";
        FakeModel m({4000, 4000}, on);
        m.pf_ms = 3;
        ie::LanesServe s(m, {});
        Req d(prompt_of(10, 9700), 400, 80);
        std::thread td([&] { d.run(s); });
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        m.record = true;
        Req p(prompt_of(80, 9750), 10, 81);   // 16 pieces of 5 rows through the pipe (another lane decodes)
        p.run(s);
        m.record = false;
        td.join();
        uint32_t pieces = 0, overlaps = 0;
        bool order = true;
        {
            std::lock_guard<std::mutex> g(m.mu);
            uint32_t last[2] = {0, 0};
            bool seen[2] = {false, false};
            for (const auto& [st, ln, p0, t0, t1] : m.tl) {
                if (ln != p.res.lane) continue;
                if (st == 0) ++pieces;
                if (seen[st] && p0 <= last[st]) order = false;
                seen[st] = true; last[st] = p0;
                if (st != 0) continue;
                for (const auto& [st2, ln2, q0, u0, u1] : m.tl)   // the piece before, on stage 1, at the same time
                    if (st2 == 1 && ln2 == ln && q0 + m.chunk == p0 && t0 < u1 && u0 < t1) ++overlaps;
            }
        }
        const std::string la = status_field(s, "lookaheads");
        const bool ok = p.res.text == text_of(ref_ids(p.ids, 10, 81, 0)) && d.res.text == text_of(ref_ids(d.ids, 400, 80, 0));
        check(ok && m.bad == 0, tag + ": the prompt and the decoding lane == their references");
        check(pieces == 16 && order, tag + ": the prompt's 16 pieces ran each stage in order");
        if (on) check(overlaps > 0 && la != "0" && !la.empty(), tag + ": " + std::to_string(overlaps) + " of its pieces ran stage 0 while the piece "
                                                            "before ran stage 1 (lookaheads " + la + ")");
        else check(overlaps == 0 && la == "0", tag + ": a lane's pieces never overlapped (lookaheads " + la + ")");
        s.shutdown();
    }
}

// P4 B34 + B33: abort_all while a lane has two pieces in flight -- its piece at 50 held on stage 1 and its lookahead at 55 held
// on stage 0, beside a decoding lane: every request ends "abort" at once, the pipe drops what waits, nothing new starts on a
// stage, the two held pieces end at their next layer boundary and shutdown waits only for them.
void test_abort_pipeline() {
    using Clock = std::chrono::steady_clock;
    auto ms_since = [](Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); };
    FakeModel m({4000, 4000}, true);
    ie::LanesServe s(m, {});
    Req d(prompt_of(10, 9800), 3000, 82);
    std::thread td([&] { d.run(s); });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    m.hold_lane = 1; m.hold_pos = 50;
    Req p(prompt_of(120, 9850), 20, 83);   // 24 pieces; lane 1 (the decoder holds lane 0)
    std::thread tp([&] { p.run(s); });
    for (int i = 0; i < 3000 && !(m.held_at[0].load() && m.held_at[1].load()); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    check(m.held_at[0].load() && m.held_at[1].load(), "abort mid-pipeline: lane 1's piece at 50 holds stage 1, its lookahead at 55 stage 0");
    size_t before = 0;
    { std::lock_guard<std::mutex> g(m.mu); before = m.pf_log.size(); }
    const auto t0 = Clock::now();
    const uint32_t held = s.abort_all();
    const double abort_ms = ms_since(t0);
    td.join(); tp.join();
    const double join_ms = ms_since(t0);
    s.shutdown();
    const double down_ms = ms_since(t0);
    size_t after = 0;
    { std::lock_guard<std::mutex> g(m.mu); after = m.pf_log.size(); }
    check(abort_ms < 50 && join_ms < 200, "abort mid-pipeline: abort_all does not block (" + std::to_string(int(abort_ms)) + " ms), every request returned (" +
                                              std::to_string(int(join_ms)) + " ms)");
    check(d.res.finish_reason == "abort" && p.res.finish_reason == "abort", "abort mid-pipeline: both requests end with abort (" +
                                                                                 d.res.finish_reason + ", " + p.res.finish_reason + ")");
    const std::string dref = text_of(ref_ids(d.ids, 3000, 82, 0));
    check(!d.res.text.empty() && dref.compare(0, d.res.text.size(), d.res.text) == 0, "abort mid-pipeline: the decoder's ids so far == its reference's");
    check(held == 2 && m.calls_after_abort.load() == 0, "abort mid-pipeline: the stages held the lane's two pieces (" + std::to_string(held) +
                                                            "), no stage started anything after the stop (" + std::to_string(m.calls_after_abort.load()) + ")");
    check(after == before && after < 24, "abort mid-pipeline: no piece started after the stop (" + std::to_string(after) + " of 24 ran stage 0)");
    check(down_ms < 400, "abort mid-pipeline: shutdown waited only for the two held pieces' layer boundary (" + std::to_string(int(down_ms)) + " ms)");
    check(m.bad == 0, "abort mid-pipeline: no stage saw a wrong position");
}

}  // namespace

int main() {
    test_q4e_rules();
    test_choose();
    test_glm_pipe_additions();
    test_card_pipe_sync();
    test_card_pipe_rows();
    test_serve();
    test_failed_lane_reuse();
    test_shared_prefix();
    test_prefill_fifo();
    test_handover_budget();
    test_abort_all();
    test_lane_pipeline();
    test_abort_pipeline();
    std::printf("%s\n", g_fail ? "LANES SERVE TEST: FAIL" : "LANES SERVE TEST: PASS");
    return g_fail ? 1 : 0;
}
