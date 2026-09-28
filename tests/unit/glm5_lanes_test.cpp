// tests/unit/glm5_lanes_test.cpp -- host-only (no GPU, no SYCL): GLM-5.3 request lanes (P4 B7, include/ie/glm5_lanes.hpp,
// docs/glm53/P4_B7_LANES.md). The card pipe's scheduling with fake stages -- every step through every stage in order with
// its own lane's residual (a cross-lane mix-up fails the pattern check), stages overlapping on different lanes, the claim
// rules (a callback resubmits once; another thread cannot submit an in-flight lane; a refused submit gives the claim back),
// pos0 validation, the error latch, a throwing callback, stop() from a callback refusing instead of deadlocking -- and the
// per-lane VRAM arithmetic (glm5_lane_bytes) against the design's 0.24 / 0.26 GiB at 32K. Does NOT link ie_core.
#undef NDEBUG
#include "ie/glm5_lanes.hpp"
#include "ie/glm5_server.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <map>
#include <mutex>
#include <set>
#include <span>
#include <stdexcept>
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

// A lane runs `steps` steps: a 3-row "prefill" at 0, then 1-row "decode" steps. Stage 0 writes a lane/position pattern into
// the step's residual, every later stage checks it and rewrites it; the last stage records the step.
struct Rig {
    uint32_t n_stages = 2, n_lanes = 3, steps = 12, wide_row = 4;
    std::atomic<int> active{0}, max_active{0};
    std::atomic<uint32_t> bad_pattern{0};
    std::mutex mu;
    std::map<uint32_t, std::vector<uint32_t>> pos_of;   // lane -> committed pos0 of each finished step, in order
    std::vector<uint32_t> done_n;
    std::string cb_err;
    ie::Glm5LanePipe* pipe = nullptr;

    std::string stage(uint32_t s, const ie::Glm5LanePipe::Step& st) {
        const int a = ++active;
        for (int m = max_active.load(); a > m && !max_active.compare_exchange_weak(m, a);) {}
        const float tag = float(st.lane * 100000 + st.pos0 * 10);
        for (uint32_t r = 0; r < st.T; ++r)
            for (uint32_t k = 0; k < wide_row; ++k) {
                float& w = st.wide[size_t(r) * wide_row + k];
                if (s > 0 && w != tag + float(s - 1)) ++bad_pattern;
                w = tag + float(s);
            }
        std::this_thread::sleep_for(std::chrono::milliseconds(2 + (st.lane + s) % 3));
        --active;
        return {};
    }
    void done(uint32_t lane, uint32_t T, uint32_t pos0) {
        std::lock_guard<std::mutex> lk(mu);
        pos_of[lane].push_back(pos0);
        if (++done_n[lane] >= steps) return;
        const int32_t id = int32_t(lane);
        if (auto e = pipe->submit(lane, &id, 1, pos0 + T); !e.empty() && cb_err.empty()) cb_err = e;
    }
};

}  // namespace

int main() {
    // ---- the pipe: 3 lanes over 2 stages, each lane driving itself from the callback ----------------------------------
    {
        Rig rig;
        ie::Glm5LanePipe pipe(rig.n_lanes, 8, rig.wide_row);
        rig.pipe = &pipe;
        rig.done_n.assign(rig.n_lanes, 0);
        check(pipe.host_bytes() == uint64_t(rig.n_lanes) * (8 * 4 + 8 * rig.wide_row * 4), "host bytes = lanes x (ids + residual)");
        check(!pipe.submit(0, nullptr, 1, 0).empty(), "submit before start is refused");
        auto e = pipe.start(rig.n_stages, [&](uint32_t s, const ie::Glm5LanePipe::Step& st) { return rig.stage(s, st); },
                            [&](uint32_t l, uint32_t T, uint32_t p) { rig.done(l, T, p); });
        check(e.empty(), "start");
        {
            std::lock_guard<std::mutex> lk(rig.mu);   // seed every lane with its 3-row step at 0
            const int32_t ids[3] = {1, 2, 3};
            for (uint32_t l = 0; l < rig.n_lanes; ++l) check(pipe.submit(l, ids, 3, 0).empty(), "seed lane " + std::to_string(l));
            check(!pipe.submit(1, ids, 1, 0).empty(), "a second submit of an in-flight lane from another thread is refused");
        }
        e = pipe.stop();
        check(e.empty() && rig.cb_err.empty(), "stop after every lane ran its steps: no error" + (e + rig.cb_err));
        bool order_ok = true;
        for (uint32_t l = 0; l < rig.n_lanes; ++l) {
            const auto& p = rig.pos_of[l];
            order_ok = order_ok && p.size() == rig.steps && p[0] == 0;
            for (size_t k = 1; k < p.size(); ++k) order_ok = order_ok && p[k] == 3 + uint32_t(k - 1);
        }
        check(order_ok, "every lane's steps finished in order at contiguous positions (0, 3, 4, ...)");
        check(pipe.steps_done() == uint64_t(rig.n_lanes) * rig.steps, "steps_done counts every step of every lane");
        check(rig.bad_pattern == 0, "every stage read its own lane's residual from the stage before (no cross-lane mix-up)");
        check(rig.max_active.load() >= 2, "the stages overlapped on different lanes (max " + std::to_string(rig.max_active.load()) + " at once)");
        for (uint32_t l = 0; l < rig.n_lanes; ++l)
            check(pipe.lane_pos(l) == 3 + rig.steps - 1, "lane " + std::to_string(l) + " committed position " + std::to_string(pipe.lane_pos(l)));
        check(!pipe.submit(0, nullptr, 1, 0).empty(), "submit after stop is refused");
    }
    // ---- claims, validation, the resubmit rule, stop from a callback ------------------------------------------------------
    {
        ie::Glm5LanePipe pipe(2, 4, 1);
        std::mutex mu;
        std::vector<std::string> seen;
        std::atomic<int> phase{0};
        auto stage = [&](uint32_t, const ie::Glm5LanePipe::Step&) -> std::string { return {}; };
        auto done = [&](uint32_t lane, uint32_t T, uint32_t pos0) {
            std::lock_guard<std::mutex> lk(mu);
            const int32_t id = 7;
            if (lane == 0 && pos0 == 0) {
                seen.push_back("resub1:" + pipe.submit(0, &id, 1, pos0 + T));        // allowed once
                seen.push_back("resub2:" + pipe.submit(0, &id, 1, pos0 + T));        // refused: already resubmitted
                seen.push_back("stop:" + pipe.stop());                               // refused, no deadlock
            }
        };
        check(pipe.start(2, stage, done).empty(), "start (claims)");
        const int32_t ids[4] = {1, 2, 3, 4};
        check(!pipe.submit(1, ids, 5, 0).empty(), "T above max_T is refused");
        check(!pipe.submit(1, ids, 0, 0).empty(), "T = 0 is refused");
        check(!pipe.submit(1, ids, 1, 9).empty(), "pos0 that is neither 0 nor the lane's position is refused");
        check(pipe.submit(1, ids, 2, 0).empty(), "the refused submits gave the claim back (lane 1 submits)");
        check(!pipe.submit(5, ids, 1, 0).empty(), "an out-of-range lane is refused");
        {
            std::lock_guard<std::mutex> lk(mu);   // hold the callback until lane 0 is submitted from here
            check(pipe.submit(0, ids, 4, 0).empty(), "lane 0 submits");
        }
        const std::string e = pipe.stop();
        check(e.empty(), "stop: " + e);
        check(seen.size() == 3 && seen[0] == "resub1:", "the callback resubmits its lane once: " + (seen.empty() ? "" : seen[0]));
        check(seen.size() == 3 && seen[1].find("already resubmitted") != std::string::npos, "a second resubmit from the callback is refused");
        check(seen.size() == 3 && seen[2].find("deadlock") != std::string::npos, "stop() from a callback refuses instead of deadlocking");
        check(pipe.lane_pos(0) == 5 && pipe.lane_pos(1) == 2, "committed positions (lane 0: 4 + 1, lane 1: 2)");
        (void)phase;
    }
    // ---- a stage error latches; the other lanes drain; stop returns it ----------------------------------------------------
    {
        ie::Glm5LanePipe pipe(2, 2, 1);
        std::mutex mu;
        std::vector<uint32_t> n(2, 0);
        std::string refused;
        auto stage = [&](uint32_t s, const ie::Glm5LanePipe::Step& st) -> std::string {
            if (s == 1 && st.lane == 1 && st.pos0 == 3) return "boom";
            if (st.lane == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));   // lane 0 outlives the error
            return {};
        };
        auto done = [&](uint32_t lane, uint32_t T, uint32_t pos0) {
            std::lock_guard<std::mutex> lk(mu);
            if (++n[lane] >= 20) return;
            const int32_t id = 1;
            if (auto e = pipe.submit(lane, &id, 1, pos0 + T); !e.empty() && refused.empty()) refused = e;
        };
        check(pipe.start(2, stage, done).empty(), "start (error latch)");
        {
            std::lock_guard<std::mutex> lk(mu);
            const int32_t ids[2] = {1, 2};
            pipe.submit(0, ids, 2, 0);
            pipe.submit(1, ids, 2, 0);
        }
        const std::string e = pipe.stop();
        check(e.find("boom") != std::string::npos && e.find("lane 1 at 3") != std::string::npos && e.find("stage 1") != std::string::npos,
              "stop returns the first stage error with its lane, position and stage: " + e);
        check(refused.find("a stage failed") != std::string::npos, "submits after the error are refused: " + refused);
        check(n[1] == 2, "the failed lane's step never reached its callback (lane 1: " + std::to_string(n[1]) + " steps)");
        check(pipe.lane_pos(1) == 3, "the failed step's position is not committed");
        // after a restart the failed lane takes only a new sequence; the other lane continues where it stopped
        n.assign(2, 100);   // the callback resubmits nothing now
        check(pipe.start(2, [](uint32_t, const ie::Glm5LanePipe::Step&) { return std::string(); }, done).empty(), "restart after the error");
        const int32_t id = 1;
        const std::string cont = pipe.submit(1, &id, 1, 3);
        check(cont.find("failed part-way") != std::string::npos, "the failed lane refuses a continuation at its old position: " + cont);
        check(pipe.submit(1, &id, 1, 0).empty(), "the failed lane takes a new sequence (pos0 0)");
        check(pipe.submit(0, &id, 1, pipe.lane_pos(0)).empty(), "the other lane continues at its committed position");
        check(pipe.stop().empty(), "stop after the restart");
        check(pipe.lane_pos(1) == 1, "the new sequence on the failed lane committed position 1");
        check(pipe.start(1, [](uint32_t, const ie::Glm5LanePipe::Step&) { return std::string(); }, nullptr).empty() &&
              pipe.submit(1, &id, 1, 1).empty() && pipe.stop().empty(), "once its new sequence has run, the lane continues normally");
    }
    // ---- a throwing callback latches an error and releases the lane ------------------------------------------------------
    {
        ie::Glm5LanePipe pipe(1, 1, 1);
        check(pipe.start(1, [](uint32_t, const ie::Glm5LanePipe::Step&) { return std::string(); },
                         [](uint32_t, uint32_t, uint32_t) { throw std::runtime_error("cb"); }).empty(), "start (throwing callback)");
        const int32_t id = 1;
        check(pipe.submit(0, &id, 1, 0).empty(), "submit (throwing callback)");
        const std::string e = pipe.stop();
        check(e.find("done callback threw: cb") != std::string::npos, "a throwing callback's error is returned by stop: " + e);
        check(pipe.start(1, [](uint32_t, const ie::Glm5LanePipe::Step&) { return std::string(); }, nullptr).empty() &&
              pipe.submit(0, &id, 1, 1).empty() && pipe.stop().empty() && pipe.lane_pos(0) == 2,
              "the pipe restarts after stop and the lane continues at its committed position");
    }
    // ---- P4 B14 rows mode: groups of 1-row steps, prefill pieces alone, the AUTO / explicit cap ---------------------------
    for (const auto& [n_lanes, cap, want_max] : std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>{{4, 0, 2}, {8, 0, 4}, {8, 3, 3}}) {
        const std::string tag_s = std::to_string(n_lanes) + " lanes, cap " + (cap ? std::to_string(cap) : std::string("AUTO"));
        const uint32_t steps = 40, wide_row = 4;
        ie::Glm5LanePipe pipe(n_lanes, 8, wide_row);
        std::mutex mu;
        std::map<uint32_t, std::vector<uint32_t>> pos_of;
        std::vector<uint32_t> done_n(n_lanes, 0);
        std::atomic<uint32_t> bad{0}, rows_T_bad{0}, plain_multi{0}, plain_decode{0};
        std::string cb_err;
        auto resub = [&](uint32_t lane, uint32_t pos0, uint32_t T) {   // (mu held)
            pos_of[lane].push_back(pos0);
            if (++done_n[lane] >= steps) return;
            const int32_t id = int32_t(lane);
            if (auto e = pipe.submit(lane, &id, 1, pos0 + T); !e.empty() && cb_err.empty()) cb_err = e;
        };
        auto stage = [&](uint32_t s, const ie::Glm5LanePipe::Step& st) -> std::string {
            if (st.T > 1) ++plain_multi; else ++plain_decode;
            const float tag = float(st.lane * 100000 + st.pos0 * 10);
            for (uint32_t r = 0; r < st.T; ++r)
                for (uint32_t k = 0; k < wide_row; ++k) {
                    float& w = st.wide[size_t(r) * wide_row + k];
                    if (s > 0 && w != tag + float(s - 1)) ++bad;
                    w = tag + float(s);
                }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            return {};
        };
        auto rows_stage = [&](uint32_t s, std::span<const ie::Glm5LanePipe::Step> st, float* gw) -> std::string {
            for (size_t i = 0; i < st.size(); ++i) {
                if (st[i].T != 1 || st[i].wide || !st[i].ids || st[i].ids[0] != int32_t(st[i].lane)) ++rows_T_bad;
                const float tag = float(st[i].lane * 100000 + st[i].pos0 * 10);
                for (uint32_t k = 0; k < wide_row; ++k) {
                    float& w = gw[i * wide_row + k];
                    if (s > 0 && w != tag + float(s - 1)) ++bad;
                    w = tag + float(s);
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            return {};
        };
        std::vector<uint32_t> last_pos(n_lanes, 0);
        auto done = [&](uint32_t lane, uint32_t T, uint32_t pos0) { std::lock_guard<std::mutex> lk(mu); resub(lane, pos0, T); };
        auto rows_done = [&](std::span<const uint32_t> lanes) {
            std::lock_guard<std::mutex> lk(mu);
            for (uint32_t l : lanes) resub(l, pipe.lane_pos(l) - 1, 1);
        };
        check(pipe.start_rows(2, stage, rows_stage, done, rows_done, 16, cap).empty() && pipe.rows_mode(), "start_rows (" + tag_s + ")");
        {
            std::lock_guard<std::mutex> lk(mu);
            const int32_t ids[3] = {1, 2, 3};
            for (uint32_t l = 0; l < n_lanes; ++l) pipe.submit(l, ids, 3, 0);
        }
        const std::string e = pipe.stop();
        check(e.empty() && cb_err.empty(), "rows: every lane ran its steps without an error (" + tag_s + ")" + e + cb_err);
        bool order_ok = true;
        for (uint32_t l = 0; l < n_lanes; ++l) {
            const auto& p = pos_of[l];
            order_ok = order_ok && p.size() == steps && p[0] == 0;
            for (size_t k = 1; k < p.size(); ++k) order_ok = order_ok && p[k] == 3 + uint32_t(k - 1);
        }
        check(order_ok, "rows: every lane's steps finished in order at contiguous positions (" + tag_s + ")");
        check(bad == 0, "rows: every stage read its rows' residual from the stage before (" + tag_s + ")");
        check(rows_T_bad == 0, "rows: a group step is 1 row a lane, its ids, the rows in the group buffer (" + tag_s + ")");
        check(plain_multi == 2 * n_lanes, "rows: every 3-row prefill piece ran alone, through the plain stage (" + tag_s + ")");
        const auto gs = pipe.group_sizes();
        uint64_t rows_total = 0, max_g = 0, multi = 0;
        for (uint32_t g = 1; g < gs.size(); ++g) { rows_total += gs[g] * g; if (gs[g]) max_g = g; if (g >= 2) multi += gs[g]; }
        check(rows_total == uint64_t(n_lanes) * steps && pipe.steps_done() == rows_total,
              "rows: the groups carried every step once (" + std::to_string(rows_total) + ")");
        check(max_g == want_max && multi > 0, "rows: the largest group is the cap " + std::to_string(want_max) + " (seen " +
                                                  std::to_string(max_g) + ", " + std::to_string(multi) + " groups of >= 2) (" + tag_s + ")");
        check(gs[want_max] * want_max * 2 >= rows_total - 2 * n_lanes,
              "rows: the regrouping keeps most decode rows in full groups (" + std::to_string(gs[want_max]) + " of size " +
                  std::to_string(want_max) + ") (" + tag_s + ")");
        check(!pipe.rows_mode(), "rows: stop leaves rows mode");
    }
    // ---- rows mode: a group's stage error fails every lane of the group; a throwing rows callback latches ----------------
    {
        ie::Glm5LanePipe pipe(3, 2, 1);
        std::mutex mu;
        std::vector<uint32_t> n(3, 0);
        std::set<uint32_t> failed_group;
        auto stage = [&](uint32_t, const ie::Glm5LanePipe::Step&) -> std::string { return {}; };
        auto rows_stage = [&](uint32_t s, std::span<const ie::Glm5LanePipe::Step> st, float*) -> std::string {
            if (s == 1 && st[0].pos0 >= 6) {
                std::lock_guard<std::mutex> lk(mu);
                if (failed_group.empty()) { for (const auto& x : st) failed_group.insert(x.lane); return "boom"; }
            }
            return {};
        };
        auto again = [&](uint32_t lane) {   // (mu held)
            if (++n[lane] >= 30) return;
            const int32_t id = 1;
            (void)pipe.submit(lane, &id, 1, pipe.lane_pos(lane));
        };
        auto done = [&](uint32_t lane, uint32_t, uint32_t) { std::lock_guard<std::mutex> lk(mu); again(lane); };
        auto rows_done = [&](std::span<const uint32_t> ls) { std::lock_guard<std::mutex> lk(mu); for (uint32_t l : ls) again(l); };
        check(pipe.start_rows(2, stage, rows_stage, done, rows_done, 16, 3).empty(), "start_rows (group error)");
        {
            std::lock_guard<std::mutex> lk(mu);
            const int32_t id = 1;
            for (uint32_t l = 0; l < 3; ++l) pipe.submit(l, &id, 1, 0);
        }
        const std::string e = pipe.stop();
        check(e.find("boom") != std::string::npos && e.find("a group of") != std::string::npos && e.find("stage 1") != std::string::npos,
              "rows: stop returns the group's stage error naming its lanes: " + e);
        check(failed_group.size() >= 2, "rows: the failing step was a group (" + std::to_string(failed_group.size()) + " lanes)");
        bool all_failed = !failed_group.empty();
        const int32_t id = 1;
        check(pipe.start(1, stage, nullptr).empty(), "restart after the group error");
        for (uint32_t l : failed_group) {
            const std::string r = pipe.submit(l, &id, 1, pipe.lane_pos(l));
            all_failed = all_failed && r.find("failed part-way") != std::string::npos;
        }
        check(all_failed, "rows: EVERY lane of the failed group takes only a new sequence");
        check(pipe.stop().empty(), "stop after the group error restart");

        ie::Glm5LanePipe p2(2, 1, 1);
        std::atomic<int> calls{0};
        check(p2.start_rows(1, stage, [](uint32_t, std::span<const ie::Glm5LanePipe::Step>, float*) { return std::string(); },
                            [&](uint32_t, uint32_t, uint32_t) {}, [&](std::span<const uint32_t>) { ++calls; throw std::runtime_error("rcb"); },
                            16, 2).empty(), "start_rows (throwing rows callback)");
        {
            // both lanes queued before stage 0 can form: submit under a paused formation is not possible, so retry until a
            // group of 2 formed (a lone lane's callback is the plain one)
            for (int k = 0; k < 50 && calls.load() == 0; ++k) {
                const int32_t i1 = 1;
                (void)p2.submit(0, &i1, 1, p2.lane_pos(0));
                (void)p2.submit(1, &i1, 1, p2.lane_pos(1));
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (!p2.error().empty()) break;
            }
        }
        const std::string e2 = p2.stop();
        check(calls.load() >= 1 && e2.find("done callback threw: rcb") != std::string::npos, "rows: a throwing rows callback's error is latched: " + e2);
        check(!p2.start_rows(1, stage, nullptr, nullptr, nullptr, 16, 0).empty(), "start_rows without a rows stage is refused");
        check(!p2.start_rows(1, stage, [](uint32_t, std::span<const ie::Glm5LanePipe::Step>, float*) { return std::string(); },
                             nullptr, nullptr, 17, 0).empty(), "start_rows with more than 16 rows a group is refused");
    }
    // ---- per-lane VRAM: GLM-5.3-Flash's layout (full layers 3, 7, ..., 43; stages [0, 23) and [23, 45)) -----------------
    {
        ie::Glm5NextConfig c;
        c.n_layers = 46; c.nextn_predict_layers = 1; c.hidden = 4096; c.n_q_heads = 64; c.kda_head_dim = 128;
        c.kv_lora_rank = 512; c.conv_kernel = 4; c.indexer_kpool = 4; c.indexer_head_dim = 128; c.indexer_top_k = 2048;
        c.attn_kind.assign(46, 0);
        for (uint32_t i = 3; i < 45; i += 4) c.attn_kind[i] = 1;
        const uint64_t kda = 64ull * 128 * 128 * 4 + 3ull * 8192 * 3 * 2;   // per KDA layer: scan fp32 + conv f16
        const uint64_t s0 = ie::glm5_lane_bytes(c, 32768, 0, 23, true), s1 = ie::glm5_lane_bytes(c, 32768, 23, 45, true);
        check(s0 == 18 * kda + 5ull * 32768 * 512 * 2 + 5ull * 8192 * 128 * 2 + 2ull * 5 * 4 * 128 * 2,
              "stage 0 (5 full + 18 KDA) one lane at 32K = " + std::to_string(s0) + " B");
        check(s1 == 16 * kda + 6ull * 32768 * 512 * 2 + 6ull * 8192 * 128 * 2 + 2ull * 6 * 4 * 128 * 2,
              "stage 1 (6 full + 16 KDA) one lane at 32K = " + std::to_string(s1) + " B");
        const double g0 = double(s0) / double(1ull << 30), g1 = double(s1) / double(1ull << 30);
        check(g0 > 0.23 && g0 < 0.25 && g1 > 0.25 && g1 < 0.27, "the design's 0.24 + 0.26 GiB per lane at 32K: " + std::to_string(g0) + " + " +
                                                                    std::to_string(g1));
        check(ie::glm5_lane_bytes(c, 2048, 0, 23, false) == 18 * kda + 5ull * 2048 * 512 * 2, "dense stage (no indexer) at ctx 2048");
    }
    std::printf("\nglm5_lanes_test: %s\n", g_fail ? "FAILURE(S)" : "PASS");
    return g_fail ? 1 : 0;
}
