// include/ie/glm5_lanes.hpp -- GLM-5.3-Flash request lanes' card pipe (P4 B7, docs/glm53/P4_B7_LANES.md; design
// ~/ds41_work/p60/p4_b6_b8_design.md section 5.2).
//
// GLM runs as TWO model objects, one per card (Glm5NextModel [0, 23) and [23, 45)); a request lane is one sequence's state on
// each of them (Glm5NextModel::select_lane). Stepped serially, each card idles while the other runs its half of a step. The
// pipe gives every stage its own host thread: while card 0 runs lane B's step, card 1 runs the rest of lane A's. A lane has at
// most one step in flight, and its step runs exactly the launches a serial step runs (Glm5NextModel::lane_stage), so with the
// CPU expert leg off (IE_G5_CPU_MISS=0) a lane's logits are its serial ones, bit for bit.
//
// No SYCL here: a stage is the caller's function, so the scheduling -- claims, hand-offs, the error latch, stop -- is
// unit-tested on the CPU (tests/unit/glm5_lanes_test.cpp).
//
// Rules (MiMo B2/B3's gate notes 6-9, docs/mimo26/P4_B3_ROWS.md):
//  * submit(lane, ...) claims the lane, check-and-set under the lock: an idle lane from any thread; a lane inside its done
//    callback only from that callback's thread, and once. A refused submit gives a fresh claim back.
//  * The claim is held THROUGH the done callback: no other thread's step rewrites the lane's results while the callback
//    reads them. A resubmit from the callback may start on stage 0 before the callback returns (stage 0 writes only the
//    lane's ids/residual buffers, which the callback does not read); the next results cannot land before it returns.
//  * pos0 is the lane's committed position (the end of its last finished step) or 0: a new sequence, which every stage
//    resets before its forward.
//  * After a stage error the failed step is dropped (its lane released) and every submit is refused with that error until
//    stop(), which returns it. Steps already past the failed stage finish normally. A step that failed part-way may have
//    advanced some of its lane's state (the KDA recurrence, the latents) and not the rest, so that lane then takes only a
//    new sequence (pos0 0, which every stage resets), also after a restart.
//  * stop() waits until no step is in flight (a callback that keeps resubmitting keeps it waiting), then joins the stage
//    threads. Called from a stage thread (a done callback) it refuses instead of deadlocking.
//  * P4 B33: cancel() is for a process stop: every step waiting for a stage is dropped at once (its lane released; one that
//    a stage already ran is half-stepped and marked failed, as a stage error marks it), a step a stage finishes after it goes
//    to no other stage and runs no done callback, and every submit is refused until the next start(). It never blocks; it
//    returns the steps a stage was running then, the only ones a stop() after it still waits for.
//  * The per-lane host buffers (the step's ids and the residual crossing the stages) are allocated AND touched by the
//    constructor: build the pipe before loading the model and the pinning floor's MemAvailable check counts them.
//  * P4 B34 lookahead (a pipe built with `ahead`): submit_ahead queues a lane's NEXT step while its one step in flight is
//    past stage 0 (on a later stage, queued for it, or between) and not in its callback, at pos0 = that step's end, in the
//    lane's second buffer set (allocated by the constructor too) -- and only while stage 0 is idle (nothing queued, nothing
//    running), so a lookahead never delays another step there. The stage queues are FIFO, so every stage runs a lane's
//    steps in submission order: its launches per card are those of the steps one after another. A lane then holds two
//    claims; the older step's callback may not resubmit (the next step is in flight already), and when it returns without
//    one the lookahead is the lane's step in flight. A lookahead is never forced: anything else answers "not taken".
//    `idle` (start / start_rows) runs unlocked whenever stage 0 is idle and a lookahead may have become possible: on stage
//    0's thread after it finished a step, and on the last stage's thread after a step there left its lane with one step in
//    flight past stage 0. Without `ahead` nothing of this exists.
//
// Rows mode (P4 B14 phase 1b, start_rows; MiMo B3's group protocol, docs/mimo26/P4_B3_ROWS.md): stage 0 gathers the queued
// lanes into a GROUP that crosses the stages as a unit, so one card step serves several lanes' decode rows.
//  * Stage 0 forms a group from its FIFO: a front lane with T > 1 (a prefill piece) or at pos0 0 (a new sequence) runs alone;
//    otherwise it takes 1-row lanes at pos0 > 0 while the group has fewer than min(cap, max_group) of them and stops at the
//    first lane that must run alone (it stays queued).
//    cap = group_cap, or AUTO (0) = ceil(lanes in flight / stages): with every lane decoding, one group per card.
//  * A group of ONE lane runs the plain StageFn with the lane's own buffers and, at the last stage, the plain DoneFn:
//    exactly the launches start()'s per-lane pipe runs. A group of G >= 2 runs RowsStageFn(stage, steps, group_wide) -- the
//    steps in group order, every one 1 row, `wide` null: the rows cross the stages in the group's own [max_group x
//    wide_row] buffer -- and, at the last stage, RowsDoneFn(lanes) once.
//  * The last stage commits every lane's position and marks every lane in its callback (each may resubmit itself once,
//    from that thread), and holds stage 0 back (it forms no group) until the callback returned: the lanes it resubmits form
//    the next group together.
//  * A stage error fails EVERY lane of the group (each is marked failed, as a lone lane's is) and latches the error.
//  * start() and its stage loop are untouched by rows mode (the other archs' scheduling is start()'s).
//  * P4 B42 regroup (set_regroup(wait_us); off by default): stage 0 merges only the decode lanes it finds QUEUED when it forms a
//    group, and a landed group's lanes resubmit together (pgate_), so k >= 3 decode groups rotating over the stages never meet
//    (one lands per step, one forms per step: the same membership forms again, forever) -- the paused serial turns' release
//    bursts used to merge them. With the regroup, before forming a group whose front is a decode lane, with fewer than `lim`
//    decode lanes queued, while a DECODE group runs at the last stage (its landing is one step away) and the decode groups in
//    flight are >= the stages (forming now would make >= 3 rotate), stage 0 waits -- at most wait_us -- for that landing's
//    callback (pgate_) or for the condition to clear, then forms from everything queued (<= lim: the AUTO cap keeps >= 2 groups).
//    One or two decoders never wait (nothing to merge); a prefill piece at the front never waits; the stats count the waits,
//    the merges (a wait that brought lanes) and the timeouts. A row's bytes do not depend on its group (the rows contract).
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace ie {

class Glm5LanePipe {
public:
    // One lane's step as a stage sees it. `wide` is the lane's residual buffer [max_T x wide_row] floats: stage s < last
    // writes its output rows there, stage s > 0 reads them (one buffer suffices: a stage reads it before it writes it).
    struct Step {
        uint32_t lane = 0, T = 0, pos0 = 0;
        const int32_t* ids = nullptr;
        float* wide = nullptr;
    };
    // Run stage `stage` of `step` on the calling (stage) thread: "" or the error. Called for stages 0..n-1 in order.
    using StageFn = std::function<std::string(uint32_t stage, const Step& step)>;
    // The lane's step finished its last stage (its position is committed): runs on the last stage's thread, may submit the
    // lane's next step (once).
    using DoneFn = std::function<void(uint32_t lane, uint32_t T, uint32_t pos0)>;
    // Rows mode: one card's part of a group's step (G >= 2 lanes, 1 row each, in group order); gwide = the group's rows
    // [max_group x wide_row] floats (row i = steps[i]).
    using RowsStageFn = std::function<std::string(uint32_t stage, std::span<const Step> steps, float* gwide)>;
    // Rows mode: a group of G >= 2 lanes finished its last stage (every position committed); may resubmit each lane once.
    using RowsDoneFn = std::function<void(std::span<const uint32_t> lanes)>;
    using IdleFn = std::function<void()>;   // P4 B34: stage 0 finished a step and has nothing queued (see the header)

    Glm5LanePipe(uint32_t n_lanes, uint32_t max_T, uint64_t wide_row, bool ahead = false);
    ~Glm5LanePipe();
    Glm5LanePipe(const Glm5LanePipe&) = delete;
    Glm5LanePipe& operator=(const Glm5LanePipe&) = delete;

    uint32_t n_lanes() const { return uint32_t(lanes_.size()); }
    uint32_t max_T() const { return max_T_; }
    uint64_t host_bytes() const;   // the per-lane buffers the constructor allocated

    std::string start(uint32_t n_stages, StageFn stage, DoneFn done, IdleFn idle = {});
    // Rows mode (see the header): max_group 1..16 rows a group, group_cap 0 = AUTO.
    std::string start_rows(uint32_t n_stages, StageFn stage, RowsStageFn rows_stage, DoneFn done, RowsDoneFn rows_done,
                           uint32_t max_group, uint32_t group_cap, IdleFn idle = {});
    bool rows_mode() const;
    // Rows mode: allocate (and touch) the group pool's host buffers now, before a start_rows with the same max_group (which
    // then keeps them); a no-op while running
    void prepare_rows(uint32_t max_group);
    // Rows mode: groups that finished every stage since the pipe's first start_rows (cumulative over resumes), by size
    std::vector<uint64_t> group_sizes() const;
    // P4 B42 (see the header): the regroup wait at stage 0, wait_us 0 = off (the default). Any time; takes effect at the next
    // group formation.
    void set_regroup(uint32_t wait_us);
    struct RegroupStats { uint64_t waits = 0, merges = 0, timeouts = 0; double wait_ms = 0; };
    RegroupStats regroup_stats() const;
    // front (P4 B39 (5)): the step goes to the FRONT of stage 0's queue instead of its back -- the owner's bounded priority (a
    // prompt's small tail piece ahead of other lanes' pieces); the claim rules are the same
    std::string submit(uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0, bool front = false);
    // P4 B34 (see the header): taken = queued; "" and not taken = not now; an error = a request no lane state allows
    std::string submit_ahead(uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0, bool& taken);
    std::string stop();
    uint32_t cancel();   // P4 B33 (see the header): drop the waiting steps; returns the steps a stage is running
    // P4 B39: wait until the lane holds no claim (no step of it in flight, queued or in its done callback) -- the other lanes'
    // steps keep running. For an owner that works on ONE lane's state beside the running pipe (the crown's drain-free serial
    // turns): the lane's last callback may still hold its claim when the owner is told the step landed. Called from a stage
    // thread it refuses instead of deadlocking.
    std::string wait_lane_idle(uint32_t lane);
    bool running() const { return !th_.empty(); }
    uint32_t lane_pos(uint32_t lane) const;   // the lane's committed position
    uint64_t steps_done() const;              // steps that finished every stage since start
    // P4 B8 (Flash-Next serving): the caller advanced the lane OUTSIDE the pipe (a prefill run serially in its turn, the
    // pipe stopped or the lane idle) and its state now ends at n_pos on every stage. Refused while the lane has a step in
    // flight, and while its last step failed part-way (P4 B9: moving a half-stepped lane's position would hide that; only
    // reset_lane or a new sequence at pos0 0 clears the failed mark).
    std::string set_lane_pos(uint32_t lane, uint32_t n_pos);
    // The caller reset the lane's state outside the pipe (a new, empty sequence): its position is 0 and the failed mark is
    // cleared. Refused while the lane has a step in flight.
    std::string reset_lane(uint32_t lane);
    std::string error() const;                // the latched stage error ("" = none); stop() returns and clears it

private:
    // A lane's steps live in slots (1, or 2 with `ahead`): slot s = ids[s * max_T ...], wide[s * max_T * wide_row ...]. The
    // stage queues hold lane * 2 + slot.
    struct Lane {
        std::vector<int32_t> ids;
        std::vector<float> wide;
        uint32_t T[2] = {0, 0}, pos0[2] = {0, 0}, n_pos = 0;
        uint32_t nfl = 0;      // claims: steps in flight, one in its callback included (0..2; 2 = a lookahead)
        uint32_t cur = 0;      // the slot of the older step in flight
        bool at0 = false;      // P4 B34: a step of the lane waits for or runs stage 0
        bool in_cb = false, resub = false;
        bool failed = false;   // its last step failed part-way: only pos0 0 (a new sequence) may follow
        std::thread::id cb_tid;
    };
    void stage_loop(uint32_t s);
    void stage_loop_rows(uint32_t s);
    void make_groups(uint32_t max_group);   // (mu held or not running)
    // the group pool: one group per step that can be past stage 0 at once (a step a lane, two with lookahead slots) + the
    // one stage 0 forms
    size_t n_groups() const { return lanes_.size() * (ahead_ ? 2 : 1) + 1; }
    void drop_claim(Lane& ln, uint32_t slot);   // (mu held) a step leaves the pipe without a callback, or its callback returned
    int32_t* slot_ids(Lane& ln, uint32_t slot) { return ln.ids.data() + size_t(slot) * max_T_; }
    float*   slot_wide(Lane& ln, uint32_t slot) { return ln.wide.data() + size_t(slot) * max_T_ * wide_row_; }
    struct Group {
        std::vector<uint32_t> lanes;
        std::vector<uint32_t> slots;   // (P4 B34) each lane's slot
        std::vector<Step> steps;
        std::vector<float> wide;   // [max_group x wide_row]
        bool decode = false;       // (P4 B42) decode rows (not a lone prefill piece / new sequence), while in flight
    };
    void free_group(uint32_t gi);   // (mu held) the group leaves the pipe: back to the pool, the regroup's accounting

    std::vector<Lane> lanes_;
    uint32_t max_T_ = 0;
    uint64_t wide_row_ = 0;
    std::vector<std::thread> th_;
    std::vector<std::deque<uint32_t>> q_;   // [stage] lanes waiting for it, FIFO
    mutable std::mutex mu_;
    std::condition_variable cv_;
    uint32_t busy_ = 0;                     // lanes with a step in flight (claimed through their callback)
    uint32_t running_ = 0;                  // P4 B33: steps (or groups) a stage thread holds: in its stage or its callback
    bool cancel_ = false;                   // P4 B33: cancel() since the last start
    bool stop_ = false;
    std::string err_;
    uint64_t steps_ = 0;
    StageFn stage_;
    DoneFn done_;
    bool ahead_ = false;                    // P4 B34: two slots a lane (submit_ahead)
    IdleFn idle_;
    bool s0_busy_ = false;                  // P4 B34: stage 0 runs a step (or group)
    // (mu held) P4 B34: stage 0 is idle and the lane may take a lookahead now -- worth calling idle_
    bool ahead_possible(const Lane& ln) const {
        return idle_ && !cancel_ && !s0_busy_ && q_[0].empty() && ln.nfl == 1 && !ln.at0 && !ln.in_cb;
    }
    // rows mode
    bool rows_ = false, pgate_ = false;
    uint32_t max_group_ = 1, group_cap_ = 0;
    std::vector<Group> groups_;              // pool: n_groups() (a step is in at most one group)
    std::vector<uint32_t> gfree_;
    std::vector<std::deque<uint32_t>> gq_;   // [stage >= 1] groups waiting for it, FIFO
    std::vector<uint64_t> gsizes_;
    RowsStageFn rows_stage_;
    RowsDoneFn rows_done_;
    // P4 B42 regroup (mu held): on with a bound; the decode groups formed and not yet freed; whether the group at the last stage
    // (running or in its callback) is a decode group, and which; the stats
    bool regroup_ = false;
    std::chrono::microseconds regroup_wait_{0};
    uint32_t dec_inflight_ = 0;
    bool last_dec_running_ = false;
    uint32_t last_gi_ = UINT32_MAX;
    RegroupStats rg_;
};

}  // namespace ie
