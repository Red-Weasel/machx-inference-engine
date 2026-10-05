// include/ie/lanes_serve.hpp -- `ie serve --parallel N` on request lanes, shared by every arch (P4 B8,
// docs/lanes/LANES_SERVE.md). First user: Qwen3.8-Flash-Next (src/engine/engine.cpp, Q4eLanesModel).
//
// The machinery MiMo B4 (docs/mimo26/P4_B4_SERVE.md) and DeepSeek-V4.1 B6b (docs/deepseek41/P4_B6B_SERVE.md) each built for
// themselves, taken out of the arch: the lane table, the admission wait and the lane choice (the reply-room rule), per-request
// lane ownership, the serial turn with the pipe paused and its handover, the serial worker thread, the prefill plan run in the
// turn (a lone prompt) or through the lane pipe beside decoding lanes, the done callback (commit, cancel, resubmit or park), a
// pipe error's recovery, teardown ordering and the /health fields. An arch supplies its state and device work through
// LanesModel (the hooks below); the protocol is V4.1 B6b's (its gated behaviour: pause/park, handover, lost lanes reset in a
// turn, the pipe never stopped from a callback, the serve mutex before the pipe's).
// P4 B39: an arch may declare a turn kind drain-free (LanesModel::serial_drains): the pipe then runs on through that turn, the
// turn waits for its own lane only (pipe_wait_lane), and the plan may put the prompt's tail through the pipe behind a snapshot
// boundary (LanesPlan::snap, LanesModel::snapshot). The crown does; every other arch keeps the paused turn.
//
// No SYCL here: the protocol is unit-tested on the CPU with a fake model (tests/unit/lanes_serve_test.cpp).
//
// CardPipe (below) is the generic two-card pipe for a model that runs as one object per card (Flash-Next, GLM): GLM B7's
// Glm5LanePipe with pause/resume on top.
#pragma once

#include "ie/glm5_lanes.hpp"
#include "ie/ngram_draft.hpp"   // Ds41NgramIndex (P4 B55 lookup rounds)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <cstdint>
#include <functional>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace ie {

// The request's sampler settings (the engine's SamplingParams, without the engine header).
struct LanesSampling {
    float    temperature = 0.7f;
    uint32_t top_k = 40;
    float    top_p = 0.95f, min_p = 0.f;
    float    presence_penalty = 0.f, frequency_penalty = 0.f, repeat_penalty = 1.f;
    uint32_t repeat_window = 64;
    uint32_t max_tokens = 512;
    bool     ignore_eos = false;
    bool penalties() const { return repeat_penalty != 1.f || presence_penalty != 0.f || frequency_penalty != 0.f; }
};

struct LanesRequest {
    const std::vector<int32_t>* ids = nullptr;   // the prompt (the caller owns it for the whole run)
    LanesSampling sp;
    uint64_t rng = 0;        // the sampler's seed: the k-th sampled id uses rng + k (the engine's serial loop)
    uint32_t snap_at = 0;    // the prompt-cache boundary the arch snapshots at (<= ids->size(); 0 = none)
    // P4 B15: the SHARED-prefix boundary -- where the conversation-independent part of the prompt (the system prompt and the
    // tools) ends, from the chat template (0 = none). An arch that supports it splits the prefill there (plan's `mark`) and
    // snapshots the state in LanesModel::mark, restorable into ANY lane; its prefill is split there with or without a hit, so a
    // restore gives the bytes a cold prefill gives.
    uint32_t share_at = 0;
    bool reply_cache = false;   // P4 B10: after a "stop"/"length" reply, LanesModel::finish runs in a serial turn
    int32_t think_close = -1;   // P4 B27: the </think> id (-1: none); LanesResult::answer_start
    // P4 B36 (B): the prompt ends with a user query, so snap_at is that query's end: the conversation's ANCHOR, which the next
    // query's prompt shares even after the template re-renders the turns between (the arch's snapshot there is kept through
    // the cache's supersede; Engine::chat sets it)
    bool anchor = false;
};

struct LanesResult {
    std::string text, finish_reason, cache_source;
    // P4 B27: the byte offset in `text` after the first generated </think> (GenerateResult::answer_start); npos when none.
    size_t answer_start = std::string::npos;
    uint32_t prompt_tokens = 0, cached_tokens = 0, completion_tokens = 0, lane = 0;
    double   prefill_ms = 0, decode_ms = 0;
};

using LanesTokenFn = std::function<bool(std::string_view)>;   // the server's callback; an empty piece = a liveness probe

// One lane as the choice sees it: owned or idle, holding a conversation or empty, its capacity (the prompt must be below it; the
// reply fits in cap - prompt), the prompt positions its OWN state serves, the end of the last prompt it holds (0: none), its
// last release (larger = more recent).
struct LanesLaneView { bool idle = false, occupied = false; uint32_t cap = 0, match = 0, last_end = 0; uint64_t tick = 0; };

// The lane for a prompt, among the IDLE lanes it fits (prompt < cap) -- V4.1 B6b's ds41_choose_lane (MiMo B4's rule, ba287bd,
// with V4.1's continuation preference; host-tested in tests/unit/ds41_serve_rules_test.cpp), verbatim:
//   1. the lanes that leave the reply room first: cap - prompt >= min(budget, cap / 4);
//   2. then the lane whose own state serves the most of the prompt, when that is at least `min_tokens` or reaches the lane's
//      last prompt end; then an empty lane; then the smallest capacity; then the least recently released; then the lower index.
// budget 0 = unlimited. -1 = no idle lane fits the prompt.
inline int lanes_choose_lane(const std::vector<LanesLaneView>& lanes, uint32_t prompt, uint32_t budget, uint32_t min_tokens) {
    const uint32_t want = budget ? budget : UINT32_MAX;
    int best = -1;
    uint64_t bk[6] = {};
    for (size_t i = 0; i < lanes.size(); ++i) {
        const LanesLaneView& l = lanes[i];
        if (!l.idle || l.cap <= prompt) continue;
        const bool roomy = l.cap - prompt >= std::min(want, l.cap / 4);
        const bool big = l.match >= min_tokens || (l.last_end && l.match >= l.last_end);
        const uint64_t k[6] = {roomy ? 0u : 1u, big ? 0u : 1u, big ? uint64_t(~l.match) : 0u, l.occupied ? 1u : 0u, l.cap, l.tick};
        bool less = best < 0;
        for (int j = 0; j < 6 && !less; ++j) { if (k[j] != bk[j]) { less = k[j] < bk[j]; break; } }
        if (less) { best = int(i); for (int j = 0; j < 6; ++j) bk[j] = k[j]; }
    }
    return best;
}

// A request's reply budget on a lane: the whole max_tokens when the prompt and it fit, else the positions left (64-bit: an
// unlimited budget cannot wrap). max_tokens 0 = unlimited.
inline uint32_t lanes_reply_budget(uint32_t prompt, uint32_t max_tokens, uint32_t cap) {
    if (prompt >= cap) return 0;
    return max_tokens == 0 || uint64_t(prompt) + max_tokens > cap ? cap - prompt : max_tokens;
}

// The repetition window (the engine's serial loop and 27B stepper): the last min(repeat_window, |hist|, 512) ids of
// prompt ++ output.
inline std::span<const int32_t> lanes_penalty_window(const std::vector<int32_t>& hist, uint32_t repeat_window) {
    const size_t nw = std::min<size_t>({size_t(repeat_window), hist.size(), size_t(512)});
    return std::span<const int32_t>(hist.data() + (hist.size() - nw), nw);
}

using LanesChunk = std::pair<uint32_t, uint32_t>;   // (pos0, rows)
struct LanesPlan {
    std::vector<LanesChunk> chunks;   // the prefill pieces covering [reused, Tp), in order; each <= the pipe's max rows
    uint32_t Tp = 0;                  // where the pipe's part of the prompt ends; prompt_end does the rest in a turn
    // P4 B15: a chunk of `chunks` ends here (reused < mark < Tp); after it lands, LanesModel::mark runs in a serial turn with the
    // lane's state exactly at `mark` (0 = none)
    uint32_t mark = 0;
    // P4 B39: a chunk of `chunks` ends here (mark < snap < Tp); after it lands, LanesModel::snapshot runs in a serial turn with the
    // lane's state exactly at `snap` (the conversation snapshot prompt_end takes at Tp == snap_at), then the chunks after it --
    // the prompt's tail, which prompt_end ran in its turn -- go through the pipe; prompt_end then runs with kept = true and no
    // rows left (0 = none). No lookahead crosses it.
    uint32_t snap = 0;
};

// What an arch provides. Threads: the host reads run under the serve mutex (the turn held, the lane idle); the serial-turn
// calls run on the serial worker with the pipe paused; sample(first = false) and keep_logits run on the pipe's done-callback
// thread for that lane's step (the lane is claimed through the callback: nothing else runs it).
class LanesModel {
public:
    virtual ~LanesModel() = default;
    virtual const char* tag() const = 0;                        // log prefix, e.g. "q4e lanes"
    virtual uint32_t n_lanes() const = 0;
    virtual uint32_t lane_cap(uint32_t lane) const = 0;         // positions the lane holds
    // host, read-only: the prompt positions the lane's OWN state would serve (its live conversation or snapshot), and whether
    // it holds a conversation at all
    virtual uint32_t own_match(uint32_t lane, std::span<const int32_t> ids) const = 0;
    virtual bool occupied(uint32_t lane) const = 0;
    virtual uint32_t last_end(uint32_t lane) const = 0;         // the end of the last prompt the lane holds (0: none)
    virtual uint32_t min_match_tokens() const { return 1024; }  // a smaller own match does not beat an empty lane (V4.1: min_slot_tokens)
    // the serial turn: select the lane and serve what its cache can ([0, reused)); reused == 0 leaves the lane reset (a new
    // sequence). `source` names what served it ("" = cache off).
    virtual std::string prefix_prepare(uint32_t lane, const LanesRequest& rq, uint32_t& reused, std::string& source) = 0;
    virtual LanesPlan plan(const LanesRequest& rq, uint32_t reused) = 0;
    // the serial turn: run chunks[0..) on the lane (the cards pipelined over them when the arch can), asking `stop` before each
    // chunk after the first; `done` = the chunks that ran
    virtual std::string prefill_serial(uint32_t lane, const LanesRequest& rq, std::span<const LanesChunk> chunks,
                                       const std::function<bool()>& stop, size_t& done) = 0;
    // the serial turn, [0, Tp) in: the arch's prompt-end work (its checkpoints, the rows [Tp, T)); leaves the prompt's last
    // logits for sample(first = true). kept = the pipe ran the prompt's last row (keep_logits kept its logits).
    virtual std::string prompt_end(uint32_t lane, const LanesRequest& rq, uint32_t Tp, uint32_t reused, bool kept) = 0;
    virtual std::string reset_lane(uint32_t lane) = 0;          // the serial turn: forget the lane's state
    // P4 B15: the serial turn, the lane's state ends exactly at pos == plan.mark: snapshot the shared prefix [0, pos) of
    // rq.ids (restorable into any lane by prefix_prepare). An error is logged, not the request's.
    virtual std::string mark(uint32_t lane, const LanesRequest& rq, uint32_t pos) {
        (void)lane; (void)rq; (void)pos;
        return {};
    }
    // P4 B39: the serial turn, the lane's state ends exactly at pos == plan.snap: the arch's conversation snapshot (what its
    // prompt_end does at Tp == snap_at: the cache insert, its checkpoints). An error is logged, not the request's.
    virtual std::string snapshot(uint32_t lane, const LanesRequest& rq, uint32_t pos) {
        (void)lane; (void)rq; (void)pos;
        return {};
    }
    // P4 B39: the serial turns' kinds, and whether a turn of that kind needs the pipe DRAINED before its device work
    // (pipe_pause: every step in flight on every card finished, the stage threads stopped; resumed at the release). Default:
    // every kind drains (B6b-B38's protocol). An arch answers false for a kind whose serial work touches only the turn's own
    // lane's state and the arch's own caches, addressed by lane index (no per-device "current lane" register a running stage
    // also sets), with no scratch a running step uses, on queues that order it behind the steps in flight; LanesServe then
    // leaves the pipe running (the other lanes' steps keep flowing), waits for the turn's own lane through pipe_wait_lane,
    // and submits the parked lanes at the release even before a handover. The turn's work must also not touch host state a
    // done callback reads without the serve mutex. kFinishStop / kFinishLength = finish after a "stop" / "length" reply.
    enum class Turn : uint8_t { kPrepare, kMark, kSnapshot, kPromptEnd, kFinishStop, kFinishLength, kReset };
    virtual bool serial_drains(Turn k) const { (void)k; return true; }
    // P4 B39: wait until the lane holds no claim in the pipe (its last done callback returned); the other lanes run on. Never
    // from a callback. "" = idle now; an error = the arch has no per-lane wait (LanesServe drains the pipe for that turn instead).
    virtual std::string pipe_wait_lane(uint32_t lane) { (void)lane; return "no per-lane wait"; }
    // P4 B15: host-only -- the prefix prefix_prepare would restore for this prompt now (into any lane; 0 = none). Called from a
    // done callback or by the turn holder, never beside the serial worker's device work (which is what mutates the caches).
    // P4 B39: with drain-free turns a callback's call CAN meet the serial worker's restore or snapshot: the arch guards its cache
    // (the crown answers 0 while its worker holds the cache; the check is re-run at the next callback).
    virtual uint32_t cache_peek(const LanesRequest& rq) const { (void)rq; return 0; }
    // P4 B29: the conversation snapshots the arch's prompt cache took so far (prompt-end + reply; not the shared-prefix
    // marks, counted by LanesServe), for /health "snapshots". Any thread.
    virtual uint64_t snapshots() const { return 0; }
    // P4 B42: the arch's pipe counters as a JSON object for /health's "pipe" (the crown: the decode groups by size and the regroup
    // waits); "" = none. Any thread.
    virtual std::string pipe_stats() const { return {}; }
    // one id from the lane's logits: first = the prompt's (prompt_end's), else the step that just finished. window = the
    // repetition window (prompt tail + output), seed = rng + the ids sampled before.
    virtual int32_t sample(uint32_t lane, bool first, const LanesSampling& sp, std::span<const int32_t> window,
                           uint64_t seed, std::string& err) = 0;
    virtual void keep_logits(uint32_t lane) = 0;                // the done callback of the piece that reached T
    virtual bool is_stop(int32_t id) const = 0;
    // P4 B10: the serial turn after a request with rq.reply_cache ended "stop" or "length" with >= 1 id (not aborted, lane not
    // lost; the lane is still the request's). out = the reply's ids; the lane's state covers prompt ++ out[0, pos - T) -- on
    // "length" the last id is not forwarded yet (pos = T + |out| - 1), on "stop" it is (pos = T + |out|). The arch's reply
    // snapshot (--parallel 1's gen-cache). An error is logged, not the request's.
    virtual std::string finish(uint32_t lane, const LanesRequest& rq, std::span<const int32_t> out, const std::string& finish,
                               uint32_t pos) {
        (void)lane; (void)rq; (void)out; (void)finish; (void)pos;
        return {};
    }
    virtual std::string detok(std::span<const int32_t> out) const = 0;   // the reply's text so far (whole)
    // the lane pipe: one step in flight per lane; `done(lane)` runs on the last stage's thread and may resubmit the lane once
    using DoneFn = std::function<void(uint32_t lane)>;
    virtual std::string pipe_start(DoneFn done) = 0;
    // P4 B14 rows mode (Glm5LanePipe::start_rows): the pipe groups the decoding lanes' 1-row steps and runs each group as one
    // card step; `rows_done(lanes)` runs once for a group of >= 2 lanes (a group of one calls `done`). rows() says whether
    // the serve module starts the pipe that way (constant for the model's life); the default is the per-lane pipe.
    using RowsDoneFn = std::function<void(std::span<const uint32_t> lanes)>;
    virtual bool rows() const { return false; }
    virtual std::string pipe_start_rows(DoneFn done, RowsDoneFn rows_done) {
        (void)done; (void)rows_done;
        return "rows not supported";
    }
    // One id per lane of a finished group (the rows callback's thread; each lane claimed through the callback): picks[i] from
    // lanes[i]'s logits with sp[i], windows[i], seeds[i] -- each exactly what sample(lanes[i], first = false, ...) returns.
    // An error fails every lane of the call. The default is sample() lane by lane.
    virtual std::string sample_rows(std::span<const uint32_t> lanes, std::span<const LanesSampling* const> sp,
                                    std::span<const std::span<const int32_t>> windows, std::span<const uint64_t> seeds,
                                    std::span<int32_t> picks) {
        for (size_t i = 0; i < lanes.size(); ++i) {
            std::string err;
            picks[i] = sample(lanes[i], false, *sp[i], windows[i], seeds[i], err);
            if (!err.empty()) return "lane " + std::to_string(lanes[i]) + ": " + err;
        }
        return {};
    }
    virtual std::string pipe_submit(uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0) = 0;
    // P4 B55: prompt-lookup rounds on a decoding lane (LanesServe::Options::look). look_rows() = the longest round the
    // arch runs (0 = none: no lookups). A round is ONE step of T rows at the lane's position -- ids[0] the lane's next
    // input, ids[1..] the drafted continuation -- that keeps every row's logits and a copy of the lane's state from
    // before it (pipe_submit_look); sample_look gives one id per row, row r from its logits with windows[r] / seeds[r]
    // (each what sample() gives for that row); pipe_submit_redo puts the lane's state back to before the round and runs
    // its first T rows again (the rows that were followed). One round (with its redo) is in flight at a time.
    virtual uint32_t look_rows() const { return 0; }
    virtual std::string pipe_submit_look(uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0) {
        (void)lane; (void)ids; (void)T; (void)pos0;
        return "lookup rounds not supported";
    }
    virtual std::string sample_look(uint32_t lane, uint32_t rows, const LanesSampling& sp, std::span<const std::span<const int32_t>> windows,
                                    std::span<const uint64_t> seeds, std::span<int32_t> picks) {
        (void)lane; (void)rows; (void)sp; (void)windows; (void)seeds; (void)picks;
        return "lookup rounds not supported";
    }
    virtual std::string pipe_submit_redo(uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0) {
        (void)lane; (void)ids; (void)T; (void)pos0;
        return "lookup rounds not supported";
    }
    // P4 B39 (5): the lane's step AHEAD of the other lanes' queued steps (a prompt's tail piece past plan.snap, <= kTailFront rows:
    // v0.2.6 ran it inside the drained prompt-end turn, before everything; through the pipe it landed behind the other prompts'
    // first pieces -- the wave leader's 7 rows waited 1.4 s behind 14 followers). Default: no priority (pipe_submit).
    virtual std::string pipe_submit_front(uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0) { return pipe_submit(lane, ids, T, pos0); }
    // P4 B34 (the lane pipeline): true = the arch's pipe takes a prefilling lane's next piece while the lane's previous piece
    // is still on a later card (Glm5LanePipe::submit_ahead); `idle` is then registered with it (card 0 is idle and a lookahead
    // may have become possible) and LanesServe offers that piece from there through pipe_submit_ahead. Called once, by
    // LanesServe's constructor. Default: no lookahead (`idle` dropped).
    virtual bool pipe_lookahead(std::function<void()> idle) { (void)idle; return false; }
    // taken = queued behind the lane's piece in flight; not taken and "" = not now; an error = a request no lane state allows
    virtual std::string pipe_submit_ahead(uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0, bool& taken) {
        (void)lane; (void)ids; (void)T; (void)pos0;
        taken = false;
        return {};
    }
    // P4 B33: the process is stopping (LanesServe::abort_all): drop the pipe's steps that have not started (CardPipe::cancel)
    // and make the device steps running now end soon (an arch whose prefill forward can stop at a layer boundary asks for it
    // here). Any thread, never blocks, never from a done callback. Returns the steps a card stage holds -- what the teardown's
    // pipe_stop still waits for. Default: nothing dropped, 0.
    virtual uint32_t abort() { return 0; }
    virtual std::string pipe_pause() = 0;          // waits for the steps in flight (the callbacks park their lanes first)
    virtual std::string pipe_resume() = 0;
    virtual bool        pipe_paused() const = 0;
    virtual std::string pipe_stop() = 0;           // never from a callback
    virtual std::string pipe_error() const = 0;
    virtual uint32_t    pipe_max_rows() const = 0;
};

// Glm5LanePipe with pause/resume: pause = stop() (it waits for no step in flight; the serve protocol parks the lanes first),
// resume = start() again with the same stages. Stage threads are made per resume: an arch whose stage threads keep per-thread
// state (an OpenMP team: V4.1, MiMo) keeps its own persistent pipe instead. A fresh submit (not from a callback) at a lane's
// position that serial work moved syncs the pipe's record of it first (Glm5LanePipe::set_lane_pos; a refused sync refuses the
// submit). The model's reset_lane calls reset_lane here too: it is what clears a lane's failed mark (a stage error part-way).
class CardPipe {
public:
    // P4 B34: ahead = the lane pipeline's second buffer set per lane (Glm5LanePipe::submit_ahead)
    CardPipe(uint32_t n_lanes, uint32_t max_rows, uint64_t wide_row, uint32_t n_stages, Glm5LanePipe::StageFn stage, bool ahead = false)
        : pipe_(n_lanes, max_rows, wide_row, ahead), n_stages_(n_stages), stage_(std::move(stage)) {}
    // P4 B14: with a rows stage, start(done, rows_done) with a rows_done runs the pipe in rows mode (groups of up to
    // max_group 1-row steps, group_cap 0 = AUTO); resume() keeps the mode of the last start.
    CardPipe(uint32_t n_lanes, uint32_t max_rows, uint64_t wide_row, uint32_t n_stages, Glm5LanePipe::StageFn stage,
             Glm5LanePipe::RowsStageFn rows_stage, uint32_t max_group, uint32_t group_cap, bool ahead = false)
        : pipe_(n_lanes, max_rows, wide_row, ahead), n_stages_(n_stages), stage_(std::move(stage)), rows_stage_(std::move(rows_stage)),
          max_group_(max_group), group_cap_(group_cap) { pipe_.prepare_rows(max_group); }   // (B14 gate #5: at load)
    std::string start(LanesModel::DoneFn done, LanesModel::RowsDoneFn rows_done = {});
    bool        rows_mode() const { return rows_on_; }
    std::vector<uint64_t> group_sizes() const { return pipe_.group_sizes(); }
    std::string submit(uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0, bool front = false);   // (front: P4 B39 (5))
    // P4 B34: the pipe's idle hook for every start / resume from now on; the lookahead submit
    void        set_idle(Glm5LanePipe::IdleFn idle) { idle_ = std::move(idle); }
    std::string submit_ahead(uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0, bool& taken) {
        return pipe_.submit_ahead(lane, ids, T, pos0, taken);
    }
    std::string reset_lane(uint32_t lane);   // the model reset the lane's state (Glm5LanePipe::reset_lane)
    std::string rewind_lane(uint32_t lane, uint32_t n_pos) { return pipe_.rewind_lane(lane, n_pos); }   // P4 B55
    std::string pause();
    std::string resume();
    bool        paused() const { return paused_; }
    std::string stop();
    uint32_t    cancel() { return pipe_.cancel(); }   // P4 B33: Glm5LanePipe::cancel (a process stop)
    std::string wait_lane(uint32_t lane) { return pipe_.wait_lane_idle(lane); }   // P4 B39: Glm5LanePipe::wait_lane_idle
    // P4 B42: the rows-mode regroup wait (Glm5LanePipe::set_regroup; 0 = off) and its counts
    void set_regroup(uint32_t wait_us) { pipe_.set_regroup(wait_us); }
    Glm5LanePipe::RegroupStats regroup_stats() const { return pipe_.regroup_stats(); }
    std::string error() const { return pipe_.error(); }
    uint32_t    max_rows() const { return pipe_.max_T(); }
    uint64_t    host_bytes() const { return pipe_.host_bytes(); }
    uint64_t    steps_done() const { return pipe_.steps_done(); }
private:
    Glm5LanePipe pipe_;
    uint32_t n_stages_;
    Glm5LanePipe::StageFn stage_;
    LanesModel::DoneFn done_;
    bool paused_ = false;
    Glm5LanePipe::RowsStageFn rows_stage_;
    uint32_t max_group_ = 1, group_cap_ = 0;
    LanesModel::RowsDoneFn rows_done_;
    bool rows_on_ = false;
    Glm5LanePipe::IdleFn idle_;   // (P4 B34)
    std::string launch();
};

class LanesServe {
public:
    struct Options {
        uint32_t mix_chunk = 0;    // cap on a prefill piece's rows while another lane is busy (0 = the plan's own pieces)
        bool     trace = false;    // one stderr line per pipe step
        // P4 B15 (the constructor reads IE_LANES_PREFILL_FIFO=0 / IE_LANES_HANDOVER_MS=<ms> over these):
        //  * prefill_fifo: prompts' pieces go through the pipe ONE prompt at a time, in arrival order (decoding lanes step between
        //    the pieces); a prompt waiting its turn re-runs prefix_prepare when LanesModel::cache_peek grows (an earlier prompt
        //    snapshotted a shared prefix it can restore). Off: every waiting prompt's pieces at once (B8-B14).
        //  * handover_ms: a turn is handed to the next waiting request with the pipe still paused only while the pipe has been
        //    paused for less than this; past it, and with decoding lanes parked, the turn is released normally (the parked lanes
        //    step once) and the waiter pauses the pipe again. 0 = always hand over (B8-B14).
        // Defaults OFF (B8-B14 scheduling): measured 2026-09-27 on the crown (4 decoders + a burst of 4 new ~13K prompts, B15
        // runs st1-st3): the FIFO cut decoder starvation (82 vs 35 decoder tokens in the burst, max gap 11 vs 28 s) but lengthened
        // the burst's TTFTs (45-91 s, window 2: 56-69 s, vs 52 s for all four with every prompt's pieces in flight). Opt in with
        // IE_LANES_PREFILL_FIFO=1 (it also re-prepares a queued prompt onto an earlier prompt's new shared prefix) and
        // IE_LANES_HANDOVER_MS=<ms>.
        bool     prefill_fifo = false;
        uint32_t handover_ms = 0;
        // P4 B34 (3) short-first (off here; the crown sets it from IE_Q35MOE_SHORT_*): a prompt whose pipe part has at most
        // max(2 x its largest piece, short_rows) rows left is SHORT. A short prompt need not wait for the FIFO window, and a long
        // prompt's next piece (and its lookahead) waits while another prompt's short work is due -- its pieces, its mark or
        // prompt-end turn -- until the guard: short_slots short pieces went into the pipe ahead of it, or short_wait_ms passed
        // since its last piece landed. Only the order of whole pieces across lanes changes: each lane's pieces, their cuts and
        // their per-card order are the same, so are the bytes.
        bool     short_first = false;
        uint32_t short_rows = 16384, short_slots = 2, short_wait_ms = 10000;
        // P4 B36 (A) re-cut (off here; the crown sets it from IE_Q35MOE_RECUT): a prefilling lane whose plan pieces are larger
        // than mix_chunk (its plan was made alone) cuts the pieces it has not sent yet into mix_chunk rows -- each plan chunk in
        // order, so the mark and Tp stay piece ends -- once another lane decodes or prefills a short prompt: a decode step then
        // waits behind at most a mix_chunk piece per card instead of a whole plan piece. Its bytes are a run's with that cut
        // sequence; a prompt that runs alone keeps its plan (--parallel 1's pieces).
        bool     recut = false;
        // P4 B36 (C) decode quota (0 = off here; the crown sets it from IE_Q35MOE_DECODE_QUOTA / _QUOTA_MAX_MS): a LONG prompt's
        // next piece (its pipe part, Tp - reused, above max(2 x its largest piece, short_rows): B34 (3)'s threshold over the whole
        // prompt, so a lead stays long to its end) waits while a decoding lane has made fewer than decode_quota steps since the
        // prompt's latest piece landed -- a lookahead too -- until quota_max_ms passed since the wait began (the guard). Only the
        // order of whole pieces across lanes changes, so the bytes are each lane's own pieces'.
        uint32_t decode_quota = 0, quota_max_ms = 2000;
        // P4 B55 prompt-lookup rounds (off here; the crown sets it from IE_Q35MOE_LANES_LOOKUP): when a decoding lane's
        // last tokens repeat an earlier span of its history (prompt ++ reply) of at least look_min tokens, the tokens that
        // followed it are run with the lane's next input as ONE step (LanesModel::pipe_submit_look) and sampled row by row
        // with the request's sampler; the reply follows the rows while the sampled id is the drafted one. Rows not followed:
        // the lane's state goes back and the followed rows run again (pipe_submit_redo). Probe first: a round is look_probe
        // rows (look_strong_rows on a repeated span of look_strong tokens) and doubles, up to the arch's look_rows(), after
        // each round followed to its end; a round with fewer than 3 rows followed blocks the lane's lookups for look_block
        // tokens, twice as long after each such round in a row (at most look_block_max). One round at a time over all lanes.
        // look_gap: after a round landed, the next round (any lane's) waits until every OTHER decoding lane made that many
        // steps, at most look_gap_ms (a round holds each card for tens of ms: the other lanes' steps wait behind it).
        uint32_t look_gap = 2, look_gap_ms = 250;
        bool     look = false;
        uint32_t look_min = 16, look_probe = 8, look_strong = 24, look_strong_rows = 32, look_block = 24, look_block_max = 384;
    };
    LanesServe(LanesModel& m, Options o);
    ~LanesServe();
    LanesServe(const LanesServe&) = delete;
    LanesServe& operator=(const LanesServe&) = delete;

    LanesResult run(const LanesRequest& rq, const LanesTokenFn& on_token);
    std::string status_json() const;   // /health fields, a JSON object (Engine::serving_status_json merges it)
    void shutdown();                   // teardown: stop admitting, join the serial worker, stop the pipe. Idempotent.
    // P4 B33: the process is stopping (Engine::abort_all, from the server's stop). Terminal and non-blocking: every request
    // ends now with "abort" (a lane whose step is on a card does not wait for it), no new device work starts -- no callback
    // submits, no turn is taken, the FIFO is emptied, a serial turn's prefill stops before its next chunk -- the pipe drops the steps
    // that have not started (LanesModel::abort), and later requests are refused. What runs on a card now finishes by itself;
    // shutdown() waits for it. Returns those steps: the pipe's plus the serial turn's device work (0 or 1). Idempotent.
    uint32_t abort_all();

    struct Lane {
        // kMark: at plan.mark, waits for its turn; kSnap (P4 B39): at plan.snap, waits for its turn
        enum class Phase : uint8_t { kIdle, kPrefill, kPromptReady, kDecode, kDone, kMark, kSnap };
        bool     busy = false;
        uint32_t cap = 0;
        Phase    phase = Phase::kIdle;
        const LanesRequest* rq = nullptr;
        LanesPlan plan; uint32_t chunk_at = 0;
        uint32_t reused = 0, pos = 0, max_new = 0, n_new = 0;
        int32_t  next = -1;
        uint32_t pend = 0;
        bool     kept = false;                 // the pipe ran the prompt's last row (its logits kept)
        std::vector<int32_t> hist;             // prompt ++ committed ids (the repetition window's source)
        std::vector<int32_t> out, outbox;
        std::string finish;
        bool     want_stop = false, lost = false, parked = false;
        uint32_t mark_at = 0;                  // P4 B15: plan.mark still due (0 = none)
        uint32_t snap_at = 0;                  // P4 B39: plan.snap still due (0 = none)
        bool     preparing = false;            // P4 B39: its prepare runs on the serial worker; its plan is not made yet
        bool     reprep = false;               // P4 B15: the FIFO head's cache_peek grew: its request re-runs prefix_prepare
        uint32_t repreps = 0;
        bool     ahead = false;                // P4 B34: chunks[chunk_at + 1] is in the pipe too (a lookahead, on_idle0)
        bool     held = false;                 // P4 B34 (3): long, its next piece waits for the short prompts (parked)
        uint32_t bypass = 0;                   //   short pieces that went into the pipe ahead of it since it was held
        uint32_t piece_max = 0;                //   its plan's largest piece (the short rule)
        bool     recut = false;                // P4 B36 (A): its pieces not yet sent were cut to mix_chunk rows
        bool     qheld = false, qfree = false;  // P4 B36 (C): its next piece waits for the decode quota; the guard let it go
        uint64_t dsteps = 0;                   //   its decode steps, never reset (the leads' baselines count them)
        std::vector<uint64_t> qbase;           //   every lane's dsteps when its latest piece landed (or its pipe part began)
        std::chrono::steady_clock::time_point t_qhold{};   // its quota wait began
        uint64_t tick = 0;
        std::chrono::steady_clock::time_point t_sub{}, t_ahead{}, t_land{};   // (t_land: its last piece landed, or its pipe part began)
        uint32_t steps = 0, chunks = 0;
        double   cb_ms = 0;
        // P4 B55 lookup rounds
        Ds41NgramIndex idx; bool idx_on = false;    // the history's n-gram index (prompt ++ committed ids)
        std::vector<int32_t> look_ids;              // the round's inputs
        uint32_t look_n = 0, look_base = 0;         // a round of look_n rows at look_base is in the pipe
        uint32_t redo_n = 0; bool redo_fly = false; // rows to run again (pending, or in the pipe)
        bool     look_end = false; std::string look_finish;   // the lane ends (with look_finish) once its redo landed
        uint32_t look_streak = 0, look_blocked = 0, look_block_len = 0, look_grow = 0;
    };

private:
    using Clock = std::chrono::steady_clock;
    void end_lane(Lane& l, const std::string& finish, bool lost);
    bool commit(Lane& l, int32_t id);
    void submit(Lane& l, uint32_t li);
    void on_idle0();                           // P4 B34: the pipe's idle hook (card 0 is idle: offer a lookahead)
    // P4 B34 (3) short-first (mu held)
    bool short_lane(const Lane& l) const;                      // its pipe part has <= max(2 x piece_max, short_rows) rows left
    bool extends_mark(const Lane& l, const Lane& e) const;     // l's prompt extends e's still-due mark (it waits for it)
    bool shorts_pending(const Lane& l, uint32_t li) const;     // another lane's short work is due (not waiting for l's mark)
    bool hold_long(Lane& l, uint32_t li);                      // l's next piece is due: true = it waits (held), else submit it
    void short_piece_in(uint32_t li);                          // a short piece went into the pipe: the held lanes' guard
    void maybe_recut(Lane& l, uint32_t li, size_t from);       // P4 B36 (A): the plan's pieces from `from` on (none sent yet)
    // P4 B36 (C) the decode quota (mu held)
    bool long_prompt(const Lane& l) const;                     // its pipe part is above max(2 x piece_max, short_rows)
    void quota_base(Lane& l);                                  // its latest piece landed (or its pipe part began): the baseline
    bool quota_due(const Lane& l, uint32_t li) const;          // a decoding lane made fewer than decode_quota steps since then
    bool quota_hold(Lane& l, uint32_t li);                     // l's next piece waits for the decoders (true), until the guard
    void quota_kick();                                         // decode steps landed: a held lane whose quota is met goes
    void on_done(uint32_t li);
    void on_done_rows(std::span<const uint32_t> lanes);
    void fifo_drop(uint32_t li);               // P4 B15: the lane leaves the prefill FIFO
    void fifo_kick();                          // P4 B15: start the FIFO head's pieces (or flag its re-prepare)
    bool fifo_waiting(uint32_t li) const;      // a prefill lane held in the FIFO (not its head, or its head re-preparing)
    bool step_landed(Lane& l, uint32_t li);
    void step_sampled(Lane& l, uint32_t li, int32_t id, const std::string& err);
    // P4 B55 (mu held)
    bool look_try(Lane& l, uint32_t li);          // a lookup round is due: look_ids / look_n / look_base are set
    void look_landed(Lane& l, uint32_t li);       // the round's callback: sample its rows, follow them, roll back or go on
    void decode_next(Lane& l, uint32_t li);       // a decode lane's next step is due (abort / park / submit)
    void pipe_failed(const std::string& e);
    // P4 B33: false = stopping, the turn not taken. P4 B39: drain = pause the pipe for the turn (every step in flight finished);
    // false = the pipe runs on (LanesModel::serial_drains answered false for the turn's kind). kind: a lane at its mark / snapshot
    // boundary (kMark / kSnapshot) goes before the kPrepare turns waiting with it (the B39 gate's wave: the followers of a shared
    // prefix cannot restore before the leader's mark lands, so every prepare turn ahead of the mark delays the whole wave); the
    // other kinds neither yield nor are yielded to (the default kReset: poll_pipe's recovery turn)
    bool take_turn(std::unique_lock<std::mutex>& lk, bool drain = true, LanesModel::Turn kind = LanesModel::Turn::kReset);
    void drain_pipe(std::unique_lock<std::mutex>& lk);   // (mu held) the pause: take_turn's drain, or lane_wait's fallback
    // P4 B39: a turn taken without the drain waits for ITS lane's steps in the pipe before its serial work (the lane's last
    // callback may still hold its claim); an arch without the per-lane wait gets the drain instead
    void lane_wait(std::unique_lock<std::mutex>& lk, uint32_t li);
    void release_turn(std::unique_lock<std::mutex>& lk);
    uint32_t pipe_end(const Lane& l) const { return l.plan.snap ? l.plan.snap : l.plan.Tp; }   // P4 B39: the pipe part's end for the short / long rules (the tail aside)
    void poll_pipe(std::unique_lock<std::mutex>& lk);
    int  choose(const std::vector<int32_t>& ids, uint32_t max_tokens) const;
    std::string serial(std::unique_lock<std::mutex>& lk, const std::function<void()>& fn,
                       const std::function<bool()>& alive = {}, std::atomic<bool>* gone = nullptr);
    uint32_t decoding() const;
    uint32_t busy() const;
    int32_t sample_into(Lane& l, uint32_t li, bool first, std::string& err);

    LanesModel& m_;
    Options     o_;
    std::vector<Lane> lanes_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    bool     piping_ = false, pause_ = false, turn_busy_ = false, stopping_ = false;
    std::deque<uint32_t> fifo_;                // P4 B15: prompts whose pieces go through the pipe, in arrival order
    bool     paused_valid_ = false;            // P4 B15: the pipe was paused by a turn at pause_t0_ and not resumed since
    std::chrono::steady_clock::time_point pause_t0_{};
    std::atomic<uint64_t> marks_{0};          // (the serial worker counts the marks of a turn's own prefill)
    uint64_t repreps_ = 0, budget_releases_ = 0;
    uint64_t lookaheads_ = 0;                  // P4 B34: pieces on_idle0 put into the pipe behind their lane's piece
    // P4 B34 (3): short pieces that went in ahead of a held long lane; guard firings (a held lane's piece released), of
    // them by the wait
    uint64_t short_bypass_ = 0, short_guard_ = 0, short_guard_wait_ = 0;
    uint64_t recuts_ = 0;                      // P4 B36 (A): lanes whose remaining plan was re-cut
    uint64_t quota_holds_ = 0, quota_guard_ = 0;   // P4 B36 (C): quota waits begun; of them ended by the guard
    int      look_owner_ = -1;                 // P4 B55: the lane whose lookup round (or its redo) is in flight or pending
    std::vector<uint64_t> look_gap_base_;      //   every lane's dsteps when the last round landed
    Clock::time_point look_gap_t0_{};
    uint64_t look_rounds_ = 0, look_drafted_ = 0, look_hits_ = 0, look_redos_ = 0;
    uint32_t turn_waiters_ = 0;
    uint64_t tick_ = 0;
    double   step_ms_ = 0;
    uint64_t steps_ = 0, tokens_ = 0, turns_ = 0, handovers_ = 0;
    double   paused_ms_ = 0;
    uint64_t drains_ = 0;                      // P4 B29: the turns' pipe_pause waits (count, total, longest)
    double   drain_ms_ = 0, drain_max_ms_ = 0;
    uint64_t turns_nodrain_ = 0;               // P4 B39: turns the pipe ran through (no pause)
    bool     turn_paused_ = false;             // P4 B39: the pipe is paused for the turn being held (paused_ms_ counts it)
    uint32_t boundary_waiters_ = 0;            // P4 B39: lanes at their mark / snapshot waiting for the turn (the prepares yield to them)
    uint64_t boundary_first_ = 0;              //   ... turns such a lane took while prepare turns were waiting
    // P4 B39 (5): requests waiting for a prepare turn (a new prompt, a re-prepare): other lanes ARRIVING -- a lone lane's parked
    // remainder is re-cut for them at its release (maybe_recut), before they are busy lanes
    uint32_t prepare_waiters_ = 0;
    // P4 B39 (5): a prompt's tail piece (past plan.snap) of at most this many rows goes to the front of the pipe's queue
    static constexpr uint32_t kTailFront = 512;
    Clock::time_point turn_t0_{};
    // the serial worker: the turn's device work runs on ONE persistent thread (V4.1 B6b: a thread that runs a forward may keep
    // per-thread state for good -- an OpenMP team -- so the HTTP request threads must not run it)
    std::thread             worker_;
    std::mutex              wmu_;
    std::condition_variable wcv_;
    const std::function<void()>* wjob_ = nullptr;
    bool                    wdone_ = false, wstop_ = false;
    std::string             werr_;
};

}  // namespace ie
