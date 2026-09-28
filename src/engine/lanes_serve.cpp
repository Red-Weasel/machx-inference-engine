// src/engine/lanes_serve.cpp -- `ie serve --parallel N` on request lanes, shared by every arch (P4 B8). See
// include/ie/lanes_serve.hpp. The protocol is DeepSeek-V4.1 B6b's (src/engine/ds41_engine.cpp, ds41_run_lanes and its serve_*
// helpers, gated 2026-09-26), with the arch's work behind LanesModel.
#include "ie/lanes_serve.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>

namespace ie {

namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

// The longest prefix of `s` that ends on a complete UTF-8 character (the engine's utf8_complete_prefix_len).
size_t utf8_complete(std::string_view s) {
    if (s.empty()) return 0;
    size_t i = s.size(), cont = 0;
    while (i > 0 && cont < 3 && (uint8_t(s[i - 1]) & 0xC0u) == 0x80u) { --i; ++cont; }
    if (i == 0) return 0;
    const uint8_t lead = uint8_t(s[i - 1]);
    const size_t need = (lead & 0x80u) == 0 ? 1 : (lead & 0xE0u) == 0xC0u ? 2 : (lead & 0xF0u) == 0xE0u ? 3 : (lead & 0xF8u) == 0xF0u ? 4 : 1;
    return (cont + 1 >= need) ? i - 1 + need : i - 1;
}
}  // namespace

// ---- CardPipe ----------------------------------------------------------------------------------------------------------------

std::string CardPipe::start(LanesModel::DoneFn done, LanesModel::RowsDoneFn rows_done) {
    done_ = std::move(done);
    rows_done_ = std::move(rows_done);
    rows_on_ = rows_stage_ && rows_done_;
    paused_ = false;
    return launch();
}

std::string CardPipe::launch() {
    LanesModel::DoneFn* d = &done_;
    if (!rows_on_) return pipe_.start(n_stages_, stage_, [d](uint32_t lane, uint32_t, uint32_t) { (*d)(lane); });
    LanesModel::RowsDoneFn* rd = &rows_done_;
    return pipe_.start_rows(n_stages_, stage_, rows_stage_, [d](uint32_t lane, uint32_t, uint32_t) { (*d)(lane); },
                            [rd](std::span<const uint32_t> lanes) { (*rd)(lanes); }, max_group_, group_cap_);
}

std::string CardPipe::submit(uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0) {
    // A fresh claim (the serve protocol submits one only for a lane it prepared in a turn: reset, restored or prefilled
    // there) at a position serial work moved syncs the pipe's record of it first, and a refused sync refuses the submit: a
    // lane with a step in flight, or one whose last step failed part-way and was not reset since (reset_lane). A callback's
    // resubmit is at the position the pipe committed, so it skips the sync and meets the pipe's own checks unchanged.
    if (pos0 != 0 && pipe_.lane_pos(lane) != pos0)
        if (auto e = pipe_.set_lane_pos(lane, pos0); !e.empty()) return "card pipe: position sync refused: " + e;
    return pipe_.submit(lane, ids, T, pos0);
}

std::string CardPipe::reset_lane(uint32_t lane) { return pipe_.reset_lane(lane); }

std::string CardPipe::pause() {
    const std::string e = pipe_.stop();
    paused_ = e.empty();
    return e;
}

std::string CardPipe::resume() {
    if (!paused_) return "card pipe: not paused";
    paused_ = false;
    return launch();
}

std::string CardPipe::stop() {
    paused_ = false;
    return pipe_.stop();
}

// ---- LanesServe --------------------------------------------------------------------------------------------------------------

LanesServe::LanesServe(LanesModel& m, Options o) : m_(m), o_(o), lanes_(m.n_lanes()) {
    for (uint32_t i = 0; i < lanes_.size(); ++i) lanes_[i].cap = m.lane_cap(i);
    if (const char* v = std::getenv("IE_LANES_PREFILL_FIFO")) o_.prefill_fifo = *v != '0';
    if (const char* v = std::getenv("IE_LANES_HANDOVER_MS")) o_.handover_ms = uint32_t(std::max(0, std::atoi(v)));
}

LanesServe::~LanesServe() { shutdown(); }

void LanesServe::shutdown() {
    // the callbacks stop resubmitting (stopping + pause); the idle serial worker ends; the pipe drains and joins its stages
    { std::lock_guard<std::mutex> lk(mu_); stopping_ = true; pause_ = true; }
    cv_.notify_all();
    { std::lock_guard<std::mutex> lk(wmu_); wstop_ = true; }
    wcv_.notify_all();
    if (worker_.joinable()) worker_.join();
    if (auto e = m_.pipe_stop(); !e.empty()) std::fprintf(stderr, "[%s] teardown pipe_stop: %s\n", m_.tag(), e.c_str());
}

uint32_t LanesServe::busy() const { uint32_t n = 0; for (const auto& l : lanes_) n += l.busy; return n; }
uint32_t LanesServe::decoding() const {
    uint32_t n = 0;
    for (const auto& l : lanes_) n += l.busy && l.phase == Lane::Phase::kDecode;
    return n;
}

// The lane's request ends (mu held): `finish` unless one is set; `lost` = its state is forgotten at the release
void LanesServe::end_lane(Lane& l, const std::string& finish, bool lost) {
    if (l.finish.empty()) l.finish = finish;
    l.lost = l.lost || lost; l.phase = Lane::Phase::kDone; l.parked = false; l.next = -1; l.reprep = false;
    fifo_drop(uint32_t(&l - lanes_.data()));
    cv_.notify_all();
}

// ---- P4 B15: the prefill FIFO (mu held throughout) ---------------------------------------------------------------------------

void LanesServe::fifo_drop(uint32_t li) {
    for (auto it = fifo_.begin(); it != fifo_.end(); ++it)
        if (*it == li) { fifo_.erase(it); return; }
}

bool LanesServe::fifo_waiting(uint32_t li) const {
    for (uint32_t x : fifo_) if (x == li) return true;
    return false;
}

// The FIFO's next moves: its first kWindow prompts (one per card stage, so both cards keep working on prefill pieces) may run.
// Stale entries leave; an entry that has not started and whose prompt the cache now serves further (an earlier prompt's
// shared-prefix snapshot) is flagged for its request to re-run prefix_prepare in a turn (once); an entry whose prompt extends
// an earlier running entry's still-due mark (the same new system prompt) waits for that mark instead of prefilling it too;
// otherwise a parked entry's next piece goes into the running pipe. Only from a done callback or by the turn holder:
// cache_peek never runs beside the serial worker's device work.
void LanesServe::fifo_kick() {
    constexpr size_t kWindow = 2;
    for (size_t i = 0; i < fifo_.size() && i < kWindow && !stopping_;) {
        const uint32_t h = fifo_[i];
        Lane& l = lanes_[h];
        if (!l.busy || (l.phase != Lane::Phase::kPrefill && l.phase != Lane::Phase::kMark)) { fifo_.erase(fifo_.begin() + long(i)); continue; }
        if (l.phase == Lane::Phase::kMark || !l.parked || l.reprep) { ++i; continue; }   // in the pipe, at its mark, re-preparing
        if (l.want_stop) { end_lane(l, "abort", false); continue; }                       // (drops it)
        if (l.chunk_at == 0 && l.repreps == 0) {
            if (m_.cache_peek(*l.rq) > l.reused) { l.reprep = true; cv_.notify_all(); ++i; continue; }
            bool wait_mark = false;   // an earlier entry is prefilling the same shared prefix: restore it once it is marked
            for (size_t j = 0; j < i && !wait_mark; ++j) {
                const Lane& e = lanes_[fifo_[j]];
                const uint32_t mk = e.mark_at;
                wait_mark = mk && l.rq->ids->size() > mk && std::equal(e.rq->ids->begin(), e.rq->ids->begin() + mk, l.rq->ids->begin());
            }
            if (wait_mark) { ++i; continue; }
        }
        if (piping_ && !pause_) { submit(l, h); cv_.notify_all(); }
        ++i;
    }
}

// A sampled id into the lane (mu held) -- the engine's emit rule: a stop id ends the lane with "stop" and is not committed;
// otherwise the id is committed (the repetition window, the outbox) and the budget may end it with "length". True while the
// lane goes on.
bool LanesServe::commit(Lane& l, int32_t id) {
    if (!l.rq->sp.ignore_eos && m_.is_stop(id)) { l.finish = "stop"; return false; }
    ++l.n_new; ++tokens_;
    l.hist.push_back(id);
    l.out.push_back(id);
    l.outbox.push_back(id);
    if (l.n_new >= l.max_new) { l.finish = "length"; return false; }
    return true;
}

// The lane's next step into the running pipe (mu held): its plan's next prefill piece, or a decode step [next]. A refused
// submit ends the request (the pipe refuses every step after a stage error).
void LanesServe::submit(Lane& l, uint32_t li) {
    l.parked = false; l.t_sub = Clock::now();
    std::string e;
    if (l.phase == Lane::Phase::kPrefill) {
        const auto [p0, t] = l.plan.chunks[l.chunk_at];
        l.pend = t; ++l.chunks;
        e = m_.pipe_submit(li, l.rq->ids->data() + p0, t, p0);
    } else {
        l.pend = 0;
        e = m_.pipe_submit(li, &l.next, 1, l.pos);
    }
    if (!e.empty()) end_lane(l, std::string("error: ") + m_.tag() + " lane " + std::to_string(li) + ": " + e, true);
}

int32_t LanesServe::sample_into(Lane& l, uint32_t li, bool first, std::string& err) {
    const LanesSampling& sp = l.rq->sp;
    const std::span<const int32_t> win = sp.penalties() ? lanes_penalty_window(l.hist, sp.repeat_window)
                                                        : std::span<const int32_t>{};
    return m_.sample(li, first, sp, win, l.rq->rng + l.n_new, err);
}

// The pipe's done callback (the last stage's thread): a prefill piece landed, or a decode step's row is sampled here.
void LanesServe::on_done(uint32_t li) {
    std::lock_guard<std::mutex> lk(mu_);
    struct Kick { LanesServe* s; ~Kick() { s->fifo_kick(); } } kick{this};   // (P4 B15; runs with mu still held)
    Lane& l = lanes_[li];
    if (!l.busy || l.phase == Lane::Phase::kDone) return;
    const auto t_cb = Clock::now();
    struct CbTime { Lane& l; Clock::time_point t0; ~CbTime() { l.cb_ms += ms_since(t0); } } cb_time{l, t_cb};
    if (!step_landed(l, li)) return;
    std::string err;
    const int32_t id = sample_into(l, li, false, err);
    step_sampled(l, li, id, err);
}

// P4 B14 rows mode: a group's done callback (the last stage's thread) -- per lane the head of on_done, ONE sample_rows over
// the group's decode rows, then per lane the tail of on_done, in group order.
void LanesServe::on_done_rows(std::span<const uint32_t> lanes) {
    std::lock_guard<std::mutex> lk(mu_);
    struct Kick { LanesServe* s; ~Kick() { s->fifo_kick(); } } kick{this};
    const auto t_cb = Clock::now();
    std::vector<uint32_t> dec;
    dec.reserve(lanes.size());
    for (uint32_t li : lanes) {
        Lane& l = lanes_[li];
        if (!l.busy || l.phase == Lane::Phase::kDone) continue;
        if (step_landed(l, li)) dec.push_back(li);
    }
    if (!dec.empty()) {
        std::vector<const LanesSampling*> sps;
        std::vector<std::span<const int32_t>> wins;
        std::vector<uint64_t> seeds;
        for (uint32_t li : dec) {   // (the windows view each lane's history: nothing commits before sample_rows returns)
            const Lane& l = lanes_[li];
            const LanesSampling& sp = l.rq->sp;
            sps.push_back(&sp);
            wins.push_back(sp.penalties() ? lanes_penalty_window(l.hist, sp.repeat_window) : std::span<const int32_t>{});
            seeds.push_back(l.rq->rng + l.n_new);
        }
        std::vector<int32_t> picks(dec.size(), 0);
        std::string err;
        try { err = m_.sample_rows(dec, sps, wins, seeds, picks); }
        catch (const std::exception& x) { err = std::string("threw: ") + x.what(); }
        for (size_t i = 0; i < dec.size(); ++i) step_sampled(lanes_[dec[i]], dec[i], picks[i], err);
    }
    const double ms = ms_since(t_cb);
    for (uint32_t li : lanes) lanes_[li].cb_ms += ms;
}

// A finished step's callback, its first half (mu held; the lane busy and not done): the trace line, and a prefill piece's
// whole handling. True = a decode step: its position advanced, its row is to be sampled.
bool LanesServe::step_landed(Lane& l, uint32_t li) {
    const double step = ms_since(l.t_sub);
    const bool prefill = l.phase == Lane::Phase::kPrefill;
    if (o_.trace) std::fprintf(stderr, "[%s step] lane %u %s %u row(s) at %u: %.1f ms, %u decoding\n", m_.tag(), li,
                               prefill ? "prefill" : "decode", prefill ? l.pend : 1u, prefill ? l.plan.chunks[l.chunk_at].first : l.pos,
                               step, decoding());
    if (prefill) {
        l.pos = l.plan.chunks[l.chunk_at].first + l.plan.chunks[l.chunk_at].second;
        ++l.chunk_at;
        if (l.mark_at && l.pos == l.mark_at && l.chunk_at < l.plan.chunks.size()) {   // P4 B15: the mark, in its request's turn
            if (l.want_stop) end_lane(l, "abort", false);
            else l.phase = Lane::Phase::kMark;
            cv_.notify_all();
            return false;
        }
        if (l.chunk_at < l.plan.chunks.size()) {
            if (l.want_stop) end_lane(l, "abort", false);               // the client left during the prompt
            else if (pause_ || stopping_) l.parked = true;
            else submit(l, li);
            cv_.notify_all();                                             // (the request thread's liveness probe)
            return false;
        }
        if (l.pos == l.rq->ids->size()) { m_.keep_logits(li); l.kept = true; }   // the prompt's last row: prompt_end samples it
        l.phase = Lane::Phase::kPromptReady;
        fifo_drop(li);                                                    // (P4 B15: the next prompt's pieces may start)
        cv_.notify_all();
        return false;
    }
    ++steps_; ++l.steps;
    step_ms_ = steps_ == 1 ? step : 0.9 * step_ms_ + 0.1 * step;
    l.pos += 1;
    return true;
}

// A decode step's callback, its second half (mu held): the sampled id (or the sampler's error) commits, ends the lane, or
// becomes its next step.
void LanesServe::step_sampled(Lane& l, uint32_t li, int32_t id, const std::string& err) {
    if (!err.empty()) { end_lane(l, std::string("error: ") + m_.tag() + " lane " + std::to_string(li) + " sample: " + err, true); return; }
    if (!commit(l, id)) { end_lane(l, l.finish, false); return; }
    l.next = id;
    if (l.want_stop) { end_lane(l, "abort", false); return; }
    if (pause_ || stopping_) { l.parked = true; cv_.notify_all(); return; }
    submit(l, li);
    cv_.notify_all();
}

// After a stage error the pipe refuses every step until it is stopped: every running request fails, and its lane is forgotten.
// A lane whose step failed may be half-stepped (card 0 advanced, card 1 not), so every lane that can have had a step in flight
// (busy, not done) is marked lost and reset in a turn by its own request before the lane is released for reuse.
void LanesServe::pipe_failed(const std::string& e) {
    std::fprintf(stderr, "[%s] the lane pipe failed: %s -- every running request fails and its lane is forgotten\n", m_.tag(), e.c_str());
    for (auto& l : lanes_) if (l.busy && l.phase != Lane::Phase::kDone) end_lane(l, std::string("error: ") + m_.tag() + " pipe: " + e, true);
}

// The serial turn (lk holds mu; returns with it held): after the current holder, the pipe PAUSED -- the callbacks park their
// lanes at their next completion and pipe_pause waits for the steps in flight. A stage error stops the pipe (the next release
// starts a new one) and fails every running request.
void LanesServe::take_turn(std::unique_lock<std::mutex>& lk) {
    ++turn_waiters_;
    cv_.wait(lk, [&] { return !turn_busy_; });
    --turn_waiters_;
    turn_busy_ = true;
    if (piping_) {
        pause_ = true;
        paused_valid_ = true; pause_t0_ = Clock::now();   // (P4 B15: the handover budget counts from here)
        lk.unlock();
        std::string e = m_.pipe_pause();
        if (!e.empty()) (void)m_.pipe_stop();   // (returns the same error and clears it)
        lk.lock();
        piping_ = false; pause_ = false;
        if (!e.empty()) pipe_failed(e);
    }
    ++turns_; turn_t0_ = Clock::now();
}

// Releases the turn (lk holds mu). Another request waiting for it takes it with the pipe still paused (no drain between the
// two turns); otherwise the parked lanes' steps go into the resumed pipe -- a new one when none is paused (the first request,
// or after a stage error).
void LanesServe::release_turn(std::unique_lock<std::mutex>& lk) {
    (void)lk;
    paused_ms_ += ms_since(turn_t0_);
    // P4 B15: hand over with the pipe still paused only within the budget, or when no decoding lane waits for it
    bool starve = false;
    if (turn_waiters_ > 0 && paused_valid_ && o_.handover_ms && ms_since(pause_t0_) >= double(o_.handover_ms))
        for (const auto& l : lanes_) if (l.busy && l.parked && l.phase == Lane::Phase::kDecode) { starve = true; break; }
    if (turn_waiters_ > 0 && !stopping_ && !starve) { ++handovers_; turn_busy_ = false; cv_.notify_all(); return; }
    if (starve) ++budget_releases_;
    std::vector<uint32_t> due;
    for (uint32_t li = 0; li < lanes_.size(); ++li) {
        Lane& l = lanes_[li];
        if (!l.busy || !l.parked) continue;
        if (l.want_stop) { end_lane(l, "abort", false); continue; }
        if (fifo_waiting(li)) continue;                                   // (the FIFO's lanes start through fifo_kick)
        due.push_back(li);
    }
    fifo_kick();   // (the pipe is not running: this only drops stale heads or flags a re-prepare)
    const bool head_go = !fifo_.empty();   // (fifo_kick below starts what may run)
    if ((!due.empty() || head_go) && !stopping_) {
        const std::string e = m_.pipe_paused() ? m_.pipe_resume()
                              : m_.rows() ? m_.pipe_start_rows([this](uint32_t li) { on_done(li); },
                                                               [this](std::span<const uint32_t> ls) { on_done_rows(ls); })
                                          : m_.pipe_start([this](uint32_t li) { on_done(li); });
        if (!e.empty()) {
            for (uint32_t li : due) end_lane(lanes_[li], std::string("error: ") + m_.tag() + " pipe: " + e, true);
            while (!fifo_.empty()) end_lane(lanes_[fifo_.front()], std::string("error: ") + m_.tag() + " pipe: " + e, true);
        } else {
            piping_ = true; paused_valid_ = false;
            for (uint32_t li : due) submit(lanes_[li], li);
            fifo_kick();
        }
    }
    turn_busy_ = false;
    cv_.notify_all();
}

// A stage error whose step ran no callback (a lone failing lane) is seen from the request threads' waits
void LanesServe::poll_pipe(std::unique_lock<std::mutex>& lk) {
    if (!piping_ || m_.pipe_error().empty()) return;
    take_turn(lk);
    release_turn(lk);
}

int LanesServe::choose(const std::vector<int32_t>& ids, uint32_t max_tokens) const {
    std::vector<LanesLaneView> v(lanes_.size());
    for (uint32_t li = 0; li < lanes_.size(); ++li) {
        const Lane& l = lanes_[li];
        v[li].idle = !l.busy; v[li].cap = l.cap; v[li].tick = l.tick;
        if (l.busy) continue;
        v[li].occupied = m_.occupied(li);
        v[li].last_end = m_.last_end(li);
        if (l.cap > ids.size()) v[li].match = m_.own_match(li, ids);
    }
    return lanes_choose_lane(v, uint32_t(ids.size()), max_tokens, std::max<uint32_t>(1u, m_.min_match_tokens()));
}

// The serial turn's device work (the turn held; lk holds mu, released while the work runs), on the serial worker. Meanwhile
// this thread probes the client every second (`alive`), and `gone` tells the work the client left. A throw comes back as the
// returned error.
std::string LanesServe::serial(std::unique_lock<std::mutex>& lk, const std::function<void()>& fn,
                               const std::function<bool()>& alive, std::atomic<bool>* gone) {
    lk.unlock();
    std::string err;
    {
        std::unique_lock<std::mutex> wl(wmu_);
        if (!worker_.joinable())
            worker_ = std::thread([this] {
                std::unique_lock<std::mutex> g(wmu_);
                for (;;) {
                    wcv_.wait(g, [&] { return wstop_ || wjob_; });
                    if (!wjob_) return;                                    // stopped (teardown: no request is left)
                    const std::function<void()>* job = wjob_;
                    g.unlock();
                    std::string e;
                    try { (*job)(); }
                    catch (const std::exception& x) { e = std::string("threw: ") + x.what(); }
                    catch (...) { e = "threw a non-std exception"; }
                    g.lock();
                    werr_ = std::move(e); wjob_ = nullptr; wdone_ = true;
                    wcv_.notify_all();
                }
            });
        wjob_ = &fn; wdone_ = false; werr_.clear();
        wcv_.notify_all();
        while (!wdone_) {
            if (wcv_.wait_for(wl, std::chrono::seconds(1)) != std::cv_status::timeout || !alive || !gone || gone->load()) continue;
            wl.unlock();
            const bool ok = alive();
            wl.lock();
            if (!ok) gone->store(true);
        }
        err = std::move(werr_); werr_.clear();
    }
    lk.lock();
    return err;
}

LanesResult LanesServe::run(const LanesRequest& rq, const LanesTokenFn& on_token) {
    LanesResult r;
    const std::vector<int32_t>& ids = *rq.ids;
    const uint32_t T = uint32_t(ids.size());
    r.prompt_tokens = T;
    uint32_t cap_max = 0;
    for (const auto& l : lanes_) cap_max = std::max(cap_max, l.cap);
    if (T == 0) { r.finish_reason = "length"; return r; }
    if (T >= cap_max) { r.finish_reason = "context_length_exceeded"; return r; }
    const auto t_req = Clock::now();
    Clock::time_point t_dec{};
    auto alive = [&] { return !on_token || on_token(std::string_view{}); };   // the server's liveness probe (an empty piece)
    auto fits = [&] { for (const auto& l : lanes_) if (!l.busy && l.cap > T) return true; return false; };

    std::unique_lock<std::mutex> lk(mu_);
    // 1. a lane, chosen under the serial turn; a prompt only the big lane holds waits for it
    int li = -1;
    for (;;) {
        while (!stopping_ && !fits()) {
            cv_.wait_for(lk, std::chrono::seconds(1));
            if (stopping_ || fits()) break;
            lk.unlock(); const bool ok = alive(); lk.lock();
            if (!ok) { r.finish_reason = "abort"; return r; }
        }
        if (stopping_) { r.finish_reason = "abort"; return r; }
        take_turn(lk);
        li = choose(ids, rq.sp.max_tokens);
        if (li >= 0) break;
        release_turn(lk);
    }
    const uint32_t lane = uint32_t(li);
    r.lane = lane;
    Lane& l = lanes_[lane];
    l.busy = true; l.phase = Lane::Phase::kPrefill; l.rq = &rq; l.finish.clear();
    l.want_stop = l.lost = l.parked = l.kept = false;
    l.outbox.clear(); l.out.clear(); l.pend = 0; l.next = -1; l.steps = l.chunks = 0; l.cb_ms = 0;
    l.hist.assign(ids.begin(), ids.end());
    l.max_new = lanes_reply_budget(T, rq.sp.max_tokens, l.cap); l.n_new = 0;   // (T < l.cap)
    uint32_t others = 0;                                               // lanes of other requests still running
    for (uint32_t k = 0; k < lanes_.size(); ++k) others += k != lane && lanes_[k].busy && lanes_[k].phase != Lane::Phase::kDone;
    const bool alone = others == 0 && turn_waiters_ == 0;
    auto release_lane = [&] {   // (mu held) the lane goes idle with its conversation; the LRU order follows the release
        l.busy = false; l.phase = Lane::Phase::kIdle; l.tick = ++tick_; l.rq = nullptr;
        l.outbox.clear(); l.hist.clear(); l.hist.shrink_to_fit();
        cv_.notify_all();
    };

    // 2. the prefix step (the turn held; the serial worker does the device work), then the plan. The prefill runs HERE when
    // every other lane is idle (the arch pipelines the cards over its chunks, until another request waits for the turn or the
    // client leaves) and otherwise through the lane pipe, beside the decoding lanes (P4 B15: one prompt's pieces at a time, in
    // arrival order, when o_.prefill_fifo). A plan's mark (the shared prefix's end) runs LanesModel::mark in a turn.
    std::string e, pe;
    uint32_t reused = 0;
    bool ready = false;                                                // the prompt is in: its first id is sampled below
    std::atomic<bool> gone{false};
    const std::function<bool()> stop = [&] {                           // (the worker asks between chunks)
        if (gone.load()) return true;
        std::lock_guard<std::mutex> g(mu_);
        return turn_waiters_ > 0 || stopping_;
    };
    int32_t first = -1;
    std::string se;                                                    // the first sample's error
    auto log_mark = [&](const std::string& me) {
        if (!me.empty()) std::fprintf(stderr, "[%s] lane %u shared-prefix mark at %u: %s\n", m_.tag(), lane, l.mark_at, me.c_str());
    };
    // on the serial worker: the prefix step, the plan and (alone) the prefill; `alone_now` = no other lane runs
    auto prep = [&](bool alone_now) {
        std::string source;
        pe = m_.prefix_prepare(lane, rq, reused, source);
        r.cache_source = source;
        if (!pe.empty()) return;
        l.reused = reused; l.pos = reused; l.chunk_at = 0;
        l.plan = m_.plan(rq, reused);
        l.mark_at = (l.plan.mark > reused && l.plan.mark < l.plan.Tp) ? l.plan.mark : 0;
        size_t kmark = l.plan.chunks.size();                            // the chunks up to and including the mark
        if (l.mark_at) {
            kmark = 0;
            for (size_t k = 0; k < l.plan.chunks.size() && !kmark; ++k)
                if (l.plan.chunks[k].first + l.plan.chunks[k].second == l.mark_at) kmark = k + 1;
            if (!kmark) {   // (an arch whose plan has no chunk ending at its mark: no mark rather than one at the wrong depth)
                std::fprintf(stderr, "[%s] lane %u: no prefill chunk ends at the mark %u; the shared prefix is not snapshotted\n",
                             m_.tag(), lane, l.mark_at);
                l.mark_at = 0; kmark = l.plan.chunks.size();
            }
        }
        if (!alone_now && o_.mix_chunk) {                              // (a capped piece computes other bits: off by default)
            std::vector<LanesChunk> cut;
            for (const auto& [p0, t] : l.plan.chunks)
                for (uint32_t q = 0; q < t; q += o_.mix_chunk) cut.emplace_back(p0 + q, std::min(o_.mix_chunk, t - q));
            l.plan.chunks.swap(cut);
        }
        if (alone_now || l.plan.chunks.empty()) {
            size_t done = 0;
            const std::span<const LanesChunk> all(l.plan.chunks);
            e = m_.prefill_serial(lane, rq, all.first(kmark), stop, done);
            if (e.empty() && !gone.load() && l.mark_at && done == kmark) {
                log_mark(m_.mark(lane, rq, l.mark_at));
                ++marks_;
                l.mark_at = 0;
                if (kmark < all.size() && !stop()) {
                    size_t d2 = 0;
                    e = m_.prefill_serial(lane, rq, all.subspan(kmark), stop, d2);
                    done += d2;
                }
            }
            l.chunk_at = uint32_t(done); l.chunks += uint32_t(done);
            if (done) l.pos = l.plan.chunks[done - 1].first + l.plan.chunks[done - 1].second;
            if (e.empty() && !gone.load() && l.chunk_at == l.plan.chunks.size()) {
                e = m_.prompt_end(lane, rq, l.plan.Tp, reused, false);
                if (e.empty()) l.pos = T;                              // the rows [Tp, T) are in: the first decode step's pos0
                if (e.empty()) { first = sample_into(l, lane, true, se); ready = true; }
            }
        }
    };
    // (mu held, the turn held) after prep: the lane ends, is ready for its first id, or waits for its pieces
    auto after_prep = [&](const std::string& we, bool initial) {
        r.cached_tokens = reused;
        if (!we.empty() && e.empty()) e = we;
        if (!pe.empty()) end_lane(l, "error: prefix cache: " + pe, true);
        else if (!e.empty()) end_lane(l, "error: " + e, true);
        else if (gone) end_lane(l, "abort", false);
        else if (ready) {
            fifo_drop(lane);
            r.prefill_ms = ms_since(t_req);
            t_dec = Clock::now();
            if (!se.empty()) end_lane(l, "error: sample: " + se, true);
            else if (!commit(l, first)) end_lane(l, l.finish, false);
            else { l.next = first; l.phase = Lane::Phase::kDecode; l.parked = true; }   // the release submits its first decode step
        } else {
            l.parked = true;                                           // the next prefill piece is due: the release submits it
            if (initial && o_.prefill_fifo && l.phase == Lane::Phase::kPrefill) fifo_.push_back(lane);
        }
    };
    const std::string we = serial(lk, [&] { prep(alone); }, alive, &gone);
    after_prep(we, true);
    const uint32_t serial_chunks = l.chunks;
    release_turn(lk);
    // 3. the rest of the prompt through the lane pipe; after each piece the server's liveness probe (a client that left ends
    // the request at the lane's next completion), and a stage error seen from here when no callback ran. P4 B15: the mark and
    // a re-prepare (the FIFO head whose cache_peek grew) each take a turn here.
    uint32_t seen = l.pos;
    for (;;) {
        if (l.phase == Lane::Phase::kMark) {
            take_turn(lk);
            if (l.phase == Lane::Phase::kMark) {
                const uint32_t at = l.pos;
                std::string me;
                const std::string wm = serial(lk, [&] { me = m_.mark(lane, rq, at); });
                log_mark(me.empty() ? wm : me);
                ++marks_;
                l.mark_at = 0;
                if (l.want_stop) end_lane(l, "abort", false);
                else { l.phase = Lane::Phase::kPrefill; l.parked = true; }   // (still the FIFO's head)
            }
            release_turn(lk);
            continue;
        }
        if (l.phase == Lane::Phase::kPrefill && l.reprep) {
            take_turn(lk);
            if (l.phase == Lane::Phase::kPrefill && l.reprep) {
                l.reprep = false; ++l.repreps; ++repreps_;
                uint32_t oth = 0;
                for (uint32_t k = 0; k < lanes_.size(); ++k) oth += k != lane && lanes_[k].busy && lanes_[k].phase != Lane::Phase::kDone;
                const bool alone_now = oth == 0 && turn_waiters_ == 0;
                e.clear(); pe.clear(); se.clear(); ready = false; first = -1;
                const std::string wr = serial(lk, [&] { prep(alone_now); }, alive, &gone);
                after_prep(wr, false);
            }
            release_turn(lk);
            seen = l.pos;
            continue;
        }
        if (l.phase != Lane::Phase::kPrefill) break;
        cv_.wait_for(lk, std::chrono::seconds(1));
        if (l.phase != Lane::Phase::kPrefill || l.reprep) continue;
        // (P4 B15) a prompt held in the FIFO behind another has no step in flight: probe its client each second, and a client
        // that left ends it here
        if (l.parked && fifo_waiting(lane) && on_token && !l.want_stop) {
            lk.unlock(); const bool ok = alive(); lk.lock();
            if (!ok && l.phase == Lane::Phase::kPrefill && l.parked && !l.reprep) { end_lane(l, "abort", false); continue; }
        }
        if (l.pos != seen && on_token && !l.want_stop) {
            seen = l.pos;
            lk.unlock(); const bool ok = alive(); lk.lock();
            if (!ok) l.want_stop = true;
        }
        poll_pipe(lk);
    }
    // 4. the prompt's end (the turn held): the arch's checkpoints and the rows past Tp, then the first id
    if (l.phase == Lane::Phase::kPromptReady) {
        take_turn(lk);
        if (l.phase == Lane::Phase::kPromptReady) {
            std::string e2;
            const bool kept = l.kept;
            const std::string we2 = serial(lk, [&] {
                e2 = m_.prompt_end(lane, rq, l.plan.Tp, reused, kept);
                if (e2.empty()) l.pos = T;
                if (e2.empty()) first = sample_into(l, lane, true, se);
            });
            if (!we2.empty() && e2.empty()) e2 = we2;
            if (!e2.empty()) end_lane(l, "error: " + e2, true);
            else if (!se.empty()) end_lane(l, "error: sample: " + se, true);
            else if (l.want_stop) end_lane(l, "abort", false);
            else {
                r.prefill_ms = ms_since(t_req);
                t_dec = Clock::now();
                if (!commit(l, first)) end_lane(l, l.finish, false);
                else { l.next = first; l.phase = Lane::Phase::kDecode; l.parked = true; }
            }
        }
        release_turn(lk);
    }
    // 5. the outcome: ids as they land in the outbox, the reply's text re-decoded whole (the engine's serial loop) and streamed
    // up to its last complete UTF-8 character; the server's callback declining cancels the lane at its next completion
    std::string text;
    size_t emitted = 0;
    bool aborted = false;
    std::vector<int32_t> got, all;
    for (;;) {
        cv_.wait_for(lk, std::chrono::seconds(1), [&] { return !l.outbox.empty() || l.phase == Lane::Phase::kDone; });
        got.swap(l.outbox);
        const bool done = l.phase == Lane::Phase::kDone;
        if (!got.empty()) {
            lk.unlock();
            if (!aborted) {
                all.insert(all.end(), got.begin(), got.end());
                text = m_.detok(all);
                const size_t safe = utf8_complete(text);
                if (safe > emitted && on_token) {
                    if (!on_token(std::string_view(text).substr(emitted, safe - emitted))) aborted = true;
                }
                if (safe > emitted) emitted = safe;
            }
            got.clear();
            lk.lock();
            if (aborted) l.want_stop = true;
        }
        if (done) break;
        poll_pipe(lk);
    }
    // 6. the finish
    r.completion_tokens = uint32_t(all.size());
    const std::string& f = l.finish;
    r.finish_reason = aborted ? "abort" : f == "stop" || f == "abort" || f.rfind("error:", 0) == 0 ? f : "length";
    if (t_dec.time_since_epoch().count()) r.decode_ms = ms_since(t_dec);
    std::fprintf(stderr, "[%s] lane %u: prompt %u (%u cached), prefill %.0f ms: %u chunk(s) in the serial turn, %u through the lane pipe; "
                         "%u tokens (reply cap %u) in %u steps; %s; %u lane(s) busy; callbacks %.0f ms; "
                         "%llu turns so far (%llu handed over), the pipe paused %.0f ms for them\n", m_.tag(), lane, T, r.cached_tokens,
                 r.prefill_ms, serial_chunks, l.chunks - serial_chunks, l.n_new, l.max_new, l.steps, r.finish_reason.c_str(), busy(), l.cb_ms,
                 (unsigned long long)turns_, (unsigned long long)handovers_, paused_ms_);
    // P4 B10: the arch's reply snapshot, in a turn (the lane is still this request's)
    if (rq.reply_cache && !l.lost && !l.out.empty() && (r.finish_reason == "stop" || r.finish_reason == "length") && !stopping_) {
        take_turn(lk);
        std::string e4;
        const std::string fin = r.finish_reason;
        const uint32_t pos = l.pos;
        const std::string we4 = serial(lk, [&] { e4 = m_.finish(lane, rq, l.out, fin, pos); });
        if (!we4.empty() && e4.empty()) e4 = we4;
        if (!e4.empty()) std::fprintf(stderr, "[%s] lane %u finish: %s\n", m_.tag(), lane, e4.c_str());
        release_turn(lk);
    }
    if (l.lost) {
        take_turn(lk);
        (void)serial(lk, [&] { if (auto e3 = m_.reset_lane(lane); !e3.empty()) std::fprintf(stderr, "[%s] reset lane %u: %s\n", m_.tag(), lane, e3.c_str()); });
        l.lost = false;
        release_turn(lk);
    }
    release_lane();
    lk.unlock();
    if (on_token && text.size() > emitted) on_token(std::string_view(text).substr(emitted));   // the serial loop's tail flush
    if (r.finish_reason.rfind("error:", 0) == 0) return r;
    r.text = std::move(text);
    return r;
}

std::string LanesServe::status_json() const {
    std::lock_guard<std::mutex> lk(mu_);
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "{\"lanes\":%zu,\"lanes_active\":%u,\"decoding\":%u,\"tokens\":%llu,\"step_ms\":%.2f,\"turns\":%llu,"
                  "\"paused_ms\":%.0f,\"handovers\":%llu,\"prefill_queue\":%zu,\"marks\":%llu,\"repreps\":%llu,"
                  "\"budget_releases\":%llu}",
                  lanes_.size(), busy(), decoding(), (unsigned long long)tokens_, step_ms_, (unsigned long long)turns_, paused_ms_,
                  (unsigned long long)handovers_, fifo_.size(), (unsigned long long)marks_.load(), (unsigned long long)repreps_,
                  (unsigned long long)budget_releases_);
    return buf;
}

}  // namespace ie
