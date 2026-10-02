// src/engine/lanes_serve.cpp -- `ie serve --parallel N` on request lanes, shared by every arch (P4 B8). See
// include/ie/lanes_serve.hpp. The protocol is DeepSeek-V4.1 B6b's (src/engine/ds41_engine.cpp, ds41_run_lanes and its serve_*
// helpers, gated 2026-09-26), with the arch's work behind LanesModel.
#include "ie/lanes_serve.hpp"

#include <algorithm>
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

// P4 B29: IE_LANES_TRACE=1 -- one stderr line per card step (a prefill piece's rows, or a decode step / group's rows), its time
// and the card's idle gap before it: where the swarm's card time goes. Off = the stages as given (no wrapper).
namespace {
struct CardTrace {
    std::mutex mu;
    Clock::time_point last[8]{};
    void line(uint32_t s, uint32_t G, uint32_t T, uint32_t lane, uint32_t pos0, Clock::time_point t0) {
        const Clock::time_point t1 = Clock::now();
        std::lock_guard<std::mutex> lk(mu);
        const double gap = s < 8 && last[s].time_since_epoch().count() ? std::chrono::duration<double, std::milli>(t0 - last[s]).count() : 0.0;
        if (s < 8) last[s] = t1;
        if (T > 1) std::fprintf(stderr, "[lanes card] card %u prefill %u rows lane %u at %u: %.1f ms (idle %.1f ms before)\n", s, T, lane,
                                pos0, std::chrono::duration<double, std::milli>(t1 - t0).count(), gap);
        else       std::fprintf(stderr, "[lanes card] card %u decode %u row(s): %.1f ms (idle %.1f ms before)\n", s, G,
                                std::chrono::duration<double, std::milli>(t1 - t0).count(), gap);
    }
};
bool lanes_trace_on() { static const bool on = [] { const char* v = std::getenv("IE_LANES_TRACE"); return v && *v == '1'; }(); return on; }
}  // namespace

std::string CardPipe::launch() {
    LanesModel::DoneFn* d = &done_;
    Glm5LanePipe::StageFn st = stage_;
    Glm5LanePipe::RowsStageFn rst = rows_stage_;
    if (lanes_trace_on()) {
        static CardTrace tr;
        st = [f = stage_](uint32_t s, const Glm5LanePipe::Step& x) {
            const auto t0 = Clock::now();
            std::string e = f(s, x);
            tr.line(s, 1, x.T, x.lane, x.pos0, t0);
            return e;
        };
        if (rows_stage_)
            rst = [f = rows_stage_](uint32_t s, std::span<const Glm5LanePipe::Step> xs, float* gw) {
                const auto t0 = Clock::now();
                std::string e = f(s, xs, gw);
                tr.line(s, uint32_t(xs.size()), 1, 0, 0, t0);
                return e;
            };
    }
    if (!rows_on_) return pipe_.start(n_stages_, st, [d](uint32_t lane, uint32_t, uint32_t) { (*d)(lane); }, idle_);
    LanesModel::RowsDoneFn* rd = &rows_done_;
    return pipe_.start_rows(n_stages_, st, rst, [d](uint32_t lane, uint32_t, uint32_t) { (*d)(lane); },
                            [rd](std::span<const uint32_t> lanes) { (*rd)(lanes); }, max_group_, group_cap_, idle_);
}

std::string CardPipe::submit(uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0, bool front) {
    // A fresh claim (the serve protocol submits one only for a lane it prepared in a turn: reset, restored or prefilled
    // there) at a position serial work moved syncs the pipe's record of it first, and a refused sync refuses the submit: a
    // lane with a step in flight, or one whose last step failed part-way and was not reset since (reset_lane). A callback's
    // resubmit is at the position the pipe committed, so it skips the sync and meets the pipe's own checks unchanged.
    if (pos0 != 0 && pipe_.lane_pos(lane) != pos0)
        if (auto e = pipe_.set_lane_pos(lane, pos0); !e.empty()) return "card pipe: position sync refused: " + e;
    return pipe_.submit(lane, ids, T, pos0, front);
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
    (void)m_.pipe_lookahead([this] { on_idle0(); });   // P4 B34 (an arch without the lane pipeline drops it)
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

// P4 B33 (see the header). The serial turn's prefill sees stopping_ through its `stop` (run_chunks asks it before each chunk);
// a request in any wait wakes on the notify and finds its lane done or take_turn refusing.
uint32_t LanesServe::abort_all() {
    std::lock_guard<std::mutex> lk(mu_);
    if (stopping_) return 0;
    stopping_ = true;
    uint32_t held = m_.abort();
    {
        std::lock_guard<std::mutex> wl(wmu_);   // (lock order mu_ -> wmu_: serial() never holds wmu_ while it takes mu_)
        if (wjob_) ++held;
    }
    for (Lane& l : lanes_)
        if (l.busy && l.phase != Lane::Phase::kDone) { l.want_stop = true; end_lane(l, "abort", false); }
    fifo_.clear();
    cv_.notify_all();
    return held;
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
// cache_peek never runs beside the serial worker's device work. P4 B34 (3) short-first: a short entry runs past the window
// too, and a long entry's next piece may wait for the short prompts (hold_long).
void LanesServe::fifo_kick() {
    constexpr size_t kWindow = 2;
    for (size_t i = 0; i < fifo_.size() && (i < kWindow || o_.short_first) && !stopping_;) {
        const uint32_t h = fifo_[i];
        Lane& l = lanes_[h];
        if (!l.busy || (l.phase != Lane::Phase::kPrefill && l.phase != Lane::Phase::kMark && l.phase != Lane::Phase::kSnap)) {
            fifo_.erase(fifo_.begin() + long(i));
            continue;
        }
        if (i >= kWindow && (l.phase != Lane::Phase::kPrefill || !short_lane(l))) { ++i; continue; }   // (P4 B34 (3))
        if (l.phase != Lane::Phase::kPrefill || !l.parked || l.reprep || l.preparing) { ++i; continue; }   // in the pipe, at its mark / snap, (re-)preparing
        if (l.want_stop) { end_lane(l, "abort", false); continue; }                       // (drops it)
        if (l.chunk_at == 0 && l.repreps == 0) {
            if (m_.cache_peek(*l.rq) > l.reused) { l.reprep = true; cv_.notify_all(); ++i; continue; }
            bool wait_mark = false;   // an earlier entry is prefilling the same shared prefix: restore it once it is marked
            for (size_t j = 0; j < i && !wait_mark; ++j) wait_mark = extends_mark(l, lanes_[fifo_[j]]);
            if (wait_mark) { ++i; continue; }
        }
        if (piping_ && !pause_ && !hold_long(l, h)) { submit(l, h); cv_.notify_all(); }
        ++i;
    }
}

// ---- P4 B34 (3): short-first (mu held throughout) ----------------------------------------------------------------------------

bool LanesServe::short_lane(const Lane& l) const {
    const uint32_t end = pipe_end(l);   // (P4 B39: the tail past plan.snap was prompt_end's; the rule counts the part before it)
    const uint32_t left = end > l.pos ? end - l.pos : 0u;
    return left <= std::max(2u * l.piece_max, o_.short_rows);
}

bool LanesServe::extends_mark(const Lane& l, const Lane& e) const {
    const uint32_t mk = e.mark_at;
    return mk && l.rq->ids->size() > mk && std::equal(e.rq->ids->begin(), e.rq->ids->begin() + mk, l.rq->ids->begin());
}

// Short work another lane has due: its pieces (in the pipe or parked for it), its mark, snapshot or prompt-end turn. A prompt
// that has not started and waits for l's own mark (the same shared prefix) does not count: l must reach that mark first. P4 B39:
// nor does a lane whose prepare is still running (no plan yet; with a drain-free prepare the pipe runs meanwhile).
bool LanesServe::shorts_pending(const Lane& l, uint32_t li) const {
    for (uint32_t k = 0; k < lanes_.size(); ++k) {
        const Lane& s = lanes_[k];
        if (k == li || !s.busy || s.preparing || !short_lane(s)) continue;
        if (s.phase == Lane::Phase::kPromptReady || s.phase == Lane::Phase::kMark || s.phase == Lane::Phase::kSnap) return true;
        if (s.phase != Lane::Phase::kPrefill) continue;
        if (s.parked && s.chunk_at == 0 && s.repreps == 0 && extends_mark(s, l)) continue;
        return true;
    }
    return false;
}

// A long prefilling lane's next piece is due: it waits (true; held, its caller parks it) while short work is due elsewhere,
// until the guard -- short_slots short pieces went in ahead of it, or short_wait_ms since its last piece landed. P4 B36 (C):
// first, while the decode quota holds it (so short-first's guard is not counted for a lane that waits for the decoders).
bool LanesServe::hold_long(Lane& l, uint32_t li) {
    if (quota_hold(l, li)) return true;
    if (!o_.short_first || l.phase != Lane::Phase::kPrefill || short_lane(l) || !shorts_pending(l, li)) return false;
    if (l.held && l.bypass >= o_.short_slots) { ++short_guard_; return false; }
    if (ms_since(l.t_land) >= double(o_.short_wait_ms)) { ++short_guard_; ++short_guard_wait_; return false; }
    if (!l.held) { l.held = true; l.bypass = 0; }
    return true;
}

// A short prompt's piece went into the pipe: one more slot ahead of every held lane; a held lane that saw short_slots of them
// gets its next piece now, behind this one (P4 B36 (C): unless the decode quota holds it; it then goes at the quota's release).
void LanesServe::short_piece_in(uint32_t li) {
    bool ahead_of = false;
    for (uint32_t k = 0; k < lanes_.size(); ++k) {
        Lane& h = lanes_[k];
        if (k == li || !h.held) continue;
        ahead_of = true;
        if (++h.bypass >= o_.short_slots && h.parked && !h.want_stop && piping_ && !pause_ && !stopping_ && !quota_hold(h, k)) {
            ++short_guard_; submit(h, k);
        }
    }
    if (ahead_of) ++short_bypass_;
}

// ---- P4 B36 (A): re-cut (mu held) ---------------------------------------------------------------------------------------------

// Options::recut: the lane's plan pieces from `from` on (none of them sent yet) become mix_chunk-row pieces, each plan chunk cut
// in order, once another lane is decoding or prefilling a short prompt (also one at its mark or prompt end). Once: no re-grow.
void LanesServe::maybe_recut(Lane& l, uint32_t li, size_t from) {
    if (!o_.recut || !o_.mix_chunk || l.recut || l.piece_max <= o_.mix_chunk || l.phase != Lane::Phase::kPrefill) return;
    // P4 B39 (5): a request waiting for its prepare turn counts as arriving -- a lone lane's remainder is released into the running
    // pipe before the arrivals are busy lanes (the drained prepare used to hold it until they were; the B39 gate's wave leader ran
    // its 3,275-row remainder as one unpipelined piece, +1.2 s)
    bool active = prepare_waiters_ > 0;
    for (uint32_t k = 0; k < lanes_.size() && !active; ++k) {
        const Lane& s = lanes_[k];
        if (k == li || !s.busy || s.preparing || s.phase == Lane::Phase::kDone) continue;   // (P4 B39: a lane still preparing has no plan)
        active = s.phase == Lane::Phase::kDecode || short_lane(s);
    }
    if (!active) return;
    std::vector<LanesChunk> cut(l.plan.chunks.begin(), l.plan.chunks.begin() + long(from));
    for (size_t i = from; i < l.plan.chunks.size(); ++i) {
        if (l.plan.snap && l.plan.chunks[i].first >= l.plan.snap) { cut.push_back(l.plan.chunks[i]); continue; }   // (P4 B39: the tail = prompt_end's piece, as cut)
        for (uint32_t q = 0; q < l.plan.chunks[i].second; q += o_.mix_chunk)
            cut.emplace_back(l.plan.chunks[i].first + q, std::min(o_.mix_chunk, l.plan.chunks[i].second - q));
    }
    std::fprintf(stderr, "[%s] lane %u re-cut at %u: %zu plan piece(s) left -> %zu of at most %u rows (another lane is decoding or "
                         "prefilling a short prompt)\n", m_.tag(), li, from < l.plan.chunks.size() ? l.plan.chunks[from].first : l.pos,
                 l.plan.chunks.size() - from, cut.size() - from, o_.mix_chunk);
    l.plan.chunks.swap(cut);
    l.recut = true; l.piece_max = o_.mix_chunk; ++recuts_;
}

// ---- P4 B36 (C): the decode quota (mu held throughout) ------------------------------------------------------------------------

// A long prompt: its whole pipe part above B34 (3)'s short threshold, so a lead stays long to its last (deepest) piece
bool LanesServe::long_prompt(const Lane& l) const {
    const uint32_t end = pipe_end(l);   // (P4 B39: as short_lane)
    const uint32_t part = end > l.reused ? end - l.reused : 0u;
    return part > std::max(2u * l.piece_max, o_.short_rows);
}

void LanesServe::quota_base(Lane& l) {
    if (!o_.decode_quota) return;
    l.qbase.resize(lanes_.size());
    for (size_t k = 0; k < lanes_.size(); ++k) l.qbase[k] = lanes_[k].dsteps;
}

// Options::decode_quota: a long prefilling lane owes the decoding lanes their steps -- one made fewer than decode_quota since
// the lane's latest piece landed (a lane that began decoding since counts its steps from then)
bool LanesServe::quota_due(const Lane& l, uint32_t li) const {
    if (!o_.decode_quota || l.phase != Lane::Phase::kPrefill || !long_prompt(l)) return false;
    for (uint32_t d = 0; d < lanes_.size(); ++d) {
        const Lane& x = lanes_[d];
        if (d == li || !x.busy || x.phase != Lane::Phase::kDecode) continue;
        if (x.dsteps - (d < l.qbase.size() ? l.qbase[d] : x.dsteps) < o_.decode_quota) return true;
    }
    return false;
}

// l's next piece is due: it waits (true) while quota_due, until the guard -- quota_max_ms since the wait began. The wait ends
// with the piece's submit; once the guard fired the piece goes regardless.
bool LanesServe::quota_hold(Lane& l, uint32_t li) {
    if (l.qfree || !quota_due(l, li)) return false;
    if (!l.qheld) { l.qheld = true; l.t_qhold = Clock::now(); ++quota_holds_; return true; }
    if (ms_since(l.t_qhold) >= double(o_.quota_max_ms)) { l.qfree = true; ++quota_guard_; return false; }
    return true;
}

// Decode steps landed and the decoders' next steps are in: a lane held for the quota whose decoders made their steps (or
// whose guard passed) takes its next piece, behind them on card 0
void LanesServe::quota_kick() {
    if (!o_.decode_quota || !piping_ || pause_ || stopping_) return;
    for (uint32_t k = 0; k < lanes_.size(); ++k) {
        Lane& h = lanes_[k];
        if (!h.busy || !h.qheld || !h.parked || h.want_stop || h.reprep || h.phase != Lane::Phase::kPrefill || hold_long(h, k)) continue;
        submit(h, k);
        cv_.notify_all();
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
    l.parked = false; l.held = false; l.bypass = 0; l.t_sub = Clock::now();
    l.qheld = l.qfree = false;   // (P4 B36 (C): its quota wait, if any, ends here)
    std::string e;
    const bool prefill = l.phase == Lane::Phase::kPrefill;
    if (prefill) {
        maybe_recut(l, li, l.chunk_at);   // (P4 B36 (A): nothing of the lane is in the pipe now)
        const auto [p0, t] = l.plan.chunks[l.chunk_at];
        l.pend = t; ++l.chunks;
        // P4 B39 (5): the prompt's tail past its snapshot boundary (<= kTailFront rows) goes ahead of the other lanes' queued
        // steps -- what the drained prompt-end turn did; one small piece per prompt, so no one starves
        const bool tail = l.plan.snap && p0 >= l.plan.snap && t <= kTailFront;
        e = tail ? m_.pipe_submit_front(li, l.rq->ids->data() + p0, t, p0) : m_.pipe_submit(li, l.rq->ids->data() + p0, t, p0);
    } else {
        l.pend = 0;
        e = m_.pipe_submit(li, &l.next, 1, l.pos);
    }
    if (!e.empty()) end_lane(l, std::string("error: ") + m_.tag() + " lane " + std::to_string(li) + ": " + e, true);
    else if (prefill && o_.short_first && short_lane(l)) short_piece_in(li);   // (P4 B34 (3))
}

// P4 B34: card 0 is idle (the pipe's idle hook: on card 0's stage thread after it finished a step, or on card 1's after a step
// there left a lane with one piece past card 0). The oldest prefilling lane whose one piece in flight has left card 0 sends its
// next piece now (the pipe checks that, and takes it only into an idle card 0), so card 0 runs it beside card 1's run of the
// piece before -- run_chunks' overlap, per lane: decode groups and every other piece keep their places. Not past the lane's
// mark or plan, and not while a turn is due (a lane at its prompt end or mark, a re-prepare: the turn's drain would wait for
// the lookahead on both cards). Not taking it changes nothing.
void LanesServe::on_idle0() {
    std::lock_guard<std::mutex> lk(mu_);
    if (!piping_ || pause_ || stopping_) return;
    for (const Lane& l : lanes_)
        if (l.busy && (l.phase == Lane::Phase::kPromptReady || l.phase == Lane::Phase::kMark || l.phase == Lane::Phase::kSnap || l.reprep)) return;
    auto offer = [&](uint32_t li) {
        Lane& l = lanes_[li];
        if (!l.busy || l.preparing || l.phase != Lane::Phase::kPrefill || l.parked || l.want_stop || l.ahead) return false;
        const size_t k = l.chunk_at;   // the piece in flight
        if (k + 1 >= l.plan.chunks.size()) return false;
        const uint32_t end_k = l.plan.chunks[k].first + l.plan.chunks[k].second;
        if ((l.mark_at && end_k == l.mark_at) || (l.snap_at && end_k == l.snap_at)) return false;   // (a turn at that depth first; P4 B39: the snapshot too)
        const bool shrt = o_.short_first && short_lane(l);
        if (o_.short_first && !shrt && shorts_pending(l, li)) return false;   // (P4 B34 (3): a long lane's next piece waits)
        if (quota_due(l, li)) return false;                                   // (P4 B36 (C): no lookahead owing the decoders)
        maybe_recut(l, li, k + 1);   // (P4 B36 (A): piece k is in the pipe, the rest not yet)
        const auto [p0, t] = l.plan.chunks[k + 1];
        bool taken = false;
        if (const std::string e = m_.pipe_submit_ahead(li, l.rq->ids->data() + p0, t, p0, taken); !e.empty())
            std::fprintf(stderr, "[%s] lane %u lookahead at %u: %s\n", m_.tag(), li, p0, e.c_str());
        if (!taken) return false;
        l.ahead = true; l.t_ahead = Clock::now(); ++l.chunks; ++lookaheads_;
        if (shrt) short_piece_in(li);
        return true;
    };
    for (uint32_t li : fifo_) if (offer(li)) return;   // (P4 B15: the FIFO's order first)
    for (uint32_t li = 0; li < lanes_.size(); ++li) if (offer(li)) return;
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
    quota_kick();   // (P4 B36 (C))
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
    // (P4 B36 (C): a prefill piece that landed in this group counts the decoders' quota from after the group's decode rows,
    // which ran beside it)
    if (o_.decode_quota && dec.size() < lanes.size())
        for (uint32_t li : lanes)
            if (lanes_[li].busy && lanes_[li].phase == Lane::Phase::kPrefill) quota_base(lanes_[li]);
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
        quota_kick();   // (P4 B36 (C): after the group's next steps went in)
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
        l.t_land = Clock::now();   // (P4 B34 (3): the short-first guard's wait counts from here)
        quota_base(l);             // (P4 B36 (C): the decoders' quota counts from here)
        if (l.ahead) {   // P4 B34: the next piece is in the pipe already (on_idle0) and is now the lane's step in flight. A
                         // lookahead never passes the mark or the plan's end, and the lane cannot end with a piece in flight:
                         // its mark, prompt end, a client that left and a pause all wait for that piece to land.
            l.ahead = false; l.t_sub = l.t_ahead; l.pend = l.plan.chunks[l.chunk_at].second;
            cv_.notify_all();                                             // (the request thread's liveness probe)
            return false;
        }
        if (l.mark_at && l.pos == l.mark_at && l.chunk_at < l.plan.chunks.size()) {   // P4 B15: the mark, in its request's turn
            if (l.want_stop) end_lane(l, "abort", false);
            else l.phase = Lane::Phase::kMark;
            cv_.notify_all();
            return false;
        }
        if (l.snap_at && l.pos == l.snap_at && l.chunk_at < l.plan.chunks.size()) {   // P4 B39: the snapshot, in its request's turn
            if (l.want_stop) end_lane(l, "abort", false);
            else l.phase = Lane::Phase::kSnap;
            cv_.notify_all();
            return false;
        }
        if (l.chunk_at < l.plan.chunks.size()) {
            if (l.want_stop) end_lane(l, "abort", false);               // the client left during the prompt
            else if (pause_ || stopping_ || hold_long(l, li)) l.parked = true;   // (P4 B34 (3): or held for the short prompts)
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
    ++steps_; ++l.steps; ++l.dsteps;
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
// starts a new one) and fails every running request. P4 B33: false once the engine is stopping (abort_all): the turn is not
// taken (or given back when the stop came during the drain) and the caller skips its device work.
// P4 B39: drain = false (the arch's serial_drains for the turn's kind) leaves the pipe running through the turn: no pause, no
// wait for the other lanes' steps; the caller waits for its own lane (lane_wait) before its serial work. A turn handed over
// with the pipe already paused runs paused either way (the chain resumes at its last release).
bool LanesServe::take_turn(std::unique_lock<std::mutex>& lk, bool drain, LanesModel::Turn kind) {
    // (the mark / snapshot of a lane goes before the prepares waiting with it; see the header. A boundary waiter is never blocked
    // by anything but the holder, so nothing starves: the prepares wait while such lanes wait, each of their turns is short)
    const bool boundary = kind == LanesModel::Turn::kMark || kind == LanesModel::Turn::kSnapshot;
    const bool yields = kind == LanesModel::Turn::kPrepare;
    ++turn_waiters_;
    if (boundary) ++boundary_waiters_;
    if (yields) ++prepare_waiters_;   // (P4 B39 (5): other lanes arriving -- maybe_recut reads it at a lone lane's release)
    cv_.wait(lk, [&] { return stopping_ || (!turn_busy_ && (!yields || boundary_waiters_ == 0)); });
    --turn_waiters_;
    if (yields) --prepare_waiters_;
    if (boundary) { --boundary_waiters_; if (!stopping_ && prepare_waiters_ > 0) ++boundary_first_; }   // (counts the prepares it went ahead of)
    if (stopping_) return false;
    turn_busy_ = true;
    if (piping_ && drain) drain_pipe(lk);
    turn_paused_ = !piping_;
    ++turns_; turn_t0_ = Clock::now();
    if (stopping_) { release_turn(lk); return false; }   // (P4 B33: the stop came while the pipe drained)
    return true;
}

// The pause (mu held by lk; released while pipe_pause waits): the callbacks park their lanes, pipe_pause waits for the steps in
// flight. P4 B29: the drain is always counted for /health.
void LanesServe::drain_pipe(std::unique_lock<std::mutex>& lk) {
    if (!piping_) return;
    pause_ = true;
    paused_valid_ = true; pause_t0_ = Clock::now();   // (P4 B15: the handover budget counts from here)
    const uint32_t dec = decoding(), act = busy();
    lk.unlock();
    std::string e = m_.pipe_pause();
    if (!e.empty()) (void)m_.pipe_stop();   // (returns the same error and clears it)
    lk.lock();
    const double dms = ms_since(pause_t0_);
    drain_ms_ += dms; drain_max_ms_ = std::max(drain_max_ms_, dms); ++drains_;
    if (lanes_trace_on())
        std::fprintf(stderr, "[%s drain] turn %llu: pipe_pause %.1f ms (%u lane(s) busy, %u decoding, %u waiting for the turn)\n",
                     m_.tag(), (unsigned long long)turns_ + 1, dms, act, dec, turn_waiters_);
    piping_ = false; pause_ = false; turn_paused_ = true;
    // (P4 B33: a stop during the drain -- the error is a step cut at a layer boundary, every lane already ended by abort_all)
    if (!e.empty() && !stopping_) pipe_failed(e);
}

// P4 B39 (the turn held, lk holds mu): with the pipe running through this turn, wait until lane li holds no step in it -- its
// last done callback may still hold its claim when this thread saw the step land. An arch without the per-lane wait gets the
// drain (once logged): a drain-free kind it did not back with pipe_wait_lane.
void LanesServe::lane_wait(std::unique_lock<std::mutex>& lk, uint32_t li) {
    if (!piping_ || stopping_) return;
    lk.unlock();
    const std::string e = m_.pipe_wait_lane(li);
    lk.lock();
    if (e.empty() || !piping_) return;
    static std::atomic<bool> said{false};
    if (!said.exchange(true))
        std::fprintf(stderr, "[%s] lane %u: %s -- the turn drains the pipe (serial_drains false without pipe_wait_lane)\n", m_.tag(), li, e.c_str());
    drain_pipe(lk);
}

// Releases the turn (lk holds mu). Another request waiting for it takes it with the pipe still paused (no drain between the
// two turns); otherwise the parked lanes' steps go into the resumed pipe -- a new one when none is paused (the first request,
// or after a stage error). P4 B39: when the pipe ran through the turn, the parked lanes that are due go into it here, before a
// handover too (nothing waits for the chain's end).
void LanesServe::release_turn(std::unique_lock<std::mutex>& lk) {
    (void)lk;
    if (turn_paused_) paused_ms_ += ms_since(turn_t0_);
    else ++turns_nodrain_;
    if (piping_ && !stopping_) {   // (P4 B39)
        for (uint32_t li = 0; li < lanes_.size(); ++li) {
            Lane& l = lanes_[li];
            if (!l.busy || !l.parked || l.preparing) continue;   // (a lane re-preparing on the worker has no pieces to send yet)
            if (l.want_stop) { end_lane(l, "abort", false); continue; }
            if (fifo_waiting(li) || hold_long(l, li)) continue;   // (the FIFO's lanes start through fifo_kick; a held lane stays parked)
            submit(l, li);
        }
        fifo_kick();
    }
    // P4 B15: hand over with the pipe still paused only within the budget, or when no decoding lane waits for it
    bool starve = false;
    if (turn_waiters_ > 0 && paused_valid_ && o_.handover_ms && ms_since(pause_t0_) >= double(o_.handover_ms))
        for (const auto& l : lanes_) if (l.busy && l.parked && l.phase == Lane::Phase::kDecode) { starve = true; break; }
    if (turn_waiters_ > 0 && !stopping_ && !starve) { ++handovers_; turn_busy_ = false; cv_.notify_all(); return; }
    if (starve) ++budget_releases_;
    if (!piping_) {
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
                for (uint32_t li : due) if (!hold_long(lanes_[li], li)) submit(lanes_[li], li);   // (P4 B34 (3): a held lane stays parked)
                fifo_kick();
            }
        }
    }
    turn_busy_ = false;
    cv_.notify_all();
}

// A stage error whose step ran no callback (a lone failing lane) is seen from the request threads' waits
void LanesServe::poll_pipe(std::unique_lock<std::mutex>& lk) {
    if (!piping_ || m_.pipe_error().empty()) return;
    if (take_turn(lk)) release_turn(lk);
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
    using Turn = LanesModel::Turn;
    for (;;) {
        while (!stopping_ && !fits()) {
            cv_.wait_for(lk, std::chrono::seconds(1));
            if (stopping_ || fits()) break;
            lk.unlock(); const bool ok = alive(); lk.lock();
            if (!ok) { r.finish_reason = "abort"; return r; }
        }
        if (stopping_ || !take_turn(lk, m_.serial_drains(Turn::kPrepare), Turn::kPrepare)) { r.finish_reason = "abort"; return r; }
        li = choose(ids, rq.sp.max_tokens);
        if (li >= 0) break;
        release_turn(lk);
    }
    const uint32_t lane = uint32_t(li);
    r.lane = lane;
    Lane& l = lanes_[lane];
    l.busy = true; l.phase = Lane::Phase::kPrefill; l.rq = &rq; l.finish.clear();
    l.want_stop = l.lost = l.parked = l.kept = l.ahead = l.held = l.recut = l.qheld = l.qfree = false; l.bypass = 0;
    l.outbox.clear(); l.out.clear(); l.pend = 0; l.next = -1; l.steps = l.chunks = 0; l.cb_ms = 0;
    // (P4 B39: the callbacks of the other lanes may read these while the prepare runs: a new prompt with no plan yet)
    l.plan = LanesPlan{}; l.pos = l.reused = l.chunk_at = l.mark_at = l.snap_at = l.piece_max = 0; l.preparing = true;
    l.hist.assign(ids.begin(), ids.end());
    l.max_new = lanes_reply_budget(T, rq.sp.max_tokens, l.cap); l.n_new = 0;   // (T < l.cap)
    uint32_t others = 0;                                               // lanes of other requests still running
    for (uint32_t k = 0; k < lanes_.size(); ++k) others += k != lane && lanes_[k].busy && lanes_[k].phase != Lane::Phase::kDone;
    const bool alone = others == 0 && turn_waiters_ == 0;
    auto release_lane = [&] {   // (mu held) the lane goes idle with its conversation; the LRU order follows the release
        l.busy = false; l.phase = Lane::Phase::kIdle; l.tick = ++tick_; l.rq = nullptr; l.preparing = false;
        l.outbox.clear(); l.hist.clear(); l.hist.shrink_to_fit();
        cv_.notify_all();
    };

    // 2. the prefix step (the turn held; the serial worker does the device work), then the plan. The prefill runs HERE when
    // every other lane is idle (the arch pipelines the cards over its chunks, until another request waits for the turn or the
    // client leaves) and otherwise through the lane pipe, beside the decoding lanes (P4 B15: one prompt's pieces at a time, in
    // arrival order, when o_.prefill_fifo). A plan's mark (the shared prefix's end) runs LanesModel::mark in a turn; P4 B39: its
    // snap (the conversation snapshot's depth) runs LanesModel::snapshot in a turn, the tail after it through the pipe.
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
    auto log_mark = [&](const std::string& me, uint32_t at) {
        if (!me.empty()) std::fprintf(stderr, "[%s] lane %u shared-prefix mark at %u: %s\n", m_.tag(), lane, at, me.c_str());
    };
    auto log_snap = [&](const std::string& me, uint32_t at) {
        if (!me.empty()) std::fprintf(stderr, "[%s] lane %u snapshot at %u: %s\n", m_.tag(), lane, at, me.c_str());
    };
    // P4 B39: what prep computes on the serial worker, committed to the lane under mu by after_prep (with a drain-free prepare
    // the other lanes' callbacks run meanwhile and read the lane's fields)
    struct Prep { LanesPlan plan; uint32_t mark_at = 0, snap_at = 0, piece_max = 0, chunk_at = 0, chunks = 0, pos = 0; } pp;
    // on the serial worker: the prefix step, the plan and (alone) the prefill; `alone_now` = no other lane runs
    auto prep = [&](bool alone_now) {
        pp = Prep{};
        std::string source;
        pe = m_.prefix_prepare(lane, rq, reused, source);
        r.cache_source = source;
        if (!pe.empty()) return;
        pp.pos = reused;
        pp.plan = m_.plan(rq, reused);
        LanesPlan& plan = pp.plan;
        pp.mark_at = (plan.mark > reused && plan.mark < plan.Tp) ? plan.mark : 0;
        pp.snap_at = (plan.snap > reused && plan.snap < plan.Tp) ? plan.snap : 0;
        // the chunk index just past a boundary (0 = no chunk ends there)
        auto ends_at = [&](uint32_t at) {
            for (size_t k = 0; k < plan.chunks.size(); ++k)
                if (plan.chunks[k].first + plan.chunks[k].second == at) return k + 1;
            return size_t(0);
        };
        size_t kmark = plan.chunks.size(), ksnap = plan.chunks.size();   // the chunks up to and including the mark / the snap
        if (pp.mark_at) {
            kmark = ends_at(pp.mark_at);
            if (!kmark) {   // (an arch whose plan has no chunk ending at its mark: no mark rather than one at the wrong depth)
                std::fprintf(stderr, "[%s] lane %u: no prefill chunk ends at the mark %u; the shared prefix is not snapshotted\n",
                             m_.tag(), lane, pp.mark_at);
                pp.mark_at = 0; kmark = plan.chunks.size();
            }
        }
        if (pp.snap_at) {
            ksnap = ends_at(pp.snap_at);
            if (!ksnap) {
                std::fprintf(stderr, "[%s] lane %u: no prefill chunk ends at the snapshot boundary %u; the conversation is not snapshotted\n",
                             m_.tag(), lane, pp.snap_at);
                pp.snap_at = 0; ksnap = plan.chunks.size();
            }
        }
        if (!alone_now && o_.mix_chunk) {                              // (a capped piece computes other bits: off by default)
            std::vector<LanesChunk> cut;
            for (const auto& [p0, t] : plan.chunks) {
                if (plan.snap && p0 >= plan.snap) { cut.emplace_back(p0, t); continue; }   // (P4 B39: the tail = prompt_end's piece, as cut)
                for (uint32_t q = 0; q < t; q += o_.mix_chunk) cut.emplace_back(p0 + q, std::min(o_.mix_chunk, t - q));
            }
            plan.chunks.swap(cut);
            if (pp.mark_at) kmark = ends_at(pp.mark_at);
            if (pp.snap_at) ksnap = ends_at(pp.snap_at);
        }
        pp.piece_max = 0;                                              // (P4 B34 (3): the short rule's piece size; the tail aside)
        for (const auto& c : plan.chunks) if (!plan.snap || c.first < plan.snap) pp.piece_max = std::max(pp.piece_max, c.second);
        if (alone_now || plan.chunks.empty()) {
            // the chunks in segments: up to the mark, up to the snap, the rest -- the arch's hook after a boundary's segment ran
            // whole, then on only while no one waits for the turn (`stop`)
            size_t done = 0, from = 0;
            const std::span<const LanesChunk> all(plan.chunks);
            struct Seg { size_t end; int kind; };   // kind: 0 = the mark, 1 = the snap, 2 = the end
            std::vector<Seg> segs;
            if (pp.mark_at) segs.push_back({kmark, 0});
            if (pp.snap_at) segs.push_back({ksnap, 1});
            segs.push_back({all.size(), 2});
            for (const Seg& sg : segs) {
                if (from < sg.end) {
                    size_t d = 0;
                    e = m_.prefill_serial(lane, rq, all.subspan(from, sg.end - from), stop, d);
                    done += d;
                    if (!e.empty() || gone.load() || done != sg.end) break;
                    from = sg.end;
                }
                if (sg.kind == 0) { log_mark(m_.mark(lane, rq, pp.mark_at), pp.mark_at); ++marks_; pp.mark_at = 0; }
                if (sg.kind == 1) { log_snap(m_.snapshot(lane, rq, pp.snap_at), pp.snap_at); pp.snap_at = 0; }
                if (sg.kind != 2 && stop()) break;
            }
            pp.chunk_at = uint32_t(done); pp.chunks = uint32_t(done);
            if (done) pp.pos = plan.chunks[done - 1].first + plan.chunks[done - 1].second;
            if (e.empty() && !gone.load() && pp.chunk_at == plan.chunks.size()) {
                e = m_.prompt_end(lane, rq, plan.Tp, reused, false);
                if (e.empty()) pp.pos = T;                             // the rows [Tp, T) are in: the first decode step's pos0
                if (e.empty()) { first = sample_into(l, lane, true, se); ready = true; }
            }
        }
    };
    // (mu held, the turn held) after prep: the lane ends, is ready for its first id, or waits for its pieces
    auto after_prep = [&](const std::string& we, bool initial) {
        r.cached_tokens = reused;
        l.preparing = false;
        if (l.phase == Lane::Phase::kDone) return;                     // (P4 B33: abort_all ended it while the turn ran)
        l.reused = reused; l.pos = pp.pos; l.chunk_at = pp.chunk_at; l.chunks += pp.chunks;
        l.plan = std::move(pp.plan); l.mark_at = pp.mark_at; l.snap_at = pp.snap_at; l.piece_max = pp.piece_max;
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
            l.t_land = Clock::now();                                   // (P4 B34 (3): the short-first guard's wait)
            quota_base(l);                                             // (P4 B36 (C): the decoders' quota)
            if (initial && o_.prefill_fifo && l.phase == Lane::Phase::kPrefill) fifo_.push_back(lane);
        }
    };
    lane_wait(lk, lane);   // (P4 B39: a drain-free turn: the lane's previous request's last callback may still hold its claim)
    const std::string we = serial(lk, [&] { prep(alone); }, alive, &gone);
    after_prep(we, true);
    const uint32_t serial_chunks = l.chunks;
    release_turn(lk);
    // 3. the rest of the prompt through the lane pipe; after each piece the server's liveness probe (a client that left ends
    // the request at the lane's next completion), and a stage error seen from here when no callback ran. P4 B15: the mark and
    // a re-prepare (the FIFO head whose cache_peek grew) each take a turn here; P4 B39: the snapshot too.
    uint32_t seen = l.pos;
    for (;;) {
        if (l.phase == Lane::Phase::kMark || l.phase == Lane::Phase::kSnap) {
            const bool snap = l.phase == Lane::Phase::kSnap;
            const Turn kind = snap ? Turn::kSnapshot : Turn::kMark;
            if (!take_turn(lk, m_.serial_drains(kind), kind)) { end_lane(l, "abort", false); continue; }   // (P4 B33: stopping)
            if (l.phase == (snap ? Lane::Phase::kSnap : Lane::Phase::kMark)) {
                lane_wait(lk, lane);
                const uint32_t at = l.pos;
                std::string me;
                const std::string wm = serial(lk, [&] { me = snap ? m_.snapshot(lane, rq, at) : m_.mark(lane, rq, at); });
                if (snap) { log_snap(me.empty() ? wm : me, at); l.snap_at = 0; }
                else      { log_mark(me.empty() ? wm : me, at); ++marks_; l.mark_at = 0; }
                if (l.phase != (snap ? Lane::Phase::kSnap : Lane::Phase::kMark)) {}   // (P4 B33: abort_all ended it meanwhile)
                else if (l.want_stop) end_lane(l, "abort", false);
                else { l.phase = Lane::Phase::kPrefill; l.parked = true; }   // (still the FIFO's head)
            }
            release_turn(lk);
            continue;
        }
        if (l.phase == Lane::Phase::kPrefill && l.reprep) {
            if (!take_turn(lk, m_.serial_drains(Turn::kPrepare), Turn::kPrepare)) { end_lane(l, "abort", false); continue; }   // (P4 B33: stopping)
            if (l.phase == Lane::Phase::kPrefill && l.reprep) {
                l.reprep = false; ++l.repreps; ++repreps_;
                uint32_t oth = 0;
                for (uint32_t k = 0; k < lanes_.size(); ++k) oth += k != lane && lanes_[k].busy && lanes_[k].phase != Lane::Phase::kDone;
                const bool alone_now = oth == 0 && turn_waiters_ == 0;
                e.clear(); pe.clear(); se.clear(); ready = false; first = -1;
                l.preparing = true;
                lane_wait(lk, lane);
                const std::string wr = serial(lk, [&] { prep(alone_now); }, alive, &gone);
                after_prep(wr, false);
            }
            release_turn(lk);
            seen = l.pos;
            continue;
        }
        if (l.phase != Lane::Phase::kPrefill) break;
        int64_t wait_ms = 1000;   // (P4 B36 (C): a lane held for the decode quota wakes at its guard's deadline)
        if (l.qheld && !l.qfree) wait_ms = std::clamp<int64_t>(int64_t(double(o_.quota_max_ms) - ms_since(l.t_qhold)) + 1, 1, 1000);
        cv_.wait_for(lk, std::chrono::milliseconds(wait_ms));
        if (l.phase != Lane::Phase::kPrefill || l.reprep) continue;
        // P4 B34 (3): a lane held for the short prompts takes its next piece once they are done or the guard's wait passed
        // (P4 B36 (C): and one held for the decode quota once the decoders made their steps or its guard's wait passed)
        if ((l.held || l.qheld) && l.parked && !l.want_stop && piping_ && !pause_ && !stopping_ && !hold_long(l, lane)) {
            submit(l, lane); cv_.notify_all();
        }
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
        const bool turn = take_turn(lk, m_.serial_drains(Turn::kPromptEnd), Turn::kPromptEnd);
        if (!turn) end_lane(l, "abort", false);                           // (P4 B33: stopping)
        if (turn && l.phase == Lane::Phase::kPromptReady) {
            lane_wait(lk, lane);
            std::string e2;
            const bool kept = l.kept;
            const std::string we2 = serial(lk, [&] {
                e2 = m_.prompt_end(lane, rq, l.plan.Tp, reused, kept);
                if (e2.empty()) l.pos = T;
                if (e2.empty()) first = sample_into(l, lane, true, se);
            });
            if (!we2.empty() && e2.empty()) e2 = we2;
            if (l.phase == Lane::Phase::kDone) {}                          // (P4 B33: abort_all ended it meanwhile)
            else if (!e2.empty()) end_lane(l, "error: " + e2, true);
            else if (!se.empty()) end_lane(l, "error: sample: " + se, true);
            else if (l.want_stop) end_lane(l, "abort", false);
            else {
                r.prefill_ms = ms_since(t_req);
                t_dec = Clock::now();
                if (!commit(l, first)) end_lane(l, l.finish, false);
                else { l.next = first; l.phase = Lane::Phase::kDecode; l.parked = true; }
            }
        }
        if (turn) release_turn(lk);
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
    // P4 B27: where the answer starts = after the first generated </think> (a special token detok drops; the prefix's
    // decode is a byte prefix of the whole reply's)
    if (rq.think_close >= 0)
        for (size_t k = 0; k < all.size(); ++k)
            if (all[k] == rq.think_close) { r.answer_start = m_.detok(std::span<const int32_t>(all).first(k + 1)).size(); break; }
    const std::string& f = l.finish;
    r.finish_reason = aborted ? "abort" : f == "stop" || f == "abort" || f.rfind("error:", 0) == 0 ? f : "length";
    if (t_dec.time_since_epoch().count()) r.decode_ms = ms_since(t_dec);
    std::fprintf(stderr, "[%s] lane %u: prompt %u (%u cached), prefill %.0f ms: %u chunk(s) in the serial turn, %u through the lane pipe; "
                         "%u tokens (reply cap %u) in %u steps; %s; %u lane(s) busy; callbacks %.0f ms; "
                         "%llu turns so far (%llu handed over, %llu ran beside the pipe), the pipe paused %.0f ms for them\n", m_.tag(), lane, T,
                 r.cached_tokens, r.prefill_ms, serial_chunks, l.chunks - serial_chunks, l.n_new, l.max_new, l.steps, r.finish_reason.c_str(), busy(),
                 l.cb_ms, (unsigned long long)turns_, (unsigned long long)handovers_, (unsigned long long)turns_nodrain_, paused_ms_);
    // P4 B10: the arch's reply snapshot, in a turn (the lane is still this request's)
    const Turn fkind = r.finish_reason == "stop" ? Turn::kFinishStop : Turn::kFinishLength;
    if (rq.reply_cache && !l.lost && !l.out.empty() && (r.finish_reason == "stop" || r.finish_reason == "length") && !stopping_ &&
        take_turn(lk, m_.serial_drains(fkind), fkind)) {
        lane_wait(lk, lane);
        std::string e4;
        const std::string fin = r.finish_reason;
        const uint32_t pos = l.pos;
        const std::string we4 = serial(lk, [&] { e4 = m_.finish(lane, rq, l.out, fin, pos); });
        if (!we4.empty() && e4.empty()) e4 = we4;
        if (!e4.empty()) std::fprintf(stderr, "[%s] lane %u finish: %s\n", m_.tag(), lane, e4.c_str());
        release_turn(lk);
    }
    if (l.lost && take_turn(lk, m_.serial_drains(Turn::kReset), Turn::kReset)) {   // (P4 B33: stopping skips the reset: the process ends and the lane with it)
        lane_wait(lk, lane);
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
    const std::string pipe = m_.pipe_stats();   // (P4 B42: the arch's pipe counters, before our lock: the pipe has its own)
    std::lock_guard<std::mutex> lk(mu_);
    char buf[1280];
    std::snprintf(buf, sizeof buf,
                  "{\"lanes\":%zu,\"lanes_active\":%u,\"decoding\":%u,\"tokens\":%llu,\"step_ms\":%.2f,\"turns\":%llu,"
                  "\"paused_ms\":%.0f,\"handovers\":%llu,\"prefill_queue\":%zu,\"marks\":%llu,\"repreps\":%llu,"
                  "\"budget_releases\":%llu,\"drains\":%llu,\"drain_ms\":%.0f,\"drain_max_ms\":%.0f,\"snapshots\":%llu,"
                  "\"lookaheads\":%llu,\"short_bypass\":%llu,\"short_guard\":%llu,\"short_guard_wait\":%llu,\"recuts\":%llu,"
                  "\"quota_holds\":%llu,\"quota_guard\":%llu,\"turns_nodrain\":%llu,\"turns_boundary_first\":%llu}",
                  lanes_.size(), busy(), decoding(), (unsigned long long)tokens_, step_ms_, (unsigned long long)turns_, paused_ms_,
                  (unsigned long long)handovers_, fifo_.size(), (unsigned long long)marks_.load(), (unsigned long long)repreps_,
                  (unsigned long long)budget_releases_, (unsigned long long)drains_, drain_ms_, drain_max_ms_,
                  (unsigned long long)m_.snapshots(), (unsigned long long)lookaheads_, (unsigned long long)short_bypass_,
                  (unsigned long long)short_guard_, (unsigned long long)short_guard_wait_, (unsigned long long)recuts_,
                  (unsigned long long)quota_holds_, (unsigned long long)quota_guard_, (unsigned long long)turns_nodrain_,
                  (unsigned long long)boundary_first_);
    std::string js(buf);
    if (!pipe.empty()) { js.pop_back(); js += ",\"pipe\":" + pipe + "}"; }
    return js;
}

}  // namespace ie
