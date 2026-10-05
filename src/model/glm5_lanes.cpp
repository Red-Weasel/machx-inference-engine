// src/model/glm5_lanes.cpp -- GLM-5.3-Flash request lanes' card pipe (P4 B7). See include/ie/glm5_lanes.hpp.
#include "ie/glm5_lanes.hpp"

#include <algorithm>
#include <exception>
#include <pthread.h>

namespace ie {

Glm5LanePipe::Glm5LanePipe(uint32_t n_lanes, uint32_t max_T, uint64_t wide_row, bool ahead)
    : lanes_(std::max<uint32_t>(n_lanes, 1)), max_T_(std::max<uint32_t>(max_T, 1)), wide_row_(wide_row), ahead_(ahead) {
    const size_t slots = ahead ? 2 : 1;   // (P4 B34: a lookahead's buffers)
    for (Lane& ln : lanes_) {   // value-initialised: every page is touched here, before the caller loads a model
        ln.ids.assign(slots * max_T_, 0);
        ln.wide.assign(slots * size_t(max_T_) * wide_row_, 0.f);
    }
}

// (mu held) one of the lane's claims ends: a step dropped (a stage error, a cancel) or its callback returned without a
// resubmit. P4 B34: the other step, if any, is the lane's step in flight now.
void Glm5LanePipe::drop_claim(Lane& ln, uint32_t slot) {
    if (--ln.nfl == 0) --busy_;
    else if (slot == ln.cur) ln.cur ^= 1u;
}

Glm5LanePipe::~Glm5LanePipe() { stop(); }

uint64_t Glm5LanePipe::host_bytes() const {
    uint64_t b = 0;
    for (const Lane& ln : lanes_) b += ln.ids.size() * sizeof(int32_t) + ln.wide.size() * sizeof(float);
    for (const Group& g : groups_) b += g.wide.size() * sizeof(float);   // rows mode's group buffers (made by start_rows)
    return b;
}

std::string Glm5LanePipe::start(uint32_t n_stages, StageFn stage, DoneFn done, IdleFn idle) {
    if (running()) return "glm5 lane pipe: already running";
    if (n_stages == 0 || !stage) return "glm5 lane pipe: no stages";
    {
        std::lock_guard<std::mutex> lk(mu_);
        q_.assign(n_stages, {});
        busy_ = 0; stop_ = false; err_.clear(); steps_ = 0;
        running_ = 0; cancel_ = false;   // (P4 B33)
        for (Lane& ln : lanes_) { ln.nfl = 0; ln.at0 = false; ln.in_cb = false; ln.resub = false; }
        stage_ = std::move(stage);
        done_ = std::move(done);
        idle_ = std::move(idle); s0_busy_ = false;
    }
    for (uint32_t s = 0; s < n_stages; ++s) {
        th_.emplace_back([this, s] { stage_loop(s); });
        const std::string name = "g5pipe-s" + std::to_string(s);
        pthread_setname_np(th_.back().native_handle(), name.c_str());
    }
    return {};
}

std::string Glm5LanePipe::start_rows(uint32_t n_stages, StageFn stage, RowsStageFn rows_stage, DoneFn done, RowsDoneFn rows_done,
                                     uint32_t max_group, uint32_t group_cap, IdleFn idle) {
    if (running()) return "glm5 lane pipe: already running";
    if (n_stages == 0 || !stage || !rows_stage) return "glm5 lane pipe: no stages";
    if (max_group < 1 || max_group > 16) return "glm5 lane pipe: a group takes 1..16 rows, not " + std::to_string(max_group);
    {
        std::lock_guard<std::mutex> lk(mu_);
        q_.assign(n_stages, {});
        gq_.assign(n_stages, {});
        busy_ = 0; stop_ = false; err_.clear(); steps_ = 0; pgate_ = false;
        running_ = 0; cancel_ = false;   // (P4 B33)
        for (Lane& ln : lanes_) { ln.nfl = 0; ln.at0 = false; ln.in_cb = false; ln.resub = false; }
        idle_ = std::move(idle); s0_busy_ = false;
        if (groups_.size() != n_groups() || max_group_ != max_group) make_groups(max_group);   // (once, not per resume)
        gfree_.clear();
        for (uint32_t i = uint32_t(groups_.size()); i-- > 0;) gfree_.push_back(i);
        if (gsizes_.size() != 17) gsizes_.assign(17, 0);   // cumulative over resumes
        for (Group& g : groups_) g.decode = false;           // (P4 B42: nothing in flight at a start)
        dec_inflight_ = 0; last_dec_running_ = false; last_gi_ = UINT32_MAX;
        max_group_ = max_group; group_cap_ = group_cap; rows_ = true;
        stage_ = std::move(stage); rows_stage_ = std::move(rows_stage);
        done_ = std::move(done); rows_done_ = std::move(rows_done);
    }
    for (uint32_t s = 0; s < n_stages; ++s) {
        th_.emplace_back([this, s] { stage_loop_rows(s); });
        const std::string name = "g5rows-s" + std::to_string(s);
        pthread_setname_np(th_.back().native_handle(), name.c_str());
    }
    return {};
}

// The rows mode's group pool, its host buffers allocated AND touched (B14 gate #5: prepare_rows makes them at load, so the load
// line's host bytes and the pinning floor count them).
void Glm5LanePipe::make_groups(uint32_t max_group) {
    groups_.assign(n_groups(), {});
    for (Group& g : groups_) {
        g.lanes.reserve(max_group);
        g.slots.reserve(max_group);
        g.steps.reserve(max_group);
        g.wide.assign(size_t(max_group) * wide_row_, 0.f);
    }
    max_group_ = max_group;
}

void Glm5LanePipe::prepare_rows(uint32_t max_group) {
    std::lock_guard<std::mutex> lk(mu_);
    if (running() || max_group < 1 || max_group > 16) return;
    make_groups(max_group);
}

bool Glm5LanePipe::rows_mode() const {
    std::lock_guard<std::mutex> lk(mu_);
    return rows_;
}

std::vector<uint64_t> Glm5LanePipe::group_sizes() const {
    std::lock_guard<std::mutex> lk(mu_);
    return gsizes_;
}

// P4 B42 (see the header)
void Glm5LanePipe::set_regroup(uint32_t wait_us) {
    std::lock_guard<std::mutex> lk(mu_);
    regroup_ = wait_us > 0;
    regroup_wait_ = std::chrono::microseconds(wait_us);
}

Glm5LanePipe::RegroupStats Glm5LanePipe::regroup_stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    return rg_;
}

// (mu held) P4 B42: every group leaves the pipe through here (its callback returned, a stage error, a cancel): the pool gets
// it back and the regroup's counts of decode groups in flight / the one at the last stage follow
void Glm5LanePipe::free_group(uint32_t gi) {
    Group& g = groups_[gi];
    if (g.decode) { --dec_inflight_; g.decode = false; }
    if (gi == last_gi_) { last_gi_ = UINT32_MAX; last_dec_running_ = false; }
    gfree_.push_back(gi);
}

std::string Glm5LanePipe::submit(uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0, bool front) {
    if (lane >= lanes_.size()) return "glm5 lane pipe: lane " + std::to_string(lane) + " of " + std::to_string(lanes_.size());
    Lane& ln = lanes_[lane];
    const std::string who = "glm5 lane pipe: lane " + std::to_string(lane);
    bool fresh = false, failed = false;
    uint32_t n_pos = 0, sl = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (q_.empty() || stop_) return "glm5 lane pipe: not running";
        if (cancel_) return "glm5 lane pipe: cancelled (the engine is stopping)";   // (P4 B33)
        if (!err_.empty()) return "glm5 lane pipe: a stage failed: " + err_;
        if (ln.nfl) {
            if (!ln.in_cb || ln.cb_tid != std::this_thread::get_id()) return who + " has a step in flight";
            if (ln.resub) return who + " was already resubmitted from its callback";
            if (ln.nfl > 1) return who + " has its next step in flight already (a lookahead)";   // (P4 B34)
        } else {
            ln.nfl = 1; ++busy_; fresh = true;
        }
        n_pos = ln.n_pos;
        failed = ln.failed;
        sl = ln.cur;      // a fresh claim: the lane's buffers are free; a resubmit: its finished step's slot
        ln.at0 = true;    // (P4 B34: no lookahead before this step is queued and has left stage 0)
    }
    auto refuse = [&](const std::string& e) {   // a refused step gives a fresh claim back (a callback's lane stays claimed)
        {
            std::lock_guard<std::mutex> lk(mu_);
            ln.at0 = false;
            if (fresh) { ln.nfl = 0; --busy_; }
        }
        if (fresh) cv_.notify_all();
        return e;
    };
    if (T == 0 || T > max_T_) return refuse(who + ": " + std::to_string(T) + " rows, the pipe takes 1.." + std::to_string(max_T_));
    if (!ids) return refuse(who + ": no ids");
    if (pos0 != 0 && failed)
        return refuse(who + ": its last step failed part-way, so only a new sequence (pos0 0) may follow");
    if (pos0 != 0 && pos0 != n_pos)
        return refuse(who + ": pos0 " + std::to_string(pos0) + " is neither 0 (a new sequence) nor its position " + std::to_string(n_pos));
    std::copy_n(ids, T, slot_ids(ln, sl));   // the claim makes this slot ours: no step of this lane uses it
    ln.T[sl] = T; ln.pos0[sl] = pos0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (pos0 == 0) ln.failed = false;  // a new sequence: every stage resets the lane before its forward
        if (!fresh) ln.resub = true;       // the callback's resubmit: its in-flight count carries over
        if (front) q_[0].push_front(lane * 2 + sl);   // (P4 B39 (5): the owner's bounded priority)
        else       q_[0].push_back(lane * 2 + sl);
    }
    cv_.notify_all();
    return {};
}

// P4 B34: the lane's next step behind its one step in flight (see the header). Under the lock throughout: the slot's buffers
// and the step's place in stage 0's queue are set together.
std::string Glm5LanePipe::submit_ahead(uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0, bool& taken) {
    taken = false;
    if (lane >= lanes_.size()) return "glm5 lane pipe: lane " + std::to_string(lane) + " of " + std::to_string(lanes_.size());
    const std::string who = "glm5 lane pipe: lane " + std::to_string(lane) + " lookahead";
    if (!ahead_) return who + ": the pipe has no lookahead slots";
    if (T == 0 || T > max_T_ || !ids) return who + ": " + std::to_string(T) + " rows, the pipe takes 1.." + std::to_string(max_T_);
    Lane& ln = lanes_[lane];
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (q_.empty() || stop_ || cancel_ || !err_.empty()) return {};   // not now
        if (s0_busy_ || !q_[0].empty()) return {};                         // stage 0 has work: never delay it
        if (ln.nfl != 1 || ln.in_cb || ln.at0 || ln.failed) return {};    // no step in flight, two, or it has not left stage 0
        const uint32_t end = ln.pos0[ln.cur] + ln.T[ln.cur];
        if (pos0 != end)
            return who + ": pos0 " + std::to_string(pos0) + " is not the end of its step in flight (" + std::to_string(end) + ")";
        const uint32_t sl = ln.cur ^ 1u;
        std::copy_n(ids, T, slot_ids(ln, sl));
        ln.T[sl] = T; ln.pos0[sl] = pos0;
        ln.nfl = 2; ln.at0 = true;
        q_[0].push_back(lane * 2 + sl);
        taken = true;
    }
    cv_.notify_all();
    return {};
}

void Glm5LanePipe::stage_loop(uint32_t s) {
    const bool last = s + 1 == q_.size();   // q_'s size is fixed while the stages run
    for (;;) {
        uint32_t l = 0, sl = 0;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return stop_ || !q_[s].empty(); });
            if (q_[s].empty()) return;        // stopping (stop() first waits for no step in flight)
            l = q_[s].front() >> 1;
            sl = q_[s].front() & 1u;
            q_[s].pop_front();
            ++running_;                       // (P4 B33: cancel() reports the steps a stage holds)
            if (s == 0) s0_busy_ = true;      // (P4 B34)
        }
        Lane& ln = lanes_[l];
        const Step st{l, ln.T[sl], ln.pos0[sl], slot_ids(ln, sl), slot_wide(ln, sl)};
        std::string e;
        try { e = stage_(s, st); }
        catch (const std::exception& ex) { e = std::string("threw: ") + ex.what(); }   // a stage thread must not end the process
        catch (...) { e = "threw a non-std exception"; }
        if (!e.empty()) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (err_.empty())
                    err_ = "lane " + std::to_string(l) + " at " + std::to_string(st.pos0) + " (" + std::to_string(st.T) +
                           " rows), stage " + std::to_string(s) + ": " + e;
                drop_claim(ln, sl);              // the failed step is dropped
                ln.failed = true;                // and the lane's state is suspect: only a new sequence may follow
                if (s == 0) { ln.at0 = false; s0_busy_ = false; }
                --running_;
            }
            cv_.notify_all();
            continue;
        }
        if (!last) {
            bool idle = false;
            {
                std::lock_guard<std::mutex> lk(mu_);
                --running_;
                if (s == 0) { ln.at0 = false; s0_busy_ = false; }
                if (cancel_) { drop_claim(ln, sl); ln.failed = true; }   // P4 B33: no other stage runs it
                else q_[s + 1].push_back(l * 2 + sl);
                idle = s == 0 && idle_ && !cancel_ && q_[0].empty();   // (P4 B34: stage 0 has nothing queued)
            }
            cv_.notify_all();
            if (idle) idle_();
            continue;
        }
        bool call = false, idle = false;
        {
            // the step's position is committed before its callback, so a resubmit from there is at the new position; the
            // lane stays claimed THROUGH the callback (only this thread may submit it, once)
            std::lock_guard<std::mutex> lk(mu_);
            ln.n_pos = st.pos0 + st.T; ++steps_;
            call = !cancel_;   // P4 B33: after cancel() no callback runs (it could start new device work)
            if (call) { ln.in_cb = true; ln.resub = false; ln.cb_tid = std::this_thread::get_id(); }
            else { drop_claim(ln, sl); --running_; if (s == 0) s0_busy_ = false; }
        }
        if (!call) { cv_.notify_all(); continue; }
        std::string ce;
        if (done_) {
            try { done_(l, st.T, st.pos0); }
            catch (const std::exception& ex) { ce = std::string("done callback threw: ") + ex.what(); }
            catch (...) { ce = "done callback threw a non-std exception"; }
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!ce.empty() && err_.empty()) err_ = "lane " + std::to_string(l) + ": " + ce;
            ln.in_cb = false;
            if (!ln.resub) drop_claim(ln, sl);
            ln.resub = false;
            --running_;
            if (s == 0) s0_busy_ = false;
            idle = ahead_possible(ln);   // (P4 B34: its lookahead, past stage 0, is its one step now)
        }
        cv_.notify_all();
        if (idle) idle_();
    }
}

// Rows mode's stage loop (see the header): stage 0 forms the groups, the stages pass them on as units, the last stage runs
// their callbacks with stage 0 held back.
void Glm5LanePipe::stage_loop_rows(uint32_t s) {
    const uint32_t n_st = uint32_t(q_.size());   // fixed while the stages run
    const bool last = s + 1 == n_st;
    for (;;) {
        uint32_t gi = 0;
        {
            std::unique_lock<std::mutex> lk(mu_);
            if (s == 0) {
                // a prefill piece (T > 1) or a step that starts a sequence (pos0 0: every stage resets the lane) runs alone
                auto alone = [&](uint32_t e) { const Lane& ln = lanes_[e >> 1]; return ln.T[e & 1u] > 1 || ln.pos0[e & 1u] == 0; };
                auto lim_now = [&] {
                    const uint32_t cap = group_cap_ ? group_cap_ : std::max<uint32_t>(1u, (busy_ + n_st - 1) / n_st);
                    return std::min(cap, max_group_);
                };
                // the decode lanes at the front of the queue, up to `upto` (the group stage 0 would form now)
                auto queued_dec = [&](uint32_t upto) {
                    uint32_t n = 0;
                    for (uint32_t e : q_[0]) { if (n >= upto || alone(e)) break; ++n; }
                    return n;
                };
                // P4 B42 (see the header): forming now would leave >= n_st decode groups rotating beside this one, the one at the
                // last stage lands within a step, and this one is not full yet
                auto need_wait = [&] {
                    if (!regroup_ || q_[0].empty() || alone(q_[0].front()) || !last_dec_running_ || dec_inflight_ < n_st) return false;
                    const uint32_t lim = lim_now();
                    return lim > 1 && queued_dec(lim) < lim;
                };
                cv_.wait(lk, [&] { return stop_ || (!q_[0].empty() && !pgate_); });
                if (q_[0].empty()) return;        // stopping (stop() first waits for no step in flight)
                if (need_wait()) {
                    const auto t0 = std::chrono::steady_clock::now();
                    const auto deadline = t0 + regroup_wait_;
                    const uint32_t before = queued_dec(max_group_);
                    ++rg_.waits;
                    for (;;) {
                        if (!cv_.wait_until(lk, deadline, [&] { return stop_ || pgate_ || !need_wait(); })) { ++rg_.timeouts; break; }
                        if (stop_) break;
                        if (!pgate_) break;                                    // the condition cleared (a landing did not resubmit, or fewer groups)
                        cv_.wait(lk, [&] { return stop_ || !pgate_; });        // a landing's callback resubmits its lanes: let them in
                        if (stop_ || !need_wait()) break;                      // (still fragmented with time left: wait for the next)
                    }
                    rg_.wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                    if (queued_dec(max_group_) > before) ++rg_.merges;
                    if (q_[0].empty()) continue;                               // (a cancel emptied the queue: back to the wait)
                    if (pgate_) continue;                                      // (a callback is still resubmitting: back to the wait)
                }
                gi = gfree_.back();               // (n_groups(): a step is in at most one)
                gfree_.pop_back();
                Group& g = groups_[gi];
                g.lanes.clear();
                g.slots.clear();
                g.steps.clear();
                const uint32_t lim = lim_now();
                const bool dec = !alone(q_[0].front());
                auto take = [&] { g.lanes.push_back(q_[0].front() >> 1); g.slots.push_back(q_[0].front() & 1u); q_[0].pop_front(); };
                if (!dec) take();
                else
                    while (!q_[0].empty() && g.lanes.size() < lim && !alone(q_[0].front())) take();
                for (size_t i = 0; i < g.lanes.size(); ++i) {   // (P4 B34: at most one step of a lane waits for stage 0)
                    Lane& ln = lanes_[g.lanes[i]];
                    const uint32_t sl = g.slots[i];
                    g.steps.push_back(Step{g.lanes[i], ln.T[sl], ln.pos0[sl], slot_ids(ln, sl), g.lanes.size() == 1 ? slot_wide(ln, sl) : nullptr});
                }
                g.decode = dec;
                if (dec) ++dec_inflight_;         // (P4 B42: freed by free_group)
                s0_busy_ = true;
            } else {
                cv_.wait(lk, [&] { return stop_ || !gq_[s].empty(); });
                if (gq_[s].empty()) return;
                gi = gq_[s].front();
                gq_[s].pop_front();
                if (last) { last_gi_ = gi; last_dec_running_ = groups_[gi].decode; }   // (P4 B42: its landing is one step away)
            }
            ++running_;                           // (P4 B33: cancel() reports the groups a stage holds)
        }
        Group& g = groups_[gi];
        const uint32_t G = uint32_t(g.lanes.size());
        std::string e;
        try {
            e = G == 1 ? stage_(s, g.steps[0]) : rows_stage_(s, std::span<const Step>(g.steps.data(), G), g.wide.data());
        }
        catch (const std::exception& ex) { e = std::string("threw: ") + ex.what(); }   // a stage thread must not end the process
        catch (...) { e = "threw a non-std exception"; }
        auto who = [&] {
            if (G == 1)
                return "lane " + std::to_string(g.steps[0].lane) + " at " + std::to_string(g.steps[0].pos0) + " (" +
                       std::to_string(g.steps[0].T) + " rows)";
            std::string w = "lanes";
            for (const Step& st : g.steps) w += " " + std::to_string(st.lane) + "@" + std::to_string(st.pos0);
            return w + " (a group of " + std::to_string(G) + ")";
        };
        if (!e.empty()) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (err_.empty()) err_ = who() + ", stage " + std::to_string(s) + ": " + e;
                for (uint32_t i = 0; i < G; ++i) {   // the group's steps are dropped, every lane's state suspect
                    Lane& ln = lanes_[g.lanes[i]];
                    drop_claim(ln, g.slots[i]); ln.failed = true;
                    if (s == 0) ln.at0 = false;
                }
                if (s == 0) s0_busy_ = false;
                free_group(gi);
                --running_;
            }
            cv_.notify_all();
            continue;
        }
        if (!last) {
            bool idle = false;
            {
                std::lock_guard<std::mutex> lk(mu_);
                --running_;
                if (s == 0) { for (uint32_t l : g.lanes) lanes_[l].at0 = false; s0_busy_ = false; }
                if (cancel_) {                     // P4 B33: no other stage runs the group
                    for (uint32_t i = 0; i < G; ++i) { drop_claim(lanes_[g.lanes[i]], g.slots[i]); lanes_[g.lanes[i]].failed = true; }
                    free_group(gi);
                } else {
                    gq_[s + 1].push_back(gi);
                }
                idle = s == 0 && idle_ && !cancel_ && q_[0].empty();   // (P4 B34: stage 0 has nothing queued)
            }
            cv_.notify_all();
            if (idle) idle_();
            continue;
        }
        bool call = false, idle = false;
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (const Step& st : g.steps) {
                Lane& ln = lanes_[st.lane];
                ln.n_pos = st.pos0 + st.T; ++steps_;
            }
            ++gsizes_[G];
            call = !cancel_;                       // P4 B33: after cancel() no callback runs (it could start new device work)
            if (call) {
                for (const Step& st : g.steps) {
                    Lane& ln = lanes_[st.lane];
                    ln.in_cb = true; ln.resub = false; ln.cb_tid = std::this_thread::get_id();
                }
                pgate_ = true;                     // stage 0 waits: the lanes this callback resubmits regroup
            } else {
                for (uint32_t i = 0; i < G; ++i) drop_claim(lanes_[g.lanes[i]], g.slots[i]);
                if (s == 0) s0_busy_ = false;
                free_group(gi);
                --running_;
            }
        }
        if (!call) { cv_.notify_all(); continue; }
        std::string ce;
        try {
            if (G == 1) { if (done_) done_(g.steps[0].lane, g.steps[0].T, g.steps[0].pos0); }
            else if (rows_done_) rows_done_(std::span<const uint32_t>(g.lanes.data(), G));
        }
        catch (const std::exception& ex) { ce = std::string("done callback threw: ") + ex.what(); }
        catch (...) { ce = "done callback threw a non-std exception"; }
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (!ce.empty() && err_.empty()) err_ = who() + ": " + ce;
            for (uint32_t i = 0; i < G; ++i) {
                Lane& ln = lanes_[g.lanes[i]];
                ln.in_cb = false;
                if (!ln.resub) drop_claim(ln, g.slots[i]);
                ln.resub = false;
            }
            pgate_ = false;
            free_group(gi);
            --running_;
            if (s == 0) s0_busy_ = false;
            for (uint32_t i = 0; i < G && !idle; ++i) idle = ahead_possible(lanes_[g.lanes[i]]);   // (P4 B34)
        }
        cv_.notify_all();
        if (idle) idle_();
    }
}

// P4 B33: a process stop -- every step (or group) waiting for a stage is dropped; what a stage holds now finishes that stage and
// goes no further (stage_loop / stage_loop_rows check cancel_). A step waiting for stage 0 has run nowhere: its lane is only
// released; one that a stage already ran is half-stepped: its lane is marked failed, as a stage error marks it.
uint32_t Glm5LanePipe::cancel() {
    uint32_t held = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (q_.empty()) return 0;   // not running (never started, stopped, or paused by its owner)
        cancel_ = true;
        for (size_t s = 0; s < q_.size(); ++s) {
            for (uint32_t e : q_[s]) {
                Lane& ln = lanes_[e >> 1];
                drop_claim(ln, e & 1u);
                if (s == 0) ln.at0 = false;
                else ln.failed = true;
            }
            q_[s].clear();
        }
        for (auto& gq : gq_) {      // rows mode: groups between stages (>= 1 stage ran them)
            for (uint32_t gi : gq) {
                const Group& g = groups_[gi];
                for (size_t i = 0; i < g.lanes.size(); ++i) { drop_claim(lanes_[g.lanes[i]], g.slots[i]); lanes_[g.lanes[i]].failed = true; }
                free_group(gi);
            }
            gq.clear();
        }
        held = running_;
    }
    cv_.notify_all();
    return held;
}

std::string Glm5LanePipe::stop() {
    if (th_.empty()) return {};
    for (const auto& t : th_)
        if (t.get_id() == std::this_thread::get_id())
            return "glm5 lane pipe: stop() from a stage thread (a done callback) would deadlock";
    {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return busy_ == 0; });
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : th_) t.join();
    th_.clear();
    std::lock_guard<std::mutex> lk(mu_);
    q_.clear();
    gq_.clear();
    stage_ = nullptr;
    done_ = nullptr;
    idle_ = nullptr;
    rows_stage_ = nullptr;
    rows_done_ = nullptr;
    rows_ = false; pgate_ = false;
    std::string e;
    std::swap(e, err_);
    return e;
}

// P4 B39 (see the header). Every claim drop notifies cv_ (the stage loops, cancel), so the wait wakes when the lane's last
// claim goes; a stopped pipe holds no claims (stop() waited for busy_ == 0).
std::string Glm5LanePipe::wait_lane_idle(uint32_t lane) {
    if (lane >= lanes_.size()) return "glm5 lane pipe: lane " + std::to_string(lane) + " of " + std::to_string(lanes_.size());
    for (const auto& t : th_)
        if (t.get_id() == std::this_thread::get_id())
            return "glm5 lane pipe: wait_lane_idle() from a stage thread (a done callback) would deadlock";
    std::unique_lock<std::mutex> lk(mu_);
    cv_.wait(lk, [&] { return th_.empty() || lanes_[lane].nfl == 0; });
    return {};
}

uint32_t Glm5LanePipe::lane_pos(uint32_t lane) const {
    std::lock_guard<std::mutex> lk(mu_);
    return lane < lanes_.size() ? lanes_[lane].n_pos : 0;
}

uint64_t Glm5LanePipe::steps_done() const {
    std::lock_guard<std::mutex> lk(mu_);
    return steps_;
}

std::string Glm5LanePipe::set_lane_pos(uint32_t lane, uint32_t n_pos) {
    std::lock_guard<std::mutex> lk(mu_);
    if (lane >= lanes_.size()) return "glm5 lane pipe: lane " + std::to_string(lane) + " of " + std::to_string(lanes_.size());
    if (lanes_[lane].nfl) return "glm5 lane pipe: lane " + std::to_string(lane) + " has a step in flight";
    if (lanes_[lane].failed)
        return "glm5 lane pipe: lane " + std::to_string(lane) + ": its last step failed part-way, so its position may not move "
               "until it is reset";
    lanes_[lane].n_pos = n_pos;
    return {};
}

std::string Glm5LanePipe::rewind_lane(uint32_t lane, uint32_t n_pos) {
    std::lock_guard<std::mutex> lk(mu_);
    if (lane >= lanes_.size()) return "glm5 lane pipe: lane " + std::to_string(lane) + " of " + std::to_string(lanes_.size());
    Lane& ln = lanes_[lane];
    const std::string who = "glm5 lane pipe: lane " + std::to_string(lane) + " rewind";
    if (ln.nfl && !(ln.in_cb && ln.cb_tid == std::this_thread::get_id() && !ln.resub && ln.nfl == 1))
        return who + ": it has a step in flight";
    if (ln.failed) return who + ": its last step failed part-way";
    if (n_pos > ln.n_pos) return who + ": " + std::to_string(n_pos) + " is past its position " + std::to_string(ln.n_pos);
    ln.n_pos = n_pos;
    return {};
}

std::string Glm5LanePipe::reset_lane(uint32_t lane) {
    std::lock_guard<std::mutex> lk(mu_);
    if (lane >= lanes_.size()) return "glm5 lane pipe: lane " + std::to_string(lane) + " of " + std::to_string(lanes_.size());
    if (lanes_[lane].nfl) return "glm5 lane pipe: lane " + std::to_string(lane) + " has a step in flight";
    lanes_[lane].n_pos = 0;
    lanes_[lane].failed = false;
    return {};
}

std::string Glm5LanePipe::error() const {
    std::lock_guard<std::mutex> lk(mu_);
    return err_;
}

}  // namespace ie
