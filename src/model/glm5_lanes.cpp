// src/model/glm5_lanes.cpp -- GLM-5.3-Flash request lanes' card pipe (P4 B7). See include/ie/glm5_lanes.hpp.
#include "ie/glm5_lanes.hpp"

#include <algorithm>
#include <exception>
#include <pthread.h>

namespace ie {

Glm5LanePipe::Glm5LanePipe(uint32_t n_lanes, uint32_t max_T, uint64_t wide_row)
    : lanes_(std::max<uint32_t>(n_lanes, 1)), max_T_(std::max<uint32_t>(max_T, 1)), wide_row_(wide_row) {
    for (Lane& ln : lanes_) {   // value-initialised: every page is touched here, before the caller loads a model
        ln.ids.assign(max_T_, 0);
        ln.wide.assign(size_t(max_T_) * wide_row_, 0.f);
    }
}

Glm5LanePipe::~Glm5LanePipe() { stop(); }

uint64_t Glm5LanePipe::host_bytes() const {
    uint64_t b = 0;
    for (const Lane& ln : lanes_) b += ln.ids.size() * sizeof(int32_t) + ln.wide.size() * sizeof(float);
    for (const Group& g : groups_) b += g.wide.size() * sizeof(float);   // rows mode's group buffers (made by start_rows)
    return b;
}

std::string Glm5LanePipe::start(uint32_t n_stages, StageFn stage, DoneFn done) {
    if (running()) return "glm5 lane pipe: already running";
    if (n_stages == 0 || !stage) return "glm5 lane pipe: no stages";
    {
        std::lock_guard<std::mutex> lk(mu_);
        q_.assign(n_stages, {});
        busy_ = 0; stop_ = false; err_.clear(); steps_ = 0;
        for (Lane& ln : lanes_) { ln.in_flight = false; ln.in_cb = false; ln.resub = false; }
        stage_ = std::move(stage);
        done_ = std::move(done);
    }
    for (uint32_t s = 0; s < n_stages; ++s) {
        th_.emplace_back([this, s] { stage_loop(s); });
        const std::string name = "g5pipe-s" + std::to_string(s);
        pthread_setname_np(th_.back().native_handle(), name.c_str());
    }
    return {};
}

std::string Glm5LanePipe::start_rows(uint32_t n_stages, StageFn stage, RowsStageFn rows_stage, DoneFn done, RowsDoneFn rows_done,
                                     uint32_t max_group, uint32_t group_cap) {
    if (running()) return "glm5 lane pipe: already running";
    if (n_stages == 0 || !stage || !rows_stage) return "glm5 lane pipe: no stages";
    if (max_group < 1 || max_group > 16) return "glm5 lane pipe: a group takes 1..16 rows, not " + std::to_string(max_group);
    {
        std::lock_guard<std::mutex> lk(mu_);
        q_.assign(n_stages, {});
        gq_.assign(n_stages, {});
        busy_ = 0; stop_ = false; err_.clear(); steps_ = 0; pgate_ = false;
        for (Lane& ln : lanes_) { ln.in_flight = false; ln.in_cb = false; ln.resub = false; }
        if (groups_.size() != lanes_.size() + 1 || max_group_ != max_group) make_groups(max_group);   // (once, not per resume)
        gfree_.clear();
        for (uint32_t i = uint32_t(groups_.size()); i-- > 0;) gfree_.push_back(i);
        if (gsizes_.size() != 17) gsizes_.assign(17, 0);   // cumulative over resumes
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
    groups_.assign(lanes_.size() + 1, {});
    for (Group& g : groups_) {
        g.lanes.reserve(max_group);
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

std::string Glm5LanePipe::submit(uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0) {
    if (lane >= lanes_.size()) return "glm5 lane pipe: lane " + std::to_string(lane) + " of " + std::to_string(lanes_.size());
    Lane& ln = lanes_[lane];
    const std::string who = "glm5 lane pipe: lane " + std::to_string(lane);
    bool fresh = false, failed = false;
    uint32_t n_pos = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (q_.empty() || stop_) return "glm5 lane pipe: not running";
        if (!err_.empty()) return "glm5 lane pipe: a stage failed: " + err_;
        if (ln.in_flight) {
            if (!ln.in_cb || ln.cb_tid != std::this_thread::get_id()) return who + " has a step in flight";
            if (ln.resub) return who + " was already resubmitted from its callback";
        } else {
            ln.in_flight = true; ++busy_; fresh = true;
        }
        n_pos = ln.n_pos;
        failed = ln.failed;
    }
    auto refuse = [&](const std::string& e) {   // a refused step gives a fresh claim back (a callback's lane stays claimed)
        if (fresh) {
            { std::lock_guard<std::mutex> lk(mu_); ln.in_flight = false; --busy_; }
            cv_.notify_all();
        }
        return e;
    };
    if (T == 0 || T > max_T_) return refuse(who + ": " + std::to_string(T) + " rows, the pipe takes 1.." + std::to_string(max_T_));
    if (!ids) return refuse(who + ": no ids");
    if (pos0 != 0 && failed)
        return refuse(who + ": its last step failed part-way, so only a new sequence (pos0 0) may follow");
    if (pos0 != 0 && pos0 != n_pos)
        return refuse(who + ": pos0 " + std::to_string(pos0) + " is neither 0 (a new sequence) nor its position " + std::to_string(n_pos));
    std::copy_n(ids, T, ln.ids.data());   // the claim makes these buffers ours: no step of this lane is in any stage
    ln.T = T; ln.pos0 = pos0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (pos0 == 0) ln.failed = false;  // a new sequence: every stage resets the lane before its forward
        if (!fresh) ln.resub = true;       // the callback's resubmit: its in-flight count carries over
        q_[0].push_back(lane);
    }
    cv_.notify_all();
    return {};
}

void Glm5LanePipe::stage_loop(uint32_t s) {
    const bool last = s + 1 == q_.size();   // q_'s size is fixed while the stages run
    for (;;) {
        uint32_t l = 0;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [&] { return stop_ || !q_[s].empty(); });
            if (q_[s].empty()) return;        // stopping (stop() first waits for no step in flight)
            l = q_[s].front();
            q_[s].pop_front();
        }
        Lane& ln = lanes_[l];
        const Step st{l, ln.T, ln.pos0, ln.ids.data(), ln.wide.data()};
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
                ln.in_flight = false; --busy_;   // the failed step is dropped
                ln.failed = true;                // and the lane's state is suspect: only a new sequence may follow
            }
            cv_.notify_all();
            continue;
        }
        if (!last) {
            { std::lock_guard<std::mutex> lk(mu_); q_[s + 1].push_back(l); }
            cv_.notify_all();
            continue;
        }
        {
            // the step's position is committed before its callback, so a resubmit from there is at the new position; the
            // lane stays claimed THROUGH the callback (only this thread may submit it, once)
            std::lock_guard<std::mutex> lk(mu_);
            ln.n_pos = st.pos0 + st.T; ++steps_;
            ln.in_cb = true; ln.resub = false; ln.cb_tid = std::this_thread::get_id();
        }
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
            if (!ln.resub) { ln.in_flight = false; --busy_; }
            ln.resub = false;
        }
        cv_.notify_all();
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
                cv_.wait(lk, [&] { return stop_ || (!q_[0].empty() && !pgate_); });
                if (q_[0].empty()) return;        // stopping (stop() first waits for no step in flight)
                gi = gfree_.back();               // (n_lanes + 1 groups; a lane is in at most one)
                gfree_.pop_back();
                Group& g = groups_[gi];
                g.lanes.clear();
                g.steps.clear();
                const uint32_t cap = group_cap_ ? group_cap_ : std::max<uint32_t>(1u, (busy_ + n_st - 1) / n_st);
                const uint32_t lim = std::min(cap, max_group_);
                // a prefill piece (T > 1) or a step that starts a sequence (pos0 0: every stage resets the lane) runs alone
                auto alone = [&](uint32_t l) { return lanes_[l].T > 1 || lanes_[l].pos0 == 0; };
                if (alone(q_[0].front())) {
                    g.lanes.push_back(q_[0].front());
                    q_[0].pop_front();
                } else {
                    while (!q_[0].empty() && g.lanes.size() < lim && !alone(q_[0].front())) {
                        g.lanes.push_back(q_[0].front());
                        q_[0].pop_front();
                    }
                }
                for (uint32_t l : g.lanes) {
                    Lane& ln = lanes_[l];
                    g.steps.push_back(Step{l, ln.T, ln.pos0, ln.ids.data(), g.lanes.size() == 1 ? ln.wide.data() : nullptr});
                }
            } else {
                cv_.wait(lk, [&] { return stop_ || !gq_[s].empty(); });
                if (gq_[s].empty()) return;
                gi = gq_[s].front();
                gq_[s].pop_front();
            }
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
                for (uint32_t l : g.lanes) {        // the group's steps are dropped, every lane's state suspect
                    lanes_[l].in_flight = false; lanes_[l].failed = true; --busy_;
                }
                gfree_.push_back(gi);
            }
            cv_.notify_all();
            continue;
        }
        if (!last) {
            { std::lock_guard<std::mutex> lk(mu_); gq_[s + 1].push_back(gi); }
            cv_.notify_all();
            continue;
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (const Step& st : g.steps) {
                Lane& ln = lanes_[st.lane];
                ln.n_pos = st.pos0 + st.T; ++steps_;
                ln.in_cb = true; ln.resub = false; ln.cb_tid = std::this_thread::get_id();
            }
            pgate_ = true;                         // stage 0 waits: the lanes this callback resubmits regroup
            ++gsizes_[G];
        }
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
            for (uint32_t l : g.lanes) {
                Lane& ln = lanes_[l];
                ln.in_cb = false;
                if (!ln.resub) { ln.in_flight = false; --busy_; }
                ln.resub = false;
            }
            pgate_ = false;
            gfree_.push_back(gi);
        }
        cv_.notify_all();
    }
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
    rows_stage_ = nullptr;
    rows_done_ = nullptr;
    rows_ = false; pgate_ = false;
    std::string e;
    std::swap(e, err_);
    return e;
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
    if (lanes_[lane].in_flight) return "glm5 lane pipe: lane " + std::to_string(lane) + " has a step in flight";
    if (lanes_[lane].failed)
        return "glm5 lane pipe: lane " + std::to_string(lane) + ": its last step failed part-way, so its position may not move "
               "until it is reset";
    lanes_[lane].n_pos = n_pos;
    return {};
}

std::string Glm5LanePipe::reset_lane(uint32_t lane) {
    std::lock_guard<std::mutex> lk(mu_);
    if (lane >= lanes_.size()) return "glm5 lane pipe: lane " + std::to_string(lane) + " of " + std::to_string(lanes_.size());
    if (lanes_[lane].in_flight) return "glm5 lane pipe: lane " + std::to_string(lane) + " has a step in flight";
    lanes_[lane].n_pos = 0;
    lanes_[lane].failed = false;
    return {};
}

std::string Glm5LanePipe::error() const {
    std::lock_guard<std::mutex> lk(mu_);
    return err_;
}

}  // namespace ie
