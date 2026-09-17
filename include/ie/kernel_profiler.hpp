#pragma once

#include <sycl/sycl.hpp>
#include <mutex>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace ie {

// ---------------------------------------------------------------------------
// KernelProfiler — collects named SYCL events for one forward step.
//
// Usage (in a tool):
//   ie::KernelProfiler prof;
//   ie::g_profiler = &prof;
//   prof.begin_step();
//   model.forward(...).wait();      // all kernels submit and complete
//   auto stats = prof.harvest();    // must call AFTER queue is idle
//   ie::g_profiler = nullptr;
// ---------------------------------------------------------------------------
class KernelProfiler {
public:
    struct Entry { const char* name; sycl::event evt; uint8_t dev; };
    // Per-device GPU activity per token: busy = sum of kernel durations,
    // span = first kernel start .. last kernel end within one token (device
    // clocks are independent, so never across devices). busy/span = how much
    // of its own window the GPU is executing; a window much wider than busy
    // means the host is feeding it slower than it runs (2026-09-04).
    struct DevSpan { uint64_t busy_ns = 0, span_ns = 0; uint32_t tokens = 0; };
    std::vector<DevSpan> dev_spans;
    struct Stat {
        std::string name;
        uint32_t calls    = 0;
        uint64_t total_ns = 0;
        uint64_t min_ns   = UINT64_MAX;
        uint64_t max_ns   = 0;
        double total_ms() const noexcept { return double(total_ns) * 1e-6; }
        double avg_ms()   const noexcept { return calls ? total_ms() / calls : 0.0; }
        double min_ms()   const noexcept { return double(min_ns)   * 1e-6; }
        double max_ms()   const noexcept { return double(max_ns)   * 1e-6; }
    };

    void push(const char* name, const sycl::event& e, const sycl::queue& q) {
        std::lock_guard<std::mutex> lk(mu_);
        entries_.push_back({name, e, dev_index(q)});
    }
    // No queue in scope (oneDNN submits internally — deepseek4's ds4_pe):
    // counted in the per-kernel buckets, excluded from the per-device spans.
    void push(const char* name, const sycl::event& e) {
        std::lock_guard<std::mutex> lk(mu_);
        entries_.push_back({name, e, kNoDev});
    }
    static constexpr uint8_t kNoDev = 0xFF;
    // Call once per decode token (before its forward) to segment the spans.
    void mark_token() { std::lock_guard<std::mutex> lk(mu_); tok_bounds_.push_back(entries_.size()); }

    void begin_step() { entries_.clear(); tok_bounds_.clear(); }

    // Harvest timing for all events collected since begin_step().
    // MUST be called after the queue is idle (event.wait() / q.wait()).
    std::vector<Stat> harvest() {
        std::vector<Stat> stats;
        stats.reserve(64);
        const size_t nd = devs_.size();
        dev_spans.assign(nd, {});
        std::vector<size_t> b = tok_bounds_;
        if (b.empty() || b[0] != 0) b.insert(b.begin(), 0);
        b.push_back(entries_.size());
        std::vector<uint64_t> lo(nd), hi(nd), busy(nd);
        for (size_t si = 0; si + 1 < b.size(); ++si) {
            std::fill(lo.begin(), lo.end(), UINT64_MAX);
            std::fill(hi.begin(), hi.end(), 0);
            std::fill(busy.begin(), busy.end(), 0);
            for (size_t i = b[si]; i < b[si + 1]; ++i) {
                auto& ent = entries_[i];
                uint64_t t0 = ent.evt.get_profiling_info<
                    sycl::info::event_profiling::command_start>();
                uint64_t t1 = ent.evt.get_profiling_info<
                    sycl::info::event_profiling::command_end>();
                uint64_t dur = (t1 > t0) ? (t1 - t0) : 0;
                auto it = std::find_if(stats.begin(), stats.end(),
                    [&](const Stat& s) { return s.name == ent.name; });
                if (it == stats.end()) {
                    stats.push_back({ent.name, 1, dur, dur, dur});
                } else {
                    it->calls++;
                    it->total_ns += dur;
                    it->min_ns = std::min(it->min_ns, dur);
                    it->max_ns = std::max(it->max_ns, dur);
                }
                if (ent.dev != kNoDev) {
                    lo[ent.dev] = std::min(lo[ent.dev], t0);
                    hi[ent.dev] = std::max(hi[ent.dev], t1);
                    busy[ent.dev] += dur;
                }
            }
            for (size_t d = 0; d < nd; ++d)
                if (hi[d] > 0) {
                    dev_spans[d].busy_ns += busy[d];
                    dev_spans[d].span_ns += hi[d] - lo[d];
                    dev_spans[d].tokens++;
                }
        }
        return stats;  // preserved in first-dispatch order
    }

private:
    uint8_t dev_index(const sycl::queue& q) {
        const sycl::device d = q.get_device();
        for (size_t i = 0; i < devs_.size(); ++i) if (devs_[i] == d) return uint8_t(i);
        devs_.push_back(d);
        return uint8_t(devs_.size() - 1);
    }
    std::mutex mu_;
    std::vector<sycl::device> devs_;
    std::vector<size_t> tok_bounds_;
    std::vector<Entry> entries_;
};

// Global profiler pointer.  null = profiling disabled (default / inference path).
// Tools set this before running a forward pass, harvest after.
inline KernelProfiler* g_profiler = nullptr;

// ps() — profiled submit.  Drop-in replacement for q.submit():
//   auto evt = ie::ps(q, "kernel_name", [&](sycl::handler& h) { ... });
// When g_profiler == nullptr the branch is branch-predicted-not-taken and
// the only overhead is a pointer compare.
template<typename F>
inline sycl::event ps(sycl::queue& q, const char* name, F&& fn) {
    auto e = q.submit(std::forward<F>(fn));
    if (g_profiler) [[unlikely]] g_profiler->push(name, e, q);
    return e;
}

}  // namespace ie
