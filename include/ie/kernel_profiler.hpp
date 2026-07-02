#pragma once

#include <sycl/sycl.hpp>

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
    struct Entry { const char* name; sycl::event evt; };
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

    void push(const char* name, const sycl::event& e) {
        entries_.push_back({name, e});
    }

    void begin_step() { entries_.clear(); }

    // Harvest timing for all events collected since begin_step().
    // MUST be called after the queue is idle (event.wait() / q.wait()).
    std::vector<Stat> harvest() {
        std::vector<Stat> stats;
        stats.reserve(64);
        for (auto& ent : entries_) {
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
        }
        return stats;  // preserved in first-dispatch order
    }

private:
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
    if (g_profiler) [[unlikely]] g_profiler->push(name, e);
    return e;
}

}  // namespace ie
