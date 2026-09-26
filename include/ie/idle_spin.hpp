// include/ie/idle_spin.hpp — #29 (docs/server_idle_spin_watchdog_2026-09-24.md): the idle-spin watchdog of `ie serve`.
//
// On 2026-09-19 `ie serve` burned ~900 % CPU for eight hours with nothing in flight and nothing queued, after a client
// disconnected mid-generation. It was never reproduced, so the cause is unknown. This makes the next occurrence
// self-diagnosing and its logging bounded: once the server has been idle for a whole window (60 s) and the process's
// CPU time grew by at least `min_cores` cores over that window, one line names the busiest threads; it repeats at most
// every `repeat_s` while the spin lasts, and /health reports {"seconds", "cores"} the whole time.
//
// The detector is arithmetic over samples, so it is unit-tested with fake ones; the /proc readers and the report
// formatter live here too, so the test can run them against a thread it spins (tests/unit/idle_spin_test.cpp). No SYCL.
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <string>
#include <string_view>
#include <vector>

namespace ie {

struct IdleSpinConfig {
    bool   enabled   = true;    // IE_IDLE_SPIN_WATCHDOG=0 turns the watchdog off
    double min_cores = 2.0;     // IE_IDLE_SPIN_CORES: the CPU rate over an idle window that counts as a spin (0 = any)
    double window_s  = 60.0;    // how long the server must have been idle; the rate is measured over the same span
    double repeat_s  = 600.0;   // a spin that persists is reported again after this long
    double sample_s  = 5.0;     // the sampling period (the server's watcher thread)
};
// The configuration from IE_IDLE_SPIN_WATCHDOG and IE_IDLE_SPIN_CORES (the values, nullptr = unset). A value that cannot
// be used keeps the default and leaves a message in `*warn`.
IdleSpinConfig idle_spin_config(const char* watchdog_env, const char* cores_env, std::string* warn);

class IdleSpinDetector {
public:
    IdleSpinDetector(IdleSpinConfig cfg, double ticks_per_s) : cfg_(cfg), tps_(ticks_per_s > 0 ? ticks_per_s : 100.0) {}
    // One sample: a monotonic time (s), the process's utime + stime (clock ticks), and whether the server was idle over
    // the WHOLE interval since the previous sample (nothing in flight or queued now, no request admitted meanwhile).
    // Returns true when a report is due: the first sample that finds a qualifying window, then every repeat_s after.
    bool update(double now_s, uint64_t cpu_ticks, bool idle);
    bool     spinning() const { return spinning_; }
    // how long the spin has lasted: from the start of the first window whose rate reached the threshold (0 when none)
    double   spin_seconds() const { return spinning_ ? last_t_ - spin_start_ : 0.0; }
    // the CPU rate over the latest full idle window, in cores (0 while the server has not been idle for a window)
    double   cores() const { return cores_; }
    uint64_t reports() const { return reports_; }
    const IdleSpinConfig& config() const { return cfg_; }

private:
    struct Sample { double t; uint64_t ticks; };
    IdleSpinConfig     cfg_;
    double             tps_;
    std::deque<Sample> idle_;          // the samples since the server last became idle; a busy interval restarts it
    bool               spinning_ = false;
    double             spin_start_ = 0, last_report_ = 0, last_t_ = 0, cores_ = 0;
    uint64_t           reports_ = 0;
};

// ---- /proc (Linux) --------------------------------------------------------------------------------------------------------
struct ThreadCpu { int tid = 0; std::string comm; char state = '?'; uint64_t utime = 0, stime = 0; };
// A /proc/<pid>/stat or /proc/<pid>/task/<tid>/stat line: the comm sits in parentheses and may hold spaces and ')'.
bool parse_proc_stat(std::string_view line, ThreadCpu& out);
uint64_t proc_self_cpu_ticks();                                    // this process's utime + stime; 0 when unreadable
// This process's threads, at most `max_threads` of them read; the ones past the bound are only counted, in `*more`.
std::vector<ThreadCpu> proc_self_threads(size_t max_threads, size_t* more);
std::string proc_self_thread_wchan(int tid);                       // where the thread sleeps in the kernel ("0": running)
int current_tid();

// ---- the report -------------------------------------------------------------------------------------------------------
struct ThreadDelta {
    int tid = 0; std::string comm; char state = '?';
    double cores = 0, usr = 0, sys = 0;   // over the interval, in cores
    bool http = false;                    // a request thread of the HTTP pool (the server tags it)
    std::string wchan;                    // filled by the server for the ones it reports
};
// The `top_n` busiest threads between two snapshots `interval_s` apart, busiest first; threads with no CPU in the
// interval are left out. Empty when `before` is empty (there is no baseline). A thread absent from `before` was born in
// the interval, so all of its CPU is counted.
std::vector<ThreadDelta> top_thread_deltas(const std::vector<ThreadCpu>& before, const std::vector<ThreadCpu>& after,
                                           double interval_s, double ticks_per_s, size_t top_n);
// The one log line. `threads` is the process's thread count, `comms` its threads' names, `omp_default_team` what
// omp_get_max_threads() says on the watcher (the team a new parallel region would start), `pid` for the stack hint.
// `baseline` says whether `top` came from two snapshots: an empty `top` then means no thread had CPU in the interval.
std::string idle_spin_report(const IdleSpinDetector& d, const std::vector<ThreadDelta>& top, double interval_s, bool baseline,
                             size_t threads, const std::vector<ThreadCpu>& comms, int omp_default_team, int pid);

}  // namespace ie
