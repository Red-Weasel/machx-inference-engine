// tests/unit/idle_spin_test.cpp -- #29 (docs/server_idle_spin_watchdog_2026-09-24.md): the idle-spin watchdog's detector,
// its /proc readers and its report line. CPU only, no SYCL:
//   g++ -std=c++20 -O1 -Wall -Wextra -pthread -I include tests/unit/idle_spin_test.cpp src/server/idle_spin.cpp
// The detector is driven with fake samples (100 ticks/s, a sample every 5 s, as the server takes them); the /proc part
// spins a real thread for 400 ms and checks the readers name it as the busiest.
#include "ie/idle_spin.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("[%s] %s\n", ok ? " ok " : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}
bool near(double a, double b, double tol = 1e-9) { return std::fabs(a - b) <= tol; }

constexpr double kTps = 100.0, kStep = 5.0;

// Feeds a detector from t0 (exclusive) to t1 (inclusive) in kStep steps at `cores` of CPU; `busy_at` marks samples whose
// interval had a request. Returns the times at which a report was due.
struct Run {
    ie::IdleSpinDetector d; double t = 0; uint64_t ticks = 0;
    explicit Run(ie::IdleSpinConfig c) : d(c, kTps) { d.update(0, 0, true); }
    std::vector<double> go(double t1, double cores, const std::vector<double>& busy_at = {}) {
        std::vector<double> due;
        while (t + kStep <= t1 + 1e-9) {
            t += kStep; ticks += uint64_t(std::llround(cores * kTps * kStep));
            bool idle = true; for (double b : busy_at) if (near(b, t)) idle = false;
            if (d.update(t, ticks, idle)) due.push_back(t);
        }
        return due;
    }
};

void config_tests() {
    std::string w;
    auto c = ie::idle_spin_config(nullptr, nullptr, &w);
    check(c.enabled && near(c.min_cores, 2.0) && near(c.window_s, 60) && near(c.repeat_s, 600) && w.empty(), "defaults: on, 2 cores over 60 s, again every 600 s");
    check(!ie::idle_spin_config("0", nullptr, &w).enabled, "IE_IDLE_SPIN_WATCHDOG=0 turns it off");
    check(ie::idle_spin_config("1", nullptr, &w).enabled, "IE_IDLE_SPIN_WATCHDOG=1 keeps it on");
    w.clear(); c = ie::idle_spin_config(nullptr, "3.5", &w); check(near(c.min_cores, 3.5) && w.empty(), "IE_IDLE_SPIN_CORES=3.5");
    w.clear(); c = ie::idle_spin_config(nullptr, "0", &w); check(near(c.min_cores, 0.0) && w.empty(), "IE_IDLE_SPIN_CORES=0 is valid (any idle window reports)");
    for (const char* bad : {"abc", "-1", "2.5x", "inf", "nan", " "}) {
        w.clear(); c = ie::idle_spin_config(nullptr, bad, &w);
        check(near(c.min_cores, 2.0) && !w.empty(), std::string("IE_IDLE_SPIN_CORES=\"") + bad + "\" keeps the default and warns");
    }
    w.clear(); c = ie::idle_spin_config(nullptr, "", &w); check(near(c.min_cores, 2.0) && w.empty(), "an empty IE_IDLE_SPIN_CORES is unset");
}

void detector_tests() {
    const ie::IdleSpinConfig def;
    {   // a quiet idle server: 0.02 cores for 30 min
        Run r(def); const auto due = r.go(1800, 0.02);
        check(due.empty() && !r.d.spinning() && near(r.d.spin_seconds(), 0), "quiet idle (0.02 cores, 30 min): no report");
        check(near(r.d.cores(), 0.02, 1e-6), "quiet idle: cores() reads the window's rate (0.02)");
    }
    {   // the 2026-09-19 shape: 9 cores idle from the start
        Run r(def); const auto due = r.go(1800, 9.0);
        check(due.size() == 3 && near(due[0], 60) && near(due[1], 660) && near(due[2], 1260),
              "9 cores idle: reported at 60 s (the first full window), then 660 and 1260 (every 600 s)");
        check(r.d.spinning() && near(r.d.cores(), 9.0, 1e-6) && near(r.d.spin_seconds(), 1800), "9 cores idle: spinning, 9.00 cores, 1800 s at t = 1800");
        check(r.d.reports() == 3, "the report counter says 3");
    }
    {   // before a full window nothing is reported, however hot
        Run r(def); const auto due = r.go(55, 20.0);
        check(due.empty() && near(r.d.cores(), 0) && !r.d.spinning(), "20 cores for 55 s: no report (the window is not full)");
    }
    {   // a request in the middle restarts the window
        Run r(def); const auto due = r.go(200, 9.0, {30});
        check(!due.empty() && near(due[0], 90), "a request at 30 s: the first report moves to 90 s");
    }
    {   // a request that came and went between samples (the server marks that interval busy) resets a running spin
        Run r(def); auto due = r.go(120, 9.0);
        check(due.size() == 1 && r.d.spinning(), "spinning at 120 s");
        due = r.go(125, 9.0, {125});
        check(due.empty() && !r.d.spinning() && near(r.d.spin_seconds(), 0) && near(r.d.cores(), 0), "a busy interval ends the spin state");
        due = r.go(185, 9.0);
        check(due.size() == 1 && near(due[0], 185), "... and the next full idle window reports again at once (185 s)");
    }
    {   // the threshold is inclusive: exactly 2 cores reports, 1.99 does not
        Run a(def); check(!a.go(120, 2.0).empty(), "exactly 2.00 cores reports");
        Run b(def); check(b.go(120, 1.99).empty(), "1.99 cores does not");
    }
    {   // a spin that stops, then starts again: the second one is reported without waiting out repeat_s
        Run r(def); auto due = r.go(100, 9.0);
        check(due.size() == 1, "spin A reported");
        due = r.go(200, 0.0);
        check(due.empty() && !r.d.spinning(), "spin A ends: the window's average falls under the threshold");
        due = r.go(400, 9.0);
        check(!due.empty() && due[0] < 300, "spin B is reported within a window of starting (not 600 s after A)");
    }
    {   // samples at irregular times: the rate is taken over the real span
        ie::IdleSpinDetector d(def, kTps);
        d.update(0, 0, true); d.update(7, 3 * 700, true); d.update(40, 3 * 4000, true);
        const bool due = d.update(61, 3 * 6100, true);
        check(due && near(d.cores(), 3.0, 1e-6) && near(d.spin_seconds(), 61), "irregular samples (0, 7, 40, 61 s) at 3 cores: reported, 3.00 cores");
    }
    {   // disabled: nothing, ever
        ie::IdleSpinConfig off = def; off.enabled = false;
        Run r(off); check(r.go(1800, 16.0).empty() && !r.d.spinning(), "disabled: 16 cores for 30 min report nothing");
    }
    {   // threshold 0: any full idle window reports (the plumbing check of the runbook)
        ie::IdleSpinConfig any = def; any.min_cores = 0;
        Run r(any); const auto due = r.go(700, 0.0);
        check(due.size() == 2 && near(due[0], 60) && near(due[1], 660), "threshold 0, a silent process: reported at 60 s and 660 s");
    }
    {   // a counter that goes backwards (it should not) is a zero delta, not a wrap
        ie::IdleSpinDetector d(def, kTps);
        d.update(0, 1000000, true); d.update(30, 10, true);
        const bool due = d.update(60, 20, true);
        check(!due && near(d.cores(), 0.0), "a CPU counter that went backwards reads 0 cores");
    }
}

void proc_tests() {
    ie::ThreadCpu t;
    check(ie::parse_proc_stat("4242 (ie) S 1 4242 4242 0 -1 4194560 12 0 3 0 1234 567 0 0 20 0 340 0 99 1 2 3", t) &&
          t.tid == 4242 && t.comm == "ie" && t.state == 'S' && t.utime == 1234 && t.stime == 567, "a stat line: tid, comm, state, utime, stime");
    check(ie::parse_proc_stat("77 (a) b) c) R 1 2 3 4 5 6 7 8 9 10 11 12 13", t) && t.comm == "a) b) c" && t.state == 'R' && t.utime == 11 && t.stime == 12,
          "a comm holding spaces and ')' (the last ')' ends it)");
    check(!ie::parse_proc_stat("", t), "an empty line is refused");
    check(!ie::parse_proc_stat("12 (x) R 1 2 3", t), "a truncated line is refused");
    check(!ie::parse_proc_stat("12 (x R 1 2 3 4 5 6 7 8 9 10 11 12 13", t), "a line with no ')' is refused");

    // the deltas: sorted, bounded, a new thread counted whole, idle threads left out, no baseline -> nothing
    const std::vector<ie::ThreadCpu> before = {{10, "ie", 'S', 100, 10}, {11, "ie", 'R', 500, 0}, {12, "omp", 'R', 50, 50}, {13, "ie", 'S', 7, 7}};
    const std::vector<ie::ThreadCpu> after = {{10, "ie", 'S', 100, 10}, {11, "ie", 'R', 950, 0}, {12, "omp", 'R', 300, 300}, {13, "ie", 'S', 8, 7}, {14, "new", 'R', 200, 0}};
    auto top = ie::top_thread_deltas(before, after, 5.0, kTps, 3);
    check(top.size() == 3 && top[0].tid == 12 && top[1].tid == 11 && top[2].tid == 14, "top 3 by CPU: 12 (1.00 cores), 11 (0.90), 14 (0.40)");
    check(near(top[0].cores, 1.0, 1e-9) && near(top[0].usr, 0.5, 1e-9) && near(top[0].sys, 0.5, 1e-9), "thread 12: 1.00 cores = usr 0.50 + sys 0.50");
    check(near(top[2].cores, 0.4, 1e-9), "a thread born in the interval counts all its CPU (0.40 cores)");
    top = ie::top_thread_deltas(before, after, 5.0, kTps, 10);
    check(top.size() == 4, "the idle thread (10) is left out");
    check(ie::top_thread_deltas({}, after, 5.0, kTps, 5).empty(), "no baseline snapshot: no deltas");

    // the real /proc of this process: spin one thread for 400 ms and find it
    const uint64_t c0 = ie::proc_self_cpu_ticks();
    size_t more = 0;
    const auto snap0 = ie::proc_self_threads(4096, &more);
    std::atomic<int> spin_tid{0}; std::atomic<bool> stop{false};
    std::thread spinner([&] {
        spin_tid = ie::current_tid();
        volatile uint64_t x = 0;
        while (!stop.load(std::memory_order_relaxed)) x = x + 1;
    });
    while (!spin_tid.load()) std::this_thread::yield();
    const auto snap1 = ie::proc_self_threads(4096, &more);
    const auto t0 = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    const auto snap2 = ie::proc_self_threads(4096, &more);
    const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    stop = true; spinner.join();
    const uint64_t c1 = ie::proc_self_cpu_ticks();
    const double tps = double(sysconf(_SC_CLK_TCK));
    check(!snap0.empty() && more == 0, "proc_self_threads reads this process's threads (" + std::to_string(snap0.size()) + ")");
    check(snap2.size() == snap0.size() + 1, "... and sees the spinner as one more thread");
    check(c1 > c0, "proc_self_cpu_ticks grows while a thread spins (" + std::to_string(c1 - c0) + " ticks)");
    const auto real = ie::top_thread_deltas(snap1, snap2, dt, tps, 5);
    check(!real.empty() && real[0].tid == spin_tid.load() && real[0].cores > 0.25,
          "the spinner is the busiest thread (" + (real.empty() ? std::string("none") : std::to_string(real[0].cores)) + " cores over " + std::to_string(dt) + " s)");
    const auto bounded = ie::proc_self_threads(1, &more);
    check(bounded.size() == 1 && more == snap0.size() - 1, "the thread read is bounded: 1 read, the rest counted");
    check(!ie::proc_self_thread_wchan(ie::current_tid()).empty(), "wchan of this thread is readable");

    // the report line
    ie::IdleSpinConfig cfg; ie::IdleSpinDetector d(cfg, kTps);
    d.update(0, 0, true); d.update(30, 27000, true); d.update(65, 58500, true);
    std::vector<ie::ThreadDelta> td(2);
    td[0].tid = 501; td[0].comm = "ie"; td[0].state = 'R'; td[0].cores = 0.99; td[0].usr = 0.40; td[0].sys = 0.59; td[0].http = true; td[0].wchan = "0";
    td[1].tid = 502; td[1].comm = "ie"; td[1].state = 'R'; td[1].cores = 0.98; td[1].usr = 0.98; td[1].sys = 0.0; td[1].wchan = "0";
    const std::vector<ie::ThreadCpu> names = {{1, "ie", 'S', 0, 0}, {2, "ie", 'S', 0, 0}, {3, "ze_worker", 'S', 0, 0}};
    const std::string line = ie::idle_spin_report(d, td, 5.0, true, 342, names, 20, 4242);
    std::printf("       %s\n", line.c_str());
    check(line.rfind("[ie] idle spin: 9.00 cores for 65 s", 0) == 0, "the line opens with the rate and the duration");
    check(line.find("threshold 2.00 cores over 60 s") != std::string::npos && line.find("report 1, again every 600 s") != std::string::npos, "... the threshold and the repeat");
    check(line.find("342 threads (ie x2, ze_worker x1), OpenMP default team 20") != std::string::npos, "... the thread count, the names and the OpenMP team");
    check(line.find("tid 501 \"ie\" R 0.99 cores (usr 0.40 sys 0.59) wchan 0 [http request thread]") != std::string::npos, "... each busy thread with usr/sys, wchan and the HTTP tag");
    check(line.find("eu-stack -p 4242") != std::string::npos, "... and the command for the stacks");
    check(line.find('\n') == std::string::npos, "one line");
    // an empty busy list: with a baseline it means no thread had CPU in the interval, not that the baseline is missing
    const std::string quiet = ie::idle_spin_report(d, {}, 5.0, true, 342, names, 20, 4242);
    check(quiet.find("(no thread had CPU time in the interval)") != std::string::npos && quiet.find("baseline") == std::string::npos,
          "an empty busy list with a baseline: \"no thread had CPU time in the interval\"");
    const std::string first = ie::idle_spin_report(d, {}, 5.0, false, 342, names, 20, 4242);
    check(first.find("(no per-thread baseline yet)") != std::string::npos, "... and without one: \"no per-thread baseline yet\"");
}

}  // namespace

int main() {
    config_tests();
    detector_tests();
    proc_tests();
    std::printf("\nIDLE SPIN: %s\n", g_fail ? "FAILURE(S)" : "PASS");
    return g_fail ? 1 : 0;
}
