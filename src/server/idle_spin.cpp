// src/server/idle_spin.cpp — see include/ie/idle_spin.hpp (#29, docs/server_idle_spin_watchdog_2026-09-24.md).
#include "ie/idle_spin.hpp"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <unordered_map>

#include <dirent.h>
#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace ie {

IdleSpinConfig idle_spin_config(const char* watchdog_env, const char* cores_env, std::string* warn) {
    IdleSpinConfig cfg;
    if (watchdog_env && std::string_view(watchdog_env) == "0") cfg.enabled = false;
    if (cores_env && *cores_env) {
        char* end = nullptr;
        errno = 0;
        const double v = std::strtod(cores_env, &end);
        if (end == cores_env || *end != '\0' || errno == ERANGE || !std::isfinite(v) || v < 0.0) {
            if (warn) {
                char b[160];
                std::snprintf(b, sizeof b, "IE_IDLE_SPIN_CORES=\"%.40s\" is not a CPU rate in cores (>= 0): the idle-spin watchdog keeps %.2f",
                              cores_env, cfg.min_cores);
                *warn = b;
            }
        } else cfg.min_cores = v;
    }
    return cfg;
}

bool IdleSpinDetector::update(double now_s, uint64_t cpu_ticks, bool idle) {
    last_t_ = now_s;
    if (!cfg_.enabled) return false;
    if (!idle || (!idle_.empty() && now_s <= idle_.back().t)) {
        // activity since the previous sample (or a clock that did not advance): an idle window starts again from here
        idle_.clear(); idle_.push_back({now_s, cpu_ticks});
        spinning_ = false; cores_ = 0;
        return false;
    }
    idle_.push_back({now_s, cpu_ticks});
    // the window's start: the LATEST sample at least window_s old, so the window is window_s .. window_s + one period
    while (idle_.size() >= 2 && idle_[1].t <= now_s - cfg_.window_s) idle_.pop_front();
    const Sample& a = idle_.front();
    const double dt = now_s - a.t;
    if (dt < cfg_.window_s) { spinning_ = false; cores_ = 0; return false; }   // not idle for a whole window yet
    cores_ = double(cpu_ticks >= a.ticks ? cpu_ticks - a.ticks : 0) / tps_ / dt;
    if (cores_ < cfg_.min_cores) { spinning_ = false; return false; }
    if (!spinning_) { spinning_ = true; spin_start_ = a.t; last_report_ = now_s; ++reports_; return true; }
    if (now_s - last_report_ >= cfg_.repeat_s) { last_report_ = now_s; ++reports_; return true; }
    return false;
}

bool parse_proc_stat(std::string_view line, ThreadCpu& out) {
    const size_t a = line.find('('), b = line.rfind(')');
    if (a == std::string_view::npos || b == std::string_view::npos || b < a) return false;
    std::string_view pid = line.substr(0, a);
    while (!pid.empty() && pid.back() == ' ') pid.remove_suffix(1);
    int tid = 0;
    if (std::from_chars(pid.data(), pid.data() + pid.size(), tid).ec != std::errc{}) return false;
    // after the comm: state(3) ppid pgrp session tty_nr tpgid flags minflt cminflt majflt cmajflt utime(14) stime(15)
    std::string_view f[13]; size_t n = 0, i = b + 1;
    while (n < 13) {
        while (i < line.size() && line[i] == ' ') ++i;
        if (i >= line.size()) return false;
        const size_t j = std::min(line.find(' ', i), line.size());
        f[n++] = line.substr(i, j - i); i = j;
    }
    uint64_t ut = 0, st = 0;
    if (std::from_chars(f[11].data(), f[11].data() + f[11].size(), ut).ec != std::errc{} ||
        std::from_chars(f[12].data(), f[12].data() + f[12].size(), st).ec != std::errc{}) return false;
    out.tid = tid; out.comm = std::string(line.substr(a + 1, b - a - 1)); out.state = f[0].empty() ? '?' : f[0][0];
    out.utime = ut; out.stime = st;
    return true;
}

namespace {
// a small /proc file's contents (a stat line is ~300 bytes); "" when it cannot be read
std::string read_small(const char* path) {
    const int fd = ::open(path, O_RDONLY | O_CLOEXEC);
    if (fd < 0) return {};
    char buf[1024]; std::string s;
    for (;;) {
        const ssize_t r = ::read(fd, buf, sizeof buf);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) break;
        s.append(buf, size_t(r));
        if (s.size() > 8192) break;
    }
    ::close(fd);
    while (!s.empty() && (s.back() == '\n' || s.back() == '\0')) s.pop_back();
    return s;
}
}  // namespace

uint64_t proc_self_cpu_ticks() {
    ThreadCpu t;
    return parse_proc_stat(read_small("/proc/self/stat"), t) ? t.utime + t.stime : 0;
}

std::vector<ThreadCpu> proc_self_threads(size_t max_threads, size_t* more) {
    std::vector<ThreadCpu> out; size_t extra = 0;
    if (DIR* d = ::opendir("/proc/self/task")) {
        while (const dirent* e = ::readdir(d)) {
            if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
            if (out.size() >= max_threads) { ++extra; continue; }
            const std::string path = std::string("/proc/self/task/") + e->d_name + "/stat";
            ThreadCpu t;
            if (parse_proc_stat(read_small(path.c_str()), t)) out.push_back(std::move(t));   // a thread gone meanwhile is skipped
        }
        ::closedir(d);
    }
    if (more) *more = extra;
    return out;
}

std::string proc_self_thread_wchan(int tid) {
    char path[96];
    std::snprintf(path, sizeof path, "/proc/self/task/%d/wchan", tid);
    std::string s = read_small(path);
    return s.empty() ? "?" : s;
}

int current_tid() { return static_cast<int>(::syscall(SYS_gettid)); }

std::vector<ThreadDelta> top_thread_deltas(const std::vector<ThreadCpu>& before, const std::vector<ThreadCpu>& after,
                                           double interval_s, double ticks_per_s, size_t top_n) {
    std::vector<ThreadDelta> out;
    if (before.empty() || interval_s <= 0 || ticks_per_s <= 0) return out;
    std::unordered_map<int, const ThreadCpu*> prev;
    for (const auto& t : before) prev[t.tid] = &t;
    const double k = 1.0 / (ticks_per_s * interval_s);
    for (const auto& t : after) {
        const auto it = prev.find(t.tid);
        const uint64_t u0 = it == prev.end() ? 0 : it->second->utime, s0 = it == prev.end() ? 0 : it->second->stime;
        const uint64_t du = t.utime >= u0 ? t.utime - u0 : 0, ds = t.stime >= s0 ? t.stime - s0 : 0;
        if (du + ds == 0) continue;
        ThreadDelta d; d.tid = t.tid; d.comm = t.comm; d.state = t.state;
        d.usr = double(du) * k; d.sys = double(ds) * k; d.cores = d.usr + d.sys;
        out.push_back(std::move(d));
    }
    std::sort(out.begin(), out.end(), [](const ThreadDelta& x, const ThreadDelta& y) { return x.cores != y.cores ? x.cores > y.cores : x.tid < y.tid; });
    if (out.size() > top_n) out.resize(top_n);
    return out;
}

std::string idle_spin_report(const IdleSpinDetector& d, const std::vector<ThreadDelta>& top, double interval_s,
                             size_t threads, const std::vector<ThreadCpu>& comms, int omp_default_team, int pid) {
    const IdleSpinConfig& c = d.config();
    char b[512];
    std::snprintf(b, sizeof b, "[ie] idle spin: %.2f cores for %.0f s with nothing in flight, queued or admitted (threshold %.2f cores over "
                               "%.0f s; report %llu, again every %.0f s while it lasts; IE_IDLE_SPIN_WATCHDOG=0 turns this off) | %zu threads",
                  d.cores(), d.spin_seconds(), c.min_cores, c.window_s, static_cast<unsigned long long>(d.reports()), c.repeat_s, threads);
    std::string s = b;
    // the threads by name, the most common first (at most 3 names)
    std::map<std::string, size_t> by_name;
    for (const auto& t : comms) ++by_name[t.comm];
    std::vector<std::pair<size_t, std::string>> names;
    for (const auto& [n, k] : by_name) names.push_back({k, n});
    std::sort(names.begin(), names.end(), [](const auto& x, const auto& y) { return x.first != y.first ? x.first > y.first : x.second < y.second; });
    for (size_t i = 0; i < names.size() && i < 3; ++i) s += (i ? ", " : " (") + names[i].second + " x" + std::to_string(names[i].first);
    if (!names.empty()) s += names.size() > 3 ? ", ...)" : ")";
    std::snprintf(b, sizeof b, ", OpenMP default team %d | busiest over the last %.1f s:", omp_default_team, interval_s);
    s += b;
    if (top.empty()) s += " (no per-thread baseline yet)";
    for (size_t i = 0; i < top.size(); ++i) {
        const auto& t = top[i];
        std::snprintf(b, sizeof b, "%s tid %d \"%.32s\" %c %.2f cores (usr %.2f sys %.2f) wchan %.40s%s", i ? ";" : "", t.tid, t.comm.c_str(),
                      t.state, t.cores, t.usr, t.sys, t.wchan.empty() ? "?" : t.wchan.c_str(), t.http ? " [http request thread]" : "");
        s += b;
    }
    std::snprintf(b, sizeof b, " | stacks: eu-stack -p %d (or gdb -p %d -batch -ex 'thread apply all bt')", pid, pid);
    s += b;
    return s;
}

}  // namespace ie
