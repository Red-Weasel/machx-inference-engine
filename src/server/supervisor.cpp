// ie supervise: the supervisor and model router (ie/supervisor.hpp, docs/serve_config.md "Supervisor").
#include "ie/supervisor.hpp"
#include "ie/card_lock.hpp"
#include "httplib/httplib.h"
#include "nlohmann/json.hpp"

#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdarg>
#include <fcntl.h>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <set>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

extern char** environ;

namespace ie::sup {
using nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {
std::string err_json(const std::string& msg, const std::string& type, const std::string& code) {
    json e{{"message", msg}, {"type", type}};
    if (!code.empty()) e["code"] = code;
    return json{{"error", e}}.dump(-1, ' ', false, json::error_handler_t::replace);
}
std::string names_of(const std::vector<ServerSpec>& servers) {
    std::string s;
    for (const auto& v : servers) s += (s.empty() ? "" : ", ") + v.name;
    return s;
}
double secs_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }
void log(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void log(const char* fmt, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "[ie supervise] %s\n", buf);
}
}  // namespace

// ---- routing -----------------------------------------------------------------------------------------------------------
Route route_model(const std::vector<ServerSpec>& servers, int default_index, const std::string* model) {
    Route r;
    if (!model) {
        if (default_index >= 0 && default_index < int(servers.size())) { r.index = default_index; return r; }
        r.status = 400;
        r.error = err_json("the request has no \"model\" and the layout names no \"default\" server; available models: " +
                           names_of(servers), "invalid_request_error", "model_required");
        return r;
    }
    for (size_t i = 0; i < servers.size(); ++i)
        if (servers[i].name == *model) { r.index = int(i); return r; }
    std::vector<int> by_id;
    for (size_t i = 0; i < servers.size(); ++i)
        if (!servers[i].model_id.empty() && servers[i].model_id == *model) by_id.push_back(int(i));
    if (by_id.size() == 1) { r.index = by_id[0]; return r; }
    if (by_id.size() > 1) {
        std::string which;
        for (int i : by_id) which += (which.empty() ? "" : ", ") + servers[size_t(i)].name;
        r.status = 400;
        r.error = err_json("model \"" + *model + "\" is served by several servers (" + which + "); ask for one by name",
                           "invalid_request_error", "model_ambiguous");
        return r;
    }
    r.status = 404;
    r.error = err_json("model \"" + *model + "\" not found; available models: " + names_of(servers),
                       "invalid_request_error", "model_not_found");
    return r;
}

Route route_request(const std::vector<ServerSpec>& servers, int default_index, const std::string& body) {
    json j = json::parse(body, nullptr, false);
    if (j.is_discarded() || !j.is_object()) {
        Route r;
        r.status = 400;
        r.error = err_json("the request body must be a JSON object", "invalid_request_error", "invalid_json");
        return r;
    }
    const auto it = j.find("model");
    if (it == j.end() || it->is_null()) return route_model(servers, default_index, nullptr);
    if (!it->is_string()) {
        Route r;
        r.status = 400;
        r.error = err_json("\"model\" must be a string", "invalid_request_error", "invalid_model");
        return r;
    }
    const std::string m = it->get<std::string>();
    return route_model(servers, default_index, &m);
}

const char* state_name(ChildState s) {
    switch (s) {
    case ChildState::Loading: return "loading";
    case ChildState::Ready: return "ready";
    case ChildState::Unhealthy: return "unhealthy";
    case ChildState::Stopping: return "stopping";
    case ChildState::Exited: return "exited";
    case ChildState::Stopped: return "stopped";
    case ChildState::LeftRunning: return "left_running";
    }
    return "?";
}

std::pair<int, std::string> aggregate_health(const std::vector<ServerSpec>& servers, const std::vector<ChildView>& views,
                                             bool stopping, int default_index) {
    json out_servers = json::object();
    size_t ready = 0, loading = 0;
    uint64_t inflight = 0, queued = 0;
    for (size_t i = 0; i < servers.size() && i < views.size(); ++i) {
        const auto& v = views[i];
        json s{{"state", state_name(v.state)}, {"model", servers[i].model_id}, {"port", servers[i].port},
               {"cards", servers[i].cards}, {"pid", v.pid}, {"restarts", v.restarts}};
        if (v.state == ChildState::Loading) s["loading_s"] = std::llround(v.since_s);
        if (v.state == ChildState::Exited || v.state == ChildState::Stopped) {
            if (v.exit_signal) s["exit_signal"] = v.exit_signal;
            else s["exit_code"] = v.exit_code;
        }
        if (!v.health.empty()) {
            json h = json::parse(v.health, nullptr, false);
            if (!h.is_discarded()) {
                s["health"] = h;
                if (h.contains("inflight") && h["inflight"].is_number_unsigned()) inflight += h["inflight"].get<uint64_t>();
                if (h.contains("queued") && h["queued"].is_number_unsigned()) queued += h["queued"].get<uint64_t>();
            }
        }
        if (v.state == ChildState::Ready) ++ready;
        if (v.state == ChildState::Loading) ++loading;
        out_servers[servers[i].name] = s;
    }
    std::string status;
    int code = 503;
    if (stopping) status = "stopping";
    else if (ready == servers.size()) { status = "ok"; code = 200; }
    else if (ready > 0) { status = "degraded"; code = 200; }
    else if (loading > 0) status = "loading";
    else status = "unavailable";
    json out{{"status", status}, {"inflight", inflight}, {"queued", queued}, {"servers", out_servers}};
    if (default_index >= 0 && default_index < int(servers.size())) out["default"] = servers[size_t(default_index)].name;
    return {code, out.dump(-1, ' ', false, json::error_handler_t::replace)};
}

std::string models_list_json(const std::vector<ServerSpec>& servers, const std::vector<ChildView>& views) {
    json data = json::array();
    for (size_t i = 0; i < servers.size(); ++i)
        data.push_back({{"id", servers[i].name}, {"object", "model"}, {"owned_by", "local"},
                        {"root", servers[i].model_id},
                        {"status", i < views.size() ? state_name(views[i].state) : "unknown"}});
    return json{{"object", "list"}, {"data", data}}.dump(-1, ' ', false, json::error_handler_t::replace);
}

// ---- the supervisor ----------------------------------------------------------------------------------------------------
namespace {
std::atomic<int> g_signal{0};
void on_signal(int sig) {
    if (g_signal.exchange(sig) != 0) {   // second signal: exit now, children keep running
        static const char msg[] = "[ie supervise] second signal: exiting now; the children are left running\n";
        ssize_t w = ::write(2, msg, sizeof msg - 1);
        (void)w;
        _exit(128 + sig);
    }
}

struct Child {
    ServerSpec spec;
    pid_t pid = -1;
    ChildState state = ChildState::Loading;
    int exit_code = -1, exit_signal = 0, restarts = 0;
    bool stop_requested = false;
    bool unhealthy_logged = false;
    std::string health;
    Clock::time_point started;
};

// One upstream exchange streamed to the front: the client thread fills it, the front's content provider drains it.
struct Pipe {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> chunks;
    bool headers = false, done = false, cancel = false, failed = false;
    int status = 0;
    httplib::Headers hdrs;
    std::string error;
    std::shared_ptr<httplib::Client> cli;
    void cancel_now() {
        std::shared_ptr<httplib::Client> c;
        {
            std::lock_guard<std::mutex> lk(mu);
            cancel = true;
            c = cli;
        }
        if (c) c->stop();   // closes the upstream socket: the child sees the client gone
        cv.notify_all();
    }
};

// Headers the front sets itself (framing, and the content type passed with the body).
bool front_sets(const std::string& k) {
    std::string l;
    for (char c : k) l += char(std::tolower(static_cast<unsigned char>(c)));
    return l == "content-length" || l == "transfer-encoding" || l == "connection" || l == "keep-alive" ||
           l == "content-type";
}

class Supervisor {
public:
    Supervisor(const std::vector<ServerSpec>& servers, const Options& opt) : opt_(opt) {
        for (const auto& s : servers) { children_.emplace_back(); children_.back().spec = s; specs_.push_back(s); }
    }

    int run() {
        // Card locks first: nothing is started when a card is taken.
        if (opt_.take_card_locks) {
            std::vector<uint32_t> cards;
            for (const auto& s : specs_) cards.insert(cards.end(), s.cards.begin(), s.cards.end());
            if (auto e = acquire_card_locks(cards, locks_); !e.empty()) { log("refusing to start: %s", e.c_str()); return 2; }
        }
        unsigned pool = 8;
        for (const auto& s : specs_) pool += s.slots;
        srv_.new_task_queue = [pool] { return new httplib::ThreadPool(pool); };
        install_routes();
        if (!srv_.bind_to_port(opt_.host, opt_.port)) {
            log("cannot bind the front endpoint %s:%d; nothing was started", opt_.host.c_str(), opt_.port);
            return 2;
        }
        struct sigaction sa {};
        sa.sa_handler = on_signal;
        sigemptyset(&sa.sa_mask);
        struct sigaction old_term {}, old_int {};
        sigaction(SIGTERM, &sa, &old_term);
        sigaction(SIGINT, &sa, &old_int);
        {
            std::lock_guard<std::mutex> lk(mu_);
            for (auto& c : children_) spawn(c);
        }
        log("front endpoint http://%s:%d/v1 routing %zu server(s): %s", opt_.host.c_str(), opt_.port, specs_.size(),
            names_of(specs_).c_str());

        std::thread monitor([this] { monitor_loop(); });
        std::thread watcher([this] {
            while (!listener_done_.load()) {
                if (g_signal.load() || admin_stop_.load()) {
                    shutdown(g_signal.load() ? "signal " + std::to_string(g_signal.load()) : std::string("admin shutdown"));
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
        srv_.listen_after_bind();
        listener_done_.store(true);
        watcher.join();
        monitor_stop_.store(true);
        monitor.join();
        sigaction(SIGTERM, &old_term, nullptr);
        sigaction(SIGINT, &old_int, nullptr);
        g_signal.store(0);

        // The exit code comes from the processes, not the state enum: a child whose pid has not been reaped is alive.
        int rc = 0, n_bad = 0;
        std::string alive, bad;
        std::lock_guard<std::mutex> lk(mu_);
        for (auto& c : children_) {
            if (c.pid > 0) {
                int st = 0;
                if (::waitpid(c.pid, &st, WNOHANG) == c.pid) {
                    c.exit_code = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
                    c.exit_signal = WIFSIGNALED(st) ? WTERMSIG(st) : 0;
                    c.pid = -1;
                } else {
                    rc = 1;
                    alive += (alive.empty() ? "" : ", ") + c.spec.name + " (pid " + std::to_string(c.pid) + ")";
                    continue;
                }
            }
            // Any exit other than code 0 is a failure, requested stop or not (a crash during teardown included).
            if (c.exit_signal != 0 || c.exit_code != 0) {
                rc = 1;
                ++n_bad;
                bad += (bad.empty() ? "" : ", ") + c.spec.name + " (" +
                       (c.exit_signal ? "signal " + std::to_string(c.exit_signal)
                                      : "exit code " + std::to_string(c.exit_code) +
                                            (c.exit_code == kDeviceLostExit ? ", device lost" : "")) + ")";
            }
        }
        if (!alive.empty()) log("stopped; STILL RUNNING (not killed): %s", alive.c_str());
        if (n_bad) log("stopped; %d child(ren) did not stop cleanly: %s", n_bad, bad.c_str());
        if (alive.empty() && !n_bad) log("stopped (every child stopped cleanly)");
        return rc;
    }

private:
    Options opt_;
    std::vector<ServerSpec> specs_;
    std::vector<Child> children_;   // guarded by mu_
    std::mutex mu_;
    CardLocks locks_;
    httplib::Server srv_;
    std::atomic<bool> stopping_{false}, admin_stop_{false}, listener_done_{false}, monitor_stop_{false};
    std::once_flag shutdown_once_;

    std::vector<ChildView> views() {
        std::lock_guard<std::mutex> lk(mu_);
        std::vector<ChildView> v;
        for (const auto& c : children_) {
            ChildView cv;
            cv.state = c.state; cv.pid = c.pid; cv.exit_code = c.exit_code; cv.exit_signal = c.exit_signal;
            cv.restarts = c.restarts; cv.health = c.health; cv.since_s = secs_since(c.started);
            v.push_back(cv);
        }
        return v;
    }

    // Start the child (mu_ held). Everything the child needs is built before fork: after fork only exec-safe calls.
    void spawn(Child& c) {
        std::vector<std::string> env;
        auto overridden = [&](const std::string& entry) {
            const auto eq = entry.find('=');
            const std::string k = entry.substr(0, eq);
            if (k == "IE_GPU_LOCK_HELD") return true;
            for (const auto& [n, v] : c.spec.env) if (n == k) return true;
            return false;
        };
        for (char** e = environ; e && *e; ++e) if (!overridden(*e)) env.emplace_back(*e);
        for (const auto& [n, v] : c.spec.env) env.push_back(n + "=" + v);
        std::string held;
        for (uint32_t card : c.spec.cards) held += (held.empty() ? "" : ",") + std::to_string(card);
        if (!held.empty()) env.push_back("IE_GPU_LOCK_HELD=" + held);
        std::vector<char*> av, ev;
        for (auto& a : c.spec.argv) av.push_back(const_cast<char*>(a.c_str()));
        av.push_back(nullptr);
        for (auto& e : env) ev.push_back(const_cast<char*>(e.c_str()));
        ev.push_back(nullptr);
        // The child's stdout+stderr go through a pipe; a reader thread writes each line to our stderr as "[name] line".
        // (one pipe per stream, so a partial stdout line is never joined with a stderr line)
        int out[2] = {-1, -1}, errp[2] = {-1, -1};
        if (::pipe2(out, O_CLOEXEC) != 0) { out[0] = out[1] = -1; }
        if (::pipe2(errp, O_CLOEXEC) != 0) { errp[0] = errp[1] = -1; }
        std::vector<int> keep;   // this child's card-lock descriptors: inherited, so the lock outlives the supervisor
        for (const auto& [card, fd] : locks_.fds())
            for (uint32_t mine : c.spec.cards) if (mine == card) keep.push_back(fd);

        const pid_t pid = ::fork();
        if (pid == 0) {
            ::setpgid(0, 0);   // its own process group: a terminal ^C reaches the supervisor only
            for (int fd : keep) ::fcntl(fd, F_SETFD, 0);
            if (out[1] >= 0) ::dup2(out[1], 1);
            if (errp[1] >= 0) ::dup2(errp[1], 2);
            // SIGPIPE ignored (inherited across exec): if the supervisor dies, the child's log writes fail with EPIPE
            // instead of killing a process that holds a model (ie serve's HTTP server ignores SIGPIPE anyway).
            struct sigaction ign {};
            ign.sa_handler = SIG_IGN;
            sigemptyset(&ign.sa_mask);
            ::sigaction(SIGPIPE, &ign, nullptr);
            sigset_t none;
            sigemptyset(&none);
            ::sigprocmask(SIG_SETMASK, &none, nullptr);
            ::execve(av[0], av.data(), ev.data());
            static const char msg[] = "[ie supervise] exec failed\n";
            ssize_t w = ::write(2, msg, sizeof msg - 1);
            (void)w;
            _exit(127);
        }
        for (int* pp : {out, errp}) {
            if (pp[1] >= 0) ::close(pp[1]);
            if (pp[0] < 0) continue;
            if (pid > 0) {
                std::thread([fd = pp[0], name = c.spec.name] {
                    std::string buf;
                    char chunk[4096];
                    auto emit = [&](const std::string& line) {
                        const std::string l = "[" + name + "] " + line + "\n";
                        ssize_t w = ::write(2, l.data(), l.size());
                        (void)w;
                    };
                    for (;;) {
                        const ssize_t n = ::read(fd, chunk, sizeof chunk);
                        if (n < 0 && errno == EINTR) continue;
                        if (n <= 0) break;
                        buf.append(chunk, size_t(n));
                        size_t nl;
                        while ((nl = buf.find('\n')) != std::string::npos) { emit(buf.substr(0, nl)); buf.erase(0, nl + 1); }
                    }
                    if (!buf.empty()) emit(buf);
                    ::close(fd);
                }).detach();
            } else ::close(pp[0]);
        }
        c.started = Clock::now();
        c.health.clear();
        c.unhealthy_logged = false;
        c.stop_requested = false;
        if (pid < 0) {
            log("server \"%s\": fork failed: %s", c.spec.name.c_str(), std::strerror(errno));
            c.pid = -1;
            c.state = ChildState::Exited;
            return;
        }
        c.pid = pid;
        c.state = ChildState::Loading;
        std::string cmd;
        for (const auto& a : c.spec.argv) cmd += (cmd.empty() ? "" : " ") + a;
        log("server \"%s\" (cards %s, port %d): started pid %d: %s", c.spec.name.c_str(), held.c_str(), c.spec.port,
            int(pid), cmd.c_str());
    }

    void monitor_loop() {
        while (!monitor_stop_.load()) {
            for (size_t i = 0; i < children_.size(); ++i) poll_child(i);
            const auto until = Clock::now() + std::chrono::duration<double>(opt_.poll_s);
            while (!monitor_stop_.load() && Clock::now() < until) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    void poll_child(size_t i) {
        std::string host;
        int port = 0;
        pid_t pid;
        {
            std::lock_guard<std::mutex> lk(mu_);
            Child& c = children_[i];
            if (c.pid <= 0) return;
            int st = 0;
            const pid_t r = ::waitpid(c.pid, &st, WNOHANG);
            if (r == c.pid) {
                c.exit_code = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
                c.exit_signal = WIFSIGNALED(st) ? WTERMSIG(st) : 0;
                c.pid = -1;
                c.health.clear();
                if (c.state == ChildState::LeftRunning) {   // terminal: it stays reported as left running
                    log("server \"%s\" (pid %d), left running, has now exited (code %d, signal %d)", c.spec.name.c_str(),
                        int(r), c.exit_code, c.exit_signal);
                    return;
                }
                const bool clean = c.stop_requested;
                c.state = clean ? ChildState::Stopped : ChildState::Exited;
                const std::string how = c.exit_signal ? "was killed by signal " + std::to_string(c.exit_signal)
                                                      : "exited with code " + std::to_string(c.exit_code) +
                                                            (c.exit_code == kDeviceLostExit ? " (device lost)" : "");
                if (clean) log("server \"%s\" (pid %d) stopped: %s", c.spec.name.c_str(), int(r), how.c_str());
                else {
                    log("server \"%s\" (pid %d) %s after %.0f s -- it is down; the other servers keep serving",
                        c.spec.name.c_str(), int(r), how.c_str(), secs_since(c.started));
                    const bool failure = c.exit_signal != 0 || c.exit_code != 0;
                    if (c.spec.restart_on_failure && failure && !stopping_.load() && c.exit_code != kDeviceLostExit &&
                        c.restarts < kMaxRestarts) {
                        ++c.restarts;
                        log("server \"%s\": restart %d of %d (\"restart\": \"on-failure\")", c.spec.name.c_str(),
                            c.restarts, kMaxRestarts);
                        spawn(c);
                    } else if (c.spec.restart_on_failure && failure) {
                        log("server \"%s\": not restarted (%s)", c.spec.name.c_str(),
                            c.exit_code == kDeviceLostExit ? "a lost device needs a person" : "restart limit reached");
                    }
                }
                return;
            }
            host = c.spec.connect_host;
            port = c.spec.port;
            pid = c.pid;
        }
        httplib::Client cli(host, port);
        cli.set_connection_timeout(1, 0);
        cli.set_read_timeout(2, 0);
        auto res = cli.Get("/health");
        std::lock_guard<std::mutex> lk(mu_);
        Child& c = children_[i];
        if (c.pid != pid) return;   // exited / restarted meanwhile
        if (c.state == ChildState::LeftRunning) return;   // terminal: never overwritten
        if (!res) return;           // not listening yet (loading) or not answering: keep the state
        c.health = res->body;
        if (res->status == 200) {
            if (c.state == ChildState::Loading)
                log("server \"%s\" (pid %d) ready after %.0f s", c.spec.name.c_str(), int(c.pid), secs_since(c.started));
            if (c.state == ChildState::Unhealthy) log("server \"%s\" healthy again", c.spec.name.c_str());
            c.state = ChildState::Ready;
            c.unhealthy_logged = false;
        } else {
            json h = json::parse(res->body, nullptr, false);
            const std::string st = !h.is_discarded() && h.contains("status") && h["status"].is_string()
                                       ? h["status"].get<std::string>() : std::string();
            if (st == "stopping") c.state = ChildState::Stopping;
            else {
                c.state = ChildState::Unhealthy;
                if (!c.unhealthy_logged)
                    log("server \"%s\" (pid %d) reports unhealthy (HTTP %d): %s -- it needs a restart", c.spec.name.c_str(),
                        int(c.pid), res->status, res->body.c_str());
                c.unhealthy_logged = true;
            }
        }
    }

    // Stop every child: its own /admin/shutdown, then wait. Never a signal.
    void shutdown(const std::string& why) {
        std::call_once(shutdown_once_, [&] {
            stopping_.store(true);
            log("shutting down (%s): stopping %zu server(s) through their /admin/shutdown", why.c_str(), children_.size());
            std::vector<std::thread> ts;
            for (size_t i = 0; i < children_.size(); ++i) ts.emplace_back([this, i] { stop_child(i); });
            for (auto& t : ts) t.join();
            srv_.stop();
        });
    }

    bool child_alive(size_t i, ChildState* st = nullptr) {
        std::lock_guard<std::mutex> lk(mu_);
        if (st) *st = children_[i].state;
        return children_[i].pid > 0;
    }

    void stop_child(size_t i) {
        std::string name, host;
        int port;
        {
            std::lock_guard<std::mutex> lk(mu_);
            name = children_[i].spec.name;
            host = children_[i].spec.connect_host;
            port = children_[i].spec.port;
        }
        ChildState st;
        if (!child_alive(i, &st)) return;
        // A loading child is not interrupted: its listener comes up after the load, then it stops in order.
        if (st == ChildState::Loading) {
            log("server \"%s\" is still loading: waiting up to %.0f s for it to become ready before stopping it",
                name.c_str(), opt_.ready_wait_s);
            const auto t0 = Clock::now();
            while (child_alive(i, &st) && st == ChildState::Loading && secs_since(t0) < opt_.ready_wait_s)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (!child_alive(i)) return;
            if (st == ChildState::Loading) {
                std::lock_guard<std::mutex> lk(mu_);
                children_[i].state = ChildState::LeftRunning;
                log("server \"%s\" (pid %d) did not finish loading within %.0f s: LEFT RUNNING (not killed); stop it "
                    "with POST http://%s:%d/admin/shutdown once it is up", name.c_str(), int(children_[i].pid),
                    opt_.ready_wait_s, host.c_str(), port);
                return;
            }
        }
        {
            std::lock_guard<std::mutex> lk(mu_);
            children_[i].stop_requested = true;
        }
        bool sent = false;
        const auto t0 = Clock::now();
        while (!sent && child_alive(i) && secs_since(t0) < opt_.stop_wait_s) {
            httplib::Client cli(host, port);
            cli.set_connection_timeout(2, 0);
            cli.set_read_timeout(10, 0);
            auto r = cli.Post("/admin/shutdown", "", "application/json");
            if (r && r->status == 200) sent = true;
            else if (child_alive(i)) std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        if (sent) log("server \"%s\": /admin/shutdown accepted, waiting for it to exit", name.c_str());
        while (child_alive(i) && secs_since(t0) < opt_.stop_wait_s)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (child_alive(i)) {
            std::lock_guard<std::mutex> lk(mu_);
            children_[i].state = ChildState::LeftRunning;
            log("server \"%s\" (pid %d) %s within %.0f s: LEFT RUNNING (not killed) -- check it, then POST "
                "http://%s:%d/admin/shutdown", name.c_str(), int(children_[i].pid),
                sent ? "did not exit" : "did not accept /admin/shutdown", opt_.stop_wait_s, host.c_str(), port);
        }
    }

    // The routed server for a request, or an error response already written.
    bool target(int index, httplib::Response& res, std::string& host, int& port) {
        std::lock_guard<std::mutex> lk(mu_);
        const Child& c = children_[size_t(index)];
        host = c.spec.connect_host;
        port = c.spec.port;
        switch (c.state) {
        case ChildState::Ready:
        case ChildState::Unhealthy:   // its own 503 says why
            return true;
        case ChildState::Loading:
            res.status = 503;
            res.set_header("Retry-After", "5");
            res.set_content(err_json("model \"" + c.spec.name + "\" is still loading", "server_error", "loading"),
                            "application/json");
            return false;
        default:
            res.status = 503;
            res.set_content(err_json("model \"" + c.spec.name + "\" is not running (" + state_name(c.state) +
                                     (c.exit_signal ? ", signal " + std::to_string(c.exit_signal)
                                                    : c.exit_code >= 0 ? ", exit code " + std::to_string(c.exit_code) : std::string()) +
                                     ")", "server_error", "server_not_running"), "application/json");
            return false;
        }
    }

    // Forward the request to the child and stream its response back unchanged.
    void proxy_post(const std::string& host, int port, const httplib::Request& req, httplib::Response& res) {
        auto p = std::make_shared<Pipe>();
        p->cli = std::make_shared<httplib::Client>(host, port);
        p->cli->set_connection_timeout(5, 0);
        p->cli->set_read_timeout(24 * 3600, 0);    // a long prefill sends nothing for minutes
        p->cli->set_write_timeout(600, 0);
        auto up = std::make_shared<httplib::Request>();
        up->method = "POST";
        up->path = req.path;
        up->body = req.body;
        for (const auto& [k, v] : req.headers) {
            std::string l;
            for (char ch : k) l += char(std::tolower(static_cast<unsigned char>(ch)));
            if (l == "content-type" || l == "accept" || l == "authorization") up->set_header(k, v);
        }
        up->response_handler = [p](const httplib::Response& r) {
            std::lock_guard<std::mutex> lk(p->mu);
            p->status = r.status;
            p->hdrs = r.headers;
            p->headers = true;
            p->cv.notify_all();
            return !p->cancel;
        };
        up->content_receiver = [p](const char* d, size_t n, uint64_t, uint64_t) {
            std::lock_guard<std::mutex> lk(p->mu);
            if (p->cancel) return false;
            p->chunks.emplace_back(d, n);
            p->cv.notify_all();
            return true;
        };
        std::thread([p, up] {
            httplib::Response r;
            httplib::Error e = httplib::Error::Success;
            const bool ok = p->cli->send(*up, r, e);
            std::lock_guard<std::mutex> lk(p->mu);
            if (!ok && !p->headers) p->error = httplib::to_string(e);
            p->failed = !ok;
            p->done = true;
            p->cv.notify_all();
        }).detach();

        // Wait for the child's status line and headers (a non-streamed reply sends them at the end).
        {
            std::unique_lock<std::mutex> lk(p->mu);
            while (!p->headers && !p->done) {
                p->cv.wait_for(lk, std::chrono::milliseconds(100));
                if (!p->headers && !p->done && req.is_connection_closed()) {
                    lk.unlock();
                    p->cancel_now();
                    lk.lock();
                }
            }
            if (!p->headers) {
                res.status = 502;
                res.set_content(err_json("the model server on port " + std::to_string(port) + " did not answer: " + p->error,
                                         "server_error", "upstream_error"), "application/json");
                return;
            }
        }
        res.status = p->status;
        std::string ctype = "application/json";
        bool chunked = false, has_length = false;
        for (const auto& [k, v] : p->hdrs) {
            std::string l;
            for (char ch : k) l += char(std::tolower(static_cast<unsigned char>(ch)));
            if (l == "content-type") ctype = v;
            if (l == "transfer-encoding" && v.find("chunked") != std::string::npos) chunked = true;
            if (l == "content-length") has_length = true;
            if (!front_sets(k)) res.set_header(k, v);
        }
        if (has_length && !chunked) {   // a whole reply (JSON, errors): wait for it and send it as is
            std::unique_lock<std::mutex> lk(p->mu);
            while (!p->done) {
                p->cv.wait_for(lk, std::chrono::milliseconds(100));
                if (!p->done && req.is_connection_closed()) { lk.unlock(); p->cancel_now(); lk.lock(); }
            }
            std::string body;
            for (auto& c : p->chunks) body += c;
            res.set_content(body, ctype);
            return;
        }
        // A stream (SSE): each piece is written through as it arrives.
        res.set_chunked_content_provider(
            ctype,
            [p](size_t, httplib::DataSink& sink) {
                std::unique_lock<std::mutex> lk(p->mu);
                if (p->chunks.empty() && !p->done) p->cv.wait_for(lk, std::chrono::milliseconds(100));
                while (!p->chunks.empty()) {
                    std::string c = std::move(p->chunks.front());
                    p->chunks.pop_front();
                    lk.unlock();
                    if (!sink.write(c.data(), c.size())) { p->cancel_now(); return false; }
                    lk.lock();
                }
                if (p->done && p->failed) return false;   // the child's stream broke: break ours too, not a clean end
                if (p->done) { lk.unlock(); sink.done(); }
                return true;
            },
            [p](bool success) { if (!success) p->cancel_now(); });
    }

    void proxy_get(const std::string& host, int port, const std::string& path, httplib::Response& res) {
        httplib::Client cli(host, port);
        cli.set_connection_timeout(2, 0);
        cli.set_read_timeout(30, 0);
        auto r = cli.Get(path);
        if (!r) {
            res.status = 502;
            res.set_content(err_json("the model server on port " + std::to_string(port) + " did not answer: " +
                                     httplib::to_string(r.error()), "server_error", "upstream_error"), "application/json");
            return;
        }
        res.status = r->status;
        std::string ctype = r->get_header_value("Content-Type");
        res.set_content(r->body, ctype.empty() ? "application/json" : ctype);
    }

    void install_routes() {
        srv_.Post("/admin/shutdown", [this](const httplib::Request& req, httplib::Response& res) {
            if (req.remote_addr != "127.0.0.1" && req.remote_addr != "::1") { res.status = 403; return; }
            res.set_content("{\"status\":\"stopping\"}", "application/json");
            log("shutdown requested by %s", req.remote_addr.c_str());
            admin_stop_.store(true);
        });
        srv_.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
            auto [code, body] = aggregate_health(specs_, views(), stopping_.load(), opt_.default_index);
            res.status = code;
            res.set_content(body, "application/json");
        });
        srv_.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
            res.set_content(models_list_json(specs_, views()), "application/json");
        });
        srv_.Get("/props", [this](const httplib::Request& req, httplib::Response& res) {
            Route r;
            if (req.has_param("model")) { const std::string m = req.get_param_value("model"); r = route_model(specs_, opt_.default_index, &m); }
            else r = route_model(specs_, opt_.default_index, nullptr);
            if (r.status != 200) { res.status = r.status; res.set_content(r.error, "application/json"); return; }
            std::string host;
            int port;
            if (!target(r.index, res, host, port)) return;
            proxy_get(host, port, "/props", res);
        });
        auto generate = [this](const httplib::Request& req, httplib::Response& res) {
            if (stopping_.load()) {
                res.status = 503;
                res.set_header("Retry-After", "1");
                res.set_content(err_json("server is shutting down", "server_error", "shutting_down"), "application/json");
                return;
            }
            const Route r = route_request(specs_, opt_.default_index, req.body);
            if (r.status != 200) { res.status = r.status; res.set_content(r.error, "application/json"); return; }
            std::string host;
            int port;
            if (!target(r.index, res, host, port)) return;
            proxy_post(host, port, req, res);
        };
        srv_.Post("/v1/chat/completions", generate);
        srv_.Post("/v1/completions", generate);
    }
};
}  // namespace

int run_supervisor(const std::vector<ServerSpec>& servers, const Options& opt) {
    if (servers.empty()) { log("the layout has no servers"); return 2; }
    Supervisor s(servers, opt);
    return s.run();
}

}  // namespace ie::sup
