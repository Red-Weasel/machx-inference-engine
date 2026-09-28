// Host-only: `ie supervise` (ie/supervisor.hpp) and the per-card locks (ie/card_lock.hpp). No GPU, no model.
//
// The children are FAKE `ie serve` processes: this binary re-run as `supervisor_test --fake-child <port> <name> <mode>
// <log>`. A fake child answers /health, /props, /v1/chat/completions (a fixed SSE stream, or a JSON echo of the request)
// and /admin/shutdown, and logs what happens to it (listening, shutdown, SIGTERM, exit) so the test can check the
// supervisor's behaviour from the child's side. Front ports 11471 / 11475 / 11478, fake children 11472-11479.
#undef NDEBUG
#include "ie/supervisor.hpp"
#include "ie/card_lock.hpp"
#include "httplib/httplib.h"
#include "nlohmann/json.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

using nlohmann::json;
using namespace std::chrono_literals;
namespace sup = ie::sup;

static bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

// The exact bytes a fake child streams: non-standard fields, a comment line, raw UTF-8 and escapes, [DONE].
static std::string expected_sse(const std::string& name) {
    return "data: {\"id\":\"c1\",\"model\":\"" + name + "\",\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n"
           "data: {\"choices\":[{\"delta\":{\"content\":\"lo \xc3\xa9 \\ud83d\\ude00\"}}],\"ie_vitals\":{\"tok_s\":12.5}}\n\n"
           ": keep-alive\n\n"
           "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}],\"truncated_tool_call\":true,"
           "\"timings\":{\"predicted_per_second\":33.1}}\n\n"
           "data: [DONE]\n\n";
}

// ---- the fake child --------------------------------------------------------------------------------------------------
static int g_log_fd = -1;
static void flog(const std::string& line) {
    const std::string l = line + "\n";
    if (g_log_fd >= 0) { ssize_t w = ::write(g_log_fd, l.data(), l.size()); (void)w; }
}
static void fake_sigterm(int) {
    static const char m[] = "SIGTERM\n";
    if (g_log_fd >= 0) { ssize_t w = ::write(g_log_fd, m, sizeof m - 1); (void)w; }
    _exit(0);
}

static int fake_child(int port, const std::string& name, const std::string& mode, const std::string& log_path) {
    g_log_fd = ::open(log_path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
    std::signal(SIGTERM, fake_sigterm);
    std::signal(SIGINT, fake_sigterm);
    const char* held = std::getenv("IE_GPU_LOCK_HELD");
    flog("pid " + std::to_string(::getpid()) + " held " + (held ? held : "-") + " pgid_is_self " +
         std::to_string(::getpgrp() == ::getpid()));
    double slowload = 0, crash = -1;
    int crash_code = 3;
    if (has(mode, "slowload=")) slowload = std::stod(mode.substr(mode.find("slowload=") + 9));
    if (has(mode, "crash=")) crash = std::stod(mode.substr(mode.find("crash=") + 6));
    if (has(mode, "code=")) crash_code = std::stoi(mode.substr(mode.find("code=") + 5));
    const bool ignore_shutdown = has(mode, "ignore_shutdown");
    const bool abort_on_shutdown = has(mode, "abort_on_shutdown");      // crashes during its requested teardown
    const bool lost_on_shutdown = has(mode, "exit75_on_shutdown");      // "device lost" during its teardown
    if (slowload > 0) std::this_thread::sleep_for(std::chrono::duration<double>(slowload));   // "loading"

    httplib::Server srv;
    std::atomic<int> inflight{0};
    srv.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(json{{"status", "ok"}, {"inflight", inflight.load()}, {"queued", 0}, {"parallel", 1}}.dump(),
                        "application/json");
    });
    srv.Get("/props", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(json{{"default_generation_settings", {{"n_ctx", 4096}}}, {"fake", name}}.dump(), "application/json");
    });
    srv.Post("/v1/chat/completions", [&](const httplib::Request& req, httplib::Response& res) {
        json b = json::parse(req.body, nullptr, false);
        if (b.is_object() && b.value("force_error", false)) {
            res.status = 400;
            res.set_content("{\"error\":{\"message\":\"bad from " + name + "\"}}", "application/json");
            return;
        }
        if (b.is_object() && b.value("endless", false)) {   // streams until the client goes away
            res.set_chunked_content_provider("text/event-stream",
                [](size_t, httplib::DataSink& sink) {
                    std::this_thread::sleep_for(50ms);
                    static const std::string ev = "data: {\"tick\":1}\n\n";
                    return sink.write(ev.data(), ev.size());
                },
                // called once the stream ends: success == false when the client went away, whether httplib noticed
                // it in sink.write or in its is_writable check before calling the provider
                [](bool success) { if (!success) flog("client gone (stream)"); });
            return;
        }
        if (b.is_object() && b.value("long_prefill", false)) {   // no header for a long time, like a big prefill
            for (int i = 0; i < 100; ++i) {
                std::this_thread::sleep_for(50ms);
                if (req.is_connection_closed()) { flog("client gone (prefill)"); return; }
            }
            res.set_content("{}", "application/json");
            return;
        }
        if (b.is_object() && b.value("stream", false)) {
            ++inflight;
            const std::string all = expected_sse(name);
            // Four writes, one of them splitting an event in the middle.
            auto parts = std::make_shared<std::vector<std::string>>(std::vector<std::string>{
                all.substr(0, 30), all.substr(30, 90), all.substr(120, 40), all.substr(160)});
            res.set_header("X-Fake-Child", name);
            res.set_chunked_content_provider("text/event-stream",
                [parts, &inflight](size_t, httplib::DataSink& sink) {
                    for (const auto& p : *parts) {
                        std::this_thread::sleep_for(60ms);
                        if (!sink.write(p.data(), p.size())) { --inflight; return false; }
                    }
                    --inflight;
                    sink.done();
                    return true;
                });
            return;
        }
        res.set_content(json{{"model", name}, {"echo", req.body}}.dump(), "application/json");
    });
    srv.Post("/admin/shutdown", [&](const httplib::Request& req, httplib::Response& res) {
        flog("shutdown from " + req.remote_addr);
        res.set_content("{\"status\":\"stopping\"}", "application/json");
        if (abort_on_shutdown) std::thread([] { std::this_thread::sleep_for(100ms); flog("abort"); std::abort(); }).detach();
        else if (lost_on_shutdown) std::thread([] { std::this_thread::sleep_for(100ms); flog("exit 75"); _exit(75); }).detach();
        else if (!ignore_shutdown) std::thread([&] { std::this_thread::sleep_for(100ms); srv.stop(); }).detach();
    });
    if (crash >= 0)
        std::thread([&, crash, crash_code] {
            std::this_thread::sleep_for(std::chrono::duration<double>(crash));
            flog("crash");
            _exit(crash_code);
        }).detach();
    const pid_t parent = ::getppid();
    std::thread([parent] {
        while (::getppid() == parent) std::this_thread::sleep_for(100ms);
        flog("test process gone: exiting");
        _exit(0);
    }).detach();
    if (!srv.bind_to_port("127.0.0.1", port)) { flog("bind failed"); return 9; }
    std::fprintf(stderr, "fake child %s listening\n", name.c_str());   // the supervisor prefixes it with [name]
    flog("listening");
    srv.listen_after_bind();
    flog("exit");
    return 0;
}

// ---- helpers ------------------------------------------------------------------------------------------------------------
static std::string g_self;
static std::string g_dir;

static std::string read_file(const std::string& p) {
    std::ifstream f(p);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static sup::ServerSpec fake_spec(const std::string& name, const std::string& id, int port, uint32_t card,
                                 const std::string& mode) {
    sup::ServerSpec s;
    s.name = name;
    s.model_id = id;
    s.port = port;
    s.cards = {card};
    const std::string log = g_dir + "/" + name + ".log";
    ::unlink(log.c_str());
    s.argv = {g_self, "--fake-child", std::to_string(port), name, mode, log};
    s.slots = 4;
    return s;
}

struct Raw { int status = 0; std::string body, ctype; httplib::Headers headers; };
static Raw post(int port, const std::string& path, const std::string& body) {
    httplib::Client cli("127.0.0.1", port);
    cli.set_read_timeout(30, 0);
    httplib::Request req;
    req.method = "POST";
    req.path = path;
    req.body = body;
    req.set_header("Content-Type", "application/json");
    Raw out;
    req.content_receiver = [&](const char* d, size_t n, uint64_t, uint64_t) { out.body.append(d, n); return true; };
    httplib::Response res;
    httplib::Error err;
    if (!cli.send(req, res, err)) { out.status = -1; return out; }
    out.status = res.status;
    out.ctype = res.get_header_value("Content-Type");
    out.headers = res.headers;
    return out;
}
static Raw get(int port, const std::string& path) {
    httplib::Client cli("127.0.0.1", port);
    cli.set_read_timeout(10, 0);
    Raw out;
    auto r = cli.Get(path);
    if (!r) { out.status = -1; return out; }
    out.status = r->status;
    out.body = r->body;
    out.ctype = r->get_header_value("Content-Type");
    return out;
}
template <class F> static bool wait_until(F f, double seconds) {
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - t0 < std::chrono::duration<double>(seconds)) {
        if (f()) return true;
        std::this_thread::sleep_for(50ms);
    }
    return f();
}
static bool pid_alive(int pid) { return pid > 0 && ::kill(pid, 0) == 0; }
static int logged_pid(const std::string& log) {
    const auto s = read_file(log);
    const auto p = s.find("pid ");
    return p == std::string::npos ? -1 : std::stoi(s.substr(p + 4));
}

// ---- pure pieces ----------------------------------------------------------------------------------------------------------
static void test_routing() {
    std::vector<sup::ServerSpec> s(3);
    s[0].name = "mimo";  s[0].model_id = "MiMo-V2.6-Flash-RL";
    s[1].name = "coder"; s[1].model_id = "Qwen3-Coder";
    s[2].name = "coder2"; s[2].model_id = "Qwen3-Coder";   // the same model twice: its id is ambiguous
    auto r = sup::route_request(s, -1, R"({"model":"coder","messages":[]})");
    assert(r.status == 200 && r.index == 1);
    r = sup::route_request(s, -1, R"({"model":"MiMo-V2.6-Flash-RL"})");   // by the unique model id
    assert(r.status == 200 && r.index == 0);
    r = sup::route_request(s, -1, R"({"model":"Qwen3-Coder"})");          // an id two servers share
    assert(r.status == 400 && has(r.error, "model_ambiguous") && has(r.error, "coder, coder2"));
    r = sup::route_request(s, -1, R"({"model":"gpt-4"})");                // unknown: 404 naming the models
    assert(r.status == 404 && has(r.error, "model_not_found") && has(r.error, "mimo, coder, coder2"));
    assert(json::parse(r.error)["error"]["message"].is_string());
    r = sup::route_request(s, -1, R"({"messages":[]})");                  // missing, no default: 400
    assert(r.status == 400 && has(r.error, "model_required") && has(r.error, "mimo, coder, coder2"));
    r = sup::route_request(s, 1, R"({"messages":[]})");                   // missing: the default
    assert(r.status == 200 && r.index == 1);
    r = sup::route_request(s, 2, R"({"model":null})");                    // null = missing
    assert(r.status == 200 && r.index == 2);
    r = sup::route_request(s, 0, R"({"model":7})");
    assert(r.status == 400 && has(r.error, "must be a string"));
    r = sup::route_request(s, 0, "not json");
    assert(r.status == 400 && has(r.error, "invalid_json"));
    r = sup::route_request(s, 0, "[1]");
    assert(r.status == 400);
    r = sup::route_request(s, 0, R"({"model":"coder"})");                 // a name beats the default
    assert(r.status == 200 && r.index == 1);
    std::puts("routing: by name, by unique id, ambiguous id, unknown, missing, default -- ok");
}

static void test_aggregate() {
    std::vector<sup::ServerSpec> s(2);
    s[0].name = "a"; s[0].model_id = "A"; s[0].port = 1; s[0].cards = {0};
    s[1].name = "b"; s[1].model_id = "B"; s[1].port = 2; s[1].cards = {1};
    std::vector<sup::ChildView> v(2);
    v[0].state = sup::ChildState::Ready; v[0].health = R"({"status":"ok","inflight":1,"queued":2})";
    v[1].state = sup::ChildState::Ready; v[1].health = R"({"status":"ok","inflight":1,"queued":0})";
    auto [code, body] = sup::aggregate_health(s, v, false, 0);
    json h = json::parse(body);
    assert(code == 200 && h["status"] == "ok" && h["inflight"] == 2 && h["queued"] == 2 && h["default"] == "a");
    assert(h["servers"]["b"]["state"] == "ready" && h["servers"]["b"]["cards"] == json::array({1}) &&
           h["servers"]["a"]["health"]["queued"] == 2);
    v[1].state = sup::ChildState::Exited; v[1].exit_code = 3; v[1].health.clear();
    std::tie(code, body) = sup::aggregate_health(s, v, false, -1);
    h = json::parse(body);
    assert(code == 200 && h["status"] == "degraded" && h["servers"]["b"]["exit_code"] == 3 && !h.contains("default"));
    v[0].state = sup::ChildState::Loading; v[0].health.clear();
    std::tie(code, body) = sup::aggregate_health(s, v, false, -1);
    assert(code == 503 && json::parse(body)["status"] == "loading");
    v[0].state = sup::ChildState::Exited;
    std::tie(code, body) = sup::aggregate_health(s, v, false, -1);
    assert(code == 503 && json::parse(body)["status"] == "unavailable");
    v[0].state = sup::ChildState::Ready;
    std::tie(code, body) = sup::aggregate_health(s, v, true, -1);
    assert(code == 503 && json::parse(body)["status"] == "stopping");
    v[1].state = sup::ChildState::Exited; v[1].exit_signal = 9;
    h = json::parse(sup::aggregate_health(s, v, false, -1).second);
    assert(h["servers"]["b"]["exit_signal"] == 9 && !h["servers"]["b"].contains("exit_code"));

    json m = json::parse(sup::models_list_json(s, v));
    assert(m["object"] == "list" && m["data"].size() == 2 && m["data"][0]["id"] == "a" && m["data"][0]["root"] == "A" &&
           m["data"][1]["status"] == "exited" && m["data"][0]["object"] == "model");
    std::puts("health aggregation (ok / degraded / loading / unavailable / stopping, sums) and /v1/models -- ok");
}

static void test_card_locks() {
    unsetenv("IE_GPU_LOCK_HELD");
    ie::CardLocks a, b, c;
    assert(ie::acquire_card_locks({0}, a).empty() && a.fds().size() == 1);
    assert(ie::card_lock_holder(0) == std::to_string(::getpid()));
    assert(ie::acquire_card_locks({1}, b).empty());                          // another card: fine
    std::string e = ie::acquire_card_locks({2, 0}, c);                       // a held card: refused, nothing kept
    assert(has(e, "card 0 is in use") && has(e, "pid " + std::to_string(::getpid())) && c.fds().empty());
    { ie::CardLocks x; assert(ie::acquire_card_locks({2}, x).empty()); }     // card 2 was not kept by the refusal
    // another process is refused too (flock is per open file description; this is the real case)
    const pid_t pid = ::fork();
    if (pid == 0) { ie::CardLocks x; _exit(ie::acquire_card_locks({0}, x).empty() ? 0 : 1); }
    int st = 0;
    ::waitpid(pid, &st, 0);
    assert(WIFEXITED(st) && WEXITSTATUS(st) == 1);
    a.release();                                                             // released: available again
    assert(ie::acquire_card_locks({0}, c).empty());
    // an ancestor's lock named in IE_GPU_LOCK_HELD is not taken again
    setenv("IE_GPU_LOCK_HELD", "0,1", 1);
    assert(ie::card_lock_inherited(0) && ie::card_lock_inherited(1) && !ie::card_lock_inherited(2));
    ie::CardLocks d;
    assert(ie::acquire_card_locks({0, 1}, d).empty() && d.fds().empty());
    setenv("IE_GPU_LOCK_HELD", "10", 1);
    assert(!ie::card_lock_inherited(1) && !ie::card_lock_inherited(0) && ie::card_lock_inherited(10));
    setenv("IE_GPU_LOCK_HELD", "all", 1);
    assert(ie::card_lock_inherited(7));
    unsetenv("IE_GPU_LOCK_HELD");
    std::puts("card locks: one process per card, different cards coexist, released on close, inherited skip -- ok");
}

// ---- the supervisor against fake children ---------------------------------------------------------------------------------
static void test_serving_and_shutdown() {
    const int front = 11471;
    std::vector<sup::ServerSpec> specs = {fake_spec("alpha", "Model-A-Q4", 11472, 0, "normal"),
                                          fake_spec("beta", "Model-B-Q8", 11473, 1, "normal"),
                                          fake_spec("gamma", "Model-C", 11474, 2, "crash=1.5")};
    sup::Options o;
    o.port = front;
    o.default_index = 0;
    o.poll_s = 0.2;
    o.ready_wait_s = 20;
    o.stop_wait_s = 15;
    std::atomic<int> rc{-100};
    const std::string err_path = g_dir + "/supervisor_stderr.txt";
    const int saved_err = ::dup(2);
    const int err_fd = ::open(err_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    ::dup2(err_fd, 2);
    ::close(err_fd);
    std::thread t([&] { rc = sup::run_supervisor(specs, o); });

    assert(wait_until([&] { return get(front, "/health").status == 200 &&
                                   json::parse(get(front, "/health").body)["status"] == "ok"; }, 20));
    // every child got its own cards' lock and its own process group
    for (const char* n : {"alpha", "beta"}) {
        const auto l = read_file(g_dir + "/" + n + ".log");
        assert(has(l, std::string("held ") + (n[0] == 'a' ? "0" : "1")) && has(l, "pgid_is_self 1"));
    }
    std::string e;
    { ie::CardLocks x; e = ie::acquire_card_locks({1}, x); }
    assert(has(e, "card 1 is in use"));                                      // held while the children run

    json models = json::parse(get(front, "/v1/models").body);
    assert(models["data"].size() == 3 && models["data"][0]["id"] == "alpha" && models["data"][1]["root"] == "Model-B-Q8");

    // routing: by name (the body arrives unchanged), by id, default, unknown
    const std::string body_a = "{\"model\":\"alpha\",  \"messages\":[{\"role\":\"user\",\"content\":\"h\xc3\xa9\"}]}";
    Raw r = post(front, "/v1/chat/completions", body_a);
    assert(r.status == 200 && has(r.ctype, "application/json"));
    json j = json::parse(r.body);
    assert(j["model"] == "alpha" && j["echo"] == body_a);
    r = post(front, "/v1/chat/completions", R"({"model":"Model-B-Q8","messages":[]})");
    assert(r.status == 200 && json::parse(r.body)["model"] == "beta");
    r = post(front, "/v1/chat/completions", R"({"messages":[]})");
    assert(r.status == 200 && json::parse(r.body)["model"] == "alpha");
    r = post(front, "/v1/chat/completions", R"({"model":"nope","messages":[]})");
    assert(r.status == 404 && has(r.body, "alpha, beta, gamma"));
    r = post(front, "/v1/chat/completions", R"({"model":"beta","force_error":true})");   // a child's error passes through
    assert(r.status == 400 && r.body == "{\"error\":{\"message\":\"bad from beta\"}}");
    r = post(front, "/v1/completions", R"({"model":"beta","prompt":"x"})");
    assert(r.status == 404);   // proxied: the fake child (like ie serve) has no /v1/completions

    // SSE: byte for byte, both servers at once
    Raw sa, sb;
    const auto t0 = std::chrono::steady_clock::now();
    std::thread ta([&] { sa = post(front, "/v1/chat/completions", R"({"model":"alpha","stream":true})"); });
    std::thread tb([&] { sb = post(front, "/v1/chat/completions", R"({"model":"beta","stream":true})"); });
    ta.join();
    tb.join();
    const double both_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    assert(sa.status == 200 && sa.body == expected_sse("alpha") && has(sa.ctype, "text/event-stream"));
    assert(sb.status == 200 && sb.body == expected_sse("beta"));
    bool fake_header = false;
    for (const auto& [k, v] : sb.headers) fake_header = fake_header || (k == "X-Fake-Child" && v == "beta");
    assert(fake_header);
    assert(both_s < 0.48);   // each stream takes >= 0.24 s: they ran at the same time

    // a client that goes away is seen by the child: mid-stream, and while the child has not answered yet
    {
        httplib::Client cli("127.0.0.1", front);
        httplib::Request q;
        q.method = "POST";
        q.path = "/v1/chat/completions";
        q.body = R"({"model":"alpha","endless":true})";
        q.set_header("Content-Type", "application/json");
        size_t got = 0;
        q.content_receiver = [&](const char*, size_t n, uint64_t, uint64_t) { got += n; return got < 60; };
        httplib::Response res;
        httplib::Error err;
        (void)cli.send(q, res, err);
        assert(got >= 60);
        assert(wait_until([&] { return has(read_file(g_dir + "/alpha.log"), "client gone (stream)"); }, 5));
        httplib::Client cli2("127.0.0.1", front);
        cli2.set_read_timeout(0, 300000);   // gives up after 0.3 s and closes the connection
        auto r2 = cli2.Post("/v1/chat/completions", R"({"model":"beta","long_prefill":true})", "application/json");
        assert(!r2);
        assert(wait_until([&] { return has(read_file(g_dir + "/beta.log"), "client gone (prefill)"); }, 5));
    }

    // /props
    r = get(front, "/props?model=beta");
    assert(r.status == 200 && json::parse(r.body)["fake"] == "beta");
    r = get(front, "/props");
    assert(r.status == 200 && json::parse(r.body)["fake"] == "alpha");
    assert(get(front, "/props?model=zzz").status == 404);

    // a crashed child is reported; the others keep serving
    assert(wait_until([&] { return json::parse(get(front, "/health").body)["servers"]["gamma"]["state"] == "exited"; }, 10));
    Raw h = get(front, "/health");
    j = json::parse(h.body);
    assert(h.status == 200 && j["status"] == "degraded" && j["servers"]["gamma"]["exit_code"] == 3);
    r = post(front, "/v1/chat/completions", R"({"model":"gamma"})");
    assert(r.status == 503 && has(r.body, "server_not_running") && has(r.body, "exit code 3"));
    assert(post(front, "/v1/chat/completions", R"({"model":"alpha"})").status == 200);

    // shutdown: each child gets its own /admin/shutdown and exits; nothing is signalled
    const int pa = logged_pid(g_dir + "/alpha.log"), pb = logged_pid(g_dir + "/beta.log");
    assert(pid_alive(pa) && pid_alive(pb));
    assert(post(front, "/admin/shutdown", "").status == 200);
    t.join();
    assert(rc == 1);   // gamma crashed
    for (const char* n : {"alpha", "beta"}) {
        const auto l = read_file(g_dir + "/" + n + ".log");
        assert(has(l, "shutdown from 127.0.0.1") && has(l, "exit") && !has(l, "SIGTERM"));
        assert(l.find("shutdown from") < l.find("\nexit"));
    }
    assert(!pid_alive(pa) && !pid_alive(pb));
    assert(get(front, "/health").status == -1);   // the front is closed after the children
    ::dup2(saved_err, 2);
    ::close(saved_err);
    const std::string err = read_file(err_path);
    assert(has(err, "[alpha] fake child alpha listening\n") && has(err, "[beta] fake child beta listening\n"));
    assert(has(err, "[ie supervise] stopped; 1 child(ren) did not stop cleanly: gamma (exit code 3)"));
    { ie::CardLocks x; assert(ie::acquire_card_locks({0, 1, 2}, x).empty()); }   // every lock released
    std::puts("supervisor: routing, byte-exact SSE passthrough (concurrent), client disconnect reaches the child, /props, crash report, "
              "ordered /admin/shutdown, locks -- ok");
}

static void test_loading_and_stubborn_children() {
    const int front = 11475;
    std::vector<sup::ServerSpec> specs = {fake_spec("slow", "S", 11476, 0, "slowload=2"),
                                          fake_spec("stubborn", "T", 11477, 1, "ignore_shutdown")};
    sup::Options o;
    o.port = front;
    o.poll_s = 0.2;
    o.ready_wait_s = 20;
    o.stop_wait_s = 2;
    std::atomic<int> rc{-100};
    std::thread t([&] { rc = sup::run_supervisor(specs, o); });
    assert(wait_until([&] { return get(front, "/health").status != -1; }, 10));
    Raw r = post(front, "/v1/chat/completions", R"({"model":"slow"})");   // still loading
    assert(r.status == 503 && has(r.body, "\"loading\""));
    assert(wait_until([&] { return json::parse(get(front, "/health").body)["servers"]["stubborn"]["state"] == "ready"; }, 10));
    assert(json::parse(get(front, "/health").body)["servers"]["slow"]["state"] == "loading");
    assert(post(front, "/admin/shutdown", "").status == 200);   // while "slow" is still loading
    t.join();
    // "stubborn" was left running: exit 1 however its state was polled meanwhile (it still answers /health 200)
    const int rc_left = rc;
    const auto ls = read_file(g_dir + "/slow.log");
    assert(ls.find("listening") < ls.find("shutdown from") && has(ls, "\nexit") && !has(ls, "SIGTERM"));
    const auto lt = read_file(g_dir + "/stubborn.log");
    assert(has(lt, "shutdown from") && !has(lt, "SIGTERM"));
    const int pt = logged_pid(g_dir + "/stubborn.log");
    int st = 0;
    const bool was_alive = pt > 0 && ::waitpid(pt, &st, WNOHANG) == 0;   // not reaped = still running
    if (pt > 0) { ::kill(pt, SIGKILL); ::waitpid(pt, nullptr, 0); }   // a fake child: clean up before asserting
    assert(was_alive);   // reported and LEFT RUNNING, not killed
    assert(rc_left == 1);
    std::puts("supervisor: a loading child finishes loading before it is stopped; a child that ignores shutdown is "
              "left running -- ok");
}

// A child that dies during its own requested shutdown is NOT a clean stop: exit 1 and the line names it.
static void test_teardown_failures() {
    struct Case { const char* mode; int want_rc; const char* want_line; };
    const Case cases[] = {{"abort_on_shutdown", 1, "did not stop cleanly: td (signal 6)"},
                          {"exit75_on_shutdown", 1, "did not stop cleanly: td (exit code 75, device lost)"},
                          {"normal", 0, "stopped (every child stopped cleanly)"}};
    for (const auto& k : cases) {
        const int front = 11478;
        const std::string err_path = g_dir + "/teardown_stderr.txt";
        const int saved_err = ::dup(2);
        const int err_fd = ::open(err_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        ::dup2(err_fd, 2);
        ::close(err_fd);
        sup::Options o;
        o.port = front;
        o.poll_s = 0.1;
        o.stop_wait_s = 10;
        std::atomic<int> rc{-100};
        std::thread t([&] { rc = sup::run_supervisor({fake_spec("td", "TD", 11479, 0, k.mode)}, o); });
        const bool up = wait_until([&] { return get(front, "/health").status == 200; }, 10);
        const bool sent = up && post(front, "/admin/shutdown", "").status == 200;
        t.join();
        ::dup2(saved_err, 2);
        ::close(saved_err);
        const std::string err = read_file(err_path);
        if (!(up && sent && rc == k.want_rc && has(err, k.want_line)))
            std::fprintf(stderr, "teardown case %s: up %d sent %d rc %d\n%s\n", k.mode, up, sent, rc.load(), err.c_str());
        assert(up && sent && rc == k.want_rc && has(err, k.want_line));
        assert(has(read_file(g_dir + "/td.log"), "shutdown from"));   // the stop was requested
    }
    std::puts("supervisor: a child that aborts or exits 75 during its requested shutdown -> exit 1, named; clean -> 0 -- ok");
}

static void test_restart_and_lock_refusal() {
    // "restart": "on-failure" -- up to kMaxRestarts, never after exit 75 (device lost)
    const int front = 11478;
    auto flaky = fake_spec("flaky", "F", 11479, 0, "crash=0.3");
    flaky.restart_on_failure = true;
    auto lost = fake_spec("lost", "L", 11472, 1, "crash=0.3,code=75");
    lost.restart_on_failure = true;
    sup::Options o;
    o.port = front;
    o.poll_s = 0.1;
    o.stop_wait_s = 5;
    std::atomic<int> rc{-100};
    std::thread t([&] { rc = sup::run_supervisor({flaky, lost}, o); });
    assert(wait_until([&] {
        Raw h = get(front, "/health");
        if (h.status == -1) return false;
        json j = json::parse(h.body);
        return j["servers"]["flaky"]["restarts"] == sup::kMaxRestarts && j["servers"]["flaky"]["state"] == "exited" &&
               j["servers"]["lost"]["state"] == "exited";
    }, 30));
    json j = json::parse(get(front, "/health").body);
    assert(j["servers"]["lost"]["restarts"] == 0 && j["servers"]["lost"]["exit_code"] == 75);
    const auto lf = read_file(g_dir + "/flaky.log");
    size_t starts = 0;
    for (size_t p = 0; (p = lf.find("listening", p)) != std::string::npos; ++p) ++starts;
    assert(starts == 1 + sup::kMaxRestarts);
    assert(post(front, "/admin/shutdown", "").status == 200);
    t.join();
    assert(rc == 1);

    // a card held by another engine process: the supervisor refuses and starts nothing
    ie::CardLocks held;
    assert(ie::acquire_card_locks({5}, held).empty());
    auto blocked = fake_spec("blocked", "B", 11473, 5, "normal");
    sup::Options o2;
    o2.port = 11471;
    assert(sup::run_supervisor({blocked}, o2) == 2);
    std::this_thread::sleep_for(300ms);
    assert(read_file(g_dir + "/blocked.log").empty());   // never started
    std::puts("supervisor: on-failure restarts (limit, not after device lost), refusal on a held card -- ok");
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--fake-child") {
        assert(argc == 6);
        return fake_child(std::stoi(argv[2]), argv[3], argv[4], argv[5]);
    }
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof buf - 1);
    assert(n > 0);
    g_self.assign(buf, size_t(n));
    char tmpl[] = "/tmp/ie_supervisor_test_XXXXXX";
    assert(::mkdtemp(tmpl));
    g_dir = tmpl;
    setenv("IE_CARD_LOCK_DIR", g_dir.c_str(), 1);
    unsetenv("IE_GPU_LOCK_HELD");

    test_routing();
    test_aggregate();
    test_card_locks();
    test_serving_and_shutdown();
    test_loading_and_stubborn_children();
    test_teardown_failures();
    test_restart_and_lock_refusal();
    std::filesystem::remove_all(g_dir);
    std::puts("supervisor_test: all passed");
    return 0;
}
