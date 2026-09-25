#include "ie/openai_server.hpp"
#include "ie/openai_proto.hpp"
#include "ie/server_admission.hpp"
#include "ie/stop_sequences.hpp"
#include "httplib/httplib.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <malloc.h>
#include <filesystem>
#include <memory>
#include <thread>
#include <mutex>
#include <unistd.h>

namespace ie {

// SIGTERM/SIGINT: the handler only records the signal (async-signal-safe); a
// watcher thread in run_openai_server turns it into an orderly stop — abort
// in-flight generations, refuse the queue, close the listener — so main()
// returns and every destructor (pinned host arenas, device state) runs. A
// second signal while stopping hard-exits: nothing must be able to wedge the
// shutdown behind a stuck forward.
static std::atomic<int> g_signal{0};
static void on_stop_signal(int sig) {
    if (g_signal.exchange(sig) != 0) _exit(128 + sig);
}

// After every generation, hand the heap's free pages back to the OS. A request's transient buffers (prefill chunks,
// image rows, bounce copies) are freed into glibc's per-thread arenas, which otherwise keep them: live, a day of
// Dream image traffic grew ie serve by ~30 GB of freed-but-held heap until the machine ran out of RAM
// (docs/deepseek41/99). IE_MALLOC_TRIM=0 turns it off.
// A generation the engine stopped as repetition is a bug report: keep the request that
// produced it (bounded, newest wins) so it can be replayed exactly (docs/deepseek41/101).
static void keep_repeating_request(const std::string& body) {
    const char* home = std::getenv("HOME");
    if (!home) return;
    std::error_code ec;
    const std::filesystem::path dir = std::filesystem::path(home) / ".cache" / "machx-ie" / "repetition";
    std::filesystem::create_directories(dir, ec);
    size_t have = 0;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) { (void)e; ++have; }
    if (have >= 20) return;
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto path = dir / (std::to_string(std::chrono::duration_cast<std::chrono::seconds>(now).count()) + ".json");
    if (FILE* f = std::fopen(path.c_str(), "wb")) {
        std::fwrite(body.data(), 1, body.size(), f);
        std::fclose(f);
        std::fprintf(stderr, "[req] the reply repeated itself and was stopped; the request is in %s\n", path.c_str());
    }
}

static double g_trim_ms = 0.0;   // the last trim's cost, for IE_MEM_REPORT
static void release_free_heap() {
    static const bool off = [] { const char* v = std::getenv("IE_MALLOC_TRIM"); return v && v[0] == '0'; }();
    if (off) return;
    const auto t0 = std::chrono::steady_clock::now();
    malloc_trim(0);
    g_trim_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// #54 (2026-09-22 02:28): a lost device cannot be recovered in-process. The orderly teardown spun on
// one core for good while the GT reset-looped, holding 29 GiB of VRAM and the pinned RAM, and the
// supervisor kept polling an unhealthy server. Once the latch trips, log it and leave: the reply that hit
// the fault has already been written, and the driver reclaims the VM when the process is gone.
// IE_EXIT_ON_DEVICE_LOST=0 keeps the old behaviour (stay up, answer 503).
static void exit_on_device_loss(const ie::DeviceFaultLatch& fault) {
    if (!fault.faulted()) return;
    if (const char* v = std::getenv("IE_EXIT_ON_DEVICE_LOST"); v && *v && std::string(v) == "0") return;
    static std::once_flag once;
    std::call_once(once, [&fault] {
        std::fprintf(stderr, "[ie] device lost (%s): exiting in 2 s so the supervisor can relaunch "
                             "(IE_EXIT_ON_DEVICE_LOST=0 keeps the process up)\n", fault.reason().c_str());
        std::fflush(stderr);
        std::thread([] { std::this_thread::sleep_for(std::chrono::seconds(2)); std::fflush(stderr); std::_Exit(75); }).detach();
    });
}

// Per-request throughput to the engine's terminal (stderr). Free: token counts
// already tracked, timing is phase-boundary clock reads (see Engine::generate).
static void log_gen_speed(const GenerateResult& r) {
    // Rate the prefill on the tokens it actually ran. Counting the cached ones as if
    // they had been prefilled reads as thousands of tok/s and hides both the real rate
    // and the fixed per-turn cost (docs/deepseek41/102).
    const uint32_t run = r.prompt_tokens - std::min(r.cached_tokens, r.prompt_tokens);
    const double ran_ms = std::max(0.0, r.prefill_ms - r.restore_ms);
    const double pf = ran_ms > 0 ? run * 1000.0 / ran_ms : 0.0;
    const double dc = r.decode_ms  > 0 ? r.completion_tokens * 1000.0 / r.decode_ms : 0.0;
    char split[192];
    if (r.cached_tokens)
        std::snprintf(split, sizeof split, " [restore %.0f ms from %s, %u new in %.0f ms = %.0f tok/s]",
                      r.restore_ms, r.cache_source.empty() ? "cache" : r.cache_source.c_str(),
                      run, ran_ms, pf);
    else
        std::snprintf(split, sizeof split, " = %.0f tok/s", pf);
    // Early vs late within ONE reply. A per-request average cannot tell decode that
    // degrades as the KV grows from a process that has simply become slower
    // (docs/deepseek41/102); this can.
    char pace[96] = "";
    if (r.early_decode_n && r.completion_tokens > r.early_decode_n + 50 && r.early_decode_ms > 0) {
        const double early = r.early_decode_n * 1000.0 / r.early_decode_ms;
        const double rest_ms = r.decode_ms - r.early_decode_ms;
        if (rest_ms > 0)
            std::snprintf(pace, sizeof pace, " (first %u at %.1f, rest at %.1f)", r.early_decode_n, early,
                          (r.completion_tokens - r.early_decode_n) * 1000.0 / rest_ms);
    }
    std::fprintf(stderr,
        "[gen] prefill %u tok (%u cached) / %.0f ms%s  |  decode %u tok / %.0f ms = %.1f tok/s%s\n",
        r.prompt_tokens, r.cached_tokens, r.prefill_ms, split, r.completion_tokens, r.decode_ms, dc, pace);
    // IE_MEM_REPORT=1: the process's anonymous RSS and glibc's heap (all arenas) after every request -- tells a
    // leak (in-use climbs) from arena fragmentation (free-in-arenas climbs) (docs/deepseek41/99)
    static const bool mem_report = std::getenv("IE_MEM_REPORT") != nullptr;
    if (mem_report) {
        long anon_kb = -1;
        if (FILE* f = std::fopen("/proc/self/status", "r")) {
            char line[256];
            while (std::fgets(line, sizeof line, f))
                if (!std::strncmp(line, "RssAnon:", 8)) { anon_kb = std::atol(line + 8); break; }
            std::fclose(f);
        }
        const struct mallinfo2 mi = mallinfo2();
        std::fprintf(stderr, "[mem] RssAnon %ld MiB | malloc heap %zu MiB: in use %zu, free in arenas %zu; mmapped %zu MiB | trim %.1f ms\n",
                     anon_kb / 1024, mi.arena >> 20, mi.uordblks >> 20, mi.fordblks >> 20, mi.hblkhd >> 20, g_trim_ms);
    }
    std::fflush(stderr);
}

// #64: a generation that ends in an error reaches the client as its finish reason ("error: ..."); say it on the log too,
// with the request's shape. Without this line a failed turn left nothing in the log (live 2026-09-24 12:59: a turn died,
// cause unknown). Every model's serve path comes through here.
static void log_gen_error(const std::string& id, const GenerateResult& r) {
    std::string why = r.finish_reason.starts_with("error:") ? r.finish_reason.substr(6) : r.finish_reason;
    if (!why.empty() && why.front() == ' ') why.erase(0, 1);
    std::fprintf(stderr, "[req] generation error: %s (%s; prompt %u tok, %u cached; completion %u tok)\n",
                 why.c_str(), id.c_str(), r.prompt_tokens, r.cached_tokens, r.completion_tokens);
    std::fflush(stderr);
}

int run_openai_server(Engine& eng, const std::string& model_id,
                      const std::string& host, int port, uint32_t max_queue) {
    httplib::Server srv;
    // Admission bound: up to eng.parallel() generations in flight at once (the
    // engine's internal FIFO gate owns the actual GPU serialization and
    // interleaving — Phase 1b; each in-flight request holds a host-RAM slot
    // stash), at most max_queue more waiting, the rest refused with 429. The
    // worker pool is sized so every admitted+queued request can hold a thread
    // and /health, /props and the 429s themselves still get one.
    Admission adm(eng.parallel(), max_queue);
    DeviceFaultLatch fault;
    const unsigned pool_threads = eng.parallel() + max_queue + 4;
    srv.new_task_queue = [pool_threads] { return new httplib::ThreadPool(pool_threads); };
    std::atomic<uint64_t> req_no{0};
    // Releases exactly once: non-copyable, so no temporary can ever run the
    // destructor a second time (a make_shared<AdmRelease>(AdmRelease{..}) did).
    std::atomic<bool> admin_stop{false};   // set by POST /admin/shutdown
    struct AdmRelease {
        Admission* a;
        explicit AdmRelease(Admission* adm_) : a(adm_) {}
        AdmRelease(const AdmRelease&) = delete;
        AdmRelease& operator=(const AdmRelease&) = delete;
        ~AdmRelease() { a->release(); }
    };
    // 503 once the device is gone: every generation would fail identically.
    auto unavailable = [&](httplib::Response& res) {
        res.status = 503;
        res.set_header("Retry-After", "30");
        res.set_content(oai::error_json("engine unavailable: " + fault.reason(),
                                        "server_error", "device_lost"),
                        "application/json");
    };
    // 429 when the wait queue is full (503 while shutting down).
    auto refuse = [&](httplib::Response& res) {
        const bool stopping = adm.stopping();
        res.status = stopping ? 503 : 429;
        res.set_header("Retry-After", "1");
        res.set_content(oai::error_json(
            stopping ? std::string("server is shutting down")
                     : "too many requests: " + std::to_string(eng.parallel()) +
                       " running and " + std::to_string(max_queue) + " waiting",
            stopping ? "server_error" : "rate_limit_error",
            stopping ? "shutting_down" : "queue_full"), "application/json");
    };

    // Clean stop for test automation: `listen_after_bind()` returns, main
    // returns, and every destructor (pinned host arenas, device state) runs —
    // a SIGTERM skips all of that. Loopback callers only.
    srv.Post("/admin/shutdown", [&](const httplib::Request& req, httplib::Response& res) {
        if (req.remote_addr != "127.0.0.1" && req.remote_addr != "::1") { res.status = 403; return; }
        res.set_content("{\"status\":\"stopping\"}", "application/json");
        std::fprintf(stderr, "[ie] shutdown requested by %s\n", req.remote_addr.c_str());
        admin_stop.store(true);   // the watcher thread below does the orderly stop
    });
    // Liveness a supervisor can act on: 200 while the device is healthy, 503
    // once a forward reported a lost/reset device (restart the process). Also
    // the only place queue depth is visible.
    srv.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        nlohmann::json h{{"status", fault.faulted() ? "unhealthy" : "ok"},
                         {"inflight", adm.inflight()}, {"queued", adm.queued()},
                         {"parallel", eng.parallel()}, {"max_queue", max_queue}};
        if (fault.faulted()) { h["reason"] = fault.reason(); res.status = 503; }
        if (adm.stopping())  { h["status"] = "stopping"; res.status = 503; }
        res.set_content(h.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace),
                        "application/json");
    });
    srv.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(oai::models_json(model_id), "application/json");
    });
    // /props — llama.cpp-compatible server-truth for the LOADED context window.
    // Dream (and other llama.cpp-shaped clients) probe exactly this shape to learn
    // n_ctx without guessing what the launcher parsed. This is the SERVER's real
    // max_ctx, so a footer/UI reading it can never disagree with the running engine.
    srv.Get("/props", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(
            "{\"default_generation_settings\":{\"n_ctx\":" +
                std::to_string(eng.max_ctx()) + "},\"total_slots\":" +
                std::to_string(eng.parallel()) + ",\"prompt_cache_slots\":" + std::to_string(eng.prompt_cache_slots()) +
                ",\"memory_residency\":" + eng.memory_residency_json() +
                ",\"vision\":" + eng.vision_status_json() + "}",   // readiness of THIS load (P11), beside the arch flag in /capabilities
            "application/json");
    });
    // 501 (not 404): clients like Seal treat 501 as a permanent capability
    // miss and stop retrying; 404 makes them retry.
    srv.Post("/v1/embeddings", [](const httplib::Request&, httplib::Response& res) {
        res.status = 501;
        res.set_content("{\"error\":{\"message\":\"embeddings not implemented\"}}",
                        "application/json");
    });

    srv.Post("/v1/chat/completions", [&](const httplib::Request& req,
                                         httplib::Response& res) {
        auto cr = oai::parse_chat_request(req.body);
        if(cr.error.empty())cr.error=eng.reasoning_effort_error(cr.reasoning_effort);
        if (!cr.error.empty()) {
            res.status = 400;
            res.set_content(nlohmann::json{{"error",{{"message",cr.error}}}}.dump(),
                            "application/json");
            return;
        }
        const std::string id = "chatcmpl-" + std::to_string(++req_no);
        const int64_t created = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        if (!cr.stream) {
            if (fault.faulted()) { unavailable(res); return; }
            if (!adm.acquire())  { refuse(res); return; }
            AdmRelease adm_rel(&adm);
            // The client may have given up (harness timeout/retry) while this
            // request waited for admission. Generating for a closed socket
            // burns GPU minutes nobody receives — drop it before prefill.
            if (req.is_connection_closed()) {
                std::fprintf(stderr, "[req] client gone while queued — dropped\n");
                return;
            }
            try {
            // `stop` sequences (Phase L): watched on the streamed text; for a
            // deepseek4 thinking request only the part after </think> counts.
            const bool separate_thinking_ns =
                (eng.arch() == ModelArch::kDeepSeek4 || eng.arch() == ModelArch::kGlm5Next || eng.arch() == ModelArch::kDeepSeek41 || eng.arch() == ModelArch::kMimo26)
                && cr.enable_thinking;
            std::string acc_ns;
            bool stopped_ns = false;
            auto r = eng.chat(cr.turns, cr.sampling,
                // Poll the socket each decoded fragment: false → Engine abort
                // (finish_reason="abort"), releasing the engine gate for queued requests.
                [&](std::string_view t) {
                    // Also answers the engine's between-chunk liveness probe
                    // during prefill (empty fragment): stop iff the client left.
                    if (adm.stopping() || req.is_connection_closed()) return false;
                    if (cr.stop.empty()) return true;
                    acc_ns += t;
                    size_t cs = 0;
                    if (separate_thinking_ns) {
                        const size_t p = acc_ns.find("</think>");
                        if (p == std::string::npos) return true;
                        cs = p + 8;
                    }
                    // a forming tool-call block is never cut by a stop sequence
                    // (the stream path holds it back the same way)
                    static const std::string kToolOpen = "<tool_call>";
                    const std::string kDsmlOpen = eng.arch() == ModelArch::kDeepSeek41
                        ? "<｜DSML｜" : "<｜DSML｜tool_calls";
                    const size_t tool_at = std::min(acc_ns.find(kToolOpen, cs),
                                                    acc_ns.find(kDsmlOpen, cs));
                    if (first_stop_match(acc_ns, cr.stop, cs, tool_at) != std::string::npos) {
                        stopped_ns = true; return false;
                    }
                    return true;
                },
                cr.enable_thinking, cr.tools_json, cr.reasoning_effort);
            release_free_heap();
            if (r.finish_reason == "repetition") keep_repeating_request(req.body);
            if (stopped_ns) {
                r.finish_reason = "stop";
                r.tool_calls_json.clear();
                size_t cut = std::string::npos;
                for (const std::string& st : cr.stop) cut = std::min(cut, r.text.find(st));
                if (cut != std::string::npos) r.text.erase(cut);
            }
            if (r.finish_reason == "abort") {
                if (adm.stopping()) {   // aborted by shutdown, not by the client: say so
                    std::fprintf(stderr, "[req] aborted by shutdown after %u tok\n",
                                 r.completion_tokens);
                    refuse(res);
                    return;
                }
                std::fprintf(stderr,
                    "[req] client disconnected mid-generation — aborted after %u tok\n",
                    r.completion_tokens);
                return;
            }
            if (r.finish_reason == "context_length_exceeded") {
                res.status = 400;
                res.set_content(
                    "{\"error\":{\"message\":\"This model's maximum context length is "
                    + std::to_string(eng.max_ctx()) + " tokens, but your messages "
                    "resulted in " + std::to_string(r.prompt_tokens) + " tokens. Reduce "
                    "the input (or compact/summarize prior context) and retry.\","
                    "\"type\":\"invalid_request_error\",\"param\":\"messages\","
                    "\"code\":\"context_length_exceeded\"}}",
                    "application/json");
                return;
            }
            if (r.finish_reason.starts_with("error:")) {
                log_gen_error(id, r);
                // an image the CLIENT sent that cannot be used (undecodable bytes, a decode failure, an aspect ratio
                // above 200, placeholders that do not match the images) is the request's fault: 400, not a server error
                const bool client_image = r.finish_reason.find("vision:") != std::string::npos &&
                    (r.finish_reason.find("not a decodable image") != std::string::npos ||
                     r.finish_reason.find("image decode failed") != std::string::npos ||
                     r.finish_reason.find("aspect ratio") != std::string::npos ||
                     r.finish_reason.find("image is empty") != std::string::npos) ||
                    r.finish_reason.find("placeholders than images") != std::string::npos;
                if (client_image) {
                    res.status = 400;
                    res.set_content(oai::error_json(r.finish_reason, "invalid_request_error", "invalid_image"), "application/json");
                    return;
                }
                fault.observe(r.finish_reason);   // latch a lost device → /health 503
                exit_on_device_loss(fault);
                res.status = fault.faulted() ? 503 : 500;
                res.set_content(oai::error_json(r.finish_reason), "application/json");
                return;
            }
            log_gen_speed(r);
            res.set_content(oai::chat_completion_json(model_id, r, id, created),
                            "application/json");
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[chat] forward EXCEPTION: %s\n", e.what());
                fault.observe(e.what());
                exit_on_device_loss(fault);
                res.status = fault.faulted() ? 503 : 500;
                res.set_content(oai::error_json(std::string("forward: ") + e.what()),
                                "application/json");
            }
            return;
        }
        // Streaming: the whole generation runs inside the FIRST provider
        // invocation.  httplib v0.18.7 semantics (write_content_chunked):
        // sink.done() sets data_available=false so the provider is never
        // called again; the provider must return TRUE — returning false
        // maps to Error::Canceled even after done().
        // Admission is decided HERE, before the response starts, so a full
        // queue is a real 429 (headers not yet sent) instead of a 200 stream
        // that stalls; the slot is released when the provider is destroyed.
        if (fault.faulted()) { unavailable(res); return; }
        if (!adm.acquire())  { refuse(res); return; }
        auto adm_rel = std::make_shared<AdmRelease>(&adm);
        const std::string raw_body = req.body;   // kept for a repetition capture
        res.set_chunked_content_provider("text/event-stream",
            [&eng, &adm, &fault, adm_rel, cr, id, created, model_id, raw_body]
            (size_t, httplib::DataSink& sink) {
              // Phase L: a forward that throws mid-stream (the non-stream path
              // already catches) must end the stream with an error frame, not
              // the server.
              try {
                if (!sink.is_writable()) {   // client gone while queued — don't generate
                    std::fprintf(stderr, "[req] stream client gone while queued — dropped\n");
                    sink.done();
                    return true;
                }
                // gpt-oss is channel-structured (Harmony): its live output is raw
                // analysis/commentary, NOT literal <tool_call>. Buffer it and emit
                // the engine's post-processed r.text at the end (final-channel answer,
                // or the canonical <tool_call> Engine::chat produced); every other
                // arch streams live token-by-token exactly as before.
                const bool harmony = (eng.arch() == ModelArch::kGptOss);
                // Accumulate the generation so a tool call (Qwen <tool_call> text)
                // can be re-emitted as structured OpenAI tool_calls. Content is
                // streamed incrementally, but we hold back the tail (len of the
                // open tag) so a forming "<tool_call>" is caught before it leaks
                // as text; once a call starts we buffer until generation ends.
                // Flips false the moment a stream write fails (client disconnected).
                // The token callback then returns false → Engine::generate aborts
                // (finish_reason="abort") → eng.chat() returns → the admission slot
                // releases → the server stops grinding and unblocks queued requests.
                bool sink_alive = true;
                std::string acc;
                size_t streamed = 0;
                bool in_tool = false;
                static const std::string OPEN = "<tool_call>";
                // deepseek4 (Phase L): the engine streams the completion WITH
                // its special tokens, so the reasoning boundary (</think>) and
                // the DSML tool-calls marker are visible here.  Reasoning goes
                // out as delta.reasoning_content until </think>; from the DSML
                // marker on, the text is held back exactly like a forming
                // <tool_call>, and the engine's parsed tool_calls are emitted
                // at the end.
                const bool ds4 = (eng.arch() == ModelArch::kDeepSeek4);
                const bool ds41 = (eng.arch() == ModelArch::kDeepSeek41);
                const bool mimo = (eng.arch() == ModelArch::kMimo26);   // the engine parses MiMo's XML tool calls (docs/mimo26 P3b)
                const bool structured = ds4 || eng.arch() == ModelArch::kGlm5Next || eng.arch() == ModelArch::kDeepSeek41 || mimo;
                const std::string DSML_OPEN = ds41 ? "<｜DSML｜" : "<｜DSML｜tool_calls";
                static const std::string THINK_CLOSE = "</think>";
                bool   in_reason       = structured && cr.enable_thinking;
                bool   in_dsml         = false;   // the held-back block is a DSML tool call
                size_t content_start   = 0;   // where content begins in `acc`
                size_t reason_streamed = 0;
                bool   stopped         = false;   // a `stop` sequence was hit
                size_t hold = (ds4 || ds41) ? std::max(OPEN.size(), DSML_OPEN.size()) : OPEN.size();
                for (const std::string& st : cr.stop) hold = std::max(hold, st.size());
                // Clamp a byte offset down to a complete-UTF-8 boundary: a
                // streamed delta must never split a multi-byte character, or
                // json::dump throws type_error.316 and aborts the server.
                auto utf8_safe = [](const std::string& s, size_t end) -> size_t {
                    if (end > s.size()) end = s.size();
                    size_t i = end;
                    while (i > 0 && (static_cast<unsigned char>(s[i - 1]) & 0xC0) == 0x80) --i;
                    if (i == 0) return end;
                    const unsigned char lead = static_cast<unsigned char>(s[i - 1]);
                    size_t need = 1;
                    if ((lead & 0xE0) == 0xC0) need = 2;
                    else if ((lead & 0xF0) == 0xE0) need = 3;
                    else if ((lead & 0xF8) == 0xF0) need = 4;
                    return (end - (i - 1) >= need) ? end : (i - 1);
                };
                auto flush_to = [&](size_t upto) {
                    upto = utf8_safe(acc, upto);
                    if (upto > streamed) {
                        auto f = oai::chat_chunk_sse(model_id, id, created,
                            std::string_view(acc).substr(streamed, upto - streamed), "");
                        if (sink.write(f.data(), f.size())) streamed = upto;
                        else sink_alive = false;   // client gone — stop generating
                    }
                };
                // Opt-in (`stream_tool_preview`): while a tool call is being written,
                // send its raw text as delta.tool_call_preview so the client can show
                // the code appearing. It is display only -- the same bytes arrive at
                // the end as structured tool_calls -- and it never touches content.
                size_t tool_streamed = 0;
                auto flush_tool_preview_to = [&](size_t upto) {
                    upto = utf8_safe(acc, upto);
                    if (upto > tool_streamed) {
                        auto f = oai::chat_chunk_sse_tool_preview(model_id, id, created,
                            std::string_view(acc).substr(tool_streamed, upto - tool_streamed));
                        if (sink.write(f.data(), f.size())) tool_streamed = upto;
                        else sink_alive = false;
                    }
                };
                auto flush_reason_to = [&](size_t upto) {
                    upto = utf8_safe(acc, upto);
                    if (upto > reason_streamed) {
                        auto f = oai::chat_chunk_sse_reasoning(model_id, id, created,
                            std::string_view(acc).substr(reason_streamed, upto - reason_streamed));
                        if (sink.write(f.data(), f.size())) reason_streamed = upto;
                        else sink_alive = false;
                    }
                };
                auto r = eng.chat(cr.turns, cr.sampling,
                    [&](std::string_view t) {
                        if (!sink_alive || adm.stopping()) return false;  // client gone /
                                                        // server stopping → abort (frees the gate)
                        // Empty fragment = the engine's between-chunk liveness
                        // probe during prefill: nothing has been written yet, so
                        // only the socket can say whether the client is still there.
                        if (t.empty()) return bool(sink.is_writable());
                        // Harmony (gpt-oss) buffers everything, so a write failure
                        // can never flip sink_alive: poll the socket instead.
                        if (harmony) return bool(sink.is_writable());   // buffered → emit r.text at end
                        acc += t;
                        if (in_reason) {
                            const size_t p = acc.find(THINK_CLOSE);
                            if (p == std::string::npos) {   // hold the tag's length back
                                flush_reason_to(acc.size() > THINK_CLOSE.size()
                                                    ? acc.size() - THINK_CLOSE.size() : 0);
                                return sink_alive;
                            }
                            flush_reason_to(p);
                            if (!sink_alive) return false;
                            in_reason     = false;
                            content_start = p + THINK_CLOSE.size();
                            streamed      = content_start;
                        }
                        if (in_tool) {                            // buffering the call
                            if (cr.stream_tool_preview) flush_tool_preview_to(acc.size());
                            return sink_alive;
                        }
                        size_t tc = acc.find(OPEN, content_start);
                        if (ds4 || ds41) {
                            const size_t td = acc.find(DSML_OPEN, content_start);
                            if (td < tc) { tc = td; in_dsml = true; }
                        }
                        const size_t stop_at = first_stop_match(acc, cr.stop, content_start, tc);
                        if (stop_at != std::string::npos) {
                            flush_to(stop_at); stopped = true; return false;
                        }
                        if (tc != std::string::npos) {
                            // V4.1's calls block opens with "\n\n<｜DSML｜ calls>": the blank line is part of the
                            // delimiter (encoding.py parse_message_from_completion_text), not content
                            size_t cut = tc;
                            if (ds41 && in_dsml)
                                for (int k = 0; k < 2 && cut > std::max(content_start, streamed) && acc[cut - 1] == '\n'; ++k) --cut;
                            flush_to(cut); in_tool = true; tool_streamed = cut;
                            if (cr.stream_tool_preview && sink_alive) flush_tool_preview_to(acc.size());
                            return sink_alive;
                        }
                        flush_to(acc.size() > hold ? acc.size() - hold : 0);
                        return sink_alive;
                    }, cr.enable_thinking, cr.tools_json, cr.reasoning_effort);
                release_free_heap();
                if (r.finish_reason == "repetition") keep_repeating_request(raw_body);
                if (r.finish_reason == "context_length_exceeded") {
                    std::string ev =
                        "data: {\"error\":{\"message\":\"context_length_exceeded: prompt "
                        + std::to_string(r.prompt_tokens) + " tokens exceeds limit "
                        + std::to_string(eng.max_ctx()) + "\",\"type\":\"invalid_request_error\","
                        "\"code\":\"context_length_exceeded\"}}\n\n";
                    sink.write(ev.data(), ev.size());
                }
                if (r.finish_reason.starts_with("error:")) {
                    log_gen_error(id, r);
                    fault.observe(r.finish_reason);   // latch a lost device → /health 503
                    exit_on_device_loss(fault);
                    const std::string ev = "data: " + oai::error_json(r.finish_reason)
                        + "\n\ndata: [DONE]\n\n";
                    sink.write(ev.data(), ev.size());
                    sink.done();
                    return true;
                }
                if (r.finish_reason == "abort")
                    std::fprintf(stderr, adm.stopping()
                        ? "[req] stream aborted by shutdown after %u tok\n"
                        : "[req] stream client disconnected mid-generation — aborted after %u tok\n",
                        r.completion_tokens);
                log_gen_speed(r);
                // If the buffered text holds valid tool call(s), emit them as a
                // structured delta + finish_reason "tool_calls". Otherwise flush
                // any held-back tail (incl. an unparseable <tool_call>) as content.
                // Harmony (gpt-oss) reassembles from the engine's post-processed
                // r.text — the final-channel answer, or the canonical <tool_call>
                // Engine::chat produced; other arches use the streamed accumulator.
                std::string tcframe = stopped ? std::string{} : (structured && !r.tool_calls_json.empty())
                    ? oai::chat_chunk_sse_tool_calls_json(model_id, id, created, r.tool_calls_json)
                    : (ds41 || mimo) ? std::string{} : oai::chat_chunk_sse_tool_calls(model_id, id, created, harmony ? r.text : acc);
                std::string fin_reason = stopped ? "stop" : r.finish_reason;
                if (!tcframe.empty()) {
                    sink.write(tcframe.data(), tcframe.size());
                    fin_reason = "tool_calls";
                } else if (harmony) {
                    auto c = oai::chat_chunk_sse(model_id, id, created, r.text, "");
                    sink.write(c.data(), c.size());
                } else if (!stopped) {
                    if (in_reason) flush_reason_to(acc.size());   // cut off inside the reasoning
                    else if (in_dsml) {
                        // a DSML tool-call block the engine could not parse (cut by
                        // max_tokens): mirror ds4_finish_completion — the prose before
                        // the block was streamed at detection, the block itself is dropped
                        // and finish_reason says the output was cut (gate L finding 1)
                    } else if (ds41) flush_to(content_start + r.text.size());
                    else flush_to(acc.size());
                }
                auto fin = oai::chat_chunk_sse(model_id, id, created, "", fin_reason,
                                               fin_reason == "length" ? r.truncated_tool_call : std::string{});
                sink.write(fin.data(), fin.size());
                // Real token usage so streaming clients get live context % + tok/s
                // (without this the client accounts zeros).
                auto us = oai::chat_chunk_sse_usage(model_id, id, created,
                                                    r.prompt_tokens, r.completion_tokens,
                                                    r.cached_tokens);
                sink.write(us.data(), us.size());
                static const std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();
              } catch (const std::exception& e) {
                std::fprintf(stderr, "[chat] stream forward EXCEPTION: %s\n", e.what());
                fault.observe(e.what());
                exit_on_device_loss(fault);
                std::string ev = "data: " + oai::error_json(std::string("forward: ") + e.what())
                                 + "\n\ndata: [DONE]\n\n";
                sink.write(ev.data(), ev.size());
                sink.done();
              }
                return true;    // provider complete; done() ends the loop
            });
    });

    // Bind BEFORE announcing, so a failure (almost always: port already in use) is
    // reported clearly instead of printing a misleading "serving" line and then
    // exiting silently — the papercut that makes `ie serve` look like it "doesn't
    // load". A launcher/manager can parse this exit code (1) + stderr to react.
    if (!srv.bind_to_port(host, port)) {
        std::fprintf(stderr,
            "[ie] ERROR: could not bind %s:%d — the port is already in use.\n"
            "      Free it (`ss -ltnp | grep :%d` to find the holder) or pick another"
            " with --port <N>.\n",
            host.c_str(), port, port);
        return 1;
    }
    std::printf("[ie] serving %s on http://%s:%d/v1 (ctx %u, parallel %u, max-queue %u)\n",
                model_id.c_str(), host.c_str(), port, eng.max_ctx(), eng.parallel(), max_queue);
    std::fflush(stdout);
    // Orderly stop on SIGTERM/SIGINT (see on_stop_signal): the watcher turns the
    // recorded signal into abort + refuse + listener close, and exits by itself
    // once the listener has returned for any reason (/admin/shutdown).
    std::signal(SIGTERM, on_stop_signal);
    std::signal(SIGINT,  on_stop_signal);
    std::atomic<bool> listening{true};
    std::thread watcher([&] {
        while (listening.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            const int sig = g_signal.load();
            if (sig == 0 && !admin_stop.load()) continue;
            std::fprintf(stderr, "[ie] %s — stopping: aborting %u in-flight, refusing %u queued\n",
                         sig ? ("signal " + std::to_string(sig)).c_str() : "admin shutdown",
                         adm.inflight(), adm.queued());
            adm.shutdown();   // in-flight generations abort at their next token
            srv.stop();
            return;
        }
    });
    const bool ok = srv.listen_after_bind();
    listening.store(false);
    watcher.join();
    std::signal(SIGTERM, SIG_DFL);
    std::signal(SIGINT,  SIG_DFL);
    return ok ? 0 : 1;
}

}  // namespace ie
