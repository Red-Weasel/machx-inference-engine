#include "ie/openai_server.hpp"
#include "ie/openai_proto.hpp"
#include "httplib/httplib.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>

namespace ie {

// Per-request throughput to the engine's terminal (stderr). Free: token counts
// already tracked, timing is phase-boundary clock reads (see Engine::generate).
static void log_gen_speed(const GenerateResult& r) {
    const double pf = r.prefill_ms > 0 ? r.prompt_tokens * 1000.0 / r.prefill_ms : 0.0;
    const double dc = r.decode_ms  > 0 ? r.completion_tokens * 1000.0 / r.decode_ms : 0.0;
    std::fprintf(stderr,
        "[gen] prefill %u tok (%u cached) / %.0f ms = %.0f tok/s  |  decode %u tok / %.0f ms = %.1f tok/s\n",
        r.prompt_tokens, r.cached_tokens, r.prefill_ms, pf, r.completion_tokens, r.decode_ms, dc);
    std::fflush(stderr);
}

int run_openai_server(Engine& eng, const std::string& model_id,
                      const std::string& host, int port) {
    httplib::Server srv;
    std::mutex gen_mu;                 // one generation at a time (v1)
    std::atomic<uint64_t> req_no{0};

    srv.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    srv.Get("/v1/models", [&](const httplib::Request&, httplib::Response& res) {
        res.set_content(oai::models_json(model_id), "application/json");
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
        if (!cr.error.empty()) {
            res.status = 400;
            res.set_content("{\"error\":{\"message\":\"" + cr.error + "\"}}",
                            "application/json");
            return;
        }
        const std::string id = "chatcmpl-" + std::to_string(++req_no);
        const int64_t created = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        if (!cr.stream) {
            std::lock_guard<std::mutex> lk(gen_mu);
            try {
            auto r = eng.chat(cr.turns, cr.sampling, {}, cr.enable_thinking,
                              cr.tools_json);
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
            log_gen_speed(r);
            res.set_content(oai::chat_completion_json(model_id, r, id, created),
                            "application/json");
            } catch (const std::exception& e) {
                std::fprintf(stderr, "[chat] forward EXCEPTION: %s\n", e.what());
                res.status = 500;
                res.set_content(std::string("{\"error\":{\"message\":\"forward: ")
                                + e.what() + "\"}}", "application/json");
            }
            return;
        }
        // Streaming: the whole generation runs inside the FIRST provider
        // invocation.  httplib v0.18.7 semantics (write_content_chunked):
        // sink.done() sets data_available=false so the provider is never
        // called again; the provider must return TRUE — returning false
        // maps to Error::Canceled even after done().
        res.set_chunked_content_provider("text/event-stream",
            [&eng, &gen_mu, cr, id, created, model_id]
            (size_t, httplib::DataSink& sink) {
                std::lock_guard<std::mutex> lk(gen_mu);
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
                std::string acc;
                size_t streamed = 0;
                bool in_tool = false;
                static const std::string OPEN = "<tool_call>";
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
                        sink.write(f.data(), f.size());
                        streamed = upto;
                    }
                };
                auto r = eng.chat(cr.turns, cr.sampling,
                    [&](std::string_view t) {
                        if (harmony) return true;     // buffered → emit r.text at end
                        acc += t;
                        if (in_tool) return true;                 // buffering the call
                        size_t tc = acc.find(OPEN);
                        if (tc != std::string::npos) { flush_to(tc); in_tool = true; return true; }
                        const size_t hold = OPEN.size();
                        flush_to(acc.size() > hold ? acc.size() - hold : 0);
                        return true;
                    }, cr.enable_thinking, cr.tools_json);
                if (r.finish_reason == "context_length_exceeded") {
                    std::string ev =
                        "data: {\"error\":{\"message\":\"context_length_exceeded: prompt "
                        + std::to_string(r.prompt_tokens) + " tokens exceeds limit "
                        + std::to_string(eng.max_ctx()) + "\",\"type\":\"invalid_request_error\","
                        "\"code\":\"context_length_exceeded\"}}\n\n";
                    sink.write(ev.data(), ev.size());
                }
                log_gen_speed(r);
                // If the buffered text holds valid tool call(s), emit them as a
                // structured delta + finish_reason "tool_calls". Otherwise flush
                // any held-back tail (incl. an unparseable <tool_call>) as content.
                // Harmony (gpt-oss) reassembles from the engine's post-processed
                // r.text — the final-channel answer, or the canonical <tool_call>
                // Engine::chat produced; other arches use the streamed accumulator.
                std::string tcframe = oai::chat_chunk_sse_tool_calls(
                    model_id, id, created, harmony ? r.text : acc);
                std::string fin_reason = r.finish_reason;
                if (!tcframe.empty()) {
                    sink.write(tcframe.data(), tcframe.size());
                    fin_reason = "tool_calls";
                } else if (harmony) {
                    auto c = oai::chat_chunk_sse(model_id, id, created, r.text, "");
                    sink.write(c.data(), c.size());
                } else {
                    flush_to(acc.size());
                }
                auto fin = oai::chat_chunk_sse(model_id, id, created, "", fin_reason);
                sink.write(fin.data(), fin.size());
                // Real token usage so streaming clients get live context % + tok/s
                // (without this the client accounts zeros).
                auto us = oai::chat_chunk_sse_usage(model_id, id, created,
                                                    r.prompt_tokens, r.completion_tokens);
                sink.write(us.data(), us.size());
                static const std::string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();
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
    std::printf("[ie] serving %s on http://%s:%d/v1\n",
                model_id.c_str(), host.c_str(), port);
    std::fflush(stdout);
    return srv.listen_after_bind() ? 0 : 1;
}

}  // namespace ie
