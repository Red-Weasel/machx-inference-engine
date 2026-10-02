// include/ie/openai_server.hpp — OpenAI-compatible HTTP server over one
// Engine.  Single model. Admission: up to EngineOptions::parallel generations
// run at once (the engine's own FIFO gate serializes/interleaves the GPU work),
// at most `max_queue` more wait, anything beyond gets HTTP 429 immediately.
// /health answers 200 {"status":"ok",inflight,queued,...} while the device is
// healthy and 503 once a forward reported a lost/reset device (the process
// must be restarted; see server_admission.hpp). SIGTERM/SIGINT abort in-flight
// generations, stop the listener and return from run_openai_server so every
// destructor runs (same as POST /admin/shutdown from loopback). P4 B33: the abort
// does not wait for queued device work (Engine::abort_all).
//
// Non-standard request fields accepted by /v1/chat/completions:
//   enable_thinking (bool; default from the server's --thinking / env, on
//     unless configured off): the model's reasoning trace, returned as
//     reasoning_content (streamed: delta.reasoning_content) by deepseek4 /
//     deepseek41 / glm5next / mimo26 and, since P4 B35, the Qwen thinking
//     templates (reasoning.hpp chatml_think_split); other templates keep it
//     in content.
//
// max_tokens semantics: when a request omits max_tokens (or sends null),
// generation runs until EOS or the context budget (kMaxTokensUnlimited in
// engine.hpp) per standard OpenAI server behavior — NOT the SamplingParams
// library default of 512. Explicit client values are honored as-is.
#pragma once
#include "ie/engine.hpp"
#include <chrono>
#include <string>
namespace ie {
// P4 B33: how the server's orderly stop began, for the caller's "[ie] stopped in X.X s (waited for N in-flight GPU steps)"
// line once the engine is destroyed: requested = a signal or /admin/shutdown stopped it; t0 = when the stop arrived
// (steady clock); gpu_steps = the device steps still running then (Engine::abort_all).
struct ServerStop {
    bool requested = false;
    std::chrono::steady_clock::time_point t0{};
    uint32_t gpu_steps = 0;
};
// Blocks until SIGTERM/SIGINT or POST /admin/shutdown. Returns non-zero on
// bind failure. `max_queue` = requests allowed to wait for a generation slot.
// `root_id` (non-empty when a layout names the server): /v1/models reports it as
// "root" beside the served name `model_id` (docs/serve_config.md). `stop` (optional) receives the stop's start.
int run_openai_server(Engine& eng, const std::string& model_id,
                      const std::string& host, int port,
                      uint32_t max_queue = 8, const std::string& root_id = "", ServerStop* stop = nullptr);
}
