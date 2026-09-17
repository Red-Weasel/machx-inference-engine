// include/ie/openai_server.hpp — OpenAI-compatible HTTP server over one
// Engine.  Single model. Admission: up to EngineOptions::parallel generations
// run at once (the engine's own FIFO gate serializes/interleaves the GPU work),
// at most `max_queue` more wait, anything beyond gets HTTP 429 immediately.
// /health answers 200 {"status":"ok",inflight,queued,...} while the device is
// healthy and 503 once a forward reported a lost/reset device (the process
// must be restarted; see server_admission.hpp). SIGTERM/SIGINT abort in-flight
// generations, stop the listener and return from run_openai_server so every
// destructor runs (same as POST /admin/shutdown from loopback).
//
// Non-standard request fields accepted by /v1/chat/completions:
//   enable_thinking (bool; default from the server's --thinking / env, on
//     unless configured off): Qwen reasoning trace; when true the raw <think>
//     text appears in content (deepseek4/glm5next: delta.reasoning_content).
//
// max_tokens semantics: when a request omits max_tokens (or sends null),
// generation runs until EOS or the context budget (kMaxTokensUnlimited in
// engine.hpp) per standard OpenAI server behavior — NOT the SamplingParams
// library default of 512. Explicit client values are honored as-is.
#pragma once
#include "ie/engine.hpp"
#include <string>
namespace ie {
// Blocks until SIGTERM/SIGINT or POST /admin/shutdown. Returns non-zero on
// bind failure. `max_queue` = requests allowed to wait for a generation slot.
int run_openai_server(Engine& eng, const std::string& model_id,
                      const std::string& host, int port,
                      uint32_t max_queue = 8);
}
