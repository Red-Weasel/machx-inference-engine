// include/ie/openai_server.hpp — OpenAI-compatible HTTP server over one
// Engine.  Single model; one generation at a time (mutex; requests queue).
//
// Non-standard request fields accepted by /v1/chat/completions:
//   enable_thinking (bool, default false): Qwen reasoning trace; when true
//     the raw <think> text appears in content.
//
// max_tokens semantics: when a request omits max_tokens (or sends null),
// generation runs until EOS or the context budget (kMaxTokensUnlimited in
// engine.hpp) per standard OpenAI server behavior — NOT the SamplingParams
// library default of 512. Explicit client values are honored as-is.
#pragma once
#include "ie/engine.hpp"
#include <string>
namespace ie {
// Blocks until the process is killed. Returns non-zero on bind failure.
int run_openai_server(Engine& eng, const std::string& model_id,
                      const std::string& host, int port);
}
