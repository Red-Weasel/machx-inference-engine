// include/ie/openai_proto.hpp — pure request/response translation for the
// OpenAI-compatible server.  No GPU, no engine dependency: unit-testable.
#pragma once
#include "ie/engine.hpp"
#include "ie/tokenizer.hpp"
#include <string>
#include <vector>

namespace ie::oai {

struct ChatRequest {
    std::vector<ChatTurn> turns;
    SamplingParams sampling;
    bool stream = false;
    bool enable_thinking = false;  // Qwen reasoning trace; opt-in per request
    std::string reasoning_effort;  // empty keeps the selected model's default
    std::string model;        // echoed back, not used for routing in v1
    std::string tools_json;   // raw OpenAI `tools` array (dumped); empty = none
    std::vector<std::string> stop;   // OpenAI `stop`: up to 4 sequences, empty = none
    std::string error;        // non-empty => 400 with this message
};

// Immutable startup defaults; configure before accepting HTTP requests.
// Environment reader throws for invalid IE_SERVE_* numeric values.
ChatRequest server_defaults_from_environment();
void configure_server_defaults(const ChatRequest& defaults);

// Parse /v1/chat/completions body. Maps: messages[] -> turns,
// temperature/top_p/top_k/max_tokens/seed/stream -> fields.
ChatRequest parse_chat_request(const std::string& body);

// Non-streaming response body.
std::string chat_completion_json(const std::string& model,
                                 const GenerateResult& r,
                                 const std::string& id, int64_t created);

// One SSE data frame carrying a `reasoning_content` delta (deepseek4 thinking).
std::string chat_chunk_sse_reasoning(const std::string& model, const std::string& id,
                                     int64_t created, std::string_view delta);
// One SSE data frame carrying structured tool_calls from an engine-parsed
// OpenAI-format array (GenerateResult::tool_calls_json); "" when the array is
// empty or unparseable.
std::string chat_chunk_sse_tool_calls_json(const std::string& model, const std::string& id,
                                           int64_t created, const std::string& tool_calls_json);

// One SSE data frame ("data: {...}\n\n") carrying a content delta; when
// finish_reason is non-empty the delta is empty and the frame closes the
// stream (caller then sends "data: [DONE]\n\n").
std::string chat_chunk_sse(const std::string& model, const std::string& id,
                           int64_t created, std::string_view delta,
                           const std::string& finish_reason);

// Streaming tool_calls delta: parses the accumulated generation's <tool_call>
// text into structured OpenAI tool_calls. Returns "" if no valid call is
// present (caller streams the buffered text as ordinary content instead).
std::string chat_chunk_sse_tool_calls(const std::string& model, const std::string& id,
                                      int64_t created, const std::string& full_text);

// Final streaming usage chunk (empty choices + a `usage` object), emitted just
// before `data: [DONE]` so streaming clients get real prompt/completion token
// counts — without it the client sees zeros (dead context % + tok/s).
std::string chat_chunk_sse_usage(const std::string& model, const std::string& id,
                                 int64_t created, uint32_t prompt_tokens,
                                 uint32_t completion_tokens,
                                 uint32_t cached_tokens = 0);

std::string models_json(const std::string& model_id);

// {"error":{"message":..,"type":..[,"code":..]}} with the message JSON-escaped
// (invalid UTF-8 replaced, never thrown on) — for every hand-built error body.
std::string error_json(std::string_view message,
                       std::string_view type = "server_error",
                       std::string_view code = {});

}  // namespace ie::oai
