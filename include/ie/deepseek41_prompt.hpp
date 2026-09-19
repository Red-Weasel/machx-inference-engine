// V4.1 text and native tool protocol. Images and internal task roles remain unsupported.
#pragma once
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace ie {

struct Ds41ChatMessage {
    std::string role;                 // "system" | "user" | "assistant"
    std::string content;
    std::string reasoning_content;    // assistant turns only; rendered per the drop-thinking rule
    bool has_images = false;          // the turn carried image content: refused (a prompt without it would be a fake)
    bool has_tool_calls = false;      // requires structured tool_calls_json below
    std::string tool_calls_json;
    std::string tool_call_id;
    std::string tools_json;          // per-message definitions, system role only
};

struct Ds41PromptOptions {
    bool thinking = true;             // thinking_mode: "thinking" (true) or "chat" (false)
    int  reasoning_effort = 75;       // 1..100; the reference's aliases: low 50, high 75 (default), max 100
    bool drop_thinking = true;        // drop reasoning_content of assistant turns before the last user turn
    bool add_bos = true;              // the leading <｜begin▁of▁sentence｜>
    bool has_tools = false;           // requires tools_json when true
    std::string tools_json;           // OpenAI tools array, attached to initial system turn
};

// "low" / "high" / "max" or a decimal 1..100; -1 with `err` set otherwise.
int ds41_reasoning_effort_of(const std::string& s, std::string& err);

// The prompt string for `msgs` (the model's own special-token text; tokenise with allow_special),
// or "" with `err` set when a message needs something this port refuses.
std::string ds41_encode_messages(const std::vector<Ds41ChatMessage>& msgs, const Ds41PromptOptions& o, std::string& err);

struct Ds41Completion {
    std::string content, reasoning_content, tool_calls_json, error;
};
// Parse only completed generations. A truncated call must never be executed.
Ds41Completion ds41_parse_completion(std::string_view text, bool thinking);
// Hide DSML (including a partial marker) from incomplete replies.
std::string ds41_visible_content(std::string_view text);
// A reply cut off inside a tool call: the call's name ("unknown" before the name was written), "" when the reply
// holds no DSML block. The call itself is never parsed or run.
std::string ds41_cut_tool_name(std::string_view text);

}  // namespace ie
