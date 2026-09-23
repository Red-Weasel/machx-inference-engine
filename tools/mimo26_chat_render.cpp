// tools/mimo26_chat_render.cpp -- ie-mimo26-chat-render: the engine's MiMo-V2.6 chat-template rendering (and completion
// parse) for tools/mimo26/check_template.py to diff against the checkpoint's own Jinja template (P3b).
//   ie-mimo26-chat-render render <case.json>      -> the prompt text on stdout
//   ie-mimo26-chat-render parse  <case.json>      -> {"reasoning", "content", "tool_calls"} as JSON on stdout
// case.json: {"messages": [{"role", "content", "reasoning_content"?, "tool_calls"? (OpenAI array), "images"? (count; the content
//             holds the server's <<ie-image>> marker per image, P6.2)}], "tools": [...]?,
//             "thinking": bool, "completion": "..." (parse only)}
#include "ie/mimo26_engine.hpp"

#include "../third_party/nlohmann/json.hpp"

#include <cstdio>
#include <fstream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 3) { std::fprintf(stderr, "usage: ie-mimo26-chat-render (render|parse) <case.json>\n"); return 2; }
    std::ifstream f(argv[2]);
    const nlohmann::ordered_json c = nlohmann::ordered_json::parse(f, nullptr, false);   // key order kept, as the server keeps it
    if (c.is_discarded()) { std::fprintf(stderr, "bad JSON\n"); return 2; }
    const bool thinking = c.value("thinking", true);
    const std::string tools = c.contains("tools") ? c["tools"].dump() : std::string();
    if (std::string(argv[1]) == "render") {
        std::vector<ie::Mimo26Message> msgs;
        for (const auto& m : c["messages"]) {
            ie::Mimo26Message x;
            x.role = m.value("role", "");
            x.content = m.contains("content") && m["content"].is_string() ? m["content"].get<std::string>() : std::string();
            // P6.2: "images": N -- the server's kChatImageMarker in the content stands where each image part sat
            if (m.contains("images") && m["images"].is_number_unsigned()) x.content = ie::mimo26_place_images(x.content, m["images"].get<size_t>());
            x.reasoning = m.contains("reasoning_content") && m["reasoning_content"].is_string() ? m["reasoning_content"].get<std::string>() : std::string();
            if (m.contains("tool_calls")) x.tool_calls_json = m["tool_calls"].dump();
            msgs.push_back(std::move(x));
        }
        std::string err;
        const std::string out = ie::mimo26_render_chat(msgs, tools, true, thinking, err);
        if (!err.empty()) { std::fprintf(stderr, "render: %s\n", err.c_str()); return 1; }
        std::fwrite(out.data(), 1, out.size(), stdout);
        return 0;
    }
    const auto p = ie::mimo26_parse_completion(c.value("completion", ""), thinking, tools);
    nlohmann::json o = {{"reasoning", p.reasoning}, {"content", p.content}, {"malformed", p.malformed_call},
                        {"tool_calls", p.tool_calls_json.empty() ? nlohmann::json::array() : nlohmann::json::parse(p.tool_calls_json)}};
    std::printf("%s\n", o.dump().c_str());
    return 0;
}
