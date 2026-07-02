#include "ie/openai_proto.hpp"
#include "nlohmann/json.hpp"

#include <cctype>
#include <string>

using nlohmann::json;
using ojson = nlohmann::ordered_json;   // preserves client key order (tools)

namespace ie::oai {

// Render one OpenAI tool_call object into the Qwen text form:
//   <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>
// `arguments` arrives as a JSON-escaped string per OpenAI; embed it as raw
// JSON (parse-and-redump if it parses, else embed as-is). Returns false on a
// structurally invalid tool_call.
static bool render_tool_call(const ojson& tc, std::string& out,
                             std::string& err) {
    if (!tc.is_object() || !tc.contains("function") || !tc["function"].is_object())
        return false;
    const auto& fn = tc["function"];
    if (!fn.contains("name") || !fn["name"].is_string()) return false;
    {
        const std::string& nm = fn["name"].get_ref<const std::string&>();
        if (nm.find('<') != std::string::npos || nm.find('>') != std::string::npos) {
            err = "invalid tool name"; return false;
        }
    }
    std::string args = "{}";
    if (fn.contains("arguments")) {
        const auto& av = fn["arguments"];
        if (av.is_string()) {
            ojson parsed = ojson::parse(av.get<std::string>(), nullptr,
                                        /*allow_exceptions=*/false);
            args = parsed.is_discarded() ? av.get<std::string>() : parsed.dump();
        } else if (!av.is_null()) {
            args = av.dump();           // already-structured arguments
        }
    }
    out += "<tool_call>\n{\"name\": ";
    out += ojson(fn["name"].get<std::string>()).dump();
    out += ", \"arguments\": ";
    out += args;
    out += "}\n</tool_call>";
    return true;
}

ChatRequest parse_chat_request(const std::string& body) {
    ChatRequest out;
    ojson j = ojson::parse(body, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_object()) { out.error = "invalid JSON"; return out; }
    out.model           = j.value("model", "default");
    out.stream          = j.value("stream", false);
    out.enable_thinking = j.value("enable_thinking", false);
    auto& sp = out.sampling;
    sp.temperature = j.value("temperature", sp.temperature);
    sp.top_p       = j.value("top_p", sp.top_p);
    sp.top_k       = j.value("top_k", sp.top_k);
    sp.min_p       = j.value("min_p", sp.min_p);
    // The sampler exposes a single repetition penalty (repeat_penalty); accept the
    // llama.cpp name and the OpenAI-ish "repetition_penalty" alias. (OpenAI's
    // frequency/presence penalties are a different scheme the sampler does not
    // implement, so they are intentionally NOT parsed here.)
    sp.repeat_penalty = j.value("repeat_penalty",
                                j.value("repetition_penalty", sp.repeat_penalty));
    if (j.contains("max_tokens") && j["max_tokens"].is_number()) {
        auto mt = j["max_tokens"].get<int64_t>();
        if (mt < 0) { out.error = "max_tokens must be >= 0"; return out; }
        sp.max_tokens = uint32_t(mt);
    } else {
        // OpenAI server semantics: omitted (or null) max_tokens means
        // "generate until EOS or the context budget" — not the library
        // default of 512, which truncated long tool calls mid-JSON with
        // finish_reason="length". Engine::generate clamps to the ctx budget.
        sp.max_tokens = kMaxTokensUnlimited;
    }
    if (j.contains("seed") && j["seed"].is_number())
        sp.seed = uint64_t(j["seed"].get<int64_t>());
    if (j.contains("tools") && !j["tools"].is_null()) {
        if (!j["tools"].is_array()) { out.error = "tools must be an array"; return out; }
        for (const auto& tool : j["tools"]) {
            if (tool.is_object() && tool.contains("function") &&
                tool["function"].is_object() &&
                tool["function"].contains("name") &&
                tool["function"]["name"].is_string()) {
                const std::string& nm = tool["function"]["name"].get_ref<const std::string&>();
                if (nm.find('<') != std::string::npos || nm.find('>') != std::string::npos) {
                    out.error = "invalid tool name"; return out;
                }
            }
        }
        if (!j["tools"].empty()) out.tools_json = j["tools"].dump();
    }
    if (!j.contains("messages") || !j["messages"].is_array() || j["messages"].empty()) {
        out.error = "messages[] required";
        return out;
    }
    for (auto& m : j["messages"]) {
        if (!m.is_object() || !m.contains("role") || !m["role"].is_string()) {
            out.error = "each message needs string role and string/array content";
            return out;
        }
        const std::string role = m["role"].get<std::string>();
        const bool has_tool_calls = role == "assistant" &&
            m.contains("tool_calls") && m["tool_calls"].is_array() &&
            !m["tool_calls"].empty();

        std::string content;
        const bool content_missing =
            !m.contains("content") || m["content"].is_null();
        if (content_missing) {
            // Strict-OpenAI assistant tool_calls messages may omit content.
            if (!has_tool_calls) {
                out.error = "each message needs string role and string/array content";
                return out;
            }
        } else {
            const auto& cv = m["content"];
            if (cv.is_string()) {
                content = cv.get<std::string>();
            } else if (cv.is_array()) {
                // OpenAI multi-part content: concatenate text parts in order.
                for (const auto& part : cv) {
                    if (part.is_object() &&
                        part.value("type", "") == "text" &&
                        part.contains("text") && part["text"].is_string()) {
                        content += part["text"].get<std::string>();
                    }
                }
            } else {
                out.error = "each message needs string role and string/array content";
                return out;
            }
        }
        if (has_tool_calls) {
            // Assistant tool calls become text <tool_call> blocks the Qwen
            // template embeds verbatim in the assistant turn.
            for (const auto& tc : m["tool_calls"]) {
                if (!content.empty()) content += "\n";
                std::string tc_err;
                if (!render_tool_call(tc, content, tc_err)) {
                    out.error = tc_err.empty()
                        ? "tool_calls entries need a function object with string name"
                        : tc_err;
                    return out;
                }
            }
        }
        // role:"tool" messages (tool_call_id ignored) pass through as tool
        // turns; the template renders them as <tool_response> user blocks.
        out.turns.push_back({role, std::move(content)});
    }
    return out;
}

static json base(const std::string& model, const std::string& id,
                 int64_t created, const char* object) {
    return json{{"id", id}, {"object", object}, {"created", created},
                {"model", model}};
}

// LENIENT fallback for a <tool_call> body that fails strict json::parse. Qwen
// fine-tunes routinely imitate the instruction template's example
// `{"name": <function-name>, "arguments": <args-json-object>}` LITERALLY — i.e.
// they emit the name UNQUOTED (`{"name": web_search, ...}`) and/or drop the outer
// closing brace. Strict parse rejects both, so the call was leaking back as raw
// text (the agent harness sees no tool_calls → stalls). This extracts the name
// (quoted OR a bare identifier) and the brace-balanced `arguments` object
// independently of the outer object's well-formedness. Returns false if no name
// is recoverable. `args_out` is always a parseable JSON-object string ("{}" if
// absent/unparseable). Numerics/strict path unchanged — this only runs after the
// strict parse has already failed.
static bool lenient_tool_call(const std::string& inner, std::string& name_out,
                              std::string& args_out) {
    auto skip_ws = [](const std::string& s, size_t i) {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) ++i;
        return i;
    };
    // ---- name ----
    const size_t kn = inner.find("\"name\"");
    if (kn == std::string::npos) return false;
    size_t i = inner.find(':', kn);
    if (i == std::string::npos) return false;
    i = skip_ws(inner, i + 1);
    if (i >= inner.size()) return false;
    if (inner[i] == '"') {                       // properly quoted name
        std::string nm; size_t j = i + 1;
        for (; j < inner.size() && inner[j] != '"'; ++j) {
            if (inner[j] == '\\' && j + 1 < inner.size()) { nm += inner[j + 1]; ++j; }
            else nm += inner[j];
        }
        name_out = nm;
    } else {                                     // bare identifier (the template-literal bug)
        size_t j = i;
        while (j < inner.size() &&
               (std::isalnum(static_cast<unsigned char>(inner[j])) ||
                inner[j] == '_' || inner[j] == '-' || inner[j] == '.')) ++j;
        if (j == i) return false;
        name_out = inner.substr(i, j - i);
    }
    if (name_out.empty()) return false;
    // ---- arguments: scan the brace-balanced object, tolerant of a missing
    // outer '}'. String-aware so braces inside quoted values don't miscount. ----
    args_out = "{}";
    const size_t ka = inner.find("\"arguments\"");
    if (ka != std::string::npos) {
        size_t a = inner.find(':', ka);
        if (a != std::string::npos) {
            a = skip_ws(inner, a + 1);
            if (a < inner.size() && inner[a] == '{') {
                int depth = 0; bool instr = false, esc = false; size_t b = a;
                for (; b < inner.size(); ++b) {
                    const char c = inner[b];
                    if (instr) {
                        if (esc) esc = false;
                        else if (c == '\\') esc = true;
                        else if (c == '"') instr = false;
                    } else if (c == '"') instr = true;
                    else if (c == '{') ++depth;
                    else if (c == '}') { if (--depth == 0) { ++b; break; } }
                }
                if (depth == 0) {
                    const std::string cand = inner.substr(a, b - a);
                    json av = json::parse(cand, nullptr, /*allow_exceptions=*/false);
                    if (av.is_object()) args_out = av.dump();
                }
            }
        }
    }
    return true;
}

// Parse the model's generated <tool_call>{"name":..,"arguments":..}</tool_call>
// text blocks (the Qwen form render_tool_call produces) back into OpenAI
// structured tool_calls. `content_out` receives the text OUTSIDE the blocks.
// Unparseable/unterminated blocks are left as content (never silently dropped),
// so a truncated tool call degrades to plain text rather than a phantom call.
// Each call carries "index" for streaming deltas; the non-stream path strips it.
static json parse_response_tool_calls(const std::string& text,
                                      std::string& content_out) {
    json calls = json::array();
    content_out.clear();
    static const std::string OPEN = "<tool_call>", CLOSE = "</tool_call>";
    size_t pos = 0;
    int idx = 0;
    while (true) {
        size_t s = text.find(OPEN, pos);
        if (s == std::string::npos) { content_out += text.substr(pos); break; }
        content_out += text.substr(pos, s - pos);
        const size_t inner_start = s + OPEN.size();
        size_t e = text.find(CLOSE, inner_start);
        if (e == std::string::npos) { content_out += text.substr(s); break; }
        const std::string inner = text.substr(inner_start, e - inner_start);
        json parsed = json::parse(inner, nullptr, /*allow_exceptions=*/false);
        if (parsed.is_object() && parsed.contains("name") && parsed["name"].is_string()) {
            json fn;
            fn["name"] = parsed["name"];
            if (parsed.contains("arguments") && !parsed["arguments"].is_null()) {
                const auto& a = parsed["arguments"];
                fn["arguments"] = a.is_string() ? a.get<std::string>() : a.dump();
            } else {
                fn["arguments"] = "{}";
            }
            calls.push_back({{"id", "call_" + std::to_string(idx)},
                             {"index", idx},
                             {"type", "function"},
                             {"function", fn}});
            ++idx;
        } else {
            // Strict parse failed — try the lenient repair (unquoted name / missing
            // outer brace) before giving up and leaving the block as raw content.
            std::string lname, largs;
            if (lenient_tool_call(inner, lname, largs)) {
                json fn;
                fn["name"] = lname;
                fn["arguments"] = largs;
                calls.push_back({{"id", "call_" + std::to_string(idx)},
                                 {"index", idx},
                                 {"type", "function"},
                                 {"function", fn}});
                ++idx;
            } else {
                content_out += text.substr(s, (e + CLOSE.size()) - s);  // keep raw
            }
        }
        pos = e + CLOSE.size();
    }
    return calls;
}

std::string chat_completion_json(const std::string& model,
                                 const GenerateResult& r,
                                 const std::string& id, int64_t created) {
    json j = base(model, id, created, "chat.completion");
    std::string content;
    json calls = parse_response_tool_calls(r.text, content);
    json msg = {{"role", "assistant"}};
    std::string finish = r.finish_reason;
    if (!calls.empty()) {
        json out = json::array();
        for (auto tc : calls) { tc.erase("index"); out.push_back(tc); }  // non-stream omits index
        msg["tool_calls"] = out;
        // OpenAI allows null content alongside tool_calls; keep any real prose.
        msg["content"] = (content.find_first_not_of(" \t\r\n") == std::string::npos)
                         ? json(nullptr) : json(content);
        finish = "tool_calls";
    } else {
        msg["content"] = r.text;
    }
    j["choices"] = json::array({{
        {"index", 0}, {"message", msg}, {"finish_reason", finish}}});
    j["usage"] = {{"prompt_tokens", r.prompt_tokens},
                  {"completion_tokens", r.completion_tokens},
                  {"total_tokens", r.prompt_tokens + r.completion_tokens}};
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

// Streaming tool_calls delta frame. Returns "" when `full_text` has no valid
// tool call (caller then streams the buffered text as ordinary content).
std::string chat_chunk_sse_tool_calls(const std::string& model, const std::string& id,
                                      int64_t created, const std::string& full_text) {
    std::string content;
    json calls = parse_response_tool_calls(full_text, content);
    if (calls.empty()) return {};
    json j = base(model, id, created, "chat.completion.chunk");
    j["choices"] = json::array({{
        {"index", 0}, {"delta", {{"role", "assistant"}, {"tool_calls", calls}}},
        {"finish_reason", json()}}});
    return "data: " + j.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
}

std::string chat_chunk_sse(const std::string& model, const std::string& id,
                           int64_t created, std::string_view delta,
                           const std::string& finish_reason) {
    json j = base(model, id, created, "chat.completion.chunk");
    json d = json::object();
    if (!delta.empty()) d["content"] = std::string(delta);
    j["choices"] = json::array({{
        {"index", 0}, {"delta", d},
        {"finish_reason", finish_reason.empty() ? json() : json(finish_reason)}}});
    return "data: " + j.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
}

std::string chat_chunk_sse_usage(const std::string& model, const std::string& id,
                                 int64_t created, uint32_t prompt_tokens,
                                 uint32_t completion_tokens) {
    json j = base(model, id, created, "chat.completion.chunk");
    j["choices"] = json::array();   // OpenAI's usage chunk carries no choices
    j["usage"] = {{"prompt_tokens", prompt_tokens},
                  {"completion_tokens", completion_tokens},
                  {"total_tokens", prompt_tokens + completion_tokens}};
    return "data: " + j.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
}

std::string models_json(const std::string& model_id) {
    return json{{"object", "list"},
                {"data", json::array({{{"id", model_id}, {"object", "model"},
                                       {"owned_by", "local"}}})}}.dump();
}

}  // namespace ie::oai
