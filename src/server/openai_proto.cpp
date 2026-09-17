#include "ie/openai_proto.hpp"
#include "ie/reasoning.hpp"
#include "nlohmann/json.hpp"

#include <cctype>
#include <cstdlib>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>

using nlohmann::json;
using ojson = nlohmann::ordered_json;   // preserves client key order (tools)

namespace ie::oai {

// Decode a base64 payload (standard alphabet, '=' padding, whitespace
// tolerated). Returns false on any other character.
static bool b64_decode(std::string_view in, std::string& out) {
    auto val = [](unsigned char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    out.clear();
    out.reserve(in.size() / 4 * 3);
    uint32_t acc = 0;
    int bits = 0;
    for (unsigned char c : in) {
        if (std::isspace(c) || c == '=') continue;
        const int v = val(c);
        if (v < 0) return false;
        acc = (acc << 6) | uint32_t(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(char((acc >> bits) & 0xff));
        }
    }
    return true;
}

// OpenAI image_url content part -> raw image bytes. Only data: URIs are
// accepted (the server never fetches remote URLs). "" on success.
static std::string parse_image_part(const json& part, std::string& bytes) {
    if (!part.contains("image_url")) return "image_url part missing image_url";
    const auto& iu = part["image_url"];
    std::string url;
    if (iu.is_string()) url = iu.get<std::string>();
    else if (iu.is_object() && iu.contains("url") && iu["url"].is_string())
        url = iu["url"].get<std::string>();
    else return "image_url needs a url string";
    const std::string_view pfx = "data:image/";
    if (url.rfind(pfx, 0) != 0)
        return "only data:image/...;base64 URLs are supported";
    const size_t comma = url.find(',');
    if (comma == std::string::npos ||
        url.find(";base64", 0) == std::string::npos || url.find(";base64") > comma)
        return "image data URI must be base64";
    if (!b64_decode(std::string_view(url).substr(comma + 1), bytes))
        return "invalid base64 in image data URI";
    if (bytes.empty()) return "empty image payload";
    return {};
}

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

namespace {
std::optional<ChatRequest> configured_defaults;

void number(const ojson& j, const char* key, float& target, double lo,
            double hi, bool exclusive_lo = false) {
    if (!j.contains(key)) return;
    if (!j[key].is_number()) throw std::runtime_error(std::string(key) + " must be numeric");
    const double v = j[key].get<double>();
    if (!std::isfinite(v) || v < lo || v > hi || (exclusive_lo && v == lo))
        throw std::runtime_error(std::string(key) + " is outside the supported range");
    const float narrowed = float(v);
    if (!std::isfinite(narrowed) || (v != 0 && narrowed == 0))
        throw std::runtime_error(std::string(key) + " is not representable as a sampling float");
    target = narrowed;
}
uint64_t integer(const ojson& j, const char* key, uint64_t fallback, uint64_t hi) {
    if (!j.contains(key)) return fallback;
    const auto& v = j[key];
    if (!v.is_number_integer() || (!v.is_number_unsigned() && v.get<int64_t>() < 0))
        throw std::runtime_error(std::string(key) + " must be a nonnegative integer");
    const auto n = v.get<uint64_t>();
    if (n > hi) throw std::runtime_error(std::string(key) + " exceeds the supported maximum");
    return n;
}
} // namespace

static ChatRequest parse_chat_request_impl(const std::string& body, ChatRequest out) {
    ojson j = ojson::parse(body, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_object()) { out.error = "invalid JSON"; return out; }
    out.model = j.value("model", "default");
    out.stream = j.value("stream", false);
    out.enable_thinking = j.value("enable_thinking", out.enable_thinking);
    if(j.contains("reasoning_effort") && !j["reasoning_effort"].is_null()) {
        if(!j["reasoning_effort"].is_string())throw std::runtime_error("reasoning_effort must be a string");
        out.reasoning_effort=j["reasoning_effort"].get<std::string>();
        if(!known_reasoning_effort(out.reasoning_effort))throw std::runtime_error("unknown reasoning_effort level");
    }
    auto& sp = out.sampling;
    number(j, "temperature", sp.temperature, 0, 2);
    number(j, "top_p", sp.top_p, 0, 1, true);
    sp.top_k = uint32_t(integer(j, "top_k", sp.top_k, 1024));
    number(j, "min_p", sp.min_p, 0, 1);
    number(j, "repetition_penalty", sp.repeat_penalty, 0, 10, true);
    number(j, "repeat_penalty", sp.repeat_penalty, 0, 10, true);
    number(j, "presence_penalty", sp.presence_penalty, -2, 2);
    number(j, "frequency_penalty", sp.frequency_penalty, -2, 2);
    sp.repeat_window = uint32_t(integer(j, "repeat_window", sp.repeat_window, 512));
    sp.repeat_window = uint32_t(integer(j, "repeat_last_n", sp.repeat_window, 512));
    if (j.contains("max_tokens") && !j["max_tokens"].is_null())
        sp.max_tokens = uint32_t(integer(j, "max_tokens", sp.max_tokens, UINT32_MAX));
    if (j.contains("seed") && !j["seed"].is_null())
        sp.seed = integer(j, "seed", sp.seed, UINT64_MAX);
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
    // `stop`: a string or an array of up to 4 strings (OpenAI); empty strings
    // are ignored, any other shape is a 400.
    if (j.contains("stop")) {
        out.stop.clear(); // explicit null/[] disables the configured stops
        const auto& st = j["stop"];
        if (st.is_null()) {
            // explicit override to no stops
        } else if (st.is_string()) {
            if (!st.get_ref<const std::string&>().empty()) out.stop.push_back(st.get<std::string>());
        } else if (st.is_array() && st.size() <= 4) {
            for (const auto& x : st) {
                if (!x.is_string()) { out.error = "stop must be a string or an array of strings"; return out; }
                if (!x.get_ref<const std::string&>().empty()) out.stop.push_back(x.get<std::string>());
            }
        } else {
            out.error = "stop must be a string or an array of up to 4 strings"; return out;
        }
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
        std::vector<std::string> images;
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
                // OpenAI multi-part content: concatenate text parts in order;
                // image_url parts (base64 data URIs) attach to the turn.
                for (const auto& part : cv) {
                    if (!part.is_object()) continue;
                    const std::string pt = part.value("type", "");
                    if (pt == "text" &&
                        part.contains("text") && part["text"].is_string()) {
                        content += part["text"].get<std::string>();
                    } else if (pt == "image_url") {
                        std::string bytes;
                        if (auto e = parse_image_part(part, bytes); !e.empty()) {
                            out.error = e;
                            return out;
                        }
                        images.push_back(std::move(bytes));
                        content += kChatImageMarker;   // position of this image in the text
                    }
                }
            } else {
                out.error = "each message needs string role and string/array content";
                return out;
            }
        }
        const std::string original_content = content;
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
        ChatTurn turn{role, std::move(content), std::move(images)};
        if (has_tool_calls) {
            turn.tool_calls_json = m["tool_calls"].dump();
            turn.content_without_tool_calls = original_content;
        }
        if (role == "assistant" && m.contains("reasoning_content") && m["reasoning_content"].is_string())
            turn.reasoning_content = m["reasoning_content"].get<std::string>();
        if (has_tool_calls) for (const auto& tc : m["tool_calls"])
            turn.tool_call_ids.push_back(tc.contains("id") && tc["id"].is_string()
                ? tc["id"].get<std::string>() : std::string{});
        if (role == "tool" && m.contains("tool_call_id") && m["tool_call_id"].is_string())
            turn.tool_call_id = m["tool_call_id"].get<std::string>();
        out.turns.push_back(std::move(turn));
    }
    return out;
}

ChatRequest server_defaults_from_environment() {
    ChatRequest defaults;
    defaults.enable_thinking = std::getenv("IE_SERVE_NO_THINK") == nullptr;
    defaults.sampling.max_tokens = 16384;
    ojson j = {{"messages", {{{"role", "user"}, {"content", "defaults"}}}}};
    if(const char* value=std::getenv("IE_SERVE_REASONING_EFFORT"))j["reasoning_effort"]=value;
    for (auto [env, key] : {std::pair{"IE_SERVE_TEMP", "temperature"},
            {"IE_SERVE_TOP_K", "top_k"}, {"IE_SERVE_TOP_P", "top_p"},
            {"IE_SERVE_MIN_P", "min_p"}, {"IE_SERVE_REPEAT_PENALTY", "repeat_penalty"},
            {"IE_SERVE_REPEAT_LAST_N", "repeat_last_n"},
            {"IE_SERVE_PRESENCE_PENALTY", "presence_penalty"},
            {"IE_SERVE_FREQUENCY_PENALTY", "frequency_penalty"},
            {"IE_SERVE_SEED", "seed"}, {"IE_SERVE_MAX_TOKENS", "max_tokens"}}) {
        if (const char* value = std::getenv(env)) {
            auto parsed = ojson::parse(value, nullptr, false);
            if (parsed.is_discarded() || !parsed.is_number())
                throw std::runtime_error(std::string(env) + " must be numeric");
            j[key] = parsed;
        }
    }
    auto out = parse_chat_request_impl(j.dump(), defaults);
    if (!out.error.empty()) throw std::runtime_error(out.error);
    if (out.sampling.max_tokens == 0) out.sampling.max_tokens = kMaxTokensUnlimited;
    out.turns.clear(); out.model.clear();
    return out;
}

void configure_server_defaults(const ChatRequest& defaults) {
    configured_defaults = defaults; // called once before HTTP worker threads start
}

ChatRequest parse_chat_request(const std::string& body) {
    try {
        return parse_chat_request_impl(body, configured_defaults
            ? *configured_defaults : server_defaults_from_environment());
    } catch (const std::exception& e) {
        ChatRequest out; out.error = std::string("invalid request setting: ") + e.what();
        return out;
    }
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
                              std::string& args_out,
                              bool* args_complete = nullptr) {
    if (args_complete) *args_complete = false;
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
                    if (av.is_object()) {
                        args_out = av.dump();
                        if (args_complete) *args_complete = true;
                    }
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
        if (e == std::string::npos) {
            // UNTERMINATED block at end of generation. Models drop the closing
            // tag (or write a bare <tool_call> as the "close" — seen live
            // 2026-08-26: Hermes delegate_task leaked as text over exactly
            // this). EOS-close repair: accept the block ONLY when the lenient
            // parse recovers a name AND a COMPLETE brace-balanced arguments
            // object — a genuinely truncated call fails that test and stays
            // text (the original safety contract).
            std::string inner_tail = text.substr(inner_start);
            // A trailing bare re-open is the model's malformed close — strip it.
            const size_t reopen = inner_tail.rfind(OPEN);
            if (reopen != std::string::npos &&
                inner_tail.find_first_not_of(" \t\r\n", reopen + OPEN.size()) ==
                    std::string::npos)
                inner_tail.erase(reopen);
            std::string lname, largs;
            bool complete = false;
            if (lenient_tool_call(inner_tail, lname, largs, &complete) && complete) {
                json fn;
                fn["name"] = lname;
                fn["arguments"] = largs;
                calls.push_back({{"id", "call_" + std::to_string(idx)},
                                 {"index", idx},
                                 {"type", "function"},
                                 {"function", fn}});
                ++idx;
            } else {
                content_out += text.substr(s);
            }
            break;
        }
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
    json calls = json::array();
    // deepseek4 hands its tool calls already parsed (Phase L); every other
    // arch is parsed from the canonical <tool_call> text as before.
    if (!r.tool_calls_json.empty()) {
        json pre = json::parse(r.tool_calls_json, nullptr, /*allow_exceptions=*/false);
        if (pre.is_array() && !pre.empty()) {
            int idx = 0;
            for (auto& tc : pre) {
                if (!tc.is_object()) { pre = json::array(); break; }
                if (!tc.contains("id")) tc["id"] = "call_" + std::to_string(idx);
                ++idx;
            }
            if (!pre.empty()) { calls = pre; content = r.text; }
        }
    }
    // A native parser's empty array is authoritative; prose must not become a call.
    if (calls.empty() && r.tool_calls_json.empty()) calls = parse_response_tool_calls(r.text, content);
    json msg = {{"role", "assistant"}};
    if (!r.reasoning_content.empty()) msg["reasoning_content"] = r.reasoning_content;
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
                  {"total_tokens", r.prompt_tokens + r.completion_tokens},
                  {"prompt_tokens_details", {{"cached_tokens", r.cached_tokens}}}};
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

std::string chat_chunk_sse_reasoning(const std::string& model, const std::string& id,
                                     int64_t created, std::string_view delta) {
    json j = base(model, id, created, "chat.completion.chunk");
    json d = json::object();
    d["reasoning_content"] = std::string(delta);
    j["choices"] = json::array({{{"index", 0}, {"delta", d}, {"finish_reason", json()}}});
    return "data: " + j.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
}

std::string chat_chunk_sse_tool_calls_json(const std::string& model, const std::string& id,
                                           int64_t created, const std::string& tool_calls_json) {
    if (tool_calls_json.empty()) return {};
    json calls = json::parse(tool_calls_json, nullptr, /*allow_exceptions=*/false);
    if (!calls.is_array() || calls.empty()) return {};
    int idx = 0;
    for (auto& tc : calls) {
        if (!tc.is_object()) return {};
        tc["index"] = idx++;
        if (!tc.contains("id")) tc["id"] = "call_" + id.substr(0, 8) + "_" + std::to_string(idx);
    }
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
                                 uint32_t completion_tokens, uint32_t cached_tokens) {
    json j = base(model, id, created, "chat.completion.chunk");
    j["choices"] = json::array();   // OpenAI's usage chunk carries no choices
    j["usage"] = {{"prompt_tokens", prompt_tokens},
                  {"completion_tokens", completion_tokens},
                  {"total_tokens", prompt_tokens + completion_tokens},
                  {"prompt_tokens_details", {{"cached_tokens", cached_tokens}}}};
    return "data: " + j.dump(-1, ' ', false, json::error_handler_t::replace) + "\n\n";
}

std::string models_json(const std::string& model_id) {
    return json{{"object", "list"},
                {"data", json::array({{{"id", model_id}, {"object", "model"},
                                       {"owned_by", "local"}}})}}.dump();
}

std::string error_json(std::string_view message, std::string_view type,
                       std::string_view code) {
    json err{{"message", std::string(message)}, {"type", std::string(type)}};
    if (!code.empty()) err["code"] = std::string(code);
    // replace: an exception text with a stray non-UTF-8 byte must still yield
    // a parseable body instead of throwing type_error.316 out of the handler.
    return json{{"error", err}}.dump(-1, ' ', false, json::error_handler_t::replace);
}

}  // namespace ie::oai
