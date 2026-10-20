// include/ie/tool_call_repair.hpp — conservative, intent-preserving repairs for model-written tool calls
// (docs/deepseek41/107). Header-only, no dependencies, so the DeepSeek V4 and V4.1 parsers (and their unit tests) share them.
//
// The contract all of this keeps: a repair may only recover what the model unambiguously MEANT, never guess. A bare word where JSON
// was promised (`string="false"` + `README.md`), an unterminated string, a name that still holds markup after trimming: those stay
// errors, because the harness's rule is that a malformed call executes nothing.
#pragma once

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

namespace ie {

// IE_TOOLCALL_REPAIR=0 restores strict, recovery-free parsing without a rebuild (the markup-in-name refusal stays on: it is a safety
// check, not a repair). Read on every call: it costs nothing next to a parse and lets tests flip it.
inline bool toolcall_repair_enabled() {
    const char* e = std::getenv("IE_TOOLCALL_REPAIR");
    return !(e && e[0] == '0' && e[1] == '\0');
}

// A tool name the model wrote with a stray tail (`run_bash>`): the identifier before it. False when what is left still holds markup
// (`"`, `<`, `>`, a newline) or is empty -- that is not a name, and the caller must report a parse error, never a call.
// `changed` (optional) says the name was trimmed.
inline bool clean_tool_name(std::string_view raw, std::string& out, bool* changed = nullptr, bool allow_trim = true) {
    auto bad = [](char c) { return c == '"' || c == '<' || c == '>' || c == '\r' || c == '\n'; };
    if (!allow_trim) {                                     // recovery off: the name is taken exactly as written, or refused
        if (changed) *changed = false;
        for (char c : raw) if (bad(c)) return false;
        out.assign(raw);
        return !raw.empty();
    }
    size_t b = 0, e = raw.size();
    while (b < e && std::isspace(static_cast<unsigned char>(raw[b]))) ++b;
    while (e > b && (std::isspace(static_cast<unsigned char>(raw[e - 1])) || raw[e - 1] == '>' || raw[e - 1] == '"')) --e;
    if (changed) *changed = (b != 0 || e != raw.size());
    if (e == b) return false;
    for (size_t i = b; i < e; ++i) if (bad(raw[i])) return false;
    out.assign(raw.substr(b, e - b));
    return true;
}

// JSON a model wrote that strict parsing rejected, fixed ONLY for the slips that have exactly one reading:
//   * a raw control character inside a string (a literal newline/tab in multi-line code) -> its escape;
//   * a backslash that starts no valid escape (`\d`, `\.`, `C:\x`) -> a literal backslash;
//   * a comma before `}` / `]`;
//   * the Python spellings True / False / None.
// Whatever else is wrong (unbalanced quotes, bare words, single-quoted strings) is left as it was, so the caller's strict parse of the
// result rejects it. Returns the candidate text; the CALLER must parse it -- this function only writes it.
inline std::string json_syntax_repair(std::string_view in) {
    std::string out;
    out.reserve(in.size() + 16);
    bool in_str = false;
    for (size_t i = 0; i < in.size(); ++i) {
        const unsigned char c = static_cast<unsigned char>(in[i]);
        if (in_str) {
            if (c == '"') { in_str = false; out += '"'; }
            else if (c == '\\') {
                if (i + 1 >= in.size()) { out += "\\\\"; continue; }
                const char n = in[i + 1];
                if (n == '"' || n == '\\' || n == '/' || n == 'b' || n == 'f' || n == 'n' || n == 'r' || n == 't') { out += '\\'; out += n; ++i; }
                else if (n == 'u') {
                    bool hex = (i + 6 <= in.size());       // `\u` + four hex digits fit in the input
                    for (size_t k = 2; hex && k < 6; ++k) hex = std::isxdigit(static_cast<unsigned char>(in[i + k])) != 0;
                    if (hex) { out.append(in.substr(i, 6)); i += 5; }
                    else out += "\\\\";                     // `\u` without four hex digits: a literal backslash; the `u` follows as itself
                } else out += "\\\\";                        // a lone backslash; the next character is handled on its own turn
            }
            else if (c < 0x20) {
                if (c == '\n') out += "\\n";
                else if (c == '\r') out += "\\r";
                else if (c == '\t') out += "\\t";
                else { char buf[8]; std::snprintf(buf, sizeof buf, "\\u%04x", c); out += buf; }
            }
            else out += static_cast<char>(c);
        } else {
            if (c == '"') { in_str = true; out += '"'; }
            else if (c == '}' || c == ']') {
                size_t k = out.size();
                while (k > 0 && std::isspace(static_cast<unsigned char>(out[k - 1]))) --k;
                if (k > 0 && out[k - 1] == ',') out.erase(k - 1, 1);   // a trailing comma
                out += static_cast<char>(c);
            }
            else if (std::isalpha(c) || c == '_') {
                size_t j = i;
                while (j < in.size() && (std::isalnum(static_cast<unsigned char>(in[j])) || in[j] == '_')) ++j;
                const std::string_view w = in.substr(i, j - i);
                if (w == "True") out += "true";
                else if (w == "False") out += "false";
                else if (w == "None") out += "null";
                else out.append(w);
                i = j - 1;
            }
            else out += static_cast<char>(c);
        }
    }
    return out;
}

// What the engine tells whoever wants to know about a tool call that needed repair or was refused. The parsers stay pure: they only
// call the observer when one is installed (the server installs a logger; unit tests install nothing).
struct ToolCallEvent {
    const char* family = "";    // "deepseek4" | "deepseek41"
    const char* kind   = "";    // "repaired" | "rejected"
    std::string detail;         // what was fixed, or the parse error
    std::string raw;            // the model's completion text (callers cut it to a sane length)
};
using ToolCallObserver = void (*)(const ToolCallEvent&);
inline ToolCallObserver& tool_call_observer() { static ToolCallObserver o = nullptr; return o; }
inline void tool_call_notify(const char* family, const char* kind, std::string detail, std::string_view raw) {
    if (auto o = tool_call_observer()) {
        ToolCallEvent ev; ev.family = family; ev.kind = kind; ev.detail = std::move(detail);
        ev.raw.assign(raw.substr(0, 6000));
        o(ev);
    }
}

}  // namespace ie
