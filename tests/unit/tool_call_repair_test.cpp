// tests/unit/tool_call_repair_test.cpp — host-only (no GPU, no model): the tool-call repair helpers (docs/deepseek41/107).
// The contract under test: a repair recovers only what the model unambiguously meant, and everything else stays an error.
#include "ie/tool_call_repair.hpp"
#include "ie/toolcall_log.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../../third_party/nlohmann/json.hpp"

using Json = nlohmann::json;
static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++failures; std::printf("  FAIL line %d: ", __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

// repair, then parse strictly; returns the parsed value or a discarded Json
static Json repaired(const std::string& in) { return Json::parse(ie::json_syntax_repair(in), nullptr, false); }

int main() {
    // ---- clean_tool_name -------------------------------------------------------------------------------------------------------
    {
        struct C { const char* in; bool ok; const char* want; bool changed; };
        const std::vector<C> cs = {
            { "run_bash", true, "run_bash", false },
            { "run_bash>", true, "run_bash", true },
            { "run_bash\">", true, "run_bash", true },
            { "  read_file  ", true, "read_file", true },
            { "mcp::search-docs.v2", true, "mcp::search-docs.v2", false },
            { "a\"><b>", false, "", true },
            { "grep\">\n<\xef\xbd\x9c" "DSML\xef\xbd\x9c parameter name=\"arguments\"", false, "", true },
            { "two words", true, "two words", false },     // a space is not markup; whether it is a declared tool is the caller's question
            { "", false, "", false },
            { ">", false, "", true },
            { "\"\"", false, "", true },
        };
        for (const auto& c : cs) {
            std::string out; bool ch = false;
            const bool ok = ie::clean_tool_name(c.in, out, &ch);
            CHECK(ok == c.ok, "clean_tool_name(%s): ok=%d want %d", c.in, ok, c.ok);
            if (ok && c.ok) CHECK(out == c.want, "clean_tool_name(%s) = [%s] want [%s]", c.in, out.c_str(), c.want);
            CHECK(ch == c.changed, "clean_tool_name(%s): changed=%d want %d", c.in, ch, c.changed);
        }
    }

    // ---- json_syntax_repair: the recoverable slips -----------------------------------------------------------------------------
    {
        Json j = repaired("{\"code\": \"line1\nline2\tx\r\n\"}");                 // raw newline / tab / CR inside a string
        CHECK(!j.is_discarded() && j["code"] == "line1\nline2\tx\r\n", "raw control chars");
        j = repaired(std::string("{\"a\": \"x") + char(0x01) + "y\"}");           // another control char -> \u0001
        CHECK(!j.is_discarded() && j["a"] == std::string("x\x01y"), "control char 0x01");
        j = repaired("{\"re\": \"\\d+\\.\\w\"}");                                  // \d \. \w are not JSON escapes: literal backslashes
        CHECK(!j.is_discarded() && j["re"] == "\\d+\\.\\w", "invalid escapes -> literal backslash");
        j = repaired("{\"p\": \"C:\\Users\\x\\users\"}");                          // \U \x invalid; \u + non-hex stays literal
        CHECK(!j.is_discarded() && j["p"] == "C:\\Users\\x\\users", "windows path");
        j = repaired("{\"e\": \"caf\\u00e9\"}");                                   // a real \uXXXX is untouched
        CHECK(!j.is_discarded() && j["e"] == "caf\xc3\xa9", "valid \\u escape kept");
        j = repaired("{\"a\": 1, \"b\": [1, 2, ], }");                              // trailing commas
        CHECK(!j.is_discarded() && j["a"] == 1 && j["b"].size() == 2, "trailing commas");
        j = repaired("{\"a\": True, \"b\": None, \"c\": False}");                  // Python literals
        CHECK(!j.is_discarded() && j["a"] == true && j["b"].is_null() && j["c"] == false, "python literals");
        j = repaired("{\"s\": \"True story, None of False\"}");                    // ...but never inside a string
        CHECK(!j.is_discarded() && j["s"] == "True story, None of False", "literals inside strings untouched");
        j = repaired("{\"path\": \"a\\\\b\"}");                                    // an already-correct escaped backslash is not doubled
        CHECK(!j.is_discarded() && j["path"] == "a\\b", "escaped backslash kept");
        j = repaired("{\"a\": \"tail\\");                                          // lone trailing backslash then EOF: still unterminated
        CHECK(j.is_discarded(), "unterminated stays invalid");
    }

    // ---- valid JSON passes through semantically unchanged (the repair is only ever applied after a strict failure, but must not
    // damage a valid document if it ever sees one) ---------------------------------------------------------------------------------
    {
        const std::vector<std::string> valid = {
            "{}", "[]", "{\"a\": 1}", "{\"a\": [1, 2.5, -3e2, true, false, null], \"b\": {\"c\": \"d\"}}",
            "{\"q\": \"say \\\"hi\\\" \\\\ / \\/ \\b\\f\\n\\r\\t\"}", "{\"u\": \"\\u00e9\\ud83d\\ude00\"}",
            "{\"cmd\": \"cd '/x y' && echo \\\"$HOME\\\"\\n\"}", "\"just a string\"", "42", "true",
        };
        for (const auto& v : valid) {
            const Json a = Json::parse(v, nullptr, false), b = repaired(v);
            CHECK(!a.is_discarded() && !b.is_discarded() && a == b, "valid doc changed by repair: %s", v.c_str());
        }
    }

    // ---- what must STAY an error (no guessing) ---------------------------------------------------------------------------------
    {
        const std::vector<std::string> unfixable = {
            "README.md",                       // a bare word where JSON was promised (the harness contract: never execute a guess)
            "{\"a\": bare}", "{\"a\": 'single'}", "{\"a\": 1", "{\"a\": \"unterminated}", "{\"a\" 1}", "[1, 2",
            "{\"a\": 1}}", "", "{\"a\": \"x\" \"b\": 2}",
        };
        for (const auto& u : unfixable) CHECK(repaired(u).is_discarded(), "should stay invalid: [%s]", u.c_str());
    }

    // ---- the observer: silent unless installed ---------------------------------------------------------------------------------
    {
        static int seen = 0; static std::string kind, raw;
        ie::tool_call_notify("deepseek4", "repaired", "x", "raw");                  // no observer: a no-op
        CHECK(seen == 0, "observer called with none installed");
        ie::tool_call_observer() = [](const ie::ToolCallEvent& e) { ++seen; kind = e.kind; raw = e.raw; };
        ie::tool_call_notify("deepseek4", "rejected", "why", std::string(9000, 'x'));
        CHECK(seen == 1 && kind == "rejected" && raw.size() == 6000, "observer got seen=%d kind=%s raw=%zu", seen, kind.c_str(), raw.size());
        ie::tool_call_observer() = nullptr;
    }

    // ---- the server's logger: one JSON line per repaired/refused call, trimmed to the call, rotated, switchable off -------------------
    {
        namespace fs = std::filesystem;
        const std::string f = (fs::temp_directory_path() / "ie_toolcall_log_test.jsonl").string();
        auto slurp = [](const std::string& p) { std::ifstream in(p); std::stringstream ss; ss << in.rdbuf(); return ss.str(); };
        fs::remove(f); fs::remove(f + ".1");
        setenv("IE_TOOLCALL_LOG", f.c_str(), 1); setenv("IE_TOOLCALL_LOG_MAX_BYTES", "700", 1);
        ie::install_toolcall_log();
        const std::string reasoning(5000, 'r');                                    // long reasoning before the call: must NOT be written out
        const std::string call = "<\xef\xbd\x9c" "DSML\xef\xbd\x9c invoke name=\"x\">\n<param>";
        ie::tool_call_notify("deepseek41", "rejected", "invalid JSON parameter", reasoning + call);
        std::string body = slurp(f);
        CHECK(!body.empty() && body.back() == '\n', "log line written");
        Json line = Json::parse(body, nullptr, false);
        CHECK(!line.is_discarded() && line["family"] == "deepseek41" && line["kind"] == "rejected" && line["detail"] == "invalid JSON parameter", "log fields");
        const std::string logged_raw = line.is_discarded() ? "" : line["raw"].get<std::string>();
        CHECK(logged_raw.find("DSML") != std::string::npos && logged_raw.size() < 160 + call.size() + 8, "raw trimmed to the call, %zu bytes", logged_raw.size());
        // invalid UTF-8 in the raw text must not throw or corrupt the file
        ie::tool_call_notify("deepseek4", "repaired", "bad utf8", std::string("\xff\xfe\x80 broken ") + call);
        // enough lines to pass the 700-byte cap -> rotation to .1
        for (int i = 0; i < 6; ++i) ie::tool_call_notify("deepseek41", "repaired", "filler filler filler filler filler filler", call);
        CHECK(fs::exists(f + ".1"), "rotated at the size cap");
        std::stringstream lines(slurp(f)); std::string l; bool all_json = true;
        while (std::getline(lines, l)) all_json = all_json && !Json::parse(l, nullptr, false).is_discarded();
        CHECK(all_json, "every line of the live log is valid JSON");
        // off switch
        fs::remove(f); fs::remove(f + ".1"); setenv("IE_TOOLCALL_LOG", "0", 1);
        ie::tool_call_notify("deepseek41", "rejected", "x", call);
        CHECK(!fs::exists(f), "IE_TOOLCALL_LOG=0 writes nothing");
        unsetenv("IE_TOOLCALL_LOG"); unsetenv("IE_TOOLCALL_LOG_MAX_BYTES"); ie::tool_call_observer() = nullptr; fs::remove(f + ".1");
    }

    if (failures) { std::printf("tool_call_repair_test: FAIL (%d checks failed)\n", failures); return 1; }
    std::printf("tool_call_repair_test: PASS\n");
    return 0;
}
