// tests/unit/ds41_tool_parse_test.cpp — host-only (no GPU, no model): ds41_parse_completion's tool-call handling (docs/deepseek41/107).
// Valid calls parse exactly as before; the slips that have ONE reading are recovered; everything else stays an error (a malformed
// call executes nothing -- Dream's DREAM-079 acceptance rule, which the bare-word case below pins).
#include "ie/deepseek41_prompt.hpp"
#include "ie/tool_call_repair.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../../third_party/nlohmann/json.hpp"

using Json = nlohmann::json;
static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++failures; std::printf("  FAIL line %d: ", __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static const std::string D = "\xef\xbd\x9c" "DSML" "\xef\xbd\x9c";                           // ｜DSML｜
static const std::string EOS = "<\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c>";

static std::string open_calls() { return "Checking.\n\n<" + D + " calls>\n"; }
static std::string close_calls() { return "</" + D + " calls>" + EOS; }
static std::string invoke(const std::string& name, const std::string& params) {
    return "<" + D + " invoke name=\"" + name + "\">\n" + params + "</" + D + " invoke>\n";
}
static std::string param(const std::string& key, bool is_string, const std::string& value) {
    return "<" + D + " parameter name=\"" + key + "\" string=\"" + (is_string ? "true" : "false") + "\">" + value + "</" + D + " parameter>\n";
}

struct Parsed { bool ok; std::string err; Json calls; };
static Parsed parse(const std::string& completion) {
    const auto c = ie::ds41_parse_completion(completion, /*thinking=*/false);
    Parsed p; p.ok = c.error.empty(); p.err = c.error;
    p.calls = c.tool_calls_json.empty() ? Json::array() : Json::parse(c.tool_calls_json);
    return p;
}
static std::string args_of(const Parsed& p, size_t i = 0) { return p.calls[i]["function"]["arguments"].get<std::string>(); }
static std::string name_of(const Parsed& p, size_t i = 0) { return p.calls[i]["function"]["name"].get<std::string>(); }

int main() {
    // ---- valid calls: unchanged behaviour ----------------------------------------------------------------------------------------
    {
        auto p = parse(open_calls() + invoke("run_bash", param("command", true, "ls -la") + param("timeout", false, "30")) + close_calls());
        CHECK(p.ok && p.calls.size() == 1 && name_of(p) == "run_bash" && args_of(p) == "{\"command\": \"ls -la\", \"timeout\": 30}",
              "valid call: ok=%d err=%s args=%s", p.ok, p.err.c_str(), p.ok ? args_of(p).c_str() : "");
        p = parse(open_calls() + invoke("a", param("x", true, "1")) + invoke("b", "") + close_calls());   // two calls, the second without arguments
        CHECK(p.ok && p.calls.size() == 2 && name_of(p, 0) == "a" && name_of(p, 1) == "b" && args_of(p, 1) == "{}", "two calls");
        p = parse(open_calls() + invoke("w", param("data", false, "{\"k\": [1, 2, {\"z\": null}]}")) + close_calls());   // valid JSON goes through verbatim
        CHECK(p.ok && args_of(p) == "{\"data\": {\"k\": [1, 2, {\"z\": null}]}}", "valid JSON verbatim: %s", p.ok ? args_of(p).c_str() : p.err.c_str());
    }

    // ---- recovered: JSON slips with one reading ---------------------------------------------------------------------------------
    {
        // the model's multi-line code in a JSON blob, written with LITERAL newlines/tabs (the commonest slip in agent harnesses)
        auto p = parse(open_calls() + invoke("run_bash", param("arguments", false, "{\"command\": \"cd /x\nls\n\tpwd\"}")) + close_calls());
        CHECK(p.ok && args_of(p) == "{\"arguments\": {\"command\": \"cd /x\\nls\\n\\tpwd\"}}", "raw newlines: ok=%d %s", p.ok, p.ok ? args_of(p).c_str() : p.err.c_str());
        p = parse(open_calls() + invoke("grep", param("arguments", false, "{\"pattern\": \"\\d+\\.\\s\", \"path\": \"C:\\Users\\x\"}")) + close_calls());
        CHECK(p.ok && args_of(p) == "{\"arguments\": {\"pattern\": \"\\\\d+\\\\.\\\\s\", \"path\": \"C:\\\\Users\\\\x\"}}", "invalid escapes: ok=%d %s", p.ok, p.ok ? args_of(p).c_str() : p.err.c_str());
        p = parse(open_calls() + invoke("f", param("a", false, "[1, 2, ]") + param("b", false, "{\"x\": True, \"y\": None,}")) + close_calls());
        CHECK(p.ok && args_of(p) == "{\"a\": [1, 2], \"b\": {\"x\": true, \"y\": null}}", "trailing commas + python literals: ok=%d %s", p.ok, p.ok ? args_of(p).c_str() : p.err.c_str());
        p = parse(open_calls() + "<" + D + " invoke name=\"run_bash>\">\n" + param("command", true, "ls") + "</" + D + " invoke>\n" + close_calls());   // stray '>' on the name
        CHECK(p.ok && name_of(p) == "run_bash", "name with stray '>': ok=%d err=%s name=%s", p.ok, p.err.c_str(), p.ok ? name_of(p).c_str() : "");
    }

    // ---- must STAY errors (a malformed call executes nothing) -------------------------------------------------------------------
    {
        // Dream's DREAM-079 acceptance case: string="false" on a bare word is NOT turned into a string
        auto p = parse(open_calls() + invoke("read_file", param("path", false, "README.md")) + close_calls());
        CHECK(!p.ok && p.calls.empty() && p.err.find("invalid JSON parameter") != std::string::npos, "bare word must stay an error: ok=%d err=%s", p.ok, p.err.c_str());
        p = parse(open_calls() + invoke("w", param("a", false, "{\"x\": \"unterminated}")) + close_calls());
        CHECK(!p.ok && p.calls.empty(), "unterminated string must stay an error");
        p = parse(open_calls() + invoke("w", param("a", false, "{'single': 'quotes'}")) + close_calls());
        CHECK(!p.ok && p.calls.empty(), "single-quoted JSON must stay an error");
        p = parse(open_calls() + invoke("a<b", param("x", true, "1")) + close_calls());                           // markup in a name
        CHECK(!p.ok && p.calls.empty(), "markup in a tool name must be an error, not a call: ok=%d name=%s", p.ok, p.ok ? name_of(p).c_str() : "");
        p = parse(open_calls() + invoke("x", param("a", true, "1") + param("a", true, "2")) + close_calls());     // duplicate parameter
        CHECK(!p.ok, "duplicate parameter must stay an error");
        // a truncated reply (no EOS / calls block not closed) executes nothing
        auto c = ie::ds41_parse_completion(open_calls() + invoke("x", param("a", true, "1")), false);
        CHECK(!c.error.empty() && c.tool_calls_json.empty(), "truncated completion must stay an error");
    }

    // ---- IE_TOOLCALL_REPAIR=0: strict again, no rebuild; the markup-in-name refusal stays ----------------------------------------
    {
        setenv("IE_TOOLCALL_REPAIR", "0", 1);
        auto p = parse(open_calls() + invoke("run_bash", param("arguments", false, "{\"command\": \"a\nb\"}")) + close_calls());
        CHECK(!p.ok && p.err.find("invalid JSON parameter") != std::string::npos, "switch off: raw newline must be an error again");
        p = parse(open_calls() + "<" + D + " invoke name=\"run_bash>\">\n" + param("command", true, "ls") + "</" + D + " invoke>\n" + close_calls());
        CHECK(!p.ok, "switch off: stray '>' name must be an error again");
        p = parse(open_calls() + invoke("a<b", param("x", true, "1")) + close_calls());
        CHECK(!p.ok, "switch off: markup name still refused");
        p = parse(open_calls() + invoke("ok", param("x", true, "1")) + close_calls());
        CHECK(p.ok && name_of(p) == "ok", "switch off: valid call unaffected");
        unsetenv("IE_TOOLCALL_REPAIR");
        p = parse(open_calls() + invoke("run_bash", param("arguments", false, "{\"command\": \"a\nb\"}")) + close_calls());
        CHECK(p.ok, "switch back on: recovered again");
    }

    // ---- the observer sees repairs and refusals (so the next real failure is captured with its raw text) -----------------------
    {
        static int repaired = 0, rejected = 0;
        ie::tool_call_observer() = [](const ie::ToolCallEvent& e) { if (std::string(e.kind) == "repaired") ++repaired; if (std::string(e.kind) == "rejected") ++rejected; };
        parse(open_calls() + invoke("w", param("a", false, "{\"x\": \"l1\nl2\"}")) + close_calls());
        parse(open_calls() + invoke("read_file", param("path", false, "README.md")) + close_calls());
        CHECK(repaired == 1 && rejected == 1, "observer counts: repaired=%d rejected=%d", repaired, rejected);
        ie::tool_call_observer() = nullptr;
    }

    if (failures) { std::printf("ds41_tool_parse_test: FAIL (%d checks failed)\n", failures); return 1; }
    std::printf("ds41_tool_parse_test: PASS\n");
    return 0;
}
