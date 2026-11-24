// tests/tool_guard_test.cpp -- ToolGuard (tagged + DSML) on hand-written replies (g++ -std=c++20 -Iinclude -I<nlohmann> tests/tool_guard_test.cpp src/engine/tool_guard.cpp)
#include "ie/tool_guard.hpp"
#include <cstdio>
using ie::ToolGuard;
static int bad = 0;
static void check(const ToolGuard& g, bool think, const char* text, bool want_ok, bool want_stop, const char* what) {
    auto s = g.start(think);
    const bool ok = g.feed(s, text);
    const bool stop = ok && g.may_stop(s);
    const bool pass = ok == want_ok && (!ok || stop == want_stop);
    if (!pass) ++bad;
    std::printf("%s %-46s ok=%d stop=%d\n", pass ? "PASS" : "FAIL", what, ok, stop);
}
int main() {
    const char* tools = R"([{"type":"function","function":{"name":"run_bash","parameters":{"type":"object","properties":{"command":{"type":"string"},"timeout":{"type":"integer"}},"required":["command"]}}},
      {"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"mode":{"type":"string","enum":["text","hex"]},"opts":{"type":"object"},"fast":{"type":"boolean"},"scale":{"type":"number"}},"required":["path"]}}},
      {"type":"function","function":{"name":"now","parameters":{"type":"object","properties":{}}}}])";
    std::string err;
    auto g = ToolGuard::make(tools, ToolGuard::xing4_tags(), err);
    if (!g) { std::printf("make failed: %s\n", err.c_str()); return 1; }
    const ToolGuard& G = *g;
    check(G, false, "Sure.<tool_call>run_bash<param_key>command</param_key><param_value>ls -la</param_value></tool_call>", true, true, "valid call");
    check(G, true, "think <tool_call>bogus</think>ok", true, true, "tags inside reasoning ignored");
    check(G, false, "<tool_call> run_bash<param_key>command</param_key><param_value>x</param_value><param_key>timeout</param_key><param_value>30</param_value></tool_call>", true, true, "two keys, int, word-start space");
    check(G, false, "<tool_call>run_bsh", false, false, "unknown tool name");
    check(G, false, "<tool_call>run_bash<param_key>cmd</param_key>", false, false, "unknown key");
    check(G, false, "<tool_call>run_bash<param_key>timeout</param_key><param_value>30</param_value></tool_call>", false, false, "close without required key");
    check(G, false, "<tool_call>run_bash<param_key>command</param_key><param_value>a</param_value><param_key>command</param_key>", false, false, "repeated key");
    check(G, false, "<tool_call>run_bash<param_key>command</param_key><param_value>a</param_value><param_key>timeout</param_key><param_value>3x", false, false, "non-integer");
    check(G, false, "<tool_call>run_bash<param_key>command</param_key><param_value>a</param_value><param_key>timeout</param_key><param_value>-</param_value>", false, false, "bare minus");
    check(G, false, "<tool_call>read_file<param_key>path</param_key><param_value>a.c</param_value><param_key>mode</param_key><param_value>hex</param_value></tool_call>", true, true, "enum ok");
    check(G, false, "<tool_call>read_file<param_key>path</param_key><param_value>a.c</param_value><param_key>mode</param_key><param_value>bin", false, false, "enum wrong");
    check(G, false, "<tool_call>read_file<param_key>path</param_key><param_value>a</param_value><param_key>opts</param_key><param_value>{\"a\": [1, 2]}</param_value></tool_call>", true, true, "json object");
    check(G, false, "<tool_call>read_file<param_key>path</param_key><param_value>a</param_value><param_key>opts</param_key><param_value>{\"a\": </param_value>", false, false, "broken json at close");
    check(G, false, "<tool_call>read_file<param_key>path</param_key><param_value>a</param_value><param_key>fast</param_key><param_value>tru", true, false, "bool prefix, mid-call: no stop");
    check(G, false, "<tool_call>read_file<param_key>path</param_key><param_value>a</param_value><param_key>scale</param_key><param_value>1.5e-3</param_value></tool_call>", true, true, "number");
    check(G, false, "<tool_call>now</tool_call>", true, true, "no-parameter tool");
    check(G, false, "<tool_call>run_bash<param_key>command</param_key><param_value>echo </param_value></tool_call><tool_call>now</tool_call>", true, true, "two calls");
    check(G, false, "<tool_call>run_bash<param_key>command</param_key><param_value>x</param_value></tool_cal", true, false, "half-written close");
    auto g2 = ToolGuard::make(tools, ToolGuard::glm_tags(), err);
    check(*g2, false, "<tool_call>run_bash<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>", true, true, "GLM spelling");
    // DeepSeek-V4.1 DSML
    auto d = ToolGuard::make_dsml(tools, err);
    const ToolGuard& D = *d;
    const std::string O = "\n\n<\uFF5CDSML\uFF5C calls>\n", I = "<\uFF5CDSML\uFF5C invoke name=\"", P = "<\uFF5CDSML\uFF5C parameter name=\"",
                      PC = "</\uFF5CDSML\uFF5C parameter>\n", IC = "</\uFF5CDSML\uFF5C invoke>\n", CC = "</\uFF5CDSML\uFF5C calls>";
    auto chk = [&](bool think, const std::string& t, bool ok, bool stop, const char* what) { check(D, think, t.c_str(), ok, stop, what); };
    chk(false, "Running it." + O + I + "run_bash\">\n" + P + "command\" string=\"true\">ls -la" + PC + IC + CC, true, true, "DSML valid call");
    chk(true, "plan</think>ok" + O + I + "run_bash\">\n" + P + "command\" string=\"true\">x" + PC + P + "timeout\" string=\"false\">30" + PC + IC + CC, true, true, "DSML int after reasoning");
    chk(false, "no blank line\n<\uFF5CDSML\uFF5C", false, false, "DSML without the blank line");
    chk(false, O + I + "run_bsh", false, false, "DSML unknown tool");
    chk(false, O + I + "run_bash\">\n" + P + "timeout\" string=\"true\">", false, false, "DSML int flagged as string");
    chk(false, O + I + "run_bash\">\n" + P + "timeout\" string=\"false\">30" + PC + IC, false, false, "DSML missing required");
    chk(false, O + I + "read_file\">\n" + P + "path\" string=\"true\">a" + PC + P + "opts\" string=\"false\">{\"a\": " + PC, false, false, "DSML broken JSON");
    chk(false, O + I + "now\">\n" + "\n" + IC + I + "now\">\n" + IC + CC, true, true, "DSML two zero-arg calls (blank line)");
    chk(false, O + I + "now\">\n" + IC + CC + "trailing", false, false, "DSML text after the block");
    chk(false, O + I + "run_bash\">\n" + P + "command\" string=\"true\">x" + PC, true, false, "DSML mid-call: no stop");
    // tool_choice
    auto gn = g->with_choice("none", err); auto gr = g->with_choice("required", err); auto gf = g->with_choice("fn:read_file", err);
    check(*gn, false, "Plain answer.", true, true, "choice none: text ok");
    check(*gn, false, "<tool_call>now</tool_call>", false, false, "choice none: no call");
    check(*gr, false, "Sure, let me", false, false, "choice required: text first refused");
    check(*gr, false, "", true, false, "choice required: no stop before a call");
    check(*gr, true, "think</think> <tool_call>now</tool_call>", true, true, "choice required: call after reasoning");
    check(*gf, false, "<tool_call>run_bash", false, false, "choice fn: other tool refused");
    check(*gf, false, "<tool_call>read_file<param_key>path</param_key><param_value>a</param_value></tool_call>", true, true, "choice fn: named tool");
    check(*gf, false, "<tool_call>\n\n\n", false, false, "name: no whitespace loop");
    check(*gr, false, "\n\n\n\n", false, false, "choice required: no whitespace loop");
    auto dr = d->with_choice("required", err);
    check(*dr, true, ("x</think>" + O + I + "now\">\n" + IC + CC).c_str(), true, true, "DSML required: block at once");
    check(*dr, false, "Let me check", false, false, "DSML required: text refused");
    // Qwen XML (Flash-Next, Qwen3.8, MiMo)
    auto x = ToolGuard::make_xml(tools, err);
    check(*x, false, "Checking.\n<tool_call>\n<function=run_bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>", true, true, "XML valid call");
    check(*x, true, "hm</think>\n\n<tool_call>\n<function=run_bash>\n<parameter=command>\nx\n</parameter>\n<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>", true, true, "XML int value");
    check(*x, false, "<tool_call>\n<function=run_bsh", false, false, "XML unknown tool");
    check(*x, false, "<tool_call>\n<function=run_bash>\n<parameter=timeout>\nthirty", false, false, "XML int gets text");
    check(*x, false, "<tool_call>\n<function=run_bash>\n<parameter=timeout>\n30\n</parameter>\n</function>", false, false, "XML missing required");
    check(*x, false, "<tool_call>\n<function=run_bash>\n<parameter=command>\nx\n</parameter>\n<parameter=command>", false, false, "XML repeated key");
    check(*x, false, "<tool_call>\n<function=run_bash>\n<parameter=command>\nx", true, false, "XML mid-call: no stop");
    check(*x, false, "<tool_call>\n\n\n\n", false, false, "XML whitespace loop");
    check(*x, false, "<tool_call>\n<function=now>\n</function>\n</tool_call>\n<tool_call>\n<function=now>\n</function>\n</tool_call>", true, true, "XML two zero-arg calls");
    // Dream's lazy tools: a request offering tool_call lets a call name any tool
    const char* dream = R"([{"type":"function","function":{"name":"tool_call","parameters":{"type":"object","properties":{"name":{"type":"string"},"arguments":{"type":"object"}},"required":["name"]}}},
                           {"type":"function","function":{"name":"task","parameters":{"type":"object","properties":{"prompt":{"type":"string"}},"required":["prompt"]}}}])";
    auto od = ToolGuard::make_dsml(dream, err); auto ox = ToolGuard::make_xml(dream, err); auto ot = ToolGuard::make(dream, ToolGuard::xing4_tags(), err);
    check(*od, false, (O + I + "run_bash\">\n" + P + "command\" string=\"true\">ls" + PC + IC + CC).c_str(), true, true, "open DSML: unlisted tool runs");
    check(*od, false, (O + I + "run_bash\">\n" + P + "timeout\" string=\"false\">{\"a\": " + PC).c_str(), false, false, "open DSML: JSON still checked");
    check(*od, false, (O + I + "run_bash\">\n" + P + "c\" string=\"true\">x" + PC + P + "c\" string=\"").c_str(), false, false, "open DSML: no key twice");
    check(*od, false, (O + I + "run bash").c_str(), false, false, "open DSML: name is an identifier");
    check(*od, false, (O + I + "task\">\n" + IC).c_str(), false, false, "open DSML: listed tool keeps its schema");
    check(*ox, false, "<tool_call>\n<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n</function>\n</tool_call>", true, true, "open XML: unlisted tool runs");
    check(*ot, false, "<tool_call>web_search<param_key>q</param_key><param_value>x</param_value></tool_call>", true, true, "open tagged: unlisted tool runs");
    check(*g, false, "<tool_call>web_searchx", false, false, "strict without the dispatcher");
    // any harness: a tool named in the conversation (no dispatcher) may be called; a made-up one may not
    {
        ie::ToolGuardPrompt pr("You can also call web_fetch with a url to read a page.");
        auto pt = ToolGuard::make(tools, ToolGuard::xing4_tags(), err);
        auto pd = ToolGuard::make_dsml(tools, err);
        auto px = ToolGuard::make_xml(tools, err);
        check(*pt, false, "<tool_call>web_fetch<param_key>url</param_key><param_value>x</param_value></tool_call>", true, true, "prompt name: tagged call");
        check(*pt, false, "<tool_call>zzz_fake<", false, false, "prompt name: made-up name refused");
        check(*pd, false, (O + I + "web_fetch\">\n" + P + "url\" string=\"true\">x" + PC + IC + CC).c_str(), true, true, "prompt name: DSML call");
        check(*px, false, "<tool_call>\n<function=web_fetch>\n<parameter=url>\nx\n</parameter>\n</function>\n</tool_call>", true, true, "prompt name: XML call");
        check(*px, false, "<tool_call>\n<function=web_fetc>", false, false, "prompt name: a prefix only is refused");
        check(*pt, false, "<tool_call>run_bash<param_key>timeout_x", false, false, "prompt name: listed tools keep their schema");
    }
    check(*g, false, "<tool_call>web_fetch<", false, false, "no prompt: unlisted refused");
    std::printf("%s (%d failed)\n", bad ? "FAILED" : "ALL PASS", bad);
    return bad ? 1 : 0;
}
