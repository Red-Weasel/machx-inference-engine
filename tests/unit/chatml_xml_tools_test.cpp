// tests/unit/chatml_xml_tools_test.cpp -- P4 B27: the Qwen3.6+ XML tool convention (tokenizer.hpp ChatmlXmlTools).
// Host-only, no GPU. Two halves:
//   1. the renderer against the GGUFs' own chat templates: tests/data/chatml_xml_tools_golden.json (gen_golden.py: the
//      OpenAI request as the server gets it, the expected prompt from the HF transformers jinja environment). Every case
//      goes through the real request parser (ie::oai::parse_chat_request) and must match byte for byte.
//   2. the parser: XML calls with typed parameters, multi-call, the unterminated tail, the hybrid salvage forms the 35B
//      wrote under the JSON preamble, a pure JSON block left to the server (the fallback), and the server frames built
//      from the result (chat_completion_json, the streaming tool_calls delta).
//   3. P4 B35, the reasoning split of a Qwen thinking reply: split_chatml_think, Engine::chat's ChatML reply
//      (finish_chatml_reply: B27 / B28's calls, then the split) and its non-stream body, the server's stream logic
//      (ChatmlThinkStream, mirrored) == the non-stream result for every fragmentation, which models split; with --gguf the
//      real tokenizers (the 27B, Flash-Next, the 35B): the decode that drops </think> (the bug), show_special (the fix).
// usage: chatml_xml_tools_test [golden.json] [--gguf PATH]...
#undef NDEBUG
#include "ie/engine.hpp"
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/openai_proto.hpp"
#include "ie/reasoning.hpp"
#include "ie/tokenizer.hpp"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <span>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;   // the request's key order IS the render (tojson keeps it)

namespace {

int fails = 0;
void check(bool ok, const char* what) {
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++fails;
}

// The first differing byte, with context, for a golden mismatch.
void show_diff(const std::string& got, const std::string& exp) {
    size_t i = 0;
    while (i < got.size() && i < exp.size() && got[i] == exp[i]) ++i;
    std::printf("    first difference at byte %zu (got %zu bytes, expected %zu)\n", i, got.size(), exp.size());
    const size_t a = i > 60 ? i - 60 : 0;
    std::printf("    got:      %s\n    expected: %s\n", json(got.substr(a, 160)).dump().c_str(), json(exp.substr(a, 160)).dump().c_str());
}

void golden(const char* path) {
    std::ifstream f(path);
    if (!f) { std::printf("  golden file %s not readable\n", path); ++fails; return; }
    json g = json::parse(f);
    for (auto& [model, d] : g["detect"].items()) {
        const ie::ChatmlXmlTools t = ie::chatml_xml_tools_from_template(g["templates"][model].get<std::string>());
        check(t.enabled == d["enabled"].get<bool>() && t.think_all_history == d["think_all_history"].get<bool>(),
              ("template detection " + model).c_str());
    }
    for (const auto& c : g["cases"]) {
        const std::string tmpl = g["templates"][c["model"].get<std::string>()].get<std::string>();
        const ie::ChatmlXmlTools xml = ie::chatml_xml_tools_from_template(tmpl);
        const bool model_has_think = tmpl.find("<think>") != std::string::npos;
        ie::oai::ChatRequest cr = ie::oai::parse_chat_request(c["request"].dump());
        assert(cr.error.empty());
        const std::string pre = c["reasoning_preamble"].get<std::string>();
        const std::string got = ie::build_chatml_prompt(cr.turns, true, cr.enable_thinking, cr.tools_json,
                                                        model_has_think, pre, xml);
        const std::string exp = c["expected"].get<std::string>();
        const bool ok = got == exp;
        check(ok, ("golden: " + c["name"].get<std::string>()).c_str());
        if (!ok) show_diff(got, exp);
    }
    // The old preamble is untouched: xml disabled == the pre-B27 render (the tokenizer_test goldens still hold).
    {
        const auto& c = g["cases"][0];
        ie::oai::ChatRequest cr = ie::oai::parse_chat_request(c["request"].dump());
        const std::string a = ie::build_chatml_prompt(cr.turns, true, true, cr.tools_json, true, {});
        const std::string b = ie::build_chatml_prompt(cr.turns, true, true, cr.tools_json, true, {}, ie::ChatmlXmlTools{});
        check(a == b && a.find("You may call one or more functions") != std::string::npos &&
              a.find("<function=") == std::string::npos, "xml disabled: the Qwen3 JSON preamble, unchanged");
        // No tools: the XML flag changes nothing.
        ie::oai::ChatRequest plain = ie::oai::parse_chat_request(R"({"messages":[{"role":"system","content":" s "},{"role":"user","content":"u\n"}]})");
        const ie::ChatmlXmlTools on{true, false};
        check(ie::build_chatml_prompt(plain.turns, true, true, {}, true, {}, on) ==
              ie::build_chatml_prompt(plain.turns, true, true, {}, true, {}), "no tools: xml flag is inert");
    }
}

const char* kTools = R"([
 {"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{
   "path":{"type":"string"},"start_line":{"type":"integer"},"line_count":{"type":"integer"},"query":{"type":"string"}},"required":["path"]}}},
 {"type":"function","function":{"name":"recall","parameters":{"type":"object","properties":{
   "query":{"type":"string"},"alternative_queries":{"type":"array","items":{"type":"string"}},"limit":{"type":"integer"},
   "all_projects":{"type":"boolean"},"ratio":{"type":"number"},"opts":{"type":"object"}},"required":["query"]}}},
 {"type":"function","function":{"name":"create_ticket","parameters":{"type":"object","properties":{
   "title":{"type":"string"},"labels":{"type":"array","items":{"type":"string"}},"assignee":{"type":"object"},
   "priority":{"type":"integer"},"blocking":{"type":"boolean"},"ratio":{"type":"number"},"maybe":{"type":["string","null"]},
   "count":{"type":["integer","null"]}},"required":["title"]}}},
 {"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}},
 {"type":"function","function":{"name":"task","parameters":{"type":"object","properties":{
   "subagent_type":{"type":"string","enum":["explorer","coder"]},"prompt":{"type":"string"}},"required":["subagent_type","prompt"]}}}])";

json args_of(const ie::ChatmlToolCalls& r, size_t i) {
    json calls = json::parse(r.tool_calls_json);
    return json::parse(calls.at(i)["function"]["arguments"].get<std::string>());
}
std::string name_of(const ie::ChatmlToolCalls& r, size_t i) {
    return json::parse(r.tool_calls_json).at(i)["function"]["name"].get<std::string>();
}

void parser() {
    // 1. One XML call, every schema type, a multi-line string, an unknown parameter.
    {
        const std::string t = "Let me read it.\n\n<tool_call>\n<function=read_file>\n<parameter=path>\ndream/core/engine.py\n</parameter>\n"
                              "<parameter=start_line>\n1\n</parameter>\n<parameter=line_count>\n150\n</parameter>\n"
                              "<parameter=query>\nline one\nline two\n</parameter>\n<parameter=extra>\n42\n</parameter>\n</function>\n</tool_call>";
        const auto r = ie::parse_chatml_xml_tool_calls(t, kTools);
        check(!r.tool_calls_json.empty() && name_of(r, 0) == "read_file", "xml: one call, name");
        const json a = args_of(r, 0);
        check(a["path"] == "dream/core/engine.py" && a["start_line"] == 1 && a["line_count"] == 150 &&
              a["query"] == "line one\nline two" && a["extra"] == 42, "xml: typed parameters (string, integer x2, multi-line, unknown JSON-looking)");
        check(r.content == "Let me read it.\n\n", "xml: the prose before the call stays content");
    }
    // 2. Two calls; number / boolean / array / object typing; a string that looks like a number stays a string.
    {
        const std::string t = "<tool_call>\n<function=recall>\n<parameter=query>\n123\n</parameter>\n<parameter=alternative_queries>\n[\"a\", \"b c\"]\n</parameter>\n"
                              "<parameter=limit>\n3\n</parameter>\n<parameter=all_projects>\ntrue\n</parameter>\n<parameter=ratio>\n0.25\n</parameter>\n"
                              "<parameter=opts>\n{\"k\": \"v\", \"m\": [1, 2]}\n</parameter>\n</function>\n</tool_call>\n"
                              "<tool_call>\n<function=read_file>\n<parameter=path>\nx.py\n</parameter>\n</function>\n</tool_call>";
        const auto r = ie::parse_chatml_xml_tool_calls(t, kTools);
        const json calls = json::parse(r.tool_calls_json);
        check(calls.size() == 2 && name_of(r, 0) == "recall" && name_of(r, 1) == "read_file", "xml: two calls");
        const json a = args_of(r, 0);
        check(a["query"] == "123" && a["alternative_queries"] == json::array({"a", "b c"}) && a["limit"] == 3 &&
              a["all_projects"] == true && a["ratio"] == 0.25 && a["opts"] == json({{"k", "v"}, {"m", {1, 2}}}),
              "xml: string-typed digits stay text; array / boolean / number / object parse");
        check(r.content == "\n", "xml: only the separator remains as content");
    }
    // 3. A call without parameters; a value with no </parameter> (repair); an unterminated last block.
    {
        const auto r = ie::parse_chatml_xml_tool_calls("<tool_call>\n<function=recall>\n</function>\n</tool_call>", kTools);
        check(!r.tool_calls_json.empty() && args_of(r, 0) == json::object(), "xml: no parameters -> {}");
        const auto rep = ie::parse_chatml_xml_tool_calls(
            "<tool_call>\n<function=read_file>\n<parameter=path>\nx.py\n<parameter=start_line>\n5\n</parameter>\n</function>\n</tool_call>", kTools);
        check(!rep.tool_calls_json.empty() && args_of(rep, 0)["path"] == "x.py" && args_of(rep, 0)["start_line"] == 5,
              "xml: a missing </parameter> is repaired at the next parameter");
        const auto cut = ie::parse_chatml_xml_tool_calls("<tool_call>\n<function=read_file>\n<parameter=path>\nx.py\n</parameter>\n</function>\n", kTools);
        check(!cut.tool_calls_json.empty() && args_of(cut, 0)["path"] == "x.py", "xml: unterminated <tool_call> with a closed </function> is accepted");
        const auto trunc = ie::parse_chatml_xml_tool_calls("<tool_call>\n<function=read_file>\n<parameter=path>\nx.p", kTools);
        check(trunc.tool_calls_json.empty() && trunc.content == "<tool_call>\n<function=read_file>\n<parameter=path>\nx.p",
              "xml: a truncated call stays text");
        const auto reopen = ie::parse_chatml_xml_tool_calls("<tool_call>\n<function=read_file>\n<parameter=path>\nx.py\n</parameter>\n</function>\n<tool_call>", kTools);
        check(!reopen.tool_calls_json.empty() && args_of(reopen, 0)["path"] == "x.py", "xml: a bare re-open as the close is stripped");
    }
    // 4. The hybrid forms seen live (runs/base_replay): salvaged into the same structured call.
    {
        const char* live[] = {
            "The user wants the file.\n\n\n<tool_call>\n<name=\"read_file\", \"arguments\": {\"path\": \"dream/core/engine.py\", \"line_count\": 150}}\n</tool_call>",
            "<tool_call>\n{\"name=\"read_file\",arguments\":{\"path\":\"dream/core/engine.py\",\"line_count\":150,\"start_line\":1}}\n</tool_call>",
            "<tool_call>\n<name=\"read_file\",arguments={\"path\":\"dream/core/engine.py\",\"line_count\":150}}\n</tool_call>",
            "<tool_call>\n{\"name=\"read_file\", \"arguments\": {\"path\": \"dream/core/engine.py\", \"line_count\": 150}}\n</tool_call>",
            "<tool_call>\n<tool_call>\n<tool_call>\n<name>read_file</name>\n<arguments>{\"path\": \"dream/core/engine.py\", \"line_count\": 150, \"start_line\": 1}</arguments>\n</tool_call>",
        };
        int ok = 0;
        for (const char* t : live) {
            const auto r = ie::parse_chatml_xml_tool_calls(t, kTools);
            if (!r.tool_calls_json.empty() && name_of(r, 0) == "read_file" && args_of(r, 0)["path"] == "dream/core/engine.py" &&
                args_of(r, 0)["line_count"] == 150) ++ok;
            else std::printf("    salvage missed: %s -> %s\n", json(t).dump().c_str(), r.tool_calls_json.c_str());
        }
        check(ok == 5, "salvage: the five live hybrid forms -> read_file(path, line_count)");
        const auto cut = ie::parse_chatml_xml_tool_calls("<tool_call>\n<name=\"read_file\", \"arguments\": {\"path\": \"dre", kTools);
        check(cut.tool_calls_json.empty(), "salvage: a cut-off hybrid stays text");
        const auto noargs = ie::parse_chatml_xml_tool_calls("<tool_call>\n<name=\"recall\"}\n</tool_call>", kTools);
        check(!noargs.tool_calls_json.empty() && name_of(noargs, 0) == "recall" && args_of(noargs, 0) == json::object(),
              "salvage: a hybrid without arguments -> {}");
    }
    // 5. The Qwen3 JSON form is parsed here too (F4); the server's JSON parser stays the parser of the other templates
    //    and never runs on an examined reply ("[]" is authoritative).
    {
        const std::string t = "<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"x.py\"}}\n</tool_call>";
        const auto r = ie::parse_chatml_xml_tool_calls(t, kTools);
        check(!r.tool_calls_json.empty() && name_of(r, 0) == "read_file" && args_of(r, 0)["path"] == "x.py" && r.content.empty(),
              "json block: parsed into the same structured call");
        ie::GenerateResult gr; gr.text = t; gr.finish_reason = "stop";
        const json body = json::parse(ie::oai::chat_completion_json("m", gr, "id1", 7));
        check(body["choices"][0]["finish_reason"] == "tool_calls" &&
              body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "read_file",
              "json block: the server's own parser still handles it when the engine set nothing (other templates)");
        ie::GenerateResult ex; ex.text = t; ex.finish_reason = "stop"; ex.tool_calls_json = "[]";
        const json b2 = json::parse(ie::oai::chat_completion_json("m", ex, "id1", 7));
        check(!b2["choices"][0]["message"].contains("tool_calls") && b2["choices"][0]["message"]["content"] == t &&
              b2["choices"][0]["finish_reason"] == "stop", "\"[]\" from the engine: no server fallback, the text is content");
        const auto none = ie::parse_chatml_xml_tool_calls("just prose, no call", kTools);
        check(none.tool_calls_json.empty() && none.content == "just prose, no call", "no block: text unchanged");
    }
    // 6. The server frames from an engine-parsed result: non-stream message and the streaming delta (ids, index).
    {
        const auto r = ie::parse_chatml_xml_tool_calls(
            "<tool_call>\n<function=read_file>\n<parameter=path>\nx.py\n</parameter>\n</function>\n</tool_call>\n"
            "<tool_call>\n<function=recall>\n<parameter=query>\nq\n</parameter>\n</function>\n</tool_call>", kTools);
        ie::GenerateResult gr; gr.text = r.content; gr.tool_calls_json = r.tool_calls_json; gr.finish_reason = "stop";
        const json body = json::parse(ie::oai::chat_completion_json("m", gr, "id1", 7));
        const json msg = body["choices"][0]["message"];
        check(body["choices"][0]["finish_reason"] == "tool_calls" && msg["tool_calls"].size() == 2 &&
              msg["tool_calls"][0]["id"] == "call_0" && msg["tool_calls"][1]["id"] == "call_1" &&
              msg["tool_calls"][1]["function"]["name"] == "recall" && msg["content"].is_null(),
              "server: chat_completion_json -> two structured calls, null content");
        const std::string frame = ie::oai::chat_chunk_sse_tool_calls_json("m", "chatcmpl-1", 7, r.tool_calls_json);
        const json d = json::parse(frame.substr(6))["choices"][0]["delta"];
        check(d["tool_calls"].size() == 2 && d["tool_calls"][0]["index"] == 0 && d["tool_calls"][1]["index"] == 1 &&
              d["tool_calls"][0]["id"].is_string() &&
              json::parse(d["tool_calls"][0]["function"]["arguments"].get<std::string>())["path"] == "x.py",
              "server: streaming tool_calls delta with index + ids");
    }
    auto P = [](const char* k, const char* v) { return std::string("<parameter=") + k + ">\n" + v + "\n</parameter>\n"; };
    const std::string pre = "<tool_call>\n<function=create_ticket>\n", post = "</function>\n</tool_call>";
    // 7. F1: only the answer span (the text after the model's </think>) is scanned; the reasoning is kept verbatim.
    {
        const std::string quote = "<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>";
        const std::string call = "<tool_call>\n<function=read_file>\n<parameter=path>\nx.py\n</parameter>\n</function>\n</tool_call>";
        const std::string head = "The user pasted this:\n" + quote + "\nIt is XML markup. ";
        const std::string t = head + "Let me read the file.\n" + call;
        const auto r = ie::parse_chatml_xml_tool_calls(t, kTools, head.size());
        check(!r.tool_calls_json.empty() && json::parse(r.tool_calls_json).size() == 1 && name_of(r, 0) == "read_file" &&
              r.content == head + "Let me read the file.\n", "F1: a block quoted in the reasoning is not a call; the real call after it is");
        const auto none = ie::parse_chatml_xml_tool_calls(t, kTools, std::string::npos);
        check(none.tool_calls_json.empty() && none.content == t, "F1: no answer span (npos: the reply never closed its reasoning) -> nothing parsed");
        const auto only_quote = ie::parse_chatml_xml_tool_calls(head + "Just markup, no call.", kTools, head.size());
        check(only_quote.tool_calls_json.empty() && only_quote.content == head + "Just markup, no call.", "F1: the quote alone before the answer -> no call");
        const auto all = ie::parse_chatml_xml_tool_calls(t, kTools, 0);
        check(!all.tool_calls_json.empty() && json::parse(all.tool_calls_json).size() == 2, "F1: answer_start 0 (thinking off) scans the whole reply");
        const auto past = ie::parse_chatml_xml_tool_calls(t, kTools, t.size() + 5);
        check(past.tool_calls_json.empty() && past.content == t, "F1: an answer_start past the end -> nothing parsed");
    }
    // 8. F2: booleans, numerics and Python literals typed by the schema.
    {
        const auto r = ie::parse_chatml_xml_tool_calls(pre + P("title", "t") + P("blocking", "True") + P("priority", "02") + P("ratio", "1e3") +
                                                       P("labels", "['gpu', 'urgent']") + P("assignee", "{'name': 'Kari'}") + post, kTools);
        check(!r.tool_calls_json.empty(), "F2: the call parses");
        const json a = args_of(r, 0);
        check(a["blocking"] == true && a["priority"] == 2 && a["ratio"] == 1000.0 && a["labels"] == json::array({"gpu", "urgent"}) &&
              a["assignee"] == json({{"name", "Kari"}}), "F2: True -> true, 02 -> 2, 1e3 -> 1000, Python list / dict -> array / object");
        const auto r2 = ie::parse_chatml_xml_tool_calls(pre + P("title", "t") + P("blocking", "FALSE") + P("priority", "two") + P("other", "1") + post, kTools);
        const json b = args_of(r2, 0);
        check(b["blocking"] == false && b["priority"] == "two" && b["other"] == 1, "F2: FALSE -> false; 'two' stays text; an unknown param keeps the JSON-looking rule");
        const auto r3 = ie::parse_chatml_xml_tool_calls(pre + P("title", "t") + P("blocking", "yes") + P("ratio", "1,5") + post, kTools);
        check(args_of(r3, 0)["blocking"] == "yes" && args_of(r3, 0)["ratio"] == "1,5", "F2: 'yes' / '1,5' are not a boolean / number, they stay text");
    }
    // 9. F3: a function the tools do not list is never a call (a quoted example, a salvage from prose).
    {
        const std::string ex = "The format is:\n\n```\n<tool_call>\n<function=example_fn>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n```\nThat's it.";
        const auto r = ie::parse_chatml_xml_tool_calls(ex, kTools);
        check(r.tool_calls_json.empty() && r.content == ex, "F3: an example call to a function not in the tools stays text");
        const std::string prose = "In the Qwen format, <tool_call> tags wrap the name and the arguments {\"a\": 1} of a call </tool_call> and that is all.";
        const auto p = ie::parse_chatml_xml_tool_calls(prose, kTools);
        check(p.tool_calls_json.empty() && p.content == prose, "F3: prose mentioning name / arguments is not salvaged into a call");
        const auto h = ie::parse_chatml_xml_tool_calls("<tool_call>\n<name=\"nope\", \"arguments\": {\"a\": 1}}\n</tool_call>", kTools);
        check(h.tool_calls_json.empty(), "F3: a hybrid with an unknown name stays text");
        const auto j = ie::parse_chatml_xml_tool_calls("<tool_call>\n{\"name\": \"nope\", \"arguments\": {}}\n</tool_call>", kTools);
        check(j.tool_calls_json.empty(), "F3: a JSON block with an unknown name stays text");
        const auto nt = ie::parse_chatml_xml_tool_calls("<tool_call>\n<function=anything>\n</function>\n</tool_call>", "");
        check(!nt.tool_calls_json.empty() && name_of(nt, 0) == "anything", "F3: no tools array given -> no name check");
    }
    // 10. F4: an XML call and a JSON call in one reply are both kept, in order; prose after the calls is content.
    {
        const std::string t = "<tool_call>\n<function=read_file>\n<parameter=path>\nx.py\n</parameter>\n</function>\n</tool_call>\n"
                              "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}\n</tool_call>\nAfter text.";
        const auto r = ie::parse_chatml_xml_tool_calls(t, kTools);
        check(!r.tool_calls_json.empty() && json::parse(r.tool_calls_json).size() == 2 && name_of(r, 0) == "read_file" &&
              name_of(r, 1) == "get_weather" && args_of(r, 1)["city"] == "Oslo" && r.content == "\n\nAfter text.",
              "F4: XML + JSON calls both kept in order; the prose after them is content");
        const auto s = ie::parse_chatml_xml_tool_calls("<tool_call>\n{\"name\": \"get_weather\", \"arguments\": \"{\\\"city\\\": \\\"Oslo\\\"}\"}\n</tool_call>", kTools);
        check(!s.tool_calls_json.empty() && args_of(s, 0)["city"] == "Oslo", "F4: the JSON form with string arguments passes them through");
    }
    // 11. F6: a type list.
    {
        const auto r = ie::parse_chatml_xml_tool_calls(pre + P("title", "t") + P("maybe", "123") + P("count", "5") + post, kTools);
        check(!r.tool_calls_json.empty() && args_of(r, 0)["maybe"] == "123" && args_of(r, 0)["count"] == 5,
              "F6: [\"string\",\"null\"] digits stay text; [\"integer\",\"null\"] parses");
    }
    // 12. B28: the reply never closed its reasoning (thinking on, no </think>: answer_start npos) but ended on its own
    //     (finish "stop", the caller's rule): the run of complete calls that ENDS the reply is accepted; a block followed
    //     by more text (one quoted while reasoning) is not; a cut-off block leaves text. The answer-span parser is unchanged.
    {
        const std::string call = "<tool_call>\n<function=read_file>\n<parameter=path>\nx.py\n</parameter>\n</function>\n</tool_call>";
        const std::string call2 = "<tool_call>\n<function=recall>\n<parameter=query>\nq\n</parameter>\n</function>\n</tool_call>";
        const std::string quote = "<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>";
        const std::string think = "The user wants me to fire off the rest. I've done 4 so far.\n\n\nFour done. Continuing:\n\n";
        const auto a = ie::parse_chatml_xml_tail_calls(think + call, kTools);
        check(!a.tool_calls_json.empty() && json::parse(a.tool_calls_json).size() == 1 && name_of(a, 0) == "read_file" &&
              args_of(a, 0)["path"] == "x.py" && a.content == think, "B28: a tail call with no </think> is accepted; the reasoning stays content");
        const auto a2 = ie::parse_chatml_xml_tail_calls(think + call + "\n", kTools);
        check(!a2.tool_calls_json.empty() && a2.content == think + "\n", "B28: trailing whitespace after the tail call is fine");
        const auto n = ie::parse_chatml_xml_tool_calls(think + call, kTools, std::string::npos);
        check(n.tool_calls_json.empty() && n.content == think + call, "B28: the answer-span parser at npos still parses nothing (the length-capped case)");
        const std::string quoted = "The user pasted this:\n" + quote + "\nIt is XML markup, not a call. I will answer in prose.";
        const auto q = ie::parse_chatml_xml_tail_calls(quoted, kTools);
        check(q.tool_calls_json.empty() && q.content == quoted, "B28: a block quoted mid-reasoning and followed by text is not a call");
        const std::string head = "The user pasted this:\n" + quote + "\nIt is markup. Let me read the file instead.\n";
        const auto tq = ie::parse_chatml_xml_tail_calls(head + call, kTools);
        check(!tq.tool_calls_json.empty() && json::parse(tq.tool_calls_json).size() == 1 && name_of(tq, 0) == "read_file" &&
              tq.content == head, "B28: a quote, text, then a tail call -> only the tail; the quote stays in content");
        const std::string cut = think + "<tool_call>\n<function=read_file>\n<parameter=path>\nx.p";
        const auto c = ie::parse_chatml_xml_tail_calls(cut, kTools);
        check(c.tool_calls_json.empty() && c.content == cut, "B28: a cut-off tail block stays text");
        const auto c2 = ie::parse_chatml_xml_tail_calls(head + "<tool_call>\n<function=read_file>\n<parameter=path>\nx", kTools);
        check(c2.tool_calls_json.empty(), "B28: a quote then a cut-off block -> nothing");
        const auto m = ie::parse_chatml_xml_tail_calls(think + call + "\n" + call2 + "\n\n", kTools);
        check(!m.tool_calls_json.empty() && json::parse(m.tool_calls_json).size() == 2 && name_of(m, 0) == "read_file" &&
              name_of(m, 1) == "recall" && m.content == think + "\n\n\n", "B28: two tail calls are both accepted, in order");
        const std::string unk = "<tool_call>\n<function=example_fn>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>";
        const auto u = ie::parse_chatml_xml_tail_calls(think + unk + "\n" + call, kTools);
        check(!u.tool_calls_json.empty() && json::parse(u.tool_calls_json).size() == 1 && name_of(u, 0) == "read_file" &&
              u.content == think + unk + "\n", "B28: an unknown-function block before the tail stays text; the tail is accepted");
        const auto qq = ie::parse_chatml_xml_tail_calls(think + quote + "\n" + call, kTools);
        check(!qq.tool_calls_json.empty() && json::parse(qq.tool_calls_json).size() == 2,
              "B28: two known blocks separated only by whitespace are two calls (indistinguishable from parallel calls)");
        const auto none = ie::parse_chatml_xml_tail_calls("just prose, no call", kTools);
        check(none.tool_calls_json.empty() && none.content == "just prose, no call", "B28: no block: text unchanged");
        // The live reply (the Dream lead's last turn, 2026-09-30 12:04): reasoning, then the call, no </think> between.
        const std::string live = "The user wants me to fire off 15 explorer subagents at once. I've done 4 so far (engine.py, system_prompt.py, "
            "settings.py, inference_coordination.py). I need to continue with the remaining 11.\n\nLet me try 8 more explorers this time "
            "(total would be 12 done, 3 remaining for next round).\n\n\nFour done (engine.py, system_prompt.py, settings.py, "
            "inference_coordination.py). Continuing with more \xe2\x80\x94 batching these:\n\n";
        const std::string live_call = "<tool_call>\n<function=task>\n<parameter=prompt>\nRead first 150 lines of /tmp/x/dream/core/tool_budget_schemas.py. "
            "Report in exactly three sentences what the file does, then one thing that could break.\n</parameter>\n<parameter=subagent_type>\n"
            "explorer\n</parameter>\n</function>\n</tool_call>";
        const auto l = ie::parse_chatml_xml_tail_calls(live + live_call, kTools);
        check(!l.tool_calls_json.empty() && json::parse(l.tool_calls_json).size() == 1 && name_of(l, 0) == "task" &&
              args_of(l, 0)["subagent_type"] == "explorer" && args_of(l, 0)["prompt"].get<std::string>().rfind("Read first 150 lines", 0) == 0 &&
              l.content == live, "B28: the live lead reply (reasoning + a task call, no </think>) -> one structured task call");
        // The server frames from the tail result: the reasoning is the content beside the call.
        ie::GenerateResult gr; gr.text = l.content; gr.tool_calls_json = l.tool_calls_json; gr.finish_reason = "stop";
        const json body = json::parse(ie::oai::chat_completion_json("m", gr, "id1", 7));
        check(body["choices"][0]["finish_reason"] == "tool_calls" && body["choices"][0]["message"]["tool_calls"].size() == 1 &&
              body["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "task" &&
              body["choices"][0]["message"]["content"] == live, "B28: chat_completion_json -> the task call, the reasoning as content");
    }
}

// ---- P4 B35: thinking goes to reasoning_content ----

// Flash-Next's thinking reply in smoke022 (runs/fn_new, M3T) with the </think> the decode dropped put back: the
// template opens <think> in the prompt, so the model writes the reasoning, "\n</think>\n\n", then the answer.
const std::string kM3TReason = "We need answer user's question: \"And the smallest of the three?\" Context: three planets "
    "Mars, Venus, Jupiter. Smallest among them is Mars? Let's verify diameters: Mars ~6779 km, Venus ~12104 km, Jupiter "
    "~139820 km. So Mars smallest. Need terse. Final: Mars.";
const std::string kM3TRaw = kM3TReason + "\n</think>\n\nMars.";
// ...and what the server returned as content (reasoning_content empty): smoke022 results.json
const std::string kM3TSmokeContent = kM3TReason + "\n\n\nMars.";

// The real tokenizers (--gguf: the 27B, Flash-Next, the 35B).
void gguf_think(const std::string& path) {
    const std::string tag = path.substr(path.rfind('/') + 1) + ": ";
    ie::GgufReader g;
    if (auto e = g.open(path); !e.empty()) { check(false, (tag + "open: " + e).c_str()); return; }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { check(false, (tag + "tokenizer: " + e).c_str()); return; }
    const int32_t close = tok.find_token("</think>");
    const int32_t keep[2] = {tok.find_token("<tool_call>"), tok.find_token("</tool_call>")};   // the engine's keep list
    check(close >= 0 && tok.is_special(close) && keep[0] >= 0 && keep[1] >= 0,
          (tag + "</think> is a special token to this tokenizer (GGUF type USER_DEFINED)").c_str());
    const std::vector<int32_t> ids = tok.encode(kM3TRaw, /*allow_special=*/true);
    check(std::count(ids.begin(), ids.end(), close) == 1, (tag + "the reply's </think> is one token").c_str());
    // The root cause, reproduced: the decode every serving path runs (skip_special, the tool-call ids kept) drops the
    // tag, so the reasoning and the answer run together -- smoke022's content, byte for byte.
    check(tok.decode(ids, true, keep) == kM3TSmokeContent,
          (tag + "decode(skip_special) drops </think>: the smoke content (reasoning + \"\\n\\n\\n\" + answer)").c_str());
    // The fix: the engine shows the tag (Engine::load, for a Qwen thinking template), every decode keeps it, and the
    // reply splits there.
    const auto* ct = g.find_kv("tokenizer.chat_template");
    check(ct && ie::chatml_think_split(ie::detect_arch(g), ct->as_string()),
          (tag + "its arch + template are a Qwen thinking template (the engine shows the tag)").c_str());
    tok.show_special(close);
    const std::string shown = tok.decode(ids, true, keep);
    const ie::ChatmlThinkSplit s = ie::split_chatml_think(shown);
    check(shown == kM3TRaw && s.reasoning == kM3TReason && s.content == "Mars.",
          (tag + "show_special: the decode keeps </think>; the split gives the reasoning and the answer").c_str());
    const std::vector<int32_t> ids2 = tok.encode("<think>\nR\n</think>\n\n<tool_call>\nx\n</tool_call><|im_end|>", true);
    check(tok.decode(ids2, true, keep) == "\nR\n</think>\n\n<tool_call>\nx\n</tool_call>",
          (tag + "show_special: <think> and <|im_end|> stay hidden, the tool-call markers kept as before").c_str());
    // GenerateResult::answer_start = the decode of the reply up to and including the tag: a byte prefix of the reply's
    const size_t k = size_t(std::find(ids.begin(), ids.end(), close) - ids.begin());
    const std::string pre = tok.decode(std::span<const int32_t>(ids).first(k + 1), true, keep);
    check(pre == kM3TReason + "\n</think>" && shown.compare(0, pre.size(), pre) == 0,
          (tag + "answer_start: the decoded prefix up to the tag is a byte prefix of the reply").c_str());
}

void think_split() {
    {
        const ie::ChatmlThinkSplit s = ie::split_chatml_think(kM3TRaw);
        check(s.reasoning == kM3TReason && s.content == "Mars.",
              "split: the smoke reply -> the reasoning before </think>, the answer after it, the template's newlines dropped");
    }
    {
        const ie::ChatmlThinkSplit s = ie::split_chatml_think("\n\nfirst\n\nsecond\n\n</think>\n\n\nanswer\n\nmore\n");
        check(s.reasoning == "first\n\nsecond" && s.content == "answer\n\nmore\n",
              "split: '\\n's go only at the boundary (around the reasoning, before the answer)");
        const ie::ChatmlThinkSplit t = ie::split_chatml_think(" R \n</think>\n\n A");
        check(t.reasoning == " R " && t.content == " A", "split: only '\\n' is stripped (the template's rstrip / lstrip of '\\n')");
    }
    {
        const ie::ChatmlThinkSplit s = ie::split_chatml_think("Let me think about this.\n\nStep one: the");
        const ie::ChatmlThinkSplit t = ie::split_chatml_think("Cut off here\n\n");
        check(s.reasoning == "Let me think about this.\n\nStep one: the" && s.content.empty() &&
              t.reasoning == "Cut off here" && t.content.empty(), "split: no </think> (cut off by max_tokens) -> all reasoning, no content");
    }
    {
        const ie::ChatmlThinkSplit a = ie::split_chatml_think("\n</think>\n\nMars.");
        const ie::ChatmlThinkSplit b = ie::split_chatml_think("R\n</think>\n\n");
        const ie::ChatmlThinkSplit c = ie::split_chatml_think("");
        check(a.reasoning.empty() && a.content == "Mars." && b.reasoning == "R" && b.content.empty() && c.reasoning.empty() &&
              c.content.empty(), "split: an empty reasoning, an empty answer, an empty reply");
        const ie::ChatmlThinkSplit d = ie::split_chatml_think("R\n</think>\n\nThe tag </think> ends it.");
        check(d.reasoning == "R" && d.content == "The tag </think> ends it.", "split: the first </think> splits; a later one is content");
    }
}

// Engine::chat's ChatML reply after generate(), as it runs it (finish_chatml_reply), then the non-stream body.
ie::GenerateResult finish(const std::string& text, size_t answer_start, const char* finish_reason, const char* tools, bool xml,
                          bool think_open, bool think_split) {
    ie::GenerateResult r;
    r.text = text; r.answer_start = answer_start; r.finish_reason = finish_reason;
    ie::finish_chatml_reply(r, tools, xml, think_open, think_split);
    return r;
}
json choice_of(const ie::GenerateResult& r) { return json::parse(ie::oai::chat_completion_json("m", r, "id1", 7))["choices"][0]; }

void think_reply() {
    const size_t npos = std::string::npos;
    const std::string get_weather = "<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>";
    const std::string read_file = "<tool_call>\n<function=read_file>\n<parameter=path>\nx.py\n</parameter>\n</function>\n</tool_call>";
    // 1. thinking on, no tools (smoke M3T): answer_start = after the tag
    {
        const ie::GenerateResult r = finish(kM3TRaw, kM3TReason.size() + 9, "stop", "", false, true, true);
        const json c = choice_of(r);
        check(r.reasoning_content == kM3TReason && r.text == "Mars." && r.tool_calls_json.empty() &&
              c["message"]["reasoning_content"] == kM3TReason && c["message"]["content"] == "Mars." && c["finish_reason"] == "stop",
              "reply: thinking on -> reasoning_content = the reasoning, content = the answer (the non-stream body)");
    }
    // 2. thinking off: the prompt pre-filled an empty think block; the reply is content, byte for byte
    {
        const std::string t = "The capital of Australia is Canberra.";
        const ie::GenerateResult r = finish(t, npos, "stop", "", false, false, false);
        const json c = choice_of(r);
        check(r.text == t && r.reasoning_content.empty() && !c["message"].contains("reasoning_content") && c["message"]["content"] == t,
              "reply: thinking off -> everything is content, unchanged");
    }
    // 3. cut off by max_tokens inside the reasoning (no </think>): all of it is reasoning, the content "" (documented rule)
    {
        const std::string t = "We need to list every pump at site C. Log 00003 says";
        const ie::GenerateResult r = finish(t, npos, "length", "", false, true, true);
        const json c = choice_of(r);
        check(r.reasoning_content == t && r.text.empty() && c["message"]["reasoning_content"] == t && c["message"]["content"] == "" &&
              c["finish_reason"] == "length", "reply: unclosed reasoning cut off by length -> all reasoning_content, content \"\"");
    }
    // 4. a call after </think> (smoke one-0): B27's scan from answer_start; the reasoning split off; null content
    {
        const std::string R = "We need answer user asks weather Oslo next 3 days metric. Need call get_weather.";
        const ie::GenerateResult r = finish(R + "\n</think>\n\n" + get_weather, R.size() + 9, "stop", kTools, true, true, true);
        const json c = choice_of(r);
        check(r.reasoning_content == R && r.text.empty() && json::parse(r.tool_calls_json).size() == 1 &&
              c["finish_reason"] == "tool_calls" && c["message"]["content"].is_null() && c["message"]["reasoning_content"] == R &&
              c["message"]["tool_calls"][0]["function"]["name"] == "get_weather" &&
              json::parse(c["message"]["tool_calls"][0]["function"]["arguments"].get<std::string>())["city"] == "Oslo",
              "reply: a call after </think> -> one structured call, the reasoning in reasoning_content, null content");
    }
    // 5. a block quoted while reasoning + a call after </think> (B27 F1): the quote stays in the reasoning, verbatim
    {
        const std::string R = "The user pasted this:\n" + get_weather + "\nIt is markup. Let me read the file.";
        const ie::GenerateResult r = finish(R + "\n</think>\n\nReading it now.\n" + read_file, R.size() + 9, "stop", kTools, true, true, true);
        check(r.reasoning_content == R && r.text == "Reading it now.\n" && json::parse(r.tool_calls_json).size() == 1 &&
              json::parse(r.tool_calls_json)[0]["function"]["name"] == "read_file",
              "reply: a block quoted in the reasoning stays there; the call after </think> is the call; the prose before it is content");
    }
    // 6. B28: no </think>, finish stop, the reply ends in a call: the call stays structured; the text before it -- a
    //    reasoning that never closed -- is reasoning_content (B28 left it in content: there was no split)
    {
        const std::string live = "The user wants me to fire off 15 explorer subagents at once. I've done 4 so far.\n\n\nFour done. "
                                 "Continuing with more \xe2\x80\x94 batching these:\n\n";
        const std::string task = "<tool_call>\n<function=task>\n<parameter=prompt>\nRead it.\n</parameter>\n<parameter=subagent_type>\n"
                                 "explorer\n</parameter>\n</function>\n</tool_call>";
        const ie::GenerateResult r = finish(live + task, npos, "stop", kTools, true, true, true);
        const json c = choice_of(r);
        check(json::parse(r.tool_calls_json).size() == 1 && json::parse(r.tool_calls_json)[0]["function"]["name"] == "task" &&
              r.reasoning_content == live.substr(0, live.size() - 2) && r.text.empty() && c["finish_reason"] == "tool_calls" &&
              c["message"]["content"].is_null(), "reply: B28 tail call without </think> -> still the call; the reasoning in reasoning_content");
    }
    // 7. B28's length rule: no </think>, finish length, a cut-off block -> no call ("[]", no server fallback), all reasoning
    {
        const std::string t = "Let me read it.\n<tool_call>\n<function=read_file>\n<parameter=path>\nx.p";
        const ie::GenerateResult r = finish(t, npos, "length", kTools, true, true, true);
        const json c = choice_of(r);
        check(r.tool_calls_json == "[]" && r.reasoning_content == t && r.text.empty() && !c["message"].contains("tool_calls") &&
              c["finish_reason"] == "length", "reply: length-capped without </think> -> no call, all reasoning");
    }
    // 8. thinking off with tools: B27's whole-reply scan (answer_start 0), no split -- unchanged
    {
        const ie::GenerateResult r = finish("Reading.\n" + read_file, npos, "stop", kTools, true, false, false);
        check(json::parse(r.tool_calls_json).size() == 1 && r.text == "Reading.\n" && r.reasoning_content.empty(),
              "reply: thinking off with tools -> the whole reply scanned, the prose content, no reasoning_content");
    }
    // 9. a <think> template the engine does not split (think_split off): B27's scan from answer_start, the text whole
    {
        const std::string R = "Quoted: " + read_file + "\nok\n";
        const ie::GenerateResult r = finish(R + "\n\nNo call.", R.size(), "stop", kTools, true, true, false);
        check(r.tool_calls_json == "[]" && r.text == R + "\n\nNo call." && r.reasoning_content.empty(),
              "reply: thinking on without the split -> the pre-B35 result (the scan from answer_start, the text whole)");
    }
    // 10. the JSON preamble (no XML form; IE_QWEN_TOOLS_JSON=1): no engine parse, the split still runs, so the server's own
    //     parser sees the answer only and a JSON block quoted while reasoning does not become a call
    {
        const std::string R = "They pasted <tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"q.py\"}}\n</tool_call> as JSON.";
        const ie::GenerateResult r = finish(R + "\n</think>\n\nIt is a JSON tool call.", R.size() + 9, "stop", kTools, false, true, true);
        const json c = choice_of(r);
        check(r.tool_calls_json.empty() && r.reasoning_content == R && r.text == "It is a JSON tool call." &&
              !c["message"].contains("tool_calls") && c["message"]["content"] == "It is a JSON tool call.",
              "reply: JSON preamble -> the quoted block stays in the reasoning, no call from it");
    }
}

// The server's stream of a Qwen thinking reply (openai_server.cpp, P4 B35: the in_reason branch, the content's start;
// B27's content phase: a <tool_call> held back; the end: the rest of an unclosed reasoning, the calls frame and the
// content's remainder, or the held-back tail), mirrored without the sink, stops and vitals. The reply arrives in
// fragments ending at `cuts`; `r` is Engine::chat's result for it.
struct Streamed { std::string reasoning, content; size_t deltas = 0; bool calls = false; };
Streamed stream_reply(const std::string& text, const std::vector<size_t>& cuts, const ie::GenerateResult& r) {
    ie::ChatmlThinkStream ts;
    Streamed o;
    static const std::string OPEN = "<tool_call>";
    const size_t hold = OPEN.size();
    size_t reason_streamed = 0, content_start = 0, streamed = 0;
    bool in_reason = true, in_tool = false;
    std::string acc;
    auto flush_to = [&](size_t upto) { if (upto > streamed) { o.content += acc.substr(streamed, upto - streamed); streamed = upto; } };
    auto fragment = [&](size_t end) {
        acc = text.substr(0, end);
        if (in_reason) {
            ts.update(acc);
            reason_streamed = std::max(reason_streamed, ts.reason_begin());
            if (ts.reason_end() > reason_streamed) {
                o.reasoning += acc.substr(reason_streamed, ts.reason_end() - reason_streamed);
                ++o.deltas;
                reason_streamed = ts.reason_end();
            }
            if (!ts.closed()) return;
            in_reason = false;
            content_start = streamed = ts.content_begin();
        } else if (streamed == content_start) {
            ts.update(acc);
            content_start = streamed = ts.content_begin();
        }
        if (in_tool) return;
        if (const size_t tc = acc.find(OPEN, content_start); tc != std::string::npos) { flush_to(tc); in_tool = true; return; }
        flush_to(acc.size() > hold ? acc.size() - hold : 0);
    };
    for (size_t c : cuts) fragment(c);
    fragment(text.size());
    if (in_reason) o.reasoning += ts.rest_of(r.reasoning_content, reason_streamed);
    const std::string parse_text = in_reason ? std::string() : acc.substr(content_start);
    const std::string tcframe = !r.tool_calls_json.empty() ? ie::oai::chat_chunk_sse_tool_calls_json("m", "id1", 7, r.tool_calls_json)
                                                           : ie::oai::chat_chunk_sse_tool_calls("m", "id1", 7, parse_text);
    if (!tcframe.empty()) {
        if (!r.tool_calls_json.empty() && r.text.size() > streamed - content_start) o.content += r.text.substr(streamed - content_start);
        o.calls = true;
    } else if (!in_reason) {
        flush_to(acc.size());
    }
    return o;
}

void think_stream() {
    const std::string quote = "<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>";
    const std::string task = "<tool_call>\n<function=task>\n<parameter=prompt>\nRead it.\n</parameter>\n<parameter=subagent_type>\n"
                             "explorer\n</parameter>\n</function>\n</tool_call>";
    struct Case { const char* name; std::string text; size_t answer_start; const char* finish; bool tools; };
    const size_t npos = std::string::npos;
    const std::vector<Case> cases = {
        {"the smoke reply", kM3TRaw, kM3TReason.size() + 9, "stop", false},
        {"newlines at the boundary", "\n\nfirst\n\nsecond\n\n</think>\n\n\nanswer\n\nmore\n", 25, "stop", false},
        {"unclosed (length)", "Let me think.\n\nStep one\n", npos, "length", false},
        {"empty reasoning", "\n</think>\n\nMars.", 9, "stop", false},
        {"empty answer", "R\n</think>\n\n", 10, "stop", false},
        {"a quoted block, then </think>", "The user pasted:\n" + quote + "\nIt is markup.\n</think>\n\nThat is an XML tool call.",
         17 + quote.size() + 23, "stop", false},
        {"a later </think> in the answer", "R\n</think>\n\nThe tag </think> ends it.", 10, "stop", false},
        {"a quoted block in an unclosed reasoning (length)", "Pasted:\n" + quote + "\nThinking more", npos, "length", true},
        {"B28: a tail call, no </think>", "I have done 4.\n\n\nFour done. Continuing:\n\n" + task, npos, "stop", true},
        {"a call after </think>", "Need the weather.\n</think>\n\nLet me check.\n" + quote, 26, "stop", true},
        {"a call right after </think>, prose after it", "Need the weather.\n</think>\n\n" + quote + "\nDone.", 26, "stop", true},
        {"prose, a call, prose", "Need it.\n</think>\n\n\nOK\n" + quote + "\nDone.", 17, "stop", true},
        {"a rejected block in the answer",
         "Explain.\n</think>\n\nThe format:\n<tool_call>\n<function=example_fn>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\nThat's it.",
         17, "stop", true},
    };
    for (const Case& c : cases) {
        // what Engine::chat returns for this reply (finish_chatml_reply: the calls taken out, the split)
        const ie::GenerateResult r = finish(c.text, c.answer_start, c.finish, c.tools ? kTools : "", c.tools, true, true);
        const bool engine_calls = !r.tool_calls_json.empty() && r.tool_calls_json != "[]";
        int bad = 0;
        size_t deltas_char = 0;
        std::vector<std::vector<size_t>> splits;
        std::vector<size_t> each;
        for (size_t i = 1; i < c.text.size(); ++i) each.push_back(i);
        splits.push_back(each);                                       // one byte at a time
        for (size_t i = 0; i <= c.text.size(); ++i) splits.push_back({i});   // every two-piece cut
        splits.push_back({});                                         // the whole reply at once
        for (const auto& cuts : splits) {
            const Streamed o = stream_reply(c.text, cuts, r);
            if (&cuts == &splits[0]) deltas_char = o.deltas;
            if (o.reasoning != r.reasoning_content || o.content != r.text || o.calls != engine_calls) {
                if (bad++ == 0) std::printf("    %s, %zu cuts: reasoning %s content %s calls %d\n", c.name, cuts.size(),
                                            json(o.reasoning).dump().c_str(), json(o.content).dump().c_str(), int(o.calls));
            }
        }
        check(bad == 0, ("stream == non-stream (reasoning, content, calls) for every fragmentation: " + std::string(c.name)).c_str());
        if (c.text.find("\nStep") != std::string::npos || c.answer_start == kM3TReason.size() + 9)
            check(deltas_char > 1, ("stream: the reasoning goes out as it comes, not at the end: " + std::string(c.name)).c_str());
    }
    // the hold-back while the reasoning is open: a <tool_call> and what follows wait for a </think> (or the end)
    {
        const std::string head = "The user pasted:\n";
        ie::ChatmlThinkStream ts;
        ts.update(head + quote + "\nIt is markup, I will answer in prose and");
        check(!ts.closed() && ts.reason_begin() == 0 && ts.reason_end() == head.size() - 1,
              "stream: an open reasoning is sent up to its first <tool_call> (its trailing '\\n' held)");
        ts.update(head + quote + "\nIt is markup, I will answer in prose and\n</think>");
        check(ts.closed() && ts.reason_end() == head.size() + quote.size() + 41 && ts.content_begin() == head.size() + quote.size() + 50,
              "stream: the </think> releases the held block as reasoning; the content starts after the tag");
        ts.update(head + quote + "\nIt is markup, I will answer in prose and\n</think>\n\n");
        check(ts.content_begin() == head.size() + quote.size() + 52, "stream: the content's leading '\\n's are skipped as they arrive");
        ts.update(head + quote + "\nIt is markup, I will answer in prose and\n</think>\n\nIt\n");
        check(ts.content_begin() == head.size() + quote.size() + 52, "stream: ...up to the first other byte");
    }
}

// Which models split: Engine::load shows their </think> and Engine::chat / the server split there.
void think_template(const char* golden_path) {
    std::ifstream f(golden_path);
    if (!f) { check(false, "think_template: golden file not readable"); return; }
    const json g = json::parse(f);
    using A = ie::ModelArch;
    const std::string t35 = g["templates"]["35b"].get<std::string>(), t27 = g["templates"]["27b"].get<std::string>();
    check(ie::chatml_think_split(A::kQwen35Moe, t35) && ie::chatml_think_split(A::kQwen35Dense, t27) &&
          ie::chatml_think_split(A::kQwen4Exp, t27), "predicate: the 35B, the 27B and Flash-Next (the 27B's template) split");
    const std::string chatml = "{{- '<|im_start|>assistant\\n<think>\\n' }}";
    check(ie::chatml_think_split(A::kQwen3Moe, chatml) && ie::chatml_think_split(A::kQwen3Next, chatml) &&
          ie::chatml_think_split(A::kQwen3Dense, chatml), "predicate: the other Qwen ChatML thinking templates split");
    check(!ie::chatml_think_split(A::kQwen35Moe, "{{- '<|im_start|>assistant\\n' }}"),
          "predicate: a template without <think> (Qwen3-Coder) -> no");
    check(!ie::chatml_think_split(A::kQwen3Dense, "<\xef\xbd\x9c" "Assistant\xef\xbd\x9c><think>\\n"),
          "predicate: R1-Distill-Qwen (its own sentinel template path) -> no");
    bool none = true;
    for (A a : {A::kDeepSeek4, A::kDeepSeek41, A::kMimo26, A::kGlm5Next, A::kGemma4, A::kGptOss, A::kLlama3, A::kUnknown})
        none = none && !ie::chatml_think_split(a, t35);
    check(none, "predicate: the archs with their own chat paths (DeepSeek-V4 / V4.1, MiMo, GLM, Gemma, gpt-oss, Llama) -> no");
}

}  // namespace

int main(int argc, char** argv) {
    std::string golden_path = "tests/data/chatml_xml_tools_golden.json";
    std::vector<std::string> ggufs;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--gguf" && i + 1 < argc) ggufs.push_back(argv[++i]);
        else if (a == "--help") { std::printf("usage: chatml_xml_tools_test [golden.json] [--gguf PATH]...\n"); return 0; }
        else golden_path = a;
    }
    std::printf("renderer vs the GGUF templates:\n");
    golden(golden_path.c_str());
    std::printf("parser:\n");
    parser();
    std::printf("B35 the reasoning split:\n");
    think_split();
    think_reply();
    think_stream();
    think_template(golden_path.c_str());
    if (!ggufs.empty()) std::printf("B35 real tokenizers:\n");
    for (const auto& p : ggufs) gguf_think(p);
    std::printf("%s (%d failures)\n", fails == 0 ? "ALL PASS" : "FAILURES", fails);
    return fails == 0 ? 0 : 1;
}
