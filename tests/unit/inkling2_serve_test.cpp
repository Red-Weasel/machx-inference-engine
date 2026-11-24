// tests/unit/inkling2_serve_test.cpp — Inkling 2 Mini's serving layer, host-only (no GPU, no weights):
//  1. the native chat template (inkling2_render_prompt) against the checkpoint's Jinja template rendered by transformers
//     (scripts/inkling2/gen_template_golden.py -> tests/data/inkling2_template_golden.json), byte for byte, with the
//     conversations taken through the server's own request parser (oai::parse_chat_request) as a client sends them;
//     and, when the model's tokenizer.json is on disk, the engine's token ids of each render against the reference's;
//  2. the request-text ranges (known_bugs 45): a marker spelled inside a request's text is text;
//  3. the effort table and the renderer's refusals;
//  4. the reply parser (Inkling2ReplyParser) on event sequences: thinking / text / tool calls, several blocks, cuts in
//     the middle of a block (max_tokens), thinking off, unparseable calls, the stream it produces.
#include "ie/inkling2_engine.hpp"
#include "ie/openai_proto.hpp"
#include "ie/reasoning.hpp"
#include "ie/serve_options.hpp"
#include "ie/server_capabilities.hpp"
#include "nlohmann/json.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

using nlohmann::json;
using ie::Ink2Mark;

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++g_fail; } } while (0)

static std::vector<ie::ChatTurn> turns_of(const json& c, std::string& tools) {
    json body = {{"messages", c["messages"]}};
    if (c.contains("tools")) body["tools"] = c["tools"];
    const auto cr = ie::oai::parse_chat_request(body.dump());
    if (!cr.error.empty()) { std::printf("  request parse error: %s\n", cr.error.c_str()); ++g_fail; }
    tools = cr.tools_json;
    return cr.turns;
}

static void golden(const char* model_dir) {
    std::ifstream f;
    for (const char* p : {"tests/data/inkling2_template_golden.json", "../tests/data/inkling2_template_golden.json",
                          "../../tests/data/inkling2_template_golden.json"}) {
        f.open(p);
        if (f) break;
        f.clear();
    }
    if (!f) { std::printf("  tests/data/inkling2_template_golden.json not found (run from the source root)\n"); ++g_fail; return; }
    const json cases = json::parse(f);
    ie::Tokenizer tok;
    bool have_tok = false;
    if (std::ifstream(std::string(model_dir) + "/tokenizer.json")) {
        const auto e = tok.load_from_hf_json(std::string(model_dir) + "/tokenizer.json", std::string(model_dir) + "/tokenizer_config.json");
        have_tok = e.empty();
        if (!have_tok) { std::printf("  tokenizer: %s\n", e.c_str()); ++g_fail; }
    } else std::printf("  (no %s/tokenizer.json: token ids not checked)\n", model_dir);
    size_t ids_ok = 0;
    for (const auto& c : cases) {
        std::string tools;
        const auto turns = turns_of(c, tools);
        const std::string effort = c["effort"].is_null() ? std::string() : c["effort"].get<std::string>();
        std::string err;
        std::vector<std::pair<size_t, size_t>> ranges;
        const std::string got = ie::inkling2_render_prompt(turns, tools, c["thinking"].get<bool>(), effort, err, &ranges);
        const std::string want = c["prompt"].get<std::string>();
        const bool ok = err.empty() && got == want;
        bool ids_same = true;
        if (have_tok && ok) {
            const auto ids = tok.encode(got, true, ranges);
            ids_same = ids == c["ids"].get<std::vector<int32_t>>();
            ids_ok += ids_same;
        }
        std::printf("  %-48s %s%s\n", c["name"].get<std::string>().c_str(), ok ? "OK" : "FAIL", have_tok && ok ? (ids_same ? ", ids OK" : ", ids DIFFER") : "");
        if (!ok || !ids_same) ++g_fail;
        if (!ok) {
            size_t d = 0;
            while (d < got.size() && d < want.size() && got[d] == want[d]) ++d;
            std::printf("    err '%s'; first difference at byte %zu\n    got : %s\n    want: %s\n", err.c_str(), d,
                        got.substr(d > 60 ? d - 60 : 0, 200).c_str(), want.substr(d > 60 ? d - 60 : 0, 200).c_str());
        }
    }
    std::printf("  template: %zu cases%s\n", cases.size(), have_tok ? (", token ids equal on " + std::to_string(ids_ok)).c_str() : "");
    // the generation-prompt switch: the same text plus the opener
    {
        std::vector<ie::ChatTurn> t(1); t[0].role = "user"; t[0].content = "Hi";
        std::string e1, e2;
        const auto a = ie::inkling2_render_prompt(t, "", true, "", e1), b = ie::inkling2_render_prompt(t, "", true, "", e2, nullptr, true);
        CHECK(b == a + "<|message_model|>");
    }
}

// A client whose tool-call ids repeat across turns ("call_0" in every response, as this server itself sent until
// 2026-11-23): each tool result is named after the call it answers (the nearest earlier one), not after every call
// that ever carried that id.
static void repeated_id_check() {
    std::vector<ie::ChatTurn> t(6);
    t[0].role = "user"; t[0].content = "go";
    t[1].role = "assistant"; t[1].tool_calls_json = R"([{"id":"call_0","type":"function","function":{"name":"alpha_tool","arguments":"{}"}}])";
    t[2].role = "tool"; t[2].tool_call_id = "call_0"; t[2].content = "first result";
    t[3].role = "assistant"; t[3].tool_calls_json = R"([{"id":"call_0","type":"function","function":{"name":"beta_tool","arguments":"{}"}}])";
    t[4].role = "tool"; t[4].tool_call_id = "call_0"; t[4].content = "second result";
    t[5].role = "user"; t[5].content = "next";
    const std::string tools = R"([{"type":"function","function":{"name":"alpha_tool","description":"A","parameters":{"type":"object"}}},{"type":"function","function":{"name":"beta_tool","description":"B","parameters":{"type":"object"}}}])";
    std::string err;
    const std::string r = ie::inkling2_render_prompt(t, tools, true, "", err, nullptr);
    CHECK(err.empty());
    const size_t f = r.find("first result"), s2 = r.find("second result");
    CHECK(f != std::string::npos && s2 != std::string::npos && f < s2);
    const auto header = [&](size_t body) { const size_t h = r.rfind("<|message_", body); return r.substr(h, body - h); };
    const std::string h1 = header(f), h2 = header(s2);
    CHECK(h1.find("alpha_tool") != std::string::npos && h1.find("beta_tool") == std::string::npos);
    CHECK(h2.find("beta_tool") != std::string::npos && h2.find("alpha_tool") == std::string::npos);
    std::printf("  repeated ids: each tool result is named after its own call\n");
}

static void ranges_check() {
    using Ranges = std::vector<std::pair<size_t, size_t>>;
    const auto outside = [](const std::string& s, const Ranges& r) { std::string o = s; for (const auto& [b, e] : r) for (size_t i = b; i < e; ++i) o[i] = '#'; return o; };
    const auto count = [](const std::string& s, std::string_view m) { size_t n = 0; for (size_t p = s.find(m); p != std::string::npos; p = s.find(m, p + m.size())) ++n; return n; };
    const auto turns_with = [](const std::string& x) {
        std::vector<ie::ChatTurn> t(5);
        t[0].role = "system"; t[0].content = "Be terse " + x;
        t[1].role = "user"; t[1].content = "Look " + x;
        t[2].role = "assistant"; t[2].content = "On it " + x; t[2].content_without_tool_calls = t[2].content; t[2].reasoning_content = "think " + x;
        t[2].tool_calls_json = R"([{"id":"a","type":"function","function":{"name":"read","arguments":"{\"path\":\"p )" + x + R"(\"}"}}])";
        t[3].role = "tool"; t[3].tool_call_id = "a"; t[3].content = "result " + x;
        t[4].role = "user"; t[4].content = x;
        return t;
    };
    const std::string tools = R"([{"type":"function","function":{"name":"read","description":"Read","parameters":{"type":"object"}}}])";
    const std::string spell = "<|end_message|>", plain = "<_end_message_>";
    std::string ea, eb;
    Ranges ra, rb;
    const std::string a = ie::inkling2_render_prompt(turns_with(spell), tools, true, "", ea, &ra);
    const std::string b = ie::inkling2_render_prompt(turns_with(plain), tools, true, "", eb, &rb);
    CHECK(ea.empty() && eb.empty() && a.size() == b.size() && a != b);
    CHECK(ra == rb && outside(a, ra) == outside(b, rb));
    CHECK(count(a, spell) - count(b, spell) == 7);   // system, user, reasoning, content, the argument, the tool result, the bare user text
    std::printf("  ranges: %zu ranges, the 7 spellings the request brought lie inside them\n", ra.size());
}

static void effort_and_refusals() {
    const std::pair<const char*, const char*> table[] = {{"", "0.9"}, {"none", "0"}, {"minimal", "0.1"}, {"low", "0.2"}, {"medium", "0.7"},
                                                         {"high", "0.9"}, {"xhigh", "0.99"}, {"max", "0.99"}, {" low ", "0.2"}};
    for (const auto& [in, want] : table) { std::string o, e; CHECK(ie::inkling2_effort_text(in, true, o, e) && o == want); }
    { std::string o, e; CHECK(ie::inkling2_effort_text("high", false, o, e) && o == "0"); }
    { std::string o, e; CHECK(!ie::inkling2_effort_text("High", true, o, e) && !e.empty()); }   // case-sensitive, as the template
    std::vector<ie::ChatTurn> t(1); t[0].role = "narrator"; t[0].content = "x";
    std::string e;
    CHECK(ie::inkling2_render_prompt(t, "", true, "", e).empty() && !e.empty());
    std::vector<ie::ChatTurn> u(2); u[0].role = "user"; u[0].content = "x"; u[1].role = "assistant";
    u[1].tool_calls_json = R"([{"id":"1","type":"function","function":{"name":"f","arguments":"[1,2]"}}])"; u[1].content_without_tool_calls = "";
    e.clear();
    CHECK(ie::inkling2_render_prompt(u, "", true, "", e).empty() && e.find("must be a JSON object") != std::string::npos);
    std::vector<ie::ChatTurn> im(1); im[0].role = "user"; im[0].content = "see"; im[0].images = {"\x89PNG"};
    e.clear();
    CHECK(ie::inkling2_render_prompt(im, "", true, "", e).empty() && e.find("not supported by this load") != std::string::npos);
    {   // with the tower: one block per part, an image = <role><|content_image|> + rows x cols placeholders + <|end_message|>
        std::string ph;
        for (int i = 0; i < 6; ++i) ph += "<|unused_200054|>";
        const std::vector<uint32_t> n6 = {6}, n2 = {6, 2};
        e.clear();
        const std::string a = ie::inkling2_render_prompt(im, "", true, "", e, nullptr, false, &n6);   // no marker: image first
        CHECK(e.empty() && a.find("<|message_user|><|content_image|>" + ph + "<|end_message|><|message_user|><|content_text|>see<|end_message|>") != std::string::npos);
        std::vector<ie::ChatTurn> mk(1); mk[0].role = "user"; mk[0].images = {"x", "y"};
        mk[0].content = "a" + std::string(ie::kChatImageMarker) + "b" + std::string(ie::kChatImageMarker);
        e.clear();
        const std::string c = ie::inkling2_render_prompt(mk, "", true, "", e, nullptr, false, &n2);
        CHECK(e.empty() && c.find("<|message_user|><|content_text|>a<|end_message|><|message_user|><|content_image|>" + ph +
                                  "<|end_message|><|message_user|><|content_text|>b<|end_message|><|message_user|><|content_image|>"
                                  "<|unused_200054|><|unused_200054|><|end_message|>") != std::string::npos);
        std::vector<ie::ChatTurn> sy(1); sy[0].role = "system"; sy[0].images = {"x"};
        e.clear();
        CHECK(ie::inkling2_render_prompt(sy, "", true, "", e, nullptr, false, &n6).empty() && e.find("user and tool") != std::string::npos);
    }
    std::printf("  effort table and refusals: done\n");
}


// "none" / "minimal" now parse for every request (Inkling 2's names); every OTHER arch must refuse them exactly as before:
// the request parser used to throw "unknown reasoning_effort level" (-> "invalid request setting: ..." and a 400), the CLI
// "--reasoning-effort requires a supported model effort level" (exit 2). The five old names take the same path as before.
static void effort_names_other_archs(const char* model_dir) {
    const std::string qwen_tmpl = "<think> reasoning_effort Reasoning effort is set to xhigh.";
    size_t checked = 0;
    for (int a = 0; a <= int(ie::ModelArch::kUnknown); ++a) {
        const auto arch = ie::ModelArch(a);
        for (const std::string& tmpl : {std::string(), qwen_tmpl}) {
            for (const char* e : {"none", "minimal"}) {
                for (bool kw : {false, true}) {
                    json body = {{"messages", json::array({{{"role", "user"}, {"content", "hi"}}})}};
                    if (kw) body["chat_template_kwargs"] = {{"reasoning_effort", e}}; else body["reasoning_effort"] = e;
                    const auto cr = ie::oai::parse_chat_request(body.dump());
                    CHECK(cr.error.empty() && cr.reasoning_effort == e);
                    const std::string got = ie::arch_reasoning_effort_error(arch, tmpl, cr.reasoning_effort);
                    if (arch == ie::ModelArch::kInkling2) CHECK(got.empty());
                    else CHECK(got == "invalid request setting: unknown reasoning_effort level");
                    ++checked;
                }
            }
            for (const char* e : {"", "low", "medium", "high", "xhigh", "max", "bogus"})
                CHECK(ie::arch_reasoning_effort_error(arch, tmpl, e) == ie::reasoning_effort_error(ie::reasoning_capabilities(arch, tmpl), e));
        }
    }
    {   // the request parser still refuses a name nobody knows, as before
        const auto cr = ie::oai::parse_chat_request(R"({"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"bogus"})");
        CHECK(cr.error == "invalid request setting: unknown reasoning_effort level");
    }
    {   // the CLI: the flag parses, the model check refuses it for anything but an Inkling 2 directory
        CHECK(ie::parse_launch_options({"--reasoning-effort", "minimal"}).defaults.reasoning_effort == "minimal");
        CHECK(ie::server_reasoning_error("/nonexistent/model.gguf", "none") == "--reasoning-effort requires a supported model effort level");
        if (std::ifstream(std::string(model_dir) + "/config.json")) CHECK(ie::server_reasoning_error(model_dir, "none").empty());
    }
    std::printf("  effort names none / minimal: %zu (arch, template, name, field) requests -- refused as before on every arch but kInkling2\n", checked);
}

struct Run { std::string stream; ie::Inkling2ReplyParser p; explicit Run(bool th) : p(th) {} };
static Run parse(std::string_view raw, bool thinking, bool complete) {
    Run r(thinking);
    for (const auto& [m, t] : ie::inkling2_events_from_text(raw)) r.stream += r.p.feed(m, t);
    r.p.finish(complete);
    return r;
}

static void parser_check() {
    {   // thinking, then the answer in two text blocks and a tool-error block
        auto r = parse("<|message_model|><|content_thinking|>Think.<|end_message|><|message_model|><|content_text|>Par<|end_message|>"
                       "<|message_model|><|content_text|>is.<|end_message|><|message_model|><|content_tool_error|>!<|end_message|>"
                       "<|content_model_end_sampling|>", true, true);
        CHECK(r.p.reasoning() == "Think." && r.p.content() == "Paris.!" && r.p.tool_calls_json() == "[]");
        CHECK(r.stream == "Think.</think>Paris.!");
    }
    {   // thinking then two calls (one with the name only in the header); arguments keep the model's key order
        auto r = parse("<|message_model|><|content_thinking|>t<|end_message|><|message_model|>get_weather<|content_invoke_tool_json|>"
                       "{\"name\":\"get_weather\",\"args\":{\"days\":2,\"city\":\"Paris\"}}<|end_message|>"
                       "<|message_model|>noop<|content_invoke_tool_json|>{\"args\":{}}<|end_message|><|content_model_end_sampling|>", true, true);
        const json calls = json::parse(r.p.tool_calls_json());
        CHECK(calls.size() == 2 && calls[0]["function"]["name"] == "get_weather" && calls[1]["function"]["name"] == "noop");
        CHECK(calls[0]["function"]["arguments"] == "{\"days\":2,\"city\":\"Paris\"}" && calls[1]["function"]["arguments"] == "{}");
        CHECK(calls[0]["type"] == "function" && calls[0]["id"].is_string() && calls[0]["id"] != calls[1]["id"]);
        CHECK(r.p.content().empty() && r.p.reasoning() == "t");
        CHECK(r.stream == "t</think><tool_call>{\"name\":\"get_weather\",\"arguments\":{\"days\":2,\"city\":\"Paris\"}}</tool_call>"
                          "<tool_call>{\"name\":\"noop\",\"arguments\":{}}</tool_call>");
        // the ids of a second reply differ from the first's (the template names a tool result after every call with its id)
        const json again = json::parse(r.p.tool_calls_json());
        CHECK(again[0]["id"] != calls[0]["id"]);
    }
    {   // cut by max_tokens inside the reasoning
        auto r = parse("<|message_model|><|content_thinking|>Let me th", true, false);
        CHECK(r.p.reasoning() == "Let me th" && r.p.content().empty() && r.p.truncated_call().empty() && r.stream == "Let me th");
    }
    {   // cut inside the answer
        auto r = parse("<|message_model|><|content_thinking|>a<|end_message|><|message_model|><|content_text|>Hal", true, false);
        CHECK(r.p.content() == "Hal" && r.stream == "a</think>Hal");
    }
    {   // cut inside a call: its name, no call, nothing of it in the content or the stream
        auto r = parse("<|message_model|><|content_text|>Reading.<|end_message|><|message_model|>read_file<|content_invoke_tool_json|>"
                       "{\"name\":\"read_file\",\"args\":{\"pa", true, false);
        CHECK(r.p.n_calls() == 0 && r.p.truncated_call() == "read_file" && r.p.content() == "Reading." && r.stream == "</think>Reading.");
        auto q = parse("<|message_model|><|content_invoke_tool_json|>{\"name\":\"grep\",\"args\":{", true, false);   // no header name
        CHECK(q.p.truncated_call() == "grep");
    }
    {   // cut inside a header: nothing
        auto r = parse("<|message_model|>read_fi", false, false);
        CHECK(r.p.content().empty() && r.p.n_calls() == 0 && r.stream.empty());
    }
    {   // thinking off: the reasoning is parsed but not streamed; no </think>
        auto r = parse("<|message_model|><|content_thinking|>hidden<|end_message|><|message_model|><|content_text|>shown<|end_message|>"
                       "<|content_model_end_sampling|>", false, true);
        CHECK(r.p.reasoning() == "hidden" && r.p.content() == "shown" && r.stream == "shown");
    }
    {   // an unparseable call that the model closed stays visible as content
        auto r = parse("<|message_model|>x<|content_invoke_tool_json|>{oops<|end_message|><|content_model_end_sampling|>", false, true);
        CHECK(r.p.n_calls() == 0 && r.p.content() == "{oops" && r.stream == "{oops");
        auto s = parse("<|message_model|>x<|content_invoke_tool_json|>{\"name\":\"x\",\"args\":\"str\"}<|end_message|>", false, true);
        CHECK(s.p.n_calls() == 0 && s.p.content() == "{\"name\":\"x\",\"args\":\"str\"}");
    }
    {   // text that SPELLS a marker but came as text tokens stays text (the engine feeds markers by token id)
        ie::Inkling2ReplyParser p(false);
        std::string s = p.feed(Ink2Mark::kModel) + p.feed(Ink2Mark::kText) + p.feed(Ink2Mark::kNone, "a <|end_message|> b") +
                        p.feed(Ink2Mark::kEnd) + p.feed(Ink2Mark::kOther) + p.feed(Ink2Mark::kEos);
        p.finish(true);
        CHECK(p.content() == "a <|end_message|> b" && s == "a <|end_message|> b");
    }
    {   // the opener appended to the prompt (IE_INKLING2_OPEN_TURN=1): the reply starts at the content kind
        ie::Inkling2ReplyParser p(true);
        std::string s = p.feed(Ink2Mark::kModel);
        for (const auto& [m, t] : ie::inkling2_events_from_text("<|content_thinking|>r<|end_message|><|message_model|><|content_text|>c<|end_message|>")) s += p.feed(m, t);
        p.finish(true);
        CHECK(p.reasoning() == "r" && p.content() == "c" && s == "r</think>c");
    }
    {   // text before any block (not the format; never dropped), and a reply with no thinking while thinking is on
        auto r = parse("plain<|content_model_end_sampling|>", true, true);
        CHECK(r.p.content() == "plain" && r.stream == "</think>plain");
    }
    {   // a call closed by the EOS without <|end_message|> is still a call
        auto r = parse("<|message_model|>f<|content_invoke_tool_json|>{\"name\":\"f\",\"args\":{\"a\":1}}<|content_model_end_sampling|>", true, true);
        CHECK(r.p.n_calls() == 1);
    }
    std::printf("  reply parser: done\n");
}

int main(int argc, char** argv) {
    const std::string default_model = std::string(std::getenv("HOME") ? std::getenv("HOME") : ".") + "/models/Inkling-2-Mini";
    const char* model_dir = argc > 1 ? argv[1] : (std::getenv("IE_INKLING2_MODEL") ? std::getenv("IE_INKLING2_MODEL") : default_model.c_str());
    golden(model_dir);
    ranges_check();
    repeated_id_check();
    effort_and_refusals();
    parser_check();
    effort_names_other_archs(model_dir);
    std::printf("inkling2_serve_test: %s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}
