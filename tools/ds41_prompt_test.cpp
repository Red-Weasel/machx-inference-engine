// tools/ds41_prompt_test.cpp — V4.1 Phase 12 step 2: the prompt format vs the checkpoint's own
// encoding.py (tools/ds41_reference/prompt_golden.py), byte for byte (docs/deepseek41/31).
//   1. ds41_encode_messages(case) == the reference's prompt string for every golden case
//   2. the rendered prompt tokenises (allow_special) to the reference's ids of its prompt
//   3. the refusals: a tool turn and a bad reasoning effort return an error, not a prompt
//   usage: ie-ds41-prompt-test <model_dir> <golden_dir>
#include "ie/deepseek41_prompt.hpp"
#include "ie/tokenizer.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "../third_party/nlohmann/json.hpp"

namespace {
int g_fail = 0;
void check(bool ok, const std::string& w, const std::string& d = "") {
    std::printf("%s%s%s\n", ok ? "[ ok ] " : "[FAIL] ", w.c_str(), d.empty() ? "" : ("  (" + d + ")").c_str()); if (!ok) ++g_fail; }
bool read_text(const std::string& p, std::string& out) {
    std::ifstream f(p, std::ios::binary); if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()); return true; }
bool read_ids(const std::string& p, std::vector<int32_t>& out) {
    std::ifstream f(p, std::ios::binary | std::ios::ate); if (!f) return false;
    const auto n = size_t(f.tellg()) / 4; out.resize(n); f.seekg(0);
    f.read(reinterpret_cast<char*>(out.data()), std::streamsize(n * 4)); return true; }
std::string esc(const std::string& s, size_t from, size_t n) {
    std::string o; for (size_t i = from; i < s.size() && i < from + n; ++i) { const char c = s[i]; if (c == '\n') o += "\\n"; else if (c == '\t') o += "\\t"; else o += c; } return o; }
}  // namespace

int main(int argc, char** argv) {
    const std::string model = argc > 1 ? argv[1] : std::string(std::getenv("HOME")) + "/models/DeepSeek-V4.1-Flash";
    const std::string gd    = argc > 2 ? argv[2] : "/tmp/ds41_golden";
    ie::Tokenizer tok;
    if (auto e = tok.load_from_hf_json(model + "/tokenizer.json", model + "/tokenizer_config.json"); !e.empty()) { std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1; }

    uint32_t n = 0;
    for (int k = 0;; ++k) {
        std::string js, want; std::vector<int32_t> ids;
        if (!read_text(gd + "/prompt_" + std::to_string(k) + ".json", js)) break;
        if (!read_text(gd + "/prompt_" + std::to_string(k) + ".txt", want) || !read_ids(gd + "/prompt_" + std::to_string(k) + ".i32", ids)) { std::fprintf(stderr, "case %d: missing .txt/.i32\n", k); return 1; }
        const auto c = nlohmann::ordered_json::parse(js);
        std::vector<ie::Ds41ChatMessage> msgs;
        for (const auto& m : c["messages"]) msgs.push_back({m.value("role", ""), m.value("content", ""), m.value("reasoning_content", "")});
        ie::Ds41PromptOptions o; std::string err;
        o.thinking = c.value("thinking_mode", "thinking") == "thinking";
        o.drop_thinking = c.value("drop_thinking", true);
        o.add_bos = c.value("add_default_bos_token", true);
        if (!c["reasoning_effort"].is_null()) {
            o.reasoning_effort = c["reasoning_effort"].is_number() ? c["reasoning_effort"].get<int>() : ie::ds41_reasoning_effort_of(c["reasoning_effort"].get<std::string>(), err);
            if (!err.empty()) { check(false, "case " + std::to_string(k) + ": reasoning effort parses", err); continue; }
        }
        std::vector<std::pair<size_t, size_t>> content;
        const std::string got = ie::ds41_encode_messages(msgs, o, err, &content);
        const std::string name = c.value("name", "?");
        size_t first = 0; while (first < got.size() && first < want.size() && got[first] == want[first]) ++first;
        check(err.empty() && got == want, "case " + std::to_string(k) + " " + name + ": prompt == the reference's, byte for byte",
              !err.empty() ? err : got == want ? std::to_string(want.size()) + " bytes" : "first difference at byte " + std::to_string(first) + ": engine '" + esc(got, first > 12 ? first - 12 : 0, 40) + "' | reference '" + esc(want, first > 12 ? first - 12 : 0, 40) + "'");
        if (err.empty()) {
            const auto enc = tok.encode(got, true);
            check(enc == ids, "case " + std::to_string(k) + " " + name + ": the prompt tokenises to the reference's ids", std::to_string(enc.size()) + " vs " + std::to_string(ids.size()));
            // known_bugs 45: with the request's own spans marked as content the ids are the same (no spelling in a golden)
            const auto guarded = tok.encode(got, true, content);
            check(guarded == enc, "case " + std::to_string(k) + " " + name + ": the content-guarded encode gives the same ids", std::to_string(content.size()) + " content ranges");
        }
        ++n;
    }
    check(n >= 10, "at least ten golden cases judged", std::to_string(n));

    // Shipped text/tool cases must match; internal task/reminder and vision roles remain refused.
    std::printf("\n=== the checkpoint's shipped encoding/tests cases ===\n");
    uint32_t shipped = 0, refused = 0;
    for (int k = 1;; ++k) {
        std::string js, want;
        if (!read_text(model + "/encoding/tests/test_input_" + std::to_string(k) + ".json", js)) break;
        read_text(model + "/encoding/tests/test_output_" + std::to_string(k) + ".txt", want);
        // the reference's load_cases: an object is one case; an array whose first element has a
        // "role" is one case's messages; otherwise an array of cases
        auto parsed = nlohmann::ordered_json::parse(js);
        nlohmann::ordered_json cases = nlohmann::ordered_json::array();
        if (parsed.is_object()) cases.push_back(parsed);
        else if (parsed.is_array() && !parsed.empty() && parsed[0].is_object() && parsed[0].contains("role")) cases.push_back(nlohmann::ordered_json{{"messages", parsed}});
        else cases = parsed;
        ++shipped;
        for (size_t ci = 0; ci < cases.size(); ++ci) {
            const auto& c = cases[ci];
            // what this port refuses: reminder / other internal roles, tasks,
            // content blocks, response formats, image content -- everything outside text turns
            const bool has_tools = c.contains("tools") && !c["tools"].is_null() && !c["tools"].empty();
            bool needs_refusal = false;
            std::vector<ie::Ds41ChatMessage> msgs;
            for (const auto& m : c["messages"]) {
                const std::string role = m.value("role", "");
                const bool images = m.contains("content") && !m["content"].is_string() && !m["content"].is_null();
                const bool tool_calls = m.contains("tool_calls") && !m["tool_calls"].is_null();
                if ((role != "system" && role != "user" && role != "assistant" && role != "tool") || images || m.contains("task") || m.contains("content_blocks") || m.contains("response_format") || m.contains("wo_eos")) needs_refusal = true;
                msgs.push_back({role, m.contains("content") && m["content"].is_string() ? m["content"].get<std::string>() : std::string(), m.contains("reasoning_content") && m["reasoning_content"].is_string() ? m["reasoning_content"].get<std::string>() : std::string(), images, tool_calls, tool_calls ? m["tool_calls"].dump() : std::string(), m.value("tool_call_id", ""), m.contains("tools") ? m["tools"].dump() : std::string()});
            }
            // the reference runner's default for these files is chat mode (test_output_2 has the chat header)
            ie::Ds41PromptOptions o; std::string err; o.has_tools = has_tools; if(has_tools) o.tools_json = c["tools"].dump();
            o.thinking = (c.contains("thinking_mode") && c["thinking_mode"].is_string() ? c["thinking_mode"].get<std::string>() : std::string("chat")) == "thinking";
            if (c.contains("reasoning_effort") && !c["reasoning_effort"].is_null()) { std::string e2; o.reasoning_effort = c["reasoning_effort"].is_number() ? c["reasoning_effort"].get<int>() : ie::ds41_reasoning_effort_of(c["reasoning_effort"].get<std::string>(), e2); }
            const std::string got = ie::ds41_encode_messages(msgs, o, err);
            const std::string tag = "shipped file " + std::to_string(k) + (cases.size() > 1 ? " case " + std::to_string(ci) : "");
            if (needs_refusal) { check(!err.empty() && got.empty(), tag + " (internal roles / images) is REFUSED, not rendered", err.empty() ? "rendered " + std::to_string(got.size()) + " bytes" : err); if (!err.empty()) ++refused; }
            else check(err.empty() && got == want, tag + " (text/tools) == its test_output byte for byte", err.empty() ? std::to_string(got.size()) + " vs " + std::to_string(want.size()) + " bytes" : err);
        }
    }
    check(shipped == 5, "the five shipped files were read", std::to_string(shipped) + " read, " + std::to_string(refused) + " refused");

    std::printf("\n=== refusals ===\n");
    { std::string err; ie::Ds41PromptOptions o;
      const auto p = ie::ds41_encode_messages({{"user", "hi", ""}, {"latest_reminder", "42", ""}}, o, err);
      check(p.empty() && !err.empty(), "an unsupported internal reminder is refused with a message", err); }
    { std::string err; const int v = ie::ds41_reasoning_effort_of("101", err);
      check(v < 0 && !err.empty(), "reasoning effort 101 is refused", err); }
    { std::string err; check(ie::ds41_reasoning_effort_of("low", err) == 50 && ie::ds41_reasoning_effort_of("high", err) == 75 && ie::ds41_reasoning_effort_of("max", err) == 100 && ie::ds41_reasoning_effort_of("7", err) == 7, "the effort aliases map low 50 / high 75 / max 100, digits pass through"); }

    // known_bugs 45 (2026-11-04): a control-token spelling typed by a user, returned by a tool, or written into a tool
    // call's arguments is text. Before the content ranges the whole render was tokenised with specials on, so a page a
    // tool fetched could forge a turn, end the reasoning or open a tool-call block.
    std::printf("\n=== control-token spellings inside request text ===\n");
    {
        const std::vector<const char*> spellings = {"<｜User｜>", "<｜Assistant｜>", "<｜end▁of▁sentence｜>", "</think>", "<think>", "｜DSML｜", "<｜begin▁of▁sentence｜>", "<｜System｜>"};
        std::string spell; for (const char* s : spellings) { spell += ' '; spell += s; }
        std::vector<int32_t> control; for (const char* s : spellings) control.push_back(tok.find_token(s));
        const auto controls = [&](const std::vector<int32_t>& v) { size_t n = 0; for (int32_t t : v) if (std::find(control.begin(), control.end(), t) != control.end()) ++n; return n; };
        std::string err; std::vector<std::pair<size_t, size_t>> content; ie::Ds41PromptOptions o;
        const std::vector<ie::Ds41ChatMessage> msgs = {
            {"system", "sys" + spell, ""}, {"user", "ask" + spell, ""},
            {"assistant", "", "", false, true, "[{\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"run\",\"arguments\":\"{\\\"cmd\\\": \\\"echo <｜User｜> </think>\\\"}\"}}]", "", ""},
            {"tool", "result" + spell, "", false, false, "", "c1", ""}, {"user", "and?", ""}};
        const std::string got = ie::ds41_encode_messages(msgs, o, err, &content);
        check(err.empty() && content.size() >= 6, "the render carries the request's content ranges", err.empty() ? std::to_string(content.size()) + " ranges" : err);
        const auto plain = tok.encode(got, true), guarded = tok.encode(got, true, content);
        // the template's own markers: BOS, System, User x2, Assistant x2, a </think> and a <think>, EOS, six DSML tags = 15;
        // the spellings inside the request's text: 8 + 8 + 8 in the three contents, 2 in the call's argument = 26 more
        check(controls(plain) == 41, "specials on everywhere read every spelling as a control token (the hole): 41", std::to_string(controls(plain)));
        check(controls(guarded) == 15, "with the content ranges only the template's 15 markers are control tokens", std::to_string(controls(guarded)));
        check(tok.decode(guarded, false) == got, "the guarded ids decode back to the prompt byte for byte");
        check(guarded.size() > plain.size(), "the spellings cost more tokens as text than as control tokens", std::to_string(guarded.size()) + " vs " + std::to_string(plain.size()));
    }

    std::printf("\n%s\n", g_fail ? "PROMPT TEST: FAILURE(S)" : "PROMPT TEST: PASS");
    return g_fail ? 1 : 0;
}
