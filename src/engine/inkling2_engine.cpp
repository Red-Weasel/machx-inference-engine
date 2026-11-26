// src/engine/inkling2_engine.cpp — Inkling 2 Mini (`inkling2_loop`) in the engine: load, the chat template, generate, parse.
// The runtime is Inkling2Model (src/model/inkling2.cpp, one card); the tokenizer is the directory's tokenizer.json; the
// template is chat_template.jinja rendered natively (inkling2_render_prompt mirrors it block for block, docs/inkling2/SPEC.md
// section 12; tests/unit/inkling2_serve_test.cpp against transformers' renders).
//
// The format: messages are <|message_user|> / <|message_model|>{name} / <|message_system|> / <|message_tool|>{name} blocks,
// each one content kind (<|content_text|>, <|content_thinking|>, <|content_invoke_tool_json|>{"name":..,"args":{..}}) closed by
// <|end_message|>; a model turn ends with <|content_model_end_sampling|> (the EOS). Thinking is a system line, "Thinking
// effort level: E" (0.9 by default, 0 = off), before the first non-system message. The template has no generation prompt:
// the prompt ends at the last block and the model opens its own turn (<|message_model|>), as the reference's generate()
// does on the rendered prompt (IE_INKLING2_OPEN_TURN=1 appends the opener instead).
//
// Served as a structured arch (openai_server.cpp): the stream carries the reasoning, "</think>", the answer, and each tool
// call as "<tool_call>{json}</tool_call>" once its block is closed; the reply is parsed token by token (markers are token
// ids, so text that spells a marker stays text) into content, reasoning_content and tool_calls.
#include "ie/engine.hpp"

#include "ie/deepseek41_generate.hpp"   // Ds41Generator::sample (the host sampler the other directory runtimes use)
#include "ie/inkling2_engine.hpp"
#include "ie/inkling2_vision.hpp"
#include "ie/lanes_auto.hpp"
#include "ie/tokenizer.hpp"
#include "ie/vision_required.hpp"   // vision_required_env (the owner's rule, said in the load log)
#include "nlohmann/json.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <concepts>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <filesystem>
#include <future>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>
#include <unordered_map>

namespace ie {

namespace {
using json = nlohmann::json;            // std::map objects: dump() sorts keys (the template's sort_keys=true)
using ojson = nlohmann::ordered_json;   // insertion order: json.dumps without sort_keys

std::string dump_c(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }
std::string dump_c(const ojson& j) { return j.dump(-1, ' ', false, ojson::error_handler_t::replace); }

// Jinja truthiness of a JSON value
bool truthy(const ojson& v) {
    if (v.is_null()) return false;
    if (v.is_boolean()) return v.get<bool>();
    if (v.is_number_integer()) return v.get<int64_t>() != 0;
    if (v.is_number_unsigned()) return v.get<uint64_t>() != 0;
    if (v.is_number_float()) return v.get<double>() != 0.0;
    if (v.is_string()) return !v.get_ref<const std::string&>().empty();
    return !v.empty();
}

size_t utf8_complete(std::string_view s) {
    if (s.empty()) return 0;
    size_t i = s.size(), cont = 0;
    while (i > 0 && cont < 3 && (uint8_t(s[i - 1]) & 0xC0u) == 0x80u) { --i; ++cont; }
    if (i == 0) return 0;
    const uint8_t lead = uint8_t(s[i - 1]);
    const size_t need = (lead & 0x80u) == 0 ? 1 : (lead & 0xE0u) == 0xC0u ? 2 : (lead & 0xF0u) == 0xE0u ? 3 : (lead & 0xF8u) == 0xF0u ? 4 : 1;
    return (cont + 1 >= need) ? i - 1 + need : i - 1;
}

constexpr std::pair<std::string_view, Ink2Mark> kMarkSpell[] = {
    {"<|message_model|>", Ink2Mark::kModel},          {"<|content_thinking|>", Ink2Mark::kThinking},
    {"<|content_text|>", Ink2Mark::kText},             {"<|content_tool_error|>", Ink2Mark::kToolError},
    {"<|content_invoke_tool_text|>", Ink2Mark::kToolText}, {"<|content_invoke_tool_json|>", Ink2Mark::kToolJson},
    {"<|end_message|>", Ink2Mark::kEnd},               {"<|content_model_end_sampling|>", Ink2Mark::kEos},
};

// The model runtime's prompt-cache rewind (Inkling2Model::truncate / earliest_restorable), used when this build's model
// interface has it; otherwise a request whose prompt does not extend the live sequence re-reads from position 0.
template <class M>
concept Ink2CanTruncate = requires(M& m, const M& cm) {
    { m.truncate(0u) } -> std::convertible_to<std::string>;
    { cm.earliest_restorable() } -> std::convertible_to<uint32_t>;
};
template <class M> bool ink2_truncate(M& m, uint32_t n, std::string& err) {
    if constexpr (Ink2CanTruncate<M>) {
        if (n < m.earliest_restorable() || n > m.n_pos()) return false;
        err = m.truncate(n);
        return err.empty();
    } else { (void)m; (void)n; (void)err; return false; }
}
template <class O> void ink2_set_rollback(O& o, uint32_t v) {
    if constexpr (requires(O& x) { x.rollback = 0u; }) o.rollback = v; else { (void)o; (void)v; }
}
constexpr bool kInk2Truncate = Ink2CanTruncate<Inkling2Model>;
}  // namespace

// ---------------------------------------------------------------------------------------------------------------- template

bool inkling2_effort_text(std::string_view effort, bool thinking, std::string& out, std::string& err) {
    if (!thinking) { out = "0"; return true; }
    size_t b = 0, e = effort.size();   // Jinja's trim: leading / trailing whitespace
    while (b < e && std::isspace(uint8_t(effort[b]))) ++b;
    while (e > b && std::isspace(uint8_t(effort[e - 1]))) --e;
    const std::string_view k = effort.substr(b, e - b);
    if (effort.empty()) { out = "0.9"; return true; }   // undefined -> the template's 0.9
    static constexpr std::pair<std::string_view, std::string_view> kMap[] = {
        {"none", "0"}, {"minimal", "0.1"}, {"low", "0.2"}, {"medium", "0.7"}, {"high", "0.9"}, {"xhigh", "0.99"}, {"max", "0.99"}};
    for (const auto& [n, v] : kMap) if (k == n) { out = v; return true; }
    err = "inkling2: unknown reasoning_effort '" + std::string(effort) + "' (none, minimal, low, medium, high, xhigh, max)";
    return false;
}

std::string inkling2_render_prompt(std::span<const ChatTurn> turns, std::string_view tools_json, bool thinking, std::string_view effort,
                                   std::string& err, std::vector<std::pair<size_t, size_t>>* content_out, bool open_turn,
                                   const std::vector<uint32_t>* image_tokens) {
    std::string eff;
    size_t next_image = 0;
    if (!inkling2_effort_text(effort, thinking, eff, err)) return {};
    PromptPiece out;
    // 1. the tool declarations: [{"description","name","parameters","type"}], sorted keys at every level, "," / ":"
    if (!tools_json.empty()) {
        const ojson tools = ojson::parse(tools_json, nullptr, false);
        if (tools.is_discarded() || !(tools.is_array() || tools.is_null())) { err = "inkling2: tools must be a JSON array"; return {}; }
        if (tools.is_array() && !tools.empty()) {
            json specs = json::array();
            for (const auto& t : tools) {
                if (!t.is_object()) { err = "inkling2: each tool must be an object"; return {}; }
                const ojson& fn = t.contains("function") ? t["function"] : t;
                if (!fn.is_object() || !fn.contains("name")) { err = "inkling2: a tool has no function name"; return {}; }
                json spec = json::object();
                spec["description"] = fn.contains("description") && truthy(fn["description"]) ? json::parse(dump_c(fn["description"])) : json("");
                spec["name"] = json::parse(dump_c(fn["name"]));
                spec["parameters"] = fn.contains("parameters") && truthy(fn["parameters"]) ? json::parse(dump_c(fn["parameters"])) : json::object();
                spec["type"] = t.contains("type") && truthy(t["type"]) ? json::parse(dump_c(t["type"])) : json("function");
                specs.push_back(std::move(spec));
            }
            out.add_text("<|message_system|>tool_declare<|content_xml|>");
            out.add_content(dump_c(specs));
            out.add_text("<|end_message|>");
        }
    }
    // the tool calls of every assistant turn, parsed once (the tool turns look their names up in ALL of them, as the template does)
    std::vector<ojson> calls(turns.size());
    for (size_t i = 0; i < turns.size(); ++i) {
        if (turns[i].role != "assistant" || turns[i].tool_calls_json.empty()) continue;
        calls[i] = ojson::parse(turns[i].tool_calls_json, nullptr, false);
        if (calls[i].is_discarded() || !(calls[i].is_array() || calls[i].is_null())) { err = "inkling2: an assistant turn's tool_calls is not a JSON array"; return {}; }
    }
    auto is_system = [](const std::string& r) { return r == "system" || r == "developer"; };
    const std::string effort_msg = "<|message_system|><|content_text|>Thinking effort level: " + eff + "<|end_message|>";
    bool emitted = false;
    for (size_t i = 0; i < turns.size(); ++i) {
        const ChatTurn& m = turns[i];
        const char* rt = m.role == "user" ? "<|message_user|>" : m.role == "assistant" ? "<|message_model|>"
                       : is_system(m.role) ? "<|message_system|>" : m.role == "tool" ? "<|message_tool|>" : nullptr;
        if (!rt) { err = "inkling2: unknown message role '" + m.role + "'"; return {}; }
        if (!m.images.empty() && !image_tokens) { err = "inkling2: images are not supported by this load (the vision tower is not loaded)"; return {}; }
        if (!m.images.empty() && m.role != "user" && m.role != "tool") { err = "inkling2: images are only allowed in user and tool turns"; return {}; }
        if (!emitted && !is_system(m.role)) { out.add_text(effort_msg); emitted = true; }
        // the header: the role token, a tool turn's name = the call this result answers: the nearest EARLIER call whose
        // id is the result's tool_call_id (ids are unique, so that is the template's match; with a client whose ids
        // repeat across turns, e.g. "call_0" in every response, taking every match would name the result after each
        // of them), else the first later one.
        PromptPiece head;
        head.add_text(rt);
        if (m.role == "tool" && !m.tool_call_id.empty()) {
            auto name_in = [&](size_t p) {
                if (!calls[p].is_array()) return false;
                for (const auto& tc : calls[p]) {
                    if (!tc.is_object() || !tc.contains("id") || !tc["id"].is_string() || tc["id"].get<std::string>() != m.tool_call_id) continue;
                    if (!tc.contains("function") || !tc["function"].is_object() || !tc["function"].contains("name")) continue;
                    const ojson& nm = tc["function"]["name"];
                    head.add_content(nm.is_string() ? nm.get<std::string>() : dump_c(nm));
                    return true;
                }
                return false;
            };
            bool found = false;
            for (size_t p = i; p-- > 0 && !found;) found = name_in(p);
            for (size_t p = i + 1; p < turns.size() && !found; ++p) found = name_in(p);
        }
        const bool model = m.role == "assistant";
        const bool has_calls = model && calls[i].is_array() && !calls[i].empty();
        const bool has_rc = model && m.reasoning_content && !m.reasoning_content->empty();
        if (has_rc) {
            out.add_piece(head); out.add_text("<|content_thinking|>"); out.add_content(*m.reasoning_content); out.add_text("<|end_message|>");
        }
        const std::string& text = model ? (m.content_without_tool_calls ? *m.content_without_tool_calls : m.content) : m.content;
        if (!m.images.empty()) {            // one block per part: the text between the images, each image its own block
            std::vector<std::string> seg;
            size_t p0 = 0, nm = 0;
            for (size_t p; (p = text.find(kChatImageMarker, p0)) != std::string::npos; p0 = p + kChatImageMarker.size(), ++nm)
                seg.push_back(text.substr(p0, p - p0));
            seg.push_back(text.substr(p0));
            if (nm != m.images.size()) {    // no usable placement: the images first, then the text without markers
                std::string all;
                for (const auto& sgm : seg) all += sgm;
                seg.assign(m.images.size() + 1, std::string());
                seg.back() = all;
            }
            for (size_t k = 0; k < seg.size(); ++k) {
                if (!seg[k].empty()) { out.add_piece(head); out.add_text("<|content_text|>"); out.add_content(seg[k]); out.add_text("<|end_message|>"); }
                if (k < m.images.size()) {
                    if (next_image >= image_tokens->size()) { err = "inkling2: internal: image token counts missing"; return {}; }
                    std::string ph;
                    for (uint32_t t = 0; t < (*image_tokens)[next_image]; ++t) ph += "<|unused_200054|>";
                    ++next_image;
                    out.add_text(rt); out.add_text("<|content_image|>"); out.add_text(ph); out.add_text("<|end_message|>");
                }
            }
        } else if (!text.empty() || !(has_calls || has_rc)) {
            out.add_piece(head); out.add_text("<|content_text|>"); out.add_content(text); out.add_text("<|end_message|>");
        }
        if (has_calls)
            for (const auto& tc : calls[i]) {
                const ojson fn = tc.is_object() && tc.contains("function") ? tc["function"] : ojson();
                if (!fn.is_object() || !fn.contains("name") || !fn["name"].is_string()) { err = "inkling2: a tool call's function name must be a string"; return {}; }
                const std::string name = fn["name"].get<std::string>();
                ojson args = fn.contains("arguments") && truthy(fn["arguments"]) ? fn["arguments"] : ojson::object();
                if (args.is_string()) {   // OpenAI sends the arguments as JSON text: parsed here, key order kept
                    const std::string& s = args.get_ref<const std::string&>();
                    if (s.find_first_not_of(" \t\r\n") == std::string::npos) args = ojson::object();
                    else args = ojson::parse(s, nullptr, false);
                }
                if (args.is_discarded() || !args.is_object()) { err = "inkling2: tool call '" + name + "' arguments must be a JSON object"; return {}; }
                std::vector<std::pair<std::string, const ojson*>> kv;   // dictsort(case_sensitive=true): code-point (= UTF-8 byte) order
                for (auto it = args.begin(); it != args.end(); ++it) kv.emplace_back(it.key(), &it.value());
                std::sort(kv.begin(), kv.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
                std::string body = "{";
                for (size_t k = 0; k < kv.size(); ++k) {
                    if (k) body += ',';
                    body += dump_c(ojson(kv[k].first)); body += ':'; body += dump_c(*kv[k].second);
                }
                body += '}';
                out.add_text("<|message_model|>"); out.add_content(name); out.add_text("<|content_invoke_tool_json|>{\"name\":");
                out.add_content(dump_c(ojson(name))); out.add_text(",\"args\":"); out.add_content(body); out.add_text("}<|end_message|>");
            }
        if (model) out.add_text("<|content_model_end_sampling|>");
    }
    if (!emitted) out.add_text(effort_msg);
    if (open_turn) out.add_text("<|message_model|>");
    if (content_out) *content_out = std::move(out.content);
    return std::move(out.text);
}

// ------------------------------------------------------------------------------------------------------------ reply parser

std::string Inkling2ReplyParser::open_answer() {
    if (thinking_ && !closed_) { closed_ = true; return "</think>"; }
    closed_ = true;
    return {};
}

void Inkling2ReplyParser::close_block() {
    if (kind_ == kCall) {
        const ojson j = ojson::parse(body_, nullptr, false);
        bool ok = !j.is_discarded() && j.is_object();
        std::string name = header_, args = "{}";
        if (ok && j.contains("name")) { if (j["name"].is_string()) name = j["name"].get<std::string>(); else ok = false; }
        const char* ak = j.is_object() && j.contains("args") ? "args" : j.is_object() && j.contains("arguments") ? "arguments" : nullptr;
        if (ok && ak) {
            ojson a = j[ak];
            if (a.is_string()) a = ojson::parse(a.get<std::string>(), nullptr, false);
            if (a.is_discarded() || !(a.is_object() || a.is_null())) ok = false;
            else if (a.is_object()) args = dump_c(a);
        }
        if (ok && !name.empty()) calls_.emplace_back(name, args);
        else content_ += body_;                               // an unparseable call stays visible
    }
    kind_ = kNone;
    body_.clear();
}

std::string Inkling2ReplyParser::feed(Ink2Mark m, std::string_view text) {
    std::string s;
    switch (m) {
    case Ink2Mark::kNone:
        switch (kind_) {
        case kNone:   s = open_answer(); kind_ = kAnswer; [[fallthrough]];   // text outside a block: the answer (never dropped)
        case kAnswer: content_ += text; s += text; break;
        case kHeader: header_ += text; break;
        case kReason: reasoning_ += text; if (thinking_ && !closed_) s = text; break;
        case kCall:   body_ += text; break;
        }
        break;
    case Ink2Mark::kModel:
        if (kind_ != kHeader) close_block();
        kind_ = kHeader; header_.clear();
        break;
    case Ink2Mark::kThinking:
        if (kind_ != kHeader) close_block();
        kind_ = kReason;
        break;
    case Ink2Mark::kText: case Ink2Mark::kToolError: case Ink2Mark::kToolText:
        if (kind_ != kHeader) close_block();
        s = open_answer(); kind_ = kAnswer;
        break;
    case Ink2Mark::kToolJson: {
        if (kind_ != kHeader) { close_block(); header_.clear(); }
        size_t b = 0, e = header_.size();
        while (b < e && std::isspace(uint8_t(header_[b]))) ++b;
        while (e > b && std::isspace(uint8_t(header_[e - 1]))) --e;
        header_ = header_.substr(b, e - b);
        s = open_answer(); kind_ = kCall; body_.clear();
        break;
    }
    case Ink2Mark::kEnd: {
        const size_t before = calls_.size();
        const bool call = kind_ == kCall;
        const std::string body = body_;
        close_block();
        if (call && calls_.size() > before) s = "<tool_call>" + dump_c(ojson{{"name", calls_.back().first}, {"arguments", ojson::parse(calls_.back().second)}}) + "</tool_call>";
        else if (call) s = body;                              // shown as answer text, as close_block kept it
        header_.clear();
        break;
    }
    case Ink2Mark::kEos:
        if (kind_ != kHeader) close_block();
        kind_ = kNone; header_.clear();
        break;
    case Ink2Mark::kOther: break;
    }
    return s;
}

void Inkling2ReplyParser::finish(bool complete) {
    if (kind_ == kCall && !complete) {                        // cut off inside a call: its name; the partial body is dropped
        truncated_ = header_;
        if (truncated_.empty()) {
            const size_t p = body_.find("\"name\"");
            if (p != std::string::npos) {
                const size_t q = body_.find('"', body_.find(':', p) + 1);
                const size_t r = q == std::string::npos ? q : body_.find('"', q + 1);
                if (r != std::string::npos) truncated_ = body_.substr(q + 1, r - q - 1);
            }
        }
        if (truncated_.empty()) truncated_ = "?";
        kind_ = kNone; body_.clear();
        return;
    }
    if (kind_ != kHeader) close_block();
    kind_ = kNone;
}

std::string Inkling2ReplyParser::tool_calls_json() const {
    // ids unique across a conversation: the template names a tool result after EVERY call carrying its id
    static std::atomic<uint64_t> serial{0};
    static const uint32_t salt = std::random_device{}();
    ojson a = ojson::array();
    for (const auto& [name, args] : calls_) {
        char id[40];
        std::snprintf(id, sizeof id, "call_%08x%llu", salt, static_cast<unsigned long long>(serial.fetch_add(1)));
        a.push_back({{"id", id}, {"type", "function"}, {"function", {{"name", name}, {"arguments", args}}}});
    }
    return dump_c(a);
}

std::vector<std::pair<Ink2Mark, std::string>> inkling2_events_from_text(std::string_view raw) {
    std::vector<std::pair<Ink2Mark, std::string>> ev;
    size_t p = 0;
    while (p < raw.size()) {
        size_t best = std::string_view::npos; Ink2Mark bm = Ink2Mark::kNone; size_t bl = 0;
        for (const auto& [sp, mk] : kMarkSpell) {
            const size_t at = raw.find(sp, p);
            if (at < best) { best = at; bm = mk; bl = sp.size(); }
        }
        if (best == std::string_view::npos) { ev.emplace_back(Ink2Mark::kNone, std::string(raw.substr(p))); break; }
        if (best > p) ev.emplace_back(Ink2Mark::kNone, std::string(raw.substr(p, best - p)));
        ev.emplace_back(bm, std::string());
        p = best + bl;
    }
    return ev;
}

bool inkling2_vision_ready(const Inkling2Bundle& b) { return b.vis && b.vis->ready(); }

Ink2Mark Inkling2Bundle::mark_of(int32_t id) const {
    for (const auto& [i, m] : marks) if (i == id) return m;
    return tok.is_special(id) ? Ink2Mark::kOther : Ink2Mark::kNone;
}

// --------------------------------------------------------------------------------------------------------------- the loop

namespace {
// The prompt-cache ids: the placeholders replaced by their images' stand-ins.
std::vector<int32_t> ink2_keyed(const std::vector<int32_t>& ids, const Ink2Images* im) {
    std::vector<int32_t> k = ids;
    if (im) for (size_t i = 0; i < im->at.size(); ++i) if (im->at[i] < k.size()) k[im->at[i]] = im->key[i];
    return k;
}
// The image rows of the prompt chunk [at, at + n) into ro (positions relative to the chunk); none: ro cleared.
void ink2_chunk_images(const Ink2Images* im, uint32_t at, uint32_t n, std::vector<float>& rows, std::vector<uint32_t>& pos,
                       Inkling2RunOptions& ro) {
    ro.image_rows = nullptr; ro.image_at = nullptr;
    if (!im || im->at.empty()) return;
    rows.clear(); pos.clear();
    const size_t H = im->rows.size() / im->at.size();
    for (size_t i = 0; i < im->at.size(); ++i)
        if (im->at[i] >= at && im->at[i] < at + n) {
            pos.push_back(im->at[i] - at);
            rows.insert(rows.end(), im->rows.begin() + i * H, im->rows.begin() + (i + 1) * H);
        }
    if (!pos.empty()) { ro.image_rows = &rows; ro.image_at = &pos; }
}

// Prefill (reusing the live sequence's longest common prefix) + sampling loop. chat: the reply is parsed (parser) and the
// stream is the parser's translation; raw generate: the decoded text as it comes.
GenerateResult inkling2_run(Inkling2Bundle& b, const std::vector<int32_t>& ids, const SamplingParams& sp, const TokenCallback& on_token,
                            Inkling2ReplyParser* parser, const Ink2Images* im = nullptr) {
    GenerateResult r;
    const std::vector<int32_t> keyed = ink2_keyed(ids, im);
    r.prompt_tokens = uint32_t(ids.size());
    if (ids.empty()) { r.finish_reason = "error: empty prompt"; return r; }
    if (ids.size() >= b.ctx) { r.finish_reason = "context_length_exceeded"; return r; }
    const uint32_t room = b.ctx - uint32_t(ids.size());
    const uint32_t max_new = (sp.max_tokens == 0 || sp.max_tokens == kMaxTokensUnlimited) ? room : std::min(sp.max_tokens, room);
    auto fail = [&](const std::string& what) {
        b.live.clear(); b.model.reset_state();
        r.finish_reason = "error: inkling2: " + what;
        return r;
    };

    const auto t_pf = std::chrono::steady_clock::now();
    // the prefix the state can keep: the longest common prefix, leaving at least one prompt row to read (its logits)
    uint32_t L = 0;
    if (b.reuse) while (L < b.live.size() && L + 1 < ids.size() && b.live[L] == keyed[L]) ++L;
    if (L < b.live.size()) {                                  // the state holds more than the common prefix: rewind
        std::string terr;
        if (L == 0 || !ink2_truncate(b.model, L, terr)) {
            if (!terr.empty()) std::fprintf(stderr, "[inkling2] truncate(%u) failed (%s): re-reading the prompt\n", L, terr.c_str());
            else if (L > 0) {
                if constexpr (Ink2CanTruncate<Inkling2Model>)
                    std::fprintf(stderr, "[inkling2] prompt cache: the common prefix (%u tokens) is before the earliest restorable position %u "
                                         "(%zu held, rollback reach %u; IE_INKLING2_ROLLBACK raises it): re-reading the whole prompt (%zu tokens)\n",
                                 L, b.model.earliest_restorable(), b.live.size(), b.rollback, ids.size());
            }
            b.model.reset_state();
            L = 0;
        }
        b.live.resize(L);
    }
    r.cached_tokens = L;
    r.cache_source = !b.reuse ? "" : L ? "live" : "none";
    Inkling2RunOptions ro;
    ro.record_routes = false;
    Inkling2Result res;
    std::vector<float> irows;
    std::vector<uint32_t> ipos;
    for (uint32_t at = L; at < ids.size();) {                 // the prompt in chunks; the liveness probe between them
        const uint32_t n = std::min<uint32_t>(b.chunk, uint32_t(ids.size()) - at);
        std::vector<int32_t> part(ids.begin() + at, ids.begin() + at + n);
        ink2_chunk_images(im, at, n, irows, ipos, ro);
        if (auto e = b.model.forward(part, ro, res); !e.empty()) return fail("prefill: " + e);
        b.live.insert(b.live.end(), keyed.begin() + at, keyed.begin() + at + n);
        at += n;
        if (at < ids.size() && on_token && !on_token(std::string_view{})) { r.finish_reason = "abort"; return r; }
    }
    r.prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pf).count();
    ro.image_rows = nullptr; ro.image_at = nullptr;

    Ds41SampleParams p;
    p.temperature = sp.temperature; p.top_k = sp.top_k; p.top_p = sp.top_p; p.min_p = sp.min_p;
    p.repeat_penalty = sp.repeat_penalty; p.repeat_window = std::min<uint32_t>(sp.repeat_window, 512);
    p.seed = sp.seed ? sp.seed : uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
    uint64_t rng = p.seed;
    const size_t window = std::min<size_t>(sp.repeat_window, 512);
    std::vector<int32_t> recent(ids.end() - std::min(ids.size(), window), ids.end());
    const bool pf_pen = (sp.presence_penalty != 0.f || sp.frequency_penalty != 0.f) && window > 0;
    std::unordered_map<int32_t, uint32_t> counts;
    auto pick = [&](std::vector<float>& lg) {
        if (pf_pen) {                                         // presence / frequency over the penalty window (llama.cpp's rule)
            counts.clear();
            for (int32_t t : recent) ++counts[t];
            for (const auto& [t, c] : counts)
                if (t >= 0 && size_t(t) < lg.size()) lg[t] -= sp.presence_penalty + sp.frequency_penalty * float(c);
        }
        return Ds41Generator::sample(lg, recent, p, rng);
    };
    auto finite = [&](const std::vector<float>& lg) { for (float v : lg) if (!std::isfinite(v)) return false; return true; };

    std::string pending;                                      // undelivered stream text (an incomplete UTF-8 tail)
    auto flush = [&]() -> bool {
        const size_t done = utf8_complete(pending);
        if (!done) return true;
        const std::string piece = pending.substr(0, done);
        pending.erase(0, done);
        return !on_token || on_token(piece);
    };
    const auto t_dec = std::chrono::steady_clock::now();
    r.finish_reason = "length";
    bool complete = false;
    if (!finite(res.last_logits)) return fail("non-finite logits at position " + std::to_string(ids.size() - 1));
    int32_t id = pick(res.last_logits);
    std::vector<int32_t> head;                                // the reply's first ids (IE_INKLING2_LOG_HEAD=1 prints them)
    static const bool log_head = [] { const char* v = std::getenv("IE_INKLING2_LOG_HEAD"); return v && *v == '1'; }();
    for (uint32_t k = 0;;) {
        if (log_head && head.size() < 12) head.push_back(id);
        if (!sp.ignore_eos && std::find(b.stop.begin(), b.stop.end(), id) != b.stop.end()) {
            r.finish_reason = "stop"; complete = true;
            if (parser) parser->feed(b.mark_of(id));
            break;
        }
        ++r.completion_tokens;
        recent.push_back(id);
        if (recent.size() > window) recent.erase(recent.begin());
        const Ink2Mark mk = b.mark_of(id);
        const std::string piece = (mk == Ink2Mark::kNone || !parser) ? b.tok.decode(std::vector<int32_t>{id}) : std::string();
        r.text += parser ? std::string() : piece;
        pending += parser ? parser->feed(mk, piece) : piece;
        if (!flush()) { r.finish_reason = "abort"; break; }
        if (++k == max_new) break;
        if (auto e = b.model.forward({id}, ro, res); !e.empty()) return fail("decode: " + e);
        b.live.push_back(id);
        if (!finite(res.last_logits)) return fail("non-finite logits at position " + std::to_string(b.live.size() - 1));
        id = pick(res.last_logits);
    }
    if (parser) parser->finish(complete);
    if (log_head) {
        std::string h;
        for (int32_t t : head) { h += '['; h += b.tok.decode(std::vector<int32_t>{t}); h += ']'; }
        std::fprintf(stderr, "[inkling2] reply head: %s\n", h.c_str());
    }
    if (!pending.empty() && on_token && r.finish_reason != "abort") on_token(pending);
    r.decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dec).count();
    return r;
}
}  // namespace

// ------------------------------------------------------------------------------------------- many conversations (parallel N)
//
// --parallel N > 1: every card holds the model once and N_card conversation slots (Inkling2Model::batch_*); one worker
// thread per card runs that card's loop: admit waiting requests into free slots (the slot whose cached tokens share the
// longest prefix with the prompt, else the least recently used), read one prompt chunk of one prefilling request, then one
// decode step for every decoding request together (continuous batching: a request joins at the next step, a finished one
// leaves). Each conversation's arithmetic is the same as alone (batch_step), so its greedy reply does not depend on its
// neighbours. A request's HTTP thread waits for its stream text (and probes the connection); a dropped client cancels it.
// Cards step independently; a new request goes to the card holding the longest common prefix, else the less loaded one.

struct Ink2Job {
    std::vector<int32_t> ids;
    std::vector<int32_t> keyed;                        // the prompt-cache ids (images' stand-ins at their placeholders)
    Ink2Images im;                                     // the request's images (empty: none)
    SamplingParams sp;
    Inkling2ReplyParser* parser = nullptr;             // chat: the reply parser (owned by the request's thread); raw: null
    // the worker's (only the card's worker touches these after submission)
    uint32_t slot = 0, at = 0, k = 0, max_new = 0;
    int      phase = 0;                                // 0 waiting, 1 reading the prompt, 2 decoding
    int32_t  next = -1;                                // the id to feed at the next step
    Ds41SampleParams p;
    uint64_t rng = 0;
    size_t   window = 0;
    bool     pf_pen = false, complete = false;
    std::vector<int32_t> recent;
    std::unordered_map<int32_t, uint32_t> counts;
    std::string pending;
    std::chrono::steady_clock::time_point t_pf, t_dec;
    // shared with the request's thread (mu)
    std::mutex mu;
    std::condition_variable cv;
    std::string out;                                   // stream text not yet delivered
    bool done = false;
    std::atomic<bool> cancel{false};
    GenerateResult r;
};

struct Ink2Card {
    const Inkling2Bundle* b = nullptr;
    sycl::queue* q = nullptr;
    Inkling2Model* model = nullptr;
    std::unique_ptr<sycl::queue> own_q;                // card 1+: its own queue and model (card 0 uses the bundle's)
    std::unique_ptr<Inkling2Model> own_model;
    uint32_t idx = 0, slots = 0, ctx = 0, chunk = 256;
    std::string name;
    std::vector<std::vector<int32_t>> live;            // per slot: the tokens its state holds
    std::vector<Ink2Job*> owner;                       // per slot: the request using it (null: free)
    std::vector<uint64_t> used_at;                     // per slot: when it was last given out (LRU)
    uint64_t tick = 0;
    std::deque<Ink2Job*> waiting;
    std::mutex mu;
    std::condition_variable cv;
    bool stop = false;
    std::atomic<uint32_t> load{0}, active{0};         // assigned (waiting + running), running
    std::atomic<uint64_t> steps{0}, rows{0};
    std::atomic<double> step_ms{0.0};
    std::thread th;                                    // last: joined in ~Ink2Serve before anything above goes
    void loop();
};

struct Ink2Serve {
    std::vector<std::unique_ptr<Ink2Card>> cards;
    std::mutex route_mu;
    bool main_lane = false;   // card 0 = ONE full-context lane for the main conversation; card 1+ = the sub-agents' lanes
    uint32_t total_slots() const { uint32_t n = 0; for (const auto& c : cards) n += c->slots; return n; }
    ~Ink2Serve() {
        for (auto& c : cards) { { std::lock_guard<std::mutex> lk(c->mu); c->stop = true; } c->cv.notify_all(); }
        for (auto& c : cards) if (c->th.joinable()) c->th.join();
        for (auto it = cards.rbegin(); it != cards.rend(); ++it) {        // card 1+ first: their models before their queues
            (*it)->own_model.reset();
            (*it)->own_q.reset();
        }
    }
};

uint32_t ink2_lcp(const std::vector<int32_t>& a, const std::vector<int32_t>& ids) {
    uint32_t L = 0;
    while (L < a.size() && L + 1 < ids.size() && a[L] == ids[L]) ++L;   // at least one prompt row is read (its logits)
    return L;
}

void Ink2Card::loop() {
    using clk = std::chrono::steady_clock;
    Inkling2RunOptions ro;
    ro.record_routes = false;
    std::vector<uint32_t> rs;
    std::vector<int32_t> rid;
    std::vector<float> lgs;
    size_t rr = 0;                                     // round-robin over prefilling requests
    const size_t V = model->config().vocab_unpadded;

    auto deliver = [&](Ink2Job& j, bool all) {        // complete UTF-8 of the pending text to the request's thread
        const size_t n = all ? j.pending.size() : utf8_complete(j.pending);
        if (!n) return;
        std::lock_guard<std::mutex> lk(j.mu);
        j.out.append(j.pending, 0, n);
        j.pending.erase(0, n);
        j.cv.notify_all();
    };
    auto finish = [&](Ink2Job& j, const std::string& reason) {   // the request leaves its slot
        if (!reason.empty()) j.r.finish_reason = reason;
        if (j.parser && !j.r.finish_reason.starts_with("error")) j.parser->finish(j.complete);
        if (j.phase == 2) j.r.decode_ms = std::chrono::duration<double, std::milli>(clk::now() - j.t_dec).count();
        if (j.r.finish_reason != "abort" && !j.r.finish_reason.starts_with("error")) deliver(j, true);
        if (j.r.finish_reason.starts_with("error")) { live[j.slot].clear(); model->batch_truncate(j.slot, 0); }
        {
            std::lock_guard<std::mutex> lk(mu);
            owner[j.slot] = nullptr;
        }
        active.fetch_sub(1); load.fetch_sub(1);
        std::lock_guard<std::mutex> lk(j.mu);
        j.done = true;
        j.cv.notify_all();
    };
    auto fail = [&](Ink2Job& j, const std::string& what) { finish(j, "error: inkling2: " + what); };
    auto finite = [](const float* lg, size_t n) { for (size_t i = 0; i < n; ++i) if (!std::isfinite(lg[i])) return false; return true; };
    // one id from these logits: the stop test, the stream text, and the next id to feed (as inkling2_run's loop)
    // the draw alone (touches only this request's state and row: several rows may draw at once); -1: non-finite logits
    auto pick = [&](Ink2Job& j, float* lgp) -> int32_t {
        if (!finite(lgp, V)) return -1;
        if (j.pf_pen) {                                 // in place: the row is this request's alone
            j.counts.clear();
            for (int32_t t : j.recent) ++j.counts[t];
            for (const auto& [t, c] : j.counts)
                if (t >= 0 && size_t(t) < V) lgp[t] -= j.sp.presence_penalty + j.sp.frequency_penalty * float(c);
        }
        return Ds41Generator::sample_row(lgp, V, j.recent, j.p, j.rng);
    };
    auto take = [&](Ink2Job& j, float* lgp, int32_t drawn = -2) {
        const int32_t id = drawn == -2 ? pick(j, lgp) : drawn;
        if (id < 0) { fail(j, "non-finite logits at position " + std::to_string(live[j.slot].size())); return; }
        if (!j.sp.ignore_eos && std::find(b->stop.begin(), b->stop.end(), id) != b->stop.end()) {
            j.complete = true;
            if (j.parser) j.parser->feed(b->mark_of(id));
            finish(j, "stop");
            return;
        }
        ++j.r.completion_tokens;
        j.recent.push_back(id);
        if (j.recent.size() > j.window) j.recent.erase(j.recent.begin());
        const Ink2Mark mk = b->mark_of(id);
        const std::string piece = (mk == Ink2Mark::kNone || !j.parser) ? b->tok.decode(std::vector<int32_t>{id}) : std::string();
        if (!j.parser) j.r.text += piece;
        j.pending += j.parser ? j.parser->feed(mk, piece) : piece;
        deliver(j, false);
        if (++j.k == j.max_new) { finish(j, "length"); return; }
        j.next = id;
    };

    static const bool step_log = [] { const char* e = std::getenv("IE_INKLING2_STEP_LOG"); return e && *e == '1'; }();
    double lg_step = 0, lg_take = 0, lg_round = 0, lg_pre = 0; uint64_t lg_n = 0, lg_rows = 0;
    // Prompt reading vs decoding (2026-11-23): a prompt chunk at depth takes 0.4-0.9 s against a 38 ms decode step, so
    // one chunk a round left the decoding slots at 2 tok/s while a 25K prompt was read (94 s). With slots decoding, a
    // chunk runs only once the decode time since the last chunk reaches (1 - share) / share of that chunk's time: the
    // prompt gets `share` of the card and the decoders the rest. No one decoding: a chunk every round, as before.
    static const double pf_share = [] {
        const char* e = std::getenv("IE_INKLING2_PREFILL_SHARE");
        const double v = e && *e ? std::atof(e) : 0.45;
        return v <= 0.0 ? 0.45 : std::min(1.0, v);
    }();
    // Accounting by wall time per round (a chunk's kernels may still run when batch_forward returns and land in the next
    // decode step): a round with a chunk costs the chunk its wall time minus the usual decode-only round.
    double pf_last_ms = 0, dec_since_pf_ms = 0, dec_round_ms = 0;
    auto lg_t = clk::now();
    for (;;) {
        if (step_log) {
            const auto now = clk::now();
            lg_round += std::chrono::duration<double, std::milli>(now - lg_t).count();
            lg_t = now;
            if (lg_n >= 200) {
                std::fprintf(stderr, "[inkling2] %s: %llu rounds: step %.2f ms (GPU batch_step), take %.2f ms, prefill %.2f ms, round %.2f ms, %.2f rows\n",
                             name.c_str(), (unsigned long long)lg_n, lg_step / lg_n, lg_take / lg_n, lg_pre / lg_n, lg_round / lg_n, double(lg_rows) / lg_n);
                lg_step = lg_take = lg_round = lg_pre = 0; lg_n = lg_rows = 0;
            }
        }
        std::vector<Ink2Job*> pre, dec;
        {
            std::unique_lock<std::mutex> lk(mu);
            auto busy = [&] {
                for (auto* j : owner) if (j) return true;
                return false;
            };
            cv.wait(lk, [&] { return stop || busy() || (!waiting.empty() && std::find(owner.begin(), owner.end(), nullptr) != owner.end()); });
            if (stop) {                                 // shutdown: every request still here ends as an abort
                for (auto* j : waiting) { std::lock_guard<std::mutex> l2(j->mu); j->r.finish_reason = "abort"; j->done = true; j->cv.notify_all(); }
                waiting.clear();
                lk.unlock();
                for (uint32_t s = 0; s < slots; ++s) if (owner[s]) finish(*owner[s], "abort");
                return;
            }
            // admit: each waiting request (in order) to the free slot sharing the longest prefix, else the least recently used
            while (!waiting.empty()) {
                int best = -1; uint32_t bl = 0;
                for (uint32_t s = 0; s < slots; ++s) {
                    if (owner[s]) continue;
                    const uint32_t L = ink2_lcp(live[s], waiting.front()->keyed);
                    if (best < 0 || L > bl || (L == bl && used_at[s] < used_at[uint32_t(best)])) { best = int(s); bl = L; }
                }
                if (best < 0) break;
                Ink2Job* j = waiting.front();
                waiting.pop_front();
                owner[uint32_t(best)] = j;
                used_at[uint32_t(best)] = ++tick;
                j->slot = uint32_t(best);
                j->phase = 1;
                active.fetch_add(1);
            }
            for (uint32_t s = 0; s < slots; ++s) {
                Ink2Job* j = owner[s];
                if (!j) continue;
                if (j->phase == 1) pre.push_back(j); else if (j->phase == 2 && j->next >= 0) dec.push_back(j);
            }
        }
        // cancelled requests (client gone) leave first
        auto drop = [&](std::vector<Ink2Job*>& v) {
            v.erase(std::remove_if(v.begin(), v.end(), [&](Ink2Job* j) { if (j->cancel.load()) { finish(*j, "abort"); return true; } return false; }), v.end());
        };
        drop(pre); drop(dec);
        // one prompt chunk of one prefilling request (the others wait their turn; decoding goes on every round)
        const auto lg_p0 = clk::now();
        const bool pf_turn = !pre.empty() && (dec.empty() || pf_share >= 1.0 ||
                                              dec_since_pf_ms >= pf_last_ms * (1.0 - pf_share) / pf_share);
        if (pf_turn) {
            Ink2Job& j = *pre[rr++ % pre.size()];
            if (j.at == 0 && j.r.prompt_tokens == 0) {          // first visit: the prompt cache, the sampler
                j.t_pf = clk::now();
                j.r.prompt_tokens = uint32_t(j.ids.size());
                auto& lv = live[j.slot];
                uint32_t L = b->reuse ? ink2_lcp(lv, j.keyed) : 0;
                if (L < lv.size()) {
                    if (L == 0 || L < model->batch_earliest_restorable(j.slot) || !model->batch_truncate(j.slot, L).empty()) {
                        if (L > 0)
                            std::fprintf(stderr, "[inkling2] %s slot %u: the common prefix (%u tokens) is before the slot's earliest restorable "
                                                 "position %u: re-reading the whole prompt (%zu tokens)\n", name.c_str(), j.slot, L,
                                         model->batch_earliest_restorable(j.slot), j.ids.size());
                        model->batch_truncate(j.slot, 0);
                        L = 0;
                    }
                    lv.resize(L);
                }
                if (lv.size() != model->batch_pos(j.slot)) { model->batch_truncate(j.slot, 0); lv.clear(); L = 0; }
                j.at = L;
                j.r.cached_tokens = L;
                j.r.cache_source = !b->reuse ? "" : L ? "live" : "none";
                j.r.lane = idx * 1000 + j.slot;
                j.r.lane_ctx = ctx;
                const uint32_t room = ctx - uint32_t(j.ids.size());
                j.max_new = (j.sp.max_tokens == 0 || j.sp.max_tokens == kMaxTokensUnlimited) ? room : std::min(j.sp.max_tokens, room);
                j.p.temperature = j.sp.temperature; j.p.top_k = j.sp.top_k; j.p.top_p = j.sp.top_p; j.p.min_p = j.sp.min_p;
                j.p.repeat_penalty = j.sp.repeat_penalty; j.p.repeat_window = std::min<uint32_t>(j.sp.repeat_window, 512);
                j.p.seed = j.sp.seed ? j.sp.seed : uint64_t(clk::now().time_since_epoch().count());
                j.rng = j.p.seed;
                j.window = std::min<size_t>(j.sp.repeat_window, 512);
                j.recent.assign(j.ids.end() - std::min(j.ids.size(), j.window), j.ids.end());
                j.pf_pen = (j.sp.presence_penalty != 0.f || j.sp.frequency_penalty != 0.f) && j.window > 0;
                if (j.at == 0) j.at = 0;
            }
            const uint32_t n = std::min<uint32_t>(chunk, uint32_t(j.ids.size()) - j.at);
            std::vector<int32_t> part(j.ids.begin() + j.at, j.ids.begin() + j.at + n);
            Inkling2Result res;
            const auto t0 = clk::now();
            std::vector<float> irows;
            std::vector<uint32_t> ipos;
            ink2_chunk_images(j.im.at.empty() ? nullptr : &j.im, j.at, n, irows, ipos, ro);
            const std::string perr = model->batch_forward(j.slot, part, ro, res);
            ro.image_rows = nullptr; ro.image_at = nullptr;
            if (!perr.empty()) fail(j, "prefill: " + perr);
            else {
                live[j.slot].insert(live[j.slot].end(), j.keyed.begin() + j.at, j.keyed.begin() + j.at + n);
                j.at += n;
                (void)t0;
                if (j.at == j.ids.size()) {                     // the prompt is in: its last logits give the first id
                    j.r.prefill_ms = std::chrono::duration<double, std::milli>(clk::now() - j.t_pf).count();
                    j.phase = 2;
                    j.t_dec = clk::now();
                    take(j, res.last_logits.data());
                }
            }
        }
        if (step_log) lg_pre += std::chrono::duration<double, std::milli>(clk::now() - lg_p0).count();
        // one decode step for every decoding request
        if (!dec.empty()) {
            rs.clear(); rid.clear();
            for (auto* j : dec) { rs.push_back(j->slot); rid.push_back(j->next); }
            // the draws on the card (2026-11-23): every row's temperature / top-k / top-p / min-p / repetition and
            // presence / frequency penalties go with the step, only the ids come back (not 0.8 MB of logits a row, and
            // no host sampling: ~1.5 ms a row of CPU at temperature > 0). IE_INKLING2_DEVICE_SAMPLING=0: the host path.
            static const bool dev_sample = [] { const char* e = std::getenv("IE_INKLING2_DEVICE_SAMPLING"); return !(e && *e == '0'); }();
            std::vector<int32_t> sids;
            const auto t0 = clk::now();
            std::string e;
            if (dev_sample) {
                std::vector<I2SampleRow> srows(dec.size());
                std::vector<int32_t> srec, spfi;
                std::vector<float> spfd;
                for (size_t i = 0; i < dec.size(); ++i) {
                    Ink2Job& j = *dec[i];
                    I2SampleRow& w = srows[i];
                    w.temp = j.p.temperature; w.top_k = j.p.top_k; w.top_p = j.p.top_p; w.min_p = j.p.min_p;
                    w.rep = j.p.repeat_window ? j.p.repeat_penalty : 1.f;
                    if (w.rep != 1.f) {
                        const size_t from = j.recent.size() > j.p.repeat_window ? j.recent.size() - j.p.repeat_window : 0;
                        w.rec_off = uint32_t(srec.size()); w.rec_n = uint32_t(j.recent.size() - from);
                        srec.insert(srec.end(), j.recent.begin() + std::ptrdiff_t(from), j.recent.end());
                    }
                    if (j.pf_pen) {
                        j.counts.clear();
                        for (int32_t t : j.recent) ++j.counts[t];
                        w.pf_off = uint32_t(spfi.size()); w.pf_n = uint32_t(j.counts.size());
                        for (const auto& [t, c] : j.counts) { spfi.push_back(t); spfd.push_back(j.sp.presence_penalty + j.sp.frequency_penalty * float(c)); }
                    }
                    j.rng += 0x9E3779B97F4A7C15ull;               // the row's own stream: one seed a draw
                    uint64_t z = j.rng;
                    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
                    w.seed = z ^ (z >> 31);
                }
                e = model->batch_step_sample(rs, rid, srows, srec, spfi, spfd, sids);
            } else e = model->batch_step(rs, rid, lgs);
            const double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            steps.fetch_add(1); rows.fetch_add(rs.size()); step_ms.store(ms);
            const auto lg_k0 = clk::now();
            if (step_log) { lg_step += ms; ++lg_n; lg_rows += rs.size(); }
            // sampled rows draw on their own threads (a temperature draw is ~1.5 ms a row over 200K logits: serial, 6 rows
            // cost 9 ms of a 47 ms round); greedy rows are an argmax and stay here. Each row's own rng: the same draws.
            std::vector<int32_t> drawn(dec.size(), -2);
            if (e.empty() && dev_sample) drawn = sids;
            else if (e.empty() && dec.size() > 1) {
                std::vector<std::future<int32_t>> fut(dec.size());
                for (size_t i = 0; i < dec.size(); ++i)
                    if (dec[i]->p.temperature > 0.f)
                        fut[i] = std::async(std::launch::async, [&, i] { return pick(*dec[i], lgs.data() + i * V); });
                for (size_t i = 0; i < dec.size(); ++i) if (fut[i].valid()) drawn[i] = fut[i].get();
            }
            for (size_t i = 0; i < dec.size(); ++i) {
                Ink2Job& j = *dec[i];
                if (!e.empty()) { fail(j, "decode: " + e); continue; }
                live[j.slot].push_back(j.next);
                j.next = -1;
                take(j, dev_sample ? nullptr : lgs.data() + i * V, drawn[i]);
            }
            if (step_log) lg_take += std::chrono::duration<double, std::milli>(clk::now() - lg_k0).count();
        }
        {
            const double rms = std::chrono::duration<double, std::milli>(clk::now() - lg_p0).count();
            if (pf_turn) {
                pf_last_ms = std::max(0.0, rms - (dec.empty() ? 0.0 : dec_round_ms));
                dec_since_pf_ms = 0;
            } else if (!dec.empty()) {
                dec_since_pf_ms += rms;
                dec_round_ms = dec_round_ms > 0 ? 0.9 * dec_round_ms + 0.1 * rms : rms;
            }
        }
    }
}

static inline uint32_t ids_size_of(const Ink2Job& j) { return uint32_t(j.ids.size()); }

// The request's side: route, wait for the stream text, probe the connection, take the result.
GenerateResult ink2_serve_run(Inkling2Bundle& b, std::vector<int32_t> ids, const SamplingParams& sp, const TokenCallback& on_token,
                              Inkling2ReplyParser* parser, const Ink2Images* im = nullptr) {
    Ink2Serve& sv = *b.serve;
    GenerateResult r0;
    r0.prompt_tokens = uint32_t(ids.size());
    if (ids.empty()) { r0.finish_reason = "error: empty prompt"; return r0; }
    uint32_t ctx = 0;
    for (const auto& c : sv.cards) ctx = std::max(ctx, c->ctx);
    if (ids.size() >= ctx) { r0.finish_reason = "context_length_exceeded"; return r0; }
    auto job = std::make_unique<Ink2Job>();
    job->keyed = ink2_keyed(ids, im);
    if (im) job->im = *im;
    job->ids = std::move(ids);
    job->sp = sp;
    job->parser = parser;
    Ink2Card* card = nullptr;
    {   // the card: the one holding the longest common prefix in a free slot (>= 64 tokens), else the less loaded one.
        // Main lane: "ie_lane": "main" always goes to card 0's full-context lane; any other request to card 1+ (the
        // sub-agents' lanes), unless its prompt does not fit their context, then to the main lane too
        std::lock_guard<std::mutex> rl(sv.route_mu);
        auto eligible = [&](const Ink2Card& c) {
            if (!sv.main_lane) return c.ctx > ids_size_of(*job);
            if (sp.lane == "main") {   // the main lane; a second "main" conversation takes a free sub-agent lane meanwhile
                const Ink2Card& m = *sv.cards[0];
                const bool main_busy = m.load.load() >= m.slots;
                return c.idx == 0 || (main_busy && c.ctx > ids_size_of(*job) && c.load.load() < c.slots);
            }
            bool small_fits = false;
            for (const auto& o : sv.cards) small_fits |= o->idx != 0 && o->ctx > ids_size_of(*job);
            return small_fits ? (c.idx != 0 && c.ctx > ids_size_of(*job)) : c.idx == 0;
        };
        uint32_t best = 0;
        for (auto& c : sv.cards) {
            if (!eligible(*c)) continue;
            std::lock_guard<std::mutex> lk(c->mu);
            for (uint32_t s = 0; s < c->slots; ++s) {
                if (c->owner[s]) continue;
                const uint32_t L = ink2_lcp(c->live[s], job->keyed);
                if (L > best) { best = L; card = c.get(); }
            }
        }
        if (best < 64) {
            card = nullptr;
            for (auto& c : sv.cards)
                if (eligible(*c) && (!card || c->load.load() < card->load.load())) card = c.get();
        }
        if (!card) { r0.finish_reason = "context_length_exceeded"; return r0; }
        card->load.fetch_add(1);
        std::lock_guard<std::mutex> lk(card->mu);
        card->waiting.push_back(job.get());
    }
    card->cv.notify_all();
    Ink2Job& j = *job;
    std::unique_lock<std::mutex> lk(j.mu);
    for (;;) {
        j.cv.wait_for(lk, std::chrono::milliseconds(100), [&] { return !j.out.empty() || j.done; });
        std::string piece = std::move(j.out);
        j.out.clear();
        const bool d = j.done;
        lk.unlock();
        if (on_token && !j.cancel.load()) {
            if (!piece.empty()) { if (!on_token(piece)) j.cancel.store(true); }
            else if (!d && !on_token(std::string_view{})) j.cancel.store(true);   // the liveness probe
        }
        if (j.cancel.load()) card->cv.notify_all();
        lk.lock();
        if (d && j.out.empty()) break;
    }
    lk.unlock();
    GenerateResult r = std::move(j.r);
    if (j.cancel.load() && r.finish_reason != "stop") r.finish_reason = "abort";
    return r;
}


// Start the per-card loops: the slot context (--slot-ctx, else min(--ctx, 32768)), the slots that fit each card (the
// memory rule: free device memory after the model, less 1 GiB, over one slot's bytes), card 1+ loaded with their own copy.
std::string inkling2_serve_start_impl(const std::string& dir, Inkling2Bundle& b, const std::vector<sycl::device>& cards, uint32_t n_cards,
                                 const Inkling2LoadOptions& lo, uint32_t asked, uint32_t slot_ctx_opt, uint32_t max_ctx,
                                 uint32_t& total, uint32_t& slot_ctx_out) {
    auto sv = std::make_shared<Ink2Serve>();
    const Inkling2Config& c = b.model.config();
    const uint32_t sctx = std::max<uint32_t>(512, std::min<uint32_t>(max_ctx, slot_ctx_opt ? slot_ctx_opt : 32768u));
    // Main lane (default with --parallel > 1 on 2+ cards and --ctx above the lanes' size; IE_INKLING2_MAIN_LANE=0 = off):
    // card 0 holds ONE lane at the full --ctx (or the most that fits) for the harness's main conversation, the other cards
    // the remaining lanes at the most positions that fit them (up to --ctx). Without it every lane was 32768, the main
    // conversation's too (a harness then compacts the owner's chat to 32K).
    const char* mlv = std::getenv("IE_INKLING2_MAIN_LANE");
    // --slot-ctx then sizes the sub-agents' lanes only (Dream's "sub-agent context": every lane after the first)
    const bool main_lane = asked >= 2 && n_cards >= 2 && max_ctx > sctx && !(mlv && std::string(mlv) == "0");
    sv->main_lane = main_lane;
    if (!main_lane && !slot_ctx_opt && max_ctx > sctx)   // said, not silent: a harness reading ctx sees the slot's
        std::fprintf(stderr, "[inkling2] NOTE: each of the %u conversation slots holds %u positions, not --ctx %u; "
                     "--slot-ctx N gives every slot more (fewer slots then fit a card)\n", asked, sctx, max_ctx);
    // slot prompt chunk: 256 on a card where it keeps every asked slot (the matrix-engine experts fill their tiles: a
    // 25K prompt arriving while 5 slots decode is read in 98 s instead of 120 at the default share, the others at 49 %
    // of their speed instead of 52 %), else 128; IE_INKLING2_SLOT_CHUNK sets it
    const char* scv = std::getenv("IE_INKLING2_SLOT_CHUNK");
    const bool chunk_set = scv && *scv;
    uint32_t chunk = chunk_set ? uint32_t(std::strtoul(scv, nullptr, 10)) : 128u;
    chunk = std::max<uint32_t>(16, std::min(chunk, b.chunk));
    const size_t per = Inkling2Model::batch_slot_bytes(c, sctx, chunk, b.model.batch_kv_bf16());
    const uint32_t big = std::min<uint32_t>(256, b.chunk);
    const size_t per_big = Inkling2Model::batch_slot_bytes(c, sctx, big, b.model.batch_kv_bf16());
    const uint64_t margin = 1ull << 30;
    total = 0;
    for (uint32_t k = 0; k < n_cards; ++k) {
        auto cd = std::make_unique<Ink2Card>();
        cd->b = &b; cd->idx = k; cd->ctx = sctx; cd->chunk = chunk;
        size_t cper = per;
        if (k == 0) { cd->q = b.q.get(); cd->model = &b.model; }
        else {
            cd->own_q = std::make_unique<sycl::queue>(sycl::context(cards[k]), cards[k], sycl::property::queue::in_order{});
            cd->own_model = std::make_unique<Inkling2Model>();
            if (auto e = cd->own_model->load(dir, *cd->own_q, lo); !e.empty()) return "inkling2 card " + std::to_string(k) + ": " + e;
            cd->q = cd->own_q.get(); cd->model = cd->own_model.get();
        }
        cd->name = "card " + std::to_string(k);
        const sycl::device dv = cd->q->get_device();
        uint32_t want = asked / n_cards + (k < asked % n_cards ? 1u : 0u);
        if (main_lane) {   // card 0: the main lane; cards 1+: the other asked - 1 lanes, spread
            const uint32_t rest = asked - 1, others = n_cards - 1;
            want = k == 0 ? 1u : rest / others + (k - 1 < rest % others ? 1u : 0u);
            if (dv.has(sycl::aspect::ext_intel_free_memory)) {
                const uint64_t fr = dv.get_info<sycl::ext::intel::info::device::free_memory>();
                const uint32_t ch = chunk_set ? chunk : big;
                uint32_t got = 0;                                         // the most positions (1024 steps) that fit `want` lanes
                const uint32_t top = (k > 0 && slot_ctx_opt) ? std::min(slot_ctx_opt, max_ctx) : max_ctx;   // sub-agents: --slot-ctx
                for (uint32_t t = top & ~1023u; t >= 4096 && !got; t -= 1024)
                    if (uint64_t(want) * Inkling2Model::batch_slot_bytes(c, t, ch, b.model.batch_kv_bf16()) + margin <= fr) got = t;
                if (got && k > 0 && slot_ctx_opt && got < (top & ~1023u))
                    std::fprintf(stderr, "[inkling2] %s: %u sub-agent lanes of --slot-ctx %u do not fit: %u positions each\n",
                                 cd->name.c_str(), want, slot_ctx_opt, got);
                if (got >= 4096) {
                    cd->ctx = std::min(got, max_ctx);
                    cd->chunk = ch;
                    cper = Inkling2Model::batch_slot_bytes(c, cd->ctx, ch, b.model.batch_kv_bf16());
                }
            }
        }
        const uint32_t cctx = cd->ctx;
        const size_t cper0 = main_lane ? cper : per;
        uint32_t fit = want;
        std::string why;
        if (main_lane && dv.has(sycl::aspect::ext_intel_free_memory)) {
            const uint64_t fr = dv.get_info<sycl::ext::intel::info::device::free_memory>();
            fit = fr > margin ? uint32_t(std::min<uint64_t>(want, (fr - margin) / cper0)) : 0u;
            why = std::to_string(fr >> 20) + " MiB free after the model";
        } else if (dv.has(sycl::aspect::ext_intel_free_memory)) {
            const uint64_t fr = dv.get_info<sycl::ext::intel::info::device::free_memory>();
            fit = fr > margin ? uint32_t(std::min<uint64_t>(want, (fr - margin) / per)) : 0u;
            why = std::to_string(fr >> 20) + " MiB free after the model";
            if (!chunk_set && big > chunk && fr > margin && (fr - margin) / per_big >= want) { cd->chunk = big; cper = per_big; }
        } else why = "free memory not reported";
        if (fit == 0) return "inkling2 " + cd->name + ": not even one conversation slot of " + std::to_string(cctx) +
                             " positions fits (" + std::to_string(cper0 >> 20) + " MiB each, " + why + "); lower --slot-ctx";
        if (fit < want)
            std::fprintf(stderr, "[inkling2] %s: %u slots of %u positions do NOT fit (%zu MiB each, %s, 1 GiB kept free): serving %u\n",
                         cd->name.c_str(), want, cctx, cper0 >> 20, why.c_str(), fit);
        if (auto e = cd->model->batch_begin(fit, cctx, cd->chunk); !e.empty()) return "inkling2 " + cd->name + ": " + e;
        cd->slots = fit;
        cd->live.assign(fit, {});
        cd->owner.assign(fit, nullptr);
        cd->used_at.assign(fit, 0);
        total += fit;
        std::fprintf(stderr, "[inkling2] %s (%s): %u conversation slots x %u positions (%zu MiB each, keys / values %s, prompt chunk %u, "
                             "%.2f GiB on the card)%s\n", cd->name.c_str(), dv.get_info<sycl::info::device::name>().c_str(), fit, cctx, cper >> 20,
                     b.model.batch_kv_bf16() ? "bf16" : "fp32", cd->chunk, (cd->model->device_bytes()) / 1073741824.0,
                     main_lane ? (k == 0 ? " = the MAIN lane (requests with \"ie_lane\": \"main\", and any prompt too long for the others)"
                                         : " = the sub-agents' lanes") : "");
        sv->cards.push_back(std::move(cd));
    }
    for (auto& cd : sv->cards) cd->th = std::thread([p = cd.get()] { p->loop(); });
    slot_ctx_out = 0;
    for (const auto& cd : sv->cards) slot_ctx_out = std::max(slot_ctx_out, cd->ctx);   // the main lane's, when there is one
    b.serve = std::move(sv);
    return {};
}
std::string inkling2_serve_status(const Inkling2Bundle& b) {
    if (!b.serve) return {};
    json j;
    uint32_t act = 0, wait = 0;
    json cards = json::array();
    for (const auto& c : b.serve->cards) {
        const uint32_t a = c->active.load(), l = c->load.load();
        act += a; wait += l > a ? l - a : 0;
        cards.push_back({{"card", c->idx}, {"slots", c->slots}, {"active", a}, {"waiting", l > a ? l - a : 0},
                         {"steps", c->steps.load()}, {"rows", c->rows.load()}, {"step_ms", std::round(c->step_ms.load() * 10) / 10}});
    }
    j["lanes_active"] = act;
    j["lanes_waiting"] = wait;
    j["inkling2_cards"] = cards;
    return dump_c(j);
}
uint32_t inkling2_serve_slots(const Inkling2Bundle& b) { return b.serve ? b.serve->total_slots() : 0u; }


// ------------------------------------------------------------------------------------------------------------ the engine

bool Engine::inkling2_dir(const std::string& path) {
    std::ifstream f(path + "/config.json");
    if (!f) return false;
    const json j = json::parse(f, nullptr, false);
    return !j.is_discarded() && j.is_object() && j.value("model_type", "") == "inkling2_loop";
}

std::string Engine::inkling2_load(const std::string& dir) {
    auto b = std::make_unique<Inkling2Bundle>();
    const auto t0 = std::chrono::steady_clock::now();
    if (auto e = b->tok.load_from_hf_json(dir + "/tokenizer.json", dir + "/tokenizer_config.json"); !e.empty()) return "inkling2 tokenizer: " + e;
    for (const auto& [sp, mk] : kMarkSpell) {
        const int32_t id = b->tok.find_token(sp);
        if (id < 0) return "inkling2: the tokenizer lacks the chat-format marker " + std::string(sp);
        b->marks.emplace_back(id, mk);
    }
    for (const char* s : {"<|content_model_end_sampling|>", "<|endoftext|>", "<|message_user|>", "<|message_system|>", "<|message_tool|>"})
        if (const int32_t v = b->tok.find_token(s); v >= 0) b->stop.push_back(v);
    {   // generation_config.json's eos_token_id (an int or a list) too
        std::ifstream g(dir + "/generation_config.json");
        const json j = g ? json::parse(g, nullptr, false) : json();
        if (j.is_object() && j.contains("eos_token_id")) {
            const json& e = j["eos_token_id"];
            auto add = [&](const json& x) { if (x.is_number_integer() && std::find(b->stop.begin(), b->stop.end(), x.get<int32_t>()) == b->stop.end()) b->stop.push_back(x.get<int32_t>()); };
            if (e.is_array()) for (const auto& x : e) add(x); else add(e);
        }
    }
    const char* ot = std::getenv("IE_INKLING2_OPEN_TURN");
    b->open_turn = ot && *ot == '1';

    std::vector<sycl::device> cards;
    for (const auto& pl : sycl::platform::get_platforms()) {
        if (pl.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : pl.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) cards.push_back(d);
    }
    if (cards.empty()) return "inkling2: no Arc GPU";
    // --parallel N > 1: many conversations, continuously batched (Ink2Serve); with --gpus 2 a copy of the model on each of
    // two cards, the conversations spread over both. --parallel 1 (and the default): the one-request path, as before.
    const bool many = opts_.parallel != kLanesAuto && opts_.parallel > 1;
    const uint32_t n_cards = many ? std::min<uint32_t>(std::max<uint32_t>(1, opts_.n_gpus), uint32_t(cards.size())) : 1u;
    if (many && opts_.n_gpus > n_cards)
        std::fprintf(stderr, "[inkling2] --gpus %u: %zu card(s) visible: serving on %u\n", opts_.n_gpus, cards.size(), n_cards);
    // --gpus 2 with an explicit --parallel 1: both cards on each token of the one conversation (Inkling2LoadOptions::tp_queue;
    // the same values as one card, ~1.3x the decode speed). IE_INKLING2_TP=0: one card. The default launch is unchanged.
    const char* tpe = std::getenv("IE_INKLING2_TP");
    const bool tp = !many && opts_.parallel == 1 && opts_.n_gpus >= 2 && cards.size() >= 2 && !(tpe && *tpe == '0');
    if (many && n_cards > 1)
        std::fprintf(stderr, "[inkling2] --gpus %u --parallel %u: a copy of the model on each card, about %u conversations each "
                             "(--gpus 2 --parallel 1 instead puts both cards on each token of one conversation)\n",
                     n_cards, opts_.parallel, (opts_.parallel + n_cards - 1) / n_cards);
    if (tp)
        std::fprintf(stderr, "[inkling2] --gpus %u --parallel 1: two cards on one conversation (attention heads, experts and the "
                             "unembed split between them; IE_INKLING2_TP=0 = one card)\n", opts_.n_gpus);
    else if (!many && opts_.n_gpus > 1)
        std::fprintf(stderr, "[inkling2] --gpus %u: this runtime runs on ONE card (the first visible; ZE_AFFINITY_MASK picks it); "
                             "--parallel N > 1 puts a copy on each card, an explicit --parallel 1 splits each token over two\n", opts_.n_gpus);
    opts_.n_gpus = tp ? 2u : n_cards;
    if (opts_.parallel == kLanesAuto) {
        opts_.parallel = 1;
        std::fprintf(stderr, "%s\n", lanes_auto_line("inkling2", 1, 0, "no request lanes for this arch: --parallel N > 1 serves N").c_str());
    }
    if (tp) {
        const sycl::context ctx2(std::vector<sycl::device>{cards[0], cards[1]});
        b->q = std::make_unique<sycl::queue>(ctx2, cards[0], sycl::property::queue::in_order{});
        b->q2 = std::make_unique<sycl::queue>(ctx2, cards[1], sycl::property::queue::in_order{});
    } else
    b->q = std::make_unique<sycl::queue>(sycl::context(cards[0]), cards[0], sycl::property::queue::in_order{});

    Inkling2LoadOptions lo;
    b->chunk = opts_.prefill_chunk_set ? std::max<uint32_t>(1, std::min<uint32_t>(opts_.prefill_chunk, 2048)) : 256u;
    // One conversation (2026-11-23): prompt chunks of 512 read 8-10 % faster than 256 (matrix-engine experts fill their
    // tiles) for ~0.6 GiB more scratch. Taken when the card's free memory covers the model file (weights + vision
    // tower, an upper bound of what lands on the card), the state at --ctx and the 512-row scratch, plus 1 GiB kept free;
    // else 256. --prefill-chunk sets it; the slot count and context are never reduced for it.
    if (!opts_.prefill_chunk_set && !many) {
        Inkling2Config cc;
        std::string why;
        if (inkling2_read_config(dir, cc).empty()) {
            uint64_t file = 0;
            std::error_code ec;
            for (const auto& de : std::filesystem::directory_iterator(dir, ec))
                if (de.path().extension() == ".safetensors" && de.path().filename().string().rfind("model", 0) == 0) file += de.file_size(ec);
            const uint32_t rb = std::max<uint32_t>(2048, 512);
            const uint64_t need = file + Inkling2Model::state_bytes_for(cc, opts_.max_ctx, 512, rb, false) +
                                  Inkling2Model::scratch_bytes_for(cc, 512, false) + (1ull << 30);
            uint64_t fr = UINT64_MAX;
            bool known = true;
            for (uint32_t k = 0; k < (tp ? 2u : 1u); ++k) {
                if (!cards[k].has(sycl::aspect::ext_intel_free_memory)) { known = false; break; }
                fr = std::min<uint64_t>(fr, cards[k].get_info<sycl::ext::intel::info::device::free_memory>());
            }
            if (known && fr >= need) b->chunk = 512;
            why = known ? std::to_string(fr >> 20) + " MiB free, " + std::to_string(need >> 20) + " MiB needed at chunk 512"
                        : "free memory not reported";
        } else why = "config not readable";
        std::fprintf(stderr, "[inkling2] prompt chunk %u (%s; --prefill-chunk sets it)\n", b->chunk, why.c_str());
    }
    lo.max_chunk = b->chunk;
    // The prompt cache rewinds the state to the longest common prefix with the previous request; truncate() reaches back
    // `rollback` positions behind the furthest one written (sliding-window key rings and conv-input rings sized for it).
    // A follow-up whose history drops the reply's reasoning shares only the previous PROMPT, so the rewind is the whole
    // previous reply: 2048 covers most replies (IE_INKLING2_ROLLBACK sets it; at least the prompt chunk).
    b->rollback = 2048;
    if (const char* rb = std::getenv("IE_INKLING2_ROLLBACK"); rb && *rb) b->rollback = uint32_t(std::strtoul(rb, nullptr, 10));
    b->rollback = std::max(b->rollback, b->chunk);
    if (many) b->rollback = b->chunk;                       // the one-request state is unused: its smallest rings
    ink2_set_rollback(lo, b->rollback);
    lo.tp_queue = b->q2.get();
    if (auto e = b->model.load(dir, *b->q, lo); !e.empty()) return "inkling2: " + e;
    std::fprintf(stderr, "[inkling2] %s; this load uses the %s\n", b->model.kv_note().c_str(),
                 many ? "batch slots (--parallel N > 1)" : "one-conversation state (--parallel 1)");
    {   // the vision tower on card 0 (its own queue in the model's context); IE_INKLING2_VISION=0: text only
        const char* ve = std::getenv("IE_INKLING2_VISION");
        const char* vt = std::getenv("IE_INKLING2_VIS_MAX_TOKENS");
        const uint32_t vmax = vt && *vt ? uint32_t(std::max(16ul, std::strtoul(vt, nullptr, 10))) : 4096u;
        if (!(ve && *ve == '0')) {
            b->vq = std::make_unique<sycl::queue>(b->q->get_context(), b->q->get_device(), sycl::property::queue::in_order{});
            auto vis = std::make_shared<Inkling2Vision>();
            // resident scratch for 1024 image tokens (IE_INKLING2_VIS_RESIDENT_TOKENS; 0 = all of vmax): weights 0.6 GiB +
            // 0.17 GiB instead of 1.28 GiB with the 4096-token scratch; larger images grow it for their encode
            const char* vr = std::getenv("IE_INKLING2_VIS_RESIDENT_TOKENS");
            const uint32_t vres = vr && *vr ? uint32_t(std::strtoul(vr, nullptr, 10)) : 1024u;
            if (auto e = vis->init(*b->vq, dir, vmax, vres); !e.empty())
                std::fprintf(stderr, "[vision] inkling2: the vision tower did not load (%s): TEXT-ONLY, image requests are refused with a 400\n", e.c_str());
            else {
                b->vis = std::move(vis);
                b->vis_max_tokens = vmax;
                std::fprintf(stderr, "[vision] inkling2: vision tower loaded on card 0 (%.2f GiB): images up to %u image tokens of 40 x 40 px "
                                     "(larger ones are scaled down; IE_INKLING2_VIS_MAX_TOKENS sets it), in user and tool turns\n",
                             b->vis->device_bytes() / 1073741824.0, vmax);
            }
        } else std::fprintf(stderr, "[vision] inkling2: IE_INKLING2_VISION=0: TEXT-ONLY, image requests are refused with a 400\n");
    }
    if (!b->model.ablation_summary().empty())   // <dir>/ie_ablation.json (or IE_INKLING2_ABLATION); IE_INKLING2_ABLATION=off = none
        std::fprintf(stderr, "[inkling2] ablation set ON (directions removed from the stream at run time; weights unchanged): %s\n",
                     b->model.ablation_summary().c_str());
    b->ctx = opts_.max_ctx;
    b->reuse = opts_.prompt_cache;
    if (many) {
        uint32_t total = 0, sctx = 0;
        if (auto e = inkling2_serve_start_impl(dir, *b, cards, n_cards, lo, opts_.parallel, opts_.slot_ctx, opts_.max_ctx, total, sctx); !e.empty())
            return e;
        if (total < opts_.parallel)
            std::fprintf(stderr, "[inkling2] --parallel %u: %u conversation slots fit; serving %u at once (more requests wait)\n",
                         opts_.parallel, total, total);
        opts_.parallel = total;                             // /props total_slots
        opts_.max_ctx = sctx;                               // /props n_ctx: the largest lane (the main lane's, when there is one)
        lane_ctx_ = sctx;                                   // /props slot_ctx: the sub-agents' lanes (Dream sizes sub-agents by it)
        if (b->serve && b->serve->main_lane) {
            uint32_t sub_ctx = 0;
            for (const auto& cd : b->serve->cards) if (cd->idx != 0) sub_ctx = sub_ctx ? std::min(sub_ctx, cd->ctx) : cd->ctx;
            if (sub_ctx) lane_ctx_ = sub_ctx;
        }
        b->ctx = sctx;
    } else if (auto e = b->model.reserve(b->ctx); !e.empty()) return "inkling2: reserving the state for ctx " + std::to_string(b->ctx) + ": " + e;
    if (!many) {
        const Inkling2Config& c = b->model.config();
        std::fprintf(stderr, "[inkling2] state for ctx %u: %.2f GiB with rollback %u (%.2f GiB at the minimum rollback %u)\n", b->ctx,
                     Inkling2Model::state_bytes_for(c, b->ctx, b->chunk, b->rollback, b->model.kv_bf16()) / 1073741824.0, b->rollback,
                     Inkling2Model::state_bytes_for(c, b->ctx, b->chunk, b->chunk, b->model.kv_bf16()) / 1073741824.0, b->chunk);
    }
    b->reuse = opts_.prompt_cache;
    prompt_cache_on_ = opts_.prompt_cache;
    // the owner's rule (IE_VISION_REQUIRED): a model should see; without the tower it is said, not refused
    if (!b->vis && vision_required_env())
        std::fprintf(stderr, "[vision] inkling2: IE_VISION_REQUIRED=1 is set and this load is text-only: noted, the load goes on\n");
    std::fprintf(stderr, "[inkling2] ready in %.1f s: Inkling 2 Mini on %s (%s), ctx %u, prompt chunk %u, %.2f GiB on the card, prompt cache %s (%s), "
                         "generation prompt: %s\n",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                 many ? (std::to_string(n_cards) + " card(s), " + std::to_string(opts_.parallel) + " conversation slots").c_str()
                      : tp ? "two cards, one conversation" : "one card",
                 (cards[0].get_info<sycl::info::device::name>() + (cards[0].has(sycl::aspect::ext_intel_pci_address)
                      ? " " + cards[0].get_info<sycl::ext::intel::info::device::pci_address>() : std::string())).c_str(), b->ctx, b->chunk, b->model.device_bytes() / 1073741824.0,
                 b->reuse ? "on" : "off", kInk2Truncate ? "rewinds with truncate()" : "extends the live sequence only",
                 b->open_turn ? "<|message_model|> appended (IE_INKLING2_OPEN_TURN=1)" : "none (the model opens its turn, as the reference)");
    arch_ = ModelArch::kInkling2;
    inkling2_ = std::move(b);
    return {};
}

GenerateResult Engine::inkling2_generate(const std::string& prompt, const SamplingParams& sp, const TokenCallback& on_token) {
    if (inkling2_->serve) return ink2_serve_run(*inkling2_, inkling2_->tok.encode(prompt, /*allow_special=*/true), sp, on_token, nullptr);
    std::lock_guard<std::mutex> lk(inkling2_mu_);
    const auto ids = inkling2_->tok.encode(prompt, /*allow_special=*/true);
    return inkling2_run(*inkling2_, ids, sp, on_token, nullptr);
}

GenerateResult Engine::inkling2_chat(std::span<const ChatTurn> turns, const SamplingParams& sp, const TokenCallback& on_token,
                                     bool enable_thinking, std::string_view tools_json, std::string_view reasoning_effort) {
    std::unique_lock<std::mutex> lk(inkling2_mu_, std::defer_lock);
    if (!inkling2_->serve) lk.lock();                     // the one-request path: one at a time; the slots: their own loops
    Inkling2Bundle& b = *inkling2_;
    GenerateResult r;
    std::string err;
    std::vector<std::pair<size_t, size_t>> content;   // known_bugs 45: the renderer's own request-text ranges
    // images: decoded and preprocessed first (their token counts shape the prompt), encoded after tokenising
    std::vector<Inkling2Image> imgs;
    std::vector<uint32_t> img_tokens;
    std::vector<uint64_t> img_hash;
    // the tower's scratch grows for this request's largest image and shrinks after (the lock spans both); when the card
    // has no room for it the image is scaled down to the resident scratch (logged), never refused
    std::unique_lock<std::mutex> vlk(b.vis_mu, std::defer_lock);
    struct VisShrink {
        Inkling2Bundle& b; std::unique_lock<std::mutex>& l;
        ~VisShrink() { if (l.owns_lock() && b.vis) b.vis->shrink(); }
    } vis_shrink{b, vlk};
    for (const auto& t : turns)
        for (const auto& bytes : t.images) {
            if (!b.vis) { r.finish_reason = "error: inkling2: images are not supported by this load (the vision tower is not loaded)"; return r; }
            Inkling2Image im;
            bool resized = false;
            if (auto e = inkling2_load_image(bytes.data(), bytes.size(), im, b.vis_max_tokens, &resized); !e.empty()) {
                r.finish_reason = "error: inkling2: " + e; return r;
            }
            if (resized) std::fprintf(stderr, "[inkling2] an image was scaled down to %u x %u tokens (the tower takes %u)\n", im.rows, im.cols, b.vis_max_tokens);
            if (!vlk.owns_lock()) vlk.lock();
            if (im.tokens() > b.vis->capacity() && !b.vis->grow(im.tokens())) {
                const uint32_t cap = b.vis->capacity();
                if (auto e = inkling2_load_image(bytes.data(), bytes.size(), im, cap, &resized); !e.empty()) { r.finish_reason = "error: inkling2: " + e; return r; }
                std::fprintf(stderr, "[inkling2] no room on the card for the tower's scratch at the image's size: scaled down to %u x %u tokens "
                                     "(the resident scratch takes %u)\n", im.rows, im.cols, cap);
            }
            uint64_t h = 1469598103934665603ull;                    // FNV-1a over the bytes: the prompt cache's image key
            for (unsigned char ch : bytes) { h ^= ch; h *= 1099511628211ull; }
            img_tokens.push_back(im.tokens());
            img_hash.push_back(h);
            imgs.push_back(std::move(im));
        }
    const std::string prompt = inkling2_render_prompt(turns, tools_json, enable_thinking, reasoning_effort, err, &content, b.open_turn,
                                                      b.vis ? &img_tokens : nullptr);
    if (!err.empty()) { r.finish_reason = "error: " + err; return r; }
    const auto ids = b.tok.encode(prompt, /*allow_special=*/true, content);   // a spelling in the request's text stays text
    Ink2Images ii;
    if (!imgs.empty()) {
        for (uint32_t i = 0; i < ids.size(); ++i) if (ids[i] == kInk2ImageToken) ii.at.push_back(i);
        size_t want = 0;
        for (uint32_t n : img_tokens) want += n;
        if (ii.at.size() != want) { r.finish_reason = "error: inkling2: internal: " + std::to_string(ii.at.size()) + " placeholders for " + std::to_string(want) + " image tokens"; return r; }
        const auto tv = std::chrono::steady_clock::now();
        size_t k = 0;
        for (size_t i = 0; i < imgs.size(); ++i) {
            std::vector<float> rows;
            if (auto e = b.vis->encode(imgs[i], rows); !e.empty()) { b.vis->shrink(); r.finish_reason = "error: inkling2: " + e; return r; }
            ii.rows.insert(ii.rows.end(), rows.begin(), rows.end());
            for (uint32_t t = 0; t < img_tokens[i]; ++t, ++k)
                ii.key.push_back(-1 - int32_t((img_hash[i] * 31 + t) & 0x3fffffffu));
        }
        b.vis->shrink();
        vlk.unlock();
        std::fprintf(stderr, "[inkling2] %zu image(s), %zu image tokens, encoded in %.0f ms\n", imgs.size(), want,
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tv).count());
    }
    const Ink2Images* iip = ii.at.empty() ? nullptr : &ii;
    Inkling2ReplyParser parser(enable_thinking);
    if (b.open_turn) parser.feed(Ink2Mark::kModel);
    r = b.serve ? ink2_serve_run(b, ids, sp, on_token, &parser, iip) : inkling2_run(b, ids, sp, on_token, &parser, iip);
    if (r.finish_reason.starts_with("error") || r.finish_reason == "context_length_exceeded") return r;
    r.text = parser.content();
    r.reasoning_content = parser.reasoning();
    r.tool_calls_json = parser.tool_calls_json();
    if (parser.n_calls() && r.finish_reason == "stop") r.finish_reason = "tool_calls";
    if (r.finish_reason == "length") r.truncated_tool_call = parser.truncated_call();
    std::string eff, eerr;
    inkling2_effort_text(reasoning_effort, enable_thinking, eff, eerr);
    std::fprintf(stderr, "[inkling2] %sprompt %u (cached %u), \"Thinking effort level: %s\", reply %u tokens, finish %s%s\n",
                 b.serve ? ("card " + std::to_string(r.lane / 1000) + " slot " + std::to_string(r.lane % 1000) + ": ").c_str() : "",
                 r.prompt_tokens, r.cached_tokens, eff.c_str(), r.completion_tokens, r.finish_reason.c_str(), parser.n_calls() ? (", " + std::to_string(parser.n_calls()) + " tool call(s)").c_str() : "");
    return r;
}

}  // namespace ie
