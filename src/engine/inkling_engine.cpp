// src/engine/inkling_engine.cpp — Inkling-Small (`inkling`) in the engine: load, the chat format, generate, parse
// (docs/inkling/00_PORT_PLAN.md P5). The runtime is InklingModel (src/model/inkling.cpp, one card); the tokenizer is the GGUF's
// (pre "inkling"); the template is the GGUF's chat_template rendered natively (inkling_render mirrors it block for block).
//
// The format: messages are <|message_user|> / <|message_model|> / <|message_system|> / <|message_tool|>{name} blocks, each one
// content kind (<|content_text|>, <|content_thinking|>, <|content_invoke_tool_json|>{"name":..,"args":{..}}) closed by
// <|end_message|>; a model turn ends with <|content_model_end_sampling|> (the EOS). Thinking is a system line, "Thinking
// effort level: E" (0.9 by default, 0 = off), before the first non-system message. The server streams Inkling as a
// structured arch: the stream is translated on the fly (thinking text, then "</think>", the text blocks, a tool block as
// "<tool_call>...</tool_call>") and the raw completion is parsed at the end into content, reasoning and tool calls.
#include "ie/engine.hpp"

#include "ie/deepseek41_generate.hpp"   // Ds41Generator::sample (the host sampler the other dir runtimes use)
#include "ie/inkling_engine.hpp"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

namespace ie {

namespace {
using json = nlohmann::json;   // std::map objects: dump() is the template's canonical_json (sorted keys, ',' ':')

size_t inkling_utf8_complete(std::string_view s) {
    if (s.empty()) return 0;
    size_t i = s.size(), cont = 0;
    while (i > 0 && cont < 3 && (uint8_t(s[i - 1]) & 0xC0u) == 0x80u) { --i; ++cont; }
    if (i == 0) return 0;
    const uint8_t lead = uint8_t(s[i - 1]);
    const size_t need = (lead & 0x80u) == 0 ? 1 : (lead & 0xE0u) == 0xC0u ? 2 : (lead & 0xF0u) == 0xE0u ? 3 : (lead & 0xF8u) == 0xF0u ? 4 : 1;
    return (cont + 1 >= need) ? i - 1 + need : i - 1;
}

std::string canon(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

// The template's reasoning_effort_text: a level name or a number in [0, 0.99], printed like Python's f"{x:.2f}" with
// trailing zeros dropped ("0", "0.1", "0.9", "0.99"). Thinking off = 0.
bool effort_text(std::string_view effort, bool thinking, std::string& out, std::string& err) {
    if (!thinking) { out = "0"; return true; }
    std::string e(effort);
    for (auto& c : e) c = char(std::tolower(uint8_t(c)));
    double v = 0.9;
    if (e.empty() || e == "high") v = 0.9;
    else if (e == "none") v = 0.0;
    else if (e == "minimal") v = 0.1;
    else if (e == "low") v = 0.2;
    else if (e == "medium") v = 0.7;
    else if (e == "xhigh" || e == "max") v = 0.99;
    else {
        char* end = nullptr;
        v = std::strtod(e.c_str(), &end);
        if (!end || *end || !(v >= 0.0 && v <= 0.99)) {
            err = "inkling: reasoning_effort '" + std::string(effort) + "' (none/minimal/low/medium/high/xhigh/max or 0.0-0.99)";
            return false;
        }
    }
    const long hund = std::lround(v * 100.0);
    if (hund == 0) out = "0";
    else if (hund % 10 == 0) out = "0." + std::to_string(hund / 10);
    else out = (hund < 10 ? "0.0" : "0.") + std::to_string(hund);
    return true;
}

// Prefill (reusing the live sequence's longest restartable prefix) + sampling loop. r.text is the raw completion; on_token
// sees the translated stream (the file header).
GenerateResult inkling_run(InklingBundle& b, const std::vector<int32_t>& ids, const SamplingParams& sp, const TokenCallback& on_token,
                           bool chat, bool thinking) {
    GenerateResult r;
    const uint32_t cap = b.model.max_ctx();
    r.prompt_tokens = uint32_t(ids.size());
    if (ids.empty()) { r.finish_reason = "error: empty prompt"; return r; }
    if (ids.size() >= cap) { r.finish_reason = "context_length_exceeded"; return r; }
    const uint32_t room = cap - uint32_t(ids.size());
    const uint32_t max_new = sp.max_tokens == 0 ? room : std::min(sp.max_tokens, room);

    const auto t_pf = std::chrono::steady_clock::now();
    uint32_t L = 0;
    if (b.reuse) while (L < b.live.size() && L + 1 < ids.size() && b.live[L] == ids[L]) ++L;   // at least one row to read
    L = b.model.restart_at(L);
    b.live.resize(L);
    r.cached_tokens = L;
    r.cache_source = L ? "live" : "none";
    std::vector<float> logits(b.cfg.vocab);
    // the prompt in chunks (two cards: overlapped); the liveness probe runs as each chunk is in
    auto alive = [&](uint32_t) { return !on_token || on_token(std::string_view{}); };
    if (auto e = b.model.prefill(ids.data() + L, uint32_t(ids.size()) - L, L, logits.data(), alive); !e.empty()) {
        b.live.clear(); b.model.restart_at(0);
        r.finish_reason = "error: inkling prefill: " + e; return r;
    }
    b.live.assign(ids.begin(), ids.begin() + b.model.pos());
    if (b.model.pos() < ids.size()) { r.finish_reason = "abort"; return r; }   // the client went away mid-prompt
    r.prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pf).count();

    Ds41SampleParams p; p.temperature = sp.temperature; p.top_k = sp.top_k; p.top_p = sp.top_p; p.min_p = sp.min_p;
    p.repeat_penalty = sp.repeat_penalty; p.repeat_window = sp.repeat_window;
    p.seed = sp.seed ? sp.seed : uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
    uint64_t rng = p.seed;
    std::vector<int32_t> recent(ids.end() - std::min<size_t>(ids.size(), 512), ids.end());
    std::string pending;                                    // undelivered stream text (an incomplete UTF-8 tail)
    enum { kNone, kThinking, kTexting, kCalling } kind = kNone;
    bool closed = false;                                    // "</think>" sent
    auto emit = [&](std::string_view s) { pending += s; };
    auto flush = [&]() -> bool {
        const size_t done = inkling_utf8_complete(pending);
        if (!done) return true;
        const std::string piece = pending.substr(0, done);
        pending.erase(0, done);
        return !on_token || on_token(piece);
    };
    const auto t_dec = std::chrono::steady_clock::now();
    r.finish_reason = "length";
    auto finite = [&](uint32_t pos) {
        for (float v : logits) if (std::isnan(v) || v == INFINITY) {
            std::fprintf(stderr, "[inkling] non-finite logits at position %u -- not sampled; the request fails and the conversation is cleared\n", pos);
            b.live.clear(); b.model.restart_at(0);
            r.finish_reason = "error: inkling: non-finite logits at position " + std::to_string(pos);
            return false;
        }
        return true;
    };
    if (!finite(uint32_t(ids.size()) - 1)) return r;
    int32_t id = Ds41Generator::sample(logits, recent, p, rng);
    for (uint32_t k = 0;;) {
        if (!sp.ignore_eos && std::find(b.stop.begin(), b.stop.end(), id) != b.stop.end()) { r.finish_reason = "stop"; break; }
        ++r.completion_tokens;
        recent.push_back(id);
        if (recent.size() > 512) recent.erase(recent.begin());
        const std::string piece = b.tok.decode(std::vector<int32_t>{id});
        r.text += piece;
        if (!chat) emit(piece);                             // raw generate: the text as it comes
        else if (id == b.msg_model) kind = kNone;
        else if (id == b.c_thinking) kind = kThinking;
        else if (id == b.c_text) { if (thinking && !closed) { emit("</think>"); closed = true; } kind = kTexting; }
        else if (id == b.c_tool) { if (thinking && !closed) { emit("</think>"); closed = true; } emit("<tool_call>"); kind = kCalling; }
        else if (id == b.end_msg) {
            if (kind == kThinking && thinking && !closed) { emit("</think>"); closed = true; }
            if (kind == kCalling) emit("</tool_call>");
            kind = kNone;
        } else if (!b.tok.is_special(id)) {
            if ((kind == kThinking && thinking) || kind == kTexting || kind == kCalling) emit(piece);
        }
        if (!flush()) { r.finish_reason = "abort"; break; }
        if (++k == max_new) break;
        const uint32_t pos = uint32_t(b.live.size());
        if (auto e = b.model.forward(&id, 1, pos, logits.data()); !e.empty()) {
            b.live.clear(); b.model.restart_at(0);
            r.finish_reason = "error: inkling decode: " + e; return r;
        }
        b.live.push_back(id);
        if (!finite(pos)) return r;
        id = Ds41Generator::sample(logits, recent, p, rng);
    }
    if (!pending.empty() && on_token) on_token(pending);
    r.decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dec).count();
    return r;
}
}  // namespace

std::string inkling_render_prompt(std::span<const ChatTurn> turns, std::string_view tools_json, bool thinking, std::string_view effort,
                           std::string& err) {
    std::string out, eff;
    if (!effort_text(effort, thinking, eff, err)) return {};
    auto role_token = [](const std::string& r) -> const char* {
        if (r == "user") return "<|message_user|>";
        if (r == "assistant") return "<|message_model|>";
        if (r == "system" || r == "developer") return "<|message_system|>";
        if (r == "tool") return "<|message_tool|>";
        return nullptr;
    };
    if (!tools_json.empty()) {
        const json tools = json::parse(tools_json, nullptr, false);
        if (tools.is_discarded() || !tools.is_array()) { err = "inkling: tools must be a JSON array"; return {}; }
        if (!tools.empty()) {
            out += "<|message_system|>tool_declare<|content_xml|>[";
            bool first = true;
            for (const auto& t : tools) {
                const json fn = t.contains("function") && t["function"].is_object() ? t["function"] : json::object();
                const json desc = fn.contains("description") && !fn["description"].is_null() && fn["description"] != "" ? fn["description"] : json("");
                const json params = fn.contains("parameters") && !fn["parameters"].is_null() && !fn["parameters"].empty() ? fn["parameters"] : json::object();
                const json name = fn.contains("name") ? fn["name"] : json(nullptr);
                const json type = t.contains("type") ? t["type"] : json("function");
                if (!first) out += ",";
                first = false;
                out += "{\"description\":" + canon(desc) + ",\"name\":" + canon(name) + ",\"parameters\":" + canon(params) +
                       ",\"type\":" + canon(type) + "}";
            }
            out += "]<|end_message|>";
        }
    }
    size_t first_idx = turns.size();
    for (size_t i = 0; i < turns.size(); ++i) if (turns[i].role != "system" && turns[i].role != "developer") { first_idx = i; break; }
    bool emitted = false;
    const std::string effort_msg = "<|message_system|><|content_text|>Thinking effort level: " + eff + "<|end_message|>";
    for (size_t i = 0; i < turns.size(); ++i) {
        const ChatTurn& m = turns[i];
        if (!m.images.empty()) { err = "inkling: this engine takes no images for Inkling"; return {}; }
        if (!emitted && i == first_idx) { out += effort_msg; emitted = true; }
        if (m.role == "tool") {
            std::string name;
            if (!m.tool_call_id.empty())                      // the name from the call this result answers
                for (size_t p = 0; p < i; ++p) {
                    if (turns[p].role != "assistant" || turns[p].tool_calls_json.empty()) continue;
                    const json calls = json::parse(turns[p].tool_calls_json, nullptr, false);
                    if (!calls.is_array()) continue;
                    for (const auto& c : calls)
                        if (c.contains("id") && c["id"].is_string() && c["id"] == m.tool_call_id && c.contains("function") &&
                            c["function"].contains("name") && c["function"]["name"].is_string())
                            name = c["function"]["name"].get<std::string>();
                }
            out += "<|message_tool|>" + name + "<|content_text|>" + m.content + "<|end_message|>";
            continue;
        }
        const char* rt = role_token(m.role);
        if (!rt) { err = "inkling: unknown message role '" + m.role + "'"; return {}; }
        std::string turn;
        const bool model = m.role == "assistant";
        if (model && m.reasoning_content && !m.reasoning_content->empty())
            turn += "<|message_model|><|content_thinking|>" + *m.reasoning_content + "<|end_message|>";
        const std::string content = model ? m.content_without_tool_calls.value_or(m.content) : m.content;
        if (!content.empty()) turn += std::string(rt) + "<|content_text|>" + content + "<|end_message|>";
        if (model && !m.tool_calls_json.empty()) {
            const json calls = json::parse(m.tool_calls_json, nullptr, false);
            if (calls.is_discarded() || !calls.is_array()) { err = "inkling: an assistant turn's tool_calls is not a JSON array"; return {}; }
            for (const auto& c : calls) {
                const json fn = c.contains("function") && c["function"].is_object() ? c["function"] : json::object();
                if (!fn.contains("name") || !fn["name"].is_string()) continue;
                std::string args = "{}";
                if (fn.contains("arguments")) {
                    const json& a = fn["arguments"];
                    if (a.is_object()) args = canon(a);
                    else if (a.is_string()) {
                        std::string s = a.get<std::string>();
                        const size_t b = s.find_first_not_of(" \t\r\n");
                        if (b != std::string::npos && s[b] == '{') {
                            const json pa = json::parse(s, nullptr, false);
                            if (!pa.is_discarded()) args = canon(pa);
                        }
                    }
                }
                const std::string name = fn["name"].get<std::string>();
                turn += "<|message_model|>" + name + "<|content_invoke_tool_json|>{\"name\":" + canon(json(name)) + ",\"args\":" + args +
                        "}<|end_message|>";
            }
        }
        out += turn;
        if (model && !turn.empty()) out += "<|content_model_end_sampling|>";
    }
    if (!emitted) out += effort_msg;
    out += "<|message_model|>";
    return out;
}

InklingReply inkling_parse_reply(std::string_view raw) {
    static const std::string kModel = "<|message_model|>", kThink = "<|content_thinking|>", kText = "<|content_text|>",
                             kTool = "<|content_invoke_tool_json|>", kEnd = "<|end_message|>", kEos = "<|content_model_end_sampling|>";
    InklingReply r;
    json calls = json::array();
    enum { kNone, kThinking, kTexting, kCalling } kind = kNone;
    std::string pend, name;
    auto finish = [&]() {
        if (kind == kThinking) r.reasoning += pend;
        else if (kind == kTexting) r.content += pend;
        else if (kind == kCalling) {
            const json j = json::parse(pend, nullptr, false);
            if (!j.is_discarded() && j.is_object() && (j.contains("name") || !name.empty())) {
                const std::string n = j.contains("name") && j["name"].is_string() ? j["name"].get<std::string>() : name;
                const json args = j.contains("args") ? j["args"] : json::object();
                calls.push_back({{"id", "call_ink_" + std::to_string(calls.size())}, {"type", "function"},
                                 {"function", {{"name", n}, {"arguments", args.dump(-1, ' ', false, json::error_handler_t::replace)}}}});
            } else r.content += pend;                         // an unparseable call stays visible
        }
        pend.clear();
    };
    size_t p = 0;
    while (p < raw.size()) {
        const size_t m = raw.find("<|", p);
        const size_t stop = m == std::string_view::npos ? raw.size() : m;
        (kind == kNone ? name : pend) += raw.substr(p, stop - p);
        if (m == std::string_view::npos) break;
        auto at = [&](const std::string& tag) { return raw.compare(m, tag.size(), tag) == 0; };
        if (at(kModel))      { finish(); kind = kNone; name.clear(); p = m + kModel.size(); }
        else if (at(kThink)) { finish(); kind = kThinking; p = m + kThink.size(); }
        else if (at(kText))  { finish(); kind = kTexting; p = m + kText.size(); }
        else if (at(kTool))  { finish(); kind = kCalling; p = m + kTool.size(); }
        else if (at(kEnd))   { finish(); kind = kNone; p = m + kEnd.size(); }
        else if (at(kEos))   { finish(); kind = kNone; p = m + kEos.size(); }
        else { (kind == kNone ? name : pend) += "<|"; p = m + 2; }
    }
    finish();
    if (!calls.empty()) r.tool_calls_json = calls.dump(-1, ' ', false, json::error_handler_t::replace);
    return r;
}

std::string Engine::inkling_load() {
    auto b = std::make_unique<InklingBundle>();
    if (auto e = read_inkling_config(gguf_, b->cfg); !e.empty()) return "inkling config: " + e;
    // the Arc GPUs on Level Zero: two cards when two are visible (each caches its own half of the layers' experts), one
    // with --gpus 1; each queue in its own single-device context (a two-device context mirrors VRAM into host RAM)
    std::vector<sycl::device> cards;
    for (const auto& pl : sycl::platform::get_platforms()) {
        if (pl.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : pl.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) cards.push_back(d);
    }
    if (cards.empty()) return "inkling: no Arc GPU";
    const uint32_t want = opts_.n_gpus ? opts_.n_gpus : 2, n = std::min<uint32_t>(std::min(want, 2u), uint32_t(cards.size()));
    if (want != n) std::fprintf(stderr, "[inkling] --gpus %u: running on %u card(s) (%zu visible, at most 2 used)\n", want, n, cards.size());
    opts_.n_gpus = n;
    b->q = std::make_unique<sycl::queue>(sycl::context(cards[0]), cards[0], sycl::property::queue::in_order{});
    if (n == 2) b->q2 = std::make_unique<sycl::queue>(sycl::context(cards[1]), cards[1], sycl::property::queue::in_order{});
    if (opts_.parallel == kLanesAuto) {
        opts_.parallel = 1;
        std::fprintf(stderr, "%s\n", lanes_auto_line("inkling", 1, 0, "no request lanes for this arch").c_str());
    } else if (opts_.parallel > 1)
        std::fprintf(stderr, "[inkling] one request lane in this version; requests take turns (--parallel %u)\n", opts_.parallel);
    b->reuse = opts_.prompt_cache;
    if (auto e = b->tok.load_from_gguf(gguf_); !e.empty()) return "inkling tokenizer: " + e;
    b->msg_model = b->tok.find_token("<|message_model|>");
    b->c_text = b->tok.find_token("<|content_text|>");
    b->c_thinking = b->tok.find_token("<|content_thinking|>");
    b->c_tool = b->tok.find_token("<|content_invoke_tool_json|>");
    b->end_msg = b->tok.find_token("<|end_message|>");
    for (int32_t v : {b->msg_model, b->c_text, b->c_thinking, b->c_tool, b->end_msg})
        if (v < 0) return "inkling: the vocabulary lacks a chat-format marker";
    b->stop = {b->tok.eos_token_id()};
    for (const char* s : {"<|message_user|>", "<|message_system|>", "<|message_tool|>"})
        if (const int32_t v = b->tok.find_token(s); v >= 0) b->stop.push_back(v);
    if (auto e = b->model.load(*b->q, gguf_, b->cfg, b->q2.get()); !e.empty()) return e;
    const uint32_t ctx = std::min<uint32_t>(opts_.max_ctx, b->cfg.ctx_train);
    if (auto e = b->model.init_runtime(ctx, b->chunk); !e.empty()) return e;
    std::fprintf(stderr, "[inkling] ready: Inkling-Small on %s, ctx %u, %.1f GiB on the card%s, %.1f GiB of experts pinned\n",
                 n == 2 ? "two cards" : "one card", ctx, b->model.device_bytes() / 1073741824.0, n == 2 ? "s" : "",
                 b->model.pinned_bytes() / 1073741824.0);
    inkling_ = std::move(b);
    return {};
}

GenerateResult Engine::inkling_generate(const std::string& prompt, const SamplingParams& sp, const TokenCallback& on_token) {
    std::lock_guard<std::mutex> lk(inkling_mu_);
    auto ids = inkling_->tok.encode(prompt, /*allow_special=*/true);
    return inkling_run(*inkling_, ids, sp, on_token, false, false);
}

GenerateResult Engine::inkling_chat(std::span<const ChatTurn> turns, const SamplingParams& sp, const TokenCallback& on_token,
                                    bool enable_thinking, std::string_view tools_json, std::string_view reasoning_effort) {
    std::lock_guard<std::mutex> lk(inkling_mu_);
    GenerateResult r;
    std::string err;
    const std::string prompt = inkling_render_prompt(turns, tools_json, enable_thinking, reasoning_effort, err);
    if (!err.empty()) { r.finish_reason = "error: " + err; return r; }
    auto ids = inkling_->tok.encode(prompt, /*allow_special=*/true);
    r = inkling_run(*inkling_, ids, sp, on_token, true, enable_thinking);
    if (r.finish_reason == "stop" || r.finish_reason == "length") {
        InklingReply pr = inkling_parse_reply(r.text);
        r.reasoning_content = std::move(pr.reasoning);
        r.text = std::move(pr.content);
        r.tool_calls_json = std::move(pr.tool_calls_json);
        if (r.tool_calls_json != "[]" && r.finish_reason == "stop") r.finish_reason = "tool_calls";
    }
    return r;
}

}  // namespace ie
