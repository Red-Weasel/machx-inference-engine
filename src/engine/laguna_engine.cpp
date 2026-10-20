// src/engine/laguna_engine.cpp — Laguna-S-2.1 (`laguna`) in the engine: load, the chat template, generate, parse
// (docs/laguna/00_PORT_PLAN.md P5). The runtime is LagunaModel (src/model/laguna.cpp, one card or two); the tokenizer is the
// GGUF's (pre "laguna"); the template is the GGUF's chat_template rendered natively (laguna_render below mirrors it line
// for line); the reply is parsed by GLM's parser -- Laguna's template is an iteration on GLM's thinking template and its
// tool calls are GLM's <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>.
#include "ie/engine.hpp"

#include "ie/deepseek41_generate.hpp"   // Ds41Generator::sample (the host sampler the other dir runtimes use)
#include "ie/glm5_server.hpp"           // parse_glm5_completion
#include "ie/laguna_engine.hpp"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

namespace ie {

namespace {
using ojson = nlohmann::ordered_json;

size_t laguna_utf8_complete(std::string_view s) {
    if (s.empty()) return 0;
    size_t i = s.size(), cont = 0;
    while (i > 0 && cont < 3 && (uint8_t(s[i - 1]) & 0xC0u) == 0x80u) { --i; ++cont; }
    if (i == 0) return 0;
    const uint8_t lead = uint8_t(s[i - 1]);
    const size_t need = (lead & 0x80u) == 0 ? 1 : (lead & 0xE0u) == 0xC0u ? 2 : (lead & 0xF0u) == 0xE0u ? 3 : (lead & 0xF8u) == 0xF0u ? 4 : 1;
    return (cont + 1 >= need) ? i - 1 + need : i - 1;
}

// Jinja's `tojson` (Python json.dumps, ensure_ascii=False, default separators ", " and ": ").
std::string py_dumps(const ojson& j) {
    if (j.is_object()) {
        std::string o = "{"; bool first = true;
        for (auto it = j.begin(); it != j.end(); ++it) { if (!first) o += ", "; first = false; o += ojson(it.key()).dump(-1, ' ', false) + ": " + py_dumps(it.value()); }
        return o + "}";
    }
    if (j.is_array()) {
        std::string o = "["; bool first = true;
        for (const auto& v : j) { if (!first) o += ", "; first = false; o += py_dumps(v); }
        return o + "]";
    }
    return j.dump(-1, ' ', false, ojson::error_handler_t::replace);
}

std::string rstrip(std::string s) { while (!s.empty() && std::isspace(uint8_t(s.back()))) s.pop_back(); return s; }
bool has_text(const std::string& s) { for (char c : s) if (!std::isspace(uint8_t(c))) return true; return false; }

// The GGUF's chat template (laguna_glm_thinking_v8 iteration), rendered natively. add_generation_prompt is always on here.
std::string laguna_render(std::span<const ChatTurn> turns, std::string_view tools_json, bool thinking, std::string& err) {
    std::string p = "〈|EOS|〉";
    std::string system = "You are a helpful, conversationally-fluent assistant made by Poolside. You are here to be helpful to users through natural language conversations.";
    size_t first = 0;
    if (!turns.empty() && turns[0].role == "system") { system = turns[0].content; first = 1; }
    ojson tools = ojson::array();
    if (!tools_json.empty()) {
        tools = ojson::parse(tools_json, nullptr, false);
        if (tools.is_discarded() || !tools.is_array()) { err = "laguna: tools must be a JSON array"; return {}; }
    }
    const bool has_sys = has_text(system), has_tools = !tools.empty();
    if (has_sys || has_tools || thinking) {
        p += "<system>";
        if (has_sys) { p += rstrip(system); if (has_tools) p += "\n\n"; }
        if (has_tools) {
            p += "### Tools\n\nYou may call functions to assist with the user query.\nAll available function signatures are listed below:\n<available_tools>\n";
            for (const auto& t : tools) p += py_dumps(t) + "\n";
            p += "</available_tools>";
        }
        p += "</system>\n";
    }
    for (size_t i = first; i < turns.size(); ++i) {
        const ChatTurn& m = turns[i];
        if (m.role == "user") p += "<user>" + m.content + "</user>\n";
        else if (m.role == "assistant") {
            p += "<assistant>";
            const std::string reasoning = m.reasoning_content.value_or("");
            p += thinking ? "<think>" + reasoning + "</think>" : std::string("</think>");
            const std::string content = m.content_without_tool_calls.value_or(m.content);
            p += content;
            if (!m.tool_calls_json.empty()) {
                const ojson calls = ojson::parse(m.tool_calls_json, nullptr, false);
                if (calls.is_discarded() || !calls.is_array()) { err = "laguna: an assistant turn's tool_calls is not a JSON array"; return {}; }
                for (const auto& c : calls) {
                    const ojson& f = c.contains("function") ? c["function"] : c;
                    if (!f.contains("name") || !f["name"].is_string()) { err = "laguna: a tool call has no name"; return {}; }
                    p += "<tool_call>" + f["name"].get<std::string>();
                    ojson args = f.contains("arguments") ? f["arguments"] : ojson::object();
                    if (args.is_string()) { args = ojson::parse(args.get<std::string>(), nullptr, false); if (args.is_discarded()) args = ojson::object(); }
                    if (args.is_object())
                        for (auto it = args.begin(); it != args.end(); ++it)
                            p += "<arg_key>" + it.key() + "</arg_key><arg_value>" +
                                 (it.value().is_string() ? it.value().get<std::string>() : py_dumps(it.value())) + "</arg_value>";
                    p += "</tool_call>";
                }
            }
            p += "</assistant>\n";
        } else if (m.role == "tool") p += "<tool_response>" + m.content + "</tool_response>\n";
        else if (m.role == "system") p += "<system>" + m.content + "</system>\n";
        else { err = "laguna: unknown role '" + m.role + "'"; return {}; }
    }
    p += thinking ? "<assistant><think>" : "<assistant></think>";
    return p;
}

// Prefill (reusing the live sequence's common prefix) + sampling loop. r.text is the raw completion (reasoning, content and
// tool-call markup as the model wrote them); the chat route parses it.
GenerateResult laguna_run(LagunaBundle& b, const std::vector<int32_t>& ids, const SamplingParams& sp, const TokenCallback& on_token) {
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
    if (!b.model.rewind(L)) { L = 0; b.model.rewind(0); }
    b.live.resize(L);
    r.cached_tokens = L;
    r.cache_source = L ? "live" : "none";
    std::vector<float> logits(b.cfg.vocab);
    // the prompt in chunks (two cards: overlapped); the liveness probe runs as each chunk is in
    auto alive = [&](uint32_t) { return !on_token || on_token(std::string_view{}); };
    if (auto e = b.model.prefill(ids.data() + L, uint32_t(ids.size()) - L, L, logits.data(), alive); !e.empty()) {
        b.live.clear(); b.model.rewind(0);
        r.finish_reason = "error: laguna prefill: " + e; return r;
    }
    b.live.assign(ids.begin(), ids.begin() + b.model.pos());
    if (b.model.pos() < ids.size()) { r.finish_reason = "abort"; return r; }   // the client went away mid-prompt
    r.prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pf).count();

    Ds41SampleParams p; p.temperature = sp.temperature; p.top_k = sp.top_k; p.top_p = sp.top_p; p.min_p = sp.min_p;
    p.repeat_penalty = sp.repeat_penalty; p.repeat_window = sp.repeat_window;
    p.seed = sp.seed ? sp.seed : uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
    uint64_t rng = p.seed;
    std::vector<int32_t> recent(ids.end() - std::min<size_t>(ids.size(), 512), ids.end());
    std::string pending;
    const auto t_dec = std::chrono::steady_clock::now();
    r.finish_reason = "length";
    auto finite = [&](uint32_t pos) {
        for (float v : logits) if (!std::isfinite(v)) {
            std::fprintf(stderr, "[laguna] non-finite logits at position %u -- not sampled; the request fails and the conversation is cleared\n", pos);
            b.live.clear(); b.model.rewind(0);
            r.finish_reason = "error: laguna: non-finite logits at position " + std::to_string(pos);
            return false;
        }
        return true;
    };
    if (!finite(uint32_t(ids.size()) - 1)) return r;
    int32_t id = Ds41Generator::sample(logits, recent, p, rng);
    for (uint32_t k = 0;;) {
        if (!sp.ignore_eos && std::find(b.eos.begin(), b.eos.end(), id) != b.eos.end()) { r.finish_reason = "stop"; break; }
        ++r.completion_tokens;
        recent.push_back(id);
        if (recent.size() > 512) recent.erase(recent.begin());
        pending += b.tok.decode(std::vector<int32_t>{id});
        if (const size_t done = laguna_utf8_complete(pending)) {
            const std::string piece = pending.substr(0, done);
            pending.erase(0, done);
            r.text += piece;
            if (on_token && !on_token(piece)) { r.finish_reason = "abort"; break; }
        }
        if (++k == max_new) break;
        const uint32_t pos = uint32_t(b.live.size());
        if (auto e = b.model.forward(&id, 1, pos, logits.data()); !e.empty()) {
            b.live.clear(); b.model.rewind(0);
            r.finish_reason = "error: laguna decode: " + e; return r;
        }
        b.live.push_back(id);
        if (!finite(pos)) return r;
        id = Ds41Generator::sample(logits, recent, p, rng);
    }
    if (!pending.empty()) { r.text += pending; if (on_token) on_token(pending); }
    r.decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dec).count();
    return r;
}
}  // namespace

std::string Engine::laguna_load() {
    auto b = std::make_unique<LagunaBundle>();
    if (auto e = read_laguna_config(gguf_, b->cfg); !e.empty()) return "laguna config: " + e;
    // the Arc GPUs on Level Zero: two cards when two are visible (each caches its own half of the layers' experts), one
    // with --gpus 1; each queue in its own single-device context (a two-device context mirrors VRAM into host RAM)
    std::vector<sycl::device> cards;
    for (const auto& pl : sycl::platform::get_platforms()) {
        if (pl.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : pl.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) cards.push_back(d);
    }
    if (cards.empty()) return "laguna: no Arc GPU";
    const uint32_t want = opts_.n_gpus ? opts_.n_gpus : 2, n = std::min<uint32_t>(std::min(want, 2u), uint32_t(cards.size()));
    if (want != n) std::fprintf(stderr, "[laguna] --gpus %u: running on %u card(s) (%zu visible, at most 2 used)\n", want, n, cards.size());
    opts_.n_gpus = n;
    b->q = std::make_unique<sycl::queue>(sycl::context(cards[0]), cards[0], sycl::property::queue::in_order{});
    if (n == 2) b->q2 = std::make_unique<sycl::queue>(sycl::context(cards[1]), cards[1], sycl::property::queue::in_order{});
    if (opts_.parallel == kLanesAuto) {
        opts_.parallel = 1;
        std::fprintf(stderr, "%s\n", lanes_auto_line("laguna", 1, 0, "no request lanes for this arch").c_str());
    } else if (opts_.parallel > 1)
        std::fprintf(stderr, "[laguna] one request lane in this version; requests take turns (--parallel %u)\n", opts_.parallel);
    b->reuse = opts_.prompt_cache;
    if (auto e = b->tok.load_from_gguf(gguf_); !e.empty()) return "laguna tokenizer: " + e;
    if (auto e = b->model.load(*b->q, gguf_, b->cfg, b->q2.get()); !e.empty()) return e;
    const uint32_t ctx = std::min<uint32_t>(opts_.max_ctx, b->cfg.ctx_train);
    if (auto e = b->model.init_runtime(ctx, b->chunk); !e.empty()) return e;
    b->eos = {b->tok.eos_token_id()};
    if (const auto* kv = gguf_.find_kv("tokenizer.ggml.eot_token_id")) b->eos.push_back(int32_t(kv->as_int()));
    std::fprintf(stderr, "[laguna] ready: Laguna-S-2.1 on %s, ctx %u, %.1f GiB on the card%s, %.1f GiB of experts pinned\n",
                 n == 2 ? "two cards" : "one card", ctx, b->model.device_bytes() / 1073741824.0, n == 2 ? "s" : "",
                 b->model.pinned_bytes() / 1073741824.0);
    laguna_ = std::move(b);
    return {};
}

GenerateResult Engine::laguna_generate(const std::string& prompt, const SamplingParams& sp, const TokenCallback& on_token) {
    std::lock_guard<std::mutex> lk(laguna_mu_);
    auto ids = laguna_->tok.encode(prompt, /*allow_special=*/true);
    return laguna_run(*laguna_, ids, sp, on_token);
}

GenerateResult Engine::laguna_chat(std::span<const ChatTurn> turns, const SamplingParams& sp, const TokenCallback& on_token,
                                   bool enable_thinking, std::string_view tools_json, std::string_view /*reasoning_effort*/) {
    std::lock_guard<std::mutex> lk(laguna_mu_);
    GenerateResult r;
    for (const auto& t : turns) if (!t.images.empty()) { r.finish_reason = "error: laguna takes no images"; return r; }
    std::string err;
    const std::string prompt = laguna_render(turns, tools_json, enable_thinking, err);
    if (!err.empty()) { r.finish_reason = "error: " + err; return r; }
    auto ids = laguna_->tok.encode(prompt, /*allow_special=*/true);
    // the template writes 〈|EOS|〉 (the BOS) itself: drop the tokenizer's own when it doubles it
    if (ids.size() > 1 && laguna_->tok.add_bos_token() && ids[0] == laguna_->tok.bos_token_id() && ids[1] == ids[0]) ids.erase(ids.begin());
    r = laguna_run(*laguna_, ids, sp, on_token);
    if (r.finish_reason == "stop" || r.finish_reason == "length") {
        auto pc = parse_glm5_completion(r.text, enable_thinking, tools_json);
        r.reasoning_content = std::move(pc.reasoning);
        r.text = std::move(pc.content);
        r.tool_calls_json = pc.tool_calls.empty() ? "[]" : std::move(pc.tool_calls);
        if (r.tool_calls_json != "[]" && r.finish_reason == "stop") r.finish_reason = "tool_calls";
    }
    return r;
}

}  // namespace ie
