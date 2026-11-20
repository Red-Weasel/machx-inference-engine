// src/engine/hy4flash_engine.cpp — Tencent Hy4-flash (`hy_v4_flash`) in the engine: load, the chat template, generate, parse.
// The runtime is Hy4fForward (src/model/hy4flash_forward.cpp) on the two cards with the 8-bit expert pack and the 8-bit
// big non-expert matrices; the tokenizer is the checkpoint's tokenizer.json; the template is chat_template.jinja rendered
// natively (hy4f_render below mirrors it macro for macro). The reply's tags are GLM's with an ":opensource" suffix and one
// outer <tool_calls> wrapper, so the stream is translated to GLM's spelling as it is written and parsed by GLM's parser.
// Images: the checkpoint's tower (src/model/hy4flash_vision.cpp) is resident on the first card; an image is its rows in the
// prompt (3-D positions, the tower's vectors as embeddings). The drafter (model.dfly_layers.*) is not ported.
#include "ie/engine.hpp"

#include "ie/deepseek41_generate.hpp"   // Ds41Generator::sample (the host sampler the other directory runtimes use)
#include "ie/glm5_server.hpp"           // parse_glm5_completion
#include "ie/hy4flash_engine.hpp"
#include "ie/lanes_auto.hpp"
#include "ie/spec_accept.hpp"
#include "ie/vision_required.hpp"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>

namespace ie {

namespace {
using ojson = nlohmann::ordered_json;

constexpr const char* kStart = "<｜hy_start:opensource｜>";
constexpr const char* kMiddle = "<｜hy_middle:opensource｜>";
constexpr const char* kEnd = "<｜hy_end:opensource｜>";
constexpr const char* kMode = "<｜reasoning_mode:opensource｜>";
constexpr std::string_view kImage = "<｜hy_image▁start:opensource｜><｜hy_image▁pad:opensource｜><｜hy_image▁end:opensource｜>";   // the template's own
// (the model's tag, the stream's)
constexpr std::pair<const char*, const char*> kTags[] = {
    {"</think:opensource>", "</think>"}, {"<think:opensource>", ""},
    {"<tool_calls:opensource>", ""}, {"</tool_calls:opensource>", ""},
    {"<tool_call:opensource>", "<tool_call>"}, {"</tool_call:opensource>", "</tool_call>"},
    {"<arg_key:opensource>", "<arg_key>"}, {"</arg_key:opensource>", "</arg_key>"},
    {"<arg_value:opensource>", "<arg_value>"}, {"</arg_value:opensource>", "</arg_value>"},
};

size_t utf8_complete(std::string_view s) {
    if (s.empty()) return 0;
    size_t i = s.size(), cont = 0;
    while (i > 0 && cont < 3 && (uint8_t(s[i - 1]) & 0xC0u) == 0x80u) { --i; ++cont; }
    if (i == 0) return 0;
    const uint8_t lead = uint8_t(s[i - 1]);
    const size_t need = (lead & 0x80u) == 0 ? 1 : (lead & 0xE0u) == 0xC0u ? 2 : (lead & 0xF0u) == 0xE0u ? 3 : (lead & 0xF8u) == 0xF0u ? 4 : 1;
    return (cont + 1 >= need) ? i - 1 + need : i - 1;
}

// Jinja's `tojson(ensure_ascii=False)` (Python json.dumps, separators ", " and ": ").
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

// the template's tool_to_json: every key but defer_loading / strict, '"key": ' + tojson(value), joined by ", "
std::string tool_to_json(const ojson& t) {
    std::string o = "{"; bool first = true;
    for (auto it = t.begin(); it != t.end(); ++it) {
        if (it.key() == "defer_loading" || it.key() == "strict") continue;
        if (!first) o += ", ";
        first = false;
        o += "\"" + it.key() + "\": " + py_dumps(it.value());
    }
    return o + "}";
}

// A turn's content: the request's text, and the template's image triple where each image stands (the server's marker when
// the markers match the images, else all of them in front -- the other vision families' rule).
void content_with_images(PromptPiece& p, const ChatTurn& m) {
    const std::string_view c = m.content; const size_t n = m.images.size();
    if (!n) { p.add_content(c); return; }
    size_t markers = 0;
    for (size_t q = c.find(kChatImageMarker); q != std::string_view::npos; q = c.find(kChatImageMarker, q + kChatImageMarker.size())) ++markers;
    if (markers == n) {
        size_t at = 0;
        for (size_t q = c.find(kChatImageMarker); q != std::string_view::npos; q = c.find(kChatImageMarker, at)) {
            p.add_content(c.substr(at, q - at)); p.add_text(kImage); at = q + kChatImageMarker.size();
        }
        p.add_content(c.substr(at));
        return;
    }
    for (size_t i = 0; i < n; ++i) p.add_text(kImage);
    size_t at = 0;
    for (size_t q = c.find(kChatImageMarker); q != std::string_view::npos; q = c.find(kChatImageMarker, at)) { p.add_content(c.substr(at, q - at)); at = q + kChatImageMarker.size(); }
    p.add_content(c.substr(at));
}

// render_tools_prompt
void tools_prompt(PromptPiece& p, const ojson& tools) {
    p.add_text("# Tools\n\nYou may call one or more functions to assist with the user query."
               "\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n");
    bool first = true;
    for (const auto& tool : tools) {
        const ojson& t = tool.is_object() && tool.contains("function") ? tool["function"] : tool;
        if (t.is_object() && t.contains("defer_loading") && t["defer_loading"].is_boolean() && t["defer_loading"].get<bool>()) continue;
        if (!first) p.add_text("\n");
        first = false;
        p.add_content(t.is_object() ? tool_to_json(t) : py_dumps(t));
    }
    p.add_text("\n</tools>\n\nFor function call returns, you should first print <tool_calls:opensource>"
               "\nFor each function call, you should return object like:\n"
               "<tool_call:opensource>{function-name}<arg_key:opensource>{arg-key-1}</arg_key:opensource><arg_value:opensource>{arg-value-1}</arg_value:opensource>"
               "<arg_key:opensource>{arg-key-2}</arg_key:opensource><arg_value:opensource>{arg-value-2}</arg_value:opensource>...</tool_call:opensource>"
               "\nAt the end of function call returns, you should print </tool_calls:opensource>");
}

std::string hy4f_render(std::span<const ChatTurn> turns, std::string_view tools_json, std::string_view effort, std::string& err,
                        std::vector<std::pair<size_t, size_t>>* content_out) {
    PromptPiece p;
    ojson tools = ojson::array();
    if (!tools_json.empty()) {
        tools = ojson::parse(tools_json, nullptr, false);
        if (tools.is_discarded() || !tools.is_array()) { err = "hy4-flash: tools must be a JSON array"; return {}; }
    }
    const bool has_tools = !tools.empty(), no_think = effort == "no_think";
    const bool preserved_thinking = has_tools;          // the template's default: reasoning of earlier turns is kept when tools are in play
    // merge_tool_responses: consecutive tool turns are one message
    struct Msg { const ChatTurn* one = nullptr; std::vector<const ChatTurn*> tools; };
    std::vector<Msg> ms;
    for (const ChatTurn& t : turns) {
        if (t.role == "tool") { if (ms.empty() || ms.back().tools.empty()) ms.emplace_back(); ms.back().tools.push_back(&t); }
        else { ms.emplace_back(); ms.back().one = &t; }
    }
    int last_user = -1;
    for (size_t i = 0; i < ms.size(); ++i) if (ms[i].one && ms[i].one->role == "user") last_user = int(i);
    const bool leading_system = !ms.empty() && ms[0].one && ms[0].one->role == "system";
    auto mode = [&]() { p.add_text(kMode); p.add_text("reasoning_effort:"); p.add_text(effort); };
    if (!leading_system) {               // no leading system: one is made for the tools and the reasoning mode
        p.add_text(kStart); p.add_text("system"); p.add_text(kMiddle);
        if (has_tools) tools_prompt(p, tools);
        mode(); p.add_text(kEnd);
    }
    std::vector<std::string> prev_ids;   // the tool-call ids of the assistant turn just rendered (complete sets only)
    bool last_is_assistant = false;
    for (size_t i = 0; i < ms.size(); ++i) {
        const Msg& g = ms[i];
        bool keep_ids = false;
        if (!g.one) {                    // a cluster of tool results, in the order of the calls when the ids say so
            std::vector<const ChatTurn*> order; bool matched = !prev_ids.empty() && prev_ids.size() == g.tools.size();
            for (size_t k = 0; matched && k < prev_ids.size(); ++k) {
                const ChatTurn* hit = nullptr; int n = 0;
                for (const ChatTurn* t : g.tools) if (!t->tool_call_id.empty() && t->tool_call_id == prev_ids[k]) { ++n; hit = t; }
                if (n == 1) order.push_back(hit); else matched = false;
            }
            p.add_text(kStart); p.add_text("tool"); p.add_text(kMiddle);
            for (const ChatTurn* t : matched ? order : g.tools) { p.add_text("<tool_response:opensource>"); p.add_content(t->content); p.add_text("</tool_response:opensource>"); }
            p.add_text(kEnd);
        } else {
            const ChatTurn& m = *g.one;
            if (m.role == "system") {
                if (!m.images.empty()) { err = "hy4-flash: a system message cannot contain images"; return {}; }
                p.add_text(kStart); p.add_text("system"); p.add_text(kMiddle);
                if (i == 0) {            // tools and the reasoning mode attach to the leading system only
                    if (has_tools) { tools_prompt(p, tools); if (!m.content.empty()) p.add_text("\n\n"); }
                    p.add_content(m.content);
                    mode();
                } else p.add_content(m.content);
                p.add_text(kEnd);
            } else if (m.role == "assistant") {
                const std::string& reasoning = m.reasoning_content ? *m.reasoning_content : std::string();
                p.add_text(kStart); p.add_text("assistant"); p.add_text(kMiddle);
                p.add_text("<think:opensource>");
                if (!no_think && (preserved_thinking || int(i) > last_user) && !reasoning.empty()) p.add_content(reasoning);
                p.add_text("</think:opensource>");
                p.add_content(m.content_without_tool_calls.value_or(m.content));
                if (!m.tool_calls_json.empty()) {
                    const ojson calls = ojson::parse(m.tool_calls_json, nullptr, false);
                    if (calls.is_discarded() || !calls.is_array()) { err = "hy4-flash: an assistant turn's tool_calls is not a JSON array"; return {}; }
                    if (!calls.empty()) {
                        bool complete = m.tool_call_ids.size() == calls.size();
                        for (const auto& id : m.tool_call_ids) complete = complete && !id.empty();
                        prev_ids = complete ? m.tool_call_ids : std::vector<std::string>{}; keep_ids = true;
                        p.add_text("<tool_calls:opensource>");
                        for (const auto& c : calls) {
                            const ojson& f = c.is_object() && c.contains("function") ? c["function"] : c;
                            if (!f.is_object() || !f.contains("name") || !f["name"].is_string()) { err = "hy4-flash: a tool call has no name"; return {}; }
                            p.add_text("<tool_call:opensource>"); p.add_content(f["name"].get_ref<const std::string&>());
                            ojson args = f.contains("arguments") ? f["arguments"] : ojson::object();
                            if (args.is_string()) { args = ojson::parse(args.get<std::string>(), nullptr, false); if (args.is_discarded()) args = ojson::object(); }
                            if (args.is_object())
                                for (auto it = args.begin(); it != args.end(); ++it) {
                                    p.add_text("<arg_key:opensource>"); p.add_content(it.key()); p.add_text("</arg_key:opensource><arg_value:opensource>");
                                    if (it.value().is_string()) p.add_content(it.value().get_ref<const std::string&>()); else p.add_content(py_dumps(it.value()));
                                    p.add_text("</arg_value:opensource>");
                                }
                            p.add_text("</tool_call:opensource>");
                        }
                        p.add_text("</tool_calls:opensource>");
                    }
                }
                p.add_text(kEnd);
            } else {                     // user, and any other role by its name
                p.add_text(kStart); p.add_content(m.role); p.add_text(kMiddle); content_with_images(p, m); p.add_text(kEnd);
            }
        }
        if (!keep_ids) prev_ids.clear();
        last_is_assistant = g.one && g.one->role == "assistant";
    }
    if (!last_is_assistant) {
        p.add_text(kStart); p.add_text("assistant"); p.add_text(kMiddle);
        p.add_text(no_think ? "<think:opensource></think:opensource>" : "<think:opensource>");
    }
    if (content_out) *content_out = std::move(p.content);
    return std::move(p.text);
}

// Prefill (reusing the live sequence's common prefix) + the sampling loop. r.text is the completion in the STREAM's
// spelling (hy4f_translate_reply's); the chat route parses it.
struct PromptImage { uint32_t pos0 = 0, n = 0; uint64_t hash = 0; Hy4fVisGeom g; std::vector<float> pv, rows; bool encoded = false; };
struct PromptExtra { std::vector<PromptImage>* images = nullptr; const std::vector<int32_t>* pos3 = nullptr; const std::vector<uint8_t>* is_embed = nullptr; };

GenerateResult hy4f_run(Hy4FlashBundle& b, const std::vector<int32_t>& ids, const SamplingParams& sp, const TokenCallback& on_token, const PromptExtra* mm = nullptr) {
    GenerateResult r;
    r.prompt_tokens = uint32_t(ids.size());
    if (ids.empty()) { r.finish_reason = "error: empty prompt"; return r; }
    if (ids.size() >= b.ctx) { r.finish_reason = "context_length_exceeded"; return r; }
    const uint32_t room = b.ctx - uint32_t(ids.size());
    const uint32_t max_new = sp.max_tokens == 0 ? room : std::min(sp.max_tokens, room);
    auto fail = [&](const std::string& what) { b.live.clear(); b.live_images.clear(); b.fw.rewind(0); r.finish_reason = "error: hy4-flash " + what; return r; };

    const auto t_pf = std::chrono::steady_clock::now();
    uint32_t L = 0;
    if (b.reuse) while (L < b.live.size() && L + 1 < ids.size() && b.live[L] == ids[L]) ++L;   // at least one row to read
    {   // an image's rows all carry one token id: the shared prefix ends where the images themselves differ
        static const std::vector<PromptImage> none;
        const std::vector<PromptImage>& now = mm && mm->images ? *mm->images : none;
        auto same = [&](uint32_t pos0, uint32_t n, uint64_t h) { for (const auto& im : now) if (im.pos0 == pos0 && im.n == n && im.hash == h) return true; return false; };
        for (const auto& li : b.live_images) if (li.pos0 < L && !same(li.pos0, li.n, li.hash)) L = li.pos0;
        for (const auto& im : now) if (im.pos0 < L) {
            bool found = false;
            for (const auto& li : b.live_images) found = found || (li.pos0 == im.pos0 && li.n == im.n && li.hash == im.hash);
            if (!found) L = im.pos0;
        }
    }
    b.fw.rewind(L);
    b.live.resize(L);
    b.live_images.erase(std::remove_if(b.live_images.begin(), b.live_images.end(), [&](const auto& li) { return li.pos0 + li.n > L; }), b.live_images.end());
    r.cached_tokens = L;
    r.cache_source = L ? "live" : "none";
    std::vector<float> logits;
    uint32_t last_rows = 1;                                // rows of the latest forward(): the drafter reads the row a token was sampled from
    for (size_t pos = L; pos < ids.size();) {             // the prompt in chunks; the liveness probe between them
        const size_t left = ids.size() - pos;
        const uint32_t T = uint32_t(b.big > b.chunk && left > std::max<size_t>(b.chunk, b.big_min) ? std::min<size_t>(b.big, left) : std::min<size_t>(b.chunk, left));
        last_rows = T;
        Hy4fRowInputs rin; std::vector<float> emb; const Hy4fRowInputs* rp = nullptr;
        if (mm && mm->images && !mm->images->empty()) {          // this chunk's image rows: the tower's vectors, in row order
            const uint32_t H = b.model.config().dim;
            for (auto& im : *mm->images) {
                if (im.pos0 + im.n <= pos || im.pos0 >= pos + T) continue;
                if (!im.encoded) {
                    if (auto e = b.vis.encode(im.pv, im.g, im.rows); !e.empty()) return fail("image: " + e);
                    im.encoded = true; std::vector<float>().swap(im.pv);
                }
                const uint32_t r0 = uint32_t(std::max<size_t>(pos, im.pos0)) - im.pos0, r1 = uint32_t(std::min<size_t>(pos + T, im.pos0 + im.n)) - im.pos0;
                emb.insert(emb.end(), im.rows.begin() + long(size_t(r0) * H), im.rows.begin() + long(size_t(r1) * H));
            }
            rin.pos3 = mm->pos3->data() + pos * 3; rin.is_embed = mm->is_embed->data() + pos; rin.embeds = emb.data(); rp = &rin;
        }
        if (auto e = b.fw.forward(ids.data() + pos, T, logits, false, rp); !e.empty()) return fail("prompt read: " + e);
        if (mm && mm->images) for (const auto& im : *mm->images) if (im.pos0 + im.n > pos && im.pos0 + im.n <= pos + T) b.live_images.push_back({im.pos0, im.n, im.hash});   // (its last row is in)
        b.live.insert(b.live.end(), ids.begin() + long(pos), ids.begin() + long(pos + T));
        pos += T;
        if (pos < ids.size() && on_token && !on_token(std::string_view{})) { r.finish_reason = "abort"; return r; }   // the client went away
    }
    r.prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pf).count();

    Ds41SampleParams p; p.temperature = sp.temperature; p.top_k = sp.top_k; p.top_p = sp.top_p; p.min_p = sp.min_p;
    p.repeat_penalty = sp.repeat_penalty; p.repeat_window = sp.repeat_window;
    p.seed = sp.seed ? sp.seed : uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
    uint64_t rng = p.seed;
    std::vector<int32_t> recent(ids.end() - long(std::min<size_t>(ids.size(), 512)), ids.end());
    std::string pending;
    const auto t_dec = std::chrono::steady_clock::now();
    r.finish_reason = "length";
    auto finite = [&](uint32_t pos) {
        for (float v : logits) if (!std::isfinite(v)) {
            std::fprintf(stderr, "[hy4-flash] non-finite logits at position %u -- not sampled; the request fails and the conversation is cleared\n", pos);
            fail("non-finite logits at position " + std::to_string(pos));
            return false;
        }
        return true;
    };
    if (!finite(uint32_t(ids.size()) - 1)) return r;
    auto finite_row = [&](const float* lr, size_t V, uint32_t pos) {
        for (size_t v = 0; v < V; ++v) if (!std::isfinite(lr[v])) {
            std::fprintf(stderr, "[hy4-flash] non-finite logits at position %u -- not sampled; the request fails and the conversation is cleared\n", pos);
            fail("non-finite logits at position " + std::to_string(pos));
            return false;
        }
        return true;
    };
    // one sampled token into the reply; false ends the request (a stop token: not part of the reply; or the client gone)
    auto emit = [&](int32_t tk) -> bool {
        if (!sp.ignore_eos && std::find(b.eos.begin(), b.eos.end(), tk) != b.eos.end()) { r.finish_reason = "stop"; return false; }
        ++r.completion_tokens;
        recent.push_back(tk);
        if (recent.size() > 512) recent.erase(recent.begin());
        const auto st = std::find_if(b.stream_text.begin(), b.stream_text.end(), [&](const auto& e) { return e.first == tk; });
        pending += st != b.stream_text.end() ? st->second : b.tok.decode(std::vector<int32_t>{tk});
        if (const size_t done = utf8_complete(pending)) {
            const std::string piece = pending.substr(0, done);
            pending.erase(0, done);
            r.text += piece;
            if (on_token && !on_token(piece)) { r.finish_reason = "abort"; return false; }
        }
        return true;
    };
    // The drafter. A round: the drafts after `id` (as many as its confidence head gives a fair chance), one pass of the
    // model over [id, drafts], and the sampler decides row by row. Temperature 0: a draft is kept when it IS the sampler's
    // token. Temperature > 0: the drafts are DRAWN from the sampler's own pipeline applied to the drafter's logits (q) and
    // kept by speculative sampling against the model's distribution p -- the reply follows p exactly (ie/spec_accept.hpp).
    const bool drafting = b.draft && b.fw.has_drafter();
    const bool ss = p.temperature > 0.f;
    const size_t V = b.model.config().vocab_size;
    struct QRow { std::vector<int32_t> ids; std::vector<double> probs; };
    std::vector<QRow> qrows(drafting ? b.fw.draft_width() : 0);
    std::vector<float> qlg, qdense(drafting && ss ? V : 0, 0.f), vlg;
    std::vector<int32_t> d, block, p_ids; std::vector<double> p_probs;
    std::function<int32_t(uint32_t, const float*)> pick;
    if (ss) pick = [&](uint32_t j, const float* lg) -> int32_t {
        QRow& qr = qrows[j];
        qlg.assign(lg, lg + V);                                    // (the pipeline applies the repeat penalty in place)
        Ds41Generator::distribution(qlg.data(), V, recent, p, qr.ids, qr.probs);
        const double u = Ds41Generator::uniform(rng); double c = 0;
        for (size_t i = 0; i < qr.ids.size(); ++i) { c += qr.probs[i]; if (u < c) return qr.ids[i]; }
        return qr.ids.back();
    };
    uint64_t n_rounds = 0, n_verified = 0, n_kept = 0, n_plain = 0; double ms_draft = 0;
    uint32_t row = last_rows - 1;                                  // the row of the latest forward() that `id` was sampled from
    int32_t id = Ds41Generator::sample(logits, recent, p, rng);
    for (uint32_t k = 0;;) {
        if (!emit(id)) break;
        if (++k == max_new) break;
        const uint32_t pos = uint32_t(b.live.size());
        d.clear();
        if (drafting && b.draft) {
            // a block is `id` and its drafts: inside the context, and no more drafts than tokens still asked for
            const uint32_t room = b.ctx > pos + 1 ? b.ctx - pos - 1 : 0, cap = std::min({b.fw.draft_width(), room, max_new - k});
            if (cap) {
                const auto td = std::chrono::steady_clock::now();
                if (auto e = b.fw.draft(row, id, d, pick, nullptr, b.draft_conf); !e.empty()) {
                    std::fprintf(stderr, "[hy4-flash] drafter: %s -- drafting is off from here on; the reply continues without it\n", e.c_str());
                    b.draft = false; d.clear();
                }
                ms_draft += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - td).count();
                if (d.size() > cap) d.resize(cap);
            }
        }
        if (d.empty()) {
            if (auto e = b.fw.forward(&id, 1, logits, false); !e.empty()) return fail("decode: " + e);
            b.live.push_back(id); row = 0; ++n_plain;
            if (!finite(pos)) return r;
            id = Ds41Generator::sample(logits, recent, p, rng);
            continue;
        }
        block.assign(1, id); block.insert(block.end(), d.begin(), d.end());
        if (auto e = b.fw.forward(block.data(), uint32_t(block.size()), vlg, true); !e.empty()) return fail("decode (verify): " + e);
        if (vlg.size() != block.size() * V) return fail("decode (verify): the pass returned " + std::to_string(vlg.size() / V) + " rows of logits for " + std::to_string(block.size()));
        uint32_t acc = 0; int32_t next = -1; bool ended = false, drop_last = false;
        for (uint32_t rr = 0; rr <= d.size(); ++rr) {
            float* lr = vlg.data() + size_t(rr) * V;
            if (!finite_row(lr, V, pos + rr)) return r;
            if (rr == d.size()) { next = Ds41Generator::sample_row(lr, V, recent, p, rng); break; }      // every draft kept: the next token
            if (ss) {
                Ds41Generator::distribution(lr, V, recent, p, p_ids, p_probs);
                const QRow& qr = qrows[rr];
                for (size_t i = 0; i < qr.ids.size(); ++i) qdense[size_t(qr.ids[i])] = float(qr.probs[i]);
                const SpecStep st = spec_accept(p_ids, p_probs, d[rr], [&](int32_t t) { return double(qdense[size_t(t)]); }, [&] { return Ds41Generator::uniform(rng); });
                for (size_t i = 0; i < qr.ids.size(); ++i) qdense[size_t(qr.ids[i])] = 0.f;
                if (!st.accepted) { next = st.token; break; }
            } else {
                const int32_t a = Ds41Generator::sample_row(lr, V, recent, p, rng);
                if (a != d[rr]) { next = a; break; }
            }
            ++acc;
            if (!emit(d[rr])) { ended = true; drop_last = true; break; }      // (a stop token's row is not part of the conversation)
            if (++k == max_new) { ended = true; break; }
        }
        const uint32_t keep = 1 + acc - (drop_last ? 1u : 0u);                // the rows of `id` and of the drafts kept
        if (keep < block.size()) b.fw.rewind(pos + keep);
        b.live.insert(b.live.end(), block.begin(), block.begin() + long(keep));
        ++n_rounds; n_verified += d.size(); n_kept += acc;
        if (ended) break;
        id = next; row = acc;                                                 // (`next` came from the row of the last draft kept)
    }
    if (drafting && n_rounds + n_plain)
        std::fprintf(stderr, "[hy4-flash] drafter: %llu rounds with drafts (%.2f verified, %.2f kept a round), %llu plain steps; %.1f ms of drafting a step%s\n",
                     static_cast<unsigned long long>(n_rounds), n_rounds ? double(n_verified) / double(n_rounds) : 0.0, n_rounds ? double(n_kept) / double(n_rounds) : 0.0,
                     static_cast<unsigned long long>(n_plain), ms_draft / double(n_rounds + n_plain), ss ? " (speculative sampling)" : "");
    if (!pending.empty()) { r.text += pending; if (on_token) on_token(pending); }
    r.decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dec).count();
    return r;
}
}  // namespace

std::string hy4f_render_prompt(std::span<const ChatTurn> turns, std::string_view tools_json, std::string_view effort, std::string& err,
                               std::vector<std::pair<size_t, size_t>>* content) {
    return hy4f_render(turns, tools_json, effort, err, content);
}

std::string hy4f_translate_reply(std::string_view raw) {
    std::string s(raw);
    for (const auto& [from, to] : kTags) {
        const std::string_view f(from), t(to);
        for (size_t at = s.find(f); at != std::string::npos; at = s.find(f, at + t.size())) s.replace(at, f.size(), t);
    }
    return s;
}

bool Engine::hy4flash_dir(const std::string& path) {
    std::ifstream f(path + "/hy4flash_ie.json");
    return bool(f);
}

std::string Engine::hy4flash_load(const std::string& dir) {
    auto b = std::make_unique<Hy4FlashBundle>();
    const auto t0 = std::chrono::steady_clock::now();
    if (auto e = b->model.load(dir); !e.empty()) return "hy4-flash: " + e;
    const Hy4fConfig& c = b->model.config();
    if (auto e = b->tok.load_from_hf_json(dir + "/tokenizer.json", dir + "/tokenizer_config.json"); !e.empty()) return "hy4-flash tokenizer: " + e;
    // stop ids: generation_config.json's eos_token_id (an int or a list), the config's, and the turn-end token
    {
        std::ifstream g(dir + "/generation_config.json");
        if (g) {
            const ojson j = ojson::parse(g, nullptr, false);
            if (!j.is_discarded() && j.contains("eos_token_id")) {
                const auto& e = j["eos_token_id"];
                if (e.is_array()) { for (const auto& x : e) if (x.is_number_integer()) b->eos.push_back(x.get<int32_t>()); }
                else if (e.is_number_integer()) b->eos.push_back(e.get<int32_t>());
            }
        }
        auto add = [&](int32_t id) { if (id >= 0 && std::find(b->eos.begin(), b->eos.end(), id) == b->eos.end()) b->eos.push_back(id); };
        add(c.eos_id);
        if (const auto e = b->tok.encode(kEnd, true); e.size() == 1) add(e[0]);
    }
    for (const auto& [from, to] : kTags)
        if (const auto e = b->tok.encode(from, true); e.size() == 1) b->stream_text.emplace_back(e[0], to);
    if (b->stream_text.size() != std::size(kTags)) return "hy4-flash: the tokenizer does not hold the reply tags (<think:opensource> ...) as single tokens";

    std::vector<sycl::device> cards;
    for (const auto& pl : sycl::platform::get_platforms()) {
        if (pl.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : pl.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) cards.push_back(d);
    }
    if (cards.empty()) return "hy4-flash: no Arc GPU";
    // every visible card up to two (the forward's layout); the experts that do not fit them are computed from RAM
    const uint32_t n = std::min<uint32_t>(2, uint32_t(cards.size()));
    if (opts_.n_gpus > 1 && opts_.n_gpus != n) std::fprintf(stderr, "[hy4-flash] --gpus %u: running on %u card(s) (%zu visible, at most 2 used)\n", opts_.n_gpus, n, cards.size());
    opts_.n_gpus = n;
    std::vector<sycl::queue*> qs;
    for (uint32_t i = 0; i < n; ++i) {
        b->queues.push_back(std::make_unique<sycl::queue>(sycl::context(cards[i]), cards[i], sycl::property::queue::in_order{}));
        qs.push_back(b->queues.back().get());
    }
    if (opts_.parallel == kLanesAuto) {
        opts_.parallel = 1;
        std::fprintf(stderr, "%s\n", lanes_auto_line("hy4-flash", 1, 0, "no request lanes for this arch").c_str());
    } else if (opts_.parallel > 1)
        std::fprintf(stderr, "[hy4-flash] one request lane in this version; requests take turns (--parallel %u)\n", opts_.parallel);

    // The image tower, resident on the first card and allocated BEFORE the forward measures the free VRAM for its expert
    // slots (1.06 GiB at the default 1024 x 1024 pixel cap: about 7 slots a layer on that card). IE_HY4F_VISION=0 leaves it
    // out; IE_HY4F_VISION_MAX_PX raises the cap to the checkpoint's 2048 x 2048 (its scratch grows with it).
    {
        std::string problem;
        const char* vv = std::getenv("IE_HY4F_VISION");
        uint64_t max_px = 1024ull * 1024ull;
        if (const char* mp = std::getenv("IE_HY4F_VISION_MAX_PX"); mp && *mp) max_px = std::strtoull(mp, nullptr, 10);
        const auto ph = b->tok.encode(kImage, true);
        if (vv && *vv == '0') problem = "the image tower is switched off (IE_HY4F_VISION=0)";
        else if (!b->model.source().find("vit.embeddings.patch_embedding.weight")) problem = "this model directory holds no vision tensors (vit.*)";
        else if (ph.size() != 3 || ph[1] != c.image_token_id) problem = "the tokenizer does not hold the template's image tokens";
        else if (auto e = b->vis.init(*qs[0], b->model, max_px); !e.empty()) problem = e;
        std::string warn;
        if (auto e = vision_required_error("hy4-flash", problem, "keep the checkpoint's vit.* shards in the model directory and do not set IE_HY4F_VISION=0", vision_required_env(), &warn); !e.empty()) return e;
        if (problem.empty()) {
            b->vis_ready = true;
            Hy4fVisGeom g; const uint32_t side = uint32_t(std::sqrt(double(b->vis.max_px()))) / 32 * 32;
            hy4f_vis_plan(side, side, g, b->vis.max_px()); b->image_tokens = g.tokens();
            std::fprintf(stderr, "[vision] hy4-flash: tower resident on card 0 (%.2f GiB with scratch for %llu pixels; an image is up to %u rows of the prompt)\n",
                         double(b->vis.device_bytes()) / 1073741824.0, static_cast<unsigned long long>(b->vis.max_px()), b->image_tokens);
        } else { b->vis_error = problem; std::fprintf(stderr, "%s\n", warn.c_str()); }
    }
    Hy4fOptions o;
    b->ctx = opts_.max_ctx;
    // 512 rows a prompt chunk: a chunk reads each missed expert once for all its rows, so longer chunks read less, but
    // their workspaces take expert slots. Measured at ctx 262,144 on a 3,908-token prompt (slots a layer on the two cards,
    // prompt read, decode): 512 -> 169 / 156, 113.5 tok/s, 16.9; 1024 -> 165 / 151, 120.4, 17.3; 2048 -> 158 / 141,
    // 124.8, 16.4. --prefill-chunk sets it, up to 1024 (a launcher's 2048 from another model's preset costs decode).
    uint32_t want_chunk = opts_.prefill_chunk_set ? opts_.prefill_chunk : 512u;
    if (want_chunk > 1024) { std::fprintf(stderr, "[hy4-flash] --prefill-chunk %u: using 1024 (longer chunks take expert slots from the cards: 158 / 141 a layer at 2048 against 165 / 151)\n", want_chunk); want_chunk = 1024; }
    b->chunk = std::max<uint32_t>(1, std::min<uint32_t>(want_chunk, b->ctx));
    o.max_ctx = b->ctx; o.max_tokens = b->chunk;
    o.q8_experts = true; o.q8_dense = true;
    // A long prompt (more than IE_HY4F_BIG_MIN rows left, default 2048) reads in chunks of IE_HY4F_BIG_CHUNK rows (default
    // 8192; 0 = off) whose workspace sits in VRAM the expert slots lend for the prompt and get back, refilled, before the
    // reply (Hy4fQ8Store::lend). Each missed expert streams once a chunk, so fewer chunks read faster: two cards, 4,882-token
    // prompt 319 -> 478 tok/s, 25,216 tokens 381 -> 471 (2026-11-19, forced-pick KL unchanged at 0.00020). Decode keeps all
    // but ~4 slots a layer (the sliding layers' key rings sized for the chunk).
    if (const char* v = std::getenv("IE_HY4F_BIG_CHUNK"); v && *v) b->big = uint32_t(std::strtoul(v, nullptr, 10)); else b->big = 8192;
    if (const char* v = std::getenv("IE_HY4F_BIG_MIN"); v && *v) b->big_min = uint32_t(std::strtoul(v, nullptr, 10));
    if (b->big > b->chunk) { b->big = std::min<uint32_t>(b->big, b->ctx); o.big_tokens = b->big; } else b->big = 0;
    // IE_HY4F_SLOTS=N: exactly N expert slots a layer on each card (default: what free VRAM holds, rounded down to a multiple of 4)
    if (const char* sv = std::getenv("IE_HY4F_SLOTS"); sv && *sv) o.q8_slots = uint32_t(std::strtoul(sv, nullptr, 10));
    // IE_HY4F_DRAFT=1: the checkpoint's drafter. OFF by default: at the owner's launch (thinking on, temperature 0.9) it
    // measured 8-16 % SLOWER than plain decode (2026-11-19, the same requests: code greedy 21.4 vs 23.2 tok/s, code 0.9
    // 18.1 vs 21.1, prose 0.9 17.0 vs 20.3) -- reasoning text keeps 1.2-1.6 drafts a round, a verify pass reads ~3x the
    // experts of a single row, and its weights take 8 expert slots a layer from the last card. It pays only on code
    // written without thinking (+40-48 % there, ie-hy4flash-run --draft).
    { const char* dv = std::getenv("IE_HY4F_DRAFT"); o.drafter = dv && *dv == '1'; }
    if (const char* dc = std::getenv("IE_HY4F_DRAFT_CONF"); dc && *dc) b->draft_conf = float(std::atof(dc));
    {   // <dir>/expert_ranking.u32: [MoE layers][experts] u32, most-used first (which experts start on the cards)
        std::ifstream rf(dir + "/expert_ranking.u32", std::ios::binary);
        uint32_t n_moe = 0; for (uint32_t L = 0; L < c.n_layers; ++L) n_moe += c.moe[L];
        std::vector<uint32_t> flat(size_t(n_moe) * c.n_experts);
        if (rf && rf.read(reinterpret_cast<char*>(flat.data()), std::streamsize(flat.size() * 4))) {
            o.ranking.assign(c.n_layers, std::vector<uint32_t>(c.n_experts)); uint32_t ord = 0;
            for (uint32_t L = 0; L < c.n_layers; ++L) {
                if (c.moe[L]) { std::copy(flat.begin() + long(size_t(ord) * c.n_experts), flat.begin() + long(size_t(ord + 1) * c.n_experts), o.ranking[L].begin()); ++ord; }
                else for (uint32_t e = 0; e < c.n_experts; ++e) o.ranking[L][e] = e;
            }
        } else std::fprintf(stderr, "[hy4-flash] no %s/expert_ranking.u32: the cards start with the first experts of each layer (slower until the residency has adapted)\n", dir.c_str());
    }
    if (auto e = b->fw.init(qs, b->model, o); !e.empty()) return "hy4-flash: " + e;
    b->reuse = opts_.prompt_cache;
    b->draft = b->fw.has_drafter();
    if (b->draft) std::fprintf(stderr, "[hy4-flash] drafter on (IE_HY4F_DRAFT=1; DFly2: up to %u drafts a step, verified while its confidence is above %.2f)\n", b->fw.draft_width(), double(b->draft_conf));
    else if (o.drafter) std::fprintf(stderr, "[hy4-flash] IE_HY4F_DRAFT=1, but the checkpoint has no drafter this layout can use; plain decode\n");
    b->big = b->fw.big_rows();
    prompt_cache_on_ = opts_.prompt_cache;
    std::string slots;
    for (size_t i = 0; i < b->fw.n_cards(); ++i) slots += (i ? " / " : "") + std::to_string(b->fw.q8_slots_of(i));
    std::fprintf(stderr, "[hy4-flash] ready in %.1f s: %u card(s), ctx %u, prompt chunk %u (long prompts %u, in lent VRAM), 8-bit experts with %s slots a layer on the cards (of %u; the rest from RAM), prompt cache %s\n",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), n, b->ctx, b->chunk, b->big ? b->big : b->chunk, slots.c_str(), c.n_experts, b->reuse ? "on" : "off");
    arch_ = ModelArch::kHy4Flash;
    hy4f_ = std::move(b);
    return {};
}

GenerateResult Engine::hy4flash_generate(const std::string& prompt, const SamplingParams& sp, const TokenCallback& on_token) {
    std::lock_guard<std::mutex> lk(hy4f_mu_);
    const auto ids = hy4f_->tok.encode(prompt, /*allow_special=*/true);
    return hy4f_run(*hy4f_, ids, sp, on_token);
}

GenerateResult Engine::hy4flash_chat(std::span<const ChatTurn> turns, const SamplingParams& sp, const TokenCallback& on_token,
                                     bool enable_thinking, std::string_view tools_json, std::string_view reasoning_effort) {
    std::lock_guard<std::mutex> lk(hy4f_mu_);
    GenerateResult r;
    const std::string_view effort = !enable_thinking ? "no_think" : reasoning_effort == "max" ? "max" : "high";
    std::string err;
    std::vector<std::pair<size_t, size_t>> content;   // known_bugs 45: the renderer's own request-text ranges
    const std::string prompt = hy4f_render(turns, tools_json, effort, err, &content);
    if (!err.empty()) { r.finish_reason = "error: " + err; return r; }
    auto ids = hy4f_->tok.encode(prompt, /*allow_special=*/true, content);   // a tag spelled in the request's text stays text
    // the template writes no BOS: drop the tokenizer's own if it adds one
    if (!ids.empty() && hy4f_->tok.add_bos_token() && ids[0] == hy4f_->model.config().bos_id) ids.erase(ids.begin());
    // images: every image's single pad token becomes its rows (the processor's expansion); the rows' 3-D positions follow
    // the reference (t = start, h = start + row, w = start + column; the text goes on one past the largest)
    std::vector<PromptImage> images;
    for (const auto& t : turns) for (const auto& bytes : t.images) {
        if (!hy4f_->vis_ready) { r.finish_reason = "error: hy4-flash cannot read images on this load: " + hy4f_->vis_error; return r; }
        PromptImage im;
        if (auto e = hy4f_load_image_mem(bytes.data(), bytes.size(), im.pv, im.g, hy4f_->vis.max_px()); !e.empty()) { r.finish_reason = "error: " + e; return r; }
        im.n = im.g.tokens(); im.hash = 1469598103934665603ull;
        for (unsigned char ch : bytes) { im.hash ^= ch; im.hash *= 1099511628211ull; }
        images.push_back(std::move(im));
    }
    if (images.empty()) r = hy4f_run(*hy4f_, ids, sp, on_token);
    else {
        const int32_t pad = hy4f_->model.config().image_token_id;
        std::vector<int32_t> full, pos3; std::vector<uint8_t> is_embed; std::vector<int32_t> ipos;
        size_t k = 0; int32_t cur = 0;
        for (int32_t id : ids) {
            if (id != pad) { full.push_back(id); is_embed.push_back(0); pos3.insert(pos3.end(), {cur, cur, cur}); ++cur; continue; }
            if (k >= images.size()) { r.finish_reason = "error: the prompt holds more image placeholders than images"; return r; }
            PromptImage& im = images[k++]; im.pos0 = uint32_t(full.size());
            full.insert(full.end(), im.n, pad); is_embed.insert(is_embed.end(), im.n, 1);
            cur = hy4f_vis_positions(im.g, cur, ipos); pos3.insert(pos3.end(), ipos.begin(), ipos.end());
        }
        if (k != images.size()) { r.finish_reason = "error: the prompt holds fewer image placeholders than images"; return r; }
        PromptExtra mm{&images, &pos3, &is_embed};
        r = hy4f_run(*hy4f_, full, sp, on_token, &mm);
    }
    if (r.finish_reason == "stop" || r.finish_reason == "length") {
        auto pc = parse_glm5_completion(hy4f_translate_reply(r.text), enable_thinking, tools_json);
        r.reasoning_content = std::move(pc.reasoning);
        r.text = std::move(pc.content);
        r.tool_calls_json = pc.tool_calls.empty() ? "[]" : std::move(pc.tool_calls);
        if (r.tool_calls_json != "[]" && r.finish_reason == "stop") r.finish_reason = "tool_calls";
    }
    return r;
}

}  // namespace ie
