// src/engine/mimo26_engine.cpp — MiMo-V2.6 behind the Engine (P3b, docs/mimo26/00_PORT_PLAN.md): the opaque-bundle
// pattern of ds41_engine.cpp. A model DIRECTORY (config.json model_type mimo_v2 + safetensors + tokenizer.json) loads
// here; chat() renders the checkpoint's chat template, runs the two-card forward with live-conversation prefix reuse,
// streams the pieces, and parses reasoning / content / XML tool calls. One request at a time.
#include "ie/engine.hpp"

#include "ie/deepseek41_generate.hpp"   // Ds41Generator::sample: the engine's host sampler
#include "ie/expert_stream.hpp"
#include "ie/mimo26_engine.hpp"
#include "ie/mimo26_host_rules.hpp"
#include "ie/ngram_draft.hpp"
#include "ie/vitals.hpp"
#include "stb/stb_image.h"   // stbi_info_from_memory (the implementation is instantiated in qwen4_image.cpp)

#include "../../third_party/nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <omp.h>
#include <string>
#include <vector>

namespace ie {

namespace {

using ojson = nlohmann::ordered_json;

// Python json.dumps(ensure_ascii=False) string escaping.
void py_escape(const std::string& s, std::string& out) {
    out += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); out += b; }
                else out += char(c);
        }
    }
    out += '"';
}

void py_dump(const ojson& j, std::string& out) {
    if (j.is_object()) {
        out += '{';
        bool first = true;
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (!first) out += ", ";
            first = false;
            py_escape(it.key(), out); out += ": "; py_dump(it.value(), out);
        }
        out += '}';
    } else if (j.is_array()) {
        out += '[';
        for (size_t i = 0; i < j.size(); ++i) { if (i) out += ", "; py_dump(j[i], out); }
        out += ']';
    } else if (j.is_string()) {
        py_escape(j.get<std::string>(), out);
    } else if (j.is_boolean()) {
        out += j.get<bool>() ? "true" : "false";
    } else if (j.is_null()) {
        out += "null";
    } else {
        out += j.dump();   // numbers
    }
}

// The template's render_value: a string as it is, anything else through tojson.
std::string render_value(const ojson& v) {
    if (v.is_string()) return v.get<std::string>();
    std::string s; py_dump(v, s); return s;
}

// One assistant turn's OpenAI tool_calls array in the template's XML form. The OpenAI `arguments` is a JSON string:
// parsed into its object and rendered as <parameter=K>V</parameter> -- the form the model itself writes (the template's
// dict branch); an arguments string that is not a JSON object goes in as it is (the template's string branch).
std::string render_tool_calls(const std::string& tool_calls_json, std::string& err) {
    if (tool_calls_json.empty()) return {};
    ojson calls = ojson::parse(tool_calls_json, nullptr, false);
    if (!calls.is_array()) { err = "tool_calls is not a JSON array"; return {}; }
    std::string out;
    for (const auto& tc0 : calls) {
        const ojson& tc = tc0.contains("function") ? tc0["function"] : tc0.contains("custom") ? tc0["custom"] : tc0;
        out += "<tool_call><function=" + tc.value("name", std::string()) + ">";
        if (tc.contains("input") && tc["input"].is_string()) {
            out += tc["input"].get<std::string>();
        } else if (tc.contains("arguments")) {
            ojson args = tc["arguments"];
            if (args.is_string()) {
                ojson parsed = ojson::parse(args.get<std::string>(), nullptr, false);
                if (parsed.is_object()) args = parsed;
            }
            if (args.is_string()) out += args.get<std::string>();
            else if (args.is_object())
                for (auto it = args.begin(); it != args.end(); ++it) out += "<parameter=" + it.key() + ">" + render_value(it.value()) + "</parameter>";
        }
        out += "</function></tool_call>";
    }
    return out;
}

// A parameter's schema type in `tools` (the OpenAI array), "" when unknown.
std::string param_type(const ojson& tools, const std::string& fn, const std::string& param) {
    if (!tools.is_array()) return {};
    for (const auto& t : tools) {
        const ojson& f = t.contains("function") ? t["function"] : t;
        if (f.value("name", std::string()) != fn || !f.contains("parameters")) continue;
        const ojson& p = f["parameters"];
        if (!p.contains("properties") || !p["properties"].contains(param)) return {};
        const ojson& s = p["properties"][param];
        if (s.contains("type") && s["type"].is_string()) return s["type"].get<std::string>();
        return {};
    }
    return {};
}

// A string parameter whose schema lists an enum: a value that is not a member but becomes one once stray quotes, '>' and
// whitespace are trimmed is snapped to that member (MiMo at temperature 1 copies the enum's JSON quote: `status"` for the
// enum ["status", ...]); anything else is left as written.
std::string enum_snap(const ojson& tools, const std::string& fn, const std::string& param, std::string v, uint32_t& repaired) {
    if (!tools.is_array()) return v;
    for (const auto& t : tools) {
        const ojson& f = t.contains("function") ? t["function"] : t;
        if (f.value("name", std::string()) != fn || !f.contains("parameters")) continue;
        const ojson& p = f["parameters"];
        if (!p.contains("properties") || !p["properties"].contains(param)) return v;
        const ojson& e = p["properties"][param];
        if (!e.contains("enum") || !e["enum"].is_array()) return v;
        for (const auto& m : e["enum"]) if (m.is_string() && m.get<std::string>() == v) return v;
        auto junk = [](char c) { return c == '"' || c == '\'' || c == '>' || c == ' ' || c == '\n'; };
        size_t a = 0, b = v.size();
        while (a < b && junk(v[a])) ++a;
        while (b > a && junk(v[b - 1])) --b;
        const std::string w = v.substr(a, b - a);
        for (const auto& m : e["enum"]) if (m.is_string() && m.get<std::string>() == w) { ++repaired; return w; }
        return v;
    }
    return v;
}

// A parameter's text as the schema says: strings stay strings; numbers / booleans / objects / arrays parse as JSON
// (kept as the string when they do not); unknown types parse only when the text is JSON-looking.
ojson typed_value(const std::string& v, const std::string& type) {
    if (type == "string") return ojson(v);
    const bool looks = !v.empty() && (v[0] == '{' || v[0] == '[' || v[0] == '-' || (v[0] >= '0' && v[0] <= '9') ||
                                      v == "true" || v == "false" || v == "null");
    if (!type.empty() || looks) {
        ojson j = ojson::parse(v, nullptr, false);
        if (!j.is_discarded()) return j;
    }
    return ojson(v);
}

std::string strip_one_newline(std::string s) {
    if (!s.empty() && s.front() == '\n') s.erase(0, 1);
    if (!s.empty() && s.back() == '\n') s.pop_back();
    return s;
}

// Complete-UTF-8 prefix length of `s`.
size_t utf8_complete(const std::string& s) {
    size_t i = s.size();
    size_t back = 0;
    while (i > 0 && back < 4) {
        const unsigned char c = uint8_t(s[i - 1]);
        if ((c & 0xC0) != 0x80) {
            const size_t need = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
            return (s.size() - (i - 1) >= need) ? s.size() : i - 1;
        }
        --i; ++back;
    }
    return s.size();
}

}  // namespace

std::string mimo26_tojson(const std::string& json_text, std::string& err) {
    ojson j = ojson::parse(json_text, nullptr, false);
    if (j.is_discarded()) { err = "not valid JSON"; return {}; }
    std::string s; py_dump(j, s); return s;
}

std::string mimo26_render_chat(const std::vector<Mimo26Message>& msgs, const std::string& tools_json, bool add_generation_prompt,
                               bool thinking, std::string& err) {
    std::string out;
    if (!tools_json.empty()) {
        ojson tools = ojson::parse(tools_json, nullptr, false);
        if (!tools.is_array()) { err = "tools is not a JSON array"; return {}; }
        if (!tools.empty()) {
            out += "<|im_start|>system\nYou are provided with the following tools:\n\n<tools>";
            for (const auto& t : tools) { out += '\n'; py_dump(t, out); }
            out += "\n</tools><|im_end|>";
        }
    }
    for (const auto& m : msgs) {
        if (m.role == "assistant") {
            out += "<|im_start|>assistant\n<think>" + m.reasoning + "</think>" + m.content;
            out += render_tool_calls(m.tool_calls_json, err);
            if (!err.empty()) return {};
            out += "<|im_end|>";
        } else {
            out += "<|im_start|>" + m.role + "\n" + m.content + "<|im_end|>";
        }
    }
    if (add_generation_prompt) out += thinking ? "<|im_start|>assistant\n<think>" : "<|im_start|>assistant\n<think></think>";
    return out;
}

Mimo26Parsed mimo26_parse_completion(const std::string& text, bool thinking, const std::string& tools_json) {
    Mimo26Parsed r;
    std::string rest = text;
    if (thinking) {
        const size_t at = text.find("</think>");
        if (at == std::string::npos) { r.reasoning = text; return r; }        // cut inside the reasoning
        r.reasoning = text.substr(0, at);
        rest = text.substr(at + 8);
    }
    const size_t tc = rest.find("<tool_call>");
    if (tc == std::string::npos) { r.content = rest; return r; }
    const ojson tools = tools_json.empty() ? ojson() : ojson::parse(tools_json, nullptr, false);
    ojson calls = ojson::array();
    std::string unparsed;   // call blocks that could not be parsed, kept as text
    size_t p = tc;
    int idx = 0;
    while (p != std::string::npos) {
        const size_t f0 = rest.find("<function=", p), f1 = f0 == std::string::npos ? f0 : rest.find('>', f0);
        const size_t fe = f1 == std::string::npos ? f1 : rest.find("</function>", f1);
        const size_t te = fe == std::string::npos ? fe : rest.find("</tool_call>", fe);
        if (te == std::string::npos) { r.malformed_call = true; unparsed += rest.substr(p); break; }
        const size_t next_call = rest.find("<tool_call>", te + 12);
        const std::string name = rest.substr(f0 + 10, f1 - (f0 + 10));
        const std::string body = rest.substr(f1 + 1, fe - (f1 + 1));
        ojson args = ojson::object();
        bool bad = false;
        for (size_t q = body.find("<parameter="); q != std::string::npos; q = body.find("<parameter=", q)) {
            const size_t k1 = body.find('>', q);
            if (k1 == std::string::npos) { bad = true; break; }
            const std::string key = body.substr(q + 11, k1 - (q + 11));
            const size_t ve = body.find("</parameter>", k1), nx = body.find("<parameter=", k1);
            if (ve != std::string::npos && (nx == std::string::npos || ve < nx)) {
                args[key] = typed_value(enum_snap(tools, name, key, strip_one_newline(body.substr(k1 + 1, ve - (k1 + 1))), r.repaired), param_type(tools, name, key));
                q = ve + 12;
                continue;
            }
            // no </parameter> before the next parameter / the function's end: the value runs there (repair)
            const size_t end = nx == std::string::npos ? body.size() : nx;
            std::string v = body.substr(k1 + 1, end - (k1 + 1));
            while (!v.empty() && (v.back() == '\n' || v.back() == ' ')) v.pop_back();
            if (v.size() >= 2 && v.compare(v.size() - 2, 2, "\">") == 0) v.resize(v.size() - 2);
            args[key] = typed_value(enum_snap(tools, name, key, strip_one_newline(v), r.repaired), param_type(tools, name, key));
            ++r.repaired;
            q = end;
        }
        if (bad) { r.malformed_call = true; unparsed += rest.substr(p, te + 12 - p); }
        else {
            char id[32]; std::snprintf(id, sizeof id, "call_%08x_%d", unsigned(std::hash<std::string>{}(name + body)), idx++);
            calls.push_back({{"id", id}, {"type", "function"}, {"function", {{"name", name}, {"arguments", args.dump()}}}});
        }
        p = next_call;
    }
    if (calls.empty()) { r.content = rest; return r; }   // nothing parseable: the whole block stays text
    std::string content = rest.substr(0, tc);
    while (!content.empty() && (content.back() == '\n' || content.back() == ' ')) content.pop_back();
    if (!unparsed.empty()) content += (content.empty() ? "" : "\n") + unparsed;
    r.content = content;
    r.tool_calls_json = calls.dump();
    return r;
}

// ---- P6.2: images (docs/mimo26/00_PORT_PLAN.md "P6.2 design") -------------------------------------------------------
constexpr std::string_view kMimo26ImagePlaceholder = "<|vision_start|><|image_pad|><|vision_end|>";   // the template's own

std::string mimo26_place_images(std::string text, size_t n_images) {
    size_t markers = 0;
    for (size_t p = text.find(kChatImageMarker); p != std::string::npos; p = text.find(kChatImageMarker, p + kChatImageMarker.size())) ++markers;
    if (markers == n_images) {
        for (size_t p = text.find(kChatImageMarker); p != std::string::npos; p = text.find(kChatImageMarker, p + kMimo26ImagePlaceholder.size()))
            text.replace(p, kChatImageMarker.size(), kMimo26ImagePlaceholder);
        return text;
    }
    for (size_t p = text.find(kChatImageMarker); p != std::string::npos; p = text.find(kChatImageMarker)) text.erase(p, kChatImageMarker.size());
    std::string front;
    for (size_t i = 0; i < n_images; ++i) front += kMimo26ImagePlaceholder;
    return front + text;
}

int32_t mimo26_image_id(uint64_t image_hash, uint32_t slot) {
    uint64_t x = image_hash + 0x9E3779B97F4A7C15ull * (uint64_t(slot) + 1);
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 27; x *= 0x94D049BB133111EBull; x ^= x >> 31;
    return -1 - int32_t(x >> 33);
}

uint64_t mimo26_image_hash(const std::string& bytes) {
    uint64_t h = 0xCBF29CE484222325ull;
    for (char c : bytes) { h ^= uint8_t(c); h *= 0x100000001B3ull; }
    return h;
}

namespace {
// The image's dimensions from its header (no decode): the token count is known before anything is encoded.
std::string mimo26_image_size(const void* bytes, size_t nbytes, uint32_t& width, uint32_t& height) {
    int w = 0, h = 0, comp = 0;
    if (!stbi_info_from_memory(static_cast<const stbi_uc*>(bytes), int(nbytes), &w, &h, &comp) || w <= 0 || h <= 0)
        return std::string("vision: not a decodable image: ") + (stbi_failure_reason() ? stbi_failure_reason() : "unknown format");
    width = uint32_t(w); height = uint32_t(h);
    return {};
}
}  // namespace

bool Engine::mimo26_dir(const std::string& path) {
    std::ifstream f(path + "/config.json");
    if (!f) return false;
    const std::string cfg((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return cfg.find("\"mimo_v2\"") != std::string::npos;
}

std::string Engine::mimo26_load(const std::string& dir) {
    auto b = std::make_unique<Mimo26Bundle>();
    const auto t0 = std::chrono::steady_clock::now();
    if (auto e = b->model.load(dir); !e.empty()) return "mimo_v2: " + e;
    if (auto e = b->tok.load_from_hf_json(dir + "/tokenizer.json", dir + "/tokenizer_config.json"); !e.empty()) return "mimo_v2 tokenizer: " + e;
    // stop ids: generation_config.json eos_token_id (an int or a list), else config.json's
    {
        std::ifstream g(dir + "/generation_config.json");
        if (g) {
            nlohmann::json gc = nlohmann::json::parse(g, nullptr, false);
            if (gc.contains("eos_token_id")) {
                const auto& e = gc["eos_token_id"];
                if (e.is_array()) for (const auto& x : e) b->eos.push_back(x.get<int32_t>());
                else if (e.is_number()) b->eos.push_back(e.get<int32_t>());
            }
        }
        if (b->eos.empty()) b->eos.push_back(b->model.config().eos_id);
    }
    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d);
    }
    if (devs.empty()) return "mimo_v2: no Arc GPU";
    for (const auto& d : devs) { b->queues.push_back(std::make_unique<sycl::queue>(sycl::context(d), d, sycl::property_list{sycl::property::queue::in_order{}})); b->qs.push_back(b->queues.back().get()); }
    Mimo26Options mo;
    mo.max_ctx = opts_.max_ctx;
    mo.max_tokens = std::min<uint32_t>(2048, opts_.max_ctx);
    // P4 B4 (docs/mimo26/P4_B4_SERVE.md): --parallel N > 1 = N lanes. Lane 0 keeps --ctx; lanes 1..N-1 get --slot-ctx positions
    // each (0 = 32,768, capped at --ctx). Their caches come out of the auto static tier, and the forward's init refuses, with
    // the numbers, when they do not fit (the load-time refusal). --parallel 1 is the pre-lane engine, byte for byte.
    // P4 B30: no --parallel = the fixed pick (lanes_auto.hpp: 4 lanes at 16K, measured best on MiMo), two cards or more
    const bool auto_n = opts_.parallel == kLanesAuto;
    const LanesAutoPlan lp = auto_n ? lanes_auto_plan(LanesArch::kMimo26, uint32_t(devs.size()), true, opts_.slot_ctx)
                                    : LanesAutoPlan{std::max<uint32_t>(1, opts_.parallel), opts_.slot_ctx, false};
    const uint32_t n_lanes = lp.n;
    const uint32_t lane_ctx = n_lanes > 1 ? std::min<uint32_t>(opts_.max_ctx, lp.slot_ctx ? lp.slot_ctx : 32768u) : 0u;
    if (auto_n) {
        opts_.parallel = n_lanes;
        std::fprintf(stderr, "%s\n", lanes_auto_line("mimo_v2", n_lanes, lane_ctx, n_lanes > 1
            ? "the fixed pick, measured best on MiMo-V2.6 (b23, 2026-09-29: 23.5-23.9 tok/s alone, 26.4 for four); every lane is "
              "reserved out of the static expert tier (the forward's lane lines below give their VRAM); images need --parallel 1"
            : "the lanes need two cards").c_str());
    }
    mo.lanes = n_lanes; mo.lane_ctx = lane_ctx;
    lane_ctx_ = lane_ctx;   // P4 B38 (0 with one lane)
    // the static expert tier: sized from each card's free VRAM (P4 lever 2: -11 % per decode token vs a fixed 48 on
    // held-out chat, PPL within noise); IE_MIMO26_STATIC=N pins a count
    mo.n_static = 0;
    if (const char* v = std::getenv("IE_MIMO26_STATIC"); v && *v) mo.n_static = uint32_t(std::atoi(v));
    if (const char* v = std::getenv("IE_MIMO26_PINNED"); v && *v) mo.n_pinned = uint32_t(std::atoi(v));
    if (const char* v = std::getenv("IE_MIMO26_STREAM"); v && *v) mo.stream_slots = uint32_t(std::atoi(v));
    // the residency ranking (P4): IE_MIMO26_RANKING names one ("0" = index order), else <model>/ie_ranking_mimo26_chat.txt
    std::string rank_path = dir + "/ie_ranking_mimo26_chat.txt";
    if (const char* rp = std::getenv("IE_MIMO26_RANKING"); rp && *rp) rank_path = std::string(rp) == "0" ? std::string() : std::string(rp);
    if (!rank_path.empty() && std::ifstream(rank_path).good()) {
        if (auto e = ds4_expert_priority_read_layers(rank_path, b->model.config().n_routed_experts, b->model.config().n_layers, mo.ranking); !e.empty())
            return "mimo_v2 ranking: " + e;
        std::fprintf(stderr, "[mimo26] residency ranking: %s\n", rank_path.c_str());
    } else std::fprintf(stderr, "[mimo26] no residency ranking: index-order expert placement (slower)\n");
    // P5: the DFlash drafter (dflash/ in the checkpoint) on the last card, allocated BEFORE the forward so the auto static
    // tier sizes around it. ON when the checkpoint ships one (A-B-A vs lookup alone: -19 % per token on the held-out
    // prompts; greedy near-ties can resolve differently, as with lookup); IE_MIMO26_DFLASH=K drafts K (1..7), 0 = off
    const bool has_dflash = std::ifstream(dir + "/dflash/config.json").good();
    b->dflash_k = has_dflash ? 7u : 0u;
    if (const char* v = std::getenv("IE_MIMO26_DFLASH"); v && *v) b->dflash_k = uint32_t(std::atoi(v));
    if (const char* v = std::getenv("IE_MIMO26_DFLASH_MINP"); v && *v) b->dflash_minp = float(std::atof(v));
    if (!b->dflash_k) std::fprintf(stderr, "[mimo26] DFlash drafter off (%s)\n", has_dflash ? "IE_MIMO26_DFLASH=0" : "no dflash/ in the checkpoint");
    if (b->dflash_k) {
        b->dflash = std::make_unique<Mimo26DFlash>();
        if (auto e = b->dflash->load(dir); !e.empty()) return "mimo_v2 " + e;
        if (auto e = b->dflash->init(*b->qs.back(), b->model.embed.w->data, b->model.config().vocab_size, b->dflash->config().window); !e.empty())
            return "mimo_v2 " + e;
        b->dflash_k = std::min(b->dflash_k, b->dflash->config().block - 1);
        if (n_lanes > 1) if (auto e = b->dflash->add_lanes(n_lanes); !e.empty()) return "mimo_v2 dflash lanes: " + e;   // (B1: one context ring per lane)
    }
    // P6.2: the vision tower, staged in pinned host memory BEFORE the forward sizes its tiers (the pinned budget then sees
    // it, and the auto static tier on the first card leaves the encode block free). Nothing of it stays on a card between
    // images. IE_MIMO26_VISION=0 leaves it out; a failure here is reported on every image request instead of failing the load.
    if (const char* v = std::getenv("IE_MIMO26_VISION"); v && std::string(v) == "0") b->vis_error = "vision is disabled (IE_MIMO26_VISION=0)";
    else if (!b->model.store().find("visual.patch_embed.proj.weight")) b->vis_error = "this checkpoint has no vision tower";
    else if (b->model.config().dim != 4096 || !b->model.config().image_token_id) b->vis_error = "this checkpoint's text width or image token differs from the mimovl tower's";
    else {
        const auto tv = std::chrono::steady_clock::now();
        std::string e = b->vis_alloc.init_with(b->qs[0]->get_context(), b->qs[0]->get_device());
        if (e.empty()) e = b->vis.load_from([m = &b->model](const std::string& n) { return m->store().find(n); });
        if (e.empty()) e = b->vis.stage_host(b->vis_alloc);
        if (e.empty()) {
            b->vis_ready = true; b->vis_error.clear();
            mo.reserve_card0 = b->vis.encode_bytes();
            std::fprintf(stderr, "[mimo26] vision tower staged in pinned host memory in %.0f ms; %.0f MiB on the first card while an image encodes, nothing between\n",
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tv).count(), double(b->vis.encode_bytes()) / 1048576.0);
        } else { b->vis_error = e; std::fprintf(stderr, "[mimo26] vision tower NOT available: %s\n", e.c_str()); }
    }
    {   // per-image token budget: the tower's scratch cap (2,048 = 8,192 patches) unless a smaller IE_MIMO26_IMAGE_TOKENS
        const uint32_t cap = MimoVisionOptions{}.max_patches / (kMimoVisMerge * kMimoVisMerge);
        b->image_tokens = cap;
        if (const char* v = std::getenv("IE_MIMO26_IMAGE_TOKENS"); v && *v) b->image_tokens = std::min<uint32_t>(cap, std::max(1, std::atoi(v)));
    }
    if (auto e = b->fwd.init(b->qs, b->model, mo); !e.empty()) return "mimo_v2 forward: " + e;
    if (b->dflash) {
        b->dflash->set_head(b->fwd.head_weights());
        if (auto e = b->fwd.set_feature_layers(b->dflash->config().target_layers, b->dflash->config().window); !e.empty()) return "mimo_v2 " + e;
        std::fprintf(stderr, "[mimo26] DFlash drafter ON: up to %u drafts per pass, cut below p %.2f, %.2f GB on the last card (IE_MIMO26_DFLASH=0 turns it off)\n",
                     b->dflash_k, b->dflash_minp, b->dflash->vram_bytes() / 1e9);
    }
    if (const char* v = std::getenv("IE_MIMO26_PROMPT_CACHE"); !opts_.prompt_cache || (v && std::string(v) == "0")) b->prefix_reuse = false;
    // P7 (#70, docs/mimo26/P7_FIX64_FIX70.md): HOST SLOTS -- other conversations' states kept in host RAM and swapped back in
    // when a prompt matches one better than the live state. IE_MIMO26_PROMPT_CACHE_GIB is their budget (16; 0 = the live
    // conversation only), IE_MIMO26_CACHE_KEEP_FREE_GIB the MemAvailable floor a new slot must leave (24); with prefix reuse
    // off (IE_MIMO26_PROMPT_CACHE=0, --no-prompt-cache) there are none
    {
        Mimo26PrefixCache::Options co;
        if (const char* g = std::getenv("IE_MIMO26_PROMPT_CACHE_GIB"); g && *g) co.budget = uint64_t(std::max(0.0, std::atof(g)) * 1073741824.0);
        if (const char* g = std::getenv("IE_MIMO26_CACHE_KEEP_FREE_GIB"); g && *g) co.keep_free = uint64_t(std::max(0.0, std::atof(g)) * 1073741824.0);
        if (!b->prefix_reuse) co.budget = 0;
        // P4 B16: at --parallel > 1 a reuse under min_tokens counts as none (batch == solo: the prefill split no longer depends
        // on which short prefix a lane happened to hold); --parallel 1 reuses any length, as before
        if (n_lanes > 1) co.min_reuse = co.min_tokens;
        if (co.min_reuse && b->prefix_reuse)
            std::fprintf(stderr, "[mimo26] lanes: a prompt reuses a lane's state or a host slot only for %u or more tokens (below that it prefills from 0)\n", co.min_reuse);
        if (auto e = b->cache.init(b->fwd, b->dflash.get(), co); !e.empty()) return "mimo_v2 host slots: " + e;
        if (b->cache.slots_on())
            std::fprintf(stderr, "[mimo26] host slots ON: up to %.1f GiB of other conversations in host RAM, kept above %.1f GiB MemAvailable "
                                 "(IE_MIMO26_PROMPT_CACHE_GIB=0 off)\n", double(co.budget) / 1073741824.0, double(co.keep_free) / 1073741824.0);
        else if (b->prefix_reuse) std::fprintf(stderr, "[mimo26] host slots off (IE_MIMO26_PROMPT_CACHE_GIB=0): only the live conversation is reused\n");
    }
    // prompt-lookup speculation (P4: held-out Dream turns -13 % per token, 76 % of drafts accepted; greedy near-ties can
    // resolve differently from one-row decoding -- 1 of 7 distinct held-out outputs identical over 256 tokens, every
    // divergence a near-tie, gate P4-1). With the DFlash drafter on it defaults OFF: the drafter alone decodes the held-out
    // prompts 7 % faster than lookup-first and ties it on verbatim copies (P5 gate finding 2, results/mimo26/p5 df4);
    // IE_MIMO26_LOOKUP=1 forces it on (a copy then takes priority over the drafter), =0 off
    b->lookup = !b->dflash;
    if (const char* v = std::getenv("IE_MIMO26_LOOKUP"); v && *v) b->lookup = std::string(v) != "0";
    if (n_lanes > 1 && b->lookup) { b->lookup = false; std::fprintf(stderr, "[mimo26] prompt-lookup speculation is --parallel 1 only: off\n"); }
    std::fprintf(stderr, "[mimo26] prompt-lookup speculation %s (a copy of >= 12 context tokens verified up to %u rows at a time; IE_MIMO26_LOOKUP=1 on, =0 off)\n",
                 b->lookup ? "ON" : "off", Mimo26Forward::kDecodeRows);
    if (n_lanes > 1) {   // P4 B4: the lanes' serving state (Mimo26Serve); the pipe starts with the first request
        b->serve.lanes.resize(n_lanes);
        for (uint32_t l = 0; l < n_lanes; ++l) b->serve.lanes[l].cap = l == 0 ? opts_.max_ctx : lane_ctx;
        if (const char* v = std::getenv("IE_MIMO26_MIX_CHUNK"); v && *v) b->serve.mix_chunk = uint32_t(std::max(1L, std::atol(v)));
        b->serve.mix_chunk = std::min(b->serve.mix_chunk, b->fwd.max_tokens());
        // P4 B5 (docs/mimo26/P4_B5_DRAFT_BUDGET.md): the draft budget -- rows per group step shared by the lanes' anchors and drafts
        // while several lanes decode (0 = the B4 rule: plain rows); the acceptance weighting; the pipe's lanes per group (0 = AUTO)
        b->serve.draft_budget = kMimo26DraftBudgetDefault;
        if (const char* v = std::getenv("IE_MIMO26_DRAFT_BUDGET"); v && *v) b->serve.draft_budget = uint32_t(std::max(0L, std::atol(v)));
        if (b->serve.draft_budget > Mimo26Forward::kDecodeRows)
            return "mimo_v2: IE_MIMO26_DRAFT_BUDGET " + std::to_string(b->serve.draft_budget) + " exceeds a group step's " + std::to_string(Mimo26Forward::kDecodeRows) + " rows";
        if (const char* v = std::getenv("IE_MIMO26_DRAFT_WEIGHT"); v && *v) b->serve.draft_weight = *v == '1';
        if (const char* v = std::getenv("IE_MIMO26_GROUP_LANES"); v && *v) b->serve.group_lanes = uint32_t(std::max(0L, std::atol(v)));
        if (!b->dflash) b->serve.draft_budget = 0;
        // P4 B16 (docs/mimo26/P4_B16_ADAPTIVE_DRAFT.md): adaptive drafting -- at and above this many decoding lanes, plain rows
        // and no drafter context feeds (0 = no cap: the B5 budget alone)
        b->serve.draft_max_lanes = kMimo26DraftMaxLanesDefault;
        if (const char* v = std::getenv("IE_MIMO26_DFLASH_MAX_LANES"); v && *v) b->serve.draft_max_lanes = uint32_t(std::max(0L, std::atol(v)));
        std::fprintf(stderr, "[mimo26] lanes: %u (lane 0 %u positions, lanes 1..%u %u each; --slot-ctx); requests decode together through the lane pipe, "
                             "a prefill chunk is %u rows while another lane is busy (IE_MIMO26_MIX_CHUNK) and %u alone; the DFlash drafter drafts for a lane "
                             "alone as at --parallel 1, and with several decoding %s (docs/mimo26/P4_B4_SERVE.md, P4_B5_DRAFT_BUDGET.md)\n",
                     n_lanes, opts_.max_ctx, n_lanes - 1, lane_ctx, b->serve.mix_chunk, b->fwd.max_tokens(),
                     !b->dflash ? "(no drafter)"
                     : !b->serve.draft_budget ? "not at all: plain rows (IE_MIMO26_DRAFT_BUDGET=0, the B4 rule)"
                     : b->serve.draft_weight ? "within a budget of rows per group step (IE_MIMO26_DRAFT_BUDGET), split by each lane's acceptance (IE_MIMO26_DRAFT_WEIGHT=1)"
                                             : "within a budget of rows per group step (IE_MIMO26_DRAFT_BUDGET), an even share per lane");
        if (b->serve.draft_budget) std::fprintf(stderr, "[mimo26] draft budget %u rows per group step\n", b->serve.draft_budget);
        if (b->dflash) {
            if (b->serve.draft_max_lanes)
                std::fprintf(stderr, "[mimo26] adaptive drafting: with %u or more lanes decoding no lane drafts or feeds the drafter's context "
                                     "(IE_MIMO26_DFLASH_MAX_LANES; 0 = no cap)\n", std::max<uint32_t>(2, b->serve.draft_max_lanes));
            else std::fprintf(stderr, "[mimo26] adaptive drafting off (the default; IE_MIMO26_DFLASH_MAX_LANES=5 turns it on): the lanes draft at any count within the budget\n");
        }
        if (b->serve.group_lanes) std::fprintf(stderr, "[mimo26] %u lane(s) per group step (IE_MIMO26_GROUP_LANES; 0 = the pipe's AUTO)\n", b->serve.group_lanes);
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::string res = "{\"arch\":\"mimo_v2\",\"cards\":" + std::to_string(devs.size()) + ",\"capacity\":" + std::to_string(opts_.max_ctx) +
                      ",\"load_s\":" + std::to_string(int(s)) + ",\"vram_gb\":[";
    for (size_t c = 0; c < devs.size(); ++c) res += std::string(c ? "," : "") + std::to_string(b->fwd.vram_bytes(c) / 1e9);
    res += "],\"pinned_gb\":" + std::to_string(b->fwd.pinned_bytes() / 1e9) + "}";
    memory_residency_json_ = res;
    std::fprintf(stderr, "[mimo26] resident in %.0f s on %zu card(s), capacity %u tokens, SWA ring %u slots, prefix reuse %s, static/pinned experts per layer",
                 s, devs.size(), opts_.max_ctx, b->fwd.ring(), b->prefix_reuse ? "on" : "off");
    for (size_t c = 0; c < devs.size(); ++c) std::fprintf(stderr, " %u/%u", b->fwd.card_static(c), b->fwd.card_pinned(c));
    std::fprintf(stderr, "\n");
    arch_ = ModelArch::kMimo26;
    mimo26_ = std::move(b);
    return {};
}

namespace {

GenerateResult mimo26_run_ids(Mimo26Bundle& b, const std::vector<int32_t>& ids, const SamplingParams& sp, const TokenCallback& on_token,
                              bool chat, bool thinking, const std::string& tools_json) {
    GenerateResult r;
    kmp_set_blocktime(0);   // the HTTP pool thread's own OpenMP team must not spin (ds41_engine.cpp)
    // ie_vitals (docs/mimo26/IE_VITALS.md): passive -- the sampler reports into `sv`, emit() copies it to the window;
    // nothing here reads the window back
    VitalsWindow* const vit = sp.vitals;
    if (vit) vit->active = true;
    Ds41SampleStats sv;
    Ds41SampleStats* const svp = vit ? &sv : nullptr;
    if (chat) r.tool_calls_json = "[]";
    const uint32_t cap = b.fwd.capacity();
    r.prompt_tokens = uint32_t(ids.size());
    if (ids.empty()) { r.finish_reason = "error: empty prompt"; return r; }
    if (ids.size() >= cap) { r.finish_reason = "context_length_exceeded"; return r; }
    const uint32_t room = cap - uint32_t(ids.size());
    const uint32_t max_new = sp.max_tokens == 0 ? room : std::min(sp.max_tokens, room);

    // what serves the prompt (P3b live reuse + P7 host slots, #70, docs/mimo26/P7_FIX64_FIX70.md): the live state's common
    // prefix while the SWA rings still hold the window before it (mimo26_servable), or another conversation's host slot
    // swapped in when it serves clearly more (the live conversation kept in a slot first); then the caches are cut there
    const auto t_pf = std::chrono::steady_clock::now();
    uint32_t L = 0;
    std::string source;
    if (b.prefix_reuse)
        if (auto e = b.cache.prepare(ids, b.live, L, source); !e.empty()) {
            b.live.clear(); b.fwd.reset(); if (b.dflash) b.dflash->reset(); b.cache.live_lost();
            r.finish_reason = "error: mimo_v2 prefix cache: " + e; return r;
        }
    b.fwd.rewind(L);
    b.live.resize(L);
    if (L == 0) b.fwd.reset();
    if (b.dflash) { if (L == 0) b.dflash->reset(); else b.dflash->rewind(L); }
    r.restore_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pf).count();
    // #64: a DRAFTER fault -- a draft or a context update that fails (a non-finite draft row, a feature fp16 cannot hold, a
    // SYCL error on its card) -- costs the drafter, not the request: logged once, drafting off for the rest of this request,
    // the drafter reset. The target's caches are intact; the next request's prefill re-syncs the drafter (the gap rule of
    // Mimo26DFlash::add_context). Only a TARGET forward failure clears the live state.
    bool df_on = b.dflash != nullptr;
    uint32_t df_fault_at = 0;   // completion tokens before the fault
    std::string df_reason;      // the fault's own words (a refused feature names its target layer and position)
    auto df_fail = [&](const std::string& what) {
        std::fprintf(stderr, "[mimo26 dflash] drafter fault at position %u (%u tokens generated): %s -- drafting off for the rest of this request, "
                             "the drafter reset; the target's caches are intact\n", b.fwd.n_pos(), r.completion_tokens, what.c_str());
        df_on = false; df_fault_at = r.completion_tokens; df_reason = what;
        b.dflash->reset();
    };
    // #74: a row about to be sampled that holds a non-finite logit is a TARGET fault and is never sampled from: the greedy
    // argmax of a NaN row is token 0 and the sampler's sort is undefined over one, so the reply would be silent garbage.
    // The caches may hold the poison (a NaN spreads through the full-attention layers): the live state is cleared, as for
    // any failed forward, and the request fails. Bit test (mimo26_scan_f32): the fast fp model may fold a float check away.
    auto sampled_ok = [&](const float* row, size_t n, uint32_t pos, const char* where) -> bool {
        const Mimo26Scan s = mimo26_scan_f32(row, n);
        if (!s.non_finite) return true;
        std::fprintf(stderr, "[mimo26] %s: %llu of %zu logits non-finite at position %u -- not sampled; the request fails and the conversation's "
                             "caches are cleared%s%s\n", where, (unsigned long long)s.non_finite, n, pos,
                     df_reason.empty() ? "" : "; earlier in this request the drafter refused: ", df_reason.c_str());
        b.live.clear(); b.fwd.reset(); if (b.dflash) b.dflash->reset(); b.cache.live_lost();
        r.finish_reason = std::string("error: mimo_v2 ") + where + ": non-finite logits at position " + std::to_string(pos) +
                          (df_reason.empty() ? std::string() : " (earlier: " + df_reason + ")");
        return false;
    };
    // the drafter's context follows every forward: the call's exported rows [p0, p0 + n) (a verify adds only its kept rows)
    auto add_ctx = [&](uint32_t n, uint32_t p0, uint32_t stride) {
        if (!df_on) return;
        std::string e;
        try { e = b.dflash->add_context(b.fwd.features(), n, p0, stride); } catch (const std::exception& x) { e = std::string("dflash: ") + x.what(); }
        if (!e.empty()) df_fail(e);
    };
    r.cached_tokens = L;
    r.cache_source = !b.prefix_reuse ? "" : source;
    // IE_MIMO26_LOGITS_FAULT=N (diagnostic, the #74 gate): the process's N-th logits row about to be sampled is overwritten
    // with NaN, as an fp16 overflow upstream leaves it
    auto inject_fault = [](float* row, size_t n) {
        static const long fault_at = [] { const char* v = std::getenv("IE_MIMO26_LOGITS_FAULT"); return v && *v ? std::atol(v) : 0L; }();
        static long n_rows = 0;
        if (++n_rows != fault_at) return;
        const uint32_t nan_bits = 0x7FC00000u;
        for (size_t i = 0; i < n; ++i) std::memcpy(row + i, &nan_bits, 4);
    };
    std::vector<float> logits;
    const uint32_t chunk = b.fwd.max_tokens();
    for (uint32_t off = L; off < ids.size(); off += chunk) {
        const uint32_t n = std::min<uint32_t>(chunk, uint32_t(ids.size()) - off);
        if (auto e = b.fwd.forward(ids.data() + off, n, off, logits, false); !e.empty()) {
            b.live.clear(); b.fwd.reset(); b.cache.live_lost();
            r.finish_reason = "error: mimo_v2 prefill: " + e; return r;
        }
        b.live.insert(b.live.end(), ids.begin() + off, ids.begin() + off + n);
        add_ctx(b.fwd.feat_rows(), off + n - b.fwd.feat_rows(), b.fwd.feat_rows());
        if (on_token && off + n < ids.size() && !on_token(std::string_view{})) { r.finish_reason = "abort"; return r; }   // liveness probe
    }
    inject_fault(logits.data(), logits.size());
    if (!sampled_ok(logits.data(), logits.size(), uint32_t(ids.size()) - 1, "prefill")) return r;
    // the live state holds the whole prompt (a slot it continued is dropped; #87: the prompt-end snapshot is taken)
    if (auto e = b.cache.prompt_done(uint32_t(ids.size())); !e.empty()) {
        b.live.clear(); b.fwd.reset(); if (b.dflash) b.dflash->reset(); b.cache.live_lost();
        r.finish_reason = "error: mimo_v2 prefix cache: " + e; return r;
    }
    r.prefill_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pf).count();

    Ds41SampleParams p; p.temperature = sp.temperature; p.top_k = sp.top_k; p.top_p = sp.top_p; p.min_p = sp.min_p;
    p.repeat_penalty = sp.repeat_penalty; p.repeat_window = sp.repeat_window;
    p.seed = sp.seed ? sp.seed : uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
    uint64_t rng = p.seed;
    std::vector<int32_t> recent(ids.end() - std::min<size_t>(ids.size(), 512), ids.end());
    std::string text, pending;
    const auto t_dec = std::chrono::steady_clock::now();
    r.finish_reason = "length";
    // one sampled token out: false = stop (an eos id, not emitted) or abort (the callback declined)
    auto emit = [&](int32_t id) -> bool {
        if (!sp.ignore_eos && std::find(b.eos.begin(), b.eos.end(), id) != b.eos.end()) { r.finish_reason = "stop"; return false; }
        ++r.completion_tokens;
        if (vit) vit->add_token(sv.has_H, sv.H, sv.margin);   // sv: the sample that produced id (the last one taken)
        recent.push_back(id);
        if (recent.size() > 512) recent.erase(recent.begin());
        pending += b.tok.decode(std::vector<int32_t>{id});
        const size_t done = utf8_complete(pending);
        if (done) {
            const std::string piece = pending.substr(0, done);
            pending.erase(0, done);
            text += piece;
            if (on_token && !on_token(piece)) { r.finish_reason = "abort"; return false; }
        }
        return true;
    };
    // Prompt-lookup speculation (P4, V4.1's docs/deepseek41/97 design): when the context's last >= 12 tokens occurred
    // before, the <= 7 tokens that followed are fed with the sampled id as one multi-row step (every row's logits); row
    // r's sample judges draft r and the drafts are kept while they equal it -- exact speculative sampling for a one-hot
    // draft, greedy included; the first mismatch's sample is the next id and rewind() drops the rejected rows (the SWA
    // ring's slots past the rewind hold only positions older than the window). No draft: a plain one-row step.
    Ds41NgramIndex idx;
    if (b.lookup) idx.reset(b.live);
    uint32_t n_pass = 0, n_rows = 0, n_acc = 0, n_plain = 0, df_pass = 0, df_acc = 0;
    const uint32_t V = b.model.config().vocab_size;
    std::vector<int32_t> rows; std::vector<float> vlg;
    int32_t id = Ds41Generator::sample(logits, recent, p, rng, svp);
    for (uint32_t k_done = 0;;) {
        if (!emit(id)) break;
        if (++k_done == max_new) break;   // the last token needs no forward
        std::vector<int32_t> d;
        if (b.lookup) {
            idx.push(id); d = idx.draft(Mimo26Forward::kDecodeRows - 1, 12);
            for (size_t i = 0; i < d.size(); ++i) if (d[i] < 0) { d.resize(i); break; }   // a copy of an image's pad run is not a draft (P6.2)
        }
        bool from_df = false;
        if (d.empty() && df_on) {
            // IE_MIMO26_DFLASH_FAULT=N (diagnostic, the #64 gate): the process's N-th draft fails as a drafter fault would
            static const long fault_at = [] { const char* v = std::getenv("IE_MIMO26_DFLASH_FAULT"); return v && *v ? std::atol(v) : 0L; }();
            static long n_drafts = 0;
            std::string e;
            try { e = ++n_drafts == fault_at ? std::string("dflash: injected fault (IE_MIMO26_DFLASH_FAULT)") : b.dflash->draft(id, b.dflash_k, d, b.dflash_minp); }
            catch (const std::exception& x) { e = std::string("dflash: ") + x.what(); }
            if (e.empty()) from_df = true;
            else { d.clear(); df_fail(e); }   // (a failed draft may hold the rows before the bad one: none is used)
        }
        if (d.size() > max_new - k_done) d.resize(max_new - k_done);
        const uint32_t pos = b.fwd.n_pos();
        if (d.empty()) {
            if (auto e = b.fwd.forward(&id, 1, pos, logits, false); !e.empty()) {
                b.live.clear(); b.fwd.reset(); b.cache.live_lost();
                r.finish_reason = "error: mimo_v2 decode: " + e; return r;
            }
            b.live.push_back(id); ++n_plain;
            add_ctx(1, pos, 1);
            inject_fault(logits.data(), logits.size());
            if (!sampled_ok(logits.data(), logits.size(), pos, "decode")) return r;
            id = Ds41Generator::sample(logits, recent, p, rng, svp);
            continue;
        }
        rows.assign(1, id); rows.insert(rows.end(), d.begin(), d.end());
        if (auto e = b.fwd.forward(rows.data(), uint32_t(rows.size()), pos, vlg, true); !e.empty()) {
            b.live.clear(); b.fwd.reset(); b.cache.live_lost();
            r.finish_reason = "error: mimo_v2 lookup verify: " + e; return r;
        }
        uint32_t acc = 0; int32_t next = -1; bool ended = false, drop_last = false;
        for (uint32_t rr = 0; rr <= d.size(); ++rr) {
            inject_fault(vlg.data() + size_t(rr) * V, V);
            if (!sampled_ok(vlg.data() + size_t(rr) * V, V, pos + rr, "verify")) return r;
            const int32_t a = Ds41Generator::sample_row(vlg.data() + size_t(rr) * V, V, recent, p, rng, svp);
            if (rr == d.size() || a != d[rr]) { next = a; break; }
            ++acc;
            if (!emit(a)) { ended = true; drop_last = true; break; }   // an eos / abort row is not part of the context
            idx.push(a);
            if (++k_done == max_new) { ended = true; break; }
        }
        const uint32_t keep = 1 + acc - (drop_last ? 1u : 0u);         // the rows of id and the accepted drafts
        b.fwd.rewind(pos + keep);
        if (keep) add_ctx(keep, pos, uint32_t(rows.size()));
        if (from_df) { ++df_pass; df_acc += acc; if (vit) vit->add_draft(uint32_t(d.size()), acc); }
        b.live.insert(b.live.end(), rows.begin(), rows.begin() + keep);
        ++n_pass; n_rows += uint32_t(rows.size()); n_acc += acc;
        if (ended) break;
        id = next;
    }
    if (b.dflash) std::fprintf(stderr, "[mimo26 dflash] %u passes, %u drafts accepted (%.2f per pass)%s\n", df_pass, df_acc, df_pass ? double(df_acc) / df_pass : 0.0,
                               df_on ? "" : ("; OFF after a drafter fault at token " + std::to_string(df_fault_at)).c_str());
    if (b.lookup) std::fprintf(stderr, "[mimo26 lookup] %u tokens: %u verify passes (%u rows, %u drafts accepted), %u plain steps\n",
                               r.completion_tokens, n_pass, n_rows, n_acc, n_plain);
    if (!pending.empty()) { text += pending; if (on_token) on_token(pending); }
    r.decode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_dec).count();
    if (!chat) { r.text = text; return r; }
    // IE_MIMO26_DUMP_REQ=<dir> (diagnostic): each chat's raw completion text, before parsing, as resp_<n>.txt
    if (const char* dd = std::getenv("IE_MIMO26_DUMP_REQ"); dd && *dd) {
        static int n_resp = 0;
        std::ofstream(std::string(dd) + "/resp_" + std::to_string(n_resp++) + ".txt", std::ios::binary) << text;
    }
    const Mimo26Parsed pc = mimo26_parse_completion(text, thinking, tools_json);
    if (pc.repaired || pc.malformed_call)
        std::fprintf(stderr, "[mimo26] tool calls: %u parameter(s) repaired (no </parameter>, or an enum value's stray quote)%s\n", pc.repaired,
                     pc.malformed_call ? "; an unparseable call block was returned as text" : "");
    r.reasoning_content = pc.reasoning;
    r.text = pc.content;
    if (!pc.tool_calls_json.empty()) { r.tool_calls_json = pc.tool_calls_json; if (r.finish_reason == "stop") r.finish_reason = "tool_calls"; }
    return r;
}

// ---- P4 B4: several requests at once on the lanes (docs/mimo26/P4_B4_SERVE.md) --------------------------------------------
// --parallel N > 1. mimo26_run_ids above stays the --parallel 1 engine, byte for byte. Here every request owns one lane; the
// lane pipe (Mimo26Forward::pipe_*, B2/B3) runs the steps -- decode groups of up to kDecodeRows rows across lanes, a prefill
// chunk alone; the pipe's done callback, on the last card's stage thread (serve_done), samples the lane's rows on the host,
// commits ids to the lane's outbox and submits the lane's next step; the request thread takes the SERIAL TURN (the pipe
// stopped) for the prefix step and the prompt-end snapshot, and otherwise only consumes its outbox. Drafter rule: a lane drafts
// with DFlash exactly as at --parallel 1 while it is the ONLY lane decoding (B4); with two or more decoding, the draft budget
// (P4 B5, docs/mimo26/P4_B5_DRAFT_BUDGET.md) shares a group step's rows between the lanes' anchors and their drafts -- budget 0
// is B4's plain one-row steps. Prompt-lookup speculation and images stay --parallel 1 only.
using SLane = Mimo26Serve::Lane;
using SPhase = Mimo26Serve::Lane::Phase;

double serve_ms(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
uint32_t serve_busy(const Mimo26Serve& s) { uint32_t n = 0; for (const auto& l : s.lanes) n += l.busy; return n; }
uint32_t serve_decoding(const Mimo26Serve& s) { uint32_t n = 0; for (const auto& l : s.lanes) n += l.busy && l.phase == SPhase::kDecode; return n; }

// The lane's request ends: `finish` unless one is set; `lost` = its device state is unusable and is cleared at the release
void serve_end(Mimo26Serve& s, SLane& l, const std::string& finish, bool lost) {
    if (l.finish.empty()) l.finish = finish;
    l.lost = l.lost || lost; l.phase = SPhase::kDone; l.parked = false; l.next = -1;
    s.cv.notify_all();
}

// A sampled id into the lane -- the serial path's emit rule: an eos id (unless ignore_eos) ends the lane with "stop" and is not
// committed; otherwise it goes to the outbox (with the stats of the sample that produced it, for ie_vitals) and the repetition
// window, and the lane ends with "length" at max_new. True while the lane goes on.
bool serve_commit(Mimo26Bundle& b, SLane& l, int32_t id, const Ds41SampleStats& st) {
    if (!l.ignore_eos && std::find(b.eos.begin(), b.eos.end(), id) != b.eos.end()) { l.finish = "stop"; return false; }
    ++l.n_new; ++b.serve.tokens;
    l.outbox.push_back({id, st.has_H, st.H, st.margin});
    l.recent.push_back(id);
    if (l.recent.size() > 512) l.recent.erase(l.recent.begin());
    if (l.n_new >= l.max_new) { l.finish = "length"; return false; }
    return true;
}

// (diagnostic, P4 B5 gate c) IE_MIMO26_GAP_LOG=1: every sampled row's pick and the row's top three raw logits, by lane and by the
// position the row predicts -- a divergence between two served runs of one prompt is then read against the first run's own gaps
// (the teacher-forced scans' near-tie rule: the other pick in the first run's top 3, within 0.75 of its top logit)
bool serve_gap_on() { static const bool on = [] { const char* v = std::getenv("IE_MIMO26_GAP_LOG"); return v && *v == '1'; }(); return on; }
void serve_gap_log(uint32_t li, uint32_t pos, const float* row, uint32_t V, int32_t pick) {
    if (V < 3) return;
    uint32_t t[3] = {0, 1, 2};
    std::sort(t, t + 3, [&](uint32_t a, uint32_t b) { return row[a] > row[b]; });
    for (uint32_t i = 3; i < V; ++i) {
        if (row[i] > row[t[0]]) { t[2] = t[1]; t[1] = t[0]; t[0] = i; }
        else if (row[i] > row[t[1]]) { t[2] = t[1]; t[1] = i; }
        else if (row[i] > row[t[2]]) t[2] = i;
    }
    std::fprintf(stderr, "[mimo26 gap] lane %u pos %u pick %d top1 %u %.6f top2 %u %.6f top3 %u %.6f\n", li, pos, pick, t[0], double(row[t[0]]),
                 t[1], double(row[t[1]]), t[2], double(row[t[2]]));
}

// #64 on a lane: a drafter fault costs the drafter, not the request (the lane's target caches are intact)
void serve_df_fail(Mimo26Bundle& b, SLane& l, uint32_t li, const std::string& what) {
    std::fprintf(stderr, "[mimo26 dflash] lane %u: drafter fault after %u tokens: %s -- drafting off for the rest of this request, the lane's drafter context reset\n",
                 li, l.n_new, what.c_str());
    l.df_on = false; l.df_reason = what;
    if (b.dflash && b.dflash->select_lane(li).empty()) b.dflash->reset();
}

// The drafter's context follows every step of a lane: the step's exported rows [p0, p0 + n), `stride` rows per layer. The
// caller's thread owns the drafter (the last card's stage thread inside the callback; the request thread with the pipe stopped).
void serve_df_ctx(Mimo26Bundle& b, SLane& l, uint32_t li, const float* feats, uint32_t n, uint32_t p0, uint32_t stride) {
    if (!l.df_on || !b.dflash || !n) return;
    // IE_MIMO26_DF_FEED=0: a lane's decode steps feed the drafter's context only while the lane is the one decoding (the feed is a
    // drafter GEMV on the last card per lane per step); when the lane is alone again, its context restarts at that position (the
    // drafter's gap rule: fewer accepted drafts at first, never a wrong token). Default 1: every step of every lane feeds it.
    // With a draft budget (B5) the lanes draft while several decode, so their contexts always follow.
    static const bool feed_always = [] { const char* v = std::getenv("IE_MIMO26_DF_FEED"); return !(v && *v == '0'); }();
    if (!feed_always && !b.serve.draft_budget && l.phase == SPhase::kDecode && serve_decoding(b.serve) > 1) return;
    // P4 B16: with IE_MIMO26_DFLASH_MAX_LANES lanes decoding or more no lane drafts, so a decode step does not feed the context
    // either; the lane is marked (df_gap) and drafts nothing until a later step's feed restarts its context there (the gap rule)
    if (l.phase == SPhase::kDecode && !mimo26_lanes_draft(serve_decoding(b.serve), b.serve.draft_max_lanes)) {
        l.df_gap = true; ++b.serve.df_feed_skipped;
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    std::string e = b.dflash->select_lane(li);
    if (e.empty()) { try { e = b.dflash->add_context(feats, n, p0, stride); } catch (const std::exception& x) { e = std::string("dflash: ") + x.what(); } }
    if (!e.empty()) serve_df_fail(b, l, li, e);
    else l.df_gap = false;
    l.dfctx_ms += serve_ms(t0);
}

// P4 B5: the drafts lane li offers in its next step while n_dec >= 2 lanes decode, within the draft budget (rows per group step).
// The groups hold the pipe's lanes per group g (IE_MIMO26_GROUP_LANES, else its AUTO: ceil(decoding / cards)). Even (the default):
// every lane floor(budget / g) rows -- its anchor and the rest drafts. Weighted (IE_MIMO26_DRAFT_WEIGHT=1), inside a done callback:
// the rows of the group being answered (its lanes' even shares, at most the budget) less what its earlier lanes resubmitted, split
// between this lane and the group's later lanes by their acceptance (mimo26_draft_k_weighted) -- so rows a lane's drafter cut
// leaves go to the lanes after it, and every group stays within the budget. Outside a callback (a turn's release) the even share.
uint32_t serve_draft_k(Mimo26Bundle& b, const SLane& l, uint32_t li, uint32_t n_dec) {
    const Mimo26Serve& s = b.serve;
    const uint32_t nc = std::max<uint32_t>(1, uint32_t(b.fwd.n_cards()));
    const uint32_t g = s.group_lanes ? s.group_lanes : std::max<uint32_t>(1, (n_dec + nc - 1) / nc);
    const uint32_t even = mimo26_draft_k_even(s.draft_budget, g, b.dflash_k);
    if (!s.draft_weight) return even;
    const std::vector<uint32_t>& G = b.fwd.pipe_cb_lanes();
    const auto it = std::find(G.begin(), G.end(), li);
    if (it == G.end()) return even;
    const uint32_t total = std::min<uint32_t>(s.draft_budget, (s.draft_budget / g) * uint32_t(G.size()));
    uint32_t pool = total > s.cb_used ? total - s.cb_used : 0;
    std::vector<double> acc{mimo26_draft_acc(l.acc_kept, l.acc_miss)};
    for (auto j = it + 1; j != G.end(); ++j) {   // the group's later lanes that will resubmit (their callbacks run next)
        const SLane& o = s.lanes[*j];
        if (!o.busy || o.phase != SPhase::kDecode || o.want_stop) continue;
        if (o.df_on && !o.df_gap) acc.push_back(mimo26_draft_acc(o.acc_kept, o.acc_miss));
        else if (pool) --pool;                   // (a lane without its drafter keeps its anchor row only)
    }
    return mimo26_draft_k_weighted(pool, acc.data(), uint32_t(acc.size()), b.dflash_k);
}

// The lane's next decode step: its next id, plus the DFlash drafts -- while it is the only lane decoding exactly as at --parallel 1
// (the B4 rule: dflash_k drafts), and while several decode the draft budget's share (P4 B5; budget 0: none, the B4 rule)
void serve_prepare_decode(Mimo26Bundle& b, SLane& l, uint32_t li) {
    Mimo26Serve& s = b.serve;
    l.rows.assign(1, l.next);
    if (!l.df_on || !b.dflash || l.df_gap) return;   // (B16: a skipped feed -- no drafts until the context is fed again)
    const uint32_t n_dec = serve_decoding(s);
    uint32_t k = b.dflash_k;
    if (n_dec != 1) {
        if (!s.draft_budget || !mimo26_lanes_draft(n_dec, s.draft_max_lanes)) return;   // (B16: adaptive drafting)
        if (!(k = serve_draft_k(b, l, li, n_dec))) return;
    }
    std::vector<int32_t> d;
    const auto t0 = std::chrono::steady_clock::now();
    std::string e = b.dflash->select_lane(li);
    if (e.empty()) { try { e = b.dflash->draft(l.next, k, d, b.dflash_minp); } catch (const std::exception& x) { e = std::string("dflash: ") + x.what(); } }
    const double ms = serve_ms(t0);
    l.draft_ms += ms; s.df_ms += ms; ++s.df_calls;
    if (n_dec != 1) { ++s.df_shared; ++l.shared_passes; }
    if (!e.empty()) { serve_df_fail(b, l, li, e); return; }
    if (d.size() > l.max_new - l.n_new) d.resize(l.max_new - l.n_new);
    l.rows.insert(l.rows.end(), d.begin(), d.end());
}

// The lane's step into the running pipe (mu held): its decode rows, or its next prefill chunk -- mix_chunk rows while another
// lane is busy or a request waits for the turn (the design's v1 mixing rule: one chunk, one decode step, in turns), else the
// forward's full chunk. mix_chunk defaults to the full chunk (no cap; the 512-row cap measured worse, docs/mimo26/P4_B4_SERVE.md).
// A refused submit ends the request (the pipe refuses everything after a stage error).
void serve_submit(Mimo26Bundle& b, SLane& l, uint32_t li) {
    Mimo26Serve& s = b.serve;
    l.parked = false; l.t_sub = std::chrono::steady_clock::now();
    std::string e;
    if (l.phase == SPhase::kPrefill) {
        const uint32_t left = uint32_t(l.prompt.size()) - l.prefill_at;
        const uint32_t cap = (serve_busy(s) > 1 || s.turn_waiters) ? s.mix_chunk : b.fwd.max_tokens();
        l.pend = std::min(left, cap); ++l.chunks;
        e = b.fwd.pipe_submit(li, l.prompt.data() + l.prefill_at, l.pend, l.prefill_at, false);
    } else {
        l.pend = 0;
        e = b.fwd.pipe_submit(li, l.rows.data(), uint32_t(l.rows.size()), b.fwd.lane_pos(li), l.rows.size() > 1);
    }
    if (!e.empty()) serve_end(s, l, "error: mimo_v2 lane " + std::to_string(li) + ": " + e, true);
}

// The pipe's done callback (the last card's stage thread): a prefill chunk landed, or a decode step's rows are sampled here.
void serve_done(Mimo26Bundle& b, uint32_t li, const std::vector<float>& logits, const float* feats, uint32_t frows) {
    Mimo26Serve& s = b.serve;
    std::lock_guard<std::mutex> lk(s.mu);
    SLane& l = s.lanes[li];
    const std::vector<uint32_t>& G = b.fwd.pipe_cb_lanes();   // (B5) the group this step ran in
    if (s.draft_weight && (G.empty() || G.front() == li)) s.cb_used = 0;   // (weighted) a new group: none of its rows resubmitted yet
    if (!l.busy || l.phase == SPhase::kDone) return;
    const auto t_cb = std::chrono::steady_clock::now();
    struct CbTime { SLane& l; std::chrono::steady_clock::time_point t0; ~CbTime() { l.cb_ms += serve_ms(t0); } } cb_time{l, t_cb};
    const double step = serve_ms(l.t_sub);
    // (diagnostic) IE_MIMO26_STEP_TRACE=1: every step's time, submit to callback, and how long after the pipe's latest start or
    // resume it landed -- the step times around the serial turns (docs/mimo26/P4_B4_SERVE.md)
    static const bool trace = [] { const char* v = std::getenv("IE_MIMO26_STEP_TRACE"); return v && *v == '1'; }();
    if (trace) std::fprintf(stderr, "[mimo26 step] lane %u %s %zu row(s) %.1f ms, %.0f ms into pipe run %llu, %u decoding\n", li,
                            l.phase == SPhase::kPrefill ? "prefill" : "decode", l.phase == SPhase::kPrefill ? size_t(l.pend) : l.rows.size(), step,
                            serve_ms(s.run_t0), (unsigned long long)s.runs, serve_decoding(s));
    if (l.phase == SPhase::kPrefill) {
        const uint32_t n = l.pend, at0 = l.prefill_at;
        serve_df_ctx(b, l, li, feats, frows, at0 + n - frows, frows);
        l.live.insert(l.live.end(), l.prompt.begin() + at0, l.prompt.begin() + at0 + n);
        l.prefill_at += n;
        if (l.prefill_at < l.prompt.size()) {
            if (l.want_stop) serve_end(s, l, "abort", false);           // the client left during the prompt
            else if (s.pause || s.stopping) l.parked = true;
            else serve_submit(b, l, li);
            s.cv.notify_all();                                          // (the request thread's liveness probe)
            return;
        }
        l.first_logits = logits;                                        // the prompt's last row, sampled under the turn
        l.phase = SPhase::kPromptReady;
        s.cv.notify_all();
        return;
    }
    // a decode step: rows[0] is the lane's anchor (committed already), rows[1..] its drafts; row r's sample judges draft r + 1
    // (the serial verify loop, exact speculative sampling for a one-hot draft)
    const uint32_t T = uint32_t(l.rows.size()), V = b.model.config().vocab_size;
    const uint32_t pos = b.fwd.lane_pos(li) - T;
    ++s.steps; ++l.steps;
    if (G.size() > 1) ++s.grouped_steps;
    s.step_ms = s.steps == 1 ? step : 0.9 * s.step_ms + 0.1 * step;
    s.rows_per_step = s.steps == 1 ? T : 0.9 * s.rows_per_step + 0.1 * T;
    uint32_t acc = 0; int32_t nxt = -1; bool ended = false, drop_last = false, rejected = false;
    Ds41SampleStats st;
    const auto t_samp = std::chrono::steady_clock::now();   // (diagnostic: the verify loop alone -- the scan, the row copy, the sampler)
    for (uint32_t r = 0; r < T; ++r) {
        const float* row = logits.data() + size_t(r) * V;
        if (const Mimo26Scan sc = mimo26_scan_f32(row, V); sc.non_finite) {   // #74: a target fault -- never sampled from; the caches may hold it
            std::fprintf(stderr, "[mimo26] lane %u decode: %llu of %u logits non-finite at position %u -- not sampled; the request fails and the lane's caches are cleared%s%s\n",
                         li, (unsigned long long)sc.non_finite, V, pos + r, l.df_reason.empty() ? "" : "; earlier the drafter refused: ", l.df_reason.c_str());
            serve_end(s, l, "error: mimo_v2 decode: non-finite logits at position " + std::to_string(pos + r) + (l.df_reason.empty() ? std::string() : " (earlier: " + l.df_reason + ")"), true);
            return;
        }
        l.scratch.assign(row, row + V);
        const int32_t a = Ds41Generator::sample_row(l.scratch.data(), V, l.recent, l.sp, l.rng, l.vitals ? &st : nullptr);
        if (serve_gap_on()) serve_gap_log(li, pos + r + 1, row, V, a);
        if (r + 1 == T || a != l.rows[r + 1]) { nxt = a; rejected = r + 1 < T; break; }
        ++acc;
        if (!serve_commit(b, l, a, st)) { ended = true; drop_last = l.finish == "stop"; break; }   // an eos row is not part of the context
    }
    l.samp_ms += serve_ms(t_samp);
    const uint32_t keep = 1 + acc - (drop_last ? 1u : 0u);              // the rows of the anchor and the accepted drafts
    if (keep < T) if (auto e = b.fwd.pipe_rewind_lane(li, pos + keep); !e.empty()) { serve_end(s, l, "error: mimo_v2 lane " + std::to_string(li) + ": " + e, true); return; }
    if (keep) serve_df_ctx(b, l, li, feats, keep, pos, frows);
    l.live.insert(l.live.end(), l.rows.begin(), l.rows.begin() + keep);
    if (T > 1) {
        ++l.df_pass; l.df_acc += acc; l.drafts.emplace_back(T - 1, acc);
        l.acc_kept = 0.75 * l.acc_kept + acc; l.acc_miss = 0.75 * l.acc_miss + (rejected ? 1.0 : 0.0);   // (B5: mimo26_draft_acc)
        s.df_offered += T - 1; s.df_accepted += acc;
    }
    if (!ended) { if (serve_commit(b, l, nxt, st)) l.next = nxt; else ended = true; }
    if (ended) { serve_end(s, l, l.finish, false); return; }
    if (l.want_stop) { serve_end(s, l, "abort", false); return; }
    if (s.pause || s.stopping) { l.parked = true; s.cv.notify_all(); return; }
    serve_prepare_decode(b, l, li);
    serve_submit(b, l, li);
    if (s.draft_weight) s.cb_used += uint32_t(l.rows.size());   // (B5, weighted: this group's rows so far)
    s.cv.notify_all();
}

// After a stage error the forward refuses every step until pipe_stop: every running request fails, the lanes' caches are cleared
void serve_pipe_failed(Mimo26Serve& s, const std::string& e) {
    std::fprintf(stderr, "[mimo26 lanes] the lane pipe failed: %s -- every running request fails and the lanes' caches are cleared\n", e.c_str());
    for (auto& l : s.lanes) if (l.busy && l.phase != SPhase::kDone) serve_end(s, l, "error: mimo_v2 pipe: " + e, true);
}

// The serial turn (lk holds mu): after the current holder; then the pipe is PAUSED -- the callbacks park their lanes at their
// next completion (pause), pipe_pause waits for the steps in flight; the stage threads stay, and their thread-local oneDNN
// contexts (the forward's and the drafter's primitives) with them. A stage error stops the pipe (pipe_stop clears it; the next
// release starts a new one) and fails every lane. IE_MIMO26_PIPE_PAUSE=0 stops the pipe at every turn and starts it again at
// the release (new stage threads: the 5b71003 behaviour, kept for the A/B in docs/mimo26/P4_B4_SERVE.md).
void serve_take_turn(Mimo26Bundle& b, std::unique_lock<std::mutex>& lk) {
    static const bool keep_stages = [] { const char* v = std::getenv("IE_MIMO26_PIPE_PAUSE"); return !(v && *v == '0'); }();
    Mimo26Serve& s = b.serve;
    ++s.turn_waiters;
    s.cv.wait(lk, [&] { return !s.turn_busy; });
    --s.turn_waiters;
    s.turn_busy = true;
    if (s.piping) {
        s.pause = true;
        lk.unlock();
        std::string e = b.fwd.pipe_pause();
        if (!e.empty() || !keep_stages) { const std::string e2 = b.fwd.pipe_stop(); if (e.empty()) e = e2; }
        lk.lock();
        s.piping = false; s.pause = false;
        if (!e.empty()) serve_pipe_failed(s, e);
    }
    ++s.turns; s.turn_t0 = std::chrono::steady_clock::now();
}

// Releases the turn: the parked lanes' steps (the drafts prepared first, while the pipe is paused and the drafter is this
// thread's) go into the resumed pipe (a new one when none is paused: the first request, or after a stage error)
void serve_release_turn(Mimo26Bundle& b, std::unique_lock<std::mutex>& lk) {
    Mimo26Serve& s = b.serve;
    std::vector<uint32_t> due;
    for (uint32_t li = 0; li < s.lanes.size(); ++li) {
        SLane& l = s.lanes[li];
        if (!l.busy || !l.parked) continue;
        if (l.want_stop) { serve_end(s, l, "abort", false); continue; }
        due.push_back(li);
    }
    if (!due.empty() && !s.stopping) {
        for (uint32_t li : due) if (s.lanes[li].phase == SPhase::kDecode) serve_prepare_decode(b, s.lanes[li], li);
        // (B5: a group holds at most the draft budget's rows -- 0 = the pipe's kDecodeRows, the B4 pipe)
        const std::string e = b.fwd.pipe_paused() ? b.fwd.pipe_resume()
            : b.fwd.pipe_start([&b](uint32_t li, const std::vector<float>& lg, const float* f, uint32_t n) { serve_done(b, li, lg, f, n); },
                               s.group_lanes, s.draft_budget);
        if (!e.empty()) {
            for (uint32_t li : due) serve_end(s, s.lanes[li], "error: mimo_v2 pipe: " + e, true);
        } else {
            s.piping = true; ++s.runs; s.run_t0 = std::chrono::steady_clock::now();
            for (uint32_t li : due) serve_submit(b, s.lanes[li], li);
        }
    }
    (void)lk;
    s.paused_ms += serve_ms(s.turn_t0);
    s.turn_busy = false;
    s.cv.notify_all();
}

// A stage error whose group ran no callback (a lone failing lane) is seen from the request threads' waits
void serve_poll_pipe(Mimo26Bundle& b, std::unique_lock<std::mutex>& lk) {
    if (!b.serve.piping || b.fwd.pipe_error().empty()) return;
    serve_take_turn(b, lk);
    serve_release_turn(b, lk);
}

// The lane for a prompt (the turn held), by mimo26_choose_lane (include/ie/mimo26_host_rules.hpp) among the idle lanes it fits
// (ids < cap): first the lanes that leave its reply room -- min(max_tokens, a quarter of the lane) past the prompt; within
// them the one whose state serves the most of the prompt (the prefix cache's reading of the lane, with the forward selected
// on it) when that is at least the cache's min_tokens (1,024: below it, the prefill saved is not worth cutting or evicting a
// conversation -- an EMPTY lane comes first); then the smallest capacity (short prompts leave lane 0 to the long ones); then
// the least recently released. -1 = none fits now.
int serve_choose(Mimo26Bundle& b, const std::vector<int32_t>& ids, uint32_t max_tokens) {
    Mimo26Serve& s = b.serve;
    std::vector<Mimo26LaneView> v(s.lanes.size());
    for (uint32_t li = 0; li < s.lanes.size(); ++li) {
        const SLane& l = s.lanes[li];
        v[li].idle = !l.busy; v[li].occupied = !l.live.empty(); v[li].cap = l.cap; v[li].tick = l.tick;
        if (l.busy || l.cap <= ids.size() || !b.prefix_reuse || l.live.empty()) continue;
        std::string e = b.fwd.select_lane(li);
        if (e.empty() && b.dflash) e = b.dflash->select_lane(li);
        if (e.empty()) v[li].match = b.cache.servable(l.live, ids);
    }
    return mimo26_choose_lane(v, uint32_t(ids.size()), max_tokens, b.cache.options().min_tokens);
}

// A lost lane (the turn held): its caches, its drafter context and its prefix bookkeeping are forgotten
void serve_clear_lane(Mimo26Bundle& b, SLane& l, uint32_t li) {
    if (b.fwd.select_lane(li).empty()) b.fwd.reset();
    if (b.dflash && b.dflash->select_lane(li).empty()) b.dflash->reset();
    b.cache.live_lost();
    l.live.clear(); l.lost = false;
}

GenerateResult mimo26_run_lanes(Mimo26Bundle& b, const std::vector<int32_t>& ids, const SamplingParams& sp, const TokenCallback& on_token,
                                bool chat, bool thinking, const std::string& tools_json) {
    GenerateResult r;
    kmp_set_blocktime(0);
    Mimo26Serve& s = b.serve;
    VitalsWindow* const vit = sp.vitals;
    if (vit) vit->active = true;
    if (chat) r.tool_calls_json = "[]";
    r.prompt_tokens = uint32_t(ids.size());
    if (ids.empty()) { r.finish_reason = "error: empty prompt"; return r; }
    for (int32_t id : ids) if (id < 0) { r.finish_reason = "error: mimo_v2 image input: images are served at --parallel 1 only (P4 B4)"; return r; }
    uint32_t cap_max = 0;
    for (const auto& l : s.lanes) cap_max = std::max(cap_max, l.cap);
    if (ids.size() >= cap_max) { r.finish_reason = "context_length_exceeded"; return r; }
    const auto t_pf = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point t_dec{};

    std::unique_lock<std::mutex> lk(s.mu);
    // 1. a lane, chosen under the serial turn; a prompt only lane 0 fits waits for lane 0
    int li = -1;
    for (;;) {
        s.cv.wait(lk, [&] { if (s.stopping) return true; for (const auto& l : s.lanes) if (!l.busy && l.cap > ids.size()) return true; return false; });
        if (s.stopping) { r.finish_reason = "abort"; return r; }
        serve_take_turn(b, lk);
        li = serve_choose(b, ids, sp.max_tokens);
        if (li >= 0) break;
        serve_release_turn(b, lk);
    }
    const uint32_t lane = uint32_t(li);
    SLane& l = s.lanes[lane];
    l.busy = true; l.phase = SPhase::kPrefill; l.finish.clear(); l.want_stop = false; l.lost = false; l.parked = false;
    l.outbox.clear(); l.drafts.clear(); l.first_logits.clear(); l.rows.clear(); l.pend = 0; l.steps = 0; l.chunks = 0;
    l.df_on = b.dflash != nullptr; l.df_pass = l.df_acc = 0; l.df_reason.clear(); l.cb_ms = l.samp_ms = l.dfctx_ms = l.draft_ms = 0;
    l.acc_kept = 4.0; l.acc_miss = 1.0; l.shared_passes = 0;   // (B5: the acceptance prior, 0.8)
    l.vitals = vit != nullptr;
    l.sp = Ds41SampleParams{}; l.sp.temperature = sp.temperature; l.sp.top_k = sp.top_k; l.sp.top_p = sp.top_p; l.sp.min_p = sp.min_p;
    l.sp.repeat_penalty = sp.repeat_penalty; l.sp.repeat_window = sp.repeat_window;
    l.sp.seed = sp.seed ? sp.seed : uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
    l.rng = l.sp.seed; l.ignore_eos = sp.ignore_eos;
    l.recent.assign(ids.end() - std::ptrdiff_t(std::min<size_t>(ids.size(), 512)), ids.end());
    const uint32_t room = l.cap - uint32_t(ids.size());
    l.max_new = sp.max_tokens == 0 ? room : std::min(sp.max_tokens, room); l.n_new = 0;
    auto release_lane = [&] {   // (mu held) the lane goes idle with its live ids; the LRU order follows the release
        l.busy = false; l.phase = SPhase::kIdle; l.tick = ++s.tick;
        l.outbox.clear(); l.drafts.clear(); l.first_logits.clear(); l.first_logits.shrink_to_fit(); l.prompt.clear(); l.prompt.shrink_to_fit();
        s.cv.notify_all();
    };
    // 2. the prefix step on the lane (the turn held): what serves the prompt -- the lane's live state, a host slot swapped in
    // (the lane's conversation kept in one first), or its prompt-end snapshot -- then the caches cut there
    {
        std::string e = b.fwd.select_lane(lane);
        if (e.empty() && b.dflash) e = b.dflash->select_lane(lane);
        uint32_t L = 0; std::string source;
        if (e.empty() && b.prefix_reuse) e = b.cache.prepare(ids, l.live, L, source);
        if (!e.empty()) {
            serve_clear_lane(b, l, lane);
            r.finish_reason = "error: mimo_v2 prefix cache: " + e;
            release_lane(); serve_release_turn(b, lk);
            return r;
        }
        b.fwd.rewind(L); l.live.resize(L);
        if (L == 0) b.fwd.reset();
        if (b.dflash) { if (L == 0) b.dflash->reset(); else b.dflash->rewind(L); }
        r.restore_ms = serve_ms(t_pf); r.cached_tokens = L; r.cache_source = b.prefix_reuse ? source : "";
        l.prompt = ids; l.prefill_at = L;
        l.parked = true;   // the first chunk is due: the turn's release submits it
    }
    serve_release_turn(b, lk);
    // 3. the prompt runs through the pipe in chunks; after each one the server's liveness probe (an empty fragment) -- a client
    // that left ends the request at the next completion
    uint32_t seen = l.prefill_at;
    while (l.phase == SPhase::kPrefill) {
        s.cv.wait_for(lk, std::chrono::seconds(1));
        if (l.phase != SPhase::kPrefill) break;
        if (l.prefill_at != seen && on_token && !l.want_stop) {
            seen = l.prefill_at;
            lk.unlock();
            const bool alive = on_token(std::string_view{});
            lk.lock();
            if (!alive) l.want_stop = true;
        }
        serve_poll_pipe(b, lk);
    }
    // 4. the prompt's end (the turn held): the snapshot (#87) and the first token
    if (l.phase == SPhase::kPromptReady) {
        serve_take_turn(b, lk);
        if (l.phase == SPhase::kPromptReady) {
            std::string e = b.fwd.select_lane(lane);
            if (e.empty() && b.dflash) e = b.dflash->select_lane(lane);
            if (e.empty()) e = b.cache.prompt_done(uint32_t(ids.size()));
            if (!e.empty()) serve_end(s, l, "error: mimo_v2 prefix cache: " + e, true);
            else {
                r.prefill_ms = serve_ms(t_pf);
                const Mimo26Scan sc = mimo26_scan_f32(l.first_logits.data(), l.first_logits.size());
                if (sc.non_finite) {
                    std::fprintf(stderr, "[mimo26] lane %u prefill: %llu of %zu logits non-finite at position %zu -- not sampled; the request fails and the lane's caches are cleared\n",
                                 lane, (unsigned long long)sc.non_finite, l.first_logits.size(), ids.size() - 1);
                    serve_end(s, l, "error: mimo_v2 prefill: non-finite logits at position " + std::to_string(ids.size() - 1), true);
                } else {
                    Ds41SampleStats st;
                    std::vector<float> raw;   // (diagnostic, IE_MIMO26_GAP_LOG: the sampler may change its copy's order; the row as it came)
                    if (serve_gap_on()) raw = l.first_logits;
                    const int32_t id = Ds41Generator::sample(l.first_logits, l.recent, l.sp, l.rng, l.vitals ? &st : nullptr);
                    if (serve_gap_on()) serve_gap_log(lane, uint32_t(ids.size()), raw.data(), uint32_t(raw.size()), id);
                    t_dec = std::chrono::steady_clock::now();
                    if (!serve_commit(b, l, id, st)) serve_end(s, l, l.finish, false);
                    else { l.next = id; l.phase = SPhase::kDecode; l.parked = true; }
                }
            }
            l.first_logits.clear(); l.first_logits.shrink_to_fit();
        }
        serve_release_turn(b, lk);
    }
    // 5. the outcome: ids as they land in the outbox, decoded to text and streamed (the callback declining cancels the lane)
    std::string text, pending;
    bool aborted = false;
    std::vector<Mimo26Serve::Tok> got; std::vector<std::pair<uint32_t, uint32_t>> gd;
    for (;;) {
        s.cv.wait_for(lk, std::chrono::seconds(1), [&] { return !l.outbox.empty() || !l.drafts.empty() || l.phase == SPhase::kDone; });
        got.swap(l.outbox); gd.swap(l.drafts);
        const bool done = l.phase == SPhase::kDone;
        if (!got.empty() || !gd.empty()) {
            lk.unlock();
            for (const auto& t : got) {
                ++r.completion_tokens;
                if (vit) vit->add_token(t.has_H, t.H, t.margin);
                pending += b.tok.decode(std::vector<int32_t>{t.id});
                const size_t d = utf8_complete(pending);
                if (d) {
                    const std::string piece = pending.substr(0, d);
                    pending.erase(0, d);
                    text += piece;
                    if (!aborted && on_token && !on_token(piece)) aborted = true;
                }
            }
            for (const auto& [off, acc] : gd) if (vit) vit->add_draft(off, acc);
            got.clear(); gd.clear();
            lk.lock();
            if (aborted) l.want_stop = true;
        }
        if (done) break;
        serve_poll_pipe(b, lk);
    }
    // 6. the finish
    r.finish_reason = aborted ? "abort" : (l.finish.empty() ? "length" : l.finish);
    if (t_dec.time_since_epoch().count()) r.decode_ms = serve_ms(t_dec);
    if (b.dflash) std::fprintf(stderr, "[mimo26 dflash] lane %u: %u passes, %u drafts accepted (%.2f per pass); %u draft calls while other lanes decoded; "
                                       "acceptance estimate %.2f%s\n", lane, l.df_pass, l.df_acc,
                               l.df_pass ? double(l.df_acc) / l.df_pass : 0.0, l.shared_passes, mimo26_draft_acc(l.acc_kept, l.acc_miss),
                               l.df_on ? "" : "; OFF after a drafter fault");
    std::fprintf(stderr, "[mimo26 lanes] lane %u: prompt %zu (%u cached%s%s) in %u chunk(s), %.0f ms; %u tokens (reply cap %u) in %u steps; %s; %u lane(s) busy; "
                         "callback %.0f ms (sampling %.0f, drafter context %.0f); drafting %.0f ms (in the callback or at a turn's release); %llu turns so far, "
                         "the pipe paused %.0f ms for them\n", lane, ids.size(),
                 r.cached_tokens, r.cache_source.empty() ? "" : " from ", r.cache_source.c_str(), l.chunks, r.prefill_ms, r.completion_tokens, l.max_new, l.steps,
                 r.finish_reason.c_str(), serve_busy(s), l.cb_ms, l.samp_ms, l.dfctx_ms, l.draft_ms, (unsigned long long)s.turns, s.paused_ms);
    if (l.lost) { serve_take_turn(b, lk); serve_clear_lane(b, l, lane); serve_release_turn(b, lk); }
    release_lane();
    lk.unlock();
    if (!pending.empty()) { text += pending; if (on_token && !aborted) on_token(pending); }
    if (!chat) { r.text = text; return r; }
    const Mimo26Parsed pc = mimo26_parse_completion(text, thinking, tools_json);
    if (pc.repaired || pc.malformed_call)
        std::fprintf(stderr, "[mimo26] tool calls: %u parameter(s) repaired (no </parameter>, or an enum value's stray quote)%s\n", pc.repaired,
                     pc.malformed_call ? "; an unparseable call block was returned as text" : "");
    r.reasoning_content = pc.reasoning;
    r.text = pc.content;
    if (!pc.tool_calls_json.empty()) { r.tool_calls_json = pc.tool_calls_json; if (r.finish_reason == "stop") r.finish_reason = "tool_calls"; }
    return r;
}

}  // namespace

std::string Engine::mimo26_serving_status_json() const {
    if (!mimo26_ || mimo26_->serve.lanes.empty()) return {};
    Mimo26Serve& s = mimo26_->serve;
    std::lock_guard<std::mutex> lk(s.mu);
    const auto [gate1, gaten] = mimo26_->fwd.pipe_gate_ms();
    char buf[768];
    // (B5) draft_*: the draft passes since load (every lane), their ms, the drafts offered / kept, and the passes with other lanes decoding
    std::snprintf(buf, sizeof buf, "{\"lanes_active\":%u,\"decoding\":%u,\"tokens\":%llu,\"step_ms\":%.1f,\"rows_per_step\":%.2f,\"turns\":%llu,\"paused_ms\":%.0f,"
                  "\"gate_ms_1\":%.1f,\"gate_ms_n\":%.1f,\"draft_budget\":%u,\"draft_calls\":%llu,\"draft_ms\":%.0f,\"draft_offered\":%llu,"
                  "\"draft_accepted\":%llu,\"draft_shared\":%llu,\"decode_steps\":%llu,\"grouped_steps\":%llu,\"draft_max_lanes\":%u,"
                  "\"draft_feeds_skipped\":%llu}", serve_busy(s), serve_decoding(s),
                  (unsigned long long)s.tokens, s.step_ms, s.rows_per_step,
                  (unsigned long long)s.turns, s.paused_ms, gate1, gaten, s.draft_budget, (unsigned long long)s.df_calls, s.df_ms,
                  (unsigned long long)s.df_offered, (unsigned long long)s.df_accepted, (unsigned long long)s.df_shared,
                  (unsigned long long)s.steps, (unsigned long long)s.grouped_steps, s.draft_max_lanes, (unsigned long long)s.df_feed_skipped);
    return buf;
}

GenerateResult Engine::mimo26_chat(std::span<const ChatTurn> turns, const SamplingParams& sp, const TokenCallback& on_token,
                                   bool enable_thinking, std::string_view tools_json, std::string_view reasoning_effort) {
    GenerateResult r;
    (void)reasoning_effort;   // validated by Engine::chat (MiMo has no effort levels)
    Mimo26Bundle& b = *mimo26_;
    // P6.2: one image of the request -- its bytes, its plan (from the header alone), and its rows once something asks
    struct Img { const std::string* bytes = nullptr; uint64_t hash = 0; MimoVisGeom geom; uint32_t n = 0, pos0 = 0; std::vector<float> rows; };
    std::vector<Img> images;
    std::vector<Mimo26Message> msgs;
    for (const auto& t : turns) {
        std::string content = t.content_without_tool_calls.value_or(t.content);
        if (!t.images.empty()) {
            if (auto why = mimo26_vision_refusal(b.vis_ready, b.vis_error, uint32_t(b.serve.lanes.size())); !why.empty()) {
                r.finish_reason = "error: mimo_v2 image input: " + why; return r;
            }
            content = mimo26_place_images(std::move(content), t.images.size());
            for (const auto& bytes : t.images) {
                Img im; im.bytes = &bytes; im.hash = mimo26_image_hash(bytes);
                uint32_t w = 0, h = 0;
                if (auto e = mimo26_image_size(bytes.data(), bytes.size(), w, h); !e.empty()) { r.finish_reason = "error: " + e; return r; }
                if (auto e = mimo26_vis_plan(h, w, im.geom, mimo26_image_max_px(b.image_tokens)); !e.empty()) { r.finish_reason = "error: " + e; return r; }
                im.n = im.geom.tokens();
                images.push_back(std::move(im));
            }
        }
        msgs.push_back({t.role, std::move(content), t.reasoning_content.value_or(""), t.tool_calls_json});
    }
    std::string err;
    const std::string prompt = mimo26_render_chat(msgs, std::string(tools_json), true, enable_thinking, err);
    if (!err.empty()) { r.finish_reason = "error: mimo_v2 chat template: " + err; return r; }
    if (const char* dd = std::getenv("IE_MIMO26_DUMP_REQ"); dd && *dd) {   // (diagnostic) the rendered prompt, req_<n>.txt
        static int n_req = 0;
        std::ofstream(std::string(dd) + "/req_" + std::to_string(n_req++) + ".txt", std::ios::binary) << prompt;
    }
    std::vector<int32_t> ids = b.tok.encode(prompt, /*allow_special=*/true);
    if (!b.serve.lanes.empty())   // P4 B4: --parallel > 1 (images were refused above: the pipe runs text positions only)
        return mimo26_run_lanes(b, ids, sp, on_token, /*chat=*/true, enable_thinking, std::string(tools_json));
    if (images.empty()) return mimo26_run_ids(b, ids, sp, on_token, /*chat=*/true, enable_thinking, std::string(tools_json));
    // the processor's expansion: each image's one <|image_pad|> becomes its N image ids (negative: the forward takes
    // their rows from the provider, the prefix cache matches them like any other ids)
    const int32_t pad = int32_t(b.model.config().image_token_id);
    std::vector<int32_t> full; full.reserve(ids.size() + images.size() * b.image_tokens);
    size_t k = 0;
    for (int32_t id : ids) {
        if (id != pad) { full.push_back(id); continue; }
        if (k >= images.size()) { r.finish_reason = "error: the prompt holds more image placeholders than images"; return r; }
        Img& im = images[k++]; im.pos0 = uint32_t(full.size());
        for (uint32_t s = 0; s < im.n; ++s) full.push_back(mimo26_image_id(im.hash, s));
    }
    if (k != images.size()) { r.finish_reason = "error: the prompt holds fewer image placeholders than images"; return r; }
    // an image is decoded and encoded on the first row asked of it: a prefix-cached image is never encoded again
    b.fwd.set_vision_provider([&b, &images](uint32_t pos, const float*& row) -> std::string {
        const uint32_t H = b.model.config().dim;
        for (auto& im : images) {
            if (pos < im.pos0 || pos - im.pos0 >= im.n) continue;
            if (im.rows.empty()) {
                const auto t0 = std::chrono::steady_clock::now();
                std::vector<float> pv; MimoVisGeom g;
                if (auto e = mimo26_load_image_mem(im.bytes->data(), im.bytes->size(), pv, g, mimo26_image_max_px(b.image_tokens)); !e.empty()) return e;
                if (g.tokens() != im.n) return "the image plan changed between the size probe and the decode";
                if (auto e = b.vis.encode_gpu(b.vis_alloc, pv, g, im.rows); !e.empty()) return e;
                if (im.rows.size() != size_t(im.n) * H) return "the tower returned " + std::to_string(im.rows.size() / H) + " rows for " + std::to_string(im.n) + " image tokens";
                std::fprintf(stderr, "[mimo26] vision: image at %u, %ux%u patches -> %u tokens, encoded in %.0f ms\n", im.pos0, g.grid_h, g.grid_w, im.n,
                             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            row = im.rows.data() + size_t(pos - im.pos0) * H;
            return {};
        }
        return "position " + std::to_string(pos) + " is outside every image of this request";
    });
    GenerateResult res = mimo26_run_ids(b, full, sp, on_token, /*chat=*/true, enable_thinking, std::string(tools_json));
    b.fwd.clear_vision();                                              // the provider captures this frame's locals
    return res;
}

GenerateResult Engine::mimo26_generate(const std::string& prompt, const SamplingParams& sp, const TokenCallback& on_token) {
    const auto ids = mimo26_->tok.encode(prompt, /*allow_special=*/true);
    if (!mimo26_->serve.lanes.empty()) return mimo26_run_lanes(*mimo26_, ids, sp, on_token, /*chat=*/false, false, {});   // P4 B4
    return mimo26_run_ids(*mimo26_, ids, sp, on_token, /*chat=*/false, false, {});
}

uint32_t Engine::mimo26_prompt_cache_slots() const {
    const Mimo26Bundle& b = *mimo26_;
    if (!b.prefix_reuse) return 0;   // no prefix reuse at all, not even the live conversation
    return b.cache.guaranteed_slots(opts_.max_ctx, b.dflash != nullptr);
}

}  // namespace ie
