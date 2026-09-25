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
    std::fprintf(stderr, "[mimo26] prompt-lookup speculation %s (a copy of >= 12 context tokens verified up to %u rows at a time; IE_MIMO26_LOOKUP=1 on, =0 off)\n",
                 b->lookup ? "ON" : "off", Mimo26Forward::kDecodeRows);
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
    opts_.parallel = 1;
    mimo26_ = std::move(b);
    return {};
}

namespace {

GenerateResult mimo26_run_ids(Mimo26Bundle& b, const std::vector<int32_t>& ids, const SamplingParams& sp, const TokenCallback& on_token,
                              bool chat, bool thinking, const std::string& tools_json) {
    GenerateResult r;
    kmp_set_blocktime(0);   // the HTTP pool thread's own OpenMP team must not spin (ds41_engine.cpp)
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
    int32_t id = Ds41Generator::sample(logits, recent, p, rng);
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
            id = Ds41Generator::sample(logits, recent, p, rng);
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
            const int32_t a = Ds41Generator::sample_row(vlg.data() + size_t(rr) * V, V, recent, p, rng);
            if (rr == d.size() || a != d[rr]) { next = a; break; }
            ++acc;
            if (!emit(a)) { ended = true; drop_last = true; break; }   // an eos / abort row is not part of the context
            idx.push(a);
            if (++k_done == max_new) { ended = true; break; }
        }
        const uint32_t keep = 1 + acc - (drop_last ? 1u : 0u);         // the rows of id and the accepted drafts
        b.fwd.rewind(pos + keep);
        if (keep) add_ctx(keep, pos, uint32_t(rows.size()));
        if (from_df) { ++df_pass; df_acc += acc; }
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

}  // namespace

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
            if (!b.vis_ready) { r.finish_reason = "error: mimo_v2 image input: " + b.vis_error; return r; }
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
    return mimo26_run_ids(*mimo26_, ids, sp, on_token, /*chat=*/false, false, {});
}

uint32_t Engine::mimo26_prompt_cache_slots() const {
    const Mimo26Bundle& b = *mimo26_;
    if (!b.prefix_reuse) return 0;   // no prefix reuse at all, not even the live conversation
    return b.cache.guaranteed_slots(opts_.max_ctx, b.dflash != nullptr);
}

}  // namespace ie
