// src/engine/ds41_engine.cpp — DeepSeek-V4.1-Flash behind the Engine (Phase 12 step 5,
// docs/deepseek41/31): the same opaque-bundle pattern as V4's Ds4Bundle. A model DIRECTORY
// (safetensors + config.json + tokenizer.json) loads here and never touches the GGUF path;
// chat() renders the V4.1 prompt format and runs Ds41Generator on the resident two-card
// runtime. Native V4.1 tools are encoded and parsed; images go through the checkpoint's own
// vision tower (Phase 57, docs/deepseek41/96). One request at a time (EngineOptions::parallel
// is 1 for this arch; the server's gate serialises).
#include "ie/engine.hpp"

#include "ie/ds41_engine.hpp"
#include "ie/deepseek41_generate.hpp"
#include "ie/deepseek41_prompt.hpp"
#include "ie/expert_stream.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <omp.h>
#include <string>
#include <vector>

namespace ie {

bool Engine::ds41_dir(const std::string& path) {
    std::ifstream f(path + "/config.json");
    if (!f) return false;
    const std::string cfg((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return cfg.find("deepseek_v41") != std::string::npos || cfg.find("DeepseekV41") != std::string::npos;
}

std::string Engine::ds41_load(const std::string& dir) {
    auto b = std::make_unique<Ds41Bundle>();
    const auto t0 = std::chrono::steady_clock::now();
    if (auto e = b->model.load(dir); !e.empty()) return "deepseek41: " + e;
    if (auto e = b->tables.load(dir); !e.empty()) return "deepseek41 engram tables (engram_tables.json + engram_token_map.i32 beside the model): " + e;
    if (auto e = b->tok.load_from_hf_json(dir + "/tokenizer.json", dir + "/tokenizer_config.json"); !e.empty()) return "deepseek41 tokenizer: " + e;
    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d);
    }
    if (devs.empty()) return "deepseek41: no Arc GPU";
    for (const auto& d : devs) { b->queues.push_back(std::make_unique<sycl::queue>(sycl::context(d), d, sycl::property_list{sycl::property::queue::in_order{}})); b->qs.push_back(b->queues.back().get()); }
    // Phase 57: the vision tower, staged in pinned host memory BEFORE the runtime sizes its pinned expert pool (the
    // live rule then sees it). Nothing of it stays on a card: encode_gpu() leases its block per image. IE_DS41_VISION=0
    // leaves it out; a failure here is reported on every image request instead of failing the load.
    if (const char* v = std::getenv("IE_DS41_VISION"); v && std::string(v) == "0") b->vis_error = "vision is disabled (IE_DS41_VISION=0)";
    else if (!b->model.config().vision_n_layers) b->vis_error = "this checkpoint has no vision tower";
    else {
        const auto tv = std::chrono::steady_clock::now();
        std::string e = b->vis_alloc.init_with(b->qs[0]->get_context(), b->qs[0]->get_device());
        if (e.empty()) e = b->vis.load_from([m = &b->model](const std::string& n) { return m->store().find(n); }, ds41_vision_options(b->model.config().dim));
        if (e.empty()) e = b->vis.stage_host(b->vis_alloc);
        if (e.empty()) {
            b->vis_ready = true; b->vis_error.clear();
            std::fprintf(stderr, "[deepseek41] vision tower staged in pinned host memory in %.0f ms; %.0f MiB on card 0 while an image encodes, nothing between\n",
                         std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tv).count(), double(b->vis.encode_bytes()) / 1048576.0);
        } else { b->vis_error = e; std::fprintf(stderr, "[deepseek41] vision tower NOT available: %s\n", e.c_str()); }
    }
    std::vector<std::vector<uint32_t>> ranking;
    // Phase 48 (docs/deepseek41/88): IE_DS41_RANKING names the residency ranking; otherwise a decode-phase chat ranking
    // beside the model (ie_ranking_decode_chat.txt) when present, else the held-out prefill profile
    std::string rank_path = dir + "/ie_ranking_heldout.txt";
    if (const char* rp = std::getenv("IE_DS41_RANKING"); rp && *rp) rank_path = rp;
    else if (std::ifstream(dir + "/ie_ranking_decode_chat.txt").good()) rank_path = dir + "/ie_ranking_decode_chat.txt";
    std::fprintf(stderr, "[deepseek41] residency ranking: %s\n", rank_path.c_str());
    if (std::ifstream(rank_path).good()) {
        if (auto e = ds4_expert_priority_read_layers(rank_path, b->model.config().n_routed_experts, b->model.config().n_layers, ranking); !e.empty()) return "deepseek41 ranking: " + e;
    } else std::fprintf(stderr, "[deepseek41] no %s: index-order expert placement (slower)\n", rank_path.c_str());
    // DSpark P4 (docs/deepseek41/58): the speculative loop's environment goes in BEFORE init_resident reads it
    // (gate P4 finding 1) -- the verify agrees with one-row steps bit for bit only with the CPU miss split off
    // (docs/55), and that split serves T = 1 misses only, so speculation leaves it idle. An explicit
    // IE_DS41_CPU_MISS wins and then the loop is not lossless, which the banner says.
    const bool spec = [] { const char* v = std::getenv("IE_DS41_SPEC"); return v && std::string(v) == "1"; }();
    if (spec) {
        setenv("IE_DS41_DECODE_MULTI", "1", 0);
        setenv("IE_DS41_CPU_MISS", "0", 0);
        const bool split_on = std::string(std::getenv("IE_DS41_CPU_MISS")) != "0";
        std::fprintf(stderr, "[deepseek41] DSpark speculation ON (temperature 0 only): the CPU miss split %s\n",
                     split_on ? "is ON by your IE_DS41_CPU_MISS -- the output may DIFFER from plain greedy (docs/58 gate finding 1)" : "off for exact losslessness");
    }
    // Phase 52 (docs/deepseek41/91): the expert tail file beside the model serves the mmap tier's hot window with one O_DIRECT
    // read per expert (no pack / permute) -- decode -6 %, prefill +6 % on Dream-style chats. Used when present;
    // IE_DS41_EXPERT_FILE names another, IE_DS41_EXPERT_FILE=0 turns it off. The tier verifies a slot byte for byte at load.
    if (const char* ef = std::getenv("IE_DS41_EXPERT_FILE"); ef && std::string(ef) == "0") unsetenv("IE_DS41_EXPERT_FILE");
    else if (!ef && std::ifstream(dir + "/ie_experts_tail.ieslot").good()) setenv("IE_DS41_EXPERT_FILE", (dir + "/ie_experts_tail.ieslot").c_str(), 0);
    Ds41Forward::ResidentOptions ro; ro.max_tokens = opts_.max_ctx;
    ro.head_fp8 = true;   // Phase 54 (founder decision, docs/deepseek41/93); IE_DS41_HEAD_FP8=0 restores the BF16 head
    // IE_DS41_VRAM_RESERVE_GIB: the per-card headroom kept free after the dense set (default 6 GiB, ResidentOptions);
    // the A/B knob for the static-expert budget -- the same name ds41_resident_test reads.
    if (const char* r = std::getenv("IE_DS41_VRAM_RESERVE_GIB"); r && *r) ro.vram_reserve = uint64_t(std::max(0.0, std::atof(r)) * 1073741824.0);
    if (auto e = b->fwd.init_resident(b->qs, b->model, b->tables, ranking, ro); !e.empty()) return "deepseek41 resident: " + e;
    b->fwd.set_logits_last_only(true);
    // Phase 46 (docs/deepseek41/86): the prefix cache, on unless the load disables it (--no-prompt-cache /
    // IE_NO_PROMPT_CACHE) or IE_DS41_PROMPT_CACHE=0; speculation keeps it off (the generator does too)
    if (const char* v = std::getenv("IE_DS41_PROMPT_CACHE"); opts_.prompt_cache && !spec && !(v && std::string(v) == "0")) {
        Ds41Forward::PrefixCacheOptions pco;
        if (const char* g = std::getenv("IE_DS41_PROMPT_CACHE_GIB")) pco.host_budget = uint64_t(std::atof(g) * 1073741824.0);
        // Phase 47 (docs/deepseek41/87): the disk store for prompt prefixes (a client's system prompt + tools), so a new
        // process does not re-prefill them. IE_DS41_PROMPT_CACHE_DIR names it ("0" or "" turns it off);
        // IE_DS41_PROMPT_CACHE_DISK_GIB its budget (8 by default)
        if (const char* d = std::getenv("IE_DS41_PROMPT_CACHE_DIR")) pco.disk_dir = std::string(d) == "0" ? std::string() : std::string(d);
        else if (const char* xdg = std::getenv("XDG_CACHE_HOME"); xdg && *xdg) pco.disk_dir = std::string(xdg) + "/machx-ie/deepseek41-prefix";
        else if (const char* home = std::getenv("HOME"); home && *home) pco.disk_dir = std::string(home) + "/.cache/machx-ie/deepseek41-prefix";
        if (const char* g = std::getenv("IE_DS41_PROMPT_CACHE_DISK_GIB")) pco.disk_budget = uint64_t(std::atof(g) * 1073741824.0);
        if (auto e = b->fwd.set_prefix_cache(true, pco); !e.empty()) return "deepseek41 prefix cache: " + e;
    }
    if (spec) {
        if (auto e = b->drafter.init(*b->qs.back(), b->model, b->fwd.dense_cache(uint32_t(b->qs.size() - 1)), {}, Ds41Drafter::Options{}); !e.empty()) return "deepseek41 drafter: " + e;
        std::fprintf(stderr, "[deepseek41] drafter %.2f GiB dense, tier %u static / %u pinned per stage\n",
                     double(b->drafter.dense_bytes()) / 1073741824.0, b->drafter.tier().n_static(), b->drafter.tier().n_pinned());
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::string res = "{\"arch\":\"deepseek_v41\",\"cards\":" + std::to_string(devs.size()) + ",\"capacity\":" + std::to_string(opts_.max_ctx) + ",\"load_s\":" + std::to_string(int(s)) + ",\"placement\":[";
    for (size_t c = 0; c < devs.size(); ++c) {
        const auto ci = b->fwd.card_info(uint32_t(c));
        res += std::string(c ? "," : "") + "{\"layers\":[" + std::to_string(ci.first_layer) + "," + std::to_string(ci.first_layer + ci.n_layers - 1) + "],\"static\":" + std::to_string(ci.n_static) + ",\"stream\":" + std::to_string(ci.n_stream) + ",\"pinned\":" + std::to_string(ci.n_pinned) + "}";
    }
    res += "]}";
    memory_residency_json_ = res;
    std::fprintf(stderr, "[deepseek41] resident in %.0f s on %zu card(s), capacity %u tokens\n", s, devs.size(), opts_.max_ctx);
    arch_ = ModelArch::kDeepSeek41;
    opts_.parallel = 1;
    if (const char* lv = std::getenv("IE_DS41_LOOKUP"); lv && std::string(lv) == "0") b->lookup = false;
    std::fprintf(stderr, "[deepseek41] prompt-lookup speculation %s (a copy of >= 12 context tokens verified up to 8 rows at a time; IE_DS41_LOOKUP=0 turns it off)\n",
                 b->lookup ? "ON" : "off");
    if (const char* po = std::getenv("IE_DS41_PROFILE_OUT"); po && *po) {
        b->profile_out = po;
        // The raw counts persist beside the ranking (<file>.counts) and are resumed here, so a profile accumulates over
        // restarts -- days of a user's own sessions (docs/89: the best ranking is one profiled on THEIR traffic), not
        // one process's.
        const auto& c = b->model.config();
        b->profile_acc.assign(c.n_layers, std::vector<uint64_t>(c.n_routed_experts, 0));
        const std::string cp = b->profile_out + ".counts";
        if (std::ifstream f(cp); f) {
            std::string magic; uint32_t nl = 0, ne = 0; uint64_t steps = 0;
            f >> magic >> nl >> ne >> steps;
            if (magic == "ie-ds41-profile-counts-v1" && nl == c.n_layers && ne == c.n_routed_experts) {
                for (auto& row : b->profile_acc) for (auto& v : row) f >> v;
                if (f) b->profile_steps = steps;
                else { b->profile_acc.assign(c.n_layers, std::vector<uint64_t>(c.n_routed_experts, 0)); std::fprintf(stderr, "[deepseek41] %s is truncated: starting the profile over\n", cp.c_str()); }
            } else std::fprintf(stderr, "[deepseek41] %s is not this model's profile (%u x %u): starting over\n", cp.c_str(), nl, ne);
        }
        std::fprintf(stderr, "[deepseek41] profiling decode routing into %s, counts in %s (resumed at %llu generated tokens)\n",
                     po, cp.c_str(), (unsigned long long)b->profile_steps);
    }
    ds41_ = std::move(b);
    return {};
}

namespace {
GenerateResult ds41_run_ids(Ds41Bundle& b, const std::vector<int32_t>& ids, const SamplingParams& sp, const TokenCallback& on_token, bool split_thinking, bool chat = false) {
    GenerateResult r;
    // Block time 0 on THIS thread, as the tier's init sets it on the loading thread: ie serve runs a request on an
    // HTTP pool thread, and that thread's team (the engram gather, every token) spun 200 ms after each region --
    // 19 workers at 40-100 % CPU through decode, beside the CPU miss path's cores (docs/deepseek41/35 step 3).
    kmp_set_blocktime(0);
    if (chat) r.tool_calls_json = "[]"; // authoritative native parser, including no calls
    Ds41SampleParams p; p.temperature = sp.temperature; p.top_k = sp.top_k; p.top_p = sp.top_p; p.min_p = sp.min_p;
    p.repeat_penalty = sp.repeat_penalty; p.repeat_window = sp.repeat_window; p.seed = sp.seed;
    const uint32_t max_new = sp.max_tokens == 0 || ids.size() + sp.max_tokens > b.fwd.capacity() ? (uint32_t(ids.size()) < b.fwd.capacity() ? b.fwd.capacity() - uint32_t(ids.size()) : 0) : sp.max_tokens;
    Ds41Generator gen(b.fwd, b.tok); if (b.drafter.ready()) gen.set_drafter(&b.drafter, b.qs.back());
    gen.set_lookup(b.lookup ? 1 : 0);
    if (!b.profile_out.empty()) gen.set_profile_decode_only(true);   // the forward's counts are this request's decode steps at the end
    std::vector<int32_t> out; Ds41GenStats st; std::string text;
    const std::string e = gen.run(ids, max_new, p, {}, [&](std::string_view piece) { text.append(piece); return on_token ? on_token(piece) : true; }, out, st);
    if (!e.empty()) { r.finish_reason = "error: " + e; return r; }
    if (!b.profile_out.empty()) {
        const auto& pr = b.fwd.profile();
        if (b.profile_acc.size() != pr.size()) b.profile_acc.assign(pr.size(), std::vector<uint64_t>(pr.empty() ? 0 : pr[0].size(), 0));
        for (size_t L = 0; L < pr.size(); ++L) for (size_t x = 0; x < pr[L].size(); ++x) b.profile_acc[L][x] += pr[L][x];
        b.profile_steps += st.n_gen;
        std::vector<std::vector<uint32_t>> orders(b.profile_acc.size()); uint64_t total = 0;
        for (size_t L = 0; L < b.profile_acc.size(); ++L) {
            orders[L].resize(b.profile_acc[L].size()); for (uint32_t x = 0; x < orders[L].size(); ++x) orders[L][x] = x;
            std::stable_sort(orders[L].begin(), orders[L].end(), [&](uint32_t a2, uint32_t b2) { return b.profile_acc[L][a2] > b.profile_acc[L][b2]; });
            for (uint64_t v : b.profile_acc[L]) total += v;
        }
        if (auto pe = ds4_expert_priority_write_layers(b.profile_out, orders, {}, {"V4.1 DECODE-PHASE profile from served requests (IE_DS41_PROFILE_OUT): " +
                std::to_string(b.profile_steps) + " generated tokens", "total selections " + std::to_string(total)}); !pe.empty())
            std::fprintf(stderr, "[deepseek41] profile write: %s\n", pe.c_str());
        {   // the counts, for the next process to resume from -- written then renamed, so a crash leaves the old file
            const std::string cp = b.profile_out + ".counts", tmp = cp + ".tmp";
            std::ofstream f(tmp);
            f << "ie-ds41-profile-counts-v1 " << b.profile_acc.size() << ' ' << (b.profile_acc.empty() ? 0 : b.profile_acc[0].size()) << ' ' << b.profile_steps << '\n';
            for (const auto& row : b.profile_acc) { for (size_t x = 0; x < row.size(); ++x) f << (x ? " " : "") << row[x]; f << '\n'; }
            f.close();
            if (!f || std::rename(tmp.c_str(), cp.c_str()) != 0) std::fprintf(stderr, "[deepseek41] profile counts: could not write %s\n", cp.c_str());
        }
    }
    r.prompt_tokens = st.n_prompt; r.completion_tokens = st.n_gen; r.cached_tokens = st.n_cached; r.prefill_ms = st.prefill_s * 1000.0; r.decode_ms = st.decode_s * 1000.0; r.restore_ms = st.restore_s * 1000.0; r.cache_source = st.cache_source; r.early_decode_ms = st.early_s * 1000.0; r.early_decode_n = st.early_n;
    r.finish_reason = st.stop_reason == "eos" || st.stop_reason == "stop" ? "stop"
                      : st.stop_reason == "callback" ? "abort"
                      : st.stop_reason == "repetition" ? "repetition" : "length";
    if (chat && r.finish_reason == "stop") {
        auto parsed = ds41_parse_completion(text + "<｜end▁of▁sentence｜>", split_thinking);
        r.text = std::move(parsed.content);
        r.reasoning_content = std::move(parsed.reasoning_content);
        r.tool_calls_json = parsed.tool_calls_json.empty() ? "[]" : std::move(parsed.tool_calls_json);
        if (!parsed.error.empty()) r.finish_reason = "error: " + parsed.error;
        else if (r.tool_calls_json != "[]") r.finish_reason = "tool_calls";
        return r;
    }
    if (split_thinking) {
        if (const auto at = text.find("</think>"); at != std::string::npos) { r.reasoning_content = text.substr(0, at); r.text = text.substr(at + 8); }
        else { r.reasoning_content = text; }                    // the budget ended inside the reasoning
    } else r.text = text;
    if (chat && r.finish_reason == "length") r.truncated_tool_call = ds41_cut_tool_name(r.text);
    if (chat) r.text = ds41_visible_content(r.text);
    return r;
}
}  // namespace

namespace {
constexpr std::string_view kDs41ImagePlaceholder = "<｜deepseek_image｜>";

// One image of a request: where its span sits, and its rows once something asks for one.
struct Ds41RequestImage {
    const std::string* bytes = nullptr;          // the image file as the request carried it
    Ds4VisGeom geom; uint32_t pos0 = 0, n = 0; uint64_t hash = 0;
    std::vector<float> rows;
};

// The id of an image position: negative, and a function of the image's bytes and the slot -- two images never share a
// prefix in the prompt cache, the same image at the same place always does.
int32_t ds41_image_id(uint64_t image_hash, uint32_t slot) {
    uint64_t x = image_hash + 0x9E3779B97F4A7C15ull * (uint64_t(slot) + 1);
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull; x ^= x >> 27; x *= 0x94D049BB133111EBull; x ^= x >> 31;
    return -1 - int32_t(x >> 33);
}

// The turn's text with one placeholder per image: where the parser's markers sit, else in front (as the V4 path does).
std::string ds41_place_images(std::string text, size_t n_images) {
    size_t markers = 0;
    for (size_t p = text.find(kChatImageMarker); p != std::string::npos; p = text.find(kChatImageMarker, p + kChatImageMarker.size())) ++markers;
    if (markers == n_images) {
        for (size_t p = text.find(kChatImageMarker); p != std::string::npos; p = text.find(kChatImageMarker, p + kDs41ImagePlaceholder.size()))
            text.replace(p, kChatImageMarker.size(), kDs41ImagePlaceholder);
        return text;
    }
    for (size_t p = text.find(kChatImageMarker); p != std::string::npos; p = text.find(kChatImageMarker)) text.erase(p, kChatImageMarker.size());
    std::string front;
    for (size_t i = 0; i < n_images; ++i) front += kDs41ImagePlaceholder;
    return front + text;
}
}  // namespace

GenerateResult Engine::ds41_chat(std::span<const ChatTurn> turns, const SamplingParams& sp, const TokenCallback& on_token,
                                 bool enable_thinking, std::string_view tools_json, std::string_view reasoning_effort) {
    GenerateResult r;
    std::vector<Ds41ChatMessage> msgs;
    std::vector<Ds41RequestImage> images;
    for (const auto& t : turns) {
        std::string content = t.content_without_tool_calls.value_or(t.content);
        if (!t.images.empty()) {
            if (!ds41_->vis_ready) { r.finish_reason = "error: deepseek41 image input: " + ds41_->vis_error; return r; }
            if (ds41_->drafter.ready()) { r.finish_reason = "error: deepseek41 image input is not supported with speculation (IE_DS41_SPEC=1)"; return r; }
            content = ds41_place_images(std::move(content), t.images.size());
            for (const auto& bytes : t.images) {
                Ds41RequestImage im; im.bytes = &bytes;
                uint32_t w = 0, h = 0;
                if (auto e = ds41_image_size(bytes.data(), bytes.size(), w, h); !e.empty()) { r.finish_reason = "error: " + e; return r; }
                im.geom = ds41_vis_plan(w, h);
                im.n = ds41_vis_tokens(im.geom.n_llm_h, im.geom.n_llm_w);
                im.hash = 0xCBF29CE484222325ull;
                for (char c : bytes) { im.hash ^= uint8_t(c); im.hash *= 0x100000001B3ull; }
                images.push_back(std::move(im));
            }
        }
        msgs.push_back({t.role, std::move(content), t.reasoning_content.value_or(""), false, !t.tool_calls_json.empty(), t.tool_calls_json, t.tool_call_id, ""});
    }
    Ds41PromptOptions po; po.thinking = enable_thinking; po.has_tools = !tools_json.empty(); po.tools_json = std::string(tools_json); std::string err;
    if (!reasoning_effort.empty()) { po.reasoning_effort = ds41_reasoning_effort_of(std::string(reasoning_effort), err); if (!err.empty()) { r.finish_reason = "error: " + err; return r; } }
    const std::string prompt = ds41_encode_messages(msgs, po, err);
    if (!err.empty()) { r.finish_reason = "error: " + err; return r; }
    auto ids = ds41_->tok.encode(prompt, /*allow_special=*/true);
    if (images.empty()) return ds41_run_ids(*ds41_, ids, sp, on_token, enable_thinking, /*chat=*/true);

    // every placeholder becomes its image's span: START, (IMAGE x w, NEW_LINE) x h, END -- all image positions
    const int32_t ph = int32_t(ds41_->model.config().image_token_id);
    std::vector<int32_t> full; full.reserve(ids.size() + images.size() * kDs41VisMaxTok);
    size_t k = 0;
    for (int32_t id : ids) {
        if (id != ph) { full.push_back(id); continue; }
        if (k >= images.size()) { r.finish_reason = "error: the prompt holds more image placeholders than images"; return r; }
        Ds41RequestImage& im = images[k++];
        im.pos0 = uint32_t(full.size());
        for (uint32_t s = 0; s < im.n; ++s) full.push_back(ds41_image_id(im.hash, s));
    }
    if (k != images.size()) { r.finish_reason = "error: the prompt holds fewer image placeholders than images"; return r; }

    Ds41Bundle& b = *ds41_;
    b.fwd.set_vision_provider([&b, &images](uint32_t pos, const float*& row) -> std::string {
        const uint32_t H = b.model.config().dim;
        for (auto& im : images) {
            if (pos < im.pos0 || pos - im.pos0 >= im.n) continue;
            if (im.rows.empty()) {                                   // first row asked of this image: decode, encode, lay out
                const auto t0 = std::chrono::steady_clock::now();
                std::vector<float> px, aligned; uint32_t ph_ = 0, pw_ = 0; Ds4VisGeom g;
                if (auto e = ds41_load_image_mem(im.bytes->data(), im.bytes->size(), px, ph_, pw_, g); !e.empty()) return e;
                if (auto e = b.vis.encode_gpu(b.vis_alloc, px.data(), ph_, pw_, aligned); !e.empty()) return e;
                std::vector<int32_t> types, perm; ds41_vis_types(g.n_llm_h, g.n_llm_w, types, perm);
                if (types.size() != im.n) return "the image plan changed between the size probe and the decode";
                if (auto e = b.vis.assemble_rows(aligned, types, perm, im.rows); !e.empty()) return e;
                std::fprintf(stderr, "[deepseek41] vision: image at %u, %ux%u patches -> %u tokens, encoded in %.0f ms\n", im.pos0, g.n_vit_h, g.n_vit_w, im.n,
                             std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
            row = im.rows.data() + size_t(pos - im.pos0) * H;
            return {};
        }
        return "position " + std::to_string(pos) + " is outside every image of this request";
    });
    GenerateResult res = ds41_run_ids(b, full, sp, on_token, enable_thinking, /*chat=*/true);
    b.fwd.clear_vision();                                             // the provider captures this frame's locals
    return res;
}

GenerateResult Engine::ds41_generate(const std::string& prompt, const SamplingParams& sp, const TokenCallback& on_token) {
    const auto ids = ds41_->tok.encode(prompt, /*allow_special=*/true);
    return ds41_run_ids(*ds41_, ids, sp, on_token, /*split_thinking=*/false);
}

uint32_t Engine::ds41_prompt_cache_slots() const {
    // V4.1 keeps other conversations in host slots under a byte budget (Phase 46): at least the live one plus one
    // saved conversation when the cache is on with a budget; the exact count depends on their lengths.
    const Ds41Bundle& b = *ds41_;
    if (!b.fwd.prefix_cache()) return 0;   // --no-prompt-cache, IE_DS41_PROMPT_CACHE=0 or speculation: nothing is reused
    return b.fwd.prefix_cache_options().host_budget > 0 ? 2 : 1;
}

}  // namespace ie
