// src/engine/ds41_engine.cpp — DeepSeek-V4.1-Flash behind the Engine (Phase 12 step 5,
// docs/deepseek41/31): the same opaque-bundle pattern as V4's Ds4Bundle. A model DIRECTORY
// (safetensors + config.json + tokenizer.json) loads here and never touches the GGUF path;
// chat() renders the V4.1 prompt format and runs Ds41Generator on the resident two-card
// runtime. Native V4.1 tools are encoded and parsed; images are refused. One request at a
// time (EngineOptions::parallel is 1 for this arch; the server's gate serialises).
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
    if (const char* po = std::getenv("IE_DS41_PROFILE_OUT"); po && *po) {
        b->profile_out = po;
        std::fprintf(stderr, "[deepseek41] profiling decode routing into %s (rewritten after every request)\n", po);
    }
    ds41_ = std::move(b);
    return {};
}

namespace {
GenerateResult ds41_run_ids(Ds41Bundle& b, const std::vector<int32_t>& ids, const SamplingParams& sp, const TokenCallback& on_token, bool split_thinking, bool chat = false) {
    GenerateResult r;
    if (chat) r.tool_calls_json = "[]"; // authoritative native parser, including no calls
    Ds41SampleParams p; p.temperature = sp.temperature; p.top_k = sp.top_k; p.top_p = sp.top_p; p.min_p = sp.min_p;
    p.repeat_penalty = sp.repeat_penalty; p.repeat_window = sp.repeat_window; p.seed = sp.seed;
    const uint32_t max_new = sp.max_tokens == 0 || ids.size() + sp.max_tokens > b.fwd.capacity() ? (uint32_t(ids.size()) < b.fwd.capacity() ? b.fwd.capacity() - uint32_t(ids.size()) : 0) : sp.max_tokens;
    Ds41Generator gen(b.fwd, b.tok); if (b.drafter.ready()) gen.set_drafter(&b.drafter, b.qs.back());
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
    }
    r.prompt_tokens = st.n_prompt; r.completion_tokens = st.n_gen; r.cached_tokens = st.n_cached; r.prefill_ms = st.prefill_s * 1000.0; r.decode_ms = st.decode_s * 1000.0;
    r.finish_reason = st.stop_reason == "eos" || st.stop_reason == "stop" ? "stop" : st.stop_reason == "callback" ? "abort" : "length";
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
    if (chat) r.text = ds41_visible_content(r.text);
    return r;
}
}  // namespace

GenerateResult Engine::ds41_chat(std::span<const ChatTurn> turns, const SamplingParams& sp, const TokenCallback& on_token,
                                 bool enable_thinking, std::string_view tools_json, std::string_view reasoning_effort) {
    GenerateResult r;
    std::vector<Ds41ChatMessage> msgs;
    for (const auto& t : turns) {
        if (!t.images.empty()) { r.finish_reason = "error: deepseek41 image input is not supported yet"; return r; }
        msgs.push_back({t.role, t.content_without_tool_calls.value_or(t.content), t.reasoning_content.value_or(""), false, !t.tool_calls_json.empty(), t.tool_calls_json, t.tool_call_id, ""});
    }
    Ds41PromptOptions po; po.thinking = enable_thinking; po.has_tools = !tools_json.empty(); po.tools_json = std::string(tools_json); std::string err;
    if (!reasoning_effort.empty()) { po.reasoning_effort = ds41_reasoning_effort_of(std::string(reasoning_effort), err); if (!err.empty()) { r.finish_reason = "error: " + err; return r; } }
    const std::string prompt = ds41_encode_messages(msgs, po, err);
    if (!err.empty()) { r.finish_reason = "error: " + err; return r; }
    const auto ids = ds41_->tok.encode(prompt, /*allow_special=*/true);
    return ds41_run_ids(*ds41_, ids, sp, on_token, enable_thinking, /*chat=*/true);
}

GenerateResult Engine::ds41_generate(const std::string& prompt, const SamplingParams& sp, const TokenCallback& on_token) {
    const auto ids = ds41_->tok.encode(prompt, /*allow_special=*/true);
    return ds41_run_ids(*ds41_, ids, sp, on_token, /*split_thinking=*/false);
}

}  // namespace ie
