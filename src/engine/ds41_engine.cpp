// src/engine/ds41_engine.cpp — DeepSeek-V4.1-Flash behind the Engine (Phase 12 step 5,
// docs/deepseek41/31): the same opaque-bundle pattern as V4's Ds4Bundle. A model DIRECTORY
// (safetensors + config.json + tokenizer.json) loads here and never touches the GGUF path;
// chat() renders the V4.1 prompt format and runs Ds41Generator on the resident two-card
// runtime. Native V4.1 tools are encoded and parsed; images go through the checkpoint's own
// vision tower (Phase 57, docs/deepseek41/96). --parallel 1: one request at a time (the server's
// admission serialises; ds41_run_ids). --parallel N > 1: N requests at once on N lanes of the
// forward, their steps through the two-card lane pipe (P4 B6b, ds41_run_lanes;
// docs/deepseek41/P4_B6B_SERVE.md).
#include "ie/engine.hpp"

#include "ie/ds41_engine.hpp"
#include "ie/ds41_serve_rules.hpp"
#include "ie/deepseek41_generate.hpp"
#include "ie/deepseek41_prompt.hpp"
#include "ie/expert_stream.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iterator>
#include <memory>
#include <omp.h>
#include <string>
#include <thread>
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
    // P4 B6b (docs/deepseek41/P4_B6B_SERVE.md): --parallel N > 1 = N lanes. Lane 0 keeps --ctx; lanes 1..N-1 hold --slot-ctx
    // positions each (0 = 32,768, capped at --ctx), and their state comes out of the static expert tier (B6a: the forward
    // refuses, with the numbers, when it does not fit). What the lane pipe cannot run is refused here, before the load's work.
    const uint32_t n_lanes = std::max<uint32_t>(1, opts_.parallel);
    const uint32_t lane_ctx = n_lanes > 1 ? std::min<uint32_t>(opts_.max_ctx, opts_.slot_ctx ? opts_.slot_ctx : 32768u) : 0u;
    if (n_lanes > 1) {
        if (spec) return "deepseek41: --parallel " + std::to_string(n_lanes) + " serves without DSpark speculation (IE_DS41_SPEC=1 is --parallel 1 only)";
        if (const char* po = std::getenv("IE_DS41_PROFILE_OUT"); po && *po) return "deepseek41: IE_DS41_PROFILE_OUT profiles --parallel 1 only (a request's reset of the routing counts would wipe the other lanes')";
        if (const char* v = std::getenv("IE_DS41_DUMP_ROUTING"); v && *v) return "deepseek41: IE_DS41_DUMP_ROUTING is --parallel 1 only (the lane pipe refuses it)";
        if (b->qs.size() < 2) return "deepseek41: --parallel > 1 needs the two-card lane pipe (" + std::to_string(b->qs.size()) + " card found)";
    }
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
    if (n_lanes > 1) { ro.lanes = n_lanes; ro.lane_ctx = lane_ctx; }   // P4 B6b (1 lane: the pre-lane forward, byte for byte)
    if (auto e = b->fwd.init_resident(b->qs, b->model, b->tables, ranking, ro); !e.empty()) return "deepseek41 resident: " + e;
    if (n_lanes > 1 && b->fwd.expert_parallel()) return "deepseek41: --parallel > 1 runs on the lane pipe, which refuses expert parallel (IE_DS41_EP)";
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
    opts_.parallel = n_lanes;   // 1 = the serial path, as before P4 B6b
    if (const char* lv = std::getenv("IE_DS41_LOOKUP"); lv && std::string(lv) == "0") b->lookup = false;
    std::fprintf(stderr, "[deepseek41] prompt-lookup speculation %s (a copy of >= 12 context tokens verified up to 8 rows at a time; IE_DS41_LOOKUP=0 turns it off)\n",
                 b->lookup ? "ON" : "off");
    if (n_lanes > 1) {   // P4 B6b: the lanes' serving state (Ds41Serve); the pipe starts with the first request
        Ds41Serve& s = b->serve;
        s.lanes.resize(n_lanes);
        for (uint32_t l = 0; l < n_lanes; ++l) s.lanes[l].cap = b->fwd.lane_capacity(l);
        s.lookup = b->lookup; s.lp = Ds41Generator::lookup_policy();
        if (s.lookup) b->fwd.set_multi_row_decode(true);   // a lane's lookup verify steps (2..8 rows) go through the pipe (B6a)
        const uint32_t cap = std::max<uint32_t>(2u, b->fwd.forward_capacity() & ~1u);
        s.mix_chunk = cap;
        if (const char* v = std::getenv("IE_DS41_MIX_CHUNK"); v && *v) s.mix_chunk = std::min<uint32_t>(cap, std::max<uint32_t>(64u, uint32_t(std::max(0L, std::atol(v)))) & ~1u);
        std::fprintf(stderr, "[deepseek41] lanes: %u (lane 0 %u positions, lanes 1..%u %u each; --slot-ctx); requests decode together through the two-card lane pipe, "
                             "prompt-lookup %s per lane; a prefill chunk is %u rows while another lane is busy (IE_DS41_MIX_CHUNK) and %u alone; a prompt arriving while "
                             "every other lane is idle prefills in its serial turn with the cards pipelined over its chunks (docs/deepseek41/P4_B6B_SERVE.md)\n",
                     n_lanes, s.lanes[0].cap, n_lanes - 1, lane_ctx, s.lookup ? "on" : "off", s.mix_chunk, cap);
    }
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
// The reply's text into the result (the serial path and the lanes alike): a chat that stopped goes through the native parser
// (content, reasoning, tool calls); otherwise the reasoning is split at </think>, and a chat cut by its budget drops a
// half-written tool call from the visible content.
void ds41_finish_text(GenerateResult& r, const std::string& text, bool chat, bool split_thinking) {
    if (chat && r.finish_reason == "stop") {
        auto parsed = ds41_parse_completion(text + "<｜end▁of▁sentence｜>", split_thinking);
        r.text = std::move(parsed.content);
        r.reasoning_content = std::move(parsed.reasoning_content);
        r.tool_calls_json = parsed.tool_calls_json.empty() ? "[]" : std::move(parsed.tool_calls_json);
        if (!parsed.error.empty()) r.finish_reason = "error: " + parsed.error;
        else if (r.tool_calls_json != "[]") r.finish_reason = "tool_calls";
        return;
    }
    if (split_thinking) {
        if (const auto at = text.find("</think>"); at != std::string::npos) { r.reasoning_content = text.substr(0, at); r.text = text.substr(at + 8); }
        else { r.reasoning_content = text; }                    // the budget ended inside the reasoning
    } else r.text = text;
    if (chat && r.finish_reason == "length") r.truncated_tool_call = ds41_cut_tool_name(r.text);
    if (chat) r.text = ds41_visible_content(r.text);
}

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
    ds41_finish_text(r, text, chat, split_thinking);
    return r;
}

// ---- P4 B6b: several requests at once on the lanes (docs/deepseek41/P4_B6B_SERVE.md) ------------------------------------------
// --parallel N > 1. ds41_run_ids above stays the --parallel 1 engine, byte for byte. Here every request owns one lane of the
// forward; the lane pipe (Ds41Forward::pipe_*, B6a) runs its prefill chunks and decode steps -- card 0 one lane's while card 1
// runs another's -- and the done callback (serve_done, on the last card's stage thread) samples the lane's rows on the host,
// judges a prompt-lookup verify step's rows, rolls the lane back, commits the ids to its outbox and submits its next step.
// The request's own thread takes the SERIAL TURN (the pipe paused) for the prefix step, the prompt's end and a prompt with
// images, and otherwise only turns its outbox into text. The rules are Ds41Generator::run's: the prefill plan
// (plan_prefill), the sampler and its state, the lookup policy and verify, the emit rule, the repetition stop.
using SLane = Ds41Serve::Lane;
using SPhase = Ds41Serve::Lane::Phase;
using SClock = std::chrono::steady_clock;

double serve_ms(SClock::time_point t0) { return std::chrono::duration<double, std::milli>(SClock::now() - t0).count(); }
uint32_t serve_busy(const Ds41Serve& s) { uint32_t n = 0; for (const auto& l : s.lanes) n += l.busy; return n; }
uint32_t serve_decoding(const Ds41Serve& s) { uint32_t n = 0; for (const auto& l : s.lanes) n += l.busy && l.phase == SPhase::kDecode; return n; }
bool serve_trace() { static const bool v = [] { const char* e = std::getenv("IE_DS41_STEP_TRACE"); return e && *e == '1'; }(); return v; }

// The lane's request ends (mu held): `finish` unless one is set; `lost` = its state is forgotten at the release
void serve_end(Ds41Serve& s, SLane& l, const std::string& finish, bool lost) {
    if (l.finish.empty()) l.finish = finish;
    l.lost = l.lost || lost; l.phase = SPhase::kDone; l.parked = false; l.next = -1;
    s.cv.notify_all();
}

// A sampled id into the lane (mu held) -- Ds41Emitter::push's rule: eos ends the lane with "stop" and is not committed;
// otherwise the id is committed (the sampler's window, the outbox, the lookup index), a reply that became one short pattern
// ends with "repetition" (counted, never emitted, as the emitter does), and the budget ends it with "length". True while the
// lane goes on.
bool serve_commit(Ds41Bundle& b, SLane& l, int32_t id) {
    if (id == b.tok.eos_token_id()) { l.finish = "stop"; return false; }
    ++l.n_new; ++b.serve.tokens;
    if (l.n_new == 1) l.t_first = SClock::now();
    else if (l.n_new == 100) l.early_ms = serve_ms(l.t_first);
    l.out.push_back(id);
    l.recent.push_back(id);
    if (l.recent.size() > size_t(l.sp.repeat_window) + 4096)    // (the sampler reads the last repeat_window only)
        l.recent.erase(l.recent.begin(), l.recent.end() - std::ptrdiff_t(l.sp.repeat_window));
    if (Ds41Generator::repeating(l.out, l.n_new)) { l.finish = "repetition"; return false; }
    l.outbox.push_back(id);
    if (l.lookup) l.idx.push(id);
    if (l.n_new >= l.max_new) { l.finish = "length"; return false; }
    return true;
}

// The lane's next step into the running pipe (mu held): its plan's next prefill chunk, a one-row prompt step up to the
// planned end, or a decode step -- [next] plus the lookup drafts (run()'s policy: a copy of >= min_match context tokens, at
// most k drafts, no more than the tokens left). A refused submit ends the request (the pipe refuses every step after a stage
// error).
void serve_submit(Ds41Bundle& b, SLane& l, uint32_t li) {
    Ds41Serve& s = b.serve;
    l.parked = false; l.t_sub = SClock::now();
    std::string e;
    if (l.phase == SPhase::kPrefill) {
        const std::vector<int32_t>& P = *l.prompt;
        if (l.chunk_at < l.plan.chunks.size()) {
            const auto [p0, t] = l.plan.chunks[l.chunk_at];
            l.pend = t; ++l.chunks;
            e = b.fwd.pipe_submit(li, P.data() + p0, t, p0, false);
        } else { l.pend = 1; e = b.fwd.pipe_submit(li, P.data() + l.pos, 1, l.pos, false); }
    } else {
        l.rows.assign(1, l.next);
        if (l.lookup) {
            std::vector<int32_t> d = l.idx.draft(s.lp.k, s.lp.min_match);
            if (d.size() > l.max_new - l.n_new) d.resize(l.max_new - l.n_new);
            l.rows.insert(l.rows.end(), d.begin(), d.end());
        }
        l.pend = 0;
        e = b.fwd.pipe_submit(li, l.rows.data(), uint32_t(l.rows.size()), l.pos, l.rows.size() > 1);
    }
    if (!e.empty()) serve_end(s, l, "error: deepseek41 lane " + std::to_string(li) + ": " + e, true);
}

// The pipe's done callback (the last card's stage thread): a prefill piece landed, or a decode step's rows are sampled here --
// rows[0] is the lane's anchor (committed already), rows[1..] its lookup drafts, and row r's sample judges draft r + 1: the
// serial loop's verify (exact speculative sampling for a one-hot draft), the lane rolled back to the rows kept.
void serve_done(Ds41Bundle& b, uint32_t li, const std::vector<float>& logits) {
    Ds41Serve& s = b.serve;
    std::lock_guard<std::mutex> lk(s.mu);
    SLane& l = s.lanes[li];
    if (!l.busy || l.phase == SPhase::kDone) return;
    struct CbTime { SLane& l; SClock::time_point t0; ~CbTime() { l.cb_ms += serve_ms(t0); } } cb_time{l, SClock::now()};
    const double step = serve_ms(l.t_sub);
    const bool prefill = l.phase == SPhase::kPrefill;
    // (diagnostic) IE_DS41_STEP_TRACE=1: every step, submit to callback
    if (serve_trace()) std::fprintf(stderr, "[ds41 step] lane %u %s %u row(s) at %u: %.1f ms, %u decoding\n", li, prefill ? "prefill" : "decode",
                                    prefill ? l.pend : uint32_t(l.rows.size()), l.pos, step, serve_decoding(s));
    if (prefill) {
        if (l.chunk_at < l.plan.chunks.size()) { l.pos = l.plan.chunks[l.chunk_at].first + l.plan.chunks[l.chunk_at].second; ++l.chunk_at; }
        else ++l.pos;
        if (l.pos < l.plan.Tp) {
            if (l.want_stop) serve_end(s, l, "abort", false);         // the client left during the prompt
            else if (s.pause || s.stopping) l.parked = true;
            else serve_submit(b, l, li);
            s.cv.notify_all();                                          // (the request thread's liveness probe)
            return;
        }
        l.first_logits = logits;                                        // the last row so far: the prompt's end is the turn's
        l.phase = SPhase::kPromptReady;
        s.cv.notify_all();
        return;
    }
    const uint32_t T = uint32_t(l.rows.size()), pos = l.pos;
    const size_t V = T ? logits.size() / T : 0;
    if (!T || !V || logits.size() != size_t(T) * V) {
        serve_end(s, l, "error: deepseek41 lane " + std::to_string(li) + ": " + std::to_string(logits.size()) + " logits for a " + std::to_string(T) + "-row step", true);
        return;
    }
    ++s.steps; ++l.steps;
    s.step_ms = s.steps == 1 ? step : 0.9 * s.step_ms + 0.1 * step;
    s.rows_per_step = s.steps == 1 ? T : 0.9 * s.rows_per_step + 0.1 * T;
    auto sample = [&](uint32_t r) {
        l.scratch.assign(logits.begin() + std::ptrdiff_t(size_t(r) * V), logits.begin() + std::ptrdiff_t(size_t(r + 1) * V));
        return Ds41Generator::sample_row(l.scratch.data(), V, l.recent, l.sp, l.rng);
    };
    if (T == 1) {
        l.pos = pos + 1; ++l.plain;
        const int32_t a = sample(0);
        if (!serve_commit(b, l, a)) { serve_end(s, l, l.finish, false); return; }
        l.next = a;
    } else {
        uint32_t L = 0; int32_t nxt = -1; bool ended = false, drop_last = false;
        for (uint32_t r = 0; r < T; ++r) {
            const int32_t a = sample(r);
            if (r + 1 == T || a != l.rows[r + 1]) { nxt = a; break; }
            ++L;
            if (!serve_commit(b, l, a)) { ended = true; drop_last = l.finish != "length"; break; }   // an eos (or a stopped) row is not kept
        }
        const uint32_t keep = 1 + L - (drop_last ? 1u : 0u);         // the rows of the anchor and the accepted drafts
        if (auto e = b.fwd.pipe_rollback(li, pos + keep); !e.empty()) { serve_end(s, l, "error: deepseek41 lane " + std::to_string(li) + " rollback: " + e, true); return; }
        l.pos = pos + keep; ++l.passes; l.pass_rows += T; l.accepted += L;
        if (ended) { serve_end(s, l, l.finish, false); return; }
        if (!serve_commit(b, l, nxt)) { serve_end(s, l, l.finish, false); return; }
        l.next = nxt;
    }
    if (l.want_stop) { serve_end(s, l, "abort", false); return; }
    if (s.pause || s.stopping) { l.parked = true; s.cv.notify_all(); return; }
    serve_submit(b, l, li);
    s.cv.notify_all();
}

// After a stage error the forward refuses every step until pipe_stop: every running request fails, and its lane is forgotten.
// A lane whose step failed is HALF-STEPPED (the B6a gate: an error on card 1 leaves card 0's latents and the look-back advanced
// with n_pos not committed), and the pipe does not reset it, as the serial forward() does. Every lane that can have had a step
// in flight is a running one (busy, not done: a done lane never resubmits), so all of them are marked lost here, and each is
// reset (serve_clear_lane: reset_state on that lane, in a turn) by its own request before the lane is released for reuse.
void serve_pipe_failed(Ds41Serve& s, const std::string& e) {
    std::fprintf(stderr, "[ds41 lanes] the lane pipe failed: %s -- every running request fails and its lane is forgotten\n", e.c_str());
    for (auto& l : s.lanes) if (l.busy && l.phase != SPhase::kDone) serve_end(s, l, "error: deepseek41 pipe: " + e, true);
}

// The serial turn (lk holds mu; returns with it held): after the current holder, the pipe PAUSED -- the callbacks park their
// lanes at their next completion, pipe_pause waits for the steps in flight, and the stage threads stay (their OpenMP teams
// and oneDNN state with them: a stage thread that exits leaves its OpenMP team behind, the adca570 leak). A stage error stops
// the pipe (the next release starts a new one) and fails every running request.
void serve_take_turn(Ds41Bundle& b, std::unique_lock<std::mutex>& lk) {
    Ds41Serve& s = b.serve;
    ++s.turn_waiters;
    s.cv.wait(lk, [&] { return !s.turn_busy; });
    --s.turn_waiters;
    s.turn_busy = true;
    if (s.piping) {
        s.pause = true;
        lk.unlock();
        std::string e = b.fwd.pipe_pause();
        if (!e.empty()) (void)b.fwd.pipe_stop();   // (returns the same error and clears it)
        lk.lock();
        s.piping = false; s.pause = false;
        if (!e.empty()) serve_pipe_failed(s, e);
    }
    ++s.turns; s.turn_t0 = SClock::now();
}

// Releases the turn (lk holds mu). Another request waiting for it takes it with the pipe still paused (no drain between the
// two turns); otherwise the parked lanes' steps go into the resumed pipe -- a new one when none is paused (the first request,
// or after a stage error).
void serve_release_turn(Ds41Bundle& b, std::unique_lock<std::mutex>& lk) {
    (void)lk;
    Ds41Serve& s = b.serve;
    s.paused_ms += serve_ms(s.turn_t0);
    if (s.turn_waiters > 0 && !s.stopping) { ++s.handovers; s.turn_busy = false; s.cv.notify_all(); return; }
    std::vector<uint32_t> due;
    for (uint32_t li = 0; li < s.lanes.size(); ++li) {
        SLane& l = s.lanes[li];
        if (!l.busy || !l.parked) continue;
        if (l.want_stop) { serve_end(s, l, "abort", false); continue; }
        due.push_back(li);
    }
    if (!due.empty() && !s.stopping) {
        const std::string e = b.fwd.pipe_paused() ? b.fwd.pipe_resume()
            : b.fwd.pipe_start([&b](uint32_t li, const std::vector<float>& lg) { serve_done(b, li, lg); });
        if (!e.empty()) for (uint32_t li : due) serve_end(s, s.lanes[li], "error: deepseek41 pipe: " + e, true);
        else { s.piping = true; for (uint32_t li : due) serve_submit(b, s.lanes[li], li); }
    }
    s.turn_busy = false;
    s.cv.notify_all();
}

// A stage error whose step ran no callback (a lone failing lane) is seen from the request threads' waits
void serve_poll_pipe(Ds41Bundle& b, std::unique_lock<std::mutex>& lk) {
    if (!b.serve.piping || b.fwd.pipe_error().empty()) return;
    serve_take_turn(b, lk);
    serve_release_turn(b, lk);
}

// The lane for a prompt (the turn held, mu held), by ds41_choose_lane (include/ie/ds41_serve_rules.hpp) among the idle lanes it
// fits (ids < capacity): first the lanes that leave its reply room -- min(max_tokens, a quarter of the lane) past the prompt;
// within them the one whose OWN state serves the most of the prompt (its live position or a checkpoint:
// Ds41Forward::prefix_servable) when that is at least the cache's min_slot_tokens (1,024: below it the prefill saved is not worth
// cutting or evicting a conversation -- an EMPTY lane comes first) or reaches the lane's last prompt end (a follow-up of its
// conversation); then the smallest capacity (short prompts leave lane 0 to the long ones); then the least recently released.
// Host slots and disk entries serve any lane. -1 = none fits now.
int serve_choose(Ds41Bundle& b, const std::vector<int32_t>& ids, uint32_t max_tokens) {
    Ds41Serve& s = b.serve;
    const uint32_t worth = b.fwd.prefix_cache() ? std::max<uint32_t>(1u, b.fwd.prefix_cache_options().min_slot_tokens) : UINT32_MAX;
    std::vector<Ds41LaneView> v(s.lanes.size());
    for (uint32_t li = 0; li < s.lanes.size(); ++li) {
        const SLane& l = s.lanes[li];
        v[li].idle = !l.busy; v[li].cap = l.cap; v[li].tick = l.tick; v[li].last_tp = l.last_tp;
        if (l.busy) continue;
        v[li].occupied = b.fwd.lane_pos(li) > 0;
        if (l.cap > ids.size()) v[li].match = b.fwd.prefix_servable(li, ids);
    }
    return ds41_choose_lane(v, uint32_t(ids.size()), max_tokens, worth);
}

// The serial turn's device work (the turn held; lk holds mu, released while the work runs): `fn` runs on the lanes' serial
// worker -- one persistent thread, so the OpenMP team the engram gather gives a thread that runs a forward is made once, not
// once per HTTP request thread (adca570). Meanwhile this thread probes the client every second (`alive`), and `gone` tells
// the work the client left. A throw inside `fn` comes back as the returned error.
std::string serve_serial(Ds41Bundle& b, std::unique_lock<std::mutex>& lk, const std::function<void()>& fn,
                         const std::function<bool()>& alive = {}, std::atomic<bool>* gone = nullptr) {
    Ds41Serve& s = b.serve;
    lk.unlock();
    std::string err;
    {
        std::unique_lock<std::mutex> wl(s.wmu);
        if (!s.worker.joinable())
            s.worker = std::thread([&s] {
                kmp_set_blocktime(0);   // its team (the engram gather) must not spin beside the CPU expert legs (docs/deepseek41/35)
                std::unique_lock<std::mutex> g(s.wmu);
                for (;;) {
                    s.wcv.wait(g, [&] { return s.wstop || s.wjob; });
                    if (!s.wjob) return;                                   // stopped (teardown: no request is left)
                    const std::function<void()>* job = s.wjob;
                    g.unlock();
                    std::string e;
                    try { (*job)(); }
                    catch (const std::exception& x) { e = std::string("threw: ") + x.what(); }
                    catch (...) { e = "threw a non-std exception"; }
                    g.lock();
                    s.werr = std::move(e); s.wjob = nullptr; s.wdone = true;
                    s.wcv.notify_all();
                }
            });
        s.wjob = &fn; s.wdone = false; s.werr.clear();
        s.wcv.notify_all();
        while (!s.wdone) {
            if (s.wcv.wait_for(wl, std::chrono::seconds(1)) != std::cv_status::timeout || !alive || !gone || gone->load()) continue;
            wl.unlock();
            const bool ok = alive();
            wl.lock();
            if (!ok) gone->store(true);
        }
        err = std::move(s.werr); s.werr.clear();
    }
    lk.lock();
    return err;
}

// A lost lane (the turn held; lk holds mu): its state is forgotten -- positions, look-back, live checkpoints, vision spans
void serve_clear_lane(Ds41Bundle& b, std::unique_lock<std::mutex>& lk, SLane& l, uint32_t li) {
    (void)serve_serial(b, lk, [&] { if (b.fwd.select_lane(li).empty()) { b.fwd.clear_vision(); b.fwd.reset_state(); } });
    l.lost = false; l.last_tp = 0;
}

// The prompt's prefill in the SERIAL TURN (on the serial worker; the lane selected): the plan's chunks from l.chunk_at -- the cards
// pipelined over them as at --parallel 1 when there are several (forward_pipelined), `stop` asked before each chunk after the
// first -- then, once every chunk ran, the one-row steps up to the planned end. Leaves l.chunk_at and l.pos where it got to
// (a stop hands the rest to the lane pipe).
std::string serve_serial_prefill(Ds41Bundle& b, SLane& l, const std::function<bool()>& stop, std::vector<float>& lg) {
    static const bool pipe_prefill = [] { const char* v = std::getenv("IE_DS41_PIPE_PREFILL"); return !(v && *v && std::string(v) == "0"); }();   // run()'s kill switch
    const std::vector<int32_t>& P = *l.prompt;
    const auto& C = l.plan.chunks;
    if (l.chunk_at < C.size()) {
        const std::vector<std::pair<uint32_t, uint32_t>> rest(C.begin() + std::ptrdiff_t(l.chunk_at), C.end());
        if (rest.size() > 1 && pipe_prefill && b.fwd.pipelined_admissible()) {
            size_t done = 0;
            if (auto e = b.fwd.forward_pipelined(P.data(), rest, lg, stop, &done); !e.empty()) return "prefill from " + std::to_string(l.reused) + " (pipelined): " + e;
            l.chunk_at += uint32_t(done); l.chunks += uint32_t(done);
        } else for (size_t k = 0; k < rest.size(); ++k) {
            if (k > 0 && stop && stop()) break;
            const auto [p0, t] = rest[k];
            if (auto e = b.fwd.forward(P.data() + p0, t, p0, lg); !e.empty()) return "prefill from " + std::to_string(l.reused) + ", chunk at " + std::to_string(p0) + ": " + e;
            ++l.chunk_at; ++l.chunks;
        }
        if (l.chunk_at) l.pos = C[l.chunk_at - 1].first + C[l.chunk_at - 1].second;
        if (l.chunk_at < C.size()) return {};
    }
    for (; l.pos < l.plan.Tp; ++l.pos)
        if (auto e = b.fwd.forward(P.data() + l.pos, 1, l.pos, lg); !e.empty()) return "prefill from " + std::to_string(l.reused) + ", token at " + std::to_string(l.pos) + ": " + e;
    return {};
}

// The prompt's end in the SERIAL TURN (on the serial worker; the lane selected) -- run()'s steps after the planned rows: the checkpoint
// before a trailing think tag, the rows [Tp, T) one at a time, the prompt-end checkpoint and the disk entry of the system
// prefix. `lg` holds the last row's logits so far and ends with the prompt's.
std::string serve_prompt_end(Ds41Bundle& b, SLane& l, std::vector<float>& lg) {
    const std::vector<int32_t>& P = *l.prompt;
    const uint32_t T = uint32_t(P.size());
    for (; l.pos < T; ++l.pos) {
        if (l.pos == l.plan.Tp && l.plan.Tp < T && l.pos > l.reused)   // everything before the trailing think tag is in: the point the next turn resumes from
            if (auto e = b.fwd.prefix_checkpoint(); !e.empty()) return "prefix checkpoint before the think tag: " + e;
        if (auto e = b.fwd.forward(P.data() + l.pos, 1, l.pos, lg); !e.empty()) return "prefill from " + std::to_string(l.reused) + ", token at " + std::to_string(l.pos) + ": " + e;
    }
    if (b.fwd.prefix_cache()) {
        if (auto e = b.fwd.prefix_checkpoint(); !e.empty()) return "prefix checkpoint: " + e;
        if (l.plan.persist_at) if (auto e = b.fwd.prefix_persist(l.plan.persist_at); !e.empty()) std::fprintf(stderr, "[ds41 lanes] prefix persist: %s\n", e.c_str());
    }
    return {};
}

GenerateResult ds41_run_lanes(Ds41Bundle& b, const std::vector<int32_t>& ids, const SamplingParams& sp, const TokenCallback& on_token,
                              bool split_thinking, bool chat, Ds41Forward::VisionProvider vis = nullptr) {
    GenerateResult r;
    if (chat) r.tool_calls_json = "[]"; // authoritative native parser, including no calls
    Ds41Serve& s = b.serve;   // (this thread runs no forward: the serial worker and the pipe's stage threads do)
    const uint32_t T = uint32_t(ids.size());
    uint32_t cap_max = 0;
    for (const auto& l : s.lanes) cap_max = std::max(cap_max, l.cap);
    if (T < 2) { r.finish_reason = "error: the prompt needs at least two tokens (the prefill takes an even count)"; return r; }
    if (T >= cap_max) { r.finish_reason = "error: prompt " + std::to_string(T) + " + 0 new tokens exceed the runtime's position capacity " + std::to_string(cap_max); return r; }
    Ds41SampleParams p; p.temperature = sp.temperature; p.top_k = sp.top_k; p.top_p = sp.top_p; p.min_p = sp.min_p;
    p.repeat_penalty = sp.repeat_penalty; p.repeat_window = sp.repeat_window; p.seed = sp.seed;
    Ds41NgramIndex idx;
    if (s.lookup) idx.reset(ids);                                      // (this thread's own work, before any turn)
    const auto t_req = SClock::now();
    SClock::time_point t_dec{};
    // P4 B19: the lane's tier counters when its decode starts (taken in the turn: the pipe is paused, the lane not in flight), so the
    // request line can give this reply's own expert traffic per card
    std::vector<Ds41Forward::LaneTierStats> tier0;
    auto tier_mark = [&](uint32_t ln) { tier0.clear(); for (uint32_t c = 0; c < b.fwd.n_cards(); ++c) tier0.push_back(b.fwd.lane_tier_stats(ln, c)); };
    auto alive = [&] { return !on_token || on_token(std::string_view{}); };   // the server's liveness probe (an empty fragment)
    auto fits = [&] { for (const auto& l : s.lanes) if (!l.busy && l.cap > T) return true; return false; };

    std::unique_lock<std::mutex> lk(s.mu);
    // 1. a lane, chosen under the serial turn; a prompt only lane 0 holds waits for lane 0
    int li = -1;
    for (;;) {
        while (!s.stopping && !fits()) {
            s.cv.wait_for(lk, std::chrono::seconds(1));
            if (s.stopping || fits()) break;
            lk.unlock(); const bool ok = alive(); lk.lock();
            if (!ok) { r.finish_reason = "abort"; return r; }
        }
        if (s.stopping) { r.finish_reason = "abort"; return r; }
        serve_take_turn(b, lk);
        li = serve_choose(b, ids, sp.max_tokens);
        if (li >= 0) break;
        serve_release_turn(b, lk);
    }
    const uint32_t lane = uint32_t(li);
    SLane& l = s.lanes[lane];
    l.busy = true; l.phase = SPhase::kPrefill; l.prompt = &ids; l.finish.clear(); l.want_stop = l.lost = l.parked = false; l.last_tp = 0;
    l.outbox.clear(); l.out.clear(); l.first_logits.clear(); l.rows.clear(); l.pend = 0; l.next = -1;
    l.steps = l.chunks = l.passes = l.pass_rows = l.accepted = l.plain = 0; l.cb_ms = l.early_ms = 0;
    l.sp = p; l.rng = p.seed ? p.seed : 0x2545F4914F6CDD1Dull;
    l.recent.assign(ids.end() - std::ptrdiff_t(std::min<size_t>(ids.size(), p.repeat_window)), ids.end());
    l.lookup = s.lookup; l.idx = std::move(idx);
    l.max_new = ds41_reply_budget(T, sp.max_tokens, l.cap); l.n_new = 0;   // (T < l.cap: the lane fits the prompt)
    uint32_t others = 0;                                               // lanes of other requests still running
    for (uint32_t k = 0; k < s.lanes.size(); ++k) others += k != lane && s.lanes[k].busy && s.lanes[k].phase != SPhase::kDone;
    const bool alone = others == 0 && s.turn_waiters == 0;
    const uint32_t chunk_cap = std::max<uint32_t>(2u, b.fwd.forward_capacity() & ~1u);
    const uint32_t pcap = alone ? chunk_cap : s.mix_chunk;
    auto release_lane = [&] {   // (mu held) the lane goes idle with its conversation; the LRU order follows the release
        l.busy = false; l.phase = SPhase::kIdle; l.tick = ++s.tick; l.prompt = nullptr;
        l.outbox.clear(); l.first_logits.clear(); l.first_logits.shrink_to_fit(); l.idx = Ds41NgramIndex{};
        s.cv.notify_all();
    };

    // 2. the prefix step (the turn held; the serial worker does the device work): what serves the prompt -- the lane's live
    // state or one of its checkpoints, a host slot swapped in (the lane's conversation kept in one first), a disk entry -- and
    // the plan from there. Then the prefill runs HERE when every other lane is idle (the cards pipelined over its chunks, as at
    // --parallel 1, until another request waits for the turn or the client leaves), when the prompt has image positions left
    // (serial-only), or when nothing precedes its end; otherwise through the lane pipe, beside the decoding lanes.
    std::string e, pe;
    uint32_t reused = 0;
    std::vector<float> lg;
    bool ready = false;                                                // the prompt is in: its first token is sampled below
    std::atomic<bool> gone{false};                                     // the client left during it
    const std::function<bool()> stop = [&] {                           // (the worker asks between chunks)
        if (gone.load()) return true;
        std::lock_guard<std::mutex> g(s.mu);
        return s.turn_waiters > 0 || s.stopping;
    };
    const std::string we = serve_serial(b, lk, [&] {
        e = b.fwd.select_lane(lane);
        if (e.empty() && vis) b.fwd.set_vision_provider(vis);
        const auto t_pc = SClock::now();
        if (e.empty() && b.fwd.prefix_cache()) {
            std::string source;
            pe = b.fwd.prefix_prepare(ids, reused, &source);
            r.restore_ms = serve_ms(t_pc); r.cache_source = source;
        }
        if (e.empty() && pe.empty()) {
            // nothing served (or the cache off): the lane forgets its old conversation now (prefix_prepare kept it in a host slot
            // when worth it). The serial forward() resets at a pos0 = 0 step by itself, but the lane pipe admits a step only at
            // the lane's n_pos, a pos0 = 0 prefill included (pipe_submit)
            if (reused == 0) b.fwd.reset_state();
            l.reused = reused; l.pos = reused; l.chunk_at = 0;
            l.plan = Ds41Generator::plan_prefill(ids, reused, pcap, b.fwd.prefix_cache(), b.tok.find_token("<｜User｜>"), b.tok.find_token("<think>"), b.tok.find_token("</think>"));
            if (pcap != chunk_cap) l.plan.persist_at = 0;   // other chunks compute other bits: a disk entry is run()'s chunking only (docs/deepseek41/103)
            bool img_left = false;
            for (uint32_t q = reused; q < T && !img_left; ++q) img_left = ids[q] < 0;
            const bool pipe_work = !l.plan.chunks.empty() || l.plan.tail < l.plan.Tp;
            if (img_left || !pipe_work || alone) {
                e = serve_serial_prefill(b, l, img_left ? std::function<bool()>() : stop, lg);
                if (e.empty() && !gone.load() && l.pos >= l.plan.Tp) { e = serve_prompt_end(b, l, lg); ready = e.empty(); }
            }
        }
        if (vis && b.fwd.select_lane(lane).empty()) b.fwd.clear_vision();   // image rows are serial-only: every image position is in by now
    }, alive, &gone);
    r.cached_tokens = reused;
    if (!we.empty() && e.empty()) e = we;
    if (!pe.empty()) serve_end(s, l, "error: prefix cache: " + pe, true);
    else if (!e.empty()) serve_end(s, l, "error: " + e, true);
    else if (gone) serve_end(s, l, "abort", false);
    else if (ready) {
        l.last_tp = l.plan.Tp;                                         // the lane holds this prompt now
        r.prefill_ms = serve_ms(t_req);
        const int32_t id = Ds41Generator::sample(lg, l.recent, l.sp, l.rng);
        t_dec = SClock::now(); tier_mark(lane);
        if (!serve_commit(b, l, id)) serve_end(s, l, l.finish, false);
        else { l.next = id; l.phase = SPhase::kDecode; l.parked = true; }   // the release submits its first decode step
    } else l.parked = true;                                            // the next prefill piece is due: the release submits it
    const uint32_t serial_chunks = l.chunks;
    serve_release_turn(b, lk);
    // 3. the rest of the prompt through the lane pipe; after each piece the server's liveness probe (a client that left ends
    // the request at the lane's next completion), and a stage error seen from here when no callback ran
    uint32_t seen = l.pos;
    while (l.phase == SPhase::kPrefill) {
        s.cv.wait_for(lk, std::chrono::seconds(1));
        if (l.phase != SPhase::kPrefill) break;
        if (l.pos != seen && on_token && !l.want_stop) {
            seen = l.pos;
            lk.unlock(); const bool ok = alive(); lk.lock();
            if (!ok) l.want_stop = true;
        }
        serve_poll_pipe(b, lk);
    }
    // 4. the prompt's end (the turn held): the think tag, the checkpoints, the disk entry, the first token
    if (l.phase == SPhase::kPromptReady) {
        serve_take_turn(b, lk);
        if (l.phase == SPhase::kPromptReady) {
            std::vector<float> lg2 = std::move(l.first_logits); l.first_logits.clear();
            std::string e2;
            const std::string we2 = serve_serial(b, lk, [&] {
                e2 = b.fwd.select_lane(lane);
                if (e2.empty()) e2 = serve_prompt_end(b, l, lg2);
            });
            if (!we2.empty() && e2.empty()) e2 = we2;
            if (!e2.empty()) serve_end(s, l, "error: " + e2, true);
            else if (l.want_stop) serve_end(s, l, "abort", false);
            else {
                l.last_tp = l.plan.Tp;
                r.prefill_ms = serve_ms(t_req);
                const int32_t id = Ds41Generator::sample(lg2, l.recent, l.sp, l.rng);
                t_dec = SClock::now(); tier_mark(lane);
                if (!serve_commit(b, l, id)) serve_end(s, l, l.finish, false);
                else { l.next = id; l.phase = SPhase::kDecode; l.parked = true; }
            }
        }
        serve_release_turn(b, lk);
    }
    // 5. the outcome: ids as they land in the outbox, decoded to text and streamed with Ds41Emitter's hold-back of incomplete
    // UTF-8; the server's callback declining cancels the lane at its next completion (no text after the declined piece)
    std::string text, pending;
    bool aborted = false;
    std::vector<int32_t> got;
    for (;;) {
        s.cv.wait_for(lk, std::chrono::seconds(1), [&] { return !l.outbox.empty() || l.phase == SPhase::kDone; });
        got.swap(l.outbox);
        const bool done = l.phase == SPhase::kDone;
        if (!got.empty()) {
            lk.unlock();
            for (const int32_t id : got) {
                if (aborted) break;
                pending += b.tok.decode(std::span<const int32_t>(&id, 1), /*skip_special=*/false);
                const size_t n = Ds41Generator::utf8_complete_prefix(pending);
                if (!n) continue;
                const std::string piece = pending.substr(0, n); pending.erase(0, n);
                text += piece;
                if (on_token && !on_token(piece)) aborted = true;
            }
            got.clear();
            lk.lock();
            if (aborted) l.want_stop = true;
        }
        if (done) break;
        serve_poll_pipe(b, lk);
    }
    // 6. the finish (run()'s stop reasons: eos -> stop, the callback -> abort, repetition, the budget -> length)
    r.prompt_tokens = T; r.completion_tokens = l.n_new;
    const std::string& f = l.finish;
    r.finish_reason = aborted ? "abort" : f == "stop" || f == "abort" || f == "repetition" || f.rfind("error:", 0) == 0 ? f : "length";
    if (t_dec.time_since_epoch().count()) r.decode_ms = serve_ms(t_dec);
    if (l.n_new >= 100 && l.early_ms > 0) { r.early_decode_ms = l.early_ms; r.early_decode_n = 100; }
    std::fprintf(stderr, "[ds41 lanes] lane %u: prompt %u (%u cached%s%s), prefill %.0f ms: %u chunk(s) in the serial turn, %u through the lane pipe; "
                         "%u tokens (reply cap %u) in %u steps (%u lookup passes: %u rows, %u drafts accepted; %u one-row steps); %s; %u lane(s) busy; callbacks %.0f ms; "
                         "%llu turns so far (%llu handed over), the pipe paused %.0f ms for them\n", lane, T, r.cached_tokens, r.cache_source.empty() ? "" : " from ",
                 r.cache_source.c_str(), r.prefill_ms, serial_chunks, l.chunks - serial_chunks, l.n_new, l.max_new, l.steps, l.passes, l.pass_rows, l.accepted, l.plain,
                 r.finish_reason.c_str(), serve_busy(s), l.cb_ms, (unsigned long long)s.turns, (unsigned long long)s.handovers, s.paused_ms);
    // P4 B19: this reply's decode steps per card -- what the tier served them from, the CPU leg's wall and compute, and the bytes
    // moved to VRAM (PCIe for the pinned misses, the mmap tier's fills). Read under mu after the lane's last callback (a lost lane
    // -- a stage error -- may still have had a step on a card: no line).
    for (uint32_t c = 0; c < (l.lost ? 0u : uint32_t(tier0.size())); ++c) {
        const auto& a = tier0[c]; const auto& z = b.fwd.lane_tier_stats(lane, c);
        const uint64_t n = z.steps - a.steps;
        if (!n) continue;
        const double pin = double(z.experts_pinned - a.experts_pinned), hit = double(z.stream_hits - a.stream_hits), cpu = double(z.experts_cpu - a.experts_cpu);
        std::fprintf(stderr, "[ds41 lanes] lane %u tier card %u: %llu decode steps (%.2f rows/step); per step %.1f static, %.1f pinned (%.1f%% stream-slot hits), %.1f on the CPU "
                             "(leg %.2f ms, compute %.2f ms = %.3f ms/expert), %.1f mmap; PCIe %.1f MiB, mmap %.1f MiB; moe %.1f ms/step\n",
                     lane, c, (unsigned long long)n, double(z.rows - a.rows) / n, double(z.experts_static - a.experts_static) / n, pin / n,
                     pin > 0 ? 100.0 * hit / pin : 0.0, cpu / n, (z.cpu_ms - a.cpu_ms) / n, (z.cpu_work_ms - a.cpu_work_ms) / n,
                     cpu > 0 ? (z.cpu_work_ms - a.cpu_work_ms) / cpu : 0.0, double(z.experts_mmap - a.experts_mmap) / n,
                     double(z.bytes_pinned - a.bytes_pinned) / n / 1048576.0, double(z.bytes_mmap - a.bytes_mmap) / n / 1048576.0, (z.moe_ms - a.moe_ms) / n);
    }
    if (l.lost) { serve_take_turn(b, lk); serve_clear_lane(b, lk, l, lane); serve_release_turn(b, lk); }
    release_lane();
    lk.unlock();
    if (r.finish_reason.rfind("error:", 0) == 0) return r;             // (as the serial path: an error carries no text)
    ds41_finish_text(r, text, chat, split_thinking);
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
    const bool lanes = !ds41_->serve.lanes.empty();   // P4 B6b: --parallel > 1
    if (images.empty()) return lanes ? ds41_run_lanes(*ds41_, ids, sp, on_token, enable_thinking, /*chat=*/true)
                                     : ds41_run_ids(*ds41_, ids, sp, on_token, enable_thinking, /*chat=*/true);

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
    Ds41Forward::VisionProvider provider = [&b, &images](uint32_t pos, const float*& row) -> std::string {
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
    };
    // P4 B6b: on the lanes the provider is set on the request's lane in its serial turn, and cleared there once every image
    // position is in (the pipe runs text positions only)
    if (lanes) return ds41_run_lanes(b, full, sp, on_token, enable_thinking, /*chat=*/true, std::move(provider));
    b.fwd.set_vision_provider(std::move(provider));
    GenerateResult res = ds41_run_ids(b, full, sp, on_token, enable_thinking, /*chat=*/true);
    b.fwd.clear_vision();                                             // the provider captures this frame's locals
    return res;
}

GenerateResult Engine::ds41_generate(const std::string& prompt, const SamplingParams& sp, const TokenCallback& on_token) {
    const auto ids = ds41_->tok.encode(prompt, /*allow_special=*/true);
    if (!ds41_->serve.lanes.empty()) return ds41_run_lanes(*ds41_, ids, sp, on_token, /*split_thinking=*/false, /*chat=*/false);   // P4 B6b
    return ds41_run_ids(*ds41_, ids, sp, on_token, /*split_thinking=*/false);
}

uint32_t Engine::ds41_prompt_cache_slots() const {
    // V4.1 keeps other conversations in host slots under a byte budget (Phase 46): at least the live one plus one
    // saved conversation when the cache is on with a budget; the exact count depends on their lengths.
    const Ds41Bundle& b = *ds41_;
    if (!b.fwd.prefix_cache()) return 0;   // --no-prompt-cache, IE_DS41_PROMPT_CACHE=0 or speculation: nothing is reused
    // P4 B6b: at --parallel N > 1 every lane keeps its own conversation live, beside the host slots
    const uint32_t live = b.serve.lanes.empty() ? 1u : uint32_t(b.serve.lanes.size());
    return live + (b.fwd.prefix_cache_options().host_budget > 0 ? 1u : 0u);
}

std::string Engine::ds41_serving_status_json() const {
    // P4 B6b (docs/deepseek41/P4_B6B_SERVE.md): DeepSeek-V4.1 at --parallel > 1 -- the lanes owned by a request, those decoding,
    // the ids committed (every lane, since load), the decode steps' submit-to-callback time (an exponential average) and rows,
    // the serial turns, the time the pipe stayed paused for them, and the turns handed straight to a waiting request
    if (!ds41_ || ds41_->serve.lanes.empty()) return {};
    Ds41Serve& s = ds41_->serve;
    std::lock_guard<std::mutex> lk(s.mu);
    char buf[320];
    std::snprintf(buf, sizeof buf, "{\"lanes_active\":%u,\"decoding\":%u,\"tokens\":%llu,\"step_ms\":%.1f,\"rows_per_step\":%.2f,\"turns\":%llu,\"paused_ms\":%.0f,"
                  "\"handovers\":%llu}", serve_busy(s), serve_decoding(s), (unsigned long long)s.tokens, s.step_ms, s.rows_per_step,
                  (unsigned long long)s.turns, s.paused_ms, (unsigned long long)s.handovers);
    return buf;
}

}  // namespace ie
