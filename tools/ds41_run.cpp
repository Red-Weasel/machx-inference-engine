// tools/ds41_run.cpp — ie-ds41-run: DeepSeek-V4.1-Flash on the resident two-card runtime, from a
// prompt to streamed text (Phase 12 step 4, docs/deepseek41/31).
//   ie-ds41-run <model> [--prompt TEXT | --chat] [--system TEXT] [--n N] [--temp T] [--top-k K]
//               [--top-p P] [--min-p M] [--repeat R] [--seed S] [--effort E|low|high|max]
//               [--chat-mode] [--raw] [--max-tokens CAP] [--tables DIR] [--ranking FILE] [--stop STR]...
//   --prompt: one user turn (a system turn with --system), rendered in the V4.1 format; --raw
//   feeds the text as is. --chat: a loop on stdin, the history re-rendered every turn (no cache
//   reuse across turns yet). The engram tables and the held-out ranking default to files beside
//   the model (engram_tables.json, engram_token_map.i32, ie_ranking_heldout.txt). Ctrl-C ends the
//   current answer and unloads. Tool calls, images and DSpark are not supported: refused, not faked.
#include "ie/deepseek41_dspark.hpp"
#include "ie/deepseek41_generate.hpp"
#include "ie/deepseek41_prompt.hpp"
#include "ie/expert_stream.hpp"

#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {
volatile std::sig_atomic_t g_stop = 0;
void on_sigint(int) { g_stop = 1; }
bool exists(const std::string& p) { return std::ifstream(p).good(); }
void usage() {
    std::fprintf(stderr, "usage: ie-ds41-run <model> (--prompt TEXT | --prompt-file FILE | --chat) [--system TEXT] [--n N] [--temp T] [--top-k K] [--top-p P] [--min-p M]\n"
                         "                   [--repeat R] [--seed S] [--effort E] [--chat-mode] [--raw] [--max-tokens CAP] [--tables DIR] [--ranking FILE] [--stop STR]...\n"); }
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(); return 2; }
    const std::string model = argv[1];
    std::string prompt, system, tables = model, ranking = model + "/ie_ranking_heldout.txt", effort = "high", prof_out, counts_out;
    std::vector<std::string> stops;
    bool chat = false, raw = false, thinking = true; uint32_t n = 512, cap = 4096;
    ie::Ds41SampleParams sp;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i]; auto val = [&]() -> std::string { if (i + 1 >= argc) { usage(); std::exit(2); } return argv[++i]; };
        if (a == "--prompt") prompt = val();
        // Phase 24: a long-context prompt does not fit on a command line -- 250k tokens is about a megabyte of
        // argv. --prompt-file reads it from disk instead; the generator chunks it on its own from there.
        else if (a == "--prompt-file") { const std::string pf = val(); std::ifstream f(pf, std::ios::binary);
            if (!f) { std::fprintf(stderr, "cannot read %s\n", pf.c_str()); return 2; }
            prompt.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()); }
        else if (a == "--chat") chat = true; else if (a == "--system") system = val();
        else if (a == "--n") n = uint32_t(std::atoi(val().c_str())); else if (a == "--temp") sp.temperature = std::stof(val());
        else if (a == "--top-k") sp.top_k = uint32_t(std::atoi(val().c_str())); else if (a == "--top-p") sp.top_p = std::stof(val());
        else if (a == "--min-p") sp.min_p = std::stof(val()); else if (a == "--repeat") sp.repeat_penalty = std::stof(val());
        else if (a == "--seed") sp.seed = std::strtoull(val().c_str(), nullptr, 10); else if (a == "--effort") effort = val();
        else if (a == "--chat-mode") thinking = false; else if (a == "--raw") raw = true; else if (a == "--max-tokens") cap = uint32_t(std::atoi(val().c_str()));
        else if (a == "--tables") tables = val(); else if (a == "--ranking") ranking = val(); else if (a == "--stop") stops.push_back(val());
        // Phase 27: write a DECODE-PHASE routing profile for this prompt, from real text rather than a
        // synthetic block -- the distribution a ranking is profiled on has to be the one it will serve.
        else if (a == "--profile-out") prof_out = val();
        // Phase 29: the RAW per-(layer, expert) decode selection counts, as CSV. The ranking file records only
        // the ORDER, which cannot answer how much selection mass a layer's top-N actually carries -- and that
        // is what decides whether a uniform static slot count per layer is leaving hit rate on the table.
        else if (a == "--counts-out") counts_out = val();
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); usage(); return 2; }
    }
    if (prompt.empty() && !chat) { usage(); return 2; }
    std::string err;
    ie::Ds41PromptOptions po; po.thinking = thinking; po.reasoning_effort = ie::ds41_reasoning_effort_of(effort, err);
    if (!err.empty()) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }

    ie::DeepSeek41Model m;
    const auto t0 = std::chrono::steady_clock::now();
    if (auto e = m.load(model); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    ie::Ds41EngramTables tb;
    if (auto e = tb.load(tables); !e.empty()) { std::fprintf(stderr, "engram tables (%s): %s\n", tables.c_str(), e.c_str()); return 1; }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_hf_json(model + "/tokenizer.json", model + "/tokenizer_config.json"); !e.empty()) { std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1; }
    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d);
    }
    if (devs.empty()) { std::fprintf(stderr, "no Arc GPU\n"); return 1; }
    std::vector<std::unique_ptr<sycl::queue>> queues; std::vector<sycl::queue*> qs;
    for (const auto& d : devs) { queues.push_back(std::make_unique<sycl::queue>(sycl::context(d), d, sycl::property_list{sycl::property::queue::in_order{}})); qs.push_back(queues.back().get()); }
    std::vector<std::vector<uint32_t>> rank;
    if (exists(ranking)) { if (auto e = ie::ds4_expert_priority_read_layers(ranking, m.config().n_routed_experts, m.config().n_layers, rank); !e.empty()) { std::fprintf(stderr, "ranking: %s\n", e.c_str()); return 1; } }
    else std::fprintf(stderr, "no ranking file at %s: index-order placement (slower)\n", ranking.c_str());
    // DSpark P4 (docs/deepseek41/58): IE_DS41_SPEC=1 -- the drafter on the head card, attached to the generator.
    // The environment the loop's losslessness needs is established HERE, before init_resident reads it (gate P4
    // finding 1): the T-row verify agrees with T one-row steps bit for bit only with the CPU miss split off
    // (docs/55), and the split serves T = 1 misses only (deepseek41_experts.cpp:517), so under speculation it is
    // idle arithmetic anyway. An explicit IE_DS41_CPU_MISS in the environment still wins (setenv does not
    // overwrite) -- and then the loop is NOT lossless, which the banner says.
    ie::Ds41Drafter drafter; const char* spec_env = std::getenv("IE_DS41_SPEC"); const bool spec = spec_env && std::string(spec_env) == "1";
    if (spec) {
        const bool had_cpu_miss = std::getenv("IE_DS41_CPU_MISS") != nullptr;
        setenv("IE_DS41_DECODE_MULTI", "1", 0);                                  // the verify step is a T = 1 + k decode row block (docs/55)
        setenv("IE_DS41_CPU_MISS", "0", 0);
        const bool split_on = std::string(std::getenv("IE_DS41_CPU_MISS")) != "0";
        std::fprintf(stderr, "[ds41 run] DSpark speculation ON (temperature 0 only): the CPU miss split %s\n", split_on
            ? "is ON by your IE_DS41_CPU_MISS -- the output may DIFFER from plain greedy (docs/58 gate finding 1)"
            : (had_cpu_miss ? "off, as you asked: the stream equals plain greedy run the same way (IE_DS41_CPU_MISS=0)"
                            : "off for exact losslessness: the stream equals plain greedy run the same way (IE_DS41_CPU_MISS=0), not the split-on default"));
    }
    ie::Ds41Forward fwd; ie::Ds41Forward::ResidentOptions opt; opt.max_tokens = cap;
    opt.head_fp8 = true;   // Phase 54: the server's head (IE_DS41_HEAD_FP8=0 for the BF16 one)
    if (auto e = fwd.init_resident(qs, m, tb, rank, opt); !e.empty()) { std::fprintf(stderr, "init_resident: %s\n", e.c_str()); return 1; }
    fwd.set_logits_last_only(true);
    if (spec) {
        if (auto e = drafter.init(*qs.back(), m, fwd.dense_cache(uint32_t(qs.size() - 1)), {}, ie::Ds41Drafter::Options{}); !e.empty()) { std::fprintf(stderr, "drafter: %s\n", e.c_str()); return 1; }
        std::fprintf(stderr, "[ds41 run] drafter %.2f GiB dense on card %zu, tier %u static / %u pinned per stage\n",
                     double(drafter.dense_bytes()) / 1073741824.0, qs.size() - 1, drafter.tier().n_static(), drafter.tier().n_pinned());
    }
    const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::fprintf(stderr, "ready in %.0f s on %zu card(s): capacity %u tokens, %s mode, effort %d, sampling temp %.2f top-k %u top-p %.2f min-p %.2f repeat %.2f\n",
                 load_s, devs.size(), cap, thinking ? "thinking" : "chat", po.reasoning_effort, sp.temperature, sp.top_k, sp.top_p, sp.min_p, sp.repeat_penalty);
    for (size_t c = 0; c < devs.size(); ++c) { const auto ci = fwd.card_info(uint32_t(c)); std::fprintf(stderr, "  card %zu: layers %u-%u, %u static + %u stream slots per layer, %u pinned\n", c, ci.first_layer, ci.first_layer + ci.n_layers - 1, ci.n_static, ci.n_stream, ci.n_pinned); }
    std::signal(SIGINT, on_sigint);
    ie::Ds41Generator gen(fwd, tok);
    if (std::getenv("IE_DS41_STAGES")) gen.set_accumulate_stages(true);
    if (!prof_out.empty() || !counts_out.empty()) gen.set_profile_decode_only(true);   // decode routing only
    if (spec) { gen.set_drafter(&drafter, qs.back());                                   // P5 (docs/59): the measurement knobs
        if (const char* v = std::getenv("IE_DS41_SPEC_K"); v && *v) gen.set_spec_k(uint32_t(std::atoi(v)));
        if (const char* v = std::getenv("IE_DS41_SPEC_CONF"); v && *v) gen.set_spec_conf(std::strtof(v, nullptr)); }

    std::vector<ie::Ds41ChatMessage> history;
    if (!system.empty()) history.push_back({"system", system, ""});
    auto answer = [&](const std::string& text_prompt) -> bool {
        const auto ids = tok.encode(text_prompt, /*allow_special=*/true);
        if (ids.size() + n > cap) { std::fprintf(stderr, "prompt %zu + %u tokens exceed --max-tokens %u\n", ids.size(), n, cap); return false; }
        std::string out_text; std::vector<int32_t> out; ie::Ds41GenStats st;
        const auto e = gen.run(ids, n, sp, stops, [&](std::string_view p) { std::fwrite(p.data(), 1, p.size(), stdout); std::fflush(stdout); out_text.append(p); return g_stop == 0; }, out, st);
        std::printf("\n");
        if (!e.empty()) { std::fprintf(stderr, "generate: %s\n", e.c_str()); return false; }
        std::fprintf(stderr, "[%u prompt tokens, prefill %.2f s = %.0f tok/s | %u generated in %.2f s = %.2f tok/s | stop: %s]\n",
                     st.n_prompt, st.prefill_s, st.prefill_s > 0 ? st.n_prompt / st.prefill_s : 0.0, st.n_gen, st.decode_s, st.decode_s > 0 ? st.n_gen / st.decode_s : 0.0, st.stop_reason.c_str());
        // IE_DS41_STAGES=1: the per-layer stage breakdown for THIS answer, so a real workload can be
        // budgeted the way the synthetic bench already can (docs/deepseek41/66). Decode-only: the counters
        // are read after generation, and prefill's contribution is dwarfed by 300 decode steps.
        if (std::getenv("IE_DS41_STAGES") && st.stages.steps) {   // Phase 50: the average over every decode step after the first
            const auto& a = st.stages; const double d = a.steps, NLTK = double(m.config().n_layers) * double(m.config().n_activated_experts);
            std::fprintf(stderr, "[stages, MEAN of %u decode steps, ms/token] wall %.1f = layers %.1f (attn %.1f + ffn_pre %.1f + moe %.1f + shared %.1f) + rest %.1f\n",
                         a.steps, a.wall / d, a.lay / d, a.att / d, a.fpre / d, a.moe / d, a.shd / d, (a.wall - a.lay) / d);
            std::fprintf(stderr, "[stages] mmap fill %.2f (read+permute batches %.2f: reads %.2f, permutes %.2f per reader) | moe prep %.2f, spawn %.2f\n",
                         a.mm_fill / d, a.mm_pack / d, a.mm_read / d, a.mm_perm / d, a.mprep / d, a.spawn / d);
            std::fprintf(stderr, "[stages] outside the layers: prep %.2f + head %.2f + untimed %.2f | host per token: sample %.2f, emit %.2f\n",
                         a.prep / d, a.head / d, (a.wall - a.lay - a.prep - a.head) / d, a.sample / d, a.emit / d);
            std::fprintf(stderr, "[stages] moe %.1f = groups %.1f + join %.1f + mmap_group %.1f + tail %.1f | cpu leg %.1f span, %.1f compute, %.1f experts -> %.2f ms/expert\n",
                         a.moe / d, a.grp / d, a.join / d, a.mmg / d, a.tail / d, a.cpu_ms / d, a.cpu_w / d, a.cpu / d, a.cpu > 0 ? a.cpu_w / a.cpu : 0.0);
            std::fprintf(stderr, "[stages] experts/token static %.1f pinned %.1f (stream hits %.1f) mmap %.1f cpu %.1f of %.0f -> hit rate %.1f%% | pinned %.1f MiB, disk %.1f MiB per token\n",
                         a.sta / d, a.pin / d, a.sh / d, a.mmx / d, a.cpu / d, NLTK, 100.0 * (a.sta + a.sh) / d / NLTK, a.bp / d / 1048576.0, a.bm / d / 1048576.0);
        }
        if (std::getenv("IE_DS41_STAGES") && st.n_gen) {
            double att = 0, fpre = 0, moe = 0, grp = 0, join = 0, mmg = 0, tail = 0, shd = 0, lay = 0, cpu_ms = 0, cpu_w = 0;
            double bp = 0, bm = 0, sta = 0, pin = 0, mmx = 0, sh = 0, cpu = 0;
            for (const auto& L : fwd.stats()) {
                att += L.attn_ms; fpre += L.ffn_pre_ms; moe += L.moe_call_ms; grp += L.moe_groups_ms;
                join += L.moe_join_ms; mmg += L.moe_mmap_group_ms; tail += L.moe_tail_ms; shd += L.shared_ms;
                lay += L.ms; cpu_ms += L.moe_cpu_ms; cpu_w += L.moe_cpu_work_ms;
                bp += double(L.bytes_pinned); bm += double(L.bytes_mmap);
                sta += L.experts_static; pin += L.experts_pinned; mmx += L.experts_mmap; sh += L.experts_stream_hit; cpu += L.experts_cpu;
            }
            const double NLTK = double(m.config().n_layers) * double(m.config().n_activated_experts);
            std::fprintf(stderr, "[stages, LAST decode step, ms] layers %.1f = attn %.1f + ffn_pre %.1f + moe %.1f + shared %.1f\n",
                         lay, att, fpre, moe, shd);
            std::fprintf(stderr, "[stages] moe %.1f = groups %.1f + join %.1f + mmap_group %.1f + tail %.1f | cpu leg %.1f span, %.1f compute\n",
                         moe, grp, join, mmg, tail, cpu_ms, cpu_w);
            std::fprintf(stderr, "[stages] experts static %.0f pinned %.0f mmap %.0f cpu %.0f of %.0f, stream hits %.0f -> hit rate %.1f%% | pinned %.1f MiB, disk %.1f MiB\n",
                         sta, pin, mmx, cpu, NLTK, sh, 100.0 * (sta + sh) / NLTK, bp / 1048576.0, bm / 1048576.0);
        }
        if (st.spec_passes) std::fprintf(stderr, "[spec: %u passes, L histogram 1..6 = %u %u %u %u %u %u, mean %.2f tokens/pass; draft %.1f + verify %.1f + rollback %.2f ms per pass]\n",
                                         st.spec_passes, st.spec_hist[1], st.spec_hist[2], st.spec_hist[3], st.spec_hist[4], st.spec_hist[5], st.spec_hist[6],
                                         double(st.n_gen) / st.spec_passes, st.spec_draft_ms / st.spec_passes, st.spec_verify_ms / st.spec_passes, st.spec_rollback_ms / st.spec_passes);
        // the history keeps the answer split at </think>: reasoning before, content after
        std::string reasoning, content = out_text;
        if (const auto at = out_text.find("</think>"); at != std::string::npos) { reasoning = out_text.substr(0, at); content = out_text.substr(at + 8); }
        history.push_back({"assistant", content, reasoning});
        if (!counts_out.empty()) {
            std::ofstream f(counts_out);
            if (!f) std::fprintf(stderr, "cannot write %s\n", counts_out.c_str());
            else {
                f << "# V4.1 decode selection counts, " << st.n_prompt << " prompt tokens, " << st.n_gen << " steps\n";
                f << "layer,expert,count\n";
                const auto& pr = fwd.profile();
                for (size_t L = 0; L < pr.size(); ++L)
                    for (size_t e = 0; e < pr[L].size(); ++e)
                        if (pr[L][e]) f << L << ',' << e << ',' << pr[L][e] << '\n';
                std::fprintf(stderr, "[wrote decode selection counts -> %s]\n", counts_out.c_str());
            }
        }
        if (!prof_out.empty()) {
            // The header names the phase, the context and the token count, because the file format records
            // none of them -- and a ranking profiled on the wrong distribution is worse than no ranking:
            // measured, one profiled on a repeated synthetic block made real-text decode 5.46 -> 3.45 tok/s.
            char note[512];
            std::snprintf(note, sizeof note,
                          "V4.1 DECODE-PHASE profile from REAL text: %u prompt tokens, %u decode steps, %.0f "
                          "selections/layer (%s)", st.n_prompt, st.n_gen,
                          double(st.n_gen) * m.config().n_activated_experts,
                          st.n_gen >= 256 ? "sufficient per docs/67" : "UNDER-SAMPLED: docs/67 asks for >= 256 steps");
            if (auto e = fwd.write_profile(prof_out, note); !e.empty()) std::fprintf(stderr, "write_profile: %s\n", e.c_str());
            else std::fprintf(stderr, "[wrote decode-phase ranking from %u real-text steps -> %s%s]\n",
                              st.n_gen, prof_out.c_str(), st.n_gen >= 256 ? "" : "  [UNDER-SAMPLED]");
        }
        return g_stop == 0;
    };

    if (!prompt.empty()) {
        std::string text;
        if (raw) text = prompt;
        else { history.push_back({"user", prompt, ""}); text = ie::ds41_encode_messages(history, po, err); if (!err.empty()) { std::fprintf(stderr, "prompt format: %s\n", err.c_str()); return 1; } }
        answer(text);
    } else {
        std::fprintf(stderr, "chat: type a message and Enter; an empty line or Ctrl-D ends the session\n");
        std::string line;
        while (!g_stop) {
            std::fprintf(stderr, "> "); if (!std::getline(std::cin, line) || line.empty()) break;
            history.push_back({"user", line, ""});
            const std::string text = ie::ds41_encode_messages(history, po, err);
            if (!err.empty()) { std::fprintf(stderr, "prompt format: %s\n", err.c_str()); break; }
            if (!answer(text)) break;
        }
    }
    if (spec) drafter.free(*qs.back());
    fwd.free_resident();
    for (auto& qp : queues) qp->wait_and_throw();
    std::fprintf(stderr, "unloaded\n");
    return 0;
}
