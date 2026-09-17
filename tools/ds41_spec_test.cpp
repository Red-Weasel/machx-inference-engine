// tools/ds41_spec_test.cpp — V4.1 DSpark P4 (docs/deepseek41/58): the speculative loop, greedy, lossless.
//   lossless: for each prompt in the range, 256 greedy tokens through the plain loop, then 256 through the
//             speculative loop from a fresh prefill; the id streams compared token for token (a hard failure
//             under IE_DS41_CPU_MISS=0; reported with the split on — docs/55: the CPU miss leg's rows differ in
//             the last bits from the GPU's); stop_reason and the text compared; the stop-string case and the
//             mid-block length case; the 2,048-token context timed both ways (information for P5).
//   stats:    512 speculative tokens per prompt in the range: the acceptance histogram, the per-position rates,
//             the confidence head bucketed (measured, not used); the records to <conf_out> when given.
//   prompts 0-5 (the golden's 12-token prompt, five raw English prompts); 6-10 = the golden's 2,048-token pp text and
//   its first 512 / 1,024 / 256 / 1,536 / 768 tokens -- document continuations, the five prompts that run the full 256
//   tokens without an eos (docs/48 P4 criterion 1 wants >= 5 x >= 256) and the long generations the stats need.
//   bench:    the steady state, one process per curve (docs/deepseek41/59): for each setting in the list -- `off`
//             (the plain loop), `k<1..5>` (the verify block's length), `c<float>` (the confidence threshold at
//             k = 5) -- `samples` runs of `warmup + measured` tokens timed from the warm-up mark, every stream
//             compared with the plain reference (docs/59 criterion 4: a scheduler must not change the output).
//   usage: ie-ds41-spec-test <model> <golden_dir> [lossless|stats] [first-last] [conf_out]
//          ie-ds41-spec-test <model> <golden_dir> bench <prompt> <warmup> <measured> <settings> [samples]
#include "ie/deepseek41_dspark.hpp"
#include "ie/deepseek41_generate.hpp"
#include "ie/expert_stream.hpp"

#include "../third_party/nlohmann/json.hpp"
#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {
int g_fail = 0;
void check(bool ok, const std::string& w, const std::string& d = "") { std::printf("%s%s%s\n", ok ? "[ ok ] " : "[FAIL] ", w.c_str(), d.empty() ? "" : ("  (" + d + ")").c_str()); if (!ok) ++g_fail; }
void info(const std::string& w, const std::string& d = "") { std::printf("[info] %s%s\n", w.c_str(), d.empty() ? "" : ("  (" + d + ")").c_str()); }
template <class T> std::vector<T> rd(const std::string& p, size_t n) { std::vector<T> v(n); std::ifstream f(p, std::ios::binary); if (!f) return {}; f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(T))); return f ? v : std::vector<T>{}; }
std::string ids_str(const std::vector<int32_t>& v, size_t a, size_t b) { std::string s; for (size_t i = a; i < b && i < v.size(); ++i) s += std::to_string(v[i]) + " "; return s; }
std::string f1(double v) { char b[32]; std::snprintf(b, sizeof b, "%.1f", v); return b; }
std::string f2(double v) { char b[32]; std::snprintf(b, sizeof b, "%.2f", v); return b; }

struct Run { std::vector<int32_t> ids; std::string text; ie::Ds41GenStats st; std::string err; };
Run gen_run(ie::Ds41Generator& g, const std::vector<int32_t>& prompt, uint32_t n, const std::vector<std::string>& stops) {
    Run r; ie::Ds41SampleParams sp;                                            // temperature 0: argmax, no penalty
    r.err = g.run(prompt, n, sp, stops, [&](std::string_view p) { r.text.append(p); return true; }, r.ids, r.st);
    return r;
}
std::string spec_line(const ie::Ds41GenStats& st) {
    if (!st.spec_passes) return "no passes";
    std::string h; for (uint32_t L = 1; L <= 6; ++L) h += std::to_string(st.spec_hist[L]) + (L < 6 ? " " : "");
    std::string pa; for (uint32_t r = 0; r < 5; ++r) pa += (st.spec_offered[r] ? f2(double(st.spec_accepted[r]) / st.spec_offered[r]) : std::string("-")) + (r < 4 ? " " : "");
    return std::to_string(st.spec_passes) + " passes, L 1..6 = " + h + ", mean " + f2(double(st.n_gen) / st.spec_passes) + " tokens/pass, per-position acceptance " + pa
         + "; per pass draft " + f1(st.spec_draft_ms / st.spec_passes) + " + verify " + f1(st.spec_verify_ms / st.spec_passes) + " + rollback " + f2(st.spec_rollback_ms / st.spec_passes) + " ms";
}
}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : std::string(std::getenv("HOME")) + "/models/DeepSeek-V4.1-Flash";
    const std::string gd  = argc > 2 ? argv[2] : dir + "/ie_golden";
    const std::string mode = argc > 3 ? argv[3] : "lossless";
    uint32_t first = 0, last = 11; if (argc > 4 && mode != "bench") { if (std::sscanf(argv[4], "%u-%u", &first, &last) != 2) { std::fprintf(stderr, "range: first-last\n"); return 2; } }
    const std::string conf_out = argc > 5 ? argv[5] : "";
    if (mode != "lossless" && mode != "stats" && mode != "bench") { std::fprintf(stderr, "mode: lossless | stats | bench\n"); return 2; }
    uint32_t b_prompt = 6, b_warm = 8, b_meas = 32, b_samples = 3; std::vector<std::string> b_sets;
    if (mode == "bench") {
        if (argc < 8) { std::fprintf(stderr, "bench <prompt> <warmup> <measured> <settings> [samples]\n"); return 2; }
        b_prompt = uint32_t(std::atoi(argv[4])); b_warm = uint32_t(std::atoi(argv[5])); b_meas = uint32_t(std::atoi(argv[6]));
        if (argc > 8) b_samples = uint32_t(std::atoi(argv[8]));
        std::string all = argv[7]; for (size_t a = 0, b; a <= all.size(); a = b + 1) { b = std::min(all.find(',', a), all.size()); if (b > a) b_sets.push_back(all.substr(a, b - a)); }
        if (b_sets.empty() || !b_samples) { std::fprintf(stderr, "bench: empty settings or samples\n"); return 2; }
    }
    const char* cm = std::getenv("IE_DS41_CPU_MISS"); const bool exact = cm && std::string(cm) == "0";   // docs/58 criterion 1

    nlohmann::json dmeta; { std::ifstream f(gd + "/decode2/d_meta.json"); if (!f) { std::fprintf(stderr, "no decode2/d_meta.json\n"); return 1; } f >> dmeta; }
    ie::DeepSeek41Model m;
    if (auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    if (!m.dspark) { std::fprintf(stderr, "the model has no DSpark drafter bound\n"); return 1; }
    ie::Ds41EngramTables tb;
    if (auto e = tb.load(gd); !e.empty()) { std::fprintf(stderr, "tables: %s\n", e.c_str()); return 1; }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_hf_json(dir + "/tokenizer.json", dir + "/tokenizer_config.json"); !e.empty()) { std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1; }
    const auto& c = m.config(); const uint32_t E = c.n_routed_experts, NL = c.n_layers;
    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) { if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue; for (const auto& d : p.get_devices(sycl::info::device_type::gpu)) if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d); }
    if (devs.empty()) { std::fprintf(stderr, "no Arc GPU\n"); return 1; }
    std::vector<std::unique_ptr<sycl::queue>> queues; std::vector<sycl::queue*> qs;
    for (const auto& d : devs) { queues.push_back(std::make_unique<sycl::queue>(sycl::context(d), d, sycl::property_list{sycl::property::queue::in_order{}})); qs.push_back(queues.back().get()); }
    std::vector<std::vector<uint32_t>> ranking; const std::string rank_in = dir + "/ie_ranking_heldout.txt";
    if (std::ifstream(rank_in).good()) if (auto e = ie::ds4_expert_priority_read_layers(rank_in, E, NL, ranking); !e.empty()) { std::fprintf(stderr, "ranking: %s\n", e.c_str()); return 1; }
    ie::Ds41Forward fwd; ie::Ds41Forward::ResidentOptions opt;
    // the position capacity the mode actually needs (the latent caches and the tier's arena are sized from it, and
    // asking for more than the run uses has made the tier's VRAM claim fail on a busy box): the 2,048-token
    // document prompt plus what will be generated plus one verify block
    opt.max_tokens = 2048 + 64 + (mode == "stats" ? 1024 : mode == "bench" ? b_warm + b_meas + 8 : 320);
    const auto t0 = std::chrono::steady_clock::now();
    if (auto e = fwd.init_resident(qs, m, tb, ranking, opt); !e.empty()) { std::fprintf(stderr, "init_resident: %s\n", e.c_str()); return 1; }
    sycl::queue& dq = *qs.back(); ie::Ds41Drafter drafter;
    if (auto e = drafter.init(dq, m, fwd.dense_cache(uint32_t(qs.size() - 1)), {}, ie::Ds41Drafter::Options{}); !e.empty()) { std::fprintf(stderr, "drafter init: %s\n", e.c_str()); fwd.free_resident(); return 1; }
    std::printf("ready in %.0f s on %zu card(s); drafter %.2f GiB dense, tier %u static / %u pinned per stage; CPU miss split %s (%s)\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), devs.size(), double(drafter.dense_bytes()) / 1073741824.0,
                drafter.tier().n_static(), drafter.tier().n_pinned(), exact ? "OFF" : "ON", exact ? "criterion 1 is a hard failure on any divergence" : "divergences are reported, not failed");
    ie::Ds41Generator plain(fwd, tok), spec(fwd, tok); spec.set_drafter(&drafter, &dq);

    // the prompts (docs/58 build item 3)
    std::vector<std::pair<std::string, std::vector<int32_t>>> prompts;
    prompts.push_back({"the golden's 12-token prompt", dmeta["prompt_ids"].get<std::vector<int32_t>>()});
    for (const char* t : {"The three primary colors are", "Once upon a time, in a small village by the sea, there lived",
                          "Write a Python function that returns the n-th Fibonacci number.\n\ndef fib(n):", "List five reasons why the sky appears blue:\n1.",
                          "Explain, step by step, how a bill becomes a law in the United States."})
        prompts.push_back({std::string("\"") + (std::strlen(t) > 40 ? std::string(t, 40) + "..." : std::string(t)) + "\"", tok.encode(t, /*allow_special=*/true)});
    if (const auto pp = rd<int32_t>(gd + "/pp_ids_2048.i32", 2048); pp.size() == 2048) {                       // prompts 6-8: the document text (long generations for the stats)
        prompts.push_back({"the 2,048-token pp text", pp});
        prompts.push_back({"the pp text's first 512 tokens", std::vector<int32_t>(pp.begin(), pp.begin() + 512)});
        prompts.push_back({"the pp text's first 1,024 tokens", std::vector<int32_t>(pp.begin(), pp.begin() + 1024)});
        prompts.push_back({"the pp text's first 256 tokens", std::vector<int32_t>(pp.begin(), pp.begin() + 256)});
        prompts.push_back({"the pp text's first 1,536 tokens", std::vector<int32_t>(pp.begin(), pp.begin() + 1536)});
        prompts.push_back({"the pp text's first 768 tokens", std::vector<int32_t>(pp.begin(), pp.begin() + 768)});
    }
    last = std::min<uint32_t>(last, uint32_t(prompts.size() - 1));

    if (mode == "lossless") {
        for (uint32_t i = first; i <= last; ++i) {
            const auto& [name, ids] = prompts[i];
            const Run a = gen_run(plain, ids, 256, {}); if (!a.err.empty()) { check(false, "prompt " + std::to_string(i) + " plain: " + a.err); continue; }
            const Run b = gen_run(spec, ids, 256, {});  if (!b.err.empty()) { check(false, "prompt " + std::to_string(i) + " speculative: " + b.err); continue; }
            size_t k = 0; while (k < a.ids.size() && k < b.ids.size() && a.ids[k] == b.ids[k]) ++k;
            const bool same = a.ids == b.ids;
            const std::string d = same ? std::to_string(a.ids.size()) + " tokens" : "first difference at " + std::to_string(k) + " of " + std::to_string(a.ids.size()) + " / " + std::to_string(b.ids.size()) + ": plain " + ids_str(a.ids, k, k + 4) + "| spec " + ids_str(b.ids, k, k + 4);
            if (a.ids.empty() && b.ids.empty())                                // eos at once: nothing to compare, and an empty == empty is not evidence
                info("prompt " + std::to_string(i) + " (" + name + "): both loops emitted 0 tokens (eos at once) — NOT evidence of losslessness (gate P4 finding 3)");
            else if (exact || same) check(same, "prompt " + std::to_string(i) + " (" + name + "): the speculative stream equals the plain greedy stream token for token (" + std::to_string(a.ids.size()) + " tokens)", d);
            else info("prompt " + std::to_string(i) + " (" + name + "): the streams differ with the CPU miss split ON (reported, docs/58 criterion 1)", d);
            check(a.st.stop_reason == b.st.stop_reason && (same ? a.text == b.text : true), "prompt " + std::to_string(i) + ": stop_reason and text agree", a.st.stop_reason + " / " + b.st.stop_reason + ", n_gen " + std::to_string(a.st.n_gen) + " / " + std::to_string(b.st.n_gen));
            std::printf("       prompt %u: plain %.1f ms/token, speculative %.1f ms/token (%s)\n", i, 1000.0 * a.st.decode_s / std::max(1u, a.st.n_gen), 1000.0 * b.st.decode_s / std::max(1u, b.st.n_gen), spec_line(b.st).c_str());
        }
        // criterion 4: a stop string; the length budget landing mid-block
        { const Run a = gen_run(plain, prompts[0].second, 64, {"."}), b = gen_run(spec, prompts[0].second, 64, {"."});
          check(a.err.empty() && b.err.empty() && a.ids == b.ids && a.text == b.text && a.st.stop_reason == "stop" && b.st.stop_reason == "stop",
                "a stop string '.' ends both loops at the same token with the same text", "'" + a.text + "' / '" + b.text + "', " + a.st.stop_reason + " / " + b.st.stop_reason + (a.err.empty() && b.err.empty() ? "" : " " + a.err + b.err)); }
        // a stop string that lands past the first verify block (gate P4 finding 8: the '.' case ends in the first pass)
        { std::vector<std::string> st_late; Run a;                            // the first candidate the plain loop hits past the first block
          for (const char* cand : {" and", " the", "2.", " is", ","}) {
              st_late.assign(1, cand); a = gen_run(plain, prompts[4].second, 64, st_late);
              if (a.err.empty() && a.st.stop_reason == "stop" && a.st.n_gen > 6) break;
          }
          const Run b = gen_run(spec, prompts[4].second, 64, st_late);
          if (a.err.empty() && a.st.stop_reason == "stop" && a.st.n_gen > 6)
              check(b.err.empty() && a.ids == b.ids && a.text == b.text && b.st.stop_reason == "stop",
                    "the stop string '" + st_late[0] + "' landing past the first block (token " + std::to_string(a.st.n_gen) + ") ends both loops identically",
                    std::to_string(a.st.n_gen) + " / " + std::to_string(b.st.n_gen) + " tokens, " + a.st.stop_reason + " / " + b.st.stop_reason + (a.text == b.text ? ", same text" : ", TEXT DIFFERS"));
          else info("the late-stop-string case did not arise with any candidate (plain: " + a.st.stop_reason + ", " + std::to_string(a.st.n_gen) + " tokens" + (a.err.empty() ? "" : ", " + a.err) + ")"); }
        { const Run a = gen_run(plain, prompts[1].second, 7, {}), b = gen_run(spec, prompts[1].second, 7, {});
          check(a.err.empty() && b.err.empty() && a.ids == b.ids && a.st.n_gen == 7 && b.st.n_gen == 7 && b.st.stop_reason == "length",
                "a token budget of 7 (inside the second block) ends both loops after exactly 7 tokens, nothing emitted past it", ids_str(a.ids, 0, 7) + "/ " + ids_str(b.ids, 0, 7) + b.st.stop_reason); }
        // information for P5: the 2,048-token context, 48 tokens both ways
        if (const auto pp = rd<int32_t>(gd + "/pp_ids_2048.i32", 2048); pp.size() == 2048) {
            const Run a = gen_run(plain, pp, 48, {}), b = gen_run(spec, pp, 48, {});
            if (!a.err.empty() || !b.err.empty()) check(false, "the 2,048-token context runs", a.err + b.err);
            else { const bool same = a.ids == b.ids;
                   if (exact || same) check(same, "the 2,048-token context: 48 speculative tokens equal the plain greedy ones", same ? "48 tokens" : "differ");
                   else info("the 2,048-token context: the streams differ with the CPU miss split ON (reported)");
                   std::printf("       2,048-token context: plain %.1f ms/token, speculative %.1f ms/token (%s)\n", 1000.0 * a.st.decode_s / std::max(1u, a.st.n_gen), 1000.0 * b.st.decode_s / std::max(1u, b.st.n_gen), spec_line(b.st).c_str()); }
        } else info("no pp_ids_2048.i32 in the golden dir: the 2,048-token timing skipped");
    } else if (mode == "stats") {
        ie::Ds41GenStats all; uint32_t n_gen = 0;
        std::vector<ie::Ds41GenStats::SpecConf> recs;
        for (uint32_t i = first; i <= last; ++i) {
            const Run b = gen_run(spec, prompts[i].second, i >= 6 ? 1024 : 512, {});   // the document prompts run long
            if (!b.err.empty()) { check(false, "prompt " + std::to_string(i) + " speculative: " + b.err); continue; }
            std::printf("       prompt %u (%s): %u tokens, %.1f ms/token, stop %s (%s)\n", i, prompts[i].first.c_str(), b.st.n_gen, 1000.0 * b.st.decode_s / std::max(1u, b.st.n_gen), b.st.stop_reason.c_str(), spec_line(b.st).c_str());
            all.spec_passes += b.st.spec_passes; n_gen += b.st.n_gen; for (uint32_t L = 0; L < 7; ++L) all.spec_hist[L] += b.st.spec_hist[L];
            for (uint32_t r = 0; r < 5; ++r) { all.spec_offered[r] += b.st.spec_offered[r]; all.spec_accepted[r] += b.st.spec_accepted[r]; }
            all.spec_draft_ms += b.st.spec_draft_ms; all.spec_verify_ms += b.st.spec_verify_ms; all.spec_rollback_ms += b.st.spec_rollback_ms;
            recs.insert(recs.end(), b.st.spec_conf.begin(), b.st.spec_conf.end());
        }
        all.n_gen = n_gen;
        std::printf("       ALL: %s\n", spec_line(all).c_str());
        check(all.spec_passes > 0, "the acceptance histogram over the range's passes (criterion 2 wants >= 1,000 over the whole set)", std::to_string(all.spec_passes) + " passes");
        // criterion 3: the confidence head bucketed per position (8 equal-width buckets between the observed min and max)
        for (uint32_t r = 0; r < 5; ++r) {
            float lo = 1e30f, hi = -1e30f; for (const auto& x : recs) if (x.pos == r) { lo = std::min(lo, x.conf); hi = std::max(hi, x.conf); }
            if (lo > hi) continue;
            uint32_t n[8] = {}, acc[8] = {};
            for (const auto& x : recs) if (x.pos == r) { const uint32_t b = std::min<uint32_t>(7, uint32_t(std::max(0.f, (x.conf - lo) / std::max(1e-6f, hi - lo)) * 8)); ++n[b]; acc[b] += x.accepted; }
            std::string line; for (uint32_t b = 0; b < 8; ++b) line += (n[b] ? f2(double(acc[b]) / n[b]) + "(" + std::to_string(n[b]) + ")" : std::string("-")) + " ";
            std::printf("       confidence, draft position %u: range [%.2f, %.2f], acceptance per bucket %s\n", r + 1, lo, hi, line.c_str());
        }
        if (!conf_out.empty()) { std::ofstream f(conf_out); f << "pos\tconf\taccepted\n"; for (const auto& x : recs) f << int(x.pos) + 1 << '\t' << x.conf << '\t' << int(x.accepted) << '\n'; std::printf("       %zu records -> %s\n", recs.size(), conf_out.c_str()); }
    }
    if (mode == "bench") {
        if (b_prompt >= prompts.size()) { check(false, "bench: prompt " + std::to_string(b_prompt) + " of " + std::to_string(prompts.size())); }
        else {
            const auto& [name, pids] = prompts[b_prompt]; const uint32_t total = b_warm + b_meas;
            std::printf("bench: prompt %u (%s, %zu tokens), %u warm-up + %u measured tokens, %u samples per setting\n",
                        b_prompt, name.c_str(), pids.size(), b_warm, b_meas, b_samples);
            plain.set_warmup_mark(b_warm); spec.set_warmup_mark(b_warm);
            const Run ref = gen_run(plain, pids, total, {});                       // the reference stream (and the plain loop's first sample)
            if (!ref.err.empty()) { check(false, "bench reference: " + ref.err); }
            else {
                std::printf("       %-8s %10s %8s %7s %7s %8s %9s %9s %8s %7s  %s\n", "setting", "ms/token", "tok/s", "mean L", "rows", "draft", "verify", "link MB", "MB/token", "union", "lossless");
                for (const auto& set : b_sets) {
                    uint32_t k = 0; float th = -1e30f; bool off = set == "off", bad = false;
                    if (!off) { if (set[0] == 'k') k = uint32_t(std::atoi(set.c_str() + 1)); else if (set[0] == 'c') { th = std::strtof(set.c_str() + 1, nullptr); k = 0; } else bad = true; }
                    if (bad) { check(false, "bench: setting '" + set + "' is not off | k<n> | c<float>"); continue; }
                    ie::Ds41Generator& g = off ? plain : spec; g.set_spec_k(k); g.set_spec_conf(th);
                    double ms_sum = 0, ms_min = 1e30, ms_max = 0; bool same = true, ok = true; ie::Ds41GenStats acc{}; uint32_t n_gen = 0, samples = 0;
                    for (uint32_t s2 = 0; s2 < b_samples; ++s2) {
                        // every setting's samples are fresh runs. The reference is the process's FIRST generation and
                        // the slots drift ~1.2 % slower over a process, so reusing it as `off`'s first sample handed
                        // the baseline the fastest slot in the process (gate P5 finding 4) -- it is only the stream
                        // to compare against now, never a timing sample.
                        const Run r = gen_run(g, pids, total, {});
                        if (!r.err.empty()) { check(false, "bench " + set + ": " + r.err); ok = false; break; }
                        if (r.ids != ref.ids) same = false;
                        if (!r.st.warm_n) { check(false, "bench " + set + ": the run ended before the warm-up mark (" + std::to_string(r.st.n_gen) + " tokens, stop " + r.st.stop_reason + ")"); ok = false; break; }
                        const double ms = 1000.0 * r.st.warm_decode_s / r.st.warm_n;
                        ms_sum += ms; ms_min = std::min(ms_min, ms); ms_max = std::max(ms_max, ms); ++samples; n_gen += r.st.n_gen;
                        acc.spec_passes += r.st.spec_passes; acc.spec_rows_verified += r.st.spec_rows_verified; acc.spec_declined += r.st.spec_declined;
                        acc.spec_link_bytes += r.st.spec_link_bytes; acc.spec_union_sum += r.st.spec_union_sum; acc.spec_union_layers += r.st.spec_union_layers;
                        acc.spec_draft_ms += r.st.spec_draft_ms; acc.spec_verify_ms += r.st.spec_verify_ms; acc.spec_rollback_ms += r.st.spec_rollback_ms;
                        for (uint32_t L = 0; L < 7; ++L) acc.spec_hist[L] += r.st.spec_hist[L];
                        for (uint32_t r2 = 0; r2 < 5; ++r2) { acc.spec_offered[r2] += r.st.spec_offered[r2]; acc.spec_accepted[r2] += r.st.spec_accepted[r2]; }
                    }
                    if (!ok || !samples) continue;
                    const double ms = ms_sum / samples, P = std::max(1u, acc.spec_passes);
                    double meanL = 0; for (uint32_t L = 1; L <= 6; ++L) meanL += double(L) * acc.spec_hist[L]; meanL = acc.spec_passes ? meanL / P : 1.0;
                    std::printf("       %-8s %10s %8s %7s %7s %8s %9s %9s %8s %7s  %s\n", set.c_str(), f1(ms).c_str(), f2(1000.0 / ms).c_str(), f2(meanL).c_str(),
                                acc.spec_passes ? f2(double(acc.spec_rows_verified) / P).c_str() : "1.00", acc.spec_passes ? f1(acc.spec_draft_ms / P).c_str() : "-",
                                acc.spec_passes ? f1(acc.spec_verify_ms / P).c_str() : "-", acc.spec_passes ? f1(double(acc.spec_link_bytes) / P / 1048576.0).c_str() : "-",
                                n_gen ? f1(double(acc.spec_link_bytes) / n_gen / 1048576.0).c_str() : "-",
                                acc.spec_union_layers ? f1(double(acc.spec_union_sum) / acc.spec_union_layers).c_str() : "-", same ? "yes" : "NO");
                    std::printf("                spread %s-%s ms over %u samples; L 1..6 = %u %u %u %u %u %u; declined %u of %u passes; rollback %s ms\n",
                                f1(ms_min).c_str(), f1(ms_max).c_str(), samples, acc.spec_hist[1], acc.spec_hist[2], acc.spec_hist[3], acc.spec_hist[4], acc.spec_hist[5], acc.spec_hist[6],
                                acc.spec_declined, acc.spec_passes, acc.spec_passes ? f2(acc.spec_rollback_ms / P).c_str() : "-");
                    if (exact || same) check(same, "bench " + set + ": the stream equals the plain reference token for token (docs/59 criterion 4)", same ? std::to_string(ref.ids.size()) + " tokens" : "DIVERGED");
                    else info("bench " + set + ": the stream differs from the plain reference with the CPU miss split ON (expected, docs/58 gate finding 1)");
                }
                std::printf("       drafter residency: %u static / %u pinned per stage (docs/59: runs are comparable only at the same residency)\n", drafter.tier().n_static(), drafter.tier().n_pinned());
            }
        }
    }
    drafter.free(dq); fwd.free_resident();
    std::printf("\nSPEC TEST (%s): %s\n", mode.c_str(), g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
