// tools/mimo26_run.cpp -- ie-mimo26-run: MiMo-V2.6 greedy generation on 1-2 cards (P2 bring-up, docs/mimo26/00_PORT_PLAN.md).
//   ie-mimo26-run <model_dir> (--prompt TEXT | --prompt-file FILE) [--n N] [--chunk C] [--ctx MAX]
//                 [--static S] [--pinned P] [--stream Q] [--no-special] [--ranking FILE] [--kprof] [--kprof-prefill]
// --static 0 sizes each card's static expert tier from its free VRAM (Mimo26Options::n_static). --lookup decodes with
// prompt-lookup speculation (greedy; the engine's mimo26_run_ids loop): a copy of >= 12 context tokens verified up to
// 8 rows at a time.
// The prompt is tokenized raw (special tokens parsed unless --no-special; no chat template), prefilled in chunks of C,
// then N tokens are generated greedily. Prints the prompt ids, the generated ids as one list, the text, and the
// prefill / decode rates. Stops early on the eos or pad token. --prompt-file may repeat: each prompt runs in turn on
// the one loaded model (the caches reset between them), its block opened by a "=== <file>" line. --kprof prints the decode
// steps' kernels (ms per token), --kprof-prefill the prefill chunks' (ms per chunk).
#include "ie/expert_stream.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/mimo26.hpp"
#include "ie/mimo26_forward.hpp"
#include "ie/ngram_draft.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: ie-mimo26-run <model_dir> (--prompt TEXT | --prompt-file FILE) [--n N] [--chunk C] [--ctx MAX] [--static S] [--pinned P] [--stream Q] [--no-special] [--ranking FILE] [--kprof] [--kprof-prefill] [--lookup]\n"); return 2; }
    const std::string model = argv[1];
    std::vector<std::pair<std::string, std::string>> prompts; uint32_t N = 32, C = 512, ctx = 4096; bool special = true; std::string ranking_path; bool kprof = false, kprof_pf = false, lookup = false;
    ie::Mimo26Options opt;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i]; auto val = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--prompt") prompts.push_back({"--prompt", val()});
        else if (a == "--prompt-file") { const std::string f = val(); std::ifstream in(f, std::ios::binary); if (!in) { std::fprintf(stderr, "cannot read %s\n", f.c_str()); return 2; } prompts.push_back({f, std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>())}); }
        else if (a == "--n") N = uint32_t(std::atol(val().c_str()));
        else if (a == "--chunk") C = uint32_t(std::atol(val().c_str()));
        else if (a == "--ctx") ctx = uint32_t(std::atol(val().c_str()));
        else if (a == "--static") opt.n_static = uint32_t(std::atol(val().c_str()));
        else if (a == "--pinned") opt.n_pinned = uint32_t(std::atol(val().c_str()));
        else if (a == "--stream") opt.stream_slots = uint32_t(std::atol(val().c_str()));
        else if (a == "--no-special") special = false;
        else if (a == "--ranking") ranking_path = val();
        else if (a == "--kprof") kprof = true;
        else if (a == "--kprof-prefill") kprof_pf = true;
        else if (a == "--lookup") lookup = true;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (prompts.empty()) { std::fprintf(stderr, "no prompt\n"); return 2; }

    ie::Mimo26Model m;
    if (auto e = m.load(model); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_hf_json(model + "/tokenizer.json", model + "/tokenizer_config.json"); !e.empty()) { std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1; }
    for (const auto& [name, text] : prompts) {
        const auto t = tok.encode(text, special);
        if (t.empty()) { std::fprintf(stderr, "%s tokenized to nothing\n", name.c_str()); return 2; }
        if (t.size() + N > ctx) { std::fprintf(stderr, "%s: prompt (%zu) + n (%u) exceeds --ctx %u\n", name.c_str(), t.size(), N, ctx); return 2; }
    }

    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d);
    }
    if (devs.empty()) { std::fprintf(stderr, "no Arc GPU\n"); return 1; }
    std::vector<std::unique_ptr<sycl::queue>> queues; std::vector<sycl::queue*> qs;
    for (const auto& d : devs) {
        const sycl::property_list pl = kprof || kprof_pf ? sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}
                                             : sycl::property_list{sycl::property::queue::in_order{}};
        queues.push_back(std::make_unique<sycl::queue>(sycl::context(d), d, pl)); qs.push_back(queues.back().get());
    }

    // the kernels recorded since begin_step(), summed and divided over `units` (decode tokens or prefill chunks)
    auto report = [&](ie::KernelProfiler& kp, size_t units, const char* unit, double wall_ms) {
        for (auto* qq : qs) qq->wait();
        auto st = kp.harvest(); ie::g_profiler = nullptr;
        std::sort(st.begin(), st.end(), [](const auto& a, const auto& b) { return a.total_ns > b.total_ns; });
        const double nt = double(std::max<size_t>(1, units));
        double sum = 0; for (const auto& x : st) sum += x.total_ms();
        std::printf("kernel profile over %zu %ss (ms per %s; GPU busy %.2f ms/%s of %.2f wall):\n", units, unit, unit, sum / nt, unit, wall_ms / nt);
        for (size_t i = 0; i < st.size() && i < 24; ++i)
            std::printf("  %-34s %9.3f ms/%s  %7.1f calls/%s  avg %.3f ms\n", st[i].name.c_str(), st[i].total_ms() / nt, unit, st[i].calls / nt, unit, st[i].avg_ms());
        for (size_t d = 0; d < kp.dev_spans.size(); ++d)
            std::printf("  device %zu: busy %.2f ms / span %.2f ms per %s\n", d, kp.dev_spans[d].busy_ns * 1e-6 / nt, kp.dev_spans[d].span_ns * 1e-6 / nt, unit);
    };
    opt.max_ctx = ctx; opt.max_tokens = std::max<uint32_t>(C, 8);
    if (!ranking_path.empty())
        if (auto e = ie::ds4_expert_priority_read_layers(ranking_path, m.config().n_routed_experts, m.config().n_layers, opt.ranking); !e.empty()) {
            std::fprintf(stderr, "ranking: %s\n", e.c_str()); return 1; }
    ie::Mimo26Forward fwd;
    const auto t_init = std::chrono::steady_clock::now();
    if (auto e = fwd.init(qs, m, opt); !e.empty()) { std::fprintf(stderr, "init: %s\n", e.c_str()); return 1; }
    std::printf("mimo26: %zu card(s), init %.1f s, VRAM", qs.size(), std::chrono::duration<double>(std::chrono::steady_clock::now() - t_init).count());
    for (size_t i = 0; i < qs.size(); ++i) std::printf(" %.2f GB", fwd.vram_bytes(i) / 1e9);
    std::printf(", pinned %.1f GB, static/pinned/stream per layer", fwd.pinned_bytes() / 1e9);
    for (size_t i = 0; i < qs.size(); ++i) std::printf(" %u/%u/%u", fwd.card_static(i), fwd.card_pinned(i), opt.stream_slots);
    std::printf(" (per card)\n");
    for (const auto& [pname, ptext] : prompts) {
    fwd.reset();
    const std::vector<int32_t> ids = tok.encode(ptext, special);
    if (prompts.size() > 1) std::printf("=== %s\n", pname.c_str());
    std::printf("prompt: %zu tokens [", ids.size());
    for (size_t i = 0; i < ids.size(); ++i) std::printf("%s%d", i ? ", " : "", ids[i]);
    std::printf("]\n");

    std::vector<float> logits;
    ie::KernelProfiler kpp;   // --kprof-prefill
    if (kprof_pf) { ie::g_profiler = &kpp; kpp.begin_step(); }
    const auto t_pf = std::chrono::steady_clock::now();
    for (uint32_t off = 0; off < ids.size(); off += C) {
        const uint32_t n = std::min<uint32_t>(C, uint32_t(ids.size()) - off);
        if (auto e = fwd.forward(ids.data() + off, n, off, logits, false); !e.empty()) { std::fprintf(stderr, "prefill at %u: %s\n", off, e.c_str()); return 1; }
    }
    const double pf_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_pf).count();
    std::printf("prefill: %.2f s (%.1f tok/s)\n", pf_s, ids.size() / pf_s);
    if (kprof_pf) report(kpp, (ids.size() + C - 1) / C, "chunk", pf_s * 1000);

    std::vector<int32_t> out;
    std::string text;
    ie::KernelProfiler kp;   // --kprof: every decode step's kernels, summed
    if (kprof) { ie::g_profiler = &kp; kp.begin_step(); }
    const auto t_dec = std::chrono::steady_clock::now();
    auto argmax = [](const float* lg, size_t V) { return int32_t(std::max_element(lg, lg + V) - lg); };
    const uint32_t V = m.config().vocab_size;
    auto put = [&](int32_t t) -> bool {   // false: an eos / pad id (recorded, not printed)
        out.push_back(t);
        if (t == m.config().eos_id || t == m.config().pad_id) return false;
        const std::string piece = tok.decode(std::vector<int32_t>{t});
        text += piece; std::fputs(piece.c_str(), stdout); std::fflush(stdout);
        return true;
    };
    uint32_t n_pass = 0, n_rows = 0, n_acc = 0, n_plain = 0;
    if (!lookup) {
        for (uint32_t k = 0; k < N; ++k) {
            const int32_t next = argmax(logits.data(), V);
            if (!put(next)) break;
            if (auto e = fwd.forward(&next, 1, fwd.n_pos(), logits, false); !e.empty()) { std::fprintf(stderr, "\ndecode step %u: %s\n", k, e.c_str()); return 1; }
        }
    } else {   // the engine's lookup loop, greedy (src/engine/mimo26_engine.cpp)
        ie::Ds41NgramIndex idx; idx.reset(ids);
        std::vector<int32_t> rows; std::vector<float> vlg;
        int32_t id = argmax(logits.data(), V);
        for (;;) {
            if (!put(id) || out.size() == N) break;
            idx.push(id);
            std::vector<int32_t> d = idx.draft(ie::Mimo26Forward::kDecodeRows - 1, 12);
            if (d.size() > N - out.size()) d.resize(N - out.size());
            const uint32_t pos = fwd.n_pos();
            if (d.empty()) {
                if (auto e = fwd.forward(&id, 1, pos, logits, false); !e.empty()) { std::fprintf(stderr, "\ndecode at %u: %s\n", pos, e.c_str()); return 1; }
                ++n_plain; id = argmax(logits.data(), V); continue;
            }
            rows.assign(1, id); rows.insert(rows.end(), d.begin(), d.end());
            if (auto e = fwd.forward(rows.data(), uint32_t(rows.size()), pos, vlg, true); !e.empty()) { std::fprintf(stderr, "\nverify at %u: %s\n", pos, e.c_str()); return 1; }
            uint32_t acc = 0; int32_t next = -1; bool ended = false, drop_last = false;
            for (uint32_t r = 0; r <= d.size(); ++r) {
                const int32_t a = argmax(vlg.data() + size_t(r) * V, V);
                if (r == d.size() || a != d[r]) { next = a; break; }
                ++acc;
                if (!put(a)) { ended = true; drop_last = true; break; }
                idx.push(a);
                if (out.size() == N) { ended = true; break; }
            }
            fwd.rewind(pos + 1 + acc - (drop_last ? 1u : 0u));
            ++n_pass; n_rows += uint32_t(rows.size()); n_acc += acc;
            if (ended) break;
            id = next;
        }
    }
    const double dec_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_dec).count();
    if (kprof) report(kp, out.size(), "token", dec_s * 1000);
    std::printf("\ngenerated: %zu tokens [", out.size());
    for (size_t i = 0; i < out.size(); ++i) std::printf("%s%d", i ? ", " : "", out[i]);
    std::printf("]\ndecode: %.1f ms/token (%.2f tok/s)\n", dec_s * 1000 / std::max<size_t>(1, out.size()), out.size() / std::max(dec_s, 1e-9));
    if (lookup) std::printf("lookup: %u verify passes (%u rows, %u drafts accepted), %u plain steps\n", n_pass, n_rows, n_acc, n_plain);
    double moe = 0, all = 0; uint32_t es = 0, ep = 0, em = 0;
    for (const auto& s : fwd.stats()) { moe += s.moe_ms; all += s.ms; es += s.experts_static; ep += s.experts_pinned; em += s.experts_mmap; }
    std::printf("last step: %.1f ms over the layers, %.1f ms in the expert tiers; experts static %u / pinned %u / mmap %u\n", all, moe, es, ep, em);
    std::fflush(stdout);
    }
    return 0;
}
