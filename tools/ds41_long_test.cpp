// tools/ds41_long_test.cpp -- V4.1 Phase 24 step 2, the deliverable's own gate: can a LONG prompt be prefilled in
// chunks, and what does decode cost afterwards. It walks the prompt up in stages and reports the length at which
// something refuses, with the refusal's own words -- so a wall is identified rather than inferred.
//   usage: ie-ds41-long-test <model> <golden_dir> [ranking] [target_tokens] [chunk] [decode_tokens]
#include "ie/deepseek41_forward.hpp"
#include "ie/expert_stream.hpp"
#include "ie/kernel_profiler.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {
double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
template <class T> std::vector<T> rd(const std::string& p, size_t n) { std::vector<T> v(n); std::ifstream f(p, std::ios::binary); if (!f) return {}; f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(T))); return f ? v : std::vector<T>{}; }
}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : std::string(std::getenv("HOME")) + "/models/DeepSeek-V4.1-Flash";
    const std::string gd  = argc > 2 ? argv[2] : dir + "/ie_golden";
    std::string rank_in = argc > 3 ? argv[3] : dir + "/ie_ranking_heldout.txt";
    if (!std::ifstream(rank_in).good()) rank_in.clear();
    const uint32_t target = argc > 4 ? uint32_t(std::atol(argv[4])) : 262144;
    const uint32_t chunk  = argc > 5 ? uint32_t(std::atol(argv[5])) : 2048;
    const uint32_t ndec   = argc > 6 ? uint32_t(std::atol(argv[6])) : 16;
    // [ctx] decouples the KV CAPACITY from the number of tokens actually prefilled. That is the control for
    // "does long context slow decode because of NC, or because the KV allocation displaces static experts from
    // VRAM": prefill the same tokens under two capacities and only the displacement differs.
    const uint32_t ctx    = argc > 7 ? uint32_t(std::atol(argv[7])) : 0;
    // [profile_out] writes a DECODE-PHASE ranking profiled at THIS context (Phase 25, docs/67). The profile is
    // reset after the prefill, so the counts hold decode selections only -- which is the whole point: a blended
    // histogram is a prefill histogram, and expert_stream.hpp measured that to be worse than index order.
    const std::string prof_out = argc > 8 ? argv[8] : "";

    ie::DeepSeek41Model m;
    if (auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    ie::Ds41EngramTables tb;
    if (auto e = tb.load(gd); !e.empty()) { std::fprintf(stderr, "tables: %s\n", e.c_str()); return 1; }
    // a long prompt built by repeating the golden 2,048-token block: the CONTENT does not matter to a capacity and
    // rate measurement, only the length and that every id is a real token this tokenizer emits
    const auto pp = rd<int32_t>(gd + "/pp_ids_2048.i32", 2048);
    if (pp.size() != 2048) { std::fprintf(stderr, "need %s/pp_ids_2048.i32\n", gd.c_str()); return 1; }
    std::vector<int32_t> ids; ids.reserve(size_t(target) + 64);
    while (ids.size() < size_t(target) + 64) ids.insert(ids.end(), pp.begin(), pp.end());

    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) { if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue; for (const auto& d : p.get_devices(sycl::info::device_type::gpu)) if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d); }
    if (devs.empty()) { std::fprintf(stderr, "no Arc GPU\n"); return 1; }
    // IE_QUEUE_PROFILING=1: queues with event profiling, so a chunk's per-kernel device time can be
    // harvested. Each submit costs ~0.4 us, so profiled rates are for SHAPE comparisons, never headlines.
    const bool qprof = std::getenv("IE_QUEUE_PROFILING") != nullptr;
    std::vector<std::unique_ptr<sycl::queue>> queues; std::vector<sycl::queue*> qs;
    for (const auto& d : devs) {
        sycl::property_list pl = qprof ? sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}
                                       : sycl::property_list{sycl::property::queue::in_order{}};
        queues.push_back(std::make_unique<sycl::queue>(sycl::context(d), d, pl)); qs.push_back(queues.back().get()); }
    std::vector<std::vector<uint32_t>> ranking;
    if (!rank_in.empty()) if (auto e = ie::ds4_expert_priority_read_layers(rank_in, m.config().n_routed_experts, m.config().n_layers, ranking); !e.empty()) { std::fprintf(stderr, "ranking: %s\n", e.c_str()); return 1; }

    ie::Ds41Forward fwd; ie::Ds41Forward::ResidentOptions opt;
    opt.max_tokens = ctx ? ctx : target + 64; opt.max_forward_tokens = chunk;
    const double t_load = now_s();
    if (auto e = fwd.init_resident(qs, m, tb, ranking, opt); !e.empty()) { std::fprintf(stderr, "init_resident: %s\n", e.c_str()); return 1; }
    std::printf("loaded ctx %u in %.1f s on %zu card(s), chunk %u\n", opt.max_tokens, now_s() - t_load, devs.size(), chunk);
    fwd.set_logits_last_only(true);

    // the stages: report a rate at each, so a wall is dated by length and the rate's shape with NC is visible
    std::vector<uint32_t> marks;
    for (uint32_t n = chunk; n < target; n *= 2) marks.push_back(n);
    marks.push_back(target);
    std::vector<float> lg; uint32_t done = 0; std::string wall;
    const double t_pf0 = now_s(); double t_prev = t_pf0; uint32_t prev_done = 0;
    // Phase 43: the whole target in one pipelined call, as the generator does (no per-stage marks); the serial walk when
    // IE_DS41_PIPE_PREFILL=0, or when a chunk profile / stage breakdown is asked for (those need one card at a time)
    const char* pv = std::getenv("IE_DS41_PIPE_PREFILL");
    if (!(pv && std::string(pv) == "0") && !qprof && !std::getenv("IE_DS41_STAGES")) {
        std::vector<std::pair<uint32_t, uint32_t>> chunks;
        for (uint32_t o = 0; o < target; o += chunk) chunks.push_back({o, std::min(chunk, target - o)});
        if (auto e = fwd.forward_pipelined(ids.data(), chunks, lg); !e.empty()) wall = e; else done = target;
        std::printf("  pipelined prefill: %zu chunks, %u tokens in %.2f s\n", chunks.size(), done, now_s() - t_pf0);
        marks.clear();
    }
    for (const uint32_t mark : marks) {
        while (done < mark) {
            const uint32_t t = std::min(chunk, mark - done);
            // THE MEASUREMENT: profile the LAST chunk of each stage. The same chunk shape runs at every
            // stage with only NC different, so differencing one kernel's ms across two stages isolates
            // exactly what scales with the latent cache -- which is the whole question.
            ie::KernelProfiler prof;
            const bool prof_this = qprof && (done + t >= mark);
            if (prof_this) { ie::g_profiler = &prof; prof.begin_step(); }
            const double tc0 = now_s();
            if (auto e = fwd.forward(ids.data() + done, t, done, lg); !e.empty()) { ie::g_profiler = nullptr; wall = "at pos0 " + std::to_string(done) + " T " + std::to_string(t) + ": " + e; break; }
            const double tc = now_s() - tc0;
            done += t;
            // IE_DS41_STAGES=1: the chunk's per-layer stages summed, so the wall the named kernels do not account for
            // is attributed (the MoE's fetch groups, the mmap fill's join, the dense stages, the untimed rest)
            if (std::getenv("IE_DS41_STAGES")) {
                double att = 0, fpre = 0, mc = 0, shd = 0, eng = 0, lay = 0, mprep = 0, grp = 0, join = 0, mmap = 0, mpack = 0, mmg = 0, tail = 0, bp = 0, bm = 0;
                double sta = 0, pin = 0, mm = 0, sh = 0;
                for (const auto& st : fwd.stats()) {
                    att += st.attn_ms; fpre += st.ffn_pre_ms; mc += st.moe_call_ms; shd += st.shared_ms; eng += st.engram_ms; lay += st.ms;
                    mprep += st.moe_prep_ms; grp += st.moe_groups_ms; join += st.moe_join_ms; mmap += st.moe_mmap_ms; mpack += st.moe_mmap_pack_ms;
                    mmg += st.moe_mmap_group_ms; tail += st.moe_tail_ms; bp += double(st.bytes_pinned); bm += double(st.bytes_mmap);
                    sta += st.experts_static; pin += st.experts_pinned; mm += st.experts_mmap; sh += st.experts_stream_hit;
                }
                std::printf("  [stages] chunk pos0 %u: wall %.0f = prep %.0f + layers %.0f + head %.0f + rest %.0f ms | layers: engram %.0f attn %.0f ffn_pre %.0f moe %.0f shared %.0f rest %.0f\n",
                            done - t, 1000.0 * tc, fwd.prep_ms(), lay, fwd.head_ms(), 1000.0 * tc - fwd.prep_ms() - lay - fwd.head_ms(),
                            eng, att, fpre, mc, shd, lay - eng - att - fpre - mc - shd);
                std::printf("  [stages]   moe %.0f = prep %.0f + groups %.0f + join %.0f + mmap_group %.0f + tail %.0f | mmap fill %.0f (read batches %.0f) | experts static %.0f pinned %.0f (stream hits %.0f) mmap %.0f | H2D pinned %.2f GiB, mmap %.2f GiB\n",
                            mc, mprep, grp, join, mmg, tail, mmap, mpack, sta, pin, sh, mm, bp / 1073741824.0, bm / 1073741824.0);
                // Phase 44: the same per layer range of each card (a pipelined chunk is bound by the slower card)
                for (size_t ci = 0; ci < qs.size(); ++ci) {
                    const auto info = fwd.card_info(uint32_t(ci));
                    double c_lay = 0, c_att = 0, c_moe = 0, c_fpre = 0, c_eng = 0, c_shd = 0;
                    for (uint32_t L = info.first_layer; L < info.first_layer + info.n_layers && L < fwd.stats().size(); ++L) {
                        const auto& st = fwd.stats()[L];
                        c_lay += st.ms; c_att += st.attn_ms; c_moe += st.moe_call_ms; c_fpre += st.ffn_pre_ms; c_eng += st.engram_ms; c_shd += st.shared_ms;
                    }
                    std::printf("  [stages]   card %zu (layers %u-%u): layers %.0f = engram %.0f + attn %.0f + ffn_pre %.0f + moe %.0f + shared %.0f + rest %.0f\n",
                                ci, info.first_layer, info.first_layer + info.n_layers - 1, c_lay, c_eng, c_att, c_fpre, c_moe, c_shd, c_lay - c_eng - c_att - c_fpre - c_moe - c_shd);
                }
                std::fflush(stdout);
            }
            if (prof_this) {
                for (auto* qq : qs) qq->wait();          // harvest() requires idle queues (kernel_profiler.hpp:21)
                auto st = prof.harvest(); ie::g_profiler = nullptr;
                std::sort(st.begin(), st.end(), [](const ie::KernelProfiler::Stat& a, const ie::KernelProfiler::Stat& b) { return a.total_ns > b.total_ns; });
                double tot = 0; for (const auto& r : st) tot += r.total_ms();
                std::printf("\n  --- one %u-token chunk at pos0 %u (NC up to %u), wall %.2f s, named kernels %.1f ms ---\n",
                            t, done - t, done, tc, tot);
                std::printf("  %-34s %7s %10s %9s %9s %9s\n", "kernel", "calls", "total ms", "avg us", "min us", "max us");
                for (size_t i = 0; i < st.size() && i < 14; ++i)
                    std::printf("  %-34s %7u %10.1f %9.1f %9.1f %9.1f\n", st[i].name.c_str(), st[i].calls,
                                st[i].total_ms(), st[i].avg_ms() * 1000.0, st[i].min_ms() * 1000.0, st[i].max_ms() * 1000.0);
                std::fflush(stdout);
            }
        }
        const double tn = now_s();
        // the STAGE rate is the one that matters: it shows the indexer's quadratic term arriving, which an
        // overall average hides
        std::printf("  prefilled %8u tokens  stage %7.1f tok/s  overall %7.1f tok/s  (stage %.2f s, %.1f s in, n_pos %u)\n",
                    done, tn > t_prev ? double(done - prev_done) / (tn - t_prev) : 0.0,
                    double(done) / (tn - t_pf0), tn - t_prev, tn - t_pf0, fwd.n_pos());
        std::fflush(stdout);
        t_prev = tn; prev_done = done;
        if (!wall.empty()) break;
    }
    const double t_pf = now_s() - t_pf0;
    if (!wall.empty()) {
        std::printf("\nWALL at %u tokens: %s\n", done, wall.c_str());
        std::printf("LONG TEST: the prefill stopped at %u of %u tokens\n", done, target);
        fwd.free_resident();
        return 2;
    }
    std::printf("\nprefill %u tokens in %.1f s = %.1f tok/s\n", done, t_pf, double(done) / t_pf);
    { uint64_t hsh = 1469598103934665603ull; for (float v : lg) { uint32_t u; std::memcpy(&u, &v, 4); hsh = (hsh ^ u) * 1099511628211ull; }
      std::printf("prefill logits fnv %016llx (%zu values, argmax %lld)\n", (unsigned long long)hsh, lg.size(), (long long)(std::max_element(lg.begin(), lg.end()) - lg.begin())); }

    // THE RESET: everything counted from here is decode at this context, and nothing else.
    if (!prof_out.empty()) fwd.reset_profile();

    // decode from the end of that prompt: the number the founder's bar is stated in
    double t_dec = 0; uint32_t got = 0;
    // DECODE IS PROFILED TOO, and at long context it is the number that decides usability. The steps run
    // at T = 1 with NC = the whole context, so any kernel whose cost is O(NC) per row shows here with no
    // row parallelism to hide it -- differencing this table between two targets names that term.
    ie::KernelProfiler dprof; const bool dprof_on = qprof && ndec > 1;
    // The per-layer STAGE breakdown and the expert counters, accumulated over the timed steps. This is what
    // attributes the time the named kernels do NOT account for: `moe_call` holds the per-layer expert fetch,
    // and its `join`/`tail` terms are the waits for the CPU miss leg.
    struct Acc { double att = 0, fpre = 0, moe = 0, grp = 0, join = 0, mmg = 0, tail = 0, shd = 0, lay = 0, cpu_ms = 0, prep = 0, head = 0, wall = 0;
                 double bp = 0, bm = 0, sta = 0, pin = 0, mm = 0, sh = 0, cpu = 0, cpu_work = 0; uint32_t n = 0; } A;
    std::vector<int32_t> dec_ids;
    for (uint32_t i = 0; i < ndec; ++i) {
        if (dprof_on && i == 1) { ie::g_profiler = &dprof; dprof.begin_step(); }   // step 0 is cold, excluded
        if (dprof_on && i >= 1) dprof.mark_token();
        const double t0 = now_s();
        int32_t nx = int32_t(std::max_element(lg.begin(), lg.end()) - lg.begin());
        dec_ids.push_back(nx);
        if (auto e = fwd.forward(&nx, 1, fwd.n_pos(), lg); !e.empty()) { ie::g_profiler = nullptr; std::printf("decode step %u: %s\n", i, e.c_str()); break; }
        const double w = now_s() - t0; t_dec += w; ++got;
        if (i >= 1) {
            ++A.n; A.wall += 1000.0 * w; A.prep += fwd.prep_ms(); A.head += fwd.head_ms();
            for (const auto& st : fwd.stats()) {
                A.att += st.attn_ms; A.fpre += st.ffn_pre_ms; A.moe += st.moe_call_ms; A.grp += st.moe_groups_ms;
                A.join += st.moe_join_ms; A.mmg += st.moe_mmap_group_ms; A.tail += st.moe_tail_ms;
                A.shd += st.shared_ms; A.lay += st.ms; A.cpu_ms += st.moe_cpu_ms; A.cpu_work += st.moe_cpu_work_ms;
                A.bp += double(st.bytes_pinned); A.bm += double(st.bytes_mmap);
                A.sta += st.experts_static; A.pin += st.experts_pinned; A.mm += st.experts_mmap;
                A.sh += st.experts_stream_hit; A.cpu += st.experts_cpu;
            }
        }
    }
    if (A.n) {
        const double d = double(A.n), NLTK = double(m.config().n_layers) * double(m.config().n_activated_experts);
        std::printf("\n  --- DECODE stage breakdown, %u steps at a %u-token context (ms/token) ---\n", A.n, done);
        std::printf("  wall %.1f = prep %.1f + layers %.1f + head %.1f   [layers: attn %.1f, ffn_pre %.1f, moe %.1f, shared %.1f]\n",
                    A.wall / d, A.prep / d, A.lay / d, A.head / d, A.att / d, A.fpre / d, A.moe / d, A.shd / d);
        std::printf("  moe %.1f = groups %.1f + join %.1f + mmap_group %.1f + tail %.1f\n",
                    A.moe / d, A.grp / d, A.join / d, A.mmg / d, A.tail / d);
        // The tail is the CPU miss leg being the critical path: tail ~= (cpu leg) - (groups). Separating the
        // leg's own COMPUTE from its span says whether the CPU got slower or merely started later, which is the
        // difference between a bandwidth problem and a scheduling one.
        std::printf("  cpu leg %.1f ms span, %.1f ms compute, %.1f experts -> %.3f ms/expert compute, %.3f ms/expert span\n",
                    A.cpu_ms / d, A.cpu_work / d, A.cpu / d,
                    A.cpu > 0 ? A.cpu_work / A.cpu : 0.0, A.cpu > 0 ? A.cpu_ms / A.cpu : 0.0);
        std::printf("  experts/token static %.1f pinned %.1f mmap %.1f cpu %.1f of %.0f, stream hits %.1f -> hit rate %.1f%%\n",
                    A.sta / d, A.pin / d, A.mm / d, A.cpu / d, NLTK, A.sh / d, 100.0 * (A.sta + A.sh) / d / NLTK);
        std::printf("  bytes/token  pinned %.1f MiB   disk %.1f MiB\n", A.bp / d / 1048576.0, A.bm / d / 1048576.0);
        std::fflush(stdout);
    }
    if (dprof_on && got > 1) {
        for (auto* qq : qs) qq->wait();
        auto st = dprof.harvest(); ie::g_profiler = nullptr;
        std::sort(st.begin(), st.end(), [](const ie::KernelProfiler::Stat& a, const ie::KernelProfiler::Stat& b) { return a.total_ns > b.total_ns; });
        const double per = double(got - 1); double tot = 0; for (const auto& r : st) tot += r.total_ms();
        std::printf("\n  --- DECODE, %u steps at a %u-token context: named kernels %.1f ms/token of %.1f ms wall ---\n",
                    uint32_t(per), done, tot / per, 1000.0 * t_dec / got);
        std::printf("  %-34s %7s %10s %9s %9s %9s\n", "kernel", "calls/tok", "ms/token", "avg us", "min us", "max us");
        for (size_t i = 0; i < st.size() && i < 16; ++i)
            std::printf("  %-34s %7.0f %10.2f %9.1f %9.1f %9.1f\n", st[i].name.c_str(), st[i].calls / per,
                        st[i].total_ms() / per, st[i].avg_ms() * 1000.0, st[i].min_ms() * 1000.0, st[i].max_ms() * 1000.0);
        std::fflush(stdout);
    }
    { uint64_t hsh = 1469598103934665603ull; for (int32_t v : dec_ids) hsh = (hsh ^ uint32_t(v)) * 1099511628211ull;
      std::printf("decode ids fnv %016llx over %zu tokens, first:", (unsigned long long)hsh, dec_ids.size());
      for (size_t i = 0; i < dec_ids.size() && i < 12; ++i) std::printf(" %d", dec_ids[i]); std::printf("\n"); }
    if (got) std::printf("decode %u tokens: %.1f ms/token = %.2f tok/s at a %u-token context\n", got, 1000.0 * t_dec / got, double(got) / t_dec, done);
    if (!prof_out.empty()) {
        // The header names the PHASE and the CONTEXT, because the file format records neither and that is
        // exactly how a prefill ranking came to be used for decode residency (docs/67).
        const uint64_t sel = uint64_t(got) * m.config().n_layers * m.config().n_activated_experts;
        char note[512];
        std::snprintf(note, sizeof note,
                      "V4.1 DECODE-PHASE profile at a %u-token context: %u decode steps, %llu selections, "
                      "%.0f per layer (%s)", done, got, (unsigned long long)sel,
                      double(got) * m.config().n_activated_experts,
                      got >= 256 ? "sufficient per docs/67" : "UNDER-SAMPLED: docs/67 asks for >= 256 steps");
        if (auto e = fwd.write_profile(prof_out, note); !e.empty()) std::fprintf(stderr, "write_profile: %s\n", e.c_str());
        else std::printf("wrote decode-phase ranking for ctx %u -> %s (%u steps, %.0f selections/layer%s)\n",
                         done, prof_out.c_str(), got, double(got) * m.config().n_activated_experts,
                         got >= 256 ? "" : "  [UNDER-SAMPLED]");
    }
    fwd.free_resident();
    std::printf("LONG TEST: %s\n", got == ndec ? "PASS" : "FAIL");
    return got == ndec ? 0 : 1;
}
