// tools/ds41_decode_test.cpp — V4.1 Phase 9: decode on the resident runtime, against the
// reference's own generate loop (golden_decode.py -> <golden>/decode2). Criteria in
// docs/deepseek41/19; design in docs/deepseek41/24.
//   1. the state after the prefill (window ring / latents / index keys of layers 0, 2, 20)
//      equals the reference's, and the prompt's engram hashes equal as integers
//   2. decode step 0 (pos 12) matches layer by layer, engram outs included
//   3. step 0's token and top-5 set match
//   4. N >= 4 greedy steps reproduce the golden's tokens (soft only under the margin rule)
//   5. negative control: step 0 with the RoPE position off by one must FAIL criterion 2
//   6. every step's hashes equal the reference's as integers
//   8. per-step device memory is bounded: free VRAM after the steps == before (within slack)
//   (7, the forward test, runs separately on the same build)
//   usage: ie-ds41-decode-test <model> <golden_dir> <golden_dir>/decode2 [ranking_in]
#include "ie/deepseek41_forward.hpp"
#include "ie/expert_stream.hpp"
#include "ie/kernel_profiler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "../third_party/nlohmann/json.hpp"

namespace {
int g_fail = 0;
void check(bool ok, const std::string& w, const std::string& d = "") {
    std::printf("%s%s%s\n", ok ? "[ ok ] " : "[FAIL] ", w.c_str(), d.empty() ? "" : ("  (" + d + ")").c_str()); if (!ok) ++g_fail; }
template <class T> std::vector<T> rd(const std::string& p, size_t n) {
    std::vector<T> v(n); std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p.c_str()); std::exit(1); }
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(T)));
    if (size_t(f.gcount()) != n * sizeof(T)) { std::fprintf(stderr, "%s: short (%zu of %zu bytes)\n", p.c_str(), size_t(f.gcount()), n * sizeof(T)); std::exit(1); }
    return v; }
bool exists(const std::string& p) { return std::ifstream(p).good(); }
double rel(const float* a, const float* b, size_t n) {
    double m = 0, s = 0; for (size_t i = 0; i < n; ++i) { m = std::max(m, std::fabs(double(a[i]) - double(b[i]))); s = std::max(s, std::fabs(double(b[i]))); }
    return s > 0 ? m / s : 0; }
double rel(const std::vector<float>& a, const std::vector<float>& b) { return rel(a.data(), b.data(), a.size()); }
std::string f3(double v) { char b[32]; std::snprintf(b, sizeof b, "%.3e", v); return b; }
std::vector<int32_t> top5(const float* row, uint32_t V) {
    std::vector<int32_t> ids(V); std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + 5, ids.end(), [&](int32_t a, int32_t b) { return row[a] > row[b]; });
    ids.resize(5); return ids;
}
}  // namespace

int main(int argc, char** argv) {
    // The near-tie and chunk-vs-step bars below were derived on the int-dot W4A8 prefill expert route. Since 2026-09-17 the
    // engine's default is the XMX W4A16 route (docs/deepseek41/82, 93), which this test measures at 26/28 and 14/15 by
    // design (docs/82); it therefore pins the route it gates unless the caller names one (IE_DS41_PREFILL_XMX=1 to run it).
    setenv("IE_DS41_PREFILL_XMX", "0", 0);
    const std::string dir = argc > 1 ? argv[1] : std::string(std::getenv("HOME")) + "/models/DeepSeek-V4.1-Flash";
    const std::string gd  = argc > 2 ? argv[2] : "/tmp/ds41_golden";
    const std::string dd  = argc > 3 ? argv[3] : gd + "/decode2";
    const std::string rank_in = argc > 4 ? argv[4] : "";

    nlohmann::json meta; { std::ifstream f(dd + "/d_meta.json"); if (!f) { std::fprintf(stderr, "no %s/d_meta.json\n", dd.c_str()); return 1; } f >> meta; }
    const std::vector<int32_t> prompt = meta["prompt_ids"].get<std::vector<int32_t>>();
    const std::vector<int32_t> all_ids = meta["all_ids"].get<std::vector<int32_t>>();
    const uint32_t T0 = uint32_t(prompt.size()), NL = meta["n_layers"], H = meta["dim"], HC = meta["hc_mult"], V = meta["vocab"];
    const uint32_t N_DEC = meta["n_decode"];
    const auto& steps = meta["steps"];
    std::printf("prompt %s -> %u tokens; golden continues %u steps: ", meta["prompt"].get<std::string>().c_str(), T0, N_DEC);
    for (uint32_t k = 0; k < N_DEC; ++k) std::printf("%s", steps[k]["text"].get<std::string>().c_str());
    std::printf("\n");

    ie::DeepSeek41Model m;
    if (auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    ie::Ds41EngramTables tb;
    if (auto e = tb.load(gd); !e.empty()) { std::fprintf(stderr, "tables: %s\n", e.c_str()); return 1; }
    const auto& c = m.config();
    const uint32_t E = c.n_routed_experts, HD = c.head_dim, IHD = c.index_head_dim, WIN = c.window_size, ENC = tb.n_hash_cols;

    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d);
    }
    if (devs.empty()) { std::fprintf(stderr, "no Arc GPU\n"); return 1; }
    std::vector<std::unique_ptr<sycl::queue>> queues; std::vector<sycl::queue*> qs;
    // IE_QUEUE_PROFILING=1: queues with event profiling, so the Phase 11 bench can harvest the
    // per-kernel device time of the timed steps through ie::KernelProfiler (~0.4 us per submit)
    const bool qprof = std::getenv("IE_QUEUE_PROFILING") != nullptr;
    for (const auto& d : devs) {
        sycl::property_list pl = qprof ? sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}
                                       : sycl::property_list{sycl::property::queue::in_order{}};
        queues.push_back(std::make_unique<sycl::queue>(sycl::context(d), d, pl)); qs.push_back(queues.back().get());
    }
    auto free_vram = [&](uint32_t ci) -> uint64_t { return devs[ci].get_info<sycl::ext::intel::info::device::free_memory>(); };

    std::vector<std::vector<uint32_t>> ranking;
    if (!rank_in.empty()) { if (auto e = ie::ds4_expert_priority_read_layers(rank_in, E, NL, ranking); !e.empty()) { std::fprintf(stderr, "ranking: %s\n", e.c_str()); return 1; } }
    ie::Ds41Forward fwd; ie::Ds41Forward::ResidentOptions opt;
    opt.max_tokens = 2048 + 64;            // Phase 11: 36 decode steps past the replayed 2048-token prompt
    const auto ti = std::chrono::steady_clock::now();
    if (auto e = fwd.init_resident(qs, m, tb, ranking, opt); !e.empty()) { std::fprintf(stderr, "init_resident: %s\n", e.c_str()); return 1; }
    std::printf("resident in %.0f s on %u card(s)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - ti).count(), uint32_t(devs.size()));

    // ---- the prefill's routing, when the golden dumped it: the state and the forced decode step are
    // judged on a prefill with the golden's experts, free of Phase 7's own-router near-tie flip at
    // layer 7 (which would otherwise carry into every ring row and latent of the later layers)
    const uint32_t TK = c.n_activated_experts;
    const bool have_proute = exists(dd + "/p_route_idx_0.i32");
    std::vector<std::vector<int32_t>> p_ridx; std::vector<std::vector<float>> p_rw, p_gap;
    if (have_proute) for (uint32_t L = 0; L < NL; ++L) {
        p_ridx.push_back(rd<int32_t>(dd + "/p_route_idx_" + std::to_string(L) + ".i32", size_t(T0) * TK));
        p_rw.push_back(rd<float>(dd + "/p_route_w_" + std::to_string(L) + ".f32", size_t(T0) * TK));
        p_gap.push_back(rd<float>(dd + "/p_route_gap_" + std::to_string(L) + ".f32", T0));
    }
    // IE_DS41_DUMP_DIR: the PREFILL's per-layer outputs (P<L>.f32) and MoE outputs (PM<L>.f32) too, for an A/B between builds
    const char* pdump = std::getenv("IE_DS41_DUMP_DIR");
    ie::Ds41Forward::Probe pprobe = pdump ? ie::Ds41Forward::Probe([&](const char* name, uint32_t L, const float* d, size_t n, sycl::queue& pq) {
        const std::string nm(name); const char* pre = nm == "layer" ? "/P" : nm == "moe" ? "/PM" : nm == "xfn" ? "/PX" : nm == "moe_own" ? "/PO" : nm == "moe_remote" ? "/PR" : nullptr;
        if (!pre) return;
        std::vector<float> got(n); pq.memcpy(got.data(), d, n * 4).wait();
        std::ofstream f(std::string(pdump) + pre + std::to_string(L) + ".f32", std::ios::binary);
        f.write(reinterpret_cast<const char*>(got.data()), std::streamsize(n * 4));
    }) : ie::Ds41Forward::Probe{};
    auto prefill = [&](bool forced, std::vector<float>& lg) -> std::string {
        fwd.reset_state();
        return fwd.forward(prompt.data(), T0, 0, lg, pprobe, forced ? &p_ridx : nullptr, forced ? &p_rw : nullptr, forced, forced ? &p_gap : nullptr);
    };
    std::vector<float> logits;
    // IE_DS41_BENCH_ONLY=1|short|long: skip the correctness criteria and run only the Phase 11
    // measurement (for a kernel profile of the decode steps alone, e.g. under unitrace -d)
    const char* bo = std::getenv("IE_DS41_BENCH_ONLY");
    const bool bench_short = !bo || std::string(bo) != "long", bench_long = !bo || std::string(bo) != "short";
    if (!bo) {   // ---- the correctness criteria (Phase 9), unchanged; not re-indented ----
    if (auto e = prefill(have_proute, logits); !e.empty()) { std::fprintf(stderr, "prefill: %s\n", e.c_str()); return 1; }
    check(fwd.n_pos() == T0, "prefill consumed the prompt", std::to_string(fwd.n_pos()) + " positions");
    std::printf("\n=== criterion 1: state after the prefill (%s routing) vs the reference's module buffers ===\n", have_proute ? "FORCED" : "own");
    const bool have_state = exists(dd + "/d_pre_window_0.f32");
    if (!have_state) std::printf("       (golden has no d_pre_* dumps -- criterion 1 NOT judged in this run)\n");
    for (uint32_t L : {0u, 2u, 20u}) {
        if (!have_state) break;
        std::vector<float> win, comp, idxk; uint32_t nc = 0;
        if (auto e = fwd.read_state(L, win, comp, idxk, nc); !e.empty()) { std::fprintf(stderr, "read_state: %s\n", e.c_str()); return 1; }
        const auto gw = rd<float>(dd + "/d_pre_window_" + std::to_string(L) + ".f32", size_t(WIN) * HD);
        const uint32_t filled = std::min(T0, WIN);          // slots [0, filled) hold positions; the rest were never written
        const double ew = rel(win.data(), gw.data(), size_t(filled) * HD);
        const double sbar = L < 10 ? 3e-3 : 1.2e-2;      // criterion 1's 3e-3; deeper layers sit on the Q8 grouped path's floor (Phase 8 card-0 bar)
        check(ew < sbar, "layer " + std::to_string(L) + " window ring equals the reference's (filled slots, bar " + f3(sbar) + ")", f3(ew) + " over " + std::to_string(filled) + " slots");
        if (exists(dd + "/d_pre_comp_" + std::to_string(L) + ".f32")) {
            const uint32_t r = m.layers()[L].kind.compress_ratio, want_nc = T0 / std::max(1u, r);
            check(nc == want_nc, "layer " + std::to_string(L) + " holds " + std::to_string(want_nc) + " latents", std::to_string(nc));
            const auto gc = rd<float>(dd + "/d_pre_comp_" + std::to_string(L) + ".f32", size_t(want_nc) * HD);
            const double ec = rel(comp.data(), gc.data(), size_t(std::min(nc, want_nc)) * HD);
            check(ec < sbar, "layer " + std::to_string(L) + " compressed latents equal the reference's", f3(ec));
            if (exists(dd + "/d_pre_idxk_" + std::to_string(L) + ".f32")) {
                const auto gk = rd<float>(dd + "/d_pre_idxk_" + std::to_string(L) + ".f32", size_t(want_nc) * IHD);
                const double ek = rel(idxk.data(), gk.data(), size_t(std::min(nc, want_nc)) * IHD);
                check(ek < sbar, "layer " + std::to_string(L) + " index keys equal the reference's", f3(ek));
            }
        }
    }
    // criterion 6 for the prompt: the hashes the engine used, as integers
    if (!exists(dd + "/p_hashes.i64")) std::printf("       (golden has no p_hashes -- prompt hashes NOT judged in this run)\n");
    else {
        const auto gh = rd<int64_t>(dd + "/p_hashes.i64", size_t(T0) * tb.layer_ids.size() * ENC);
        const auto& eh = fwd.last_hashes();     // [engram layer][T * ENC]
        uint64_t bad = 0;
        for (size_t li = 0; li < eh.size(); ++li)
            for (uint32_t t = 0; t < T0; ++t) for (uint32_t k = 0; k < ENC; ++k)
                if (eh[li][size_t(t) * ENC + k] != gh[(size_t(t) * tb.layer_ids.size() + li) * ENC + k]) ++bad;
        check(bad == 0, "prompt engram hashes equal the reference's, as integers", std::to_string(bad) + " differ");
    }

    // ---- decode steps ------------------------------------------------------------------------
    const uint64_t vram0 = free_vram(0), vram1 = devs.size() > 1 ? free_vram(1) : 0;
    std::vector<double> layer_err(NL, -1), engram_err(NL, -1);
    auto probe = [&](const char* name, uint32_t L, const float* d, size_t n, sycl::queue& pq) {
        std::vector<float> got(n); pq.memcpy(got.data(), d, n * 4).wait();
        if (std::string(name) == "layer")  layer_err[L]  = rel(got, rd<float>(dd + "/d_layer_out_" + std::to_string(L) + ".f32", n));
        if (std::string(name) == "engram") engram_err[L] = rel(got, rd<float>(dd + "/d_engram_out_" + std::to_string(L) + ".f32", n));
    };
    std::printf("\n=== decode: step 0 at pos %u, layer by layer ===\n", T0);
    int32_t fed = all_ids[T0];                                   // the golden's first generated token (" Berlin")
    // the golden's routing at this step, when dumped: own-router divergences are judged to the first
    // flip (a near-tie), and a second pass with the routing FORCED checks the arithmetic at every layer
    const bool have_route = exists(dd + "/d_route_idx_0.i32");
    std::vector<std::vector<int32_t>> g_ridx; std::vector<std::vector<float>> g_rw, g_gap;
    if (have_route) for (uint32_t L = 0; L < NL; ++L) {
        g_ridx.push_back(rd<int32_t>(dd + "/d_route_idx_" + std::to_string(L) + ".i32", TK));
        g_rw.push_back(rd<float>(dd + "/d_route_w_" + std::to_string(L) + ".f32", TK));
        g_gap.push_back(rd<float>(dd + "/d_route_gap_" + std::to_string(L) + ".f32", 1));
    } else std::printf("       (golden has no d_route_* dumps -- own-router judged over all 40 layers, no forced pass)\n");
    auto bar_of = [](uint32_t L) { return L < 20 ? 1.2e-2 : 3.5e-2; };
    if (have_route && have_proute) {
        // the decode ARITHMETIC: the golden's experts at the prefill and at this step, judged at all 40 layers
        std::vector<double> f_err(NL, -1), f_eng(NL, -1);
        const char* dump = std::getenv("IE_DS41_DUMP_DIR");          // Phase 11 isolation: the forced step's layer outputs
        auto fprobe = [&](const char* name, uint32_t L, const float* d, size_t n, sycl::queue& pq) {
            std::vector<float> got(n); pq.memcpy(got.data(), d, n * 4).wait();
            if (dump && std::string(name) == "moe") { std::ofstream f(std::string(dump) + "/M" + std::to_string(L) + ".f32", std::ios::binary); f.write(reinterpret_cast<const char*>(got.data()), std::streamsize(n * 4)); }
            if (dump && std::string(name) == "layer") { std::ofstream f(std::string(dump) + "/L" + std::to_string(L) + ".f32", std::ios::binary); f.write(reinterpret_cast<const char*>(got.data()), std::streamsize(n * 4)); }
            if (std::string(name) == "layer")  f_err[L] = rel(got, rd<float>(dd + "/d_layer_out_" + std::to_string(L) + ".f32", n));
            if (std::string(name) == "engram") f_eng[L] = rel(got, rd<float>(dd + "/d_engram_out_" + std::to_string(L) + ".f32", n));
        };
        std::vector<float> f_logits;
        if (auto e = fwd.forward(&fed, 1, T0, f_logits, fprobe, &g_ridx, &g_rw, true, &g_gap); !e.empty()) { std::fprintf(stderr, "decode step 0 (forced): %s\n", e.c_str()); return 1; }
        uint32_t bad = 0; double w0 = 0, w1 = 0;
        for (uint32_t L = 0; L < NL; ++L) { if (f_err[L] < 0 || f_err[L] >= bar_of(L)) ++bad; (L < 20 ? w0 : w1) = std::max(L < 20 ? w0 : w1, f_err[L]); }
        std::printf("       per-layer stream error, prefill and step FORCED to the golden's routing:");
        for (uint32_t L = 0; L < NL; ++L) std::printf("%s%2u:%.2e", L % 8 ? "  " : "\n         ", L, f_err[L]);
        std::printf("\n");
        check(bad == 0, "step 0, FORCED routing: every layer within its bar (1.2e-2 to layer 19, 3.5e-2 after)", "card 0 worst " + f3(w0) + ", card 1 worst " + f3(w1) + ", " + std::to_string(bad) + " over");
        for (uint32_t L : {1u, 14u}) check(f_eng[L] >= 0 && f_eng[L] < 1.2e-2, "step 0, FORCED: engram output at layer " + std::to_string(L) + " within 1.2e-2", f3(f_eng[L]));
        const auto g = rd<float>(dd + "/d_logits_step1.f32", V);
        const double lr = rel(f_logits.data(), g.data(), V);
        check(lr < 5e-2, "step 0, FORCED routing: logits within 5e-2 of the golden (the Q8 path's regression bar)", "rel " + f3(lr));
    } else std::printf("       (no forced pass: the golden lacks p_route_*/d_route_* dumps)\n");
    // the engine's OWN router from here: a fresh own-router prefill, then step 0 judged to the first flip
    if (auto e = prefill(false, logits); !e.empty()) { std::fprintf(stderr, "prefill (own): %s\n", e.c_str()); return 1; }
    const auto s0 = std::chrono::steady_clock::now();
    if (auto e = fwd.forward(&fed, 1, T0, logits, probe, have_route ? &g_ridx : nullptr, have_route ? &g_rw : nullptr, false, have_route ? &g_gap : nullptr); !e.empty()) { std::fprintf(stderr, "decode step 0: %s\n", e.c_str()); return 1; }
    const double step0_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count();
    // Phase 7's per-layer bars on the streaming path were 5e-3 to layer 19 and 1e-2 after; the
    // resident path runs the Q8 grouped MoE, whose floor is ~15x the fp16 path's (docs/22), so
    // the bars here are the Phase 8 forced-pass regression bars per card: 1.2e-2 / 3.5e-2.
    int32_t first_flip = -1;
    if (have_route) for (uint32_t L = 0; L < NL; ++L) if (fwd.stats()[L].routing_diff) { first_flip = int32_t(L); break; }
    std::printf("       per-layer stream error vs d_layer_out (%.2f s), own router%s:", step0_s,
                first_flip >= 0 ? (" -- first routing flip at layer " + std::to_string(first_flip)).c_str() : "");
    for (uint32_t L = 0; L < NL; ++L) std::printf("%s%2u:%.2e", L % 8 ? "  " : "\n         ", L, layer_err[L]);
    std::printf("\n");
    {
        uint32_t bad = 0; double worst = 0;
        for (uint32_t L = 0; L < NL; ++L) {
            if (first_flip >= 0 && int32_t(L) >= first_flip) break;
            if (layer_err[L] < 0 || layer_err[L] >= bar_of(L)) ++bad;
            worst = std::max(worst, layer_err[L]);
        }
        check(bad == 0, "step 0, own router: every layer up to the first routing flip within its bar (1.2e-2 to layer 19, 3.5e-2 after)",
              "worst " + f3(worst) + ", " + std::to_string(bad) + " over, judged " + std::to_string(first_flip >= 0 ? first_flip : int32_t(NL)) + " layers");
        if (first_flip >= 0) check(fwd.stats()[first_flip].flip_gap_max < 3e-3, "step 0: the first routing flip is at a near-tie",
                                   "layer " + std::to_string(first_flip) + ", golden 6th-vs-7th margin " + f3(fwd.stats()[first_flip].flip_gap_max));
        if (first_flip < 0 || first_flip > 1) check(engram_err[1] >= 0 && engram_err[1] < 1.2e-2, "step 0: engram output at layer 1 within 1.2e-2", f3(engram_err[1]));
        if (first_flip < 0 || first_flip > 14) check(engram_err[14] >= 0 && engram_err[14] < 1.2e-2, "step 0: engram output at layer 14 within 1.2e-2", f3(engram_err[14]));
    }
    {   // criterion 3: the step's token and top-5 set, from the engine's own router
        const auto g = rd<float>(dd + "/d_logits_step1.f32", V);     // the reference's logits AFTER feeding step 0's token
        const double lr = rel(logits.data(), g.data(), V);
        const auto e5 = top5(logits.data(), V), g5 = top5(g.data(), V);
        std::vector<int32_t> es(e5), gs(g5); std::sort(es.begin(), es.end()); std::sort(gs.begin(), gs.end());
        const int32_t want = steps[1]["token"].get<int32_t>();
        std::printf("       engine top-5: %d %d %d %d %d   golden: %d %d %d %d %d   logits rel %.3e (own router, after the flip at layer %d)\n",
                    e5[0], e5[1], e5[2], e5[3], e5[4], g5[0], g5[1], g5[2], g5[3], g5[4], lr, first_flip);
        check(e5[0] == want, "step 0: next token is the golden's " + steps[1]["text"].get<std::string>(), "engine " + std::to_string(e5[0]) + " vs " + std::to_string(want));
        check(es == gs, "step 0: top-5 SET equals the golden's");
    }
    // criterion 6 at step 0
    if (!exists(dd + "/d_hashes.i64")) std::printf("       (golden has no d_hashes -- step-0 hashes NOT judged in this run)\n");
    else {
        const auto gh = rd<int64_t>(dd + "/d_hashes.i64", tb.layer_ids.size() * ENC);
        const auto& eh = fwd.last_hashes(); uint64_t hb = 0;
        for (size_t li = 0; li < eh.size(); ++li) for (uint32_t k = 0; k < ENC; ++k) if (eh[li][k] != gh[li * ENC + k]) ++hb;
        check(hb == 0, "step 0: the new token's engram hashes equal the reference's (the 4-gram reaches into the prompt)", std::to_string(hb) + " differ");
    }
    // criterion 4: greedy continuation from the engine's own argmax, steps 1..N-1
    std::printf("\n=== criterion 4: greedy steps ===\n");
    std::vector<int32_t> engine_ids(all_ids.begin(), all_ids.begin() + T0 + 1);
    uint32_t diverged_at = 0; bool diverged = false;
    for (uint32_t k = 1; k < N_DEC; ++k) {
        const int32_t nxt = int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
        engine_ids.push_back(nxt);
        const int32_t want = all_ids[T0 + k];
        const auto g = rd<float>(dd + "/d_logits_step" + std::to_string(k) + ".f32", V);    // logits that produced all_ids[T0+k]
        const auto g5 = top5(g.data(), V);
        const double margin = g[g5[0]] - g[g5[1]], lr = rel(logits.data(), g.data(), V);
        std::printf("       step %u: engine chose %d (golden %d), golden top-1/top-2 margin %.3f, logits rel %.3e\n", k, nxt, want, margin, lr);
        if (nxt != want && !diverged) { diverged = true; diverged_at = k; }
        // hashes for this step, if the golden dumped them
        if (auto e = fwd.forward(&want, 1, T0 + k, logits); !e.empty()) { std::fprintf(stderr, "decode step %u: %s\n", k, e.c_str()); return 1; }
        const std::string hp = dd + "/s" + std::to_string(k) + "_hashes.i64";
        if (exists(hp)) {
            const auto gh = rd<int64_t>(hp, tb.layer_ids.size() * ENC); const auto& eh = fwd.last_hashes(); uint64_t hb = 0;
            for (size_t li = 0; li < eh.size(); ++li) for (uint32_t kk = 0; kk < ENC; ++kk) if (eh[li][kk] != gh[li * ENC + kk]) ++hb;
            check(hb == 0, "step " + std::to_string(k) + ": engram hashes equal the reference's", std::to_string(hb) + " differ");
        }
    }
    {   // the last step's logits -> the golden's final token
        const int32_t nxt = int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
        engine_ids.push_back(nxt);
        const int32_t want = all_ids[T0 + N_DEC];
        std::printf("       step %u: engine chose %d (golden %d)\n", N_DEC, nxt, want);
        if (nxt != want && !diverged) { diverged = true; diverged_at = N_DEC; }
    }
    if (!diverged) check(true, "greedy: all " + std::to_string(N_DEC) + " continuation tokens equal the golden's");
    else {
        // the margin rule: a divergence is soft only if the golden's own margin at that step is under the accumulated logit error
        const auto g = rd<float>(dd + "/d_logits_step" + std::to_string(diverged_at) + ".f32", V);
        const auto g5 = top5(g.data(), V); const double margin = g[g5[0]] - g[g5[1]];
        std::printf("       diverged at step %u: golden margin %.3f\n", diverged_at, margin);
        check(false, "greedy: continuation equals the golden's (diverged at step " + std::to_string(diverged_at) + ", margin " + f3(margin) + ")");
    }
    std::printf("       engine: ");
    for (size_t i = T0; i < engine_ids.size(); ++i) std::printf("%d ", engine_ids[i]);
    std::printf("| golden: "); for (size_t i = T0; i < all_ids.size(); ++i) std::printf("%d ", all_ids[i]); std::printf("\n");

    // criterion 8: what the steps left on the device
    {
        const int64_t d0 = int64_t(vram0) - int64_t(free_vram(0)), d1 = devs.size() > 1 ? int64_t(vram1) - int64_t(free_vram(1)) : 0;
        check(d0 < int64_t(64ull << 20) && d1 < int64_t(64ull << 20), "per-step device memory is bounded: free VRAM after the steps within 64 MiB of before",
              std::to_string(d0 >> 20) + " / " + std::to_string(d1 >> 20) + " MiB held");
    }

    // criterion 5: the negative control -- redo the prefill and step 0 with the RoPE position off by one
    std::printf("\n=== criterion 5: negative control (step 0 with the position off by one) ===\n");
    if (auto e = prefill(false, logits); !e.empty()) { std::fprintf(stderr, "prefill (control): %s\n", e.c_str()); return 1; }
    std::fill(layer_err.begin(), layer_err.end(), -1.0);
    fwd.set_rope_offset_diagnostic(1);
    if (auto e = fwd.forward(&fed, 1, T0, logits, probe); !e.empty()) { std::fprintf(stderr, "decode (control): %s\n", e.c_str()); return 1; }
    fwd.set_rope_offset_diagnostic(0);
    uint32_t over = 0; double worst = 0;
    for (uint32_t L = 0; L < NL; ++L) { const double bar = L < 20 ? 1.2e-2 : 3.5e-2; if (layer_err[L] >= bar) ++over; worst = std::max(worst, layer_err[L]); }
    check(over > 0, "with the position off by one, criterion 2 FAILS (position bookkeeping is load-bearing)", std::to_string(over) + " layers over their bar, worst " + f3(worst));
    }   // ---- end of the correctness criteria ----

    // ---- Phase 11, step 1 (docs/deepseek41/29): the decode step's cost, measured ---------------
    // Steps 0-3 with the per-stage breakdown (the resident test's stage timers, at T = 1), then
    // 32 more steps as ms/token and tok/s with the bytes each token moved per tier; after the
    // golden prompt, and after the replayed 2048-token pp text (full 128-row rings, 2048 latents
    // in the ratio-1 caches). IE_DS41_LAUNCH_MARKS=1 prints "@@layer L" markers on stderr after
    // every layer of step 0 after the golden prompt, for the launch count per layer under
    // SYCL_UR_TRACE=2 (tools/ds41_reference/ur_launch_count.py).
    std::printf("\n=== Phase 11: the decode step, measured (docs/deepseek41/29) ===\n");
    fwd.set_logits_last_only(true);
    const bool marks = std::getenv("IE_DS41_LAUNCH_MARKS") != nullptr;
    const uint32_t N_BREAK = 4, N_RATE = 32;
    auto bench = [&](const std::string& label, uint32_t pos_start, bool mark_step0) -> bool {
        std::printf("       %s\n", label.c_str());
        std::printf("       %4s | %5s %6s %6s %7s %7s (%5s %5s %6s %5s %6s %5s) %6s %5s %5s %5s %7s | %3s/%3s/%3s | %8s %8s\n",
                    "step", "prep", "engram", "attn", "ffn_pre", "moe", "prep", "spawn", "groups", "join", "mmapgr", "tail", "shared", "misc", "setup", "head", "total", "sta", "pin", "mm", "pin MiB", "mm MiB");
        std::printf("       (sta/pin/mm = experts by tier; of pin, the stream-slot hits moved nothing: printed as hit%% = (sta + stream hits) / %u)\n", NL * TK);
        ie::Ds41Forward::Probe mprobe = [](const char* name, uint32_t L, const float*, size_t, sycl::queue&) {
            if (std::string(name) == "layer") std::fprintf(stderr, "@@layer %u\n", L); };
        std::vector<double> wall_ms; uint64_t bp_sum = 0, bm_sum = 0; uint32_t sta_sum = 0, pin_sum = 0, mm_sum = 0, sh_sum = 0, cpu_sum = 0; double grp_sum = 0, cpu_ms_sum = 0, cpu_work_sum = 0;
        // expert parallel (docs/deepseek41/38 criterion 4): per token, the sum over layers of the slower / faster card's
        // served experts and link misses, the remote tier's wall and the staging + add time
        uint32_t ep_emax = 0, ep_emin = 0, ep_mmax = 0, ep_mmin = 0; double ep_remote = 0, ep_xfer = 0; bool ep_seen = false;
        // Phase 21 step 0 (docs/deepseek41/60): the DISTRIBUTION of a layer's link misses over the two cards, which
        // is what bounds any cross-card balance -- a miss is one atomic 18.8 MB fetch, so (1,0) and (2,1) cannot be
        // improved while (2,0) and (3,0) can. Counted over the measured steps only, capped at 7 per side.
        uint64_t miss_hist[8][8] = {}; uint64_t miss_cur = 0, miss_bal = 0;
        double ept[9] = {0};                                                     // Phase 19 step 0: the EP timeline, summed over layers and timed steps
        double ep_maxsum = 0, ep_minsum = 0;                                     // per layer: max / min of (the owner at its import, the remote's rows landed) -- the E[max] geometry, measured
        uint32_t file_sum = 0;                                                   // mmap experts filled from the expert file (Phase 15 C2)
        ie::KernelProfiler prof;                                            // the named kernels of the timed steps
        for (uint32_t k = 0; k < N_BREAK + N_RATE; ++k) {
            const int32_t nxt = int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
            const bool mk = mark_step0 && k == 0;
            if (qprof && k == N_BREAK) { ie::g_profiler = &prof; prof.begin_step(); }
            if (qprof && k >= N_BREAK) prof.mark_token();
            if (mk) std::fprintf(stderr, "@@step begin\n");
            const auto t0 = std::chrono::steady_clock::now();
            if (auto e = fwd.forward(&nxt, 1, pos_start + k, logits, mk ? mprobe : ie::Ds41Forward::Probe{}); !e.empty()) { std::fprintf(stderr, "bench step %u: %s\n", k, e.c_str()); return false; }
            const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (mk) std::fprintf(stderr, "@@step end\n");
            double eng = 0, att = 0, fpre = 0, mcall = 0, shd = 0, lay = 0, mprep = 0, mspawn = 0, mgrp = 0, mjoin = 0, mmg = 0, mtail = 0;
            uint64_t bp = 0, bm = 0; uint32_t es = 0, ep = 0, em = 0, sh = 0, ec = 0; double mcpu = 0, mcpuw = 0;
            uint32_t emax = 0, emin = 0, mmax = 0, mmin = 0, efile = 0; double rem = 0, xfer = 0; double et[9] = {0};
            for (const auto& s : fwd.stats()) {
                efile += s.experts_mmap_file;
                if (s.ep_t_rows > 0) { ep_maxsum += std::max(s.ep_t_import_in, s.ep_t_rows); ep_minsum += std::min(s.ep_t_import_in, s.ep_t_rows); }
                et[0] += s.ep_t_stage; et[1] += s.ep_t_spawn; et[2] += s.ep_t_remote_in; et[3] += s.ep_t_remote_out; et[4] += s.ep_t_rows; et[5] += s.ep_t_owner_in; et[6] += s.ep_t_import_in; et[7] += s.ep_t_import_out; et[8] += s.moe_call_ms;
                if (s.experts_owner + s.experts_remote) { ep_seen = true; emax += std::max(s.experts_owner, s.experts_remote); emin += std::min(s.experts_owner, s.experts_remote);
                                                          mmax += std::max(s.miss_owner, s.miss_remote); mmin += std::min(s.miss_owner, s.miss_remote); rem += s.moe_remote_ms; xfer += s.ep_xfer_ms;
                                                          if (k >= N_BREAK) {                              // Phase 21 step 0: this layer's split, and the atomic-balanced ideal
                                                              const uint32_t hi = std::max(s.miss_owner, s.miss_remote), lo = std::min(s.miss_owner, s.miss_remote);
                                                              ++miss_hist[std::min<uint32_t>(hi, 7)][std::min<uint32_t>(lo, 7)];
                                                              miss_cur += hi; miss_bal += (hi + lo + 1) / 2;
                                                          } }
                eng += s.engram_ms; att += s.attn_ms; fpre += s.ffn_pre_ms; mcall += s.moe_call_ms; shd += s.shared_ms; lay += s.ms;
                mprep += s.moe_prep_ms; mspawn += s.moe_spawn_ms; mgrp += s.moe_groups_ms; mjoin += s.moe_join_ms; mmg += s.moe_mmap_group_ms; mtail += s.moe_tail_ms;
                bp += s.bytes_pinned; bm += s.bytes_mmap; es += s.experts_static; ep += s.experts_pinned; em += s.experts_mmap; sh += s.experts_stream_hit; ec += s.experts_cpu; mcpu += s.moe_cpu_ms; mcpuw += s.moe_cpu_work_ms;
            }
            const double misc = lay - eng - att - fpre - mcall - shd;             // inside the layers, untimed (routing checks, profile, frees)
            const double setup = wall - fwd.prep_ms() - fwd.head_ms() - lay;     // outside them: each card's rope tables, scratch, stream copies
            if (k < N_BREAK)
                std::printf("       %4u | %5.1f %6.1f %6.1f %7.1f %7.1f (%5.1f %5.1f %6.1f %5.1f %6.1f %5.1f) %6.1f %5.1f %5.1f %5.1f %7.1f | %3u/%3u/%3u | %8.1f %8.1f | hit%% %.1f\n",
                            k, fwd.prep_ms(), eng, att, fpre, mcall, mprep, mspawn, mgrp, mjoin, mmg, mtail, shd, misc, setup, fwd.head_ms(), wall,
                            es, ep, em, double(bp) / double(1 << 20), double(bm) / double(1 << 20), 100.0 * double(es + sh) / double(NL * TK)),
                std::printf("            cpu experts %u (cpu leg %.1f ms to the join, compute %.1f ms)\n", ec, mcpu, mcpuw);
            else { wall_ms.push_back(wall); bp_sum += bp; bm_sum += bm; sta_sum += es; pin_sum += ep; mm_sum += em; sh_sum += sh; grp_sum += mgrp; cpu_sum += ec; cpu_ms_sum += mcpu; cpu_work_sum += mcpuw;
                   ep_emax += emax; ep_emin += emin; ep_mmax += mmax; ep_mmin += mmin; ep_remote += rem; ep_xfer += xfer; file_sum += efile; for (int i = 0; i < 9; ++i) ept[i] += et[i]; }
        }
        if (qprof) {
            for (auto& qp : queues) qp->wait_and_throw();
            auto st = prof.harvest(); ie::g_profiler = nullptr;
            std::sort(st.begin(), st.end(), [](const ie::KernelProfiler::Stat& a, const ie::KernelProfiler::Stat& b) { return a.total_ns > b.total_ns; });
            double all = 0; for (const auto& e : st) all += e.total_ms();
            std::printf("       named kernels over the %u timed steps (device time; unnamed = oneDNN and the tier's DMAs): %.1f ms total = %.1f ms/token\n", N_RATE, all, all / N_RATE);
            // Phase 23 (docs/deepseek41/63): min and max per call as well as the mean. If the card's clock is ramping
            // out of a power-gated state after each of the ~40 fetch stalls a token contains, a kernel's fastest call
            // should approach its standalone bench while its mean does not -- a spread the mean alone hides.
            std::printf("       %-34s %8s %9s %9s %9s %9s %6s\n", "kernel", "calls", "total ms", "avg us", "min us", "max us", "%");
            for (size_t i = 0; i < st.size() && i < 28; ++i)
                std::printf("       %-34s %8u %9.1f %9.1f %9.1f %9.1f %5.1f%%\n", st[i].name.c_str(), st[i].calls, st[i].total_ms(), st[i].avg_ms() * 1000.0,
                            st[i].min_ms() * 1000.0, st[i].max_ms() * 1000.0, 100.0 * st[i].total_ms() / all);
            for (size_t d = 0; d < prof.dev_spans.size(); ++d)
                std::printf("       card %zu: busy %.1f ms, span %.1f ms over %u tokens -> the GPU executes %.0f%% of its own window\n", d, prof.dev_spans[d].busy_ns / 1e6, prof.dev_spans[d].span_ns / 1e6, prof.dev_spans[d].tokens,
                            prof.dev_spans[d].span_ns ? 100.0 * double(prof.dev_spans[d].busy_ns) / double(prof.dev_spans[d].span_ns) : 0.0);
        }
        const double sum = std::accumulate(wall_ms.begin(), wall_ms.end(), 0.0), mean = sum / double(wall_ms.size());
        const double mn = *std::min_element(wall_ms.begin(), wall_ms.end()), mx = *std::max_element(wall_ms.begin(), wall_ms.end());
        std::printf("       steps %u-%u: %.1f ms/token mean (min %.1f, max %.1f) = %.2f tok/s | per token: %.1f MiB from the pinned tier, %.1f MiB from disk; experts static/pinned/mmap %.1f/%.1f/%.1f of %u, stream-slot hits %.1f -> decode hit rate %.1f%% | pinned bytes over the groups' wall: %.1f GB/s\n",
                    N_BREAK, N_BREAK + N_RATE - 1, mean, mn, mx, 1000.0 / mean, double(bp_sum) / N_RATE / double(1 << 20), double(bm_sum) / N_RATE / double(1 << 20),
                    double(sta_sum) / N_RATE, double(pin_sum) / N_RATE, double(mm_sum) / N_RATE, NL * TK, double(sh_sum) / N_RATE, 100.0 * double(sta_sum + sh_sum) / double(N_RATE) / double(NL * TK),
                    grp_sum > 0 ? double(bp_sum) / (grp_sum / 1000.0) / 1e9 : 0.0);
        if (file_sum) std::printf("       expert file: %.1f of the %.1f mmap experts per token filled by one direct pread (the rest by the pack path)\n", double(file_sum) / N_RATE, double(mm_sum) / N_RATE);
        if (ep_seen) std::printf("       expert parallel: per token the slower card served %.1f experts and the faster %.1f (of %u), link misses slower %.1f / faster %.1f; the remote tiers' wall %.1f ms, staging + add %.1f ms\n",
                                 double(ep_emax) / N_RATE, double(ep_emin) / N_RATE, NL * TK, double(ep_mmax) / N_RATE, double(ep_mmin) / N_RATE, ep_remote / N_RATE, ep_xfer / N_RATE);
        if (ep_seen) std::printf("       ep timeline, ms per token summed over the layers from the MoE call's start (per layer = /%u): staging D2H done %.1f | helper entered %.1f, remote tier in %.1f / out %.1f, rows on the host %.1f | owner tier in %.1f, import in %.1f / out %.1f | call end %.1f\n",
                                 NL, ept[0] / N_RATE, ept[1] / N_RATE, ept[2] / N_RATE, ept[3] / N_RATE, ept[4] / N_RATE, ept[5] / N_RATE, ept[6] / N_RATE, ept[7] / N_RATE, ept[8] / N_RATE),
                     std::printf("       ep E[max]: per token, the sum over layers of the LATER of (owner at its import, remote rows landed) %.1f ms vs the earlier %.1f ms -- the difference is what the per-layer imbalance costs\n", ep_maxsum / N_RATE, ep_minsum / N_RATE);
        // Phase 21 step 0 (docs/deepseek41/60), criterion 1: the ceiling a cross-card balance could reach. The
        // marginal cost of a miss ON THE CRITICAL PATH is estimated from the two sides' own measurement -- the
        // E[max] gap divided by the miss gap -- and the ideal moves misses until the two sides differ by at most one.
        if (ep_seen && miss_cur > miss_bal) {
            const double per_miss = (ep_mmax > ep_mmin) ? (ep_maxsum - ep_minsum) / double(ep_mmax - ep_mmin) : 0.0;
            std::printf("       ep miss split, per layer over the %u measured steps (rows = the fuller side, columns = the emptier):\n", N_RATE);
            for (uint32_t hi = 0; hi < 8; ++hi) {
                uint64_t row = 0; for (uint32_t lo = 0; lo <= hi; ++lo) row += miss_hist[hi][lo];
                if (!row) continue;
                std::printf("            %u misses:", hi);
                for (uint32_t lo = 0; lo <= hi; ++lo) if (miss_hist[hi][lo]) std::printf("  (%u,%u) x%llu", hi, lo, (unsigned long long)miss_hist[hi][lo]);
                std::printf("\n");
            }
            std::printf("       ep balance CEILING: today the critical path carries %.2f misses per token, an atomic per-layer balance would carry %.2f (-%.2f); at %.2f ms per critical miss that is **%.1f ms/token** (%.1f -> %.1f ms/token, %.2f -> %.2f tok/s)\n",
                        double(miss_cur) / N_RATE, double(miss_bal) / N_RATE, double(miss_cur - miss_bal) / N_RATE, per_miss,
                        per_miss * double(miss_cur - miss_bal) / N_RATE, mean, mean - per_miss * double(miss_cur - miss_bal) / N_RATE,
                        1000.0 / mean, 1000.0 / std::max(1.0, mean - per_miss * double(miss_cur - miss_bal) / N_RATE));
        }
        if (cpu_sum) std::printf("       CPU miss path: %.1f experts per token on the CPU, its leg %.1f ms per token to the join, compute %.1f ms = %.2f ms per expert\n", double(cpu_sum) / N_RATE, cpu_ms_sum / N_RATE, cpu_work_sum / N_RATE, cpu_sum ? cpu_work_sum / double(cpu_sum) : 0.0);
        return true;
    };
    if (bench_short) {
        if (auto e = prefill(false, logits); !e.empty()) { std::fprintf(stderr, "prefill (bench): %s\n", e.c_str()); return 1; }
        if (!bench("after the golden prompt (" + std::to_string(T0) + " tokens), own router, greedy", T0, marks)) return 1;
    }
    if (bench_long) {
        const auto pp2048 = rd<int32_t>(gd + "/pp_ids_2048.i32", 2048);
        fwd.reset_state();
        fwd.set_bounded_replay(true);                                  // as deployed (Phase 10, the default), stated
        const auto tp = std::chrono::steady_clock::now();
        if (auto e = fwd.forward(pp2048.data(), 2048, 0, logits); !e.empty()) { std::fprintf(stderr, "prefill (pp2048, replay): %s\n", e.c_str()); return 1; }
        std::printf("       replayed 2048-token prefill: %.2f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count());
        if (!bench("after the replayed 2048-token pp text, own router, greedy", 2048, false)) return 1;
    }
    check(true, "Phase 11: " + std::to_string(N_BREAK + N_RATE) + " decode steps after each prompt ran and were measured (numbers above)");

    fwd.free_resident();
    for (auto& qp : queues) qp->wait_and_throw();
    std::printf("\n%s\n", g_fail ? "DECODE TEST: FAILURE(S)" : "DECODE TEST: PASS");
    return g_fail ? 1 : 0;
}
