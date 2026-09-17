// tools/ds41_replay_test.cpp — V4.1 Phase 10: Decoder SWA Bounded Replay, mode on vs the exact
// path, on the resident runtime (docs/deepseek41/27; the report's §2.2 and §3.2.2).
//   2. prompts that fit the window (12 and 128 tokens): bit-identical logits on vs off
//   3. twenty prefixes of the pp2048 text (five multiples of 128, fifteen not -- gate 10: the five
//      alone hid two bugs): replay must be CLOSER to the exact path than the truncated reference
//      (the last 128 tokens as a fresh prompt, i.e. no global context), KL(e||r) < KL(e||t) over
//      the softmax; the next tokens and the agreement rate are reported
//   4. layer 20's latents and index keys equal the exact path's for all T entries, bit for bit,
//      and its window ring slot by slot (positions T-128..T-1 at slot p % 128 on both paths);
//      at T = 130 (offset 2) and 450 (offset 322) the segment's sliding mask itself, read back
//      through the probe, is the lower-triangular [128 x 128] pattern at layers 20 and 30
//   5. speed on vs off at the five 128-multiples, cold and warm
//   7. four decode steps after the 1930- and 2048-token prefills, teacher-forced on the exact path's
//      greedy tokens: replay closer to exact than the truncated reference at every step (gate 10
//      finding 8), the exact path's top-1/top-2 margin printed beside every step
//   (1, the exact path untouched, is the existing tests; 6, memory, is the resident test's unload)
//   usage: ie-ds41-replay-test <model> <golden_dir> [ranking_in]
#include "ie/deepseek41_forward.hpp"
#include "ie/expert_stream.hpp"

#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
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
    if (size_t(f.gcount()) != n * sizeof(T)) { std::fprintf(stderr, "%s: short\n", p.c_str()); std::exit(1); }
    return v; }
double rel(const float* a, const float* b, size_t n) {
    double m = 0, s = 0; for (size_t i = 0; i < n; ++i) { m = std::max(m, std::fabs(double(a[i]) - double(b[i]))); s = std::max(s, std::fabs(double(b[i]))); }
    return s > 0 ? m / s : 0; }
std::string f3(double v) { char b[32]; std::snprintf(b, sizeof b, "%.3e", v); return b; }
std::vector<int32_t> top5(const float* row, uint32_t V) {
    std::vector<int32_t> ids(V); std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + 5, ids.end(), [&](int32_t a, int32_t b) { return row[a] > row[b]; });
    ids.resize(5); return ids;
}
double kl_softmax(const std::vector<float>& a, const std::vector<float>& b) {   // KL(softmax(a) || softmax(b)), nats
    double ma = -1e300, mb = -1e300; for (float v : a) ma = std::max(ma, double(v)); for (float v : b) mb = std::max(mb, double(v));
    double za = 0, zb = 0; for (size_t i = 0; i < a.size(); ++i) { za += std::exp(double(a[i]) - ma); zb += std::exp(double(b[i]) - mb); }
    double kl = 0;
    for (size_t i = 0; i < a.size(); ++i) { const double la = double(a[i]) - ma - std::log(za), lb = double(b[i]) - mb - std::log(zb); kl += std::exp(la) * (la - lb); }
    return kl; }
double secs(const std::chrono::steady_clock::time_point& t) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); }
}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : std::string(std::getenv("HOME")) + "/models/DeepSeek-V4.1-Flash";
    const std::string gd  = argc > 2 ? argv[2] : "/tmp/ds41_golden";
    const std::string rank_in = argc > 3 ? argv[3] : "";

    nlohmann::json meta; { std::ifstream f(gd + "/model/m_meta.json"); if (!f) { std::fprintf(stderr, "no m_meta.json\n"); return 1; } f >> meta; }
    const std::vector<int32_t> prompt12 = meta["ids"].get<std::vector<int32_t>>();
    const uint32_t NL = meta["n_layers"], V = meta["vocab"];

    ie::DeepSeek41Model m;
    if (auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    ie::Ds41EngramTables tb;
    if (auto e = tb.load(gd); !e.empty()) { std::fprintf(stderr, "tables: %s\n", e.c_str()); return 1; }
    const auto& c = m.config();
    const uint32_t E = c.n_routed_experts, WIN = c.window_size, DEC0 = NL / 2;

    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d);
    }
    if (devs.empty()) { std::fprintf(stderr, "no Arc GPU\n"); return 1; }
    std::vector<std::unique_ptr<sycl::queue>> queues; std::vector<sycl::queue*> qs;
    for (const auto& d : devs) { queues.push_back(std::make_unique<sycl::queue>(sycl::context(d), d, sycl::property_list{sycl::property::queue::in_order{}})); qs.push_back(queues.back().get()); }

    std::vector<std::vector<uint32_t>> ranking;
    if (!rank_in.empty()) { if (auto e = ie::ds4_expert_priority_read_layers(rank_in, E, NL, ranking); !e.empty()) { std::fprintf(stderr, "ranking: %s\n", e.c_str()); return 1; } }
    ie::Ds41Forward fwd; ie::Ds41Forward::ResidentOptions opt;
    opt.max_tokens = 2048 + 16;            // criterion 7 decodes four steps past a 2048-token prompt
    const auto ti = std::chrono::steady_clock::now();
    if (auto e = fwd.init_resident(qs, m, tb, ranking, opt); !e.empty()) { std::fprintf(stderr, "init_resident: %s\n", e.c_str()); return 1; }
    fwd.set_logits_last_only(true);
    std::printf("resident in %.0f s on %u card(s); window %u, decoder from layer %u\n", secs(ti), uint32_t(devs.size()), WIN, DEC0);

    const auto pp2048 = rd<int32_t>(gd + "/pp_ids_2048.i32", 2048);
    const auto pp512  = rd<int32_t>(gd + "/pp_ids_512.i32", 512);
    std::vector<float> lg;
    auto prefill = [&](const std::vector<int32_t>& in, bool replay, const ie::Ds41Forward::Probe& probe = {}) -> double {
        fwd.set_bounded_replay(replay);
        const auto t0 = std::chrono::steady_clock::now();
        if (auto e = fwd.forward(in.data(), uint32_t(in.size()), 0, lg, probe); !e.empty()) { std::fprintf(stderr, "prefill(%zu, replay=%d): %s\n", in.size(), int(replay), e.c_str()); std::exit(1); }
        return secs(t0);
    };

    // ---- criterion 2: prompts that fit the window are the exact path, bit for bit ------------
    std::printf("\n=== criterion 2: prompts within the window, mode on vs off ===\n");
    for (const auto* pr : {&prompt12, (const std::vector<int32_t>*)nullptr}) {
        std::vector<int32_t> in = pr ? *pr : std::vector<int32_t>(pp512.begin(), pp512.begin() + WIN);
        prefill(in, false); std::vector<float> off = lg;
        prefill(in, true);
        check(off == lg, "T = " + std::to_string(in.size()) + ": replay on == off, bit for bit", off == lg ? "" : "rel " + f3(rel(lg.data(), off.data(), V)));
    }

    // ---- criteria 3, 4, 5: prefixes of the pp2048 text -----------------------------------------
    // For each prefix: the exact path, the replay, and the TRUNCATED reference (its last 128 tokens
    // as a fresh prompt: what the decoder would see with no global context at all). Replay's whole
    // point is layer 20's compressed cache over the full prompt, so it must land closer to the exact
    // path than the truncated reference does: KL(e||r) < KL(e||t). Layer 20's state: latents and
    // index keys bit for bit; the ring slot by slot (both paths hold positions T-128..T-1 at slot
    // p % 128; the segment's rows are the same stream rows, so the bar is the GEMM's rounding).
    std::printf("\n=== criteria 3-5: prefixes of the pp2048 text, replay vs exact vs the truncated (last-128) reference ===\n");
    std::printf("       %6s | %7s %7s %6s | %-24s | %9s %9s | %7s | %s\n", "T", "exact s", "repl. s", "warm s", "next exact/replay/trunc", "KL(e||r)", "KL(e||t)", "margin", "layer-20 state");
    const std::vector<uint32_t> prefixes = {130u, 190u, 256u, 258u, 322u, 450u, 512u, 600u, 706u, 898u, 1024u, 1030u, 1150u, 1290u, 1536u, 1538u, 1800u, 1930u, 2046u, 2048u};
    uint32_t agree = 0, closer = 0, judged = 0; double kl_r_sum = 0, kl_t_sum = 0, kl_r_max = 0;
    // one prefix of one text: exact, replay (with the mask probe when asked), warm when timed, truncated
    auto judge_prefix = [&](const std::string& name, const std::vector<int32_t>& text, uint32_t T, bool timed, bool probe_mask) -> int {
        std::vector<int32_t> in(text.begin(), text.begin() + T);
        const double s_exact = prefill(in, false); std::vector<float> lg_exact = lg;
        std::vector<float> win_e, comp_e, idx_e; uint32_t nc_e = 0;
        if (auto e = fwd.read_state(DEC0, win_e, comp_e, idx_e, nc_e); !e.empty()) { std::fprintf(stderr, "read_state: %s\n", e.c_str()); return 1; }
        // the segment's sliding mask, read back at layers 20 and 30 for the two bug regimes (offset 2:
        // future keys; offset 322: no keys at all, before the fix): must be exactly lower-triangular
        std::vector<uint32_t> mask_bad; uint32_t mask_seen = 0;
        ie::Ds41Forward::Probe mprobe = [&](const char* name, uint32_t L, const float* d, size_t n, sycl::queue& pq) {
            if (std::string(name) != "swa_mask" || (L != DEC0 && L != DEC0 + 10)) return;
            std::vector<float> mk(n); pq.memcpy(mk.data(), d, n * 4).wait(); ++mask_seen;
            const uint32_t Tq = uint32_t(std::lround(std::sqrt(double(n)))); uint32_t bad = 0;
            for (uint32_t i = 0; i < Tq; ++i) for (uint32_t kk = 0; kk < Tq; ++kk) if ((mk[size_t(i) * Tq + kk] == 0.f) != (kk <= i)) ++bad;
            mask_bad.push_back(bad); };
        const double s_replay = prefill(in, true, probe_mask ? mprobe : ie::Ds41Forward::Probe{}); std::vector<float> lg_replay = lg;
        if (probe_mask) {
            const uint32_t worst = mask_bad.empty() ? 0u : *std::max_element(mask_bad.begin(), mask_bad.end());
            check(mask_seen == 2 && worst == 0, name + " T = " + std::to_string(T) + " (offset " + std::to_string(T - WIN) + "): the segment's sliding mask is exactly lower-triangular at layers " + std::to_string(DEC0) + " and " + std::to_string(DEC0 + 10),
                  std::to_string(mask_seen) + " masks read, worst " + std::to_string(worst) + " wrong entries of " + std::to_string(WIN * WIN));
        }
        std::vector<float> win_r, comp_r, idx_r; uint32_t nc_r = 0;
        if (auto e = fwd.read_state(DEC0, win_r, comp_r, idx_r, nc_r); !e.empty()) { std::fprintf(stderr, "read_state: %s\n", e.c_str()); return 1; }
        const double s_warm = timed ? prefill(in, true) : 0.0;
        std::vector<int32_t> tail(in.end() - WIN, in.end());
        prefill(tail, false); std::vector<float> lg_trunc = lg;
        const auto e5 = top5(lg_exact.data(), V), r5 = top5(lg_replay.data(), V), t5 = top5(lg_trunc.data(), V);
        const double margin = lg_exact[e5[0]] - lg_exact[e5[1]];
        const double kl_r = kl_softmax(lg_exact, lg_replay), kl_t = kl_softmax(lg_exact, lg_trunc);
        const double ring_rel = rel(win_r.data(), win_e.data(), win_e.size());
        const bool same = e5[0] == r5[0];
        const bool state_ok = nc_e == T && nc_r == T && comp_e == comp_r && idx_e == idx_r && ring_rel < 1e-5;
        const std::string st = state_ok ? "identical, ring rel " + f3(ring_rel)
                                        : "nc " + std::to_string(nc_e) + "/" + std::to_string(nc_r) + (comp_e == comp_r ? "" : " latents differ") + (idx_e == idx_r ? "" : " keys differ") + " ring rel " + f3(ring_rel);
        std::printf("       %6u | %7.2f %7.2f %6.2f | %7d / %-7d / %-7d | %9.4f %9.4f | %7.3f | %s\n", T, s_exact, s_replay, s_warm, e5[0], r5[0], t5[0],
                    kl_r, kl_t, margin, st.c_str());
        check(state_ok, name + " T = " + std::to_string(T) + ": layer " + std::to_string(DEC0) + " latents and index keys equal the exact path's bit for bit, the ring slot by slot within 1e-5");
        check(kl_r < kl_t, name + " T = " + std::to_string(T) + ": replay is closer to the exact path than the truncated reference (KL e||r < KL e||t)",
              "KL " + f3(kl_r) + " vs " + f3(kl_t));
        if (same) ++agree; if (kl_r < kl_t) ++closer; kl_r_sum += kl_r; kl_t_sum += kl_t; kl_r_max = std::max(kl_r_max, kl_r); ++judged;
        return 0;
    };
    for (uint32_t T : prefixes) if (judge_prefix("pp2048", pp2048, T, T % 128 == 0, T == 130 || T == 450)) return 1;
    // a second text (gate 10 round 2, finding 3: pp_ids_512 is a prefix of pp_ids_2048, so the twenty
    // above are cuts of ONE text): a 2048-token slice of the held-out corpus, four prefixes
    std::printf("       second text: corpus_ids_65536[32768:34816]\n");
    {
        const auto corpus = rd<int32_t>(gd + "/corpus_ids_65536.i32", 65536);
        const std::vector<int32_t> text2(corpus.begin() + 32768, corpus.begin() + 32768 + 2048);
        for (uint32_t T : {300u, 706u, 1290u, 2048u}) if (judge_prefix("corpus", text2, T, false, false)) return 1;
    }
    std::printf("       next token agrees with the exact path's at %u of %u prefixes; replay closer than truncated at %u of %u; mean KL(e||r) %.4f (max %.4f), KL(e||t) %.4f\n",
                agree, judged, closer, judged, kl_r_sum / double(judged), kl_r_max, kl_t_sum / double(judged));
    // the absolute companion bar (gate 10 round 2, finding 4): the relative test cannot catch a
    // regression that raises both KLs. A REGRESSION DETECTOR, not the evidence of correctness (that
    // is the mask assertion, the ring, and the ordering above): calibrated once on the fixed build
    // over both texts (24 prefixes: mean 0.0052, max 0.0567 at pp2048 T = 322, run 4 of 2026-09-13)
    // with ~3.5x headroom, and recorded as such in docs/deepseek41/28.
    check(kl_r_sum / double(judged) < 0.02 && kl_r_max < 0.2, "over every prefix judged, KL(exact||replay) mean under 0.02 and max under 0.2 (regression bar, calibrated: 0.0052 / 0.0567)",
          "mean " + f3(kl_r_sum / double(judged)) + ", max " + f3(kl_r_max));

    // ---- criterion 7: decode after a replayed prefill, exact vs replay vs truncated -------------
    // Teacher-forced on the exact path's four greedy tokens, so the three paths score the same
    // sequence at every step; replay must be closer to exact than the truncated reference at each.
    // At 1930 (offset 1802, not a multiple of the window) and at 2048. The exact path's top-1/top-2
    // margin stands beside every step: a divergence at a near-tie is benign, one at a wide margin is not.
    for (uint32_t T : {1930u, 2048u}) {
        std::printf("\n=== criterion 7: four decode steps after the %u-token prefill, teacher-forced on the exact path's tokens ===\n", T);
        std::vector<int32_t> in(pp2048.begin(), pp2048.begin() + T);
        std::vector<int32_t> toks;
        std::vector<std::vector<float>> lg_e(5), lg_r(5), lg_t(5);
        prefill(in, false); lg_e[0] = lg;
        for (uint32_t k = 0; k < 4; ++k) {
            const int32_t nxt = int32_t(std::max_element(lg.begin(), lg.end()) - lg.begin()); toks.push_back(nxt);
            if (auto e = fwd.forward(&nxt, 1, T + k, lg); !e.empty()) { std::fprintf(stderr, "decode (exact) step %u: %s\n", k, e.c_str()); return 1; }
            lg_e[k + 1] = lg;
        }
        prefill(in, true); lg_r[0] = lg;
        for (uint32_t k = 0; k < 4; ++k) { if (auto e = fwd.forward(&toks[k], 1, T + k, lg); !e.empty()) { std::fprintf(stderr, "decode (replay) step %u: %s\n", k, e.c_str()); return 1; } lg_r[k + 1] = lg; }
        { std::vector<int32_t> tail(in.end() - WIN, in.end()); prefill(tail, false); lg_t[0] = lg;
          for (uint32_t k = 0; k < 4; ++k) { if (auto e = fwd.forward(&toks[k], 1, WIN + k, lg); !e.empty()) { std::fprintf(stderr, "decode (truncated) step %u: %s\n", k, e.c_str()); return 1; } lg_t[k + 1] = lg; } }
        std::printf("       exact path's greedy tokens: %d %d %d %d\n", toks[0], toks[1], toks[2], toks[3]);
        if (T == 2048) {   // the generate test's reference (gate 15 finding 1: never a hard-coded continuation of one text)
            std::ofstream f(gd + "/replay_pp2048_greedy4.i32", std::ios::binary); f.write(reinterpret_cast<const char*>(toks.data()), 16);
            std::printf("       written to %s/replay_pp2048_greedy4.i32 for the generate test\n", gd.c_str());
        }
        std::printf("       %4s | %-24s | %7s | %9s %9s\n", "step", "argmax exact/replay/trunc", "margin", "KL(e||r)", "KL(e||t)");
        bool all_closer = true; uint32_t diverged = 0;
        for (uint32_t k = 0; k < 5; ++k) {
            const auto e5 = top5(lg_e[k].data(), V), r5 = top5(lg_r[k].data(), V), t5 = top5(lg_t[k].data(), V);
            const double margin = lg_e[k][e5[0]] - lg_e[k][e5[1]];
            const double kl_r = kl_softmax(lg_e[k], lg_r[k]), kl_t = kl_softmax(lg_e[k], lg_t[k]);
            std::printf("       %4u | %7d / %-7d / %-7d | %7.3f | %9.4f %9.4f%s\n", k, e5[0], r5[0], t5[0], margin, kl_r, kl_t, e5[0] == r5[0] ? "" : "   <- replay's argmax differs");
            all_closer = all_closer && kl_r < kl_t; if (e5[0] != r5[0]) ++diverged;
        }
        check(all_closer, "after the " + std::to_string(T) + "-token prefill, replay is closer to the exact path than the truncated reference at every one of the 5 scored steps",
              std::to_string(diverged) + " of 5 argmax differ (reported, not required)");
    }

    fwd.free_resident();
    for (auto& qp : queues) qp->wait_and_throw();
    std::printf("\n%s\n", g_fail ? "REPLAY TEST: FAILURE(S)" : "REPLAY TEST: PASS");
    return g_fail ? 1 : 0;
}
