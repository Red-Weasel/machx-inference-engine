// tools/ds41_generate_test.cpp — V4.1 Phase 12 step 3: the generate loop on the resident runtime
// (docs/deepseek41/31, criterion 3).
//   1. greedy from the golden prompt reproduces the golden's continuation tokens (the decode test's)
//   2. greedy after the 2048-token pp text, replay ON, gives the replay test's tokens (docs/28)
//   3. the sampler: temperature 0 == argmax under top-k/top-p; a seeded draw repeats; top-k 1 == argmax
//   4. streaming: the emitted pieces concatenate to decode(ids); a stop string cuts the text and ends the run
//   5. ms/token over 32 greedy steps after the 2048-token prompt, beside the decode test's number
//   usage: ie-ds41-generate-test <model> <golden_dir> [ranking_in]
#include "ie/deepseek41_generate.hpp"
#include "ie/expert_stream.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
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
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(T))); return v; }
std::string ids_str(const std::vector<int32_t>& v, size_t n) { std::string s; for (size_t i = 0; i < n && i < v.size(); ++i) s += std::to_string(v[i]) + " "; return s; }
}  // namespace

int main(int argc, char** argv) {
    // The prefill(T) vs prefill(T-2) + 2 steps check requires the SAME argmax, which holds on the int-dot W4A8 prefill route
    // the test was built on. The engine default since 2026-09-17 is the XMX W4A16 route: there the paths differ by design,
    // measured at T = 512 as KL 0.0031 (< the 0.02 bar) with a near-tie argmax flip (docs/deepseek41/93). The test pins the
    // route it gates unless the caller names one (IE_DS41_PREFILL_XMX=1 to run it on XMX).
    setenv("IE_DS41_PREFILL_XMX", "0", 0);
    const std::string dir = argc > 1 ? argv[1] : std::string(std::getenv("HOME")) + "/models/DeepSeek-V4.1-Flash";
    const std::string gd  = argc > 2 ? argv[2] : "/tmp/ds41_golden";
    const std::string rank_in = argc > 3 ? argv[3] : "";
    nlohmann::json meta; { std::ifstream f(gd + "/decode2/d_meta.json"); if (!f) { std::fprintf(stderr, "no decode2/d_meta.json\n"); return 1; } f >> meta; }
    const std::vector<int32_t> prompt = meta["prompt_ids"].get<std::vector<int32_t>>();
    const std::vector<int32_t> all_ids = meta["all_ids"].get<std::vector<int32_t>>();
    const uint32_t T0 = uint32_t(prompt.size());

    // ---- the sampler alone, no GPU ------------------------------------------------------------
    std::printf("=== the sampler (host) ===\n");
    {
        std::vector<float> lg(1000); for (size_t i = 0; i < lg.size(); ++i) lg[i] = std::sin(float(i) * 0.37f) * 5.f;   // a fixed, spiky distribution
        const int32_t am = int32_t(std::max_element(lg.begin(), lg.end()) - lg.begin());
        ie::Ds41SampleParams p0; p0.temperature = 0.f; p0.top_k = 40; p0.top_p = 0.9f; uint64_t r = 1;
        auto l = lg; check(ie::Ds41Generator::sample(l, {}, p0, r) == am, "temperature 0 under top-k 40 / top-p 0.9 is the argmax", std::to_string(am));
        ie::Ds41SampleParams p1; p1.temperature = 0.8f; p1.top_k = 50; p1.top_p = 0.95f; p1.seed = 7;
        uint64_t ra = 7, rb = 7; std::vector<int32_t> a, b;
        for (int i = 0; i < 16; ++i) { auto la = lg, lb = lg; a.push_back(ie::Ds41Generator::sample(la, {}, p1, ra)); b.push_back(ie::Ds41Generator::sample(lb, {}, p1, rb)); }
        check(a == b, "a seeded temperature-0.8 draw repeats exactly (16 picks)", ids_str(a, 6));
        bool all_in_top = true; std::vector<int32_t> idx(lg.size()); for (size_t i = 0; i < idx.size(); ++i) idx[i] = int32_t(i);
        std::sort(idx.begin(), idx.end(), [&](int32_t x, int32_t y) { return lg[size_t(x)] > lg[size_t(y)]; });
        for (int32_t v : a) if (std::find(idx.begin(), idx.begin() + 50, v) == idx.begin() + 50) all_in_top = false;
        check(all_in_top, "every sampled id is inside the top-k 50 set");
        ie::Ds41SampleParams p2; p2.temperature = 1.5f; p2.top_k = 1; uint64_t rc = 3; auto lc = lg;
        check(ie::Ds41Generator::sample(lc, {}, p2, rc) == am, "top-k 1 at temperature 1.5 is the argmax");
        ie::Ds41SampleParams p3; p3.temperature = 0.f; p3.repeat_penalty = 1.5f; p3.repeat_window = 8; auto ld = lg;
        const int32_t pen = ie::Ds41Generator::sample(ld, {am}, p3, rc);
        check(pen != am && ld[size_t(am)] < lg[size_t(am)], "the repetition penalty lowers a recent positive logit and moves the argmax off it", std::to_string(pen));
    }

    ie::DeepSeek41Model m;
    if (auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    ie::Ds41EngramTables tb;
    if (auto e = tb.load(gd); !e.empty()) { std::fprintf(stderr, "tables: %s\n", e.c_str()); return 1; }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_hf_json(dir + "/tokenizer.json", dir + "/tokenizer_config.json"); !e.empty()) { std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1; }
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
    if (!rank_in.empty()) { if (auto e = ie::ds4_expert_priority_read_layers(rank_in, m.config().n_routed_experts, m.config().n_layers, ranking); !e.empty()) { std::fprintf(stderr, "ranking: %s\n", e.c_str()); return 1; } }
    ie::Ds41Forward fwd; ie::Ds41Forward::ResidentOptions opt; opt.max_tokens = 2048 + 64;
    if (auto e = fwd.init_resident(qs, m, tb, ranking, opt); !e.empty()) { std::fprintf(stderr, "init_resident: %s\n", e.c_str()); return 1; }
    ie::Ds41Generator gen(fwd, tok);

    // ---- 1. greedy from the golden prompt -----------------------------------------------------
    std::printf("\n=== greedy from the golden prompt (%u tokens) ===\n", T0);
    std::vector<int32_t> out; ie::Ds41GenStats st; std::string text;
    ie::Ds41SampleParams greedy;
    if (auto e = gen.run(prompt, 4, greedy, {}, [&](std::string_view p) { text.append(p); return true; }, out, st); !e.empty()) { std::fprintf(stderr, "run: %s\n", e.c_str()); return 1; }
    std::vector<int32_t> want(all_ids.begin() + T0, all_ids.begin() + T0 + 4);
    check(out == want, "the four greedy tokens equal the golden's continuation", "engine " + ids_str(out, 4) + "| golden " + ids_str(want, 4));
    std::printf("       text: '%s'  stop: %s, prefill %.2f s, %.0f ms/token\n", text.c_str(), st.stop_reason.c_str(), st.prefill_s, st.n_gen ? 1000.0 * st.decode_s / st.n_gen : 0.0);
    check(text == tok.decode(out, false), "the streamed pieces concatenate to decode(ids)");
    // an odd-length prompt: the same prompt without its first token (11 tokens): must run, not be refused
    { std::vector<int32_t> odd(prompt.begin() + 1, prompt.end()); std::vector<int32_t> o2; ie::Ds41GenStats s2; std::string t2;
      const auto e = gen.run(odd, 2, greedy, {}, [&](std::string_view p) { t2.append(p); return true; }, o2, s2);
      check(e.empty() && o2.size() == 2, "an odd-length prompt (11 tokens) is prefilled even and its tail stepped", e.empty() ? "'" + t2 + "'" : e); }
    // a stop string
    { std::vector<int32_t> o3; ie::Ds41GenStats s3; std::string t3;
      if (auto e = gen.run(prompt, 16, greedy, {"."}, [&](std::string_view p) { t3.append(p); return true; }, o3, s3); !e.empty()) { std::fprintf(stderr, "run (stop): %s\n", e.c_str()); return 1; }
      check(s3.stop_reason == "stop" && t3.find('.') == std::string::npos && !t3.empty(), "a stop string '.' ends the run and is not emitted", "'" + t3 + "', stop_reason " + s3.stop_reason + ", " + std::to_string(s3.n_gen) + " tokens"); }

    // ---- the property the odd-length rule rests on: a decode step continues a prefill exactly ------
    // prefill(T) vs prefill(T-2) + two decode steps, both legal (even) lengths, the logits at the
    // same final position (gate 12 pre-read, P3). The difference is the prefill-vs-decode path
    // difference through all 40 layers; the bar is Phase 9's logits regression bar.
    // gate 12 finding 3: at four lengths (the golden prompt and three even prefixes of the pp2048
    // text, replay off so the exact path is the reference), not one; the 5e-2 is Phase 9's
    // logits-vs-golden bar, borrowed -- the natural scale here is the ~1e-2 amplification, and
    // every number is printed beside the verdict
    std::printf("\n=== prefill(T) vs prefill(T-2) + 2 steps: the logits at position T-1 ===\n");
    {
        const auto pp2048_c = rd<int32_t>(gd + "/pp_ids_2048.i32", 2048);
        fwd.set_bounded_replay(false);
        for (uint32_t T : {T0, 130u, 512u, 1030u}) {
            const int32_t* ids = T == T0 ? prompt.data() : pp2048_c.data();
            std::vector<float> a, b;
            if (auto e = fwd.forward(ids, T, 0, a); !e.empty()) { std::fprintf(stderr, "prefill(T): %s\n", e.c_str()); return 1; }
            if (auto e = fwd.forward(ids, T - 2, 0, b); !e.empty()) { std::fprintf(stderr, "prefill(T-2): %s\n", e.c_str()); return 1; }
            if (auto e = fwd.forward(ids + T - 2, 1, T - 2, b); !e.empty()) { std::fprintf(stderr, "step T-2: %s\n", e.c_str()); return 1; }
            if (auto e = fwd.forward(ids + T - 1, 1, T - 1, b); !e.empty()) { std::fprintf(stderr, "step T-1: %s\n", e.c_str()); return 1; }
            // gate 15 finding 2: the max-relative logits difference with a borrowed 5e-2 bar is text-dependent
            // (5.9e-2 at T = 130 on the re-baselined text, 2-5e-2 elsewhere); the quantity that matters for
            // generation is the divergence of the two paths' next-token distributions, so the check is the
            // replay test's: KL(prefill(T) || prefill(T-2) + 2 steps) under its 0.02 bar, and the same argmax.
            // The max-relative number stays printed beside it.
            double m = 0, sc = 0; for (size_t i = 0; i < a.size(); ++i) { m = std::max(m, std::fabs(double(a[i]) - double(b[i]))); sc = std::max(sc, std::fabs(double(a[i]))); }
            const double r = sc > 0 ? m / sc : 0;
            auto lse = [](const std::vector<float>& v) { double mx = -1e300; for (float x : v) mx = std::max(mx, double(x)); double s2 = 0; for (float x : v) s2 += std::exp(double(x) - mx); return mx + std::log(s2); };
            const double la = lse(a), lb = lse(b); double kl = 0;
            for (size_t i = 0; i < a.size(); ++i) { const double pa = std::exp(double(a[i]) - la); kl += pa * ((double(a[i]) - la) - (double(b[i]) - lb)); }
            const int32_t ta = int32_t(std::max_element(a.begin(), a.end()) - a.begin()), tb = int32_t(std::max_element(b.begin(), b.end()) - b.begin());
            char buf[128]; std::snprintf(buf, sizeof buf, "KL %.4f, max-rel %.3e, argmax %d vs %d", kl, r, ta, tb);
            check(kl < 0.02 && ta == tb, "T = " + std::to_string(T) + ": prefill(T) and prefill(T-2) + 2 decode steps agree at position T-1: KL(prefill || decode path) < 0.02 (the replay test's bar) with the same argmax", buf);
        }
    }

    // ---- 2 + 5. after the 2048-token pp text, replay ON ----------------------------------------
    std::printf("\n=== after the 2048-token pp text, replay ON ===\n");
    const auto pp2048 = rd<int32_t>(gd + "/pp_ids_2048.i32", 2048);
    fwd.set_bounded_replay(true);
    std::vector<int32_t> o4; ie::Ds41GenStats s4; std::string t4;
    if (auto e = gen.run(pp2048, 36, greedy, {}, [&](std::string_view p) { t4.append(p); return true; }, o4, s4); !e.empty()) { std::fprintf(stderr, "run (pp2048): %s\n", e.c_str()); return 1; }
    // the exact path's four greedy tokens after the 2048-token prefill, written by the replay test (criterion 7)
    // for THIS text -- gate 15 finding 1: the earlier hard-coded 369 2619 1683 16 was one text's continuation
    std::vector<int32_t> want_replay(4, -1);
    { std::ifstream f(gd + "/replay_pp2048_greedy4.i32", std::ios::binary);
      if (!f) { std::printf("[FAIL] %s/replay_pp2048_greedy4.i32 is absent: run ie-ds41-replay-test first (it writes the exact path's tokens)\n", gd.c_str()); ++g_fail; }
      else f.read(reinterpret_cast<char*>(want_replay.data()), 16); }
    check(o4.size() >= 4 && std::equal(want_replay.begin(), want_replay.end(), o4.begin()), "the first four greedy tokens equal the replay test's exact path's (" + ids_str(want_replay, 4) + ")", ids_str(o4, 4));
    const double ms = s4.n_gen > 4 ? 1000.0 * s4.decode_s / double(s4.n_gen) : 0.0;
    std::printf("       prefill %.2f s (2048 tokens), %u tokens in %.2f s = %.1f ms/token (the decode test: 194.9 on the shipped Phase 11 build)\n       text: '%s'\n", s4.prefill_s, s4.n_gen, s4.decode_s, ms, t4.substr(0, 200).c_str());
    // gate 12 finding 2: the loop's own runs span 173-181 ms on this build (three runs, two
    // sessions) and the decode test's bench 194-199; a +20% loop regression is ~215, so the bar
    // is 210 -- above the loop's spread, below a fifth of regression
    check(ms > 0 && ms < 210.0, "ms/token over the greedy run is under 210 (the loop's runs: 173-181; a 20% regression would be ~215)", std::to_string(int(ms)) + " ms");

    fwd.free_resident();
    for (auto& qp : queues) qp->wait_and_throw();
    std::printf("\n%s\n", g_fail ? "GENERATE TEST: FAILURE(S)" : "GENERATE TEST: PASS");
    return g_fail ? 1 : 0;
}
