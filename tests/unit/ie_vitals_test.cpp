// tests/unit/ie_vitals_test.cpp -- ie_vitals (docs/mimo26/IE_VITALS.md), CPU only:
//  1. sample_row's entropy / margin against hand-computed values (and a long-double reference on LM-like rows);
//  2. the stats pointer never changes the pick or the RNG stream;
//  3. VitalsWindow's counts, the window / summary JSON, sse_add_field;
//  4. the request parser: ie_vitals absent / false / true.
#undef NDEBUG
#include "ie/deepseek41_generate.hpp"
#include "ie/openai_proto.hpp"
#include "ie/vitals.hpp"
#include "nlohmann/json.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using nlohmann::json;

namespace {
int fails = 0;
void near(double got, double want, double tol, const char* what) {
    if (std::fabs(got - want) > tol) { std::printf("FAIL %s: got %.9f want %.9f\n", what, got, want); ++fails; }
}
void check(bool ok, const char* what) { if (!ok) { std::printf("FAIL %s\n", what); ++fails; } }

ie::Ds41SampleStats stats_of(std::vector<float> lg, float temp, uint32_t top_k, const std::vector<int32_t>& recent = {}, float pen = 1.f) {
    ie::Ds41SampleParams sp; sp.temperature = temp; sp.top_k = top_k; sp.top_p = 0.95f; sp.repeat_penalty = pen;
    uint64_t rng = 7; ie::Ds41SampleStats st;
    ie::Ds41Generator::sample_row(lg.data(), lg.size(), recent, sp, rng, &st);
    return st;
}
json frame_json(const std::string& f) {   // "data: {...}\n\n"
    check(f.rfind("data: ", 0) == 0 && f.size() > 8 && f.substr(f.size() - 2) == "\n\n", "SSE frame shape");
    return json::parse(f.substr(6, f.size() - 8));
}
}  // namespace

int main() {
    const double ln2 = std::log(2.0);
    // 1. hand-computed: softmax(ln4, ln2, 0, 0) = (1/2, 1/4, 1/8, 1/8): H = 1.75 ln 2, margin 1/4
    {
        auto st = stats_of({float(std::log(4.0)), float(ln2), 0.f, 0.f}, 1.f, 0);
        check(st.has_H, "T=1 reports H");
        near(st.H, 1.75 * ln2, 1e-6, "H of (1/2,1/4,1/8,1/8)");
        near(st.margin, 0.25, 1e-6, "margin of (1/2,1/4,1/8,1/8)");
        // the same distribution at T = 2 (logits doubled)
        auto s2 = stats_of({float(2 * std::log(4.0)), float(2 * ln2), 0.f, 0.f}, 2.f, 0);
        near(s2.H, 1.75 * ln2, 1e-6, "H at T=2");
        // top_k 2 keeps (2/3, 1/3): H = ln3 - (2/3) ln2, margin 1/3
        auto s3 = stats_of({float(std::log(4.0)), float(ln2), 0.f, 0.f}, 1.f, 2);
        near(s3.H, std::log(3.0) - 2.0 / 3.0 * ln2, 1e-6, "H under top_k 2");
        near(s3.margin, 1.0 / 3.0, 1e-6, "margin under top_k 2");
        // uniform over 8: H = ln 8 (> the 2-nat cut), margin 0
        auto s4 = stats_of(std::vector<float>(8, 0.f), 1.f, 0);
        near(s4.H, 3 * ln2, 1e-6, "H uniform 8");
        near(s4.margin, 0.0, 1e-9, "margin uniform");
        // repeat penalty 2 on id 0: ln4 -> ln2, softmax = (1/3, 1/3, 1/6, 1/6): H = (2/3) ln3 + (1/3) ln6 (post-penalty)
        auto s5 = stats_of({float(std::log(4.0)), float(ln2), 0.f, 0.f}, 1.f, 0, {0}, 2.f);
        near(s5.H, 2.0 / 3.0 * std::log(3.0) + 1.0 / 3.0 * std::log(6.0), 1e-6, "H after the repeat penalty");
        near(s5.margin, 0.0, 1e-6, "margin after the repeat penalty");
        // greedy: no entropy (no extra vocabulary pass)
        ie::Ds41SampleStats g; g.has_H = true;
        std::vector<float> lg{1.f, 3.f, 2.f}; ie::Ds41SampleParams sp; sp.temperature = 0.f; uint64_t rng = 1;
        check(ie::Ds41Generator::sample_row(lg.data(), 3, {}, sp, rng, &g) == 1 && !g.has_H, "greedy: argmax, has_H false");
    }
    // 1b + 2. LM-like rows at the MiMo vocabulary size: H vs a long-double reference; the pick and the RNG stream with
    // stats == without
    {
        const size_t V = 152576;
        std::mt19937 g(20260925);
        struct Case { float temp, top_p, min_p; uint32_t top_k; };
        const Case cases[] = {{1.f, 0.95f, 0.f, 0}, {0.7f, 0.8f, 0.f, 0}, {1.f, 0.95f, 0.f, 40}, {1.f, 1.f, 0.05f, 0}, {0.6f, 0.95f, 0.f, 20}};
        int rows = 0;
        for (int r = 0; r < 24; ++r) {
            std::vector<float> lg(V);
            std::normal_distribution<float> nd(0.f, 2.5f);
            for (auto& v : lg) v = nd(g);
            for (int k = 0; k < 1 + r % 12; ++k) lg[g() % V] += 10.f + float(g() % 1000) / 100.f;
            for (const Case& c : cases) {
                ie::Ds41SampleParams sp; sp.temperature = c.temp; sp.top_p = c.top_p; sp.min_p = c.min_p; sp.top_k = c.top_k;
                const std::vector<int32_t> recent{int32_t(g() % V), int32_t(g() % V)};
                sp.repeat_penalty = 1.1f;
                for (int s = 0; s < 4; ++s) {
                    auto a = lg, b = lg;
                    uint64_t ra = 1000 + s, rb = 1000 + s;
                    ie::Ds41SampleStats st;
                    const int32_t pa = ie::Ds41Generator::sample_row(a.data(), V, recent, sp, ra);
                    const int32_t pb = ie::Ds41Generator::sample_row(b.data(), V, recent, sp, rb, &st);
                    check(pa == pb && ra == rb && a == b, "stats change neither the pick, the RNG nor the row");
                    if (s) continue;
                    // reference over the penalised row b, the kept set = top_k largest (or all)
                    std::vector<long double> x(b.begin(), b.end());
                    if (c.top_k) { std::vector<long double> y = x; std::nth_element(y.begin(), y.begin() + c.top_k - 1, y.end(), std::greater<>());
                                   const long double cut = y[c.top_k - 1]; std::vector<long double> k; for (auto v : x) if (v >= cut) k.push_back(v); x = k; }
                    long double mx = *std::max_element(x.begin(), x.end()) / c.temp, z = 0;
                    for (auto v : x) z += std::exp(v / c.temp - mx);
                    long double H = 0, p1 = 0, p2 = 0;
                    for (auto v : x) { const long double p = std::exp(v / c.temp - mx) / z; H -= p * std::log(p); if (p > p1) { p2 = p1; p1 = p; } else if (p > p2) p2 = p; }
                    near(st.H, double(H), 1e-6, "H vs long-double reference");
                    near(st.margin, double(p1 - p2), 1e-6, "margin vs long-double reference");
                    ++rows;
                }
            }
        }
        std::printf("reference rows checked: %d\n", rows);
        // the cost of asking (information only): Dream's setting (T 1, top-p 0.95, top-k 0), the same rows with and without
        std::vector<std::vector<float>> R;
        for (int r = 0; r < 40; ++r) {
            std::vector<float> lg(V); std::normal_distribution<float> nd(0.f, 2.5f);
            for (auto& v : lg) v = nd(g);
            for (int k = 0; k < 1 + r % 12; ++k) lg[g() % V] += 10.f + float(g() % 1000) / 100.f;
            R.push_back(std::move(lg));
        }
        ie::Ds41SampleParams sp; sp.temperature = 1.f; sp.top_p = 0.95f;
        double t[2] = {0, 0};
        for (int rep = 0; rep < 3; ++rep)
            for (int with = 0; with < 2; ++with)
                for (const auto& row : R) {
                    auto a = row; uint64_t rng = 5; ie::Ds41SampleStats st;
                    const auto t0 = std::chrono::steady_clock::now();
                    ie::Ds41Generator::sample_row(a.data(), V, {}, sp, rng, with ? &st : nullptr);
                    t[with] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                }
        std::printf("sample_row at Dream's setting: %.3f ms/row without stats, %.3f ms/row with (+%.3f ms)\n",
                    t[0] / (3 * R.size()), t[1] / (3 * R.size()), (t[1] - t[0]) / (3 * R.size()));
    }
    // 3. the window
    {
        ie::VitalsWindow w;
        check(w.window_empty() && !w.active, "fresh window empty, inactive");
        w.add_token(true, 1.0, 0.5); w.add_token(true, 3.0, 0.1); w.add_token(false, 99, -1);
        w.add_draft(5, 3); w.add_draft(2, 0);
        check(w.n == 3 && w.n_H == 2 && w.n_hi == 1, "window counts");
        json j = json::parse(ie::oai::vitals_window_json(w));
        check(j["n"] == 3 && j["n_hi"] == 1 && j["draft"]["offered"] == 7 && j["draft"]["accepted"] == 3, "window json counts");
        near(j["H_mean"].get<double>(), 2.0, 1e-9, "window H_mean");
        near(j["H_max"].get<double>(), 3.0, 1e-9, "window H_max");
        near(j["margin_min"].get<double>(), 0.1, 1e-9, "window margin_min");
        w.reset_window();
        check(w.window_empty() && w.tot_n == 3 && w.tot_offered == 7, "reset keeps the totals");
        w.add_token(false, 0, 0);
        json k = json::parse(ie::oai::vitals_window_json(w));
        check(k["n"] == 1 && k["H_mean"].is_null() && k["H_max"].is_null() && k["margin_min"].is_null(), "greedy window: H null");
        ie::GenerateResult r; r.cached_tokens = 100; r.cache_source = "prompt end"; r.prefill_ms = 12.5; r.restore_ms = 1.25;
        r.completion_tokens = 4; r.decode_ms = 200;
        json s = json::parse(ie::oai::vitals_summary_json(w, r));
        check(s["tokens"] == 4 && s["n_hi"] == 1 && s["draft_offered"] == 7 && s["draft_accepted"] == 3 && s["cached_tokens"] == 100 &&
              s["cache_source"] == "prompt end", "summary fields");
        near(s["H_mean"].get<double>(), 2.0, 1e-9, "summary H_mean (tokens with H only)");
        near(s["decode_tps"].get<double>(), 20.0, 1e-9, "summary decode_tps");
        near(s["prefill_ms"].get<double>(), 12.5, 1e-9, "summary prefill_ms");
        r.cache_source.clear(); r.decode_ms = 0;
        json s2 = json::parse(ie::oai::vitals_summary_json(w, r));
        check(!s2.contains("cache_source") && !s2.contains("decode_tps"), "summary omits what is unavailable");
        // sse_add_field: one more member, the rest unchanged
        const std::string f = ie::oai::chat_chunk_sse("m", "id1", 5, "hi", "");
        const std::string g2 = ie::oai::sse_add_field(f, "ie_vitals", ie::oai::vitals_window_json(w));
        json a = frame_json(f), b = frame_json(g2);
        check(b.contains("ie_vitals") && b["ie_vitals"]["n"] == 1, "added field present");
        b.erase("ie_vitals");
        check(a == b, "the rest of the frame unchanged");
    }
    // 4. the parser
    {
        setenv("IE_SERVE_MAX_TOKENS", "100", 1);
        const std::string msgs = R"("messages":[{"role":"user","content":"x"}])";
        check(!ie::oai::parse_chat_request("{" + msgs + "}").ie_vitals, "absent: off");
        auto f = ie::oai::parse_chat_request("{\"ie_vitals\":false," + msgs + "}");
        check(f.error.empty() && !f.ie_vitals, "false: off");
        auto t = ie::oai::parse_chat_request("{\"ie_vitals\":true,\"seed\":3," + msgs + "}");
        check(t.error.empty() && t.ie_vitals && t.sampling.seed == 3 && t.sampling.vitals == nullptr, "true: on (the parser sets no pointer)");
    }
    std::printf("%s (%d failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
