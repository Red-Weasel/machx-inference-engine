// tools/ds41_forward_test.cpp — V4.1 Phase 7: the whole text forward vs the full-model golden.
// Criteria: docs/deepseek41/18_PHASE7_CRITERIA_2026-09-12.md
#include "ie/deepseek41_forward.hpp"

#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <chrono>
#include <numeric>
#include <vector>

#include "../third_party/nlohmann/json.hpp"

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what, const std::string& detail = "") {
    std::printf("%s%s%s\n", ok ? "[ ok ] " : "[FAIL] ", what.c_str(), detail.empty() ? "" : ("  (" + detail + ")").c_str());
    if (!ok) ++g_fail;
}
template <class T> std::vector<T> rd(const std::string& p, size_t n) {
    std::vector<T> v(n); std::ifstream f(p, std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", p.c_str()); std::exit(1); }
    f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(T)));
    if (size_t(f.gcount()) != n * sizeof(T)) { std::fprintf(stderr, "%s: short\n", p.c_str()); std::exit(1); }
    return v;
}
double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0, s = 0;
    for (size_t i = 0; i < a.size(); ++i) { m = std::max(m, std::fabs(double(a[i]) - double(b[i]))); s = std::max(s, std::fabs(double(b[i]))); }
    return s > 0 ? m / s : 0.0;
}
}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : std::string(std::getenv("HOME")) + "/models/DeepSeek-V4.1-Flash";
    const std::string gd  = argc > 2 ? argv[2] : "/tmp/ds41_golden";     // engram tables live here
    const std::string md  = argc > 3 ? argv[3] : gd + "/model";           // golden_model.py output

    nlohmann::json meta; { std::ifstream f(md + "/m_meta.json"); f >> meta; }
    const uint32_t T = meta["L"], NL = meta["n_layers"], H = meta["dim"], HC = meta["hc_mult"], V = meta["vocab"];
    std::vector<int32_t> ids = meta["ids"].get<std::vector<int32_t>>();
    const std::string want_tok = meta["next_token_text"];
    const int32_t want_id = meta["next_token"];
    std::printf("prompt %s -> %u tokens; golden next token %d %s\n", meta["prompt"].get<std::string>().c_str(), T, want_id, want_tok.c_str());

    ie::DeepSeek41Model m;
    if (auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    ie::Ds41EngramTables tb;
    if (auto e = tb.load(gd); !e.empty()) { std::fprintf(stderr, "tables: %s\n", e.c_str()); return 1; }

    sycl::device dev;
    for (const auto& p : sycl::platform::get_platforms())
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) { dev = d; goto found; }
    std::fprintf(stderr, "no Arc GPU\n"); return 1;
found:
    sycl::queue q(dev, sycl::property::queue::in_order{});
    const bool has_free = dev.has(sycl::aspect::ext_intel_free_memory);
    auto free_vram = [&]() -> uint64_t { return has_free ? dev.get_info<sycl::ext::intel::info::device::free_memory>() : 0; };
    const uint64_t free0 = free_vram();
    std::printf("%s, %.3f GiB free\n", dev.get_info<sycl::info::device::name>().c_str(), double(free0) / (1ull << 30));

    ie::Ds41Forward fwd;
    if (auto e = fwd.init(q, m, tb); !e.empty()) { std::fprintf(stderr, "init: %s\n", e.c_str()); return 1; }

    // per-layer comparison as the layers go by
    std::vector<double> layer_err(NL, -1), engram_err(NL, -1);
    double final_err = -1;   // the collapsed stream the head sees: criterion 1's judgement of layer 39
    const size_t FLAT = size_t(T) * HC * H;
    auto probe = [&](const char* name, uint32_t L, const float* d, size_t n, sycl::queue& pq) {
        std::vector<float> got(n); pq.memcpy(got.data(), d, n * 4).wait();
        if (std::string(name) == "layer")  layer_err[L]  = rel(got, rd<float>(md + "/m_layer_out_" + std::to_string(L) + ".f32", FLAT));
        if (std::string(name) == "engram") engram_err[L] = rel(got, rd<float>(md + "/m_engram_out_" + std::to_string(L) + ".f32", FLAT));
        if (std::string(name) == "final") {
            const auto g = rd<float>(md + "/m_final_collapsed.f32", size_t(T) * H);
            final_err = rel(got, g);
        }
    };
    // per-layer routing goldens (indices, weights, and the 6th-vs-7th score gap per token)
    const uint32_t TK = m.config().n_activated_experts;
    std::vector<std::vector<int32_t>> g_ridx; std::vector<std::vector<float>> g_rw, g_gap;
    const bool have_routing = std::ifstream(md + "/m_route_idx_0.i32").good();
    if (have_routing)
        for (uint32_t L = 0; L < NL; ++L) {
            g_ridx.push_back(rd<int32_t>(md + "/m_route_idx_" + std::to_string(L) + ".i32", size_t(T) * TK));
            g_rw.push_back(rd<float>(md + "/m_route_w_" + std::to_string(L) + ".f32", size_t(T) * TK));
            g_gap.push_back(rd<float>(md + "/m_route_gap_" + std::to_string(L) + ".f32", T));
        }
    const bool force = argc > 4 && std::string(argv[4]) == "force";
    if (force && !have_routing) { std::fprintf(stderr, "force needs routing goldens\n"); return 1; }
    std::printf("mode: %s\n", force ? "FORCED routing (diagnostic: expert arithmetic only)" : "engine's own router");
    std::vector<float> logits;
    const auto t0 = std::chrono::steady_clock::now();
    if (auto e = fwd.prefill(ids.data(), T, logits, probe, have_routing ? &g_ridx : nullptr, have_routing ? &g_rw : nullptr, force,
                             have_routing ? &g_gap : nullptr);
        !e.empty()) { std::fprintf(stderr, "prefill: %s\n", e.c_str()); return 1; }
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    std::printf("\n%-5s %-4s %-12s %8s %8s %6s %8s %5s %9s %9s  %s\n", "layer", "r", "kind", "engram", "stream", "exps", "MiB", "rdiff", "flip gap", "min gap", "ms");
    size_t bad = 0;
    // AMENDED (disclosed in 18_PHASE7_CRITERIA): with the engine's own router, the layer-by-
    // layer comparison is only meaningful UP TO the first routing flip -- after it the golden's
    // routing was computed on a different stream, so later "differences" compare different
    // computations. So in own-router mode the per-layer bar applies through the first flip,
    // the first flip must be a near-tie (golden margin under the band), and the top-1 must hold.
    int32_t first_flip = -1;
    for (uint32_t L = 0; L < NL; ++L) if (fwd.stats()[L].routing_diff) { first_flip = int32_t(L); break; }
    for (uint32_t L = 0; L < NL; ++L) {
        const auto& k = m.layers()[L].kind; const auto& s = fwd.stats()[L];
        const char* kind = k.is_kv_source ? "kv+idx src" : k.is_index_source ? "idx src" : k.compress_ratio ? "consumer" : "window";
        // Layer 39 is judged on the COLLAPSED stream the head sees (criterion 1), not on its raw
        // stream with a slack bar -- the gate found the old 1e-1 unused (raw 1.46e-3).
        const double err = L == 39 ? final_err : layer_err[L];
        const double tol = L < 20 ? 5e-3 : 1e-2;
        const bool judged = force || first_flip < 0 || int32_t(L) < first_flip;   // own-router: only up to the first flip
        const bool ok = !judged || (err <= tol && (engram_err[L] < 0 || engram_err[L] <= 5e-3));
        if (!ok) ++bad;
        const std::string eng = engram_err[L] < 0 ? "-" : std::to_string(engram_err[L]).substr(0, 8);
        float min_gap = 0.f;
        if (have_routing) { min_gap = 1e30f; for (float g : g_gap[L]) min_gap = std::min(min_gap, g); }
        std::printf("%s %-4u %-4u %-12s %8s %8.2e %6u %8.0f %5u %9.2e %9.2e  %.0f%s\n", ok ? "  " : (judged ? "!!" : ".."), L, k.compress_ratio, kind,
                    eng.c_str(), layer_err[L], s.experts_uploaded, double(s.expert_bytes + s.dense_bytes) / (1 << 20),
                    s.routing_diff, double(s.flip_gap_max), double(min_gap), s.ms,
                    ok ? "" : (judged ? "   <-- over tolerance" : "   (after first flip: informational)"));
    }
    std::printf("       final collapsed stream (what the head sees; layer 39's judgement) rel %.3e\n", final_err);
    check(bad == 0, force ? "FORCED routing: every layer's output stream within its tolerance (5e-3 to L19, 1e-2 after; L39 on the collapsed stream)"
                          : "own router: every layer's stream within tolerance up to the first routing flip",
          std::to_string(bad) + " layers over");
    if (have_routing)   // criterion 5: at layer 0 the input IS the embedding, so nothing can excuse a mismatch
        check(fwd.stats()[0].routing_diff == 0, "layer 0 routing equals the golden's for every token (no accumulated error to excuse it)",
              std::to_string(fwd.stats()[0].routing_diff) + " tokens differ");
    if (!force && have_routing) {
        if (first_flip < 0) std::printf("       own router: NO routing flip in 40 layers\n");
        else {
            const float gap = fwd.stats()[first_flip].flip_gap_max;
            // band: the golden's own margin must be inside the accumulated stream error (~5e-4 of
            // scale by layer 7); 3e-3 in sqrtsoftplus+bias units is ~6x that, and a flip at a
            // margin above it would mean the router disagrees with the reference on a clear call
            check(gap < 3e-3, "own router: the FIRST routing flip (layer " + std::to_string(first_flip) + ") is at a near-tie",
                  "golden 6th-vs-7th margin " + std::to_string(gap) + ", " + std::to_string(fwd.stats()[first_flip].routing_diff) + " token(s)");
        }
    }
    // Not a check: the runtime refuses above the threshold (an error return, exercised by the gate's
    // mutant), so a check here could never fail. Stated as the fact it is.
    std::printf("       candidate top-k: no-op at this length (T=%u; the runtime refuses above %u compressed positions) -- NAMED GAP\n",
                T, m.config().candidate_topk_blocks * m.config().candidate_block_size);

    // logits + the token
    const auto g_logits = rd<float>(md + "/m_logits.f32", size_t(T) * V);
    const double lr = rel(logits, g_logits);
    if (force || first_flip < 0) check(lr < 1e-2, "logits within 1e-2 of the golden over [T, vocab]", "rel " + std::to_string(lr));
    else std::printf("       logits rel %.3e vs the golden (informational after a routing flip)\n", lr);
    const float* last = logits.data() + size_t(T - 1) * V; const float* g_last = g_logits.data() + size_t(T - 1) * V;
    std::vector<int32_t> top(V); std::iota(top.begin(), top.end(), 0);
    std::partial_sort(top.begin(), top.begin() + 5, top.end(), [&](int a, int b) { return last[a] > last[b]; });
    std::vector<int32_t> gtop(V); std::iota(gtop.begin(), gtop.end(), 0);
    std::partial_sort(gtop.begin(), gtop.begin() + 5, gtop.end(), [&](int a, int b) { return g_last[a] > g_last[b]; });
    std::printf("       engine top-5 ids: %d %d %d %d %d  (logit %.3f)   golden: %d %d %d %d %d  (%.3f)\n",
                top[0], top[1], top[2], top[3], top[4], last[top[0]], gtop[0], gtop[1], gtop[2], gtop[3], gtop[4], g_last[gtop[0]]);
    check(top[0] == want_id, "TOP-1 TOKEN IS " + want_tok, "engine " + std::to_string(top[0]) + " vs golden " + std::to_string(want_id));
    std::vector<int32_t> s5(top.begin(), top.begin() + 5), g5(gtop.begin(), gtop.begin() + 5);
    std::sort(s5.begin(), s5.end()); std::sort(g5.begin(), g5.end());
    if (force || first_flip < 0) check(s5 == g5, "top-5 SET equals the golden's");
    else std::printf("       top-5 set %s the golden's (informational after a routing flip)\n", s5 == g5 ? "equals" : "differs from");

    q.wait_and_throw();
    const uint64_t free1 = free_vram();
    check(has_free && (int64_t(free0) - int64_t(free1)) < int64_t(128ull << 20), "free VRAM returns to start within 128 MiB",
          std::to_string(double(int64_t(free0) - int64_t(free1)) / (1 << 20)).substr(0, 7) + " MiB unreturned");
    uint64_t total_bytes = fwd.head_bytes(); for (const auto& s : fwd.stats()) total_bytes += s.expert_bytes + s.dense_bytes;
    std::printf("       whole forward: %.1f s, %.2f GiB of weights moved (per-layer streaming, no residency)\n", secs, double(total_bytes) / (1ull << 30));
    const std::string verdict = g_fail ? "FORWARD TEST: " + std::to_string(g_fail) + " FAILURE(S)"
                                       : "FORWARD TEST: PASS — the engine says " + want_tok;
    std::printf("\n%s\n", verdict.c_str());
    return g_fail ? 1 : 0;
}
