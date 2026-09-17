// tools/ds41_resident_test.cpp — V4.1 Phase 8: the resident runtime. Correctness on the golden
// prompt (own router, judged to the first flip, top-1 " Berlin"; then FORCED routing, judged at
// every one of the 40 layers so the second card is checked too), bytes by tier per forward with a
// warm second forward, MemAvailable accounting for the pinned tier, then prefill tok/s at 512 and
// 2048 real tokens with the MoE's phase breakdown and the hit rate by selections for the ranking
// in use, for index order (the control) and for a held-out profile. Every Arc card found is used,
// each on its own single-device context, layers split contiguously across them.
//   usage: ie-ds41-resident-test <model> <golden_dir> <golden_dir>/model [profile_out] [ranking_in] [corpus_ids]
//   corpus_ids: int32 token ids, >= 65,536 of them, disjoint from the measured prompts. When given,
//   the run first prefills the corpus in max_tokens chunks with the engine's own router and writes
//   THAT profile to profile_out (held-out, criterion 3); without it, profile_out receives the
//   in-sample profile of this run's own prompts, labelled as such.
#include "ie/deepseek41_forward.hpp"
#include "ie/expert_stream.hpp"

#include <algorithm>
#include <array>
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
    if (size_t(f.gcount()) != n * sizeof(T)) { std::fprintf(stderr, "%s: short\n", p.c_str()); std::exit(1); }
    return v; }
double rel(const float* a, const float* b, size_t n) {
    double m = 0, s = 0; for (size_t i = 0; i < n; ++i) { m = std::max(m, std::fabs(double(a[i]) - double(b[i]))); s = std::max(s, std::fabs(double(b[i]))); }
    return s > 0 ? m / s : 0; }
double rel(const std::vector<float>& a, const std::vector<float>& b) { return rel(a.data(), b.data(), a.size()); }
uint64_t mem_available_kib() { std::ifstream f("/proc/meminfo"); std::string k; uint64_t v; std::string u;
    while (f >> k >> v >> u) if (k == "MemAvailable:") return v; return 0; }
double gib(uint64_t b) { return double(b) / double(1ull << 30); }
std::string f3(double v) { char b[32]; std::snprintf(b, sizeof b, "%.3e", v); return b; }
// most-selected first, per layer -- the order write_profile stores
std::vector<std::vector<uint32_t>> rank_from(const std::vector<std::vector<uint64_t>>& counts) {
    std::vector<std::vector<uint32_t>> r(counts.size());
    for (size_t L = 0; L < counts.size(); ++L) {
        r[L].resize(counts[L].size()); std::iota(r[L].begin(), r[L].end(), 0u);
        std::stable_sort(r[L].begin(), r[L].end(), [&](uint32_t a, uint32_t b) { return counts[L][a] > counts[L][b]; });
    }
    return r;
}
std::vector<int32_t> top5(const float* row, uint32_t V) {
    std::vector<int32_t> ids(V); std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + 5, ids.end(), [&](int32_t a, int32_t b) { return row[a] > row[b]; });
    ids.resize(5); return ids;
}
}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : std::string(std::getenv("HOME")) + "/models/DeepSeek-V4.1-Flash";
    const std::string gd  = argc > 2 ? argv[2] : "/tmp/ds41_golden";
    const std::string md  = argc > 3 ? argv[3] : gd + "/model";
    const std::string prof_out  = argc > 4 ? argv[4] : "";
    const std::string rank_in   = argc > 5 ? argv[5] : "";
    const std::string corpus_in = argc > 6 ? argv[6] : "";

    nlohmann::json meta; { std::ifstream f(md + "/m_meta.json"); f >> meta; }
    const uint32_t T = meta["L"], NL = meta["n_layers"], H = meta["dim"], HC = meta["hc_mult"], V = meta["vocab"];
    std::vector<int32_t> ids = meta["ids"].get<std::vector<int32_t>>();
    const int32_t want_id = meta["next_token"]; const std::string want_tok = meta["next_token_text"];

    ie::DeepSeek41Model m;
    if (auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    ie::Ds41EngramTables tb;
    if (auto e = tb.load(gd); !e.empty()) { std::fprintf(stderr, "tables: %s\n", e.c_str()); return 1; }
    const auto& c = m.config();
    const uint32_t E = c.n_routed_experts, TK = c.n_activated_experts;

    // every Arc card, each queue on an EXPLICIT single-device context: sycl::queue(device)
    // binds to the platform default context and a two-card box then mirrors every VRAM
    // allocation into host RAM (the 83554c2 lesson). Level Zero only: the OpenCL platform
    // lists the same two cards again, without the free-memory aspect this test budgets from.
    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d);
    }
    if (devs.empty()) { std::fprintf(stderr, "no Arc GPU\n"); return 1; }
    std::vector<std::unique_ptr<sycl::queue>> queues;
    std::vector<sycl::queue*> qs;
    for (const auto& d : devs) {
        // IE_QUEUE_PROFILING=1: the attention kernel's GPU timestamps land in the per-layer stage dump
        if (std::getenv("IE_QUEUE_PROFILING"))
            queues.push_back(std::make_unique<sycl::queue>(sycl::context(d), d, sycl::property_list{sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}}));
        else
            queues.push_back(std::make_unique<sycl::queue>(sycl::context(d), d, sycl::property_list{sycl::property::queue::in_order{}}));
        qs.push_back(queues.back().get());
    }
    const uint32_t NCARD = uint32_t(devs.size());
    auto free_vram = [&](uint32_t ci) -> uint64_t { return devs[ci].get_info<sycl::ext::intel::info::device::free_memory>(); };
    std::vector<uint64_t> f0(NCARD);
    for (uint32_t ci = 0; ci < NCARD; ++ci) { f0[ci] = free_vram(ci);
        std::printf("card %u: %s, %.2f GiB VRAM free\n", ci, devs[ci].get_info<sycl::info::device::name>().c_str(), gib(f0[ci])); }
    const uint64_t ma0 = mem_available_kib();
    std::printf("MemAvailable %.1f GiB\n", double(ma0) / (1u << 20));

    std::vector<std::vector<uint32_t>> ranking;
    if (!rank_in.empty()) {
        if (auto e = ie::ds4_expert_priority_read_layers(rank_in, E, NL, ranking); !e.empty()) { std::fprintf(stderr, "ranking: %s\n", e.c_str()); return 1; }
        std::printf("ranking in use: %s\n", rank_in.c_str());
    } else std::printf("ranking in use: index order (the control)\n");
    std::vector<std::vector<uint32_t>> index_order(NL, std::vector<uint32_t>(E));
    for (auto& r : index_order) std::iota(r.begin(), r.end(), 0u);
    const auto& rk_use = ranking.empty() ? index_order : ranking;

    auto fwd_p = std::make_unique<ie::Ds41Forward>(); ie::Ds41Forward& fwd = *fwd_p; ie::Ds41Forward::ResidentOptions opt;
    // IE_DS41_VRAM_RESERVE_GIB: the per-card headroom kept free for scratch (default 3). Fewer
    // static slots, but a test of whether the driver was evicting under pressure.
    if (const char* r = std::getenv("IE_DS41_VRAM_RESERVE_GIB")) { opt.vram_reserve = uint64_t(std::atof(r) * double(1ull << 30)); std::printf("vram_reserve %s GiB per card\n", r); }
    const auto ti = std::chrono::steady_clock::now();
    if (auto e = fwd.init_resident(qs, m, tb, ranking, opt); !e.empty()) { std::fprintf(stderr, "init_resident: %s\n", e.c_str()); return 1; }
    fwd.set_logits_last_only(true);            // this test reads the last row only (top-1, top-5); saves 1 GiB of VRAM + copy at pp2048
    const double init_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - ti).count();
    const uint64_t ma1 = mem_available_kib();
    std::printf("resident in %.0f s on %u card(s): dense %.2f GiB + experts %.2f GiB VRAM, %.1f GiB pinned host; MemAvailable %.1f -> %.1f GiB\n",
                init_s, NCARD, gib(fwd.dense_bytes()), gib(fwd.expert_vram_bytes()), gib(fwd.expert_pinned_bytes()),
                double(ma0) / (1u << 20), double(ma1) / (1u << 20));
    std::vector<ie::Ds41Forward::CardInfo> cards(NCARD);
    std::vector<uint32_t> owner(NL, 0);
    for (uint32_t ci = 0; ci < NCARD; ++ci) {
        cards[ci] = fwd.card_info(ci); const auto& info = cards[ci];
        for (uint32_t L = info.first_layer; L < info.first_layer + info.n_layers; ++L) owner[L] = ci;
        std::printf("       card %u: layers %u..%u, dense %.2f GiB, experts %.2f GiB VRAM (%u static + %u stream per layer), %u pinned per layer = %.1f GiB; VRAM free %.2f -> %.2f GiB\n",
                    ci, info.first_layer, info.first_layer + info.n_layers - 1, gib(info.dense_bytes), gib(info.expert_vram_bytes),
                    info.n_static, info.n_stream, info.n_pinned, gib(info.pinned_bytes), gib(f0[ci]), gib(free_vram(ci)));
    }
    const double pin_drop = double(ma0 - ma1) / (1u << 20);
    check(std::fabs(pin_drop - gib(fwd.expert_pinned_bytes())) < 0.05 * gib(fwd.expert_pinned_bytes()) + 2.0,
          "MemAvailable dropped by the pinned tier's bytes (within 5% + 2 GiB)",
          std::to_string(pin_drop).substr(0, 6) + " GiB dropped vs " + std::to_string(gib(fwd.expert_pinned_bytes())).substr(0, 6) + " pinned");
    // criterion 3: the placement IS the ranking -- rank [0, static) in VRAM, then pinned, then mmap
    {
        uint32_t bad = 0;
        const uint32_t nparts = fwd.expert_parallel() == 1 ? uint32_t(cards.size()) : 1u;   // docs/38: rank position i -> card i % n, its position i / n
        for (uint32_t L = 0; L < NL; ++L) {
            for (uint32_t r = 0; r < E; ++r) {
                const auto& ci = cards[nparts > 1 ? r % nparts : owner[L]]; const uint32_t rr = r / nparts;
                const uint32_t want = rr < ci.n_static ? 0u : rr < ci.n_static + ci.n_pinned ? 1u : 2u;
                if (fwd.expert_tier(L, rk_use[L][r]) != want) ++bad;
            }
        }
        check(bad == 0, nparts > 1 ? "expert placement follows the ranking in use per card (rank position i on card i % n): static, pinned, mmap, every layer"
                                   : "expert placement follows the ranking in use: rank [0,static) VRAM, then pinned, then mmap, every layer",
              std::to_string(bad) + " (layer, expert) pairs misplaced");
    }
    // where a selection count lands, by tier: `actual` asks the runtime; otherwise by rank under each card's slots
    auto shares = [&](const std::vector<std::vector<uint64_t>>& cnt, const std::vector<std::vector<uint32_t>>& rk, bool actual) {
        uint64_t by[3] = {0, 0, 0}, tot = 0;
        for (uint32_t L = 0; L < NL; ++L) {
            const auto& ci = cards[owner[L]];
            for (uint32_t r = 0; r < E; ++r) {
                const uint32_t e = rk[L][r]; const uint64_t n = cnt[L][e]; if (!n) continue;
                const uint32_t t = actual ? fwd.expert_tier(L, e) : (r < ci.n_static ? 0u : r < ci.n_static + ci.n_pinned ? 1u : 2u);
                by[t] += n; tot += n;
            }
        }
        std::array<double, 3> s{}; if (tot) for (int i = 0; i < 3; ++i) s[i] = 100.0 * double(by[i]) / double(tot);
        return s;
    };

    std::vector<float> logits;
    auto run = [&](const std::vector<int32_t>& in, const ie::Ds41Forward::Probe& pr,
                   const std::vector<std::vector<int32_t>>* ridx, const std::vector<std::vector<float>>* rw,
                   const std::vector<std::vector<float>>* gap, bool force) -> double {
        const auto t0 = std::chrono::steady_clock::now();
        if (auto e = fwd.prefill(in.data(), uint32_t(in.size()), logits, pr, ridx, rw, force, gap); !e.empty()) {
            std::fprintf(stderr, "prefill: %s\n", e.c_str()); std::exit(1); }
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    // ---- held-out routing profile (criterion 3) ---------------------------------------------
    std::vector<std::vector<uint32_t>> held_rank; bool have_held = false;
    if (!corpus_in.empty()) {
        if (prof_out.empty()) { std::fprintf(stderr, "corpus_ids needs profile_out\n"); return 1; }
        std::ifstream f(corpus_in, std::ios::binary | std::ios::ate);
        if (!f) { std::fprintf(stderr, "cannot open %s\n", corpus_in.c_str()); return 1; }
        const size_t n_all = size_t(f.tellg()) / 4; f.seekg(0);
        std::vector<int32_t> corp(n_all); f.read(reinterpret_cast<char*>(corp.data()), std::streamsize(n_all * 4));
        const uint32_t CH = opt.max_tokens, nch = uint32_t(n_all / CH), used = nch * CH;
        std::printf("\n=== held-out routing profile: %s -> %u tokens in %u chunks of %u (own router) ===\n", corpus_in.c_str(), used, nch, CH);
        check(used >= 65536, "profile corpus has >= 64K tokens", std::to_string(used) + " tokens");
        fwd.reset_profile();
        const auto tp = std::chrono::steady_clock::now();
        for (uint32_t ch = 0; ch < nch; ++ch) {
            std::vector<int32_t> chunk(corp.begin() + size_t(ch) * CH, corp.begin() + size_t(ch + 1) * CH);
            run(chunk, {}, nullptr, nullptr, nullptr, false);
        }
        const double ps = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count();
        std::printf("       profiled in %.0f s (%.1f tok/s)\n", ps, used / ps);
        held_rank = rank_from(fwd.profile()); have_held = true;
        const auto e = fwd.write_profile(prof_out, "V4.1 held-out own-router profile: " + corpus_in + ", " + std::to_string(used) + " tokens");
        check(e.empty(), "held-out profile written -> " + prof_out, e);
        fwd.reset_profile();
    }

    // ---- correctness on the golden prompt, own router ---------------------------------------
    std::vector<std::vector<int32_t>> g_ridx; std::vector<std::vector<float>> g_rw, g_gap;
    for (uint32_t L = 0; L < NL; ++L) {
        g_ridx.push_back(rd<int32_t>(md + "/m_route_idx_" + std::to_string(L) + ".i32", size_t(T) * TK));
        g_rw.push_back(rd<float>(md + "/m_route_w_" + std::to_string(L) + ".f32", size_t(T) * TK));
        g_gap.push_back(rd<float>(md + "/m_route_gap_" + std::to_string(L) + ".f32", T));
    }
    std::vector<double> layer_err(NL, -1);
    auto probe_into = [&](std::vector<double>& dst) {
        return [&](const char* name, uint32_t L, const float* d, size_t n, sycl::queue& pq) {
            if (std::string(name) != "layer") return;
            std::vector<float> got(n); pq.memcpy(got.data(), d, n * 4).wait();
            dst[L] = rel(got, rd<float>(md + "/m_layer_out_" + std::to_string(L) + ".f32", size_t(T) * HC * H));
        };
    };
    std::printf("\n=== correctness: golden prompt, own router, Q8 grouped path ===\n");
    const double s1 = run(ids, probe_into(layer_err), &g_ridx, &g_rw, &g_gap, false);
    int32_t first_flip = -1; for (uint32_t L = 0; L < NL; ++L) if (fwd.stats()[L].routing_diff) { first_flip = int32_t(L); break; }
    double worst = 0; for (uint32_t L = 0; L < NL; ++L) if (first_flip < 0 || int32_t(L) < first_flip) worst = std::max(worst, layer_err[L]);
    std::printf("       layers 0..%d (up to the first flip): worst stream error %.3e; first flip at layer %d\n",
                first_flip < 0 ? int(NL) - 1 : first_flip - 1, worst, first_flip);
    check(worst < 8e-3, "own router: every layer up to the first routing flip within the Q8-path bar (8e-3)", "worst " + std::to_string(worst));
    if (first_flip >= 0) check(fwd.stats()[first_flip].flip_gap_max < 3e-3, "the first flip is at a near-tie",
                               "layer " + std::to_string(first_flip) + ", golden margin " + std::to_string(fwd.stats()[first_flip].flip_gap_max));
    const float* last = logits.data() + (logits.size() - V);
    const int32_t top1 = int32_t(std::max_element(last, last + V) - last);
    check(top1 == want_id, "TOP-1 TOKEN IS " + want_tok, "engine " + std::to_string(top1) + " vs golden " + std::to_string(want_id));
    uint64_t mm_bytes = 0, mm_exp = 0, st_exp = 0, pn_exp = 0, dense_moved = 0;
    for (const auto& s : fwd.stats()) { mm_bytes += s.bytes_mmap; mm_exp += s.experts_mmap; st_exp += s.experts_static; pn_exp += s.experts_pinned; dense_moved += s.dense_bytes; }
    std::printf("       forward %.2f s; occupied experts by tier: %llu static, %llu pinned, %llu mmap; %.2f GiB mmap->VRAM; dense bytes moved: %llu\n",
                s1, (unsigned long long)st_exp, (unsigned long long)pn_exp, (unsigned long long)mm_exp, gib(mm_bytes), (unsigned long long)dense_moved);
    check(dense_moved == 0, "dense bytes moved per forward is 0 (resident), from the layer stats");
    // warm second forward: identical logits, no dense movement, the pinned tier's stream slots already hold the hits
    std::vector<float> l1 = logits;
    const double s2 = run(ids, {}, nullptr, nullptr, nullptr, false);
    check(l1 == logits, "a warm second forward is bit-identical", std::to_string(s2).substr(0, 5) + " s");

    // ---- FORCED routing: the golden's experts, the engine's arithmetic, judged at all 40 layers.
    // The own-router judgement stops at the first flip (layer 7) and so says nothing per layer
    // about the second card (Phase 8 gate, finding 3). Bars: top-5 ORDER is discrete; the logits
    // bar 5e-2 and the boundary factor 1.5 are REGRESSION bars set after the gate's probe measured
    // 4.39e-2 and no jump at 19->20 (8.36e-3 -> 8.20e-3), not derived from the Q8 path's floor.
    std::printf("\n=== correctness: golden prompt, FORCED routing, every layer ===\n");
    std::vector<double> f_err(NL, -1);
    const double sf = run(ids, probe_into(f_err), &g_ridx, &g_rw, &g_gap, true);
    std::printf("       per-layer stream error vs the golden (%.2f s):", sf);
    for (uint32_t L = 0; L < NL; ++L) std::printf("%s%2u:%.2e", L % 8 ? "  " : "\n         ", L, f_err[L]);
    std::printf("\n");
    // Per-card worst bars (gate note): the Q8 path's error grows ~1.06x per layer, so one bar
    // for all 40 would let a slow card-1 regression through. Regression bars from the measured
    // 9.02e-3 (card 0, layer 18) and 2.76e-2 (card 1, layer 38), each with ~25% headroom.
    const double card_bar[2] = {1.2e-2, 3.5e-2};
    for (uint32_t ci = 0; ci < NCARD; ++ci) {
        double w = 0; uint32_t at = 0;
        for (uint32_t L = cards[ci].first_layer; L < cards[ci].first_layer + cards[ci].n_layers; ++L) if (f_err[L] > w) { w = f_err[L]; at = L; }
        const double bar = ci < 2 ? card_bar[ci] : card_bar[1];
        check(w < bar, "FORCED routing: card " + std::to_string(ci) + " worst layer error under its regression bar " + f3(bar),
              f3(w) + " at layer " + std::to_string(at));
    }
    for (uint32_t ci = 1; ci < NCARD; ++ci) {
        const uint32_t b = cards[ci].first_layer;
        check(f_err[b] <= 1.5 * f_err[b - 1], "no error jump across the card boundary (layer " + std::to_string(b) + " <= 1.5 x layer " + std::to_string(b - 1) + ")",
              f3(f_err[b - 1]) + " -> " + f3(f_err[b]));
    }
    const auto g_logits = rd<float>(md + "/m_logits.f32", size_t(T) * V);
    const double lr = rel(logits.data() + (logits.size() - V), g_logits.data() + size_t(T - 1) * V, V);   // last row (this test keeps one)
    const auto e5 = top5(logits.data() + (logits.size() - V), V), g5 = top5(g_logits.data() + size_t(T - 1) * V, V);
    std::printf("       engine top-5: %d %d %d %d %d   golden: %d %d %d %d %d\n", e5[0], e5[1], e5[2], e5[3], e5[4], g5[0], g5[1], g5[2], g5[3], g5[4]);
    check(e5 == g5, "FORCED routing: top-5 ORDER equals the golden's");
    check(lr < 5e-2, "FORCED routing: logits within 5e-2 of the golden over [T, vocab] (regression bar, Q8 path)", "rel " + f3(lr));

    // ---- prefill throughput, real text ---------------------------------------------------------
    std::printf("\n=== prefill throughput (%u card(s)) ===\n", NCARD);
    for (uint32_t n : {512u, 2048u}) {
        const std::string p = gd + "/pp_ids_" + std::to_string(n) + ".i32";
        if (!std::ifstream(p).good()) { std::printf("       (no %s)\n", p.c_str()); continue; }
        const auto in = rd<int32_t>(p, n);
        const auto c0 = fwd.profile();
        const double cold = run(in, {}, nullptr, nullptr, nullptr, false);
        auto pass = fwd.profile();
        for (uint32_t L = 0; L < NL; ++L) for (uint32_t e = 0; e < E; ++e) pass[L][e] -= c0[L][e];
        uint64_t mmb = 0, mme = 0, pne = 0, ste = 0; double moe_ms = 0, prep = 0, mm = 0, pk = 0, grp = 0, rdm = 0, pmm = 0, mmg = 0, ovl = 0, tail = 0, spawn = 0, join = 0;
        for (const auto& s : fwd.stats()) { mmb += s.bytes_mmap; mme += s.experts_mmap; pne += s.experts_pinned; ste += s.experts_static; moe_ms += s.moe_ms;
                                            prep += s.moe_prep_ms; mm += s.moe_mmap_ms; pk += s.moe_mmap_pack_ms; grp += s.moe_groups_ms;
                                            rdm += s.moe_mmap_read_ms; pmm += s.moe_mmap_permute_ms; mmg += s.moe_mmap_group_ms; tail += s.moe_tail_ms; spawn += s.moe_spawn_ms; join += s.moe_join_ms;
                                            ovl += std::max(s.moe_mmap_ms, s.moe_groups_ms); }
        double eng = 0, att = 0, fpre = 0, shd = 0, lay = 0, mcall = 0;      // the dense body by stage, COLD pass (same stats as above)
        for (const auto& s : fwd.stats()) { eng += s.engram_ms; att += s.attn_ms; fpre += s.ffn_pre_ms; shd += s.shared_ms; lay += s.ms; mcall += s.moe_call_ms; }
        const double fwd_prep_ms_cold = fwd.prep_ms(), fwd_head_ms_cold = fwd.head_ms();
        auto dump_stages = [&](const std::string& tag) {          // per-layer stage ms, for the cold/warm question
            std::ofstream f(gd + "/stages_pp" + std::to_string(n) + "_" + tag + ".txt");
            f << "L attn ffn_pre moe_call moe_prep spawn groups join mmap_group tail fill shared engram total attn_kq attn_kexec\n";
            for (uint32_t L = 0; L < NL; ++L) { const auto& s = fwd.stats()[L];
                f << L << ' ' << s.attn_ms << ' ' << s.ffn_pre_ms << ' ' << s.moe_call_ms << ' ' << s.moe_prep_ms << ' ' << s.moe_spawn_ms << ' ' << s.moe_groups_ms << ' '
                  << s.moe_join_ms << ' ' << s.moe_mmap_group_ms << ' ' << s.moe_tail_ms << ' ' << s.moe_mmap_ms << ' ' << s.shared_ms << ' ' << s.engram_ms << ' ' << s.ms
                  << ' ' << s.attn_kernel_queue_ms << ' ' << s.attn_kernel_exec_ms << "\n"; }
        };
        dump_stages("cold");
        const double warm = run(in, {}, nullptr, nullptr, nullptr, false);
        const double c_prep = fwd_prep_ms_cold, c_head = fwd_head_ms_cold;
        dump_stages("warm");
        double w_eng = 0, w_att = 0, w_fpre = 0, w_shd = 0, w_lay = 0, w_mcall = 0, w_moe = 0;   // the same, WARM pass
        for (const auto& s : fwd.stats()) { w_eng += s.engram_ms; w_att += s.attn_ms; w_fpre += s.ffn_pre_ms; w_shd += s.shared_ms; w_lay += s.ms; w_mcall += s.moe_call_ms; w_moe += s.moe_ms; }
        std::printf("       pp%-5u cold %.2f s = %.1f tok/s | warm %.2f s = %.1f tok/s | occupied experts/forward: %llu static %llu pinned %llu mmap, %.1f GiB mmap->VRAM\n",
                    n, cold, n / cold, warm, n / warm, (unsigned long long)ste, (unsigned long long)pne, (unsigned long long)mme, gib(mmb));
        std::printf("       pp%-5u MoE %.0f ms of the cold pass = prep %.0f + spawn %.0f + groups %.0f + join %.0f + mmap group %.0f + tail %.0f  [sum %.0f]; the reader's fill leg %.0f (host %.0f: reads %.0f + permute %.0f per reader) ran beside groups + join\n",
                    n, moe_ms, prep, spawn, grp, join, mmg, tail, prep + spawn + grp + join + mmg + tail, mm, pk, rdm, pmm);
        const auto a = shares(pass, rk_use, true), i0 = shares(pass, index_order, false);
        std::printf("       pp%-5u COLD: layers %.0f ms = engram %.0f + attention %.0f + ffn-pre (hc, norm, router) %.0f + MoE call %.0f (tier-internal %.0f) + shared %.0f + rest %.0f; outside the layers %.0f = host prep (hashes, gathers, embed) %.0f + head %.0f + other %.0f\n",
                    n, lay, eng, att, fpre, mcall, moe_ms, shd, lay - eng - att - fpre - mcall - shd, cold * 1000 - lay, c_prep, c_head, cold * 1000 - lay - c_prep - c_head);
        std::printf("       pp%-5u WARM: layers %.0f ms = engram %.0f + attention %.0f + ffn-pre %.0f + MoE call %.0f (tier-internal %.0f) + shared %.0f + rest %.0f; outside the layers %.0f = host prep %.0f + head %.0f + other %.0f\n",
                    n, w_lay, w_eng, w_att, w_fpre, w_mcall, w_moe, w_shd, w_lay - w_eng - w_att - w_fpre - w_mcall - w_shd, warm * 1000 - w_lay, fwd.prep_ms(), fwd.head_ms(), warm * 1000 - w_lay - fwd.prep_ms() - fwd.head_ms());
        std::printf("       pp%-5u selections by tier, VRAM/pinned/mmap %%: in use %.1f/%.1f/%.1f | index order %.1f/%.1f/%.1f",
                    n, a[0], a[1], a[2], i0[0], i0[1], i0[2]);
        if (have_held) { const auto h = shares(pass, held_rank, false); std::printf(" | held-out profile %.1f/%.1f/%.1f", h[0], h[1], h[2]); }
        std::printf("\n");
    }
    if (!prof_out.empty() && !have_held) {
        if (auto e = fwd.write_profile(prof_out, "V4.1 own-router prefill profile, IN-SAMPLE (golden prompt x2 + pp512 x2 + pp2048 x2)"); !e.empty()) std::fprintf(stderr, "profile: %s\n", e.c_str());
        else std::printf("       wrote IN-SAMPLE routing profile -> %s\n", prof_out.c_str());
    }
    // unload in three steps, free VRAM read after each: what free_resident returns, what only the
    // destructor returns, and what the runtime keeps (its pool) until the process ends
    std::vector<uint64_t> f_before(NCARD), f_after(NCARD), f_dtor(NCARD);
    for (uint32_t ci = 0; ci < NCARD; ++ci) f_before[ci] = free_vram(ci);
    fwd.free_resident();
    for (auto& qp : queues) qp->wait_and_throw();
    for (uint32_t ci = 0; ci < NCARD; ++ci) f_after[ci] = free_vram(ci);
    fwd_p.reset();
    for (auto& qp : queues) qp->wait_and_throw();
    for (uint32_t ci = 0; ci < NCARD; ++ci) f_dtor[ci] = free_vram(ci);
    const uint64_t ma2 = mem_available_kib();
    for (uint32_t ci = 0; ci < NCARD; ++ci) {
        const uint64_t f2 = f_dtor[ci];
        std::printf("       card %u free VRAM: start %.3f GiB, before unload %.3f, after free_resident %.3f, after destructor %.3f GiB\n",
                    ci, gib(f0[ci]), gib(f_before[ci]), gib(f_after[ci]), gib(f_dtor[ci]));
        // What stays after a full unload is the runtime's per-process residue -- oneDNN's per-shape
        // primitive cache and workspaces, the JIT'd kernel bundles, the USM pool -- measured at
        // ~115 MiB with the 12-token shapes alone and 220-270 MiB with the pp512/pp2048 shapes
        // (docs/deepseek41/23, runs U-W), never growing across passes of the same shapes. The bar
        // covers that residue; the engine's OWN leaks are what the second cycle below measures.
        check(int64_t(f0[ci]) - int64_t(f2) < int64_t(384ull << 20), "VRAM returned on unload, card " + std::to_string(ci) + " (within the runtime's measured residue)",
              std::to_string(double(int64_t(f0[ci]) - int64_t(f2)) / (1 << 20)).substr(0, 7) + " MiB unreturned");
    }
    // IE_DS41_TWO_CYCLES=1: load again, run the golden prompt, unload again. With the runtime's
    // caches already warm, any VRAM the second unload fails to return is the engine's.
    if (std::getenv("IE_DS41_TWO_CYCLES")) {
        std::printf("\n=== second load/forward/unload cycle: the engine's own leak, runtime residue excluded ===\n");
        auto fwd2 = std::make_unique<ie::Ds41Forward>();
        if (auto e = fwd2->init_resident(qs, m, tb, ranking, opt); !e.empty()) { std::fprintf(stderr, "init_resident (cycle 2): %s\n", e.c_str()); return 1; }
        fwd2->set_logits_last_only(true);
        std::vector<float> lg2;
        if (auto e = fwd2->prefill(ids.data(), T, lg2); !e.empty()) { std::fprintf(stderr, "prefill (cycle 2): %s\n", e.c_str()); return 1; }
        for (uint32_t n : {512u, 2048u}) {       // the same shapes as cycle 1, so the residue is already paid
            const std::string p = gd + "/pp_ids_" + std::to_string(n) + ".i32";
            if (!std::ifstream(p).good()) continue;
            const auto in = rd<int32_t>(p, n);
            if (auto e = fwd2->prefill(in.data(), n, lg2); !e.empty()) { std::fprintf(stderr, "pp (cycle 2): %s\n", e.c_str()); return 1; }
        }
        fwd2->free_resident(); fwd2.reset();
        for (auto& qp : queues) qp->wait_and_throw();
        for (uint32_t ci = 0; ci < NCARD; ++ci) {
            const uint64_t f3 = free_vram(ci);
            std::printf("       card %u free VRAM after cycle 1 %.3f GiB, after cycle 2 %.3f GiB\n", ci, gib(f_dtor[ci]), gib(f3));
            check(int64_t(f_dtor[ci]) - int64_t(f3) < int64_t(64ull << 20), "second cycle returns to the first cycle's level, card " + std::to_string(ci) + " (the engine leaks nothing)",
                  std::to_string(double(int64_t(f_dtor[ci]) - int64_t(f3)) / (1 << 20)).substr(0, 7) + " MiB grown");
        }
    }
    // Criterion 7's RAM half is judged ACROSS THE PROCESS BOUNDARY (docs/deepseek41/21, amendment
    // 2): the Level Zero runtime holds freed pinned allocations until exit, so the in-process
    // figure is information for the launcher, which compares MemAvailable before launch and
    // after exit. It was a [FAIL] here on every run for exactly that reason.
    std::printf("       MemAvailable in-process after unload: %.1f GiB (%.1f before load; %.1f GiB not yet returned inside the process — judge at the process boundary)\n",
                double(ma2) / (1u << 20), double(ma0) / (1u << 20), double(int64_t(ma0) - int64_t(ma2)) / (1u << 20));
    std::printf("\n%s\n", g_fail ? "RESIDENT TEST: FAILURE(S)" : "RESIDENT TEST: PASS");
    return g_fail ? 1 : 0;
}
