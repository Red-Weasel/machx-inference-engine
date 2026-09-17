// tools/ds41_multi_test.cpp — DSpark P2 (docs/deepseek41/48 E, P2): a T-row decode step at pos0 equals T
// consecutive one-row steps from the same state, BIT FOR BIT -- every logits row, every layer's state (the
// window ring slot by slot, the latents, the index keys, nc) and the engram hashes -- for T in {2, 3, 6}, at an
// even and an odd pos0 (the ratio-2 parity), on the 12-token golden prompt and past the ring's wrap on the
// 2,048-token pp text. The reference is the one-row path itself (greedy), reached again by a fresh prefill
// (deterministic). Criterion 2: with the per-row ring causality switched off the T-row step must DIFFER.
// usage: ie-ds41-multi-test <model dir> <golden dir> [ranking file]   (IE_DS41_DECODE_MULTI is set here)
#include "ie/deepseek41_forward.hpp"
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
std::vector<int32_t> rd_i32(const std::string& p, size_t n) { std::vector<int32_t> v(n); std::ifstream f(p, std::ios::binary); if (!f) return {}; f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * 4)); return f ? v : std::vector<int32_t>{}; }

struct State { std::vector<std::vector<float>> win, comp, idxk; std::vector<uint32_t> nc; std::vector<std::vector<int64_t>> hashes; std::vector<float> logits; std::vector<int32_t> toks; };

std::string snapshot(const ie::Ds41Forward& fwd, uint32_t NL, State& s) {
    s.win.assign(NL, {}); s.comp.assign(NL, {}); s.idxk.assign(NL, {}); s.nc.assign(NL, 0);
    for (uint32_t L = 0; L < NL; ++L) if (auto e = fwd.read_state(L, s.win[L], s.comp[L], s.idxk[L], s.nc[L]); !e.empty()) return "read_state(" + std::to_string(L) + "): " + e;
    return {};
}
template <class T> bool same(const std::vector<T>& a, const std::vector<T>& b) { return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0); }
}  // namespace

int main(int argc, char** argv) {
    setenv("IE_DS41_DECODE_MULTI", "1", 1);
    const std::string dir = argc > 1 ? argv[1] : std::string(std::getenv("HOME")) + "/models/DeepSeek-V4.1-Flash";
    const std::string gd  = argc > 2 ? argv[2] : dir + "/ie_golden";
    std::string rank_in = argc > 3 ? argv[3] : dir + "/ie_ranking_heldout.txt";
    if (!std::ifstream(rank_in).good()) rank_in.clear();
    nlohmann::json meta; { std::ifstream f(gd + "/decode2/d_meta.json"); if (!f) { std::fprintf(stderr, "no %s/decode2/d_meta.json\n", gd.c_str()); return 1; } f >> meta; }
    const std::vector<int32_t> prompt = meta["prompt_ids"].get<std::vector<int32_t>>();
    const uint32_t NL = meta["n_layers"], V = meta["vocab"];

    ie::DeepSeek41Model m;
    if (auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    ie::Ds41EngramTables tb;
    if (auto e = tb.load(gd); !e.empty()) { std::fprintf(stderr, "tables: %s\n", e.c_str()); return 1; }
    const auto& c = m.config(); const uint32_t E = c.n_routed_experts;
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
    if (!rank_in.empty()) if (auto e = ie::ds4_expert_priority_read_layers(rank_in, E, NL, ranking); !e.empty()) { std::fprintf(stderr, "ranking: %s\n", e.c_str()); return 1; }
    ie::Ds41Forward fwd; ie::Ds41Forward::ResidentOptions opt; opt.max_tokens = 2048 + 64;
    const auto t0 = std::chrono::steady_clock::now();
    if (auto e = fwd.init_resident(qs, m, tb, ranking, opt); !e.empty()) { std::fprintf(stderr, "init_resident: %s\n", e.c_str()); return 1; }
    std::printf("resident in %.0f s on %zu card(s)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), devs.size());

    const auto pp2048 = rd_i32(gd + "/pp_ids_2048.i32", 2048);
    struct Scen { std::string name; const std::vector<int32_t>* ids; };
    std::vector<Scen> scens{{"the 12-token golden prompt", &prompt}};
    if (!pp2048.empty()) scens.push_back({"the 2,048-token pp text (replayed; the ring has wrapped)", &pp2048});

    // run `k0` greedy one-row steps after a fresh prefill of `ids`; returns the logits of the last forward and fills `toks`
    auto prefill_to = [&](const std::vector<int32_t>& ids, uint32_t k0, std::vector<float>& logits, std::vector<int32_t>& toks) -> std::string {
        fwd.reset_state(); fwd.set_logits_last_only(true); toks.clear();
        if (auto e = fwd.forward(ids.data(), uint32_t(ids.size()), 0, logits); !e.empty()) return "prefill: " + e;
        for (uint32_t k = 0; k < k0; ++k) {
            const int32_t nxt = int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
            if (auto e = fwd.forward(&nxt, 1, uint32_t(ids.size()) + k, logits); !e.empty()) return "step " + std::to_string(k) + ": " + e;
            toks.push_back(nxt);
        }
        return {};
    };
    std::vector<float> logits; std::vector<int32_t> pre_toks;
    // diagnostic: row 0 of each layer's output in the one-row step at pos0 vs in the T-row step -- where does the difference start, and how big is it?
    const uint32_t FLAT = meta["hc_mult"].get<uint32_t>() * meta["dim"].get<uint32_t>();
    std::vector<std::vector<float>> lay_ref(NL), lay_got(NL); bool diag_done = false;
    const uint32_t HDIM = meta["dim"].get<uint32_t>();
    std::vector<std::vector<float>> xfn_ref(NL), xfn_got(NL), moe_ref(NL), moe_got(NL);   // the MoE input and output, row 0
    auto mk_probe = [&](std::vector<std::vector<float>>& dst, std::vector<std::vector<float>>& dx, std::vector<std::vector<float>>& dm) { return ie::Ds41Forward::Probe([&dst, &dx, &dm, FLAT, HDIM](const char* name, uint32_t L, const float* d, size_t, sycl::queue& pq) {
        const std::string nm(name);
        if (nm == "layer") { dst[L].resize(FLAT); pq.memcpy(dst[L].data(), d, size_t(FLAT) * 4).wait(); }
        else if (nm == "xfn") { dx[L].resize(HDIM); pq.memcpy(dx[L].data(), d, size_t(HDIM) * 4).wait(); }
        else if (nm == "moe") { dm[L].resize(HDIM); pq.memcpy(dm[L].data(), d, size_t(HDIM) * 4).wait(); } }); };
    for (const auto& sc : scens) for (uint32_t k0 = 0; k0 < 2; ++k0) for (uint32_t T : {2u, 3u, 6u}) {
        const uint32_t pos0 = uint32_t(sc.ids->size()) + k0;
        const std::string tag = sc.name + ", pos0 " + std::to_string(pos0) + " (" + (pos0 % 2 ? "odd" : "even") + "), T = " + std::to_string(T);
        // ---- the reference: T one-row steps
        State ref;
        if (auto e = prefill_to(*sc.ids, k0, logits, pre_toks); !e.empty()) { check(false, tag + ": reference " + e); continue; }
        ref.hashes.assign(tb.layer_ids.size(), {});
        int32_t nxt = int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
        bool bad = false;
        for (uint32_t i = 0; i < T; ++i) {
            ref.toks.push_back(nxt);
            if (auto e = fwd.forward(&nxt, 1, pos0 + i, logits, (!diag_done && i == 0) ? mk_probe(lay_ref, xfn_ref, moe_ref) : ie::Ds41Forward::Probe{}); !e.empty()) { check(false, tag + ": reference step " + std::to_string(i) + ": " + e); bad = true; break; }
            ref.logits.insert(ref.logits.end(), logits.begin(), logits.end());
            const auto& hs = fwd.last_hashes(); for (size_t li = 0; li < hs.size(); ++li) ref.hashes[li].insert(ref.hashes[li].end(), hs[li].begin(), hs[li].end());
            nxt = int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
        }
        if (bad) continue;
        if (auto e = snapshot(fwd, NL, ref); !e.empty()) { check(false, tag + ": " + e); continue; }
        // ---- the T-row step from the same state
        State got;
        if (auto e = prefill_to(*sc.ids, k0, logits, pre_toks); !e.empty()) { check(false, tag + ": multi " + e); continue; }
        fwd.set_logits_last_only(false);
        std::vector<float> lg;
        const auto tm = std::chrono::steady_clock::now();
        if (auto e = fwd.forward(ref.toks.data(), T, pos0, lg, !diag_done ? mk_probe(lay_got, xfn_got, moe_got) : ie::Ds41Forward::Probe{}); !e.empty()) { check(false, tag + ": the T-row step: " + e); continue; }
        if (!diag_done) { diag_done = true; std::printf("       row 0 of each layer's output, the T-row step vs the one-row step (max |diff| / max |ref|):\n");
            auto rel = [](const std::vector<float>& a, const std::vector<float>& b) { double m = 0, s = 0; for (size_t i = 0; i < a.size() && i < b.size(); ++i) { m = std::max(m, std::fabs(double(a[i]) - double(b[i]))); s = std::max(s, std::fabs(double(b[i]))); } return s > 0 ? m / s : 0.0; };
            for (uint32_t L = 0; L < NL; ++L) std::printf("       L%02u layer %.2e xfn %.2e moe %.2e%s", L, rel(lay_got[L], lay_ref[L]), rel(xfn_got[L], xfn_ref[L]), rel(moe_got[L], moe_ref[L]), (L % 3 == 2) ? "\n" : " | ");
            std::printf("\n"); }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tm).count();
        got.logits = lg; got.hashes = fwd.last_hashes();
        if (auto e = snapshot(fwd, NL, got); !e.empty()) { check(false, tag + ": " + e); continue; }
        // ---- compare, bit for bit
        size_t ndiff = 0; double mx = 0; bool sz = got.logits.size() == ref.logits.size() && got.logits.size() == size_t(T) * V;
        if (sz) for (size_t i = 0; i < got.logits.size(); ++i) if (got.logits[i] != ref.logits[i]) { ++ndiff; mx = std::max(mx, std::fabs(double(got.logits[i]) - double(ref.logits[i]))); }
        char b[160]; std::snprintf(b, sizeof b, "%zu of %zu logits differ, max |diff| %.3e; the T-row step %.1f ms", ndiff, got.logits.size(), mx, ms);
        check(sz && ndiff == 0, tag + ": the T logits rows equal the T one-row steps' bit for bit", b);
        uint32_t bad_layers = 0; std::string first;
        for (uint32_t L = 0; L < NL; ++L) {
            const bool ok = same(got.win[L], ref.win[L]) && same(got.comp[L], ref.comp[L]) && same(got.idxk[L], ref.idxk[L]) && got.nc[L] == ref.nc[L];
            if (!ok) { ++bad_layers; if (first.empty()) first = "layer " + std::to_string(L) + (same(got.win[L], ref.win[L]) ? "" : " ring") + (same(got.comp[L], ref.comp[L]) ? "" : " latents") + (same(got.idxk[L], ref.idxk[L]) ? "" : " index-keys") + (got.nc[L] == ref.nc[L] ? "" : " nc"); }
        }
        check(bad_layers == 0, tag + ": every layer's state (ring, latents, index keys, nc) identical", bad_layers ? std::to_string(bad_layers) + " layers differ, first " + first : "40 layers");
        bool hs_ok = got.hashes.size() == ref.hashes.size(); for (size_t li = 0; hs_ok && li < got.hashes.size(); ++li) hs_ok = same(got.hashes[li], ref.hashes[li]);
        check(hs_ok, tag + ": the engram hashes of the T rows identical");
        // ---- criterion 2: the negative control at the largest T on each scenario's even pos0
        if (T == 6 && k0 == 0) {
            if (auto e = prefill_to(*sc.ids, k0, logits, pre_toks); !e.empty()) { check(false, tag + ": control " + e); continue; }
            fwd.set_logits_last_only(false); fwd.set_multi_nocausal_diagnostic(true);
            std::vector<float> lg2; const std::string e = fwd.forward(ref.toks.data(), T, pos0, lg2);
            fwd.set_multi_nocausal_diagnostic(false);
            if (!e.empty()) { check(false, tag + ": the control step: " + e); continue; }
            size_t nd2 = 0; for (size_t i = 0; i < lg2.size() && i < ref.logits.size(); ++i) if (lg2[i] != ref.logits[i]) ++nd2;
            check(nd2 > 0, tag + ": NEGATIVE CONTROL -- the ring causality off makes the T-row step differ", std::to_string(nd2) + " logits differ");
        }
    }
    fwd.free_resident();
    std::printf("\nMULTI TEST: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
