// tools/ds41_rollback_test.cpp — DSpark P3 (docs/deepseek41/56): after a T = 6 step at pos0, rollback_to(pos0 + L) for
// L in 1..6 leaves EXACTLY the state L one-row steps leave (every layer's ring / latents / index keys / nc, n_pos, the
// id sequence), and one further one-row step from it gives the never-speculated path's logits bit for bit. Criterion 3:
// with the ring restore skipped the state must differ for some L < 6 past the wrap.
// usage: ie-ds41-rollback-test <model dir> <golden dir> [short|long|both] [ranking file]
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
struct State { std::vector<std::vector<float>> win, comp, idxk; std::vector<uint32_t> nc; uint32_t n_pos = 0; std::vector<int32_t> ids; };
std::string snapshot(const ie::Ds41Forward& fwd, uint32_t NL, State& s) {
    s.win.assign(NL, {}); s.comp.assign(NL, {}); s.idxk.assign(NL, {}); s.nc.assign(NL, 0); s.n_pos = fwd.n_pos(); s.ids = fwd.all_ids();
    for (uint32_t L = 0; L < NL; ++L) if (auto e = fwd.read_state(L, s.win[L], s.comp[L], s.idxk[L], s.nc[L]); !e.empty()) return "read_state(" + std::to_string(L) + "): " + e;
    return {};
}
template <class T> bool same(const std::vector<T>& a, const std::vector<T>& b) { return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0); }
// the ring is compared over its LIVE slots only -- those holding positions <= n_pos - 1 (all of them once wrapped): the
// prefill rewrites the last min(T, window) slots and reset_state keeps the rest, so unwritten slots carry the previous
// run's history in either path and no step ever reads them (the mask closes positions < 0)
std::string diff_state(const State& a, const State& b, uint32_t NL, uint32_t WIN, uint32_t HD) {
    if (a.n_pos != b.n_pos) return "n_pos " + std::to_string(a.n_pos) + " vs " + std::to_string(b.n_pos);
    if (a.ids != b.ids) return "the id sequence differs (" + std::to_string(a.ids.size()) + " vs " + std::to_string(b.ids.size()) + " ids)";
    uint32_t bad = 0; std::string first; const size_t live = size_t(std::min(WIN, a.n_pos)) * HD;
    for (uint32_t L = 0; L < NL; ++L) {
        const bool ring_ok = a.win[L].size() == b.win[L].size() && a.win[L].size() >= live && (live == 0 || std::memcmp(a.win[L].data(), b.win[L].data(), live * sizeof(float)) == 0);
        const bool ok = ring_ok && same(a.comp[L], b.comp[L]) && same(a.idxk[L], b.idxk[L]) && a.nc[L] == b.nc[L];
        if (!ok) { ++bad; if (first.empty()) first = "layer " + std::to_string(L) + (ring_ok ? "" : " ring") + (same(a.comp[L], b.comp[L]) ? "" : " latents") + (same(a.idxk[L], b.idxk[L]) ? "" : " index-keys") + (a.nc[L] == b.nc[L] ? "" : " nc"); }
    }
    return bad ? std::to_string(bad) + " layers differ, first " + first : "";
}
}  // namespace

int main(int argc, char** argv) {
    setenv("IE_DS41_DECODE_MULTI", "1", 1);
    const std::string dir = argc > 1 ? argv[1] : std::string(std::getenv("HOME")) + "/models/DeepSeek-V4.1-Flash";
    const std::string gd  = argc > 2 ? argv[2] : dir + "/ie_golden";
    const std::string which = argc > 3 ? argv[3] : "both";
    std::string rank_in = argc > 4 ? argv[4] : dir + "/ie_ranking_heldout.txt";
    if (!std::ifstream(rank_in).good()) rank_in.clear();
    nlohmann::json meta; { std::ifstream f(gd + "/decode2/d_meta.json"); if (!f) { std::fprintf(stderr, "no %s/decode2/d_meta.json\n", gd.c_str()); return 1; } f >> meta; }
    const std::vector<int32_t> prompt = meta["prompt_ids"].get<std::vector<int32_t>>();
    const uint32_t NL = meta["n_layers"]; const uint32_t WIN = 128, HD = 512;   // the ring's shape (window_size, head_dim), from the config below
    ie::DeepSeek41Model m;
    if (auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    ie::Ds41EngramTables tb;
    if (auto e = tb.load(gd); !e.empty()) { std::fprintf(stderr, "tables: %s\n", e.c_str()); return 1; }
    const auto& c = m.config(); const uint32_t E = c.n_routed_experts;
    if (c.window_size != WIN || c.head_dim != HD) { std::fprintf(stderr, "unexpected window / head_dim\n"); return 1; }
    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) { if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue; for (const auto& d : p.get_devices(sycl::info::device_type::gpu)) if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d); }
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
    std::vector<Scen> scens;
    if (which != "long") scens.push_back({"the 12-token golden prompt", &prompt});
    if (which != "short" && !pp2048.empty()) scens.push_back({"the 2,048-token pp text (the ring has wrapped)", &pp2048});
    const uint32_t T = 6;
    std::vector<float> logits; std::vector<int32_t> toks;
    auto prefill_to = [&](const std::vector<int32_t>& ids, uint32_t k0, std::vector<float>& lg) -> std::string {
        fwd.reset_state(); fwd.set_logits_last_only(true);
        if (auto e = fwd.forward(ids.data(), uint32_t(ids.size()), 0, lg); !e.empty()) return "prefill: " + e;
        for (uint32_t k = 0; k < k0; ++k) { const int32_t nxt = int32_t(std::max_element(lg.begin(), lg.end()) - lg.begin()); if (auto e = fwd.forward(&nxt, 1, uint32_t(ids.size()) + k, lg); !e.empty()) return "step " + std::to_string(k) + ": " + e; }
        return {};
    };
    double snap_ms_sum = 0; uint32_t snap_n = 0;
    for (const auto& sc : scens) for (uint32_t k0 = 0; k0 < 2; ++k0) {
        const uint32_t pos0 = uint32_t(sc.ids->size()) + k0;
        // the reference tokens of the T greedy one-row steps from S (and their logits after step L, for the continuation)
        if (auto e = prefill_to(*sc.ids, k0, logits); !e.empty()) { check(false, sc.name + ": " + e); continue; }
        toks.clear(); std::vector<std::vector<float>> ref_logits_after(T + 1); std::vector<State> ref_state(T + 1);
        int32_t nxt = int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
        for (uint32_t i = 0; i < T; ++i) { toks.push_back(nxt); if (auto e = fwd.forward(&nxt, 1, pos0 + i, logits); !e.empty()) { check(false, sc.name + ": reference step " + std::to_string(i) + ": " + e); break; } nxt = int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin()); }
        // per L: the reference state after L steps + one continuation step's logits and state
        for (uint32_t L = 1; L <= T; ++L) {
            const std::string tag = sc.name + ", pos0 " + std::to_string(pos0) + " (" + (pos0 % 2 ? "odd" : "even") + "), rollback to L = " + std::to_string(L);
            if (auto e = prefill_to(*sc.ids, k0, logits); !e.empty()) { check(false, tag + ": " + e); continue; }
            for (uint32_t i = 0; i < L; ++i) if (auto e = fwd.forward(&toks[i], 1, pos0 + i, logits); !e.empty()) { check(false, tag + ": ref " + e); break; }
            State ref; if (auto e = snapshot(fwd, NL, ref); !e.empty()) { check(false, tag + ": " + e); continue; }
            const int32_t cont = int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());
            std::vector<float> ref_cont; if (auto e = fwd.forward(&cont, 1, pos0 + L, ref_cont); !e.empty()) { check(false, tag + ": ref continuation " + e); continue; }
            State ref2; if (auto e = snapshot(fwd, NL, ref2); !e.empty()) { check(false, tag + ": " + e); continue; }
            // the T-row step, the rollback, the same continuation
            if (auto e = prefill_to(*sc.ids, k0, logits); !e.empty()) { check(false, tag + ": " + e); continue; }
            fwd.set_logits_last_only(false); std::vector<float> lg6;
            if (auto e = fwd.forward(toks.data(), T, pos0, lg6); !e.empty()) { check(false, tag + ": the T-row step: " + e); continue; }
            const auto tr = std::chrono::steady_clock::now();
            if (auto e = fwd.rollback_to(pos0 + L); !e.empty()) { check(false, tag + ": rollback_to: " + e); continue; }
            snap_ms_sum += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr).count(); ++snap_n;
            State got; if (auto e = snapshot(fwd, NL, got); !e.empty()) { check(false, tag + ": " + e); continue; }
            const std::string d1 = diff_state(got, ref, NL, WIN, HD);
            check(d1.empty(), tag + ": the state after the rollback equals L one-row steps' (ring, latents, index keys, nc, n_pos)", d1.empty() ? "40 layers" : d1);
            fwd.set_logits_last_only(true); std::vector<float> got_cont;
            if (auto e = fwd.forward(&cont, 1, pos0 + L, got_cont); !e.empty()) { check(false, tag + ": continuation " + e); continue; }
            size_t nd = 0; for (size_t i = 0; i < got_cont.size() && i < ref_cont.size(); ++i) if (got_cont[i] != ref_cont[i]) ++nd;
            check(got_cont.size() == ref_cont.size() && nd == 0, tag + ": one further one-row step's logits bit-identical to the never-speculated path", std::to_string(nd) + " of " + std::to_string(got_cont.size()) + " differ");
            State got2; if (auto e = snapshot(fwd, NL, got2); !e.empty()) { check(false, tag + ": " + e); continue; }
            const std::string d2 = diff_state(got2, ref2, NL, WIN, HD);
            check(d2.empty(), tag + ": ... and its state", d2.empty() ? "40 layers" : d2);
            // criterion 3: the negative control past the wrap, at one L < T
            if (pos0 >= 128 && L == 2) {
                if (auto e = prefill_to(*sc.ids, k0, logits); !e.empty()) { check(false, tag + ": control " + e); continue; }
                fwd.set_logits_last_only(false); std::vector<float> lgc;
                if (auto e = fwd.forward(toks.data(), T, pos0, lgc); !e.empty()) { check(false, tag + ": control step: " + e); continue; }
                fwd.set_rollback_noring_diagnostic(true); const std::string er = fwd.rollback_to(pos0 + L); fwd.set_rollback_noring_diagnostic(false);
                if (!er.empty()) { check(false, tag + ": control rollback: " + er); continue; }
                State gc; if (auto e = snapshot(fwd, NL, gc); !e.empty()) { check(false, tag + ": " + e); continue; }
                const std::string dc = diff_state(gc, ref, NL, WIN, HD);
                check(!dc.empty(), tag + ": NEGATIVE CONTROL -- skipping the ring restore leaves a different state", dc.empty() ? "identical (the control has no teeth)" : dc);
            }
        }
    }
    if (snap_n) std::printf("       rollback_to: %.3f ms mean over %u calls\n", snap_ms_sum / snap_n, snap_n);
    fwd.free_resident();
    std::printf("\nROLLBACK TEST: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
