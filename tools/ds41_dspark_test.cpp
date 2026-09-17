// tools/ds41_dspark_test.cpp — DSpark P1 (docs/deepseek41/57, the golden of docs/54): the drafter in the engine against
// the Python golden. The 12-token prompt is prefilled (the backbone), the drafter's rings are seeded from the golden's
// forced `sp_prefill_main_hidden`, then positions 12..15 in order with the golden's forced `main_hidden` and input id:
// main_x, the rings, the opened columns, the embedded rows, the stage boundaries with FORCED routing, the head, the
// Markov chain, the confidence -- each against its file and bar; then the engine's own router's flips; then the two
// negative controls. usage: ie-ds41-dspark-test <model dir> <golden dir> [ranking file]
#include "ie/deepseek41_dspark.hpp"
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
template <class T> std::vector<T> rd(const std::string& p, size_t n) { std::vector<T> v(n); std::ifstream f(p, std::ios::binary); if (!f) return {}; f.read(reinterpret_cast<char*>(v.data()), std::streamsize(n * sizeof(T))); return f ? v : std::vector<T>{}; }
double rel(const float* a, const float* b, size_t n) { double m = 0, s = 0; for (size_t i = 0; i < n; ++i) { m = std::max(m, std::fabs(double(a[i]) - double(b[i]))); s = std::max(s, std::fabs(double(b[i]))); } return s > 0 ? m / s : 0; }
double rel(const std::vector<float>& a, const std::vector<float>& b) { return a.size() == b.size() && !a.empty() ? rel(a.data(), b.data(), a.size()) : 1e9; }
std::string f3(double v) { char b[32]; std::snprintf(b, sizeof b, "%.3e", v); return b; }
}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : std::string(std::getenv("HOME")) + "/models/DeepSeek-V4.1-Flash";
    const std::string gd  = argc > 2 ? argv[2] : dir + "/ie_golden";
    const std::string sd  = gd + "/dspark";
    std::string rank_in = argc > 3 ? argv[3] : dir + "/ie_ranking_heldout.txt";
    if (!std::ifstream(rank_in).good()) rank_in.clear();
    nlohmann::json meta; { std::ifstream f(sd + "/sp_meta.json"); if (!f) { std::fprintf(stderr, "no %s/sp_meta.json\n", sd.c_str()); return 1; } f >> meta; }
    nlohmann::json dmeta; { std::ifstream f(gd + "/decode2/d_meta.json"); if (!f) { std::fprintf(stderr, "no decode2/d_meta.json\n"); return 1; } f >> dmeta; }
    const std::vector<int32_t> prompt = dmeta["prompt_ids"].get<std::vector<int32_t>>();
    const uint32_t H = meta["dim"], HC = meta["hc_mult"], WIN = meta["window"], B = meta["block_size"], NS = meta["n_stages"], V = meta["vocab"], TK = meta["topk"], R = meta["markov_rank"], HD = meta["head_dim"];
    const uint32_t NT = uint32_t(meta["target_layer_ids"].size()), T0 = uint32_t(prompt.size());
    const auto& positions = meta["positions"];
    std::printf("dspark golden: %zu positions, block %u, %u stages, %u/%u experts\n", positions.size(), B, NS, uint32_t(meta["n_routed"]), TK);

    ie::DeepSeek41Model m;
    if (auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    check(bool(m.dspark), "the drafter's six extras are bound (main_proj, main_norm, norm, markov embed / head, confidence proj)");
    check(m.unclaimed().empty(), "no unclaimed tensors after the binds", m.unclaimed().empty() ? "" : "first: " + m.unclaimed()[0]);
    ie::Ds41EngramTables tb;
    if (auto e = tb.load(gd); !e.empty()) { std::fprintf(stderr, "tables: %s\n", e.c_str()); return 1; }
    const auto& c = m.config(); const uint32_t E = c.n_routed_experts, NL = c.n_layers;
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
    // the drafter on the head card (the last queue)
    sycl::queue& dq = *qs.back();
    ie::Ds41Drafter drafter; ie::Ds41Drafter::Options dopt;
    const auto t1 = std::chrono::steady_clock::now();
    if (auto e = drafter.init(dq, m, fwd.dense_cache(uint32_t(qs.size() - 1)), {}, dopt); !e.empty()) { std::fprintf(stderr, "drafter init: %s\n", e.c_str()); drafter.free(dq); fwd.free_resident(); return 1; }
    const double init_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    std::printf("drafter: %.2f GiB dense on card %zu, tier static %u / pinned %u per stage, init %.1f s\n", double(drafter.dense_bytes()) / 1073741824.0, qs.size() - 1, drafter.tier().n_static(), drafter.tier().n_pinned(), init_s);

    // the backbone prefill (the drafter's inputs come from the golden, forced)
    std::vector<float> logits; fwd.set_logits_last_only(true); fwd.set_capture_main_hidden(true);
    if (auto e = fwd.forward(prompt.data(), T0, 0, logits); !e.empty()) { std::fprintf(stderr, "prefill: %s\n", e.c_str()); return 1; }
    { const auto& mh = fwd.main_hidden(); const auto g = rd<float>(sd + "/sp_prefill_main_hidden.f32", size_t(T0) * NT * H);
      check(mh.size() == g.size(), "the backbone captured main_hidden for the prompt rows [T, n_targets * dim]", std::to_string(mh.size()) + " vs " + std::to_string(g.size()));
      if (const char* dd = std::getenv("IE_DS41_DSPARK_DUMP")) { std::ofstream f(std::string(dd) + "/mh_engine.f32", std::ios::binary); f.write(reinterpret_cast<const char*>(mh.data()), std::streamsize(mh.size() * 4)); }
      // information only: with the engine's OWN router the prefill's stream flips from layer 7 on (docs/18 amended: the
      // per-layer comparison holds up to the first flip; the forward test reads 1.7e-1 .. 2.3e-1 at layers 29-34), so
      // the golden's forced main_hidden drives the bars below; the own-input preview at the end reports what it drafts
      if (mh.size() == g.size()) std::printf("       the engine's own prefill main_hidden vs the golden's: %s (information; own-router flips from layer 7)\n", f3(rel(mh, g)).c_str()); }
    // seed the rings from the golden's prefill main_hidden; compare the rings' filled slots
    { const auto g = rd<float>(sd + "/sp_prefill_main_hidden.f32", size_t(T0) * NT * H);
      if (auto e = drafter.seed(dq, g.data(), T0, 0); !e.empty()) { check(false, "seed: " + e); }
      { const auto gx = rd<float>(sd + "/sp_prefill_main_x.f32", size_t(T0) * H); const double r = rel(drafter.last_main_x(), gx);
        check(r < 3e-3, "main_x of the prompt rows (main_norm(main_proj(main_hidden))) within 3e-3", f3(r)); }
      for (uint32_t s = 0; s < NS; ++s) { const auto gw = rd<float>(sd + "/sp_prefill_window_" + std::to_string(s) + ".f32", size_t(WIN) * HD);
          std::vector<float> ring(size_t(WIN) * HD); dq.memcpy(ring.data(), drafter.ring(s), ring.size() * 4).wait();
          const double r = rel(ring.data(), gw.data(), size_t(T0) * HD);                       // slots 0..T0-1
          check(r < 3e-3, "stage " + std::to_string(s) + ": the ring after the prefill seeding (slots 0.." + std::to_string(T0 - 1) + ") within 3e-3", f3(r)); } }
    // the positions, in order
    std::vector<std::vector<int32_t>> own_flips(NS);
    for (const auto& P : positions) {
        const uint32_t pos = P["pos"]; const int32_t in_id = P["input_id"]; const std::string sp = sd + "/sp" + std::to_string(pos) + "_";
        const auto mh = rd<float>(sp + "main_hidden.f32", size_t(NT) * H);
        // the probes: every stage boundary into host copies
        std::vector<std::vector<float>> got_attn_in(NS), got_attn_out(NS), got_ffn_in(NS), got_moe_out(NS), got_layer_out(NS), got_ffn_pre(NS); std::vector<float> got_embed, got_head_in, got_logits_base, got_mask;
        ie::Ds41Drafter::Probe probe = [&](const char* name, uint32_t s, const float* d, size_t n, sycl::queue& pq) {
            std::vector<float>* dst = nullptr; const std::string nm(name);
            if (nm == "attn_in") dst = &got_attn_in[s]; else if (nm == "attn_out") dst = &got_attn_out[s]; else if (nm == "ffn_in") dst = &got_ffn_in[s]; else if (nm == "moe_out") dst = &got_moe_out[s];
            else if (nm == "layer_out") dst = &got_layer_out[s]; else if (nm == "ffn_pre") dst = &got_ffn_pre[s]; else if (nm == "embed") dst = &got_embed; else if (nm == "head_in") dst = &got_head_in; else if (nm == "logits_base") dst = &got_logits_base; else if (nm == "mask") dst = &got_mask;
            if (!dst) return; dst->resize(n); pq.memcpy(dst->data(), d, n * 4).wait(); };
        // forced routing per stage
        std::vector<std::vector<int32_t>> fidx(NS); std::vector<std::vector<float>> fw(NS); std::vector<const int32_t*> fip(NS); std::vector<const float*> fwp(NS);
        for (uint32_t s = 0; s < NS; ++s) { fidx[s] = rd<int32_t>(sp + "route_idx_" + std::to_string(s) + ".i32", size_t(B) * TK); fw[s] = rd<float>(sp + "route_w_" + std::to_string(s) + ".f32", size_t(B) * TK); fip[s] = fidx[s].data(); fwp[s] = fw[s].data(); }
        std::vector<int32_t> ids; std::vector<float> lg, conf;
        if (auto e = drafter.draft(dq, mh.data(), in_id, pos, ids, lg, conf, probe, &fip, &fwp); !e.empty()) { check(false, "pos " + std::to_string(pos) + ": draft: " + e); continue; }
        const std::string tag = "pos " + std::to_string(pos);
        std::printf("       %s: %.1f ms per pass (forced routing)\n", tag.c_str(), drafter.last_pass_ms());
        // main_x is checked through the ring (its only consumer); the rings after the pass
        for (uint32_t s = 0; s < NS; ++s) { const auto gw = rd<float>(sp + "window_" + std::to_string(s) + ".f32", size_t(WIN) * HD);
            std::vector<float> ring(size_t(WIN) * HD); dq.memcpy(ring.data(), drafter.ring(s), ring.size() * 4).wait();
            const double r = rel(ring.data(), gw.data(), size_t(std::min(WIN, pos + 1)) * HD);
            check(r < 3e-3, tag + ", stage " + std::to_string(s) + ": the ring after the pass (slots 0.." + std::to_string(pos) + ") within 3e-3", f3(r)); }
        { const auto g = rd<float>(sp + "embed.f32", size_t(B) * HC * H); check(g.size() == got_embed.size() && std::memcmp(g.data(), got_embed.data(), g.size() * 4) == 0, tag + ": the five embedded rows bit-identical"); }
        for (uint32_t s = 0; s < NS; ++s) {
            const std::string st = tag + ", stage " + std::to_string(s);
            const double r1 = rel(got_attn_in[s], rd<float>(sp + "attn_in_" + std::to_string(s) + ".f32", size_t(B) * H)); check(r1 < 1.2e-2, st + ": attn_in within 1.2e-2", f3(r1));
            const double r2 = rel(got_attn_out[s], rd<float>(sp + "attn_out_" + std::to_string(s) + ".f32", size_t(B) * H)); check(r2 < 1.2e-2, st + ": attn_out within 1.2e-2", f3(r2));
            const double r3 = rel(got_ffn_in[s], rd<float>(sp + "ffn_in_" + std::to_string(s) + ".f32", size_t(B) * H)); check(r3 < 1.2e-2, st + ": ffn_in within 1.2e-2", f3(r3));
            const double r4 = rel(got_moe_out[s], rd<float>(sp + "moe_out_" + std::to_string(s) + ".f32", size_t(B) * H)); check(r4 < 1.2e-2, st + ": moe_out (forced routing) within 1.2e-2", f3(r4));
            const double r5 = rel(got_layer_out[s], rd<float>(sp + "layer_out_" + std::to_string(s) + ".f32", size_t(B) * HC * H)); check(r5 < 1.2e-2, st + ": layer_out within 1.2e-2", f3(r5));
            const double r6 = rel(got_ffn_pre[s], rd<float>(sp + "ffn_pre_" + std::to_string(s) + ".f32", size_t(B) * HC)); check(r6 < 1.2e-2, st + ": ffn_pre within 1.2e-2", f3(r6));
        }
        { const double r = rel(got_head_in, rd<float>(sp + "head_in.f32", size_t(B) * H)); check(r < 1.2e-2, tag + ": head_in within 1.2e-2", f3(r)); }
        { const double r = rel(got_logits_base, rd<float>(sp + "logits_base.f32", size_t(B) * V)); check(r < 5e-2, tag + ": logits_base within 5e-2", f3(r)); }
        { const auto g = rd<float>(sp + "logits.f32", size_t(B) * V); const double r = rel(lg, g); check(r < 5e-2, tag + ": logits after the Markov bias within 5e-2", f3(r)); }
        { const auto g = rd<int32_t>(sp + "output_ids.i32", B + 1); bool same = g.size() == ids.size() && std::memcmp(g.data(), ids.data(), g.size() * 4) == 0;
          std::string d; for (uint32_t i = 0; i < ids.size(); ++i) d += std::to_string(ids[i]) + (i + 1 < ids.size() ? " " : ""); d += " | golden"; for (int32_t v : g) d += " " + std::to_string(v);
          if (!same) { // a near-tie: the row's draft margin under the logits bar x max|logits|
              const auto& dm = P["draft_margin"]; double mxl = 0; for (float v : lg) mxl = std::max(mxl, std::fabs(double(v)));
              bool near = true; for (uint32_t i = 1; i <= B && i < g.size(); ++i) if (ids[i] != g[i] && double(dm[i - 1]) >= 5e-2 * mxl) near = false;
              check(near, tag + ": output_ids equal the golden's (a mismatch only at a near-tie under the logits bar)", d); if (near) std::printf("       (near-tie mismatch tolerated)\n"); }
          else check(true, tag + ": output_ids equal the golden's", d); }
        { const auto g = rd<float>(sp + "confidence.f32", B); double m = 0, s = 0; for (uint32_t i = 0; i < B; ++i) { m = std::max(m, std::fabs(double(conf[i]) - g[i])); s = std::max(s, std::fabs(double(g[i]))); }
          check(m / s < 1e-2, tag + ": confidence within 1e-2 relative (1e-3 reported)", f3(m / s)); }
        { const auto g = rd<float>(sp + "main_x.f32", H); const double r = rel(drafter.last_main_x(), g); check(r < 3e-3, tag + ": main_x within 3e-3", f3(r)); }
        // criterion 2: the opened columns per draft row (the mask's zero entries: ring slots <= pos, the five draft columns) equal the golden's
        for (uint32_t s = 0; s < NS; ++s) {
            std::vector<int32_t> gt; { std::ifstream f(sp + "topk_" + std::to_string(s) + ".i32", std::ios::binary | std::ios::ate); const auto n = size_t(f.tellg()) / 4; gt.resize(n); f.seekg(0); f.read(reinterpret_cast<char*>(gt.data()), std::streamsize(n * 4)); }
            const uint32_t NKV = WIN + B; std::vector<int32_t> open; bool rows_equal = got_mask.size() == size_t(B) * NKV;
            for (uint32_t t = 0; t < B && rows_equal; ++t) { std::vector<int32_t> row; for (uint32_t kk = 0; kk < NKV; ++kk) if (got_mask[size_t(t) * NKV + kk] == 0.f) row.push_back(int32_t(kk)); open.insert(open.end(), row.begin(), row.end()); }
            check(rows_equal && open == gt, tag + ", stage " + std::to_string(s) + ": the opened columns of the five draft rows equal the golden's topk exactly", std::to_string(open.size()) + " vs " + std::to_string(gt.size()) + " entries");
        }
        { const auto g = rd<float>(sp + "markov_embed.f32", size_t(B) * R); const auto& e = drafter.last_markov_embed();
          check(e.size() == g.size() && std::memcmp(e.data(), g.data(), g.size() * 4) == 0, tag + ": the Markov embedding rows bit-identical", f3(rel(e, g))); }
        { const auto g = rd<float>(sp + "markov_bias.f32", size_t(B) * V); const double r = rel(drafter.last_markov_bias(), g); check(r < 1e-3, tag + ": the Markov bias rows within 1e-3", f3(r)); }
        // the engine's own router: a second pass, flips against the golden per stage
        { std::vector<int32_t> ids2; std::vector<float> lg2, conf2;
          if (auto e = drafter.draft(dq, nullptr, in_id, pos, ids2, lg2, conf2); e.empty()) {
              const auto& own = drafter.last_routing();
              for (uint32_t s = 0; s < NS; ++s) { const auto gap = rd<float>(sp + "route_gap_" + std::to_string(s) + ".f32", B); uint32_t flips = 0, bad = 0;
                  for (uint32_t t = 0; t < B; ++t) { std::vector<int32_t> a(own[s].begin() + t * TK, own[s].begin() + (t + 1) * TK), b(fidx[s].begin() + t * TK, fidx[s].begin() + (t + 1) * TK); std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end()); if (a != b) { ++flips; if (gap[t] >= 3e-3f) ++bad; } }
                  check(bad == 0, tag + ", stage " + std::to_string(s) + ": own-router flips only at near-ties (gap < 3e-3)", std::to_string(flips) + " flips, " + std::to_string(bad) + " at a large gap"); }
              std::printf("       %s: own router drafts", tag.c_str()); for (int32_t v : ids2) std::printf(" %d", v); std::printf(" (%.1f ms per pass)\n", drafter.last_pass_ms()); }
          else check(false, tag + ": the own-router pass: " + e); }
    }
    // the negative controls at the first position
    if (!positions.empty()) {
        const auto& P = positions[0]; const uint32_t pos = P["pos"]; const int32_t in_id = P["input_id"]; const std::string sp = sd + "/sp" + std::to_string(pos) + "_";
        const auto mh = rd<float>(sp + "main_hidden.f32", size_t(NT) * H);
        std::vector<std::vector<int32_t>> fidx(NS); std::vector<std::vector<float>> fw(NS); std::vector<const int32_t*> fip(NS); std::vector<const float*> fwp(NS);
        for (uint32_t s = 0; s < NS; ++s) { fidx[s] = rd<int32_t>(sp + "route_idx_" + std::to_string(s) + ".i32", size_t(B) * TK); fw[s] = rd<float>(sp + "route_w_" + std::to_string(s) + ".f32", size_t(B) * TK); fip[s] = fidx[s].data(); fwp[s] = fw[s].data(); }
        std::vector<float> ao0; ie::Ds41Drafter::Probe probe = [&](const char* name, uint32_t s, const float* d, size_t n, sycl::queue& pq) { if (std::string(name) == "attn_out" && s == 0) { ao0.resize(n); pq.memcpy(ao0.data(), d, n * 4).wait(); } };
        const auto g = rd<float>(sp + "attn_out_0.f32", size_t(B) * H);
        std::vector<int32_t> ids; std::vector<float> lg, conf;
        drafter.set_rope_offset_diagnostic(1); const std::string e1 = drafter.draft(dq, nullptr, in_id, pos, ids, lg, conf, probe, &fip, &fwp); drafter.set_rope_offset_diagnostic(0);
        check(e1.empty() && rel(ao0, g) >= 1.2e-2, "NEGATIVE CONTROL: the draft rows' RoPE position off by one breaks attn_out_0", e1.empty() ? f3(rel(ao0, g)) : e1);
        drafter.set_causal_drafts_diagnostic(true); const std::string e2 = drafter.draft(dq, nullptr, in_id, pos, ids, lg, conf, probe, &fip, &fwp); drafter.set_causal_drafts_diagnostic(false);
        check(e2.empty() && rel(ao0, g) >= 1.2e-2, "NEGATIVE CONTROL: a causal mask among the five draft columns breaks attn_out_0 (the golden's control_rel: " + f3(double(P["control_rel"])) + ")", e2.empty() ? f3(rel(ao0, g)) : e2);
    }
    // information for P4: the drafter on the ENGINE's own inputs at pos T0 -- its prefill capture seeds the rings, its
    // decode step at T0 gives main_hidden(T0) and the token at T0 + 1 -- against the golden's drafts at that position
    {
        const std::vector<float> mh_p = fwd.main_hidden();                                                      // the prompt rows, still held
        const int32_t t0 = int32_t(std::max_element(logits.begin(), logits.end()) - logits.begin());            // the backbone's sample at T0
        std::vector<float> lg1; std::string e = drafter.seed(dq, mh_p.data(), T0, 0);
        if (e.empty()) e = fwd.forward(&t0, 1, T0, lg1);
        std::vector<int32_t> ids; std::vector<float> lg, conf;
        if (e.empty()) { const std::vector<float> mh1 = fwd.main_hidden(); const int32_t t1 = int32_t(std::max_element(lg1.begin(), lg1.end()) - lg1.begin());
            e = drafter.draft(dq, mh1.data(), t1, T0, ids, lg, conf); }
        if (!e.empty()) std::printf("       own-input preview at pos %u: %s\n", T0, e.c_str());
        else { std::string got, want; for (int32_t v : ids) got += std::to_string(v) + " ";
            for (const auto& P : positions) if (uint32_t(P["pos"]) == T0) for (int32_t v : P["output_ids"].get<std::vector<int32_t>>()) want += std::to_string(v) + " ";
            std::printf("       own-input preview at pos %u (the engine's own prefill capture, decode step and router): drafts %s| golden %s(information; %.1f ms per pass)\n", T0, got.c_str(), want.c_str(), drafter.last_pass_ms()); }
    }
    drafter.free(dq); fwd.free_resident();
    std::printf("\nDSPARK TEST: %s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
