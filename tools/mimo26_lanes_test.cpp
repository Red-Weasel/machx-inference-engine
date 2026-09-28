// tools/mimo26_lanes_test.cpp -- ie-mimo26-lanes-test: the gate of MiMo-V2.6 lanes with serial interleaving (P4 B1,
// ~/ds41_work/p60/p4_mimo_batching_design.md). One model loaded with L lanes (Mimo26Options::lanes, the drafter's rings via
// Mimo26DFlash::add_lanes). Every prompt runs SOLO on lane 0 first -- the pre-lane path -- then all of them INTERLEAVED over
// the L lanes: one forward of one lane at a time (a prefill chunk or a decode row), round robin (Mimo26LaneSet::next), a lane
// that finishes taking the next queued prompt mid-run (on a dirty lane: reset, a fresh sequence). Each interleaved sequence
// must be BYTE-IDENTICAL to its solo run: the logits after the prompt and after every decode step (fp32 bytes, FNV), the
// greedy tokens, and with the drafter its drafts + probabilities before every step and its context bookkeeping at the end.
// The CPU expert leg is off (IE_DS41_CPU_MISS=0: its split follows the stream-slot history, so its rows are not
// bit-reproducible run to run; --cpu-miss keeps it on, and then identity is not expected).
// --slot-arm adds the host-slot interplay: a lane's conversation is kept in a host slot (Mimo26PrefixCache, the per-lane
// bookkeeping) when an unrelated prompt arrives on that lane, restored into ANOTHER lane for its continuation, and the
// continuation must be identical to a control that never left lane 0.
// --pipeline (P4 B2, docs/mimo26/P4_B2_PIPELINE.md) replaces the serial interleave by CARD-PIPELINED lanes
// (Mimo26Forward::pipe_*): one stage thread per card, lane B's step on card 0 while lane A's runs on card 1, every lane driving
// itself from the last stage's callback (the drafter's work there too). Same per-sequence identity checks against solo. It also
// times the steady window where every lane decodes (aggregate tokens/s vs solo), prints each lane's stream-slot hits and each
// card's per-engine-class busy from /proc/self/fdinfo over the solo and the pipelined decode windows. --aba runs the solo arm
// again after the pipelined one (A-B-A; a short warm-up precedes the first arm). With --cpu-miss the identity lines are
// informational (the CPU leg's split follows the stream-slot history); both cards' legs stay on 8-19 as served, --split gives
// each card's leg its own E-cores (Mimo26Options::split_cpu_cores, for the whole process -- the solo arms too; measured slower).
// --group-lanes G (P4 B3, docs/mimo26/P4_B3_ROWS.md): the pipe's stage 0 groups up to G waiting lanes' rows into ONE forward (0 =
// auto, ceil(lanes in flight / cards): 4 lanes on 2 cards run as two 2-row groups; 1 = B2's one lane per group); the tier print
// then shows each card's group steps (hits per group). --stream-slots S sets the tier's stream slots per layer (default 8).
// --invariant sets IE_MIMO26_ROWS_INVARIANT=1 (the o-proj and the head per lane inside a group: the serial launches), and
// first runs the SPLIT-ROWS check (P4 B16, ~/ds41_work/p60/mimo_ident/ANALYSIS.md section 2): on lane 0 with the first
// prompt, for every M in --invariant-m (default 9,16,38,46,64,239) and a start P = --invariant-start (default 8, != 0):
//   A  one forward of the P + M tokens at position 0 (all rows' logits),
//   B  P tokens at 0, then the M tokens at P (a lane prefix of P reused, the remainder prefilled),
//   C  P + 16 tokens at 0, rewound to P, then the M tokens at P (the reused rows computed by a different chunk).
// Rows [P, P + M) of B and C against A, fp32 bytes: identical rows, the first differing row and the largest |diff|. The result is
// REPORTED ([info] and a SPLIT-ROWS line), not gated: whether MiMo's T > 8 prefill kernels are split-invariant is what it
// measures. --invariant-only stops after it.
//   usage: ie-mimo26-lanes-test <model_dir> --prompt-file F [--prompt-file F ...] [--lanes L] [--n N] [--ctx C]
//          [--lane-ctx C2] [--chunk C] [--no-dflash] [--slot-arm] [--cpu-miss] [--pipeline] [--aba] [--split]
//          [--group-lanes G] [--stream-slots S] [--invariant] [--invariant-m M1,M2,...] [--invariant-start P] [--invariant-only]
#include "ie/expert_stream.hpp"      // ds4_expert_priority_read_layers
#include "ie/mimo26.hpp"
#include "ie/mimo26_dflash.hpp"
#include "ie/mimo26_forward.hpp"
#include "ie/mimo26_lanes.hpp"
#include "ie/mimo26_prefix_cache.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <dirent.h>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what, const std::string& detail = "") {
    std::printf("[%s] %s%s%s\n", ok ? " ok " : "FAIL", what.c_str(), detail.empty() ? "" : "  ", detail.c_str());
    std::fflush(stdout);
    if (!ok) ++g_fail;
}
uint64_t fnv(const std::vector<float>& v) {
    uint64_t h = 1469598103934665603ull;
    for (float x : v) { uint32_t u; std::memcpy(&u, &x, 4); h = (h ^ u) * 1099511628211ull; }
    return h;
}
double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
int32_t argmax(const std::vector<float>& lg) { return int32_t(std::max_element(lg.begin(), lg.end()) - lg.begin()); }

// B0 M1 (the premise): this process's DRM clients' engine cycles from /proc/self/fdinfo (xe: drm-cycles-<class> and
// drm-total-cycles-<class> per client; a client's busy share of a class = its cycle delta / the total-cycle delta, divided by
// drm-engine-capacity-<class>). Deltas are ACCUMULATED over windows (begin/end pairs), so several solo decodes add up.
struct FdBusy {
    struct Cnt { uint64_t cyc = 0, tot = 0; uint32_t cap = 1; };
    using Snap = std::map<std::string, Cnt>;                    // "pdev|client|class"
    std::map<std::string, std::pair<uint64_t, uint64_t>> acc;   // key -> (sum of cycle deltas, sum of total-cycle deltas)
    std::map<std::string, uint32_t> caps;
    Snap at;
    static Snap snap() {
        Snap s;
        DIR* d = opendir("/proc/self/fdinfo");
        if (!d) return s;
        std::map<std::string, bool> seen;                       // one client may sit behind several fds
        while (dirent* e = readdir(d)) {
            if (e->d_name[0] == '.') continue;
            std::ifstream in(std::string("/proc/self/fdinfo/") + e->d_name);
            std::string line, pdev, client; std::map<std::string, Cnt> m;
            while (std::getline(in, line)) {
                const size_t c = line.find(':'); if (c == std::string::npos) continue;
                const std::string k = line.substr(0, c), v = line.substr(c + 1);
                const uint64_t n = std::strtoull(v.c_str(), nullptr, 10);
                if (k == "drm-pdev") { const size_t b = v.find_first_not_of(" \t"); pdev = b == std::string::npos ? "" : v.substr(b); }
                else if (k == "drm-client-id") client = std::to_string(n);
                else if (k.rfind("drm-total-cycles-", 0) == 0) m[k.substr(17)].tot = n;
                else if (k.rfind("drm-cycles-", 0) == 0) m[k.substr(11)].cyc = n;
                else if (k.rfind("drm-engine-capacity-", 0) == 0) m[k.substr(20)].cap = uint32_t(std::max<uint64_t>(1, n));
            }
            if (client.empty() || pdev.empty() || seen[pdev + "|" + client]) continue;
            seen[pdev + "|" + client] = true;
            for (const auto& [cls, cn] : m) s[pdev + "|" + client + "|" + cls] = cn;
        }
        closedir(d);
        return s;
    }
    void begin() { at = snap(); }
    void end() {
        const Snap now = snap();
        for (const auto& [k, c] : now) {
            const auto it = at.find(k);
            if (it == at.end() || c.tot <= it->second.tot || c.cyc < it->second.cyc) continue;
            acc[k].first += c.cyc - it->second.cyc; acc[k].second += c.tot - it->second.tot; caps[k] = c.cap;
        }
    }
    // per card (pdev) and class: the clients' busy % summed; and Dream's figure (per client the max over classes, summed)
    void print(const char* label) const {
        std::map<std::string, std::map<std::string, double>> by, cl;   // pdev -> class -> %; pdev|client -> class -> %
        for (const auto& [k, v] : acc) {
            if (!v.second) continue;
            const size_t a = k.find('|'), b = k.rfind('|');
            const double pct = 100.0 * double(v.first) / double(v.second) / caps.at(k);
            by[k.substr(0, a)][k.substr(b + 1)] += pct;
            cl[k.substr(0, b)][k.substr(b + 1)] = pct;
        }
        for (const auto& [pdev, m] : by) {
            double dream = 0;
            for (const auto& [ck, cm] : cl)
                if (ck.rfind(pdev + "|", 0) == 0) { double mx = 0; for (const auto& [c, p] : cm) mx = std::max(mx, p); dream += mx; }
            std::printf("fdinfo busy [%s] card %s:", label, pdev.c_str());
            for (const auto& [c, p] : m) std::printf(" %s %.1f%%", c.c_str(), p);
            std::printf("  (Dream's figure, per client the max over classes, summed: %.1f%%)\n", dream);
            for (const auto& [ck, cm] : cl)
                if (ck.rfind(pdev + "|", 0) == 0) {
                    std::printf("    client %s:", ck.substr(pdev.size() + 1).c_str());
                    for (const auto& [c, p] : cm) if (p > 0) std::printf(" %s %.1f%%", c.c_str(), p);
                    std::printf("\n");
                }
        }
        if (by.empty()) std::printf("fdinfo busy [%s]: no DRM client cycles in /proc/self/fdinfo\n", label);
    }
};

// One sequence's record: the logits after its prompt and after every decode step, its tokens, the drafter's drafts and
// probabilities before every step and its context bookkeeping at the end.
struct Trace {
    std::vector<uint64_t> logits;
    std::vector<int32_t> tokens, drafts;
    std::vector<float> probs;
    uint32_t n_pos = 0, ctx_lo = 0, ctx_end = 0, ctx_hi = 0;
    double prefill_ms = 0, decode_ms = 0;
    std::vector<double> step_ms;   // each decode step (timing only; not compared)
};
std::string first_diff(const Trace& a, const Trace& b) {
    for (size_t i = 0; i < std::min(a.logits.size(), b.logits.size()); ++i)
        if (a.logits[i] != b.logits[i]) return "logits differ first at step " + std::to_string(i);
    if (a.logits.size() != b.logits.size()) return "step counts differ";
    if (a.tokens != b.tokens) return "tokens differ";
    if (a.drafts != b.drafts) return "drafts differ";
    if (a.probs.size() != b.probs.size() || (!a.probs.empty() && std::memcmp(a.probs.data(), b.probs.data(), a.probs.size() * 4)))
        return "draft probabilities differ";
    if (a.n_pos != b.n_pos || a.ctx_lo != b.ctx_lo || a.ctx_end != b.ctx_end || a.ctx_hi != b.ctx_hi) return "bookkeeping differs";
    return {};
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: ie-mimo26-lanes-test <model_dir> --prompt-file F [--prompt-file F ...] [--lanes L] [--n N] [--ctx C] "
                             "[--lane-ctx C2] [--chunk C] [--no-dflash] [--slot-arm] [--cpu-miss] [--pipeline] [--aba] [--split] "
                             "[--group-lanes G] [--stream-slots S] [--invariant] [--invariant-m M1,M2,...] [--invariant-start P] [--invariant-only]\n");
        return 2;
    }
    const std::string dir = argv[1];
    std::vector<std::string> files;
    uint32_t lanes = 2, N = 64, ctx = 16384, lane_ctx = 0, C = 2048, group_lanes = 0, stream_slots = 8;
    bool use_df = true, slot_arm = false, cpu_miss = false, pipeline = false, aba = false, split = false, invariant = false;
    std::vector<uint32_t> inv_m{9, 16, 38, 46, 64, 239};   // (B16) the split-rows check's remainder sizes
    uint32_t inv_start = 8; bool inv_only = false;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--prompt-file") files.push_back(val());
        else if (a == "--lanes") lanes = uint32_t(std::atol(val().c_str()));
        else if (a == "--n") N = uint32_t(std::atol(val().c_str()));
        else if (a == "--ctx") ctx = uint32_t(std::atol(val().c_str()));
        else if (a == "--lane-ctx") lane_ctx = uint32_t(std::atol(val().c_str()));
        else if (a == "--chunk") C = uint32_t(std::atol(val().c_str()));
        else if (a == "--no-dflash") use_df = false;
        else if (a == "--slot-arm") slot_arm = true;
        else if (a == "--cpu-miss") cpu_miss = true;
        else if (a == "--pipeline") pipeline = true;
        else if (a == "--aba") aba = true;
        else if (a == "--split") split = true;
        else if (a == "--group-lanes") group_lanes = uint32_t(std::atol(val().c_str()));
        else if (a == "--stream-slots") stream_slots = uint32_t(std::atol(val().c_str()));
        else if (a == "--invariant") invariant = true;
        else if (a == "--invariant-m") {
            inv_m.clear();
            std::stringstream ss(val()); std::string t;
            while (std::getline(ss, t, ',')) if (!t.empty()) inv_m.push_back(uint32_t(std::atol(t.c_str())));
        }
        else if (a == "--invariant-start") inv_start = uint32_t(std::atol(val().c_str()));
        else if (a == "--invariant-only") { invariant = true; inv_only = true; }
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (files.empty() || lanes == 0 || N == 0) { std::fprintf(stderr, "need --prompt-file, --lanes >= 1, --n >= 1\n"); return 2; }
    if (!cpu_miss) setenv("IE_DS41_CPU_MISS", "0", 1);
    if (invariant) setenv("IE_MIMO26_ROWS_INVARIANT", "1", 1);
    ie::Mimo26Model m;
    if (auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_hf_json(dir + "/tokenizer.json", dir + "/tokenizer_config.json"); !e.empty()) { std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1; }
    std::vector<std::vector<int32_t>> prompts;
    const uint32_t cap_small = lanes > 1 && lane_ctx ? std::min(ctx, lane_ctx) : ctx;
    for (const auto& f : files) {
        std::ifstream in(f, std::ios::binary);
        if (!in) { std::fprintf(stderr, "cannot read %s\n", f.c_str()); return 2; }
        prompts.push_back(tok.encode(std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()), true));
        if (prompts.back().empty() || prompts.back().size() + N > cap_small) {
            std::fprintf(stderr, "%s: %zu tokens + %u do not fit ctx %u\n", f.c_str(), prompts.back().size(), N, cap_small); return 2; }
    }

    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d);
    }
    if (devs.empty()) { std::fprintf(stderr, "no Arc GPU\n"); return 1; }
    std::vector<std::unique_ptr<sycl::queue>> queues; std::vector<sycl::queue*> qs;
    for (const auto& d : devs) { queues.push_back(std::make_unique<sycl::queue>(sycl::context(d), d, sycl::property_list{sycl::property::queue::in_order{}})); qs.push_back(queues.back().get()); }
    // the engine's configuration (mimo26_load): 2,048-row chunks, the auto static tier, the chat ranking when installed
    ie::Mimo26Options opt; opt.max_ctx = ctx; opt.max_tokens = std::min<uint32_t>(C, ctx); opt.n_static = 0;
    opt.lanes = lanes; opt.lane_ctx = lane_ctx; opt.stream_slots = stream_slots;
    opt.split_cpu_cores = pipeline && cpu_miss && split;   // B2 --split: each card's CPU leg on its own E-cores (measured slower)
    if (const std::string rp = dir + "/ie_ranking_mimo26_chat.txt"; std::ifstream(rp).good())
        if (auto e = ie::ds4_expert_priority_read_layers(rp, m.config().n_routed_experts, m.config().n_layers, opt.ranking); !e.empty()) { std::fprintf(stderr, "ranking: %s\n", e.c_str()); return 1; }
    const auto t_init = std::chrono::steady_clock::now();
    ie::Mimo26DFlash df_obj;                     // on the last card, before the forward (as mimo26_load)
    ie::Mimo26DFlash* df = nullptr;
    if (use_df && std::ifstream(dir + "/dflash/config.json").good()) {
        if (auto e = df_obj.load(dir); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
        if (auto e = df_obj.init(*qs.back(), m.embed.w->data, m.config().vocab_size, df_obj.config().window); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
        if (auto e = df_obj.add_lanes(lanes); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
        df = &df_obj;
    }
    ie::Mimo26Forward fwd;
    if (auto e = fwd.init(qs, m, opt); !e.empty()) { std::printf("init: %s\n", e.c_str()); return 1; }
    if (df) {
        df->set_head(fwd.head_weights());
        if (auto e = fwd.set_feature_layers(df->config().target_layers, df->config().window); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
    }
    std::printf("mimo26: %zu card(s), init %.1f s, %u lane(s): lane 0 ctx %u, the others ctx %u, SWA ring %u, drafter %s, CPU miss leg %s, "
                "stream slots %u%s%s\n",
                qs.size(), ms_since(t_init) / 1000, fwd.n_lanes(), ctx, lanes > 1 && lane_ctx ? lane_ctx : ctx, fwd.ring(), df ? "ON" : "off", cpu_miss ? "ON" : "off",
                stream_slots, pipeline ? (group_lanes ? (", groups of up to " + std::to_string(group_lanes) + " lane(s)").c_str() : ", groups auto (lanes in flight / cards)") : "",
                invariant ? ", rows invariant (o-proj and head per lane)" : "");
    std::printf("VRAM per card:");
    for (size_t c = 0; c < qs.size(); ++c) std::printf(" %.2f GB (one extra lane %.3f GB, static/pinned %u/%u)", fwd.vram_bytes(c) / 1e9, fwd.lane_bytes(c) / 1e9, fwd.card_static(c), fwd.card_pinned(c));
    if (df) std::printf("; drafter ring per lane %.3f GB", df->lane_bytes() / 1e9);
    std::printf("\n");
    if (lanes > 1) {   // (d) the logged lane VRAM against the lane arithmetic (the design's ~0.68 GB per card at 32K)
        uint64_t sum = 0;
        for (size_t c = 0; c < qs.size(); ++c) sum += fwd.lane_bytes(c);
        std::vector<ie::Mimo26LaneLayer> all;
        for (uint32_t L = 0; L < m.config().n_layers; ++L) all.push_back({m.config().is_swa(L), m.config().n_kv(L)});
        const uint64_t want = ie::mimo26_lane_kv_bytes(all, m.config().head_dim, m.config().v_head_dim, fwd.ring(), lane_ctx ? lane_ctx : ctx);
        check(sum == want, "one extra lane's caches over the cards equal the lane arithmetic",
              std::to_string(sum) + " B = " + std::to_string(double(sum) / 1e9) + " GB; at ctx 32,768 the arithmetic gives " +
              std::to_string(double(ie::mimo26_lane_kv_bytes(all, m.config().head_dim, m.config().v_head_dim, fwd.ring(), 32768)) / 1e9) + " GB");
    }

    auto select = [&](uint32_t l) -> bool {
        std::string e = fwd.select_lane(l);
        if (e.empty() && df) e = df->select_lane(l);
        if (!e.empty()) { std::printf("select lane %u: %s\n", l, e.c_str()); return false; }
        return true;
    };
    auto ctx_add = [&](uint32_t n, uint32_t p0, uint32_t stride) -> bool {
        if (!df) return true;
        if (auto e = df->add_context(fwd.features(), n, p0, stride); !e.empty()) { std::printf("dflash context at %u: %s\n", p0, e.c_str()); return false; }
        return true;
    };
    // one step of a sequence on the selected lane: the prefill chunk at `at` or (at == prompt end) a decode row of `next`
    std::vector<float> lg;
    auto prefill_chunk = [&](const std::vector<int32_t>& ids, uint32_t& at, Trace& t) -> bool {
        const auto t0 = std::chrono::steady_clock::now();
        const uint32_t n = std::min<uint32_t>(opt.max_tokens, uint32_t(ids.size()) - at);
        if (auto e = fwd.forward(ids.data() + at, n, at, lg, false); !e.empty()) { std::printf("prefill at %u: %s\n", at, e.c_str()); return false; }
        if (!ctx_add(fwd.feat_rows(), at + n - fwd.feat_rows(), fwd.feat_rows())) return false;
        at += n;
        t.prefill_ms += ms_since(t0);
        if (at == ids.size()) t.logits.push_back(fnv(lg));
        return true;
    };
    auto decode_step = [&](int32_t id, Trace& t) -> bool {
        const auto t0 = std::chrono::steady_clock::now();
        if (df) {
            std::vector<int32_t> d;
            if (auto e = df->draft(id, df->config().block - 1, d, 0.f); !e.empty()) { std::printf("draft at %u: %s\n", fwd.n_pos(), e.c_str()); return false; }
            t.drafts.insert(t.drafts.end(), d.begin(), d.end());
            t.probs.insert(t.probs.end(), df->last_probs().begin(), df->last_probs().end());
        }
        const uint32_t pos = fwd.n_pos();
        if (auto e = fwd.forward(&id, 1, pos, lg, false); !e.empty()) { std::printf("decode at %u: %s\n", pos, e.c_str()); return false; }
        if (!ctx_add(1, pos, 1)) return false;
        t.logits.push_back(fnv(lg));
        t.decode_ms += ms_since(t0);
        return true;
    };
    auto finish_trace = [&](Trace& t) {
        t.n_pos = fwd.n_pos();
        if (df) { t.ctx_lo = df->ctx_lo(); t.ctx_end = df->ctx_end(); t.ctx_hi = df->ctx_hi(); }
    };

    // ---- (B16) --invariant: the split-rows check -- rows [P, P + M) prefilled after a reused prefix vs in one call --------
    if (invariant) {
        const uint32_t V = m.config().vocab_size, P = inv_start;
        const std::vector<int32_t>& ids = prompts[0];
        if (P == 0) { std::fprintf(stderr, "--invariant-start must be > 0 (the split's start position)\n"); return 2; }
        if (!select(0)) return 1;
        std::vector<float> la, lb, lc, tmp;
        for (const uint32_t M : inv_m) {
            if (M == 0 || P + 16 + M > ids.size() || P + 16 + M > opt.max_tokens) {
                std::printf("[info] split rows M=%u P=%u: skipped (the first prompt has %zu tokens, a forward at most %u rows)\n", M, P, ids.size(), opt.max_tokens);
                continue;
            }
            std::string e;
            fwd.reset(); e = fwd.forward(ids.data(), P + M, 0, la, true);                                           // A
            if (e.empty()) { fwd.reset(); e = fwd.forward(ids.data(), P, 0, tmp, false); }                         // B
            if (e.empty()) e = fwd.forward(ids.data() + P, M, P, lb, true);
            if (e.empty()) { fwd.reset(); e = fwd.forward(ids.data(), P + 16, 0, tmp, false); }                    // C
            if (e.empty()) { fwd.rewind(P); e = fwd.forward(ids.data() + P, M, P, lc, true); }
            if (!e.empty()) { std::printf("split rows M=%u: %s\n", M, e.c_str()); return 1; }
            auto cmp = [&](const std::vector<float>& x, uint32_t& same, int& first, double& maxd) {
                same = 0; first = -1; maxd = 0;
                for (uint32_t r = 0; r < M; ++r) {
                    const float* a = la.data() + size_t(P + r) * V; const float* b = x.data() + size_t(r) * V;
                    if (!std::memcmp(a, b, size_t(V) * 4)) { ++same; continue; }
                    if (first < 0) first = int(r);
                    for (uint32_t i = 0; i < V; ++i) maxd = std::max(maxd, double(std::fabs(a[i] - b[i])));
                }
            };
            uint32_t sb, sc; int fb, fc; double db, dc;
            cmp(lb, sb, fb, db); cmp(lc, sc, fc, dc);
            std::printf("SPLIT-ROWS M=%u P=%u: B (prefix %u, remainder at %u) == A %u/%u rows (first diff row %d, max |d| %.3g); "
                        "C (prefix from a %u-row call) == A %u/%u rows (first diff row %d, max |d| %.3g)\n",
                        M, P, P, P, sb, M, fb, db, P + 16, sc, M, fc, dc);
            std::fflush(stdout);
        }
        fwd.reset();
        if (df) df->reset();
        if (inv_only) { std::printf("\n--invariant-only: stopped after the split-rows check\n"); return 0; }
    }

    // with the CPU leg on, a lane's rows are not bit-reproducible run to run: identity is reported, not gated
    auto check_id = [&](bool ok, const std::string& what, const std::string& detail) {
        if (!cpu_miss) { check(ok, what, detail); return; }
        std::printf("[info] %s: %s%s%s\n", what.c_str(), ok ? "identical" : "DIFFERS", detail.empty() ? "" : "  ", detail.c_str());
        std::fflush(stdout);
    };
    // per lane and card: the decode steps' experts and how many pinned ones a stream slot already held (the B1 hypothesis)
    auto print_tier = [&](const char* label, uint32_t n_l) {
        for (uint32_t l = 0; l < n_l; ++l)
            for (size_t c = 0; c < fwd.n_cards(); ++c) {
                const auto& t = fwd.lane_tier_stats(l, c);
                if (!t.steps) continue;
                std::printf("tier [%s] lane %u card %zu: %llu decode steps; per step %.1f static, %.1f pinned, of them %.1f stream-slot hits (%.1f%%) and "
                            "%.1f on the CPU; moe %.1f ms/step\n", label, l, c, (unsigned long long)t.steps, double(t.experts_static) / t.steps,
                            double(t.experts_pinned) / t.steps, double(t.stream_hits) / t.steps,
                            t.experts_pinned ? 100.0 * double(t.stream_hits) / double(t.experts_pinned) : 0.0, double(t.experts_cpu) / t.steps, t.moe_ms / t.steps);
            }
        for (size_t c = 0; c < fwd.n_cards(); ++c) {   // B3: the steps that carried several lanes' rows, per card
            const auto& t = fwd.group_tier_stats(c);
            if (!t.steps) continue;
            std::printf("tier [%s] GROUPS card %zu: %llu group steps of %.2f rows; per step %.1f static, %.1f pinned, of them %.1f stream-slot hits (%.1f%%) and "
                        "%.1f on the CPU; moe %.1f ms/step\n", label, c, (unsigned long long)t.steps, double(t.rows) / t.steps, double(t.experts_static) / t.steps,
                        double(t.experts_pinned) / t.steps, double(t.stream_hits) / t.steps,
                        t.experts_pinned ? 100.0 * double(t.stream_hits) / double(t.experts_pinned) : 0.0, double(t.experts_cpu) / t.steps, t.moe_ms / t.steps);
        }
        fwd.reset_lane_tier_stats();
    };
    constexpr uint32_t kWarmSteps = 4;   // decode steps of each sequence left out of the timing (oneDNN primitives, first slots)
    auto steady_rate = [&](const std::vector<Trace>& v) {   // solo: tokens/s over the decode steps after the first kWarmSteps
        double ms = 0; uint32_t n = 0;
        for (const auto& t : v) for (size_t k = kWarmSteps; k < t.step_ms.size(); ++k) { ms += t.step_ms[k]; ++n; }
        return ms > 0 ? n / (ms / 1000) : 0.0;
    };

    // ---- solo: every prompt alone on lane 0 (the pre-lane path) --------------------------------------------------------
    FdBusy fd_solo, fd_pipe;
    auto run_solo = [&](std::vector<Trace>& out, const char* label) -> double {   // wall ms, < 0 on a failure
        out.assign(prompts.size(), Trace{});
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < prompts.size(); ++i) {
            if (!select(0)) return -1;
            fwd.reset(); if (df) df->reset();
            Trace& t = out[i];
            for (uint32_t at = 0; at < prompts[i].size(); ) if (!prefill_chunk(prompts[i], at, t)) return -1;
            int32_t id = argmax(lg);
            fd_solo.begin();
            for (uint32_t k = 0; k < N; ++k) {
                t.tokens.push_back(id);
                if (k + 1 == N) break;
                const double d0 = t.decode_ms;
                if (!decode_step(id, t)) return -1;
                t.step_ms.push_back(t.decode_ms - d0);
                id = argmax(lg);
            }
            fd_solo.end();
            finish_trace(t);
            std::printf("solo%s %zu (%s): %zu prompt tokens, prefill %.0f ms, decode %.1f ms/token, tokens %d %d %d %d ...\n", label, i, files[i].c_str(),
                        prompts[i].size(), t.prefill_ms, t.decode_ms / std::max<size_t>(1, N - 1), t.tokens[0], t.tokens[1 % N], t.tokens[2 % N], t.tokens[3 % N]);
        }
        return ms_since(t0);
    };
    if (aba) {   // warm-up before the first arm: a short prefill and a few decode steps, discarded
        if (!select(0)) return 1;
        fwd.reset(); if (df) df->reset();
        const std::vector<int32_t> w(prompts[0].begin(), prompts[0].begin() + std::ptrdiff_t(std::min<size_t>(512, prompts[0].size())));
        Trace t;
        for (uint32_t at = 0; at < w.size(); ) if (!prefill_chunk(w, at, t)) return 1;
        int32_t id = argmax(lg);
        for (uint32_t k = 0; k < 8; ++k) { if (!decode_step(id, t)) return 1; id = argmax(lg); }
        fwd.reset_lane_tier_stats();
        std::printf("warm-up: 512-token prefill + 8 decode steps, discarded\n");
    }
    std::vector<Trace> solo;
    const double solo_ms = run_solo(solo, aba ? " A1" : "");
    if (solo_ms < 0) return 1;
    print_tier("solo", 1);
    const double solo_rate = steady_rate(solo);

    ie::Mimo26LaneSet set(lanes);
    std::vector<Trace> inter(prompts.size());
    std::vector<int> seq_of(lanes, -1);           // the prompt each lane runs
    std::deque<size_t> queue; for (size_t i = 0; i < prompts.size(); ++i) queue.push_back(i);
    uint32_t steps = 0, switches = 0, readmits = 0; int last = -1;
    std::string order;
    auto admit = [&]() {
        std::vector<uint32_t> got;
        while (!queue.empty()) {
            const int l = set.admit(prompts[queue.front()], N, 0);
            if (l < 0) break;
            if (seq_of[size_t(l)] >= 0) ++readmits;
            seq_of[size_t(l)] = int(queue.front());
            queue.pop_front();
            got.push_back(uint32_t(l));
        }
        return got;
    };
    if (!pipeline) {
    // ---- interleaved: the prompts over the lanes, one lane's forward at a time; a finished lane takes the next prompt -------
    admit();
    const auto t_inter = std::chrono::steady_clock::now();
    double dec_wall = 0; uint32_t dec_steps = 0;
    for (int l; (l = set.next()) >= 0; ) {
        if (!select(uint32_t(l))) return 1;
        if (l != last) ++switches;
        last = l;
        if (order.size() < 160) order += std::to_string(l);
        ie::Mimo26Lane& ln = set.lane(uint32_t(l));
        Trace& t = inter[size_t(seq_of[size_t(l)])];
        ++steps;
        if (ln.phase == ie::Mimo26Lane::Phase::kPrefill) {
            if (ln.prefill_at == 0) { fwd.reset(); if (df) df->reset(); ln.live.clear(); }   // a fresh sequence on a possibly dirty lane
            const uint32_t at0 = ln.prefill_at;
            if (!prefill_chunk(ln.prompt, ln.prefill_at, t)) return 1;
            ln.live.insert(ln.live.end(), ln.prompt.begin() + at0, ln.prompt.begin() + ln.prefill_at);
            if (ln.prefill_at < ln.prompt.size()) continue;
            ln.phase = ie::Mimo26Lane::Phase::kDecode;
        } else {
            const auto t0 = std::chrono::steady_clock::now();
            if (!decode_step(ln.next, t)) return 1;
            ln.live.push_back(ln.next);
            dec_wall += ms_since(t0); ++dec_steps;
        }
        const int32_t id = argmax(lg);
        t.tokens.push_back(id);
        if (!ie::mimo26_lane_commit(ln, id, {}, true)) {   // (ignore_eos: every sequence runs its N tokens, as solo)
            finish_trace(t);
            set.release(uint32_t(l));
            admit();                                        // mid-run: the finished lane takes the next queued prompt
        }
    }
    const double inter_ms = ms_since(t_inter);
    std::printf("interleaved: %u forwards over %u lane(s), %u lane switches, %u mid-run re-admissions; lane order %s%s\n",
                steps, lanes, switches, readmits, order.c_str(), order.size() >= 160 ? " ..." : "");
    print_tier("interleaved", lanes);
    for (size_t i = 0; i < prompts.size(); ++i) {
        const std::string d = first_diff(inter[i], solo[i]);
        check_id(d.empty(), "sequence " + std::to_string(i) + " (" + files[i] + ") interleaved == solo: " + std::to_string(inter[i].logits.size()) +
                            " logits rows, " + std::to_string(inter[i].tokens.size()) + " tokens" + (df ? ", drafts and drafter bookkeeping" : ""), d);
    }
    if (lanes > 1) check(readmits > 0 || prompts.size() <= lanes, "a lane finished mid-run and took the next queued prompt", std::to_string(readmits) + " re-admission(s)");
    uint32_t n_tok = 0; double solo_dec = 0; for (const auto& t : solo) { n_tok += uint32_t(t.tokens.size()); solo_dec += t.decode_ms; }
    std::printf("timing: solo %.1f s for all prompts (decode %.1f ms/step), interleaved %.1f s (decode %.1f ms/step): interleaved/solo wall %.3f; "
                "aggregate %.2f vs %.2f tokens/s over the whole run (B1 serial interleave: no gain expected)\n",
                solo_ms / 1000, solo_dec / std::max<uint32_t>(1, n_tok - uint32_t(prompts.size())), inter_ms / 1000, dec_wall / std::max(1u, dec_steps),
                inter_ms / solo_ms, n_tok / (inter_ms / 1000), n_tok / (solo_ms / 1000));
    } else {
    // ---- pipelined (B2): every lane drives itself from the last stage's callback; card 0 runs one lane while card 1 runs another --
    std::mutex cb_mu;                                  // the callback's state (one stage thread calls it; main holds it while seeding)
    std::string cb_err;
    std::vector<uint32_t> pend(lanes, 0);             // the rows of each lane's step in flight
    std::vector<std::chrono::steady_clock::time_point> t_sub(lanes);
    const auto t_pipe = std::chrono::steady_clock::now();
    std::vector<uint32_t> wave;                        // the first wave's lanes: the steady window is while all of them decode
    std::vector<uint32_t> dec_n(lanes, 0);
    std::vector<double> done_ms;                       // every decode completion (ms since t_pipe) while the window may be open
    double win0 = -1, win1 = -1;
    auto fail = [&](const std::string& e) { if (cb_err.empty()) cb_err = e; };
    auto submit = [&](uint32_t l) -> bool {           // the lane's next step: its next prefill chunk, or a decode row (the draft first)
        ie::Mimo26Lane& ln = set.lane(l);
        std::string e;
        t_sub[l] = std::chrono::steady_clock::now();
        if (ln.phase == ie::Mimo26Lane::Phase::kPrefill) {
            pend[l] = std::min<uint32_t>(opt.max_tokens, uint32_t(ln.prompt.size()) - ln.prefill_at);
            e = fwd.pipe_submit(l, ln.prompt.data() + ln.prefill_at, pend[l], ln.prefill_at, false);
        } else {
            if (df) {
                Trace& t = inter[size_t(seq_of[l])];
                std::vector<int32_t> d;
                if (auto de = df->select_lane(l); !de.empty()) { fail(de); return false; }
                if (auto de = df->draft(ln.next, df->config().block - 1, d, 0.f); !de.empty()) { fail("draft: " + de); return false; }
                t.drafts.insert(t.drafts.end(), d.begin(), d.end());
                t.probs.insert(t.probs.end(), df->last_probs().begin(), df->last_probs().end());
            }
            pend[l] = 1;
            e = fwd.pipe_submit(l, &ln.next, 1, fwd.lane_pos(l), false);
        }
        if (!e.empty()) { fail(e); return false; }
        return true;
    };
    auto start = [&](uint32_t l) -> bool {            // a fresh sequence on a possibly dirty lane (in the callback)
        if (auto e = fwd.pipe_reset_lane(l); !e.empty()) { fail(e); return false; }
        if (df) { if (auto e = df->select_lane(l); !e.empty()) { fail(e); return false; } df->reset(); }
        set.lane(l).live.clear();
        return submit(l);
    };
    auto on_done = [&](uint32_t l, const std::vector<float>& logits, const float* feats, uint32_t rows) {
        std::lock_guard<std::mutex> lk(cb_mu);
        if (!cb_err.empty()) return;
        ie::Mimo26Lane& ln = set.lane(l);
        Trace& t = inter[size_t(seq_of[l])];
        const double step = ms_since(t_sub[l]);
        ++steps;
        if (df) if (auto e = df->select_lane(l); !e.empty()) { fail(e); return; }
        if (ln.phase == ie::Mimo26Lane::Phase::kPrefill) {
            const uint32_t n = pend[l], at0 = ln.prefill_at;
            if (df) if (auto e = df->add_context(feats, rows, at0 + n - rows, rows); !e.empty()) { fail("dflash context: " + e); return; }
            ln.live.insert(ln.live.end(), ln.prompt.begin() + at0, ln.prompt.begin() + at0 + n);
            ln.prefill_at += n; t.prefill_ms += step;
            if (ln.prefill_at < ln.prompt.size()) { submit(l); return; }
            t.logits.push_back(fnv(logits));
            ln.phase = ie::Mimo26Lane::Phase::kDecode;
        } else {
            const uint32_t pos = fwd.lane_pos(l) - 1;
            if (df) if (auto e = df->add_context(feats, 1, pos, 1); !e.empty()) { fail("dflash context: " + e); return; }
            t.logits.push_back(fnv(logits)); t.decode_ms += step; t.step_ms.push_back(step);
            ln.live.push_back(ln.next);
            const double at = ms_since(t_pipe);
            ++dec_n[l];
            if (win1 < 0) {
                done_ms.push_back(at);
                if (win0 < 0 && std::all_of(wave.begin(), wave.end(), [&](uint32_t w) { return dec_n[w] >= kWarmSteps; })) { win0 = at; fd_pipe.begin(); }
            }
        }
        const int32_t id = argmax(logits);
        t.tokens.push_back(id);
        if (ie::mimo26_lane_commit(ln, id, {}, true)) { submit(l); return; }
        t.n_pos = fwd.lane_pos(l);
        if (df) { t.ctx_lo = df->ctx_lo(); t.ctx_end = df->ctx_end(); t.ctx_hi = df->ctx_hi(); }
        if (win1 < 0 && win0 >= 0 && std::find(wave.begin(), wave.end(), l) != wave.end()) { win1 = ms_since(t_pipe); fd_pipe.end(); }
        set.release(l);
        for (uint32_t nl : admit()) start(nl);        // mid-run: the finished lane takes the next queued prompt
    };
    // every lane starts empty (before the stages run: the drafter is the callback's once they do)
    for (uint32_t l = 0; l < lanes; ++l) { if (!select(l)) return 1; fwd.reset(); if (df) df->reset(); }
    if (!select(0)) return 1;
    if (auto e = fwd.pipe_start(on_done, group_lanes); !e.empty()) { std::printf("pipe_start: %s\n", e.c_str()); return 1; }
    {
        std::lock_guard<std::mutex> lk(cb_mu);
        wave = admit();
        for (uint32_t l : wave) { set.lane(l).live.clear(); if (!submit(l)) break; }
    }
    const std::string pe = fwd.pipe_stop();
    const double pipe_ms = ms_since(t_pipe);
    if (!pe.empty() || !cb_err.empty()) { std::printf("pipeline: %s%s\n", pe.c_str(), cb_err.c_str()); return 1; }
    std::printf("pipelined: %u forwards over %u lane(s), %u mid-run re-admissions, %.1f s\n", steps, lanes, readmits, pipe_ms / 1000);
    print_tier("pipelined", lanes);
    for (size_t i = 0; i < prompts.size(); ++i) {
        const std::string d = first_diff(inter[i], solo[i]);
        check_id(d.empty(), "sequence " + std::to_string(i) + " (" + files[i] + ") pipelined == solo: " + std::to_string(inter[i].logits.size()) +
                            " logits rows, " + std::to_string(inter[i].tokens.size()) + " tokens" + (df ? ", drafts and drafter bookkeeping" : ""), d);
    }
    if (lanes > 1) check(readmits > 0 || prompts.size() <= lanes, "a lane finished mid-run and took the next queued prompt", std::to_string(readmits) + " re-admission(s)");
    // the steady window: from every first-wave lane's kWarmSteps-th decode step to the first of them finishing
    uint32_t in_win = 0;
    for (double t : done_ms) if (t > win0 && t <= win1) ++in_win;
    const double agg = win1 > win0 && win0 >= 0 ? in_win / ((win1 - win0) / 1000) : 0.0;
    double lat = 0; uint32_t nlat = 0;
    for (const auto& t : inter) for (size_t k = kWarmSteps; k < t.step_ms.size(); ++k) { lat += t.step_ms[k]; ++nlat; }
    std::printf("timing: steady window %.1f s with all %zu first-wave lanes decoding: %u decode steps = %.2f tokens/s aggregate, %.1f ms per lane step "
                "(submit to logits); solo%s %.2f tokens/s (%.1f ms/step) -> x%.3f\n",
                (win1 - win0) / 1000, wave.size(), in_win, agg, nlat ? lat / nlat : 0.0, aba ? " A1" : "", solo_rate, solo_rate > 0 ? 1000 / solo_rate : 0.0,
                solo_rate > 0 ? agg / solo_rate : 0.0);
    if (aba) {
        std::vector<Trace> solo2;
        if (run_solo(solo2, " A2") < 0) return 1;
        print_tier("solo A2", 1);
        const double r2 = steady_rate(solo2);
        for (size_t i = 0; i < prompts.size(); ++i) check_id(first_diff(solo2[i], solo[i]).empty(), "solo A2 == solo A1, sequence " + std::to_string(i), first_diff(solo2[i], solo[i]));
        std::printf("timing A-B-A: solo A1 %.2f, pipelined %.2f aggregate, solo A2 %.2f tokens/s -> pipelined / mean(A1, A2) = x%.3f\n",
                    solo_rate, agg, r2, (solo_rate + r2) > 0 ? agg / ((solo_rate + r2) / 2) : 0.0);
    }
    fd_pipe.print("pipelined decode window");
    }
    fd_solo.print("solo decode");

    // ---- --slot-arm: a lane's conversation through a host slot into another lane ----------------------------------------
    if (slot_arm) {
        size_t pa = 0;   // the longest prompt (a kept state must be >= min_tokens), the continuation from another prompt
        for (size_t i = 1; i < prompts.size(); ++i) if (prompts[i].size() > prompts[pa].size()) pa = i;
        const size_t pb = (pa + 1) % prompts.size();
        const std::vector<int32_t>& A = prompts[pa];
        const std::vector<int32_t> tail(prompts[pb].begin(), prompts[pb].begin() + std::ptrdiff_t(std::min<size_t>(300, prompts[pb].size())));
        ie::Mimo26PrefixCache cache;
        const ie::Mimo26PrefixCache::Options def;
        const uint32_t la = lanes > 1 ? 1 : 0, lb = lanes > 2 ? 2 : 0;   // A runs on la, comes back on lb
        std::vector<std::vector<int32_t>> live(lanes);
        for (uint32_t l = 0; l < lanes; ++l) {   // every lane starts empty for the cache (its own bookkeeping)
            if (!select(l)) return 1;
            fwd.reset(); if (df) df->reset();
        }
        if (!select(0)) return 1;
        if (auto e = cache.init(fwd, df, def); !e.empty()) { std::printf("cache init: %s\n", e.c_str()); return 1; }
        auto request = [&](uint32_t l, const std::vector<int32_t>& ids, uint32_t& L, std::string& src) -> bool {   // mimo26_run_ids' prompt
            if (!select(l)) return false;
            if (auto e = cache.prepare(ids, live[l], L, src); !e.empty()) { std::printf("prepare: %s\n", e.c_str()); return false; }
            fwd.rewind(L); live[l].resize(L);
            if (L == 0) fwd.reset();
            if (df) { if (L == 0) df->reset(); else df->rewind(L); }
            Trace scratch;
            for (uint32_t at = L; at < ids.size(); ) if (!prefill_chunk(ids, at, scratch)) return false;
            live[l].insert(live[l].end(), ids.begin() + L, ids.end());
            if (auto e = cache.prompt_done(uint32_t(ids.size())); !e.empty()) { std::printf("prompt_done: %s\n", e.c_str()); return false; }
            return true;
        };
        auto greedy = [&](uint32_t l, uint32_t n, Trace& t) -> bool {
            t = Trace{}; t.logits.push_back(fnv(lg));
            int32_t id = argmax(lg);
            for (uint32_t k = 0; k < n; ++k) {
                t.tokens.push_back(id);
                if (!decode_step(id, t)) return false;
                live[l].push_back(id);
                id = argmax(lg);
            }
            finish_trace(t);
            return true;
        };
        // control: A on lane 0, 8 greedy steps, the continuation, 16 greedy steps -- no slot involved
        Trace c8, c16, t8, t16;
        uint32_t L = 0; std::string src;
        if (!request(0, A, L, src) || !greedy(0, 8, c8)) return 1;
        const std::vector<int32_t> cont = [&] { std::vector<int32_t> v = live[0]; v.insert(v.end(), tail.begin(), tail.end()); return v; }();
        if (!request(0, cont, L, src) || !greedy(0, 16, c16)) return 1;
        check(L == live[0].size() - tail.size() - 16 && src == "live", "slot arm control: the continuation is served from lane 0's live state",
              "served " + std::to_string(L) + " from " + src);
        // arm: A on lane la, an unrelated prompt on la (A kept in a host slot), the continuation on lane lb (A restored there)
        live[0].clear(); if (!select(0)) return 1;
        fwd.reset(); if (df) df->reset();
        cache.live_lost();
        if (!request(la, A, L, src) || !greedy(la, 8, t8)) return 1;
        check(first_diff(t8, c8).empty(), "slot arm: A on lane " + std::to_string(la) + " == A on lane 0 (8 steps)", first_diff(t8, c8));
        const std::vector<int32_t> kept = live[la];
        if (!request(la, prompts[pb], L, src)) return 1;
        const auto s1 = cache.stats();
        // (the held-out prompts share a chat-template prefix: the other prompt may reuse that much of A -- a branch, so A is kept first)
        check(L < kept.size() && s1.slots == 1, "slot arm: another prompt on lane " + std::to_string(la) + " keeps A (" + std::to_string(kept.size()) + " positions) in a host slot",
              std::to_string(s1.bytes >> 20) + " MiB in " + std::to_string(int(s1.last_save_ms)) + " ms; it reused " + std::to_string(L) + " of A");
        if (!request(lb, cont, L, src) || !greedy(lb, 16, t16)) return 1;
        const auto s2 = cache.stats();
        check(L == kept.size() && src.rfind("slot ", 0) == 0, "slot arm: the continuation on lane " + std::to_string(lb) + " restores A from its slot",
              "served " + std::to_string(L) + " from " + src + " in " + std::to_string(int(s2.last_load_ms)) + " ms");
        check(first_diff(t16, c16).empty(), "slot arm: the continuation's logits, 16 tokens" + std::string(df ? ", drafts and drafter bookkeeping" : "") +
                                            " identical to the control that never left lane 0", first_diff(t16, c16));
        cache.free_all();
    }

    if (df) df->free_all();
    fwd.free_all();
    std::printf("\nMIMO26 LANES TEST: %s\n", g_fail ? "FAILURE(S)" : "PASS");
    return g_fail ? 1 : 0;
}
