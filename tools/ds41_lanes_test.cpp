// tools/ds41_lanes_test.cpp -- ie-ds41-lanes-test: the gate of DeepSeek-V4.1-Flash lanes and the card pipe (P4 B6a,
// docs/deepseek41/P4_B6A_LANES.md; the shape of ie-mimo26-lanes-test). One runtime loaded with L lanes
// (Ds41Forward::ResidentOptions::lanes / lane_ctx). Every prompt runs SOLO on lane 0 first -- the pre-lane path -- then all of
// them over the L lanes: serially interleaved (one forward of one lane at a time, round robin, select_lane) or, with
// --pipeline, CARD-PIPELINED (Ds41Forward::pipe_*: one stage thread per card, lane B's step on card 0 while lane A's runs on
// card 1, every lane driving itself from the last stage's callback). A lane that finishes takes the next queued prompt
// mid-run (a fresh sequence on a dirty lane). Each sequence must be BYTE-IDENTICAL to its solo run: the logits after the
// prompt and after every decode step (fp32 bytes, FNV -- every row's for a verify step), the greedy tokens, the final position.
// --lookup runs the generator's prompt-lookup loop (Phase 58: a copy of >= 12 context tokens drafted, up to 7 drafts verified
// as one 2..8-row step, greedy acceptance, rollback) in both arms, so the pipe's multi-row verify steps and pipe_rollback are
// gated too. The CPU expert leg is off (IE_DS41_CPU_MISS=0: its split follows the stream-slot history, so its rows are not
// bit-reproducible run to run); --cpu-miss keeps it on (the served configuration) and then identity is reported, not gated.
// --aba runs the solo arm again after the pipelined one (A-B-A; a short warm-up precedes the first arm) and reports the
// steady-window aggregate tokens/s against the mean of the solo arms. --slot-arm: a lane's conversation kept in a host slot
// (the per-lane live checkpoints, the shared slots) when an unrelated prompt arrives on that lane, restored into ANOTHER
// lane, the continuation identical to a control that never left lane 0.
//   usage: ie-ds41-lanes-test <model_dir> (--prompt-file F | --user-file F)... [--lanes L] [--n N] [--ctx C] [--lane-ctx C2]
//          [--chunk C] [--lookup] [--cpu-miss] [--pipeline] [--aba] [--slot-arm] [--tables DIR] [--ranking FILE]
//   --prompt-file: the text as is (a rendered prompt); --user-file: the text as one user turn in the V4.1 chat format (no
//   thinking). The engram tables default to the model directory, the ranking to ie_ranking_decode_chat.txt beside the model
//   (else ie_ranking_heldout.txt), the expert tail file beside the model is used when present -- ie serve's configuration.
#include "ie/deepseek41.hpp"
#include "ie/deepseek41_engram.hpp"
#include "ie/deepseek41_forward.hpp"
#include "ie/deepseek41_prompt.hpp"
#include "ie/expert_stream.hpp"      // ds4_expert_priority_read_layers
#include "ie/ngram_draft.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <dirent.h>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
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
double ms_since(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
int32_t argmax_row(const std::vector<float>& lg, size_t V, size_t r) { return int32_t(std::max_element(lg.begin() + std::ptrdiff_t(r * V), lg.begin() + std::ptrdiff_t((r + 1) * V)) - (lg.begin() + std::ptrdiff_t(r * V))); }
bool exists(const std::string& p) { return std::ifstream(p).good(); }
double mem_available_gib() {
    std::ifstream f("/proc/meminfo"); std::string k; unsigned long long v = 0; std::string unit;
    while (f >> k >> v >> unit) if (k == "MemAvailable:") return double(v) / 1048576.0;
    return 0;
}

// The prompt's chunks (pos0, T), the generator's plain plan: pieces of `cap` from 0 (the first even), never leaving 2..8 rows,
// a remainder of 1..8 fed one token at a time. Solo and the lanes use the same plan, so their steps are the same launches.
std::vector<std::pair<uint32_t, uint32_t>> plan_chunks(uint32_t P, uint32_t cap) {
    std::vector<std::pair<uint32_t, uint32_t>> ch;
    uint32_t off = 0;
    while (P - off > (off == 0 ? 0u : ie::kDs41MaxDecodeRows)) {
        uint32_t t = std::min(cap, P - off); const uint32_t rest = P - off - t;
        if (rest >= 1 && rest <= ie::kDs41MaxDecodeRows && t > 2 * (ie::kDs41MaxDecodeRows + 1)) t -= ie::kDs41MaxDecodeRows + 1 - rest;
        if (off == 0 && (t & 1u)) { if (t == 1) break; --t; }
        ch.push_back({off, t}); off += t;
    }
    for (; off < P; ++off) ch.push_back({off, 1});
    return ch;
}

// B0 M1 (the premise): this process's DRM clients' engine cycles from /proc/self/fdinfo (xe: drm-cycles-<class> and
// drm-total-cycles-<class> per client); deltas accumulated over begin/end windows.
struct FdBusy {
    struct Cnt { uint64_t cyc = 0, tot = 0; uint32_t cap = 1; };
    using Snap = std::map<std::string, Cnt>;                    // "pdev|client|class"
    std::map<std::string, std::pair<uint64_t, uint64_t>> acc;
    std::map<std::string, uint32_t> caps;
    Snap at;
    static Snap snap() {
        Snap s;
        DIR* d = opendir("/proc/self/fdinfo");
        if (!d) return s;
        std::map<std::string, bool> seen;
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
    void print(const char* label) const {
        std::map<std::string, std::map<std::string, double>> by, cl;
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
            std::printf("  (per client the max over classes, summed: %.1f%%)\n", dream);
        }
        if (by.empty()) std::printf("fdinfo busy [%s]: no DRM client cycles in /proc/self/fdinfo\n", label);
    }
};

// One sequence's record: the logits after its prompt and after every decode step (every row's for a verify step), its
// tokens, its final position; the timings (not compared).
struct Trace {
    std::vector<uint64_t> logits;
    std::vector<int32_t> tokens;
    uint32_t n_pos = 0;
    double prefill_ms = 0, decode_ms = 0;
    std::vector<double> step_ms; std::vector<uint32_t> step_tok;   // each decode step: its wall and the tokens it committed
    uint32_t passes = 0, plain = 0, accepted = 0;                  // lookup: verify passes, one-row steps, accepted drafts
};
std::string first_diff(const Trace& a, const Trace& b) {
    for (size_t i = 0; i < std::min(a.logits.size(), b.logits.size()); ++i)
        if (a.logits[i] != b.logits[i]) return "logits differ first at step " + std::to_string(i);
    if (a.logits.size() != b.logits.size()) return "step counts differ (" + std::to_string(a.logits.size()) + " vs " + std::to_string(b.logits.size()) + ")";
    if (a.tokens != b.tokens) return "tokens differ";
    if (a.n_pos != b.n_pos) return "final positions differ";
    return {};
}

// The prompt-lookup policy the generator ships (docs/deepseek41/97): a copy of >= 12 tokens, up to 7 drafts per pass.
constexpr uint32_t kLookupK = 7, kLookupMin = 12;
constexpr uint32_t kWarmSteps = 4;   // decode steps of each sequence left out of the timing (first slots, oneDNN primitives)

// One sequence's decode state, the same for solo, interleaved and pipelined: the caches hold [0, pos); `id` is the token
// whose forward is due (already in `tokens`); a verify step's rows are [id, drafts...].
struct Seq {
    const std::vector<int32_t>* prompt = nullptr;
    Trace* t = nullptr;
    uint32_t at = 0;                       // prefill cursor (the next chunk index)
    std::vector<std::pair<uint32_t, uint32_t>> chunks;
    uint32_t pos = 0, k = 0;               // positions in the caches; tokens generated so far
    int32_t id = -1;
    std::vector<int32_t> rows;             // the decode step in flight (1 row, or 1 + drafts)
    ie::Ds41NgramIndex idx;
    std::chrono::steady_clock::time_point t_sub;
    bool decoding() const { return at >= chunks.size(); }
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: ie-ds41-lanes-test <model_dir> (--prompt-file F | --user-file F)... [--lanes L] [--n N] [--ctx C] [--lane-ctx C2] "
                             "[--chunk C] [--lookup] [--cpu-miss] [--pipeline] [--aba] [--slot-arm] [--tables DIR] [--ranking FILE]\n");
        return 2;
    }
    const std::string dir = argv[1];
    std::vector<std::pair<std::string, bool>> files;   // (path, rendered as a user turn)
    std::string tables = dir, ranking_path;
    uint32_t lanes = 2, N = 64, ctx = 16384, lane_ctx = 0, chunk = 2048;
    bool lookup = false, cpu_miss = false, pipeline = false, aba = false, slot_arm = false;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--prompt-file") files.push_back({val(), false});
        else if (a == "--user-file") files.push_back({val(), true});
        else if (a == "--lanes") lanes = uint32_t(std::atol(val().c_str()));
        else if (a == "--n") N = uint32_t(std::atol(val().c_str()));
        else if (a == "--ctx") ctx = uint32_t(std::atol(val().c_str()));
        else if (a == "--lane-ctx") lane_ctx = uint32_t(std::atol(val().c_str()));
        else if (a == "--chunk") chunk = uint32_t(std::atol(val().c_str()));
        else if (a == "--lookup") lookup = true;
        else if (a == "--cpu-miss") cpu_miss = true;
        else if (a == "--pipeline") pipeline = true;
        else if (a == "--aba") aba = true;
        else if (a == "--slot-arm") slot_arm = true;
        else if (a == "--tables") tables = val();
        else if (a == "--ranking") ranking_path = val();
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (files.empty() || lanes == 0 || N < 2) { std::fprintf(stderr, "need a prompt, --lanes >= 1, --n >= 2\n"); return 2; }
    if (!cpu_miss) setenv("IE_DS41_CPU_MISS", "0", 1);
    // ie serve's configuration (Engine::ds41_load): the expert tail file beside the model, the chat decode ranking when present
    if (const char* ef = std::getenv("IE_DS41_EXPERT_FILE"); ef && std::string(ef) == "0") unsetenv("IE_DS41_EXPERT_FILE");
    else if (!ef && exists(dir + "/ie_experts_tail.ieslot")) setenv("IE_DS41_EXPERT_FILE", (dir + "/ie_experts_tail.ieslot").c_str(), 0);
    if (ranking_path.empty()) ranking_path = exists(dir + "/ie_ranking_decode_chat.txt") ? dir + "/ie_ranking_decode_chat.txt" : dir + "/ie_ranking_heldout.txt";
    const double mem0 = mem_available_gib();

    ie::DeepSeek41Model m;
    if (auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    ie::Ds41EngramTables tb;
    if (auto e = tb.load(tables); !e.empty()) { std::fprintf(stderr, "engram tables (%s): %s\n", tables.c_str(), e.c_str()); return 1; }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_hf_json(dir + "/tokenizer.json", dir + "/tokenizer_config.json"); !e.empty()) { std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1; }
    std::vector<std::vector<int32_t>> prompts; std::vector<std::string> names;
    const uint32_t cap_small = lanes > 1 ? std::min(ctx, lane_ctx ? lane_ctx : 32768u) : ctx;   // ResidentOptions::lane_ctx's default
    for (const auto& [f, user] : files) {
        std::ifstream in(f, std::ios::binary);
        if (!in) { std::fprintf(stderr, "cannot read %s\n", f.c_str()); return 2; }
        std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        if (user) {
            std::vector<ie::Ds41ChatMessage> msgs = {{"user", text, "", false, false, "", "", ""}};
            ie::Ds41PromptOptions po; po.thinking = false; std::string err;
            text = ie::ds41_encode_messages(msgs, po, err);
            if (!err.empty()) { std::fprintf(stderr, "%s: prompt format: %s\n", f.c_str(), err.c_str()); return 2; }
        }
        prompts.push_back(tok.encode(text, /*allow_special=*/true)); names.push_back(f);
        if (prompts.back().size() < 2 || prompts.back().size() + N + ie::kDs41MaxDecodeRows > cap_small) {
            std::fprintf(stderr, "%s: %zu tokens + %u + a verify block do not fit ctx %u\n", f.c_str(), prompts.back().size(), N, cap_small); return 2; }
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
    std::vector<std::vector<uint32_t>> ranking;
    if (exists(ranking_path)) { if (auto e = ie::ds4_expert_priority_read_layers(ranking_path, m.config().n_routed_experts, m.config().n_layers, ranking); !e.empty()) { std::fprintf(stderr, "ranking: %s\n", e.c_str()); return 1; } }
    else std::fprintf(stderr, "no ranking file at %s: index-order placement (slower)\n", ranking_path.c_str());
    const auto t_init = std::chrono::steady_clock::now();
    ie::Ds41Forward fwd; ie::Ds41Forward::ResidentOptions ro;
    ro.max_tokens = ctx; ro.head_fp8 = true; ro.lanes = lanes;   // Phase 54: the server's FP8 head
    if (lane_ctx) ro.lane_ctx = lane_ctx;
    if (auto e = fwd.init_resident(qs, m, tb, ranking, ro); !e.empty()) { std::printf("init_resident: %s\n", e.c_str()); return 1; }
    fwd.set_logits_last_only(true);
    if (lookup) fwd.set_multi_row_decode(true);   // the generator's lookup loop admits the 2..8-row verify steps this way
    const uint32_t cap = std::min(chunk, fwd.forward_capacity());
    const double mem1 = mem_available_gib();
    std::printf("ds41: %zu card(s), init %.1f s, %u lane(s): lane 0 ctx %u, the others ctx %u; chunk %u; lookup %s, CPU miss leg %s; expert file %s; ranking %s\n",
                qs.size(), ms_since(t_init) / 1000, fwd.n_lanes(), fwd.lane_capacity(0), lanes > 1 ? fwd.lane_capacity(1) : fwd.lane_capacity(0), cap,
                lookup ? "ON" : "off", cpu_miss ? "ON" : "off", std::getenv("IE_DS41_EXPERT_FILE") ? "on" : "off", ranking_path.c_str());
    std::printf("MemAvailable: %.1f GiB before the load, %.1f after\n", mem0, mem1);
    for (uint32_t c = 0; c < fwd.n_cards(); ++c) {
        const auto ci = fwd.card_info(c);
        std::printf("card %u: layers %u-%u, %u static + %u stream slots per layer, %u pinned; dense %.2f GB, expert VRAM %.2f GB; one extra lane %.1f MiB\n",
                    c, ci.first_layer, ci.first_layer + ci.n_layers - 1, ci.n_static, ci.n_stream, ci.n_pinned, ci.dense_bytes / 1e9, ci.expert_vram_bytes / 1e9, fwd.lane_bytes(c) / 1048576.0);
    }
    if (lanes > 1) {   // the logged lane VRAM against the state arithmetic (rings + latents + index keys + the ratio-2 halves)
        const auto& cfg = m.config();
        for (uint32_t c = 0; c < fwd.n_cards(); ++c) {
            const auto ci = fwd.card_info(c); uint64_t want = 0;
            for (uint32_t L = ci.first_layer; L < ci.first_layer + ci.n_layers; ++L) {
                const auto& k = m.layers()[L].kind;
                want += uint64_t(cfg.window_size) * cfg.head_dim * 4;
                if (k.is_kv_source) { want += uint64_t(fwd.lane_capacity(1)) * (cfg.head_dim + cfg.index_head_dim) * 4; if (k.compress_ratio > 1) want += 2ull * k.compress_ratio * cfg.head_dim * 4; }
            }
            check(fwd.lane_bytes(c) == want, "card " + std::to_string(c) + ": one extra lane's device state equals the arithmetic", std::to_string(fwd.lane_bytes(c)) + " B = " + std::to_string(fwd.lane_bytes(c) / 1048576.0) + " MiB");
        }
    }

    const size_t V = m.config().vocab_size;
    auto select = [&](uint32_t l) -> bool { if (auto e = fwd.select_lane(l); !e.empty()) { std::printf("select lane %u: %s\n", l, e.c_str()); return false; } return true; };
    // with the CPU leg on, a lane's rows are not bit-reproducible run to run: identity is reported, not gated
    auto check_id = [&](bool ok, const std::string& what, const std::string& detail) {
        if (!cpu_miss) { check(ok, what, detail); return; }
        std::printf("[info] %s: %s%s%s\n", what.c_str(), ok ? "identical" : "DIFFERS", detail.empty() ? "" : "  ", detail.c_str());
        std::fflush(stdout);
    };
    auto print_tier = [&](const char* label, uint32_t n_l) {
        for (uint32_t l = 0; l < n_l; ++l)
            for (uint32_t c = 0; c < fwd.n_cards(); ++c) {
                const auto& t = fwd.lane_tier_stats(l, c);
                if (!t.steps) continue;
                std::printf("tier [%s] lane %u card %u: %llu decode steps (%.2f rows/step); per step %.1f static, %.1f pinned, of them %.1f stream-slot hits (%.1f%%), %.1f on the CPU, %.1f mmap; moe %.1f ms/step"
                            "; CPU leg %.2f ms (compute %.2f, %.3f ms/expert), PCIe %.1f MiB, mmap %.1f MiB per step\n",
                            label, l, c, (unsigned long long)t.steps, double(t.rows) / t.steps, double(t.experts_static) / t.steps, double(t.experts_pinned) / t.steps,
                            double(t.stream_hits) / t.steps, t.experts_pinned ? 100.0 * double(t.stream_hits) / double(t.experts_pinned) : 0.0, double(t.experts_cpu) / t.steps,
                            double(t.experts_mmap) / t.steps, t.moe_ms / t.steps, t.cpu_ms / t.steps, t.cpu_work_ms / t.steps,
                            t.experts_cpu ? t.cpu_work_ms / double(t.experts_cpu) : 0.0, double(t.bytes_pinned) / t.steps / 1048576.0, double(t.bytes_mmap) / t.steps / 1048576.0);
            }
        fwd.reset_lane_tier_stats();
    };
    auto seq_init = [&](Seq& s, const std::vector<int32_t>& prompt, Trace& t) {
        s = Seq{}; s.prompt = &prompt; s.t = &t; s.chunks = plan_chunks(uint32_t(prompt.size()), cap); s.at = 0; s.pos = 0; s.k = 0; s.id = -1;
    };
    // the prompt is in: the first token from its last logits, the lookup index over the prompt + it
    auto seq_prompt_done = [&](Seq& s, const std::vector<float>& lg) {
        s.t->logits.push_back(fnv(lg));
        s.pos = uint32_t(s.prompt->size()); s.id = argmax_row(lg, V, 0); s.t->tokens.push_back(s.id); s.k = 1;
        if (lookup) { s.idx.reset(*s.prompt); s.idx.push(s.id); }
    };
    // the next decode step's rows: [id] or [id, drafts...] (a pass commits at most drafts + 1 tokens: capped to the budget)
    auto seq_next_rows = [&](Seq& s) {
        s.rows.assign(1, s.id);
        if (!lookup || s.k + 1 >= N) return;
        std::vector<int32_t> d = s.idx.draft(kLookupK, kLookupMin);
        if (d.size() > N - s.k - 1) d.resize(N - s.k - 1);
        s.rows.insert(s.rows.end(), d.begin(), d.end());
    };
    // a decode step's logits are in: record them, commit the tokens, and return how many positions to keep (rows accepted + 1)
    auto seq_step_done = [&](Seq& s, const std::vector<float>& lg, double step_wall) -> uint32_t {
        s.t->logits.push_back(fnv(lg));
        const uint32_t m_ = uint32_t(s.rows.size()) - 1;   // drafts
        uint32_t L = 0; int32_t next = -1;
        for (uint32_t r = 0; r <= m_; ++r) {
            const int32_t a = argmax_row(lg, V, r);
            if (r == m_ || a != s.rows[1 + r]) { next = a; break; }
            ++L; s.t->tokens.push_back(s.rows[1 + r]); if (lookup) s.idx.push(s.rows[1 + r]); ++s.k;
        }
        s.id = next; s.t->tokens.push_back(s.id); if (lookup) s.idx.push(s.id); ++s.k;
        if (m_) { ++s.t->passes; s.t->accepted += L; } else ++s.t->plain;
        s.t->decode_ms += step_wall; s.t->step_ms.push_back(step_wall); s.t->step_tok.push_back(L + 1);
        return 1 + L;
    };

    // ---- the serial API: one sequence's step on the selected lane (solo, and the interleaved arm) --------------------------
    std::vector<float> lg;
    auto serial_prefill_chunk = [&](Seq& s) -> bool {
        const auto t0 = std::chrono::steady_clock::now();
        const auto [p0, T] = s.chunks[s.at];
        if (auto e = fwd.forward(s.prompt->data() + p0, T, p0, lg); !e.empty()) { std::printf("prefill at %u (%u rows): %s\n", p0, T, e.c_str()); return false; }
        s.t->prefill_ms += ms_since(t0);
        if (++s.at == s.chunks.size()) seq_prompt_done(s, lg);
        return true;
    };
    auto serial_decode_step = [&](Seq& s) -> bool {
        seq_next_rows(s);
        const auto t0 = std::chrono::steady_clock::now();
        const bool verify = s.rows.size() > 1;
        if (verify) fwd.set_logits_last_only(false);
        const std::string e = fwd.forward(s.rows.data(), uint32_t(s.rows.size()), s.pos, lg);
        if (verify) fwd.set_logits_last_only(true);
        if (!e.empty()) { std::printf("decode at %u (%zu rows): %s\n", s.pos, s.rows.size(), e.c_str()); return false; }
        const uint32_t keep = seq_step_done(s, lg, ms_since(t0));
        if (verify) if (auto re = fwd.rollback_to(s.pos + keep); !re.empty()) { std::printf("rollback to %u: %s\n", s.pos + keep, re.c_str()); return false; }
        s.pos += keep;
        return true;
    };
    auto finish_trace = [&](Seq& s) { s.t->n_pos = fwd.n_pos(); };
    // solo: tokens/s over the decode steps after the first kWarmSteps -- of every prompt, or of the prompts `only` names
    auto steady_rate = [&](const std::vector<Trace>& v, const std::vector<size_t>* only = nullptr) {
        double ms = 0; uint32_t n = 0;
        for (size_t i = 0; i < v.size(); ++i) {
            if (only && std::find(only->begin(), only->end(), i) == only->end()) continue;
            const auto& t = v[i];
            for (size_t k = kWarmSteps; k < t.step_ms.size(); ++k) { ms += t.step_ms[k]; n += t.step_tok[k]; }
        }
        return ms > 0 ? n / (ms / 1000) : 0.0;
    };
    auto summary = [&](const Trace& t) {
        std::string s = std::to_string(t.tokens.size()) + " tokens";
        if (lookup) s += ", " + std::to_string(t.passes) + " verify passes (" + std::to_string(t.accepted) + " drafts accepted), " + std::to_string(t.plain) + " one-row steps";
        return s;
    };

    // ---- solo: every prompt alone on lane 0 (the pre-lane path) --------------------------------------------------------
    FdBusy fd_solo, fd_pipe;
    auto run_solo = [&](std::vector<Trace>& out, const char* label) -> double {   // wall ms, < 0 on a failure
        out.assign(prompts.size(), Trace{});
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < prompts.size(); ++i) {
            if (!select(0)) return -1;
            fwd.reset_state();
            Seq s; seq_init(s, prompts[i], out[i]);
            while (!s.decoding()) if (!serial_prefill_chunk(s)) return -1;
            fd_solo.begin();
            while (s.k < N) if (!serial_decode_step(s)) return -1;
            fd_solo.end();
            finish_trace(s);
            const Trace& t = out[i];
            std::printf("solo%s %zu (%s): %zu prompt tokens in %zu chunks, prefill %.0f ms, decode %.1f ms/token (%zu steps), %s; tokens %d %d %d %d ...\n", label, i, names[i].c_str(),
                        prompts[i].size(), s.chunks.size(), t.prefill_ms, t.decode_ms / std::max<size_t>(1, t.tokens.size() - 1), t.step_ms.size(), summary(t).c_str(),
                        t.tokens[0], t.tokens[1 % t.tokens.size()], t.tokens[2 % t.tokens.size()], t.tokens[3 % t.tokens.size()]);
        }
        return ms_since(t0);
    };
    if (aba) {   // warm-up before the first arm: a short prefill and a few decode steps, discarded
        if (!select(0)) return 1;
        fwd.reset_state();
        const std::vector<int32_t> w(prompts[0].begin(), prompts[0].begin() + std::ptrdiff_t(std::min<size_t>(512, prompts[0].size()) & ~size_t(1)));
        Trace t; Seq s; seq_init(s, w, t);
        while (!s.decoding()) if (!serial_prefill_chunk(s)) return 1;
        for (uint32_t k = 0; k < 8 && s.k < N; ++k) if (!serial_decode_step(s)) return 1;
        fwd.reset_lane_tier_stats();
        std::printf("warm-up: %zu-token prefill + 8 decode steps, discarded\n", w.size());
    }
    std::vector<Trace> solo;
    const double solo_ms = run_solo(solo, aba ? " A1" : "");
    if (solo_ms < 0) return 1;
    print_tier("solo", 1);
    const double solo_rate = steady_rate(solo);

    // ---- the lanes: the prompts queued, admitted to idle lanes, a finished lane taking the next prompt --------------------
    std::vector<Trace> inter(prompts.size());
    std::vector<Seq> seq(lanes); std::vector<int> seq_of(lanes, -1);
    std::deque<size_t> queue; for (size_t i = 0; i < prompts.size(); ++i) queue.push_back(i);
    uint32_t steps = 0, switches = 0, readmits = 0;
    auto admit = [&]() {   // every idle lane takes the next queued prompt; returns the lanes that got one
        std::vector<uint32_t> got;
        for (uint32_t l = 0; l < lanes && !queue.empty(); ++l) {
            if (seq_of[l] >= 0 && seq[l].k < N) continue;                         // busy
            if (seq_of[l] >= 0) ++readmits;                                          // a finished lane takes the next prompt (a dirty lane)
            seq_of[l] = int(queue.front()); seq_init(seq[l], prompts[queue.front()], inter[size_t(queue.front())]); queue.pop_front();
            got.push_back(l);
        }
        return got;
    };
    auto busy_lanes = [&]() { uint32_t n = 0; for (uint32_t l = 0; l < lanes; ++l) if (seq_of[l] >= 0 && seq[l].k < N) ++n; return n; };
    if (!pipeline) {
        // ---- interleaved: one lane's forward at a time, round robin over the busy lanes ----------------------------------
        for (uint32_t l : admit()) { if (!select(l)) return 1; fwd.reset_state(); }
        const auto t_inter = std::chrono::steady_clock::now();
        int last = -1; std::string order;
        for (uint32_t l = 0; busy_lanes(); l = (l + 1) % lanes) {
            if (seq_of[l] < 0 || seq[l].k >= N) continue;
            if (!select(l)) return 1;
            if (int(l) != last) ++switches;
            last = int(l);
            if (order.size() < 160) order += std::to_string(l);
            ++steps;
            Seq& s = seq[l];
            if (!s.decoding()) { if (!serial_prefill_chunk(s)) return 1; continue; }
            if (!serial_decode_step(s)) return 1;
            if (s.k >= N) {
                finish_trace(s);
                for (uint32_t nl : admit()) { if (!select(nl)) return 1; fwd.reset_state(); }   // mid-run: the finished lane takes the next queued prompt
                if (!select(l)) return 1;
            }
        }
        const double inter_ms = ms_since(t_inter);
        std::printf("interleaved: %u forwards over %u lane(s), %u lane switches, %u mid-run re-admissions; lane order %s%s\n",
                    steps, lanes, switches, readmits, order.c_str(), order.size() >= 160 ? " ..." : "");
        print_tier("interleaved", lanes);
        for (size_t i = 0; i < prompts.size(); ++i) {
            const std::string d = first_diff(inter[i], solo[i]);
            check_id(d.empty(), "sequence " + std::to_string(i) + " (" + names[i] + ") interleaved == solo: " + std::to_string(inter[i].logits.size()) + " logits rows, " + summary(inter[i]), d);
        }
        if (lanes > 1) check(readmits > 0 || prompts.size() <= lanes, "a lane finished mid-run and took the next queued prompt", std::to_string(readmits) + " re-admission(s)");
        uint32_t n_tok = 0; for (const auto& t : solo) n_tok += uint32_t(t.tokens.size());
        std::printf("timing: solo %.1f s for all prompts, interleaved %.1f s: interleaved/solo wall %.3f; aggregate %.2f vs %.2f tokens/s over the whole run (serial interleave: no gain expected)\n",
                    solo_ms / 1000, inter_ms / 1000, inter_ms / solo_ms, n_tok / (inter_ms / 1000), n_tok / (solo_ms / 1000));
    } else {
        // ---- pipelined: every lane drives itself from the last stage's callback; card 0 runs one lane while card 1 runs another --
        std::mutex cb_mu;                                  // the callback's state (one stage thread calls it; main holds it while seeding)
        std::string cb_err;
        const auto t_pipe = std::chrono::steady_clock::now();
        std::vector<uint32_t> wave;                        // the first wave's lanes: the steady window is while all of them decode
        std::vector<size_t> wave_p;                        // ... and the prompts they run (the window's solo baseline)
        std::vector<uint32_t> dec_n(lanes, 0);
        std::vector<std::pair<double, uint32_t>> done_ev;  // every decode completion (ms since t_pipe, tokens committed) while the window may be open
        double win0 = -1, win1 = -1;
        auto fail = [&](const std::string& e) { if (cb_err.empty()) cb_err = e; };
        auto submit = [&](uint32_t l) -> bool {           // the lane's next step: its next prefill chunk, or a decode step
            Seq& s = seq[l];
            std::string e;
            s.t_sub = std::chrono::steady_clock::now();
            if (!s.decoding()) { const auto [p0, T] = s.chunks[s.at]; e = fwd.pipe_submit(l, s.prompt->data() + p0, T, p0, false); }
            else { seq_next_rows(s); e = fwd.pipe_submit(l, s.rows.data(), uint32_t(s.rows.size()), s.pos, s.rows.size() > 1); }
            if (!e.empty()) { fail(e); return false; }
            return true;
        };
        auto start = [&](uint32_t l) -> bool {            // a fresh sequence on a possibly dirty lane (in the callback)
            if (auto e = fwd.pipe_reset_lane(l); !e.empty()) { fail(e); return false; }
            return submit(l);
        };
        auto on_done = [&](uint32_t l, const std::vector<float>& logits) {
            std::lock_guard<std::mutex> lk(cb_mu);
            if (!cb_err.empty()) return;
            Seq& s = seq[l];
            const double step = ms_since(s.t_sub);
            ++steps;
            if (!s.decoding()) {
                s.t->prefill_ms += step;
                if (++s.at < s.chunks.size()) { submit(l); return; }
                seq_prompt_done(s, logits);
            } else {
                const bool verify = s.rows.size() > 1;
                const uint32_t keep = seq_step_done(s, logits, step);
                if (verify) if (auto e = fwd.pipe_rollback(l, s.pos + keep); !e.empty()) { fail("pipe_rollback to " + std::to_string(s.pos + keep) + ": " + e); return; }
                s.pos += keep;
                const double at = ms_since(t_pipe);
                ++dec_n[l];
                if (win1 < 0) {
                    done_ev.push_back({at, keep});
                    if (win0 < 0 && std::all_of(wave.begin(), wave.end(), [&](uint32_t w) { return dec_n[w] >= kWarmSteps; })) { win0 = at; fd_pipe.begin(); }
                }
            }
            if (s.k < N) { submit(l); return; }
            s.t->n_pos = fwd.lane_pos(l);
            if (win1 < 0 && win0 >= 0 && std::find(wave.begin(), wave.end(), l) != wave.end()) { win1 = ms_since(t_pipe); fd_pipe.end(); }
            for (uint32_t nl : admit()) start(nl);        // mid-run: the finished lane takes the next queued prompt
        };
        for (uint32_t l = 0; l < lanes; ++l) { if (!select(l)) return 1; fwd.reset_state(); }   // every lane starts empty
        if (!select(0)) return 1;
        if (auto e = fwd.pipe_start(on_done); !e.empty()) { std::printf("pipe_start: %s\n", e.c_str()); return 1; }
        {
            std::lock_guard<std::mutex> lk(cb_mu);
            wave = admit();
            for (uint32_t l : wave) wave_p.push_back(size_t(seq_of[l]));
            for (uint32_t l : wave) if (!submit(l)) break;
        }
        const std::string pe = fwd.pipe_stop();
        const double pipe_ms = ms_since(t_pipe);
        if (!pe.empty() || !cb_err.empty()) { std::printf("pipeline: %s%s\n", pe.c_str(), cb_err.c_str()); fwd.free_resident(); return 1; }
        std::printf("pipelined: %u forwards over %u lane(s), %u mid-run re-admissions, %.1f s\n", steps, lanes, readmits, pipe_ms / 1000);
        print_tier("pipelined", lanes);
        for (size_t i = 0; i < prompts.size(); ++i) {
            const std::string d = first_diff(inter[i], solo[i]);
            check_id(d.empty(), "sequence " + std::to_string(i) + " (" + names[i] + ") pipelined == solo: " + std::to_string(inter[i].logits.size()) + " logits rows, " + summary(inter[i]), d);
        }
        if (lanes > 1) check(readmits > 0 || prompts.size() <= lanes, "a lane finished mid-run and took the next queued prompt", std::to_string(readmits) + " re-admission(s)");
        if (lookup) { uint32_t p = 0; for (const auto& t : inter) p += t.passes; check(p > 0, "verify steps of 2..8 rows went through the pipe", std::to_string(p) + " passes"); }
        // the steady window: from every first-wave lane's kWarmSteps-th decode step to the first of them finishing
        uint32_t in_win = 0;
        for (const auto& [t, n] : done_ev) if (t > win0 && t <= win1) in_win += n;
        const double agg = win1 > win0 && win0 >= 0 ? in_win / ((win1 - win0) / 1000) : 0.0;
        double lat = 0; uint32_t nlat = 0;
        for (const auto& t : inter) for (size_t k = kWarmSteps; k < t.step_ms.size(); ++k) { lat += t.step_ms[k]; ++nlat; }
        std::printf("timing: steady window %.1f s with all %zu first-wave lanes decoding: %u tokens = %.2f tokens/s aggregate, %.1f ms per lane step (submit to logits); "
                    "solo%s %.2f tokens/s -> x%.3f\n", (win1 - win0) / 1000, wave.size(), in_win, agg, nlat ? lat / nlat : 0.0, aba ? " A1" : "", solo_rate, solo_rate > 0 ? agg / solo_rate : 0.0);
        // the window holds the first wave's prompts only, so the like-for-like solo rate is over those prompts; and the
        // whole run (every prompt's prefill and decode, the re-admissions and the one-lane tail) against the solo arm's wall
        std::string wl; for (size_t p : wave_p) wl += (wl.empty() ? "" : ", ") + std::to_string(p);
        const double solo_rate_w = steady_rate(solo, &wave_p);
        std::printf("timing: the window against solo%s over the first wave's own prompts (%s): %.2f tokens/s -> x%.3f; whole run: solo%s %.1f s, pipelined %.1f s -> x%.3f\n",
                    aba ? " A1" : "", wl.c_str(), solo_rate_w, solo_rate_w > 0 ? agg / solo_rate_w : 0.0, aba ? " A1" : "", solo_ms / 1000, pipe_ms / 1000, solo_ms / pipe_ms);
        if (aba) {
            std::vector<Trace> solo2;
            const double solo2_ms = run_solo(solo2, " A2");
            if (solo2_ms < 0) return 1;
            print_tier("solo A2", 1);
            const double r2 = steady_rate(solo2), r2w = steady_rate(solo2, &wave_p);
            for (size_t i = 0; i < prompts.size(); ++i) check_id(first_diff(solo2[i], solo[i]).empty(), "solo A2 == solo A1, sequence " + std::to_string(i), first_diff(solo2[i], solo[i]));
            std::printf("timing A-B-A: solo A1 %.2f, pipelined %.2f aggregate, solo A2 %.2f tokens/s -> pipelined / mean(A1, A2) = x%.3f\n",
                        solo_rate, agg, r2, (solo_rate + r2) > 0 ? agg / ((solo_rate + r2) / 2) : 0.0);
            std::printf("timing A-B-A over the first wave's own prompts (%s): solo A1 %.2f, pipelined %.2f, solo A2 %.2f tokens/s -> x%.3f\n",
                        wl.c_str(), solo_rate_w, agg, r2w, (solo_rate_w + r2w) > 0 ? agg / ((solo_rate_w + r2w) / 2) : 0.0);
            std::printf("timing A-B-A whole run: solo A1 %.1f s, pipelined %.1f s, solo A2 %.1f s -> mean(A1, A2) / pipelined = x%.3f\n",
                        solo_ms / 1000, pipe_ms / 1000, solo2_ms / 1000, (solo_ms + solo2_ms) / 2 / pipe_ms);
        }
        fd_pipe.print("pipelined decode window");
    }
    fd_solo.print("solo decode");

    // ---- --slot-arm: a lane's conversation through a host slot into another lane (the per-lane live checkpoints) --------
    if (slot_arm) {
        size_t pa = 0;   // the longest prompt (a kept state must be >= min_slot_tokens), the continuation from another prompt
        for (size_t i = 1; i < prompts.size(); ++i) if (prompts[i].size() > prompts[pa].size()) pa = i;
        const size_t pb = (pa + 1) % prompts.size();
        const std::vector<int32_t>& A = prompts[pa];
        const std::vector<int32_t> tail(prompts[pb].begin() + 1, prompts[pb].begin() + 1 + std::ptrdiff_t(std::min<size_t>(300, prompts[pb].size() - 1)));   // (past its BOS)
        const uint32_t la = lanes > 1 ? 1 : 0, lb = lanes > 2 ? 2 : 0;   // A runs on la, comes back on lb
        for (uint32_t l = 0; l < lanes; ++l) { if (!select(l)) return 1; fwd.reset_state(); }
        if (!select(0)) return 1;
        if (auto e = fwd.set_prefix_cache(true); !e.empty()) { std::printf("set_prefix_cache: %s\n", e.c_str()); return 1; }
        std::vector<std::vector<int32_t>> live(lanes);
        auto request = [&](uint32_t l, const std::vector<int32_t>& ids, uint32_t& L, std::string& src) -> bool {   // the generator's prompt path
            if (!select(l)) return false;
            if (auto e = fwd.prefix_prepare(ids, L, &src); !e.empty()) { std::printf("prefix_prepare: %s\n", e.c_str()); return false; }
            // the rest as the generator runs it: chunks of cap from L (a pos0 = 0 first chunk even), never leaving 2..8, singles at the end
            std::vector<std::pair<uint32_t, uint32_t>> ch; uint32_t off = L; const uint32_t P = uint32_t(ids.size());
            while (P - off > (off == 0 ? 0u : ie::kDs41MaxDecodeRows)) {
                uint32_t t = std::min(cap, P - off); const uint32_t rest = P - off - t;
                if (rest >= 1 && rest <= ie::kDs41MaxDecodeRows && t > 2 * (ie::kDs41MaxDecodeRows + 1)) t -= ie::kDs41MaxDecodeRows + 1 - rest;
                if (off == 0 && (t & 1u)) { if (t == 1) break; --t; }
                ch.push_back({off, t}); off += t;
            }
            for (; off < P; ++off) ch.push_back({off, 1});
            for (const auto& [p0, T] : ch) if (auto e = fwd.forward(ids.data() + p0, T, p0, lg); !e.empty()) { std::printf("forward at %u: %s\n", p0, e.c_str()); return false; }
            live[l] = ids;
            if (auto e = fwd.prefix_checkpoint(); !e.empty()) { std::printf("prefix_checkpoint: %s\n", e.c_str()); return false; }
            return true;
        };
        auto greedy = [&](uint32_t l, uint32_t n, Trace& t) -> bool {
            t = Trace{}; t.logits.push_back(fnv(lg));
            int32_t id = argmax_row(lg, V, 0);
            for (uint32_t k = 0; k < n; ++k) {
                t.tokens.push_back(id); live[l].push_back(id);
                if (auto e = fwd.forward(&id, 1, fwd.n_pos(), lg); !e.empty()) { std::printf("decode: %s\n", e.c_str()); return false; }
                t.logits.push_back(fnv(lg)); id = argmax_row(lg, V, 0);
            }
            t.n_pos = fwd.n_pos();
            return true;
        };
        // control: A on lane 0, 8 greedy steps, the continuation, 16 greedy steps -- no slot involved
        Trace c8, c16, t8, t16;
        uint32_t L = 0; std::string src;
        if (!request(0, A, L, src) || !greedy(0, 8, c8)) return 1;
        const std::vector<int32_t> cont = [&] { std::vector<int32_t> v = live[0]; v.insert(v.end(), tail.begin(), tail.end()); return v; }();
        if (!request(0, cont, L, src) || !greedy(0, 16, c16)) return 1;
        check(L == live[0].size() - tail.size() - 16 && src == "live", "slot arm control: the continuation is served from lane 0's live state", "served " + std::to_string(L) + " from " + src);
        // arm: A on lane la, an unrelated prompt on la (A kept in a host slot), the continuation on lane lb (A restored there)
        if (!select(0)) return 1;
        fwd.reset_state(); live[0].clear();
        if (!request(la, A, L, src) || !greedy(la, 8, t8)) return 1;
        check(first_diff(t8, c8).empty(), "slot arm: A on lane " + std::to_string(la) + " == A on lane 0 (8 steps)", first_diff(t8, c8));
        const std::vector<int32_t> kept = live[la];
        if (!request(la, prompts[pb], L, src)) return 1;
        const auto s1 = fwd.prefix_cache_stats();
        check(L < kept.size() && s1.host_slots == 1, "slot arm: another prompt on lane " + std::to_string(la) + " keeps A (" + std::to_string(kept.size()) + " positions) in a host slot",
              std::to_string(s1.host_bytes >> 20) + " MiB in " + std::to_string(int(s1.last_save_ms)) + " ms; it reused " + std::to_string(L) + " of A");
        if (!request(lb, cont, L, src) || !greedy(lb, 16, t16)) return 1;
        const auto s2 = fwd.prefix_cache_stats();
        check(L == kept.size() && src == "host slot", "slot arm: the continuation on lane " + std::to_string(lb) + " restores A from its slot",
              "served " + std::to_string(L) + " from " + src + " in " + std::to_string(int(s2.last_restore_ms)) + " ms");
        check(first_diff(t16, c16).empty(), "slot arm: the continuation's logits and 16 tokens identical to the control that never left lane 0", first_diff(t16, c16));
        fwd.set_prefix_cache(false);
    }

    std::printf("MemAvailable at the end: %.1f GiB (%.1f before the load, %.1f after it)\n", mem_available_gib(), mem0, mem1);
    fwd.free_resident();
    for (auto& qp : queues) qp->wait_and_throw();
    std::printf("\nDS41 LANES TEST: %s\n", g_fail ? "FAILURE(S)" : "PASS");
    return g_fail ? 1 : 0;
}
