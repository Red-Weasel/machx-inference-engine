// tools/glm5next_lanes_test.cpp -- ie-glm-lanes-test: the gate of GLM-5.3-Flash request lanes and their two-card decode
// pipe (P4 B7, docs/glm53/P4_B7_LANES.md; design ~/ds41_work/p60/p4_b6_b8_design.md section 5.2).
//
// Loads the model as ie-glm5next-run does (two cards, blocks [0, 23) + [23, 45), the expert cache at IE_G5_ECACHE_MB or
// 22.5 GiB per card, banks pinned when IE_G5_PIN_BANKS=1 as served) with L request lanes per stage
// (Glm5NextModel::set_seq_lanes: lanes 1..L-1 at --lane-ctx, their bytes taken out of the expert cache). Every prompt runs
// SOLO on lane 0 first -- the pre-lane path -- then all of them over the L lanes:
//  * default (or --serial): INTERLEAVED, one lane's step at a time (a prefill chunk or a decode row), round robin;
//  * --pipeline: CARD-PIPELINED (Glm5LanePipe): one host thread per card, lane B's step on card 0 while lane A's runs on
//    card 1, every lane driving itself from the last stage's callback. --pipeline --serial runs both, serial first.
// A lane that finishes takes the next queued prompt mid-run (a new sequence on a dirty lane: pos 0 resets it). Each lane's
// sequence must be BYTE-IDENTICAL to its solo run: the fp16 logits after the prompt and after every decode step (FNV over
// the bytes) and the greedy tokens. The CPU expert leg is off (IE_G5_CPU_MISS=0: its CPU/GPU split follows the expert
// cache's history, so its rows are not reproducible run to run); --cpu-miss keeps it on (the served config) and then the
// identity lines are informational. --aba runs the solo arm again after the lanes (A-B-A; a short warm-up precedes the first
// arm) and reports the steady-window aggregate tokens/s against solo: the window starts when every first-wave lane has done
// 4 decode steps and ends when the first of them finishes; the inter-token gap's p50 / p95 per arm (P4 B14). Also printed: the lane VRAM against its arithmetic
// (glm5_lane_bytes), the expert cache, each card's engine busy from /proc/self/fdinfo over the decode windows, the expert
// cache hits and CPU experts per phase, and MemAvailable's minimum per phase against the 40 GiB floor. The per-lane host
// buffers are allocated before the model loads, so the pinning floor's MemAvailable check counts them. --dry prints the
// prompts' token counts and one extra lane's planned VRAM per stage, then exits without touching a device.
//   usage: ie-glm-lanes-test <model-00001-of-N.gguf> --prompt-file F [--prompt-file F ...] [--lanes L] [--n N] [--ctx C]
//          [--lane-ctx C2] [--chunk C] [--max-prompt P] [--gpus 1|2] [--cpu-miss] [--pipeline] [--serial] [--aba] [--dry]
#include "ie/allocator.hpp"
#include "ie/gguf.hpp"
#include "ie/glm5_lanes.hpp"
#include "ie/glm5_server.hpp"
#include "ie/glm5next.hpp"
#include "ie/model_config.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <dirent.h>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace ie;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& what, const std::string& detail = "") {
    std::printf("[%s] %s%s%s\n", ok ? " ok " : "FAIL", what.c_str(), detail.empty() ? "" : "  ", detail.c_str());
    std::fflush(stdout);
    if (!ok) ++g_fail;
}
double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}
uint64_t fnv16(const std::vector<sycl::half>& v) {
    uint64_t h = 1469598103934665603ull;
    for (const sycl::half& x : v) { uint16_t u; std::memcpy(&u, &x, 2); h = (h ^ u) * 1099511628211ull; }
    return h;
}
int32_t argmax16(const std::vector<sycl::half>& v) {   // ie-glm5next-run's argmax: the first maximum
    float mx = -1e30f; int32_t am = 0;
    for (uint32_t i = 0; i < v.size(); ++i) if (float(v[i]) > mx) { mx = float(v[i]); am = int32_t(i); }
    return am;
}

Glm5NextModel* g_term_m[2] = {};
uint32_t g_term_n = 0;
void on_term(int sig) {   // ie-glm5next-run's handler: drain the cards before the process goes
    for (uint32_t i = 0; i < g_term_n; ++i) if (g_term_m[i]) g_term_m[i]->shutdown();
    _exit(128 + sig);
}

// MemAvailable sampled every 200 ms; the minimum per phase.
struct MemWatch {
    std::atomic<bool> quit{false};
    std::thread th;
    std::mutex mu;
    std::string cur = "start";
    std::vector<std::pair<std::string, double>> mins;   // phase -> min GiB, in phase order
    static double read_gib() {
        std::ifstream f("/proc/meminfo");
        std::string k; uint64_t v = 0; std::string u;
        while (f >> k >> v >> u) if (k == "MemAvailable:") return double(v) / 1048576.0;
        return 0;
    }
    void sample() {
        const double g = read_gib();
        std::lock_guard<std::mutex> lk(mu);
        if (mins.empty() || mins.back().first != cur) mins.emplace_back(cur, g);
        else mins.back().second = std::min(mins.back().second, g);
    }
    void start() { th = std::thread([this] { while (!quit) { sample(); std::this_thread::sleep_for(std::chrono::milliseconds(200)); } }); }
    void phase(const std::string& p) { { std::lock_guard<std::mutex> lk(mu); cur = p; } sample(); }
    void stop() { quit = true; if (th.joinable()) th.join(); }
    double min_from(size_t i0) {
        std::lock_guard<std::mutex> lk(mu);
        double m = 1e30;
        for (size_t i = i0; i < mins.size(); ++i) m = std::min(m, mins[i].second);
        return m;
    }
    void print() {
        std::lock_guard<std::mutex> lk(mu);
        std::printf("MemAvailable minimum per phase (GiB):");
        for (const auto& [p, g] : mins) std::printf(" %s %.2f;", p.c_str(), g);
        std::printf("\n");
    }
    ~MemWatch() { stop(); }
};

// Each card's engine busy for this process's DRM clients, from /proc/self/fdinfo (xe: drm-cycles-<class> and
// drm-total-cycles-<class> per client), accumulated over begin/end windows (ie-mimo26-lanes-test's FdBusy).
struct FdBusy {
    struct Cnt { uint64_t cyc = 0, tot = 0; uint32_t cap = 1; };
    using Snap = std::map<std::string, Cnt>;
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
        std::map<std::string, std::map<std::string, double>> by;
        for (const auto& [k, v] : acc) {
            if (!v.second) continue;
            const size_t a = k.find('|'), b = k.rfind('|');
            by[k.substr(0, a)][k.substr(b + 1)] += 100.0 * double(v.first) / double(v.second) / caps.at(k);
        }
        for (const auto& [pdev, m] : by) {
            std::printf("fdinfo busy [%s] card %s:", label, pdev.c_str());
            for (const auto& [c, p] : m) if (p > 0) std::printf(" %s %.1f%%", c.c_str(), p);
            std::printf("\n");
        }
        if (by.empty()) std::printf("fdinfo busy [%s]: no DRM client cycles in /proc/self/fdinfo\n", label);
    }
};

// One sequence's record.
struct Trace {
    std::vector<uint64_t> logits;   // FNV of the fp16 logits after the prompt and after every decode step
    std::vector<int32_t> tokens;
    double prefill_ms = 0, decode_ms = 0;
    std::vector<double> step_ms;    // each decode step (timing only; not compared)
    uint32_t n_pos = 0;
};
std::string first_diff(const Trace& a, const Trace& b) {
    for (size_t i = 0; i < std::min(a.logits.size(), b.logits.size()); ++i)
        if (a.logits[i] != b.logits[i]) return "logits differ first at row " + std::to_string(i);
    if (a.logits.size() != b.logits.size()) return "row counts differ";
    if (a.tokens != b.tokens) return "tokens differ";
    if (a.n_pos != b.n_pos) return "positions differ";
    return {};
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: ie-glm-lanes-test <model-00001-of-N.gguf> --prompt-file F [--prompt-file F ...] [--lanes L] [--n N] "
                             "[--ctx C] [--lane-ctx C2] [--chunk C] [--max-prompt P] [--gpus 1|2] [--cpu-miss] [--pipeline] [--serial] "
                             "[--aba] [--dry]\n");
        return 2;
    }
    std::vector<std::string> files;
    uint32_t lanes = 2, N = 48, ctx = 32768, lane_ctx = 32768, C = 1024, max_prompt = 0, n_st = 2;
    bool cpu_miss = false, pipeline = false, serial = false, aba = false, dry = false;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--prompt-file") files.push_back(val());
        else if (a == "--lanes") lanes = uint32_t(std::atol(val().c_str()));
        else if (a == "--n") N = uint32_t(std::atol(val().c_str()));
        else if (a == "--ctx") ctx = uint32_t(std::atol(val().c_str()));
        else if (a == "--lane-ctx") lane_ctx = uint32_t(std::atol(val().c_str()));
        else if (a == "--chunk") C = uint32_t(std::atol(val().c_str()));
        else if (a == "--max-prompt") max_prompt = uint32_t(std::atol(val().c_str()));
        else if (a == "--gpus") n_st = uint32_t(std::atol(val().c_str()));
        else if (a == "--cpu-miss") cpu_miss = true;
        else if (a == "--pipeline") pipeline = true;
        else if (a == "--serial") serial = true;
        else if (a == "--aba") aba = true;
        else if (a == "--dry") dry = true;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (!pipeline) serial = true;
    lane_ctx = std::min(lane_ctx ? lane_ctx : ctx, ctx);
    if (files.empty() || lanes == 0 || N < 2 || C == 0 || (n_st != 1 && n_st != 2)) {
        std::fprintf(stderr, "need --prompt-file, --lanes >= 1, --n >= 2, --chunk >= 1, --gpus 1|2\n"); return 2;
    }
    if (!cpu_miss) setenv("IE_G5_CPU_MISS", "0", 1);   // before the first forward reads it
    if (n_st == 2 && !std::getenv("IE_G5_ECACHE_MB")) setenv("IE_G5_ECACHE_MB", "23040", 0);   // ie-glm5next-run's per-card default

    GgufReader g;
    if (auto e = g.open(argv[1]); !e.empty()) { std::fprintf(stderr, "open: %s\n", e.c_str()); return 1; }
    Glm5NextConfig cfg;
    if (auto e = read_glm5next_config(g, cfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }
    Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tokenizer: %s\n", e.c_str()); return 1; }
    std::vector<std::vector<int32_t>> prompts;
    const uint32_t cap_small = lanes > 1 ? lane_ctx : ctx;
    for (const auto& f : files) {
        std::ifstream in(f, std::ios::binary);
        if (!in) { std::fprintf(stderr, "cannot read %s\n", f.c_str()); return 2; }
        prompts.push_back(tok.encode(std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()), true));
        if (max_prompt && prompts.back().size() > max_prompt) prompts.back().resize(max_prompt);
        if (prompts.back().empty() || prompts.back().size() + N > cap_small) {
            std::fprintf(stderr, "%s: %zu tokens + %u do not fit ctx %u\n", f.c_str(), prompts.back().size(), N, cap_small); return 2;
        }
    }
    const uint32_t H = cfg.hidden, hc = cfg.hc_count, V = cfg.vocab, n_tf = cfg.n_transformer_layers();
    if (dry) {   // sizing only: no device is touched
        for (size_t i = 0; i < prompts.size(); ++i) std::printf("prompt %zu (%s): %zu tokens\n", i, files[i].c_str(), prompts[i].size());
        const bool sp = ctx > cfg.indexer_top_k + cfg.indexer_kpool - 1;
        for (uint32_t s = 0; s < n_st; ++s) {
            const uint32_t lo = (n_st == 2 && s == 1) ? 23 : 0, hi = (n_st == 2 && s == 0) ? 23 : n_tf;
            std::printf("stage %u [%u, %u): one extra lane at ctx %u = %.1f MiB\n", s, lo, hi, lane_ctx,
                        glm5_lane_bytes(cfg, lane_ctx, lo, hi, sp) / 1048576.0);
        }
        return 0;
    }

    // ---- host buffers BEFORE the model loads (the pinning floor's MemAvailable check counts them) ---------------------------
    std::unique_ptr<Glm5LanePipe> pipe;
    if (pipeline) pipe = std::make_unique<Glm5LanePipe>(lanes, C, uint64_t(hc) * H);
    std::vector<std::vector<sycl::half>> logits(lanes, std::vector<sycl::half>(V));   // every lane's tail logits
    std::vector<float> wide_h(size_t(C) * hc * H);                                    // the serial path's residual hand-off
    const uint64_t host_b = (pipe ? pipe->host_bytes() : 0) + uint64_t(lanes) * V * 2 + wide_h.size() * 4;
    MemWatch mw;
    mw.start();
    mw.phase("load");
    std::printf("glm-lanes: %zu prompt(s), %u lane(s), N %u, ctx %u (lanes 1..%u: %u), chunk %u, %u card(s), CPU miss leg %s, %s%s\n",
                prompts.size(), lanes, N, ctx, lanes - 1, lane_ctx, C, n_st, cpu_miss ? "ON (served)" : "off (identity)",
                pipeline ? (serial ? "serial interleave + pipeline" : "pipeline") : "serial interleave", aba ? ", A-B-A" : "");
    std::printf("host buffers allocated before the load: %.1f MiB (pipe per-lane ids + residual %.1f MiB, logits %.2f MiB, serial "
                "residual %.1f MiB)\n", host_b / 1048576.0, (pipe ? pipe->host_bytes() : 0) / 1048576.0,
                uint64_t(lanes) * V * 2 / 1048576.0, wide_h.size() * 4 / 1048576.0);

    // ---- load: ie-glm5next-run's two-card recipe ---------------------------------------------------------------------------
    uint32_t split = 23;
    if (const char* v = std::getenv("IE_G5_SPLIT")) { const uint32_t s = uint32_t(std::atoi(v)); if (s >= 1 && s < n_tf) split = s; }
    DeviceAllocator alloc[2];
    Glm5NextModel m_st[2];
    g_term_m[0] = &m_st[0]; g_term_m[1] = &m_st[1]; g_term_n = n_st;
    std::signal(SIGTERM, on_term);
    std::signal(SIGINT, on_term);
    // teardown: stop the pipe, then drain EVERY card before any of them frees (MiMo B2 gate note 6)
    struct Teardown {
        Glm5LanePipe* p; Glm5NextModel* m; uint32_t n;
        ~Teardown() { if (p) p->stop(); for (uint32_t i = 0; i < n; ++i) m[i].shutdown(); }
    } teardown{pipe.get(), m_st, n_st};
    const auto t_load = std::chrono::steady_clock::now();
    uint32_t lo_of[2] = {0, 0}, hi_of[2] = {n_tf, n_tf};
    for (uint32_t s = 0; s < n_st; ++s) {
        if (auto e = alloc[s].init("B70", s); !e.empty()) { std::fprintf(stderr, "gpu%u: %s\n", s, e.c_str()); return 1; }
        lo_of[s] = (n_st == 2 && s == 1) ? split : 0;
        hi_of[s] = (n_st == 2 && s == 0) ? split : n_tf;
        std::printf("[glm-lanes] loading blocks [%u, %u) on GPU %u ...\n", lo_of[s], hi_of[s], s);
        std::fflush(stdout);
        m_st[s].set_seq_lanes(lanes, lane_ctx);
        if (auto e = m_st[s].load(alloc[s], g, cfg, 0, lo_of[s], hi_of[s]); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
        if (auto e = m_st[s].init_runtime(ctx, std::max(C, 8u)); !e.empty()) { std::fprintf(stderr, "runtime: %s\n", e.c_str()); return 1; }
    }
    {
        std::future<std::string> fw[2];
        for (uint32_t s = 0; s < n_st; ++s) fw[s] = std::async(std::launch::async, [&m_st, s] { return m_st[s].warm_banks(); });
        for (uint32_t s = 0; s < n_st; ++s) if (auto e = fw[s].get(); !e.empty()) { std::fprintf(stderr, "warm: %s\n", e.c_str()); return 1; }
    }
    std::printf("[glm-lanes] loaded in %.1f s\n", ms_since(t_load) / 1000);
    const bool sparse = ctx > cfg.indexer_top_k + cfg.indexer_kpool - 1 || std::getenv("IE_G5_FORCE_SPARSE") != nullptr;
    for (uint32_t s = 0; s < n_st; ++s) {
        std::printf("stage %u [%u, %u): device %.3f GiB, expert cache %.3f GiB, pinned banks %.2f GiB; one extra lane %.1f MiB "
                    "(%.1f MiB for %u extra lane(s), out of the expert cache)\n",
                    s, lo_of[s], hi_of[s], m_st[s].device_bytes() / 1073741824.0, m_st[s].expert_cache_bytes() / 1073741824.0,
                    m_st[s].pinned_bank_bytes() / 1073741824.0, m_st[s].lane_bytes() / 1048576.0,
                    m_st[s].lane_bytes() * (lanes - 1) / 1048576.0, lanes - 1);
        if (lanes > 1) {
            const uint64_t want = glm5_lane_bytes(cfg, lane_ctx, lo_of[s], hi_of[s], sparse);
            check(m_st[s].lane_bytes() == want, "stage " + std::to_string(s) + ": one extra lane's device bytes equal glm5_lane_bytes",
                  std::to_string(m_st[s].lane_bytes()) + " B; at ctx 32,768 the arithmetic gives " +
                  std::to_string(double(glm5_lane_bytes(cfg, 32768, lo_of[s], hi_of[s], true)) / 1073741824.0) + " GiB");
        }
        if (pipeline)
            if (const std::string r = m_st[s].pipe_refusal(); !r.empty()) { std::fprintf(stderr, "pipe: stage %u: %s\n", s, r.c_str()); return 1; }
        if (pipeline) m_st[s].set_cpu_partition(true);   // the stages run concurrently (the q* workers' log states it)
    }
    const size_t mw_loaded = [&] { mw.phase("ready"); std::lock_guard<std::mutex> lk(mw.mu); return mw.mins.size() - 1; }();

    // ---- the stage body both paths run (identical launches), and the serial step --------------------------------------------
    auto stage = [&](uint32_t s, uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0, float* wide) -> std::string {
        const bool last = s + 1 == n_st;
        return m_st[s].lane_stage(lane, ids, T, pos0, s ? wide : nullptr, last ? nullptr : wide, last ? logits[lane].data() : nullptr);
    };
    auto serial_step = [&](uint32_t lane, const int32_t* ids, uint32_t T, uint32_t pos0) -> std::string {
        for (uint32_t s = 0; s < n_st; ++s)
            if (auto e = stage(s, lane, ids, T, pos0, wide_h.data()); !e.empty()) return "stage " + std::to_string(s) + ": " + e;
        return {};
    };
    auto check_id = [&](bool ok, const std::string& what, const std::string& detail) {   // with the CPU leg on: informational
        if (!cpu_miss) { check(ok, what, detail); return; }
        std::printf("[info] %s: %s%s%s\n", what.c_str(), ok ? "identical" : "DIFFERS", detail.empty() ? "" : "  ", detail.c_str());
        std::fflush(stdout);
    };
    // expert-cache and CPU-leg counters per phase (deltas)
    struct Ctr { uint64_t hits = 0, misses = 0, cpu = 0; };
    auto ctr_now = [&] { Ctr c; for (uint32_t s = 0; s < n_st; ++s) { c.hits += m_st[s].ecache_hits; c.misses += m_st[s].ecache_misses; c.cpu += m_st[s].n_cpu_experts; } return c; };
    auto ctr_print = [&](const char* label, const Ctr& a, uint32_t dec_steps) {
        const Ctr b = ctr_now();
        const uint64_t h = b.hits - a.hits, m = b.misses - a.misses, c = b.cpu - a.cpu;
        std::printf("experts [%s]: cache %llu hits / %llu misses (%.1f%% hit), %llu CPU experts (%.1f per decode step)\n", label,
                    (unsigned long long)h, (unsigned long long)m, 100.0 * double(h) / double(std::max<uint64_t>(1, h + m)),
                    (unsigned long long)c, dec_steps ? double(c) / dec_steps : 0.0);
    };
    constexpr uint32_t kWarmSteps = 4;   // decode steps of each sequence left out of the timing
    auto steady_rate = [&](const std::vector<Trace>& v) {
        double ms = 0; uint32_t n = 0;
        for (const auto& t : v) for (size_t k = kWarmSteps; k < t.step_ms.size(); ++k) { ms += t.step_ms[k]; ++n; }
        return ms > 0 ? n / (ms / 1000) : 0.0;
    };
    // P4 B14: the p50 / p95 of a set of inter-token gaps (ms), "p50 X / p95 Y ms"
    auto gap_pct = [](std::vector<double> g) {
        if (g.empty()) return std::string("no gaps");
        std::sort(g.begin(), g.end());
        auto at = [&](double p) { return g[std::min(g.size() - 1, size_t(p / 100.0 * double(g.size() - 1) + 0.5))]; };
        char b[96]; std::snprintf(b, sizeof b, "p50 %.1f / p95 %.1f ms over %zu gaps", at(50), at(95), g.size());
        return std::string(b);
    };
    auto solo_gaps = [&](const std::vector<Trace>& v) {   // a solo sequence's gap between tokens is its decode step
        std::vector<double> g;
        for (const auto& t : v) for (size_t k = kWarmSteps; k < t.step_ms.size(); ++k) g.push_back(t.step_ms[k]);
        return g;
    };
    auto preview = [&](const Trace& t) {
        const size_t n = std::min<size_t>(t.tokens.size(), 24);
        std::string s = tok.decode(std::span<const int32_t>(t.tokens.data(), n));
        for (char& c : s) if (c == '\n' || c == '\r') c = ' ';
        return s;
    };

    // ---- solo: every prompt alone on lane 0 (the pre-lane path) ------------------------------------------------------------
    FdBusy fd_solo, fd_lanes;
    auto run_solo = [&](std::vector<Trace>& out, const char* label) -> double {   // wall ms, < 0 on a failure
        out.assign(prompts.size(), Trace{});
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < prompts.size(); ++i) {
            const auto& ids = prompts[i];
            Trace& t = out[i];
            for (uint32_t at = 0; at < ids.size(); ) {
                const uint32_t n = std::min<uint32_t>(C, uint32_t(ids.size()) - at);
                const auto tp = std::chrono::steady_clock::now();
                if (auto e = serial_step(0, ids.data() + at, n, at); !e.empty()) { std::printf("solo prefill at %u: %s\n", at, e.c_str()); return -1; }
                t.prefill_ms += ms_since(tp);
                at += n;
            }
            t.logits.push_back(fnv16(logits[0]));
            int32_t id = argmax16(logits[0]);
            fd_solo.begin();
            for (uint32_t k = 0; k < N; ++k) {
                t.tokens.push_back(id);
                if (k + 1 == N) break;
                const auto tp = std::chrono::steady_clock::now();
                if (auto e = serial_step(0, &id, 1, uint32_t(ids.size()) + k); !e.empty()) { std::printf("solo decode: %s\n", e.c_str()); return -1; }
                const double ms = ms_since(tp);
                t.decode_ms += ms; t.step_ms.push_back(ms);
                t.logits.push_back(fnv16(logits[0]));
                id = argmax16(logits[0]);
            }
            fd_solo.end();
            t.n_pos = m_st[n_st - 1].lane_pos(0);
            std::printf("solo%s %zu (%s): %zu prompt tokens, prefill %.0f ms (%.1f tok/s), decode %.1f ms/token: \"%s\"\n", label, i,
                        files[i].c_str(), ids.size(), t.prefill_ms, ids.size() / (t.prefill_ms / 1000), t.decode_ms / (N - 1), preview(t).c_str());
            std::fflush(stdout);
        }
        return ms_since(t0);
    };
    if (aba) {   // warm-up before the first arm (JIT, the cache, the q* workers): discarded
        const std::vector<int32_t> w(prompts[0].begin(), prompts[0].begin() + std::ptrdiff_t(std::min<size_t>({512, prompts[0].size(), C})));
        if (auto e = serial_step(0, w.data(), uint32_t(w.size()), 0); !e.empty()) { std::printf("warm-up: %s\n", e.c_str()); return 1; }
        int32_t id = argmax16(logits[0]);
        for (uint32_t k = 0; k < 8; ++k) {
            if (auto e = serial_step(0, &id, 1, uint32_t(w.size()) + k); !e.empty()) { std::printf("warm-up: %s\n", e.c_str()); return 1; }
            id = argmax16(logits[0]);
        }
        std::printf("warm-up: %zu-token prefill + 8 decode steps, discarded\n", w.size());
    }
    mw.phase(aba ? "solo-A1" : "solo");
    Ctr c0 = ctr_now();
    std::vector<Trace> solo;
    const double solo_ms = run_solo(solo, aba ? " A1" : "");
    if (solo_ms < 0) return 1;
    uint32_t n_tok = 0; double solo_dec = 0;
    for (const auto& t : solo) { n_tok += uint32_t(t.tokens.size()); solo_dec += t.decode_ms; }
    ctr_print("solo", c0, n_tok - uint32_t(prompts.size()));
    const double solo_rate = steady_rate(solo);
    std::printf("solo%s inter-token gap: %s\n", aba ? " A1" : "", gap_pct(solo_gaps(solo)).c_str());

    // ---- the lanes' host bookkeeping: admission to idle lanes (index order), a finished lane takes the next queued prompt -----
    struct LaneRun { int seq = -1; bool busy = false, prefill = true; uint32_t at = 0, n_new = 0; int32_t next = -1; };
    std::vector<LaneRun> lr;
    std::deque<size_t> queue;
    uint32_t readmits = 0;
    auto reset_runs = [&] {
        lr.assign(lanes, LaneRun{}); queue.clear(); readmits = 0;
        for (size_t i = 0; i < prompts.size(); ++i) queue.push_back(i);
    };
    auto admit = [&]() {
        std::vector<uint32_t> got;
        for (uint32_t l = 0; l < lanes && !queue.empty(); ++l) {
            if (lr[l].busy) continue;
            if (lr[l].seq >= 0) ++readmits;
            lr[l] = LaneRun{int(queue.front()), true, true, 0, 0, -1};
            queue.pop_front();
            got.push_back(l);
        }
        return got;
    };
    auto compare = [&](const std::vector<Trace>& got, const char* mode) {
        for (size_t i = 0; i < prompts.size(); ++i) {
            const std::string d = first_diff(got[i], solo[i]);
            check_id(d.empty(), "sequence " + std::to_string(i) + " (" + files[i] + ") " + mode + " == solo: " +
                                std::to_string(got[i].logits.size()) + " logits rows, " + std::to_string(got[i].tokens.size()) + " tokens", d);
        }
        if (lanes > 1) check(readmits > 0 || prompts.size() <= lanes, std::string(mode) + ": a lane finished mid-run and took the next queued prompt",
                             std::to_string(readmits) + " re-admission(s)");
    };

    // ---- serial interleave (B7a): one lane's step at a time, round robin ----------------------------------------------------
    if (serial) {
        mw.phase("interleaved");
        c0 = ctr_now();
        reset_runs();
        std::vector<Trace> inter(prompts.size());
        admit();
        uint32_t steps = 0, switches = 0, dec_steps = 0; int last = -1;
        std::string order;
        const auto t_inter = std::chrono::steady_clock::now();
        for (;;) {
            int l = -1;
            for (uint32_t k = 1; k <= lanes; ++k) {
                const uint32_t c = last < 0 ? k - 1 : (uint32_t(last) + k) % lanes;
                if (lr[c].busy) { l = int(c); break; }
            }
            if (l < 0) break;
            if (l != last) ++switches;
            last = l;
            if (order.size() < 120) order += std::to_string(l);
            LaneRun& r = lr[size_t(l)];
            Trace& t = inter[size_t(r.seq)];
            const auto& ids = prompts[size_t(r.seq)];
            ++steps;
            const auto tp = std::chrono::steady_clock::now();
            if (r.prefill) {
                const uint32_t n = std::min<uint32_t>(C, uint32_t(ids.size()) - r.at);
                if (auto e = serial_step(uint32_t(l), ids.data() + r.at, n, r.at); !e.empty()) { std::printf("interleaved prefill: %s\n", e.c_str()); return 1; }
                r.at += n; t.prefill_ms += ms_since(tp);
                if (r.at < ids.size()) continue;
                r.prefill = false;
            } else {
                if (auto e = serial_step(uint32_t(l), &r.next, 1, uint32_t(ids.size()) + r.n_new - 1); !e.empty()) { std::printf("interleaved decode: %s\n", e.c_str()); return 1; }
                const double ms = ms_since(tp);
                t.decode_ms += ms; t.step_ms.push_back(ms); ++dec_steps;
            }
            t.logits.push_back(fnv16(logits[size_t(l)]));
            r.next = argmax16(logits[size_t(l)]);
            t.tokens.push_back(r.next);
            if (++r.n_new < N) continue;
            t.n_pos = m_st[n_st - 1].lane_pos(uint32_t(l));
            r.busy = false;
            admit();                                   // mid-run: the finished lane takes the next queued prompt
        }
        const double inter_ms = ms_since(t_inter);
        std::printf("interleaved: %u steps over %u lane(s), %u lane switches, %u mid-run re-admissions, %.1f s; lane order %s%s\n",
                    steps, lanes, switches, readmits, inter_ms / 1000, order.c_str(), order.size() >= 120 ? " ..." : "");
        ctr_print("interleaved", c0, dec_steps);
        compare(inter, "interleaved");
        std::printf("timing: solo %.1f s for all prompts (decode %.1f ms/step), interleaved %.1f s: interleaved/solo wall %.3f (serial "
                    "interleave: no gain expected)\n", solo_ms / 1000, solo_dec / std::max<uint32_t>(1, n_tok - uint32_t(prompts.size())),
                    inter_ms / 1000, inter_ms / solo_ms);
    }

    // ---- card-pipelined (B7b): every lane drives itself from the last stage's callback --------------------------------------
    if (pipeline) {
        mw.phase("pipelined");
        c0 = ctr_now();
        reset_runs();
        std::vector<Trace> inter(prompts.size());
        std::mutex cb_mu;                               // the callback's state (the last stage calls it; main holds it while seeding)
        std::string cb_err;
        std::vector<std::chrono::steady_clock::time_point> t_sub(lanes);
        std::vector<uint32_t> wave, dec_n(lanes, 0);
        std::vector<double> done_ms;                    // every decode completion (ms since t_pipe) while the window may be open
        std::vector<std::vector<double>> lane_done(lanes);   // P4 B14: the same, per lane (the window's inter-token gaps)
        double win0 = -1, win1 = -1;
        uint32_t steps = 0, dec_steps = 0;
        const auto t_pipe = std::chrono::steady_clock::now();
        auto fail = [&](const std::string& e) { if (cb_err.empty()) cb_err = e; };
        auto submit = [&](uint32_t l) -> bool {        // the lane's next step: a prefill chunk or a decode row (cb_mu held)
            LaneRun& r = lr[l];
            const auto& ids = prompts[size_t(r.seq)];
            t_sub[l] = std::chrono::steady_clock::now();
            const std::string e = r.prefill
                ? pipe->submit(l, ids.data() + r.at, std::min<uint32_t>(C, uint32_t(ids.size()) - r.at), r.at)
                : pipe->submit(l, &r.next, 1, uint32_t(ids.size()) + r.n_new - 1);
            if (!e.empty()) { fail(e); return false; }
            return true;
        };
        auto on_done = [&](uint32_t l, uint32_t T, uint32_t) {
            std::lock_guard<std::mutex> lk(cb_mu);
            if (!cb_err.empty()) return;
            LaneRun& r = lr[l];
            Trace& t = inter[size_t(r.seq)];
            const auto& ids = prompts[size_t(r.seq)];
            const double step = ms_since(t_sub[l]);
            ++steps;
            if (r.prefill) {
                r.at += T; t.prefill_ms += step;
                if (r.at < ids.size()) { submit(l); return; }
                r.prefill = false;
            } else {
                t.decode_ms += step; t.step_ms.push_back(step); ++dec_steps;
                const double at = ms_since(t_pipe);
                ++dec_n[l];
                if (win1 < 0) {
                    done_ms.push_back(at);
                    lane_done[l].push_back(at);
                    if (win0 < 0 && std::all_of(wave.begin(), wave.end(), [&](uint32_t w) { return dec_n[w] >= kWarmSteps; })) { win0 = at; fd_lanes.begin(); }
                }
            }
            t.logits.push_back(fnv16(logits[l]));
            r.next = argmax16(logits[l]);
            t.tokens.push_back(r.next);
            if (++r.n_new < N) { submit(l); return; }
            t.n_pos = m_st[n_st - 1].lane_pos(l);        // this callback runs on the last stage's thread, which owns it
            if (win1 < 0 && win0 >= 0 && std::find(wave.begin(), wave.end(), l) != wave.end()) { win1 = ms_since(t_pipe); fd_lanes.end(); }
            r.busy = false;
            for (uint32_t nl : admit()) submit(nl);    // mid-run: the finished lane takes the next queued prompt (pos 0: a reset)
        };
        auto stage_fn = [&](uint32_t s, const Glm5LanePipe::Step& st) { return stage(s, st.lane, st.ids, st.T, st.pos0, st.wide); };
        if (auto e = pipe->start(n_st, stage_fn, on_done); !e.empty()) { std::printf("pipe start: %s\n", e.c_str()); return 1; }
        {
            std::lock_guard<std::mutex> lk(cb_mu);
            wave = admit();
            for (uint32_t l : wave) if (!submit(l)) break;
        }
        const std::string pe = pipe->stop();
        const double pipe_ms = ms_since(t_pipe);
        if (!pe.empty() || !cb_err.empty()) { std::printf("pipeline: %s%s%s\n", pe.c_str(), pe.empty() || cb_err.empty() ? "" : "; ", cb_err.c_str()); return 1; }
        std::printf("pipelined: %u steps over %u lane(s) on %u stage thread(s), %u mid-run re-admissions, %.1f s\n", steps, lanes, n_st,
                    readmits, pipe_ms / 1000);
        ctr_print("pipelined", c0, dec_steps);
        compare(inter, "pipelined");
        uint32_t in_win = 0;
        for (double t : done_ms) if (t > win0 && t <= win1) ++in_win;
        const double agg = win1 > win0 && win0 >= 0 ? in_win / ((win1 - win0) / 1000) : 0.0;
        double lat = 0; uint32_t nlat = 0;
        for (const auto& t : inter) for (size_t k = kWarmSteps; k < t.step_ms.size(); ++k) { lat += t.step_ms[k]; ++nlat; }
        std::printf("timing: steady window %.1f s with all %zu first-wave lanes decoding: %u decode steps = %.2f tokens/s aggregate, "
                    "%.1f ms per lane step (submit to logits); solo%s %.2f tokens/s (%.1f ms/step) -> x%.3f\n",
                    (win1 - win0) / 1000, wave.size(), in_win, agg, nlat ? lat / nlat : 0.0, aba ? " A1" : "", solo_rate,
                    solo_rate > 0 ? 1000 / solo_rate : 0.0, solo_rate > 0 ? agg / solo_rate : 0.0);
        {   // P4 B14: each lane's gaps between consecutive tokens inside the steady window, pooled and per lane
            std::vector<double> all;
            std::string per;
            for (uint32_t l = 0; l < lanes; ++l) {
                std::vector<double> g;
                for (size_t k = 1; k < lane_done[l].size(); ++k)
                    if (lane_done[l][k - 1] > win0 && lane_done[l][k] <= win1) g.push_back(lane_done[l][k] - lane_done[l][k - 1]);
                all.insert(all.end(), g.begin(), g.end());
                per += "; lane " + std::to_string(l) + " " + gap_pct(g);
            }
            std::printf("pipelined inter-token gap in the window: %s%s\n", gap_pct(all).c_str(), per.c_str());
        }
        if (aba) {
            mw.phase("solo-A2");
            c0 = ctr_now();
            std::vector<Trace> solo2;
            if (run_solo(solo2, " A2") < 0) return 1;
            ctr_print("solo A2", c0, n_tok - uint32_t(prompts.size()));
            const double r2 = steady_rate(solo2);
            std::printf("solo A2 inter-token gap: %s\n", gap_pct(solo_gaps(solo2)).c_str());
            for (size_t i = 0; i < prompts.size(); ++i)
                check_id(first_diff(solo2[i], solo[i]).empty(), "solo A2 == solo A1, sequence " + std::to_string(i), first_diff(solo2[i], solo[i]));
            std::printf("timing A-B-A: solo A1 %.2f, pipelined %.2f aggregate, solo A2 %.2f tokens/s -> pipelined / mean(A1, A2) = x%.3f\n",
                        solo_rate, agg, r2, (solo_rate + r2) > 0 ? agg / ((solo_rate + r2) / 2) : 0.0);
        }
        fd_lanes.print("pipelined decode window");
    }
    fd_solo.print("solo decode");

    mw.phase("end");
    const double floor_min = mw.min_from(mw_loaded);
    mw.stop();
    mw.print();
    check(floor_min >= 40.0, "MemAvailable stayed at or above the 40 GiB floor after the load", std::to_string(floor_min) + " GiB minimum");
    std::printf("\nGLM LANES TEST: %s\n", g_fail ? "FAILURE(S)" : "PASS");
    return g_fail ? 1 : 0;
}
