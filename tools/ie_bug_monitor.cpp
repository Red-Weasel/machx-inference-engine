// tools/ie_bug_monitor.cpp — comprehensive 8-chain live dashboard.
//
// Single-screen, constantly-refreshing terminal UI that runs 8 fully
// independent forward(T=1) chains (A..H) — each with its own KvCache and
// DeltaNetState — and per-iter computes a per-layer hash of every
// exposed persistent buffer for each chain (DN state, DN conv state,
// KV-K, KV-V, logits).  Hashes are compared across chains.  When ANY
// chain's combined hash diverges from chain A, the dashboard:
//   * Flips the global status to "✗ DIVERGED".
//   * Marks the divergent chains red in the chain table.
//   * Walks the per-layer hash arrays in producer order
//       (DN_state[L=0..29] → DN_conv[L=0..29] → KV_K[L=0..9] →
//        KV_V[L=0..9] → logits)
//     and prints which buffer first differs for each divergent chain.
//   * Records the FIRST-EVER divergence (iter, chain, buffer tag) and
//     pins it in the dashboard for the rest of the run.
//
// THROUGHPUT panel reports both prompt-processing tok/s (single-chain,
// measured during a one-time T=PROMPT_T warmup) and decode tok/s
// (single-chain equivalent + 8-chain combined, computed live).
//
// LATENCY graph shows per-iter wall ms across 8 chains as a moving
// sparkline plus p50/p95/max statistics.
//
// TOP KERNELS shows kernels by this-iter time (summed across the 8
// forwards in the iter) with per-kernel mini-sparklines and all-time
// totals.
//
// MEMORY estimates GPU usage: model + 8 KV + 8 DN-state + 8 DN-conv +
// workspace.  Color-coded against the 32 GB B70 budget.
//
// FOCUS RECOMMENDATIONS auto-detect: bug fired? hot kernel >10%? p99
// jitter >2× p50? memcpy bloat? memory pressure?
//
// Footer: Ctrl-C exits cleanly (cursor restored, final summary printed).
//
// NOTE: Workspace intermediates (ws_x_normed_, ws_qkv_, ws_qkv_silu_,
// etc.) are SHARED across chains and are overwritten as chains run
// sequentially.  Capturing them per-chain would require getter methods
// on QwenModel — out of scope for this tool.  The persistent state
// hashed here (DN state, conv state, KV) IS sufficient to detect and
// localize divergence to a layer + buffer kind.

#include "ie/allocator.hpp"
#include "ie/deltanet_state.hpp"
#include "ie/gguf.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/kv_cache.hpp"
#include "ie/qwen36.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

// ===== ANSI ============================================================
constexpr const char* HIDE_CUR = "\033[?25l";
constexpr const char* SHOW_CUR = "\033[?25h";
constexpr const char* CLR_SCR  = "\033[2J\033[H";
constexpr const char* HOME     = "\033[H";
constexpr const char* CLR_EOL  = "\033[K";
constexpr const char* CLR_EOS  = "\033[J";

constexpr const char* R     = "\033[0m";
constexpr const char* DIM   = "\033[2m";
constexpr const char* BOLD  = "\033[1m";
constexpr const char* INV   = "\033[7m";
constexpr const char* GR    = "\033[32m";
constexpr const char* GR_B  = "\033[32;1m";
constexpr const char* RD_B  = "\033[31;1m";
constexpr const char* YL    = "\033[33m";
constexpr const char* YL_B  = "\033[33;1m";
constexpr const char* BL    = "\033[34;1m";

const char* SPK[9] = {" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
const char* BAR_FULL  = "█";
const char* BAR_LIGHT = "░";

// ===== Chain constants =================================================
constexpr int N_CHAINS = 8;
constexpr const char* CHAIN_LBL[N_CHAINS] = {
    "A", "B", "C", "D", "E", "F", "G", "H"
};

// ===== Helpers =========================================================
double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint64_t hash64(const void* data, size_t bytes,
                 uint64_t seed = 0xcbf29ce484222325ULL) {
    uint64_t h = seed;
    const uint8_t*  p = static_cast<const uint8_t*>(data);
    const uint64_t* w = reinterpret_cast<const uint64_t*>(p);
    const size_t    nw = bytes / 8;
    for (size_t i = 0; i < nw; ++i) { h ^= w[i]; h *= 0x100000001b3ULL; }
    for (size_t i = nw * 8; i < bytes; ++i) {
        h ^= uint64_t(p[i]); h *= 0x100000001b3ULL;
    }
    return h;
}

std::string short_name(const std::string& full) {
    if (full.find("MemoryCopy") != std::string::npos) {
        if (full.find("(M2D)") != std::string::npos) return "memcpy(M2D)";
        if (full.find("(D2M)") != std::string::npos) return "memcpy(D2M)";
        if (full.find("(D2D)") != std::string::npos) return "memcpy(D2D)";
        return "memcpy";
    }
    auto p = full.find("ie::");
    size_t start = (p == std::string::npos) ? 0 : p + 4;
    size_t end   = full.find('(', start);
    if (end == std::string::npos) end = full.size();
    std::string s = full.substr(start, end - start);
    if (s.size() > 28) s = s.substr(0, 28);
    return s;
}

std::string spark_line(const std::deque<double>& vals,
                        double vmin, double vmax) {
    std::string out;
    if (vmax <= vmin) vmax = vmin + 1e-12;
    for (double v : vals) {
        double f = (v - vmin) / (vmax - vmin);
        int idx = std::clamp(int(f * 8.0) + 1, 1, 8);
        if (v <= 0.0) idx = 0;
        out += SPK[idx];
    }
    return out;
}

std::string h_bar(double frac, int width, const char* color) {
    frac = std::clamp(frac, 0.0, 1.0);
    int filled = int(frac * width);
    std::string out;
    out += color;
    for (int i = 0; i < filled; ++i) out += BAR_FULL;
    out += DIM;
    for (int i = filled; i < width; ++i) out += BAR_LIGHT;
    out += R;
    return out;
}

double dq_pct(std::deque<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[size_t(p * (v.size() - 1))];
}
double dq_max(const std::deque<double>& v) {
    double m = 0.0; for (double x : v) m = std::max(m, x); return m;
}
double dq_min(const std::deque<double>& v) {
    if (v.empty()) return 0.0;
    double m = v.front(); for (double x : v) m = std::min(m, x); return m;
}

std::string fmt_bytes(uint64_t b) {
    char buf[32];
    if (b >= (uint64_t)1 << 30)
        std::snprintf(buf, sizeof(buf), "%.2f GB", double(b) / (1ull << 30));
    else if (b >= (uint64_t)1 << 20)
        std::snprintf(buf, sizeof(buf), "%.1f MB", double(b) / (1ull << 20));
    else if (b >= 1024)
        std::snprintf(buf, sizeof(buf), "%.0f KB", double(b) / 1024.0);
    else
        std::snprintf(buf, sizeof(buf), "%lu B", (unsigned long)b);
    return buf;
}

std::string fmt_us(double ns) {
    char buf[32];
    if (ns < 1e3)        std::snprintf(buf, sizeof(buf), "%.0f ns",  ns);
    else if (ns < 1e6)   std::snprintf(buf, sizeof(buf), "%.1f µs",  ns / 1e3);
    else if (ns < 1e9)   std::snprintf(buf, sizeof(buf), "%.2f ms",  ns / 1e6);
    else                 std::snprintf(buf, sizeof(buf), "%.2f s",   ns / 1e9);
    return buf;
}

std::string fmt_clock(double seconds) {
    int s = int(seconds);
    int h = s / 3600;
    int m = (s % 3600) / 60;
    int sec = s % 60;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d", h, m, sec);
    return buf;
}

constexpr const char* kSampleText =
    "I am by birth a Genevese; and my family is one of the most distinguished "
    "of that republic. My ancestors had been for many years counsellors and "
    "syndics; and my father had filled several public situations with honour "
    "and reputation. He was respected by all who knew him for his integrity "
    "and indefatigable attention to public business. He passed his younger "
    "days perpetually occupied by the affairs of his country; and it was not "
    "until the decline of life that he thought of marrying, and bestowing on "
    "the state sons who might carry his virtues and his name down to "
    "posterity. As the circumstances of his marriage illustrate his "
    "character, I cannot refrain from relating them.\n\n"
    "During the years that followed, the household preserved a quiet discipline "
    "which gave dignity to ordinary labor. The library was small, but every "
    "volume had been chosen with care, and the books were read until their "
    "margins carried traces of many hands. At evening the shutters were closed, "
    "the lamp was trimmed, and the younger children listened while letters from "
    "distant friends were read aloud. These letters spoke of voyages, harvests, "
    "public debates, and the patient work by which families keep faith with one "
    "another across time and weather.\n\n"
    "I learned early that knowledge is not gathered by haste alone. A page "
    "understood clearly was worth more than a chapter passed over in restless "
    "curiosity. My teachers encouraged questions, but they also required proof, "
    "comparison, and a willingness to revise an opinion when the evidence did "
    "not support it. In that habit I found a kind of freedom: the mind became "
    "less anxious when it could distinguish a bright guess from a settled fact.\n\n"
    "When I was older, I travelled beyond the familiar streets of my childhood "
    "and saw how much of human life depends on arrangements too common to be "
    "praised. Roads, bridges, ledgers, workshops, schools, and markets seemed "
    "plain enough at first glance, yet each required memory, trust, and daily "
    "attention. A careless hand could waste what many careful hands had built. "
    "This observation made me cautious in judgment and more grateful for the "
    "uncelebrated skill that supports a peaceful city.\n\n"
    "The strongest impression of those years was not a single event, but a "
    "gradual conviction that character is measured in repeated choices. A "
    "person may speak generously in public and still fail in private duties; "
    "another may say little and yet become indispensable by doing necessary "
    "work at the proper hour. I admired the latter kind of excellence. It did "
    "not glitter, but it endured, and it left the world more orderly than it "
    "found it.\n\n"
    "Thus my education joined affection with inquiry. I loved the people who "
    "had formed me, but I also learned to examine my own certainties. Whenever "
    "a new subject drew my attention, I tried to ask what could be tested, what "
    "must be inferred, and what ought to remain undecided. This discipline did "
    "not diminish wonder. On the contrary, it made wonder steadier, because it "
    "rested on patient attention rather than surprise alone.";

std::string read_text_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return {};
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

volatile std::sig_atomic_t g_quit = 0;
void on_sigint(int) { g_quit = 1; }

// ===== Per-chain hash structures =======================================
struct ChainHashes {
    uint64_t              combined = 0;
    std::vector<uint64_t> dn_state;     // n_layers_linear (30)
    std::vector<uint64_t> dn_conv;      // n_layers_linear (30)
    std::vector<uint64_t> kv_k;         // n_layers_full   (10)
    std::vector<uint64_t> kv_v;         // n_layers_full   (10)
    uint64_t              logits = 0;
};

// Diff vs reference chain (chain A).  first_diff_tag identifies the
// chronologically-first per-layer/per-buffer hash that differs.
struct ChainDiff {
    bool        diverged = false;
    std::string first_diff_tag;
};

ChainDiff diff_vs_ref(const ChainHashes& ref, const ChainHashes& x) {
    ChainDiff d{};
    if (ref.combined == x.combined) return d;
    d.diverged = true;
    char buf[64];
    auto check_layer = [&](const std::vector<uint64_t>& a,
                            const std::vector<uint64_t>& b,
                            const char* prefix) -> bool {
        for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
            if (a[i] != b[i]) {
                std::snprintf(buf, sizeof(buf), "%s[L=%zu]", prefix, i);
                d.first_diff_tag = buf;
                return true;
            }
        }
        return false;
    };
    if (check_layer(ref.dn_state, x.dn_state, "DN_state")) return d;
    if (check_layer(ref.dn_conv,  x.dn_conv,  "DN_conv"))  return d;
    if (check_layer(ref.kv_k,     x.kv_k,     "KV_K"))     return d;
    if (check_layer(ref.kv_v,     x.kv_v,     "KV_V"))     return d;
    if (ref.logits != x.logits) {
        d.first_diff_tag = "logits";
        return d;
    }
    d.first_diff_tag = "(combined diff but per-layer match)";
    return d;
}

// ===== Per-chain runtime ===============================================
struct Chain {
    ie::KvCache       kv;
    ie::DeltaNetState dn;
    sycl::half*       d_logits = nullptr;
    ChainHashes       hashes;
};

struct HashScratch {
    std::vector<uint8_t> dn_state;
    std::vector<uint8_t> dn_conv;
    std::vector<uint8_t> kv_layer;     // sized to max populated K or V layer
    std::vector<sycl::half> logits;    // host-side fp16
};

// Hash all persistent state buffers of one chain.  Per-layer DN state
// + DN conv + KV-K + KV-V + a small logits sample.
void hash_chain(sycl::queue& q, Chain& c,
                 const ie::DeltaNetStateConfig& dnc,
                 const ie::KvCacheConfig& kvc,
                 uint32_t vocab,
                 HashScratch& s) {
    // ----- DN state -----
    const size_t per_dn_layer =
        size_t(dnc.n_v_heads) * dnc.v_head_dim * dnc.k_head_dim *
        sizeof(float);
    const size_t dn_total = per_dn_layer * dnc.n_layers_linear;
    if (s.dn_state.size() < dn_total) s.dn_state.resize(dn_total);
    q.memcpy(s.dn_state.data(), c.dn.state_ptr(), dn_total).wait();
    c.hashes.dn_state.assign(dnc.n_layers_linear, 0);
    for (uint32_t L = 0; L < dnc.n_layers_linear; ++L) {
        c.hashes.dn_state[L] =
            hash64(s.dn_state.data() + L * per_dn_layer, per_dn_layer);
    }

    // ----- DN conv state -----
    const size_t per_conv_layer =
        size_t(dnc.conv_channels) * (dnc.conv_kernel - 1) *
        sizeof(sycl::half);
    const size_t conv_total = per_conv_layer * dnc.n_layers_linear;
    if (s.dn_conv.size() < conv_total) s.dn_conv.resize(conv_total);
    q.memcpy(s.dn_conv.data(), c.dn.conv_state_ptr(), conv_total).wait();
    c.hashes.dn_conv.assign(dnc.n_layers_linear, 0);
    for (uint32_t L = 0; L < dnc.n_layers_linear; ++L) {
        c.hashes.dn_conv[L] =
            hash64(s.dn_conv.data() + L * per_conv_layer, per_conv_layer);
    }

    // ----- KV cache (populated portion only) -----
    c.hashes.kv_k.assign(kvc.n_layers_full, 0);
    c.hashes.kv_v.assign(kvc.n_layers_full, 0);
    const uint64_t per_kvhead_elems =
        uint64_t(kvc.max_ctx) * kvc.head_dim;
    const uint64_t per_layer_elems =
        uint64_t(kvc.n_kv_heads) * per_kvhead_elems;
    for (uint32_t L = 0; L < kvc.n_layers_full; ++L) {
        const uint32_t len = c.kv.length(L);
        if (len == 0) continue;
        const size_t bytes = size_t(len) * kvc.head_dim *
                              sizeof(sycl::half);
        if (s.kv_layer.size() < bytes) s.kv_layer.resize(bytes);
        uint64_t kh = 0xcbf29ce484222325ULL;
        uint64_t vh = 0xcbf29ce484222325ULL;
        for (uint32_t kvh = 0; kvh < kvc.n_kv_heads; ++kvh) {
            const sycl::half* k_src = c.kv.k_ptr() +
                L * per_layer_elems + kvh * per_kvhead_elems;
            const sycl::half* v_src = c.kv.v_ptr() +
                L * per_layer_elems + kvh * per_kvhead_elems;
            q.memcpy(s.kv_layer.data(), k_src, bytes).wait();
            kh = hash64(s.kv_layer.data(), bytes, kh);
            q.memcpy(s.kv_layer.data(), v_src, bytes).wait();
            vh = hash64(s.kv_layer.data(), bytes, vh);
        }
        c.hashes.kv_k[L] = kh;
        c.hashes.kv_v[L] = vh;
    }

    // ----- Logits (small) -----
    if (s.logits.size() < vocab) s.logits.resize(vocab);
    q.memcpy(s.logits.data(), c.d_logits,
             vocab * sizeof(sycl::half)).wait();
    c.hashes.logits = hash64(s.logits.data(), vocab * sizeof(sycl::half));

    // ----- Combined -----
    uint64_t h = 0xcbf29ce484222325ULL;
    auto fold = [&](const std::vector<uint64_t>& v) {
        for (auto x : v) { h ^= x; h *= 0x100000001b3ULL; }
    };
    fold(c.hashes.dn_state);
    fold(c.hashes.dn_conv);
    fold(c.hashes.kv_k);
    fold(c.hashes.kv_v);
    h ^= c.hashes.logits; h *= 0x100000001b3ULL;
    c.hashes.combined = h;
}

// ===== Kernel rolling state ============================================
struct KernelRoll {
    double             ns_this_iter = 0.0;
    int                calls_this_iter = 0;
    double             all_time_ns = 0.0;
    uint64_t           all_time_calls = 0;
    std::deque<double> ns_window;
    double             max_seen_ns = 0.0;
};

// ===== Dashboard =======================================================
struct Dashboard {
    // Static
    std::string device_name;
    uint32_t    max_iters = 0;
    int         graph_n   = 60;
    uint64_t    model_bytes_est = 22ull << 30;
    uint64_t    kv_bytes_total = 0;
    uint64_t    dn_bytes_total = 0;
    uint64_t    workspace_bytes_est = 0;
    uint64_t    gpu_mem_total_b = uint64_t(32) << 30;

    // Throughput
    double      prompt_T = 0.0;
    double      prompt_ms = 0.0;
    double      prompt_tok_s = 0.0;

    // Live
    uint32_t    iter = 0;
    double      app_start_ms = 0.0;
    int         total_match_iters = 0;
    int         total_diff_iters  = 0;
    int         first_div_iter = -1;
    int         first_div_chain = -1;
    std::string first_div_tag;

    // Per-chain snapshot
    std::array<uint64_t,  N_CHAINS> chain_combined{};
    std::array<ChainDiff, N_CHAINS> chain_diff{};
    int         n_unique_outputs = 1;  // # of distinct combined hashes

    std::deque<double> wall_window;        // total iter ms (8 forwards)
    std::deque<int>    diverge_strip;      // 0/1 any-diverged per iter
    std::map<std::string, KernelRoll> kernels;
};

// Auto-detected hotspot bullets.
std::vector<std::pair<const char*, std::string>>
focus_notes(const Dashboard& d) {
    std::vector<std::pair<const char*, std::string>> notes;

    if (d.first_div_iter >= 0) {
        char buf[200];
        std::snprintf(buf, sizeof(buf),
            "BUG FIRED: iter %d, chain %s, first divergent buffer = %s. "
            "%lld iters since.",
            d.first_div_iter,
            CHAIN_LBL[d.first_div_chain],
            d.first_div_tag.c_str(),
            (long long)(int(d.iter) - d.first_div_iter));
        notes.push_back({RD_B, buf});
    } else if (d.iter > 0) {
        char buf[120];
        std::snprintf(buf, sizeof(buf),
            "Determinism: %d iters all %d/8 chains matched. Bug not yet observed.",
            d.total_match_iters, N_CHAINS);
        notes.push_back({GR, buf});
    }

    if (d.wall_window.size() >= 16) {
        double p50 = dq_pct(d.wall_window, 0.50);
        double p99 = dq_pct(d.wall_window, 0.99);
        if (p50 > 0 && p99 > 2.0 * p50) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                "Latency jitter: p99 %.0f ms = %.1fx p50 %.0f ms — "
                "tail-latency variance is high.",
                p99, p99 / p50, p50);
            notes.push_back({YL, buf});
        }
    }

    std::vector<std::pair<std::string, double>> ranked;
    double total_kn = 0.0;
    for (auto& [n, k] : d.kernels) {
        ranked.push_back({n, k.ns_this_iter});
        total_kn += k.ns_this_iter;
    }
    std::sort(ranked.begin(), ranked.end(),
        [](auto& a, auto& b){ return a.second > b.second; });
    if (ranked.size() >= 2 && total_kn > 0) {
        double pct1 = ranked[0].second / total_kn;
        double pct2 = ranked[1].second / total_kn;
        if (pct1 > 0.10) {
            char buf[180];
            std::snprintf(buf, sizeof(buf),
                "Top kernel: %s = %.1f%% of compute. "
                "Top-2 (+%s = %.1f%%) = %.1f%% combined.",
                ranked[0].first.c_str(), 100.0 * pct1,
                ranked[1].first.c_str(), 100.0 * pct2,
                100.0 * (pct1 + pct2));
            notes.push_back({YL, buf});
        }
    }

    auto m = d.kernels.find("memcpy(M2D)");
    if (m != d.kernels.end() && m->second.ns_this_iter > 1.0e6) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
            "memcpy(M2D) = %s/iter — host→device transfer is heavy.",
            fmt_us(m->second.ns_this_iter).c_str());
        notes.push_back({YL, buf});
    }

    uint64_t used = d.model_bytes_est + d.kv_bytes_total +
                    d.dn_bytes_total + d.workspace_bytes_est;
    double frac = double(used) / double(d.gpu_mem_total_b);
    if (frac > 0.85) {
        char buf[140];
        std::snprintf(buf, sizeof(buf),
            "GPU memory at %.0f%% — may be tight for additional context.",
            100.0 * frac);
        notes.push_back({RD_B, buf});
    } else if (frac > 0.50) {
        char buf[120];
        std::snprintf(buf, sizeof(buf),
            "GPU memory at %.0f%% — comfortable.", 100.0 * frac);
        notes.push_back({GR, buf});
    }

    if (notes.empty()) notes.push_back({DIM, "No anomalies detected."});
    return notes;
}

void render(const Dashboard& d) {
    std::string out;
    out.reserve(16384);
    out += HOME;

    // ---- Header ----
    {
        char buf[256];
        const double up_s = (now_ms() - d.app_start_ms) / 1000.0;
        const char* status_color =
            (d.first_div_iter < 0) ? GR_B : RD_B;
        const char* status_text =
            (d.first_div_iter < 0) ? "✓ ALL 8 CHAINS MATCH"
                                    : "✗ DIVERGENCE DETECTED";
        std::snprintf(buf, sizeof(buf),
            "%s%s ie-bug-monitor %s%s — 8-chain Live Dashboard%s%s\n",
            BL, INV, R, BL, R, CLR_EOL);
        out += buf;
        std::snprintf(buf, sizeof(buf),
            " %sModel%s Qwen3.6-35B-A3B-Q4_K_M   %sDevice%s %s   "
            "%sStatus%s %s%s%s   %suptime%s %s%s\n",
            DIM, R,
            DIM, R, d.device_name.c_str(),
            DIM, R, status_color, status_text, R,
            DIM, R, fmt_clock(up_s).c_str(),
            CLR_EOL);
        out += buf;
        std::snprintf(buf, sizeof(buf),
            " %siter%s %s%u%s/%u   %schains%s %d (A..H) parallel   "
            "%sunique outputs%s %s%d%s%s\n",
            DIM, R, BOLD, d.iter, R, d.max_iters,
            DIM, R, N_CHAINS,
            DIM, R,
            (d.n_unique_outputs > 1 ? RD_B : GR),
            d.n_unique_outputs, R,
            CLR_EOL);
        out += buf;
        out += DIM;
        out += " ──────────────────────────────────────────────────────────────────────────────";
        out += R;
        out += CLR_EOL;
        out += "\n";
    }

    // ---- THROUGHPUT ----
    {
        char buf[256];
        double cur_ms = d.wall_window.empty() ? 0.0 : d.wall_window.back();
        double per_chain_ms = cur_ms / double(N_CHAINS);
        double decode_per_chain_eq = (per_chain_ms > 0)
            ? 1000.0 / per_chain_ms : 0.0;
        double decode_8combined = (cur_ms > 0)
            ? double(N_CHAINS) * 1000.0 / cur_ms : 0.0;
        out += "\n";
        std::snprintf(buf, sizeof(buf),
            "%sTHROUGHPUT%s%s\n", BL, R, CLR_EOL);
        out += buf;
        std::snprintf(buf, sizeof(buf),
            "  %sprompt processing%s   %s%6.1f tok/s%s  (T=%.0f, "
            "%.1f ms, single chain)%s\n",
            DIM, R, BOLD, d.prompt_tok_s, R,
            d.prompt_T, d.prompt_ms, CLR_EOL);
        out += buf;
        std::snprintf(buf, sizeof(buf),
            "  %sdecode (1-chain eq)%s %s%6.1f tok/s%s  (engine speed if alone)%s\n",
            DIM, R, BOLD, decode_per_chain_eq, R, CLR_EOL);
        out += buf;
        std::snprintf(buf, sizeof(buf),
            "  %sdecode (8 chains)%s   %s%6.1f tok/s%s  (combined this iter)%s\n",
            DIM, R, BOLD, decode_8combined, R, CLR_EOL);
        out += buf;
    }

    // ---- DETERMINISM (8 chains) ----
    {
        char buf[256];
        out += "\n";
        const char* hdr_color = (d.n_unique_outputs > 1) ? RD_B : BL;
        std::snprintf(buf, sizeof(buf),
            "%sDETERMINISM%s  (8 parallel chains, independent KV+DN+conv state)%s\n",
            hdr_color, R, CLR_EOL);
        out += buf;

        std::snprintf(buf, sizeof(buf),
            "  %s%-6s %-18s %-12s %s%s\n",
            DIM, "chain", "combined hash", "vs A", "first divergent buffer", CLR_EOL);
        out += buf;
        out += "  ";
        out += DIM;
        out += "──────────────────────────────────────────────────────────────────────";
        out += R;
        out += CLR_EOL;
        out += "\n";

        for (int c = 0; c < N_CHAINS; ++c) {
            const char* color;
            const char* status;
            const char* tag;
            char tag_buf[80];
            if (c == 0) {
                color = (d.n_unique_outputs > 1) ? GR_B : GR;
                status = "ref";
                tag = "—";
            } else if (!d.chain_diff[c].diverged) {
                color = GR;
                status = "✓ match";
                tag = "—";
            } else {
                color = RD_B;
                status = "✗ DIFF";
                std::snprintf(tag_buf, sizeof(tag_buf), "%s",
                              d.chain_diff[c].first_diff_tag.c_str());
                tag = tag_buf;
            }
            std::snprintf(buf, sizeof(buf),
                "  %s%-6s%s %s%016lx%s %s%-12s%s %s%s\n",
                BOLD, CHAIN_LBL[c], R,
                color, (unsigned long)d.chain_combined[c], R,
                color, status, R,
                tag, CLR_EOL);
            out += buf;
        }

        // Convergence summary + history strip
        std::snprintf(buf, sizeof(buf),
            "  %sconvergence%s %s%d%s/%d match chain A   "
            "%sany-divergence history (last %d):%s ",
            DIM, R,
            (d.n_unique_outputs > 1 ? RD_B : GR),
            // count of chains that match A (including A)
            int(N_CHAINS - std::count_if(d.chain_diff.begin(), d.chain_diff.end(),
                [](const ChainDiff& x){ return x.diverged; })),
            R,
            N_CHAINS,
            DIM, int(d.diverge_strip.size()), R);
        out += buf;
        for (int v : d.diverge_strip) {
            if (v == 0) { out += GR; out += SPK[1]; }
            else        { out += RD_B; out += SPK[8]; }
        }
        out += R;
        out += CLR_EOL;
        out += "\n";

        if (d.first_div_iter >= 0) {
            std::snprintf(buf, sizeof(buf),
                "  %sfirst-ever divergence:%s %siter %d, chain %s, buffer %s%s%s\n",
                DIM, R, RD_B,
                d.first_div_iter,
                CHAIN_LBL[d.first_div_chain],
                d.first_div_tag.c_str(),
                R, CLR_EOL);
            out += buf;
        } else {
            std::snprintf(buf, sizeof(buf),
                "  %sfirst-ever divergence:%s %s—%s%s\n",
                DIM, R, GR, R, CLR_EOL);
            out += buf;
        }
    }

    // ---- LATENCY ----
    {
        char buf[256];
        out += "\n";
        std::snprintf(buf, sizeof(buf),
            "%sLATENCY%s  (per-iter wall ms across %d chains, last %d iters)%s\n",
            BL, R, N_CHAINS, int(d.wall_window.size()), CLR_EOL);
        out += buf;
        out += "  ";
        if (!d.wall_window.empty()) {
            double mn = dq_min(d.wall_window);
            double mx = dq_max(d.wall_window);
            std::string sl = spark_line(d.wall_window,
                std::max(0.0, mn * 0.9), mx);
            out += GR;
            out += sl;
            out += R;
        }
        out += CLR_EOL;
        out += "\n";
        double cur = d.wall_window.empty() ? 0.0 : d.wall_window.back();
        double p50 = dq_pct(d.wall_window, 0.50);
        double p95 = dq_pct(d.wall_window, 0.95);
        double mx  = dq_max(d.wall_window);
        std::snprintf(buf, sizeof(buf),
            "  %scur%s %.0f ms   %sp50%s %.0f   %sp95%s %.0f   "
            "%smax%s %.0f   %sper-chain%s %.1f ms%s\n",
            DIM, R, cur,
            DIM, R, p50,
            DIM, R, p95,
            DIM, R, mx,
            DIM, R, cur / double(N_CHAINS),
            CLR_EOL);
        out += buf;
    }

    // ---- TOP KERNELS ----
    {
        char buf[512];
        out += "\n";
        std::snprintf(buf, sizeof(buf),
            "%sTOP KERNELS%s  (per iter, summed across %d chains)%s\n",
            BL, R, N_CHAINS, CLR_EOL);
        out += buf;

        std::vector<std::pair<std::string, KernelRoll>> ranked(
            d.kernels.begin(), d.kernels.end());
        double total = 0.0;
        for (auto& [_, k] : ranked) total += k.ns_this_iter;
        std::sort(ranked.begin(), ranked.end(),
            [](auto& a, auto& b){
                return a.second.ns_this_iter > b.second.ns_this_iter; });

        const int kKernelsShown = 8;
        std::snprintf(buf, sizeof(buf),
            "  %s%-26s %10s  %5s  %5s  %9s  %s%s\n",
            DIM, "kernel", "this iter", "%", "calls",
            "all-time", "trend (60)", R);
        out += buf;
        out += "  ";
        out += DIM;
        out += "─────────────────────────────────────────────────────────────────────────────────";
        out += R;
        out += CLR_EOL;
        out += "\n";

        int shown = 0;
        for (auto& [name, k] : ranked) {
            if (shown >= kKernelsShown) break;
            ++shown;
            double pct = (total > 0.0) ? 100.0 * k.ns_this_iter / total : 0.0;
            const char* color = (pct >= 10.0) ? YL_B
                                                : (pct >= 5.0 ? YL : DIM);
            std::string spk = spark_line(k.ns_window,
                                          0.0,
                                          std::max(1.0, k.max_seen_ns));
            std::snprintf(buf, sizeof(buf),
                "  %s%-26s%s  %s%9s%s  %s%4.1f%%%s  %5d  %9s  %s%s%s%s\n",
                color, name.c_str(), R,
                BOLD, fmt_us(k.ns_this_iter).c_str(), R,
                color, pct, R,
                k.calls_this_iter,
                fmt_us(k.all_time_ns).c_str(),
                color, spk.c_str(), R,
                CLR_EOL);
            out += buf;
        }
        for (int i = shown; i < kKernelsShown; ++i) {
            out += " "; out += CLR_EOL; out += "\n";
        }
        std::snprintf(buf, sizeof(buf),
            "  %stotal kernel%s %s/iter  (%d kernels tracked)%s\n",
            DIM, R, fmt_us(total).c_str(),
            int(d.kernels.size()), CLR_EOL);
        out += buf;
    }

    // ---- MEMORY ----
    {
        char buf[256];
        out += "\n";
        std::snprintf(buf, sizeof(buf),
            "%sMEMORY%s  (estimate: model + 8 KV + 8 DN-state + 8 DN-conv + workspace)%s\n",
            BL, R, CLR_EOL);
        out += buf;
        const uint64_t used = d.model_bytes_est + d.kv_bytes_total +
                              d.dn_bytes_total + d.workspace_bytes_est;
        const double frac = double(used) / double(d.gpu_mem_total_b);
        const char* color =
            (frac > 0.85) ? RD_B : (frac > 0.70 ? YL : GR);
        std::snprintf(buf, sizeof(buf),
            "  GPU mem  %s  %s%5.1f%%%s  %s/%s%s\n",
            h_bar(frac, 30, color).c_str(),
            color, 100.0 * frac, R,
            fmt_bytes(used).c_str(),
            fmt_bytes(d.gpu_mem_total_b).c_str(),
            CLR_EOL);
        out += buf;
        std::snprintf(buf, sizeof(buf),
            "  %smodel%s %s   %sKV (×8 chains)%s %s   "
            "%sDN+conv (×8)%s %s   %sworkspace%s %s%s\n",
            DIM, R, fmt_bytes(d.model_bytes_est).c_str(),
            DIM, R, fmt_bytes(d.kv_bytes_total).c_str(),
            DIM, R, fmt_bytes(d.dn_bytes_total).c_str(),
            DIM, R, fmt_bytes(d.workspace_bytes_est).c_str(),
            CLR_EOL);
        out += buf;
    }

    // ---- FOCUS ----
    {
        out += "\n";
        out += BL;
        out += "FOCUS RECOMMENDATIONS";
        out += R;
        out += CLR_EOL;
        out += "\n";
        auto notes = focus_notes(d);
        for (auto& [color, msg] : notes) {
            out += "  ";
            out += color; out += msg; out += R;
            out += CLR_EOL; out += "\n";
        }
        for (size_t i = notes.size(); i < 5; ++i) {
            out += " "; out += CLR_EOL; out += "\n";
        }
    }

    // ---- Footer ----
    out += "\n";
    out += DIM;
    out += " Ctrl-C to exit. ";
    out += R;
    out += CLR_EOL;
    out += "\n";
    out += CLR_EOS;

    std::fwrite(out.data(), 1, out.size(), stdout);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf_path =
        "/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf";
    std::string text_path;
    uint32_t    max_iters = 1024;
    uint32_t    prompt_T  = 128;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf"      && i + 1 < argc) gguf_path = argv[++i];
        else if (a == "--max-iters" && i + 1 < argc) max_iters = std::atoi(argv[++i]);
        else if (a == "--text"      && i + 1 < argc) text_path = argv[++i];
        else if (a == "--prompt-T"  && i + 1 < argc) prompt_T  = std::atoi(argv[++i]);
    }

    std::signal(SIGINT, on_sigint);

    std::printf("ie-bug-monitor (8-chain) — initializing...\n"
                "  GGUF      : %s\n  max iters : %u\n  prompt T  : %u\n",
                gguf_path.c_str(), max_iters, prompt_T);
    std::fflush(stdout);

    // Engine init
    ie::DeviceAllocator alloc;
    if (auto err = alloc.init(); !err.empty()) {
        std::fprintf(stderr, "alloc: %s\n", err.c_str()); return 1;
    }
    auto& q = alloc.queue();
    std::string device_name =
        q.get_device().get_info<sycl::info::device::name>();
    std::printf("  device    : %s\n", device_name.c_str());
    std::fflush(stdout);

    ie::GgufReader g;
    if (auto err = g.open(gguf_path); !err.empty()) {
        std::fprintf(stderr, "gguf open: %s\n", err.c_str()); return 1;
    }
    ie::Tokenizer tok;
    if (auto err = tok.load_from_gguf(g); !err.empty()) {
        std::fprintf(stderr, "tokenizer: %s\n", err.c_str()); return 1;
    }

    std::printf("  loading model (~22 GB)...\n");
    std::fflush(stdout);
    const double t_load = now_ms();
    ie::QwenConfig cfg;
    ie::QwenModel  model;
    if (auto err = model.load(alloc, g, cfg); !err.empty()) {
        std::fprintf(stderr, "model.load: %s\n", err.c_str()); return 1;
    }
    std::printf("  loaded in %.1f s\n", (now_ms() - t_load) / 1000.0);

    std::string corpus = text_path.empty() ? kSampleText
                                            : read_text_file(text_path);
    auto ids = tok.encode(corpus, /*allow_special=*/false);
    if (tok.add_bos_token() && tok.bos_token_id() >= 0 &&
        (ids.empty() || ids.front() != tok.bos_token_id())) {
        ids.insert(ids.begin(), tok.bos_token_id());
    }
    if (uint32_t(ids.size()) < prompt_T + max_iters) {
        std::fprintf(stderr,
            "corpus has only %zu tokens; need >= %u (prompt_T %u + max_iters %u)\n",
            ids.size(), prompt_T + max_iters, prompt_T, max_iters);
        if (uint32_t(ids.size()) <= prompt_T + 8) {
            std::fprintf(stderr, "fatal\n"); return 1;
        }
        max_iters = std::min(max_iters, uint32_t(ids.size()) - prompt_T - 1);
        std::fprintf(stderr, "  capping max_iters to %u\n", max_iters);
    }

    if (auto err = model.ensure_workspace(prompt_T); !err.empty()) {
        std::fprintf(stderr, "ws: %s\n", err.c_str()); return 1;
    }
    const uint32_t max_ctx = std::max<uint32_t>(prompt_T + max_iters + 4, 64);
    if (auto err = model.ensure_attn_partials(max_ctx); !err.empty()) {
        std::fprintf(stderr, "attn_partials: %s\n", err.c_str()); return 1;
    }
    const uint32_t L_full = cfg.n_layers / cfg.full_attn_interval;
    const uint32_t L_lin  = cfg.n_layers - L_full;

    ie::KvCacheConfig kvcfg{};
    kvcfg.n_layers_full = L_full;
    kvcfg.n_kv_heads    = cfg.n_kv_heads;
    kvcfg.max_ctx       = max_ctx;
    kvcfg.head_dim      = cfg.head_dim;
    kvcfg.use_int8      = false;
    ie::DeltaNetStateConfig dncfg{
        L_lin, cfg.ssm_n_v_heads, cfg.ssm_head_dim, cfg.ssm_head_dim,
        cfg.ssm_inner * 2, cfg.ssm_conv_kernel
    };

    // ---- Allocate 8 chains ----
    std::printf("  allocating %d chains... ", N_CHAINS);
    std::fflush(stdout);
    std::array<Chain, N_CHAINS> chains;
    for (int c = 0; c < N_CHAINS; ++c) {
        if (auto e = chains[c].kv.init(alloc, kvcfg); !e.empty()) {
            std::fprintf(stderr, "kv[%d]: %s\n", c, e.c_str()); return 1;
        }
        if (auto e = chains[c].dn.init(alloc, dncfg); !e.empty()) {
            std::fprintf(stderr, "dn[%d]: %s\n", c, e.c_str()); return 1;
        }
        chains[c].d_logits =
            sycl::malloc_device<sycl::half>(cfg.vocab, q);
        chains[c].kv.reset();
        chains[c].dn.reset(q);
    }
    std::printf("done\n");

    auto* d_ids = sycl::malloc_device<int32_t>(prompt_T + max_iters, q);
    q.memcpy(d_ids, ids.data(),
             (prompt_T + max_iters) * sizeof(int32_t)).wait();

    HashScratch scratch;

    // ---- Phase 0: prompt processing measurement (single chain) ----
    std::printf("  measuring prompt processing (T=%u)...\n", prompt_T);
    std::fflush(stdout);
    chains[0].kv.reset();
    chains[0].dn.reset(q);
    const double t_p0 = now_ms();
    model.forward(q, d_ids, prompt_T, /*start_pos=*/0,
                  chains[0].kv, chains[0].dn,
                  chains[0].d_logits).wait();
    const double prompt_ms = now_ms() - t_p0;
    const double prompt_tok_s = double(prompt_T) / (prompt_ms / 1000.0);
    std::printf("  prompt: %.1f ms → %.1f tok/s\n", prompt_ms, prompt_tok_s);

    // Reset all chains so decode starts from clean state.
    for (auto& c : chains) {
        c.kv.reset();
        c.dn.reset(q);
    }

    // ---- Profiler ----
    ie::KernelProfiler prof;
    ie::g_profiler = &prof;

    // ---- Dashboard state ----
    Dashboard d;
    d.device_name        = device_name;
    d.max_iters          = max_iters;
    d.app_start_ms       = now_ms();
    d.prompt_T           = double(prompt_T);
    d.prompt_ms          = prompt_ms;
    d.prompt_tok_s       = prompt_tok_s;
    {
        const uint64_t kv_one = uint64_t(L_full) * cfg.n_kv_heads *
            max_ctx * cfg.head_dim * sizeof(sycl::half) * 2;
        const uint64_t dn_state_one = uint64_t(L_lin) * cfg.ssm_n_v_heads *
            cfg.ssm_head_dim * cfg.ssm_head_dim * sizeof(float);
        const uint64_t dn_conv_one  = uint64_t(L_lin) *
            cfg.ssm_inner * 2 * (cfg.ssm_conv_kernel - 1) *
            sizeof(sycl::half);
        d.kv_bytes_total      = kv_one * N_CHAINS;
        d.dn_bytes_total      = (dn_state_one + dn_conv_one) * N_CHAINS;
        d.workspace_bytes_est = uint64_t(prompt_T) << 20;  // rough
    }

    std::printf("  starting 8-chain decode loop...\n");
    std::fflush(stdout);

    std::printf("%s%s", HIDE_CUR, CLR_SCR);
    std::fflush(stdout);

    // ---- Main decode loop ----
    for (uint32_t k = 0; k < max_iters && !g_quit; ++k) {
        prof.begin_step();

        const auto t0 = std::chrono::steady_clock::now();
        // Each chain decodes the same token at position k starting from
        // its freshly-reset state.  KV cache grows from length=0 as the
        // model writes each new slot.  This matches ie-bug-live's flow
        // and avoids reading uninitialized cache slots.
        for (int c = 0; c < N_CHAINS; ++c) {
            model.forward(q,
                d_ids + k,
                /*T=*/1, /*start_pos=*/k,
                chains[c].kv, chains[c].dn,
                chains[c].d_logits).wait();
        }
        const auto t1 = std::chrono::steady_clock::now();
        const double wall_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Hash each chain's persistent state.
        for (int c = 0; c < N_CHAINS; ++c) {
            hash_chain(q, chains[c], dncfg, kvcfg, cfg.vocab, scratch);
        }

        // Compare each chain vs A.
        for (int c = 0; c < N_CHAINS; ++c) {
            d.chain_combined[c] = chains[c].hashes.combined;
            if (c == 0) {
                d.chain_diff[0] = ChainDiff{};
            } else {
                d.chain_diff[c] = diff_vs_ref(chains[0].hashes,
                                              chains[c].hashes);
            }
        }
        // Count unique outputs (combined hashes).
        {
            std::vector<uint64_t> uniq;
            for (auto h : d.chain_combined) {
                if (std::find(uniq.begin(), uniq.end(), h) == uniq.end())
                    uniq.push_back(h);
            }
            d.n_unique_outputs = int(uniq.size());
        }

        const bool any_diverged =
            std::any_of(d.chain_diff.begin(), d.chain_diff.end(),
                [](const ChainDiff& x){ return x.diverged; });
        if (any_diverged) {
            ++d.total_diff_iters;
            if (d.first_div_iter < 0) {
                d.first_div_iter = int(k);
                // First divergent chain (lowest-index ≠ A).
                for (int c = 1; c < N_CHAINS; ++c) {
                    if (d.chain_diff[c].diverged) {
                        d.first_div_chain = c;
                        d.first_div_tag   = d.chain_diff[c].first_diff_tag;
                        break;
                    }
                }
            }
        } else {
            ++d.total_match_iters;
        }

        d.iter = k + 1;
        d.wall_window.push_back(wall_ms);
        while (int(d.wall_window.size()) > d.graph_n)
            d.wall_window.pop_front();
        d.diverge_strip.push_back(any_diverged ? 1 : 0);
        while (int(d.diverge_strip.size()) > d.graph_n)
            d.diverge_strip.pop_front();

        // Harvest profile.
        auto stats = prof.harvest();
        for (auto& [_, kr] : d.kernels) {
            kr.ns_this_iter    = 0.0;
            kr.calls_this_iter = 0;
        }
        for (auto& s : stats) {
            std::string nm = short_name(s.name);
            auto& kr = d.kernels[nm];
            kr.ns_this_iter    += double(s.total_ns);
            kr.calls_this_iter += int(s.calls);
            kr.all_time_ns     += double(s.total_ns);
            kr.all_time_calls  += s.calls;
            kr.max_seen_ns = std::max(kr.max_seen_ns, double(s.total_ns));
        }
        for (auto& [_, kr] : d.kernels) {
            kr.ns_window.push_back(kr.ns_this_iter);
            while (int(kr.ns_window.size()) > d.graph_n)
                kr.ns_window.pop_front();
        }

        render(d);
    }

    // Restore terminal
    std::printf("%s\n", SHOW_CUR);
    std::fflush(stdout);

    // Final summary
    std::printf("\n%s════ run complete ════%s\n", BL, R);
    std::printf("  iters run        : %u of %u\n", d.iter, d.max_iters);
    std::printf("  prompt           : %.1f tok/s (T=%u, %.1f ms)\n",
                d.prompt_tok_s, prompt_T, d.prompt_ms);
    if (!d.wall_window.empty()) {
        std::printf("  decode (8-chain) : last iter %.0f ms = %.1f tok/s combined\n",
                    d.wall_window.back(),
                    double(N_CHAINS) * 1000.0 / d.wall_window.back());
        std::printf("                     per-chain equivalent %.1f tok/s\n",
                    double(N_CHAINS) * 1000.0 /
                    d.wall_window.back() / 1.0);
    }
    std::printf("  match iters      : %d\n", d.total_match_iters);
    std::printf("  diverged iters   : %d\n", d.total_diff_iters);
    if (d.first_div_iter >= 0) {
        std::printf("  first divergence : iter %d, chain %s, buffer %s\n",
                    d.first_div_iter,
                    CHAIN_LBL[d.first_div_chain],
                    d.first_div_tag.c_str());
        std::printf("  %s>>> BUG REPRODUCED <<<%s\n", RD_B, R);
    } else {
        std::printf("  no divergence in this run.\n");
    }

    // Cleanup
    for (auto& c : chains) {
        if (c.d_logits) sycl::free(c.d_logits, q);
    }
    sycl::free(d_ids, q);
    ie::g_profiler = nullptr;
    return d.first_div_iter < 0 ? 0 : 1;
}
