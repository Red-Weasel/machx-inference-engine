// ie-attn-profile — per-sub-kernel attention profiler with live terminal feed.
//
// Instruments full_attention_fa2_decode (fp16) and full_attention_fa2_decode_int8
// using SYCL event profiling to measure append / partial / combine independently.
// No model or GGUF needed — allocates synthetic KV caches and runs the kernels
// directly at the requested context lengths.
//
// Usage:
//   ie-attn-profile [--ctx 1024,4096,16384] [--steps 20] [--layers 10]
//                   [--mode fp16|int8|both] [--warmup 4]
//                   [--q-heads 16] [--kv-heads 2] [--head-dim 256]

#include <ie/ops.hpp>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

// ── ANSI palette ──────────────────────────────────────────────────────────────
#define R   "\033[0m"
#define B   "\033[1m"
#define DIM "\033[2m"
#define CYN "\033[36m"
#define GRN "\033[32m"
#define YLW "\033[33m"
#define RED "\033[31m"
#define MAG "\033[35m"
#define WHT "\033[97m"
#define BLU "\033[34m"

static volatile bool g_stop = false;
static void on_signal(int) { g_stop = true; }

// ── tiny helpers ──────────────────────────────────────────────────────────────
static std::string fmt_ctx(uint32_t v) {
    if (v >= 1000) {
        char buf[16];
        snprintf(buf, sizeof(buf), "%u,%03u", v / 1000, v % 1000);
        return buf;
    }
    return std::to_string(v);
}

static const char* pct_color(double pct) {
    if (pct > 85.0) return RED;
    if (pct > 70.0) return YLW;
    return GRN;
}

static const char* delta_color(double delta_pct) {
    if (delta_pct >  15.0) return RED;
    if (delta_pct >   5.0) return YLW;
    if (delta_pct <  -5.0) return GRN;
    return DIM;
}

static std::string progress_bar(uint32_t done, uint32_t total, int width = 24) {
    int filled = (total > 0) ? int(double(done) / total * width) : 0;
    std::string s = "[";
    for (int i = 0; i < width; ++i)
        s += (i < filled) ? "\xe2\x96\x88" : "\xe2\x96\x91";  // ▓ / ░
    s += "]";
    return s;
}

// ── header ────────────────────────────────────────────────────────────────────
static void print_header(const std::string& dev, uint32_t q_h, uint32_t kv_h,
                          uint32_t hd, uint32_t layers, uint32_t steps,
                          uint32_t warmup) {
    auto now   = std::chrono::system_clock::now();
    auto now_t = std::chrono::system_clock::to_time_t(now);
    char ts[32];
    strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", localtime(&now_t));

    const int W = 76;
    auto line = [&](const std::string& s) {
        int pad = W - 2 - int(s.size());
        printf("  \xe2\x95\x91 %s%*s \xe2\x95\x91\n", s.c_str(), pad > 0 ? pad : 0, "");
    };

    printf("\n");
    printf("  \xe2\x95\x94");
    for (int i = 0; i < W - 2; ++i) printf("\xe2\x95\x90");
    printf("\xe2\x95\x97\n");

    line(std::string(B "ie-attn-profile" R "  \xe2\x94\x80  ") + dev + "  \xe2\x94\x80  " + ts);
    line("Qwen3.6-35B-A3B FA-2 decode: sub-kernel breakdown (append / partial / combine)");
    {
        char cfg[128];
        snprintf(cfg, sizeof(cfg),
                 "q_heads=%u  kv_heads=%u  head_dim=%u  layers=%u  steps=%u  warmup=%u",
                 q_h, kv_h, hd, layers, steps, warmup);
        line(cfg);
    }

    printf("  \xe2\x95\x9a");
    for (int i = 0; i < W - 2; ++i) printf("\xe2\x95\x90");
    printf("\xe2\x95\x9d\n\n");

    // Column header
    printf("  " B "%-8s  %-4s  %-6s  %-10s  %-12s  %-11s  %-10s  %s\n" R,
           "ctx", "mode", "step", "append ms", "partial ms", "combine ms",
           "total ms", "breakdown");
    printf("  ");
    for (int i = 0; i < 82; ++i) printf("\xe2\x94\x80");
    printf("\n");
    fflush(stdout);
}

// ── print one data row ────────────────────────────────────────────────────────
static void print_row(uint32_t ctx, bool is_int8, const char* step_label,
                      double app, double par, double comb,
                      bool is_avg, double fp16_par_for_delta) {
    double tot    = app + par + comb;
    double par_pct = (tot > 0.0) ? par / tot * 100.0 : 0.0;
    const char* mode_col = is_int8 ? CYN : GRN;
    const char* row_col  = is_avg  ? B   : "";

    // ctx column: bold if avg row
    printf("  %s%-8s%s  %s%-4s%s  %-6s  "
           DIM "%-10.3f" R "  %s%-12.3f%s  "
           DIM "%-11.3f" R "  %s%-10.3f%s  ",
           is_avg ? B : "", fmt_ctx(ctx).c_str(), R,
           mode_col, is_int8 ? "int8" : "fp16", R,
           step_label,
           app,
           pct_color(par_pct), par, R,
           comb,
           row_col, tot, R);

    // breakdown tags
    printf("%s%.0f%%%s partial", pct_color(par_pct), par_pct, R);

    if (fp16_par_for_delta > 0.0 && is_int8) {
        double delta = (par - fp16_par_for_delta) / fp16_par_for_delta * 100.0;
        printf("  %s%+.1f%% vs fp16%s", delta_color(delta), delta, R);
    }

    if (is_avg) printf("  " B "◀ avg" R);
    printf("\n");
    fflush(stdout);
}

// ── run one ctx × mode sweep ──────────────────────────────────────────────────
struct AvgResult {
    uint32_t ctx;
    bool     is_int8;
    double   append_ms, partial_ms, combine_ms;
};

static AvgResult run_sweep(sycl::queue& q,
                            uint32_t ctx_target,
                            bool use_int8,
                            uint32_t n_warmup,
                            uint32_t n_steps,
                            uint32_t n_layers,
                            // buffers
                            const sycl::half* q_buf,
                            const sycl::half* k_buf,
                            const sycl::half* v_buf,
                            sycl::half*       k_cache,
                            sycl::half*       v_cache,
                            int8_t*           k_int8,
                            int8_t*           v_int8,
                            sycl::half*       k_scales,
                            sycl::half*       v_scales,
                            sycl::half*       y_buf,
                            float*            parts,
                            uint32_t          n_q_heads,
                            uint32_t          n_kv_heads,
                            uint32_t          head_dim,
                            uint32_t          max_ctx,
                            double            fp16_par_avg) {
    std::vector<double> app_v, par_v, comb_v;
    app_v.reserve(n_steps);
    par_v.reserve(n_steps);
    comb_v.reserve(n_steps);

    for (uint32_t s = 0; s < n_warmup + n_steps && !g_stop; ++s) {
        bool measuring = (s >= n_warmup);
        uint32_t pos   = ctx_target + s;

        ie::AttnProfileData prof;

        for (uint32_t layer = 0; layer < n_layers && !g_stop; ++layer) {
            if (!use_int8) {
                ie::full_attention_fa2_decode(q, q_buf, k_buf, v_buf,
                                              k_cache, v_cache, y_buf, parts,
                                              pos, n_q_heads, n_kv_heads,
                                              head_dim, max_ctx, {},
                                              measuring ? &prof : nullptr);
            } else {
                ie::full_attention_fa2_decode_int8(q, q_buf, k_buf, v_buf,
                                                   k_int8, v_int8,
                                                   k_scales, v_scales,
                                                   nullptr, nullptr,
                                                   y_buf, parts,
                                                   pos, n_q_heads, n_kv_heads,
                                                   head_dim, max_ctx, {},
                                                   measuring ? &prof : nullptr);
            }
        }

        if (!measuring) {
            // Drain warmup before next iter so warmup kernels can't bleed
            // into the next step's submission.  Required for accurate
            // per-call profiling — `prof` aggregation only holds for
            // events that have actually completed.
            q.wait();
            continue;
        }

        // Wait for all kernels to actually finish so the host-side prof
        // accumulators (filled from event start/end timestamps) are valid.
        q.wait();

        double a = prof.append_ms()  / n_layers;
        double p = prof.partial_ms() / n_layers;
        double c = prof.combine_ms() / n_layers;
        app_v.push_back(a);
        par_v.push_back(p);
        comb_v.push_back(c);

        char step_label[12];
        snprintf(step_label, sizeof(step_label), "%u/%u", s - n_warmup + 1, n_steps);
        print_row(ctx_target, use_int8, step_label, a, p, c, false, fp16_par_avg);
    }

    if (app_v.empty()) return {ctx_target, use_int8, 0, 0, 0};

    auto avg_v = [](const std::vector<double>& v) {
        return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    };
    double a = avg_v(app_v), p = avg_v(par_v), c = avg_v(comb_v);

    // separator before avg line
    printf("  " DIM);
    for (int i = 0; i < 82; ++i) printf("\xe2\x94\x80");
    printf(R "\n");

    print_row(ctx_target, use_int8, "avg", a, p, c, true, fp16_par_avg);

    printf("  ");
    for (int i = 0; i < 82; ++i) printf("\xe2\x94\x81");
    printf("\n\n");
    fflush(stdout);

    return {ctx_target, use_int8, a, p, c};
}

// ── summary table ─────────────────────────────────────────────────────────────
static void print_summary(const std::vector<AvgResult>& res, uint32_t n_layers) {
    printf("\n  " B "SUMMARY  (per-layer average, %u layers simulated per step)\n" R, n_layers);
    printf("  ");
    for (int i = 0; i < 82; ++i) printf("\xe2\x95\x90");
    printf("\n");
    printf("  " B "%-8s  %-4s  %-11s  %-12s  %-11s  %-10s  %-8s  %s\n" R,
           "ctx", "mode", "append ms", "partial ms", "combine ms",
           "total ms", "par%", "vs fp16 partial");
    printf("  ");
    for (int i = 0; i < 82; ++i) printf("\xe2\x95\x90");
    printf("\n");

    uint32_t prev_ctx = 0;
    double   fp16_par = 0.0;
    for (const auto& r : res) {
        if (r.ctx != prev_ctx) {
            if (prev_ctx) {
                printf("  " DIM);
                for (int i = 0; i < 82; ++i) printf("\xe2\x94\x80");
                printf(R "\n");
            }
            prev_ctx = r.ctx;
            fp16_par = 0.0;
        }
        double tot = r.append_ms + r.partial_ms + r.combine_ms;
        double par_pct = (tot > 0.0) ? r.partial_ms / tot * 100.0 : 0.0;
        const char* mc = r.is_int8 ? CYN : GRN;

        printf("  %s%-8s%s  %s%-4s%s  "
               DIM "%-11.3f" R "  %s%-12.3f%s  "
               DIM "%-11.3f" R "  " B "%-10.3f" R "  "
               "%s%-7.1f%%%s",
               B, fmt_ctx(r.ctx).c_str(), R,
               mc, r.is_int8 ? "int8" : "fp16", R,
               r.append_ms,
               pct_color(par_pct), r.partial_ms, R,
               r.combine_ms, r.append_ms + r.partial_ms + r.combine_ms,
               pct_color(par_pct), par_pct, R);

        if (!r.is_int8) {
            fp16_par = r.partial_ms;
            printf("\n");
        } else if (fp16_par > 0.0) {
            double delta = (r.partial_ms - fp16_par) / fp16_par * 100.0;
            printf("  %s%+.1f%%%s\n", delta_color(delta), delta, R);
        } else {
            printf("\n");
        }
    }
    printf("  ");
    for (int i = 0; i < 82; ++i) printf("\xe2\x95\x90");
    printf("\n\n");
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    signal(SIGINT,  on_signal);
    signal(SIGTERM, on_signal);

    std::vector<uint32_t> ctx_list = {1024, 4096, 8192, 16384};
    uint32_t n_steps   = 20;
    uint32_t n_warmup  = 4;
    uint32_t n_layers  = 10;
    uint32_t n_q_heads = 16;
    uint32_t n_kv_heads = 2;
    uint32_t head_dim  = 256;
    bool do_fp16 = true;
    bool do_int8 = true;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--ctx") && i + 1 < argc) {
            ctx_list.clear();
            char* p = argv[++i];
            while (*p) {
                ctx_list.push_back(uint32_t(strtoul(p, &p, 10)));
                if (*p == ',') ++p;
            }
        } else if (!strcmp(argv[i], "--steps")   && i + 1 < argc) { n_steps    = uint32_t(atoi(argv[++i])); }
        else if (!strcmp(argv[i], "--warmup")     && i + 1 < argc) { n_warmup   = uint32_t(atoi(argv[++i])); }
        else if (!strcmp(argv[i], "--layers")     && i + 1 < argc) { n_layers   = uint32_t(atoi(argv[++i])); }
        else if (!strcmp(argv[i], "--q-heads")    && i + 1 < argc) { n_q_heads  = uint32_t(atoi(argv[++i])); }
        else if (!strcmp(argv[i], "--kv-heads")   && i + 1 < argc) { n_kv_heads = uint32_t(atoi(argv[++i])); }
        else if (!strcmp(argv[i], "--head-dim")   && i + 1 < argc) { head_dim   = uint32_t(atoi(argv[++i])); }
        else if (!strcmp(argv[i], "--mode") && i + 1 < argc) {
            const char* m = argv[++i];
            do_fp16 = (strstr(m, "fp16") || !strcmp(m, "both"));
            do_int8 = (strstr(m, "int8") || !strcmp(m, "both"));
        }
    }

    // Device + profiling queue
    sycl::device device;
    try { device = sycl::device{sycl::gpu_selector_v}; }
    catch (...) { fprintf(stderr, "No GPU found.\n"); return 1; }

    // In-order queue: profiler measures per-kernel start/end timestamps,
    // so kernels MUST execute serially.  Out-of-order would let the runtime
    // overlap layers (deps = {}) and pollute the per-call timings.
    sycl::queue q{device,
                  sycl::property_list{sycl::property::queue::enable_profiling{},
                                      sycl::property::queue::in_order{}}};

    auto dev_name = device.get_info<sycl::info::device::name>();
    print_header(dev_name, n_q_heads, n_kv_heads, head_dim, n_layers, n_steps, n_warmup);

    // Buffer allocation
    uint32_t max_ctx = *std::max_element(ctx_list.begin(), ctx_list.end())
                     + n_warmup + n_steps + 8;

    size_t q_sz      = size_t(n_q_heads)  * head_dim;
    size_t kv_sz     = size_t(n_kv_heads) * head_dim;
    size_t cache_sz  = size_t(n_kv_heads) * max_ctx * head_dim;
    size_t scales_sz = size_t(n_kv_heads) * max_ctx;
    // partials: worst-case super_chunks × q_heads × (head_dim+2)
    size_t n_super_max = (size_t(max_ctx) / 64 / 8 + 2);
    size_t parts_sz    = n_super_max * n_q_heads * (head_dim + 2);

    auto* q_buf   = sycl::malloc_device<sycl::half>(q_sz,    q);
    auto* k_buf   = sycl::malloc_device<sycl::half>(kv_sz,   q);
    auto* v_buf   = sycl::malloc_device<sycl::half>(kv_sz,   q);
    auto* k_cache = sycl::malloc_device<sycl::half>(cache_sz,  q);
    auto* v_cache = sycl::malloc_device<sycl::half>(cache_sz,  q);
    auto* y_buf   = sycl::malloc_device<sycl::half>(q_sz,    q);
    auto* parts   = sycl::malloc_device<float>(parts_sz,     q);
    auto* k_int8  = sycl::malloc_device<int8_t>(cache_sz,    q);
    auto* v_int8  = sycl::malloc_device<int8_t>(cache_sz,    q);
    auto* k_scl   = sycl::malloc_device<sycl::half>(scales_sz, q);
    auto* v_scl   = sycl::malloc_device<sycl::half>(scales_sz, q);

    // Initialize with non-trivial values so GPU cache behavior is realistic
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(cache_sz), [=](sycl::id<1> i) {
            float v = float(i[0] % 251) * 0.008f - 1.0f;
            k_cache[i] = sycl::half(v);
            v_cache[i] = sycl::half(v * 0.5f);
            k_int8[i]  = int8_t(i[0] % 255 - 127);
            v_int8[i]  = int8_t(i[0] % 255 - 127);
        });
    });
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(scales_sz), [=](sycl::id<1> i) {
            k_scl[i] = sycl::half(0.01f);
            v_scl[i] = sycl::half(0.01f);
        });
    });
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(q_sz), [=](sycl::id<1> i) {
            q_buf[i] = sycl::half(float(i[0] % 256) * 0.01f - 1.28f);
            if (i[0] < kv_sz) {
                k_buf[i] = sycl::half(float(i[0] % 256) * 0.01f - 1.28f);
                v_buf[i] = sycl::half(float(i[0] % 256) * 0.01f - 1.28f);
            }
        });
    });
    q.wait();

    printf("  " DIM "Buffers allocated (max_ctx=%u). Profiling...\n\n" R, max_ctx);
    fflush(stdout);

    // Main sweep
    std::vector<AvgResult> all_results;
    for (uint32_t ctx : ctx_list) {
        if (g_stop) break;
        double fp16_par_avg = 0.0;

        if (do_fp16) {
            auto r = run_sweep(q, ctx, false, n_warmup, n_steps, n_layers,
                               q_buf, k_buf, v_buf, k_cache, v_cache,
                               k_int8, v_int8, k_scl, v_scl, y_buf, parts,
                               n_q_heads, n_kv_heads, head_dim, max_ctx, 0.0);
            fp16_par_avg = r.partial_ms;
            all_results.push_back(r);
        }
        if (do_int8 && !g_stop) {
            auto r = run_sweep(q, ctx, true, n_warmup, n_steps, n_layers,
                               q_buf, k_buf, v_buf, k_cache, v_cache,
                               k_int8, v_int8, k_scl, v_scl, y_buf, parts,
                               n_q_heads, n_kv_heads, head_dim, max_ctx,
                               fp16_par_avg);
            all_results.push_back(r);
        }
    }

    if (!all_results.empty())
        print_summary(all_results, n_layers);

    // Cleanup
    sycl::free(q_buf,   q);  sycl::free(k_buf,   q);  sycl::free(v_buf,  q);
    sycl::free(k_cache, q);  sycl::free(v_cache, q);  sycl::free(y_buf,  q);
    sycl::free(parts,   q);  sycl::free(k_int8,  q);  sycl::free(v_int8, q);
    sycl::free(k_scl,   q);  sycl::free(v_scl,   q);
    return 0;
}
