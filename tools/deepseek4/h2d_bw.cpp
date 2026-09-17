// Minimal H2D / D2H bandwidth probe for the DeepSeek-V4 expert-streaming feasibility question.
// Measures sustained PCIe transfer rate for chunk sizes matching a single MoE expert
// (~10.4 MB: gate 3.06 + up 3.06 + down 4.25) and for a whole-layer working set (~64 MB = 6 experts).
#include <sycl/sycl.hpp>
#include <cstdio>
#include <chrono>
#include <vector>
#include <algorithm>

using clk = std::chrono::steady_clock;

static double time_copy(sycl::queue& q, void* dst, const void* src, size_t bytes, int iters) {
    // warmup
    for (int i = 0; i < 3; ++i) q.memcpy(dst, src, bytes).wait();
    std::vector<double> ts;
    for (int i = 0; i < iters; ++i) {
        auto t0 = clk::now();
        q.memcpy(dst, src, bytes).wait();
        auto t1 = clk::now();
        ts.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];   // median
}

int main(int argc, char** argv) {
    sycl::queue q{sycl::gpu_selector_v};
    auto dev = q.get_device();
    printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    printf("global mem: %.1f GB\n",
           dev.get_info<sycl::info::device::global_mem_size>() / 1e9);

    const size_t MB = 1024ull * 1024ull;
    struct Case { const char* name; size_t bytes; };
    Case cases[] = {
        {"1 expert (10.4 MB)",        (size_t)(10.4 * MB)},
        {"6 experts / 1 layer (62 MB)", (size_t)(62.2 * MB)},
        {"256 MB",                     256 * MB},
        {"1 GB",                      1024 * MB},
    };

    // pinned host allocation — the realistic path for a streaming engine
    size_t maxb = 1024 * MB;
    void* h_pinned = sycl::malloc_host(maxb, q);
    void* h_pageable = malloc(maxb);
    void* d = sycl::malloc_device(maxb, q);
    if (!h_pinned || !h_pageable || !d) { printf("alloc failed\n"); return 1; }
    memset(h_pinned, 0xA5, maxb);
    memset(h_pageable, 0xA5, maxb);

    printf("\n%-28s %12s %12s %12s\n", "case", "H2D pinned", "H2D pageable", "D2H pinned");
    printf("%-28s %12s %12s %12s\n", "", "GB/s", "GB/s", "GB/s");
    for (auto& c : cases) {
        int iters = c.bytes > 256 * MB ? 5 : 20;
        double t_pin  = time_copy(q, d, h_pinned,   c.bytes, iters);
        double t_page = time_copy(q, d, h_pageable, c.bytes, iters);
        double t_d2h  = time_copy(q, h_pinned, d,   c.bytes, iters);
        printf("%-28s %12.2f %12.2f %12.2f\n", c.name,
               c.bytes / t_pin / 1e9, c.bytes / t_page / 1e9, c.bytes / t_d2h / 1e9);
    }

    // Sustained streaming estimate: what a decode step would actually cost.
    // 43 layers x 6 experts x 10.37 MB = ~2.68 GB per token if nothing is cached.
    double bw = (1024.0 * MB) / time_copy(q, d, h_pinned, 1024 * MB, 5) / 1e9;
    double per_token_gb = 2.68;
    printf("\nsustained H2D (1 GB, pinned): %.2f GB/s\n", bw);
    printf("full expert set per token: %.2f GB -> %.1f ms -> %.1f tok/s (0%% cache hit)\n",
           per_token_gb, per_token_gb / bw * 1000.0, bw / per_token_gb);
    for (double hit : {0.25, 0.50, 0.75, 0.90}) {
        double eff = per_token_gb * (1.0 - hit);
        printf("  at %2.0f%% expert-cache hit: %.2f GB -> %.1f ms -> %.1f tok/s\n",
               hit * 100, eff, eff / bw * 1000.0, bw / eff);
    }

    sycl::free(h_pinned, q); sycl::free(d, q); free(h_pageable);
    return 0;
}
