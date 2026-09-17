// tests/unit/ds4_vision_gpu_test.cpp — P2 gate (docs/deepseek4/70_VISION_EXP_PORT_PLAN.md):
// GPU vision tower + aligner + block layout vs the numpy oracle golden
// (tests/oracle/ds4_vision_oracle.py, itself gated against the official torch
// modules). fp16 GEMM/attention drift vs fp32 → relative-L2 + per-row cosine
// for the aligner rows; the block layout (types/perm) must match exactly.
// Env: IE_DS4_VIS_SIDECAR (default: the apetersson Reference-Native-GGUF
// sidecar under ~/models), IE_DS4_VIS_GOLDEN (.bin). Skips without the golden.

#include "ie/allocator.hpp"
#include "ie/ds4_vision.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {
struct Golden {
    uint32_t n_vit_h, n_vit_w, n_llm_h, n_llm_w, start_pos, n_rows, hidden, n_patches;
    std::vector<float> pixels, aligned, rows;
    std::vector<int32_t> types, perm;
};

bool read_golden(const char* path, Golden& g) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return false;
    char magic[8];
    if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "DS4VORC1", 8)) { std::fclose(f); return false; }
    uint32_t h[8];
    if (std::fread(h, 4, 8, f) != 8) { std::fclose(f); return false; }
    g.n_vit_h = h[0]; g.n_vit_w = h[1]; g.n_llm_h = h[2]; g.n_llm_w = h[3];
    g.start_pos = h[4]; g.n_rows = h[5]; g.hidden = h[6]; g.n_patches = h[7];
    auto rd = [&](auto& v, size_t n) {
        v.resize(n);
        return std::fread(v.data(), sizeof(v[0]), n, f) == n;
    };
    const size_t H = size_t(g.n_vit_h) * 14, W = size_t(g.n_vit_w) * 14;
    const bool ok = rd(g.pixels, 3 * H * W) && rd(g.types, g.n_rows) &&
                    rd(g.perm, size_t(g.n_llm_h) * g.n_llm_w) &&
                    rd(g.aligned, size_t(g.n_llm_h) * g.n_llm_w * g.hidden) &&
                    rd(g.rows, size_t(g.n_rows) * g.hidden);
    std::fclose(f);
    return ok;
}

struct Cmp { double rel, max_d, min_cos, p05_cos, mean_abs; };
Cmp compare(const std::vector<float>& a, const std::vector<float>& b, uint32_t D) {
    double sum_sq = 0, ref_sq = 0, max_d = 0, sum_abs = 0;
    std::vector<double> cos;
    for (size_t r = 0; r < a.size() / D; ++r) {
        double dot = 0, na = 0, nb = 0;
        for (uint32_t j = 0; j < D; ++j) {
            const double x = a[r * D + j], y = b[r * D + j], d = x - y;
            sum_sq += d * d; ref_sq += x * x; max_d = std::max(max_d, std::fabs(d)); sum_abs += std::fabs(d);
            dot += x * y; na += x * x; nb += y * y;
        }
        cos.push_back(dot / std::sqrt(na * nb));
    }
    std::sort(cos.begin(), cos.end());
    const double p05 = cos.empty() ? 0.0 : cos[std::min(cos.size() - 1, cos.size() / 20)];
    return {std::sqrt(sum_sq / ref_sq), max_d, cos.empty() ? 0.0 : cos.front(), p05,
            a.empty() ? 0.0 : sum_abs / double(a.size())};
}
}  // namespace

#include "ie/ops.hpp"
#include <sycl/sycl.hpp>
// IE_DS4_VIS_PROBE=1: run the block-0 GEMM + bias shapes on fresh buffers first
// (DEVICE_LOST bisection, 2026-09-01).
static int probe() {
    ie::DeviceAllocator alloc;
    if (auto e = alloc.init("B70", 0); !e.empty()) { std::fprintf(stderr, "gpu: %s\n", e.c_str()); return 1; }
    sycl::queue& q = alloc.queue();
    const uint32_t M = 896, K = 1024, N = 3072;
    auto* A = static_cast<sycl::half*>(alloc.malloc(size_t(M) * K * 2));
    auto* B = static_cast<sycl::half*>(alloc.malloc(size_t(K) * N * 2));
    auto* C = static_cast<float*>(alloc.malloc(size_t(M) * N * 4));
    auto* b = static_cast<float*>(alloc.malloc(size_t(N) * 4));
    auto* y = static_cast<sycl::half*>(alloc.malloc(size_t(M) * N * 2));
    q.parallel_for(sycl::range<1>(size_t(M) * K), [=](sycl::id<1> i) { A[i] = sycl::half(0.01f); }).wait();
    q.parallel_for(sycl::range<1>(size_t(K) * N), [=](sycl::id<1> i) { B[i] = sycl::half(0.02f); }).wait();
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) { b[i] = 1.f; }).wait();
    std::fprintf(stderr, "[probe] buffers ok\n");
    try {
        ie::gemm_fp16(q, A, B, C, M, N, K); q.wait();
        float c0 = 0; q.memcpy(&c0, C, 4).wait();
        std::fprintf(stderr, "[probe] gemm ok C[0]=%g (expect %g)\n", c0, 0.01f * 0.02f * K);
    } catch (const std::exception& e) { std::fprintf(stderr, "[probe] gemm THREW: %s\n", e.what()); return 1; }
    try {
        q.parallel_for(sycl::range<1>(size_t(M) * N), [=](sycl::id<1> i) {
            float v = C[i] + (b ? b[uint32_t(i % N)] : 0.f);
            y[i] = sycl::half(v);
        }).wait();
        std::fprintf(stderr, "[probe] bias(no act) ok\n");
        q.parallel_for(sycl::range<1>(size_t(M) * N), [=](sycl::id<1> i) {
            float v = C[i] + (b ? b[uint32_t(i % N)] : 0.f);
            v = 0.5f * v * (1.0f + sycl::erf(v * 0.70710678118654752f));
            y[i] = sycl::half(v);
        }).wait();
        std::fprintf(stderr, "[probe] bias(erf) ok\n");
        const int act = 0;
        q.parallel_for(sycl::range<1>(size_t(M) * N), [=](sycl::id<1> i) {
            float v = C[i] + (b ? b[uint32_t(i % N)] : 0.f);
            if (act == 1) v = 0.5f * v * (1.0f + sycl::erf(v * 0.70710678118654752f));
            y[i] = sycl::half(v);
        }).wait();
        std::fprintf(stderr, "[probe] bias(act flag) ok\n");
    } catch (const std::exception& e) { std::fprintf(stderr, "[probe] bias THREW: %s\n", e.what()); return 1; }
    try {
        ie::ds4vis_detail::bias_act_f16_probe(&q, C, b, y, M, N, 0);
        std::fprintf(stderr, "[probe] encoder-TU bias(act 0) ok\n");
        q.memset(C, 0, size_t(M) * N * 4).wait();
        std::fprintf(stderr, "[probe] canary after encoder-TU bias ok\n");
        ie::ds4vis_detail::bias_act_f16_probe(&q, C, b, y, M, N, 1);
        q.memset(C, 0, size_t(M) * N * 4).wait();
        std::fprintf(stderr, "[probe] encoder-TU bias(act 1) + canary ok\n");
    } catch (const std::exception& e) { std::fprintf(stderr, "[probe] encoder-TU bias THREW: %s\n", e.what()); return 1; }
    return 0;
}

// IE_DS4_VIS_PROBE=2: mimic the encoder's allocation pattern (many small + big
// device buffers, pinned staging alloc/free) and memset everything.
static int probe2() {
    ie::DeviceAllocator alloc;
    if (auto e = alloc.init("B70", 0); !e.empty()) { std::fprintf(stderr, "gpu: %s\n", e.c_str()); return 1; }
    sycl::queue& q = alloc.queue();
    std::vector<std::pair<void*, size_t>> bufs;
    void* stage = sycl::malloc_host(80u << 20, q);
    auto up = [&](size_t bytes) {
        void* d = alloc.malloc(bytes);
        std::memset(stage, 1, std::min<size_t>(bytes, 80u << 20));
        q.memcpy(d, stage, std::min<size_t>(bytes, 80u << 20)).wait();
        bufs.push_back({d, bytes});
    };
    up(588 * 1024 * 2); up(4096);
    for (int L = 0; L < 32; ++L) {
        up(4096); up(4096); up(3072 * 1024 * 2); up(3072 * 4); up(1024 * 1024 * 2); up(4096);
        up(5632 * 1024 * 2); up(2816 * 1024 * 2);
    }
    up(4096); up(size_t(9216) * 4096 * 2); up(16384); up(size_t(4096) * 4096 * 2); up(16384);
    if (std::getenv("IE_DS4_VIS_PROBE_FREESTAGE")) sycl::free(stage, q);
    std::fprintf(stderr, "[probe2] %zu weight buffers uploaded\n", bufs.size());
    const size_t Mp = 896, Bp = 112;
    const size_t scratch[] = {Mp * 588 * 2, Mp * 1024 * 4, Mp * 1024 * 2, Mp * 5632 * 4, Mp * 3072 * 2,
                              Mp * 1024 * 2, Mp * 2816 * 2, Mp * 32 * 4, Mp * 32 * 4, Bp * 9216 * 2, Bp * 4096 * 2, Bp * 4096 * 4};
    size_t k = 0;
    for (size_t bytes : scratch) {
        void* d = alloc.malloc(bytes);
        try { q.memset(d, 0, bytes).wait(); std::fprintf(stderr, "[probe2] scratch %zu (%zu B) memset ok\n", k, bytes); }
        catch (const std::exception& e) { std::fprintf(stderr, "[probe2] scratch %zu (%zu B) THREW %s\n", k, bytes, e.what()); return 1; }
        ++k;
    }
    std::fprintf(stderr, "[probe2] all ok\n");
    return 0;
}

int main() {
    if (const char* pr = std::getenv("IE_DS4_VIS_PROBE")) return std::atoi(pr) == 2 ? probe2() : probe();
    const char* gp = std::getenv("IE_DS4_VIS_GOLDEN");
    if (!gp) { std::fprintf(stderr, "SKIP: IE_DS4_VIS_GOLDEN not set\n"); return 0; }
    const char* sp = std::getenv("IE_DS4_VIS_SIDECAR");
    const std::string sidecar = sp ? sp : std::string(std::getenv("HOME")) +
        "/models/apetersson-DeepSeek-V4-Flash-Vision-Exp-Abliterated/Reference-Native-GGUF/"
        "DeepSeek-V4-Flash-Vision-Exp-Abliterated-Native.safetensors";
    Golden g;
    if (!read_golden(gp, g)) { std::fprintf(stderr, "FAIL: cannot read golden %s\n", gp); return 1; }
    const uint32_t H = g.n_vit_h * 14, W = g.n_vit_w * 14, D = g.hidden;
    if (D != ie::kDs4VisOut) { std::fprintf(stderr, "FAIL: golden hidden %u\n", D); return 1; }

    // ---- image loader vs the golden's pixels (IE_DS4_VIS_IMAGE=<file the golden was made from>)
    bool loader_ok = true;
    if (const char* ip = std::getenv("IE_DS4_VIS_IMAGE")) {
        std::vector<float> px; uint32_t LH = 0, LW = 0; ie::Ds4VisGeom geom;
        if (auto e = ie::ds4_load_image(ip, px, LH, LW, geom); !e.empty()) { std::fprintf(stderr, "loader: %s\n", e.c_str()); return 1; }
        size_t bad = 0; double maxd = 0;
        if (LH != H || LW != W) { loader_ok = false; }
        else for (size_t i = 0; i < px.size(); ++i) {
            const double d = std::fabs(double(px[i]) - double(g.pixels[i]));
            maxd = std::max(maxd, d); if (d > 2.0 / 127.5 + 1e-6) ++bad;   // > 2 u8 LSB
        }
        // PNG: the PIL bicubic/pad emulation is bit-exact. JPEG: stb and libjpeg
        // decode differ by a few LSB on a handful of pixels (measured: max 3.2
        // LSB, ~1e-4 of the pixels on carrots/corn) — allow that, nothing more.
        loader_ok = loader_ok && maxd <= 8.0 / 127.5 && bad <= px.size() / 1000;
        std::printf("loader: %ux%u vs golden %ux%u  n_llm %ux%u  max|d|=%.4f (%.2f LSB)  >2LSB=%zu  %s\n",
                    LW, LH, W, H, geom.n_llm_w, geom.n_llm_h, maxd, maxd * 127.5, bad, loader_ok ? "ok" : "MISMATCH");
    }

    // ---- block layout: exact ----------------------------------------------
    std::vector<int32_t> types, perm;
    ie::ds4_vis_build_block(g.n_llm_h, g.n_llm_w, g.start_pos, types, perm);
    const bool types_ok = types == g.types, perm_ok = perm == g.perm;
    const uint32_t gt = ie::ds4_vis_grid_tokens(g.n_llm_h, g.n_llm_w) + (3 - g.start_pos % 4);
    std::printf("layout: types %s (%zu vs %u) perm %s (%zu) grid_tokens+pad=%u\n",
                types_ok ? "ok" : "MISMATCH", types.size(), g.n_rows, perm_ok ? "ok" : "MISMATCH",
                perm.size(), gt);

    // ---- tower ---------------------------------------------------------------
    // The allocator must outlive the tower: its pinned host buffers are freed
    // on the allocator's queue.
    ie::DeviceAllocator alloc;
    if (auto e = alloc.init("B70", 0); !e.empty()) { std::fprintf(stderr, "gpu: %s\n", e.c_str()); return 1; }
    ie::Ds4Vision vis;
    if (auto e = vis.load(sidecar); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    std::vector<float> a1, a2;
    auto t0 = std::chrono::steady_clock::now();
    if (auto e = vis.encode_gpu(alloc, g.pixels.data(), H, W, a1); !e.empty()) { std::fprintf(stderr, "gpu: %s\n", e.c_str()); return 1; }
    auto t1 = std::chrono::steady_clock::now();   // includes upload
    if (auto e = vis.encode_gpu(alloc, g.pixels.data(), H, W, a2); !e.empty()) { std::fprintf(stderr, "gpu2: %s\n", e.c_str()); return 1; }
    auto t2 = std::chrono::steady_clock::now();   // warm
    if (a1.size() != g.aligned.size()) { std::fprintf(stderr, "FAIL: aligned size %zu vs %zu\n", a1.size(), g.aligned.size()); return 1; }
    bool finite = true;
    for (float v : a2) finite = finite && std::isfinite(v);
    if (const char* dp = std::getenv("IE_DS4_VIS_DUMP")) {   // raw GPU aligner rows for offline analysis
        if (FILE* df = std::fopen(dp, "wb")) { std::fwrite(a2.data(), 4, a2.size(), df); std::fclose(df); }
    }
    const Cmp ca = compare(g.aligned, a2, D);
    const Cmp rr = compare(a1, a2, D);   // run-to-run

    // ---- assembled rows ------------------------------------------------------
    std::vector<float> rows;
    if (auto e = vis.assemble_rows(a2, types, perm, rows); !e.empty()) { std::fprintf(stderr, "rows: %s\n", e.c_str()); return 1; }
    Cmp cr{1, 1, 0, 0, 1};
    if (rows.size() == g.rows.size()) cr = compare(g.rows, rows, D);

    std::printf("aligner vs oracle: rows=%u rel_l2=%.3e max|d|=%.3e mean|d|=%.3e min_row_cos=%.6f p05_row_cos=%.6f finite=%d | "
                "run-to-run rel=%.1e | block rows rel_l2=%.3e min_cos=%.6f | cold %.2fs warm %.3fs\n",
                unsigned(a2.size() / D), ca.rel, ca.max_d, ca.mean_abs, ca.min_cos, ca.p05_cos, int(finite), rr.rel,
                cr.rel, cr.min_cos,
                std::chrono::duration<double>(t1 - t0).count(),
                std::chrono::duration<double>(t2 - t1).count());
    // Bar (docs/deepseek4/70_VISION_EXP_PORT_PLAN.md, P2 criterion 1, with the
    // measured yardstick): the official modules run in bf16 sit at min row-cos
    // 0.57-0.83 and mean|d| 2.2e-3..7.4e-3 vs the fp32 golden; this fp16 path
    // is ~10x closer on every fixture (worst single row: a low-norm row of the
    // white-background fixture, cos 0.9907). Bar: rel_l2 < 2e-2, p05 row-cos
    // > 0.999, min row-cos > 0.99, mean|d| < 1e-3, bit-identical run to run.
    const bool pass = loader_ok && types_ok && perm_ok && finite && ca.rel < 2e-2 && ca.p05_cos > 0.999 &&
                      ca.min_cos > 0.99 && ca.mean_abs < 1e-3 && rr.rel == 0.0 &&
                      rows.size() == g.rows.size() && cr.rel < 2e-2 && cr.min_cos > 0.99;
    std::printf(pass ? "GATE PASSED\n" : "GATE FAILED\n");
    return pass ? 0 : 1;
}
