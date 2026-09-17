// tests/unit/qwen4_vision_gpu_test.cpp — P3 gate (docs/qwen4/16_vision_port.md):
// GPU vision tower vs the CPU reference path (itself numpy-oracle-gated) on the
// golden input image. fp16 GEMM/attention accumulation drifts vs the fp32/f64
// CPU math, so the gate is relative-L2 + per-row cosine, not bitwise.
// Env: IE_MMPROJ (default ~/models/.../mmproj-F16.gguf), IE_VIS_GOLDEN (.bin).

#include "ie/allocator.hpp"
#include "ie/qwen4_vision.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
    const char* mm = std::getenv("IE_MMPROJ");
    std::string mmproj = mm ? mm :
        std::string(std::getenv("HOME")) + "/models/Qwen3.8-Flash-Next-GGUF/mmproj-F16.gguf";
    const char* gp = std::getenv("IE_VIS_GOLDEN");
    if (!gp) { std::fprintf(stderr, "SKIP: IE_VIS_GOLDEN not set\n"); return 0; }
    FILE* f = std::fopen(gp, "rb");
    if (!f) { std::fprintf(stderr, "FAIL: cannot open %s\n", gp); return 1; }
    uint32_t hdr[3];
    if (std::fread(hdr, 4, 3, f) != 3) return 1;
    const uint32_t H = hdr[0], W = hdr[1];
    std::vector<float> img(size_t(3) * H * W);
    if (std::fread(img.data(), 4, img.size(), f) != img.size()) return 1;
    std::fclose(f);

    ie::Qwen4Vision vis;
    if (auto e = vis.load(mmproj); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    ie::DeviceAllocator alloc;
    if (auto e = alloc.init("B70", 0); !e.empty()) { std::fprintf(stderr, "gpu: %s\n", e.c_str()); return 1; }

    std::vector<float> cpu, gpu;
    if (auto e = vis.encode(img.data(), H, W, cpu); !e.empty()) { std::fprintf(stderr, "cpu: %s\n", e.c_str()); return 1; }
    auto t0 = std::chrono::steady_clock::now();
    if (auto e = vis.encode_gpu(alloc, img.data(), H, W, gpu); !e.empty()) { std::fprintf(stderr, "gpu: %s\n", e.c_str()); return 1; }
    auto t1 = std::chrono::steady_clock::now();   // includes weight upload
    std::vector<float> gpu2;
    if (auto e = vis.encode_gpu(alloc, img.data(), H, W, gpu2); !e.empty()) { std::fprintf(stderr, "gpu2: %s\n", e.c_str()); return 1; }
    auto t2 = std::chrono::steady_clock::now();   // warm

    if (gpu.size() != cpu.size()) { std::fprintf(stderr, "FAIL: size\n"); return 1; }
    double sum_sq = 0, ref_sq = 0, max_d = 0, min_cos = 1.0;
    const uint32_t D = ie::kVisOutD;
    for (size_t r = 0; r < cpu.size() / D; ++r) {
        double dot = 0, na = 0, nb = 0;
        for (uint32_t j = 0; j < D; ++j) {
            const double a = cpu[r * D + j], b = gpu[r * D + j], d = a - b;
            sum_sq += d * d; ref_sq += a * a;
            max_d = std::max(max_d, std::fabs(d));
            dot += a * b; na += a * a; nb += b * b;
        }
        min_cos = std::min(min_cos, dot / std::sqrt(na * nb));
    }
    const double rel = std::sqrt(sum_sq / ref_sq);
    std::printf("gpu vs cpu: rel_l2=%.3e max|d|=%.3e min_row_cos=%.6f  "
                "cold %.2fs warm %.3fs\n",
                rel, max_d, min_cos,
                std::chrono::duration<double>(t1 - t0).count(),
                std::chrono::duration<double>(t2 - t1).count());
    if (rel > 2e-2 || min_cos < 0.999) { std::printf("GATE FAILED\n"); return 1; }
    std::printf("GATE PASSED\n");
    return 0;
}
