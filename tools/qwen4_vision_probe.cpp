// tools/qwen4_vision_probe.cpp — standalone Qwen3.8 vision encode probe (CPU path).
// Usage: ie-qwen4-vision-probe <mmproj.gguf> <image> [--dump out.bin] [--gpu]
// Prints grid/token counts and embedding stats; --dump writes [Nm, out_d] f32
// (out_d = the mmproj's projector width: 2560 Flash-Next, 5120 the 27B, 2048 the 35B).
// --gpu (2026-10-21): also encode on the first B70 (encode_gpu, what the servers run) and compare it with the CPU path:
// relative L2, the lowest per-row cosine and the row it is on, the largest |difference| (qwen4_vision_gpu_test's measures,
// on any picture size).

#include "ie/allocator.hpp"
#include "ie/qwen4_vision.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <mmproj.gguf> <image> [--dump out.bin]\n", argv[0]); return 2; }
    const char* dump = nullptr;
    bool gpu = false;
    for (int i = 3; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[i + 1];
        if (!std::strcmp(argv[i], "--gpu")) gpu = true;
    }

    ie::Qwen4Vision vis;
    auto t0 = std::chrono::steady_clock::now();
    if (auto e = vis.load(argv[1]); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    auto t1 = std::chrono::steady_clock::now();

    std::vector<float> img;
    uint32_t H = 0, W = 0;
    if (auto e = ie::qwen4_load_image(argv[2], img, H, W); !e.empty()) {
        std::fprintf(stderr, "image: %s\n", e.c_str()); return 1;
    }
    std::printf("[img] resized to %ux%u  grid %ux%u  merged tokens %u\n",
                W, H, H / 16, W / 16, (H / 32) * (W / 32));
    auto t2 = std::chrono::steady_clock::now();
    std::vector<float> emb;
    if (auto e = vis.encode(img.data(), H, W, emb); !e.empty()) {
        std::fprintf(stderr, "encode: %s\n", e.c_str()); return 1;
    }
    auto t3 = std::chrono::steady_clock::now();

    const size_t nm = emb.size() / vis.out_d();
    double mu = 0, sq = 0;
    for (float v : emb) mu += v;
    mu /= double(emb.size());
    for (float v : emb) sq += (v - mu) * (v - mu);
    std::printf("[emb] [%zu, %u]  mean=%.6f std=%.6f  first4: %.4f %.4f %.4f %.4f\n",
                nm, vis.out_d(), mu, std::sqrt(sq / double(emb.size())),
                emb[0], emb[1], emb[2], emb[3]);
    std::printf("[time] load %.2fs  encode %.2fs\n",
                std::chrono::duration<double>(t1 - t0).count(),
                std::chrono::duration<double>(t3 - t2).count());
    if (gpu) {
        ie::DeviceAllocator alloc;
        if (auto e = alloc.init("B70", 0); !e.empty()) { std::fprintf(stderr, "gpu: %s\n", e.c_str()); return 1; }
        std::vector<float> g;
        if (auto e = vis.encode_gpu(alloc, img.data(), H, W, g); !e.empty()) { std::fprintf(stderr, "encode_gpu: %s\n", e.c_str()); return 1; }
        if (g.size() != emb.size()) { std::fprintf(stderr, "encode_gpu: %zu values, the CPU path %zu\n", g.size(), emb.size()); return 1; }
        const uint32_t D = vis.out_d();
        double num = 0, den = 0, max_d = 0, min_cos = 2;
        size_t min_row = 0;
        for (size_t r = 0; r < nm; ++r) {
            double dot = 0, na = 0, nb = 0;
            for (uint32_t j = 0; j < D; ++j) {
                const double a = emb[r * D + j], b = g[r * D + j];
                num += (a - b) * (a - b); den += a * a; max_d = std::max(max_d, std::fabs(a - b));
                dot += a * b; na += a * a; nb += b * b;
            }
            const double c = dot / std::sqrt(na * nb + 1e-30);
            if (c < min_cos) { min_cos = c; min_row = r; }
        }
        std::printf("[gpu] rel L2 %.3e  min row cosine %.6f (row %zu of %zu)  max |d| %.4f\n", std::sqrt(num / den), min_cos,
                    min_row, nm, max_d);
    }
    if (dump) {
        FILE* f = std::fopen(dump, "wb");
        if (!f) { std::fprintf(stderr, "cannot write %s\n", dump); return 1; }
        std::fwrite(emb.data(), 4, emb.size(), f);
        std::fclose(f);
        std::printf("[dump] %s\n", dump);
    }
    return 0;
}
