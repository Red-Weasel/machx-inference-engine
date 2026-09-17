// tools/qwen4_vision_probe.cpp — standalone qwen4exp vision encode probe.
// Usage: ie-qwen4-vision-probe <mmproj.gguf> <image> [--dump out.bin]
// Prints grid/token counts and embedding stats; --dump writes [Nm, 2560] f32.

#include "ie/qwen4_vision.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s <mmproj.gguf> <image> [--dump out.bin]\n", argv[0]); return 2; }
    const char* dump = nullptr;
    for (int i = 3; i + 1 < argc; ++i)
        if (!std::strcmp(argv[i], "--dump")) dump = argv[i + 1];

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

    const size_t nm = emb.size() / ie::kVisOutD;
    double mu = 0, sq = 0;
    for (float v : emb) mu += v;
    mu /= double(emb.size());
    for (float v : emb) sq += (v - mu) * (v - mu);
    std::printf("[emb] [%zu, %u]  mean=%.6f std=%.6f  first4: %.4f %.4f %.4f %.4f\n",
                nm, ie::kVisOutD, mu, std::sqrt(sq / double(emb.size())),
                emb[0], emb[1], emb[2], emb[3]);
    std::printf("[time] load %.2fs  encode %.2fs\n",
                std::chrono::duration<double>(t1 - t0).count(),
                std::chrono::duration<double>(t3 - t2).count());
    if (dump) {
        FILE* f = std::fopen(dump, "wb");
        if (!f) { std::fprintf(stderr, "cannot write %s\n", dump); return 1; }
        std::fwrite(emb.data(), 4, emb.size(), f);
        std::fclose(f);
        std::printf("[dump] %s\n", dump);
    }
    return 0;
}
