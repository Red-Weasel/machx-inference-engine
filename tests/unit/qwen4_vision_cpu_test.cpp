// tests/unit/qwen4_vision_cpu_test.cpp — P1 gate (docs/qwen4/16_vision_port.md):
// CPU vision tower output must match the numpy golden oracle on the same GGUF
// and synthetic image. Regenerate golden with:
//   python3 tests/oracle/qwen4_vision_oracle.py --out <golden.bin>
// Env: IE_MMPROJ (default ~/models/.../mmproj-F16.gguf), IE_VIS_GOLDEN (.bin).

#include "ie/qwen4_vision.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    if (std::fread(hdr, 4, 3, f) != 3) { std::fprintf(stderr, "FAIL: short golden\n"); return 1; }
    const uint32_t H = hdr[0], W = hdr[1], Nm = hdr[2];
    std::vector<float> img(size_t(3) * H * W), gold(size_t(Nm) * ie::kVisOutD);
    if (std::fread(img.data(), 4, img.size(), f) != img.size() ||
        std::fread(gold.data(), 4, gold.size(), f) != gold.size()) {
        std::fprintf(stderr, "FAIL: truncated golden\n"); return 1;
    }
    std::fclose(f);

    ie::Qwen4Vision vis;
    if (auto e = vis.load(mmproj); !e.empty()) {
        std::fprintf(stderr, "FAIL: load: %s\n", e.c_str()); return 1;
    }
    std::vector<float> out;
    if (auto e = vis.encode(img.data(), H, W, out); !e.empty()) {
        std::fprintf(stderr, "FAIL: encode: %s\n", e.c_str()); return 1;
    }
    if (out.size() != gold.size()) {
        std::fprintf(stderr, "FAIL: shape %zu vs golden %zu\n", out.size(), gold.size());
        return 1;
    }
    double max_abs = 0, sum_sq = 0, ref_sq = 0;
    size_t argmax = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        const double d = double(out[i]) - gold[i];
        if (std::fabs(d) > max_abs) { max_abs = std::fabs(d); argmax = i; }
        sum_sq += d * d;
        ref_sq += double(gold[i]) * gold[i];
    }
    const double rel = std::sqrt(sum_sq / (ref_sq > 0 ? ref_sq : 1.0));
    std::printf("vision cpu vs oracle: max|d|=%.3e (at %zu: %f vs %f)  rel_l2=%.3e  [%u x %u -> %u tok]\n",
                max_abs, argmax, out[argmax], gold[argmax], rel, W, H, Nm);
    if (max_abs > 1e-3 || rel > 1e-4) { std::printf("GATE FAILED\n"); return 1; }
    std::printf("GATE PASSED\n");
    return 0;
}
