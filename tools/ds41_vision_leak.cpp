// tools/ds41_vision_leak.cpp — host-memory growth probe for the V4.1 vision path (docs/deepseek41/99).
// Runs the serving path's per-image work (ds41_load_image_mem -> Ds4Vision::encode_gpu -> assemble_rows) on the
// given images again and again and prints the process's RssAnon after every pass, so a per-image leak shows as a
// steady climb. Loads only the tower, never the language model.
//   usage: ie-ds41-vision-leak <model_dir> <passes> <image> [image...]
#include "ie/allocator.hpp"
#include "ie/ds41_vision.hpp"
#include "ie/safetensors.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {
long rss_anon_kb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("RssAnon:", 0) == 0) return std::strtol(line.c_str() + 8, nullptr, 10);
    return -1;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) { std::printf("usage: %s <model_dir> <passes> <image> [image...]\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    const int passes = std::atoi(argv[2]);
    std::vector<std::vector<uint8_t>> images;
    for (int i = 3; i < argc; ++i) {
        std::ifstream f(argv[i], std::ios::binary);
        images.emplace_back((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (images.back().empty()) { std::printf("cannot read %s\n", argv[i]); return 2; }
    }
    ie::SafetensorsModel model;
    if (auto e = model.open(dir); !e.empty()) { std::printf("open: %s\n", e.c_str()); return 2; }
    ie::DeviceAllocator alloc;
    if (auto e = alloc.init("B70", 0); !e.empty()) { std::printf("device: %s\n", e.c_str()); return 2; }
    ie::Ds4Vision vis;
    if (auto e = vis.load_from([&](const std::string& n) { return model.find(n); }, ie::ds41_vision_options(5120)); !e.empty()) { std::printf("load: %s\n", e.c_str()); return 2; }
    if (auto e = vis.stage_host(alloc); !e.empty()) { std::printf("stage: %s\n", e.c_str()); return 2; }
    std::printf("staged; RssAnon %ld MiB\n", rss_anon_kb() / 1024);
    const long start = rss_anon_kb();
    for (int k = 0; k < passes; ++k) {
        for (const auto& bytes : images) {
            std::vector<float> px, aligned, rows; uint32_t H = 0, W = 0; ie::Ds4VisGeom g;
            if (auto e = ie::ds41_load_image_mem(bytes.data(), bytes.size(), px, H, W, g); !e.empty()) { std::printf("load image: %s\n", e.c_str()); return 1; }
            if (auto e = vis.encode_gpu(alloc, px.data(), H, W, aligned); !e.empty()) { std::printf("encode: %s\n", e.c_str()); return 1; }
            std::vector<int32_t> types, perm; ie::ds41_vis_types(g.n_llm_h, g.n_llm_w, types, perm);
            if (auto e = vis.assemble_rows(aligned, types, perm, rows); !e.empty()) { std::printf("assemble: %s\n", e.c_str()); return 1; }
        }
        std::printf("pass %3d: RssAnon %ld MiB (%+ld MiB since staging)\n", k + 1, rss_anon_kb() / 1024, (rss_anon_kb() - start) / 1024);
        std::fflush(stdout);
    }
    return 0;
}
