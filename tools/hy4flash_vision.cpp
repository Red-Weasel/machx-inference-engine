// tools/hy4flash_vision.cpp — ie-hy4flash-vision: the engine's Hy4-flash image tower against the reference's
// (scripts/hy4flash_oracle/vision_oracle.py writes the oracle directory).
//
//   ie-hy4flash-vision <ie_dir> --oracle <dir> [--max-px N]
// Runs the tower on the oracle's own pixel_values and compares the hidden state after the captured blocks and the
// projector's output; then loads <dir>/image.png through the engine's loader and compares its pixel_values and the
// tower's output on them (the loader's resize is PIL's bicubic, as the reference's PIL processor).
#include "ie/hy4flash.hpp"
#include "ie/hy4flash_vision.hpp"
#include "nlohmann/json.hpp"

#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace {
bool read_f32(const std::string& p, std::vector<float>& v) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const auto n = size_t(f.tellg()); v.resize(n / 4); f.seekg(0);
    return bool(f.read(reinterpret_cast<char*>(v.data()), std::streamsize(v.size() * 4)));
}
// relative error: rms(a - b) / rms(b), and the largest absolute difference
void cmp(const char* what, const std::vector<float>& a, const std::vector<float>& b, double& worst) {
    if (a.size() != b.size()) { std::printf("%-34s SIZE MISMATCH (%zu vs %zu)\n", what, a.size(), b.size()); worst = 1e9; return; }
    double d2 = 0, r2 = 0, mx = 0;
    for (size_t i = 0; i < a.size(); ++i) { const double d = double(a[i]) - b[i]; d2 += d * d; r2 += double(b[i]) * b[i]; mx = std::max(mx, std::fabs(d)); }
    const double rel = std::sqrt(d2 / std::max(r2, 1e-30));
    std::printf("%-34s relative error %.3e, largest difference %.4f (reference rms %.4f)\n", what, rel, mx, std::sqrt(r2 / double(b.size())));
    worst = std::max(worst, rel);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: ie-hy4flash-vision <ie_dir> --oracle <dir> [--max-px N]\n"); return 2; }
    const std::string dir = argv[1]; std::string oracle; uint64_t max_px = 1024ull * 1024ull;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--oracle" && i + 1 < argc) oracle = argv[++i];
        else if (a == "--max-px" && i + 1 < argc) max_px = std::stoull(argv[++i]);
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    ie::Hy4FlashModel m;
    if (auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    sycl::device dev; bool found = false;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (!found && d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) { dev = d; found = true; }
    }
    if (!found) { std::fprintf(stderr, "no Arc GPU\n"); return 1; }
    sycl::queue q(sycl::context(dev), dev, sycl::property::queue::in_order{});
    ie::Hy4fVision vis;
    auto t0 = std::chrono::steady_clock::now();
    if (auto e = vis.init(q, m, max_px); !e.empty()) { std::fprintf(stderr, "init: %s\n", e.c_str()); return 1; }
    std::printf("tower loaded in %.1f s: %.2f GiB on the card (weights + scratch for %llu pixels)\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                double(vis.device_bytes()) / 1073741824.0, static_cast<unsigned long long>(vis.max_px()));
    std::ifstream mf(oracle + "/meta.json");
    if (!mf) { std::fprintf(stderr, "no %s/meta.json\n", oracle.c_str()); return 1; }
    const auto meta = nlohmann::json::parse(mf);
    ie::Hy4fVisGeom g; g.grid_h = meta["grid_h"].get<uint32_t>(); g.grid_w = meta["grid_w"].get<uint32_t>(); g.h = g.grid_h * 16; g.w = g.grid_w * 16;
    std::vector<float> pv, merged, ref;
    if (!read_f32(oracle + "/pixel_values.f32", pv)) { std::fprintf(stderr, "no pixel_values.f32\n"); return 1; }
    std::vector<uint32_t> capture; for (const auto& b : meta["blocks"]) capture.push_back(b.get<uint32_t>());
    std::vector<std::vector<float>> captured;
    double worst = 0;
    t0 = std::chrono::steady_clock::now();
    if (auto e = vis.encode(pv, g, merged, capture, &captured); !e.empty()) { std::fprintf(stderr, "encode: %s\n", e.c_str()); return 1; }
    std::printf("grid %u x %u: %u patches -> %u tokens, encoded in %.0f ms (with the checkpoints' read-backs)\n", g.grid_h, g.grid_w, g.patches(), g.tokens(),
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    for (size_t i = 0; i < capture.size() && i < captured.size(); ++i) {
        char nm[64]; std::snprintf(nm, sizeof nm, "/block_%02u.f32", capture[i]);
        if (read_f32(oracle + nm, ref)) { char w[64]; std::snprintf(w, sizeof w, "after block %u", capture[i]); cmp(w, captured[i], ref, worst); }
    }
    if (!read_f32(oracle + "/merged.f32", ref)) { std::fprintf(stderr, "no merged.f32\n"); return 1; }
    cmp("projector output (the text rows)", merged, ref, worst);
    // the engine's own loader on the oracle's image
    {
        std::ifstream f(oracle + "/image.png", std::ios::binary);
        if (f) {
            const std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            std::vector<float> pv2, merged2; ie::Hy4fVisGeom g2;
            if (auto e = ie::hy4f_load_image_mem(bytes.data(), bytes.size(), pv2, g2, max_px); !e.empty()) { std::fprintf(stderr, "loader: %s\n", e.c_str()); return 1; }
            std::printf("loader: %u x %u canvas, grid %u x %u\n", g2.w, g2.h, g2.grid_h, g2.grid_w);
            double wl = 0; cmp("loader pixel_values", pv2, pv, wl);
            t0 = std::chrono::steady_clock::now();
            if (auto e = vis.encode(pv2, g2, merged2); !e.empty()) { std::fprintf(stderr, "encode: %s\n", e.c_str()); return 1; }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            cmp("projector output, engine's loader", merged2, ref, worst);
            std::printf("one image end to end on the card: %.0f ms\n", ms);
        }
    }
    const bool pass = worst < 2e-2;
    std::printf("VISION GATE: %s (worst relative error %.3e, tolerance 2e-2)\n", pass ? "PASS" : "FAIL", worst);
    return pass ? 0 : 1;
}
