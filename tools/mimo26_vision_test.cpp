// tools/mimo26_vision_test.cpp — the P6.1 gate for MiMo-V2.6's vision input (docs/mimo26/00_PORT_PLAN.md).
//
//   ie-mimo26-vision-test plan   <golden_dir>   the resize plan vs the reference's smart_resize sweep (sweep.tsv)
//   ie-mimo26-vision-test pixels <golden_dir>   pixel_values of every fixture_k.png vs pixel_values_k.f32
//   ie-mimo26-vision-test tower  <golden_dir> <model_dir>   the GPU tower vs the fp32 goldens (P6.1b)
//
// The goldens come from tools/mimo26/vision_ref.py. Exit 0 only when every check passes.
#include "ie/mimo26_vision.hpp"
#include "ie/allocator.hpp"
#include "ie/safetensors.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::vector<char> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int check_plan(const std::string& dir) {
    std::ifstream f(dir + "/sweep.tsv");
    if (!f) { std::printf("FAIL: no %s/sweep.tsv\n", dir.c_str()); return 1; }
    std::string line;
    std::getline(f, line);   // header
    int rows = 0, bad = 0;
    while (std::getline(f, line)) {
        std::istringstream in(line);
        std::string hs, ws, hb, wb;
        std::getline(in, hs, '\t'); std::getline(in, ws, '\t'); std::getline(in, hb, '\t'); std::getline(in, wb, '\t');
        if (hs.empty()) continue;
        ++rows;
        ie::MimoVisGeom g;
        const std::string err = ie::mimo26_vis_plan(uint32_t(std::stoul(hs)), uint32_t(std::stoul(ws)), g);
        const bool ref_err = hb == "error";
        const bool ok = ref_err ? !err.empty() : err.empty() && g.h == std::stoul(hb) && g.w == std::stoul(wb);
        if (!ok) {
            ++bad;
            std::printf("  mismatch %s x %s: reference %s x %s, engine %u x %u %s\n", hs.c_str(), ws.c_str(), hb.c_str(),
                        wb.c_str(), g.h, g.w, err.c_str());
        }
    }
    std::printf("plan: %d sizes, %d mismatches\n", rows, bad);
    return rows >= 100 && bad == 0 ? 0 : 1;
}

int check_pixels(const std::string& dir) {
    static const float kStd[3] = {0.26862954f, 0.26130258f, 0.27577711f};
    int fails = 0, fixtures = 0;
    for (int k = 0;; ++k) {
        const std::vector<char> png = read_file(dir + "/fixture_" + std::to_string(k) + ".png");
        if (png.empty()) break;
        ++fixtures;
        std::vector<float> pv;
        ie::MimoVisGeom g;
        const std::string err = ie::mimo26_load_image_mem(png.data(), png.size(), pv, g);
        const std::vector<char> ref = read_file(dir + "/pixel_values_" + std::to_string(k) + ".f32");
        if (!err.empty() || ref.size() != pv.size() * sizeof(float)) {
            std::printf("fixture %d: FAIL %s (engine %zu values, reference %zu)\n", k, err.c_str(), pv.size(),
                        ref.size() / sizeof(float));
            ++fails;
            continue;
        }
        const float* r = reinterpret_cast<const float*>(ref.data());
        double worst = 0.0;
        size_t over = 0;
        for (size_t i = 0; i < pv.size(); ++i) {
            const int c = int((i % ie::kMimoVisPatchDim) / (2 * ie::kMimoVisPatch * ie::kMimoVisPatch));
            const double lsb = std::fabs(double(pv[i]) - double(r[i])) * kStd[c] * 255.0;   // in u8 steps
            worst = std::max(worst, lsb);
            over += lsb > 2.0;
        }
        std::printf("fixture %d: grid %u x %u, %u patches, %u tokens; worst %.3f u8 LSB, %zu values beyond 2\n", k,
                    g.grid_h, g.grid_w, g.patches(), g.tokens(), worst, over);
        fails += over != 0;
    }
    std::printf("pixels: %d fixtures, %d failing\n", fixtures, fails);
    return fixtures == 3 && fails == 0 ? 0 : 1;
}

struct Cmp { double rel = 0, min_cos = 1, p05 = 1; size_t rows = 0, b99 = 0, b95 = 0; };

// rel-L2 and the row-cosine distribution of a against r ([rows, cols]); `skip` columns left out of both
Cmp compare(const float* a, const float* r, size_t rows, size_t cols, const std::vector<uint32_t>& skip = {}) {
    Cmp c;
    c.rows = rows;
    std::vector<char> keep(cols, 1);
    for (uint32_t s : skip) if (s < cols) keep[s] = 0;
    double num = 0, den = 0;
    std::vector<double> cs(rows);
    for (size_t i = 0; i < rows; ++i) {
        double dot = 0, na = 0, nr = 0;
        for (size_t j = 0; j < cols; ++j) {
            if (!keep[j]) continue;
            const double x = a[i * cols + j], y = r[i * cols + j];
            dot += x * y; na += x * x; nr += y * y; num += (x - y) * (x - y); den += y * y;
        }
        cs[i] = (na > 0 && nr > 0) ? dot / std::sqrt(na * nr) : (na == nr ? 1.0 : 0.0);
        if (!std::isfinite(cs[i])) cs[i] = -1.0;                // an inf / NaN row fails every bar (it is not < 0.99 as NaN)
        c.b99 += cs[i] < 0.99;
        c.b95 += cs[i] < 0.95;
    }
    c.rel = std::isfinite(den) && std::isfinite(num) ? (den > 0 ? std::sqrt(num / den) : 0.0) : std::numeric_limits<double>::infinity();   // a NaN on either side fails
    std::sort(cs.begin(), cs.end());
    c.min_cos = cs.front();
    const double pos = 0.05 * double(rows - 1);                 // torch.quantile's linear interpolation
    const size_t lo = size_t(pos), hi = std::min(rows - 1, lo + 1);
    c.p05 = cs[lo] + (cs[hi] - cs[lo]) * (pos - double(lo));
    return c;
}

bool bar(const Cmp& c) { return c.rel < 2e-2 && c.p05 > 0.999 && double(c.b99) <= 0.005 * double(c.rows) && c.b95 == 0; }

void show(const char* what, const Cmp& c) {
    std::printf("    %-22s rel-L2 %.3e  min row-cos %.5f  p05 %.6f  rows<0.99 %zu/%zu  rows<0.95 %zu  %s\n", what, c.rel,
                c.min_cos, c.p05, c.b99, c.rows, c.b95, bar(c) ? "ok" : "BELOW THE BAR");
}

double free_mib(sycl::queue& q) {
    try { return double(q.get_device().get_info<sycl::ext::intel::info::device::free_memory>()) / 1048576.0; }
    catch (...) { return -1.0; }
}

double yard_rel(const std::string& json, int k) {               // yardstick.json: {"k": {..., "rel_l2": x, ...}}
    const size_t at = json.find("\"" + std::to_string(k) + "\"");
    const size_t end = at == std::string::npos ? at : json.find('}', at);                    // stay inside this fixture's object
    size_t r = at == std::string::npos ? at : json.find("\"rel_l2\":", at);
    if (r != std::string::npos && end != std::string::npos && r > end) r = std::string::npos;
    return r == std::string::npos ? -1.0 : std::strtod(json.c_str() + r + 9, nullptr);
}

int check_tower(const std::string& dir, const std::string& model_dir) {
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a) { return std::chrono::duration<double, std::milli>(clk::now() - a).count(); };
    ie::SafetensorsModel model;
    if (auto e = model.open(model_dir); !e.empty()) { std::printf("open %s: %s\n", model_dir.c_str(), e.c_str()); return 2; }
    ie::DeviceAllocator alloc;
    const char* card = std::getenv("IE_MIMO26_VIS_CARD");
    if (auto e = alloc.init("B70", card ? uint32_t(std::atoi(card)) : 0u); !e.empty()) { std::printf("device: %s\n", e.c_str()); return 2; }
    ie::MimoVision vis;
    if (auto e = vis.load_from([&](const std::string& n) { return model.find(n); }); !e.empty()) { std::printf("load: %s\n", e.c_str()); return 2; }
    const double free0 = free_mib(alloc.queue());
    auto t0 = clk::now();
    if (auto e = vis.stage_host(alloc); !e.empty()) { std::printf("stage: %s\n", e.c_str()); return 2; }
    std::printf("tower: staged in pinned host memory in %.0f ms; encode block %.0f MiB; device free %.0f MiB\n", ms(t0),
                double(vis.encode_bytes()) / 1048576.0, free0);
    const std::vector<char> yj = read_file(dir + "/yardstick.json");
    const std::string yard(yj.begin(), yj.end());
    const std::vector<uint32_t> blocks = {0, 5, 9, 27}, dominant = {102, 572, 1031, 1040};
    int fails = 0;
    for (int k = 0; k < 3; ++k) {
        const std::vector<char> png = read_file(dir + "/fixture_" + std::to_string(k) + ".png");
        std::vector<float> pv, m1, m2;
        ie::MimoVisGeom g;
        if (auto e = ie::mimo26_load_image_mem(png.data(), png.size(), pv, g); !e.empty()) { std::printf("fixture %d: %s\n", k, e.c_str()); return 2; }
        std::vector<std::vector<float>> caps;
        t0 = clk::now();
        if (auto e = vis.encode_gpu(alloc, pv, g, m1, blocks, &caps); !e.empty()) { std::printf("encode: %s\n", e.c_str()); return 2; }
        const double t_first = ms(t0), free1 = free_mib(alloc.queue());
        t0 = clk::now();
        if (auto e = vis.encode_gpu(alloc, pv, g, m2); !e.empty()) { std::printf("encode: %s\n", e.c_str()); return 2; }
        const double t_second = ms(t0), free2 = free_mib(alloc.queue());
        const bool same = m1.size() == m2.size() && std::memcmp(m1.data(), m2.data(), m1.size() * 4) == 0;
        std::printf("fixture %d: %u patches -> %u tokens; encode %.0f ms then %.0f ms; run to run %s; device free %.0f -> %.0f MiB\n",
                    k, g.patches(), g.tokens(), t_first, t_second, same ? "bit-identical" : "DIFFERENT", free1, free2);
        fails += !same;
        fails += free2 < free0 - 256.0;                     // (e) the block is returned (one-time runtime state allowed)
        const std::vector<char> ref = read_file(dir + "/merged_" + std::to_string(k) + ".f32");
        if (ref.size() != m1.size() * 4) { std::printf("  merged: golden size mismatch\n"); ++fails; continue; }
        const Cmp c = compare(m1.data(), reinterpret_cast<const float*>(ref.data()), g.tokens(), 4096);
        show("merged (binding)", c);
        fails += !bar(c);
        const double y = yard_rel(yard, k);
        std::printf("    the module's own BF16: rel-L2 %.3e -> engine %s it%s\n", y, c.rel < y ? "closer than" : "NOT closer than",
                    k == 2 ? " (fixture 2: not binding)" : "");
        if (k != 2) fails += !(c.rel < y);
        for (size_t i = 0; i < blocks.size() && i < caps.size(); ++i) {
            const std::vector<char> rb = read_file(dir + "/block" + std::to_string(blocks[i]) + "_" + std::to_string(k) + ".f32");
            if (rb.size() != caps[i].size() * 4) { std::printf("  block %u: golden size mismatch\n", blocks[i]); ++fails; continue; }
            const auto* rf = reinterpret_cast<const float*>(rb.data());
            const std::string a = "block " + std::to_string(blocks[i]) + " (binding)", b = "block " + std::to_string(blocks[i]) + " raw";
            const Cmp ex = compare(caps[i].data(), rf, g.patches(), 1280, dominant);
            show(a.c_str(), ex);
            show(b.c_str(), compare(caps[i].data(), rf, g.patches(), 1280));
            fails += !bar(ex);
        }
    }
    {   // (f) encode time at the 1,024-token budget (recorded)
        const std::vector<char> png = read_file(dir + "/fixture_1.png");
        std::vector<float> pv, m;
        ie::MimoVisGeom g;
        ie::mimo26_load_image_mem(png.data(), png.size(), pv, g, uint64_t(1024) * 4 * 256);
        double best = 1e30;
        for (int r = 0; r < 3; ++r) {
            t0 = clk::now();
            if (auto e = vis.encode_gpu(alloc, pv, g, m); !e.empty()) { std::printf("budget encode: %s\n", e.c_str()); return 2; }
            best = std::min(best, ms(t0));
        }
        std::printf("budget: fixture 1 at <= 1,024 tokens -> %u x %u patches, %u tokens: best of 3 encodes %.0f ms\n",
                    g.grid_h, g.grid_w, g.tokens(), best);
    }
    std::printf("tower: %s\n", fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: ie-mimo26-vision-test plan|pixels <golden_dir> | tower <golden_dir> <model_dir>\n");
        return 2;
    }
    const std::string what = argv[1], dir = argv[2];
    if (what == "plan") return check_plan(dir);
    if (what == "pixels") return check_pixels(dir);
    if (what == "tower") return argc > 3 ? check_tower(dir, argv[3]) : 2;
    std::printf("unknown check %s\n", what.c_str());
    return 2;
}
