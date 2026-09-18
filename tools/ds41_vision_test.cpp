// tools/ds41_vision_test.cpp — V4.1 Phase 55 gate: the vision tower against the checkpoint's own vision.py +
// image_processor.py (tools/ds41_reference/golden_vision.py, fp32 on the shipped weights). docs/deepseek41/94.
//   1. the image plan: every (w, h) of geom.txt, exact
//   2. the loader (decode + PIL contain / grey pad / normalise) against the reference's pixel tensor
//   3. the tower + aligner on the REFERENCE pixels (so 3 does not inherit 2): rel-L2, per-row cosine
//   4. the span layout (types) and the assembled rows (delimiters = the checkpoint's learned rows)
//   5. bit-identical run to run; the transient device block is returned after every encode
//   6. Phase 56: the engram hash with image positions (negative ids) against NgramHashState run with a token mask
//   usage: ie-ds41-vision-test <model_dir> <golden_dir>
#include "ie/allocator.hpp"
#include "ie/deepseek41_engram.hpp"
#include "ie/ds41_vision.hpp"
#include "ie/safetensors.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace {
int g_fail = 0, g_n = 0;
void check(bool ok, const char* what) { ++g_n; if (!ok) ++g_fail; std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what); }

struct Golden { uint32_t n_vit_h, n_vit_w, n_llm_h, n_llm_w, n_types, hidden, n_patches; std::vector<float> pixels, rows; std::vector<int32_t> types; };
bool read_golden(const std::string& path, Golden& g) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[8]; uint32_t h[8];
    bool ok = std::fread(magic, 1, 8, f) == 8 && !std::memcmp(magic, "DS41VOR1", 8) && std::fread(h, 4, 8, f) == 8;
    if (ok) {
        g.n_vit_h = h[0]; g.n_vit_w = h[1]; g.n_llm_h = h[2]; g.n_llm_w = h[3]; g.n_types = h[4]; g.hidden = h[5]; g.n_patches = h[6];
        g.pixels.resize(size_t(3) * g.n_vit_h * 14 * g.n_vit_w * 14); g.rows.resize(size_t(g.n_llm_h) * g.n_llm_w * g.hidden); g.types.resize(g.n_types);
        ok = std::fread(g.pixels.data(), 4, g.pixels.size(), f) == g.pixels.size() && std::fread(g.rows.data(), 4, g.rows.size(), f) == g.rows.size() &&
             std::fread(g.types.data(), 4, g.types.size(), f) == g.types.size();
    }
    std::fclose(f);
    return ok;
}

struct Cmp { double rel, min_cos, p05_cos, mean_abs, ref_abs; size_t below99; double below99_norm; };
Cmp compare(const std::vector<float>& a, const std::vector<float>& ref, uint32_t D) {
    double sq = 0, rsq = 0, ma = 0, ra = 0, low_norm = 0, all_norm = 0; size_t low = 0; std::vector<double> cs;
    for (size_t r = 0; r * D < ref.size(); ++r) {
        double dot = 0, na = 0, nb = 0;
        for (uint32_t d = 0; d < D; ++d) {
            const double x = a[r * D + d], y = ref[r * D + d];
            dot += x * y; na += x * x; nb += y * y; sq += (x - y) * (x - y); rsq += y * y; ma += std::fabs(x - y); ra += std::fabs(y);
        }
        cs.push_back(dot / std::sqrt(na * nb));
        all_norm += std::sqrt(nb);
        if (cs.back() < 0.99) { ++low; low_norm += std::sqrt(nb); }
    }
    const double mean_norm = all_norm / double(cs.size());
    std::sort(cs.begin(), cs.end());
    return {std::sqrt(sq / rsq), cs.front(), cs[cs.size() / 20], ma / double(ref.size()), ra / double(ref.size()), low,
            low ? low_norm / double(low) / mean_norm : 0.0};
}

double free_mib(sycl::queue& q) {
    try { return double(q.get_device().get_info<sycl::ext::intel::info::device::free_memory>()) / 1048576.0; }
    catch (...) { return -1.0; }
}
}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : std::string(std::getenv("HOME")) + "/models/DeepSeek-V4.1-Flash";
    const std::string gd  = argc > 2 ? argv[2] : std::string(std::getenv("HOME")) + "/ds41_work/vision_golden";
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a) { return std::chrono::duration<double, std::milli>(clk::now() - a).count(); };

    // ---- 1. the plan ------------------------------------------------------------------------------------------
    {
        std::ifstream f(gd + "/geom.txt"); std::string line; uint32_t n = 0, bad = 0, max_patches = 0;
        while (std::getline(f, line)) {
            std::istringstream ss(line); uint32_t w, h, lh, lw, bh, bw;
            if (!(ss >> w >> h >> lh >> lw >> bh >> bw)) continue;
            const ie::Ds4VisGeom g = ie::ds41_vis_plan(w, h);
            ++n; max_patches = std::max(max_patches, g.n_vit_h * g.n_vit_w);
            if (g.n_llm_h != lh || g.n_llm_w != lw || g.best_h != bh || g.best_w != bw) {
                if (++bad <= 5) std::printf("    plan(%u, %u): engine llm %ux%u canvas %ux%u, reference llm %ux%u canvas %ux%u\n", w, h, g.n_llm_h, g.n_llm_w, g.best_h, g.best_w, lh, lw, bh, bw);
            }
        }
        std::printf("plan: %u sizes, %u mismatches, largest grid %u patches (cap %u)\n", n, bad, max_patches, ie::kDs41VisMaxPatches);
        check(n > 100 && bad == 0, "the image plan matches plan_image_grid on every size");
        check(max_patches <= ie::kDs41VisMaxPatches, "every planned grid fits the tower's scratch");
    }

    // ---- 6. the engram hash with dead tokens (tools/ds41_reference/golden_engram_mask.py) -------------------------
    {
        FILE* f = std::fopen((gd + "/engram_mask.bin").c_str(), "rb");
        char magic[8]; uint32_t hd[3] = {0, 0, 0};
        bool ok = f && std::fread(magic, 1, 8, f) == 8 && !std::memcmp(magic, "DS41EMK1", 8) && std::fread(hd, 4, 3, f) == 3;
        const uint32_t L = hd[0], NL = hd[1], NC = hd[2];
        std::vector<int32_t> ids(L); std::vector<int64_t> ref(size_t(NL) * L * NC);
        ok = ok && std::fread(ids.data(), 4, L, f) == L && std::fread(ref.data(), 8, ref.size(), f) == ref.size();
        if (f) std::fclose(f);
        ie::Ds41EngramTables tb;
        const std::string te = ok ? tb.load(dir) : std::string("no engram_mask.bin");
        size_t bad = 0, n_img = 0;
        if (te.empty() && ok && NC == tb.n_hash_cols && NL == tb.layer_ids.size()) {
            for (int32_t id : ids) n_img += id < 0;
            for (uint32_t li = 0; li < NL; ++li) {
                std::vector<int64_t> out(size_t(L) * NC);
                ie::ds41_engram_hash(tb, ids.data(), L, li, out.data());
                for (size_t i = 0; i < out.size(); ++i) bad += out[i] != ref[size_t(li) * L * NC + i];
            }
            std::printf("engram hash with image positions: %u tokens (%zu image), %u layers x %u columns, %zu mismatches\n", L, n_img, NL, NC, bad);
        } else std::printf("engram hash golden: %s\n", te.empty() ? "malformed or another layout" : te.c_str());
        check(ok && te.empty() && n_img > 0 && bad == 0, "engram: a negative id is a DEAD token -- every hash equals NgramHashState's under its token mask");
    }

    // ---- the tower ------------------------------------------------------------------------------------------------
    ie::SafetensorsModel model;
    if (auto e = model.open(dir); !e.empty()) { std::printf("open %s: %s\n", dir.c_str(), e.c_str()); return 2; }
    ie::DeviceAllocator alloc;
    if (auto e = alloc.init("B70", std::getenv("IE_DS41_VIS_CARD") ? uint32_t(std::atoi(std::getenv("IE_DS41_VIS_CARD"))) : 0); !e.empty()) { std::printf("device: %s\n", e.c_str()); return 2; }
    ie::Ds4Vision vis;
    const uint32_t HID = 5120;
    if (auto e = vis.load_from([&](const std::string& n) { return model.find(n); }, ie::ds41_vision_options(HID)); !e.empty()) { std::printf("load: %s\n", e.c_str()); return 2; }
    const double free0 = free_mib(alloc.queue());
    auto t0 = clk::now();
    if (auto e = vis.stage_host(alloc); !e.empty()) { std::printf("stage: %s\n", e.c_str()); return 2; }
    std::printf("tower: staged in pinned host memory in %.0f ms; encode block %.0f MiB; device free %.0f MiB\n", ms(t0), double(vis.encode_bytes()) / 1048576.0, free0);

    double free_steady = -1.0;
    for (int k = 0; k < 3; ++k) {
        Golden g;
        if (!read_golden(gd + "/golden_" + std::to_string(k) + ".bin", g)) { std::printf("golden_%d.bin: missing or malformed\n", k); ++g_fail; continue; }
        std::printf("fixture %d: vit %ux%u (%u patches), llm %ux%u, %u tokens\n", k, g.n_vit_h, g.n_vit_w, g.n_patches, g.n_llm_h, g.n_llm_w, g.n_types);
        // 2. the loader
        std::ifstream f(gd + "/img_" + std::to_string(k) + ".png", std::ios::binary);
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        std::vector<float> px; uint32_t H = 0, W = 0; ie::Ds4VisGeom geom;
        const std::string le = ie::ds41_load_image_mem(bytes.data(), bytes.size(), px, H, W, geom);
        bool geom_ok = le.empty() && geom.n_vit_h == g.n_vit_h && geom.n_vit_w == g.n_vit_w && geom.n_llm_h == g.n_llm_h && geom.n_llm_w == g.n_llm_w && px.size() == g.pixels.size();
        check(geom_ok, "loader: the canvas and both grids match the reference");
        if (geom_ok) {
            size_t bad = 0; double maxd = 0, sum = 0;
            for (size_t i = 0; i < px.size(); ++i) { const double d = std::fabs(double(px[i]) - g.pixels[i]); maxd = std::max(maxd, d); sum += d; if (d > 2.0 / 127.5 + 1e-6) ++bad; }
            std::printf("    pixels: max|d| %.4f (1 LSB = %.4f), mean|d| %.2e, %zu of %zu beyond 2 LSB\n", maxd, 1.0 / 127.5, sum / double(px.size()), bad, px.size());
            check(double(bad) <= 1e-3 * double(px.size()) && sum / double(px.size()) < 1e-3, "loader: pixels within 2 u8 LSB of PIL on all but <= 0.1 % of samples");
        }
        // 3. the tower on the reference pixels
        std::vector<float> a1, a2;
        t0 = clk::now();
        std::string e = vis.encode_gpu(alloc, g.pixels.data(), g.n_vit_h * 14, g.n_vit_w * 14, a1);
        const double cold = ms(t0);
        if (!e.empty()) { std::printf("    encode: %s\n", e.c_str()); ++g_fail; continue; }
        const double free1 = free_mib(alloc.queue());
        t0 = clk::now();
        e = vis.encode_gpu(alloc, g.pixels.data(), g.n_vit_h * 14, g.n_vit_w * 14, a2);
        const double warm = ms(t0);
        bool finite = true; for (float v : a1) finite &= std::isfinite(v);
        const Cmp c = compare(a1, g.rows, HID);
        std::printf("    rows vs reference: rel_l2 %.3e, min row-cos %.6f, p05 row-cos %.6f, mean|d| %.2e (|ref| %.2e); encode %.0f ms, again %.0f ms; device free %.0f MiB\n",
                    c.rel, c.min_cos, c.p05_cos, c.mean_abs, c.ref_abs, cold, warm, free1);
        check(a1.size() == g.rows.size() && finite, "tower: every aligner row is finite");
        if (c.below99) std::printf("    %zu rows below cos 0.99; their reference norm is %.2fx the mean row's\n", c.below99, c.below99_norm);
        // The bar against the exact (fp32) function. Its yardstick is the shipped module in its shipping precision:
        // bf16 vs fp32 measures rel-L2 5.1e-2, min row-cos 0.607 and 26 of 943 rows below 0.99 on fixture 1 (flat
        // dark background; p55/bf16_yardstick.log), so a 0.99 floor on EVERY row is tighter than the model's own run.
        check(c.rel < 2e-2 && c.p05_cos > 0.999 && c.below99 * 200 <= a1.size() / HID && c.min_cos > 0.95 && c.mean_abs < 2e-2 * c.ref_abs,
              "tower: rel-L2 < 2e-2, p05 row-cos > 0.999, <= 0.5 % of rows below 0.99, none below 0.95");
        check(e.empty() && a1 == a2, "tower: bit-identical run to run");
        // the first encode leaves the runtime's one-time state behind (kernel images, pools: ~70 MiB measured); the
        // 1.5 GiB block must not stay, and no later encode may leave anything
        if (free_steady < 0) free_steady = free1;
        check(free0 < 0 || (free0 - free1 < 256.0 && std::fabs(free1 - free_steady) < 16.0), "transient: the device block is returned after every encode");
        // 4. the span
        std::vector<int32_t> types, perm; ie::ds41_vis_types(g.n_llm_h, g.n_llm_w, types, perm);
        bool ty = types.size() == g.types.size() && types.size() == ie::ds41_vis_tokens(g.n_llm_h, g.n_llm_w);
        for (size_t i = 0; ty && i < types.size(); ++i) {     // reference enum: START 0, IMAGE 1, NEW_LINE 2, END 3
            const int32_t want = g.types[i] == 0 ? ie::kDs4VisStart : g.types[i] == 1 ? ie::kDs4VisImage : g.types[i] == 2 ? ie::kDs4VisNewline : ie::kDs4VisEnd;
            ty = types[i] == want;
        }
        check(ty, "span: START, (IMAGE x w, NEW_LINE) x h, END -- as image_token_types");
        std::vector<float> rows;
        e = vis.assemble_rows(a1, types, perm, rows);
        const ie::SafeTensorInfo* st = model.find("image_start");
        bool rows_ok = e.empty() && rows.size() == types.size() * HID && st;
        for (uint32_t d = 0; rows_ok && d < HID; ++d) {
            uint16_t hb; std::memcpy(&hb, st->data + size_t(d) * 2, 2);
            const uint32_t u = uint32_t(hb) << 16; float v; std::memcpy(&v, &u, 4);
            rows_ok = rows[d] == v;
        }
        rows_ok = rows_ok && !std::memcmp(&rows[size_t(1) * HID], a1.data(), HID * 4);     // the first IMAGE slot is aligner row 0
        check(rows_ok, "span: row 0 is the checkpoint's image_start, row 1 is aligner row 0");
    }
    std::printf("\nVISION TEST: %s (%d/%d)\n", g_fail ? "FAIL" : "PASS", g_n - g_fail, g_n);
    return g_fail ? 1 : 0;
}
