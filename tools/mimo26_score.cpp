// tools/mimo26_score.cpp -- ie-mimo26-score: MiMo-V2.6 logits at EVERY position of given token sequences, teacher forced
// (the engine side of the P2 gate's per-position comparison, docs/mimo26/00_PORT_PLAN.md).
//   ie-mimo26-score <model_dir> <ids.txt> <out.bin> [--ubatch U] [--static S] [--pinned P] [--stream Q] [--ranking FILE]
// --ubatch U (default min(longest sequence, 2048)): each sequence runs as forward calls of at most U rows (P3a).
// --image FILE (repeatable, P6.2): the <|image_pad|> ids (151655) of the sequences, in order, each become the image's N
// image ids (the processor's expansion); the tower encodes the image and the forward takes those rows from it. The
// expanded sequences are written beside the MLOG as <out.bin>.ids.txt (ids < 0 = image rows, in order) for
// tools/mimo26/ref_forward.py --splice. --image-tokens N caps an image at N tokens (default 2048, the tower's cap).
// --dump-rows DIR writes each image's tower rows (the engine's, [N, 4096] f32) as DIR/rows_<k>.f32 -- spliced into the
// reference instead of the fp32 goldens, they isolate the language model's numerics from the tower's. --rows-from DIR
// does the opposite: the forward takes each image's rows from DIR/rows_<k>.f32 (e.g. the fp32 goldens) instead of encoding.
// ids.txt: one sequence per line, token ids separated by spaces or commas (brackets ignored). out.bin: "MLOG",
// n_seq (u32), vocab (u32), then per sequence n (u32), ids (n x i32), logits (n x vocab f32) -- the format of
// tools/mimo26/oracle_logits.cpp, compared by tools/mimo26/compare_logits.py. Each sequence starts from an empty cache.
#include "ie/expert_stream.hpp"
#include "ie/allocator.hpp"
#include "ie/mimo26.hpp"
#include "ie/mimo26_engine.hpp"    // mimo26_image_id / mimo26_image_hash (P6.2)
#include "ie/mimo26_forward.hpp"
#include "ie/mimo26_vision.hpp"
#include "stb/stb_image.h"          // stbi_info_from_memory (implementation in qwen4_image.cpp)

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: ie-mimo26-score <model_dir> <ids.txt> <out.bin> [--static S] [--pinned P] [--stream Q] [--ranking FILE] [--image FILE]... [--image-tokens N]\n"); return 2; }
    const std::string model = argv[1];
    ie::Mimo26Options opt;
    std::string ranking_path;   // --ranking: a per-layer expert priority file (ie-mimo26-profile)
    uint32_t U = 0;
    std::vector<std::string> image_files; uint32_t image_tokens = 2048; std::string dump_rows, rows_from;
    for (int i = 4; i < argc; ++i) {
        const std::string a = argv[i]; auto val = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--ubatch") U = uint32_t(std::atol(val().c_str()));
        else if (a == "--static") opt.n_static = uint32_t(std::atol(val().c_str()));
        else if (a == "--pinned") opt.n_pinned = uint32_t(std::atol(val().c_str()));
        else if (a == "--stream") opt.stream_slots = uint32_t(std::atol(val().c_str()));
        else if (a == "--ranking") ranking_path = val();
        else if (a == "--image") image_files.push_back(val());
        else if (a == "--image-tokens") image_tokens = uint32_t(std::atol(val().c_str()));
        else if (a == "--dump-rows") dump_rows = val();
        else if (a == "--rows-from") rows_from = val();
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    std::vector<std::vector<int32_t>> seqs;
    {
        std::ifstream in(argv[2]);
        std::string line;
        while (std::getline(in, line)) {
            for (char& c : line) if (c == ',' || c == '[' || c == ']') c = ' ';
            std::istringstream ss(line);
            std::vector<int32_t> s; long v;
            while (ss >> v) s.push_back(int32_t(v));
            if (!s.empty()) seqs.push_back(std::move(s));
        }
    }
    if (seqs.empty()) { std::fprintf(stderr, "no sequences in %s\n", argv[2]); return 2; }
    // P6.2: the images, planned from their headers; every <|image_pad|> of the sequences (in order) expands to its N ids
    struct ToolImage { std::string bytes; uint64_t hash = 0; ie::MimoVisGeom geom; uint32_t n = 0; size_t seq = 0; uint32_t pos0 = 0; std::vector<float> rows; };
    constexpr int32_t kImagePad = 151655;
    std::vector<ToolImage> images;
    for (const auto& fn : image_files) {
        std::ifstream in(fn, std::ios::binary);
        if (!in) { std::fprintf(stderr, "cannot read %s\n", fn.c_str()); return 2; }
        ToolImage im; im.bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
        im.hash = ie::mimo26_image_hash(im.bytes);
        int w = 0, h = 0, comp = 0;
        if (!stbi_info_from_memory(reinterpret_cast<const stbi_uc*>(im.bytes.data()), int(im.bytes.size()), &w, &h, &comp)) { std::fprintf(stderr, "%s: not an image\n", fn.c_str()); return 2; }
        if (auto e = ie::mimo26_vis_plan(uint32_t(h), uint32_t(w), im.geom, ie::mimo26_image_max_px(image_tokens)); !e.empty()) { std::fprintf(stderr, "%s: %s\n", fn.c_str(), e.c_str()); return 2; }
        im.n = im.geom.tokens();
        std::fprintf(stderr, "image %zu: %s %dx%d -> %ux%u canvas, %ux%u patches, %u image tokens\n", images.size(), fn.c_str(), w, h, im.geom.w, im.geom.h, im.geom.grid_h, im.geom.grid_w, im.n);
        images.push_back(std::move(im));
    }
    if (!images.empty()) {
        size_t k = 0;
        for (size_t si = 0; si < seqs.size(); ++si) {
            std::vector<int32_t> full;
            for (int32_t id : seqs[si]) {
                if (id != kImagePad) { full.push_back(id); continue; }
                if (k >= images.size()) { std::fprintf(stderr, "sequence %zu: more <|image_pad|> ids than --image files\n", si); return 2; }
                ToolImage& im = images[k++]; im.seq = si; im.pos0 = uint32_t(full.size());
                for (uint32_t t = 0; t < im.n; ++t) full.push_back(ie::mimo26_image_id(im.hash, t));
            }
            seqs[si] = std::move(full);
        }
        if (k != images.size()) { std::fprintf(stderr, "%zu --image file(s) have no <|image_pad|> in the sequences\n", images.size() - k); return 2; }
        std::ofstream ids_out(std::string(argv[3]) + ".ids.txt");
        for (const auto& s : seqs) { for (size_t i = 0; i < s.size(); ++i) ids_out << (i ? " " : "") << s[i]; ids_out << "\n"; }
    }
    size_t max_n = 0; for (const auto& s : seqs) max_n = std::max(max_n, s.size());

    ie::Mimo26Model m;
    if (auto e = m.load(model); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d);
    }
    if (devs.empty()) { std::fprintf(stderr, "no Arc GPU\n"); return 1; }
    std::vector<std::unique_ptr<sycl::queue>> queues; std::vector<sycl::queue*> qs;
    for (const auto& d : devs) { queues.push_back(std::make_unique<sycl::queue>(sycl::context(d), d, sycl::property_list{sycl::property::queue::in_order{}})); qs.push_back(queues.back().get()); }
    opt.max_ctx = uint32_t(std::max<size_t>(max_n, 64));
    if (!U) U = std::min<uint32_t>(opt.max_ctx, 2048);
    opt.max_tokens = std::min(U, opt.max_ctx);
    ie::Mimo26Forward fwd;
    if (!ranking_path.empty())
        if (auto e = ie::ds4_expert_priority_read_layers(ranking_path, m.config().n_routed_experts, m.config().n_layers, opt.ranking); !e.empty()) {
            std::fprintf(stderr, "ranking: %s\n", e.c_str()); return 1; }
    ie::DeviceAllocator vis_alloc; ie::MimoVision vis;   // the allocator outlives the tower (it frees the pinned memory)
    if (!images.empty()) {   // the tower: staged before the forward sizes its tiers, its encode block reserved on the first card
        if (m.config().image_token_id != uint32_t(kImagePad)) { std::fprintf(stderr, "config image_token_id %u != %d\n", m.config().image_token_id, kImagePad); return 1; }
        std::string e = vis_alloc.init_with(qs[0]->get_context(), qs[0]->get_device());
        if (e.empty()) e = vis.load_from([&m](const std::string& n) { return m.store().find(n); });
        if (e.empty()) e = vis.stage_host(vis_alloc);
        if (!e.empty()) { std::fprintf(stderr, "vision tower: %s\n", e.c_str()); return 1; }
        opt.reserve_card0 = vis.encode_bytes();
    }
    if (auto e = fwd.init(qs, m, opt); !e.empty()) { std::fprintf(stderr, "init: %s\n", e.c_str()); return 1; }
    size_t cur_seq = 0;
    if (!images.empty()) {   // encode every image up front; the provider serves the current sequence's rows
        for (auto& im : images) {
            std::vector<float> pv; ie::MimoVisGeom g;
            if (auto e = ie::mimo26_load_image_mem(im.bytes.data(), im.bytes.size(), pv, g, ie::mimo26_image_max_px(image_tokens)); !e.empty()) { std::fprintf(stderr, "image: %s\n", e.c_str()); return 1; }
            if (g.tokens() != im.n) { std::fprintf(stderr, "image plan changed between the header and the decode\n"); return 1; }
            if (!rows_from.empty()) {   // the rows come from files (e.g. the fp32 goldens), not from the tower
                std::ifstream in(rows_from + "/rows_" + std::to_string(&im - &images[0]) + ".f32", std::ios::binary);
                im.rows.assign(size_t(im.n) * m.config().dim, 0.f);
                in.read(reinterpret_cast<char*>(im.rows.data()), std::streamsize(im.rows.size() * 4));
                if (!in || in.gcount() != std::streamsize(im.rows.size() * 4)) { std::fprintf(stderr, "--rows-from: rows_%zu.f32 must hold %u x %u f32\n", size_t(&im - &images[0]), im.n, m.config().dim); return 1; }
            } else if (auto e = vis.encode_gpu(vis_alloc, pv, g, im.rows); !e.empty()) { std::fprintf(stderr, "encode: %s\n", e.c_str()); return 1; }
            if (!dump_rows.empty()) {
                std::ofstream o(dump_rows + "/rows_" + std::to_string(&im - &images[0]) + ".f32", std::ios::binary);
                o.write(reinterpret_cast<const char*>(im.rows.data()), std::streamsize(im.rows.size() * 4));
            }
        }
        fwd.set_vision_provider([&](uint32_t pos, const float*& row) -> std::string {
            for (const auto& im : images)
                if (im.seq == cur_seq && pos >= im.pos0 && pos - im.pos0 < im.n) { row = im.rows.data() + size_t(pos - im.pos0) * m.config().dim; return {}; }
            return "position " + std::to_string(pos) + " of sequence " + std::to_string(cur_seq) + " is outside every image";
        });
    }

    const uint32_t V = m.config().vocab_size;
    std::FILE* f = std::fopen(argv[3], "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }
    const uint32_t hdr[2] = {uint32_t(seqs.size()), V};
    std::fwrite("MLOG", 1, 4, f); std::fwrite(hdr, 4, 2, f);
    std::vector<float> logits;
    for (size_t si = 0; si < seqs.size(); ++si) {
        const auto& s = seqs[si];
        cur_seq = si;
        fwd.reset();
        const uint32_t n = uint32_t(s.size());
        std::fwrite(&n, 4, 1, f);
        std::fwrite(s.data(), 4, n, f);
        for (uint32_t off = 0; off < n; off += opt.max_tokens) {
            const uint32_t m = std::min(opt.max_tokens, n - off);
            if (auto e = fwd.forward(s.data() + off, m, off, logits, true); !e.empty()) { std::fprintf(stderr, "sequence %zu at %u: %s\n", si, off, e.c_str()); return 1; }
            std::fwrite(logits.data(), 4, size_t(m) * V, f);
        }
        std::fprintf(stderr, "sequence %zu: %u tokens\n", si, n);
    }
    std::fclose(f);
    return 0;
}
