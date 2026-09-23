// tools/mimo26_dflash_check.cpp -- ie-mimo26-dflash-check: P5 gate (d) (docs/mimo26/00_PORT_PLAN.md). The engine's DFlash
// drafter (src/model/mimo26_dflash.cpp) fed the SAME context features as tools/mimo26/dflash_accept.py: the residual dumps
// of one ie-mimo26-score call (IE_MIMO26_DUMP + IE_MIMO26_DUMP_LAYERS = the drafter's target layers); at each continuation
// position p, the drafts after ids[p] with the context [p - window, p). Writes "p d1 .. d7" per position -- the reference
// writes the same with dflash_accept.py --print-drafts, and the two files are compared row by row (top-1 per draft row).
//   ie-mimo26-dflash-check <model_dir> <dump_dir> <ids.txt> <n_prompt> <out.txt> [--call N] [--max-pos M] [--plain]
// --plain: dflash.py's forward as shipped (no sinks, value scale 1) -- compare with dflash_accept.py --print-variant plain.
#include "ie/mimo26.hpp"
#include "ie/mimo26_dflash.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: ie-mimo26-dflash-check <model_dir> <dump_dir> <ids.txt> <n_prompt> <out.txt> [--call N] [--max-pos M] [--plain]\n");
        return 2;
    }
    const std::string model = argv[1], dump = argv[2], ids_path = argv[3], out_path = argv[5];
    const uint32_t n_prompt = uint32_t(std::atol(argv[4]));
    uint32_t call = 0, max_pos = 250;
    bool plain = false;
    for (int i = 6; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--call" && i + 1 < argc) call = uint32_t(std::atol(argv[++i]));
        else if (a == "--max-pos" && i + 1 < argc) max_pos = uint32_t(std::atol(argv[++i]));
        else if (a == "--plain") plain = true;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    std::vector<int32_t> ids;   // dflash_accept.py's reader: commas and brackets are spaces
    {
        std::ifstream f(ids_path);
        std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        for (char& ch : s) if (ch == ',' || ch == '[' || ch == ']') ch = ' ';
        std::istringstream is(s);
        for (long v; is >> v;) ids.push_back(int32_t(v));
    }
    ie::Mimo26Model m;
    if (auto e = m.load(model); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    const uint32_t H = m.config().dim, V = m.config().vocab_size, T = uint32_t(ids.size());

    sycl::device dev;
    bool found = false;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu))
            if (!found && d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) { dev = d; found = true; }
    }
    if (!found) { std::fprintf(stderr, "no Arc GPU\n"); return 1; }
    sycl::queue q(sycl::context(dev), dev, sycl::property_list{sycl::property::queue::in_order{}});

    ie::Mimo26DFlash df;
    if (auto e = df.load(model); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
    const auto& dc = df.config();
    if (auto e = df.init(q, m.embed.w->data, V, dc.window); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
    // the target's lm_head as the forward uploads it: bf16 -> fp32 -> host fp16
    sycl::half* head = sycl::malloc_device<sycl::half>(size_t(V) * H, q);
    if (!head) { std::fprintf(stderr, "lm_head allocation failed\n"); return 1; }
    {
        const uint32_t RB = 1024;
        std::vector<float> row(size_t(RB) * H);
        std::vector<sycl::half> h(row.size());
        for (uint32_t r0 = 0; r0 < V; r0 += RB) {
            const uint32_t n = std::min(RB, V - r0);
            ie::mimo26_bf16_to_f32(m.lm_head.w->data + size_t(r0) * H * 2, n * H, row.data());
            for (size_t i = 0; i < size_t(n) * H; ++i) h[i] = sycl::half(row[i]);
            q.memcpy(head + size_t(r0) * H, h.data(), size_t(n) * H * 2).wait();
        }
    }
    df.set_head(head);
    df.set_reference_form(plain);

    const uint32_t nf = uint32_t(dc.target_layers.size());
    std::vector<float> feats(size_t(nf) * T * H);   // [nf][T][H]
    for (uint32_t f = 0; f < nf; ++f) {
        const std::string fn = dump + "/x_L" + std::to_string(dc.target_layers[f]) + "_s" + std::to_string(call) + ".f32";
        std::ifstream in(fn, std::ios::binary);
        if (!in.read(reinterpret_cast<char*>(feats.data() + size_t(f) * T * H), std::streamsize(size_t(T) * H * 4))) {
            std::fprintf(stderr, "cannot read %u x %u floats from %s\n", T, H, fn.c_str()); return 1;
        }
    }
    std::vector<float> buf;
    auto add = [&](uint32_t a, uint32_t b) -> std::string {   // positions [a, b), layer-major
        const uint32_t n = b - a;
        buf.resize(size_t(nf) * n * H);
        for (uint32_t f = 0; f < nf; ++f) std::memcpy(buf.data() + size_t(f) * n * H, feats.data() + (size_t(f) * T + a) * H, size_t(n) * H * 4);
        return df.add_context(buf.data(), n, a);
    };
    if (auto e = add(n_prompt > dc.window ? n_prompt - dc.window : 0u, n_prompt); !e.empty()) { std::fprintf(stderr, "%s\n", e.c_str()); return 1; }
    std::FILE* out = std::fopen(out_path.c_str(), "w");
    if (!out) { std::fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1; }
    const uint32_t end = std::min<uint32_t>(T - dc.block, n_prompt + max_pos);   // dflash_accept.py's range
    std::vector<int32_t> d;
    for (uint32_t p = n_prompt; p < end; ++p) {
        if (auto e = df.draft(ids[p], dc.block - 1, d); !e.empty()) { std::fprintf(stderr, "draft at %u: %s\n", p, e.c_str()); return 1; }
        std::fprintf(out, "%u", p);
        for (int32_t t : d) std::fprintf(out, " %d", t);
        std::fprintf(out, "\n");
        if (auto e = add(p, p + 1); !e.empty()) { std::fprintf(stderr, "context at %u: %s\n", p, e.c_str()); return 1; }
    }
    std::fclose(out);
    std::printf("drafted %u positions [%u, %u) -> %s\n", end - n_prompt, n_prompt, end, out_path.c_str());
    df.free_all();
    sycl::free(head, q);
    return 0;
}
