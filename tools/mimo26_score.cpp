// tools/mimo26_score.cpp -- ie-mimo26-score: MiMo-V2.6 logits at EVERY position of given token sequences, teacher forced
// (the engine side of the P2 gate's per-position comparison, docs/mimo26/00_PORT_PLAN.md).
//   ie-mimo26-score <model_dir> <ids.txt> <out.bin> [--ubatch U] [--static S] [--pinned P] [--stream Q] [--ranking FILE]
// --ubatch U (default min(longest sequence, 2048)): each sequence runs as forward calls of at most U rows (P3a).
// ids.txt: one sequence per line, token ids separated by spaces or commas (brackets ignored). out.bin: "MLOG",
// n_seq (u32), vocab (u32), then per sequence n (u32), ids (n x i32), logits (n x vocab f32) -- the format of
// tools/mimo26/oracle_logits.cpp, compared by tools/mimo26/compare_logits.py. Each sequence starts from an empty cache.
#include "ie/expert_stream.hpp"
#include "ie/mimo26.hpp"
#include "ie/mimo26_forward.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: ie-mimo26-score <model_dir> <ids.txt> <out.bin> [--static S] [--pinned P] [--stream Q] [--ranking FILE]\n"); return 2; }
    const std::string model = argv[1];
    ie::Mimo26Options opt;
    std::string ranking_path;   // --ranking: a per-layer expert priority file (ie-mimo26-profile)
    uint32_t U = 0;
    for (int i = 4; i < argc; ++i) {
        const std::string a = argv[i]; auto val = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); } return argv[++i]; };
        if (a == "--ubatch") U = uint32_t(std::atol(val().c_str()));
        else if (a == "--static") opt.n_static = uint32_t(std::atol(val().c_str()));
        else if (a == "--pinned") opt.n_pinned = uint32_t(std::atol(val().c_str()));
        else if (a == "--stream") opt.stream_slots = uint32_t(std::atol(val().c_str()));
        else if (a == "--ranking") ranking_path = val();
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
    if (auto e = fwd.init(qs, m, opt); !e.empty()) { std::fprintf(stderr, "init: %s\n", e.c_str()); return 1; }

    const uint32_t V = m.config().vocab_size;
    std::FILE* f = std::fopen(argv[3], "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", argv[3]); return 1; }
    const uint32_t hdr[2] = {uint32_t(seqs.size()), V};
    std::fwrite("MLOG", 1, 4, f); std::fwrite(hdr, 4, 2, f);
    std::vector<float> logits;
    for (size_t si = 0; si < seqs.size(); ++si) {
        const auto& s = seqs[si];
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
