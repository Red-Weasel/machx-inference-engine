// tools/mimo26_profile.cpp -- ie-mimo26-profile: a MiMo-V2.6 expert-residency ranking from real traffic (P4 lever 1,
// docs/mimo26/00_PORT_PLAN.md). Each sequence is prefilled; the routed selections at the COUNTED positions (the
// assistant turns of a chat -- the decode workload; V4.1: a ranking is only valid for the workload it was profiled on)
// are summed per (layer, expert) and written most-selected-first in the per-layer priority format
// (ds4_expert_priority_read_layers). The dense layer 0 gets the identity order.
//   ie-mimo26-profile <model_dir> <seqs.jsonl> <out_ranking.txt> [--ubatch U] [--static S]
// --static 0 sizes the static expert tier from free VRAM (MiMo-V2.6-Pro: 48 per layer does not fit a 32 GB card).
// seqs.jsonl: one {"ids": [...], "count": [[a, b], ...]} per line (count positions a <= p < b).
#include "ie/expert_stream.hpp"
#include "ie/mimo26.hpp"
#include "ie/mimo26_forward.hpp"

#include "../third_party/nlohmann/json.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: ie-mimo26-profile <model_dir> <seqs.jsonl> <out_ranking.txt> [--ubatch U] [--static S]\n"); return 2; }
    const std::string model = argv[1], seqs_path = argv[2], out_path = argv[3];
    uint32_t U = 2048;
    ie::Mimo26Options opt;
    for (int i = 4; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--ubatch" && i + 1 < argc) U = uint32_t(std::atol(argv[++i]));
        else if (a == "--static" && i + 1 < argc) opt.n_static = uint32_t(std::atol(argv[++i]));
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    struct Seq { std::vector<int32_t> ids; std::vector<uint8_t> mask; };
    std::vector<Seq> seqs;
    size_t max_n = 0, counted = 0;
    {
        std::ifstream in(seqs_path);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            const auto j = nlohmann::json::parse(line, nullptr, false);
            if (j.is_discarded() || !j.contains("ids")) { std::fprintf(stderr, "bad line in %s\n", seqs_path.c_str()); return 2; }
            Seq s;
            s.ids = j["ids"].get<std::vector<int32_t>>();
            s.mask.assign(s.ids.size(), 0);
            for (const auto& r : j.value("count", nlohmann::json::array())) {
                const size_t a = r[0].get<size_t>(), b = std::min<size_t>(r[1].get<size_t>(), s.ids.size());
                for (size_t p = a; p < b; ++p) s.mask[p] = 1;
            }
            counted += size_t(std::count(s.mask.begin(), s.mask.end(), uint8_t(1)));
            max_n = std::max(max_n, s.ids.size());
            seqs.push_back(std::move(s));
        }
    }
    if (seqs.empty()) { std::fprintf(stderr, "no sequences\n"); return 2; }

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
    opt.max_tokens = std::min<uint32_t>(U, opt.max_ctx);
    ie::Mimo26Forward fwd;
    if (auto e = fwd.init(qs, m, opt); !e.empty()) { std::fprintf(stderr, "init: %s\n", e.c_str()); return 1; }
    fwd.set_profile(true);
    std::printf("mimo26 profile: %zu sequences, %zu counted positions, longest %zu\n", seqs.size(), counted, max_n);
    std::vector<float> logits;
    const auto t0 = std::chrono::steady_clock::now();
    size_t done_tok = 0;
    for (size_t si = 0; si < seqs.size(); ++si) {
        const auto& s = seqs[si];
        fwd.reset();
        for (uint32_t off = 0; off < s.ids.size(); off += opt.max_tokens) {
            const uint32_t n = std::min<uint32_t>(opt.max_tokens, uint32_t(s.ids.size()) - off);
            fwd.set_profile_rows(s.mask.data() + off);
            if (auto e = fwd.forward(s.ids.data() + off, n, off, logits, false); !e.empty()) { std::fprintf(stderr, "sequence %zu at %u: %s\n", si, off, e.c_str()); return 1; }
        }
        done_tok += s.ids.size();
        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("[%zu/%zu] %zu tokens, %.0f tok/s\n", si + 1, seqs.size(), s.ids.size(), done_tok / el);
        std::fflush(stdout);
    }
    const auto& pr = fwd.profile();
    std::vector<std::vector<uint32_t>> orders(pr.size());
    uint64_t total = 0;
    for (size_t L = 0; L < pr.size(); ++L) {
        orders[L].resize(pr[L].size());
        std::iota(orders[L].begin(), orders[L].end(), 0u);
        std::stable_sort(orders[L].begin(), orders[L].end(), [&](uint32_t a, uint32_t b) { return pr[L][a] > pr[L][b]; });
        for (uint64_t v : pr[L]) total += v;
    }
    if (auto e = ie::ds4_expert_priority_write_layers(out_path, orders, {}, {
            "MiMo-V2.6 expert ranking from ie-mimo26-profile: " + std::to_string(seqs.size()) + " sequences of " + seqs_path,
            std::to_string(counted) + " counted positions (assistant turns), " + std::to_string(total) + " routed selections"}); !e.empty()) {
        std::fprintf(stderr, "write: %s\n", e.c_str()); return 1;
    }
    // coverage: the share of counted selections the top-N experts of each layer take (what a static tier of N would hit)
    for (uint32_t N : {32u, 48u, 64u, 96u, 128u}) {
        uint64_t hit = 0, all = 0;
        for (size_t L = 0; L < pr.size(); ++L) {
            for (uint32_t i = 0; i < pr[L].size(); ++i) { all += pr[L][orders[L][i]]; if (i < N) hit += pr[L][orders[L][i]]; }
        }
        std::printf("top-%u per layer covers %.1f %% of the counted selections\n", N, all ? 100.0 * double(hit) / double(all) : 0.0);
    }
    std::printf("wrote %s\n", out_path.c_str());
    return 0;
}
