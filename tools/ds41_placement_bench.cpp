// tools/ds41_placement_bench.cpp — ie-ds41-placement-bench: Phase 13 step 1 (docs/deepseek41/34): the
// decode hit rate, bytes per token and ms/token under a residency placement, at two contexts (the
// table of docs/deepseek41/35 step 1; committed at gate 13's request so the table can be re-derived).
//   IE_DS41_STREAM_SLOTS=N    stream (LRU) slots per layer; the static count derives from the VRAM budget
//   IE_DS41_CTX=2048|4096     the long context: the pp2048 text once, or twice (4096)
//   usage: ie-ds41-placement-bench <model> <golden_dir> <ranking>
#include "ie/deepseek41_generate.hpp"
#include "ie/expert_stream.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

int main(int argc, char** argv) {
    const std::string dir = argv[1], gd = argv[2], rank_in = argv[3];
    const uint32_t stream_slots = std::getenv("IE_DS41_STREAM_SLOTS") ? uint32_t(std::atoi(std::getenv("IE_DS41_STREAM_SLOTS"))) : 8u;
    const uint32_t ctx = std::getenv("IE_DS41_CTX") ? uint32_t(std::atoi(std::getenv("IE_DS41_CTX"))) : 2048u;
    nlohmann::json meta; { std::ifstream f(gd + "/decode2/d_meta.json"); f >> meta; }
    const std::vector<int32_t> prompt = meta["prompt_ids"].get<std::vector<int32_t>>();
    std::vector<int32_t> pp(ctx); { std::ifstream f(gd + "/pp_ids_2048.i32", std::ios::binary); f.read(reinterpret_cast<char*>(pp.data()), 2048 * 4); }
    if (ctx == 4096) for (uint32_t i = 0; i < 2048; ++i) pp[2048 + i] = pp[i];

    ie::DeepSeek41Model m; if (auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    ie::Ds41EngramTables tb; if (auto e = tb.load(gd); !e.empty()) { std::fprintf(stderr, "tables: %s\n", e.c_str()); return 1; }
    std::vector<sycl::device> devs;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu)) if (d.get_info<sycl::info::device::name>().find("Arc") != std::string::npos) devs.push_back(d);
    }
    std::vector<std::unique_ptr<sycl::queue>> queues; std::vector<sycl::queue*> qs;
    for (const auto& d : devs) { queues.push_back(std::make_unique<sycl::queue>(sycl::context(d), d, sycl::property_list{sycl::property::queue::in_order{}})); qs.push_back(queues.back().get()); }
    std::vector<std::vector<uint32_t>> ranking;
    if (auto e = ie::ds4_expert_priority_read_layers(rank_in, m.config().n_routed_experts, m.config().n_layers, ranking); !e.empty()) { std::fprintf(stderr, "ranking: %s\n", e.c_str()); return 1; }
    ie::Ds41Forward fwd; ie::Ds41Forward::ResidentOptions opt; opt.stream_slots = stream_slots; opt.max_tokens = ctx + 64;
    const auto t0 = std::chrono::steady_clock::now();
    if (auto e = fwd.init_resident(qs, m, tb, ranking, opt); !e.empty()) { std::fprintf(stderr, "init_resident: %s\n", e.c_str()); return 1; }
    fwd.set_logits_last_only(true);
    std::printf("placement: stream_slots %u -> card 0: %u static + %u stream, card 1: %u static + %u stream (load %.0f s)\n", stream_slots,
                fwd.card_info(0).n_static, fwd.card_info(0).n_stream, fwd.card_info(1).n_static, fwd.card_info(1).n_stream, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    const uint32_t NL = m.config().n_layers, TK = m.config().n_activated_experts;
    auto bench = [&](const char* label, const std::vector<int32_t>& ids) {
        std::vector<float> lg;
        fwd.set_bounded_replay(true);
        if (auto e = fwd.forward(ids.data(), uint32_t(ids.size()), 0, lg); !e.empty()) { std::fprintf(stderr, "prefill: %s\n", e.c_str()); std::exit(1); }
        std::vector<double> wall; uint64_t bp = 0, bm = 0; uint32_t sta = 0, sh = 0, mm = 0;
        for (uint32_t k = 0; k < 36; ++k) {
            const int32_t nxt = int32_t(std::max_element(lg.begin(), lg.end()) - lg.begin());
            const auto ta = std::chrono::steady_clock::now();
            if (auto e = fwd.forward(&nxt, 1, uint32_t(ids.size()) + k, lg); !e.empty()) { std::fprintf(stderr, "step: %s\n", e.c_str()); std::exit(1); }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ta).count();
            if (k < 4) continue;
            wall.push_back(ms);
            for (const auto& s : fwd.stats()) { bp += s.bytes_pinned; bm += s.bytes_mmap; sta += s.experts_static; sh += s.experts_stream_hit; mm += s.experts_mmap; }
        }
        const double mean = std::accumulate(wall.begin(), wall.end(), 0.0) / wall.size();
        std::printf("%-28s %6.1f ms/token = %5.2f tok/s | pinned %6.1f MiB/token, disk %6.1f MiB/token | static %5.1f + stream hits %5.1f of %u = hit rate %4.1f%% | mmap %4.1f\n",
                    label, mean, 1000.0 / mean, double(bp) / 32 / (1 << 20), double(bm) / 32 / (1 << 20), sta / 32.0, sh / 32.0, NL * TK, 100.0 * (sta + sh) / 32.0 / (NL * TK), mm / 32.0);
    };
    bench("after the 12-token prompt", prompt);
    bench((std::string("after ") + std::to_string(ctx) + "-token context").c_str(), pp);
    fwd.free_resident();
    for (auto& q : queues) q->wait_and_throw();
    return 0;
}
