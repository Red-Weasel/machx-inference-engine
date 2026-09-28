// tools/q27_lanes_equiv_test.cpp -- P4 B18: the 27B split's request-lane steps against --parallel 1's forward, bit for bit, on
// the GPU (the T > 1 vs T == 1 reduction-order hunt; the host rules are tests/unit/q27_lanes_test.cpp).
//
// Method (teacher-forced, synthetic ids, no tokenizer):
//  1. Reference: each of N sequences alone on lane 0 through --parallel 1's calls: the prompt with Engine::generate's
//     prefill_to rule (forward_pipelined for a range longer than one chunk, else forward chunk by chunk), then `steps` decode
//     rows with forward(T = 1). A 64-bit hash + the argmax of every logits row is kept (prefill last row + every step).
//  2. Lanes: sequence i on lane i (lane 0 = the --ctx state, lanes 1.. at --slot-ctx, so the KV layouts differ): its prompt
//     through forward_stage piece by piece with the plan's kinds (q27_prefill_ops: a 1-row pipelined piece through the prefill
//     kernels), card 0 then card 1; then every step the N lanes are cut into groups whose size cycles 1, 2, ..., N (a group
//     of 1 = forward_stage T = 1, of G >= 2 = forward_stage_rows), card 0 then card 1 per group.
//  3. Every row's hash must equal the reference's: the prefill pieces (forward_stage vs forward_pipelined / forward), the
//     lone decode row (forward_slots' walk vs forward(T = 1)) and every group size 2..N (the batched int-dot, the dual kernels,
//     the per-row alpha/beta, attention on another KV layout).
//
// usage: ie-q27-lanes-equiv --gguf <27B.gguf> [--lanes 16] [--depth 1025] [--steps 48] [--ctx 8192] [--slot-ctx 4096]
//                           [--chunk 512] [--no-pipeline]
// Exit 0 = every row bit-identical.
#include "ie/allocator.hpp"
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/q27_lanes.hpp"
#include "ie/qwen35_split.hpp"

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using ie::Qwen35SplitModel;

namespace {
std::vector<int32_t> synth_prompt(uint32_t seq, uint32_t len) {
    std::vector<int32_t> ids(len);
    for (uint32_t t = 0; t < len; ++t) ids[t] = int32_t(1000u + (seq * 7919u + t * 2654435761u) % 49000u);
    return ids;
}
int32_t synth_next(uint32_t seq, uint32_t step) { return int32_t(1000u + (seq * 104729u + step * 40503u) % 49000u); }

struct Row { uint64_t h = 0; uint32_t argmax = 0; };
Row digest(const sycl::half* host, uint32_t V) {
    Row r;
    uint64_t h = 0xcbf29ce484222325ull;
    float best = -1e30f;
    for (uint32_t v = 0; v < V; ++v) {
        uint16_t b;
        std::memcpy(&b, &host[v], 2);
        h = (h ^ b) * 0x100000001B3ull;
        const float f = float(host[v]);
        if (f > best) { best = f; r.argmax = v; }
    }
    r.h = h;
    return r;
}
}  // namespace

int main(int argc, char** argv) {
    std::string gguf;
    uint32_t n_lanes = 16, depth = 1025, steps = 48, max_ctx = 8192, slot_ctx = 4096, chunk = 512;
    bool pipeline = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto num = [&] { return uint32_t(std::atoi(argv[++i])); };
        if      (a == "--gguf" && i + 1 < argc)     gguf = argv[++i];
        else if (a == "--lanes" && i + 1 < argc)    n_lanes = num();
        else if (a == "--depth" && i + 1 < argc)    depth = num();
        else if (a == "--steps" && i + 1 < argc)    steps = num();
        else if (a == "--ctx" && i + 1 < argc)      max_ctx = num();
        else if (a == "--slot-ctx" && i + 1 < argc) slot_ctx = num();
        else if (a == "--chunk" && i + 1 < argc)    chunk = num();
        else if (a == "--no-pipeline")              pipeline = false;
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (gguf.empty() || n_lanes < 2 || n_lanes > Qwen35SplitModel::kMaxRows || depth < 2 || chunk == 0 ||
        depth + steps + 8 > slot_ctx || slot_ctx > max_ctx) {
        std::fprintf(stderr, "usage: --gguf F [--lanes 2..16] [--depth D >= 2] [--steps S] [--ctx C] [--slot-ctx S >= D + S + 8]\n");
        return 2;
    }
    ie::GgufReader g;
    if (auto m = g.open(gguf); !m.empty()) { std::fprintf(stderr, "gguf: %s\n", m.c_str()); return 1; }
    ie::Qwen35Config qcfg;
    if (auto m = ie::read_qwen35_config(g, qcfg); !m.empty()) { std::fprintf(stderr, "config: %s\n", m.c_str()); return 1; }
    ie::DeviceFleet fleet;
    if (auto m = fleet.init(2, "B70", false, /*shared_ctx=*/false); !m.empty()) { std::fprintf(stderr, "fleet: %s\n", m.c_str()); return 1; }
    ie::LayerPlan plan = ie::LayerPlan::contiguous(qcfg.n_transformer_layers(), 2);
    Qwen35SplitModel model;
    if (auto m = model.load(fleet, plan, g, qcfg, max_ctx, false); !m.empty()) { std::fprintf(stderr, "model: %s\n", m.c_str()); return 1; }
    if (auto m = model.init_lanes(n_lanes, slot_ctx, chunk, 1536ull << 20); !m.empty()) { std::fprintf(stderr, "lanes: %s\n", m.c_str()); return 1; }
    const std::string rows_why = model.rows_off_reason();
    if (!rows_why.empty()) { std::fprintf(stderr, "rows off: %s\n", rows_why.c_str()); return 1; }
    const uint32_t V = qcfg.dense.vocab, H = qcfg.dense.hidden;
    sycl::queue& hq = fleet.dev(model.head_dev()).queue();
    std::printf("loaded: %u lanes (lane 0 ctx %u, lanes 1.. ctx %u), depth %u, steps %u, chunk %u, pipeline %s\n", n_lanes,
                model.lane_ctx(0), model.lane_ctx(1), depth, steps, chunk, pipeline ? "on" : "off");

    // 1. the reference: --parallel 1's calls on lane 0, one sequence at a time
    std::vector<Row> ref(size_t(n_lanes) * (steps + 1));
    std::vector<sycl::half> lg(V);
    const auto t_ref = std::chrono::steady_clock::now();
    model.select_lane(0);
    for (uint32_t s = 0; s < n_lanes; ++s) {
        const std::vector<int32_t> ids = synth_prompt(s, depth);
        if (pipeline && depth > chunk) {
            if (auto m = model.forward_pipelined(ids.data(), depth, 0, true, chunk, lg.data()); !m.empty()) { std::fprintf(stderr, "ref pipelined: %s\n", m.c_str()); return 1; }
        } else {
            for (uint32_t p = 0; p < depth; p += chunk) {
                const uint32_t n = std::min(chunk, depth - p);
                if (auto m = model.forward(ids.data() + p, n, p, p == 0, lg.data()); !m.empty()) { std::fprintf(stderr, "ref forward: %s\n", m.c_str()); return 1; }
            }
        }
        ref[size_t(s) * (steps + 1)] = digest(lg.data(), V);
        for (uint32_t t = 0; t < steps; ++t) {
            const int32_t tok = synth_next(s, t);
            if (auto m = model.forward(&tok, 1, depth + t, false, lg.data()); !m.empty()) { std::fprintf(stderr, "ref step: %s\n", m.c_str()); return 1; }
            ref[size_t(s) * (steps + 1) + 1 + t] = digest(lg.data(), V);
        }
    }
    const double ref_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_ref).count();

    // 2. the lanes
    std::vector<sycl::half> xh(uint64_t(std::max<uint32_t>(chunk, n_lanes)) * H);
    uint64_t rows = 0, bad = 0, bad_argmax = 0, pf_bad = 0, pk1 = 0;
    std::vector<uint64_t> bad_by_g(n_lanes + 1, 0), rows_by_g(n_lanes + 1, 0);
    auto check = [&](uint32_t s, uint32_t k, const sycl::half* dev_row, uint32_t G) {
        hq.memcpy(lg.data(), dev_row, uint64_t(V) * 2).wait();
        const Row got = digest(lg.data(), V), want = ref[size_t(s) * (steps + 1) + k];
        ++rows; ++rows_by_g[G];
        if (got.h != want.h) {
            ++bad; ++bad_by_g[G];
            if (got.argmax != want.argmax) ++bad_argmax;
            if (bad <= 10) std::printf("  MISMATCH seq %u row %u (group of %u): argmax %u vs %u\n", s, k, G, got.argmax, want.argmax);
        }
    };
    const auto t_lanes = std::chrono::steady_clock::now();
    for (uint32_t s = 0; s < n_lanes; ++s) {
        const std::vector<int32_t> ids = synth_prompt(s, depth);
        const auto ops = ie::q27_prefill_ops(depth, 0, depth, 0, false, chunk, pipeline);
        for (const ie::Q27Op& o : ops) {
            const ie::Q27Piece& p = o.piece;
            pk1 += p.rows == 1 && p.pk;
            for (uint32_t dev = 0; dev < 2; ++dev)
                if (auto m = model.forward_stage(dev, s, ids.data() + p.pos0, p.rows, p.pos0, xh.data(), p.pk); !m.empty()) {
                    std::fprintf(stderr, "lane %u prefill: %s\n", s, m.c_str()); return 1;
                }
        }
        const uint64_t b0 = bad;
        check(s, 0, model.logits(), 1);
        pf_bad += bad != b0;
    }
    uint32_t G = 1;
    std::vector<uint32_t> lanes(n_lanes), pos(n_lanes);
    std::vector<int32_t> toks(n_lanes);
    for (uint32_t t = 0; t < steps; ++t) {
        for (uint32_t s = 0; s < n_lanes; ++s) { lanes[s] = s; pos[s] = depth + t; toks[s] = synth_next(s, t); }
        for (uint32_t a = 0; a < n_lanes; a += G) {
            const uint32_t g = std::min(G, n_lanes - a);
            for (uint32_t dev = 0; dev < 2; ++dev) {
                const std::string m = g == 1
                    ? model.forward_stage(dev, lanes[a], &toks[a], 1, pos[a], xh.data(), false)
                    : model.forward_stage_rows(dev, std::span<const uint32_t>(&lanes[a], g), &toks[a], &pos[a], xh.data());
                if (!m.empty()) { std::fprintf(stderr, "step %u group at %u of %u: %s\n", t, a, g, m.c_str()); return 1; }
            }
            for (uint32_t r = 0; r < g; ++r)
                check(a + r, 1 + t, g == 1 ? model.logits() : model.rows_logits() + uint64_t(r) * V, g);
        }
        G = G % n_lanes + 1;
    }
    const double lanes_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_lanes).count();
    std::printf("reference %.1f s, lanes %.1f s; %llu 1-row prefill-kernel pieces\n", ref_s, lanes_s, (unsigned long long)pk1);
    std::printf("prefill last rows: %u of %u bit-identical\n", n_lanes - uint32_t(pf_bad), n_lanes);
    for (uint32_t g = 1; g <= n_lanes; ++g)
        if (rows_by_g[g]) std::printf("  group of %2u: %llu rows, %llu differ\n", g, (unsigned long long)rows_by_g[g], (unsigned long long)bad_by_g[g]);
    std::printf("%s: %llu rows, %llu differ (%llu with another argmax)\n", bad ? "FAIL" : "PASS", (unsigned long long)rows,
                (unsigned long long)bad, (unsigned long long)bad_argmax);
    return bad ? 1 : 0;
}
