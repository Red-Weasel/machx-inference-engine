// tools/ie_qwen35moe_bench.cpp — crown (Qwen3.6-35B-A3B, kQwen35Moe) all-Q8_0
// LAYER-SPLIT decode/prefill benchmark (>=2 GPUs). ie-bench is single-GPU only;
// this is the matched 2-card harness for the crown split (Qwen35MoeSplitModel),
// methodology-aligned with `llama-bench -p512 -n128 -sm layer` for the head-to-head.
//
// One model load yields clean pp512 + tg128: a prefill warmup pass (JIT discarded),
// a timed prefill, a decode warmup, then the timed decode. Wall-clock around the
// host-synchronous split forward (host ids in, host logits out — same shape as
// GptOssTpModel::forward / Engine::forward_step's split branch).
//
// Usage:
//   ie-qwen35moe-bench --gguf <q8 crown.gguf> [--gpus 2] [--ctx 4096]
//                      [--prefill 512] [--decode 128] [--warmup 8]
//
// Discipline (project rules): run on a CLEAN box (GPU idle-clocked, swap≈0, CPU
// quiet — the cross-card x handoff is host-bounced so CPU contention fakes a
// slowdown). Discard the FIRST whole run after any rebuild (JIT). Crown prefill
// is §1-capped at T≤512, so --prefill defaults to 512 (one batch, no chunking).

#include "ie/allocator.hpp"
#include "ie/dense_split.hpp"          // LayerPlan
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/qwen35moe_split.hpp"

#include <sycl/sycl.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
constexpr const char* B = "\033[1m";
constexpr const char* Z = "\033[0m";
double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

int main(int argc, char** argv) {
    std::string gguf_path =
        "/home/weezy/models/LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Claude-Genesis-GGUF/"
        "Qwen3.6-35B-A3B-Uncensored-Claude-Genesis-V4-Q8_0.gguf";
    uint32_t n_gpus       = 2;
    uint32_t max_ctx      = 4096;
    uint32_t prefill      = 512;
    uint32_t decode_steps = 128;
    uint32_t warmup       = 8;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf"    && i + 1 < argc) gguf_path     = argv[++i];
        else if (a == "--gpus"    && i + 1 < argc) n_gpus        = std::atoi(argv[++i]);
        else if (a == "--ctx"     && i + 1 < argc) max_ctx       = std::atoi(argv[++i]);
        else if (a == "--prefill" && i + 1 < argc) prefill       = std::atoi(argv[++i]);
        else if (a == "--decode"  && i + 1 < argc) decode_steps  = std::atoi(argv[++i]);
        else if (a == "--warmup"  && i + 1 < argc) warmup        = std::atoi(argv[++i]);
    }

    std::printf("%sie-qwen35moe-bench%s — crown all-Q8_0 2-card decode/prefill tok/s\n", B, Z);
    std::printf("  gguf    : %s\n", gguf_path.c_str());
    std::printf("  gpus    : %u\n  ctx     : %u\n  prefill : %u\n  decode  : %u (+%u warmup)\n",
                n_gpus, max_ctx, prefill, decode_steps, warmup);

    ie::GgufReader g;
    if (auto err = g.open(gguf_path); !err.empty()) {
        std::fprintf(stderr, "gguf open: %s\n", err.c_str()); return 1;
    }
    if (ie::detect_arch(g) != ie::ModelArch::kQwen35Moe) {
        std::fprintf(stderr, "not a qwen35moe (crown) gguf\n"); return 1;
    }
    ie::QwenConfig cfg;   // crown defaults (GGUF metadata verified to match)

    if (prefill + warmup + decode_steps + 4 > max_ctx) {
        max_ctx = prefill + warmup + decode_steps + 4;
        std::printf("  ctx grew to %u (prefill + decode slack)\n", max_ctx);
    }

    const double t0 = now_ms();
    std::printf("  loading (layer-split, %u cards)...\n", n_gpus);
    ie::DeviceFleet fleet;
    if (auto err = fleet.init(n_gpus); !err.empty()) {
        std::fprintf(stderr, "fleet.init: %s\n", err.c_str()); return 1;
    }
    if (fleet.size() < 2) {
        std::fprintf(stderr, "split needs >=2 GPUs (have %u)\n", fleet.size()); return 1;
    }
    ie::LayerPlan plan = ie::LayerPlan::contiguous(cfg.n_layers, n_gpus);
    ie::Qwen35MoeSplitModel m;
    if (auto err = m.load(fleet, plan, g, cfg, max_ctx); !err.empty()) {
        std::fprintf(stderr, "model.load: %s\n", err.c_str()); return 1;
    }
    std::printf("  loaded in %.1f s on %u cards (%s)\n", (now_ms() - t0) / 1000.0, fleet.size(),
                fleet.dev(0).queue().get_device()
                    .get_info<sycl::info::device::name>().c_str());

    const uint32_t V = cfg.vocab;
    std::vector<sycl::half> logits(V);

    // Synthetic token ids — forward() cost is routing-pattern-independent (each
    // layer touches its own top-8 experts however the ids are sliced).
    std::vector<int32_t> ids(max_ctx);
    for (uint32_t i = 0; i < max_ctx; ++i)
        ids[i] = int32_t(((uint64_t(i) * 1103515245u) + 12345u) % V);

    // ---- Prefill warmup (untimed — first run JITs the prefill-shape kernels) ----
    if (prefill > 0)
        if (auto e = m.forward(ids.data(), prefill, 0, /*reset_kv=*/true, logits.data());
            !e.empty()) { std::fprintf(stderr, "prefill warmup: %s\n", e.c_str()); return 1; }

    // ---- Timed prefill (pp) ----
    const double tpp0 = now_ms();
    if (prefill > 0)
        if (auto e = m.forward(ids.data(), prefill, 0, /*reset_kv=*/true, logits.data());
            !e.empty()) { std::fprintf(stderr, "prefill forward: %s\n", e.c_str()); return 1; }
    const double pp_ms = now_ms() - tpp0;
    uint32_t pos = prefill;
    if (prefill > 0)
        std::printf("  %sPREFILL: %u tok in %.1f ms = %.1f tok/s%s\n",
                    B, prefill, pp_ms, prefill * 1000.0 / pp_ms, Z);

    // ---- Warmup decode (untimed) ----
    for (uint32_t s = 0; s < warmup; ++s) {
        if (auto e = m.forward(ids.data() + (pos % max_ctx), 1, pos, false, logits.data());
            !e.empty()) { std::fprintf(stderr, "warmup decode: %s\n", e.c_str()); return 1; }
        ++pos;
    }

    // ---- Timed decode (tg) ----
    const double tg0 = now_ms();
    for (uint32_t s = 0; s < decode_steps; ++s) {
        if (auto e = m.forward(ids.data() + (pos % max_ctx), 1, pos, false, logits.data());
            !e.empty()) { std::fprintf(stderr, "decode forward: %s\n", e.c_str()); return 1; }
        ++pos;
    }
    const double tg_ms   = now_ms() - tg0;
    const double ms_step = tg_ms / decode_steps;
    std::printf("  %sDECODE : %.3f ms/tok = %.2f tok/s%s  (%u steps, pos~%u)\n",
                B, ms_step, 1000.0 / ms_step, Z, decode_steps, pos);

    std::printf("\n# TSV\tgpus\tctx\tprefill\tpp_tok_s\tdecode_steps\ttg_ms_tok\ttg_tok_s\n");
    std::printf("# %u\t%u\t%u\t%.2f\t%u\t%.4f\t%.2f\n",
                n_gpus, max_ctx, prefill, prefill > 0 ? prefill * 1000.0 / pp_ms : 0.0,
                decode_steps, ms_step, 1000.0 / ms_step);
    return 0;
}
