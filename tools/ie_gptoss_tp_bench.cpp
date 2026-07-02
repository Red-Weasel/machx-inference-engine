// tools/ie_gptoss_tp_bench.cpp — gpt-oss TENSOR-PARALLEL decode benchmark (>=2 GPUs).
//
// The only multi-GPU decode-tok/s tool in the tree: ie-bench is single-GPU only.
// Loads a gpt-oss GGUF (multi-shard ok) across N cards via GptOssTpModel, runs a
// T=1 prefill to a target ctx, then times `--decode` autoregressive T=1 steps →
// decode tok/s. With IE_GPTOSS_TP_TIMING=1 it ALSO prints the per-section
// attribution (attention / the two host-bounced all-reduces / the card-0 router
// round-trip / MoE gemv incl. host-RAM spill / lm_head) — Step-0 of the 120b
// decode-optimization: is decode comm-bound, spill-bound, or kernel-bound?
//
// Usage:
//   ie-gptoss-tp-bench --gguf <shard1.gguf> [--gpus 2] [--ctx 4096]
//                      [--prefill 512] [--decode 64] [--warmup 8]
//
// Step-0 protocol (run on a CLEAN box — GPU idle, CPU quiet — see the all-reduce
// is CPU-bounced so CPU contention fakes a slowdown):
//   true tok/s :  ie-gptoss-tp-bench --gguf <s1> --gpus 2 --prefill 512 --decode 64
//   breakdown  :  IE_GPTOSS_TP_TIMING=1 ie-gptoss-tp-bench ... (adds barriers)
//   spill A/B  :  IE_GPTOSS_TP_RESERVE_GB=5 ...  vs  IE_GPTOSS_TP_RESERVE_GB=2 ...
//   comm A/B   :  IE_GPTOSS_TP_REPLICATE_ATTN=1 ...   (drops 1 of 2 all-reduces/layer)

#include "ie/allocator.hpp"
#include "ie/gguf.hpp"
#include "ie/gptoss.hpp"
#include "ie/gptoss_tp.hpp"
#include "ie/model_config.hpp"

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
        "/home/weezy/models/lmstudio-community/gpt-oss-120b-GGUF/"
        "gpt-oss-120b-MXFP4-00001-of-00002.gguf";
    uint32_t n_gpus       = 2;
    uint32_t max_ctx      = 4096;
    uint32_t prefill      = 512;
    uint32_t decode_steps = 64;
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

    const bool tp_timing = std::getenv("IE_GPTOSS_TP_TIMING") != nullptr;
    std::printf("%sie-gptoss-tp-bench%s — multi-GPU decode tok/s + comm breakdown\n", B, Z);
    std::printf("  gguf    : %s\n", gguf_path.c_str());
    std::printf("  gpus    : %u\n  ctx     : %u\n  prefill : %u\n  decode  : %u (+%u warmup)\n",
                n_gpus, max_ctx, prefill, decode_steps, warmup);
    std::printf("  timing  : %s   replicate_attn: %s   reserve_gb(env): %s\n",
                tp_timing ? "ON (barriers added)" : "off",
                std::getenv("IE_GPTOSS_TP_REPLICATE_ATTN") ? "ON (Phase-1)" : "off (head-shard)",
                std::getenv("IE_GPTOSS_TP_RESERVE_GB") ? std::getenv("IE_GPTOSS_TP_RESERVE_GB")
                                                       : "default(5/3)");

    ie::GgufReader g;
    if (auto err = g.open(gguf_path); !err.empty()) {
        std::fprintf(stderr, "gguf open: %s\n", err.c_str()); return 1;
    }
    if (ie::detect_arch(g) != ie::ModelArch::kGptOss) {
        std::fprintf(stderr, "not a gpt-oss gguf\n"); return 1;
    }
    ie::GptOssConfig gcfg;
    if (auto err = ie::read_gptoss_config(g, gcfg); !err.empty()) {
        std::fprintf(stderr, "read_gptoss_config: %s\n", err.c_str()); return 1;
    }

    if (prefill + warmup + decode_steps + 4 > max_ctx) {
        max_ctx = prefill + warmup + decode_steps + 4;
        std::printf("  ctx grew to %u (prefill + decode slack)\n", max_ctx);
    }

    const double t0 = now_ms();
    std::printf("  loading (TP, %u cards)...\n", n_gpus);
    ie::DeviceFleet fleet;
    if (auto err = fleet.init(n_gpus); !err.empty()) {
        std::fprintf(stderr, "fleet.init: %s\n", err.c_str()); return 1;
    }
    if (fleet.size() < 2) {
        std::fprintf(stderr, "TP needs >=2 GPUs (have %u)\n", fleet.size()); return 1;
    }
    ie::GptOssTpModel gtp;
    if (auto err = gtp.load(fleet, g, gcfg, max_ctx); !err.empty()) {
        std::fprintf(stderr, "gtp.load: %s\n", err.c_str()); return 1;
    }
    std::printf("  loaded in %.1f s on %u cards (%s)\n", (now_ms() - t0) / 1000.0, fleet.size(),
                fleet.dev(0).queue().get_device()
                    .get_info<sycl::info::device::name>().c_str());

    const uint32_t V = gcfg.dense.vocab;
    std::vector<sycl::half> logits(V);

    // Synthetic token ids — forward() cost is routing-pattern-independent (each
    // layer touches its own top-4 experts however the ids are sliced).
    std::vector<int32_t> ids(max_ctx);
    for (uint32_t i = 0; i < max_ctx; ++i)
        ids[i] = int32_t(((uint64_t(i) * 1103515245u) + 12345u) % V);

    // ---- Prefill (BATCHED T>1 — the now-default correct path) ----
    if (tp_timing) gtp.reset_tp_timing();
    const double tpp0 = now_ms();
    if (prefill > 0)
        if (auto m = gtp.forward(ids.data(), prefill, 0, /*reset_kv=*/true, logits.data());
            !m.empty()) { std::fprintf(stderr, "prefill forward: %s\n", m.c_str()); return 1; }
    const double pp_ms = now_ms() - tpp0;
    uint32_t pos = prefill;
    if (prefill > 0) {
        std::printf("  prefill %u tok in %.1f ms (%.1f tok/s, BATCHED)\n",
                    prefill, pp_ms, prefill * 1000.0 / pp_ms);
        if (tp_timing) { std::printf("  [PREFILL section breakdown]\n"); gtp.report_tp_timing(1); }
    }

    // ---- Warmup decode (untimed — JIT / caches still settling) ----
    for (uint32_t s = 0; s < warmup; ++s) {
        if (auto m = gtp.forward(ids.data() + (pos % max_ctx), 1, pos, false, logits.data());
            !m.empty()) { std::fprintf(stderr, "warmup forward: %s\n", m.c_str()); return 1; }
        ++pos;
    }

    // ---- Timed decode ----
    if (tp_timing) gtp.reset_tp_timing();
    const double tg0 = now_ms();
    for (uint32_t s = 0; s < decode_steps; ++s) {
        if (auto m = gtp.forward(ids.data() + (pos % max_ctx), 1, pos, false, logits.data());
            !m.empty()) { std::fprintf(stderr, "decode forward: %s\n", m.c_str()); return 1; }
        ++pos;
    }
    const double tg_ms   = now_ms() - tg0;
    const double ms_step = tg_ms / decode_steps;
    std::printf("\n  %sDECODE: %.3f ms/tok  =  %.2f tok/s%s  (%u steps, pos~%u)\n",
                B, ms_step, 1000.0 / ms_step, Z, decode_steps, pos);

    if (tp_timing) gtp.report_tp_timing(decode_steps);

    std::printf("\n# TSV\tgpus\tctx\tprefill\tdecode_steps\ttg_ms_tok\ttg_tok_s\n");
    std::printf("# %u\t%u\t%u\t%u\t%.4f\t%.2f\n",
                n_gpus, max_ctx, prefill, decode_steps, ms_step, 1000.0 / ms_step);
    return 0;
}
