// tools/ie_bench.cpp — Phase 9 gate measurement.
//
// Measures both prompt-processing (PP, prefill tok/s) and token-generation
// (TG, decode tok/s) at multiple ctx lengths, on a single model load.
// Sweeps a list of prefill lengths; for each length:
//   1. Reset KV + DeltaNet state, set start_pos = 0
//   2. Run a single forward(T = T_pp) → PP latency / tok/s
//   3. Run N decode steps of forward(T = 1, start_pos += 1) → TG latency / tok/s
//   4. Reset for the next sweep point
//
// Output is one row per prefill length, both human-readable and a TSV trailer
// suitable for piping to a plotter.
//
// Usage:
//   ie-bench --gguf <path> [--ctx 32768] [--prefill 1,256,1024,4096,16384]
//            [--decode 32] [--warmup 2]
//
// Discipline: this tool loads the full ~22 GB model; do not run concurrently
// with another model-loading process. KV cache memory at 32k ctx is ~5.2 GB
// in addition to the model weights.

#include "ie/allocator.hpp"
#include "ie/deltanet_state.hpp"
#include "ie/dense_transformer.hpp"
#include "ie/gguf.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/kv_cache.hpp"
#include "ie/model_config.hpp"
#include "ie/ops.hpp"
#include "ie/qwen35_dense.hpp"
#include "ie/qwen36.hpp"
#include "ie/qwen3moe.hpp"
#include "ie/gptoss.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr const char* B = "\033[1m";
constexpr const char* G = "\033[32m";
constexpr const char* Y = "\033[33m";
constexpr const char* Z = "\033[0m";

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::vector<uint32_t> parse_csv_uint(const std::string& s) {
    std::vector<uint32_t> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(uint32_t(std::atoi(tok.c_str())));
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf_path = "/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf";
    uint32_t max_ctx = 32768 + 64;       // enough for the largest prefill + some decode
    std::vector<uint32_t> prefill_pts = {1, 256, 1024, 4096, 16384};
    uint32_t decode_steps = 32;
    uint32_t warmup_decode = 4;
    bool fastforward = false;            // skip real prefill, fake start_pos via kv.set_length
    bool int8_kv = false;                // use INT8 KV cache (FA-2 decode path only)
    bool profile = false;                // print per-section profile after each forward
    bool kprofile = false;               // per-kernel timing for the prefill forward
    bool kprofile_decode = false;        // per-kernel timing for ONE decode (T=1) step
    uint32_t kprofile_verify = 0;        // per-kernel timing for ONE forward(T=K) — the
                                         // spec-decode VERIFY shape (small-T batched path)
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf"    && i + 1 < argc) gguf_path     = argv[++i];
        else if (a == "--ctx"     && i + 1 < argc) max_ctx       = std::atoi(argv[++i]);
        else if (a == "--prefill" && i + 1 < argc) prefill_pts   = parse_csv_uint(argv[++i]);
        else if (a == "--decode"  && i + 1 < argc) decode_steps  = std::atoi(argv[++i]);
        else if (a == "--warmup"  && i + 1 < argc) warmup_decode = std::atoi(argv[++i]);
        else if (a == "--fastforward")              fastforward   = true;
        else if (a == "--int8-kv")                  int8_kv       = true;
        else if (a == "--profile")                  profile       = true;
        else if (a == "--kprofile")                 kprofile      = true;
        else if (a == "--kprofile-decode")          kprofile_decode = true;
        else if (a == "--kprofile-verify" && i + 1 < argc) kprofile_verify = std::atoi(argv[++i]);
    }

    std::printf("%sie-bench%s — Phase 9 gate (PP / TG @ ctx sweep)\n", B, Z);
    std::printf("  gguf       : %s\n", gguf_path.c_str());
    std::printf("  ctx        : %u\n", max_ctx);
    std::printf("  prefill_pts:");
    for (auto p : prefill_pts) std::printf(" %u", p);
    std::printf("\n  decode_steps: %u (+%u warmup)\n", decode_steps, warmup_decode);

    // Per-kernel profiling reads SYCL event timestamps → the queue must be created
    // with enable_profiling, which is now OPT-IN (default OFF; it taxes every submit
    // ~1.76 ms on the in-order L0 queue — the decode submission-bound regression).
    // Self-enable it ONLY for the kprofile paths so plain PP/TG bench the fast path.
    if (kprofile || kprofile_decode || kprofile_verify) setenv("IE_QUEUE_PROFILING", "1", 1);

    // 1. Open GGUF + load model
    ie::GgufReader g;
    if (auto err = g.open(gguf_path); !err.empty()) {
        std::fprintf(stderr, "gguf open: %s\n", err.c_str()); return 1;
    }
    ie::DeviceAllocator alloc;
    if (auto err = alloc.init(); !err.empty()) {
        std::fprintf(stderr, "alloc: %s\n", err.c_str()); return 1;
    }
    auto& q = alloc.queue();
    std::printf("  device     : %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    // Arch-tagged dispatch (P2 Task 5): qwen35moe → QwenModel (crown path,
    // unchanged), qwen3 → DenseModel (all layers full-attn, no DeltaNet).
    const ie::ModelArch arch = ie::detect_arch(g);
    if (arch == ie::ModelArch::kUnknown) {
        const auto* a = g.find_kv("general.architecture");
        std::fprintf(stderr, "unsupported architecture: %.*s\n",
                     a ? int(a->as_string().size()) : 9,
                     a ? a->as_string().data() : "<missing>");
        return 1;
    }
    const bool is_dense = ie::is_dense_arch(arch);
    const bool is_q35   = (arch == ie::ModelArch::kQwen35Dense);
    const bool is_q3moe = (arch == ie::ModelArch::kQwen3Moe);   // qwen3moe (Coder-30B-A3B)
    const bool is_gptoss = (arch == ie::ModelArch::kGptOss);    // gpt-oss (OpenAI MoE)
    std::printf("  arch       : %s\n", arch == ie::ModelArch::kLlama3 ? "llama (dense)"
                                     : is_dense ? "qwen3 (dense)"
                                     : is_q35   ? "qwen35 (dense-hybrid)"
                                     : is_q3moe ? "qwen3moe (dense-attn + top-k MoE)"
                                     : is_gptoss ? "gpt-oss (OpenAI MoE)" : "qwen35moe");
    if (is_dense && (profile || kprofile)) {
        std::fprintf(stderr, "note: --profile/--kprofile are crown-model only;"
                             " ignored for the dense arch (use --kprofile-decode"
                             " for a per-kernel DECODE profile on any arch)\n");
        profile = kprofile = false;
    }
    // --kprofile-decode rides the arch-agnostic KernelProfiler (g_profiler +
    // ie::ps wrappers) — works for both crown and dense. No model.set_profile
    // dependency, so no crown-only guard needed.

    const double t0 = now_ms();
    std::printf("  loading model...\n");
    ie::QwenConfig  cfg;
    ie::QwenModel   model;
    ie::DenseConfig dcfg;
    ie::DenseModel  dmodel;
    ie::Qwen35Config     qcfg;
    ie::Qwen35DenseModel q35model;
    ie::Qwen3MoeConfig   mcfg;
    ie::Qwen3MoeModel    mmodel;
    ie::GptOssConfig     gcfg;
    ie::GptOssModel      gmodel;
    uint32_t vocab = 0;
    if (is_q3moe) {
        if (auto err = ie::read_qwen3moe_config(g, mcfg); !err.empty()) {
            std::fprintf(stderr, "read_qwen3moe_config: %s\n", err.c_str()); return 1;
        }
        if (auto err = mmodel.load(alloc, g, mcfg); !err.empty()) {
            std::fprintf(stderr, "model.load: %s\n", err.c_str()); return 1;
        }
        vocab = mcfg.dense.vocab;
    } else if (is_gptoss) {
        if (auto err = ie::read_gptoss_config(g, gcfg); !err.empty()) {
            std::fprintf(stderr, "read_gptoss_config: %s\n", err.c_str()); return 1;
        }
        if (auto err = gmodel.load(alloc, g, gcfg); !err.empty()) {
            std::fprintf(stderr, "model.load: %s\n", err.c_str()); return 1;
        }
        vocab = gcfg.dense.vocab;
    } else if (is_dense) {
        if (auto err = ie::read_dense_config_auto(g, dcfg); !err.empty()) {
            std::fprintf(stderr, "read_dense_config: %s\n", err.c_str()); return 1;
        }
        if (auto err = dmodel.load(alloc, g, dcfg); !err.empty()) {
            std::fprintf(stderr, "model.load: %s\n", err.c_str()); return 1;
        }
        vocab = dcfg.vocab;
    } else if (is_q35) {
        if (auto err = ie::read_qwen35_config(g, qcfg); !err.empty()) {
            std::fprintf(stderr, "read_qwen35_config: %s\n", err.c_str()); return 1;
        }
        if (auto err = q35model.load(alloc, g, qcfg); !err.empty()) {
            std::fprintf(stderr, "model.load: %s\n", err.c_str()); return 1;
        }
        vocab = qcfg.dense.vocab;
    } else {
        if (auto err = model.load(alloc, g, cfg); !err.empty()) {
            std::fprintf(stderr, "model.load: %s\n", err.c_str()); return 1;
        }
        vocab = cfg.vocab;
    }
    std::printf("  loaded in %.1f s\n", (now_ms() - t0) / 1000.0);

    // 2. Caches sized for max_ctx. Dense: KV for every layer; DeltaNetState
    //    stays uninit (reset() no-ops).
    const uint32_t n_kv_layers =
        is_q3moe ? mcfg.dense.n_layers              // every layer is full-attn
      : is_gptoss ? gcfg.dense.n_layers             // gpt-oss: every layer full KV (SWA is a mask)
      : is_dense ? dcfg.n_layers
      : is_q35   ? qcfg.n_transformer_layers() / qcfg.full_attn_interval
                 : cfg.n_layers / cfg.full_attn_interval;
    ie::KvCache kv;
    {
        ie::KvCacheConfig kvcfg{};
        kvcfg.n_layers_full = n_kv_layers;
        kvcfg.n_kv_heads    = is_q3moe ? mcfg.dense.n_kv_heads : is_gptoss ? gcfg.dense.n_kv_heads : is_dense ? dcfg.n_kv_heads : is_q35 ? qcfg.dense.n_kv_heads : cfg.n_kv_heads;
        kvcfg.max_ctx       = max_ctx;
        kvcfg.head_dim      = is_q3moe ? mcfg.dense.head_dim : is_gptoss ? gcfg.dense.head_dim : is_dense ? dcfg.head_dim : is_q35 ? qcfg.dense.head_dim : cfg.head_dim;
        kvcfg.use_int8      = int8_kv;
        if (auto err = kv.init(alloc, kvcfg); !err.empty()) {
            std::fprintf(stderr, "kv: %s\n", err.c_str()); return 1;
        }
        std::printf("  kv mode    : %s\n", int8_kv ? "INT8 + fp16 shadow" : "fp16");
    }
    ie::DeltaNetState dn;
    if (is_q35) {
        const uint32_t n_tx   = qcfg.n_transformer_layers();
        const uint32_t L_lin  = n_tx - n_tx / qcfg.full_attn_interval;
        const uint32_t conv_ch = qcfg.ssm_inner + 2u * qcfg.ssm_n_k_heads * qcfg.ssm_state;
        if (auto err = dn.init(alloc, ie::DeltaNetStateConfig{
                L_lin, qcfg.ssm_n_v_heads, qcfg.ssm_state, qcfg.ssm_state,
                conv_ch, qcfg.ssm_conv_kernel}); !err.empty()) {
            std::fprintf(stderr, "dn: %s\n", err.c_str()); return 1;
        }
    } else if (!is_dense && !is_q3moe && !is_gptoss) {
        const uint32_t L_full = cfg.n_layers / cfg.full_attn_interval;
        const uint32_t L_lin  = cfg.n_layers - L_full;
        if (auto err = dn.init(alloc, ie::DeltaNetStateConfig{
                L_lin, cfg.ssm_n_v_heads, cfg.ssm_head_dim, cfg.ssm_head_dim,
                cfg.ssm_inner * 2, cfg.ssm_conv_kernel}); !err.empty()) {
            std::fprintf(stderr, "dn: %s\n", err.c_str()); return 1;
        }
    }

    // 3. Workspace sized for the largest prefill we'll run
    uint32_t max_T_ws = 1;
    for (auto p : prefill_pts) max_T_ws = std::max(max_T_ws, p);
    {
        auto wserr = is_q3moe ? mmodel.ensure_workspace(max_T_ws)
                   : is_gptoss ? gmodel.ensure_workspace(max_T_ws)
                   : is_dense ? dmodel.ensure_workspace(max_T_ws)
                   : is_q35   ? q35model.ensure_workspace(max_T_ws)
                              : model.ensure_workspace(max_T_ws);
        if (!wserr.empty()) { std::fprintf(stderr, "ws: %s\n", wserr.c_str()); return 1; }
        auto aperr = is_q3moe ? mmodel.ensure_attn_partials(max_ctx)
                   : is_gptoss ? gmodel.ensure_attn_partials(max_ctx)
                   : is_dense ? dmodel.ensure_attn_partials(max_ctx)
                   : is_q35   ? q35model.ensure_attn_partials(max_ctx)
                              : model.ensure_attn_partials(max_ctx);
        if (!aperr.empty()) { std::fprintf(stderr, "attn_partials: %s\n", aperr.c_str()); return 1; }
    }

    // 4. Persistent device buffers (sized for the largest prefill)
    auto* d_logits  = sycl::malloc_device<sycl::half>(vocab, q);
    auto* d_ids_buf = sycl::malloc_device<int32_t>(max_ctx, q);

    // Arch-agnostic forward shim (same shape as Engine::forward_step).
    auto forward = [&](const int32_t* d_in, uint32_t T, uint32_t pos) {
        return is_q3moe ? mmodel.forward(q, d_in, T, pos, kv, d_logits)
             : is_gptoss ? gmodel.forward(q, d_in, T, pos, kv, d_logits)
             : is_dense ? dmodel.forward(q, d_in, T, pos, kv, d_logits)
             : is_q35   ? q35model.forward(q, d_in, T, pos, kv, dn, d_logits)
                        : model.forward(q, d_in, T, pos, kv, dn, d_logits);
    };

    // 5. Build a synthetic prefill of arbitrary token IDs. We don't need a
    // real prompt — the cost of forward() is the same regardless of the
    // routing pattern (each layer touches its own active 8 experts however
    // you slice it). Use a stable mod-pattern that stays inside vocab.
    std::vector<int32_t> ids(max_ctx);
    for (uint32_t i = 0; i < max_ctx; ++i) {
        ids[i] = int32_t(((i * 1103515245u) + 12345u) % vocab);
    }
    q.memcpy(d_ids_buf, ids.data(), uint64_t(max_ctx) * sizeof(int32_t)).wait();

    // 6. Run the sweep. Each sweep point: reset → prefill → decode.
    std::printf("\n%s%-8s  %-12s  %-10s  %-12s  %-10s%s\n",
                B, "Tpp", "PP_ms", "PP_tok/s", "TG_ms/tok", "TG_tok/s", Z);
    std::vector<std::tuple<uint32_t, double, double, double, double>> rows;

    for (uint32_t T_pp : prefill_pts) {
        if (T_pp + warmup_decode + decode_steps > max_ctx) {
            std::fprintf(stderr, "skip T_pp=%u: exceeds max_ctx=%u\n", T_pp, max_ctx);
            continue;
        }

        // Reset state
        kv.reset();
        dn.reset(q);
        uint32_t cur_pos = 0;

        // ---- Prefill (or fastforward — fake start_pos for long-ctx decode bench) ----
        double pp_ms;
        double pp_tok_per_s;
        if (fastforward) {
            // Just bump KV length per layer to T_pp; cache contents stay zero/stale,
            // but the FA-2 / naive attention timing is identical.
            for (uint32_t L = 0; L < n_kv_layers; ++L) {
                kv.set_length(L, T_pp);
            }
            cur_pos = T_pp;
            pp_ms = 0.0;
            pp_tok_per_s = 0.0;
        } else {
            if (profile) model.set_profile(true);
            ie::KernelProfiler kprof;
            if (kprofile) { ie::g_profiler = &kprof; kprof.begin_step(); }
            q.wait();
            const double t_pp0 = now_ms();
            forward(d_ids_buf, T_pp, cur_pos).wait();
            pp_ms = now_ms() - t_pp0;
            cur_pos += T_pp;
            pp_tok_per_s = (T_pp > 0) ? (T_pp * 1000.0 / pp_ms) : 0.0;
            if (kprofile) {
                ie::g_profiler = nullptr;
                auto stats = kprof.harvest();
                std::sort(stats.begin(), stats.end(),
                          [](const auto& a, const auto& b) {
                              return a.total_ns > b.total_ns;
                          });
                double tot = 0;
                for (auto& s : stats) tot += s.total_ms();
                std::printf("\n  --- kernel profile @ prefill T_pp=%u "
                            "(GPU total %.1f ms) ---\n", T_pp, tot);
                for (auto& s : stats) {
                    std::printf("    %-26s %8.2f ms  (%5.1f%%)  %5u calls  "
                                "%7.3f avg\n",
                                s.name.c_str(), s.total_ms(),
                                100.0 * s.total_ms() / tot, s.calls, s.avg_ms());
                }
            }
            if (profile) {
                std::printf("\n  --- profile @ prefill T_pp=%u ---\n", T_pp);
                model.dump_profile(stdout);
                model.set_profile(false);
            }
        }

        // ---- Decode warmup (don't time these — caches/code may still be cold) ----
        for (uint32_t s = 0; s < warmup_decode; ++s) {
            int32_t* one = d_ids_buf + (cur_pos % max_ctx);  // pick any token id
            forward(one, 1, cur_pos).wait();
            cur_pos += 1;
        }

        // ---- Decode timed steps ----
        if (profile) model.set_profile(true);
        q.wait();
        const double t_tg0 = now_ms();
        for (uint32_t s = 0; s < decode_steps; ++s) {
            int32_t* one = d_ids_buf + (cur_pos % max_ctx);
            forward(one, 1, cur_pos).wait();
            cur_pos += 1;
        }
        const double tg_ms_total = now_ms() - t_tg0;
        const double tg_ms_per_step = tg_ms_total / decode_steps;
        const double tg_tok_per_s   = 1000.0 / tg_ms_per_step;

        // ---- Per-kernel DECODE profile (one extra fully-warm T=1 step) ----
        // Runs AFTER prefill + warmup + timed decode, so JIT is discarded and
        // caches are hot. Arch-agnostic: g_profiler + ie::ps wrappers only.
        if (kprofile_decode) {
            ie::KernelProfiler kprof;
            ie::g_profiler = &kprof;
            kprof.begin_step();
            q.wait();
            int32_t* one = d_ids_buf + (cur_pos % max_ctx);
            forward(one, 1, cur_pos).wait();
            cur_pos += 1;
            ie::g_profiler = nullptr;
            auto stats = kprof.harvest();
            std::sort(stats.begin(), stats.end(),
                      [](const auto& a, const auto& b) {
                          return a.total_ns > b.total_ns;
                      });
            double tot = 0;
            for (auto& s : stats) tot += s.total_ms();
            std::printf("\n  --- DECODE kernel profile @ pos=%u (T=1, one warm "
                        "step; GPU total %.3f ms) ---\n", cur_pos - 1, tot);
            std::printf("    %-26s %10s  %7s  %6s  %10s\n",
                        "kernel", "ms/token", "%", "calls", "avg_ms");
            for (auto& s : stats) {
                std::printf("    %-26s %10.4f  %6.1f%%  %6u  %10.4f\n",
                            s.name.c_str(), s.total_ms(),
                            100.0 * s.total_ms() / tot, s.calls, s.avg_ms());
            }
            std::printf("    %-26s %10.4f  %6.1f%%\n", "TOTAL", tot, 100.0);
        }

        // ---- Per-kernel VERIFY profile: ONE forward(T=K) (the spec-decode
        // verify shape — small-T batched int-dot path). Lets us localize where
        // the verify cost lives vs a T=1 decode (the spec round's dominant cost).
        if (kprofile_verify >= 2 && kprofile_verify <= 16) {
            const uint32_t K = kprofile_verify;
            ie::KernelProfiler kprof;
            ie::g_profiler = &kprof;
            kprof.begin_step();
            q.wait();
            forward(d_ids_buf, K, cur_pos).wait();
            cur_pos += K;
            ie::g_profiler = nullptr;
            auto stats = kprof.harvest();
            std::sort(stats.begin(), stats.end(),
                      [](const auto& a, const auto& b) { return a.total_ns > b.total_ns; });
            double tot = 0;
            for (auto& s : stats) tot += s.total_ms();
            std::printf("\n  --- VERIFY kernel profile @ pos=%u (T=%u, one warm "
                        "forward; GPU total %.3f ms) ---\n", cur_pos - K, K, tot);
            std::printf("    %-26s %10s  %7s  %6s  %10s\n",
                        "kernel", "ms", "%", "calls", "avg_ms");
            for (auto& s : stats)
                std::printf("    %-26s %10.4f  %6.1f%%  %6u  %10.4f\n",
                            s.name.c_str(), s.total_ms(),
                            100.0 * s.total_ms() / tot, s.calls, s.avg_ms());
            std::printf("    %-26s %10.4f  %6.1f%%\n", "TOTAL", tot, 100.0);
        }

        if (profile) {
            std::printf("\n  --- profile @ T_pp=%u (%u decode steps) ---\n",
                        T_pp, decode_steps);
            model.dump_profile(stdout);
            model.set_profile(false);  // turn off for next sweep point
        }

        std::printf("  %-8u  %-12.2f  %-10.1f  %-12.2f  %-10.1f\n",
                    T_pp, pp_ms, pp_tok_per_s, tg_ms_per_step, tg_tok_per_s);
        std::fflush(stdout);
        rows.emplace_back(T_pp, pp_ms, pp_tok_per_s, tg_ms_per_step, tg_tok_per_s);
    }

    // TSV trailer for plotting.
    std::printf("\n# TSV: T_pp\tpp_ms\tpp_tok_s\ttg_ms_tok\ttg_tok_s\n");
    for (auto& r : rows) {
        std::printf("# %u\t%.3f\t%.2f\t%.4f\t%.2f\n",
                    std::get<0>(r), std::get<1>(r), std::get<2>(r),
                    std::get<3>(r), std::get<4>(r));
    }

    sycl::free(d_logits, q);
    sycl::free(d_ids_buf, q);
    return 0;
}
