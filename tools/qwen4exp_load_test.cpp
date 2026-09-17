// tools/qwen4exp_load_test.cpp — Qwen3.8-Flash-Next load + placement gate.
//
// Loads the real 4-shard UD-Q4_K_XL onto ONE B70 (v0 placement contract in
// include/ie/qwen4exp.hpp) and audits what actually landed: every device
// pointer non-null per the block schedule, device residency inside budget,
// host banks bound to the mmap, and a placement summary. This is the
// campaign's "model loaded" gate — forward comes on top.
//
// usage: ie-qwen4exp-load-test <model-00001-of-0000N.gguf> [gpu_ordinal]
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"
#include "ie/qwen4exp.hpp"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <ctime>

using namespace ie;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "\033[32mOK\033[0m" : "\033[31mFAIL\033[0m", what);
    if (!ok) ++g_fail;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.gguf> [gpu]\n", argv[0]); return 2; }
    const uint32_t ordinal = argc > 2 ? uint32_t(std::atoi(argv[2])) : 0;

    GgufReader g;
    if (auto e = g.open(argv[1]); !e.empty()) { std::fprintf(stderr, "open: %s\n", e.c_str()); return 1; }
    if (detect_arch(g) != ModelArch::kQwen4Exp) { std::fprintf(stderr, "not qwen4exp\n"); return 1; }
    Qwen4ExpConfig cfg;
    if (auto e = read_qwen4exp_config(g, cfg); !e.empty()) {
        std::fprintf(stderr, "config: %s\n", e.c_str()); return 1;
    }

    DeviceAllocator alloc;
    if (auto e = alloc.init("B70", ordinal); !e.empty()) {
        std::fprintf(stderr, "gpu init: %s\n", e.c_str()); return 1;
    }
    std::printf("\n\033[1mQwen3.8-Flash-Next load gate\033[0m  GPU %u: %s\n\n", ordinal,
                alloc.device().get_info<sycl::info::device::name>().c_str());

    Qwen4ExpModel m;
    const std::time_t t0 = std::time(nullptr);
    if (auto e = m.load(alloc, g, cfg); !e.empty()) {
        std::fprintf(stderr, "load: %s\n", e.c_str()); return 1;
    }
    const long secs = long(std::time(nullptr) - t0);

    std::printf("\033[1m1. placement audit\033[0m\n");
    uint32_t bad = 0;
    for (uint32_t L = 0; L < cfg.n_layers; ++L) {
        const auto& w = m.layer(L);
        bool ok = w.hc_attn_up && w.hc_attn_down && w.hc_attn_norm && w.hc_attn_inject &&
                  w.hc_ffn_up && w.hc_ffn_down && w.hc_ffn_norm && w.hc_ffn_inject &&
                  w.ffn_gate_inp && w.ffn_gate_inp_shexp && w.ffn_gate_shexp &&
                  w.ffn_up_shexp && w.ffn_down_shexp &&
                  w.gate_exps && w.up_exps && w.down_exps;
        if (cfg.is_full_attn(L))
            ok = ok && w.attn_q && w.attn_k && w.attn_v && w.attn_output &&
                 w.attn_q_norm && w.attn_k_norm && w.idx_q_proj && w.idx_k_proj &&
                 w.idx_q_norm && w.idx_k_norm && !w.attn_qkv && !w.ssm_out;
        else
            ok = ok && w.attn_qkv && w.attn_gate && w.ssm_alpha && w.ssm_beta &&
                 w.ssm_a && w.ssm_dt && w.ssm_conv && w.ssm_norm && w.ssm_out &&
                 !w.attn_q && !w.idx_q_proj;
        if (cfg.is_ple_layer(L))
            ok = ok && w.ple_key && w.ple_value && w.ple_conv &&
                 w.ple_norm_query && w.ple_norm_key && w.ple_norm_conv;
        else
            ok = ok && !w.ple_key;
        if (!ok) { ++bad; std::printf("      block %u placement wrong\n", L); }
    }
    check(bad == 0, "all 48 blocks match the device/host schedule");
    check(m.token_embd && m.lm_head && m.out_hc_up && m.out_hc_down && m.out_hc_norm,
          "globals resident (embd, lm_head, output_hc_*)");
    check(m.ple_table != nullptr, "PLE table bound host-side (26.8 GiB, IQ4_NL)");

    std::printf("\n\033[1m2. residency\033[0m\n");
    const double dev_gib  = m.device_bytes() / 1073741824.0;
    const double host_gib = m.host_bank_bytes() / 1073741824.0;
    std::printf("      device (F16/F32 weights): %.2f GiB   host (banks+PLE): %.2f GiB   load %lds\n",
                dev_gib, host_gib, secs);
    // Measured on the real file 2026-08-26: 9.35 GiB (the UD quant keeps ~4 GiB
    // of expert banks in Q8_0, which live HOST-side — less device-bound Q8_0
    // than the pre-load projection assumed).
    check(dev_gib > 8.5 && dev_gib < 12.0, "device residency within v0 envelope (8.5-12 GiB)");
    check(host_gib > 85.0 && host_gib < 100.0, "host banks + PLE within envelope (85-100 GiB)");

    // A trivial device round-trip so "loaded" means TOUCHED, not just mapped:
    // read back the first row of token_embd and require it finite + nonzero.
    std::printf("\n\033[1m3. device touch\033[0m\n");
    {
        std::vector<sycl::half> row(cfg.hidden);
        alloc.queue().memcpy(row.data(), m.token_embd, cfg.hidden * sizeof(sycl::half)).wait();
        double s = 0;
        for (auto v : row) s += std::abs(double(float(v)));
        check(s > 0.0 && s < 1e9, "token_embd row 0 readable from device, finite, nonzero");
    }

    std::printf("\n\033[1m%s\033[0m  (%d failure%s)\n\n",
                g_fail ? "\033[31mGATE FAILED\033[0m" : "\033[32mGATE PASSED\033[0m",
                g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
