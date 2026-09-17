// tools/glm5next_load_test.cpp — GLM-5.3-Flash load + placement gate (P1).
//
// Loads the real 6-shard UD-Q4_K_XL's non-expert set onto ONE B70 (v0
// placement contract in include/ie/glm5next.hpp) and audits what actually
// landed: every device pointer non-null per the block schedule, device
// residency inside budget, host banks bound to the mmap, and a placement
// summary. This is the port's "model loaded" gate — forward comes on top.
//
// usage: ie-glm5next-load-test <model-00001-of-00006.gguf> [gpu_ordinal]
#include "ie/gguf.hpp"
#include "ie/glm5next.hpp"
#include "ie/model_config.hpp"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <map>
#include <string>

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
    if (detect_arch(g) != ModelArch::kGlm5Next) { std::fprintf(stderr, "not glm5next\n"); return 1; }
    Glm5NextConfig cfg;
    if (auto e = read_glm5next_config(g, cfg); !e.empty()) {
        std::fprintf(stderr, "config: %s\n", e.c_str()); return 1;
    }

    DeviceAllocator alloc;
    if (auto e = alloc.init("B70", ordinal); !e.empty()) {
        std::fprintf(stderr, "gpu init: %s\n", e.c_str()); return 1;
    }
    std::printf("\n\033[1mGLM-5.3-Flash load gate\033[0m  GPU %u: %s\n\n", ordinal,
                alloc.device().get_info<sycl::info::device::name>().c_str());

    Glm5NextModel m;
    const std::time_t t0 = std::time(nullptr);
    if (auto e = m.load(alloc, g, cfg); !e.empty()) {
        std::fprintf(stderr, "load: %s\n", e.c_str()); return 1;
    }
    const long secs = long(std::time(nullptr) - t0);

    std::printf("\033[1m1. placement audit\033[0m\n");
    const uint32_t n_tf = cfg.n_transformer_layers();
    uint32_t bad = 0;
    for (uint32_t L = 0; L < n_tf; ++L) {
        const auto& w = m.layer(L);
        bool ok = w.hc_attn_fn && w.hc_attn_base && w.hc_attn_scale &&
                  w.hc_ffn_fn && w.hc_ffn_base && w.hc_ffn_scale &&
                  w.attn_norm && w.ffn_norm;
        if (cfg.is_full_attn(L)) {
            ok = ok && w.q_a && w.q_a_norm && w.q_b && w.kv_a && w.kv_a_norm &&
                 w.k_b && w.v_b && w.attn_out &&
                 w.idx_k && w.idx_k_norm_w && w.idx_k_norm_b && w.idx_q_b &&
                 w.idx_proj && w.idx_ape && w.idx_gate &&
                 !w.kda_q && !w.conv_q && !w.ssm_a;
        } else {
            ok = ok && w.kda_q && w.kda_k && w.kda_v && w.kda_o &&
                 w.conv_q && w.conv_k && w.conv_v && w.ssm_a && w.dt_bias &&
                 w.f_a && w.f_b && w.g_a && w.g_b && w.beta && w.ssm_norm &&
                 !w.q_a && !w.idx_k;
        }
        if (cfg.is_dense_layer(L)) {
            ok = ok && w.ffn_gate && w.ffn_up && w.ffn_down &&
                 !w.router && !w.gate_exps;
        } else {
            ok = ok && w.router && w.probs_bias &&
                 w.shexp_gate && w.shexp_up && w.shexp_down &&
                 w.gate_exps && w.up_exps && w.down_exps && !w.ffn_gate;
        }
        if (!ok) { ++bad; std::printf("      block %2u placement WRONG\n", L); }
    }
    check(bad == 0, "all 45 transformer blocks bound per schedule");
    check(m.token_embd != nullptr, "token_embd on device");
    check(bool(m.lm_head) && m.output_norm != nullptr, "lm_head + output_norm on device");
    {
        const auto& mtp = m.layer(n_tf);
        check(mtp.nextn_eh && mtp.nextn_enorm && mtp.nextn_hnorm && mtp.nextn_shn &&
              mtp.gate_exps && mtp.up_exps && mtp.down_exps,
              "MTP block bound as host views (not uploaded)");
    }

    std::printf("\n\033[1m2. residency\033[0m\n");
    const double dev_gib  = m.device_bytes() / 1073741824.0;
    const double host_gib = m.host_bank_bytes() / 1073741824.0;
    std::printf("      device %.3f GiB | host banks %.3f GiB | load %ld s\n",
                dev_gib, host_gib, secs);
    // v0 F16-everything projection: ~19 GiB. A big drift means a binding or
    // layout bug, not a tuning choice — hence the band.
    check(dev_gib > 15.0 && dev_gib < 24.0, "device residency within the v0 F16 band (15-24 GiB)");
    // 44 expert-bank triples (43 MoE + MTP) at UD-Q4 sizes.
    check(host_gib > 150.0 && host_gib < 185.0, "host banks within the expected band (150-185 GiB)");

    // Where the bytes sit, summarized by tensor family.
    std::printf("\n\033[1m3. placement summary (top families)\033[0m\n");
    std::map<std::string, uint64_t> fam;
    for (const auto& r : m.placement()) {
        std::string f = r.name;
        if (f.rfind("blk.", 0) == 0) f = f.substr(f.find('.', 4) + 1);
        f += std::string(" [") + r.where + "]";
        fam[f] += r.bytes;
    }
    std::multimap<uint64_t, std::string, std::greater<>> by_size;
    for (const auto& [k, v] : fam) by_size.emplace(v, k);
    int shown = 0;
    for (const auto& [v, k] : by_size) {
        if (++shown > 14) break;
        std::printf("      %-44s %8.3f GiB\n", k.c_str(), v / 1073741824.0);
    }

    std::printf("\n\033[1m%s\033[0m  (%d failure%s)\n\n",
                g_fail ? "\033[31mGATE FAILED\033[0m" : "\033[32mGATE PASSED\033[0m",
                g_fail, g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
