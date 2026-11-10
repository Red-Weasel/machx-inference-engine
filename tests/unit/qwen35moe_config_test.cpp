// tests/unit/qwen35moe_config_test.cpp -- read_qwen35moe_config (2026-11-08). Host-only, header parse, no GPU.
//   default run: every qwen35moe file in $IE_Q35M_CONFIG_FILES (':'-separated 35B-A3B-shaped GGUFs) must read
//   back QwenConfig's defaults EXACTLY -- the reader leaves the daily 35B's load unchanged. Absent files are skipped.
//   with arguments: prints the shape each named file reads as (the generic mini oracle's qwen35moe mini, for one).
#include "ie/gguf.hpp"
#include "ie/qwen36.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {
std::string show(const ie::QwenConfig& c) {
    char b[512];
    std::snprintf(b, sizeof(b), "layers %u hidden %u heads %u/%u head_dim %u rope %u@%.0f experts %u of %u ffn %u interval %u "
                  "ssm %u (v %u k %u d %u conv %u) vocab %u eps %g", c.n_layers, c.hidden, c.n_q_heads, c.n_kv_heads,
                  c.head_dim, c.rope_dim, double(c.rope_theta), c.experts_topk, c.n_experts, c.expert_ffn,
                  c.full_attn_interval, c.ssm_inner, c.ssm_n_v_heads, c.ssm_n_k_heads, c.ssm_head_dim, c.ssm_conv_kernel,
                  c.vocab, double(c.rms_eps));
    return b;
}
bool same(const ie::QwenConfig& a, const ie::QwenConfig& b) {
    return a.n_layers == b.n_layers && a.hidden == b.hidden && a.n_q_heads == b.n_q_heads && a.n_kv_heads == b.n_kv_heads &&
           a.head_dim == b.head_dim && a.vocab == b.vocab && a.rope_dim == b.rope_dim && a.rope_theta == b.rope_theta &&
           a.n_experts == b.n_experts && a.experts_topk == b.experts_topk && a.expert_ffn == b.expert_ffn &&
           a.full_attn_interval == b.full_attn_interval && a.ssm_inner == b.ssm_inner && a.ssm_n_v_heads == b.ssm_n_v_heads &&
           a.ssm_n_k_heads == b.ssm_n_k_heads && a.ssm_head_dim == b.ssm_head_dim &&
           a.ssm_conv_kernel == b.ssm_conv_kernel && a.rms_eps == b.rms_eps;
}
}  // namespace

int main(int argc, char** argv) {
    const ie::QwenConfig def{};
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            ie::GgufReader g;
            if (auto e = g.open(argv[i]); !e.empty()) { std::printf("%s: %s\n", argv[i], e.c_str()); return 1; }
            ie::QwenConfig c;
            const std::string e = ie::read_qwen35moe_config(g, c);
            std::printf("%s\n  %s%s\n", argv[i], e.empty() ? show(c).c_str() : e.c_str(), same(c, def) ? "  (== defaults)" : "");
            const std::string r = ie::qwen35moe_shape_check(c);
            std::printf("  shape check: %s\n", r.empty() ? "accepted" : r.c_str());
        }
        return 0;
    }
    // The files come from $IE_Q35M_CONFIG_FILES (':'-separated qwen35moe GGUFs of the 35B-A3B shape); unset = nothing to check.
    std::string list;
    if (const char* v = std::getenv("IE_Q35M_CONFIG_FILES"); v && *v) list = v;
    int fails = 0, checked = 0;
    size_t p = 0;
    while (p <= list.size()) {
        const size_t q = list.find(':', p);
        const std::string f = list.substr(p, q == std::string::npos ? std::string::npos : q - p);
        p = (q == std::string::npos) ? list.size() + 1 : q + 1;
        struct stat st {};
        if (f.empty() || stat(f.c_str(), &st) != 0) { std::printf("  SKIP (absent) %s\n", f.c_str()); continue; }
        ie::GgufReader g;
        if (auto e = g.open(f); !e.empty()) { std::printf("  FAIL open %s: %s\n", f.c_str(), e.c_str()); ++fails; continue; }
        ie::QwenConfig c;
        const std::string e = ie::read_qwen35moe_config(g, c);
        const bool ok = e.empty() && same(c, def) && ie::qwen35moe_shape_check(c).empty();
        std::printf("  [%s] %s\n       %s\n", ok ? "OK" : "FAIL", f.c_str(), e.empty() ? show(c).c_str() : e.c_str());
        ++checked;
        if (!ok) ++fails;
    }
    std::printf("qwen35moe_config_test: %d file(s) checked, %d failure(s)%s\n", checked, fails,
                checked ? "" : " (no file present: nothing proven)");
    return fails ? 1 : 0;
}
