// tests/model_config_test.cpp — P2 Task 1: detect_arch + read_dense_config
// against the two real on-disk GGUFs. Host-only (mmap reads, no GPU).
// Skips-with-warning (exit 0) if a GGUF path is absent so CI on other
// boxes stays green.
#undef NDEBUG  // build is Release (-DNDEBUG); asserts must stay live here
#include "ie/model_config.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

std::string home_path(const char* rel) {
    const char* home = std::getenv("HOME");
    return std::string(home ? home : "") + "/" + rel;
}

// Returns true if the GGUF opened; false → caller skips with a warning.
bool open_or_skip(ie::GgufReader& g, const std::string& path) {
    const std::string err = g.open(path);
    if (!err.empty()) {
        std::fprintf(stderr,
                     "model_config_test: SKIP (cannot open %s: %s)\n",
                     path.c_str(), err.c_str());
        return false;
    }
    return true;
}

}  // namespace

int main() {
    int checked = 0;

    // --- crown GGUF: arch must be detected as qwen35moe ------------------
    {
        ie::GgufReader g;
        if (open_or_skip(g, home_path(".seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf"))) {
            assert(ie::detect_arch(g) == ie::ModelArch::kQwen35Moe);
            ++checked;
        }
    }

    // --- qwen3-8b dense: arch + full DenseConfig --------------------------
    {
        ie::GgufReader g;
        if (open_or_skip(g, home_path(".seal/models/Qwen3-8B-Q4_K_M.gguf"))) {
            assert(ie::detect_arch(g) == ie::ModelArch::kQwen3Dense);

            ie::DenseConfig cfg;
            const std::string err = ie::read_dense_config(g, "qwen3", cfg);
            if (!err.empty()) {
                std::fprintf(stderr, "read_dense_config failed: %s\n", err.c_str());
                return 1;
            }
            assert(cfg.n_layers   == 36);
            assert(cfg.hidden     == 4096);
            assert(cfg.n_q_heads  == 32);
            assert(cfg.n_kv_heads == 8);
            assert(cfg.head_dim   == 128);
            assert(cfg.ffn        == 12288);
            assert(cfg.vocab      == 151936);
            assert(cfg.rope_dim   == 128);   // no rope.dimension_count key → full rotary
            assert(std::fabs(cfg.rope_theta - 1e6f) <= 1.f);
            assert(std::fabs(cfg.rms_eps - 1e-6f) <= 1e-9f);
            assert(cfg.ctx_train  == 40960);

            // Asking for a wrong arch prefix must report the first missing
            // REQUIRED key, not silently succeed.
            ie::DenseConfig bogus;
            const std::string err2 = ie::read_dense_config(g, "noarch", bogus);
            assert(!err2.empty());
            assert(err2.find("noarch.block_count") != std::string::npos);
            assert(cfg.arch == ie::ModelArch::kQwen3Dense);
            ++checked;
        }
    }

    // --- Llama-3.1-8B-Instruct: arch + DenseConfig (P3a Task 0 values) -----
    {
        ie::GgufReader g;
        const std::string p = home_path(
            "models/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/"
            "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf");
        if (open_or_skip(g, p)) {
            assert(ie::detect_arch(g) == ie::ModelArch::kLlama3);
            assert(std::string(ie::arch_key(ie::ModelArch::kLlama3)) == "llama");

            ie::DenseConfig cfg;
            const std::string err = ie::read_dense_config_auto(g, cfg);
            if (!err.empty()) {
                std::fprintf(stderr, "read_dense_config(llama) failed: %s\n", err.c_str());
                return 1;
            }
            assert(cfg.arch       == ie::ModelArch::kLlama3);
            assert(cfg.n_layers   == 32);
            assert(cfg.hidden     == 4096);
            assert(cfg.n_q_heads  == 32);
            assert(cfg.n_kv_heads == 8);
            assert(cfg.head_dim   == 128);   // attention.key_length absent → hidden/n_q
            assert(cfg.ffn        == 14336);
            assert(cfg.vocab      == 128256);
            assert(cfg.rope_dim   == 128);
            assert(std::fabs(cfg.rope_theta - 5e5f) <= 1.f);
            assert(std::fabs(cfg.rms_eps - 1e-5f) <= 1e-9f);
            assert(cfg.ctx_train  == 131072);
            ++checked;
        }
    }

    // --- qwen3.6-27B dense-hybrid (qwen35): arch + Qwen35Config -----------
    {
        ie::GgufReader g;
        const std::string p =
            home_path("models/bartowski/Qwen3.6-27B-GGUF/Qwen_Qwen3.6-27B-Q4_K_M.gguf");
        if (open_or_skip(g, p)) {
            assert(ie::detect_arch(g) == ie::ModelArch::kQwen35Dense);

            ie::Qwen35Config cfg;
            const std::string err = ie::read_qwen35_config(g, cfg);
            if (!err.empty()) {
                std::fprintf(stderr, "read_qwen35_config failed: %s\n", err.c_str());
                return 1;
            }
            // Echo every field so the real GGUF is the ground-truth record.
            std::printf("qwen35: layers=%u hidden=%u qh=%u kvh=%u hd=%u ffn=%u "
                        "vocab=%u | interval=%u ssm_inner=%u ssm_state=%u "
                        "n_k=%u n_v=%u conv=%u v_hd=%u | rope_sec=[%d,%d,%d,%d]\n",
                        cfg.dense.n_layers, cfg.dense.hidden, cfg.dense.n_q_heads,
                        cfg.dense.n_kv_heads, cfg.dense.head_dim, cfg.dense.ffn,
                        cfg.dense.vocab, cfg.full_attn_interval, cfg.ssm_inner,
                        cfg.ssm_state, cfg.ssm_n_k_heads, cfg.ssm_n_v_heads,
                        cfg.ssm_conv_kernel, cfg.ssm_v_head_dim,
                        cfg.rope_sections[0], cfg.rope_sections[1],
                        cfg.rope_sections[2], cfg.rope_sections[3]);
            std::fflush(stdout);  // survive a failing assert below

            // Structural invariants — values verified against the real GGUF KV
            // table (bartowski Qwen3.6-27B Q4_K_M, 2026-06-11).
            assert(cfg.dense.n_layers == 65);     // block_count INCLUDES 1 NextN/MTP layer
            assert(cfg.nextn_predict_layers == 1);
            assert(cfg.n_transformer_layers() == 64);
            assert(cfg.dense.hidden   == 5120);
            assert(cfg.dense.ffn      == 17408);
            assert(cfg.dense.n_q_heads  == 24);   // full-attention head counts
            assert(cfg.dense.n_kv_heads == 4);
            assert(cfg.dense.head_dim == 256);    // attention.key_length
            assert(cfg.full_attn_interval == 4);
            assert(cfg.ssm_state      == 128);    // k head dim
            assert(cfg.ssm_n_v_heads  == 48);
            assert(cfg.ssm_n_k_heads  == 16);
            assert(cfg.ssm_inner      == 6144);
            assert(cfg.ssm_v_head_dim == 128);    // 6144 / 48
            assert(cfg.ssm_conv_kernel == 4);
            assert(cfg.rope_sections[0] == 11 && cfg.rope_sections[1] == 11 &&
                   cfg.rope_sections[2] == 10 && cfg.rope_sections[3] == 0);

            // Recurrent (linear) vs full-attention mask over the 64 transformer
            // layers (NextN excluded): every 4th layer (il = 3,7,...,63) is full
            // attention → 16 full, 48 linear.
            int n_full = 0, n_lin = 0;
            for (uint32_t il = 0; il < cfg.n_transformer_layers(); ++il)
                (cfg.recurrent_layer(il) ? n_lin : n_full)++;
            assert(n_full == 16);
            assert(n_lin  == 48);
            assert(cfg.recurrent_layer(0) && !cfg.recurrent_layer(3));
            ++checked;
        }
    }

    if (checked == 0) {
        std::fprintf(stderr, "model_config_test: SKIP (no local GGUFs found)\n");
        return 0;
    }
    std::printf("model_config_test: all OK (%d gguf(s) checked)\n", checked);
    return 0;
}
