// tools/ie_qwen36_spec.cpp — Crown (Qwen3.6-35B-A3B) MTP self-speculative decode.
//
// Mirrors ie-qwen35-spec but for the Crown (QwenModel). The Crown gained native
// MTP via QwenModel::load_mtp_head + spec_generate (the head is full-attn + MoE).
// Verifies token-for-token LOSSLESS vs plain greedy.
//
// usage: ie-qwen36-spec --gguf <crown-MTP.gguf> [--prompt <text>] [--ntok N] [--K K]
//        (needs the MTP-converted GGUF: unsloth/Qwen3.6-35B-A3B-MTP-GGUF)

#include "ie/allocator.hpp"
#include "ie/deltanet_state.hpp"
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/ops.hpp"
#include "ie/qwen36.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {
int argmax_row(const sycl::half* row, uint32_t vocab) {
    float best = float(row[0]); int arg = 0;
    for (uint32_t v = 1; v < vocab; ++v) { float val = float(row[v]); if (val > best) { best = val; arg = int(v); } }
    return arg;
}
}  // namespace

int main(int argc, char** argv) {
    // Two GGUFs: the main (verify) model loads from a STANDARD-quant crown GGUF
    // (the engine can't read the unsloth UD quant — Q5_K MoE/Q8_0 attn/F32 ssm);
    // only the blk.40 MTP head comes from the unsloth MTP GGUF (head experts get
    // dequanted to fp16 in load_mtp_head, so its UD quant is fine for the head).
    std::string gguf_path = "/home/weezy/.seal/models/Qwen3.6-35B-A3B-Q4_K_M.gguf";
    std::string head_path = "/home/weezy/models/unsloth-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf";
    std::string prompt = "The capital of France is";
    uint32_t ntok = 48, K = 4;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf"      && i + 1 < argc) gguf_path = argv[++i];
        else if (a == "--head-gguf" && i + 1 < argc) head_path = argv[++i];
        else if (a == "--prompt"    && i + 1 < argc) prompt    = argv[++i];
        else if (a == "--ntok"      && i + 1 < argc) ntok      = std::atoi(argv[++i]);
        else if (a == "--K"         && i + 1 < argc) K         = std::atoi(argv[++i]);
    }
    std::printf("ie-qwen36-spec — Crown MTP self-speculative decode (LOSSLESS gate)\n");

    ie::GgufReader g, head_g;
    if (auto e = g.open(gguf_path);      !e.empty()) { std::fprintf(stderr, "gguf: %s\n", e.c_str()); return 1; }
    if (auto e = head_g.open(head_path); !e.empty()) { std::fprintf(stderr, "head gguf: %s\n", e.c_str()); return 1; }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tok: %s\n", e.c_str()); return 1; }

    ie::DeviceAllocator alloc;
    if (auto e = alloc.init(); !e.empty()) { std::fprintf(stderr, "alloc: %s\n", e.c_str()); return 1; }
    auto& q = alloc.queue();
    const uint32_t max_ctx = 4096;

    ie::QwenConfig cfg;                 // defaults == the Crown's config
    ie::QwenModel model;
    std::printf("  loading Crown (main=%s)...\n", gguf_path.c_str());
    if (auto e = model.load(alloc, g, cfg); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    if (auto e = model.load_mtp_head(head_g, max_ctx); !e.empty()) { std::fprintf(stderr, "load_mtp_head: %s\n", e.c_str()); return 1; }
    std::printf("  blk.%u MTP head loaded (from %s).\n", cfg.n_layers, head_path.c_str());

    const uint32_t H = cfg.hidden, vocab = cfg.vocab;
    const uint32_t L_full = cfg.n_layers / cfg.full_attn_interval;
    const uint32_t L_lin  = cfg.n_layers - L_full;

    auto ids = tok.encode(prompt, /*allow_special=*/false);
    const uint32_t P0 = uint32_t(ids.size());
    if (P0 == 0) { std::fprintf(stderr, "empty prompt\n"); return 1; }

    auto make_kv = [&](ie::KvCache& kv) {
        ie::KvCacheConfig c{}; c.n_layers_full = L_full; c.n_kv_heads = cfg.n_kv_heads;
        c.max_ctx = max_ctx; c.head_dim = cfg.head_dim; return kv.init(alloc, c);
    };
    auto make_dn = [&](ie::DeltaNetState& dn) {
        return dn.init(alloc, ie::DeltaNetStateConfig{
            L_lin, cfg.ssm_n_v_heads, cfg.ssm_head_dim, cfg.ssm_head_dim,
            cfg.ssm_inner * 2, cfg.ssm_conv_kernel});
    };

    if (auto e = model.ensure_workspace(std::max<uint32_t>(P0, K)); !e.empty()) { std::fprintf(stderr, "ws: %s\n", e.c_str()); return 1; }
    if (auto e = model.ensure_attn_partials(max_ctx); !e.empty()) { std::fprintf(stderr, "partials: %s\n", e.c_str()); return 1; }

    auto* d_logits = sycl::malloc_device<sycl::half>(vocab, q);
    auto* d_ids    = sycl::malloc_device<int32_t>(std::max<uint32_t>(P0, K) + 8, q);
    std::vector<sycl::half> row(vocab);

    // (A) Plain greedy — the oracle.
    std::vector<int32_t> plain;
    double plain_tps = 0;
    {
        ie::KvCache kv; ie::DeltaNetState dn;
        if (auto e = make_kv(kv); !e.empty()) { std::fprintf(stderr, "kv: %s\n", e.c_str()); return 1; }
        if (auto e = make_dn(dn); !e.empty()) { std::fprintf(stderr, "dn: %s\n", e.c_str()); return 1; }
        kv.reset(); dn.reset(q);
        q.memcpy(d_ids, ids.data(), P0 * sizeof(int32_t)).wait();
        model.forward(q, d_ids, P0, 0, kv, dn, d_logits).wait();
        q.memcpy(row.data(), d_logits, uint64_t(vocab) * sizeof(sycl::half)).wait();
        int32_t next = argmax_row(row.data(), vocab);
        uint32_t pos = P0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t s = 0; s < ntok; ++s) {
            plain.push_back(next);
            q.memcpy(d_ids, &next, sizeof(int32_t)).wait();
            model.forward(q, d_ids, 1, pos, kv, dn, d_logits).wait();
            ++pos;
            q.memcpy(row.data(), d_logits, uint64_t(vocab) * sizeof(sycl::half)).wait();
            next = argmax_row(row.data(), vocab);
        }
        auto t1 = std::chrono::steady_clock::now();
        plain_tps = ntok / std::chrono::duration<double>(t1 - t0).count();
        std::printf("  plain greedy : %.2f tok/s\n", plain_tps);
    }

    // (B) Spec greedy via the new QwenModel::spec_generate.
    std::vector<int32_t> spec;
    double spec_tps = 0;
    {
        ie::KvCache kv; ie::DeltaNetState dn;
        if (auto e = make_kv(kv); !e.empty()) { std::fprintf(stderr, "kv: %s\n", e.c_str()); return 1; }
        if (auto e = make_dn(dn); !e.empty()) { std::fprintf(stderr, "dn: %s\n", e.c_str()); return 1; }
        kv.reset(); dn.reset(q);
        // prefill prompt with hidden_pre_norm → h_last + tn at the last position.
        auto* d_hpf = sycl::malloc_device<sycl::half>(uint64_t(P0) * H, q);
        auto* h_last = sycl::malloc_device<sycl::half>(uint64_t(H), q);
        q.memcpy(d_ids, ids.data(), P0 * sizeof(int32_t)).wait();
        model.forward(q, d_ids, P0, 0, kv, dn, d_logits, nullptr, d_hpf).wait();
        q.memcpy(row.data(), d_logits, uint64_t(vocab) * sizeof(sycl::half)).wait();
        int32_t tn = argmax_row(row.data(), vocab);
        q.memcpy(h_last, d_hpf + uint64_t(P0 - 1) * H, uint64_t(H) * sizeof(sycl::half)).wait();

        auto t0 = std::chrono::steady_clock::now();
        auto e = model.spec_generate(q, kv, dn, h_last, tn, P0, ntok, K,
                                     [&](int32_t id) { spec.push_back(id); return true; });
        auto t1 = std::chrono::steady_clock::now();
        if (!e.empty()) { std::fprintf(stderr, "spec_generate: %s\n", e.c_str()); return 1; }
        spec_tps = spec.size() / std::chrono::duration<double>(t1 - t0).count();
        sycl::free(d_hpf, q); sycl::free(h_last, q);
        std::printf("  spec  greedy : %.2f tok/s\n", spec_tps);
    }

    // (C) Lossless gate.
    bool lossless = (spec.size() >= plain.size());
    for (size_t i = 0; i < plain.size() && i < spec.size(); ++i)
        if (plain[i] != spec[i]) { lossless = false; break; }
    std::printf("  LOSSLESS     : %s\n", lossless ? "YES (token-for-token identical)" : "NO");
    std::printf("  speedup      : %.2fx\n", spec_tps / plain_tps);
    return lossless ? 0 : 2;
}
