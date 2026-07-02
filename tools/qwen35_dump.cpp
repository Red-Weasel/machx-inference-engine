// tools/qwen35_dump.cpp — per-layer residual dumper for the qwen35 dense-hybrid
// (Qwen3.6-27B). The engine-side counterpart of tools/ie-llama-dump for
// tools/diff_layers.sh: loads Qwen35DenseModel directly (NOT via Engine, so we
// can set the dump prefix), runs ONE prefill of the prompt, and writes
// <prefix>_L00.bin (post-embedding) + _L01..L64.bin (residual after each layer)
// as fp32 [T,H]. Also prints argmax + top-5 of the last token for greedy parity.
//
// Mirrors src/engine/engine.cpp's kQwen35Dense load branch exactly. One model
// on the GPU at a time (GPU exclusivity).
//
// usage: ie-qwen35-dump --gguf <27b.gguf> --dump <prefix> -p "prompt" [--ctx N]

#include "ie/allocator.hpp"
#include "ie/deltanet_state.hpp"
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/model_config.hpp"
#include "ie/ops.hpp"
#include "ie/qwen35_dense.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string gguf_path, dump_prefix, prompt = "The capital of France is";
    uint32_t max_ctx = 512;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf" && i + 1 < argc) gguf_path   = argv[++i];
        else if (a == "--dump" && i + 1 < argc) dump_prefix = argv[++i];
        else if (a == "-p"     && i + 1 < argc) prompt      = argv[++i];
        else if (a == "--ctx"  && i + 1 < argc) max_ctx     = std::atoi(argv[++i]);
    }
    if (gguf_path.empty() || dump_prefix.empty()) {
        std::fprintf(stderr, "usage: %s --gguf <27b.gguf> --dump <prefix> [-p \"prompt\"] [--ctx N]\n", argv[0]);
        return 2;
    }

    ie::GgufReader g;
    if (auto e = g.open(gguf_path); !e.empty()) { std::fprintf(stderr, "gguf: %s\n", e.c_str()); return 1; }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tok: %s\n", e.c_str()); return 1; }
    ie::DeviceAllocator alloc;
    if (auto e = alloc.init(); !e.empty()) { std::fprintf(stderr, "alloc: %s\n", e.c_str()); return 1; }
    auto& q = alloc.queue();

    ie::Qwen35Config qcfg;
    if (auto e = ie::read_qwen35_config(g, qcfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }
    ie::Qwen35DenseModel model;
    std::printf("loading 27B (~18 GB to device)...\n");
    if (auto e = model.load(alloc, g, qcfg); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }

    const uint32_t n_tx   = qcfg.n_transformer_layers();          // 64
    const uint32_t L_full = n_tx / qcfg.full_attn_interval;       // 16
    const uint32_t L_lin  = n_tx - L_full;                        // 48
    ie::KvCache kv;
    if (auto e = kv.init(alloc, ie::KvCacheConfig{L_full, qcfg.dense.n_kv_heads, max_ctx, qcfg.dense.head_dim});
        !e.empty()) { std::fprintf(stderr, "kv: %s\n", e.c_str()); return 1; }
    const uint32_t conv_ch = qcfg.ssm_inner + 2u * qcfg.ssm_n_k_heads * qcfg.ssm_state;  // 10240
    ie::DeltaNetState dn;
    if (auto e = dn.init(alloc, ie::DeltaNetStateConfig{
            L_lin, qcfg.ssm_n_v_heads, qcfg.ssm_state, qcfg.ssm_state, conv_ch, qcfg.ssm_conv_kernel});
        !e.empty()) { std::fprintf(stderr, "dn: %s\n", e.c_str()); return 1; }
    kv.reset();
    dn.reset(q);

    const auto ids = tok.encode(prompt, /*allow_special=*/true);
    if (ids.empty()) { std::fprintf(stderr, "empty encoding\n"); return 1; }
    std::printf("prompt='%s' tokens=", prompt.c_str());
    for (auto t : ids) std::printf("%d ", t);
    std::printf("(%zu)\n", ids.size());

    if (auto e = model.ensure_workspace(uint32_t(ids.size())); !e.empty()) { std::fprintf(stderr, "ws: %s\n", e.c_str()); return 1; }
    if (auto e = model.ensure_attn_partials(max_ctx); !e.empty()) { std::fprintf(stderr, "partials: %s\n", e.c_str()); return 1; }
    model.set_dump_prefix(dump_prefix);

    const uint32_t vocab = qcfg.dense.vocab;
    auto* d_logits = sycl::malloc_device<sycl::half>(vocab, q);
    auto* d_ids    = sycl::malloc_device<int32_t>(ids.size(), q);
    q.memcpy(d_ids, ids.data(), ids.size() * sizeof(int32_t)).wait();

    // Single-chunk prefill (prompt ≤ ctx, ≤ 256) → forward dumps L00..L64.
    model.forward(q, d_ids, uint32_t(ids.size()), 0, kv, dn, d_logits).wait();

    // argmax + top-5 of the last token (greedy parity vs ie-llama-dump).
    std::vector<sycl::half> hl(vocab);
    q.memcpy(hl.data(), d_logits, vocab * sizeof(sycl::half)).wait();
    std::vector<std::pair<float, int>> v(vocab);
    for (uint32_t i = 0; i < vocab; ++i) v[i] = {float(hl[i]), int(i)};
    std::partial_sort(v.begin(), v.begin() + 5, v.end(), [](auto& a, auto& b){ return a.first > b.first; });
    auto piece = [&](int id){ return tok.decode(std::vector<int32_t>{id}, /*skip_special=*/false); };
    std::printf("argmax: id=%d logit=%.3f piece='%s'\n", v[0].second, v[0].first, piece(v[0].second).c_str());
    for (int r = 0; r < 5; ++r)
        std::printf("  top%d id=%-7d logit=%8.3f '%s'\n", r + 1, v[r].second, v[r].first, piece(v[r].second).c_str());
    std::printf("dumped L00..L%u to %s_LNN.bin\n", n_tx, dump_prefix.c_str());

    sycl::free(d_logits, q); sycl::free(d_ids, q);
    return 0;
}
