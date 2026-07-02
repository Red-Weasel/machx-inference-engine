// tools/ie_qwen35_mtp_accept.cpp — STEP 0 de-risk gate for MTP self-speculative
// decoding on the Qwen3.6-27B (`qwen35`, 65 blocks). Measures the greedy
// ACCEPTANCE RATE of the native MTP / NextN head at blk.64.
//
// MTP recipe (DeepSeek/Qwen NextN), teacher-forced over a fixed token seq
// t_0..t_{N-1}, per position i:
//   h_i  = main-model final-layer hidden at i (residual AFTER blk.63, BEFORE
//          output_norm). Captured via Qwen35DenseModel::set_dump_prefix — the
//          "_L64.bin" dump IS that pre-output_norm residual for ALL positions.
//          (Fully ADDITIVE: no edit to the model class; crown trivially safe.)
//   e_i  = embed(t_{i+1})  (teacher-forced next token; same token_embd table)
//   x_i  = eh_proj( concat( enorm(e_i)[5120], hnorm(h_i)[5120] ) )  → [5120]
//   run x_i through the blk.64 FULL-attention transformer layer (its OWN KV):
//     attn_norm → joint attn_q → split_q_gate_per_head → attn_k/v → q/k rms_norm
//     → rope_partial → full_attention → sigmoid_gate → attn_output → +residual
//     → post_attention_norm → SwiGLU FFN → +residual
//   out_i  = shared_head_norm(layer_out_i) → SHARED lm_head (output.weight)
//   draft_i = argmax(out_i)  = predicted t_{i+2}
//
// ACCEPTANCE = mean( draft_i == t_{i+2} ) over valid i (0 .. N-3).
//
// The blk.64 weights are Q8_0 matrices (dequanted to F16 at load, riding the
// F16 GEMV) + F32 norms; reuses the exact full-attn leaf ops from
// src/model/qwen35_dense.cpp. lm_head = the main model's output.weight (we
// re-load it here independently as a plain F16 GEMV weight, so this tool never
// touches the model's private SoA path).
//
// usage: ie-qwen35-mtp-accept --gguf <27b.gguf> [--text <file>] [--max-tokens N]

#include "ie/allocator.hpp"
#include "ie/deltanet_state.hpp"
#include "ie/dense_transformer.hpp"   // DenseQuantPtr
#include "ie/dequant.hpp"
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/model_config.hpp"
#include "ie/ops.hpp"
#include "ie/qwen35_dense.hpp"
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

// Same built-in clean-prose sample ie-perplexity uses (stable, in-distribution).
const char* kSampleText =
    "I am by birth a Genevese; and my family is one of the most distinguished "
    "of that republic. My ancestors had been for many years counsellors and "
    "syndics; and my father had filled several public situations with honour "
    "and reputation. He was respected by all who knew him for his integrity "
    "and indefatigable attention to public business. He passed his younger "
    "days perpetually occupied by the affairs of his country; and it was not "
    "until the decline of life that he thought of marrying, and bestowing on "
    "the state sons who might carry his virtues and his name down to "
    "posterity. As the circumstances of his marriage illustrate his "
    "character, I cannot refrain from relating them.\n\n"
    "During the years that followed, the household preserved a quiet discipline "
    "which gave dignity to ordinary labor. The library was small, but every "
    "volume had been chosen with care, and the books were read until their "
    "margins carried traces of many hands. At evening the shutters were closed, "
    "the lamp was trimmed, and the younger children listened while letters from "
    "distant friends were read aloud. These letters spoke of voyages, harvests, "
    "public debates, and the patient work by which families keep faith with one "
    "another across time and weather.\n\n"
    "I learned early that knowledge is not gathered by haste alone. A page "
    "understood clearly was worth more than a chapter passed over in restless "
    "curiosity. My teachers encouraged questions, but they also required proof, "
    "comparison, and a willingness to revise an opinion when the evidence did "
    "not support it. In that habit I found a kind of freedom: the mind became "
    "less anxious when it could distinguish a bright guess from a settled fact.";

std::string read_text_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return {};
    std::stringstream ss; ss << f.rdbuf();
    return ss.str();
}

// Dequant a Q8_0 GGUF weight [K,N] (ggml shape[0]=K contiguous → [N,K] row-major)
// to a device F16 [K,N] buffer via dequant_q8_0_to_Bt (the transposed dequant),
// consumable directly by gemv_fp16 / gemm_fp16 as a load-time [K,N] weight.
sycl::half* dequant_q8_0_weight(ie::DeviceAllocator& alloc,
                                const ie::GgufTensorInfo* t, std::string& err) {
    if (!t) { err = "tensor not found"; return nullptr; }
    if (t->dtype != ie::DType::kQ8_0) { err = "expected Q8_0"; return nullptr; }
    const uint32_t K = uint32_t(t->shape[0]);
    const uint32_t N = uint32_t(t->shape[1]);
    void* packed = alloc.malloc(t->nbytes);
    if (!packed) { err = "malloc packed"; return nullptr; }
    alloc.queue().memcpy(packed, t->data, t->nbytes).wait();
    auto* d = static_cast<sycl::half*>(
        alloc.malloc(uint64_t(K) * N * sizeof(sycl::half)));
    if (!d) { alloc.free(packed); err = "malloc fp16"; return nullptr; }
    ie::dequant_q8_0_to_Bt(alloc.queue(), packed, d, K, N).wait();
    alloc.free(packed);
    return d;
}

float* upload_f32(ie::DeviceAllocator& alloc, const ie::GgufTensorInfo* t,
                  std::string& err) {
    if (!t) { err = "tensor not found"; return nullptr; }
    if (t->dtype != ie::DType::kF32) { err = "expected F32"; return nullptr; }
    void* d = alloc.malloc(t->nbytes);
    if (!d) { err = "malloc f32"; return nullptr; }
    alloc.queue().memcpy(d, t->data, t->nbytes).wait();
    return static_cast<float*>(d);
}

// Local F16 [M,K] @ [K,N](=W_kn) -> [M,N] FP16, via gemm_fp16 (fp32 C) + cast.
// All MTP-head matmuls are F16 weights with M>=2, so the prefill GEMM path is
// always the right one. `cscr` is a grown-on-demand fp32 scratch (+8 rows for
// gemm_fp16's TM=8 store overrun).
struct GemmScratch { float* c = nullptr; uint64_t cap = 0; };
void f16gemm(sycl::queue& q, GemmScratch& s, const sycl::half* A,
             const sycl::half* W_kn, sycl::half* y,
             uint32_t M, uint32_t K, uint32_t N) {
    const uint64_t need = (uint64_t(M) + 8) * N;
    if (need > s.cap) {
        if (s.c) sycl::free(s.c, q);
        s.c = sycl::malloc_device<float>(need, q);
        s.cap = s.c ? need : 0;
    }
    sycl::event e = ie::gemm_fp16(q, A, W_kn, s.c, M, N, K);
    ie::cast_fp32_to_fp16(q, s.c, y, uint64_t(M) * N, {e}).wait();
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf_path =
        "/home/weezy/models/bartowski/Qwen3.6-27B-GGUF/Qwen_Qwen3.6-27B-Q4_K_M.gguf";
    std::string text_path;
    uint32_t max_tokens = 256;
    int concat_order = 0;   // 0 = [enorm(e); hnorm(h)]  (spec default), 1 = swapped

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf"       && i + 1 < argc) gguf_path  = argv[++i];
        else if (a == "--text"       && i + 1 < argc) text_path  = argv[++i];
        else if (a == "--max-tokens" && i + 1 < argc) max_tokens = std::atoi(argv[++i]);
        else if (a == "--swap-concat")                concat_order = 1;
    }

    std::printf("ie-qwen35-mtp-accept — STEP 0 MTP acceptance gate\n");
    std::printf("  gguf        : %s\n", gguf_path.c_str());
    std::printf("  concat order: %s\n",
                concat_order == 0 ? "[enorm(e) ; hnorm(h)]" : "[hnorm(h) ; enorm(e)]");

    ie::GgufReader g;
    if (auto e = g.open(gguf_path); !e.empty()) { std::fprintf(stderr, "gguf: %s\n", e.c_str()); return 1; }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tok: %s\n", e.c_str()); return 1; }
    ie::DeviceAllocator alloc;
    if (auto e = alloc.init(); !e.empty()) { std::fprintf(stderr, "alloc: %s\n", e.c_str()); return 1; }
    auto& q = alloc.queue();
    std::printf("  device      : %s\n",
                q.get_device().get_info<sycl::info::device::name>().c_str());

    ie::Qwen35Config qcfg;
    if (auto e = ie::read_qwen35_config(g, qcfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }

    const ie::DenseConfig& dc = qcfg.dense;
    const uint32_t H    = dc.hidden;                  // 5120
    const uint32_t HD   = dc.head_dim;                // 256
    const uint32_t N_q  = dc.n_q_heads  * HD;         // 6144
    const uint32_t N_qg = N_q * 2u;                   // 12288 (joint Q|gate)
    const uint32_t N_kv = dc.n_kv_heads * HD;         // 1024
    const uint32_t F    = dc.ffn;                     // 17408
    const uint32_t rope_n = dc.rope_dim;              // 64 (partial)
    const float    eps  = dc.rms_eps;
    const uint32_t vocab = dc.vocab;
    const uint32_t mtp_blk = qcfg.n_transformer_layers();   // 64
    std::printf("  H=%u HD=%u N_q=%u N_kv=%u F=%u vocab=%u  MTP blk=%u\n",
                H, HD, N_q, N_kv, F, vocab, mtp_blk);

    // --- main model (blk.0..63) ---
    ie::Qwen35DenseModel model;
    std::printf("  loading 27B main model (~18 GB to device)...\n");
    if (auto e = model.load(alloc, g, qcfg); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }

    // --- tokenize corpus ---
    std::string corpus = text_path.empty() ? kSampleText : read_text_file(text_path);
    if (corpus.empty()) { std::fprintf(stderr, "empty corpus\n"); return 1; }
    auto ids = tok.encode(corpus, /*allow_special=*/false);
    if (tok.add_bos_token() && tok.bos_token_id() >= 0 &&
        (ids.empty() || ids.front() != tok.bos_token_id()))
        ids.insert(ids.begin(), tok.bos_token_id());
    if (ids.size() > max_tokens) ids.resize(max_tokens);
    const uint32_t N = uint32_t(ids.size());
    if (N < 3) { std::fprintf(stderr, "need >=3 tokens, got %u\n", N); return 1; }
    std::printf("  tokens      : %u\n", N);

    // --- caches for the main model (single-chunk prefill of the whole seq) ---
    const uint32_t n_tx   = qcfg.n_transformer_layers();          // 64
    const uint32_t L_full = n_tx / qcfg.full_attn_interval;       // 16
    const uint32_t L_lin  = n_tx - L_full;                        // 48
    const uint32_t max_ctx = N + 4;
    ie::KvCache kv;
    if (auto e = kv.init(alloc, ie::KvCacheConfig{L_full, dc.n_kv_heads, max_ctx, HD}); !e.empty())
        { std::fprintf(stderr, "kv: %s\n", e.c_str()); return 1; }
    const uint32_t conv_ch = qcfg.ssm_inner + 2u * qcfg.ssm_n_k_heads * qcfg.ssm_state;
    ie::DeltaNetState dn;
    if (auto e = dn.init(alloc, ie::DeltaNetStateConfig{
            L_lin, qcfg.ssm_n_v_heads, qcfg.ssm_state, qcfg.ssm_state, conv_ch, qcfg.ssm_conv_kernel});
        !e.empty()) { std::fprintf(stderr, "dn: %s\n", e.c_str()); return 1; }
    kv.reset(); dn.reset(q);

    if (auto e = model.ensure_workspace(N); !e.empty()) { std::fprintf(stderr, "ws: %s\n", e.c_str()); return 1; }
    if (auto e = model.ensure_attn_partials(max_ctx); !e.empty()) { std::fprintf(stderr, "partials: %s\n", e.c_str()); return 1; }

    // Capture the main model's per-layer residuals via the dump mechanism. We
    // only consume "_L<n_tx>.bin" (post-blk.63 residual = pre-output_norm hidden,
    // all N positions, fp32 [N,H]). Write to a temp prefix in /tmp.
    char prefix[256];
    std::snprintf(prefix, sizeof(prefix), "/tmp/ie_mtp_h_%d", int(getpid()));
    model.set_dump_prefix(prefix);

    auto* d_ids = sycl::malloc_device<int32_t>(N, q);
    q.memcpy(d_ids, ids.data(), N * sizeof(int32_t)).wait();
    auto* d_logits_main = sycl::malloc_device<sycl::half>(vocab, q);

    std::printf("  prefilling main model (T=%u)...\n", N);
    model.forward(q, d_ids, N, 0, kv, dn, d_logits_main).wait();

    // Read back h_0..h_{N-1} (pre-output_norm) as fp32 [N,H].
    char hpath[300];
    std::snprintf(hpath, sizeof(hpath), "%s_L%02u.bin", prefix, n_tx);
    std::vector<float> h_all(uint64_t(N) * H);
    {
        std::ifstream f(hpath, std::ios::binary);
        if (!f) { std::fprintf(stderr, "could not open hidden dump %s\n", hpath); return 1; }
        f.read(reinterpret_cast<char*>(h_all.data()), uint64_t(N) * H * sizeof(float));
        if (!f) { std::fprintf(stderr, "short read on %s\n", hpath); return 1; }
    }
    // Clean up the dumps (_L00..L64 .bin/.meta).
    for (uint32_t L = 0; L <= n_tx; ++L) {
        char p[320];
        std::snprintf(p, sizeof(p), "%s_L%02u.bin", prefix, L);  std::remove(p);
        std::snprintf(p, sizeof(p), "%s_L%02u.meta", prefix, L); std::remove(p);
    }
    std::printf("  captured pre-output_norm hidden [%u,%u]\n", N, H);

    // ====================================================================
    // Load blk.64 (MTP / NextN head) weights.
    // ====================================================================
    char buf[64];
    auto MT = [&](const char* name) {
        std::snprintf(buf, sizeof(buf), "blk.%u.%s", mtp_blk, name);
        return g.find_tensor(buf);
    };
    std::string err;
    auto reqf16 = [&](const char* name) -> sycl::half* {
        sycl::half* p = dequant_q8_0_weight(alloc, MT(name), err);
        if (!err.empty()) { std::fprintf(stderr, "blk.64.%s: %s\n", name, err.c_str()); std::exit(1); }
        return p;
    };
    auto reqf32 = [&](const char* name) -> float* {
        float* p = upload_f32(alloc, MT(name), err);
        if (!err.empty()) { std::fprintf(stderr, "blk.64.%s: %s\n", name, err.c_str()); std::exit(1); }
        return p;
    };

    // MTP-specific
    sycl::half* eh_proj = reqf16("nextn.eh_proj.weight");   // [K=10240, N=5120]
    float* enorm        = reqf32("nextn.enorm.weight");     // [5120]
    float* hnorm        = reqf32("nextn.hnorm.weight");     // [5120]
    float* shead_norm   = reqf32("nextn.shared_head_norm.weight");  // [5120]
    // full-attn transformer layer
    sycl::half* w_attn_q   = reqf16("attn_q.weight");       // [5120, 12288] joint Q|gate
    sycl::half* w_attn_k   = reqf16("attn_k.weight");       // [5120, 1024]
    sycl::half* w_attn_v   = reqf16("attn_v.weight");       // [5120, 1024]
    sycl::half* w_attn_out = reqf16("attn_output.weight");  // [6144, 5120]
    float* attn_norm       = reqf32("attn_norm.weight");
    float* post_attn_norm  = reqf32("post_attention_norm.weight");
    float* attn_q_norm     = reqf32("attn_q_norm.weight");  // [256]
    float* attn_k_norm     = reqf32("attn_k_norm.weight");  // [256]
    sycl::half* w_ffn_gate  = reqf16("ffn_gate.weight");    // [5120, 17408]
    sycl::half* w_ffn_up    = reqf16("ffn_up.weight");      // [5120, 17408]
    sycl::half* w_ffn_down  = reqf16("ffn_down.weight");    // [17408, 5120]

    // lm_head = the main model's output.weight. Re-load it here as a plain F16
    // GEMV weight (the GGUF stores it Q6_K — dequant to fp16 [K=H, N=vocab]).
    sycl::half* w_lm_head = nullptr;
    {
        const auto* o = g.find_tensor("output.weight");
        if (!o) { std::fprintf(stderr, "output.weight not found\n"); return 1; }
        const uint32_t K = uint32_t(o->shape[0]), Nv = uint32_t(o->shape[1]);
        void* packed = alloc.malloc(o->nbytes);
        q.memcpy(packed, o->data, o->nbytes).wait();
        w_lm_head = static_cast<sycl::half*>(alloc.malloc(uint64_t(K) * Nv * sizeof(sycl::half)));
        if (o->dtype == ie::DType::kQ6_K)      ie::dequant_q6_K_to_Bt(q, packed, w_lm_head, K, Nv).wait();
        else if (o->dtype == ie::DType::kQ8_0) ie::dequant_q8_0_to_Bt(q, packed, w_lm_head, K, Nv).wait();
        else if (o->dtype == ie::DType::kQ4_K) ie::dequant_q4_K_to_Bt(q, packed, w_lm_head, K, Nv).wait();
        else { std::fprintf(stderr, "output.weight dtype unsupported\n"); return 1; }
        alloc.free(packed);
        std::printf("  lm_head     : output.weight [%u,%u] dequanted to fp16\n", K, Nv);
    }
    std::printf("  blk.64 MTP head loaded.\n");

    // ====================================================================
    // MTP forward over the whole sequence in ONE T=N chunk (its own KV).
    // We run all positions i=0..N-1 in one batched forward (teacher-forced).
    // x_i depends on h_i (captured) + embed(t_{i+1}); only i <= N-2 has a
    // teacher-forced next token. We build x for i in [0, N-2] (M = N-1 rows),
    // then run blk.64 over those M positions and compare draft_i to t_{i+2}.
    // ====================================================================
    const uint32_t M = N - 1;   // positions 0..N-2 (need t_{i+1} for embed)
    GemmScratch gs;             // fp32 scratch reused across all F16 GEMMs

    // device scratch
    auto* d_h    = sycl::malloc_device<sycl::half>(uint64_t(M) * H, q);       // hnorm input (h_i)
    auto* d_e    = sycl::malloc_device<sycl::half>(uint64_t(M) * H, q);       // enorm input (embed t_{i+1})
    auto* d_hn   = sycl::malloc_device<sycl::half>(uint64_t(M) * H, q);
    auto* d_en   = sycl::malloc_device<sycl::half>(uint64_t(M) * H, q);
    auto* d_cat  = sycl::malloc_device<sycl::half>(uint64_t(M) * 2u * H, q);  // [M, 10240]
    auto* d_x    = sycl::malloc_device<sycl::half>(uint64_t(M) * H, q);       // eh_proj out (residual stream)
    auto* d_xn   = sycl::malloc_device<sycl::half>(uint64_t(M) * H, q);       // normed scratch
    auto* d_qg   = sycl::malloc_device<sycl::half>(uint64_t(M) * N_qg, q);
    auto* d_q    = sycl::malloc_device<sycl::half>(uint64_t(M) * N_q, q);
    auto* d_gate = sycl::malloc_device<sycl::half>(uint64_t(M) * N_q, q);
    auto* d_k    = sycl::malloc_device<sycl::half>(uint64_t(M) * N_kv, q);
    auto* d_v    = sycl::malloc_device<sycl::half>(uint64_t(M) * N_kv, q);
    auto* d_ao   = sycl::malloc_device<sycl::half>(uint64_t(M) * N_q, q);
    auto* d_blk  = sycl::malloc_device<sycl::half>(uint64_t(M) * H, q);       // attn/ffn out-proj
    auto* d_fg   = sycl::malloc_device<sycl::half>(uint64_t(M) * F, q);
    auto* d_fu   = sycl::malloc_device<sycl::half>(uint64_t(M) * F, q);
    auto* d_fh   = sycl::malloc_device<sycl::half>(uint64_t(M) * F, q);
    auto* d_pos  = sycl::malloc_device<int32_t>(M, q);
    auto* d_logits = sycl::malloc_device<sycl::half>(uint64_t(M) * vocab, q);
    // MTP own KV cache (1 full-attn layer)
    auto* mtp_kc = sycl::malloc_device<sycl::half>(uint64_t(dc.n_kv_heads) * max_ctx * HD, q);
    auto* mtp_vc = sycl::malloc_device<sycl::half>(uint64_t(dc.n_kv_heads) * max_ctx * HD, q);

    // h_i (i in [0,M)) → d_h (cast fp32→fp16)
    {
        std::vector<sycl::half> hh(uint64_t(M) * H);
        for (uint64_t r = 0; r < uint64_t(M) * H; ++r) hh[r] = sycl::half(h_all[r]);
        q.memcpy(d_h, hh.data(), uint64_t(M) * H * sizeof(sycl::half)).wait();
    }
    // embed(t_{i+1}) for i in [0,M) → d_e. token_embd is Q4_K in this GGUF.
    {
        std::vector<int32_t> next_ids(M);
        for (uint32_t i = 0; i < M; ++i) next_ids[i] = ids[i + 1];
        auto* d_next = sycl::malloc_device<int32_t>(M, q);
        q.memcpy(d_next, next_ids.data(), M * sizeof(int32_t)).wait();
        const auto* te = g.find_tensor("token_embd.weight");
        // The embedding table is uploaded by the model; re-upload here packed for
        // the lookup kernel (independent of the model's private buffer).
        void* te_dev = alloc.malloc(te->nbytes);
        q.memcpy(te_dev, te->data, te->nbytes).wait();
        if (te->dtype == ie::DType::kQ4_K)      ie::embedding_lookup_q4k(q, d_next, te_dev, d_e, M, H).wait();
        else if (te->dtype == ie::DType::kQ6_K) ie::embedding_lookup_q6k(q, d_next, te_dev, d_e, M, H).wait();
        else if (te->dtype == ie::DType::kQ8_0) ie::embedding_lookup_q8_0(q, d_next, te_dev, d_e, M, H).wait();
        else { std::fprintf(stderr, "token_embd dtype unsupported\n"); return 1; }
        alloc.free(te_dev);
        sycl::free(d_next, q);
    }

    // enorm(e), hnorm(h)  (plain RMSNorm, F32 weight)
    ie::rms_norm_f32w(q, d_e, enorm, d_en, M, H, eps).wait();
    ie::rms_norm_f32w(q, d_h, hnorm, d_hn, M, H, eps).wait();

    // concat → [M, 2H]. order 0: [enorm(e) ; hnorm(h)]; order 1 swapped.
    {
        const sycl::half* first  = (concat_order == 0) ? d_en : d_hn;
        const sycl::half* second = (concat_order == 0) ? d_hn : d_en;
        q.parallel_for(sycl::range<1>(uint64_t(M) * H), [=](sycl::id<1> i) {
            const uint64_t r = uint64_t(i) / H, c = uint64_t(i) % H;
            d_cat[r * 2u * H + c]      = first[i];
            d_cat[r * 2u * H + H + c]  = second[i];
        }).wait();
    }

    // x = eh_proj(cat) : [M,2H] @ [2H, H] → [M, H].  eh_proj is F16 [K=2H, N=H].
    f16gemm(q, gs, d_cat, eh_proj, d_x, M, 2u * H, H);

    // positions 0..M-1 for the MTP layer's own attention.
    {
        std::vector<int32_t> pos(M);
        for (uint32_t i = 0; i < M; ++i) pos[i] = int32_t(i);
        q.memcpy(d_pos, pos.data(), M * sizeof(int32_t)).wait();
    }

    // ---- blk.64 full-attention transformer layer over d_x (residual stream) ----
    // pre-attn norm
    ie::rms_norm_f32w(q, d_x, attn_norm, d_xn, M, H, eps).wait();
    // joint Q|gate → split per head
    f16gemm(q, gs, d_xn, w_attn_q, d_qg, M, H, N_qg);
    ie::split_q_gate_per_head(q, d_qg, d_q, d_gate, M, dc.n_q_heads, HD).wait();
    // K, V
    f16gemm(q, gs, d_xn, w_attn_k, d_k, M, H, N_kv);
    f16gemm(q, gs, d_xn, w_attn_v, d_v, M, H, N_kv);
    // per-head Q/K rms_norm
    ie::rms_norm_f32w(q, d_q, attn_q_norm, d_q, M * dc.n_q_heads,  HD, eps).wait();
    ie::rms_norm_f32w(q, d_k, attn_k_norm, d_k, M * dc.n_kv_heads, HD, eps).wait();
    // partial RoPE
    ie::rope_partial(q, d_q, d_pos, d_q, M, dc.n_q_heads,  HD, rope_n, dc.rope_theta).wait();
    ie::rope_partial(q, d_k, d_pos, d_k, M, dc.n_kv_heads, HD, rope_n, dc.rope_theta).wait();
    // full attention (its own KV, start_pos=0, T=M)
    ie::full_attention(q, d_q, d_k, d_v, mtp_kc, mtp_vc, d_ao,
                       M, 0, dc.n_q_heads, dc.n_kv_heads, HD, max_ctx).wait();
    // sigmoid gate
    ie::sigmoid_gate(q, d_ao, d_gate, d_ao, uint64_t(M) * N_q).wait();
    // out-proj → d_blk
    f16gemm(q, gs, d_ao, w_attn_out, d_blk, M, N_q, H);
    // residual + post-attention norm: d_x += d_blk; d_xn = rms(d_x)
    ie::residual_add(q, d_x, d_blk, d_x, uint64_t(M) * H).wait();
    ie::rms_norm_f32w(q, d_x, post_attn_norm, d_xn, M, H, eps).wait();
    // SwiGLU FFN
    f16gemm(q, gs, d_xn, w_ffn_gate, d_fg, M, H, F);
    f16gemm(q, gs, d_xn, w_ffn_up,   d_fu, M, H, F);
    ie::swiglu(q, d_fg, d_fu, d_fh, uint64_t(M) * F).wait();
    f16gemm(q, gs, d_fh, w_ffn_down, d_blk, M, F, H);
    ie::residual_add(q, d_x, d_blk, d_x, uint64_t(M) * H).wait();

    // shared_head_norm → shared lm_head → logits[M, vocab]
    ie::rms_norm_f32w(q, d_x, shead_norm, d_xn, M, H, eps).wait();
    f16gemm(q, gs, d_xn, w_lm_head, d_logits, M, H, vocab);

    // ====================================================================
    // Score: draft_i = argmax(logits_i) should == t_{i+2}, for i in [0, N-3].
    // ====================================================================
    std::vector<sycl::half> hlog(uint64_t(M) * vocab);
    q.memcpy(hlog.data(), d_logits, uint64_t(M) * vocab * sizeof(sycl::half)).wait();

    uint32_t accepted = 0, total = 0;
    double conf_sum = 0.0;
    std::vector<std::pair<int,int>> sample;   // (draft, actual) for first 10
    for (uint32_t i = 0; i + 2 < N; ++i) {     // need t_{i+2}
        const sycl::half* row = hlog.data() + uint64_t(i) * vocab;
        float best = float(row[0]); int arg = 0;
        for (uint32_t v = 1; v < vocab; ++v) {
            float val = float(row[v]);
            if (val > best) { best = val; arg = int(v); }
        }
        // softmax confidence of the top-1
        float m = best;
        double se = 0.0;
        for (uint32_t v = 0; v < vocab; ++v) se += std::exp(double(float(row[v])) - double(m));
        const double conf = 1.0 / se;   // exp(best-m)=1 / sum
        conf_sum += conf;

        const int actual = ids[i + 2];
        if (arg == actual) ++accepted;
        ++total;
        if (sample.size() < 10) sample.emplace_back(arg, actual);
    }

    const double accept_rate = total ? double(accepted) / total : 0.0;
    std::printf("\n=== RESULT ===\n");
    std::printf("  scored positions : %u\n", total);
    std::printf("  accepted (draft==t_{i+2}) : %u\n", accepted);
    std::printf("  ACCEPTANCE RATE  : %.4f\n", accept_rate);
    std::printf("  mean top-1 conf  : %.4f\n", total ? conf_sum / total : 0.0);
    std::printf("  random baseline  : %.3e  (1/vocab)\n", 1.0 / double(vocab));
    std::printf("\n  first %zu (draft, actual) pairs  [d='draft piece' a='actual piece']:\n", sample.size());
    for (size_t k = 0; k < sample.size(); ++k) {
        auto pd = tok.decode(std::vector<int32_t>{sample[k].first},  /*skip_special=*/false);
        auto pa = tok.decode(std::vector<int32_t>{sample[k].second}, /*skip_special=*/false);
        std::printf("    i=%2zu  draft=%-7d a=%-7d  %s  d='%s' a='%s'\n",
                    k, sample[k].first, sample[k].second,
                    sample[k].first == sample[k].second ? "MATCH" : "     ",
                    pd.c_str(), pa.c_str());
    }
    std::printf("\n  GATE: %s (>=0.5 → STEP 0 PASSES)\n",
                accept_rate >= 0.5 ? "PASS" : "FAIL");

    sycl::free(d_ids, q); sycl::free(d_logits_main, q);
    return 0;
}
