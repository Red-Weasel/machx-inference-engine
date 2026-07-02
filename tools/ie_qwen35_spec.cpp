// tools/ie_qwen35_spec.cpp — STEP 1b+2: FULL greedy MTP self-speculative decode
// loop on the Qwen3.6-27B (`qwen35`, 65 blocks), proven LOSSLESS vs plain greedy.
//
// The 27B (target) is the source of truth. The native MTP / NextN head at blk.64
// drafts K tokens autoregressively; the target verifies them in ONE forward(T=K)
// and accepts the longest argmax-matching prefix; on partial accept the target's
// DeltaNet + KV state is rolled back to exactly the accepted position. Greedy
// acceptance ⇒ the emitted token sequence is BIT-IDENTICAL to plain greedy decode.
//
// MTP draft recipe (per draft step j, autoregressive):
//   h_in  = (j==0) main-model pre-output_norm hidden at last committed pos
//                  else the blk.64 residual-stream output from draft step j-1
//   e_in  = embed(prev_tok)   (j==0: prev_tok = tn = target argmax @ last pos;
//                              else prev_tok = the token drafted at step j-1)
//   x = eh_proj( concat( enorm(e_in), hnorm(h_in) ) )            → [1,H]
//   run blk.64 FULL-attn transformer layer over x (its OWN 1-layer KV, T=1,
//     pos = p+j) → residual stream mtp_h  → shared_head_norm → shared lm_head
//   g_j = argmax  = drafted token (a candidate for committed-pos p+j+1)
//
// Verify: target forward(T=K, start_pos=p, all_logits=L[K,vocab],
//   hidden_pre_norm=Hd[K,H]) on input [tn, g_1, .., g_{K-1}].
//   L[j] = argmax = the TRUE token after input position j.
//   Accept longest n: g_j == argmax(L[j-1]) for j=1..n (where g_0 := tn always
//   "accepted" — it IS the verified-correct token at pos p). Then commit
//   tn,g_1,..,g_n PLUS the bonus argmax(L[n]); next round's tn' = argmax(L[n]),
//   h_last' = Hd[n]. State must reflect exactly committed prefix p..p+n (the
//   bonus token is NOT yet processed — it becomes next round's tn).
//
// usage: ie-qwen35-spec --gguf <27b.gguf> [--prompt <text>] [--ntok N] [--K K]

#include "ie/allocator.hpp"
#include "ie/deltanet_state.hpp"
#include "ie/dense_transformer.hpp"
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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// ---- Q8_0 weight → device F16 [K,N] (transposed dequant), as in the accept tool.
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

// (The draft path is all M=1 GEMVs → routed through gemv_fp16, not a GEMM.
// GemmScratch is retained as an unused MtpHead member placeholder for clarity.)
struct GemmScratch { float* c = nullptr; uint64_t cap = 0; };

// argmax over a fp16 [vocab] row on host.
int argmax_row(const sycl::half* row, uint32_t vocab) {
    float best = float(row[0]); int arg = 0;
    for (uint32_t v = 1; v < vocab; ++v) {
        float val = float(row[v]);
        if (val > best) { best = val; arg = int(v); }
    }
    return arg;
}

}  // namespace

// ===========================================================================
// MTP draft head — owns its blk.64 weights + a private 1-layer KV cache.
// Drafts K tokens autoregressively given an initial (h_last, tn).
// ===========================================================================
struct MtpHead {
    // geometry
    uint32_t H, HD, N_q, N_qg, N_kv, F, rope_n, vocab, mtp_blk;
    float eps, rope_theta;
    uint32_t n_q_heads, n_kv_heads;

    // weights
    sycl::half *eh_proj, *w_attn_q, *w_attn_k, *w_attn_v, *w_attn_out;
    sycl::half *w_ffn_gate, *w_ffn_up, *w_ffn_down, *w_lm_head;
    float *enorm, *hnorm, *shead_norm, *attn_norm, *post_attn_norm,
          *attn_q_norm, *attn_k_norm;
    void* te_dev = nullptr; ie::DType te_dtype = ie::DType::kCount;

    // its own 1-layer KV (max_ctx)
    sycl::half *mtp_kc = nullptr, *mtp_vc = nullptr;
    uint32_t max_ctx = 0;

    // per-step T=1 scratch
    sycl::half *d_h, *d_e, *d_hn, *d_en, *d_cat, *d_x, *d_xn, *d_qg, *d_q,
               *d_gate, *d_k, *d_v, *d_ao, *d_blk, *d_fg, *d_fu, *d_fh,
               *d_logits1;
    int32_t* d_pos1;
    int32_t* d_tok1;
    GemmScratch gs;

    void alloc_scratch(sycl::queue& q) {
        auto A = [&](uint64_t n){ return sycl::malloc_device<sycl::half>(n, q); };
        d_h=A(H); d_e=A(H); d_hn=A(H); d_en=A(H); d_cat=A(2u*H); d_x=A(H);
        d_xn=A(H); d_qg=A(N_qg); d_q=A(N_q); d_gate=A(N_q); d_k=A(N_kv);
        d_v=A(N_kv); d_ao=A(N_q); d_blk=A(H); d_fg=A(F); d_fu=A(F); d_fh=A(F);
        d_logits1=A(vocab);
        d_pos1 = sycl::malloc_device<int32_t>(1, q);
        d_tok1 = sycl::malloc_device<int32_t>(1, q);
        mtp_kc = A(uint64_t(n_kv_heads) * max_ctx * HD);
        mtp_vc = A(uint64_t(n_kv_heads) * max_ctx * HD);
    }

    // Run blk.64 over a single residual-stream vector d_x (already eh_proj'd),
    // at attention position `pos` (writes into mtp KV at that pos). Leaves the
    // post-FFN residual stream in d_x, and the lm_head logits in d_logits1.
    //
    // PERF: ALL matmuls are M=1 GEMVs → use gemv_fp16 (a real GEMV kernel), not
    // gemm_fp16 (a prefill GEMM tuned for M>=2 — disastrous at M=1). The in-order
    // queue chains everything; we only sync once (in draft()) to read logits.
    void run_layer(sycl::queue& q, int32_t pos) {
        q.memcpy(d_pos1, &pos, sizeof(int32_t));
        ie::rms_norm_f32w(q, d_x, attn_norm, d_xn, 1, H, eps);
        ie::gemv_fp16(q, d_xn, w_attn_q, d_qg, H, N_qg);
        ie::split_q_gate_per_head(q, d_qg, d_q, d_gate, 1, n_q_heads, HD);
        ie::gemv_fp16(q, d_xn, w_attn_k, d_k, H, N_kv);
        ie::gemv_fp16(q, d_xn, w_attn_v, d_v, H, N_kv);
        ie::rms_norm_f32w(q, d_q, attn_q_norm, d_q, n_q_heads,  HD, eps);
        ie::rms_norm_f32w(q, d_k, attn_k_norm, d_k, n_kv_heads, HD, eps);
        ie::rope_partial(q, d_q, d_pos1, d_q, 1, n_q_heads,  HD, rope_n, rope_theta);
        ie::rope_partial(q, d_k, d_pos1, d_k, 1, n_kv_heads, HD, rope_n, rope_theta);
        ie::full_attention(q, d_q, d_k, d_v, mtp_kc, mtp_vc, d_ao,
                           1, uint32_t(pos), n_q_heads, n_kv_heads, HD, max_ctx);
        ie::sigmoid_gate(q, d_ao, d_gate, d_ao, uint64_t(N_q));
        ie::gemv_fp16(q, d_ao, w_attn_out, d_blk, N_q, H);
        ie::residual_add(q, d_x, d_blk, d_x, uint64_t(H));
        ie::rms_norm_f32w(q, d_x, post_attn_norm, d_xn, 1, H, eps);
        ie::gemv_fp16(q, d_xn, w_ffn_gate, d_fg, H, F);
        ie::gemv_fp16(q, d_xn, w_ffn_up,   d_fu, H, F);
        ie::swiglu(q, d_fg, d_fu, d_fh, uint64_t(F));
        ie::gemv_fp16(q, d_fh, w_ffn_down, d_blk, F, H);
        ie::residual_add(q, d_x, d_blk, d_x, uint64_t(H));
        ie::rms_norm_f32w(q, d_x, shead_norm, d_xn, 1, H, eps);
        ie::gemv_fp16(q, d_xn, w_lm_head, d_logits1, H, vocab);
    }

    // Embed a single token into d_e (fp16 [H]).
    void embed(sycl::queue& q, int32_t tok) {
        q.memcpy(d_tok1, &tok, sizeof(int32_t));
        if (te_dtype == ie::DType::kQ4_K)      ie::embedding_lookup_q4k(q, d_tok1, te_dev, d_e, 1, H);
        else if (te_dtype == ie::DType::kQ6_K) ie::embedding_lookup_q6k(q, d_tok1, te_dev, d_e, 1, H);
        else                                   ie::embedding_lookup_q8_0(q, d_tok1, te_dev, d_e, 1, H);
    }

    // Build x = eh_proj(concat(enorm(e_in), hnorm(h_in))) into d_x.
    // h_src is a device fp16 [H] pointer (h_last or prior mtp residual stream).
    void build_x(sycl::queue& q, const sycl::half* h_src, int32_t e_tok) {
        embed(q, e_tok);                                  // d_e = embed(e_tok)
        q.memcpy(d_h, h_src, uint64_t(H) * sizeof(sycl::half));
        ie::rms_norm_f32w(q, d_e, enorm, d_en, 1, H, eps);
        ie::rms_norm_f32w(q, d_h, hnorm, d_hn, 1, H, eps);
        // concat [enorm(e) ; hnorm(h)]  (spec-validated order)
        q.parallel_for(sycl::range<1>(uint64_t(H)), [=, en=d_en, hn=d_hn, cat=d_cat, H=H](sycl::id<1> i) {
            cat[uint64_t(i)]       = en[uint64_t(i)];
            cat[uint64_t(i) + H]   = hn[uint64_t(i)];
        });
        ie::gemv_fp16(q, d_cat, eh_proj, d_x, 2u * H, H);
    }

    // Draft K tokens from (h_last device-ptr, tn). Returns K drafted tokens in
    // `out` (out[0]=g_1 .. out[K-1]=g_K), grows mtp KV from pos = p_base.
    // The verify INPUT sequence is [tn, out[0], .., out[K-2]] (length K).
    double t_buildlayer = 0, t_copyback = 0, t_argmax = 0; uint64_t n_steps = 0;
    void draft(sycl::queue& q, const sycl::half* h_last, int32_t tn,
               uint32_t p_base, uint32_t K, std::vector<int32_t>& out) {
        out.clear();
        int32_t e_tok = tn;
        const sycl::half* h_src = h_last;
        std::vector<sycl::half> row(vocab);
        for (uint32_t j = 0; j < K; ++j) {
            auto a0 = std::chrono::steady_clock::now();
            build_x(q, h_src, e_tok);            // x from (h_src, e_tok)
            run_layer(q, int32_t(p_base + j));   // advance mtp KV at pos p_base+j
            q.wait();
            auto a1 = std::chrono::steady_clock::now();
            q.memcpy(row.data(), d_logits1, uint64_t(vocab) * sizeof(sycl::half)).wait();
            auto a2 = std::chrono::steady_clock::now();
            int32_t g = argmax_row(row.data(), vocab);
            auto a3 = std::chrono::steady_clock::now();
            t_buildlayer += std::chrono::duration<double>(a1-a0).count();
            t_copyback   += std::chrono::duration<double>(a2-a1).count();
            t_argmax     += std::chrono::duration<double>(a3-a2).count();
            ++n_steps;
            out.push_back(g);
            // next step: h = this step's residual stream (d_x), e = g
            h_src = d_x;     // d_x is overwritten next build_x, but build_x reads
                             // h_src into d_h FIRST (memcpy) before touching d_x,
                             // so passing d_x is safe across the boundary.
            e_tok = g;
        }
    }
};

int main(int argc, char** argv) {
    std::string gguf_path =
        "/home/weezy/models/bartowski/Qwen3.6-27B-GGUF/Qwen_Qwen3.6-27B-Q4_K_M.gguf";
    std::string prompt =
        "You are a cybersecurity expert. Explain in detail how a buffer overflow "
        "vulnerability works and how an attacker can exploit it to gain control.";
    uint32_t ntok = 64;
    uint32_t K = 4;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf"   && i + 1 < argc) gguf_path = argv[++i];
        else if (a == "--prompt" && i + 1 < argc) prompt    = argv[++i];
        else if (a == "--ntok"   && i + 1 < argc) ntok      = std::atoi(argv[++i]);
        else if (a == "--K"      && i + 1 < argc) K         = std::atoi(argv[++i]);
    }

    std::printf("ie-qwen35-spec — STEP 1b+2 MTP self-speculative decode (LOSSLESS gate)\n");
    std::printf("  gguf : %s\n  ntok : %u   K : %u\n", gguf_path.c_str(), ntok, K);

    ie::GgufReader g;
    if (auto e = g.open(gguf_path); !e.empty()) { std::fprintf(stderr, "gguf: %s\n", e.c_str()); return 1; }
    ie::Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tok: %s\n", e.c_str()); return 1; }
    ie::DeviceAllocator alloc;
    if (auto e = alloc.init(); !e.empty()) { std::fprintf(stderr, "alloc: %s\n", e.c_str()); return 1; }
    auto& q = alloc.queue();
    std::printf("  device : %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    ie::Qwen35Config qcfg;
    if (auto e = ie::read_qwen35_config(g, qcfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }
    const ie::DenseConfig& dc = qcfg.dense;
    const uint32_t H = dc.hidden, HD = dc.head_dim;
    const uint32_t N_q = dc.n_q_heads * HD, N_qg = N_q * 2u, N_kv = dc.n_kv_heads * HD;
    const uint32_t F = dc.ffn, rope_n = dc.rope_dim, vocab = dc.vocab;
    const float eps = dc.rms_eps;
    const uint32_t mtp_blk = qcfg.n_transformer_layers();   // 64

    ie::Qwen35DenseModel model;
    std::printf("  loading 27B main model...\n");
    if (auto e = model.load(alloc, g, qcfg); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }

    // --- prompt tokens ---
    auto ids = tok.encode(prompt, /*allow_special=*/false);
    if (tok.add_bos_token() && tok.bos_token_id() >= 0 &&
        (ids.empty() || ids.front() != tok.bos_token_id()))
        ids.insert(ids.begin(), tok.bos_token_id());
    const uint32_t P0 = uint32_t(ids.size());
    std::printf("  prompt tokens : %u\n", P0);

    // --- caches ---
    const uint32_t n_tx = qcfg.n_transformer_layers();
    const uint32_t L_full = n_tx / qcfg.full_attn_interval;
    const uint32_t L_lin  = n_tx - L_full;
    const uint32_t max_ctx = P0 + ntok + 8 + K;
    const uint32_t conv_ch = qcfg.ssm_inner + 2u * qcfg.ssm_n_k_heads * qcfg.ssm_state;

    auto make_kv = [&](ie::KvCache& kv) -> std::string {
        return kv.init(alloc, ie::KvCacheConfig{L_full, dc.n_kv_heads, max_ctx, HD});
    };
    auto make_dn = [&](ie::DeltaNetState& dn) -> std::string {
        return dn.init(alloc, ie::DeltaNetStateConfig{
            L_lin, qcfg.ssm_n_v_heads, qcfg.ssm_state, qcfg.ssm_state, conv_ch, qcfg.ssm_conv_kernel});
    };

    if (auto e = model.ensure_workspace(std::max<uint32_t>(P0, K)); !e.empty()) { std::fprintf(stderr, "ws: %s\n", e.c_str()); return 1; }
    if (auto e = model.ensure_attn_partials(max_ctx); !e.empty()) { std::fprintf(stderr, "partials: %s\n", e.c_str()); return 1; }

    auto* d_logits = sycl::malloc_device<sycl::half>(vocab, q);
    auto* d_ids_scr = sycl::malloc_device<int32_t>(std::max<uint32_t>(P0, K) + 8, q);

    // ======================================================================
    // (A) Plain greedy decode — the oracle.
    // ======================================================================
    auto plain_greedy = [&](std::vector<int32_t>& outtoks, double& tokps) {
        ie::KvCache kv; ie::DeltaNetState dn;
        if (auto e = make_kv(kv); !e.empty()) { std::fprintf(stderr, "kv: %s\n", e.c_str()); std::exit(1); }
        if (auto e = make_dn(dn); !e.empty()) { std::fprintf(stderr, "dn: %s\n", e.c_str()); std::exit(1); }
        kv.reset(); dn.reset(q);

        // prefill the prompt
        q.memcpy(d_ids_scr, ids.data(), P0 * sizeof(int32_t)).wait();
        model.forward(q, d_ids_scr, P0, 0, kv, dn, d_logits).wait();
        std::vector<sycl::half> row(vocab);
        q.memcpy(row.data(), d_logits, uint64_t(vocab) * sizeof(sycl::half)).wait();
        int32_t next = argmax_row(row.data(), vocab);

        outtoks.clear();
        uint32_t pos = P0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t s = 0; s < ntok; ++s) {
            outtoks.push_back(next);
            q.memcpy(d_ids_scr, &next, sizeof(int32_t)).wait();
            model.forward(q, d_ids_scr, 1, pos, kv, dn, d_logits).wait();
            ++pos;
            q.memcpy(row.data(), d_logits, uint64_t(vocab) * sizeof(sycl::half)).wait();
            next = argmax_row(row.data(), vocab);
        }
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        tokps = ntok / secs;
    };

    // ======================================================================
    // (B) MTP head setup.
    // ======================================================================
    MtpHead mtp;
    mtp.H=H; mtp.HD=HD; mtp.N_q=N_q; mtp.N_qg=N_qg; mtp.N_kv=N_kv; mtp.F=F;
    mtp.rope_n=rope_n; mtp.vocab=vocab; mtp.mtp_blk=mtp_blk; mtp.eps=eps;
    mtp.rope_theta=dc.rope_theta; mtp.n_q_heads=dc.n_q_heads;
    mtp.n_kv_heads=dc.n_kv_heads; mtp.max_ctx=max_ctx;
    {
        char buf[64]; std::string err;
        auto MT = [&](const char* name) {
            std::snprintf(buf, sizeof(buf), "blk.%u.%s", mtp_blk, name);
            return g.find_tensor(buf);
        };
        auto reqf16 = [&](const char* n) -> sycl::half* {
            auto* p = dequant_q8_0_weight(alloc, MT(n), err);
            if (!err.empty()) { std::fprintf(stderr, "blk.64.%s: %s\n", n, err.c_str()); std::exit(1); }
            return p;
        };
        auto reqf32 = [&](const char* n) -> float* {
            auto* p = upload_f32(alloc, MT(n), err);
            if (!err.empty()) { std::fprintf(stderr, "blk.64.%s: %s\n", n, err.c_str()); std::exit(1); }
            return p;
        };
        mtp.eh_proj    = reqf16("nextn.eh_proj.weight");
        mtp.enorm      = reqf32("nextn.enorm.weight");
        mtp.hnorm      = reqf32("nextn.hnorm.weight");
        mtp.shead_norm = reqf32("nextn.shared_head_norm.weight");
        mtp.w_attn_q   = reqf16("attn_q.weight");
        mtp.w_attn_k   = reqf16("attn_k.weight");
        mtp.w_attn_v   = reqf16("attn_v.weight");
        mtp.w_attn_out = reqf16("attn_output.weight");
        mtp.attn_norm      = reqf32("attn_norm.weight");
        mtp.post_attn_norm = reqf32("post_attention_norm.weight");
        mtp.attn_q_norm    = reqf32("attn_q_norm.weight");
        mtp.attn_k_norm    = reqf32("attn_k_norm.weight");
        mtp.w_ffn_gate = reqf16("ffn_gate.weight");
        mtp.w_ffn_up   = reqf16("ffn_up.weight");
        mtp.w_ffn_down = reqf16("ffn_down.weight");
        // shared lm_head = output.weight
        const auto* o = g.find_tensor("output.weight");
        const uint32_t Ko = uint32_t(o->shape[0]), Nv = uint32_t(o->shape[1]);
        void* packed = alloc.malloc(o->nbytes);
        q.memcpy(packed, o->data, o->nbytes).wait();
        mtp.w_lm_head = static_cast<sycl::half*>(alloc.malloc(uint64_t(Ko) * Nv * sizeof(sycl::half)));
        if (o->dtype == ie::DType::kQ6_K)      ie::dequant_q6_K_to_Bt(q, packed, mtp.w_lm_head, Ko, Nv).wait();
        else if (o->dtype == ie::DType::kQ8_0) ie::dequant_q8_0_to_Bt(q, packed, mtp.w_lm_head, Ko, Nv).wait();
        else if (o->dtype == ie::DType::kQ4_K) ie::dequant_q4_K_to_Bt(q, packed, mtp.w_lm_head, Ko, Nv).wait();
        else { std::fprintf(stderr, "output.weight dtype unsupported\n"); return 1; }
        alloc.free(packed);
        // token_embd (independent upload for the lookup)
        const auto* te = g.find_tensor("token_embd.weight");
        mtp.te_dtype = te->dtype;
        mtp.te_dev = alloc.malloc(te->nbytes);
        q.memcpy(mtp.te_dev, te->data, te->nbytes).wait();
        mtp.alloc_scratch(q);
    }
    std::printf("  blk.64 MTP head loaded.\n");

    // ======================================================================
    // (C) Spec greedy decode.
    // ======================================================================
    auto spec_greedy = [&](std::vector<int32_t>& outtoks, double& tokps,
                           double& mean_accept, double& tok_per_fwd) {
        ie::KvCache kv; ie::DeltaNetState dn;
        if (auto e = make_kv(kv);   !e.empty()) { std::fprintf(stderr, "kv: %s\n", e.c_str()); std::exit(1); }
        if (auto e = make_dn(dn);   !e.empty()) { std::fprintf(stderr, "dn: %s\n", e.c_str()); std::exit(1); }
        kv.reset(); dn.reset(q);

        // Per-position DeltaNet checkpoint — eliminates the rollback re-forward.
        ie::Qwen35SpecCheckpoint ckpt;
        if (auto e = ckpt.init(alloc, dn, K); !e.empty()) {
            std::fprintf(stderr, "ckpt: %s\n", e.c_str()); std::exit(1);
        }

        // verify-mode scratch: all_logits [K,vocab], hidden_pre_norm [K,H]
        auto* d_all  = sycl::malloc_device<sycl::half>(uint64_t(K) * vocab, q);
        auto* d_hid  = sycl::malloc_device<sycl::half>(uint64_t(K) * H, q);
        auto* d_hlast = sycl::malloc_device<sycl::half>(uint64_t(H), q);   // current h_last
        std::vector<sycl::half> Lrow(uint64_t(K) * vocab), Hrow(uint64_t(K) * H);

        // --- prefill prompt, also grab pre-output_norm hidden at last pos ---
        // forward(T=P0) with hidden_pre_norm over all P0 → take row P0-1.
        auto* d_hid_pf = sycl::malloc_device<sycl::half>(uint64_t(P0) * H, q);
        q.memcpy(d_ids_scr, ids.data(), P0 * sizeof(int32_t)).wait();
        model.forward(q, d_ids_scr, P0, 0, kv, dn, d_logits,
                      /*all_logits=*/nullptr, /*hidden_pre_norm=*/d_hid_pf).wait();
        std::vector<sycl::half> row(vocab);
        q.memcpy(row.data(), d_logits, uint64_t(vocab) * sizeof(sycl::half)).wait();
        int32_t tn = argmax_row(row.data(), vocab);   // target argmax @ last committed pos
        q.memcpy(d_hlast, d_hid_pf + uint64_t(P0 - 1) * H,
                 uint64_t(H) * sizeof(sycl::half)).wait();
        sycl::free(d_hid_pf, q);

        outtoks.clear();
        uint32_t p = P0;          // committed prefix length (= next write position)
        uint64_t n_verifies = 0, n_accepted_total = 0;
        double t_verify = 0, t_rollback = 0; uint64_t n_verify_fwd = 0, n_rollback_fwd = 0;

        auto t0 = std::chrono::steady_clock::now();
        while (outtoks.size() < ntok) {
            // ---- 1. DRAFT K tokens with the MTP head (its own KV grows from p) ----
            // Reset the MTP KV each round: lengths are positional, full_attention
            // reads [0, pos]; we re-seed from pos=p, but earlier MTP KV rows (<p)
            // are stale drafts. To keep it correct & simple we ZERO-rewind: the
            // MTP attention only needs causal context over the CURRENT draft burst
            // plus the committed prefix's MTP keys. We DON'T have committed-prefix
            // MTP keys (we never ran the MTP head over accepted tokens), so the
            // MTP head must attend only within the draft burst. => run draft with
            // pos starting at 0 in a FRESH per-round MTP KV window.
            std::vector<int32_t> drafted;
            mtp.draft(q, d_hlast, tn, /*p_base=*/0, K, drafted);
            // drafted[0..K-1] = g_1..g_K ; verify input = [tn, g_1, .., g_{K-1}]

            // ---- 2. VERIFY with the target forward(T=K, start_pos=p) ----
            // CHECKPOINT MODE: the verify snapshots the DeltaNet recurrent+conv
            // state AFTER each of the K positions, per linear layer, so a partial
            // accept commits via commit_to_n() — NO rollback re-forward. KV length
            // before the verify (= p for every full layer; truncate-on-commit).
            std::vector<uint32_t> kv_len_snap(L_full);
            for (uint32_t l = 0; l < L_full; ++l) kv_len_snap[l] = kv.length(l);

            std::vector<int32_t> vin(K);
            vin[0] = tn;
            for (uint32_t j = 1; j < K; ++j) vin[j] = drafted[j - 1];
            q.memcpy(d_ids_scr, vin.data(), K * sizeof(int32_t)).wait();
            auto vt0 = std::chrono::steady_clock::now();
            model.forward(q, d_ids_scr, K, p, kv, dn, d_logits,
                          /*all_logits=*/d_all, /*hidden_pre_norm=*/d_hid,
                          /*ckpt=*/&ckpt).wait();
            auto vt1 = std::chrono::steady_clock::now();
            t_verify += std::chrono::duration<double>(vt1-vt0).count(); ++n_verify_fwd;
            q.memcpy(Lrow.data(), d_all, uint64_t(K) * vocab * sizeof(sycl::half)).wait();
            q.memcpy(Hrow.data(), d_hid, uint64_t(K) * H * sizeof(sycl::half)).wait();

            // L[j] = argmax = TRUE token after input position j.
            // committed tokens this round: tn (pos p, already truth), then accept
            // g_j == argmax(L[j-1]) for j=1.. while matching.
            std::vector<int32_t> targ(K);
            for (uint32_t j = 0; j < K; ++j)
                targ[j] = argmax_row(Lrow.data() + uint64_t(j) * vocab, vocab);

            uint32_t n = 0;   // number of DRAFTED tokens accepted (g_1..g_n)
            for (uint32_t j = 1; j < K; ++j) {
                if (drafted[j - 1] == targ[j - 1]) ++n;   // g_j matches target@(j-1)
                else break;
            }
            // committed this round: tn + g_1..g_n + bonus(=targ[n]).
            // committed token ids, in order:
            std::vector<int32_t> committed;
            committed.push_back(tn);
            for (uint32_t j = 0; j < n; ++j) committed.push_back(drafted[j]);
            const int32_t bonus = targ[n];                 // argmax(L[n]) = next tn'
            // accepted position count for state = tn + g_1..g_n = n+1 input rows.
            const uint32_t accepted = n + 1;               // rows of vin to keep

            n_verifies++;
            n_accepted_total += accepted;   // tokens the target "produced" per fwd
                                            // (committed before the bonus)

            // ---- 3. COMMIT state to exactly p+accepted (committed prefix WITHOUT
            //         the bonus token — bonus becomes next tn). NO re-forward:
            //         pick the per-position snapshot taken at verify-row accepted-1
            //         and copy it into the live DeltaNet state; truncate the KV
            //         length to p+accepted (full_attention reads [0,len); extra
            //         verify keys past accepted are overwritten next round). ----
            if (accepted < K) {
                auto rt0 = std::chrono::steady_clock::now();
                if (auto e = ckpt.commit_to_n(q, dn, accepted); !e.empty()) {
                    std::fprintf(stderr, "commit_to_n: %s\n", e.c_str()); std::exit(1);
                }
                for (uint32_t l = 0; l < L_full; ++l)
                    kv.set_length(l, kv_len_snap[l] + accepted);
                auto rt1 = std::chrono::steady_clock::now();
                t_rollback += std::chrono::duration<double>(rt1-rt0).count(); ++n_rollback_fwd;
            }
            // else: all K verify rows accepted → dn/kv already at p+K = p+accepted.

            // ---- 4. advance bookkeeping: emit committed[1..] (the NEW tokens) ----
            // committed[0]=tn was ALREADY emitted in the previous round (it is the
            // bonus of the previous verify). EXCEPTION: the very first round, tn
            // came from prefill and has not been emitted → emit it too.
            // Unify: emit g_1..g_n + bonus; emit tn only on round 0.
            if (n_verifies == 1) outtoks.push_back(tn);
            for (uint32_t j = 0; j < n; ++j) {
                if (outtoks.size() >= ntok) break;
                outtoks.push_back(drafted[j]);
            }
            if (outtoks.size() < ntok) outtoks.push_back(bonus);

            // next round: tn' = bonus, h_last' = pre-output_norm hidden at the
            // accepted boundary row = Hrow[accepted-1] (the row whose argmax is
            // the bonus token).
            tn = bonus;
            q.memcpy(d_hlast, d_hid + uint64_t(accepted - 1) * H,
                     uint64_t(H) * sizeof(sycl::half)).wait();
            p += accepted;
        }
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();
        if (outtoks.size() > ntok) outtoks.resize(ntok);
        tokps = outtoks.size() / secs;
        mean_accept = n_verifies ? double(n_accepted_total) / n_verifies : 0.0;
        tok_per_fwd = mean_accept;   // tokens committed per target forward (verify)
                                     // (rollback re-forward not counted — see note)
        std::printf("  [target profile] verify fwd: %.1f ms x%llu  |  commit (no re-fwd): %.2f ms x%llu  | RE-FORWARDS: 0\n",
                    n_verify_fwd ? 1e3*t_verify/n_verify_fwd : 0.0, (unsigned long long)n_verify_fwd,
                    n_rollback_fwd ? 1e3*t_rollback/n_rollback_fwd : 0.0, (unsigned long long)n_rollback_fwd);

        sycl::free(d_all, q); sycl::free(d_hid, q); sycl::free(d_hlast, q);
    };

    // ======================================================================
    // Run both, compare.
    // ======================================================================
    std::vector<int32_t> plain_toks, spec_toks;
    double plain_tps = 0, spec_tps = 0, mean_acc = 0, tpf = 0;

    std::printf("\n[plain] greedy decode (oracle), N=%u ...\n", ntok);
    plain_greedy(plain_toks, plain_tps);          // warm-up + measured (same run)
    plain_greedy(plain_toks, plain_tps);          // 2nd run = warm measurement
    std::printf("  plain greedy : %.2f tok/s\n", plain_tps);

    std::printf("\n[spec] MTP self-speculative decode, K=%u ...\n", K);
    spec_greedy(spec_toks, spec_tps, mean_acc, tpf);   // warm-up
    spec_greedy(spec_toks, spec_tps, mean_acc, tpf);   // measured
    std::printf("  spec greedy  : %.2f tok/s   mean accepted/verify : %.3f\n",
                spec_tps, mean_acc);
    std::printf("  [draft profile] steps=%llu  build+layer=%.1f ms/step  copyback=%.1f ms/step  argmax=%.1f ms/step\n",
                (unsigned long long)mtp.n_steps,
                mtp.n_steps ? 1e3*mtp.t_buildlayer/mtp.n_steps : 0.0,
                mtp.n_steps ? 1e3*mtp.t_copyback/mtp.n_steps : 0.0,
                mtp.n_steps ? 1e3*mtp.t_argmax/mtp.n_steps : 0.0);

    // ---- LOSSLESS comparison ----
    bool lossless = (plain_toks.size() == spec_toks.size());
    int first_diff = -1;
    if (lossless) {
        for (size_t i = 0; i < plain_toks.size(); ++i)
            if (plain_toks[i] != spec_toks[i]) { lossless = false; first_diff = int(i); break; }
    } else {
        first_diff = int(std::min(plain_toks.size(), spec_toks.size()));
    }

    std::printf("\n=== RESULT ===\n");
    std::printf("  LOSSLESS                  : %s\n", lossless ? "YES (token-for-token identical)" : "NO");
    if (!lossless) {
        std::printf("  first differing position  : %d\n", first_diff);
        std::printf("  plain[%d]=%d  spec[%d]=%d\n", first_diff,
                    first_diff < int(plain_toks.size()) ? plain_toks[first_diff] : -1,
                    first_diff, first_diff < int(spec_toks.size()) ? spec_toks[first_diff] : -1);
        auto dump = [&](const char* lbl, const std::vector<int32_t>& t) {
            std::printf("  %s (first 24): ", lbl);
            for (size_t i = 0; i < std::min<size_t>(24, t.size()); ++i) std::printf("%d ", t[i]);
            std::printf("\n");
        };
        dump("plain", plain_toks);
        dump("spec ", spec_toks);
    }
    std::printf("  K                         : %u\n", K);
    std::printf("  mean accepted / verify    : %.3f\n", mean_acc);
    std::printf("  plain greedy tok/s        : %.2f\n", plain_tps);
    std::printf("  spec  greedy tok/s        : %.2f\n", spec_tps);
    std::printf("  net speedup               : %.2fx\n", plain_tps > 0 ? spec_tps / plain_tps : 0.0);
    {
        auto txt = tok.decode(spec_toks, /*skip_special=*/false);
        std::printf("\n  spec output text:\n%s\n", txt.c_str());
    }

    sycl::free(d_logits, q); sycl::free(d_ids_scr, q);
    return lossless ? 0 : 2;
}
