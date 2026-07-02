// tools/ie_gemma4_spec.cpp — Gemma-4 MTP self-speculative decode (Pieces 3+4).
//
// The gemma-4-31B (target, `gemma4`, 60 layers, Q4_0) is the source of truth. The
// official 930M `gemma4-assistant` MTP draft head (4 layers, hidden 1024, Q8_0)
// drafts K tokens autoregressively; the target verifies them in ONE forward(T=K)
// and accepts the longest argmax-matching prefix. Greedy acceptance ⇒ the emitted
// sequence is BIT-IDENTICAL to plain greedy decode (the LOSSLESS gate).
//
// What makes the head special vs the 27B NextN head:
//   * EAGLE recurrence: input = (token, target_hidden); output = (logits, h_next),
//     and h_next feeds the NEXT draft step's `inp_h`.  The embed uses the TARGET's
//     tok_embd (×√backbone); pre_proj fuses concat(embed, inp_h)[2*backbone]→1024.
//   * Q-ONLY attention SHARING the TARGET's KV cache (no wk/wv): SWA head layers
//     read target L(n-2), the global head layer reads target L(n-1).  All K draft
//     steps use the SAME position p (= committed length) — shared-memory semantics
//     (ref: llama.cpp common/speculative.cpp draft-mtp, "same position for all
//     draft tokens").  ⇒ read_attention_gemma (read-only, no append).
//   * NO DeltaNet → rollback is implicit: the target writes KV by absolute position,
//     so the next forward at start_pos = p+accepted overwrites the stale draft keys.
//
// Correctness gate: acceptance rate. Even a buggy head stays LOSSLESS (we only
// commit verified tokens), so a healthy acceptance (~0.6, matching llama on B70)
// is the signal the head forward is correct.
//
// usage: ie-gemma4-spec --gguf <31b.gguf> --head <mtp-head.gguf>
//        [--prompt <text>] [--ntok N] [--K K]
// Authoritative head forward: ~/llama.cpp/src/models/gemma4-assistant.cpp.

#include "ie/allocator.hpp"
#include "ie/dequant.hpp"
#include "ie/gemma4.hpp"
#include "ie/gemma4_assistant.hpp"
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/model_config.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"   // block_q8_0 / block_q8_1x
#include "ie/tokenizer.hpp"

#include <sycl/sycl.hpp>

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

// SoA-Q8_0 weight (int8 qs col-contiguous + raw fp16 d per 32-block) for the
// int-dot W8A8 decode GEMV (gemv_q8_0_soa_q8). HALVES the head's per-step bytes
// vs the fp16 dequant — and is CLOSER to the original Q8_0 (int-dot, no fp16
// round-trip). Repack mirrors qwen35_split.cpp::build_split.
struct Q8W { int8_t* qs = nullptr; uint16_t* d = nullptr; uint32_t K = 0, N = 0; };

Q8W repack_q8_0_soa(ie::DeviceAllocator& alloc, const ie::GgufTensorInfo* t, std::string& err) {
    Q8W w{};
    if (!t) { err = "tensor not found"; return w; }
    if (t->dtype != ie::DType::kQ8_0) { err = "expected Q8_0"; return w; }
    w.K = uint32_t(t->shape[0]); w.N = uint32_t(t->shape[1]);
    if (w.K % 32 != 0) { err = "Q8_0 SoA: K%32!=0"; return w; }
    const uint32_t K = w.K, N = w.N, bpc = K / 32;
    const auto* blocks = reinterpret_cast<const ie::block_q8_0*>(t->data);
    std::vector<int8_t>   qs(uint64_t(N) * K);
    std::vector<uint16_t> dd(uint64_t(N) * bpc);
    for (uint64_t n = 0; n < N; ++n)
        for (uint32_t b = 0; b < bpc; ++b) {
            const ie::block_q8_0& blk = blocks[n * bpc + b];
            dd[n * bpc + b] = *reinterpret_cast<const uint16_t*>(&blk.d);
            for (int i = 0; i < 32; ++i) qs[n * K + uint64_t(b) * 32 + i] = blk.qs[i];
        }
    w.qs = static_cast<int8_t*>(alloc.malloc(qs.size()));
    w.d  = static_cast<uint16_t*>(alloc.malloc(dd.size() * sizeof(uint16_t)));
    if (!w.qs || !w.d) { err = "Q8_0 SoA alloc"; return w; }
    alloc.queue().memcpy(w.qs, qs.data(), qs.size()).wait();
    alloc.queue().memcpy(w.d,  dd.data(), dd.size() * sizeof(uint16_t)).wait();
    return w;
}

float* upload_f32(ie::DeviceAllocator& alloc, const ie::GgufTensorInfo* t, std::string& err) {
    if (!t) { err = "tensor not found"; return nullptr; }
    if (t->dtype != ie::DType::kF32) { err = "expected F32"; return nullptr; }
    void* d = alloc.malloc(t->nbytes);
    if (!d) { err = "malloc f32"; return nullptr; }
    alloc.queue().memcpy(d, t->data, t->nbytes).wait();
    return static_cast<float*>(d);
}

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
// Gemma-4 MTP draft head — owns its 4-layer weights (Q8_0 → fp16) and reads the
// TARGET model's KV cache (shared, read-only). Drafts K tokens autoregressively.
// ===========================================================================
struct GemmaMtpHead {
    // geometry (head)
    uint32_t H = 0;          // head hidden (1024)
    uint32_t BB = 0;         // backbone / target hidden (5376)
    uint32_t F = 0;          // head ffn (8192)
    uint32_t vocab = 0;
    uint32_t n_q = 0;        // query heads (32)
    float    eps = 1e-6f;
    uint32_t target_n_layers = 0;

    struct LW {
        bool     is_swa = true;
        uint32_t head_dim = 0, n_kv = 0, n_rot = 0;
        float    theta = 1e4f;
        float    out_scale = 1.0f;
        float   *attn_norm=nullptr, *attn_q_norm=nullptr, *post_attn_norm=nullptr,
                *ffn_norm=nullptr, *post_ffw_norm=nullptr, *rope_freqs=nullptr;
        Q8W wq, wo, fg, fu, fd;
    };
    std::vector<LW> L;

    // top-level
    Q8W pre_proj;    // [2*BB, H]
    Q8W post_proj;   // [H, BB]
    Q8W tok_embd;    // [H, vocab]  (head output/logits proj)
    float *output_norm=nullptr;
    void* d_actq8=nullptr;          // block_q8_1x activation stream (max head K)

    // target (for the embed input — uses the TARGET's tok_embd)
    const void* tgt_te=nullptr; ie::DType tgt_te_dt=ie::DType::kCount;

    // scratch (T=1)
    sycl::half *d_xemb=nullptr, *d_xh=nullptr, *d_inph=nullptr, *d_cur=nullptr,
               *d_curn=nullptr, *d_q=nullptr, *d_ao=nullptr, *d_blk=nullptr,
               *d_attnout=nullptr, *d_fg=nullptr, *d_fu=nullptr, *d_fh=nullptr,
               *d_hnext=nullptr, *d_logits=nullptr;
    int32_t *d_pos=nullptr, *d_tok=nullptr;
    uint32_t max_Nq = 0;

    uint32_t max_K = 0;   // widest head GEMV K (for the activation q8 stream)

    void alloc_scratch(sycl::queue& q) {
        auto A = [&](uint64_t n){ return sycl::malloc_device<sycl::half>(n, q); };
        d_xemb=A(BB); d_xh=A(2u*BB); d_inph=A(BB); d_cur=A(H); d_curn=A(H);
        d_q=A(max_Nq); d_ao=A(max_Nq); d_blk=A(H); d_attnout=A(H);
        d_fg=A(F); d_fu=A(F); d_fh=A(F); d_hnext=A(BB); d_logits=A(vocab);
        d_pos = sycl::malloc_device<int32_t>(1, q);
        d_tok = sycl::malloc_device<int32_t>(1, q);
        d_actq8 = sycl::malloc_device<ie::block_q8_1x>(max_K / 32, q);
    }

    static void scale_inplace(sycl::queue& q, sycl::half* x, float c, uint64_t n) {
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i){ x[i] = sycl::half(float(x[i]) * c); });
    }

    // y[N] = Q8_0-SoA W[K,N] · q8(in[K])  (int-dot W8A8 decode GEMV).
    void gemv(sycl::queue& q, const sycl::half* in, const Q8W& W, sycl::half* y) {
        sycl::event qe = ie::quantize_q8_1(q, in, d_actq8, W.K);
        ie::gemv_q8_0_soa_q8(q, d_actq8, W.qs, W.d, y, W.K, W.N, {qe});
    }

    // x = pre_proj( concat( target_embd(tok)×√BB , inp_h ) ) → d_cur [H].
    void build_x(sycl::queue& q, const sycl::half* inp_h, int32_t tok) {
        q.memcpy(d_tok, &tok, sizeof(int32_t));
        if (tgt_te_dt == ie::DType::kQ6_K)      ie::embedding_lookup_q6k(q, d_tok, tgt_te, d_xemb, 1, BB);
        else if (tgt_te_dt == ie::DType::kQ4_K) ie::embedding_lookup_q4k(q, d_tok, tgt_te, d_xemb, 1, BB);
        else                                    ie::embedding_lookup_q8_0(q, d_tok, tgt_te, d_xemb, 1, BB);
        scale_inplace(q, d_xemb, std::sqrt(float(BB)), BB);
        // concat [embed(BB) ; inp_h(BB)] → d_xh[2*BB]
        q.memcpy(d_xh,       d_xemb, uint64_t(BB) * sizeof(sycl::half));
        q.memcpy(d_xh + BB,  inp_h,  uint64_t(BB) * sizeof(sycl::half));
        gemv(q, d_xh, pre_proj, d_cur);
    }

    // Run the 4 head layers over d_cur at draft position `p` (read-only attention
    // over the target's KV[0, p)). Leaves draft logits in d_logits and the next
    // recurrence hidden in d_hnext.
    void run_layers(sycl::queue& q, const ie::Gemma4Model& tgt, uint32_t p) {
        const int32_t pos = int32_t(p);
        q.memcpy(d_pos, &pos, sizeof(int32_t));
        for (uint32_t il = 0; il < L.size(); ++il) {
            const LW& w = L[il];
            const uint32_t HD = w.head_dim, Nq = n_q * HD;
            const uint32_t Lsh = w.is_swa ? (target_n_layers - 2) : (target_n_layers - 1);

            ie::rms_norm_f32w(q, d_cur, w.attn_norm, d_curn, 1, H, eps);
            gemv(q, d_curn, w.wq, d_q);
            ie::rms_norm_f32w(q, d_q, w.attn_q_norm, d_q, n_q, HD, eps);     // per-head QK-norm
            scale_inplace(q, d_q, std::sqrt(float(HD)), uint64_t(Nq));       // net attn scale 1.0
            if (!w.is_swa && w.rope_freqs)
                ie::rope_partial_ff(q, d_q, d_pos, d_q, 1, n_q, HD, w.n_rot, w.theta, w.rope_freqs);
            else
                ie::rope_partial(q, d_q, d_pos, d_q, 1, n_q, HD, w.n_rot, w.theta);
            ie::read_attention_gemma(q, d_q, tgt.kcache(Lsh), tgt.vcache(Lsh), d_ao,
                                     /*ctx_len=*/p, n_q, w.n_kv, HD, tgt.kv_ctx());
            gemv(q, d_ao, w.wo, d_blk);
            ie::rms_norm_f32w(q, d_blk, w.post_attn_norm, d_blk, 1, H, eps);
            ie::residual_add(q, d_cur, d_blk, d_attnout, uint64_t(H));

            ie::rms_norm_f32w(q, d_attnout, w.ffn_norm, d_curn, 1, H, eps);
            gemv(q, d_curn, w.fg, d_fg);
            gemv(q, d_curn, w.fu, d_fu);
            ie::geglu(q, d_fg, d_fu, d_fh, uint64_t(F));
            gemv(q, d_fh, w.fd, d_blk);
            ie::rms_norm_f32w(q, d_blk, w.post_ffw_norm, d_blk, 1, H, eps);
            ie::residual_add(q, d_attnout, d_blk, d_cur, uint64_t(H));
            if (w.out_scale != 1.0f) scale_inplace(q, d_cur, w.out_scale, uint64_t(H));
        }
        ie::rms_norm_f32w(q, d_cur, output_norm, d_curn, 1, H, eps);
        gemv(q, d_curn, tok_embd,  d_logits);   // draft logits
        gemv(q, d_curn, post_proj, d_hnext);    // recurrence hidden
    }

    // Draft K tokens from (h_last device-ptr [BB], tn). out[0..K-1] = drafted.
    double t_step = 0; uint64_t n_steps = 0;
    void draft(sycl::queue& q, const ie::Gemma4Model& tgt, const sycl::half* h_last,
               int32_t tn, uint32_t p, uint32_t K, std::vector<int32_t>& out) {
        out.clear();
        q.memcpy(d_inph, h_last, uint64_t(BB) * sizeof(sycl::half)).wait();
        int32_t tok = tn;
        std::vector<sycl::half> row(vocab);
        for (uint32_t j = 0; j < K; ++j) {
            auto a0 = std::chrono::steady_clock::now();
            build_x(q, d_inph, tok);
            run_layers(q, tgt, p);
            q.wait();
            q.memcpy(row.data(), d_logits, uint64_t(vocab) * sizeof(sycl::half)).wait();
            int32_t g = argmax_row(row.data(), vocab);
            t_step += std::chrono::duration<double>(std::chrono::steady_clock::now() - a0).count();
            ++n_steps;
            out.push_back(g);
            // recurrence: inp_h ← h_next, tok ← g
            q.memcpy(d_inph, d_hnext, uint64_t(BB) * sizeof(sycl::half)).wait();
            tok = g;
        }
    }
};

int main(int argc, char** argv) {
    std::string gguf_path = "/home/weezy/models/google/gemma-4-31B-it-qat-q4_0-gguf/gemma-4-31B_q4_0-it.gguf";
    std::string head_path = "/home/weezy/models/google/gemma-4-31B-it-qat-q4_0-gguf/mtp-gemma-4-31B-it-Q8_0.gguf";
    std::string prompt = "Explain in detail how a buffer overflow vulnerability works and how it can be exploited.";
    uint32_t ntok = 64, K = 4;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--gguf"   && i + 1 < argc) gguf_path = argv[++i];
        else if (a == "--head"   && i + 1 < argc) head_path = argv[++i];
        else if (a == "--prompt" && i + 1 < argc) prompt    = argv[++i];
        else if (a == "--ntok"   && i + 1 < argc) ntok      = std::atoi(argv[++i]);
        else if (a == "--K"      && i + 1 < argc) K         = std::atoi(argv[++i]);
    }

    std::printf("ie-gemma4-spec — MTP self-speculative decode (LOSSLESS gate)\n");
    std::printf("  target : %s\n  head   : %s\n  ntok : %u   K : %u\n",
                gguf_path.c_str(), head_path.c_str(), ntok, K);

    ie::GgufReader g, gh;
    if (auto e = g.open(gguf_path);  !e.empty()) { std::fprintf(stderr, "gguf: %s\n", e.c_str()); return 1; }
    if (auto e = gh.open(head_path); !e.empty()) { std::fprintf(stderr, "head gguf: %s\n", e.c_str()); return 1; }

    ie::Tokenizer tok;
    if (auto e = tok.load_from_gguf(g); !e.empty()) { std::fprintf(stderr, "tok: %s\n", e.c_str()); return 1; }
    ie::DeviceAllocator alloc;
    if (auto e = alloc.init(); !e.empty()) { std::fprintf(stderr, "alloc: %s\n", e.c_str()); return 1; }
    auto& q = alloc.queue();
    std::printf("  device : %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    ie::GemmaConfig cfg;
    if (auto e = ie::read_gemma4_config(g, cfg); !e.empty()) { std::fprintf(stderr, "config: %s\n", e.c_str()); return 1; }
    ie::GemmaAssistantConfig hcfg;
    if (auto e = ie::read_gemma4_assistant_config(gh, hcfg); !e.empty()) { std::fprintf(stderr, "head config: %s\n", e.c_str()); return 1; }

    const uint32_t vocab = cfg.vocab;
    ie::Gemma4Model model;
    std::printf("  loading 31B target...\n");
    if (auto e = model.load(alloc, g, cfg); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }

    // prompt tokens
    auto ids = tok.encode(prompt, /*allow_special=*/true);
    const uint32_t P0 = uint32_t(ids.size());
    std::printf("  prompt tokens : %u\n", P0);

    const uint32_t max_ctx = P0 + ntok + 8 + K;
    if (auto e = model.ensure_workspace(std::max<uint32_t>(P0, K)); !e.empty()) { std::fprintf(stderr, "ws: %s\n", e.c_str()); return 1; }
    if (auto e = model.ensure_kv(max_ctx); !e.empty()) { std::fprintf(stderr, "kv: %s\n", e.c_str()); return 1; }

    auto* d_ids = sycl::malloc_device<int32_t>(std::max<uint32_t>(P0, K) + 8, q);
    auto* d_logits = sycl::malloc_device<sycl::half>(vocab, q);
    ie::KvCache kv;   // unused — Gemma self-manages KV

    // ======================================================================
    // (A) Plain greedy decode — the oracle.
    // ======================================================================
    auto plain_greedy = [&](std::vector<int32_t>& outtoks, double& tokps) {
        std::vector<sycl::half> row(vocab);
        q.memcpy(d_ids, ids.data(), P0 * sizeof(int32_t)).wait();
        model.forward(q, d_ids, P0, 0, kv, d_logits).wait();
        q.memcpy(row.data(), d_logits, uint64_t(vocab) * sizeof(sycl::half)).wait();
        int32_t next = argmax_row(row.data(), vocab);

        outtoks.clear();
        uint32_t pos = P0;
        auto t0 = std::chrono::steady_clock::now();
        for (uint32_t s = 0; s < ntok; ++s) {
            outtoks.push_back(next);
            q.memcpy(d_ids, &next, sizeof(int32_t)).wait();
            model.forward(q, d_ids, 1, pos, kv, d_logits).wait();
            ++pos;
            q.memcpy(row.data(), d_logits, uint64_t(vocab) * sizeof(sycl::half)).wait();
            next = argmax_row(row.data(), vocab);
        }
        tokps = ntok / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    };

    // ======================================================================
    // (B) MTP head setup (Q8_0 → fp16, correctness-first).
    // ======================================================================
    GemmaMtpHead mtp;
    mtp.H = hcfg.hidden; mtp.BB = hcfg.n_embd_backbone; mtp.F = hcfg.ffn;
    mtp.vocab = hcfg.vocab; mtp.n_q = hcfg.n_q_heads; mtp.eps = hcfg.rms_eps;
    mtp.target_n_layers = model.n_layers();
    mtp.tgt_te = model.token_embd(); mtp.tgt_te_dt = model.token_embd_dtype();
    if (mtp.vocab != vocab) { std::fprintf(stderr, "vocab mismatch head=%u target=%u\n", mtp.vocab, vocab); return 1; }
    {
        std::string err; char buf[96];
        auto HT  = [&](const char* n) { return gh.find_tensor(n); };
        auto HTl = [&](uint32_t l, const char* n) { std::snprintf(buf, sizeof(buf), "blk.%u.%s", l, n); return gh.find_tensor(buf); };
        auto qw = [&](const ie::GgufTensorInfo* t, const char* what) {
            Q8W w = repack_q8_0_soa(alloc, t, err);
            if (!err.empty()) { std::fprintf(stderr, "%s: %s\n", what, err.c_str()); std::exit(1); }
            mtp.max_K = std::max(mtp.max_K, w.K); return w; };
        auto f32 = [&](const ie::GgufTensorInfo* t, const char* what) {
            auto* p = upload_f32(alloc, t, err);
            if (!err.empty()) { std::fprintf(stderr, "%s: %s\n", what, err.c_str()); std::exit(1); } return p; };

        mtp.max_K = 0;
        mtp.pre_proj    = qw(HT("nextn.pre_projection.weight"),  "pre_projection");
        mtp.post_proj   = qw(HT("nextn.post_projection.weight"), "post_projection");
        mtp.tok_embd    = qw(HT("token_embd.weight"),            "token_embd");
        mtp.output_norm = f32(HT("output_norm.weight"),           "output_norm");
        float* rope_freqs_shared = nullptr;
        if (const auto* rf = HT("rope_freqs.weight")) rope_freqs_shared = f32(rf, "rope_freqs");

        mtp.L.resize(hcfg.n_layers);
        mtp.max_Nq = 0;
        for (uint32_t l = 0; l < hcfg.n_layers; ++l) {
            auto& w = mtp.L[l];
            w.is_swa   = hcfg.is_swa[l] != 0;
            w.head_dim = hcfg.head_dim[l];
            w.n_kv     = hcfg.n_kv_heads[l];
            w.n_rot    = hcfg.n_rot[l];
            w.theta    = hcfg.rope_theta[l];
            mtp.max_Nq = std::max(mtp.max_Nq, mtp.n_q * w.head_dim);
            w.attn_norm      = f32(HTl(l, "attn_norm.weight"), "attn_norm");
            w.attn_q_norm    = f32(HTl(l, "attn_q_norm.weight"), "attn_q_norm");
            w.post_attn_norm = f32(HTl(l, "post_attention_norm.weight"), "post_attn");
            w.ffn_norm       = f32(HTl(l, "ffn_norm.weight"), "ffn_norm");
            w.post_ffw_norm  = f32(HTl(l, "post_ffw_norm.weight"), "post_ffw_norm");
            if (!w.is_swa) w.rope_freqs = rope_freqs_shared;
            if (const auto* os = HTl(l, "layer_output_scale.weight"))
                w.out_scale = *reinterpret_cast<const float*>(os->data);
            w.wq = qw(HTl(l, "attn_q.weight"),     "attn_q");
            w.wo = qw(HTl(l, "attn_output.weight"), "attn_output");
            w.fg = qw(HTl(l, "ffn_gate.weight"),   "ffn_gate");
            w.fu = qw(HTl(l, "ffn_up.weight"),     "ffn_up");
            w.fd = qw(HTl(l, "ffn_down.weight"),   "ffn_down");
        }
        mtp.alloc_scratch(q);
    }
    std::printf("  head loaded: %zu layers, H=%u BB=%u F=%u\n", mtp.L.size(), mtp.H, mtp.BB, mtp.F);

    // ======================================================================
    // (C) Spec greedy decode.
    // ======================================================================
    auto spec_greedy = [&](std::vector<int32_t>& outtoks, double& tokps, double& mean_accept) {
        auto* d_all   = sycl::malloc_device<sycl::half>(uint64_t(K) * vocab, q);
        auto* d_hid   = sycl::malloc_device<sycl::half>(uint64_t(K) * mtp.BB, q);
        auto* d_hlast = sycl::malloc_device<sycl::half>(uint64_t(mtp.BB), q);
        std::vector<sycl::half> Lrow(uint64_t(K) * vocab);

        // prefill the prompt; grab pre-output_norm hidden at last pos.
        auto* d_hid_pf = sycl::malloc_device<sycl::half>(uint64_t(P0) * mtp.BB, q);
        q.memcpy(d_ids, ids.data(), P0 * sizeof(int32_t)).wait();
        model.forward(q, d_ids, P0, 0, kv, d_logits, /*hidden_pre_norm=*/d_hid_pf).wait();
        std::vector<sycl::half> row(vocab);
        q.memcpy(row.data(), d_logits, uint64_t(vocab) * sizeof(sycl::half)).wait();
        int32_t tn = argmax_row(row.data(), vocab);
        q.memcpy(d_hlast, d_hid_pf + uint64_t(P0 - 1) * mtp.BB, uint64_t(mtp.BB) * sizeof(sycl::half)).wait();
        sycl::free(d_hid_pf, q);

        outtoks.clear();
        uint32_t p = P0;
        uint64_t n_verifies = 0, n_accepted_total = 0;
        double t_draft = 0, t_verify = 0, t_host = 0;
        auto t0 = std::chrono::steady_clock::now();
        while (outtoks.size() < ntok) {
            // ---- 1. DRAFT K tokens (head reads target KV[0,p), fixed pos p) ----
            auto dt0 = std::chrono::steady_clock::now();
            std::vector<int32_t> drafted;
            mtp.draft(q, model, d_hlast, tn, p, K, drafted);
            auto dt1 = std::chrono::steady_clock::now();
            t_draft += std::chrono::duration<double>(dt1 - dt0).count();

            // ---- 2. VERIFY: target forward(T=K, start_pos=p) on [tn, g_1..g_{K-1}] ----
            std::vector<int32_t> vin(K);
            vin[0] = tn;
            for (uint32_t j = 1; j < K; ++j) vin[j] = drafted[j - 1];
            q.memcpy(d_ids, vin.data(), K * sizeof(int32_t)).wait();
            model.forward(q, d_ids, K, p, kv, d_logits, /*hidden_pre_norm=*/d_hid,
                          /*all_logits=*/d_all).wait();
            auto vt1 = std::chrono::steady_clock::now();
            t_verify += std::chrono::duration<double>(vt1 - dt1).count();
            q.memcpy(Lrow.data(), d_all, uint64_t(K) * vocab * sizeof(sycl::half)).wait();

            std::vector<int32_t> targ(K);
            for (uint32_t j = 0; j < K; ++j)
                targ[j] = argmax_row(Lrow.data() + uint64_t(j) * vocab, vocab);
            t_host += std::chrono::duration<double>(std::chrono::steady_clock::now() - vt1).count();

            // accept g_j == argmax(L[j-1]); committed = tn + g_1..g_n + bonus.
            uint32_t n = 0;
            for (uint32_t j = 1; j < K; ++j) { if (drafted[j - 1] == targ[j - 1]) ++n; else break; }
            const int32_t bonus = targ[n];
            const uint32_t accepted = n + 1;   // input rows kept (tn + g_1..g_n)

            n_verifies++; n_accepted_total += accepted;

            // ---- 3. emit NEW tokens; advance state (gemma KV needs NO rollback:
            //         next forward at start_pos=p+accepted overwrites stale keys) ----
            if (n_verifies == 1) outtoks.push_back(tn);
            for (uint32_t j = 0; j < n; ++j) { if (outtoks.size() >= ntok) break; outtoks.push_back(drafted[j]); }
            if (outtoks.size() < ntok) outtoks.push_back(bonus);

            tn = bonus;
            q.memcpy(d_hlast, d_hid + uint64_t(accepted - 1) * mtp.BB, uint64_t(mtp.BB) * sizeof(sycl::half)).wait();
            p += accepted;
        }
        if (outtoks.size() > ntok) outtoks.resize(ntok);
        tokps = outtoks.size() / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        mean_accept = n_verifies ? double(n_accepted_total) / n_verifies : 0.0;
        std::printf("  [round profile] verifies=%llu  draft=%.1f ms  verify=%.1f ms  host=%.1f ms  (per round)\n",
                    (unsigned long long)n_verifies,
                    n_verifies ? 1e3 * t_draft / n_verifies : 0.0,
                    n_verifies ? 1e3 * t_verify / n_verifies : 0.0,
                    n_verifies ? 1e3 * t_host / n_verifies : 0.0);
        sycl::free(d_all, q); sycl::free(d_hid, q); sycl::free(d_hlast, q);
    };

    // ======================================================================
    // Run both, compare.
    // ======================================================================
    std::vector<int32_t> plain_toks, spec_toks;
    double plain_tps = 0, spec_tps = 0, mean_acc = 0;

    std::printf("\n[plain] greedy decode (oracle), N=%u ...\n", ntok);
    plain_greedy(plain_toks, plain_tps);
    plain_greedy(plain_toks, plain_tps);
    std::printf("  plain greedy : %.2f tok/s\n", plain_tps);

    std::printf("\n[spec] MTP self-speculative decode, K=%u ...\n", K);
    spec_greedy(spec_toks, spec_tps, mean_acc);
    spec_greedy(spec_toks, spec_tps, mean_acc);
    std::printf("  spec greedy  : %.2f tok/s   mean accepted/verify : %.3f\n", spec_tps, mean_acc);
    std::printf("  [draft] steps=%llu  %.2f ms/step\n",
                (unsigned long long)mtp.n_steps, mtp.n_steps ? 1e3 * mtp.t_step / mtp.n_steps : 0.0);

    bool lossless = (plain_toks.size() == spec_toks.size());
    int first_diff = -1;
    if (lossless) {
        for (size_t i = 0; i < plain_toks.size(); ++i)
            if (plain_toks[i] != spec_toks[i]) { lossless = false; first_diff = int(i); break; }
    } else first_diff = int(std::min(plain_toks.size(), spec_toks.size()));

    std::printf("\n=== RESULT ===\n");
    std::printf("  LOSSLESS                  : %s\n", lossless ? "YES (token-for-token identical)" : "NO");
    if (!lossless) {
        std::printf("  first differing position  : %d\n", first_diff);
        auto dump = [&](const char* lbl, const std::vector<int32_t>& t) {
            std::printf("  %s (first 24): ", lbl);
            for (size_t i = 0; i < std::min<size_t>(24, t.size()); ++i) std::printf("%d ", t[i]);
            std::printf("\n");
        };
        dump("plain", plain_toks); dump("spec ", spec_toks);
    }
    std::printf("  K                         : %u\n", K);
    std::printf("  mean accepted / verify    : %.3f  (>1.0 ⇒ head forward correct)\n", mean_acc);
    std::printf("  plain greedy tok/s        : %.2f\n", plain_tps);
    std::printf("  spec  greedy tok/s        : %.2f\n", spec_tps);
    std::printf("  net speedup               : %.2fx\n", plain_tps > 0 ? spec_tps / plain_tps : 0.0);
    {
        auto txt = tok.decode(spec_toks, /*skip_special=*/true, {});
        std::printf("\n  spec output text:\n%s\n", txt.c_str());
    }

    sycl::free(d_logits, q); sycl::free(d_ids, q);
    return lossless ? 0 : 2;
}
