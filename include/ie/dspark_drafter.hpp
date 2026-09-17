// include/ie/dspark_drafter.hpp — dspark EAGLE-style block-diffusion drafter (M1).
//
// M1 SCOPE: the drafter FORWARD on recorded target-tap inputs, gated against the
// PrismML fork's CPU golden vectors (docs/q2_0_optimization/09_dspark_m1_port_plan.md).
// Loads its own drafter GGUF (arch "dspark") standalone; NO target model, NO
// spec_generate integration, NO qwen35_dense edits (those are M2 — see the
// M2 hook comments in src/model/dspark_drafter.cpp).
//
// One round = one `draft_block()` call and is fully SELF-CONTAINED: a ctx row's
// K/V depend only on its own tap row + absolute position and ctx rows never
// re-enter the residual stream (fork dspark.cpp header + plan §3 equivalence
// note), so the cumulative context is recomputed each call and NO persistent KV
// cache is needed for the M1 forward gate.
//
// Forward (verified vs fork src/models/dspark.cpp; block_size=4, 6 layers,
// H=5120, GQA 40/4, HD=128, vocab=248320):
//   A. target_ctx = RMSNorm_hidden_norm(fc(ctx_feat))          once, pre-loop
//   B. x = token_embd[draft_tokens] + log_snr_embed            pre-loop
//   C. per layer: h=rms(attn_norm,x);
//        K,V = Wk,Wv @ concat(target_ctx, h);  Q = Wq @ h  (draft rows only)
//        per-head QK-RMSNorm(128) -> NEOX RoPE(128, base 1e7) -> NON-CAUSAL attn
//        (4 queries over all Lc+4 keys) -> Wo -> +x -> square SwiGLU FFN -> +x
//   D. h_norm = rms(output_norm, x);  base_logits = output @ h_norm  (own Q4_1 head)
//   E. host markov resample -> block draft ids
//
// Reuses existing engine ops (include/ie/ops.hpp): gemm_fp16, rms_norm_f32w,
// rope_partial (NEOX), swiglu, residual_add, embedding_lookup_q2_0. The one new
// piece of device code is the NON-CAUSAL attention kernel (all existing engine
// attention kernels are causal), which lives privately in the .cpp.
#pragma once

#include "ie/allocator.hpp"
#include "ie/dtype.hpp"
#include "ie/gguf.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace ie {

struct DsparkDrafter {
    bool             loaded = false;
    DeviceAllocator* alloc  = nullptr;

    // --- geometry (all read from the drafter GGUF metadata) ---
    uint32_t H = 0, n_layer = 0, n_head = 0, n_head_kv = 0, HD = 0;
    uint32_t N_q = 0, N_kv = 0, F = 0, vocab = 0;
    uint32_t n_capture = 0, n_embd_cap = 0, block_size = 0, markov_rank = 0;
    int32_t  mask_token_id = 0;
    float    rope_theta = 0.f, eps = 0.f;
    float    min_log_snr = 0.f, max_log_snr = 0.f;
    bool     log_snr_on = false;
    uint32_t max_ctx_rows = 0;   // largest Lc the scratch is sized for

    // Target layer ids whose residual outputs the drafter conditions on (M2). Read
    // from dspark.dspark.target_layers; the engine target must capture ws_x_ after
    // each of these layers into the [rows, n_capture, H] tap feed (capture order).
    std::vector<int32_t> target_layers;   // [n_capture], e.g. {1,16,31,46,61}

    // --- device weights (F16 [K,N] row-major, ready for gemm_fp16/gemv_fp16) ---
    sycl::half* w_fc = nullptr;          // [n_embd_cap, H]   dspark.fc (Q4_1 -> F16)
    float*      hidden_norm = nullptr;   // [H]               dspark.hidden_norm (F32)
    struct Layer {
        sycl::half *wq=nullptr, *wk=nullptr, *wv=nullptr, *wo=nullptr;
        sycl::half *w_gate=nullptr, *w_up=nullptr, *w_down=nullptr;
        float *attn_norm=nullptr, *ffn_norm=nullptr, *q_norm=nullptr, *k_norm=nullptr;
    };
    std::vector<Layer> layers;
    float*      output_norm = nullptr;   // [H]
    sycl::half* w_head = nullptr;         // [H, vocab]  drafter's OWN Q4_1 head -> F16
    void*       token_embd = nullptr;     // Q2_0 raw (shared layout with target)
    DType       token_embd_dtype = DType::kCount;
    sycl::half* d_snr = nullptr;          // [block_size, H] precomputed log-SNR rows

    // --- host-resident markov head (rank-256 bigram; applied host-side for M1) ---
    std::vector<float> markov_w1;         // [vocab*rank] (rank fastest) from head_a (BF16)
    std::vector<float> markov_w2;         // [vocab*rank] (rank fastest) from head_b (Q4_1)
    bool has_markov = false;

    // --- device scratch (sized once at load for max_ctx_rows + block_size) ---
    float*      d_feat_f32 = nullptr;     // [max_ctx_rows, n_embd_cap]
    sycl::half* d_feat_f16 = nullptr;     // [max_ctx_rows, n_embd_cap]
    sycl::half* d_target_ctx = nullptr;   // [max_ctx_rows, H]
    sycl::half* d_x = nullptr;            // [block_size, H]  draft residual
    sycl::half* d_h = nullptr;            // [block_size, H]  attn_norm(x)
    sycl::half* d_ain = nullptr;          // [max_rows, H]    concat(target_ctx, h)
    sycl::half* d_k = nullptr;            // [max_rows, N_kv]
    sycl::half* d_v = nullptr;            // [max_rows, N_kv]
    sycl::half* d_q = nullptr;            // [block_size, N_q]
    sycl::half* d_attn = nullptr;         // [block_size, N_q]
    sycl::half* d_tmp = nullptr;          // [block_size, H]  (Wo / down output)
    sycl::half* d_fg = nullptr;           // [block_size, F]
    sycl::half* d_fu = nullptr;           // [block_size, F]
    sycl::half* d_fh = nullptr;           // [block_size, F]
    sycl::half* d_hidden = nullptr;       // [block_size, H]  result_norm (pre-head)
    float*      C_small = nullptr;        // [roundup(max_rows,8), H] gemm f32 out
    float*      C_head = nullptr;         // [roundup(block_size,8), vocab] head f32 out
    int32_t*    d_pos_all = nullptr;      // [max_rows]
    int32_t*    d_pos_draft = nullptr;    // [block_size]
    int32_t*    d_tokens = nullptr;       // [block_size]

    std::vector<void*> owned;

    ~DsparkDrafter();

    // Load all weights + precompute log-SNR/markov from the drafter GGUF `g`
    // (must stay alive only for the duration of load). `max_ctx_rows_` is the
    // largest cumulative context-row count any draft_block() call will use.
    // Returns "" on success, else error text.
    std::string load(DeviceAllocator& a, const GgufReader& g, uint32_t max_ctx_rows_);

    // Run ONE self-contained drafter round.
    //   ctx_feat     : host [n_ctx_rows * n_embd_cap] f32 raw target taps
    //   ctx_pos      : host [n_ctx_rows] absolute positions
    //   draft_tokens : host [block_size] (row 0 = anchor id, rest = mask_token_id)
    //   draft_pos    : host [block_size] absolute positions
    // Outputs (resized as needed):
    //   out_hidden : [block_size * H]  pre-head hidden (result_norm) f32
    //   out_logits : [block_size * vocab] base logits (pre-markov) f32
    //   out_ids    : [block_size] post-markov argmax draft ids
    std::string draft_block(const float* ctx_feat, const int32_t* ctx_pos,
                            int32_t n_ctx_rows,
                            const int32_t* draft_tokens, const int32_t* draft_pos,
                            std::vector<float>& out_hidden,
                            std::vector<float>& out_logits,
                            std::vector<int32_t>& out_ids);
};

}  // namespace ie
