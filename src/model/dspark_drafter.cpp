// src/model/dspark_drafter.cpp — dspark drafter forward (M1).
//
// Verified against the PrismML fork's canonical forward
// (models/llama.cpp-prism/src/models/dspark.cpp) and the block-draft
// protocol + host markov resample (common/speculative.cpp:825-1447). See
// include/ie/dspark_drafter.hpp for the stage map and
// docs/q2_0_optimization/09_dspark_m1_port_plan.md for the port plan.
//
// M1 reuses existing engine leaf ops (include/ie/ops.hpp) and adds exactly one
// new device kernel: NON-CAUSAL attention (dspark_attn below). No existing
// kernels or models are modified.
//
// ===========================================================================
// M2 INTEGRATION HOOKS (NOT built here — list only, per plan §5):
//   1. Target 5-tap export: qwen35_dense.cpp forward() must memcpy ws_x_ rows
//      after layers {1,16,31,46,61} into [T,5,H] on prefill AND every verify.
//   2. Draft-provider hook in spec_generate (qwen35_dense.cpp:~1465): swap
//      mtp_.draft for DsparkDrafter; verify T=m+1 variable; next-round
//      conditioning = tap rows of the accepted prefix.
//   3. --spec-draft <dspark.gguf> CLI gate (mirror gemma4 --spec-head).
//   4. Markov SYCL kernel (host loop here is correct but bandwidth-bound;
//      ~127 MB w2 traffic per position — M2 needs a fused device resample).
//   5. Production LM head: reuse the target's Q2_0 head (0 extra VRAM) instead
//      of the drafter's own Q4_1 head used here for the gate (plan §5.3).
//   6. Confidence head: loaded-but-unused in the shipped fork (M3).
// ===========================================================================

#include "ie/dspark_drafter.hpp"

#include "ie/dequant_ref.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <cmath>
#include <cstring>
#include <limits>

namespace ie {

namespace {

inline uint32_t round_up8(uint32_t x) { return (x + 7u) & ~7u; }

inline float bf16_to_f32(uint16_t b) {
    uint32_t bits = uint32_t(b) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Non-causal GQA attention for the tiny dspark case: `T` draft queries attend
// over ALL `Lk` keys (ctx rows + draft rows), no positional mask. One work-item
// per (query, q-head); each runs a scalar online-softmax over the key window.
// GQA mapping (kv_head = h / (n_q/n_kv)) and scale (1/sqrt(HD)) match the
// engine's causal full_attention convention (src/ops/attention.cpp). HD<=128.
sycl::event dspark_attn(sycl::queue& q,
                        const sycl::half* Q, const sycl::half* K,
                        const sycl::half* V, sycl::half* Y,
                        uint32_t T, uint32_t Lk,
                        uint32_t n_q, uint32_t n_kv, uint32_t HD) {
    const float    scale = 1.0f / std::sqrt(float(HD));
    const uint32_t gqa   = n_q / n_kv;
    return q.parallel_for(sycl::range<2>(T, n_q), [=](sycl::id<2> idx) {
        const uint32_t t  = uint32_t(idx[0]);
        const uint32_t h  = uint32_t(idx[1]);
        const uint32_t kv = h / gqa;

        const sycl::half* qrow = Q + (size_t)(t * n_q + h) * HD;
        float qv[128];
        for (uint32_t d = 0; d < HD; ++d) qv[d] = float(qrow[d]);

        float m = -std::numeric_limits<float>::infinity();
        float l = 0.f;
        float out[128];
        for (uint32_t d = 0; d < HD; ++d) out[d] = 0.f;

        for (uint32_t i = 0; i < Lk; ++i) {
            const sycl::half* krow = K + (size_t)(i * n_kv + kv) * HD;
            float s = 0.f;
            for (uint32_t d = 0; d < HD; ++d) s += qv[d] * float(krow[d]);
            s *= scale;

            const float m_new = sycl::fmax(m, s);
            const float alpha = sycl::exp(m - m_new);
            const float e     = sycl::exp(s - m_new);

            const sycl::half* vrow = V + (size_t)(i * n_kv + kv) * HD;
            for (uint32_t d = 0; d < HD; ++d) out[d] = out[d] * alpha + e * float(vrow[d]);
            l = l * alpha + e;
            m = m_new;
        }

        const float inv = 1.f / l;
        sycl::half* yrow = Y + (size_t)(t * n_q + h) * HD;
        for (uint32_t d = 0; d < HD; ++d) yrow[d] = sycl::half(out[d] * inv);
    });
}

}  // namespace

DsparkDrafter::~DsparkDrafter() {
    if (alloc) for (void* p : owned) alloc->free(p);
}

std::string DsparkDrafter::load(DeviceAllocator& a, const GgufReader& g,
                                uint32_t max_ctx_rows_) {
    if (loaded) return {};
    alloc = &a;
    auto& q = a.queue();
    std::string err;

    // ---- geometry from GGUF metadata (arch-prefixed "dspark.*") ----
    auto ku = [&](const char* key, uint32_t& dst, bool required) -> bool {
        const auto* kv = g.find_kv(key);
        if (!kv) { if (required) err = std::string("missing key: ") + key; return false; }
        dst = uint32_t(kv->as_uint());
        return true;
    };
    auto kf = [&](const char* key, float& dst, bool required) -> bool {
        const auto* kv = g.find_kv(key);
        if (!kv) { if (required) err = std::string("missing key: ") + key; return false; }
        dst = float(kv->as_float());
        return true;
    };

    ku("dspark.block_count",            n_layer,   true);
    ku("dspark.embedding_length",       H,         true);
    ku("dspark.attention.head_count",   n_head,    true);
    ku("dspark.attention.head_count_kv",n_head_kv, true);
    ku("dspark.attention.key_length",   HD,        true);
    ku("dspark.feed_forward_length",    F,         true);
    kf("dspark.rope.freq_base",         rope_theta,true);
    kf("dspark.attention.layer_norm_rms_epsilon", eps, true);
    ku("dspark.dspark.block_size",      block_size,true);
    if (!err.empty()) return "dspark meta: " + err;

    { const auto* kv = g.find_kv("dspark.dspark.mask_token_id");
      if (!kv) return "dspark meta: missing dspark.dspark.mask_token_id";
      mask_token_id = int32_t(kv->as_int()); }
    { uint32_t mr = 0; ku("dspark.dspark.markov_rank", mr, false); markov_rank = mr; }
    { const auto* kv = g.find_kv("dspark.dspark.log_snr_conditioning");
      log_snr_on = kv && kv->as_int() != 0; }
    if (log_snr_on) {
        if (!kf("dspark.dspark.min_log_snr", min_log_snr, true) ||
            !kf("dspark.dspark.max_log_snr", max_log_snr, true))
            return "dspark meta: log_snr on but min/max missing";
    }

    N_q  = n_head    * HD;
    N_kv = n_head_kv * HD;

    // vocab + n_capture come from the tensor shapes (dspark has no vocab KV).
    const auto* te = g.find_tensor("token_embd.weight");
    if (!te) return "token_embd.weight not found";
    vocab = uint32_t(te->shape[1]);
    const auto* fc = g.find_tensor("dspark.fc.weight");
    if (!fc) return "dspark.fc.weight not found";
    n_embd_cap = uint32_t(fc->shape[0]);
    if (n_embd_cap % H != 0) return "dspark.fc: n_embd_cap not a multiple of H";
    n_capture = n_embd_cap / H;

    // Target capture layer ids (M2 tap export). n_capture is fixed by the fc width;
    // the ids tell the engine target WHICH layer outputs to feed as ctx taps.
    {
        const auto* kv = g.find_kv("dspark.dspark.target_layers");
        if (kv && kv->type == GgufValueType::kArray) {
            auto arr = kv->as_pod_array<int32_t>();
            target_layers.assign(arr.begin(), arr.end());
        }
        if (target_layers.size() != n_capture)
            return "dspark meta: target_layers count != n_capture (fc width)";
    }

    max_ctx_rows = max_ctx_rows_;
    const uint32_t max_rows = max_ctx_rows + block_size;

    // ---- weight dequant/upload helpers ----
    // Q4_1 [K,N] GGUF (N rows of K contiguous, blocks of 32 along K) -> device
    // F16 [K,N] row-major (element (k,n) at k*N+n), ready for gemm_fp16/gemv_fp16.
    auto q41_to_Bt = [&](const char* name) -> sycl::half* {
        const auto* t = g.find_tensor(name);
        if (!t) { err = std::string(name) + ": not found"; return nullptr; }
        if (t->dtype != DType::kQ4_1) { err = std::string(name) + ": expected Q4_1"; return nullptr; }
        const uint32_t K = uint32_t(t->shape[0]);
        const uint32_t N = uint32_t(t->shape[1]);
        if (K % 32 != 0) { err = std::string(name) + ": K not a multiple of 32"; return nullptr; }
        const uint32_t bpc = K / 32;                 // blocks per column
        std::vector<sycl::half> host((size_t)K * N);
        const auto* blocks = reinterpret_cast<const block_q4_1*>(t->data);
        float tmp[32];
        for (uint32_t n = 0; n < N; ++n) {
            const block_q4_1* col = blocks + (size_t)n * bpc;
            for (uint32_t b = 0; b < bpc; ++b) {
                ref::dequant_q4_1(&col[b], tmp);
                const uint32_t k0 = b * 32;
                for (uint32_t j = 0; j < 32; ++j)
                    host[(size_t)(k0 + j) * N + n] = sycl::half(tmp[j]);
            }
        }
        auto* d = static_cast<sycl::half*>(a.malloc((size_t)K * N * sizeof(sycl::half)));
        if (!d) { err = std::string(name) + ": device malloc"; return nullptr; }
        q.memcpy(d, host.data(), (size_t)K * N * sizeof(sycl::half)).wait();
        owned.push_back(d);
        return d;
    };
    auto f32_up = [&](const char* name, uint32_t expect_n) -> float* {
        const auto* t = g.find_tensor(name);
        if (!t) { err = std::string(name) + ": not found"; return nullptr; }
        if (t->dtype != DType::kF32) { err = std::string(name) + ": expected F32"; return nullptr; }
        if (uint32_t(t->shape[0]) != expect_n) { err = std::string(name) + ": bad shape"; return nullptr; }
        void* d = a.malloc(t->nbytes);
        if (!d) { err = std::string(name) + ": device malloc"; return nullptr; }
        q.memcpy(d, t->data, t->nbytes).wait();
        owned.push_back(d);
        return static_cast<float*>(d);
    };

    w_fc = q41_to_Bt("dspark.fc.weight");                 if (!err.empty()) return err;
    hidden_norm = f32_up("dspark.hidden_norm.weight", H); if (!err.empty()) return err;
    output_norm = f32_up("output_norm.weight", H);        if (!err.empty()) return err;
    w_head = q41_to_Bt("output.weight");                  if (!err.empty()) return err;

    layers.resize(n_layer);
    char buf[64];
    auto nm = [&](const char* t, uint32_t i) { std::snprintf(buf, sizeof(buf), "blk.%u.%s", i, t); return buf; };
    for (uint32_t i = 0; i < n_layer; ++i) {
        Layer& L = layers[i];
        L.attn_norm = f32_up(nm("attn_norm.weight", i), H);  if (!err.empty()) return err;
        L.ffn_norm  = f32_up(nm("ffn_norm.weight", i), H);   if (!err.empty()) return err;
        L.q_norm    = f32_up(nm("attn_q_norm.weight", i), HD);if (!err.empty()) return err;
        L.k_norm    = f32_up(nm("attn_k_norm.weight", i), HD);if (!err.empty()) return err;
        L.wq = q41_to_Bt(nm("attn_q.weight", i));            if (!err.empty()) return err;
        L.wk = q41_to_Bt(nm("attn_k.weight", i));            if (!err.empty()) return err;
        L.wv = q41_to_Bt(nm("attn_v.weight", i));            if (!err.empty()) return err;
        L.wo = q41_to_Bt(nm("attn_output.weight", i));       if (!err.empty()) return err;
        L.w_gate = q41_to_Bt(nm("ffn_gate.weight", i));      if (!err.empty()) return err;
        L.w_up   = q41_to_Bt(nm("ffn_up.weight", i));        if (!err.empty()) return err;
        L.w_down = q41_to_Bt(nm("ffn_down.weight", i));      if (!err.empty()) return err;
    }

    // ---- token_embd: keep raw Q2_0 on device for embedding_lookup_q2_0 ----
    token_embd_dtype = te->dtype;
    if (token_embd_dtype != DType::kQ2_0)
        return "token_embd: expected Q2_0 (Prism ternary), got " + std::string(type_name(token_embd_dtype));
    token_embd = a.malloc(te->nbytes);
    if (!token_embd) return "token_embd: device malloc";
    q.memcpy(token_embd, te->data, te->nbytes).wait();
    owned.push_back(token_embd);

    // ---- log-SNR conditioning: precompute the two distinct rows on host ----
    // (anchor t=1000, mask t=0) through the BF16 MLP, upload a [block_size,H]
    // matrix (row 0 = anchor, rows>0 = mask; anchor rule: draft_row % bsz == 0).
    {
        std::vector<sycl::half> snr_mat((size_t)block_size * H, sycl::half(0));
        if (log_snr_on) {
            const auto* w1t = g.find_tensor("dspark.log_snr_fc1.weight");
            const auto* b1t = g.find_tensor("dspark.log_snr_fc1.bias");
            const auto* w2t = g.find_tensor("dspark.log_snr_fc2.weight");
            const auto* b2t = g.find_tensor("dspark.log_snr_fc2.bias");
            if (!w1t || !b1t || !w2t || !b2t) return "dspark: log_snr tensors missing";
            const uint32_t n_freq = uint32_t(w1t->shape[0]);      // 128
            const uint32_t half   = n_freq / 2;                   // 64
            if (uint32_t(w1t->shape[1]) != H || uint32_t(w2t->shape[0]) != H ||
                uint32_t(w2t->shape[1]) != H) return "dspark: log_snr shape mismatch";
            const uint16_t* W1 = reinterpret_cast<const uint16_t*>(w1t->data);  // [H][n_freq]
            const uint16_t* W2 = reinterpret_cast<const uint16_t*>(w2t->data);  // [H][H]
            const float*    B1 = reinterpret_cast<const float*>(b1t->data);
            const float*    B2 = reinterpret_cast<const float*>(b2t->data);

            auto snr_row = [&](float log_snr, sycl::half* dst) {
                const float t = (log_snr - min_log_snr) / (max_log_snr - min_log_snr) * 1000.0f;
                std::vector<float> feat(n_freq);
                for (uint32_t i = 0; i < half; ++i) {
                    const float fr = std::exp(-std::log(10000.0f) * float(i) / float(half));
                    feat[i]        = std::sin(t * fr);
                    feat[half + i] = std::cos(t * fr);
                }
                std::vector<float> h1(H);
                for (uint32_t o = 0; o < H; ++o) {
                    float acc = B1[o];
                    const uint16_t* wr = W1 + (size_t)o * n_freq;
                    for (uint32_t k = 0; k < n_freq; ++k) acc += bf16_to_f32(wr[k]) * feat[k];
                    h1[o] = acc / (1.0f + std::exp(-acc));       // SiLU
                }
                for (uint32_t o = 0; o < H; ++o) {
                    float acc = B2[o];
                    const uint16_t* wr = W2 + (size_t)o * H;
                    for (uint32_t k = 0; k < H; ++k) acc += bf16_to_f32(wr[k]) * h1[k];
                    dst[o] = sycl::half(acc);
                }
            };
            std::vector<sycl::half> anchor(H), mask(H);
            snr_row(max_log_snr, anchor.data());
            snr_row(min_log_snr, mask.data());
            for (uint32_t r = 0; r < block_size; ++r) {
                const sycl::half* src = (r % block_size == 0) ? anchor.data() : mask.data();
                std::memcpy(snr_mat.data() + (size_t)r * H, src, (size_t)H * sizeof(sycl::half));
            }
        }
        d_snr = static_cast<sycl::half*>(a.malloc((size_t)block_size * H * sizeof(sycl::half)));
        if (!d_snr) return "dspark: d_snr malloc";
        q.memcpy(d_snr, snr_mat.data(), (size_t)block_size * H * sizeof(sycl::half)).wait();
        owned.push_back(d_snr);
    }

    // ---- markov head (host fp32, rank fastest) ----
    if (markov_rank > 0) {
        const auto* a_t = g.find_tensor("dspark.markov_head_a.weight"); // BF16 [rank, vocab]
        const auto* b_t = g.find_tensor("dspark.markov_head_b.weight"); // Q4_1 [rank, vocab]
        if (a_t && b_t) {
            const uint32_t R = uint32_t(a_t->shape[0]);
            if (R != markov_rank || uint32_t(a_t->shape[1]) != vocab)
                return "dspark: markov_head_a shape mismatch";
            markov_w1.resize((size_t)vocab * R);
            const uint16_t* w1raw = reinterpret_cast<const uint16_t*>(a_t->data);
            for (size_t i = 0; i < (size_t)vocab * R; ++i) markov_w1[i] = bf16_to_f32(w1raw[i]);

            if (b_t->dtype != DType::kQ4_1) return "dspark: markov_head_b expected Q4_1";
            if (R % 32 != 0) return "dspark: markov_rank not a multiple of 32 (Q4_1 head_b)";
            markov_w2.resize((size_t)vocab * R);
            ref::dequant_q4_1_buffer(b_t->data, (size_t)vocab * R, markov_w2.data());
            has_markov = true;
        }
    }

    // ---- device scratch ----
    auto AF16 = [&](size_t n) -> sycl::half* {
        auto* p = static_cast<sycl::half*>(a.malloc(n * sizeof(sycl::half)));
        if (p) owned.push_back(p);
        return p;
    };
    auto AF32 = [&](size_t n) -> float* {
        auto* p = static_cast<float*>(a.malloc(n * sizeof(float)));
        if (p) owned.push_back(p);
        return p;
    };
    auto AI32 = [&](size_t n) -> int32_t* {
        auto* p = static_cast<int32_t*>(a.malloc(n * sizeof(int32_t)));
        if (p) owned.push_back(p);
        return p;
    };
    d_feat_f32   = AF32((size_t)max_ctx_rows * n_embd_cap);
    d_feat_f16   = AF16((size_t)max_ctx_rows * n_embd_cap);
    d_target_ctx = AF16((size_t)max_ctx_rows * H);
    d_x          = AF16((size_t)block_size * H);
    d_h          = AF16((size_t)block_size * H);
    d_ain        = AF16((size_t)max_rows * H);
    d_k          = AF16((size_t)max_rows * N_kv);
    d_v          = AF16((size_t)max_rows * N_kv);
    d_q          = AF16((size_t)block_size * N_q);
    d_attn       = AF16((size_t)block_size * N_q);
    d_tmp        = AF16((size_t)block_size * H);
    d_fg         = AF16((size_t)block_size * F);
    d_fu         = AF16((size_t)block_size * F);
    d_fh         = AF16((size_t)block_size * F);
    d_hidden     = AF16((size_t)block_size * H);
    C_small      = AF32((size_t)round_up8(max_rows) * H);
    C_head       = AF32((size_t)round_up8(block_size) * vocab);
    d_pos_all    = AI32(max_rows);
    d_pos_draft  = AI32(block_size);
    d_tokens     = AI32(block_size);
    if (!d_feat_f32 || !d_feat_f16 || !d_target_ctx || !d_x || !d_h || !d_ain ||
        !d_k || !d_v || !d_q || !d_attn || !d_tmp || !d_fg || !d_fu || !d_fh ||
        !d_hidden || !C_small || !C_head || !d_pos_all || !d_pos_draft || !d_tokens)
        return "dspark: scratch alloc failed";

    loaded = true;
    return {};
}

std::string DsparkDrafter::draft_block(const float* ctx_feat, const int32_t* ctx_pos,
                                       int32_t n_ctx_rows_i,
                                       const int32_t* draft_tokens, const int32_t* draft_pos,
                                       std::vector<float>& out_hidden,
                                       std::vector<float>& out_logits,
                                       std::vector<int32_t>& out_ids) {
    if (!loaded) return "draft_block: not loaded";
    if (n_ctx_rows_i < 0) return "draft_block: negative n_ctx_rows";
    const uint32_t Lc = uint32_t(n_ctx_rows_i);
    if (Lc > max_ctx_rows) return "draft_block: n_ctx_rows exceeds max_ctx_rows";
    const uint32_t bs = block_size;
    const uint32_t Lk = Lc + bs;
    auto& q = alloc->queue();

    // ---- upload inputs ----
    if (Lc > 0)
        q.memcpy(d_feat_f32, ctx_feat, (size_t)Lc * n_embd_cap * sizeof(float));
    q.memcpy(d_tokens, draft_tokens, (size_t)bs * sizeof(int32_t));
    q.memcpy(d_pos_draft, draft_pos, (size_t)bs * sizeof(int32_t));
    {
        std::vector<int32_t> pos_all(Lk);
        for (uint32_t i = 0; i < Lc; ++i) pos_all[i] = ctx_pos[i];
        for (uint32_t j = 0; j < bs; ++j) pos_all[Lc + j] = draft_pos[j];
        q.memcpy(d_pos_all, pos_all.data(), (size_t)Lk * sizeof(int32_t)).wait();
    }

    // gemm helper: C_small = A[M,K] @ B[K,N] (fp32), then cast valid [M*N] -> dst.
    auto mm_to_f16 = [&](const sycl::half* A, const sycl::half* B, sycl::half* dst,
                         uint32_t M, uint32_t N, uint32_t K) {
        gemm_fp16(q, A, B, C_small, M, N, K);
        cast_fp32_to_fp16(q, C_small, dst, (size_t)M * N);
    };

    // ================= STAGE A: target-context projection (once) =============
    if (Lc > 0) {
        cast_fp32_to_fp16(q, d_feat_f32, d_feat_f16, (size_t)Lc * n_embd_cap);
        // fc: [Lc, n_embd_cap] @ [n_embd_cap, H] -> C_small[Lc, H]; cast into d_ain
        gemm_fp16(q, d_feat_f16, w_fc, C_small, Lc, H, n_embd_cap);
        cast_fp32_to_fp16(q, C_small, d_ain, (size_t)Lc * H);
        rms_norm_f32w(q, d_ain, hidden_norm, d_target_ctx, Lc, H, eps);
    }

    // ================= STAGE B: draft embeddings + log-SNR ====================
    embedding_lookup_q2_0(q, d_tokens, token_embd, d_x, bs, H);
    residual_add(q, d_x, d_snr, d_x, (size_t)bs * H);

    // ================= STAGE C: 6 layers ======================================
    for (uint32_t il = 0; il < n_layer; ++il) {
        const Layer& L = layers[il];

        // h = rms(attn_norm, x)
        rms_norm_f32w(q, d_x, L.attn_norm, d_h, bs, H, eps);

        // a_in = concat(target_ctx[Lc], h[bs])  (ctx rows never see attn_norm)
        if (Lc > 0)
            q.memcpy(d_ain, d_target_ctx, (size_t)Lc * H * sizeof(sycl::half));
        q.memcpy(d_ain + (size_t)Lc * H, d_h, (size_t)bs * H * sizeof(sycl::half));

        // K, V over all Lk rows; Q over draft rows only.
        mm_to_f16(d_ain, L.wk, d_k, Lk, N_kv, H);
        mm_to_f16(d_ain, L.wv, d_v, Lk, N_kv, H);
        mm_to_f16(d_h,   L.wq, d_q, bs, N_q,  H);

        // per-head QK-RMSNorm(HD)
        rms_norm_f32w(q, d_q, L.q_norm, d_q, bs * n_head,    HD, eps);
        rms_norm_f32w(q, d_k, L.k_norm, d_k, Lk * n_head_kv, HD, eps);

        // NEOX RoPE, full HD, base rope_theta; Q at draft pos, K at [ctx|draft].
        rope_partial(q, d_q, d_pos_draft, d_q, bs, n_head,    HD, HD, rope_theta);
        rope_partial(q, d_k, d_pos_all,   d_k, Lk, n_head_kv, HD, HD, rope_theta);

        // non-causal attention: bs queries over Lk keys
        dspark_attn(q, d_q, d_k, d_v, d_attn, bs, Lk, n_head, n_head_kv, HD);

        // out proj + residual
        mm_to_f16(d_attn, L.wo, d_tmp, bs, H, N_q);
        residual_add(q, d_x, d_tmp, d_x, (size_t)bs * H);

        // square SwiGLU FFN + residual
        rms_norm_f32w(q, d_x, L.ffn_norm, d_h, bs, H, eps);
        mm_to_f16(d_h, L.w_gate, d_fg, bs, F, H);
        mm_to_f16(d_h, L.w_up,   d_fu, bs, F, H);
        swiglu(q, d_fg, d_fu, d_fh, (size_t)bs * F);
        mm_to_f16(d_fh, L.w_down, d_tmp, bs, H, F);
        residual_add(q, d_x, d_tmp, d_x, (size_t)bs * H);
    }

    // ================= STAGE D: head =========================================
    rms_norm_f32w(q, d_x, output_norm, d_hidden, bs, H, eps);        // result_norm
    gemm_fp16(q, d_hidden, w_head, C_head, bs, vocab, H);            // base logits (f32)

    out_hidden.resize((size_t)bs * H);
    out_logits.resize((size_t)bs * vocab);
    {
        std::vector<sycl::half> hid((size_t)bs * H);
        q.memcpy(hid.data(), d_hidden, (size_t)bs * H * sizeof(sycl::half));
        q.memcpy(out_logits.data(), C_head, (size_t)bs * vocab * sizeof(float)).wait();
        for (size_t i = 0; i < (size_t)bs * H; ++i) out_hidden[i] = float(hid[i]);
    }

    // ================= STAGE E: host markov resample =========================
    // step_logit[k][v] = base[k][v] + sum_r w1[prev*R+r]*w2[v*R+r]; argmax
    // (lowest v wins ties); prev = anchor for k==0 else the sampled k-1 token.
    out_ids.assign(bs, 0);
    const int32_t anchor = draft_tokens[0];
    int32_t prev = anchor;
    const uint32_t R = markov_rank;
    for (uint32_t k = 0; k < bs; ++k) {
        const float* base = out_logits.data() + (size_t)k * vocab;
        int32_t best_id = 0;
        float   best_v  = -std::numeric_limits<float>::infinity();
        if (has_markov) {
            const float* emb = markov_w1.data() + (size_t)prev * R;
            for (uint32_t v = 0; v < vocab; ++v) {
                const float* w2row = markov_w2.data() + (size_t)v * R;
                float bias = 0.f;
                for (uint32_t r = 0; r < R; ++r) bias += emb[r] * w2row[r];
                const float logit = base[v] + bias;
                if (logit > best_v) { best_v = logit; best_id = int32_t(v); }
            }
        } else {
            for (uint32_t v = 0; v < vocab; ++v)
                if (base[v] > best_v) { best_v = base[v]; best_id = int32_t(v); }
        }
        out_ids[k] = best_id;
        prev = best_id;
    }
    return {};
}

}  // namespace ie
