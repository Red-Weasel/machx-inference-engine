// include/ie/deepseek41_dspark.hpp — DeepSeek-V4.1-Flash's DSpark drafter (docs/deepseek41/48 §A, 54 §6, 57).
//
// The reference's `forward_spec` (inference/model.py:1275-1282) on the engine's kernels: from the backbone's
// `main_hidden` (the mean over the hc copies of the stream ENTERING layers 37, 38, 39, concatenated) and the
// last sampled token, five draft rows -- the token, then four noise tokens -- run three DSpark blocks (window-only
// attention whose ring holds the BACKBONE's main_kv and whose five rows see every filled slot and each other;
// a 128-expert / top-3 MoE with the shared expert; the hyper-connections), then the tied head, the sequential
// Markov bias + argmax chain and the confidence head. Card-local: it lives on the card that owns the backbone's
// last layers and the head.
#pragma once
#include "ie/deepseek41_experts.hpp"
#include "ie/deepseek41_weights.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace ie {

class Ds41Drafter {
public:
    // `cache` is the card's dense cache (the three stages upload through it like backbone layers); the tier's
    // residency is `n_static` / `n_pinned` slots per stage (0 = from the card's free VRAM after `vram_reserve`
    // and the given pinned bytes). `ranking` may be empty (index order).
    struct Options { uint32_t n_static = 0, n_pinned = 0, stream_slots = 4; uint64_t vram_reserve = 2ull << 30; uint64_t host_pin_bytes = 8ull << 30; };
    std::string init(sycl::queue& q, const DeepSeek41Model& m, Ds41DenseCache& cache,
                     const std::vector<std::vector<uint32_t>>& ranking, const Options& opt);
    void free(sycl::queue& q);
    bool ready() const { return ready_; }

    // The rings: each stage's main_kv for positions [pos0, pos0 + n) from main_hidden [n, dim * n_targets] (host
    // fp32) -- the prefill's seeding (the last min(n, window) rows land) and, in a loop, every accepted row.
    std::string seed(sycl::queue& q, const float* main_hidden, uint32_t n, uint32_t pos0);

    // One pass at position `pos` (the rings hold positions <= pos): the five rows [token, noise x 4] at positions
    // pos + 1 .. pos + 5. `ids` gets [token, d1 .. d5], `logits` the five rows [5, vocab] after the Markov bias
    // (host), `confidence` the five scores. `forced_idx` / `forced_w` (per stage, [5 * top_k]) replace the router
    // (the golden's routing, as the backbone tests do); `probe(name, stage, dev, n, q)` sees the stage boundaries:
    // "main_x", "attn_in", "attn_out", "ffn_in", "moe_out", "layer_out", "ffn_pre", "head_in", "logits_base".
    using Probe = std::function<void(const char* name, uint32_t stage, const float* dev, size_t n, sycl::queue& q)>;
    std::string draft(sycl::queue& q, const float* main_hidden, int32_t token, uint32_t pos,
                      std::vector<int32_t>& ids, std::vector<float>& logits, std::vector<float>& confidence,
                      const Probe& probe = {}, const std::vector<const int32_t*>* forced_idx = nullptr,
                      const std::vector<const float*>* forced_w = nullptr);
    // the routing the last pass's own router chose, per stage [5 * top_k] (for the flip count against the golden)
    const std::vector<std::vector<int32_t>>& last_routing() const { return last_idx_; }
    // the last seed's main_x rows [n, dim], the last pass's Markov embedding rows [5, R] and bias rows [5, V] (host; docs/57 criteria 1 and 4)
    const std::vector<float>& last_main_x() const { return last_main_x_; }
    const std::vector<float>& last_markov_embed() const { return last_embed_; }
    const std::vector<float>& last_markov_bias() const { return last_bias_; }
    // diagnostics (the negative controls of docs/57 criterion 6)
    void set_rope_offset_diagnostic(int32_t off) { rope_off_ = off; }
    void set_causal_drafts_diagnostic(bool on) { causal_diag_ = on; }
    uint64_t dense_bytes() const { return dense_bytes_; }
    const Ds41ExpertTier& tier() const { return tier_; }
    uint32_t block_size() const { return B_; }
    const float* ring(uint32_t s) const { return rings_.at(s); }                    // stage s's ring [window, HD] (device)
    double last_pass_ms() const { return last_ms_; }

private:
    const DeepSeek41Model* m_ = nullptr; Ds41DenseCache* cache_ = nullptr;
    bool ready_ = false; uint32_t L0_ = 0, nS_ = 0, B_ = 0, TK_ = 0, E_ = 0, NT_ = 0;   // the first stage's layer id, stages, block, top-k, experts, targets
    Ds41ExpertTier tier_;
    Ds41Fp8Mat main_proj_; float* main_norm_ = nullptr; float* norm_ = nullptr;      // device
    sycl::half* markov_head_ = nullptr;                                             // device [V, R] fp16
    std::vector<uint16_t> markov_embed_bf16_;                                       // host [V, R] bf16 (a gather per step)
    std::vector<float> conf_proj_;                                                  // host [dim + R] fp32
    std::vector<float*> rings_;                                                     // per stage [window, HD]
    float* rope_inv_ = nullptr; int32_t* rope_pos_ = nullptr; float* rope_cs_ = nullptr; float* rope_sn_ = nullptr;   // [1 + B] positions
    std::vector<void*> scratch_;                                                    // everything to free
    struct Bufs { float *hA, *hB, *pre_mix, *xa, *xn, *qr, *qrn, *qq, *kvw, *oo, *oa, *ao, *xf, *xfn, *moe, *shg, *shu, *shh, *mask, *r_logits, *r_w, *a_pre, *a_post, *a_comb, *f_pre, *f_post, *f_comb, *hcs, *lg, *bias, *xhead; int32_t* r_i; sycl::half *x16, *t16, *e16; };
    Bufs bb_{}; bool bufs_ready_ = false;                                           // the pass's buffers, allocated on the first pass
    std::vector<std::vector<int32_t>> last_idx_;
    std::vector<float> last_main_x_, last_embed_, last_bias_;
    uint64_t dense_bytes_ = 0; double last_ms_ = 0; int32_t rope_off_ = 0; bool causal_diag_ = false;
    void* take(sycl::queue& q, size_t bytes);
};

}  // namespace ie
