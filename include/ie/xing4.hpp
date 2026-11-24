// include/ie/xing4.hpp -- Xing4.0-29B-A4B (GGUF arch `xing4_0`): DeepSeek-V3 MLA + MoE with DeepSeek-V4 style
// hyper-connections (4 residual streams, Sinkhorn mixing), served from a Q8_0 GGUF with every weight on the cards.
//
// Per layer (reference: the release's modeling_xing4_0.py; llama.cpp src/models/xing4-0.cpp):
//   hc_pre (streams -> pre/post/comb, collapsed = sum_i pre_i S_i) -> RMSNorm -> MLA attention -> S' = post (x) out + comb S
//   hc_pre (ffn)                                                     -> RMSNorm -> dense SwiGLU (layers < 2) or MoE
//   MoE: 64 routed experts, top-4 by sigmoid(logit) + bias, weights = normalised sigmoids x 2.0, + 1 shared expert.
// After the last layer: mean of the 4 streams -> RMSNorm -> lm_head.
//
// Placement: layers [0, split) on card 0, [split, L) and the head on card 1 (one card: everything on card 0). Weights are
// Q8_0 blocks from the file repacked to SoA (qs[n*K + k] int8, d[n*(K/32) + b] fp16): the decode GEMVs read 1.0625 bytes a
// weight. KV cache per layer: [ctx, 576] fp16 (512 normalised latent + 64 roped key), the absorbed MLA form.
#pragma once

#include "ie/gguf.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ie {

struct Xing4Config {
    uint32_t n_layers = 40, dim = 3584, n_heads = 32, q_lora = 768, kv_lora = 512, nope = 128, rope = 64, v_dim = 128;
    uint32_t ffn = 9216, moe_ffn = 1024, n_experts = 64, top_k = 4, n_shared = 1, n_dense = 2, vocab = 131072, hc = 4;
    uint32_t sinkhorn_iters = 20;
    float    norm_eps = 1e-6f, hc_eps = 1e-6f, routed_scale = 2.0f, clamp_lo = -30.f, clamp_hi = 30.f;
    float    rope_theta = 10000.f, yarn_factor = 64.f, yarn_beta_fast = 32.f, yarn_beta_slow = 1.f, yarn_mscale_all = 1.f;
    uint32_t yarn_orig_ctx = 4096;
    bool     norm_topk = true;
    uint32_t qk_dim() const { return nope + rope; }
    uint32_t lat_w() const { return kv_lora + rope; }          // 576: one cache row
};

std::string xing4_read_config(const GgufReader& g, Xing4Config& c);

// One Q8_0 matrix in SoA form on a card: N output rows of K inputs (several stacked for 3-D tensors).
// fmt as Xg4W (include/ie/xing4_ops.hpp): 0 Q8_0, 1 Q6_K, 2 Q5_K, 3 Q4_K -- the file's own format where the kernels read it,
// otherwise rounded to Q8_0 at load.
struct Xg4Mat {
    int8_t* qs = nullptr; uint16_t* d = nullptr; uint8_t* qh = nullptr; uint8_t* sc = nullptr; uint32_t N = 0, K = 0, fmt = 0;
    explicit operator bool() const { return qs != nullptr; }
};

struct Xing4Options {
    uint32_t max_ctx = 262144;      // keys a lane at most
    uint32_t kv_tokens = 262144;    // the paged cache's keys, all lanes together (64-key pages)
    uint32_t max_lanes = 16;        // requests at once
    uint32_t max_tokens = 512;      // rows of one forward call (a prompt chunk, or one row per lane)
    uint32_t split = 0;             // first layer on card 1 (0 = half the layers)
};

// One row of a forward call: the token and the lane (request) it belongs to. A lane's rows go in order; each takes the
// lane's next position.
struct Xing4Row { int32_t token; uint32_t lane; };

class Xing4Model {
public:
    Xing4Model();
    ~Xing4Model();
    Xing4Model(const Xing4Model&) = delete;
    Xing4Model& operator=(const Xing4Model&) = delete;

    std::string load(const std::string& gguf_path, const std::vector<sycl::queue*>& qs, const Xing4Options& o);
    const Xing4Config& config() const { return cfg_; }
    const GgufReader& gguf() const { return g_; }

    // A batch of rows from any lanes. logits: [n_logit_rows, vocab] fp32 on the host for the rows listed in logit_rows
    // (ascending indices into rows). The lanes keep the rows in their caches.
    std::string forward_rows(const Xing4Row* rows, uint32_t T, float* logits, const uint32_t* logit_rows, uint32_t n_logit_rows);
    // lane 0 only: ids[0 .. T) after what lane 0 holds; logits for the last n_logits rows
    std::string forward(const int32_t* ids, uint32_t T, float* logits, uint32_t n_logits);
    void lane_truncate(uint32_t lane, uint32_t n);              // forget the lane's keys past n (0 = free the lane)
    // dst takes src's first n keys (n <= lane_len(src)): the full pages are SHARED (no copy, no memory -- a page is freed when
    // its last lane lets go), the partial last page is copied. dst's old keys are dropped. One forward at a time (the caller).
    std::string lane_share(uint32_t dst, uint32_t src, uint32_t n);
    uint32_t shared_pages() const { uint32_t k = 0; for (uint16_t r : page_ref_) k += r > 1; return k; }
    uint32_t lane_len(uint32_t lane) const { return lane < lane_len_.size() ? lane_len_[lane] : 0; }
    uint32_t free_keys() const { return uint32_t(free_pages_.size()) * 64u; }
    void rewind(uint32_t n) { lane_truncate(0, n); }
    uint32_t position() const { return lane_len(0); }
    uint64_t vram_bytes(size_t card) const;

    // Per-kernel timing (IE_XG4_PROFILE=1 + profiling queues) is printed by the tool; this is the per-card time of the
    // last forward's work, in ms, measured with events at the card boundaries.
    double last_card_ms(size_t card) const { return card < card_ms_.size() ? card_ms_[card] : 0.0; }

    std::string profile_report();     // IE_XG4_PROFILE=1 with profiling queues: kernel time by kind since the last report
    void profile_reset();

    struct Layer;
    struct Card;
private:
    void prof(const char* name, const sycl::event& e);
    bool prof_on_ = false;
    bool intdot_ = true;
    bool p2p_ = false;                                          // the cards share a context: the streams are pushed card to card                                        // IE_XG4_INTDOT=0: fp16 activations in the matrix kernels
    std::vector<std::pair<const char*, sycl::event>> prof_ev_;
    std::string upload_layer(Card& c, uint32_t L);
    std::string run_layer(Card& c, uint32_t L, uint32_t T, uint32_t max_keys);
    Xg4Mat q8_mat(Card& c, const GgufTensorInfo* t, uint32_t n_rows, uint32_t k, std::string& err);
    void q8_fill(Card& c, const GgufTensorInfo* t, Xg4Mat& dst, uint32_t row0, uint32_t n_rows, std::string& err);
    bool alloc_mat(Card& c, Xg4Mat& m, uint32_t fmt, uint32_t n_rows, uint32_t k);
    float* f32_vec(Card& c, const GgufTensorInfo* t, size_t n, std::string& err);

    Xing4Config cfg_;
    Xing4Options opt_;
    GgufReader g_;
    std::vector<std::unique_ptr<Card>> cards_;
    std::vector<std::unique_ptr<Layer>> layers_;
    std::vector<uint32_t> layer_card_;
    std::vector<float> rope_inv_freq_;
    float attn_scale_ = 1.f;
    uint32_t maxp_ = 0, n_pages_ = 0;
    std::vector<uint32_t> lane_len_;
    std::vector<std::vector<int32_t>> lane_pages_;
    std::vector<int32_t> free_pages_, tab_host_;
    std::vector<uint16_t> page_ref_;                            // lanes holding each page (a shared prefix: several)
    int32_t page_take();                                        // a free page, held once (-1: none)
    void page_drop(int32_t p);                                  // one lane lets go; the last returns it to the free list
    void page_copy(int32_t to, int32_t from, uint32_t rows);    // rows [0, rows) of every layer's page
    bool tab_dirty_ = true;
    std::vector<double> card_ms_;
    std::vector<float> host_x_;                                 // the streams on their way between the cards
    std::vector<int32_t> rinfo_;                                // [lane T][pos T][token T][logit rows][attention groups]
    uint32_t agrp_off_ = 0, n_agrp_ = 0;                        // the prompt attention's row groups (xg4_attn_rows4) in rinfo_
};

}  // namespace ie
