// include/ie/mimo26_dflash.hpp — MiMo-V2.6's bundled DFlash drafter (P5, docs/mimo26/00_PORT_PLAN.md "P5 DESIGN").
//
// A block drafter: after each verify, the target's residual stream after layers [0, 11, 23, 35, 47] at the accepted rows
// (and at the prompt's last rows after a prefill) -> concat [5 * 4096] -> fc -> hidden_norm = a context feature per
// position; every drafter layer k/v-projects the features (k_norm, RoPE on the first 64 of 128 dims, theta 1e4) into its
// own KV ring (the last 1024 positions). A draft is ONE forward of the block [anchor token, 7 x mask] (the target's
// embeddings; the mask row from dflash/mask_embedding.pt -- the target's own row for the mask id is all zeros) through the
// 5 dense layers -- q from the block, keys/values = the ring + the block, bidirectional inside the block, per-head sinks,
// the attention output x attention_value_scale (0.612) -- then the final norm and the TARGET's lm_head -> 7 draft tokens.
// The offline study (tools/mimo26/dflash_accept.py) measured 0.9 / 2.4 / 4.5 accepted drafts at k = 1 / 3 / 7 on held-out
// Dream turns; a drafter error only lowers acceptance (the verify decides every token).
#pragma once

#include "ie/safetensors.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace ie {

struct Mimo26DFlashConfig {
    uint32_t hidden = 0, inter = 0, n_layers = 0, n_q = 0, n_kv = 0, head_dim = 0, rope_dim = 0;
    uint32_t block = 0, window = 0, mask_id = 0;
    float    rope_theta = 0.f, eps = 0.f, value_scale = 1.f;
    std::vector<uint32_t> target_layers;   // the target layers whose residual rows are the context features
};

class Mimo26DFlash {
public:
    ~Mimo26DFlash();
    // Reads <model_dir>/dflash/config.json, maps dflash_draft_model.safetensors, reads mask_embedding.pt's raw storage.
    std::string load(const std::string& model_dir);
    // Uploads the weights (fp16) to `q`'s device and allocates the ring (window + max_ctx_rows slots) and workspaces for
    // context updates of up to max_ctx_rows rows. `embed` is the target's BF16 embedding table (host, row-major
    // [vocab, hidden]). Call BEFORE the target forward's init, so an auto-sized static expert tier sees this VRAM taken.
    std::string init(sycl::queue& q, const uint8_t* embed_bf16, uint32_t vocab, uint32_t max_ctx_rows);
    // The target's fp16 lm_head on the same device, [vocab, hidden] (Mimo26Forward::head_weights(), after its init).
    void set_head(const sycl::half* lm_head) { head_ = lm_head; }
    const Mimo26DFlashConfig& config() const { return cfg_; }
    // Context features for positions [pos0, pos0 + T): host fp32, LAYER-major [n_target_layers][stride][hidden] (one block
    // of `stride` >= T rows per target layer, as the forward exports them -- a verify exports all its rows and adds only the
    // accepted first T; 0 = T). pos0 < ctx_end() replaces the rows at and after pos0 (a rewind); pos0 > ctx_end() leaves a
    // gap: the context restarts at pos0 (drafts attend only to positions the ring holds, so a gap costs acceptance, never
    // correctness).
    std::string add_context(const float* feats, uint32_t T, uint32_t pos0, uint32_t stride = 0);
    uint32_t ctx_end() const { return ctx_end_; }
    // Drop the positions at and after L (the target's prefix reuse); the ring's slots older than hi - R are gone.
    void     rewind(uint32_t L);
    void     reset() { ctx_end_ = ctx_lo_ = hi_ = 0; }
    // The drafts after `anchor` (the token at position p = ctx_end(), not yet in the target's caches): up to block - 1
    // greedy tokens, `k` of them returned -- cut before the first draft whose softmax probability under the drafter is
    // below `min_p` (0 keeps all k: a verify row costs ~25 ms, so a likely rejection is cheaper not drafted).
    std::string draft(int32_t anchor, uint32_t k, std::vector<int32_t>& out, float min_p = 0.f);
    const std::vector<float>& last_probs() const { return probs_; }   // the last draft's top-1 probabilities, all k rows
    // P5 gate (d): dflash.py's forward as shipped -- no attention sinks, value scale 1 -- for the reference comparison
    void set_reference_form(bool on) { ref_form_ = on; }
    uint64_t vram_bytes() const { return bytes_; }
    void free_all();

private:
    struct Layer {
        sycl::half *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr, *gate = nullptr, *up = nullptr, *down = nullptr;
        float *in_norm = nullptr, *post_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *sink = nullptr;
        sycl::half *kc = nullptr, *vc = nullptr;   // the context ring [n_kv, R, head_dim]
    };
    template <class T> T* dev(size_t n);
    SafetensorsReader file_;
    Mimo26DFlashConfig cfg_;
    std::vector<float> mask_emb_;          // [hidden]
    sycl::queue* q_ = nullptr;
    const uint8_t* embed_ = nullptr;
    const sycl::half* head_ = nullptr;
    uint32_t vocab_ = 0, R_ = 0, max_rows_ = 0;
    uint32_t ctx_end_ = 0, ctx_lo_ = 0, hi_ = 0;   // the context = positions [ctx_lo_, ctx_end_); hi_ = the highest written + 1
    std::vector<Layer> L_;
    sycl::half *fc_ = nullptr; float *hnorm_ = nullptr, *fnorm_ = nullptr;
    // workspaces (max_rows_ rows for context updates, block rows for a draft)
    float *feat32_ = nullptr, *ctx32_ = nullptr, *x32_ = nullptr, *t32_ = nullptr, *q32_ = nullptr, *kv32_ = nullptr, *g32_ = nullptr, *u32_ = nullptr, *lg32_ = nullptr;
    sycl::half *feat16_ = nullptr, *ctx16_ = nullptr, *x16_ = nullptr, *qh_ = nullptr, *kh_ = nullptr, *vh_ = nullptr, *att_ = nullptr, *h16_ = nullptr;
    int32_t* pos_ = nullptr;
    int32_t* top_id_ = nullptr; float* top_p_ = nullptr;   // [block]: the head rows' argmax and its softmax probability
    std::vector<float> probs_;
    bool ref_form_ = false;
    std::vector<void*> owned_;
    uint64_t bytes_ = 0;
};

}  // namespace ie
