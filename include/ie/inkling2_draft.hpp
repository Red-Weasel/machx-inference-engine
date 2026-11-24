// include/ie/inkling2_draft.hpp — Inkling 2 Mini's draft head (mtp.safetensors, "DSpark"): 3 dense layers over the
// target's mixed features + a recurrent (rnn_hidden) Markov head; drafts block_size - 1 = 7 tokens per pass.
//
// Forward (vLLM's vllm/models/inkling2/nvidia/dspark.py, checked by scripts/inkling2/dspark_harness.py):
//   context: per position, the target features (Inkling2RunOptions::feature_visits = mtp_config.target_layer_ids:
//     the partial state entering calls 5, 23, 41 and the final-norm output); layer i mixes them with softmax(fuse_weight[i])
//     + RMSNorm * dflare_norm, then K = rope(k_norm_ctx(k_proj_ctx f)), V = v_proj_ctx f at the position. Local layers
//     (0, 2) keep a ring of the last sliding_window_size positions; layer 1 keeps every position.
//   block at the anchor position p: row 0 = embed_norm(embed[anchor]), rows 1..7 = mask_embed; pre-norm layers with
//     q / k head norms and neox RoPE (theta 1e6), attention over [context, block] without a causal mask, SiLU MLP;
//     final norm. Row k drafts the token at p + k: logits = softcap(unembed h_k + fold(markov latent)), the Markov state
//     carried over the 7 steps from the anchor (z = [state, w1[prev], h_k, f_rnn], f_rnn = the Markov mix at p - 1).
// fp32 activations, bf16 weights, fp16 context K / V. The target's unembed is shared (not copied).
#pragma once
#include <sycl/sycl.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ie {

class Inkling2Draft {
public:
    Inkling2Draft();
    ~Inkling2Draft();
    // dir: the checkpoint (config.json mtp_config, mtp.safetensors, the target's embed / embed_norm). unembed: the
    // target's device bf16 [vocab, hidden] rows (vocab = the unpadded vocabulary it reads out). max_ctx: positions.
    std::string init(sycl::queue& q, const std::string& dir, const uint16_t* unembed, uint32_t vocab, uint32_t max_ctx);
    bool ready() const;
    uint32_t block() const;                                  // 8: the anchor + 7 drafts
    const std::vector<uint32_t>& feature_visits() const;     // for Inkling2RunOptions::feature_visits
    uint64_t device_bytes() const;
    // Append the context of positions [n_ctx(), n_ctx() + rows): rows [first, first + rows) of features [taps][T][hidden]
    // (Inkling2RunOptions::features of a forward with T rows).
    std::string add_context(const std::vector<float>& features, uint32_t T, uint32_t first, uint32_t rows);
    uint32_t n_ctx() const;
    // Rewind the context to n positions (at most 64 back from the furthest written; further: reset needed).
    std::string truncate(uint32_t n);
    void reset();
    // Greedy drafts (block() - 1 tokens) after `anchor`, the token at position n_ctx().
    std::string draft(int32_t anchor, std::vector<int32_t>& out);
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace ie
