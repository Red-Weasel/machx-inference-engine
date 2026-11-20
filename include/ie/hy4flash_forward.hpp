// include/ie/hy4flash_forward.hpp — Hy4-flash device forward, bring-up form (docs/hy4flash/00_PORT_AND_ABLITERATION_PLAN.md
// phase 2): correctness against the reference first, the speed work later.
//
//   residual     4 streams [T, hc, dim] fp32. A sublayer reads x = sum_i pre_i * stream_i and writes stream_i += post_i * y.
//   attention    gated MLA in the ABSORBED form: the cache holds one latent per position ([kv_lora | rotated rope] = 576
//                fp16), a query head scores it with (W_k^T q_nope | q_rot), the head's value is W_v . sum_k p_k latent_k.
//                Sliding layers see the last `window` keys; indexed layers every key up to the recall budget, then the
//                groups the compressed indexer recalls (scored on the host in this form).
//   FFN          layer 0 dense; the others the routed experts in the latent width through Ds41ExpertTier (MXFP4 pack),
//                between routed_expert_down_proj / up_proj, plus the shared expert, closed by combine_output_norm.
//   n-gram       the hashed table rows are read from the source mapping on the host (fp8 round trip, per-head norm).
//
// One lane, no pipeline; layers split over the cards in order. Queues must be in-order.
#pragma once
#include "ie/deepseek41_experts.hpp"
#include "ie/hy4flash.hpp"
#include "ie/hy4flash_q8.hpp"

#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace ie {

struct Hy4fOptions {
    uint32_t max_ctx = 4096;        // positions the caches hold
    uint32_t max_tokens = 256;      // rows per forward() call
    // a prompt chunk of up to this many rows (> max_tokens) runs in VRAM the expert slots lend for it (8-bit experts only;
    // 0 = none): a long prompt streams each missed expert once per chunk, so fewer, bigger chunks read it faster
    uint32_t big_tokens = 0;
    uint32_t split_layer = 0;       // two cards: the first layer of the second card (0 = half the layers)
    uint32_t n_static = 0;          // experts a layer kept in VRAM (0 = all of them when they fit the tier's own check)
    uint32_t n_pinned = 0;          // experts a layer pinned in RAM (0 = the rest)
    uint32_t stream_slots = 8;
    std::vector<std::vector<uint32_t>> ranking;   // [n_layers][E] most-important-first; empty = identity
    // The routed experts computed from the SOURCE's BF16 weights (uploaded per expert, per call) instead of the 4-bit pack:
    // the yardstick the pack's quality is measured against. Slow -- for chunked evaluation, not for decode. Needs no pack.
    bool exact_experts = false;
    // The routed experts as 8-bit integers (Hy4fQ8Store: built at load from the source, no pack; W8A16 kernels). Inside the
    // model's own numerical noise against the reference, unlike the MXFP4 pack. q8_slots = VRAM slots a layer.
    bool q8_experts = false;
    // 0 = as many as each card's free VRAM holds after everything else is allocated, less a 1.4 GiB margin (the
    // default). A number asks for that many; a card that cannot hold it gets what fits and a warning (2026-11-18: 188
    // on a card with room for 175 overfilled it and decode crawled).
    uint32_t q8_slots = 0;
    std::vector<uint32_t> q8_slots_card;          // per card, when the cards' free VRAM differs (empty = q8_slots on every card)
    // The stored width of the experts' integers: gate / up, and down (8, 5 or 4; Hy4fQFmt). 8 / 8 = the 8-bit pack.
    uint32_t q_bits_gu = 8, q_bits_dn = 8;
    // The big non-expert matrices (attention's gate / o / q_b, the shared expert, the latent maps, layer 0's FFN) as 8-bit
    // integers with one fp16 scale per 32 weights, converted at load: half the bytes a decode token reads outside the
    // experts (12.3 GB a token as fp16 -- more than twice the experts') and 5 GiB of VRAM back for expert slots.
    // Decode and T <= 4 run W8A16 kernels; longer chunks expand a matrix to fp16 scratch and use the fp16 GEMM.
    bool q8_dense = false;
    // The checkpoint's drafter (DFly2; model.dfly_layers.*): its 0.85 B weights go on the LAST card before the expert slots
    // are sized, so a load that will not draft must not ask for it (about 1 GiB: some 7 expert slots a layer there).
    bool drafter = false;
    // With exact_experts: every expert matrix is rounded to a weight FORMAT on the card before it is used, so a format's
    // quality can be measured (NLL against the unrounded run) without building its pack or its kernels. Rows are cut into
    // blocks along the input dimension. "" = none;
    //   mxfp4  4-bit float (E2M1), one power-of-two scale per 32            (the pack's rule: 4.25 bits a weight)
    //   mxfp6  6-bit float (E2M3), one power-of-two scale per 32            (6.25)
    //   fp8    8-bit float (E4M3), one power-of-two scale per 32            (8.25)
    //   int8   8-bit integer, one fp scale per 32                           (Q8_0's shape, 8.5)
    //   q6s    6-bit integer -32..31, one fp scale per 16                   (Q6_K's shape, 6.56 as Q6_K stores it)
    //   q5a    5-bit integer with scale and offset per 32                   (Q5_K's shape, 5.5)
    //   q4a    4-bit integer with scale and offset per 16                   (Q4_K's shape, 4.5)
    // The K-quant shapes are min/max fits with unquantised scales: llama.cpp's search does better on the fit and a little
    // worse on the 6-bit scales, so read them as approximations.
    //   q3a / q2a  3- / 2-bit integer with scale and offset per 16      (about 3.5 / 2.5 with 6-bit scales)
    //   t158   ternary, one scale per 32 (1.58 bits + scale);  b1  sign, one scale per 32 (1 bit + scale)
    std::string emulate;
    // Bits by usage: experts flagged in emulate_hot_set[MoE layer ordinal][expert] take emulate_hot ("exact" = unrounded)
    // instead of `emulate`. Empty = every expert takes `emulate`.
    std::string emulate_hot;
    std::vector<std::vector<uint8_t>> emulate_hot_set;
    // Diagnostic (the real-weights oracle of the first K layers): stop after this many layers (0 = run all). forward()
    // then returns no logits; the stream dump gets one more entry, the streams LEAVING the last layer run.
    uint32_t stop_after = 0;
};

// ---- host-side arithmetic (src/model/hy4flash_host.cpp, compiled with the precise float model) ----------------------
// cos / sin of one position for a layer type: n = rope_dim / 2 frequencies, theta^(-2i / rope_dim). pos3 = (t, h, w): the
// interleaved M-RoPE takes frequency i from plane i % 3 while i < 3 * mrope[1 or 2], else from the time plane.
void hy4f_rope_row(float theta, const uint32_t mrope[3], uint32_t rope_dim, const int32_t pos3[3], float* cos_out, float* sin_out);
// x -> nearest FP8 E4M3 value -> x, one scale per vector (the reference's _fp8_round_trip), in place.
void hy4f_fp8_round_trip(float* x, uint32_t n);
// The same rounding kept as data: `row` = [fp32 scale | n E4M3 codes (sign, 4-bit exponent, 3-bit mantissa)].
// hy4f_fp8_decode_row(encode(x)) equals hy4f_fp8_round_trip(x) bit for bit.
void hy4f_fp8_encode_row(const float* x, uint32_t n, uint8_t* row);
void hy4f_fp8_decode_row(const uint8_t* row, uint32_t n, float* x);
// The router: sigmoid scores, the top_k of (score + bias), weights = the unbiased scores normalised and scaled.
void hy4f_route(const float* logits, const float* bias, uint32_t E, uint32_t top_k, bool norm, float scale, int32_t* idx, float* w);

// The compressed indexer of one indexed layer, kept on the host: token keys as they arrive, a group key when its
// `ratio` tokens are in, and the recall of one query.
struct Hy4fIndexer {
    uint32_t ratio = 0, heads = 0, hd = 0, group_topk = 0, rope_dim = 0;
    float    theta = 0.f, eps = 0.f; uint32_t mrope[3] = {0, 0, 0};
    std::vector<float> tok_norm, kv, kg, key_norm, ape;   // [hd], [hd, hd], [hd, hd], [hd], [ratio, hd]
    std::vector<float> tok_keys;                           // [n_tokens, hd]  normed
    std::vector<float> grp_keys;                           // [n_groups, hd]  pooled, normed, rotated at the group's first position
    uint32_t n_tokens() const { return hd ? uint32_t(tok_keys.size() / hd) : 0; }
    uint32_t pending = 0;                                  // token keys computed on the card and not read back yet (they follow the n_tokens() held here)
    uint32_t n_groups() const { return hd ? uint32_t(grp_keys.size() / hd) : 0; }
    void rewind(uint32_t n);                               // keep the first n tokens
    // raw [hd] = token_key_projection(x) of the token at position n_tokens()
    // pos3_all (optional): [n_tokens, 3] the 3-D positions of every token so far -- a group's key is rotated at its FIRST
    // token's position; nullptr = position = token index on all three axes (text)
    void append(const float* raw, const int32_t* pos3_all = nullptr);
    // q [heads, hd] = query_projection(q_resid) (unrotated), head_w [heads] = head_weight_projection(x) (unscaled), for
    // the query at position pos. True: `allow` [pos + 1] (when given) holds 1 for the keys this query may attend. False: the context
    // is within the budget for this query -- every key up to pos is allowed and `allow` is not written.
    // `picked` (optional) receives the group_topk - 1 earlier groups chosen, ascending.
    bool recall(const float* q, const float* head_w, uint32_t pos, uint8_t* allow, std::vector<float>& scratch, std::vector<uint32_t>* picked = nullptr, int threads = 1, const int32_t* pos3 = nullptr) const;   // pos3: the query's 3-D position (nullptr = pos on all axes)
    // The same for T rows at positions pos0.. (q [T, heads, hd], head_w [T, heads], allow [T] rows `stride` apart): a
    // chunk's rows over a team, one row's groups over the team; a row within the budget gets 1 for every key up to it.
    // picked (optional): per row, the groups chosen (empty for a row within the budget).
    // The pick alone, from scores computed elsewhere (the card): scores [T] rows `stride` apart, row t's first
    // (pos0 + t) / ratio entries valid. picked[t] = the group_topk - 1 best groups, ascending (empty within the budget).
    void select_rows(const float* scores, size_t stride, uint32_t pos0, uint32_t T, std::vector<std::vector<uint32_t>>& picked, int threads) const;
    void recall_rows(const float* q, const float* head_w, uint32_t pos0, uint32_t T, uint8_t* allow, size_t stride,
                     std::vector<std::vector<uint32_t>>* picked, int threads, const int32_t* pos3_rows = nullptr) const;
};

// Rows that are not plain text tokens (an image's rows): their 3-D positions and their embeddings.
struct Hy4fRowInputs {
    const int32_t* pos3 = nullptr;      // [T, 3] (t, h, w) of every row of the call; nullptr = each row one past the largest axis of the row before it
    const float*   embeds = nullptr;    // [n, dim] fp32: the embeddings of the rows flagged in is_embed, in order
    const uint8_t* is_embed = nullptr;  // [T] 1 = the row's embedding comes from `embeds` (and the row owns no n-gram)
};
class Hy4fForward {
public:
    Hy4fForward();
    ~Hy4fForward();
    Hy4fForward(const Hy4fForward&) = delete;
    Hy4fForward& operator=(const Hy4fForward&) = delete;

    // One queue per card (in-order). "" on success.
    std::string init(const std::vector<sycl::queue*>& qs, const Hy4FlashModel& m, const Hy4fOptions& o);
    // ids [T] continue the sequence at position n_pos(). logits: [T, vocab] (all_rows) or [vocab] (the last row).
    std::string forward(const int32_t* ids, uint32_t T, std::vector<float>& logits, bool all_rows, const Hy4fRowInputs* in = nullptr);
    // the position a text row appended now would take (one past the largest axis of the last row; 0 at the start)
    int32_t  next_position() const { const size_t n = pos3_.size(); return n < 3 ? 0 : std::max({pos3_[n - 3], pos3_[n - 2], pos3_[n - 1]}) + 1; }
    void     reset() { rewind(0); }
    void     rewind(uint32_t n);
    uint32_t n_pos() const { return n_pos_; }
    // Diagnostic: every forward() appends the streams ENTERING each layer, [n_layers, T, hc, dim] fp32, to *dst.
    // last_row_only: [n_layers, hc, dim] of the call's last row (what an activation recording of a prompt keeps).
    void set_stream_dump(std::vector<float>* dst, bool last_row_only = false) { dump_ = dst; dump_last_ = last_row_only; }
    // Diagnostic: experts picked per layer of the last call, [n_layers][T * top_k] (empty rows for the dense layer).
    const std::vector<std::vector<int32_t>>& last_routes() const { return routes_; }
    // Diagnostic (the mini gate): the reference's discrete choices by absolute position -- routes [MoE layer ordinal][n_pos][top_k]
    // (ascending), recall [indexed layer ordinal][n_pos][group_topk] (the own group last). Every row's own choice is compared
    // with them (ref_diffs()); with `force` the reference's choice is the one used, so a near-tie that fell the other way
    // does not hide whether the arithmetic agrees.
    void set_reference_routes(const int32_t* r, uint32_t n_pos, bool force, uint32_t n_moe_layers = ~0u) { ref_routes_ = r; ref_n_ = n_pos; force_routes_ = force; ref_route_layers_ = n_moe_layers; }
    void set_reference_recall(const int32_t* r, uint32_t n_pos, bool force) { ref_recall_ = r; ref_n_ = n_pos; force_recall_ = force; }
    struct RefDiffs { uint32_t route_rows = 0, recall_rows = 0; std::vector<std::pair<uint32_t, uint32_t>> routes, recalls; };   // (layer, position)
    const RefDiffs& ref_diffs() const { return ref_diffs_; }
    // Run-time ablation (the search for an abliteration): after the embedding and after every write into the streams (the
    // n-gram step, attention, the FFN) the components along the k orthonormal rows of `basis` ([k, dim] fp32) are removed
    // from every stream, scaled by `strength` (1 = removed). k = 0 switches it off. What scripts/hy4flash_ablit/overlay.py
    // bakes into the writers, without a reload per candidate.
    std::string set_ablation(const float* basis, uint32_t k, float strength = 1.f);
    // Run-time expert mask (the same search): the listed (layer, expert) pairs are never routed to -- the router takes its
    // next candidates instead. An empty list clears it. Baked into a model by a large negative e_score_correction_bias.
    std::string set_expert_mask(const std::vector<std::pair<uint32_t, uint32_t>>& layer_expert);
    // Run-time steering (a "control vector"): after layer L has written, alpha * vectors[L, i, :] is added to stream i of
    // every row. vectors: [n_layers, hc, dim] fp32 (zeros = leave the layer alone); nullptr or alpha 0 switches it off.
    // Not expressible in this model's weights (its sublayers have no bias behind their output norms): an engine feature.
    std::string set_steering(const float* vectors, float alpha);
    // Calibration recorder (for data-aware rounding of the experts): from now on, every multi-token call with the 8-bit
    // expert path computes ALL its expert jobs on the CPU and accumulates, per MoE layer, the Gram matrices of the
    // experts' latent inputs (xtx) and of their hidden activations pooled over the layer's experts (hth). Slow by design.
    void record_cov(bool on, bool inputs_only = false) { cov_on_ = on && !inputs_only; cov_x_ = on; }
    // (inputs_only: only the latent inputs' Gram matrix, at normal speed -- the residents stay on the cards)
    // <prefix>.xtx.f64 / <prefix>.hth.f64: float64 [n_moe_layers, K, K]; <prefix>.json: the sample counts. "" on success.
    std::string write_cov(const std::string& prefix) const;
    // The checkpoint's drafter (DFly2: a block of [the token just sampled, mask x N] through 5 layers that read the target's
    // keys; reference: vLLM models/hy_v4/nvidia/dfly*.py, docs/research/2026-11-19-hy4-dfly-drafter-reference.md).
    bool     has_drafter() const;
    uint32_t draft_width() const;               // drafts a round proposes
    // After a forward(): the greedy drafts for the positions after `bonus`, the token sampled from row `row` of that
    // call (the call's later rows, if any, must have been rewound away). Changes nothing the next forward() relies on.
    // pick(k, logits [vocab]): the token of draft k (default: the largest logit) -- a sampler keeps the logits it was
    // shown for the acceptance test. conf: the checkpoint's confidence head per draft (its acceptance rate as trained).
    // stop_below > 0: the chain ends before the draft at which the product of the confidences so far falls under it
    // (fewer drafts than draft_width(), possibly none: every further draft costs a read of the 742 MB head).
    std::string draft(uint32_t row, int32_t bonus, std::vector<int32_t>& drafts, const std::function<int32_t(uint32_t, const float*)>& pick = {},
                      std::vector<float>* conf = nullptr, float stop_below = 0.f);
    uint64_t vram_bytes(size_t card) const;
    size_t   n_cards() const { return cards_.size(); }
    struct Stats { double ms_total = 0, ms_attn = 0, ms_ffn = 0, ms_engram = 0, ms_indexer = 0; uint64_t calls = 0, rows = 0, q8_hits = 0, q8_uploads = 0, q8_cpu = 0, q8_streamed = 0, q8_promotions = 0, q8_rebalanced = 0; double q8_ms_cpu = 0; double q8_build_s = 0, q8_ms_upload = 0, q8_ms_kernel = 0, ms_router = 0, ms_ffn_dense = 0; };
    const Stats& stats() const { return st_; }
    uint32_t dense_q8_mats() const { return n_q8_mats_; }
    uint32_t q8_slots_of(size_t card) const { return card < q8_slots_used_.size() ? q8_slots_used_[card] : 0; }   // expert slots a layer on that card
    bool pack_in_ram() const;                 // every card's mapped expert pack has been paged in (the background thread finished)
    double pack_ram_gib() const;      // non-expert matrices held as 8-bit (q8_dense)
    // IE_HY4F_PROF=1: every step of a layer waited for and timed on its own (slower overall; the shares are what matter)
    const std::vector<std::pair<std::string, double>>& profile() const { return prof_; }
    // rows a call may hold in VRAM the expert slots lend (0: none; calls hold up to max_tokens)
    uint32_t big_rows() const { return big_ok_ ? rows_max() : 0; }
    void free_all();

private:
    struct Mat { const sycl::half* w16 = nullptr; const int8_t* qs = nullptr; const uint16_t* d = nullptr; };   // fp16, or 8-bit SoA
    struct Dense {
        float *hc_a_fn = nullptr, *hc_a_base = nullptr, *hc_a_scale = nullptr, *hc_f_fn = nullptr, *hc_f_base = nullptr, *hc_f_scale = nullptr;
        float *attn_norm = nullptr, *ffn_norm = nullptr, *q_a_norm = nullptr, *kv_a_norm = nullptr, *post_norm = nullptr, *sinks = nullptr;
        sycl::half *q_a = nullptr, *kv_a = nullptr, *wk_t = nullptr, *wv = nullptr;
        Mat q_b, gate, o;
        sycl::half *ix_q = nullptr, *ix_head_w = nullptr, *ix_tok_k = nullptr;
        float* ix_pend = nullptr;                                   // [ix_pend_cap_, ix_head_dim]: the pending token keys of this layer's indexer
        float* ixg_ring = nullptr; mutable uint32_t ixg_ring_seeded = ~0u;   // [8, ix_head_dim]: the normed keys of the decode row's group
        float *ixg_tok_norm = nullptr, *ixg_kv = nullptr, *ixg_kg = nullptr, *ixg_key_norm = nullptr, *ixg_ape = nullptr;   // the indexer's pooling weights on the card (the decode recall on the card)
        float* ix_grp = nullptr;                                    // [ctx / ratio + 2, ix_head_dim]: the indexer's group keys (the host's, mirrored)
        // The latent cache [kv_cap, kv_lora + rope]. An indexed layer keeps every position (kv_cap = ctx). A sliding layer
        // reads only its last `sliding_window` keys: kv_win = the card holds the rows kv_base .. (2 * (window + chunk) of
        // them) and every row is also kept in host RAM (kv_host, for a rewind below kv_base).
        sycl::half *kv = nullptr; uint32_t kv_cap = 0; mutable uint32_t kv_base = 0; bool kv_win = false; int32_t kv_host = -1; uint32_t kv_stage = 0;
        // The heads this card computes of the layer: all of them, or -- a sliding layer split over two cards (tensor
        // parallel) -- one half: the home card (tp, it also owns the layer's norms, streams and FFN) holds heads 0 .. nh - 1,
        // the other card the rest. kv_mirror: this half writes the layer's keys to host RAM.
        uint32_t nh = 0, h0 = 0; bool tp = false, kv_mirror = false;
        Mat mlp_gate, mlp_up, mlp_down; float* mlp_post_norm = nullptr;
        float *router_w = nullptr, *moe_norm = nullptr;
        Mat lat_down, lat_up, sh_gate, sh_up, sh_down;
        std::vector<float> router_b;                                // host [E]
        // n-gram layer
        sycl::half *eg_value_w = nullptr; std::vector<sycl::half*> eg_key_w;
        float *eg_value_b = nullptr, *eg_key_b = nullptr, *eg_norm_key = nullptr, *eg_norm_query = nullptr, *eg_scales = nullptr, *eg_pre_gamma = nullptr;   // [dim] or [hc, dim]
        std::vector<float> eg_head_norms;                           // host [n_heads, head_dim]
        int32_t ix = -1;                                            // index into indexers_
    };
    struct Card;
    template <class T> T* dev(Card& c, size_t n);
    void mm(Card& c, const Mat& w, const sycl::half* x16, float* y32, uint32_t T, uint32_t N, uint32_t K);   // y32 [T, N] = x16 [T, K] . W^T
    std::string upload_card(Card& c);
    std::string alloc_rows(Card& c, size_t T); std::string rows_setup(Card& c); std::string rows_mode(bool big);
    uint32_t rows_max() const { return std::max(opt_.max_tokens, opt_.big_tokens); }
    bool big_ok_ = false;
    std::string run_card(Card& c, size_t ci, const int32_t* ids, uint32_t T, uint32_t pos0, std::vector<float>& logits, bool all_rows);
    std::string attention(Card& c, const Dense& d, const Hy4fLayer& lw, uint32_t L, uint32_t T, uint32_t pos0);        // attention_core + the output norm
    std::string attention_core(Card& c, const Dense& d, const Hy4fLayer& lw, uint32_t L, uint32_t T, uint32_t pos0);   // -> c.o32 [T, dim]: the heads of d through o
    std::string attention_tp(Card& home, Card& help, const Dense& dh, const Dense& dp, const Hy4fLayer& lw, uint32_t L, uint32_t T, uint32_t pos0);
    void stage_helper(Card& p, uint32_t T, uint32_t pos0);
    struct TpWorker; std::unique_ptr<TpWorker> tpw_;
    struct Dfly; std::unique_ptr<Dfly> dfly_;       // the drafter's weights and scratch on the last card, the target's last recall choices   // the host thread that drives the other card's half of a split layer for a chunk
    std::string engram(Card& c, const Dense& d, const Hy4fEngram& eg, uint32_t T, uint32_t pos0);
    std::string exact_moe(Card& c, const Hy4fLayer& lw, uint32_t T);
    std::string q8_moe(Card& c, uint32_t L, uint32_t T);

    const Hy4FlashModel* m_ = nullptr;
    Hy4fOptions opt_;
    std::vector<std::unique_ptr<Card>> cards_;
    std::vector<Hy4fIndexer> indexers_;
    std::vector<int32_t> hist_;                 // every token fed, by position (the n-gram layers read back two)
    std::vector<float> h_x_, h_stage_, h_cos_s_, h_sin_s_, h_cos_i_, h_sin_i_, h_emb_, h_scr_;
    std::vector<uint8_t> h_allow_;
    std::vector<int32_t> pos3_;               // [n_pos, 3] every row's 3-D position (text: the same value on all axes)
    std::vector<uint8_t> h_egmask_; bool eg_any_masked_ = false;   // this call: rows that own an n-gram (0 = an image row or the pad id)
    std::vector<int32_t> h_sel_, h_cnt_;
    std::vector<std::pair<sycl::half*, size_t>> kv_mirror_;   // per windowed layer: every key row [ctx, kv_lora + rope] in host RAM (mapped, touched as written)
    std::vector<uint32_t> ix_dev_groups_;     // per indexer: group keys the card holds (never more than the host's)
    std::vector<float> h_ixsc_;               // the card's group scores of a call, read back [T, groups]      // an indexed layer past the recall budget: the keys each row attends ([T, S], -1 pads), their count
    std::vector<int32_t> h_ridx_, h_lo_, h_hi_; std::vector<float> h_rw_;
    std::vector<std::vector<int32_t>> routes_;
    bool tail_fused_ = false; const float* tp_add_ = nullptr;      // a decode pass's attention end fused with the post-add (and the other card's half)
    std::vector<float>* dump_ = nullptr; bool dump_last_ = false;
    uint32_t n_pos_ = 0, ctx_ = 0;
    const int32_t *ref_routes_ = nullptr, *ref_recall_ = nullptr; uint32_t ref_n_ = 0; bool force_routes_ = false, force_recall_ = false;
    RefDiffs ref_diffs_;
    uint32_t ref_route_layers_ = ~0u; bool stopped_ = false;
    std::vector<std::pair<std::string, double>> prof_; std::chrono::steady_clock::time_point prof_t_; bool prof_on_ = false;
    void tick(sycl::queue& q, const char* name);
    uint32_t n_q8_mats_ = 0;
    std::vector<uint32_t> q8_slots_used_;
    uint32_t ix_pend_cap_ = 0; std::vector<float> h_ixp_;
    bool cov_on_ = false, cov_x_ = false; std::vector<double> cov_xtx_, cov_hth_; std::vector<uint64_t> cov_nx_, cov_nh_; std::vector<float> h_hall_;
    std::vector<int32_t> h_rng_;
    std::vector<float> steer_; float steer_alpha_ = 0.f; uint64_t steer_version_ = 0;
    std::map<uint32_t, std::vector<int32_t>> eg_lookup_; std::vector<uint64_t> h_eg_rows_; std::vector<const uint8_t*> h_eg_addr_;   // the n-gram step's host tables
    std::vector<std::vector<uint8_t>> mask_; std::vector<float> h_bias_;      // [n_layers][E] 1 = masked; the masked bias of the layer in hand
    std::vector<float> abl_basis_; uint32_t abl_k_ = 0; float abl_strength_ = 1.f; uint64_t abl_version_ = 0;
    Stats st_;
};

}  // namespace ie
