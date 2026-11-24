// include/ie/inkling2.hpp — Inkling 2 Mini (Hugging Face `inkling2_loop`) text model, port phase 1.
//
// A correct, unoptimised forward pass on ONE card, gated against the publisher's reference (scripts/inkling2/gate.sh).
// The op-level specification is docs/inkling2/SPEC.md; section numbers in comments refer to it. Ops:
// include/ie/inkling2_ops.hpp. Tool: tools/ie_inkling2_run.cpp.
//
// Placement (phase 1): dense matrices on the device as their BF16 bits (decoded in the kernels; exact), small
// vectors (norms, scales, conv taps, attn_res / message queries, router bias) as fp32, expert banks in their stored
// form (NVFP4 nibbles + E4M3 scales + per-expert factor on layers 0..19, MXFP8 bytes + UE8M0 exponents on 20..23)
// decoded inside the expert kernels, the embedding table on the host (rows gathered per call). About 17 GiB.
//
// State (SPEC 6): one cache entry per (layer, pass), 48 in all, entry index = Inkling2Config::entry(L, pass): keys
// and values plus the four 3-row conv histories (k, v, attn_sconv, mlp_sconv). forward() appends at the entries'
// current length, in chunks of at most max_chunk tokens, so one call serves a prompt and a single-token decode step.
// Global entries (8) keep every position (row = position, grown by doubling or reserve()). Sliding entries (40) are
// rings of window + rollback rows (row = position % rows; rollback >= max_chunk): a chunk's own writes must not
// overwrite keys its first query still sees, and truncate() must find the keys it rewinds to. The four conv histories
// per entry are rings of the conv INPUTS, rollback + 3 rows, so a rewound position finds its 3 predecessors.
#pragma once

#include "ie/inkling2_ops.hpp"
#include "ie/safetensors.hpp"

#include <sycl/sycl.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ie {

struct Inkling2Config {
    uint32_t hidden = 0, n_layers = 0, vocab = 0, vocab_unpadded = 0;
    uint32_t n_heads = 0, n_kv_heads = 0, head_dim = 0, d_rel = 0, rel_extent = 0, window = 0, sconv_k = 0;
    uint32_t block = 0, n_experts = 0, top_k = 0, inter = 0;
    uint32_t loop_start = 0, loop_end = 0, passes = 0, gate_rank = 0;
    float    log_n_floor = 0, log_alpha = 0, eps = 0, qk_clip = 0, rel_clip = 0, route_scale = 0;
    float    gate_softcap = 0, up_softcap = 0, final_softcap = 0;

    bool     is_global(uint32_t L) const { return (L + 1) % block == 0; }               // SPEC 0: 5, 11, 17, 23
    bool     is_loop(uint32_t L) const { return L >= loop_start && L < loop_end; }      // 6..17
    bool     is_boundary(uint32_t L) const { return L % block == 0; }                   // 0, 6, 12, 18
    uint32_t n_loop() const { return loop_end - loop_start; }
    uint32_t n_calls() const { return n_layers + (passes - 1) * n_loop(); }             // 48 layer evaluations
    // SPEC 6 cache entry of (layer, pass); pass is 1-based and only > 1 on loop layers.
    uint32_t entry(uint32_t L, uint32_t pass) const {
        return (pass <= 1 || !is_loop(L)) ? L : n_layers + (pass - 2) * n_loop() + (L - loop_start);
    }
};

// Parse config.json text (the `text_config` object) and refuse anything this runtime does not implement.
// Returns "" on success.
std::string inkling2_parse_config(const std::string& json_text, Inkling2Config& c);
std::string inkling2_read_config(const std::string& model_dir, Inkling2Config& c);

// The 48 layer calls in execution order (SPEC 5.1): 0-5, (6..17) pass 1, 2, 3, then 18-23.
struct Inkling2Call { uint32_t layer = 0, pass = 1; };
std::vector<Inkling2Call> inkling2_schedule(const Inkling2Config& c);

// An ablation set (abliteration at run time, the weights untouched): groups of orthonormal unit directions removed from
// the writes into the stream. A write is the embedding output x0, a call's attention output after attn_sconv, or a call's
// MoE output after mlp_sconv (SPEC 4.1: the only vectors that enter the partial state h and, through it, the block list).
// At a write a group names: w -= strength * sum_k (w . r_k) r_k. Every read-out (attn_res, messages, final) takes RMS-
// normalised convex mixes of block states and partials, which are sums of writes, so with a group on EVERY write the
// whole stream (and everything read from it) has no component along the group's directions -- the run-time equivalent
// of orthogonalising every writer's output matrix. Without a set the forward is the unmodified one, bit for bit.
//
// File: <name>.json  {"format": "ie-ablation-v1", "arch": "inkling2_loop", "hidden": 2304, "file": "<blob>.f32",
//   "rows": K, "groups": [{"rows": [first, count], "strength": 1.0, "embed": true, "attn": "all" | [call, ...],
//   "moe": "all" | [call, ...]}], ...}. The blob is K x hidden float32 (relative paths: next to the json). Calls are the
// 48 layer calls in execution order (inkling2_schedule). A write may belong to one group at most.
// Steering: a group with "mode": "add" (default "remove") instead adds a fixed vector to each write it names:
// w += strength * sum_k r_k (its rows need not be orthonormal; strength may be negative). A write may belong to one
// remove group and one add group; both: w' = (w - s_r sum_k (w . r_k) r_k) + s_a sum_j a_j. The recorder keeps the
// writes before both (kinds 1, 2) and the partial state after both (kind 0).
struct Inkling2Ablation {
    struct Group {
        uint32_t row0 = 0, k = 0;
        float    strength = 1.f;
        bool     embed = false;
        bool     keep_norm = false;          // rescale each ablated write to its original length (experiment)
        bool     add = false;                // "mode": "add" (steering): w += strength * sum_k r_k
        std::vector<uint8_t> attn, moe;      // [n_calls]: 1 = this call's write is ablated
    };
    std::vector<float> dirs;                 // [rows, hidden]
    uint32_t rows = 0;
    std::vector<Group> groups;
    std::string path, summary;
};
// Read and check an ablation file (unit, orthonormal rows within each group to 1e-3; one group per write). "" on success.
std::string inkling2_read_ablation(const std::string& json_path, const Inkling2Config& c, Inkling2Ablation& out);

struct Inkling2LoadOptions {
    // The ablation set: "" = <model dir>/ie_ablation.json when that file exists (IE_INKLING2_ABLATION overrides: a path,
    // or "off"); "off" = none; otherwise a path to an ablation json.
    std::string ablation;
    // Emulate the publisher's reference arithmetic: bf16 rounding at every point it rounds (SPEC 9), expert weights
    // rounded to bf16 after decoding, and the NVFP4 factor from bf16(amax) / bf16(denominator) (SPEC 9, trap 24).
    // For comparing single calls with the goldens below their bf16 noise; NOT the engine's arithmetic.
    bool ref_bf16 = false;
    // Largest chunk forward() runs at once (prompts are split): sets the scratch size and the sliding rings' slack.
    uint32_t max_chunk = 256;
    // How far truncate() can rewind behind the furthest position ever written (0 = max_chunk; never less than
    // max_chunk). Sizes the sliding rings (window + rollback rows) and the conv input rings (rollback + 3 rows).
    uint32_t rollback = 0;
    // Attention keys / values cached in bf16 (half the bytes; logits ~5e-3 relL2 from fp32, 15-30 % further from the
    // reference at 8.5K). kv_bf16: the one-conversation state (default fp32: quality first); kv_bf16_batch: the batch
    // slots (default bf16: memory per slot decides how many conversations fit). IE_INKLING2_KV=fp32 / bf16 sets both.
    // The fused fp32 path only: the bf16 emulation mode and IE_INKLING2_UNFUSED keep fp32.
    bool kv_bf16 = false;
    bool kv_bf16_batch = true;
    // Two cards on ONE conversation (tensor parallel): a second in-order queue on the other card, in the SAME context as
    // load()'s queue (peer access is enabled here). Each card holds a full copy of the weights; per layer call the
    // attention heads are split between the cards (each its 12 query / 4 key-value heads: half the K/V cache each),
    // and at one token per step also the attention output projection's rows, the 8 picked experts (4 each) and the
    // unembed's rows -- every split joined by a direct card-to-card copy, so each card ends a call with the same full
    // vectors and the arithmetic of every value is the one-card arithmetic (attention partial merges aside: fp noise).
    // forward() etc. drive both cards (the second from a worker thread). The batch_* calls are refused. nullptr = one card.
    sycl::queue* tp_queue = nullptr;
};

struct Inkling2RunOptions {
    const std::vector<std::vector<std::array<int32_t, 8>>>* forced_routes = nullptr;  // [calls][T][8]: impose picks
    const std::vector<std::vector<float>>* teacher = nullptr;   // [calls][T * H]: call i's output replaced by teacher[i]
    bool keep_calls = false;                                    // return each call's output
    uint32_t chunk = 0;                                         // prefill chunk (0 = max_chunk; clamped to it)
    bool record_routes = true;                                  // read the picks / margins back after every call
    // Recorder: the last record_last positions of this forward, per call, the vectors of kInk2RecKinds (fp32), into
    // record_out [record_last][calls][kInk2RecKinds][hidden] (resized). 0 = off.
    uint32_t record_last = 0;
    std::vector<float>* record_out = nullptr;
    // Images (include/ie/inkling2_vision.hpp, Inkling2Vision::encode): image_rows [n, hidden] replace the inputs at the
    // positions image_at [n] (indices into THIS forward's ids, the placeholder tokens) after embed_norm and before any
    // embedding ablation -- the reference's inputs_embeds.masked_scatter. nullptr = none.
    const std::vector<float>* image_rows = nullptr;
    const std::vector<uint32_t>* image_at = nullptr;
    // The draft head's target features (mtp_config.target_layer_ids): for each visit v in feature_visits, the partial
    // state entering call v (the reference's aux hidden state), or for v == the number of calls the final-norm output;
    // every row of this forward, into features [taps][T][hidden] (resized; fp32). nullptr = off. Not with batch slots.
    const std::vector<uint32_t>* feature_visits = nullptr;
    std::vector<float>* features = nullptr;
    // The softcapped logits of every row of the forward (a draft verify), into Inkling2Result::all_logits [T][vocab];
    // the forward must fit one chunk. Not with batch slots or two cards.
    bool all_logits = false;
};
// Recorded per call: 0 the partial state after the call (h_out, after any ablation), 1 the attention write (after
// attn_sconv, BEFORE any ablation), 2 the MoE write (after mlp_sconv, before any ablation), 3 the MoE read-out u (after
// the mlp controller; the input of mlp_norm).
constexpr uint32_t kInk2RecKinds = 4;

struct Inkling2Result {
    std::vector<float>                last_logits;     // [vocab_unpadded], softcapped, last position of the call
    std::vector<float>                all_logits;      // [T][vocab_unpadded] with Inkling2RunOptions::all_logits
    std::vector<std::vector<float>>   calls;           // [calls][T * H] when keep_calls
    std::vector<std::vector<int32_t>> own_routes;      // [calls][T * 8]: this run's own top-8 (ascending ids)
    std::vector<std::vector<int32_t>> used_routes;     // [calls][T * 8]: the picks actually used
    std::vector<std::vector<float>>   route_margin;    // [calls][T]: 8th minus 9th selection score
};

class Inkling2Model {
public:
    Inkling2Model();
    ~Inkling2Model();
    Inkling2Model(const Inkling2Model&) = delete;
    Inkling2Model& operator=(const Inkling2Model&) = delete;

    // Read config.json and model-00001-of-00001.safetensors from `dir`, upload to q's device. "" on success.
    std::string load(const std::string& dir, sycl::queue& q, const Inkling2LoadOptions& opt = {});

    // Run ids at positions [n_pos(), n_pos() + T) through the model, appending to the per-(layer, pass) state.
    // Teacher forcing and forced routes index calls of THIS forward. "" on success.
    std::string forward(const std::vector<int32_t>& ids, const Inkling2RunOptions& opt, Inkling2Result& out);

    void     reset_state();                              // forget every cached position
    // Rewind the context to its first n positions (prompt-cache reuse): the next forward() continues at position n
    // exactly as if positions >= n had never been fed. Works for earliest_restorable() <= n <= n_pos(); n = 0 always.
    std::string truncate(uint32_t n);
    // The smallest n truncate() accepts: (furthest position ever written) - rollback, floored at 0. Global entries keep
    // every position; the limit comes from the sliding rings and the conv input rings.
    uint32_t earliest_restorable() const;
    // BENCHMARK ONLY: pretend n positions are cached (all keys, values and conv inputs zero) so decode speed at a long
    // context can be timed without prefilling it. The outputs that follow are meaningless.
    std::string fill_zero_context_for_benchmark(uint32_t n);
    // One decode step: feed `id` at position n_pos() and return the softcapped logits [vocab_unpadded].
    std::string decode_step(int32_t id, std::vector<float>& logits);
    std::string reserve(uint32_t ctx);                   // size the global entries for ctx positions now
    // Replace the ablation set (nullptr = none: the unmodified forward). Takes effect at the next forward; the cached
    // state is NOT recomputed (reset_state() before comparing). "" on success.
    std::string set_ablation(const Inkling2Ablation* a);
    const std::string& ablation_summary() const;         // "" = none
    // Device bytes of the state (all 48 entries) and of the per-chunk scratch, for a context / chunk (no device needed).
    static size_t state_bytes_for(const Inkling2Config& c, uint32_t ctx, uint32_t max_chunk);   // rollback = max_chunk
    static size_t state_bytes_for(const Inkling2Config& c, uint32_t ctx, uint32_t max_chunk, uint32_t rollback,
                                  bool kv_bf16 = false);
    bool kv_bf16() const;                                // the one-conversation state's keys / values are bf16
    bool batch_kv_bf16() const;                          // the batch slots' keys / values are bf16
    std::string kv_note() const;                         // "fp32 (one conversation), bf16 (batch slots) - why" for load logs
    static size_t scratch_bytes_for(const Inkling2Config& c, uint32_t max_chunk, bool teacher);
    uint32_t n_pos() const;
    size_t   device_bytes() const;                       // weights + state + scratch now allocated
    const uint16_t* unembed_device() const;              // the bf16 unembed rows [vocab_unpadded, hidden] on the card

    // ---- batched decoding: up to `slots` independent conversations on this card, one pass per decode step for all of
    // them. Each conversation's arithmetic is the same as alone (forward() on a fresh state, then one token at a time),
    // bit for bit. The single state above is untouched; forward() etc. keep working between batch calls. Needs the fp32
    // path (not ref_bf16) and ablation groups of at most 4 directions.
    // Allocate the slots' state for up to ctx positions each (replaces an earlier batch). "" on success. chunk: the
    // slots' prefill chunk, which is also how far a slot can rewind (batch_truncate): its rings are window + chunk rows.
    std::string batch_begin(uint32_t slots, uint32_t ctx, uint32_t chunk = 64);
    // Reset slot s and run `ids` through it as forward() would on a fresh state (opt's recorder applies).
    std::string batch_prefill(uint32_t slot, const std::vector<int32_t>& ids, const Inkling2RunOptions& opt, Inkling2Result& out);
    // Append `ids` to slot s (as forward() on its state); out.last_logits = the last position's.
    std::string batch_forward(uint32_t slot, const std::vector<int32_t>& ids, const Inkling2RunOptions& opt, Inkling2Result& out);
    // Rewind slot s to its first n positions (as truncate()); n >= batch_earliest_restorable(s).
    std::string batch_truncate(uint32_t slot, uint32_t n);
    uint32_t batch_earliest_restorable(uint32_t slot) const;
    uint32_t batch_slots() const;
    uint32_t batch_ctx() const;
    // One token per listed slot (distinct; ids[i] for slots[i]) at that slot's next position, all in one pass.
    // logits: [slots.size()][vocab_unpadded], softcapped.
    std::string batch_step(const std::vector<uint32_t>& slots, const std::vector<int32_t>& ids, std::vector<float>& logits);
    // The same step with the draws on the card (i2_sample_rows): rows[i] for slots[i]; recent / pf_*: the arrays the rows'
    // rec_off / pf_off index. out[i]: the id drawn for slots[i], -1 for non-finite logits. No logits come back.
    std::string batch_step_sample(const std::vector<uint32_t>& slots, const std::vector<int32_t>& ids, const std::vector<I2SampleRow>& rows,
                                  const std::vector<int32_t>& recent, const std::vector<int32_t>& pf_id, const std::vector<float>& pf_d,
                                  std::vector<int32_t>& out);
    uint32_t batch_pos(uint32_t slot) const;              // positions cached in a slot
    // BENCHMARK ONLY: slot s pretends n zero positions are cached (as fill_zero_context_for_benchmark).
    std::string batch_fill_zero_for_benchmark(uint32_t slot, uint32_t n);
    void     batch_end();                                 // free the slots' state
    // Device bytes one slot needs at ctx positions (its state, partials, logits), rollback = max_chunk.
    static size_t batch_slot_bytes(const Inkling2Config& c, uint32_t ctx, uint32_t max_chunk, bool kv_bf16 = true);
    const Inkling2Config& config() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace ie
