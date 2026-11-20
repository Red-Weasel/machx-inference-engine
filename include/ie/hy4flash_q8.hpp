// include/ie/hy4flash_q8.hpp — Hy4-flash routed experts as 8-bit integers (docs/hy4flash, 2026-11-18).
//
// Why: measured against Tencent's reference, 8-bit integer experts (one fp16 scale per 32 weights of a row: the engine's
// Q8_0-SoA code) sit inside the model's own numerical noise (KL 0.012 against 0.010 for full precision and 0.019 for the
// reference run in bfloat16), while the 4-bit float pack the MXFP4 tier needs does not (0.125). This store is the fast
// path at that quality: mapped from the 8-bit pack scripts/hy4flash_oracle/pack_q8.py writes (<ie_dir>/experts-q8-LNN.bin,
// the slot layout below) or, for a layer with no pack file, built at load from the source's BF16 experts; computed by the engine's
// W8A16 kernels (gemv_q8_0_soa_f16_rows / _rows_dual: fp16 activations, nothing else rounded).
//
// One store per card, for that card's MoE layers. Every expert of those layers lives in host RAM as one SLOT
//   [gate qs | up qs | down qs | gate d | up d | down d]      qs int8 [N][K], d fp16 bits [N][K / 32]
// (gate / up: N = moe_inter, K = latent; down: N = latent, K = moe_inter; 7.5 MB at the release's widths), and a number
// of VRAM slots per layer hold the experts in use: the first by a usage ranking, the rest replaced least-recently-used.
#pragma once
#include "ie/hy4flash.hpp"

#include <sycl/sycl.hpp>
#include <atomic>
#include <condition_variable>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#include <deque>
#include <memory>
#include <mutex>
#include <cstdint>
#include <thread>
#include <string>
#include <vector>

namespace ie {

// The width of the stored integers, per matrix kind (gate / up share one, down has its own). A row of K weights is
//   8 bits  int8 [K]                                                    (K bytes;      |q| <= 127)
//   5 bits  nibbles [K / 2] then one high-bit word per 32 [K / 8]        (5 K / 8;      q + 16 in 0..31, |q| <= 15)
//   4 bits  nibbles [K / 2]                                              (K / 2;        q + 8 in 0..15, |q| <= 7)
// weight 2j of a block in the low nibble of byte j, 2j + 1 in the high one; bit i of the high-bit word = weight i's
// fifth bit. Always one fp16 scale per 32 weights. Every kernel below takes the format; 8 / 8 is the default.
struct Hy4fQFmt {
    uint32_t gu = 8, dn = 8;
    static uint32_t row_bytes(uint32_t bits, uint32_t K) { return bits == 8 ? K : bits == 5 ? K / 2 + K / 8 : K / 2; }
    static bool known(uint32_t bits) { return bits == 8 || bits == 5 || bits == 4; }
};

class Hy4fQ8Store {
public:
    ~Hy4fQ8Store();
    // Builds the RAM slots of layers [first_layer, first_layer + n_layers) from the source (OpenMP), allocates
    // `vram_slots` VRAM slots a layer and fills them with the first experts of ranking[layer]. "" on success.
    std::string init(sycl::queue& q, const Hy4FlashModel& m, uint32_t first_layer, uint32_t n_layers, uint32_t vram_slots,
                     const std::vector<std::vector<uint32_t>>& ranking, Hy4fQFmt fmt = {}, uint64_t lend_bytes = 0);
    // Lending (lend_bytes > 0 at init): every layer's slab is two pieces of physical VRAM, the tail at least lend_bytes. lend()
    // maps every layer's tail into one contiguous range and returns it (the experts there become misses, the layers keep
    // the slots below); take_back() maps the tails home and fills those slots with the best-scoring experts not resident
    // (copies enqueued on q). The caller waits for every queue that may read either range first. A long prompt reads with a
    // big workspace in the lent range; decode gets every slot back.
    uint64_t lendable() const { return lend_on_ ? uint64_t(nL_) * (res_bytes_ - tail_off_) : 0; }
    uint8_t* lend(sycl::queue& q);
    void take_back(sycl::queue& q, uint64_t token);
    bool lent() const { return lent_; }
    uint32_t live_slots() const { return n_live_; }
    // The pack file of a layer is <ie_dir>/experts-q8-LNN.bin at 8 / 8 and experts-q<gu><dn>-LNN.bin otherwise (q55, q48 ..).
    Hy4fQFmt fmt() const { return fmt_; }
    // A mapped pack is paged into RAM by a background thread started at init (IE_HY4F_PRELOAD=0: not): an expert the
    // cards do not hold is computed on the CPU from this mapping, and one whose pages are still on the drive costs a
    // 7.5 MB read instead of 0.12 ms (measured 2026-11-18: the same 15.8 % misses ran at 11.9 tok/s cold, 16.3 warm).
    bool warm_done() const { return warm_done_.load(); }
    uint64_t warm_bytes() const { return warm_bytes_.load(); }
    void free_all(sycl::queue& q);
    struct Planes { const int8_t *gate_qs, *up_qs, *down_qs; const uint16_t *gate_d, *up_d, *down_d; };
    // The VRAM planes of expert e of model layer L, uploading it over the layer's least recently used slot when it is not
    // resident. Every use is enqueued on `q` (in order) before the next acquire of that layer may replace the slot.
    std::string acquire(sycl::queue& q, uint32_t L, uint32_t e, Planes& out);
    // Decode with a CPU leg: is expert e of layer L in VRAM (then acquire() is a hit); its planes in RAM (host pointers).
    bool resident(uint32_t L, uint32_t e) const { return L >= L0_ && L < L0_ + nL_ && e < E_ && slot_of_[L - L0_][e] >= 0; }
    Planes ram_planes(uint32_t L, uint32_t e) const;
    // Residency follows use. Every pick adds 1 to its expert's score, which halves every kHalfLife tokens (so a score
    // is about 370 x the expert's picks per token). An expert outside VRAM takes the slot of the layer's lowest-scoring
    // resident when its score exceeds that one's by kPromoteGain: an upload is 7.5 MB (0.64 ms), a miss on the CPU
    // 0.125 ms, so a swap has to save about five misses, which a score difference of 3-4 does within a few hundred
    // tokens. (History, 2026-11-18: "3 misses in 64 tokens" uploaded 43 experts a token; "twice the victim's score + 1"
    // swapped on a single pick against an unused resident and almost never against a used one.)
    //   promote()    one missed expert during decode; slots used since `mark` are in this call's kernels and stay
    //   rebalance()  after a chunk of a prompt has been read: up to `budget` swaps for the layer, best candidates first
    // The residents the ranking put in VRAM start with kSeedScore (a prior that one unused prompt overturns).
    void note_pick(uint32_t L, uint32_t e, uint64_t token);
    uint64_t mark() const { return clock_; }
    bool promote(sycl::queue& q, uint32_t L, uint32_t e, uint64_t token, uint64_t mark, Planes& out);
    uint32_t rebalance(sycl::queue& q, uint32_t L, uint64_t token, uint32_t budget);
    //   plan_promotions()  rebalance's choice and bookkeeping without the copies: (expert, slot) pairs whose bytes the caller
    //                      writes into slab(L) + slot * slot_bytes() itself (a prompt chunk copies them from its staging area)
    std::vector<std::pair<uint32_t, int32_t>> plan_promotions(uint32_t L, uint64_t token, uint32_t budget);
    // promote_async(): promote()'s choice of victim, but the 7.5 MB upload goes to a worker thread with its own queue, so
    // the compute queue never waits for it (a synchronous promote() held the card ~0.75 ms an upload, ~3.3 uploads a
    // decode token). The expert stays a miss -- computed on the CPU -- until activate_pending() makes it resident; forward()
    // calls that at the start of every call, so placement changes at the same points for the same tokens however long the
    // copy takes (it waits there for an unfinished copy: a token is ~40 ms, a copy ~0.75 ms). IE_HY4F_ASYNC_PROMOTE=0: promote().
    bool promote_async(sycl::queue& q, uint32_t L, uint32_t e, uint64_t token, uint64_t mark);
    void activate_pending();
    static constexpr float kHalfLife = 256.f, kPromoteGain = 3.f, kSeedScore = 1.f;
    // A chunk's kernels address experts as (the layer's VRAM slab, a slot index): the slot of a resident expert (marked
    // used, counted as a hit) or -1; the slab; where the six planes sit inside a slot.
    int32_t slot_touch(uint32_t L, uint32_t e) {
        const int32_t s = slot_of_[L - L0_][e];
        if (s >= 0) { stamp_[L - L0_][size_t(s)] = ++clock_; ++st_.hits; }
        return s;
    }
    const uint8_t* slab(uint32_t L) const { return dev_[L - L0_]; }
    struct Offsets { uint64_t up, down, gate_d, up_d, down_d; };
    Offsets offsets() const { return {off_up_, off_down_, off_gate_d_, off_up_d_, off_down_d_}; }
    uint64_t vram_bytes() const { return vram_bytes_; }
    uint64_t ram_bytes() const { return ram_bytes_; }
    uint64_t slot_bytes() const { return slot_bytes_; }
    bool ram_pinned(uint32_t L) const { return L >= L0_ && L - L0_ < pinned_.size() && pinned_[L - L0_]; }
    const uint8_t* ram_slot(uint32_t L, uint32_t e) const { return ram_[L - L0_] + size_t(e) * slot_bytes_; }
    // bytes of one expert at a format (the slot layout above, padded to 64)
    static uint64_t slot_size(uint32_t MI, uint32_t LAT, Hy4fQFmt fmt) {
        const uint64_t raw = 2 * uint64_t(MI) * Hy4fQFmt::row_bytes(fmt.gu, LAT) + uint64_t(LAT) * Hy4fQFmt::row_bytes(fmt.dn, MI) + 2 * uint64_t(MI) * (LAT / 32) * 2 + uint64_t(LAT) * (MI / 32) * 2;
        return (raw + 63) / 64 * 64;
    }
    uint32_t vram_slots() const { return n_slots_; }
    struct Stats { uint64_t hits = 0, uploads = 0, promotions = 0; double build_s = 0; uint32_t mapped_layers = 0; };   // mapped_layers: read from the pack, not converted
    const Stats& stats() const { return st_; }
    void reset_stats() { st_.hits = st_.uploads = st_.promotions = 0; }

private:
    uint32_t L0_ = 0, nL_ = 0, E_ = 0, MI_ = 0, LAT_ = 0, n_slots_ = 0, n_live_ = 0, lend_live_ = 0;
    bool lend_on_ = false, lent_ = false;
    std::vector<sycl::ext::oneapi::experimental::physical_mem> ph_head_, ph_tail_;   // [layer] (lending)
    uint64_t res_bytes_ = 0, tail_off_ = 0; uintptr_t lend_va_ = 0;  // a layer's reserved range, where its tail starts; the lent range
    std::unique_ptr<sycl::context> vm_ctx_;
    Hy4fQFmt fmt_;
    std::thread warm_; std::atomic<bool> warm_stop_{false}, warm_done_{false}; std::atomic<uint64_t> warm_bytes_{0};
    void stop_warm();
    struct Pending { uint32_t l, e; int32_t slot; uint64_t seq; sycl::event after; };   // after: a barrier on the compute queue
    std::vector<Pending> pend_;                 // promoted since the last activate_pending()
    std::unique_ptr<sycl::queue> up_q_;         // the upload worker's queue (the store's device and context)
    std::thread up_thr_; std::mutex up_mu_; std::condition_variable up_cv_, up_done_cv_;
    std::deque<Pending> up_jobs_; std::vector<uint8_t> up_fail_; uint64_t up_submitted_ = 0, up_finished_ = 0, up_base_ = 0; bool up_stop_ = false;
    void stop_uploader();
    bool pending(uint32_t l, uint32_t e) const { for (const auto& p : pend_) if (p.l == l && p.e == e) return true; return false; }
    uint64_t slot_bytes_ = 0, off_up_ = 0, off_down_ = 0, off_gate_d_ = 0, off_up_d_ = 0, off_down_d_ = 0;
    std::vector<uint8_t*> ram_;                 // [layer] E slots
    std::vector<uint8_t>  mapped_;              // [layer] 1 = ram_[layer] is a mapping of the pack file (munmap), 0 = our own memory (free)
    std::vector<uint8_t>  registered_;          // [layer] 1 = the mapping is registered with reg_ctx_ for direct copies (prepare_for_device_copy)
    std::unique_ptr<sycl::context> reg_ctx_;
    std::vector<uint8_t>  pinned_;              // [layer] 1 = ram_ is pinned host memory of pin_ctx_ (IE_HY4F_PIN_PACK=1): the cards DMA from it directly
    std::unique_ptr<sycl::context> pin_ctx_;
    void unmap(size_t l);
    std::vector<uint8_t*> dev_;                 // [layer] n_slots slots (device USM)
    std::vector<std::vector<int32_t>> slot_of_; // [layer][expert] -> VRAM slot or -1
    std::vector<std::vector<int32_t>> in_slot_; // [layer][slot] -> expert or -1
    std::vector<std::vector<uint64_t>> stamp_;  // [layer][slot] last use
    std::vector<std::vector<float>> score_; std::vector<std::vector<uint64_t>> score_tok_;      // [layer][expert]: decayed pick count, the token it was last brought up to date at
    float eff(uint32_t l, uint32_t e, uint64_t token) const;
    uint64_t clock_ = 0, vram_bytes_ = 0, ram_bytes_ = 0;
    Stats st_;
};

// One decode token's routed experts in two launches (src/ops/hy4flash_q8_moe.cpp): y [latent] (overwritten) =
// sum_j w[j] * down_j . ( silu(min(gate_j . x, lim)) * clamp(up_j . x, +-lim) ) over the n_jobs <= 16 experts whose VRAM
// planes are pl[j]. x16 [latent] fp16, hj [n_jobs * moe_inter] fp16 scratch; all 32-byte aligned device memory.
sycl::event hy4f_q8_moe_decode(sycl::queue& q, const sycl::half* x16, const Hy4fQ8Store::Planes* pl, const float* w, uint32_t n_jobs,
                               sycl::half* hj, float* y, uint32_t LAT, uint32_t MI, float lim, Hy4fQFmt fmt = {});

// A CHUNK's routed experts that are in VRAM, three launches for the whole chunk (src/ops/hy4flash_q8_moe.cpp):
// n_jobs (row, expert) pairs sorted by row -- job_slot [n_jobs] the expert's slot in `slab`, job_row [n_jobs] its token
// row, job_w [n_jobs] its routing weight, row_start [T + 1]: the jobs of row t are [row_start[t], row_start[t + 1]).
// y [T, latent] (overwritten) = per row, the weighted sum of its jobs; a row without jobs gets 0. x16 [T, latent] fp16;
// hj [n_jobs, moe_inter] fp16 and yj [n_jobs, latent] fp32 scratch. All device memory.
sycl::event hy4f_q8_moe_rows(sycl::queue& q, const sycl::half* x16, const uint8_t* slab, uint64_t slot_bytes, Hy4fQ8Store::Offsets off,
                             const int32_t* job_slot, const int32_t* job_row, const float* job_w, const int32_t* row_start,
                             uint32_t n_jobs, uint32_t T, sycl::half* hj, float* yj, float* y, uint32_t LAT, uint32_t MI, float lim, Hy4fQFmt fmt = {});
// The same for a chunk's MISSED experts on the CPU: y [T, latent] += per row, the weighted sum of its jobs; jobs in any
// order (sorted by expert keeps an expert's weights in cache for its rows). x [T, latent] fp32 on the host.
void hy4f_q8_moe_cpu_rows(const float* x, const Hy4fQ8Store::Planes* pl, const int32_t* job_row, const float* w, uint32_t n_jobs,
                          uint32_t T, float* y, uint32_t LAT, uint32_t MI, float lim, int threads, Hy4fQFmt fmt = {}, float* h_all = nullptr);
// (h_all [n_jobs, moe_inter], optional: every job's hidden activation, for the calibration recorder)
// A [K, K] (float64, row-major) += R^T R for n rows R [n, K] fp32 -- the Gram accumulation of the calibration recorder.
void hy4f_gram_add(const float* R, size_t n, uint32_t K, double* A, int threads);

// A decode token's small fp16 products as direct kernels (src/ops/hy4flash_small.cpp), with the contracts of
// gemm_nt_f16_onednn (K a multiple of 32), gemm_bmm_nt_f16_onednn (IPG a multiple of 8) and gemm_fp16_onednn.
sycl::event hy4f_f16_gemv_f32(sycl::queue& q, const sycl::half* x16, const sycl::half* w16, float* y, uint32_t K, uint32_t N, uint32_t T);
sycl::event hy4f_f32_gemv_f32(sycl::queue& q, const float* x32, const float* w32, float* y, uint32_t K, uint32_t N, uint32_t T);   // fp32 x fp32 (the router), K % 8 == 0
sycl::event hy4f_f16_bmm_f32(sycl::queue& q, const sycl::half* a16, const sycl::half* w16, float* y, uint32_t M, uint32_t G, uint32_t IPG, uint32_t OPG);
sycl::event hy4f_f16_bmm_place(sycl::queue& q, const sycl::half* x16, const sycl::half* w16, sycl::half* a16, uint32_t M, uint32_t G, uint32_t IPG, uint32_t OPG, uint32_t LW);
sycl::event hy4f_f16_bmm_gate(sycl::queue& q, const sycl::half* c16, const sycl::half* w16, const float* g32, sycl::half* y16, uint32_t M, uint32_t G, uint32_t IPG, uint32_t OPG);
// A decode pass of 2 .. 8 rows: the resident experts any row picked, each read once (job_slot / job_mask [n_jobs] on the
// card: the expert's slot, a bit per row that picked it; row_jobs / row_w [T, TK]: pick k's job or -1, its weight).
// hj [n_jobs, T, MI] fp16 and yj [n_jobs, T, LAT] fp32 are scratch; y [T, LAT] = each row's resident picks summed in pick order.
sycl::event hy4f_q8_moe_mrows(sycl::queue& q, const sycl::half* x16, const uint8_t* slab, uint64_t slot_bytes, Hy4fQ8Store::Offsets off,
                              const int32_t* job_slot, const uint32_t* job_mask, const int32_t* row_jobs, const float* row_w, uint32_t n_jobs, uint32_t T, uint32_t TK,
                              sycl::half* hj, float* yj, float* y, uint32_t LAT, uint32_t MI, float lim, Hy4fQFmt fmt);
// y [T, N] fp32 = x16 [T, K] . W^T, T <= 8, the 8-bit matrix read once for all rows
sycl::event hy4f_q8_gemv_rows_f32(sycl::queue& q, const sycl::half* x16, const int8_t* qs, const uint16_t* d, float* y, uint32_t K, uint32_t N, uint32_t T);
sycl::event hy4f_f16_scores_sel(sycl::queue& q, const sycl::half* a16, const sycl::half* kv, const int32_t* sel, float* scores, uint32_t B, uint32_t NH, uint32_t LW, uint32_t S);
sycl::event hy4f_f16_mix_sel(sycl::queue& q, const sycl::half* p16, const sycl::half* kv, const int32_t* sel, const int32_t* hi, sycl::half* ctx16, uint32_t B, uint32_t NH, uint32_t LW, uint32_t S);
sycl::event hy4f_ix_rotate(sycl::queue& q, const float* q32, const float* cs, float* qrot, uint32_t T, uint32_t heads, uint32_t hd, uint32_t rope_dim);
sycl::event hy4f_ix_scores(sycl::queue& q, const float* qrot, const float* hw, const float* grp, const int32_t* own, float* score, uint32_t T, uint32_t heads, uint32_t hd, uint32_t GC, float hs);
sycl::event hy4f_f16_mix_f16(sycl::queue& q, const sycl::half* p16, const sycl::half* b16, sycl::half* c16, uint32_t M, uint32_t N, uint32_t K);
// The attention of a few rows (decode, T <= 8) in one launch: the scores against the allowed keys, the sink softmax and the
// weighted sum of the keys -- what hy4f_f16_bmm_f32 + sink_softmax + hy4f_f16_mix_f16 (or the gathered-key path) do in
// three launches. ctx16 [rows, NH, LW] = sum_j p[r, h, j] * K_j over j in [lo[r], hi[r]], p = softmax(a16[r, h, :] . K_j *
// scaling | the head's sink logit), K_j = keys + j * LW, or keys + sel[r * S + j] * LW when sel is given. fp32 scores,
// probabilities and sums. LW a multiple of 8 and at most kAttnMaxLW; at most kAttnMaxKeys keys a row.
constexpr uint32_t kAttnMaxKeys = 4096, kAttnMaxLW = 640;
sycl::event hy4f_attn_decode(sycl::queue& q, const sycl::half* a16, const sycl::half* keys, const int32_t* sel, uint32_t S, const int32_t* lo, const int32_t* hi,
                             const float* sinks, sycl::half* ctx16, uint32_t rows, uint32_t NH, uint32_t LW, float scaling, bool use_sink);
// The same attention on the XMX units (src/ops/hy4flash_attn_xmx.cpp): split-key flash decoding, keys LW = 576 wide for the
// scores, the context over the first KL = 512 (W_v is zero under the rest, written as zeros), heads a multiple of 32.
// n_axis: the keys of the axis when sel is null (the row's allowed ones are [lo[r], hi[r]]); with sel, row r's keys are
// sel[r * S + 0 .. hi[r]]. Workspace per queue, grown on demand.
bool hy4f_attn_xmx_fits(uint32_t NH, uint32_t LW, uint32_t KL);
sycl::event hy4f_ix_select_rows(sycl::queue& q, const float* score, size_t stride, uint32_t rows, uint32_t pos0, uint32_t keep,
                                uint32_t ratio, uint32_t S, int32_t* sel, int32_t* lo, int32_t* hi);
// prompt rows of an indexed layer: each row against its chosen keys, flash-style on the XMX units (no workspace)
sycl::event hy4f_attn_rows_xmx(sycl::queue& q, const sycl::half* a16, const sycl::half* keys, const int32_t* sel, uint32_t S,
                               const int32_t* lo, const int32_t* hi, const float* sinks, sycl::half* ctx16,
                               uint32_t rows, uint32_t NH, float scaling, bool use_sink);
// A prompt chunk's experts expert-major on the XMX units (src/ops/hy4flash_moe_grp.cpp; 8-bit slots):
sycl::event hy4f_moe_grp_gu(sycl::queue& q, const sycl::half* x16, const uint64_t* job_base, const int32_t* job_j0, const int32_t* job_n,
                            const int32_t* j_row, uint32_t n_jobs, Hy4fQ8Store::Offsets off, sycl::half* hj, uint32_t LAT, uint32_t MI, float lim);
sycl::event hy4f_moe_grp_down(sycl::queue& q, const sycl::half* hj, const uint64_t* job_base, const int32_t* job_j0, const int32_t* job_n,
                              const float* j_w, uint32_t n_jobs, Hy4fQ8Store::Offsets off, float* yj, uint32_t LAT, uint32_t MI);
sycl::event hy4f_moe_grp_sum(sycl::queue& q, const float* yj, const int32_t* pos, uint32_t T, uint32_t TKp, float* y, uint32_t LAT);
// Fused small steps of a decode pass (src/ops/hy4flash_fuse.cpp), the parts' own arithmetic:
sycl::event hy4f_hc_apply_norm(sycl::queue& q, const float* S, const float* part, uint32_t n_parts, const float* base, const float* scale,
                               float* pre, float* post, float* x, const float* w, sycl::half* xn16, float* xn32,
                               uint32_t T, uint32_t hc, uint32_t H, uint32_t HH, float eps, float hc_eps, float mag, float norm_eps);
sycl::event hy4f_norm_post_add(sycl::queue& q, float* o, const float* add, const float* axpy_x, const float* w, const float* post, float* S,
                               uint32_t T, uint32_t hc, uint32_t H, float eps);
// The indexer of a decode row on the card (src/ops/hy4flash_ix_gpu.cpp): a completed group's pooled key, and the row's
// exact recall (top keep groups by score, the lower index first among equals) written as the attention's key list.
sycl::event hy4f_ix_norm_into(sycl::queue& q, const float* raw, const float* tok_norm, float* out, uint32_t hd, float eps);
sycl::event hy4f_ix_group_build(sycl::queue& q, const float* nk, const float* kv, const float* kg, const float* key_norm,
                                const float* ape, const float* cs, float* grp_out, uint32_t ratio, uint32_t hd, uint32_t rope_dim, float eps);
sycl::event hy4f_ix_select(sycl::queue& q, const float* score, uint32_t GC, uint32_t keep, uint32_t ratio, uint32_t p, uint32_t S,
                           int32_t* sel, int32_t* lo, int32_t* hi);
sycl::event hy4f_attn_decode_xmx(sycl::queue& q, const sycl::half* a16, const sycl::half* keys, const int32_t* sel, uint32_t S,
                                 uint32_t n_axis, const int32_t* lo, const int32_t* hi, const float* sinks, sycl::half* ctx16,
                                 uint32_t rows, uint32_t NH, float scaling, bool use_sink);

// BF16 n-gram table rows [rows, hd] -> FP8 rows [rows, 4 + hd] (hy4f_fp8_encode_row), over an OpenMP team. Every row is
// decoded again and compared with hy4f_fp8_round_trip of the source row; returns the number of rows that differ (0).
uint64_t hy4f_engram_pack_fp8(const uint8_t* bf16, uint64_t rows, uint32_t hd, uint8_t* out);

// Page n addresses of a memory-mapped file in at once (first and last byte of `bytes` at each, over an OpenMP team): the
// n-gram tables are 127 GB on the drive and a token needs 64 scattered rows of them -- read one page fault at a time that
// was 6-45 ms a token on text the page cache had not seen (2026-11-18).
void hy4f_prefault(const uint8_t* const* addrs, size_t n, size_t bytes, int threads);

// The CPU leg (src/ops/hy4flash_q8_cpu.cpp): y [latent] += sum_j w[j] * expert_j(x) for n_jobs experts given by their
// RAM planes (Hy4fQ8Store::ram_planes). x [latent] fp32 on the host. AVX2 integer dot products over an OpenMP team.
// threads <= 0: min(the OpenMP default, 16).
void hy4f_q8_moe_cpu(const float* x, const Hy4fQ8Store::Planes* pl, const float* w, uint32_t n_jobs, float* y,
                     uint32_t LAT, uint32_t MI, float lim, int threads, Hy4fQFmt fmt = {});

// The big NON-expert matrices as 8-bit (Hy4fOptions::q8_dense; src/ops/hy4flash_q8_moe.cpp): y [T, N] fp32 = x16 [T, K] . W^T
// for T <= 4 rows (each row reads the weights once), and the matrix back as fp16 into a scratch buffer for a many-row GEMM.
// K a multiple of 32; qs and x16 16-byte aligned.
sycl::event hy4f_q8_gemv_f32(sycl::queue& q, const sycl::half* x16, const int8_t* qs, const uint16_t* d, float* y,
                             uint32_t K, uint32_t N, uint32_t T);
sycl::event hy4f_q8_dequant_f16(sycl::queue& q, const int8_t* qs, const uint16_t* d, sycl::half* w16, uint32_t N, uint32_t K);
// hy4f_q8_from_bf16 over an OpenMP team (rows are independent)
void hy4f_q8_from_bf16_mt(const uint8_t* bf16, uint32_t N, uint32_t K, int8_t* qs, uint16_t* d);

// One row-block's worth of the conversion (exposed for the unit check): BF16 [N, K] -> qs int8 [N][K], d fp16 bits
// [N][K / 32]; d = fp16(max|w| / 127) of the block, qs = round(w / d) clamped to +-127.
void hy4f_q8_from_bf16(const uint8_t* bf16, uint32_t N, uint32_t K, int8_t* qs, uint16_t* d);
// The same for any stored width (plain rounding to |q| <= 2^(bits - 1) - 1, the row layout of Hy4fQFmt); `container8`
// keeps those integers in the 8-bit layout -- the same arithmetic through the 8-bit kernels, to check the low-bit ones.
void hy4f_qn_from_bf16(const uint8_t* bf16, uint32_t N, uint32_t K, uint32_t bits, bool container8, uint8_t* rows, uint16_t* d);

}  // namespace ie
