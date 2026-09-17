// include/ie/hyv4.hpp — GLM-5.3-Flash (`hyv4`) model runtime.
//
// Port plan: docs/glm53/PORT_PLAN.md. Reference graphs: llama.cpp PR 27754
// (build_kda_layer, build_indexer) + the engine's own glm52_run.cpp (absorbed
// MLA + sigmoid-bias routing, same family) + deepseek4.cpp (Sinkhorn HC).
//
// v0 bring-up placement (single B70, correctness before speed — the exact
// qwen4exp v0 recipe):
//  - DEVICE, F16: every non-expert matrix (KDA q/k/v/o + low-rank gates,
//    MLA lora stack + absorbed k_b/v_b, indexer mats, shexp, dense FFN,
//    embeddings, lm_head), dequantized host-side at load and TRANSPOSED to
//    [K, N] row-major per the gemv_fp16/gemm_fp16 contract (ops.hpp) unless
//    noted. ~19 GiB projected — fits one 32 GB card. Q8-SoA residency is a
//    later perf phase (qwen4exp-certified pattern).
//  - DEVICE, F32: norms (incl. the indexer's LayerNorm WITH bias), Sinkhorn
//    HC fn/base/scale, router gate_inp + exp_probs_b (fp32 — fragile
//    routing chain, ds4/qwen4exp precedent), KDA ssm_a / dt_bias, indexer
//    proj (F32 in the file — bf16 flips near-tie rankings, PR warning) and
//    compressor ape.
//  - HOST (GGUF mmap, raw quant): the 3 routed-expert banks per MoE block
//    (gate/up mostly Q4_K, down mostly Q5_K — UD dynamic quants vary per
//    layer; the bank views carry each tensor's own dtype), streamed through
//    a per-layer VRAM slot cache at P2+.
//  - The MTP block (blk.45, nextn.*) is BOUND (shape-checked, bank recorded)
//    but NOT uploaded in v0 — it only serves speculative decode (P3).
#pragma once

#include "ie/allocator.hpp"
#include "ie/deltanet_state.hpp"
#include "ie/gguf.hpp"
#include "ie/model_config.hpp"

#include <sycl/sycl.hpp>

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <functional>

namespace ie {

// Device weights for one block. Pointers are F16 [K, N]-transposed unless
// noted. Null when the tensor does not exist on this block (KDA vs full-attn
// vs dense-FFN vs MoE — Hyv4Config::is_full_attn()/is_dense_layer(); the
// MTP block additionally carries NO hyper-connection tensors).
// One dense 2-D projection, in exactly one residency:
//   qs/d  — Q8_0-SoA planes (qs[n*K+b*32+i] int8, d[n*(K/32)+b] f16 bits),
//           GGUF-native N-major, consumed by gemv_q8_0_soa_f16_g/_rows
//           (W8A16 — quality-neutral, the qwen4exp-certified default), OR
//   w     — F16 [K, N]-transposed for gemv/gemm_fp16 (fallback for non-Q8_0
//           files and IE_HY4_DENSE_Q8=0).
struct Hyv4DW {
    sycl::half* w  = nullptr;
    int8_t*     qs = nullptr;
    uint16_t*   d  = nullptr;
    // Raw packed K-quant residency (P4): the GGUF bytes verbatim on device,
    // served by gemv_q{4,5,6}_K at T==1 and dequant_*_to_Bt + gemm_fp16 at
    // T>1. 2.9x smaller than the f16 expansion — the freed VRAM is ecache.
    const uint8_t* raw = nullptr;
    DType          rdt = DType::kCount;
    explicit operator bool() const noexcept { return w || qs || raw; }
};

struct Hyv4Layer {
    // -- hyper-connections (Sinkhorn, deepseek4 flavor): F32, GGUF-native ---
    // fn [hc*H=16384, mix=24]; absent on the MTP block.
    float* hc_attn_fn    = nullptr;
    float* hc_attn_base  = nullptr;   // [24]
    float* hc_attn_scale = nullptr;   // [3]
    float* hc_ffn_fn     = nullptr;
    float* hc_ffn_base   = nullptr;
    float* hc_ffn_scale  = nullptr;

    float* attn_norm = nullptr;       // [H] RMS gamma
    float* ffn_norm  = nullptr;       // [H]

    // -- KDA (linear blocks; PR build_kda_layer) ----------------------------
    // q/k/v [H -> 8192] (64 heads x 128); SEPARATE projections + SEPARATE
    // per-channel causal conv1d (kernel 4) per stream, SiLU AFTER conv.
    Hyv4DW kda_q;
    Hyv4DW kda_k;
    Hyv4DW kda_v;
    Hyv4DW kda_o;                     // [8192 -> H]
    // depthwise conv weights, F16, k-major [4, 8192] (GGUF [4, 1, 8192]).
    sycl::half* conv_q = nullptr;
    sycl::half* conv_k = nullptr;
    sycl::half* conv_v = nullptr;
    float* ssm_a   = nullptr;         // [64] per-head A
    float* dt_bias = nullptr;         // [8192] per (head, channel)
    Hyv4DW f_a;                       // [H -> 128]   gate low-rank in
    Hyv4DW f_b;                       // [128 -> 8192] gate low-rank out
    Hyv4DW g_a;                       // [H -> 128]   output-gate low-rank in
    Hyv4DW g_b;                       // [128 -> 8192] output-gate low-rank out
    Hyv4DW beta;                      // [H -> 64] per-head delta step
    sycl::half* ssm_norm = nullptr;   // [128] per-head gated-RMS weight

    // -- MLA + DSA indexer (full-attn blocks; NOPE — no rope anywhere) ------
    Hyv4DW q_a;                       // [H -> 1536]
    float*      q_a_norm = nullptr;   // [1536]
    Hyv4DW q_b;                       // [1536 -> 16384] (64 heads x 256)
    Hyv4DW kv_a;                      // [H -> 512] (latent only — no rope part)
    float*      kv_a_norm = nullptr;  // [512]
    // Absorbed per-head projections, F16 GGUF-NATIVE element order (the
    // absorb-form GEMM layout is a P2 decision; shapes are [256,512,64] /
    // [512,256,64]).
    sycl::half* k_b = nullptr;
    sycl::half* v_b = nullptr;
    Hyv4DW attn_out;                  // [16384 -> H]
    Hyv4DW attn_gate;                 // [H -> 16384] sigmoid attention gate
    float* attn_sinks = nullptr;      // [64] per-head sink logits, F32
    sycl::half* idx_k      = nullptr; // indexer.attn_k [H -> 128]
    float*      idx_k_norm_w = nullptr;  // [128] LayerNorm WITH bias, eps 1e-6
    float*      idx_k_norm_b = nullptr;  // [128]
    sycl::half* idx_q_b    = nullptr; // indexer.attn_q_b [1536 -> 4096] (32 x 128)
    float*      idx_proj   = nullptr; // [H, 32] per-head weights, F32 kept F32
    float*      idx_ape    = nullptr; // indexer_compressor_ape [128, 4]
    sycl::half* idx_gate   = nullptr; // indexer_compressor_gate [H -> 128]

    // -- FFN: dense (blocks 0-2) OR MoE (blocks 3+) -------------------------
    Hyv4DW ffn_gate;                  // dense [H -> 12288]
    Hyv4DW ffn_up;
    Hyv4DW ffn_down;                  // [12288 -> H]
    float* router     = nullptr;      // ffn_gate_inp [H, E] fp32, GGUF-native rows
    float* probs_bias = nullptr;      // exp_probs_b.bias [E]
    Hyv4DW shexp_gate;                // [H -> 2048]
    Hyv4DW shexp_up;
    Hyv4DW shexp_down;                // [2048 -> H]

    // -- routed expert banks: HOST, raw GGUF quant (streamed at P2+) --------
    const GgufTensorInfo* gate_exps = nullptr;  // [H, 2048, 288]
    const GgufTensorInfo* up_exps   = nullptr;  // [H, 2048, 288]
    const GgufTensorInfo* down_exps = nullptr;  // [2048, H, 288]
    // Fill sources: the mmap bytes by default; PINNED host copies when
    // IE_HY4_PIN_BANKS is on (load-time; page cache released as copied) —
    // a miss fill is then one async DMA at ~28 GB/s with zero CPU bytes.
    const uint8_t* gate_src = nullptr;
    const uint8_t* up_src   = nullptr;
    const uint8_t* down_src = nullptr;
    // pread fill addressing (fd + file offset per bank tensor; -1 = unresolved,
    // fill falls back to memcpy-from-mmap)
    int gate_fd = -1, up_fd = -1, down_fd = -1;
    uint64_t gate_foff = 0, up_foff = 0, down_foff = 0;
    // Pinned repack: per-expert CONTIGUOUS [gate|up|down] slices (the exact
    // ecache slot layout) — a miss fill is ONE async DMA.
    const uint8_t* bank_pin = nullptr;
    bool bank_pinned = false;

    // -- NextN / MTP projections (blk.45 only; bound, NOT uploaded in v0) ---
    const GgufTensorInfo* nextn_eh    = nullptr;  // [2H -> H]
    const GgufTensorInfo* nextn_enorm = nullptr;  // [H]
    const GgufTensorInfo* nextn_hnorm = nullptr;  // [H]
    const GgufTensorInfo* nextn_shn   = nullptr;  // [H] shared_head_norm
};

class Hyv4Model {
public:
    ~Hyv4Model();
    // Drain copy+compute queues, then join the fill worker. Must run BEFORE
    // freeing pinned bounce buffers — an in-flight H2D against a freed pin
    // is the BCS timeout that leaves the display B70 stuck in GT C0 @ 2800.
    void shutdown() noexcept;

    // Loads onto `alloc`'s device. `g` must outlive the model: the routed
    // expert banks stay host-resident inside its mmap. `vram_budget_bytes`
    // gates projected device residency BEFORE any upload (0 = default
    // 28 GiB); IE_ALLOW_OOM=1 bypasses. `layer_lo`/`layer_hi` restrict this
    // instance to blocks [lo, hi) for a later 2-GPU pipeline split (globals
    // bind by role: token_embd with layer 0's owner, lm_head with the tail).
    // The MTP block (n_layers-1 when nextn_predict_layers==1) is never part
    // of the uploaded range; its tensors are bound as host views.
    std::string load(DeviceAllocator& alloc, const GgufReader& g,
                     const Hyv4Config& cfg, uint64_t vram_budget_bytes = 0,
                     uint32_t layer_lo = 0, uint32_t layer_hi = UINT32_MAX);

    const Hyv4Config& config() const noexcept { return cfg_; }
    const Hyv4Layer&  layer(uint32_t il) const { return layers_[il]; }

    // Globals (device).
    sycl::half* token_embd = nullptr;  // [vocab, H] GGUF-native row-gather
    Hyv4DW      lm_head;               // [H -> vocab]
    float*      output_norm = nullptr; // [H]
    // learned iHC output head (output_hc_*): collapses the 4 streams
    float* hh_fn_ = nullptr;           // [hc, hc*H]
    float* hh_base_ = nullptr;         // [hc]
    float* hh_scale_ = nullptr;        // [1]

    uint64_t device_bytes() const noexcept { return dev_bytes_; }

    // One DMA sweep over this stage's pinned banks to re-establish the
    // device's page mappings. Call AFTER every stage has loaded: a later
    // stage's host-USM allocations evict the earlier stage's mappings, and
    // the first forward then re-faults 80+ GiB at ~0.3 GB/s (traced
    // 2026-08-28: stage A first forward 39-48s vs stage B 1.4s).
    std::string warm_banks();
    // Pre-fill the expert cache from a usage profile (IE_HY4_EPROFILE_OUT
    // dump, u64[n_layers x E] pick counts): each owned MoE layer loads its
    // top-C measured experts so the first prompt starts warm instead of
    // relearning from 0% hit. Slots keep LRU stamp 0 (oldest) — an expert
    // the live traffic never touches is the first evicted. Call after
    // init_runtime, before serving.
    std::string warm_cache(const char* profile_path);
    uint64_t host_bank_bytes() const noexcept { return host_bytes_; }

    sycl::queue& queue() const { return alloc_->queue(); }

    // -- expert-parallel decode (IE_HY4_EP_DECODE=1, needs the shared P2P
    // context from IE_HY4_EP=1) ---------------------------------------------
    // Each card computes the experts with (e & 1) == parity of EVERY pinned
    // MoE layer: the owner ships the T<=4 activation rows to the peer, both
    // cards fill misses from the shared host banks on their own PCIe link
    // concurrently, and the peer's fp32 partial is added into the owner's
    // accumulator once (deterministic; fp32 add-order change vs single-card,
    // PPL-gated). Call on BOTH models after every stage loaded + warmed:
    // this model allocates its half-caches for the PEER's layers.
    std::string ep_enable(Hyv4Model& peer, uint32_t parity);

    // -- runtime (v0, single GPU; correctness before speed) -----------------
    // init_runtime: MLA latent caches (12 full-attn layers), KDA state (34
    // linear layers, via DeltaNetState), all workspaces for chunks up to
    // `max_chunk` tokens. v0 attention is EXACT dense MLA over the latent,
    // which IS the sparse path while start_pos + T <= indexer.top_k +
    // kpool - 1 (2051 on the real file — hyv4_n_select in the reference);
    // beyond that forward() refuses honestly (the indexer lands later).
    std::string init_runtime(uint32_t max_ctx, uint32_t max_chunk);
    // Run T tokens at absolute position `start_pos`. When `logits_out` is
    // non-null the LAST row's fp16 logits [vocab] are written there (device
    // pointer; the model-owned logits() buffer is also valid). When
    // `all_logits` is non-null it receives fp32 logits for EVERY row
    // ([T, vocab], device) — the PPL scorer's input. NOTE: gemm_fp16 stores
    // full 8-row output tiles, so `all_logits` must be allocated for
    // round_up(T, 8) rows.
    std::string forward(const int32_t* tokens_host, uint32_t T, uint32_t start_pos,
                        sycl::half* logits_out, float* all_logits = nullptr);
    // Pipeline stage entry (2-GPU layer split; the qwen4exp forward_range
    // pattern, serial v0): wide_in_host == nullptr -> embed here (needs
    // layer_lo == 0); wide_out_host != nullptr -> stop after our blocks and
    // download the wide streams [T, hc, H] fp32; otherwise run the final
    // merge + lm_head (needs layer_hi == n_transformer_layers).
    std::string forward_range(const int32_t* tokens_host, uint32_t T,
                              uint32_t start_pos, const float* wide_in_host,
                              float* wide_out_host, sycl::half* logits_out,
                              float* all_logits = nullptr);
    void        reset_state();
    sycl::half* logits() const noexcept { return logits_; }

    // -- MTP / NextN draft head (spec decode; tail stage only) --------------
    // The GGUF's blk.45 is a full MLA+MoE decoder block WITHOUT hyper-
    // connections plus the nextn fusion tensors. Neither llama.cpp nor
    // transformers implements it; the wiring below is the glm52_run MTP
    // (ik_llama build_deepseek2_mtp) with clean single residuals:
    //   x = eh_proj([RMS(embed(tok), enorm) | RMS(hidden, hnorm)])
    //   x += MLA(RMS(x));  x += MoE+shexp(RMS(x));  logits = head(RMS(x, shn))
    // `hidden` is the main model's stream MEAN row (what feeds output_norm).
    // The MTP keeps its OWN latent cache over decode positions only
    // (mtp_base_..); NOPE attention makes the suffix history well-posed.
    bool mtp_loaded() const noexcept { return mtp_eh_.w || mtp_eh_.qs; }
    // One draft step: fuse (hidden_row, tok) at absolute position `pos`,
    // run blk.45, leave fp16 logits in logits(). hidden_row is a DEVICE fp32
    // [H] row (mean_rows() of the last forward, or mtp_hidden() to chain).
    std::string mtp_step(const float* hidden_row, int32_t tok, uint32_t pos);
    const float* mtp_hidden() const noexcept { return mtp_x_; }   // pre-head x
    const float* mean_rows() const noexcept { return mean_; }     // [T, H] of last forward
    void mtp_reset(uint32_t base) noexcept { mtp_base_ = base; mtp_len_ = base; }

    // -- spec-decode state snapshot (KDA cannot rewind) ---------------------
    // snapshot(): D2D-save every owned KDA layer's scan+conv state + depth
    // markers. restore(): put them back (partial-accept path replays the
    // accepted rows through a normal forward afterwards).
    std::string snapshot_state();
    std::string restore_state();
    // Cheaper partial-accept path: with spec-verify mode ON, a small-T
    // forward saves each KDA layer's scan/conv INPUT rows; commit_verify(n)
    // then rebuilds the exact state of the accepted prefix from the last
    // snapshot + tiny per-layer re-scans (no replay forward). n in [1, T];
    // n == T is a no-op (the live state already is the full window's).
    void set_spec_verify(bool on) noexcept { spec_verify_ = on; }
    std::string commit_verify(uint32_t n_commit);

    // Per-tensor placement rows for the load report/audit tool.
    struct PlacementRow { std::string name; const char* where; const char* dt; uint64_t bytes; };
    const std::vector<PlacementRow>& placement() const noexcept { return placement_; }

private:
    void free_all();
    std::string run_block(uint32_t L, uint32_t T, uint32_t start_pos);
    std::string run_moe(uint32_t L, uint32_t T);
    void ihc_pre_norm(sycl::queue& q, const float* fn, const float* base,
                      const float* scale, const float* norm_w, uint32_t T);
    // Dense projection dispatch: gemv_fp16 at T==1 (the decode-critical
    // shape — gemm_fp16's 128-row tiles waste 127/128 of their compute
    // there), gemm_fp16 otherwise. Exactly one of y16/y32 may be null; the
    // other is filled (via a cast through the mm scratch when the producing
    // kernel's natural dtype differs).
    void mm(const sycl::half* A, const Hyv4DW& W, uint32_t T,
            uint32_t N, uint32_t K, sycl::half* y16, float* y32);
    // Two same-shape Q8-SoA mats, one activation, one launch (rows_dual).
    // Falls back to two mm() if either side is F16. y16/y32 nullable per
    // output the same way as mm().
    void mm_dual(const sycl::half* A, const Hyv4DW& Wa, const Hyv4DW& Wb,
                 uint32_t T, uint32_t N, uint32_t K,
                 sycl::half* ya16, float* ya32,
                 sycl::half* yb16, float* yb32);

    bool shutting_down_ = false;
    DeviceAllocator* alloc_ = nullptr;
    Hyv4Config   cfg_{};
    std::vector<Hyv4Layer> layers_;
    std::vector<void*> owned_;
    std::vector<void*> owned_host_;          // pinned bank copies
    uint64_t dev_bytes_  = 0;
    uint64_t host_bytes_ = 0;
    uint64_t pinned_bytes_ = 0;
    uint32_t layer_lo_ = 0, layer_hi_ = 0;
    // Diagnostic (IE_HY4_DUMP_WIDE): per-forward host staging of the residual
    // after every block's attn mix / ffn mix (see forward_range / run_block).
    std::vector<float>    dump_h_;
    std::vector<uint32_t> dump_hdr_;
    uint64_t              dump_cur_ = 0;
    std::vector<uint16_t> dmoe_h_;       // IE_HY4_DUMP_MOE per-expert staging
    std::vector<uint32_t> dmoe_hdr_;
    uint64_t              dmoe_cur_ = 0;
    std::vector<PlacementRow> placement_;

    // ---- runtime state ----
    uint32_t max_ctx_ = 0, max_chunk_ = 0;
    DeltaNetState dn_;                       // KDA scan + conv states
    std::vector<int32_t> lin_idx_, full_idx_;  // layer -> cache-local index
    sycl::half* lat_cache_ = nullptr;        // [n_full, max_ctx, kv_lora] f16
    uint32_t    kv_len_ = 0;                 // latent depth (uniform, = pos)
    // workspaces (device USM; sized for max_chunk_ = MT tokens)
    int32_t* d_tokens_ = nullptr;            // [MT]
    float* wide_a_ = nullptr;                // [MT, 4, H] fp32 streams
    float* wide_b_ = nullptr;                //   (hc_mix double buffer)
    float* post_ = nullptr;                  // [MT, 4]
    float* ihc_pre_ = nullptr;               // [MT, 4] iHC pre gates
    float* ihc_flat_ = nullptr;              // [MT, hc*H] flat-RMS-normed streams
    float* ihc_mix_ = nullptr;               // [MT, 2*hc] fn projections
    float* comb_ = nullptr;                  // [MT, 4, 4]
    float* coll_ = nullptr;                  // [MT, H]
    float* xn_   = nullptr;                  // [MT, H] normed collapse
    sycl::half* x16_ = nullptr;              // [MT, H]
    float* mix_out_ = nullptr;               // [MT, H] sublayer output
    // KDA
    float* kda_f32a_ = nullptr;              // [MT, 8192] (q gemm / scan q)
    float* kda_f32b_ = nullptr;              // (k)
    float* kda_f32c_ = nullptr;              // (v)
    sycl::half *kda_h16a_ = nullptr, *kda_h16b_ = nullptr, *kda_h16c_ = nullptr;
    float* kda_pre_  = nullptr;              // [MT, 8192] gate pre-activation
    float* kda_g_    = nullptr;              // [MT, 8192] log decay
    sycl::half* kda_lo16_ = nullptr;         // [MT, 128] low-rank mid
    float* kda_beta_ = nullptr;              // [MT, 64]
    sycl::half* kda_z16_ = nullptr;          // [MT, 8192] output gate
    float* kda_out_  = nullptr;              // [MT, 8192] scan out
    sycl::half* kda_y16_ = nullptr;          // [MT, 8192] gated-normed
    // mm() dispatch scratch (see mm): f16 row for T==1 f32-consumers,
    // f32 plane for T>1 f16-consumers.
    sycl::half* mmscr16_ = nullptr;          // [16384]
    sycl::half* dense_bt_ = nullptr;         // [6144*18432] T>1 raw-dense Bt scratch
    float*      mmscr32_ = nullptr;          // [MT, 16384]
    // MLA
    sycl::half* mla_qa16_ = nullptr;         // [MT, q_lora]
    float* mla_qb_  = nullptr;               // [MT, 64*256]
    float* mla_qlat_ = nullptr;              // [MT, 64, kv_lora]
    sycl::half* mla_kv16_ = nullptr;         // [MT, CW=kv_lora+rope] (cache row staging)
    float* mla_kv32_ = nullptr;              // [MT, CW] pre-norm/rope kv projection
    float* mla_qcat_ = nullptr;              // [MT, 64, CW] q_lat(512) | q_pe(64)
    float* gate32_ = nullptr;                // [MT, 64*256] attention gate logits
    float* mla_att_ = nullptr;               // [MT, 64, kv_lora]
    sycl::half* mla_o16_ = nullptr;          // [MT, 64*256]
    // MoE
    float* moe_rl_   = nullptr;              // [MT, E] router logits
    int32_t* moe_top_ = nullptr;             // [MT, top_k]
    float* moe_topw_ = nullptr;              // [MT, top_k]
    int32_t* moe_idx_ = nullptr;             // gather index buffer [MT*top_k]
    float* moe_w_    = nullptr;              // scatter weights     [MT*top_k]
    sycl::half* moe_xg_ = nullptr;           // [MT, H] gathered rows
    float* moe_gu_[2] = {nullptr, nullptr};  // [MT, EF] gate / up
    sycl::half* moe_h16_ = nullptr;          // [MT, EF]
    float* moe_dn_  = nullptr;               // [MT, H]
    float* moe_acc_ = nullptr;               // [MT, H] fp32 accumulator
    sycl::half* emat_g16_ = nullptr;         // dequanted expert gate  Bt [H, EF]
    sycl::half* emat_u16_ = nullptr;         //                    up  Bt [H, EF]
    sycl::half* emat_d16_ = nullptr;         //                  down  Bt [EF, H]
    std::vector<int32_t> h_top_;             // host routing staging
    std::vector<float>   h_topw_;
    std::vector<int32_t> h_idx_;             // flat per-expert gather lists
    std::vector<float>   h_w_;               //   (stable across async H2D)
    // Per-layer LRU raw-quant expert slot cache: a slot holds one expert's
    // gate|up|down slices verbatim; on use they are dequanted from VRAM
    // (no H2D on a hit). Slices are immutable weight copies — cache state
    // never affects numerics, only which bytes cross PCIe.
    struct ECache {
        uint8_t* base = nullptr;             // C * slot_bytes (device)
        uint64_t gate_off = 0, up_off = 0, down_off = 0, slot_bytes = 0;
        std::vector<int16_t>  slot_of;       // [n_experts] -> slot or -1
        std::vector<int32_t>  expert_in;     // [C] -> expert id or -1
        std::vector<uint64_t> last_use;      // [C] LRU stamps
        // R9 prefetch (IE_HY4_PREFETCH): a prefetched slot's fill has had no
        // reader yet — its first demand reader must carry this event.
        std::vector<std::shared_future<sycl::event>> pf_ev;   // [C]
        std::vector<uint8_t> pf_pending;                      // [C]
    };
    std::vector<ECache> ecache_;             // one per MoE layer (empty: dense)
    uint64_t ecache_clock_ = 0;
    uint32_t ecache_slots_ = 0;
    // -- expert-parallel decode state (ep_enable) ---------------------------
    Hyv4Model* ep_peer_ = nullptr;
    uint32_t ep_parity_ = 0;
    std::vector<ECache> ep_cache_;       // this card's half-caches for PEER layers
    uint64_t ep_clock_ = 0;
    sycl::half* ep_x16_ = nullptr;       // peer role: activation landing [8, H]
    sycl::half* ep_xg_ = nullptr;        // peer role: gathered rows [8, H]
    sycl::half* ep_scratch_ = nullptr;   // peer role: solo yg[EF] yu[EF] yd[H];
                                         // grouped T=1 yg[8,EF] yu[8,EF] yd[8,H]
    int32_t* ep_idx_ = nullptr;          // peer role: pick token idx [8*top_k]
    float* ep_w_ = nullptr;              // peer role: pick weights [8*top_k]
    float* ep_acc_ = nullptr;            // peer role: fp32 partial [8, H]
    float* ep_ret_ = nullptr;            // OWNER role: peer partial lands here
    float* ep_vacc_ = nullptr;           // IE_HY4_EP_VERIFY: local recompute of the peer half
    sycl::event ep_add_ev_;              // owner's last partial-add (ret reuse fence)
    // peer role: PER-LAYER host staging for the async idx/w uploads. One
    // shared buffer raced: the owner's host thread runs ahead of the peer
    // queue and overwrote layer L's weights before pq consumed them —
    // peer experts got the wrong layer's routing weights (PPL 1.87 -> 3.10,
    // caught by the chunk=1 PPL gate 2026-08-29).
    std::vector<int32_t> ep_hidx_;       // [n_layers * ep_tkmax_]
    std::vector<float> ep_hw_;
    uint32_t ep_tkmax_ = 0;
    // peer role, mmap bounce: OWN ring sized for the PEER stage's largest
    // slice (the shared pin_ring_ is sized for THIS stage's — overrun risk)
    // EP worker: runs ep_partial OFF the owner's host thread so peer staging
    // and submission overlap the owner's local layer work (host-serial EP
    // measured 1.85 tok/s vs 3.57 local-only, 2026-08-29 06:34).
    struct EpReq {
        const Hyv4Layer* w; uint32_t L, T, n_exp, npick;
        const uint32_t *exps, *ecnt; const int32_t* idx_flat; const float* w_flat;
        const sycl::half* x_host; sycl::event x_ev, ret_dep; float* ret_host;
        // pf_only: no compute — warm ep_cache_[L] with the owner's predicted
        // experts of OUR parity (fills ride our fill thread's pf path).
        bool pf_only = false;
        int32_t pf_pred[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
        // IE_HY4_EP_PUSH: x was already PUSHED into our ep_x16_ (P2P D2D
        // write, owner-waited); push the partial straight into ret_dev on
        // the owner's card instead of bouncing through ret_host.
        bool x_pushed = false;
        float* ret_dev = nullptr;
        std::promise<sycl::event> done;
    };
    struct EpThread {
        std::thread th; std::mutex mu; std::condition_variable cv;
        std::deque<EpReq> reqs; bool stop = false;
        void start(Hyv4Model* m);
        std::shared_future<sycl::event> push(EpReq&& r);
        ~EpThread();
    };
    EpThread ep_thread_;
    // owner-side stable per-layer staging for the request arrays
    std::shared_future<sycl::event> ep_ret_fut_;   // pending peer chain (owner)
    std::vector<uint32_t> ep_oexps_, ep_oecnt_;
    std::vector<int32_t> ep_oidx_;
    std::vector<float> ep_ow_;
    uint8_t* ep_pin_[2] = {nullptr, nullptr};
    sycl::event ep_pin_ev_[2];
    uint32_t ep_pin_idx_ = 0;
    // owner role: pinned host staging for the peer hop — the platform's P2P
    // READ path corrupts (2026-08-10 finding); every EP transfer is a
    // D2H + H2D push pair through these, never a direct D2D copy.
    sycl::half* ep_xstage_ = nullptr;    // x16 rows out [8, H]
    float* ep_rstage_ = nullptr;         // peer partial back [8, H]
    struct PreadPool;   // defined below
    // pread the expert slice [gate|up|down] into `dst` via the pool; falls
    // back to the pooled memcpy on unresolved fds; loud error on IO failure.
    // `pool`/`serial_cp` override the shared prp_/cpool_ for callers on other
    // threads (the prefetch worker) — the pools' rendezvous is single-caller.
    std::string pread_fill_(int gfd, uint64_t goff, const uint8_t* gs, uint64_t gsl,
                            int ufd, uint64_t uoff, const uint8_t* us, uint64_t usl,
                            int dfd, uint64_t doff, const uint8_t* ds2, uint64_t dsl,
                            uint8_t* dst, PreadPool* pool = nullptr,
                            bool serial_cp = false);
    // peer-side prefetch: warm ep_cache_[L] with predicted experts of OUR
    // parity (fills ride our fill thread's pf path; runs on the EpThread,
    // the sole ep_cache_ mutator).
    void ep_prefetch(uint32_t L, const int32_t* p8);
    // peer-side compute of this card's parity half of the OWNER's layer `w`
    std::string ep_partial(const Hyv4Layer& w, uint32_t L, uint32_t T,
                           const uint32_t* exps, const uint32_t* ecnt, uint32_t n_exp,
                           const int32_t* idx_flat, const float* w_flat, uint32_t npick,
                           const sycl::half* x_host, const sycl::event& x_ev,
                           const sycl::event& ret_dep,
                           float* ret_host, bool x_pushed, float* ret_dev,
                           sycl::event* done);
    // Pinned staging ring for miss fills: H2D from pageable mmap runs at
    // ~12 GB/s on this box, from pinned at ~28 (measured 2026-08-28); the
    // ring bounces mmap->pinned on the CPU while the previous entry's H2D
    // is in flight.
    // Threaded pread staging (glm52's PreadPool, hardened): the kernel copies
    // page-cache (or NVMe) bytes straight into the pinned ring at 27-29 GB/s
    // aggregate — no mmap memcpy, no giant pinned tier. Any short/failed read
    // fails the whole batch and the fill path returns an error (no DMA of
    // garbage — Codex adversarial finding on the glm52 original).
    struct PreadPool {
        struct Job { int fd; uint64_t off, len; uint8_t* dst; };
        std::vector<std::thread> ws;
        std::mutex mu; std::condition_variable cv, cvd;
        const Job* jobs = nullptr; size_t njobs = 0, next = 0, done = 0;
        bool stop = false, fail = false;
        void start(int n);
        bool run(const Job* j, size_t n);   // blocking; false on any failure
        void shutdown();
        ~PreadPool() { shutdown(); }
    };
    PreadPool prp_;
    static constexpr uint32_t kPinRing = 8;
    uint8_t* pin_ring_[kPinRing] = {};
    sycl::event pin_ev_[kPinRing];
    uint64_t pin_sz_ = 0;
    uint32_t pin_idx_ = 0;
    // R9 prefetch: worker-owned staging (IE_HY4_PREFETCH) — never shared with
    // the inline decode fills.
    uint8_t* pf_pin_[2] = {};
    sycl::event pf_pin_ev_[2];
    uint32_t pf_pin_idx_ = 0;
    PreadPool pf_prp_;
    int32_t* pf_pred_pin_ = nullptr;     // pinned landing for the layer-ahead
                                         // prediction (collected end-of-call)
    // IE_HY4_PP_TILES: prefill grouped-tiles wave jobs (device + host staging)
    int32_t* pp_jobs_ = nullptr;
    std::vector<int32_t> pp_hjobs_;
    // IE_HY4_PP_STREAM: prefill layer-streaming — double-buffered whole-layer
    // bank loads (3 big sequential H2Ds) replace the per-miss cache stream.
    // Each buffer holds [gate | up | down] regions of one MoE layer.
    uint8_t* pp_stream_[2] = {};
    uint64_t pp_stream_sz_ = 0;
    int32_t pp_stream_layer_[2] = {-1, -1};
    // Last compute that READ this stream buffer. Next fill of the same
    // buffer waits on this — NOT on live compute of the other buffer
    // (that fence serialized H2D behind the other GPU's work; A-wait 16 s).
    sycl::event pp_stream_free_[2];
    // Stream-owned pinned bounce. 256 MiB pieces. Ring is sized at init to
    // the layer bank / piece so a layer never wraps (wrap was 18 s ring_wait
    // at depth 4 and 8). ~20 × 256 MiB ≈ 5 GiB/card.
    static constexpr uint64_t kPpPinSz = 256ull << 20;
    std::vector<uint8_t*> pp_pin_;
    std::vector<sycl::event> pp_pin_ev_;
    // Load runs on the fill worker; the future's event is the layer's last
    // piece H2D (get() blocks only until the worker has SUBMITTED it).
    std::shared_future<sycl::event> pp_stream_fut_[2];
    // Dedicated in-order transfer queue: miss H2Ds overlap compute; the
    // first slot-reading kernel per expert carries the fill event as a dep,
    // and fills depend on a compute-fence barrier (eviction safety — the
    // victim slot may be read by still-in-flight earlier kernels).
    std::unique_ptr<sycl::queue> copyq_;
    // Main-thread-only. copyq_ belongs to the fill worker during PP_STREAM;
    // a jobs H2D on copyq_ races the worker and, with wait(), sits behind
    // the next layer's DMA (GPU1 ring_wait 16 s).
    std::unique_ptr<sycl::queue> jobsq_;
    // Fill worker: runs ring-wait + pooled copy + H2D submit OFF the main
    // thread (measured 2026-08-28: the pooled copies were 17.4s of a 38.9s
    // 2-chunk prefill, host-blocking). Owns pin_ring_/pin_ev_/cpool_/copyq_
    // exclusively once started; the main thread's ensure() does slot
    // bookkeeping and hands over a request; consumers block on the future
    // only if they outrun the fill stream.
    struct FillReq {
        uint8_t* dst;
        const uint8_t *g, *u, *d2;
        uint64_t gsl, usl, dsl;
        int gfd = -1, ufd = -1, dfd = -1;      // pread addressing (else memcpy)
        uint64_t goff = 0, uoff = 0, doff = 0;
        bool bounce = true;    // pinned bounce (decode trickle) vs direct
                               // pageable H2D (prefill bulk — no CPU bytes,
                               // frees DDR for the DMA itself)
        bool pf = false;       // prefetch: use the pf-owned ring + pool (the
                               // shared ring/pools race the main thread's
                               // concurrent inline decode fills)
        bool stream = false;   // IE_HY4_PP_STREAM: whole-layer region load —
                               // g/u/d are full [E x slice] regions, chunked
                               // through the pin ring (pread -> pinned -> H2D)
        sycl::event fence;
        std::promise<sycl::event> done;
    };
    struct FillThread {
        std::thread th;
        std::mutex mu;
        std::condition_variable cv;
        std::deque<FillReq> reqs;
        bool stop = false;
        void start(Hyv4Model* m);
        std::shared_future<sycl::event> push(FillReq&& r);
        ~FillThread();
    };
    FillThread fill_;
    // Tiny persistent copy pool: a single-threaded memcpy tops out at the
    // same ~12 GB/s the pageable path already got; 3 workers + the caller
    // reach ~30+.
    struct CopyPool {
        std::vector<std::thread> th;
        std::mutex mu;
        std::condition_variable cv, cv_done;
        const uint8_t* src = nullptr;
        uint8_t* dst = nullptr;
        uint64_t n = 0;
        uint32_t seq = 0, done = 0, workers = 0;
        bool stop = false;
        void start(uint32_t nw);
        void copy(uint8_t* d, const void* s, uint64_t bytes);
        ~CopyPool();
    };
    CopyPool cpool_;
public:
    uint64_t ecache_hits = 0, ecache_misses = 0;   // perf counters
    // R9 shadow-router counting (IE_HY4_SHADOW_COUNT): layer-ahead proxy score.
    uint64_t shadow_hits_ = 0, shadow_total_ = 0;
    std::vector<int32_t> shadow_pred_;
    std::vector<uint8_t> shadow_has_;
    std::vector<uint64_t> shadow_lhits_, shadow_ltotal_;   // per-layer accuracy
    // depth-2 proxy scoring (counting only): predict L+2's routing from L's xn_.
    uint64_t shadow2_hits_ = 0, shadow2_total_ = 0;
    std::vector<int32_t> shadow2_pred_;
    std::vector<uint8_t> shadow2_has_;
    uint64_t pf_issued_ = 0, pf_used_ = 0;   // IE_HY4_PREFETCH diagnostics
    // Host wall-time buckets over run_moe (seconds; reset by caller).
    double t_ring_wait = 0, t_pool_copy = 0, t_fill_submit = 0, t_compute_submit = 0;
    // IE_HY4_PPS_SYNC probe (numbers-invalid): per-layer stream decomposition —
    // get = block until the layer's H2Ds are SUBMITTED, wait = until LANDED.
    double t_pps_get = 0, t_pps_wait = 0;
    bool traced_first_ = false;   // IE_HY4_TRACE_FIRST: one per-layer trace done
    // Expert-selection profile: counts[L * n_experts + e] over every routed
    // pick this model instance served (host-side, from the routing D2H the
    // MoE loop already does). Free to keep on; read by the priority tooling.
    std::vector<uint64_t> expert_counts;
    std::vector<uint64_t> miss_counts;   // per (layer, expert) CACHE MISSES — the host-tier ranking signal
    // IE_HY4_CPU_MISS: host Q8×Q4/Q5 path for decode misses (no 16 MiB H2D).
    std::vector<sycl::half> cpu_x16h_;
    std::vector<float> cpu_xf_, cpu_y_;
    std::vector<float> cpu_acc_ring_[2];
    float* cpu_acc_dev_[2] = {};
    uint32_t cpu_acc_sel_ = 0;
    double t_cpu_compute = 0, t_cpu_join = 0;   // q* CPU branch: worker time / main-thread wait
    uint64_t n_cpu_experts = 0;
    // q* CPU branch runs on ONE persistent worker thread so its OpenMP team
    // persists across layers (a std::async per layer re-created the team each
    // call: 1.7-3 ms/expert vs the bench's 0.25). Job = a std::function; the
    // worker owns cpu_x*/cpu_y_/cbuf while a job is in flight.
    struct CpuWorker {
        std::thread th;
        std::mutex mu;
        std::condition_variable cv;
        std::function<void()> job;
        bool has_job = false, done = true, quit = false;
        int core_lo = -1, core_hi = -1;   // affinity range (inherited by the OMP team)
        void start();
        void submit(std::function<void()> j);
        void wait();
        ~CpuWorker();
    } cpu_worker_;
private:
    // final
    float* mean_ = nullptr;                  // [MT, H]
    sycl::half* logits_ = nullptr;           // [vocab]
    sycl::half* lgs16_ = nullptr;            // [16, vocab] q8 lm_head row batch
    // MTP (tail stage; null when not loaded)
    Hyv4DW mtp_eh_;                          // eh_proj [2H -> H]
    float* mtp_enorm_ = nullptr;             // [H]
    float* mtp_hnorm_ = nullptr;             // [H]
    float* mtp_shn_ = nullptr;               // [H]
    sycl::half* mtp_lat_ = nullptr;          // [max_ctx, kv_lora] own latent cache
    sycl::half* mtp_e16_ = nullptr;          // [H] embed row
    sycl::half* mtp_cat_ = nullptr;          // [2H] fused input
    float* mtp_x_ = nullptr;                 // [H] running residual
    sycl::half* mtp_x16_ = nullptr;          // [H]
    uint32_t mtp_base_ = 0, mtp_len_ = 0;    // ingested suffix [base, len)
    // KDA snapshot (spec decode)
    float* snap_dn_ = nullptr;
    sycl::half* snap_conv_ = nullptr;
    uint32_t snap_kv_len_ = 0, snap_mtp_len_ = 0;
    // spec-verify input capture ([n_lin][SPECMAX][...], see commit_verify)
    static constexpr uint32_t kSpecMax = 8;
    bool spec_verify_ = false;
    uint32_t spec_T_ = 0, spec_pos_ = 0;
    float* sv_q_ = nullptr;                  // [n_lin, kSpecMax, DI] f32
    float* sv_k_ = nullptr;
    float* sv_v_ = nullptr;
    float* sv_g_ = nullptr;
    float* sv_beta_ = nullptr;               // [n_lin, kSpecMax, NH]
    sycl::half* sv_ci_ = nullptr;            // [n_lin, 3, kSpecMax, DI] conv ins
};

// Shipped router GEMV (fp32). T=1 uses SLM-staged x; T>1 is the global-x
// leaf. Same per-expert FMA lattice either way.
sycl::event hyv4_router_logits(sycl::queue& q, const float* x, const float* w,
                               float* rl, uint32_t T, uint32_t H, uint32_t E,
                               const std::vector<sycl::event>& deps = {});

}  // namespace ie
