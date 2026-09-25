// src/engine/engine.cpp — product-facing ie::Engine: load once, streaming
// generate/chat.  Load sequence mirrors tools/ie_bench.cpp; decode loop
// mirrors tools/forward_test.cpp.  Prefill is chunked at opts.prefill_chunk
// (hard cap 256 per docs/known_bugs.md — never submit larger T).
#include "ie/engine.hpp"
#include "ie/ds41_engine.hpp"
#include "ie/mimo26_engine.hpp"
#include "ie/ds41_vision.hpp"        // kDs41VisMaxTok (vision_status_json)
#include "../../third_party/nlohmann/json.hpp"
#include "ie/reasoning.hpp"
#include <future>
#include "ie/qwen4exp.hpp"
#include "ie/glm5next.hpp"
#include "ie/glm5_server.hpp"
#include "ie/server_capabilities.hpp"
#include <charconv>
#include <cmath>
#include "ie/deepseek4.hpp"
#include "ie/dspark_drafter.hpp"
#include "ie/memory_plan.hpp"
#include "ie/ops.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

namespace ie {

// DeepSeek-V4-Flash state. Opaque to engine.hpp (see the forward declaration
// there) so the whole deepseek4 stack — model, both runtimes, expert cache —
// stays out of every translation unit that only wants an Engine.
struct Q4eBundle {
    Qwen4ExpConfig  cfg;
    DeviceAllocator a1;          // second card (split mode only)
    Qwen4ExpModel   A, B;        // A: [0,24) or whole model; B: [24,48)
    bool            split = false;
    bool            p2p   = false;  // wide handoff pushes device-to-device
    std::vector<float> wide_b;   // second parity buffer (pipelined prefill)
    std::vector<float> wide;     // pipeline handoff [chunk, 4, hidden]
    // Agent-turn prefix reuse via the engine's generic prompt-cache flow:
    // snapshot at the stable conversation depth, restore when the next
    // prompt extends those tokens (all-or-nothing, like DS4 — the DeltaNet
    // state cannot be truncated).
    std::vector<int32_t> snap_tokens;

    void reset_all() {
        A.reset_state();
        if (split) B.reset_state();
    }
    std::string fwd(const int32_t* t, uint32_t T, uint32_t pos) {
        if (split) {
            std::string m = A.forward_range(t, T, pos, nullptr, wide.data(), nullptr);
            if (!m.empty()) return m;
            return B.forward_range(t, T, pos, p2p ? nullptr : wide.data(),
                                   nullptr, nullptr, /*wide_in_device=*/p2p);
        }
        return A.forward(t, T, pos, nullptr);
    }
    // Chunk-PIPELINED prefill over the split (run2's certified overlap: A
    // runs chunk k+1 while B runs chunk k; B is sequential with itself via
    // the future chain; double host wide buffers). P2P peer handoff is OFF
    // inside (single peer target would clobber B's in-flight chunk) and
    // restored after. 1024-token chunks — the run2-certified pipeline shape
    // (the serial path's big-chunk default trades misses for no overlap).
    // Caller owns the pos==0 reset semantics (mirrors fwd()).
    std::string fwd_pipelined(const int32_t* t, uint32_t count, uint32_t pos) {
        constexpr uint32_t PC = 2048;
        if (!split || count <= PC) return "q4e pipeline: not applicable";
        // P2P + pipelining now coexist via B's double wide banks: A pushes
        // chunk k into bank k&1 while B reads chunk k-1 from the other bank.
        // Bank alloc failure (or no P2P) falls back to the host double-bounce.
        // Bank guard includes B.max_chunk() >= PC: the peer push lands in B's
        // [max_chunk, 4, H] bank BEFORE B's own T guard could refuse it.
        const bool banks = p2p && B.max_chunk() >= PC &&
                           B.alloc_wide_bank2().empty();
        if (p2p && !banks) A.set_wide_peer(nullptr);
        const uint64_t wsz = uint64_t(PC) * cfg.hc_count * cfg.hidden;
        if (wide.size() < wsz) wide.resize(wsz);
        if (!banks && wide_b.size() < wsz) wide_b.resize(wsz);
        float* wb2[2] = {wide.data(), banks ? wide.data() : wide_b.data()};
        std::future<std::string> bfut;
        std::string err;
        uint32_t done = 0;
        for (uint32_t k = 0; done < count; ++k) {
            const uint32_t n = std::min(PC, count - done);
            float* wb = wb2[k & 1];
            if (banks) A.set_wide_peer(B.wide_bank(k & 1));
            if (auto e = A.forward_range(t + done, n, pos + done, nullptr,
                                         wb, nullptr); !e.empty()) {
                err = "A: " + e; break;
            }
            if (bfut.valid())
                if (auto e = bfut.get(); !e.empty()) { err = "B: " + e; break; }
            Qwen4ExpModel* Bp = &B;
            const uint32_t bn = n, bpos = pos + done, bank = k & 1;
            bfut = std::async(std::launch::async, [Bp, wb, bn, bpos, banks, bank]() {
                if (banks) Bp->use_wide_bank(bank);
                return Bp->forward_range(nullptr, bn, bpos,
                                         banks ? nullptr : wb, nullptr, nullptr,
                                         /*wide_in_device=*/banks);
            });
            done += n;
        }
        if (bfut.valid()) {
            const std::string e = bfut.get();
            if (err.empty() && !e.empty()) err = "B: " + e;
        }
        if (banks) B.use_wide_bank(0);
        if (p2p) A.set_wide_peer(B.wide_device());
        return err;
    }
    std::string snap(uint32_t depth) {
        std::string m = A.snapshot(depth);
        if (m.empty() && split) m = B.snapshot(depth);
        return m;
    }
    std::string rest() {
        std::string m = A.restore();
        if (m.empty() && split) m = B.restore();
        return m;
    }
};

struct Glm5Bundle {
    Glm5NextConfig cfg;
    DeviceAllocator a1;
    Glm5NextModel A, B;
    bool split = false;
    std::vector<float> wide;
    ~Glm5Bundle() { A.shutdown(); if (split) B.shutdown(); }
    std::string fwd(const int32_t* ids, uint32_t n, uint32_t pos) {
        if (!split) return A.forward(ids, n, pos, nullptr);
        if (auto m=A.forward_range(ids,n,pos,nullptr,wide.data(),nullptr); !m.empty()) return m;
        return B.forward_range(ids,n,pos,wide.data(),nullptr,nullptr);
    }
};

struct Ds4Bundle {
    DeepSeek4Config                     cfg{};
    DeepSeek4Model                      model;      // points into Engine::gguf_'s mmap
    std::unique_ptr<DeepSeek4Runtime>   rt;         // single card
    std::unique_ptr<DeepSeek4TpRuntime> tp;         // >1 card, driven in lockstep
    std::vector<float>                  logits;     // host fp32 [vocab], reused per step
    // Prompt cache (docs/deepseek4/72 Phase L3): N conversations' states in
    // PINNED HOST memory, one Ds4HostSnapshot per card, with the tokens each was
    // taken over.  Lookup = the deepest slot whose tokens are an exact prefix of
    // the request (all-or-nothing at that depth: a DS4 snapshot cannot be
    // truncated); insert = the slot this conversation already occupied (it grew
    // by a turn) or the least recently used one.  Declared after rt/tp so it is
    // destroyed first (its buffers are pinned on the runtimes' queues).
    struct Slot {
        std::vector<int32_t>         tokens;   // empty <=> no snapshot
        std::vector<Ds4HostSnapshot> cards;
        uint64_t                     stamp = 0;
    };
    std::vector<Slot> slots;
    uint64_t          slot_clock = 0;
};

// GPUs available to the deepseek4 runtimes, counted the way THEY index them:
// deepseek4.cpp builds its device list from LEVEL-ZERO GPUs only and
// Ds4Options::device_ordinal is an index into that list. count_matching_gpus()
// cannot stand in — it walks every backend, so each B70 is seen twice (once
// level_zero, once opencl) and it answers 4 on a two-card box. The name filter
// then drops the iGPU, which shares that list. $IE_GPU_FILTER overrides it, and
// $DS4_GPU / $DS4_TP_GPUS override the ordinals themselves.
static uint32_t ds4_count_gpus() {
    std::string_view filter = kGpuNameFilter;
    if (const char* f = std::getenv("IE_GPU_FILTER"); f && *f) filter = f;
    uint32_t n = 0;
    for (const auto& d : sycl::device::get_devices()) {
        if (!d.is_gpu() || d.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        if (filter.empty() ||
            d.get_info<sycl::info::device::name>().find(filter) != std::string::npos) ++n;
    }
    return n;
}

size_t utf8_complete_prefix_len(std::string_view s) {
    if (s.empty()) return 0;
    size_t i = s.size();
    size_t cont = 0;
    while (i > 0 && cont < 3 && (uint8_t(s[i - 1]) & 0xC0u) == 0x80u) { --i; ++cont; }
    if (i == 0) return 0;
    const uint8_t lead = uint8_t(s[i - 1]);
    size_t need = (lead & 0x80u) == 0 ? 1 : (lead & 0xE0u) == 0xC0u ? 2
                : (lead & 0xF0u) == 0xE0u ? 3 : (lead & 0xF8u) == 0xF0u ? 4 : 1;
    return (cont + 1 >= need) ? i - 1 + need : i - 1;
}

std::unique_ptr<Engine> Engine::load(const std::string& gguf_path,
                                     const EngineOptions& opts_in,
                                     std::string& err) {
    if (opts_in.max_ctx < 9 || (opts_in.slot_ctx && opts_in.slot_ctx < 9)) {
        err = "context and nonzero slot context must be at least 9 tokens";
        return nullptr;
    }
    auto e = std::unique_ptr<Engine>(new Engine());
    EngineOptions opts = opts_in;   // mutable working copy (the planner resolves n_gpus)
    e->opts_ = opts;
    e->model_path_ = gguf_path;
    if (ds41_dir(gguf_path)) {                       // DeepSeek-V4.1-Flash: a safetensors directory, its own runtime
        if (auto m = e->ds41_load(gguf_path); !m.empty()) { err = m; return nullptr; }
        return e;
    }
    if (mimo26_dir(gguf_path)) {                     // MiMo-V2.6: a safetensors directory, its own runtime
        if (auto m = e->mimo26_load(gguf_path); !m.empty()) { err = m; return nullptr; }
        return e;
    }
    if (auto m = e->gguf_.open(gguf_path); !m.empty())   { err = "gguf: " + m;  return nullptr; }

    // Arch dispatch (detected BEFORE the allocator so the qwen4exp 2-GPU
    // P2P path can join a shared context): qwen35moe → crown QwenModel path
    // (behavior unchanged), qwen3 → DenseModel (every layer full-attention;
    // no DeltaNetState).
    e->arch_ = detect_arch(e->gguf_);
    if (!server_supports_arch(e->arch_)) {
        const auto* a = e->gguf_.find_kv("general.architecture");
        err = "unsupported architecture: " +
              (a ? std::string(a->as_string()) : std::string("<missing>"));
        return nullptr;
    }
    // IE_P2P=1 + qwen4exp: both stage allocators join ONE shared context
    // with peer access — the A->B wide handoff then PUSHES device-to-device
    // (bit-eq certified by ie-qwen4exp-run2's gate) instead of bouncing
    // through host. The sycl handles are refcounted, so the queues keep the
    // shared context alive; no storage needed here.
    std::vector<sycl::device> q4e_p2p_gpus;
    std::unique_ptr<sycl::context> q4e_p2p_ctx;
    const bool q4e_p2p = e->arch_ == ModelArch::kQwen4Exp &&
                         std::getenv("IE_P2P") != nullptr &&
                         gpu_p2p_shared_context("B70", q4e_p2p_gpus, q4e_p2p_ctx);
    if (q4e_p2p) {
        if (auto m = e->alloc_.init_with(*q4e_p2p_ctx, q4e_p2p_gpus[0]); !m.empty()) {
            err = "alloc: " + m; return nullptr;
        }
    } else if (auto m = e->alloc_.init(); !m.empty())   { err = "alloc: " + m; return nullptr; }
    e->tp_logits_host_.bind(e->alloc_.queue().get_context());   // pinned logits staging

    // VRAM-aware placement: when n_gpus==0 (auto, the CLI default) sense per-card
    // VRAM and pick single vs multi-GPU; an explicit --gpus N is respected. Only
    // arches with a working >1-GPU path may be split (gemma4/qwen3moe/crown are
    // single-GPU today). Resolves opts.n_gpus to a concrete >=1 for the branches.
    if (e->arch_ == ModelArch::kGlm5Next) {
        if (e->alloc_.device().get_backend() != sycl::backend::ext_oneapi_level_zero) {
            err="glm5next requires Level Zero devices; set ONEAPI_DEVICE_SELECTOR=level_zero:gpu"; return nullptr;
        }
        const uint32_t avail = ds4_count_gpus();
        if (!avail || opts.n_gpus > std::min(avail,2u)) {
            err = "glm5next requires one or two available physical Level Zero GPUs"; return nullptr;
        }
        if (!opts.n_gpus) opts.n_gpus = std::min(avail,2u);
        e->opts_.n_gpus = opts.n_gpus;
        if (opts.int8_kv || opts.spec || !opts.spec_draft.empty() || !opts.spec_head.empty()) {
            err = "glm5next server does not support int8 KV or speculative decoding"; return nullptr;
        }
        // These research switches add unbudgeted allocations or require a
        // different executor. Refuse them rather than inherit runner state.
        for (const char* key : {"IE_G5_MTP","IE_G5_EP_DECODE","IE_G5_PP_STREAM","IE_G5_ECACHE_PAD"}) {
            const char* v=std::getenv(key);
            if (v) {
                err=std::string("glm5next server does not support ")+key+"; unset it before serving"; return nullptr;
            }
        }
        std::fprintf(stderr,"[mem-plan] glm5next: %u physical GPU(s); host expert streaming, per-stage resident/context/cache guard\n", opts.n_gpus);
    } else if (e->arch_ == ModelArch::kQwen4Exp) {
        // Flash-Next plans its own placement (per-card dense halves + VRAM
        // expert caches sized adaptively inside init_runtime); the generic
        // planner would sum all 103 GiB against per-card VRAM and refuse.
        const uint32_t avail = std::max<uint32_t>(1u, ds4_count_gpus());
        if (opts.n_gpus == 0) opts.n_gpus = std::min(avail, 2u);
        opts.n_gpus     = std::min(std::max<uint32_t>(1u, opts.n_gpus), std::min(avail, 2u));
        e->opts_.n_gpus = opts.n_gpus;
        std::fprintf(stderr, "[mem-plan] qwen4exp: engine planner bypassed (routed experts "
                             "stream through per-card VRAM caches); %u of %u GPU(s)\n",
                     opts.n_gpus, avail);
    } else if (e->arch_ == ModelArch::kDeepSeek4) {
        // The generic planner CANNOT plan this arch and would refuse every load:
        // it sums every GGUF tensor (151 GB for UD-Q8_K_XL) against per-card
        // VRAM, but deepseek4 is designed to hold only the always-resident set
        // plus a routed-expert slot cache in VRAM and STREAM the rest out of a
        // pinned host arena. Its own planner (ds4_plan_residency_tp, run inside
        // DeepSeek4Runtime::load) does the real per-card arithmetic and refuses
        // an over-cap plan by name. So: resolve the card count here and let the
        // arch's own planner own the fit decision.
        const uint32_t avail = std::max<uint32_t>(1u, ds4_count_gpus());
        if (opts.n_gpus == 0) opts.n_gpus = avail;               // auto → every card
        opts.n_gpus     = std::min(std::max<uint32_t>(1u, opts.n_gpus), avail);
        e->opts_.n_gpus = opts.n_gpus;
        std::fprintf(stderr, "[mem-plan] deepseek4: engine planner bypassed (routed experts "
                             "stream from pinned host RAM); %u of %u GPU(s)\n",
                     opts.n_gpus, avail);
    } else {
        const bool mgpu = arch_can_multigpu(e->arch_);
        PlacementPlan plan = plan_placement(e->gguf_, e->arch_, opts.n_gpus,
                                            opts.max_ctx, opts.int8_kv, mgpu,
                                            e->alloc_.device());
        std::fprintf(stderr, "[mem-plan] %s\n", plan.note.c_str());
        opts.n_gpus     = plan.n_gpus;
        e->opts_.n_gpus = plan.n_gpus;
        // Hard gate: when the planner is confident the footprint won't fit even
        // after splitting (and the user neither forced --gpus nor set the
        // override), refuse here with an actionable message instead of crashing
        // deep in an OOM during weight upload. IE_ALLOW_OOM=1 attempts anyway.
        if (should_block_oom(plan, std::getenv("IE_ALLOW_OOM") != nullptr)) {
            const uint64_t need = plan.weights_bytes + plan.kv_bytes + plan.ws_bytes;
            err = "won't fit in VRAM: needs ~" + human_gb(need) + " but have " +
                  human_gb(plan.per_card_bytes) + "/card x " +
                  std::to_string(plan.avail_gpus) + " GPU(s). Options: use a smaller "
                  "quant, add --gpus N (if the arch splits), or lower --ctx. "
                  "Set IE_ALLOW_OOM=1 to attempt anyway.";
            return nullptr;
        }
    }
    uint32_t vocab = 0;
    if (is_dense_arch(e->arch_)) {
        // Reads <arch>.* for the actual arch ("qwen3" or "qwen2" — both ride this
        // dense path; the forward gates QK-norm vs attention-bias on presence).
        if (auto m = read_dense_config_auto(e->gguf_, e->dcfg_); !m.empty()) { err = "config: " + m; return nullptr; }
        if (opts.n_gpus > 1) {
            // Multi-GPU tensor-parallel: split every layer across n_gpus cards.
            // The single-GPU dense_/kv_ are NOT loaded (the model may not fit one
            // card). int8 KV is unsupported on the TP path (fp16 only).
            if (auto m = e->fleet_.init(opts.n_gpus); !m.empty()) { err = "fleet: " + m; return nullptr; }
            if (e->fleet_.size() < 2) { err = "tensor-parallel needs >=2 GPUs (have " +
                                              std::to_string(e->fleet_.size()) + ")"; return nullptr; }
            if (auto m = e->tp_model_.load(e->fleet_, e->gguf_, e->dcfg_, opts.max_ctx); !m.empty())
                { err = "tp model: " + m; return nullptr; }
            e->tp_ = true;
            e->tp_ids_host_.resize(opts.prefill_chunk);
            e->tp_logits_host_.resize(e->dcfg_.vocab);
        } else {
            if (auto m = e->dense_.load(e->alloc_, e->gguf_, e->dcfg_); !m.empty())  { err = "model: " + m;  return nullptr; }
            KvCacheConfig kvcfg{};
            kvcfg.n_layers_full = e->dcfg_.n_layers;     // all layers full-attention
            kvcfg.n_kv_heads    = e->dcfg_.n_kv_heads;
            kvcfg.max_ctx       = opts.max_ctx;
            kvcfg.head_dim      = e->dcfg_.head_dim;
            kvcfg.use_int8      = opts.int8_kv;
            if (auto m = e->kv_.init(e->alloc_, kvcfg); !m.empty()) { err = "kv: " + m; return nullptr; }
            // dn_ deliberately NOT inited — dense arch has no linear-attn layers;
            // DeltaNetState::reset() on the uninited state is a safe no-op.
            if (auto m = e->dense_.ensure_workspace(opts.prefill_chunk); !m.empty()) { err = "ws: " + m; return nullptr; }
            if (auto m = e->dense_.ensure_attn_partials(opts.max_ctx); !m.empty())  { err = "partials: " + m; return nullptr; }
        }
        vocab = e->dcfg_.vocab;
    } else if (e->arch_ == ModelArch::kQwen35Dense) {
        // qwen35 dense-hybrid (Qwen3.6-27B): gated-DeltaNet linear layers +
        // gated full-attn every full_attn_interval-th layer + dense SwiGLU.
        // Hybrid caches like the crown: KV holds the full-attn layers, the
        // DeltaNet state holds the linear ones.
        Qwen35Config qcfg;
        if (auto m = read_qwen35_config(e->gguf_, qcfg); !m.empty()) { err = "config: " + m; return nullptr; }
        // Q8_0-packed path (Qwen35SplitModel). --gpus>1 → layer-split across cards;
        // --gpus 1 + IE_QWEN35_Q8=1 → SINGLE-GPU Q8 (fits ~28.6 GB on one 32 GB card,
        // no cross-card serial tax → faster decode + big context since the 27B's KV
        // is tiny). The plain --gpus 1 default stays on the F16 Qwen35DenseModel (for
        // non-Q8 27B GGUFs that would otherwise need this packing).
        // TENSOR-PARALLEL path: IE_QWEN35_TP=1 + --gpus>1 → both cards on the same
        // token (Megatron). Mutually exclusive with the layer-split Q8 path below.
        const bool use_tp = opts.n_gpus > 1 && std::getenv("IE_QWEN35_TP");
        const bool use_split = !use_tp && (opts.n_gpus > 1 || std::getenv("IE_QWEN35_Q8"));
        if (use_tp) {
            // P2P transfers are OFF everywhere (2026-08-16): on this client bridge
            // peer READS return garbage (standalone test: 100% corrupt) and peer
            // writes proved unreliable under load — the 2026-08-15 "TP 20.8" was
            // measured on corrupted output and is retracted. Host-staged TP is the
            // real path: 19.11 plain / 22.70 with --spec (correct text, tau~2).
            // The P2P machinery stays behind enable_p2p_transfers for future
            // hardware; IE_NO_P2P=1 remains the global kill switch.
            if (auto m = e->fleet_.init(opts.n_gpus, "B70", /*enable_p2p_transfers=*/false); !m.empty()) { err = "fleet: " + m; return nullptr; }
            if (e->fleet_.size() < 2) { err = "tensor-parallel needs >=2 GPUs (have " +
                                              std::to_string(e->fleet_.size()) + ")"; return nullptr; }
            if (auto m = e->qwen35_tp_model_.load(e->fleet_, e->gguf_, qcfg, opts.max_ctx); !m.empty())
                { err = "tp model: " + m; return nullptr; }
            e->qwen35_tp_ = true;
            e->tp_ids_host_.resize(std::max<uint32_t>(opts.prefill_chunk, 1u));
            e->tp_logits_host_.resize(qcfg.dense.vocab);
            if (opts.spec) {
                if (auto m = e->qwen35_tp_model_.load_mtp_head(e->gguf_, opts.max_ctx);
                    !m.empty())
                    std::fprintf(stderr,
                        "[spec] TP MTP head load failed (continuing without): %s\n",
                        m.c_str());
                else
                    e->spec_ = true;
            }
            vocab = qcfg.dense.vocab;
        } else if (use_split) {
            const uint32_t ndev = opts.n_gpus > 1 ? opts.n_gpus : 1;
            if (auto m = e->fleet_.init(ndev, "B70", false, /*shared_ctx=*/false); !m.empty()) { err = "fleet: " + m; return nullptr; }
            if (ndev > 1 && e->fleet_.size() < 2) { err = "layer-split needs >=2 GPUs (have " +
                                              std::to_string(e->fleet_.size()) + ")"; return nullptr; }
            const uint32_t n_tx = qcfg.n_transformer_layers();
            LayerPlan plan = LayerPlan::contiguous(n_tx, ndev);
            if (auto m = e->qwen35_split_model_.load(e->fleet_, plan, e->gguf_, qcfg, opts.max_ctx, opts.int8_kv); !m.empty())
                { err = "model: " + m; return nullptr; }
            e->qwen35_split_ = true;
            e->tp_ids_host_.resize(std::max<uint32_t>(opts.prefill_chunk, 1u));
            e->tp_logits_host_.resize(qcfg.dense.vocab);
            // --int8-kv coexists with the prompt cache: copy_prefix_from snapshots the
            // int8 shadow + scales and regenerates fp16 on restore (int8 KV snapshot).
            if ((opts.prompt_cache || std::getenv("IE_PROMPT_CACHE")) && !std::getenv("IE_NO_PROMPT_CACHE")) {
                FleetPrefixCacheConfig pcfg;
                pcfg.max_prefix_len = opts.max_ctx;
                if (const char* s = std::getenv("IE_PROMPT_CACHE_MAX_PREFIX"))
                    if (unsigned v = unsigned(std::atoi(s))) pcfg.max_prefix_len = v;
                // Entry cap (IE_PROMPT_CACHE_MAX_ENTRIES): the cache's VRAM
                // worst case is max_entries × a full-depth endpoint. At long
                // --ctx that unbounded product is what overcommitted VRAM and
                // took the GPU driver (and the desktop) down on 2026-08-26 —
                // the load-time budget check below now accounts for it.
                if (const char* s = std::getenv("IE_PROMPT_CACHE_MAX_ENTRIES"))
                    if (unsigned v = unsigned(std::atoi(s))) pcfg.max_entries = v;
                e->split_cache_cfg_ = pcfg;
                if (auto m = e->fleet_cache_.init(e->qwen35_split_model_, pcfg); !m.empty()) {
                    err = "27B-split prompt-cache: " + m; return nullptr;
                }
                e->prompt_cache_on_ = true;
            }
            // --spec on the split: NextN/MTP head onto head_dev (+~0.6 GB there).
            if (opts.spec) {
                if (auto m = e->qwen35_split_model_.load_mtp_head(e->gguf_, opts.max_ctx); !m.empty())
                    std::fprintf(stderr,
                        "[spec] split MTP head load failed (continuing without): %s\n",
                        m.c_str());
                else
                    e->spec_ = true;
            }
            // --- Load-time VRAM budget check (added after the 2026-08-26
            // desktop crash). This stack has NO free-VRAM query and driver
            // OOM at run time is NOT graceful — it killed the xe driver and
            // rebooted the machine. So the worst case is computed HERE, from
            // arithmetic, and an over-committed config is refused with the
            // numbers instead of loaded on hope. Worst case per card =
            // weights + live KV @max_ctx + DeltaNet + decode banks + prompt
            // cache at its configured caps + a flat workspace reserve. ---
            {
                auto& model = e->qwen35_split_model_;
                const uint32_t n_par = std::max<uint32_t>(opts.parallel, 1u);
                uint32_t sctx = opts.slot_ctx ? opts.slot_ctx : 65536u;
                sctx = std::min<uint32_t>(sctx, opts.max_ctx);
                const std::vector<uint64_t> wbytes = model.device_bytes();
                bool over = false;
                std::string detail;
                for (uint32_t d = 0; d < e->fleet_.size(); ++d) {
                    const uint64_t gmem = e->fleet_.dev(d).device()
                        .get_info<sycl::info::device::global_mem_size>();
                    uint64_t kv_live = 0, kv_slot = 0, dnb = 0;
                    if (model.dev_has_kv(d)) {
                        const auto& kc = model.kv_cache(d).config();
                        const uint64_t per_tok = uint64_t(kc.n_layers_full) *
                            kc.n_kv_heads * kc.head_dim * 2 /*K+V*/ * 2 /*fp16*/;
                        kv_live = per_tok * kc.max_ctx;
                        kv_slot = per_tok * sctx;
                    }
                    if (model.dev_has_dn(d)) {
                        const auto& dn = model.dn_state(d);
                        dnb = dn.state_elems_per_layer() * dn.config().n_layers_linear * 4 +
                              dn.conv_elems_per_layer()  * dn.config().n_layers_linear * 2;
                    }
                    const uint64_t banks = (n_par > 1) ? n_par * (kv_slot + dnb) : 0;
                    uint64_t cache = 0;
                    if (e->prompt_cache_on_) {
                        const uint64_t ep_tok = std::min<uint64_t>(
                            e->split_cache_cfg_.max_prefix_len, opts.max_ctx);
                        uint64_t per_tok = 0;
                        if (model.dev_has_kv(d)) {
                            const auto& kc = model.kv_cache(d).config();
                            per_tok = uint64_t(kc.n_layers_full) * kc.n_kv_heads *
                                      kc.head_dim * 2 * 2;
                        }
                        cache = uint64_t(e->split_cache_cfg_.max_entries) *
                                (per_tok * ep_tok + dnb);
                    }
                    const uint64_t reserve = uint64_t(2) << 30;   // workspaces/partials/logits
                    const uint64_t total = (d < wbytes.size() ? wbytes[d] : 0) +
                                           kv_live + dnb + banks + cache + reserve;
                    char line[256];
                    std::snprintf(line, sizeof line,
                        "[budget] card %u: weights %.1f + kv %.1f + banks %.1f + "
                        "cache-max %.1f + reserve 2.0 = %.1f GB of %.1f GB\n",
                        d, (d < wbytes.size() ? wbytes[d] : 0) / 1e9, kv_live / 1e9,
                        banks / 1e9, cache / 1e9, total / 1e9, gmem / 1e9);
                    std::fputs(line, stderr);
                    detail += line;
                    if (double(total) > 0.92 * double(gmem)) over = true;
                }
                if (over) {
                    err = "VRAM budget: worst case exceeds 92% of a card — refusing "
                          "to load (this exact overcommit crashed the machine on "
                          "2026-08-26). Lower --ctx, --parallel, --slot-ctx, "
                          "IE_PROMPT_CACHE_MAX_PREFIX, or IE_PROMPT_CACHE_MAX_ENTRIES.\n" +
                          detail;
                    return nullptr;
                }
            }
            // --parallel>1: allocate the per-slot decode banks and spawn the
            // joint-step scheduler (Phase 2b). Bank alloc failure is a clean
            // load-time refusal — shrink --slot-ctx or --parallel to fit.
            if (opts.parallel > 1) {
                uint32_t sctx = opts.slot_ctx ? opts.slot_ctx : 65536u;
                sctx = std::min<uint32_t>(sctx, opts.max_ctx);
                if (auto m = e->qwen35_split_model_.alloc_slot_banks(opts.parallel, sctx);
                    !m.empty()) {
                    err = "slot banks (--parallel " + std::to_string(opts.parallel) +
                          ", slot-ctx " + std::to_string(sctx) + "): " + m;
                    return nullptr;
                }
                std::fprintf(stderr,
                    "[batch] %u decode slot banks @ ctx %u (joint-step scheduler on)\n",
                    opts.parallel, sctx);
                e->stepper_spawn_();
            }
            vocab = qcfg.dense.vocab;
        } else {
        if (auto m = e->qwen35_.load(e->alloc_, e->gguf_, qcfg); !m.empty()) { err = "model: " + m; return nullptr; }
        const uint32_t n_tx   = qcfg.n_transformer_layers();          // 64 (skips NextN)
        const uint32_t L_full = n_tx / qcfg.full_attn_interval;       // 16
        const uint32_t L_lin  = n_tx - L_full;                        // 48
        {
            KvCacheConfig kvcfg{};
            kvcfg.n_layers_full = L_full;
            kvcfg.n_kv_heads    = qcfg.dense.n_kv_heads;
            kvcfg.max_ctx       = opts.max_ctx;
            kvcfg.head_dim      = qcfg.dense.head_dim;
            kvcfg.use_int8      = opts.int8_kv;
            if (auto m = e->kv_.init(e->alloc_, kvcfg); !m.empty()) { err = "kv: " + m; return nullptr; }
        }
        // conv_channels computed DIRECTLY (R1: never ssm_inner*2 for this arch).
        const uint32_t conv_ch = qcfg.ssm_inner + 2u * qcfg.ssm_n_k_heads * qcfg.ssm_state;  // 10240
        if (auto m = e->dn_.init(e->alloc_, DeltaNetStateConfig{
                L_lin, qcfg.ssm_n_v_heads, qcfg.ssm_state, qcfg.ssm_state,
                conv_ch, qcfg.ssm_conv_kernel}); !m.empty()) {
            err = "dn: " + m; return nullptr;
        }
        if (auto m = e->qwen35_.ensure_workspace(opts.prefill_chunk); !m.empty()) { err = "ws: " + m; return nullptr; }
        if (auto m = e->qwen35_.ensure_attn_partials(opts.max_ctx); !m.empty())  { err = "partials: " + m; return nullptr; }
        // MTP self-speculative decode (single-GPU kQwen35Dense, --spec). Loads the
        // native MTP/NextN head (~0.6 GB) only when requested → default memory
        // unchanged. Greedy-only; generate() falls back to plain decode otherwise.
        if (opts.spec) {
            if (auto m = e->qwen35_.load_mtp_head(e->gguf_, opts.max_ctx); !m.empty()) {
                err = "mtp head: " + m; return nullptr;
            }
            e->spec_ = true;
        }
        // dspark separate-draft speculative decode (single-GPU kQwen35Dense,
        // --spec-draft <dspark.gguf>). Loads a target-conditioned DsparkDrafter
        // (mirrors the gemma4 --spec-head precedent) and tells the target which
        // layer outputs to export as conditioning taps. Greedy-only; generate()
        // falls back to plain decode for temperature > 0. Mutually independent of
        // --spec (native MTP): --spec-draft takes precedence in generate().
        if (!opts.spec_draft.empty()) {
            GgufReader draft_g;
            if (auto m = draft_g.open(opts.spec_draft); !m.empty()) {
                err = "spec-draft gguf: " + m; return nullptr;
            }
            e->dspark_drafter_ = std::make_unique<DsparkDrafter>();
            // Cumulative context can reach the full sequence length (recompute-all
            // M1 drafter), so size the drafter scratch for max_ctx rows.
            if (auto m = e->dspark_drafter_->load(e->alloc_, draft_g, opts.max_ctx);
                !m.empty()) { err = "dspark drafter: " + m; return nullptr; }
            if (e->dspark_drafter_->vocab != qcfg.dense.vocab) {
                err = "spec-draft: drafter/target vocab mismatch"; return nullptr;
            }
            if (e->dspark_drafter_->H != qcfg.dense.hidden) {
                err = "spec-draft: drafter/target hidden-size mismatch"; return nullptr;
            }
            e->qwen35_.set_tap_layers(e->dspark_drafter_->target_layers);
            e->spec_draft_ = true;
            std::fprintf(stderr, "[spec] dspark drafter: %s (block_size=%u, taps=%zu)\n",
                         opts.spec_draft.c_str(), e->dspark_drafter_->block_size,
                         e->dspark_drafter_->target_layers.size());
        }
        vocab = qcfg.dense.vocab;
        }
    } else if (e->arch_ == ModelArch::kQwen3Moe) {
        // Qwen3 standard MoE (Qwen3-Coder-30B-A3B): dense QK-norm attention +
        // top-k MoE FFN. Every layer full-attn (KV for all layers); no DeltaNet.
        Qwen3MoeConfig q3mcfg;
        if (auto m = read_qwen3moe_config(e->gguf_, q3mcfg); !m.empty()) { err = "config: " + m; return nullptr; }
        // TENSOR-PARALLEL opt-in (IE_QWEN3MOE_TP=1 + --gpus>1): head-shard the
        // attention so BOTH cards compute it concurrently over half the heads +
        // half the KV (the long-ctx decode lever, where the layer-split is bound by
        // attention-at-depth at single-card parity) + expert-shard the MoE. Mirrors
        // the crown IE_QWEN35_TP opt-in; the DEFAULT stays the layer-split below.
        const bool q3moe_tp = opts.n_gpus > 1 && std::getenv("IE_QWEN3MOE_TP");
        if (q3moe_tp) {
            if (auto m = e->fleet_.init(opts.n_gpus); !m.empty()) { err = "fleet: " + m; return nullptr; }
            if (e->fleet_.size() < 2) { err = "tensor-parallel needs >=2 GPUs (have " +
                                              std::to_string(e->fleet_.size()) + ")"; return nullptr; }
            if (auto m = e->q3moe_tp_model_.load(e->fleet_, e->gguf_, q3mcfg, opts.max_ctx, opts.int8_kv); !m.empty())
                { err = "q3moe-tp model: " + m; return nullptr; }
            e->q3moe_tp_ = true;
            e->tp_ids_host_.resize(std::max<uint32_t>(opts.prefill_chunk, 1u));
            e->tp_logits_host_.resize(q3mcfg.dense.vocab);
            if ((opts.prompt_cache || std::getenv("IE_PROMPT_CACHE")) && !std::getenv("IE_NO_PROMPT_CACHE")) {
                FleetPrefixCacheConfig pcfg;
                pcfg.max_prefix_len = opts.max_ctx;
                if (const char* s = std::getenv("IE_PROMPT_CACHE_MAX_PREFIX"))
                    if (unsigned v = unsigned(std::atoi(s))) pcfg.max_prefix_len = v;
                if (auto m = e->fleet_cache_.init(e->q3moe_tp_model_, pcfg); !m.empty()) {
                    err = "q3moe-tp prompt-cache: " + m; return nullptr;
                }
                e->prompt_cache_on_ = true;
            }
        } else if (opts.n_gpus > 1) {
            // Multi-GPU LAYER-SPLIT (kQwen3Moe): per-card weights + full-attn KV,
            // host-logits bounce like the crown split. Splits the 130k KV across
            // cards. The single-GPU q3moe_ is NOT loaded on this path.
            if (auto m = e->fleet_.init(opts.n_gpus, "B70", false, /*shared_ctx=*/false); !m.empty()) { err = "fleet: " + m; return nullptr; }
            if (e->fleet_.size() < 2) { err = "layer-split needs >=2 GPUs (have " +
                                              std::to_string(e->fleet_.size()) + ")"; return nullptr; }
            LayerPlan plan = LayerPlan::contiguous(q3mcfg.dense.n_layers, opts.n_gpus);
            if (auto m = e->q3moe_split_model_.load(e->fleet_, plan, e->gguf_, q3mcfg, opts.max_ctx, opts.int8_kv); !m.empty())
                { err = "model: " + m; return nullptr; }
            e->q3moe_split_ = true;
            e->tp_ids_host_.resize(std::max<uint32_t>(opts.prefill_chunk, 1u));
            e->tp_logits_host_.resize(q3mcfg.dense.vocab);
            // --int8-kv coexists with the prompt cache: copy_prefix_from snapshots the
            // int8 shadow + scales and regenerates fp16 on restore (int8 KV snapshot).
            if ((opts.prompt_cache || std::getenv("IE_PROMPT_CACHE")) && !std::getenv("IE_NO_PROMPT_CACHE")) {
                FleetPrefixCacheConfig pcfg;
                pcfg.max_prefix_len = opts.max_ctx;
                if (const char* s = std::getenv("IE_PROMPT_CACHE_MAX_PREFIX"))
                    if (unsigned v = unsigned(std::atoi(s))) pcfg.max_prefix_len = v;
                if (auto m = e->fleet_cache_.init(e->q3moe_split_model_, pcfg); !m.empty()) {
                    err = "q3moe-split prompt-cache: " + m; return nullptr;
                }
                e->prompt_cache_on_ = true;
            }
        } else {
            if (auto m = e->q3moe_.load(e->alloc_, e->gguf_, q3mcfg); !m.empty()) { err = "model: " + m; return nullptr; }
            KvCacheConfig kvcfg{};
            kvcfg.n_layers_full = q3mcfg.dense.n_layers;
            kvcfg.n_kv_heads    = q3mcfg.dense.n_kv_heads;
            kvcfg.max_ctx       = opts.max_ctx;
            kvcfg.head_dim      = q3mcfg.dense.head_dim;
            kvcfg.use_int8      = opts.int8_kv;
            if (auto m = e->kv_.init(e->alloc_, kvcfg); !m.empty()) { err = "kv: " + m; return nullptr; }
            if (auto m = e->q3moe_.ensure_workspace(opts.prefill_chunk); !m.empty()) { err = "ws: " + m; return nullptr; }
        }
        vocab = q3mcfg.dense.vocab;
    } else if (e->arch_ == ModelArch::kGptOss) {
        // gpt-oss (OpenAI MoE, 20b/120b): GQA attention with per-head sinks +
        // alternating SWA(even)/full(odd) + top-4 MXFP4 MoE. Every layer carries
        // full KV (n_kv_heads=8, head_dim=64); SWA is a softmax-mask applied inside
        // attention, not a cache-geometry concern → KV is full like q3moe.
        GptOssConfig gcfg;
        if (auto m = read_gptoss_config(e->gguf_, gcfg); !m.empty()) { err = "config: " + m; return nullptr; }
        if (opts.n_gpus > 1) {
            // Multi-GPU TENSOR-PARALLEL (the 120b lever): MXFP4 experts sharded
            // across the fleet, attention replicated (Phase 1). Host-bounce like
            // the other TP paths; int8-KV unsupported (fp16 all-reduce only).
            if (auto m = e->fleet_.init(opts.n_gpus); !m.empty()) { err = "fleet: " + m; return nullptr; }
            if (e->fleet_.size() < 2) { err = "tensor-parallel needs >=2 GPUs (have " +
                                              std::to_string(e->fleet_.size()) + ")"; return nullptr; }
            if (auto m = e->gptoss_tp_model_.load(e->fleet_, e->gguf_, gcfg, opts.max_ctx); !m.empty())
                { err = "gptoss-tp model: " + m; return nullptr; }
            e->gptoss_tp_ = true;
            e->tp_ids_host_.resize(std::max<uint32_t>(opts.prefill_chunk, 1u));
            e->tp_logits_host_.resize(gcfg.dense.vocab);
            // Fleet prompt/KV cache for gpt-oss TP (council's second model) — was
            // re-prefilling the whole convo every turn. Same FleetPrefixCache as the
            // 80B/crown-split paths; gpt-oss is KV-only (no DeltaNet). Default-ON,
            // opt-out IE_NO_PROMPT_CACHE. Cap at max_ctx; oversize snapshot fails
            // alloc → skipped gracefully.
            if ((opts.prompt_cache || std::getenv("IE_PROMPT_CACHE")) && !std::getenv("IE_NO_PROMPT_CACHE")) {
                FleetPrefixCacheConfig pcfg;
                pcfg.max_prefix_len = opts.max_ctx;
                if (const char* s = std::getenv("IE_PROMPT_CACHE_MAX_PREFIX"))
                    if (unsigned v = unsigned(std::atoi(s))) pcfg.max_prefix_len = v;
                if (auto m = e->fleet_cache_.init(e->gptoss_tp_model_, pcfg); !m.empty()) {
                    err = "gptoss-tp prompt-cache: " + m; return nullptr;
                }
                e->prompt_cache_on_ = true;
            }
        } else {
            if (auto m = e->gptoss_.load(e->alloc_, e->gguf_, gcfg); !m.empty()) { err = "model: " + m; return nullptr; }
            KvCacheConfig kvcfg{};
            kvcfg.n_layers_full = gcfg.dense.n_layers;
            kvcfg.n_kv_heads    = gcfg.dense.n_kv_heads;
            kvcfg.max_ctx       = opts.max_ctx;
            kvcfg.head_dim      = gcfg.dense.head_dim;
            kvcfg.use_int8      = opts.int8_kv;
            if (auto m = e->kv_.init(e->alloc_, kvcfg); !m.empty()) { err = "kv: " + m; return nullptr; }
            if (auto m = e->gptoss_.ensure_workspace(opts.prefill_chunk); !m.empty()) { err = "ws: " + m; return nullptr; }
            if (auto m = e->gptoss_.ensure_attn_partials(opts.max_ctx); !m.empty()) { err = "partials: " + m; return nullptr; }
        }
        vocab = gcfg.dense.vocab;
    } else if (e->arch_ == ModelArch::kQwen3Next) {
        // Qwen3-Next-80B (hybrid gated-DeltaNet + gated full-attn + 512-expert MoE
        // + shared expert). 46 GB Q4 → MUST layer-split across the fleet (won't fit
        // one 32 GB card). Like the TP path: the model owns the split weights +
        // per-card KV/DeltaNet, forward bounces host logits into d_logits_.
        Qwen3NextConfig ncfg;
        if (auto m = read_qwen3next_config(e->gguf_, ncfg); !m.empty()) { err = "config: " + m; return nullptr; }
        const uint32_t n_gpus = opts.n_gpus ? opts.n_gpus : 1;
        if (auto m = e->fleet_.init(n_gpus, "B70", false, /*shared_ctx=*/false); !m.empty()) { err = "fleet: " + m; return nullptr; }
        const uint32_t n_layers = ncfg.hybrid.n_transformer_layers();
        LayerPlan plan = LayerPlan::contiguous(n_layers, n_gpus);
        if (auto m = e->next_model_.load(e->fleet_, plan, e->gguf_, ncfg, opts.max_ctx, opts.int8_kv); !m.empty())
            { err = "qwen3next model: " + m; return nullptr; }
        e->next_ = true;
        e->tp_ids_host_.resize(std::max<uint32_t>(opts.prefill_chunk, 1u));
        e->tp_logits_host_.resize(ncfg.hybrid.dense.vocab);
        // Fleet prompt/KV cache (default-ON; opt out IE_NO_PROMPT_CACHE → byte-identical
        // OFF path). Reuses prompt_cache_on_; generate() next_ branch restores per-card state.
        if ((opts.prompt_cache || std::getenv("IE_PROMPT_CACHE")) && !std::getenv("IE_NO_PROMPT_CACHE")) {
            FleetPrefixCacheConfig pcfg;
            // Cap at the model's context, not the 8192 default: any conversation
            // past 8k tokens was rejected by insert() → never cached → every turn
            // re-prefilled the WHOLE context ("0 cached" in the [gen] log). Snapshots
            // are depth-sized (insert sets kc.max_ctx=N) and a too-big snapshot fails
            // alloc → skips gracefully (KvCache::init null-checks), so a big cap
            // enables incremental prefill on long chats with no crash risk.
            pcfg.max_prefix_len = opts.max_ctx;
            if (const char* s = std::getenv("IE_PROMPT_CACHE_MAX_PREFIX"))
                if (unsigned v = unsigned(std::atoi(s))) pcfg.max_prefix_len = v;
            if (auto m = e->fleet_cache_.init(e->next_model_, pcfg); !m.empty()) {
                err = "fleet prompt-cache: " + m; return nullptr;
            }
            e->prompt_cache_on_ = true;
        }
        vocab = ncfg.hybrid.dense.vocab;
    } else if (e->arch_ == ModelArch::kGemma4) {
        // Gemma 4 (gemma4): dense 31B + MoE 26B-A4B, single-GPU. Self-manages its
        // per-layer KV (variable head geometry) + workspace; forward_step calls it
        // like the dense path (writes device logits directly). SPM tokenizer +
        // <start_of_turn> template + softcap. Correctness-first forward (slow).
        GemmaConfig gcfg;
        if (auto m = read_gemma4_config(e->gguf_, gcfg); !m.empty()) { err = "config: " + m; return nullptr; }
        if (auto m = e->gemma_model_.load(e->alloc_, e->gguf_, gcfg); !m.empty()) { err = "gemma4 model: " + m; return nullptr; }
        if (auto m = e->gemma_model_.ensure_workspace(opts.prefill_chunk); !m.empty()) { err = "ws: " + m; return nullptr; }
        if (auto m = e->gemma_model_.ensure_kv(opts.max_ctx); !m.empty()) { err = "kv: " + m; return nullptr; }
        // VRAM-guarded single-endpoint prompt/KV cache (task #6). Gemma self-manages
        // its per-layer KV (variable head geometry → fits neither KvCache nor
        // FleetPrefixCache), so the snapshot lives IN the model (snapshot_kv/
        // restore_kv) — no separate cache object to init here, just flip the shared
        // gate. generate()'s kGemma4 restore + snapshot branches drive it. Default-ON;
        // opt-out IE_NO_PROMPT_CACHE → byte-identical OFF path (no snapshot ever taken,
        // full prefill each turn). A too-big snapshot fails its VRAM alloc and is
        // skipped gracefully (no crash), so long-ctx sessions are safe.
        if ((opts.prompt_cache || std::getenv("IE_PROMPT_CACHE")) && !std::getenv("IE_NO_PROMPT_CACHE"))
            e->prompt_cache_on_ = true;
        // MTP self-speculative decode (--spec): load gemma's official MTP draft head
        // from a SEPARATE GGUF. Path = opts.spec_head, else convention: the first
        // mtp-*.gguf next to the target. Greedy-only; falls back to plain decode.
        // Google ships gemma-4 to be run WITH this head — strictly lossless, beats llama.
        if (opts.spec) {
            std::string head_path = opts.spec_head;
            if (head_path.empty()) {
                namespace fs = std::filesystem;
                std::error_code ec;
                fs::path dir = fs::path(gguf_path).parent_path();
                for (fs::directory_iterator it(dir, ec), end; !ec && it != end; ++it) {
                    const std::string fn = it->path().filename().string();
                    if (fn.rfind("mtp", 0) == 0 && fn.size() > 5 &&
                        fn.substr(fn.size()-5) == ".gguf") { head_path = it->path().string(); break; }
                }
                if (head_path.empty()) { err = "spec: no MTP head GGUF found next to target (pass --spec-head)"; return nullptr; }
            }
            GgufReader head_g;
            if (auto m = head_g.open(head_path); !m.empty()) { err = "spec head gguf: " + m; return nullptr; }
            if (auto m = e->gemma_model_.load_mtp_head(head_g, opts.max_ctx); !m.empty()) { err = m; return nullptr; }
            e->spec_ = true;
            std::fprintf(stderr, "[spec] gemma4 MTP head: %s\n", head_path.c_str());
        }
        vocab = gcfg.vocab;
    } else if (e->arch_ == ModelArch::kQwen35Moe && opts.n_gpus > 1) {
        // Crown (kQwen35Moe) all-Q8_0 LAYER-SPLIT across the fleet. The ~34 GiB Q8_0
        // crown won't fit one 32 GB card and the single-GPU QwenModel has no other
        // multi-GPU path. Per-card weights + KV/DeltaNet; host-logits bounce like the
        // 27B split. The single-GPU crown (model_) is untouched → PPL gate safe.
        QwenConfig cfg;
        if (auto m = e->fleet_.init(opts.n_gpus, "B70", false, /*shared_ctx=*/false); !m.empty()) { err = "fleet: " + m; return nullptr; }
        if (e->fleet_.size() < 2) { err = "layer-split needs >=2 GPUs (have " +
                                          std::to_string(e->fleet_.size()) + ")"; return nullptr; }
        LayerPlan plan = LayerPlan::contiguous(cfg.n_layers, opts.n_gpus);
        if (auto m = e->qwen35moe_split_model_.load(e->fleet_, plan, e->gguf_, cfg, opts.max_ctx, opts.int8_kv); !m.empty())
            { err = "model: " + m; return nullptr; }
        e->qwen35moe_split_ = true;
        e->tp_ids_host_.resize(std::max<uint32_t>(opts.prefill_chunk, 1u));
        e->tp_logits_host_.resize(e->qwen35moe_split_model_.config().vocab);
        // Fleet prompt/KV cache for the crown Q8 2-card split — the daily-driver
        // path that previously re-prefilled the WHOLE conversation every turn
        // ("0 cached"). Reuses the proven FleetPrefixCache (per-card Kv+DeltaNet
        // snapshot/restore) + the shared prompt_cache_on_ gate; generate()'s
        // qwen35moe_split_ branch restores/snapshots per card. Default-ON; opt out
        // IE_NO_PROMPT_CACHE → byte-identical OFF path. Cap at max_ctx (not the 8192
        // default) so long chats are cached; a too-big snapshot fails alloc and is
        // skipped gracefully (no crash).
        // --int8-kv now coexists with the prompt cache: FleetPrefixCache's
        // copy_prefix_from snapshots the per-card int8 KV shadow + scales and
        // regenerates fp16 on restore, so int8-KV chats also get re-prefill savings.
        if ((opts.prompt_cache || std::getenv("IE_PROMPT_CACHE")) && !std::getenv("IE_NO_PROMPT_CACHE")) {
            FleetPrefixCacheConfig pcfg;
            pcfg.max_prefix_len = opts.max_ctx;
            if (const char* s = std::getenv("IE_PROMPT_CACHE_MAX_PREFIX"))
                if (unsigned v = unsigned(std::atoi(s))) pcfg.max_prefix_len = v;
            if (auto m = e->fleet_cache_.init(e->qwen35moe_split_model_, pcfg); !m.empty()) {
                err = "crown-split prompt-cache: " + m; return nullptr;
            }
            e->prompt_cache_on_ = true;
        }
        vocab = e->qwen35moe_split_model_.config().vocab;
    } else if (e->arch_ == ModelArch::kGlm5Next) {
        auto* b=(e->glm5_=std::make_unique<Glm5Bundle>()).get();
        if (auto m=read_glm5next_config(e->gguf_,b->cfg); !m.empty()) { err="glm5next config: "+m; return nullptr; }
        if (!opts.max_ctx || (b->cfg.ctx_train && opts.max_ctx>b->cfg.ctx_train) ||
            !opts.prefill_chunk || opts.prefill_chunk>8192) {
            err="glm5next requires context within the model training limit and prefill chunk 1..8192"; return nullptr;
        }
        const uint32_t chunk=std::min(opts.prefill_chunk,opts.max_ctx);
        e->opts_.prefill_chunk=chunk;
        b->split=opts.n_gpus==2;
        const uint32_t n_tf=b->cfg.n_transformer_layers();
        uint32_t split=(n_tf+1)/2;
        if (const char* v=std::getenv("IE_G5_SPLIT")) {
            auto r=std::from_chars(v,v+std::strlen(v),split);
            if (r.ec!=std::errc{} || *r.ptr || !split || split>=n_tf) { err="invalid IE_G5_SPLIT"; return nullptr; }
        }
        Glm5MemoryPolicy memory;
        if (auto m=glm5_memory_policy(std::getenv("IE_G5_PIN_BANKS"),std::getenv("IE_G5_ECACHE_MB"),
                std::getenv("IE_G5_PIN_FLOOR_GIB"),std::getenv("IE_G5_PIN_MAX_GIB"),
                std::getenv("IE_G5_PIN_OVERHEAD"),memory); !m.empty()) { err=m; return nullptr; }
        // Mmap experts are reclaimable page-cache pages. Only staging must
        // fit in host RAM; optional bank pinning separately enforces its own
        // live MemAvailable floor and falls back per layer to mmap.
        uint64_t host_available=0;
        { std::ifstream mem("/proc/meminfo"); std::string line;
          while(std::getline(mem,line)) if(line.starts_with("MemAvailable:")) {
              const auto first=line.find_first_of("0123456789");
              if(first!=line.npos)std::from_chars(line.data()+first,line.data()+line.size(),host_available);
              host_available*=1024;break;
          }
        }
        uint64_t max_expert=0,host_experts=0;
        for(const auto& t:e->gguf_.tensors()) if(std::string_view(t.name).find("_exps.")!=std::string_view::npos) {
            host_experts+=t.nbytes;
            max_expert=std::max(max_expert,t.nbytes/std::max(1u,b->cfg.n_experts));
        }
        // A slot contains gate/up/down. Ring 8 + optional 16 fill-lane
        // buffers + two prefetch buffers, with the measured pin overhead.
        const uint64_t host_reserve=(8ull<<30)+max_expert*3*26*opts.n_gpus*2;
        if(!host_available || host_available<host_reserve) {
            err="glm5next: insufficient available host RAM for load scratch and expert staging";return nullptr;
        }
        std::fprintf(stderr,"[mem-plan] glm5next: %.2f GiB mmap expert banks; host staging/load reserve %.2f GiB, available %.2f GiB\n",host_experts/1073741824.,host_reserve/1073741824.,host_available/1073741824.);
        if (b->split) {
            if (auto m=b->a1.init(kGpuNameFilter,1); !m.empty()) { err="glm5next gpu1: "+m; return nullptr; }
            if(b->a1.device().get_backend()!=sycl::backend::ext_oneapi_level_zero ||
               b->a1.device()==e->alloc_.device()) {
                err="glm5next requires distinct Level Zero GPUs; set ONEAPI_DEVICE_SELECTOR=level_zero:gpu";return nullptr;
            }
            b->wide.resize(uint64_t(std::max(chunk,8u))*b->cfg.hc_count*b->cfg.hidden);
        }
        // Every stage is checked BEFORE either uploads weights. The runtime's
        // own exact resident-weight guard receives only the remaining budget.
        const uint64_t pin_cap=glm5_pin_budget(host_available,host_reserve,opts.n_gpus,memory);
        std::fprintf(stderr,"[mem-plan] glm5next: host pinning %s; per-stage pin cap %.2f GiB, host floor %.2f GiB + staging reserve\n",
                     memory.pin_banks?"on":"off (explicit override)",pin_cap/1073741824.,memory.floor_gib);
        uint64_t weight_budget[2]{}, cache_budget[2]{};
        for (uint32_t stage=0;stage<opts.n_gpus;++stage) {
            auto& al=stage?b->a1:e->alloc_;
            const uint32_t lo=stage?split:0, hi=b->split&&!stage?split:n_tf;
            uint64_t usable=usable_card_vram(al.device());
            try {
                if (al.device().has(sycl::aspect::ext_intel_free_memory))
                    usable=std::min(usable,uint64_t(al.device().get_info<sycl::ext::intel::info::device::free_memory>()*0.90));
            } catch (...) {} // some Level Zero drivers cannot report free memory
            const uint64_t reserve=glm5_runtime_reserve(b->cfg,opts.max_ctx,std::max(chunk,8u),lo,hi);
            uint64_t weights=0, experts=0, max_slot=0, moe_layers=0;
            for(uint32_t l=lo;l<hi;++l) {
                uint64_t layer_bytes=0;
                for(const char* name:{"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"})
                    if(const auto* t=e->gguf_.find_tensor("blk."+std::to_string(l)+"."+name))layer_bytes+=t->nbytes;
                if(layer_bytes) { ++moe_layers; experts+=layer_bytes;
                    max_slot=std::max(max_slot,layer_bytes/std::max(1u,b->cfg.n_experts)); }
            }
            if(auto x=Glm5NextModel::plan_device_weights(e->gguf_,b->cfg,lo,hi,weights);!x.empty()) {
                err="glm5next weight plan: "+x;return nullptr;
            }
            const uint64_t cache=glm5_cache_budget(usable,weights,reserve,experts,memory);
            const uint64_t minimum_cache=moe_layers*max_slot*b->cfg.n_experts_used;
            if(cache<minimum_cache) {
                err="glm5next stage "+std::to_string(stage)+" expert cache cannot hold one token's routed experts; reduce context/prefill or increase IE_G5_ECACHE_MB";return nullptr;
            }
            cache_budget[stage]=cache;
            std::fprintf(stderr,"[mem-plan] glm5next stage %u: weights <= %.2f GiB, runtime reserve %.2f GiB, expert cache %.2f GiB, available %.2f GiB\n",stage,weights/1073741824.,reserve/1073741824.,cache/1073741824.,usable/1073741824.);
            if (!glm5_residency_fits(weights,reserve,cache,usable)) {
                err="glm5next stage "+std::to_string(stage)+" resident weights + context/workspaces + expert cache exceed available VRAM; reduce context, prefill chunk, or IE_G5_ECACHE_MB"; return nullptr;
            }
            weight_budget[stage]=usable-reserve-cache;
        }
        for (uint32_t stage=0;stage<opts.n_gpus;++stage) {
            auto& m=stage?b->B:b->A; auto& al=stage?b->a1:e->alloc_;
            const uint32_t lo=stage?split:0, hi=b->split&&!stage?split:n_tf;
            m.configure_server_memory(memory,cache_budget[stage],pin_cap,host_reserve);
            m.set_cpu_threads(opts.cpu_threads);
            if (auto x=m.load(al,e->gguf_,b->cfg,weight_budget[stage],lo,hi);!x.empty()) { err="glm5next load: "+x;return nullptr; }
            if (auto x=m.init_runtime(opts.max_ctx,std::max(chunk,8u));!x.empty()) { err="glm5next runtime: "+x;return nullptr; }
        }
        uint64_t pinned=0, mapped=0;
        std::string stages="[";
        for(uint32_t stage=0;stage<opts.n_gpus;++stage) {
            const auto& m=stage?b->B:b->A;
            pinned+=m.pinned_bank_bytes();mapped+=m.host_bank_bytes()-m.pinned_bank_bytes();
            if(stage)stages+=",";
            stages+=std::to_string(m.expert_cache_bytes());
            std::fprintf(stderr,"[mem-plan] glm5next stage %u actual: %.2f GiB pinned host banks, %.2f GiB mmap banks, %.2f GiB GPU expert cache\n",
                stage,m.pinned_bank_bytes()/1073741824.,(m.host_bank_bytes()-m.pinned_bank_bytes())/1073741824.,m.expert_cache_bytes()/1073741824.);
        }
        e->memory_residency_json_="{\"host_pinned_bytes\":"+std::to_string(pinned)+
            ",\"host_mmap_bytes\":"+std::to_string(mapped)+",\"gpu_expert_cache_bytes\":"+stages+"]}";
        if(mapped)std::fprintf(stderr,"[mem-plan] glm5next: PARTIAL host residency; %.2f GiB remain reclaimable mmap banks and may read from disk\n",mapped/1073741824.);
        if(auto m=b->A.warm_banks();!m.empty()){err=m;return nullptr;}
        if(b->split) if(auto m=b->B.warm_banks();!m.empty()){err=m;return nullptr;}
        if(const char* profile=std::getenv("IE_G5_WARM_CACHE")) {
            if(auto m=b->A.warm_cache(profile);!m.empty()){err=m;return nullptr;}
            if(b->split) if(auto m=b->B.warm_cache(profile);!m.empty()){err=m;return nullptr;}
        }
        vocab=e->glm5_vocab_=b->cfg.vocab;
        e->prompt_cache_on_=false;
        if(opts.prompt_cache) std::fprintf(stderr,"[glm5next] prefix cache unavailable; requests reset and prefill independently\n");
    } else if (e->arch_ == ModelArch::kQwen4Exp) {
        // Qwen3.8-Flash-Next: HC wide-residual DeltaNet/QSA hybrid, host-
        // streamed routed experts behind per-card VRAM LRU caches. 1 GPU runs
        // the whole model; 2 GPUs run the layer-split pipeline [0,24)/[24,48)
        // (bit-identical to single-GPU — the ie-qwen4exp-run2 gate). The
        // engine's KvCache/DeltaNetState/prefix machinery is UNUSED: the model
        // owns its state, and q4e_forward resets it at pos==0.
        Q4eBundle* b = (e->q4e_ = std::make_unique<Q4eBundle>()).get();
        if (auto m = read_qwen4exp_config(e->gguf_, b->cfg); !m.empty()) {
            err = "config: " + m; e->q4e_.reset(); return nullptr;
        }
        // Big-chunk default (2026-08-27): prefill is dominated by expert-slot
        // H2D, and the per-chunk expert union is ~all 512 experts regardless
        // of chunk size — bigger chunks amortize the uploads. Probe (single
        // B70, 4096 toks): chunk 1024 = 293 tok/s / 40.6K misses; 2048 = 346;
        // 4096 = 404-413 tok/s / 15.2K misses. The gptoss big-chunk pattern:
        // arch default, IE_Q4E_PREFILL_CHUNK overrides (=1024 reverts; lower
        // it if VRAM-bound — the MoE mega buffers grow ~T*n_used*EF).
        uint32_t chunk = std::min(4096u, opts.max_ctx);
        if (const char* c = std::getenv("IE_Q4E_PREFILL_CHUNK")) {
            const int v = std::atoi(c);
            if (v >= 64 && uint32_t(v) <= opts.max_ctx) chunk = uint32_t(v);
        }
        e->opts_.prefill_chunk = chunk;
        b->split = opts.n_gpus >= 2;
        const uint32_t hi = b->split ? b->cfg.n_layers / 2 : b->cfg.n_layers;
        if (auto m = b->A.load(e->alloc_, e->gguf_, b->cfg, 0, 0, hi); !m.empty()) {
            err = "qwen4exp A: " + m; e->q4e_.reset(); return nullptr;
        }
        if (auto m = b->A.init_runtime(opts.max_ctx, chunk); !m.empty()) {
            err = "qwen4exp A runtime: " + m; e->q4e_.reset(); return nullptr;
        }
        if (b->split) {
            b->p2p = q4e_p2p;
            if (b->p2p) {
                if (auto m = b->a1.init_with(*q4e_p2p_ctx, q4e_p2p_gpus[1]); !m.empty()) {
                    err = "qwen4exp gpu1: " + m; e->q4e_.reset(); return nullptr;
                }
            } else if (auto m = b->a1.init("B70", 1); !m.empty()) {
                err = "qwen4exp gpu1: " + m; e->q4e_.reset(); return nullptr;
            }
            if (auto m = b->B.load(b->a1, e->gguf_, b->cfg, 0, hi, b->cfg.n_layers); !m.empty()) {
                err = "qwen4exp B: " + m; e->q4e_.reset(); return nullptr;
            }
            if (auto m = b->B.init_runtime(opts.max_ctx, chunk); !m.empty()) {
                err = "qwen4exp B runtime: " + m; e->q4e_.reset(); return nullptr;
            }
            if (b->p2p) b->A.set_wide_peer(b->B.wide_device());
            b->wide.resize(uint64_t(chunk) * b->cfg.hc_count * b->cfg.hidden);
        }
        e->q4e_vocab_ = b->cfg.vocab;
        vocab = b->cfg.vocab;
        e->prompt_cache_on_ = true;   // snapshot/restore via the generic flow
    } else if (e->arch_ == ModelArch::kDeepSeek4) {
        // DeepSeek-V4-Flash (deepseek4): MLA with q/kv/o LoRA, a lightning
        // indexer that picks the top-k KV entries, per-layer KV compressors,
        // hyper-connections, and a 256-expert MoE with one shared expert. Only a
        // few GB per card is always resident; the routed experts live in a pinned
        // host arena and stream into a VRAM slot cache on demand. The runtime
        // therefore owns its own device(s), queues and caches — the engine's
        // KvCache/DeltaNetState/prefix cache are all UNUSED on this path, and
        // forward_step bounces host logits into d_logits_ like the fleet paths.
        Ds4Bundle* b = (e->ds4_ = std::make_unique<Ds4Bundle>()).get();
        if (auto m = read_deepseek4_config(e->gguf_, b->cfg); !m.empty()) {
            err = "config: " + m; e->ds4_.reset(); return nullptr;
        }
        // Vision-Exp sidecar: $IE_DS4_VISION, else the one `*Native.safetensors`
        // beside the GGUF. Uploaded on GPU 0 NOW so the runtime's expert-arena
        // derivation below sees it (vram_reserved_extra). A GGUF without the
        // exp_probs_b_vl bias is text-only even with a sidecar present.
        // This block runs BEFORE model.load(): DeepSeek4Model copies cfg by
        // value and the runtime takes ITS copy, so an rms_eps written here after
        // the load would only change the log line (P0 gate finding 7).
        {
            namespace fs = std::filesystem;
            std::string side;
            if (const char* v = std::getenv("IE_DS4_VISION")) side = v;
            else {
                const fs::path dir = fs::path(e->model_path_).parent_path();
                std::error_code ec;
                for (const auto& ent : fs::directory_iterator(dir, ec)) {
                    const std::string nm = ent.path().filename().string();
                    static constexpr std::string_view kSuffix = "-Native.safetensors";
                    if (nm.size() > kSuffix.size() &&
                        std::string_view(nm).substr(nm.size() - kSuffix.size()) == kSuffix) {
                        side = ent.path().string(); break;
                    }
                }
            }
            if (!side.empty() && !std::getenv("IE_DS4_NO_VISION")) {
                auto v = std::make_unique<Ds4Vision>();
                if (auto m = v->load(side); !m.empty()) {
                    err = "deepseek4 vision sidecar: " + m; e->ds4_.reset(); return nullptr;
                }
                if (auto m = v->upload(e->alloc_); !m.empty()) {
                    err = "deepseek4 vision upload: " + m; e->ds4_.reset(); return nullptr;
                }
                e->ds4_vis_ = std::move(v);
                e->ds4_vis_path_ = side;
                // Official Vision-Exp config: rms_norm_eps = 1e-20 (the GGUF key
                // carries the 0731 value 1e-6). config.json beside the GGUF wins
                // over the file key; $IE_DS4_RMS_EPS (already applied) wins over both.
                if (!std::getenv("IE_DS4_RMS_EPS")) {
                    const fs::path cj = fs::path(side).parent_path() / "config.json";
                    if (std::ifstream f{cj}) {
                        std::string body((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
                        if (auto p = body.find("\"rms_norm_eps\""); p != std::string::npos)
                            if (auto c = body.find(':', p); c != std::string::npos)
                                b->cfg.rms_eps = std::strtof(body.c_str() + c + 1, nullptr);
                    }
                }
                std::fprintf(stderr, "[ds4] vision sidecar %s uploaded on GPU 0 (%.2f GB); rms_eps %g\n",
                             side.c_str(), double(Ds4Vision::device_bytes()) / 1e9, double(b->cfg.rms_eps));
            }
        }
        if (auto m = b->model.load(e->gguf_, b->cfg); !m.empty()) {
            err = "deepseek4 model: " + m; e->ds4_.reset(); return nullptr;
        }
        Ds4Options o;
        // Largest T one forward() may submit. generate() chunks the prefill at
        // opts_.prefill_chunk and decodes at T=1, so this is exactly that bound.
        o.max_seq     = std::max<uint32_t>(1u, opts.prefill_chunk);
        // deepseek4 prefill is DMA-bound per chunk (a 512-token chunk streams
        // ~26 GB of experts per card, a 1024-token chunk ~31 GB), so the chunk
        // size sets the tok/s almost directly: measured 2026-09-02 on 2x B70,
        // pp1024 328 vs 270 tok/s and pp2048 305 vs 244 at 1024- vs 512-token
        // chunks, decode unchanged (28.8 vs 27.7-29.6), one expert slot fewer
        // (87 vs 88) for the larger batch workspace (docs/deepseek4/72 Phase H).
        // The 256 default in EngineOptions is a Qwen linear-attention bound
        // (docs/known_bugs.md) and does not apply here.  $IE_DS4_PREFILL_CHUNK
        // overrides in either direction.  2048 since docs/deepseek4/72 Phase N
        // (with the XMX attention of Phase K the chunk is DMA-bound again and a
        // 2048-token chunk halves the expert bytes per token; the batch
        // workspace cap kDs4BatchTokenCap moved with it).
        {
            uint32_t want = 2048u;
            if (const char* c = std::getenv("IE_DS4_PREFILL_CHUNK"); c && *c)
                want = std::max<uint32_t>(1u, uint32_t(std::atoi(c)));
            if (o.max_seq != want) {
                std::fprintf(stderr, "[ds4] prefill chunk %u -> %u (%s)\n", o.max_seq, want,
                             std::getenv("IE_DS4_PREFILL_CHUNK") ? "$IE_DS4_PREFILL_CHUNK"
                                                                  : "deepseek4 default, DMA-bound chunks");
                o.max_seq = want;
                e->opts_.prefill_chunk = want;
            }
        }
        if (e->ds4_vis_) {
            // An image block (<= 384 rows + 3 pad) must be prefilled in ONE
            // forward(): floor the chunk at 512 for a vision-capable load.
            if (o.max_seq < 512u) {
                std::fprintf(stderr, "[ds4] vision: prefill chunk %u -> 512 (an image block must fit one forward)\n",
                             o.max_seq);
                o.max_seq = 512u;
                e->opts_.prefill_chunk = 512u;
            }
            o.vram_reserved_extra = Ds4Vision::device_bytes();
        }
        o.max_context = opts.max_ctx;
        // Options with no EngineOptions equivalent keep their Ds4Options default
        // of 0/empty, which is what makes them resolve through the DS4_* env vars
        // the runtime already reads ($DS4_PIN_CAP_GB, $DS4_STREAM_SLOTS,
        // $DS4_EXPERT_PRIORITY_FILE, $DS4_EXPERT_PROFILE) — so serving behaves
        // exactly like the test/bench harness. pin_layers has no such fallback
        // inside the runtime, so it is read here (same var the harness uses).
        if (const char* s = std::getenv("DS4_PIN_LAYERS")) {
            const int v = std::atoi(s);
            if (v > 0) o.pin_layers = uint32_t(v);
        }
        if (opts.n_gpus > 1) {
            // Tensor-parallel: n_cards runtimes, each holding
            // expert_ffn / n_cards of EVERY routed expert plus its share of the
            // split non-expert set, driven in lockstep with one cross-card
            // reduction per layer. This is the ONLY supported way to run a sliced
            // load (a lone sliced runtime refuses its own forward).
            Ds4TpOptions to;
            to.base    = o;             // device_ordinal/n_cards/card stay at the
            to.n_cards = opts.n_gpus;   // defaults — load() refuses otherwise
            // $DS4_TP_GPUS ("0,1") pins the card→GPU map, same var the harness
            // uses; empty is the identity map {0..n_cards-1}.
            if (const char* s = std::getenv("DS4_TP_GPUS"); s && *s) {
                std::string cur;
                for (const char* p = s;; ++p) {
                    if (*p == ',' || *p == 0) {
                        if (!cur.empty()) to.device_ordinals.push_back(uint32_t(std::atoi(cur.c_str())));
                        cur.clear();
                        if (*p == 0) break;
                    } else cur.push_back(*p);
                }
                if (to.device_ordinals.size() != to.n_cards) {
                    err = "DS4_TP_GPUS lists " + std::to_string(to.device_ordinals.size()) +
                          " GPU(s) but --gpus is " + std::to_string(to.n_cards);
                    e->ds4_.reset(); return nullptr;
                }
            }
            if (std::getenv("IE_DS4_DRY_RUN")) {
                err = "deepseek4 dry-run (IE_DS4_DRY_RUN): stopped before the runtime load";
                std::fprintf(stderr,
                    "[ds4] DRY RUN: %llu tensors bound | %u layers, %u experts (top-%u), hidden %u,"
                    " vocab %u | TENSOR-PARALLEL over %u card(s) | max_seq %u, max_context %u\n",
                    (unsigned long long)b->model.n_bound(), b->cfg.n_layers, b->cfg.n_experts,
                    b->cfg.n_experts_used, b->cfg.hidden, b->cfg.vocab, to.n_cards,
                    o.max_seq, o.max_context);
                e->ds4_.reset(); return nullptr;
            }
            b->tp = std::make_unique<DeepSeek4TpRuntime>();
            if (auto m = b->tp->load(b->model, to); !m.empty()) {
                err = "deepseek4 tp runtime: " + m; e->ds4_.reset(); return nullptr;
            }
        } else {
            // Single card. $DS4_GPU picks which one (the harness's variable);
            // absent, card 0, like every other single-GPU path.
            if (const char* s = std::getenv("DS4_GPU")) o.device_ordinal = uint32_t(std::atoi(s));
            if (std::getenv("IE_DS4_DRY_RUN")) {
                err = "deepseek4 dry-run (IE_DS4_DRY_RUN): stopped before the runtime load";
                std::fprintf(stderr,
                    "[ds4] DRY RUN: %llu tensors bound | %u layers, %u experts (top-%u), hidden %u,"
                    " vocab %u | SINGLE card, GPU %u | max_seq %u, max_context %u\n",
                    (unsigned long long)b->model.n_bound(), b->cfg.n_layers, b->cfg.n_experts,
                    b->cfg.n_experts_used, b->cfg.hidden, b->cfg.vocab, o.device_ordinal,
                    o.max_seq, o.max_context);
                e->ds4_.reset(); return nullptr;
            }
            b->rt = std::make_unique<DeepSeek4Runtime>();
            if (auto m = b->rt->load(b->model, o); !m.empty()) {
                err = "deepseek4 runtime: " + m; e->ds4_.reset(); return nullptr;
            }
        }
        // Prompt cache (task: DS4 prefix reuse). Before this, deepseek4 was
        // excluded from prefix caching BY OMISSION — there was no guard here, so
        // prompt_cache_on_ stayed false and ds4_forward's `pos == 0 →
        // reset_context()` fired on every request. Turn 10 of a conversation
        // re-prefilled turns 1-9: 2 000 tokens at the measured pp512 = 52.91
        // tok/s is 37.8 s of TTFT (docs/deepseek4/60_CONTINUOUS_BATCHING.md §6.4).
        //
        // The snapshot is model-owned and single-endpoint, following Gemma4 —
        // the DS4 caches are nothing like KvCache/DeltaNetState, so PrefixCache
        // and FleetPrefixCache cannot be reused for them.
        //
        // THE DEPTH CAP IS LOAD-BEARING, NOT A TUNABLE. A snapshot costs
        // 22.55 MiB + 6.88 kB per cached token PER CARD (13.76 before the fp16
        // compressed entries of docs/deepseek4/73 Phase 2), and it is drawn from
        // the same ~2 GiB reserve the expert arena is derived from
        // (`global_mem - resident - 2 GiB`, deepseek4.cpp). At --ctx 200000 an
        // uncapped snapshot would be 2.8 GB and would take the arena with it —
        // and a shrunken arena is measured in decode throughput (residency is
        // what bought 3.79 → 24.80 tok/s). The cap was 8 192 (~135 MB/card)
        // until docs/deepseek4/72 Phase L: an agent conversation is routinely
        // 16-64K tokens and re-prefilling its tail every turn (at ~360 tok/s,
        // 45 s per turn at 24K) is the single largest agentic latency on this
        // path. 65 536 tokens costs at most ~925 MB/card, inside the reserve,
        // and the operator's --ctx bounds it below that. Override either way
        // with $IE_DS4_PROMPT_CACHE_MAX_PREFIX and watch VRAM.
        if ((opts.prompt_cache || std::getenv("IE_PROMPT_CACHE")) &&
            !std::getenv("IE_NO_PROMPT_CACHE")) {
            e->ds4_cache_max_prefix_ = std::min<uint32_t>(opts.max_ctx, 65536u);
            if (const char* s = std::getenv("IE_DS4_PROMPT_CACHE_MAX_PREFIX"))
                if (unsigned v = unsigned(std::atoi(s)))
                    e->ds4_cache_max_prefix_ = std::min<uint32_t>(opts.max_ctx, v);
            e->prompt_cache_on_ = true;
            // Host-resident slots (Phase L3): each holds one conversation's state
            // in pinned host memory; $IE_DS4_PROMPT_CACHE_SLOTS sets how many.
            uint32_t n_slots = 4;
            if (const char* s = std::getenv("IE_DS4_PROMPT_CACHE_SLOTS"))
                if (int v = std::atoi(s); v > 0) n_slots = uint32_t(v);
            e->ds4_->slots.resize(n_slots);
            std::fprintf(stderr,
                "[ds4-cache] prompt cache on: %u host slots, prefix depth cap %u tokens (a full-depth snapshot is ~%u MB/card per slot)\n",
                n_slots, e->ds4_cache_max_prefix_,
                unsigned((22.55 * 1048576.0 + 6880.0 * e->ds4_cache_max_prefix_) / 1e6));
        }
        b->logits.resize(b->cfg.vocab);
        e->ds4_vocab_ = b->cfg.vocab;
        e->tp_ids_host_.resize(std::max<uint32_t>(opts.prefill_chunk, 1u));
        e->tp_logits_host_.resize(b->cfg.vocab);
        vocab = b->cfg.vocab;
    } else {
        QwenConfig cfg;
        if (auto m = e->model_.load(e->alloc_, e->gguf_, cfg); !m.empty()) { err = "model: " + m; return nullptr; }
        cfg = e->model_.config();

        // Caches sized for max_ctx (pattern from tools/ie_bench.cpp).
        const uint32_t L_full = cfg.n_layers / cfg.full_attn_interval;
        const uint32_t L_lin  = cfg.n_layers - L_full;
        {
            KvCacheConfig kvcfg{};
            kvcfg.n_layers_full = L_full;
            kvcfg.n_kv_heads    = cfg.n_kv_heads;
            kvcfg.max_ctx       = opts.max_ctx;
            kvcfg.head_dim      = cfg.head_dim;
            kvcfg.use_int8      = opts.int8_kv;
            if (auto m = e->kv_.init(e->alloc_, kvcfg); !m.empty()) { err = "kv: " + m; return nullptr; }
        }
        if (auto m = e->dn_.init(e->alloc_, DeltaNetStateConfig{
                L_lin, cfg.ssm_n_v_heads, cfg.ssm_head_dim, cfg.ssm_head_dim,
                cfg.ssm_inner * 2, cfg.ssm_conv_kernel}); !m.empty()) {
            err = "dn: " + m; return nullptr;
        }
        // Crown long-ctx oneDNN-MoE lever (DEFAULT-ON for max_ctx>=8192): size the
        // forward workspace to the BIG prefill chunk NOW. ensure_workspace decides
        // whether to allocate the oneDNN MoE buffers + 8192 scratch from its max_T
        // arg, and it only re-runs when max_T grows — so if we sized it to the prompt
        // length at first forward, a prompt in [4096, 8192) would miss the lever (and
        // hit the per-token cliff). Sizing to min(8192,max_ctx) here makes ANY prompt
        // >=4096 engage oneDNN. Opt-out IE_QWEN36_NO_MOE_ONEDNN; explicit
        // IE_QWEN36_MOE_ONEDNN forces it below 8192 (testing).
        uint32_t crown_ws = opts.prefill_chunk;
        if (std::getenv("IE_QWEN36_NO_MOE_ONEDNN") == nullptr &&
            (opts.max_ctx >= 8192u || std::getenv("IE_QWEN36_MOE_ONEDNN")))
            crown_ws = std::max<uint32_t>(crown_ws, std::min<uint32_t>(8192u, opts.max_ctx));
        if (auto m = e->model_.ensure_workspace(crown_ws); !m.empty()) { err = "ws: " + m; return nullptr; }
        if (auto m = e->model_.ensure_attn_partials(opts.max_ctx); !m.empty())  { err = "partials: " + m; return nullptr; }
        // Prompt/KV cache — crown (kQwen35Moe) single-GPU only; PrefixCache::init is
        // typed to QwenConfig. Default-ON; opt out IE_NO_PROMPT_CACHE → byte-identical OFF path.
        if ((opts.prompt_cache || std::getenv("IE_PROMPT_CACHE")) && !std::getenv("IE_NO_PROMPT_CACHE")) {
            PrefixCacheConfig pcfg;
            pcfg.max_prefix_len = opts.max_ctx;   // was 4096 — cache long single-GPU sessions too
            if (const char* s = std::getenv("IE_PROMPT_CACHE_MAX_PREFIX"))
                if (unsigned v = unsigned(std::atoi(s))) pcfg.max_prefix_len = v;
            if (auto m = e->prefix_cache_.init(e->alloc_, cfg, pcfg); !m.empty()) {
                err = "prompt-cache: " + m; return nullptr;
            }
            e->prompt_cache_on_ = true;
        }
        vocab = cfg.vocab;
    }
    if (auto m = e->tok_.load_from_gguf(e->gguf_); !m.empty()) { err = "tokenizer: " + m; return nullptr; }

    auto& q = e->alloc_.queue();
    e->d_logits_ = sycl::malloc_device<sycl::half>(vocab, q);
    e->d_ids_    = sycl::malloc_device<int32_t>(opts.max_ctx, q);
    e->d_pick_   = sycl::malloc_device<int32_t>(1, q);
    e->d_recent_ = sycl::malloc_device<int32_t>(512, q);
    if (!e->d_logits_ || !e->d_ids_ || !e->d_pick_ || !e->d_recent_) { err = "device alloc"; return nullptr; }
    // Stop set, keyed FIRST on the Wave-1 template family (a Mistral/DeepSeek
    // GGUF detects as kLlama3/kQwen3Dense but needs its own stop token), then on
    // arch_. Every CURRENT model is kAuto → it keeps its exact arch-based branch.
    using TF = DenseConfig::TemplateFamily;
    const TF tf = is_dense_arch(e->arch_) ? e->dcfg_.template_family : TF::kAuto;
    if (tf == TF::kMistral) {
        // Mistral turns close on </s> (which is also the GGUF eos, caught at
        // sampling); set it explicitly so a non-eos </s> id still stops.
        e->stop_ids_[0] = e->tok_.find_token("</s>");
        e->stop_ids_[1] = -1;
    } else if (tf == TF::kDeepSeek) {
        // R1-Distill assistant turns close on <｜end▁of▁sentence｜> (the GGUF eos,
        // already caught at sampling); set it explicitly for robustness.
        e->stop_ids_[0] = e->tok_.find_token(
            "<\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c>");
        e->stop_ids_[1] = -1;
    } else if (tf == TF::kGranite) {
        // Granite assistant turns close on <|end_of_text|> (=GGUF eos id 0,
        // already caught at sampling); set it explicitly for robustness.
        e->stop_ids_[0] = e->tok_.find_token("<|end_of_text|>");
        e->stop_ids_[1] = -1;
    } else if (tf == TF::kChatML && e->arch_ != ModelArch::kQwen3Dense) {
        // ChatML-template GGUF on a non-Qwen3 dense arch (Phi-4: arch phi3→kLlama3,
        // but its template/eos are ChatML). It would otherwise fall to the kLlama3
        // branch and set absent <|eot_id|>/<|end_of_text|>. Phi-4 closes turns on
        // <|im_end|> (=GGUF eos, already caught at sampling); <|endoftext|> is the
        // model's second EOG token — set BOTH so a non-eos EOG id still stops.
        // Qwen3-dense (also a kChatML template) is EXCLUDED so it keeps its exact
        // prior stop set (<|im_end|> only); adding <|endoftext|>=151643 there would
        // change its behavior. Phi-3.5-mini (also phi3) will share this branch.
        e->stop_ids_[0] = e->tok_.find_token("<|im_end|>");
        e->stop_ids_[1] = e->tok_.find_token("<|endoftext|>");
    } else if (e->arch_ == ModelArch::kLlama3) {
        e->stop_ids_[0] = e->tok_.find_token("<|eot_id|>");
        e->stop_ids_[1] = e->tok_.find_token("<|end_of_text|>");
    } else if (e->arch_ == ModelArch::kGemma4) {
        // Gemma 4 model turns close on <turn|> (id 106); <eos>=1 caught at sampling.
        e->stop_ids_[0] = e->tok_.find_token("<turn|>");
        e->stop_ids_[1] = -1;
    } else if (e->arch_ == ModelArch::kDeepSeek4) {
        // DeepSeek-V4-Flash assistant turns close on <｜end▁of▁sentence｜> — the
        // same literal build_deepseek4_prompt() emits after an assistant turn,
        // and the GGUF eos (already caught at sampling). Set explicitly so a
        // non-eos occurrence of that id still stops, mirroring the kDeepSeek
        // (R1-Distill) branch above.
        e->stop_ids_[0] = e->tok_.find_token(
            "<\xef\xbd\x9c" "end" "\xe2\x96\x81" "of" "\xe2\x96\x81" "sentence" "\xef\xbd\x9c>");
        e->stop_ids_[1] = -1;
    } else if (e->arch_ == ModelArch::kGlm5Next) {
        e->stop_ids_[0] = e->tok_.find_token("<|user|>");
        e->stop_ids_[1] = e->tok_.find_token("<|observation|>");
    } else if (e->arch_ == ModelArch::kGptOss) {
        // gpt-oss Harmony: the assistant FINAL turn closes on <|return|>=200002 (the
        // GGUF eos, caught at sampling); tool calls close on <|call|>=200012. <|end|>
        // (200007) is NOT a stop — it closes the analysis channel mid-generation.
        e->stop_ids_[0] = e->tok_.find_token("<|call|>");   // 200012
        e->stop_ids_[1] = -1;
    } else {
        e->stop_ids_[0] = e->tok_.find_token("<|im_end|>");
        e->stop_ids_[1] = -1;
    }
    e->tool_call_ids_[0] = e->tok_.find_token("<tool_call>");
    e->tool_call_ids_[1] = e->tok_.find_token("</tool_call>");
    if (e->tool_call_ids_[0] == -1 || e->tool_call_ids_[1] == -1)
        fprintf(stderr, "[ie] warning: model vocab lacks <tool_call> markers;"
                        " tool-call text may lose framing\n");
    return e;
}

sycl::event Engine::ds4_forward(sycl::queue& q, const int32_t* ids, uint32_t T,
                                uint32_t pos) {
    // Host-bounce, like every fleet path: `ids` is a device pointer into d_ids_,
    // the runtime wants host ids and writes host logits. Two differences from the
    // others: the logits arrive fp32 (the whole deepseek4 activation path is
    // fp32) so they are narrowed into tp_logits_host_ for the shared fp16 GPU
    // sampler, and the per-layer caches are cleared HERE at pos==0 — the runtime
    // has no reset_kv flag, it has reset_context().
    if (tp_ids_host_.size() < T) tp_ids_host_.resize(T);
    q.memcpy(tp_ids_host_.data(), ids, T * sizeof(int32_t)).wait();
    if (pos == 0) {
        if (ds4_->tp) { ds4_->tp->reset_context(); ds4_->tp->clear_vision(); }
        else          { ds4_->rt->reset_context(); ds4_->rt->clear_vision(); }
        if (ds4_vis_active_) ds4_apply_vision();
    }
    const uint32_t V = ds4_vocab_;
    if (ds4_->logits.size() < V)      ds4_->logits.resize(V);
    if (tp_logits_host_.size() < V)   tp_logits_host_.resize(V);
    const std::string m =
        ds4_->tp ? ds4_->tp->forward(tp_ids_host_.data(), T, pos, ds4_->logits.data(),
                                     /*last_only=*/true)
                 : ds4_->rt->forward(tp_ids_host_.data(), T, pos, ds4_->logits.data(),
                                     /*last_only=*/true);
    if (!m.empty()) {
        std::fprintf(stderr, "deepseek4 forward: %s\n", m.c_str());
        if (ds4_err_.empty()) ds4_err_ = m;   // latched; generate() stops on it
    }
    for (uint32_t v = 0; v < V; ++v) tp_logits_host_[v] = sycl::half(ds4_->logits[v]);
    return q.memcpy(d_logits_, tp_logits_host_.data(), uint64_t(V) * sizeof(sycl::half));
}

std::string Engine::q4e_stage_vision(const std::vector<int32_t>& ids) {
    // Lazy mmproj: env IE_MMPROJ, else mmproj*.gguf beside the model file.
    if (!q4e_vis_) {
        if (q4e_mmproj_.empty()) {
            if (const char* mp = std::getenv("IE_MMPROJ")) q4e_mmproj_ = mp;
            else {
                namespace fs = std::filesystem;
                const fs::path dir = fs::path(model_path_).parent_path();
                for (const auto* cand : {"mmproj-F16.gguf", "mmproj-BF16.gguf"}) {
                    if (fs::exists(dir / cand)) { q4e_mmproj_ = (dir / cand).string(); break; }
                    if (fs::exists(dir.parent_path() / cand)) {
                        q4e_mmproj_ = (dir.parent_path() / cand).string(); break;
                    }
                }
            }
            if (q4e_mmproj_.empty())
                return "vision: no mmproj*.gguf next to the model (or set IE_MMPROJ)";
        }
        auto v = std::make_unique<Qwen4Vision>();
        if (auto e = v->load(q4e_mmproj_); !e.empty()) return "vision: " + e;
        q4e_vis_ = std::move(v);
        std::fprintf(stderr, "[vision] loaded %s\n", q4e_mmproj_.c_str());
    }
    const int32_t pad = q4e_->cfg.ple_image_token ? int32_t(q4e_->cfg.ple_image_token)
                                                  : 248056;
    // Pad runs in prompt order must match the pending images one-to-one.
    q4e_vis_rows_.clear();
    q4e_vis_spans_.clear();
    std::vector<Qwen4VisGrid> grids;
    size_t img = 0;
    for (uint32_t t = 0; t < ids.size(); ) {
        if (ids[t] != pad) { ++t; continue; }
        uint32_t n = 0;
        while (t + n < ids.size() && ids[t + n] == pad) ++n;
        if (img >= q4e_pending_imgs_.size())
            return "vision: more image_pad runs than attached images";
        auto& im = q4e_pending_imgs_[img];
        const uint32_t gh2 = im.H / 32, gw2 = im.W / 32;
        if (n != gh2 * gw2)
            return "vision: image_pad run length mismatch (client-injected pads?)";
        DeviceAllocator& va = q4e_->split ? q4e_->a1 : alloc_;
        std::vector<float> emb;
        if (auto e = q4e_vis_->encode_gpu(va, im.px.data(), im.H, im.W, emb);
            !e.empty()) return "vision encode: " + e;
        q4e_vis_spans_.push_back({t, n});
        grids.push_back({t, gh2, gw2});
        q4e_vis_rows_.insert(q4e_vis_rows_.end(), emb.begin(), emb.end());
        ++img;
        t += n;
    }
    if (img != q4e_pending_imgs_.size())
        return "vision: fewer image_pad runs than attached images";
    qwen4_build_mrope3(uint32_t(ids.size()), grids, q4e_pos3_, q4e_delta_);
    q4e_pending_imgs_.clear();
    q4e_vis_active_ = true;
    return {};
}

void Engine::q4e_apply_vision() {
    if (!q4e_ || !q4e_vis_active_) return;
    const uint32_t H = q4e_->cfg.hidden;
    size_t row = 0;
    for (const auto& [t0, n] : q4e_vis_spans_) {
        if (auto e = q4e_->A.set_vision(q4e_vis_rows_.data() + row * H, t0, n);
            !e.empty() && q4e_err_.empty()) q4e_err_ = e;
        row += n;
    }
    const uint32_t np = uint32_t(q4e_pos3_.size() / 3);
    q4e_->A.set_mrope(q4e_pos3_.data(), np, q4e_delta_);
    if (q4e_->split) q4e_->B.set_mrope(q4e_pos3_.data(), np, q4e_delta_);
}

// ---- DeepSeek-V4-Flash-Vision-Exp ---------------------------------------
// Placeholder token: the official tokenizer names id 129264 `<｜deepseek_image｜>`;
// the GGUF vocab (converted from the 0731 tokenizer) labels the same id
// `<｜image2｜>`. chat() emits whatever text the loaded vocab has for the id.
static constexpr int32_t kDs4ImagePlaceholderId = 129264;

std::string Engine::ds4_expand_vision(std::vector<int32_t>& ids) {
    ds4_vis_spans_.clear();
    ds4_vis_rows_.clear();
    std::vector<int32_t> out;
    out.reserve(ids.size() + size_t(ds4_pending_imgs_.size()) * (kDs4VisMaxTok + 4));
    size_t img = 0;
    for (int32_t id : ids) {
        if (id != kDs4ImagePlaceholderId) { out.push_back(id); continue; }
        if (img >= ds4_pending_imgs_.size())
            return "vision: more image placeholders in the prompt than attached images";
        Ds4PendingImage& im = ds4_pending_imgs_[img++];
        const uint32_t start = uint32_t(out.size());
        ds4_vis_build_block(im.geom.n_llm_h, im.geom.n_llm_w, start, im.types, im.perm);
        if (im.types.size() > opts_.prefill_chunk)
            return "vision: image block of " + std::to_string(im.types.size()) +
                   " rows exceeds the prefill chunk " + std::to_string(opts_.prefill_chunk);
        const uint32_t compress_pad = kDs4VisPadTo - 1 - start % kDs4VisPadTo;
        ds4_vis_spans_.push_back({start, uint32_t(im.types.size()), compress_pad});
        // Every block row carries the placeholder id: the embedding row is
        // overwritten by the splice and the routers never hash an image row.
        out.insert(out.end(), im.types.size(), kDs4ImagePlaceholderId);
    }
    if (img != ds4_pending_imgs_.size())
        return "vision: fewer image placeholders in the prompt than attached images";
    ids.swap(out);
    ds4_vis_active_ = true;
    return {};
}

std::string Engine::ds4_stage_vision() {
    if (!ds4_vis_) return "vision: no sidecar loaded";
    ds4_vis_rows_.clear();
    const auto t0 = std::chrono::steady_clock::now();
    uint32_t n_patches = 0;
    for (Ds4PendingImage& im : ds4_pending_imgs_) {
        n_patches += (im.H / kDs4VisPatch) * (im.W / kDs4VisPatch);
        std::vector<float> aligned, rows;
        if (auto e = ds4_vis_->encode_gpu(alloc_, im.px.data(), im.H, im.W, aligned); !e.empty())
            return "vision encode: " + e;
        if (auto e = ds4_vis_->assemble_rows(aligned, im.types, im.perm, rows); !e.empty())
            return "vision rows: " + e;
        ds4_vis_rows_.insert(ds4_vis_rows_.end(), rows.begin(), rows.end());
    }
    std::fprintf(stderr, "[vision] %zu image(s), %u patches, %zu rows encoded in %.0f ms\n",
                 ds4_pending_imgs_.size(), n_patches, ds4_vis_rows_.size() / size_t(ds4_->cfg.hidden),
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    ds4_pending_imgs_.clear();
    return {};
}

void Engine::ds4_apply_vision() {
    if (!ds4_ || !ds4_vis_active_) return;
    const uint32_t H = ds4_->cfg.hidden;
    size_t row = 0;
    for (const Ds4VisSpan& sp : ds4_vis_spans_) {
        const float* rows = ds4_vis_rows_.data() + row * H;
        const std::string e = ds4_->tp ? ds4_->tp->set_vision(rows, sp.t0, sp.n, sp.start_off)
                                       : ds4_->rt->set_vision(rows, sp.t0, sp.n, sp.start_off);
        if (!e.empty() && ds4_err_.empty()) ds4_err_ = e;
        row += sp.n;
    }
}

sycl::event Engine::glm5_forward(sycl::queue& q,const int32_t* ids,uint32_t T,uint32_t pos) {
    if(!glm5_err_.empty()) return {};
    tp_ids_host_.resize(T);
    q.memcpy(tp_ids_host_.data(),ids,uint64_t(T)*sizeof(int32_t)).wait();
    if(pos==0) { glm5_->A.reset_state(); if(glm5_->split) glm5_->B.reset_state(); }
    if(auto m=glm5_->fwd(tp_ids_host_.data(),T,pos);!m.empty()) { glm5_err_=m;return {}; }
    auto& tail=glm5_->split?glm5_->B:glm5_->A;
    tp_logits_host_.resize(glm5_vocab_);
    tail.queue().memcpy(tp_logits_host_.data(),tail.logits(),uint64_t(glm5_vocab_)*2).wait();
    for(auto v:tp_logits_host_) if(!std::isfinite(float(v))) {glm5_err_="non-finite logits";return {};}
    return q.memcpy(d_logits_,tp_logits_host_.data(),uint64_t(glm5_vocab_)*2);
}

sycl::event Engine::q4e_forward(sycl::queue& q, const int32_t* ids, uint32_t T,
                                uint32_t pos) {
    // Host-bounce like ds4_forward: device ids in, model-owned state, fp16
    // logits bounced from the tail card into the engine's sampler buffer.
    if (tp_ids_host_.size() < T) tp_ids_host_.resize(T);
    q.memcpy(tp_ids_host_.data(), ids, T * sizeof(int32_t)).wait();
    if (pos == 0) {
        q4e_->A.reset_state();
        if (q4e_->split) q4e_->B.reset_state();
        q4e_apply_vision();   // reset just wiped staging; re-arm for this request
    }
    const uint32_t V = q4e_vocab_;
    if (tp_logits_host_.size() < V) tp_logits_host_.resize(V);
    Q4eBundle& b = *q4e_;
    Qwen4ExpModel& tail = b.split ? b.B : b.A;
    std::string m = b.fwd(tp_ids_host_.data(), T, pos);
    if (!m.empty()) {
        std::fprintf(stderr, "qwen4exp forward: %s\n", m.c_str());
        if (q4e_err_.empty()) q4e_err_ = m;
    } else {
        tail.queue().memcpy(tp_logits_host_.data(), tail.logits(),
                            uint64_t(V) * sizeof(sycl::half)).wait();
    }
    return q.memcpy(d_logits_, tp_logits_host_.data(), uint64_t(V) * sizeof(sycl::half));
}

Engine::~Engine() {
    // Stop the joint-step scheduler before any device teardown.
    if (stepper_.th.joinable()) {
        {
            std::lock_guard<std::mutex> l(stepper_.mu);
            stepper_.shutdown = true;
        }
        stepper_.cv.notify_all();
        stepper_.th.join();
    }
    if (stepper_.d_pick || stepper_.d_recent) {
        sycl::queue& hq = qwen35_split_model_.fleet()
            ? qwen35_split_model_.fleet()->dev(qwen35_split_model_.head_dev()).queue()
            : alloc_.queue();
        if (stepper_.d_pick)   sycl::free(stepper_.d_pick, hq);
        if (stepper_.d_recent) sycl::free(stepper_.d_recent, hq);
    }
    if (!alloc_.ready()) return;
    auto& q = alloc_.queue();
    // Free the dspark drafter (device weights + scratch) while alloc_ is still live.
    dspark_drafter_.reset();
    if (d_logits_) sycl::free(d_logits_, q);
    if (d_ids_)    sycl::free(d_ids_, q);
    if (d_pick_)   sycl::free(d_pick_, q);
    if (d_recent_) sycl::free(d_recent_, q);
}

void Engine::stepper_spawn_() {
    stepper_.slots.assign(opts_.parallel, {});
    sycl::queue& hq = qwen35_split_model_.fleet()
                          ->dev(qwen35_split_model_.head_dev()).queue();
    stepper_.d_pick   = sycl::malloc_device<int32_t>(opts_.parallel, hq);
    stepper_.d_recent = sycl::malloc_device<int32_t>(512, hq);
    if (!stepper_.d_pick || !stepper_.d_recent) {
        // ~2 KB — failure here means the card is beyond full anyway. Without
        // the scratch the stepper cannot sample; run without banks instead
        // (generate() falls back to the time-slice path when n_slot_banks()==0).
        std::fprintf(stderr, "[batch] stepper scratch alloc failed — "
                             "falling back to time-slice scheduling\n");
        qwen35_split_model_.free_slot_banks();
        return;
    }
    stepper_.th = std::thread([this] { stepper_loop_(); });
}

void Engine::stepper_loop_() {
    auto& model = qwen35_split_model_;
    const uint32_t V = model.config().dense.vocab;
    sycl::queue& hq = model.fleet()->dev(model.head_dev()).queue();
    constexpr uint32_t kMaxRepeatWindow = 512;
    struct RowCtx {
        SamplingParams sp;
        uint64_t rng;
        uint32_t step;
        uint32_t nw;
        int32_t  win[kMaxRepeatWindow];
    };
    std::vector<RowCtx>   rows;
    std::vector<uint32_t> idx, banks, poss;
    std::vector<int32_t>  toks, picks;

    std::unique_lock<std::mutex> lk(stepper_.mu);
    for (;;) {
        stepper_.cv.wait(lk, [&] {
            if (stepper_.shutdown) return true;
            for (auto& s : stepper_.slots)
                if (s.running && !s.finished) return true;
            return false;
        });
        if (stepper_.shutdown) return;

        // Build the step group + snapshot everything sampling needs, so no
        // slot field is touched outside the lock.
        idx.clear(); banks.clear(); poss.clear(); toks.clear(); rows.clear();
        for (uint32_t i = 0; i < stepper_.slots.size(); ++i) {
            auto& s = stepper_.slots[i];
            if (!s.running || s.finished) continue;
            if (s.want_stop) {
                s.finished = true;
                s.finish_reason = "abort";
                continue;
            }
            idx.push_back(i);
            banks.push_back(i);          // bank id == slot index
            poss.push_back(s.pos);
            toks.push_back(s.pending);
            RowCtx rc;
            rc.sp = s.sp; rc.rng = s.rng; rc.step = s.made;
            rc.nw = 0;
            if (s.sp.repeat_penalty != 1.0f || s.sp.presence_penalty != 0.f || s.sp.frequency_penalty != 0.f) {
                const uint32_t total = uint32_t(s.hist.size());
                rc.nw = std::min({s.sp.repeat_window, total, kMaxRepeatWindow});
                for (uint32_t k = 0; k < rc.nw; ++k)
                    rc.win[k] = s.hist[total - rc.nw + k];
            }
            rows.push_back(rc);
        }
        if (idx.empty()) { stepper_.cv.notify_all(); continue; }
        lk.unlock();

        gate_.acquire();
        std::string err = model.forward_slots(uint32_t(idx.size()), toks.data(),
                                              poss.data(), banks.data(),
                                              /*out_logits_host=*/nullptr);
        picks.assign(idx.size(), 0);
        if (err.empty()) {
            sycl::half* lg = model.slot_logits_dev();
            for (size_t r = 0; r < idx.size(); ++r) {
                const RowCtx& rc = rows[r];
                sycl::half* row = lg + r * uint64_t(V);
                if (rc.nw > 0) {
                    hq.memcpy(stepper_.d_recent, rc.win, rc.nw * sizeof(int32_t));
                    sampling_penalties(hq, row, V, stepper_.d_recent, rc.nw,
                                       rc.sp.repeat_penalty, rc.sp.presence_penalty, rc.sp.frequency_penalty);
                }
                sample_softmax_topk_topp(hq, row, stepper_.d_pick + r, V,
                                         rc.sp.temperature, rc.sp.top_k,
                                         rc.sp.top_p, rc.sp.min_p,
                                         rc.rng + rc.step);
            }
            hq.memcpy(picks.data(), stepper_.d_pick,
                      idx.size() * sizeof(int32_t)).wait();
        }
        gate_.release();

        lk.lock();
        for (size_t r = 0; r < idx.size(); ++r) {
            auto& s = stepper_.slots[idx[r]];
            if (!err.empty()) {
                s.finished = true;
                s.finish_reason = "error: slot-step: " + err;
                continue;
            }
            const int32_t pick = picks[r];
            s.pos  += 1;
            s.made += 1;
            // Solo-loop semantics: EOS/stop tokens end the generation WITHOUT
            // being delivered; any other pick is delivered, then the budget
            // check may end it with "length".
            if (!s.sp.ignore_eos &&
                (pick == tok_.eos_token_id() || pick == stop_ids_[0] ||
                 pick == stop_ids_[1])) {
                s.finished = true;
                s.finish_reason = "stop";
                continue;
            }
            s.hist.push_back(pick);
            s.pending = pick;
            s.outbox.push_back(pick);
            if (s.made >= s.budget) {
                s.finished = true;
                s.finish_reason = "length";
            }
        }
        stepper_.cv.notify_all();
    }
}

namespace {
// Restore the longest cached prefix into a multi-GPU hybrid model's per-card
// KvCache + DeltaNetState. Returns the restored depth M (>0) on FULL success; 0
// on a miss OR any per-card failure — the caller then does a clean pos=0 full
// prefill, which resets every card via reset_kv=(pos==0), wiping any half-restore.
// Shared by the 80B-next and crown-Q8-split paths (identical per-card contract).
template <class Model>
uint32_t fleet_cache_restore(Model& m, FleetPrefixCache& cache,
                             const std::vector<int32_t>& ids, const char* tag) {
    auto hit = cache.find_longest_match(ids);
    const uint32_t cap = uint32_t(ids.size()) - 1;   // always leave >=1 token to prefill
    const uint32_t M   = std::min<uint32_t>(hit.match_len, cap);
    if (!(M > 0 && hit.kv && hit.dn)) return 0;
    // DeltaNet safety: dn_state.copy_from restores the FULL recurrent state at the
    // snapshot depth (== hit.match_len) — it has no positional prefix, so KV must land
    // at that SAME depth. If `cap` clamped M below match_len (the exact full-length
    // repeat, match_len == ids.size()), restoring KV@M with DN@match_len would
    // double-apply the last token in the recurrence → corrupt state. Skip (full
    // prefill) for DeltaNet-bearing models; KV-only models are position-keyed and safe.
    if (M < hit.match_len)
        for (uint32_t d = 0; d < m.n_devices(); ++d)
            if (m.dev_has_dn(d)) return 0;
    DeviceFleet* fleet = m.fleet();
    std::string emsg;
    for (uint32_t dev = 0; dev < m.n_devices(); ++dev) {
        sycl::queue& dq = fleet->dev(dev).queue();
        if (m.dev_has_kv(dev)) {
            if (!(*hit.kv)[dev]) { emsg = "missing kv snapshot"; break; }
            if (auto e = m.kv_cache(dev).copy_prefix_from(dq, *(*hit.kv)[dev], M); !e.empty()) { emsg = e; break; }
        }
        if (m.dev_has_dn(dev)) {
            if (!(*hit.dn)[dev]) { emsg = "missing dn snapshot"; break; }
            if (auto e = m.dn_state(dev).copy_from(dq, *(*hit.dn)[dev]); !e.empty()) { emsg = e; break; }
        }
    }
    if (emsg.empty()) return M;
    std::fprintf(stderr, "[%s] restore failed (%s); full prefill\n", tag, emsg.c_str());
    return 0;
}

// Phase 1b: per-slot suspend/resume state for a mid-flight generation on the
// 27B split path. Suspended state goes to HOST RAM, not VRAM: at agent depth a
// suspended slot's KV is ~0.7 GB/card and up to 3 slots may be suspended at
// once, which does not fit beside a live long-ctx cache; host RAM absorbs it
// and the PCIe round-trip on in-order queues is cheap relative to a decode
// slice. Slot-owned (never in the LRU prompt-cache trie), so an active
// generation can never be evicted. Vectors resize exactly per stash (capacity
// is retained, so steady-state stashes do not reallocate).
struct HostSlotStash {
    struct Card {
        std::vector<uint8_t> k, v;      // [L_full][n_kv_h][depth][head_dim] fp16, packed
        std::vector<uint8_t> dns, dnc;  // DeltaNet fp32 state + fp16 conv, flat
    };
    std::vector<Card> cards;
    uint32_t depth = 0;

    // Snapshot the model's live per-card state at depth N (state covers tokens
    // [0..N)). K/V rows are copied per (layer, kv_head) slice — the live cache
    // is strided by max_ctx, the stash packs by N. int8-kv is not supported
    // (--parallel>1 + --int8-kv is refused at CLI parse). Returns "" on
    // success; on failure depth stays 0 and the live state is untouched.
    template <class Model>
    std::string stash(Model& m, uint32_t N) {
        depth = 0;
        DeviceFleet* fleet = m.fleet();
        cards.resize(m.n_devices());
        for (uint32_t dev = 0; dev < m.n_devices(); ++dev) {
            sycl::queue& q = fleet->dev(dev).queue();
            Card& c = cards[dev];
            if (m.dev_has_kv(dev)) {
                const KvCacheConfig& kc = m.kv_cache(dev).config();
                if (kc.use_int8) return "slot stash: int8-kv unsupported";
                const uint64_t slice  = uint64_t(N) * kc.head_dim * sizeof(sycl::half);
                const uint64_t stride = uint64_t(kc.max_ctx) * kc.head_dim;   // halves
                const uint64_t nsl    = uint64_t(kc.n_layers_full) * kc.n_kv_heads;
                c.k.resize(nsl * slice);
                c.v.resize(nsl * slice);
                const sycl::half* ks = m.kv_cache(dev).k_ptr();
                const sycl::half* vs = m.kv_cache(dev).v_ptr();
                for (uint64_t s = 0; s < nsl; ++s) {
                    q.memcpy(c.k.data() + s * slice, ks + s * stride, slice);
                    q.memcpy(c.v.data() + s * slice, vs + s * stride, slice);
                }
            }
            if (m.dev_has_dn(dev)) {
                const DeltaNetState& d = m.dn_state(dev);
                const uint64_t sb = d.state_elems_per_layer() *
                                    d.config().n_layers_linear * sizeof(float);
                const uint64_t cb = d.conv_elems_per_layer() *
                                    d.config().n_layers_linear * sizeof(sycl::half);
                c.dns.resize(sb);
                c.dnc.resize(cb);
                q.memcpy(c.dns.data(), d.state_ptr(), sb);
                q.memcpy(c.dnc.data(), d.conv_state_ptr(), cb);
            }
            q.wait();   // d2h copies must land before the gate is released
        }
        depth = N;
        return {};
    }

    // Bitwise compare of two stashes at the same depth — the ground truth for
    // suspend/resume losslessness (generation-level text comparison cannot
    // serve: decode kernels have per-invocation nondeterminism, see the
    // campaign doc; copies either preserve bytes or they don't). Vectors are
    // exact-sized per stash, so whole-vector compares are valid.
    static std::string equal(const HostSlotStash& a, const HostSlotStash& b,
                             bool& eq) {
        eq = false;
        if (a.depth == 0 || a.depth != b.depth) return "equal: depth mismatch";
        if (a.cards.size() != b.cards.size())   return "equal: card-count mismatch";
        for (size_t i = 0; i < a.cards.size(); ++i) {
            const Card& x = a.cards[i];
            const Card& y = b.cards[i];
            if (x.k.size() != y.k.size() || x.dns.size() != y.dns.size())
                return "equal: payload-shape mismatch card " + std::to_string(i);
            if (x.k != y.k || x.v != y.v || x.dns != y.dns || x.dnc != y.dnc)
                return {};
        }
        eq = true;
        return {};
    }

    // Restore the stashed state into the model's live per-card caches and set
    // the per-layer KV lengths to the stash depth (the device-path
    // copy_prefix_from does this internally; a raw h2d copy must do it
    // explicitly). KV rows beyond `depth` are left stale — kernels never read
    // past the current position (same contract as the prefix cache).
    template <class Model>
    std::string unstash(Model& m) {
        if (depth == 0) return "slot unstash: nothing stashed";
        DeviceFleet* fleet = m.fleet();
        for (uint32_t dev = 0; dev < m.n_devices(); ++dev) {
            sycl::queue& q = fleet->dev(dev).queue();
            Card& c = cards[dev];
            if (m.dev_has_kv(dev)) {
                KvCache& live = m.kv_cache(dev);
                const KvCacheConfig& kc = live.config();
                const uint64_t slice  = uint64_t(depth) * kc.head_dim * sizeof(sycl::half);
                const uint64_t stride = uint64_t(kc.max_ctx) * kc.head_dim;
                const uint64_t nsl    = uint64_t(kc.n_layers_full) * kc.n_kv_heads;
                for (uint64_t s = 0; s < nsl; ++s) {
                    q.memcpy(live.k_ptr() + s * stride, c.k.data() + s * slice, slice);
                    q.memcpy(live.v_ptr() + s * stride, c.v.data() + s * slice, slice);
                }
                for (uint32_t l = 0; l < kc.n_layers_full; ++l) live.set_length(l, depth);
            }
            if (m.dev_has_dn(dev)) {
                DeltaNetState& d = m.dn_state(dev);
                q.memcpy(d.state_ptr(), c.dns.data(), c.dns.size());
                q.memcpy(d.conv_state_ptr(), c.dnc.data(), c.dnc.size());
            }
            q.wait();   // h2d must land before the holder's next forward
        }
        return {};
    }
};
}  // namespace

GenerateResult Engine::generate(const std::string& prompt,
                                const SamplingParams& sp,
                                const TokenCallback& on_token,
                                uint32_t cache_prefix_len,
                                bool reply_cache) {
    if (arch_ == ModelArch::kDeepSeek41) return ds41_generate(prompt, sp, on_token);
    if (arch_ == ModelArch::kMimo26) return mimo26_generate(prompt, sp, on_token);
    GenerateResult res;
    auto& q = alloc_.queue();
    q4e_vis_active_ = false;   // re-armed below iff this request carries images
    ds4_vis_active_ = false;
    auto ids = tok_.encode(prompt, /*allow_special=*/true);
    if (ds4_ && !ds4_pending_imgs_.empty()) {
        if (auto e = ds4_expand_vision(ids); !e.empty()) {
            ds4_pending_imgs_.clear();
            res.finish_reason = "error: " + e;
            return res;
        }
    }
    if (std::getenv("IE_DUMP_PROMPT")) {
        std::fprintf(stderr, "[prompt %zu toks]", ids.size());
        for (size_t i = 0; i < ids.size() && i < 24; ++i) std::fprintf(stderr, " %d", ids[i]);
        std::fprintf(stderr, "\n[prompt text] %.300s\n", prompt.c_str());
    }
    res.prompt_tokens = uint32_t(ids.size());
    const uint32_t budget = opts_.max_ctx - 8;
    // Prompt alone exceeds the context window: no generation is possible.
    // Distinct from "length" (generation hit max_tokens) so the server can
    // surface a clear context_length_exceeded error and the agent can compact
    // + retry, instead of receiving an empty "length" completion that reads as
    // a generic failure. Does NOT cap context — budget is the full max_ctx.
    if (ids.size() >= budget) { res.finish_reason = "context_length_exceeded"; return res; }
    if (ids.empty()) { res.finish_reason = "length"; return res; }

    // --- Multi-slot gate (Phase 1b): all GPU work below runs under the FIFO
    // ticket lock; the 27B-split prefill/decode loops yield it at slice
    // boundaries when contended. RAII so every return path releases. Solo
    // requests pay one uncontended lock — nothing else changes for them. ---
    gate_.acquire();
    struct GateGuard {
        Gate* g;
        bool held = true;
        void release_early() { if (held) { g->release(); held = false; } }
        ~GateGuard() { if (held) g->release(); }
    } gate_guard_{&gate_};

    glm5_err_.clear(); // request-local reset under the generation gate
    // The other two forward-failure latches are cleared here too, UNDER the
    // gate: cleared before it, a queued request (--parallel > 1) could wipe a
    // running one's latched error between its forward and its check.
    ds4_err_.clear();   // deepseek4 latch (inert on every other arch)
    q4e_err_.clear();   // qwen4exp latch, same contract

    // Vision staging (qwen4exp): encode the images chat() left pending against
    // this prompt's pad runs. GPU work — runs under the gate like the rest.
    if (q4e_ && !q4e_pending_imgs_.empty()) {
        if (auto e = q4e_stage_vision(ids); !e.empty()) {
            q4e_pending_imgs_.clear();
            res.finish_reason = "error: " + e;
            return res;
        }
    }
    if (ds4_ && ds4_vis_active_) {
        if (auto e = ds4_stage_vision(); !e.empty()) {
            ds4_pending_imgs_.clear();
            ds4_vis_active_ = false;
            res.finish_reason = "error: " + e;
            return res;
        }
    }

    // --- dspark separate-draft GREEDY spec decode (single-GPU kQwen35Dense,
    //     --spec-draft). Lossless (greedy only); temperature>0 falls through to
    //     plain decode. Takes precedence over --spec (MTP) when both are set. ---
    if (spec_draft_ && dspark_drafter_ && dspark_drafter_->loaded &&
        arch_ == ModelArch::kQwen35Dense && !qwen35_split_ && !qwen35_tp_ &&
        sp.temperature == 0.0f &&
        (sp.repeat_window == 0 || (sp.repeat_penalty == 1.f && sp.presence_penalty == 0.f && sp.frequency_penalty == 0.f))) {
        const uint32_t H    = qwen35_.config().dense.hidden;
        const uint32_t V    = vocab();
        const uint32_t nc   = dspark_drafter_->n_capture;
        const uint32_t ncap = dspark_drafter_->n_embd_cap;   // nc * H
        const uint32_t P0   = uint32_t(ids.size());

        // Full prefill (spec v1 does not use the prompt cache), exporting the
        // dspark conditioning taps for EVERY prompt position into the cumulative
        // host feed (positions 0..P0-1). Grown in place each round thereafter.
        kv_.reset(); dn_.reset(q);
        q.memcpy(d_ids_, ids.data(), ids.size() * sizeof(int32_t)).wait();
        uint32_t pf_chunk = std::min<uint32_t>(512u, opts_.max_ctx);
        if (const char* c = std::getenv("IE_QWEN35_PREFILL_CHUNK")) {
            const int v = std::atoi(c);
            if (v >= 1 && uint32_t(v) <= opts_.max_ctx) pf_chunk = uint32_t(v);
        }
        auto* d_taps_pf = sycl::malloc_device<sycl::half>(uint64_t(nc) * pf_chunk * H, q);
        if (!d_taps_pf) { res.finish_reason = "error: spec-draft prefill alloc"; return res; }

        const uint32_t reserve_rows = P0 + std::min<uint32_t>(sp.max_tokens, budget - P0);
        std::vector<float>   feat;  feat.reserve(size_t(reserve_rows) * ncap);
        std::vector<int32_t> pos;   pos.reserve(reserve_rows);
        std::vector<sycl::half> chunk_taps(size_t(nc) * pf_chunk * H);

        const auto t_pf0 = std::chrono::steady_clock::now();
        uint32_t cpos = 0;
        while (cpos < P0) {
            const uint32_t n = std::min<uint32_t>(pf_chunk, P0 - cpos);
            qwen35_.forward(q, d_ids_ + cpos, n, cpos, kv_, dn_, d_logits_,
                            /*all_logits=*/nullptr, /*hidden_pre_norm=*/nullptr,
                            /*ckpt=*/nullptr, /*taps_out=*/d_taps_pf).wait();
            q.memcpy(chunk_taps.data(), d_taps_pf,
                     uint64_t(nc) * n * H * sizeof(sycl::half)).wait();
            const size_t base = cpos;   // rows already staged (== positions done)
            feat.resize((base + n) * size_t(ncap));
            pos.resize(base + n);
            for (uint32_t r = 0; r < n; ++r) {
                float* dst = feat.data() + (base + r) * size_t(ncap);
                for (uint32_t c = 0; c < nc; ++c) {
                    const sycl::half* src = chunk_taps.data() + (uint64_t(c) * n + r) * H;
                    float* dcol = dst + size_t(c) * H;
                    for (uint32_t h = 0; h < H; ++h) dcol[h] = float(src[h]);
                }
                pos[base + r] = int32_t(cpos + r);
            }
            cpos += n;
        }
        sycl::free(d_taps_pf, q);
        res.prefill_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_pf0).count();

        // target argmax @ last prompt position (d_logits_ holds the last row) → the
        // round-1 anchor (identical to the plain-greedy first generated token).
        std::vector<sycl::half> lrow(V);
        q.memcpy(lrow.data(), d_logits_, uint64_t(V) * sizeof(sycl::half)).wait();
        int32_t tn = 0; { float best = float(lrow[0]);
            for (uint32_t v = 1; v < V; ++v) { float x = float(lrow[v]); if (x > best) { best = x; tn = int32_t(v); } } }

        std::vector<int32_t> generated;
        std::string decoded_all;
        size_t emitted = 0;
        res.finish_reason = "length";
        const uint32_t max_new = std::min<uint32_t>(sp.max_tokens, budget - P0);
        const auto t_dec0 = std::chrono::steady_clock::now();
        auto emit = [&](int32_t pick) -> bool {
            if (!sp.ignore_eos &&
                (pick == tok_.eos_token_id() || pick == stop_ids_[0] || pick == stop_ids_[1])) {
                res.finish_reason = "stop"; return false;
            }
            generated.push_back(pick);
            decoded_all = tok_.decode(std::span<const int32_t>(generated), /*skip_special=*/true,
                                      /*keep_special=*/std::span<const int32_t>(tool_call_ids_, 2));
            const size_t safe = utf8_complete_prefix_len(decoded_all);
            if (safe > emitted && on_token) {
                if (!on_token(std::string_view(decoded_all).substr(emitted, safe - emitted))) {
                    res.finish_reason = "abort"; emitted = safe; return false;
                }
                emitted = safe;
            }
            return true;
        };
        Qwen35DenseModel::SpecStats stats;
        std::string serr = qwen35_.spec_generate_dspark(q, kv_, dn_, *dspark_drafter_,
                                                        feat, pos, tn, P0, max_new, emit, &stats);
        res.decode_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_dec0).count();
        if (!serr.empty() && res.finish_reason != "abort" && res.finish_reason != "stop")
            std::fprintf(stderr, "[spec] %s\n", serr.c_str());
        std::fprintf(stderr,
            "[spec-draft] rounds=%llu drafted=%llu accepted=%llu bonus=%llu tau=%.3f\n",
            (unsigned long long)stats.rounds, (unsigned long long)stats.drafted,
            (unsigned long long)stats.accepted, (unsigned long long)stats.bonus, stats.tau);
        if (on_token && decoded_all.size() > emitted)
            on_token(std::string_view(decoded_all).substr(emitted));
        res.text = std::move(decoded_all);
        res.completion_tokens = uint32_t(generated.size());
        return res;
    }

    // --- MTP self-speculative GREEDY decode on the 2-card fleet paths (--spec) ---
    // Same lossless-greedy contract as the single-GPU branch below; each fleet
    // backend owns its prefill + draft/verify loop and per-card checkpoints.
    // Context cutoff: spec verify re-reads the KV K× per round, so the
    // weight-amortization win decays with context and eventually inverts.
    //
    // The original 2048 (2026-08-15) was measured while the SPEC prefill path was
    // silently running 256-token chunks (the big-chunk fix applied only to the
    // non-spec path — see the spec_pf comment below), which made spec look far
    // worse than it was. Re-measured 2026-08-23 with prefill fixed, TP+spec K3,
    // prefill matched in both arms so this is a clean decode comparison:
    //   3.5K  17.26 -> 22.59  (+31%)
    //   5K    16.35 -> 17.31  (+5.9%)
    //   8K    15.07 -> 12.37  (-18%)
    //   13K   12.89 -> 11.01  (-15%)
    // The true crossover is ~6K. 4096 sits inside the winning region with margin
    // rather than on the boundary. Above the cutoff, fall through to plain decode
    // (still greedy; still correct).
    static const uint32_t spec_max_prompt = []{
        const char* e = std::getenv("IE_QWEN35_SPEC_MAX_PROMPT");
        return (e && std::atoi(e) > 0) ? uint32_t(std::atoi(e)) : 4096u;
    }();
    const bool fleet_spec_loaded =
        (qwen35_split_ && qwen35_split_model_.mtp_loaded()) ||
        (qwen35_tp_ && qwen35_tp_model_.mtp_loaded());
    if (spec_ && fleet_spec_loaded &&
        arch_ == ModelArch::kQwen35Dense && sp.temperature == 0.0f &&
        (sp.repeat_window == 0 || (sp.repeat_penalty == 1.f && sp.presence_penalty == 0.f && sp.frequency_penalty == 0.f)) &&
        uint32_t(ids.size()) <= spec_max_prompt) {
        const uint32_t P0 = uint32_t(ids.size());
        std::vector<int32_t> generated;
        std::string decoded_all;
        size_t emitted = 0;
        res.finish_reason = "length";
        const uint32_t max_new = std::min<uint32_t>(sp.max_tokens, budget - P0);
        const uint32_t K = std::min<uint32_t>(16u, std::max<uint32_t>(2u, opts_.spec_k));
        auto emit = [&](int32_t pick) -> bool {
            if (!sp.ignore_eos &&
                (pick == tok_.eos_token_id() || pick == stop_ids_[0] || pick == stop_ids_[1])) {
                res.finish_reason = "stop"; return false;
            }
            generated.push_back(pick);
            decoded_all = tok_.decode(std::span<const int32_t>(generated), /*skip_special=*/true,
                                      /*keep_special=*/std::span<const int32_t>(tool_call_ids_, 2));
            const size_t safe = utf8_complete_prefix_len(decoded_all);
            if (safe > emitted && on_token) {
                if (!on_token(std::string_view(decoded_all).substr(emitted, safe - emitted))) {
                    res.finish_reason = "abort"; emitted = safe; return false;
                }
                emitted = safe;
            }
            return true;
        };
        double pf_ms = 0.0;
        // Prefill chunk for the SPEC path. The tuned chunk selection further down
        // (§ TP big-chunk, 3072) computes a LOCAL value used only by the non-spec
        // path, while this call passed the raw option — default 256. Since the TP
        // model is only ever driven through spec_generate, that meant TP+spec
        // prefilled a 3,514-token prompt in 14 chunks and measured 315 tok/s
        // against the certified 894. Apply the same rule here, without overriding
        // a chunk the caller set explicitly.
        uint32_t spec_pf = std::max<uint32_t>(1u, opts_.prefill_chunk);
        if (opts_.prefill_chunk <= 256u) {           // i.e. still the default
            if (arch_ == ModelArch::kQwen35Dense && qwen35_tp_)
                spec_pf = std::min<uint32_t>(3072u, opts_.max_ctx);
            else if (arch_ == ModelArch::kQwen35Dense)
                spec_pf = std::min<uint32_t>(512u, opts_.max_ctx);   // split optimum
        }
        if (const char* c = std::getenv("IE_QWEN35_PREFILL_CHUNK")) {
            const int v = std::atoi(c);
            if (v >= 1 && uint32_t(v) <= opts_.max_ctx) spec_pf = uint32_t(v);
        }
        const auto t0 = std::chrono::steady_clock::now();
        std::string serr = qwen35_tp_
            ? qwen35_tp_model_.spec_generate(
                  ids.data(), P0, spec_pf, max_new, K, emit, &pf_ms)
            : qwen35_split_model_.spec_generate(
                  ids.data(), P0, spec_pf, max_new, K, emit, &pf_ms);
        const double total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        res.prefill_ms = pf_ms;
        res.decode_ms  = std::max(0.0, total_ms - pf_ms);
        if (!serr.empty() && res.finish_reason != "abort" && res.finish_reason != "stop")
            std::fprintf(stderr, qwen35_tp_ ? "[spec-tp] %s\n" : "[spec-split] %s\n",
                         serr.c_str());
        if (on_token && decoded_all.size() > emitted)
            on_token(std::string_view(decoded_all).substr(emitted));
        res.prompt_tokens = P0;
        res.text = std::move(decoded_all);
        res.completion_tokens = uint32_t(generated.size());
        return res;
    }

    // --- MTP self-speculative GREEDY decode (single-GPU kQwen35Dense, --spec) ---
    // Only when spec is loaded AND sampling is greedy (temperature == 0): the
    // lossless guarantee holds for argmax decode only. Any temperature > 0 falls
    // through to the normal sampled decode below (byte-identical to non-spec).
    if (spec_ && qwen35_.mtp_loaded() && arch_ == ModelArch::kQwen35Dense &&
        !qwen35_split_ && !qwen35_tp_ && sp.temperature == 0.0f &&
        (sp.repeat_window == 0 || (sp.repeat_penalty == 1.f && sp.presence_penalty == 0.f && sp.frequency_penalty == 0.f))) {
        const uint32_t H = vocab() ? qwen35_.config().dense.hidden : 0;
        const uint32_t V = vocab();
        // Full prefill (spec v1 does not use the prompt cache). Grab the
        // pre-output_norm hidden over all prompt positions → take the last row.
        kv_.reset(); dn_.reset(q);
        q.memcpy(d_ids_, ids.data(), ids.size() * sizeof(int32_t)).wait();
        uint32_t pf_chunk = std::min<uint32_t>(512u, opts_.max_ctx);
        if (const char* c = std::getenv("IE_QWEN35_PREFILL_CHUNK")) {
            const int v = std::atoi(c);
            if (v >= 1 && uint32_t(v) <= opts_.max_ctx) pf_chunk = uint32_t(v);
        }
        const uint32_t P0 = uint32_t(ids.size());
        auto* d_hid_pf = sycl::malloc_device<sycl::half>(uint64_t(P0) * H, q);
        auto* d_hlast  = sycl::malloc_device<sycl::half>(uint64_t(H), q);
        if (!d_hid_pf || !d_hlast) {
            if (d_hid_pf) sycl::free(d_hid_pf, q);
            if (d_hlast)  sycl::free(d_hlast, q);
            res.finish_reason = "error: spec prefill alloc"; return res;
        }
        const auto t_pf0 = std::chrono::steady_clock::now();
        // The hidden_pre_norm buffer must cover [0..P0); chunked prefill writes each
        // chunk's rows at the chunk's absolute offset (forward writes rows for the
        // T submitted positions starting at row 0 of the passed pointer → offset it).
        uint32_t pos = 0;
        while (pos < P0) {
            const uint32_t n = std::min<uint32_t>(pf_chunk, P0 - pos);
            qwen35_.forward(q, d_ids_ + pos, n, pos, kv_, dn_, d_logits_,
                            /*all_logits=*/nullptr,
                            /*hidden_pre_norm=*/d_hid_pf + uint64_t(pos) * H).wait();
            pos += n;
        }
        res.prefill_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_pf0).count();
        // target argmax @ last prompt position (d_logits_ holds the last row)
        std::vector<sycl::half> lrow(V);
        q.memcpy(lrow.data(), d_logits_, uint64_t(V) * sizeof(sycl::half)).wait();
        int32_t tn = 0; { float best = float(lrow[0]);
            for (uint32_t v = 1; v < V; ++v) { float x = float(lrow[v]); if (x > best) { best = x; tn = int32_t(v); } } }
        q.memcpy(d_hlast, d_hid_pf + uint64_t(P0 - 1) * H, uint64_t(H) * sizeof(sycl::half)).wait();
        sycl::free(d_hid_pf, q);

        std::vector<int32_t> generated;
        std::string decoded_all;
        size_t emitted = 0;
        res.finish_reason = "length";
        const uint32_t max_new = std::min<uint32_t>(sp.max_tokens, budget - P0);
        const uint32_t K = std::max<uint32_t>(1u, opts_.spec_k);
        const auto t_dec0 = std::chrono::steady_clock::now();
        // emit() is called per committed token IN ORDER. Returning false stops the
        // loop (stop token: not surfaced to the user, mirrors plain decode's break
        // before push_back). max_tokens is enforced inside spec_generate via max_new.
        auto emit = [&](int32_t pick) -> bool {
            if (!sp.ignore_eos &&
                (pick == tok_.eos_token_id() || pick == stop_ids_[0] || pick == stop_ids_[1])) {
                res.finish_reason = "stop"; return false;
            }
            generated.push_back(pick);
            decoded_all = tok_.decode(std::span<const int32_t>(generated), /*skip_special=*/true,
                                      /*keep_special=*/std::span<const int32_t>(tool_call_ids_, 2));
            const size_t safe = utf8_complete_prefix_len(decoded_all);
            if (safe > emitted && on_token) {
                if (!on_token(std::string_view(decoded_all).substr(emitted, safe - emitted))) {
                    res.finish_reason = "abort"; emitted = safe; return false;
                }
                emitted = safe;
            }
            return true;
        };
        std::string serr = qwen35_.spec_generate(q, kv_, dn_, d_hlast, tn, P0,
                                                 max_new, K, emit);
        sycl::free(d_hlast, q);
        res.decode_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_dec0).count();
        if (!serr.empty() && res.finish_reason != "abort" && res.finish_reason != "stop")
            std::fprintf(stderr, "[spec] %s\n", serr.c_str());
        if (on_token && decoded_all.size() > emitted)
            on_token(std::string_view(decoded_all).substr(emitted));
        res.text = std::move(decoded_all);
        res.completion_tokens = uint32_t(generated.size());
        return res;
    }

    // --- Gemma-4 MTP self-speculative GREEDY decode (kGemma4, --spec) ---
    // Same shape as the qwen35 path but: backbone-wide hidden, NO DeltaNet (gemma
    // self-manages KV; rollback is implicit), and the official gemma4-assistant head
    // (loaded above). Strictly token-lossless vs plain greedy; beats llama on B70.
    if (spec_ && gemma_model_.mtp_loaded() && arch_ == ModelArch::kGemma4 &&
        sp.temperature == 0.0f &&
        (sp.repeat_window == 0 || (sp.repeat_penalty == 1.f && sp.presence_penalty == 0.f && sp.frequency_penalty == 0.f))) {
        const uint32_t H = gemma_model_.hidden();   // backbone (e.g. 5376)
        const uint32_t V = vocab();
        q.memcpy(d_ids_, ids.data(), ids.size() * sizeof(int32_t)).wait();
        const uint32_t P0 = uint32_t(ids.size());
        auto* d_hid_pf = sycl::malloc_device<sycl::half>(uint64_t(P0) * H, q);
        auto* d_hlast  = sycl::malloc_device<sycl::half>(uint64_t(H), q);
        if (!d_hid_pf || !d_hlast) {
            if (d_hid_pf) sycl::free(d_hid_pf, q);
            if (d_hlast)  sycl::free(d_hlast, q);
            res.finish_reason = "error: spec prefill alloc"; return res;
        }
        const uint32_t pf_chunk = std::max<uint32_t>(1u, opts_.prefill_chunk);
        const auto t_pf0 = std::chrono::steady_clock::now();
        uint32_t pos = 0;
        while (pos < P0) {
            const uint32_t n = std::min<uint32_t>(pf_chunk, P0 - pos);
            gemma_model_.forward(q, d_ids_ + pos, n, pos, kv_, d_logits_,
                                 /*hidden_pre_norm=*/d_hid_pf + uint64_t(pos) * H,
                                 /*all_logits=*/nullptr).wait();
            pos += n;
        }
        res.prefill_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_pf0).count();
        std::vector<sycl::half> lrow(V);
        q.memcpy(lrow.data(), d_logits_, uint64_t(V) * sizeof(sycl::half)).wait();
        int32_t tn = 0; { float best = float(lrow[0]);
            for (uint32_t v = 1; v < V; ++v) { float x = float(lrow[v]); if (x > best) { best = x; tn = int32_t(v); } } }
        q.memcpy(d_hlast, d_hid_pf + uint64_t(P0 - 1) * H, uint64_t(H) * sizeof(sycl::half)).wait();
        sycl::free(d_hid_pf, q);

        std::vector<int32_t> generated;
        std::string decoded_all;
        size_t emitted = 0;
        res.finish_reason = "length";
        const uint32_t max_new = std::min<uint32_t>(sp.max_tokens, budget - P0);
        const uint32_t K = std::max<uint32_t>(1u, opts_.spec_k);
        const auto t_dec0 = std::chrono::steady_clock::now();
        auto emit = [&](int32_t pick) -> bool {
            if (!sp.ignore_eos &&
                (pick == tok_.eos_token_id() || pick == stop_ids_[0] || pick == stop_ids_[1])) {
                res.finish_reason = "stop"; return false;
            }
            generated.push_back(pick);
            decoded_all = tok_.decode(std::span<const int32_t>(generated), /*skip_special=*/true,
                                      /*keep_special=*/std::span<const int32_t>(tool_call_ids_, 2));
            const size_t safe = utf8_complete_prefix_len(decoded_all);
            if (safe > emitted && on_token) {
                if (!on_token(std::string_view(decoded_all).substr(emitted, safe - emitted))) {
                    res.finish_reason = "abort"; emitted = safe; return false;
                }
                emitted = safe;
            }
            return true;
        };
        std::string serr = gemma_model_.spec_generate(q, d_hlast, tn, P0, max_new, K, emit);
        sycl::free(d_hlast, q);
        res.decode_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_dec0).count();
        if (!serr.empty() && res.finish_reason != "abort" && res.finish_reason != "stop")
            std::fprintf(stderr, "[spec] %s\n", serr.c_str());
        if (on_token && decoded_all.size() > emitted)
            on_token(std::string_view(decoded_all).substr(emitted));
        res.text = std::move(decoded_all);
        res.completion_tokens = uint32_t(generated.size());
        return res;
    }

    // Prompt/KV cache (crown single-GPU; gated). Restore the longest cached prefix
    // instead of re-prefilling it. find_longest_match returns an endpoint whose
    // tokens are EXACTLY a prefix of `ids`, so its DeltaNet snapshot is valid at
    // that depth (state at pos N is a function of tokens [0..N)). Always leave >=1
    // token to prefill so d_logits_ holds the last prompt token's logits for the
    // first decode step (restore copies state, not logits). Default-OFF → the path
    // below is byte-identical to the prior stateless behavior.
    uint32_t restored = 0;
    if (prompt_cache_on_ && next_) {
        // 80B fleet path: restore the longest cached prefix into every card's live
        // kv_[dev]/dn_[dev]. Hit (M>0) → forward starts at pos=M≠0, reset_kv=false,
        // uses the restored state; miss/partial → restored=0, pos=0 resets all cards.
        restored = fleet_cache_restore(next_model_, fleet_cache_, ids, "fleet-cache");
    } else if (prompt_cache_on_ && qwen35moe_split_) {
        // Crown Q8 2-card split — identical per-card contract as the 80B path. This
        // is the fix for "0 cached every turn" on the daily-driver 2-card crown.
        restored = fleet_cache_restore(qwen35moe_split_model_, fleet_cache_, ids, "crown-split-cache");
    } else if (prompt_cache_on_ && gptoss_tp_) {
        // gpt-oss tensor-parallel — KV-only (no DeltaNet); the DN half of the restore
        // is a no-op (dev_has_dn=false on every card).
        restored = fleet_cache_restore(gptoss_tp_model_, fleet_cache_, ids, "gptoss-tp-cache");
    } else if (prompt_cache_on_ && q3moe_split_) {
        // qwen3moe (Coder/Tongyi) layer-split — KV-only.
        restored = fleet_cache_restore(q3moe_split_model_, fleet_cache_, ids, "q3moe-split-cache");
    } else if (prompt_cache_on_ && q3moe_tp_) {
        // qwen3moe (Coder/Tongyi) tensor-parallel — KV-only, head-sharded per card.
        restored = fleet_cache_restore(q3moe_tp_model_, fleet_cache_, ids, "q3moe-tp-cache");
    } else if (prompt_cache_on_ && qwen35_split_) {
        // 27B DeltaNet+dense layer-split — snapshots per-card KV + DeltaNet (like crown).
        restored = fleet_cache_restore(qwen35_split_model_, fleet_cache_, ids, "27b-split-cache");
    } else if (prompt_cache_on_ && arch_ == ModelArch::kGemma4) {
        // Gemma4 VRAM-guarded single-endpoint snapshot (task #6). LCP-match the
        // incoming ids against the stored snapshot tokens; restore that depth (leave
        // >=1 token to prefill so d_logits_ holds the last prompt token's logits).
        // Gemma self-manages KV and never resets in forward_step (attention reads
        // [0,pos]) → restoring [0,M) then prefilling from pos=M is exact (KV is purely
        // positional; the matched tokens are identical). Miss → M=0, full prefill from
        // pos=0 overwrites the cache. MUST precede the single-GPU crown branch below
        // (gemma is !tp_ && !next_ and would otherwise hit the QwenConfig-typed
        // prefix_cache_ that was never inited for gemma).
        const std::vector<int32_t>& snaptok = gemma_model_.snapshot_tokens();
        const uint32_t D = gemma_model_.snapshot_depth();
        if (D > 0 && !snaptok.empty()) {
            uint32_t lim = std::min<uint32_t>(D, uint32_t(snaptok.size()));
            lim = std::min<uint32_t>(lim, uint32_t(ids.size()));
            uint32_t lcp = 0;
            while (lcp < lim && ids[lcp] == snaptok[lcp]) ++lcp;
            const uint32_t cap = uint32_t(ids.size()) - 1;   // prefill >=1 token
            const uint32_t M   = std::min<uint32_t>(lcp, cap);
            if (M > 0) restored = gemma_model_.restore_kv(q, M);
        }
    } else if (prompt_cache_on_ && arch_ == ModelArch::kQwen4Exp) {
        // Same all-or-nothing contract as DS4 below: the DeltaNet/PLE state
        // cannot be truncated, so restore only when the snapshot tokens are a
        // STRICT prefix of the new prompt (strict → the volatile tail always
        // recomputes, so sampling never sees stale logits).
        const std::vector<int32_t>& snaptok = q4e_->snap_tokens;
        const uint32_t D = q4e_->A.snapshot_depth();
        if (D > 0 && snaptok.size() == size_t(D) && size_t(D) < ids.size()) {
            bool is_prefix = true;
            for (uint32_t i = 0; i < D && is_prefix; ++i)
                is_prefix = (ids[i] == snaptok[i]);
            if (is_prefix) {
                if (std::string rm = q4e_->rest(); rm.empty()) restored = D;
                else std::fprintf(stderr, "[q4e-cache] restore: %s\n", rm.c_str());
            }
        }
    } else if (prompt_cache_on_ && arch_ == ModelArch::kDeepSeek4) {
        // DeepSeek-V4 single-endpoint snapshot (runtime-owned device state, tokens
        // here). MUST precede the single-GPU branch below for the same reason
        // gemma4 does: DS4 is !tp_ && !next_ and would otherwise reach the
        // QwenConfig-typed prefix_cache_ that was never inited for it.
        //
        // ALL-OR-NOTHING, and that is the difference from every other branch here.
        // Gemma4 restores the first M rows of a positional KV slab; the fleet
        // caches copy a prefix of theirs. A DS4 snapshot cannot be truncated: the
        // CSA/HCA compressors carry a partially-filled window buffer, a monotone
        // emitted-entry count and a ping-pong overlap slice, and there is no
        // function from "state at depth D" to "state at depth M < D". So a
        // partial match restores NOTHING — restoring D rows of a conversation the
        // caller did not ask for would answer with somebody else's context and
        // never look like an error.
        const uint32_t cap = uint32_t(ids.size()) - 1;      // always prefill >= 1 token,
                                                            // so d_logits_ holds the last
                                                            // prompt token's logits
        // A vision request never restores (its image rows are not in the
        // cached state) and never snapshots (see the insert below).  Among the
        // host slots, the DEEPEST exact prefix wins.
        if (!ds4_vis_active_) {
            int best = -1; uint32_t bestD = 0;
            for (size_t k = 0; k < ds4_->slots.size(); ++k) {
                const auto& sl = ds4_->slots[k];
                const uint32_t D = uint32_t(sl.tokens.size());
                if (D == 0 || D > cap || D <= bestD) continue;
                bool is_prefix = true;
                for (uint32_t i = 0; i < D && is_prefix; ++i) is_prefix = (ids[i] == sl.tokens[i]);
                if (is_prefix) { best = int(k); bestD = D; }
            }
            if (best >= 0) {
                auto& sl = ds4_->slots[size_t(best)];
                if (ds4_->tp) restored = ds4_->tp->restore_from_host(sl.cards);
                else          restored = sl.cards.empty() ? 0u : ds4_->rt->restore_from_host(sl.cards[0]);
                if (restored == bestD) sl.stamp = ++ds4_->slot_clock;
                else { restored = 0; sl.tokens.clear(); }   // a broken slot never matches again
            }
        }
    } else if (prompt_cache_on_ && !tp_ && !next_) {
        auto hit = prefix_cache_.find_longest_match(ids);
        const uint32_t cap = uint32_t(ids.size()) - 1;          // prefill >=1 token
        // Crown here is the DeltaNet hybrid: dn_.copy_from restores the full recurrent
        // state at match_len, so a cap-clamped M (exact full-length repeat) would leave
        // KV one token behind DN → double-applied recurrence. Skip the restore in that
        // case (M<match_len) → clean full prefill.
        const uint32_t M   = (hit.match_len <= cap) ? hit.match_len : 0u;
        kv_.reset();
        dn_.reset(q);
        if (M > 0 && hit.kv && hit.dn) {
            std::string e1 = kv_.copy_prefix_from(q, *hit.kv, M);
            std::string e2 = e1.empty() ? dn_.copy_from(q, *hit.dn) : e1;
            if (e1.empty() && e2.empty()) {
                restored = M;                                   // suffix prefill starts here
            } else {
                kv_.reset(); dn_.reset(q);                      // partial restore → clean full prefill
                std::fprintf(stderr, "[prompt-cache] restore failed (%s); full prefill\n",
                             (!e1.empty() ? e1 : e2).c_str());
            }
        }
    } else if (!tp_) {
        // Stateless path: reset caches, full prefill. The TP path owns its per-card
        // KV and resets inside forward_step (reset_kv at pos==0), so kv_/dn_ (not
        // inited under tp_) are left alone. dn_.reset is a no-op for dense.
        kv_.reset();
        dn_.reset(q);
    }
    q.memcpy(d_ids_, ids.data(), ids.size() * sizeof(int32_t)).wait();
    // Prefill chunk. The global default is 256 (the §1 BMG DeltaNet-recurrence
    // HW-bug cap — docs/known_bugs.md). The gated-DeltaNet family now runs at 512:
    //   - qwen3next: EMPIRICALLY CLEAN at 512 (25/25 bit-identical; ie-qwen3next-ppl).
    //   - crown (qwen36 35B, kQwen35Moe) + 27B (qwen35, kQwen35Dense): RAISED to 512
    //     on 2026-06-20 after the §1 non-determinism was found NOT REPRODUCIBLE on
    //     NEO 26.14/26.18 + kernel 6.17.0-35: ie-bug-monitor 1024-iter ×3 (24 chains)
    //     = 0 divergence; crown PPL chunk-512 ×3 bit-identical (16.27) + no collapse
    //     at 512/1024.
    // A 512-tok single-call prefill is ~1.1× faster than 2×256 on >256-tok prompts
    // (closes the pp512 gap vs llama.cpp SYCL) at zero decode/short-prompt cost.
    // Revert to 256 (or A/B) via the env vars; never exceed max_ctx; 1024+ is NOT
    // yet determinism-validated (single clean run only).
    uint32_t pf_chunk = opts_.prefill_chunk;
    if (next_) {
        pf_chunk = std::min<uint32_t>(512u, opts_.max_ctx);
        // 80B oneDNN-MoE lever (Step 2c): feed a BIG prefill chunk so the MoE sees
        // large rows/expert (≈T·K/E), where per-expert oneDNN GEMM beats int-dot.
        // The DeltaNet recurrence is internally sub-chunked to ≤512 inside forward()
        // (§1-safe, Step 2a). OPT-IN IE_QWEN3NEXT_MOE_ONEDNN — multi-card-safe via the
        // per-device oneDNN ctx map (Step 1). M=T·10/512 needs ~8K for M≈160, so match
        // the chunk to the MoE T-gate. DEFAULT-ON for long-ctx (max_ctx>=8192) — the
        // clean-box A/B confirmed 1.53× over int-dot at M≈160 (919 vs 601 tok/s @8K,
        // new-old-new); opt-out IE_QWEN3NEXT_NO_MOE_ONEDNN; explicit IE_QWEN3NEXT_MOE_ONEDNN
        // force-enables below 8192 (testing). Short-ctx (<8192) stays 512 — unchanged.
        // IE_QWEN3NEXT_PREFILL_CHUNK still overrides either way.
        if (std::getenv("IE_QWEN3NEXT_NO_MOE_ONEDNN") == nullptr &&
            (opts_.max_ctx >= 8192u || std::getenv("IE_QWEN3NEXT_MOE_ONEDNN")))
            pf_chunk = std::min<uint32_t>(8192u, opts_.max_ctx);
        if (const char* c = std::getenv("IE_QWEN3NEXT_PREFILL_CHUNK")) {
            const int v = std::atoi(c);
            if (v >= 1 && uint32_t(v) <= opts_.max_ctx) pf_chunk = uint32_t(v);
        }
    } else if (arch_ == ModelArch::kQwen35Moe || arch_ == ModelArch::kQwen35Dense) {
        pf_chunk = std::min<uint32_t>(512u, opts_.max_ctx);
        // qwen35 TP (2026-08-16): no pipeline → big chunks amortize the per-chunk
        // weight dequant + gemm weight re-reads. Measured @2K: 295→500 tok/s
        // (chunk 512→2048/3072 + fa2 tile at 1024); @9K: 279→426 (chunk 3072).
        // 4096 REGRESSED @9K (223.9). Text byte-identical at 2K and 9K vs the
        // chunk-512 reference. Split keeps 512 (its pipeline's measured optimum).
        if (arch_ == ModelArch::kQwen35Dense && qwen35_tp_)
            pf_chunk = std::min<uint32_t>(3072u, opts_.max_ctx);
        // Crown (kQwen35Moe) long-ctx oneDNN-MoE lever: feed a BIG prefill chunk so
        // the MoE sees large rows/expert (oneDNN engages at T>=minT). The DeltaNet
        // recurrence is internally sub-chunked to <=512 launches inside forward()
        // (§1-safe). DEFAULT-ON for long-ctx configs (max_ctx>=8192, matching the
        // model's load-time buffer alloc); opt-out IE_QWEN36_NO_MOE_ONEDNN; explicit
        // IE_QWEN36_MOE_ONEDNN force-enables below 8192 (testing). 27B dense
        // (kQwen35Dense) has no MoE so it stays at 512. IE_QWEN35_PREFILL_CHUNK
        // still overrides.
        if (arch_ == ModelArch::kQwen35Moe &&
            std::getenv("IE_QWEN36_NO_MOE_ONEDNN") == nullptr &&
            (opts_.max_ctx >= 8192u || std::getenv("IE_QWEN36_MOE_ONEDNN")))
            pf_chunk = std::min<uint32_t>(8192u, opts_.max_ctx);
        if (const char* c = std::getenv("IE_QWEN35_PREFILL_CHUNK")) {
            const int v = std::atoi(c);
            if (v >= 1 && uint32_t(v) <= opts_.max_ctx) pf_chunk = uint32_t(v);
        }
    } else if (arch_ == ModelArch::kGptOss) {
        // gpt-oss is NOT DeltaNet → the §1 256-cap does not apply. The default-256
        // chunk was the prefill wall: each tiny chunk re-dequants all active experts
        // and pays the TP all-reduce/launch overhead. A big chunk amortizes both
        // across many tokens (~7× fewer expert-dequant passes on a long prompt).
        // 4096 is the measured sweet spot on 2×B70 @65k ctx: prefill 256→433 tok/s
        // (12.7× the T=1 path); 8192 spills the TP MoE workspace to host RAM and
        // REGRESSES (301 tok/s, decode thrashes). Override via IE_GPTOSS_PREFILL_CHUNK
        // (=256 reverts; lower it if VRAM-bound at very long ctx; the workspace grows ~T·K).
        pf_chunk = std::min<uint32_t>(4096u, opts_.max_ctx);
        if (const char* c = std::getenv("IE_GPTOSS_PREFILL_CHUNK")) {
            const int v = std::atoi(c);
            if (v >= 1 && uint32_t(v) <= opts_.max_ctx) pf_chunk = uint32_t(v);
        }
    }
    if (q4e_ && restored > 0) {
        // Restore path skips the pos==0 reset, so staging from any previous
        // request must be dropped by hand; this request's (if any) re-arms.
        q4e_->A.clear_vision();
        if (q4e_->split) q4e_->B.clear_vision();
        q4e_apply_vision();
    }
    res.cached_tokens = restored;
    // Snapshot boundary: the STABLE conversation depth (cache_prefix_len, no gen-prompt/
    // think suffix) so the next turn restores it; 0 → full depth (exact-repeat reuse).
    // The DeltaNet state is position-dependent, so we must snapshot when kv_/dn_ hold
    // EXACTLY the boundary state — hence split the prefill there.
    uint32_t snap_at = (prompt_cache_on_ && cache_prefix_len > 0)
        ? std::min<uint32_t>(cache_prefix_len, uint32_t(ids.size()))
        : uint32_t(ids.size());
    // DS4 only: never snapshot deeper than the VRAM cap set at load. Clamping the
    // BOUNDARY rather than skipping the insert means a long conversation still
    // caches its first ds4_cache_max_prefix_ tokens instead of caching nothing.
    if (prompt_cache_on_ && arch_ == ModelArch::kDeepSeek4 && ds4_cache_max_prefix_)
        snap_at = std::min<uint32_t>(snap_at, ds4_cache_max_prefix_);
    if (ds4_vis_active_) snap_at = uint32_t(ids.size());   // no snapshot boundary inside a block
    uint32_t pos = restored;            // cache restore (if any) already loaded [0..restored)
    // Per-request suspend buffer (host RAM) for gate yields — 27B split only.
    HostSlotStash slot_stash;
    // qwen4exp per-generation slots (A + B stage when split): same yield
    // contract as the 27B stash, over Qwen4ExpModel::stash_slot/unstash_slot
    // (byte-exactness gated by ie-qwen4exp-slot-test).
    Qwen4ExpModel::SlotState q4e_sa, q4e_sb;
    auto q4e_stash = [&](uint32_t depth) -> std::string {
        std::string e = q4e_->A.stash_slot(q4e_sa, depth);
        if (e.empty() && q4e_->split) e = q4e_->B.stash_slot(q4e_sb, depth);
        return e;
    };
    auto q4e_unstash = [&]() -> std::string {
        std::string e = q4e_->A.unstash_slot(q4e_sa);
        if (e.empty() && q4e_->split) e = q4e_->B.unstash_slot(q4e_sb);
        return e;
    };
    std::string slot_err;               // fatal yield failure (unstash after handoff)
    bool pf_abort = false;              // client left during prefill (liveness probe)
    const auto t_pf0 = std::chrono::steady_clock::now();
    auto prefill_to = [&](uint32_t end) {
        if (pf_abort) return;   // client left during an earlier range: nothing more runs
        // 27B-split PIPELINED prefill — DEFAULT ON since the 2026-08-15 A/B: 9K-token
        // prompt pp 219.5 → 424.7 (+93%, near the theoretical 2×), byte-identical
        // output (gated with oneDNN pinned off on both sides). Card 1 runs chunk c
        // while card 0 runs c+1; engages only past one chunk; serial fallback on any
        // error/topology mismatch. Kill switch: IE_QWEN35_NO_PIPELINE=1
        // (IE_QWEN35_PIPELINE=0 also disables).
        static const bool split_pipeline = []{
            if (std::getenv("IE_QWEN35_NO_PIPELINE")) return false;
            const char* e = std::getenv("IE_QWEN35_PIPELINE");
            return !e || std::atoi(e) != 0;
        }();
        // Contended gate → take the serial chunk loop below instead: it has
        // yield points between chunks, the pipelined range does not.
        if (split_pipeline && qwen35_split_ && !gate_.contended() &&
            end > pos && (end - pos) > pf_chunk) {
            if (auto m = qwen35_split_model_.forward_pipelined(
                    ids.data() + pos, end - pos, pos, /*reset_kv=*/(pos == 0),
                    pf_chunk, tp_logits_host_.data()); m.empty()) {
                pos = end;
                q.memcpy(d_logits_, tp_logits_host_.data(),
                         uint64_t(vocab()) * sizeof(sycl::half)).wait();
                return;
            } else {
                std::fprintf(stderr, "[pipeline] %s — serial fallback\n", m.c_str());
            }
        }
        // qwen4exp split: chunk-pipelined range (run2's certified overlap;
        // bit-identical to serial — same kernels, same order, exact fp32
        // handoff). Serial fallback on any error. IE_Q4E_NO_PIPELINE=1
        // disables.
        static const bool q4e_pipeline =
            std::getenv("IE_Q4E_NO_PIPELINE") == nullptr;
        if (q4e_pipeline && q4e_ && q4e_->split && !gate_.contended() &&
            end > pos && (end - pos) > 1024) {
            if (pos == 0) { q4e_->A.reset_state(); q4e_->B.reset_state(); }
            if (auto m = q4e_->fwd_pipelined(ids.data() + pos, end - pos, pos);
                m.empty()) {
                pos = end;
                const uint32_t V = q4e_vocab_;
                if (tp_logits_host_.size() < V) tp_logits_host_.resize(V);
                q4e_->B.queue().memcpy(tp_logits_host_.data(),
                                       q4e_->B.logits(),
                                       uint64_t(V) * sizeof(sycl::half)).wait();
                q.memcpy(d_logits_, tp_logits_host_.data(),
                         uint64_t(V) * sizeof(sycl::half)).wait();
                return;
            } else {
                std::fprintf(stderr, "[q4e-pipeline] serial fallback\n");
            }
        }
        while (pos < end && slot_err.empty() && !pf_abort) {
            uint32_t n = std::min<uint32_t>(pf_chunk, end - pos);
            if (ds4_vis_active_) {
                // Never cut an image block: the runtime refuses a straddling
                // chunk, so end this chunk right before a block that would not
                // fit whole. Blocks are <= pf_chunk (checked at expand time).
                for (const auto& sp : ds4_vis_spans_)
                    if (sp.t0 > pos && sp.t0 < pos + n && sp.t0 + sp.n > pos + n) n = sp.t0 - pos;
            }
            forward_step(q, d_ids_ + pos, n, pos).wait();
            pos += n;
            if (!glm5_err_.empty()) { slot_err="glm5next: "+glm5_err_; break; }
            // Client-liveness probe between chunks: an EMPTY fragment asks the
            // callback whether to continue, so a caller that disconnected during
            // a long prefill stops it here instead of after the first token.
            // (The pipelined split ranges above run whole and are not probed.)
            if (on_token && pos < end && !on_token(std::string_view{})) { pf_abort = true; break; }
            // Yield between prefill chunks when contended (27B split): bounds a
            // queued request's wait to ~one chunk instead of a whole long
            // prefill. d_ids_ is engine-shared — re-upload our remaining ids
            // after getting the gate back. A STASH failure skips the yield
            // (we still hold the gate, state untouched — safe to continue); an
            // UNSTASH failure after re-acquiring is fatal for this generation
            // (the live state belongs to another slot now) → slot_err aborts.
            if (qwen35_split_ && gate_.contended() && pos < end) {
                std::string se = slot_stash.stash(qwen35_split_model_, pos);
                if (se.empty()) {
                    gate_.release();
                    gate_.acquire();
                    se = slot_stash.unstash(qwen35_split_model_);
                    if (!se.empty()) { slot_err = se; break; }
                    if (pos < ids.size())
                        q.memcpy(d_ids_ + pos, ids.data() + pos,
                                 (ids.size() - pos) * sizeof(int32_t)).wait();
                } else {
                    std::fprintf(stderr, "[slot] prefill yield skipped (%s)\n",
                                 se.c_str());
                }
            }
            if (q4e_ && gate_.contended() && pos < end) {
                std::string se = q4e_stash(pos);
                if (se.empty()) {
                    gate_.release();
                    gate_.acquire();
                    se = q4e_unstash();
                    if (!se.empty()) { slot_err = se; break; }
                    if (pos < ids.size())
                        q.memcpy(d_ids_ + pos, ids.data() + pos,
                                 (ids.size() - pos) * sizeof(int32_t)).wait();
                } else {
                    std::fprintf(stderr, "[q4e-slot] prefill yield skipped (%s)\n",
                                 se.c_str());
                }
            }
        }
    };
    prefill_to(snap_at);
    // Snapshot the boundary state (kv_/dn_ or per-card kv_[dev]/dn_[dev] now hold
    // [0..snap_at)) — only if we actually prefilled new tokens up to it (snap_at >
    // restored). This else-if chain MIRRORS the restore dispatch above so exactly one
    // cache owns each model; the single-GPU prefix_cache_.insert must not fire on a
    // multi-GPU path (its kv_/dn_ are uninitialized there).
    if (prompt_cache_on_ && snap_at > restored && !pf_abort) {
        const std::vector<int32_t> pref(ids.begin(), ids.begin() + snap_at);
        std::string m;
        const char* tag = nullptr;
        if (next_)                   { m = fleet_cache_.insert(next_model_, pref);            tag = "fleet-cache"; }
        else if (qwen35moe_split_)   { m = fleet_cache_.insert(qwen35moe_split_model_, pref); tag = "crown-split-cache"; }
        else if (gptoss_tp_)         { m = fleet_cache_.insert(gptoss_tp_model_, pref);       tag = "gptoss-tp-cache"; }
        else if (q3moe_split_)       { m = fleet_cache_.insert(q3moe_split_model_, pref);     tag = "q3moe-split-cache"; }
        else if (q3moe_tp_)          { m = fleet_cache_.insert(q3moe_tp_model_, pref);        tag = "q3moe-tp-cache"; }
        else if (qwen35_split_)      { m = fleet_cache_.insert(qwen35_split_model_, pref);    tag = "27b-split-cache"; }
        else if (arch_ == ModelArch::kGemma4) { m = gemma_model_.snapshot_kv(q, snap_at, pref); tag = "gemma4-cache"; }
        else if (arch_ == ModelArch::kQwen4Exp) {
            // Never snapshot a vision request: a later prompt with the SAME pad
            // pattern but DIFFERENT image bytes would restore silently wrong KV
            // (token ids cannot distinguish the images).
            if (!q4e_vis_active_) {
                tag = "q4e-cache";
                m = q4e_->snap(snap_at);
                if (m.empty()) q4e_->snap_tokens = pref;
                else           q4e_->snap_tokens.clear();
            }
        }
        else if (arch_ == ModelArch::kDeepSeek4) {
            tag = "ds4-cache";
            // The runtime holds the device state; the tokens live beside it. They
            // are written ONLY on success, so a failed capture leaves no snapshot
            // AND no tokens claiming there is one — the restore above requires
            // snap_tokens.size() == snapshot_depth() and would refuse a mismatch.
            if (ds4_vis_active_) {
                // Never snapshot a vision request: the token ids cannot tell
                // one image's bytes from another's (same rule as qwen4exp).
                tag = nullptr;
            } else if (!ds4_->slots.empty()) {
                // The slot this conversation already lives in (its tokens are a
                // prefix of the new boundary), else an empty one, else the LRU.
                size_t pick = 0; bool found = false;
                for (size_t k = 0; k < ds4_->slots.size() && !found; ++k) {
                    const auto& sl = ds4_->slots[k];
                    if (sl.tokens.empty() || sl.tokens.size() > pref.size()) continue;
                    if (std::equal(sl.tokens.begin(), sl.tokens.end(), pref.begin())) { pick = k; found = true; }
                }
                for (size_t k = 0; k < ds4_->slots.size() && !found; ++k)
                    if (ds4_->slots[k].tokens.empty()) { pick = k; found = true; }
                if (!found)
                    for (size_t k = 1; k < ds4_->slots.size(); ++k)
                        if (ds4_->slots[k].stamp < ds4_->slots[pick].stamp) pick = k;
                auto& sl = ds4_->slots[pick];
                if (ds4_->tp) m = ds4_->tp->snapshot_to_host(sl.cards);
                else { sl.cards.resize(1); m = ds4_->rt->snapshot_to_host(sl.cards[0]); }
                if (m.empty()) { sl.tokens = pref; sl.stamp = ++ds4_->slot_clock; }
                else           sl.tokens.clear();
            }
        }
        else if (!tp_)               { m = prefix_cache_.insert(q, pref, kv_, dn_);           tag = "prompt-cache"; }
        if (!m.empty() && tag) std::fprintf(stderr, "[%s] insert: %s\n", tag, m.c_str());
    }
    prefill_to(uint32_t(ids.size()));   // volatile suffix (gen-prompt/think), if any
    res.prefill_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_pf0).count();
    // A failed deepseek4 prefill leaves d_logits_ holding whatever the last good
    // step wrote. Sampling it would produce text that reads like an answer, so
    // stop here with the runtime's own diagnostic instead.
    if (!ds4_err_.empty()) { res.finish_reason = "error: deepseek4: " + ds4_err_; return res; }
    if (!glm5_err_.empty()) { res.finish_reason="error: glm5next: "+glm5_err_; return res; }
    if (!q4e_err_.empty()) { res.finish_reason = "error: qwen4exp: " + q4e_err_; return res; }

    uint64_t rng = sp.seed ? sp.seed
        : uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
    std::vector<int32_t> generated;
    std::string decoded_all;
    size_t emitted = 0;
    res.finish_reason = "length";
    const uint32_t max_new = std::min<uint32_t>(sp.max_tokens, budget - uint32_t(ids.size()));
    // Decode slice between gate yields under contention (27B split). ~32 tokens
    // ≈ 2.3 s at 14 tok/s against a ~0.1 s host-stash round-trip at agent
    // depth. Override: IE_PARALLEL_SLICE.
    const uint32_t slice_tokens = [] {
        const char* e = std::getenv("IE_PARALLEL_SLICE");
        const int v = e ? std::atoi(e) : 0;
        return v > 0 ? uint32_t(v) : 32u;
    }();
    if (!slot_err.empty()) {
        res.finish_reason = "error: slot: " + slot_err;
        return res;
    }
    if (pf_abort) { res.finish_reason = "abort"; return res; }
    const auto t_dec0 = std::chrono::steady_clock::now();

    // --- Joint-step batched decode (Phase 2b). With slot banks allocated,
    // this request samples its FIRST token from the live prefill logits (same
    // math as step 0 of the loop below), banks its state, releases its gate
    // ticket, and consumes the rest of its tokens from the stepper. The
    // stash-yield loop below serves --parallel 1 (banks absent) AND any
    // request whose context is too deep for a slot bank — a prompt past
    // slot_ctx falls back to time-slicing instead of failing at bank_store.
    if (qwen35_split_ && qwen35_split_model_.n_slot_banks() > 0 &&
        uint64_t(ids.size()) + max_new + 8 <= qwen35_split_model_.slot_ctx()) {
        auto dec_ms = [&] {
            return std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_dec0).count();
        };
        if (sp.repeat_penalty != 1.0f || sp.presence_penalty != 0.f || sp.frequency_penalty != 0.f) {
            constexpr uint32_t kMaxRepeatWindow = 512;
            const uint32_t P  = uint32_t(ids.size());
            const uint32_t nw = std::min({sp.repeat_window, P, kMaxRepeatWindow});
            if (nw > 0) {
                int32_t win[kMaxRepeatWindow];
                for (uint32_t k = 0; k < nw; ++k) win[k] = ids[P - nw + k];
                q.memcpy(d_recent_, win, nw * sizeof(int32_t)).wait();
                sampling_penalties(q, d_logits_, vocab(), d_recent_, nw,
                                   sp.repeat_penalty, sp.presence_penalty, sp.frequency_penalty);
            }
        }
        sample_softmax_topk_topp(q, d_logits_, d_pick_, vocab(), sp.temperature,
                                 sp.top_k, sp.top_p, sp.min_p, rng).wait();
        int32_t pick = 0;
        q.memcpy(&pick, d_pick_, sizeof(pick)).wait();
        if (!sp.ignore_eos &&
            (pick == tok_.eos_token_id() || pick == stop_ids_[0] || pick == stop_ids_[1])) {
            res.finish_reason = "stop";
            res.decode_ms = dec_ms();
            return res;
        }
        generated.push_back(pick);
        decoded_all = tok_.decode(std::span<const int32_t>(generated), true,
                                  std::span<const int32_t>(tool_call_ids_, 2));
        {
            const size_t safe = utf8_complete_prefix_len(decoded_all);
            if (safe > emitted && on_token) {
                if (!on_token(std::string_view(decoded_all).substr(emitted, safe - emitted))) {
                    res.finish_reason = "abort";
                    res.text = std::move(decoded_all);
                    res.completion_tokens = 1;
                    res.decode_ms = dec_ms();
                    return res;
                }
                emitted = safe;
            }
        }
        if (max_new <= 1) {
            res.finish_reason = "length";
            res.text = std::move(decoded_all);
            res.completion_tokens = 1;
            res.decode_ms = dec_ms();
            return res;
        }
        // Claim a bank (admission bounds concurrency to the bank count).
        uint32_t bank = UINT32_MAX;
        {
            std::lock_guard<std::mutex> l(stepper_.mu);
            for (uint32_t i = 0; i < stepper_.slots.size(); ++i)
                if (!stepper_.slots[i].used) { bank = i; stepper_.slots[i].used = true; break; }
        }
        if (bank == UINT32_MAX) {
            res.finish_reason = "error: no free decode slot";
            res.decode_ms = dec_ms();
            return res;
        }
        struct BankRelease {
            Engine* e; uint32_t b;
            ~BankRelease() {
                std::lock_guard<std::mutex> l(e->stepper_.mu);
                e->stepper_.slots[b] = {};
            }
        } bank_rel{this, bank};
        if (auto m = qwen35_split_model_.bank_store(bank, pos); !m.empty()) {
            res.finish_reason = "error: bank-store: " + m;
            res.decode_ms = dec_ms();
            return res;
        }
        {
            std::lock_guard<std::mutex> l(stepper_.mu);
            auto& s = stepper_.slots[bank];
            s.running = true;
            s.pos = pos;
            s.pending = pick;
            s.sp = sp;
            s.rng = rng;
            s.made = 1;                       // step-0 token counted
            s.budget = max_new;
            s.hist.assign(ids.begin(), ids.end());
            s.hist.push_back(pick);
        }
        gate_guard_.release_early();          // the stepper takes the gate per step
        stepper_.cv.notify_all();
        std::string fin;
        {
            std::unique_lock<std::mutex> l(stepper_.mu);
            for (;;) {
                stepper_.cv.wait(l, [&] {
                    auto& s = stepper_.slots[bank];
                    return !s.outbox.empty() || s.finished;
                });
                bool got = false;
                {
                    auto& s = stepper_.slots[bank];
                    while (!s.outbox.empty()) {
                        generated.push_back(s.outbox.front());
                        s.outbox.pop_front();
                        got = true;
                    }
                }
                if (got) {
                    l.unlock();
                    decoded_all = tok_.decode(std::span<const int32_t>(generated), true,
                                              std::span<const int32_t>(tool_call_ids_, 2));
                    const size_t safe = utf8_complete_prefix_len(decoded_all);
                    bool aborted = false;
                    if (safe > emitted && on_token) {
                        if (!on_token(std::string_view(decoded_all).substr(emitted, safe - emitted)))
                            aborted = true;
                        else
                            emitted = safe;
                    }
                    l.lock();
                    if (aborted) stepper_.slots[bank].want_stop = true;
                }
                if (stepper_.slots[bank].finished) {
                    fin = stepper_.slots[bank].finish_reason;
                    break;
                }
            }
        }
        stepper_.cv.notify_all();             // wake the stepper past our slot
        if (on_token && decoded_all.size() > emitted)
            on_token(std::string_view(decoded_all).substr(emitted));
        res.text = std::move(decoded_all);
        res.completion_tokens = uint32_t(generated.size());
        res.finish_reason = fin.empty() ? "length" : fin;
        res.decode_ms = dec_ms();
        return res;
    }
    for (uint32_t step = 0; step < max_new; ++step) {
        if (sp.repeat_penalty != 1.0f || sp.presence_penalty != 0.f || sp.frequency_penalty != 0.f) {
            // llama.cpp-style penalty_last_n: the window is the last N tokens of
            // (prompt ids ++ generated), so it covers the PROMPT TAIL and fires
            // from the very first decode step — not just already-generated tokens.
            // d_recent_ is sized kMaxRepeatWindow; clamp nw to match.
            constexpr uint32_t kMaxRepeatWindow = 512;
            const uint32_t P = uint32_t(ids.size());
            const uint32_t G = uint32_t(generated.size());
            const uint32_t nw = std::min({sp.repeat_window, P + G, kMaxRepeatWindow});
            if (nw > 0) {
                int32_t win[kMaxRepeatWindow];
                const uint32_t start = (P + G) - nw;   // window = C[start .. P+G-1]
                for (uint32_t k = 0; k < nw; ++k) {
                    const uint32_t i = start + k;       // index into ids++generated
                    win[k] = (i < P) ? ids[i] : generated[i - P];
                }
                q.memcpy(d_recent_, win, nw * sizeof(int32_t)).wait();
                sampling_penalties(q, d_logits_, vocab(), d_recent_, nw, sp.repeat_penalty, sp.presence_penalty, sp.frequency_penalty);
            }
        }
        // Debug (IE_DEBUG_TOPK=1): for the first few steps dump the top-3 logits
        // AND (after sampling) the picked token + its RANK in the full distribution
        // — diagnoses the qwen35 temp>0 chat bug. rank ≤ top_k → legit; rank ≫ top_k
        // → the sampler selected outside the top-k/top-p set (selection bug). Zero
        // cost when the env is unset. NOTE: the sampler tombstones d_logits_, so the
        // host copy MUST be taken before sampling.
        static const bool dbg_topk = std::getenv("IE_DEBUG_TOPK") != nullptr;
        std::vector<float> dbg_logits;
        auto dbg_tok = [&](int32_t tid) {
            return tok_.decode(std::span<const int32_t>(&tid, 1), true,
                               std::span<const int32_t>{});
        };
        if (dbg_topk && step < 4) {
            std::vector<sycl::half> hl(vocab());
            q.memcpy(hl.data(), d_logits_, vocab() * sizeof(sycl::half)).wait();
            dbg_logits.resize(vocab());
            for (uint32_t i = 0; i < vocab(); ++i) dbg_logits[i] = float(hl[i]);
            std::vector<std::pair<float, int>> v(vocab());
            for (uint32_t i = 0; i < vocab(); ++i) v[i] = {dbg_logits[i], int(i)};
            std::partial_sort(v.begin(), v.begin() + 3, v.end(),
                              [](auto& a, auto& b) { return a.first > b.first; });
            std::fprintf(stderr, "[topk step %u] top-3:", step);
            for (int j = 0; j < 3; ++j)
                std::fprintf(stderr, "  (%.2f '%s')", v[j].first, dbg_tok(v[j].second).c_str());
            std::fprintf(stderr, "\n");
        }
        // In-order queue: the sampler is ordered after the forward that wrote
        // d_logits_, so the 4-byte pick readback below is the ONE host sync per
        // token (was four: sampler wait, pick D2H, pick H2D, forward wait).
        sample_softmax_topk_topp(q, d_logits_, d_pick_, vocab(),
                                 sp.temperature, sp.top_k, sp.top_p, sp.min_p,
                                 rng + step);
        int32_t pick = 0;
        q.memcpy(&pick, d_pick_, sizeof(pick)).wait();
        if (dbg_topk && step < 4 && !dbg_logits.empty()) {
            const float pl = dbg_logits[pick];
            uint32_t rank = 0;
            for (float x : dbg_logits) if (x > pl) ++rank;
            std::fprintf(stderr, "[topk step %u] PICKED id=%d logit=%.3f rank=%u '%s'\n",
                         step, pick, pl, rank, dbg_tok(pick).c_str());
        }
        if (!sp.ignore_eos &&
            (pick == tok_.eos_token_id() || pick == stop_ids_[0] || pick == stop_ids_[1])) {
            res.finish_reason = "stop"; break;
        }
        generated.push_back(pick);
        // Cooperative yield (Phase 1b): when other requests hold gate tickets,
        // hand the GPUs over at slice boundaries — stash to host RAM, release,
        // re-acquire (FIFO puts us behind the waiters), restore, continue. The
        // yield point is exact: state covers [0..pos), the sampled token is
        // appended but not yet forwarded, and the code below re-uploads it to
        // d_ids_ before the next forward regardless.
        //
        // Test harness: IE_TEST_FORCE_YIELD=K drives the same stash machinery
        // deterministically every K tokens — stash, CLOBBER the live state by
        // forwarding a wrong token (what another slot's activity does), then
        // unstash; IE_TEST_STASH_VERIFY=1 additionally re-stashes and
        // bit-compares (state-level losslessness proof, immune to the decode
        // kernels' per-invocation nondeterminism). Re-read per token (not a
        // static) so the test tool can toggle between generations.
        const uint32_t force_yield = [] {
            const char* e = std::getenv("IE_TEST_FORCE_YIELD");
            const int v = e ? std::atoi(e) : 0;
            return v > 0 ? uint32_t(v) : 0u;
        }();
        const bool test_yield = force_yield && ((step + 1) % force_yield) == 0;
        const bool real_yield = !force_yield && gate_.contended() &&
                                ((step + 1) % slice_tokens) == 0;
        // A yield (real or test) releases the gate: another request's sampler
        // may overwrite the engine-wide d_pick_ meanwhile, and the test path
        // clobbers d_ids_[pos] on purpose. Either way the token must then be
        // re-uploaded from the host `pick` below, not copied from d_pick_.
        const bool yielded = (qwen35_split_ || q4e_) && (test_yield || real_yield);
        if (qwen35_split_ && (test_yield || real_yield)) {
            std::string se = slot_stash.stash(qwen35_split_model_, pos);
            if (se.empty() && test_yield) {
                const int32_t wrong = (pick + 1 < int32_t(vocab())) ? pick + 1 : 0;
                q.memcpy(d_ids_ + pos, &wrong, sizeof(wrong)).wait();
                forward_step(q, d_ids_ + pos, /*T=*/1, pos).wait();
                se = slot_stash.unstash(qwen35_split_model_);
                if (se.empty() && std::getenv("IE_TEST_STASH_VERIFY")) {
                    static HostSlotStash re_stash;   // test-only, single-thread
                    se = re_stash.stash(qwen35_split_model_, pos);
                    bool eq = false;
                    if (se.empty())
                        se = HostSlotStash::equal(slot_stash, re_stash, eq);
                    if (se.empty() && !eq)
                        se = "restore not bit-exact at depth " + std::to_string(pos);
                    if (se.empty())
                        std::fprintf(stderr, "[stash-verify] depth %u OK\n", pos);
                }
            } else if (se.empty()) {
                gate_.release();
                gate_.acquire();
                se = slot_stash.unstash(qwen35_split_model_);
            }
            if (!se.empty()) {
                res.finish_reason = "error: slot-stash: " + se;
                break;
            }
        }
        if (q4e_ && (test_yield || real_yield)) {
            std::string se = q4e_stash(pos);
            if (se.empty() && test_yield) {
                // Corrupt-then-restore probe (engine-level restore check):
                // advance the live state with a WRONG token, then unstash.
                const int32_t wrong = (pick + 1 < int32_t(vocab())) ? pick + 1 : 0;
                q.memcpy(d_ids_ + pos, &wrong, sizeof(wrong)).wait();
                forward_step(q, d_ids_ + pos, /*T=*/1, pos).wait();
                se = q4e_unstash();
            } else if (se.empty()) {
                gate_.release();
                gate_.acquire();
                se = q4e_unstash();
            }
            if (!se.empty()) {
                res.finish_reason = "error: q4e-slot: " + se;
                break;
            }
        }
        // The token never leaves the device on the common path: D2D copy of the
        // pick into the id buffer (ordered after the sampler), then
        // the next forward is ENQUEUED without waiting so the host work below
        // (detokenize + stream callback) overlaps the GPU. The next iteration's
        // pick readback is the sync point, so a device-side failure surfaces
        // there — one token later, same exception. The host-runtime arches
        // (deepseek4/glm5next/qwen4exp) run their forward synchronously inside
        // forward_step and latch errors before it returns, unchanged.
        if (yielded) q.memcpy(d_ids_ + pos, &pick, sizeof(pick)).wait();   // gate was released: see above
        else         q.memcpy(d_ids_ + pos, d_pick_, sizeof(int32_t));
        forward_step(q, d_ids_ + pos, /*T=*/1, pos);
        pos += 1;
        // deepseek4 keeps its special tokens in the decoded text (Phase L): the
        // reasoning boundary </think> and the DSML tool-call markers are
        // special tokens, and Engine::chat parses the completion by them.
        decoded_all = tok_.decode(std::span<const int32_t>(generated),
                                  /*skip_special=*/arch_ != ModelArch::kDeepSeek4 && arch_ != ModelArch::kGlm5Next,
                                  /*keep_special=*/std::span<const int32_t>(tool_call_ids_, 2));
        const size_t safe = utf8_complete_prefix_len(decoded_all);
        if (safe > emitted && on_token) {
            if (!on_token(std::string_view(decoded_all).substr(emitted, safe - emitted))) {
                res.finish_reason = "abort"; emitted = safe; break;
            }
            emitted = safe;
        }
        // Same reason as after the prefill: never sample a failed step's stale
        // logits. The tokens produced BEFORE the failure are kept and returned.
        if (!ds4_err_.empty()) { res.finish_reason = "error: deepseek4: " + ds4_err_; break; }
        if (!glm5_err_.empty()) { res.finish_reason="error: glm5next: "+glm5_err_; break; }
        if (!q4e_err_.empty()) { res.finish_reason = "error: qwen4exp: " + q4e_err_; break; }
    }
    res.decode_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_dec0).count();
    // --- Reply snapshot (gen-cache). The next turn of this conversation sends
    // this prompt AND this reply back; snapshotting the state at the END of the
    // reply lets the trie restore [prompt ++ reply) and prefill only the
    // template tail + the new user message, instead of re-prefilling the whole
    // reply every turn. Trie-backed caches only (single-GPU prefix_cache_, the
    // fleet caches): one more LRU endpoint there is a pure win. The single-
    // endpoint caches (gemma4, qwen4exp, deepseek4 slots) would OVERWRITE the
    // prompt-boundary snapshot, which a regenerate — or a thinking model, whose
    // history drops the <think> block so the reply re-tokenizes differently —
    // still needs. Depth == pos: every generated token was forwarded (EOS is
    // never pushed), so the live state is exactly [prompt ++ reply).
    // Only when the caller (chat()) says the template can hit it: on Qwen's
    // thinking-capable templates the generation prompt carries a <think> block
    // that the history render of the same turn does not, so the next turn's
    // tokens diverge right after the assistant header and a prompt++reply
    // endpoint would only waste an LRU slot (measured 2026-09-10 on the 27B
    // split: turn-2/3 hit depth identical with and without it).
    // Kill switch: IE_NO_GEN_CACHE=1. Capacity: IE_PROMPT_CACHE_MAX_ENTRIES.
    static const bool gen_cache_off = std::getenv("IE_NO_GEN_CACHE") != nullptr;
    if (reply_cache && prompt_cache_on_ && !gen_cache_off && !generated.empty() &&
        (res.finish_reason == "stop" || res.finish_reason == "length") &&
        uint64_t(ids.size()) + generated.size() == pos) {
        std::vector<int32_t> full(ids.begin(), ids.end());
        full.insert(full.end(), generated.begin(), generated.end());
        const auto t_gc0 = std::chrono::steady_clock::now();
        std::string m;
        const char* tag = nullptr;
        if (next_)                 { m = fleet_cache_.insert(next_model_, full);            tag = "fleet-cache"; }
        else if (qwen35moe_split_) { m = fleet_cache_.insert(qwen35moe_split_model_, full); tag = "crown-split-cache"; }
        else if (gptoss_tp_)       { m = fleet_cache_.insert(gptoss_tp_model_, full);       tag = "gptoss-tp-cache"; }
        else if (q3moe_split_)     { m = fleet_cache_.insert(q3moe_split_model_, full);     tag = "q3moe-split-cache"; }
        else if (q3moe_tp_)        { m = fleet_cache_.insert(q3moe_tp_model_, full);        tag = "q3moe-tp-cache"; }
        else if (qwen35_split_)    { m = fleet_cache_.insert(qwen35_split_model_, full);    tag = "27b-split-cache"; }
        else if (!tp_ && arch_ != ModelArch::kGemma4 && arch_ != ModelArch::kQwen4Exp &&
                 arch_ != ModelArch::kDeepSeek4 && arch_ != ModelArch::kGlm5Next) {
            m = prefix_cache_.insert(q, full, kv_, dn_); tag = "prompt-cache";
        }
        if (tag) {
            const double gc_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_gc0).count();
            if (m.empty())
                std::fprintf(stderr, "[%s] reply snapshot at depth %zu (%.1f ms)\n",
                             tag, full.size(), gc_ms);
            else
                std::fprintf(stderr, "[%s] reply snapshot skipped: %s\n", tag, m.c_str());
        }
    }
    // Phase 2 (empty-response defect): a generation that decodes to empty or
    // whitespace-only text despite producing tokens is the exact signature of
    // the 2026-08-25 agent failures (825/964 tokens decoded, nothing
    // delivered). Dump the evidence unconditionally — it is rare, cheap, and
    // undiagnosable after the fact without the ids.
    if (!generated.empty() &&
        decoded_all.find_first_not_of(" \t\r\n") == std::string::npos) {
        const std::string raw =
            tok_.decode(generated, /*skip_special=*/false, std::span<const int32_t>{});
        std::fprintf(stderr,
            "[empty-gen] %zu tokens decoded to empty/whitespace text; first ids:",
            generated.size());
        for (size_t i = 0; i < generated.size() && i < 32; ++i)
            std::fprintf(stderr, " %d", generated[i]);
        std::fprintf(stderr, "\n[empty-gen] raw-with-specials (first 400): %.400s\n",
                     raw.c_str());
    }
    if (on_token && decoded_all.size() > emitted)
        on_token(std::string_view(decoded_all).substr(emitted));
    // Harmony: res.text = the FINAL-channel answer only (tokens after the LAST
    // <|message|>=200008). The analysis/thinking trace is dropped from res.text
    // (still streamed live via on_token above — clean-streaming is a v1 follow-up).
    if (arch_ == ModelArch::kGptOss && !generated.empty()) {
        // Decode WITH the Harmony markers to detect a tool call (commentary
        // channel: "…to=functions.NAME …<|message|>{json}<|call|>"). If present,
        // re-emit it as the canonical <tool_call> the OpenAI server already parses
        // into structured tool_calls; otherwise take the final-channel answer.
        const std::string raw = tok_.decode(generated, /*skip_special=*/false,
                                             std::span<const int32_t>{});
        bool is_tool = false;
        const size_t tp = raw.find("to=functions.");
        if (tp != std::string::npos) {
            const size_t ns = tp + 13;  // strlen("to=functions.")
            const size_t ne = raw.find_first_of(" \t\r\n<", ns);
            const std::string name = raw.substr(ns, ne == std::string::npos ? ne : ne - ns);
            const size_t mp = raw.find("<|message|>", tp);
            if (!name.empty() && mp != std::string::npos) {
                const size_t as = mp + 11;  // strlen("<|message|>")
                size_t ae = raw.find("<|call|>", as);
                if (ae == std::string::npos) ae = raw.size();
                std::string args = raw.substr(as, ae - as);
                const size_t l = args.find_first_not_of(" \t\r\n");
                const size_t r = args.find_last_not_of(" \t\r\n");
                args = (l == std::string::npos) ? std::string("{}") : args.substr(l, r - l + 1);
                if (args.empty()) args = "{}";
                decoded_all = "<tool_call>\n{\"name\": \"" + name +
                              "\", \"arguments\": " + args + "}\n</tool_call>";
                is_tool = true;
            }
        }
        if (!is_tool) {
            const int32_t msg_id = tok_.find_token("<|message|>");   // 200008
            size_t last_msg = generated.size();
            for (size_t k = 0; k < generated.size(); ++k)
                if (generated[k] == msg_id) last_msg = k;
            if (msg_id >= 0 && last_msg + 1 < generated.size()) {
                std::span<const int32_t> fin(generated.data() + last_msg + 1,
                                             generated.size() - last_msg - 1);
                decoded_all = tok_.decode(fin, /*skip_special=*/true, std::span<const int32_t>{});
            }
        }
    }
    res.text = std::move(decoded_all);
    res.completion_tokens = uint32_t(generated.size());
    return res;
}

// deepseek4 (docs/deepseek4/72 Phase L): the decode loop kept the special
// tokens, so `res.text` is the raw completion minus the EOS the loop never
// pushes.  Parse it by the model's own format (parse_deepseek4_completion: the
// port of the official parser) into content / reasoning_content / tool_calls.
// A completion the parser rejects — cut off by max_tokens, aborted by the
// client or a stop sequence, or malformed — keeps its text and only has the
// reasoning split off at </think>, so nothing is ever silently dropped.
static void ds4_finish_completion(const Tokenizer& tok, GenerateResult& res, bool thinking) {
    if (res.text.empty()) return;
    std::string raw = res.text;
    if (res.finish_reason == "stop") raw += std::string(tok.token_str(tok.eos_token_id()));
    DeepSeek4Completion pc = parse_deepseek4_completion(raw, thinking);
    if (pc.error.empty()) {
        res.text              = std::move(pc.content);
        res.reasoning_content = std::move(pc.reasoning_content);
        if (pc.tool_calls_json != "[]") res.tool_calls_json = std::move(pc.tool_calls_json);
        return;
    }
    if (res.finish_reason == "stop")
        std::fprintf(stderr, "[ds4-chat] completion did not parse (%s); returned as text\n",
                     pc.error.c_str());
    static const std::string kClose = "</think>";
    if (thinking) {
        const size_t p = res.text.find(kClose);
        if (p != std::string::npos) {
            res.reasoning_content = res.text.substr(0, p);
            res.text.erase(0, p + kClose.size());
        } else {   // cut off inside the reasoning: all of it is reasoning
            res.reasoning_content = std::move(res.text);
            res.text.clear();
        }
    }
    // A completion cut inside a tool-call block (max_tokens, or a client abort)
    // must not hand the half-written DSML markup to the client as prose: keep
    // the prose before the block, drop the block, and let finish_reason
    // ("length" / "abort") say the output was cut.  Seen in the Phase M soak.
    static const std::string kToolsOpen = "<\xef\xbd\x9c" "DSML" "\xef\xbd\x9c" "tool_calls";
    if (const size_t t = res.text.find(kToolsOpen); t != std::string::npos) {
        res.text.erase(t);
        while (!res.text.empty() && (res.text.back() == '\n' || res.text.back() == ' ')) res.text.pop_back();
        std::fprintf(stderr, "[ds4-chat] completion cut inside a tool call (finish %s); the partial call was dropped\n",
                     res.finish_reason.c_str());
    }
}

uint32_t Engine::prompt_cache_slots() const {
    if (mimo26_) return mimo26_prompt_cache_slots();
    if (ds41_) return ds41_prompt_cache_slots();
    return prompt_cache_on_ ? 1u : 0u;   // the other arches: the live conversation's prefix, when the cache is on
}

std::string Engine::vision_status_json() const {
    bool ready = false; std::string reason; long tokens = -1;
    switch (arch_) {
        case ModelArch::kMimo26:
            if (mimo26_) { ready = mimo26_->vis_ready; reason = ready ? "" : mimo26_->vis_error; tokens = long(mimo26_->image_tokens); }
            else reason = "model not loaded";
            break;
        case ModelArch::kDeepSeek41:
            if (ds41_) { ready = ds41_->vis_ready; reason = ready ? "" : ds41_->vis_error; tokens = long(kDs41VisMaxTok); }
            else reason = "model not loaded";
            break;
        case ModelArch::kDeepSeek4:
            ready = ds4_vis_ != nullptr;
            if (!ready) reason = "this deepseek4 load has no vision sidecar (*-Native.safetensors beside the GGUF, or $IE_DS4_VISION)";
            break;
        case ModelArch::kQwen4Exp:
            ready = true;
            break;
        default:
            reason = "this architecture has no vision input";
    }
    nlohmann::json j{{"ready", ready}, {"reason", reason}};
    if (tokens >= 0) j["image_tokens"] = tokens;
    return j.dump();
}

std::string Engine::reasoning_effort_error(std::string_view effort) const {
    if (arch_ == ModelArch::kDeepSeek41 || arch_ == ModelArch::kMimo26) return ie::reasoning_effort_error(reasoning_capabilities(arch_), effort);   // no GGUF template
    const auto* ct=gguf_.find_kv("tokenizer.chat_template");
    return ie::reasoning_effort_error(reasoning_capabilities(arch_,
        ct && ct->type==GgufValueType::kString?ct->as_string():std::string_view{}),effort);
}

GenerateResult Engine::chat(std::span<const ChatTurn> turns,
                            const SamplingParams& sp,
                            const TokenCallback& on_token,
                            bool enable_thinking,
                            std::string_view tools_json,
                            std::string_view reasoning_effort) {
    if(auto error=reasoning_effort_error(reasoning_effort);!error.empty()) {
        GenerateResult r;r.finish_reason="error: "+error;return r;
    }
    if (arch_ == ModelArch::kDeepSeek41) return ds41_chat(turns, sp, on_token, enable_thinking, tools_json, reasoning_effort);
    if (arch_ == ModelArch::kMimo26) return mimo26_chat(turns, sp, on_token, enable_thinking, tools_json, reasoning_effort);
    // Vision guard: only the qwen4exp path (ChatML branch below) consumes
    // turn images; every other arch must refuse rather than silently drop.
    q4e_pending_imgs_.clear();
    ds4_pending_imgs_.clear();
    bool has_images = false;
    for (const auto& t : turns) has_images |= !t.images.empty();
    const bool ds4_vision = arch_ == ModelArch::kDeepSeek4 && ds4_vis_ != nullptr;
    if (has_images && arch_ != ModelArch::kQwen4Exp && !ds4_vision) {
        GenerateResult res;
        res.finish_reason = arch_ == ModelArch::kDeepSeek4
            ? "error: this deepseek4 load has no vision sidecar (*-Native.safetensors beside the GGUF, or $IE_DS4_VISION)"
            : "error: this model has no vision support";
        return res;
    }
    // Per-request image staging lives in engine members that are written
    // BEFORE the generation gate; with --parallel > 1 two requests could
    // clobber each other's images. Refuse plainly instead of racing.
    if (has_images && opts_.parallel > 1) {
        GenerateResult res;
        res.finish_reason = "error: image inputs need --parallel 1 "
                            "(per-request image staging is engine-global, not slot-safe)";
        return res;
    }
    if (arch_ == ModelArch::kGlm5Next) {
        std::string error;
        const std::string prompt=build_glm5_prompt(turns,enable_thinking,tools_json,error,reasoning_effort);
        if(!error.empty()) { GenerateResult r;r.finish_reason="error: "+error;return r; }
        auto r=generate(prompt,sp,on_token);
        auto parsed=parse_glm5_completion(r.text,enable_thinking,tools_json);
        r.text=std::move(parsed.content);r.reasoning_content=std::move(parsed.reasoning);
        r.tool_calls_json=std::move(parsed.tool_calls);
        if(!r.tool_calls_json.empty() && r.finish_reason=="stop")r.finish_reason="tool_calls";
        return r;
    }
    // Wave-1 template-family dispatch. dcfg_.template_family is read from the
    // GGUF chat_template (model_config.cpp). For every CURRENT model this is
    // byte-identical to the old `arch_==kLlama3` branch: today no current GGUF
    // resolves to kMistral/kDeepSeek, llama-3 GGUFs resolve to kLlama3 (or kAuto,
    // which falls back to arch_==kLlama3), and crown/qwen3/qwen3moe are
    // kAuto+non-llama → ChatML. Only NEW model families (Mistral [INST],
    // DeepSeek-R1-Distill sentinels) take the new branches.
    if (arch_ == ModelArch::kDeepSeek4) {
        // DeepSeek-V4-Flash. This is NOT the R1-Distill path: the model ships no
        // jinja template and build_deepseek4_prompt() is the port of the official
        // encoding_dsv4.py (thinking convention, reasoning-effort preamble, DSML
        // tools block, <tool_result> merging) — see tokenizer.cpp for the exact
        // ported/deferred split. Checked BEFORE the template-family dispatch so a
        // GGUF carrying the Unsloth substitute template can never route here.
        for (const auto& t : turns) {
            if (t.role != "system" && t.role != "user" &&
                t.role != "assistant" && t.role != "tool") {
                GenerateResult res;
                res.finish_reason = "error: role '" + t.role +
                                    "' is not supported on the deepseek4 path";
                return res;
            }
        }
        // Vision-Exp: decode each image now (official resize/pad; its geometry
        // sets the block size) and prepend one placeholder token per image to
        // the turn text; generate() expands the placeholders into blocks.
        std::span<const ChatTurn> ds4_turns = turns;
        std::vector<ChatTurn> ds4_vis_turns;
        if (has_images) {
            ds4_vis_turns.assign(turns.begin(), turns.end());
            const std::string ph(tok_.token_str(kDs4ImagePlaceholderId));
            for (auto& t : ds4_vis_turns) {
                for (const auto& bytes : t.images) {
                    Ds4PendingImage im;
                    if (auto e = ds4_load_image_mem(bytes.data(), bytes.size(), im.px, im.H, im.W, im.geom);
                        !e.empty()) {
                        GenerateResult res;
                        res.finish_reason = "error: " + e;
                        return res;
                    }
                    ds4_pending_imgs_.push_back(std::move(im));
                }
                // The parser marks where each image sat among the text parts;
                // honour that (the reference does). Markers missing or
                // mismatched (a caller that built ChatTurn by hand): prepend.
                size_t markers = 0;
                for (size_t p = t.content.find(kChatImageMarker); p != std::string::npos;
                     p = t.content.find(kChatImageMarker, p + kChatImageMarker.size())) ++markers;
                if (markers == t.images.size()) {
                    for (size_t p = t.content.find(kChatImageMarker); p != std::string::npos;
                         p = t.content.find(kChatImageMarker, p + ph.size()))
                        t.content.replace(p, kChatImageMarker.size(), ph);
                } else {
                    for (size_t p = t.content.find(kChatImageMarker); p != std::string::npos;
                         p = t.content.find(kChatImageMarker)) t.content.erase(p, kChatImageMarker.size());
                    std::string runs;
                    for (size_t i = 0; i < t.images.size(); ++i) runs += ph;
                    t.content = runs + t.content;
                }
            }
            ds4_turns = ds4_vis_turns;
        }
        DeepSeek4ChatOptions opt;
        opt.thinking   = enable_thinking;
        opt.tools_json = tools_json;
        if(!reasoning_effort.empty())opt.reasoning_effort=reasoning_effort;
        //
        // Stable conversation boundary = the same render WITHOUT the trailing
        // generation prompt. That is the prefix the NEXT turn shares, so the
        // snapshot is taken there and the few-token gen-prompt suffix is
        // re-prefilled fresh each turn. Without this the cache would only ever
        // reuse an EXACT full repeat — which no chat client ever sends. Mirrors
        // the chatml / harmony / gemma paths.
        uint32_t cache_prefix_len = 0;
        if (prompt_cache_on_ && !has_images) {   // vision requests never touch the cache
            const std::string stable =
                build_deepseek4_prompt(ds4_turns, /*add_generation_prompt=*/false, opt);
            cache_prefix_len = uint32_t(tok_.encode(stable, /*allow_special=*/true).size());
        }
        GenerateResult res =
            generate(build_deepseek4_prompt(ds4_turns, /*add_generation_prompt=*/true, opt),
                     sp, on_token, cache_prefix_len);
        ds4_finish_completion(tok_, res, enable_thinking);
        return res;
    }
    using TF = DenseConfig::TemplateFamily;
    const TF tf = is_dense_arch(arch_) ? dcfg_.template_family : TF::kAuto;
    if (tf == TF::kMistral) {
        // Mistral [INST] (+ [SYSTEM_PROMPT] for Devstral/v3+). Tools are not
        // rendered into the Mistral template (v1). BOS (<s>) added by encode().
        if (!tools_json.empty()) {
            GenerateResult res;
            res.finish_reason = "error: tool calling not supported on the Mistral path (v1)";
            return res;
        }
        return generate(build_mistral_prompt(turns, /*add_generation_prompt=*/true,
                                             dcfg_.mistral_sysprompt != 0),
                        sp, on_token);
    }
    if (tf == TF::kDeepSeek) {
        // DeepSeek-R1-Distill sentinel template (reasoning; emits <think>). Tools
        // are not rendered (v1). BOS (<｜begin▁of▁sentence｜>) added by encode().
        if (!tools_json.empty()) {
            GenerateResult res;
            res.finish_reason = "error: tool calling not supported on the DeepSeek path (v1)";
            return res;
        }
        return generate(build_deepseek_prompt(turns, /*add_generation_prompt=*/true,
                                              enable_thinking),
                        sp, on_token);
    }
    if (tf == TF::kGranite) {
        // Granite-3.x <|start_of_role|> turns. Tools not rendered (v1).
        if (!tools_json.empty()) {
            GenerateResult res;
            res.finish_reason = "error: tool calling not supported on the Granite path (v1)";
            return res;
        }
        return generate(build_granite_prompt(turns, /*add_generation_prompt=*/true),
                        sp, on_token);
    }
    if (arch_ == ModelArch::kGemma4) {
        // Gemma 4 <|turn>{role}\n…<turn|> turns. Tools not rendered (v1). BOS by encode().
        if (!tools_json.empty()) {
            GenerateResult res;
            res.finish_reason = "error: tool calling not supported on the Gemma path (v1)";
            return res;
        }
        const std::string full = build_gemma_prompt(turns, /*add_generation_prompt=*/true);
        // Stable conversation boundary = the same render WITHOUT the trailing
        // generation prompt. That's the prefix the NEXT turn shares, so the VRAM-
        // guarded snapshot is taken there (the gen-prompt suffix is a few tokens,
        // re-prefilled fresh each turn). Mirrors the chatml/harmony paths.
        uint32_t cache_prefix_len = 0;
        if (prompt_cache_on_) {
            const std::string stable = build_gemma_prompt(turns, /*add_generation_prompt=*/false);
            cache_prefix_len = uint32_t(tok_.encode(stable, /*allow_special=*/true).size());
        }
        return generate(full, sp, on_token, cache_prefix_len);
    }
    if (arch_ == ModelArch::kGptOss) {
        // gpt-oss OpenAI Harmony. Tools render as the developer-message
        // `namespace functions { … }` block; the model's commentary tool call is
        // translated back to a canonical <tool_call> in generate(). No BOS (encode
        // prepends none for this GGUF). enable_thinking maps to the reasoning effort.
        const std::string_view effort = !reasoning_effort.empty()?reasoning_effort:
            enable_thinking ? "high" : "low";
        const std::string full = build_harmony_prompt(turns, /*add_generation_prompt=*/true,
                                                      effort, tools_json);
        // Stable boundary (no trailing assistant/analysis header) so the prompt cache
        // snapshots the shared prefix the next turn reuses — mirrors the chatml path.
        uint32_t cache_prefix_len = 0;
        if (prompt_cache_on_) {
            const std::string stable = build_harmony_prompt(turns, /*add_generation_prompt=*/false,
                                                            effort, tools_json);
            cache_prefix_len = uint32_t(tok_.encode(stable, /*allow_special=*/true).size());
        }
        return generate(full, sp, on_token, cache_prefix_len);
    }
    const bool want_llama3 =
        (tf == TF::kLlama3) || (tf == TF::kAuto && arch_ == ModelArch::kLlama3);
    if (want_llama3) {
        // v1: tools are not rendered into the llama template. enable_thinking is
        // ignored (no <think> convention). BOS is prepended by encode().
        if (!tools_json.empty()) {
            GenerateResult res;
            res.finish_reason = "error: tool calling not supported on the llama path (v1)";
            return res;
        }
        return generate(build_llama3_prompt(turns, /*add_generation_prompt=*/true), sp, on_token);
    }
    // Inject the <think> block ONLY for reasoning models whose CHAT TEMPLATE uses
    // it. Qwen3-Coder HAS <think> tokens in vocab (151667/151668) but is a
    // non-reasoning Instruct model — its template never emits <think>, and
    // injecting the empty-think convention makes it free-continue ("Human:"). Gate
    // on the template string, not token presence.
    const auto* ct = gguf_.find_kv("tokenizer.chat_template");
    const bool model_has_think = ct && ct->type == GgufValueType::kString &&
                                 ct->as_string().find("<think>") != std::string_view::npos;
    // Qwen3.8-style template surface (gate on the template string, same philosophy
    // as model_has_think — Qwen3.6-and-earlier templates have neither feature):
    //  1. reasoning_effort: default xhigh injects a sentence at the top of the
    //     system block when thinking is on; medium injects nothing; the template
    //     aliases "high"→"xhigh". Override: IE_REASONING_EFFORT=xhigh|medium|low.
    //  2. developer role: folded into system (3.6 templates raise on it).
    std::string_view effort_preamble;
    if (ct && ct->type == GgufValueType::kString && enable_thinking &&
        ct->as_string().find("reasoning_effort") != std::string_view::npos) {
        const std::string eff = [&]{
            const char* e = std::getenv("IE_REASONING_EFFORT");
            std::string v = !reasoning_effort.empty()?std::string(reasoning_effort):(e ? e : "xhigh");
            return v == "high" ? std::string("xhigh") : v;
        }();
        if (eff == "xhigh")
            effort_preamble = "Reasoning effort is set to xhigh. Please think carefully "
                              "through the task, validate key assumptions, consider plausible "
                              "alternatives, and prioritize correctness, consistency, and "
                              "clarity in the final answer.";
        else if (eff == "low")
            effort_preamble = "Reasoning effort is set to low. Keep your thinking brief and "
                              "focused, moving directly to the conclusion without unnecessary "
                              "elaboration.";
        // medium (or unknown): no injection, matching the vendor template.
    }
    std::span<const ChatTurn> render_turns = turns;
    // Vision (qwen4exp, docs/qwen4/16_vision_port.md §4): decode each image NOW
    // (its merged grid sets the pad-run length), stash pixels for generate()'s
    // encode, and rewrite the turn's content with the sentinel run the
    // tokenizer maps to ids 248053 / 248056×N / 248054.
    std::vector<ChatTurn> vis_turns;
    if (has_images) {
        vis_turns.assign(turns.begin(), turns.end());
        for (auto& t : vis_turns) {
            std::string runs;
            for (const auto& bytes : t.images) {
                Q4ePendingImage im;
                if (auto e = qwen4_load_image_mem(bytes.data(), bytes.size(),
                                                  im.px, im.H, im.W); !e.empty()) {
                    GenerateResult res;
                    res.finish_reason = "error: " + e;
                    return res;
                }
                const uint32_t nm = (im.H / 32) * (im.W / 32);
                runs += "<|vision_start|>";
                for (uint32_t i = 0; i < nm; ++i) runs += "<|image_pad|>";
                runs += "<|vision_end|>";
                q4e_pending_imgs_.push_back(std::move(im));
            }
            for (size_t p = t.content.find(kChatImageMarker); p != std::string::npos;
                 p = t.content.find(kChatImageMarker)) t.content.erase(p, kChatImageMarker.size());
            if (!runs.empty()) t.content = runs + t.content;
        }
        render_turns = vis_turns;
    }
    std::vector<ChatTurn> folded;   // only materialized if a developer turn exists
    if (ct && ct->type == GgufValueType::kString &&
        ct->as_string().find("developer") != std::string_view::npos) {
        for (size_t ti = 0; ti < render_turns.size(); ++ti) {
            if (render_turns[ti].role == "developer") {
                folded.assign(render_turns.begin(), render_turns.end());
                for (auto& t : folded) if (t.role == "developer") t.role = "system";
                render_turns = folded;
                break;
            }
        }
    }
    const std::string full = build_chatml_prompt(render_turns, /*add_generation_prompt=*/true,
                                                 enable_thinking, tools_json, model_has_think,
                                                 effort_preamble);
    // Stable conversation boundary = the same render WITHOUT the generation prompt /
    // think suffix. That's the prefix the NEXT turn shares, so the prompt cache snapshots
    // there (the gen-prompt/think suffix is re-prefilled fresh each turn — a few tokens).
    uint32_t cache_prefix_len = 0;
    if (prompt_cache_on_) {
        const std::string stable = build_chatml_prompt(render_turns, /*add_generation_prompt=*/false,
                                                       enable_thinking, tools_json, model_has_think,
                                                       effort_preamble);
        cache_prefix_len = uint32_t(tok_.encode(stable, /*allow_special=*/true).size());
    }
    // Reply snapshots can only hit when the history render of an assistant turn
    // (<|im_start|>assistant\n + content) equals the generation prompt + reply:
    // true for plain ChatML (non-thinking models such as Qwen3-Coder), false for
    // thinking-capable templates, whose generation prompt adds a <think> block
    // (empty when thinking is off) that history never carries.
    return generate(full, sp, on_token, cache_prefix_len, /*reply_cache=*/!model_has_think);
}

}  // namespace ie
