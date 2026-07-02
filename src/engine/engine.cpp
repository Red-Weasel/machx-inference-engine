// src/engine/engine.cpp — product-facing ie::Engine: load once, streaming
// generate/chat.  Load sequence mirrors tools/ie_bench.cpp; decode loop
// mirrors tools/forward_test.cpp.  Prefill is chunked at opts.prefill_chunk
// (hard cap 256 per docs/known_bugs.md — never submit larger T).
#include "ie/engine.hpp"
#include "ie/memory_plan.hpp"
#include "ie/ops.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <vector>

namespace ie {

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
    auto e = std::unique_ptr<Engine>(new Engine());
    EngineOptions opts = opts_in;   // mutable working copy (the planner resolves n_gpus)
    e->opts_ = opts;
    if (auto m = e->gguf_.open(gguf_path); !m.empty())   { err = "gguf: " + m;  return nullptr; }
    if (auto m = e->alloc_.init(); !m.empty())           { err = "alloc: " + m; return nullptr; }

    // Arch dispatch: qwen35moe → crown QwenModel path (behavior unchanged),
    // qwen3 → DenseModel (every layer full-attention; no DeltaNetState).
    e->arch_ = detect_arch(e->gguf_);
    if (e->arch_ == ModelArch::kUnknown) {
        const auto* a = e->gguf_.find_kv("general.architecture");
        err = "unsupported architecture: " +
              (a ? std::string(a->as_string()) : std::string("<missing>"));
        return nullptr;
    }

    // VRAM-aware placement: when n_gpus==0 (auto, the CLI default) sense per-card
    // VRAM and pick single vs multi-GPU; an explicit --gpus N is respected. Only
    // arches with a working >1-GPU path may be split (gemma4/qwen3moe/crown are
    // single-GPU today). Resolves opts.n_gpus to a concrete >=1 for the branches.
    {
        const bool mgpu = is_dense_arch(e->arch_) ||
                          e->arch_ == ModelArch::kQwen35Dense ||
                          e->arch_ == ModelArch::kQwen3Next ||
                          e->arch_ == ModelArch::kGptOss ||
                          e->arch_ == ModelArch::kQwen35Moe ||
                          e->arch_ == ModelArch::kQwen3Moe;
        PlacementPlan plan = plan_placement(e->gguf_, e->arch_, opts.n_gpus,
                                            opts.max_ctx, opts.int8_kv, mgpu,
                                            e->alloc_.device());
        std::fprintf(stderr, "[mem-plan] %s\n", plan.note.c_str());
        opts.n_gpus     = plan.n_gpus;
        e->opts_.n_gpus = plan.n_gpus;
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
            if (auto m = e->fleet_.init(opts.n_gpus); !m.empty()) { err = "fleet: " + m; return nullptr; }
            if (e->fleet_.size() < 2) { err = "tensor-parallel needs >=2 GPUs (have " +
                                              std::to_string(e->fleet_.size()) + ")"; return nullptr; }
            if (auto m = e->qwen35_tp_model_.load(e->fleet_, e->gguf_, qcfg, opts.max_ctx); !m.empty())
                { err = "tp model: " + m; return nullptr; }
            e->qwen35_tp_ = true;
            e->tp_ids_host_.resize(std::max<uint32_t>(opts.prefill_chunk, 1u));
            e->tp_logits_host_.resize(qcfg.dense.vocab);
            vocab = qcfg.dense.vocab;
        } else if (use_split) {
            const uint32_t ndev = opts.n_gpus > 1 ? opts.n_gpus : 1;
            if (auto m = e->fleet_.init(ndev); !m.empty()) { err = "fleet: " + m; return nullptr; }
            if (ndev > 1 && e->fleet_.size() < 2) { err = "layer-split needs >=2 GPUs (have " +
                                              std::to_string(e->fleet_.size()) + ")"; return nullptr; }
            const uint32_t n_tx = qcfg.n_transformer_layers();
            LayerPlan plan = LayerPlan::contiguous(n_tx, ndev);
            if (auto m = e->qwen35_split_model_.load(e->fleet_, plan, e->gguf_, qcfg, opts.max_ctx); !m.empty())
                { err = "model: " + m; return nullptr; }
            e->qwen35_split_ = true;
            e->tp_ids_host_.resize(std::max<uint32_t>(opts.prefill_chunk, 1u));
            e->tp_logits_host_.resize(qcfg.dense.vocab);
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
            if (auto m = e->q3moe_tp_model_.load(e->fleet_, e->gguf_, q3mcfg, opts.max_ctx); !m.empty())
                { err = "q3moe-tp model: " + m; return nullptr; }
            e->q3moe_tp_ = true;
            e->tp_ids_host_.resize(std::max<uint32_t>(opts.prefill_chunk, 1u));
            e->tp_logits_host_.resize(q3mcfg.dense.vocab);
        } else if (opts.n_gpus > 1) {
            // Multi-GPU LAYER-SPLIT (kQwen3Moe): per-card weights + full-attn KV,
            // host-logits bounce like the crown split. Splits the 130k KV across
            // cards. The single-GPU q3moe_ is NOT loaded on this path.
            if (auto m = e->fleet_.init(opts.n_gpus); !m.empty()) { err = "fleet: " + m; return nullptr; }
            if (e->fleet_.size() < 2) { err = "layer-split needs >=2 GPUs (have " +
                                              std::to_string(e->fleet_.size()) + ")"; return nullptr; }
            LayerPlan plan = LayerPlan::contiguous(q3mcfg.dense.n_layers, opts.n_gpus);
            if (auto m = e->q3moe_split_model_.load(e->fleet_, plan, e->gguf_, q3mcfg, opts.max_ctx); !m.empty())
                { err = "model: " + m; return nullptr; }
            e->q3moe_split_ = true;
            e->tp_ids_host_.resize(std::max<uint32_t>(opts.prefill_chunk, 1u));
            e->tp_logits_host_.resize(q3mcfg.dense.vocab);
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
        if (auto m = e->fleet_.init(n_gpus); !m.empty()) { err = "fleet: " + m; return nullptr; }
        const uint32_t n_layers = ncfg.hybrid.n_transformer_layers();
        LayerPlan plan = LayerPlan::contiguous(n_layers, n_gpus);
        if (auto m = e->next_model_.load(e->fleet_, plan, e->gguf_, ncfg, opts.max_ctx); !m.empty())
            { err = "qwen3next model: " + m; return nullptr; }
        e->next_ = true;
        e->tp_ids_host_.resize(std::max<uint32_t>(opts.prefill_chunk, 1u));
        e->tp_logits_host_.resize(ncfg.hybrid.dense.vocab);
        // Fleet prompt/KV cache (default-ON; opt out IE_NO_PROMPT_CACHE → byte-identical
        // OFF path). Reuses prompt_cache_on_; generate() next_ branch restores per-card state.
        if ((opts.prompt_cache || std::getenv("IE_PROMPT_CACHE")) && !std::getenv("IE_NO_PROMPT_CACHE")) {
            if (auto m = e->fleet_cache_.init(e->next_model_, FleetPrefixCacheConfig{}); !m.empty()) {
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
        if (auto m = e->fleet_.init(opts.n_gpus); !m.empty()) { err = "fleet: " + m; return nullptr; }
        if (e->fleet_.size() < 2) { err = "layer-split needs >=2 GPUs (have " +
                                          std::to_string(e->fleet_.size()) + ")"; return nullptr; }
        LayerPlan plan = LayerPlan::contiguous(cfg.n_layers, opts.n_gpus);
        if (auto m = e->qwen35moe_split_model_.load(e->fleet_, plan, e->gguf_, cfg, opts.max_ctx); !m.empty())
            { err = "model: " + m; return nullptr; }
        e->qwen35moe_split_ = true;
        e->tp_ids_host_.resize(std::max<uint32_t>(opts.prefill_chunk, 1u));
        e->tp_logits_host_.resize(e->qwen35moe_split_model_.config().vocab);
        vocab = e->qwen35moe_split_model_.config().vocab;
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
            if (auto m = e->prefix_cache_.init(e->alloc_, cfg, PrefixCacheConfig{}); !m.empty()) {
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

Engine::~Engine() {
    if (!alloc_.ready()) return;
    auto& q = alloc_.queue();
    if (d_logits_) sycl::free(d_logits_, q);
    if (d_ids_)    sycl::free(d_ids_, q);
    if (d_pick_)   sycl::free(d_pick_, q);
    if (d_recent_) sycl::free(d_recent_, q);
}

GenerateResult Engine::generate(const std::string& prompt,
                                const SamplingParams& sp,
                                const TokenCallback& on_token,
                                uint32_t cache_prefix_len) {
    GenerateResult res;
    auto& q = alloc_.queue();
    auto ids = tok_.encode(prompt, /*allow_special=*/true);
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

    // --- MTP self-speculative GREEDY decode (single-GPU kQwen35Dense, --spec) ---
    // Only when spec is loaded AND sampling is greedy (temperature == 0): the
    // lossless guarantee holds for argmax decode only. Any temperature > 0 falls
    // through to the normal sampled decode below (byte-identical to non-spec).
    if (spec_ && qwen35_.mtp_loaded() && arch_ == ModelArch::kQwen35Dense &&
        !qwen35_split_ && !qwen35_tp_ && sp.temperature == 0.0f) {
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
        sp.temperature == 0.0f) {
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
        // 80B fleet path: restore the longest cached prefix into EVERY card's live
        // kv_[dev]/dn_[dev]. On a hit (M>0) the forward starts at pos=M≠0 →
        // reset_kv=false → it uses the restored state. On a miss (M==0) we touch
        // nothing; pos=0 → the forward self-resets every card. Restore is all-or-
        // nothing across cards: a stale (unrestored) card would silently corrupt a
        // hit (no reset happens), so any per-card failure/missing-snapshot falls
        // back to restored=0 (clean full prefill — pos=0 resets all cards).
        auto hit = fleet_cache_.find_longest_match(ids);
        const uint32_t cap = uint32_t(ids.size()) - 1;          // prefill >=1 token
        const uint32_t M   = std::min<uint32_t>(hit.match_len, cap);
        if (M > 0 && hit.kv && hit.dn) {
            DeviceFleet* fleet = next_model_.fleet();
            bool ok = true;
            std::string emsg;
            for (uint32_t dev = 0; dev < next_model_.n_devices() && ok; ++dev) {
                sycl::queue& dq = fleet->dev(dev).queue();
                if (next_model_.dev_has_kv(dev)) {
                    if (!(*hit.kv)[dev]) { ok = false; emsg = "missing kv snapshot"; break; }
                    if (auto e1 = next_model_.kv_cache(dev).copy_prefix_from(dq, *(*hit.kv)[dev], M);
                        !e1.empty()) { ok = false; emsg = e1; break; }
                }
                if (next_model_.dev_has_dn(dev)) {
                    if (!(*hit.dn)[dev]) { ok = false; emsg = "missing dn snapshot"; break; }
                    if (auto e2 = next_model_.dn_state(dev).copy_from(dq, *(*hit.dn)[dev]);
                        !e2.empty()) { ok = false; emsg = e2; break; }
                }
            }
            if (ok) {
                restored = M;                                   // suffix prefill starts here
            } else {
                // Partial restore → restored stays 0; the pos==0 forward resets every
                // card, wiping the half-restored state before the clean full prefill.
                std::fprintf(stderr, "[fleet-cache] restore failed (%s); full prefill\n", emsg.c_str());
            }
        }
    } else if (prompt_cache_on_ && !tp_ && !next_) {
        auto hit = prefix_cache_.find_longest_match(ids);
        const uint32_t cap = uint32_t(ids.size()) - 1;          // prefill >=1 token
        const uint32_t M   = std::min<uint32_t>(hit.match_len, cap);
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
    res.cached_tokens = restored;
    // Snapshot boundary: the STABLE conversation depth (cache_prefix_len, no gen-prompt/
    // think suffix) so the next turn restores it; 0 → full depth (exact-repeat reuse).
    // The DeltaNet state is position-dependent, so we must snapshot when kv_/dn_ hold
    // EXACTLY the boundary state — hence split the prefill there.
    const uint32_t snap_at = (prompt_cache_on_ && cache_prefix_len > 0)
        ? std::min<uint32_t>(cache_prefix_len, uint32_t(ids.size()))
        : uint32_t(ids.size());
    uint32_t pos = restored;            // cache restore (if any) already loaded [0..restored)
    const auto t_pf0 = std::chrono::steady_clock::now();
    auto prefill_to = [&](uint32_t end) {
        while (pos < end) {
            const uint32_t n = std::min<uint32_t>(pf_chunk, end - pos);
            forward_step(q, d_ids_ + pos, n, pos).wait();
            pos += n;
        }
    };
    prefill_to(snap_at);
    // Snapshot the boundary state (kv_/dn_ now hold [0..snap_at)) — only if we actually
    // prefilled new tokens up to it (snap_at > restored).
    if (prompt_cache_on_ && !tp_ && !next_ && snap_at > restored) {
        const std::vector<int32_t> pref(ids.begin(), ids.begin() + snap_at);
        if (auto m = prefix_cache_.insert(q, pref, kv_, dn_); !m.empty())
            std::fprintf(stderr, "[prompt-cache] insert: %s\n", m.c_str());
    }
    // 80B fleet snapshot: kv_[dev]/dn_[dev] now hold [0..snap_at) on every card.
    if (prompt_cache_on_ && next_ && snap_at > restored) {
        const std::vector<int32_t> pref(ids.begin(), ids.begin() + snap_at);
        if (auto m = fleet_cache_.insert(next_model_, pref); !m.empty())
            std::fprintf(stderr, "[fleet-cache] insert: %s\n", m.c_str());
    }
    prefill_to(uint32_t(ids.size()));   // volatile suffix (gen-prompt/think), if any
    res.prefill_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_pf0).count();

    uint64_t rng = sp.seed ? sp.seed
        : uint64_t(std::chrono::steady_clock::now().time_since_epoch().count());
    std::vector<int32_t> generated;
    std::string decoded_all;
    size_t emitted = 0;
    res.finish_reason = "length";
    const uint32_t max_new = std::min<uint32_t>(sp.max_tokens, budget - uint32_t(ids.size()));
    const auto t_dec0 = std::chrono::steady_clock::now();
    for (uint32_t step = 0; step < max_new; ++step) {
        if (sp.repeat_penalty != 1.0f && !generated.empty()) {
            // d_recent_ is sized kMaxRepeatWindow; clamp nw to match.
            constexpr uint32_t kMaxRepeatWindow = 512;
            const uint32_t nw = std::min({sp.repeat_window,
                                          uint32_t(generated.size()),
                                          kMaxRepeatWindow});
            q.memcpy(d_recent_, generated.data() + generated.size() - nw, nw * sizeof(int32_t)).wait();
            repetition_penalty(q, d_logits_, vocab(), d_recent_, nw, sp.repeat_penalty);
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
        sample_softmax_topk_topp(q, d_logits_, d_pick_, vocab(),
                                 sp.temperature, sp.top_k, sp.top_p, sp.min_p,
                                 rng + step).wait();
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
        decoded_all = tok_.decode(std::span<const int32_t>(generated), /*skip_special=*/true,
                                  /*keep_special=*/std::span<const int32_t>(tool_call_ids_, 2));
        const size_t safe = utf8_complete_prefix_len(decoded_all);
        if (safe > emitted && on_token) {
            if (!on_token(std::string_view(decoded_all).substr(emitted, safe - emitted))) {
                res.finish_reason = "abort"; emitted = safe; break;
            }
            emitted = safe;
        }
        q.memcpy(d_ids_ + pos, &pick, sizeof(pick)).wait();
        forward_step(q, d_ids_ + pos, /*T=*/1, pos).wait();
        pos += 1;
    }
    res.decode_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t_dec0).count();
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

GenerateResult Engine::chat(std::span<const ChatTurn> turns,
                            const SamplingParams& sp,
                            const TokenCallback& on_token,
                            bool enable_thinking,
                            std::string_view tools_json) {
    // Wave-1 template-family dispatch. dcfg_.template_family is read from the
    // GGUF chat_template (model_config.cpp). For every CURRENT model this is
    // byte-identical to the old `arch_==kLlama3` branch: today no current GGUF
    // resolves to kMistral/kDeepSeek, llama-3 GGUFs resolve to kLlama3 (or kAuto,
    // which falls back to arch_==kLlama3), and crown/qwen3/qwen3moe are
    // kAuto+non-llama → ChatML. Only NEW model families (Mistral [INST],
    // DeepSeek-R1-Distill sentinels) take the new branches.
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
        return generate(build_gemma_prompt(turns, /*add_generation_prompt=*/true), sp, on_token);
    }
    if (arch_ == ModelArch::kGptOss) {
        // gpt-oss OpenAI Harmony. Tools render as the developer-message
        // `namespace functions { … }` block; the model's commentary tool call is
        // translated back to a canonical <tool_call> in generate(). No BOS (encode
        // prepends none for this GGUF). enable_thinking maps to the reasoning effort.
        const std::string_view effort = enable_thinking ? "high" : "low";
        return generate(build_harmony_prompt(turns, /*add_generation_prompt=*/true,
                                             effort, tools_json),
                        sp, on_token);
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
    const std::string full = build_chatml_prompt(turns, /*add_generation_prompt=*/true,
                                                 enable_thinking, tools_json, model_has_think);
    // Stable conversation boundary = the same render WITHOUT the generation prompt /
    // think suffix. That's the prefix the NEXT turn shares, so the prompt cache snapshots
    // there (the gen-prompt/think suffix is re-prefilled fresh each turn — a few tokens).
    uint32_t cache_prefix_len = 0;
    if (prompt_cache_on_) {
        const std::string stable = build_chatml_prompt(turns, /*add_generation_prompt=*/false,
                                                       enable_thinking, tools_json, model_has_think);
        cache_prefix_len = uint32_t(tok_.encode(stable, /*allow_special=*/true).size());
    }
    return generate(full, sp, on_token, cache_prefix_len);
}

}  // namespace ie
