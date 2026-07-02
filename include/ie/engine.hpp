// include/ie/engine.hpp — product-facing API: load once, generate with a
// streaming callback.  v1: one implicit session; server treats requests as
// stateless (reset + full-conversation prefill per request).
#pragma once
#include "ie/allocator.hpp"
#include "ie/deltanet_state.hpp"
#include "ie/dense_transformer.hpp"
#include "ie/dense_tp.hpp"
#include "ie/qwen35_dense.hpp"
#include "ie/qwen35_split.hpp"
#include "ie/qwen35moe_split.hpp"
#include "ie/qwen35_tp.hpp"
#include "ie/qwen3moe.hpp"
#include "ie/qwen3moe_split.hpp"
#include "ie/qwen3moe_tp.hpp"
#include "ie/qwen3next.hpp"
#include "ie/gptoss.hpp"
#include "ie/gptoss_tp.hpp"
#include "ie/gemma4.hpp"
#include "ie/gguf.hpp"
#include "ie/kv_cache.hpp"
#include "ie/prefix_cache.hpp"
#include "ie/fleet_prefix_cache.hpp"
#include "ie/model_config.hpp"
#include "ie/qwen36.hpp"
#include "ie/tokenizer.hpp"
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace ie {

// SamplingParams::max_tokens value meaning "no explicit cap": generate until
// EOS or the context budget. Safe because Engine::generate clamps with
// min(max_tokens, ctx_budget - prompt_tokens) after rejecting prompts that
// already exceed the budget.
inline constexpr uint32_t kMaxTokensUnlimited = UINT32_MAX;

struct SamplingParams {
    float    temperature    = 0.7f;
    uint32_t top_k          = 40;
    float    top_p          = 0.95f;
    float    min_p          = 0.0f;
    float    repeat_penalty = 1.0f;   // 1.0 = off
    uint32_t repeat_window  = 64;     // clamped to 512 internally
    uint64_t seed           = 0;      // 0 = derive from steady_clock
    uint32_t max_tokens     = 512;
    bool     ignore_eos     = false;  // benchmarking: keep decoding exactly
                                      // max_tokens, never stop on eos/stop ids
                                      // (fixed-budget tg, == llama --ignore-eos).
};

struct EngineOptions {
    uint32_t max_ctx       = 8192;
    bool     int8_kv       = false;
    uint32_t prefill_chunk = 256;    // hard cap per docs/known_bugs.md
    uint32_t n_gpus        = 1;      // >1 → tensor-parallel split (dense archs only)
    bool     prompt_cache  = true;   // reuse cached prefix KV+DN across turns (crown
                                     // single-GPU + Qwen3-Next-80B fleet). Default-ON;
                                     // opt out via env IE_NO_PROMPT_CACHE=1 (or =false).
    bool     spec          = false;  // MTP self-speculative decode (Qwen3.6-27B
                                     // kQwen35Dense, single-GPU, GREEDY only). Loads
                                     // the native MTP/NextN head (~0.6 GB) and runs
                                     // draft+verify; lossless vs plain greedy. CLI: --spec.
    uint32_t spec_k        = 2;      // draft length. K=2 is the MEASURED optimum for the
                                     // 27B (verify cost grows ~linearly with K but accept
                                     // saturates): clean-box K=2 1.12× / K=3 1.06× / K=4
                                     // 0.98× (a LOSS). Lossless at all K. Override --spec-k.
    std::string spec_head;           // gemma4 MTP draft head GGUF (separate file,
                                     // mtp-gemma-4-*-Q8_0.gguf). Empty → convention:
                                     // first mtp-*.gguf next to the target. CLI: --spec-head.
};

struct GenerateResult {
    std::string text;
    uint32_t    prompt_tokens     = 0;
    uint32_t    cached_tokens     = 0; // prompt tokens served from the prefix cache
    uint32_t    completion_tokens = 0;
    std::string finish_reason;        // "stop" | "length" | "abort"
    double      prefill_ms        = 0; // wall time of the prefill phase
    double      decode_ms         = 0; // wall time of the decode loop
};

// on_token receives each newly-decoded UTF-8-complete text fragment.
// Return false to abort generation (finish_reason="abort").
using TokenCallback = std::function<bool(std::string_view)>;

class Engine {
public:
    static std::unique_ptr<Engine> load(const std::string& gguf_path,
                                        const EngineOptions& opts,
                                        std::string& err);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    const Tokenizer&  tokenizer() const noexcept { return tok_; }
    // Crown-arch (qwen35moe) config view; meaningless when arch() is dense.
    const QwenConfig& config()    const noexcept { return model_.config(); }
    ModelArch arch()    const noexcept { return arch_; }
    uint32_t  max_ctx() const noexcept { return opts_.max_ctx; }
    uint32_t  vocab()   const noexcept {
        if (is_dense_arch(arch_))             return dcfg_.vocab;   // valid in both single-GPU + TP
        if (arch_ == ModelArch::kQwen35Dense)
            return qwen35_tp_    ? qwen35_tp_model_.config().dense.vocab
                 : qwen35_split_ ? qwen35_split_model_.config().dense.vocab
                                 : qwen35_.config().dense.vocab;
        if (arch_ == ModelArch::kQwen3Moe)    return q3moe_tp_    ? q3moe_tp_model_.config().dense.vocab
                                                 : q3moe_split_ ? q3moe_split_model_.config().dense.vocab
                                                                  : q3moe_.config().dense.vocab;
        if (arch_ == ModelArch::kGptOss)      return gptoss_tp_ ? gptoss_tp_model_.config().dense.vocab
                                                                : gptoss_.config().dense.vocab;
        if (arch_ == ModelArch::kQwen3Next)   return next_model_.config().hybrid.dense.vocab;
        if (arch_ == ModelArch::kGemma4)      return gemma_model_.config().vocab;
        if (qwen35moe_split_)                 return qwen35moe_split_model_.config().vocab;
        return model_.config().vocab;
    }

    // cache_prefix_len: token count of the STABLE conversation prefix (the prompt
    // without the trailing generation-prompt/think suffix). The prompt cache
    // snapshots its endpoint there so the NEXT turn — which shares that stable
    // prefix but diverges at the gen-prompt — restores it. 0 = snapshot at full
    // depth (exact-repeat reuse only; used by direct/non-chat callers).
    GenerateResult generate(const std::string& prompt,
                            const SamplingParams& sp,
                            const TokenCallback& on_token = {},
                            uint32_t cache_prefix_len = 0);

    // `tools_json`: raw OpenAI `tools` array; empty = no tools (template
    // output byte-identical to the pre-tools behavior).
    GenerateResult chat(std::span<const ChatTurn> turns,
                        const SamplingParams& sp,
                        const TokenCallback& on_token = {},
                        bool enable_thinking = true,
                        std::string_view tools_json = {});

private:
    Engine() = default;
    // Arch-tagged backend pair (no virtual interface yet — two backends):
    // model_ is engaged when arch_ == kQwen35Moe (crown path, unchanged);
    // dense_ when arch_ == kQwen3Dense (dn_ stays uninit — reset() no-ops).
    sycl::event forward_step(sycl::queue& q, const int32_t* ids, uint32_t T,
                             uint32_t pos) {
        if (tp_) {
            // Multi-GPU tensor-parallel (dense only). `ids` is a device pointer
            // (d_ids_+pos); DenseModelTP wants host ids and writes host logits.
            // reset_kv on the first step of a generate (pos==0). The summed logits
            // are copied into d_logits_ so the existing GPU sampler is unchanged.
            if (tp_ids_host_.size() < T) tp_ids_host_.resize(T);
            q.memcpy(tp_ids_host_.data(), ids, T * sizeof(int32_t)).wait();
            if (auto m = tp_model_.forward(tp_ids_host_.data(), T, pos,
                                           /*reset_kv=*/(pos == 0),
                                           tp_logits_host_.data()); !m.empty())
                std::fprintf(stderr, "TP forward: %s\n", m.c_str());
            return q.memcpy(d_logits_, tp_logits_host_.data(),
                            uint64_t(vocab()) * sizeof(sycl::half));
        }
        if (next_) {
            // Qwen3-Next-80B layer-split fleet. Like the TP path: host ids in,
            // host logits out (model owns per-card KV/DeltaNet), bounced into
            // d_logits_ so the existing GPU sampler is unchanged.
            if (tp_ids_host_.size() < T) tp_ids_host_.resize(T);
            q.memcpy(tp_ids_host_.data(), ids, T * sizeof(int32_t)).wait();
            if (auto m = next_model_.forward(tp_ids_host_.data(), T, pos,
                                             /*reset_kv=*/(pos == 0),
                                             tp_logits_host_.data()); !m.empty())
                std::fprintf(stderr, "qwen3next forward: %s\n", m.c_str());
            return q.memcpy(d_logits_, tp_logits_host_.data(),
                            uint64_t(vocab()) * sizeof(sycl::half));
        }
        if (qwen35_tp_) {
            // Qwen3.6-27B TENSOR-PARALLEL fleet (both cards on the same token).
            // Same host-bounce as the split path: host ids in, host logits out.
            if (tp_ids_host_.size() < T) tp_ids_host_.resize(T);
            q.memcpy(tp_ids_host_.data(), ids, T * sizeof(int32_t)).wait();
            if (auto m = qwen35_tp_model_.forward(tp_ids_host_.data(), T, pos,
                                                  /*reset_kv=*/(pos == 0),
                                                  tp_logits_host_.data()); !m.empty())
                std::fprintf(stderr, "qwen35tp forward: %s\n", m.c_str());
            return q.memcpy(d_logits_, tp_logits_host_.data(),
                            uint64_t(vocab()) * sizeof(sycl::half));
        }
        if (qwen35_split_) {
            // Qwen3.6-27B layer-split fleet (same host-bounce as the next_ path):
            // host ids in, host logits out (model owns per-card KV/DeltaNet),
            // bounced into d_logits_ so the existing GPU sampler is unchanged.
            if (tp_ids_host_.size() < T) tp_ids_host_.resize(T);
            q.memcpy(tp_ids_host_.data(), ids, T * sizeof(int32_t)).wait();
            if (auto m = qwen35_split_model_.forward(tp_ids_host_.data(), T, pos,
                                                     /*reset_kv=*/(pos == 0),
                                                     tp_logits_host_.data()); !m.empty())
                std::fprintf(stderr, "qwen35split forward: %s\n", m.c_str());
            return q.memcpy(d_logits_, tp_logits_host_.data(),
                            uint64_t(vocab()) * sizeof(sycl::half));
        }
        if (qwen35moe_split_) {
            // Crown (kQwen35Moe) all-Q8_0 layer-split fleet. Same host-bounce as the
            // 27B split path: host ids in, host logits out (model owns per-card
            // KV/DeltaNet), bounced into d_logits_ so the GPU sampler is unchanged.
            if (tp_ids_host_.size() < T) tp_ids_host_.resize(T);
            q.memcpy(tp_ids_host_.data(), ids, T * sizeof(int32_t)).wait();
            if (auto m = qwen35moe_split_model_.forward(tp_ids_host_.data(), T, pos,
                                                        /*reset_kv=*/(pos == 0),
                                                        tp_logits_host_.data()); !m.empty())
                std::fprintf(stderr, "qwen35moe_split forward: %s\n", m.c_str());
            return q.memcpy(d_logits_, tp_logits_host_.data(),
                            uint64_t(vocab()) * sizeof(sycl::half));
        }
        if (q3moe_split_) {
            // Qwen3 MoE (kQwen3Moe) layer-split fleet (Coder-30B / Tongyi). Same
            // host-bounce as the crown split: host ids in, host logits out (model
            // owns per-card KV), bounced into d_logits_ so the GPU sampler is unchanged.
            if (tp_ids_host_.size() < T) tp_ids_host_.resize(T);
            q.memcpy(tp_ids_host_.data(), ids, T * sizeof(int32_t)).wait();
            if (auto m = q3moe_split_model_.forward(tp_ids_host_.data(), T, pos,
                                                    /*reset_kv=*/(pos == 0),
                                                    tp_logits_host_.data()); !m.empty())
                std::fprintf(stderr, "q3moe_split forward: %s\n", m.c_str());
            return q.memcpy(d_logits_, tp_logits_host_.data(),
                            uint64_t(vocab()) * sizeof(sycl::half));
        }
        if (q3moe_tp_) {
            // Qwen3 MoE (kQwen3Moe) TENSOR-PARALLEL fleet (Coder-30B / Tongyi):
            // head-sharded attention (both cards compute over half the heads + half
            // the KV) + expert-sharded MoE. Same host-bounce as the other TP paths.
            if (tp_ids_host_.size() < T) tp_ids_host_.resize(T);
            q.memcpy(tp_ids_host_.data(), ids, T * sizeof(int32_t)).wait();
            if (auto m = q3moe_tp_model_.forward(tp_ids_host_.data(), T, pos,
                                                 /*reset_kv=*/(pos == 0),
                                                 tp_logits_host_.data()); !m.empty())
                std::fprintf(stderr, "q3moe_tp forward: %s\n", m.c_str());
            return q.memcpy(d_logits_, tp_logits_host_.data(),
                            uint64_t(vocab()) * sizeof(sycl::half));
        }
        if (gptoss_tp_) {
            // gpt-oss tensor-parallel fleet (MXFP4 experts sharded, attn replicated).
            // Same host-bounce as the other TP paths.
            if (tp_ids_host_.size() < T) tp_ids_host_.resize(T);
            q.memcpy(tp_ids_host_.data(), ids, T * sizeof(int32_t)).wait();
            if (auto m = gptoss_tp_model_.forward(tp_ids_host_.data(), T, pos,
                                                  /*reset_kv=*/(pos == 0),
                                                  tp_logits_host_.data()); !m.empty())
                std::fprintf(stderr, "gptoss-tp forward: %s\n", m.c_str());
            return q.memcpy(d_logits_, tp_logits_host_.data(),
                            uint64_t(vocab()) * sizeof(sycl::half));
        }
        if (is_dense_arch(arch_))
            return dense_.forward(q, ids, T, pos, kv_, d_logits_);
        if (arch_ == ModelArch::kQwen35Dense)
            return qwen35_.forward(q, ids, T, pos, kv_, dn_, d_logits_);
        if (arch_ == ModelArch::kQwen3Moe)
            return q3moe_.forward(q, ids, T, pos, kv_, d_logits_);
        if (arch_ == ModelArch::kGptOss)
            return gptoss_.forward(q, ids, T, pos, kv_, d_logits_);
        if (arch_ == ModelArch::kGemma4)
            // Gemma self-manages per-layer KV (kv_ unused); writes device logits
            // directly. Each gen overwrites the cache from slot 0, and attention
            // reads only [0,pos], so no explicit reset is needed at pos==0.
            return gemma_model_.forward(q, ids, T, pos, kv_, d_logits_);
        return model_.forward(q, ids, T, pos, kv_, dn_, d_logits_);
    }
    GgufReader      gguf_;
    DeviceAllocator alloc_;
    ModelArch       arch_ = ModelArch::kQwen35Moe;
    QwenModel       model_;
    DenseModel      dense_;
    Qwen35DenseModel qwen35_;
    Qwen3MoeModel   q3moe_;
    GptOssModel     gptoss_;          // gpt-oss (OpenAI MoE): sinks + alternating SWA + top-4 MXFP4 MoE
    Gemma4Model     gemma_model_;     // Gemma 4 (self-manages per-layer KV)
    DenseConfig     dcfg_;            // dense config (set at load; used by vocab() in both modes)
    KvCache         kv_;
    DeltaNetState   dn_;
    // Prompt/KV cache (crown kQwen35Moe single-GPU only; gated on opts_.prompt_cache).
    // Token-trie of (KV slab, DeltaNet snapshot) endpoints; lets generate() restore a
    // cached prefix instead of re-prefilling it. Inert unless prompt_cache_on_.
    PrefixCache     prefix_cache_;
    bool            prompt_cache_on_ = false;
    Tokenizer       tok_;
    EngineOptions   opts_;
    // Multi-GPU tensor-parallel path (dense archs, opts.n_gpus > 1). When tp_ is
    // set, the single-GPU dense_/kv_ are NOT loaded; tp_model_ owns the split
    // weights + per-card KV, and forward_step bounces logits into d_logits_.
    bool            tp_ = false;
    bool            next_ = false;        // Qwen3-Next-80B layer-split fleet path
    bool            qwen35_split_ = false;// Qwen3.6-27B (kQwen35Dense) layer-split fleet path
    bool            qwen35moe_split_ = false;// crown (kQwen35Moe) all-Q8_0 layer-split fleet path
    bool            q3moe_split_ = false;  // Qwen3 MoE (kQwen3Moe) layer-split fleet path
    bool            q3moe_tp_ = false;    // Qwen3 MoE (kQwen3Moe) TENSOR-PARALLEL path (head-shard attn + expert-shard MoE)
    bool            qwen35_tp_ = false;   // Qwen3.6-27B (kQwen35Dense) TENSOR-PARALLEL path
    bool            gptoss_tp_ = false;   // gpt-oss (kGptOss) MoE TENSOR-PARALLEL path
    bool            spec_ = false;        // MTP self-speculative decode (kQwen35Dense single-GPU)
    DeviceFleet     fleet_;
    DenseModelTP    tp_model_;
    Qwen3NextModel  next_model_;          // owns split weights + per-card KV/DeltaNet
    Qwen35SplitModel qwen35_split_model_; // 27B layer-split: per-card weights + KV/DeltaNet
    Qwen35MoeSplitModel qwen35moe_split_model_; // crown all-Q8_0 layer-split: per-card MoE + KV/DeltaNet
    Qwen3MoeSplitModel q3moe_split_model_;      // qwen3moe layer-split: per-card Q4_K/Q6_K MoE + full-attn KV
    Qwen3MoeTpModel q3moe_tp_model_;            // qwen3moe tensor-parallel: head-shard attn (halved KV) + expert-shard MoE
    Qwen35TpModel   qwen35_tp_model_;     // 27B tensor-parallel: per-card sharded FFN + KV/DeltaNet
    GptOssTpModel   gptoss_tp_model_;     // gpt-oss MoE tensor-parallel: per-card sharded MXFP4 experts + KV
    // 80B fleet prompt/KV cache (kQwen3Next; gated on prompt_cache_on_). Per-endpoint
    // vector<KvCache>+vector<DeltaNetState> snapshots, one per card — the crown design
    // run once per device. Declared AFTER next_model_/fleet_ so it destructs FIRST,
    // freeing its per-card snapshots while the fleet allocators are still alive.
    FleetPrefixCache fleet_cache_;
    std::vector<int32_t>    tp_ids_host_;
    std::vector<sycl::half> tp_logits_host_;
    sycl::half*     d_logits_ = nullptr;
    int32_t*        d_ids_    = nullptr;
    int32_t*        d_pick_   = nullptr;
    int32_t*        d_recent_ = nullptr;
    // Per-arch generation stop tokens (besides eos): qwen → {<|im_end|>, -1};
    // llama → {<|eot_id|>, <|end_of_text|>}. Set at load.
    int32_t         stop_ids_[2] = {-1, -1};
    // <tool_call>/</tool_call> ids: preserved through skip-special decode so
    // text-embedded tool calls survive into the OpenAI response content.
    int32_t         tool_call_ids_[2] = {-1, -1};
};

size_t utf8_complete_prefix_len(std::string_view s);

}  // namespace ie
