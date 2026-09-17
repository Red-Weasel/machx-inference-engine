// include/ie/engine.hpp — product-facing API: load once, generate with a
// streaming callback.  v1: one implicit session; server treats requests as
// stateless (reset + full-conversation prefill per request).
#pragma once
#include "ie/qwen4_vision.hpp"
#include "ie/ds4_vision.hpp"
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
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace ie {

// DeepSeek-V4-Flash (kDeepSeek4) state, held behind an opaque handle so this
// header does not have to pull in deepseek4.hpp — the bound model, BOTH runtimes
// (single-card + tensor-parallel), the pinned host arena and the expert cache.
// Defined in engine.cpp; non-null only on the deepseek4 path.
struct Ds4Bundle;

// Qwen3.8-Flash-Next (kQwen4Exp) state, same opaque-handle pattern: config +
// one Qwen4ExpModel per card (1-GPU whole model, or the bit-identical 2-GPU
// layer-split pipeline). Defined in engine.cpp; non-null only on that path.
struct Q4eBundle;
struct Glm5Bundle;
// DeepSeek-V4.1-Flash (kDeepSeek41): a model DIRECTORY (safetensors), the resident two-card
// runtime, the tokenizer.json tokenizer and the V4.1 prompt format. src/engine/ds41_engine.cpp.
struct Ds41Bundle;

// Pinned (USM host) staging buffer for the host-bounced logits row. Every
// multi-card / host-runtime path (DenseModelTP, qwen3next, qwen35 TP/split,
// deepseek4, glm5next, qwen4exp) lands the vocab row here and the engine queue
// copies it to d_logits_ once per token; with a pageable std::vector that H2D
// is a driver-staged copy. Bound to the engine allocator's context once at
// load (Engine::create); before bind(), or if the USM allocation fails, it
// degrades to pageable malloc so nothing else changes. Same resize/size/data
// surface as the vector it replaces; contents survive a growing resize.
class PinnedHalfBuf {
public:
    PinnedHalfBuf() = default;
    ~PinnedHalfBuf() { release(); }
    PinnedHalfBuf(const PinnedHalfBuf&) = delete;
    PinnedHalfBuf& operator=(const PinnedHalfBuf&) = delete;
    void bind(const sycl::context& c) { ctx_ = c; }
    void resize(size_t n) {
        if (n <= cap_) { n_ = n; return; }
        sycl::half* np = nullptr;
        bool np_pinned = false;
        if (ctx_) {
            np = static_cast<sycl::half*>(sycl::malloc_host(n * sizeof(sycl::half), *ctx_));
            np_pinned = (np != nullptr);
        }
        if (!np) np = static_cast<sycl::half*>(std::malloc(n * sizeof(sycl::half)));
        if (!np) throw std::bad_alloc();
        if (n_) std::memcpy(np, p_, n_ * sizeof(sycl::half));
        release();
        p_ = np; pinned_ = np_pinned; cap_ = n; n_ = n;
    }
    size_t            size()  const noexcept { return n_; }
    sycl::half*       data()        noexcept { return p_; }
    const sycl::half* data()  const noexcept { return p_; }
    sycl::half*       begin()       noexcept { return p_; }
    sycl::half*       end()         noexcept { return p_ + n_; }
    sycl::half&       operator[](size_t i)       noexcept { return p_[i]; }
    const sycl::half& operator[](size_t i) const noexcept { return p_[i]; }
    bool              pinned() const noexcept { return pinned_; }
private:
    void release() {
        if (!p_) return;
        if (pinned_) sycl::free(p_, *ctx_); else std::free(p_);
        p_ = nullptr; cap_ = n_ = 0; pinned_ = false;
    }
    std::optional<sycl::context> ctx_;
    sycl::half* p_   = nullptr;
    size_t      n_   = 0;
    size_t      cap_ = 0;
    bool        pinned_ = false;
};

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
    float    presence_penalty = 0.0f;
    float    frequency_penalty = 0.0f;
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
    uint32_t cpu_threads   = 0; // 0 preserves the model CPU expert team default
    bool     int8_kv       = false;
    uint32_t prefill_chunk = 256;    // hard cap per docs/known_bugs.md
    uint32_t n_gpus        = 1;      // >1 → tensor-parallel split (dense archs only)
    uint32_t parallel      = 1;      // concurrent generations the engine admits (1-4).
                                     // >1 → generations run under the internal FIFO
                                     // gate; on the 27B split path decode is BATCHED
                                     // (joint-step scheduler over per-slot state
                                     // banks), other archs take whole-generation
                                     // FIFO turns. CLI: --parallel.
    uint32_t slot_ctx      = 0;      // per-slot ctx budget for the decode banks
                                     // (--parallel>1, split path). 0 = auto:
                                     // min(max_ctx, 65536). Each of the `parallel`
                                     // banks holds a full KV at this depth, so it
                                     // bounds VRAM; bank alloc failure is a clean
                                     // load-time refusal. CLI: --slot-ctx.
    bool     prompt_cache  = true;   // reuse cached prefix KV+DN across turns (crown
                                     // single-GPU + Qwen3-Next-80B fleet). Default-ON;
                                     // opt out via env IE_NO_PROMPT_CACHE=1 (or =false).
    bool     spec          = false;  // MTP self-speculative decode (Qwen3.6-27B
                                     // kQwen35Dense, GREEDY only; single/split/TP). Loads
                                     // the native MTP/NextN head (~0.6 GB) and runs
                                     // draft+verify; lossless vs plain greedy. CLI: --spec.
    uint32_t spec_k        = 2;      // draft length. K=2 is the MEASURED optimum for the
                                     // 27B (verify cost grows ~linearly with K but accept
                                     // saturates): clean-box K=2 1.12× / K=3 1.06× / K=4
                                     // 0.98× (a LOSS). Lossless at all K. Override --spec-k.
    std::string spec_head;           // gemma4 MTP draft head GGUF (separate file,
                                     // mtp-gemma-4-*-Q8_0.gguf). Empty → convention:
                                     // first mtp-*.gguf next to the target. CLI: --spec-head.
    std::string spec_draft;          // dspark separate-draft GGUF (kQwen35Dense target,
                                     // e.g. *-dspark-Q4_1.gguf). Loads a target-conditioned
                                     // DsparkDrafter; lossless GREEDY spec. CLI: --spec-draft.
};

struct GenerateResult {
    std::string text;
    // deepseek4 (docs/deepseek4/72 Phase L): the completion parsed by the
    // model's own format — reasoning split off, tool calls as an OpenAI-format
    // JSON array ("" when none).  Other archs leave both empty.
    std::string reasoning_content;
    std::string tool_calls_json;
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
    const std::string& memory_residency_json() const noexcept { return memory_residency_json_; }
    // Crown-arch (qwen35moe) config view; meaningless when arch() is dense.
    const QwenConfig& config()    const noexcept { return model_.config(); }
    ModelArch arch()    const noexcept { return arch_; }
    // A DeepSeek-V4.1 model directory (config.json names deepseek_v41)? Such a path loads without GGUF.
    static bool ds41_dir(const std::string& path);
    uint32_t  max_ctx() const noexcept { return opts_.max_ctx; }
    uint32_t  parallel() const noexcept { return opts_.parallel; }
    uint32_t  vocab()   const noexcept {
        // deepseek4 first: its config is inside the opaque Ds4Bundle, so the
        // count is cached here at load (and none of the branches below apply).
        if (arch_ == ModelArch::kDeepSeek4)   return ds4_vocab_;
        if (arch_ == ModelArch::kQwen4Exp)    return q4e_vocab_;
        if (arch_ == ModelArch::kGlm5Next)    return glm5_vocab_;
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
    // reply_cache: also snapshot the state at the end of the reply, keyed by
    // prompt++reply, on the trie-backed prompt caches. Only chat() sets it, and
    // only for templates whose history render of an assistant turn equals the
    // generation prompt + content (otherwise the next turn's tokens diverge
    // right after the assistant header and the endpoint could never hit).
    GenerateResult generate(const std::string& prompt,
                            const SamplingParams& sp,
                            const TokenCallback& on_token = {},
                            uint32_t cache_prefix_len = 0,
                            bool reply_cache = false);

    // `tools_json`: raw OpenAI `tools` array; empty = no tools (template
    // output byte-identical to the pre-tools behavior).
    GenerateResult chat(std::span<const ChatTurn> turns,
                        const SamplingParams& sp,
                        const TokenCallback& on_token = {},
                        bool enable_thinking = true,
                        std::string_view tools_json = {},
                        std::string_view reasoning_effort = {});
    // Metadata-only admission validation; safe before sending HTTP/SSE headers.
    std::string reasoning_effort_error(std::string_view effort) const;

private:
    Engine() = default;
    // --- Multi-slot gate (agent serving, Phase 1b). Every generation runs its
    // GPU work under this FIFO ticket lock; when contended, the 27B-split
    // decode/prefill loops yield it at slice boundaries (host-staged slot
    // stash), so up to EngineOptions::parallel requests interleave. With one
    // request in flight, acquire/release are uncontended and no yield fires —
    // behavior and cost identical to the pre-gate engine. ---
    struct Gate {
        std::mutex mu;
        std::condition_variable cv;
        uint64_t next = 0;        // next ticket to hand out
        uint64_t serving = 0;     // ticket currently allowed to run
        std::atomic<uint64_t> inflight{0};   // holders + waiters, lock-free read
        void acquire() {
            std::unique_lock<std::mutex> l(mu);
            const uint64_t my = next++;
            inflight.store(next - serving, std::memory_order_relaxed);
            cv.wait(l, [&] { return serving == my; });
        }
        void release() {
            {
                std::lock_guard<std::mutex> l(mu);
                ++serving;
                inflight.store(next - serving, std::memory_order_relaxed);
            }
            cv.notify_all();
        }
        // Approximate by design: read without the lock. A stale read costs at
        // worst one extra slice before yielding or one unneeded stash.
        bool contended() const {
            return inflight.load(std::memory_order_relaxed) > 1;
        }
    } gate_;

    // --- Joint-step decode scheduler (decode-throughput campaign, Phase 2b).
    // Spawned when --parallel>1 on the 27B split path. Request threads prefill
    // exactly as before (exclusive gate, chunk-yield), take their first token
    // from the live prefill logits, bank_store their state, RELEASE their gate
    // ticket, and enroll here; the stepper then drives batched decode: one
    // gate hold per step, forward_slots over the enrolled group, per-slot
    // repeat-penalty + sampling on-device from the slot-logits rows, picks
    // published to per-slot outboxes. Cancellation = want_stop, honored at
    // step boundaries. ---
    // Effective split-path prompt-cache caps (recorded at init so the VRAM
    // budget check can price the cache's worst case).
    FleetPrefixCacheConfig split_cache_cfg_{};
    struct BatchStepper {
        struct Slot {
            bool     used = false;        // bank ownership (request lifetime)
            bool     running = false;     // enrolled in the step group
            bool     want_stop = false;   // request-side abort
            bool     finished = false;
            uint32_t pos = 0;             // position of the pending token
            int32_t  pending = 0;         // token to forward next step
            SamplingParams sp{};
            uint64_t rng = 0;
            uint32_t made = 0, budget = 0;
            std::vector<int32_t> hist;    // prompt ++ generated (repeat window)
            std::deque<int32_t>  outbox;  // picks for the request thread
            std::string finish_reason;
        };
        std::mutex              mu;
        std::condition_variable cv;       // shared: stepper + request threads
        std::vector<Slot>       slots;
        std::thread             th;
        bool                    shutdown = false;
        int32_t* d_pick   = nullptr;      // [parallel] on head_dev
        int32_t* d_recent = nullptr;      // [512] repeat window on head_dev
    } stepper_;
    void stepper_spawn_();
    void stepper_loop_();
    // Arch-tagged backend pair (no virtual interface yet — two backends):
    // model_ is engaged when arch_ == kQwen35Moe (crown path, unchanged);
    // dense_ when arch_ == kQwen3Dense (dn_ stays uninit — reset() no-ops).
    sycl::event forward_step(sycl::queue& q, const int32_t* ids, uint32_t T,
                             uint32_t pos) {
        // DeepSeek-V4-Flash. Out of line because the runtimes are opaque here;
        // same host-bounce contract as the fleet paths (host ids in, host logits
        // out) except the logits arrive fp32 and are narrowed on the way in.
        if (glm5_) return glm5_forward(q, ids, T, pos);
        if (q4e_) return q4e_forward(q, ids, T, pos);
        if (ds4_) return ds4_forward(q, ids, T, pos);
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
    // deepseek4 forward, defined in engine.cpp (Ds4Bundle is incomplete here).
    sycl::event ds4_forward(sycl::queue& q, const int32_t* ids, uint32_t T, uint32_t pos);
    // deepseek41 (src/engine/ds41_engine.cpp): the directory load and the chat/generate routes
    std::string    ds41_load(const std::string& dir);
    GenerateResult ds41_chat(std::span<const ChatTurn> turns, const SamplingParams& sp, const TokenCallback& on_token,
                            bool enable_thinking, std::string_view tools_json, std::string_view reasoning_effort);
    GenerateResult ds41_generate(const std::string& prompt, const SamplingParams& sp, const TokenCallback& on_token);
    // qwen4exp forward, defined in engine.cpp (Q4eBundle is incomplete here).
    sycl::event q4e_forward(sycl::queue& q, const int32_t* ids, uint32_t T, uint32_t pos);
    sycl::event glm5_forward(sycl::queue& q, const int32_t* ids, uint32_t T, uint32_t pos);
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
    bool            spec_draft_ = false;  // dspark separate-draft spec decode (kQwen35Dense single-GPU)
    std::unique_ptr<DsparkDrafter> dspark_drafter_;  // loaded only with --spec-draft
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
    // DeepSeek-V4-Flash. Owns the bound model (which points into gguf_'s mmap),
    // the per-card runtimes, their pinned host arena and their VRAM expert cache.
    // Declared HERE, after every fleet member and long after gguf_, so it
    // destructs before them — releasing the cards' memory while the reader whose
    // mmap its tensors point at is still alive.
    std::unique_ptr<Ds4Bundle> ds4_;
    std::unique_ptr<Ds41Bundle> ds41_;   // deepseek41 only; frees its runtime in its own destructor
    uint32_t        ds4_vocab_ = 0;       // cached: vocab() cannot see into Ds4Bundle
    // DeepSeek-V4-Flash-Vision-Exp (docs/deepseek4/70_VISION_EXP_PORT_PLAN.md P3/P4).
    // load() finds the vision sidecar beside the GGUF and uploads the tower on
    // GPU 0 BEFORE the runtime plans its expert arena; chat() decodes each image
    // (official resize/pad) and prepends one placeholder token per image;
    // generate() expands every placeholder into its N-layout block (CPU), then
    // encodes the images under the GPU gate and stages the rows + spans here;
    // ds4_forward applies them right after its pos==0 reset. Declared AFTER
    // alloc_: the tower frees pinned host memory on alloc_'s queue.
    std::unique_ptr<Ds4Vision> ds4_vis_;
    std::string     ds4_vis_path_;        // resolved sidecar ("" = text-only file)
    struct Ds4PendingImage {
        std::vector<float>   px;          // [3, H, W] normalised
        uint32_t             H = 0, W = 0;
        Ds4VisGeom           geom;
        std::vector<int32_t> types, perm; // block layout, set by ds4_expand_vision
    };
    std::vector<Ds4PendingImage> ds4_pending_imgs_;
    struct Ds4VisSpan { uint32_t t0, n, start_off; };
    std::vector<Ds4VisSpan> ds4_vis_spans_;
    std::vector<float>      ds4_vis_rows_;     // concat block rows, span order
    bool                    ds4_vis_active_ = false;
    std::string ds4_expand_vision(std::vector<int32_t>& ids);   // placeholders -> blocks (CPU)
    std::string ds4_stage_vision();                             // encode + assemble rows (GPU)
    void        ds4_apply_vision();
    // Qwen3.8-Flash-Next bundle (same destruction-order rationale as ds4_).
    std::string memory_residency_json_ = "null";
    std::unique_ptr<Glm5Bundle> glm5_; // destroyed before allocators and GGUF mmap
    uint32_t glm5_vocab_ = 0;
    std::string glm5_err_;
    std::unique_ptr<Q4eBundle> q4e_;
    uint32_t        q4e_vocab_ = 0;
    std::string     q4e_err_;             // forward-failure latch (mirrors ds4_err_)
    // Vision (qwen4exp only; docs/qwen4/16_vision_port.md §4). chat() decodes
    // each image and rewrites the turn's content with the pad run; generate()
    // encodes the pending images and stages the splice + M-RoPE table here;
    // q4e_forward applies them right AFTER its pos==0 reset (which would
    // otherwise wipe them). Staging clears at the next generate().
    std::unique_ptr<Qwen4Vision> q4e_vis_;
    std::string     model_path_;          // gguf path from load() (mmproj discovery)
    std::string     q4e_mmproj_;          // resolved mmproj path ("" = none found)
    struct Q4ePendingImage { std::vector<float> px; uint32_t H, W; };
    std::vector<Q4ePendingImage> q4e_pending_imgs_;
    std::vector<float>   q4e_vis_rows_;   // concat merged rows, span order
    std::vector<std::pair<uint32_t, uint32_t>> q4e_vis_spans_;  // (t0, n)
    std::vector<int32_t> q4e_pos3_;       // [3, n_prompt]
    int32_t              q4e_delta_ = 0;
    bool                 q4e_vis_active_ = false;
    // (engine.cpp) Encode the pending images against the tokenized prompt and
    // fill the staging above; apply staging to the bundle (A splice + A/B
    // mrope). Apply runs at q4e_forward's pos==0 (after its reset) or right
    // after a prompt-cache restore.
    std::string q4e_stage_vision(const std::vector<int32_t>& ids);
    void        q4e_apply_vision();
    // Deepest prefix the deepseek4 prompt cache will snapshot. Capped at load
    // because the snapshot (22.55 MiB + 6.88 kB/token PER CARD at the default fp16 values) is drawn from
    // the same reserve the expert arena is derived from, and trading expert
    // residency for cached prefix is a decode-throughput loss. See the load site.
    uint32_t        ds4_cache_max_prefix_ = 0;
    // Last deepseek4 forward error. forward_step cannot return one (it returns a
    // sycl::event), and continuing past a failed forward would sample stale
    // logits into text that LOOKS like output — so the failure is latched here
    // and generate() stops on it. Cleared at the top of every generate().
    std::string     ds4_err_;
    std::vector<int32_t>    tp_ids_host_;
    PinnedHalfBuf           tp_logits_host_;   // pinned; bound in Engine::create
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
