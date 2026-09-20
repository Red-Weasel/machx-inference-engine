// include/ie/deepseek41_generate.hpp — the generate loop on Ds41Forward (Phase 12 step 3,
// docs/deepseek41/31): prefill (bounded replay as configured; an odd-length prompt is prefilled
// even and its last token stepped, since a prefill's length must be a multiple of the compress
// ratio), then one step per token with host-side sampling over the fp32 logits (temperature,
// top-k, top-p, min-p, repetition penalty over a recent window -- the semantics of the engine's
// sample_softmax_topk_topp / repetition_penalty kernels, on the host because Ds41Forward hands
// its logits back as fp32 on the host), streaming detokenisation that never emits a partial
// UTF-8 sequence, and the stop conditions: eos, the token budget, stop strings.
#pragma once
#include "ie/deepseek41_dspark.hpp"
#include "ie/deepseek41_forward.hpp"
#include "ie/tokenizer.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace ie {

struct Ds41SampleParams {
    float    temperature = 0.f;     // <= 0: argmax
    uint32_t top_k = 0;             // 0: off
    float    top_p = 1.f;           // 1: off
    float    min_p = 0.f;           // 0: off
    float    repeat_penalty = 1.f;  // 1: off; > 1 divides positive logits of recent ids, multiplies negative ones
    uint32_t repeat_window = 64;    // recent ids the penalty sees (prompt tail + output)
    uint64_t seed = 42;
};

struct Ds41GenStats {
    double   prefill_s = 0, decode_s = 0;
    double   restore_s = 0;         // of prefill_s: restoring the cached prefix
    uint32_t n_prompt = 0, n_gen = 0;
    uint32_t n_cached = 0;          // Phase 46: prompt tokens served from the prefix cache (not run)
    std::string cache_source;       // "live" | "checkpoint" | "host slot" | "none" (empty: the cache is off)
    std::string stop_reason;        // "eos" | "length" | "stop" | "callback" | "error"
    // DSpark P5 (docs/deepseek41/59): the steady-state window -- the decode time and token count after the warm-up
    // mark, because the cold steps right after a prefill dominate a short run (docs/58 criterion 6)
    double   warm_decode_s = 0; uint32_t warm_n = 0;
    // DSpark P4 (docs/deepseek41/58): the speculative loop's accounting (all zero when the plain loop ran)
    uint32_t spec_passes = 0, spec_hist[7] = {};           // passes by their L (1 .. 6 rows committed per pass)
    uint32_t spec_rows_verified = 0, spec_declined = 0;    // P5: Σ (1 + k_pass) over passes; passes that declined to speculate
    uint64_t spec_link_bytes = 0;                          // P5: Σ over the verify steps of the layers' bytes_pinned + bytes_mmap
    uint64_t spec_union_sum = 0; uint32_t spec_union_layers = 0;   // P5: Σ experts_uploaded over those steps' layers, and their count
    uint32_t spec_offered[5] = {}, spec_accepted[5] = {};  // per draft position, the judged drafts only
    struct SpecConf { uint8_t pos, accepted; float conf; };
    std::vector<SpecConf> spec_conf;                       // one record per judged draft (the confidence head, measured)
    double spec_draft_ms = 0, spec_verify_ms = 0, spec_rollback_ms = 0;
    // Phase 58 (docs/deepseek41/97): prompt-lookup speculation -- passes that verified a copied draft, their rows and
    // accepted drafts, and the one-row steps between them
    uint32_t lookup_passes = 0, lookup_rows = 0, lookup_accepted = 0, lookup_plain = 0;
    double   lookup_verify_ms = 0, lookup_plain_ms = 0;
    // Phase 50: the forward's per-layer stages summed over every plain decode step (set_accumulate_stages)
    struct StageAcc { uint32_t steps = 0; double att = 0, fpre = 0, moe = 0, grp = 0, join = 0, mmg = 0, tail = 0, shd = 0, lay = 0, cpu_ms = 0, cpu_w = 0, bp = 0, bm = 0, sta = 0, pin = 0, mmx = 0, sh = 0, cpu = 0, wall = 0, prep = 0, head = 0, sample = 0, emit = 0, mm_fill = 0, mm_pack = 0, mm_read = 0, mm_perm = 0, spawn = 0, mprep = 0; } stages;
};

class Ds41Generator {
public:
    Ds41Generator(Ds41Forward& fwd, const Tokenizer& tok) : fwd_(fwd), tok_(tok) {}
    // Generates up to `max_new` tokens after `prompt_ids`; `on_piece` receives the text as it
    // becomes complete UTF-8 (return false to stop); `out_ids` gets the generated ids (the
    // stopping eos excluded). Returns "" or a diagnostic (also in st.stop_reason = "error").
    std::string run(const std::vector<int32_t>& prompt_ids, uint32_t max_new, const Ds41SampleParams& sp,
                    const std::vector<std::string>& stops,
                    const std::function<bool(std::string_view)>& on_piece,
                    std::vector<int32_t>& out_ids, Ds41GenStats& st);
    // The sampler alone (host): the chosen id for `logits`, with the penalty over `recent`.
    static int32_t sample(std::vector<float>& logits, const std::vector<int32_t>& recent, const Ds41SampleParams& sp, uint64_t& rng_state);
    static int32_t sample_row(float* logits, size_t V, const std::vector<int32_t>& recent, const Ds41SampleParams& sp, uint64_t& rng_state);
    // DSpark P4 (docs/deepseek41/58): with a drafter attached, run() speculates at temperature <= 0 -- five drafts
    // verified in one T = 6 step, greedy acceptance with the plain sampler, rollback_to, the rings re-seeded;
    // token for token the plain loop's stream. At temperature > 0 the plain loop runs (said once on stderr).
    void set_drafter(Ds41Drafter* d, sycl::queue* q) { drafter_ = d; dq_ = q; }
    // DSpark P5 (docs/deepseek41/59) — measurement knobs: they change how many rows the verify carries, never any
    // arithmetic. The verify block's length (0 = the drafter's whole block); the confidence threshold (the leading
    // run of drafts scoring at or above it is what gets verified; a run of 0 makes the pass a plain one-row step);
    // and the warm-up mark the steady-state window starts at.
    // Phase 25/27: reset the forward's routing profile at the PREFILL/DECODE boundary, so what it accumulates
    // is decode-phase routing for THIS prompt. A profile taken over the whole run is a prefill histogram
    // (expert_stream.hpp:707), and a profile taken on synthetic text is worse than useless -- measured: a
    // ranking profiled on a repeated block made real-text decode 37% SLOWER (5.46 -> 3.45 tok/s).
    void set_profile_decode_only(bool on) { profile_decode_only_ = on; }

    void set_spec_k(uint32_t k) { spec_k_ = k; }
    void set_spec_conf(float th) { spec_conf_ = th; }
    void set_warmup_mark(uint32_t n) { warm_mark_ = n; }
    void set_accumulate_stages(bool on) { acc_stages_ = on; }   // Phase 50: needs IE_DS41_STAGES=1 for the stage timers to be valid
    // Phase 58: prompt-lookup speculation for this generator: -1 = the IE_DS41_LOOKUP environment (the default), 0 off, 1 on
    void set_lookup(int on) { lookup_ = on; }
private:
    Ds41Forward& fwd_;
    const Tokenizer& tok_;
    Ds41Drafter* drafter_ = nullptr; sycl::queue* dq_ = nullptr;
    bool                profile_decode_only_ = false;
    uint32_t spec_k_ = 0, warm_mark_ = 0; float spec_conf_ = -1e30f;
    bool acc_stages_ = false;
    int  lookup_ = -1;
};

}  // namespace ie
