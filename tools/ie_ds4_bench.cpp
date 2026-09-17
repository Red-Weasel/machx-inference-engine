// tools/ie_ds4_bench.cpp — the DeepSeek-V4-Flash performance baseline harness.
//
// WHAT THIS IS FOR.  A number to optimise against, and enough attribution to
// know WHICH number to attack.  It reports time-to-first-token, prefill (pp) at
// several prompt lengths, decode (tg) at several context depths with its
// distribution rather than just a mean, and the expert-streaming counters that
// decide whether this model is compute-bound or DMA-bound.
//
// EVERY NUMBER IS MEASURED.  Wall clocks come from steady_clock around the call
// being timed; DMA bytes, transfer-queue busy time and cache hits/misses come
// from `Ds4ExpertCache::Stats`, which the engine fills from the transfer queue's
// own profiling events; per-kernel GPU time comes from `ie::KernelProfiler` over
// the SYCL events the `ie::ps` wrappers already push (the same mechanism
// `tools/ie_bench.cpp --kprofile-decode` uses).  A metric this harness cannot
// measure prints the literal value `unavailable` with a `[ds4-note]` saying why.
// Nothing is modelled, estimated or defaulted into a plausible-looking figure —
// with ONE exception, `*.overlap_est_frac`, which is derived arithmetic over two
// measured times and carries `_est` in its name for exactly that reason.
//
// THE OUTPUT IS MEANT TO BE DIFFED.  Every metric is one line
//     [ds4-bench] <key>=<value>
// with no formatting inside the value, so `--baseline <previous stdout>` can
// mechanically compare two runs and emit
//     [ds4-delta] <key> base=<..> now=<..> pct=<..>
// That is the whole point: run, change something, run again, read the delta.
//
// THE OPTIMISATION LOOP.  `--profile-out <path>` turns on real expert-selection
// counting (`Ds4ExpertProfile`) and writes the measured residency ranking;
// `--priority-in <path>` loads that ranking back as the order in which experts
// claim the static, never-evicted VRAM slots.  `--loop` does both in one
// process — profile, release, reload with the measured order, re-bench — and
// reports the hit-rate and tok/s delta.  Prior static analysis of the shipped
// tensors found index order already captures 97.5-99.0% of the available gain on
// the only three layers whose usage is computable without a forward pass, so the
// honest expectation is a SMALL win.  The point is that the loop is closed and
// the result is measured either way.
//
// WHAT IT WILL NOT DO.  It does not load a model unless told which one, it does
// not size itself past `--max-seq` (it chunks and says so), and it never prints
// a throughput figure for a run that errored.
//
// Usage:
//   ie-ds4-bench --mini                       # miniature fixture, no big model
//   ie-ds4-bench --gguf <shard1.gguf> --kprofile --profile-out rank.txt
//   DS4_TP_GPUS=0,1 ie-ds4-bench --gguf <shard1.gguf> --baseline prev.txt

#include "ie/deepseek4.hpp"
#include "ie/expert_stream.hpp"
#include "ie/gguf.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/model_config.hpp"

#include "deepseek4/ds4_mini_gguf.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace ie;

double now_s() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::vector<uint32_t> parse_csv_u32(const std::string& s) {
    std::vector<uint32_t> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(uint32_t(std::strtoul(tok.c_str(), nullptr, 10)));
    }
    return out;
}

// ---------------------------------------------------------------------------
// The metric block
// ---------------------------------------------------------------------------
// One `key=value` per line, printed as it is produced (so a run that dies
// half-way still leaves usable data) and retained so `--baseline` can diff it.
// Keys are hierarchical and stable; values carry no units, no separators and no
// commentary — the unit is part of the key (`_ms`, `_bytes`, `_frac`, `tok_s`).
class Emit {
public:
    void s(const std::string& k, const std::string& v) {
        kv_[k] = v;
        std::printf("[ds4-bench] %s=%s\n", k.c_str(), v.c_str());
        std::fflush(stdout);
    }
    void d(const std::string& k, double v, int prec = 3) {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.*f", prec, v);
        s(k, buf);
    }
    void u(const std::string& k, uint64_t v) { s(k, std::to_string(v)); }
    void b(const std::string& k, bool v) { s(k, v ? "1" : "0"); }
    // A metric that could not be measured on this run.  It still occupies its
    // key so a diff shows it disappearing or appearing, and the reason goes to a
    // note line rather than into the value.
    void na(const std::string& k, const std::string& why) {
        s(k, "unavailable");
        note(k + ": " + why);
    }
    static void note(const std::string& t) {
        std::printf("[ds4-note] %s\n", t.c_str());
        std::fflush(stdout);
    }
    const std::map<std::string, std::string>& all() const noexcept { return kv_; }

private:
    std::map<std::string, std::string> kv_;
};

// Reads a previous run's stdout back into a key->value map.
std::map<std::string, std::string> read_baseline(const std::string& path, std::string& err) {
    std::map<std::string, std::string> out;
    std::ifstream f(path);
    if (!f) { err = "cannot open " + path; return out; }
    std::string line;
    const std::string tag = "[ds4-bench] ";
    while (std::getline(f, line)) {
        if (line.rfind(tag, 0) != 0) continue;
        const std::string body = line.substr(tag.size());
        const size_t eq = body.find('=');
        if (eq == std::string::npos) continue;
        out[body.substr(0, eq)] = body.substr(eq + 1);
    }
    if (out.empty()) err = path + " contained no [ds4-bench] lines";
    return out;
}

bool numeric(const std::string& v, double& out) {
    if (v.empty() || v == "unavailable") return false;
    char* end = nullptr;
    const double d = std::strtod(v.c_str(), &end);
    if (end == v.c_str() || *end != 0) return false;
    out = d;
    return true;
}

// ---------------------------------------------------------------------------
// One handle over "single card" and "n cards in lockstep"
// ---------------------------------------------------------------------------
// Dispatch only — it computes nothing, so neither path can acquire behaviour the
// other does not have.  Copied in spirit from deepseek4_forward_test.cpp's
// FwdDriver so the two agree on what "card 0" means.
struct Driver {
    DeepSeek4Runtime*   single = nullptr;
    DeepSeek4TpRuntime* tp     = nullptr;

    std::string forward(const int32_t* ids, uint32_t T, uint32_t pos0, float* lg) {
        return tp ? tp->forward(ids, T, pos0, lg) : single->forward(ids, T, pos0, lg);
    }
    void     reset_context() { if (tp) tp->reset_context(); else single->reset_context(); }
    uint32_t n_cards() const { return tp ? tp->n_cards() : 1u; }
    DeepSeek4Runtime& card(uint32_t c) { return tp ? tp->card(c) : *single; }
    DeepSeek4Runtime& card0() { return card(0); }
    bool ready() const { return tp != nullptr || single != nullptr; }
};

// Expert-cache counters for ONE forward, SUMMED OVER EVERY CARD.  The wall clock
// is NOT summed — the cards run concurrently inside one tp.forward(), so the host
// clock around that call is the only correct wall.
//
// READ THAT PAIRING CAREFULLY, BECAUSE IT HAS ALREADY MISLED ONCE.  A ratio built
// from a summed counter and an unsummed wall is an N-CARD figure over a ONE-CARD
// clock: `dma_ms / wall_ms` on two cards reads as twice the per-card PCIe
// utilisation.  Every emitter that forms such a ratio also emits the per-card
// form beside it, and the per-card one is what answers "is the link busy".
struct StepStat {
    double   wall_ms     = 0.0;
    double   dma_ms      = 0.0;   // transfer-queue busy, summed over cards
    uint64_t bytes       = 0;
    uint64_t hits        = 0;
    uint64_t misses      = 0;
    uint64_t acquires    = 0;
    uint64_t unavailable = 0;
    uint64_t spec_issued = 0;
    uint64_t spec_hits   = 0;

    void add(const StepStat& o) {
        wall_ms += o.wall_ms; dma_ms += o.dma_ms; bytes += o.bytes;
        hits += o.hits; misses += o.misses; acquires += o.acquires;
        unavailable += o.unavailable; spec_issued += o.spec_issued; spec_hits += o.spec_hits;
    }
    double hit_rate() const {
        const uint64_t n = hits + misses;
        return n ? double(hits) / double(n) : 0.0;
    }
};

// Reads the per-card cache counters left behind by the forward that just ran.
// `Ds4ExpertCache::reset_stats()` runs at the top of every forward_prologue, so
// these describe exactly one forward.
StepStat collect(Driver& drv, double wall_ms) {
    StepStat s;
    s.wall_ms = wall_ms;
    for (uint32_t c = 0; c < drv.n_cards(); ++c) {
        const Ds4ExpertCache::Stats& st = drv.card(c).cache().stats();
        s.dma_ms      += st.dma_seconds * 1e3;
        s.bytes       += st.bytes_fetched;
        s.hits        += st.hits;
        s.misses      += st.misses;
        s.acquires    += st.acquires;
        s.unavailable += st.unavailable;
        s.spec_issued += st.spec_issued;
        s.spec_hits   += st.spec_hits;
    }
    return s;
}

// ---------------------------------------------------------------------------
// Kernel-time attribution
// ---------------------------------------------------------------------------
// The buckets below are a NAME TABLE, not an inference: every entry is a kernel
// this engine actually submits, filed by the block of the forward pass that
// submits it.  `ds4_dense_*` / `ds4_grouped_*` carry an `@site` suffix from
// src/model/deepseek4.cpp (attention projections vs the shared expert vs the
// lm_head) precisely so this table does not have to guess.  Anything not listed
// lands in `other` and is still printed by name, so an unclassified kernel shows
// up as a visible bucket rather than silently distorting a share.
const char* const kAttnKernels[] = {
    "ds4_attention", "ds4_rope_apply", "ds4_rope_cos_sin", "ds4_sliding_causal_mask",
    "ds4_concat_mask", "ds4_compress_pool", "ds4_indexer_score", "ds4_indexer_topk",
    "ds4_block_bias_dense", "ds4_block_bias_topk_fill", "ds4_block_bias_topk_scatter",
// THE Q8 AND SCALAR NAMES BELONG HERE TOO.  With IE_DS4_DENSE_Q8=1 (the default
// since 2026-08-03) every dense projection submits as `ds4_dense_q8@…`, and none
// of those names was listed — so the whole always-resident dense set fell into
// `other` and the per-bucket shares of a Q8 profile described a different engine
// than the one that ran.  `ds4_dense_f16_scalar@…` is the fallback route
// deepseek4.cpp used to submit under the oneDNN route's own name; it is listed
// separately here for the same reason it is named separately there.
    "ds4_unweighted_rms_norm", "ds4_grouped_linear",
    "ds4_dense_f16@attn", "ds4_dense_packed@attn", "ds4_dense_q8@attn",
    "ds4_dense_f16_scalar@attn", "ds4_grouped_f16_scalar@attn",
    "ds4_grouped_f16@attn", "ds4_grouped_packed@attn", "ds4_grouped_q8@attn"};
const char* const kMoeRoutedKernels[] = {
    "ds4_router_topk", "ds4_router_hash", "gemv_iq3_xxs", "gemv_mxfp4", "gemv_mxfp4_x2",
    "ds4_expert_accum", "quant_q8_1", "cast_f32_f16", "cast_f16_f32"};
const char* const kMoeSharedKernels[] = {
    "ds4_dense_f16@shexp", "ds4_dense_packed@shexp", "ds4_dense_q8@shexp",
    "ds4_dense_f16_scalar@shexp", "ds4_grouped_f16_scalar@shexp",
    "ds4_grouped_f16@shexp", "ds4_grouped_packed@shexp", "ds4_grouped_q8@shexp"};
const char* const kHyperKernels[] = {
    "ds4_hyper_connection", "ds4_hc_gemv", "ds4_hc_tail", "ds4_hc_tail_norm", "ds4_hc_mix",
    "ds4_hyper_head"};
const char* const kHeadKernels[] = {
    "ds4_dense_f16@lmhead", "ds4_dense_packed@lmhead", "ds4_dense_q8@lmhead",
    "ds4_dense_f16_scalar@lmhead", "ds4_grouped_f16_scalar@lmhead",
    "ds4_grouped_f16@lmhead", "ds4_grouped_packed@lmhead", "ds4_grouped_q8@lmhead"};

template <size_t N>
bool in_set(const char* const (&set)[N], const std::string& n) {
    for (size_t i = 0; i < N; ++i) if (n == set[i]) return true;
    return false;
}

// `ds4_swiglu_clamped` is submitted by BOTH the routed-expert loop and the
// shared expert, from deepseek4_ops.cpp, which carries no site tag.  Both are
// MoE, so the attention/MoE/hyper split is unaffected; only the routed-vs-shared
// sub-split is, and this says so out loud instead of picking one.
const char* bucket_of(const std::string& n) {
    if (in_set(kAttnKernels, n))      return "attention";
    if (in_set(kMoeRoutedKernels, n)) return "moe_routed";
    if (in_set(kMoeSharedKernels, n)) return "moe_shared";
    if (in_set(kHyperKernels, n))     return "hyper";
    if (in_set(kHeadKernels, n))      return "lm_head";
    if (n == "ds4_swiglu_clamped" || n == "ds4_swiglu_clamped_h") return "moe_mixed";
    // Its own bucket, and not filed under any site: `ds4_q8_soa_to_f16` runs at
    // EVERY site (it has no @site suffix) and it is the one kernel whose cost is
    // the entire justification for the requantised T>1 route.  The derived cost
    // is ~23 ms/card/chunk over the real per-card set; if this bucket is not
    // approximately that, the route is not what the code comments say it is.
    if (n == "ds4_q8_soa_to_f16") return "dense_materialise";
    return "other";
}

// ---------------------------------------------------------------------------
// Distribution of a sample of per-step times
// ---------------------------------------------------------------------------
struct Dist {
    double mean = 0, p50 = 0, p10 = 0, p90 = 0, min = 0, max = 0, stdev = 0;
    uint32_t n = 0;
};

Dist distribution(std::vector<double> v) {
    Dist d;
    if (v.empty()) return d;
    std::sort(v.begin(), v.end());
    d.n   = uint32_t(v.size());
    d.min = v.front();
    d.max = v.back();
    // Nearest-rank percentile: no interpolation, so every reported value is a
    // step that actually happened.
    auto pct = [&](double p) {
        size_t i = size_t(p * double(v.size()));
        if (i >= v.size()) i = v.size() - 1;
        return v[i];
    };
    d.p10 = pct(0.10);
    d.p50 = pct(0.50);
    d.p90 = pct(0.90);
    double sum = 0;
    for (double x : v) sum += x;
    d.mean = sum / double(v.size());
    double ss = 0;
    for (double x : v) ss += (x - d.mean) * (x - d.mean);
    d.stdev = v.size() > 1 ? std::sqrt(ss / double(v.size() - 1)) : 0.0;
    return d;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
struct Args {
    std::string gguf;
    std::string mini_path;
    bool        mini        = false;
    bool        mini_keep   = false;
    uint32_t    gpu         = 0;
    std::vector<uint32_t> tp_gpus;
    std::vector<uint32_t> pp{128, 512, 2048, 8192};
    std::vector<uint32_t> ctx{128, 1024, 4096};
    uint32_t    decode      = 32;
    uint32_t    warmup      = 2;
    uint32_t    max_seq     = 512;
    uint32_t    max_ctx     = 0;       // 0 = derive from pp/ctx
    uint32_t    slots       = 0;
    uint32_t    stream_slots = 0;
    uint64_t    cache_bytes = 0;
    uint32_t    pin_layers  = 0;
    bool        prefetch    = false;  // matches Ds4Options default (off since 2026-08-03)
    bool        kprofile    = false;
    bool        kprofile_prefill = false;
    bool        loop        = false;
    std::string profile_out;
    std::string priority_in;
    std::string baseline;
    std::string tag;
    bool        pp_set = false, ctx_set = false, decode_set = false, max_seq_set = false;
};

std::vector<uint32_t> parse_tp_gpus() {
    const char* s = std::getenv("DS4_TP_GPUS");
    return s ? parse_csv_u32(s) : std::vector<uint32_t>{};
}

void usage() {
    std::printf(
        "ie-ds4-bench — DeepSeek-V4-Flash baseline harness\n"
        "  --gguf <path>        model shard 1 (default $DS4_GGUF)\n"
        "  --mini [path]        write and bench the miniature fixture instead\n"
        "  --mini-keep          do not delete the fixture afterwards\n"
        "  --gpu <n>            single-card ordinal (default $DS4_GPU, else 0);\n"
        "                       $DS4_TP_GPUS=0,1 drives both cards instead\n"
        "  --pp a,b,c           prefill lengths        (default 128,512,2048,8192)\n"
        "  --ctx a,b,c          decode context depths  (default 128,1024,4096)\n"
        "  --decode <n>         timed decode steps per depth (default 32)\n"
        "  --warmup <n>         untimed decode steps per depth (default 2)\n"
        "  --max-seq <n>        Ds4Options::max_seq; prefill past it is CHUNKED\n"
        "  --max-ctx <n>        Ds4Options::max_context (default: derived)\n"
        "  --slots <n>          VRAM expert slots/layer (0 = derive)\n"
        "  --stream-slots <n>   evictable slots/layer; the rest are STATIC and are\n"
        "                       filled in residency-priority order ($DS4_STREAM_SLOTS)\n"
        "  --cache-mb <n>       VRAM expert cache budget\n"
        "  --pin-layers <n>     pin only the first n layers ($DS4_PIN_LAYERS)\n"
        "  --no-prefetch        disable idle-time speculative prefetch (now the default)\n"
        "  --prefetch           re-enable it; helps a single cold chunk, costs decode\n"
        "  --kprofile           per-kernel GPU attribution of one warm decode step\n"
        "  --kprofile-prefill   per-kernel GPU attribution of an EXTRA prefill run\n"
        "                       (forces the TP cards to take turns -- read the shares,\n"
        "                       not that pass's wall)\n"
        "  --profile-out <p>    count expert selections and write the ranking to p\n"
        "  --priority-in <p>    load with a measured residency ranking from p\n"
        "  --loop               profile -> reload with the ranking -> re-bench -> delta\n"
        "  --baseline <file>    diff every metric against a previous run's stdout\n"
        "  --tag <str>          free-form label carried into the block\n");
}

// ---------------------------------------------------------------------------
// Per-pass results, kept so --loop can diff pass A against pass B
// ---------------------------------------------------------------------------
struct TgPoint {
    uint32_t depth    = 0;
    double   tok_s    = 0;
    double   hit_rate = 0;
    double   mb_tok   = 0;
};
struct PassResult {
    bool     ok       = false;
    double   ttft_ms  = 0;    // warm
    double   tok_s    = 0;    // aggregate over every timed decode step
    double   hit_rate = 0;    // aggregate over every timed decode step
    double   mb_tok   = 0;
    std::vector<TgPoint> tg;
};

// ---------------------------------------------------------------------------
// The benchmark body — one loaded runtime in, one metric block out
// ---------------------------------------------------------------------------
PassResult run_pass(Driver& drv, const DeepSeek4Config& cfg, const Args& a,
                    const std::string& pfx, Emit& em) {
    PassResult res;
    std::vector<float> logits(cfg.vocab);

    // Deterministic prompt ids: the same sequence for every run and every pass,
    // so a diff between two runs is not a diff between two prompts.
    auto prompt_ids = [&](uint32_t n, uint32_t salt) {
        std::vector<int32_t> v(n);
        for (uint32_t i = 0; i < n; ++i)
            v[i] = int32_t((uint64_t(i) * 1103515245ull + 12345ull + salt * 7919ull) % cfg.vocab);
        return v;
    };
    auto argmax = [&]() {
        int32_t best = 0;
        float   bv   = -1e30f;
        for (uint32_t i = 0; i < cfg.vocab; ++i)
            if (logits[i] > bv) { bv = logits[i]; best = int32_t(i); }
        return best;
    };

    // Prefill `T` tokens from the CURRENT context position, chunked at max_seq
    // because the workspace is sized for max_seq and forward() refuses beyond
    // it.  Returns "" or the engine's own diagnostic.
    auto prefill = [&](const std::vector<int32_t>& ids, uint32_t pos0,
                       StepStat& acc, uint32_t& chunks) -> std::string {
        chunks = 0;
        uint32_t done = 0;
        while (done < ids.size()) {
            const uint32_t n = std::min<uint32_t>(a.max_seq, uint32_t(ids.size()) - done);
            const double t0 = now_s();
            const std::string e = drv.forward(ids.data() + done, n, pos0 + done, logits.data());
            const double ms = (now_s() - t0) * 1e3;
            if (!e.empty()) return e;
            acc.add(collect(drv, ms));
            ++chunks;
            done += n;
        }
        return {};
    };

    // ---------------------------------------------------------------- TTFT --
    // "generate() entry to the first sampled token", excluding load: one prefill
    // of the prompt plus the argmax that turns its logits into a token.  COLD is
    // the first call this process ever makes, so it carries the SYCL JIT of
    // every kernel in the model; WARM is the same work with the code cache hot.
    // They are reported separately because the difference is entirely JIT and
    // reporting one number would misrepresent both.
    const uint32_t ttft_prompt = std::min<uint32_t>(a.pp.empty() ? 128u : a.pp.front(),
                                                    a.max_seq);
    em.u(pfx + "ttft.prompt_tokens", ttft_prompt);
    {
        const std::vector<int32_t> ids = prompt_ids(ttft_prompt, 1);
        for (int warm = 0; warm < 2; ++warm) {
            drv.reset_context();
            StepStat acc;
            uint32_t chunks = 0;
            const double t0 = now_s();
            const std::string e = prefill(ids, 0, acc, chunks);
            if (!e.empty()) {
                em.na(pfx + (warm ? "ttft.warm_ms" : "ttft.cold_ms"), "forward failed: " + e);
                Emit::note("TTFT aborted; no throughput figure will be printed for this pass.");
                return res;
            }
            const double pre_ms = (now_s() - t0) * 1e3;
            const double s0 = now_s();
            const int32_t tok = argmax();
            const double samp_ms = (now_s() - s0) * 1e3;
            const char* k = warm ? "warm" : "cold";
            em.d(pfx + "ttft." + k + "_ms", pre_ms + samp_ms, 3);
            em.d(pfx + "ttft." + k + ".prefill_ms", pre_ms, 3);
            em.d(pfx + "ttft." + k + ".sample_ms", samp_ms, 3);
            em.u(pfx + "ttft." + k + ".first_token_id", uint64_t(uint32_t(tok)));
            if (warm) res.ttft_ms = pre_ms + samp_ms;
        }
        Emit::note("ttft.cold includes SYCL JIT of every kernel; ttft.warm does not. "
                   "Neither includes model load.");
    }

    // ------------------------------------------------------------- PREFILL --
    // Per length, from an EMPTY context.  A single number would be misleading
    // here: sliding_window bounds the local branch, while the CSA (ratio 4) and
    // HCA (ratio 128) compressors only start emitting entries at 4 and 128
    // tokens respectively, and the CSA layers' top-k indexer only has something
    // to select from once the compressed list is long.  Short and long prompts
    // therefore run measurably different code.
    em.u(pfx + "pp.chunk_tokens", a.max_seq);
    for (uint32_t T : a.pp) {
        const std::string kp = pfx + "pp." + std::to_string(T) + ".";
        if (T > a.max_ctx) {
            em.s(kp + "status", "skipped_exceeds_max_context");
            em.na(kp + "tok_s", "prompt " + std::to_string(T) + " > max_context " +
                                 std::to_string(a.max_ctx));
            continue;
        }
        drv.reset_context();
        const std::vector<int32_t> ids = prompt_ids(T, 2);
        StepStat acc;
        uint32_t chunks = 0;
        const double t0 = now_s();
        const std::string e = prefill(ids, 0, acc, chunks);
        const double ms = (now_s() - t0) * 1e3;
        if (!e.empty()) {
            em.s(kp + "status", "error");
            em.na(kp + "tok_s", "forward failed: " + e);
            continue;
        }
        em.s(kp + "status", "ok");
        em.u(kp + "chunks", chunks);
        em.d(kp + "ms", ms, 3);
        em.d(kp + "tok_s", double(T) * 1e3 / ms, 3);
        em.d(kp + "ms_per_tok", ms / double(T), 5);
        em.u(kp + "dma_bytes", acc.bytes);
        em.d(kp + "dma_busy_ms", acc.dma_ms, 3);
        em.d(kp + "dma_busy_frac", acc.dma_ms / ms, 4);
        // ---- THE SAME NUMBERS PER CARD, BECAUSE THE SUMS ABOVE MISLEAD -----
        // `collect()` SUMS every counter over the cards; `ms` is one wall clock
        // around one forward and is NOT summed (it cannot be — the cards run
        // concurrently).  So `dma_busy_frac` is N cards' transfer-queue busy
        // time over ONE wall, which reads as N x the real link utilisation and
        // was in fact misread that way.  A card is the thing that has a PCIe
        // link, so the per-card figures below are the ones that answer "is the
        // link the bottleneck".
        const double ncards = double(drv.n_cards());
        em.u(kp + "cards", drv.n_cards());
        em.u(kp + "dma_bytes_per_card", uint64_t(double(acc.bytes) / ncards));
        em.d(kp + "dma_busy_ms_per_card", acc.dma_ms / ncards, 3);
        em.d(kp + "dma_busy_frac_per_card", acc.dma_ms / ncards / ms, 4);
        em.d(kp + "dma_GB_s_when_busy", acc.dma_ms > 0 ? double(acc.bytes) / 1e6 / acc.dma_ms : 0.0, 3);
        em.d(kp + "dma_MB_per_tok_per_card", double(acc.bytes) / ncards / double(T) / 1e6, 3);
        em.u(kp + "cache_hits", acc.hits);
        em.u(kp + "cache_misses", acc.misses);
        // ---- the wall-clock decomposition of the LAST forward (one chunk) ----
        // wall = prologue + span + reduce + epilogue + gap (Ds4TpForwardTrace);
        // per card: busy = Σ its segment intervals, waits = host time blocked
        // in q.wait() by category.  Under TP only.
        if (drv.tp) {
            const ie::Ds4TpForwardTrace& ft = drv.tp->forward_trace();
            const std::string fp = kp + "ftrace.";
            const double gap = ft.wall_seconds - ft.prologue_seconds - ft.span_seconds -
                               ft.reduce_seconds - ft.epilogue_seconds;
            em.d(fp + "wall_ms", ft.wall_seconds * 1e3, 3);
            em.d(fp + "span_ms", ft.span_seconds * 1e3, 3);
            em.d(fp + "reduce_ms", ft.reduce_seconds * 1e3, 3);
            em.d(fp + "prologue_ms", ft.prologue_seconds * 1e3, 3);
            em.d(fp + "epilogue_ms", ft.epilogue_seconds * 1e3, 3);
            em.d(fp + "gap_ms", gap * 1e3, 3);
            em.d(fp + "span_pre_ms", ft.phase_span[0] * 1e3, 3);
            em.d(fp + "span_mid_ms", ft.phase_span[1] * 1e3, 3);
            em.d(fp + "span_post_ms", ft.phase_span[2] * 1e3, 3);
            em.d(fp + "overlap_ms", ft.overlap_seconds * 1e3, 3);
            em.u(fp + "segments_overlapped", ft.overlapped);
            em.u(fp + "segments", ft.total);
            em.u(fp + "sequential", ft.sequential ? 1u : 0u);
            for (uint32_t c = 0; c < ft.n_cards; ++c) {
                const std::string cp = fp + "card" + std::to_string(c) + ".";
                if (c < ft.card_busy.size()) em.d(cp + "busy_ms", ft.card_busy[c] * 1e3, 3);
                if (c < ft.card_waits.size()) {
                    const auto& w = ft.card_waits[c];
                    em.d(cp + "wait_router_ms", w.router * 1e3, 3);
                    em.d(cp + "wait_moe_ms", w.moe * 1e3, 3);
                    em.d(cp + "wait_post_ms", w.post * 1e3, 3);
                    em.d(cp + "wait_other_ms", w.other * 1e3, 3);
                    em.d(cp + "wait_total_ms", w.total() * 1e3, 3);
                    em.u(cp + "waits", w.count());
                }
            }
        }
        em.d(kp + "cache_hit_rate", acc.hit_rate(), 4);
        em.u(kp + "cache_acquires", acc.acquires);
        // Acquire IDS, per card per layer.  This is the number the expert-major
        // residency loop exists to minimise: with the batch whole and the
        // EXPERTS grouped it equals the distinct experts a layer touched, so a
        // value near n_experts means the prompt really does touch nearly every
        // expert, and one far below it means the working set is concentrated.
        // The token-chunked predecessor had no such bound.
        {
            const uint32_t nl = drv.card0().layers_pinned();
            const double   den = ncards * double(nl ? nl : 1u);
            em.d(kp + "acquire_ids_per_layer_per_card", double(acc.hits + acc.misses) / den, 2);
            em.d(kp + "misses_per_layer_per_card", double(acc.misses) / den, 2);
        }
        em.u(kp + "prefetch_issued", acc.spec_issued);
        em.u(kp + "prefetch_hits", acc.spec_hits);
        em.u(kp + "kv_state_bytes_e2", drv.card0().kv_state_bytes(2));
        em.u(kp + "kv_state_bytes_e4", drv.card0().kv_state_bytes(4));

        // ---- per-kernel attribution of the PREFILL ------------------------
        // `--kprofile` alone attributes a DECODE step, which is a different
        // shape of work: T=1 GEMVs against T-token GEMMs, no chunking, and a
        // routed-expert set of at most top_k per layer.  Nothing in this
        // harness had ever attributed a PREFILL, which is why "where does the
        // prefill wall go" was answerable only as "not DMA".  This repeats the
        // prompt from an empty context with the profiler on.
        //
        // READ THE CAVEAT.  `ie::g_profiler` is a process-global whose entry
        // list is a bare std::vector, so `DeepSeek4TpRuntime::forward` takes the
        // cards in TURNS while it is set (deepseek4.cpp, `ftrace_.sequential`).
        // The per-kernel GPU durations are therefore still real per-card device
        // times and their SHARES are the point; the wall of this extra pass is
        // NOT comparable to the timed one above and is emitted only so the two
        // are visibly different rather than silently conflated.
        if (!a.kprofile_prefill) continue;
        drv.reset_context();
        KernelProfiler pprof;
        g_profiler = &pprof;
        pprof.begin_step();
        StepStat pacc;
        uint32_t pchunks = 0;
        const double pk0 = now_s();
        const std::string pe = prefill(ids, 0, pacc, pchunks);
        const double pwall_ms = (now_s() - pk0) * 1e3;
        g_profiler = nullptr;
        if (!pe.empty()) { em.na(kp + "kprof.gpu_ms", "profiled prefill failed: " + pe); continue; }
        std::vector<KernelProfiler::Stat> pst = pprof.harvest();
        std::sort(pst.begin(), pst.end(),
                  [](const KernelProfiler::Stat& x, const KernelProfiler::Stat& y) {
                      return x.total_ns > y.total_ns;
                  });
        double pgpu_ms = 0;
        for (const KernelProfiler::Stat& s : pst) pgpu_ms += s.total_ms();
        em.d(kp + "kprof.wall_ms_SEQUENTIAL_CARDS", pwall_ms, 3);
        em.d(kp + "kprof.gpu_ms", pgpu_ms, 3);
        em.d(kp + "kprof.dma_busy_ms", pacc.dma_ms, 3);
        em.u(kp + "kprof.kernels", uint64_t(pst.size()));
        {
            std::map<std::string, double>   pb;
            std::map<std::string, uint64_t> pc;
            for (const KernelProfiler::Stat& s : pst) {
                pb[bucket_of(s.name)] += s.total_ms();
                pc[bucket_of(s.name)] += s.calls;
                em.d(kp + "kprof.kernel." + s.name + ".ms", s.total_ms(), 4);
                em.u(kp + "kprof.kernel." + s.name + ".calls", s.calls);
            }
            for (const std::pair<const std::string, double>& b : pb) {
                em.d(kp + "kgroup." + b.first + ".ms", b.second, 3);
                em.d(kp + "kgroup." + b.first + ".frac", pgpu_ms > 0 ? b.second / pgpu_ms : 0.0, 4);
                em.u(kp + "kgroup." + b.first + ".calls", pc[b.first]);
            }
        }
        // What the timed (concurrent) prefill above did NOT spend on the GPU or
        // the transfer queue.  Under the sequential profiled drive `gpu_ms` is
        // the sum over cards, so compare it against the profiled wall, not the
        // timed one — and treat the remainder as host + submission, not as a
        // kernel nobody named.
        em.d(kp + "kprof.host_ms_est",
             std::max(0.0, pwall_ms - std::max(pgpu_ms, pacc.dma_ms)), 3);
    }
    if (a.kprofile_prefill)
        Emit::note("pp.*.kprof.* comes from an EXTRA prefill run with the process-global kernel "
                   "profiler on, which forces the TP cards to take turns. Its wall is therefore "
                   "NOT the concurrent prefill wall reported as pp.*.ms; only the per-kernel and "
                   "per-bucket GPU times and their shares are meant to be read.");

    // -------------------------------------------------------------- DECODE --
    // Per context depth, because the KV state grows with context and the CSA
    // layers run a top-k indexer over a compressed list that grows with it too.
    // The distribution is reported rather than a mean: expert streaming makes
    // per-token cost bursty (a token whose 6 experts are all resident and a
    // token that misses on all 6 are different events), and a mean hides that.
    double all_tok = 0, all_s = 0;
    uint64_t all_hit = 0, all_miss = 0, all_bytes = 0;
    bool tp_kprof_noted = false;
    for (uint32_t depth : a.ctx) {
        const std::string kd = pfx + "tg." + std::to_string(depth) + ".";
        // depth + warmup + timed steps + the one extra profiled step --kprofile
        // adds, + 1 slack.  forward() refuses past max_context rather than
        // overrunning, and being refused at the last step would throw away the
        // whole depth, so the check is made before any of it runs.
        const uint64_t need = uint64_t(depth) + a.warmup + a.decode + (a.kprofile ? 1u : 0u) + 1u;
        if (need > a.max_ctx) {
            em.s(kd + "status", "skipped_exceeds_max_context");
            em.na(kd + "tok_s", "depth+steps " + std::to_string(need) + " > max_context " +
                                 std::to_string(a.max_ctx));
            continue;
        }
        drv.reset_context();
        StepStat pre;
        uint32_t chunks = 0;
        std::string e;
        if (depth) e = prefill(prompt_ids(depth, 3), 0, pre, chunks);
        if (!e.empty()) {
            em.s(kd + "status", "error_prefill");
            em.na(kd + "tok_s", "prefill failed: " + e);
            continue;
        }
        uint32_t pos = depth;
        int32_t  tok = depth ? argmax() : 0;

        for (uint32_t w = 0; w < a.warmup; ++w) {
            e = drv.forward(&tok, 1, pos, logits.data());
            if (!e.empty()) break;
            ++pos;
            tok = argmax();
        }
        if (!e.empty()) {
            em.s(kd + "status", "error_warmup");
            em.na(kd + "tok_s", "warmup step failed: " + e);
            continue;
        }

        std::vector<double> step_ms;
        step_ms.reserve(a.decode);
        StepStat acc;
        const double d0 = now_s();
        for (uint32_t s = 0; s < a.decode; ++s) {
            const double t0 = now_s();
            e = drv.forward(&tok, 1, pos, logits.data());
            const double ms = (now_s() - t0) * 1e3;
            if (!e.empty()) break;
            step_ms.push_back(ms);
            acc.add(collect(drv, ms));
            ++pos;
            tok = argmax();
        }
        const double total_s = now_s() - d0;
        if (!e.empty()) {
            em.s(kd + "status", "error_decode");
            em.na(kd + "tok_s", "decode step failed: " + e);
            continue;
        }
        const Dist di = distribution(step_ms);
        em.s(kd + "status", "ok");
        em.u(kd + "steps", di.n);
        em.d(kd + "tok_s", double(di.n) / total_s, 4);
        em.d(kd + "ms_per_tok_mean", di.mean, 4);
        em.d(kd + "ms_per_tok_p50", di.p50, 4);
        em.d(kd + "ms_per_tok_p10", di.p10, 4);
        em.d(kd + "ms_per_tok_p90", di.p90, 4);
        em.d(kd + "ms_per_tok_min", di.min, 4);
        em.d(kd + "ms_per_tok_max", di.max, 4);
        em.d(kd + "ms_per_tok_stdev", di.stdev, 4);
        em.d(kd + "burstiness_p90_over_p50", di.p50 > 0 ? di.p90 / di.p50 : 0.0, 4);
        em.d(kd + "dma_bytes_per_tok", double(acc.bytes) / double(di.n), 1);
        em.d(kd + "dma_busy_ms_per_tok", acc.dma_ms / double(di.n), 4);
        em.d(kd + "dma_busy_frac", acc.dma_ms / (total_s * 1e3), 4);
        em.u(kd + "cache_hits", acc.hits);
        em.u(kd + "cache_misses", acc.misses);
        em.d(kd + "cache_hit_rate", acc.hit_rate(), 4);
        em.u(kd + "cache_acquires", acc.acquires);
        em.u(kd + "cache_unavailable", acc.unavailable);
        em.u(kd + "prefetch_issued", acc.spec_issued);
        em.u(kd + "prefetch_hits", acc.spec_hits);
        // Context cost, split into the two pieces that have different fixes: the
        // compressed entry arrays, which grow with context, and everything else
        // (sliding ring, window buffers, overlap slices), which does not.
        const uint64_t kv_state = drv.card0().kv_state_bytes(2);
        const uint64_t kv_grow  = drv.card0().kv_growing_bytes(2);
        const uint64_t ctxlen   = drv.card0().context_length();
        em.u(kd + "kv_state_bytes_e2", kv_state);
        em.u(kd + "kv_state_bytes_e4", drv.card0().kv_state_bytes(4));
        em.u(kd + "kv_growing_bytes_e2", kv_grow);
        em.u(kd + "kv_const_bytes_e2", kv_state >= kv_grow ? kv_state - kv_grow : 0);
        em.u(kd + "kv_allocated_bytes", drv.card0().kv_allocated_bytes());
        em.u(kd + "context_length", ctxlen);
        em.d(kd + "kv_growth_bytes_per_ctx_tok_e2",
             ctxlen ? double(kv_grow) / double(ctxlen) : 0.0, 2);

        TgPoint p;
        p.depth = depth;
        p.tok_s = double(di.n) / total_s;
        p.hit_rate = acc.hit_rate();
        p.mb_tok = double(acc.bytes) / double(di.n) / 1e6;
        res.tg.push_back(p);
        all_tok += double(di.n);
        all_s   += total_s;
        all_hit += acc.hits;
        all_miss += acc.misses;
        all_bytes += acc.bytes;

        // ---- per-kernel attribution of ONE extra fully warm T=1 step ----
        // Same mechanism and the same placement as ie_bench.cpp's
        // --kprofile-decode: after prefill, warmup and the timed run, so JIT is
        // discarded and the caches are hot.  The DS4 compute queue is already
        // constructed with enable_profiling, so nothing about the timed steps
        // above changes when this is on or off.
        if (!a.kprofile) continue;
        if (drv.n_cards() > 1 && !tp_kprof_noted) {
            tp_kprof_noted = true;
            Emit::note("kprofile under TP: g_profiler is process-global and the cards run "
                       "sequentially inside one forward, so attention / moe / hyper buckets are "
                       "the SUM over cards, while lm_head is card 0 only (the head is mirrored "
                       "and only card 0 computes it). Do not read a per-card share off these.");
        }
        KernelProfiler kprof;
        g_profiler = &kprof;
        kprof.begin_step();
        const double k0 = now_s();
        e = drv.forward(&tok, 1, pos, logits.data());
        const double kwall_ms = (now_s() - k0) * 1e3;
        g_profiler = nullptr;
        if (!e.empty()) {
            em.na(kd + "kprof.gpu_ms", "profiled step failed: " + e);
            continue;
        }
        ++pos;
        tok = argmax();
        const StepStat kst = collect(drv, kwall_ms);
        std::vector<KernelProfiler::Stat> st = kprof.harvest();
        std::sort(st.begin(), st.end(),
                  [](const KernelProfiler::Stat& x, const KernelProfiler::Stat& y) {
                      return x.total_ns > y.total_ns;
                  });
        double gpu_ms = 0;
        for (const KernelProfiler::Stat& s : st) gpu_ms += s.total_ms();
        em.d(kd + "kprof.wall_ms", kwall_ms, 4);
        em.d(kd + "kprof.gpu_ms", gpu_ms, 4);
        em.d(kd + "kprof.dma_busy_ms", kst.dma_ms, 4);
        em.u(kd + "kprof.kernels", uint64_t(st.size()));
        std::map<std::string, double> bucket;
        std::map<std::string, uint64_t> bcalls;
        for (const KernelProfiler::Stat& s : st) {
            bucket[bucket_of(s.name)] += s.total_ms();
            bcalls[bucket_of(s.name)] += s.calls;
            em.d(kd + "kprof.kernel." + s.name + ".ms", s.total_ms(), 4);
            em.u(kd + "kprof.kernel." + s.name + ".calls", s.calls);
        }
        for (const std::pair<const std::string, double>& b : bucket) {
            em.d(kd + "kgroup." + b.first + ".ms", b.second, 4);
            em.d(kd + "kgroup." + b.first + ".frac", gpu_ms > 0 ? b.second / gpu_ms : 0.0, 4);
            em.u(kd + "kgroup." + b.first + ".calls", bcalls[b.first]);
        }
        // The one derived figure in this harness.  DMA and compute run on two
        // queues; `wall` is what the caller waits for.  If they were perfectly
        // serialised, wall == gpu + dma; if perfectly overlapped, wall ==
        // max(gpu, dma).  The fraction of the SMALLER of the two that the wall
        // says must have been hidden is therefore
        //     (gpu + dma - wall) / min(gpu, dma)
        // clamped to [0, 1].  It is an ESTIMATE — wall also contains host time
        // (routing ids copied to host, the chunking search, submission) that
        // belongs to neither queue, which biases it DOWN.  Read it as a lower
        // bound on overlap, and read `dma_busy_frac` next to it, which is
        // directly measured.
        const double lo = std::min(gpu_ms, kst.dma_ms);
        if (lo <= 0.0) {
            em.na(kd + "overlap_est_frac",
                  kst.dma_ms <= 0.0 ? "no expert DMA in the profiled step (everything resident)"
                                    : "no GPU kernel time recorded in the profiled step");
        } else {
            const double f = (gpu_ms + kst.dma_ms - kwall_ms) / lo;
            em.d(kd + "overlap_est_frac", std::max(0.0, std::min(1.0, f)), 4);
        }
        em.d(kd + "kprof.host_ms_est", std::max(0.0, kwall_ms - std::max(gpu_ms, kst.dma_ms)), 4);
    }

    if (all_s > 0 && all_tok > 0) {
        res.ok       = true;
        res.tok_s    = all_tok / all_s;
        res.hit_rate = (all_hit + all_miss) ? double(all_hit) / double(all_hit + all_miss) : 0.0;
        res.mb_tok   = double(all_bytes) / all_tok / 1e6;
        em.d(pfx + "tg.all.tok_s", res.tok_s, 4);
        em.d(pfx + "tg.all.cache_hit_rate", res.hit_rate, 4);
        em.d(pfx + "tg.all.dma_mb_per_tok", res.mb_tok, 4);
        em.u(pfx + "tg.all.steps", uint64_t(all_tok));
    } else {
        em.na(pfx + "tg.all.tok_s", "no decode depth completed");
    }

    // Cross-card cost, cumulative over every forward this pass made.  One
    // reduction per layer per forward is the lockstep invariant; a count that
    // does not match that is the first thing to look at if TP throughput is off.
    if (drv.tp) {
        em.u(pfx + "tp.reductions", drv.tp->reductions());
        em.d(pfx + "tp.reduce_ms_total", drv.tp->reduce_seconds() * 1e3, 3);
        em.u(pfx + "tp.stage_bytes", drv.tp->stage_bytes());
        Emit::note("tp.* are cumulative over the whole pass, not per token. The cards DO run "
                   "concurrently now (DeepSeek4TpRuntime posts each layer segment to a per-card "
                   "worker and joins at the reduction; deepseek4_residency_test 15h asserts the "
                   "intervals intersect) EXCEPT while the kernel profiler is on, which forces "
                   "them to take turns.");
    }
    return res;
}

// ---------------------------------------------------------------------------
// Load + report the residency configuration the numbers were measured under
// ---------------------------------------------------------------------------
std::string load_runtime(const DeepSeek4Model& m, const Args& a, const Ds4Options& base,
                         DeepSeek4Runtime& rt, DeepSeek4TpRuntime& tprt, Driver& drv,
                         const std::string& pfx, Emit& em) {
    const double t0 = now_s();
    std::string err;
    if (a.tp_gpus.size() > 1) {
        Ds4TpOptions to;
        to.base                  = base;
        to.base.device_ordinal   = 0;
        to.base.n_cards          = 1;
        to.base.card             = 0;
        to.n_cards               = uint32_t(a.tp_gpus.size());
        to.device_ordinals       = a.tp_gpus;
        to.same_device_rehearsal = false;
        err = tprt.load(m, to);
        if (err.empty()) drv.tp = &tprt;
    } else {
        err = rt.load(m, base);
        if (err.empty()) drv.single = &rt;
    }
    const double load_s = now_s() - t0;
    em.d(pfx + "load.seconds", load_s, 2);
    if (!err.empty()) { em.s(pfx + "load.status", "error"); return err; }
    em.s(pfx + "load.status", "ok");

    uint64_t vram = 0, host = 0;
    for (uint32_t c = 0; c < drv.n_cards(); ++c) {
        DeepSeek4Runtime& rc = drv.card(c);
        vram += rc.resident_bytes() + rc.cache().device_bytes();
        host += rc.arena().total_bytes();
        const std::string kc = pfx + "card." + std::to_string(c) + ".";
        em.s(kc + "device", rc.queue().get_device().get_info<sycl::info::device::name>());
        em.u(kc + "resident_bytes", rc.resident_bytes());
        em.u(kc + "expert_cache_bytes", rc.cache().device_bytes());
        em.u(kc + "pinned_host_bytes", rc.arena().total_bytes());
        em.u(kc + "slots_per_layer", rc.cache().slots_per_layer());
        em.u(kc + "static_slots", rc.cache().static_slots());
        em.u(kc + "stream_slots", rc.cache().stream_slots());
    }
    const Ds4ResidencyPlan& p = drv.card0().residency();
    em.u(pfx + "residency.vram_bytes_total", vram);
    em.u(pfx + "residency.pinned_host_bytes_total", host);
    em.u(pfx + "residency.resident_saved_bytes", drv.card0().resident_saved_bytes());
    em.u(pfx + "residency.pinned_experts_per_layer", p.pinned_experts);
    em.u(pfx + "residency.layers_pinned", drv.card0().layers_pinned());
    const char* origin = "index";
    switch (drv.card0().priority_origin()) {
        case DeepSeek4Runtime::PriorityOrigin::kIndex:       origin = "index"; break;
        case DeepSeek4Runtime::PriorityOrigin::kOptionField: origin = "option"; break;
        case DeepSeek4Runtime::PriorityOrigin::kFile:        origin = "file"; break;
    }
    em.s(pfx + "residency.priority_origin", origin);
    em.s(pfx + "residency.priority_source",
         drv.card0().priority_source().empty() ? "-" : drv.card0().priority_source());
    return {};
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (const char* g = std::getenv("DS4_GGUF")) a.gguf = g;
    if (const char* g = std::getenv("DS4_GPU")) a.gpu = uint32_t(std::atoi(g));
    if (const char* g = std::getenv("DS4_PIN_LAYERS")) a.pin_layers = uint32_t(std::atoi(g));
    if (const char* g = std::getenv("DS4_DECODE_STEPS")) {
        a.decode = uint32_t(std::atoi(g)); a.decode_set = true;
    }
    if (const char* g = std::getenv("DS4_EXPERT_PRIORITY_FILE")) a.priority_in = g;
    a.tp_gpus = parse_tp_gpus();

    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? argv[++i] : std::string(); };
        if      (s == "--gguf")        a.gguf = next();
        else if (s == "--mini")      { a.mini = true;
                                       if (i + 1 < argc && argv[i + 1][0] != '-') a.mini_path = next(); }
        else if (s == "--mini-keep")   a.mini_keep = true;
        else if (s == "--gpu")         a.gpu = uint32_t(std::atoi(next().c_str()));
        else if (s == "--pp")        { a.pp = parse_csv_u32(next()); a.pp_set = true; }
        else if (s == "--ctx")       { a.ctx = parse_csv_u32(next()); a.ctx_set = true; }
        else if (s == "--decode")    { a.decode = uint32_t(std::atoi(next().c_str())); a.decode_set = true; }
        else if (s == "--warmup")      a.warmup = uint32_t(std::atoi(next().c_str()));
        else if (s == "--max-seq")   { a.max_seq = uint32_t(std::atoi(next().c_str())); a.max_seq_set = true; }
        else if (s == "--max-ctx")     a.max_ctx = uint32_t(std::atoi(next().c_str()));
        else if (s == "--slots")       a.slots = uint32_t(std::atoi(next().c_str()));
        else if (s == "--stream-slots") a.stream_slots = uint32_t(std::atoi(next().c_str()));
        else if (s == "--cache-mb")    a.cache_bytes = uint64_t(std::atoll(next().c_str())) << 20;
        else if (s == "--pin-layers")  a.pin_layers = uint32_t(std::atoi(next().c_str()));
        else if (s == "--no-prefetch") a.prefetch = false;
        else if (s == "--prefetch")    a.prefetch = true;
        else if (s == "--kprofile")    a.kprofile = true;
        else if (s == "--kprofile-prefill") a.kprofile_prefill = true;
        else if (s == "--loop")        a.loop = true;
        else if (s == "--profile-out") a.profile_out = next();
        else if (s == "--priority-in") a.priority_in = next();
        else if (s == "--baseline")    a.baseline = next();
        else if (s == "--tag")         a.tag = next();
        else if (s == "-h" || s == "--help") { usage(); return 0; }
        else { std::fprintf(stderr, "unknown argument: %s\n", s.c_str()); usage(); return 2; }
    }

    // The miniature fixture is 6 layers of hidden 256 with 8 experts; the real
    // model's default sweep points do not fit it, so they shrink UNLESS the
    // caller asked for specific ones.
    std::vector<std::vector<uint8_t>> mini_keepalive;
    if (a.mini) {
        if (a.mini_path.empty()) {
            const char* td = std::getenv("TMPDIR");
            a.mini_path = std::string(td ? td : "/tmp") + "/ds4_bench_mini.gguf";
        }
        ds4mini::Cfg mc;
        const std::string we = ds4mini::write_gguf(mc, a.mini_path, mini_keepalive);
        mini_keepalive.clear();
        if (!we.empty()) { std::fprintf(stderr, "mini fixture: %s\n", we.c_str()); return 1; }
        a.gguf = a.mini_path;
        if (!a.pp_set)      a.pp  = {8, 32, 136};
        if (!a.ctx_set)     a.ctx = {0, 32, 136};
        if (!a.decode_set)  a.decode = 8;
        if (!a.max_seq_set) a.max_seq = 192;
        // < n_experts so the cache really misses, and split so the STATIC
        // partition is non-empty: with the default min_stream_slots (12) a
        // 4-slot cache plans zero static slots, under which the residency
        // priority order has nothing to order and the profile -> priority loop
        // would be a guaranteed no-op rather than a measurement.
        if (!a.slots)        a.slots = 6;
        if (!a.stream_slots) a.stream_slots = 2;
        if (!a.cache_bytes) a.cache_bytes = 64ull << 20;
        // The fixture's own §6 sizing.  Derived-from-the-sweep would be tight
        // enough that a compressed-entry list one row longer than expected turns
        // into a refusal, and a refusal here would look like a harness bug.
        if (!a.max_ctx)     a.max_ctx = 512;
    }
    if (!a.decode) a.decode = 1;   // a zero-step decode has no distribution to report
    if (a.gguf.empty()) {
        std::fprintf(stderr, "no model: pass --gguf <path>, set $DS4_GGUF, or use --mini\n");
        return 2;
    }
    // Same gate the Phase 5 test uses, for the same reason: loading the real
    // model is 120 GB of disk, tens of minutes and both GPUs, and it happens
    // only when the owner asks for it in as many words.  The miniature fixture
    // is exempt — it is a ~200 MB synthetic file this tool writes itself.
    if (!a.mini && !std::getenv("DS4_RUN_REAL")) {
        std::fprintf(stderr,
            "REFUSED: %s is a real model load (VRAM on every named card + a pinned host\n"
            "arena + a long file walk).  Set DS4_RUN_REAL=1 to authorise it, or use --mini\n"
            "to exercise this harness on the miniature fixture instead.\n", a.gguf.c_str());
        return 2;
    }
    if (a.pp.empty()) a.pp = {128};
    if (a.ctx.empty()) a.ctx = {0};
    if (!a.max_ctx) {
        uint32_t need = 0;
        for (uint32_t t : a.pp) need = std::max(need, t);
        for (uint32_t d : a.ctx) need = std::max(need, d + a.warmup + a.decode + 2);
        a.max_ctx = std::max(need + 16u, a.max_seq + 16u);
    }

    Emit em;
    em.u("schema", 1);
    em.s("run.tool", "ie-ds4-bench");
    em.s("run.build", std::string(__DATE__) + " " + __TIME__);
    em.s("run.tag", a.tag.empty() ? "-" : a.tag);
    em.s("run.model_path", a.gguf);
    em.b("run.mini_fixture", a.mini);
    em.u("run.max_seq", a.max_seq);
    em.u("run.max_context", a.max_ctx);
    em.u("run.decode_steps", a.decode);
    em.u("run.decode_warmup", a.warmup);
    em.b("run.prefetch", a.prefetch);
    em.b("run.kprofile", a.kprofile);
    em.b("run.kprofile_prefill", a.kprofile_prefill);
    em.u("run.pin_layers", a.pin_layers);
    em.u("run.slots_per_layer_req", a.slots);
    em.u("run.min_stream_slots_req", a.stream_slots);
    em.u("run.expert_cache_bytes_req", a.cache_bytes);
    if (a.mini)
        Emit::note("MINIATURE FIXTURE: 6 layers, hidden 256, 8 experts, pseudo-random weights. "
                   "These numbers prove the harness computes its metrics. They say NOTHING about "
                   "the real DeepSeek-V4-Flash-0731 throughput and must never be quoted as a "
                   "baseline for it.");

    GgufReader g;
    if (const std::string e = g.open(a.gguf); !e.empty()) {
        em.s("run.status", "gguf_open_failed");
        Emit::note("gguf open: " + e);
        return 1;
    }
    DeepSeek4Config cfg;
    if (const std::string e = read_deepseek4_config(g, cfg); !e.empty()) {
        em.s("run.status", "config_failed");
        Emit::note("config: " + e);
        return 1;
    }
    DeepSeek4Model model;
    if (const std::string e = model.load(g, cfg); !e.empty()) {
        em.s("run.status", "bind_failed");
        Emit::note("bind: " + e);
        return 1;
    }
    em.u("model.tensors_bound", model.n_bound());
    em.u("model.n_layers", cfg.n_layers);
    em.u("model.hidden", cfg.hidden);
    em.u("model.vocab", cfg.vocab);
    em.u("model.n_experts", cfg.n_experts);
    em.u("model.n_experts_used", cfg.n_experts_used);
    em.u("model.expert_ffn", cfg.expert_ffn);
    em.u("model.sliding_window", cfg.sliding_window);
    em.u("model.hash_layer_count", cfg.hash_layer_count);
    em.u("model.n_cards", a.tp_gpus.size() > 1 ? uint32_t(a.tp_gpus.size()) : 1u);
    Emit::note("pp.<T> is prefill of T tokens from an EMPTY context, chunked at run.max_seq "
               "(pp.chunk_tokens). It is reported PER LENGTH on purpose: the local branch is "
               "bounded by sliding_window=" + std::to_string(cfg.sliding_window) +
               ", the CSA (ratio 4) compressor emits its first entry at 4 tokens and the HCA "
               "(ratio 128) one at 128, and the CSA layers' top-k indexer only has a long "
               "compressed list to select from at long prompts. One prefill number would "
               "average three different regimes.");
    Emit::note("tg.<D> is decode from a context of D tokens. kv_state_bytes_e2 / _e4 are the "
               "live per-card cache state at that depth measured by Ds4LayerCache itself, "
               "quoted at 2 B/element (the reference/fp16 figure) and at 4 (what this fp32 "
               "engine allocates). kv_growing_bytes_e2 is the part that grows with context "
               "(the compressed entry arrays); kv_const_bytes_e2 is the remainder. None of "
               "these is a per-token constant multiplied out — they are read off the caches.");

    Ds4Options base;
    base.device_ordinal      = a.gpu;
    base.max_seq             = a.max_seq;
    base.max_context         = a.max_ctx;
    base.slots_per_layer     = a.slots;
    base.min_stream_slots    = a.stream_slots;
    base.expert_cache_bytes  = a.cache_bytes;
    base.pin_layers          = a.pin_layers;
    base.prefetch            = a.prefetch;
    base.expert_priority_file = a.priority_in;
    // Counting is only turned on when a ranking is actually wanted out of it: it
    // is a measurement tool, not part of inference, and leaving it on would put
    // a host-side counter update in the routing path of every benchmarked token.
    base.profile_experts     = !a.profile_out.empty() || a.loop;

    // ---------------------------------------------------------------- pass A --
    PassResult A;
    std::string rank_path = a.profile_out;
    {
        DeepSeek4Runtime   rt;
        DeepSeek4TpRuntime tprt;
        Driver drv;
        const std::string pfx = a.loop ? "A." : "";
        if (const std::string e = load_runtime(model, a, base, rt, tprt, drv, pfx, em); !e.empty()) {
            em.s("run.status", "load_failed");
            Emit::note("load: " + e);
            return 1;
        }
        A = run_pass(drv, cfg, a, pfx, em);

        if (base.profile_experts) {
            const Ds4ExpertProfile& prof = drv.card0().expert_profile();
            em.b(pfx + "profile.active", prof.active());
            em.u(pfx + "profile.selections", prof.selections());
            em.u(pfx + "profile.rejected", prof.rejected());
            if (drv.n_cards() > 1)
                Emit::note("expert profile is card 0's. Every card runs the identical router on "
                           "mirrored weights and DeepSeek4TpRuntime refuses the forward pass if "
                           "their selected ids or weights differ, so the cards' counts are the "
                           "same measurement, not independent ones.");
            if (prof.active()) {
                // Concentration of the measured utilisation, which is what
                // decides whether a measured order can beat index order at all.
                const std::vector<uint32_t> order = prof.priority();
                uint64_t top = 0, tot = 0;
                const uint32_t nstat = drv.card0().residency().static_slots;
                for (uint32_t i = 0; i < order.size(); ++i) {
                    const uint64_t c = prof.total(order[i]);
                    tot += c;
                    if (i < nstat) top += c;
                }
                em.u(pfx + "profile.static_slots", nstat);
                em.d(pfx + "profile.top_static_share", tot ? double(top) / double(tot) : 0.0, 5);
                em.d(pfx + "profile.uniform_share",
                     order.empty() ? 0.0 : double(nstat) / double(order.size()), 5);
            }
            if (rank_path.empty() && a.loop) {
                const char* td = std::getenv("TMPDIR");
                rank_path = std::string(td ? td : "/tmp") + "/ds4_bench_priority.txt";
            }
            if (!rank_path.empty()) {
                const std::string we = prof.write(rank_path);
                em.s(pfx + "profile.written", we.empty() ? rank_path : "error");
                if (!we.empty()) Emit::note("profile write: " + we);
            }
        }
    }  // pass A's runtime releases here — its VRAM and pinned host pages must be
       // gone before pass B allocates its own, or the second load is being asked
       // for memory the first one still holds.

    // ---------------------------------------------------------------- pass B --
    // The other half of the loop: reload with the MEASURED residency order and
    // run the identical benchmark, so the delta below is attributable to the
    // ordering and to nothing else.
    if (a.loop) {
        if (rank_path.empty()) {
            em.na("B.load.status", "pass A produced no ranking file");
        } else {
            Ds4Options bopt = base;
            // Counting stays ON, even though pass B's counts are thrown away.
            // Pass A had to count, so switching it off here would put a per-token
            // host-side counter update in A's routing path and not in B's, and
            // the tok/s delta below would then contain the profiler's own cost.
            // The ONLY difference between the two passes must be the ordering.
            bopt.profile_experts      = true;
            bopt.expert_priority_file = rank_path;
            DeepSeek4Runtime   rt;
            DeepSeek4TpRuntime tprt;
            Driver drv;
            if (const std::string e = load_runtime(model, a, bopt, rt, tprt, drv, "B.", em);
                !e.empty()) {
                em.s("B.load.status", "error");
                Emit::note("pass B load: " + e);
            } else {
                const PassResult B = run_pass(drv, cfg, a, "B.", em);
                std::printf("\n");
                auto delta = [&](const char* name, double x, double y) {
                    std::printf("[ds4-delta] loop.%s base=%.6f now=%.6f abs=%+.6f pct=%+.3f\n",
                                name, x, y, y - x,
                                x != 0.0 ? 100.0 * (y - x) / x : 0.0);
                };
                if (A.ok && B.ok) {
                    delta("tg.all.cache_hit_rate", A.hit_rate, B.hit_rate);
                    delta("tg.all.tok_s", A.tok_s, B.tok_s);
                    delta("tg.all.dma_mb_per_tok", A.mb_tok, B.mb_tok);
                    delta("ttft.warm_ms", A.ttft_ms, B.ttft_ms);
                    for (const TgPoint& pa : A.tg)
                        for (const TgPoint& pb : B.tg)
                            if (pa.depth == pb.depth) {
                                const std::string k1 = "tg." + std::to_string(pa.depth) + ".tok_s";
                                const std::string k2 = "tg." + std::to_string(pa.depth) +
                                                       ".cache_hit_rate";
                                delta(k1.c_str(), pa.tok_s, pb.tok_s);
                                delta(k2.c_str(), pa.hit_rate, pb.hit_rate);
                            }
                    Emit::note("loop delta = (measured expert order) minus (index order). "
                               "Static analysis of the shipped tensors put the available gain at "
                               "0.03 percentage points of layer-0-2 hit rate, so a small or zero "
                               "delta is the EXPECTED result, not a broken loop.");
                } else {
                    Emit::note("loop delta not computed: one of the two passes did not complete.");
                }
            }
        }
    }

    // ------------------------------------------------------------- baseline --
    if (!a.baseline.empty()) {
        std::string err;
        const std::map<std::string, std::string> b = read_baseline(a.baseline, err);
        std::printf("\n");
        if (!err.empty()) {
            Emit::note("baseline: " + err);
        } else {
            uint32_t compared = 0, changed = 0, missing = 0;
            for (const std::pair<const std::string, std::string>& kv : em.all()) {
                const std::map<std::string, std::string>::const_iterator it = b.find(kv.first);
                if (it == b.end()) { ++missing; continue; }
                double x = 0, y = 0;
                if (!numeric(it->second, x) || !numeric(kv.second, y)) {
                    if (it->second != kv.second)
                        std::printf("[ds4-delta] %s base=%s now=%s\n", kv.first.c_str(),
                                    it->second.c_str(), kv.second.c_str());
                    continue;
                }
                ++compared;
                if (x == y) continue;
                ++changed;
                std::printf("[ds4-delta] %s base=%.6f now=%.6f abs=%+.6f pct=%+.3f\n",
                            kv.first.c_str(), x, y, y - x, x != 0.0 ? 100.0 * (y - x) / x : 0.0);
            }
            std::printf("[ds4-delta] summary compared=%u changed=%u keys_absent_from_baseline=%u\n",
                        compared, changed, missing);
        }
    }

    // A pass that did not complete is a failed run, not a quiet one: a harness
    // that exits 0 after printing `unavailable` would let a broken configuration
    // pass a script unnoticed.
    em.s("run.status", A.ok ? "ok" : "incomplete");
    if (a.mini && !a.mini_keep) std::remove(a.mini_path.c_str());
    return A.ok ? 0 : 1;
}
