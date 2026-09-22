// src/model/deepseek41_forward.cpp — V4.1 text forward, Transformer.forward unrolled.
// See include/ie/deepseek41_forward.hpp. Every call pattern here was verified in isolation in
// tools/ds41_{block,attn2,engram,moe}_test.cpp; this file composes them with the shared-cache
// plumbing between source and consumer layers, runs each card's layer range in turn, and
// (Phase 9) carries the per-layer state a single-token step needs across calls.
#include "ie/deepseek41_forward.hpp"
#include "ie/deepseek41_candidate.hpp"
#include "ie/deepseek41_attn_xmx.hpp"

#include "ie/deepseek4.hpp"
#include "ie/deepseek4_attn.hpp"
#include "ie/deepseek4_experts.hpp"
#include "ie/deepseek4_ops.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/deepseek41_upload.hpp"
#include "ie/expert_stream.hpp"
#include "ie/ops.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <sys/mman.h>
#include <thread>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>

namespace ie {

namespace {
inline float bf16f(uint16_t h) { const uint32_t b = uint32_t(h) << 16; float f; std::memcpy(&f, &b, 4); return f; }
}  // namespace

std::string Ds41Forward::init(sycl::queue& q, const DeepSeek41Model& m, const Ds41EngramTables& tables) {
    m_ = &m; tb_ = &tables;
    if (tables.layer_ids.size() != m.config().engram_layer_ids.size())
        return "engram tables do not match the config's engram_layer_ids";
    profile_.assign(m.config().n_layers, std::vector<uint64_t>(m.config().n_routed_experts, 0));
    for (auto& cp : cards_) free_state(*cp);
    cards_.clear();
    cards_.push_back(std::make_unique<Card>());
    cards_[0]->q = &q; cards_[0]->L0 = 0; cards_[0]->L1 = m.config().n_layers;
    resident_ = false;
    reset_state();
    return {};
}

namespace {
uint64_t ds41_mem_available_bytes() {
    std::FILE* f = std::fopen("/proc/meminfo", "r"); if (!f) return 0;
    char line[256]; unsigned long long v = 0; uint64_t out = 0;
    while (std::fgets(line, sizeof line, f)) if (std::sscanf(line, "MemAvailable: %llu kB", &v) == 1) { out = uint64_t(v) * 1024ull; break; }
    std::fclose(f); return out;
}
// expert parallel: the per-layer helper threads are joined and the owner's import hook cleared on
// every exit from the layer, a SYCL exception out of moe() included (gate 14 finding 7)
struct EpHelpersDone {
    std::vector<std::thread>* h; Ds41ExpertTier* t; std::function<void()> wait_posted;   // Phase 19: the persistent helpers' jobs too
    ~EpHelpersDone() { for (auto& th : *h) if (th.joinable()) th.join(); if (wait_posted) wait_posted(); t->set_ep_import({}); }
};
// Phase 19 (docs/deepseek41/50): IE_DS41_EP_THREAD=0 restores a std::thread per layer per remote card (the kill switch)
static bool ds41_ep_thread_on() { static const bool on = [] { const char* v = std::getenv("IE_DS41_EP_THREAD"); return !(v && *v && std::string(v) == "0"); }(); return on; }
}  // namespace

std::string Ds41Forward::init_resident(const std::vector<sycl::queue*>& qs, const DeepSeek41Model& m, const Ds41EngramTables& tables,
                                       const std::vector<std::vector<uint32_t>>& ranking_in, const ResidentOptions& opt) {
    if (qs.empty()) return "init_resident: no queues";
    if (auto e = init(*qs[0], m, tables); !e.empty()) return e;
    const auto& c = m.config();
    if (qs.size() > c.n_layers) return "init_resident: more cards than layers";
    cap_pos_ = opt.max_tokens;
    // Phase 24 (docs/deepseek41/64): the single-call bound for every per-call scratch buffer; the KV keeps max_tokens.
    fwd_cap_ = opt.max_forward_tokens ? opt.max_forward_tokens : opt.max_tokens;
    if (const char* v = std::getenv("IE_DS41_MAX_FWD")) if (const uint32_t n = uint32_t(std::atoi(v))) fwd_cap_ = n;
    if (fwd_cap_ > cap_pos_) fwd_cap_ = cap_pos_;
    if (fwd_cap_ < cap_pos_)
        std::fprintf(stderr, "[ds41 forward] one forward carries at most %u tokens (the context holds %u): the expert batch workspace and the EP staging are sized for %u, the KV for %u\n",
                     fwd_cap_, cap_pos_, fwd_cap_, cap_pos_);
    cards_.clear();
    for (size_t i = 0; i < qs.size(); ++i) {
        cards_.push_back(std::make_unique<Card>());
        cards_[i]->q = qs[i];
        cards_[i]->L0 = uint32_t(c.n_layers * i / qs.size());
        cards_[i]->L1 = uint32_t(c.n_layers * (i + 1) / qs.size());
    }

    std::vector<std::vector<uint32_t>> ranking = ranking_in;
    if (ranking.empty()) {
        ranking.assign(c.n_layers, std::vector<uint32_t>(c.n_routed_experts));
        for (auto& r : ranking) std::iota(r.begin(), r.end(), 0u);
    }
    Ds4SlotLayout lay;
    if (auto e = ds41_slot_layout(c.dim, c.moe_inter_dim, lay); !e.empty()) return e;
    // expert parallel (docs/deepseek41/38): the option, else the env; needs two cards
    ep_ = opt.expert_parallel;
    if (!ep_) if (const char* v = std::getenv("IE_DS41_EP")) { const std::string s(v); ep_ = s == "control" ? 2u : (!s.empty() && s != "0") ? 1u : 0u; }
    if (ep_ && qs.size() < 2) ep_ = 0;                                          // one card: the ordinary path
    // expert parallel + the CPU miss split (docs/deepseek41/42 step 3): the two tiers' workers each
    // take half of the E-cores (8-13 / 14-19 on this box: the 12 E-cores split in two), unless
    // IE_DS41_CPU_CORES names the set explicitly, in which case both tiers share it as given
    const bool cpu_split = [] { const char* v = std::getenv("IE_DS41_CPU_MISS"); return !(v && *v && std::string(v) == "0"); }();   // ON since gate 20 (docs/52); =0 the kill switch
    const bool split_cores = ep_ == 1 && cpu_split && !std::getenv("IE_DS41_CPU_CORES");
    const uint32_t n_cards = uint32_t(qs.size());
    uint64_t pin_cap = opt.host_pin_cap;
    if (!pin_cap) if (const char* v = std::getenv("IE_DS41_PIN_CAP_GIB")) pin_cap = uint64_t(std::atof(v) * 1073741824.0);   // the A/B knob against the live rule (docs/40)
    if (!pin_cap) {                                       // lever C1 (docs/deepseek41/40): the live rule, its reasoning printed
        std::string why; bool ok = false;
        pin_cap = ds4_host_pin_cap_live(why, &ok);
        if (!ok || !pin_cap) return "init_resident: the live pinned-host cap could not be derived (" + why + "); pass host_pin_cap";
        // Phase 53 (docs/deepseek41/92, founder-approved 2026-09-17): 10 GiB more than the shared rule's total bound -- +15
        // experts per layer out of the NVMe tier, +5-7 % chat decode, measured at a 30 GiB MemAvailable floor.
        // IE_DS41_PIN_EXTRA_GIB overrides (0 restores the shared rule).
        uint64_t extra = 10ull << 30;
        if (const char* v = std::getenv("IE_DS41_PIN_EXTRA_GIB")) extra = uint64_t(std::max(0.0, std::atof(v)) * 1073741824.0);
        if (extra) { pin_cap += extra; why += "; + " + std::to_string(extra >> 30) + " GiB V4.1 extra"; }
        // the rule reserves against the box at LOAD time; this term keeps at least kFreeAfterPin available for the run's
        // whole life (a build or a desktop app started later; the watchdog and the 2026-08-27 freeze live at the low end
        // of what a pin leaves). 40 GiB before Phase 53; 30 GiB with the extra, the floor it was measured at -- a box with
        // less free memory at load gets a smaller pin, not a squeeze.
        constexpr uint64_t kFreeAfterPin = 30ull << 30;
        const uint64_t avail = ds41_mem_available_bytes();
        const uint64_t cap_free = avail > kFreeAfterPin ? avail - kFreeAfterPin : 0;
        if (cap_free < pin_cap) { pin_cap = cap_free; why += "; then min with MemAvailable - 30 GiB kept free after the pin"; }
        if (!pin_cap) return "init_resident: less than 30 GiB would stay available after any pin (" + why + ")";
        std::fprintf(stderr, "[ds41 forward] pinned host cap %.1f GiB from the live rule: %s\n", double(pin_cap) / 1073741824.0, why.c_str());
    }

    for (size_t ci = 0; ci < cards_.size(); ++ci) {
        Card& card = *cards_[ci]; sycl::queue& q = *card.q;
        const uint32_t nL = card.L1 - card.L0;
        // the tier's layer range and budgets: this card's layers, or (expert parallel) every layer
        // with this card's parity share of the experts -- the same VRAM and pinned RAM in total
        const uint32_t tL0 = ep_ == 1 ? 0u : card.L0, tnL = ep_ == 1 ? c.n_layers : nL;
        // expert parallel keeps the full stream-slot count per card per layer: halving it (4 + 4 = 8
        // as before) cost 4 static slots less but measured pp2048 300 vs 308-319 tok/s and decode
        // 151 vs 149 ms/token (docs/deepseek41/39); IE_DS41_EP_STREAM=N overrides for experiments
        uint32_t stream = opt.stream_slots;
        if (const char* v = std::getenv("IE_DS41_STREAM_SLOTS"); v && *v && std::atoi(v) >= 2) stream = uint32_t(std::atoi(v));   // Phase 51: the A/B knob
        if (ep_ == 1) if (const char* v = std::getenv("IE_DS41_EP_STREAM")) stream = std::max(1u, uint32_t(std::atoi(v)));
        // dense path first, so the expert budget is measured against what is actually left
        for (uint32_t L = card.L0; L < card.L1; ++L)
            if (auto e = card.cache.upload_layer(q, m, L); !e.empty()) return "dense cache: " + e;
        if (card.L1 == c.n_layers)
            if (auto e = card.cache.upload_head(q, m, [&] { const char* v = std::getenv("IE_DS41_HEAD_FP8"); return v && *v ? std::string(v) == "1" : opt.head_fp8; }()); !e.empty()) return "dense cache: " + e;
        if (auto e = ensure_state(card); !e.empty()) return e;      // before the budget: it is part of the reserve's job
        // expert parallel: the staging buffers and the control arm's empty tier come out of the
        // measured VRAM too, so they go BEFORE the budget query (the control arm's warm pp2048 pass
        // read 11.6 s against 7.3 with them allocated after it: the eviction signature of Phase 9)
        if (ep_) {
            const size_t n = size_t(fwd_cap_) * c.dim, nr = size_t(fwd_cap_) * c.n_activated_experts;   // Phase 24: per-call staging
            card.ep_x = sycl::malloc_device<float>(n, q); card.ep_hx = sycl::malloc_host<float>(n, q);
            card.ep_ypc = sycl::malloc_device<sycl::half>(nr * c.dim, q); card.ep_hyc = sycl::malloc_host<sycl::half>(nr * c.dim, q);
            card.ep_rows = sycl::malloc_device<int32_t>(nr, q);
            if (!card.ep_x || !card.ep_hx || !card.ep_ypc || !card.ep_hyc || !card.ep_rows) return "init_resident: expert-parallel staging alloc failed";
            q.memset(card.ep_x, 0, n * 4); q.memset(card.ep_ypc, 0, nr * c.dim * 2); q.memset(card.ep_rows, 0, nr * 4); q.wait();
        }

        uint32_t n_static = opt.n_static, n_pinned = opt.n_pinned;
        if (n_static + n_pinned == 0) {
            const auto& dev = q.get_device();
            const uint64_t free_now = dev.has(sycl::aspect::ext_intel_free_memory)
                                    ? dev.get_info<sycl::ext::intel::info::device::free_memory>() : 0;
            if (!free_now) return "init_resident: device does not report free memory; pass n_static/n_pinned explicitly";
            // The reserve is headroom for ONE forward's scratch, so it has to follow the per-forward bound: the
            // default 6 GiB covers 2,048 tokens, and a caller who raises IE_DS41_MAX_FWD would otherwise load
            // fine and then OOM on its first big call (seen building Phase 24 step 2 at a 8,192 bound).
            // What must stay free for ONE forward, computed rather than guessed: the tier's expert batch workspace
            // (allocated inside tier.init AFTER this budget, so it was never subtracted) plus the call's own scratch,
            // which measures ~0.9 MB per token (the two hc streams, q/o at 64 heads x 512, the masks). The
            // configured floor still wins at small bounds, so the default 2,048-token sizing is unchanged.
            const uint64_t batch_ws = ds4_expert_batch_ws_bytes(fwd_cap_, c.n_activated_experts, c.dim, c.moe_inter_dim);
            const uint64_t scratch  = uint64_t(fwd_cap_) * 920ull * 1024ull;
            // The context-scaled part of the scratch, which the 920 KB/token above does NOT cover: `scores`,
            // `mcomp` and `mask` are [STRIP, NC] and the candidate stage's keep-mask is [fwd_cap, NC/block].
            // Getting this wrong is what made step 1's reserve short by 3x at 250k, so it is computed from the
            // same STRIP default the forward uses rather than estimated per token.
            const uint64_t nc_max = cap_pos_;
            const uint64_t strip  = std::min<uint64_t>(fwd_cap_, [] { const char* v = std::getenv("IE_DS41_STRIP"); const long n = v && *v ? std::atol(v) : 256; return n > 0 ? uint64_t(n) : uint64_t(1) << 40; }());
            const uint64_t cand_bytes = (c.candidate_source_layer >= 0 && c.candidate_block_size)
                                        ? uint64_t(fwd_cap_) * ((nc_max + c.candidate_block_size - 1) / c.candidate_block_size)
                                          + strip * ((nc_max + c.candidate_block_size - 1) / c.candidate_block_size) * 4ull : 0ull;
            const uint64_t nc_scratch = strip * nc_max * 4ull * 3ull + cand_bytes;
            const uint64_t reserve  = std::max<uint64_t>(opt.vram_reserve, batch_ws + scratch + nc_scratch);
            if (reserve > opt.vram_reserve)
                std::fprintf(stderr, "[ds41 forward] card %zu: keeping %.2f GiB free for one %u-token forward (batch workspace %.2f + scratch %.2f + context-scaled %.2f at strip %llu)\n",
                             ci, double(reserve) / 1073741824.0, fwd_cap_, double(batch_ws) / 1073741824.0, double(scratch) / 1073741824.0,
                             double(nc_scratch) / 1073741824.0, (unsigned long long)strip);
            const uint64_t for_experts = free_now > reserve ? free_now - reserve : 0;
            const uint64_t per_slot_all_layers = uint64_t(tnL) * lay.bytes;     // one slot in every layer of THIS tier
            const uint32_t experts_here = ep_ == 1 ? (c.n_routed_experts - ci + n_cards - 1) / n_cards : c.n_routed_experts;
            const uint32_t slots = uint32_t(std::min<uint64_t>(experts_here, for_experts / per_slot_all_layers));
            if (slots <= stream) return "init_resident: VRAM left for experts (" + std::to_string(for_experts >> 20) + " MiB) gives " +
                                        std::to_string(slots) + " slots/layer, not enough for the stream partition";
            n_static = slots - stream;
            const uint64_t host_cap = ep_ == 1 ? pin_cap / n_cards : pin_cap * nL / c.n_layers;   // this card's share of the pinned tier
            n_pinned = uint32_t(std::min<uint64_t>(experts_here - n_static, host_cap / per_slot_all_layers));
        }
        if (split_cores) card.tier.set_cpu_cores(ci == 0 ? "8-13" : "14-19");
        // The static claim is derived from the free VRAM this card reports, and that estimate can
        // overshoot what the driver will actually hand out -- by the allocator's own padding, or because
        // something else took a few hundred MB since the query. When it does, the allocation used to
        // abort the process with an uncaught Level Zero OUT_OF_DEVICE_MEMORY, which is how a load that
        // works at one context fails at another (a SMALLER context leaves more VRAM free, so MORE static
        // slots are claimed, so the claim is likelier to overshoot -- the founder's Dream launch loaded at
        // ctx 32768 and died at 4096). Give back one slot per layer at a time instead of dying: the
        // experts that lose a static home fall through to the pinned tier, which is a speed cost of a few
        // ms/token, not a correctness change.
        {
            std::string te; bool ok = false;
            for (uint32_t give_back = 0; give_back <= 8 && n_static > give_back; ++give_back) {
                const uint32_t ns = n_static - give_back;
                try {
                    te = card.tier.init(q, m, tL0, tnL, ranking, ns, n_pinned, stream, fwd_cap_, ep_ == 1 ? uint32_t(ci) : 0u, ep_ == 1 ? n_cards : 1u);
                } catch (const sycl::exception& ex) { te = std::string("threw: ") + ex.what(); }
                if (te.empty()) {
                    if (give_back) std::fprintf(stderr, "[ds41 forward] card %zu: the VRAM claim overshot; %u static slots per layer instead of %u (the rest stream from the pinned tier)\n", ci, ns, n_static);
                    n_static = ns; ok = true; break;
                }
                if (te.find("OUT_OF_DEVICE_MEMORY") == std::string::npos && te.find("malloc_device failed") == std::string::npos) break;   // not a budget problem
                card.tier.free_storage(q);                       // drop whatever the failed attempt did take
            }
            if (!ok) return "expert tier (layers " + std::to_string(tL0) + ".." + std::to_string(tL0 + tnL - 1) + "): " + te;
        }
        // the control arm's empty tier consumes NO resources -- no cache, arena, workspace or staging
        // (gate 14 finding 1: inside the budget it cost 5 static slots per layer; after the budget with
        // one slot per layer it cost 0.7 GiB of the reserve and evicted in the warm pp2048 pass)
        if (ep_ == 2)
            if (auto e = card.ctl_tier.init(q, m, 0, c.n_layers, ranking, 0, 0, 0, fwd_cap_, c.n_routed_experts, c.n_routed_experts); !e.empty())
                return "control tier: " + e;
    }
    if (ep_ && ds41_ep_thread_on()) for (auto& cp : cards_) {          // Phase 19: one parked helper per card, its block time set once
        Card* cd = cp.get();
        cd->ep_thread = std::thread([cd] {
            kmp_set_blocktime(0);
            for (;;) {
                std::function<void()> job;
                { std::unique_lock<std::mutex> lk(cd->ep_mu); cd->ep_cv.wait(lk, [&] { return cd->ep_pending || cd->ep_stop; }); if (cd->ep_stop) return; job = std::move(cd->ep_job); cd->ep_pending = false; }
                job();
                { std::lock_guard<std::mutex> lk(cd->ep_mu); cd->ep_done = true; }
                cd->ep_cv.notify_all();
            }
        });
    }
    // The engram tables are read one random 256-byte row per (token, column) (ds41_engram_gather), but the tiers' init
    // above advised the whole store MADV_NORMAL for the dense path -- and under it every row's fault read around a 128 KB
    // window: ~12.5 GB of NVMe reads per 2,048-token chunk, 2.2-2.5 s of the first card's host prep, which made card 0
    // the pipeline's slow stage (busy 29 s vs 16 s at 10k tokens). Back to MADV_RANDOM for these ranges: one page a row.
    for (const uint32_t Le : tables.layer_ids)
        for (const SafeTensorInfo* t : {m.layers()[Le].engram_embed.w, m.layers()[Le].engram_embed.s}) {
            if (!t || !t->data || !t->nbytes) continue;
            const uintptr_t a0 = reinterpret_cast<uintptr_t>(t->data) & ~uintptr_t(4095);
            const uintptr_t a1 = (reinterpret_cast<uintptr_t>(t->data) + t->nbytes + 4095) & ~uintptr_t(4095);
            if (madvise(reinterpret_cast<void*>(a0), a1 - a0, MADV_RANDOM) != 0) return "init_resident: madvise(MADV_RANDOM) on the engram table " + t->name + " failed";
        }
    g_ds4_attn_split_fixed64 = true;   // DSpark P2: the T-row step's attention combine is the one-row step's (bit-identical)
    if (ep_) std::fprintf(stderr, "[ds41 forward] expert parallel %s on %u cards\n", ep_ == 2 ? "CONTROL ARM (the machinery, no parallelism)" : "ON (parity share of every layer's experts per card)", n_cards);
    resident_ = true;
    return {};
}

std::string Ds41Forward::ensure_state(Card& card) {
    if (card.state_ready) return {};
    const auto& c = m_->config();
    sycl::queue& q = *card.q;
    const uint32_t HD = c.head_dim, IHD = c.index_head_dim, WIN = c.window_size;
    card.state.assign(c.n_layers, LayerState{});
    auto alloc = [&](size_t n) -> float* { float* p = sycl::malloc_device<float>(n, q); if (p) q.memset(p, 0, n * 4); return p; };
    for (uint32_t L = card.L0; L < card.L1; ++L) {
        auto& st = card.state[L]; const auto& k = m_->layers()[L].kind;
        if (!(st.win_kv = alloc(size_t(WIN) * HD))) return "state: window ring alloc failed";
        if (k.is_kv_source) {
            if (!(st.comp_kv = alloc(size_t(cap_pos_) * HD))) return "state: latent cache alloc failed";
            if (!(st.idx_k = alloc(size_t(cap_pos_) * IHD))) return "state: index key cache alloc failed";
            if (k.compress_ratio > 1) {
                if (!(st.part_kv = alloc(size_t(k.compress_ratio) * HD)) || !(st.part_gate = alloc(size_t(k.compress_ratio) * HD)))
                    return "state: partial group alloc failed";
            }
        }
    }
    if (!(card.imp_ckv = alloc(size_t(cap_pos_) * HD)) || !(card.imp_ik = alloc(size_t(cap_pos_) * IHD))) return "state: import alloc failed";
    // Phase 18 (docs/deepseek41/46 term 2): the two RoPE inv_freq tables resident (the host's
    // ds4_rope_inv_freq, uploaded once), the step's positions (pinned host + device) and their cos / sin
    {
        const uint32_t RD = c.rope_head_dim, HR = RD / 2;
        Ds4RopeConfig rmain; rmain.theta = c.rope_theta; rmain.yarn = false;
        Ds4RopeConfig rcomp; rcomp.theta = c.compress_rope_theta; rcomp.yarn = true; rcomp.factor = c.rope_factor;
        rcomp.beta_fast = float(c.beta_fast); rcomp.beta_slow = float(c.beta_slow); rcomp.original_max_pos = c.original_seq_len;
        std::vector<float> inv = ds4_rope_inv_freq(rmain, RD);
        const std::vector<float> ic = ds4_rope_inv_freq(rcomp, RD); inv.insert(inv.end(), ic.begin(), ic.end());
        card.rope_inv = sycl::malloc_device<float>(inv.size(), q); card.rope_pos = sycl::malloc_device<int32_t>(kDs41RopeSlots, q); card.rope_hpos = sycl::malloc_host<int32_t>(kDs41RopeSlots, q);
        card.rope_cs = sycl::malloc_device<float>(size_t(kDs41RopeSlots) * HR, q); card.rope_sn = sycl::malloc_device<float>(size_t(kDs41RopeSlots) * HR, q);
        if (!card.rope_inv || !card.rope_pos || !card.rope_hpos || !card.rope_cs || !card.rope_sn) return "state: rope table alloc failed";
        q.memcpy(card.rope_inv, inv.data(), inv.size() * 4); q.memset(card.rope_pos, 0, kDs41RopeSlots * 4); std::fill_n(card.rope_hpos, kDs41RopeSlots, 0);
    }
    q.wait_and_throw();
    card.state_ready = true;
    return {};
}

// P3 (docs/deepseek41/56): the rollback snapshots -- what a T-row step overwrites. Allocated the first time such a
// step runs (gate P4: an engine that never speculates should hold none of it), freed with the rest of the state.
std::string Ds41Forward::ensure_snapshots(Card& card) {
    if (card.snaps_ready) return {};
    const auto& c = m_->config(); const uint32_t HD = c.head_dim;
    sycl::queue& q = *card.q;
    auto alloc = [&](size_t n) -> float* { float* p = sycl::malloc_device<float>(n, q); if (p) q.memset(p, 0, n * 4); return p; };
    for (uint32_t L = card.L0; L < card.L1; ++L) {
        auto& st = card.state[L]; const auto& k = m_->layers()[L].kind;
        if (!st.snap_ring && !(st.snap_ring = alloc(size_t(kDs41MaxDecodeRows) * HD))) return "state: ring snapshot alloc failed";
        if (k.is_kv_source && k.compress_ratio > 1 && !st.snap_ckv) {
            if (!(st.snap_ckv = alloc(size_t(kDs41MaxDecodeRows) * HD)) || !(st.snap_cg = alloc(size_t(kDs41MaxDecodeRows) * HD)) ||
                !(st.snap_part_kv = alloc(HD)) || !(st.snap_part_gate = alloc(HD))) return "state: compressor snapshot alloc failed";
        }
    }
    q.wait_and_throw();
    card.snaps_ready = true;
    return {};
}

void Ds41Forward::free_state(Card& card) {
    if (!card.state_ready) return;
    sycl::queue& q = *card.q;
    for (auto& st : card.state) {
        for (float* p : {st.win_kv, st.comp_kv, st.idx_k, st.part_kv, st.part_gate, st.snap_ring, st.snap_ckv, st.snap_cg, st.snap_part_kv, st.snap_part_gate}) if (p) sycl::free(p, q);
        st = LayerState{};
    }
    if (card.imp_ckv) sycl::free(card.imp_ckv, q); if (card.imp_ik) sycl::free(card.imp_ik, q);
    card.imp_ckv = nullptr; card.imp_ik = nullptr; card.snaps_ready = false;
    for (auto& lst : card.dec_scratch) for (auto& [p, n] : lst) sycl::free(p, q);
    card.dec_scratch.clear(); card.dec_cursor.clear();
    for (void* p : {static_cast<void*>(card.rope_inv), static_cast<void*>(card.rope_pos), static_cast<void*>(card.rope_hpos),
                    static_cast<void*>(card.rope_cs), static_cast<void*>(card.rope_sn)}) if (p) sycl::free(p, q);
    card.rope_inv = nullptr; card.rope_pos = nullptr; card.rope_hpos = nullptr; card.rope_cs = nullptr; card.rope_sn = nullptr;
    card.state_ready = false;
}

void Ds41Forward::reset_state() {
    for (auto& cp : cards_) for (auto& st : cp->state) { st.nc = 0; st.part_valid = false; }
    n_pos_ = 0; all_ids_.clear();
    pc_clear_live();   // Phase 46: every live checkpoint's latents are about to be rewritten
}

std::string Ds41Forward::read_state(uint32_t L, std::vector<float>& win, std::vector<float>& comp, std::vector<float>& idxk, uint32_t& nc) const {
    const auto& c = m_->config();
    for (const auto& cp : cards_) {
        if (L < cp->L0 || L >= cp->L1) continue;
        if (!cp->state_ready) return "read_state: no state yet";
        const auto& st = cp->state[L]; sycl::queue& q = *cp->q;
        win.assign(size_t(c.window_size) * c.head_dim, 0.f);
        q.memcpy(win.data(), st.win_kv, win.size() * 4).wait();
        nc = st.nc; comp.clear(); idxk.clear();
        if (st.comp_kv && nc) {
            comp.resize(size_t(nc) * c.head_dim); idxk.resize(size_t(nc) * c.index_head_dim);
            q.memcpy(comp.data(), st.comp_kv, comp.size() * 4); q.memcpy(idxk.data(), st.idx_k, idxk.size() * 4); q.wait();
        }
        return {};
    }
    return "read_state: layer " + std::to_string(L) + " is on no card";
}

void Ds41Forward::ep_thread_stop(Card& card) {
    if (!card.ep_thread.joinable()) return;
    { std::lock_guard<std::mutex> lk(card.ep_mu); card.ep_stop = true; }
    card.ep_cv.notify_all(); card.ep_thread.join(); card.ep_stop = false;
}

void Ds41Forward::free_resident() {
    // DRAIN BEFORE FREEING. Every allocation below is referenced by kernels that may still be in
    // flight: an aborted generation (a client disconnect, or the server's own /admin/shutdown
    // "aborting 1 in-flight") returns from the generate loop while the last forward's submissions
    // are still queued. Freeing under them leaves the driver holding a job whose memory is gone,
    // which the GuC then times out -- `Timedout job ... in no process [-1]`, GT reset loops, and a
    // card that refuses new contexts until the box is rebooted. That wedged the second B70 twice on
    // 2026-09-14 (once from hard-aborted test processes, once from Dream's shutdown mid-reply).
    // The helpers are stopped first so nothing new is submitted, then every queue is drained; a
    // teardown drain reports async errors instead of throwing, because there is nothing left to save.
    for (auto& cp : cards_) ep_thread_stop(*cp);
    for (auto& cp : cards_) if (cp->q) {
        try { cp->q->wait_and_throw(); }
        catch (const sycl::exception& e) { std::fprintf(stderr, "[ds41 forward] teardown drain: %s\n", e.what()); }
    }
    set_prefix_cache(false);   // Phase 46: its pinned blocks belong to these queues' contexts
    for (auto& cp : cards_) {
        free_state(*cp); cp->tier.free_storage(*cp->q); cp->ctl_tier.free_storage(*cp->q); cp->cache.free_all(*cp->q);
        for (void** b : {reinterpret_cast<void**>(&cp->ep_x), reinterpret_cast<void**>(&cp->ep_hx), reinterpret_cast<void**>(&cp->ep_ypc),
                         reinterpret_cast<void**>(&cp->ep_hyc), reinterpret_cast<void**>(&cp->ep_rows)}) if (*b) { sycl::free(*b, *cp->q); *b = nullptr; }
    }
    ep_ = 0;
    resident_ = false;
}

uint64_t Ds41Forward::dense_bytes() const { uint64_t b = 0; for (const auto& cp : cards_) b += cp->cache.bytes(); return b; }
uint64_t Ds41Forward::expert_vram_bytes() const { uint64_t b = 0; for (const auto& cp : cards_) b += cp->tier.vram_bytes(); return b; }
uint64_t Ds41Forward::expert_pinned_bytes() const { uint64_t b = 0; for (const auto& cp : cards_) b += cp->tier.pinned_bytes(); return b; }

Ds41Forward::CardInfo Ds41Forward::card_info(uint32_t ci) const {
    CardInfo info;
    if (ci >= cards_.size()) return info;
    const Card& card = *cards_[ci];
    info.first_layer = card.L0; info.n_layers = card.L1 - card.L0;
    info.n_static = card.tier.n_static(); info.n_stream = card.tier.n_stream(); info.n_pinned = card.tier.n_pinned();
    info.dense_bytes = card.cache.bytes(); info.expert_vram_bytes = card.tier.vram_bytes(); info.pinned_bytes = card.tier.pinned_bytes();
    return info;
}

uint32_t Ds41Forward::expert_tier(uint32_t L, uint32_t e) const {
    if (!resident_) return 2u;
    if (ep_ == 1) { for (const auto& cp : cards_) if (const uint32_t t = cp->tier.tier_of(L, e); t != 3u) return t; return 2u; }
    for (const auto& cp : cards_) if (L >= cp->L0 && L < cp->L1) return cp->tier.tier_of(L, e);
    return 2u;
}

void Ds41Forward::reset_profile() { for (auto& r : profile_) std::fill(r.begin(), r.end(), 0u); }

std::string Ds41Forward::write_profile(const std::string& path, const std::string& note) const {
    const uint32_t E = m_->config().n_routed_experts;
    std::vector<std::vector<uint32_t>> orders(profile_.size());
    std::vector<uint64_t> counts;
    uint64_t total = 0;
    for (size_t L = 0; L < profile_.size(); ++L) {
        orders[L].resize(E); std::iota(orders[L].begin(), orders[L].end(), 0u);
        std::stable_sort(orders[L].begin(), orders[L].end(), [&](uint32_t a, uint32_t b) { return profile_[L][a] > profile_[L][b]; });
        for (uint64_t v : profile_[L]) total += v;
    }
    return ds4_expert_priority_write_layers(path, orders, counts, {note, "total selections " + std::to_string(total)});
}

void Ds41Forward::set_vision_span(uint32_t pos0, std::vector<float> rows) {
    VisSpan sp; sp.pos0 = pos0; sp.n = uint32_t(rows.size() / m_->config().dim); sp.rows = std::move(rows);
    vis_spans_.push_back(std::move(sp));
}

std::string Ds41Forward::forward(const int32_t* ids, uint32_t T, uint32_t pos0, std::vector<float>& logits_out,
                                 const Probe& probe, const std::vector<std::vector<int32_t>>* expected,
                                 const std::vector<std::vector<float>>* expected_w, bool force_routing,
                                 const std::vector<std::vector<float>>* expected_gap) {
    // Phase 46: a call that fails or throws part-way may have appended latents / written rings on some layers without
    // committing n_pos -- with the prefix cache on, that live state must never be reused, so it is dropped
    try {
        std::string e = forward_impl(ids, T, pos0, logits_out, probe, expected, expected_w, force_routing, expected_gap, nullptr);
        if (!e.empty() && pc_on_) reset_state();
        return e;
    } catch (...) { if (pc_on_) reset_state(); throw; }
}

std::string Ds41Forward::forward_impl(const int32_t* ids, uint32_t T, uint32_t pos0, std::vector<float>& logits_out,
                                      const Probe& probe, const std::vector<std::vector<int32_t>>* expected,
                                      const std::vector<std::vector<float>>* expected_w, bool force_routing,
                                      const std::vector<std::vector<float>>* expected_gap, const Stage* stage) {
    if (force_routing && (!expected || !expected_w)) return "force_routing needs expected_routing and expected_weights";
    if (cards_.empty()) return "forward: not initialised";
    // Phase 43: a pipeline stage runs cards [card_first, card_last) only; the whole-model call is the one stage [0, n)
    const size_t card_first = stage ? stage->card_first : 0, card_last = stage ? stage->card_last : cards_.size();
    const bool first_stage = card_first == 0, last_stage = card_last == cards_.size();
    if (stage && (probe || expected || force_routing)) return "forward: a pipeline stage takes no probe or expected routing";
    if (stage && (card_first >= card_last || card_last > cards_.size() || !first_stage != bool(stage->in) || !last_stage != bool(stage->out)))
        return "forward: malformed pipeline stage";
    const auto& m = *m_;
    const auto& c = m.config();
    const uint32_t H = c.dim, HC = c.hc_mult, QR = c.q_lora_rank, HD = c.head_dim, NH = c.n_heads;
    const uint32_t QH = NH * HD, RD = c.rope_head_dim, WIN = c.window_size;
    const uint32_t G = c.o_groups, OPG = c.o_lora_rank, IPG = QH / G, OR = G * OPG;
    const uint32_t IH = c.index_n_heads, IHD = c.index_head_dim, IQ = IH * IHD;
    const uint32_t EF = c.moe_inter_dim, E = c.n_routed_experts, TK = c.n_activated_experts;
    const uint32_t FLAT = HC * H;
    const uint32_t ENC = tb_->n_hash_cols, EHD = tb_->head_dim, EIN = ENC * EHD, EOUT = H * (HC + 1);
    const uint32_t V = c.vocab_size;
    const float LOWEST = std::numeric_limits<float>::lowest();
    if (!stage) stats_.assign(c.n_layers, {});
    else for (size_t ci = card_first; ci < card_last; ++ci) for (uint32_t L = cards_[ci]->L0; L < cards_[ci]->L1; ++L) stats_[L] = {};   // its own layers only: the other stages write theirs meanwhile

    // ---- the two admitted regimes --------------------------------------------------------------
    const bool decode = pos0 > 0;
    if (!decode) {
        for (uint32_t L = 0; L < c.n_layers; ++L)
            if (const uint32_t r = m.layers()[L].kind.compress_ratio; r > 1 && T % r != 0)
                return "forward: a prefill's T must be a multiple of every compress_ratio (a partial group at the prompt's end is decode's, not yet admitted)";
        if (first_stage) reset_state();   // a later stage never resets: the first stage did, and runs the next chunk on its cards meanwhile
    } else {
        static const bool multi = [] { const char* v = std::getenv("IE_DS41_DECODE_MULTI"); return v && *v && std::string(v) != "0"; }();   // DSpark P2, opt-in until its gate
        // Phase 24 step 2 (docs/deepseek41/65): a PREFILL CONTINUATION -- T > the DSpark row cap at pos0 > 0, so a long
        // prompt can be fed in chunks. The compressor, the latent append and every position table were already written
        // general in T and pos0 for P2's T-row decode; what a chunk needs beyond them is its window keys as a dense
        // segment (the 128-slot ring cannot hold both the previous window and T > WIN new rows) -- built below.
        // Phase 24 step 2: DEFAULT ON since 2026-09-16 -- the server never set the opt-in, so every chat prompt longer than
        // one 2,048-token forward (any request carrying Dream's tool schemas) was refused at its second chunk.
        // IE_DS41_CONT=0 is the kill switch.
        static const bool cont_on = [] { const char* v = std::getenv("IE_DS41_CONT"); return !(v && *v && std::string(v) == "0"); }();
        if (T != 1 && !((multi || multi_rows_) && T >= 2 && T <= kDs41MaxDecodeRows) && !(cont_on && T > kDs41MaxDecodeRows)) return "forward: at pos0 > 0 only single-token steps are admitted (T == 1; up to " + std::to_string(kDs41MaxDecodeRows) + " with IE_DS41_DECODE_MULTI=1)";
        if (!stage && pos0 != n_pos_) return "forward: step at pos0 " + std::to_string(pos0) + " but the state holds " + std::to_string(n_pos_) + " positions";   // forward_pipelined checks the whole chunk list up front
    }
    if (pos0 + T > cap_pos_) return "forward: position " + std::to_string(pos0 + T) + " exceeds the state's capacity " + std::to_string(cap_pos_) + " (max_tokens)";
    if (T > fwd_cap_) return "forward: " + std::to_string(T) + " tokens in one call exceeds this runtime's per-forward limit " + std::to_string(fwd_cap_) +
                             " (the context holds " + std::to_string(cap_pos_) + "; a longer prompt must be fed in chunks of at most that, which Phase 24 step 2 admits -- or raise IE_DS41_MAX_FWD and pay the scratch in VRAM)";
    // Decoder SWA Bounded Replay: the decoder half over the last `WIN` prompt tokens only (docs/27)
    const uint32_t dec0 = c.n_layers / 2;
    const bool replay = bounded_replay_ && !decode && T > WIN;
    // Phase 24 step 2: a prefill continuation chunk -- the position tables, the compressor, the latent append and the
    // state updates are general in T and pos0, and the window segment is built here. The card-0 page fault this path
    // once hit was the P3 snapshot writing T rows into an 8-row buffer (fixed, 9a04274); the continuation is proven
    // against a regrouping control there and gated by ie-ds41-cont-test (15/15).
    const bool cont = decode && T > kDs41MaxDecodeRows;
    // Phase 35 (docs/deepseek41/75): a host wait after a launch serializes the host with the device -- the next
    // kernel cannot be enqueued until this one has finished -- and the decode path had ~12 of them per layer that
    // fed no host read. They stay only where a host read follows (the routing readback, the logits, host-vector
    // uploads) or a diagnostic needs them: a probe, or IE_DS41_STAGES=1 for the per-stage timers.
    static const bool stages_env = [] { const char* e = std::getenv("IE_DS41_STAGES"); return e && *e; }();
    const bool hostsync = stages_env || bool(probe);
    auto W = [&](sycl::event e) { if (hostsync) e.wait(); };
    // IE_DS41_TRACE=1: one line per phase, which is how the step-2 fault was localised to the setup block.
    static const bool trace = [] { const char* v = std::getenv("IE_DS41_TRACE"); return v && *v && std::string(v) != "0"; }();
    auto TR = [&](const char* what, uint32_t L) { if (trace) { std::fprintf(stderr, "[trace] T=%u pos0=%u cont=%d L=%u %s\n", T, pos0, int(cont), L, what); std::fflush(stderr); } };
    TR("enter", 0);
    if (replay && !m.layers()[dec0].kind.is_kv_source)
        return "forward: bounded replay needs the first decoder layer (" + std::to_string(dec0) + ") to be a kv source";
    if (replay && expected) return "forward: expected routing is judged on the exact path only (set_bounded_replay(false))";
    // a kv source ABOVE the transition would compress the segment alone (latents 0..127 against an
    // absolute indexer threshold: every key admitted, silently wrong); this model has none (2/8/14/20)
    if (replay) for (uint32_t L = dec0 + 1; L < c.n_layers; ++L)
        if (m.layers()[L].kind.is_kv_source) return "forward: bounded replay needs every kv source at or below the first decoder layer (layer " + std::to_string(L) + " is one)";
    // the head runs over the segment, so the logits would come back with 128 rows instead of T --
    // refused rather than handed to a caller indexing row T-1
    if (replay && !logits_last_only_) return "forward: bounded replay needs set_logits_last_only(true) (the head sees the 128-row segment)";
    uint32_t T_carry = T;                 // the rows the next card receives: T, or the segment after the transition
    uint32_t off_carry = 0;               // ... and those rows' offset into the prompt (0 until a replay transition)
    if (stage && T >= 2 && T <= kDs41MaxDecodeRows) return "forward: a pipeline stage carries no multi-row decode step";
    if (!stage) for (auto& cp : cards_) if (auto e = ensure_state(*cp); !e.empty()) return e;   // forward_pipelined ensures every card's before any stage starts
    if (T >= 2 && T <= kDs41MaxDecodeRows && pos0 > 0)                                  // P3, on first use (never for a continuation)
        for (auto& cp : cards_) if (auto e = ensure_snapshots(*cp); !e.empty()) return e;
    // the sequence the engram look-back reads: extended for this call's hashes, committed with
    // n_pos_ only on success (Phase 9 gate, finding 12: an errored step must not desync it).
    // Phase 43: only the first stage owns it -- a later stage never reads it, and it is being extended meanwhile.
    const size_t ids_before = first_stage ? all_ids_.size() : 0;
    if (first_stage) all_ids_.insert(all_ids_.end(), ids, ids + T);
    struct RollBack { std::vector<int32_t>& v; size_t n; bool ok = false; ~RollBack() { if (!ok) v.resize(n); } } ids_guard{all_ids_, ids_before, !first_stage};
    if (!first_stage)
        for (uint32_t Le : tb_->layer_ids) if (Le >= cards_[card_first]->L0) return "forward: a later pipeline stage holds an engram layer (" + std::to_string(Le) + "); its host prep is the first stage's";

    // ---- host-side preparation, shared by every card ---------------------------------------
    const auto t_prep = std::chrono::steady_clock::now();
    // engram hashes + gathers (they depend only on the ids). A step's hashes come from the last
    // max_ngram ids of the sequence: the look-back never reaches further, so the window's last
    // row equals the full sequence's value at that position.
    std::vector<std::vector<sycl::half>> engram_in16(tb_->layer_ids.size());
    if (first_stage) last_hashes_.assign(tb_->layer_ids.size(), {});
    for (size_t li = 0; first_stage && li < tb_->layer_ids.size(); ++li) {
        std::vector<int64_t> hash;
        if (!decode) {
            hash.resize(size_t(T) * ENC);
            ds41_engram_hash(*tb_, ids, T, uint32_t(li), hash.data());
        } else {
            const uint32_t n = uint32_t(all_ids_.size()), nw = std::min(n, tb_->max_ngram_size - 1 + T);   // the look-back plus this step's T rows
            std::vector<int64_t> hw(size_t(nw) * ENC);
            ds41_engram_hash(*tb_, all_ids_.data() + (n - nw), nw, uint32_t(li), hw.data());
            hash.assign(hw.end() - std::ptrdiff_t(size_t(T) * ENC), hw.end());                    // the last T rows
        }
        last_hashes_[li] = hash;
        std::vector<float> emb(size_t(T) * EIN);
        ds41_engram_gather(m.layers()[tb_->layer_ids[li]].engram_embed, hash.data(), T, ENC, EHD, emb.data());
        engram_in16[li].resize(emb.size());
        for (size_t i = 0; i < emb.size(); ++i) engram_in16[li][i] = sycl::half(emb[i]);
    }
    // embed -> hc_mult copies: the hyper-connection stream. It lives on whichever card is
    // running and is bounced through this host copy at every card boundary, with the previous
    // layer's ffn_pre mix and the source caches a consumer on the next card would need.
    std::vector<float> hh(first_stage ? size_t(T) * FLAT : 0);
    if (first_stage) {
        const auto* EW = reinterpret_cast<const uint16_t*>(m.embed.w->data);
        for (uint32_t t = 0; t < T; ++t) {
            const float* vrow = nullptr;                       // an image position: the tower's row, not the table's
            if (ids[t] < 0) {
                const uint32_t p = pos0 + t;
                for (const auto& sp : vis_spans_) if (p >= sp.pos0 && p - sp.pos0 < sp.n) { vrow = sp.rows.data() + size_t(p - sp.pos0) * H; break; }
                if (!vrow && vis_provider_) { if (std::string ve = vis_provider_(p, vrow); !ve.empty()) return "forward: vision: " + ve; }
                if (!vrow) return "forward: position " + std::to_string(p) + " is an image position and no vision span covers it";
            }
            for (uint32_t d = 0; d < H; ++d) {
                const float v = vrow ? vrow[d] : bf16f(EW[size_t(ids[t]) * H + d]);
                for (uint32_t cpy = 0; cpy < HC; ++cpy) hh[(size_t(t) * HC + cpy) * H + d] = v;
            }
        }
    }
    // image rows among the last `n` rows of this call, as (local row, count) runs: the engram passes them through
    // untouched and the router selects by the image bias there (model.py: engram_mask, Gate.bias_vl)
    auto image_runs = [&](uint32_t n) {
        std::vector<std::pair<uint32_t, uint32_t>> runs;
        for (uint32_t t = T - n; t < T; ++t) {
            if (ids[t] >= 0) continue;
            if (!runs.empty() && runs.back().first + runs.back().second == t - (T - n)) ++runs.back().second;
            else runs.push_back({t - (T - n), 1});
        }
        return runs;
    };
    std::vector<float> hpm(size_t(T) * HC, 0.f);
    for (uint32_t t = 0; t < T; ++t) hpm[size_t(t) * HC] = 1.f;
    std::vector<float> h_imp_ckv, h_imp_ik;                      // the current source's caches, at a card boundary
    std::vector<int32_t> h_sh_topk(size_t(T) * c.index_topk);
    uint32_t bounce_nc = 0, sh_topk_w = 0; bool bounce_topk = false;
    std::vector<int32_t> h_idx(size_t(T) * TK); std::vector<float> h_w(size_t(T) * TK);
    if (!first_stage) {   // Phase 43: the stream and the boundary caches, exactly as the previous card left them
        StageCarry& in = *stage->in;
        hh = std::move(in.hh); hpm = std::move(in.hpm); h_imp_ckv = std::move(in.imp_ckv); h_imp_ik = std::move(in.imp_ik); h_sh_topk = std::move(in.sh_topk);
        T_carry = in.T_carry; off_carry = in.off_carry; bounce_nc = in.bounce_nc; bounce_topk = in.bounce_topk;
    }
    if (first_stage) prep_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_prep).count();

    // ---- one card's layer range -------------------------------------------------------------
    auto run_card = [&](Card& card) -> std::string {
    sycl::queue& q = *card.q;
    // the oneDNN projections named for the kernel profiler (oneDNN submits internally, so ie::ps
    // cannot wrap them; the event is pushed instead -- inert unless a tool set ie::g_profiler)
    auto pe = [](const char* name, sycl::event e) { if (ie::g_profiler) [[unlikely]] ie::g_profiler->push(name, e); return e; };

    // ---- allocation helpers ------------------------------------------------------------
    auto f32 = [&](size_t n) { return sycl::malloc_device<float>(n, q); };
    auto f16 = [&](size_t n) { return sycl::malloc_device<sycl::half>(n, q); };
    auto i32 = [&](size_t n) { return sycl::malloc_device<int32_t>(n, q); };
    std::vector<void*> scratch;
    // Phase 18 (docs/deepseek41/46 term 2): at decode the scratch comes from the card's persistent
    // list in the order the step asks for it (the same sizes every step: T = 1 and the state's
    // capacity); a size that does not match falls back to a fresh allocation, freed as before.
    // ON by default (docs/47: bit-identical by the dumps, -2.9 ms/token measured); `IE_DS41_SETUP_FAST=0` restores
    // the per-step allocations, the five RoPE round trips and the source-cache bounce (the kill switch).
    static const bool setup_fast = [] { const char* v = std::getenv("IE_DS41_SETUP_FAST"); return !(v && *v && std::string(v) == "0"); }();
    const bool dec_cache = decode && setup_fast;
    // Phase 43: the persistent lists serve DECODE-sized steps only. A continuation chunk's scratch (~0.9 MB per token
    // plus its context-scaled strips) used to stay allocated after the prefill -- one set per distinct chunk length,
    // inside the 6 GiB per-forward reserve -- which is where a larger chunk would run out. Same bytes, same arithmetic.
    const bool dec_keep = dec_cache && T <= kDs41MaxDecodeRows;
    if (dec_keep && card.dec_scratch.size() <= T) { card.dec_scratch.resize(T + 1); card.dec_cursor.resize(T + 1, 0); }   // one list per step length (P2)
    if (dec_keep) card.dec_cursor[T] = 0;
    auto take = [&](size_t bytes) -> void* {
        if (!dec_keep) return nullptr;
        auto& ds = card.dec_scratch[T]; size_t& cur = card.dec_cursor[T];
        if (cur < ds.size()) return ds[cur].second == bytes ? ds[cur++].first : nullptr;
        void* p = sycl::malloc_device<uint8_t>(bytes, q); if (!p) return nullptr;
        ds.push_back({p, bytes}); ++cur; return p;
    };
    auto S32 = [&](size_t n) { if (void* cp = take(n * 4)) return static_cast<float*>(cp); float* p = f32(n); scratch.push_back(p); return p; };
    auto S16 = [&](size_t n) { if (void* cp = take(n * 2)) return static_cast<sycl::half*>(cp); sycl::half* p = f16(n); scratch.push_back(p); return p; };
    auto SI  = [&](size_t n) { if (void* cp = take(n * 4)) return static_cast<int32_t*>(cp); int32_t* p = i32(n); scratch.push_back(p); return p; };
    auto to16 = [&](const float* s, sycl::half* d, size_t n) {
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { d[i] = sycl::half(s[i]); });
    };
    uint64_t layer_dense_bytes = 0;
    auto dense = [&](const Ds41Tensor& t) {                 // FP8 + 32x32 scale -> fp16
        const uint32_t N = uint32_t(t.w->shape[0]), K = uint32_t(t.w->shape[1]);
        const uint32_t bn = N / uint32_t(t.s->shape[0]), bk = K / uint32_t(t.s->shape[1]);
        uint8_t* dw = sycl::malloc_device<uint8_t>(size_t(N) * K, q);
        uint8_t* ds = sycl::malloc_device<uint8_t>(t.s->nbytes, q);
        sycl::half* o = f16(size_t(N) * K);
        q.memcpy(dw, t.w->data, size_t(N) * K); q.memcpy(ds, t.s->data, t.s->nbytes);
        ds41_dense_dequant_f16(q, dw, ds, N, K, bn, bk, o).wait();
        sycl::free(dw, q); sycl::free(ds, q);
        layer_dense_bytes += t.w->nbytes + t.s->nbytes;
        return o;
    };
    std::vector<sycl::half> hbuf16; std::vector<float> hbuf32;
    auto bf16_16 = [&](const Ds41Tensor& t) {              // BF16 -> fp16
        const size_t n = size_t(t.w->numel()); const auto* s = reinterpret_cast<const uint16_t*>(t.w->data);
        hbuf16.resize(n); for (size_t i = 0; i < n; ++i) hbuf16[i] = sycl::half(bf16f(s[i]));
        sycl::half* d = f16(n); q.memcpy(d, hbuf16.data(), n * 2).wait();
        layer_dense_bytes += t.w->nbytes; return d;
    };
    auto bf16_32 = [&](const Ds41Tensor& t) {              // BF16 -> fp32
        const size_t n = size_t(t.w->numel()); const auto* s = reinterpret_cast<const uint16_t*>(t.w->data);
        hbuf32.resize(n); for (size_t i = 0; i < n; ++i) hbuf32[i] = bf16f(s[i]);
        float* d = f32(n); q.memcpy(d, hbuf32.data(), n * 4).wait();
        layer_dense_bytes += n * 4; return d;                  // DEVICE bytes (gate finding 6)
    };
    auto f32_up = [&](const Ds41Tensor& t) {               // F32 verbatim
        const size_t n = size_t(t.w->numel()); float* d = f32(n);
        q.memcpy(d, t.w->data, n * 4).wait(); layer_dense_bytes += t.w->nbytes; return d;
    };

    // ---- RoPE tables: main for ratio-0 layers, compress (YaRN) for the rest ----------------
    Ds4RopeConfig rmain; rmain.theta = c.rope_theta; rmain.yarn = false;
    Ds4RopeConfig rcomp; rcomp.theta = c.compress_rope_theta; rcomp.yarn = true; rcomp.factor = c.rope_factor;
    rcomp.beta_fast = float(c.beta_fast); rcomp.beta_slow = float(c.beta_slow); rcomp.original_max_pos = c.original_seq_len;
    auto rope_tables = [&](const Ds4RopeConfig& cfg, const std::vector<int32_t>& pos, float*& cs, float*& sn) {
        const auto inv = ds4_rope_inv_freq(cfg, RD);
        float* d_inv = f32(inv.size()); q.memcpy(d_inv, inv.data(), inv.size() * 4);
        int32_t* d_pos = SI(pos.size()); q.memcpy(d_pos, pos.data(), pos.size() * 4).wait();
        cs = S32(pos.size() * (RD / 2)); sn = S32(pos.size() * (RD / 2));
        ds4_rope_cos_sin(q, d_inv, d_pos, cs, sn, uint32_t(pos.size()), RD / 2, 1.0f).wait();
        sycl::free(d_inv, q); return d_pos;
    };
    // this card's rows' positions: the prompt's, or the segment's after a transition on an earlier card
    std::vector<int32_t> pos_t(T_carry); std::iota(pos_t.begin(), pos_t.end(), int32_t(pos0 + off_carry));
    // q/kv positions: the real ones, or (decode, criterion 5's negative control) shifted by rope_off_
    std::vector<int32_t> pos_qk(pos_t);
    if (decode && rope_off_) for (auto& p : pos_qk) p = std::max(0, p + rope_off_);
    float *cos_m, *sin_m, *cos_c, *sin_c, *cos_g2, *sin_g2, *cos_g1, *sin_g1;
    int32_t* d_pos_t;
    // Phase 18's one-copy RoPE path writes 4T + n2 entries into card.rope_hpos, which is kDs41RopeSlots long --
    // 36 of 40 at T = 8, and that fit was load-bearing for the DSpark row cap without anything saying so. A Phase 24
    // continuation carries T up to the forward bound, so the capacity is now CHECKED and a longer call takes the
    // general host-built tables below. (Before this check, a decode call with T > 8 wrote 4,608 entries into 40 and
    // surfaced as UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY from the copy that followed.)
    const bool rope_fast = dec_cache && (4ull * T + (uint64_t(T) + 1) / 2 + 1) <= kDs41RopeSlots;
    if (rope_fast) {
        // Phase 18 (docs/deepseek41/46 term 2): the step's five position entries in one copy and two
        // launches of the same kernel, no host waits -- a table value depends only on its position and
        // inv_freq, so these are the values the five calls below compute (bit-identical)
        const uint32_t HR = RD / 2;
        // slots: main [0, T) the real positions (masks, top-k), [T, 2T) q/kv's; compress [2T, 3T) q/kv's, [3T, 3T + n2) the
        // ratio-2 latents closed by this step (the group's first token: pos + 1 - ratio, model.py:757), [3T + n2, 4T + n2) the
        // ratio-1 latents (one per row). At T = 1 this is exactly the five entries of Phase 18.
        uint32_t n2 = 0;
        for (uint32_t r = 0; r < T; ++r) { card.rope_hpos[r] = pos_t[r]; card.rope_hpos[T + r] = pos_qk[r]; card.rope_hpos[2 * T + r] = pos_qk[r]; }
        for (uint32_t r = 0; r < T; ++r) if ((pos0 + r) % 2 == 1) card.rope_hpos[3 * T + n2++] = int32_t(pos0 + r - 1);
        for (uint32_t r = 0; r < T; ++r) card.rope_hpos[3 * T + n2 + r] = int32_t(pos0 + r);
        const uint32_t nslot = 4 * T + n2;
        q.memcpy(card.rope_pos, card.rope_hpos, nslot * 4);
        ds4_rope_cos_sin(q, card.rope_inv, card.rope_pos, card.rope_cs, card.rope_sn, 2 * T, HR, 1.0f);
        ds4_rope_cos_sin(q, card.rope_inv + HR, card.rope_pos + 2 * T, card.rope_cs + size_t(2) * T * HR, card.rope_sn + size_t(2) * T * HR, nslot - 2 * T, HR, 1.0f);
        d_pos_t = card.rope_pos;
        cos_m  = card.rope_cs + size_t(T) * HR;            sin_m  = card.rope_sn + size_t(T) * HR;
        cos_c  = card.rope_cs + size_t(2) * T * HR;        sin_c  = card.rope_sn + size_t(2) * T * HR;
        cos_g2 = card.rope_cs + size_t(3) * T * HR;        sin_g2 = card.rope_sn + size_t(3) * T * HR;
        cos_g1 = card.rope_cs + (size_t(3) * T + n2) * HR; sin_g1 = card.rope_sn + (size_t(3) * T + n2) * HR;
    } else {
    d_pos_t = rope_tables(rmain, pos_t, cos_m, sin_m);      // the real positions (masks, top-k)
    rope_tables(rmain, pos_qk, cos_m, sin_m);
    rope_tables(rcomp, pos_qk, cos_c, sin_c);
    if (!decode) {
        rope_tables(rcomp, ds4_compress_positions(T / 2, 2, 0), cos_g2, sin_g2);
        rope_tables(rcomp, ds4_compress_positions(T, 1, 0), cos_g1, sin_g1);
    } else {
        // a latent stands for the first token of its group: pos + 1 - ratio (model.py:757); one entry per latent this step emits
        std::vector<int32_t> g2; for (uint32_t r = 0; r < T; ++r) if ((pos0 + r) % 2 == 1) g2.push_back(int32_t(pos0 + r - 1));
        if (g2.empty()) g2.push_back(int32_t(pos0 >= 1 ? pos0 - 1 : 0));   // unused when no group closes
        std::vector<int32_t> g1(T); std::iota(g1.begin(), g1.end(), int32_t(pos0));
        rope_tables(rcomp, g2, cos_g2, sin_g2);
        rope_tables(rcomp, g1, cos_g1, sin_g1);
    }
    }

    // ---- the stream and the previous layer's mix, from the host copy --------------------------
    uint32_t Tl = T_carry;                // query rows in this card's layers (== T until a replay transition)
    uint32_t seg_off = off_carry;         // the rows' offset into the prompt: the ring slot of row r is (pos0 + seg_off + r) % window
    // The sliding mask compares a KEY ROW INDEX with a query position (ds4_sliding_causal_mask), so
    // after a transition the segment's queries need positions relative to the segment's first row --
    // the offset absolute positions saw future keys at small offsets and NO sliding keys past 256
    // (gate 10 finding 1). The absolute positions stay with the indexer's causal threshold.
    auto rel_positions = [&](uint32_t n) { std::vector<int32_t> rel(n); std::iota(rel.begin(), rel.end(), 0);
                                          int32_t* d = SI(n); q.memcpy(d, rel.data(), size_t(n) * 4).wait(); return d; };
    int32_t* d_pos_mask = seg_off ? rel_positions(Tl) : d_pos_t;
    float* hA = S32(size_t(T) * FLAT); float* hB = S32(size_t(T) * FLAT);
    q.memcpy(hA, hh.data(), size_t(Tl) * FLAT * 4).wait();
    float* h = hA; float* h_next = hB;
    float* pre_mix = S32(size_t(T) * HC);
    q.memcpy(pre_mix, hpm.data(), size_t(Tl) * HC * 4).wait();

    // ---- scratch (max sizes) -------------------------------------------------------------------
    const uint32_t TW = std::max({QR, QH, OR, IQ, EF, EIN});
    // compressed entries a query may see: at pos0 > 0 that is what the caches actually hold by the end of this call,
    // NOT the declared capacity -- sizing this scratch by max_tokens is what made a large --ctx cost VRAM it never used.
    // PHASE 24 STEP 3: the indexer score, the block bias, the mask build and the attention are all ROW-INDEPENDENT
    // -- nothing reduces across query rows -- so the compressed segment's work runs in strips of STRIP rows and the
    // three buffers that scale as rows x NC are allocated for a STRIP, not for T. At NC = 250k those three are 3 MB
    // per row (scores + mcomp + mask), i.e. 6.2 GB at T = 2048, which no reserve on this card can hold; at STRIP =
    // 256 they are 768 MB regardless of the chunk size. Re-ordering nothing, this must be bit-identical.
    static const uint32_t strip_w = [] { const char* v = std::getenv("IE_DS41_STRIP"); const long n = v && *v ? std::atol(v) : 256; return n > 0 ? uint32_t(n) : 0u; }();
    const uint32_t STRIP = strip_w ? std::min(strip_w, T) : T;          // 0 disables striping (the kill switch)
    const size_t NCMAX = decode ? std::min<size_t>(cap_pos_, size_t(pos0) + T) : size_t(T);
    const size_t NKVMAX = cont ? size_t(WIN) + size_t(T) + NCMAX : decode ? size_t(WIN) + NCMAX : size_t(T) + NCMAX;
    sycl::half *x16 = S16(size_t(T) * H), *t16 = S16(size_t(T) * TW), *lat16 = S16(size_t(T) * HD);
    float *qr = S32(size_t(T) * QR), *qrn = S32(size_t(T) * QR), *qq = S32(size_t(T) * QH);
    float *kvw = S32(size_t(T) * HD), *kvwn = S32(size_t(T) * HD), *oo = S32(size_t(T) * QH), *oa = S32(size_t(T) * OR);
    float *ao = S32(size_t(T) * H), *xa = S32(size_t(T) * H), *xn = S32(size_t(T) * H), *xf = S32(size_t(T) * H), *xfn = S32(size_t(T) * H);
    float *a_pre = S32(size_t(T) * HC), *a_post = S32(size_t(T) * HC), *a_comb = S32(size_t(T) * HC * HC);
    float *f_pre = S32(size_t(T) * HC), *f_post = S32(size_t(T) * HC), *f_comb = S32(size_t(T) * HC * HC);
    float *ckv_p = S32(size_t(T) * HD), *cg_p = S32(size_t(T) * HD), *pooled = S32(size_t(T) * HD), *latent = S32(size_t(T) * HD);
    float *chunk_kv = S32(size_t(T + 1) * HD), *chunk_gate = S32(size_t(T + 1) * HD);   // the ratio-2 groups closed at decode: a held half + T rows
    float *ik = S32(size_t(T) * IHD), *iq = S32(size_t(T) * IQ), *wp = S32(size_t(T) * IH), *scores = S32(size_t(STRIP) * NCMAX);
    const size_t NKW = cont ? size_t(WIN) + T : size_t(T);               // a continuation's window key count (prev window ++ chunk)
    float *mwin = S32(size_t(T) * NKW), *mcomp = S32(size_t(STRIP) * NCMAX), *mask = S32(size_t(STRIP) * NKVMAX);
    float *wkeys = cont ? S32(NKW * HD) : nullptr;                        // Phase 24: the dense window segment
    float *moe = S32(size_t(T) * H), *shg = S32(size_t(T) * EF), *shu = S32(size_t(T) * EF), *shh = S32(size_t(T) * H);
    float *r_logits = S32(size_t(T) * E), *r_w = S32(size_t(T) * TK); int32_t* r_i = SI(size_t(T) * TK);
    float *ekv = S32(size_t(T) * EOUT), *keyc = S32(size_t(T) * HC * H), *valc = S32(size_t(T) * H), *egate = S32(size_t(T) * HC);
    float* hcs = decode ? S32(size_t(T) * ds41_hc_mixes_scratch_floats(HC)) : nullptr;   // Phase 11: the mixes' decode shape (partials), T rows (P2)
    float* zero_bias = S32(size_t(2) * HD); q.memset(zero_bias, 0, size_t(2) * HD * 4).wait();
    // The two-level candidate top-k's state. `cand_keep` is per QUERY ROW and must survive from the
    // candidate source layer (20) to its consumers (24/28/32/36), so it is full-T; `cand_bscore` is a strip's
    // own block scores. Both are allocated ONLY when the context can actually exceed the candidate pool
    // (2048 blocks x 8 = 16,384 latents), so nothing changes at the lengths that have always worked.
    const uint32_t cand_bs = c.candidate_block_size ? c.candidate_block_size : 1u;
    // IE_DS41_CAND=0 runs the fine top-k over the WHOLE latent cache, i.e. the one-level selection. It is the
    // kill switch and the A/B arm: above 16,384 latents it is much more expensive but it is the exact answer at
    // the consumer layers, which is what makes the two-level stage's approximation measurable rather than assumed.
    static const bool cand_on = [] { const char* v = std::getenv("IE_DS41_CAND"); return !(v && *v && std::string(v) == "0"); }();
    const bool cand_live = cand_on && c.candidate_source_layer >= 0 && c.candidate_topk_blocks &&
                           NCMAX > size_t(c.candidate_topk_blocks) * cand_bs;
    const size_t cand_nbmax = cand_live ? (NCMAX + cand_bs - 1) / cand_bs : 0;
    uint8_t* cand_keep = cand_live ? reinterpret_cast<uint8_t*>(S32((size_t(T) * cand_nbmax + 3) / 4)) : nullptr;
    float* cand_bscore = cand_live ? S32(size_t(STRIP) * cand_nbmax) : nullptr;
    // the index-source's picks, shared with its consumers (one slot, as in SharedAttentionRuntime);
    // a card that starts mid-model inherits the previous card's, with the source caches it read
    int32_t* sh_topk = SI(size_t(T) * c.index_topk);
    // Phase 26: the same picks in ASCENDING index order, for the gathered decode attention. Sorted only when
    // `sh_topk` actually changes -- at an index-source layer or after a cross-card bounce -- which is 8-9 times
    // per token rather than once per compressed layer.
    int32_t* topk_asc = SI(size_t(T) * c.index_topk);
    int32_t* topk_nv  = SI(size_t(T));                 // Phase 33: each row's valid-pick count, from the sort
    bool asc_stale = true;
    struct Src { const float* ckv = nullptr; const float* ik = nullptr; uint32_t nc = 0; } cur;
    if (bounce_nc) {
        q.memcpy(card.imp_ckv, h_imp_ckv.data(), size_t(bounce_nc) * HD * 4);
        q.memcpy(card.imp_ik, h_imp_ik.data(), size_t(bounce_nc) * IHD * 4);
        cur = {card.imp_ckv, card.imp_ik, bounce_nc};
    }
    if (bounce_topk) { q.memcpy(sh_topk, h_sh_topk.data(), h_sh_topk.size() * 4); asc_stale = true; }
    q.wait();
    DS4ExpertWorkspace ws;
    if (!resident_) if (auto e = ds4_expert_ws_alloc(q, H, EF, ws); !e.empty()) return e;

    // =========================================================================================
    for (uint32_t L = card.L0; L < card.L1; ++L) {
        const auto t0 = std::chrono::steady_clock::now();
        // DSpark P1: the drafter reads the mean over hc of the stream entering the target layers (model.py:1264-1266)
        if (const auto& tl = c.dspark_target_layer_ids; capture_main_hidden_ && !tl.empty() && c.dspark_block_size) {
            const auto it = std::find(tl.begin(), tl.end(), L);
            if (it != tl.end()) {
                const uint32_t ti = uint32_t(it - tl.begin()), NT = uint32_t(tl.size());
                std::vector<float> hh_rows(size_t(Tl) * FLAT); q.memcpy(hh_rows.data(), h, hh_rows.size() * 4).wait();
                if (main_hidden_.size() != size_t(T) * NT * H) main_hidden_.assign(size_t(T) * NT * H, 0.f);
                for (uint32_t t = 0; t < Tl; ++t) for (uint32_t d = 0; d < H; ++d) {
                    float acc = 0.f; for (uint32_t cp = 0; cp < HC; ++cp) acc += hh_rows[(size_t(t) * HC + cp) * H + d];
                    main_hidden_[(size_t(T - Tl + t) * NT + ti) * H + d] = acc / float(HC);
                }
            }
        }
        const auto& Lw = m.layers()[L];
        const auto& k = Lw.kind;
        auto& st = card.state[L];
        TR("layer start", L);
        layer_dense_bytes = 0;
        std::vector<void*> lw;                                  // this layer's device weights
        auto keep = [&](void* p) { lw.push_back(p); return p; };
        bool comp_done = false;
        if (replay && L == dec0) {
            // THE REPLAY TRANSITION. This layer's compressor runs over the WHOLE prompt from the
            // encoder output (the global latents and index keys every later layer reads); then the
            // stream and the previous layer's mix are cut down to the last WIN tokens and the rest
            // of this layer and every later one run over that segment (docs/deepseek41/27).
            const Ds41LayerDense* rdt = resident_ ? card.cache.layer(L) : nullptr;
            if (resident_ && !rdt) return "resident: layer " + std::to_string(L) + " is not in the dense cache";
            if (k.compress_ratio != 1) return "layer " + std::to_string(L) + ": bounded replay expects a ratio-1 source at the decoder's first layer";
            if (st.nc + Tl > cap_pos_) return "layer " + std::to_string(L) + ": latent cache full";
            float* n_at_t = rdt ? rdt->n_at : (float*)keep(bf16_32(Lw.attn_norm));
            sycl::half* w_ckv_t = rdt ? rdt->comp_wkv : (sycl::half*)keep(bf16_16(Lw.comp_wkv));
            float* n_c_t = rdt ? rdt->n_c : (float*)keep(bf16_32(Lw.comp_norm));
            sycl::half* w_ik_t = rdt ? rdt->idx_wk : (sycl::half*)keep(bf16_16(Lw.idx_wk)); float* n_ik_t = rdt ? rdt->n_ik : (float*)keep(bf16_32(Lw.idx_k_norm));
            ds41_hc_collapse(q, h, pre_mix, xa, Tl, H, HC);
            ds4_rms_norm(q, xa, n_at_t, xn, Tl, H, c.norm_eps);
            to16(xn, x16, size_t(Tl) * H);
            gemm_nt_f16_onednn(q, x16, w_ckv_t, ckv_p, Tl, HD, H);
            ds4_rms_norm(q, ckv_p, n_c_t, latent, Tl, HD, c.norm_eps);
            float* dst_k = st.idx_k + size_t(st.nc) * IHD; float* dst_c = st.comp_kv + size_t(st.nc) * HD;
            to16(latent, lat16, size_t(Tl) * HD);
            gemm_nt_f16_onednn(q, lat16, w_ik_t, ik, Tl, IHD, HD);
            ds4_rms_norm(q, ik, n_ik_t, dst_k, Tl, IHD, c.norm_eps);
            ds4_rope_apply(q, dst_k, cos_g1, sin_g1, dst_k, Tl, 1, IHD, RD, +1.0f);
            q.memcpy(dst_c, latent, size_t(Tl) * HD * 4);
            ds4_rope_apply(q, dst_c, cos_g1, sin_g1, dst_c, Tl, 1, HD, RD, +1.0f).wait();
            st.nc += Tl; cur = {st.comp_kv, st.idx_k, st.nc}; comp_done = true;
            // the segment: the last WIN rows of the stream and of the mix, moved to the base of fresh
            // buffers; the per-query tables and positions advanced by the same offset
            const uint32_t Tq = WIN, off = Tl - Tq;
            float* h_seg = (h == hA) ? hB : hA;
            q.memcpy(h_seg, h + size_t(off) * FLAT, size_t(Tq) * FLAT * 4);
            float* pm_seg = S32(size_t(Tq) * HC);
            q.memcpy(pm_seg, pre_mix + size_t(off) * HC, size_t(Tq) * HC * 4).wait();
            h = h_seg; h_next = (h_seg == hA) ? hB : hA; pre_mix = pm_seg;
            d_pos_t += off; cos_m += size_t(off) * (RD / 2); sin_m += size_t(off) * (RD / 2); cos_c += size_t(off) * (RD / 2); sin_c += size_t(off) * (RD / 2);
            Tl = Tq; seg_off += off; d_pos_mask = rel_positions(Tq);
        }

        auto ms_from = [](const std::chrono::steady_clock::time_point& t) {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(); };
        // ---- engram, before the block ---------------------------------------------------
        if (k.has_engram) {
            const auto te = std::chrono::steady_clock::now();
            const auto it = std::find(tb_->layer_ids.begin(), tb_->layer_ids.end(), L);
            const size_t li = size_t(it - tb_->layer_ids.begin());
            const Ds41LayerDense* rde = resident_ ? card.cache.layer(L) : nullptr;
            sycl::half* w_e = rde ? rde->engram_wkv : (sycl::half*)keep(dense(Lw.engram_wkv));
            q.memcpy(t16, engram_in16[li].data() + size_t(T - Tl) * EIN, size_t(Tl) * EIN * 2).wait();   // the last Tl rows of T
            ((rde && rde->f8_engram_wkv.w) ? (Tl == 1 ? gemv_fp8_e4m3_f16_tagged(q, t16, rde->f8_engram_wkv.w, rde->f8_engram_wkv.s, ekv, EIN, EOUT, "engram_wkv")
                                              : (decode && Tl <= kDs41MaxDecodeRows) ? gemv_fp8_rows(q, t16, EIN, rde->f8_engram_wkv.w, rde->f8_engram_wkv.s, ekv, Tl, EIN, EOUT)
                                                       : pe("ds41.onednn.engram", gemm_nt_f16_onednn(q, t16, const_cast<sycl::half*>(card.cache.f16(q, rde->f8_engram_wkv)), ekv, Tl, EOUT, EIN)))
                                                    : pe("ds41.onednn.engram", gemm_nt_f16_onednn(q, t16, w_e, ekv, Tl, EOUT, EIN))).wait();
            ie::ps(q, "ds41_engram_split", [&](sycl::handler& hh_) { hh_.parallel_for(sycl::range<1>(size_t(Tl) * HC * H), [=](sycl::id<1> i) {
                const uint32_t t = uint32_t(i[0] / (size_t(HC) * H)), r = uint32_t(i[0] % (size_t(HC) * H));
                keyc[i] = ekv[size_t(t) * EOUT + r];
            }); });
            ie::ps(q, "ds41_engram_split", [&](sycl::handler& hh_) { hh_.parallel_for(sycl::range<1>(size_t(Tl) * H), [=](sycl::id<1> i) {
                const uint32_t t = uint32_t(i[0] / H), d = uint32_t(i[0] % H);
                valc[i] = ekv[size_t(t) * EOUT + size_t(HC) * H + d];
            }); }).wait();
            std::vector<float> qk(size_t(HC) * H);
            {
                const auto* qw = reinterpret_cast<const uint16_t*>(Lw.engram_q.w->data);
                const auto* kw = reinterpret_cast<const uint16_t*>(Lw.engram_k.w->data);
                for (size_t i = 0; i < qk.size(); ++i) qk[i] = bf16f(qw[i]) * bf16f(kw[i]);
            }
            float* dqk = resident_ ? card.cache.layer(L)->engram_qk : (float*)keep(f32(qk.size()));
            if (!resident_) q.memcpy(dqk, qk.data(), qk.size() * 4).wait();
            ds41_engram_gate(q, h, keyc, valc, dqk, h_next, egate, Tl, H, HC, c.norm_eps).wait();
            for (const auto& [r0, rn] : image_runs(Tl))        // token_mask False shuts the gate: the row passes through
                q.memcpy(h_next + size_t(r0) * FLAT, h + size_t(r0) * FLAT, size_t(rn) * FLAT * 4);
            q.wait();
            std::swap(h, h_next);
            stats_[L].engram_ms = ms_from(te);
            if (probe) probe("engram", L, h, size_t(Tl) * FLAT, q);
        }

        // ---- dense weights for this layer: from the resident cache, or uploaded for this forward
        const Ds41LayerDense* rd = resident_ ? card.cache.layer(L) : nullptr;
        // Phase 17 (docs/deepseek41/43): a dense projection from the resident FP8 pair -- the FP8 GEMV at
        // T = 1 (named by its own ie::ps), the fp16 scratch + oneDNN GEMM at T > 1; the fp16 path otherwise
        auto dense_proj = [&](const char* name, const sycl::half* act, sycl::half* w16, const Ds41Fp8Mat* f8, float* out, uint32_t T_, uint32_t N_, uint32_t K_) -> sycl::event {
            if (f8 && f8->w) {
                // profiled by SITE at T = 1: "ds41.onednn.q_b" -> "q_b", so the decode table splits the fp8 GEMV by shape
                if (T_ == 1) return gemv_fp8_e4m3_f16_tagged(q, act, f8->w, f8->s, out, K_, N_, std::strncmp(name, "ds41.onednn.", 12) == 0 ? name + 12 : name);
                if (decode && T_ <= kDs41MaxDecodeRows) return gemv_fp8_rows(q, act, K_, f8->w, f8->s, out, T_, K_, N_);   // P2: row-identical
                w16 = const_cast<sycl::half*>(card.cache.f16(q, *f8));
            }
            if (decode && T_ <= kDs41MaxDecodeRows) return gemv_f16_rows(q, act, K_, w16, out, T_, K_, N_);                 // P2: row-identical
            return pe(name, gemm_nt_f16_onednn(q, act, w16, out, T_, N_, K_));
        };
        if (resident_ && !rd) return "resident: layer " + std::to_string(L) + " is not in the dense cache";
        sycl::half *w_qa = rd ? rd->wq_a : (sycl::half*)keep(dense(Lw.wq_a)), *w_qb = rd ? rd->wq_b : (sycl::half*)keep(dense(Lw.wq_b));
        sycl::half *w_kv = rd ? rd->wkv : (sycl::half*)keep(dense(Lw.wkv)), *w_oa = rd ? rd->wo_a : (sycl::half*)keep(dense(Lw.wo_a));
        sycl::half *w_ob = rd ? rd->wo_b : (sycl::half*)keep(dense(Lw.wo_b));
        sycl::half *s_w1 = rd ? rd->sh_w1 : (sycl::half*)keep(dense(Lw.sh_w1)), *s_w3 = rd ? rd->sh_w3 : (sycl::half*)keep(dense(Lw.sh_w3));
        sycl::half *s_w2 = rd ? rd->sh_w2 : (sycl::half*)keep(dense(Lw.sh_w2));
        float *n_q = rd ? rd->n_q : (float*)keep(bf16_32(Lw.q_norm)), *n_kv = rd ? rd->n_kv : (float*)keep(bf16_32(Lw.kv_norm));
        float *n_at = rd ? rd->n_at : (float*)keep(bf16_32(Lw.attn_norm)), *n_ff = rd ? rd->n_ff : (float*)keep(bf16_32(Lw.ffn_norm));
        float* sinks = rd ? rd->sinks : (float*)keep(f32_up(Lw.attn_sink));
        float *a_fn = rd ? rd->a_fn : (float*)keep(f32_up(Lw.hc_attn_fn)), *a_bs = rd ? rd->a_bs : (float*)keep(f32_up(Lw.hc_attn_base));
        float *a_sc = rd ? rd->a_sc : (float*)keep(f32_up(Lw.hc_attn_scale));
        float *f_fn = rd ? rd->f_fn : (float*)keep(f32_up(Lw.hc_ffn_fn)), *f_bs = rd ? rd->f_bs : (float*)keep(f32_up(Lw.hc_ffn_base));
        float *f_sc = rd ? rd->f_sc : (float*)keep(f32_up(Lw.hc_ffn_scale));
        float* g_w = rd ? rd->g_w : (float*)keep(bf16_32(Lw.gate_w)); float* g_b = rd ? rd->g_b : (float*)keep(f32_up(Lw.gate_bias));

        // ---- hyper-connection, attention site ----------------------------------------------
        const auto ta = std::chrono::steady_clock::now();
        ds41_hc_mixes(q, h, a_fn, a_bs, a_sc, a_pre, a_post, a_comb, Tl, H, HC, c.hc_sinkhorn_iters, c.norm_eps, c.hc_eps, {}, hcs);
        ds41_hc_collapse(q, h, pre_mix, xa, Tl, H, HC);
        W(ds4_rms_norm(q, xa, n_at, xn, Tl, H, c.norm_eps));

        // ---- attention, by kind --------------------------------------------------------------
        const uint32_t R = k.compress_ratio;
        const float* cs_t = R ? cos_c : cos_m; const float* sn_t = R ? sin_c : sin_m;
        to16(xn, x16, size_t(Tl) * H);
        dense_proj("ds41.onednn.q_a", x16, w_qa, rd ? &rd->f8_wq_a : nullptr, qr, Tl, QR, H);
        ds4_rms_norm(q, qr, n_q, qrn, Tl, QR, c.norm_eps);
        to16(qrn, t16, size_t(Tl) * QR);
        dense_proj("ds41.onednn.q_b", t16, w_qb, rd ? &rd->f8_wq_b : nullptr, qq, Tl, QH, QR);
        ds4_rope_apply(q, qq, cs_t, sn_t, qq, Tl, NH, HD, RD, +1.0f);
        dense_proj("ds41.onednn.kv", x16, w_kv, rd ? &rd->f8_wkv : nullptr, kvw, Tl, HD, H);
        ds4_rms_norm(q, kvw, n_kv, kvwn, Tl, HD, c.norm_eps);
        ds4_rope_apply(q, kvwn, cs_t, sn_t, kvwn, Tl, 1, HD, RD, +1.0f);
        // the window ring: the last min(Tl, window) rows land at slot = position % window -- AFTER the attention
        // (DSpark P2, docs/deepseek41/55): a T-row step's rows would evict the T - 1 oldest keys its earlier rows
        // still see, so the attention reads this step's rows from `kvwn` at their slots' column indices (the
        // one-row step's layout, bit for bit) and the old ring elsewhere; the write follows.
        const uint32_t ring_nwr = std::min(Tl, WIN), ring_first = Tl - ring_nwr, ring_base = (pos0 + seg_off + ring_first) % WIN;
        // P3: this step can be rolled back. The snapshot buffers hold kDs41MaxDecodeRows rows (state alloc), because
        // rollback exists for the speculative verify step, which is bounded by that cap -- so the cap is load-bearing
        // HERE TOO, and a Phase 24 continuation (T far above it) must not snapshot: it is a prefill chunk, nothing
        // rolls it back, and copying T rows into an 8-row buffer is an out-of-bounds write (it faulted card 0 once).
        const bool snap = decode && T >= 2 && T <= kDs41MaxDecodeRows;
        if (snap) {
            st.snap_nc = st.nc; st.snap_part_valid = st.part_valid;
            if (st.snap_part_kv && st.part_valid) { q.memcpy(st.snap_part_kv, st.part_kv, size_t(HD) * 4); q.memcpy(st.snap_part_gate, st.part_gate, size_t(HD) * 4); }
        }
        auto ring_write = [&] {
            float* ring = st.win_kv; const uint32_t nwr = ring_nwr, first = ring_first;
            ie::ps(q, "ds41_ring_write", [&](sycl::handler& hh_) { hh_.parallel_for(sycl::range<1>(size_t(nwr) * HD), [=](sycl::id<1> i) {
                const uint32_t r = uint32_t(i[0] / HD), d = uint32_t(i[0] % HD);
                const uint32_t p = pos0 + seg_off + first + r;
                ring[size_t(p % WIN) * HD + d] = kvwn[size_t(first + r) * HD + d];
            }); });
        };
        if (!decode) ds4_sliding_causal_mask(q, d_pos_mask, mwin, Tl, Tl, WIN);
        // Phase 24 step 2: a continuation's window segment is the live previous window in POSITION order followed by
        // this chunk's keys, so the keys a row needs are contiguous and the mask is the ordinary sliding-causal rule.
        // Key index k stands for position (pos0 - nprev + k) across BOTH halves, which is what makes this simple.
        const uint32_t cont_nprev = cont ? std::min(WIN - 1u, pos0) : 0u, cont_nkw = cont_nprev + Tl;
        if (cont) {
            { const float* ring = st.win_kv; float* wk = wkeys; const uint32_t np = cont_nprev, p0 = pos0, Wc = WIN;
              if (np) ie::ps(q, "ds41_cont_wkeys", [&](sycl::handler& hh_) { hh_.parallel_for(sycl::range<1>(size_t(np) * HD), [=](sycl::id<1> i) {
                  const uint32_t k = uint32_t(i[0] / HD), d = uint32_t(i[0] % HD);
                  wk[size_t(k) * HD + d] = ring[size_t((p0 - np + k) % Wc) * HD + d]; }); });
              q.memcpy(wkeys + size_t(cont_nprev) * HD, kvwn, size_t(Tl) * HD * 4); }
            { float* mw = mwin; const uint32_t np = cont_nprev, nkw = cont_nkw, Wc = WIN; const int32_t p0 = int32_t(pos0);
              ie::ps(q, "ds41_cont_wmask", [&](sycl::handler& hh_) { hh_.parallel_for(sycl::range<1>(size_t(Tl) * nkw), [=](sycl::id<1> i) {
                  const uint32_t t = uint32_t(i[0] / nkw), k = uint32_t(i[0] % nkw);
                  const int32_t kp = p0 - int32_t(np) + int32_t(k), qp = p0 + int32_t(t);
                  mw[i] = (kp <= qp && kp >= qp - int32_t(Wc - 1)) ? 0.f : LOWEST; }); }); }
        }
        if (hostsync) q.wait();
        if (probe && !decode && seg_off) probe("swa_mask", L, mwin, size_t(Tl) * Tl, q);   // the segment's [Tl, Tl] sliding mask

        TR("window keys done", L);
        uint32_t NC = 0; sycl::event attn_ev;
        if (R) {
            const float* cs_g = R == 2 ? cos_g2 : cos_g1; const float* sn_g = R == 2 ? sin_g2 : sin_g1;
            if (k.is_kv_source && !comp_done) {
                // compressor -> latent (RoPE-free), then index key, then the RoPE'd cache
                sycl::half* w_ckv = rd ? rd->comp_wkv : (sycl::half*)keep(bf16_16(Lw.comp_wkv));
                float* n_c = rd ? rd->n_c : (float*)keep(bf16_32(Lw.comp_norm));
                if (decode && Tl <= kDs41MaxDecodeRows) gemv_f16_rows(q, x16, H, w_ckv, ckv_p, Tl, H, HD);   // P2: row-identical at decode
                else pe("ds41.onednn.comp_kv", gemm_nt_f16_onednn(q, x16, w_ckv, ckv_p, Tl, HD, H));
                uint32_t n_new = 0;                                   // latents this call emits
                if (!decode) {
                    n_new = Tl / R;
                    if (R > 1) {
                        sycl::half* w_cg = rd ? rd->comp_wgate : (sycl::half*)keep(bf16_16(Lw.comp_wgate));
                        pe("ds41.onednn.comp_gate", gemm_nt_f16_onednn(q, x16, w_cg, cg_p, Tl, HD, H));   // prefill
                        ds4_compress_pool(q, ckv_p, cg_p, zero_bias, nullptr, nullptr, pooled, n_new, R, HD, false);
                        ds4_rms_norm(q, pooled, n_c, latent, n_new, HD, c.norm_eps);
                    } else {
                        ds4_rms_norm(q, ckv_p, n_c, latent, n_new, HD, c.norm_eps);
                    }
                    st.part_valid = false;
                } else if (R == 1) {
                    n_new = Tl;                                            // one latent per row (P2: T rows)
                    ds4_rms_norm(q, ckv_p, n_c, latent, Tl, HD, c.norm_eps);
                } else {
                    // ratio 2: an even position opens a group (its row and gate held), the odd one closes it. A T-row
                    // step: the held half (if this step starts odd) followed by the T rows, pooled pairwise; a trailing
                    // even row is held. Each group is pooled from the same two rows a 1-row step would pool: bit-identical.
                    sycl::half* w_cg = rd ? rd->comp_wgate : (sycl::half*)keep(bf16_16(Lw.comp_wgate));
                    if (Tl <= kDs41MaxDecodeRows) gemv_f16_rows(q, x16, H, w_cg, cg_p, Tl, H, HD);   // P2: row-identical at decode
                    else pe("ds41.onednn.comp_gate", gemm_nt_f16_onednn(q, x16, w_cg, cg_p, Tl, HD, H));
                    const bool starts_odd = (pos0 + 1) % R == 0;
                    if (starts_odd != st.part_valid) return "layer " + std::to_string(L) + (starts_odd ? ": closing a ratio-2 group with no open half" : ": opening a ratio-2 group over a held half");
                    if (snap) { q.memcpy(st.snap_ckv, ckv_p, size_t(Tl) * HD * 4); q.memcpy(st.snap_cg, cg_p, size_t(Tl) * HD * 4); }   // P3: the rows, for the half after a rollback
                    uint32_t len = 0;
                    if (st.part_valid) { q.memcpy(chunk_kv, st.part_kv, size_t(HD) * 4); q.memcpy(chunk_gate, st.part_gate, size_t(HD) * 4); len = 1; }
                    q.memcpy(chunk_kv + size_t(len) * HD, ckv_p, size_t(Tl) * HD * 4); q.memcpy(chunk_gate + size_t(len) * HD, cg_p, size_t(Tl) * HD * 4); len += Tl;
                    const uint32_t n_pairs = len / R;
                    if (n_pairs) {
                        ds4_compress_pool(q, chunk_kv, chunk_gate, zero_bias, nullptr, nullptr, pooled, n_pairs, R, HD, false);
                        ds4_rms_norm(q, pooled, n_c, latent, n_pairs, HD, c.norm_eps);
                    }
                    if (len % R) { q.memcpy(st.part_kv, chunk_kv + size_t(len - 1) * HD, size_t(HD) * 4); q.memcpy(st.part_gate, chunk_gate + size_t(len - 1) * HD, size_t(HD) * 4).wait(); st.part_valid = true; }
                    else { q.wait(); st.part_valid = false; }
                    n_new = n_pairs;
                }
                if (n_new) {
                    if (st.nc + n_new > cap_pos_) return "layer " + std::to_string(L) + ": latent cache full";
                    sycl::half* w_ik = rd ? rd->idx_wk : (sycl::half*)keep(bf16_16(Lw.idx_wk)); float* n_ik = rd ? rd->n_ik : (float*)keep(bf16_32(Lw.idx_k_norm));
                    float* dst_k = st.idx_k + size_t(st.nc) * IHD; float* dst_c = st.comp_kv + size_t(st.nc) * HD;
                    to16(latent, lat16, size_t(n_new) * HD);
                    if (decode && n_new <= kDs41MaxDecodeRows) gemv_f16_rows(q, lat16, HD, w_ik, ik, n_new, HD, IHD);   // P2: row-identical at decode
                    else pe("ds41.onednn.idx_k", gemm_nt_f16_onednn(q, lat16, w_ik, ik, n_new, IHD, HD));
                    ds4_rms_norm(q, ik, n_ik, dst_k, n_new, IHD, c.norm_eps);
                    ds4_rope_apply(q, dst_k, cs_g, sn_g, dst_k, n_new, 1, IHD, RD, +1.0f);
                    q.memcpy(dst_c, latent, size_t(n_new) * HD * 4);
                    W(ds4_rope_apply(q, dst_c, cs_g, sn_g, dst_c, n_new, 1, HD, RD, +1.0f));
                    st.nc += n_new;
                }
                cur = {st.comp_kv, st.idx_k, st.nc};
            }
            if (!cur.ckv) return "layer " + std::to_string(L) + ": consumer with no source cache";
            NC = cur.nc;
            if (k.is_index_source) {
                sycl::half* w_iqb = rd ? rd->idx_wq_b : (sycl::half*)keep(dense(Lw.idx_wq_b)); sycl::half* w_iwp = rd ? rd->idx_weights : (sycl::half*)keep(bf16_16(Lw.idx_weights));
                dense_proj("ds41.onednn.idx_q", t16, w_iqb, rd ? &rd->f8_idx_wq_b : nullptr, iq, Tl, IQ, QR);          // t16 == qrn (fp16)
                ds4_rope_apply(q, iq, cos_c, sin_c, iq, Tl, IH, IHD, RD, +1.0f);
                if (decode && Tl <= kDs41MaxDecodeRows) gemv_f16_rows(q, x16, H, w_iwp, wp, Tl, H, IH);   // P2: row-identical at decode
                else pe("ds41.onednn.idx_w", gemm_nt_f16_onednn(q, x16, w_iwp, wp, Tl, IH, H));
                sh_topk_w = std::min(c.index_topk, NC);
            }
            // THE STRIP LOOP (Phase 24 step 3). Everything in here is per-query-row: the indexer's score is a
            // dot product for one row against every latent, its top-k is that row's own selection, the bias and
            // the mask are that row's, and the attention's softmax reduces over keys, never over rows. So a strip
            // computes exactly what the full call computed for the same rows -- bit for bit -- while `scores`,
            // `mcomp` and `mask` shrink from rows x NC to STRIP x NC. `iq` / `wp` / `sh_topk` stay full-size and
            // their GEMMs stay outside, both because they do not scale with NC and because a oneDNN matmul is free
            // to block its K reduction differently at a different M, which striping them could disturb.
            for (uint32_t so = 0; so < Tl; so += STRIP) {
                const uint32_t sT = std::min(STRIP, Tl - so);
                // The XMX attention converts a fp32 `b` segment (the latent cache) into its fp16 staging, and
                // that staging is a property of the LAYER, not of the strip -- `xmx_kv_prepared` exists to say
                // so. Without it every strip re-staged the whole NC axis, 8x the work at the default width for
                // bytes that had not changed. This is the one place striping added cost rather than saving it.
                const bool kv_staged = so > 0;
                if (k.is_index_source) {
                    // Phase 45 (docs/deepseek41/85): at a layer that consumes the candidate pool, only the keys inside the
                    // row's kept blocks are scored; the rest get the mask's value directly -- bit-identical to scoring
                    // every key and masking (ds41_candidate_apply), ~NC / 16,384 fewer dot products. IE_DS41_CAND_SKIP=0
                    // scores every key again.
                    static const bool cand_skip_on = [] { const char* e = std::getenv("IE_DS41_CAND_SKIP"); return !(e && *e && std::string(e) == "0"); }();
                    const bool cand_skip = cand_skip_on && cand_live && NC > size_t(c.candidate_topk_blocks) * cand_bs &&
                                           int32_t(L) > c.candidate_source_layer && IH >= 16 && cand_bs % 2 == 0;
                    if (cand_skip)
                        ds4_indexer_score_candidates(q, iq + size_t(so) * IQ, cur.ik, wp + size_t(so) * IH, scores, sT, IH, IHD, NC, 1.0f / std::sqrt(float(IHD)), 1.0f / std::sqrt(float(IH)),
                                                     cand_keep + size_t(so) * ((size_t(NC) + cand_bs - 1) / cand_bs), cand_bs,
                                                     -std::numeric_limits<float>::infinity());   // ds41_candidate_apply's value (kDs41CandNegInf)
                    else
                    ds4_indexer_score(q, iq + size_t(so) * IQ, cur.ik, wp + size_t(so) * IH, scores, sT, IH, IHD, NC, 1.0f / std::sqrt(float(IHD)), 1.0f / std::sqrt(float(IH)));
                    // THE TWO-LEVEL CANDIDATE TOP-K (model.py:569-610). The source layer publishes the kept
                    // blocks from its own scores and then runs its own fine top-k UNMASKED (the reference falls
                    // through, it does not mask itself); the later index sources mask to those blocks first.
                    // Layers 2/8/14 sit before the source and do neither -- the reference does not restrict them,
                    // and their fine top-k is the radix shape, which is O(n_keys).
                    if (cand_live && NC > size_t(c.candidate_topk_blocks) * cand_bs) {
                        const size_t cnb = (size_t(NC) + cand_bs - 1) / cand_bs;
                        if (int32_t(L) == c.candidate_source_layer) {
                            cand_noop_ = false;
                            ds41_candidate_block_scores(q, scores, d_pos_t + so, cand_bscore, sT, NC, cand_bs, R);
                            ds41_candidate_select(q, cand_bscore, cand_keep + size_t(so) * cnb, sT, NC, cand_bs, c.candidate_topk_blocks).wait();
                        } else if (int32_t(L) > c.candidate_source_layer && !cand_skip) {
                            ds41_candidate_apply(q, cand_keep + size_t(so) * cnb, scores, sT, NC, cand_bs).wait();
                        }
                    }
                    W(ds4_indexer_topk(q, scores, d_pos_t + so, sh_topk + size_t(so) * c.index_topk, sT, NC, c.index_topk, R));
                    asc_stale = true;
                }
                // Phase 44 (docs/deepseek41/84): IE_DS41_CONT_GATHER=1 -- a continuation strip attends over each row's
                // live keys gathered (its <= 128 window keys and its picks), not the dense segment ++ every latent, so
                // neither the [strip, NC] bias nor the dense mask is built. DEFAULT ON after its gate (PPL +0.0005 nats, t = 0.62;
                // cont 15/15; 32k prefill 278 -> 311 tok/s); IE_DS41_CONT_GATHER=0 restores the contiguous walk.
                static const bool cont_gather = [] { const char* e = std::getenv("IE_DS41_CONT_GATHER"); return !(e && *e && std::string(e) == "0"); }();
                const bool gat = cont && cont_gather && sh_topk_w && ds4_attention_xmx_eligible(q.get_device(), sT, NH, HD);
                if (!gat) ds4_block_bias_topk(q, sh_topk + size_t(so) * c.index_topk, mcomp, sT, NC, sh_topk_w);
                if (gat) {
                    attn_ev = ds41_attention_cont_xmx(q, qq + size_t(so) * QH, wkeys, cont_nkw, cont_nprev, WIN, mwin, so,
                                                      cur.ckv, NC, sh_topk + size_t(so) * c.index_topk, sh_topk_w,
                                                      sinks, oo + size_t(so) * QH, sT, NH, HD, 1.0f / std::sqrt(float(HD)));
                    if (hostsync) attn_ev.wait();
                } else if (cont) {   // Phase 24 step 2: the dense window segment ++ the accumulated latents
                    const uint32_t NKV = cont_nkw + NC, nkw = cont_nkw, NCc = NC, r0 = so;
                    W(ie::ps(q, "ds41_cont_mask", [&](sycl::handler& hh_) { hh_.parallel_for(sycl::range<1>(size_t(sT) * NKV), [=](sycl::id<1> i) {
                        const uint32_t t = uint32_t(i[0] / NKV), kk = uint32_t(i[0] % NKV);
                        mask[i] = kk < nkw ? mwin[size_t(r0 + t) * nkw + kk] : mcomp[size_t(t) * NCc + (kk - nkw)]; }); }));
                    Ds4KvSegs segs; segs.a = wkeys; segs.n_a = cont_nkw; segs.b = cur.ckv; segs.n_b = NC; segs.b_f16 = false;
                    attn_ev = ds4_attention_segs(q, qq + size_t(so) * QH, segs, mask, sinks, oo + size_t(so) * QH, sT, NH, HD, 1.0f / std::sqrt(float(HD)), kv_staged);
                } else if (!decode) {
                    const uint32_t NKV = Tl + NC, NCc = NC, Tc = Tl, r0 = so;
                    q.parallel_for(sycl::range<1>(size_t(sT) * NKV), [=](sycl::id<1> i) {
                        const uint32_t t = uint32_t(i[0] / NKV), kk = uint32_t(i[0] % NKV);
                        mask[i] = kk < Tc ? mwin[size_t(r0 + t) * Tc + kk] : mcomp[size_t(t) * NCc + (kk - Tc)];
                    }); if (hostsync) q.wait();
                    Ds4KvSegs segs; segs.a = kvwn; segs.n_a = Tl; segs.b = cur.ckv; segs.n_b = NC; segs.b_f16 = false;
                    attn_ev = ds4_attention_segs(q, qq + size_t(so) * QH, segs, mask, sinks, oo + size_t(so) * QH, sT, NH, HD, 1.0f / std::sqrt(float(HD)), kv_staged);
                } else {
                    // the whole ring as one segment, [sT, NKV]: after this step's writes slot kk holds the latest position
                    // q = kk + WIN * floor((p_last - kk) / WIN) with p_last = pos0 + Tl - 1; it is open for row r iff
                    // 0 <= q <= pos0 + r (P2: per-row ring causality; at T = 1 this is `kk <= pos` while the ring fills
                    // and every slot once it has wrapped); every stored key carries its own RoPE
                    // slot kk of the ring holds, for row t: this step's row j = (kk - base) mod WIN when j < T and j <= t (position pos0 + j);
                    // otherwise the old ring's position q = kk + WIN * floor((pos0 - 1 - kk) / WIN) -- open iff it exists and lies in
                    // row t's window [pos0 + t - (WIN - 1), pos0 + t] (at T = 1 this is the ring-filling rule and every slot once wrapped)
                    // `ts` is the row's index in this strip, `t` its index in the call -- the causal rules are all in `t`.
                    const uint32_t NKV = WIN + NC; const int32_t pos = int32_t(pos0); const uint32_t NCc = NC, WINc = WIN, Tc = Tl, base = ring_base, r0 = so; const bool nocausal = nocausal_diag_;
                    ie::ps(q, "ds41_decode_mask", [&](sycl::handler& hh_) { hh_.parallel_for(sycl::range<1>(size_t(sT) * NKV), [=](sycl::id<1> i) {
                        const uint32_t ts = uint32_t(i[0] / NKV), kk = uint32_t(i[0] % NKV), t = r0 + ts;
                        if (kk < WINc) {
                            const uint32_t j = (kk + WINc - base) % WINc; int32_t qpos;
                            if (j < Tc) qpos = (j <= t || nocausal) ? pos + int32_t(j) : pos + int32_t(j) - int32_t(WINc);
                            else { const int32_t d = pos - 1 - int32_t(kk); qpos = int32_t(kk) + int32_t(WINc) * (d >= 0 ? d / int32_t(WINc) : -((-d + int32_t(WINc) - 1) / int32_t(WINc))); }
                            mask[i] = (qpos >= 0 && qpos >= pos + int32_t(t) - int32_t(WINc - 1)) ? 0.f : LOWEST;
                        } else mask[i] = mcomp[size_t(ts) * NCc + (kk - WINc)];
                    }); }); if (hostsync) q.wait();
                    Ds4KvSegs segs; segs.a = st.win_kv; segs.n_a = WIN; segs.b = cur.ckv; segs.n_b = NC; segs.b_f16 = false;
                    segs.a_new = kvwn + size_t(ring_first) * HD; segs.a_new_base = ring_base; segs.a_new_n = ring_nwr; segs.a_new_nocausal = nocausal_diag_;
                    // PHASE 26: hand the split kernel the indexer's picks so segment b costs O(index_topk)
                    // instead of O(NC). `mcomp` is still built and still passed -- the gathered path reads it to
                    // test each pick's bias, so a pick that is not live is dropped exactly as the dense scan
                    // dropped it -- but the kernel no longer WALKS the NC columns to discover they are masked.
                    // The sort READS sh_topk_w entries per row -- the count ds4_indexer_topk actually emitted,
                    // min(index_topk, NC), NOT index_topk, or stale picks from an earlier layer would leak in
                    // (the first attempt did that: 28 -> 23/5 on the decode gate, worst layer 9.4e-1) -- and
                    // WRITES the whole pitch, valid picks ascending then -1. n_picks is therefore the PITCH,
                    // index_topk: the kernel addresses row t at b_picks + t * n_picks, and handing it the
                    // count instead put row 1 and up at the wrong offset whenever NC < index_topk (multi test:
                    // T-1 rows wrong on the 12-token prompt, none on the 2,048-token one).
                    // IE_DS41_GATHER=0: the dense mask scan (the pre-Phase-26 path) -- the A/B arm for row-identity.
                    static const bool gather_on = [] { const char* e = std::getenv("IE_DS41_GATHER"); return !(e && *e && std::string(e) == "0"); }();
                    if (gather_on && sh_topk_w) {
                        if (asc_stale) { W(ds41_sort_picks_asc(q, sh_topk, topk_asc, topk_nv, Tl, sh_topk_w, c.index_topk)); asc_stale = false; }
                        segs.b_picks = topk_asc + size_t(so) * c.index_topk; segs.n_picks = c.index_topk; segs.b_pick_n = topk_nv + so;
                    }
                    // Phase 34 (docs/deepseek41/74): the XMX gathered flash-decoding path -- needs the picks; IE_DS41_ATTN_XMX=0
                    // restores the fp32 split kernel exactly.
                    if (segs.b_picks && ds41_attention_decode_xmx_eligible(q.get_device(), sT, NH, HD))
                        attn_ev = ds41_attention_decode_xmx(q, qq + size_t(so) * QH, segs, mask, sinks, oo + size_t(so) * QH, sT, NH, HD, 1.0f / std::sqrt(float(HD)));
                    else
                    attn_ev = ds4_attention_segs(q, qq + size_t(so) * QH, segs, mask, sinks, oo + size_t(so) * QH, sT, NH, HD, 1.0f / std::sqrt(float(HD)), kv_staged);
                }
            }
        } else if (cont) {   // Phase 24 step 2: window-only layer, the same dense segment
            static const bool cont_gather = [] { const char* e = std::getenv("IE_DS41_CONT_GATHER"); return !(e && *e && std::string(e) == "0"); }();
            if (cont_gather && ds4_attention_xmx_eligible(q.get_device(), Tl, NH, HD))   // Phase 44: each row's <= 128 window keys
                attn_ev = ds41_attention_cont_xmx(q, qq, wkeys, cont_nkw, cont_nprev, WIN, mwin, 0, nullptr, 0, nullptr, 0,
                                                  sinks, oo, Tl, NH, HD, 1.0f / std::sqrt(float(HD)));
            else {
            Ds4KvSegs segs0; segs0.a = wkeys; segs0.n_a = cont_nkw; segs0.b = nullptr; segs0.n_b = 0; segs0.b_f16 = false;
            attn_ev = ds4_attention_segs(q, qq, segs0, mwin, sinks, oo, Tl, NH, HD, 1.0f / std::sqrt(float(HD)));
            }
        } else if (!decode) {
            attn_ev = ds4_attention(q, qq, kvwn, mwin, sinks, oo, Tl, NH, HD, Tl, 1.0f / std::sqrt(float(HD)));
        } else {
            const int32_t pos = int32_t(pos0); const uint32_t WINc = WIN, Tc = Tl, base = ring_base; const bool nocausal = nocausal_diag_;   // [Tl, WIN], the same ring rule (P2)
            ie::ps(q, "ds41_decode_mask", [&](sycl::handler& hh_) { hh_.parallel_for(sycl::range<1>(size_t(Tl) * WIN), [=](sycl::id<1> i) {
                const uint32_t t = uint32_t(i[0] / WINc), kk = uint32_t(i[0] % WINc);
                const uint32_t j = (kk + WINc - base) % WINc; int32_t qpos;
                if (j < Tc) qpos = (j <= t || nocausal) ? pos + int32_t(j) : pos + int32_t(j) - int32_t(WINc);
                else { const int32_t d = pos - 1 - int32_t(kk); qpos = int32_t(kk) + int32_t(WINc) * (d >= 0 ? d / int32_t(WINc) : -((-d + int32_t(WINc) - 1) / int32_t(WINc))); }
                mask[i] = (qpos >= 0 && qpos >= pos + int32_t(t) - int32_t(WINc - 1)) ? 0.f : LOWEST;
            }); }); if (hostsync) q.wait();
            Ds4KvSegs segs0; segs0.a = st.win_kv; segs0.n_a = WIN; segs0.b = nullptr; segs0.n_b = 0; segs0.b_f16 = false;
            segs0.a_new = kvwn + size_t(ring_first) * HD; segs0.a_new_base = ring_base; segs0.a_new_n = ring_nwr; segs0.a_new_nocausal = nocausal_diag_;
            if (ds41_attention_decode_xmx_eligible(q.get_device(), Tl, NH, HD))   // Phase 34: the window-only layers too (2 blocks)
                attn_ev = ds41_attention_decode_xmx(q, qq, segs0, mask, sinks, oo, Tl, NH, HD, 1.0f / std::sqrt(float(HD)));
            else
            attn_ev = ds4_attention_segs(q, qq, segs0, mask, sinks, oo, Tl, NH, HD, 1.0f / std::sqrt(float(HD)));
        }
        TR("attention done", L);
        if (snap) {                                          // P3: the slots this write evicts, in row order
            float* sr = st.snap_ring; const float* ring = st.win_kv; const uint32_t nwr = ring_nwr, base = ring_base;
            ie::ps(q, "ds41_ring_snapshot", [&](sycl::handler& hh_) { hh_.parallel_for(sycl::range<1>(size_t(nwr) * HD), [=](sycl::id<1> i) {
                const uint32_t r = uint32_t(i[0] / HD), d = uint32_t(i[0] % HD); sr[size_t(r) * HD + d] = ring[size_t((base + r) % WIN) * HD + d]; }); });
        }
        ring_write();                                        // the step's rows land in the ring for the next step
        ds4_rope_apply(q, oo, cs_t, sn_t, oo, Tl, NH, HD, RD, -1.0f);
        to16(oo, t16, size_t(Tl) * QH);
        // Phase 32: o_a in FP8 at decode -- the checkpoint's own FP8 bytes with the LUT decode, the same weight values the
        // fp16 copy held, 1.34 GB/token instead of 2.68. T == 1 the grouped GEMV, 2..8 rows the grouped rows kernel (P2
        // row-identity kept). Prefill keeps the fp16 block-diagonal bmm on the cache's on-demand fp16 view.
        if (decode && Tl <= kDs41MaxDecodeRows) {
            if (rd && rd->f8_wo_a.w) {
                if (Tl == 1) gemv_fp8_e4m3_f16_grouped(q, t16, rd->f8_wo_a.w, rd->f8_wo_a.s, oa, IPG, OR, OPG, "o_a");
                else gemv_fp8_rows_grouped(q, t16, QH, rd->f8_wo_a.w, rd->f8_wo_a.s, oa, Tl, IPG, OR, OPG);
            } else gemv_f16_rows(q, t16, QH, w_oa, oa, Tl, IPG, OR, OPG);   // P2: the block-diagonal o_a, row-identical at decode
        } else {
            if (!w_oa && rd && rd->f8_wo_a.w) w_oa = const_cast<sycl::half*>(card.cache.f16(q, rd->f8_wo_a));
            pe("ds41.onednn.o_a", gemm_bmm_nt_f16_onednn(q, t16, w_oa, oa, Tl, G, IPG, OPG));
        }
        to16(oa, t16, size_t(Tl) * OR);
        W(dense_proj("ds41.onednn.o_b", t16, w_ob, rd ? &rd->f8_wo_b : nullptr, ao, Tl, H, OR));

        W(ds4_hc_mix(q, h, a_post, a_comb, ao, h_next, Tl, H, HC));
        std::swap(h, h_next);
        stats_[L].attn_ms = ms_from(ta);
        if (q.has_property<sycl::property::queue::enable_profiling>()) {
            // the attention kernel's own GPU clock: did it run slowly, or start late?
            const auto sub = attn_ev.get_profiling_info<sycl::info::event_profiling::command_submit>();
            const auto st0 = attn_ev.get_profiling_info<sycl::info::event_profiling::command_start>();
            const auto en0 = attn_ev.get_profiling_info<sycl::info::event_profiling::command_end>();
            stats_[L].attn_kernel_queue_ms = double(st0 - sub) / 1e6; stats_[L].attn_kernel_exec_ms = double(en0 - st0) / 1e6;
        }

        // ---- hyper-connection, FFN site; then the MoE ---------------------------------------
        const auto tf = std::chrono::steady_clock::now();
        ds41_hc_mixes(q, h, f_fn, f_bs, f_sc, f_pre, f_post, f_comb, Tl, H, HC, c.hc_sinkhorn_iters, c.norm_eps, c.hc_eps, {}, hcs);
        ds41_hc_collapse(q, h, a_pre, xf, Tl, H, HC);
        W(ds4_rms_norm(q, xf, n_ff, xfn, Tl, H, c.norm_eps));

        W(ds4_router_topk(q, xfn, g_w, g_b, r_logits, r_w, r_i, Tl, H, E, TK, c.route_scale));   // the readback below waits
        q.memcpy(h_idx.data(), r_i, size_t(Tl) * TK * 4); q.memcpy(h_w.data(), r_w, size_t(Tl) * TK * 4).wait();   // Tl rows: the rest of h_idx is stale
        if (const auto runs = image_runs(Tl); !runs.empty()) {
            // image-span tokens select by their own bias (training's noaux_tc_for_vl); the weights stay the unbiased
            // scores'. Second pass with that bias, its rows taken at the image positions, device and host kept equal.
            if (!Lw.gate_bias_vl.w) return "forward: the prompt holds image positions and layer " + std::to_string(L) + " has no ffn.gate.bias_vl";
            float* g_bv = (rd && rd->g_b_vl) ? rd->g_b_vl : (float*)keep(f32_up(Lw.gate_bias_vl));
            W(ds4_router_topk(q, xfn, g_w, g_bv, r_logits, r_w, r_i, Tl, H, E, TK, c.route_scale));
            std::vector<int32_t> vi(size_t(Tl) * TK); std::vector<float> vw(size_t(Tl) * TK);
            q.memcpy(vi.data(), r_i, vi.size() * 4); q.memcpy(vw.data(), r_w, vw.size() * 4).wait();
            for (const auto& [r0, rn] : runs) {
                std::copy(vi.begin() + size_t(r0) * TK, vi.begin() + size_t(r0 + rn) * TK, h_idx.begin() + size_t(r0) * TK);
                std::copy(vw.begin() + size_t(r0) * TK, vw.begin() + size_t(r0 + rn) * TK, h_w.begin() + size_t(r0) * TK);
            }
            q.memcpy(r_i, h_idx.data(), size_t(Tl) * TK * 4); q.memcpy(r_w, h_w.data(), size_t(Tl) * TK * 4).wait();
        }
        stats_[L].ffn_pre_ms = ms_from(tf);
        if (expected && L < expected->size()) {
            uint32_t diff = 0;
            for (uint32_t t = 0; t < Tl; ++t) {
                std::vector<int32_t> a(h_idx.begin() + t * TK, h_idx.begin() + (t + 1) * TK);
                std::vector<int32_t> b((*expected)[L].begin() + t * TK, (*expected)[L].begin() + (t + 1) * TK);
                std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
                if (a != b) {
                    ++diff;
                    if (expected_gap && L < expected_gap->size())
                        stats_[L].flip_gap_max = std::max(stats_[L].flip_gap_max, (*expected_gap)[L][t]);
                }
            }
            stats_[L].routing_diff = diff;
            if (force_routing) {
                // Golden INDICES, but weights derived from the ENGINE's own router scores, so the
                // weight arithmetic (unbiased gather, /(sum+1e-20), x route_scale) is exercised at
                // every layer even in forced mode (gate finding 5). Where the engine's selection
                // already matched the golden's, these must equal the engine's own weights.
                // ds4_router_topk's `logits` are the RAW x @ W^Tl (its header); the score is
                // sqrt(softplus(logit)), computed here exactly as the kernel does.
                std::vector<float> lg(size_t(Tl) * E);
                q.memcpy(lg.data(), r_logits, lg.size() * 4).wait();
                for (auto& v : lg) v = std::sqrt(std::log1p(std::exp(-std::fabs(v))) + std::max(v, 0.f));
                for (uint32_t t = 0; t < Tl; ++t) {
                    // the self-check is meaningful only where the engine's SET equals the golden's:
                    // the weight is normalised over the selected set's sum, so a token whose
                    // selection differs (the near-tie flips) has a different denominator by design
                    std::vector<int32_t> a(h_idx.begin() + t * TK, h_idx.begin() + (t + 1) * TK);
                    std::vector<int32_t> b((*expected)[L].begin() + t * TK, (*expected)[L].begin() + (t + 1) * TK);
                    std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
                    const bool same_set = a == b;
                    double sum = 0;
                    for (uint32_t k2 = 0; k2 < TK; ++k2) sum += lg[size_t(t) * E + (*expected)[L][size_t(t) * TK + k2]];
                    for (uint32_t k2 = 0; k2 < TK; ++k2) {
                        const int32_t e = (*expected)[L][size_t(t) * TK + k2];
                        const float w = float(lg[size_t(t) * E + e] / (sum + 1e-20) * c.route_scale);
                        if (same_set)
                            for (uint32_t k3 = 0; k3 < TK; ++k3)
                                if (h_idx[size_t(t) * TK + k3] == e && std::fabs(h_w[size_t(t) * TK + k3] - w) > 1e-5f)
                                    return "layer " + std::to_string(L) + ": engine router weight " + std::to_string(h_w[size_t(t) * TK + k3]) +
                                           " != weight rebuilt from its own scores " + std::to_string(w) + " (expert " + std::to_string(e) + ")";
                    }
                    for (uint32_t k2 = 0; k2 < TK; ++k2) {
                        const int32_t e = (*expected)[L][size_t(t) * TK + k2];
                        h_idx[size_t(t) * TK + k2] = e;
                        h_w[size_t(t) * TK + k2] = float(lg[size_t(t) * E + e] / (sum + 1e-20) * c.route_scale);
                    }
                }
            }
        }
        if (!force_routing && L < profile_.size())
            for (size_t i = 0; i < size_t(Tl) * TK; ++i) if (const int32_t e = h_idx[i]; e >= 0 && uint32_t(e) < E) ++profile_[L][e];
        // Phase 18 (docs/deepseek41/46 term 1): the shared expert's projections into `shh`. At decode
        // they are enqueued AHEAD of the tier call (after the EP staging's wait, so the remote card's
        // start is not delayed): they read xfn and scratch the tier never touches, and they run while
        // the first group's DMAs are in flight, where the queue idled. The add and the hc mix stay
        // after the tier, so the sum is formed in the same order -- bit-identical.
        // ON by default (docs/47: bit-identical by the dumps, -4.9 ms/token measured); `IE_DS41_SHARED_EARLY=0`
        // restores the sequential order (the kill switch).
        static const bool shared_early_on = [] { const char* v = std::getenv("IE_DS41_SHARED_EARLY"); return !(v && *v && std::string(v) == "0"); }();
        const bool shared_early = decode && resident_ && shared_early_on;
        auto shared_expert = [&] {
            to16(xfn, x16, size_t(Tl) * H);
            dense_proj("ds41.onednn.sh_gate", x16, s_w1, rd ? &rd->f8_sh_w1 : nullptr, shg, Tl, EF, H);
            dense_proj("ds41.onednn.sh_up", x16, s_w3, rd ? &rd->f8_sh_w3 : nullptr, shu, Tl, EF, H);
            {
                const float Lim = c.swiglu_limit;
                ie::ps(q, "ds41_shared_swiglu", [&](sycl::handler& hh_) { hh_.parallel_for(sycl::range<1>(size_t(Tl) * EF), [=](sycl::id<1> i) {
                    float gg = shg[i], uu = shu[i];
                    if (Lim > 0.f) { gg = sycl::fmin(gg, Lim); uu = sycl::fmax(sycl::fmin(uu, Lim), -Lim); }
                    shg[i] = (gg / (1.0f + sycl::exp(-gg))) * uu;
                }); });
            }
            to16(shg, t16, size_t(Tl) * EF);
            dense_proj("ds41.onednn.sh_down", t16, s_w2, rd ? &rd->f8_sh_w2 : nullptr, shh, Tl, H, EF);
        };
        if (resident_) {
            const auto tm = std::chrono::steady_clock::now();
            // expert parallel (docs/deepseek41/38), bit-exact with the whole tier: every other card
            // runs its tier's share of this layer on a helper thread -- x staged out through the two
            // cards' pinned buffers -- in EXPORT mode (its packed fp16 rows, no scatter); this card
            // runs its own tier with an IMPORT hook that, right before the one scatter, joins the
            // helpers and drops their rows into the same packed positions of its own workspace.
            std::vector<std::thread> helpers; std::vector<Card*> posted; std::vector<std::string> herr(cards_.size()); double xfer_ms = 0;
            const size_t nxh = size_t(Tl) * H;
            auto wait_posted = [&] { for (Card* pc : posted) { std::unique_lock<std::mutex> lk(pc->ep_mu); pc->ep_cv.wait(lk, [&] { return pc->ep_done; }); } posted.clear(); };
            EpHelpersDone helpers_done{&helpers, &card.tier, wait_posted};   // gate 14 finding 7: joined and the hook cleared on every exit, exceptions included
            if (ep_) {
                const auto tx = std::chrono::steady_clock::now();
                q.memcpy(card.ep_hx, xfn, nxh * 4).wait();
                xfer_ms += ms_from(tx); stats_[L].ep_t_stage = ms_from(tm);
                static const bool ep_seq = std::getenv("IE_DS41_EP_SEQ") != nullptr;   // diagnosis: the remote tier inline, no concurrency
                for (size_t r = 0; r < cards_.size(); ++r) {
                    Card* rc = cards_[r].get(); if (rc == &card) continue;
                    auto remote = [&, rc, r, nxh] {
                        stats_[L].ep_t_spawn = std::max(stats_[L].ep_t_spawn, ms_from(tm));
                        kmp_set_blocktime(0);
                        std::memcpy(rc->ep_hx, card.ep_hx, nxh * 4);
                        rc->q->memcpy(rc->ep_x, rc->ep_hx, nxh * 4);
                        Ds41ExpertTier& rt = ep_ == 2 ? rc->ctl_tier : rc->tier;
                        const auto tr = std::chrono::steady_clock::now(); stats_[L].ep_t_remote_in = std::max(stats_[L].ep_t_remote_in, ms_from(tm));
                        rt.set_ep_export(true);
                        herr[r] = rt.moe(*rc->q, L, rc->ep_x, h_idx.data(), h_w.data(), Tl, nullptr, c.swiglu_limit);
                        rt.set_ep_export(false);
                        rc->ep_ms = ms_from(tr); stats_[L].ep_t_remote_out = std::max(stats_[L].ep_t_remote_out, ms_from(tm));
                        if (!herr[r].empty()) return;
                        // gather its rows contiguously and bring them to its pinned buffer
                        const auto& rows = rt.ep_rows(); const uint32_t nr = uint32_t(rows.size());
                        if (nr) {
                            rc->q->memcpy(rc->ep_rows, rows.data(), size_t(nr) * 4);
                            const sycl::half* yp = rt.packed_yp(); sycl::half* ypc = rc->ep_ypc; const int32_t* rws = rc->ep_rows; const uint32_t HH = H;
                            rc->q->parallel_for(sycl::range<2>(nr, HH), [=](sycl::id<2> id) { ypc[id[0] * HH + id[1]] = yp[size_t(rws[id[0]]) * HH + id[1]]; });
                            rc->q->memcpy(rc->ep_hyc, rc->ep_ypc, size_t(nr) * HH * 2).wait();
                        }
                        stats_[L].ep_t_rows = std::max(stats_[L].ep_t_rows, ms_from(tm));
                    };
                    if (ep_seq) remote();
                    else if (rc->ep_thread.joinable()) { { std::lock_guard<std::mutex> lk(rc->ep_mu); rc->ep_job = remote; rc->ep_pending = true; rc->ep_done = false; } rc->ep_cv.notify_all(); posted.push_back(rc); }
                    else helpers.emplace_back(remote);
                }
                card.tier.set_ep_import([&](sycl::queue& iq, sycl::half* yp, uint32_t /*TK*/) -> std::string {
                    stats_[L].ep_t_import_in = ms_from(tm);
                    for (auto& t : helpers) t.join(); helpers.clear(); wait_posted();
                    for (size_t r = 0; r < cards_.size(); ++r) if (!herr[r].empty()) return "card " + std::to_string(r) + ": " + herr[r];
                    const auto tx = std::chrono::steady_clock::now();
                    for (size_t r = 0; r < cards_.size(); ++r) {
                        Card* rc = cards_[r].get(); if (rc == &card) continue;
                        const auto& rows = (ep_ == 2 ? rc->ctl_tier : rc->tier).ep_rows(); const uint32_t nr = uint32_t(rows.size());
                        if (!nr) continue;
                        std::memcpy(card.ep_hyc, rc->ep_hyc, size_t(nr) * H * 2);
                        iq.memcpy(card.ep_ypc, card.ep_hyc, size_t(nr) * H * 2);
                        iq.memcpy(card.ep_rows, rows.data(), size_t(nr) * 4);
                        const sycl::half* ypc = card.ep_ypc; const int32_t* rws = card.ep_rows; const uint32_t HH = H;
                        iq.parallel_for(sycl::range<2>(nr, HH), [=](sycl::id<2> id) { yp[size_t(rws[id[0]]) * HH + id[1]] = ypc[id[0] * HH + id[1]]; });
                        iq.wait();                                                   // the landing buffers are reused per remote card
                    }
                    xfer_ms += ms_from(tx); stats_[L].ep_t_import_out = ms_from(tm);
                    return {};
                });
            }
            if (probe) probe("xfn", L, xfn, nxh, q);                                  // the MoE input, for A/B dumps
            // Phase 18 term 1: enqueued from inside the tier right after group 0's fetch is issued (enqueued
            // BEFORE the call, the tier's own prep enqueues blocked the host behind these kernels: +4.5 ms/token)
            if (shared_early) card.tier.set_pre_groups(shared_expert);
            if (ep_) stats_[L].ep_t_owner_in = ms_from(tm);
            // docs/48 P0: the routing per decode step and layer, each selected expert with its tier (0 static / 1 pinned / 2 mmap)
            static FILE* rdump = [] { const char* v = std::getenv("IE_DS41_DUMP_ROUTING"); return v && *v ? std::fopen(v, "a") : nullptr; }();
            if (rdump && decode) { std::fprintf(rdump, "%u %u", pos0, L); for (uint32_t k2 = 0; k2 < TK; ++k2) std::fprintf(rdump, " %d:%u", h_idx[k2], expert_tier(L, uint32_t(h_idx[k2]))); std::fputc('\n', rdump); }
            const std::string own = card.tier.moe(q, L, xfn, h_idx.data(), h_w.data(), Tl, moe, c.swiglu_limit);
            card.tier.set_ep_import({}); card.tier.set_pre_groups({});
            for (auto& t : helpers) t.join(); wait_posted();
            if (!own.empty()) return "layer " + std::to_string(L) + " tier: " + own;
            for (size_t r = 0; r < cards_.size(); ++r) if (!herr[r].empty()) return "layer " + std::to_string(L) + " tier (card " + std::to_string(r) + "): " + herr[r];
            stats_[L].moe_call_ms = ms_from(tm); stats_[L].ep_xfer_ms = xfer_ms;
            if (probe) probe("moe", L, moe, size_t(Tl) * H, q);   // the routed MoE output, after the expert-parallel merge
            const auto& ts = card.tier.last();
            stats_[L].experts_static = ts.experts_static; stats_[L].experts_pinned = ts.experts_pinned;
            stats_[L].experts_mmap = ts.experts_mmap; stats_[L].bytes_mmap = ts.bytes_mmap_to_vram; stats_[L].bytes_pinned = ts.bytes_pinned_to_vram; stats_[L].experts_stream_hit = ts.stream_hits; stats_[L].experts_cpu = ts.experts_cpu; stats_[L].moe_cpu_ms = ts.ms_cpu; stats_[L].moe_cpu_work_ms = ts.ms_cpu_work; stats_[L].moe_ms = ts.ms;
            stats_[L].moe_prep_ms = ts.ms_prep; stats_[L].moe_mmap_ms = ts.ms_mmap; stats_[L].moe_mmap_pack_ms = ts.ms_mmap_pack; stats_[L].moe_groups_ms = ts.ms_groups;
            stats_[L].moe_mmap_read_ms = ts.ms_mmap_read; stats_[L].moe_mmap_permute_ms = ts.ms_mmap_permute; stats_[L].moe_mmap_group_ms = ts.ms_mmap_group; stats_[L].moe_tail_ms = ts.ms_tail; stats_[L].moe_spawn_ms = ts.ms_spawn; stats_[L].moe_join_ms = ts.ms_join;
            stats_[L].experts_uploaded = ts.experts_mmap; stats_[L].expert_bytes = ts.bytes_mmap_to_vram; stats_[L].experts_mmap_file = ts.experts_mmap_file;
            if (ep_) {   // the other cards' tiers: counts summed in, walls kept apart, the owner/remote split recorded
                stats_[L].experts_owner = ts.experts_static + ts.experts_pinned + ts.experts_mmap; stats_[L].miss_owner = ts.experts_pinned - ts.stream_hits;
                for (const auto& cp : cards_) {
                    if (cp.get() == &card) continue;
                    const auto& rs = (ep_ == 2 ? cp->ctl_tier : cp->tier).last();
                    stats_[L].experts_static += rs.experts_static; stats_[L].experts_pinned += rs.experts_pinned; stats_[L].experts_mmap += rs.experts_mmap;
                    stats_[L].bytes_mmap += rs.bytes_mmap_to_vram; stats_[L].bytes_pinned += rs.bytes_pinned_to_vram; stats_[L].experts_stream_hit += rs.stream_hits;
                    stats_[L].experts_uploaded += rs.experts_mmap; stats_[L].expert_bytes += rs.bytes_mmap_to_vram; stats_[L].experts_mmap_file += rs.experts_mmap_file;
                    stats_[L].experts_remote += rs.experts_static + rs.experts_pinned + rs.experts_mmap; stats_[L].miss_remote += rs.experts_pinned - rs.stream_hits;
                    stats_[L].experts_cpu += rs.experts_cpu; stats_[L].moe_cpu_ms = std::max(stats_[L].moe_cpu_ms, rs.ms_cpu); stats_[L].moe_cpu_work_ms += rs.ms_cpu_work;   // Phase 20: the remote tier's CPU leg counted too
                    stats_[L].moe_remote_ms = std::max(stats_[L].moe_remote_ms, cp->ep_ms);
                }
            }
        } else {
        // union of the routed experts -> a compact bank, indices remapped to bank slots
        std::vector<int32_t> uniq(h_idx); std::sort(uniq.begin(), uniq.end()); uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        std::unordered_map<int32_t, int32_t> slot; slot.reserve(uniq.size() * 2);
        std::vector<Ds41Tensor> s1, s3, s2;
        for (size_t i = 0; i < uniq.size(); ++i) {
            slot[uniq[i]] = int32_t(i);
            s1.push_back(Lw.exp_w1[uniq[i]]); s3.push_back(Lw.exp_w3[uniq[i]]); s2.push_back(Lw.exp_w2[uniq[i]]);
        }
        std::vector<int32_t> local(h_idx.size());
        for (size_t i = 0; i < h_idx.size(); ++i) local[i] = slot[h_idx[i]];
        DS4ExpertBank bg, bu, bd;
        const uint32_t EU = uint32_t(uniq.size());
        if (auto e = ds41_expert_bank_upload(q, s1, H, EF, EU, bg); !e.empty()) return "layer " + std::to_string(L) + " w1: " + e;
        if (auto e = ds41_expert_bank_upload(q, s3, H, EF, EU, bu); !e.empty()) return "layer " + std::to_string(L) + " w3: " + e;
        if (auto e = ds41_expert_bank_upload(q, s2, EF, H, EU, bd); !e.empty()) return "layer " + std::to_string(L) + " w2: " + e;
        stats_[L].experts_uploaded = EU;
        stats_[L].expert_bytes = uint64_t(EU) * (bg.mx_qs_stride + bg.mx_e_stride + bu.mx_qs_stride + bu.mx_e_stride + bd.mx_qs_stride + bd.mx_e_stride);
        if (auto e = ds4_experts_forward(q, bg, bu, bd, xfn, local.data(), h_w.data(), moe, Tl, H, EF, TK, c.swiglu_limit, ws, true); !e.empty())
            return "layer " + std::to_string(L) + " experts: " + e;
        ds4_expert_bank_free(q, bg); ds4_expert_bank_free(q, bu); ds4_expert_bank_free(q, bd);
        }   // streaming MoE

        // shared expert
        const auto tsh = std::chrono::steady_clock::now();
        if (!shared_early) shared_expert();
        ie::ps(q, "ds41_shared_add", [&](sycl::handler& hh_) { hh_.parallel_for(sycl::range<1>(size_t(Tl) * H), [=](sycl::id<1> i) { moe[i] += shh[i]; }); });
        W(ds4_hc_mix(q, h, f_post, f_comb, moe, h_next, Tl, H, HC));
        std::swap(h, h_next);
        stats_[L].shared_ms = ms_from(tsh);
        std::swap(pre_mix, f_pre);                       // this layer's ffn_pre feeds the next layer's
                                                          // collapse; the old pre_mix buffer becomes
                                                          // the scratch the next hc_mixes writes into

        for (void* p : lw) sycl::free(p, q);
        stats_[L].dense_bytes = layer_dense_bytes;
        stats_[L].ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (probe) probe("layer", L, h, size_t(Tl) * FLAT, q);
    }
    // =========================================================================================

    if (card.L1 == c.n_layers) {
        // ---- collapse with the last ffn_pre, norm, head ----------------------------------------
        const auto t_head = std::chrono::steady_clock::now();
        ds41_hc_collapse(q, h, pre_mix, xa, Tl, H, HC);
        layer_dense_bytes = 0;                                     // the head and norm count too (finding 6)
        const float* n_out = resident_ ? card.cache.final_norm() : bf16_32(m.final_norm);
        W(ds4_rms_norm(q, xa, n_out, xn, Tl, H, c.norm_eps));
        if (probe) probe("final", 0, xa, size_t(Tl) * H, q);
        {
            // Phase 54: the FP8 head when the cache holds one (decode rows and a last-row prefill read its bytes; a full-logits
            // prefill dequantises it into the cache's fp16 scratch)
            const Ds41Fp8Mat* h8 = resident_ && card.cache.head_fp8().w ? &card.cache.head_fp8() : nullptr;
            sycl::half* w_head = h8 ? nullptr : resident_ ? const_cast<sycl::half*>(card.cache.head()) : bf16_16(m.lm_head);   // [V, H]
            const uint32_t TL = logits_last_only_ ? 1u : Tl, t0r = Tl - TL;     // the rows the caller asked for
            float* lg = f32(size_t(TL) * V);
            to16(xn, x16, size_t(Tl) * H);
            if (h8 && TL == 1) W(gemv_fp8_e4m3_f16_tagged(q, x16 + size_t(t0r) * H, h8->w, h8->s, lg, H, V, "ds41_head_fp8"));
            else if (h8 && decode && TL <= kDs41MaxDecodeRows) W(gemv_fp8_rows(q, x16 + size_t(t0r) * H, H, h8->w, h8->s, lg, TL, H, V));
            else if (h8) W(pe("ds41.onednn.head", gemm_nt_f16_onednn(q, x16 + size_t(t0r) * H, card.cache.f16(q, *h8), lg, TL, V, H)));
            else if (decode && TL <= kDs41MaxDecodeRows) W(gemv_f16_rows(q, x16 + size_t(t0r) * H, H, w_head, lg, TL, H, V));   // P2: row-identical at decode
            else W(pe("ds41.onednn.head", gemm_nt_f16_onednn(q, x16 + size_t(t0r) * H, w_head, lg, TL, V, H)));
            logits_out.resize(size_t(TL) * V);
            q.memcpy(logits_out.data(), lg, logits_out.size() * 4).wait();
            if (!resident_ && w_head) sycl::free(w_head, q);
            sycl::free(lg, q);
        }
        head_bytes_ = layer_dense_bytes;
        head_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_head).count();
        if (!resident_) sycl::free(const_cast<float*>(n_out), q);
    } else {
        // ---- hand the state to the next card through the host --------------------------------
        q.memcpy(hh.data(), h, size_t(Tl) * FLAT * 4); q.memcpy(hpm.data(), pre_mix, size_t(Tl) * HC * 4); T_carry = Tl; off_carry = seg_off;
        // Phase 18 (docs/deepseek41/46 term 2): the bounce is dead when the next card's first layer is a kv
        // source AND an index source -- it overwrites `cur` and `sh_topk` before any consumer reads them
        // (this split: layer 20; the dumps at 0.0 with the copy gone are the proof). Decode only.
        const auto& nk = m.layers()[card.L1].kind;
        const bool dead_bounce = dec_cache && nk.is_kv_source && nk.is_index_source;
        bounce_nc = dead_bounce ? 0u : cur.nc; bounce_topk = false;
        if (cur.ckv && cur.nc && !dead_bounce) {
            h_imp_ckv.resize(size_t(cur.nc) * HD); h_imp_ik.resize(size_t(cur.nc) * IHD);
            q.memcpy(h_imp_ckv.data(), cur.ckv, h_imp_ckv.size() * 4); q.memcpy(h_imp_ik.data(), cur.ik, h_imp_ik.size() * 4);
            q.memcpy(h_sh_topk.data(), sh_topk, h_sh_topk.size() * 4); bounce_topk = true;
        }
        q.wait_and_throw();
    }
    if (!resident_) ds4_expert_ws_free(q, ws);
    for (void* p : scratch) sycl::free(p, q);
    // Phase 46: a prefill chunk's end is a prefix-cache checkpoint for this card (its rings copied after its launches)
    if (pc_on_ && T > kDs41MaxDecodeRows)
        for (size_t ci = 0; ci < cards_.size(); ++ci) if (cards_[ci].get() == &card) { pc_capture(ci, pos0 + T); break; }
    return {};
    };   // run_card

    for (size_t ci = card_first; ci < card_last; ++ci)
        if (auto e = run_card(*cards_[ci]); !e.empty()) return e;
    ids_guard.ok = true;
    if (!last_stage) {    // Phase 43: hand the boundary to the next stage; the positions are committed by the last one
        StageCarry& out = *stage->out;
        out.hh = std::move(hh); out.hpm = std::move(hpm); out.imp_ckv = std::move(h_imp_ckv); out.imp_ik = std::move(h_imp_ik); out.sh_topk = std::move(h_sh_topk);
        out.T_carry = T_carry; out.off_carry = off_carry; out.bounce_nc = bounce_nc; out.bounce_topk = bounce_topk;
        return {};
    }
    n_pos_ = pos0 + T;
    snap_pos0_ = pos0; snap_T_ = T; snap_valid_ = decode && T >= 2;   // P3
    return {};
}

// Phase 43 (docs/deepseek41/83): the cards as pipeline stages over a chunked prefill. Stage 0 (card 0) runs on this
// thread; stage c on its own thread, fed through a one-slot mailbox, so card c-1 is at most one finished chunk ahead of
// card c. Every stage calls forward_impl on its card with the chunk's (ids, T, pos0) -- the same launches in the same
// order as forward(), on state no other stage touches (per-card KV, tiers, scratch; stats_ per layer).
std::string Ds41Forward::forward_pipelined(const int32_t* ids, const std::vector<std::pair<uint32_t, uint32_t>>& chunks, std::vector<float>& logits_out) {
    if (!pipelined_admissible()) return "forward_pipelined: needs resident mode on 2+ cards, no expert parallel, no drafter capture";
    if (chunks.empty()) return {};
    const size_t NS = cards_.size(), NK = chunks.size();
    uint32_t p = chunks[0].first;
    if (p != 0 && p != n_pos_) return "forward_pipelined: the first chunk starts at " + std::to_string(p) + " but the state holds " + std::to_string(n_pos_) + " positions";
    for (const auto& [pos0, T] : chunks) {
        if (pos0 != p) return "forward_pipelined: chunk at " + std::to_string(pos0) + " is not contiguous (expected " + std::to_string(p) + ")";
        if (pos0 > 0 && T <= kDs41MaxDecodeRows) return "forward_pipelined: a chunk of " + std::to_string(T) + " rows at " + std::to_string(pos0) + " is a decode step; run it with forward()";
        p += T;
    }
    for (auto& cp : cards_) if (auto e = ensure_state(*cp); !e.empty()) return e;
    stats_.assign(m_->config().n_layers, {});
    struct Mailbox { std::mutex mu; std::condition_variable cv; bool full = false; StageCarry carry; };
    std::vector<std::unique_ptr<Mailbox>> box(NS);                    // box[s]: stage s's input (s >= 1)
    for (auto& b : box) b = std::make_unique<Mailbox>();
    std::atomic<bool> abort{false};
    std::vector<std::string> err(NS);
    auto wake_all = [&] { for (auto& b : box) { std::lock_guard<std::mutex> lk(b->mu); } for (auto& b : box) b->cv.notify_all(); };
    // IE_DS41_PIPE_TRACE=1: each stage's busy time and its waits (for the previous stage's carry / for the next stage to
    // take its own), printed once -- the slower stage is the one that sets the chunk rate
    static const bool trace = [] { const char* v = std::getenv("IE_DS41_PIPE_TRACE"); return v && *v && std::string(v) == "1"; }();
    std::vector<double> busy_ms(NS, 0), in_ms(NS, 0), out_ms(NS, 0);
    auto ms_from = [](const std::chrono::steady_clock::time_point& t) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(); };
    auto run_stage = [&](size_t s) {
        StageCarry in, out;
        std::vector<float> scratch_logits;
        for (size_t k = 0; k < NK; ++k) {
            if (s > 0) {
                const auto tw = std::chrono::steady_clock::now();
                Mailbox& b = *box[s];
                std::unique_lock<std::mutex> lk(b.mu);
                b.cv.wait(lk, [&] { return b.full || abort.load(); });
                if (!b.full) return;                                       // aborted upstream
                in = std::move(b.carry); b.full = false;
                lk.unlock(); b.cv.notify_all();
                in_ms[s] += ms_from(tw);
            }
            Stage st{s, s + 1, s > 0 ? &in : nullptr, s + 1 < NS ? &out : nullptr};
            const auto [pos0, T] = chunks[k];
            const auto tb = std::chrono::steady_clock::now();
            std::string e;
            try { e = forward_impl(ids + pos0, T, pos0, s + 1 < NS ? scratch_logits : logits_out, {}, nullptr, nullptr, false, nullptr, &st); }
            catch (const std::exception& ex) { e = std::string("threw: ") + ex.what(); }   // a stage thread must not terminate the process
            catch (...) { e = "threw a non-std exception"; }
            busy_ms[s] += ms_from(tb);
            if (e.empty() && abort.load()) return;
            if (!e.empty()) { err[s] = "chunk at " + std::to_string(pos0) + " (" + std::to_string(T) + " tokens), card " + std::to_string(s) + ": " + e; abort = true; wake_all(); return; }
            if (s + 1 < NS) {
                const auto tw = std::chrono::steady_clock::now();
                Mailbox& b = *box[s + 1];
                std::unique_lock<std::mutex> lk(b.mu);
                b.cv.wait(lk, [&] { return !b.full || abort.load(); });
                if (abort.load()) return;
                b.carry = std::move(out); b.full = true;
                lk.unlock(); b.cv.notify_all();
                out_ms[s] += ms_from(tw);
            }
        }
    };
    std::vector<std::thread> th;
    for (size_t s = 1; s < NS; ++s) th.emplace_back(run_stage, s);
    run_stage(0);
    for (auto& t : th) t.join();
    if (trace)
        for (size_t s = 0; s < NS; ++s)
            std::fprintf(stderr, "[ds41 pipe] stage %zu (card %zu): busy %.2f s, waiting for its input %.2f s, for the next stage %.2f s, over %zu chunks\n",
                         s, s, busy_ms[s] / 1e3, in_ms[s] / 1e3, out_ms[s] / 1e3, NK);
    for (const auto& e : err) if (!e.empty()) { if (pc_on_) reset_state(); return "forward_pipelined: " + e; }   // Phase 46: see forward()
    return {};
}

std::string Ds41Forward::rollback_to(uint32_t n_pos) {
    if (!snap_valid_) return "rollback_to: no multi-row step to roll back";
    if (n_pos < snap_pos0_ || n_pos > snap_pos0_ + snap_T_) return "rollback_to: " + std::to_string(n_pos) + " is outside the last step [" + std::to_string(snap_pos0_) + ", " + std::to_string(snap_pos0_ + snap_T_) + "]";
    const auto& c = m_->config(); const uint32_t HD = c.head_dim, WIN = c.window_size, T = snap_T_, L = n_pos - snap_pos0_, pos0 = snap_pos0_;
    for (auto& cp : cards_) {
        Card& card = *cp; sycl::queue& q = *card.q;
        for (uint32_t Ly = card.L0; Ly < card.L1; ++Ly) {
            auto& st = card.state[Ly]; const auto& k = m_->layers()[Ly].kind;
            if (!noring_diag_)                                                     // the evicted keys back into the slots of the rows not accepted
                for (uint32_t r = L; r < T; ++r) q.memcpy(st.win_kv + size_t((pos0 + r) % WIN) * HD, st.snap_ring + size_t(r) * HD, size_t(HD) * 4);
            if (k.is_kv_source) {
                if (k.compress_ratio <= 1) { st.nc = st.snap_nc + L; st.part_valid = false; }
                else {
                    const uint32_t before = st.snap_part_valid ? 1u : 0u, acc = before + L;
                    st.nc = st.snap_nc + acc / k.compress_ratio;
                    if (acc % k.compress_ratio) {                                    // an odd accepted sequence: its last element is the open half
                        if (L >= 1) { q.memcpy(st.part_kv, st.snap_ckv + size_t(L - 1) * HD, size_t(HD) * 4); q.memcpy(st.part_gate, st.snap_cg + size_t(L - 1) * HD, size_t(HD) * 4); }
                        else        { q.memcpy(st.part_kv, st.snap_part_kv, size_t(HD) * 4); q.memcpy(st.part_gate, st.snap_part_gate, size_t(HD) * 4); }
                        st.part_valid = true;
                    } else st.part_valid = false;
                }
            }
        }
        q.wait_and_throw();
    }
    all_ids_.resize(n_pos); n_pos_ = n_pos; snap_valid_ = false;
    return {};
}

}  // namespace ie
