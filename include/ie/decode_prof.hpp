#pragma once
// P4 B22: per-token decode breakdown for the Qwen layer-split models (27B qwen35 split, 35B qwen35moe split), without
// unitrace (it cannot query event timestamps on this driver stack).
//
// IE_DECODE_PROF=1 turns on two layers:
//  (A) host phase clock, always cheap: timestamps at the step's EXISTING sync points only (no added waits), so the wall
//      split is the production timeline: setup (positions/ids/embedding), each card's segment (submit + run, ends at
//      the card's drain), the card hand-off copy, head + logits bounce, and the engine time between two forward calls.
//  (B) device op clock, when the queues also have profiling (IE_QUEUE_PROFILING=1): each op's returned kernel event is
//      tagged with a category; at the step's end the device timestamps give each category's kernel time and the idle
//      gap before each op. Ops that submit several kernels (attention = append + partial + combine) return only the
//      LAST kernel's event: their earlier kernels fall in that op's "gap-before" column.
// Only T == 1 steps are counted. Every IE_DECODE_PROF_WINDOW (default 32) tokens it prints the per-token means, resets.
// Off (the default): one predictable branch per call site; no events are held, nothing is waited.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ie {

class DecodeProf {
public:
    enum Cat : uint8_t { kGemv, kGemvSmall, kQuant, kAttn, kAttnPre, kDnRec, kDnConv, kDnMisc, kNorm, kElem, kMoe, kRouter, kShexpGate,
                         kCopy, kNCat };
    enum Ph : uint8_t { kSetup, kCard0, kHand, kCard1, kHead, kOutside, kNPh };

    explicit DecodeProf(const char* tag) : tag_(tag) {
        const char* e = std::getenv("IE_DECODE_PROF");
        on_ = e && *e == '1';
        if (const char* w = std::getenv("IE_DECODE_PROF_WINDOW")) window_ = uint32_t(std::max(1, std::atoi(w)));
        dev_clock_ = on_ && std::getenv("IE_QUEUE_PROFILING");
    }
    bool on() const { return on_; }
    bool active() const { return active_; }

    // Start of a forward call. Only T == 1 is measured; a T > 1 call breaks the "outside" chain.
    void begin(uint32_t T, uint32_t pos) {
        if (!on_) return;
        active_ = (T == 1);
        const auto now = clk::now();
        if (!active_) { have_ret_ = false; return; }
        if (have_ret_) cur_ph_[kOutside] = ms(last_ret_, now);
        else cur_ph_[kOutside] = -1.0;
        t_ = now;
        pos_ = pos;
        if (n_ == 0) pos_first_ = pos;
        for (auto& v : ev_) v.clear();
    }
    // Close host phase p at "now" (called right after an existing sync point).
    void phase(Ph p) {
        if (!active_) return;
        const auto now = clk::now();
        cur_ph_[p] += ms(t_, now);
        t_ = now;
    }
    // Tag an op's kernel event with a category (device clock layer only).
    void tag(uint32_t dev, const sycl::event& e, Cat c) {
        if (!active_ || !dev_clock_ || dev >= 2) return;
        ev_[dev].push_back({e, c});
    }
    // The host waited on dev's queue here: the idle time before dev's next tagged op is a sync bubble, not a launch gap.
    void sync(uint32_t dev) {
        if (!active_ || dev >= 2 || ev_[dev].empty()) return;
        ev_[dev].push_back({sycl::event{}, kNCat});
    }
    // End of a forward call (all queues idle: the logits are on the host).
    void end() {
        if (!active_) return;
        const auto now = clk::now();
        last_ret_ = now;
        have_ret_ = true;
        active_ = false;
        for (int p = 0; p < kNPh; ++p) {
            if (p == kOutside) { if (cur_ph_[p] >= 0) { sum_ph_[p] += cur_ph_[p]; ++n_out_; } }
            else sum_ph_[p] += cur_ph_[p];
            cur_ph_[p] = 0;
        }
        for (uint32_t d = 0; d < 2; ++d) harvest(d);
        pos_last_ = pos_;
        if (++n_ == window_) print();
    }

private:
    using clk = std::chrono::steady_clock;
    static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
    struct Ev { sycl::event e; Cat c; };

    void harvest(uint32_t d) {
        auto& v = ev_[d];
        if (v.empty()) return;
        uint64_t prev_end = 0, first = 0;
        bool after_sync = false, any = false;
        for (size_t i = 0; i < v.size(); ++i) {
            if (v[i].c == kNCat) { after_sync = true; continue; }
            const uint64_t s = v[i].e.get_profiling_info<sycl::info::event_profiling::command_start>();
            const uint64_t f = v[i].e.get_profiling_info<sycl::info::event_profiling::command_end>();
            if (!any) first = s;
            busy_[d][v[i].c] += double(f > s ? f - s : 0) * 1e-6;
            ++cnt_[d][v[i].c];
            if (any && s > prev_end) (after_sync ? sync_gap_[d] : gap_[d][v[i].c]) += double(s - prev_end) * 1e-6;
            any = true; after_sync = false;
            if (f > prev_end) prev_end = f;
        }
        span_[d] += double(prev_end > first ? prev_end - first : 0) * 1e-6;
        ++dev_tok_[d];
        v.clear();
    }

    void print() {
        static const char* cn[kNCat] = {"gemv", "gemv_small(ssm alpha/beta)", "quant_act", "attn(last krn)", "attn_pre(qk norm,rope,split,gate)",
                                        "dn_recurrence", "dn_conv", "dn_misc", "norm(rms/resid+rms)", "elem(swiglu,resid)",
                                        "moe_experts", "moe_router", "shexp_gate", "copy"};
        static const char* pn[kNPh] = {"setup(pos,ids,embed)", "card0 segment", "hand-off", "card1 segment",
                                       "head+logits D2H", "engine outside fwd"};
        const double n = double(n_);
        double wall = 0;
        for (int p = 0; p < kNPh; ++p) if (p != kOutside) wall += sum_ph_[p];
        std::fprintf(stderr, "\n[decode-prof %s] %u tokens at pos %u..%u (per-token ms)\n", tag_, n_, pos_first_, pos_last_);
        for (int p = 0; p < kNPh; ++p) {
            const double v = p == kOutside ? (n_out_ ? sum_ph_[p] / n_out_ : 0) : sum_ph_[p] / n;
            std::fprintf(stderr, "  host %-24s %8.3f\n", pn[p], v);
        }
        const double outside = n_out_ ? sum_ph_[kOutside] / n_out_ : 0;
        std::fprintf(stderr, "  host forward wall        %8.3f   step (fwd+outside) %8.3f = %.2f tok/s\n", wall / n,
                     wall / n + outside, 1000.0 / (wall / n + outside));
        for (uint32_t d = 0; d < 2; ++d) {
            if (!dev_tok_[d]) continue;
            const double m = double(dev_tok_[d]);
            double b = 0, g = 0;
            for (int c = 0; c < kNCat; ++c) { b += busy_[d][c]; g += gap_[d][c]; }
            std::fprintf(stderr, "  dev%u span %8.3f  busy %8.3f  gaps %8.3f  sync-bubbles %7.3f  (ops/token %.0f)\n", d,
                         span_[d] / m, b / m, g / m, sync_gap_[d] / m, [&] { double k = 0; for (int c = 0; c < kNCat; ++c) k += cnt_[d][c]; return k / m; }());
            for (int c = 0; c < kNCat; ++c)
                if (cnt_[d][c])
                    std::fprintf(stderr, "    %-34s busy %8.3f  gap-before %7.3f  n %5.0f\n", cn[c], busy_[d][c] / m,
                                 gap_[d][c] / m, double(cnt_[d][c]) / m);
        }
        std::fflush(stderr);
        n_ = 0; n_out_ = 0;
        for (auto& v : sum_ph_) v = 0;
        for (uint32_t d = 0; d < 2; ++d) {
            span_[d] = 0; sync_gap_[d] = 0; dev_tok_[d] = 0;
            for (int c = 0; c < kNCat; ++c) { busy_[d][c] = gap_[d][c] = 0; cnt_[d][c] = 0; }
        }
    }

    const char* tag_;
    bool on_ = false, dev_clock_ = false, active_ = false, have_ret_ = false;
    uint32_t window_ = 32, n_ = 0, n_out_ = 0, pos_ = 0, pos_first_ = 0, pos_last_ = 0;
    clk::time_point t_{}, last_ret_{};
    double cur_ph_[kNPh] = {}, sum_ph_[kNPh] = {};
    std::vector<Ev> ev_[2];
    double busy_[2][kNCat] = {}, gap_[2][kNCat] = {}, span_[2] = {}, sync_gap_[2] = {};
    uint64_t cnt_[2][kNCat] = {}, dev_tok_[2] = {};
};

}  // namespace ie
