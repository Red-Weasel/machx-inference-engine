// include/ie/vitals.hpp -- per-request "vital signs" (docs/mimo26/IE_VITALS.md): a passive accumulator the generator
// fills after each committed token and the server reads into the SSE stream when the request asked for it
// ("ie_vitals": true). It never feeds back into sampling: the sampler only WRITES the numbers it reports.
#pragma once
#include <algorithm>
#include <cstdint>

namespace ie {

struct VitalsWindow {
    // A token whose sampling distribution has an entropy above this many nats counts as "high". Diagnostic and
    // uncalibrated: 2 nats is the entropy of ~7.4 equally likely choices.
    static constexpr double kHighEntropyNats = 2.0;

    bool active = false;   // set by a generator that fills the window (MiMo-V2.6); others leave it false

    // Since the last reset_window() (one SSE report).
    uint32_t n = 0;                 // committed tokens
    uint32_t n_H = 0;               // of them, with an entropy (a sampled row at temperature > 0)
    uint32_t n_hi = 0;              // of them, entropy > kHighEntropyNats
    double   sum_H = 0, max_H = 0;  // nats
    double   min_margin = 1;        // lowest p(top-1) - p(top-2)
    uint32_t draft_offered = 0, draft_accepted = 0;   // DFlash draft tokens verified / kept

    // The whole request.
    uint32_t tot_n = 0, tot_n_H = 0, tot_n_hi = 0, tot_offered = 0, tot_accepted = 0;
    double   tot_sum_H = 0;

    void add_token(bool has_H, double H, double margin) {
        ++n; ++tot_n;
        if (!has_H) return;
        ++n_H; ++tot_n_H;
        sum_H += H; tot_sum_H += H;
        max_H = std::max(max_H, H);
        min_margin = std::min(min_margin, margin);
        if (H > kHighEntropyNats) { ++n_hi; ++tot_n_hi; }
    }
    void add_draft(uint32_t offered, uint32_t accepted) {
        draft_offered += offered; draft_accepted += accepted;
        tot_offered += offered; tot_accepted += accepted;
    }
    bool window_empty() const { return n == 0 && draft_offered == 0; }
    void reset_window() { n = n_H = n_hi = 0; sum_H = max_H = 0; min_margin = 1; draft_offered = draft_accepted = 0; }
};

}  // namespace ie
