// tests/unit/ds41_prefill_plan_test.cpp -- P4 B6b (docs/deepseek41/P4_B6B_SERVE.md): Ds41Generator::plan_prefill, the prefill
// plan that run() and the lanes' serving path share. On random prompts (a first and a last user message start, an optional
// trailing think tag, any reused prefix and chunk cap):
//   (1) the planned branch equals the inline plan run() carried before the extraction (a verbatim copy below, d0e39c7);
//   (2) every plan is admissible step by step, as the lane pipe judges a step (Ds41Forward::pipe_submit): contiguous chunks
//       from the reused positions, a pos0 = 0 chunk of even length (V4.1's compress ratios are 1 and 2), a continuation of
//       more than kDs41MaxDecodeRows rows, at most `cap` rows; the rest one row at a time, never more than kDs41MaxDecodeRows
//       of them before the planned end once a chunk ran; Tp = T - 1 exactly when a planned prompt of 4+ tokens ends in a
//       think tag; persist_at a chunk end.
// CPU only.
#include "ie/deepseek41_generate.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <utility>
#include <vector>

namespace {

constexpr int32_t kUser = 128803, kThink = 128821, kThinkEnd = 128822;   // any distinct ids do; the plan only compares them

struct Ref { std::vector<std::pair<uint32_t, uint32_t>> chunks; uint32_t off = 0, Tp = 0, persist_at = 0; };

// run()'s planned branch at d0e39c7 (src/model/deepseek41_generate.cpp), verbatim but for the tokenizer lookups
Ref reference_plan(const std::vector<int32_t>& prompt_ids, uint32_t reused, uint32_t cap) {
    const uint32_t T = uint32_t(prompt_ids.size());
    uint32_t persist_at = 0;
    const bool planned = true;
    uint32_t Tp = T;
    if (planned && T >= 4) {
        const int32_t last = prompt_ids[T - 1];
        if (last >= 0 && (last == kThink || last == kThinkEnd)) Tp = T - 1;
    }
    std::vector<uint32_t> cuts;
    if (const int32_t u = kUser; u >= 0) {
        const auto f = std::find(prompt_ids.begin(), prompt_ids.end(), u);
        const auto l = std::find(prompt_ids.rbegin(), prompt_ids.rend(), u);
        const uint32_t first = f != prompt_ids.end() ? uint32_t(f - prompt_ids.begin()) : 0u;
        const uint32_t lastu = l != prompt_ids.rend() ? uint32_t(prompt_ids.size() - 1 - size_t(l - prompt_ids.rbegin())) : 0u;
        uint32_t prev = reused;
        for (uint32_t h : {first, lastu}) {
            const bool is_first = h == first;
            if (reused == 0) h &= ~1u;
            if (h == 0 || h <= prev + 16 || h + 1 >= Tp) continue;
            if (is_first) persist_at = h;
            cuts.push_back(h); prev = h;
        }
    }
    cuts.push_back(Tp);
    std::vector<std::pair<uint32_t, uint32_t>> chunks;
    uint32_t off = reused;
    for (const uint32_t end : cuts) {
        const bool last = end == Tp;
        while (end - off > (off == 0 ? 0u : ie::kDs41MaxDecodeRows)) {
            uint32_t t = std::min(cap, end - off);
            uint32_t rest = end - off - t;
            if (rest >= 1 && rest <= ie::kDs41MaxDecodeRows && t > 2 * (ie::kDs41MaxDecodeRows + 1)) { t -= ie::kDs41MaxDecodeRows + 1 - rest; rest = end - off - t; }
            if (off == 0 && (t & 1u)) { if (t == 1) break; --t; }
            if (off == 0 && !last && end - off - t >= 1 && end - off - t <= ie::kDs41MaxDecodeRows) t -= 2;
            chunks.push_back({off, t}); off += t;
        }
        if (!last && off != end) { chunks.clear(); cuts.clear(); break; }
    }
    if (cuts.empty()) {
        persist_at = 0; off = reused;
        while (Tp - off > (off == 0 ? 0u : ie::kDs41MaxDecodeRows)) {
            uint32_t t = std::min(cap, Tp - off); const uint32_t rest = Tp - off - t;
            if (rest >= 1 && rest <= ie::kDs41MaxDecodeRows && t > 2 * (ie::kDs41MaxDecodeRows + 1)) t -= ie::kDs41MaxDecodeRows + 1 - rest;
            if (off == 0 && (t & 1u)) { if (t == 1) break; --t; }
            chunks.push_back({off, t}); off += t;
        }
    }
    return {chunks, off, Tp, persist_at};
}

int fails = 0;
void check(bool ok, const char* what, uint32_t T, uint32_t reused, uint32_t cap, bool planned) {
    if (ok) return;
    if (++fails <= 20) std::printf("FAIL %s (T %u, reused %u, cap %u, planned %d)\n", what, T, reused, cap, int(planned));
}

}  // namespace

int main() {
    std::mt19937_64 rng(20260926);
    const uint32_t caps[] = {2048, 1024, 512, 64, 18, 20};
    uint64_t plans = 0, chunks = 0, with_persist = 0, with_tag = 0;
    for (int it = 0; it < 200000; ++it) {
        // a prompt: filler ids, a first user message start, maybe a later one, maybe a trailing think tag
        const uint32_t T = 2 + uint32_t(rng() % (it % 3 == 0 ? 40000 : 3000));
        std::vector<int32_t> ids(T);
        for (auto& x : ids) x = int32_t(rng() % 100000);
        if (T > 3 && rng() % 8) ids[rng() % T] = kUser;
        if (T > 3 && rng() % 2) ids[T / 2 + rng() % (T - T / 2)] = kUser;
        if (T > 3 && rng() % 3) ids[T - 1] = rng() % 2 ? kThink : kThinkEnd;
        const uint32_t cap = caps[rng() % 6];
        const uint32_t reused = rng() % 3 ? 0u : uint32_t(rng() % T);           // < T: the last token always runs
        for (const bool planned : {true, false}) {
            if (!planned && reused) continue;                                    // (the plain plan never follows a reuse)
            const ie::Ds41PrefillPlan pl = ie::Ds41Generator::plan_prefill(ids, reused, cap, planned, kUser, kThink, kThinkEnd);
            ++plans; chunks += pl.chunks.size();
            if (planned) {
                const Ref ref = reference_plan(ids, reused, cap);
                check(pl.chunks == ref.chunks && pl.tail == ref.off && pl.Tp == ref.Tp && pl.persist_at == ref.persist_at,
                      "the planned branch equals run()'s inline plan (d0e39c7)", T, reused, cap, planned);
            }
            const bool tag = planned && T >= 4 && (ids[T - 1] == kThink || ids[T - 1] == kThinkEnd);
            with_tag += tag;
            check(pl.Tp == (tag ? T - 1 : T), "Tp = T - 1 exactly for a planned prompt ending in a think tag", T, reused, cap, planned);
            uint32_t p = reused;
            bool ok = true;
            for (const auto& [p0, t] : pl.chunks) {
                ok = ok && p0 == p && t >= 1 && t <= cap && p0 + t <= pl.Tp;
                ok = ok && (p0 == 0 ? t % 2 == 0 : t > ie::kDs41MaxDecodeRows);
                p = p0 + t;
            }
            check(ok, "contiguous chunks from reused, each admissible (pos0 = 0: even; later: > kDs41MaxDecodeRows rows; <= cap)", T, reused, cap, planned);
            check(pl.tail == p, "the tail starts where the chunks end", T, reused, cap, planned);
            check(pl.tail <= pl.Tp, "the chunks end at or before the planned end", T, reused, cap, planned);
            // what follows the chunks runs one row at a time: a remainder of at most kDs41MaxDecodeRows before the planned end
            // (planned), or the even prefix's decode-sized gap plus the odd tail (not planned); no row at pos0 = 0
            if (planned) check(pl.Tp - pl.tail <= ie::kDs41MaxDecodeRows, "at most kDs41MaxDecodeRows one-row steps before the planned end", T, reused, cap, planned);
            else check(T - pl.tail <= ie::kDs41MaxDecodeRows + 1, "at most kDs41MaxDecodeRows + 1 one-row steps (the gap and the odd tail)", T, reused, cap, planned);
            check(pl.tail > 0, "a one-row step never runs at position 0 (a prefill needs an even T)", T, reused, cap, planned);
            if (pl.persist_at) {
                ++with_persist;
                bool at_end = false;
                for (const auto& [p0, t] : pl.chunks) at_end = at_end || p0 + t == pl.persist_at;
                check(planned && at_end, "persist_at is a chunk end of a planned prefill", T, reused, cap, planned);
            }
        }
    }
    std::printf("%llu plans (%llu chunks; %llu with a disk-entry end, %llu ending in a think tag): %s\n", (unsigned long long)plans,
                (unsigned long long)chunks, (unsigned long long)with_persist, (unsigned long long)with_tag, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
