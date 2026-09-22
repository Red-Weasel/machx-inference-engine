// src/model/deepseek41_generate.cpp — see include/ie/deepseek41_generate.hpp.
#include "ie/deepseek41_generate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <unordered_map>

namespace ie {
namespace {
// splitmix64: a seeded stream of uniforms for the multinomial draw
inline uint64_t next_u64(uint64_t& s) { uint64_t z = (s += 0x9E3779B97F4A7C15ull); z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull; return z ^ (z >> 31); }
inline double uniform01(uint64_t& s) { return double(next_u64(s) >> 11) * (1.0 / 9007199254740992.0); }
// the longest prefix of `b` that ends on a complete UTF-8 sequence
size_t utf8_complete_prefix(const std::string& b) {
    if (b.empty()) return 0;
    size_t i = b.size();
    // walk back over up to 3 continuation bytes to the leader of the last sequence
    size_t back = 0;
    while (i > 0 && back < 3 && (uint8_t(b[i - 1]) & 0xC0) == 0x80) { --i; ++back; }
    if (i == 0) return back == 0 ? b.size() : 0;      // only continuation bytes: hold them
    const uint8_t lead = uint8_t(b[i - 1]);
    size_t need = 1;
    if ((lead & 0x80) == 0) need = 1; else if ((lead & 0xE0) == 0xC0) need = 2; else if ((lead & 0xF0) == 0xE0) need = 3; else if ((lead & 0xF8) == 0xF0) need = 4; else return b.size();   // invalid lead: pass through
    const size_t have = 1 + back;
    return have >= need ? b.size() : i - 1;           // the sequence is complete, or hold it back
}

// Phase 58 (docs/deepseek41/97): where the context's suffix occurred before. For every n-gram (n = 2..6) its last
// kOcc ends; a draft is the continuation after the EARLIER occurrence of the longest suffix found whose match extends
// furthest back (ties: the most recent), taken only when that match is at least `min_match` tokens. "Furthest back"
// rather than "most recent": in a rewrite of a list, the line just written shares every fixed field with the next one,
// and only the original line (in the context) shares its number too. tools/ds41_reference/lookup_policy.py is the
// same logic, priced on real traces.
class Ds41NgramIndex {
public:
    static constexpr uint32_t kNmin = 2, kNmax = 6, kOcc = 16, kMaxBack = 64;
    void reset(const std::vector<int32_t>& s) { seq_.clear(); for (auto& m : occ_) m.clear(); seq_.reserve(s.size() + 4096); for (int32_t t : s) push(t); }
    void push(int32_t t) {
        seq_.push_back(t); const size_t end = seq_.size();
        for (uint32_t n = kNmin; n <= kNmax && n <= end; ++n) {
            auto& l = occ_[n][key(end - n, n)];
            if (l.size() == kOcc) l.erase(l.begin());
            l.push_back(uint32_t(end));
        }
    }
    std::vector<int32_t> draft(uint32_t K, uint32_t min_match) const {
        std::vector<int32_t> d;
        const size_t end = seq_.size();
        for (uint32_t n = kNmax; n >= kNmin; --n) {
            if (end < size_t(n) + 1) continue;
            const auto it = occ_[n].find(key(end - n, n));
            if (it == occ_[n].end() || it->second.size() < 2) continue;         // back() is the suffix itself
            size_t best_e = 0; uint32_t best_m = 0;
            for (size_t i = 0; i + 1 < it->second.size(); ++i) {
                const size_t e = it->second[i];
                if (!std::equal(seq_.begin() + std::ptrdiff_t(e - n), seq_.begin() + std::ptrdiff_t(e), seq_.begin() + std::ptrdiff_t(end - n))) continue;   // a hash collision
                uint32_t m = n;
                while (m < kMaxBack && end >= size_t(m) + 1 && e >= size_t(m) + 1 && seq_[end - m - 1] == seq_[e - m - 1]) ++m;
                if (m >= best_m) { best_m = m; best_e = e; }                   // later entries are more recent: ties go to them
            }
            if (!best_e) continue;
            if (best_m < min_match) return d;
            for (size_t c = best_e; c < end && d.size() < K && seq_[c] >= 0; ++c) d.push_back(seq_[c]);   // never draft an image position
            return d;
        }
        return d;
    }
private:
    uint64_t key(size_t at, uint32_t n) const {
        uint64_t h = 0xCBF29CE484222325ull ^ n;
        for (uint32_t i = 0; i < n; ++i) { h ^= uint32_t(seq_[at + i]); h *= 0x100000001B3ull; }
        return h;
    }
    std::vector<int32_t> seq_;
    std::unordered_map<uint64_t, std::vector<uint32_t>> occ_[kNmax + 1];
};
}  // namespace

int32_t Ds41Generator::sample(std::vector<float>& lg, const std::vector<int32_t>& recent, const Ds41SampleParams& sp, uint64_t& rng) {
    return sample_row(lg.data(), lg.size(), recent, sp, rng);
}

int32_t Ds41Generator::sample_row(float* lg, size_t V, const std::vector<int32_t>& recent, const Ds41SampleParams& sp, uint64_t& rng) {
    if (sp.repeat_penalty != 1.f && sp.repeat_window)
        for (size_t i = recent.size() > sp.repeat_window ? recent.size() - sp.repeat_window : 0; i < recent.size(); ++i) {
            const int32_t id = recent[i];
            if (id < 0 || size_t(id) >= V) continue;
            lg[size_t(id)] = lg[size_t(id)] > 0 ? lg[size_t(id)] / sp.repeat_penalty : lg[size_t(id)] * sp.repeat_penalty;
        }
    if (sp.temperature <= 0.f) return int32_t(std::max_element(lg, lg + V) - lg);
    std::vector<int32_t> idx(V); std::iota(idx.begin(), idx.end(), 0);
    const size_t keep = sp.top_k && sp.top_k < V ? sp.top_k : V;
    std::partial_sort(idx.begin(), idx.begin() + std::ptrdiff_t(keep), idx.end(), [&](int32_t a, int32_t b) { return lg[size_t(a)] > lg[size_t(b)]; });
    idx.resize(keep);
    std::vector<double> p(keep); double mx = lg[size_t(idx[0])] / sp.temperature, z = 0;
    for (size_t i = 0; i < keep; ++i) { p[i] = std::exp(double(lg[size_t(idx[i])]) / sp.temperature - mx); z += p[i]; }
    for (auto& v : p) v /= z;
    size_t n = keep;
    if (sp.top_p < 1.f) { double c = 0; n = 0; while (n < keep) { c += p[n]; ++n; if (c >= sp.top_p) break; } }
    if (sp.min_p > 0.f) { const double floor = sp.min_p * p[0]; size_t m = 0; while (m < n && p[m] >= floor) ++m; n = std::max<size_t>(1, m); }
    double zz = 0; for (size_t i = 0; i < n; ++i) zz += p[i];
    const double r = uniform01(rng) * zz; double c = 0;
    for (size_t i = 0; i < n; ++i) { c += p[i]; if (r < c) return idx[i]; }
    return idx[n - 1];
}

namespace {
// The plain loop's per-token emission -- eos, the ids, the UTF-8 hold-back, the stop strings -- one token at a time in
// order; the speculative loop (docs/deepseek41/58) feeds it the committed tokens the same way.
struct Ds41Emitter {
    const Tokenizer& tok; const std::vector<std::string>& stops; const std::function<bool(std::string_view)>& on_piece;
    std::vector<int32_t>& out_ids; std::vector<int32_t>& recent; Ds41GenStats& st;
    std::string pending, tail;                                    // bytes not yet complete UTF-8; emitted text tail for the stop strings
    size_t longest_stop = 0;
    uint32_t warm_mark = 0;                                       // P5: the committed token the steady-state window starts at
    std::chrono::steady_clock::time_point warm_t0{};
    std::chrono::steady_clock::time_point decode_t0{};            // set at the first committed token
    Ds41Emitter(const Tokenizer& t, const std::vector<std::string>& s, const std::function<bool(std::string_view)>& cb,
                std::vector<int32_t>& o, std::vector<int32_t>& r, Ds41GenStats& stats)
        : tok(t), stops(s), on_piece(cb), out_ids(o), recent(r), st(stats) { for (const auto& x : stops) longest_stop = std::max(longest_stop, x.size()); }
    // A reply that has become one short pattern repeated is finished, whatever it says next: live
    // (2026-09-19) a corrupted image encode made the model emit one word 11,584 times, 14 minutes of
    // decoding. 1,024 tokens in, every 64th, the last 512 are checked for a period of at most 16.
    bool repeating() const {
        constexpr uint32_t kMin = 1024, kWindow = 512, kMaxPeriod = 16;
        if (st.n_gen < kMin || st.n_gen % 64) return false;
        const size_t n = out_ids.size();
        if (n < size_t(kWindow) + kMaxPeriod) return false;
        for (uint32_t p = 1; p <= kMaxPeriod; ++p) {
            bool same = true;
            for (size_t i = n - kWindow; i < n && same; ++i) same = out_ids[i] == out_ids[i - p];
            if (same) return true;
        }
        return false;
    }
    bool push(int32_t id) {                                       // false: the run ended (st.stop_reason set)
        if (id == tok.eos_token_id()) { st.stop_reason = "eos"; return false; }
        out_ids.push_back(id); recent.push_back(id); ++st.n_gen;
        // Early-window rate, for the decode-variance question.
        if (st.n_gen == 1) decode_t0 = std::chrono::steady_clock::now();
        else if (st.n_gen == 100) {
            st.early_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - decode_t0).count();
            st.early_n = 100;
        }
        // P5 (docs/deepseek41/59): the window opens at the mark and closes at every later token, so warm_decode_s /
        // warm_n describe the run from the mark onwards -- the cold steps after the prefill excluded.
        if (warm_mark && st.n_gen == warm_mark) warm_t0 = std::chrono::steady_clock::now();
        else if (warm_mark && st.n_gen > warm_mark) { st.warm_decode_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - warm_t0).count(); st.warm_n = st.n_gen - warm_mark; }
        if (repeating()) { st.stop_reason = "repetition"; return false; }
        pending += tok.decode(std::span<const int32_t>(&id, 1), /*skip_special=*/false);
        const size_t n = utf8_complete_prefix(pending);
        if (n) {
            std::string piece = pending.substr(0, n); pending.erase(0, n);
            bool stopped = false;
            if (!stops.empty()) {
                const std::string window = tail + piece;
                size_t cut = std::string::npos;
                for (const auto& s : stops) { const size_t at = window.find(s); if (at != std::string::npos && (cut == std::string::npos || at < cut)) cut = at; }
                if (cut != std::string::npos) { piece = cut > tail.size() ? window.substr(tail.size(), cut - tail.size()) : std::string(); stopped = true; }
            }
            if (!piece.empty() && !on_piece(piece)) { st.stop_reason = "callback"; return false; }
            tail += piece; if (tail.size() > longest_stop) tail.erase(0, tail.size() - longest_stop);
            if (stopped) { st.stop_reason = "stop"; return false; }
        }
        return true;
    }
};
double ms_since(const std::chrono::steady_clock::time_point& t) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(); }
}  // namespace

std::string Ds41Generator::run(const std::vector<int32_t>& prompt_ids, uint32_t max_new, const Ds41SampleParams& sp,
                               const std::vector<std::string>& stops, const std::function<bool(std::string_view)>& on_piece,
                               std::vector<int32_t>& out_ids, Ds41GenStats& st) {
    st = Ds41GenStats{}; out_ids.clear();
    const uint32_t T = uint32_t(prompt_ids.size());
    if (T < 2) { st.stop_reason = "error"; return "the prompt needs at least two tokens (the prefill takes an even count)"; }
    if (T + max_new > fwd_.capacity()) { st.stop_reason = "error"; return "prompt " + std::to_string(T) + " + " + std::to_string(max_new) + " new tokens exceed the runtime's position capacity " + std::to_string(fwd_.capacity()); }
    const bool spec = drafter_ && drafter_->ready() && sp.temperature <= 0.f;   // DSpark P4
    if (drafter_ && drafter_->ready() && sp.temperature > 0.f) std::fprintf(stderr, "[ds41 generate] speculation off: temperature > 0 (docs/deepseek41/58: temperature 0 only)\n");
    const uint32_t K = spec ? drafter_->block_size() : 0;
    if (spec && T + max_new + K > fwd_.capacity()) { st.stop_reason = "error"; return "prompt " + std::to_string(T) + " + " + std::to_string(max_new) + " new tokens + the " + std::to_string(K) + "-row verify block exceed the runtime's position capacity " + std::to_string(fwd_.capacity()); }
    if (spec && spec_k_ > K) { st.stop_reason = "error"; return "IE_DS41_SPEC_K " + std::to_string(spec_k_) + " exceeds the drafter's block " + std::to_string(K); }
    fwd_.set_logits_last_only(true); fwd_.set_capture_main_hidden(spec);
    std::vector<float> logits;
    const auto t0 = std::chrono::steady_clock::now();
    // Phase 46 (docs/deepseek41/86): the prefix cache -- the state is left holding prompt_ids[0, reused) and only the
    // rest is run. Off under speculation (the drafter's rings are not checkpointed).
    uint32_t reused = 0;
    if (!spec && fwd_.prefix_cache()) {
        const auto tr = std::chrono::steady_clock::now();
        if (auto e = fwd_.prefix_prepare(prompt_ids, reused, &st.cache_source); !e.empty()) { st.stop_reason = "error"; return "prefix cache: " + e; }
        st.restore_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - tr).count();
        st.n_cached = reused;
    }
    const uint32_t Te = T & ~1u;                                   // the even prefix
    // PHASE 24: THE PROMPT IS FED IN CHUNKS when it is longer than one forward can carry, so a caller never has
    // to know that limit exists. The chunk is `forward_capacity()` rounded DOWN to even, because a ratio-2
    // compressor group must not be split across calls: an even chunk starting at an even position always leaves
    // the held half closed, so `starts_odd == part_valid` holds by construction. Te and the chunk are both even,
    // so the final piece is even too.
    const uint32_t cap = std::max<uint32_t>(2u, fwd_.forward_capacity() & ~1u);
    uint32_t pf_last_T = Te, pf_last_pos = 0;                      // the final chunk -- what seeds the drafter
    // Phase 43 (docs/deepseek41/83): the chunks run with the cards as pipeline stages (bit-identical, both links streaming
    // at once: 32k prefill 165 -> 278 tok/s). ON by default; IE_DS41_PIPE_PREFILL=0 is the kill switch. The drafter's
    // capture keeps the serial loop.
    static const bool pipe_prefill = [] { const char* v = std::getenv("IE_DS41_PIPE_PREFILL"); return !(v && *v && std::string(v) == "0"); }();
    uint32_t persist_at = 0;                                       // Phase 47: the system prefix's aligned end, for the disk
    const bool planned = fwd_.prefix_cache() && !spec;
    // Phase 57 (docs/deepseek41/96): a chat prompt ends `<｜Assistant｜><think>` (or `</think>`), and the NEXT turn renders
    // this same assistant turn with the other tag -- the two prompts part ways at T - 1, so the checkpoint at T below is
    // never reusable and every turn fell back to the last user message's start (re-running the whole message: with an
    // image, its encode and hundreds of tokens). The plan ends one token early, a checkpoint is taken there, and the tag
    // runs as a one-row step.
    uint32_t Tp = T;
    if (planned && T >= 4) {
        const int32_t last = prompt_ids[T - 1];
        if (last >= 0 && (last == tok_.find_token("<think>") || last == tok_.find_token("</think>"))) Tp = T - 1;
    }
    if (planned) {
        // Phase 46/47: with the cache on, the prompt runs from `reused` (0 = a fresh prefill) in chunks whose boundaries
        // include where the FIRST and the LAST user message begin (a special token, so a token boundary in every
        // rendering): the first is the system prompt + tools shared by every conversation of a client, the last the point
        // a regenerated or edited reply resumes from. Chunks of `cap`, never leaving 2..8 rows before a boundary (a chunk
        // shortened to leave 9), a pos0 = 0 chunk of even length (a prefill's T is a multiple of the ratio-2 groups), a
        // remainder of 1..8 rows at the very end fed one token at a time. A continuation may start at an odd position.
        std::vector<uint32_t> cuts;
        if (const int32_t u = tok_.find_token("<｜User｜>"); u >= 0) {
            const auto f = std::find(prompt_ids.begin(), prompt_ids.end(), u);
            const auto l = std::find(prompt_ids.rbegin(), prompt_ids.rend(), u);
            const uint32_t first = f != prompt_ids.end() ? uint32_t(f - prompt_ids.begin()) : 0u;
            const uint32_t lastu = l != prompt_ids.rend() ? uint32_t(prompt_ids.size() - 1 - size_t(l - prompt_ids.rbegin())) : 0u;
            uint32_t prev = reused;
            for (uint32_t h : {first, lastu}) {
                const bool is_first = h == first;
                if (reused == 0) h &= ~1u;                         // a pos0 = 0 chunk ending here must have even length
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
            while (end - off > (off == 0 ? 0u : kDs41MaxDecodeRows)) {
                uint32_t t = std::min(cap, end - off);
                uint32_t rest = end - off - t;
                if (rest >= 1 && rest <= kDs41MaxDecodeRows && t > 2 * (kDs41MaxDecodeRows + 1)) { t -= kDs41MaxDecodeRows + 1 - rest; rest = end - off - t; }
                if (off == 0 && (t & 1u)) { if (t == 1) break; --t; }
                if (off == 0 && !last && end - off - t >= 1 && end - off - t <= kDs41MaxDecodeRows) t -= 2;   // keep the boundary a chunk end
                chunks.push_back({off, t}); off += t;
            }
            if (!last && off != end) { chunks.clear(); cuts.clear(); break; }   // cannot land on this boundary: plain plan below
        }
        if (cuts.empty()) {                                        // the plain plan (no boundaries), from `reused`
            persist_at = 0; off = reused;
            while (Tp - off > (off == 0 ? 0u : kDs41MaxDecodeRows)) {
                uint32_t t = std::min(cap, Tp - off); const uint32_t rest = Tp - off - t;
                if (rest >= 1 && rest <= kDs41MaxDecodeRows && t > 2 * (kDs41MaxDecodeRows + 1)) t -= kDs41MaxDecodeRows + 1 - rest;
                if (off == 0 && (t & 1u)) { if (t == 1) break; --t; }
                chunks.push_back({off, t}); off += t;
            }
        }
        if (chunks.size() > 1 && pipe_prefill && fwd_.pipelined_admissible()) {
            if (auto e = fwd_.forward_pipelined(prompt_ids.data(), chunks, logits); !e.empty()) { st.stop_reason = "error"; return "prefill from " + std::to_string(reused) + " (pipelined): " + e; }
        } else for (const auto& [p0, t] : chunks)
            if (auto e = fwd_.forward(prompt_ids.data() + p0, t, p0, logits); !e.empty()) { st.stop_reason = "error"; return "prefill from " + std::to_string(reused) + ", chunk at " + std::to_string(p0) + ": " + e; }
        for (; off < T; ++off) {
            if (off == Tp && Tp < T && off > reused)               // everything before the trailing think tag is in: the point the next turn resumes from
                if (auto e = fwd_.prefix_checkpoint(); !e.empty()) { st.stop_reason = "error"; return "prefix checkpoint before the think tag: " + e; }
            if (auto e = fwd_.forward(prompt_ids.data() + off, 1, off, logits); !e.empty()) { st.stop_reason = "error"; return "prefill from " + std::to_string(reused) + ", token at " + std::to_string(off) + ": " + e; }
        }
    } else if (Te > cap && pipe_prefill && !spec && fwd_.pipelined_admissible()) {
        std::vector<std::pair<uint32_t, uint32_t>> chunks;
        uint32_t off = 0;
        for (; off < Te; ) { const uint32_t t = std::min(cap, Te - off); if (off > 0 && t <= kDs41MaxDecodeRows) break; chunks.push_back({off, t}); off += t; }
        if (auto e = fwd_.forward_pipelined(prompt_ids.data(), chunks, logits); !e.empty()) { st.stop_reason = "error"; return "prefill (pipelined): " + e; }
        for (; off < Te; ++off)                                    // the tail gap, one token at a time as below
            if (auto e = fwd_.forward(prompt_ids.data() + off, 1, off, logits); !e.empty()) { st.stop_reason = "error"; return "prefill tail at " + std::to_string(off) + ": " + e; }
    } else if (Te > cap) {
        for (uint32_t off = 0; off < Te; ) {
            uint32_t t = std::min(cap, Te - off);
            // THE TAIL GAP, and it is a real one: at pos0 > 0 the forward admits T == 1 always and
            // T > kDs41MaxDecodeRows as a continuation, but 2..kDs41MaxDecodeRows only under
            // IE_DS41_DECODE_MULTI (that band is the DSpark verify path, opt-in). So a prompt whose even
            // prefix leaves a remainder in that band used to fail AFTER the whole prefill had run --
            // measured on a 223,236-token prompt, which is 109 x 2048 + 4, refused at the last chunk 37
            // minutes in. Such a tail is fed ONE TOKEN AT A TIME instead: always admitted, the most
            // heavily verified path there is, and at most kDs41MaxDecodeRows extra calls.
            if (off > 0 && t <= kDs41MaxDecodeRows) t = 1;
            // spec: only the LAST chunk's rows can reach the drafter's rings, so capture only there
            if (spec) fwd_.set_capture_main_hidden(off + t >= Te);
            if (auto e = fwd_.forward(prompt_ids.data() + off, t, off, logits); !e.empty()) {
                st.stop_reason = "error";
                return "prefill chunk at " + std::to_string(off) + " (" + std::to_string(t) + " of " +
                       std::to_string(Te) + " tokens): " + e; }
            pf_last_T = t; pf_last_pos = off;
            off += t;
        }
    } else if (auto e = fwd_.forward(prompt_ids.data(), Te, 0, logits); !e.empty()) { st.stop_reason = "error"; return "prefill: " + e; }
    std::vector<float> mh_p;                                       // spec: main_hidden(p) for the next draft
    if (spec) {                                                    // the rings from the prompt's capture (the last min(Te, window) rows land)
        const auto& mh = fwd_.main_hidden(); const size_t row = mh.size() / pf_last_T;
        if (auto e = drafter_->seed(*dq_, mh.data(), pf_last_T, pf_last_pos); !e.empty()) { st.stop_reason = "error"; return "drafter seed: " + e; }
        mh_p.assign(mh.end() - std::ptrdiff_t(row), mh.end());
    }
    if (!planned && Te < T) { if (auto e = fwd_.forward(&prompt_ids[Te], 1, Te, logits); !e.empty()) { st.stop_reason = "error"; return "prefill's odd tail: " + e; }
                  // the tail position's main_kv must reach the rings too: `draft` writes no ring slot (only `seed`
                  // does, deepseek41_dspark.cpp), and the draft mask opens every slot up to pos (gate P4 finding 2)
                  if (spec) { mh_p = fwd_.main_hidden();
                      if (auto e = drafter_->seed(*dq_, mh_p.data(), 1, Te); !e.empty()) { st.stop_reason = "error"; return "drafter seed (the prompt's odd tail): " + e; } } }
    // Phase 46: the prompt's end is the checkpoint the next turn of this conversation most likely resumes from
    if (fwd_.prefix_cache()) {
        if (auto e = fwd_.prefix_checkpoint(); !e.empty()) { st.stop_reason = "error"; return "prefix checkpoint: " + e; }
        // Phase 47: the system prompt + tools, once per distinct prefix, to disk for the next process (skipped quickly when
        // it is already there, or too short, or the disk store is off)
        if (persist_at) if (auto e = fwd_.prefix_persist(persist_at); !e.empty()) std::fprintf(stderr, "[ds41 generate] prefix persist: %s\n", e.c_str());
    }
    st.prefill_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    st.n_prompt = T;
    // THE PREFILL/DECODE BOUNDARY: from here the routing profile accumulates decode selections only, which is
    // what a residency ranking must be built from (docs/deepseek41/67).
    if (profile_decode_only_) fwd_.reset_profile();

    std::vector<int32_t> recent(prompt_ids.end() - std::min<size_t>(prompt_ids.size(), sp.repeat_window), prompt_ids.end());
    uint64_t rng = sp.seed ? sp.seed : 0x2545F4914F6CDD1Dull;
    Ds41Emitter em(tok_, stops, on_piece, out_ids, recent, st); em.warm_mark = warm_mark_;
    const auto t1 = std::chrono::steady_clock::now();
    // Phase 58 (docs/deepseek41/97): prompt-lookup speculation. IE_DS41_LOOKUP=1 turns it on; IE_DS41_LOOKUP_K caps the
    // drafts per pass (rows = 1 + K <= kDs41MaxDecodeRows), IE_DS41_LOOKUP_MIN is the copy length a draft needs.
    struct LookupPolicy { bool on = false; uint32_t k = 7, min_match = 12; };
    static const LookupPolicy lp = [] {
        LookupPolicy p;
        if (const char* v = std::getenv("IE_DS41_LOOKUP")) p.on = std::string(v) == "1";
        if (const char* v = std::getenv("IE_DS41_LOOKUP_K")) p.k = uint32_t(std::max(1, std::min(int(kDs41MaxDecodeRows) - 1, std::atoi(v))));
        if (const char* v = std::getenv("IE_DS41_LOOKUP_MIN")) p.min_match = uint32_t(std::max(2, std::atoi(v)));
        if (p.on) std::fprintf(stderr, "[ds41 lookup] prompt-lookup speculation ON: up to %u drafts per pass, a copy of >= %u tokens\n", p.k, p.min_match);
        return p;
    }();
    const bool lookup_on = lookup_ < 0 ? lp.on : lookup_ == 1;
    if (!spec && lookup_on) {
        // The caches hold 0 .. pos-1; `id` is sampled and not yet fed. A pass drafts d (a copy from the context), feeds
        // [id, d] as one multi-row step with every row's logits, and row r judges d[r] with the plain sampler and the
        // running `recent`: sampling each row from p and keeping the drafts while they equal the sample is exact
        // speculative sampling for a one-hot draft (greedy included). The first mismatch's sample is the next `id`;
        // rollback_to keeps the rows of id and the accepted drafts. No draft: a plain one-row step (the CPU miss leg).
        Ds41NgramIndex idx; idx.reset(prompt_ids);
        fwd_.set_multi_row_decode(true);
        // where a step's time goes, summed over the layers: [0] the one-row steps, [1] the verify steps (docs/97)
        struct Acc { double wall = 0, attn = 0, moe = 0, pre = 0, shd = 0; double bp = 0, bm = 0, sta = 0, pin = 0, mmx = 0, cpu = 0, hit = 0; uint32_t steps = 0, rows = 0; } acc[2];
        auto account = [&](int k, uint32_t rows_, double wall) {
            Acc& a = acc[k]; ++a.steps; a.rows += rows_; a.wall += wall;
            for (const auto& L : fwd_.stats()) {
                a.attn += L.attn_ms; a.moe += L.moe_call_ms; a.pre += L.ffn_pre_ms; a.shd += L.shared_ms;
                a.bp += double(L.bytes_pinned); a.bm += double(L.bytes_mmap); a.sta += L.experts_static; a.pin += L.experts_pinned;
                a.mmx += L.experts_mmap; a.cpu += L.experts_cpu; a.hit += L.experts_stream_hit;
            }
        };
        const uint32_t V = uint32_t(logits.size());
        uint32_t pos = T, k_done = 0;
        int32_t id = sample(logits, recent, sp, rng);
        std::vector<int32_t> rows; std::vector<float> vlg;
        for (;;) {
            if (!em.push(id)) break;
            idx.push(id);
            if (++k_done == max_new) { st.stop_reason = "length"; break; }
            std::vector<int32_t> d = idx.draft(lp.k, lp.min_match);
            if (d.size() > max_new - k_done) d.resize(max_new - k_done);
            if (d.empty()) {
                const auto ts = std::chrono::steady_clock::now();
                if (auto e = fwd_.forward(&id, 1, pos, logits); !e.empty()) { st.stop_reason = "error"; return "step at " + std::to_string(pos) + ": " + e; }
                const double w = ms_since(ts); st.lookup_plain_ms += w; ++st.lookup_plain; account(0, 1, w);
                ++pos; id = sample(logits, recent, sp, rng);
                continue;
            }
            rows.assign(1, id); rows.insert(rows.end(), d.begin(), d.end());
            const auto tv = std::chrono::steady_clock::now();
            fwd_.set_logits_last_only(false);
            const std::string ev = fwd_.forward(rows.data(), uint32_t(rows.size()), pos, vlg);
            fwd_.set_logits_last_only(true);
            if (!ev.empty()) { st.stop_reason = "error"; return "lookup verify at " + std::to_string(pos) + " (" + std::to_string(rows.size()) + " rows): " + ev; }
            account(1, uint32_t(rows.size()), ms_since(tv));
            uint32_t L = 0; int32_t next = -1; bool ended = false, drop_last = false;
            for (uint32_t r = 0; r <= d.size(); ++r) {
                const int32_t a = sample_row(vlg.data() + size_t(r) * V, V, recent, sp, rng);
                if (r == d.size() || a != d[r]) { next = a; break; }
                ++L;
                if (!em.push(a)) { ended = true; drop_last = true; break; }   // eos / stop / callback: not part of the output
                idx.push(a);
                if (++k_done == max_new) { st.stop_reason = "length"; ended = true; break; }
            }
            const uint32_t keep = 1 + L - (drop_last ? 1u : 0u);                // rows: id + the accepted drafts
            if (auto e = fwd_.rollback_to(pos + keep); !e.empty()) { st.stop_reason = "error"; return "lookup rollback to " + std::to_string(pos + keep) + ": " + e; }
            st.lookup_verify_ms += ms_since(tv); ++st.lookup_passes; st.lookup_rows += uint32_t(rows.size()); st.lookup_accepted += L;
            if (ended) break;
            pos += 1 + L; id = next;
        }
        std::fprintf(stderr, "[ds41 lookup] %u tokens: %u passes (%u rows, %u drafts accepted, %.1f ms each), %u plain steps (%.1f ms each)\n",
                     st.n_gen, st.lookup_passes, st.lookup_rows, st.lookup_accepted, st.lookup_passes ? st.lookup_verify_ms / st.lookup_passes : 0.0,
                     st.lookup_plain, st.lookup_plain ? st.lookup_plain_ms / st.lookup_plain : 0.0);
        for (int k = 0; k < 2; ++k) {
            const Acc& a = acc[k]; if (!a.steps) continue; const double n = a.steps;
            std::fprintf(stderr, "[ds41 lookup]   %s: %.2f rows/step, per step: wall %.1f ms = attention %.1f + router %.1f + shared %.1f + MoE %.1f (sum over layers); "
                                 "experts static %.1f pinned %.1f (stream hits %.1f, CPU %.1f) mmap %.1f; link %.0f MiB pinned + %.0f MiB mmap\n",
                         k ? "verify  " : "one-row ", a.rows / n, a.wall / n, a.attn / n, a.pre / n, a.shd / n, a.moe / n,
                         a.sta / n, a.pin / n, a.hit / n, a.cpu / n, a.mmx / n, a.bp / n / 1048576.0, a.bm / n / 1048576.0);
        }
    } else if (!spec) {
        uint32_t pos = T;
        for (uint32_t k = 0; k < max_new; ++k) {
            const auto t_s = std::chrono::steady_clock::now();
            const int32_t id = sample(logits, recent, sp, rng);
            const auto t_e = std::chrono::steady_clock::now();
            if (!em.push(id)) break;
            if (acc_stages_ && k > 0) { st.stages.sample += std::chrono::duration<double, std::milli>(t_e - t_s).count(); st.stages.emit += ms_since(t_e); }
            if (k + 1 == max_new) { st.stop_reason = "length"; break; }
            const auto ts = std::chrono::steady_clock::now();
            if (auto e = fwd_.forward(&id, 1, pos, logits); !e.empty()) { st.stop_reason = "error"; return "step " + std::to_string(k) + ": " + e; }
            if (acc_stages_ && k > 0) {                             // step 0 is the cold one after the prefill
                auto& a = st.stages; ++a.steps; a.wall += ms_since(ts); a.prep += fwd_.prep_ms(); a.head += fwd_.head_ms();
                for (const auto& L : fwd_.stats()) {
                    a.att += L.attn_ms; a.fpre += L.ffn_pre_ms; a.moe += L.moe_call_ms; a.grp += L.moe_groups_ms; a.join += L.moe_join_ms;
                    a.mmg += L.moe_mmap_group_ms; a.tail += L.moe_tail_ms; a.shd += L.shared_ms; a.lay += L.ms; a.cpu_ms += L.moe_cpu_ms;
                    a.cpu_w += L.moe_cpu_work_ms; a.bp += double(L.bytes_pinned); a.bm += double(L.bytes_mmap); a.sta += L.experts_static;
                    a.pin += L.experts_pinned; a.mmx += L.experts_mmap; a.sh += L.experts_stream_hit; a.cpu += L.experts_cpu;
                    a.mm_fill += L.moe_mmap_ms; a.mm_pack += L.moe_mmap_pack_ms; a.mm_read += L.moe_mmap_read_ms; a.mm_perm += L.moe_mmap_permute_ms;
                    a.spawn += L.moe_spawn_ms; a.mprep += L.moe_prep_ms;
                }
            }
            ++pos;
        }
    } else {
        // DSpark P4 (docs/deepseek41/58, the algebra): the caches hold 0 .. p; t1 is the token for p + 1 (the plain
        // loop's sample); draft at p; verify [t1, d1 .. dK] at p + 1 with every row's logits; row r judges d(r+1)
        // with the plain sampler and the running recent; the first rejection's argmax (or row K's) is the bonus,
        // the next pass's t1; rollback_to(p + 1 + L); the rings re-seeded from the verify's capture rows 0 .. L-1.
        const uint32_t V = uint32_t(logits.size());
        const uint32_t Kmax = spec_k_ ? std::min(spec_k_, K) : K;             // P5 (docs/59 knob 1): the verify block's length
        uint32_t p = T - 1, k_done = 0;
        int32_t t1 = sample(logits, recent, sp, rng);
        std::vector<int32_t> ids; std::vector<float> dlg, conf, vlg;
        bool ended = false;
        while (!ended && k_done < max_new) {
            const auto td = std::chrono::steady_clock::now();
            if (auto e = drafter_->draft(*dq_, mh_p.data(), t1, p, ids, dlg, conf); !e.empty()) { st.stop_reason = "error"; return "draft at " + std::to_string(p) + ": " + e; }
            st.spec_draft_ms += ms_since(td);
            if (!em.push(t1)) break;
            if (++k_done == max_new) { st.stop_reason = "length"; break; }
            // P5 (docs/59 knob 2): verify the leading run of drafts the confidence head scores at or above the
            // threshold. A run of 0 makes this pass a plain one-row step -- the drafter's cost is still paid.
            uint32_t kp = 0; while (kp < Kmax && conf[kp] >= spec_conf_) ++kp;
            const auto tv = std::chrono::steady_clock::now();
            if (kp) fwd_.set_logits_last_only(false);
            const std::string ev = fwd_.forward(ids.data(), 1 + kp, p + 1, vlg);
            fwd_.set_logits_last_only(true);
            if (!ev.empty()) { st.stop_reason = "error"; return "verify at " + std::to_string(p + 1) + " (" + std::to_string(1 + kp) + " rows): " + ev; }
            st.spec_verify_ms += ms_since(tv); st.spec_rows_verified += 1 + kp; st.spec_declined += kp ? 0 : 1;
            // the verify step's cost over the links, and the experts its rows' union opened per layer (the resident
            // tier's counters -- `experts_uploaded` is the streaming path's field and stays 0 here)
            for (const auto& ls : fwd_.stats()) { st.spec_link_bytes += ls.bytes_pinned + ls.bytes_mmap;
                st.spec_union_sum += ls.experts_static + ls.experts_pinned + ls.experts_mmap; ++st.spec_union_layers; }
            uint32_t L = 1; int32_t bonus = -1; bool drop_last = false;
            for (uint32_t r = 0; r <= kp; ++r) {
                const int32_t a = sample_row(vlg.data() + size_t(r) * V, V, recent, sp, rng);
                if (r == kp) { bonus = a; break; }
                ++st.spec_offered[r]; const bool ok = ids[r + 1] == a;
                st.spec_conf.push_back({uint8_t(r), uint8_t(ok), conf[r]});
                if (!ok) { bonus = a; break; }
                ++st.spec_accepted[r]; ++L;
                if (!em.push(a)) { ended = true; drop_last = true; break; }   // eos / stop / callback: this position is not part of the output
                if (++k_done == max_new) { st.stop_reason = "length"; ended = true; break; }
            }
            ++st.spec_passes; ++st.spec_hist[std::min<uint32_t>(L, 6)];
            // The block is rolled back even when the run ends inside it (gate P4 finding 4), so the caches, n_pos
            // and all_ids_ always describe exactly the emitted sequence -- `run` re-prefills today, but a future
            // prompt-cache reuse would inherit the rejected rows.
            const uint32_t L_keep = L - (drop_last ? 1u : 0u);
            if (kp) {                                                         // a one-row step took no snapshot: nothing to roll back
                const auto tr = std::chrono::steady_clock::now();
                if (auto e = fwd_.rollback_to(p + 1 + L_keep); !e.empty()) { st.stop_reason = "error"; return "rollback at " + std::to_string(p + 1 + L_keep) + ": " + e; }
                st.spec_rollback_ms += ms_since(tr);
            }
            if (ended) break;
            const auto& mh = fwd_.main_hidden(); const size_t row = mh.size() / (1 + kp);
            if (auto e = drafter_->seed(*dq_, mh.data(), L, p + 1); !e.empty()) { st.stop_reason = "error"; return "drafter seed at " + std::to_string(p + 1) + ": " + e; }
            mh_p.assign(mh.begin() + std::ptrdiff_t(size_t(L - 1) * row), mh.begin() + std::ptrdiff_t(size_t(L) * row));
            p += L; t1 = bonus;
        }
    }
    if (st.stop_reason.empty()) st.stop_reason = "length";
    st.decode_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
    // Phase 58 (docs/deepseek41/97): IE_DS41_LOOKUP_SHADOW=1 -- replay this request's greedy output as a prompt-lookup
    // speculative decoder would have run it (draft = the continuation of the most recent earlier occurrence of the
    // last n tokens, n = 4..2; a pass verifies 1 + k rows and commits the matching drafts + one token). Reads only the
    // finished token sequence: the output is untouched. Printed per request for the cost model to price.
    static const bool shadow = [] { const char* v = std::getenv("IE_DS41_LOOKUP_SHADOW"); return v && std::string(v) == "1"; }();
    // IE_DS41_LOOKUP_DUMP=<dir>: the request's token stream for offline policy search -- u32 n_prompt, u32 n_out, then
    // i32 prompt ids, i32 output ids (tools/ds41_reference/lookup_policy.py)
    if (const char* dd = std::getenv("IE_DS41_LOOKUP_DUMP"); dd && *dd && !out_ids.empty()) {
        static uint32_t n_dump = 0;
        char name[64]; std::snprintf(name, sizeof name, "/req_%05u.ids", n_dump++);
        if (FILE* f = std::fopen((std::string(dd) + name).c_str(), "wb")) {
            const uint32_t h[2] = {uint32_t(prompt_ids.size()), uint32_t(out_ids.size())};
            std::fwrite(h, 4, 2, f); std::fwrite(prompt_ids.data(), 4, prompt_ids.size(), f); std::fwrite(out_ids.data(), 4, out_ids.size(), f);
            std::fclose(f);
        }
    }
    if (shadow && !spec && !out_ids.empty()) {
        std::vector<int32_t> seq(prompt_ids); seq.insert(seq.end(), out_ids.begin(), out_ids.end());
        const size_t P = prompt_ids.size(), N = out_ids.size();
        for (const uint32_t K : {3u, 7u}) {                        // drafts per pass; 1 + K <= kDs41MaxDecodeRows
            uint32_t passes = 0, drafted_passes = 0, rows = 0, hist[9] = {};
            for (size_t g = 0; g < N; ) {                          // the pass that commits out_ids[g ..]; its last committed token is seq[P + g - 1]
                const size_t end = P + g;                          // context = seq[0, end)
                std::vector<int32_t> draft;
                for (uint32_t n = 4; n >= 2 && draft.empty(); --n) {
                    if (end < n + 1) continue;
                    for (size_t j = end - n; j-- > 0; ) {          // most recent earlier occurrence of seq[end-n, end)
                        if (!std::equal(seq.begin() + std::ptrdiff_t(j), seq.begin() + std::ptrdiff_t(j + n), seq.begin() + std::ptrdiff_t(end - n))) continue;
                        for (size_t c = j + n; c < end && draft.size() < K; ++c) draft.push_back(seq[c]);
                        break;
                    }
                }
                uint32_t L = 0;
                while (L < draft.size() && g + L < N && draft[L] == out_ids[g + L]) ++L;
                const uint32_t commit = uint32_t(std::min<size_t>(L + 1, N - g));
                ++passes; rows += 1 + uint32_t(draft.size()); if (!draft.empty()) ++drafted_passes;
                ++hist[std::min(commit, 8u)];
                g += commit;
            }
            // price with the measured multi-row decode costs (docs/deepseek41/59, split off): 1 row 86 ms, +41 ms per row
            const double plain_ms = 86.0 * double(N), spec_ms = 86.0 * passes + 41.0 * double(rows - passes);
            std::fprintf(stderr, "[ds41 lookup shadow] k<=%u: %zu tokens in %u passes (%u drafted), %.2f tokens/pass, %u rows verified; "
                                 "commits/pass 1:%u 2:%u 3:%u 4:%u 5:%u 6:%u 7:%u 8:%u; priced %.0f ms vs plain %.0f ms (x%.2f)\n",
                         K, N, passes, drafted_passes, double(N) / passes, rows, hist[1], hist[2], hist[3], hist[4], hist[5], hist[6], hist[7], hist[8],
                         spec_ms, plain_ms, plain_ms / spec_ms);
        }
    }
    return {};
}

}  // namespace ie
