// include/ie/ngram_draft.hpp -- the prompt-lookup draft index (V4.1 Phase 58, docs/deepseek41/97; shared with MiMo-V2.6,
// docs/mimo26/00_PORT_PLAN.md): where the context's suffix occurred before, and the continuation after it.
#pragma once

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace ie {

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

}  // namespace ie
