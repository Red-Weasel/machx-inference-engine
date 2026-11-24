#pragma once
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>
#include <utility>
#include <vector>

class IntervalSet {
public:
    void add(int64_t lo, int64_t hi) {
        if (lo >= hi) return;
        auto it = m_.upper_bound(lo);
        if (it != m_.begin()) {
            auto p = std::prev(it);
            if (p->second >= lo) {
                lo = p->first; hi = std::max(hi, p->second);
                total_ -= p->second - p->first;
                it = m_.erase(p);
            }
        }
        while (it != m_.end() && it->first <= hi) {
            hi = std::max(hi, it->second);
            total_ -= it->second - it->first;
            it = m_.erase(it);
        }
        m_.emplace(lo, hi);
        total_ += hi - lo;
    }

    void remove(int64_t lo, int64_t hi) {
        if (lo >= hi) return;
        auto it = m_.upper_bound(lo);
        if (it != m_.begin()) {
            auto p = std::prev(it);
            if (p->second > lo) {
                int64_t a = p->first, b = p->second;
                total_ -= b - a;
                m_.erase(p);
                if (a < lo) { m_.emplace(a, lo); total_ += lo - a; }
                if (b > hi) { m_.emplace(hi, b); total_ += b - hi; }
                it = m_.upper_bound(lo);
            }
        }
        while (it != m_.end() && it->first < hi) {
            int64_t a = it->first, b = it->second;
            total_ -= b - a;
            it = m_.erase(it);
            if (b > hi) { m_.emplace(hi, b); total_ += b - hi; break; }
        }
    }

    bool contains(int64_t x) const {
        auto it = m_.upper_bound(x);
        if (it == m_.begin()) return false;
        return x < std::prev(it)->second;
    }

    int64_t covered() const { return total_; }

    std::vector<std::pair<int64_t, int64_t>> intervals() const { return {m_.begin(), m_.end()}; }

private:
    std::map<int64_t, int64_t> m_;
    int64_t total_ = 0;
};
