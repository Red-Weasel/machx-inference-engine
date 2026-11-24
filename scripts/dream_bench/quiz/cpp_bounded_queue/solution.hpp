#pragma once
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>

template <class T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t capacity) : cap_(capacity) {
        if (capacity == 0) throw std::invalid_argument("capacity must be positive");
    }

    bool push(T v) {
        std::unique_lock lk(m_);
        not_full_.wait(lk, [&] { return closed_ || q_.size() < cap_; });
        if (closed_) return false;
        q_.push_back(std::move(v));
        not_empty_.notify_one();
        return true;
    }

    std::optional<T> pop() {
        std::unique_lock lk(m_);
        not_empty_.wait(lk, [&] { return closed_ || !q_.empty(); });
        return take_(lk);
    }

    bool try_push(T v) {
        std::lock_guard lk(m_);
        if (closed_ || q_.size() >= cap_) return false;
        q_.push_back(std::move(v));
        not_empty_.notify_one();
        return true;
    }

    std::optional<T> try_pop() {
        std::unique_lock lk(m_);
        return take_(lk);
    }

    void close() {
        { std::lock_guard lk(m_); closed_ = true; }
        not_full_.notify_all();
        not_empty_.notify_all();
    }

    size_t size() const { std::lock_guard lk(m_); return q_.size(); }

private:
    std::optional<T> take_(std::unique_lock<std::mutex>&) {
        if (q_.empty()) return std::nullopt;
        std::optional<T> v(std::move(q_.front()));
        q_.pop_front();
        not_full_.notify_one();
        return v;
    }

    const size_t cap_;
    mutable std::mutex m_;
    std::condition_variable not_full_, not_empty_;
    std::deque<T> q_;
    bool closed_ = false;
};
