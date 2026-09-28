#pragma once
// Per-card locks (docs/serve_config.md, "Card locks"): at most one engine process per physical card.
//
// Card N's lock is an exclusive flock(2) on <dir>/ie-card-N.lock (dir = $IE_CARD_LOCK_DIR, default /tmp). flock is
// released by the kernel when the last descriptor closes, so a crashed holder never leaves a stale lock; the file only
// carries the holder's pid for the error message.
//
// Who takes them:
//   - `ie serve|run --cards LIST` takes LIST's locks at startup and refuses to start when one is held.
//   - `ie supervise` takes every child's cards before starting any child, then hands each child its own cards'
//     descriptors (inherited across exec) and IE_GPU_LOCK_HELD=<its cards>.
//   - scripts/ie-run-guarded --cards LIST takes LIST's locks; without --cards it keeps its single-flight lock
//     (/tmp/ie-gpu.lock) and also takes cards 0..7, so a whole-machine run and a per-card run exclude each other.
//     Both export IE_GPU_LOCK_HELD (LIST, or "all") for the process they start.
//   - IE_GPU_LOCK_HELD names the cards an ancestor already holds for this process ("all" or "0,1"): those are not
//     taken again (a second open file description of the same file would conflict with the inherited one).
// `ie serve` without --cards takes no lock (as before).
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/file.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace ie {

inline std::string card_lock_dir() {
    const char* d = std::getenv("IE_CARD_LOCK_DIR");
    return d && *d ? std::string(d) : std::string("/tmp");
}
inline std::string card_lock_path(uint32_t card) { return card_lock_dir() + "/ie-card-" + std::to_string(card) + ".lock"; }

// True when IE_GPU_LOCK_HELD says an ancestor holds `card`'s lock for this process.
inline bool card_lock_inherited(uint32_t card) {
    const char* h = std::getenv("IE_GPU_LOCK_HELD");
    if (!h || !*h) return false;
    const std::string s(h);
    if (s == "all") return true;
    size_t pos = 0;
    while (pos <= s.size()) {
        const size_t comma = s.find(',', pos);
        const std::string item = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        if (item == std::to_string(card)) return true;
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return false;
}

// The locks one process holds; closing the descriptors releases them.
class CardLocks {
public:
    CardLocks() = default;
    CardLocks(const CardLocks&) = delete;
    CardLocks& operator=(const CardLocks&) = delete;
    CardLocks(CardLocks&& o) noexcept : fds_(std::move(o.fds_)) { o.fds_.clear(); }
    CardLocks& operator=(CardLocks&& o) noexcept { release(); fds_ = std::move(o.fds_); o.fds_.clear(); return *this; }
    ~CardLocks() { release(); }
    void release() { for (auto& [c, fd] : fds_) ::close(fd); fds_.clear(); }
    const std::vector<std::pair<uint32_t, int>>& fds() const { return fds_; }   // (card, descriptor), O_CLOEXEC
    std::vector<std::pair<uint32_t, int>> fds_;
};

// The pid a lock file names ("" when unreadable).
inline std::string card_lock_holder(uint32_t card) {
    const int fd = ::open(card_lock_path(card).c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return {};
    char buf[32] = {};
    const ssize_t n = ::read(fd, buf, sizeof buf - 1);
    ::close(fd);
    std::string s(buf, n > 0 ? size_t(n) : 0);
    while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
    return s;
}

// Take `cards`' locks without waiting (the inherited ones are skipped). "" = all taken (added to `out`); otherwise
// nothing new is kept and the error names the card and the holder's pid.
inline std::string acquire_card_locks(const std::vector<uint32_t>& cards, CardLocks& out) {
    CardLocks taken;
    for (uint32_t c : cards) {
        if (card_lock_inherited(c)) continue;
        const std::string path = card_lock_path(c);
        const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0666);
        if (fd < 0) return "card " + std::to_string(c) + ": cannot open lock " + path + ": " + std::strerror(errno);
        if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {
            const int e = errno;
            ::close(fd);
            if (e != EWOULDBLOCK) return "card " + std::to_string(c) + ": cannot lock " + path + ": " + std::strerror(e);
            const std::string pid = card_lock_holder(c);
            return "card " + std::to_string(c) + " is in use by another engine process" +
                   (pid.empty() ? std::string() : " (pid " + pid + ")") + " -- lock " + path +
                   "; one engine process per card (docs/serve_config.md, Card locks)";
        }
        const std::string pid = std::to_string(::getpid()) + "\n";
        if (::ftruncate(fd, 0) == 0) { ssize_t w = ::pwrite(fd, pid.data(), pid.size(), 0); (void)w; }
        taken.fds_.emplace_back(c, fd);
    }
    for (auto& p : taken.fds_) out.fds_.push_back(p);
    taken.fds_.clear();
    return {};
}

}  // namespace ie
