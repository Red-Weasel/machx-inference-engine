// tests/unit/gguf_madvise_test.cpp — GgufReader must leave kernel readahead ON.
//
// WHY THIS TEST EXISTS
// --------------------
// GgufReader::open used to advise every shard mapping MADV_RANDOM ("tensor data
// is touched sparsely").  MADV_RANDOM sets the VMA's ra_pages to zero, which
// turns kernel readahead OFF: every mmap fault then fetches exactly one page and
// blocks on it.  That is wrong for BOTH regions of a GGUF:
//
//   * the header region (KV table + tensor-info table) is walked strictly
//     front-to-back by parse_shard itself, on every open, for every shard;
//   * the tensor-data region is streamed whole-tensor, ascending, by every
//     model loader in the engine.
//
// Neither is a sparse access pattern, so the hint only ever cost throughput.
// Measured on this box the difference was ~0.04 MB/s vs the drive's sequential
// rate — a serial chain of one-page reads.
//
// WHAT IS ASSERTED
// ----------------
// Only the deterministic part: that the mappings GgufReader hands out do NOT
// carry VM_RAND_READ.  Linux reports that as "rr" in the VmFlags line of
// /proc/self/smaps, so this is an exact, non-timing check that cannot flake on a
// loaded box.  The throughput figures below it are printed as diagnostics and
// are deliberately NOT asserted: this box's model lives on a contended spinning
// USB disk and any rate threshold would be noise, not a regression signal.
//
// No GPU, no model load — this is mmap + header parse + a bounded byte read.
#undef NDEBUG  // build is Release (-DNDEBUG); asserts must stay live here
#include "ie/gguf.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Shard 1 is metadata-only (5 MB, 0 tensors); opening it makes GgufReader map
// and header-parse all four shards, which is exactly the path this guards.
constexpr const char* kDs4Dir =
    "${IE_MODELS_DIR}/DeepSeek-V4-Flash-0731-GGUF/UD-Q3_K_XL/";
constexpr const char* kDs4Stem = "DeepSeek-V4-Flash-0731-UD-Q3_K_XL-0000";

std::string shard_path(int n) {
    return std::string(kDs4Dir) + kDs4Stem + char('0' + n) + "-of-00004.gguf";
}

// One /proc/self/smaps entry, header line plus the VmFlags line.
struct MapEntry {
    uint64_t    start = 0, end = 0, pgoff = 0;
    std::string path;
    std::string vmflags;
};

// Parse /proc/self/smaps. Only the header lines ("start-end perms offset ...")
// and VmFlags lines are kept; every other field is skipped.
std::vector<MapEntry> read_smaps() {
    std::vector<MapEntry> out;
    std::FILE* f = std::fopen("/proc/self/smaps", "r");
    if (!f) return out;
    char line[4096];
    while (std::fgets(line, sizeof(line), f)) {
        uint64_t s = 0, e = 0, off = 0;
        char perms[8] = {0};
        int path_pos = 0;
        // Header lines are the only ones matching "<hex>-<hex> <perms> <hex> ...".
        if (std::sscanf(line, "%lx-%lx %7s %lx %*s %*s %n", &s, &e, perms, &off,
                        &path_pos) >= 4) {
            MapEntry m;
            m.start = s; m.end = e; m.pgoff = off;
            if (path_pos > 0) {
                std::string p(line + path_pos);
                while (!p.empty() && (p.back() == '\n' || p.back() == ' ')) p.pop_back();
                m.path = p;
            }
            out.push_back(std::move(m));
        } else if (std::strncmp(line, "VmFlags:", 8) == 0 && !out.empty()) {
            std::string v(line + 8);
            while (!v.empty() && (v.back() == '\n' || v.back() == ' ')) v.pop_back();
            out.back().vmflags = v;
        }
    }
    std::fclose(f);
    return out;
}

// The smaps entry whose address range contains `addr`, or nullptr.
const MapEntry* entry_for(const std::vector<MapEntry>& v, const void* addr) {
    const auto a = reinterpret_cast<uint64_t>(addr);
    for (const auto& m : v) if (a >= m.start && a < m.end) return &m;
    return nullptr;
}

// True if VmFlags advertises VM_RAND_READ, i.e. MADV_RANDOM is in force. The
// token is exactly "rr"; match on whole space-separated tokens so that no other
// flag can alias it.
bool has_rand_read(const std::string& vmflags) {
    size_t i = 0;
    while (i < vmflags.size()) {
        while (i < vmflags.size() && vmflags[i] == ' ') ++i;
        size_t j = i;
        while (j < vmflags.size() && vmflags[j] != ' ') ++j;
        if (vmflags.compare(i, j - i, "rr") == 0) return true;
        i = j;
    }
    return false;
}

// Drop `len` bytes at `off` of `path` from the page cache so the next read of
// that range actually reaches the disk. Clean, unmapped page-cache pages only —
// which is what an untouched tensor region is.
void evict(const std::string& path, uint64_t off, uint64_t len) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return;
    ::posix_fadvise(fd, off_t(off), off_t(len), POSIX_FADV_DONTNEED);
    ::close(fd);
}

// Device-level bytes this process has read, from /proc/self/io. This counts what
// actually came off the block device, so readahead over-reads are included and
// page-cache hits are not.
uint64_t read_bytes_now() {
    std::FILE* f = std::fopen("/proc/self/io", "r");
    if (!f) return 0;
    char line[256];
    uint64_t v = 0;
    while (std::fgets(line, sizeof(line), f))
        if (std::sscanf(line, "read_bytes: %lu", &v) == 1) break;
    std::fclose(f);
    return v;
}

double secs_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main() {
    // Line-buffer stdout: the diagnostics below are the whole point of a failing
    // run, and a full buffer would be discarded by assert()'s abort().
    ::setvbuf(stdout, nullptr, _IOLBF, 0);

    // ---- cold header parse ------------------------------------------------
    // Evict a generous prefix of every shard so the header walk that open()
    // performs (KV table for shard 1, tensor-info table for each shard) is read
    // from the disk rather than from cache.
    for (int n = 1; n <= 4; ++n) evict(shard_path(n), 0, 256ull << 20);

    const uint64_t io0 = read_bytes_now();
    const auto     t0  = std::chrono::steady_clock::now();

    ie::GgufReader g;
    const std::string err = g.open(shard_path(1));

    const double   open_s  = secs_since(t0);
    const uint64_t open_io = read_bytes_now() - io0;

    if (!err.empty()) {
        std::fprintf(stderr, "gguf_madvise_test: SKIP (cannot open %s: %s)\n",
                     shard_path(1).c_str(), err.c_str());
        std::puts("gguf_madvise_test: SKIPPED");
        return 0;
    }

    std::printf("header parse (4 shards, cold): %.2f s, %.2f MB off disk\n",
                open_s, double(open_io) / (1 << 20));
    std::printf("  tensors=%llu kvs=%llu\n",
                (unsigned long long)g.n_tensors(), (unsigned long long)g.n_kv());

    // ---- bounded cold stream through the mapping --------------------------
    // Pick a file range deep inside shard 2's tensor data, evict it, then walk it
    // one byte per page in ascending order — the same shape of access every model
    // loader makes over a weight tensor.
    constexpr uint64_t kOff    = 4ull << 30;   // 4 GiB into a 49 GiB shard
    constexpr uint64_t kWindow = 256ull << 20;
    constexpr uint64_t kBudget = 32ull << 20;  // stop after this many bytes ...
    constexpr double   kMaxSec = 8.0;          // ... or this long, whichever first

    evict(shard_path(2), kOff, kWindow);

    // Map file offset -> address via the reader's own mapping of shard 2, found
    // by path in smaps. GgufReader mapped the whole file at offset 0, but the
    // VMA may have been split, so match on the containing pgoff range.
    const auto  maps  = read_smaps();
    const auto  sp2   = shard_path(2);
    const void* start = nullptr;
    uint64_t    avail = 0;
    for (const auto& m : maps) {
        const uint64_t span = m.end - m.start;
        if (m.path == sp2 && kOff >= m.pgoff && kOff < m.pgoff + span) {
            start = reinterpret_cast<const void*>(m.start + (kOff - m.pgoff));
            avail = m.pgoff + span - kOff;
            break;
        }
    }

    if (start) {
        const long pg   = ::sysconf(_SC_PAGESIZE);
        const uint64_t n = (avail < kBudget) ? avail : kBudget;
        const auto* p    = static_cast<const volatile uint8_t*>(start);

        const uint64_t s_io0 = read_bytes_now();
        const auto     s_t0  = std::chrono::steady_clock::now();
        uint64_t touched = 0;
        uint8_t  sink    = 0;
        for (uint64_t i = 0; i < n; i += uint64_t(pg)) {
            sink ^= p[i];
            touched = i + uint64_t(pg);
            // Check the clock every 16 pages: often enough that kMaxSec is a real
            // bound even when each fault costs tens of ms, rare enough to stay
            // free once the pages are warm.
            if ((i & (16 * uint64_t(pg) - 1)) == 0 && secs_since(s_t0) > kMaxSec) break;
        }
        const double   s_s  = secs_since(s_t0);
        const uint64_t s_io = read_bytes_now() - s_io0;

        std::printf("stream (cold, ascending, 1 byte/page): %.3f MB of mapping in %.2f s\n",
                    double(touched) / (1 << 20), s_s);
        std::printf("  off disk: %.3f MB  ->  %.4f MB/s   (sink=%u)\n",
                    double(s_io) / (1 << 20), (double(s_io) / (1 << 20)) / s_s,
                    unsigned(sink));
    } else {
        std::printf("stream: SKIP (no smaps entry for shard 2 covering offset %llu)\n",
                    (unsigned long long)kOff);
    }

    // ---- the actual assertion ---------------------------------------------
    // Readahead must be ON for the primary mapping (reached via a KV payload,
    // shard 1 has no tensors) and for a secondary shard mapping (reached via any
    // tensor, since every tensor of this model lives in shards 2..4).
    assert(g.n_kv() > 0);
    const auto smaps = read_smaps();

    const void*     kv_p  = g.kvs()[0].payload;
    const MapEntry* prim  = entry_for(smaps, kv_p);
    assert(prim && "no smaps entry for the primary shard mapping");
    std::printf("primary   VmFlags:%s\n", prim->vmflags.c_str());
    assert(!has_rand_read(prim->vmflags) &&
           "GgufReader left MADV_RANDOM (VmFlags rr) on the primary mapping — "
           "kernel readahead is disabled and every fault costs one 4 KiB read");

    assert(g.n_tensors() > 0);
    const void*     t_p   = g.tensors()[0].data;
    const MapEntry* sec   = entry_for(smaps, t_p);
    assert(sec && "no smaps entry for the secondary shard mapping");
    std::printf("secondary VmFlags:%s\n", sec->vmflags.c_str());
    assert(!has_rand_read(sec->vmflags) &&
           "GgufReader left MADV_RANDOM (VmFlags rr) on a split-shard mapping");

    std::puts("gguf_madvise_test: all OK (readahead enabled on every shard mapping)");
    return 0;
}
