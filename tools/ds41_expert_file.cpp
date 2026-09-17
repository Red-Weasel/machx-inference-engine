// tools/ds41_expert_file.cpp — ie-ds41-expert-file: write the mmap tier's experts (the ranking's tail from
// --cutoff on, every layer) in the engine's slot layout to one file, so a fill is one O_DIRECT pread into
// the staging slot instead of pread + permute (docs/deepseek41/40, C2). The slots are the bytes
// ds41_slot_pack_pread produces, byte for byte.
//   usage: ie-ds41-expert-file <model> <ranking> [--cutoff N=297] [--end M=N+44] [--out PATH=<model>/ie_experts_tail.ieslot]
//          [--threads T=8] --confirm
//   The window [cutoff, end) of every layer's ranking: the hot part of the mmap tier's tail (the ranking is by
//   frequency, so a pass touches mostly its head); the whole tail (~60 GB) would leave the system volume under the
//   50 GB floor, and an mmap expert the file does not hold takes the pack path (the init line prints the coverage).
//   The tool prints the projected size and the free space it would leave on the volume and refuses to
//   write without --confirm or if the volume would be left under 50 GB (the main NVMe is the system volume).
#include "ie/deepseek41.hpp"
#include "ie/deepseek41_experts.hpp"
#include "ie/expert_stream.hpp"

#include <atomic>
#include <mutex>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/statvfs.h>
#include <thread>
#include <unistd.h>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: ie-ds41-expert-file <model> <ranking> [--cutoff N] [--out PATH] [--threads T] --confirm\n"); return 2; }
    const std::string dir = argv[1], ranking = argv[2];
    uint32_t cutoff = 297, end = 0, nthreads = 8; std::string out = dir + "/" + "ie_experts_tail.ieslot"; bool confirm = false;
    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--cutoff" && i + 1 < argc) cutoff = uint32_t(std::atoi(argv[++i]));
        else if (a == "--end" && i + 1 < argc) end = uint32_t(std::atoi(argv[++i]));
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--threads" && i + 1 < argc) nthreads = uint32_t(std::atoi(argv[++i]));
        else if (a == "--confirm") confirm = true;
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    ie::DeepSeek41Model m; if (auto e = m.load(dir); !e.empty()) { std::fprintf(stderr, "load: %s\n", e.c_str()); return 1; }
    const auto& c = m.config(); const uint32_t NL = c.n_layers, E = c.n_routed_experts;
    if (!end) end = cutoff + 44; if (end > E) end = E;
    if (cutoff >= end) { std::fprintf(stderr, "the window [%u, %u) is empty (expert count %u)\n", cutoff, end, E); return 2; }
    ie::Ds4SlotLayout lay; if (auto e = ie::ds41_slot_layout(c.dim, c.moe_inter_dim, lay); !e.empty()) { std::fprintf(stderr, "layout: %s\n", e.c_str()); return 1; }
    if (lay.bytes % 4096) { std::fprintf(stderr, "the slot (%llu bytes) is not a 4 KiB multiple; O_DIRECT reads of it would need padding -- not written\n", (unsigned long long)lay.bytes); return 1; }
    std::vector<std::vector<uint32_t>> rank;
    if (auto e = ie::ds4_expert_priority_read_layers(ranking, E, NL, rank); !e.empty()) { std::fprintf(stderr, "ranking: %s\n", e.c_str()); return 1; }
    const uint64_t per_layer = end - cutoff, n_slots = uint64_t(NL) * per_layer;
    const uint64_t table_off = 4096, table_bytes = uint64_t(NL) * E * 8, slots_off = ((table_off + table_bytes + 4095) / 4096) * 4096;
    const uint64_t total = slots_off + n_slots * lay.bytes;
    struct statvfs sv{}; const std::string vol = out.substr(0, out.find_last_of('/'));
    if (statvfs(vol.c_str(), &sv) != 0) { std::perror("statvfs"); return 1; }
    const double free_gb = double(sv.f_bavail) * sv.f_frsize / 1e9, size_gb = double(total) / 1e9, left_gb = free_gb - size_gb;
    std::printf("tail window: ranking positions %u..%u of every layer = %llu experts per layer, %llu slots of %.1f MiB = %.1f GB\n", cutoff, end - 1, (unsigned long long)per_layer, (unsigned long long)n_slots, lay.bytes / 1048576.0, size_gb);
    std::printf("volume %s: %.1f GB free now, %.1f GB after the write\n", vol.c_str(), free_gb, left_gb);
    if (left_gb < 50.0) { std::fprintf(stderr, "REFUSED: the volume would be left under 50 GB\n"); return 3; }
    if (!confirm) { std::fprintf(stderr, "not written: pass --confirm to write %s\n", out.c_str()); return 4; }

    const int fd = open(out.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT, 0644);
    if (fd < 0) { std::perror("open out"); return 1; }
    // the header + table, in one page-aligned buffer
    std::vector<uint64_t> table(size_t(NL) * E, 0);
    for (uint32_t l = 0; l < NL; ++l) for (uint32_t i = cutoff; i < end; ++i) table[size_t(l) * E + rank[l][i]] = slots_off + (uint64_t(l) * per_layer + (i - cutoff)) * lay.bytes;
    void* head = nullptr; if (posix_memalign(&head, 4096, size_t(slots_off)) != 0) return 1; std::memset(head, 0, size_t(slots_off));
    ie::Ds41ExpertFileHeader hd{}; std::memcpy(hd.magic, ie::kDs41ExpertFileMagic, 8); hd.H = c.dim; hd.EF = c.moe_inter_dim; hd.n_layers = NL; hd.n_experts = E;
    hd.slot_bytes = lay.bytes; hd.table_off = table_off; hd.n_slots = n_slots; hd.cutoff = cutoff;
    std::memcpy(head, &hd, sizeof hd); std::memcpy(static_cast<uint8_t*>(head) + table_off, table.data(), table_bytes);
    if (pwrite(fd, head, size_t(slots_off), 0) != ssize_t(slots_off)) { std::perror("pwrite header"); return 1; }
    // the slots, packed by the same path the tier uses, T threads
    std::atomic<uint64_t> next{0}, done{0}; std::atomic<bool> failed{false}; std::string first_err; std::mutex em;
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> ts;
    for (uint32_t t = 0; t < nthreads; ++t) ts.emplace_back([&] {
        void* buf = nullptr; void* bounce = nullptr;
        if (posix_memalign(&buf, 4096, size_t(lay.bytes)) != 0 || posix_memalign(&bounce, 4096, size_t(ie::ds41_pack_bounce_bytes(lay))) != 0) { failed = true; return; }
        for (;;) {
            const uint64_t s = next.fetch_add(1); if (s >= n_slots || failed) break;
            const uint32_t l = uint32_t(s / per_layer), i = cutoff + uint32_t(s % per_layer), e = rank[l][i];
            const auto& Lw = m.layers()[l];
            if (auto er = ie::ds41_slot_pack_pread(lay, m.store(), Lw.exp_w1[e], Lw.exp_w3[e], Lw.exp_w2[e], buf, bounce); !er.empty()) { std::lock_guard<std::mutex> g(em); if (first_err.empty()) first_err = "layer " + std::to_string(l) + " expert " + std::to_string(e) + ": " + er; failed = true; break; }
            const uint64_t off = table[size_t(l) * E + e];
            if (pwrite(fd, buf, size_t(lay.bytes), off_t(off)) != ssize_t(lay.bytes)) { std::lock_guard<std::mutex> g(em); if (first_err.empty()) first_err = std::string("pwrite: ") + std::strerror(errno); failed = true; break; }
            const uint64_t d = ++done; if (d % 250 == 0 || d == n_slots) { const double s_el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); std::printf("  %llu / %llu slots, %.1f s, %.2f GB/s written\r", (unsigned long long)d, (unsigned long long)n_slots, s_el, d * double(lay.bytes) / s_el / 1e9); std::fflush(stdout); }
        }
        std::free(buf); std::free(bounce);
    });
    for (auto& t : ts) t.join();
    std::printf("\n");
    if (failed) { std::fprintf(stderr, "FAILED: %s -- the file is incomplete; removing it\n", first_err.c_str()); close(fd); unlink(out.c_str()); return 1; }
    if (fsync(fd) != 0) { std::perror("fsync"); return 1; }
    close(fd);
    const double s_el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("wrote %s: %.1f GB in %.1f s (%.2f GB/s)\n", out.c_str(), size_gb, s_el, size_gb / s_el);
    return 0;
}
