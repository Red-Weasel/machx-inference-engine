// include/ie/hugetlb_pool.hpp -- the host's hugetlb pool as the V4.1 staging tier sees it (2026-11-05, known_bugs 46's
// neighbour): how many 2 MiB pages the kernel reserved, how many are free, and WHO holds the rest. The tier's WARNING used
// to say "Reserve vm.nr_hugepages >= 144" while the pool already held 320 -- a PostgreSQL in a container had 141 of them,
// so one tier got its pages and the other ran on movable memory for a week without anyone seeing why. Header-only, host:
// tests/unit/hugetlb_pool_test.cpp (the parsers on captured text; the live scan reads /proc).
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace ie {

struct HugetlbPool {
    uint64_t total = 0, free = 0, rsvd = 0;   // pages
    uint64_t page_kb = 0;                     // Hugepagesize
    uint64_t self_kb = 0;                     // this process's own HugetlbPages (its other tiers' slots)
    struct Holder { std::string name; uint64_t kb = 0; int pid = 0; };
    std::vector<Holder> holders;              // the largest per-process mappings (a shared segment shows in each process)
    uint64_t pages(uint64_t kb) const { return page_kb ? kb / page_kb : 0; }
    // the pages neither free nor this process's: what everything else on the box holds (exact, unlike the mappings)
    uint64_t others() const { const uint64_t mine = pages(self_kb); return total > free + mine ? total - free - mine : 0; }
};

// `MemInfo` text (/proc/meminfo): HugePages_Total / HugePages_Free / HugePages_Rsvd / Hugepagesize.
inline void hugetlb_parse_meminfo(std::string_view text, HugetlbPool& p) {
    const auto field = [&](std::string_view key) -> uint64_t {
        const auto at = text.find(key);
        if (at == std::string_view::npos) return 0;
        size_t i = at + key.size();
        while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) ++i;
        uint64_t v = 0;
        while (i < text.size() && text[i] >= '0' && text[i] <= '9') v = v * 10 + uint64_t(text[i++] - '0');
        return v;
    };
    p.total = field("HugePages_Total:"); p.free = field("HugePages_Free:"); p.rsvd = field("HugePages_Rsvd:");
    p.page_kb = field("Hugepagesize:");
}

// One /proc/<pid>/status text: the process name and its HugetlbPages in kB. False when it holds none.
inline bool hugetlb_parse_status(std::string_view text, std::string& name, uint64_t& kb) {
    name.clear(); kb = 0;
    if (const auto at = text.find("Name:"); at != std::string_view::npos) {
        size_t i = at + 5;
        while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) ++i;
        const auto end = text.find('\n', i);
        name = std::string(text.substr(i, end == std::string_view::npos ? std::string_view::npos : end - i));
    }
    const auto at = text.find("HugetlbPages:");
    if (at == std::string_view::npos) return false;
    size_t i = at + 13;
    while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) ++i;
    while (i < text.size() && text[i] >= '0' && text[i] <= '9') kb = kb * 10 + uint64_t(text[i++] - '0');
    return kb > 0;
}

// The live pool: /proc/meminfo and every /proc/<pid>/status (readable ones), the largest `max_holders` holders kept.
inline HugetlbPool hugetlb_pool_status(size_t max_holders = 4) {
    HugetlbPool p;
    { std::ifstream f("/proc/meminfo"); std::stringstream ss; ss << f.rdbuf(); hugetlb_parse_meminfo(ss.str(), p); }
    { std::ifstream f("/proc/self/status"); std::stringstream ss; ss << f.rdbuf(); std::string n; hugetlb_parse_status(ss.str(), n, p.self_kb); }
    if (DIR* d = opendir("/proc")) {
        while (const dirent* e = readdir(d)) {
            if (e->d_name[0] < '1' || e->d_name[0] > '9') continue;
            std::ifstream f(std::string("/proc/") + e->d_name + "/status");
            if (!f) continue;
            std::stringstream ss; ss << f.rdbuf();
            std::string name; uint64_t kb = 0;
            if (hugetlb_parse_status(ss.str(), name, kb)) p.holders.push_back({name, kb, std::atoi(e->d_name)});
        }
        closedir(d);
    }
    std::sort(p.holders.begin(), p.holders.end(), [](const auto& a, const auto& b) { return a.kb > b.kb; });
    if (p.holders.size() > max_holders) p.holders.resize(max_holders);
    return p;
}

// One line for the operator: the pool, who holds it, and what this tier wanted. `need_pages` = this tier's pages.
inline std::string hugetlb_pool_advice(const HugetlbPool& p, uint64_t need_pages) {
    std::string s = "the pool: " + std::to_string(p.total) + " pages reserved, " + std::to_string(p.free) + " free";
    if (p.rsvd) s += " (" + std::to_string(p.rsvd) + " promised)";
    s += ", this engine holds " + std::to_string(p.pages(p.self_kb)) + ", everything else " + std::to_string(p.others());
    s += "; this tier wanted " + std::to_string(need_pages) + " and there is one tier per card";
    if (!p.holders.empty()) {
        s += "; the largest mappings:";
        for (size_t i = 0; i < p.holders.size(); ++i)
            s += (i ? ", " : " ") + p.holders[i].name + " " + std::to_string(p.pages(p.holders[i].kb)) + " pages (pid " + std::to_string(p.holders[i].pid) + ")";
    }
    s += ". vm.nr_hugepages must cover every card's tier plus what everything else holds, or the tier runs on movable memory.";
    return s;
}

}  // namespace ie
