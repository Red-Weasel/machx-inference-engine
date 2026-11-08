// tests/unit/hugetlb_pool_test.cpp -- the hugetlb pool report the V4.1 staging tier prints with its WARNING
// (include/ie/hugetlb_pool.hpp). Host-only:  g++ -std=c++20 -O1 -Wall -Wextra -I include tests/unit/hugetlb_pool_test.cpp
#include "ie/hugetlb_pool.hpp"

#include <cstdio>
#include <string>

namespace {
int g_fail = 0;
void check(bool ok, const char* what) { std::printf("[%s] %s\n", ok ? " ok " : "FAIL", what); if (!ok) ++g_fail; }
}  // namespace

int main() {
    using namespace ie;
    HugetlbPool p;
    hugetlb_parse_meminfo("MemTotal:       261115532 kB\nHugePages_Total:     320\nHugePages_Free:      114\nHugePages_Rsvd:      112\n"
                          "HugePages_Surp:        0\nHugepagesize:       2048 kB\nHugetlb:          655360 kB\n", p);
    check(p.total == 320 && p.free == 114 && p.rsvd == 112 && p.page_kb == 2048, "meminfo: total / free / reserved / page size");
    check(p.pages(294912) == 144 && p.pages(77824) == 38, "kB -> pages");
    std::string name; uint64_t kb = 0;
    check(hugetlb_parse_status("Name:\tpostgres\nUmask:\t0077\nState:\tS (sleeping)\nHugetlbPages:\t   77824 kB\nThreads:\t1\n", name, kb) &&
          name == "postgres" && kb == 77824, "status: a holder's name and kB");
    check(!hugetlb_parse_status("Name:\tbash\nHugetlbPages:\t       0 kB\n", name, kb) && name == "bash" && kb == 0, "status: a process holding none");
    check(!hugetlb_parse_status("Name:\tkthreadd\nThreads:\t1\n", name, kb), "status: no HugetlbPages line");
    p.holders = {{"ie", 294912, 122455}, {"postgres", 77824, 7733}};
    p.self_kb = 294912;   // this engine's other tier
    check(p.others() == 320 - 114 - 144, "others = the pool minus the free pages minus this engine's: 62 (the three postgres mappings of one segment)");
    const std::string advice = hugetlb_pool_advice(p, 144);
    check(advice.find("320 pages reserved, 114 free") != std::string::npos && advice.find("112 promised") != std::string::npos, "advice: the pool");
    check(advice.find("this engine holds 144, everything else 62") != std::string::npos, "advice: this engine's pages and everything else's, exact");
    check(advice.find("ie 144 pages (pid 122455)") != std::string::npos && advice.find("postgres 38 pages (pid 7733)") != std::string::npos, "advice: the largest mappings in pages");
    check(advice.find("wanted 144") != std::string::npos && advice.find("one tier per card") != std::string::npos, "advice: what the tier wanted");
    HugetlbPool empty; empty.page_kb = 2048;
    check(hugetlb_pool_advice(empty, 144).find("largest mappings") == std::string::npos && empty.others() == 0, "advice: no holders, no mapping clause; others 0");
    HugetlbPool odd; odd.page_kb = 2048; odd.total = 10; odd.free = 8; odd.self_kb = 8192;   // free + self past the total: never negative
    check(odd.others() == 0, "others never underflows");
    // the live scan: whatever this box has, it reads without failing and the holders fit the pool
    const HugetlbPool live = hugetlb_pool_status();
    uint64_t held = 0; for (const auto& h : live.holders) held += live.pages(h.kb);
    check(live.page_kb == 0 || held <= live.total, "live: the holders do not exceed the pool");
    std::printf("  live: %llu pages, %llu free, %zu holder(s)\n", (unsigned long long)live.total, (unsigned long long)live.free, live.holders.size());
    std::printf("\nHUGETLB POOL: %s\n", g_fail ? "FAILURE(S)" : "PASS");
    return g_fail ? 1 : 0;
}
