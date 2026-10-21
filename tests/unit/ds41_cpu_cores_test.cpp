// tests/unit/ds41_cpu_cores_test.cpp — the CPU expert leg's default cores (include/ie/deepseek41_cards.hpp).
// No GPU. This box (Core Ultra 7 265K) must keep the set the leg was tuned on, 8-19 and its halves 8-13 / 14-19;
// a homogeneous CPU (e.g. a single-socket EPYC) gets one thread per physical core past the first quarter.

#include "ie/deepseek41_cards.hpp"

#include <cstdio>
#include <sched.h>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "OK" : "FAIL", what);
    if (!ok) ++g_fail;
}
void check_eq(const std::string& got, const std::string& want, const char* what) {
    std::printf("  [%s] %s: \"%s\"%s%s%s\n", got == want ? "OK" : "FAIL", what, got.c_str(),
                got == want ? "" : " (want \"", got == want ? "" : want.c_str(), got == want ? "" : "\")");
    if (got != want) ++g_fail;
}

std::vector<int> range(int a, int b) { std::vector<int> v; for (int c = a; c <= b; ++c) v.push_back(c); return v; }
std::vector<std::vector<int>> one_thread_cores(int n) { std::vector<std::vector<int>> v; for (int c = 0; c < n; ++c) v.push_back({c}); return v; }
std::vector<std::vector<int>> smt_cores(int n) { std::vector<std::vector<int>> v; for (int c = 0; c < n; ++c) v.push_back({c, c + n}); return v; }

}  // namespace

int main() {
    using namespace ie;
    std::printf("parse / format\n");
    check(ds41_parse_cores("8-19,0,1") == [] { auto v = range(8, 19); v.push_back(0); v.push_back(1); return v; }(), "\"8-19,0,1\" parses");
    check(ds41_parse_cores("8-19\n") == range(8, 19), "a sysfs line's newline is ignored");
    check(ds41_parse_cores("").empty(), "\"\" is no cores");
    check_eq(ds41_format_cores(ds41_parse_cores("8-19,0,1")), "8-19,0-1", "format joins runs");
    check_eq(ds41_format_cores({5}), "5", "one core");

    std::printf("this box: hybrid Core Ultra 7 265K, P-cores 0-7, E-cores 8-19, no SMT\n");
    check_eq(ds41_cpu_leg_cores_from("8-19\n", one_thread_cores(20), range(0, 19)), "8-19", "the E-cores (the old default)");
    { auto allowed = range(0, 6); for (int c : range(8, 19)) allowed.push_back(c);
      check_eq(ds41_cpu_leg_cores_from("8-19\n", one_thread_cores(20), allowed), "8-19", "under taskset 0-6,8-19"); }
    check_eq(ds41_cpu_cores_part("8-19", 0, 2), "8-13", "expert parallel, card 0 (the old \"8-13\")");
    check_eq(ds41_cpu_cores_part("8-19", 1, 2), "14-19", "expert parallel, card 1 (the old \"14-19\")");
    check_eq(ds41_cpu_cores_part("8-19", 1, 2) + "," + ds41_cpu_cores_part("8-19", 0, 2), "14-19,8-13", "MiMo card 1's rotated order");
    check_eq(ds41_cpu_leg_cores_from("8-19\n", one_thread_cores(20), range(0, 7)), "2-7", "E-cores outside the mask: the P-core rule");

    std::printf("homogeneous CPUs\n");
    check_eq(ds41_cpu_leg_cores_from("", smt_cores(32), range(0, 63)), "8-31", "EPYC 32 cores / 64 threads: 24 cores, no SMT sibling");
    check_eq(ds41_cpu_leg_cores_from("", smt_cores(64), range(0, 127)), "16-63", "EPYC 64 cores / 128 threads");
    check_eq(ds41_cpu_leg_cores_from("", one_thread_cores(16), range(0, 15)), "4-15", "16 cores, SMT off");
    check_eq(ds41_cpu_leg_cores_from("", smt_cores(32), range(0, 15)), "4-15", "under taskset 0-15");
    check_eq(ds41_cpu_leg_cores_from("", smt_cores(32), range(32, 47)), "36-47", "only the siblings allowed: they are the cores");
    check_eq(ds41_cpu_leg_cores_from("", one_thread_cores(1), {0}), "0", "one core: the leg shares it");
    check_eq(ds41_cpu_cores_part("8-31", 0, 4), "8-13", "4 cards under expert parallel: card 0");
    check_eq(ds41_cpu_cores_part("8-31", 3, 4), "26-31", "... card 3");
    check_eq(ds41_cpu_cores_part("5", 2, 4), "5", "fewer cores than parts");

    std::printf("q*: the shares scaled to another box\n");
    auto near = [](float a, float b) { return a > b - 1e-4f && a < b + 1e-4f; };
    check(ds41_qstar_scale(0.30f, 1.0) == 0.30f, "factor 1: q* exactly as swept");
    check(near(ds41_qstar_scale(0.30f, 0.2), 0.0789f), "factor 0.2 (a link at a fifth): q* 0.30 -> 0.079");
    check(near(ds41_qstar_scale(0.60f, 0.2), 0.2308f), "... the parity share 0.60 -> 0.231");
    check(near(ds41_qstar_scale(0.30f, 4.0), 0.6316f), "factor 4 (a slow CPU): q* 0.30 -> 0.632");
    check(ds41_qstar_scale(0.f, 0.2) == 0.f && ds41_qstar_scale(1.f, 0.2) == 1.f, "0 and 1 stay");
    check(ds41_qstar_factor(0, 50) == 1.0 && ds41_qstar_factor(27, 0) == 1.0, "a rate unknown: factor 1");
    if (kDs41RefCpuGBs > 0) {
        check(ds41_qstar_factor(kDs41RefLinkGBs, kDs41RefCpuGBs) == 1.0, "this box's own rates: factor 1");
        check(ds41_qstar_factor(kDs41RefLinkGBs * 0.55, kDs41RefCpuGBs) == 1.0, "a link 45 % slower: still inside the band");
        check(ds41_qstar_factor(kDs41RefLinkGBs, kDs41RefCpuGBs * 0.55) == 1.0, "a leg 45 % slower: still inside the band");
        const double f = ds41_qstar_factor(6.5, kDs41RefCpuGBs * 1.5);
        std::printf("  (an x4 PCIe 4.0 riser, 6.5 GB/s, and a leg 1.5x this box's: factor %.3f, q* 0.30 -> %.3f)\n", f, ds41_qstar_scale(0.30f, f));
        check(f < 0.5 && ds41_qstar_scale(0.30f, f) < 0.10f, "x4 risers: q* well under 0.10");
    }

    std::printf("the live machine\n");
    const std::string live = ds41_cpu_leg_default_cores();
    std::printf("  ds41_cpu_leg_default_cores() = \"%s\"\n", live.c_str());
    check(!ds41_parse_cores(live).empty(), "at least one core");
    if (std::FILE* f = std::fopen("/sys/devices/cpu_atom/cpus", "r")) {
        char buf[256] = {}; const bool got = std::fgets(buf, sizeof buf, f) != nullptr; std::fclose(f);
        if (got && std::string(buf) == "8-19\n") {
            check_eq(live, "8-19", "this box: the E-cores, as before");
            // the regression of the first default-mode serve: a tier's cpu_start moves the host thread to the other cores,
            // and the NEXT card's default must not read that narrowed mask (it gave "2-7")
            cpu_set_t p; CPU_ZERO(&p); for (int c = 0; c < 8; ++c) CPU_SET(c, &p);
            if (sched_setaffinity(0, sizeof p, &p) == 0) check_eq(ds41_cpu_leg_default_cores(), "8-19", "after this thread moved to cores 0-7");
        }
    }

    std::printf("%s (%d failure(s))\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
