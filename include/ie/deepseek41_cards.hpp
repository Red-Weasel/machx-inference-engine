// include/ie/deepseek41_cards.hpp — the cards DeepSeek-V4.1-Flash runs on (2026-10-20, docs/known_bugs.md §5.9).
//
// Every Level Zero GPU whose name holds "Arc" and that is not integrated graphics (Core Ultra names its iGPU "Intel(R)
// Arc(TM) Graphics", so the name alone would make it a pipeline stage), in the platform's order. ie serve and the V4.1
// tools take their cards from here, so a test runs on exactly the cards a server would.
//
// IE_DS41_LOGICAL_CARDS=k (a TEST switch) lists every card k times in a row: on two B70s, k = 2 runs the 4-stage layer
// pipeline a 4-card box runs -- layers 0-9 / 10-19 on card 0, 20-29 / 30-39 on card 1, every boundary crossed through
// the host as between real cards. Each copy gets its own single-device context and queue, so it owns its allocations
// as a separate card would; the copies of one card share its VRAM (init_resident splits the budget among the queues
// on one device) and its PCIe link, so the emulation proves the arithmetic, not the speed.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ie {

struct Ds41Cards {
    std::vector<sycl::device> devs;   // one entry per pipeline stage (a physical card repeated with IE_DS41_LOGICAL_CARDS)
    size_t   n_physical = 0;          // the discrete cards found
    size_t   n_integrated = 0;        // integrated GPUs skipped
    uint32_t logical = 1;             // IE_DS41_LOGICAL_CARDS (1 = off)
    std::string error;                // "" = usable
};

// want = 0: every card; N > 0: the first N entries (logical ones with IE_DS41_LOGICAL_CARDS), an error when fewer exist.
Ds41Cards ds41_cards(uint32_t want = 0);

// One in-order queue per entry, each on its own single-device context (the 83554c2 lesson: a two-device context
// mirrors VRAM into host RAM).
std::vector<std::unique_ptr<sycl::queue>> ds41_card_queues(const std::vector<sycl::device>& devs);

// ---- the CPU expert leg's cores (2026-10-20: 4-card boxes are often single-socket EPYCs, not this Core Ultra 7 265K) ----
// The default when neither IE_DS41_CPU_CORES nor a tier's override names them: a hybrid Intel CPU's E-cores
// (/sys/devices/cpu_atom/cpus: 8-19 here, the set the leg was tuned on and always had), else one hardware thread of
// each physical core past the first quarter -- the host, reader and stage threads keep those and every SMT sibling (an
// EPYC with 32 cores gives the leg 24). Only CPUs the process could run on when it started are chosen (its affinity
// mask, read before main: the engine narrows its own threads' masks later).
std::string ds41_cpu_leg_default_cores();
// The rule itself, for the test: atom = the cpu_atom list ("" off a hybrid CPU); cores = each physical core's hardware
// threads; allowed = the CPUs the process may run on.
std::string ds41_cpu_leg_cores_from(const std::string& atom, const std::vector<std::vector<int>>& cores,
                                    const std::vector<int>& allowed);
// Part i of n of a core list, contiguous and as even as the count allows ("8-19" in 2: "8-13" / "14-19"): each
// expert-parallel tier takes one. With fewer cores than parts, part i is core i % count.
std::string ds41_cpu_cores_part(const std::string& list, uint32_t i, uint32_t n);
std::vector<int> ds41_parse_cores(const std::string& list);    // "8-19,0,1" -> {8..19, 0, 1}
std::string ds41_format_cores(const std::vector<int>& cores);  // {8..13, 0} -> "8-13,0"

// ---- q*, the pinned misses' PCIe share, for another box (2026-10-20: cards on x4 risers) ----
// The CPU leg's shares (q* 0.30 one row, 0.20 a verify, 0.30 a continuation's cost ratio) were swept on this box: links of
// 26-28 GB/s host->card from pinned memory (init_resident's probe) and a leg of kDs41RefCpuGBs on its 12 E-cores (the
// tier's probe). Each share is a rate ratio -- q / (1 - q) = what PCIe moves / what the CPU computes per ms -- so another
// box scales each one's odds by (its link / this box's) x (this box's leg / its leg). Within a factor of 2 of this box
// the factor is 1 and every share stays exactly as swept (probe noise never moves them). Expert parallel's 0.60 is left
// alone: it was swept with half the E-cores per tier, a reference not measured.
constexpr double kDs41RefLinkGBs = 27.0;
constexpr double kDs41RefCpuGBs  = 60.0;  // 59.2 / 61.0 measured on the two tiers, 2026-10-20 (srv/runs/d3_new_2c_default)
double ds41_qstar_factor(double link_gbs, double cpu_gbs);   // 1 inside the band, or when either rate is unknown
float  ds41_qstar_scale(float q, double factor);             // q's odds times factor

}  // namespace ie
