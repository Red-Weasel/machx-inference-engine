// src/model/deepseek41_cards.cpp — the cards DeepSeek-V4.1-Flash runs on (include/ie/deepseek41_cards.hpp).
#include "ie/deepseek41_cards.hpp"

#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <sched.h>
#include <unistd.h>

namespace ie {

namespace {
// ZE_DEVICE_PROPERTY_FLAG_INTEGRATED, asked through dlsym as in deepseek41_experts.cpp; when the loader cannot be
// asked the device counts as discrete (the behaviour before the check)
bool integrated_gpu(const sycl::device& d) {
    void* lib = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_NOLOAD);
    if (!lib) lib = dlopen("libze_loader.so.1", RTLD_NOW);
    auto get_props = lib ? reinterpret_cast<ze_result_t (*)(ze_device_handle_t, ze_device_properties_t*)>(
                               dlsym(lib, "zeDeviceGetProperties")) : nullptr;
    if (!get_props) return false;
    ze_device_properties_t p{};
    p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    if (get_props(sycl::get_native<sycl::backend::ext_oneapi_level_zero>(d), &p) != ZE_RESULT_SUCCESS) return false;
    return (p.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) != 0;
}
}  // namespace

Ds41Cards ds41_cards(uint32_t want) {
    Ds41Cards r;
    std::vector<sycl::device> phys;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
        for (const auto& d : p.get_devices(sycl::info::device_type::gpu)) {
            if (d.get_info<sycl::info::device::name>().find("Arc") == std::string::npos) continue;
            if (integrated_gpu(d)) { ++r.n_integrated; continue; }
            phys.push_back(d);
        }
    }
    r.n_physical = phys.size();
    if (phys.empty()) { r.error = r.n_integrated ? "no discrete Arc GPU (only integrated graphics)" : "no Arc GPU"; return r; }
    if (const char* v = std::getenv("IE_DS41_LOGICAL_CARDS"); v && *v) {
        const long k = std::atol(v);
        if (k < 1 || k > 8) { r.error = "IE_DS41_LOGICAL_CARDS=" + std::string(v) + ": expected 1..8"; return r; }
        r.logical = uint32_t(k);
    }
    for (const auto& d : phys) for (uint32_t i = 0; i < r.logical; ++i) r.devs.push_back(d);
    if (r.logical > 1)
        std::fprintf(stderr, "[ds41] IE_DS41_LOGICAL_CARDS=%u (a test switch): each of the %zu card(s) runs as %u pipeline stages, "
                             "%zu in all; the copies of a card share its VRAM and its link\n", r.logical, phys.size(), r.logical, r.devs.size());
    if (want) {
        if (want > r.devs.size()) {
            r.error = "--gpus " + std::to_string(want) + ": only " + std::to_string(r.devs.size()) + " card(s) visible" +
                      (r.logical > 1 ? " (with IE_DS41_LOGICAL_CARDS)" : "") + "; `ie cards` lists them";
            r.devs.clear(); return r;
        }
        r.devs.resize(want);
    }
    return r;
}

std::vector<std::unique_ptr<sycl::queue>> ds41_card_queues(const std::vector<sycl::device>& devs) {
    std::vector<std::unique_ptr<sycl::queue>> qs;
    for (const auto& d : devs) qs.push_back(std::make_unique<sycl::queue>(sycl::context(d), d, sycl::property_list{sycl::property::queue::in_order{}}));
    return qs;
}

std::vector<int> ds41_parse_cores(const std::string& list) {
    std::vector<int> out;
    for (size_t i = 0; i < list.size();) {
        size_t j = list.find(',', i); if (j == std::string::npos) j = list.size();
        const std::string tok = list.substr(i, j - i); const size_t d = tok.find('-');
        if (tok.find_first_of("0123456789") == std::string::npos) { i = j + 1; continue; }   // "" or a stray "\n"
        if (d == std::string::npos) out.push_back(std::atoi(tok.c_str()));
        else for (int c = std::atoi(tok.substr(0, d).c_str()); c <= std::atoi(tok.substr(d + 1).c_str()); ++c) out.push_back(c);
        i = j + 1;
    }
    return out;
}

std::string ds41_format_cores(const std::vector<int>& cores) {
    std::string s;
    for (size_t i = 0; i < cores.size();) {
        size_t j = i + 1;
        while (j < cores.size() && cores[j] == cores[j - 1] + 1) ++j;
        if (!s.empty()) s += ',';
        s += std::to_string(cores[i]);
        if (j - i > 1) s += '-' + std::to_string(cores[j - 1]);
        i = j;
    }
    return s;
}

std::string ds41_cpu_leg_cores_from(const std::string& atom, const std::vector<std::vector<int>>& cores,
                                    const std::vector<int>& allowed) {
    auto ok = [&](int c) { return std::find(allowed.begin(), allowed.end(), c) != allowed.end(); };
    std::vector<int> pick;
    for (int c : ds41_parse_cores(atom)) if (ok(c)) pick.push_back(c);
    if (!pick.empty()) return ds41_format_cores(pick);
    std::vector<int> first;                                  // one allowed thread per physical core, in core order
    for (const auto& core : cores)
        for (int c : core) if (ok(c)) { first.push_back(c); break; }
    std::sort(first.begin(), first.end());
    const size_t skip = first.size() > 1 ? first.size() / 4 : 0;
    return ds41_format_cores(std::vector<int>(first.begin() + std::ptrdiff_t(skip), first.end()));
}

namespace {
// The CPUs the process may run on, read once before main: the engine narrows its own threads' masks later (the first
// tier's cpu_start moves the host thread off the leg's cores), and a mask read then made the second card's leg "2-7"
// (caught 2026-10-20 in the first default-mode serve of this rule)
const std::vector<int> g_allowed_at_start = [] {
    std::vector<int> v;
    cpu_set_t set; CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) == 0) { for (int c = 0; c < CPU_SETSIZE; ++c) if (CPU_ISSET(c, &set)) v.push_back(c); }
    else for (int c = 0; c < int(sysconf(_SC_NPROCESSORS_ONLN)); ++c) v.push_back(c);
    return v;
}();
}  // namespace

std::string ds41_cpu_leg_default_cores() {
    auto read = [](const std::string& path) {
        std::string s;
        if (std::FILE* f = std::fopen(path.c_str(), "r")) { char buf[4096]; if (std::fgets(buf, sizeof buf, f)) s = buf; std::fclose(f); }
        return s;
    };
    const std::vector<int>& allowed = g_allowed_at_start;
    std::vector<std::vector<int>> cores;
    std::vector<bool> seen(CPU_SETSIZE, false);
    for (int c : allowed) {
        if (seen[size_t(c)]) continue;
        std::vector<int> sib = ds41_parse_cores(read("/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/thread_siblings_list"));
        if (std::find(sib.begin(), sib.end(), c) == sib.end()) sib = {c};   // no topology: the CPU is its own core
        for (int s : sib) if (s >= 0 && s < CPU_SETSIZE) seen[size_t(s)] = true;
        cores.push_back(sib);
    }
    return ds41_cpu_leg_cores_from(read("/sys/devices/cpu_atom/cpus"), cores, allowed);
}

double ds41_qstar_factor(double link_gbs, double cpu_gbs) {
    if (!(link_gbs > 0) || !(cpu_gbs > 0) || !(kDs41RefCpuGBs > 0)) return 1.0;
    const double f = (link_gbs / kDs41RefLinkGBs) * (kDs41RefCpuGBs / cpu_gbs);
    return f >= 0.5 && f <= 2.0 ? 1.0 : f;
}

float ds41_qstar_scale(float q, double factor) {
    if (factor == 1.0 || !(factor > 0) || q <= 0.f || q >= 1.f) return q;
    const double o = double(q) / (1.0 - double(q)) * factor;
    return float(o / (1.0 + o));
}

std::string ds41_cpu_cores_part(const std::string& list, uint32_t i, uint32_t n) {
    const std::vector<int> all = ds41_parse_cores(list);
    if (all.empty() || n == 0) return list;
    if (all.size() < n) return std::to_string(all[i % all.size()]);
    return ds41_format_cores(std::vector<int>(all.begin() + std::ptrdiff_t(all.size() * i / n), all.begin() + std::ptrdiff_t(all.size() * (i + 1) / n)));
}

}  // namespace ie
