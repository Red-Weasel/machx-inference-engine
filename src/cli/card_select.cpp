// --cards: map card numbers to the runtime's Level Zero indices, pin the process to them, verify (ie/card_select.hpp).
#include "ie/card_select.hpp"
#include <level_zero/ze_api.h>
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace ie {

std::string enumerate_level_zero_gpus(std::vector<LevelZeroGpu>& out) {
    out.clear();
    if (ze_result_t r = zeInit(ZE_INIT_FLAG_GPU_ONLY); r != ZE_RESULT_SUCCESS)
        return "Level Zero init failed (0x" + std::to_string(uint32_t(r)) + ")";
    uint32_t n_drv = 0;
    if (zeDriverGet(&n_drv, nullptr) != ZE_RESULT_SUCCESS) return "zeDriverGet failed";
    std::vector<ze_driver_handle_t> drivers(n_drv);
    if (n_drv && zeDriverGet(&n_drv, drivers.data()) != ZE_RESULT_SUCCESS) return "zeDriverGet failed";
    uint32_t index = 0;   // the runtime numbers the backend's devices across its drivers, in this order
    for (auto drv : drivers) {
        uint32_t n = 0;
        if (zeDeviceGet(drv, &n, nullptr) != ZE_RESULT_SUCCESS) return "zeDeviceGet failed";
        std::vector<ze_device_handle_t> devs(n);
        if (n && zeDeviceGet(drv, &n, devs.data()) != ZE_RESULT_SUCCESS) return "zeDeviceGet failed";
        for (auto d : devs) {
            ze_device_properties_t p{};
            p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            if (zeDeviceGetProperties(d, &p) != ZE_RESULT_SUCCESS) return "zeDeviceGetProperties failed";
            if (p.type != ZE_DEVICE_TYPE_GPU) continue;
            ze_pci_ext_properties_t pci{};
            pci.stype = ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES;
            if (zeDevicePciGetPropertiesExt(d, &pci) != ZE_RESULT_SUCCESS)
                return std::string("cannot read the PCI address of ") + p.name;
            char bdf[32];
            std::snprintf(bdf, sizeof bdf, "%04x:%02x:%02x.%x", pci.address.domain, pci.address.bus,
                          pci.address.device, pci.address.function);
            out.push_back({index++, bdf, p.name, (p.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED) != 0});
        }
    }
    return {};
}

std::string apply_card_selection(const std::vector<uint32_t>& cards, std::vector<std::string>& pcis) {
    // A selector or affinity mask from the environment already renumbers or hides devices; combined with --cards the
    // numbers would no longer mean the physical cards. "level_zero:gpu" / "level_zero:*" hide nothing and are replaced.
    if (const char* s = std::getenv("ONEAPI_DEVICE_SELECTOR"); s && *s) {
        const std::string v = s;
        if (v != "level_zero:gpu" && v != "level_zero:*")
            return "--cards cannot be combined with ONEAPI_DEVICE_SELECTOR=" + v + " (unset it; --cards sets it)";
    }
    if (const char* m = std::getenv("ZE_AFFINITY_MASK"); m && *m)
        return std::string("--cards cannot be combined with ZE_AFFINITY_MASK=") + m + " (unset it)";
    std::vector<LevelZeroGpu> gpus;
    if (auto e = enumerate_level_zero_gpus(gpus); !e.empty()) return "--cards: " + e;
    std::string err;
    const std::string sel = card_selector_for(gpus, cards, err, &pcis);
    if (sel.empty()) return err;
    setenv("ONEAPI_DEVICE_SELECTOR", sel.c_str(), 1);
    std::string list;
    for (size_t i = 0; i < cards.size(); ++i) list += (i ? ", " : "") + std::to_string(cards[i]) + " (" + pcis[i] + ")";
    std::fprintf(stderr, "[ie] --cards: using card %s; ONEAPI_DEVICE_SELECTOR=%s\n", list.c_str(), sel.c_str());
    return {};
}

std::string verify_card_selection(const std::vector<std::string>& pcis) {
    std::vector<std::string> seen;
    try {
        for (const auto& d : sycl::device::get_devices()) {
            if (!d.is_gpu() || d.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
            if (!d.has(sycl::aspect::ext_intel_pci_address)) return "--cards: a visible GPU does not report its PCI address";
            seen.push_back(d.get_info<sycl::ext::intel::info::device::pci_address>());
        }
    } catch (const sycl::exception& e) {
        return std::string("--cards: device enumeration failed: ") + e.what();
    }
    auto want = pcis;
    std::sort(want.begin(), want.end());
    std::sort(seen.begin(), seen.end());
    if (seen == want) return {};
    auto join = [](const std::vector<std::string>& v) {
        std::string s;
        for (const auto& x : v) s += (s.empty() ? "" : ", ") + x;
        return s.empty() ? std::string("none") : s;
    };
    return "--cards: the runtime exposes GPU(s) " + join(seen) + " but the chosen cards are " + join(want) +
           "; refusing to start on the wrong card";
}

}  // namespace ie
