// include/ie/preflight.hpp — load-time capability preflight (metadata-only).
//
// Before any tensor upload, answer two questions from GGUF metadata + a cheap
// device-info VRAM query: (1) does the engine have a kernel for every weight
// dtype in this file, and (2) does the estimated footprint fit in VRAM. The
// point is an instant verdict — "IQ4_XS experts → no kernel → won't load" in
// ~1s — instead of a multi-minute disk read that hard-fails deep in load.
//
// This unit touches NO numerics: it reads metadata, reuses the memory_plan
// estimators + the dtype table, and reports. It never uploads a tensor, never
// launches a kernel, and allocates nothing on the device beyond the VRAM-size
// query. It must run even on an UNLOADABLE model — that is the whole point.
#pragma once

#include "ie/model_config.hpp"   // ModelArch, detect_arch
#include "ie/dtype.hpp"          // DType

#include <string>
#include <cstdint>

namespace ie {

class GgufReader;

// True iff the engine has SOME consumption path for this weight dtype today
// (Global Constraint #4 set). Pure; unit-tested. NOT arch-role-aware.
bool is_loadable_weight_dtype(DType d) noexcept;

struct PreflightReport {
    bool        will_load = true;   // no blocking dtype/arch/VRAM issue found
    std::string text;               // full multi-line human report
};

// Metadata-only. The only GPU interaction permitted is a device-info VRAM
// query (usable_card_vram / count_matching_gpus). Never uploads or runs kernels.
PreflightReport preflight_model(const GgufReader& g,
                                uint32_t max_ctx, bool int8_kv,
                                uint32_t requested_gpus);

}  // namespace ie
