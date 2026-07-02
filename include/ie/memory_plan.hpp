// include/ie/memory_plan.hpp — VRAM-aware placement planner.
//
// At load time the engine no longer relies on hardcoded "this fits / this must
// split" judgments: it senses per-card VRAM, estimates the resident footprint
// (weights + KV reserve + workspace) and auto-picks single-GPU vs multi-GPU.
// An explicit --gpus N always wins; --gpus 0 (the CLI default) means "auto".
#pragma once

#include "ie/model_config.hpp"   // ModelArch

#include <sycl/sycl.hpp>

#include <cstdint>
#include <string>
#include <string_view>

namespace ie {

class GgufReader;

// Default GPU name filter — matches DeviceAllocator::init / DeviceFleet::init.
// Override at runtime with IE_GPU_FILTER (device names drift across drivers).
inline constexpr std::string_view kGpuNameFilter = "B70";

struct PlacementPlan {
    uint32_t    n_gpus         = 1;     // resolved concrete card count (>=1)
    bool        forced         = false; // user pinned --gpus (planner did not decide)
    uint64_t    weights_bytes  = 0;     // estimated resident weight bytes
    uint64_t    kv_bytes       = 0;     // estimated KV reserve at max_ctx
    uint64_t    ws_bytes       = 0;     // workspace/activation margin
    uint64_t    per_card_bytes = 0;     // usable VRAM per card (after safety)
    uint32_t    avail_gpus     = 1;     // matching GPUs the box has
    bool        fits           = true;  // false → estimated to OOM even multi-GPU
    std::string note;                   // one-line human explanation (logged)
};

// Count GPUs whose name contains `name_filter` (mirrors DeviceFleet::init).
uint32_t count_matching_gpus(std::string_view name_filter = kGpuNameFilter);

// Usable VRAM per card = global_mem_size(dev) * safety (headroom for driver/display).
uint64_t usable_card_vram(const sycl::device& dev, double safety = 0.90);

// Sum of all GGUF tensor byte sizes (≈ resident weight bytes with SoA-only).
uint64_t estimate_weight_bytes(const GgufReader& g);

// Conservative KV-cache reserve at max_ctx. Reads GGUF metadata
// (<arch>.block_count / head_count_kv / key_length / full_attention_interval)
// so hybrid arches don't grossly over-count. int8_kv halves the K/V bytes.
uint64_t estimate_kv_bytes(const GgufReader& g, uint32_t max_ctx, bool int8_kv);

// Resolve placement. `requested`: 0 = auto, >0 = forced (clamped to avail).
// `multigpu_ok`: this arch has a working >1-GPU path.
PlacementPlan plan_placement(const GgufReader& g, ModelArch arch,
                             uint32_t requested, uint32_t max_ctx, bool int8_kv,
                             bool multigpu_ok, const sycl::device& dev,
                             std::string_view name_filter = kGpuNameFilter);

}  // namespace ie
