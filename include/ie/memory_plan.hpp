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

// Flat workspace/activation margin folded into the resident footprint (dequant
// scratch, prefill activations, sampler buffers). Shared by the planner and the
// preflight report so the footprint math has exactly one source.
inline constexpr uint64_t kWorkspaceMarginBytes = uint64_t(1536) * 1024 * 1024;

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

// Format a raw byte count as "%.1f GB" — the shared footprint formatter used by
// both the planner note and the load-time OOM gate (one place for the math).
std::string human_gb(uint64_t bytes);

// True → refuse to load: the planner is confident the footprint won't fit
// (fits==false) and the OOM override (IE_ALLOW_OOM) is off. NOTE: `plan.forced`
// (user pinned --gpus) does NOT bypass this — pinning the card count is a
// separate intent from consenting to OOM, so a forced-but-over-budget load
// (e.g. a 222k KV that can't fit on the pinned cards) still gets refused.
bool should_block_oom(const PlacementPlan& plan, bool allow_oom_env) noexcept;

// True → this arch has a working >1-GPU (layer-split / tensor-parallel) path
// today; gemma4 and unknown arches are single-GPU. Feeds plan_placement's
// `multigpu_ok` and the preflight fit check — one definition so the two loaders
// can't drift apart.
bool arch_can_multigpu(ModelArch arch) noexcept;

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
