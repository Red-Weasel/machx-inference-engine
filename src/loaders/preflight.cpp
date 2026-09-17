// src/loaders/preflight.cpp — load-time capability preflight (see header).
//
// Metadata-only: reads GGUF tensor infos + `general.architecture`, reuses the
// memory_plan estimators and the dtype table, queries device VRAM size, and
// reports. No tensor upload, no kernel launch, no device allocation.

#include "ie/preflight.hpp"
#include "ie/server_capabilities.hpp"

#include "ie/gguf.hpp"
#include "ie/memory_plan.hpp"    // estimate_*, usable_card_vram, count_matching_gpus, plan_placement
#include "ie/model_config.hpp"   // detect_arch, ModelArch, is_dense_arch
#include "ie/dequant_ref.hpp"    // ie::ref host dequant — capability probe below

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace ie {

// Capability probe for the kIQ3_XXS whitelist entry below. The whitelist is a
// hand-written list, so no entry is self-checking: a dtype can be added before
// its decode path exists, and preflight then reports "loadable" for a model
// that cannot load. IQ3_XXS is the newest and least-settled entry (it briefly
// sat on the whitelist with no dequantiser at all), so bind the claim to the
// thing it depends on — naming the host dequantiser here makes its removal a
// BUILD error in this file rather than a false "WILL LOAD" at runtime.
// Half-guarantee, stated plainly: this pins the ie::ref decode entry point and
// its signature, NOT dense_dispatch.hpp's is_host_fallback_dtype table (an
// internal header this TU must not include), which is the other half of the
// load path.
static_assert(std::is_same_v<decltype(&ref::dequant_iq3_xxs_buffer),
                             void (*)(const void*, std::size_t, float*)>,
              "IQ3_XXS is on the loadable whitelist but its host dequantiser "
              "(ie::ref::dequant_iq3_xxs_buffer) is missing or changed shape — "
              "drop kIQ3_XXS from is_loadable_weight_dtype in the same edit");

// The engine's v1 weight-dtype consumption set (Global Constraint #4). A dtype
// outside this whitelist (every other K-quant, all IQ*/TQ* trellis quants, the
// legacy Q4_1/Q5_0/Q5_1/Q8_1, Q2_K/Q3_K/Q8_K, and non-weight int/f64 ids) has
// no kernel today → the model will hard-fail deep in load. The one exception is
// integer INDEX tensors, which are not weights at all and are classified by
// is_index_dtype below rather than here. Pure + NOT arch-role-aware (the report
// caveat covers per-arch role acceptance).
bool is_loadable_weight_dtype(DType d) noexcept {
    switch (d) {
        case DType::kF32:
        case DType::kF16:
        case DType::kBF16:
        case DType::kQ4_0:
        case DType::kQ8_0:
        case DType::kQ4_K:
        case DType::kQ5_K:
        case DType::kQ6_K:
        case DType::kMXFP4:
        // Prism ternary 2-bit (Q2_0) — native SoA W2A8 int-dot decode
        // (gemv_q2_0_soa_q8); qwen35 dense path (Ternary-Bonsai-27B).
        case DType::kQ2_0:
        // Prism 1-bit (Q1_0) — loaded via exact load-time expansion to the
        // Q2_0 SoA path (sign codes {0,2}, same d); qwen35 dense (Bonsai-27B).
        case DType::kQ1_0:
        // EXL3 (QTIP trellis) — native on-GPU decode (gemv_exl3); dense + qwen3next.
        case DType::kEXL3:
        // Host universal dequant→fp16 fallback (dense path; see
        // dense_dispatch.hpp upload_host_dequant_to_fp16). Loads via the slow
        // host dequant, not a fast device kernel yet.
        case DType::kQ4_1:
        case DType::kQ5_0:
        case DType::kQ5_1:
        case DType::kQ2_K:
        case DType::kQ3_K:
        case DType::kIQ4_NL:
        case DType::kIQ4_XS:
        // IQ3_XXS — the dominant expert dtype of the DeepSeek-V4-Flash
        // UD-Q3_K_XL GGUF (ffn_gate_exps / ffn_up_exps). Loads via the same host
        // dequant fallback: ie::ref::dequant_iq3_xxs_buffer (pinned by the
        // static_assert above) reached through dense_dispatch's host-fallback
        // upload. Like every other entry here this asserts a path EXISTS, not
        // that its numerics are validated.
        case DType::kIQ3_XXS:
            return true;
        // NOTE (P1 gate 2026-09-01): kIQ2_XXS / kSTQ1_0 are deliberately NOT
        // whitelisted yet. The hyv4 port streams them as raw host expert banks
        // (never through this loader); whitelisting them here without a
        // dense_dispatch host-dequant case would re-create the IQ3_XXS
        // false-loadable regression documented in preflight_test.cpp.
        default:
            return false;
    }
}

namespace {

// human_gb, kWorkspaceMarginBytes, and arch_can_multigpu all live in
// memory_plan.hpp — reused here so the footprint math and the splittable-arch
// set have a single home shared with Engine::load.

// First GPU matching the name filter (IE_GPU_FILTER override, else "B70"),
// falling back to any GPU (mirrors count_matching_gpus). Returns false if the
// box has no GPU at all.
bool first_matching_gpu(sycl::device& out) {
    std::string_view filter = kGpuNameFilter;
    if (const char* f = std::getenv("IE_GPU_FILTER"); f && *f) filter = f;
    for (const auto& d : sycl::device::get_devices()) {
        if (!d.is_gpu()) continue;
        if (filter.empty() ||
            d.get_info<sycl::info::device::name>().find(filter) != std::string::npos) {
            out = d;
            return true;
        }
    }
    for (const auto& d : sycl::device::get_devices())
        if (d.is_gpu()) { out = d; return true; }
    return false;
}

// Integer index tensors — a class apart from weights, not a widening of the
// weight whitelist. GGML's I8/I16/I32/I64 ids never carry quantised weights;
// they are lookup tables consumed as integers and never dequantised. The case
// in hand is deepseek4's hash router: blk.0-2 `ffn_gate_tid2eid`, a token-id →
// expert-id table (3 tensors, 0.01 GB). Putting I32 in is_loadable_weight_dtype
// would instead assert that an I32 *weight* is dequantisable — false, and it
// would hide a genuinely malformed file. Whether the arch actually consumes a
// given index table stays covered by the report's role caveat.
bool is_index_dtype(DType d) noexcept {
    switch (d) {
        case DType::kI8:
        case DType::kI16:
        case DType::kI32:
        case DType::kI64:
            return true;
        default:
            return false;
    }
}

struct DtypeStat {
    DType                    dtype;
    uint64_t                 bytes = 0;
    uint32_t                 count = 0;
    std::vector<std::string> examples;  // up to 5 tensor names
};

}  // namespace

PreflightReport preflight_model(const GgufReader& g,
                                uint32_t max_ctx, bool int8_kv,
                                uint32_t requested_gpus) {
    PreflightReport r;
    std::ostringstream os;

    // ---- 1) Arch line ----------------------------------------------------
    const ModelArch arch = detect_arch(g);
    const auto* arch_kv  = g.find_kv("general.architecture");
    const std::string arch_str = arch_kv ? std::string(arch_kv->as_string()) : "<missing>";
    const bool recognized = (arch != ModelArch::kUnknown);
    os << "arch: " << arch_str << (recognized ? " (recognized)\n" : " (UNRECOGNIZED)\n");

    std::string block_reason;  // first blocking issue → verdict reason
    if (!server_supports_arch(arch)) {
        r.will_load = false;
        block_reason = "architecture '" + arch_str + "' has no server backend";
    }

    // ---- 2) Quant / dtype section ---------------------------------------
    std::vector<DtypeStat> stats;  // first-seen order
    for (const auto& t : g.tensors()) {
        auto it = std::find_if(stats.begin(), stats.end(),
                               [&](const DtypeStat& s) { return s.dtype == t.dtype; });
        if (it == stats.end()) {
            stats.push_back(DtypeStat{t.dtype, 0, 0, {}});
            it = stats.end() - 1;
        }
        it->bytes += t.nbytes;
        it->count += 1;
        if (it->examples.size() < 5) it->examples.emplace_back(t.name);
    }

    // Dominant weight dtype = the one holding the most bytes (the quant at a glance).
    if (!stats.empty()) {
        const auto* dom = &stats.front();
        for (const auto& s : stats) if (s.bytes > dom->bytes) dom = &s;
        os << "quant: dominant weight dtype " << type_name(dom->dtype)
           << " (" << human_gb(dom->bytes) << " across " << dom->count << " tensors)\n";
    } else {
        os << "quant: no tensors found in file\n";
    }

    os << "dtypes present:";
    for (const auto& s : stats)
        os << ' ' << type_name(s.dtype) << "[" << s.count << "]";
    os << '\n';

    std::vector<std::string> unsupported_names;  // for the verdict reason
    for (const auto& s : stats) {
        if (is_loadable_weight_dtype(s.dtype)) continue;
        if (is_index_dtype(s.dtype)) {
            os << "  index " << type_name(s.dtype) << " (" << s.count
               << " tensors, " << human_gb(s.bytes) << ") — integer lookup table, "
                  "never dequantised → not a load blocker. examples:\n";
            for (const auto& n : s.examples) os << "    " << n << '\n';
            continue;
        }
        os << "  UNSUPPORTED " << type_name(s.dtype) << " (" << s.count
           << " tensors, " << human_gb(s.bytes) << ") → will not load. examples:\n";
        for (const auto& n : s.examples) os << "    " << n << '\n';
        unsupported_names.emplace_back(std::string(type_name(s.dtype)));
        r.will_load = false;
    }
    if (!unsupported_names.empty() && block_reason.empty()) {
        std::string joined;
        for (size_t i = 0; i < unsupported_names.size(); ++i)
            joined += (i ? ", " : "") + unsupported_names[i];
        block_reason = "unsupported weight dtype(s): " + joined + " (no kernel)";
    }

    // ---- 3) VRAM section -------------------------------------------------
    const bool streaming = server_streams_experts(arch);
    if (streaming) {
        os << "placement: streaming experts; total GGUF weight bytes are not the VRAM footprint.\n"
              "Resident weights, host experts, context and workspace must pass the runtime's memory plan.\n";
    } else {
        const uint64_t weights = estimate_weight_bytes(g);
        const uint64_t kv      = estimate_kv_bytes(g, max_ctx, int8_kv);
        const uint64_t ws      = kWorkspaceMarginBytes;  // shared with plan_placement
        const uint64_t need    = weights + kv + ws;

        os << "VRAM need: weights " << human_gb(weights) << " + KV@" << max_ctx
           << (int8_kv ? " (int8)" : "") << " " << human_gb(kv)
           << " + workspace " << human_gb(ws) << " = " << human_gb(need) << '\n';

        sycl::device dev;
        if (!first_matching_gpu(dev)) {
            os << "VRAM: no GPU detected (skipped)\n";
            // Per spec: absence of a GPU does NOT flip will_load.
        } else {
            const PlacementPlan plan = plan_placement(g, arch, requested_gpus,
                                                      max_ctx, int8_kv,
                                                      arch_can_multigpu(arch), dev);
            if (plan.per_card_bytes == 0) {
                os << "VRAM: per-card size unavailable — fit check skipped\n";
            } else {
                const uint64_t total = uint64_t(plan.n_gpus) * plan.per_card_bytes;
                const bool fits = need <= total;
                os << "VRAM: per-card usable " << human_gb(plan.per_card_bytes)
                   << " x " << plan.n_gpus << " card(s) (box has " << plan.avail_gpus
                   << " matching) = " << human_gb(total) << " usable\n";
                os << "placement: " << plan.note << '\n';
                if (!fits) {
                    r.will_load = false;
                    if (block_reason.empty())
                        block_reason = "does not fit in VRAM (needs " + human_gb(need) +
                                       ", have " + human_gb(total) + " across " +
                                       std::to_string(plan.n_gpus) + " card(s))";
                }
            }
        }

    }
    // ---- 4) Verdict line -------------------------------------------------
    if (r.will_load) {
        os << (streaming ? "VERDICT: metadata checks passed; runtime memory plan required\n"
                         : "VERDICT: WILL LOAD (fast path)\n");
    } else {
        os << "VERDICT: WILL NOT LOAD — "
           << (block_reason.empty() ? std::string("blocking issue found") : block_reason)
           << '\n';
        // Options hint tailored to the blocker (always includes the general levers).
        os << "options:";
        if (!unsupported_names.empty())
            os << " use a different/smaller quant (e.g. Q4_K / Q6_K / Q8_0);";
        os << " --gpus N if the arch splits; lower --ctx to shrink the KV reserve.\n";
    }

    // ---- 5) Caveat line --------------------------------------------------
    os << "note: conservative check — per-arch dtype-role acceptance is not fully "
          "modeled, so \"WILL LOAD\" is a strong indicator, not a guarantee.\n";

    r.text = os.str();
    return r;
}

}  // namespace ie
