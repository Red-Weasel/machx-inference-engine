// src/loaders/memory_plan.cpp — VRAM-aware placement planner (see header).

#include "ie/memory_plan.hpp"
#include "ie/gguf.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace ie {

namespace {

// Read a u32 GGUF metadata value by exact key; returns `fallback` if absent.
uint32_t kv_u32(const GgufReader& g, const std::string& key, uint32_t fallback) {
    if (const auto* kv = g.find_kv(key)) return static_cast<uint32_t>(kv->as_uint());
    return fallback;
}

std::string human_gb(uint64_t bytes) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f GB", double(bytes) / (1024.0 * 1024.0 * 1024.0));
    return buf;
}

}  // namespace

uint32_t count_matching_gpus(std::string_view name_filter) {
    if (const char* f = std::getenv("IE_GPU_FILTER"); f && *f) name_filter = f;  // see DeviceAllocator::init
    uint32_t n = 0;
    for (const auto& d : sycl::device::get_devices()) {
        if (!d.is_gpu()) continue;
        if (name_filter.empty() ||
            d.get_info<sycl::info::device::name>().find(name_filter) != std::string::npos)
            ++n;
    }
    // Mirror DeviceAllocator's ordinal-0 fallback: if nothing matches the filter,
    // any single GPU is still usable (count it so single-GPU planning works).
    if (n == 0) {
        for (const auto& d : sycl::device::get_devices())
            if (d.is_gpu()) { n = 1; break; }
    }
    return n;
}

uint64_t usable_card_vram(const sycl::device& dev, double safety) {
    uint64_t total = 0;
    try {
        total = dev.get_info<sycl::info::device::global_mem_size>();
    } catch (const sycl::exception&) {
        return 0;  // unknown → caller treats as "can't plan", stays single-GPU
    }
    return uint64_t(double(total) * safety);
}

uint64_t estimate_weight_bytes(const GgufReader& g) {
    uint64_t sum = 0;
    for (const auto& t : g.tensors()) sum += t.nbytes;
    return sum;
}

uint64_t estimate_kv_bytes(const GgufReader& g, uint32_t max_ctx, bool int8_kv) {
    const auto* arch_kv = g.find_kv("general.architecture");
    if (!arch_kv) return 0;
    const std::string a = std::string(arch_kv->as_string());
    const std::string p = a + ".";

    const uint32_t n_layers   = kv_u32(g, p + "block_count", 0);
    const uint32_t n_kv_heads = kv_u32(g, p + "attention.head_count_kv", 0);
    uint32_t head_dim         = kv_u32(g, p + "attention.key_length", 0);
    if (head_dim == 0) {
        const uint32_t hidden  = kv_u32(g, p + "embedding_length", 0);
        const uint32_t n_qhead = kv_u32(g, p + "attention.head_count", 0);
        if (hidden && n_qhead) head_dim = hidden / n_qhead;
    }
    if (!n_layers || !n_kv_heads || !head_dim) return 0;

    // Hybrid arches (crown qwen35moe, qwen35 27B, qwen3next) keep KV only on the
    // full-attention layers (every `full_attention_interval`-th). When the key is
    // absent (dense / gemma4) every layer keeps KV.
    const uint32_t interval = kv_u32(g, p + "full_attention_interval", 1);
    const uint32_t n_full   = interval > 1 ? (n_layers + interval - 1) / interval : n_layers;

    // K and V, fp16 (or int8 + a small fp16 scale shadow ≈ +1/32 per element).
    const uint64_t per_layer = uint64_t(n_kv_heads) * max_ctx * head_dim;
    const uint64_t elt = int8_kv ? 1 : 2;
    uint64_t bytes = per_layer * n_full * 2 /*K+V*/ * elt;
    if (int8_kv) bytes += per_layer * n_full * 2 * sizeof(sycl::half) / 32;  // block scales
    return bytes;
}

PlacementPlan plan_placement(const GgufReader& g, ModelArch /*arch*/,
                             uint32_t requested, uint32_t max_ctx, bool int8_kv,
                             bool multigpu_ok, const sycl::device& dev,
                             std::string_view name_filter) {
    PlacementPlan plan;
    plan.weights_bytes  = estimate_weight_bytes(g);
    plan.kv_bytes       = estimate_kv_bytes(g, max_ctx, int8_kv);
    // Workspace/activation margin: a flat 1.5 GB covers dequant scratch, prefill
    // activations and sampler buffers for the chunk sizes we use.
    plan.ws_bytes       = uint64_t(1536) * 1024 * 1024;
    plan.per_card_bytes = usable_card_vram(dev);
    plan.avail_gpus     = count_matching_gpus(name_filter);

    // Forced: the user pinned --gpus N — respect it (clamped to available).
    if (requested > 0) {
        plan.forced = true;
        plan.n_gpus = std::min(requested, std::max<uint32_t>(plan.avail_gpus, 1));
        plan.note   = "forced --gpus " + std::to_string(requested) +
                      " (using " + std::to_string(plan.n_gpus) + ")";
        return plan;
    }

    const uint64_t need = plan.weights_bytes + plan.kv_bytes + plan.ws_bytes;
    const uint64_t card = plan.per_card_bytes;

    // No reliable VRAM reading → conservative single-GPU (preserves old behavior).
    if (card == 0) {
        plan.n_gpus = 1;
        plan.note   = "VRAM size unavailable → single-GPU (legacy behavior)";
        return plan;
    }

    if (need <= card) {
        plan.n_gpus = 1;
        plan.note = "fits 1 card (" + human_gb(need) + " of " + human_gb(card) + ")";
        return plan;
    }

    // Over one card. Split if the arch supports it and enough cards exist.
    if (multigpu_ok && plan.avail_gpus >= 2) {
        // Smallest card count that holds the footprint (cap at available).
        uint32_t want = uint32_t((need + card - 1) / card);
        plan.n_gpus = std::min(std::max<uint32_t>(want, 2), plan.avail_gpus);
        plan.fits   = need <= uint64_t(plan.n_gpus) * card;
        plan.note   = "needs " + human_gb(need) + " > " + human_gb(card) +
                      "/card → split across " + std::to_string(plan.n_gpus) + " GPUs" +
                      (plan.fits ? "" : " (STILL TIGHT — may OOM)");
        return plan;
    }

    // Can't split (single-GPU-only arch, or <2 cards). Run on one card and warn.
    plan.n_gpus = 1;
    plan.fits   = false;
    plan.note   = "needs " + human_gb(need) + " > " + human_gb(card) + "/card but " +
                  (multigpu_ok ? "<2 GPUs available" : "arch is single-GPU only") +
                  " → single-GPU (MAY OOM)";
    return plan;
}

}  // namespace ie
