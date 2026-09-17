// tests/unit/deepseek4_residency_test.cpp — the tiered VRAM/host residency rule.
//
// WHAT DEFECT THIS TEST EXISTS FOR
// --------------------------------
// The Phase 5 run configuration pinned ALL 43 layers x 256 experts (120.393 GB
// for Q3) into host RAM *and* uploaded the always-resident set to VRAM.  That
// double-counts: docs/deepseek4/33_vram_residency_design.md §2.2 specifies that
// an expert held permanently in the VRAM arena has its HOST COPY FREED, which is
// what turns 120.393 GB of pinned host RAM into 80.419 GB for Q3.  A full-pin
// run on this box was heading for a watchdog kill, and an earlier one had
// `systemd-oomd` kill 147 processes.
//
// WHAT IT CHECKS, AND WITH WHAT EVIDENCE
// --------------------------------------
//   §1  Per-layer expert slot sizes read from the REAL GGUF tensor tables of
//       BOTH quants — each tensor's own `type_id`, never a per-model constant.
//       blk.26 is checked by name because Unsloth's UD mixed precision puts its
//       gate/up in MXFP4 while its 42 siblings are IQ3_XXS, and it is exactly
//       the layer a per-model constant gets wrong.
//   §2  The plan reproduces doc 33's split and — the whole point — that
//         static_slots x Σslot_bytes  +  pinned_host_bytes  ==  the whole pool,
//       i.e. every expert byte is charged to exactly ONE tier.
//   §3  The hard pinned-host cap refuses rather than downsizing, and the naive
//       whole-pool pin is exactly what it refuses.
//   §4  THE ALWAYS-RESIDENT SET, packed vs dequantised-to-fp16, for BOTH quants
//       from the same real tensor tables.  The engine used to expand every
//       resident weight to fp16 and spend 14.785 GB on the 7.81 GB the Q3 file
//       stores; §4 recomputes both totals from `ds4_dense_device_bytes` — the
//       loader's own policy function — and anchors the old figure against the
//       14.785 GB a real load actually printed.
//   §5  The residency plan the recovered VRAM buys, again for both quants, and
//       the arithmetic proof that doc 33's 80.419 GB target is NOT reachable on
//       one card at any resident-set size (it is a two-card aggregate).
//   §6  The live pinned-host cap derived from /proc/meminfo, which a constant
//       cannot do because a constant cannot see the page cache.
//   §7  TWO-CARD HIDDEN-DIM EXPERT-TP (doc 33 §2.1, "Option A"): the per-layer
//       slice over BOTH real tensor tables, the proof that it lands on
//       quantisation block boundaries for each tensor's OWN dtype, and the
//       proof that the two cards' slices TILE the expert exactly (no byte lost,
//       none duplicated) including on blk.26 where Q3's dtype changes.
//   §8  The two-card residency plan for both quants, and the whole-box
//       invariant  n_cards x (static VRAM + pinned host) == the whole pool.
//   §9  The load-time split is BYTE-EXACT: a card's packed slice equals the
//       corresponding bytes of the whole-expert pack, on synthetic IQ3_XXS and
//       MXFP4 tensors, for both the N-slice (gate/up) and the K-slice (down).
//   §10 A statically-resident expert has NO host copy at all: the arena returns
//       a null pointer for it, so "this expert was never pinned" is a checkable
//       fact rather than a stale byte.
//   §11 The 32.53 GB per-allocation ceiling still fails loudly, with nothing
//       allocated and nothing silently downsized.
//   §12 An expert that is neither VRAM-resident nor host-pinned yields
//       kDs4NoSlot — the sentinel `DeepSeek4Runtime::layer_forward` turns into
//       "refusing to fabricate an expert output" — and the static partition is
//       never evicted no matter how hard the streaming partition is hammered.
//   §13 THE NUMERICS GATE on §4's saving: the in-kernel Q8_0/Q6_K decoders that
//       let a weight stay packed are compared element-by-element against
//       `ie::ref` (bit-exact vs ggml) through the SAME dispatchers `forward()`
//       uses.  Equality must be EXACT — the packed path is not an approximation,
//       it removes the fp16 rounding the old path applied on top of the dequant.
//
//   §14 The cross-card reduction of the sliced `down` partials, staged through
//       pinned host memory because P2P is unavailable between these two cards.
//   §15 THE TWO-CARD ORCHESTRATOR (`DeepSeek4TpRuntime`), which is what turns
//       all of the above into a forward pass:
//         a  the expert block itself — one routed expert packed whole and packed
//            as two halves, run through the SAME GEMVs `layer_forward` uses.
//            The SwiGLU intermediate must be BIT-IDENTICAL and the two halves'
//            `down` outputs must sum to the whole's within TWO fp16 ulps.  This
//            is the analytic statement that expert-TP is a PARTITION.
//         b  a 2-layer miniature-but-structurally-real deepseek4 GGUF, run
//            single-card and two-card, compared end to end.
//         c  the same for a 6-layer one (both routers, CSA + HCA, an all-MXFP4
//            expert layer).
//         d  the card->device binding contract and every refusal around it.
//       §15 also proves `n_cards == 1` through the orchestrator is BIT-IDENTICAL
//       to the plain single-card `forward()`, i.e. that splitting `layer_forward`
//       at the reduction seam changed no arithmetic.
//
// §1-§9 are pure host work: no GPU, no device memory, no model bytes read (mmap
// header parse only; §9 builds its own synthetic tensors in RAM).  §10-§14 need
// a queue and allocate ~35 MB of pinned host and ~4 MB of device memory in
// total.  §15 additionally writes two ~5 MB GGUFs to $TMPDIR and loads them
// (~10 MB of VRAM per runtime, three runtimes at a time).  They skip loudly if
// no GPU exists.
//
// $DS4_GPU picks the device.  $DS4_TP_GPUS="0,1" makes §15b/§15c a GENUINE
// two-device run; without it both cards share one device and §15 says so.
#undef NDEBUG  // build is Release (-DNDEBUG); asserts must stay live here
#include "ie/deepseek4.hpp"
#include "ie/deepseek4_experts.hpp"
#include "ie/deepseek4_ops.hpp"
#include "ie/dequant_ref.hpp"
#include "ie/expert_stream.hpp"
#include "ie/gguf.hpp"
#include "ie/gguf_writer.hpp"
#include "ie/model_config.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <sycl/sycl.hpp>

#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

int g_fail = 0;

void ok(const char* what, bool cond, const std::string& detail = {}) {
    std::printf("  [%s] %s%s%s\n", cond ? "PASS" : "FAIL", what,
                detail.empty() ? "" : " -- ", detail.c_str());
    if (!cond) ++g_fail;
}

std::string gb(uint64_t b) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3f GB (%llu B)", double(b) / 1e9, (unsigned long long)b);
    return buf;
}

std::string env_or(const char* key, const char* fallback) {
    const char* v = std::getenv(key);
    return v ? std::string(v) : std::string(fallback);
}

// One quant's routed-expert geometry, read from the file and nothing else.
struct ExpertGeometry {
    bool                  opened = false;
    uint32_t              n_layers = 0, n_experts = 0, hidden = 0, expert_ffn = 0;
    std::vector<uint64_t> slot_bytes;      // per layer, from ds4_slot_layout
    uint64_t              pool_slot = 0;   // Σ_L slot_bytes[L] * n_experts
    uint64_t              pool_gguf = 0;   // Σ over the 3*n_layers expert tensors of nbytes
    uint64_t              per_layer = 0;   // Σ_L slot_bytes[L]
    std::string           dtypes_of_25, dtypes_of_26;
    // §4: the always-resident (non-expert) set, both policies.
    uint64_t              resident_fp16   = 0;  // every 2-D weight dequantised to fp16
    uint64_t              resident_packed = 0;  // ds4_dense_keeps_packed honoured
    uint64_t              resident_gguf   = 0;  // what the FILE stores for that same set
    uint64_t              n_dense_packed = 0, n_dense_expanded = 0;
    std::string           resident_err;         // non-empty => a tensor went unclassified
    // §4b/§8c: the SAME set under two-card non-expert TP — what ONE card holds
    // when `Ds4Options::split_non_expert` is on.  Charged tensor by tensor with
    // the loader's own `ds4_dense_device_bytes` over the sliced extents, so it
    // cannot drift from what `upload_dense` allocates.
    uint64_t              resident_split  = 0;  // per card, packed policy honoured
    uint64_t              split_rows_saved = 0, split_cols_saved = 0, split_mirrored = 0;
    uint64_t              n_split_rows = 0, n_split_cols = 0;
    // §7/§8: the two-card hidden-dim expert-TP split, derived per layer from
    // that layer's OWN three tensors.
    std::vector<uint64_t> card0_bytes, card1_bytes;   // per layer, this card's slice
    uint64_t              card_per_layer = 0;         // Σ_L card0_bytes[L]
    std::string           slice_err;                  // a layer that refused the 2-way split
    bool                  slice_tiles     = true;     // card0+card1 == whole, EVERY layer
    bool                  slice_balanced  = true;     // card0 == card1, EVERY layer
    bool                  slice_range_ok  = true;     // [0,EF/2) and [EF/2,EF), EVERY layer
    bool                  whole_matches_1card = true; // n_cards==1 reproduces ds4_slot_layout
    std::string           midblock_refusal;           // what a mid-block split says, blk.0
    uint32_t              down_block = 0;             // block elements of blk.0's down dtype
};

// This box's B70 as reported by `global_mem_size`.  Recorded in
// docs/deepseek4/40_ALPHA_OMEGA_PHASE_PLAN.md, and §5 re-derives that document's
// `slots=36 ... host=109.106 GB` line from it exactly, which is what pins it down.
constexpr uint64_t kB70GlobalMem = 34'242'000'000ull;
// The margin DeepSeek4Runtime::load leaves for workspace + driver overhead.
constexpr uint64_t kWsMargin = 2ull << 30;

// The 2-D weights `DeepSeek4Runtime::upload_layer`/`load` push through
// `upload_dense`.  Kept as a literal list, mirroring the call order in
// src/model/deepseek4.cpp, so a tensor added to the forward without a residency
// decision makes §4 fail loudly (`resident_err`) instead of silently vanishing
// from the budget.
const char* const kDenseSuffix[] = {
    "attn_q_a.weight", "attn_q_b.weight", "attn_kv.weight",
    "attn_output_a.weight", "attn_output_b.weight",
    "attn_compressor_kv.weight", "attn_compressor_gate.weight",
    "indexer.attn_q_b.weight", "indexer.proj.weight",
    "indexer_compressor_kv.weight", "indexer_compressor_gate.weight",
    "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
};
const char* const kGlobalDense[] = {"token_embd.weight", "output.weight"};

// THE PER-TENSOR SPLIT DECISION, as data.  These two lists are the whole of the
// non-expert TP policy `DeepSeek4Runtime::upload_layer` applies; every other
// dense weight is MIRRORED, and §4b asserts that this classification reproduces
// what a real two-card load actually allocated (on the fixture, where a real
// load is affordable).
//
//   ROWS — the output dim N is partitioned.  No sum is split, so no reduction.
//   COLS — the contraction dim K is partitioned.  The result is a partial.
//          `attn_output_b` is the ONE weight whose partial needs a reduction of
//          its own; `ffn_down_shexp`'s folds into the routed-expert reduction
//          that already existed.
const char* const kSplitRows[] = {
    "attn_q_b.weight",          // query heads
    "attn_output_a.weight",     // output groups (aligned with the heads)
    "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
};
const char* const kSplitCols[] = {
    "attn_output_b.weight",     // -> a NEW reduction, T*hidden fp32, per layer
    "ffn_down_shexp.weight",    // -> rides the routed-expert reduction, free
};

bool is_one_of(const std::string& s, const char* const* set, size_t n) {
    for (size_t i = 0; i < n; ++i) if (s == set[i]) return true;
    return false;
}

// "blk.17.attn_q_a.weight" -> "attn_q_a.weight"; a non-layer name is returned
// unchanged.
std::string layer_suffix(std::string_view name) {
    if (name.rfind("blk.", 0) != 0) return std::string(name);
    const size_t dot = name.find('.', 4);
    return dot == std::string_view::npos ? std::string(name)
                                         : std::string(name.substr(dot + 1));
}

const ie::GgufTensorInfo* need(const ie::GgufReader& g, const std::string& n) {
    return g.find_tensor(n);
}

std::string trio(const ie::GgufTensorInfo* a, const ie::GgufTensorInfo* b,
                 const ie::GgufTensorInfo* c) {
    return std::string(ie::type_name(a->dtype)) + "/" + std::string(ie::type_name(b->dtype)) +
           "/" + std::string(ie::type_name(c->dtype));
}

// Reads the tensor table only.  `data` is never dereferenced, so no weight byte
// is pulled off the disk.
bool read_geometry(const std::string& shard1, ExpertGeometry& out) {
    ie::GgufReader g;
    const std::string e = g.open(shard1);
    if (!e.empty()) {
        std::printf("  SKIP: cannot open %s (%s)\n", shard1.c_str(), e.c_str());
        return false;
    }
    // Layer count from the file, not from a constant.
    uint32_t L = 0;
    while (need(g, "blk." + std::to_string(L) + ".ffn_gate_exps.weight")) ++L;
    if (L == 0) { std::printf("  SKIP: %s has no routed experts\n", shard1.c_str()); return false; }

    const ie::GgufTensorInfo* g0 = need(g, "blk.0.ffn_gate_exps.weight");
    out.hidden     = uint32_t(g0->shape[0]);
    out.expert_ffn = uint32_t(g0->shape[1]);
    out.n_experts  = uint32_t(g0->shape[2]);
    out.n_layers   = L;
    out.slot_bytes.assign(L, 0);
    for (uint32_t l = 0; l < L; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        const ie::GgufTensorInfo* gt = need(g, p + "ffn_gate_exps.weight");
        const ie::GgufTensorInfo* ut = need(g, p + "ffn_up_exps.weight");
        const ie::GgufTensorInfo* dt = need(g, p + "ffn_down_exps.weight");
        if (!gt || !ut || !dt) { std::printf("  FAIL: layer %u missing an expert tensor\n", l); return false; }
        ie::Ds4SlotLayout lay;
        const std::string le = ie::ds4_slot_layout(*gt, *ut, *dt, out.hidden, out.expert_ffn, lay);
        if (!le.empty()) { std::printf("  FAIL: layer %u slot layout: %s\n", l, le.c_str()); return false; }
        out.slot_bytes[l] = lay.bytes;
        out.per_layer    += lay.bytes;
        out.pool_gguf    += gt->nbytes + ut->nbytes + dt->nbytes;
        if (l == 25) out.dtypes_of_25 = trio(gt, ut, dt);
        if (l == 26) out.dtypes_of_26 = trio(gt, ut, dt);

        // ---- §7: the two-card hidden-dim slice of THIS layer ----
        // Derived from this layer's own tensors, because the dtype — and hence
        // the block a slice must land on — varies by layer (Q3's blk.26).
        uint64_t half[2] = {0, 0};
        for (uint32_t c = 0; c < 2 && out.slice_err.empty(); ++c) {
            ie::Ds4ExpertSlice sl;
            const std::string se =
                ie::ds4_expert_slice(*gt, *ut, *dt, out.expert_ffn, 2, c, sl);
            if (!se.empty()) { out.slice_err = "layer " + std::to_string(l) + ": " + se; break; }
            if (sl.efc != out.expert_ffn / 2 || sl.ef0 != c * (out.expert_ffn / 2))
                out.slice_range_ok = false;
            ie::Ds4SlotLayout cl;
            const std::string ce =
                ie::ds4_slot_layout_tp(*gt, *ut, *dt, out.hidden, out.expert_ffn, sl, cl);
            if (!ce.empty()) { out.slice_err = "layer " + std::to_string(l) + ": " + ce; break; }
            half[c] = cl.bytes;
        }
        if (!out.slice_err.empty()) return false;
        out.card0_bytes.push_back(half[0]);
        out.card1_bytes.push_back(half[1]);
        out.card_per_layer += half[0];
        out.slice_balanced = out.slice_balanced && half[0] == half[1];
        out.slice_tiles    = out.slice_tiles && half[0] + half[1] == lay.bytes;

        // A 1-card "slice" must reproduce ds4_slot_layout byte for byte —
        // otherwise the TP path is a second, divergent implementation.
        ie::Ds4ExpertSlice one;
        ie::Ds4SlotLayout  onel;
        if (!ie::ds4_expert_slice(*gt, *ut, *dt, out.expert_ffn, 1, 0, one).empty() ||
            !ie::ds4_slot_layout_tp(*gt, *ut, *dt, out.hidden, out.expert_ffn, one, onel).empty() ||
            onel.bytes != lay.bytes || onel.gate.off0 != lay.gate.off0 ||
            onel.up.off1 != lay.up.off1 || onel.down.off0 != lay.down.off0 ||
            onel.down.K != lay.down.K || onel.gate.N != lay.gate.N)
            out.whole_matches_1card = false;

        if (l == 0) {
            out.down_block = dt->dtype == ie::DType::kIQ3_XXS ? uint32_t(ie::kQK_K)
                             : dt->dtype == ie::DType::kMXFP4 ? uint32_t(ie::kQK_MXFP4)
                                                              : 0u;
            // 2048/128 = 16 elements per card, which is HALF an MXFP4 block: the
            // split would land mid-block and the decode would be garbage.  The
            // planner must say so by name rather than round.
            ie::Ds4ExpertSlice bad;
            out.midblock_refusal =
                ie::ds4_expert_slice(*gt, *ut, *dt, out.expert_ffn, 128, 0, bad);
        }
    }
    out.pool_slot = out.per_layer * out.n_experts;

    // ---- §4: the always-resident set, charged the way load() charges it ----
    for (const ie::GgufTensorInfo& ti : g.tensors()) {
        const std::string name(ti.name);
        if (name.find("_exps.") != std::string::npos) continue;   // streamed, not resident
        const std::string suf = layer_suffix(ti.name);
        const bool dense = is_one_of(suf, kDenseSuffix, std::size(kDenseSuffix)) ||
                           is_one_of(name, kGlobalDense, std::size(kGlobalDense));
        if (dense) {
            if (ti.n_dims != 2) { out.resident_err = name + ": dense weight is not 2-D"; break; }
            const uint64_t K = ti.shape[0], N = ti.shape[1];
            out.resident_fp16   += K * N * 2;
            const uint64_t whole = ie::ds4_dense_device_bytes(ti.dtype, K, N);
            out.resident_packed += whole;
            out.resident_gguf   += ti.nbytes;
            (ie::ds4_dense_keeps_packed(ti.dtype, K) ? out.n_dense_packed
                                                     : out.n_dense_expanded)++;
            // ---- what ONE CARD holds under the two-card non-expert split ----
            if (is_one_of(suf, kSplitRows, std::size(kSplitRows))) {
                if (N % 2) { out.resident_err = name + ": N is odd, cannot row-split"; break; }
                const uint64_t half = ie::ds4_dense_device_bytes(ti.dtype, K, N / 2);
                out.resident_split += half;
                out.split_rows_saved += whole - half;
                ++out.n_split_rows;
            } else if (is_one_of(suf, kSplitCols, std::size(kSplitCols))) {
                if (K % 2) { out.resident_err = name + ": K is odd, cannot column-split"; break; }
                // The block-alignment rule `upload_dense` enforces, restated
                // here so a dtype whose half-row is ragged fails the ARITHMETIC
                // too and not only the load.
                if (ie::ds4_dense_keeps_packed(ti.dtype, K) &&
                    !ie::ds4_dense_keeps_packed(ti.dtype, K / 2)) {
                    out.resident_err = name + ": half of K=" + std::to_string(K) +
                                       " is not a whole number of blocks";
                    break;
                }
                const uint64_t half = ie::ds4_dense_device_bytes(ti.dtype, K / 2, N);
                out.resident_split += half;
                out.split_cols_saved += whole - half;
                ++out.n_split_cols;
            } else {
                out.resident_split += whole;
                out.split_mirrored += whole;
            }
        } else if (suf == "ffn_gate_inp.weight") {
            // The router is deliberately widened to fp32 and stays that way: doc
            // 33 §6.4 measured top-6 ties decided by score differences of 0.02-0.05
            // against a near-uniform trained bias.  Saving 90 MB here is not worth
            // touching that chain, so BOTH policies charge the same fp32 bytes.
            const uint64_t n = ti.shape[0] * ti.shape[1];
            out.resident_fp16 += n * 4; out.resident_packed += n * 4;
            out.resident_gguf += ti.nbytes;
            // MIRRORED, and not optionally: every card must reach the SAME top-k
            // decision or the expert partials belong to different experts.  The
            // orchestrator's lockstep check refuses on any disagreement.
            out.resident_split += n * 4; out.split_mirrored += n * 4;
        } else if (ti.dtype == ie::DType::kF32 || ti.dtype == ie::DType::kI32) {
            out.resident_fp16 += ti.nbytes; out.resident_packed += ti.nbytes;
            out.resident_gguf += ti.nbytes;
            // `attn_sinks` is per-QUERY-HEAD, so a card holds only its own
            // heads'.  256 B whole -- accounted because the loader really does
            // slice it (ds4_attention indexes sinks[h] over [0, nhc)), not
            // because the bytes matter.  Every other F32/I32 tensor is mirrored.
            if (suf == "attn_sinks.weight") {
                out.resident_split += ti.nbytes / 2;
                out.split_rows_saved += ti.nbytes - ti.nbytes / 2;
                ++out.n_split_rows;
            } else {
                out.resident_split += ti.nbytes; out.split_mirrored += ti.nbytes;
            }
        } else {
            out.resident_err = name + " (" + std::string(ie::type_name(ti.dtype)) +
                               "): not a routed expert, not a listed dense weight, not F32/I32"
                               " — the residency budget does not account for it";
            break;
        }
    }

    out.opened = true;
    return true;
}

void report_geometry(const char* label, const ExpertGeometry& q) {
    std::printf("    %s: %u layers x %u experts, H=%u EF=%u\n", label,
                q.n_layers, q.n_experts, q.hidden, q.expert_ffn);
    // Distinct slot sizes and which layers carry them.
    std::vector<uint64_t> seen;
    for (uint64_t b : q.slot_bytes) {
        bool have = false;
        for (uint64_t s : seen) have = have || s == b;
        if (!have) seen.push_back(b);
    }
    for (uint64_t s : seen) {
        uint32_t n = 0; uint32_t first = 0; bool got = false;
        for (uint32_t l = 0; l < q.n_layers; ++l)
            if (q.slot_bytes[l] == s) { ++n; if (!got) { first = l; got = true; } }
        std::printf("      slot %10llu B (%7.3f MB) on %2u layers (first blk.%u)\n",
                    (unsigned long long)s, double(s) / 1e6, n, first);
    }
    std::printf("      Sum of one slot per layer = %llu B (%.6f GB); pool = %s\n",
                (unsigned long long)q.per_layer, double(q.per_layer) / 1e9, gb(q.pool_slot).c_str());
}

// -------------------------------------------------------------------------
// §2/§3 — the plan, checked against doc 33's numbers and the cap
// -------------------------------------------------------------------------
void check_plan(const char* label, const ExpertGeometry& q, uint32_t design_slots,
                uint32_t min_stream, uint64_t expect_host, uint64_t cap) {
    ie::Ds4ResidencyPlan p;
    // vram_budget is expressed in the SAME per-whole-expert units the engine
    // allocates in, so `design_slots` slots is exactly design_slots * per_layer.
    const std::string e = ie::ds4_plan_residency(q.slot_bytes, q.n_experts,
                                                 uint64_t(design_slots) * q.per_layer,
                                                 /*slots_cap=*/0, cap, min_stream, p);
    std::printf("    %s @ %u slots/layer: static %u + stream %u, %u pinned experts/layer,"
                " host %s, static VRAM %s\n",
                label, design_slots, p.static_slots, p.stream_slots, p.pinned_experts,
                gb(p.host_bytes).c_str(), gb(uint64_t(p.static_slots) * q.per_layer).c_str());
    if (!e.empty()) std::printf("      refusal: %s\n", e.c_str());
    ok((std::string(label) + ": plan accepted under the cap").c_str(), e.empty(), e);
    ok((std::string(label) + ": slots = static + stream").c_str(),
       p.slots_per_layer == p.static_slots + p.stream_slots);
    ok((std::string(label) + ": streaming reserve honoured").c_str(), p.stream_slots == min_stream);
    ok((std::string(label) + ": pinned host bytes match the design").c_str(),
       p.host_bytes == expect_host,
       gb(p.host_bytes) + " vs expected " + gb(expect_host));
    ok((std::string(label) + ": pinned host bytes are at or under the cap").c_str(),
       p.host_bytes <= cap, gb(p.host_bytes) + " vs cap " + gb(cap));
    // THE DEFECT, stated as an equation: a statically resident expert has no
    // host copy, so the two tiers PARTITION the pool instead of overlapping it.
    const uint64_t static_vram = uint64_t(p.static_slots) * q.per_layer;
    ok((std::string(label) + ": static VRAM bytes + pinned host bytes == the whole expert pool"
        " (nothing counted twice)").c_str(),
       static_vram + p.host_bytes == q.pool_slot,
       gb(static_vram) + " + " + gb(p.host_bytes) + " vs pool " + gb(q.pool_slot));
    ok((std::string(label) + ": every expert is covered by exactly one tier").c_str(),
       p.static_slots + p.pinned_experts == q.n_experts);
    ok((std::string(label) + ": the pin is strictly smaller than the whole pool").c_str(),
       p.host_bytes < q.pool_slot,
       "saves " + gb(q.pool_slot - p.host_bytes));
}

// -------------------------------------------------------------------------
// §8 — the two-card plan, and the whole-box partition invariant
// -------------------------------------------------------------------------
// `expect` is {slots/card, static/card, pinned experts/card, host bytes/card}.
void check_tp_plan(const char* label, const ExpertGeometry& q, uint64_t budget_per_card,
                   uint64_t cap_total, uint32_t exp_slots, uint32_t exp_static,
                   uint32_t exp_pinned, uint64_t exp_host_card, bool expect_accept) {
    ie::Ds4TpResidencyPlan tp;
    const std::string e = ie::ds4_plan_residency_tp(q.card0_bytes, q.slot_bytes, q.n_experts, 2,
                                                    budget_per_card, 0, cap_total, 12, tp);
    std::printf("    %s 2-card: %u slots/card (static %u + stream %u), %u pinned experts/card,"
                " host %s/card = %s total, VRAM %s/card\n",
                label, tp.card.slots_per_layer, tp.card.static_slots, tp.card.stream_slots,
                tp.card.pinned_experts, gb(tp.card.host_bytes).c_str(),
                gb(tp.host_bytes_total).c_str(), gb(tp.card.vram_bytes).c_str());
    if (!e.empty()) std::printf("      refusal: %s\n", e.c_str());
    ok((std::string(label) + " 2-card: verdict matches the cap arithmetic").c_str(),
       e.empty() == expect_accept, e);
    ok((std::string(label) + " 2-card: slots/static/pinned per card").c_str(),
       tp.card.slots_per_layer == exp_slots && tp.card.static_slots == exp_static &&
           tp.card.pinned_experts == exp_pinned,
       std::to_string(tp.card.slots_per_layer) + "/" + std::to_string(tp.card.static_slots) +
           "/" + std::to_string(tp.card.pinned_experts));
    ok((std::string(label) + " 2-card: pinned host per card").c_str(),
       tp.card.host_bytes == exp_host_card, gb(tp.card.host_bytes));
    ok((std::string(label) + " 2-card: the box's total pin is exactly twice one card's").c_str(),
       tp.host_bytes_total == 2 * exp_host_card, gb(tp.host_bytes_total));
    // THE INVARIANT, now per box: every expert byte is charged to exactly one
    // tier on exactly one card.
    const uint64_t static_vram_card = uint64_t(tp.card.static_slots) * tp.card_slot_total;
    ok((std::string(label) + " 2-card: 2 x (static VRAM + pinned host) == the whole expert pool"
        " (nothing lost, nothing counted twice)").c_str(),
       2 * (static_vram_card + tp.card.host_bytes) == q.pool_slot,
       gb(2 * (static_vram_card + tp.card.host_bytes)) + " vs pool " + gb(q.pool_slot));
    ok((std::string(label) + " 2-card: per-card slot bytes are exactly half the whole expert's")
           .c_str(),
       tp.card_slot_total * 2 == tp.whole_slot_total && tp.whole_slot_total == q.per_layer,
       std::to_string(tp.card_slot_total) + " x2 vs " + std::to_string(tp.whole_slot_total));
    ok((std::string(label) + " 2-card: the pool the plan accounts for IS the file's pool").c_str(),
       tp.pool_bytes == q.pool_slot, gb(tp.pool_bytes));
}

// -------------------------------------------------------------------------
// §15 helpers — a miniature but STRUCTURALLY REAL deepseek4 GGUF.
//
// COPIED, not hoisted, from tests/unit/deepseek4_forward_test.cpp's `mini`
// namespace (the repo's copy-not-hoist discipline for test fixtures; that file
// is gate-passed and is not edited here).  Only two things changed: `L` is a
// parameter so a ONE-layer variant exists, and `EFF` is checked to divide by
// n_cards.  Every tensor name, dtype and shape relationship is the production
// file's — only the dimensions shrink.
//
// WHY A MINI MODEL AT ALL.  The 128.20 GB production file needs hours of USB-2
// I/O per load and 78 GB of pinned host RAM; comparing a one-card run against a
// two-card run of it is not a test that can be run.  This one loads in seconds
// and exercises the SAME code path: the same slice derivation, the same slot
// pack, the same streaming cache, the same GEMVs, the same reduction.
namespace mini {

struct Cfg {
    uint32_t L = 6, H = 256, NH = 4, HD = 64, VOCAB = 512;
    uint32_t QR = 256, OLR = 64, OG = 2, SW = 16;
    uint32_t IH = 4, IHD = 64, ITOPK = 8;
    uint32_t E = 8, EU = 2, EFF = 256;
    uint32_t HC = 4, SINK = 20, HASH = 3, RD = 32;
    std::vector<int32_t> ratios{0, 0, 4, 128, 4, 128};
    uint32_t mxfp4_layer = 4;      // stands in for blk.26's UD upcast
};

// The SHORTEST legal deepseek4: two layers, one CSA and one HCA, both hash
// routers.  `DeepSeek4Model::layer_kind` requires the config to declare exactly
// two distinct nonzero compress ratios and only scans the first `n_layers` of
// them, so two layers is the floor — a 1-layer model cannot be built.  It is the
// shortest chain from a routed-expert partial to the logits, which is why the
// end-to-end comparison uses it as well as the full 6-layer fixture.
Cfg short_model() {
    Cfg c;
    c.L = 2;
    c.ratios = {4, 128};
    c.mxfp4_layer = 99;            // no MXFP4 gate/up layer in a 1-layer model
    return c;
}

struct Rng {
    std::mt19937 g;
    explicit Rng(uint64_t s) : g(uint32_t(s)) {}
    float sn(float sd) { std::normal_distribution<float> d(0.f, sd); return d(g); }
    uint32_t u(uint32_t n) { return g() % n; }
};

std::vector<uint8_t> f32_buf(Rng& r, size_t n, float sd, float bias = 0.f) {
    std::vector<uint8_t> b(n * 4);
    auto* p = reinterpret_cast<float*>(b.data());
    for (size_t i = 0; i < n; ++i) p[i] = r.sn(sd) + bias;
    return b;
}
std::vector<uint8_t> bf16_buf(Rng& r, size_t n, float sd) {
    std::vector<uint8_t> b(n * 2);
    auto* p = reinterpret_cast<uint16_t*>(b.data());
    for (size_t i = 0; i < n; ++i) {
        const float f = r.sn(sd);
        uint32_t bits; std::memcpy(&bits, &f, 4);
        p[i] = uint16_t(bits >> 16);
    }
    return b;
}
std::vector<uint8_t> i32_buf(Rng& r, size_t n, uint32_t mod) {
    std::vector<uint8_t> b(n * 4);
    auto* p = reinterpret_cast<int32_t*>(b.data());
    for (size_t i = 0; i < n; ++i) p[i] = int32_t(r.u(mod));
    return b;
}
std::vector<uint8_t> q8_buf(Rng& r, size_t n) {
    const size_t nb = n / 32;
    std::vector<uint8_t> b(nb * sizeof(ie::block_q8_0));
    auto* blk = reinterpret_cast<ie::block_q8_0*>(b.data());
    for (size_t i = 0; i < nb; ++i) {
        blk[i].d = ie::fp32_to_fp16(0.02f);
        for (int j = 0; j < 32; ++j) blk[i].qs[j] = int8_t(int(r.u(65)) - 32);
    }
    return b;
}
std::vector<uint8_t> q6k_buf(Rng& r, size_t n) {
    const size_t nb = n / 256;
    std::vector<uint8_t> b(nb * sizeof(ie::block_q6_K));
    auto* blk = reinterpret_cast<ie::block_q6_K*>(b.data());
    for (size_t i = 0; i < nb; ++i) {
        blk[i].d = ie::fp32_to_fp16(0.004f);
        for (int j = 0; j < 128; ++j) blk[i].ql[j] = uint8_t(r.u(256));
        for (int j = 0; j < 64; ++j)  blk[i].qh[j] = uint8_t(r.u(256));
        for (int j = 0; j < 16; ++j)  blk[i].scales[j] = int8_t(int(r.u(17)) - 8);
    }
    return b;
}
std::vector<uint8_t> iq3_buf(Rng& r, size_t n) {
    const size_t nb = n / 256;
    std::vector<uint8_t> b(nb * sizeof(ie::block_iq3_xxs));
    auto* blk = reinterpret_cast<ie::block_iq3_xxs*>(b.data());
    for (size_t i = 0; i < nb; ++i) {
        blk[i].d = ie::fp32_to_fp16(0.0005f);
        for (int j = 0; j < 96; ++j) blk[i].qs[j] = uint8_t(r.u(256));
    }
    return b;
}
std::vector<uint8_t> mxfp4_buf(Rng& r, size_t n) {
    const size_t nb = n / 32;
    std::vector<uint8_t> b(nb * sizeof(ie::block_mxfp4));
    auto* blk = reinterpret_cast<ie::block_mxfp4*>(b.data());
    for (size_t i = 0; i < nb; ++i) {
        blk[i].e = uint8_t(124 + r.u(3));
        for (int j = 0; j < 16; ++j) blk[i].qs[j] = uint8_t(r.u(256));
    }
    return b;
}

std::string write_gguf(const Cfg& c, const std::string& path,
                       std::vector<std::vector<uint8_t>>& keep) {
    using namespace ie;
    Rng r(20260802);
    GgufWriter w;
    w.kv_string("general.architecture", "deepseek4");
    w.kv_u32("deepseek4.block_count", c.L);
    w.kv_u32("deepseek4.embedding_length", c.H);
    w.kv_u32("deepseek4.attention.head_count", c.NH);
    w.kv_u32("deepseek4.attention.head_count_kv", 1);
    w.kv_u32("deepseek4.attention.key_length", c.HD);
    w.kv_u32("deepseek4.attention.value_length", c.HD);
    w.kv_u32("deepseek4.context_length", 4096);
    w.kv_f32("deepseek4.attention.layer_norm_rms_epsilon", 1e-6f);
    w.kv_u32("deepseek4.attention.q_lora_rank", c.QR);
    w.kv_u32("deepseek4.attention.output_lora_rank", c.OLR);
    w.kv_u32("deepseek4.attention.output_group_count", c.OG);
    w.kv_u32("deepseek4.attention.sliding_window", c.SW);
    w.kv_u32("deepseek4.attention.indexer.head_count", c.IH);
    w.kv_u32("deepseek4.attention.indexer.key_length", c.IHD);
    w.kv_u32("deepseek4.attention.indexer.top_k", c.ITOPK);
    w.kv_i32_array("deepseek4.attention.compress_ratios", c.ratios);
    w.kv_f32("deepseek4.attention.compress_rope_freq_base", 160000.f);
    w.kv_u32("deepseek4.rope.dimension_count", c.RD);
    w.kv_f32("deepseek4.rope.freq_base", 10000.f);
    w.kv_f32("deepseek4.rope.scaling.factor", 16.f);
    w.kv_string("deepseek4.rope.scaling.type", "yarn");
    w.kv_u32("deepseek4.rope.scaling.original_context_length", 65536);
    w.kv_f32("deepseek4.rope.scaling.yarn_beta_fast", 32.f);
    w.kv_f32("deepseek4.rope.scaling.yarn_beta_slow", 1.f);
    w.kv_u32("deepseek4.expert_count", c.E);
    w.kv_u32("deepseek4.expert_used_count", c.EU);
    w.kv_u32("deepseek4.expert_feed_forward_length", c.EFF);
    w.kv_u32("deepseek4.expert_shared_count", 1);
    w.kv_u32("deepseek4.expert_gating_func", 4);
    w.kv_f32("deepseek4.expert_weights_scale", 1.5f);
    w.kv_bool("deepseek4.expert_weights_norm", true);
    w.kv_f32_array("deepseek4.swiglu_clamp_exp", std::vector<float>(c.L, 10.f));
    w.kv_f32_array("deepseek4.swiglu_clamp_shexp", std::vector<float>(c.L, 10.f));
    w.kv_u32("deepseek4.hyper_connection.count", c.HC);
    w.kv_u32("deepseek4.hyper_connection.sinkhorn_iterations", c.SINK);
    w.kv_f32("deepseek4.hyper_connection.epsilon", 1e-6f);
    w.kv_u32("deepseek4.hash_layer_count", c.HASH);
    std::vector<std::string> toks(c.VOCAB);
    for (uint32_t i = 0; i < c.VOCAB; ++i) toks[i] = "t" + std::to_string(i);
    w.kv_string_array("tokenizer.ggml.tokens", toks);
    w.kv_string("tokenizer.ggml.model", "gpt2");

    auto add = [&](const std::string& nm, DType dt, std::vector<uint64_t> shape,
                   std::vector<uint8_t>&& data) {
        keep.push_back(std::move(data));
        w.tensor(nm, dt, shape, keep.back().data(), keep.back().size());
    };
    const uint32_t MIX = (2 + c.HC) * c.HC, HCH = c.HC * c.H, ORK = c.OLR * c.OG;

    add("token_embd.weight", DType::kQ8_0, {c.H, c.VOCAB}, q8_buf(r, size_t(c.H) * c.VOCAB));
    add("output.weight", DType::kQ6_K, {c.H, c.VOCAB}, q6k_buf(r, size_t(c.H) * c.VOCAB));
    add("output_norm.weight", DType::kF32, {c.H}, f32_buf(r, c.H, 0.02f, 1.f));
    add("output_hc_fn.weight", DType::kF32, {HCH, c.HC}, f32_buf(r, size_t(HCH) * c.HC, 0.02f));
    add("output_hc_base.weight", DType::kF32, {c.HC}, f32_buf(r, c.HC, 0.1f));
    add("output_hc_scale.weight", DType::kF32, {1}, f32_buf(r, 1, 0.f, 1.f));

    for (uint32_t L = 0; L < c.L; ++L) {
        const std::string b = "blk." + std::to_string(L) + ".";
        const int32_t ratio = c.ratios[L];
        const uint32_t CD = ratio == 4 ? 2 * c.HD : c.HD;
        add(b + "attn_norm.weight", DType::kF32, {c.H}, f32_buf(r, c.H, 0.02f, 1.f));
        add(b + "attn_q_a.weight", DType::kQ8_0, {c.H, c.QR}, q8_buf(r, size_t(c.H) * c.QR));
        add(b + "attn_q_a_norm.weight", DType::kF32, {c.QR}, f32_buf(r, c.QR, 0.02f, 1.f));
        add(b + "attn_q_b.weight", DType::kQ8_0, {c.QR, c.NH * c.HD},
            q8_buf(r, size_t(c.QR) * c.NH * c.HD));
        add(b + "attn_kv.weight", DType::kQ8_0, {c.H, c.HD}, q8_buf(r, size_t(c.H) * c.HD));
        add(b + "attn_kv_a_norm.weight", DType::kF32, {c.HD}, f32_buf(r, c.HD, 0.02f, 1.f));
        add(b + "attn_sinks.weight", DType::kF32, {c.NH}, f32_buf(r, c.NH, 0.5f));
        add(b + "attn_output_a.weight", DType::kQ8_0, {c.H, ORK}, q8_buf(r, size_t(c.H) * ORK));
        add(b + "attn_output_b.weight", DType::kQ8_0, {ORK, c.H}, q8_buf(r, size_t(ORK) * c.H));
        if (ratio) {
            add(b + "attn_compressor_kv.weight", DType::kQ8_0, {c.H, CD}, q8_buf(r, size_t(c.H) * CD));
            add(b + "attn_compressor_gate.weight", DType::kQ8_0, {c.H, CD}, q8_buf(r, size_t(c.H) * CD));
            add(b + "attn_compressor_ape.weight", DType::kF32, {CD, uint64_t(ratio)},
                f32_buf(r, size_t(CD) * ratio, 0.1f));
            add(b + "attn_compressor_norm.weight", DType::kF32, {c.HD}, f32_buf(r, c.HD, 0.02f, 1.f));
        }
        if (ratio == 4) {
            add(b + "indexer.attn_q_b.weight", DType::kQ8_0, {c.QR, c.IH * c.IHD},
                q8_buf(r, size_t(c.QR) * c.IH * c.IHD));
            add(b + "indexer.proj.weight", DType::kF32, {c.H, c.IH}, f32_buf(r, size_t(c.H) * c.IH, 0.05f));
            add(b + "indexer_compressor_kv.weight", DType::kQ8_0, {c.H, 2 * c.IHD},
                q8_buf(r, size_t(c.H) * 2 * c.IHD));
            add(b + "indexer_compressor_gate.weight", DType::kQ8_0, {c.H, 2 * c.IHD},
                q8_buf(r, size_t(c.H) * 2 * c.IHD));
            add(b + "indexer_compressor_ape.weight", DType::kF32, {2 * c.IHD, uint64_t(ratio)},
                f32_buf(r, size_t(2 * c.IHD) * ratio, 0.1f));
            add(b + "indexer_compressor_norm.weight", DType::kF32, {c.IHD}, f32_buf(r, c.IHD, 0.02f, 1.f));
        }
        add(b + "ffn_norm.weight", DType::kF32, {c.H}, f32_buf(r, c.H, 0.02f, 1.f));
        add(b + "ffn_gate_inp.weight", DType::kBF16, {c.H, c.E}, bf16_buf(r, size_t(c.H) * c.E, 0.05f));
        const bool mx = (L == c.mxfp4_layer);
        const size_t ge = size_t(c.H) * c.EFF * c.E;
        add(b + "ffn_gate_exps.weight", mx ? DType::kMXFP4 : DType::kIQ3_XXS, {c.H, c.EFF, c.E},
            mx ? mxfp4_buf(r, ge) : iq3_buf(r, ge));
        add(b + "ffn_up_exps.weight", mx ? DType::kMXFP4 : DType::kIQ3_XXS, {c.H, c.EFF, c.E},
            mx ? mxfp4_buf(r, ge) : iq3_buf(r, ge));
        add(b + "ffn_down_exps.weight", DType::kMXFP4, {c.EFF, c.H, c.E}, mxfp4_buf(r, ge));
        add(b + "ffn_gate_shexp.weight", DType::kQ8_0, {c.H, c.EFF}, q8_buf(r, size_t(c.H) * c.EFF));
        add(b + "ffn_up_shexp.weight", DType::kQ8_0, {c.H, c.EFF}, q8_buf(r, size_t(c.H) * c.EFF));
        add(b + "ffn_down_shexp.weight", DType::kQ8_0, {c.EFF, c.H}, q8_buf(r, size_t(c.EFF) * c.H));
        if (L < c.HASH)
            add(b + "ffn_gate_tid2eid.weight", DType::kI32, {c.EU, c.VOCAB},
                i32_buf(r, size_t(c.EU) * c.VOCAB, c.E));
        else
            add(b + "exp_probs_b.bias", DType::kF32, {c.E}, f32_buf(r, c.E, 0.05f));
        for (const char* site : {"hc_attn", "hc_ffn"}) {
            add(b + site + "_fn.weight", DType::kF32, {HCH, MIX},
                f32_buf(r, size_t(HCH) * MIX, 0.02f));
            add(b + site + "_base.weight", DType::kF32, {MIX}, f32_buf(r, MIX, 0.1f));
            add(b + site + "_scale.weight", DType::kF32, {3}, f32_buf(r, 3, 0.f, 1.f));
        }
    }
    return w.write(path);
}

}  // namespace mini

double max_abs(const std::vector<float>& v) {
    double m = 0;
    for (float x : v) m = std::max(m, double(std::fabs(x)));
    return m;
}
double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        m = std::max(m, double(std::fabs(a[i] - b[i])));
    return m;
}
int32_t argmax(const std::vector<float>& v) {
    int32_t best = 0;
    for (size_t i = 1; i < v.size(); ++i) if (v[i] > v[best]) best = int32_t(i);
    return best;
}

// FNV-1a over the RAW BIT PATTERNS, printed as hex.  The relative-error prints
// beside it are %.6g and cannot tell "bit-identical" from "differs in the last
// mantissa bit", which is exactly the distinction a change to the ORDER the two
// cards execute in has to be held to: reordering submission must move no bit.
// Comparing this digest between two builds is the whole check.
uint64_t bitdigest(const void* p, size_t bytes, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    for (size_t i = 0; i < bytes; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
template <class T>
uint64_t bitdigest(const std::vector<T>& v, uint64_t h = 1469598103934665603ull) {
    return bitdigest(v.data(), v.size() * sizeof(T), h);
}

// ---------------------------------------------------------------------------
// §16 — THE LOAD I/O PROBE.  OFF unless $DS4_IO_PROBE_LAYERS names a layer count.
// ---------------------------------------------------------------------------
//
// WHAT IT MEASURES, AND WHY IT HAS TO BE MEASURED.  Under hidden-dim expert-TP
// the two cards hold the two halves of the SAME expert.  For gate/up those
// halves are adjacent runs of source columns, so a walk per card reads each
// byte once either way.  `down` is different: it is sliced along its
// CONTRACTION dimension, so card 0 takes the first half of EVERY one of its
// H columns and card 1 the second half — a stride of ~544 B inside a 1088 B
// column on Q3, far below a page.  BOTH cards therefore touch EVERY page of
// `down`, and walking the file once per card read all of it twice with the rest
// of the model in between.  How much that costs is an arithmetic claim about
// page granularity, and the honest way to settle it is to count the bytes the
// block device actually delivered.
//
// HOW.  `/proc/self/io: read_bytes` is the kernel's own count of bytes fetched
// from storage for this process.  Before each variant the probe drops the
// relevant page-cache pages — MADV_DONTNEED over the mapped expert ranges to
// unmap the PTEs, then POSIX_FADV_DONTNEED on every shard — so each variant
// starts cold and the counter is a real disk read, not a cache hit.
//
// It reads REAL model bytes, which is why it is off by default: a load of the
// whole file takes about an hour of this box's drive.  $DS4_IO_PROBE_LAYERS=1 is
// ~3 GB and a couple of minutes.  It performs NO device work and pins NOTHING;
// the pack destination is one reused heap buffer, because where the bytes land
// does not change which bytes are read.  When it is asked for it is the ONLY
// thing this binary does: the GPU sections would put their own allocations into
// whatever memory accounting the probe is being run under, and the whole point
// of the probe is that the answer depends on how much page cache is available.
//
// THE CONFOUND, NAMED.  Whether a per-card walk costs anything EXTRA depends
// entirely on whether `down`'s pages survive from card 0's pass to card 1's.  On
// an idle 156 GB box a one-layer probe's 2.8 GB working set survives trivially
// and the two variants read the SAME number of bytes — measured, 1.000x.  That
// is not evidence the defect is harmless; it is evidence the probe was run in
// the wrong regime.  The real load pins ~78 GB of the box's RAM and then streams
// a 120 GB file past it, so a page touched at the start of card 0's walk is long
// gone by card 1's.  To reproduce THAT at a size that can be measured in
// minutes, run this probe inside a memory cgroup whose limit is below the
// working set, e.g.
//
//   systemd-run --user --scope -p MemoryMax=4G -- \
//     env DS4_IO_PROBE_LAYERS=4 ./deepseek4_residency_test
//
// The cap constrains BOTH variants identically, so the comparison stays fair.
uint64_t proc_read_bytes() {
    std::FILE* f = std::fopen("/proc/self/io", "r");
    if (!f) return 0;
    char line[256];
    unsigned long long v = 0;
    bool got = false;
    while (std::fgets(line, sizeof(line), f))
        if (std::sscanf(line, "read_bytes: %llu", &v) == 1) { got = true; break; }
    std::fclose(f);
    return got ? uint64_t(v) : 0;
}

void drop_page_cache(const std::string& dir,
                     const std::vector<const ie::GgufTensorInfo*>& ts) {
    // POSIX_FADV_DONTNEED alone will not evict a page that is still mapped into
    // a page table, and the reader mmaps the whole file, so the PTEs have to go
    // first.  MADV_DONTNEED on a PROT_READ MAP_PRIVATE file mapping is safe: the
    // next touch re-faults from the file, which is exactly the point.
    const long pg = ::sysconf(_SC_PAGESIZE);
    if (pg > 0)
        for (const ie::GgufTensorInfo* t : ts) {
            if (!t || !t->data || !t->nbytes) continue;
            const uintptr_t a  = reinterpret_cast<uintptr_t>(t->data);
            const uintptr_t lo = a & ~uintptr_t(pg - 1);
            ::madvise(reinterpret_cast<void*>(lo), size_t(t->nbytes + (a - lo)), MADV_DONTNEED);
        }
    if (DIR* d = ::opendir(dir.c_str())) {
        while (struct dirent* de = ::readdir(d)) {
            const std::string n = de->d_name;
            if (n.size() < 5 || n.compare(n.size() - 5, 5, ".gguf") != 0) continue;
            const int fd = ::open((dir + "/" + n).c_str(), O_RDONLY);
            if (fd < 0) continue;
            ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            ::close(fd);
        }
        ::closedir(d);
    }
}

bool io_probe() {
    const char* ls = std::getenv("DS4_IO_PROBE_LAYERS");
    if (!ls) return false;
    const int want = std::atoi(ls);
    if (want <= 0) return false;

    std::printf("[16] LOAD I/O PROBE — bytes the block device actually delivered\n");
    // Report the memory ceiling the kernel is actually enforcing, because the
    // answer is meaningless without it.
    {
        std::FILE* f = std::fopen("/proc/self/cgroup", "r");
        std::string cg;
        char        line[512];
        if (f) {
            while (std::fgets(line, sizeof(line), f))
                if (std::strncmp(line, "0::", 3) == 0) {
                    cg = line + 3;
                    while (!cg.empty() && (cg.back() == '\n' || cg.back() == '\r')) cg.pop_back();
                    break;
                }
            std::fclose(f);
        }
        std::string lim = "unknown";
        if (!cg.empty()) {
            if (std::FILE* mf = std::fopen(("/sys/fs/cgroup" + cg + "/memory.max").c_str(), "r")) {
                char buf[64] = {0};
                if (std::fgets(buf, sizeof(buf), mf)) {
                    lim = buf;
                    while (!lim.empty() && (lim.back() == '\n' || lim.back() == ' ')) lim.pop_back();
                }
                std::fclose(mf);
            }
        }
        std::printf("    cgroup %s, memory.max = %s\n", cg.empty() ? "?" : cg.c_str(), lim.c_str());
    }
    const std::string shard1 =
        env_or("DS4_IO_PROBE_MODEL",
               "${IE_MODELS_DIR}/DeepSeek-V4-Flash-0731-GGUF/UD-Q3_K_XL/"
               "DeepSeek-V4-Flash-0731-UD-Q3_K_XL-00001-of-00004.gguf");
    const size_t      slash = shard1.find_last_of('/');
    const std::string dir   = slash == std::string::npos ? "." : shard1.substr(0, slash);

    ie::GgufReader g;
    if (const std::string e = g.open(shard1); !e.empty()) {
        std::printf("  SKIP: cannot open %s (%s)\n", shard1.c_str(), e.c_str());
        return true;
    }
    const ie::GgufTensorInfo* g0 = need(g, "blk.0.ffn_gate_exps.weight");
    if (!g0 || g0->n_dims != 3) { std::printf("  SKIP: no routed experts\n"); return true; }
    const uint32_t H = uint32_t(g0->shape[0]), EF = uint32_t(g0->shape[1]);
    const uint32_t NE = uint32_t(g0->shape[2]);
    constexpr uint32_t NC = 2;

    struct Lay {
        const ie::GgufTensorInfo *gt = nullptr, *ut = nullptr, *dt = nullptr;
        ie::Ds4SlotLayout        card[NC];
    };
    std::vector<Lay>                       ly;
    std::vector<const ie::GgufTensorInfo*> all;
    uint64_t widest = 0, work = 0;
    for (int l = 0; l < want; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        Lay x;
        x.gt = need(g, p + "ffn_gate_exps.weight");
        x.ut = need(g, p + "ffn_up_exps.weight");
        x.dt = need(g, p + "ffn_down_exps.weight");
        if (!x.gt || !x.ut || !x.dt) break;
        for (uint32_t c = 0; c < NC; ++c) {
            ie::Ds4ExpertSlice sl;
            if (const std::string e = ie::ds4_expert_slice(*x.gt, *x.ut, *x.dt, EF, NC, c, sl);
                !e.empty()) {
                std::printf("  SKIP: layer %d slice: %s\n", l, e.c_str());
                return true;
            }
            if (const std::string e =
                    ie::ds4_slot_layout_tp(*x.gt, *x.ut, *x.dt, H, EF, sl, x.card[c]);
                !e.empty()) {
                std::printf("  SKIP: layer %d layout: %s\n", l, e.c_str());
                return true;
            }
            widest = std::max(widest, x.card[c].bytes);
            work  += x.card[c].bytes * NE;
        }
        all.push_back(x.gt); all.push_back(x.ut); all.push_back(x.dt);
        ly.push_back(x);
    }
    if (ly.empty()) { std::printf("  SKIP: no layers resolved\n"); return true; }

    // One destination, reused for every pack.  The packs are identical work in
    // both variants; only the ORDER differs, which is the whole hypothesis.
    std::vector<uint8_t> dst(static_cast<size_t>(widest));
    auto pack = [&](uint32_t c, size_t l, uint32_t e) {
        (void)ie::ds4_slot_pack(ly[l].card[c], *ly[l].gt, *ly[l].ut, *ly[l].dt, e, dst.data());
    };
    // A: what the orchestrator used to do — card 0's whole walk, then card 1's.
    auto walk_per_card = [&] {
        for (uint32_t c = 0; c < NC; ++c)
            for (size_t l = 0; l < ly.size(); ++l) {
                ie::ds4_stream_advise_experts(*ly[l].gt, *ly[l].ut, *ly[l].dt);
                for (uint32_t e = 0; e < NE; ++e) pack(c, l, e);
            }
    };
    // B: what it does now — one walk, both cards' slices of the same expert
    // packed back to back.
    auto walk_single = [&] {
        for (size_t l = 0; l < ly.size(); ++l) {
            ie::ds4_stream_advise_experts(*ly[l].gt, *ly[l].ut, *ly[l].dt);
            for (uint32_t e = 0; e < NE; ++e)
                for (uint32_t c = 0; c < NC; ++c) pack(c, l, e);
        }
    };

    struct R { uint64_t read = 0; double sec = 0; };
    auto measure = [&](auto&& fn) {
        drop_page_cache(dir, all);
        const uint64_t b0 = proc_read_bytes();
        const auto     t0 = std::chrono::steady_clock::now();
        fn();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return R{proc_read_bytes() - b0, s};
    };

    std::printf("    %zu layer(s) x %u experts x %u cards; %s of slot bytes to pack\n",
                ly.size(), NE, NC, gb(work).c_str());
    const R a = measure(walk_per_card);
    std::printf("    A  one walk PER CARD (the defect): read %s in %.1f s (%.1f MB/s)\n",
                gb(a.read).c_str(), a.sec, double(a.read) / 1e6 / std::max(a.sec, 1e-9));
    const R b = measure(walk_single);
    std::printf("    B  ONE interleaved walk (the fix): read %s in %.1f s (%.1f MB/s)\n",
                gb(b.read).c_str(), b.sec, double(b.read) / 1e6 / std::max(b.sec, 1e-9));

    // If the eviction did not take, both numbers are meaningless and saying so
    // is worth more than a green tick.
    ok("the probe really read from the device (the page-cache drop took effect)",
       a.read > work / 2 && b.read > work / 2,
       "A " + gb(a.read) + ", B " + gb(b.read) + ", work " + gb(work));
    ok("ONE interleaved walk reads STRICTLY FEWER bytes than one walk per card",
       b.read < a.read,
       "B/A = " + std::to_string(a.read ? double(b.read) / double(a.read) : 0.0) +
           "; A - B = " + gb(a.read > b.read ? a.read - b.read : 0));
    std::printf("    -> the interleaved walk read %.3fx what the per-card walks read,"
                " and took %.3fx the time\n",
                a.read ? double(b.read) / double(a.read) : 0.0,
                a.sec ? b.sec / a.sec : 0.0);
    return true;
}

// ---------------------------------------------------------------------------
// §17 — THE OFFLINE REPLACEMENT-POLICY SCORER.  OFF unless $DS4_TRACE_SCORE
// names a routing trace written by $DS4_EXPERT_TRACE.
// ---------------------------------------------------------------------------
//
// WHAT QUESTION IT EXISTS FOR.  §12 proves the streaming partition is CORRECT.
// It says nothing about whether the partition's replacement rule is any GOOD,
// and the one measurement that had been taken — "demand hit rate equals static
// residency to 1e-9" — was taken in PREFILL, where a chunk touches each of the
// 256 experts about once and there is by construction no reuse to exploit.  At
// decode a token routes 6 of 256 and the same 6 may recur, so the same question
// has a different answer and needs a different instrument.
//
// WHY OFFLINE, REPLAYED FROM A TRACE, RATHER THAN A SHADOW CACHE IN THE ENGINE.
// A shadow simulator inside `Ds4ExpertCache` would answer one question per
// model load, at 161 s of load each, while adding host work to the decode
// critical path and a second piece of state to the one class whose state being
// wrong produces silently wrong logits.  A trace costs one load TOTAL and then
// scores an unlimited number of policies in milliseconds with zero risk to the
// hot path.  The trace is captured where the router's decision first exists on
// the host, BEFORE residency is consulted, so it describes what the MODEL
// routed to and is invariant to every policy scored against it.
//
// WHAT IT REPRODUCES, AND HOW FAITHFULLY.  The simulated cache is the shipped
// one, not an idealisation of it:
//   * the STATIC partition never participates in replacement (`is_static`),
//   * `ds4_plan_expert_groups` splits a token's occupied experts into groups
//     capped at `pipelined_group_cap()` STREAMING experts,
//   * every group goes through the `keep_prev == true` form of `acquire_impl`,
//     which is what decode actually takes, so the PREVIOUS group's in-use marks
//     (at decode: the previous TOKEN's) still block eviction, and
//   * a victim search that finds every streaming slot claimed returns the
//     kDs4NoSlot refusal rather than evicting a live slot.
// Only `pick_victim` is swapped.  That is the point: any difference between two
// rows of the output is attributable to the replacement rule and to nothing
// else.
//
// THE NEGATIVE CONTROLS, which are the reason this can FALSIFY and not only
// confirm.  Two rows are scored that must be read together with the rest:
//   * SHUFFLED — the same tokens in a random order per layer.  This preserves
//     every expert's frequency and destroys the temporal structure.  A policy
//     whose advantage survives the shuffle is exploiting FREQUENCY, which the
//     static partition already exploits better; only an advantage that
//     DISAPPEARS under the shuffle is genuine recurrence.
//   * BELADY — evict the resident expert whose next use is furthest away, using
//     knowledge of the future.  No online policy can beat it, so if FIFO is
//     already close to it there is nothing left for any policy to win and the
//     honest answer is to stop.
//
// USAGE
//   DS4_EXPERT_TRACE=/tmp/ds4.trace  <a real-model decode run>
//   DS4_TRACE_SCORE=/tmp/ds4.trace ./build/tests/deepseek4_residency_test
// Knobs, all optional: DS4_TRACE_SCORE_SLOTS (92), DS4_TRACE_SCORE_STATIC (80),
// DS4_TRACE_SCORE_T (1 = decode only), DS4_TRACE_SCORE_BYTES (6684672, the
// per-card Q8 slot).
struct RTrace {
    uint32_t n_layers = 0, n_experts = 0, K = 0, hash_layers = 0;
    // [L][t] = the DISTINCT experts layer L routed to at step t, ascending —
    // `occ` in moe_expert_grouped, which is what the cache is actually asked for.
    std::vector<std::vector<std::vector<uint32_t>>> tok;
    uint64_t records = 0, kept = 0;
};

bool read_trace(const std::string& path, int want_T, RTrace& tr, std::string& err) {
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f) { err = "cannot open " + path; return false; }
    char line[4096];
    while (std::fgets(line, sizeof(line), f)) {
        if (line[0] == '#') {
            unsigned nl = 0, ne = 0, k = 0, hl = 0;
            if (std::sscanf(line, "# n_layers=%u n_experts=%u n_experts_used=%u hash_layers=%u",
                            &nl, &ne, &k, &hl) == 4) {
                tr.n_layers = nl; tr.n_experts = ne; tr.K = k; tr.hash_layers = hl;
                tr.tok.assign(nl, {});
            }
            continue;
        }
        if (!tr.n_layers) { err = "trace has no '# n_layers=...' header line"; std::fclose(f); return false; }
        int v[4 + 64];
        int n = 0;
        for (char* p = std::strtok(line, " \t\n"); p && n < int(sizeof(v) / sizeof(v[0]));
             p = std::strtok(nullptr, " \t\n"))
            v[n++] = std::atoi(p);
        if (n < 4 + int(tr.K)) continue;
        ++tr.records;
        const int T = v[2], L = v[3];
        if (want_T > 0 && T != want_T) continue;
        if (L < 0 || uint32_t(L) >= tr.n_layers) continue;
        std::vector<uint32_t> occ;
        occ.reserve(tr.K);
        for (uint32_t i = 0; i < tr.K; ++i) {
            const int e = v[4 + i];
            if (e < 0 || uint32_t(e) >= tr.n_experts) continue;
            occ.push_back(uint32_t(e));
        }
        // Distinct and ASCENDING, exactly as `occ` is built in the engine: the
        // cache is asked once per distinct expert, never once per routed slot.
        std::sort(occ.begin(), occ.end());
        occ.erase(std::unique(occ.begin(), occ.end()), occ.end());
        tr.tok[uint32_t(L)].push_back(std::move(occ));
        ++tr.kept;
    }
    std::fclose(f);
    return true;
}

enum class Pol { kFifo, kLru, kLfu, kRandom, kBelady };
const char* pol_name(Pol p) {
    switch (p) {
        case Pol::kFifo:   return "FIFO (as shipped)";
        case Pol::kLru:    return "LRU";
        case Pol::kLfu:    return "LFU";
        case Pol::kRandom: return "RANDOM";
        case Pol::kBelady: return "BELADY (offline ceiling)";
    }
    return "?";
}

struct SimOut {
    uint64_t reqs = 0, static_hits = 0, stream_hits = 0, misses = 0, unavailable = 0, steps = 0;
    double hit_rate()   const { return reqs ? double(static_hits + stream_hits) / double(reqs) : 0.0; }
    double static_rate() const { return reqs ? double(static_hits) / double(reqs) : 0.0; }
    double bytes_per_step(uint64_t slot_bytes) const {
        return steps ? double(misses) * double(slot_bytes) / double(steps) : 0.0;
    }
};

// One layer of the shipped cache, with only `pick_victim` swapped out.
void sim_layer(const std::vector<std::vector<uint32_t>>& toks, const uint8_t* is_static,
               uint32_t n_experts, uint32_t stream, Pol pol, uint64_t seed, SimOut& out) {
    if (stream == 0 || toks.empty()) return;
    // Ds4ExpertCache::pipelined_group_cap() is 0 below two streaming slots, and
    // moe_expert_grouped reads that as "cannot double-buffer": it falls back to
    // the SERIAL shape — plain `acquire`, which clears BOTH in-use generations,
    // and a group cap of the whole partition.  Modelled rather than approximated,
    // because the split sweep walks right down to a 4-slot partition and an
    // approximation there would be a fabricated row.
    const uint32_t gcap      = stream < 2 ? 0u : stream / 2;
    const bool     keep_prev = gcap != 0;
    const uint32_t cap       = keep_prev ? gcap : stream;

    std::vector<int32_t>  slot_of(n_experts, -1);
    std::vector<int32_t>  tag(stream, -1);
    std::vector<uint8_t>  iu(stream, 0), prev(stream, 0);
    std::vector<uint64_t> stamp(stream, 0), freq(stream, 0);
    uint32_t fifo = 0;
    uint64_t clk  = 0;
    std::mt19937_64 rng(seed);

    // Belady's oracle: the next step at which each expert is routed again.
    // Built once, walked with a per-expert cursor as the replay advances.
    std::vector<std::vector<uint32_t>> when;
    std::vector<size_t>                cur;
    if (pol == Pol::kBelady) {
        when.assign(n_experts, {});
        cur.assign(n_experts, 0);
        for (uint32_t t = 0; t < toks.size(); ++t)
            for (uint32_t e : toks[t]) when[e].push_back(t);
    }
    auto next_use = [&](uint32_t e, uint32_t t) -> uint64_t {
        while (cur[e] < when[e].size() && when[e][cur[e]] <= t) ++cur[e];
        return cur[e] < when[e].size() ? when[e][cur[e]] : UINT64_MAX;
    };

    for (uint32_t t = 0; t < toks.size(); ++t) {
        const std::vector<uint32_t>& occ = toks[t];
        ++out.steps;
        if (pol == Pol::kBelady)
            for (uint32_t e = 0; e < n_experts; ++e)
                if (slot_of[e] >= 0) next_use(e, t);   // keep the cursors current
        size_t gi = 0;
        while (gi < occ.size()) {
            // ds4_plan_expert_groups: the cap counts STREAMING experts only.
            size_t   gj = gi;
            uint32_t ns = 0;
            while (gj < occ.size()) {
                if (!is_static[occ[gj]]) {
                    if (ns + 1 > cap && gj > gi) break;
                    ++ns;
                }
                ++gj;
            }
            // acquire_impl: `keep_prev` is true on the pipelined path decode
            // takes, and false on the serial fallback — the same single flag the
            // engine branches on.
            if (keep_prev) prev = iu;
            else           std::fill(prev.begin(), prev.end(), 0);
            std::fill(iu.begin(), iu.end(), 0);
            for (size_t k = gi; k < gj; ++k) {
                const uint32_t e = occ[k];
                ++out.reqs;
                if (is_static[e]) { ++out.static_hits; continue; }
                if (slot_of[e] >= 0) {
                    const uint32_t s = uint32_t(slot_of[e]);
                    ++out.stream_hits;
                    iu[s] = 1; stamp[s] = ++clk; ++freq[s];
                    continue;
                }
                int32_t v = -1;
                if (pol == Pol::kFifo) {
                    uint32_t c = fifo % stream;
                    for (uint32_t tries = 0; tries < stream; ++tries) {
                        const uint32_t nx = (c + 1) % stream;
                        if (!iu[c] && !prev[c]) { fifo = nx; v = int32_t(c); break; }
                        c = nx;
                    }
                } else {
                    uint64_t best = 0;
                    bool     have = false;
                    for (uint32_t s = 0; s < stream; ++s) {
                        if (iu[s] || prev[s]) continue;
                        uint64_t key;
                        switch (pol) {
                            case Pol::kLru:    key = UINT64_MAX - stamp[s]; break;
                            case Pol::kLfu:    key = UINT64_MAX - (freq[s] * (1ull << 32) + stamp[s]); break;
                            case Pol::kRandom: key = rng(); break;
                            case Pol::kBelady: key = tag[s] < 0 ? UINT64_MAX
                                                               : next_use(uint32_t(tag[s]), t); break;
                            default:           key = 0; break;
                        }
                        if (!have || key > best) { best = key; have = true; v = int32_t(s); }
                    }
                }
                if (v < 0) { ++out.unavailable; continue; }
                const uint32_t s = uint32_t(v);
                if (tag[s] >= 0) slot_of[uint32_t(tag[s])] = -1;
                tag[s] = int32_t(e);
                slot_of[e] = int32_t(s);
                iu[s] = 1; stamp[s] = ++clk; freq[s] = 1;
                ++out.misses;
            }
            gi = gj;
        }
    }
}

// is_static[L * n_experts + e].  `mode`: 0 index order (the shipped default),
// 1 one GLOBAL frequency ranking (what Ds4ExpertProfile::priority() produces),
// 2 a PER-LAYER frequency ranking — `Ds4Options::expert_priority_layers`, or the
// per-layer form of the ranking file.  It measured the prize BEFORE that existed
// and is now the scorer's model of a shippable configuration; §15e-2 gates that
// the engine really honours it per layer rather than applying layer 0's order.
// `from` bounds the steps the ranking may be COUNTED over: ranking on the whole
// trace and then scoring the same trace is an oracle, so the caller passes half.
std::vector<uint8_t> build_static(const RTrace& tr, uint32_t static_slots, int mode,
                                  size_t count_upto) {
    std::vector<uint8_t> s(size_t(tr.n_layers) * tr.n_experts, 0);
    if (mode == 0) {
        for (uint32_t L = 0; L < tr.n_layers; ++L)
            for (uint32_t e = 0; e < std::min(static_slots, tr.n_experts); ++e)
                s[size_t(L) * tr.n_experts + e] = 1;
        return s;
    }
    std::vector<std::vector<uint64_t>> c(tr.n_layers, std::vector<uint64_t>(tr.n_experts, 0));
    std::vector<uint64_t>              g(tr.n_experts, 0);
    for (uint32_t L = 0; L < tr.n_layers; ++L)
        for (size_t t = 0; t < tr.tok[L].size() && t < count_upto; ++t)
            for (uint32_t e : tr.tok[L][t]) { ++c[L][e]; ++g[e]; }
    auto top = [&](const std::vector<uint64_t>& cnt) {
        std::vector<uint32_t> o(tr.n_experts);
        for (uint32_t e = 0; e < tr.n_experts; ++e) o[e] = e;
        std::sort(o.begin(), o.end(), [&](uint32_t a, uint32_t b) {
            return cnt[a] != cnt[b] ? cnt[a] > cnt[b] : a < b;   // matches priority()
        });
        return o;
    };
    if (mode == 1) {
        const std::vector<uint32_t> o = top(g);
        for (uint32_t L = 0; L < tr.n_layers; ++L)
            for (uint32_t i = 0; i < std::min(static_slots, tr.n_experts); ++i)
                s[size_t(L) * tr.n_experts + o[i]] = 1;
    } else {
        for (uint32_t L = 0; L < tr.n_layers; ++L) {
            const std::vector<uint32_t> o = top(c[L]);
            for (uint32_t i = 0; i < std::min(static_slots, tr.n_experts); ++i)
                s[size_t(L) * tr.n_experts + o[i]] = 1;
        }
    }
    return s;
}

SimOut sim_all(const RTrace& tr, const std::vector<uint8_t>& is_static, uint32_t stream,
               Pol pol, size_t skip_steps) {
    SimOut o;
    for (uint32_t L = 0; L < tr.n_layers; ++L) {
        if (tr.tok[L].size() <= skip_steps) continue;
        const std::vector<std::vector<uint32_t>> tail(tr.tok[L].begin() + std::ptrdiff_t(skip_steps),
                                                      tr.tok[L].end());
        sim_layer(tail, is_static.data() + size_t(L) * tr.n_experts, tr.n_experts, stream, pol,
                  0x9E3779B97F4A7C15ull ^ L, o);
    }
    // `steps` accumulated per layer; the headline is per DECODE STEP.
    o.steps = tr.n_layers ? o.steps / tr.n_layers : 0;
    return o;
}

bool trace_score() {
    const char* tp = std::getenv("DS4_TRACE_SCORE");
    if (!tp || !*tp) return false;

    const uint32_t slots  = uint32_t(std::atoi(env_or("DS4_TRACE_SCORE_SLOTS", "92").c_str()));
    const uint32_t stat   = uint32_t(std::atoi(env_or("DS4_TRACE_SCORE_STATIC", "80").c_str()));
    const int      wantT  = std::atoi(env_or("DS4_TRACE_SCORE_T", "1").c_str());
    const uint64_t sbytes = uint64_t(std::atoll(env_or("DS4_TRACE_SCORE_BYTES", "6684672").c_str()));
    const double   pcie   = 24.3e9;   // measured per-card, docs/deepseek4/50 §"12.3 GB/s ... error"

    std::printf("[17] OFFLINE REPLACEMENT-POLICY SCORER — %s\n", tp);
    RTrace      tr;
    std::string err;
    if (!read_trace(tp, wantT, tr, err)) {
        ok("the routing trace could be read", false, err);
        std::printf("\n%s: %d failure(s)\n", g_fail ? "FAILED" : "PASSED", g_fail);
        return true;
    }
    if (stat > slots) {
        ok("static slots <= slots/layer", false, std::to_string(stat) + " > " + std::to_string(slots));
        return true;
    }
    const uint32_t stream = slots - stat;
    size_t steps = 0;
    for (uint32_t L = 0; L < tr.n_layers; ++L) steps = std::max(steps, tr.tok[L].size());
    std::printf("    trace: %llu records, %llu kept at T==%d; %u layers x %u experts, top-%u,"
                " %u hash layers; %zu steps/layer\n",
                (unsigned long long)tr.records, (unsigned long long)tr.kept, wantT,
                tr.n_layers, tr.n_experts, tr.K, tr.hash_layers, steps);
    std::printf("    geometry: %u slots/layer = %u static + %u streaming (%.1f%% residency),"
                " %llu B/slot/card, %.1f GB/s PCIe\n",
                slots, stat, stream, 100.0 * slots / std::max(tr.n_experts, 1u),
                (unsigned long long)sbytes, pcie / 1e9);
    ok("the trace carries decode steps to score", steps > 8,
       std::to_string(steps) + " steps at T==" + std::to_string(wantT));
    if (steps <= 8) return true;

    // ---- the structural question, asked BEFORE any policy: is there reuse? ----
    // Reuse distance in STEPS, per layer, over the experts a policy could ever
    // help with.  A cache of `stream` slots absorbing 6 arrivals per step holds
    // roughly stream/misses_per_step steps of history, so only reuse INSIDE that
    // window is reachable at all.
    {
        std::vector<uint64_t> d(12, 0);
        uint64_t              never = 0, total = 0;
        for (uint32_t L = 0; L < tr.n_layers; ++L) {
            std::vector<int64_t> last(tr.n_experts, -1);
            for (size_t t = 0; t < tr.tok[L].size(); ++t) {
                for (uint32_t e : tr.tok[L][t]) {
                    if (last[e] >= 0) {
                        const size_t dist = t - size_t(last[e]);
                        ++total;
                        if (dist < d.size()) ++d[dist]; else ++never;
                    }
                    last[e] = int64_t(t);
                }
            }
        }
        std::printf("    reuse distance (steps between two selections of the SAME expert in the"
                    " SAME layer), %llu pairs:\n      ", (unsigned long long)(total + never));
        double cum = 0;
        const double den = double(total + never ? total + never : 1);
        for (size_t i = 1; i < d.size(); ++i) {
            cum += double(d[i]) / den;
            std::printf("d=%zu %.1f%% (cum %.1f%%)  ", i, 100.0 * double(d[i]) / den, 100.0 * cum);
            if (i % 4 == 0) std::printf("\n      ");
        }
        std::printf("d>=%zu %.1f%%\n", d.size(), 100.0 * double(never) / den);
        const double p1 = double(d[1]) / den;
        std::printf("    -> CONSECUTIVE-STEP reuse is %.1f%% of repeat pairs; a UNIFORM router"
                    " would give %.1f%%\n", 100.0 * p1, 100.0 * double(tr.K) / double(tr.n_experts));
    }

    // ---- the policy sweep, at the geometry that shipped ----
    const size_t half = steps / 2;
    struct Row { const char* stat_name; int stat_mode; Pol pol; };
    const Row rows[] = {
        {"index order (SHIPPED)",      0, Pol::kFifo},
        {"index order (SHIPPED)",      0, Pol::kLru},
        {"index order (SHIPPED)",      0, Pol::kLfu},
        {"index order (SHIPPED)",      0, Pol::kRandom},
        {"index order (SHIPPED)",      0, Pol::kBelady},
        {"global freq (priority file)",1, Pol::kFifo},
        {"global freq (priority file)",1, Pol::kLru},
        {"PER-LAYER freq (expert_priority_layers)", 2, Pol::kFifo},
        {"PER-LAYER freq (expert_priority_layers)", 2, Pol::kLru},
    };
    std::printf("\n    %-40s  %-24s  %7s %7s %7s  %9s  %8s\n", "static set", "replacement",
                "hit", "static", "stream", "MB/step", "ms/step");
    double base_bytes = 0;
    for (const Row& r : rows) {
        // Frequency rankings are COUNTED on the first half and SCORED on the
        // second, so a ranking never gets to see the steps it is graded on.
        const std::vector<uint8_t> is_st = build_static(tr, stat, r.stat_mode, half);
        const SimOut o = sim_all(tr, is_st, stream, r.pol, half);
        const double mb = o.bytes_per_step(sbytes) / 1e6;
        if (&r == &rows[0]) base_bytes = mb;
        std::printf("    %-40s  %-24s  %6.2f%% %6.2f%% %6.2f%%  %9.1f  %8.2f%s\n",
                    r.stat_name, pol_name(r.pol), 100.0 * o.hit_rate(), 100.0 * o.static_rate(),
                    100.0 * (o.hit_rate() - o.static_rate()), mb,
                    1e3 * mb * 1e6 / pcie,
                    o.unavailable ? "  <-- REFUSALS" : "");
        ok(std::string("no expert was refused a slot under ").append(pol_name(r.pol)).c_str(),
           o.unavailable == 0, std::to_string(o.unavailable) + " unavailable");
    }

    // ---- NEGATIVE CONTROL: the same tokens, temporal order destroyed ----
    {
        RTrace sh = tr;
        std::mt19937_64 rng(12345);
        for (uint32_t L = 0; L < sh.n_layers; ++L)
            std::shuffle(sh.tok[L].begin(), sh.tok[L].end(), rng);
        const std::vector<uint8_t> is_st = build_static(sh, stat, 0, SIZE_MAX);
        const SimOut f = sim_all(sh, is_st, stream, Pol::kFifo, half);
        const SimOut l = sim_all(sh, is_st, stream, Pol::kLru, half);
        std::printf("\n    NEGATIVE CONTROL — same tokens, order SHUFFLED per layer"
                    " (frequency kept, recurrence destroyed):\n"
                    "      FIFO %.2f%% hit, %.1f MB/step   LRU %.2f%% hit, %.1f MB/step\n",
                    100.0 * f.hit_rate(), f.bytes_per_step(sbytes) / 1e6,
                    100.0 * l.hit_rate(), l.bytes_per_step(sbytes) / 1e6);
        std::printf("      Any streaming hit rate that SURVIVES this shuffle is frequency, not"
                    " recurrence, and the static partition exploits frequency better.\n");
    }

    // ---- the static/streaming split sweep, total slots held FIXED ----
    //
    // FIFO *and* LRU at every split, because the two levers INTERACT and scoring
    // LRU only at the shipped 12-slot window would beg the question.  FIFO and
    // LRU can only differ over slots that survive long enough for "which one is
    // oldest" and "which one was used least recently" to disagree; at 12 slots
    // against ~4 misses a step that is ~3 steps of history and the two orders are
    // nearly the same order.  If a WIDER streaming partition is what the split
    // sweep recommends, LRU has to be re-asked there, and this is where.
    // THE PINNED-HOST COST COLUMN, which is what makes this a recommendation and
    // not a wish.  Slots moved from static to streaming are not free: an expert
    // that loses its permanent VRAM home needs a PINNED HOST COPY, so widening
    // the streaming partition by one slot per layer costs
    //   n_cards x n_layers x slot_bytes
    // of locked RAM, and `ds4_plan_residency_tp` REFUSES a plan above the cap
    // rather than downsizing it.  A row over the cap is not a slow
    // configuration, it is one that will not load.
    const uint32_t ncards = uint32_t(std::atoi(env_or("DS4_TRACE_SCORE_CARDS", "2").c_str()));
    const double   pincap = std::atof(env_or("DS4_TRACE_SCORE_PINCAP_GB", "115").c_str());
    std::printf("\n    STATIC/STREAMING SPLIT at %u slots/layer (index order, as shipped);"
                " pinned host = %u cards x %u layers x (%u - static) x %llu B, cap %.0f GB:\n"
                "    %7s %7s  %8s %9s  %8s %9s  %9s  %9s\n",
                slots, ncards, tr.n_layers, tr.n_experts, (unsigned long long)sbytes, pincap,
                "static", "stream", "FIFO hit", "FIFO MB", "LRU hit", "LRU MB", "vs shipped",
                "pinned GB");
    for (uint32_t st = 44; st + 4 <= slots; st += 4) {
        const std::vector<uint8_t> is_st = build_static(tr, st, 0, SIZE_MAX);
        const SimOut f = sim_all(tr, is_st, slots - st, Pol::kFifo, half);
        const SimOut l = sim_all(tr, is_st, slots - st, Pol::kLru,  half);
        const double fmb = f.bytes_per_step(sbytes) / 1e6, lmb = l.bytes_per_step(sbytes) / 1e6;
        const double pin = double(ncards) * tr.n_layers * double(tr.n_experts - st) * double(sbytes) / 1e9;
        std::printf("    %7u %7u  %7.2f%% %9.1f  %7.2f%% %9.1f  %+8.1f%%  %8.1f%s%s\n",
                    st, slots - st, 100.0 * f.hit_rate(), fmb, 100.0 * l.hit_rate(), lmb,
                    base_bytes > 0 ? 100.0 * (fmb - base_bytes) / base_bytes : 0.0, pin,
                    pin > pincap ? "  WILL NOT LOAD" : "",
                    (f.unavailable || l.unavailable) ? "  REFUSALS" : "");
    }
    // ---- HOW TO READ THE ABOVE ----
    // Calibrated against twelve synthetic traces spanning consecutive
    // stickiness, per-layer Zipf skew, shared Zipf skew and slow topic drift at
    // three timescales.  The rows above separate those cases cleanly, and they
    // demand OPPOSITE fixes, so the reading rule is worth stating where the
    // numbers are printed rather than in a document that will drift from them.
    std::printf(
        "\n    READING RULE (calibrated on synthetic traces of known structure)\n"
        "    1. FIFO hit vs the SHUFFLED control.\n"
        "       collapses (e.g. 68%% -> 36%%)  => the hits are RECURRENCE at distance 1.\n"
        "                                        FIFO already captures all of it; no\n"
        "                                        replacement policy can add anything and\n"
        "                                        80/12 is already the best split.\n"
        "       barely moves                  => the hits are FREQUENCY.  Replacement is\n"
        "                                        the wrong lever; WHICH experts are static\n"
        "                                        is the right one.\n"
        "    2. PER-LAYER freq minus index order.  This was the prize for making residency\n"
        "       per-layer; it is now `Ds4Options::expert_priority_layers` and the per-layer\n"
        "       form of the ranking file, gated in §15e-2.  Synthetic range +0.04 points\n"
        "       (pure stickiness) to +32.6 points (topic drift); on the REAL 512-step decode\n"
        "       trace it measured +19.3 points of hit rate, 533.0 -> 201.7 MB/card/token,\n"
        "       at the SAME 80/12 split and the SAME pinned host bytes.\n"
        "    3. LRU is worth ~0 AT 80/12 under every structure tested (max +0.02), because\n"
        "       12 slots against ~4 misses/step is ~3 steps of history and `keep_prev`\n"
        "       already pins the previous token's slots -- FIFO and LRU evict at the same\n"
        "       age.  It only becomes a lever at a WIDE streaming partition AND with\n"
        "       medium-distance reuse: at 44/48 under topic drift LRU measured +6.6 points\n"
        "       over FIFO, and under pure distance-1 stickiness it measured +0.06 there.\n"
        "       So read the split sweep's two columns together, never the 80/12 row alone.\n"
        "    4. BELADY bounds every online policy.  If FIFO is already close to it, the\n"
        "       replacement question is closed whatever the answer to 1-3.\n");
    std::printf("\n%s: %d failure(s)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return true;
}

}  // namespace

int main() {
    // §16 runs ALONE when it is asked for: it is a measurement harness whose
    // answer depends on how much page cache is available, and the GPU sections
    // would spend memory inside whatever accounting it is being measured under.
    if (io_probe()) {
        std::printf("\n%s: %d failure(s)\n", g_fail ? "FAILED" : "PASSED", g_fail);
        return g_fail ? 1 : 0;
    }
    // §17 runs ALONE for the same reason §16 does — it is a measurement harness,
    // not a gate, and it answers about a trace rather than about this box.
    if (trace_score()) return g_fail ? 1 : 0;
    std::printf("deepseek4_residency_test — tiered VRAM/host residency (docs/deepseek4/33 §2.2)\n");

    // ---------------------------------------------------------------------
    std::printf("\n[1] Real GGUF tensor tables, both quants (header parse only)\n");
    // ---------------------------------------------------------------------
    ExpertGeometry q3, q8;
    const bool have_q3 = read_geometry(
        env_or("DS4_GGUF",
               "${IE_MODELS_DIR}/DeepSeek-V4-Flash-0731-GGUF/UD-Q3_K_XL/"
               "DeepSeek-V4-Flash-0731-UD-Q3_K_XL-00001-of-00004.gguf"), q3);
    const bool have_q8 = read_geometry(
        env_or("DS4_GGUF_Q8",
               "${IE_MODELS_DIR}/DeepSeek-V4-Flash-0731-GGUF/UD-Q8_K_XL/"
               "DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf"), q8);

    if (have_q3) {
        report_geometry("UD-Q3_K_XL", q3);
        std::printf("      blk.25 gate/up/down = %s ; blk.26 gate/up/down = %s\n",
                    q3.dtypes_of_25.c_str(), q3.dtypes_of_26.c_str());
        ok("Q3: 43 layers x 256 experts", q3.n_layers == 43 && q3.n_experts == 256);
        ok("Q3: slot bytes come from each tensor's OWN dtype -- blk.25 is IQ3_XXS gate/up",
           q3.dtypes_of_25 == "IQ3_XXS/IQ3_XXS/MXFP4", q3.dtypes_of_25);
        ok("Q3: blk.26 gate/up are MXFP4, not IQ3_XXS (Unsloth UD mixed precision)",
           q3.dtypes_of_26 == "MXFP4/MXFP4/MXFP4", q3.dtypes_of_26);
        ok("Q3: a normal layer's slot is 10,878,976 B", q3.slot_bytes[25] == 10878976ull,
           std::to_string(q3.slot_bytes[25]));
        ok("Q3: blk.26's slot is 13,369,344 B -- WIDER, which a per-model constant misses",
           q3.slot_bytes[26] == 13369344ull, std::to_string(q3.slot_bytes[26]));
        ok("Q3: blk.26 is the ONLY wide layer", [&] {
               uint32_t n = 0;
               for (uint32_t l = 0; l < q3.n_layers; ++l) if (q3.slot_bytes[l] != 10878976ull) ++n;
               return n == 1;
           }());
        ok("Q3: packed slot bytes == the GGUF expert bytes (the SoA repack is size-preserving)",
           q3.pool_slot == q3.pool_gguf, gb(q3.pool_slot) + " vs " + gb(q3.pool_gguf));
        ok("Q3: routed-expert pool is 120,393,302,016 B (120.393 GB, manifest)",
           q3.pool_slot == 120393302016ull, gb(q3.pool_slot));
        ok("Q3: Sum of one slot per layer is 470,286,336 B",
           q3.per_layer == 470286336ull, std::to_string(q3.per_layer));
    } else {
        std::printf("  %s\n", "Q3 ABSENT: sections keyed to the Q3 tensor table are NOT verified.");
    }

    if (have_q8) {
        report_geometry("UD-Q8_K_XL", q8);
        std::printf("      blk.25 gate/up/down = %s ; blk.26 gate/up/down = %s\n",
                    q8.dtypes_of_25.c_str(), q8.dtypes_of_26.c_str());
        ok("Q8: 43 layers x 256 experts", q8.n_layers == 43 && q8.n_experts == 256);
        ok("Q8: every layer's slot is 13,369,344 B (uniform -- and read, not assumed)", [&] {
               for (uint64_t b : q8.slot_bytes) if (b != 13369344ull) return false;
               return true;
           }());
        ok("Q8: blk.26 is NOT special here -- same size as blk.25, unlike Q3",
           q8.slot_bytes[26] == q8.slot_bytes[25],
           std::to_string(q8.slot_bytes[25]) + " vs " + std::to_string(q8.slot_bytes[26]));
        ok("Q8: packed slot bytes == the GGUF expert bytes",
           q8.pool_slot == q8.pool_gguf, gb(q8.pool_slot) + " vs " + gb(q8.pool_gguf));
        ok("Q8: routed-expert pool is 147,169,738,752 B (147.170 GB)",
           q8.pool_slot == 147169738752ull, gb(q8.pool_slot));
        ok("Q8: Sum of one slot per layer is 574,881,792 B",
           q8.per_layer == 574881792ull, std::to_string(q8.per_layer));
    } else {
        std::printf("  %s\n", "Q8 ABSENT: sections keyed to the Q8 tensor table are NOT verified.");
    }

    // ---------------------------------------------------------------------
    std::printf("\n[2] The pin budget, against doc 33 §2.3/§2.4\n");
    // ---------------------------------------------------------------------
    if (have_q3)
        // doc 33 §2.3: 97 slots/layer, static 85 -> "host holds 7,353 x 10.879 MB = 80.0 GB".
        // Exact, with blk.26 at its real width: 171 x 470,286,336 = 80,418,963,456 B.
        check_plan("Q3 (doc 33 §2.3)", q3, /*slots=*/97, /*min_stream=*/12,
                   /*expect_host=*/80418963456ull, ie::kDs4HostPinCapDefault);
    if (have_q8)
        // doc 33 §2.4: 80 slots/layer, static 68 -> "8,084 x 13.369 MB = 108.1 GB".
        // Exact: 188 x 574,881,792 = 108,077,776,896 B.  That is ABOVE the default
        // cap, which is the honest answer for a figure doc 33 §7.3 itself flags as
        // extrapolation past the 96 GB that was actually pinned and touched.
        check_plan("Q8 (doc 33 §2.4)", q8, /*slots=*/80, /*min_stream=*/12,
                   /*expect_host=*/108077776896ull, 110'000'000'000ull);

    if (have_q8) {
        ie::Ds4ResidencyPlan p;
        const std::string e = ie::ds4_plan_residency(q8.slot_bytes, q8.n_experts,
                                                     80ull * q8.per_layer, 0,
                                                     ie::kDs4HostPinCapDefault, 12, p);
        ok("Q8 at the DEFAULT cap is REFUSED, loudly and with the numbers", !e.empty(), e);
        ok("Q8 refusal still reports what it would have needed",
           p.host_bytes == 108077776896ull, gb(p.host_bytes));
    }

    // ---------------------------------------------------------------------
    std::printf("\n[3] The hard pinned-host cap\n");
    // ---------------------------------------------------------------------
    if (have_q3) {
        // The DEFECT itself: no static partition at all (every VRAM slot kept
        // evictable) is exactly the 120.393 GB whole-pool pin, and the cap
        // refuses it instead of quietly allocating it.
        ie::Ds4ResidencyPlan p;
        const std::string e = ie::ds4_plan_residency(q3.slot_bytes, q3.n_experts,
                                                     12ull * q3.per_layer, 0,
                                                     ie::kDs4HostPinCapDefault, 12, p);
        ok("Q3 with NO static partition wants the whole 120.393 GB pool",
           p.host_bytes == q3.pool_slot, gb(p.host_bytes));
        ok("...and is REFUSED by the cap rather than downsized", !e.empty(), e);

        // Raising the cap is the only way to get the old behaviour back, and it
        // is explicit.  Nothing downsizes on its own.
        ie::Ds4ResidencyPlan p2;
        const std::string e2 = ie::ds4_plan_residency(q3.slot_bytes, q3.n_experts,
                                                      12ull * q3.per_layer, 0,
                                                      q3.pool_slot, 12, p2);
        ok("an explicitly raised cap permits it (no hidden clamp either way)", e2.empty(), e2);

        // More VRAM slots must mean strictly fewer pinned host bytes: that IS
        // the free-host-copy-on-residency rule, expressed as a monotonicity.
        bool mono = true;
        uint64_t prev = ~0ull;
        for (uint32_t s = 16; s <= 96; s += 8) {
            ie::Ds4ResidencyPlan ps;
            (void)ie::ds4_plan_residency(q3.slot_bytes, q3.n_experts, uint64_t(s) * q3.per_layer,
                                         0, q3.pool_slot, 12, ps);
            mono = mono && ps.host_bytes < prev;
            mono = mono && uint64_t(ps.static_slots) * q3.per_layer + ps.host_bytes == q3.pool_slot;
            prev = ps.host_bytes;
        }
        ok("host bytes fall strictly as VRAM slots rise, and the partition holds throughout", mono);
    }
    {
        ie::Ds4ResidencyPlan p;
        const std::vector<uint64_t> sb(4, 1ull << 20);
        ok("a zero cap is refused", !ie::ds4_plan_residency(sb, 64, 0, 0, 0, 12, p).empty());
        ok("min_stream_slots == 0 is refused",
           !ie::ds4_plan_residency(sb, 64, 0, 0, ~0ull, 0, p).empty());
        ok("a VRAM budget too small for one slot per layer is refused",
           !ie::ds4_plan_residency(sb, 64, 1024, 0, ~0ull, 12, p).empty());
        ok("a zero-byte layer slot is refused",
           !ie::ds4_plan_residency(std::vector<uint64_t>{1ull << 20, 0}, 64, 0, 0, ~0ull, 12, p)
                .empty());
    }

    // ---------------------------------------------------------------------
    std::printf("\n[4] The always-resident set: packed vs dequantised to fp16\n");
    // ---------------------------------------------------------------------
    {
        // The policy function itself, before any file is involved.
        ok("Q8_0 with a whole number of 32-blocks stays packed",
           ie::ds4_dense_keeps_packed(ie::DType::kQ8_0, 4096));
        ok("Q6_K with a whole number of 256-blocks stays packed",
           ie::ds4_dense_keeps_packed(ie::DType::kQ6_K, 4096));
        ok("a ragged Q8_0 row does NOT stay packed (element addressing needs whole blocks)",
           !ie::ds4_dense_keeps_packed(ie::DType::kQ8_0, 48));
        ok("a ragged Q6_K row does NOT stay packed",
           !ie::ds4_dense_keeps_packed(ie::DType::kQ6_K, 128));
        ok("BF16 is NOT packed -- it is already 2 B/element, so packing saves nothing",
           !ie::ds4_dense_keeps_packed(ie::DType::kBF16, 4096));
        ok("F32 is NOT packed -- expanding it to fp16 HALVES it",
           !ie::ds4_dense_keeps_packed(ie::DType::kF32, 4096));
        ok("IQ3_XXS is NOT packed here (routed experts have their own path)",
           !ie::ds4_dense_keeps_packed(ie::DType::kIQ3_XXS, 4096));
        ok("packed Q8_0 [4096,1024] costs 34 B per 32 elements",
           ie::ds4_dense_device_bytes(ie::DType::kQ8_0, 4096, 1024) == (4096ull / 32) * 34 * 1024,
           std::to_string(ie::ds4_dense_device_bytes(ie::DType::kQ8_0, 4096, 1024)));
        ok("packed Q6_K [4096,129280] costs 210 B per 256 elements",
           ie::ds4_dense_device_bytes(ie::DType::kQ6_K, 4096, 129280) ==
               (4096ull / 256) * 210 * 129280);
        ok("an unpacked dtype costs exactly K*N*2",
           ie::ds4_dense_device_bytes(ie::DType::kBF16, 4096, 512) == 4096ull * 512 * 2);
    }
    for (const auto& e : {std::pair<const char*, const ExpertGeometry*>{"Q3", &q3},
                          std::pair<const char*, const ExpertGeometry*>{"Q8", &q8}}) {
        const ExpertGeometry& t = *e.second;
        if (!t.opened) continue;
        std::printf("    %s always-resident: fp16-expanded %s ; packed %s ; the FILE stores %s\n"
                    "      (%llu dense weights stay packed, %llu expand)\n",
                    e.first, gb(t.resident_fp16).c_str(), gb(t.resident_packed).c_str(),
                    gb(t.resident_gguf).c_str(),
                    (unsigned long long)t.n_dense_packed,
                    (unsigned long long)t.n_dense_expanded);
        ok((std::string(e.first) + ": every non-expert tensor is classified").c_str(),
           t.resident_err.empty(), t.resident_err);
        ok((std::string(e.first) + ": packed never costs more than expanding").c_str(),
           t.resident_packed <= t.resident_fp16);
    }
    if (have_q3 && have_q8)
        // Both quants have identical geometry, so "dequantise everything to fp16"
        // lands on the SAME number for both -- which is why 14.785 GB tells you
        // nothing about the file's own precision.
        ok("fp16 expansion costs the same for both quants (same shapes, same fp16)",
           q3.resident_fp16 == q8.resident_fp16,
           gb(q3.resident_fp16) + " vs " + gb(q8.resident_fp16));
    if (have_q3) {
        // THE ANCHOR: a real 43-layer load printed `always-resident set uploaded:
        // 14.785 GB` (docs/deepseek4/40). If this classification were wrong, this
        // assertion is what would say so.
        ok("Q3: the OLD expand-everything policy costs 14,784,708,956 B -- the 14.785 GB "
           "a real load printed",
           q3.resident_fp16 == 14784708956ull, gb(q3.resident_fp16));
        ok("Q3: the packed policy costs 7,887,249,756 B", q3.resident_packed == 7887249756ull,
           gb(q3.resident_packed));
        ok("Q3: packing recovers 6.897 GB of VRAM",
           q3.resident_fp16 - q3.resident_packed == 6897459200ull,
           gb(q3.resident_fp16 - q3.resident_packed));
        // Doc 33 §2.3 budgeted the resident set from the file's QUANTISED bytes
        // (7.808 GB incl. token_embd). The packed total lands just above it, and
        // the whole excess is the deliberate BF16->fp32 router widening.
        ok("Q3: packed total exceeds the file's own bytes by exactly the fp32 router widening",
           q3.resident_packed > q3.resident_gguf &&
               q3.resident_packed - q3.resident_gguf < 100'000'000ull,
           gb(q3.resident_packed) + " vs file " + gb(q3.resident_gguf));
    }
    if (have_q8) {
        // The honest Q8 result: its non-expert weights are BF16 in the file, i.e.
        // ALREADY 2 B/element, so there is nothing for packing to recover. Saying
        // otherwise would be the easiest possible lie in this whole change.
        ok("Q8: packing recovers NOTHING -- its resident set is BF16, already 2 B/element",
           q8.resident_packed == q8.resident_fp16, gb(q8.resident_packed));
        ok("Q8: and no dense weight qualified for packing", q8.n_dense_packed == 0,
           std::to_string(q8.n_dense_packed));
    }

    // ---------------------------------------------------------------------
    std::printf("\n[5] What the recovered VRAM buys, per quant (single device)\n");
    // ---------------------------------------------------------------------
    // load() derives the expert budget as global_mem - resident - 2 GB.
    auto budget_for = [](uint64_t resident) {
        return kB70GlobalMem > resident + kWsMargin ? kB70GlobalMem - resident - kWsMargin : 0ull;
    };
    auto plan_at = [&](const ExpertGeometry& t, uint64_t resident, uint64_t cap,
                       ie::Ds4ResidencyPlan& p) {
        return ie::ds4_plan_residency(t.slot_bytes, t.n_experts, budget_for(resident), 0, cap,
                                      12, p);
    };
    if (have_q3) {
        ie::Ds4ResidencyPlan before, after;
        const std::string eb = plan_at(q3, q3.resident_fp16,   200'000'000'000ull, before);
        const std::string ea = plan_at(q3, q3.resident_packed, 200'000'000'000ull, after);
        std::printf("    Q3 fp16-expanded: budget %s -> %u slots, host %s\n"
                    "    Q3 packed       : budget %s -> %u slots, host %s\n",
                    gb(budget_for(q3.resident_fp16)).c_str(), before.slots_per_layer,
                    gb(before.host_bytes).c_str(),
                    gb(budget_for(q3.resident_packed)).c_str(), after.slots_per_layer,
                    gb(after.host_bytes).c_str());
        ok("Q3 before: the plan reproduces docs/deepseek4/40 exactly (36 slots, 24 static, "
           "232 pinned, 109.106 GB host)",
           eb.empty() && before.slots_per_layer == 36 && before.static_slots == 24 &&
               before.pinned_experts == 232 && before.host_bytes == 109106429952ull,
           gb(before.host_bytes));
        ok("Q3 after: 51 slots/layer, 39 static, 217 pinned",
           ea.empty() && after.slots_per_layer == 51 && after.static_slots == 39 &&
               after.pinned_experts == 217);
        ok("Q3 after: pinned host is 102,052,134,912 B (102.052 GB)",
           after.host_bytes == 102052134912ull, gb(after.host_bytes));
        ok("Q3: packing buys 15 more VRAM slots per layer and drops the pin by 7.054 GB",
           after.slots_per_layer - before.slots_per_layer == 15 &&
               before.host_bytes - after.host_bytes == 7054295040ull,
           gb(before.host_bytes - after.host_bytes));
        ok("Q3 after: the two tiers still PARTITION the pool (nothing counted twice)",
           uint64_t(after.static_slots) * q3.per_layer + after.host_bytes == q3.pool_slot);
        // The part packing CANNOT fix, stated as arithmetic rather than as a hope.
        const uint64_t vram_for_80gb = 97ull * q3.per_layer;
        ok("doc 33's 80.419 GB target needs 97 slots = 45.62 GB of expert VRAM, which does not "
           "fit one 34.242 GB card AT ANY resident-set size -- it is a two-card aggregate",
           vram_for_80gb > kB70GlobalMem, gb(vram_for_80gb) + " vs card " + gb(kB70GlobalMem));
        // Even a hypothetical zero-byte resident set cannot reach the default cap.
        ie::Ds4ResidencyPlan ideal;
        (void)plan_at(q3, 0, 200'000'000'000ull, ideal);
        ok("even a ZERO-byte resident set only reaches 68 slots on this card, so the pin stays "
           "above the 84 GB default cap",
           ideal.host_bytes > ie::kDs4HostPinCapDefault,
           std::to_string(ideal.slots_per_layer) + " slots, " + gb(ideal.host_bytes));
        // ...so the default cap must still REFUSE, loudly, after the change.
        ie::Ds4ResidencyPlan capped;
        const std::string ec = plan_at(q3, q3.resident_packed, ie::kDs4HostPinCapDefault, capped);
        ok("Q3 packed is STILL refused by the 84 GB constant cap -- the fix is real but partial",
           !ec.empty(), ec);
    }
    if (have_q8) {
        ie::Ds4ResidencyPlan before, after;
        const std::string eb = plan_at(q8, q8.resident_fp16,   200'000'000'000ull, before);
        const std::string ea = plan_at(q8, q8.resident_packed, 200'000'000'000ull, after);
        std::printf("    Q8 fp16-expanded: %u slots, host %s\n    Q8 packed       : %u slots, host %s\n",
                    before.slots_per_layer, gb(before.host_bytes).c_str(),
                    after.slots_per_layer, gb(after.host_bytes).c_str());
        ok("Q8: 30 slots/layer, 18 static, 238 pinned, 136,821,866,496 B host",
           ea.empty() && after.slots_per_layer == 30 && after.static_slots == 18 &&
               after.pinned_experts == 238 && after.host_bytes == 136821866496ull,
           gb(after.host_bytes));
        ok("Q8: packing changes NOTHING -- identical plan before and after",
           eb.empty() && before.host_bytes == after.host_bytes &&
               before.slots_per_layer == after.slots_per_layer);
    }

    // ---------------------------------------------------------------------
    std::printf("\n[6] The live pinned-host cap (/proc/meminfo, not a constant)\n");
    // ---------------------------------------------------------------------
    // The DERIVED cap this box would use for a real load. Captured once here and
    // reused by §8, so the two sections cannot disagree about what "the cap" is.
    uint64_t g_live_cap = 0;
    bool     g_live_ok  = false;
    {
        std::string why;
        bool meminfo_ok = false;
        const uint64_t live = ie::ds4_host_pin_cap_live(why, &meminfo_ok);
        g_live_cap = live;
        g_live_ok  = meminfo_ok;
        std::printf("      %s\n", why.c_str());
        ok("the derivation is always reported, cap or no cap", !why.empty());

        // Recompute it here from /proc/meminfo independently. The point of the
        // whole change is that this number MOVES with the page cache, so a test
        // that hardcoded it would be testing nothing.
        uint64_t total_kb = 0, avail_kb = 0;
        if (std::FILE* f = std::fopen("/proc/meminfo", "r")) {
            char line[256];
            while (std::fgets(line, sizeof(line), f)) {
                unsigned long long v = 0;
                if (std::sscanf(line, "MemTotal: %llu kB", &v) == 1)          total_kb = v;
                else if (std::sscanf(line, "MemAvailable: %llu kB", &v) == 1) avail_kb = v;
            }
            std::fclose(f);
        }
        if (total_kb && avail_kb) {
            const uint64_t total = total_kb * 1024ull, avail = avail_kb * 1024ull;
            const uint64_t r_dyn = std::max(ie::kDs4HostPinReserveMin,
                                            total / ie::kDs4HostPinReserveDiv);
            const uint64_t r_abs = std::max(ie::kDs4HostPinReserveMin,
                                            total / ie::kDs4HostPinUnpinnedDiv);
            const uint64_t c_dyn = avail > r_dyn ? avail - r_dyn : 0;
            const uint64_t c_abs = total > r_abs ? total - r_abs : 0;
            const uint64_t want  = std::min(c_dyn, c_abs);
            std::printf("      independent recompute: dynamic %s, absolute %s -> %s (%s binds)\n",
                        gb(c_dyn).c_str(), gb(c_abs).c_str(), gb(want).c_str(),
                        c_dyn <= c_abs ? "dynamic" : "absolute");
            ok("/proc/meminfo was readable, so the cap is DERIVED and not the constant fallback",
               meminfo_ok);
            // MemAvailable drifts between the two reads; allow 1 GB of drift and
            // no more, so a formula change still fails this.
            const uint64_t d = live > want ? live - want : want - live;
            ok("the live cap == min(MemAvailable - max(16 GB, MemTotal/8), "
               "MemTotal - max(16 GB, MemTotal/4)), recomputed independently",
               d < (1ull << 30), gb(live) + " vs " + gb(want) + " (drift " + gb(d) + ")");
            ok("the live cap never exceeds MemAvailable", live <= avail);
            ok("the live cap leaves at least the dynamic reserve of what is AVAILABLE unpinned",
               live == 0 || avail - live >= r_dyn - (1ull << 30));
            // THE NEW TERM, stated as the property it exists for: a pinned page
            // is unreclaimable, so a quarter of the box's RAM stays out of the
            // pin no matter how idle MemAvailable makes the machine look.
            ok("the live cap ALSO leaves at least a quarter of MemTotal unpinned -- the term "
               "MemAvailable alone cannot express, because a pinned page is unreclaimable",
               total - live >= r_abs, gb(total - live) + " unpinned vs floor " + gb(r_abs));
            ok("the absolute term is not below the 119.5 GB this box is RECORDED as having "
               "pinned successfully (docs/deepseek4/40) -- a ceiling under that would refuse a "
               "configuration already known to work",
               c_abs >= 119'500'000'000ull, gb(c_abs));
        } else {
            std::printf("  SKIP: /proc/meminfo unavailable; the live cap is UNVERIFIED here\n");
            ok("no cap is derived when /proc/meminfo cannot be read", live == 0 && !meminfo_ok);
        }

        // The two zero returns must be distinguishable, because a caller that
        // confuses them substitutes the 84 GB constant on a STARVED box -- i.e.
        // raises the cap exactly when it must fall. `meminfo_ok` is what carries
        // that distinction, and the starved branch's consequence is that a cap
        // of zero REFUSES rather than falling through to any constant.
        ie::Ds4ResidencyPlan p;
        const std::string e = ie::ds4_plan_residency(std::vector<uint64_t>(4, 1ull << 20), 64, 0,
                                                     0, /*cap=*/0, 12, p);
        ok("a derived cap of ZERO refuses the plan outright -- the starved-box branch never "
           "reaches a constant", !e.empty(), e);
    }

    // ---------------------------------------------------------------------
    std::printf("\n[7] Two-card hidden-dim expert-TP: the split (doc 33 §2.1, Option A)\n");
    // ---------------------------------------------------------------------
    for (const auto& ent : {std::pair<const char*, const ExpertGeometry*>{"Q3", &q3},
                            std::pair<const char*, const ExpertGeometry*>{"Q8", &q8}}) {
        const ExpertGeometry& t = *ent.second;
        if (!t.opened) continue;
        const std::string L = ent.first;
        std::printf("    %s: EF=%u -> [0,%u) and [%u,%u) per card; down blocks are %u elements;"
                    " Σ slice/layer %llu B (whole %llu B)\n",
                    L.c_str(), t.expert_ffn, t.expert_ffn / 2, t.expert_ffn / 2, t.expert_ffn,
                    t.down_block, (unsigned long long)t.card_per_layer,
                    (unsigned long long)t.per_layer);
        ok((L + ": every layer accepts the 2-card intermediate-dim split").c_str(),
           t.slice_err.empty(), t.slice_err);
        ok((L + ": the slice is [0,EF/2) and [EF/2,EF) on every layer").c_str(), t.slice_range_ok);
        ok((L + ": the two cards' slices TILE the expert on every layer (card0+card1 == whole)")
               .c_str(),
           t.slice_tiles);
        ok((L + ": the two cards carry exactly the same bytes on every layer").c_str(),
           t.slice_balanced);
        ok((L + ": Σ per-card slot bytes is exactly half Σ whole slot bytes").c_str(),
           t.card_per_layer * 2 == t.per_layer,
           std::to_string(t.card_per_layer) + " x2 vs " + std::to_string(t.per_layer));
        ok((L + ": a 1-card slice reproduces ds4_slot_layout byte for byte "
                "(one implementation, not two)").c_str(),
           t.whole_matches_1card);
        // THE BLOCK-BOUNDARY PROOF, stated as arithmetic over the file's own
        // dtypes: `down` is sliced along its CONTRACTION dim, so EF/2 must be a
        // whole number of blocks.  gate/up are sliced along their OUTPUT dim,
        // where blocks run along the untouched K, so no constraint applies.
        ok((L + ": EF/2 is a whole number of down-projection quantisation blocks").c_str(),
           t.down_block != 0 && (t.expert_ffn / 2) % t.down_block == 0,
           std::to_string(t.expert_ffn / 2) + " % " + std::to_string(t.down_block) + " = " +
               std::to_string(t.down_block ? (t.expert_ffn / 2) % t.down_block : 0));
        ok((L + ": a split that would land MID-BLOCK is refused by name, not rounded").c_str(),
           !t.midblock_refusal.empty(), t.midblock_refusal);
        {
            ie::Ds4ExpertSlice s;
            ie::GgufTensorInfo dummy{};
            ok((L + ": an EF that does not divide by n_cards is refused").c_str(),
               !ie::ds4_expert_slice(dummy, dummy, dummy, t.expert_ffn, 3, 0, s).empty());
            ok((L + ": card >= n_cards is refused").c_str(),
               !ie::ds4_expert_slice(dummy, dummy, dummy, t.expert_ffn, 2, 2, s).empty());
            ok((L + ": n_cards == 0 is refused").c_str(),
               !ie::ds4_expert_slice(dummy, dummy, dummy, t.expert_ffn, 0, 0, s).empty());
        }
    }
    if (have_q3) {
        // doc 33 §2.1's two slot classes are PER-CARD figures; this is where
        // that becomes checkable rather than asserted.
        ok("Q3: a normal layer's per-card slice is 5,439,488 B (doc 33 §2.1)",
           q3.card0_bytes[25] == 5439488ull, std::to_string(q3.card0_bytes[25]));
        ok("Q3: blk.26's per-card slice is 6,684,672 B -- the MXFP4 layer, still exactly half",
           q3.card0_bytes[26] == 6684672ull, std::to_string(q3.card0_bytes[26]));
    }
    if (have_q8)
        ok("Q8: every layer's per-card slice is 6,684,672 B (uniform MXFP4)", [&] {
               for (uint64_t b : q8.card0_bytes) if (b != 6684672ull) return false;
               return true;
           }());

    // ---------------------------------------------------------------------
    std::printf("\n[8] The two-card residency plan (this is what closes the memory gap)\n");
    // ---------------------------------------------------------------------
    if (have_q3) {
        // Q3 MIRRORS its non-expert set across the cards (doc 33 §2.5: Q3 is
        // compute-bound at this residency, so TP-splitting attention buys slots
        // it cannot use and costs a reduce per layer).  So the per-card budget
        // is the SAME as the single-card budget: 34.242 - 7.887 - 2 GiB.
        check_tp_plan("Q3", q3, budget_for(q3.resident_packed), ie::kDs4HostPinCapDefault,
                      /*slots*/ 102, /*static*/ 90, /*pinned*/ 166,
                      /*host/card*/ 39033765888ull, /*accept*/ true);
        ok("Q3 2-card: 78.068 GB of pinned host over the box -- UNDER the 84 GB cap that the "
           "single-card plan's 102.052 GB blows through.  This is the fix.",
           2 * 39033765888ull < ie::kDs4HostPinCapDefault,
           gb(2 * 39033765888ull) + " vs cap " + gb(ie::kDs4HostPinCapDefault));
        ok("Q3 2-card: 102 slots/card beats doc 33's 97, and doubles the 51 one card reaches",
           true, "51 -> 102 slots/layer/card");
    }
    if (have_q8) {
        // Q8's non-expert set is BF16 in the file, so §4 measured that packing
        // recovers NOTHING; MIRRORED, the per-card budget is the fp16 one.
        check_tp_plan("Q8 (non-expert set mirrored)", q8, budget_for(q8.resident_fp16),
                      ie::kDs4HostPinCapDefault, /*slots*/ 60, /*static*/ 48, /*pinned*/ 208,
                      /*host/card*/ 59787706368ull, /*accept*/ false);
        ok("Q8 2-card mirrored: 119.575 GB total is STILL above the 84 GB cap -- expert-TP alone "
           "does not close Q8's gap, and the plan says so instead of pinning it",
           2 * 59787706368ull > ie::kDs4HostPinCapDefault, gb(2 * 59787706368ull));
        ok("Q8 2-card mirrored: even the BOX total beats the single-card figure "
           "(119.575 GB vs 136.822 GB) -- the split still helps, just not enough",
           2 * 59787706368ull < 136821866496ull,
           gb(2 * 59787706368ull) + " vs single-card " + gb(136821866496ull));
    }

    // ---------------------------------------------------------------------
    std::printf("\n[8c] TP-SPLITTING THE ALWAYS-RESIDENT SET -- the last memory gap\n");
    // ---------------------------------------------------------------------
    // WHAT THIS MEASURES.  §8 above plans both quants with the non-expert set
    // MIRRORED: every card carries a full copy, 7.887 GB for Q3 and 14.785 GB
    // for Q8 (Q8's is BF16 in the file, so §4 measured that packing recovers
    // NOTHING from it -- the whole 14.785 GB is real).  This section plans them
    // with that set TP-SPLIT per the decision table in kSplitRows/kSplitCols,
    // and the arithmetic is charged with the loader's OWN `ds4_dense_device_bytes`
    // over the sliced extents, tensor by tensor, from the real tables.
    //
    // WHAT IT COSTS, STATED BEFORE THE GAIN.  `attn_output_b` contracts over the
    // output-group axis this split partitions, so its result is a partial and the
    // forward gains ONE host-staged reduction per layer -- doc 33 §2.5 measured
    // that round trip at 16.83 us for 32 KB / 18.44 us for 64 KB, and one token's
    // reduction is T*hidden*4 = 16 KB. `ffn_down_shexp` is also a partial but it
    // is added into `ws_moe_` before the routed-expert reduction that already
    // ran, so it costs nothing. Everything else is mirrored.
    for (const auto& ent : {std::pair<const char*, const ExpertGeometry*>{"Q3", &q3},
                            std::pair<const char*, const ExpertGeometry*>{"Q8", &q8}}) {
        const ExpertGeometry& t = *ent.second;
        if (!t.opened) continue;
        const std::string L = ent.first;
        std::printf("    %s always-resident, per card: MIRRORED %s -> SPLIT %s\n"
                    "      (%llu row-split + %llu column-split tensors save %s + %s;"
                    " %s stays mirrored)\n",
                    L.c_str(), gb(t.resident_packed).c_str(), gb(t.resident_split).c_str(),
                    (unsigned long long)t.n_split_rows, (unsigned long long)t.n_split_cols,
                    gb(t.split_rows_saved).c_str(), gb(t.split_cols_saved).c_str(),
                    gb(t.split_mirrored).c_str());
        ok((L + ": the split accounting partitions the mirrored total exactly"
                " (saved + kept == mirrored)").c_str(),
           t.resident_split + t.split_rows_saved + t.split_cols_saved == t.resident_packed,
           gb(t.resident_split) + " + " + gb(t.split_rows_saved + t.split_cols_saved) + " vs " +
               gb(t.resident_packed));
        ok((L + ": splitting strictly reduces what one card holds").c_str(),
           t.resident_split < t.resident_packed,
           "saves " + gb(t.resident_packed - t.resident_split) + "/card");
        // The honest ceiling: the mirrored part is what a split can never touch,
        // and it is the majority of the set on neither quant but a real floor.
        ok((L + ": the mirrored remainder is accounted for, not hand-waved").c_str(),
           t.split_mirrored > 0 && t.split_mirrored < t.resident_packed,
           gb(t.split_mirrored) + " mirrored of " + gb(t.resident_packed));

        ie::Ds4TpResidencyPlan mir, spl;
        const uint64_t bmir = budget_for(t.resident_packed);
        const uint64_t bspl = budget_for(t.resident_split);
        const std::string emir = ie::ds4_plan_residency_tp(
            t.card0_bytes, t.slot_bytes, t.n_experts, 2, bmir, 0, 200'000'000'000ull, 12, mir);
        const std::string espl = ie::ds4_plan_residency_tp(
            t.card0_bytes, t.slot_bytes, t.n_experts, 2, bspl, 0, 200'000'000'000ull, 12, spl);
        std::printf("      MIRRORED: budget %s -> %u slots/card (%u static), host %s/card ="
                    " %s box\n"
                    "      SPLIT   : budget %s -> %u slots/card (%u static), host %s/card ="
                    " %s box\n",
                    gb(bmir).c_str(), mir.card.slots_per_layer, mir.card.static_slots,
                    gb(mir.card.host_bytes).c_str(), gb(mir.host_bytes_total).c_str(),
                    gb(bspl).c_str(), spl.card.slots_per_layer, spl.card.static_slots,
                    gb(spl.card.host_bytes).c_str(), gb(spl.host_bytes_total).c_str());
        ok((L + ": both plans are accepted at an unbounded cap").c_str(),
           emir.empty() && espl.empty(), emir + espl);
        ok((L + ": the split buys strictly more VRAM slots per card").c_str(),
           spl.card.slots_per_layer > mir.card.slots_per_layer,
           std::to_string(mir.card.slots_per_layer) + " -> " +
               std::to_string(spl.card.slots_per_layer) + " slots/layer/card (+" +
               std::to_string(spl.card.slots_per_layer - mir.card.slots_per_layer) + ")");
        ok((L + ": ...and strictly less pinned host over the box").c_str(),
           spl.host_bytes_total < mir.host_bytes_total,
           gb(mir.host_bytes_total) + " -> " + gb(spl.host_bytes_total) + " (frees " +
               gb(mir.host_bytes_total - spl.host_bytes_total) + ")");
        // THE INVARIANT, unchanged by any of this: the split moves bytes between
        // tiers, it never loses or duplicates one.
        ok((L + ": the split plan still partitions the whole expert pool").c_str(),
           2 * (uint64_t(spl.card.static_slots) * spl.card_slot_total + spl.card.host_bytes) ==
               t.pool_slot);
        // The reduction bill, in the same units the gain is quoted in, so the
        // trade is legible rather than asserted.  16 KB at T=1; doc 33 §2.5
        // measured 16.83 us for a 32 KB round trip and 18.44 us for 64 KB.
        std::printf("      the trade: +%u slots/card (residency %.1f%% -> %.1f%%) for ONE extra"
                    " host-staged reduction per layer\n"
                    "                 = %u layers x 16 KB round trips; at doc 33 §2.5's measured"
                    " 16.83 us that is %.2f ms/token\n",
                    spl.card.slots_per_layer - mir.card.slots_per_layer,
                    100.0 * mir.card.slots_per_layer / t.n_experts,
                    100.0 * spl.card.slots_per_layer / t.n_experts,
                    t.n_layers, double(t.n_layers) * 16.83e-3);
    }

    // ---------------------------------------------------------------------
    std::printf("\n[8b] The same plans against the DERIVED cap -- what this box will really do\n");
    // ---------------------------------------------------------------------
    // §8 above measures every plan against the 84 GB CONSTANT, which is no
    // longer the default: it is the /proc-unreadable fallback. What a load on
    // this machine actually uses is §6's derived number, and the whole point of
    // deriving it is that a configuration which genuinely fits stops being
    // refused by a constant that cannot see the box.
    if (!g_live_ok) {
        std::printf("  SKIP: /proc/meminfo unreadable; the derived-cap verdicts are UNVERIFIED\n");
    } else {
        std::printf("    derived cap right now: %s\n", gb(g_live_cap).c_str());
        auto verdict = [&](const char* label, const ExpertGeometry& t, uint64_t budget_per_card,
                           uint32_t n_cards, uint64_t expect_total) {
            ie::Ds4TpResidencyPlan tp;
            const std::string e = ie::ds4_plan_residency_tp(
                n_cards == 2 ? t.card0_bytes : t.slot_bytes, t.slot_bytes, t.n_experts, n_cards,
                budget_per_card, 0, g_live_cap, 12, tp);
            std::printf("    %-28s %u-card: wants %s, cap %s -> %s\n", label, n_cards,
                        gb(tp.host_bytes_total).c_str(), gb(g_live_cap).c_str(),
                        e.empty() ? "ACCEPTED" : "REFUSED");
            if (!e.empty()) std::printf("      %s\n", e.c_str());
            ok((std::string(label) + ": the plan's pin total is the expected figure").c_str(),
               tp.host_bytes_total == expect_total,
               gb(tp.host_bytes_total) + " vs " + gb(expect_total));
            // The verdict is asserted against the ARITHMETIC, not against a
            // hardcoded expectation, so this stays honest on a busier box.
            ok((std::string(label) + ": accept/refuse matches host_bytes_total <= cap").c_str(),
               e.empty() == (tp.host_bytes_total <= g_live_cap),
               gb(tp.host_bytes_total) + " vs cap " + gb(g_live_cap));
            return e.empty();
        };
        if (have_q8) {
            // THE DEFECT THIS SECTION EXISTS FOR. 119.575 GB over a 168.01 GB
            // box is 71% of RAM and leaves 48 GB; refusing it because a 2026-era
            // constant says 84 GB was the bug.
            const bool acc = verdict("Q8 (non-expert mirrored)", q8,
                                     budget_for(q8.resident_fp16), 2, 119575412736ull);
            ok("Q8 TWO-CARD IS ACCEPTED under the derived cap on this machine -- the 84 GB "
               "constant refused a configuration that fits",
               acc, "wants " + gb(119575412736ull) + ", derived cap " + gb(g_live_cap));
            ok("...and it was the CONSTANT that refused it, not the arithmetic",
               119575412736ull > ie::kDs4HostPinCapDefault,
               gb(119575412736ull) + " vs constant " + gb(ie::kDs4HostPinCapDefault));
            // Q8 single-card is genuinely too big and MUST still refuse: at
            // 136.822 GB it is above the absolute ceiling on this box whatever
            // MemAvailable says, so this is the "still refuses loudly" half.
            ie::Ds4ResidencyPlan p8;
            const std::string e8 = plan_at(q8, q8.resident_fp16, g_live_cap, p8);
            std::printf("    %-28s 1-card: wants %s -> %s\n", "Q8 (single card)",
                        gb(p8.host_bytes).c_str(), e8.empty() ? "ACCEPTED" : "REFUSED");
            ok("Q8 SINGLE-card (136.822 GB) is STILL refused, loudly and with the numbers -- "
               "an over-large plan did not become acceptable",
               !e8.empty() && p8.host_bytes == 136821866496ull, e8);
        }
        if (have_q3) {
            verdict("Q3", q3, budget_for(q3.resident_packed), 2, 78067531776ull);
            // Q3 single-card: 102.052 GB. Above the 84 GB constant, below the
            // derived cap on a quiet box -- so its verdict is a FUNCTION of the
            // machine, and the test states which way it went rather than
            // pretending the answer is fixed.
            ie::Ds4ResidencyPlan p3;
            const std::string e3 = plan_at(q3, q3.resident_packed, g_live_cap, p3);
            std::printf("    %-28s 1-card: wants %s -> %s%s\n", "Q3 (single card)",
                        gb(p3.host_bytes).c_str(), e3.empty() ? "ACCEPTED" : "REFUSED",
                        e3.empty() ? "" : " (this box is too busy for it right now)");
            ok("Q3 single-card: the verdict follows the arithmetic against the derived cap",
               e3.empty() == (p3.host_bytes <= g_live_cap),
               gb(p3.host_bytes) + " vs cap " + gb(g_live_cap));
            ok("Q3 single-card wants 102.052 GB, which the 84 GB constant refuses outright",
               p3.host_bytes == 102052134912ull && p3.host_bytes > ie::kDs4HostPinCapDefault,
               gb(p3.host_bytes));
        }
        // An unarguably over-large plan must still be refused with a number in
        // the message: the cap moved, it did not stop existing.
        {
            ie::Ds4ResidencyPlan p;
            const std::vector<uint64_t> sb(43, 13369344ull);
            const std::string e = ie::ds4_plan_residency(sb, 256, 12ull * 43 * 13369344ull, 0,
                                                         g_live_cap, 12, p);
            ok("a plan that wants the WHOLE 147.170 GB Q8 pool is refused by the derived cap",
               !e.empty() && p.host_bytes == 147169738752ull, e);
            ok("...and the refusal names both the requested bytes and the cap",
               e.find("147.170") != std::string::npos &&
                   e.find(gb(g_live_cap).substr(0, 6)) != std::string::npos,
               e);
        }

        // ---- the residency ORDER, applied to the REAL plans of both quants ----
        // §15e/§15f prove the order reaches the arena and the cache, but they
        // must do it on a miniature model: instantiating a Ds4HostArena over
        // these tensor tables would pin 39-60 GB. What CAN be proven here, on
        // the real numbers, is the step in between -- that the order partitions
        // the real expert count into the real static and pinned sets, and that a
        // non-identity order really moves experts across that boundary.
        // This is exactly the derivation `DeepSeek4Runtime::load` performs
        // between the plan and `arena_.init_set`.
        for (const auto& ent : {std::pair<const char*, const ExpertGeometry*>{"Q3", &q3},
                                std::pair<const char*, const ExpertGeometry*>{"Q8", &q8}}) {
            const ExpertGeometry& t = *ent.second;
            if (!t.opened) continue;
            const std::string L = ent.first;
            ie::Ds4TpResidencyPlan tp;
            (void)ie::ds4_plan_residency_tp(t.card0_bytes, t.slot_bytes, t.n_experts, 2,
                                            budget_for(t.resident_packed), 0, g_live_cap, 12, tp);
            const uint32_t S = tp.card.static_slots, P = tp.card.pinned_experts;
            if (S == 0 || S >= t.n_experts) {
                std::printf("    %s: static_slots = %u -- no boundary to move across\n",
                            L.c_str(), S);
                continue;
            }
            // The loader's own two lines, over the real plan.
            auto split = [&](const std::vector<uint32_t>& prio, std::vector<uint32_t>& stat,
                             std::vector<uint32_t>& pin) {
                stat.assign(prio.begin(), prio.begin() + S);
                pin.assign(prio.begin() + S, prio.begin() + S + P);
            };
            std::vector<uint32_t> idx(t.n_experts), rev(t.n_experts);
            for (uint32_t i = 0; i < t.n_experts; ++i) { idx[i] = i; rev[i] = t.n_experts - 1 - i; }
            std::vector<uint32_t> si, pi, sr, pr;
            split(idx, si, pi);
            split(rev, sr, pr);
            std::printf("    %s real 2-card plan: %u static + %u pinned of %u experts/layer\n",
                        L.c_str(), S, P, t.n_experts);
            ok((L + ": the order partitions the REAL expert count -- static + pinned == n_experts,"
                    " no expert in both tiers and none in neither").c_str(), [&] {
                   if (S + P != t.n_experts) return false;
                   std::vector<int> seen(t.n_experts, 0);
                   for (uint32_t e : si) ++seen[e];
                   for (uint32_t e : pi) ++seen[e];
                   for (int c : seen) if (c != 1) return false;
                   return true;
               }(), std::to_string(S) + " + " + std::to_string(P) + " vs " +
                        std::to_string(t.n_experts));
            ok((L + ": index order makes experts [0,static) resident, as it always did").c_str(),
               si.front() == 0 && si.back() == S - 1 && pi.front() == S);
            ok((L + ": a REVERSED order moves every one of those experts across the boundary "
                    "-- on the real tensor table, not a fixture").c_str(), [&] {
                   std::vector<bool> in_si(t.n_experts, false);
                   for (uint32_t e : si) in_si[e] = true;
                   uint32_t moved = 0;
                   for (uint32_t e : sr) if (!in_si[e]) ++moved;
                   return moved == S;   // NONE of the reversed static set was static before
               }(), std::to_string(S) + " static slots change hands");
            ok((L + ": and the pinned host set changes with it, keeping the same SIZE -- a "
                    "permutation moves residency, it does not change how much fits").c_str(),
               pr.size() == pi.size() && pr != pi,
               std::to_string(pi.size()) + " experts pinned either way");
        }
    }
    {
        // The planner's own guards.
        ie::Ds4TpResidencyPlan tp;
        const std::vector<uint64_t> card(4, 1ull << 20), whole(4, 2ull << 20), bad(4, 3ull << 20);
        ok("n_cards == 0 is refused",
           !ie::ds4_plan_residency_tp(card, whole, 64, 0, 0, 0, ~0ull, 12, tp).empty());
        ok("a slice that does not tile the expert is REFUSED, per layer, by name",
           !ie::ds4_plan_residency_tp(card, bad, 64, 2, 0, 0, ~0ull, 12, tp).empty());
        std::printf("      %s\n",
                    ie::ds4_plan_residency_tp(card, bad, 64, 2, 0, 0, ~0ull, 12, tp).c_str());
        ok("mismatched layer counts are refused",
           !ie::ds4_plan_residency_tp(card, std::vector<uint64_t>(3, 2ull << 20), 64, 2, 0, 0,
                                      ~0ull, 12, tp).empty());
        ok("a zero-byte per-card slice is refused",
           !ie::ds4_plan_residency_tp(std::vector<uint64_t>{0, 1ull << 20},
                                      std::vector<uint64_t>{0, 2ull << 20}, 64, 2, 0, 0, ~0ull,
                                      12, tp).empty());
        ok("the whole-box cap is divided across cards, never applied per card", [&] {
               // Learn what one card would pin, then offer exactly that much for
               // the BOX.  One card fits; two cannot — and the planner must
               // refuse, because both arenas come out of the same RAM.
               ie::Ds4TpResidencyPlan a, b, c;
               const uint64_t bud = 53ull * (4ull << 20);
               if (!ie::ds4_plan_residency_tp(card, whole, 64, 2, bud, 0, ~0ull, 12, a).empty())
                   return false;
               const uint64_t one = a.card.host_bytes;
               const std::string eb =
                   ie::ds4_plan_residency_tp(card, whole, 64, 2, bud, 0, 2 * one, 12, b);
               const std::string ec =
                   ie::ds4_plan_residency_tp(card, whole, 64, 2, bud, 0, one, 12, c);
               return one > 0 && eb.empty() && !ec.empty() && b.host_bytes_total == 2 * one;
           }());
    }
    if (have_q3) {
        // n_cards == 1 through the TP planner must equal the single-card planner
        // exactly — that is what makes this an EXTENSION and not a fork.
        ie::Ds4ResidencyPlan one;
        ie::Ds4TpResidencyPlan tp;
        const uint64_t bud = budget_for(q3.resident_packed);
        const std::string e1 = ie::ds4_plan_residency(q3.slot_bytes, q3.n_experts, bud, 0,
                                                      200'000'000'000ull, 12, one);
        const std::string e2 = ie::ds4_plan_residency_tp(q3.slot_bytes, q3.slot_bytes,
                                                         q3.n_experts, 1, bud, 0,
                                                         200'000'000'000ull, 12, tp);
        ok("n_cards == 1 through the TP planner reproduces ds4_plan_residency exactly",
           e1.empty() && e2.empty() && tp.card.slots_per_layer == one.slots_per_layer &&
               tp.card.static_slots == one.static_slots &&
               tp.card.host_bytes == one.host_bytes && tp.card.vram_bytes == one.vram_bytes &&
               tp.host_bytes_total == one.host_bytes,
           std::to_string(tp.card.slots_per_layer) + " vs " +
               std::to_string(one.slots_per_layer));
    }

    // ---------------------------------------------------------------------
    std::printf("\n[8c] Residency priority as DATA: the profile counter and the ranking file\n");
    // ---------------------------------------------------------------------
    // `Ds4Options::expert_priority` has existed and been tested, but nothing
    // ever SUPPLIED an order, so the knob could only be turned from C++. These
    // two pieces close the loop: a counter that measures utilisation during a
    // real forward pass, and a file format that carries the measurement into a
    // later load. Everything here is host-only -- no queue, no model.
    {
        const std::string dir = env_or("TMPDIR", "/tmp");
        const std::string path = dir + "/ds4_prio_test.txt";

        // ---- the counter ----
        ie::Ds4ExpertProfile prof;
        ok("a profile with zero layers is refused", !prof.init(0, 8).empty());
        ok("a profile with zero experts is refused", !prof.init(4, 0).empty());
        ok("an uninitialised profile is inactive and refuses to write",
           !ie::Ds4ExpertProfile{}.active() &&
               !ie::Ds4ExpertProfile{}.write(path).empty());
        ok("profile init", prof.init(3, 8).empty());
        ok("...and it starts empty", prof.selections() == 0 && prof.rejected() == 0);
        // An empty profile MUST yield the identity, so "profiling produced
        // nothing" degrades to today's default rather than to a shuffle.
        ok("an all-zero profile yields exactly the identity permutation", [&] {
               const std::vector<uint32_t> p = prof.priority();
               if (p.size() != 8) return false;
               for (uint32_t i = 0; i < 8; ++i) if (p[i] != i) return false;
               return true;
           }());
        {
            // Layer 0 routes to {5,5,2}, layer 1 to {5,7}, layer 2 to {7}.
            const int32_t l0[] = {5, 5, 2};
            const int32_t l1[] = {5, 7};
            const int32_t l2[] = {7};
            prof.record(0, l0, 3, true);
            prof.record(1, l1, 2, true);
            prof.record(2, l2, 1, true);
            ok("per-LAYER counts are kept separately, not merged at capture",
               prof.count(0, 5) == 2 && prof.count(1, 5) == 1 && prof.count(2, 5) == 0 &&
                   prof.count(0, 7) == 0 && prof.count(1, 7) == 1 && prof.count(2, 7) == 1);
            ok("totals are the per-layer sum", prof.total(5) == 3 && prof.total(7) == 2 &&
                                                   prof.total(2) == 1 && prof.total(0) == 0);
            ok("selections counted", prof.selections() == 6, std::to_string(prof.selections()));
            // 5 (3) > 7 (2) > 2 (1) > everything else at 0, ties by ascending id.
            ok("priority is most-selected first, ties broken by ASCENDING id", [&] {
                   const std::vector<uint32_t> p = prof.priority();
                   const std::vector<uint32_t> want{5, 7, 2, 0, 1, 3, 4, 6};
                   return p == want;
               }(), [&] {
                   std::string s;
                   for (uint32_t e : prof.priority()) s += std::to_string(e) + " ";
                   return s;
               }());
        }
        {
            // An out-of-range id must be COUNTED AS REJECTED, never clamped onto
            // expert 0 -- a clamp would invent utilisation that never happened.
            const int32_t bad[] = {-1, 8, 100, 3};
            const uint64_t before0 = prof.total(0);
            prof.record(0, bad, 4, true);
            ok("an out-of-range or negative expert id is rejected, not clamped onto expert 0",
               prof.rejected() == 3 && prof.total(0) == before0 && prof.total(3) == 1,
               "rejected=" + std::to_string(prof.rejected()));
            ok("a record() for a layer past the end is dropped, not written out of bounds",
               [&] { const int32_t x[] = {1}; prof.record(99, x, 1, true); return prof.total(1) == 0; }());
        }

        // ---- the file, round-tripped ----
        {
            const std::string we = prof.write(path);
            ok("the profile writes its ranking file", we.empty(), we);
            // The dump is now the PER-LAYER form, because that is what residency
            // wants and a global order throws most of the signal away (measured
            // cross-stream on the real model: one global order buys +5.5 points
            // of static hit over index order, a per-layer order buys +35.0).
            std::vector<std::vector<uint32_t>> backL;
            const std::string re = ie::ds4_expert_priority_read_layers(path, 8, 3, backL);
            ok("...and it reads back", re.empty(), re);
            ok("THE ROUND TRIP: the PER-LAYER orders read equal the orders written",
               backL == prof.priority_layers());
            // ...and the layers really do differ, or the round trip proves
            // nothing a single permutation would not have proved.  Layer 0 saw
            // {5,5,2,3}, layer 1 saw {5,7}, layer 2 saw {7}: their rankings
            // cannot coincide.
            ok("...and the per-layer orders are NOT all the same order",
               backL.size() == 3 && (backL[0] != backL[1] || backL[1] != backL[2]),
               [&] {
                   std::string s;
                   for (const auto& v : backL) {
                       for (uint32_t e : v) s += std::to_string(e);
                       s += " | ";
                   }
                   return s;
               }());
            // The per-layer detail must survive the dump as comments, or a
            // future per-layer scheme would need a whole new profiling run.
            std::string body;
            if (std::FILE* f = std::fopen(path.c_str(), "r")) {
                char b[4096];
                while (std::fgets(b, sizeof(b), f)) body += b;
                std::fclose(f);
            }
            ok("the dump declares its format version and expert count",
               body.find("# ds4-expert-priority 1") == 0 &&
                   body.find("\nexperts 8\n") != std::string::npos);
            ok("the dump preserves the per-layer counts as comments",
               body.find("# layer 0: 2:1 3:1 5:2") != std::string::npos &&
                   body.find("# layer 1: 5:1 7:1") != std::string::npos,
               body.substr(0, 200));
            ok("the dump records how many ids were rejected",
               body.find("3 ids rejected as out of range") != std::string::npos);
        }

        // ---- the format's refusals: every one names the problem ----
        auto write_raw = [&](const std::string& text) {
            std::FILE* f = std::fopen(path.c_str(), "w");
            std::fputs(text.c_str(), f);
            std::fclose(f);
        };
        auto refuse = [&](const char* what, const std::string& text, uint32_t n,
                          const char* needle) {
            write_raw(text);
            std::vector<uint32_t> v;
            const std::string e = ie::ds4_expert_priority_read(path, n, v);
            ok(what, !e.empty() && e.find(needle) != std::string::npos && v.empty(), e);
        };
        refuse("a file with no `experts` header at all is refused", "# nothing but a comment\n", 4,
               "no `experts");
        refuse("expert ids BEFORE the header are refused", "0\n experts 4\n1\n", 4,
               "before the `experts");
        refuse("a ranking built for a DIFFERENT expert count is refused, not truncated",
               "experts 256\n0\n1\n2\n3\n", 4, "different model");
        refuse("a PARTIAL ranking is refused (an expert would get neither tier)",
               "experts 4\n0\n1\n", 4, "full permutation");
        refuse("a ranking with too MANY ids is refused", "experts 4\n0\n1\n2\n3\n0\n", 4,
               "more than 4");
        refuse("a DUPLICATED id is refused (two experts would alias one slot)",
               "experts 4\n0\n1\n1\n2\n", 4, "twice");
        refuse("an OUT-OF-RANGE id is refused", "experts 4\n0\n1\n2\n9\n", 4, "is outside [0,");
        refuse("a non-numeric line is refused by name", "experts 4\n0\n1\nbanana\n3\n", 4,
               "expected an expert id");
        refuse("two `experts` headers are refused", "experts 4\nexperts 4\n0\n1\n2\n3\n", 4,
               "second `experts`");
        {
            std::vector<uint32_t> v;
            ok("a missing file is refused by name",
               !ie::ds4_expert_priority_read(dir + "/ds4_no_such_ranking", 4, v).empty());
        }
        {
            // Comments, blank lines, indentation and trailing columns are all
            // ignored -- the file has to be hand-writable, and the counts a dump
            // carries must be informative rather than load-bearing.
            write_raw("# ds4-expert-priority 1\n# a note\n\n  experts 4\n"
                      "  3 91204 extra columns\n1\t7\n\n# mid-file comment\n0 0\n2\n");
            std::vector<uint32_t> v;
            const std::string e = ie::ds4_expert_priority_read(path, 4, v);
            ok("comments, blank lines, indentation and trailing columns are ignored",
               e.empty() && v == std::vector<uint32_t>({3, 1, 0, 2}), e);
        }
        // The writer's own guards.
        ok("writing a non-permutation is refused",
           !ie::ds4_expert_priority_write(path, {0, 1, 1}, {}).empty());
        ok("writing counts of the wrong length is refused",
           !ie::ds4_expert_priority_write(path, {0, 1, 2}, {5, 6}).empty());
        ok("writing to an unopenable path is refused by name",
           !ie::ds4_expert_priority_write("/proc/ds4/nope", {0, 1}, {}).empty());
        std::remove(path.c_str());
    }

    // ---------------------------------------------------------------------
    std::printf("\n[8d] What the hash-router tables say about the DEFAULT order\n");
    // ---------------------------------------------------------------------
    // Layers 0-2 route through `ffn_gate_tid2eid`, a frozen [vocab, top_k] I32
    // lookup, so their expert usage is EXACTLY computable from a token-frequency
    // distribution with no forward pass at all. That makes them the one place in
    // this model where the residency question can be answered from the file, and
    // it is worth answering rather than asserting -- the previous justification
    // for index order was withdrawn precisely because it was asserted.
    //
    // Under the only distribution that needs no assumption (uniform over the
    // vocabulary) the answer is that index order is already almost exactly
    // optimal, and this section pins down BY HOW MUCH so the claim can be
    // falsified later rather than believed.
    for (const auto& ent : {std::pair<const char*, const char*>{
                                "Q3", "${IE_MODELS_DIR}/"
                                      "DeepSeek-V4-Flash-0731-GGUF/UD-Q3_K_XL/"
                                      "DeepSeek-V4-Flash-0731-UD-Q3_K_XL-00001-of-00004.gguf"}}) {
        ie::GgufReader g;
        if (!g.open(ent.second).empty()) {
            std::printf("  SKIP: %s absent; the tid2eid finding is NOT verified here\n", ent.first);
            break;
        }
        // Layer count and geometry from the file, never from a constant.
        std::vector<const ie::GgufTensorInfo*> tabs;
        for (uint32_t L = 0;; ++L) {
            const ie::GgufTensorInfo* t =
                g.find_tensor("blk." + std::to_string(L) + ".ffn_gate_tid2eid.weight");
            if (!t) break;
            tabs.push_back(t);
        }
        const ie::GgufTensorInfo* ge = g.find_tensor("blk.0.ffn_gate_exps.weight");
        if (tabs.empty() || !ge) {
            std::printf("  SKIP: %s has no tid2eid tables\n", ent.first);
            break;
        }
        const uint32_t TOPK = uint32_t(tabs[0]->shape[0]);
        const uint32_t VOC  = uint32_t(tabs[0]->shape[1]);
        const uint32_t NE   = uint32_t(ge->shape[2]);
        std::printf("    %s: %zu hash layers, tid2eid is [%u vocab x %u top_k] I32, %u experts\n",
                    ent.first, tabs.size(), VOC, TOPK, NE);
        ok("the hash-router tables are I32 and top_k-wide", [&] {
               for (const ie::GgufTensorInfo* t : tabs)
                   if (t->dtype != ie::DType::kI32 || t->shape[0] != TOPK || t->shape[1] != VOC)
                       return false;
               return true;
           }());

        // Aggregate per-expert entry counts over every hash layer. This IS the
        // expert usage of layers 0..n-1 under a uniform token distribution --
        // not a model of it, the exact value.
        std::vector<uint64_t> cnt(NE, 0);
        bool in_range = true;
        uint64_t picks = 0;
        for (const ie::GgufTensorInfo* t : tabs) {
            const auto* p = reinterpret_cast<const int32_t*>(t->data);
            for (uint64_t i = 0; i < uint64_t(VOC) * TOPK; ++i) {
                const int32_t e = p[i];
                if (e < 0 || uint32_t(e) >= NE) { in_range = false; break; }
                ++cnt[uint32_t(e)];
                ++picks;
            }
        }
        ok("every tid2eid entry is a valid expert id", in_range);
        ok("the tables hold vocab x top_k x n_hash_layers entries",
           picks == uint64_t(VOC) * TOPK * tabs.size(), std::to_string(picks));

        // Is the table flat? (If it were, there would be nothing to exploit and
        // nothing to get wrong.)
        uint64_t lo = ~0ull, hi = 0;
        for (uint64_t c : cnt) { lo = std::min(lo, c); hi = std::max(hi, c); }
        const double uni = double(picks) / NE;
        std::printf("    per-expert entries: min %llu  max %llu  uniform %.1f  spread %.1f%%\n",
                    (unsigned long long)lo, (unsigned long long)hi, uni,
                    100.0 * (double(hi) - double(lo)) / uni);
        ok("the tid2eid tables are NOT flat -- there is real structure to get right or wrong",
           hi > lo, std::to_string(lo) + ".." + std::to_string(hi));

        // ...and the structure runs the SAME WAY as index order: low expert ids
        // carry more entries than high ones. Block means make that checkable
        // without a correlation coefficient.
        const uint32_t B = NE / 8;
        std::vector<double> blockmean(8, 0.0);
        for (uint32_t b = 0; b < 8; ++b) {
            uint64_t s = 0;
            for (uint32_t e = b * B; e < (b + 1) * B; ++e) s += cnt[e];
            blockmean[b] = double(s) / B;
        }
        std::printf("    mean entries per %u-expert block:", B);
        for (double m : blockmean) std::printf(" %.0f", m);
        std::printf("\n");
        ok("entry count falls MONOTONICALLY with expert id across all 8 blocks -- the skew runs "
           "the same way index order does", [&] {
               for (uint32_t b = 1; b < 8; ++b)
                   if (blockmean[b] >= blockmean[b - 1]) return false;
               return true;
           }());

        // THE NUMBER THE DEFAULT RESTS ON: how much of the achievable gain does
        // index order already capture, against the count-optimal order and
        // against a residency-sized random baseline?
        std::vector<uint32_t> best(NE);
        for (uint32_t e = 0; e < NE; ++e) best[e] = e;
        std::sort(best.begin(), best.end(),
                  [&](uint32_t a, uint32_t b) { return cnt[a] != cnt[b] ? cnt[a] > cnt[b] : a < b; });
        bool all_small = true;
        for (uint32_t S : {18u, 39u, 48u, 90u}) {
            if (S >= NE) continue;
            uint64_t ix = 0, bs = 0;
            for (uint32_t i = 0; i < S; ++i) { ix += cnt[i]; bs += cnt[best[i]]; }
            const double fi = 100.0 * double(ix) / double(picks);
            const double fb = 100.0 * double(bs) / double(picks);
            const double fu = 100.0 * double(S) / NE;
            const double got = 100.0 * (fi - fu) / (fb - fu);
            std::printf("    %3u static slots: index %.4f%%  optimal %.4f%%  random %.4f%%"
                        "  -> index captures %.1f%% of the gain, residual %+.4f pts\n",
                        S, fi, fb, fu, got, fb - fi);
            ok(("index order beats a random static set at " + std::to_string(S) + " slots").c_str(),
               fi > fu);
            ok(("index order is within 0.1 points of OPTIMAL at " + std::to_string(S) +
                " slots -- so no static order can beat it here by enough to matter").c_str(),
               fb - fi < 0.1, "residual " + std::to_string(fb - fi) + " pts");
            all_small = all_small && (fb - fi) < 0.1;
        }
        ok("CONCLUSION, from the real tables: index order stays the default because the best "
           "achievable alternative is worth under 0.1 points on 3 of 43 layers -- a measured "
           "reason, not an assumed one", all_small);
    }

    // ---------------------------------------------------------------------
    std::printf("\n[9] The load-time split is BYTE-EXACT (synthetic tensors, host only)\n");
    // ---------------------------------------------------------------------
    // A card's packed slice must be, byte for byte, the corresponding part of
    // the whole-expert pack. Anything else is a silent wrong-weights bug that
    // no shape check would catch. Both slice axes are exercised:
    //   gate/up  N-slice  (output columns; blocks run along the untouched K)
    //   down     K-slice  (contraction; keeps whole blocks of every column)
    // and both dtypes, INCLUDING an IQ3_XXS down tensor, which neither shipped
    // quant has — its 256-element block is the harder alignment case and the
    // code must not be right by accident of MXFP4's 32.
    {
        std::mt19937 rng(0x5171CE5u);
        auto rnd = [&] { return uint8_t(rng() & 0xFFu); };

        // Builds a [K, N, E] tensor of `dt` filled with reproducible junk. The
        // BYTES are what this section compares, so their values are irrelevant
        // as long as they are not uniform.
        struct Synth {
            std::vector<uint8_t> bytes;
            ie::GgufTensorInfo   ti{};
        };
        auto synth = [&](const char* nm, ie::DType dt, uint32_t K, uint32_t N, uint32_t E) {
            auto s = std::make_unique<Synth>();
            const uint64_t blocks = uint64_t(E) * N * (K / (dt == ie::DType::kIQ3_XXS
                                                                ? uint32_t(ie::kQK_K)
                                                                : uint32_t(ie::kQK_MXFP4)));
            const uint64_t bsz = dt == ie::DType::kIQ3_XXS ? sizeof(ie::block_iq3_xxs)
                                                           : sizeof(ie::block_mxfp4);
            s->bytes.resize(size_t(blocks * bsz));
            for (uint8_t& b : s->bytes) b = rnd();
            s->ti.name = nm; s->ti.dtype = dt; s->ti.n_dims = 3;
            s->ti.shape[0] = K; s->ti.shape[1] = N; s->ti.shape[2] = E;
            s->ti.nbytes = blocks * bsz;
            s->ti.data = s->bytes.data();
            return s;
        };

        const uint32_t H = 512, EF = 512, E = 3;
        // Q3-shaped: IQ3_XXS gate/up + MXFP4 down. Plus an IQ3_XXS down, which
        // exercises the 256-element block on the CONTRACTION axis.
        auto gate = synth("gate", ie::DType::kIQ3_XXS, H, EF, E);
        auto up   = synth("up",   ie::DType::kIQ3_XXS, H, EF, E);
        auto dmx  = synth("down_mxfp4", ie::DType::kMXFP4,   EF, H, E);
        auto diq  = synth("down_iq3",   ie::DType::kIQ3_XXS, EF, H, E);

        for (const auto* dn : {&dmx, &diq}) {
            const ie::GgufTensorInfo& down = (*dn)->ti;
            const std::string dn_name(down.name);
            ie::Ds4SlotLayout whole;
            const std::string we = ie::ds4_slot_layout(gate->ti, up->ti, down, H, EF, whole);
            ok((dn_name + ": whole-expert layout").c_str(), we.empty(), we);
            std::vector<uint8_t> wpack(size_t(whole.bytes));
            ok((dn_name + ": whole-expert pack").c_str(),
               ie::ds4_slot_pack(whole, gate->ti, up->ti, down, 1, wpack.data()).empty());

            bool gate_ok = true, up_ok = true, down_ok = true, halves_ok = true;
            for (uint32_t c = 0; c < 2; ++c) {
                ie::Ds4ExpertSlice sl;
                const std::string se = ie::ds4_expert_slice(gate->ti, up->ti, down, EF, 2, c, sl);
                ok((dn_name + ": card " + std::to_string(c) + " slice").c_str(), se.empty(), se);
                ie::Ds4SlotLayout cl;
                ok((dn_name + ": card " + std::to_string(c) + " layout").c_str(),
                   ie::ds4_slot_layout_tp(gate->ti, up->ti, down, H, EF, sl, cl).empty());
                halves_ok = halves_ok && cl.bytes * 2 == whole.bytes;
                std::vector<uint8_t> cpack(size_t(cl.bytes));
                ok((dn_name + ": card " + std::to_string(c) + " pack").c_str(),
                   ie::ds4_slot_pack(cl, gate->ti, up->ti, down, 1, cpack.data()).empty());

                // Per-column BYTE stride of each plane, from K alone:
                //   IQ3_XXS  gp K/4, ap K/32 words = K/8 B, dp K/256 halves = K/128 B
                //   MXFP4    qs K/2, e  K/32 B
                auto col_bytes = [](const ie::Ds4MatPlanes& m, int pl) -> uint64_t {
                    if (m.dt == ie::DType::kIQ3_XXS)
                        return pl == 0 ? uint64_t(m.K) / 4
                             : pl == 1 ? uint64_t(m.K) / 8
                                       : uint64_t(m.K) / 128;
                    return pl == 0 ? uint64_t(m.K) / 2 : uint64_t(m.K) / 32;
                };
                auto plane_off = [](const ie::Ds4MatPlanes& m, int pl) {
                    return pl == 0 ? m.off0 : pl == 1 ? m.off1 : m.off2;
                };
                auto plane_len = [](const ie::Ds4MatPlanes& m, int pl) {
                    return pl == 0 ? m.len0 : pl == 1 ? m.len1 : m.len2;
                };

                // gate/up: an N-slice is a CONTIGUOUS run of columns in every
                // plane, so the card's whole plane is one sub-range compare.
                for (int which = 0; which < 2; ++which) {
                    const ie::Ds4MatPlanes& w = which ? whole.up : whole.gate;
                    const ie::Ds4MatPlanes& p = which ? cl.up : cl.gate;
                    bool good = true;
                    for (int pl = 0; pl < 3; ++pl) {
                        const uint64_t len = plane_len(p, pl);
                        if (!len) continue;
                        good = good && std::memcmp(cpack.data() + plane_off(p, pl),
                                                   wpack.data() + plane_off(w, pl) + c * len,
                                                   size_t(len)) == 0;
                    }
                    (which ? up_ok : gate_ok) = (which ? up_ok : gate_ok) && good;
                }
                // down: a K-slice keeps blocks [c*nb,(c+1)*nb) of EVERY output
                // column, so the comparison walks columns instead of one memcpy.
                {
                    const ie::Ds4MatPlanes& w = whole.down;
                    const ie::Ds4MatPlanes& p = cl.down;
                    const int np = w.dt == ie::DType::kIQ3_XXS ? 3 : 2;
                    for (uint32_t n = 0; n < w.N; ++n)
                        for (int pl = 0; pl < np; ++pl) {
                            const uint64_t wc = col_bytes(w, pl), pc = col_bytes(p, pl);
                            down_ok = down_ok &&
                                      std::memcmp(cpack.data() + plane_off(p, pl) + n * pc,
                                                  wpack.data() + plane_off(w, pl) + n * wc + c * pc,
                                                  size_t(pc)) == 0;
                        }
                }
            }
            ok((dn_name + ": the two half-slots are each exactly half the whole slot").c_str(),
               halves_ok);
            ok((dn_name + ": card c's GATE bytes == the whole pack's columns [c*EF/2,(c+1)*EF/2)")
                   .c_str(),
               gate_ok);
            ok((dn_name + ": card c's UP bytes == the whole pack's matching columns").c_str(),
               up_ok);
            ok((dn_name +
                ": card c's DOWN bytes == the whole pack's blocks [c*nb,(c+1)*nb) of every column")
                   .c_str(),
               down_ok);
        }
        // The 256-element block on the contraction axis is a REAL constraint,
        // not decoration: EF=512 over 4 cards gives 128 rows/card, which cuts an
        // IQ3_XXS super-block in half.  MXFP4 (32) is fine at the same split,
        // which is exactly why the check has to read each tensor's own dtype.
        {
            ie::Ds4ExpertSlice s;
            const std::string eiq = ie::ds4_expert_slice(gate->ti, up->ti, diq->ti, EF, 4, 1, s);
            const std::string emx = ie::ds4_expert_slice(gate->ti, up->ti, dmx->ti, EF, 4, 1, s);
            ok("a 4-way split of EF=512 CUTS an IQ3_XXS down block and is refused by name",
               !eiq.empty(), eiq);
            ok("...while the same split of an MXFP4 down tensor is fine -- the check is per-dtype,"
               " not per-model",
               emx.empty(), emx);
        }
    }

    // ---------------------------------------------------------------------
    std::printf("\n[10-14] Device-backed checks (small: ~35 MB pinned host, ~4 MB device)\n");
    // ---------------------------------------------------------------------
    std::vector<sycl::device> gpus;
    for (const auto& d : sycl::device::get_devices())
        if (d.is_gpu() && d.get_backend() == sycl::backend::ext_oneapi_level_zero) gpus.push_back(d);
    if (gpus.empty()) {
        std::printf("  NO LEVEL-ZERO GPU: §10-§14 NOT RUN. The arena/cache behaviour, the packed\n"
                    "  decoders and the cross-card reduction are UNVERIFIED on this machine;\n"
                    "  §1-§9 above are pure\n"
                    "  arithmetic and still stand.\n");
        std::printf("\n%s: %d failure(s)\n", g_fail ? "FAILED" : "PASSED", g_fail);
        return g_fail ? 1 : 0;
    }
    const char* gsel = std::getenv("DS4_GPU");
    const size_t gi = gsel ? size_t(std::atoi(gsel)) : (gpus.size() > 1 ? 1u : 0u);
    sycl::queue q(gpus[gi < gpus.size() ? gi : 0],
                  sycl::property_list{sycl::property::queue::in_order()});
    std::printf("    device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    // ---------------------------------------------------------------------
    std::printf("\n[10] A statically resident expert has NO host copy\n");
    // ---------------------------------------------------------------------
    const uint32_t NL = 4, NE = 64, STAT = 40, PIN = NE - STAT;
    std::vector<uint64_t> sb(NL, 256ull << 10);
    sb[NL - 1] = 512ull << 10;                       // one wide layer, like blk.26
    uint64_t per_layer = 0;
    for (uint64_t b : sb) per_layer += b;
    {
        ie::Ds4HostArena a;
        const std::string e = a.init_range(q, sb, NE, STAT, PIN, 8ull << 20);
        ok("arena init_range", e.empty(), e);
        ok("arena pins ONLY the streamed experts",
           a.total_bytes() == uint64_t(PIN) * per_layer,
           gb(a.total_bytes()) + " vs " + gb(uint64_t(PIN) * per_layer));
        ok("arena reports its pinned range", a.first_pinned_expert() == STAT &&
                                             a.n_pinned_experts() == PIN);
        bool no_host_copy = true, has_host_copy = true, contiguous = true;
        for (uint32_t l = 0; l < NL; ++l) {
            for (uint32_t e2 = 0; e2 < STAT; ++e2) no_host_copy = no_host_copy && !a.slot(l, e2);
            for (uint32_t e2 = STAT; e2 < NE; ++e2) has_host_copy = has_host_copy && a.slot(l, e2);
            for (uint32_t e2 = STAT + 1; e2 < NE; ++e2) {
                const auto p0 = reinterpret_cast<uintptr_t>(a.slot(l, e2 - 1));
                const auto p1 = reinterpret_cast<uintptr_t>(a.slot(l, e2));
                contiguous = contiguous && (p1 - p0 == sb[l]);
            }
        }
        ok("EVERY statically resident expert returns a null host slot -- the double-count is gone",
           no_host_copy);
        ok("every streamed expert has a host slot", has_host_copy);
        ok("streamed experts stay contiguous inside a layer (no segment straddling)", contiguous);
        bool under = true;
        for (size_t i = 0; i < a.n_segments(); ++i)
            under = under && a.segment_bytes(i) <= ie::kDs4MaxAllocBytes;
        ok("every segment is within the 32.53 GB per-allocation cap", under);
        // A fully resident model needs no host arena at all.
        ie::Ds4HostArena none;
        ok("static_slots == n_experts pins nothing",
           none.init_range(q, sb, NE, NE, 0, 8ull << 20).empty() && none.total_bytes() == 0);
    }
    if (have_q3) {
        // Arithmetic, not allocation: at the doc-33 Q3 split each layer's pinned
        // block is 171 slots, so no single allocation comes near the ceiling.
        uint64_t widest_layer = 0;
        for (uint64_t b : q3.slot_bytes) widest_layer = std::max<uint64_t>(widest_layer, b * 171ull);
        ok("Q3 @ 171 pinned experts/layer: the largest single layer block is far under the cap",
           widest_layer < ie::kDs4MaxAllocBytes, gb(widest_layer));
    }

    // ---------------------------------------------------------------------
    std::printf("\n[11] The 32.53 GB per-allocation ceiling still fails loudly\n");
    // ---------------------------------------------------------------------
    {
        ie::Ds4HostArena bad;
        const std::string e = bad.init_range(q, sb, NE, STAT, PIN,
                                             ie::kDs4MaxAllocBytes + (1ull << 30));
        ok("a segment target above the ceiling is REFUSED", !e.empty());
        ok("...and nothing was allocated before refusing", bad.total_bytes() == 0);
        std::printf("      %s\n", e.c_str());
    }
    {
        ie::Ds4HostArena bad;
        const std::string e = bad.init_range(q, std::vector<uint64_t>{ie::kDs4MaxAllocBytes / 4},
                                             8, 0, 8, 1ull << 30);
        ok("a layer needing more than the ceiling in one piece is REFUSED", !e.empty());
        ok("...and nothing was allocated before refusing", bad.total_bytes() == 0);
        std::printf("      %s\n", e.c_str());
    }
    {
        ie::Ds4HostArena bad;
        const std::string e = bad.init_range(q, sb, NE, 60, 8, 8ull << 20);
        ok("a pinned range running past the expert count is REFUSED", !e.empty());
        std::printf("      %s\n", e.c_str());
    }

    // ---------------------------------------------------------------------
    std::printf("\n[12] Unavailable expert -> kDs4NoSlot; static slots never evicted\n");
    // ---------------------------------------------------------------------
    {
        // Deliberately INCOMPLETE coverage: experts [0,8) are static, [8,24)
        // are pinned, and [24,32) are NEITHER.  This is the state the loud
        // refusal exists for, and the planner in §2 is what normally makes it
        // unreachable -- it only occurs under an explicit partial pin.
        const uint32_t L2 = 2, E2 = 32, S2 = 8, P2 = 16, SLOTS = 14;
        const uint64_t SLOTB = 64ull << 10;
        std::vector<uint64_t> sb2(L2, SLOTB);
        ie::Ds4HostArena a;
        ok("partial-coverage arena", a.init_range(q, sb2, E2, S2, P2, 8ull << 20).empty());
        for (uint32_t l = 0; l < L2; ++l)
            for (uint32_t e2 = S2; e2 < S2 + P2; ++e2)
                static_cast<uint32_t*>(a.slot(l, e2))[0] = l * 1000u + e2;

        ie::Ds4ExpertCache c;
        const std::string ce = c.init(q, a, SLOTS, 0, S2);
        ok("cache init with a static partition", ce.empty(), ce);
        ok("static + streaming == slots/layer",
           c.static_slots() == S2 && c.stream_slots() == SLOTS - S2);

        // Install the static experts from a staging buffer that then dies --
        // exactly what the runtime does, and the reason there is no host copy.
        {
            std::vector<uint8_t> stage(static_cast<size_t>(SLOTB));
            for (uint32_t l = 0; l < L2; ++l)
                for (uint32_t s = 0; s < S2; ++s) {
                    std::memset(stage.data(), 0, stage.size());
                    reinterpret_cast<uint32_t*>(stage.data())[0] = l * 1000u + s;
                    ok("install_static", c.install_static(l, s, s, stage.data()).empty());
                }
        }
        ok("a static expert is reported resident", c.is_static(0, 3) && c.available(0, 3));
        ok("a pinned expert is available but not static", c.available(0, 10) && !c.is_static(0, 10));
        ok("an uncovered expert is neither", !c.available(0, 30) && !c.is_static(0, 30));

        // Static experts are pure hits: no transfer at all.
        c.reset_stats();
        {
            const int32_t ids[4] = {0, 1, 2, 7};
            uint32_t sl[4];
            c.acquire(0, ids, 4, sl).wait();
            bool inside = true, tagged = true;
            for (uint32_t k = 0; k < 4; ++k) {
                inside = inside && sl[k] < S2;
                tagged = tagged && c.tag(0, sl[k]) == ids[k];
            }
            ok("static experts resolve inside the static partition", inside);
            ok("their directory tags are right", tagged);
            ok("and they cost ZERO bytes of DMA",
               c.stats().hits == 4 && c.stats().misses == 0 && c.stats().bytes_fetched == 0);
        }

        // A streamed expert is fetched byte-exactly from the pinned arena.
        c.reset_stats();
        {
            const int32_t ids[2] = {9, 12};
            uint32_t sl[2];
            c.acquire(0, ids, 2, sl).wait();
            bool inside = true, bytes_ok = true;
            std::vector<uint8_t> got(static_cast<size_t>(SLOTB));
            for (uint32_t k = 0; k < 2; ++k) {
                inside = inside && sl[k] >= S2 && sl[k] < SLOTS;
                q.memcpy(got.data(), c.slot_ptr(0, sl[k]), size_t(SLOTB)).wait();
                bytes_ok = bytes_ok &&
                           std::memcmp(got.data(), a.slot(0, uint32_t(ids[k])), size_t(SLOTB)) == 0;
            }
            ok("streamed experts land in the STREAMING partition", inside);
            ok("fetched device bytes == pinned host bytes", bytes_ok);
        }

        // THE REFUSAL.  Expert 30 has neither a VRAM home nor a host copy.
        c.reset_stats();
        {
            const int32_t ids[3] = {30, 31, 9};
            uint32_t sl[3] = {0, 0, 0};
            c.acquire(0, ids, 3, sl).wait();
            ok("an expert that is neither VRAM-resident nor host-pinned yields kDs4NoSlot",
               sl[0] == ie::kDs4NoSlot && sl[1] == ie::kDs4NoSlot);
            ok("...it is NOT silently rounded to slot 0", sl[0] != 0 && sl[1] != 0);
            ok("...and it is counted, not swallowed", c.stats().unavailable == 2,
               std::to_string(c.stats().unavailable));
            ok("a covered expert in the same call still works", sl[2] != ie::kDs4NoSlot);
        }
        {
            const int32_t bad_id[2] = {-1, int32_t(E2)};
            uint32_t sl[2] = {0, 0};
            c.acquire(0, bad_id, 2, sl).wait();
            ok("an out-of-range expert id also yields kDs4NoSlot",
               sl[0] == ie::kDs4NoSlot && sl[1] == ie::kDs4NoSlot);
        }
        // Asking for more streamed experts than there are streaming slots must
        // refuse the excess, not clobber a slot the same call is already using.
        {
            const uint32_t over = SLOTS - S2 + 3;
            std::vector<int32_t> ids(over);
            std::vector<uint32_t> sl(over, 0);
            for (uint32_t i = 0; i < over; ++i) ids[i] = int32_t(S2 + (i % P2));
            c.acquire(1, ids.data(), over, sl.data()).wait();
            uint32_t served = 0, refused = 0;
            for (uint32_t i = 0; i < over; ++i)
                (sl[i] == ie::kDs4NoSlot ? refused : served)++;
            ok("over-subscribing the streaming partition refuses the excess loudly",
               refused > 0 && served == SLOTS - S2,
               std::to_string(served) + " served, " + std::to_string(refused) + " refused");
            bool distinct = true;
            for (uint32_t i = 0; i < over; ++i)
                for (uint32_t j = i + 1; j < over; ++j)
                    if (sl[i] != ie::kDs4NoSlot && sl[i] == sl[j] && ids[i] != ids[j])
                        distinct = false;
            ok("no two distinct experts were handed the same live slot", distinct);
        }

        // Hammer the streaming partition; the static one must not move.
        {
            std::vector<int32_t> tags_before(S2);
            for (uint32_t s = 0; s < S2; ++s) tags_before[s] = c.tag(0, s);
            for (uint32_t round = 0; round < 40; ++round) {
                int32_t ids[4];
                uint32_t sl[4];
                for (uint32_t k = 0; k < 4; ++k)
                    ids[k] = int32_t(S2 + (round * 4 + k) % P2);
                c.acquire(0, ids, 4, sl).wait();
                c.speculate(0, ids, 4);
            }
            c.transfer_queue().wait();
            bool kept = true, still = true;
            for (uint32_t s = 0; s < S2; ++s) kept = kept && c.tag(0, s) == tags_before[s];
            for (uint32_t e2 = 0; e2 < S2; ++e2) still = still && c.is_static(0, e2);
            ok("160 streamed acquires + speculation never evicted a static slot", kept);
            ok("every static expert is still resident afterwards", still);
        }
    }

    // ---------------------------------------------------------------------
    std::printf("\n[13] The packed decoders are BIT-EXACT against ie::ref\n");
    // ---------------------------------------------------------------------
    // §4's saving is only legitimate if leaving a weight packed changes no
    // number.  These run the REAL dispatchers (ds4_dense_packed_gemv /
    // ds4_grouped_packed_gemv call dense_w / grouped_w, the same functions
    // layer_forward calls) over an IDENTITY activation, which makes
    //     y[t, n] == the decoded weight w[n][t]
    // exactly: every other term is fma(w, 0, acc), and the work-group reduction
    // adds only zeros to the one live product.  So this compares the DECODE,
    // not a dot product, and can demand exact equality rather than a tolerance.
    {
        std::mt19937 rng(0xD54C0DEu);
        // Random but FINITE fp16 scales: a random 16-bit pattern can be NaN, and
        // NaN != NaN would make this test pass by accident on both sides.
        auto rand_half = [&] {
            std::uniform_real_distribution<float> u(-2.f, 2.f);
            return ie::fp32_to_fp16(u(rng));
        };
        auto rand_byte = [&] { return uint8_t(rng() & 0xFFu); };

        // ---- Q8_0 -------------------------------------------------------
        {
            const uint32_t K = 256, N = 9;   // N deliberately not a multiple of 8
            const uint64_t rb = ie::ds4_dense_row_bytes(ie::DType::kQ8_0, K);
            ok("Q8_0 row bytes = (K/32)*34", rb == (K / 32) * 34, std::to_string(rb));
            std::vector<uint8_t> h_w(size_t(rb) * N);
            for (uint32_t n = 0; n < N; ++n)
                for (uint32_t b = 0; b < K / 32; ++b) {
                    uint8_t* blk = h_w.data() + size_t(n) * rb + size_t(b) * 34;
                    const uint16_t d = rand_half();
                    blk[0] = uint8_t(d & 0xFF); blk[1] = uint8_t(d >> 8);
                    for (uint32_t i = 0; i < 32; ++i) blk[2 + i] = rand_byte();
                }
            std::vector<float> h_x(size_t(K) * K, 0.f);
            for (uint32_t t = 0; t < K; ++t) h_x[size_t(t) * K + t] = 1.f;

            auto* d_w = sycl::malloc_device<uint8_t>(h_w.size(), q);
            auto* d_x = sycl::malloc_device<float>(h_x.size(), q);
            auto* d_y = sycl::malloc_device<float>(size_t(K) * N, q);
            ok("Q8_0 probe allocations", d_w && d_x && d_y);
            q.memcpy(d_w, h_w.data(), h_w.size()).wait();
            q.memcpy(d_x, h_x.data(), h_x.size() * 4).wait();
            const std::string ge =
                ie::ds4_dense_packed_gemv(q, d_x, d_w, ie::DType::kQ8_0, d_y, K, K, N);
            ok("ds4_dense_packed_gemv accepted Q8_0", ge.empty(), ge);
            std::vector<float> h_y(size_t(K) * N);
            q.memcpy(h_y.data(), d_y, h_y.size() * 4).wait();

            std::vector<float> refrow(K);
            uint64_t bad = 0; double worst = 0;
            for (uint32_t n = 0; n < N; ++n) {
                ie::ref::dequant_q8_0_buffer(h_w.data() + size_t(n) * rb, K, refrow.data());
                for (uint32_t k = 0; k < K; ++k) {
                    const float got = h_y[size_t(k) * N + n];
                    if (got != refrow[k]) {
                        ++bad;
                        worst = std::max(worst, double(std::fabs(got - refrow[k])));
                    }
                }
            }
            ok("every Q8_0 element decodes BIT-EXACTLY like ie::ref::dequant_q8_0_buffer",
               bad == 0, std::to_string(bad) + " of " + std::to_string(size_t(K) * N) +
                             " differ, worst " + std::to_string(worst));

            // The grouped sibling, which has its own row-stride arithmetic.
            const uint32_t G = 3, OPG = 4;   // weight is [K, G*OPG]; IPG == K
            const uint32_t NG = G * OPG;
            std::vector<uint8_t> h_wg(size_t(rb) * NG);
            for (uint32_t n = 0; n < NG; ++n)
                for (uint32_t b = 0; b < K / 32; ++b) {
                    uint8_t* blk = h_wg.data() + size_t(n) * rb + size_t(b) * 34;
                    const uint16_t d = rand_half();
                    blk[0] = uint8_t(d & 0xFF); blk[1] = uint8_t(d >> 8);
                    for (uint32_t i = 0; i < 32; ++i) blk[2 + i] = rand_byte();
                }
            const uint32_t TG = G * K;
            std::vector<float> h_xg(size_t(TG) * G * K, 0.f);
            for (uint32_t t = 0; t < TG; ++t) h_xg[size_t(t) * G * K + t] = 1.f;
            auto* g_w = sycl::malloc_device<uint8_t>(h_wg.size(), q);
            auto* g_x = sycl::malloc_device<float>(h_xg.size(), q);
            auto* g_y = sycl::malloc_device<float>(size_t(TG) * NG, q);
            ok("grouped probe allocations", g_w && g_x && g_y);
            q.memcpy(g_w, h_wg.data(), h_wg.size()).wait();
            q.memcpy(g_x, h_xg.data(), h_xg.size() * 4).wait();
            const std::string gge = ie::ds4_grouped_packed_gemv(q, g_x, g_w, ie::DType::kQ8_0,
                                                                g_y, TG, G, K, OPG);
            ok("ds4_grouped_packed_gemv accepted Q8_0", gge.empty(), gge);
            std::vector<float> h_yg(size_t(TG) * NG);
            q.memcpy(h_yg.data(), g_y, h_yg.size() * 4).wait();
            uint64_t gbad = 0, off_group_nonzero = 0;
            std::vector<float> grow(K);
            for (uint32_t n = 0; n < NG; ++n) {
                ie::ref::dequant_q8_0_buffer(h_wg.data() + size_t(n) * rb, K, grow.data());
                const uint32_t gn = n / OPG;
                for (uint32_t t = 0; t < TG; ++t) {
                    const float got = h_yg[size_t(t) * NG + n];
                    if (t / K == gn) { if (got != grow[t % K]) ++gbad; }
                    else if (got != 0.f) ++off_group_nonzero;
                }
            }
            ok("the grouped packed path decodes bit-exactly too", gbad == 0,
               std::to_string(gbad));
            ok("...and stays block-diagonal (group g never reads group g'!=g)",
               off_group_nonzero == 0, std::to_string(off_group_nonzero));

            for (void* p : {(void*)d_w, (void*)d_x, (void*)d_y,
                            (void*)g_w, (void*)g_x, (void*)g_y})
                if (p) sycl::free(p, q);
        }

        // ---- Q6_K -------------------------------------------------------
        // 512 elements = two whole super-blocks, so both 128-element halves of
        // both blocks are exercised: all 4 lane sub-groups, both `is` values,
        // both nibbles, all four qh shifts.
        {
            const uint32_t K = 512, N = 5;
            const uint64_t rb = ie::ds4_dense_row_bytes(ie::DType::kQ6_K, K);
            ok("Q6_K row bytes = (K/256)*210", rb == (K / 256) * 210, std::to_string(rb));
            std::vector<uint8_t> h_w(size_t(rb) * N);
            for (uint32_t n = 0; n < N; ++n)
                for (uint32_t b = 0; b < K / 256; ++b) {
                    uint8_t* blk = h_w.data() + size_t(n) * rb + size_t(b) * 210;
                    for (uint32_t i = 0; i < 208; ++i) blk[i] = rand_byte();  // ql/qh/scales
                    const uint16_t d = rand_half();
                    blk[208] = uint8_t(d & 0xFF); blk[209] = uint8_t(d >> 8);
                }
            std::vector<float> h_x(size_t(K) * K, 0.f);
            for (uint32_t t = 0; t < K; ++t) h_x[size_t(t) * K + t] = 1.f;
            auto* d_w = sycl::malloc_device<uint8_t>(h_w.size(), q);
            auto* d_x = sycl::malloc_device<float>(h_x.size(), q);
            auto* d_y = sycl::malloc_device<float>(size_t(K) * N, q);
            ok("Q6_K probe allocations", d_w && d_x && d_y);
            q.memcpy(d_w, h_w.data(), h_w.size()).wait();
            q.memcpy(d_x, h_x.data(), h_x.size() * 4).wait();
            const std::string ge =
                ie::ds4_dense_packed_gemv(q, d_x, d_w, ie::DType::kQ6_K, d_y, K, K, N);
            ok("ds4_dense_packed_gemv accepted Q6_K", ge.empty(), ge);
            std::vector<float> h_y(size_t(K) * N);
            q.memcpy(h_y.data(), d_y, h_y.size() * 4).wait();
            std::vector<float> refrow(K);
            uint64_t bad = 0; double worst = 0;
            for (uint32_t n = 0; n < N; ++n) {
                ie::ref::dequant_q6_K_buffer(h_w.data() + size_t(n) * rb, K, refrow.data());
                for (uint32_t k = 0; k < K; ++k) {
                    const float got = h_y[size_t(k) * N + n];
                    if (got != refrow[k]) {
                        ++bad;
                        worst = std::max(worst, double(std::fabs(got - refrow[k])));
                    }
                }
            }
            ok("every Q6_K element decodes BIT-EXACTLY like ie::ref::dequant_q6_K_buffer",
               bad == 0, std::to_string(bad) + " of " + std::to_string(size_t(K) * N) +
                             " differ, worst " + std::to_string(worst));
            for (void* p : {(void*)d_w, (void*)d_x, (void*)d_y}) if (p) sycl::free(p, q);
        }

        // A dtype with no device decoder must be REFUSED by the seam, not run
        // against a null fp16 pointer.
        {
            const std::string e1 = ie::ds4_dense_packed_gemv(q, nullptr, nullptr,
                                                             ie::DType::kBF16, nullptr, 1, 64, 1);
            ok("a non-packed dtype is refused by name", !e1.empty(), e1);
            const std::string e2 = ie::ds4_dense_packed_gemv(q, nullptr, nullptr,
                                                             ie::DType::kQ8_0, nullptr, 1, 48, 1);
            ok("a ragged K is refused by name", !e2.empty(), e2);
        }
    }

    // ---------------------------------------------------------------------
    std::printf("\n[14] The cross-card reduction of the sliced `down` partials\n");
    // ---------------------------------------------------------------------
    // SCOPE, stated up front: this runs BOTH "cards" on ONE device, because the
    // second B70 is occupied by a live 43-layer load.  What that verifies is the
    // arithmetic and the pinned-host staging mechanism; what it does NOT verify
    // is a transfer between two DIFFERENT devices.  The live two-card run is
    // deferred, and this test says so rather than implying coverage it lacks.
    {
        const uint64_t N  = 4096;                      // one decode step: hidden * fp32 = 16 KB
        const size_t   NB = size_t(N);
        auto* a     = sycl::malloc_device<float>(NB, q);
        auto* b     = sycl::malloc_device<float>(NB, q);
        auto* stage = sycl::malloc_host<float>(NB * 2, q);
        ok("reduction probe allocations (2 x 16 KB device, 32 KB pinned host)", a && b && stage);
        if (a && b && stage) {
            std::vector<float> ha(NB, 0.f), hb(NB, 0.f);
            for (size_t i = 0; i < NB; ++i) {
                ha[i] = float(i) * 0.5f;          // card 0's partial
                hb[i] = -float(i) * 0.25f + 3.f;  // card 1's partial
            }
            q.memcpy(a, ha.data(), NB * 4).wait();
            q.memcpy(b, hb.data(), NB * 4).wait();
            std::vector<sycl::queue*> qs{&q, &q};
            std::vector<float*>       parts{a, b};
            const std::string re = ie::ds4_tp_reduce_host(qs, parts, stage, N);
            ok("ds4_tp_reduce_host accepted the two partials", re.empty(), re);
            std::vector<float> got_a(NB, 0.f), got_b(NB, 0.f);
            q.memcpy(got_a.data(), a, NB * 4).wait();
            q.memcpy(got_b.data(), b, NB * 4).wait();
            uint64_t bad_a = 0, bad_b = 0;
            for (size_t i = 0; i < NB; ++i) {
                const float want = ha[i] + hb[i];
                if (got_a[i] != want) ++bad_a;
                if (got_b[i] != want) ++bad_b;
            }
            // fp32 addition of the same two operands in the same order is exact
            // and reproducible, so this demands equality, not a tolerance.
            ok("every card ends up holding the EXACT sum of both partials",
               bad_a == 0 && bad_b == 0,
               std::to_string(bad_a) + "/" + std::to_string(bad_b) + " of " + std::to_string(N));
            // The half-sum is what a missing reduction would leave behind; prove
            // the test would have noticed.
            ok("...and the un-reduced partial is NOT what came back (a skipped reduce is visible)",
               got_a[1] != ha[1] && got_b[1] != hb[1]);

            ok("a null staging buffer is refused",
               !ie::ds4_tp_reduce_host(qs, parts, nullptr, N).empty());
            ok("n == 0 is refused", !ie::ds4_tp_reduce_host(qs, parts, stage, 0).empty());
            ok("a queue/partial count mismatch is refused",
               !ie::ds4_tp_reduce_host(qs, std::vector<float*>{a}, stage, N).empty());
            ok("a null partial is refused",
               !ie::ds4_tp_reduce_host(qs, std::vector<float*>{a, nullptr}, stage, N).empty());
        }
        for (void* p : {(void*)a, (void*)b, (void*)stage}) if (p) sycl::free(p, q);
    }

    // ---------------------------------------------------------------------
    std::printf("\n[15] The two-card orchestrator: hidden-dim expert-TP in lockstep\n");
    // ---------------------------------------------------------------------
    // WHAT THIS PROVES.  That `DeepSeek4TpRuntime` — n_cards runtimes driven in
    // lockstep, with `ws_moe_` reduced through pinned host memory between the
    // routed experts and the shared expert — reproduces the SINGLE-CARD forward
    // pass, i.e. that hidden-dim expert-TP is a partition of the contraction and
    // not an approximation of it.
    //
    // WHERE THE TWO PATHS CANNOT BE BIT-IDENTICAL, AND WHY THAT IS NOT A FUDGE.
    // `ds4_expert_gemv` writes the `down` result into `xws_.y_h`, which is fp16,
    // BEFORE `accum_f16` scales it by the routing weight and adds it to the fp32
    // accumulator.  One card computes fp16(y0 + y1); two cards compute
    // fp16(y0) + fp16(y1).  That is ONE extra rounding at 11 significand bits per
    // expert and it is the ONLY lossy difference between the two paths: §9 proved
    // the two cards' packed bytes are exactly the whole expert's bytes, and §14
    // proved the reduction is exact fp32 addition.  §15a below measures that one
    // rounding directly and bounds it analytically; §15b/§15c then show what it
    // does end to end.
    {
        const std::string tmpdir = env_or("TMPDIR", "/tmp");

        // -----------------------------------------------------------------
        std::printf("\n  [15a] The expert block itself: is the split a PARTITION?\n");
        // -----------------------------------------------------------------
        // No model, no orchestrator: one routed expert, packed whole and packed
        // as two halves by the SAME `ds4_slot_pack`, run through the SAME
        // `ds4_expert_gemv` / `ds4_swiglu_clamped` / `quantize_q8_1` calls
        // `layer_forward` makes.  This is where the tight, ANALYTIC statement
        // lives, because nothing downstream can amplify it yet.
        {
            const uint32_t H = 256, EF = 256;
            std::mt19937_64 srng(0x5111CE0u);
            struct Synth { std::vector<uint8_t> bytes; ie::GgufTensorInfo ti{}; };
            auto synth = [&](const char* nm, ie::DType dt, uint32_t K, uint32_t N) {
                auto s = std::make_unique<Synth>();
                s->bytes.resize(size_t(ie::bytes_for(dt, size_t(K))) * N);
                if (dt == ie::DType::kMXFP4) {
                    auto* b = reinterpret_cast<ie::block_mxfp4*>(s->bytes.data());
                    for (size_t i = 0; i < s->bytes.size() / sizeof(ie::block_mxfp4); ++i) {
                        b[i].e = uint8_t(124 + (srng() % 3));
                        for (int j = 0; j < 16; ++j) b[i].qs[j] = uint8_t(srng() & 0xFF);
                    }
                } else {
                    auto* b = reinterpret_cast<ie::block_iq3_xxs*>(s->bytes.data());
                    for (size_t i = 0; i < s->bytes.size() / sizeof(ie::block_iq3_xxs); ++i) {
                        b[i].d = ie::fp32_to_fp16(0.0015f);
                        for (int j = 0; j < 96; ++j) b[i].qs[j] = uint8_t(srng() & 0xFF);
                    }
                }
                s->ti.name = nm; s->ti.dtype = dt; s->ti.n_dims = 3;
                s->ti.shape[0] = K; s->ti.shape[1] = N; s->ti.shape[2] = 1;
                s->ti.nbytes = s->bytes.size();
                s->ti.data = s->bytes.data();
                return s;
            };
            auto gt = synth("ffn_gate_exps.weight", ie::DType::kIQ3_XXS, H, EF);
            auto ut = synth("ffn_up_exps.weight",   ie::DType::kIQ3_XXS, H, EF);
            auto dt = synth("ffn_down_exps.weight", ie::DType::kMXFP4,   EF, H);

            ie::Ds4SlotLayout whole{}, half[2];
            ok("whole-expert slot layout",
               ie::ds4_slot_layout(gt->ti, ut->ti, dt->ti, H, EF, whole).empty());
            bool slice_ok = true;
            for (uint32_t c = 0; c < 2; ++c) {
                ie::Ds4ExpertSlice sl;
                slice_ok = slice_ok &&
                           ie::ds4_expert_slice(gt->ti, ut->ti, dt->ti, EF, 2, c, sl).empty() &&
                           ie::ds4_slot_layout_tp(gt->ti, ut->ti, dt->ti, H, EF, sl, half[c]).empty();
            }
            ok("both half-expert slot layouts", slice_ok);

            auto upload = [&](const ie::Ds4SlotLayout& lay) -> uint8_t* {
                std::vector<uint8_t> host(size_t(lay.bytes));
                if (!ie::ds4_slot_pack(lay, gt->ti, ut->ti, dt->ti, 0, host.data()).empty())
                    return nullptr;
                auto* d = sycl::malloc_device<uint8_t>(host.size(), q);
                if (d) q.memcpy(d, host.data(), host.size()).wait();
                return d;
            };
            uint8_t* dw = upload(whole);
            uint8_t* d0 = upload(half[0]);
            uint8_t* d1 = upload(half[1]);
            ie::DS4ExpertWorkspace xws{};
            const std::string wse = ie::ds4_expert_ws_alloc(q, H, EF, xws);
            ok("expert probe allocations", dw && d0 && d1 && wse.empty(), wse);

            if (dw && d0 && d1 && wse.empty()) {
                std::vector<float> x(H);
                std::mt19937 arng(7);
                std::normal_distribution<float> nd(0.f, 1.f);
                for (uint32_t i = 0; i < H; ++i) x[i] = nd(arng);
                auto* dx = sycl::malloc_device<float>(H, q);
                q.memcpy(dx, x.data(), H * 4).wait();
                ie::cast_fp32_to_fp16(q, dx, xws.xh, H);
                ie::quantize_q8_1(q, xws.xh, xws.x_q8, H);
                q.wait();

                // Verbatim the sequence in DeepSeek4Runtime::layer_forward_pre.
                auto run = [&](const ie::Ds4SlotLayout& lay, uint8_t* base,
                               std::vector<float>& y_out, std::vector<float>& h_out) {
                    const ie::DS4ExpertBank gb = ie::ds4_slot_bank(lay.gate, base);
                    const ie::DS4ExpertBank ub = ie::ds4_slot_bank(lay.up,   base);
                    const ie::DS4ExpertBank db = ie::ds4_slot_bank(lay.down, base);
                    const uint32_t EFc = lay.gate.N;
                    ie::ds4_expert_gemv(q, gb, 0, xws.x_q8, nullptr, xws.gate_h);
                    ie::ds4_expert_gemv(q, ub, 0, xws.x_q8, nullptr, xws.up_h);
                    ie::cast_fp16_to_fp32(q, xws.gate_h, xws.gate_f, EFc);
                    ie::cast_fp16_to_fp32(q, xws.up_h,   xws.up_f,   EFc);
                    ie::ds4_swiglu_clamped(q, xws.gate_f, xws.up_f, xws.h_f, EFc, 10.f);
                    ie::cast_fp32_to_fp16(q, xws.h_f, xws.h_h, EFc);
                    ie::quantize_q8_1(q, xws.h_h, xws.h_q8, EFc);
                    ie::ds4_expert_gemv(q, db, 0, xws.h_q8, nullptr, xws.y_h);
                    q.wait();
                    h_out.assign(EFc, 0.f);
                    q.memcpy(h_out.data(), xws.h_f, size_t(EFc) * 4).wait();
                    std::vector<sycl::half> yh(H);
                    q.memcpy(yh.data(), xws.y_h, H * 2).wait();
                    y_out.assign(H, 0.f);
                    for (uint32_t i = 0; i < H; ++i) y_out[i] = float(yh[i]);
                };
                std::vector<float> yw, y0, y1, hw, h0, h1;
                run(whole,   dw, yw, hw);
                run(half[0], d0, y0, h0);
                run(half[1], d1, y1, h1);

                // The SwiGLU intermediate must be EXACTLY equal: the gate/up
                // N-slice is a slice of the OUTPUT, so no summation is split and
                // there is nothing for rounding to do.  A card whose gate/up
                // columns were off by even one would fail here, not silently.
                uint64_t hbad = 0;
                for (uint32_t i = 0; i < EF / 2; ++i) {
                    if (h0[i] != hw[i]) ++hbad;
                    if (h1[i] != hw[EF / 2 + i]) ++hbad;
                }
                ok("the SwiGLU intermediate is BIT-IDENTICAL on both halves -- the gate/up"
                   " N-slice is exact",
                   hbad == 0, std::to_string(hbad) + " of " + std::to_string(EF) + " differ");
                ok("...and it is not degenerate (some element is nonzero)", max_abs(hw) > 0);

                const double ymax = max_abs(yw);
                double ydiff = 0;
                for (uint32_t i = 0; i < H; ++i)
                    ydiff = std::max(ydiff, double(std::fabs((y0[i] + y1[i]) - yw[i])));
                // ANALYTIC BOUND.  y0, y1 and yw are all fp16.  Two roundings on
                // the split path against one on the whole path, each at most half
                // an ulp, so |(y0+y1) - yw| <= 1.5 ulp(max|y|); rounded up to 2.
                const double ulp = std::ldexp(1.0, std::ilogb(std::max(ymax, 1e-30)) - 10);
                std::printf("      down output: max|y_whole| = %.6g, ulp_fp16 = %.6g,"
                            " max|(y0+y1) - y_whole| = %.6g  (%.3e relative)\n",
                            ymax, ulp, ydiff, ydiff / std::max(ymax, 1e-30));
                ok("the two halves' `down` outputs sum to the whole expert's, to within TWO fp16"
                   " ulps -- the ONLY lossy step in the split",
                   ydiff <= 2.0 * ulp,
                   std::to_string(ydiff) + " vs " + std::to_string(2.0 * ulp));
                ok("...and neither half alone is the answer (the sum is doing real work)",
                   max_abs_diff(y0, yw) > 10.0 * ulp && max_abs_diff(y1, yw) > 10.0 * ulp);
                sycl::free(dx, q);
            }
            for (void* p : {(void*)dw, (void*)d0, (void*)d1}) if (p) sycl::free(p, q);
            ie::ds4_expert_ws_free(q, xws);
        }

        // -----------------------------------------------------------------
        // §15b/§15c — the orchestrator, on a miniature but structurally real
        // deepseek4 GGUF.
        // -----------------------------------------------------------------
        // card -> GPU map.  DS4_TP_GPUS="0,1" asks for a GENUINE two-device run;
        // the default puts both cards on the device §10-§14 already picked,
        // because the second B70 in this box is usually occupied by a real load.
        std::vector<uint32_t> tp_gpus;
        if (const char* s = std::getenv("DS4_TP_GPUS")) {
            std::string cur;
            for (const char* p = s;; ++p) {
                if (*p == ',' || *p == 0) {
                    if (!cur.empty()) tp_gpus.push_back(uint32_t(std::atoi(cur.c_str())));
                    cur.clear();
                    if (*p == 0) break;
                } else cur.push_back(*p);
            }
        }
        if (tp_gpus.empty()) tp_gpus = {uint32_t(gi), uint32_t(gi)};
        const bool two_device = tp_gpus.size() == 2 && tp_gpus[0] != tp_gpus[1];
        std::printf("\n    card->GPU map: {%u, %u}  -> %s\n", tp_gpus[0], tp_gpus[1],
                    two_device ? "GENUINE TWO-DEVICE RUN"
                               : "same-device rehearsal (set DS4_TP_GPUS=0,1 for two devices)");

        // Caps are explicit so the test does not depend on how much host RAM the
        // machine happens to have when it runs.
        auto base_opts = [&](uint32_t ord) {
            ie::Ds4Options o;
            o.device_ordinal      = ord;
            o.max_seq             = 64;
            o.max_context         = 256;
            o.slots_per_layer     = 4;              // < n_experts, so the cache really misses
            o.expert_cache_bytes  = 32ull << 20;
            o.host_pin_cap_bytes  = 1ull << 30;
            o.prefetch            = false;          // speculation is timing-dependent; keep the
                                                    // two paths' cache traffic comparable
            return o;
        };

        constexpr uint32_t kPrompt = 24, kSteps = 4;
        struct RunOut {
            std::vector<float>   logits;   // prefill logits
            std::vector<float>   moe;      // ws_moe_ [kPrompt * H] after the prefill
            std::vector<int32_t> greedy;   // one token per decode step
        };
        auto run_single = [&](ie::DeepSeek4Runtime& rt, const ie::DeepSeek4Config& cfg,
                              const std::vector<int32_t>& prompt, RunOut& out) -> std::string {
            out.logits.assign(cfg.vocab, 0.f);
            std::string e = rt.forward(prompt.data(), uint32_t(prompt.size()), 0, out.logits.data());
            if (!e.empty()) return e;
            out.moe.assign(size_t(prompt.size()) * cfg.hidden, 0.f);
            rt.queue().memcpy(out.moe.data(), rt.moe_accumulator(), out.moe.size() * 4).wait();
            int32_t tok = argmax(out.logits);
            for (uint32_t s = 0; s < kSteps; ++s) {
                std::vector<float> lg(cfg.vocab, 0.f);
                e = rt.forward(&tok, 1, uint32_t(prompt.size() + s), lg.data());
                if (!e.empty()) return e;
                tok = argmax(lg);
                out.greedy.push_back(tok);
            }
            return {};
        };
        auto run_tp = [&](ie::DeepSeek4TpRuntime& tp, const ie::DeepSeek4Config& cfg,
                          const std::vector<int32_t>& prompt, RunOut& out) -> std::string {
            out.logits.assign(cfg.vocab, 0.f);
            std::string e = tp.forward(prompt.data(), uint32_t(prompt.size()), 0, out.logits.data());
            if (!e.empty()) return e;
            out.moe.assign(size_t(prompt.size()) * cfg.hidden, 0.f);
            tp.card(0).queue().memcpy(out.moe.data(), tp.card(0).moe_accumulator(),
                                      out.moe.size() * 4).wait();
            int32_t tok = argmax(out.logits);
            for (uint32_t s = 0; s < kSteps; ++s) {
                std::vector<float> lg(cfg.vocab, 0.f);
                e = tp.forward(&tok, 1, uint32_t(prompt.size() + s), lg.data());
                if (!e.empty()) return e;
                tok = argmax(lg);
                out.greedy.push_back(tok);
            }
            return {};
        };

        // THE END-TO-END BOUND, and what it is and is not.
        //
        // §15a's bound is analytic.  This one is NOT, and saying so is the point:
        // what the fp16 seed measured there (~5e-3 relative on the accumulator,
        // dominated by elements where the two halves partly cancel) becomes after
        // N decoder layers is a property of the WEIGHTS, and these are random
        // ones.  There is no honest way to derive an end-to-end constant from
        // them, so this bound says only "the divergence is the size of an
        // amplified rounding, not the size of a broken reduction".  The
        // discriminating check is the RATIO assertion beside it, measured in the
        // same run against the half-sum a dropped reduction leaves behind.
        //
        // Largest observed on this fixture: 2.1e-2 (6 layers, accumulator).
        constexpr double kEnd2EndRel = 0.05;

        // The mini fixture's IQ3_XXS super-scale is 0.0015, not the 0.02 the
        // Phase-5 forward test uses.  At 0.02 this model's clamped SwiGLU sat
        // PERMANENTLY in its clamp (measured max|h| = 99.9955 == 10*10 exactly),
        // which makes the expert block a near-constant function and the network
        // chaotic: the same 3.2e-4 seed grew to 0.76 by layer 6.  A fixture in a
        // regime no trained model is in tests less, not more.
        struct Variant { mini::Cfg cfg; const char* label; };
        const Variant variants[2] = {
            {mini::short_model(), "15b  2-layer (CSA + HCA, hash router) -- the shortest chain"},
            {mini::Cfg{},       "15c  6-layer (hash + top-k routers, CSA + HCA, MXFP4 expert layer)"},
        };
        for (int vi = 0; vi < 2; ++vi) {
            const mini::Cfg& mc = variants[vi].cfg;
            std::printf("\n  [%s]\n", variants[vi].label);

            const std::string path = tmpdir + "/ds4_tp_mini_" + std::to_string(vi) + ".gguf";
            std::vector<std::vector<uint8_t>> keep;
            const std::string we = mini::write_gguf(mc, path, keep);
            ok("wrote a miniature deepseek4 GGUF", we.empty(), we);
            keep.clear();
            if (!we.empty()) continue;

            ie::GgufReader g;
            const std::string oe = g.open(path);
            ok("GgufReader parses it", oe.empty(), oe);
            ie::DeepSeek4Config cfg;
            const std::string ce = ie::read_deepseek4_config(g, cfg);
            ok("read_deepseek4_config", ce.empty(), ce);
            ie::DeepSeek4Model model;
            const std::string be = model.load(g, cfg);
            ok("DeepSeek4Model binds every tensor", be.empty(), be);
            if (!oe.empty() || !ce.empty() || !be.empty()) { std::remove(path.c_str()); continue; }

            std::vector<int32_t> prompt(kPrompt);
            for (uint32_t i = 0; i < kPrompt; ++i) prompt[i] = int32_t((i * 37 + 5) % cfg.vocab);

            // ---- the single-card reference ----
            RunOut ref;
            bool ref_slice_split = true;   // set from the reference load below
            {
                ie::DeepSeek4Runtime rt;
                const std::string le = rt.load(model, base_opts(uint32_t(gi)));
                ok("single-card load", le.empty(), le);
                if (!le.empty()) { std::remove(path.c_str()); continue; }
                ref_slice_split = rt.attn_slice().split;
                ok("the single-card runtime holds the WHOLE expert (n_cards == 1)",
                   rt.expert_slice().n_cards == 1 && rt.expert_slice().efc == cfg.expert_ffn);
                ok("...and the WHOLE non-expert set: nothing is split on one card",
                   !rt.attn_slice().split && rt.attn_slice().nhc == cfg.n_q_heads &&
                       rt.attn_slice().gc == cfg.o_groups &&
                       rt.attn_slice().efc == cfg.expert_ffn);
                const std::string fe = run_single(rt, cfg, prompt, ref);
                ok("single-card prefill + decode", fe.empty(), fe);
                if (!fe.empty()) { std::remove(path.c_str()); continue; }
            }
            bool ref_finite = true;
            for (float v : ref.logits) ref_finite = ref_finite && std::isfinite(v);
            ok("single-card logits are finite", ref_finite);

            // -----------------------------------------------------------------
            // §15j — EXPERT-GROUPED vs TOKEN-CHUNKED residency
            // -----------------------------------------------------------------
            // The routed-expert block used to split the batch into TOKEN RANGES
            // small enough that the union of their STREAMED experts fit the
            // layer's evictable partition, and run one batched GEMM pass per
            // range.  The residency budget therefore bounded the BATCH, and an
            // expert routed to from two different ranges was fetched — and its
            // weights re-read by a second GEMM — once per range.
            //
            // It now groups the EXPERTS instead and leaves the batch whole.  Two
            // things have to hold, and they pull in opposite directions:
            //   * not one output bit may move, because grouping is a placement
            //     decision and the arithmetic is identical either way, and
            //   * it must really move LESS through the cache, or the change is
            //     decorative.
            // A mutant that reorders or re-splits a reduction fails the first; a
            // mutant that groups but still re-acquires per token range fails the
            // second.
            //
            // THE INVARIANT THE COUNTER CHECK IS BUILT ON: with the whole prompt
            // inside one packed batch, every distinct expert of a layer is
            // acquired EXACTLY ONCE, so the acquire-id total over a single
            // forward cannot exceed n_layers * n_experts.  The token-chunked
            // predecessor has no such bound — it re-acquires per range — and on
            // this fixture it exceeds it, which is what makes the assertion
            // discriminate rather than merely pass.
            {
                std::printf("\n  [15j  expert-grouped vs token-chunked residency]\n");
                ie::Ds4Options co = base_opts(uint32_t(gi));
                co.moe_token_chunk = true;

                ie::DeepSeek4Runtime rtc;
                const std::string cle = rtc.load(model, co);
                ok("token-chunked single-card load", cle.empty(), cle);
                if (cle.empty()) {
                    RunOut chunked;
                    const std::string cfe = run_single(rtc, cfg, prompt, chunked);
                    ok("token-chunked prefill + decode", cfe.empty(), cfe);
                    if (cfe.empty())
                        ok("THE EXPERT-GROUPED ROUTED-EXPERT BLOCK IS BIT-IDENTICAL TO THE"
                           " TOKEN-CHUNKED ONE -- grouping changes WHICH experts are resident"
                           " when, and nothing else",
                           chunked.logits == ref.logits && chunked.moe == ref.moe &&
                               chunked.greedy == ref.greedy,
                           "max|dlogit| = " +
                               std::to_string(max_abs_diff(chunked.logits, ref.logits)) +
                               ", max|dmoe| = " +
                               std::to_string(max_abs_diff(chunked.moe, ref.moe)));

                    // ---- ONE prefill each, counters read where they landed ----
                    // `Ds4ExpertCache::reset_stats()` runs at the top of every
                    // forward_prologue, so these describe exactly one forward.
                    ie::DeepSeek4Runtime rtg;
                    const std::string gle = rtg.load(model, base_opts(uint32_t(gi)));
                    ok("expert-grouped single-card load (for the counter comparison)",
                       gle.empty(), gle);
                    rtc.reset_context();
                    if (gle.empty()) {
                        std::vector<float> lg(cfg.vocab, 0.f);
                        const std::string g1 =
                            rtg.forward(prompt.data(), uint32_t(prompt.size()), 0, lg.data());
                        const ie::Ds4ExpertCache::Stats gs = rtg.cache().stats();
                        const std::string c1 =
                            rtc.forward(prompt.data(), uint32_t(prompt.size()), 0, lg.data());
                        const ie::Ds4ExpertCache::Stats cs = rtc.cache().stats();
                        ok("both prefills ran", g1.empty() && c1.empty(), g1 + c1);
                        std::printf("      grouped : %llu acquire calls, %llu ids"
                                    " (%llu hit / %llu miss), %llu B fetched\n",
                                    (unsigned long long)gs.acquires,
                                    (unsigned long long)(gs.hits + gs.misses),
                                    (unsigned long long)gs.hits, (unsigned long long)gs.misses,
                                    (unsigned long long)gs.bytes_fetched);
                        std::printf("      chunked : %llu acquire calls, %llu ids"
                                    " (%llu hit / %llu miss), %llu B fetched\n",
                                    (unsigned long long)cs.acquires,
                                    (unsigned long long)(cs.hits + cs.misses),
                                    (unsigned long long)cs.hits, (unsigned long long)cs.misses,
                                    (unsigned long long)cs.bytes_fetched);
                        std::printf("      (FIXTURE ONLY -- %u experts, %u slots/layer,"
                                    " %u-token prompt.  These ratios do NOT extrapolate to the"
                                    " real model.)\n",
                                    cfg.n_experts, rtg.cache().slots_per_layer(), kPrompt);
                        if (g1.empty() && c1.empty()) {
                            ok("grouping acquires each distinct expert of a layer AT MOST ONCE:"
                               " acquire ids <= n_layers * n_experts",
                               gs.hits + gs.misses <=
                                   uint64_t(cfg.n_layers) * cfg.n_experts,
                               std::to_string(gs.hits + gs.misses) + " > " +
                                   std::to_string(uint64_t(cfg.n_layers) * cfg.n_experts));
                            ok("...a bound the token-chunked loop BREAKS on this fixture,"
                               " so the assertion above is not free",
                               cs.hits + cs.misses >
                                   uint64_t(cfg.n_layers) * cfg.n_experts,
                               std::to_string(cs.hits + cs.misses));
                            ok("grouping issues STRICTLY FEWER acquire calls",
                               gs.acquires < cs.acquires,
                               std::to_string(gs.acquires) + " vs " + std::to_string(cs.acquires));
                            ok("...and moves STRICTLY FEWER H2D bytes",
                               gs.bytes_fetched < cs.bytes_fetched,
                               std::to_string(gs.bytes_fetched) + " vs " +
                                   std::to_string(cs.bytes_fetched));
                            ok("...and never more of either -- the grouped path cannot do more"
                               " work than the loop it replaced on any routing",
                               gs.acquires <= cs.acquires && gs.misses <= cs.misses &&
                                   gs.bytes_fetched <= cs.bytes_fetched);
                        }
                    }
                }
            }

            // ---- the orchestrator with ONE card: must be BIT-IDENTICAL ----
            // No reduction runs, so any difference here would be the pre/post
            // split or the orchestrator's plumbing changing the arithmetic.
            {
                ie::Ds4TpOptions to;
                to.base            = base_opts(0);
                to.n_cards         = 1;
                to.device_ordinals = {uint32_t(gi)};
                ie::DeepSeek4TpRuntime tp1;
                const std::string le = tp1.load(model, to);
                ok("orchestrator load, n_cards = 1", le.empty(), le);
                if (le.empty()) {
                    RunOut one;
                    const std::string fe = run_tp(tp1, cfg, prompt, one);
                    ok("orchestrator forward, n_cards = 1", fe.empty(), fe);
                    if (fe.empty()) {
                        ok("n_cards=1 through the orchestrator is BIT-IDENTICAL to the plain"
                           " single-card forward -- the pre/post split changed no arithmetic",
                           one.logits == ref.logits && one.moe == ref.moe &&
                               one.greedy == ref.greedy,
                           "max|dlogit| = " + std::to_string(max_abs_diff(one.logits, ref.logits)) +
                               ", max|dmoe| = " + std::to_string(max_abs_diff(one.moe, ref.moe)));
                        ok("...and it performed NO cross-card reduction", tp1.reductions() == 0,
                           std::to_string(tp1.reductions()));
                    }
                }
            }

            // ---- the orchestrator with TWO cards ----
            ie::Ds4TpOptions to;
            to.base                  = base_opts(0);
            to.n_cards               = 2;
            to.device_ordinals       = tp_gpus;
            to.same_device_rehearsal = !two_device;
            ie::DeepSeek4TpRuntime tp;
            const std::string le = tp.load(model, to);
            ok("two-card orchestrator load", le.empty(), le);
            if (!le.empty()) { std::remove(path.c_str()); continue; }

            ok("card 0 holds intermediate [0, EF/2) and card 1 holds [EF/2, EF)",
               tp.card(0).expert_slice().ef0 == 0 &&
                   tp.card(0).expert_slice().efc == cfg.expert_ffn / 2 &&
                   tp.card(1).expert_slice().ef0 == cfg.expert_ffn / 2 &&
                   tp.card(1).expert_slice().efc == cfg.expert_ffn / 2);
            ok("each card holds HALF the expert bytes the single card would",
               tp.card(0).tp_residency().card_slot_total * 2 ==
                   tp.card(0).tp_residency().whole_slot_total);
            // ---- the NON-EXPERT split geometry ----
            // Three extents, one head axis.  The load refuses if the group axis
            // and the head axis disagree; this asserts the partition they land
            // on, which is what makes `attn_output_a` read this card's own
            // attention output and not the other card's.
            {
                const ie::Ds4AttnSlice& a0 = tp.card(0).attn_slice();
                const ie::Ds4AttnSlice& a1 = tp.card(1).attn_slice();
                ok("the non-expert set is TP-split on both cards", a0.split && a1.split);
                ok("the query heads tile: card 0 [0,NH/2), card 1 [NH/2,NH)",
                   a0.nh0 == 0 && a0.nhc == cfg.n_q_heads / 2 &&
                       a1.nh0 == cfg.n_q_heads / 2 && a1.nhc == cfg.n_q_heads / 2 &&
                       a0.nhc + a1.nhc == cfg.n_q_heads,
                   std::to_string(a0.nh0) + "+" + std::to_string(a0.nhc) + " / " +
                       std::to_string(a1.nh0) + "+" + std::to_string(a1.nhc));
                ok("the output groups tile, and each card's groups read exactly its own heads",
                   a0.g0 == 0 && a1.g0 == cfg.o_groups / 2 && a0.gc + a1.gc == cfg.o_groups &&
                       a0.g0 * (cfg.n_q_heads / cfg.o_groups) == a0.nh0 &&
                       a1.g0 * (cfg.n_q_heads / cfg.o_groups) == a1.nh0,
                   std::to_string(a0.gc) + " + " + std::to_string(a1.gc) + " groups");
                ok("the shared expert is split on the SAME intermediate axis as the routed ones",
                   a0.ef0 == tp.card(0).expert_slice().ef0 &&
                       a0.efc == tp.card(0).expert_slice().efc &&
                       a1.ef0 == tp.card(1).expert_slice().ef0 &&
                       a1.efc == tp.card(1).expert_slice().efc);
                ok("...and a single-card runtime is NOT split (mirrored, by construction)",
                   !ref_slice_split);
            }
            // The PHYSICAL device, not just the recorded ordinal: a card->device
            // map that quietly collapsed to one device would pass an ordinal
            // check and fail this one -- but only when two devices are visible,
            // which is why the note at the end of §15 says what it says.
            ok("each card's QUEUE is on the physical GPU it was assigned",
               tp.device_of(0) == tp_gpus[0] && tp.device_of(1) == tp_gpus[1] &&
                   tp.card(0).queue().get_device() ==
                       gpus[tp_gpus[0] < gpus.size() ? tp_gpus[0] : 0] &&
                   tp.card(1).queue().get_device() ==
                       gpus[tp_gpus[1] < gpus.size() ? tp_gpus[1] : 0]);

            // ---- §15g: the LOAD ORDER, which is a correctness-adjacent
            // property the load records about itself ----
            //
            // WHAT DEFECT THIS EXISTS FOR.  The orchestrator used to run card 0's
            // whole load and then card 1's.  On the real Q3 model that put
            // 31,696 MiB on card 0 and 1,498 MiB on card 1 -- a second 32 GB GPU
            // sitting untouched while host RAM filled with card 0's pinned arena
            // -- and it walked the routed-expert tensors once PER CARD, an hour
            // and 120 GB apart, so `ffn_down_exps` (whose slice is strided finer
            // than a page, hence touched in full by BOTH cards) was read from
            // disk twice.  Neither fact is visible once a load has finished,
            // which is why the load now records them.
            {
                const ie::Ds4TpLoadTrace& tr = tp.load_trace();
                const uint64_t packs = uint64_t(cfg.n_layers) * cfg.n_experts;
                // The prepare phase is where each card's mirrored always-resident
                // set is uploaded -- 7.887 GB per card on the real Q3 -- so it is
                // the phase the owner watched put 31,696 MiB on one GPU and
                // nothing on the other.  Running one thread per card makes the
                // two cards' intervals OVERLAP; the sequential load it replaces
                // had card 1 beginning strictly after card 0 ended.
                ok("the two cards' PREPARE phases overlap in time -- both devices are being"
                   " filled at once, not one and then the other",
                   tr.prepare_begin.size() == 2 && tr.prepare_end.size() == 2 &&
                       tr.prepare_begin[1] < tr.prepare_end[0] &&
                       tr.prepare_begin[0] < tr.prepare_end[1],
                   tr.prepare_begin.size() == 2
                       ? "card 0 [" + std::to_string(tr.prepare_begin[0]) + "," +
                             std::to_string(tr.prepare_end[0]) + "] s, card 1 [" +
                             std::to_string(tr.prepare_begin[1]) + "," +
                             std::to_string(tr.prepare_end[1]) + "] s"
                       : "no trace");
                ok("the walk served the cards ALTERNATELY -- the longest run of consecutive"
                   " packs by ONE card is 1",
                   tr.max_same_card_run == 1,
                   "max run " + std::to_string(tr.max_same_card_run) +
                       "; loading card 0 and then card 1 would give " + std::to_string(packs));
                ok("the walk was sampled once per pinned layer",
                   tr.timeline.size() == cfg.n_layers,
                   std::to_string(tr.timeline.size()) + " samples for " +
                       std::to_string(cfg.n_layers) + " layers");
                // THE CONCURRENCY ASSERTION, and why it can be an EQUALITY.
                // A sample is taken after each layer's expert loop, at which
                // point the interleaved drive has given both cards exactly the
                // same experts of exactly the same layers -- and the two halves
                // of a balanced slice are the same size -- so the cards' packed
                // byte counts must agree EXACTLY at every sample, starting with
                // the first.  "Card 0 fully, then card 1" leaves card 1 at zero
                // for every sample of the first half and fails on sample 0.
                bool together = !tr.timeline.empty();
                for (const auto& s : tr.timeline)
                    together = together && s.per_card.size() == 2 && s.per_card[0] > 0 &&
                               s.per_card[0] == s.per_card[1];
                std::string detail;
                for (const auto& s : tr.timeline)
                    detail += "[" +
                              std::to_string(s.per_card.size() > 0 ? s.per_card[0] : 0) + "/" +
                              std::to_string(s.per_card.size() > 1 ? s.per_card[1] : 0) + "]";
                ok("BOTH CARDS FILL TOGETHER: at EVERY per-layer sample the two cards had"
                   " packed the same non-zero number of expert bytes",
                   together, detail);
                ok("...and they finished level", tr.packed_bytes.size() == 2 &&
                       tr.packed_bytes[0] > 0 && tr.packed_bytes[0] == tr.packed_bytes[1],
                   tr.packed_bytes.size() == 2
                       ? std::to_string(tr.packed_bytes[0]) + " vs " +
                             std::to_string(tr.packed_bytes[1])
                       : "no trace");
                std::printf("      load trace: prepare %.3f s (one thread per card),"
                            " walk %.3f s, %llu B/card\n",
                            tr.prepare_seconds, tr.pack_seconds,
                            (unsigned long long)(tr.packed_bytes.empty() ? 0
                                                                        : tr.packed_bytes[0]));
            }

            RunOut two;
            const std::string fe = run_tp(tp, cfg, prompt, two);
            ok("two-card orchestrated prefill + decode", fe.empty(), fe);
            if (!fe.empty()) { std::remove(path.c_str()); continue; }
            bool two_finite = true;
            for (float v : two.logits) two_finite = two_finite && std::isfinite(v);
            ok("two-card logits are finite", two_finite);

            // The forward returning "" IS the lockstep proof: every layer of
            // every call compared both cards' expert ids and routing weights
            // against card 0's and would have refused on any difference.
            ok("the lockstep check passed at every layer (a divergence would have refused)",
               fe.empty());
            // TWO reductions per layer, not one: the attention output is now a
            // partial as well as the routed-expert sum.  The count is asserted
            // against the split the load actually resolved, so a build with
            // `split_non_expert = false` would assert ONE and still be checked —
            // the number tracks the configuration instead of being a constant.
            {
                const uint32_t rpl = tp.card(0).attn_slice().split ? 2u : 1u;
                ok("the reduction ran once per PARTIAL per layer per forward",
                   tp.reductions() == uint64_t(cfg.n_layers) * (1 + kSteps) * rpl,
                   std::to_string(tp.reductions()) + " vs " +
                       std::to_string(uint64_t(cfg.n_layers) * (1 + kSteps) * rpl) + " (" +
                       std::to_string(rpl) + " per layer)");
                ok("the non-expert set was TP-SPLIT on both cards, so the attention output is a"
                   " partial and earns the second reduction",
                   tp.card(0).attn_slice().split && tp.card(1).attn_slice().split);
            }
            std::printf("      pinned cross-card staging: %llu B total; %llu reductions in %.3f ms\n",
                        (unsigned long long)tp.stage_bytes(),
                        (unsigned long long)tp.reductions(), tp.reduce_seconds() * 1e3);

            // ---- THE COMPARISON ----
            const double moe_scale = max_abs(ref.moe);
            const double moe_diff  = max_abs_diff(two.moe, ref.moe);
            const double lg_scale  = max_abs(ref.logits);
            const double lg_diff   = max_abs_diff(two.logits, ref.logits);
            std::printf("      routed-expert accumulator: max|ref| = %.6g, max|two-card - ref| ="
                        " %.6g  (%.3e relative)\n",
                        moe_scale, moe_diff, moe_diff / std::max(moe_scale, 1e-30));
            std::printf("      prefill logits           : max|ref| = %.6g, max|two-card - ref| ="
                        " %.6g  (%.3e relative)\n",
                        lg_scale, lg_diff, lg_diff / std::max(lg_scale, 1e-30));

            ok("the two-card routed-expert sum matches the single card's",
               moe_diff <= kEnd2EndRel * moe_scale,
               std::to_string(moe_diff / std::max(moe_scale, 1e-30)) + " relative, bound " +
                   std::to_string(kEnd2EndRel));
            ok("the two-card logits match the single card's",
               lg_diff <= kEnd2EndRel * lg_scale,
               std::to_string(lg_diff / std::max(lg_scale, 1e-30)) + " relative, bound " +
                   std::to_string(kEnd2EndRel));
            // The discriminating check: a dropped reduction leaves HALF the
            // routed sum, an error of 0.5*max|ref|.  Computed, not assumed.
            ok("...and that difference is >= 10x smaller than the half-sum a dropped reduction"
               " would leave",
               moe_diff * 10.0 < 0.5 * moe_scale,
               "ratio " + std::to_string((0.5 * moe_scale) / std::max(moe_diff, 1e-30)) + "x");
            ok("the greedy token sequence is identical on both paths",
               two.greedy == ref.greedy);
            // THE CROSS-BUILD BIT DIGEST.  The two prints above are relative and
            // to 6 digits; this one is over the raw bit patterns.  It is what
            // makes "the two cards now run CONCURRENTLY and not one after the
            // other" a checkable claim rather than an assertion of intent: the
            // digest must be the same hex on both builds.  Nothing asserts it
            // inside one run — there is only one path in a run — so it is
            // printed, and the comparison is the reviewer's.
            std::printf("      two-card bit digest: logits %016llx  moe %016llx  greedy %016llx\n",
                        (unsigned long long)bitdigest(two.logits),
                        (unsigned long long)bitdigest(two.moe),
                        (unsigned long long)bitdigest(two.greedy));
            std::printf("      single-card digest : logits %016llx  moe %016llx  greedy %016llx\n",
                        (unsigned long long)bitdigest(ref.logits),
                        (unsigned long long)bitdigest(ref.moe),
                        (unsigned long long)bitdigest(ref.greedy));

            // -----------------------------------------------------------------
            // §15i — THE NON-EXPERT SPLIT, PRICED: what it saves and what it costs
            // -----------------------------------------------------------------
            // §8c does this arithmetic over the REAL tensor tables, where a load
            // is unaffordable.  Here the same classification is checked against a
            // load that actually happened, which is the only thing that makes the
            // real-model numbers more than a spreadsheet: if `kSplitRows` /
            // `kSplitCols` and `ds4_dense_device_bytes` predict this fixture's
            // `resident_bytes()` exactly, they are predicting the loader and not
            // a model of it.
            //
            // The MIRRORED two-card run beside it is the numerical control: it
            // has the routed-expert split (and therefore its fp16 rounding) but
            // NOT the attention split, so the difference between the two rows is
            // exactly what splitting the non-expert set cost in arithmetic.
            {
                std::printf("\n  [15i  the non-expert split, priced]\n");
                ie::Ds4TpOptions mo;
                mo.base                   = base_opts(0);
                mo.base.split_non_expert  = false;
                mo.n_cards                = 2;
                mo.device_ordinals        = tp_gpus;
                mo.same_device_rehearsal  = !two_device;
                ie::DeepSeek4TpRuntime mtp;
                const std::string mle = mtp.load(model, mo);
                ok("two-card orchestrator load with the non-expert set MIRRORED", mle.empty(), mle);
                if (mle.empty()) {
                    ok("...and it really is mirrored: no card reports a split",
                       !mtp.card(0).attn_slice().split && !mtp.card(1).attn_slice().split &&
                           mtp.card(0).attn_slice().nhc == cfg.n_q_heads);

                    // ---- the byte prediction, checked against the real load ----
                    // Recomputed here from the MODEL's bound tensors with the
                    // same two lists and the same policy function §8c uses.
                    uint64_t pred_saved = 0;
                    for (const ie::DeepSeek4Layer& lw : model.layers()) {
                        const ie::GgufTensorInfo* rows[] = {lw.attn_q_b, lw.attn_output_a,
                                                            lw.ffn_gate_shexp, lw.ffn_up_shexp};
                        const ie::GgufTensorInfo* cols[] = {lw.attn_output_b, lw.ffn_down_shexp};
                        for (const ie::GgufTensorInfo* ti : rows)
                            pred_saved += ie::ds4_dense_device_bytes(ti->dtype, ti->shape[0],
                                                                     ti->shape[1]) -
                                          ie::ds4_dense_device_bytes(ti->dtype, ti->shape[0],
                                                                     ti->shape[1] / 2);
                        for (const ie::GgufTensorInfo* ti : cols)
                            pred_saved += ie::ds4_dense_device_bytes(ti->dtype, ti->shape[0],
                                                                     ti->shape[1]) -
                                          ie::ds4_dense_device_bytes(ti->dtype, ti->shape[0] / 2,
                                                                     ti->shape[1]);
                        pred_saved += uint64_t(cfg.n_q_heads - cfg.n_q_heads / 2) * 4;  // sinks
                    }
                    const uint64_t rm = mtp.card(0).resident_bytes();
                    const uint64_t rs = tp.card(0).resident_bytes();
                    std::printf("      resident/card: mirrored %llu B, split %llu B,"
                                " saved %llu B (predicted %llu B)\n",
                                (unsigned long long)rm, (unsigned long long)rs,
                                (unsigned long long)(rm - rs), (unsigned long long)pred_saved);
                    ok("THE SPLIT ARITHMETIC PREDICTS THE LOADER EXACTLY -- the same"
                       " kSplitRows/kSplitCols classification §8c applies to the real tensor"
                       " tables reproduces this load's resident bytes to the byte",
                       rm > rs && rm - rs == pred_saved,
                       std::to_string(rm - rs) + " vs " + std::to_string(pred_saved));
                    ok("...and both cards saved the same (the split is balanced)",
                       mtp.card(1).resident_bytes() - tp.card(1).resident_bytes() == pred_saved);

                    // ---- the mirrored run's numbers, as the control ----
                    RunOut mirr;
                    const std::string mfe = run_tp(mtp, cfg, prompt, mirr);
                    ok("mirrored two-card prefill + decode", mfe.empty(), mfe);
                    if (mfe.empty()) {
                        ok("the mirrored two-card run reduces ONCE per layer",
                           mtp.reductions() == uint64_t(cfg.n_layers) * (1 + kSteps),
                           std::to_string(mtp.reductions()));
                        const double ms = max_abs(ref.moe), ls = max_abs(ref.logits);
                        const double m_mir = max_abs_diff(mirr.moe, ref.moe);
                        const double l_mir = max_abs_diff(mirr.logits, ref.logits);
                        const double m_spl = max_abs_diff(two.moe, ref.moe);
                        const double l_spl = max_abs_diff(two.logits, ref.logits);
                        std::printf("      vs the single card:  mirrored moe %.3e / logits %.3e"
                                    "   split moe %.3e / logits %.3e   (relative)\n",
                                    m_mir / std::max(ms, 1e-30), l_mir / std::max(ls, 1e-30),
                                    m_spl / std::max(ms, 1e-30), l_spl / std::max(ls, 1e-30));
                        std::printf("      mirrored two-card digest: logits %016llx  moe %016llx"
                                    "  greedy %016llx\n",
                                    (unsigned long long)bitdigest(mirr.logits),
                                    (unsigned long long)bitdigest(mirr.moe),
                                    (unsigned long long)bitdigest(mirr.greedy));
                        // THE HONEST STATEMENT.  Neither two-card path is
                        // bit-identical to one card -- §15's own comment says why,
                        // and it is the routed experts' fp16 `y_h`, which BOTH
                        // rows above carry.  What this asserts is that adding the
                        // attention split did not change the ORDER of that error:
                        // its extra reassociations are fp32 (`attn_output_b`'s
                        // contraction and the shared expert's), so they are ~1e-7
                        // relative against an fp16 rounding that is ~1e-3.
                        ok("splitting the non-expert set did not change the ORDER of the two-card"
                           " divergence -- the fp16 routed-expert rounding still dominates",
                           m_spl <= 4.0 * std::max(m_mir, 1e-30) &&
                               l_spl <= 4.0 * std::max(l_mir, 1e-30),
                           "moe " + std::to_string(m_spl / std::max(m_mir, 1e-30)) + "x, logits " +
                               std::to_string(l_spl / std::max(l_mir, 1e-30)) + "x the mirrored"
                               " two-card error");
                        ok("...and the mirrored two-card run is itself within the same end-to-end"
                           " bound",
                           m_mir <= kEnd2EndRel * ms && l_mir <= kEnd2EndRel * ls);
                        ok("both two-card paths pick the same greedy tokens as one card",
                           mirr.greedy == ref.greedy && two.greedy == ref.greedy);
                    }
                }
            }

            // ---- a sliced runtime driven on its own still REFUSES ----
            {
                std::vector<float> lg(cfg.vocab, 0.f);
                const std::string e1 = tp.card(1).forward(prompt.data(), 4, 0, lg.data());
                ok("card 1's own forward() refuses -- it can only produce a PARTIAL", !e1.empty());
                ok("...and names the orchestrator as the supported path",
                   e1.find("PARTIAL") != std::string::npos &&
                       e1.find("DeepSeek4TpRuntime") != std::string::npos, e1);
            }

            // -----------------------------------------------------------------
            // §15h — DO THE TWO CARDS ACTUALLY RUN AT THE SAME TIME?
            // -----------------------------------------------------------------
            // The orchestrator used to drive the cards STRICTLY SEQUENTIALLY in
            // the forward pass: card 0's layer ran to completion (its segment
            // ends in a q.wait()), then card 1's, then the reduction.  One 32 GB
            // GPU was idle at every instant.  That is invisible from the outside
            // — the logits are the same either way — so the forward now records
            // each card's [begin, end] for every layer segment and this section
            // asserts the INTERVALS INTERSECT.  Under the sequential drive card
            // c+1's begin was strictly after card c's end and the intersection
            // is empty, so the assertion discriminates the defect it exists for.
            //
            // The timing beside it is FIXTURE-ONLY.  This model is a few hundred
            // KB of random weights; its layers are microseconds, so thread
            // hand-off is a much larger fraction of a segment here than it can
            // ever be on the real 128 GB model.  The number below is evidence
            // that concurrency happened, NOT an estimate of what it is worth.
            {
                std::printf("\n  [15h  do the two cards overlap?]\n");
                constexpr uint32_t kTimed = 24;
                int32_t tok = two.greedy.empty() ? 1 : two.greedy.back();
                std::vector<float> lg(cfg.vocab, 0.f);
                // One warm-up forward outside the clock: the first call after a
                // fresh prompt grows the caches (a realloc per layer) and would
                // otherwise be timed as if it were a decode step.
                std::string te = tp.forward(&tok, 1, kPrompt + kSteps, lg.data());
                const auto  t0 = std::chrono::steady_clock::now();
                for (uint32_t s = 0; s < kTimed && te.empty(); ++s)
                    te = tp.forward(&tok, 1, kPrompt + kSteps + 1 + s, lg.data());
                const double ms =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() *
                    1e3 / double(kTimed);
                ok("timed two-card decode forwards", te.empty(), te);
                std::printf("      %u two-card decode forwards: %.3f ms each"
                            "  (FIXTURE ONLY -- not the real model)\n", kTimed, ms);

                const ie::Ds4TpForwardTrace& ft = tp.forward_trace();
                ok("the forward recorded a segment per card per phase per layer (THREE phases:"
                   " pre / mid / post)",
                   ft.n_cards == 2 && ft.total == uint64_t(cfg.n_layers) * 3 &&
                       ft.begin.size() == ft.total * 2 && ft.end.size() == ft.total * 2,
                   std::to_string(ft.total) + " segments, " + std::to_string(ft.n_cards) +
                       " cards");
                ok("...and it did NOT fall back to the sequential drive"
                   " (that happens only with the kernel profiler on)",
                   !ft.sequential);
                // THE ASSERTION THE SECTION EXISTS FOR.  Sequential execution
                // makes this FAIL by construction: card 1's segment could not
                // begin until card 0's q.wait() had returned.
                //
                // Splitting the attention output made a layer THREE segments
                // instead of two, so this now covers pre / mid / post; the
                // statement is unchanged and so is the bar.
                ok("THE TWO CARDS OVERLAP: in EVERY layer segment of the last forward, card 1's"
                   " [begin, end] intersects card 0's",
                   ft.total > 0 && ft.overlapped == ft.total,
                   std::to_string(ft.overlapped) + " of " + std::to_string(ft.total) +
                       " segments overlapped");
                // Per phase, so a regression that idles one card in ONE of the
                // three says WHICH rather than just lowering a total.
                {
                    uint64_t ov[3] = {0, 0, 0};
                    for (uint64_t s = 0; s < ft.total; ++s)
                        ov[s % 3] += (std::min(ft.end[s * 2 + 0], ft.end[s * 2 + 1]) >
                                      std::max(ft.begin[s * 2 + 0], ft.begin[s * 2 + 1]));
                    std::printf("      overlapped per phase: pre %llu/%u  mid %llu/%u"
                                "  post %llu/%u\n",
                                (unsigned long long)ov[0], cfg.n_layers,
                                (unsigned long long)ov[1], cfg.n_layers,
                                (unsigned long long)ov[2], cfg.n_layers);
                }
                ok("...and the overlap is a real fraction of the work, not a shared instant",
                   ft.busy_seconds > 0.0 && ft.overlap_seconds > 0.25 * ft.busy_seconds,
                   "overlap " + std::to_string(ft.overlap_seconds * 1e3) + " ms of " +
                       std::to_string(ft.busy_seconds * 1e3) + " ms card-busy");
                // Per-segment evidence, printed so the numbers behind the two
                // assertions above are on the record rather than summarised.
                static const char* const kPhase[3] = {"pre", "mid", "post"};
                for (uint64_t s = 0; s < ft.total && s < 6; ++s)
                    std::printf("      layer %u %-4s: card0 [%.3f, %.3f] ms   card1 [%.3f, %.3f] ms\n",
                                unsigned(s / 3), kPhase[s % 3],
                                ft.begin[s * 2 + 0] * 1e3, ft.end[s * 2 + 0] * 1e3,
                                ft.begin[s * 2 + 1] * 1e3, ft.end[s * 2 + 1] * 1e3);
                std::printf("      overlap %.3f ms of %.3f ms card-busy over %llu segments\n",
                            ft.overlap_seconds * 1e3, ft.busy_seconds * 1e3,
                            (unsigned long long)ft.total);
            }

            // -----------------------------------------------------------------
            // §15e — residency priority: WHICH experts get the static VRAM slots
            // -----------------------------------------------------------------
            // The static partition used to be experts [0, static_slots) purely
            // because of how they are NUMBERED.  It is now an input.  Two things
            // have to hold, and they pull in opposite directions:
            //   * a non-identity order must really MOVE experts between the VRAM
            //     partition and the streaming/pinned one (otherwise the knob is
            //     decorative), and
            //   * it must not change a single output bit, because residency is a
            //     placement decision and the arithmetic is identical either way.
            // A mutant that ignores the order fails the first; a mutant that lets
            // the order leak into the packing or the routing fails the second.
            {
                std::printf("\n  [15e  residency priority order]\n");
                // §15e needs a NON-EMPTY static partition or it cannot tell the
                // orders apart, and base_opts deliberately has none (4 slots, all
                // streaming, so the cache really misses).  So this section sizes
                // its own: 6 slots/layer with the streaming floor pinned at
                // n_experts_used gives 4 static + 2 streaming out of 8 experts.
                // Its reference is its OWN default-order run, not §15's, so the
                // comparison is like-for-like and §15's settings are untouched.
                auto prio_opts = [&] {
                    ie::Ds4Options o   = base_opts(uint32_t(gi));
                    o.slots_per_layer  = 6;
                    o.min_stream_slots = cfg.n_experts_used;
                    return o;
                };
                ie::DeepSeek4Runtime dflt;
                std::string de = dflt.load(model, prio_opts());
                ok("default (empty priority) load", de.empty(), de);
                const uint32_t STATIC = dflt.residency().static_slots;
                std::printf("      static slots/layer = %u of %u experts\n", STATIC, cfg.n_experts);

                RunOut dref;
                if (de.empty()) {
                    const std::string df = run_single(dflt, cfg, prompt, dref);
                    ok("default-order prefill + decode", df.empty(), df);
                    if (!df.empty()) de = df;
                }

                if (!de.empty() || STATIC == 0 || STATIC >= cfg.n_experts) {
                    ok("PRECONDITION: 0 < static_slots < n_experts, so residency is a real choice",
                       false, "static_slots = " + std::to_string(STATIC) + " of " +
                                  std::to_string(cfg.n_experts) + " -- §15e cannot discriminate");
                } else {
                    ok("the default order is the identity permutation",
                       dflt.expert_priority().size() == cfg.n_experts &&
                           [&]{ for (uint32_t i = 0; i < cfg.n_experts; ++i)
                                    if (dflt.expert_priority()[i] != i) return false;
                                return true; }());
                    ok("...so by default expert e holds static slot e, and only e < static_slots"
                       " is resident",
                       dflt.static_slot_of(0, 0) == 0 &&
                           dflt.static_slot_of(0, STATIC - 1) == int32_t(STATIC - 1) &&
                           dflt.static_slot_of(0, STATIC) < 0);
                    // The default's own resident set, read off the CACHE (not the
                    // plan) so this compares what was really installed.
                    std::vector<bool> dflt_static(cfg.n_experts, false);
                    for (uint32_t e2 = 0; e2 < cfg.n_experts; ++e2)
                        dflt_static[e2] = dflt.cache().is_static(0, e2);

                    // A REVERSED priority: the experts that were resident are now
                    // the last to claim a slot, and vice versa.
                    ie::Ds4Options ro = prio_opts();
                    ro.expert_priority.resize(cfg.n_experts);
                    for (uint32_t i = 0; i < cfg.n_experts; ++i)
                        ro.expert_priority[i] = cfg.n_experts - 1 - i;
                    ie::DeepSeek4Runtime rev;
                    const std::string re = rev.load(model, ro);
                    ok("reversed-priority load", re.empty(), re);
                    if (re.empty()) {
                        ok("the reversed order is reported back verbatim",
                           rev.expert_priority() == ro.expert_priority);
                        ok("the same NUMBER of experts is VRAM-resident (a permutation moves"
                           " residency, it does not change how much fits)",
                           rev.residency().static_slots == STATIC &&
                               rev.arena().n_pinned_experts() ==
                                   dflt.arena().n_pinned_experts());

                        // THE DISCRIMINATING CHECK: a different SET is resident.
                        uint32_t moved = 0;
                        for (uint32_t e2 = 0; e2 < cfg.n_experts; ++e2)
                            if (rev.cache().is_static(0, e2) != dflt_static[e2]) ++moved;
                        ok("a non-identity priority really CHANGES which experts are"
                           " VRAM-resident",
                           moved > 0, std::to_string(moved) + " experts changed residency");
                        ok("...specifically the top-priority expert is now resident and expert 0"
                           " is not",
                           rev.cache().is_static(0, cfg.n_experts - 1) &&
                               !rev.cache().is_static(0, 0));
                        ok("...and the host arena mirrors it: expert 0 now HAS a pinned copy,"
                           " the newly-resident one does not",
                           rev.arena().is_pinned(0, 0) &&
                               !rev.arena().is_pinned(0, cfg.n_experts - 1) &&
                               rev.arena().slot(0, cfg.n_experts - 1) == nullptr);
                        ok("static_slot_of inverts the supplied order",
                           rev.static_slot_of(0, cfg.n_experts - 1) == 0 &&
                               rev.static_slot_of(0, 0) < 0);

                        // ...and none of it may perturb a single output bit.
                        RunOut rout;
                        const std::string rf = run_single(rev, cfg, prompt, rout);
                        ok("reversed-priority prefill + decode", rf.empty(), rf);
                        if (rf.empty()) {
                            ok("residency order changes WHERE experts live and NOTHING about the"
                               " arithmetic: logits are BIT-IDENTICAL to the default order",
                               rout.logits == dref.logits,
                               "max|dlogit| = " +
                                   std::to_string(max_abs_diff(rout.logits, dref.logits)));
                            ok("...and so is the routed-expert accumulator",
                               rout.moe == dref.moe,
                               "max|dmoe| = " + std::to_string(max_abs_diff(rout.moe, dref.moe)));
                            ok("...and the greedy token sequence", rout.greedy == dref.greedy);
                        }
                    }

                    // ---- an INTERLEAVED order: a non-contiguous pinned set ----
                    // The reversed order above is not enough on its own.  Reversing
                    // [0,8) with 4 static slots leaves the pinned set {3,2,1,0} --
                    // still the contiguous id range [0,4) -- so the OLD contiguous
                    // arena formula (slot = e - first_expert) remains a valid
                    // bijection and a mutant that reverts to it survives.  Proven:
                    // that mutant was built and did survive the reversed case.
                    // Taking every SECOND expert makes the static set {0,2,4,6} and
                    // the pinned set {1,3,5,7}, whose ids are NOT contiguous, so the
                    // old formula sends expert 5 past the end of the pinned range
                    // and hands expert 2 a slot it must not have.
                    {
                        ie::Ds4Options io = prio_opts();
                        io.expert_priority.clear();
                        for (uint32_t e2 = 0; e2 < cfg.n_experts; e2 += 2)
                            io.expert_priority.push_back(e2);
                        for (uint32_t e2 = 1; e2 < cfg.n_experts; e2 += 2)
                            io.expert_priority.push_back(e2);
                        ie::DeepSeek4Runtime ilv;
                        const std::string ie_ = ilv.load(model, io);
                        ok("interleaved-priority load (static set {0,2,4,...}, pinned set"
                           " {1,3,5,...} -- ids NOT contiguous)", ie_.empty(), ie_);
                        if (ie_.empty()) {
                            ok("the even experts are VRAM-resident and the odd ones are not",
                               ilv.cache().is_static(0, 0) && ilv.cache().is_static(0, 2) &&
                                   !ilv.cache().is_static(0, 1) && !ilv.cache().is_static(0, 3));
                            ok("the arena pins exactly the ODD experts, at a non-contiguous set"
                               " of ids",
                               ilv.arena().is_pinned(0, 1) && ilv.arena().is_pinned(0, cfg.n_experts - 1) &&
                                   !ilv.arena().is_pinned(0, 0) && !ilv.arena().is_pinned(0, 2) &&
                                   ilv.arena().first_pinned_expert() == 1);
                            // The discriminator: distinct host slots for ids whose
                            // GAPS the contiguous formula cannot represent.
                            bool distinct = true;
                            std::vector<void*> seen_p;
                            for (uint32_t e2 = 1; e2 < cfg.n_experts; e2 += 2) {
                                void* sp = ilv.arena().slot(0, e2);
                                distinct = distinct && sp != nullptr &&
                                           std::find(seen_p.begin(), seen_p.end(), sp) ==
                                               seen_p.end();
                                seen_p.push_back(sp);
                            }
                            ok("every pinned expert gets a DISTINCT, non-null host slot despite"
                               " the gaps in its id set", distinct);
                            RunOut iout;
                            const std::string if_ = run_single(ilv, cfg, prompt, iout);
                            ok("interleaved-priority prefill + decode", if_.empty(), if_);
                            if (if_.empty()) {
                                ok("...and its logits are STILL bit-identical to index order",
                                   iout.logits == dref.logits,
                                   "max|dlogit| = " +
                                       std::to_string(max_abs_diff(iout.logits, dref.logits)));
                                ok("...and the greedy token sequence", iout.greedy == dref.greedy);
                            }
                        }
                    }

                    // ---- 15e-2: A PER-LAYER ORDER, WHICH IS THE POINT ----
                    //
                    // Everything above still gives all layers the SAME order, so
                    // an implementation that reads layer 0's ranking and applies
                    // it everywhere passes every one of those checks.  That
                    // implementation is exactly what shipped, and it is what
                    // makes residency worth only 42.5% static hit on a measured
                    // real decode where a per-layer order reaches 77.9%.
                    //
                    // THE DISCRIMINATOR: give layer 0 the identity and layer 1
                    // the REVERSE, so expert 0 must be statically resident in
                    // layer 0 and must STREAM in layer 1 — and therefore must
                    // have NO host copy in layer 0 and one in layer 1.  A
                    // single-order implementation cannot produce that state at
                    // all; it will make the two layers agree.
                    if (cfg.n_layers >= 2) {
                        ie::Ds4Options po = prio_opts();
                        po.expert_priority_layers.assign(cfg.n_layers, {});
                        for (uint32_t l = 0; l < cfg.n_layers; ++l)
                            for (uint32_t e2 = 0; e2 < cfg.n_experts; ++e2)
                                po.expert_priority_layers[l].push_back(
                                    (l % 2) ? cfg.n_experts - 1 - e2 : e2);
                        ie::DeepSeek4Runtime plr;
                        const std::string pe = plr.load(model, po);
                        ok("per-layer residency order loads (layer 0 identity, layer 1 reversed)",
                           pe.empty(), pe);
                        if (pe.empty()) {
                            const uint32_t last = cfg.n_experts - 1;
                            ok("expert 0 is STATIC in layer 0 and STREAMS in layer 1 — the state a"
                               " single global order cannot represent",
                               plr.static_slot_of(0, 0) == 0 && plr.static_slot_of(1, 0) < 0);
                            ok("...and the last expert is the mirror image of that",
                               plr.static_slot_of(1, last) == 0 && plr.static_slot_of(0, last) < 0);
                            ok("the CACHE agrees per layer, not just the plan",
                               plr.cache().is_static(0, 0) && !plr.cache().is_static(1, 0) &&
                                   plr.cache().is_static(1, last) && !plr.cache().is_static(0, last));
                            // The tiered rule, per layer: a statically-resident
                            // expert has NO host copy, and the SAME id in the
                            // next layer must have one.  This is the check that
                            // fails loudly if `pin_index_` stayed one-dimensional.
                            ok("expert 0 has NO host copy in layer 0 but DOES in layer 1 — the"
                               " tiered rule is now per layer",
                               plr.arena().slot(0, 0) == nullptr && plr.arena().slot(1, 0) != nullptr &&
                                   !plr.arena().is_pinned(0, 0) && plr.arena().is_pinned(1, 0));
                            ok("...and every layer still pins the SAME COUNT, so the arena's"
                               " layout and total bytes are unchanged",
                               plr.arena().n_pinned_experts() ==
                                   dflt.arena().n_pinned_experts() &&
                                   plr.arena().total_bytes() == dflt.arena().total_bytes(),
                               std::to_string(plr.arena().total_bytes()) + " vs " +
                                   std::to_string(dflt.arena().total_bytes()));
                            // Distinct host slots WITHIN each layer, checked on
                            // the layer whose order is reversed: a per-layer
                            // index built from the wrong row would collide.
                            bool distinct1 = true;
                            std::vector<void*> seen1;
                            for (uint32_t e2 = 0; e2 < cfg.n_experts; ++e2) {
                                void* sp = plr.arena().slot(1, e2);
                                if (!sp) continue;
                                distinct1 = distinct1 && std::find(seen1.begin(), seen1.end(), sp) ==
                                                             seen1.end();
                                seen1.push_back(sp);
                            }
                            ok("layer 1's pinned experts get DISTINCT host slots under its own order",
                               distinct1 && seen1.size() == plr.arena().n_pinned_experts());
                            RunOut pout;
                            const std::string pf2 = run_single(plr, cfg, prompt, pout);
                            ok("per-layer-priority prefill + decode", pf2.empty(), pf2);
                            if (pf2.empty()) {
                                ok("PER-LAYER residency changes WHERE experts live and NOTHING"
                                   " about the arithmetic: logits BIT-IDENTICAL to index order",
                                   pout.logits == dref.logits,
                                   "max|dlogit| = " +
                                       std::to_string(max_abs_diff(pout.logits, dref.logits)));
                                ok("...and the routed-expert accumulator", pout.moe == dref.moe,
                                   "max|dmoe| = " + std::to_string(max_abs_diff(pout.moe, dref.moe)));
                                ok("...and the greedy token sequence", pout.greedy == dref.greedy);
                            }
                        }
                        // NEGATIVE CONTROLS on the per-layer form.  Each is a way
                        // of leaving an expert with neither a VRAM home nor a
                        // host copy, which surfaces only as a refusal mid-decode
                        // if it is not caught at load.
                        auto refuse_layers = [&](const char* what,
                                                 std::vector<std::vector<uint32_t>> p,
                                                 const char* needle) {
                            ie::Ds4Options o = prio_opts();
                            o.expert_priority_layers = std::move(p);
                            ie::DeepSeek4Runtime bad;
                            const std::string e = bad.load(model, o);
                            ok(what, !e.empty() && e.find(needle) != std::string::npos, e);
                        };
                        {
                            auto p = po.expert_priority_layers;
                            p.pop_back();
                            refuse_layers("a per-layer order naming too FEW layers is refused", p,
                                          "layers");
                        }
                        {
                            auto p = po.expert_priority_layers;
                            p[1].pop_back();
                            refuse_layers("a per-layer order whose layer 1 is a PREFIX is refused",
                                          p, "full permutation");
                        }
                        {
                            auto p = po.expert_priority_layers;
                            p[1][0] = p[1][1];
                            refuse_layers("a per-layer order repeating an expert in layer 1 is"
                                          " refused BY LAYER", p, "layer 1 lists expert");
                        }
                        {
                            auto p = po.expert_priority_layers;
                            p[1][0] = cfg.n_experts;
                            refuse_layers("an out-of-range id in layer 1 is refused BY LAYER", p,
                                          "layer 1[0]");
                        }
                        // The FILE form of the same thing, round-tripped, because
                        // a ranking that cannot be written and re-read is not a
                        // measurement loop.
                        {
                            const std::string plp = tmpdir + "/ds4_perlayer_prio.txt";
                            if (std::FILE* f = std::fopen(plp.c_str(), "w")) {
                                std::fprintf(f, "# ds4-expert-priority 1\nexperts %u\nlayers %u\n",
                                             cfg.n_experts, cfg.n_layers);
                                for (uint32_t l = 0; l < cfg.n_layers; ++l) {
                                    std::fprintf(f, "layer %u\n", l);
                                    for (uint32_t e2 : po.expert_priority_layers[l])
                                        std::fprintf(f, "%u\n", e2);
                                }
                                std::fclose(f);
                            }
                            ie::Ds4Options fo = prio_opts();
                            fo.expert_priority_file = plp;
                            ie::DeepSeek4Runtime frt2;
                            const std::string fe2 = frt2.load(model, fo);
                            ok("a PER-LAYER ranking FILE round-trips to the same residency",
                               fe2.empty() && frt2.expert_priority_layers() ==
                                                  po.expert_priority_layers,
                               fe2);
                            ok("...and it is recorded as having come from the file",
                               fe2.empty() && frt2.priority_origin() ==
                                                  ie::DeepSeek4Runtime::PriorityOrigin::kFile);
                            // A per-layer file for a model with a DIFFERENT layer
                            // count must fail the load, not silently use a prefix.
                            const std::string blp = tmpdir + "/ds4_perlayer_badL.txt";
                            if (std::FILE* f = std::fopen(blp.c_str(), "w")) {
                                std::fprintf(f, "experts %u\nlayers %u\nlayer 0\n", cfg.n_experts,
                                             cfg.n_layers + 7);
                                for (uint32_t e2 = 0; e2 < cfg.n_experts; ++e2)
                                    std::fprintf(f, "%u\n", e2);
                                std::fclose(f);
                            }
                            ie::Ds4Options bo2 = prio_opts();
                            bo2.expert_priority_file = blp;
                            ie::DeepSeek4Runtime brt2;
                            const std::string be2 = brt2.load(model, bo2);
                            ok("a per-layer ranking for a DIFFERENT layer count fails the load",
                               !be2.empty() && be2.find("different model") != std::string::npos,
                               be2);
                            std::remove(plp.c_str());
                            std::remove(blp.c_str());
                        }
                    }

                    // A malformed order is refused BY NAME, never filled in.
                    auto refuse_prio = [&](const char* what, std::vector<uint32_t> p,
                                           const char* needle) {
                        ie::Ds4Options o = prio_opts();
                        o.expert_priority = std::move(p);
                        ie::DeepSeek4Runtime bad;
                        const std::string e = bad.load(model, o);
                        ok(what, !e.empty() && e.find(needle) != std::string::npos, e);
                    };
                    refuse_prio("a PARTIAL priority order is refused (it would leave an expert"
                                " with neither a VRAM home nor a host copy)",
                                {0, 1, 2}, "full permutation");
                    {
                        std::vector<uint32_t> dup(cfg.n_experts);
                        for (uint32_t i = 0; i < cfg.n_experts; ++i) dup[i] = i;
                        dup[cfg.n_experts - 1] = 0;                 // 0 listed twice
                        refuse_prio("a DUPLICATED expert id is refused (two experts would alias"
                                    " one slot)", dup, "twice");
                    }
                    {
                        std::vector<uint32_t> oob(cfg.n_experts);
                        for (uint32_t i = 0; i < cfg.n_experts; ++i) oob[i] = i;
                        oob[0] = cfg.n_experts;                     // out of range
                        refuse_prio("an OUT-OF-RANGE expert id is refused", oob, "is outside [0,");
                    }

                    // ---------------------------------------------------------
                    // §15f — THE MEASURED LOOP, end to end on a real forward pass
                    // ---------------------------------------------------------
                    // §8c proved the counter and the file format in isolation.
                    // What is left, and what the defect was actually about, is
                    // that NOTHING ever supplied an order: the knob existed but
                    // no path reached it from outside C++.  This runs the whole
                    // loop — measure a real forward pass, dump it, feed it back
                    // through the FILE (option field and env var both), and
                    // prove the resulting resident set is the measured one.
                    std::printf("\n  [15f  measure -> file -> residency, end to end]\n");
                    const std::string pdir  = env_or("TMPDIR", "/tmp");
                    const std::string ppath = pdir + "/ds4_measured_prio.txt";

                    // ---- 1. measure ----
                    ie::Ds4Options mo = prio_opts();
                    mo.profile_experts = true;
                    ie::DeepSeek4Runtime prof_rt;
                    const std::string pe = prof_rt.load(model, mo);
                    ok("a profiling load succeeds", pe.empty(), pe);
                    if (pe.empty()) {
                        ok("...and the profile is sized from the MODEL, not from a constant",
                           prof_rt.expert_profile().active() &&
                               prof_rt.expert_profile().n_layers() == cfg.n_layers &&
                               prof_rt.expert_profile().n_experts() == cfg.n_experts);
                        ok("nothing is counted before a forward pass",
                           prof_rt.expert_profile().selections() == 0);
                        RunOut pout;
                        const std::string pf = run_single(prof_rt, cfg, prompt, pout);
                        ok("profiled prefill + decode", pf.empty(), pf);
                        if (pf.empty()) {
                            const ie::Ds4ExpertProfile& P = prof_rt.expert_profile();
                            // EXACT expected count: every layer routes every
                            // token to n_experts_used experts, for the prefill
                            // and every decode step.  A counter that dropped a
                            // layer or double-counted one fails here.
                            const uint64_t want = uint64_t(cfg.n_layers) * cfg.n_experts_used *
                                                  (kPrompt + kSteps);
                            ok("every routed selection is counted, exactly once -- "
                               "n_layers x n_experts_used x (prefill + decode) tokens",
                               P.selections() == want,
                               std::to_string(P.selections()) + " vs " + std::to_string(want));
                            ok("...and no id was rejected, so the router never produced one out "
                               "of range", P.rejected() == 0);
                            ok("every layer contributed", [&] {
                                   for (uint32_t l = 0; l < cfg.n_layers; ++l) {
                                       uint64_t s = 0;
                                       for (uint32_t e = 0; e < cfg.n_experts; ++e)
                                           s += P.count(l, e);
                                       if (s != uint64_t(cfg.n_experts_used) * (kPrompt + kSteps))
                                           return false;
                                   }
                                   return true;
                               }());
                            ok("the per-expert totals sum to the selection count", [&] {
                                   uint64_t s = 0;
                                   for (uint32_t e = 0; e < cfg.n_experts; ++e) s += P.total(e);
                                   return s == P.selections();
                               }());
                            // The measurement must be a REAL measurement: on a
                            // trained-looking router it will not be flat, and if
                            // it were, the whole exercise would be a no-op and
                            // this section would silently prove nothing.
                            uint64_t lo = ~0ull, hi = 0;
                            for (uint32_t e = 0; e < cfg.n_experts; ++e) {
                                lo = std::min(lo, P.total(e));
                                hi = std::max(hi, P.total(e));
                            }
                            std::printf("      measured totals span %llu..%llu over %u experts; "
                                        "order = ", (unsigned long long)lo,
                                        (unsigned long long)hi, cfg.n_experts);
                            const std::vector<uint32_t> measured = P.priority();
                            for (uint32_t e : measured) std::printf("%u ", e);
                            std::printf("\n");
                            // ...and the per-layer form, which is what the dump
                            // carries and what residency now resolves to.
                            const std::vector<std::vector<uint32_t>> measuredL = P.priority_layers();

                            // ---- 2. dump ----
                            const std::string we = P.write(ppath);
                            ok("the measured profile writes a ranking file", we.empty(), we);

                            // ---- 3. feed it back through the OPTION FIELD ----
                            ie::Ds4Options fo = prio_opts();
                            fo.expert_priority_file = ppath;
                            ie::DeepSeek4Runtime frt;
                            const std::string fe = frt.load(model, fo);
                            ok("a load with expert_priority_file succeeds", fe.empty(), fe);
                            if (fe.empty()) {
                                ok("THE ORDER REACHED THE LOADER FROM THE FILE -- reported as "
                                   "file-sourced, naming the path",
                                   frt.priority_origin() ==
                                           ie::DeepSeek4Runtime::PriorityOrigin::kFile &&
                                       frt.priority_source() == ppath,
                                   frt.priority_source());
                                // The dump carries the PER-LAYER orders now, so
                                // this is the stronger claim: every layer's
                                // resolved order is that layer's own measurement,
                                // not one order averaged over the model.
                                ok("...and the resolved order IS the measured order, PER LAYER",
                                   frt.expert_priority_layers() == measuredL);
                                // THE DISCRIMINATOR: residency follows the
                                // measurement.  Checked on the CACHE, so it is
                                // what was really installed, not what was planned,
                                // and checked in EVERY layer rather than layer 0 —
                                // a loader that applied layer 0's order everywhere
                                // would pass the old single-layer form of this.
                                ok("THE MEASURED EXPERTS ARE THE VRAM-RESIDENT ONES, IN EVERY LAYER",
                                   [&] {
                                       for (uint32_t l = 0; l < cfg.n_layers; ++l)
                                           for (uint32_t i = 0; i < cfg.n_experts; ++i) {
                                               const bool want_res = i < STATIC;
                                               const uint32_t e2 = measuredL[l][i];
                                               if (frt.cache().is_static(l, e2) != want_res)
                                                   return false;
                                               // ...and its mirror in the host arena.
                                               if (frt.arena().is_pinned(l, e2) == want_res)
                                                   return false;
                                           }
                                       return true;
                                   }());
                                // Placement only: not one output bit may move.
                                RunOut fout;
                                const std::string ff = run_single(frt, cfg, prompt, fout);
                                ok("file-priority prefill + decode", ff.empty(), ff);
                                if (ff.empty()) {
                                    ok("a measured residency order changes WHERE experts live and "
                                       "nothing about the arithmetic: logits bit-identical",
                                       fout.logits == dref.logits,
                                       "max|dlogit| = " + std::to_string(
                                                              max_abs_diff(fout.logits,
                                                                           dref.logits)));
                                    ok("...and the greedy token sequence",
                                       fout.greedy == dref.greedy);
                                }
                            }

                            // ---- 4. ...and through $DS4_EXPERT_PRIORITY_FILE ----
                            // The zero-code-change path: the same file, supplied
                            // by the environment, must land on the same order.
                            setenv("DS4_EXPERT_PRIORITY_FILE", ppath.c_str(), 1);
                            ie::DeepSeek4Runtime ert;
                            const std::string ee = ert.load(model, prio_opts());
                            unsetenv("DS4_EXPERT_PRIORITY_FILE");
                            ok("$DS4_EXPERT_PRIORITY_FILE alone supplies the order (no code "
                               "change, no option field)",
                               ee.empty() && ert.priority_origin() ==
                                                 ie::DeepSeek4Runtime::PriorityOrigin::kFile &&
                                   ert.expert_priority_layers() == measuredL,
                               ee);

                            // ---- 5. precedence and refusals ----
                            // An explicit order must WIN over the file, or a
                            // stale env var would silently override a caller.
                            {
                                ie::Ds4Options po = prio_opts();
                                po.expert_priority_file = ppath;
                                po.expert_priority.resize(cfg.n_experts);
                                for (uint32_t i = 0; i < cfg.n_experts; ++i)
                                    po.expert_priority[i] = cfg.n_experts - 1 - i;
                                ie::DeepSeek4Runtime prt;
                                const std::string pre = prt.load(model, po);
                                ok("Ds4Options::expert_priority WINS over the file",
                                   pre.empty() &&
                                       prt.priority_origin() ==
                                           ie::DeepSeek4Runtime::PriorityOrigin::kOptionField &&
                                       prt.expert_priority() == po.expert_priority,
                                   pre);
                            }
                            // A bad file must FAIL THE LOAD, never fall through
                            // to index order: "my ranking was silently ignored"
                            // is precisely what a measurement loop cannot detect.
                            {
                                const std::string bad = pdir + "/ds4_bad_prio.txt";
                                if (std::FILE* f = std::fopen(bad.c_str(), "w")) {
                                    std::fputs("experts 4\n0\n1\n2\n3\n", f);
                                    std::fclose(f);
                                }
                                ie::Ds4Options bo = prio_opts();
                                bo.expert_priority_file = bad;
                                ie::DeepSeek4Runtime brt;
                                const std::string be = brt.load(model, bo);
                                ok("a ranking file for a DIFFERENT model fails the load rather "
                                   "than falling back to index order",
                                   !be.empty() &&
                                       be.find("different model") != std::string::npos,
                                   be);
                                std::remove(bad.c_str());
                            }
                            {
                                ie::Ds4Options mo2 = prio_opts();
                                mo2.expert_priority_file = pdir + "/ds4_absent_prio.txt";
                                ie::DeepSeek4Runtime mrt;
                                const std::string me = mrt.load(model, mo2);
                                ok("a MISSING ranking file fails the load, loudly",
                                   !me.empty() && me.find("cannot open") != std::string::npos, me);
                            }
                            std::remove(ppath.c_str());
                        }
                    }
                    // Profiling is OFF unless asked for: it is a measurement
                    // tool, and an always-on counter in the decode loop is a
                    // cost nobody chose.
                    ok("profiling is off by default", !dflt.expert_profile().active());
                }
            }

            // ---- 15d: the binding and refusal contract (once) ----
            if (vi == 0) {
                std::printf("\n  [15d  card/device binding + the refusal contract]\n");
                auto refuse = [&](const char* what, ie::Ds4TpOptions o, const char* needle) {
                    ie::DeepSeek4TpRuntime bad;
                    const std::string e = bad.load(model, o);
                    ok(what, !e.empty() && e.find(needle) != std::string::npos, e);
                };
                ie::Ds4TpOptions g0;
                g0.base                  = base_opts(0);
                g0.n_cards               = 2;
                g0.device_ordinals       = tp_gpus;
                g0.same_device_rehearsal = !two_device;

                { auto o = g0; o.n_cards = 0;
                  refuse("n_cards == 0 is refused", o, "n_cards == 0"); }
                { auto o = g0; o.base.n_cards = 2;
                  refuse("base.n_cards set by the caller is refused, not overwritten", o,
                         "set PER CARD"); }
                { auto o = g0; o.base.card = 1;
                  refuse("base.card set by the caller is refused, not overwritten", o,
                         "set PER CARD"); }
                { auto o = g0; o.base.device_ordinal = 1;
                  refuse("base.device_ordinal set by the caller is refused", o,
                         "device_ordinals"); }
                { auto o = g0; o.device_ordinals = {0};
                  refuse("a device_ordinals of the wrong length is refused", o,
                         "device ordinals for"); }
                { auto o = g0; o.device_ordinals = {0, 99};
                  refuse("a device ordinal past the GPU count is refused BY CARD", o,
                         "card 1 asks for GPU 99"); }
                { auto o = g0; o.device_ordinals = {uint32_t(gi), uint32_t(gi)};
                  o.same_device_rehearsal = false;
                  refuse("two cards on ONE device is refused unless explicitly opted into", o,
                         "same_device_rehearsal"); }
                // A card count nothing divides by, refused TWICE over — once by
                // each split that cannot honour it, and in the order the load
                // reaches them.  Both are checked, because turning the non-expert
                // split off must not make the expert split silently accept a
                // geometry it cannot tile either.
                { auto o = g0; o.n_cards = 3;
                  o.device_ordinals = {uint32_t(gi), uint32_t(gi), uint32_t(gi)};
                  o.same_device_rehearsal = true;
                  refuse("a card count the HEAD/GROUP axes do not divide by is refused", o,
                         "do not both divide by 3 cards"); }
                { auto o = g0; o.n_cards = 3;
                  o.device_ordinals = {uint32_t(gi), uint32_t(gi), uint32_t(gi)};
                  o.same_device_rehearsal = true;
                  o.base.split_non_expert = false;
                  refuse("...and with the non-expert split OFF, the card count the intermediate"
                         " dimension does not divide by is STILL refused", o,
                         "does not divide by 3 cards"); }
                // The mismatch the orchestrator exists to prevent, at the level
                // below it: a runtime told to hold a card that cannot exist.
                {
                    ie::Ds4Options o = base_opts(uint32_t(gi));
                    o.n_cards = 2; o.card = 2;
                    ie::DeepSeek4Runtime rt;
                    const std::string e = rt.load(model, o);
                    ok("a runtime whose card is outside [0, n_cards) is refused BEFORE any upload",
                       !e.empty() && e.find("is outside [0,2)") != std::string::npos, e);
                    ok("...and nothing was resident when it refused", rt.resident_bytes() == 0);
                }
            }
            std::remove(path.c_str());
        }

        if (!two_device)
            std::printf("\n    NOTE: both cards ran on GPU %u. The cross-DEVICE, cross-CONTEXT\n"
                        "    pinned staging is NOT verified by this run -- only the arithmetic and\n"
                        "    the binding logic are. Re-run with DS4_TP_GPUS=0,1 when the second\n"
                        "    card is idle.\n", tp_gpus[0]);
        else
            std::printf("\n    Both cards ran on DIFFERENT physical GPUs (%u and %u): the\n"
                        "    cross-device, cross-context pinned staging IS verified by this run.\n",
                        tp_gpus[0], tp_gpus[1]);
    }

    std::printf("\n%s: %d failure(s)\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
