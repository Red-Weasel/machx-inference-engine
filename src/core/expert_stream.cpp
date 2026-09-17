// src/core/expert_stream.cpp — DeepSeek-V4-Flash expert streaming (Phase 5).
//
// See include/ie/expert_stream.hpp for the design rationale.  Two rules govern
// every line here:
//
//   1. A FETCH IS ONE MEMCPY.  No repack, no gather, no dtype branch on the
//      fetch path.  Everything a fetch could possibly have to reformat is done
//      once, at load time, by ds4_slot_pack.
//   2. AN ALLOCATION FAILURE IS LOUD.  Neither arena silently downsizes, falls
//      back to pageable memory, or continues with fewer bytes than it promised.
//      The 32.53 GB per-allocation ceiling is checked BEFORE the call, so the
//      diagnostic names the number that was too big rather than reporting a
//      null pointer after the fact.

#include "ie/expert_stream.hpp"

#include "ie/quant_blocks.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>

#include <sys/mman.h>
#include <unistd.h>

namespace ie {
namespace {

constexpr uint64_t kAlign = 64;

uint64_t align_up(uint64_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

std::string bytes_str(uint64_t b) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3f GB (%llu B)", double(b) / 1e9, (unsigned long long)b);
    return buf;
}

// Plane sizes for one [K, N] matrix of the given dtype, in bytes.
// Returns false for a dtype with no device GEMV in this engine — the same
// whitelist ds4_expert_bank_upload enforces, applied per tensor, never per role.
bool plane_sizes(DType dt, uint32_t K, uint32_t N, uint64_t& l0, uint64_t& l1, uint64_t& l2) {
    switch (dt) {
        case DType::kIQ3_XXS:
            if (K % kQK_K) return false;
            l0 = uint64_t(N) * (uint64_t(K) / 4);            // gp  uint8
            l1 = uint64_t(N) * (uint64_t(K) / 32) * 4u;      // ap  uint32
            l2 = uint64_t(N) * (uint64_t(K) / kQK_K) * 2u;   // dp  uint16
            return true;
        case DType::kMXFP4:
            if (K % kQK_MXFP4) return false;
            l0 = uint64_t(N) * (uint64_t(K) / 2);            // mx_qs uint8
            l1 = uint64_t(N) * (uint64_t(K) / 32);           // mx_e  uint8
            l2 = 0;
            return true;
        default:
            return false;
    }
}

// Elements per quantisation block for the dtypes that have a slot layout.  0
// for anything else, which `plane_sizes` has already rejected.
uint32_t block_elems(DType dt) {
    switch (dt) {
        case DType::kIQ3_XXS: return kQK_K;
        case DType::kMXFP4:   return kQK_MXFP4;
        default:              return 0;
    }
}

// `fileK`/`fileN` are the tensor's own dimensions; `k0`/`K` and `n0`/`N` are the
// sub-matrix this card owns (the whole thing when n_cards == 1).
std::string lay_mat(const GgufTensorInfo& ti, uint32_t fileK, uint32_t fileN,
                    uint32_t k0, uint32_t K, uint32_t n0, uint32_t N,
                    uint64_t& cursor, Ds4MatPlanes& m) {
    const std::string nm(ti.name);
    if (ti.n_dims != 3) return nm + ": expected a 3-D expert tensor";
    if (ti.shape[0] != fileK || ti.shape[1] != fileN)
        return nm + ": shape [" + std::to_string(ti.shape[0]) + "," +
               std::to_string(ti.shape[1]) + "] != expected [" + std::to_string(fileK) + "," +
               std::to_string(fileN) + "]";
    if (uint64_t(k0) + K > fileK || uint64_t(n0) + N > fileN)
        return nm + ": slice [" + std::to_string(k0) + "+" + std::to_string(K) + "," +
               std::to_string(n0) + "+" + std::to_string(N) + ") runs past the tensor";
    uint64_t l0 = 0, l1 = 0, l2 = 0;
    if (!plane_sizes(ti.dtype, K, N, l0, l1, l2))
        return nm + ": expert dtype " + std::string(type_name(ti.dtype)) +
               " has no streaming slot layout in this engine";
    m.dt = ti.dtype; m.K = K; m.N = N;
    m.src_K = fileK; m.src_N = fileN; m.k0 = k0; m.n0 = n0;
    m.off0 = cursor;             m.len0 = l0; cursor = align_up(cursor + l0);
    m.off1 = cursor;             m.len1 = l1; cursor = align_up(cursor + l1);
    if (l2) { m.off2 = cursor;   m.len2 = l2; cursor = align_up(cursor + l2); }
    else    { m.off2 = 0;        m.len2 = 0; }
    return {};
}

// Packs expert `e`'s sub-matrix [k0,k0+K) x [n0,n0+N) of one matrix into the
// slot.  This is the load-time repack the fetch path is forbidden to do.
//
// Both dtypes store a [K,N] expert as N contiguous COLUMNS of K/block blocks,
// and both SoA plane forms keep that per-column grouping, so:
//   * an N-slice is a contiguous run of source columns (no block constraint);
//   * a K-slice keeps blocks [k0/block, (k0+K)/block) of EVERY column, which is
//     only expressible when k0 and K are whole multiples of the block —
//     enforced by ds4_expert_slice, not assumed here.
void pack_mat(const Ds4MatPlanes& m, const GgufTensorInfo& ti, uint32_t e, uint8_t* dst) {
    if (m.dt == DType::kIQ3_XXS) {
        const uint64_t src_spc = uint64_t(m.src_K) / kQK_K;   // super-blocks per source column
        const uint32_t spc     = m.K / kQK_K;                 // ...per sliced column
        const uint32_t sk0     = m.k0 / kQK_K;
        const auto* base = reinterpret_cast<const block_iq3_xxs*>(ti.data) +
                           (uint64_t(e) * m.src_N + m.n0) * src_spc;
        uint8_t*  gp = dst + m.off0;
        auto*     ap = reinterpret_cast<uint32_t*>(dst + m.off1);
        auto*     dp = reinterpret_cast<uint16_t*>(dst + m.off2);
        for (uint32_t n = 0; n < m.N; ++n) {
            const block_iq3_xxs* col = base + uint64_t(n) * src_spc + sk0;
            uint8_t*  gcol = gp + uint64_t(n) * (uint64_t(m.K) / 4);
            uint32_t* acol = ap + uint64_t(n) * (uint64_t(m.K) / 32);
            uint16_t* dcol = dp + uint64_t(n) * spc;
            for (uint32_t s = 0; s < spc; ++s) {
                dcol[s] = col[s].d;
                std::memcpy(gcol + uint64_t(s) * 64, col[s].qs, 64);      // 8 grid bytes x 8 sub-blocks
                std::memcpy(acol + uint64_t(s) * 8,  col[s].qs + 64, 32); // 8 scale/sign words
            }
        }
        return;
    }
    // MXFP4 — the gpt-oss plane split, verbatim.
    const uint32_t bpc     = m.K / kQK_MXFP4;
    const uint64_t src_bpc = uint64_t(m.src_K) / kQK_MXFP4;
    const uint32_t bk0     = m.k0 / kQK_MXFP4;
    const auto* base = reinterpret_cast<const block_mxfp4*>(ti.data) +
                       (uint64_t(e) * m.src_N + m.n0) * src_bpc;
    uint8_t* qs = dst + m.off0;
    uint8_t* ep = dst + m.off1;
    for (uint32_t n = 0; n < m.N; ++n) {
        const block_mxfp4* col = base + uint64_t(n) * src_bpc + bk0;
        for (uint32_t b = 0; b < bpc; ++b) {
            const uint64_t di = uint64_t(n) * bpc + b;
            std::memcpy(qs + di * 16, col[b].qs, 16);
            ep[di] = col[b].e;
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// slot layout
// ---------------------------------------------------------------------------
std::string ds4_expert_slice(const GgufTensorInfo& gate, const GgufTensorInfo& up,
                             const GgufTensorInfo& down, uint32_t EF,
                             uint32_t n_cards, uint32_t card, Ds4ExpertSlice& out) {
    out = Ds4ExpertSlice{};
    if (n_cards == 0)    return "ds4_expert_slice: n_cards == 0";
    if (card >= n_cards) return "ds4_expert_slice: card " + std::to_string(card) +
                                " is outside [0," + std::to_string(n_cards) + ")";
    if (EF % n_cards)
        return "ds4_expert_slice: the intermediate dimension " + std::to_string(EF) +
               " does not divide by " + std::to_string(n_cards) +
               " cards — hidden-dim expert-TP would have to round, which loses weights";
    const uint32_t efc = EF / n_cards;
    const uint32_t ef0 = card * efc;

    // gate/up are sliced along their OUTPUT dim, which never cuts a block; only
    // their shape is checked here (lay_mat repeats it) so a transposed file is
    // caught before the block arithmetic below reads the wrong dimension.
    for (const GgufTensorInfo* ti : {&gate, &up}) {
        if (ti->n_dims != 3 || ti->shape[1] != EF)
            return std::string(ti->name) + ": expected [*, " + std::to_string(EF) +
                   ", *] for an intermediate-dim slice";
        if (block_elems(ti->dtype) == 0)
            return std::string(ti->name) + ": dtype " + std::string(type_name(ti->dtype)) +
                   " has no streaming slot layout in this engine";
    }
    // `down` is sliced along its CONTRACTION dim, and THAT cuts blocks.
    if (down.n_dims != 3 || down.shape[0] != EF)
        return std::string(down.name) + ": expected [" + std::to_string(EF) +
               ", *, *] for an intermediate-dim slice";
    const uint32_t blk = block_elems(down.dtype);
    if (blk == 0)
        return std::string(down.name) + ": dtype " + std::string(type_name(down.dtype)) +
               " has no streaming slot layout in this engine";
    if (ef0 % blk || efc % blk)
        return std::string(down.name) + ": " + std::string(type_name(down.dtype)) +
               " quantises " + std::to_string(blk) + " elements per block, but a " +
               std::to_string(n_cards) + "-card split of the intermediate dimension " +
               std::to_string(EF) + " puts card " + std::to_string(card) + " at [" +
               std::to_string(ef0) + "," + std::to_string(ef0 + efc) +
               ") — that lands mid-block, which would decode garbage.  Refusing.";

    out.n_cards = n_cards; out.card = card; out.ef0 = ef0; out.efc = efc;
    return {};
}

std::string ds4_slot_layout_tp(const GgufTensorInfo& gate, const GgufTensorInfo& up,
                               const GgufTensorInfo& down, uint32_t H, uint32_t EF,
                               const Ds4ExpertSlice& sl, Ds4SlotLayout& out) {
    out = Ds4SlotLayout{};
    if (sl.efc == 0 || uint64_t(sl.ef0) + sl.efc > EF)
        return "ds4_slot_layout_tp: slice [" + std::to_string(sl.ef0) + "," +
               std::to_string(uint64_t(sl.ef0) + sl.efc) + ") is not inside [0," +
               std::to_string(EF) + ")";
    uint64_t cursor = 0;
    std::string e;
    // gate/up: [H, EF] sliced on N.  down: [EF, H] sliced on K.  Same `efc`
    // either way — that is what makes the two halves of an expert a partition.
    if (e = lay_mat(gate, H, EF, 0, H, sl.ef0, sl.efc, cursor, out.gate); !e.empty()) return e;
    if (e = lay_mat(up,   H, EF, 0, H, sl.ef0, sl.efc, cursor, out.up);   !e.empty()) return e;
    if (e = lay_mat(down, EF, H, sl.ef0, sl.efc, 0, H, cursor, out.down); !e.empty()) return e;
    out.bytes = cursor;
    return {};
}

std::string ds4_slot_layout(const GgufTensorInfo& gate, const GgufTensorInfo& up,
                            const GgufTensorInfo& down, uint32_t H, uint32_t EF,
                            Ds4SlotLayout& out) {
    Ds4ExpertSlice whole;
    whole.n_cards = 1; whole.card = 0; whole.ef0 = 0; whole.efc = EF;
    return ds4_slot_layout_tp(gate, up, down, H, EF, whole, out);
}

DS4ExpertBank ds4_slot_bank(const Ds4MatPlanes& m, void* base) {
    auto* p = static_cast<uint8_t*>(base);
    DS4ExpertBank b;
    b.dtype = m.dt; b.K = m.K; b.N = m.N; b.E = 1;
    if (m.dt == DType::kIQ3_XXS) {
        b.gp = p + m.off0;
        b.ap = reinterpret_cast<uint32_t*>(p + m.off1);
        b.dp = reinterpret_cast<uint16_t*>(p + m.off2);
        b.gp_stride = m.len0;
        b.ap_stride = m.len1 / 4;
        b.dp_stride = m.len2 / 2;
    } else {
        b.mx_qs = p + m.off0;
        b.mx_e  = p + m.off1;
        b.mx_qs_stride = m.len0;
        b.mx_e_stride  = m.len1;
    }
    return b;
}

std::string ds4_slot_pack(const Ds4SlotLayout& lay, const GgufTensorInfo& gate,
                          const GgufTensorInfo& up, const GgufTensorInfo& down,
                          uint32_t e, void* dst) {
    if (!gate.data || !up.data || !down.data)
        return "ds4_slot_pack: an expert tensor has no mmap backing";
    if (e >= gate.shape[2] || e >= up.shape[2] || e >= down.shape[2])
        return "ds4_slot_pack: expert index " + std::to_string(e) + " out of range";
    auto* p = static_cast<uint8_t*>(dst);
    pack_mat(lay.gate, gate, e, p);
    pack_mat(lay.up,   up,   e, p);
    pack_mat(lay.down, down, e, p);
    return {};
}

// ---------------------------------------------------------------------------
// kernel readahead over the mmap (see the header for the measurements)
// ---------------------------------------------------------------------------
void ds4_stream_advise(const void* p, uint64_t nbytes) noexcept {
    if (!p || nbytes == 0) return;
    const long pg = ::sysconf(_SC_PAGESIZE);
    if (pg <= 0) return;
    // madvise requires a page-aligned start; round the start DOWN and lengthen
    // to match, so the whole requested range is still covered.
    const uintptr_t a  = reinterpret_cast<uintptr_t>(p);
    const uintptr_t lo = a & ~uintptr_t(pg - 1);
    const uint64_t  n  = nbytes + (a - lo);
    if (n > uint64_t(SIZE_MAX)) return;
    // MADV_NORMAL, not MADV_WILLNEED: measured, the latter does nothing while
    // the VMA is still MADV_RANDOM (header).  The return value is deliberately
    // ignored — a refusal means the pages fault in on demand exactly as before,
    // which is slow but never wrong.  Nothing to report, nothing to fail.
    (void)::madvise(reinterpret_cast<void*>(lo), size_t(n), MADV_NORMAL);
}

void ds4_stream_advise_experts(const GgufTensorInfo& gate, const GgufTensorInfo& up,
                               const GgufTensorInfo& down) noexcept {
    for (const GgufTensorInfo* ti : {&gate, &up, &down})
        ds4_stream_advise(ti->data, ti->nbytes);
}

// ---------------------------------------------------------------------------
// live pinned-host ceiling
// ---------------------------------------------------------------------------
uint64_t ds4_host_pin_cap_live(std::string& why, bool* meminfo_ok) {
    if (meminfo_ok) *meminfo_ok = false;
    std::FILE* f = std::fopen("/proc/meminfo", "r");
    if (!f) {
        why = "/proc/meminfo unreadable — cannot derive a live pinned-host cap";
        return 0;
    }
    uint64_t total_kb = 0, avail_kb = 0, cached_kb = 0;
    char line[256];
    while (std::fgets(line, sizeof(line), f)) {
        unsigned long long v = 0;
        if (std::sscanf(line, "MemTotal: %llu kB", &v) == 1)          total_kb = v;
        else if (std::sscanf(line, "MemAvailable: %llu kB", &v) == 1) avail_kb = v;
        else if (std::sscanf(line, "Cached: %llu kB", &v) == 1)       cached_kb = v;
    }
    std::fclose(f);
    if (total_kb == 0 || avail_kb == 0) {
        why = "/proc/meminfo has no MemTotal/MemAvailable — cannot derive a live pinned-host cap";
        return 0;
    }
    if (meminfo_ok) *meminfo_ok = true;
    const uint64_t total  = total_kb * 1024ull;
    const uint64_t avail  = avail_kb * 1024ull;
    const uint64_t cached = cached_kb * 1024ull;

    // Term 1 — what is free NOW, less a working set for the page cache and the
    // desktop.  This is the term that moves with the cache, and the one that
    // would have refused the 120.393 GB pin behind the 2026-08-02 oomd kill.
    const uint64_t reserve_dyn = std::max(kDs4HostPinReserveMin, total / kDs4HostPinReserveDiv);
    const uint64_t cap_dyn     = avail > reserve_dyn ? avail - reserve_dyn : 0;
    // Term 2 — RAM that stays unpinned whatever the box currently looks like.
    // A pinned page is unreclaimable, so this is the floor under the pool the
    // kernel can still reclaim from for the whole of the run.
    const uint64_t reserve_abs = std::max(kDs4HostPinReserveMin, total / kDs4HostPinUnpinnedDiv);
    const uint64_t cap_abs     = total > reserve_abs ? total - reserve_abs : 0;

    const uint64_t cap = std::min(cap_dyn, cap_abs);
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "MemTotal %.2f GB, MemAvailable %.2f GB (page cache %.2f GB of it); "
                  "available-%.2f = %.3f GB, total-%.2f = %.3f GB -> cap %.3f GB (%s bound)",
                  double(total) / 1e9, double(avail) / 1e9, double(cached) / 1e9,
                  double(reserve_dyn) / 1e9, double(cap_dyn) / 1e9,
                  double(reserve_abs) / 1e9, double(cap_abs) / 1e9,
                  double(cap) / 1e9, cap_dyn <= cap_abs ? "available" : "total");
    why = buf;
    return cap;
}

// ---------------------------------------------------------------------------
// residency priority as data — the profile file
// ---------------------------------------------------------------------------
namespace {

// A permutation check shared by the reader and by anything that supplies an
// order.  Names the offending id rather than repairing the vector.
std::string check_permutation(const std::vector<uint32_t>& v, uint32_t n,
                              const std::string& what) {
    if (v.size() != n)
        return what + " has " + std::to_string(v.size()) +
               " entries but the model has " + std::to_string(n) +
               " experts — it must be a full permutation, not a prefix";
    std::vector<bool> seen(n, false);
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] >= n)
            return what + "[" + std::to_string(i) + "] = " +
                   std::to_string(v[i]) + " is outside [0," + std::to_string(n) + ")";
        if (seen[v[i]])
            return what + " lists expert " + std::to_string(v[i]) +
                   " twice (at position " + std::to_string(i) + ") — it must be a permutation";
        seen[v[i]] = true;
    }
    return {};
}

}  // namespace

std::string ds4_expert_priority_write_layers(const std::string& path,
                                             const std::vector<std::vector<uint32_t>>& orders,
                                             const std::vector<uint64_t>& counts,
                                             const std::vector<std::string>& notes) {
    if (orders.empty()) return "ds4_expert_priority_write_layers: no layers";
    const uint32_t ne = uint32_t(orders[0].size());
    for (size_t l = 0; l < orders.size(); ++l)
        if (const std::string e = check_permutation(orders[l], ne,
                                                    "layer " + std::to_string(l) + "'s order");
            !e.empty())
            return e;
    if (!counts.empty() && counts.size() != ne)
        return "ds4_expert_priority_write_layers: " + std::to_string(counts.size()) +
               " counts for " + std::to_string(ne) + " experts";
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return "ds4_expert_priority_write_layers: cannot create " + path;
    std::fprintf(f, "# ds4-expert-priority 1\n");
    for (const std::string& n : notes) std::fprintf(f, "# %s\n", n.c_str());
    std::fprintf(f, "experts %u\nlayers %zu\n", ne, orders.size());
    std::fprintf(f, "# per-layer residency order, most-selected first WITHIN EACH LAYER\n");
    for (size_t l = 0; l < orders.size(); ++l) {
        std::fprintf(f, "layer %zu\n", l);
        for (uint32_t e : orders[l]) std::fprintf(f, "%u\n", e);
    }
    const bool bad = std::ferror(f) != 0;
    if (std::fclose(f) != 0 || bad)
        return "ds4_expert_priority_write_layers: write to " + path + " failed";
    return {};
}

std::string ds4_expert_priority_read_layers(const std::string& path, uint32_t n_experts,
                                            uint32_t n_layers,
                                            std::vector<std::vector<uint32_t>>& out) {
    out.clear();
    if (n_experts == 0) return "ds4_expert_priority_read_layers: n_experts == 0";
    if (n_layers == 0)  return "ds4_expert_priority_read_layers: n_layers == 0";

    // Does this file carry per-layer blocks at all?  Asked by reading the
    // header, not by guessing from the line count, so a truncated per-layer file
    // is an error rather than a plausible-looking single permutation.
    bool per_layer = false;
    {
        std::FILE* f = std::fopen(path.c_str(), "r");
        if (!f) return "ds4_expert_priority_read_layers: cannot open " + path;
        char line[4096];
        while (std::fgets(line, sizeof(line), f)) {
            const char* p = line;
            while (*p == ' ' || *p == '\t') ++p;
            if (*p == '#') continue;
            unsigned long long v = 0;
            if (std::sscanf(p, "layers %llu", &v) == 1) { per_layer = true; break; }
        }
        std::fclose(f);
    }
    if (!per_layer) {
        // The existing single-permutation file, replicated.  Delegated so there
        // is exactly ONE implementation of "is this ranking valid for this
        // model", with exactly one set of error messages.
        std::vector<uint32_t> one;
        if (std::string e = ds4_expert_priority_read(path, n_experts, one); !e.empty()) return e;
        out.assign(n_layers, std::move(one));
        return {};
    }

    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return "ds4_expert_priority_read_layers: cannot open " + path;
    char        line[4096];
    uint64_t    lineno   = 0;
    bool        have_e   = false, have_l = false;
    int64_t     cur      = -1;      // index of the block being filled
    std::string err;
    while (std::fgets(line, sizeof(line), f)) {
        ++lineno;
        const char* p = line;
        while (*p == ' ' || *p == '\t') ++p;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue;
        unsigned long long v = 0;
        if (std::sscanf(p, "experts %llu", &v) == 1) {
            if (have_e) { err = path + ":" + std::to_string(lineno) + ": a second `experts` header"; break; }
            have_e = true;
            if (uint32_t(v) != n_experts) {
                err = path + ": ranking is for " + std::to_string(v) + " experts but this model has " +
                      std::to_string(n_experts) + " — refusing to use a ranking built for a"
                      " different model";
                break;
            }
            continue;
        }
        if (std::sscanf(p, "layers %llu", &v) == 1) {
            if (have_l) { err = path + ":" + std::to_string(lineno) + ": a second `layers` header"; break; }
            have_l = true;
            if (uint32_t(v) != n_layers) {
                err = path + ": ranking is for " + std::to_string(v) + " layers but this model has " +
                      std::to_string(n_layers) + " — refusing to use a ranking built for a"
                      " different model";
                break;
            }
            out.assign(n_layers, {});
            continue;
        }
        if (std::sscanf(p, "layer %llu", &v) == 1) {
            if (!have_e || !have_l) {
                err = path + ":" + std::to_string(lineno) +
                      ": a `layer` block before the `experts`/`layers` headers";
                break;
            }
            if (uint32_t(v) >= n_layers) {
                err = path + ":" + std::to_string(lineno) + ": layer " + std::to_string(v) +
                      " is outside [0," + std::to_string(n_layers) + ")";
                break;
            }
            if (!out[v].empty()) {
                err = path + ":" + std::to_string(lineno) + ": layer " + std::to_string(v) +
                      " appears twice";
                break;
            }
            cur = int64_t(v);
            continue;
        }
        if (cur < 0) {
            err = path + ":" + std::to_string(lineno) +
                  ": expert ids before any `layer <i>` marker";
            break;
        }
        char* end = nullptr;
        const unsigned long id = std::strtoul(p, &end, 10);
        if (end == p) {
            err = path + ":" + std::to_string(lineno) + ": expected an expert id, got \"" +
                  std::string(p).substr(0, 32) + "\"";
            break;
        }
        std::vector<uint32_t>& blk = out[size_t(cur)];
        if (blk.size() == n_experts) {
            err = path + ":" + std::to_string(lineno) + ": layer " + std::to_string(cur) +
                  " has more than " + std::to_string(n_experts) + " expert ids";
            break;
        }
        blk.push_back(uint32_t(id));
    }
    std::fclose(f);
    if (err.empty() && !have_e) err = path + ": no `experts <n>` header";
    if (err.empty())
        for (uint32_t l = 0; l < n_layers && err.empty(); ++l)
            err = check_permutation(out[l], n_experts,
                                    "layer " + std::to_string(l) + "'s ranking in " + path);
    if (!err.empty()) { out.clear(); return err; }
    return {};
}

std::string ds4_expert_priority_read(const std::string& path, uint32_t n_experts,
                                     std::vector<uint32_t>& out) {
    out.clear();
    if (n_experts == 0) return "ds4_expert_priority_read: n_experts == 0";
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return "ds4_expert_priority_read: cannot open " + path;

    char     line[4096];
    uint32_t declared = 0;
    bool     have_hdr = false;
    uint64_t lineno   = 0;
    std::string err;
    while (std::fgets(line, sizeof(line), f)) {
        ++lineno;
        const char* p = line;
        while (*p == ' ' || *p == '\t') ++p;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue;
        unsigned long long v = 0;
        if (std::sscanf(p, "experts %llu", &v) == 1) {
            if (have_hdr) {
                err = path + ":" + std::to_string(lineno) +
                      ": a second `experts` header — the file declares its expert count once";
                break;
            }
            have_hdr = true;
            declared = uint32_t(v);
            if (declared != n_experts) {
                err = path + ": ranking is for " + std::to_string(declared) +
                      " experts but this model has " + std::to_string(n_experts) +
                      " — refusing to use a ranking built for a different model";
                break;
            }
            continue;
        }
        if (!have_hdr) {
            err = path + ":" + std::to_string(lineno) +
                  ": expert ids before the `experts <n>` header — the count must be declared first";
            break;
        }
        char* end = nullptr;
        const unsigned long id = std::strtoul(p, &end, 10);
        if (end == p) {
            err = path + ":" + std::to_string(lineno) +
                  ": expected an expert id, got \"" + std::string(p).substr(0, 32) + "\"";
            break;
        }
        if (out.size() == n_experts) {
            err = path + ":" + std::to_string(lineno) + ": more than " +
                  std::to_string(n_experts) + " expert ids";
            break;
        }
        out.push_back(uint32_t(id));
    }
    std::fclose(f);
    if (!err.empty()) { out.clear(); return err; }
    if (!have_hdr) { out.clear(); return path + ": no `experts <n>` header"; }
    if (const std::string e = check_permutation(out, n_experts, "the ranking in " + path);
        !e.empty()) {
        out.clear();
        return e;
    }
    return {};
}

std::string ds4_expert_priority_write(const std::string& path,
                                      const std::vector<uint32_t>& order,
                                      const std::vector<uint64_t>& counts,
                                      const std::vector<std::string>& notes) {
    if (order.empty()) return "ds4_expert_priority_write: empty order";
    if (const std::string e =
            check_permutation(order, uint32_t(order.size()), "the order to write");
        !e.empty())
        return e;
    if (!counts.empty() && counts.size() != order.size())
        return "ds4_expert_priority_write: " + std::to_string(counts.size()) +
               " counts for " + std::to_string(order.size()) + " experts";
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return "ds4_expert_priority_write: cannot create " + path;
    std::fprintf(f, "# ds4-expert-priority 1\n");
    for (const std::string& n : notes) std::fprintf(f, "# %s\n", n.c_str());
    std::fprintf(f, "experts %zu\n", order.size());
    std::fprintf(f, "# <expert id>%s   most-selected first\n",
                 counts.empty() ? "" : "  <total selections>");
    for (uint32_t e : order) {
        if (counts.empty()) std::fprintf(f, "%u\n", e);
        else std::fprintf(f, "%u %llu\n", e, (unsigned long long)counts[e]);
    }
    const bool bad = std::ferror(f) != 0;
    if (std::fclose(f) != 0 || bad)
        return "ds4_expert_priority_write: write to " + path + " failed";
    return {};
}

// ---------------------------------------------------------------------------
// Ds4ExpertProfile
// ---------------------------------------------------------------------------
std::string Ds4ExpertProfile::init(uint32_t n_layers, uint32_t n_experts) {
    c_.clear();
    n_layers_ = n_experts_ = 0;
    sel_ = rej_ = 0;
    if (n_layers == 0)  return "Ds4ExpertProfile: n_layers == 0";
    if (n_experts == 0) return "Ds4ExpertProfile: n_experts == 0";
    c_.assign(size_t(n_layers) * n_experts, 0);
    d_.assign(size_t(n_layers) * n_experts, 0);
    n_layers_  = n_layers;
    n_experts_ = n_experts;
    return {};
}

void Ds4ExpertProfile::reset() noexcept {
    std::fill(c_.begin(), c_.end(), 0ull);
    std::fill(d_.begin(), d_.end(), 0ull);
    sel_ = rej_ = dsel_ = 0;
}

void Ds4ExpertProfile::record(uint32_t L, const int32_t* ids, uint32_t n, bool decode) noexcept {
    if (!active() || L >= n_layers_ || !ids) return;
    uint64_t* row = c_.data() + size_t(L) * n_experts_;
    uint64_t* drow = d_.data() + size_t(L) * n_experts_;
    for (uint32_t i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if (e < 0 || uint32_t(e) >= n_experts_) { ++rej_; continue; }
        ++row[uint32_t(e)];
        ++sel_;
        if (decode) { ++drow[uint32_t(e)]; ++dsel_; }
    }
}

uint64_t Ds4ExpertProfile::count(uint32_t L, uint32_t e) const noexcept {
    if (L >= n_layers_ || e >= n_experts_) return 0;
    return c_[size_t(L) * n_experts_ + e];
}

uint64_t Ds4ExpertProfile::count_decode(uint32_t L, uint32_t e) const noexcept {
    if (L >= n_layers_ || e >= n_experts_) return 0;
    return d_[size_t(L) * n_experts_ + e];
}

uint64_t Ds4ExpertProfile::total(uint32_t e) const noexcept {
    if (e >= n_experts_) return 0;
    uint64_t t = 0;
    for (uint32_t L = 0; L < n_layers_; ++L) t += c_[size_t(L) * n_experts_ + e];
    return t;
}

namespace {
// Descending count, ascending id on a tie — so an all-zero profile is the
// identity, not an arbitrary shuffle.  std::sort is not stable, hence the
// explicit id comparison rather than relying on the initial ordering.
std::vector<uint32_t> rank_by(const std::vector<uint64_t>& cnt) {
    std::vector<uint32_t> order(cnt.size());
    for (uint32_t e = 0; e < cnt.size(); ++e) order[e] = e;
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return cnt[a] != cnt[b] ? cnt[a] > cnt[b] : a < b;
    });
    return order;
}
}  // namespace

std::vector<uint32_t> Ds4ExpertProfile::priority() const {
    // DECODE counts when the run saw decode; blended only when it did not.  See
    // the header: a blended ranking measured WORSE than index order on decode.
    const std::vector<uint64_t>& src = dsel_ ? d_ : c_;
    std::vector<uint64_t> tot(n_experts_, 0);
    for (uint32_t L = 0; L < n_layers_; ++L)
        for (uint32_t e = 0; e < n_experts_; ++e) tot[e] += src[size_t(L) * n_experts_ + e];
    return rank_by(tot);
}

std::vector<std::vector<uint32_t>> Ds4ExpertProfile::priority_layers() const {
    const std::vector<uint64_t>& src = dsel_ ? d_ : c_;
    std::vector<std::vector<uint32_t>> out(n_layers_);
    for (uint32_t L = 0; L < n_layers_; ++L) {
        std::vector<uint64_t> row(src.begin() + size_t(L) * n_experts_,
                                  src.begin() + size_t(L + 1) * n_experts_);
        out[L] = rank_by(row);
    }
    return out;
}

std::string Ds4ExpertProfile::write(const std::string& path) const {
    if (!active()) return "Ds4ExpertProfile::write: profile was never initialised";
    std::vector<uint64_t> tot(n_experts_);
    for (uint32_t e = 0; e < n_experts_; ++e) tot[e] = total(e);
    std::vector<std::string> notes;
    {
        char buf[320];
        std::snprintf(buf, sizeof(buf),
                      "measured by ie::Ds4ExpertProfile: %llu selections over %u layers x %u"
                      " experts (%llu ids rejected as out of range); %llu of them at DECODE,"
                      " and the ranking below is built from THOSE",
                      (unsigned long long)sel_, n_layers_, n_experts_,
                      (unsigned long long)rej_, (unsigned long long)dsel_);
        notes.emplace_back(buf);
        if (!dsel_)
            notes.emplace_back("NO DECODE SELECTIONS WERE SEEN — this ranking is built from"
                               " PREFILL counts, and a prefill ranking measured WORSE than index"
                               " order on decode (31.16% vs 31.47% static hit).  Capture a run"
                               " that actually decodes before using this for residency.");
    }
    // The per-layer detail, as comments: informative to a human, ignored by the
    // reader, and NOT lost at the point of capture.  Zero counts are omitted so
    // a sparse profile stays readable.
    for (uint32_t L = 0; L < n_layers_; ++L) {
        std::string s = "layer " + std::to_string(L) + ":";
        bool any = false;
        for (uint32_t e = 0; e < n_experts_; ++e) {
            const uint64_t v = c_[size_t(L) * n_experts_ + e];
            if (!v) continue;
            s += " " + std::to_string(e) + ":" + std::to_string(v);
            any = true;
        }
        notes.push_back(any ? s : s + " (no selections)");
    }
    // The GLOBAL order is kept as a comment and the PER-LAYER one is what the
    // file actually carries, because they are worth very different amounts:
    // measured cross-stream on the real model, one global order buys +5.5 points
    // of static hit over index order while a per-layer order buys +35.0.
    {
        const std::vector<uint32_t> g = priority();
        std::string s = "global order (informational; the blocks below are what is READ):";
        for (size_t i = 0; i < g.size() && i < 16; ++i) s += " " + std::to_string(g[i]);
        notes.push_back(s + " ...");
    }
    return ds4_expert_priority_write_layers(path, priority_layers(), tot, notes);
}

// ---------------------------------------------------------------------------
// residency plan
// ---------------------------------------------------------------------------
std::string ds4_plan_residency(const std::vector<uint64_t>& layer_slot_bytes,
                               uint32_t n_experts, uint64_t vram_budget_bytes,
                               uint32_t slots_cap, uint64_t host_pin_cap_bytes,
                               uint32_t min_stream_slots, Ds4ResidencyPlan& out) {
    out = Ds4ResidencyPlan{};
    if (layer_slot_bytes.empty()) return "ds4_plan_residency: no layers";
    if (n_experts == 0)           return "ds4_plan_residency: n_experts == 0";
    if (min_stream_slots == 0)    return "ds4_plan_residency: min_stream_slots == 0";

    // Σ over layers of ONE slot each, read from the per-layer slot sizes the
    // caller derived from each tensor's own dtype.  Q3 = 42*10,878,976 +
    // 13,369,344 (blk.26) = 470,286,336 B; Q8 = 43*13,369,344 = 574,881,792 B.
    uint64_t per_layer = 0;
    for (uint32_t l = 0; l < layer_slot_bytes.size(); ++l) {
        if (layer_slot_bytes[l] == 0)
            return "ds4_plan_residency: layer " + std::to_string(l) + " has a zero-byte expert slot";
        per_layer += layer_slot_bytes[l];
    }

    uint32_t slots = slots_cap ? std::min(slots_cap, n_experts) : n_experts;
    if (vram_budget_bytes)
        slots = uint32_t(std::min<uint64_t>(slots, vram_budget_bytes / per_layer));
    if (slots == 0)
        return "ds4_plan_residency: VRAM budget " + bytes_str(vram_budget_bytes) +
               " cannot hold even one slot per layer (" + bytes_str(per_layer) + ")";

    // Maximise the static partition: every static slot is one expert per layer
    // whose host copy is never allocated, and doc 33 §2.2/§3.1 measured that it
    // costs nothing in hit rate to move a slot from streaming to static.
    out.stream_slots     = std::min(slots, min_stream_slots);
    out.static_slots     = slots - out.stream_slots;
    out.slots_per_layer  = slots;
    out.pinned_experts   = n_experts - out.static_slots;
    out.layer_slot_total = per_layer;
    out.vram_bytes       = uint64_t(slots) * per_layer;
    out.host_bytes       = uint64_t(out.pinned_experts) * per_layer;

    if (out.host_bytes > host_pin_cap_bytes) {
        const uint64_t affordable = host_pin_cap_bytes / per_layer;      // experts/layer
        const uint64_t want_slots = (affordable >= n_experts ? 0 : uint64_t(n_experts) - affordable) +
                                    min_stream_slots;
        return "ds4_plan_residency: the routed experts that are NOT permanently VRAM-resident need " +
               bytes_str(out.host_bytes) + " of pinned host RAM (" +
               std::to_string(out.pinned_experts) + " of " + std::to_string(n_experts) +
               " experts per layer; only " + std::to_string(out.static_slots) +
               " are held permanently in the " + std::to_string(slots) +
               "-slot VRAM arena), above the " + bytes_str(host_pin_cap_bytes) +
               " pinned-host cap.  " + std::to_string(want_slots) +
               " VRAM slots per layer would bring it under the cap.  Refusing to load: "
               "pinning past the cap is how systemd-oomd killed 147 processes on this box.";
    }
    return {};
}

std::string ds4_plan_residency_tp(const std::vector<uint64_t>& card_slot_bytes,
                                  const std::vector<uint64_t>& whole_slot_bytes,
                                  uint32_t n_experts, uint32_t n_cards,
                                  uint64_t vram_budget_per_card, uint32_t slots_cap,
                                  uint64_t host_pin_cap_total, uint32_t min_stream_slots,
                                  Ds4TpResidencyPlan& out) {
    out = Ds4TpResidencyPlan{};
    if (n_cards == 0) return "ds4_plan_residency_tp: n_cards == 0";
    if (card_slot_bytes.size() != whole_slot_bytes.size())
        return "ds4_plan_residency_tp: " + std::to_string(card_slot_bytes.size()) +
               " per-card layers vs " + std::to_string(whole_slot_bytes.size()) + " whole layers";
    // THE INVARIANT, checked before anything is planned: the cards' slices must
    // TILE the expert.  A slice that is a byte short (padding that does not
    // halve, a dtype whose planes round up) would leave weights in no card's
    // arena, and the symptom would be a wrong logit hours into a load.
    for (size_t l = 0; l < card_slot_bytes.size(); ++l) {
        if (card_slot_bytes[l] == 0)
            return "ds4_plan_residency_tp: layer " + std::to_string(l) +
                   " has a zero-byte per-card expert slice";
        if (card_slot_bytes[l] * n_cards != whole_slot_bytes[l])
            return "ds4_plan_residency_tp: layer " + std::to_string(l) + ": " +
                   std::to_string(n_cards) + " x " + bytes_str(card_slot_bytes[l]) +
                   " per card != " + bytes_str(whole_slot_bytes[l]) +
                   " for the whole expert — the split loses or duplicates bytes";
        out.card_slot_total  += card_slot_bytes[l];
        out.whole_slot_total += whole_slot_bytes[l];
    }
    // The pinned-host cap is one pool of RAM shared by every card's arena, so it
    // is divided here.  Applying it per card would authorise n_cards times the
    // pin that was actually approved.
    const std::string e = ds4_plan_residency(card_slot_bytes, n_experts, vram_budget_per_card,
                                             slots_cap, host_pin_cap_total / n_cards,
                                             min_stream_slots, out.card);
    out.n_cards          = n_cards;
    out.vram_bytes_total = out.card.vram_bytes * n_cards;
    out.host_bytes_total = out.card.host_bytes * n_cards;
    out.pool_bytes       = uint64_t(n_experts) * out.whole_slot_total;
    return e;
}

// ---------------------------------------------------------------------------
// cross-card reduction, through host
// ---------------------------------------------------------------------------
// The size gate shared by ds4_tp_reduce_host (memcpy hops from here up) and
// Ds4TpReducer (the sliced schedule from here up).
constexpr uint64_t kDs4CopyEngineMinN = uint64_t(1) << 16;

std::string ds4_tp_reduce_host(const std::vector<sycl::queue*>& qs,
                               const std::vector<float*>& parts,
                               const std::vector<float*>& stages, uint64_t n) {
    if (qs.empty())              return "ds4_tp_reduce_host: no queues";
    if (qs.size() != parts.size())
        return "ds4_tp_reduce_host: " + std::to_string(qs.size()) + " queues vs " +
               std::to_string(parts.size()) + " partials";
    if (qs.size() != stages.size())
        return "ds4_tp_reduce_host: " + std::to_string(qs.size()) + " queues vs " +
               std::to_string(stages.size()) + " staging buffers";
    if (n == 0)                  return "ds4_tp_reduce_host: n == 0";
    for (size_t c = 0; c < qs.size(); ++c) {
        if (!qs[c] || !parts[c])
            return "ds4_tp_reduce_host: card " + std::to_string(c) + " has a null queue or partial";
        if (!stages[c])
            return "ds4_tp_reduce_host: card " + std::to_string(c) +
                   " has a null host staging buffer";
    }

    const size_t nb = size_t(n) * sizeof(float);   // host-to-host fan-out below

    // BOTH HOPS ARE KERNELS, NOT `memcpy`, AND THAT IS THE WHOLE POINT.
    //
    // A `q.memcpy` is dispatched to a Level-Zero COPY ENGINE.  So is every
    // host-to-device burst the expert stream issues on the transfer queue, and
    // those bursts are megabytes while these hops are kilobytes.  A 32 KB hop
    // that lands behind a 4 MB expert burst waits for the burst, and because
    // these queues are in-order everything the card enqueues afterwards waits
    // with it.  A `parallel_for` that dereferences the pinned host pointer moves
    // the same bytes over the same PCIe link on the COMPUTE engine, which the
    // expert stream does not use — so it does not queue behind it.
    //
    // MEASURED on this box (2x B70, PCIe Gen5 x8/card).  One reduction, DRAINED
    // inside the timed region so nothing is merely deferred; the transfer queue
    // carries one expert-sized burst per reduction and is drained each time, so
    // the backlog is bounded the way `acquire` bounds it.  THE FOUR FORMS WERE
    // ROTATED THROUGH THE FIRST POSITION so none of them permanently stands in
    // front of a fresh burst — the first version of this measurement did not
    // rotate, and it read a difference that was pure ordering.
    //
    //   n = 8192 (decode: ws_moe_ + the sliced shared expert)
    //     stream    | memcpy/memcpy | memcpy/kernel | kernel/memcpy | BOTH kernels
    //     ----------+---------------+---------------+---------------+-------------
    //      idle     |     33.1 us   |     36.6 us   |     36.9 us   |    39.3 us
    //     20 GB/s   |     48.6 us   |     50.7 us   |     35.6 us   |    36.8 us
    //     41 GB/s   |    196.4 us   |    199.1 us   |    185.5 us   |    45.3 us
    //   n = 4096 (decode: the attention-output reduction)
    //      idle     |     29.0 us   |     31.2 us   |     31.2 us   |    32.6 us
    //     23 GB/s   |     40.7 us   |     41.9 us   |     30.5 us   |    28.5 us
    //     42 GB/s   |    186.9 us   |    192.5 us   |    176.4 us   |    39.0 us
    //
    // CONVERTING ONE LEG BUYS NOTHING — that is what the middle two columns say,
    // and it is worth stating because it is the obvious half-measure.  Whichever
    // hop is left as a `memcpy` is enough to put the reduction back in the copy
    // engine's queue.  The reduction has to leave that engine entirely or not at
    // all.
    //
    // WHAT IT COSTS, STATED RATHER THAN BURIED: on a genuinely IDLE link the
    // kernel form is 3.6-6.2 us slower at decode sizes, and at prefill sizes
    // (n = 2,097,152, where the hop is simply PCIe-bandwidth-bound and no
    // queueing delay is material) about 5% slower: 2611.7 us vs 2737.5 us.
    // (2026-08: accepted unconditionally.  2026-09-02, docs/deepseek4/72 Phase
    // F: at prefill sizes the reduction is a third of a 1024-token chunk, so
    // the hops are size-gated to memcpys from kCopyEngineMinN up — the decode
    // regime this table describes is unchanged below it.)
    //
    // NO VALUE CHANGES.  A `memcpy` of n floats and `dst[i] = src[i]` over n
    // floats are both byte copies and neither does arithmetic; the gate asserts
    // that against the memcpy form bit for bit rather than trusting the
    // argument.  A float4 form measured 8% better at n = 262144 and 0% at the
    // decode sizes that matter, so the scalar form is kept: it needs no
    // divisibility or alignment precondition to state and refuse on.
    // PREFILL SIZES ARE THE OTHER REGIME (docs/deepseek4/72 Phase F, measured
    // 2026-09-02): at n = 2,097,152 (T = 512, 8 MB per card) the kernel hop is
    // PCIe-bandwidth-bound at the compute engine's store rate — the D2H leg
    // alone was ~4.5 ms of a 5.4 ms reduction, 86 times per 512-token chunk —
    // while the copy engine moves the same 8 MB in ~0.35 ms and the worst
    // queueing behind one expert burst (6.7 MB) is ~0.3 ms.  So above
    // kCopyEngineMinN both legs are memcpys; below it (decode) the table above
    // stands and both legs stay kernels.  Either form is a byte copy.
    // CAVEAT measured the same night: a copy-engine hop queues behind whatever
    // else is on the copy engine — a 400 MB cross-layer prestage burst put in
    // front of it cost +355 ms of reductions per chunk (that prestage is gone,
    // falsified on its own account).  Demand expert bursts are 6.7 MB, ~0.3 ms
    // of queueing at worst, which is what the size gate accepts.
    constexpr uint64_t kCopyEngineMinN = kDs4CopyEngineMinN;
    auto hop = [](sycl::queue& q, float* dst, const float* src, uint64_t cnt) {
        if (cnt >= kCopyEngineMinN) {
            q.memcpy(dst, src, size_t(cnt) * sizeof(float));
            return;
        }
        q.parallel_for(sycl::range<1>(size_t(cnt)), [=](sycl::id<1> i) { dst[i] = src[i]; });
    };

    // IE_DS4_REDUCE_TRACE=1: where a reduction's wall goes, printed every 256
    // calls for n >= kCopyEngineMinN (prefill).  `drain` is the wait for work
    // the caller left in flight on the compute queues BEFORE the D2H is even
    // issued — the part that is not the reduction's own cost.
    static const bool trace = std::getenv("IE_DS4_REDUCE_TRACE") != nullptr;
    static double t_drain = 0, t_d2h = 0, t_sum = 0, t_h2d = 0;
    static uint64_t n_trace = 0;
    const bool tr = trace && n >= kCopyEngineMinN;
    auto now = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    double t0 = 0, t1 = 0, t2 = 0, t3 = 0;
    if (tr) { t0 = now(); for (sycl::queue* q : qs) q->wait(); t1 = now(); }

    // D2H: every card into ITS OWN pinned buffer, in its own context.  Issued to
    // every card FIRST and waited for afterwards, so the reads are in flight
    // together rather than one after the other.
    for (size_t c = 0; c < qs.size(); ++c) hop(*qs[c], stages[c], parts[c], n);
    // LOAD-BEARING: the host sum below reads these buffers.  Host USM written by
    // a kernel is coherent for the host once that kernel's event has completed,
    // which is exactly what this wait establishes.
    for (sycl::queue* q : qs) q->wait();
    if (tr) t2 = now();
    // Sum on the host, into card 0's buffer, then hand every other card a copy
    // in the context its queue can actually read — in ONE pass over the rows,
    // the fan-out fused into the add (same adds in the same order as the
    // separate add + memcpy, so the bytes are the same; deepseek4_stream_gate_test
    // §8 holds it to the bit at n = 8192 and n = 131072).
    //
    // MEASURED 2026-09-02 (docs/deepseek4/72 Phase F): fusing the fan-out took
    // a 512-token chunk's 86 reductions from 541 to 467 ms.  The pass itself
    // stays ~1.2 ms at n = 1-4 M whether or not it is split across threads
    // (tried with OpenMP, 8 threads: no change) — it is bound by reading the
    // pinned host buffers, not by the adds — so it is left single-threaded.
    if (qs.size() == 2) {
        float* a0 = stages[0];
        float* a1 = stages[1];
        for (uint64_t i = 0; i < n; ++i) {
            const float v = a0[i] + a1[i];
            a0[i] = v; a1[i] = v;
        }
    } else {
        for (size_t c = 1; c < qs.size(); ++c) {
            const float* src = stages[c];
            for (uint64_t i = 0; i < n; ++i) stages[0][i] += src[i];
        }
        for (size_t c = 1; c < qs.size(); ++c) std::memcpy(stages[c], stages[0], nb);
    }
    // H2D: each card from its own buffer, and NOT waited for — see the contract
    // in the header.  Whatever the caller enqueues on `qs[c]` next already runs
    // after this write, because the queue is in-order; the wait bought a host
    // round trip per card per layer and no ordering at all.  The two things it
    // DID cover are re-established explicitly: the next call's own D2H wait
    // above precedes the next host write to `stages[c]`, and a caller that frees
    // or resizes `stages[c]` must drain `qs[c]` first.
    //
    if (tr) t3 = now();
    for (size_t c = 0; c < qs.size(); ++c) hop(*qs[c], parts[c], stages[c], n);
    if (tr) {
        const double t4 = now();
        t_drain += t1 - t0; t_d2h += t2 - t1; t_sum += t3 - t2; t_h2d += t4 - t3;
        if (++n_trace % 256 == 0)
            std::fprintf(stderr, "[ds4-reduce] n=%llu calls=%llu avg ms: drain %.3f  d2h %.3f  sum %.3f  h2d-issue %.3f\n",
                         (unsigned long long)n, (unsigned long long)n_trace,
                         t_drain / n_trace * 1e3, t_d2h / n_trace * 1e3,
                         t_sum / n_trace * 1e3, t_h2d / n_trace * 1e3);
    }
    return {};
}

std::string ds4_tp_reduce_host(const std::vector<sycl::queue*>& qs,
                               const std::vector<float*>& parts,
                               float* stage, uint64_t n) {
    if (!stage) return "ds4_tp_reduce_host: null host staging buffer";
    if (n == 0) return "ds4_tp_reduce_host: n == 0";
    std::vector<float*> stages(qs.size());
    for (size_t c = 0; c < qs.size(); ++c) stages[c] = stage + c * n;
    return ds4_tp_reduce_host(qs, parts, stages, n);
}

// ---------------------------------------------------------------------------
// Ds4TpReducer
// ---------------------------------------------------------------------------

namespace {
inline void ds4_cpu_pause() noexcept {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::this_thread::yield();
#endif
}
}  // namespace

Ds4TpReducer::~Ds4TpReducer() { reset(); }

void Ds4TpReducer::reset() {
    stop_.store(true, std::memory_order_release);
    { std::lock_guard<std::mutex> lk(m_); gen_.fetch_add(1, std::memory_order_release); }
    cv_.notify_all();
    for (std::thread& t : threads_) if (t.joinable()) t.join();
    threads_.clear();
    // Nothing of ours may still be in flight on the auxiliary queues.  Callers
    // include a noexcept release(): a lost device must not throw out of here.
    for (sycl::queue& q : aux_) {
        try { q.wait(); } catch (const sycl::exception& e) {
            std::fprintf(stderr, "[ds4-reduce] aux queue drain failed on reset: %s\n", e.what());
        }
    }
    aux_.clear();
    // Back to the constructed state: a later init() must not hand its fresh
    // workers a stale generation or the last job's (freed) buffers (gate I/J v2
    // finding 2 — workers also start from the live generation, see worker()).
    stop_.store(false, std::memory_order_release);
    gen_.store(0, std::memory_order_release);
    done_.store(0, std::memory_order_relaxed);
    job_a0_ = job_a1_ = nullptr;
    job_n_  = 0;
    n_threads_ = 0;
    sliced_    = false;
}

std::string Ds4TpReducer::init(const std::vector<sycl::queue*>& qs, uint32_t n_threads) {
    if (!threads_.empty() || !aux_.empty()) return "Ds4TpReducer: init called twice";
    static const bool on = [] {
        const char* v = std::getenv("IE_DS4_REDUCE_SLICED");
        return !(v && *v && std::atoi(v) == 0);
    }();
    sliced_ = on && qs.size() == 2;
    if (!sliced_) return {};
    for (sycl::queue* q : qs) {
        if (!q) return "Ds4TpReducer: null queue";
        aux_.emplace_back(q->get_context(), q->get_device(),
                          sycl::property_list{sycl::property::queue::in_order()});
    }
    if (const char* t = std::getenv("IE_DS4_REDUCE_THREADS"); t && *t)   // bisection knob
        n_threads = uint32_t(std::max(1, std::atoi(t)));
    n_threads_ = std::max(1u, n_threads);
    threads_.reserve(n_threads_);
    for (uint32_t t = 0; t < n_threads_; ++t) threads_.emplace_back([this, t] { worker(t); });
    return {};
}

// Each worker spins briefly on the generation counter (the slices of one
// reduction arrive ~0.3 ms apart, so it stays hot through a reduction) and
// then sleeps on the condition variable (the reductions are tens of ms
// apart, so it burns nothing between layers).
void Ds4TpReducer::worker(uint32_t tid) {
    uint64_t seen = gen_.load(std::memory_order_acquire);   // the live generation, never a phantom job
    for (;;) {
        uint64_t g = gen_.load(std::memory_order_acquire);
        for (int spin = 0; spin < 25000 && g == seen && !stop_.load(std::memory_order_relaxed); ++spin) {
            ds4_cpu_pause();
            g = gen_.load(std::memory_order_acquire);
        }
        if (g == seen) {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [&] {
                return stop_.load(std::memory_order_acquire) ||
                       gen_.load(std::memory_order_acquire) != seen;
            });
            g = gen_.load(std::memory_order_acquire);
        }
        if (stop_.load(std::memory_order_acquire)) return;
        seen = g;
        const uint64_t per = (job_n_ + n_threads_ - 1) / n_threads_;
        const uint64_t b = std::min<uint64_t>(job_n_, uint64_t(tid) * per);
        const uint64_t e = std::min<uint64_t>(job_n_, b + per);
        float* a0 = job_a0_;
        float* a1 = job_a1_;
        for (uint64_t i = b; i < e; ++i) {   // the host form's adds, same order per element
            const float v = a0[i] + a1[i];
            a0[i] = v; a1[i] = v;
        }
        done_.fetch_add(1, std::memory_order_release);
    }
}

void Ds4TpReducer::sum_threaded(float* a0, float* a1, uint64_t n) {
    job_a0_ = a0; job_a1_ = a1; job_n_ = n;
    done_.store(0, std::memory_order_relaxed);
    { std::lock_guard<std::mutex> lk(m_); gen_.fetch_add(1, std::memory_order_release); }
    cv_.notify_all();
    while (done_.load(std::memory_order_acquire) < n_threads_) ds4_cpu_pause();
}

std::string Ds4TpReducer::reduce(const std::vector<sycl::queue*>& qs,
                                 const std::vector<float*>& parts,
                                 const std::vector<float*>& stages, uint64_t n) {
    if (!sliced_ || n < kDs4CopyEngineMinN || qs.size() != 2)
        return ds4_tp_reduce_host(qs, parts, stages, n);
    if (aux_.size() != 2 || threads_.empty()) return "Ds4TpReducer: reduce before init";
    if (parts.size() != 2 || stages.size() != 2)
        return "Ds4TpReducer: 2 queues vs " + std::to_string(parts.size()) + " partials / " +
               std::to_string(stages.size()) + " staging buffers";
    for (size_t c = 0; c < 2; ++c)
        if (!qs[c] || !parts[c] || !stages[c])
            return "Ds4TpReducer: card " + std::to_string(c) +
                   " has a null queue, partial or host staging buffer";

    static const bool trace = std::getenv("IE_DS4_REDUCE_TRACE") != nullptr;
    static double t_drain = 0, t_own = 0;
    static uint64_t n_trace = 0;
    auto now = [] { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); };
    const double t0 = trace ? now() : 0;

    static const uint64_t kSlice = [] {   // 1M floats = 4 MB per card; IE_DS4_REDUCE_SLICE overrides (bisection knob)
        const char* v = std::getenv("IE_DS4_REDUCE_SLICE");
        return (v && *v && std::atoll(v) > 0) ? uint64_t(std::atoll(v)) : (uint64_t(1) << 20);
    }();
    const uint32_t NS = uint32_t((n + kSlice - 1) / kSlice);
    std::vector<sycl::event> d2h[2], h2d[2];
    for (int c = 0; c < 2; ++c) { d2h[c].reserve(NS); h2d[c].reserve(NS); }
    // Every D2H slice issued up front, in order, on the compute queues: the
    // copies land back to back while the host works on the earliest one.
    for (uint32_t k = 0; k < NS; ++k) {
        const uint64_t a = uint64_t(k) * kSlice, len = std::min<uint64_t>(kSlice, n - a);
        for (int c = 0; c < 2; ++c)
            d2h[c].push_back(qs[c]->memcpy(stages[c] + a, parts[c] + a, size_t(len) * sizeof(float)));
    }
    double t1 = 0;
    for (uint32_t k = 0; k < NS; ++k) {
        const uint64_t a = uint64_t(k) * kSlice, len = std::min<uint64_t>(kSlice, n - a);
        // LOAD-BEARING, as in the host form: host USM written by a copy is
        // coherent for the host once that copy's event has completed.
        d2h[0][k].wait();
        d2h[1][k].wait();
        if (trace && k == 0) t1 = now();
        sum_threaded(stages[0] + a, stages[1] + a, len);
        for (int c = 0; c < 2; ++c)
            h2d[c].push_back(aux_[c].memcpy(parts[c] + a, stages[c] + a, size_t(len) * sizeof(float)));
    }
    // The compute queue orders everything the caller enqueues next after the
    // last H2D of this card — the same guarantee the in-order H2D gave.
    for (int c = 0; c < 2; ++c) qs[c]->ext_oneapi_submit_barrier(h2d[c]);
    if (trace) {
        const double t2 = now();
        t_drain += t1 - t0; t_own += t2 - t1;
        if (++n_trace % 256 == 0)
            std::fprintf(stderr, "[ds4-reduce-sliced] n=%llu calls=%llu avg ms: drain(to slice 0) %.3f  own %.3f  (%u slices)\n",
                         (unsigned long long)n, (unsigned long long)n_trace,
                         t_drain / n_trace * 1e3, t_own / n_trace * 1e3, NS);
    }
    return {};
}

// ---------------------------------------------------------------------------
// Ds4HostArena
// ---------------------------------------------------------------------------
std::string Ds4HostArena::init(sycl::queue& q, const std::vector<uint64_t>& layer_slot_bytes,
                               uint32_t n_experts, uint64_t segment_target) {
    return init_range(q, layer_slot_bytes, n_experts, 0, n_experts, segment_target);
}

std::string Ds4HostArena::init_range(sycl::queue& q, const std::vector<uint64_t>& layer_slot_bytes,
                                     uint32_t n_experts, uint32_t first_expert, uint32_t n_pinned,
                                     uint64_t segment_target) {
    if (uint64_t(first_expert) + n_pinned > n_experts) {
        free_storage();
        return "Ds4HostArena: pinned range [" + std::to_string(first_expert) + "," +
               std::to_string(uint64_t(first_expert) + n_pinned) + ") is outside the " +
               std::to_string(n_experts) + " experts of this model";
    }
    std::vector<uint32_t> ids(n_pinned);
    for (uint32_t i = 0; i < n_pinned; ++i) ids[i] = first_expert + i;
    return init_set(q, layer_slot_bytes, n_experts, ids, segment_target);
}

std::string Ds4HostArena::init_set(sycl::queue& q, const std::vector<uint64_t>& layer_slot_bytes,
                                   uint32_t n_experts, const std::vector<uint32_t>& pinned_ids,
                                   uint64_t segment_target) {
    // The all-layers-alike case, expressed as the general one so there is a
    // single layout implementation rather than two that can drift apart.
    return init_set_per_layer(q, layer_slot_bytes, n_experts,
                              std::vector<std::vector<uint32_t>>(
                                  layer_slot_bytes.empty() ? 1 : layer_slot_bytes.size(),
                                  pinned_ids),
                              segment_target);
}

std::string Ds4HostArena::init_set_per_layer(
    sycl::queue& q, const std::vector<uint64_t>& layer_slot_bytes, uint32_t n_experts,
    const std::vector<std::vector<uint32_t>>& pinned_ids_per_layer, uint64_t segment_target) {
    free_storage();
    if (layer_slot_bytes.empty()) return "Ds4HostArena: no layers";
    if (n_experts == 0)           return "Ds4HostArena: n_experts == 0";
    if (pinned_ids_per_layer.size() != layer_slot_bytes.size())
        return "Ds4HostArena: " + std::to_string(pinned_ids_per_layer.size()) +
               " pinned sets for " + std::to_string(layer_slot_bytes.size()) + " layers";
    const size_t np = pinned_ids_per_layer[0].size();
    if (np > n_experts)
        return "Ds4HostArena: " + std::to_string(np) +
               " experts to pin but the model has only " + std::to_string(n_experts);
    if (segment_target == 0 || segment_target > kDs4MaxAllocBytes)
        return "Ds4HostArena: segment target " + bytes_str(segment_target) +
               " exceeds the verified max_mem_alloc_size ceiling " + bytes_str(kDs4MaxAllocBytes);

    // Build [layer][expert] -> slot-index.  A duplicate would alias two experts
    // onto one slot and surface only as a wrong logit, so it is named and
    // refused here.  A layer pinning a DIFFERENT NUMBER than its siblings is
    // refused too: the layout's safety argument is that a layer occupies exactly
    // `n_pinned` contiguous slots, and a ragged set silently breaks it.
    std::vector<int32_t> idx(size_t(layer_slot_bytes.size()) * n_experts, -1);
    for (size_t l = 0; l < pinned_ids_per_layer.size(); ++l) {
        const std::vector<uint32_t>& ids = pinned_ids_per_layer[l];
        if (ids.size() != np)
            return "Ds4HostArena: layer " + std::to_string(l) + " pins " +
                   std::to_string(ids.size()) + " experts but layer 0 pins " +
                   std::to_string(np) + " — every layer must pin the same COUNT";
        int32_t* row = idx.data() + l * n_experts;
        for (size_t i = 0; i < ids.size(); ++i) {
            const uint32_t e = ids[i];
            if (e >= n_experts)
                return "Ds4HostArena: expert id " + std::to_string(e) + " at position " +
                       std::to_string(i) + " of layer " + std::to_string(l) +
                       "'s pinned set is outside [0," + std::to_string(n_experts) + ")";
            if (row[e] >= 0)
                return "Ds4HostArena: expert id " + std::to_string(e) +
                       " appears twice in layer " + std::to_string(l) + "'s pinned set (positions " +
                       std::to_string(row[e]) + " and " + std::to_string(i) +
                       ") — two experts would alias one host slot";
            row[e] = int32_t(i);
        }
    }

    q_            = &q;
    n_experts_    = n_experts;
    pin_index_    = std::move(idx);
    n_pinned_     = uint32_t(np);
    first_expert_ = n_experts;                       // lowest pinned id in LAYER 0, or n_experts
    for (uint32_t e = 0; e < n_experts; ++e)
        if (pin_index_[e] >= 0) { first_expert_ = e; break; }
    if (n_pinned_ == 0) first_expert_ = 0;
    slot_bytes_   = layer_slot_bytes;
    const uint32_t L = uint32_t(slot_bytes_.size());
    layer_seg_.assign(L, 0);
    layer_off_.assign(L, 0);
    if (n_pinned_ == 0) return {};   // everything is statically resident; nothing to pin

    // Pack WHOLE LAYERS into segments.  A layer is n_pinned contiguous slots,
    // so an expert can never straddle a segment boundary by construction.  A
    // single layer larger than the target still gets its own segment (and is
    // rejected only if it exceeds the hard device ceiling).
    std::vector<std::pair<uint32_t, uint32_t>> plan;   // (first_layer, n_layers)
    uint64_t cur = 0;
    uint32_t first = 0;
    for (uint32_t l = 0; l < L; ++l) {
        const uint64_t need = slot_bytes_[l] * n_pinned_;
        if (need > kDs4MaxAllocBytes)
            return "Ds4HostArena: layer " + std::to_string(l) + " needs " + bytes_str(need) +
                   " in one piece, above the " + bytes_str(kDs4MaxAllocBytes) +
                   " per-allocation ceiling";
        if (cur != 0 && cur + need > segment_target) {
            plan.emplace_back(first, l - first);
            first = l; cur = 0;
        }
        cur += need;
    }
    plan.emplace_back(first, L - first);

    for (const auto& [f, n] : plan) {
        Seg s; s.first_layer = f; s.n_layers = n;
        for (uint32_t l = f; l < f + n; ++l) {
            layer_seg_[l] = uint32_t(segs_.size());
            layer_off_[l] = s.bytes;
            s.bytes += slot_bytes_[l] * n_pinned_;
        }
        if (s.bytes > kDs4MaxAllocBytes) {
            free_storage();
            return "Ds4HostArena: segment " + std::to_string(segs_.size()) + " would be " +
                   bytes_str(s.bytes) + ", above the " + bytes_str(kDs4MaxAllocBytes) + " ceiling";
        }
        s.p = static_cast<uint8_t*>(sycl::malloc_host(size_t(s.bytes), q));
        if (!s.p) {
            const uint64_t got = total_bytes_;
            free_storage();
            return "Ds4HostArena: sycl::malloc_host failed for segment " +
                   std::to_string(segs_.size()) + " of " + bytes_str(s.bytes) +
                   " (layers " + std::to_string(f) + ".." + std::to_string(f + n - 1) +
                   "); " + bytes_str(got) + " had already been pinned";
        }
        total_bytes_ += s.bytes;
        segs_.push_back(s);
    }
    return {};
}

void Ds4HostArena::free_storage() noexcept {
    if (q_) for (Seg& s : segs_) if (s.p) sycl::free(s.p, *q_);
    segs_.clear();
    slot_bytes_.clear();
    layer_seg_.clear();
    layer_off_.clear();
    pin_index_.clear();
    total_bytes_  = 0;
    n_experts_    = 0;
    first_expert_ = 0;
    n_pinned_     = 0;
    q_            = nullptr;
}

void* Ds4HostArena::slot(uint32_t L, uint32_t e) const noexcept {
    if (L >= slot_bytes_.size()) return nullptr;
    // Outside the pinned SET there is deliberately no storage: those experts
    // live permanently in VRAM and a host copy of them would be dead bytes.
    // The set need not be contiguous — which expert wins a static VRAM slot is
    // chosen by priority order, not by index (see Ds4Options::expert_priority).
    // ...and the set is now PER LAYER, so the question is "does e have a copy in
    // THIS layer", not "in the model".
    if (e >= n_experts_) return nullptr;
    const int32_t i = pin_index_[size_t(L) * n_experts_ + e];
    if (i < 0) return nullptr;
    const Seg& s = segs_[layer_seg_[L]];
    return s.p + layer_off_[L] + uint64_t(i) * slot_bytes_[L];
}

uint64_t Ds4HostArena::max_segment_bytes() const noexcept {
    uint64_t m = 0;
    for (const Seg& s : segs_) m = std::max(m, s.bytes);
    return m;
}

// ---------------------------------------------------------------------------
// Ds4ExpertCache
// ---------------------------------------------------------------------------
std::string Ds4ExpertCache::init(sycl::queue& compute, const Ds4HostArena& arena,
                                 uint32_t slots_per_layer, uint64_t vram_budget_bytes,
                                 uint32_t static_slots, uint32_t bank_slots) {
    free_storage();
    if (arena.n_layers() == 0) return "Ds4ExpertCache: arena not initialised";
    if (slots_per_layer == 0)  return "Ds4ExpertCache: slots_per_layer == 0";

    arena_     = &arena;
    cq_        = &compute;
    n_layers_  = arena.n_layers();
    n_experts_ = arena.n_experts();

    uint64_t per_slot_total = 0;   // sum over layers of one slot each
    uint64_t widest = 0;
    for (uint32_t l = 0; l < n_layers_; ++l) {
        per_slot_total += arena.slot_bytes(l);
        widest = std::max(widest, arena.slot_bytes(l));
    }
    slots_ = std::min(slots_per_layer, n_experts_);
    bank_slots_  = bank_slots;
    bank_stride_ = widest;
    if (vram_budget_bytes) {
        const uint64_t fit = vram_budget_bytes / per_slot_total;
        slots_ = uint32_t(std::min<uint64_t>(slots_, fit));
        if (slots_ == 0) {
            const std::string e = "Ds4ExpertCache: VRAM budget " + bytes_str(vram_budget_bytes) +
                                  " cannot hold even one slot per layer (" +
                                  bytes_str(per_slot_total) + ")";
            free_storage();
            return e;
        }
    }
    // The static partition must fit inside the arena that was actually sized.
    // A budget-driven reduction of slots_ that swallowed the static slots would
    // silently un-resident experts whose host copy was never allocated, so it is
    // an error rather than a clamp.
    if (static_slots > slots_) {
        const std::string e = "Ds4ExpertCache: " + std::to_string(static_slots) +
                              " static slots requested but only " + std::to_string(slots_) +
                              " slots/layer fit (VRAM budget " + bytes_str(vram_budget_bytes) + ")";
        free_storage();
        return e;
    }
    static_ = static_slots;

    // Per-LAYER device allocation: 43 allocations of ~1 GB rather than one 44 GB
    // block, so the device ceiling is never approached and a partial failure
    // names the layer.
    const uint64_t widest_alloc = widest * slots_;
    if (widest_alloc > kDs4MaxAllocBytes) {
        const std::string e = "Ds4ExpertCache: a layer arena would be " + bytes_str(widest_alloc) +
                              ", above the " + bytes_str(kDs4MaxAllocBytes) + " ceiling";
        free_storage();
        return e;
    }

    dev_.assign(n_layers_, nullptr);
    for (uint32_t l = 0; l < n_layers_; ++l) {
        const uint64_t need = arena.slot_bytes(l) * slots_;
        dev_[l] = static_cast<uint8_t*>(sycl::malloc_device(size_t(need), compute));
        if (!dev_[l]) {
            const std::string e = "Ds4ExpertCache: malloc_device failed for layer " +
                                  std::to_string(l) + " arena of " + bytes_str(need) + " (" +
                                  bytes_str(device_bytes_) + " already allocated)";
            free_storage();
            return e;
        }
        device_bytes_ += need;
    }

    for (int b = 0; b < 2 && bank_slots_; ++b) {
        bank_[b] = static_cast<uint8_t*>(sycl::malloc_device(size_t(bank_stride_) * bank_slots_, compute));
        if (!bank_[b]) {
            // Not fatal: without banks the grouped prefill path is the old one.
            std::fprintf(stderr, "[ds4] prefill fetch banks OFF: malloc_device failed for bank %d of %s"
                                 " — prefill runs the plain grouped path\n",
                         b, bytes_str(bank_stride_ * bank_slots_).c_str());
            for (int k = 0; k < b; ++k) {
                sycl::free(bank_[k], compute);
                device_bytes_ -= bank_stride_ * bank_slots_;
                bank_[k] = nullptr;
            }
            bank_slots_ = 0;
            break;
        }
        device_bytes_ += bank_stride_ * bank_slots_;
    }
    if (bank_slots_)
        std::fprintf(stderr, "[ds4] prefill fetch banks: 2 x %u slots (%s); prefill misses bypass the LRU\n",
                     bank_slots_, bytes_str(uint64_t(2) * bank_slots_ * bank_stride_).c_str());

    slot_of_.assign(uint64_t(n_layers_) * n_experts_, -1);
    tag_of_.assign(uint64_t(n_layers_) * slots_, -1);
    fill_cursor_.assign(n_layers_, static_);
    fifo_.assign(n_layers_, static_);        // replacement never re-enters [0, static_)
    stamp_.assign(uint64_t(n_layers_) * slots_, 0);
    tick_.assign(n_layers_, 0);
    {   // $DS4_EVICT=lru|fifo. LRU is the default since the 2026-09-01 A/B
        // (docs/deepseek4/71 lever 1): tg4096 22.04 vs 20.93 tok/s (+5.3%, n=3
        // each, every LRU run above every FIFO run), decode hit 0.733 vs 0.698,
        // pp512 unchanged (221.9 both). FIFO stays as the A/B arm.
        const char* ev = std::getenv("DS4_EVICT");
        lru_ = !(ev && std::string(ev) == "fifo");
        if (ev) std::fprintf(stderr, "[ds4] expert cache eviction: %s\n", lru_ ? "lru" : "fifo");
    }
    inuse_.assign(uint64_t(n_layers_) * slots_, 0);
    inuse_prev_.assign(uint64_t(n_layers_) * slots_, 0);
    from_spec_.assign(uint64_t(n_layers_) * slots_, 0);
    slot_ev_.assign(uint64_t(n_layers_) * slots_, sycl::event{});
    slot_seq_.assign(uint64_t(n_layers_) * slots_, 0);
    seq_ = 0;

    // The seam: a SECOND in-order queue on the compute queue's own context and
    // device.  Same context ⇒ its events are legal `depends_on` arguments in a
    // compute submission, which is what makes the H2D overlap the GEMVs.
    // `enable_profiling` is OPT-IN, matching src/core/allocator.cpp:41-47 and the
    // compute queue's own gate.  It was unconditional here, and it is not free:
    // measured on this box a kernel submit costs 1.43 us with profiling off and
    // 1.85 us with it on, and `submit` is 70-87% of card-busy time during decode.
    // A DS4 decode token issues thousands of submissions, so a flat +0.42 us each
    // is real. Gating the compute queue alone measured card-busy 1.672 -> 0.974 ms
    // (-42%); the transfer queue carries the expert H2D traffic and needs the same
    // treatment or `dma_busy_ms` keeps paying for instrumentation nobody read.
    //
    // Set IE_QUEUE_PROFILING=1 to restore it — `--kprofile` and the DMA accounting
    // both require it, and the bench refuses gracefully when it is absent rather
    // than throwing out of harvest().
    const bool qprof = std::getenv("IE_QUEUE_PROFILING") != nullptr;
    xq_store_.emplace_back(compute.get_context(), compute.get_device(),
                           qprof ? sycl::property_list{sycl::property::queue::in_order(),
                                                       sycl::property::queue::enable_profiling()}
                                 : sycl::property_list{sycl::property::queue::in_order()});
    xq_ = &xq_store_.back();
    return {};
}

void Ds4ExpertCache::free_storage() noexcept {
    if (cq_) for (uint8_t* p : dev_) if (p) sycl::free(p, *cq_);
    if (cq_) for (uint8_t*& p : bank_) if (p) { sycl::free(p, *cq_); p = nullptr; }
    bank_slots_ = 0; bank_stride_ = 0;
    dev_.clear();
    slot_of_.clear(); tag_of_.clear(); fifo_.clear(); fill_cursor_.clear();
    inuse_.clear(); inuse_prev_.clear(); from_spec_.clear();
    slot_ev_.clear(); slot_seq_.clear(); seq_ = 0;
    pending_.clear();
    xq_ = nullptr;
    xq_store_.clear();
    arena_ = nullptr; cq_ = nullptr;
    n_layers_ = n_experts_ = slots_ = static_ = 0;
    device_bytes_ = 0;
}

std::string Ds4ExpertCache::install_static(uint32_t L, uint32_t e, uint32_t s, const void* src) {
    if (!arena_ || !cq_)   return "Ds4ExpertCache::install_static: cache not initialised";
    if (L >= n_layers_)    return "Ds4ExpertCache::install_static: layer " + std::to_string(L) +
                                  " >= " + std::to_string(n_layers_);
    if (e >= n_experts_)   return "Ds4ExpertCache::install_static: expert " + std::to_string(e) +
                                  " >= " + std::to_string(n_experts_);
    if (s >= static_)      return "Ds4ExpertCache::install_static: slot " + std::to_string(s) +
                                  " is outside the static partition [0," + std::to_string(static_) + ")";
    if (!src)              return "Ds4ExpertCache::install_static: null source";
    const uint64_t sb = arena_->slot_bytes(L);
    cq_->memcpy(dev_[L] + uint64_t(s) * sb, src, size_t(sb)).wait();
    int32_t& t = tag_of(L, s);
    if (t >= 0) slot_of(L, uint32_t(t)) = -1;
    t = int32_t(e);
    slot_of(L, e) = int32_t(s);
    return {};
}

bool Ds4ExpertCache::available(uint32_t L, uint32_t e) const noexcept {
    if (!arena_ || L >= n_layers_ || e >= n_experts_) return false;
    // Resident right now (static or cached), or fetchable from the pinned arena.
    return slot_of_[uint64_t(L) * n_experts_ + e] >= 0 || arena_->slot(L, e) != nullptr;
}

bool Ds4ExpertCache::is_static(uint32_t L, uint32_t e) const noexcept {
    if (!arena_ || L >= n_layers_ || e >= n_experts_) return false;
    const int32_t w = slot_of_[uint64_t(L) * n_experts_ + e];
    return w >= 0 && uint32_t(w) < static_;
}

bool Ds4ExpertCache::is_resident(uint32_t L, uint32_t e) const noexcept {
    if (!arena_ || L >= n_layers_ || e >= n_experts_) return false;
    return slot_of_[uint64_t(L) * n_experts_ + e] >= 0;
}

uint32_t Ds4ExpertCache::pick_victim(uint32_t L, const uint8_t* iu,
                                     const uint8_t* iu_prev) noexcept {
    const uint32_t stream = slots_ - static_;
    if (stream == 0) return kDs4NoSlot;
    if (lru_) {
        // Least recently touched streaming slot outside both live generations.
        // O(stream) per miss (stream <= ~90): negligible next to the 6.7 MB copy.
        const uint64_t* st = stamp_.data() + uint64_t(L) * slots_;
        uint32_t best = kDs4NoSlot;
        uint64_t best_stamp = ~uint64_t(0);
        for (uint32_t s = static_; s < slots_; ++s) {
            if (iu[s] || (iu_prev && iu_prev[s])) continue;
            if (st[s] < best_stamp) { best_stamp = st[s]; best = s; }
        }
        return best;
    }
    uint32_t v = fifo_[L];
    if (v < static_ || v >= slots_) v = static_;
    for (uint32_t tries = 0; tries < stream; ++tries) {
        const uint32_t next = static_ + (v - static_ + 1) % stream;
        if (!iu[v] && !(iu_prev && iu_prev[v])) { fifo_[L] = next; return v; }
        v = next;
    }
    return kDs4NoSlot;   // every streaming slot is claimed by a LIVE group
}

void* Ds4ExpertCache::slot_ptr(uint32_t L, uint32_t slot) noexcept {
    if (L >= n_layers_) return nullptr;
    if (slot >= kDs4BankSlotBase) {
        const uint32_t v = slot - kDs4BankSlotBase;
        if (!bank_slots_ || v >= 2u * bank_slots_) return nullptr;
        return bank_[v / bank_slots_] + uint64_t(v % bank_slots_) * bank_stride_;
    }
    if (slot >= slots_) return nullptr;
    return dev_[L] + uint64_t(slot) * arena_->slot_bytes(L);
}

const void* Ds4ExpertCache::slot_ptr(uint32_t L, uint32_t slot) const noexcept {
    if (L >= n_layers_) return nullptr;
    if (slot >= kDs4BankSlotBase) {
        const uint32_t v = slot - kDs4BankSlotBase;
        if (!bank_slots_ || v >= 2u * bank_slots_) return nullptr;
        return bank_[v / bank_slots_] + uint64_t(v % bank_slots_) * bank_stride_;
    }
    if (slot >= slots_) return nullptr;
    return dev_[L] + uint64_t(slot) * arena_->slot_bytes(L);
}

sycl::event Ds4ExpertCache::acquire_prefill(uint32_t L, const int32_t* ids, uint32_t n,
                                            uint32_t* out_slots, uint32_t bank) {
    sycl::event ev;
    if (L >= n_layers_ || !bank_slots_ || bank > 1) {
        for (uint32_t k = 0; k < n; ++k) out_slots[k] = kDs4NoSlot;
        return ev;
    }
    ++st_.acquires;
    uint64_t best_seq = 0;
    auto observe = [&](uint32_t s) {   // a hit on a slot whose demand/speculative fill is in flight
        const uint64_t q = slot_seq_[uint64_t(L) * slots_ + s];
        if (q > best_seq) { best_seq = q; ev = slot_ev_[uint64_t(L) * slots_ + s]; }
    };
    const uint64_t sb = arena_->slot_bytes(L);
    uint32_t j = 0;
    for (uint32_t k = 0; k < n; ++k) {
        const int32_t e = ids[k];
        if (e < 0 || uint32_t(e) >= n_experts_) { out_slots[k] = kDs4NoSlot; ++st_.unavailable; continue; }
        const int32_t where = slot_of(L, uint32_t(e));
        if (where >= 0) {
            // Scan-resistant: no recency bump, no in-use mark — nothing evicts
            // an LRU slot during a prefill group, its misses go to the banks.
            ++st_.hits;
            if (uint32_t(where) < static_) ++st_.static_hits; else ++st_.stream_hits;
            out_slots[k] = uint32_t(where);
            observe(uint32_t(where));
            continue;
        }
        const void* src = arena_->slot(L, uint32_t(e));
        if (!src) { out_slots[k] = kDs4NoSlot; ++st_.unavailable; continue; }
        // POPULATE, never evict: an EMPTY streaming slot takes the expert as a
        // normal LRU fill (so a fresh cache fills up during the first prefill
        // and later chunks/prefills/decode hit it); only a full cache sends the
        // miss to the bank.  Empty slots exist only until the cache first fills,
        // so the cursor scan is a one-time cost per layer.
        {
            uint32_t s = fill_cursor_[L], found = kDs4NoSlot;
            for (uint32_t tries = 0; tries < slots_ - static_ && s < slots_; ++tries, ++s)
                if (tag_of(L, s) < 0) { found = s; break; }
            fill_cursor_[L] = (found == kDs4NoSlot) ? slots_ : found + 1;
            if (found != kDs4NoSlot) {
                tag_of(L, found) = e;
                slot_of(L, uint32_t(e)) = int32_t(found);
                touch(L, found);
                from_spec_[uint64_t(L) * slots_ + found] = 0;
                const sycl::event fe = xq_->memcpy(dev_[L] + uint64_t(found) * sb, src, size_t(sb));
                slot_ev_[uint64_t(L) * slots_ + found]  = fe;
                slot_seq_[uint64_t(L) * slots_ + found] = ++seq_;
                pending_.push_back(fe);
                ++st_.misses;
                st_.bytes_fetched += sb;
                out_slots[k] = found;
                observe(found);
                continue;
            }
        }
        if (j >= bank_slots_) { out_slots[k] = kDs4NoSlot; ++st_.unavailable; continue; }
        uint8_t* dst = bank_[bank] + uint64_t(j) * bank_stride_;
        const sycl::event fe = xq_->memcpy(dst, src, size_t(sb));
        pending_.push_back(fe);
        ++st_.misses;
        st_.bytes_fetched += sb;
        out_slots[k] = kDs4BankSlotBase + bank * bank_slots_ + j;
        ++j;
        // xq_ is in-order: the last fill's event covers every earlier bank fill.
        best_seq = ++seq_;
        ev = fe;
    }
    return ev;
}

int32_t Ds4ExpertCache::tag(uint32_t L, uint32_t slot) const noexcept {
    if (L >= n_layers_ || slot >= slots_) return -1;
    return tag_of_[uint64_t(L) * slots_ + slot];
}

sycl::event Ds4ExpertCache::acquire(uint32_t L, const int32_t* ids, uint32_t n,
                                    uint32_t* out_slots) {
    return acquire_impl(L, ids, n, out_slots, /*keep_prev=*/false);
}

sycl::event Ds4ExpertCache::acquire_pipelined(uint32_t L, const int32_t* ids, uint32_t n,
                                              uint32_t* out_slots) {
    ++st_.pipelined;
    return acquire_impl(L, ids, n, out_slots, /*keep_prev=*/true);
}

sycl::event Ds4ExpertCache::acquire_impl(uint32_t L, const int32_t* ids, uint32_t n,
                                         uint32_t* out_slots, bool keep_prev) {
    sycl::event ev;
    if (L >= n_layers_) return ev;
    ++st_.acquires;

    // The in-use marks exist so that neither speculate() nor a LATER acquire can
    // evict a slot something still reads.
    //   serial    (keep_prev == false): only this call's claims are live, which
    //             is sound because the caller drains the compute queue between
    //             groups.  Both generations are cleared, so victim choice is
    //             bit-identical to the pre-pipeline behaviour.
    //   pipelined (keep_prev == true) : the PREVIOUS call's claims are still
    //             being read by the compute queue right now, so they roll into
    //             the previous generation and stay protected for this call.
    uint8_t* iu   = inuse_.data()      + uint64_t(L) * slots_;
    uint8_t* prev = inuse_prev_.data() + uint64_t(L) * slots_;
    if (keep_prev) std::memcpy(prev, iu, slots_);
    else           std::memset(prev, 0, slots_);
    std::memset(iu, 0, slots_);

    // The latest-sequence fill event among the slots this call hands out.  See
    // the `slot_ev_` note in the header: a HIT on a slot whose fill is still in
    // flight has to be covered too, or the caller's `depends_on` is a no-op.
    uint64_t best_seq = 0;
    auto     observe  = [&](uint32_t s) {
        const uint64_t q = slot_seq_[uint64_t(L) * slots_ + s];
        if (q > best_seq) { best_seq = q; ev = slot_ev_[uint64_t(L) * slots_ + s]; }
    };

    const uint64_t sb = arena_->slot_bytes(L);
    for (uint32_t k = 0; k < n; ++k) {
        const int32_t e = ids[k];
        if (e < 0 || uint32_t(e) >= n_experts_) {
            out_slots[k] = kDs4NoSlot; ++st_.unavailable; continue;
        }
        int32_t& where = slot_of(L, uint32_t(e));
        if (where >= 0) {
            ++st_.hits;
            // Which partition served it.  The static partition is [0, static_)
            // by construction — see pick_victim, which never enters it.
            if (uint32_t(where) < static_) ++st_.static_hits; else ++st_.stream_hits;
            uint8_t& fs = from_spec_[uint64_t(L) * slots_ + uint32_t(where)];
            if (fs) { ++st_.spec_hits; fs = 0; }
            out_slots[k] = uint32_t(where);
            iu[where] = 1;
            touch(L, uint32_t(where));
            observe(uint32_t(where));
            continue;
        }
        // Miss.  Not resident, so it has to come from the pinned arena — and if
        // it was never pinned there is no honest way to produce its output.
        const void* src = arena_->slot(L, uint32_t(e));
        if (!src) { out_slots[k] = kDs4NoSlot; ++st_.unavailable; continue; }
        const uint32_t v = pick_victim(L, iu, prev);
        if (v == kDs4NoSlot) { out_slots[k] = kDs4NoSlot; ++st_.unavailable; continue; }
        int32_t& t = tag_of(L, v);
        if (t >= 0) slot_of(L, uint32_t(t)) = -1;
        t = e;
        where = int32_t(v);
        out_slots[k] = v;
        iu[v] = 1;
        touch(L, v);
        from_spec_[uint64_t(L) * slots_ + v] = 0;

        const sycl::event fe = xq_->memcpy(dev_[L] + uint64_t(v) * sb, src, size_t(sb));
        slot_ev_[uint64_t(L) * slots_ + v]  = fe;
        slot_seq_[uint64_t(L) * slots_ + v] = ++seq_;
        pending_.push_back(fe);
        ++st_.misses;
        st_.bytes_fetched += sb;
        observe(v);
    }
    return ev;
}

uint32_t Ds4ExpertCache::speculate(uint32_t L, const int32_t* ids, uint32_t n) {
    if (L >= n_layers_ || n == 0) return 0;
    // Idle-time-only: a speculative fetch must never delay a demand fetch.  If
    // anything is still outstanding on the transfer queue, decline.  Under this
    // rule a wrong prediction consumes bandwidth that was provably going to be
    // wasted, so it costs power and nothing else.
    if (!pending_.empty()) {
        const auto st = pending_.back()
                            .get_info<sycl::info::event::command_execution_status>();
        if (st != sycl::info::event_command_status::complete) return 0;
    }
    const uint64_t sb = arena_->slot_bytes(L);
    const uint8_t* iu   = inuse_.data()      + uint64_t(L) * slots_;
    const uint8_t* prev = inuse_prev_.data() + uint64_t(L) * slots_;
    uint32_t issued = 0;
    for (uint32_t k = 0; k < n; ++k) {
        const int32_t e = ids[k];
        if (e < 0 || uint32_t(e) >= n_experts_) continue;
        if (slot_of(L, uint32_t(e)) >= 0) continue;      // already resident
        const void* src = arena_->slot(L, uint32_t(e));
        if (!src) continue;                              // no host copy to speculate from
        // BOTH live generations are off limits.  Under the pipelined shape the
        // previous group's slots are being read by the compute queue right now,
        // and a speculative fill into one of them is the same silent
        // wrong-weights corruption a demand fetch into one would be.
        const uint32_t v = pick_victim(L, iu, prev);
        if (v == kDs4NoSlot) break;                       // every streaming slot is in use
        int32_t& t = tag_of(L, v);
        if (t >= 0) slot_of(L, uint32_t(t)) = -1;
        t = e;
        slot_of(L, uint32_t(e)) = int32_t(v);
        touch(L, v);
        from_spec_[uint64_t(L) * slots_ + v] = 1;
        const sycl::event fe = xq_->memcpy(dev_[L] + uint64_t(v) * sb, src, size_t(sb));
        // Recorded exactly as a demand fetch is, so that a later acquire which
        // HITS this slot before the copy lands returns an event that covers it.
        slot_ev_[uint64_t(L) * slots_ + v]  = fe;
        slot_seq_[uint64_t(L) * slots_ + v] = ++seq_;
        pending_.push_back(fe);
        ++issued;
        ++st_.spec_issued;
        st_.spec_bytes += sb;
    }
    return issued;
}

void ds4_plan_expert_groups(const Ds4ExpertCache& c, uint32_t L,
                            const uint32_t* occ, size_t n, uint32_t stream_cap,
                            std::vector<uint32_t>& bounds) {
    bounds.clear();
    if (n == 0) return;
    bounds.push_back(0);
    size_t gi = 0;
    while (gi < n) {
        size_t   gj = gi;
        uint32_t n_stream = 0;
        while (gj < n) {
            if (!c.is_static(L, occ[gj])) {
                // `gj > gi` keeps a group from ever being empty: a single expert
                // that already exceeds the cap gets a group of its own and is
                // refused by name inside acquire, which is strictly better than
                // silently dropping it from the batch.
                if (n_stream + 1 > stream_cap && gj > gi) break;
                ++n_stream;
            }
            ++gj;
        }
        bounds.push_back(uint32_t(gj));
        gi = gj;
    }
}

uint32_t ds4_order_hits_first(const Ds4ExpertCache& c, uint32_t L, const uint32_t* occ,
                              size_t n, uint32_t cap, std::vector<uint32_t>& ordered,
                              std::vector<uint32_t>& bounds) {
    ordered.clear();
    bounds.clear();
    if (n == 0) return 0;
    ordered.reserve(n);
    for (size_t i = 0; i < n; ++i) if (c.is_resident(L, occ[i]))  ordered.push_back(occ[i]);
    const uint32_t nh = uint32_t(ordered.size());
    for (size_t i = 0; i < n; ++i) if (!c.is_resident(L, occ[i])) ordered.push_back(occ[i]);
    if (nh) { bounds.push_back(0); bounds.push_back(nh); }
    if (nh < n) {
        std::vector<uint32_t> mb;
        ds4_plan_expert_groups(c, L, ordered.data() + nh, n - nh, cap, mb);
        for (uint32_t b : mb)
            if (bounds.empty() || b + nh != bounds.back()) bounds.push_back(b + nh);
    }
    return nh;
}

void Ds4ExpertCache::collect_dma_time() {
    if (pending_.empty()) return;
    xq_->wait();
    // The transfer queue only carries profiling when IE_QUEUE_PROFILING is set
    // (see the queue construction above).  Querying an event from a queue without
    // the property THROWS, so this has to ask first — and it must NOT silently
    // report 0 s of DMA, which would read as "streaming is free" in every number
    // derived from `dma_seconds`.  `dma_unavailable` is what the caller reports.
    const bool have_prof =
        xq_->has_property<sycl::property::queue::enable_profiling>();
    if (!have_prof) {
        st_.dma_unavailable = true;
        pending_.clear();
        return;
    }
    for (sycl::event& e : pending_) {
        const auto t0 = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        const auto t1 = e.get_profiling_info<sycl::info::event_profiling::command_end>();
        st_.dma_seconds += double(t1 - t0) * 1e-9;
    }
    pending_.clear();
}

}  // namespace ie
