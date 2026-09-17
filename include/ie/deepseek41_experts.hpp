// include/ie/deepseek41_experts.hpp — V4.1 routed experts on V4's residency machinery.
//
// V4's Ds4HostArena (pinned host slots), Ds4ExpertCache (static + streaming VRAM slots, fetch
// from the arena) and the expert-major grouped GEMM path are all keyed by (layer, expert) and a
// slot layout, not by model. After Phase 4's nibble repack a V4.1 expert IS a V4 MXFP4 slot, so
// V4.1 needs only (a) its own slot layout and packer, (b) a loader that fills the arena and the
// static slots from safetensors, and (c) a third tier for the experts that fit neither VRAM nor
// pinned RAM on this box: those are uploaded per call into a transient bank and run in the same
// grouped launch -- a DS4GemmJob takes any bank -- so the cache needs no change.
#pragma once

#include "ie/deepseek4_experts.hpp"
#include "ie/deepseek41.hpp"
#include "ie/expert_stream.hpp"
#include "ie/safetensors.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <string>
#include <vector>

namespace ie {

// One expert's slot: gate (w1), up (w3), down (w2) as MXFP4 planes in V4's convention --
// nibble plane at off0, E8M0 plane at off1, each 64 B aligned. K/N are contraction/output.
std::string ds41_slot_layout(uint32_t H, uint32_t EF, Ds4SlotLayout& out);

// The permute-free on-disk tail (docs/deepseek41/40, C2): the mmap tier's experts written once in
// the engine's slot layout, so a fill is one O_DIRECT pread straight into the staging slot. The
// file is a 4 KiB header, a (layer, expert) -> byte-offset table (n_layers * n_experts u64, 0 =
// absent), then the slots at 4 KiB-aligned offsets. Written by ie-ds41-expert-file; read by the
// tier when IE_DS41_EXPERT_FILE names it (default: none -- the pack path).
constexpr char kDs41ExpertFileMagic[8] = {'I', 'E', 'S', 'L', 'O', 'T', '0', '1'};
struct Ds41ExpertFileHeader {
    char     magic[8];
    uint32_t H = 0, EF = 0, n_layers = 0, n_experts = 0;
    uint64_t slot_bytes = 0;      // == ds41_slot_layout(H, EF).bytes
    uint64_t table_off = 0;       // byte offset of the offset table
    uint64_t n_slots = 0;
    uint32_t cutoff = 0;          // the ranking position from which experts were written (informational)
    uint32_t reserved = 0;
};
// Read one slot -- or its slice-th of n_slices page-granular slices -- from the file at byte offset
// `off` into `dst` (4 KiB-aligned, e.g. an imported staging slot) with O_DIRECT preads on `dfd`;
// `ns_read`, when given, accumulates the syscall time.
std::string ds41_slot_pread_file(int dfd, uint64_t off, void* dst, uint64_t bytes, uint32_t slice, uint32_t n_slices, uint64_t* ns_read);
// The vectorised nibble-plane permute against its scalar reference on a random plane of `bytes`
// (a multiple of 16): true iff byte-identical. The tier test runs it on the model's plane sizes.
bool ds41_permute_plane_selftest(uint64_t bytes, uint32_t seed);

// Pack one expert (w1, w3, w2 as bound) into `dst`, laid out by ds41_slot_layout: the nibble
// planes are permuted from V4.1's sequential order into the engine's interleaved one (the Phase
// 4 shuffle, on the host), the scale planes copied verbatim. `dst` may be a pinned arena slot
// or a staging buffer. Returns "" or a shape diagnostic.
std::string ds41_slot_pack(const Ds4SlotLayout& lay, const Ds41Tensor& w1, const Ds41Tensor& w3,
                           const Ds41Tensor& w2, void* dst);

// The same pack, byte for byte, with the planes READ from the shard file -- O_DIRECT when the
// shard has that descriptor, buffered pread otherwise -- into `bounce` (page-aligned, at least
// ds41_pack_bounce_bytes(lay)), then the nibble planes permuted from the bounce into `dst` and
// the scales copied. Nothing touches the mapping and nothing READS `dst`: measured
// (docs/deepseek41/23) the mapping's page-fault path is ~2 GB/s here, buffered pread into and
// in-place permute from a pinned slot the same, while this NVMe reads 9-10 GB/s either way
// into ordinary memory. Safe to call from several threads on distinct `dst`/`bounce`.
uint64_t    ds41_pack_bounce_bytes(const Ds4SlotLayout& lay);
// `ns_read` / `ns_permute`, when given, receive this call's time in the read syscalls and in
// the permute+copy (nanoseconds), for the phase accounting.
// `slice` / `n_slices`: read and permute only the slice-th of n_slices equal (16-byte aligned)
// parts of every plane -- the decode shape (Phase 11), where a layer has one or two mmap experts
// and the reader threads would otherwise sit idle while one of them walks 17.9 MiB alone.
std::string ds41_slot_pack_pread(const Ds4SlotLayout& lay, const SafetensorsModel& store,
                                 const Ds41Tensor& w1, const Ds41Tensor& w3, const Ds41Tensor& w2,
                                 void* dst, void* bounce, uint64_t* ns_read = nullptr, uint64_t* ns_permute = nullptr,
                                 uint32_t slice = 0, uint32_t n_slices = 1);

struct Ds41TierStats {
    uint32_t experts_static = 0, experts_pinned = 0, experts_mmap = 0;   // occupied experts by tier, this call
    uint32_t stream_hits = 0;            // of experts_pinned: served from a stream slot with no transfer (the cache's counter)
    uint32_t experts_cpu = 0;            // of experts_pinned: misses computed on the CPU from the arena (Phase 13, T = 1 only)
    uint32_t experts_mmap_file = 0;      // of experts_mmap: filled by one direct pread from the expert file (Phase 15 C2)
    double   ms_cpu = 0;                 // the CPU leg's wall, dispatch to join (overlaps the groups)
    double   ms_cpu_work = 0;            // the worker's own compute time inside that leg
    uint64_t bytes_pinned_to_vram = 0;   // arena -> stream slot (H2D, no repack)
    uint64_t bytes_mmap_to_vram   = 0;   // mmap  -> transient bank (read + repack + H2D)
    double   ms = 0;
    // phase breakdown of `ms` (docs/deepseek41/23, criterion 1): prep = host counting sort +
    // the gather/quant submits; mmap = the whole mmap-tier fill (reads + DMAs), wall; mmap_pack =
    // the host pread + permute batches alone, wall (their DMAs overlap the next batch);
    // groups = the fetch/GEMM group loop + scatter, wall
    double   ms_prep = 0, ms_mmap = 0, ms_mmap_pack = 0, ms_groups = 0;
    // inside mmap_pack, summed over the reader threads then divided by their count (so the two
    // are on the wall-time scale of mmap_pack): the read syscalls, and the permute + scale copy
    double   ms_mmap_read = 0, ms_mmap_permute = 0;
    // the mmap fill runs on a reader thread, overlapped with the static/pinned groups; the mmap
    // experts' own GEMMs run last, after the fill: ms ~= prep + max(mmap, groups) + mmap_group
    double   ms_mmap_group = 0;
    double   ms_tail = 0;                // the scatter and the transient bank's free, after the mmap group (gate 8b finding 1)
    // the main thread's chain is prep -> spawn -> groups -> join -> mmap_group -> tail, and those six
    // sum to `ms` by construction; `mmap` is the reader's leg, overlapped with groups (+ join)
    double   ms_spawn = 0, ms_join = 0;
};

// The three tiers for a contiguous range of layers on one device.
class Ds41ExpertTier {
public:
    ~Ds41ExpertTier();
    // `ranking[L]` (model layer index) lists all E experts most-important-first; ranks
    // [0, n_static) become static VRAM slots, [n_static, n_static + n_pinned) pinned host
    // slots, the rest the mmap tier. `stream_slots` evictable VRAM slots per layer serve the
    // pinned tier's misses. `max_tokens` sizes the batch workspace.
    // Expert parallel (docs/deepseek41/38): with `n_parts` > 1 this tier serves only the experts
    // at ranking positions == `part` (mod n_parts) of every layer -- the counts above apply to
    // that subset, in its ranking order -- and the others are tier 3, "not this tier's": never
    // resident, skipped by moe(), their packed rows given weight 0 and zeroed so the scatter
    // adds nothing. The default (one part) is byte-for-byte the whole tier; `part == n_parts`
    // is the empty subset (no expert of any layer: the control arm of docs/38).
    std::string init(sycl::queue& q, const DeepSeek41Model& m, uint32_t first_layer, uint32_t n_layers,
                     const std::vector<std::vector<uint32_t>>& ranking,
                     uint32_t n_static, uint32_t n_pinned, uint32_t stream_slots, uint32_t max_tokens,
                     uint32_t part = 0, uint32_t n_parts = 1);
    void free_storage(sycl::queue& q);

    // y[T, H] = routed MoE for model layer L (must be in this tier's range); y is zeroed here.
    // x is the post-ffn_norm stream [T, H] fp32; ridx/rw are HOST [T*top_k] from the router.
    std::string moe(sycl::queue& q, uint32_t L, const float* x, const int32_t* ridx, const float* rw,
                    uint32_t T, float* y, float swiglu_limit);

    const Ds41TierStats& last() const { return st_; }
    // Expert parallel, bit-exact (docs/deepseek41/38): the tier that serves the OTHER card's share
    // runs moe() with `ep_export` set -- everything as usual, but no scatter: its packed rows
    // (fp16, as its GEMMs wrote them) stay in the batch workspace, `ep_rows()` lists which packed
    // rows are its experts', and `packed_yp()` is the buffer. The owner's tier runs with an
    // `ep_import` hook, called on its queue after its own rows are written and before its
    // scatter: the hook drops the other tier's rows into the same packed positions (the packing
    // is the same counting sort of the same routing on both), the alien weights stay REAL, and
    // the one scatter then accumulates the six products in the same order as the whole tier.
    void set_ep_export(bool on) { ep_export_ = on; }
    // The CPU miss split's cores for THIS tier, set before init (docs/deepseek41/42 step 3: under
    // expert parallel each card's tier gets its own half of the E-cores); empty = the env / default.
    void set_cpu_cores(std::string cores) { cpu_cores_override_ = std::move(cores); }
    void set_ep_import(std::function<std::string(sycl::queue&, sycl::half* yp, uint32_t rows)> fn) { ep_import_ = std::move(fn); }
    // Phase 18 (docs/deepseek41/46 term 1): called once per moe() on the caller's thread right after the
    // first group's fetch is issued -- work enqueued on the queue there runs under that group's DMAs
    void set_pre_groups(std::function<void()> fn) { pre_groups_ = std::move(fn); }
    const std::vector<int32_t>& ep_rows() const { return ep_rows_; }
    sycl::half* packed_yp() const { return static_cast<sycl::half*>(bws_.yp); }
    uint64_t vram_bytes() const { return cache_.device_bytes(); }
    uint64_t pinned_bytes() const { return arena_.total_bytes(); }
    uint32_t n_static() const { return cache_.static_slots(); }
    uint32_t n_stream() const { return cache_.stream_slots(); }
    uint32_t n_pinned() const { return np_; }
    // 0 static VRAM, 1 pinned host, 2 mmap, 3 not this tier's part -- for model layer L in this
    // tier's range (2 otherwise)
    uint32_t tier_of(uint32_t L, uint32_t e) const {
        return ready_ && L >= L0_ && L < L0_ + nL_ && e < E_ ? tier_[L - L0_][e] : 2u;
    }

private:
    const DeepSeek41Model* m_ = nullptr;
    uint32_t L0_ = 0, nL_ = 0, E_ = 0, TK_ = 0, H_ = 0, EF_ = 0, np_ = 0;
    Ds4SlotLayout   lay_;
    Ds4HostArena    arena_;
    Ds4ExpertCache  cache_;
    DS4ExpertBatchWs bws_{};
    // Phase 18 (docs/deepseek41/46 term 1): the packing's per-call H2D copies staged through pinned host
    // memory, so their enqueue never blocks the host thread behind work already on the in-order queue
    // (from a pageable vector it did, and held the first group's fetch behind the shared expert's kernels)
    int32_t* h_row_tok_ = nullptr; int32_t* h_tk2p_ = nullptr; float* h_w_pk_ = nullptr; float* h_w_pk2_ = nullptr;
    // Pinned staging slots: the mmap tier's per-call path and the static installs at load both
    // pread-pack into them (kReaders at a time) and DMA from them; two batches of readers fit,
    // so batch k's DMAs overlap batch k+1's reads.
    std::vector<uint8_t*> mm_stage_;
    uint32_t mm_stage_slots_ = 16;
    // Phase 15 C2 (docs/deepseek41/40): with IE_DS41_EXPERT_FILE the staging slots are anonymous
    // huge-page memory imported into the driver (O_DIRECT cannot land in host USM: EFAULT), the
    // file's (layer, expert) -> offset table is held here, and an mmap expert the file covers is
    // filled by one direct pread instead of pread + permute. Without the file: byte-for-byte today's.
    bool                  stage_imported_ = false;
    void*                 ze_drv_ = nullptr; void* ze_release_ = nullptr;
    int                   mm_file_fd_ = -1;
    std::vector<uint64_t> mm_file_off_;       // [layer of this tier][expert] -> byte offset, 0 = absent; empty = no file
    std::vector<void*> bounce_;                // per reader thread, page-aligned ordinary memory
    // The mmap tier's DMAs go on their own in-order queue (same device and context) so they run
    // beside the compute queue's GEMMs instead of in line with them.
    std::unique_ptr<sycl::queue> dq_;
    // The mmap tier's transient bank, kept across calls and grown when a call needs more: a
    // per-call malloc_device of ~1 GiB cost 3.4 s per pp2048 pass (docs/deepseek41/23, run J).
    uint8_t* mm_dev_ = nullptr; size_t mm_dev_cap_ = 0;
    std::vector<std::vector<uint8_t>> tier_;   // [layer][expert]: 0 static, 1 pinned, 2 mmap, 3 not this part
    bool ep_export_ = false;                   // expert parallel: leave the packed rows, skip the scatter
    std::function<std::string(sycl::queue&, sycl::half*, uint32_t)> ep_import_;   // ... or import the other tier's rows before it
    std::function<void()> pre_groups_;                                             // Phase 18: the owner's shared expert, under the first fetch
    std::vector<int32_t> ep_rows_;             // the last call's packed rows that are this tier's experts'
    Ds41TierStats   st_;
    bool ready_ = false;
    bool empty_ = false;                       // the empty subset (part == n_parts): no storage at all, moe() returns at once
    bool share_ = false;                       // a parity share of every layer (expert parallel): the CPU split's q* default is 0.60 then (docs/52)
    // ---- the decode miss path (docs/deepseek41/34): at T = 1, a share of the pinned MISSES is
    // computed on the CPU from the arena by a persistent worker whose OpenMP team sits on its own
    // cores, concurrently with the GPU groups. Phase 20 (docs/deepseek41/49): each CPU expert's output
    // lands as ITS PACKED fp16 ROW (the routing weight stays real; the one scatter applies it), so an
    // exporting tier's CPU rows travel like its GPU rows and the split works under expert parallel.
    // IE_DS41_CPU_MISS=1 enables; IE_DS41_QSTAR = the PCIe share of the misses (default 0.30 =
    // 1.2 / (1.2 + 2.75) experts per ms, measured); IE_DS41_CPU_CORES (default 8-19, the E-cores).
    struct CpuMiss {
        bool on = false; float qstar = 0.30f; std::vector<int> cores; int nthreads = 0;
        std::thread th; std::mutex mu; std::condition_variable cv; bool stop = false, pending = false, done = true;
        const Ds4SlotLayout* lay = nullptr; float limit = 0.f;
        std::vector<std::pair<const void*, uint32_t>> work;   // (arena slot, its packed row)
        std::vector<float> x, scratch, out;                   // host: the activation, EF*2, H
        double work_ms = 0;                                   // the last request's compute time
        sycl::half* h_rows = nullptr;                         // pinned host: the rows, [top_k, H] fp16, in work order
    };
    CpuMiss cpu_;
    std::string cpu_cores_override_;
    // The mmap fill's reader: a PERSISTENT thread per affinity mode ([0] the calling thread's partition -- decode --, [1]
    // every core -- prefill with the CPU leg on), so its OpenMP team is created once. It used to be a fresh std::thread
    // per moe() call, and libiomp5 never reclaimed the teams those threads opened: a served process reached 94,178
    // threads and grew ~10,000 threads / ~0.5 GB of stacks per 120-token reply (2026-09-16, docs/deepseek41/88).
    struct MmReader {
        std::thread th; std::mutex mu; std::condition_variable cv;
        std::function<void()> job; bool pending = false, done = true, stop = false;
    };
    MmReader mmr_[2];
    void mm_reader_post(int which, std::function<void()> job);   // starts the thread on first use (it inherits the caller's affinity)
    void mm_reader_wait(int which);
    void mm_reader_stop();
    void cpu_start(sycl::queue& q);
    void cpu_stop();
    void cpu_run();
};

}  // namespace ie
