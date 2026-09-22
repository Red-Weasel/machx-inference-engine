// include/ie/expert_stream.hpp — DeepSeek-V4-Flash expert streaming (Phase 5).
//
// WHY THIS EXISTS — STREAMING IS FORCED, NOT AN OPTIMISATION
// ----------------------------------------------------------
// The routed experts of DeepSeek-V4-Flash-0731 UD-Q3_K_XL are 120.393 GB
// (docs/deepseek4/21_tensor_manifest_verified.md).  The box has 64 GB of VRAM
// across two B70s and Level-Zero P2P is UNAVAILABLE between them
// (`can_access_peer == 0` both directions, docs/deepseek4/31 §2.4), so the two
// caches are disjoint and neither can borrow the other's.  No quantisation this
// engine can decode makes 120 GB fit.  The expert bytes therefore have to cross
// PCIe every token, and the only question the design controls is whether the
// DMA engine ever goes idle.
//
// WHAT IS IMPLEMENTED HERE
// ------------------------
//   Ds4HostArena   — segmented PINNED host arena.  Every segment is smaller
//                    than the device's max_mem_alloc_size (32.53 GB measured,
//                    docs/deepseek4/32_pinned_host_ceiling_verified.md) and a
//                    whole number of experts; an expert NEVER straddles a
//                    segment boundary, so one fetch is one contiguous memcpy.
//                    A failed segment allocation is a hard, named error.
//   Ds4ExpertCache — per-card VRAM slot cache, statically partitioned per layer
//                    with FIFO replacement inside a layer, driven by ONE host
//                    directory, plus a dedicated in-order TRANSFER queue on the
//                    same device+context as the compute queue so H2D overlaps
//                    compute (0.90 overlap fraction measured on this hardware,
//                    docs/deepseek4/31 §0.2).
//
// THE SEAM, AND WHY IT IS THIS ONE
// --------------------------------
// `DeviceAllocator` (include/ie/allocator.hpp) holds exactly one in-order queue
// per device, and every transfer in the engine shares the compute queue, where
// the in-order property serialises it by construction.  A second queue is the
// smallest thing that has to change.  `allocator.hpp` is a gate-passed Phase-0
// file, so instead of editing it this class constructs its own transfer queue
// from the compute queue's OWN context and device:
//
//     sycl::queue(compute.get_context(), compute.get_device(), in_order)
//
// Same context ⇒ the event returned by a transfer `memcpy` is legal to consume
// via `h.depends_on(ev)` in a compute submission.  Nothing in `allocator.hpp`
// or `memory_plan.cpp` moves, and the engine's existing single-queue model is
// untouched for every other architecture.
//
// LAYOUT DISCIPLINE (docs/deepseek4/31 §2.5)
// ------------------------------------------
// The DMA is a dumb byte copy: no repack can happen per fetch (that would need
// the CPU to reformat ~25 GB/s).  So the host arena stores each expert ALREADY
// in the SoA plane form `ds4_expert_gemv` reads, with the three matrices'
// planes packed back-to-back inside one slot.  `Ds4SlotLayout` is that packing,
// and `ds4_slot_bank()` re-derives a single-expert `DS4ExpertBank` view over a
// slot without copying anything.  Fetching an expert is therefore exactly one
// `memcpy` of `Ds4SlotLayout::bytes`.
#pragma once

#include "ie/deepseek4_experts.hpp"
#include "ie/dtype.hpp"
#include "ie/gguf.hpp"

#include <sycl/sycl.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace ie {

// Measured single-allocation ceiling on this device (max_mem_alloc_size =
// 32.53 GB).  Both the host arena and the device arena refuse to ask for more
// than this in one call — loudly.
inline constexpr uint64_t kDs4MaxAllocBytes = 32'530'000'000ull;

// Recommended host segment size.  Whole layers are packed into a segment until
// the next one would exceed this, so a segment is ~16 GB and always < the cap.
inline constexpr uint64_t kDs4HostSegmentTarget = 16ull << 30;

// FALLBACK ceiling on the pinned host spill, used ONLY when /proc/meminfo
// cannot be read.  It is deliberately pessimistic — a process that cannot see
// how much memory the box has must not guess high — and it is NOT the default
// any more: `ds4_host_pin_cap_live` below is, and the loader only reaches this
// constant when the derivation is impossible.
//
// 84.0 GB is the Q3 residency plan's 80.419 GB (docs/deepseek4/33 §2.3,
// recomputed from the real tensor table) plus headroom, comfortably under the
// 96 GB cumulative pin that was reached and touched (docs/deepseek4/32).  It is
// enforced by refusing to load, never by silently pinning less.
inline constexpr uint64_t kDs4HostPinCapDefault = 84'000'000'000ull;

// The DYNAMIC reserve: memory left unpinned relative to what is available RIGHT
// NOW — page cache, the desktop, and whatever else the box is doing.
// max(this, MemTotal / kDs4HostPinReserveDiv) — the fixed floor keeps a small
// machine from deriving a cap that leaves nothing, and the fraction keeps a
// large one from leaving only 16 GB.
inline constexpr uint64_t kDs4HostPinReserveMin = 16ull << 30;
inline constexpr uint64_t kDs4HostPinReserveDiv = 8;

// The ABSOLUTE reserve: RAM that must stay unpinned no matter how idle the box
// looks.  See the derivation note below for why a second, MemTotal-relative term
// is needed at all — MemAvailable alone would authorise ~95% of RAM on a
// freshly-booted box, and a pinned page is one the kernel can never take back.
inline constexpr uint64_t kDs4HostPinUnpinnedDiv = 4;

// Derives the pinned-host ceiling from /proc/meminfo instead of a constant.
//
//     cap = min( MemAvailable - max(kDs4HostPinReserveMin, MemTotal/8),
//                MemTotal     - max(kDs4HostPinReserveMin, MemTotal/4) )
//
// WHY TWO TERMS, AND WHAT THE HAZARD ACTUALLY IS
// ----------------------------------------------
// The failure this guards against is on record, and it is NOT an out-of-memory
// kill.  `journalctl`, 2026-08-02 00:28:49 on this box:
//
//   systemd-oomd: Killed /user.slice/.../app-code-9571.scope due to memory
//   pressure for /user.slice/user-1000.slice/user@1000.service being
//   70.24% > 50.00% for > 20s WITH RECLAIM ACTIVITY
//   systemd[7446]: app-code-9571.scope: systemd-oomd killed 147 process(es)
//
// `systemd-oomd` is enabled and active here, and `user@1000.service` carries
// ManagedOOMMemoryPressure=kill at a 50% PSI limit over
// DefaultMemoryPressureDurationSec=20s.  So the thing that kills the desktop is
// SUSTAINED RECLAIM — the kernel evicting page cache continuously — not
// exhaustion.  A pinned page is unswappable AND unreclaimable, so every pinned
// byte permanently shrinks the only pool reclaim can draw on.  The cap's job is
// therefore to leave enough reclaimable memory that the box never enters that
// state, which is a different question from "does it fit".
//
//   * The DYNAMIC term is what catches the recorded incident.  ~47 GB of the
//     memory that looked "available" that night was GGUF page cache, and the pin
//     grew into it while the loader was still reading through the same file —
//     squeezed from both sides (docs/deepseek4/40).  MemAvailable moves with the
//     cache, so subtracting a reserve from it is exactly the check a constant
//     cannot perform.  It also binds hardest when the box is busy, which is when
//     it should.
//   * The ABSOLUTE term is what a busy-box check cannot do: on an idle box
//     MemAvailable approaches MemTotal, and `MemAvailable - reserve` would
//     authorise a pin that leaves the kernel nothing elastic for the rest of the
//     run.  MemTotal/4 is a POLICY floor, not a derivation, and it is stated as
//     one.  Its two sanity bounds are real numbers: it must not fall below the
//     119.5 GB that docs/deepseek4/40 records as successfully pinned on this box
//     once the cache had drained, and on this 168.01 GB box it leaves 42.00 GB,
//     which is 3.2x the 13.2 GB that was non-reclaimable at the time of writing.
//
// The one structural change that makes this safe rather than merely bounded:
// `DeepSeek4Runtime::load` allocates the WHOLE arena before it reads a single
// expert byte (src/model/deepseek4.cpp, `arena_.init_set` precedes the pack
// loop).  The incident's "pin grows while the cache is being filled" mode cannot
// recur in that order — the pin either lands entirely, up front, or the load
// refuses before any streaming starts.
//
// `why` always receives the derivation, success or failure, so the number is
// never a mystery in a log.  `meminfo_ok`, when given, distinguishes the two
// zero returns that a caller MUST NOT confuse:
//   *meminfo_ok == false : /proc/meminfo was unreadable.  Nothing is known;
//                          fall back to kDs4HostPinCapDefault.
//   *meminfo_ok == true, return 0 : the box genuinely has less free than the
//                          reserve.  NOTHING may be pinned, and substituting the
//                          84 GB constant here would raise the cap on a machine
//                          that is already starved — the exact inversion this
//                          out-parameter exists to prevent.
uint64_t ds4_host_pin_cap_live(std::string& why, bool* meminfo_ok = nullptr);

// Evictable slots reserved per layer.  Everything above this in the VRAM arena
// is STATIC: uploaded once, never evicted, and — the whole point — never given
// a pinned host copy.  12 is the split doc 33 §2.3/§2.4 chose for both quants
// (97 -> 85+12, 80 -> 68+12) and is 2x the top-6 routing width.
inline constexpr uint32_t kDs4MinStreamSlots = 12;
// Prefill fetch banks: slots per bank (2 banks), see Ds4ExpertCache::acquire_prefill.
inline constexpr uint32_t kDs4PrefillBankSlots = 32;
// A grouped call with at least this many tokens is a prefill sweep and may use
// the banks; below it (decode, small chunks) the LRU eviction path is used.
inline constexpr uint32_t kDs4BankMinTokens = 64;

// Written into acquire()'s `out_slots` for an expert this cache cannot serve —
// out of range, or neither statically resident nor host-pinned, or no evictable
// slot left.  It is deliberately NOT a valid slot index: a caller that ignores
// it will fault rather than silently compute with the wrong expert's weights.
inline constexpr uint32_t kDs4NoSlot = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// Slot layout — one expert's gate+up+down, SoA planes packed contiguously.
// ---------------------------------------------------------------------------

// The planes of ONE weight matrix inside a slot.  Offsets are byte offsets from
// the slot base and are 64 B aligned so `uint32_t`/`uint16_t` plane loads stay
// naturally aligned after a raw byte memcpy.
//   IQ3_XXS : p0 = gp (uint8, N*K/4), p1 = ap (uint32, N*K/32), p2 = dp (uint16, N*K/256)
//   MXFP4   : p0 = mx_qs (uint8, N*K/2), p1 = mx_e (uint8, N*K/32), p2 unused
//
// `K`/`N` are the dimensions of the sub-matrix THIS CARD holds; `src_K`/`src_N`
// are the tensor's own dimensions in the file and `k0`/`n0` say where the slice
// starts.  They are equal and zero under single-card residency.  Carrying them
// here rather than passing a slice descriptor to the packer is deliberate: the
// layout and the repack read the SAME four numbers, so they cannot drift.
struct Ds4MatPlanes {
    DType    dt = DType::kF32;
    uint32_t K = 0, N = 0;
    uint32_t src_K = 0, src_N = 0;
    uint32_t k0 = 0, n0 = 0;
    uint64_t off0 = 0, len0 = 0;
    uint64_t off1 = 0, len1 = 0;
    uint64_t off2 = 0, len2 = 0;
};

struct Ds4SlotLayout {
    Ds4MatPlanes gate, up, down;
    uint64_t     bytes = 0;   // total, 64 B aligned
};

// Derives the layout from the three GGUF expert tensors' OWN dtypes.  Nothing
// here keys off the layer index: blk.26 (MXFP4 gate/up) and blk.25 (IQ3_XXS
// gate/up) produce different layouts through the same call.
// Returns "" on success, else a diagnostic.
std::string ds4_slot_layout(const GgufTensorInfo& gate, const GgufTensorInfo& up,
                            const GgufTensorInfo& down, uint32_t H, uint32_t EF,
                            Ds4SlotLayout& out);

// A single-expert (E == 1) bank view over `base`, which must point at a slot
// laid out by `ds4_slot_layout`.  No allocation, no copy — pointer arithmetic.
DS4ExpertBank ds4_slot_bank(const Ds4MatPlanes& m, void* base);

// Repacks expert `e` of the three GGUF tensors into `dst` (one slot's worth of
// bytes).  This is the load-time work the fetch path must never do.  When `lay`
// was produced by `ds4_slot_layout_tp` this writes ONLY this card's slice.
std::string ds4_slot_pack(const Ds4SlotLayout& lay, const GgufTensorInfo& gate,
                          const GgufTensorInfo& up, const GgufTensorInfo& down,
                          uint32_t e, void* dst);

// ---------------------------------------------------------------------------
// KERNEL READAHEAD OVER THE MMAP — why the load was disk-bound
// ---------------------------------------------------------------------------
// `GgufReader::open` maps every shard and immediately advises the whole mapping
// MADV_RANDOM (src/loaders/gguf_reader.cpp:230 and :256, "tensor data is touched
// sparsely").  MADV_RANDOM sets `ra_pages = 0` on the VMA, which switches kernel
// readahead OFF: every mmap fault then fetches exactly ONE page and waits for
// it.  That hint is right for the sparse metadata probing the reader itself
// does, and catastrophically wrong for this loader, which afterwards streams
// ~128 GB through the same mapping in ascending order.
//
// MEASURED on this box (2026-08-02; model on /dev/sdb, a USB-2 SPINNING disk):
//   iostat -x sdb during a real two-card load : rareq-sz = 3.91 kB, r_await 64 ms
//                                               -> ONE 4 kB PAGE PER SEEK
//   /proc/<pid>/io over a real two-card load  : 0.0388 MB/s aggregate over 206 s
// At that rate the 128 GB model would take over a month, and the load in fact
// never got past its FIRST dense tensor.
//
// WHICH ADVICE ACTUALLY FIXES IT — MEASURED, NOT ASSUMED
// ------------------------------------------------------
// scratchpad/ra_probe.c streams 32 MB cold from a real shard through a mapping
// first advised MADV_RANDOM exactly as the reader does, then re-advised:
//
//   (leave MADV_RANDOM)  < 0.37 MB/s   (timed out at 90 s — the bug)
//   MADV_WILLNEED        < 0.37 MB/s   (timed out at 90 s — NO EFFECT AT ALL)
//   MADV_SEQUENTIAL        3.81 MB/s
//   MADV_SEQUENTIAL+WILLNEED, 8 MB pipelined chunks
//                          7.43 MB/s
//   MADV_NORMAL       5.62 .. 10.25 MB/s over four cold runs
//   dd bs=32M iflag=direct, same drive, same minutes
//                     5.8 .. 7.2 MB/s
//
// So: MADV_WILLNEED — the obvious "prefetch this" call — does NOTHING on a VMA
// that is still MADV_RANDOM, because the readahead it forces is bounded by the
// VMA's own ra_pages, which MADV_RANDOM has set to zero.  Asking for the pages
// is useless; what has to be undone is the advice that disabled readahead.
// MADV_NORMAL restores the default window (read_ahead_kb = 1024 here) and alone
// reaches the drive's sequential rate.  Pipelined WILLNEED on top of it added
// nothing measurable, so it is not done: the simplest thing that works.
//
// (Note the sequential baseline is 5.8-7.2 MB/s TODAY, not the 36.4 MB/s
// measured earlier, because an unrelated download is writing to the same
// spindle at ~15 MB/s and holding it at >90% utilisation.  The relevant
// comparison is against dd taken in the same minutes, which this matches.)
//
// Options weighed and rejected:
//   * explicit pread() of whole-expert byte ranges into a staging buffer is the
//     strongest fix, but the file descriptor and each shard's mmap base live in
//     `GgufReader` and are not exposed — reaching them means editing the GGUF
//     reader, which this change does not own.
//   * one sequential pass that reads each whole expert ONCE and hands both cards
//     their slice from the same buffer is better still (it halves total bytes
//     read; see ds4_stream_advise_experts), but needs DeepSeek4Runtime::load
//     split into phases across the TP orchestrator.  Independent of, and
//     composable with, this fix.
//
// ADVISORY ONLY.  madvise cannot change WHAT is read — an ignored or failed hint
// simply means pages fault in on demand as before.  These calls therefore cannot
// alter a single loaded byte, only the rate, and deliberately return void and
// swallow errors: there is no failure mode to report, and promoting a
// performance hint to a hard error would be wrong.
void ds4_stream_advise(const void* p, uint64_t nbytes) noexcept;

// Re-enables readahead over one layer's three routed-expert tensors, whole.
//
// Whole tensors, not this card's slice, and that is deliberate.  Under expert-TP
// `down` is sliced along its CONTRACTION dim, so the card's half is `efc/blk`
// blocks out of every one of `H` source columns — a stride so fine (392 B taken
// of every 784 B for IQ3_XXS) that it dirties EVERY page of the expert's `down`
// block regardless.  Advising the whole contiguous range therefore costs no
// extra I/O and removes every seek.  `gate`/`up` are sliced on their output dim,
// so the two cards' halves are adjacent runs — contiguous when taken whole.
//
// NOTE (not fixed here — see the report): the TP orchestrator loads card 0
// fully, then card 1, so these ranges are streamed TWICE, once per card, and at
// 128 GB the page cache cannot bridge the two passes.  This makes each pass
// sequential; it does not make them one pass.
void ds4_stream_advise_experts(const GgufTensorInfo& gate, const GgufTensorInfo& up,
                               const GgufTensorInfo& down) noexcept;

// ---------------------------------------------------------------------------
// Hidden-dim expert tensor parallelism (docs/deepseek4/33 §2.1, "Option A")
// ---------------------------------------------------------------------------
//
// Both cards select the SAME `n_experts_used` experts for a token; each holds
// half of every expert's intermediate dimension.  With
// `moe_intermediate_size == 2048` card 0 owns gate/up columns [0,1024) and down
// rows [0,1024), card 1 the complement.  Per-card expert bytes are therefore
// exactly `1/n_cards` of the pool, which is the whole point: it is the only
// lever that brings either quant's pinned-host spill under the cap on this box.
//
// WHY THE HIDDEN DIM AND NOT THE EXPERT INDEX (doc 31/33's "Option B").  Both
// schemes need the SAME cross-card reduction — under Option B a token's 6
// experts are spread over both cards, so the two per-card partial sums of
// `Σ_k w_k · expert_k(x)` still have to be added — and both move the same bytes
// per token per card and cover the same number of distinct experts.  What
// separates them is BALANCE: under Option B the experts-per-card is
// Binomial(6, 1/2), so one card does 4-6 of them while the other waits, whereas
// Option A gives each card exactly half of every expert.  Option A also reuses
// the slicing `src/model/gptoss_tp.cpp` already proves.
//
// BLOCK BOUNDARIES.  gate/up are sliced along their OUTPUT dim (N == EF); the
// quantisation blocks run along K, and the SoA planes are per-column
// contiguous, so an N-slice can never cut a block at any offset.  `down` is
// sliced along its CONTRACTION dim (K == EF), which DOES cut blocks unless
// `ef0` and `efc` are whole multiples of the dtype's block: 256 elements for
// IQ3_XXS, 32 for MXFP4.  `ds4_expert_slice` checks that against each tensor's
// OWN dtype and refuses by name rather than rounding.
struct Ds4ExpertSlice {
    uint32_t n_cards = 1;
    uint32_t card    = 0;
    uint32_t ef0     = 0;   // first intermediate column/row this card owns
    uint32_t efc     = 0;   // how many of them
};

// Derives the slice for `card` of `n_cards` and PROVES it lands on block
// boundaries for the three tensors' own dtypes.  Returns "" or a diagnostic
// naming the tensor, its dtype, its block size and the offending offset.
std::string ds4_expert_slice(const GgufTensorInfo& gate, const GgufTensorInfo& up,
                             const GgufTensorInfo& down, uint32_t EF,
                             uint32_t n_cards, uint32_t card, Ds4ExpertSlice& out);

// Slot layout for the slice `sl` only.  `ds4_slot_layout` is exactly this with
// the whole-expert slice, and produces byte-identical output.
std::string ds4_slot_layout_tp(const GgufTensorInfo& gate, const GgufTensorInfo& up,
                               const GgufTensorInfo& down, uint32_t H, uint32_t EF,
                               const Ds4ExpertSlice& sl, Ds4SlotLayout& out);

// ---------------------------------------------------------------------------
// Cross-card reduction of the sliced `down` partials — through HOST memory
// ---------------------------------------------------------------------------
//
// Under hidden-dim expert-TP each card's `down` GEMV contracts over half the
// intermediate dimension, so each produces a PARTIAL [n] fp32 result and the
// true output is their sum.  Level-Zero P2P is UNAVAILABLE between the two B70s
// (`can_access_peer == 0` both directions; a D2D copy fails with
// `UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`), so the sum has to go through host
// memory.  Measured round trip on this box: 18.44 us median for 64 KB, 16.83 us
// for 32 KB (docs/deepseek4/33 §2.5) — latency-dominated, so a decode-step
// payload of `hidden * 4` = 16 KB costs ~16 us and 43 of them cost ~0.7 ms.
//
// `stages[c]` is card c's OWN pinned host buffer of `n` floats, allocated with
// `sycl::malloc_host(n, qs[c])` — i.e. in THAT queue's context.
//
// WHY PER CARD AND NOT ONE BUFFER.  A USM allocation belongs to the context it
// was made in, and handing card 1's queue a pointer that only card 0's context
// knows is undefined behaviour.  Giving every card a buffer in its own queue's
// context makes the transfer legal by construction, on one device or on two.
// The cost is `n_cards - 1` extra host-to-host copies of `n` floats (64 KB at
// hidden=4096, decode) against an 18.44 us PCIe round trip.
//
// AND IT IS NOT MERELY BELT-AND-BRACES, THOUGH IT LOOKS LIKE IT ON THIS STACK.
// An earlier version of this note said the two runtimes hold two DIFFERENT
// contexts, because `DeepSeek4Runtime` builds its queue from a bare
// `sycl::device` and that was read as "the implicit per-device context".  It is
// not: MEASURED on oneAPI 2026.1 here, `sycl::queue(dev0)` and
// `sycl::queue(dev1)` compare EQUAL on `get_context()`, and that context reports
// TWO devices — `queue(device)` uses the PLATFORM default context, which spans
// every device of the platform.  So one shared buffer would happen to work
// today.  It is still not what this does, for the reason above: the guarantee
// SYCL offers is about the context an allocation was made in, and a run that
// rests on the platform default context happening to span both cards breaks the
// day anything hands these queues an explicit single-device context.
//
// Pageable memory would silently drop the transfer onto a staged, much slower
// path, which is exactly the "no pageable fallback" rule this subsystem is built
// on.  MEASURED: staging in plain `malloc` memory works — both B70s report
// `aspect::usm_system_allocations` — and is 1.8x SLOWER than pinned (21.5 us vs
// 12.0 us for a 16 KB round trip).  It is not an option, it is a regression.
//
// BOTH HOPS ARE KERNELS, NOT COPIES.  `memcpy` runs on a Level-Zero copy engine,
// which is the same engine the expert stream's multi-megabyte H2D bursts occupy;
// a 32 KB hop behind a 4 MB burst waits for the burst.  Under a ~41 GB/s expert
// stream that turns a 39 us reduction into a 196 us one.  A `parallel_for` over
// the pinned host pointer moves the same bytes over the same link on the COMPUTE
// engine and does not queue behind it.  BOTH hops have to move: converting only
// one was measured and buys nothing, because whichever is left is enough to put
// the reduction back in that queue.  The full table, the sizes swept, the idle
// cost and the rotation the measurement needed are in the implementation.
//
// PREFILL SIZES (n >= 65536, docs/deepseek4/72 Phase F, 2026-09-02) are the
// other regime: there the hop is bandwidth-bound and the compute-engine kernel
// moves 8 MB at a fraction of the copy engine's rate, while the worst queueing
// behind one 6.7 MB expert burst is ~0.3 ms — so above that size both hops ARE
// memcpys, and the host sum's fan-out is fused into the add (same adds, same
// order, so the bytes are the same; gated bit for bit at both sizes by
// deepseek4_stream_gate_test §8).
//
// THE COMPLETION CONTRACT, WHICH IS NOT "BLOCKING".  On return the sum has been
// SUBMITTED to every card's queue, not completed: anything the caller enqueues
// on `qs[c]` afterwards sees the full sum, because these queues are in-order,
// but the host must not assume the write has landed.  Two obligations follow,
// and both are the caller's:
//   * do not FREE or RESIZE `stages[c]` without draining `qs[c]` first — the
//     return kernel reads those pages, and `sycl::free` is not ordered against
//     the queue;
//   * do not write `stages[c]` from the host without draining `qs[c]` — for the
//     per-layer decode loop the next call's own D2H wait already does this, so
//     back-to-back reductions need nothing extra.
// The only wait left inside is the D2H one, which is a genuine data dependency:
// the sum happens on the host.
//
// WHY THE SUM IS STILL ON THE HOST, WHICH IS A MEASURED CHOICE AND NOT AN
// OVERSIGHT.  Summing on a card instead needs every card's staged partial to be
// readable by that card, i.e. ONE `sycl::context` spanning both devices.  That
// context DOES construct on this box, host USM allocated in it IS readable and
// writable by kernels on both cards, and a reduction built on it is correct and
// bit-identical across cards (verified).  It is not taken here because it is
// worth ~6 us of a ~24 us hop — ~0.5 ms of a 77 ms token — and it would move
// every allocation in `DeepSeek4Runtime` into a shared context for that.  Two
// things measured along the way are worth knowing before anyone tries:
//   * a cross-DEVICE sycl::event is NOT a legal `depends_on`/`ext_oneapi_submit_
//     barrier` argument even inside one context — the Level Zero driver calls
//     abort() in cmdlist_hw.inl.  So the host barrier below cannot be removed;
//     it is not laziness, it is the only synchronisation this driver offers.
//   * a kernel on card 0 that dereferences card 1's DEVICE USM inside a shared
//     context does not fail — it reads ZEROS, silently.  P2P remains off
//     (`can_access_peer == 0` both directions, re-verified), and this is what
//     "off" looks like from inside a kernel.
std::string ds4_tp_reduce_host(const std::vector<sycl::queue*>& qs,
                               const std::vector<float*>& parts,
                               const std::vector<float*>& stages, uint64_t n);

// Single-buffer form: `stage` is `n_cards * n` floats of pinned host memory, and
// every queue must share ONE context (e.g. every "card" on one device).  Exactly
// the call above with `stages[c] = stage + c*n`.
std::string ds4_tp_reduce_host(const std::vector<sycl::queue*>& qs,
                               const std::vector<float*>& parts,
                               float* stage, uint64_t n);

// ---------------------------------------------------------------------------
// Ds4TpReducer — the prefill-size reduction, sliced (docs/deepseek4/72 Phase I)
// ---------------------------------------------------------------------------
//
// Same bytes as `ds4_tp_reduce_host` (every element is a0[i] + a1[i] in fp32,
// fanned out to both cards), different schedule for n >= 65536 and exactly two
// cards: the D2H is issued in slices of <= 1M floats on each compute queue, the
// host waits for one slice at a time and sums it on a small thread pool while
// the later slices are still landing, and each summed slice goes back on an
// AUXILIARY in-order queue of the same device; the compute queue then takes a
// barrier on those H2D events, so everything the caller enqueues on it next is
// ordered after the write exactly as before.  A same-device cross-queue event
// is the dependency the expert stream already runs on (compute queue waiting
// on the transfer queue's fetch); a cross-DEVICE event is never used.
//
// MEASURED before building (scratchpad/p2p/reduce_bench2.cpp, bit-identical to
// the memcpy form): 16 MB 3.07 -> 1.73-1.92 ms, 32 MB 6.14 -> 3.79, 8 MB 1.52
// -> 0.84; at decode sizes it is SLOWER (n = 8192: 74 vs 32 us), so below the
// size gate `reduce()` is `ds4_tp_reduce_host`, untouched.  The P2P form was
// measured the same morning and lost at every size on this bridge (peer
// pushes run at 5.5 GB/s per direction) — see the doc.
//
// Contract (in addition to ds4_tp_reduce_host's): `init()` before use, with
// the same queues `reduce()` will be called with; a caller that frees or
// resizes `stages[c]` drains `qs[c]` first — the barrier on `qs[c]` completes
// only after the auxiliary queue's H2D, so that one drain covers both.
// `IE_DS4_REDUCE_SLICED=0` routes every size through ds4_tp_reduce_host.
class Ds4TpReducer {
public:
    Ds4TpReducer() = default;
    ~Ds4TpReducer();
    Ds4TpReducer(const Ds4TpReducer&) = delete;
    Ds4TpReducer& operator=(const Ds4TpReducer&) = delete;

    // One auxiliary in-order queue per card (same device and context as qs[c])
    // and `n_threads` summing threads.  Empty string on success.
    std::string init(const std::vector<sycl::queue*>& qs, uint32_t n_threads = 6);
    // Joins the threads and drops the auxiliary queues; `init()` may follow.
    // DeepSeek4TpRuntime::release() calls it so a reload rebinds the queues.
    void reset();
    std::string reduce(const std::vector<sycl::queue*>& qs,
                       const std::vector<float*>& parts,
                       const std::vector<float*>& stages, uint64_t n);
    bool sliced() const noexcept { return sliced_; }

private:
    void sum_threaded(float* a0, float* a1, uint64_t n);
    void worker(uint32_t tid);

    std::vector<sycl::queue>  aux_;
    std::vector<std::thread>  threads_;
    std::mutex                m_;
    std::condition_variable   cv_;
    std::atomic<uint64_t>     gen_{0};
    std::atomic<uint32_t>     done_{0};
    std::atomic<bool>         stop_{false};
    float*                    job_a0_ = nullptr;
    float*                    job_a1_ = nullptr;
    uint64_t                  job_n_  = 0;
    uint32_t                  n_threads_ = 0;
    bool                      sliced_ = false;
};

// ---------------------------------------------------------------------------
// Tiered residency plan — VRAM first, host RAM only for the spill
// ---------------------------------------------------------------------------
//
// docs/deepseek4/33_vram_residency_design.md §2.2.  The VRAM expert arena is
// split into two kinds of slot:
//
//   STATIC    uploaded once at load, never evicted, HOST COPY NEVER ALLOCATED.
//   STREAMING a FIFO cache over the experts that did not fit, backed by pinned
//             host RAM.
//
// Keeping a pinned host copy of an expert that is permanently in VRAM is pure
// double-counting: it can never be fetched, so the bytes are dead. Freeing them
// is what turns "120.393 GB pinned" into "80.419 GB pinned" for Q3.
//
// WHICH experts are static is arbitrary, and that is a measured result rather
// than a convenience: the trained `exp_probs_b` bias is near-uniform (§6.4,
// centred spread 0.2-0.6%) and measured hit rate tracks residency to within one
// point across 9.4-100% (§3.1), so a static slot and a cache slot are equally
// likely to hold the expert you need. Slot s therefore holds expert s.
//
// Every field is derived from the REAL per-layer slot bytes, never from a layer
// count or a per-model constant: Q3 is 10,878,976 B/expert on 42 layers but
// 13,369,344 B on blk.26 (Unsloth UD mixed precision puts its gate/up in MXFP4,
// not IQ3_XXS), and Q8 is 13,369,344 B on all 43.
struct Ds4ResidencyPlan {
    uint32_t slots_per_layer  = 0;   // total VRAM slots per layer
    uint32_t static_slots     = 0;   // never evicted, no host copy
    uint32_t stream_slots     = 0;   // FIFO, backed by the pinned arena
    uint32_t pinned_experts   = 0;   // experts per layer WITH a pinned host copy
    uint64_t layer_slot_total = 0;   // Σ_L slot_bytes[L] — one slot in every layer
    uint64_t vram_bytes       = 0;   // slots_per_layer * layer_slot_total
    uint64_t host_bytes       = 0;   // pinned_experts   * layer_slot_total
};

// Plans the split.  `slots_cap == 0` means "up to n_experts"; `vram_budget_bytes
// == 0` means "do not let VRAM bind it".  Static slots are MAXIMISED (which
// minimises host bytes) subject to leaving `min_stream_slots` evictable.
//
// Returns "" on success.  Returns a diagnostic — and leaves `out` filled in, so
// the caller can print the numbers — when the spill cannot be brought under
// `host_pin_cap_bytes`.  It never quietly reduces coverage: an expert with
// neither a VRAM home nor a pinned host copy would have to be refused at the
// first token, hours after the load began, so it is refused here instead.
std::string ds4_plan_residency(const std::vector<uint64_t>& layer_slot_bytes,
                               uint32_t n_experts, uint64_t vram_budget_bytes,
                               uint32_t slots_cap, uint64_t host_pin_cap_bytes,
                               uint32_t min_stream_slots, Ds4ResidencyPlan& out);

// The same plan across `n_cards` cards under hidden-dim expert-TP.  Every card
// runs the IDENTICAL plan over its own half of every expert, so one
// `Ds4ResidencyPlan` describes all of them and the totals are per-card figures
// times `n_cards`.
//
// The pinned-host cap is a WHOLE-BOX resource — both cards' arenas come out of
// the same RAM — so `host_pin_cap_total` is divided by `n_cards` before the
// per-card plan is made, never applied per card.
struct Ds4TpResidencyPlan {
    uint32_t         n_cards          = 1;
    Ds4ResidencyPlan card{};             // what ONE card does; slot bytes are its slice
    uint64_t         card_slot_total  = 0;   // Σ_L this card's slot bytes
    uint64_t         whole_slot_total = 0;   // Σ_L a whole expert's slot bytes
    uint64_t         vram_bytes_total = 0;   // card.vram_bytes * n_cards
    uint64_t         host_bytes_total = 0;   // card.host_bytes * n_cards
    uint64_t         pool_bytes       = 0;   // n_experts * whole_slot_total
};

// `card_slot_bytes[L]` comes from `ds4_slot_layout_tp`, `whole_slot_bytes[L]`
// from `ds4_slot_layout`.  The planner REFUSES unless
// `n_cards * card_slot_bytes[L] == whole_slot_bytes[L]` for every layer: a slice
// that does not tile the expert exactly is a lost or duplicated byte, and it is
// caught here rather than as a wrong logit hours later.  `n_cards == 1` is the
// single-device case and produces exactly what `ds4_plan_residency` would.
std::string ds4_plan_residency_tp(const std::vector<uint64_t>& card_slot_bytes,
                                  const std::vector<uint64_t>& whole_slot_bytes,
                                  uint32_t n_experts, uint32_t n_cards,
                                  uint64_t vram_budget_per_card, uint32_t slots_cap,
                                  uint64_t host_pin_cap_total, uint32_t min_stream_slots,
                                  Ds4TpResidencyPlan& out);

// ---------------------------------------------------------------------------
// Residency priority as DATA — the profile file and the counter that writes it
// ---------------------------------------------------------------------------
//
// `Ds4Options::expert_priority` decides which experts win the static, never-
// evicted VRAM slots.  Static analysis cannot fill it in: the trained
// `exp_probs_b` balancing bias records how much correction the router NEEDED,
// not the residual imbalance, and its suppressed experts are layer-idiosyncratic
// (adjacent-layer Jaccard 0.006).  Utilisation is measured by COUNTING
// SELECTIONS, which needs a real forward pass.  These two pieces close that
// loop: `Ds4ExpertProfile` counts, and the file format below carries the count
// back into a later load with no code change.
//
// THE FILE FORMAT (`ds4-expert-priority 1`), by example:
//
//     # ds4-expert-priority 1
//     # any number of free-form comment lines, ignored by the reader
//     experts 256
//     # <expert id>  <total selections>
//     17 91204
//     3  90887
//     ...                       (exactly `experts` lines, a permutation)
//
// Reader rules, all enforced by name rather than repaired:
//   * blank lines, and lines whose first non-blank character is '#', are ignored
//   * exactly one `experts N` header is required and N must equal the model's
//     expert count — a ranking for a different model is refused, not truncated
//   * on every other line the FIRST whitespace-separated field is the expert id
//     and the rest of the line is ignored, so the counts a dump carries are
//     informative and never load-bearing
//   * the ids must be a full permutation of [0, N): a missing id would leave an
//     expert with neither a VRAM home nor a host copy, and the only place that
//     surfaces is a refusal mid-decode
// Order is most-important-first: line 1 gets static VRAM slot 0.
// The PER-LAYER form of the ranking file, and the one a decode-derived residency
// order needs: layer 4's hot experts are not layer 40's, so one permutation for
// the whole model throws away most of the signal (measured on the real model: a
// per-layer top-80 carries a median 90.8% of that layer's decode selections, one
// global top-80 carries 42.5%).
//
// ACCEPTS BOTH FORMATS, which is the whole point — a file with no `layers <n>`
// header is the existing single permutation and is replicated to every layer, so
// every ranking ever written by `ds4_expert_priority_write` keeps working and
// keeps meaning exactly what it meant.  With the header, the file carries
// `n_layers` blocks, each introduced by `layer <i>` and each a full permutation:
//
//     experts 256
//     layers 43
//     layer 0
//     <256 ids, most-selected first>
//     layer 1
//     ...
//
// A block count, a block index or a permutation that does not check out is a
// hard, named error — never a silent fall-through to index order, for the same
// reason the single-permutation reader refuses: "my measured ranking was quietly
// ignored" is precisely the failure a measurement loop cannot see from outside.
std::string ds4_expert_priority_read_layers(const std::string& path, uint32_t n_experts,
                                            uint32_t n_layers,
                                            std::vector<std::vector<uint32_t>>& out);

// Writes the per-layer form read by the above.  `counts` (optional) and `notes`
// are carried as comments so the measurement that produced the order is not lost
// at the point of capture.
std::string ds4_expert_priority_write_layers(const std::string& path,
                                             const std::vector<std::vector<uint32_t>>& orders,
                                             const std::vector<uint64_t>& counts,
                                             const std::vector<std::string>& notes);

std::string ds4_expert_priority_read(const std::string& path, uint32_t n_experts,
                                     std::vector<uint32_t>& out);

// Writes `order` in that format.  `counts`, when non-empty, must have one entry
// per expert INDEXED BY EXPERT ID (not by rank) and is emitted as the second
// column.  `note` lines are written as comments verbatim.
std::string ds4_expert_priority_write(const std::string& path,
                                      const std::vector<uint32_t>& order,
                                      const std::vector<uint64_t>& counts,
                                      const std::vector<std::string>& notes = {});

// Counts routed-expert selections per layer during a real forward pass.
//
// WHY PER LAYER when `expert_priority` is a single global order.  The residency
// mechanism is global (one static set, applied to every layer's arena), so the
// order this produces is the per-expert TOTAL.  The per-layer detail is kept
// anyway, and written into the dump as comments, because it is the measurement —
// discarding it at the point of capture would make a future per-layer residency
// scheme need a whole new profiling run.
class Ds4ExpertProfile {
public:
    // `n_layers`/`n_experts` == 0 is refused; a profile that silently counts
    // nothing is worse than one that never started.
    std::string init(uint32_t n_layers, uint32_t n_experts);
    bool        active() const noexcept { return n_experts_ != 0; }
    void        reset() noexcept;

    // One layer's routing decision: `ids[0..n)` are the selected expert ids for
    // however many tokens the caller has, flattened.  An id outside
    // [0, n_experts) is counted in `rejected()` and NOT clamped onto a valid
    // expert — a clamp would quietly invent utilisation for expert 0.
    //
    // `decode` SPLITS THE COUNTS BY PHASE, and it is not a nicety.  Residency is
    // chosen once at load and then has to serve both phases, but the two phases
    // want opposite things and prefill outnumbers decode: a `--ctx 4096
    // --decode 512` run records 1,155,840 prefill selections against 132,612
    // decode ones, so a blended histogram is 90% a prefill histogram.  MEASURED
    // consequence on the real model: the ranking derived from the blended counts
    // scores 31.16% static hit on decode, against 31.47% for plain INDEX ORDER.
    // It is worse than doing nothing, and nothing about the file it is written
    // to says so.  Counting the phases apart is what makes the measurement loop
    // produce a ranking for the phase it is going to be judged on.
    void record(uint32_t L, const int32_t* ids, uint32_t n, bool decode) noexcept;

    uint32_t n_layers()   const noexcept { return n_layers_; }
    uint32_t n_experts()  const noexcept { return n_experts_; }
    // `count`/`total` are over BOTH phases; the `_decode` forms are the ones a
    // residency ranking should be built from.
    uint64_t count(uint32_t L, uint32_t e) const noexcept;
    uint64_t total(uint32_t e) const noexcept;      // Σ over layers
    uint64_t count_decode(uint32_t L, uint32_t e) const noexcept;
    uint64_t selections()        const noexcept { return sel_; }
    uint64_t decode_selections() const noexcept { return dsel_; }
    uint64_t rejected()          const noexcept { return rej_; }

    // Most-selected first, ties broken by ASCENDING expert id.  That tie rule is
    // load-bearing: it makes an empty profile yield exactly the identity
    // permutation, so "profiling produced nothing" degrades to today's default
    // instead of to an arbitrary shuffle.
    //
    // Built from the DECODE counts when the run recorded any, and from the
    // blended counts only when it recorded none (a prefill-only run, where a
    // prefill ranking is the only thing on offer and is what was asked for).
    std::vector<uint32_t> priority() const;
    // The PER-LAYER form, `n_layers` permutations, each ranked by that layer's
    // own counts.  This is what the residency order actually wants: on the real
    // model each layer's own top 80 of 256 carries 83.8-95.8% of that layer's
    // decode selections, while one global top 80 carries 42.5%.
    std::vector<std::vector<uint32_t>> priority_layers() const;

    // Dumps `priority_layers()` in the per-layer file format, with `priority()`'s
    // global order and the per-layer counts retained as comments so nothing the
    // run measured is lost.
    std::string write(const std::string& path) const;

private:
    std::vector<uint64_t> c_;          // [n_layers * n_experts], BOTH phases
    std::vector<uint64_t> d_;          // [n_layers * n_experts], DECODE only
    uint32_t              n_layers_ = 0, n_experts_ = 0;
    uint64_t              sel_ = 0, rej_ = 0, dsel_ = 0;
};

// ---------------------------------------------------------------------------
// Segmented pinned host arena
// ---------------------------------------------------------------------------
class Ds4HostArena {
public:
    Ds4HostArena() = default;
    ~Ds4HostArena() { free_storage(); }
    Ds4HostArena(const Ds4HostArena&)            = delete;
    Ds4HostArena& operator=(const Ds4HostArena&) = delete;

    // `layer_slot_bytes[L]` is that layer's slot size (layers differ: blk.26 is
    // wider).  Allocates ceil(total / segment_target) pinned segments, each a
    // whole number of experts and each < kDs4MaxAllocBytes.  A segment that
    // fails to allocate is reported by name and size; nothing is silently
    // downsized and no pageable fallback is substituted.
    //
    // Pins EVERY expert.  Correct only when nothing is permanently VRAM-resident;
    // see init_range for the tiered case.
    std::string init(sycl::queue& q, const std::vector<uint64_t>& layer_slot_bytes,
                     uint32_t n_experts, uint64_t segment_target = kDs4HostSegmentTarget);

    // Pins only experts [first_expert, first_expert + n_pinned) of every layer.
    // The experts BELOW first_expert are the statically VRAM-resident ones, and
    // deliberately get no host backing at all — that is the storage this whole
    // class exists to not spend.  `n_experts` stays the model's full expert
    // count because it is what the cache directory is addressed by.
    // `slot()` returns nullptr for any expert outside the pinned range, which is
    // what makes "this expert has no host copy" a checkable fact rather than an
    // out-of-bounds read.
    std::string init_range(sycl::queue& q, const std::vector<uint64_t>& layer_slot_bytes,
                           uint32_t n_experts, uint32_t first_expert, uint32_t n_pinned,
                           uint64_t segment_target = kDs4HostSegmentTarget);

    // As init_range, but the pinned experts are an ARBITRARY SET rather than a
    // contiguous run — the general form, of which init_range is the special case
    // (and is implemented in terms of).
    //
    // WHY THIS EXISTS.  Which experts live permanently in VRAM used to be
    // "experts [0, static_slots)", i.e. chosen purely because of how they are
    // NUMBERED.  That is only defensible if utilisation is exactly uniform, and
    // nothing in this repo had measured that — the supporting benchmark ran on a
    // randomly-initialised model, where uniformity is true by construction and
    // therefore unfalsifiable.  Residency order is now an INPUT
    // (`Ds4Options::expert_priority`), so a measured order can be supplied
    // without touching this class again.  `pinned_ids` is the tail of that
    // priority order: the experts that did NOT win a static VRAM slot and so
    // need a pinned host copy.
    //
    // Duplicates and out-of-range ids are rejected by name — a repeated id would
    // silently alias two experts onto one slot, which is exactly the class of
    // bug that shows up only as a wrong logit.
    std::string init_set(sycl::queue& q, const std::vector<uint64_t>& layer_slot_bytes,
                         uint32_t n_experts, const std::vector<uint32_t>& pinned_ids,
                         uint64_t segment_target = kDs4HostSegmentTarget);

    // As init_set, but the pinned SET MAY DIFFER PER LAYER — the general form of
    // which init_set is the all-layers-alike special case.
    //
    // WHY.  init_set takes ONE id list and applies it to all 43 layers, so the
    // experts that win a permanent VRAM home are the same 80 ids in every layer.
    // That is only defensible if the layers agree about which experts are hot,
    // and on the real model they emphatically do not: measured on a 512-step
    // decode of DeepSeek-V4-Flash, each layer's own top-80 carries 83.8-95.8% of
    // that layer's selections (median 90.8%), while ONE global top-80 carries
    // 42.5% and index order 31.5%.  Layer 4's hot set is not layer 40's.
    //
    // COSTS NOTHING EXTRA.  Every layer still pins the SAME NUMBER of experts,
    // so the segment plan, the per-layer offsets and the total pinned bytes are
    // bit-for-bit what init_set produced — only WHICH expert lands in which slot
    // index changes, and that is a pure relabelling of `pin_index_`.  A ragged
    // set (layers pinning different counts) is refused rather than accommodated:
    // the layout's whole safety argument is that a layer is `n_pinned`
    // contiguous slots.
    std::string init_set_per_layer(sycl::queue& q, const std::vector<uint64_t>& layer_slot_bytes,
                                   uint32_t n_experts,
                                   const std::vector<std::vector<uint32_t>>& pinned_ids_per_layer,
                                   uint64_t segment_target = kDs4HostSegmentTarget);
    void free_storage() noexcept;

    void*    slot(uint32_t L, uint32_t e) const noexcept;
    uint64_t slot_bytes(uint32_t L) const noexcept { return L < slot_bytes_.size() ? slot_bytes_[L] : 0; }
    uint32_t n_layers()  const noexcept { return uint32_t(slot_bytes_.size()); }
    uint32_t n_experts() const noexcept { return n_experts_; }
    // Lowest expert id with a host copy IN LAYER 0 (== first_expert for a
    // contiguous init).  Layer 0 specifically, because the pinned set is now per
    // layer and "the lowest pinned id" is no longer a property of the model; it
    // is used only to make a refusal message concrete, never to address memory.
    uint32_t first_pinned_expert() const noexcept { return first_expert_; }
    uint32_t n_pinned_experts()    const noexcept { return n_pinned_; }
    // True iff expert `e` has a pinned host copy IN LAYER `L` — the set- and
    // layer-generalised form of "is e inside the pinned range", for callers that
    // must not assume contiguity or that the layers agree.
    bool     is_pinned(uint32_t L, uint32_t e) const noexcept {
        return L < slot_bytes_.size() && e < n_experts_ &&
               pin_index_[size_t(L) * n_experts_ + e] >= 0;
    }
    uint64_t total_bytes() const noexcept { return total_bytes_; }
    size_t   n_segments() const noexcept { return segs_.size(); }
    uint64_t segment_bytes(size_t i) const noexcept { return segs_[i].bytes; }
    // Largest single allocation actually requested — the number criterion 5 is about.
    uint64_t max_segment_bytes() const noexcept;

private:
    struct Seg {
        uint8_t* p     = nullptr;
        uint64_t bytes = 0;
        uint32_t first_layer = 0, n_layers = 0;   // whole layers only
    };
    sycl::queue*          q_ = nullptr;
    std::vector<Seg>      segs_;
    std::vector<uint64_t> slot_bytes_;      // per layer
    std::vector<uint32_t> layer_seg_;       // layer -> segment index
    std::vector<uint64_t> layer_off_;       // layer -> byte offset inside its segment
    // [layer][expert] -> slot index inside that layer's run, -1 = no host copy.
    // Was [expert]; it is per layer because which experts earn a permanent VRAM
    // home is a per-layer question (see init_set_per_layer).
    std::vector<int32_t>  pin_index_;
    uint32_t              n_experts_    = 0;   // model total, for addressing
    uint32_t              first_expert_ = 0;   // lowest expert id WITH a host copy
    uint32_t              n_pinned_     = 0;   // how many are backed here
    uint64_t              total_bytes_  = 0;
};

// ---------------------------------------------------------------------------
// Per-card VRAM expert cache
// ---------------------------------------------------------------------------
//
// Policy (docs/deepseek4/31 §2.2, verified reasoning rather than assumed):
// per-layer STATIC partition, FIFO inside a layer.  NOT a global LRU — the
// access pattern is cyclic (layer 0..42, then layer 0 again) which is the
// textbook sequential-flooding pathology for LRU, and `noaux_tc` + the trained
// `exp_probs_b` bias equalise expert utilisation, under which no policy that
// keeps the cache full beats residency.
//
// The directory is HOST-side.  That is free: the router's expert ids have to
// reach the host anyway to address the arena, exactly as in the existing
// gpt-oss / qwen3moe decode paths.
class Ds4ExpertCache {
public:
    Ds4ExpertCache() = default;
    ~Ds4ExpertCache() { free_storage(); }
    Ds4ExpertCache(const Ds4ExpertCache&)            = delete;
    Ds4ExpertCache& operator=(const Ds4ExpertCache&) = delete;

    // Allocates `slots_per_layer` slots for every layer, one device allocation
    // per layer (each far below kDs4MaxAllocBytes), and constructs the transfer
    // queue on `compute`'s context+device.  `vram_budget_bytes == 0` means "use
    // slots_per_layer as given"; otherwise slots_per_layer is REDUCED until the
    // total fits the budget, and the value actually used is reported by
    // slots_per_layer().  Fewer than 1 slot per layer is a hard error.
    //
    // Slots [0, static_slots) are the STATIC partition: install_static() fills
    // them once and nothing ever evicts them.  Replacement runs only over
    // [static_slots, slots_per_layer).  static_slots == 0 is a pure streaming
    // cache, which is the pre-tiering behaviour.
    // `bank_slots` = slots per prefill fetch bank (2 banks), 0 = off.  The
    // caller sizes them out of ITS VRAM budget before deriving slots_per_layer
    // (DeepSeek4Runtime::load_prepare); init only allocates, and if a bank's
    // allocation fails it drops the banks and keeps going (old grouped path).
    std::string init(sycl::queue& compute, const Ds4HostArena& arena,
                     uint32_t slots_per_layer, uint64_t vram_budget_bytes = 0,
                     uint32_t static_slots = 0, uint32_t bank_slots = 0);
    void free_storage() noexcept;

    // Uploads one slot's worth of bytes for layer `L` into static slot `s` and
    // records it as expert `e`, permanently.  Blocking: the caller's staging
    // buffer is free the moment this returns, and freeing it is the point.
    std::string install_static(uint32_t L, uint32_t e, uint32_t s, const void* src);

    bool         ready() const noexcept { return arena_ != nullptr; }
    sycl::queue& transfer_queue() noexcept { return *xq_; }
    uint32_t     slots_per_layer() const noexcept { return slots_; }
    uint32_t     static_slots() const noexcept { return static_; }
    uint32_t     stream_slots() const noexcept { return slots_ - static_; }
    uint64_t     device_bytes() const noexcept { return device_bytes_; }

    // Can expert `e` of layer `L` be served at all — i.e. is it resident now, or
    // does it have a pinned host copy to fetch from?  False means the only
    // honest response is to refuse; there is no fallback that is not a lie.
    bool available(uint32_t L, uint32_t e) const noexcept;
    // Is it resident in the never-evicted partition?  Such an expert costs no
    // streaming slot, which is what the caller's chunking bound turns on.
    bool is_static(uint32_t L, uint32_t e) const noexcept;
    // Is it resident right now (static or cached), i.e. would `acquire*` serve
    // it without a transfer?  What the hits-first prefill order turns on
    // (docs/deepseek4/72 Phase J).
    bool is_resident(uint32_t L, uint32_t e) const noexcept;

    // Makes experts `ids[0..n)` of layer `L` resident and writes their slot
    // indices to `out_slots`.  Misses are issued as H2D memcpys on the TRANSFER
    // queue; the returned event is the last of them (the queue is in-order, so
    // waiting on it covers every earlier one).  A call with no misses returns a
    // default-constructed (already-complete) event.
    //
    // An expert that CANNOT be served gets `kDs4NoSlot` and no transfer.  The
    // caller must check for it and refuse; it is never rounded down to slot 0.
    //
    // The compute submission that reads these slots MUST `depends_on()` the
    // returned event.  That is the whole DMA/compute overlap mechanism.
    sycl::event acquire(uint32_t L, const int32_t* ids, uint32_t n, uint32_t* out_slots);

    // ----------------------------------------------------------------------
    // THE 2-STAGE FETCH PIPELINE — why `acquire` alone leaves half the link idle
    // ----------------------------------------------------------------------
    //
    // MEASURED, and this is the defect the pipeline exists for.  On a pp512
    // prefill the transfer queue moves 60.7 GB per card in 2,499 ms of BUSY time
    // (24.3 GB/s, 92-95% of this box's per-card PCIe ceiling) inside a 5,518 ms
    // wall.  The link is therefore running at very nearly its hardware rate and
    // is IDLE 55% OF THE TIME.  The bandwidth is not the problem; the duty cycle
    // is.  The cause is structural, not a tuning matter: the caller's group loop
    // is `acquire(g) -> wait -> compute(g) -> drain -> acquire(g+1)`, so the DMA
    // engine and the EUs strictly alternate and neither ever runs during the
    // other.  A standalone probe of exactly that shape on this hardware
    // (scratchpad/stream_probe2.cpp, card 1, 192 slot fetches of 6,684,672 B
    // against a matched memory-bound kernel) reproduces it and measures the fix:
    //
    //     shape                 serial    1 group ahead   2 groups ahead
    //     48 groups x  4 slots  97.70 ms     50.76 ms        53.54 ms
    //     32 groups x  6 slots 101.43 ms     57.62 ms        55.38 ms
    //     24 groups x  8 slots 105.09 ms     59.63 ms        56.39 ms
    //     16 groups x 12 slots 103.02 ms     57.22 ms        56.83 ms   <- today
    //
    // 1.76-1.92x, and a THIRD stage buys nothing — two groups resident at once is
    // the whole win, so this is a double buffer and not a ring.  The same probe
    // measured the copy rate under concurrent compute at 28.34 GB/s against
    // 28.01 GB/s alone, i.e. the compute does not erode the transfer, which is
    // what makes the max() rather than the sum the reachable target.
    //
    // WHAT HAS TO BE TRUE FOR IT TO BE SAFE.  Group g+1's H2D lands while group
    // g's GEMMs are still reading their slots, so the replacement policy must not
    // hand group g+1 a slot group g is using.  `acquire` clears the layer's
    // in-use marks on entry, which is exactly right for the serial shape and
    // exactly wrong for this one.  `acquire_pipelined` keeps ONE generation of
    // marks: the slots the PREVIOUS call for this layer claimed stay protected
    // for the duration of this one.  Two generations is all that is needed, and
    // all that is sound, because the caller drains the compute queue at the end
    // of every group: when group g+1 is issued, group g-1's compute has
    // provably completed and its slots are free.
    //
    // The corollary is a hard capacity rule: TWO groups must fit the streaming
    // partition at once, so a group may claim at most `pipelined_group_cap()`
    // streaming slots.  `ds4_plan_expert_groups` below is the partitioner that
    // enforces it.  There is no silent widening — a group that cannot be served
    // gets kDs4NoSlot per expert, the same loud refusal as everywhere else.
    //
    // Returns the last H2D event of this group, exactly as `acquire` does.
    sycl::event acquire_pipelined(uint32_t L, const int32_t* ids, uint32_t n,
                                  uint32_t* out_slots);

    // Prefill fetch BANKS (docs/deepseek4/72 Phase A, 2026-09-01).  A prefill
    // group's misses are filled into one of two dedicated banks instead of
    // evicting LRU slots, and its hits read their LRU slot WITHOUT a recency
    // bump — so a 256-expert sweep neither thrashes the decode working set nor
    // starts the next prefill cold, and every resident expert is a free hit
    // (a prefill needs all of them anyway).  Bank slot ids are
    // kDs4BankSlotBase + bank*bank_slots() + j and resolve through slot_ptr().
    // `bank` alternates per group (g & 1); the caller's per-group compute
    // drain is what makes refilling a bank safe (write-after-read).
    // bank_slots() == 0 means banks are off ($DS4_PREFILL_BANKS=0, or they did not fit).
    static constexpr uint32_t kDs4BankSlotBase = 1u << 30;
    uint32_t    bank_slots() const noexcept { return bank_slots_; }
    sycl::event acquire_prefill(uint32_t L, const int32_t* ids, uint32_t n,
                                uint32_t* out_slots, uint32_t bank);


    // THE DECODE REGIME — WHAT THIS PIPELINE DOES AND DOES NOT DO AT T == 1
    // ----------------------------------------------------------------------
    // Everything above was measured on a PREFILL shape, where a chunk's `occ` is
    // most of the expert pool and a layer splits into many groups.  Decode is the
    // opposite shape and it defeats the pipeline structurally, not by degrees:
    // one token routes to `n_experts_used` == 6 experts, which is at or below
    // `pipelined_group_cap()` (6, at the shipped 12 streaming slots), so
    // `ds4_plan_expert_groups` returns EXACTLY ONE GROUP and the caller's
    // `if (g + 1 < NG) issue(g + 1)` is never taken.  Measured: 256 of 256
    // synthetic decode routings give NG == 1 (deepseek4_stream_gate_test §7a).
    //
    // FORCING MORE GROUPS DOES NOT FIX IT, AND THE REASON IS A RATIO, NOT A
    // TUNING.  Splitting the group only lets a fetch overlap the GEMVs of the
    // group before it, and at decode those GEMVs are the cheapest thing in the
    // layer: at M == 1 an expert GEMV reads its 6,684,672 B of weights from VRAM
    // exactly once, which this box does at 1,167 GB/s (0.0057 ms), against
    // 27.4 GB/s to fetch the same bytes over PCIe (0.2437 ms).  The consumer is
    // 43x faster than the producer, so the whole intra-layer overlap window is
    // ~2% of the fetch it would have to hide.  Measured across group caps 6/3/2/1
    // at decode shape, every difference was inside run-to-run noise.
    //
    // WHAT IS LEFT IS A HARD STRUCTURAL FACT.  The host cannot call `acquire`
    // until the router's expert ids are on the host, and getting them there
    // drains the compute queue.  So at the instant the H2D burst is issued there
    // is nothing enqueued for it to overlap except those same GEMVs.  Decode
    // expert DMA is therefore ~95% EXPOSED and no scheduling change inside this
    // class can hide it.  The only lever that shortens it is moving fewer bytes —
    // i.e. residency, which is `ds4_plan_residency` above and NOT this pipeline.

    // The largest number of STREAMING slots one group may claim if two groups
    // are to be resident at once.  Returns 0 when the streaming partition is too
    // small to double-buffer at all (fewer than 2 evictable slots), which the
    // caller must read as "run the serial shape" rather than as a cap of zero.
    uint32_t pipelined_group_cap() const noexcept {
        return stream_slots() < 2 ? 0u : stream_slots() / 2;
    }

    // Idle-time-only speculation (docs/deepseek4/31 §1.4 Rank 1): issues fetches
    // for `ids` ONLY if the transfer queue has no outstanding work, and never
    // evicts a slot the most recent acquire() for that layer is using.  A wrong
    // prediction therefore costs power and nothing else.  Returns how many
    // fetches it issued.
    //
    // MEASURED UTILITY IN PREFILL: NONE, and the reason is structural.  A pp512
    // run issued 1,900 speculative fetches for 18 hits (0.9%).  Two independent
    // causes, both fatal: the predictor is layer L's token-0 routing used for
    // layer L+1, and adjacent-layer expert overlap on this model is Jaccard
    // 0.006 (docs/deepseek4/33 §6.4); and even a correct prediction is evicted
    // long before it is needed, because layer L+1's own group loop churns every
    // streaming slot several times over before reaching the group that would
    // have hit.  At ~6.68 MB a fetch that is ~12.7 GB of link time bought with
    // nothing, ~9.5% of the prefill DMA budget.
    //
    // A HYPOTHESIS THAT WAS TESTED AND FALSIFIED, recorded so it is not tried
    // again.  The idle-time rule was expected to make this self-DISABLING once
    // `acquire_pipelined` kept the transfer queue busy.  IT DOES NOT: the caller
    // speculates AFTER draining the compute queue at the end of a group, and by
    // that instant the one group of lookahead has already landed, so the queue
    // is idle there in BOTH shapes.  Measured, identical counts
    // (deepseek4_stream_gate_test §6c): 36 speculative fetches issued serially,
    // 36 issued pipelined.
    //
    // The consequence is a live decision rather than a footnote.  Under the
    // pipeline the link is the binding resource, so these bytes come straight
    // off throughput: the same section measures the pipelined loop at 27.74 ms
    // with speculation against 20.33 ms without, for 56% extra link bytes.  On
    // the real model the ratio is 12.7 GB of speculation against 121.5 GB of
    // demand traffic, so ~10%.  `Ds4Options::prefetch` SHOULD BE OFF for prefill
    // once the pipeline lands.  This function is left honest and unchanged —
    // making it quietly refuse would be a silent policy change hiding in an API
    // whose name promises the opposite.  A speculation that does fire respects
    // both live generations of in-use marks.
    uint32_t speculate(uint32_t L, const int32_t* ids, uint32_t n);

    void*       slot_ptr(uint32_t L, uint32_t slot) noexcept;
    const void* slot_ptr(uint32_t L, uint32_t slot) const noexcept;

    // Resident expert id in `slot`, or -1.  Exposed so a test can prove the
    // directory and the bytes agree.
    int32_t tag(uint32_t L, uint32_t slot) const noexcept;

    struct Stats {
        uint64_t acquires      = 0;
        uint64_t hits          = 0;
        // `hits` SPLIT BY WHICH PARTITION SERVED IT.  static_hits + stream_hits
        // == hits, always.
        //
        // WHY THE SPLIT IS WORTH COUNTING.  A single hit rate cannot tell
        // "residency is holding the right experts" apart from "the streaming
        // cache is absorbing the misses", and those have OPPOSITE fixes: the
        // first is answered by changing the residency ORDER, the second by
        // changing the replacement policy or the split.  Worse, they trade off
        // against each other — a hotter static set removes hot experts from the
        // streaming pool, so the streaming term FALLS as the static term rises —
        // and a lone total can move the wrong way for the right reason, or the
        // right way for the wrong one.
        //
        // This was added after a measured A/B moved the total hit rate 0.6920 ->
        // 0.6815 under a residency order whose static term was independently
        // computed to RISE from 31.5% to 44.1%.  With only the total there was
        // no way to tell which of the two terms was wrong; with the split it is
        // one run and one subtraction.  Two counters on a branch that already
        // exists is a cheap price for never having that argument again.
        uint64_t static_hits   = 0;
        uint64_t stream_hits   = 0;
        uint64_t misses        = 0;
        uint64_t unavailable   = 0;   // ids that got kDs4NoSlot
        uint64_t bytes_fetched = 0;
        uint64_t spec_issued   = 0;
        uint64_t spec_bytes    = 0;
        uint64_t spec_hits     = 0;   // acquire hits on a slot a speculation filled
        double   dma_seconds   = 0.0; // from transfer-queue profiling events
        // True once collect_dma_time() has been asked for a measurement it cannot
        // make, i.e. the transfer queue was built without `enable_profiling`
        // (the default since 2026-08-03 — it costs ~0.42 us per submission).
        // `dma_seconds` then stays 0, which is NOT the same as "no DMA happened",
        // and every consumer must distinguish the two rather than reporting a
        // free-looking zero. Set IE_QUEUE_PROFILING=1 to get real numbers.
        bool     dma_unavailable = false;
        // How many of `acquires` came through acquire_pipelined.  Emitted so a
        // run can PROVE which shape it took: a pipelined build that silently
        // fell back to the serial one would otherwise look identical in every
        // other counter, and the whole 1.8x rests on the distinction.
        uint64_t pipelined     = 0;
    };
    const Stats& stats() const noexcept { return st_; }
    void         reset_stats() noexcept { st_ = Stats{}; }
    // Drains the profiling events accumulated so far into stats().dma_seconds.
    // Call after a token (it waits on the transfer queue).
    void collect_dma_time();

private:
    int32_t& slot_of(uint32_t L, uint32_t e) noexcept { return slot_of_[uint64_t(L) * n_experts_ + e]; }
    int32_t& tag_of(uint32_t L, uint32_t s) noexcept  { return tag_of_[uint64_t(L) * slots_ + s]; }
    // FIFO victim inside layer L's STREAMING partition, skipping slots this
    // acquire already claimed.  kDs4NoSlot when every streaming slot is claimed
    // (or there are none) — the old code clobbered a live slot instead.
    //
    // `iu` is this call's claims; `iu_prev`, when non-null, is the PREVIOUS
    // group's, still being read by the compute queue under the pipelined shape.
    // Passing null is the serial shape and picks exactly what it always did.
    uint32_t pick_victim(uint32_t L, const uint8_t* iu,
                         const uint8_t* iu_prev = nullptr) noexcept;
    // Shared body of acquire / acquire_pipelined.  `keep_prev` is the single
    // difference: false clears both generations of in-use marks (serial), true
    // rolls the current generation into the previous one (pipelined).
    sycl::event acquire_impl(uint32_t L, const int32_t* ids, uint32_t n,
                             uint32_t* out_slots, bool keep_prev);

    const Ds4HostArena*        arena_ = nullptr;
    sycl::queue*               cq_ = nullptr;
    std::vector<sycl::queue>   xq_store_;   // holds the transfer queue (queue is not default-constructible)
    sycl::queue*               xq_ = nullptr;
    std::vector<uint8_t*>      dev_;        // one allocation per layer
    std::vector<int32_t>       slot_of_;    // [n_layers * n_experts]  -1 = absent
    std::vector<int32_t>       tag_of_;     // [n_layers * slots]      -1 = empty
    std::vector<uint32_t>      fifo_;       // per-layer replacement cursor (FIFO policy)
    // LRU policy ($DS4_EVICT=lru; FreeToken review, docs/deepseek4/71 lever 1):
    // per-slot recency stamp from a per-layer tick, bumped on every hit and
    // fill; the victim is the least-recent streaming slot not in a live group.
    bool                       lru_ = true;
    uint32_t                   bank_slots_  = 0;      // slots per prefill bank (0 = off)
    uint64_t                   bank_stride_ = 0;      // widest slot bytes over layers
    uint8_t*                   bank_[2] = {nullptr, nullptr};
    std::vector<uint32_t>      fill_cursor_;   // [n_layers] next streaming slot to try for an EMPTY-slot fill
    std::vector<uint64_t>      stamp_;      // [n_layers * slots]
    std::vector<uint64_t>      tick_;       // [n_layers]
    void touch(uint32_t L, uint32_t s) noexcept { stamp_[uint64_t(L) * slots_ + s] = ++tick_[L]; }
    std::vector<uint8_t>       inuse_;      // [n_layers * slots] 1 if last acquire used it
    std::vector<uint8_t>       inuse_prev_; // [n_layers * slots] 1 if the acquire BEFORE it did
    std::vector<uint8_t>       from_spec_;  // [n_layers * slots] 1 if filled speculatively
    // THE FILL EVENT PER SLOT, AND THE BUG IT CLOSES.
    //
    // `acquire` used to return only the last MISS's memcpy event, on the
    // reasoning that the in-order transfer queue makes it cover every earlier
    // transfer in the same call.  That is true, and it is not enough: an id can
    // resolve to a HIT on a slot whose fill is STILL IN FLIGHT — `speculate`
    // publishes `slot_of`/`tag_of` at enqueue time, not at completion, and under
    // the pipeline a group's own fills are deliberately still running when the
    // group after it is issued.  If such a call then has no miss of its own it
    // returns a default-constructed (already-complete) event, the caller's
    // `depends_on` becomes a no-op, and the GEMM reads a slot mid-DMA.  That is
    // a silent wrong-weights bug, not a stall.
    //
    // So every slot carries the event that last wrote it plus a monotonic
    // sequence number, and an acquire returns the LATEST-sequence event among
    // the slots it hands out — exact rather than conservative.  Returning "the
    // most recent transfer overall" would also be safe on an in-order queue but
    // would make each group wait for the NEXT group's fetches, which is exactly
    // the overlap the pipeline exists to create.
    std::vector<sycl::event>   slot_ev_;    // [n_layers * slots] fill event
    std::vector<uint64_t>      slot_seq_;   // [n_layers * slots] 0 = never streamed
    uint64_t                   seq_ = 0;    // enqueue counter on the transfer queue
    std::vector<sycl::event>   pending_;    // profiling events not yet drained
    bool                       xq_prof_ = false;   // the transfer queue profiles (IE_QUEUE_PROFILING)
    // A fill's event. Every one is kept for collect_dma_time() only when the transfer queue profiles; otherwise only
    // the latest is needed (speculate()'s "anything outstanding" check). A caller that never collects -- the V4.1
    // tier -- grew pending_ by one event per miss for the life of the process: ~2.5 KB of runtime state each,
    // +6.3 MiB per 100-token served reply (+17 MiB with the CPU leg off, all misses over PCIe).
    void note_fill(const sycl::event& e) { if (!xq_prof_) pending_.clear(); pending_.push_back(e); }
    uint32_t                   n_layers_ = 0, n_experts_ = 0, slots_ = 0, static_ = 0;
    uint64_t                   device_bytes_ = 0;
    Stats                      st_{};
};

// ---------------------------------------------------------------------------
// Group partitioning for the fetch pipeline
// ---------------------------------------------------------------------------
//
// `occ[0..n)` is one layer's OCCUPIED routed-expert ids for one chunk, ascending
// and distinct — `moe_expert_grouped`'s `occ`.  This splits it into consecutive
// groups, each of which claims at most `stream_cap` STREAMING slots, and fills
// `bounds` with G+1 offsets so group g is `occ[bounds[g] .. bounds[g+1])`.
//
// Statically resident experts are never evicted, cost no slot, and are therefore
// not counted — the same rule the caller's inline loop already applied, moved
// here so the pipeline and the partition cannot drift apart.
//
// TWO PROPERTIES THIS MUST HAVE, AND BOTH ARE GATED:
//   * a group is NEVER empty, so an expert that on its own exceeds the cap still
//     reaches `acquire` and gets its named kDs4NoSlot refusal rather than being
//     dropped from the batch and silently omitted from the sum;
//   * the groups TILE `occ` exactly — bounds[0] == 0, bounds[G] == n, strictly
//     increasing — so no expert is computed twice and none is skipped.
//
// Passing `stream_cap == c.stream_slots()` reproduces the serial partition
// exactly; `c.pipelined_group_cap()` is the double-buffered one.
// Hits-first order for the bank path (docs/deepseek4/72 Phase J): the resident
// experts of `occ` first, in their given order, as ONE group that needs no
// transfer, then the rest in their given order cut by ds4_plan_expert_groups
// with `cap`.  `ordered` receives the permuted ids, `bounds` the group bounds
// over `ordered`; returns the number of leading hits (0 = no hits group, and
// then `bounds` is exactly the planner's).
uint32_t ds4_order_hits_first(const Ds4ExpertCache& c, uint32_t L, const uint32_t* occ,
                              size_t n, uint32_t cap, std::vector<uint32_t>& ordered,
                              std::vector<uint32_t>& bounds);

void ds4_plan_expert_groups(const Ds4ExpertCache& c, uint32_t L,
                            const uint32_t* occ, size_t n, uint32_t stream_cap,
                            std::vector<uint32_t>& bounds);

}  // namespace ie
