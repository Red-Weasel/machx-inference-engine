// src/core/allocator.cpp

#include <cstdlib>
#include "ie/allocator.hpp"

#include <algorithm>
#include <cstdio>
#include <mutex>
#include <string>
#include <vector>

namespace ie {

// Runtime defaults the engine relies on, applied with overwrite=0 so an explicit
// environment always wins. Runs once from a static initializer of this TU (any
// binary that links DeviceAllocator gets it before main) and again, harmlessly,
// from init()/init_with() in case the object was not pulled in.
//
//   SYCL_CACHE_PERSISTENT=1 — DPC++ on-disk cache of JIT-compiled device code.
//   Measured 2026-09-10 (Qwen3-4B Q4_K_M, ie-bench --prefill 64 --decode 8, one
//   B70): with this AND the driver's own compiler cache disabled the first
//   forward pays 5.2 s of JIT (PP 5205 ms vs 130 ms; wall 11.5 s vs 4.3 s). The
//   driver cache (NEO_CACHE_PERSISTENT, default on, ~/.cache/neo_compiler_cache,
//   1 GiB cap) normally hides that on a warm box; this is the second layer for
//   when it is evicted, disabled, or absent (fresh container, new driver).
//   NOT here: SYCL_UR_L0_RESTRICT_USM_RESIDENCY_TO_P2P. It was tried as a
//   default against the multi-device-context VRAM mirror (every device
//   allocation in a two-device context costs its size in host RAM, see
//   DeviceAllocator::init below) and measured inert on this stack (2026-09-11,
//   oneAPI 2026.1 / compute-runtime 26.22: 2 x 4 GiB in a two-device context ->
//   8 GiB of host RAM with the flag 0 or 1, peer access on or off; per-device
//   contexts -> 0.02 GiB). Only single-device contexts avoid the mirror.
void apply_runtime_defaults() {
    static std::once_flag once;
    std::call_once(once, [] {
        ::setenv("SYCL_CACHE_PERSISTENT", "1", /*overwrite=*/0);
    });
}
namespace {
struct RuntimeDefaultsInit { RuntimeDefaultsInit() { apply_runtime_defaults(); } };
const RuntimeDefaultsInit runtime_defaults_init_;
}  // namespace

bool gpu_p2p_available(const std::vector<sycl::device>& devs, const char* tag) {
    // Fewer than two devices is not a peer question: there is no pair to refuse,
    // so it is vacuously true and says nothing.  Answered BEFORE the cache, not
    // after — a single-device caller must not be handed a two-card verdict.
    if (devs.size() < 2) return true;
    static bool probed = false;
    static bool ok     = true;
    if (probed) return ok;
    probed = true;

    // BOTH directions of every pair.  Peer access is not symmetric by
    // definition, and the path that reads A's memory from B is not the one that
    // reads B's from A, so "usable" here means every card can reach every other.
    size_t fa = 0, fb = 0;
    ok = true;
    for (size_t a = 0; a < devs.size() && ok; ++a)
        for (size_t b = 0; b < devs.size(); ++b) {
            if (a == b) continue;
            bool can = false;
            // A backend that cannot answer throws rather than returning false.
            // An unanswerable peer question is a "no": the only safe thing to do
            // with one is take the host path.
            //
            // The copy is not a stray one: `ext_oneapi_can_access_peer` is not
            // declared const in oneAPI 2026.1, and `sycl::device` is a handle, so
            // copying it costs a refcount and keeps the caller's vector const.
            sycl::device from = devs[a];
            try {
                can = from.ext_oneapi_can_access_peer(
                    devs[b], sycl::ext::oneapi::peer_access::access_supported);
            } catch (const sycl::exception&) {
                can = false;
            }
            if (!can) { ok = false; fa = a; fb = b; break; }
        }

    if (ok)
        std::fprintf(stderr, "[%s] P2P available between all %zu devices\n", tag, devs.size());
    else
        std::fprintf(stderr,
                     "[%s] P2P unavailable between device %zu and %zu (can_access_peer == 0) "
                     "— using host-staged transfers\n",
                     tag, fa, fb);
    return ok;
}

bool gpu_p2p_shared_context(std::string_view name_filter,
                            std::vector<sycl::device>& gpus_out,
                            std::unique_ptr<sycl::context>& ctx_out) {
    if (const char* f = std::getenv("IE_GPU_FILTER"); f && *f) name_filter = f;
    std::vector<sycl::device> gpus;
    for (const auto& p : sycl::platform::get_platforms()) {
        if (p.get_info<sycl::info::platform::name>().find("Level-Zero") ==
            std::string::npos) continue;
        for (const auto& d : p.get_devices())
            if (d.is_gpu() &&
                (name_filter.empty() ||
                 d.get_info<sycl::info::device::name>().find(name_filter) !=
                     std::string::npos))
                gpus.push_back(d);
    }
    if (gpus.size() > 2) gpus.resize(2);
    if (gpus.size() < 2) return false;
    if (!gpu_p2p_available(gpus, "p2p-ctx")) return false;
    try {
        gpus[0].ext_oneapi_enable_peer_access(gpus[1]);
        gpus[1].ext_oneapi_enable_peer_access(gpus[0]);
        ctx_out = std::make_unique<sycl::context>(gpus);
    } catch (const sycl::exception& e) {
        std::fprintf(stderr, "[p2p-ctx] enable/context failed: %s — "
                     "using host-staged transfers\n", e.what());
        return false;
    }
    gpus_out = std::move(gpus);
    return true;
}

std::string DeviceAllocator::init(std::string_view name_filter, uint32_t ordinal) {
    // Env override: device names drift across drivers (the old "0xe223" stopped
    // matching the L0-V2 "Intel(R) Arc(TM) Pro B70 Graphics" name). IE_GPU_FILTER
    // lets the user re-point the filter without a rebuild.
    if (const char* f = std::getenv("IE_GPU_FILTER"); f && *f) name_filter = f;
    sycl::device picked;
    bool found = false;
    uint32_t seen = 0;   // count of matching GPUs so far
    for (const auto& d : sycl::device::get_devices()) {
        if (!d.is_gpu()) continue;
        if (name_filter.empty() ||
            d.get_info<sycl::info::device::name>().find(name_filter) != std::string::npos) {
            if (seen == ordinal) { picked = d; found = true; break; }
            ++seen;
        }
    }
    if (!found && ordinal == 0) {
        // Fall back to any GPU (only for the first device — ordinal>0 must match).
        for (const auto& d : sycl::device::get_devices()) {
            if (d.is_gpu()) { picked = d; found = true; break; }
        }
    }
    if (!found) return "no SYCL GPU device available at ordinal " + std::to_string(ordinal);

    try {
        const auto pci = picked.get_info<sycl::ext::intel::info::device::pci_address>();
        std::string boot = "/sys/bus/pci/devices/" + pci + "/boot_vga";
        FILE* f = std::fopen(boot.c_str(), "r");
        char bv = '0';
        if (f) { bv = static_cast<char>(std::fgetc(f)); std::fclose(f); }
        std::fprintf(stderr, "[gpu] ordinal %u  %s  pci %s%s\n",
                     ordinal, picked.get_info<sycl::info::device::name>().c_str(),
                     pci.c_str(),
                     bv == '1' ? "  DISPLAY (boot_vga) — compute+Xorg on this card "
                                 "is what pins 2800 MHz/C0 after a run"
                               : "");
    } catch (...) {}

    apply_runtime_defaults();
    try {
        // In-order queue: each submission implicitly depends on the previous,
        // so the model code can drop most per-kernel .wait() calls. Keeps
        // launch latency on a single thread of execution and lets the SYCL
        // runtime overlap host-side dispatch with device-side execution.
        //
        // enable_profiling is OPT-IN (IE_QUEUE_PROFILING=1). On the in-order
        // Level-Zero queue it adds large per-submit HOST overhead (~1.76 ms/kernel
        // measured) → decode is submission-bound, not GPU-bound (crown GPU-busy
        // 11.5 ms but wall 276 ms = 3.6 tok/s vs the ~81 tok/s GPU ceiling). Only
        // the kernel profiler / ie-bench --kprofile need it, so default OFF.
        //
        // EXPLICIT single-device context. sycl::queue(device) binds the queue to
        // the platform's DEFAULT context, which spans EVERY visible device — and
        // on this stack (oneAPI 2026.1 / compute-runtime 26.22 / xe) a device
        // allocation made in a multi-device context is given a resident
        // system-memory placement as well as its VRAM one: measured 2026-09-11,
        // 20 x 1 GiB or 1 x 20 GiB of malloc_device costs 20 GiB of host RAM
        // when both B70s are visible (ONEAPI_DEVICE_SELECTOR=level_zero:gpu),
        // 0.04 GiB when one is. That mirror was the "phantom" ~57 GiB every
        // two-card GLM / DeepSeek run lost (docs/glm53/DECODE_HOST_WAITS_
        // 2026-09-10.md) and the reason GLM's last stage-1 layers could not be
        // pinned. A context of exactly this device keeps VRAM in VRAM. Callers
        // that need one context across cards use init_with() and pay the mirror
        // knowingly.
        const sycl::context own_ctx(picked);
        if (std::getenv("IE_QUEUE_PROFILING"))
            queue_ = sycl::queue(own_ctx, picked,
                                 sycl::property_list{sycl::property::queue::in_order{},
                                                     sycl::property::queue::enable_profiling{}});
        else
            queue_ = sycl::queue(own_ctx, picked, sycl::property_list{sycl::property::queue::in_order{}});
    } catch (sycl::exception& e) {
        return std::string("queue creation failed: ") + e.what();
    }
    return {};
}

std::string DeviceAllocator::init_with(const sycl::context& ctx, const sycl::device& dev) {
    apply_runtime_defaults();
    try {
        // Same properties as init(): in-order; profiling opt-in (see init()).
        if (std::getenv("IE_QUEUE_PROFILING"))
            queue_ = sycl::queue(ctx, dev,
                                 sycl::property_list{sycl::property::queue::in_order{},
                                                     sycl::property::queue::enable_profiling{}});
        else
            queue_ = sycl::queue(ctx, dev, sycl::property_list{sycl::property::queue::in_order{}});
    } catch (sycl::exception& e) {
        return std::string("shared-context queue creation failed: ") + e.what();
    }
    return {};
}

namespace {
struct DevAllocRec { const void* key; const void* p; size_t n; uint32_t seq; std::string tag; };
std::mutex              g_da_mu;
std::vector<DevAllocRec> g_da;
thread_local std::string g_da_tag;
}   // namespace

bool dev_alloc_auditing() {
    static const bool on = std::getenv("IE_ALLOC_AUDIT") != nullptr;
    return on;
}
void dev_alloc_tag(const char* tag) {
    if (dev_alloc_auditing()) g_da_tag = tag ? tag : "";
}
void dev_alloc_record(const void* key, const void* p, size_t n) {
    std::lock_guard<std::mutex> lk(g_da_mu);
    g_da.push_back({key, p, n, uint32_t(g_da.size()), g_da_tag});
}
std::string dev_alloc_owner(const void* p) {
    std::lock_guard<std::mutex> lk(g_da_mu);
    const auto* a = static_cast<const uint8_t*>(p);
    std::string out;
    for (const DevAllocRec& r : g_da) {
        const auto* b = static_cast<const uint8_t*>(r.p);
        if (a >= b && a < b + r.n) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "%s#%u[%p +%llu of %llu]",
                          r.tag.empty() ? "?" : r.tag.c_str(), r.seq, r.p,
                          (unsigned long long)(a - b), (unsigned long long)r.n);
            if (!out.empty()) out += " AND ";
            out += buf;
        }
    }
    return out.empty() ? std::string("<no tracked allocation>") : out;
}
void dev_alloc_audit(const char* what) {
    if (!dev_alloc_auditing()) return;
    std::lock_guard<std::mutex> lk(g_da_mu);
    std::vector<const DevAllocRec*> v;
    for (const DevAllocRec& r : g_da) v.push_back(&r);
    std::sort(v.begin(), v.end(), [](const DevAllocRec* a, const DevAllocRec* b) {
        if (a->key != b->key) return a->key < b->key;
        return a->p < b->p;
    });
    std::fprintf(stderr, "[alloc-audit] %s: %zu device allocations\n", what, v.size());
    uint64_t noverlap = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        const auto* b = static_cast<const uint8_t*>(v[i]->p);
        const char* rel = "";
        char gap[64] = "";
        if (i && v[i - 1]->key == v[i]->key) {
            const auto* pb = static_cast<const uint8_t*>(v[i - 1]->p);
            const std::ptrdiff_t d = b - (pb + v[i - 1]->n);
            if (d < 0) { rel = "  *** OVERLAPS PREVIOUS ***"; ++noverlap; }
            std::snprintf(gap, sizeof gap, " gap %+lld", (long long)d);
        }
        std::fprintf(stderr, "[alloc-audit]   dev %p  %p .. %p  %10.3f MiB  #%u %s%s%s\n",
                     v[i]->key, v[i]->p, (const void*)(b + v[i]->n),
                     double(v[i]->n) / 1048576.0, v[i]->seq,
                     v[i]->tag.empty() ? "?" : v[i]->tag.c_str(), gap, rel);
    }
    std::fprintf(stderr, "[alloc-audit] %s: %llu overlapping pairs\n", what,
                 (unsigned long long)noverlap);
}

void* DeviceAllocator::malloc(size_t nbytes) {
    if (!queue_) return nullptr;
    void* p = sycl::malloc_device(nbytes, *queue_);
    if (p && dev_alloc_auditing()) dev_alloc_record(this, p, nbytes);
    return p;
}

void DeviceAllocator::free(void* p) noexcept {
    if (!queue_ || !p) return;
    sycl::free(p, *queue_);
}

Tensor DeviceAllocator::alloc_tensor(DType dtype, std::span<const uint64_t> shape) {
    Tensor t{};
    if (shape.empty() || shape.size() > kMaxDims) return t;
    t.dtype = dtype;
    t.device = Device::kGpu;
    t.n_dims = static_cast<uint32_t>(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) t.shape[i] = shape[i];
    uint64_t row_elems = shape[0];
    uint64_t hi = 1;
    for (size_t i = 1; i < shape.size(); ++i) hi *= shape[i];
    t.nbytes = bytes_for(dtype, row_elems) * hi;
    if (t.nbytes == 0) return Tensor{};
    t.data = malloc(t.nbytes);
    if (!t.data) return Tensor{};
    return t;
}

void DeviceAllocator::free_tensor(Tensor& t) noexcept {
    if (t.device == Device::kGpu) {
        free(t.data);
    }
    t.data = nullptr;
    t.nbytes = 0;
    t.n_dims = 0;
}

// ---- DeviceFleet (P-A multi-GPU) --------------------------------------
std::string DeviceFleet::init(uint32_t n_request, std::string_view name_filter,
                              bool enable_p2p_transfers, bool shared_ctx) {
    if (n_request == 0) return "DeviceFleet: n_request == 0";
    if (const char* f = std::getenv("IE_GPU_FILTER"); f && *f) name_filter = f;  // see DeviceAllocator::init
    // Count matching GPUs so we bind min(n_request, available).
    uint32_t available = 0;
    for (const auto& d : sycl::device::get_devices()) {
        if (!d.is_gpu()) continue;
        if (name_filter.empty() ||
            d.get_info<sycl::info::device::name>().find(name_filter) != std::string::npos)
            ++available;
    }
    if (available == 0) return "DeviceFleet: no GPU matches filter '" +
                                std::string(name_filter) + "'";
    const uint32_t n = std::min(n_request, available);
    devs_.clear();
    devs_.resize(n);   // DeviceAllocator default-constructs (no queue yet)
    for (uint32_t i = 0; i < n; ++i)
        if (auto e = devs_[i].init(name_filter, i); !e.empty())
            return "DeviceFleet dev " + std::to_string(i) + ": " + e;
    // Multi-GPU: rebind every queue into ONE shared sycl::context (2026-08-16).
    // Host USM from any fleet queue becomes legally dereferenceable by every
    // fleet device, and events chain across queues — the prerequisites for the
    // zero-copy all-reduce. IE_NO_SHARED_CTX=1 falls back to per-device contexts
    // (and thereby the host-staged AR) if a driver issue ever surfaces.
    //
    // Only when asked (`shared_ctx`): on this stack every device allocation made
    // in a two-device context is mirrored 1:1 into host RAM (see
    // DeviceAllocator::init), so a shared context costs the fleet its whole
    // weight set in RAM. The tensor-parallel paths need it — their all-reduce
    // pinned-slot exchange is what keeps 27B TP prefill at 503 tok/s instead of
    // 280 host-staged (measured 2026-09-11, 16K prompt) — the pipeline splits do
    // not: they only copy_across (host bounce), and the 27B split measured
    // identical prefill/decode/text with per-device contexts and 29 GiB less
    // host RAM taken at load.
    if (n > 1 && shared_ctx && !std::getenv("IE_NO_SHARED_CTX")) {
        try {
            std::vector<sycl::device> bound;
            bound.reserve(n);
            for (uint32_t i = 0; i < n; ++i) bound.push_back(devs_[i].device());
            sycl::context ctx(bound);
            for (uint32_t i = 0; i < n; ++i)
                if (auto e = devs_[i].init_with(ctx, bound[i]); !e.empty())
                    return "DeviceFleet dev " + std::to_string(i) + ": " + e;
            shared_ctx_ = ctx;
        } catch (const sycl::exception& e) {
            std::fprintf(stderr, "[gpu] shared context unavailable (%s) — per-device contexts\n",
                         e.what());
            shared_ctx_.reset();
        }
    }
    // Says which transfer path this fleet is on, once, before anything uses it.
    // `copy_across` and `all_reduce_sum_fp16` below stage through host memory
    // unconditionally, so a false answer here does not change what runs — it is
    // what makes the cost of that path explicable instead of just slow.
    if (n > 1) {
        std::vector<sycl::device> bound;
        bound.reserve(n);
        for (uint32_t i = 0; i < n; ++i) bound.push_back(devs_[i].device());
        p2p_ = gpu_p2p_available(bound, "gpu") && enable_p2p_transfers;
        if (std::getenv("IE_NO_P2P")) p2p_ = false;   // kill switch
        if (p2p_) {
            // Enable peer access on every ordered pair so cross-device USM memcpy
            // goes direct over PCIe (live 2026-08-15 via the host-bridge whitelist
            // kernel patch). Any failure drops back to the host-bounce paths.
            try {
                for (uint32_t a = 0; a < n; ++a)
                    for (uint32_t b = 0; b < n; ++b)
                        if (a != b)
                            devs_[a].device().ext_oneapi_enable_peer_access(devs_[b].device());
            } catch (const sycl::exception& e) {
                std::fprintf(stderr, "[gpu] peer-access enable failed (%s) — host-staged\n",
                             e.what());
                p2p_ = false;
            }
        }
    }
    return {};
}

void DeviceFleet::copy_across(uint32_t si, void* dst, uint32_t di,
                              const void* src, size_t nbytes) {
    if (si == di) {                          // same device — direct copy
        devs_[di].queue().memcpy(dst, src, nbytes).wait();
        return;
    }
    if (p2p_) {
        // Direct GPU→GPU as a PUSH from the SOURCE queue. Two reasons, both
        // measured 2026-08-16 on the 2×B70/whitelist-patched client bridge:
        // (1) peer READS return garbage on this root complex (standalone test:
        // 100% corrupt pulls, 0% corrupt pushes — posted writes route, read
        // completions do not); (2) the source's in-order queue naturally orders
        // the copy after the producer kernels — no cross-queue race.
        devs_[si].queue().memcpy(dst, src, nbytes).wait();
        return;
    }
    // Host bounce: src(dev si) → host → dst(dev di). Robust on any board; the
    // layer-split boundary copy is small so this is cheap.
    std::vector<unsigned char> host(nbytes);
    devs_[si].queue().memcpy(host.data(), src, nbytes).wait();
    devs_[di].queue().memcpy(dst, host.data(), nbytes).wait();
}

void DeviceFleet::ensure_ar_pin(uint64_t n_elem) {
    if (n_elem <= ar_cap_ && !ar_pin_.empty()) return;
    // Growing the ring frees the old pinned buffers. The zero-copy AR leaves
    // consumers of those buffers IN FLIGHT (no host wait per AR), so drain every
    // queue before freeing — rare event (capacity only ever steps up: decode H,
    // then verify T×H), and skipping the drain is use-after-free under async AR.
    for (auto& d : devs_) {
        try { d.queue().wait(); } catch (const sycl::exception&) {}
    }
    free_ar_pin();
    const uint32_t nd = static_cast<uint32_t>(devs_.size());
    ar_pin_.assign(kArGen, std::vector<sycl::half*>(nd, nullptr));
    for (uint32_t gtmp = 0; gtmp < kArGen; ++gtmp)
        for (uint32_t d = 0; d < nd; ++d)
            ar_pin_[gtmp][d] = sycl::malloc_host<sycl::half>(n_elem, devs_[d].queue());
    ar_acc_.assign(n_elem, 0.0f);
    ar_cap_ = n_elem;
    ar_gen_ = 0;
}

void DeviceFleet::free_ar_pin() {
    for (uint32_t gtmp = 0; gtmp < ar_pin_.size(); ++gtmp)
        for (uint32_t d = 0; d < ar_pin_[gtmp].size() && d < devs_.size(); ++d)
            if (ar_pin_[gtmp][d]) sycl::free(ar_pin_[gtmp][d], devs_[d].queue());
    ar_pin_.clear();
    ar_cap_ = 0;
}

DeviceFleet::~DeviceFleet() {
    // Drain any in-flight async scatters before freeing the pinned staging.
    //
    // THE DRAIN CAN THROW, AND THIS IS A DESTRUCTOR.  A queue whose device the
    // Level-Zero runtime has already declared lost fails its wait with a
    // sycl::exception, and an exception leaving a destructor is std::terminate —
    // the process dies with `terminate called after throwing an instance of
    // 'sycl::_V1::exception'` and the ACTUAL failure, which happened earlier and
    // elsewhere, is gone. Report it and carry on tearing down instead: the pinned
    // pages still have to be released, and a lost device cannot be made to
    // finish the work either way.
    for (auto& d : devs_) {
        try {
            d.queue().wait();
        } catch (const sycl::exception& e) {
            std::fprintf(stderr, "[gpu] DeviceFleet teardown: queue drain failed: %s\n", e.what());
        }
    }
    free_ar_pin();
    if (ar2_scratch_ && !devs_.empty()) {
        try { sycl::free(ar2_scratch_, devs_[0].queue()); } catch (const sycl::exception&) {}
        ar2_scratch_ = nullptr; ar2_cap_ = 0;
    }
    for (uint32_t d = 0; d < 2 && d < devs_.size(); ++d)
        if (zc_scr_[d]) { devs_[d].free(zc_scr_[d]); zc_scr_[d] = nullptr; }
    zc_scr_cap_ = 0;
}

void DeviceFleet::all_reduce_sum_fp16(const std::vector<sycl::half*>& bufs,
                                      uint64_t n_elem) {
    const uint32_t nd = static_cast<uint32_t>(bufs.size());
    if (nd <= 1 || n_elem == 0) return;   // nothing to reduce
    // TIMING PROBE ONLY (output is mathematically wrong): IE_AR_SKIP=1 makes the
    // all-reduce a no-op so the tok/s delta prices the AR's total overhead.
    static const bool ar_skip = []{
        const char* e = std::getenv("IE_AR_SKIP");
        return e && std::atoi(e) == 1;
    }();
    if (ar_skip) return;
    const size_t bytes = n_elem * sizeof(sycl::half);

    // 2-device P2P fast path (2026-08-15): dev0 pulls dev1's partial DIRECT over
    // PCIe, sums on-GPU as half(f32(a)+f32(b)) — the same value the host loop
    // produces for nd=2 (fp32 accumulate in device order) — then dev1 pulls the
    // result back DIRECT. Removes both host DMA hops and the CPU sum from the
    // per-layer TP hot path.
    if (p2p_ && nd == 2) {
        auto& q0 = devs_[0].queue();
        if (n_elem > ar2_cap_) {
            if (ar2_scratch_) sycl::free(ar2_scratch_, q0);
            ar2_scratch_ = sycl::malloc_device<sycl::half>(n_elem, q0);
            ar2_cap_ = ar2_scratch_ ? n_elem : 0;
        }
        if (ar2_scratch_) {
            // PUSH-ONLY choreography (2026-08-16: peer READS are broken on this
            // bridge — standalone test 100% corrupt pulls, 0% corrupt pushes;
            // this was the root cause of a full night of garbled TP output that
            // byte-gates missed because BOTH TP modes emitted the same garbage):
            //   1. q1 PUSHES dev1's partial into dev0 scratch (q1 in-order =
            //      after dev1's producer — ordering for free).
            //   2. q0 sums on dev0 (depends_on the push; q0 in-order covers d0's
            //      own producer). Same d0+d1 fp32 order as the host loop.
            //   3. q0 PUSHES the sum into bufs[1] (write direction — works).
            //   4. q1 barrier depends on the push-back so dev1's next consumer
            //      is ordered. Host does not block.
            sycl::half* d0 = bufs[0];
            sycl::half* scr = ar2_scratch_;
            auto& q1 = devs_[1].queue();
            const sycl::event push = q1.memcpy(scr, bufs[1], bytes);   // dev1 → dev0 (PUSH)
            const sycl::event sum = q0.submit([&](sycl::handler& h) {
                h.depends_on(push);
                h.parallel_for(sycl::range<1>(n_elem), [=](sycl::id<1> i) {
                    d0[i] = sycl::half(float(d0[i]) + float(scr[i]));
                });
            });
            const sycl::event back = q0.submit([&](sycl::handler& h) {
                h.depends_on(sum);
                h.memcpy(bufs[1], bufs[0], bytes);             // dev0 → dev1 (PUSH)
            });
            q1.ext_oneapi_submit_barrier({back});
            return;
        }
        // scratch alloc failed → fall through to the host-staged path
    }
    ensure_ar_pin(n_elem);
    sycl::half** stage = ar_pin_[ar_gen_].data();   // this generation's per-dev buffers
    ar_gen_ = (ar_gen_ + 1) % kArGen;

    // ZERO-COPY shared-context path (2026-08-16), nd==2: each device DMAs its
    // partial into its pinned slot, then each device's SUM KERNEL reads the
    // PEER's slot directly (legal now: one shared context) with a cross-queue
    // event dependency — NO host wait anywhere, so the host thread keeps
    // submitting layers ahead instead of blocking per all-reduce.
    //   Bit-identity: host loop computes half(f32(d0)+f32(d1)); dev0 computes
    //   f(b0)+f(s1), dev1 computes f(b1)+f(s0) — same two operands, and IEEE
    //   fp32 addition of two operands is commutative → identical halves.
    //   WAR across rounds: generation ring + the cross-queue dependency chain
    //   (my DMA#n+4 ≥ my sum#n+3 ≥ peer DMA#n+3 ≥ peer sum#n+2 ≥ peer sum#n).
    // History: this variant WITHOUT the shared context was falsified 2026-08-16
    // (host USM registered in dev1's context dereferenced by dev0's kernel is UB
    // on Level Zero) — the shared-context fleet is what makes it legal.
    // DEFAULT OFF (IE_AR_ZC=1 re-enables for future driver stacks): BOTH async
    // variants were falsified on this board 2026-08-16 — pure-EU peer-slot reads
    // gave tau=1.000 (total corruption) and the copy-engine hop below still gave
    // tau~1.1 (probabilistic race). Together with the P2P night this pins the
    // real defect: CROSS-QUEUE EVENT DEPENDENCIES are not reliably honored on
    // this driver/board; only host-mediated sync is trustworthy. The code stays
    // as the reference implementation for hardware where they work.
    static const bool ar_zc_on = []{
        const char* e = std::getenv("IE_AR_ZC");
        return e && std::atoi(e) == 1;
    }();
    if (shared_ctx_ && nd == 2 && ar_zc_on) {
        // COPY-ENGINE-ONLY choreography. The pure-EU variant (dev0 kernel
        // directly summing dev1's pinned slot) was FALSIFIED here 2026-08-16
        // even WITH the shared context: tau collapsed to 1.000 (total
        // corruption) — EU loads of host USM freshly DMA-written by the OTHER
        // device return garbage on this board, while a copy-engine H2D hop of
        // the same buffer is clean. So: peer slot → own DEVICE scratch by DMA
        // (cross-queue event dep), then sum from device memory. Still no host
        // wait anywhere.
        if (n_elem > zc_scr_cap_) {
            for (auto& d : devs_) {
                try { d.queue().wait(); } catch (const sycl::exception&) {}
            }
            for (uint32_t d = 0; d < 2; ++d) {
                if (zc_scr_[d]) devs_[d].free(zc_scr_[d]);
                zc_scr_[d] = static_cast<sycl::half*>(
                    devs_[d].malloc(n_elem * sizeof(sycl::half)));
            }
            zc_scr_cap_ = (zc_scr_[0] && zc_scr_[1]) ? n_elem : 0;
        }
        if (zc_scr_cap_ >= n_elem) {
            auto& q0 = devs_[0].queue();
            auto& q1 = devs_[1].queue();
            sycl::half* b0 = bufs[0];    sycl::half* b1 = bufs[1];
            sycl::half* s0 = stage[0];   sycl::half* s1 = stage[1];
            sycl::half* z0 = zc_scr_[0]; sycl::half* z1 = zc_scr_[1];
            const sycl::event g0 = q0.memcpy(s0, b0, bytes);   // own partial → own slot
            const sycl::event g1 = q1.memcpy(s1, b1, bytes);
            q0.submit([&](sycl::handler& h) {                  // peer slot → own scratch
                h.depends_on(g1); h.memcpy(z0, s1, bytes); });
            q1.submit([&](sycl::handler& h) {
                h.depends_on(g0); h.memcpy(z1, s0, bytes); });
            q0.submit([&](sycl::handler& h) {                  // sum from DEVICE memory
                h.parallel_for(sycl::range<1>(n_elem), [=](sycl::id<1> i) {
                    b0[i] = sycl::half(float(b0[i]) + float(z0[i]));
                });
            });
            q1.submit([&](sycl::handler& h) {
                h.parallel_for(sycl::range<1>(n_elem), [=](sycl::id<1> i) {
                    b1[i] = sycl::half(float(b1[i]) + float(z1[i]));
                });
            });
            return;
        }
        // scratch alloc failed → fall through to the host-staged path
    }

    // Gather ALL devices concurrently (overlap the per-card D→H DMAs) into pinned
    // staging, then wait — the consumer (CPU sum or device sum) must see final data.
    std::vector<sycl::event> ev(nd);
    for (uint32_t d = 0; d < nd; ++d)
        ev[d] = devs_[d].queue().memcpy(stage[d], bufs[d], bytes);
    for (uint32_t d = 0; d < nd; ++d) ev[d].wait();

    // DEVICE-SUM path (2026-08-16, nd==2 + shared context): after the host
    // gather waits (which are the ONLY cross-device ordering this board can be
    // trusted with — cross-queue event deps are falsified above), each device
    // DMAs the PEER's staged partial into its own scratch and sums ON-DEVICE.
    // Kills the scalar CPU fp32 sum + writeback loops, which at prefill
    // (n_elem = chunk×H ≈ 2.6M) cost 5-15 ms per all-reduce — measured 25.6%
    // of TP prefill wall. Ordering: every submission here is on the consuming
    // device's own in-order queue, AFTER the host observed both gathers; ring
    // wraparound is safe because AR #n's H2D precedes (in-order) that queue's
    // #n+1 gather, which the host waited before returning from call #n+1.
    // Bit-identity: stage[d] is a byte-copy of bufs[d], so half(f32(b_d) +
    // f32(s_peer)) uses the same two operands as the CPU loop's d0+d1 fp32
    // accumulate — IEEE addition of two operands is commutative → identical.
    if (shared_ctx_ && nd == 2) {
        if (n_elem > zc_scr_cap_) {
            for (auto& d : devs_) {
                try { d.queue().wait(); } catch (const sycl::exception&) {}
            }
            for (uint32_t d = 0; d < 2; ++d) {
                if (zc_scr_[d]) devs_[d].free(zc_scr_[d]);
                zc_scr_[d] = static_cast<sycl::half*>(
                    devs_[d].malloc(n_elem * sizeof(sycl::half)));
            }
            zc_scr_cap_ = (zc_scr_[0] && zc_scr_[1]) ? n_elem : 0;
        }
        if (zc_scr_cap_ >= n_elem) {
            for (uint32_t d = 0; d < 2; ++d) {
                auto& q = devs_[d].queue();
                sycl::half* b = bufs[d];
                sycl::half* sp = stage[1 - d];    // PEER's gathered partial
                sycl::half* z = zc_scr_[d];
                q.memcpy(z, sp, bytes);           // in-order; gathers already waited
                q.submit([&](sycl::handler& h) {
                    h.parallel_for(sycl::range<1>(n_elem), [=](sycl::id<1> i) {
                        b[i] = sycl::half(float(b[i]) + float(z[i]));
                    });
                });
            }
            return;
        }
        // scratch alloc failed → CPU-sum fallback below
    }

    // fp32 accumulate (device order 0..nd → BIT-IDENTICAL to the prior version),
    // then write the sum back into each device's own pinned buffer for the scatter.
    std::fill_n(ar_acc_.begin(), n_elem, 0.0f);
    for (uint32_t d = 0; d < nd; ++d) {
        const sycl::half* s = stage[d];
        for (uint64_t i = 0; i < n_elem; ++i) ar_acc_[i] += float(s[i]);
    }
    for (uint32_t d = 0; d < nd; ++d) {
        sycl::half* s = stage[d];
        for (uint64_t i = 0; i < n_elem; ++i) s[i] = sycl::half(ar_acc_[i]);
    }

    // Scatter the sum back to ALL devices — ASYNC: NO host .wait(). Each device's
    // queue is in-order, so its next-submitted consumer (residual_add on the same
    // queue) runs after this scatter; the generation ring keeps the host from
    // overwriting a buffer whose DMA is still in flight. Removes one host-device
    // sync barrier per all-reduce (the per-layer decode hot path). Math unchanged.
    for (uint32_t d = 0; d < nd; ++d)
        devs_[d].queue().memcpy(bufs[d], stage[d], bytes);
}

}  // namespace ie
