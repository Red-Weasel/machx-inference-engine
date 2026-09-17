// src/model/deepseek41_experts.cpp — see the header.
#include "ie/deepseek41_experts.hpp"
#include "ie/cpu_moe_mxfp4.hpp"

#include <cmath>
#include <sched.h>
#include "ie/deepseek41_upload.hpp"
#include "ie/deepseek4_ops.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <dlfcn.h>
#include <fcntl.h>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <immintrin.h>
#include <omp.h>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>

#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

namespace ie {

namespace {
inline uint64_t align64(uint64_t x) { return (x + 63) & ~uint64_t(63); }
// Readers in flight over the staging slots: NVMe queue depth for the mmap tier and the load.
constexpr uint32_t kReaders = 8;

constexpr uint64_t kPage = 4096;
// Read file bytes [off, off+len) into `buf`: O_DIRECT through `dfd` when it exists (the aligned
// superset lands in buf, and *at says where the wanted bytes start), buffered `fd` otherwise
// (*at = 0). O_DIRECT reads may come back short only at end of file, so the loop stops once
// the wanted bytes are in.
std::string read_range(int fd, int dfd, void* buf, uint64_t off, uint64_t len, uint64_t* at) {
    uint8_t* d = static_cast<uint8_t*>(buf);
    uint64_t rd_off = off, want = len; *at = 0;
    if (dfd >= 0) {
        const uint64_t aoff = off & ~(kPage - 1), aend = (off + len + kPage - 1) & ~(kPage - 1);
        rd_off = aoff; want = aend - aoff; *at = off - aoff; fd = dfd;
    }
    const uint64_t need = *at + len;                       // bytes that must be present
    uint64_t got = 0;
    while (got < want) {
        const ssize_t r = ::pread(fd, d + got, size_t(want - got), off_t(rd_off + got));
        if (r < 0) { if (errno == EINTR) continue; return std::string("pread: ") + std::strerror(errno); }
        if (r == 0) { if (got >= need) break; return "pread: unexpected end of file"; }
        got += uint64_t(r);
    }
    return {};
}
// the Phase 4 shuffle, src -> dst over whole planes: engine_byte[j] = nib(v[j/2], j&1) | nib(v[j/2+8], j&1) << 4
inline void permute_plane_ref(const uint8_t* src, uint8_t* qs, uint64_t len0) {   // the scalar reference
    for (uint64_t blk = 0; blk < len0 / 16; ++blk) {
        const uint8_t* v = src + blk * 16; uint8_t* o = qs + blk * 16;
        for (int j = 0; j < 16; ++j) {
            const int sh = (j & 1) * 4;
            o[j] = uint8_t(((v[j >> 1] >> sh) & 0xF) | (((v[(j >> 1) + 8] >> sh) & 0xF) << 4));
        }
    }
}
// The same shuffle, one 16-byte block per SSE2 vector (Phase 15 C0, docs/deepseek41/40): the even
// outputs are (low nibbles of bytes 0-7) | (low nibbles of bytes 8-15) << 4, the odd outputs the
// high nibbles likewise, interleaved. The scalar loop ran instruction-bound at 3.8 ms per slot on
// one core against memcpy's 0.9 (step 0(d)); ds41_permute_plane_selftest checks this against the
// reference on random planes.
inline void permute_plane(const uint8_t* src, uint8_t* qs, uint64_t len0) {
#if defined(__SSE2__)
    const __m128i m4 = _mm_set1_epi8(0x0F), mh = _mm_set1_epi8(int8_t(0xF0));
    for (uint64_t blk = 0; blk < len0 / 16; ++blk) {
        const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + blk * 16));
        const __m128i hi = _mm_srli_si128(v, 8);                                        // bytes 8-15 in the low half
        const __m128i ev = _mm_or_si128(_mm_and_si128(v, m4), _mm_slli_epi16(_mm_and_si128(hi, m4), 4));   // even outputs
        const __m128i od = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(v, 4), m4), _mm_and_si128(hi, mh));    // odd outputs
        _mm_storeu_si128(reinterpret_cast<__m128i*>(qs + blk * 16), _mm_unpacklo_epi8(ev, od));
    }
#else
    permute_plane_ref(src, qs, len0);
#endif
}
std::string plane_check(const Ds4MatPlanes& m, const Ds41Tensor& t, const char* nm) {
    if (!t.w || !t.s) return std::string(nm) + ": not bound";
    if (t.w->shape != std::vector<int64_t>{m.N, m.K / 2}) return std::string(nm) + ": plane is not [N, K/2]";
    if (t.w->nbytes != m.len0 || t.s->nbytes != m.len1) return std::string(nm) + ": byte count disagrees with the layout";
    return {};
}
// Run `pack(i, thread)` for i in [0, n) on kReaders threads; the first error wins, the rest stop.
std::string parallel_fill(uint32_t n, const std::function<std::string(uint32_t, uint32_t)>& pack) {
    std::string err; std::atomic<bool> failed{false};
    // Said once: this file MUST be compiled with -fopenmp (src/CMakeLists.txt sets it per source);
    // without it every pragma here is ignored and the whole fill runs on one thread -- which is
    // how runs C-F of docs/deepseek41/23 measured a "parallel" pack at one thread's speed.
    static std::atomic<bool> reported{false};
    #pragma omp parallel for schedule(dynamic) num_threads(kReaders)
    for (int64_t i = 0; i < int64_t(n); ++i) {
        if (i == 0 && n >= kReaders && omp_get_num_threads() < int(kReaders) && !reported.exchange(true))
            std::fprintf(stderr, "[ds41 tier] WARNING: parallel_fill got %d thread(s) for %u readers (max %d, level %d) -- compiled without OpenMP?\n",
                         omp_get_num_threads(), kReaders, omp_get_max_threads(), omp_get_level());
        if (failed.load(std::memory_order_relaxed)) continue;
        const std::string e = pack(uint32_t(i), uint32_t(omp_get_thread_num()));
        if (!e.empty()) {
            failed.store(true, std::memory_order_relaxed);
            #pragma omp critical
            { if (err.empty()) err = e; }
        }
    }
    return err;
}
}  // namespace

bool ds41_permute_plane_selftest(uint64_t bytes, uint32_t seed) {
    std::vector<uint8_t> src(bytes), a(bytes), b(bytes);
    uint32_t x = seed ? seed : 1u; for (auto& v : src) { x = x * 1664525u + 1013904223u; v = uint8_t(x >> 24); }
    permute_plane_ref(src.data(), a.data(), bytes); permute_plane(src.data(), b.data(), bytes);
    return a == b;
}

std::string ds41_slot_pread_file(int dfd, uint64_t off, void* dst, uint64_t bytes, uint32_t slice, uint32_t n_slices, uint64_t* ns_read) {
    if (dfd < 0) return "expert file: not open";
    if (n_slices == 0 || slice >= n_slices) return "expert file: slice out of range";
    if ((off | bytes | reinterpret_cast<uintptr_t>(dst)) & (kPage - 1)) return "expert file: offset, size or destination not 4 KiB-aligned";
    const uint64_t pages = bytes / kPage, p0 = pages * slice / n_slices, p1 = pages * (slice + 1) / n_slices;
    uint8_t* d = static_cast<uint8_t*>(dst) + p0 * kPage; const uint64_t want = (p1 - p0) * kPage; uint64_t got = 0;
    const auto t0 = std::chrono::steady_clock::now();
    while (got < want) {
        const ssize_t r = ::pread(dfd, d + got, size_t(want - got), off_t(off + p0 * kPage + got));
        if (r < 0) { if (errno == EINTR) continue; return std::string("expert file pread: ") + std::strerror(errno); }
        if (r == 0) return "expert file pread: unexpected end of file";
        got += uint64_t(r);
    }
    if (ns_read) *ns_read += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    return {};
}

std::string ds41_slot_layout(uint32_t H, uint32_t EF, Ds4SlotLayout& out) {
    if (H % 32 || EF % 32) return "ds41_slot_layout: H and EF must be multiples of 32";
    auto mat = [](Ds4MatPlanes& m, uint32_t K, uint32_t N, uint64_t& cur) {
        m.dt = DType::kMXFP4; m.K = K; m.N = N; m.src_K = K; m.src_N = N; m.k0 = 0; m.n0 = 0;
        m.off0 = cur; m.len0 = uint64_t(N) * (K / 2);  cur = align64(cur + m.len0);   // nibbles
        m.off1 = cur; m.len1 = uint64_t(N) * (K / 32); cur = align64(cur + m.len1);   // E8M0
        m.off2 = 0;   m.len2 = 0;
    };
    uint64_t cur = 0;
    mat(out.gate, H, EF, cur); mat(out.up, H, EF, cur); mat(out.down, EF, H, cur);
    out.bytes = cur;
    return {};
}

std::string ds41_slot_pack(const Ds4SlotLayout& lay, const Ds41Tensor& w1, const Ds41Tensor& w3,
                           const Ds41Tensor& w2, void* dst) {
    auto one = [&](const Ds4MatPlanes& m, const Ds41Tensor& t, const char* nm) -> std::string {
        if (auto e = plane_check(m, t, nm); !e.empty()) return e;
        uint8_t* qs = static_cast<uint8_t*>(dst) + m.off0;
        const uint8_t* src = t.w->data;
        // the Phase 4 shuffle, on the host: engine_byte[j] = nib(v[j/2], j&1) | nib(v[j/2+8], j&1) << 4
        // Parallel over the 16-byte blocks (each writes only its own 16 output bytes). Measured
        // (docs/deepseek41/23): this loop's time is the page-fault read of the mapping under it,
        // not the shuffle -- twenty threads moved it 5%; ds41_slot_pack_pread is the fix.
        const int64_t n_blk = int64_t(m.len0 / 16);
        #pragma omp parallel for schedule(static)
        for (int64_t blk = 0; blk < n_blk; ++blk) {
            const uint8_t* v = src + blk * 16; uint8_t* o = qs + blk * 16;
            for (int j = 0; j < 16; ++j) {
                const int sh = (j & 1) * 4;
                o[j] = uint8_t(((v[j >> 1] >> sh) & 0xF) | (((v[(j >> 1) + 8] >> sh) & 0xF) << 4));
            }
        }
        std::memcpy(static_cast<uint8_t*>(dst) + m.off1, t.s->data, m.len1);
        return {};
    };
    if (auto e = one(lay.gate, w1, "w1"); !e.empty()) return e;
    if (auto e = one(lay.up,   w3, "w3"); !e.empty()) return e;
    return one(lay.down, w2, "w2");
}

uint64_t ds41_pack_bounce_bytes(const Ds4SlotLayout& lay) {
    const uint64_t biggest = std::max({lay.gate.len0, lay.up.len0, lay.down.len0, lay.gate.len1, lay.up.len1, lay.down.len1});
    return (biggest + 2 * kPage + kPage - 1) & ~(kPage - 1);       // the aligned superset of any one plane
}

std::string ds41_slot_pack_pread(const Ds4SlotLayout& lay, const SafetensorsModel& store,
                                 const Ds41Tensor& w1, const Ds41Tensor& w3, const Ds41Tensor& w2,
                                 void* dst, void* bounce, uint64_t* ns_read, uint64_t* ns_permute,
                                 uint32_t slice, uint32_t n_slices) {
    if (!bounce || (reinterpret_cast<uintptr_t>(bounce) & (kPage - 1))) return "pack_pread: bounce must be page-aligned";
    if (n_slices == 0 || slice >= n_slices) return "pack_pread: slice out of range";
    if (lay.gate.len0 % 16 || lay.up.len0 % 16 || lay.down.len0 % 16) return "pack_pread: a nibble plane is not whole 16-byte blocks (the slices would drop its tail)";   // gate 11 finding 3
    auto* bb = static_cast<uint8_t*>(bounce);
    uint64_t nr = 0, np = 0;
    auto tick = [] { return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()); };
    auto one = [&](const Ds4MatPlanes& m, const Ds41Tensor& t, const char* nm) -> std::string {
        if (auto e = plane_check(m, t, nm); !e.empty()) return e;
        // this slice's byte range of each plane: 16-byte blocks of the nibble plane, bytes of the scales
        const uint64_t nb = m.len0 / 16, s0 = (nb * slice / n_slices) * 16, s1 = (nb * (slice + 1) / n_slices) * 16;
        const uint64_t c0 = m.len1 * slice / n_slices, c1 = m.len1 * (slice + 1) / n_slices;
        int fd = -1, dfd = -1; uint64_t off = 0, at = 0;
        if (!store.locate(t.w->data, fd, dfd, off)) return std::string(nm) + ": nibble plane is not inside a shard mapping";
        uint64_t t0 = tick(), t1 = t0;
        if (s1 > s0) {
            if (auto e = read_range(fd, dfd, bb, off + s0, s1 - s0, &at); !e.empty()) return std::string(nm) + " nibbles: " + e;
            t1 = tick(); nr += t1 - t0;
            permute_plane(bb + at, static_cast<uint8_t*>(dst) + m.off0 + s0, s1 - s0);
            t0 = tick(); np += t0 - t1;
        }
        if (c1 > c0) {
            if (!store.locate(t.s->data, fd, dfd, off)) return std::string(nm) + ": scale plane is not inside a shard mapping";
            if (auto e = read_range(fd, dfd, bb, off + c0, c1 - c0, &at); !e.empty()) return std::string(nm) + " scales: " + e;
            t1 = tick(); nr += t1 - t0;
            std::memcpy(static_cast<uint8_t*>(dst) + m.off1 + c0, bb + at, c1 - c0);
            np += tick() - t1;
        }
        return {};
    };
    struct Done { uint64_t *r, *p, &nr, &np; ~Done() { if (r) *r += nr; if (p) *p += np; } } done{ns_read, ns_permute, nr, np};
    if (auto e = one(lay.gate, w1, "w1"); !e.empty()) return e;
    if (auto e = one(lay.up,   w3, "w3"); !e.empty()) return e;
    return one(lay.down, w2, "w2");
}

Ds41ExpertTier::~Ds41ExpertTier() = default;

void Ds41ExpertTier::free_storage(sycl::queue& q) {
    mm_reader_stop();
    cpu_stop(); if (cpu_.h_rows) { sycl::free(cpu_.h_rows, q); cpu_.h_rows = nullptr; }
    if (bws_.max_tokens) ds4_expert_batch_ws_free(q, bws_);
    for (void** p : {reinterpret_cast<void**>(&h_row_tok_), reinterpret_cast<void**>(&h_tk2p_), reinterpret_cast<void**>(&h_w_pk_), reinterpret_cast<void**>(&h_w_pk2_)}) if (*p) { sycl::free(*p, q); *p = nullptr; }
    for (auto* sp : mm_stage_) if (sp) {
        if (stage_imported_) { if (ze_drv_ && ze_release_) reinterpret_cast<ze_result_t (*)(ze_driver_handle_t, void*)>(ze_release_)(static_cast<ze_driver_handle_t>(ze_drv_), sp); munmap(sp, size_t(lay_.bytes)); }
        else sycl::free(sp, q);
    }
    mm_stage_.clear(); stage_imported_ = false; ze_drv_ = nullptr; ze_release_ = nullptr;
    if (mm_file_fd_ >= 0) { close(mm_file_fd_); mm_file_fd_ = -1; } mm_file_off_.clear();
    for (void* b : bounce_) std::free(b);
    bounce_.clear();
    if (mm_dev_) { sycl::free(mm_dev_, q); mm_dev_ = nullptr; mm_dev_cap_ = 0; }
    if (dq_) { dq_->wait_and_throw(); dq_.reset(); }
    cache_.free_storage(); arena_.free_storage();
    ready_ = false; empty_ = false;
}

std::string Ds41ExpertTier::init(sycl::queue& q, const DeepSeek41Model& m, uint32_t first_layer, uint32_t n_layers,
                                 const std::vector<std::vector<uint32_t>>& ranking,
                                 uint32_t n_static, uint32_t n_pinned, uint32_t stream_slots, uint32_t max_tokens,
                                 uint32_t part, uint32_t n_parts) {
    // The OpenMP block time is 0 on every thread of this tier that forks a team (this one, the
    // mmap reader thread of each moe() call, the CPU worker): libiomp5's default lets a team spin
    // 200 ms after every region, and the reader team's spin on the P-cores displaces the host
    // thread that submits the GPU work -- gate 13 isolated it at ~26 ms/token idle (183-189 -> 159-160)
    // and ~190 ms/token under a 400% CPU load (docs/deepseek41/35 step 3). Numerics-free; on by default.
    kmp_set_blocktime(0);
    const auto& c = m.config();
    m_ = &m; L0_ = first_layer; nL_ = n_layers; E_ = c.n_routed(first_layer); TK_ = c.n_activated(first_layer);   // the layer kind's counts (the MTP stages: 128 / 3)
    H_ = c.dim; EF_ = c.moe_inter_dim; np_ = n_pinned; share_ = n_parts > 1;
    if (first_layer + n_layers > c.n_layers + c.n_mtp_layers) return "tier: layer range out of the text path";
    if (n_parts == 0 || part > n_parts) return "tier: part " + std::to_string(part) + " of " + std::to_string(n_parts) + " parts";
    // part == n_parts: the empty subset -- no expert of any layer (the expert-parallel control arm)
    if (auto e = ds41_slot_layout(H_, EF_, lay_); !e.empty()) return e;

    if (part == n_parts) {   // the empty subset: a tier that CONSUMES NO RESOURCES (the control arm of docs/38, gate 14 finding 1)
        tier_.assign(n_layers, std::vector<uint8_t>(E_, 3)); empty_ = true; ready_ = true; return {};
    }
    const std::vector<uint64_t> slot_bytes(n_layers, lay_.bytes);
    // this tier's experts per layer, in ranking order: the whole ranking, or its positions == part (mod n_parts)
    std::vector<std::vector<uint32_t>> sub(n_layers);
    tier_.assign(n_layers, std::vector<uint8_t>(E_, 3));
    std::vector<std::vector<uint32_t>> pinned_ids(n_layers);
    for (uint32_t l = 0; l < n_layers; ++l) {
        const auto& r = ranking.at(L0_ + l);
        if (r.size() != E_) return "tier: ranking for layer " + std::to_string(L0_ + l) + " is not a full permutation";
        for (uint32_t i = part; i < E_; i += n_parts) sub[l].push_back(r[i]);
        if (n_static + n_pinned > sub[l].size()) return "tier: static + pinned exceeds this part's expert count";
        for (uint32_t e : sub[l]) tier_[l][e] = 2;
        for (uint32_t i = 0; i < n_static; ++i) tier_[l][sub[l][i]] = 0;
        for (uint32_t i = n_static; i < n_static + n_pinned; ++i) { tier_[l][sub[l][i]] = 1; pinned_ids[l].push_back(sub[l][i]); }
    }
    // The safetensors reader advises its mappings MADV_RANDOM at open (one 4 KB page per
    // fault). Expert bytes no longer go through the mapping at all -- every fill below reads
    // with pread (ds41_slot_pack_pread) -- but the dense path still does; keep the whole-mapping
    // MADV_NORMAL for it. One call per shard. No WILLNEED.
    if (!m.store().advise(MADV_NORMAL)) return "tier: madvise(MADV_NORMAL) failed on a shard mapping";
    mm_stage_.resize(mm_stage_slots_);
    const char* file_env = std::getenv("IE_DS41_EXPERT_FILE");
    const std::string file_path = (file_env && L0_ < c.n_layers) ? file_env : "";   // the file holds the text layers' experts; an MTP tier (the drafter's) skips it
    if (!file_path.empty()) {
        // the staging slots as anonymous huge-page memory imported into the driver (docs/40 step 0(c):
        // O_DIRECT into host USM is refused with EFAULT; imported anonymous memory takes it and DMAs)
        ze_driver_handle_t drv = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context().get_platform());
        void* lib = dlopen("libze_loader.so.1", RTLD_NOW | RTLD_NOLOAD); if (!lib) lib = dlopen("libze_loader.so.1", RTLD_NOW);
        auto getext = lib ? reinterpret_cast<ze_result_t (*)(ze_driver_handle_t, const char*, void**)>(dlsym(lib, "zeDriverGetExtensionFunctionAddress")) : nullptr;
        ze_result_t (*imp)(ze_driver_handle_t, void*, size_t) = nullptr; ze_result_t (*rel)(ze_driver_handle_t, void*) = nullptr;
        if (!getext || getext(drv, "zexDriverImportExternalPointer", reinterpret_cast<void**>(&imp)) != ZE_RESULT_SUCCESS || !imp
                    || getext(drv, "zexDriverReleaseImportedPointer", reinterpret_cast<void**>(&rel)) != ZE_RESULT_SUCCESS || !rel)
            return "tier: IE_DS41_EXPERT_FILE needs the driver's import extension (zexDriverImportExternalPointer), which is not available";
        ze_drv_ = drv; ze_release_ = reinterpret_cast<void*>(rel); stage_imported_ = true;
        for (auto& sp : mm_stage_) {
            void* m = mmap(nullptr, size_t(lay_.bytes), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (m == MAP_FAILED) return "tier: staging mmap failed";
            madvise(m, size_t(lay_.bytes), MADV_HUGEPAGE); std::memset(m, 0, size_t(lay_.bytes));
            if (const ze_result_t r = imp(drv, m, size_t(lay_.bytes)); r != ZE_RESULT_SUCCESS) return "tier: staging import failed (0x" + std::to_string(unsigned(r)) + ")";
            sp = static_cast<uint8_t*>(m);
        }
    } else
        for (auto& sp : mm_stage_) { sp = sycl::malloc_host<uint8_t>(lay_.bytes, q); if (!sp) return "tier: pinned staging alloc failed"; }
    bounce_.assign(kReaders, nullptr);
    for (auto& b : bounce_)
        if (posix_memalign(&b, size_t(kPage), size_t(ds41_pack_bounce_bytes(lay_))) != 0) return "tier: bounce alloc failed";
    dq_ = std::make_unique<sycl::queue>(q.get_context(), q.get_device(), sycl::property_list{sycl::property::queue::in_order{}});
    const auto& store = m.store();

    // pinned tier: arena slots, pread-packed straight into the pinned slot, kReaders at a time
    if (n_pinned) {
        if (auto e = arena_.init_set_per_layer(q, slot_bytes, E_, pinned_ids); !e.empty()) return "tier arena: " + e;
        for (uint32_t l = 0; l < n_layers; ++l) {
            const auto& Lw = m.layers()[L0_ + l];
            const auto e = parallel_fill(uint32_t(pinned_ids[l].size()), [&](uint32_t i, uint32_t th) -> std::string {
                const uint32_t ex = pinned_ids[l][i];
                void* dst = arena_.slot(l, ex);
                if (!dst) return "tier arena: no slot for pinned expert " + std::to_string(ex);
                if (auto s = ds41_slot_pack_pread(lay_, store, Lw.exp_w1[ex], Lw.exp_w3[ex], Lw.exp_w2[ex], dst, bounce_[th]); !s.empty())
                    return "tier pack (layer " + std::to_string(L0_ + l) + ", expert " + std::to_string(ex) + "): " + s;
                return {};
            });
            if (!e.empty()) return e;
        }
    } else {
        if (auto e = arena_.init_set_per_layer(q, slot_bytes, E_, std::vector<std::vector<uint32_t>>(n_layers)); !e.empty())
            return "tier arena (empty): " + e;
    }
    // VRAM: static + stream slots per layer. Static installs go through the pinned staging
    // slots, a batch of readers at a time, then one DMA each.
    if (auto e = cache_.init(q, arena_, n_static + stream_slots, 0, n_static, 0); !e.empty()) return "tier cache: " + e;
    // Phase 15 C2: the expert file -- validated, coverage counted, one slot checked byte for byte
    if (!file_path.empty()) {
        const int fd = open(file_path.c_str(), O_RDONLY), dfd = open(file_path.c_str(), O_RDONLY | O_DIRECT);
        if (fd < 0 || dfd < 0) return "tier: IE_DS41_EXPERT_FILE " + file_path + ": " + std::strerror(errno);
        Ds41ExpertFileHeader hd{};
        if (pread(fd, &hd, sizeof hd, 0) != ssize_t(sizeof hd) || std::memcmp(hd.magic, kDs41ExpertFileMagic, 8) != 0) { close(fd); close(dfd); return "tier: expert file: not an IESLOT01 file"; }
        if (hd.H != H_ || hd.EF != EF_ || hd.n_layers != c.n_layers || hd.n_experts != E_ || hd.slot_bytes != lay_.bytes) { close(fd); close(dfd);
            return "tier: expert file: layout mismatch (H " + std::to_string(hd.H) + " EF " + std::to_string(hd.EF) + " layers " + std::to_string(hd.n_layers) + " experts " + std::to_string(hd.n_experts) + " slot " + std::to_string(hd.slot_bytes) + ")"; }
        std::vector<uint64_t> table(size_t(hd.n_layers) * hd.n_experts);
        if (pread(fd, table.data(), table.size() * 8, off_t(hd.table_off)) != ssize_t(table.size() * 8)) { close(fd); close(dfd); return "tier: expert file: table read failed"; }
        close(fd);
        mm_file_off_.assign(size_t(n_layers) * E_, 0);
        uint32_t covered = 0, wanted = 0; int32_t sl0 = -1; uint32_t se0 = 0;
        for (uint32_t l = 0; l < n_layers; ++l) for (uint32_t e = 0; e < E_; ++e) if (tier_[l][e] == 2) {
            ++wanted; const uint64_t o = table[size_t(L0_ + l) * E_ + e];
            if (o) { mm_file_off_[size_t(l) * E_ + e] = o; ++covered; if (sl0 < 0) { sl0 = int32_t(l); se0 = e; } }
        }
        if (covered) {   // the sample: the pack path's bytes == the file's, byte for byte
            void* a = nullptr; void* b = nullptr;
            if (posix_memalign(&a, kPage, size_t(lay_.bytes)) != 0 || posix_memalign(&b, kPage, size_t(lay_.bytes)) != 0) { close(dfd); return "tier: expert file: sample alloc failed"; }
            const auto& Lw = m.layers()[L0_ + uint32_t(sl0)];
            std::string e1 = ds41_slot_pack_pread(lay_, store, Lw.exp_w1[se0], Lw.exp_w3[se0], Lw.exp_w2[se0], a, bounce_[0]);
            std::string e2 = e1.empty() ? ds41_slot_pread_file(dfd, mm_file_off_[size_t(sl0) * E_ + se0], b, lay_.bytes, 0, 1, nullptr) : std::string();
            const bool same = e1.empty() && e2.empty() && std::memcmp(a, b, size_t(lay_.bytes)) == 0;
            std::free(a); std::free(b);
            if (!same) { close(dfd); mm_file_off_.clear(); return "tier: expert file REFUSED: layer " + std::to_string(L0_ + uint32_t(sl0)) + " expert " + std::to_string(se0) + " differs from the pack path" + (e1.empty() ? "" : " (" + e1 + ")") + (e2.empty() ? "" : " (" + e2 + ")"); }
        }
        mm_file_fd_ = dfd;
        std::fprintf(stderr, "[ds41 tier] expert file %s: covers %u of this tier's %u mmap experts (layers %u..%u); staging imported; one slot verified byte for byte\n",
                     file_path.c_str(), covered, wanted, L0_, L0_ + n_layers - 1);
    }
    cpu_start(q);
    for (uint32_t l = 0; l < n_layers; ++l) {
        const auto& Lw = m.layers()[L0_ + l]; const auto& r = sub[l];
        for (uint32_t b0 = 0; b0 < n_static; b0 += mm_stage_slots_) {
            const uint32_t bn = std::min(mm_stage_slots_, n_static - b0);
            const auto e = parallel_fill(bn, [&](uint32_t i, uint32_t th) -> std::string {
                const uint32_t s = b0 + i;
                if (auto err = ds41_slot_pack_pread(lay_, store, Lw.exp_w1[r[s]], Lw.exp_w3[r[s]], Lw.exp_w2[r[s]], mm_stage_[i], bounce_[th]); !err.empty())
                    return "tier static pack (layer " + std::to_string(L0_ + l) + ", expert " + std::to_string(r[s]) + "): " + err;
                return {};
            });
            if (!e.empty()) return e;
            for (uint32_t i = 0; i < bn; ++i)
                if (auto err = cache_.install_static(l, r[b0 + i], b0 + i, mm_stage_[i]); !err.empty()) return "tier install_static: " + err;
        }
    }
    if (auto e = ds4_expert_batch_ws_alloc(q, max_tokens, TK_, H_, EF_, bws_); !e.empty()) return "tier batch ws: " + e;
    { const size_t n = size_t(max_tokens) * TK_;
      h_row_tok_ = sycl::malloc_host<int32_t>(n, q); h_tk2p_ = sycl::malloc_host<int32_t>(n, q); h_w_pk_ = sycl::malloc_host<float>(n, q); h_w_pk2_ = sycl::malloc_host<float>(n, q);
      if (!h_row_tok_ || !h_tk2p_ || !h_w_pk_ || !h_w_pk2_) return "tier: pinned packing staging alloc failed"; }
    ready_ = true;
    return {};
}

void Ds41ExpertTier::cpu_start(sycl::queue& q) {
    const char* on = std::getenv("IE_DS41_CPU_MISS");
    cpu_.on = !(on && *on && std::string(on) == "0");   // ON since gate 20 (docs/52): -7.6 ms/token under EP; =0 the kill switch
    if (!cpu_.on) return;
    if (share_) cpu_.qstar = 0.60f;                         // under expert parallel (a parity share): the sweep's optimum (docs/52); 0.30 for a whole tier
    if (const char* qs = std::getenv("IE_DS41_QSTAR")) cpu_.qstar = std::max(0.f, std::min(1.f, float(std::atof(qs))));
    std::string cores = !cpu_cores_override_.empty() ? cpu_cores_override_ : std::getenv("IE_DS41_CPU_CORES") ? std::getenv("IE_DS41_CPU_CORES") : "8-19";
    cpu_.cores.clear();
    for (size_t i = 0; i < cores.size();) {                // "8-19,0,1" -> the core list
        size_t j = cores.find(',', i); if (j == std::string::npos) j = cores.size();
        const std::string tok = cores.substr(i, j - i); const size_t d = tok.find('-');
        if (d == std::string::npos) cpu_.cores.push_back(std::atoi(tok.c_str()));
        else for (int c = std::atoi(tok.substr(0, d).c_str()); c <= std::atoi(tok.substr(d + 1).c_str()); ++c) cpu_.cores.push_back(c);
        i = j + 1;
    }
    cpu_.nthreads = int(cpu_.cores.size());
    cpu_.lay = &lay_; cpu_.x.assign(H_, 0.f); cpu_.scratch.assign(size_t(2) * EF_, 0.f); cpu_.out.assign(H_, 0.f);
    cpu_.h_rows = sycl::malloc_host<sycl::half>(size_t(TK_) * H_, q);
    cpu_.stop = false; cpu_.pending = false; cpu_.done = true;
    // the reader threads (this thread's OpenMP team) keep off the worker's cores: pinned to the rest
    {
        std::vector<int> rest; const long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
        for (int c = 0; c < ncpu; ++c) if (std::find(cpu_.cores.begin(), cpu_.cores.end(), c) == cpu_.cores.end()) rest.push_back(c);
        if (!rest.empty()) {
            cpu_set_t set; CPU_ZERO(&set); for (int c : rest) CPU_SET(c, &set);
            sched_setaffinity(0, sizeof set, &set);                          // the host thread
            #pragma omp parallel num_threads(kReaders)
            { sched_setaffinity(0, sizeof set, &set); }                        // its reader team
        }
    }
    cpu_.th = std::thread([this] { cpu_run(); });
    std::fprintf(stderr, "[ds41 tier] CPU miss path ON: q* %.2f (PCIe share of the pinned misses), %d threads on cores %s\n", cpu_.qstar, cpu_.nthreads, cores.c_str());
}

void Ds41ExpertTier::mm_reader_post(int which, std::function<void()> job) {
    MmReader& r = mmr_[which];
    if (!r.th.joinable()) {
        r.stop = false; r.pending = false; r.done = true;
        r.th = std::thread([&r] {
            kmp_set_blocktime(0);                                  // once: this thread is the team's master for its whole life
            for (;;) {
                std::unique_lock<std::mutex> lk(r.mu);
                r.cv.wait(lk, [&] { return r.pending || r.stop; });
                if (r.stop) return;
                r.pending = false; auto job = std::move(r.job); lk.unlock();
                job();
                lk.lock(); r.done = true; lk.unlock(); r.cv.notify_all();
            }
        });
    }
    { std::lock_guard<std::mutex> lk(r.mu); r.job = std::move(job); r.pending = true; r.done = false; }
    r.cv.notify_all();
}

void Ds41ExpertTier::mm_reader_wait(int which) {
    MmReader& r = mmr_[which];
    if (!r.th.joinable()) return;
    std::unique_lock<std::mutex> lk(r.mu);
    r.cv.wait(lk, [&] { return r.done; });
}

void Ds41ExpertTier::mm_reader_stop() {
    for (auto& r : mmr_) {
        if (!r.th.joinable()) continue;
        { std::unique_lock<std::mutex> lk(r.mu); r.cv.wait(lk, [&] { return r.done; }); r.stop = true; }
        r.cv.notify_all(); r.th.join();
    }
}

void Ds41ExpertTier::cpu_stop() {
    if (!cpu_.th.joinable()) return;
    { std::lock_guard<std::mutex> lk(cpu_.mu); cpu_.stop = true; }
    cpu_.cv.notify_all(); cpu_.th.join();
}

// The worker: one OpenMP team, every thread pinned to its core once, then requests forever.
void Ds41ExpertTier::cpu_run() {
    const auto& cores = cpu_.cores;
    kmp_set_blocktime(0);                 // this thread's team too (see init)
    #pragma omp parallel num_threads(cpu_.nthreads)
    {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(cores[size_t(omp_get_thread_num()) % cores.size()], &set);
        sched_setaffinity(0, sizeof set, &set);
    }
    for (;;) {
        std::unique_lock<std::mutex> lk(cpu_.mu);
        cpu_.cv.wait(lk, [&] { return cpu_.pending || cpu_.stop; });
        if (cpu_.stop) return;
        cpu_.pending = false; lk.unlock();
        const auto tw = std::chrono::steady_clock::now();
        size_t i_row = 0;
        for (const auto& [slot, row] : cpu_.work) {
            cpu_expert_mxfp4(slot, *cpu_.lay, cpu_.x.data(), cpu_.scratch.data(), cpu_.out.data(), cpu_.limit, cpu_.nthreads);
            sycl::half* dst = cpu_.h_rows + i_row * H_; for (uint32_t i = 0; i < H_; ++i) dst[i] = sycl::half(cpu_.out[i]);   // the row, as the GPU stores its own (fp16)
            (void)row; ++i_row;
        }
        cpu_.work_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tw).count();
        lk.lock(); cpu_.done = true; lk.unlock(); cpu_.cv.notify_all();
    }
}

std::string Ds41ExpertTier::moe(sycl::queue& q, uint32_t L, const float* x, const int32_t* ridx, const float* rw,
                                uint32_t T, float* y, float swiglu_limit) {
    if (!ready_) return "tier: not initialised";
    if (L < L0_ || L >= L0_ + nL_) return "tier: layer " + std::to_string(L) + " is not in this tier";
    if (empty_) { st_ = Ds41TierStats{}; ep_rows_.clear(); if (pre_groups_) pre_groups_(); if (!ep_export_ && y) q.memset(y, 0, size_t(T) * H_ * 4).wait(); return {}; }   // nothing to serve (the caller's pre-groups work still runs: gate 18 finding 3)
    if (T > bws_.max_tokens) return "tier: T exceeds the batch workspace";
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t fetched0 = cache_.stats().bytes_fetched;   // the pinned tier's H2D bytes, counted at the cache's misses
    const uint64_t shits0 = cache_.stats().stream_hits;       // ... and the pinned experts a stream slot already held
    const uint32_t l = L - L0_, H = H_, EF = EF_, K = TK_, E = E_;
    const auto& Lw = m_->layers()[L];
    st_ = Ds41TierStats{};

    // ---- counting sort by expert (V4's packing) -------------------------------------------
    bws_.routes.resize(T);
    for (uint32_t t = 0; t < T; ++t) {
        bws_.routes[t].clear();
        for (uint32_t k = 0; k < K; ++k) {
            const int32_t e = ridx[size_t(t) * K + k];
            if (e < 0 || uint32_t(e) >= E) return "tier: expert id out of range";
            bws_.routes[t].push_back({uint32_t(e), rw[size_t(t) * K + k]});
        }
    }
    build_moe_packing(bws_.routes, E, K, bws_.pk);
    const uint32_t TK = T * K;
    std::copy_n(bws_.pk.sorted_idx.data(), TK, h_row_tok_);     q.memcpy(bws_.row_tok, h_row_tok_, TK * 4);   // pinned sources: the enqueues do not block (Phase 18)
    std::copy_n(bws_.pk.tk_to_packed.data(), TK, h_tk2p_);     q.memcpy(bws_.tk2p, h_tk2p_, TK * 4);
    std::copy_n(bws_.pk.weights_packed.data(), TK, h_w_pk_);   q.memcpy(bws_.w_pk, h_w_pk_, TK * 4);
    ds4_expert_gather_cast(q, x, bws_.row_tok, bws_.xp, TK, H);
    quantize_q8_1(q, bws_.xp, bws_.xp_q8, TK * H);
    if (!ep_export_) q.memset(y, 0, size_t(T) * H * 4);            // export mode has no y: the rows are the product
    const auto& off = bws_.pk.expert_offsets;
    auto ms_since = [](const std::chrono::steady_clock::time_point& t) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count(); };

    // ---- partition the occupied experts: static/pinned run through the stream-slot groups
    // now; the mmap experts are filled MEANWHILE on a reader thread (DMAs on dq_) and run as
    // one last group once the fill has landed. Before this the fill and the groups ran back to
    // back, 8.4 s + 7.0 s at pp2048 (docs/deepseek41/23, run G).
    std::vector<uint32_t> occ, mm; occ.reserve(E); bool alien = false; ep_rows_.clear();
    for (uint32_t e = 0; e < E; ++e) {
        if (off[e + 1] <= off[e]) continue;
        if (tier_[l][e] == 3) {
            // another tier's expert (expert parallel). With an importer those rows arrive from the
            // other tier before the scatter and keep their real weights (bit-exact with the whole
            // tier); without one they carry weight 0 and a zeroed row, so the scatter adds nothing
            if (!ep_import_) for (uint32_t rrow = off[e]; rrow < off[e + 1]; ++rrow) bws_.pk.weights_packed[rrow] = 0.f;
            q.memset(static_cast<sycl::half*>(bws_.yp) + uint64_t(off[e]) * H, 0, size_t(off[e + 1] - off[e]) * H * 2);
            alien = true; continue;
        }
        if (ep_export_) for (uint32_t rrow = off[e]; rrow < off[e + 1]; ++rrow) ep_rows_.push_back(int32_t(rrow));
        if (tier_[l][e] == 2) mm.push_back(e); else occ.push_back(e);
    }
    if (alien && !ep_import_) { std::copy_n(bws_.pk.weights_packed.data(), TK, h_w_pk2_); q.memcpy(bws_.w_pk, h_w_pk2_, TK * 4); }
    // ---- the decode miss path: a share of the pinned MISSES goes to the CPU worker, concurrently
    bool cpu_leg = false; std::chrono::steady_clock::time_point tc{};
    if (cpu_.on && T == 1) {   // Phase 20 (docs/49): in both modes -- the CPU rows land in the packed workspace like the GPU's
        std::vector<uint32_t> miss;
        for (uint32_t e : occ) if (tier_[l][e] == 1 && !cache_.is_resident(l, e)) miss.push_back(e);
        const uint32_t n_pcie = uint32_t(std::lround(cpu_.qstar * double(miss.size())));
        if (miss.size() > n_pcie) {
            cpu_.work.clear();
            for (size_t i = n_pcie; i < miss.size(); ++i) {
                const uint32_t e = miss[i];
                cpu_.work.emplace_back(arena_.slot(l, e), off[e]);   // its packed row (one row per expert at T = 1); the weight stays real
                occ.erase(std::find(occ.begin(), occ.end(), e));
            }
            cpu_.limit = swiglu_limit;
            q.memcpy(cpu_.x.data(), x, size_t(H) * 4).wait();          // the activation, host side
            { std::lock_guard<std::mutex> lk(cpu_.mu); cpu_.pending = true; cpu_.done = false; }
            cpu_.cv.notify_all();
            cpu_leg = true; tc = std::chrono::steady_clock::now();
            st_.experts_cpu = uint32_t(cpu_.work.size());
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    st_.ms_prep = ms_since(t0);
    uint8_t* mm_dev = nullptr; std::string mm_err;
    int reader = -1;                                      // the persistent reader running this call's fill, or none
    struct WaitAtExit { Ds41ExpertTier& t; int& w; ~WaitAtExit() { if (w >= 0) t.mm_reader_wait(w); } } join_guard{*this, reader};   // gate 8b finding 4
    if (!mm.empty()) {
        const size_t need = mm.size() * lay_.bytes;
        if (need > mm_dev_cap_) {
            if (mm_dev_) sycl::free(mm_dev_, q);
            mm_dev_ = sycl::malloc_device<uint8_t>(need, q); mm_dev_cap_ = mm_dev_ ? need : 0;
            if (!mm_dev_) return "tier mmap: device alloc failed";
        }
        mm_dev = mm_dev_;
        // Phase 20 (docs/52): the core partition confines the host thread -- and every thread it spawns -- to the
        // P-cores; at prefill the readers are the work, so they are spawned with every core allowed (T = 1 keeps
        // the partition: the CPU leg's team owns the E-cores then)
        cpu_set_t saved_aff; bool relax = false;
        if (cpu_.on && T > 1 && sched_getaffinity(0, sizeof saved_aff, &saved_aff) == 0) {
            cpu_set_t all; CPU_ZERO(&all); const long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
            for (long c = 0; c < ncpu && c < CPU_SETSIZE; ++c) CPU_SET(int(c), &all);
            relax = sched_setaffinity(0, sizeof all, &all) == 0;
        }
        // the persistent reader for this call's affinity: [1] all cores (relaxed above), [0] the caller's partition
        reader = relax ? 1 : 0;
        mm_reader_post(reader, [&] {
            try {
                sycl::queue& dq = *dq_;
                std::vector<sycl::event> inflight(mm_stage_slots_);
                double read_ms = 0;
                std::vector<uint64_t> ns_read(kReaders, 0), ns_perm(kReaders, 0);   // per reader thread
                std::atomic<uint32_t> n_file{0};                                       // experts filled from the expert file
                for (size_t b0 = 0; b0 < mm.size(); b0 += kReaders) {
                    const size_t bn = std::min<size_t>(kReaders, mm.size() - b0);
                    for (size_t i = 0; i < bn; ++i) inflight[(b0 + i) % mm_stage_slots_].wait();   // those slots' previous DMAs are done
                    const auto tr = std::chrono::steady_clock::now();
                    // decode (Phase 11): a batch of one or two experts is split into slices so every
                    // reader thread reads and permutes a part of each -- ~6 ms per expert on one thread
                    const uint32_t ns = uint32_t(bn) < kReaders ? kReaders / uint32_t(bn) : 1u;
                    const auto err = parallel_fill(uint32_t(bn) * ns, [&](uint32_t j, uint32_t th) -> std::string {
                        const uint32_t i = j / ns, sl = j % ns, e = mm[b0 + i];
                        const uint64_t foff = mm_file_off_.empty() ? 0 : mm_file_off_[size_t(l) * E_ + e];
                        if (foff) {   // Phase 15 C2: one direct pread into the (imported) staging slot, no bounce, no permute
                            if (auto er = ds41_slot_pread_file(mm_file_fd_, foff, mm_stage_[(b0 + i) % mm_stage_slots_], lay_.bytes, sl, ns, &ns_read[th]); !er.empty())
                                return "tier mmap file (expert " + std::to_string(e) + ", slice " + std::to_string(sl) + "): " + er;
                            if (sl == 0) n_file.fetch_add(1);
                            return {};
                        }
                        if (auto er = ds41_slot_pack_pread(lay_, m_->store(), Lw.exp_w1[e], Lw.exp_w3[e], Lw.exp_w2[e], mm_stage_[(b0 + i) % mm_stage_slots_], bounce_[th],
                                                           &ns_read[th], &ns_perm[th], sl, ns); !er.empty())
                            return "tier mmap pack (expert " + std::to_string(e) + ", slice " + std::to_string(sl) + "): " + er;
                        return {};
                    });
                    if (!err.empty()) { mm_err = err; break; }
                    read_ms += ms_since(tr);
                    for (size_t i = 0; i < bn; ++i) {
                        const size_t sl = (b0 + i) % mm_stage_slots_;
                        inflight[sl] = dq.memcpy(mm_dev + (b0 + i) * lay_.bytes, mm_stage_[sl], lay_.bytes);
                    }
                }
                for (auto& ev : inflight) ev.wait();
                dq.wait_and_throw();
                st_.ms_mmap_pack = read_ms;                   // the host read+permute batches, wall (DMAs overlap the next batch)
                uint64_t sr = 0, sp = 0; for (uint32_t t = 0; t < kReaders; ++t) { sr += ns_read[t]; sp += ns_perm[t]; }
                st_.ms_mmap_read = double(sr) / 1e6 / kReaders; st_.ms_mmap_permute = double(sp) / 1e6 / kReaders;
                st_.bytes_mmap_to_vram = uint64_t(mm.size()) * lay_.bytes;
                st_.experts_mmap = uint32_t(mm.size()); st_.experts_mmap_file = n_file.load();
                st_.ms_mmap = ms_since(t1);
            } catch (const std::exception& ex) { mm_err = std::string("tier mmap reader: ") + ex.what(); }
        });
        if (relax) sched_setaffinity(0, sizeof saved_aff, &saved_aff);   // the host thread back on its partition
    }

    // ---- groups: consecutive runs of the static/pinned occupied experts whose PINNED members
    // fit the stream slots. EVERY pinned expert in a group holds a stream slot for the group's
    // duration -- a hit that is already resident keeps its slot in-use just as a miss does --
    // so the cap counts all of them, not only the misses. (The first version counted misses
    // alone; on a warm second call the hits filled the partition and the misses had nowhere
    // to land.) Two groups are live at once under the fetch pipeline (group g+1's H2D lands
    // while group g's GEMMs read their slots), so a group may claim at most
    // pipelined_group_cap() -- half the streaming partition. With too few evictable slots to
    // double-buffer, fall back to the serial shape and the full partition.
    const auto t2 = std::chrono::steady_clock::now();
    st_.ms_spawn = ms_since(t1);                          // the transient bank's alloc + the reader's start
    const bool pipe = cache_.pipelined_group_cap() != 0;
    const uint32_t cap = pipe ? cache_.pipelined_group_cap() : cache_.stream_slots();
    std::vector<uint32_t> gb{0};
    { uint32_t need = 0;
      for (uint32_t i = 0; i < occ.size(); ++i) {
          const bool takes_slot = tier_[l][occ[i]] == 1;
          if (takes_slot && need == cap) { gb.push_back(i); need = 0; }
          if (takes_slot) ++need;
      }
      gb.push_back(uint32_t(occ.size())); }
    const size_t NG = occ.empty() ? 0 : gb.size() - 1;
    auto* xq8 = static_cast<block_q8_1x*>(bws_.xp_q8);
    auto* hq8 = static_cast<block_q8_1x*>(bws_.h_q8);
    // DSpark P2 (docs/deepseek41/55): a multi-row decode step (2..8 rows) runs an expert shared by several rows as one
    // job PER ROW, so every row takes the one-row job's arithmetic (the grouped GEMM's M-tile changes the summation
    // otherwise): bit-identical to T one-row steps; the expert's slot is read once per row from VRAM (a decode-only cost)
    const bool row_jobs = T >= 2 && T <= 8;
    // Phase 42 (docs/deepseek41/82): the PREFILL groups (T > 8 rows: strips of a chunk, never a decode or DSpark
    // step) take the fused MXFP4 -> XMX route -- W4A16 with fp32 accumulation, the route DS4's model has run since
    // 2026-08-09 on a PPL gate that measured it better than the int-dot W4A8 kernel and 4-10x faster. gate/up land
    // fp32 in g_f/u_f, the fp32 -> fp16 SwiGLU feeds down as fp16 rows, down lands fp32 in y_f32 and the fp32
    // scatter applies the routing weights. Decode (T = 1, the CPU leg) and DSpark rows (2..8, per-row int-dot jobs)
    // are untouched bit for bit. IE_DS4_EXPERT_XMX=0 keeps the int-dot route everywhere; an EP tier keeps it too
    // (its rows are exported as fp16 from yp).
    // OPT-IN (IE_DS41_PREFILL_XMX=1) after the Phase 42 gate: +8-14 % prefill and better per-layer bars, but the
    // decode test's own-router flip at a 4.6e-3 margin (above its 3e-3 near-tie bar) and the cont test's
    // chunks-of-16 arm (a W4A16 chunk against W4A8 one-token steps) miss their bars as written -- the founder's call.
    // DEFAULT ON since 2026-09-17 (founder decision on docs/83's perplexity: +0.0022 nats, t = 1.81; +5-8 % prefill on the
    // pipelined stack). IE_DS41_PREFILL_XMX=0 restores the int-dot W4A8 route.
    static const bool prefill_xmx = [] { const char* e = std::getenv("IE_DS41_PREFILL_XMX"); return !(e && *e && std::string(e) == "0"); }();
    const bool xmx = prefill_xmx && ds4_expert_xmx_on() && T > 8 && !ep_export_ && !ep_import_ && bws_.g_f && bws_.u_f && bws_.y_f32;
    static thread_local std::vector<DS4XmxGemmJob> xmx_jobs;
    auto push_job = [&](const DS4ExpertBank& bank, uint32_t n_e, const void* x_in, uint32_t x_row_blocks, sycl::half* out, uint32_t out_stride, uint32_t o) {
        if (!row_jobs || n_e <= 1) { bws_.jobs.push_back({bank, 0, n_e, x_in, out, o}); return; }
        for (uint32_t r = 0; r < n_e; ++r) bws_.jobs.push_back({bank, 0, 1u, static_cast<const block_q8_1x*>(x_in) + uint64_t(r) * x_row_blocks, out + uint64_t(r) * out_stride, o + r});
    };
    // THE FETCH PIPELINE (V4's, measured 1.8-1.9x on this box): group g+1's acquire is issued
    // BEFORE group g's GEMMs, so the copy engine and the EUs overlap instead of alternating.
    // acquire_pipelined keeps the previous group's slots protected; the drain at the end of each
    // group is what makes issuing g+2 safe. Serial `acquire` when the partition cannot double-buffer.
    std::vector<std::vector<int32_t>>  gids(NG);
    std::vector<std::vector<uint32_t>> gslot(NG);
    std::vector<sycl::event>           gev(NG);
    std::string gerr;
    auto issue = [&](size_t g) {
        for (uint32_t i = gb[g]; i < gb[g + 1]; ++i) gids[g].push_back(int32_t(occ[i]));
        gslot[g].assign(gids[g].size(), kDs4NoSlot);
        if (!gids[g].empty())
            gev[g] = pipe ? cache_.acquire_pipelined(l, gids[g].data(), uint32_t(gids[g].size()), gslot[g].data())
                          : cache_.acquire(l, gids[g].data(), uint32_t(gids[g].size()), gslot[g].data());
        for (size_t i = 0; i < gslot[g].size(); ++i)
            if (gslot[g][i] == kDs4NoSlot && gerr.empty())
                gerr = "tier: expert " + std::to_string(gids[g][i]) + " has no VRAM slot after acquire";
    };
    auto fail = [&](const std::string& e) { if (reader >= 0) { mm_reader_wait(reader); reader = -1; } return e; };
    if (NG) issue(0);
    if (pre_groups_) pre_groups_();                       // Phase 18: the caller's work under group 0's DMAs
    for (size_t g = 0; g < NG; ++g) {
        if (pipe && g + 1 < NG) issue(g + 1);            // overlaps the compute below
        if (!gerr.empty()) return fail(gerr);
        std::unordered_map<int32_t, uint32_t> slot_of;
        for (size_t i = 0; i < gids[g].size(); ++i) {
            slot_of[gids[g][i]] = gslot[g][i];
            if (tier_[l][gids[g][i]] == 0) ++st_.experts_static; else ++st_.experts_pinned;
        }
        const sycl::event fetched = gev[g];
        q.submit([&](sycl::handler& h) { h.depends_on(fetched); h.single_task([] {}); });
        // gate + up jobs
        bws_.jobs.clear(); xmx_jobs.clear();
        for (uint32_t i = gb[g]; i < gb[g + 1]; ++i) {
            const uint32_t e = occ[i]; const uint32_t o = off[e], n_e = off[e + 1] - o;
            const void* xe = xq8 + uint64_t(o) * (H / 32);
            void* base = cache_.slot_ptr(l, slot_of[int32_t(e)]);
            if (xmx) {
                xmx_jobs.push_back({ds4_slot_bank(lay_.gate, base), 0, n_e, o, bws_.g_f + uint64_t(o) * EF});
                xmx_jobs.push_back({ds4_slot_bank(lay_.up,   base), 0, n_e, o, bws_.u_f + uint64_t(o) * EF});
                continue;
            }
            push_job(ds4_slot_bank(lay_.gate, base), n_e, xe, H / 32, bws_.g_h + uint64_t(o) * EF, EF, o);
            push_job(ds4_slot_bank(lay_.up,   base), n_e, xe, H / 32, bws_.u_h + uint64_t(o) * EF, EF, o);
        }
        if (xmx) { if (auto e = ds4_expert_gemm_xmx_grouped(q, xmx_jobs.data(), uint32_t(xmx_jobs.size()), bws_.xp, H, bws_.w16, bws_.w16_cap); !e.empty()) return fail("tier gate/up (xmx): " + e); }
        else if (auto e = ds4_expert_gemm_q8_grouped(q, bws_.jobs.data(), uint32_t(bws_.jobs.size()), bws_.grp); !e.empty()) return fail("tier gate/up: " + e);
        // The group's packed rows are contiguous in expert order; an mmap expert's rows that
        // fall inside the range get a throwaway activation here that its own group below
        // recomputes -- never read in between.
        const uint32_t r0 = off[occ[gb[g]]], nr = off[occ[gb[g + 1] - 1] + 1] - r0;
        const uint64_t HN = uint64_t(nr) * EF;
        if (xmx) ds4_swiglu_clamped_to_f16(q, bws_.g_f + uint64_t(r0) * EF, bws_.u_f + uint64_t(r0) * EF, bws_.h_h + uint64_t(r0) * EF, HN, swiglu_limit);
        else {
        ds4_swiglu_clamped_h(q, bws_.g_h + uint64_t(r0) * EF, bws_.u_h + uint64_t(r0) * EF, bws_.h_h + uint64_t(r0) * EF, HN, swiglu_limit);
        quantize_q8_1(q, bws_.h_h + uint64_t(r0) * EF, hq8 + uint64_t(r0) * (EF / 32), uint32_t(HN));
        }
        // down jobs
        bws_.jobs.clear(); xmx_jobs.clear();
        for (uint32_t i = gb[g]; i < gb[g + 1]; ++i) {
            const uint32_t e = occ[i]; const uint32_t o = off[e], n_e = off[e + 1] - o;
            const void* he = hq8 + uint64_t(o) * (EF / 32);
            if (xmx) { xmx_jobs.push_back({ds4_slot_bank(lay_.down, cache_.slot_ptr(l, slot_of[int32_t(e)])), 0, n_e, o, bws_.y_f32 + uint64_t(o) * H}); continue; }
            push_job(ds4_slot_bank(lay_.down, cache_.slot_ptr(l, slot_of[int32_t(e)])), n_e, he, EF / 32, static_cast<sycl::half*>(bws_.yp) + uint64_t(o) * H, H, o);
        }
        if (xmx) { if (auto e = ds4_expert_gemm_xmx_grouped(q, xmx_jobs.data(), uint32_t(xmx_jobs.size()), bws_.h_h, EF, bws_.w16, bws_.w16_cap); !e.empty()) return fail("tier down (xmx): " + e); }
        else if (auto e = ds4_expert_gemm_q8_grouped(q, bws_.jobs.data(), uint32_t(bws_.jobs.size()), bws_.grp); !e.empty()) return fail("tier down: " + e);
        q.wait_and_throw();   // the drain: group g's slots are free for group g+2's fetch
        if (!pipe && g + 1 < NG) issue(g + 1);
    }
    st_.ms_groups = ms_since(t2);

    // ---- the mmap group: after the fill has landed (the reader waited on its own queue) ----
    const auto tj = std::chrono::steady_clock::now();
    if (reader >= 0) { mm_reader_wait(reader); reader = -1; }
    st_.ms_join = ms_since(tj);                           // the fill's excess over the groups, when it is the longer leg
    if (!mm_err.empty()) return mm_err;
    const auto t3 = std::chrono::steady_clock::now();
    if (!mm.empty()) {
        auto mm_bank = [&](const Ds4MatPlanes& pl, size_t i) { return ds4_slot_bank(pl, mm_dev + i * lay_.bytes); };
        bws_.jobs.clear(); xmx_jobs.clear();
        for (size_t p = 0; p < mm.size(); ++p) {
            const uint32_t e = mm[p]; const uint32_t o = off[e], n_e = off[e + 1] - o;
            const void* xe = xq8 + uint64_t(o) * (H / 32);
            if (xmx) {
                xmx_jobs.push_back({mm_bank(lay_.gate, p), 0, n_e, o, bws_.g_f + uint64_t(o) * EF});
                xmx_jobs.push_back({mm_bank(lay_.up,   p), 0, n_e, o, bws_.u_f + uint64_t(o) * EF});
                continue;
            }
            push_job(mm_bank(lay_.gate, p), n_e, xe, H / 32, bws_.g_h + uint64_t(o) * EF, EF, o);
            push_job(mm_bank(lay_.up,   p), n_e, xe, H / 32, bws_.u_h + uint64_t(o) * EF, EF, o);
        }
        if (xmx) { if (auto e = ds4_expert_gemm_xmx_grouped(q, xmx_jobs.data(), uint32_t(xmx_jobs.size()), bws_.xp, H, bws_.w16, bws_.w16_cap); !e.empty()) return fail("tier mmap gate/up (xmx): " + e); }
        else if (auto e = ds4_expert_gemm_q8_grouped(q, bws_.jobs.data(), uint32_t(bws_.jobs.size()), bws_.grp); !e.empty()) return fail("tier mmap gate/up: " + e);
        for (size_t p = 0; p < mm.size(); ++p) {          // these rows are scattered: per expert
            const uint32_t e = mm[p]; const uint32_t o = off[e], n_e = off[e + 1] - o;
            const uint64_t HN = uint64_t(n_e) * EF;
            if (xmx) { ds4_swiglu_clamped_to_f16(q, bws_.g_f + uint64_t(o) * EF, bws_.u_f + uint64_t(o) * EF, bws_.h_h + uint64_t(o) * EF, HN, swiglu_limit); continue; }
            ds4_swiglu_clamped_h(q, bws_.g_h + uint64_t(o) * EF, bws_.u_h + uint64_t(o) * EF, bws_.h_h + uint64_t(o) * EF, HN, swiglu_limit);
            quantize_q8_1(q, bws_.h_h + uint64_t(o) * EF, hq8 + uint64_t(o) * (EF / 32), uint32_t(HN));
        }
        bws_.jobs.clear(); xmx_jobs.clear();
        for (size_t p = 0; p < mm.size(); ++p) {
            const uint32_t e = mm[p]; const uint32_t o = off[e], n_e = off[e + 1] - o;
            if (xmx) { xmx_jobs.push_back({mm_bank(lay_.down, p), 0, n_e, o, bws_.y_f32 + uint64_t(o) * H}); continue; }
            push_job(mm_bank(lay_.down, p), n_e, hq8 + uint64_t(o) * (EF / 32), EF / 32, static_cast<sycl::half*>(bws_.yp) + uint64_t(o) * H, H, o);
        }
        if (xmx) { if (auto e = ds4_expert_gemm_xmx_grouped(q, xmx_jobs.data(), uint32_t(xmx_jobs.size()), bws_.h_h, EF, bws_.w16, bws_.w16_cap); !e.empty()) return fail("tier mmap down (xmx): " + e); }
        else if (auto e = ds4_expert_gemm_q8_grouped(q, bws_.jobs.data(), uint32_t(bws_.jobs.size()), bws_.grp); !e.empty()) return fail("tier mmap down: " + e);
        q.wait_and_throw();
        st_.ms_mmap_group = ms_since(t3);
    }
    const auto t4 = std::chrono::steady_clock::now();
    if (cpu_leg) {   // Phase 20: the CPU experts' rows land in their packed positions before the export wait / the scatter
        { std::unique_lock<std::mutex> lk(cpu_.mu); cpu_.cv.wait(lk, [&] { return cpu_.done; }); }
        st_.ms_cpu = ms_since(tc); st_.ms_cpu_work = cpu_.work_ms;
        for (size_t i = 0; i < cpu_.work.size(); ++i)
            q.memcpy(static_cast<sycl::half*>(bws_.yp) + uint64_t(cpu_.work[i].second) * H, cpu_.h_rows + i * H, size_t(H) * 2);
    }
    if (ep_export_) { q.wait(); }                                                         // the rows are the product; no scatter
    else {
        if (ep_import_) if (auto e = ep_import_(q, static_cast<sycl::half*>(bws_.yp), TK); !e.empty()) return "tier ep import: " + e;
        if (xmx) ds4_expert_scatter_accum_f32(q, bws_.y_f32, bws_.tk2p, bws_.w_pk, y, T, K, H).wait();
        else ds4_expert_scatter_accum(q, bws_.yp, bws_.tk2p, bws_.w_pk, y, T, K, H).wait();
    }
    st_.ms_tail = ms_since(t4);
    st_.bytes_pinned_to_vram = cache_.stats().bytes_fetched - fetched0;      // actual: a hit in a stream slot moves nothing
    st_.stream_hits = uint32_t(cache_.stats().stream_hits - shits0);
    st_.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return {};
}

}  // namespace ie
