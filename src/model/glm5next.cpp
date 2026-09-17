// src/model/glm5next.cpp — GLM-5.3-Flash (`glm5next`) loader (v0).
//
// Placement contract in include/ie/glm5next.hpp; port plan in
// docs/glm53/PORT_PLAN.md. This translation unit is LOAD ONLY — the forward
// lands on top of it (port P2) so the placement can gate first, exactly the
// qwen4exp bring-up sequence.

#include "ie/glm5next.hpp"

#include "ie/cpu_moe_gemv.hpp"
#include "ie/deepseek4.hpp"       // ds4_hc_mix (the DecoderLayer residual mix)
#include "ie/deepseek4_ops.hpp"
#include "ie/dequant.hpp"
#include "ie/dequant_ref.hpp"
#include "ie/dtype.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/deepseek4_attn.hpp"   // ds4_indexer_score / ds4_indexer_topk (P2)
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <algorithm>
#include <cmath>
#include <pthread.h>
#include <sched.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <stdexcept>
#include <sys/mman.h>
#include <vector>

namespace ie {

namespace {

inline float bf16_to_fp32(uint16_t v) {
    uint32_t u = uint32_t(v) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

// Dequantize one whole GGUF tensor to fp32, GGUF element order. Covers every
// dtype the real UD-Q4_K_XL file uses for DEVICE-bound tensors (F32/F16/Q8_0)
// plus the K-quants so a differently-quantized glm5next GGUF still loads.
std::string dequant_tensor_fp32(const GgufTensorInfo* t, std::vector<float>& out) {
    uint64_t n = 1;
    for (uint32_t d = 0; d < t->n_dims; ++d) n *= t->shape[d];
    out.resize(n);
    switch (t->dtype) {
        case DType::kF32:
            std::memcpy(out.data(), t->data, n * sizeof(float));
            return {};
        case DType::kF16: {
            const auto* s = reinterpret_cast<const uint16_t*>(t->data);
            for (uint64_t i = 0; i < n; ++i)
                out[i] = float(sycl::bit_cast<sycl::half>(s[i]));
            return {};
        }
        case DType::kBF16: {
            const auto* s = reinterpret_cast<const uint16_t*>(t->data);
            for (uint64_t i = 0; i < n; ++i) out[i] = bf16_to_fp32(s[i]);
            return {};
        }
        case DType::kQ8_0:  ref::dequant_q8_0_buffer(t->data, n, out.data()); return {};
        case DType::kQ4_K:  ref::dequant_q4_K_buffer(t->data, n, out.data()); return {};
        case DType::kQ5_K:  ref::dequant_q5_K_buffer(t->data, n, out.data()); return {};
        case DType::kQ6_K:  ref::dequant_q6_K_buffer(t->data, n, out.data()); return {};
        case DType::kQ5_1:  ref::dequant_q5_1_buffer(t->data, n, out.data()); return {};
        case DType::kQ3_K:    ref::dequant_q3_K_buffer(t->data, n, out.data()); return {};
        case DType::kIQ4_XS:  ref::dequant_iq4_xs_buffer(t->data, n, out.data()); return {};
        case DType::kIQ3_XXS: ref::dequant_iq3_xxs_buffer(t->data, n, out.data()); return {};
        default:
            return std::string("glm5next load: unsupported device-tensor dtype ") +
                   std::string(type_name(t->dtype));
    }
}

}  // namespace

static void pin_copy_thread();

void Glm5NextModel::CopyPool::start(uint32_t nw) {
    if (!th.empty()) return;   // idempotent (load-time pinning starts it early)
    workers = nw;
    for (uint32_t w = 0; w < nw; ++w)
        th.emplace_back([this, w] {
            pin_copy_thread();
            uint32_t last = 0;
            for (;;) {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop || seq != last; });
                if (stop) return;
                last = seq;
                const uint8_t* s = src; uint8_t* d = dst; const uint64_t nn = n;
                lk.unlock();
                // worker w owns stripe w of (workers + 1); the caller takes
                // the last stripe so all cores copy concurrently.
                const uint64_t per = (nn + workers) / (workers + 1);
                const uint64_t o = uint64_t(w) * per;
                if (o < nn) std::memcpy(d + o, s + o, std::min(per, nn - o));
                lk.lock();
                if (++done == workers) cv_done.notify_one();
            }
        });
}
void Glm5NextModel::CopyPool::copy(uint8_t* d, const void* s, uint64_t bytes) {
    if (th.empty()) { std::memcpy(d, s, bytes); return; }
    {
        std::lock_guard<std::mutex> lk(mu);
        src = static_cast<const uint8_t*>(s); dst = d; n = bytes;
        done = 0; ++seq;
    }
    cv.notify_all();
    const uint64_t per = (bytes + workers) / (workers + 1);
    const uint64_t o = uint64_t(workers) * per;
    if (o < bytes) std::memcpy(d + o, static_cast<const uint8_t*>(s) + o, bytes - o);
    std::unique_lock<std::mutex> lk(mu);
    cv_done.wait(lk, [&] { return done == workers; });
}
Glm5NextModel::CopyPool::~CopyPool() {
    { std::lock_guard<std::mutex> lk(mu); stop = true; }
    cv.notify_all();
    for (auto& t : th) t.join();
}

// IE_G5_PIN_CPUS=1: pin copy-path threads (fill worker, pread pools) to the
// P-core set 0-7. The unpinned scheduler migrates them across P/E cores and
// the H2D submission path collapses (measured 2026-08-29: card0 10.6 GB/s
// unpinned vs 21 pinned, card1 20 vs 26, standalone). Compute threads stay
// free — a full-process 8-core pin cost decode 5.95 -> 4.24.
static void pin_copy_thread() {
    static const bool on = [] {
        const char* v = std::getenv("IE_G5_PIN_CPUS");
        return v && v[0] == '1';
    }();
    if (!on) return;
    cpu_set_t s;
    CPU_ZERO(&s);
    for (int c = 0; c < 8; ++c) CPU_SET(c, &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

void Glm5NextModel::FillThread::start(Glm5NextModel* m) {
    owner = m;
    th = std::thread([this, m] {
        pin_copy_thread();
        for (;;) {
            FillReq r;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop || !reqs.empty(); });
                if (stop && reqs.empty()) return;
                r = std::move(reqs.front());
                reqs.pop_front();
            }
            if (r.pf) {
                // Prefetch: pf-owned ring + pool — the shared pin_ring_/prp_/
                // cpool_ belong to the main thread's inline fills at decode.
                uint8_t* pin = m->pf_pin_[m->pf_pin_idx_];
                m->pf_pin_ev_[m->pf_pin_idx_].wait();
                if (auto e2 = m->pread_fill_(r.gfd, r.goff, r.g, r.gsl,
                                             r.ufd, r.uoff, r.u, r.usl,
                                             r.dfd, r.doff, r.d2, r.dsl, pin,
                                             &m->pf_prp_, /*serial_cp=*/true);
                    !e2.empty()) {
                    // The slot is already CLAIMED (bookkeeping on the issuing
                    // thread) — a bare return would leave it poisoned. The
                    // mmap memcpy path cannot fail; fall back loudly.
                    std::fprintf(stderr, "[glm5next] pf pread failed (%s) — memcpy fallback\n",
                                 e2.c_str());
                    (void)m->pread_fill_(-1, 0, r.g, r.gsl, -1, 0, r.u, r.usl,
                                         -1, 0, r.d2, r.dsl, pin,
                                         nullptr, /*serial_cp=*/true);
                }
                m->pf_pin_ev_[m->pf_pin_idx_] = m->copyq_->memcpy(
                    r.dst, pin, r.gsl + r.usl + r.dsl, {r.fence});
                r.done.set_value(m->pf_pin_ev_[m->pf_pin_idx_]);
                m->pf_pin_idx_ = (m->pf_pin_idx_ + 1) % 2;
            } else if (r.stream) {
                // IE_G5_PP_STREAM whole-layer load: three [E x slice] regions
                // chunked through the pin ring — pread into pinned, H2D on
                // the in-order copyq_. Striped mmap memcpy was slower
                // (pool_copy 14 s vs pread 10 s). First piece carries the
                // buffer-reuse fence; copyq_ in-order covers the rest.
                static const bool use_pread = [] {
                    const char* v = std::getenv("IE_G5_PREAD");
                    return !(v && v[0] == '0');
                }();
                m->prp_.start(6);
                sycl::event last;
                bool first = true;
                uint32_t pb = 0;   // stream-owned bounce cursor
                auto region = [&](int fd, uint64_t foff, const uint8_t* src,
                                  uint64_t len, uint8_t* dst) {
                    for (uint64_t o = 0; o < len; o += kPpPinSz) {
                        const uint64_t pc = std::min(kPpPinSz, len - o);
                        uint8_t* pin = m->pp_pin_[pb];
                        auto tw0 = std::chrono::steady_clock::now();
                        m->pp_pin_ev_[pb].wait();
                        auto tw1 = std::chrono::steady_clock::now();
                        bool ok = false;
                        if (use_pread && fd >= 0) {
                            constexpr uint64_t SL = 4ull << 20;
                            std::vector<PreadPool::Job> jobs;
                            jobs.reserve(size_t((pc + SL - 1) / SL));
                            for (uint64_t s = 0; s < pc; s += SL)
                                jobs.push_back({fd, foff + o + s,
                                                std::min(SL, pc - s), pin + s});
                            ok = m->prp_.run(jobs.data(), jobs.size());
                            if (!ok)
                                std::fprintf(stderr, "[glm5next] pp-stream pread "
                                             "failed — memcpy fallback\n");
                        }
                        if (!ok) std::memcpy(pin, src + o, pc);
                        m->t_ring_wait += std::chrono::duration<double>(tw1 - tw0).count();
                        m->t_pool_copy += std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - tw1).count();
                        last = m->copyq_->memcpy(dst + o, pin, pc,
                            first ? std::vector<sycl::event>{r.fence}
                                  : std::vector<sycl::event>{});
                        first = false;
                        m->pp_pin_ev_[pb] = last;
                        pb = (pb + 1) % uint32_t(m->pp_pin_.size());
                    }
                };
                region(r.gfd, r.goff, r.g, r.gsl, r.dst);
                region(r.ufd, r.uoff, r.u, r.usl, r.dst + r.gsl);
                region(r.dfd, r.doff, r.d2, r.dsl, r.dst + r.gsl + r.usl);
                r.done.set_value(last);
            } else if (r.bounce) {
                uint8_t* pin = m->pin_ring_[m->pin_idx_];
                auto tw0 = std::chrono::steady_clock::now();
                m->pin_ev_[m->pin_idx_].wait();
                auto tw1 = std::chrono::steady_clock::now();
                if (auto e2 = m->pread_fill_(r.gfd, r.goff, r.g, r.gsl,
                                             r.ufd, r.uoff, r.u, r.usl,
                                             r.dfd, r.doff, r.d2, r.dsl, pin);
                    !e2.empty()) {
                    std::fprintf(stderr, "[glm5next] %s\n", e2.c_str());
                    r.done.set_value(sycl::event{});
                    continue;
                }
                auto tw2 = std::chrono::steady_clock::now();
                m->pin_ev_[m->pin_idx_] = m->copyq_->memcpy(
                    r.dst, pin, r.gsl + r.usl + r.dsl, {r.fence});
                m->t_ring_wait += std::chrono::duration<double>(tw1 - tw0).count();
                m->t_pool_copy += std::chrono::duration<double>(tw2 - tw1).count();
                r.done.set_value(m->pin_ev_[m->pin_idx_]);
                m->pin_idx_ = (m->pin_idx_ + 1) % kPinRing;
            } else {
                // direct pageable H2D from the mmap: the runtime stages
                // internally; no host bytes on our threads at all.
                m->copyq_->memcpy(r.dst, r.g, r.gsl, {r.fence});
                m->copyq_->memcpy(r.dst + r.gsl, r.u, r.usl);
                sycl::event ev = m->copyq_->memcpy(r.dst + r.gsl + r.usl, r.d2, r.dsl);
                r.done.set_value(ev);
            }
        }
    });
}
std::shared_future<sycl::event> Glm5NextModel::FillThread::push(FillReq&& r) {
    // G2: plain bounce fills go to the lanes when they run (pf / stream keep
    // this thread — their rings have one consumer)
    if (owner && owner->lanes_.running() && r.bounce && !r.pf && !r.stream)
        return owner->lanes_.push(std::move(r));
    auto fut = r.done.get_future().share();
    {
        std::lock_guard<std::mutex> lk(mu);
        reqs.push_back(std::move(r));
    }
    cv.notify_one();
    return fut;
}
Glm5NextModel::FillThread::~FillThread() {
    { std::lock_guard<std::mutex> lk(mu); stop = true; }
    cv.notify_all();
    if (th.joinable()) th.join();
}

namespace {
// G1: the q* CPU-miss worker of each stage needs its own cores. P-cores are the
// CPUs sharing the highest acpi_cppc/highest_perf (this rig: 90 on 0-7, 68 on
// the 12 E-cores 8-19); the first stage gets the P block, the tail stage the E
// block. Contiguous blocks only (CpuWorker pins a lo-hi range); a flat or
// missing signal falls back to halves; fewer than 4 CPUs -> no pinning.
// Counts hardware_concurrency(), not the process's allowed set: under a cpuset
// or an inherited taskset the pin can fail — CpuWorker::start reports that and
// runs unpinned.
struct G5CoreSplit { int a_lo = -1, a_hi = -1, b_lo = -1, b_hi = -1; };
G5CoreSplit g5_auto_core_split() {
    const int n = int(std::thread::hardware_concurrency());
    std::vector<int> perf(size_t(std::max(n, 0)), -1);
    int best = -1;
    for (int c = 0; c < n; ++c) {
        char path[96];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/acpi_cppc/highest_perf", c);
        if (FILE* f = std::fopen(path, "r")) {
            int v = -1;
            if (std::fscanf(f, "%d", &v) == 1) perf[size_t(c)] = v;
            std::fclose(f);
        }
        best = std::max(best, perf[size_t(c)]);
    }
    int p_lo = -1, p_hi = -1, e_lo = -1, e_hi = -1;
    for (int c = 0; c < n; ++c) {
        if (best > 0 && perf[size_t(c)] == best) { if (p_lo < 0) p_lo = c; p_hi = c; }
        else                                       { if (e_lo < 0) e_lo = c; e_hi = c; }
    }
    if (p_lo == 0 && p_hi >= 0 && p_hi + 1 == e_lo && e_hi == n - 1) return {p_lo, p_hi, e_lo, e_hi};
    if (n >= 4) return {0, n / 2 - 1, n / 2, n - 1};
    return {};
}
// One segment of an expert from the bank file into a pinned buffer: exact
// bytes, EINTR retried; false on a short read.  Falls back to memcpy from the
// mmap when there is no fd.
bool g5_lane_read(int fd, uint64_t off, const uint8_t* src, uint64_t len, uint8_t* dst) {
    if (fd < 0) { std::memcpy(dst, src, len); return true; }
    uint64_t got = 0;
    while (got < len) {
        const ssize_t r = pread(fd, dst + got, len - got, off + got);
        if (r < 0) { if (errno == EINTR) continue; return false; }
        if (r == 0) return false;
        got += uint64_t(r);
    }
    return true;
}
}  // namespace

std::string Glm5NextModel::FillLanes::start(Glm5NextModel* owner, uint32_t n, uint64_t pin_sz) {
    if (!workers.empty() || n == 0) return {};
    m = owner;
    for (uint32_t i = 0; i < kBufs; ++i) {
        bufs[i] = static_cast<uint8_t*>(sycl::malloc_host(pin_sz, m->alloc_->queue()));
        if (!bufs[i]) { shutdown(); return "glm5next fill lanes: pinned bounce alloc failed"; }
    }
    for (uint32_t i = 0; i < n; ++i) workers.emplace_back([this] {
        pin_copy_thread();
        for (;;) {
            Seg sg;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop || !segs.empty(); });
                if (stop && segs.empty()) return;
                sg = std::move(segs.front());
                segs.pop_front();
            }
            const auto tw1 = std::chrono::steady_clock::now();
            if (!g5_lane_read(sg.fd, sg.off, sg.src, sg.len, sg.dst)) sg.req->failed.store(true);
            const auto tw2 = std::chrono::steady_clock::now();
            {
                std::lock_guard<std::mutex> lk(tmu);
                m->t_pool_copy += std::chrono::duration<double>(tw2 - tw1).count();
            }
            if (sg.req->left.fetch_sub(1) == 1) {   // the expert's last segment: issue its H2D
                Req& q = *sg.req;
                if (q.failed.load()) {
                    std::fprintf(stderr, "[glm5next] fill lane: pread failed (short read / IO error) — refusing DMA\n");
                    q.r.done.set_value(sycl::event{});
                } else {
                    const uint64_t n = q.r.gsl + q.r.usl + q.r.dsl;
                    // Diagnostic (IE_G5_LANE_H2D_ON_Q=1, det7): issue the H2D on the
                    // COMPUTE queue instead of copyq_ — same engine as the readers, no
                    // cross-engine visibility question. det6 showed one fused kernel
                    // reading correct gate bytes and stale up bytes of the same slot.
                    static const bool h2d_on_q = std::getenv("IE_G5_LANE_H2D_ON_Q") != nullptr;
                    sycl::event ev = h2d_on_q
                        ? m->alloc_->queue().memcpy(q.r.dst, q.pin, n, {q.r.fence})
                        : m->copyq_->memcpy(q.r.dst, q.pin, n, {q.r.fence});
                    // Diagnostic (IE_G5_LANE_VERIFY=1, det3): before releasing the
                    // compute, prove both legs of THIS fill — the landed slot must
                    // equal the pinned buffer (DMA leg) and the pinned buffer must
                    // equal a fresh serial pread of the expert (pread leg). Any
                    // mismatch is printed with its first differing offset.
                    static const bool verify = std::getenv("IE_G5_LANE_VERIFY") != nullptr;
                    if (verify) {
                        thread_local std::vector<uint8_t> slot, fresh;
                        slot.resize(n); fresh.resize(n);
                        ev.wait();
                        m->copyq_->memcpy(slot.data(), q.r.dst, n).wait();
                        bool okf = g5_lane_read(q.r.gfd, q.r.goff, q.r.g,  q.r.gsl, fresh.data())
                                && g5_lane_read(q.r.ufd, q.r.uoff, q.r.u,  q.r.usl, fresh.data() + q.r.gsl)
                                && g5_lane_read(q.r.dfd, q.r.doff, q.r.d2, q.r.dsl, fresh.data() + q.r.gsl + q.r.usl);
                        auto first_diff = [&](const uint8_t* a, const uint8_t* b) -> int64_t {
                            for (uint64_t i = 0; i < n; ++i) if (a[i] != b[i]) return int64_t(i);
                            return -1;
                        };
                        const int64_t d_dma = first_diff(slot.data(), q.pin);
                        const int64_t d_pre = okf ? first_diff(fresh.data(), q.pin) : -2;
                        static std::atomic<uint64_t> n_ver{0}, n_bad{0};
                        const uint64_t k = ++n_ver;
                        if (d_dma >= 0 || d_pre != -1) {
                            ++n_bad;
                            std::fprintf(stderr, "[glm5next] LANE VERIFY MISMATCH #%llu (fill %llu): buf %u dst %p bytes %llu — DMA leg first diff %lld, pread leg first diff %lld (%s)\n",
                                         (unsigned long long)n_bad.load(), (unsigned long long)k, q.buf, (void*)q.r.dst,
                                         (unsigned long long)n, (long long)d_dma, (long long)d_pre,
                                         d_pre == -2 ? "fresh pread failed" : "");
                        } else if ((k % 20000) == 0) {
                            std::fprintf(stderr, "[glm5next] lane verify: %llu fills checked, %llu mismatches\n",
                                         (unsigned long long)k, (unsigned long long)n_bad.load());
                        }
                    }
                    q.r.done.set_value(ev);
                }
            }
        }
    });
    return {};
}

std::shared_future<sycl::event> Glm5NextModel::FillLanes::push(FillReq&& r) {
    // Single producer (the submit thread): take the next ring buffer, waiting
    // for the H2D that last read it (kBufs deep, so rarely).
    auto req = std::make_shared<Req>();
    req->r = std::move(r);
    auto fut = req->r.done.get_future().share();
    const uint32_t b = buf_next;
    buf_next = (buf_next + 1) % kBufs;
    // The buffer's previous request must have ISSUED its H2D (its promise
    // resolved) and that H2D must have finished reading the buffer — waiting
    // only on a stored event would race a request whose segments are still in
    // flight (its event not stored yet).
    const auto tw0 = std::chrono::steady_clock::now();
    if (buf_fut[b].valid()) { buf_fut[b].wait(); sycl::event prev = buf_fut[b].get(); prev.wait(); }
    {
        std::lock_guard<std::mutex> lk(tmu);
        m->t_ring_wait += std::chrono::duration<double>(std::chrono::steady_clock::now() - tw0).count();
    }
    buf_fut[b] = fut;
    req->pin = bufs[b];
    req->buf = b;
    const FillReq& fr = req->r;
    std::vector<Seg> cut;
    auto seg = [&](int fd, uint64_t off, const uint8_t* src, uint64_t len, uint8_t* dst) {
        for (uint64_t o = 0; o < len; o += kSeg)
            cut.push_back({req, fd, off + o, src + o, std::min(kSeg, len - o), dst + o});
    };
    seg(fr.gfd, fr.goff, fr.g,  fr.gsl, req->pin);
    seg(fr.ufd, fr.uoff, fr.u,  fr.usl, req->pin + fr.gsl);
    seg(fr.dfd, fr.doff, fr.d2, fr.dsl, req->pin + fr.gsl + fr.usl);
    req->left.store(uint32_t(cut.size()));
    {
        std::lock_guard<std::mutex> lk(mu);
        for (auto& c : cut) segs.push_back(std::move(c));
    }
    cv.notify_all();
    return fut;
}

void Glm5NextModel::FillLanes::shutdown() noexcept {
    { std::lock_guard<std::mutex> lk(mu); stop = true; }
    cv.notify_all();
    for (std::thread& t : workers) if (t.joinable()) t.join();
    if (m && m->copyq_) { try { m->copyq_->wait(); } catch (...) {} }   // nothing may still read a buffer
    for (uint32_t i = 0; i < kBufs; ++i)
        if (bufs[i] && m) { sycl::free(bufs[i], m->alloc_->queue()); bufs[i] = nullptr; }
    workers.clear();
    for (auto& f : buf_fut) f = {};
    stop = false;
    buf_next = 0;
}

Glm5NextModel::~Glm5NextModel() { free_all(); }

void Glm5NextModel::shutdown() noexcept {
    if (shutting_down_) return;
    shutting_down_ = true;
    {
        std::lock_guard<std::mutex> lk(fill_.mu);
        fill_.stop = true;
    }
    fill_.cv.notify_all();
    if (fill_.th.joinable()) fill_.th.join();
    lanes_.shutdown();
    try { if (copyq_) copyq_->wait(); } catch (...) {}
    try { if (jobsq_) jobsq_->wait(); } catch (...) {}
    try { if (alloc_ && alloc_->ready()) alloc_->queue().wait(); } catch (...) {}
}

void Glm5NextModel::free_all() {
    if (!alloc_) return;
    // Drain DMA before touching pin buffers (see shutdown()).
    shutdown();
    copyq_.reset();
    jobsq_.reset();
    for (void* p : owned_) if (p) alloc_->free(p);
    for (void* p : owned_host_) if (p) sycl::free(p, alloc_->queue());
    owned_host_.clear();
    for (auto*& p : pin_ring_)
        if (p) { sycl::free(p, alloc_->queue()); p = nullptr; }
    for (auto*& p : pp_pin_)
        if (p) { sycl::free(p, alloc_->queue()); p = nullptr; }
    for (auto*& p : pf_pin_)
        if (p) { sycl::free(p, alloc_->queue()); p = nullptr; }
    for (auto*& p : ep_pin_)
        if (p) { sycl::free(p, alloc_->queue()); p = nullptr; }
    if (pf_pred_pin_) { sycl::free(pf_pred_pin_, alloc_->queue()); pf_pred_pin_ = nullptr; }
    if (ep_xstage_) { sycl::free(ep_xstage_, alloc_->queue()); ep_xstage_ = nullptr; }
    if (ep_rstage_) { sycl::free(ep_rstage_, alloc_->queue()); ep_rstage_ = nullptr; }
    owned_.clear();
    layers_.clear();
    alloc_ = nullptr;
}

std::string Glm5NextModel::load(DeviceAllocator& alloc, const GgufReader& g,
                                const Glm5NextConfig& cfg, uint64_t vram_budget_bytes,
                                uint32_t layer_lo, uint32_t layer_hi) {
    return load_impl(&alloc,g,cfg,vram_budget_bytes,layer_lo,layer_hi,nullptr);
}

std::string Glm5NextModel::plan_device_weights(const GgufReader& g, const Glm5NextConfig& cfg,
                                              uint32_t lo, uint32_t hi, uint64_t& bytes) {
    bytes = 0;
    Glm5NextModel scratch;
    return scratch.load_impl(nullptr,g,cfg,0,lo,hi,&bytes);
}

std::string Glm5NextModel::load_impl(DeviceAllocator* alloc, const GgufReader& g,
                                    const Glm5NextConfig& cfg, uint64_t vram_budget_bytes,
                                    uint32_t layer_lo, uint32_t layer_hi, uint64_t* projected_bytes) {
    if (alloc) dev_alloc_tag("weights");
    alloc_ = alloc;
    cfg_   = cfg;
    layers_.assign(cfg.n_layers, {});
    // The uploaded range never includes the MTP block: it only serves
    // speculative decode (P3) and its tensors are bound as host views below.
    const uint32_t n_tf = cfg.n_transformer_layers();
    layer_lo_ = std::min(layer_lo, n_tf);
    layer_hi_ = std::min(layer_hi, n_tf);
    if (layer_lo_ >= layer_hi_) return "glm5next load: empty layer range";
    if (vram_budget_bytes == 0) vram_budget_bytes = 28ull << 30;

    char buf[96];
    auto Tl = [&](uint32_t L, const char* n) -> const GgufTensorInfo* {
        std::snprintf(buf, sizeof(buf), "blk.%u.%s", L, n);
        return g.find_tensor(buf);
    };
    std::string err;

    // ---- pass 1: bind every tensor + project device residency BEFORE any
    // upload, so the VRAM guard fires with zero bytes allocated. -------------
    // Layout classes (see the header comment on Glm5NextLayer):
    //  kNative   — GGUF element order as-is (token_embd, k_b/v_b, 1-D, conv)
    //  kTransKN  — transposed to [K, N] row-major for gemv_fp16/gemm_fp16
    enum class Lay { kNative, kTransKN, kQ8Soa };
    struct Job {
        const GgufTensorInfo* t;
        sycl::half** dst_h;    // exactly one of dst_h / dst_f / dst_dw is set
        float**      dst_f;
        Glm5DW*      dst_dw = nullptr;
        Lay          lay = Lay::kNative;
    };
    std::vector<Job> jobs;
    uint64_t projected = 0;

    auto want = [&](const GgufTensorInfo* t, const char* name,
                    sycl::half** h, float** f, Lay lay = Lay::kNative) -> bool {
        if (!t) { err = std::string("glm5next load: missing tensor ") + name; return false; }
        uint64_t n = 1;
        for (uint32_t d = 0; d < t->n_dims; ++d) n *= t->shape[d];
        const uint64_t bytes = n * (h ? sizeof(sycl::half) : sizeof(float));
        projected += bytes;
        placement_.push_back({std::string(t->name), "device", h ? "F16" : "F32", bytes});
        jobs.push_back({t, h, f, nullptr, lay});
        return true;
    };
    // Dense 2-D projection: Q8_0-SoA when the file stores Q8_0 (halves the
    // resident reads — W8A16, quality-neutral; IE_G5_DENSE_Q8=0 opts out),
    // F16 [K,N]-transposed otherwise.
    const bool q8_dense = [] {
        const char* v = std::getenv("IE_G5_DENSE_Q8");
        return !(v && v[0] == '0');
    }();
    auto want_dw = [&](const GgufTensorInfo* t, const char* name,
                       Glm5DW* dw) -> bool {
        if (!t) { err = std::string("glm5next load: missing tensor ") + name; return false; }
        if (q8_dense && t->dtype == DType::kQ8_0 && t->n_dims == 2) {
            const uint64_t K = t->shape[0], N = t->shape[1];
            const uint64_t bytes = N * K + N * (K / 32) * 2;
            projected += bytes;
            placement_.push_back({std::string(t->name), "device", "Q8soa", bytes});
            jobs.push_back({t, nullptr, nullptr, dw, Lay::kQ8Soa});
            return true;
        }
        return want(t, name, &dw->w, nullptr, Lay::kTransKN);
    };
    auto host_bank = [&](uint32_t L, const char* n,
                         const GgufTensorInfo** dst) -> bool {
        const GgufTensorInfo* t = Tl(L, n);
        if (!t) { err = std::string("glm5next load: missing expert bank blk.") +
                        std::to_string(L) + "." + n; return false; }
        *dst = t;
        host_bytes_ += t->nbytes;
        placement_.push_back({std::string(t->name), "host", type_name(t->dtype).data(), t->nbytes});
        return true;
    };
    // Globals by role. The tail stage also loads token_embd, but only when
    // MTP spec decode is requested (IE_G5_MTP=1, set by the runner for
    // --spec): the fusion embeds the drafted token there. Default skips the
    // whole MTP kit — 1.18 GiB embd + blk.45 decoder + its ecache slot cut.
    const bool mtp_want = [] {
        const char* v = std::getenv("IE_G5_MTP");
        return v && v[0] == '1';
    }();
    // Which stage hosts the kit: the tail (default) or, with
    // IE_G5_MTP_STAGE=head, the first stage — the pipelined-draft caller then
    // drafts on card 0 while card 1 runs the real token (G5' lever 3). The
    // host stage needs token_embd (the fusion embeds the drafted token) and
    // lm_head + shared_head_norm (the draft's logits); it costs that card the
    // blk.45 decoder, the head, and the MTP block's share of the expert cache.
    static const bool mtp_head = [] {
        const char* v = std::getenv("IE_G5_MTP_STAGE");
        return v && (v[0] == 'h' || v[0] == 'H' || v[0] == '0');
    }();
    mtp_here_ = mtp_want && (mtp_head ? layer_lo_ == 0 : layer_hi_ == n_tf);
    if (layer_lo_ == 0 || mtp_here_) {
        if (!want(g.find_tensor("token_embd.weight"), "token_embd.weight",
                  &token_embd, nullptr)) return err;
    }
    if (layer_hi_ == n_tf || mtp_here_) {
        if (!want_dw(g.find_tensor("output.weight"), "output.weight", &lm_head)) return err;
        if (!want(g.find_tensor("output_norm.weight"), "output_norm.weight",
                  nullptr, &output_norm)) return err;
    }

    for (uint32_t L = layer_lo_; L < layer_hi_; ++L) {
        Glm5NextLayer& w = layers_[L];
        // Hyper-connections + norms — every transformer block.
        if (!want(Tl(L, "hc_attn_fn.weight"),    "hc_attn_fn",    nullptr, &w.hc_attn_fn)) return err;
        if (!want(Tl(L, "hc_attn_base.weight"),  "hc_attn_base",  nullptr, &w.hc_attn_base)) return err;
        if (!want(Tl(L, "hc_attn_scale.weight"), "hc_attn_scale", nullptr, &w.hc_attn_scale)) return err;
        if (!want(Tl(L, "hc_ffn_fn.weight"),     "hc_ffn_fn",     nullptr, &w.hc_ffn_fn)) return err;
        if (!want(Tl(L, "hc_ffn_base.weight"),   "hc_ffn_base",   nullptr, &w.hc_ffn_base)) return err;
        if (!want(Tl(L, "hc_ffn_scale.weight"),  "hc_ffn_scale",  nullptr, &w.hc_ffn_scale)) return err;
        if (!want(Tl(L, "attn_norm.weight"), "attn_norm", nullptr, &w.attn_norm)) return err;
        if (!want(Tl(L, "ffn_norm.weight"),  "ffn_norm",  nullptr, &w.ffn_norm)) return err;

        if (cfg.is_full_attn(L)) {
            // MLA lora stack + absorbed heads.
            if (!want_dw(Tl(L, "attn_q_a.weight"), "attn_q_a", &w.q_a)) return err;
            if (!want(Tl(L, "attn_q_a_norm.weight"), "attn_q_a_norm", nullptr, &w.q_a_norm)) return err;
            if (!want_dw(Tl(L, "attn_q_b.weight"), "attn_q_b", &w.q_b)) return err;
            if (!want_dw(Tl(L, "attn_kv_a_mqa.weight"), "attn_kv_a_mqa", &w.kv_a)) return err;
            if (!want(Tl(L, "attn_kv_a_norm.weight"), "attn_kv_a_norm", nullptr, &w.kv_a_norm)) return err;
            if (!want(Tl(L, "attn_k_b.weight"), "attn_k_b", &w.k_b, nullptr)) return err;
            if (!want(Tl(L, "attn_v_b.weight"), "attn_v_b", &w.v_b, nullptr)) return err;
            if (!want_dw(Tl(L, "attn_output.weight"), "attn_output", &w.attn_out)) return err;
            // Lightning indexer + gated compressor.
            if (!want(Tl(L, "indexer.attn_k.weight"),   "indexer.attn_k",   &w.idx_k, nullptr, Lay::kTransKN)) return err;
            if (!want(Tl(L, "indexer.k_norm.weight"),   "indexer.k_norm",   nullptr, &w.idx_k_norm_w)) return err;
            if (!want(Tl(L, "indexer.k_norm.bias"),     "indexer.k_norm.bias", nullptr, &w.idx_k_norm_b)) return err;
            if (!want(Tl(L, "indexer.attn_q_b.weight"), "indexer.attn_q_b", &w.idx_q_b, nullptr, Lay::kTransKN)) return err;
            if (!want(Tl(L, "indexer.proj.weight"),     "indexer.proj",     nullptr, &w.idx_proj)) return err;
            if (!want(Tl(L, "indexer_compressor_ape.weight"),  "indexer_compressor_ape",  nullptr, &w.idx_ape)) return err;
            if (!want(Tl(L, "indexer_compressor_gate.weight"), "indexer_compressor_gate", &w.idx_gate, nullptr, Lay::kTransKN)) return err;
        } else {
            // KDA: separate q/k/v + per-stream conv, low-rank gates, scan params.
            if (!want_dw(Tl(L, "attn_q.weight"), "attn_q(kda)", &w.kda_q)) return err;
            if (!want_dw(Tl(L, "attn_k.weight"), "attn_k(kda)", &w.kda_k)) return err;
            if (!want_dw(Tl(L, "attn_v.weight"), "attn_v(kda)", &w.kda_v)) return err;
            if (!want_dw(Tl(L, "attn_output.weight"), "attn_output(kda)", &w.kda_o)) return err;
            if (!want(Tl(L, "ssm_conv1d_q.weight"), "ssm_conv1d_q", &w.conv_q, nullptr)) return err;
            if (!want(Tl(L, "ssm_conv1d_k.weight"), "ssm_conv1d_k", &w.conv_k, nullptr)) return err;
            if (!want(Tl(L, "ssm_conv1d_v.weight"), "ssm_conv1d_v", &w.conv_v, nullptr)) return err;
            if (!want(Tl(L, "ssm_a"),       "ssm_a",       nullptr, &w.ssm_a)) return err;
            if (!want(Tl(L, "ssm_dt.bias"), "ssm_dt.bias", nullptr, &w.dt_bias)) return err;
            if (!want_dw(Tl(L, "ssm_f_a.weight"), "ssm_f_a", &w.f_a)) return err;
            if (!want_dw(Tl(L, "ssm_f_b.weight"), "ssm_f_b", &w.f_b)) return err;
            if (!want_dw(Tl(L, "ssm_g_a.weight"), "ssm_g_a", &w.g_a)) return err;
            if (!want_dw(Tl(L, "ssm_g_b.weight"), "ssm_g_b", &w.g_b)) return err;
            if (!want_dw(Tl(L, "ssm_beta.weight"), "ssm_beta", &w.beta)) return err;
            if (!want(Tl(L, "ssm_norm.weight"), "ssm_norm", &w.ssm_norm, nullptr)) return err;
        }

        if (cfg.is_dense_layer(L)) {
            if (!want_dw(Tl(L, "ffn_gate.weight"), "ffn_gate", &w.ffn_gate)) return err;
            if (!want_dw(Tl(L, "ffn_up.weight"), "ffn_up", &w.ffn_up)) return err;
            if (!want_dw(Tl(L, "ffn_down.weight"), "ffn_down", &w.ffn_down)) return err;
        } else {
            if (!want(Tl(L, "ffn_gate_inp.weight"), "ffn_gate_inp", nullptr, &w.router)) return err;
            if (!want(Tl(L, "exp_probs_b.bias"),    "exp_probs_b.bias", nullptr, &w.probs_bias)) return err;
            if (!want_dw(Tl(L, "ffn_gate_shexp.weight"), "ffn_gate_shexp", &w.shexp_gate)) return err;
            if (!want_dw(Tl(L, "ffn_up_shexp.weight"), "ffn_up_shexp", &w.shexp_up)) return err;
            if (!want_dw(Tl(L, "ffn_down_shexp.weight"), "ffn_down_shexp", &w.shexp_down)) return err;
            if (!host_bank(L, "ffn_gate_exps.weight", &w.gate_exps)) return err;
            if (!host_bank(L, "ffn_up_exps.weight",   &w.up_exps))   return err;
            if (!host_bank(L, "ffn_down_exps.weight", &w.down_exps)) return err;
        }
    }

    // MTP block: the tail stage uploads blk.45's full decoder set (MLA +
    // router + shexp + norms; NO hyper-connections on this block, indexer
    // skipped like the main v0 path) plus the nextn fusion tensors — the
    // spec-decode draft head. Its expert bank streams through the ecache
    // like any other MoE layer.
    if (mtp_here_ && cfg.nextn_predict_layers == 1) {
        const uint32_t M = n_tf;   // blk.45 on the real file
        Glm5NextLayer& w = layers_[M];
        if (!want(Tl(M, "attn_norm.weight"), "attn_norm(45)", nullptr, &w.attn_norm)) return err;
        if (!want(Tl(M, "ffn_norm.weight"),  "ffn_norm(45)",  nullptr, &w.ffn_norm)) return err;
        if (!want_dw(Tl(M, "attn_q_a.weight"), "attn_q_a(45)", &w.q_a)) return err;
        if (!want(Tl(M, "attn_q_a_norm.weight"), "attn_q_a_norm(45)", nullptr, &w.q_a_norm)) return err;
        if (!want_dw(Tl(M, "attn_q_b.weight"), "attn_q_b(45)", &w.q_b)) return err;
        if (!want_dw(Tl(M, "attn_kv_a_mqa.weight"), "attn_kv_a_mqa(45)", &w.kv_a)) return err;
        if (!want(Tl(M, "attn_kv_a_norm.weight"), "attn_kv_a_norm(45)", nullptr, &w.kv_a_norm)) return err;
        if (!want(Tl(M, "attn_k_b.weight"), "attn_k_b(45)", &w.k_b, nullptr)) return err;
        if (!want(Tl(M, "attn_v_b.weight"), "attn_v_b(45)", &w.v_b, nullptr)) return err;
        if (!want_dw(Tl(M, "attn_output.weight"), "attn_output(45)", &w.attn_out)) return err;
        if (!want(Tl(M, "ffn_gate_inp.weight"), "ffn_gate_inp(45)", nullptr, &w.router)) return err;
        if (!want(Tl(M, "exp_probs_b.bias"),    "exp_probs_b(45)",  nullptr, &w.probs_bias)) return err;
        if (!want_dw(Tl(M, "ffn_gate_shexp.weight"), "ffn_gate_shexp(45)", &w.shexp_gate)) return err;
        if (!want_dw(Tl(M, "ffn_up_shexp.weight"), "ffn_up_shexp(45)", &w.shexp_up)) return err;
        if (!want_dw(Tl(M, "ffn_down_shexp.weight"), "ffn_down_shexp(45)", &w.shexp_down)) return err;
        if (!want_dw(Tl(M, "nextn.eh_proj.weight"), "nextn.eh_proj", &mtp_eh_)) return err;
        if (!want(Tl(M, "nextn.enorm.weight"),   "nextn.enorm",   nullptr, &mtp_enorm_)) return err;
        if (!want(Tl(M, "nextn.hnorm.weight"),   "nextn.hnorm",   nullptr, &mtp_hnorm_)) return err;
        if (!want(Tl(M, "nextn.shared_head_norm.weight"), "nextn.shn", nullptr, &mtp_shn_)) return err;
        if (!host_bank(M, "ffn_gate_exps.weight", &w.gate_exps)) return err;
        if (!host_bank(M, "ffn_up_exps.weight",   &w.up_exps))   return err;
        if (!host_bank(M, "ffn_down_exps.weight", &w.down_exps)) return err;
        // keep the plain views bound too (audit tool checks them)
        w.nextn_eh    = Tl(M, "nextn.eh_proj.weight");
        w.nextn_enorm = Tl(M, "nextn.enorm.weight");
        w.nextn_hnorm = Tl(M, "nextn.hnorm.weight");
        w.nextn_shn   = Tl(M, "nextn.shared_head_norm.weight");
    }

    // Planning and loading use these same destination types and tensor roles.
    // Exit before touching a GPU queue or any mapped weight payload.
    if (projected_bytes) { *projected_bytes = projected; return {}; }

    // ---- VRAM guard: projected weights only (KV/workspaces come later and
    // small); refuse before the first byte moves. ----------------------------
    if (projected > vram_budget_bytes && std::getenv("IE_ALLOW_OOM") == nullptr) {
        std::snprintf(buf, sizeof(buf), "%.2f GiB > budget %.2f GiB",
                      projected / 1073741824.0, vram_budget_bytes / 1073741824.0);
        return std::string("glm5next load: projected device weights ") + buf +
               " (IE_ALLOW_OOM=1 to attempt anyway)";
    }

    // ---- pass 2: dequant + upload. -----------------------------------------
    std::vector<float>      f32;
    std::vector<sycl::half> h16;
    sycl::queue& q = alloc->queue();
    for (const Job& j : jobs) {
        if (j.lay == Lay::kQ8Soa) {
            // Byte-faithful Q8_0 -> SoA repack (the qwen35/qwen4exp layout:
            // qs[n*K + b*32 + i], d[n*bpc + b]).
            const uint32_t K = uint32_t(j.t->shape[0]), N = uint32_t(j.t->shape[1]);
            const uint32_t bpc = K / 32;
            const auto* blocks = reinterpret_cast<const block_q8_0*>(j.t->data);
            std::vector<int8_t>   qs(uint64_t(N) * K);
            std::vector<uint16_t> dd(uint64_t(N) * bpc);
            for (uint64_t nn = 0; nn < N; ++nn)
                for (uint32_t b = 0; b < bpc; ++b) {
                    const block_q8_0& blk = blocks[nn * bpc + b];
                    std::memcpy(&dd[nn * bpc + b], &blk.d, 2);
                    std::memcpy(&qs[nn * K + uint64_t(b) * 32], blk.qs, 32);
                }
            auto* dqs = static_cast<int8_t*>(alloc->malloc(qs.size()));
            auto* ddd = static_cast<uint16_t*>(alloc->malloc(dd.size() * 2));
            if (!dqs || !ddd) return "glm5next load: q8 soa alloc failed";
            q.memcpy(dqs, qs.data(), qs.size()).wait();
            q.memcpy(ddd, dd.data(), dd.size() * 2).wait();
            owned_.push_back(dqs); owned_.push_back(ddd);
            j.dst_dw->qs = dqs; j.dst_dw->d = ddd;
            dev_bytes_ += qs.size() + dd.size() * 2;
            continue;
        }
        if (auto e = dequant_tensor_fp32(j.t, f32); !e.empty()) return e;
        const uint64_t n = f32.size();
        void* d = nullptr;
        if (j.dst_f) {                       // F32 tensor, GGUF-native order
            d = alloc->malloc(n * sizeof(float));
            if (!d) return "glm5next load: device malloc failed (f32)";
            q.memcpy(d, f32.data(), n * sizeof(float)).wait();
            *j.dst_f = static_cast<float*>(d);
            dev_bytes_ += n * sizeof(float);
        } else if (j.lay == Lay::kTransKN && j.t->n_dims == 2) {
            // gemv_fp16 layout: W[k*N+n] from GGUF's W[n*K+k].
            const uint64_t K = j.t->shape[0], N = j.t->shape[1];
            h16.resize(n);
            for (uint64_t nn = 0; nn < N; ++nn)
                for (uint64_t k = 0; k < K; ++k)
                    h16[k * N + nn] = sycl::half(f32[nn * K + k]);
            d = alloc->malloc(n * sizeof(sycl::half));
            if (!d) return "glm5next load: device malloc failed (f16)";
            q.memcpy(d, h16.data(), n * sizeof(sycl::half)).wait();
            *j.dst_h = static_cast<sycl::half*>(d);
            dev_bytes_ += n * sizeof(sycl::half);
        } else {                             // kNative F16 (embd, k_b/v_b, conv, 1-D)
            h16.resize(n);
            for (uint64_t i = 0; i < n; ++i) h16[i] = sycl::half(f32[i]);
            d = alloc->malloc(n * sizeof(sycl::half));
            if (!d) return "glm5next load: device malloc failed (f16)";
            q.memcpy(d, h16.data(), n * sizeof(sycl::half)).wait();
            *j.dst_h = static_cast<sycl::half*>(d);
            dev_bytes_ += n * sizeof(sycl::half);
        }
        owned_.push_back(d);
    }

    // ---- pass 3: bank fill sources — pin the owned banks into host USM ----
    // (IE_G5_PIN_BANKS=0 opts out). Copies the mmap bytes into pinned
    // allocations layer by layer, releasing the page cache behind itself
    // (madvise DONTNEED) so RAM holds ONE copy, not two. A miss fill then
    // needs no CPU bounce at all. Stops pinning (gracefully, per layer) if
    // MemAvailable would drop under the 40 GiB floor — the 2026-08-27
    // pinned-livelock lesson.
    {
        for (uint32_t L = layer_lo_; L < cfg.n_layers; ++L) {
            Glm5NextLayer& w2 = layers_[L];
            if (!w2.gate_exps) continue;
            w2.gate_src = static_cast<const uint8_t*>(w2.gate_exps->data);
            w2.up_src   = static_cast<const uint8_t*>(w2.up_exps->data);
            w2.down_src = static_cast<const uint8_t*>(w2.down_exps->data);
            g.locate(w2.gate_src, w2.gate_fd, w2.gate_foff);
            g.locate(w2.up_src,   w2.up_fd,   w2.up_foff);
            g.locate(w2.down_src, w2.down_fd, w2.down_foff);
        }
        // Founder 2026-08-29: banks stay on mmap (page cache — reclaimable,
        // "used" RAM stays ~20GB) by default. IE_G5_PIN_BANKS=1 opts back in
        // to pinned host USM (26 vs ~8-12 GB/s fills, +160GB used RAM).
        const char* pv = std::getenv("IE_G5_PIN_BANKS");
        if (server_memory_ ? memory_policy_.pin_banks : (pv && pv[0] == '1')) {
            cpool_.start(3);
            auto avail_gib = [] {
                FILE* f = std::fopen("/proc/meminfo", "r");
                if (!f) return 0.0;
                char k[64]; uint64_t v; char u[16];
                double out = 0;
                while (std::fscanf(f, "%63s %lu %15s\n", k, &v, u) == 3)
                    if (std::strcmp(k, "MemAvailable:") == 0) { out = v / 1048576.0; break; }
                std::fclose(f);
                return out;
            };
            const long psz = sysconf(_SC_PAGESIZE);
            // floor tunable (IE_G5_PIN_FLOOR_GIB, default 40): the last stage-B
            // layers land on mmap+bounce otherwise — ~1.5ms/miss slower at decode
            double floor_gib = 40.0;
            if (const char* fv = std::getenv("IE_G5_PIN_FLOOR_GIB"))
                floor_gib = std::max(8.0, std::atof(fv));
            // Per-stage pinned-bytes cap (IE_G5_PIN_MAX_GIB): bounds TOTAL used
            // RAM regardless of MemAvailable — the founder's 200+GB "big no"
            // (2026-08-29). Every stage gets its own cap, so EP always has
            // pinned layers on BOTH cards.
            double cap_gib = 1e9;
            if (const char* cv = std::getenv("IE_G5_PIN_MAX_GIB"))
                cap_gib = std::max(4.0, std::atof(cv));
            if (server_memory_) {
                floor_gib = memory_policy_.floor_gib;
                cap_gib = server_pin_cap_ / 1073741824.0;
            }
            uint32_t n_pinned = 0, n_skipped = 0;
            // IE_G5_NO_PIN_MTP=1: leave the MTP block's bank on mmap. Tried as the
            // default 2026-09-04 (n11) to hand the tail stage more pinned
            // layers: it gained ONE (17 vs 16), chunk-1 prefill did not move
            // (32.8 vs 33.6), and short-context pipedraft decode fell 16%
            // (13.20 vs 15.66) as the draft's experts faulted from a page cache
            // the other layers' pinning evicts. Pinned by default; the switch
            // stays for the day the RAM arithmetic changes.
            static const bool pin_mtp = std::getenv("IE_G5_NO_PIN_MTP") == nullptr;
            const uint32_t n_tf_pin = cfg.n_transformer_layers();
            for (uint32_t L = layer_lo_; L < cfg.n_layers; ++L) {
                Glm5NextLayer& w2 = layers_[L];
                if (!w2.gate_exps) continue;
                if (L == n_tf_pin && !pin_mtp) continue;
                const GgufTensorInfo* ts[3] = {w2.gate_exps, w2.up_exps, w2.down_exps};
                const uint8_t** dsts[3] = {&w2.gate_src, &w2.up_src, &w2.down_src};
                const double need_gib =
                    (ts[0]->nbytes + ts[1]->nbytes + ts[2]->nbytes) / 1073741824.0;
                // The floor check reads MemAvailable, which COUNTS this layer's own
                // page-cache pages (reclaimable) — the madvise below frees them.
                // Charge the layer at its REAL cost: pinned host USM costs 1.00x
                // its size (measured 2026-09-11: 8 / 16 / 100 GiB of malloc_host
                // -> MemAvailable drop 1.001-1.007x). The 1.37x charged here from
                // 2026-09-03 ("160 GiB pinned consumed ~219") was the multi-device
                // context mirroring VRAM into host RAM, fixed in the allocator
                // (83554c2), never the pinning; left at 1.37 it skipped stage B's
                // last layer whenever the head stage also pinned the MTP kit.
                // IE_G5_PIN_OVERHEAD (>= 1.0) still overrides; the floor is
                // untouched.
                static const double kPinOverhead = [] {
                    const char* v = std::getenv("IE_G5_PIN_OVERHEAD");
                    return v ? std::max(1.0, std::atof(v)) : 1.0;
                }();
                const double pin_overhead = server_memory_ ? memory_policy_.pin_overhead : kPinOverhead;
                // The floor is checked BEFORE each pin; the run's own transients
                // (workspaces, staging, page cache of the mmap'd dense weights,
                // ~1 GiB measured 2026-09-11 with every bank pinned) land after
                // the last one and would leave the low-water mark below the
                // floor. Reserve them here so "floor" bounds the run, not just
                // the pin decision. Raises the effective floor only.
                constexpr double kPinTransientGib = 2.0;
                if (avail_gib() < need_gib * pin_overhead + floor_gib + kPinTransientGib) { ++n_skipped; continue; }
                // hard per-stage cap (Codex finding: parsed but unenforced)
                if (double(pinned_bytes_) / 1073741824.0 + need_gib > cap_gib) { ++n_skipped; continue; }
                (void)dsts;
                const uint64_t E2 = cfg.n_experts;
                const uint64_t gsl2 = ts[0]->nbytes / E2, usl2 = ts[1]->nbytes / E2,
                               dsl2 = ts[2]->nbytes / E2;
                const uint64_t stride = gsl2 + usl2 + dsl2;
                uint8_t* p = nullptr;
                try { p = static_cast<uint8_t*>(sycl::malloc_host(stride * E2, alloc->queue())); }
                catch (const sycl::exception& ex) {
                    std::fprintf(stderr, "[glm5next] layer %u host pin allocation failed: %s; left on mmap\n", L, ex.what());
                }
                if (!p) { ++n_skipped; continue; }
                // per-expert CONTIGUOUS repack = the ecache slot layout
                for (uint64_t e2 = 0; e2 < E2; ++e2) {
                    uint8_t* dst2 = p + e2 * stride;
                    cpool_.copy(dst2,               static_cast<const uint8_t*>(ts[0]->data) + e2 * gsl2, gsl2);
                    cpool_.copy(dst2 + gsl2,        static_cast<const uint8_t*>(ts[1]->data) + e2 * usl2, usl2);
                    cpool_.copy(dst2 + gsl2 + usl2, static_cast<const uint8_t*>(ts[2]->data) + e2 * dsl2, dsl2);
                }
                for (int i = 0; i < 3; ++i) {   // release the file-cache copy
                    uintptr_t b = reinterpret_cast<uintptr_t>(ts[i]->data);
                    uintptr_t lo = (b + psz - 1) & ~uintptr_t(psz - 1);
                    uintptr_t hi = (b + ts[i]->nbytes) & ~uintptr_t(psz - 1);
                    if (hi > lo)
                        madvise(reinterpret_cast<void*>(lo), hi - lo, MADV_DONTNEED);
                }
                owned_host_.push_back(p);
                pinned_bytes_ += stride * E2;
                w2.bank_pin = p;
                w2.bank_pinned = true;
                ++n_pinned;
                std::fprintf(stderr, "[glm5next] layer %u pinned; stage total %.2f GiB\n",
                             L, pinned_bytes_ / 1073741824.0);
            }
            std::fprintf(stderr,
                         "[glm5next] pinned banks: %u layers (%.1f GiB), %u left on mmap\n",
                         n_pinned, pinned_bytes_ / 1073741824.0, n_skipped);
        }
    }
    return {};
}

// ===========================================================================
// Runtime (v0, single GPU — correctness before speed)
// ===========================================================================

namespace {

// hc_mult exact copies of the embedding row into the wide fp32 streams —
// "no scaling, no one-hot into stream 0" (PR graph, hc_init).
sycl::event k_embed_streams(sycl::queue& q, const int32_t* toks,
                            const sycl::half* embd, float* streams,
                            uint32_t T, uint32_t H, uint32_t hc,
                            const std::vector<sycl::event>& deps) {
    return ie::ps(q, "g5_embed", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<2>(T, H), [=](sycl::id<2> id) {
            const uint32_t t = uint32_t(id[0]), d = uint32_t(id[1]);
            const float v = float(embd[uint64_t(toks[t]) * H + d]);
            for (uint32_t s = 0; s < hc; ++s)
                streams[(uint64_t(t) * hc + s) * H + d] = v;
        });
    });
}

sycl::event k_sigmoid(sycl::queue& q, float* x, uint64_t n,
                      const std::vector<sycl::event>& deps) {
    return ie::ps(q, "g5_sigmoid", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            x[i] = 1.f / (1.f + sycl::native::exp(-x[i]));
        });
    });
}

// Router logits, fp32 end to end (fragile chain): rl[t, e] = x[t]·W[e].
// W is the GGUF-native fp32 ffn_gate_inp — E rows of H.
// T=1 SLM-shared-x (2–8 experts/WG) measured 2.7× slower than this 288-WG
// leaf: x is L2-resident; occupancy wins. Keep one SG per expert.
sycl::event k_router_logits(sycl::queue& q, const float* x, const float* w,
                            float* rl, uint32_t T, uint32_t H, uint32_t E,
                            const std::vector<sycl::event>& deps) {
    constexpr int SG = 32;
    return ie::ps(q, "g5_router", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({T, uint64_t(E) * SG}, {1, SG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t t = uint32_t(it.get_group(0));
            const uint32_t e = uint32_t(it.get_group(1));
            const uint32_t l = uint32_t(it.get_local_id(1));
            auto sg = it.get_sub_group();
            const float* xr = x + uint64_t(t) * H;
            const float* wr = w + uint64_t(e) * H;
            float acc = 0.f;
            // Load ahead to expose independent memory operations. Keep a single
            // accumulator and the original lane/element order: routing is FP32 exact.
            constexpr uint32_t AHEAD = 16;
            uint32_t k = l;
            for (; k < H && H - k > (AHEAD - 1) * SG; k += AHEAD * SG) {
                float xv[AHEAD], wv[AHEAD];
                #pragma unroll
                for (uint32_t j = 0; j < AHEAD; ++j) {
                    xv[j] = xr[k + j * SG];
                    wv[j] = wr[k + j * SG];
                }
                #pragma unroll
                for (uint32_t j = 0; j < AHEAD; ++j)
                    acc = sycl::fma(xv[j], wv[j], acc);
            }
            for (; k < H; k += SG) acc = sycl::fma(xr[k], wr[k], acc);
            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (l == 0) rl[uint64_t(t) * E + e] = acc;
        });
    });
}

// Sigmoid + selection-bias top-k (glm52_run's k_router_topk, at 288 experts).
// Selection is by sigmoid(logit) + exp_probs_b; the returned weights are the
// UNBIASED sigmoids, normalised over the top-k (weights_norm) and scaled
// (weights_scale 2.5). Ties break to the lower expert id.
// Preserve the ordinary FP32 normalization order. When every selected sigmoid
// underflows to zero, normalize relative exponentials instead of producing 0/0.
static inline void router_normalize(const float* logits, const int32_t* ids,
                                    float* weights, uint32_t count,
                                    float sum, float scale) {
    if (sum == 0.f) {
        float largest = -INFINITY;
        for (uint32_t k = 0; k < count; ++k)
            largest = sycl::fmax(largest, logits[ids[k]]);
        for (uint32_t k = 0; k < count; ++k) {
            weights[k] = sycl::exp(logits[ids[k]] - largest);
            sum += weights[k];
        }
    }
    for (uint32_t k = 0; k < count; ++k)
        weights[k] = weights[k] / sum * scale;
}

// The fixed per-lane count exposes the score scan to the compiler. Cache the
// unbiased sigmoid separately: subtracting the bias from the score is not exact.
template <uint32_t PER>
static sycl::event k_router_topk_small(sycl::queue& q, const float* rl,
                          const float* bias, int32_t* top, float* topw,
                          uint32_t NE, uint32_t TOPK, float wscale, uint32_t T,
                          const std::vector<sycl::event>& deps) {
    constexpr uint32_t SG = 32;
    const uint32_t per = PER ? PER : (NE + SG - 1) / SG;
    return ie::ps(q, "g5_topk", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(T) * SG, SG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t t = uint32_t(it.get_group(0));
            auto sg = it.get_sub_group();
            const uint32_t lane = uint32_t(sg.get_local_id()[0]);
            const float* rlt = rl + uint64_t(t) * NE;
            float score[PER ? PER : 16], probability[PER ? PER : 16];
            #pragma unroll (PER ? PER : 1)
            for (uint32_t i = 0; i < per; ++i) {
                const uint32_t e = lane * per + i;
                const bool valid = PER ? true : e < NE;
                const float p = valid ? 1.f / (1.f + sycl::exp(-rlt[e])) : 0.f;
                probability[i] = p;
                score[i] = valid ? p + bias[e] : -INFINITY;
            }
            float sum = 0.f;
            for (uint32_t k = 0; k < TOPK; ++k) {
                float best = -INFINITY;
                uint32_t index = 0;
                #pragma unroll (PER ? PER : 1)
                for (uint32_t i = 0; i < per; ++i)
                    if (score[i] > best) { best = score[i]; index = i; }
                const float maximum = sycl::reduce_over_group(sg, best, sycl::maximum<float>());
                const uint32_t candidate = best == maximum
                    ? sycl::min(lane * per + index, NE) : NE;
                const uint32_t winner = sycl::reduce_over_group(sg, candidate,
                                                               sycl::minimum<uint32_t>());
                const uint32_t owner = winner / per;
                float selected = 0.f;
                if (owner == lane) {
                    const uint32_t i = winner - lane * per;
                    selected = probability[i];
                    score[i] = -INFINITY;
                }
                const float weight = sycl::select_from_group(sg, selected, owner);
                sum += weight;
                if (lane == 0) {
                    top[uint64_t(t) * TOPK + k] = int32_t(winner);
                    topw[uint64_t(t) * TOPK + k] = weight;
                }
            }
            if (lane == 0)
                router_normalize(rlt, top + uint64_t(t) * TOPK,
                                 topw + uint64_t(t) * TOPK, TOPK, sum, wscale);
        });
    });
}

// Rare larger expert counts must not overrun a fixed private score array.
// One work item owns a row and its selection history, so this fallback needs
// no scratch allocation or cross-workgroup synchronization.
static sycl::event k_router_topk_large(sycl::queue& q, const float* rl,
                          const float* bias, int32_t* top, float* topw,
                          uint32_t NE, uint32_t TOPK, float wscale, uint32_t T,
                          const std::vector<sycl::event>& deps) {
    return ie::ps(q, "g5_topk_large", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(T), [=](sycl::id<1> id) {
            const uint64_t t = id[0];
            const float* rlt = rl + t * NE;
            int32_t* ids = top + t * TOPK;
            float* weights = topw + t * TOPK;
            float sum = 0.f;
            for (uint32_t k = 0; k < TOPK; ++k) {
                float best = -INFINITY;
                uint32_t winner = NE;
                for (uint32_t e = 0; e < NE; ++e) {
                    bool used = false;
                    for (uint32_t j = 0; j < k; ++j) used |= ids[j] == int32_t(e);
                    if (used) continue;
                    float score = 1.f / (1.f + sycl::exp(-rlt[e])) + bias[e];
                    if (sycl::isnan(score)) score = -INFINITY;
                    if (score > best || (score == best && e < winner)) {
                        best = score; winner = e;
                    }
                }
                const float weight = 1.f / (1.f + sycl::exp(-rlt[winner]));
                ids[k] = int32_t(winner);
                weights[k] = weight;
                sum += weight;
            }
            router_normalize(rlt, ids, weights, TOPK, sum, wscale);
        });
    });
}

sycl::event k_router_topk(sycl::queue& q, const float* rl, const float* bias,
                          int32_t* top, float* topw, uint32_t NE, uint32_t TOPK,
                          float wscale, uint32_t T,
                          const std::vector<sycl::event>& deps) {
    if (NE == 0 || NE > uint32_t(INT32_MAX) || TOPK == 0 || TOPK > NE)
        throw std::invalid_argument("router requires 0 < top-k <= expert count");
    if (T == 0) return q.ext_oneapi_submit_barrier(deps);
    if (NE == 288) return k_router_topk_small<9>(q, rl, bias, top, topw, NE, TOPK, wscale, T, deps);
    if (NE == 256) return k_router_topk_small<8>(q, rl, bias, top, topw, NE, TOPK, wscale, T, deps);
    if (NE == 512) return k_router_topk_small<16>(q, rl, bias, top, topw, NE, TOPK, wscale, T, deps);
    if (NE <= 512) return k_router_topk_small<0>(q, rl, bias, top, topw, NE, TOPK, wscale, T, deps);
    return k_router_topk_large(q, rl, bias, top, topw, NE, TOPK, wscale, T, deps);
}


sycl::event k_gather_rows(sycl::queue& q, const sycl::half* x, const int32_t* idx,
                          sycl::half* out, uint32_t n_rows, uint32_t H,
                          const std::vector<sycl::event>& deps) {
    return ie::ps(q, "g5_gather", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<2>(n_rows, H), [=](sycl::id<2> id) {
            out[uint64_t(id[0]) * H + id[1]] =
                x[uint64_t(idx[id[0]]) * H + id[1]];
        });
    });
}

// acc[idx[i]] += w[i] * dn[i]  — one work-item per (row, d); rows carry
// DISTINCT tokens within one expert (a token picks an expert at most once),
// so there are no intra-launch collisions; cross-expert accumulation is
// ordered by the sequential per-expert launches (deterministic).
sycl::event k_scatter_add(sycl::queue& q, const float* dn, const int32_t* idx,
                          const float* w, float* acc, uint32_t n_rows, uint32_t H,
                          const std::vector<sycl::event>& deps) {
    return ie::ps(q, "g5_scatter", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<2>(n_rows, H), [=](sycl::id<2> id) {
            const uint32_t r = uint32_t(id[0]), d = uint32_t(id[1]);
            acc[uint64_t(idx[r]) * H + d] += w[r] * dn[uint64_t(r) * H + d];
        });
    });
}

// Decode variant: ONE f16 row into the accumulator at weight w[0].
sycl::event k_scatter_add_h(sycl::queue& q, const sycl::half* dn, const int32_t* idx,
                            const float* w, float* acc, uint32_t H,
                            const std::vector<sycl::event>& deps) {
    return ie::ps(q, "g5_scatter_h", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(H), [=](sycl::id<1> id) {
            acc[uint64_t(idx[0]) * H + id[0]] += w[0] * float(dn[id[0]]);
        });
    });
}

// Multi-row f16 scatter (the rows-kernel MoE path).
sycl::event k_scatter_add_h_rows(sycl::queue& q, const sycl::half* dn,
                                 const int32_t* idx, const float* w, float* acc,
                                 uint32_t n_rows, uint32_t H,
                                 const std::vector<sycl::event>& deps) {
    return ie::ps(q, "g5_scatter_hr", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<2>(n_rows, H), [=](sycl::id<2> id) {
            const uint32_t r = uint32_t(id[0]), d = uint32_t(id[1]);
            acc[uint64_t(idx[r]) * H + d] += w[r] * float(dn[uint64_t(r) * H + d]);
        });
    });
}

// q_lat[t, h, l] = Σ_d k_b[h][l][d] · q_head[t, h*256 + d]   (absorb form).
// k_b is the GGUF-native f16 [64][512][256] (d fastest).
sycl::event k_mla_kabsorb(sycl::queue& q, const float* q_head, const sycl::half* k_b,
                          float* q_lat, uint32_t T, uint32_t NH, uint32_t LAT,
                          uint32_t HD, const std::vector<sycl::event>& deps) {
    constexpr int SG = 16, LOOK = 8;
    return ie::ps(q, "g5_kabsorb", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({uint64_t(T) * NH, uint64_t(LAT) * SG},
                                         {1, SG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t th = uint32_t(it.get_group(0));
            const uint32_t t = th / NH, hh = th % NH;
            const uint32_t l = uint32_t(it.get_group(1));
            const uint32_t lane = uint32_t(it.get_local_id(1));
            auto sg = it.get_sub_group();
            const float* qh = q_head + (uint64_t(t) * NH + hh) * HD;
            const sycl::half* kb = k_b + (uint64_t(hh) * LAT + l) * HD;
            float acc = 0.f;
            // Load ahead without changing the ordered FMA chain.
            for (uint32_t base = lane; base < HD; base += SG * LOOK) {
                float weights[LOOK], inputs[LOOK];
                #pragma unroll
                for (int j = 0; j < LOOK; ++j) {
                    const uint32_t i = base + j * SG;
                    if (i < HD) {
                        weights[j] = float(kb[i]);
                        inputs[j] = qh[i];
                    }
                }
                #pragma unroll
                for (int j = 0; j < LOOK; ++j)
                    if (base + j * SG < HD)
                        acc = sycl::fma(weights[j], inputs[j], acc);
            }
            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (lane == 0) q_lat[(uint64_t(t) * NH + hh) * LAT + l] = acc;
        });
    });
}

// o16[t, h*256 + o] = Σ_l v_b[h][o][l] · att[t, h, l]   (absorb form).
// v_b is the GGUF-native f16 [64][256][512] (l fastest).
sycl::event k_mla_vabsorb(sycl::queue& q, const float* att, const sycl::half* v_b,
                          sycl::half* o16, uint32_t T, uint32_t NH, uint32_t LAT,
                          uint32_t HD, const std::vector<sycl::event>& deps) {
    constexpr int SG = 16, LOOK = 8;
    return ie::ps(q, "g5_vabsorb", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({uint64_t(T) * NH, uint64_t(HD) * SG},
                                         {1, SG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t th = uint32_t(it.get_group(0));
            const uint32_t t = th / NH, hh = th % NH;
            const uint32_t o = uint32_t(it.get_group(1));
            const uint32_t lane = uint32_t(it.get_local_id(1));
            auto sg = it.get_sub_group();
            const float* ar = att + (uint64_t(t) * NH + hh) * LAT;
            const sycl::half* vb = v_b + (uint64_t(hh) * HD + o) * LAT;
            float acc = 0.f;
            // Load ahead without changing the ordered FMA chain.
            for (uint32_t base = lane; base < LAT; base += SG * LOOK) {
                float weights[LOOK], inputs[LOOK];
                #pragma unroll
                for (int j = 0; j < LOOK; ++j) {
                    const uint32_t i = base + j * SG;
                    if (i < LAT) {
                        weights[j] = float(vb[i]);
                        inputs[j] = ar[i];
                    }
                }
                #pragma unroll
                for (int j = 0; j < LOOK; ++j)
                    if (base + j * SG < LAT)
                        acc = sycl::fma(weights[j], inputs[j], acc);
            }
            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (lane == 0) o16[uint64_t(t) * NH * HD + hh * HD + o] = sycl::half(acc);
        });
    });
}

// Dense causal MLA attention over the latent cache (absorbed MQA: the key AND
// value of position j are the same latent row). One work-group per (t, h);
// scores staged in SLM (n_ctx <= 2051 by the caller's refusal), two-pass
// softmax in fp32.  att[t, h, :] = Σ_j p_j · lat[j, :].
sycl::event k_mla_attend(sycl::queue& q, const float* q_lat, const sycl::half* lat,
                         float* att, uint32_t T, uint32_t NH, uint32_t LAT,
                         uint32_t pos0, float scale, uint32_t slm_cap,
                         const std::vector<sycl::event>& deps) {
    constexpr uint32_t WG = 128;
    return ie::ps(q, "g5_attend", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> p_slm(slm_cap, h);
        h.parallel_for(sycl::nd_range<2>({uint64_t(T) * NH, WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t th = uint32_t(it.get_group(0));
            const uint32_t t = th / NH, hh = th % NH;
            const uint32_t lane = uint32_t(it.get_local_id(1));
            const uint32_t n = pos0 + t + 1;          // causal: j in [0, n)
            const float* qr = q_lat + (uint64_t(t) * NH + hh) * LAT;

            // pass A: scores into SLM (each lane strides positions)
            float mx = -1e30f;
            for (uint32_t j = lane; j < n; j += WG) {
                const sycl::half* lj = lat + uint64_t(j) * LAT;
                float s = 0.f;
                for (uint32_t d = 0; d < LAT; ++d) s = sycl::fma(float(lj[d]), qr[d], s);
                s *= scale;
                p_slm[j] = s;
                mx = sycl::fmax(mx, s);
            }
            mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
            float sum = 0.f;
            for (uint32_t j = lane; j < n; j += WG) {
                const float e = sycl::native::exp(p_slm[j] - mx);
                p_slm[j] = e;
                sum += e;
            }
            sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
            const float inv = 1.f / sum;
            sycl::group_barrier(it.get_group());

            // pass B: lane owns LAT/WG dims of the accumulator
            float* ar = att + (uint64_t(t) * NH + hh) * LAT;
            for (uint32_t d = lane; d < LAT; d += WG) {
                float acc = 0.f;
                for (uint32_t j = 0; j < n; ++j)
                    acc = sycl::fma(p_slm[j], float(lat[uint64_t(j) * LAT + d]), acc);
                ar[d] = acc * inv;
            }
        });
    });
}

// x[t, d] = mean_s streams[t, s, d]  — the final unweighted stream merge
// ("no hc_head tensor here", PR graph).
// ===========================================================================
// P2 — DSA lightning indexer + sparse MLA attention (ctx > n_sel).
// Semantics verified against modeling_glm5_next.py (Glm5NextIndexer.forward,
// get_pooled_states, append_visible_tail). See docs/glm53/CAMPAIGN_2026-09-03.md.
// ===========================================================================

// LayerNorm WITH BIAS over the last dim (the indexer's k_norm, eps 1e-6 — the
// port plan warns this is a LayerNorm, NOT an RMS norm like everything else).
sycl::event k_g5_idx_lnb(sycl::queue& q, const sycl::half* x, const float* w,
                         const float* b, sycl::half* out, uint32_t T, uint32_t D,
                         float eps, const std::vector<sycl::event>& deps) {
    constexpr uint32_t WG = 128;
    return ie::ps(q, "g5_idx_lnb", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({uint64_t(T), WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t t = uint32_t(it.get_group(0));
            const uint32_t lane = uint32_t(it.get_local_id(1));
            const sycl::half* xr = x + uint64_t(t) * D;
            float s = 0.f;
            for (uint32_t d = lane; d < D; d += WG) s += float(xr[d]);
            s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
            const float mean = s / float(D);
            float v = 0.f;
            for (uint32_t d = lane; d < D; d += WG) {
                const float z = float(xr[d]) - mean;
                v += z * z;
            }
            v = sycl::reduce_over_group(it.get_group(), v, sycl::plus<float>());
            const float inv = sycl::rsqrt(v / float(D) + eps);
            for (uint32_t d = lane; d < D; d += WG)
                out[uint64_t(t) * D + d] =
                    sycl::half((float(xr[d]) - mean) * inv * w[d] + b[d]);
        });
    });
}

// Gated compressor pooling. For pool p (tokens [p*KP, p*KP+KP)) and dim d:
//   logits[i] = gate[tok_i][d] + ape[i][d];  prob = softmax_i(logits)
//   pool_key[p][d] = sum_i prob[i] * k[tok_i][d]
// The softmax is PER DIMENSION over the KP tokens (reference: softmax(dim=2)).
// Tokens below `start_pos` come from the open-pool roll buffer, indexed by
// (j % KP) — a pool's members are exactly the tokens sharing its pool id.
sycl::event k_g5_pool_key(sycl::queue& q, const sycl::half* kchunk,
                          const sycl::half* gchunk, const sycl::half* kroll,
                          const sycl::half* groll, const float* ape,
                          sycl::half* pool_key, uint32_t p0, uint32_t npool,
                          uint32_t KP, uint32_t D, uint32_t start_pos,
                          const std::vector<sycl::event>& deps) {
    return ie::ps(q, "g5_pool_key", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<2>(npool, D), [=](sycl::id<2> id) {
            const uint32_t p = p0 + uint32_t(id[0]);
            const uint32_t d = uint32_t(id[1]);
            float mx = -1e30f;
            float lg[8];                    // KP <= 8
            for (uint32_t i = 0; i < KP; ++i) {
                const uint32_t j = p * KP + i;
                const float g = j >= start_pos
                    ? float(gchunk[uint64_t(j - start_pos) * D + d])
                    : float(groll[uint64_t(j % KP) * D + d]);
                lg[i] = g + ape[uint64_t(i) * D + d];
                mx = sycl::fmax(mx, lg[i]);
            }
            float sum = 0.f;
            for (uint32_t i = 0; i < KP; ++i) { lg[i] = sycl::exp(lg[i] - mx); sum += lg[i]; }
            const float inv = 1.f / sum;
            float acc = 0.f;
            for (uint32_t i = 0; i < KP; ++i) {
                const uint32_t j = p * KP + i;
                const float kv = j >= start_pos
                    ? float(kchunk[uint64_t(j - start_pos) * D + d])
                    : float(kroll[uint64_t(j % KP) * D + d]);
                acc += lg[i] * inv * kv;
            }
            pool_key[uint64_t(p) * D + d] = sycl::half(acc);
        });
    });
}

// Selected pool ids -> ASCENDING raw token ids, plus the incomplete tail.
//
// Ascending is deliberate: with the whole context selected (ctx <= n_sel) the
// gather then visits exactly the positions the dense path visits, in the same
// order, so the two paths are BIT-IDENTICAL and the dense PPL ruler gates the
// sparse path. It also keeps the latent reads monotone.
//
// tail (append_visible_tail): visible_count = pos+1, tail_count = (pos+1) % KP,
// tail tokens are [pos+1-tail_count, pos]. Those are exactly the members of the
// still-open pool, which the complete-pool selection cannot cover.
sycl::event k_g5_pool_expand(sycl::queue& q, const int32_t* sel, const int32_t* pos,
                             int32_t* tok, int32_t* n_valid, uint32_t T, uint32_t sk,
                             uint32_t sort_w, uint32_t KP, uint32_t n_sel, uint32_t kv_len,
                             const std::vector<sycl::event>& deps) {
    constexpr uint32_t WG = 256;
    return ie::ps(q, "g5_pool_expand", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<int32_t, 1> s_slm(sort_w, h);
        h.parallel_for(sycl::nd_range<2>({uint64_t(T), WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t t = uint32_t(it.get_group(0));
            const uint32_t lane = uint32_t(it.get_local_id(1));
            const int32_t* sr = sel + uint64_t(t) * sk;
            // load, mapping the -1 sentinel (and the pad up to the power-of-two
            // sort width) to INT_MAX so they sort to the end and emit -1
            for (uint32_t i = lane; i < sort_w; i += WG)
                s_slm[i] = (i < sk && sr[i] >= 0) ? sr[i] : INT32_MAX;
            sycl::group_barrier(it.get_group());
            // bitonic sort ascending (sort_w is a power of two: index_topk/KP)
            for (uint32_t k2 = 2; k2 <= sort_w; k2 <<= 1)
                for (uint32_t j2 = k2 >> 1; j2 > 0; j2 >>= 1) {
                    for (uint32_t i = lane; i < sort_w; i += WG) {
                        const uint32_t ix = i ^ j2;
                        if (ix > i) {
                            const bool up = (i & k2) == 0;
                            const int32_t a = s_slm[i], b = s_slm[ix];
                            if ((a > b) == up) { s_slm[i] = b; s_slm[ix] = a; }
                        }
                    }
                    sycl::group_barrier(it.get_group());
                }
            // COMPACT expansion: valid pool tokens first (ascending), then the
            // tail, then -1 padding, and n_valid[t] = the prefix length.
            // Two things this buys: (1) the attend kernel loops over n_valid
            // instead of all n_sel slots — at position 5 that is 6 iterations,
            // not 2051 (measured: 6 ms/token at ctx 131072, pos 0); (2) the
            // valid prefix is exactly the dense kernel's j = 0..n-1 order, so
            // per-lane softmax partials match the dense path and the two are
            // BIT-IDENTICAL wherever selection is complete (ctx <= n_sel).
            // Sorted pools are [valid ascending..., INT32_MAX...]; count them.
            uint32_t nv_p = 0;
            for (uint32_t i = lane; i < sort_w; i += WG) nv_p += s_slm[i] != INT32_MAX;
            nv_p = sycl::reduce_over_group(it.get_group(), nv_p, sycl::plus<uint32_t>());
            const int32_t pz = pos[t];
            const uint32_t tail_count = uint32_t(pz + 1) % KP;
            const uint32_t nfull = nv_p * KP;
            const uint32_t nv = nfull + tail_count;
            int32_t* tr = tok + uint64_t(t) * n_sel;
            for (uint32_t i = lane; i < n_sel; i += WG) {
                int32_t v = -1;
                if (i < nfull) {
                    const int32_t c = s_slm[i / KP] * int32_t(KP) + int32_t(i % KP);
                    if (uint32_t(c) < kv_len) v = c;
                } else if (i < nv) {
                    v = pz + 1 - int32_t(tail_count) + int32_t(i - nfull);
                }
                tr[i] = v;
            }
            if (lane == 0) n_valid[t] = int32_t(nv);
        });
    });
}

// Per-head indexer weights: y[T, IH] = x[T, H] @ w_proj[H, IH].
// w_proj stays FP32 on purpose — the port plan records that bf16 here flips
// near-tie pool rankings, and a flipped pool is a DISCRETE selection change.
sycl::event k_g5_idx_wproj(sycl::queue& q, const sycl::half* x, const float* wp,
                           float* y, uint32_t T, uint32_t H, uint32_t IH,
                           const std::vector<sycl::event>& deps) {
    // One sub-group per (t, head): lanes stride k, then a sub-group reduce.
    // The first version ran ONE work-item per head with a serial 4096-long
    // loop — 32 items on a 32-core GPU, 0.27 ms per call, 3 ms/token across
    // the 12 MLA layers. Deterministic: fixed lane assignment + sub-group
    // reduce, the same contract every GEMV in this file relies on.
    // indexer.proj comes through the loader's F32 branch in GGUF-NATIVE order
    // W[n*K+k]: element (head o, input k) is wp[o*H + k].
    // v3 (2026-09-04, Astra review): v2 put all 32 heads' sub-groups in ONE
    // 512-item work-group per token — at decode T=1 that is one Xe core of 32,
    // 43.6 us/call at 12 GB/s over a 512 KiB matrix. One work-group = one
    // sub-group = one (token, head) now; 32 work-groups spread the cores.
    // Same lane -> k sequence and the same sub-group reduce: byte-identical.
    constexpr uint32_t SG = 16;
    return ie::ps(q, "g5_idx_wproj", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({uint64_t(T) * IH, SG}, {1, SG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t g = uint32_t(it.get_group(0));
            const uint32_t t = g / IH, o = g % IH;
            const uint32_t lane = uint32_t(it.get_local_id(1));
            const sycl::half* xr = x + uint64_t(t) * H;
            const float* wr = wp + uint64_t(o) * H;
            float acc = 0.f;
            for (uint32_t k = lane; k < H; k += SG)
                acc = sycl::fma(float(xr[k]), wr[k], acc);
            acc = sycl::reduce_over_group(it.get_sub_group(), acc, sycl::plus<float>());
            if (lane == 0) y[uint64_t(t) * IH + o] = acc;
        });
    });
}

// Query positions for this chunk — ds4_indexer_topk derives its causal pool
// threshold (positions[t] + 1) / compress_rate from these.
sycl::event k_g5_idx_pos(sycl::queue& q, int32_t* pos, uint32_t T, uint32_t start_pos,
                         const std::vector<sycl::event>& deps) {
    return ie::ps(q, "g5_idx_pos", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(T), [=](sycl::id<1> id) {
            pos[id[0]] = int32_t(start_pos + uint32_t(id[0]));
        });
    });
}

// MLA attention over a GATHERED index list (the sparse twin of k_mla_attend).
// SLM holds n_sel scores — the same 2051-float budget the dense path already
// sized for, independent of context length. -1 entries are masked out.
// v3 (2026-09-04): SPLIT-K. v1 = one work-group per (token, head): 64 WGs at
// T=1, each re-reading all 2051 selected rows with a 1 KiB stride — 1.99 ms/call,
// 36% of decode GPU at 24K. v2 = one WG per TOKEN with all heads sharing SLM
// tiles: the locality was right and the arithmetic was wrong — at T=1 that is
// ONE work-group on ONE of 32 Xe cores, and it measured SLOWER (2.30 ms).
// v3 keeps v2's per-WG structure (rows tiled through SLM, online softmax,
// coalesced accumulate) and restores parallelism by splitting the selected
// rows into kSplit slices: one WG per (token, head-group, slice). Each WG
// emits a partial (m, l, o[512]) for its heads; a combine kernel merges the
// slices with the standard online-softmax rescale. The measured GLM dispatch
// uses 16 heads/WG at T=1 (32 WGs) and 32 heads/WG for larger batches. The
// original four-head path remains available for other shapes. This grouping
// preserves v3 arithmetic; v3 itself was not bit-identical to v1/v2.
template<uint32_t HEADS, bool SHARE_EXP>
sycl::event k_g5_mla_attend_sel_impl(sycl::queue& q, const float* q_lat,
                                const sycl::half* lat, const int32_t* sel,
                                const int32_t* n_valid, float* att, float* part,
                                uint32_t T, uint32_t NH, uint32_t LAT, uint32_t n_sel,
                                uint32_t kv_len, float scale,
                                const std::vector<sycl::event>& deps) {
    constexpr uint32_t SG = 16;
    constexpr uint32_t HPW = HEADS;              // heads per work-group
    constexpr uint32_t kSplit = 8;           // slices of the selected rows
    constexpr uint32_t TILE = 32;
    const uint32_t WG = HPW * SG;
    const uint32_t n_hg = NH / HPW;
    // Each partial stores m, l, six padding floats, then o[LAT].
    const uint32_t PSTR = LAT + 8;      // (m, l, pad x6) then o[LAT]: 32-byte aligned blocks
    auto e1 = ie::ps(q, "g5_attend_sel", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<sycl::half, 1> tile(TILE * LAT, h);     // 32 KiB
        sycl::local_accessor<float, 1> ps(HPW * TILE, h);
        sycl::local_accessor<int32_t, 1> rowj(TILE, h);                // this tile's row ids
        h.parallel_for(sycl::nd_range<2>({uint64_t(T) * n_hg * kSplit, WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t g = uint32_t(it.get_group(0));
            const uint32_t t = g / (n_hg * kSplit);
            const uint32_t hg = (g / kSplit) % n_hg;
            const uint32_t sl = g % kSplit;
            const uint32_t lid = uint32_t(it.get_local_id(1));
            const uint32_t hl = lid / SG, lane = lid % SG;
            const uint32_t hh = hg * HPW + hl;
            auto sg = it.get_sub_group();
            const int32_t* sr = sel + uint64_t(t) * n_sel;
            const uint32_t nv = sycl::min(uint32_t(n_valid[t]), n_sel);
            // this slice's row range [r0, r1)
            const uint32_t per = (nv + kSplit - 1) / kSplit;
            const uint32_t r0 = sycl::min(sl * per, nv), r1 = sycl::min(r0 + per, nv);
            // v3.2 (2026-09-04 n19 asm): v3.1 read the SLM tile as 32 separate
            // 16-bit loads per row per pass (load.slm.d16u32), and the compiler
            // put a scoreboard wait after nearly every one (65 sync.allrd for
            // 64 loads) — ~40 exposed cycles x 64 x 256 rows per lane ~= the
            // whole 0.79 ms/call. Now lane l owns four CONTIGUOUS 8-dim blocks
            // (dims 8l..8l+7, 128+8l.., 256+8l.., 384+8l..): a row is four
            // 16-byte loads per pass, 16 lanes at 16-byte stride = 32 distinct
            // banks. The per-lane partial sums group different dims than v3.1,
            // so this is gate P (PPL), not byte identity.
            const float* qr = q_lat + (uint64_t(t) * NH + hh) * LAT;
            float qreg[32];
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const sycl::vec<float, 8> qv =
                    *reinterpret_cast<const sycl::vec<float, 8>*>(qr + b * 128 + lane * 8);
                #pragma unroll
                for (int u = 0; u < 8; ++u) qreg[b * 8 + u] = qv[u];
            }
            float oacc[32];
            #pragma unroll
            for (int k = 0; k < 32; ++k) oacc[k] = 0.f;
            float m_run = -1e30f, l_run = 0.f;
            sycl::half* tp = tile.get_multi_ptr<sycl::access::decorated::no>().get();
            for (uint32_t s0 = r0; s0 < r1; s0 += TILE) {
                const uint32_t nt = sycl::min(TILE, r1 - s0);
                // Stage row IDs once. The first LAT/8 work-items load
                // contiguous vectors, with eight independent rows in flight.
                // Additional heads reuse the same tile after the WG barrier.
                if (lid < TILE) rowj[lid] = (lid < nt) ? sr[s0 + lid] : -1;
                sycl::group_barrier(it.get_group());
                if (lid < LAT / 8) for (uint32_t n0 = 0; n0 < nt; n0 += 8) {
                    sycl::vec<sycl::half, 8> v[8];
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        const uint32_t n = n0 + uint32_t(u);
                        const int32_t j = (n < nt) ? rowj[n] : -1;
                        if (j >= 0 && uint32_t(j) < kv_len)
                            v[u] = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(
                                       lat + uint64_t(j) * LAT + lid * 8);
                        else
                            v[u] = sycl::vec<sycl::half, 8>(sycl::half(0.f));
                    }
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        const uint32_t n = n0 + uint32_t(u);
                        if (n < nt)
                            *reinterpret_cast<sycl::vec<sycl::half, 8>*>(tp + n * LAT + lid * 8) = v[u];
                    }
                }
                sycl::group_barrier(it.get_group());
                float tmx = -1e30f;
                // Two independent row chains hide dot-product latency without
                // changing any row's FMA order or subgroup reduction.
                constexpr int RP = 2;
                for (uint32_t rb = 0; rb < nt; rb += RP) {
                    float acc[RP] = {};
                    #pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        sycl::vec<sycl::half, 8> tv[RP];
                        #pragma unroll
                        for (int rr = 0; rr < RP; ++rr)
                            if (rb + rr < nt)
                                tv[rr] = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(
                                    tp + (rb + rr) * LAT + b * 128 + lane * 8);
                        #pragma unroll
                        for (int u = 0; u < 8; ++u) {
                            #pragma unroll
                            for (int rr = 0; rr < RP; ++rr)
                                if (rb + rr < nt)
                                    acc[rr] = sycl::fma(float(tv[rr][u]), qreg[b * 8 + u], acc[rr]);
                        }
                    }
                    #pragma unroll
                    for (int rr = 0; rr < RP; ++rr) {
                        if (rb + rr < nt) {
                            const int32_t j = sr[s0 + rb + rr];
                            const float sum = sycl::reduce_over_group(sg, acc[rr], sycl::plus<float>());
                            const float v = (j >= 0 && uint32_t(j) < kv_len) ? sum * scale : -1e30f;
                            if (lane == 0) ps[hl * TILE + rb + rr] = v;
                            tmx = sycl::fmax(tmx, v);
                        }
                    }
                }
                const float m_new = sycl::fmax(m_run, tmx);
                const float corr = sycl::native::exp(m_run - m_new);
                l_run *= corr;
                #pragma unroll
                for (int k = 0; k < 32; ++k) oacc[k] *= corr;
                sycl::group_barrier(sg);
                if constexpr (SHARE_EXP) {
                    // Compute each coefficient once per head, then reuse it
                    // across lanes while keeping the ordered value-FMA chain.
                    for (uint32_t r = lane; r < nt; r += SG) {
                        const float v = ps[hl * TILE + r];
                        ps[hl * TILE + r] = v > -1e29f ? sycl::native::exp(v - m_new) : 0.f;
                    }
                    sycl::group_barrier(sg);
                }
                for (uint32_t r = 0; r < nt; ++r) {
                    const float v = ps[hl * TILE + r];
                    float e;
                    if constexpr (SHARE_EXP) e = v;
                    else e = v > -1e29f ? sycl::native::exp(v - m_new) : 0.f;
                    l_run += e;
                    #pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        const sycl::vec<sycl::half, 8> tv =
                            *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(tp + r * LAT + b * 128 + lane * 8);
                        #pragma unroll
                        for (int u = 0; u < 8; ++u)
                            oacc[b * 8 + u] = sycl::fma(e, float(tv[u]), oacc[b * 8 + u]);
                    }
                }
                m_run = m_new;
                sycl::group_barrier(it.get_group());
            }
            float* pp = part + (uint64_t(uint64_t(t) * NH + hh) * kSplit + sl) * PSTR;
            if (lane == 0) { pp[0] = m_run; pp[1] = l_run; }
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                sycl::vec<float, 8> ov;
                #pragma unroll
                for (int u = 0; u < 8; ++u) ov[u] = oacc[b * 8 + u];
                *reinterpret_cast<sycl::vec<float, 8>*>(pp + 8 + b * 128 + lane * 8) = ov;
            }
        });
    });
    // combine: one sub-group per (token, head), merge the kSplit partials
    return ie::ps(q, "g5_attend_combine", [&](sycl::handler& h) {
        h.depends_on({e1});
        h.parallel_for(sycl::nd_range<2>({uint64_t(T) * NH, SG}, {1, SG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t th = uint32_t(it.get_group(0));
            const uint32_t lane = uint32_t(it.get_local_id(1));
            const float* pb = part + uint64_t(th) * kSplit * PSTR;
            float M = -1e30f;
            for (uint32_t s = 0; s < kSplit; ++s) M = sycl::fmax(M, pb[s * PSTR]);
            float L = 0.f;
            float o[32];
            #pragma unroll
            for (int k = 0; k < 32; ++k) o[k] = 0.f;
            for (uint32_t s = 0; s < kSplit; ++s) {
                const float ms = pb[s * PSTR], ls = pb[s * PSTR + 1];
                const float w = (ls > 0.f) ? sycl::native::exp(ms - M) : 0.f;
                L += ls * w;
                #pragma unroll
                for (int k = 0; k < 32; ++k)
                    o[k] = sycl::fma(w, pb[s * PSTR + 8 + lane + k * SG], o[k]);
            }
            const float inv = L > 0.f ? 1.f / L : 0.f;
            float* ar = att + uint64_t(th) * LAT;
            #pragma unroll
            for (int k = 0; k < 32; ++k) ar[lane + k * SG] = o[k] * inv;
        });
    });
}


sycl::event k_g5_mla_attend_sel(sycl::queue& q, const float* q_lat,
                                const sycl::half* lat, const int32_t* sel,
                                const int32_t* n_valid, float* att, float* part,
                                uint32_t T, uint32_t NH, uint32_t LAT, uint32_t n_sel,
                                uint32_t kv_len, float scale,
                                const std::vector<sycl::event>& deps) {
    // Measured GLM shape: increase tile reuse without changing split count,
    // per-head arithmetic, partial layout, or the combine kernel.
    if (NH == 64 && LAT == 512) {
        if (T == 1)
            return k_g5_mla_attend_sel_impl<16, true>(q, q_lat, lat, sel, n_valid,
                att, part, T, NH, LAT, n_sel, kv_len, scale, deps);
        return k_g5_mla_attend_sel_impl<32, true>(q, q_lat, lat, sel, n_valid,
            att, part, T, NH, LAT, n_sel, kv_len, scale, deps);
    }
    return k_g5_mla_attend_sel_impl<4, false>(q, q_lat, lat, sel, n_valid,
        att, part, T, NH, LAT, n_sel, kv_len, scale, deps);
}
sycl::event k_stream_mean(sycl::queue& q, const float* streams, float* x,
                          uint32_t T, uint32_t H, uint32_t hc,
                          const std::vector<sycl::event>& deps) {
    return ie::ps(q, "g5_mean", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<2>(T, H), [=](sycl::id<2> id) {
            const uint32_t t = uint32_t(id[0]), d = uint32_t(id[1]);
            float s = 0.f;
            for (uint32_t j = 0; j < hc; ++j)
                s += streams[(uint64_t(t) * hc + j) * H + d];
            x[uint64_t(t) * H + d] = s / float(hc);
        });
    });
}

}  // namespace

sycl::event glm5_router_logits(sycl::queue& q, const float* x, const float* w,
                               float* rl, uint32_t T, uint32_t H, uint32_t E,
                               const std::vector<sycl::event>& deps) {
    return k_router_logits(q, x, w, rl, T, H, E, deps);
}

sycl::event glm5_router_topk(sycl::queue& q, const float* logits, const float* bias,
                            int32_t* indices, float* weights, uint32_t experts,
                            uint32_t top_k, float scale, uint32_t tokens,
                            const std::vector<sycl::event>& deps) {
    return k_router_topk(q, logits, bias, indices, weights, experts, top_k, scale, tokens, deps);
}

std::string Glm5NextModel::init_runtime(uint32_t max_ctx, uint32_t max_chunk) {
    dev_alloc_tag("workspace");
    max_ctx_ = max_ctx;
    max_chunk_ = max_chunk;
    const uint32_t H = cfg_.hidden, hc = cfg_.hc_count;
    // gemm_fp16 stores full 8x16 output tiles; every [MT, N] workspace is
    // padded to the next row-tile so a ragged T's tail tile lands in-buffer.
    const uint32_t MT = (max_chunk + 7) & ~7u;
    const uint32_t n_tf = cfg_.n_transformer_layers();
    const uint32_t DI = cfg_.n_q_heads * cfg_.kda_head_dim;   // 8192
    const uint32_t LAT = cfg_.kv_lora_rank;                   // 512
    const uint32_t EF = cfg_.expert_ffn;                      // 2048

    // Cache indices are RANGE-LOCAL: a pipeline stage sizes its KV/scan
    // state and expert cache only for the blocks it owns.
    lin_idx_.assign(cfg_.n_layers, -1);
    full_idx_.assign(cfg_.n_layers, -1);
    uint32_t n_lin = 0, n_full = 0;
    for (uint32_t L = layer_lo_; L < layer_hi_; ++L) {
        if (cfg_.is_full_attn(L)) full_idx_[L] = int32_t(n_full++);
        else                      lin_idx_[L]  = int32_t(n_lin++);
    }
    if (n_lin == 0) n_lin = 1;   // DeltaNetState refuses 0 layers

    DeltaNetStateConfig dc{};
    dc.n_layers_linear = n_lin;
    dc.n_v_heads = cfg_.n_q_heads;
    dc.v_head_dim = cfg_.kda_head_dim;
    dc.k_head_dim = cfg_.kda_head_dim;
    dc.conv_channels = 3 * DI;      // q|k|v conv states, per-stream slices
    dc.conv_kernel = cfg_.conv_kernel;
    if (auto e = dn_.init(*alloc_, dc); !e.empty()) return e;

    auto dev = [&](uint64_t bytes) -> void* {
        void* p = alloc_->malloc(bytes);
        if (p) owned_.push_back(p);
        return p;
    };
    auto f32buf = [&](float*& p, uint64_t n) { p = static_cast<float*>(dev(n * 4)); return p != nullptr; };
    auto h16buf = [&](sycl::half*& p, uint64_t n) { p = static_cast<sycl::half*>(dev(n * 2)); return p != nullptr; };

    bool ok = true;
    lat_cache_ = static_cast<sycl::half*>(dev(uint64_t(n_full) * max_ctx * LAT * 2));
    // ---- P2 DSA sparse path ------------------------------------------------
    // Engages ONLY above the selection width. At ctx <= n_sel the whole context
    // is selected, so the dense path is exactly right and stays bit-identical
    // (that is what makes the 1.8694 PPL ruler a valid gate for both).
    n_sel_    = cfg_.indexer_top_k + cfg_.indexer_kpool - 1;
    select_k_ = cfg_.indexer_top_k / cfg_.indexer_kpool;
    n_pools_  = (max_ctx + cfg_.indexer_kpool - 1) / cfg_.indexer_kpool;
    // IE_G5_FORCE_SPARSE=1 runs the indexer at ctx <= n_sel, where every valid
    // pool fits inside select_k and the ascending gather visits exactly the
    // positions the dense path visits, in the same order => the two must be
    // BIT-IDENTICAL. That is the gate for the whole sparse path.
    sparse_   = max_ctx > n_sel_ || std::getenv("IE_G5_FORCE_SPARSE") != nullptr;
    if (sparse_) {
        const uint32_t IH = cfg_.indexer_n_heads, IHD = cfg_.indexer_head_dim,
                       KP = cfg_.indexer_kpool;
        // Scores are the one workspace that scales with BOTH context and chunk
        // (T x n_pools). DS4 lost 250K to exactly this class, so it is strip-
        // counted: kIdxStrip query rows at a time, never max_chunk.
        const uint32_t strip = std::min<uint32_t>(kIdxStrip, MT);
        pool_key_  = static_cast<sycl::half*>(dev(uint64_t(n_full) * n_pools_ * IHD * 2));
        idx_k_     = static_cast<sycl::half*>(dev(uint64_t(MT) * IHD * 2));
        idx_gate_  = static_cast<sycl::half*>(dev(uint64_t(MT) * IHD * 2));
        idx_kroll_ = static_cast<sycl::half*>(dev(uint64_t(n_full) * KP * IHD * 2));
        idx_groll_ = static_cast<sycl::half*>(dev(uint64_t(n_full) * KP * IHD * 2));
        idx_sel_   = static_cast<int32_t*>(dev(uint64_t(MT) * select_k_ * 4));
        idx_tok_   = static_cast<int32_t*>(dev(uint64_t(MT) * n_sel_ * 4));
        idx_pos_   = static_cast<int32_t*>(dev(uint64_t(MT) * 4));
        idx_nv_    = static_cast<int32_t*>(dev(uint64_t(MT) * 4));
        // split-K attention partials: [MT][NH][8 slices][LAT+2] fp32 (v3)
        // split-K attention partials: [strip][NH][8 slices][LAT+2] fp32. Sized for
        // ONE kIdxStrip-row strip (67 MiB), not max_chunk — at MT=1024 that was
        // 1.004 GiB/card of scratch (Astra review 2026-09-04). The attention
        // call below runs in strips to match.
        ok = ok && f32buf(idx_part_, uint64_t(strip) * cfg_.n_q_heads * 8 * (LAT + 8));
        roll_bytes_ = uint64_t(n_full) * KP * IHD * 2;
        snap_kroll_ = static_cast<sycl::half*>(dev(roll_bytes_));
        snap_groll_ = static_cast<sycl::half*>(dev(roll_bytes_));
        sv_idx_k_ = static_cast<sycl::half*>(dev(uint64_t(n_full) * kSpecMax * IHD * 2));
        sv_idx_g_ = static_cast<sycl::half*>(dev(uint64_t(n_full) * kSpecMax * IHD * 2));
        ok = ok && f32buf(idx_q_, uint64_t(MT) * IH * IHD) &&
                   f32buf(idx_w_, uint64_t(MT) * IH) &&
                   f32buf(idx_scores_, uint64_t(strip) * n_pools_);
        ok = ok && pool_key_ && idx_k_ && idx_gate_ && idx_kroll_ && idx_groll_ &&
                   idx_sel_ && idx_tok_ && idx_pos_ && idx_nv_ &&
                   snap_kroll_ && snap_groll_ && sv_idx_k_ && sv_idx_g_;
        if (!ok) return "glm5next init_runtime: DSA indexer alloc failed";
        std::fprintf(stderr,
                     "[glm5next] DSA sparse attention: ctx %u, %u pools of %u, "
                     "select %u -> %u tokens; pool keys %.2f GiB, latent KV %.2f GiB\n",
                     max_ctx, n_pools_, KP, select_k_, n_sel_,
                     double(uint64_t(n_full) * n_pools_ * IHD * 2) / 1073741824.0,
                     double(uint64_t(n_full) * max_ctx * LAT * 2) / 1073741824.0);
    }
    ok = ok && lat_cache_;
    d_tokens_ = static_cast<int32_t*>(dev(uint64_t(MT) * 4));
    ok = ok && d_tokens_;
    ok = ok && f32buf(wide_a_, uint64_t(MT) * hc * H) && f32buf(wide_b_, uint64_t(MT) * hc * H);
    for (uint32_t s = 0; s < kCpuAccRing; ++s) ok = ok && f32buf(cpu_acc_dev_[s], H);
    ok = ok && f32buf(post_, uint64_t(MT) * hc) && f32buf(comb_, uint64_t(MT) * hc * hc);
    ok = ok && f32buf(coll_, uint64_t(MT) * H) && f32buf(xn_, uint64_t(MT) * H);
    ok = ok && h16buf(x16_, uint64_t(MT) * H) && f32buf(mix_out_, uint64_t(MT) * H);
    ok = ok && f32buf(kda_f32a_, uint64_t(MT) * DI) && f32buf(kda_f32b_, uint64_t(MT) * DI) &&
               f32buf(kda_f32c_, uint64_t(MT) * DI);
    ok = ok && h16buf(kda_h16a_, uint64_t(MT) * DI) && h16buf(kda_h16b_, uint64_t(MT) * DI) &&
               h16buf(kda_h16c_, uint64_t(MT) * DI);
    ok = ok && f32buf(kda_pre_, uint64_t(MT) * DI) && f32buf(kda_g_, uint64_t(MT) * DI);
    ok = ok && h16buf(kda_lo16_, uint64_t(MT) * 128);
    ok = ok && f32buf(kda_beta_, uint64_t(MT) * cfg_.n_q_heads);
    ok = ok && h16buf(kda_z16_, uint64_t(MT) * DI);
    ok = ok && f32buf(kda_out_, uint64_t(MT) * DI) && h16buf(kda_y16_, uint64_t(MT) * DI);
    ok = ok && h16buf(mmscr16_, 16384) && f32buf(mmscr32_, uint64_t(MT) * 16384);
    ok = ok && h16buf(mla_qa16_, uint64_t(MT) * cfg_.q_lora_rank);
    ok = ok && f32buf(mla_qb_, uint64_t(MT) * cfg_.n_q_heads * cfg_.key_len_mla);
    ok = ok && f32buf(mla_qlat_, uint64_t(MT) * cfg_.n_q_heads * LAT);
    ok = ok && h16buf(mla_kv16_, uint64_t(MT) * LAT);
    ok = ok && f32buf(mla_att_, uint64_t(MT) * cfg_.n_q_heads * LAT);
    ok = ok && h16buf(mla_o16_, uint64_t(MT) * cfg_.n_q_heads * cfg_.value_len_mla);
    ok = ok && f32buf(moe_rl_, uint64_t(MT) * cfg_.n_experts);
    moe_top_ = static_cast<int32_t*>(dev(uint64_t(MT) * cfg_.n_experts_used * 4));
    moe_idx_ = static_cast<int32_t*>(dev(uint64_t(MT) * cfg_.n_experts_used * 4));
    ok = ok && moe_top_ && moe_idx_;
    ok = ok && f32buf(moe_topw_, uint64_t(MT) * cfg_.n_experts_used) &&
               f32buf(moe_w_, uint64_t(MT) * cfg_.n_experts_used);
    ok = ok && h16buf(moe_xg_, uint64_t(MT) * H);
    ok = ok && f32buf(moe_gu_[0], uint64_t(MT) * std::max(EF, cfg_.ffn)) &&
               f32buf(moe_gu_[1], uint64_t(MT) * std::max(EF, cfg_.ffn));
    ok = ok && h16buf(moe_h16_, uint64_t(MT) * std::max(EF, cfg_.ffn));
    ok = ok && f32buf(moe_dn_, uint64_t(MT) * H) && f32buf(moe_acc_, uint64_t(MT) * H);
    ok = ok && h16buf(emat_g16_, uint64_t(H) * EF) && h16buf(emat_u16_, uint64_t(H) * EF) &&
               h16buf(emat_d16_, uint64_t(EF) * H);
    ok = ok && f32buf(mean_, uint64_t(MT) * H);
    ok = ok && h16buf(logits_, cfg_.vocab);
    if (layer_hi_ == n_tf) ok = ok && h16buf(lgs16_, uint64_t(16) * cfg_.vocab);
    if (mtp_loaded()) {
        ok = ok && h16buf(mtp_lat_, uint64_t(max_ctx) * LAT);
        ok = ok && h16buf(mtp_e16_, H) && h16buf(mtp_cat_, 2 * H);
        ok = ok && f32buf(mtp_x_, H) && h16buf(mtp_x16_, H);
    }
    // KDA spec-decode snapshot: scan state (fp32) + conv taps (f16).
    ok = ok && f32buf(snap_dn_, uint64_t(n_lin) * dn_.state_elems_per_layer());
    snap_conv_ = static_cast<sycl::half*>(dev(uint64_t(n_lin) * dn_.conv_elems_per_layer() * 2));
    ok = ok && snap_conv_;
    // spec-verify input capture (commit_verify's re-scan source).
    ok = ok && f32buf(sv_q_, uint64_t(n_lin) * kSpecMax * DI) &&
               f32buf(sv_k_, uint64_t(n_lin) * kSpecMax * DI) &&
               f32buf(sv_v_, uint64_t(n_lin) * kSpecMax * DI) &&
               f32buf(sv_g_, uint64_t(n_lin) * kSpecMax * DI) &&
               f32buf(sv_beta_, uint64_t(n_lin) * kSpecMax * cfg_.n_q_heads);
    sv_ci_ = static_cast<sycl::half*>(dev(uint64_t(n_lin) * 3 * kSpecMax * DI * 2));
    ok = ok && sv_ci_;
    if (!ok) return "glm5next init_runtime: device alloc failed";

    // Per-layer LRU expert slot cache. Budget from IE_G5_ECACHE_MB (default
    // 10 GiB); slots split evenly across the MoE layers, floored at top_k so
    // one token's routing always fits, disabled below that.
    {
        uint64_t budget = 10240ull << 20;
        if (const char* v = std::getenv("IE_G5_ECACHE_MB")) budget = uint64_t(std::atoll(v)) << 20;
        if (server_memory_) budget = server_cache_bytes_;
        // EP decode: half the VRAM budget goes to this card's half-caches for
        // the PEER's layers (allocated later by ep_enable) — shrink ours now.
        if (std::getenv("IE_G5_EP_DECODE")) budget /= 2;
        // The stage that hosts the MTP kit also serves blk.45 from its ecache:
        // the tail by default ([layer_lo_, n_tf] is contiguous) or, with
        // IE_G5_MTP_STAGE=head, the first stage — then the range below spans
        // the OTHER stage's layers too, and ec_skip() keeps them out.
        const uint32_t eco_hi = mtp_loaded() ? n_tf + 1 : layer_hi_;
        auto ec_skip = [&](uint32_t L) {
            return cfg_.is_dense_layer(L) || (L >= layer_hi_ && L != n_tf);
        };
        uint32_t n_moe = 0;
        uint64_t slot_bytes_max = 0;
        std::vector<uint64_t> lslot(cfg_.n_layers, 0);
        for (uint32_t L = layer_lo_; L < eco_hi; ++L)
            if (!ec_skip(L)) {
                ++n_moe;
                const Glm5NextLayer& w = layers_[L];
                lslot[L] = w.gate_exps->nbytes / cfg_.n_experts +
                           w.up_exps->nbytes / cfg_.n_experts +
                           w.down_exps->nbytes / cfg_.n_experts;
                slot_bytes_max = std::max(slot_bytes_max, lslot[L]);
            }
        uint32_t slots = n_moe && slot_bytes_max ? uint32_t(budget / (uint64_t(n_moe) * slot_bytes_max)) : 0;
        slots = std::min(slots, cfg_.n_experts);
        if (slots >= cfg_.n_experts_used) {
            pin_sz_ = slot_bytes_max;
            for (uint32_t r = 0; r < kPinRing; ++r) {
                pin_ring_[r] = static_cast<uint8_t*>(
                    sycl::malloc_host(pin_sz_, alloc_->queue()));
                if (!pin_ring_[r]) return "glm5next init_runtime: pinned ring alloc failed";
            }
            cpool_.start(3);
            copyq_ = std::make_unique<sycl::queue>(
                alloc_->queue().get_context(), alloc_->queue().get_device(),
                sycl::property::queue::in_order{});
            jobsq_ = std::make_unique<sycl::queue>(
                alloc_->queue().get_context(), alloc_->queue().get_device(),
                sycl::property::queue::in_order{});
            // G1: passive OpenMP waiting for the q* CPU workers, set from inside
            // and from HERE (init runs on the main thread, one stage after the
            // other) rather than from CpuWorker::start, which a pipelined stage
            // may reach on a second host thread while the other stage's thread
            // is inside getenv — setenv/getenv are not safe concurrently.
            // libiomp5 reads its environment at first use and the workers'
            // first expert is the process's first OpenMP region (the only
            // pragma is in cpu_moe_gemv.cpp), so this lands in time; the
            // recipe measured 9.60 tok/s with it. A user's setting wins.
            setenv("OMP_WAIT_POLICY", "passive", 0);
            // G1 gate finding (2026-09-02 16:57, verified by the evaluator's
            // standalone probe): with libiomp5's default affinity handling the
            // SECOND stage's team is reset to the "full mask" the runtime
            // captured from the FIRST master (0-7 here), so pinning the worker
            // to 8-19 was silently undone — all 16 threads ran on the P-cores
            // while the log claimed otherwise. KMP_AFFINITY=disabled makes the
            // runtime leave thread masks alone; each team then inherits its
            // master's sched_setaffinity mask, which CpuWorker::start reads
            // back and reports after the first region.
            setenv("KMP_AFFINITY", "disabled", 0);
            fill_.start(this);
            {
                // The det2 divergence is CLOSED (2026-09-03): it was the VRAM page
                // alias on layer 28 (see the alias self-test below), not the lanes —
                // with the aliased slots quarantined, lanes 8 vs inline is byte-
                // identical over 2050 predictions (results/glm53-2026-09-03/perf/k6).
                // Still opt-in because with IE_G5_PIN_BANKS there is no host staging
                // for them to parallelize (44.5 vs 44.4 tok/s); they are the +26%
                // prefill lever on the mmap path only.
                uint32_t n_lanes = 0;
                if (const char* v = std::getenv("IE_G5_FILL_LANES")) n_lanes = uint32_t(std::max(0, std::atoi(v)));
                if (n_lanes) {
                    if (std::string e = lanes_.start(this, n_lanes, pin_sz_); !e.empty()) return "glm5next init_runtime: " + e;
                    std::fprintf(stderr, "[glm5next] fill lanes: %u workers over 2 MiB segments, %u x %.1f MiB pinned buffers\n",
                                 n_lanes, unsigned(FillLanes::kBufs), double(pin_sz_) / (1024.0 * 1024.0));
                }
            }
            if (std::getenv("IE_G5_PREFETCH")) {
                // Sized for ANY layer's slice, not just this stage's max —
                // peer-side prefetch stages PEER-stage slices (blk 11's
                // 17.56 MiB overran a local-sbm buffer: EFAULT'd preads).
                const uint64_t pf_sz = std::max<uint64_t>(pin_sz_, 20ull << 20);
                for (uint32_t r = 0; r < 2; ++r) {
                    pf_pin_[r] = static_cast<uint8_t*>(
                        sycl::malloc_host(pf_sz, alloc_->queue()));
                    if (!pf_pin_[r]) return "glm5next init_runtime: pf ring alloc failed";
                }
            }
            if (std::getenv("IE_G5_PP_TILES") || std::getenv("IE_G5_PP_STREAM")) {
                // 16 device slices × 24 KB. Each tiles batch H2Ds its slice
                // on jobsq_ (main-thread, not copyq_) WITHOUT waiting.
                // Compute kernels depend on that event.
                pp_jobs_ = static_cast<int32_t*>(dev(16ull * 2048ull * 3 * 4));
                if (!pp_jobs_) return "glm5next init_runtime: pp_jobs alloc failed";
                pp_hjobs_.reserve(2048 * 3);
            }
            if (std::getenv("IE_G5_PP_STREAM")) {
                // biggest layer bank (gate+up+down) across owned MoE layers
                uint64_t mx = 0;
                for (uint32_t L = layer_lo_; L < eco_hi; ++L)
                    if (!ec_skip(L))
                        mx = std::max(mx, layers_[L].gate_exps->nbytes +
                                          layers_[L].up_exps->nbytes +
                                          layers_[L].down_exps->nbytes);
                pp_stream_sz_ = mx;
                for (int b2 = 0; b2 < 2; ++b2) {
                    pp_stream_[b2] = static_cast<uint8_t*>(dev(mx));
                    if (!pp_stream_[b2])
                        return "glm5next init_runtime: pp_stream alloc failed "
                               "(shrink IE_G5_ECACHE_MB — stream needs 2 layer banks)";
                }
                const uint32_t npin = std::max(8u,
                    uint32_t((mx + kPpPinSz - 1) / kPpPinSz));
                pp_pin_.assign(npin, nullptr);
                pp_pin_ev_.assign(npin, {});
                for (uint32_t b2 = 0; b2 < npin; ++b2) {
                    pp_pin_[b2] = static_cast<uint8_t*>(
                        sycl::malloc_host(kPpPinSz, alloc_->queue()));
                    if (!pp_pin_[b2])
                        return "glm5next init_runtime: pp_pin alloc failed";
                }
                std::fprintf(stderr,
                             "[glm5next] pp-stream: 2 x %.2f GiB layer buffers, "
                             "%u x %.0f MiB pin\n",
                             double(mx) / 1073741824.0, npin,
                             double(kPpPinSz) / (1024.0 * 1024.0));
            }
            ecache_slots_ = slots;
            // Per-layer slot budget (IE_G5_SLOT_PROFILE = u64[n_layers x E]
            // pick counts, the IE_G5_EPROFILE_OUT dump): greedy allocation by
            // marginal hits per byte — the k-th slot at layer L is worth that
            // layer's k-th-hottest expert count. Flat-routing layers earn more
            // slots than skewed ones. Uniform split without a profile.
            std::vector<uint32_t> slots_l(cfg_.n_layers, slots);
            // IE_G5_UNIFORM_SLOTS=1: the pre-greedy uniform division (bisect
            // aid — 2026-08-29 divergence hunt: output text changed with slot
            // GEOMETRY, which correct caching never does).
            if (!std::getenv("IE_G5_UNIFORM_SLOTS")) {
                // Budget-exact balanced fill by ACTUAL per-layer slot bytes.
                // The uniform slot_bytes_MAX division strands ~17% of the
                // budget (Opus 5 report, 2026-08-29): stage A 72 -> 87 slots
                // at identical VRAM.
                uint64_t left = budget;
                for (uint32_t L = layer_lo_; L < eco_hi; ++L)
                    if (lslot[L]) { slots_l[L] = 0; }
                bool grew = true;
                // The MTP block (blk.n_tf) routes 8 experts per DRAFT step and is
                // not touched by prefill at all, yet the balanced fill gave it a
                // full equal share: with --pipedraft stage A dropped 78 -> 69
                // slots on every real layer and 100K prefill ran 15% slower
                // (2026-09-04 n8 vs j3). Cap it at the small-T floor instead;
                // IE_G5_MTP_SLOTS overrides.
                const uint32_t mtp_cap = [&] {
                    const char* v = std::getenv("IE_G5_MTP_SLOTS");
                    return v ? uint32_t(std::atoi(v)) : std::max<uint32_t>(32, cfg_.n_experts_used);
                }();
                while (grew) {
                    grew = false;
                    uint32_t pick = UINT32_MAX, mn = UINT32_MAX;
                    for (uint32_t L = layer_lo_; L < eco_hi; ++L) {
                        const uint32_t cap = (L == n_tf) ? std::min(mtp_cap, cfg_.n_experts) : cfg_.n_experts;
                        if (lslot[L] && slots_l[L] < cap &&
                            lslot[L] <= left && slots_l[L] < mn) { mn = slots_l[L]; pick = L; }
                    }
                    if (pick != UINT32_MAX) { left -= lslot[pick]; ++slots_l[pick]; grew = true; }
                }
                uint32_t mnv = UINT32_MAX, mxv = 0;
                for (uint32_t L = layer_lo_; L < eco_hi; ++L)
                    if (lslot[L]) { mnv = std::min(mnv, slots_l[L]); mxv = std::max(mxv, slots_l[L]); }
                if (mnv != UINT32_MAX && mnv < cfg_.n_experts_used)
                    return "glm5next init_runtime: ecache budget too small";
                slots = mnv == UINT32_MAX ? slots : mnv;   // report + eprio floor
            }
            if (const char* pf = std::getenv("IE_G5_SLOT_PROFILE")) {
                std::vector<uint64_t> counts(uint64_t(cfg_.n_layers) * cfg_.n_experts);
                FILE* pfd = std::fopen(pf, "rb");
                const bool ok_prof = pfd &&
                    std::fread(counts.data(), 8, counts.size(), pfd) == counts.size();
                if (pfd) std::fclose(pfd);
                // Floor 32, not top_k: a decode step activates ~8-32 experts
                // per layer, and a layer with fewer slots than its active set
                // degenerates to the full-drain fence path (measured: hit
                // 68->72% but decode 8.2 -> 7.0 with floor 8).
                const uint32_t fl = std::max<uint32_t>(32, cfg_.n_experts_used);
                uint64_t floor_cost = 0;
                std::vector<std::vector<uint64_t>> sorted(cfg_.n_layers);
                std::vector<uint64_t> lbytes(cfg_.n_layers, 0);
                if (ok_prof) {
                    for (uint32_t L = layer_lo_; L < eco_hi; ++L) {
                        if (ec_skip(L)) continue;
                        const Glm5NextLayer& w = layers_[L];
                        lbytes[L] = w.gate_exps->nbytes / cfg_.n_experts +
                                    w.up_exps->nbytes / cfg_.n_experts +
                                    w.down_exps->nbytes / cfg_.n_experts;
                        floor_cost += uint64_t(fl) * lbytes[L];
                        sorted[L].assign(counts.begin() + uint64_t(L) * cfg_.n_experts,
                                         counts.begin() + uint64_t(L + 1) * cfg_.n_experts);
                        std::sort(sorted[L].begin(), sorted[L].end(), std::greater<uint64_t>());
                    }
                }
                if (ok_prof && floor_cost <= budget) {
                    for (uint32_t L = layer_lo_; L < eco_hi; ++L)
                        if (!ec_skip(L)) slots_l[L] = fl;
                    uint64_t left = budget - floor_cost;
                    // (value, layer) max-heap; value = count/byte of the next slot
                    using Cand = std::pair<double, uint32_t>;
                    std::priority_queue<Cand> pq;
                    for (uint32_t L = layer_lo_; L < eco_hi; ++L)
                        if (!ec_skip(L) && fl < cfg_.n_experts)
                            pq.push({double(sorted[L][fl]) / double(lbytes[L]), L});
                    while (!pq.empty()) {
                        const uint32_t L = pq.top().second;
                        pq.pop();
                        if (lbytes[L] > left) continue;
                        left -= lbytes[L];
                        ++slots_l[L];
                        if (slots_l[L] < cfg_.n_experts)
                            pq.push({double(sorted[L][slots_l[L]]) / double(lbytes[L]), L});
                    }
                    uint32_t mn = UINT32_MAX, mx = 0;
                    for (uint32_t L = layer_lo_; L < eco_hi; ++L)
                        if (!ec_skip(L)) {
                            mn = std::min(mn, slots_l[L]);
                            mx = std::max(mx, slots_l[L]);
                        }
                    std::fprintf(stderr,
                                 "[glm5next] slot profile %s: per-layer slots %u..%u\n",
                                 pf, mn, mx);
                } else if (pf[0]) {
                    std::fprintf(stderr, "[glm5next] slot profile %s unreadable — uniform slots\n", pf);
                }
            }
            ecache_.resize(cfg_.n_layers);
            uint64_t ec_bytes = 0;
            // IE_G5_ECACHE_PAD=<MiB>: dead padding before each layer's cache
            // (2026-08-29 divergence probe — shifts every device address while
            // keeping all counts identical; a text flip under padding proves
            // ADDRESS-dependence, i.e. an OOB read or address-dependent rot).
            const uint64_t ec_pad = [] {
                const char* v = std::getenv("IE_G5_ECACHE_PAD");
                return v ? uint64_t(std::atoll(v)) << 20 : 0ull;
            }();
            for (uint32_t L = layer_lo_; L < eco_hi; ++L) {
                if (ec_skip(L)) continue;
                char ectag[32];
                std::snprintf(ectag, sizeof ectag, "ecache L%u", L);
                dev_alloc_tag(ectag);
                if (ec_pad) (void)dev(ec_pad);
                const Glm5NextLayer& w = layers_[L];
                ECache& ec = ecache_[L];
                const uint64_t gsl = w.gate_exps->nbytes / cfg_.n_experts;
                const uint64_t usl = w.up_exps->nbytes / cfg_.n_experts;
                const uint64_t dsl = w.down_exps->nbytes / cfg_.n_experts;
                ec.gate_off = 0; ec.up_off = gsl; ec.down_off = gsl + usl;
                ec.slot_bytes = gsl + usl + dsl;
                ec.base = static_cast<uint8_t*>(dev(uint64_t(slots_l[L]) * ec.slot_bytes));
                if (!ec.base) return "glm5next init_runtime: ecache alloc failed";
                ec.slot_of.assign(cfg_.n_experts, -1);
                ec.expert_in.assign(slots_l[L], -1);
                ec.last_use.assign(slots_l[L], 0);
                ec.slot_bad.assign(slots_l[L], 0);
                if (std::getenv("IE_G5_PREFETCH")) {   // empty == prefetch off
                    ec.pf_ev.assign(slots_l[L], {});
                    ec.pf_pending.assign(slots_l[L], 0);
                }
                ec_bytes += uint64_t(slots_l[L]) * ec.slot_bytes;
            }
            expert_cache_bytes_ = ec_bytes;
            std::fprintf(stderr, "[glm5next] expert cache: %u slots/layer, %.2f GiB\n",
                         slots, double(ec_bytes) / 1073741824.0);
            // ---- VRAM 2 MiB page-alias self-test (IE_G5_NO_ALIAS_TEST=1 skips) ----
            // 2026-09-03, card 1: two virtual 2 MiB pages of ONE 1044 MiB
            // malloc_device allocation (layer 28's expert cache, +0 and
            // +1069547520) read back each other's bytes. Every fill into one slot
            // silently rewrote what the other slot reads, so 181 of 49125 expert
            // reads in the PPL ruler took a different expert's Q4_K blocks, whose
            // scales overflow fp16 in the GEMV and were clamped to +-65504 — the
            // "layer 28 fp16 overflow" of 2026-09-01. It is stable across runs and
            // moves with the allocation layout, so the engine cannot assume it
            // away: test the mapping and refuse to use what fails.
            //
            // What the test has to do, and why each part is there:
            //  * TWO PASSES over the WHOLE set of caches. A page written and read
            //    straight back cannot reveal an alias — its partner has not been
            //    written yet. Tags are unique across all of the stage's caches, so
            //    a cross-allocation alias is caught as well.
            //  * The bulk pattern is written by a KERNEL, over every word. A small
            //    probe is useless: 8-byte tags total ~90 kB, stay in L2 and read
            //    back from cache. Writing all 22 GiB spills it, and costs one
            //    round trip of VRAM bandwidth (~0.2 s).
            //  * Word 0 of each page is then re-written by the COPY ENGINE, and
            //    the verify kernel reads with the EUs. That is exactly the pair
            //    the fill (copyq_ H2D) and the GEMV (EU load) use, so a mapping
            //    that differs BETWEEN engines is caught too, not just a shared
            //    physical page.
            if (!std::getenv("IE_G5_NO_ALIAS_TEST")) {
                sycl::queue& qq = alloc_->queue();
                constexpr uint64_t PG = 2ull << 20;
                constexpr uint64_t WPP = PG / 8;            // words per 2 MiB page
                constexpr uint64_t kCeMark = 0xFFFFFF;      // word 0's copy-engine tag
                constexpr uint32_t kMaxRep = 64;
                struct Alloc { uint32_t L; uint64_t words, tag0, npages; };
                std::vector<Alloc> als;
                uint64_t tag_base = 1;
                for (uint32_t L2 = layer_lo_; L2 < eco_hi; ++L2) {
                    if (ec_skip(L2)) continue;
                    const uint64_t nw = uint64_t(ecache_[L2].expert_in.size()) *
                                        ecache_[L2].slot_bytes / 8;
                    const uint64_t np = (nw + WPP - 1) / WPP;
                    als.push_back({L2, nw, tag_base, np});
                    tag_base += np;
                }
                uint32_t* rep  = static_cast<uint32_t*>(dev(4));
                uint64_t* repv = static_cast<uint64_t*>(dev(uint64_t(kMaxRep) * 2 * 8));
                if (rep && repv) {
                    const auto at0 = std::chrono::steady_clock::now();
                    qq.memset(rep, 0, 4).wait();
                    for (const Alloc& a : als) {          // pass 1: tag every word
                        uint64_t* p = reinterpret_cast<uint64_t*>(ecache_[a.L].base);
                        const uint64_t t0 = a.tag0;
                        qq.parallel_for<class G5AliasTagK>(sycl::range<1>(a.words), [=](sycl::id<1> id) {
                            const uint64_t i = id[0];
                            p[i] = ((t0 + i / WPP) << 24) | (i % WPP);
                        });
                    }
                    qq.wait();
                    std::vector<uint64_t> ce(tag_base);   // pass 1b: copy engine, word 0
                    for (const Alloc& a : als) {
                        uint8_t* b = ecache_[a.L].base;
                        for (uint64_t k = 0; k < a.npages; ++k) {
                            ce[a.tag0 + k - 1] = ((a.tag0 + k) << 24) | kCeMark;
                            copyq_->memcpy(b + k * PG, &ce[a.tag0 + k - 1], 8);
                        }
                    }
                    copyq_->wait();
                    for (const Alloc& a : als) {          // pass 2: verify with the EUs
                        const uint64_t* p = reinterpret_cast<const uint64_t*>(ecache_[a.L].base);
                        const uint64_t t0 = a.tag0;
                        uint32_t* rp = rep; uint64_t* rv = repv;
                        qq.parallel_for<class G5AliasChkK>(sycl::range<1>(a.words), [=](sycl::id<1> id) {
                            const uint64_t i = id[0];
                            const uint64_t w = i % WPP;
                            const uint64_t want = ((t0 + i / WPP) << 24) | (w ? w : kCeMark);
                            const uint64_t got = p[i];
                            if (got == want) return;
                            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed,
                                             sycl::memory_scope::device> c(*rp);
                            const uint32_t k = c.fetch_add(1u);
                            if (k < kMaxRep) { rv[2 * k] = t0 + i / WPP; rv[2 * k + 1] = got; }
                        });
                    }
                    qq.wait();
                    uint32_t nrep = 0;
                    std::vector<uint64_t> hv2(uint64_t(kMaxRep) * 2, 0);
                    qq.memcpy(&nrep, rep, 4).wait();
                    qq.memcpy(hv2.data(), repv, uint64_t(kMaxRep) * 2 * 8).wait();
                    auto where = [&](uint64_t t, uint32_t& L3, uint32_t& sl, uint64_t& off) {
                        for (const Alloc& a : als)
                            if (t >= a.tag0 && t < a.tag0 + a.npages) {
                                L3 = a.L; off = (t - a.tag0) * PG;
                                sl = uint32_t(off / ecache_[a.L].slot_bytes);
                                return true;
                            }
                        return false;
                    };
                    std::vector<uint64_t> seen;
                    for (uint32_t k = 0; k < std::min<uint32_t>(nrep, kMaxRep); ++k) {
                        const uint64_t t = hv2[2 * k], got = hv2[2 * k + 1];
                        if (std::find(seen.begin(), seen.end(), t) != seen.end()) continue;
                        seen.push_back(t);
                        uint32_t L3 = 0, sl = 0; uint64_t off = 0;
                        if (!where(t, L3, sl, off)) continue;
                        ecache_[L3].slot_bad[sl] = 1;
                        uint32_t L4 = 0, sl4 = 0; uint64_t off4 = 0;
                        if (where(got >> 24, L4, sl4, off4)) {
                            ecache_[L4].slot_bad[sl4] = 1;
                            std::fprintf(stderr,
                                         "[glm5next] VRAM PAGE ALIAS: L%u slot %u (cache +%llu) reads "
                                         "what was written to L%u slot %u (cache +%llu) — both slots "
                                         "quarantined\n", L3, sl, (unsigned long long)off,
                                         L4, sl4, (unsigned long long)off4);
                        } else {
                            std::fprintf(stderr,
                                         "[glm5next] VRAM PAGE FAULTY: L%u slot %u (cache +%llu) read "
                                         "back 0x%llx — slot quarantined\n", L3, sl,
                                         (unsigned long long)off, (unsigned long long)got);
                        }
                    }
                    uint32_t quarantined = 0;
                    for (uint32_t L2 = layer_lo_; L2 < eco_hi; ++L2)
                        if (!ec_skip(L2))
                            for (uint8_t b2 : ecache_[L2].slot_bad) quarantined += b2;
                    std::fprintf(stderr,
                                 "[glm5next] vram alias test: %.2f GiB in %.2f s, %u bad words, "
                                 "%zu bad pages, %u slots quarantined\n",
                                 double(ec_bytes) / 1073741824.0,
                                 std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - at0).count(),
                                 nrep, seen.size(), quarantined);
                }
            }

            // IE_G5_ECACHE_MEMTEST=1: pattern-test every expert-cache slot with
            // the model out of the way. Write a PRNG pattern H2D on copyq_ (the
            // fill path's own queue), then read it back TWICE — once on copyq_,
            // once on q (the compute path's queue). 2026-09-03 det8: two fixed
            // 2 MiB pages of layer 28's cache hand the GEMV high-entropy
            // non-model bytes on 181 of 49125 reads. Both readbacks wrong = the
            // write never landed; copyq_ right and q wrong = a visibility fault.
            if (std::getenv("IE_G5_ECACHE_MEMTEST")) {
                // TWO PHASES ON PURPOSE. Writing a slot and reading it straight
                // back cannot see an ALIAS (two virtual pages of the allocation
                // backed by one physical page) — the second write has not
                // happened yet. So: write EVERY page of EVERY cache first, then
                // read them all back. A page that comes back holding another
                // page's pattern names the alias and its partner.
                // 2026-09-03 det8: layer 28's cache hands the GEMV genuine model
                // bytes that belong 5 MiB into a DIFFERENT slot, on 181 of 49125
                // reads, at two fixed 2 MiB-aligned addresses.
                sycl::queue& qq = alloc_->queue();
                const uint64_t PG = 2ull << 20;
                struct Pg { uint32_t L, slot; uint64_t off; uint8_t* addr; uint64_t tag; };
                std::vector<Pg> pages;
                for (uint32_t L = layer_lo_; L < eco_hi; ++L) {
                    if (ec_skip(L)) continue;
                    ECache& ec = ecache_[L];
                    const uint64_t tot = uint64_t(ec.expert_in.size()) * ec.slot_bytes;
                    for (uint64_t off = 0; off < tot; off += PG)
                        pages.push_back({L, uint32_t(off / ec.slot_bytes), off,
                                         ec.base + off, uint64_t(pages.size()) + 1});
                }
                std::vector<uint8_t> buf(PG);
                auto fill_pat = [&](uint64_t tag, uint64_t len) {
                    uint64_t st = 0x9E3779B97F4A7C15ull ^ (tag * 0xD1B54A32D192ED03ull);
                    for (uint64_t i2 = 0; i2 + 8 <= len; i2 += 8) {
                        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
                        std::memcpy(buf.data() + i2, &st, 8);
                    }
                };
                std::fprintf(stderr, "[ecache-memtest] phase 1: writing %zu pages of 2 MiB\n",
                             pages.size());
                for (const Pg& pg : pages) {
                    const uint64_t len = std::min(PG, uint64_t(ecache_[pg.L].slot_bytes) *
                                                     ecache_[pg.L].expert_in.size() - pg.off);
                    fill_pat(pg.tag, len);
                    copyq_->memcpy(pg.addr, buf.data(), len).wait();
                }
                std::fprintf(stderr, "[ecache-memtest] phase 2: reading them back\n");
                std::vector<uint8_t> got(PG);
                uint64_t bad = 0;
                for (const Pg& pg : pages) {
                    const uint64_t len = std::min(PG, uint64_t(ecache_[pg.L].slot_bytes) *
                                                     ecache_[pg.L].expert_in.size() - pg.off);
                    qq.memcpy(got.data(), pg.addr, len).wait();
                    fill_pat(pg.tag, len);
                    if (std::memcmp(buf.data(), got.data(), len) == 0) continue;
                    ++bad;
                    // whose pattern came back? try every page's tag
                    uint64_t who = 0;
                    for (const Pg& o : pages) {
                        fill_pat(o.tag, 4096);
                        if (std::memcmp(buf.data(), got.data(), 4096) == 0) { who = o.tag; break; }
                    }
                    const Pg* w2 = nullptr;
                    for (const Pg& o : pages) if (o.tag == who) { w2 = &o; break; }
                    std::fprintf(stderr,
                                 "[ecache-memtest] ALIAS: L%u slot %u page %p (cache +%llu) "
                                 "came back holding %s\n",
                                 pg.L, pg.slot, (void*)pg.addr, (unsigned long long)pg.off,
                                 w2 ? "" : "an unknown pattern");
                    if (w2)
                        std::fprintf(stderr,
                                     "[ecache-memtest]        the pattern written to L%u slot %u "
                                     "page %p (cache +%llu, slot +%llu)\n",
                                     w2->L, w2->slot, (void*)w2->addr,
                                     (unsigned long long)w2->off,
                                     (unsigned long long)(w2->off - uint64_t(w2->slot) *
                                                          ecache_[w2->L].slot_bytes));
                }
                std::fprintf(stderr, "[ecache-memtest] %zu pages, %llu ALIASED/WRONG\n",
                             pages.size(), (unsigned long long)bad);
                if (layer_hi_ >= cfg_.n_layers || layer_lo_ > 0)
                    return "glm5next init_runtime: ecache memtest complete (diagnostic exit)";
            }
            dev_alloc_tag("post-ecache");
            {
                char at[64];
                std::snprintf(at, sizeof at, "stage layers [%u, %u)", layer_lo_, layer_hi_);
                dev_alloc_audit(at);
            }
            // Profile-guided residency (IE_G5_EPRIO=<counts file>): preload
            // each owned MoE layer's hottest experts into its slots and make
            // the top fraction STICKY (last_use = UINT64_MAX — the LRU victim
            // scan can never pick them). Measured 2026-08-28: per-layer
            // top-55 covers 74.8% of picks vs LRU's achieved 62.6%.
            if (const char* pf = std::getenv("IE_G5_EPRIO")) {
                std::vector<uint64_t> counts(uint64_t(cfg_.n_layers) * cfg_.n_experts);
                FILE* f = std::fopen(pf, "rb");
                if (f && std::fread(counts.data(), 8, counts.size(), f) == counts.size()) {
                    uint32_t sticky_pct = 60;
                    if (const char* sp = std::getenv("IE_G5_STICKY_PCT"))
                        sticky_pct = uint32_t(std::atoi(sp));
                    uint32_t n_sticky = 0;
                    sycl::queue& q = alloc_->queue();
                    for (uint32_t L = layer_lo_; L < eco_hi; ++L) {
                        if (ec_skip(L)) continue;
                        ECache& ec = ecache_[L];
                        const uint32_t lsl = uint32_t(ec.expert_in.size());
                        n_sticky = lsl * sticky_pct / 100;
                        const Glm5NextLayer& w = layers_[L];
                        std::vector<uint32_t> order(cfg_.n_experts);
                        for (uint32_t e = 0; e < cfg_.n_experts; ++e) order[e] = e;
                        const uint64_t* row = counts.data() + uint64_t(L) * cfg_.n_experts;
                        std::sort(order.begin(), order.end(),
                                  [&](uint32_t a, uint32_t b) { return row[a] > row[b]; });
                        const uint64_t gsl = w.gate_exps->nbytes / cfg_.n_experts;
                        const uint64_t usl = w.up_exps->nbytes / cfg_.n_experts;
                        const uint64_t dsl = w.down_exps->nbytes / cfg_.n_experts;
                        for (uint32_t s = 0; s < lsl; ++s) {
                            if (!ec.slot_bad.empty() && ec.slot_bad[s]) continue;
                            const uint32_t e = order[s];
                            uint8_t* dst = ec.base + uint64_t(s) * ec.slot_bytes;
                            q.memcpy(dst + ec.gate_off,
                                     static_cast<const uint8_t*>(w.gate_exps->data) + uint64_t(e) * gsl, gsl);
                            q.memcpy(dst + ec.up_off,
                                     static_cast<const uint8_t*>(w.up_exps->data) + uint64_t(e) * usl, usl);
                            q.memcpy(dst + ec.down_off,
                                     static_cast<const uint8_t*>(w.down_exps->data) + uint64_t(e) * dsl, dsl);
                            ec.expert_in[s] = int32_t(e);
                            ec.slot_of[e] = int16_t(s);
                            ec.last_use[s] = s < n_sticky ? UINT64_MAX : ++ecache_clock_;
                        }
                        q.wait();
                    }
                    std::fprintf(stderr, "[glm5next] eprio preload: %u/%u sticky per layer (%s)\n",
                                 n_sticky, slots, pf);
                }
                if (f) std::fclose(f);
            }
        }
    }

    h_top_.resize(uint64_t(MT) * cfg_.n_experts_used);
    h_topw_.resize(uint64_t(MT) * cfg_.n_experts_used);
    h_idx_.resize(uint64_t(MT) * cfg_.n_experts_used);
    h_w_.resize(uint64_t(MT) * cfg_.n_experts_used);
    reset_state();
    return {};
}

void Glm5NextModel::reset_state() {
    if (!alloc_) return;
    sycl::queue& q = alloc_->queue();
    dn_.reset(q);
    kv_len_ = 0;
    q.wait();
}

void Glm5NextModel::mm(const sycl::half* A, const Glm5DW& W, uint32_t T,
                       uint32_t N, uint32_t K, sycl::half* y16, float* y32) {
    sycl::queue& q = alloc_->queue();
    if (W.qs) {   // Q8_0-SoA residency: W8A16 GEMV, T-rows variant for T>1
        sycl::half* y = y16 ? y16 : (T == 1 ? mmscr16_
                                            : reinterpret_cast<sycl::half*>(mmscr32_));
        if (T == 1) gemv_q8_0_soa_f16_g(q, A, W.qs, W.d, y, K, N, {});
        else        gemv_q8_0_soa_f16_rows(q, A, W.qs, W.d, y, K, N, T, {});
        if (y32) cast_fp16_to_fp32(q, y, y32, uint64_t(T) * N, {});
        return;
    }
    if (T == 1) {
        sycl::half* y = y16 ? y16 : mmscr16_;
        gemv_fp16(q, A, W.w, y, K, N, {});
        if (y32) cast_fp16_to_fp32(q, y, y32, N, {});
    } else {
        float* c = y32 ? y32 : mmscr32_;
        gemm_fp16(q, A, W.w, c, T, N, K, {});
        if (y16) cast_fp32_to_fp16(q, c, y16, uint64_t(T) * N, {});
    }
}

void Glm5NextModel::mm_dual(const sycl::half* A, const Glm5DW& Wa, const Glm5DW& Wb,
                            uint32_t T, uint32_t N, uint32_t K,
                            sycl::half* ya16, float* ya32,
                            sycl::half* yb16, float* yb32) {
    if (!(Wa.qs && Wb.qs)) {
        mm(A, Wa, T, N, K, ya16, ya32);
        mm(A, Wb, T, N, K, yb16, yb32);
        return;
    }
    sycl::queue& q = alloc_->queue();
    // moe_h16_ is free during shexp gate+up; KDA dual writes the caller's y16.
    sycl::half* ya = ya16 ? ya16 : moe_h16_;
    sycl::half* yb = yb16 ? yb16 : moe_h16_ + uint64_t(T) * N;
    gemv_q8_0_soa_f16_rows_dual(q, A, Wa.qs, Wa.d, ya, Wb.qs, Wb.d, yb, K, N, T, {});
    if (ya32) cast_fp16_to_fp32(q, ya, ya32, uint64_t(T) * N, {});
    if (yb32) cast_fp16_to_fp32(q, yb, yb32, uint64_t(T) * N, {});
}

std::string Glm5NextModel::run_moe(uint32_t L, uint32_t T) {
    sycl::queue& q = alloc_->queue();
    const Glm5NextLayer& w = layers_[L];
    const uint32_t H = cfg_.hidden, E = cfg_.n_experts, TK = cfg_.n_experts_used;
    const uint32_t EF = cfg_.expert_ffn;
    const float clamp = cfg_.swiglu_clamp_exp[L];

    // Router (fp32 chain) + selection. The D2H wait is EVENT-scoped: the
    // shared-expert chain is submitted after it, so the GPU chews on shexp
    // while the host does the per-expert grouping below (no full drain).
    k_router_logits(q, xn_, w.router, moe_rl_, T, H, E, {});
    k_router_topk(q, moe_rl_, w.probs_bias, moe_top_, moe_topw_, E, TK,
                  cfg_.expert_weights_scale, T, {});
    q.memcpy(h_top_.data(), moe_top_, uint64_t(T) * TK * 4);
    auto e_d2h = q.memcpy(h_topw_.data(), moe_topw_, uint64_t(T) * TK * 4);
    q.memset(moe_acc_, 0, uint64_t(T) * H * 4);

    // Shared expert (unscaled; added into the accumulator first).
    {
        mm_dual(x16_, w.shexp_gate, w.shexp_up, T, EF, H,
                nullptr, moe_gu_[0], nullptr, moe_gu_[1]);
        ds4_swiglu_clamped(q, moe_gu_[0], moe_gu_[1], moe_gu_[0],
                           uint64_t(T) * EF, cfg_.swiglu_clamp_shexp[L], {});
        cast_fp32_to_fp16(q, moe_gu_[0], moe_h16_, uint64_t(T) * EF, {});
        mm(moe_h16_, w.shexp_down, T, H, EF, nullptr, moe_dn_);
        // identity scatter: token i adds its own row at weight 1
        ie::ps(q, "g5_shexp_add", [&](sycl::handler& h) {
            const float* dn = moe_dn_; float* acc = moe_acc_;
            h.parallel_for(sycl::range<2>(T, H), [=](sycl::id<2> id) {
                acc[uint64_t(id[0]) * H + id[1]] += dn[uint64_t(id[0]) * H + id[1]];
            });
        });
    }
    {
        const auto rw0 = std::chrono::steady_clock::now();
        e_d2h.wait();   // routing landed; shexp may still be in flight
        t_route_wait += std::chrono::duration<double>(std::chrono::steady_clock::now() - rw0).count();
    }

    // R9 prediction SUBMIT (consumed at end-of-call): the next MoE layer's
    // router on THIS layer's xn_, D2H into pinned staging with no wait — the
    // in-order queue completes it well before the waves finish, so the
    // end-of-call collect is free (the old per-layer .wait() was a full
    // queue drain x42/token). moe_rl_/moe_top_/moe_topw_ reuse is safe:
    // their real contents landed in h_top_/h_topw_ above.
    static const bool shadow_cnt = std::getenv("IE_G5_SHADOW_COUNT") != nullptr;
    static const int pf_depth = [] {
        const char* v = std::getenv("IE_G5_PREFETCH");
        if (!v) return 0;
        const int d = std::atoi(v);
        return d >= 1 ? d : 1;
    }();
    const bool pf_on = pf_depth >= 1;
    uint32_t pred_ln = UINT32_MAX, pred_ln2 = UINT32_MAX;
    sycl::event pred_ev, pred2_ev;
    // Rows to predict from. Spec-verify (T=2) touches up to 2*top_k distinct
    // experts per layer, so prefetching there RAISES the hit rate — measured
    // 69.0 -> 86.2% (9.9k issued, 64% consumed). It is still a LOSS: decode
    // 3.74 tok/s vs the 3.95-5.44 band with it off, twice, because decode is
    // bandwidth-bound and the 36% unconsumed prefetches are pure wasted PCIe
    // traffic (~0.5 GB/token) on the critical path. Hit rate is not the
    // objective function; bytes moved is. Opt in with IE_G5_PF_SPEC=1.
    static const bool pf_spec = std::getenv("IE_G5_PF_SPEC") != nullptr;
    const uint32_t pf_rows = (T == 1) ? 1 : (pf_spec && T <= kSpecMax ? T : 0);
    if ((shadow_cnt || pf_on) && pf_rows) {
        uint32_t Ln = L + 1;
        while (Ln < layer_hi_ && cfg_.is_dense_layer(Ln)) ++Ln;
        if (Ln < layer_hi_) {
            if (!pf_pred_pin_)   // rows 0..kSpecMax-1 = depth 1, row kSpecMax = depth 2
                pf_pred_pin_ = static_cast<int32_t*>(
                    sycl::malloc_host(uint64_t(kSpecMax + 1) * TK * 4, q));
            k_router_logits(q, xn_, layers_[Ln].router, moe_rl_, pf_rows, H, E, {});
            k_router_topk(q, moe_rl_, layers_[Ln].probs_bias, moe_top_, moe_topw_,
                          E, TK, cfg_.expert_weights_scale, pf_rows, {});
            pred_ev = q.memcpy(pf_pred_pin_, moe_top_,
                               uint64_t(pf_rows) * TK * 4);
            pred_ln = Ln;
            // depth-2 proxy: L+2's router on the same xn_ (63.2% measured).
            if (shadow_cnt || pf_depth >= 2) {
                uint32_t Ln2 = Ln + 1;
                while (Ln2 < layer_hi_ && cfg_.is_dense_layer(Ln2)) ++Ln2;
                if (Ln2 < layer_hi_) {
                    k_router_logits(q, xn_, layers_[Ln2].router, moe_rl_, 1, H, E, {});
                    k_router_topk(q, moe_rl_, layers_[Ln2].probs_bias, moe_top_,
                                  moe_topw_, E, TK, cfg_.expert_weights_scale, 1, {});
                    pred2_ev = q.memcpy(pf_pred_pin_ + uint64_t(kSpecMax) * TK,
                                        moe_top_, uint64_t(TK) * 4);
                    pred_ln2 = Ln2;
                }
            }
        }
    }

    // Routed experts, grouped by expert id (ascending — deterministic order).
    // The per-expert gather/weight lists are packed into the STABLE flat host
    // arrays h_idx_/h_w_ and uploaded once, so nothing in the loop below
    // waits: the in-order queue serializes every slot fill, dequant, gemm and
    // scatter without a single host sync.
    std::vector<std::vector<int32_t>> picks(E);
    if (expert_counts.empty()) expert_counts.assign(uint64_t(cfg_.n_layers) * E, 0);
    if (miss_counts.empty()) miss_counts.assign(uint64_t(cfg_.n_layers) * E, 0);
    for (uint32_t t = 0; t < T; ++t)
        for (uint32_t k = 0; k < TK; ++k) {
            const uint32_t e = uint32_t(h_top_[uint64_t(t) * TK + k]);
            picks[e].push_back(int32_t(t));
            ++expert_counts[uint64_t(L) * E + e];
        }
    uint32_t off = 0;
    std::vector<uint32_t> e_off(E, 0);
    for (uint32_t e = 0; e < E; ++e) {
        e_off[e] = off;
        for (int32_t t : picks[e]) {
            h_idx_[off] = t;
            float wv = 0.f;
            for (uint32_t k = 0; k < TK; ++k)
                if (h_top_[uint64_t(t) * TK + k] == int32_t(e))
                    wv = h_topw_[uint64_t(t) * TK + k];
            h_w_[off] = wv;
            ++off;
        }
    }
    q.memcpy(moe_idx_, h_idx_.data(), uint64_t(off) * 4);
    q.memcpy(moe_w_, h_w_.data(), uint64_t(off) * 4);

    // ---- expert-parallel decode: the peer card computes the other parity
    // half of this layer out of ITS half-cache, filling misses on ITS PCIe
    // link concurrently with ours. Partial lands in ep_ret_ and is added
    // once after the local experts (deterministic order).
    sycl::event ep_ret_ev;
    bool ep_used = false;
    const bool epv = std::getenv("IE_G5_EP_VERIFY") != nullptr;
    std::vector<std::vector<int32_t>> saved_picks(epv ? E : 0);
    if (ep_peer_ && T <= 4 &&
        L < ep_peer_->ep_cache_.size() && ep_peer_->ep_cache_[L].base) {
        std::vector<uint32_t> pe, pc;
        std::vector<int32_t> pidx;
        std::vector<float> pw;
        for (uint32_t e = 0; e < E; ++e) {
            if (picks[e].empty() || (e & 1u) == ep_parity_) continue;
            if (epv) saved_picks[e] = picks[e];
            pe.push_back(e);
            pc.push_back(uint32_t(picks[e].size()));
            for (size_t j = 0; j < picks[e].size(); ++j) {
                pidx.push_back(picks[e][j]);
                pw.push_back(h_w_[e_off[e] + j]);
            }
            picks[e].clear();   // the local machinery below skips them
        }
        if (!pe.empty()) {
            // IE_G5_EP_PUSH: P2P D2D WRITES are certified on this stack (only
            // reads corrupt) — push x straight into the peer's ep_x16_ and
            // have the peer push its partial straight into our ep_ret_.
            // 4 PCIe hops/layer (D2H + wait + H2D each way) become 2 direct
            // pushes; each side waits only its OWN events (trusted pattern).
            static const bool ep_push = std::getenv("IE_G5_EP_PUSH") != nullptr;
            sycl::event xev;
            if (ep_push) {
                xev = q.memcpy(ep_peer_->ep_x16_, x16_, uint64_t(T) * H * 2);
                xev.wait();   // OUR event: trusted; x is in the peer's VRAM
            } else {
                xev = q.memcpy(ep_xstage_, x16_, uint64_t(T) * H * 2);  // D2H (our link)
                xev.wait();   // cross-device event deps are NOT trusted (2026-08-29
                              // oracle: stale-x corruption on stage-B peer layers)
            }
            // copy the lists into stable owner-side staging (the worker reads
            // them after this scope dies), then hand off to the PEER's worker
            // thread — the peer chain runs concurrently with our local half.
            ep_oexps_.assign(pe.begin(), pe.end());
            ep_oecnt_.assign(pc.begin(), pc.end());
            ep_oidx_.assign(pidx.begin(), pidx.end());
            ep_ow_.assign(pw.begin(), pw.end());
            EpReq r;
            r.w = &w; r.L = L; r.T = T;
            r.n_exp = uint32_t(pe.size()); r.npick = uint32_t(pidx.size());
            r.exps = ep_oexps_.data(); r.ecnt = ep_oecnt_.data();
            r.idx_flat = ep_oidx_.data(); r.w_flat = ep_ow_.data();
            r.x_host = ep_xstage_; r.x_ev = xev; r.ret_dep = ep_add_ev_;
            r.ret_host = ep_rstage_;
            r.x_pushed = ep_push;
            r.ret_dev = ep_push ? ep_ret_ : nullptr;
            ep_peer_->ep_thread_.start(ep_peer_);
            ep_ret_fut_ = ep_peer_->ep_thread_.push(std::move(r));
            ep_used = true;
        }
    }

    ECache* ec = (L < ecache_.size() && ecache_[L].base) ? &ecache_[L] : nullptr;
    if (!ec) return "glm5next moe: expert cache disabled (IE_G5_ECACHE_MB too small)";
    float* acc_dst = moe_acc_;   // IE_G5_EP_VERIFY redirects compute_e here
    const uint64_t gsl = w.gate_exps->nbytes / E;
    const uint64_t usl = w.up_exps->nbytes / E;
    const uint64_t dsl = w.down_exps->nbytes / E;

    // IE_G5_PP_STREAM (T>4): whole-layer bank double-buffer replaces the
    // per-miss cache stream — 3 big sequential H2Ds per layer, the NEXT MoE
    // layer's loading behind THIS layer's compute. Buffer reuse is fenced
    // behind all submitted compute (the old occupant's readers).
    static const bool pp_stream_env = std::getenv("IE_G5_PP_STREAM") != nullptr;
    const bool pp_stream = pp_stream_env && pp_stream_[0] && T > 4;
    const auto pps_call_t0 = std::chrono::steady_clock::now();
    const uint8_t *str_g = nullptr, *str_u = nullptr, *str_d = nullptr;
    sycl::event stream_dep;
    if (pp_stream) {
        auto load_layer = [&](uint32_t Ls, int buf) {
            const Glm5NextLayer& wl = layers_[Ls];
            FillReq r;
            r.stream = true;
            r.dst = pp_stream_[buf];
            r.g  = wl.gate_src; r.gsl = wl.gate_exps->nbytes;
            r.gfd = wl.gate_fd; r.goff = wl.gate_foff;
            r.u  = wl.up_src;   r.usl = wl.up_exps->nbytes;
            r.ufd = wl.up_fd;   r.uoff = wl.up_foff;
            r.d2 = wl.down_src; r.dsl = wl.down_exps->nbytes;
            r.dfd = wl.down_fd; r.doff = wl.down_foff;
            r.fence = pp_stream_free_[buf];
            pp_stream_fut_[buf] = fill_.push(std::move(r));
            pp_stream_layer_[buf] = int32_t(Ls);
        };
        const int b = int(L & 1);
        if (pp_stream_layer_[b] != int32_t(L)) load_layer(L, b);
        // Queue the other buffer's next layer BEFORE get() so the fill
        // worker does not idle in the get() bubble (stamp: 18 s ring_wait).
        uint32_t Ln2 = L + 1;
        while (Ln2 < layer_hi_ && cfg_.is_dense_layer(Ln2)) ++Ln2;
        if (Ln2 < layer_hi_ && int(Ln2 & 1) != b &&
            pp_stream_layer_[Ln2 & 1] != int32_t(Ln2))
            load_layer(Ln2, int(Ln2 & 1));
        static const bool pps_sync = std::getenv("IE_G5_PPS_SYNC") != nullptr;
        auto tg0 = std::chrono::steady_clock::now();
        stream_dep = pp_stream_fut_[b].get();
        auto tg1 = std::chrono::steady_clock::now();
        t_pps_get += std::chrono::duration<double>(tg1 - tg0).count();
        if (pps_sync) {   // probe: block until the layer is ON DEVICE
            stream_dep.wait();
            t_pps_wait += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - tg1).count();
        }
        // ONE barrier orders ALL of this layer's compute behind the bank
        // load (in-order q) — attaching stream_dep to every expert's first
        // reader (~263 event refs/layer) is redundant dep-graph weight.
        q.ext_oneapi_submit_barrier({stream_dep});
        str_g = pp_stream_[b];
        str_u = str_g + w.gate_exps->nbytes;
        str_d = str_u + w.up_exps->nbytes;
    }

    // Idempotent miss fill: pooled CPU copy into the pinned ring, H2D on the
    // TRANSFER queue (overlaps compute). The fill depends on a compute-fence
    // barrier (the victim slot may still be read by in-flight kernels); the
    // per-expert fill event feeds pass 2's first slot-reading kernel.
    std::vector<std::shared_future<sycl::event>> fevf(E);
    std::vector<uint8_t> filled(E, 0);
    // `pending[e]` = e is routed this call and not yet computed. The victim
    // scan avoids pending slots (Belady-flavored: their next use is CLOSER
    // than anything else's) — without this, overflow layers evict the later
    // waves' cached experts and inter-chunk reuse collapses to zero.
    std::vector<uint8_t> pending(E, 0);
    std::vector<int32_t> act_pos(E, -1);   // e -> index in this call's schedule
    std::string fill_err;                  // pread failure -> abort (no garbage DMA)
    // Wave-lagged fill fence. A per-miss submit_barrier on the in-order compute
    // queue serializes DMA behind ALL submitted compute (measured: pinned banks
    // 39 -> 24 tok/s prefill). Instead: fills of wave w depend on the barrier
    // captured at wave w-1's START (covers computes through wave w-2), and the
    // victim scan skips any slot read by waves w-1/w — so fills stream during
    // the previous wave's compute without ever overwriting an in-flight read.
    std::vector<int32_t> slot_wave(ec->expert_in.size(), -2);  // slot -> last wave using it (this call)
    uint32_t cur_wave = 0;
    sycl::event wave_fence = q.ext_oneapi_submit_barrier();    // pre-call readers
    // A slot whose prefetch H2D may not be SUBMITTED yet is unevictable — a
    // demand fill submitted before it would be overwritten by the late DMA.
    // Ready (submitted) prefetches are released in the sweep below the waves.
    auto pf_busy = [&](uint32_t s) {
        return !ec->pf_pending.empty() && ec->pf_pending[s];
    };
    // ... and a slot whose VRAM pages failed init_runtime's alias self-test is
    // unusable for the life of the process: filling it also rewrites the page
    // its partner slot reads.
    auto slot_blocked = [&](uint32_t s) {
        return pf_busy(s) || (!ec->slot_bad.empty() && ec->slot_bad[s]);
    };
    auto ensure = [&](uint32_t e) {
        if (ec->slot_of[e] >= 0) return;
        filled[e] = 1;
        ++ecache_misses;
        ++miss_counts[uint64_t(L) * E + e];
        // Belady over this call's known schedule: prefer a slot whose expert
        // is DONE or never routed (LRU among those); else evict the pending
        // expert whose next use is FARTHEST (max act_pos). Plain LRU here is
        // the sequential-scan worst case (measured: 0%% inter-chunk reuse).
        uint32_t victim = UINT32_MAX;
        uint64_t oldest = UINT64_MAX;
        for (uint32_t s = 0; s < ec->expert_in.size(); ++s) {
            if (slot_blocked(s)) continue;
            if (slot_wave[s] >= int32_t(cur_wave) - 1) continue;   // in-flight reader
            if (ec->expert_in[s] >= 0 && pending[ec->expert_in[s]]) continue;
            if (ec->last_use[s] < oldest) { oldest = ec->last_use[s]; victim = s; }
        }
        if (victim == UINT32_MAX) {
            int32_t far = -1;
            for (uint32_t s = 0; s < ec->expert_in.size(); ++s) {
                if (slot_blocked(s)) continue;
                if (slot_wave[s] >= int32_t(cur_wave) - 1) continue;
                if (act_pos[ec->expert_in[s]] > far) { far = act_pos[ec->expert_in[s]]; victim = s; }
            }
        }
        sycl::event fence = wave_fence;
        if (victim == UINT32_MAX) {
            // Tiny cache: no slot outside the in-flight window. Evict per the
            // old rules behind a full drain — correct, serial for this fill.
            oldest = UINT64_MAX;
            for (uint32_t s = 0; s < ec->expert_in.size(); ++s) {
                if (slot_blocked(s)) continue;
                if (ec->expert_in[s] >= 0 && pending[ec->expert_in[s]]) continue;
                if (ec->last_use[s] < oldest) { oldest = ec->last_use[s]; victim = s; }
            }
            if (victim == UINT32_MAX) {
                int32_t far = -1;
                for (uint32_t s = 0; s < ec->expert_in.size(); ++s) {
                    if (slot_blocked(s)) continue;
                    if (act_pos[ec->expert_in[s]] > far) { far = act_pos[ec->expert_in[s]]; victim = s; }
                }
            }
            if (victim == UINT32_MAX) { fill_err = "glm5next moe: no evictable slot (pf saturation)"; return; }
            fence = q.ext_oneapi_submit_barrier();
        }
        if (ec->expert_in[victim] >= 0) ec->slot_of[ec->expert_in[victim]] = -1;
        ec->expert_in[victim] = int32_t(e);
        ec->slot_of[e] = int16_t(victim);
        ec->last_use[victim] = ++ecache_clock_;
        slot_wave[victim] = int32_t(cur_wave);
        const uint8_t* gs = w.gate_src + uint64_t(e) * gsl;
        const uint8_t* us = w.up_src + uint64_t(e) * usl;
        const uint8_t* ds2 = w.down_src + uint64_t(e) * dsl;
        uint8_t* dst = ec->base + uint64_t(victim) * ec->slot_bytes + ec->gate_off;
        if (w.bank_pinned) {
            // Pinned repacked source: ONE async DMA per miss, zero host bytes.
            sycl::event ev = copyq_->memcpy(
                dst, w.bank_pin + uint64_t(e) * (gsl + usl + dsl),
                gsl + usl + dsl, {fence});
            std::promise<sycl::event> pr;
            pr.set_value(ev);
            fevf[e] = pr.get_future().share();
            return;
        }
        if (T <= 4 && !lanes_.running()) {
            // Decode/verify trickle: INLINE bounce fill — the fill thread's
            // per-miss wake/promise handoff measured decode 6.43 -> 5.26.
            // With the G2 fill lanes on, the request goes to them instead
            // (below): N experts in flight beat the inline serial chain.
            uint8_t* pin = pin_ring_[pin_idx_];
            auto tw0 = std::chrono::steady_clock::now();
            pin_ev_[pin_idx_].wait();
            auto tw1 = std::chrono::steady_clock::now();
            t_ring_wait += std::chrono::duration<double>(tw1 - tw0).count();
            if (auto e2 = pread_fill_(w.gate_fd, w.gate_foff + uint64_t(e) * gsl, gs, gsl,
                                      w.up_fd,   w.up_foff   + uint64_t(e) * usl, us, usl,
                                      w.down_fd, w.down_foff + uint64_t(e) * dsl, ds2, dsl,
                                      pin); !e2.empty()) { fill_err = e2; return; }
            t_pool_copy += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - tw1).count();
            pin_ev_[pin_idx_] = copyq_->memcpy(dst, pin, gsl + usl + dsl, {fence});
            std::promise<sycl::event> pr;
            pr.set_value(pin_ev_[pin_idx_]);
            fevf[e] = pr.get_future().share();
            pin_idx_ = (pin_idx_ + 1) % kPinRing;
            return;
        }
        FillReq r;
        r.dst = dst;
        r.g = gs; r.u = us; r.d2 = ds2;
        r.gsl = gsl; r.usl = usl; r.dsl = dsl;
        r.gfd = w.gate_fd; r.goff = w.gate_foff + uint64_t(e) * gsl;
        r.ufd = w.up_fd;   r.uoff = w.up_foff   + uint64_t(e) * usl;
        r.dfd = w.down_fd; r.doff = w.down_foff + uint64_t(e) * dsl;
        r.bounce = true;   // direct pageable-H2D FALSIFIED (3 GB/s from the worker; 2026-08-28)
        r.fence = fence;
        fevf[e] = fill_.push(std::move(r));
    };
    // Diagnostic (IE_G5_DUMP_MOE=<layer>, needs IE_G5_DUMP_WIDE): stage that
    // layer's per-expert x / gate / up / down tensors as the kernels see them
    // (async D2H on the in-order q; forward_range flushes). Record layout:
    // u32{L,e,nt,path,r0,nrows,kind,len} + i32 tok[nrows] + u16 data[len].
    static const int dmoe_layer = [] {
        const char* v = std::getenv("IE_G5_DUMP_MOE");
        return v ? std::atoi(v) : -1;
    }();
    const bool dmoe_on = dmoe_layer == int(L) && !dump_h_.empty();
    if (dmoe_on) {
        dmoe_h_.assign(uint64_t(E) * T * (2 * EF + 2 * H) + 16, 0);
        dmoe_hdr_.clear();
        dmoe_cur_ = 0;
    }
    auto dmoe = [&](uint32_t kind, const sycl::half* src, uint64_t len, uint32_t e,
                    uint32_t nt, uint32_t path, uint32_t r0, uint32_t nrows) {
        if (!dmoe_on || dmoe_cur_ + len > dmoe_h_.size()) return;
        q.memcpy(dmoe_h_.data() + dmoe_cur_, src, len * 2);
        dmoe_hdr_.insert(dmoe_hdr_.end(), {L, e, nt, path, r0, nrows, kind, uint32_t(len)});
        for (uint32_t i = 0; i < nrows; ++i) dmoe_hdr_.push_back(uint32_t(picks[e][r0 + i]));
        dmoe_cur_ += len;
    };
    // One compute unit: everything the old pass-2 body did for expert e.
    auto compute_e = [&](uint32_t e) -> std::string {
        const uint32_t nt = uint32_t(picks[e].size());
        const int32_t* idx_d = moe_idx_ + e_off[e];
        const float*   w_d   = moe_w_ + e_off[e];
        k_gather_rows(q, x16_, idx_d, moe_xg_, nt, H, {});
        const uint8_t *g_raw, *u_raw, *d_raw;
        if (pp_stream) {   // whole-layer stream regions, e-strided
            g_raw = str_g + uint64_t(e) * gsl;
            u_raw = str_u + uint64_t(e) * usl;
            d_raw = str_d + uint64_t(e) * dsl;
        } else {
            uint8_t* base = ec->base + uint64_t(ec->slot_of[e]) * ec->slot_bytes;
            g_raw = base + ec->gate_off;
            u_raw = base + ec->up_off;
            d_raw = base + ec->down_off;
        }

        // IE_G5_SLOT_VERIFY=<layer|-1>: read the slot back THROUGH THE POINTER
        // THE KERNELS ARE ABOUT TO USE and compare it with the host bank. The
        // fill-time verifier (a D2H issued right after the fill's DMA) cannot
        // see a write that lands BETWEEN the fill and the compute; this can.
        // 2026-09-02 det6/e124: layer 28 computes ~2 MiB of expert 124's up
        // slice from bytes that are not the model's, in the DEFAULT fill path.
        static const int sv_layer = [] {
            const char* v = std::getenv("IE_G5_SLOT_VERIFY");
            return v ? std::atoi(v) : -2;
        }();
        if (sv_layer != -2 && (sv_layer < 0 || uint32_t(sv_layer) == L) && !pp_stream) {
            if (filled[e]) { sycl::event fe = fevf[e].get(); fe.wait(); }
            uint8_t* sbase = ec->base + uint64_t(ec->slot_of[e]) * ec->slot_bytes;
            std::vector<uint8_t> hv(ec->slot_bytes);
            q.memcpy(hv.data(), sbase, ec->slot_bytes).wait();
            const struct { const char* nm; uint64_t off, len; const uint8_t* src; } rg[3] = {
                {"gate", ec->gate_off, gsl, w.gate_src + uint64_t(e) * gsl},
                {"up",   ec->up_off,   usl, w.up_src   + uint64_t(e) * usl},
                {"down", ec->down_off, dsl, w.down_src + uint64_t(e) * dsl}};
            for (const auto& r2 : rg) {
                const uint8_t* got = hv.data() + r2.off;
                ++sv_checks_;
                if (!std::memcmp(got, r2.src, r2.len)) continue;
                uint64_t nbad = 0, first = UINT64_MAX, last = 0;
                for (uint64_t i = 0; i < r2.len; ++i)
                    if (got[i] != r2.src[i]) { ++nbad; if (first == UINT64_MAX) first = i; last = i; }
                ++sv_bad_;
                std::fprintf(stderr,
                             "[slot-verify] L%u e%u slot %d %s: %llu/%llu bytes differ, "
                             "slice range [%llu, %llu], dev %p..%p (slot base %p, "
                             "region base %p, filled=%u, tok %llu)\n",
                             L, e, int(ec->slot_of[e]), r2.nm,
                             (unsigned long long)nbad, (unsigned long long)r2.len,
                             (unsigned long long)first, (unsigned long long)last,
                             (void*)(sbase + r2.off + first), (void*)(sbase + r2.off + last),
                             (void*)sbase, (void*)(sbase + r2.off), unsigned(filled[e]),
                             (unsigned long long)sv_checks_);
                if (dev_alloc_auditing())
                    std::fprintf(stderr, "[slot-verify]   owner of %p: %s\n",
                                 (void*)(sbase + r2.off + first),
                                 dev_alloc_owner(sbase + r2.off + first).c_str());
                // IE_G5_SLOT_DUMP=<prefix>: write the region as the KERNEL sees
                // it plus the model's own bytes, so the foreign content can be
                // identified offline (another expert? KV? zeros? activations?).
                static const char* sv_dump = std::getenv("IE_G5_SLOT_DUMP");
                static int sv_dumps_left = 2;
                if (sv_dump && sv_dumps_left > 0) {
                    --sv_dumps_left;
                    char fn[512];
                    for (int which = 0; which < 2; ++which) {
                        std::snprintf(fn, sizeof fn, "%s.L%u.e%u.slot%d.%s.%s",
                                      sv_dump, L, e, int(ec->slot_of[e]), r2.nm,
                                      which ? "exp" : "got");
                        if (FILE* f = std::fopen(fn, "wb")) {
                            std::fwrite(which ? r2.src : got, 1, r2.len, f);
                            std::fclose(f);
                        }
                    }
                    std::fprintf(stderr, "[slot-verify]   dumped %s.L%u.e%u.slot%d.%s.{got,exp} (%llu B)\n",
                                 sv_dump, L, e, int(ec->slot_of[e]), r2.nm,
                                 (unsigned long long)r2.len);
                }
            }
        }

        auto gemv_raw = [&](const sycl::half* A, const uint8_t* raw, DType dt,
                            sycl::half* yv, uint32_t K, uint32_t N,
                            const std::vector<sycl::event>& deps) -> std::string {
            switch (dt) {
                case DType::kQ4_K: gemv_q4_K(q, A, raw, yv, K, N, deps); break;
                case DType::kQ5_K: gemv_q5_K(q, A, raw, yv, K, N, deps); break;
                case DType::kQ6_K: gemv_q6_K(q, A, raw, yv, K, N, deps); break;
                // UD-Q3_K_XL banks (gate/up IQ3_XXS + one Q3_K layer, down
                // IQ4_XS): raw-slot kernels, no repack — fills unchanged.
                case DType::kQ3_K:    gemv_q3_K(q, A, raw, yv, K, N, deps); break;
                case DType::kIQ4_XS:  gemv_iq4_xs(q, A, raw, yv, K, N, deps); break;
                case DType::kIQ3_XXS: gemv_iq3_xxs_raw(q, A, raw, yv, K, N, deps); break;
                default:
                    return std::string("glm5next moe: unsupported bank dtype ") +
                           std::string(type_name(dt));
            }
            return {};
        };
        if (nt <= 4) {
            // Small-T fast path (decode AND spec verify/replay windows):
            // native raw-quant GEMVs straight off the slot, per row — no
            // dequant materialization (that chain moves ~7x the bytes per
            // expert; at nt<=4 the repeated raw reads are still cheaper).
            // moe_h16_ partitions: [0,EF) gate, [EF,2EF) up, [2EF,2EF+H) down.
            sycl::half* yg = moe_h16_;
            sycl::half* yu = moe_h16_ + EF;
            sycl::half* yd = moe_h16_ + 2 * EF;
            for (uint32_t r = 0; r < nt; ++r) {
                const sycl::half* xr = moe_xg_ + uint64_t(r) * H;
                // the first slot reader carries the fill event (cross-queue);
                // stream mode rides the layer-start barrier instead
                const std::vector<sycl::event> dep = (r == 0 && filled[e])
                    ? std::vector<sycl::event>{fevf[e].get()} : std::vector<sycl::event>{};
                if (w.gate_exps->dtype == DType::kQ4_K && w.up_exps->dtype == DType::kQ4_K) {
                    gemv_q4_K_dual(q, xr, g_raw, u_raw, yg, yu, H, EF, "g5_moe_gu", dep);
                } else {
                    if (auto err = gemv_raw(xr, g_raw, w.gate_exps->dtype, yg, H, EF, dep); !err.empty()) return err;
                    if (auto err = gemv_raw(xr, u_raw, w.up_exps->dtype,   yu, H, EF, {}); !err.empty()) return err;
                }
                dmoe(0, xr, H, e, nt, 0, r, 1); dmoe(1, yg, EF, e, nt, 0, r, 1); dmoe(2, yu, EF, e, nt, 0, r, 1);
                {   // Diagnostic (IE_G5_NAN_PROBE): per-expert stage check on the
                    // small-T f16 path — where does the non-finite value enter?
                    static const bool nan_probe = std::getenv("IE_G5_NAN_PROBE") != nullptr;
                    // IE_G5_NAN_LAYER=<L>: probe ONE layer (the q.wait() per
                    // expert costs ~40x on all 46) — the 2026-09-02 hunt only
                    // needs layer 28. Unset = every layer, as before.
                    static const int nan_layer = [] {
                        const char* v = std::getenv("IE_G5_NAN_LAYER");
                        return v ? std::atoi(v) : -1;
                    }();
                    static int left = [] {
                        const char* v = std::getenv("IE_G5_NAN_MAX");
                        return v ? std::atoi(v) : 6;
                    }();
                    if (nan_probe && (nan_layer < 0 || nan_layer == int(L)) && left > 0) {
                        q.wait();
                        std::vector<sycl::half> hg(EF), hu(EF);
                        q.memcpy(hg.data(), yg, EF * 2).wait();
                        q.memcpy(hu.data(), yu, EF * 2).wait();
                        uint32_t nfg = 0, nfu = 0; float mg = 0, mu = 0;
                        for (uint32_t i = 0; i < EF; ++i) {
                            const float a = float(hg[i]), b = float(hu[i]);
                            if (!std::isfinite(a)) ++nfg; else mg = std::max(mg, std::fabs(a));
                            if (!std::isfinite(b)) ++nfu; else mu = std::max(mu, std::fabs(b));
                        }
                        if (nfg || nfu || mg > 3e4f || mu > 3e4f) {
                            std::fprintf(stderr, "[nan-probe] L%u expert %u row %u: gate nonfinite %u max %.3g | up nonfinite %u max %.3g (clamp %.3g)\n",
                                         L, e, r, nfg, mg, nfu, mu, clamp);
                            --left;
                        }
                    }
                }
                ds4_swiglu_clamped_h(q, yg, yu, yg, EF, clamp, {});
                if (auto err = gemv_raw(yg, d_raw, w.down_exps->dtype, yd, EF, H, {}); !err.empty()) return err;
                dmoe(3, yd, H, e, nt, 0, r, 1);
                k_scatter_add_h(q, yd, idx_d + r, w_d + r, acc_dst, H, {});
            }
            return std::string{};
        }

        // Rows fast path (prefill): direct raw-quant multi-row GEMVs — no
        // dequant materialization (~1/7th the bytes per expert). Q6_K banks
        // (3 down layers) fall through to the dequant+gemm chain below.
        const bool rows_ok =
            (w.gate_exps->dtype == DType::kQ4_K || w.gate_exps->dtype == DType::kQ5_K) &&
            (w.up_exps->dtype == DType::kQ4_K || w.up_exps->dtype == DType::kQ5_K) &&
            (w.down_exps->dtype == DType::kQ4_K || w.down_exps->dtype == DType::kQ5_K);
        if (rows_ok) {
            auto rows = [&](const sycl::half* A, const uint8_t* raw, DType dt,
                            sycl::half* yv, uint32_t K, uint32_t N, uint32_t R,
                            const std::vector<sycl::event>& deps) {
                // gemm_q4_K_xmx at M=16 K=4096 N=2048 is 4× slower than
                // gemv_q4_K_rows on B70 (0.245 vs 0.062 ms) and max|d|=1
                // vs the rows leaf — do not dispatch it on this path.
                if (dt == DType::kQ4_K) gemv_q4_K_rows(q, A, raw, yv, K, N, R, deps);
                else                    gemv_q5_K_rows(q, A, raw, yv, K, N, R, deps);
            };
            const uint32_t MTR = (max_chunk_ + 7) & ~7u;   // moe_h16_ row count
            sycl::half* yg = moe_h16_;
            sycl::half* yu = moe_h16_ + uint64_t(MTR) * EF;
            sycl::half* yd = yu + uint64_t(MTR) * EF;
            const std::vector<sycl::event> dep = filled[e]
                ? std::vector<sycl::event>{fevf[e].get()} : std::vector<sycl::event>{};
            rows(moe_xg_, g_raw, w.gate_exps->dtype, yg, H, EF, nt, dep);
            dmoe(0, moe_xg_, uint64_t(nt) * H, e, nt, 1, 0, nt); dmoe(1, yg, uint64_t(nt) * EF, e, nt, 1, 0, nt);
            rows(moe_xg_, u_raw, w.up_exps->dtype,   yu, H, EF, nt, {});
            dmoe(2, yu, uint64_t(nt) * EF, e, nt, 1, 0, nt);
            ds4_swiglu_clamped_h(q, yg, yu, yg, uint64_t(nt) * EF, clamp, {});
            rows(yg, d_raw, w.down_exps->dtype, yd, EF, H, nt, {});
            dmoe(3, yd, uint64_t(nt) * H, e, nt, 1, 0, nt);
            k_scatter_add_h_rows(q, yd, idx_d, w_d, acc_dst, nt, H, {});
            return std::string{};
        }

        auto emat = [&](const uint8_t* raw, DType dt, sycl::half* dst,
                        uint32_t K, uint32_t N,
                        const std::vector<sycl::event>& deps) -> std::string {
            switch (dt) {
                case DType::kQ4_K: dequant_q4_K_to_Bt(q, raw, dst, K, N, deps); break;
                case DType::kQ5_K: dequant_q5_K_to_Bt(q, raw, dst, K, N, deps); break;
                case DType::kQ6_K: dequant_q6_K_to_Bt(q, raw, dst, K, N, deps); break;
                case DType::kQ3_K:    dequant_q3_K_to_Bt(q, raw, dst, K, N, deps); break;
                case DType::kIQ4_XS:  dequant_iq4_xs_to_Bt(q, raw, dst, K, N, deps); break;
                case DType::kIQ3_XXS: dequant_iq3_xxs_to_Bt(q, raw, dst, K, N, deps); break;
                default:
                    return std::string("glm5next moe: unsupported bank dtype ") +
                           std::string(type_name(dt));
            }
            return {};
        };
        const std::vector<sycl::event> edep = filled[e]
            ? std::vector<sycl::event>{fevf[e].get()} : std::vector<sycl::event>{};
        if (auto err = emat(g_raw, w.gate_exps->dtype, emat_g16_, H, EF, edep); !err.empty()) return err;
        if (auto err = emat(u_raw, w.up_exps->dtype,   emat_u16_, H, EF, {}); !err.empty()) return err;
        if (auto err = emat(d_raw, w.down_exps->dtype, emat_d16_, EF, H, {}); !err.empty()) return err;

        gemm_fp16(q, moe_xg_, emat_g16_, moe_gu_[0], nt, EF, H, {});
        gemm_fp16(q, moe_xg_, emat_u16_, moe_gu_[1], nt, EF, H, {});
        ds4_swiglu_clamped(q, moe_gu_[0], moe_gu_[1], moe_gu_[0], uint64_t(nt) * EF, clamp, {});
        cast_fp32_to_fp16(q, moe_gu_[0], moe_h16_, uint64_t(nt) * EF, {});
        gemm_fp16(q, moe_h16_, emat_d16_, moe_dn_, nt, H, EF, {});
        k_scatter_add(q, moe_dn_, idx_d, w_d, acc_dst, nt, H, {});
        return std::string{};
    };

    // Grouped T==1 decode: ONE launch each for gate/up (all P experts), swiglu,
    // down, and the ordered reduce — vs ~5 launches x P for the per-expert
    // chain. Kernels run the solo lattices verbatim and the reduce reproduces
    // the sequential scatter's fp32 rounding. OPT-IN (IE_G5_GROUPED=1) until
    // the post-reboot oracle clears it: on the 2026-08-29 rotten driver the
    // EP oracle flagged (L14, tok 7) in both grouped runs with VARYING value
    // (race signature — suspect: up to 8 cross-queue fill events on one
    // launch vs the certified single-event pattern) — unattributable there.
    // Eligibility: gate/up Q4_K, down Q5_K/Q6_K (blk 11 falls back).
    // IE_G5_GROUPED: 0/unset off; 1 strict single-launch (bit-identical);
    // 2 ready/late split — ready experts (hit or fill COMPLETE) launch as one
    // grouped job NOW, stragglers run solo so each starts the moment ITS
    // fill lands (strict grouped waits for the SLOWEST fill, which killed
    // the prefetcher's overlap: 7.15 -> 5.14 measured 2026-08-29). Mode 2
    // reorders the fp32 accumulation (EP-noise class) — text/PPL gated.
    static const int grouped_mode = [] {
        const char* v = std::getenv("IE_G5_GROUPED");
        return v ? std::atoi(v) : 0;
    }();
    const bool grouped_on = grouped_mode >= 1;
    static const bool no_q6_slm = std::getenv("IE_NO_Q6K_SLM") != nullptr;
    const bool grouped = grouped_on && T == 1 &&
        w.gate_exps->dtype == DType::kQ4_K && w.up_exps->dtype == DType::kQ4_K &&
        (w.down_exps->dtype == DType::kQ5_K ||
         (w.down_exps->dtype == DType::kQ6_K && !no_q6_slm));
    // IE_G5_PP_TILES: prefill launch-storm fix — a wave's nt>4 experts run
    // as tile batches (3 launches + per-expert scatters vs ~5 x nE); nt<=4
    // experts keep the small-T solo fold, so routing == the old path and the
    // whole layer stays bit-identical. Q6_K-down / Q5_K-gate layers solo.
    static const bool pp_tiles_env = std::getenv("IE_G5_PP_TILES") != nullptr;
    // Stream+tiles measured 39.9 tok/s (GPU1 ring_wait 16 s) vs ~44-48
    // without — overlapping tiles compute with layer H2D starves BCS.
    // Keep tiles opt-in (IE_G5_PP_TILES), not implicit with PP_STREAM.
    const bool pp_tiles = pp_tiles_env && pp_jobs_ && T > 4 &&
        w.gate_exps->dtype == DType::kQ4_K && w.up_exps->dtype == DType::kQ4_K &&
        w.down_exps->dtype == DType::kQ5_K;
    auto compute_grouped = [&](const std::vector<uint32_t>& act_v,
                               uint32_t w0, uint32_t w1) -> std::string {
        static bool once = [] {
            std::fprintf(stderr, "[glm5next] grouped MoE decode active\n");
            return true;
        }();
        (void)once;
        // Every expert owns row r = i - w0 (ascending expert order), late or
        // not; the single ordered reduce at the end sums rows ascending, so
        // the fp32 rounding is timing-independent (mode 2's old late-solo
        // scatters made the accumulation order — and the tokens — depend on
        // DMA completion timing; caught 2026-08-29, g2a vs g2b text drift).
        const uint32_t P = w1 - w0;
        if (!P) return std::string{};
        int32_t slots[8]; float wts[8];
        uint8_t is_late[8] = {}; uint8_t has_ev[8] = {};
        sycl::event fill_ev[8];
        for (uint32_t i = w0; i < w1; ++i) {
            const uint32_t e = act_v[i];
            const uint32_t r = i - w0;
            slots[r] = ec->slot_of[e];
            wts[r]   = h_w_[e_off[e]];   // T==1: one pick per expert
            // One event per filled expert: with prefetch in the mix,
            // submission order on copyq_ is not knowable host-side (pf fills
            // come from the fill worker), so a "last event covers all"
            // collapse can leave a NEWER demand fill unfenced.
            if (filled[e]) {
                fill_ev[r] = fevf[e].get(); has_ev[r] = 1;
                // ready/late split (mode 2): an INCOMPLETE fill would stall
                // the whole grouped launch — that row launches separately,
                // gated on its own fill, after the ready runs.
                if (grouped_mode >= 2 &&
                    fill_ev[r].get_info<sycl::info::event::command_execution_status>()
                        != sycl::info::event_command_status::complete)
                    is_late[r] = 1;
            }
        }
        const uint32_t MTR = (max_chunk_ + 7) & ~7u;
        sycl::half* yg = moe_h16_;
        sycl::half* yu = moe_h16_ + uint64_t(MTR) * EF;
        sycl::half* yd = yu + uint64_t(MTR) * EF;
        auto run_rows = [&](uint32_t r0, uint32_t len,
                            const std::vector<sycl::event>& dep) {
            moe_gemv_q4_K_gu_grouped(q, x16_, ec->base, ec->slot_bytes,
                                     ec->gate_off, ec->up_off, slots + r0, len,
                                     yg + uint64_t(r0) * EF, yu + uint64_t(r0) * EF,
                                     H, EF, dep);
            ds4_swiglu_clamped_h(q, yg + uint64_t(r0) * EF, yu + uint64_t(r0) * EF,
                                 yg + uint64_t(r0) * EF, uint64_t(len) * EF, clamp, {});
            if (w.down_exps->dtype == DType::kQ5_K)
                moe_gemv_q5_K_down_grouped(q, yg + uint64_t(r0) * EF, ec->base,
                                           ec->slot_bytes, ec->down_off, slots + r0,
                                           len, yd + uint64_t(r0) * H, EF, H, {});
            else
                moe_gemv_q6_K_down_grouped(q, yg + uint64_t(r0) * EF, ec->base,
                                           ec->slot_bytes, ec->down_off, slots + r0,
                                           len, yd + uint64_t(r0) * H, EF, H, {});
        };
        // Ready rows first (contiguous ascending runs), then late rows —
        // submission order can't change the result (disjoint rows; the
        // in-order queue runs the reduce after every write).
        for (uint32_t r = 0; r < P;) {
            if (is_late[r]) { ++r; continue; }
            const uint32_t r0 = r;
            std::vector<sycl::event> dep;
            for (; r < P && !is_late[r]; ++r)
                if (has_ev[r]) dep.push_back(fill_ev[r]);
            run_rows(r0, r - r0, dep);
        }
        for (uint32_t r = 0; r < P; ++r)
            if (is_late[r]) run_rows(r, 1, {fill_ev[r]});
        moe_reduce_waccum(q, yd, wts, P, acc_dst, H, {});
        return std::string{};
    };

    // WAVES of at most `slots` experts (qwen4exp's wave-flush shape): bump
    // this wave's hits, fill its misses (H2D-paced, on the transfer queue),
    // then submit its computes — the async submits let the NEXT wave's fills
    // overlap this wave's GPU work, and a wave can never evict its own
    // fills (the 18:11 DEVICE_LOST class). Compute order stays ascending
    // expert id across waves — numerics identical to the single-pass loop.
    std::vector<uint32_t> act;
    for (uint32_t e = 0; e < E; ++e) if (!picks[e].empty()) act.push_back(e);
    // G1 (docs/glm53/CAMPAIGN_2026-09-02.md): ON by default — the 2026-08-31
    // recipe (6.65 -> 9.60 tok/s) with its two enablers, the persistent
    // CpuWorker and the per-stage core partition, which now self-configures
    // (g5_auto_core_split below). IE_G5_CPU_MISS=0 restores the all-fill path.
    static const bool cpu_miss_on = [] {
        const char* v = std::getenv("IE_G5_CPU_MISS");
        const bool on = !(v && v[0] == '0');
        // With q* on, the IE_G5_DUMP_WIDE records carry the CPU path's Q8_K
        // numerics and IE_G5_DUMP_MOE omits the CPU-computed experts; an
        // all-fill comparison (det bisect, "== P1.nll") needs IE_G5_CPU_MISS=0.
        if (on && (std::getenv("IE_G5_DUMP_WIDE") || std::getenv("IE_G5_DUMP_MOE")))
            std::fprintf(stderr, "[glm5next] note: q* CPU-miss split is ON — dumps carry q* numerics; "
                                 "set IE_G5_CPU_MISS=0 for an all-fill comparison\n");
        return on;
    }();
    const bool use_cpu_miss = cpu_miss_on && T == 1 && !pp_stream
        && w.gate_exps->dtype == DType::kQ4_K
        && w.up_exps->dtype == DType::kQ4_K
        && w.down_exps->dtype == DType::kQ5_K;
    std::vector<uint32_t> cpu_act;
    sycl::event cpu_xev;
    // q* split (FreeToken §3.2, arXiv 2608.16157): of the m misses, q* =
    // round(m · B_P/B_H) are FILLED over PCIe and computed on the GPU (they
    // also become resident for future hits); the remaining m-q* execute on
    // the CPU from the host bank, concurrently, leaving residency unchanged.
    // B_P/B_H measured on this rig 26.5 / 60.2 GB/s (cpu_moe_gemv_bench)
    // -> 0.44. IE_G5_QSTAR overrides (1.0 == all-fill, 0.0 == grok's all-CPU
    // mode, which measured 3.59 tok/s vs 6.65 baseline: serial CPU compute
    // after the waves PLUS a DMA fill of every CPU expert). The CPU branch
    // runs on a worker thread launched BEFORE the wave submits.
    // Default 0.25, not the bandwidth ratio 0.44: at T = 1 the PCIe leg is a
    // per-expert LATENCY chain (~1.1 ms: pread + DMA, serial), not a full-rate
    // stream, and the CPU leg runs 0.55-0.7 ms/expert in parallel. Sweep
    // 2026-09-02 (glm1b, decode-96, no other env): 0.15 -> 9.07, 0.25 -> 9.55 /
    // 9.67, 0.35 -> 9.46, 0.44 -> 8.34 / 8.80 / 8.92 tok/s. `qf >= 1` below keeps
    // one admission per layer so the cache still learns.
    // Per-stage override (IE_G5_QSTAR_A / IE_G5_QSTAR_B, lever 4 of the
    // pipelined draft: stage A's link idles while B runs, so A can fill more).
    // Default 0.25 -> 0.40 (2026-09-04 m3, pinned banks, counted host waits):
    // 0.25 10.61 / 0.40 12.02 / 0.60 11.55 / 0.80 10.28 / all-fill 9.35 tok/s.
    // The CPU-join wait falls monotonically with q* but the misses that leave
    // the CPU go to the fill path, and on the tail stage that is the unpinned
    // layers' pread — the two exposed waits trade, and 0.40 is where they
    // cross. (0.25 was tuned on 09-02 when host staging drowned the knob.)
    static const float qstar_all = [] {
        const char* v = std::getenv("IE_G5_QSTAR");
        return v ? std::strtof(v, nullptr) : 0.40f;
    }();
    static const float qstar_a = [] {
        const char* v = std::getenv("IE_G5_QSTAR_A");
        return v ? std::strtof(v, nullptr) : qstar_all;
    }();
    static const float qstar_b = [] {
        const char* v = std::getenv("IE_G5_QSTAR_B");
        return v ? std::strtof(v, nullptr) : qstar_all;
    }();
    const float qstar = layer_lo_ == 0 ? qstar_a : qstar_b;
    static const bool qstar_install = std::getenv("IE_G5_QSTAR_INSTALL") != nullptr;
    // The CPU job captures run_moe locals by reference; an error return between
    // submit and the join below must not leave it running over dead stack
    // (G1 gate finding 4). Armed at submit, disarmed at the normal join.
    struct CpuJoinGuard {
        CpuWorker* w = nullptr;
        ~CpuJoinGuard() { if (w) w->wait(); }
    } cpu_join;
    if (use_cpu_miss) {
        std::vector<uint32_t> miss;
        for (uint32_t e : act) if (ec->slot_of[e] < 0) miss.push_back(e);
        const uint32_t m = uint32_t(miss.size());
        uint32_t qf = uint32_t(std::lround(double(qstar) * m));
        if (m && qf == 0) qf = 1;            // always keep the cache warming
        if (qf > m) qf = m;
        std::vector<char> is_cpu(E, 0);
        for (uint32_t i = qf; i < m; ++i) is_cpu[miss[i]] = 1;
        std::vector<uint32_t> gpu_act;
        gpu_act.reserve(act.size());
        for (uint32_t e : act) (is_cpu[e] ? cpu_act : gpu_act).push_back(e);
        act.swap(gpu_act);
        if (!cpu_act.empty()) {
            cpu_x16h_.resize(H);
            cpu_xf_.resize(H);
            cpu_y_.resize(H);
            // The H2D that consumed this ring slot kCpuAccRing layers ago was
            // submitted WITHOUT a wait (below); the host runs ahead of q, so it may
            // not have read the buffer yet — zeroing and refilling it here raced
            // that DMA (glm1b pq vs pq2, 2026-09-02: q* path non-reproducible from
            // token 131 on). Wait for that copy before touching the slot.
            cpu_acc_ev_[cpu_acc_sel_].wait();
            cpu_acc_ring_[cpu_acc_sel_].assign(H, 0.f);
            cpu_xev = q.memcpy(cpu_x16h_.data(), x16_, uint64_t(H) * 2);
            const auto* gbase = reinterpret_cast<const block_q4_K*>(w.gate_src);
            const auto* ubase = reinterpret_cast<const block_q4_K*>(w.up_src);
            const auto* dbase = reinterpret_cast<const block_q5_K*>(w.down_src);
            // IE_G5_PIN_BANKS: the pinned bank is per-expert CONTIGUOUS
            // [gate|up|down] with stride gsl+usl+dsl, and the loader
            // madvise(MADV_DONTNEED)s the mmap that the three base pointers
            // above address. Reading them here would fault EVERY CPU expert
            // back off NVMe — pinning and q* were silently incompatible.
            const uint8_t* pin_bank  = w.bank_pinned ? w.bank_pin : nullptr;
            const uint64_t pin_stride = gsl + usl + dsl;
            const uint64_t gblk = gsl / sizeof(block_q4_K);
            const uint64_t ublk = usl / sizeof(block_q4_K);
            const uint64_t dblk = dsl / sizeof(block_q5_K);
            float* cbuf = cpu_acc_ring_[cpu_acc_sel_].data();
            const std::vector<uint32_t>& cact = cpu_act;
            if (cpu_worker_.core_lo < 0) {
                const char* cr = std::getenv(layer_lo_ == 0 ? "IE_G5_CPU_CORES_A"
                                                            : "IE_G5_CPU_CORES_B");
                int lo = -1, hi = -1;
                cpu_worker_.stage = layer_lo_ == 0 ? "A" : "B";
                if (cr && std::sscanf(cr, "%d-%d", &lo, &hi) == 2) {
                    cpu_worker_.core_lo = lo; cpu_worker_.core_hi = hi;
                    std::fprintf(stderr, "[glm5next] q* CPU worker (stage %s): cores %d-%d (IE_G5_CPU_CORES_%s)\n",
                                 layer_lo_ == 0 ? "A" : "B", lo, hi, layer_lo_ == 0 ? "A" : "B");
                } else {
                    // No env. Both teams on the P-cores when the stages run
                    // serially (G1 fix round, 2026-09-02: 9.42 vs 8.32 tok/s
                    // for the P/E split — the E-core team is the longer leg at
                    // 0.83 ms/expert, and passive waits keep the idle team off
                    // the cores); the tail stage takes the E-cores only when the
                    // caller declared concurrent stages (set_cpu_partition), where
                    // two teams on one block measured 1.44 vs 0.25 ms/expert
                    // (probe 2026-08-31).
                    // Concurrent stages (set_cpu_partition, --pipedraft) were
                    // measured too, 2026-09-02 glm5p/glm5q: tail team on the
                    // E-cores 10.6-11.1 tok/s (8 or 12 threads — the E block is
                    // bandwidth-bound, not thread-bound) vs both teams on the
                    // P-cores 11.4-12.0. So the P-cores host both teams in both
                    // modes; the E block is reachable only through the env.
                    static const G5CoreSplit cs = g5_auto_core_split();
                    cpu_worker_.core_lo = cs.a_lo; cpu_worker_.core_hi = cs.a_hi;
                    if (cpu_worker_.core_lo >= 0)
                        std::fprintf(stderr, "[glm5next] q* CPU worker (stage %s): cores %d-%d (auto: P-cores for both stages, %s; IE_G5_CPU_CORES_%s overrides)\n",
                                     layer_lo_ == 0 ? "A" : "B", cpu_worker_.core_lo, cpu_worker_.core_hi,
                                     cpu_partition_ ? "concurrent stages — measured 11.4-12.0 vs 10.6-11.1 with the tail on E"
                                                    : "serial stages — measured 9.4 vs 8.3 with the tail on E",
                                     layer_lo_ == 0 ? "A" : "B");
                }
            }
            // Team size = the cores the worker is pinned to (12 on the E-core
            // block, 8 on the P-cores); 8 when unpinned (the kernel's default).
            const int nthr = cpu_threads_ ? int(cpu_threads_) : (cpu_worker_.core_lo >= 0
                ? cpu_worker_.core_hi - cpu_worker_.core_lo + 1 : 8);
            cpu_join.w = &cpu_worker_;
            cpu_worker_.submit(
                [this, cpu_xev, gbase, ubase, dbase, gblk, ublk, dblk, cbuf,
                 pin_bank, pin_stride, gsl, usl,
                 &cact, &e_off, H, EF, clamp, nthr]() mutable {
                    cpu_xev.wait();
                    const auto tcb0 = std::chrono::steady_clock::now();
                    for (uint32_t i = 0; i < H; ++i) cpu_xf_[i] = float(cpu_x16h_[i]);
                    // G1b (IE_G5_CPU_Q8S=1): per-32 activation scales — oracle 2x
                    // less error than the per-256 block_q8_K at +5% time (opt-in
                    // until the PPL-trace gate on a clean boot, 2026-09-02).
                    static const bool q8s = [] {
                        const char* v = std::getenv("IE_G5_CPU_Q8S");
                        return v && v[0] == '1';
                    }();
                    for (uint32_t e : cact) {
                        const block_q4_K* gp;
                        const block_q4_K* upp;
                        const block_q5_K* dp;
                        if (pin_bank) {
                            const uint8_t* eb = pin_bank + uint64_t(e) * pin_stride;
                            gp  = reinterpret_cast<const block_q4_K*>(eb);
                            upp = reinterpret_cast<const block_q4_K*>(eb + gsl);
                            dp  = reinterpret_cast<const block_q5_K*>(eb + gsl + usl);
                        } else {
                            gp  = gbase + e * gblk;
                            upp = ubase + e * ublk;
                            dp  = dbase + e * dblk;
                        }
                        if (q8s)
                            cpu_moe_expert_q8s_nt(cpu_xf_.data(), gp, upp, dp,
                                                  H, EF, clamp, cpu_y_.data(), nthr);
                        else
                            cpu_moe_expert_q8_nt(cpu_xf_.data(), gp, upp, dp,
                                                 H, EF, clamp, cpu_y_.data(), nthr);
                        const float wt = h_w_[e_off[e]];
                        for (uint32_t i = 0; i < H; ++i) cbuf[i] += wt * cpu_y_[i];
                    }
                    t_cpu_compute += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - tcb0).count();
                    n_cpu_experts += cact.size();
                });
        }
    }
    for (uint32_t i = 0; i < act.size(); ++i) { pending[act[i]] = 1; act_pos[act[i]] = int32_t(i); }
    // Two overlapping miss waves occupy up to 2*Wsz slots. Half-cache
    // waves leave almost no room for pending resident experts, so a long
    // sequential prefill scan reloads them before their turn. In warm,
    // overflowing prefill, cap waves at 16 to leave a retained set while
    // keeping enough fills in flight to overlap compute (8 was slower).
    // Cold/fitting batches and decode retain the original wave width.
    uint32_t Wsz = std::max<uint32_t>(8, uint32_t(ec->expert_in.size()) / 2);
    static const uint32_t pp_wave = [] {
        const char* v = std::getenv("IE_G5_PP_WAVE");
        return v ? uint32_t(std::max(0, std::atoi(v))) : 16u;
    }();   // 0 restores half-cache waves for comparisons
    if (pp_wave && T > 4 && act.size() > ec->expert_in.size() &&
        !pp_stream && !pp_tiles && !pf_on && !shadow_cnt &&
        std::any_of(act.begin(), act.end(), [&](uint32_t e) { return ec->slot_of[e] >= 0; }))
        Wsz = std::min(Wsz, pp_wave);
    sycl::event prev_wave_barrier = wave_fence;
    // Tiles jobs: 16 device slices, async H2D on copyq_ (no wait — a wait
    // sits behind the next layer's DMA). Host ring keeps each batch's bytes
    // alive until that slice is reused. Cursor lives across waves.
    constexpr uint32_t kJobBufs = 16;
    constexpr uint32_t kJobSpan = 2048 * 3;
    uint32_t job_cur = 0;
    std::vector<int32_t> hring[kJobBufs];
    for (uint32_t w0 = 0; w0 < act.size(); w0 += Wsz) {
        const uint32_t w1 = std::min<uint32_t>(w0 + Wsz, uint32_t(act.size()));
        cur_wave = w0 / Wsz;
        // B_w = everything submitted so far (computes through wave w-1). This
        // wave's fills use B_{w-1} (lagged one wave) so they overlap the
        // previous wave's compute; the slot_wave victim guard keeps them off
        // any slot waves w-1/w are reading.
        sycl::event Bw = q.ext_oneapi_submit_barrier();
        wave_fence = prev_wave_barrier;
        prev_wave_barrier = Bw;
        if (!pp_stream) {
            for (uint32_t i = w0; i < w1; ++i) {
                const int16_t slot = ec->slot_of[act[i]];
                if (slot >= 0) {
                    ++ecache_hits;
                    slot_wave[slot] = int32_t(cur_wave);
                    if (ec->last_use[slot] != UINT64_MAX)
                        ec->last_use[slot] = ++ecache_clock_;
                    // Prefetched slot's first demand reader carries the fill event.
                    if (pf_busy(uint32_t(slot))) {
                        fevf[act[i]] = ec->pf_ev[slot];
                        filled[act[i]] = 1;
                        ec->pf_pending[slot] = 0;
                        ++pf_used_;
                    }
                }
            }
            for (uint32_t i = w0; i < w1; ++i) ensure(act[i]);
        }
        auto tc0 = std::chrono::steady_clock::now();
        if (grouped) {
            for (uint32_t i = w0; i < w1; ++i) ensure(act[i]);   // idempotent refill
            if (auto err = compute_grouped(act, w0, w1); !err.empty()) return err;
            for (uint32_t i = w0; i < w1; ++i) pending[act[i]] = 0;
        } else if (pp_tiles) {
            const uint32_t MTR = (max_chunk_ + 7) & ~7u;
            sycl::half* yg = moe_h16_;
            sycl::half* yu = moe_h16_ + uint64_t(MTR) * EF;
            sycl::half* yd = yu + uint64_t(MTR) * EF;
            uint32_t i = w0;
            while (i < w1) {
                while (i < w1 && picks[act[i]].size() <= 4) {   // small-T solo
                    if (!pp_stream) ensure(act[i]);
                    if (auto err = compute_e(act[i]); !err.empty()) return err;
                    pending[act[i]] = 0;
                    ++i;
                }
                if (i >= w1) break;
                // batch of consecutive nt>4 experts (contiguous picks), row-
                // capped at MTR (the wave-local y buffers).
                const uint32_t b0 = i;
                const uint32_t pick0 = e_off[act[i]];
                uint32_t rows = 0;
                const uint32_t slot = job_cur % kJobBufs;
                auto& hj = hring[slot];
                hj.clear();
                std::vector<sycl::event> dep;
                while (i < w1 && picks[act[i]].size() > 4) {
                    const uint32_t e = act[i];
                    const uint32_t nt = uint32_t(picks[e].size());
                    if (rows + nt > MTR ||
                        hj.size() / 3 + (nt + 15) / 16 > 256) break;
                    if (!pp_stream) ensure(e);
                    for (uint32_t r0 = 0; r0 < nt; r0 += 16) {
                        hj.push_back(int32_t(e_off[e] + r0));
                        hj.push_back(int32_t(std::min<uint32_t>(16, nt - r0)));
                        hj.push_back(pp_stream ? int32_t(e)
                                               : int32_t(ec->slot_of[e]));
                    }
                    if (!pp_stream && filled[e]) dep.push_back(fevf[e].get());
                    rows += nt;
                    ++i;
                }
                if (hj.empty()) {   // can't batch (defensive): solo it
                    if (!pp_stream) ensure(act[i]);
                    if (auto err = compute_e(act[i]); !err.empty()) return err;
                    pending[act[i]] = 0;
                    ++i;
                    continue;
                }
                const uint32_t G = uint32_t(hj.size() / 3);
                int32_t* jdst = pp_jobs_ + uint64_t(slot) * kJobSpan;
                dep.push_back(jobsq_->memcpy(jdst, hj.data(), hj.size() * 4));
                ++job_cur;
                if (pp_stream) {
                    moe_gemv_q4_K_rows_tiles(q, x16_, moe_idx_, pick0, str_g,
                                             gsl, 0, jdst, yg, H, EF, G, dep);
                    moe_gemv_q4_K_rows_tiles(q, x16_, moe_idx_, pick0, str_u,
                                             usl, 0, jdst, yu, H, EF, G, {});
                    ds4_swiglu_clamped_h(q, yg, yu, yg, uint64_t(rows) * EF, clamp, {});
                    moe_gemv_q5_K_rows_tiles(q, yg, nullptr, pick0, str_d,
                                             dsl, 0, jdst, yd, EF, H, G, {});
                } else {
                    moe_gemv_q4_K_rows_tiles(q, x16_, moe_idx_, pick0, ec->base,
                                             ec->slot_bytes, ec->gate_off, jdst,
                                             yg, H, EF, G, dep);
                    moe_gemv_q4_K_rows_tiles(q, x16_, moe_idx_, pick0, ec->base,
                                             ec->slot_bytes, ec->up_off, jdst,
                                             yu, H, EF, G, {});
                    ds4_swiglu_clamped_h(q, yg, yu, yg, uint64_t(rows) * EF, clamp, {});
                    moe_gemv_q5_K_rows_tiles(q, yg, nullptr, pick0, ec->base,
                                             ec->slot_bytes, ec->down_off, jdst,
                                             yd, EF, H, G, {});
                }
                for (uint32_t j = b0; j < i; ++j) {   // ordered per-expert adds
                    const uint32_t e = act[j];
                    k_scatter_add_h_rows(q, yd + uint64_t(e_off[e] - pick0) * H,
                                         moe_idx_ + e_off[e], moe_w_ + e_off[e],
                                         acc_dst, uint32_t(picks[e].size()), H, {});
                    pending[e] = 0;
                }
            }
        } else
        for (uint32_t i = w0; i < w1; ++i) {
            if (!pp_stream)   // stream mode has no cache to fill
                ensure(act[i]);   // idempotent: refills if a full-pending fallback evicted it
            if (auto err = compute_e(act[i]); !err.empty()) return err;
            pending[act[i]] = 0;
        }
        t_compute_submit += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tc0).count();
        if (!fill_err.empty()) return fill_err;
    }
    if (!cpu_act.empty()) {
        const auto tw0 = std::chrono::steady_clock::now();
        cpu_join.w = nullptr;
        cpu_worker_.wait();   // CPU branch ran concurrently with the GPU waves
        t_cpu_join += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tw0).count();
        auto& cbuf = cpu_acc_ring_[cpu_acc_sel_];
        // q* semantics: CPU-executed experts leave residency unchanged (the
        // q* fills are the cache's admissions). IE_G5_QSTAR_INSTALL=1 restores
        // grok's post-compute install (a second DMA per CPU expert).
        for (uint32_t e : qstar_install ? cpu_act : std::vector<uint32_t>{}) {
            ensure(e);
            if (ec->slot_of[e] >= 0 && filled[e] && !ec->pf_pending.empty()) {
                const uint32_t s = uint32_t(ec->slot_of[e]);
                ec->pf_ev[s] = fevf[e];
                ec->pf_pending[s] = 1;
            }
        }
        if (!fill_err.empty()) return fill_err;
        // Submit the 16 KB H2D on the in-order compute queue WITHOUT wait:
        // memcpy.wait() drained shexp+GPU MoE of THIS layer (kprof 36 ms
        // GPU/tok vs ~240 ms wall). Same-q submit is ordered before the
        // add; ping-pong host+device so the next layer can pack. Do NOT
        // put this on copyq_ (cross-queue dep vs ensure() fills deadlocked).
        float* dacc = cpu_acc_dev_[cpu_acc_sel_];
        cpu_acc_ev_[cpu_acc_sel_] = q.memcpy(dacc, cbuf.data(), uint64_t(H) * 4);
        float* acc = acc_dst;
        ie::ps(q, "g5_cpu_acc", [&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(H), [=](sycl::id<1> id) {
                acc[id[0]] += dacc[id[0]];
            });
        });
        cpu_acc_sel_ = (cpu_acc_sel_ + 1) % kCpuAccRing;
        static bool once = [] {
            std::fprintf(stderr, "[glm5next] CPU-miss MoE decode active (IE_G5_CPU_MISS=1)\n");
            return true;
        }();
        (void)once;
    }

    // Release wrong-prediction prefetches whose H2D is COMPLETE. Submitted
    // is enough for REFILL safety (in-order copyq_ serializes a later fill),
    // but a released slot's next HIT launches its reader with NO dep on the
    // pf DMA (pf_busy false -> no event adoption) — reader on q races the
    // in-flight H2D on copyq_ and can read a half-filled slot (the rare
    // timing-dependent token flips, 2026-08-29 d1/d3).
    if (!ec->pf_pending.empty()) {
        for (uint32_t s = 0; s < ec->pf_pending.size(); ++s)
            if (ec->pf_pending[s] &&
                ec->pf_ev[s].wait_for(std::chrono::seconds(0)) == std::future_status::ready &&
                ec->pf_ev[s].get().get_info<sycl::info::event::command_execution_status>()
                    == sycl::info::event_command_status::complete)
                ec->pf_pending[s] = 0;
    }

    // R9 layer-ahead proxy COLLECT (submitted before the waves; the wait is
    // ~free by now). IE_G5_SHADOW_COUNT scores the proxy (measured 76.8%,
    // 2026-08-29); IE_G5_PREFETCH streams the predicted misses into Ln's
    // cache through the fill thread, overlapping everything after this call
    // (post-moe, attention, the EP wait). A prefetched slot's first demand
    // reader picks up the fill event via pf_pending.
    if ((shadow_cnt || pf_on) && pf_rows) {
        if (shadow_pred_.empty()) {
            shadow_pred_.assign(uint64_t(cfg_.n_layers) * TK, -1);
            shadow_has_.assign(cfg_.n_layers, 0);
        }
        if (shadow_cnt && shadow_has_[L]) {
            if (shadow_lhits_.empty()) {
                shadow_lhits_.assign(cfg_.n_layers, 0);
                shadow_ltotal_.assign(cfg_.n_layers, 0);
            }
            uint32_t ov = 0;
            for (uint32_t k = 0; k < TK; ++k)
                for (uint32_t j = 0; j < TK; ++j)
                    if (shadow_pred_[uint64_t(L) * TK + k] == h_top_[j]) { ++ov; break; }
            shadow_hits_ += ov;
            shadow_total_ += TK;
            shadow_lhits_[L] += ov;
            shadow_ltotal_[L] += TK;
        }
        shadow_has_[L] = 0;
        if (shadow_cnt) {
            if (shadow2_pred_.empty()) {
                shadow2_pred_.assign(uint64_t(cfg_.n_layers) * TK, -1);
                shadow2_has_.assign(cfg_.n_layers, 0);
            }
            if (shadow2_has_[L]) {
                uint32_t ov = 0;
                for (uint32_t k = 0; k < TK; ++k)
                    for (uint32_t j = 0; j < TK; ++j)
                        if (shadow2_pred_[uint64_t(L) * TK + k] == h_top_[j]) { ++ov; break; }
                shadow2_hits_ += ov;
                shadow2_total_ += TK;
            }
            shadow2_has_[L] = 0;
            if (pred_ln2 != UINT32_MAX) {
                pred2_ev.wait();
                std::copy(pf_pred_pin_ + uint64_t(kSpecMax) * TK,
                          pf_pred_pin_ + uint64_t(kSpecMax + 1) * TK,
                          shadow2_pred_.begin() + uint64_t(pred_ln2) * TK);
                shadow2_has_[pred_ln2] = 1;
            }
        }
        // Issue prefetches for one predicted layer. Slots holding a predicted
        // expert are protected; the barrier fences the fill DMA behind every
        // submitted reader (earlier tokens' kernels included).
        auto issue_pf = [&](uint32_t Ln, const int32_t* p8) {
            ECache* ecn = (Ln < ecache_.size() && ecache_[Ln].base)
                              ? &ecache_[Ln] : nullptr;
            const Glm5NextLayer& wn = layers_[Ln];
            if (!ecn || !wn.gate_exps || ecn->pf_pending.empty()) return;
            sycl::event pfence;
            bool have_fence = false;
            std::vector<uint8_t> pred(E, 0);
            for (uint32_t k = 0; k < TK; ++k)
                if (p8[k] >= 0) pred[p8[k]] = 1;
            const uint64_t gsl2 = wn.gate_exps->nbytes / E;
            const uint64_t usl2 = wn.up_exps->nbytes / E;
            const uint64_t dsl2 = wn.down_exps->nbytes / E;
            for (uint32_t k = 0; k < TK; ++k) {
                const int32_t pe = p8[k];
                if (pe < 0 || ecn->slot_of[pe] >= 0) continue;
                if (ep_peer_ && ((uint32_t(pe) & 1u) != ep_parity_)) continue;
                uint32_t victim = UINT32_MAX;
                uint64_t oldest = UINT64_MAX;
                for (uint32_t s = 0; s < ecn->expert_in.size(); ++s) {
                    if (ecn->pf_pending[s]) continue;
                    if (!ecn->slot_bad.empty() && ecn->slot_bad[s]) continue;
                    if (ecn->expert_in[s] >= 0 && pred[ecn->expert_in[s]]) continue;
                    if (ecn->last_use[s] < oldest) { oldest = ecn->last_use[s]; victim = s; }
                }
                if (victim == UINT32_MAX) break;
                if (!have_fence) { pfence = q.ext_oneapi_submit_barrier(); have_fence = true; }
                if (ecn->expert_in[victim] >= 0)
                    ecn->slot_of[ecn->expert_in[victim]] = -1;
                ecn->expert_in[victim] = pe;
                ecn->slot_of[pe] = int16_t(victim);
                ecn->last_use[victim] = ++ecache_clock_;
                FillReq r;
                r.dst = ecn->base + uint64_t(victim) * ecn->slot_bytes + ecn->gate_off;
                r.g = wn.gate_src + uint64_t(pe) * gsl2;
                r.u = wn.up_src + uint64_t(pe) * usl2;
                r.d2 = wn.down_src + uint64_t(pe) * dsl2;
                r.gsl = gsl2; r.usl = usl2; r.dsl = dsl2;
                r.gfd = wn.gate_fd; r.goff = wn.gate_foff + uint64_t(pe) * gsl2;
                r.ufd = wn.up_fd;   r.uoff = wn.up_foff + uint64_t(pe) * usl2;
                r.dfd = wn.down_fd; r.doff = wn.down_foff + uint64_t(pe) * dsl2;
                r.bounce = true;
                r.pf = true;   // pf-owned ring/pool — the shared ones
                               // race the inline fills (08:20 deadlock:
                               // two callers corrupt PreadPool's counters)
                r.fence = pfence;
                ecn->pf_ev[victim] = fill_.push(std::move(r));
                ecn->pf_pending[victim] = 1;
                ++pf_issued_;
            }
        };
        if (pred_ln != UINT32_MAX) {
            pred_ev.wait();
            std::copy(pf_pred_pin_, pf_pred_pin_ + TK,
                      shadow_pred_.begin() + uint64_t(pred_ln) * TK);
            shadow_has_[pred_ln] = 1;
            // One issue pass per predicted row: at T>1 the layer will demand
            // the union of every row's picks. issue_pf skips already-resident
            // experts, so later rows only add what the earlier ones missed.
            if (pf_on)
                for (uint32_t r = 0; r < pf_rows; ++r)
                    issue_pf(pred_ln, pf_pred_pin_ + uint64_t(r) * TK);
            // Ship the OTHER parity's predictions to the serving peer: its
            // EpThread warms its half-cache on ITS link before the Ln
            // compute request arrives (FIFO order guarantees precedence).
            if (pf_on && ep_peer_ && pred_ln < ep_peer_->ep_cache_.size() &&
                ep_peer_->ep_cache_[pred_ln].base) {
                EpReq r;
                r.pf_only = true;
                r.L = pred_ln;
                r.T = 1;
                r.w = &layers_[pred_ln];
                int n = 0;
                for (uint32_t k = 0; k < TK && n < 8; ++k) {
                    const int32_t pe = pf_pred_pin_[k];
                    if (pe >= 0 && (uint32_t(pe) & 1u) != ep_parity_)
                        r.pf_pred[n++] = pe;
                }
                if (n) {
                    ep_peer_->ep_thread_.start(ep_peer_);
                    ep_peer_->ep_thread_.push(std::move(r));
                }
            }
        }
        // Depth-2: two layers of fill lead time (proxy accuracy 63.2%,
        // 2026-08-29). Closer depth-1 predictions at Ln refine later; already-
        // resident experts are skipped.
        if (pf_depth >= 2 && pred_ln2 != UINT32_MAX) {
            pred2_ev.wait();
            issue_pf(pred_ln2, pf_pred_pin_ + uint64_t(kSpecMax) * TK);
        }
    }
    if (ep_used) {
        static const bool ep_push2 = std::getenv("IE_G5_EP_PUSH") != nullptr;
        ep_ret_ev = ep_ret_fut_.get();   // worker finished submitting the chain
        ep_ret_ev.wait();                // peer ret landed (host-side sync)
        if (!ep_push2)   // push mode: partial is ALREADY in ep_ret_ (P2P write)
            q.memcpy(ep_ret_, ep_rstage_, uint64_t(T) * H * 4);  // H2D (our link)
        float* acc = moe_acc_;
        const float* ret = ep_ret_;
        ep_add_ev_ = ie::ps(q, "g5_ep_add", [&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(uint64_t(T) * H), [=](sycl::id<1> i) {
                acc[i[0]] += ret[i[0]];
            });
        });
        // IE_G5_EP_VERIFY: recompute the peer half LOCALLY and diff against
        // the peer's returned partial — per-layer bug localization.
        if (epv) {
            if (!ep_vacc_) ep_vacc_ = static_cast<float*>(alloc_->malloc(8ull * H * 4));
            q.memset(ep_vacc_, 0, uint64_t(T) * H * 4);
            acc_dst = ep_vacc_;
            for (uint32_t e = 0; e < E; ++e) {
                if (saved_picks[e].empty()) continue;
                picks[e] = saved_picks[e];       // e_off/moe_idx_/moe_w_ still valid
                // Verify-loop eviction race (found by the CPU arbiter,
                // 2026-08-29: peer == ground truth, THIS recompute was the
                // corrupt side): fills here ran behind the STALE lagged
                // wave fence and earlier verify reads were never marked
                // in-flight, so a later verify fill could overwrite a slot
                // a still-running verify gemv was reading. Fresh barrier +
                // mark each read slot.
                wave_fence = q.ext_oneapi_submit_barrier();
                ensure(e);
                if (ec->slot_of[e] >= 0)
                    slot_wave[ec->slot_of[e]] = int32_t(cur_wave);
                if (auto err = compute_e(e); !err.empty()) return err;
            }
            acc_dst = moe_acc_;
            q.wait();
            std::vector<float> pa(uint64_t(T) * H), pb(uint64_t(T) * H);
            q.memcpy(pa.data(), ep_ret_, uint64_t(T) * H * 4).wait();
            q.memcpy(pb.data(), ep_vacc_, uint64_t(T) * H * 4).wait();
            double mx = 0, sum = 0;
            for (uint64_t i = 0; i < uint64_t(T) * H; ++i) {
                const double d = std::abs(double(pa[i]) - double(pb[i]));
                mx = std::max(mx, d); sum += d;
            }
            std::fprintf(stderr, "[epv] L%u peers=%zu max|d| %.6f mean %.8f\n",
                         L, size_t(0), mx, sum / (uint64_t(T) * H));
            // IE_G5_EPV_DUMP=<dir>: on the FIRST mismatch, dump everything a
            // CPU ground-truth arbiter needs — x, the peer expert list with
            // routing weights, pa (peer partial), pb (local recompute). The
            // arbiter (tools/glm5next_epv_ref) recomputes from the disk-
            // verified bank bytes and says WHICH side is wrong.
            static bool dumped = false;
            const char* dd = std::getenv("IE_G5_EPV_DUMP");
            if (dd && !dumped && mx > 1e-3) {
                dumped = true;
                std::vector<sycl::half> xh(uint64_t(T) * H);
                q.memcpy(xh.data(), x16_, uint64_t(T) * H * 2).wait();
                char p[512];
                auto wr = [&](const char* n, const void* d2, size_t bytes) {
                    std::snprintf(p, sizeof(p), "%s/%s", dd, n);
                    if (FILE* f = std::fopen(p, "wb")) {
                        std::fwrite(d2, 1, bytes, f);
                        std::fclose(f);
                    }
                };
                wr("x.f16", xh.data(), xh.size() * 2);
                wr("pa.f32", pa.data(), pa.size() * 4);
                wr("pb.f32", pb.data(), pb.size() * 4);
                std::vector<int32_t> el;
                std::vector<float> ew;
                for (uint32_t e = 0; e < E; ++e) {
                    if (saved_picks[e].empty()) continue;
                    el.push_back(int32_t(e));
                    ew.push_back(h_w_[e_off[e]]);
                }
                wr("experts.i32", el.data(), el.size() * 4);
                wr("weights.f32", ew.data(), ew.size() * 4);
                std::snprintf(p, sizeof(p), "%s/meta.txt", dd);
                if (FILE* f = std::fopen(p, "w")) {
                    std::fprintf(f, "L %u T %u H %u EF %u n_exp %zu clamp %g\n",
                                 L, T, H, EF, el.size(), double(clamp));
                    std::fclose(f);
                }
                std::fprintf(stderr, "[epv] DUMPED L%u to %s\n", L, dd);
            }
        }
    }
    if (pp_stream) {
        pp_stream_free_[int(L & 1)] = q.ext_oneapi_submit_barrier();
        static const bool pps_sync2 = std::getenv("IE_G5_PPS_SYNC") != nullptr;
        if (pps_sync2) {   // probe: drain + per-layer wall (numbers-invalid)
            q.wait();
            std::fprintf(stderr, "[pps] L%u dt %d/%d/%d ms %.1f\n", L,
                         int(w.gate_exps->dtype), int(w.up_exps->dtype),
                         int(w.down_exps->dtype),
                         std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - pps_call_t0)
                             .count() * 1e3);
        }
    }
    return {};
}

std::string Glm5NextModel::run_block(uint32_t L, uint32_t T, uint32_t start_pos) {
    sycl::queue& q = alloc_->queue();
    const Glm5NextLayer& w = layers_[L];
    const uint32_t H = cfg_.hidden, hc = cfg_.hc_count;
    const uint32_t DI = cfg_.n_q_heads * cfg_.kda_head_dim;
    const uint32_t NH = cfg_.n_q_heads, HD = cfg_.kda_head_dim;
    const uint32_t LAT = cfg_.kv_lora_rank;

    // ---- attention site: HC pre + norm --------------------------------------
    ds4_hyper_connection_norm(q, wide_a_, w.hc_attn_fn, w.hc_attn_base, w.hc_attn_scale,
                              w.attn_norm, post_, comb_, coll_, xn_,
                              T, H, hc, cfg_.hc_sinkhorn_iters,
                              cfg_.rms_eps, cfg_.hc_eps, cfg_.rms_eps, {});
    cast_fp32_to_fp16(q, xn_, x16_, uint64_t(T) * H, {});

    if (cfg_.is_full_attn(L)) {
        // ---- MLA (absorbed, NOPE) ------------------------------------------
        const uint32_t QL = cfg_.q_lora_rank, HDM = cfg_.key_len_mla;
        mm(x16_, w.q_a, T, QL, H, mla_qa16_, nullptr);
        rms_norm_f32w(q, mla_qa16_, w.q_a_norm, mla_qa16_, T, QL, cfg_.rms_eps, {});
        mm(mla_qa16_, w.q_b, T, NH * HDM, QL, nullptr, mla_qb_);
        k_mla_kabsorb(q, mla_qb_, w.k_b, mla_qlat_, T, NH, LAT, HDM, {});

        mm(x16_, w.kv_a, T, LAT, H, mla_kv16_, nullptr);
        rms_norm_f32w(q, mla_kv16_, w.kv_a_norm, mla_kv16_, T, LAT, cfg_.rms_eps, {});
        // append this chunk's latents to the cache (D2D; in-order queue
        // sequences it before the attend below — no host sync needed)
        sycl::half* lay_cache = lat_cache_ +
            (uint64_t(full_idx_[L]) * max_ctx_ + start_pos) * LAT;
        q.memcpy(lay_cache, mla_kv16_, uint64_t(T) * LAT * 2);

        const float scale = 1.f / std::sqrt(float(HDM));
        const uint32_t n_sel = cfg_.indexer_top_k + cfg_.indexer_kpool - 1;
        // IE_G5_FORCE_DENSE=1: dense causal attention with the SLM score buffer
        // sized to the whole context. Only usable while max_ctx floats fit in a
        // work-group's SLM (~8-16k), but that is enough to give the sparse path
        // a TRUE reference at a context where its coverage is already partial —
        // the only way to tell "my indexer selects badly" from "something else
        // degrades with depth".
        static const bool force_dense = std::getenv("IE_G5_FORCE_DENSE") != nullptr;
        if (!sparse_ || force_dense) {
            // ctx <= n_sel: every key is selected, so dense IS the sparse answer.
            k_mla_attend(q, mla_qlat_, lat_cache_ + uint64_t(full_idx_[L]) * max_ctx_ * LAT,
                         mla_att_, T, NH, LAT, start_pos, scale,
                         force_dense ? max_ctx_ : n_sel, {});
        } else {
            // ---- DSA lightning indexer (modeling_glm5_next.py Glm5NextIndexer) --
            const uint32_t IH = cfg_.indexer_n_heads, IHD = cfg_.indexer_head_dim,
                           KP = cfg_.indexer_kpool;
            const uint32_t fi = uint32_t(full_idx_[L]);
            sycl::half* prow  = pool_key_  + uint64_t(fi) * n_pools_ * IHD;
            sycl::half* kroll = idx_kroll_ + uint64_t(fi) * KP * IHD;
            sycl::half* groll = idx_groll_ + uint64_t(fi) * KP * IHD;
            const Glm5DW dw_k{w.idx_k, nullptr, nullptr};
            const Glm5DW dw_g{w.idx_gate, nullptr, nullptr};
            const Glm5DW dw_q{w.idx_q_b, nullptr, nullptr};
            // indexer key (LayerNorm WITH bias) and compressor gate for this chunk
            mm(x16_, dw_k, T, IHD, H, idx_k_, nullptr);
            k_g5_idx_lnb(q, idx_k_, w.idx_k_norm_w, w.idx_k_norm_b, idx_k_, T, IHD,
                         cfg_.ln_eps, {});
            mm(x16_, dw_g, T, IHD, H, idx_gate_, nullptr);
            // build every pool whose LAST token landed in this chunk (a pool is
            // immutable once complete, so each is built exactly once, ever)
            const uint32_t p_lo = start_pos / KP, p_hi = (start_pos + T) / KP;
            if (p_hi > p_lo)
                k_g5_pool_key(q, idx_k_, idx_gate_, kroll, groll, w.idx_ape, prow,
                              p_lo, p_hi - p_lo, KP, IHD, start_pos, {});
            // Carry the still-open pool's members into the next chunk. Only the
            // members that arrived in THIS chunk get copied: the open pool can
            // start before start_pos (T=1 decode at pos 5 with KP=4 leaves two
            // earlier members), and those were written by the chunks that
            // produced them. Computing the loop from `open` alone underflows
            // `T - open` when open > T and walks the source pointer off the map.
            // spec-verify: keep this window's key/gate so a PARTIAL accept can
            // put the committed rows back into the roll (commit_verify)
            if (spec_verify_ && T <= kSpecMax) {
                q.memcpy(sv_idx_k_ + uint64_t(fi) * kSpecMax * IHD, idx_k_, uint64_t(T) * IHD * 2);
                q.memcpy(sv_idx_g_ + uint64_t(fi) * kSpecMax * IHD, idx_gate_, uint64_t(T) * IHD * 2);
            }
            const uint32_t end = start_pos + T;
            const uint32_t open_base = end - (end % KP);   // first token of the open pool
            for (uint32_t j = std::max(open_base, start_pos); j < end; ++j) {
                q.memcpy(kroll + uint64_t(j % KP) * IHD,
                         idx_k_ + uint64_t(j - start_pos) * IHD, IHD * 2);
                q.memcpy(groll + uint64_t(j % KP) * IHD,
                         idx_gate_ + uint64_t(j - start_pos) * IHD, IHD * 2);
            }
            // indexer query (from the q_a LoRA residual, not x) + per-head weights
            mm(mla_qa16_, dw_q, T, IH * IHD, QL, nullptr, idx_q_);
            k_g5_idx_wproj(q, x16_, w.idx_proj, idx_w_, T, H, IH, {});
            k_g5_idx_pos(q, idx_pos_, T, start_pos, {});
            // score -> top-k pools, in strips so the score workspace stays counted
            const uint32_t n_keys = (start_pos + T + KP - 1) / KP;
            const float sms = 1.f / std::sqrt(float(IHD));
            const float wsc = 1.f / std::sqrt(float(IH));
            // ds4_indexer_topk emits min(index_topk, n_keys) per query, so early
            // in a sequence (few complete pools) its ROW STRIDE is not select_k_.
            // Reading it as if it were hands the gather wild token ids.
            const uint32_t sk = std::min(select_k_, n_keys);
            for (uint32_t t0 = 0; t0 < T; t0 += kIdxStrip) {
                const uint32_t ts = std::min<uint32_t>(kIdxStrip, T - t0);
                ds4_indexer_score(q, idx_q_ + uint64_t(t0) * IH * IHD, prow,
                                  idx_w_ + uint64_t(t0) * IH, idx_scores_,
                                  ts, IH, IHD, n_keys, sms, wsc, {});
                ds4_indexer_topk(q, idx_scores_, idx_pos_ + t0,
                                 idx_sel_ + uint64_t(t0) * sk,
                                 ts, n_keys, sk, KP, {});
            }
            k_g5_pool_expand(q, idx_sel_, idx_pos_, idx_tok_, idx_nv_, T, sk, select_k_,
                             KP, n_sel_, start_pos + T, {});
            // kIdxStrip-row strips: the partials buffer holds one strip. Every
            // query row's computation is independent, so this is byte-identical
            // to the single full-T launch it replaces.
            for (uint32_t t0 = 0; t0 < T; t0 += kIdxStrip) {
                const uint32_t ts = std::min<uint32_t>(kIdxStrip, T - t0);
                k_g5_mla_attend_sel(q, mla_qlat_ + uint64_t(t0) * NH * LAT,
                                    lat_cache_ + uint64_t(fi) * max_ctx_ * LAT,
                                    idx_tok_ + uint64_t(t0) * n_sel_, idx_nv_ + t0,
                                    mla_att_ + uint64_t(t0) * NH * LAT, idx_part_,
                                    ts, NH, LAT, n_sel_, start_pos + T, scale, {});
            }
        }
        k_mla_vabsorb(q, mla_att_, w.v_b, mla_o16_, T, NH, LAT, cfg_.value_len_mla, {});
        mm(mla_o16_, w.attn_out, T, H, NH * cfg_.value_len_mla, nullptr, mix_out_);
    } else {
        // ---- KDA -----------------------------------------------------------
        const int32_t li = lin_idx_[L];
        mm_dual(x16_, w.kda_q, w.kda_k, T, DI, H,
                kda_h16a_, nullptr, kda_h16b_, nullptr);
        mm(x16_, w.kda_v, T, DI, H, kda_h16c_, nullptr);
        const bool sv = spec_verify_ && T <= kSpecMax;
        if (sv) {   // conv INPUT rows, per stream (commit_verify re-runs the conv)
            sycl::half* ci = sv_ci_ + uint64_t(li) * 3 * kSpecMax * DI;
            q.memcpy(ci,                    kda_h16a_, uint64_t(T) * DI * 2);
            q.memcpy(ci + kSpecMax * DI,    kda_h16b_, uint64_t(T) * DI * 2);
            q.memcpy(ci + 2 * kSpecMax * DI, kda_h16c_, uint64_t(T) * DI * 2);
        }
        // per-stream causal conv (fused SiLU) with per-stream state slices.
        // The conv kernel must NOT run in place (items read earlier x rows);
        // Write the half-rounded result directly to float, saving three cast launches.
        sycl::half* cs = dn_.conv_state_ptr() +
                         uint64_t(li) * dn_.conv_elems_per_layer();
        const uint32_t cs_per = (cfg_.conv_kernel - 1) * DI;
        depthwise_conv1d_causal_f32(q, kda_h16a_, w.conv_q, cs,              kda_f32a_, T, DI, cfg_.conv_kernel, {});
        depthwise_conv1d_causal_f32(q, kda_h16b_, w.conv_k, cs + cs_per,     kda_f32b_, T, DI, cfg_.conv_kernel, {});
        depthwise_conv1d_causal_f32(q, kda_h16c_, w.conv_v, cs + 2 * cs_per, kda_f32c_, T, DI, cfg_.conv_kernel, {});
        // L2 norm per head (reference constant 1e-6); q also takes 1/sqrt(hd)
        kda_l2norm(q, kda_f32a_, kda_f32a_, T * NH, HD, 1.f / std::sqrt(float(HD)), 1e-6f, {});
        kda_l2norm(q, kda_f32b_, kda_f32b_, T * NH, HD, 1.f, 1e-6f, {});
        // decay gate (low-rank) + beta. f_a and g_a are the same  H→128
        // shape on the same x16_; stash g_a in moe_h16_ (free until FFN)
        // so g_b after the scan does not re-read the dense Q8 row.
        mm_dual(x16_, w.f_a, w.g_a, T, 128, H, kda_lo16_, nullptr, moe_h16_, nullptr);
        mm(kda_lo16_, w.f_b, T, DI, 128, nullptr, kda_pre_);
        kda_gate(q, kda_pre_, w.dt_bias, w.ssm_a, kda_g_, T, NH, HD,
                 cfg_.kda_gate_lower_bound, {});
        mm(x16_, w.beta, T, NH, H, nullptr, kda_beta_);
        k_sigmoid(q, kda_beta_, uint64_t(T) * NH, {});
        if (sv) {   // scan INPUT rows (post-l2norm/gate/sigmoid)
            q.memcpy(sv_q_ + uint64_t(li) * kSpecMax * DI, kda_f32a_, uint64_t(T) * DI * 4);
            q.memcpy(sv_k_ + uint64_t(li) * kSpecMax * DI, kda_f32b_, uint64_t(T) * DI * 4);
            q.memcpy(sv_v_ + uint64_t(li) * kSpecMax * DI, kda_f32c_, uint64_t(T) * DI * 4);
            q.memcpy(sv_g_ + uint64_t(li) * kSpecMax * DI, kda_g_, uint64_t(T) * DI * 4);
            q.memcpy(sv_beta_ + uint64_t(li) * kSpecMax * NH, kda_beta_, uint64_t(T) * NH * 4);
        }
        // the scan
        float* st = dn_.state_ptr() + uint64_t(li) * dn_.state_elems_per_layer();
        kda_recurrence(q, kda_f32a_, kda_f32b_, kda_f32c_, kda_g_, kda_beta_,
                       st, kda_out_, 1, T, NH, HD, HD, {});
        // output gate: per-head RMS * sigmoid(g_b(g_a(x)))
        mm(moe_h16_, w.g_b, T, DI, 128, kda_z16_, nullptr);
        gated_rms_norm(q, kda_out_, kda_z16_, w.ssm_norm, kda_y16_,
                       T * NH, HD, cfg_.rms_eps, /*sigmoid_gate=*/true, {});
        mm(kda_y16_, w.kda_o, T, H, DI, nullptr, mix_out_);
    }

    // HC post (comb^T · streams + post · sub) into the other wide buffer.
    ds4_hc_mix(q, wide_a_, post_, comb_, mix_out_, wide_b_, T, H, hc, {});
    std::swap(wide_a_, wide_b_);
    if (!dump_h_.empty()) {   // IE_G5_DUMP_WIDE (forward_range): async D2H on the in-order q
        const uint64_t n = uint64_t(T) * hc * H;
        if (dump_cur_ + n <= dump_h_.size()) {
            q.memcpy(dump_h_.data() + dump_cur_, wide_a_, n * 4);
            dump_cur_ += n;
            dump_hdr_.insert(dump_hdr_.end(), {L, 0u, T, start_pos, uint32_t(n)});
        }
    }
    {   // Diagnostic (IE_G5_NAN_PROBE): mixer-stage check, pairs with the
        // per-layer probe in forward_range to split mixer vs FFN.
        static const bool nan_probe = std::getenv("IE_G5_NAN_PROBE") != nullptr;
        static int left = 2;
        if (nan_probe && left > 0 && T == 1) {
            q.wait();
            std::vector<float> hw(uint64_t(hc) * H);
            q.memcpy(hw.data(), wide_a_, hw.size() * 4).wait();
            uint32_t nf = 0;
            for (float v : hw) if (!std::isfinite(v)) ++nf;
            std::vector<sycl::half> mo(H);
            q.memcpy(mo.data(), mix_out_, H * 2).wait();
            uint32_t nfm = 0;
            for (auto v : mo) if (!std::isfinite(float(v))) ++nfm;
            if (nf || nfm) {
                std::fprintf(stderr, "[nan-probe] pos %u layer %u AFTER MIXER: wide non-finite %u, mixer_out non-finite %u\n",
                             start_pos, L, nf, nfm);
                --left;
            }
        }
    }

    // ---- FFN site -----------------------------------------------------------
    ds4_hyper_connection_norm(q, wide_a_, w.hc_ffn_fn, w.hc_ffn_base, w.hc_ffn_scale,
                              w.ffn_norm, post_, comb_, coll_, xn_,
                              T, H, hc, cfg_.hc_sinkhorn_iters,
                              cfg_.rms_eps, cfg_.hc_eps, cfg_.rms_eps, {});
    cast_fp32_to_fp16(q, xn_, x16_, uint64_t(T) * H, {});

    if (cfg_.is_dense_layer(L)) {
        const uint32_t FF = cfg_.ffn;
        mm(x16_, w.ffn_gate, T, FF, H, nullptr, moe_gu_[0]);
        mm(x16_, w.ffn_up,   T, FF, H, nullptr, moe_gu_[1]);
        ds4_swiglu_clamped(q, moe_gu_[0], moe_gu_[1], moe_gu_[0],
                           uint64_t(T) * FF, cfg_.swiglu_clamp_exp[L], {});
        cast_fp32_to_fp16(q, moe_gu_[0], moe_h16_, uint64_t(T) * FF, {});
        mm(moe_h16_, w.ffn_down, T, H, FF, nullptr, moe_acc_);
    } else {
        if (auto e = run_moe(L, T); !e.empty()) return e;
    }

    ds4_hc_mix(q, wide_a_, post_, comb_, moe_acc_, wide_b_, T, H, hc, {});
    std::swap(wide_a_, wide_b_);
    if (!dump_h_.empty()) {   // IE_G5_DUMP_WIDE (forward_range): async D2H on the in-order q
        const uint64_t n = uint64_t(T) * hc * H;
        if (dump_cur_ + n <= dump_h_.size()) {
            q.memcpy(dump_h_.data() + dump_cur_, wide_a_, n * 4);
            dump_cur_ += n;
            dump_hdr_.insert(dump_hdr_.end(), {L, 1u, T, start_pos, uint32_t(n)});
        }
    }
    {
        static const bool pps_sync = std::getenv("IE_G5_PPS_SYNC") != nullptr;
        if (pps_sync && T > 4) {   // probe: per-BLOCK drain+wall (invalid #s)
            static thread_local auto bt0 = std::chrono::steady_clock::now();
            q.wait();
            auto bt1 = std::chrono::steady_clock::now();
            std::fprintf(stderr, "[ppb-blk] L%u ms %.1f\n", L,
                         std::chrono::duration<double>(bt1 - bt0).count() * 1e3);
            bt0 = std::chrono::steady_clock::now();
        }
    }
    return {};
}

std::string Glm5NextModel::mtp_step(const float* hidden_row, int32_t tok, uint32_t pos) {
    if (!mtp_loaded()) return "glm5next mtp: not loaded on this stage";
    if (pos >= max_ctx_ || pos < mtp_base_) return "glm5next mtp: bad pos";
    const uint32_t n_sel = cfg_.indexer_top_k + cfg_.indexer_kpool - 1;
    // The MTP head attends DENSELY over its own latent cache, so v0 REFUSED
    // once this generation had taken more than n_sel MTP steps (the cap is
    // relative to mtp_base_, NOT to the prompt length — a long prefill alone
    // does not trigger it, and the runner treats the error as fatal, so it was
    // a hard stop on long GENERATIONS rather than the silent no-op first
    // supposed; the 0/64 acceptance at ctx 32768 had a different cause, the
    // out-of-bounds draft seed in the runner). It does not need the full
    // history: this is a
    // DRAFTER, stage B's argmax decides every token, and a rejected draft costs
    // only a redo — so an approximate drafter is lossless BY CONSTRUCTION, and
    // next-token signal is concentrated in recent context anyway. Attend over
    // the most recent n_sel positions instead of refusing. (The main path's 46
    // layers use the real DSA indexer; giving the MTP block its own indexer +
    // pool keys is the exact-but-expensive alternative.)
    const uint32_t mtp_w0 = (pos + 1 > mtp_base_ + n_sel) ? (pos + 1 - n_sel) : mtp_base_;
    sycl::queue& q = alloc_->queue();
    const uint32_t H = cfg_.hidden, NH = cfg_.n_q_heads, LAT = cfg_.kv_lora_rank;
    const uint32_t HDM = cfg_.key_len_mla, QL = cfg_.q_lora_rank;
    const uint32_t n_tf = cfg_.n_transformer_layers();
    const Glm5NextLayer& w = layers_[n_tf];

    // fusion: x = eh_proj([RMS(embed(tok), enorm) | RMS(hidden, hnorm)])
    {
        const sycl::half* emb = token_embd;
        sycl::half* e16 = mtp_e16_;
        const uint64_t row = uint64_t(tok) * H;
        ie::ps(q, "g5_mtp_embed", [&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(H), [=](sycl::id<1> i) { e16[i] = emb[row + i]; });
        });
    }
    rms_norm_f32w(q, mtp_e16_, mtp_enorm_, mtp_cat_, 1, H, cfg_.rms_eps, {});
    cast_fp32_to_fp16(q, hidden_row, mtp_x16_, H, {});
    rms_norm_f32w(q, mtp_x16_, mtp_hnorm_, mtp_cat_ + H, 1, H, cfg_.rms_eps, {});
    mm(mtp_cat_, mtp_eh_, 1, H, 2 * H, mtp_x16_, mtp_x_);

    auto add_row = [&](const float* src) {
        float* dst = mtp_x_;
        ie::ps(q, "g5_mtp_add", [&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(H), [=](sycl::id<1> i) { dst[i] += src[i]; });
        });
    };

    // attn sublayer: x += MLA(RMS(x)) over the MTP's own latent suffix.
    rms_norm_f32w(q, mtp_x16_, w.attn_norm, x16_, 1, H, cfg_.rms_eps, {});
    mm(x16_, w.q_a, 1, QL, H, mla_qa16_, nullptr);
    rms_norm_f32w(q, mla_qa16_, w.q_a_norm, mla_qa16_, 1, QL, cfg_.rms_eps, {});
    mm(mla_qa16_, w.q_b, 1, NH * HDM, QL, nullptr, mla_qb_);
    k_mla_kabsorb(q, mla_qb_, w.k_b, mla_qlat_, 1, NH, LAT, HDM, {});
    mm(x16_, w.kv_a, 1, LAT, H, mla_kv16_, nullptr);
    rms_norm_f32w(q, mla_kv16_, w.kv_a_norm, mla_kv16_, 1, LAT, cfg_.rms_eps, {});
    q.memcpy(mtp_lat_ + uint64_t(pos) * LAT, mla_kv16_, uint64_t(LAT) * 2);
    k_mla_attend(q, mla_qlat_, mtp_lat_ + uint64_t(mtp_w0) * LAT, mla_att_,
                 1, NH, LAT, pos - mtp_w0, 1.f / std::sqrt(float(HDM)), n_sel, {});
    k_mla_vabsorb(q, mla_att_, w.v_b, mla_o16_, 1, NH, LAT, cfg_.value_len_mla, {});
    mm(mla_o16_, w.attn_out, 1, H, NH * cfg_.value_len_mla, nullptr, mix_out_);
    add_row(mix_out_);

    // ffn sublayer: x += (shexp + routed MoE)(RMS(x)).
    cast_fp32_to_fp16(q, mtp_x_, mtp_x16_, H, {});
    rms_norm_f32w(q, mtp_x16_, w.ffn_norm, x16_, 1, H, cfg_.rms_eps, {});
    cast_fp16_to_fp32(q, x16_, xn_, H, {});
    if (auto e = run_moe(n_tf, 1); !e.empty()) return e;
    add_row(moe_acc_);

    // head: logits = lm_head(RMS(x, shared_head_norm)).
    cast_fp32_to_fp16(q, mtp_x_, mtp_x16_, H, {});
    rms_norm_f32w(q, mtp_x16_, mtp_shn_, x16_, 1, H, cfg_.rms_eps, {});
    if (lm_head.qs)
        gemv_q8_0_soa_f16_g(q, x16_, lm_head.qs, lm_head.d, logits_, H, cfg_.vocab, {});
    else
        gemv_fp16(q, x16_, lm_head.w, logits_, H, cfg_.vocab, {});
    q.wait();
    mtp_len_ = pos + 1;
    return {};
}

std::string Glm5NextModel::snapshot_state() {
    sycl::queue& q = alloc_->queue();
    const uint64_t sb = uint64_t(dn_.config().n_layers_linear) * dn_.state_elems_per_layer() * 4;
    const uint64_t cb = uint64_t(dn_.config().n_layers_linear) * dn_.conv_elems_per_layer() * 2;
    q.memcpy(snap_dn_, dn_.state_ptr(), sb);
    q.memcpy(snap_conv_, dn_.conv_state_ptr(), cb);
    if (sparse_ && snap_kroll_) {   // DSA open-pool roll (see the header note)
        q.memcpy(snap_kroll_, idx_kroll_, roll_bytes_);
        q.memcpy(snap_groll_, idx_groll_, roll_bytes_);
    }
    q.wait();
    snap_kv_len_ = kv_len_;
    snap_mtp_len_ = mtp_len_;
    return {};
}

std::string Glm5NextModel::commit_verify(uint32_t n) {
    if (!spec_verify_ || spec_T_ == 0) return "glm5next commit: no spec-verify forward recorded";
    if (n == 0 || n > spec_T_) return "glm5next commit: bad n";
    if (n == spec_T_) return {};   // full accept: live state IS the committed state
    sycl::queue& q = alloc_->queue();
    const uint32_t DI = cfg_.n_q_heads * cfg_.kda_head_dim;
    const uint32_t NH = cfg_.n_q_heads, HD = cfg_.kda_head_dim;
    // Rewind to the pre-verify snapshot, then re-advance exactly n rows from
    // the captured per-layer inputs — the same kernels in the same order, so
    // the committed state is bit-identical to a T=n forward's.
    const uint64_t sb = uint64_t(dn_.config().n_layers_linear) * dn_.state_elems_per_layer() * 4;
    const uint64_t cb = uint64_t(dn_.config().n_layers_linear) * dn_.conv_elems_per_layer() * 2;
    q.memcpy(dn_.state_ptr(), snap_dn_, sb);
    q.memcpy(dn_.conv_state_ptr(), snap_conv_, cb);
    if (sparse_ && snap_kroll_) {
        // Drop the rejected rows' key/gate, then RE-INSTALL the accepted ones:
        // the restore puts back the PRE-verify roll, which predates rows
        // [spec_pos_, spec_pos_+n). kSpecMax (8) exceeds kpool (4), so a window
        // can wrap the roll and a selective restore is not enough — replay the
        // committed rows from the spec capture instead.
        q.memcpy(idx_kroll_, snap_kroll_, roll_bytes_);
        q.memcpy(idx_groll_, snap_groll_, roll_bytes_);
        const uint32_t KP = cfg_.indexer_kpool, IHD = cfg_.indexer_head_dim;
        const uint32_t cend = spec_pos_ + n;
        const uint32_t ob = cend - (cend % KP);
        for (uint32_t L2 = layer_lo_; L2 < layer_hi_; ++L2) {
            if (full_idx_[L2] < 0) continue;
            const uint32_t fi = uint32_t(full_idx_[L2]);
            sycl::half* kr = idx_kroll_ + uint64_t(fi) * KP * IHD;
            sycl::half* gr = idx_groll_ + uint64_t(fi) * KP * IHD;
            const sycl::half* sk = sv_idx_k_ + uint64_t(fi) * kSpecMax * IHD;
            const sycl::half* sg = sv_idx_g_ + uint64_t(fi) * kSpecMax * IHD;
            for (uint32_t j = std::max(ob, spec_pos_); j < cend; ++j) {
                q.memcpy(kr + uint64_t(j % KP) * IHD, sk + uint64_t(j - spec_pos_) * IHD, uint64_t(IHD) * 2);
                q.memcpy(gr + uint64_t(j % KP) * IHD, sg + uint64_t(j - spec_pos_) * IHD, uint64_t(IHD) * 2);
            }
        }
    }
    const uint32_t cs_per = (cfg_.conv_kernel - 1) * DI;
    for (uint32_t L = layer_lo_; L < layer_hi_; ++L) {
        const int32_t li = lin_idx_[L];
        if (li < 0) continue;
        const Glm5NextLayer& w = layers_[L];
        sycl::half* cs = dn_.conv_state_ptr() + uint64_t(li) * dn_.conv_elems_per_layer();
        sycl::half* ci = sv_ci_ + uint64_t(li) * 3 * kSpecMax * DI;
        depthwise_conv1d_causal(q, ci,                     w.conv_q, cs,              kda_y16_, n, DI, cfg_.conv_kernel, {});
        depthwise_conv1d_causal(q, ci + kSpecMax * DI,     w.conv_k, cs + cs_per,     kda_y16_, n, DI, cfg_.conv_kernel, {});
        depthwise_conv1d_causal(q, ci + 2 * kSpecMax * DI, w.conv_v, cs + 2 * cs_per, kda_y16_, n, DI, cfg_.conv_kernel, {});
        float* st = dn_.state_ptr() + uint64_t(li) * dn_.state_elems_per_layer();
        kda_recurrence(q, sv_q_ + uint64_t(li) * kSpecMax * DI,
                       sv_k_ + uint64_t(li) * kSpecMax * DI,
                       sv_v_ + uint64_t(li) * kSpecMax * DI,
                       sv_g_ + uint64_t(li) * kSpecMax * DI,
                       sv_beta_ + uint64_t(li) * kSpecMax * NH,
                       st, kda_out_, 1, n, NH, HD, HD, {});
    }
    q.wait();
    kv_len_ = spec_pos_ + n;
    return {};
}

std::string Glm5NextModel::restore_state() {
    sycl::queue& q = alloc_->queue();
    const uint64_t sb = uint64_t(dn_.config().n_layers_linear) * dn_.state_elems_per_layer() * 4;
    const uint64_t cb = uint64_t(dn_.config().n_layers_linear) * dn_.conv_elems_per_layer() * 2;
    q.memcpy(dn_.state_ptr(), snap_dn_, sb);
    q.memcpy(dn_.conv_state_ptr(), snap_conv_, cb);
    if (sparse_ && snap_kroll_) {
        q.memcpy(idx_kroll_, snap_kroll_, roll_bytes_);
        q.memcpy(idx_groll_, snap_groll_, roll_bytes_);
    }
    q.wait();
    kv_len_ = snap_kv_len_;
    mtp_len_ = snap_mtp_len_;
    return {};
}

void Glm5NextModel::PreadPool::start(int n) {
    if (!ws.empty()) return;
    for (int i = 0; i < n; ++i) ws.emplace_back([this] {
        pin_copy_thread();
        for (;;) {
            size_t j;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop || next < njobs; });
                if (stop) return;
                j = next++;
            }
            const Job& job = jobs[j];
            uint64_t got = 0;
            bool ok = true;
            while (got < job.len) {
                ssize_t r = pread(job.fd, job.dst + got, job.len - got, job.off + got);
                if (r < 0) { if (errno == EINTR) continue; ok = false; break; }
                if (r == 0) { ok = false; break; }   // EOF short of len
                got += uint64_t(r);
            }
            {
                std::lock_guard<std::mutex> lk(mu);
                if (!ok) fail = true;
                if (++done == njobs) cvd.notify_all();
            }
        }
    });
}
bool Glm5NextModel::PreadPool::run(const Job* j, size_t n) {
    if (!n) return true;
    std::unique_lock<std::mutex> lk(mu);
    jobs = j; njobs = n; next = 0; done = 0; fail = false;
    cv.notify_all();
    cvd.wait(lk, [&] { return done == njobs; });
    return !fail;
}
void Glm5NextModel::PreadPool::shutdown() {
    {
        std::lock_guard<std::mutex> lk(mu);
        stop = true;
    }
    cv.notify_all();
    for (auto& w : ws) if (w.joinable()) w.join();
    ws.clear();
}

std::string Glm5NextModel::pread_fill_(int gfd, uint64_t goff, const uint8_t* gs, uint64_t gsl,
                                       int ufd, uint64_t uoff, const uint8_t* us, uint64_t usl,
                                       int dfd, uint64_t doff, const uint8_t* ds2, uint64_t dsl,
                                       uint8_t* dst, PreadPool* pool, bool serial_cp) {
    static const bool use_pread = [] {
        const char* v = std::getenv("IE_G5_PREAD");
        return !(v && v[0] == '0');
    }();
    if (!use_pread || gfd < 0 || ufd < 0 || dfd < 0) {   // memcpy path
        if (serial_cp) {   // off-thread caller: cpool_ rendezvous is unshareable
            std::memcpy(dst,             gs,  gsl);
            std::memcpy(dst + gsl,       us,  usl);
            std::memcpy(dst + gsl + usl, ds2, dsl);
            return {};
        }
        cpool_.copy(dst,             gs,  gsl);
        cpool_.copy(dst + gsl,       us,  usl);
        cpool_.copy(dst + gsl + usl, ds2, dsl);
        return {};
    }
    PreadPool& prp = pool ? *pool : prp_;
    prp.start(6);
    constexpr uint64_t SL = 2ull << 20;
    PreadPool::Job jobs[64];
    size_t n = 0;
    auto seg = [&](int fd, uint64_t off, uint64_t len, uint8_t* d) {
        for (uint64_t o = 0; o < len; o += SL)
            jobs[n++] = {fd, off + o, std::min(SL, len - o), d + o};
    };
    seg(gfd, goff, gsl, dst);
    seg(ufd, uoff, usl, dst + gsl);
    seg(dfd, doff, dsl, dst + gsl + usl);
    if (!prp.run(jobs, n))
        return "glm5next: pread fill failed (short read / IO error) — refusing DMA";
    return {};
}

// Peer-side prefetch (EpThread only — the sole ep_cache_ mutator): warm
// ep_cache_[L] with predicted experts of OUR parity through OUR fill
// thread's pf path (our PCIe link). Bank sources live on the layer's OWNER.
void Glm5NextModel::ep_prefetch(uint32_t L, const int32_t* p8) {
    if (!ep_peer_ || L >= ep_cache_.size() || !ep_cache_[L].base) return;
    ECache& ec = ep_cache_[L];
    if (ec.pf_pending.empty()) return;
    const Glm5NextLayer& w = ep_peer_->layers_[L];
    if (!w.gate_exps) return;
    const uint32_t E = cfg_.n_experts;
    sycl::queue& pq = alloc_->queue();
    // release prefetches whose H2D is COMPLETE (same rule as run_moe — a
    // released slot's next hit reader carries no dep, so submitted-only
    // release races the in-flight DMA)
    for (uint32_t s = 0; s < ec.pf_pending.size(); ++s)
        if (ec.pf_pending[s] &&
            ec.pf_ev[s].wait_for(std::chrono::seconds(0)) == std::future_status::ready &&
            ec.pf_ev[s].get().get_info<sycl::info::event::command_execution_status>()
                == sycl::info::event_command_status::complete)
            ec.pf_pending[s] = 0;
    sycl::event fence;
    bool have_fence = false;
    std::vector<uint8_t> pred(E, 0);
    for (uint32_t k = 0; k < 8; ++k)
        if (p8[k] >= 0) pred[p8[k]] = 1;
    const uint64_t gsl = w.gate_exps->nbytes / E;
    const uint64_t usl = w.up_exps->nbytes / E;
    const uint64_t dsl = w.down_exps->nbytes / E;
    for (uint32_t k = 0; k < 8; ++k) {
        const int32_t pe = p8[k];
        if (pe < 0 || ec.slot_of[pe] >= 0) continue;
        if ((uint32_t(pe) & 1u) != ep_parity_) continue;   // our half only
        uint32_t victim = UINT32_MAX;
        uint64_t oldest = UINT64_MAX;
        for (uint32_t s = 0; s < ec.expert_in.size(); ++s) {
            if (ec.pf_pending[s]) continue;
            if (ec.expert_in[s] >= 0 && pred[ec.expert_in[s]]) continue;
            if (ec.last_use[s] < oldest) { oldest = ec.last_use[s]; victim = s; }
        }
        if (victim == UINT32_MAX) break;
        if (!have_fence) { fence = pq.ext_oneapi_submit_barrier(); have_fence = true; }
        if (ec.expert_in[victim] >= 0)
            ec.slot_of[ec.expert_in[victim]] = -1;
        ec.expert_in[victim] = pe;
        ec.slot_of[pe] = int16_t(victim);
        ec.last_use[victim] = ++ep_clock_;
        FillReq r;
        r.dst = ec.base + uint64_t(victim) * ec.slot_bytes + ec.gate_off;
        r.g = w.gate_src + uint64_t(pe) * gsl;
        r.u = w.up_src + uint64_t(pe) * usl;
        r.d2 = w.down_src + uint64_t(pe) * dsl;
        r.gsl = gsl; r.usl = usl; r.dsl = dsl;
        r.gfd = w.gate_fd; r.goff = w.gate_foff + uint64_t(pe) * gsl;
        r.ufd = w.up_fd;   r.uoff = w.up_foff + uint64_t(pe) * usl;
        r.dfd = w.down_fd; r.doff = w.down_foff + uint64_t(pe) * dsl;
        r.bounce = true;
        r.pf = true;
        r.fence = fence;
        ec.pf_ev[victim] = fill_.push(std::move(r));
        ec.pf_pending[victim] = 1;
        ++pf_issued_;
    }
}

void Glm5NextModel::EpThread::start(Glm5NextModel* m) {
    if (th.joinable()) return;
    th = std::thread([this, m] {
        for (;;) {
            EpReq r;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop || !reqs.empty(); });
                if (stop && reqs.empty()) return;
                r = std::move(reqs.front());
                reqs.pop_front();
            }
            if (r.pf_only) {
                m->ep_prefetch(r.L, r.pf_pred);
                r.done.set_value(sycl::event{});
                continue;
            }
            sycl::event done_ev;
            auto err = m->ep_partial(*r.w, r.L, r.T, r.exps, r.ecnt, r.n_exp,
                                     r.idx_flat, r.w_flat, r.npick,
                                     r.x_host, r.x_ev, r.ret_dep, r.ret_host,
                                     r.x_pushed, r.ret_dev, &done_ev);
            if (!err.empty()) std::fprintf(stderr, "[glm5next] ep worker: %s\n", err.c_str());
            r.done.set_value(done_ev);
        }
    });
}
std::shared_future<sycl::event> Glm5NextModel::EpThread::push(EpReq&& r) {
    auto fut = r.done.get_future().share();
    {
        std::lock_guard<std::mutex> lk(mu);
        reqs.push_back(std::move(r));
    }
    cv.notify_one();
    return fut;
}
Glm5NextModel::EpThread::~EpThread() {
    { std::lock_guard<std::mutex> lk(mu); stop = true; }
    cv.notify_all();
    if (th.joinable()) th.join();
}

std::string Glm5NextModel::warm_banks() {
    if (!copyq_) return {};
    const uint64_t sz = 128ull << 20;
    uint8_t* scratch = static_cast<uint8_t*>(sycl::malloc_device(sz, *copyq_));
    if (!scratch) return "glm5next warm_banks: scratch alloc failed";
    for (uint32_t L = layer_lo_; L < cfg_.n_layers; ++L) {
        const Glm5NextLayer& w = layers_[L];
        if (!w.bank_pinned) continue;
        const uint64_t total =
            w.gate_exps->nbytes + w.up_exps->nbytes + w.down_exps->nbytes;
        for (uint64_t off = 0; off < total; off += sz)
            copyq_->memcpy(scratch, w.bank_pin + off, std::min(sz, total - off));
    }
    // The host-USM staging pins lose their device mappings the same way
    // (a later stage's allocations evict them; H2Ds then re-fault at
    // ~0.3 GB/s — the pp-stream stage-A collapse, 2026-08-29). Sweep the
    // FULL allocation of each: mapping re-establishment is per-page.
    auto sweep = [&](const uint8_t* p, uint64_t n) {
        if (!p) return;
        for (uint64_t off = 0; off < n; off += sz)
            copyq_->memcpy(scratch, p + off, std::min(sz, n - off));
    };
    for (auto* p : pin_ring_) sweep(p, pin_sz_);
    for (auto* p : pp_pin_) sweep(p, kPpPinSz);
    for (auto* p : pf_pin_) sweep(p, std::max<uint64_t>(pin_sz_, 20ull << 20));
    copyq_->wait();
    sycl::free(scratch, *copyq_);
    return {};
}

std::string Glm5NextModel::warm_cache(const char* profile_path) {
    if (!copyq_ || ecache_.empty()) return {};
    const uint32_t E = cfg_.n_experts;
    std::vector<uint64_t> counts(uint64_t(cfg_.n_layers) * E);
    FILE* f = std::fopen(profile_path, "rb");
    if (!f) return std::string("glm5next warm_cache: cannot open ") + profile_path;
    const bool ok = std::fread(counts.data(), 8, counts.size(), f) == counts.size();
    std::fclose(f);
    if (!ok) return "glm5next warm_cache: short profile (want u64[n_layers x E])";
    std::vector<std::shared_future<sycl::event>> pend;
    uint64_t bytes = 0;
    for (uint32_t L = layer_lo_; L < uint32_t(ecache_.size()); ++L) {
        ECache& ec = ecache_[L];
        if (!ec.base || !layers_[L].gate_exps) continue;
        const Glm5NextLayer& w = layers_[L];
        const uint64_t gsl = w.gate_exps->nbytes / E;
        const uint64_t usl = w.up_exps->nbytes / E;
        const uint64_t dsl = w.down_exps->nbytes / E;
        std::vector<uint32_t> rank(E);
        for (uint32_t e = 0; e < E; ++e) rank[e] = e;
        const uint64_t* row = counts.data() + uint64_t(L) * E;
        std::stable_sort(rank.begin(), rank.end(),
                         [&](uint32_t a, uint32_t b) { return row[a] > row[b]; });
        const uint32_t C = uint32_t(ec.expert_in.size());
        for (uint32_t s = 0; s < C; ++s) {
            const uint32_t e = rank[s];
            if (!row[e]) break;                    // never-used tail stays cold
            if (ec.slot_of[e] >= 0) continue;
            ec.expert_in[s] = int32_t(e);
            ec.slot_of[e] = int16_t(s);
            ec.last_use[s] = 0;
            FillReq r;
            r.dst = ec.base + uint64_t(s) * ec.slot_bytes + ec.gate_off;
            r.g = w.gate_src + uint64_t(e) * gsl; r.gsl = gsl;
            r.gfd = w.gate_fd; r.goff = w.gate_foff + uint64_t(e) * gsl;
            r.u = w.up_src + uint64_t(e) * usl;   r.usl = usl;
            r.ufd = w.up_fd;  r.uoff = w.up_foff + uint64_t(e) * usl;
            r.d2 = w.down_src + uint64_t(e) * dsl; r.dsl = dsl;
            r.dfd = w.down_fd; r.doff = w.down_foff + uint64_t(e) * dsl;
            pend.push_back(fill_.push(std::move(r)));
            bytes += gsl + usl + dsl;
        }
    }
    for (auto& p : pend) { sycl::event ev = p.get(); ev.wait(); }
    std::fprintf(stderr, "[glm5next] warm cache: %zu experts, %.2f GiB\n",
                 pend.size(), double(bytes) / 1073741824.0);
    return {};
}

std::string Glm5NextModel::ep_enable(Glm5NextModel& peer, uint32_t parity) {
    if (!copyq_) return "glm5next ep_enable: expert cache disabled";
    ep_peer_ = &peer;
    ep_parity_ = parity;
    const uint32_t H = cfg_.hidden, EF = cfg_.expert_ffn, E = cfg_.n_experts;
    const uint32_t TKMAX = 8 * cfg_.n_experts_used;
    auto dev = [&](uint64_t n) -> void* {
        void* p = alloc_->malloc(n);
        if (p) owned_.push_back(p);   // free_all() owns EP buffers too
        return p;
    };
    ep_x16_ = static_cast<sycl::half*>(dev(8ull * H * 2));
    ep_xg_ = static_cast<sycl::half*>(dev(8ull * H * 2));
    ep_scratch_ = static_cast<sycl::half*>(dev(8ull * (2ull * EF + H) * 2));
    ep_idx_ = static_cast<int32_t*>(dev(uint64_t(TKMAX) * 4));
    ep_w_ = static_cast<float*>(dev(uint64_t(TKMAX) * 4));
    ep_acc_ = static_cast<float*>(dev(8ull * H * 4));
    ep_ret_ = static_cast<float*>(dev(8ull * H * 4));
    if (!ep_x16_ || !ep_xg_ || !ep_scratch_ || !ep_idx_ || !ep_w_ || !ep_acc_ || !ep_ret_)
        return "glm5next ep_enable: buffer alloc failed";
    // Half-caches for the PEER's pinned MoE layers — the second half of this
    // card's IE_G5_ECACHE_MB (init_runtime kept only half for our own layers).
    uint64_t budget = 10240ull << 20;
    if (const char* v = std::getenv("IE_G5_ECACHE_MB")) budget = uint64_t(std::atoll(v)) << 20;
    budget /= 2;
    uint32_t n_pl = 0;
    uint64_t sbm = 0;
    const uint32_t n_tf = cfg_.n_transformer_layers();
    for (uint32_t L = 0; L < n_tf; ++L) {
        if (L >= peer.ecache_.size() || !peer.ecache_[L].base) continue;
        ++n_pl;
        sbm = std::max(sbm, peer.ecache_[L].slot_bytes);
    }
    if (!n_pl) return "glm5next ep_enable: peer has no MoE layers";
    uint32_t slots = uint32_t(budget / (uint64_t(n_pl) * sbm));
    slots = std::min(slots, E);
    // T<=4 spec windows can route up to 4*top_k picks; half-parity ~16. The
    // fill loop refuses if a call's expert set exceeds the slot count.
    if (slots < 4 * cfg_.n_experts_used)
        return "glm5next ep_enable: budget too small (" + std::to_string(slots) + " slots)";
    ep_cache_.resize(cfg_.n_layers);
    uint64_t total = 0;
    for (uint32_t L = 0; L < n_tf; ++L) {
        if (L >= peer.ecache_.size() || !peer.ecache_[L].base) continue;
        const ECache& pec = peer.ecache_[L];
        ECache& ec = ep_cache_[L];
        ec.gate_off = pec.gate_off; ec.up_off = pec.up_off; ec.down_off = pec.down_off;
        ec.slot_bytes = pec.slot_bytes;
        ec.base = static_cast<uint8_t*>(dev(uint64_t(slots) * ec.slot_bytes));
        if (!ec.base) return "glm5next ep_enable: ep cache alloc failed";
        ec.slot_of.assign(E, -1);
        ec.expert_in.assign(slots, -1);
        ec.last_use.assign(slots, 0);
        if (std::getenv("IE_G5_PREFETCH")) {
            ec.pf_ev.assign(slots, {});
            ec.pf_pending.assign(slots, 0);
        }
        total += uint64_t(slots) * ec.slot_bytes;
    }
    ep_tkmax_ = TKMAX;
    ep_hidx_.resize(uint64_t(cfg_.n_layers) * TKMAX);
    ep_hw_.resize(uint64_t(cfg_.n_layers) * TKMAX);
    for (int r = 0; r < 2; ++r) {
        ep_pin_[r] = static_cast<uint8_t*>(sycl::malloc_host(sbm, alloc_->queue()));
        if (!ep_pin_[r]) return "glm5next ep_enable: bounce ring alloc failed";
    }
    ep_xstage_ = static_cast<sycl::half*>(sycl::malloc_host(8ull * H * 2, alloc_->queue()));
    ep_rstage_ = static_cast<float*>(sycl::malloc_host(8ull * H * 4, alloc_->queue()));
    if (!ep_xstage_ || !ep_rstage_) return "glm5next ep_enable: staging alloc failed";
    std::fprintf(stderr,
                 "[glm5next] EP decode: parity %u, %u peer-layer half-caches, %u slots (%.2f GiB)\n",
                 parity, n_pl, slots, double(total) / 1073741824.0);
    return {};
}

// Peer-side half of one MoE layer at T<=4: fill misses from the owner's
// pinned host bank on OUR PCIe link, run the small-T gemv chain out of OUR
// half-cache, and land the fp32 partial in the owner's ep_ret_. Runs on this
// card's queues; `x_ev` = the owner's activation P2P copy, `ret_dep` = the
// owner's previous partial-add (ret_dst reuse fence).
std::string Glm5NextModel::ep_partial(const Glm5NextLayer& w, uint32_t L, uint32_t T,
                                      const uint32_t* exps, const uint32_t* ecnt, uint32_t n_exp,
                                      const int32_t* idx_flat, const float* w_flat, uint32_t npick,
                                      const sycl::half* x_host, const sycl::event& x_ev,
                                      const sycl::event& ret_dep,
                                      float* ret_host, bool x_pushed, float* ret_dev,
                                      sycl::event* done) {
    sycl::queue& pq = alloc_->queue();
    ECache& ec = ep_cache_[L];
    const uint32_t H = cfg_.hidden, EF = cfg_.expert_ffn, E = cfg_.n_experts;
    const float clamp = cfg_.swiglu_clamp_exp[L];
    const uint64_t gsl = w.gate_exps->nbytes / E;
    const uint64_t usl = w.up_exps->nbytes / E;
    const uint64_t dsl = w.down_exps->nbytes / E;

    int32_t* hidx = ep_hidx_.data() + uint64_t(L) * ep_tkmax_;
    float* hw = ep_hw_.data() + uint64_t(L) * ep_tkmax_;
    std::copy(idx_flat, idx_flat + npick, hidx);
    std::copy(w_flat, w_flat + npick, hw);
    pq.memcpy(ep_idx_, hidx, uint64_t(npick) * 4);
    pq.memcpy(ep_w_, hw, uint64_t(npick) * 4);
    pq.memset(ep_acc_, 0, uint64_t(T) * H * 4);
    // Cross-device event deps are unreliable on this stack (one direction
    // silently no-ops — the 06:16 oracle's stale-x signature). HOST-wait the
    // two incoming dependencies instead; the in-order queue orders the rest.
    const_cast<sycl::event&>(x_ev).wait();
    const_cast<sycl::event&>(ret_dep).wait();
    sycl::event fence = pq.ext_oneapi_submit_barrier();
    if (!x_pushed)   // push mode: owner already P2P-wrote x into our ep_x16_
        pq.memcpy(ep_x16_, x_host, uint64_t(T) * H * 2);   // H2D on OUR link (push pair)

    std::vector<sycl::event> fev(n_exp);
    std::vector<uint8_t> fillv(n_exp, 0);
    std::vector<uint8_t> cur(E, 0);
    for (uint32_t i = 0; i < n_exp; ++i) cur[exps[i]] = 1;
    for (uint32_t i = 0; i < n_exp; ++i) {
        const uint32_t e = exps[i];
        if (ec.slot_of[e] >= 0) {
            ++ecache_hits;
            const int16_t sl = ec.slot_of[e];
            ec.last_use[sl] = ++ep_clock_;
            // prefetched slot: first demand reader carries the fill event
            if (!ec.pf_pending.empty() && ec.pf_pending[sl]) {
                fev[i] = ec.pf_ev[sl].get();
                fillv[i] = 1;
                ec.pf_pending[sl] = 0;
                ++pf_used_;
            }
            continue;
        }
        uint32_t v = UINT32_MAX;
        uint64_t old = UINT64_MAX;
        for (uint32_t s = 0; s < ec.expert_in.size(); ++s) {
            if (!ec.pf_pending.empty() && ec.pf_pending[s]) continue;   // DMA maybe unsubmitted
            if (ec.expert_in[s] >= 0 && cur[ec.expert_in[s]]) continue;
            if (ec.last_use[s] < old) { old = ec.last_use[s]; v = s; }
        }
        if (v == UINT32_MAX) return "glm5next ep: half-cache smaller than one call";
        if (ec.expert_in[v] >= 0) ec.slot_of[ec.expert_in[v]] = -1;
        ec.expert_in[v] = int32_t(e);
        ec.slot_of[e] = int16_t(v);
        ec.last_use[v] = ++ep_clock_;
        ++ecache_misses;
        if (miss_counts.empty()) miss_counts.assign(uint64_t(cfg_.n_layers) * E, 0);
        ++miss_counts[uint64_t(L) * E + e];
        fillv[i] = 1;
        uint8_t* dst = ec.base + uint64_t(v) * ec.slot_bytes;
        if (w.bank_pinned) {
            fev[i] = copyq_->memcpy(dst,
                                    w.bank_pin + uint64_t(e) * (gsl + usl + dsl),
                                    gsl + usl + dsl, {fence});
        } else {
            // mmap mode: bounce through the EP ring (blocking pooled copy from
            // the owner's mmap source, then one DMA on our link). The EP ring
            // is sized for the PEER stage's largest slice — the shared
            // pin_ring_ is not (Q6_K down layers overran it: 05:07 SIGSEGV).
            uint8_t* pin = ep_pin_[ep_pin_idx_];
            ep_pin_ev_[ep_pin_idx_].wait();
            if (auto e2 = pread_fill_(w.gate_fd, w.gate_foff + uint64_t(e) * gsl,
                                      w.gate_src + uint64_t(e) * gsl, gsl,
                                      w.up_fd,   w.up_foff   + uint64_t(e) * usl,
                                      w.up_src   + uint64_t(e) * usl, usl,
                                      w.down_fd, w.down_foff + uint64_t(e) * dsl,
                                      w.down_src + uint64_t(e) * dsl, dsl,
                                      pin); !e2.empty()) return e2;
            ep_pin_ev_[ep_pin_idx_] = copyq_->memcpy(dst, pin, gsl + usl + dsl, {fence});
            fev[i] = ep_pin_ev_[ep_pin_idx_];
            ep_pin_idx_ = (ep_pin_idx_ + 1) % 2;
        }
    }

    // Grouped T==1 (mirrors run_moe's compute_grouped — same kernels, same
    // ordered reduce → bit-identical to the per-expert chain below).
    static const bool ep_grouped_on = [] {
        const char* v = std::getenv("IE_G5_GROUPED");
        return v && v[0] == '1';
    }();
    static const bool ep_no_q6_slm = std::getenv("IE_NO_Q6K_SLM") != nullptr;
    if (ep_grouped_on && T == 1 && n_exp <= 8 &&
        w.gate_exps->dtype == DType::kQ4_K && w.up_exps->dtype == DType::kQ4_K &&
        (w.down_exps->dtype == DType::kQ5_K ||
         (w.down_exps->dtype == DType::kQ6_K && !ep_no_q6_slm))) {
        static bool once = [] {
            std::fprintf(stderr, "[glm5next] grouped MoE decode active (EP peer)\n");
            return true;
        }();
        (void)once;
        int32_t slots[8]; float wts[8];
        // One event per filled expert (see compute_grouped: pf fills break
        // the last-covers-all assumption).
        std::vector<sycl::event> dep;
        for (uint32_t i = 0; i < n_exp; ++i) {
            slots[i] = ec.slot_of[exps[i]];
            wts[i]   = hw[i];                 // T==1: flat order == exps order
            if (fillv[i]) dep.push_back(fev[i]);
        }
        sycl::half* gg = ep_scratch_;
        sycl::half* gu = ep_scratch_ + 8ull * EF;
        sycl::half* gd = ep_scratch_ + 16ull * EF;
        moe_gemv_q4_K_gu_grouped(pq, ep_x16_, ec.base, ec.slot_bytes,
                                 ec.gate_off, ec.up_off, slots, n_exp,
                                 gg, gu, H, EF, dep);
        ds4_swiglu_clamped_h(pq, gg, gu, gg, uint64_t(n_exp) * EF, clamp, {});
        if (w.down_exps->dtype == DType::kQ5_K)
            moe_gemv_q5_K_down_grouped(pq, gg, ec.base, ec.slot_bytes,
                                       ec.down_off, slots, n_exp, gd, EF, H, {});
        else
            moe_gemv_q6_K_down_grouped(pq, gg, ec.base, ec.slot_bytes,
                                       ec.down_off, slots, n_exp, gd, EF, H, {});
        moe_reduce_waccum(pq, gd, wts, n_exp, ep_acc_, H, {});
        *done = pq.memcpy(ret_dev ? ret_dev : (float*)ret_host, ep_acc_,
                          uint64_t(T) * H * 4);   // ret_dev = P2P push to owner
        return {};
    }

    sycl::half* yg = ep_scratch_;
    sycl::half* yu = ep_scratch_ + EF;
    sycl::half* yd = ep_scratch_ + 2 * EF;
    auto gemv_raw = [&](const sycl::half* A, const uint8_t* raw, DType dt,
                        sycl::half* yv, uint32_t K, uint32_t N,
                        const std::vector<sycl::event>& deps) -> std::string {
        switch (dt) {
            case DType::kQ4_K: gemv_q4_K(pq, A, raw, yv, K, N, deps); break;
            case DType::kQ5_K: gemv_q5_K(pq, A, raw, yv, K, N, deps); break;
            case DType::kQ6_K: gemv_q6_K(pq, A, raw, yv, K, N, deps); break;
            case DType::kQ3_K:    gemv_q3_K(pq, A, raw, yv, K, N, deps); break;
            case DType::kIQ4_XS:  gemv_iq4_xs(pq, A, raw, yv, K, N, deps); break;
            case DType::kIQ3_XXS: gemv_iq3_xxs_raw(pq, A, raw, yv, K, N, deps); break;
            default:
                return std::string("glm5next ep: unsupported bank dtype ") +
                       std::string(type_name(dt));
        }
        return {};
    };
    uint32_t off = 0;
    for (uint32_t i = 0; i < n_exp; ++i) {
        const uint32_t e = exps[i];
        const uint32_t nt = ecnt[i];
        k_gather_rows(pq, ep_x16_, ep_idx_ + off, ep_xg_, nt, H, {});
        const uint8_t* base = ec.base + uint64_t(ec.slot_of[e]) * ec.slot_bytes;
        const uint8_t* g_raw = base + ec.gate_off;
        const uint8_t* u_raw = base + ec.up_off;
        const uint8_t* d_raw = base + ec.down_off;
        for (uint32_t r = 0; r < nt; ++r) {
            const sycl::half* xr = ep_xg_ + uint64_t(r) * H;
            const std::vector<sycl::event> dep = (r == 0 && fillv[i])
                ? std::vector<sycl::event>{fev[i]} : std::vector<sycl::event>{};
            if (w.gate_exps->dtype == DType::kQ4_K && w.up_exps->dtype == DType::kQ4_K) {
                gemv_q4_K_dual(pq, xr, g_raw, u_raw, yg, yu, H, EF, "g5_moe_gu", dep);
            } else {
                if (auto err = gemv_raw(xr, g_raw, w.gate_exps->dtype, yg, H, EF, dep); !err.empty()) return err;
                if (auto err = gemv_raw(xr, u_raw, w.up_exps->dtype,   yu, H, EF, {}); !err.empty()) return err;
            }
            ds4_swiglu_clamped_h(pq, yg, yu, yg, EF, clamp, {});
            if (auto err = gemv_raw(yg, d_raw, w.down_exps->dtype, yd, EF, H, {}); !err.empty()) return err;
            k_scatter_add_h(pq, yd, ep_idx_ + off + r, ep_w_ + off + r, ep_acc_, H, {});
        }
        off += nt;
    }
    *done = pq.memcpy(ret_dev ? ret_dev : (float*)ret_host, ep_acc_,
                      uint64_t(T) * H * 4);   // ret_dev = P2P push to owner
    return {};
}

std::string Glm5NextModel::forward(const int32_t* tokens_host, uint32_t T,
                                   uint32_t start_pos, sycl::half* logits_out,
                                   float* all_logits) {
    return forward_range(tokens_host, T, start_pos, nullptr, nullptr,
                         logits_out, all_logits);
}


void Glm5NextModel::CpuWorker::start() {
    if (th.joinable()) return;
    th = std::thread([this] {
        const bool pinned = core_lo >= 0 && core_hi >= core_lo;
        if (pinned) {
            // IE_G5_CPU_CORES=lo-hi per stage (or the auto split): pin the
            // worker; its OpenMP team inherits this mask ONLY with
            // KMP_AFFINITY=disabled (set in init_runtime) — otherwise libiomp5
            // re-pins the team to the first master's mask (G1 gate, 2026-09-02).
            // Probe 2026-08-31: both teams "close"-bound on the same 8 P-cores
            // -> 1.44 ms/expert vs 0.25 in isolation.
            cpu_set_t set; CPU_ZERO(&set);
            for (int c = core_lo; c <= core_hi; ++c) CPU_SET(c, &set);
            if (sched_setaffinity(0, sizeof(set), &set) != 0)
                std::fprintf(stderr, "[glm5next] q* CPU worker: sched_setaffinity(%d-%d) failed: %s — running unpinned\n",
                             core_lo, core_hi, std::strerror(errno));
        }
        bool first = true;
        for (;;) {
            std::function<void()> j;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [this] { return has_job || quit; });
                if (quit && !has_job) return;
                j = std::move(job);
                has_job = false;
            }
            j();
            if (first) {
                // Report the mask the kernel actually holds AFTER the first
                // OpenMP region (the runtime may have changed it) — the log
                // states the real placement, not the requested one.
                first = false;
                cpu_set_t now; CPU_ZERO(&now);
                if (sched_getaffinity(0, sizeof(now), &now) == 0) {
                    std::string s; int run_lo = -1;
                    for (int c = 0; c <= CPU_SETSIZE; ++c) {
                        const bool in = c < CPU_SETSIZE && CPU_ISSET(c, &now);
                        if (in && run_lo < 0) run_lo = c;
                        if (!in && run_lo >= 0) {
                            if (!s.empty()) s += ",";
                            s += std::to_string(run_lo);
                            if (c - 1 != run_lo) s += "-" + std::to_string(c - 1);
                            run_lo = -1;
                        }
                    }
                    std::fprintf(stderr, "[glm5next] q* CPU worker (stage %s): mask after the first region = %s (%s %d-%d)\n",
                                 stage, s.c_str(), pinned ? "requested" : "unpinned; requested", core_lo, core_hi);
                }
            }
            {
                std::lock_guard<std::mutex> lk(mu);
                done = true;
            }
            cv.notify_all();
        }
    });
}
void Glm5NextModel::CpuWorker::submit(std::function<void()> j) {
    start();
    {
        std::lock_guard<std::mutex> lk(mu);
        job = std::move(j);
        has_job = true;
        done = false;
    }
    cv.notify_all();
}
void Glm5NextModel::CpuWorker::wait() {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [this] { return done; });
}
Glm5NextModel::CpuWorker::~CpuWorker() {
    {
        std::lock_guard<std::mutex> lk(mu);
        quit = true;
    }
    cv.notify_all();
    if (th.joinable()) th.join();
}

std::string Glm5NextModel::forward_range(const int32_t* tokens_host, uint32_t T,
                                         uint32_t start_pos, const float* wide_in_host,
                                         float* wide_out_host, sycl::half* logits_out,
                                         float* all_logits) {
    if (!lat_cache_) return "glm5next forward: init_runtime not called";
    if (T == 0 || T > max_chunk_) return "glm5next forward: bad T";
    if (start_pos != kv_len_) return "glm5next forward: non-contiguous position";
    // v0 is EXACT dense MLA, which equals the sparse path only while every
    // position is selected (n_select = top_k + kpool - 1 = 2051 on the real
    // file). Refuse honestly beyond it — the indexer is a later phase.
    const uint32_t n_sel = cfg_.indexer_top_k + cfg_.indexer_kpool - 1;
    // v0 refused past n_sel because dense MLA only equals the sparse answer
    // while every position is selected. With the DSA indexer live that bound
    // is gone; without it the refusal still stands.
    if (!sparse_ && start_pos + T > n_sel)
        return "glm5next forward: dense MLA == sparse only to " + std::to_string(n_sel) +
               " positions (raise --ctx to engage the DSA indexer)";
    if (start_pos + T > max_ctx_) return "glm5next forward: exceeds max_ctx";

    sycl::queue& q = alloc_->queue();
    const uint32_t H = cfg_.hidden, hc = cfg_.hc_count;
    const uint32_t n_tf = cfg_.n_transformer_layers();
    // IE_G5_TRACE_FIRST=1: time every layer of THIS stage's first forward
    // (the ~50s stage-A first-forward anomaly hunt). The per-layer q.wait()
    // distorts overlap — diagnosis only.
    const bool trace = !traced_first_ && std::getenv("IE_G5_TRACE_FIRST") != nullptr;
    const auto tf0 = std::chrono::steady_clock::now();
    auto since = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - tf0).count();
    };
    if (wide_in_host == nullptr) {
        if (layer_lo_ != 0) return "glm5next forward_range: no wide input for a tail stage";
        q.memcpy(d_tokens_, tokens_host, uint64_t(T) * 4).wait();
        k_embed_streams(q, d_tokens_, token_embd, wide_a_, T, H, hc, {});
    } else {
        q.memcpy(wide_a_, wide_in_host, uint64_t(T) * hc * H * 4);
    }
    if (trace) { q.wait(); std::fprintf(stderr, "[trace s%u] embed/wide-in %.2fs\n", layer_lo_, since()); }

    if (spec_verify_) { spec_T_ = T; spec_pos_ = start_pos; }
    // Diagnostic (IE_G5_DUMP_WIDE=<path>): cross-process determinism bisect —
    // stage this forward's residual after every block's attn mix and ffn mix
    // (run_block pushes async D2H copies on the in-order queue), then append
    // the records to <path>.<layer_lo_>: header {L, stage, T, start_pos, n}
    // + n floats each. The host write lands after the forward's final wait.
    static const char* dump_path = std::getenv("IE_G5_DUMP_WIDE");
    if (dump_path) {
        dump_h_.assign(uint64_t(2) * (layer_hi_ - layer_lo_) * T * hc * H, 0.f);
        dump_cur_ = 0;
        dump_hdr_.clear();
    }
    for (uint32_t L = layer_lo_; L < layer_hi_; ++L) {
        if (auto e = run_block(L, T, start_pos); !e.empty()) return e;
        {   // Diagnostic: IE_G5_NAN_PROBE=1 reports the first layer whose
            // residual stream goes non-finite (decode T=1 NaN hunt, 2026-09-01).
            static const bool nan_probe = std::getenv("IE_G5_NAN_PROBE") != nullptr;
            static int nan_left = 2;
            if (nan_probe && nan_left > 0) {
                q.wait();
                std::vector<float> hw(uint64_t(T) * cfg_.hc_count * cfg_.hidden);
                q.memcpy(hw.data(), wide_a_, hw.size() * 4).wait();
                uint32_t nf = 0;
                for (float v : hw) if (!std::isfinite(v)) ++nf;
                if (nf) {
                    std::fprintf(stderr, "[nan-probe] pos %u T %u: layer %u output has %u non-finite of %zu\n",
                                 start_pos, T, L, nf, hw.size());
                    --nan_left;
                }
            }
        }
        if (trace) { q.wait(); std::fprintf(stderr, "[trace s%u] L%u %.2fs\n", layer_lo_, L, since()); }
    }
    if (trace) traced_first_ = true;
    if (dump_path) {
        q.wait();
        const std::string p = std::string(dump_path) + "." + std::to_string(layer_lo_);
        if (FILE* f = std::fopen(p.c_str(), "ab")) {
            const uint64_t n = uint64_t(T) * hc * H;
            for (size_t r = 0; r < dump_hdr_.size() / 5; ++r) {
                std::fwrite(dump_hdr_.data() + r * 5, 4, 5, f);
                std::fwrite(dump_h_.data() + r * n, 4, n, f);
            }
            std::fclose(f);
        }
        if (!dmoe_hdr_.empty()) {   // IE_G5_DUMP_MOE records (see run_moe)
            const std::string pm = std::string(dump_path) + ".moe." + std::to_string(layer_lo_);
            if (FILE* f = std::fopen(pm.c_str(), "ab")) {
                size_t hi = 0; uint64_t off = 0;
                while (hi < dmoe_hdr_.size()) {
                    const uint32_t nrows = dmoe_hdr_[hi + 5], len = dmoe_hdr_[hi + 7];
                    std::fwrite(dmoe_hdr_.data() + hi, 4, 8 + nrows, f);
                    std::fwrite(dmoe_h_.data() + off, 2, len, f);
                    hi += 8 + nrows; off += len;
                }
                std::fclose(f);
            }
            dmoe_hdr_.clear(); dmoe_cur_ = 0;
        }
        dump_h_.clear();
    }
    kv_len_ = start_pos + T;

    if (wide_out_host) {
        q.memcpy(wide_out_host, wide_a_, uint64_t(T) * hc * H * 4);
        const auto ww0 = std::chrono::steady_clock::now();
        q.wait();
        t_wide_wait += std::chrono::duration<double>(std::chrono::steady_clock::now() - ww0).count();
        return {};
    }
    if (layer_hi_ != n_tf) return "glm5next forward_range: mid-pipe stage needs wide_out";

    // final merge: unweighted stream mean -> output_norm -> lm_head
    k_stream_mean(q, wide_a_, mean_, T, H, hc, {});
    cast_fp32_to_fp16(q, mean_, x16_, uint64_t(T) * H, {});
    rms_norm_f32w(q, x16_, output_norm, x16_, T, H, cfg_.rms_eps, {});
    const uint32_t V = cfg_.vocab;
    if (all_logits) {
        if (lm_head.qs)
            for (uint32_t t0 = 0; t0 < T; t0 += 16) {
                const uint32_t bs = std::min(16u, T - t0);
                gemv_q8_0_soa_f16_rows(q, x16_ + uint64_t(t0) * H, lm_head.qs,
                                       lm_head.d, lgs16_, H, V, bs, {});
                cast_fp16_to_fp32(q, lgs16_, all_logits + uint64_t(t0) * V,
                                  uint64_t(bs) * V, {});
            }
        else
            gemm_fp16(q, x16_, lm_head.w, all_logits, T, V, H, {});
    }
    if (lm_head.qs)
        gemv_q8_0_soa_f16_g(q, x16_ + uint64_t(T - 1) * H, lm_head.qs, lm_head.d,
                            logits_, H, V, {});
    else
        gemv_fp16(q, x16_ + uint64_t(T - 1) * H, lm_head.w, logits_, H, V, {});
    if (logits_out && logits_out != logits_)
        q.memcpy(logits_out, logits_, uint64_t(V) * 2);
    {
        const auto fw0 = std::chrono::steady_clock::now();
        q.wait();
        t_final_wait += std::chrono::duration<double>(std::chrono::steady_clock::now() - fw0).count();
    }
    return {};
}

}  // namespace ie
