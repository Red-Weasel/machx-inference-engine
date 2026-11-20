// src/model/hy4flash_q8.cpp — the 8-bit integer expert store of Hy4-flash. See include/ie/hy4flash_q8.hpp.
// Compiled with -fopenmp (src/CMakeLists.txt): the load-time conversion of 148.6 B weights runs on every core.
#include "ie/hy4flash_q8.hpp"
#include "ie/hy4flash_forward.hpp"

#include <atomic>
#include <thread>
#include <fcntl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace ie {

namespace {
inline float bf16_at(const uint8_t* p) { uint16_t b; std::memcpy(&b, p, 2); const uint32_t u = uint32_t(b) << 16; float f; std::memcpy(&f, &u, 4); return f; }
// fp32 -> fp16 bits, round to nearest even (F16C: this file is built with -mf16c)
inline uint16_t f16_bits(float f) { return uint16_t(_cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)); }
inline float f16_value(uint16_t h) { return _cvtsh_ss(h); }
}  // namespace

void hy4f_q8_from_bf16(const uint8_t* bf16, uint32_t N, uint32_t K, int8_t* qs, uint16_t* d) {
    const uint32_t nb = K / 32;
    for (uint32_t n = 0; n < N; ++n) {
        const uint8_t* row = bf16 + size_t(n) * K * 2;
        for (uint32_t b = 0; b < nb; ++b) {
            float w[32]; float amax = 0.f;
            for (uint32_t i = 0; i < 32; ++i) { w[i] = bf16_at(row + (size_t(b) * 32 + i) * 2); amax = std::max(amax, std::fabs(w[i])); }
            uint16_t db = f16_bits(amax / 127.0f);
            float dv = f16_value(db);
            if (dv * 127.0f < amax) { db = uint16_t(db + 1); dv = f16_value(db); }      // the rounded scale must still reach the largest weight
            d[size_t(n) * nb + b] = db;
            const float inv = dv > 0.f ? 1.0f / dv : 0.f;
            int8_t* o = qs + size_t(n) * K + size_t(b) * 32;
            for (uint32_t i = 0; i < 32; ++i) { const float r = std::nearbyint(w[i] * inv); o[i] = int8_t(std::max(-127.f, std::min(127.f, r))); }
        }
    }
}

void hy4f_q8_from_bf16_mt(const uint8_t* bf16, uint32_t N, uint32_t K, int8_t* qs, uint16_t* d) {
    const uint32_t nb = K / 32, step = 64;
    #pragma omp parallel for schedule(dynamic, 1)
    for (uint32_t n0 = 0; n0 < N; n0 += step) {
        const uint32_t n = std::min(step, N - n0);
        hy4f_q8_from_bf16(bf16 + size_t(n0) * K * 2, n, K, qs + size_t(n0) * K, d + size_t(n0) * nb);
    }
}

void hy4f_qn_from_bf16(const uint8_t* bf16, uint32_t N, uint32_t K, uint32_t bits, bool container8, uint8_t* rows, uint16_t* d) {
    const uint32_t nb = K / 32; const float qmax = float((1u << (bits - 1)) - 1u);
    const uint32_t rb = Hy4fQFmt::row_bytes(container8 ? 8 : bits, K); const int off = bits == 5 ? 16 : 8;
    for (uint32_t n = 0; n < N; ++n) {
        const uint8_t* row = bf16 + size_t(n) * K * 2; uint8_t* out = rows + size_t(n) * rb;
        for (uint32_t b = 0; b < nb; ++b) {
            float w[32]; float amax = 0.f;
            for (uint32_t i = 0; i < 32; ++i) { w[i] = bf16_at(row + (size_t(b) * 32 + i) * 2); amax = std::max(amax, std::fabs(w[i])); }
            uint16_t db = f16_bits(amax / qmax);
            float dv = f16_value(db);
            if (dv * qmax < amax) { db = uint16_t(db + 1); dv = f16_value(db); }
            d[size_t(n) * nb + b] = db;
            const float inv = dv > 0.f ? 1.0f / dv : 0.f;
            int qv[32];
            for (uint32_t i = 0; i < 32; ++i) qv[i] = int(std::max(-qmax, std::min(qmax, std::nearbyint(w[i] * inv))));
            if (container8 || bits == 8) { for (uint32_t i = 0; i < 32; ++i) out[size_t(b) * 32 + i] = uint8_t(int8_t(qv[i])); continue; }
            uint32_t hb = 0;
            for (uint32_t j = 0; j < 16; ++j) {
                const uint32_t a = uint32_t(qv[2 * j] + off), c = uint32_t(qv[2 * j + 1] + off);
                out[size_t(b) * 16 + j] = uint8_t((a & 15u) | ((c & 15u) << 4));
                hb |= ((a >> 4) & 1u) << (2 * j); hb |= ((c >> 4) & 1u) << (2 * j + 1);
            }
            if (bits == 5) std::memcpy(out + (K / 2) + size_t(b) * 4, &hb, 4);
        }
    }
}

void Hy4fQ8Store::stop_warm() { warm_stop_.store(true); if (warm_.joinable()) warm_.join(); }

void Hy4fQ8Store::stop_uploader() {
    { std::lock_guard<std::mutex> lk(up_mu_); up_stop_ = true; }
    up_cv_.notify_all();
    if (up_thr_.joinable()) up_thr_.join();                 // (the worker drains its queue first)
    up_q_.reset();
    pend_.clear(); up_jobs_.clear(); up_fail_.clear(); up_submitted_ = up_finished_ = up_base_ = 0; up_stop_ = false;
}

bool Hy4fQ8Store::promote_async(sycl::queue& q, uint32_t L, uint32_t e, uint64_t token, uint64_t mark) {
    const uint32_t l = L - L0_;
    if (score_.size() != nL_ || pending(l, e)) return false;
    int32_t victim = -1; float vmin = 0.f;
    for (uint32_t i = 0; i < n_live_; ++i) {
        if (stamp_[l][i] > mark) continue;                       // in this call's kernels, or an upload on its way
        const int32_t ve = in_slot_[l][i];
        const float sv = ve < 0 ? -1.f : eff(l, uint32_t(ve), token);
        if (victim < 0 || sv < vmin) { vmin = sv; victim = int32_t(i); }
    }
    if (victim < 0 || eff(l, e, token) < std::max(vmin, 0.f) + kPromoteGain) return false;
    if (!up_thr_.joinable()) {
        try { up_q_ = std::make_unique<sycl::queue>(q.get_context(), q.get_device(), sycl::property::queue::in_order{}); }
        catch (const sycl::exception&) { return false; }
        up_thr_ = std::thread([this] {
            for (;;) {
                Pending j;
                { std::unique_lock<std::mutex> lk(up_mu_); up_cv_.wait(lk, [&] { return up_stop_ || !up_jobs_.empty(); }); if (up_jobs_.empty()) return; j = up_jobs_.front(); up_jobs_.pop_front(); }
                bool ok = true;
                // (after the compute queue's barrier: no kernel enqueued before the promotion still reads the old expert)
                try { up_q_->memcpy(dev_[j.l] + size_t(j.slot) * slot_bytes_, ram_[j.l] + size_t(j.e) * slot_bytes_, slot_bytes_, j.after).wait(); }
                catch (const sycl::exception& x) { ok = false; std::fprintf(stderr, "[hy4-flash] an expert upload failed (%s): the expert stays on the CPU\n", x.what()); }
                { std::lock_guard<std::mutex> lk(up_mu_); if (!ok) up_fail_[size_t(j.seq - up_base_)] = 1; ++up_finished_; }
                up_done_cv_.notify_all();
            }
        });
    }
    if (in_slot_[l][size_t(victim)] >= 0) slot_of_[l][size_t(in_slot_[l][size_t(victim)])] = -1;   // the victim is a miss from now on
    in_slot_[l][size_t(victim)] = -1;
    stamp_[l][size_t(victim)] = UINT64_MAX;                     // nobody's victim and in nobody's kernel until it is live
    Pending p{l, e, victim, 0, q.ext_oneapi_submit_barrier()};
    { std::lock_guard<std::mutex> lk(up_mu_); p.seq = up_submitted_++; up_fail_.push_back(0); up_jobs_.push_back(p); }
    up_cv_.notify_one();
    pend_.push_back(p); ++st_.promotions;
    return true;
}

void Hy4fQ8Store::activate_pending() {
    if (pend_.empty()) return;
    std::unique_lock<std::mutex> lk(up_mu_);
    up_done_cv_.wait(lk, [&] { return up_finished_ == up_submitted_; });
    for (const auto& p : pend_) {
        if (up_fail_[size_t(p.seq - up_base_)]) { stamp_[p.l][size_t(p.slot)] = ++clock_; continue; }   // (the slot stays empty)
        slot_of_[p.l][p.e] = p.slot; in_slot_[p.l][size_t(p.slot)] = int32_t(p.e); stamp_[p.l][size_t(p.slot)] = ++clock_;
    }
    pend_.clear(); up_fail_.clear(); up_base_ = up_submitted_;
}

uint64_t hy4f_engram_pack_fp8(const uint8_t* bf16, uint64_t rows, uint32_t hd, uint8_t* out) {
    uint64_t bad = 0;
    #pragma omp parallel for schedule(static) reduction(+ : bad)
    for (uint64_t r = 0; r < rows; ++r) {
        float x[256], y[256];
        const uint8_t* src = bf16 + r * hd * 2; uint8_t* dst = out + r * (4 + size_t(hd));
        for (uint32_t k = 0; k < hd; ++k) x[k] = bf16_at(src + size_t(k) * 2);
        hy4f_fp8_encode_row(x, hd, dst);
        hy4f_fp8_decode_row(dst, hd, y);
        hy4f_fp8_round_trip(x, hd);
        bad += std::memcmp(x, y, size_t(hd) * 4) != 0;
    }
    return bad;
}

void Hy4fQ8Store::unmap(size_t l) {
    if (l < registered_.size() && registered_[l] && reg_ctx_) {
        try { sycl::ext::oneapi::experimental::release_from_device_copy(ram_[l], *reg_ctx_); } catch (...) {}
        registered_[l] = 0;
    }
    if (l < pinned_.size() && pinned_[l] && pin_ctx_) { sycl::free(ram_[l], *pin_ctx_); pinned_[l] = 0; ram_[l] = nullptr; return; }
    if (l < mapped_.size() && mapped_[l]) ::munmap(ram_[l], size_t(E_) * slot_bytes_); else std::free(ram_[l]);
}

Hy4fQ8Store::~Hy4fQ8Store() {
    stop_warm(); stop_uploader();
    for (size_t l = 0; l < ram_.size(); ++l) {
        if (!ram_[l]) continue;
        unmap(l);
    }
}

void Hy4fQ8Store::free_all(sycl::queue& q) {
    stop_warm(); stop_uploader();
    namespace ex = sycl::ext::oneapi::experimental;
    if (lend_on_ && vm_ctx_) {
        if (lent_) for (uint32_t l = 0; l < nL_; ++l) ex::unmap(reinterpret_cast<void*>(lend_va_ + l * tail_bytes_), tail_bytes_, *vm_ctx_);
        for (size_t l = 0; l < dev_.size(); ++l) if (dev_[l]) {
            if (tail_off_) ex::unmap(dev_[l], tail_off_, *vm_ctx_);
            if (!lent_) ex::unmap(dev_[l] + tail_off_, tail_bytes_, *vm_ctx_);
            for (uint32_t i = 0; i < ctx_bytes_.size(); ++i) if (!ctx_gone_[i] && l < ph_ctx_[i].size()) ex::unmap(dev_[l] + ctx_off(i), ctx_bytes_[i], *vm_ctx_);
            ex::free_virtual_mem(reinterpret_cast<uintptr_t>(dev_[l]), res_bytes_, *vm_ctx_);
        }
        if (lend_va_) ex::free_virtual_mem(lend_va_, uint64_t(nL_) * tail_bytes_, *vm_ctx_);
        ph_head_.clear(); ph_tail_.clear(); ph_ctx_.clear(); ctx_bytes_.clear(); ctx_gone_.clear(); lend_va_ = 0; lent_ = false; lend_on_ = false;
    } else for (uint8_t* p : dev_) if (p) sycl::free(p, q);
    dev_.clear();
    for (size_t l = 0; l < ram_.size(); ++l) {
        if (!ram_[l]) continue;
        unmap(l);
    }
    ram_.clear(); mapped_.clear(); registered_.clear(); pinned_.clear(); vram_bytes_ = ram_bytes_ = 0;
}

std::string Hy4fQ8Store::init(sycl::queue& q, const Hy4FlashModel& m, uint32_t first_layer, uint32_t n_layers, uint32_t vram_slots,
                              const std::vector<std::vector<uint32_t>>& ranking, Hy4fQFmt fmt, uint64_t lend_bytes,
                              const std::vector<uint64_t>& ctx_pieces) {
    if (!Hy4fQFmt::known(fmt.gu) || !Hy4fQFmt::known(fmt.dn)) return "expert widths other than 8, 5 and 4 bits are not built";
    // IE_HY4F_QN_CONTAINER8=1 (a check of the low-bit kernels): round to the asked widths, store and compute as 8-bit
    static const bool container8 = [] { const char* v = std::getenv("IE_HY4F_QN_CONTAINER8"); return v && *v == '1'; }();
    const Hy4fQFmt asked = fmt;
    if (container8) fmt = {};
    fmt_ = fmt;
    const auto& cfg = m.config();
    L0_ = first_layer; nL_ = n_layers; E_ = cfg.n_experts; MI_ = cfg.moe_inter_dim; LAT_ = cfg.latent_dim;
    if (LAT_ % 32 || MI_ % 32) return "expert widths are not multiples of 32";
    n_slots_ = std::min(vram_slots, E_);
    if (n_slots_ < cfg.top_k) return "fewer VRAM slots a layer than experts a token picks";
    n_live_ = n_slots_; n_usable_ = n_slots_;
    const uint64_t gq = uint64_t(MI_) * Hy4fQFmt::row_bytes(fmt.gu, LAT_), dq = uint64_t(LAT_) * Hy4fQFmt::row_bytes(fmt.dn, MI_), gd = uint64_t(MI_) * (LAT_ / 32) * 2, dd = uint64_t(LAT_) * (MI_ / 32) * 2;
    const uint64_t src_g = uint64_t(MI_) * LAT_ * 2, src_d = uint64_t(LAT_) * MI_ * 2;        // a source matrix as BF16
    off_up_ = gq; off_down_ = 2 * gq; off_gate_d_ = 2 * gq + dq; off_up_d_ = off_gate_d_ + gd; off_down_d_ = off_up_d_ + gd;
    slot_bytes_ = (off_down_d_ + dd + 63) / 64 * 64;
    const auto t0 = std::chrono::steady_clock::now();
    ram_.assign(nL_, nullptr); dev_.assign(nL_, nullptr); mapped_.assign(nL_, 0); registered_.assign(nL_, 0); pinned_.assign(nL_, 0);
    slot_of_.assign(nL_, std::vector<int32_t>(E_, -1)); in_slot_.assign(nL_, std::vector<int32_t>(n_slots_, -1)); stamp_.assign(nL_, std::vector<uint64_t>(n_slots_, 0));
    for (uint32_t l = 0; l < nL_; ++l) {
        const Hy4fLayer& lw = m.layers()[L0_ + l];
        if (!lw.moe) return "layer " + std::to_string(L0_ + l) + " has no routed experts";
        // the pack file of this layer, when it is there and whole: mapped, not converted (a load in seconds)
        {
            char nm[64];
            if (fmt.gu == 8 && fmt.dn == 8) std::snprintf(nm, sizeof nm, "/experts-q8-L%02u.bin", L0_ + l);
            else std::snprintf(nm, sizeof nm, "/experts-q%u%u-L%02u.bin", fmt.gu, fmt.dn, L0_ + l);
            const std::string path = m.dir() + nm;
            struct stat sb{};
            if (!(container8 && (asked.gu != 8 || asked.dn != 8)) && ::stat(path.c_str(), &sb) == 0) {
                if (uint64_t(sb.st_size) != uint64_t(E_) * slot_bytes_) return "8-bit pack file " + path + " has " + std::to_string(sb.st_size) + " bytes, " + std::to_string(uint64_t(E_) * slot_bytes_) + " expected (re-run pack_q8.py)";
                const int fd = ::open(path.c_str(), O_RDONLY);
                if (fd < 0) return "cannot open " + path;
                // IE_HY4F_PIN_PACK (default on; =0 maps the pack): the layer in pinned host memory, read with parallel preads, the page cache dropped behind
                // it (one copy in RAM): the cards DMA a missed expert straight from it (~25 GB/s a card; the mapping 6.4) --
                // a prompt chunk streams its missed experts that way. DeepSeek-V4.1's expert tier is pinned the same way.
                static const bool pin_pack = [] { const char* v = std::getenv("IE_HY4F_PIN_PACK"); return !(v && *v == '0'); }();   // default on (2026-11-19: a longer load is the smaller cost)
                if (pin_pack) {
                    if (!pin_ctx_) pin_ctx_ = std::make_unique<sycl::context>(q.get_context());
                    const size_t bytes = size_t(E_) * slot_bytes_;
                    uint8_t* hp = sycl::malloc_host<uint8_t>(bytes, *pin_ctx_);
                    if (!hp) { ::close(fd); return "pinned host memory for the experts of layer " + std::to_string(L0_ + l) + " could not be allocated (IE_HY4F_PIN_PACK=0 maps the pack instead)"; }
                    std::atomic<bool> bad{false}; const size_t part = (bytes / 8 + 4095) / 4096 * 4096;
                    std::vector<std::thread> rd;
                    for (int t = 0; t < 8; ++t) rd.emplace_back([&, t] {
                        for (size_t o = size_t(t) * part; o < std::min(bytes, size_t(t + 1) * part);) {
                            const size_t want = std::min(std::min(bytes, size_t(t + 1) * part) - o, size_t(64) << 20);
                            const ssize_t got = ::pread(fd, hp + o, want, off_t(o));
                            if (got <= 0) { bad = true; return; }
                            o += size_t(got);
                        }
                    });
                    for (auto& th : rd) th.join();
                    ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
                    ::close(fd);
                    if (bad) { sycl::free(hp, *pin_ctx_); return "cannot read " + path; }
                    ram_[l] = hp; pinned_[l] = 1; mapped_[l] = 1; ++st_.mapped_layers;
                } else {
                void* mp = ::mmap(nullptr, size_t(E_) * slot_bytes_, PROT_READ, MAP_SHARED, fd, 0);
                ::close(fd);
                if (mp == MAP_FAILED) return "cannot map " + path;
                ram_[l] = static_cast<uint8_t*>(mp); mapped_[l] = 1; ++st_.mapped_layers;
                // registered with the driver for direct copies: an expert's upload from the mapping goes 6.4 -> 13.9 GB/s
                // (1.24 -> 0.57 ms) and no longer passes through the driver's CPU staging copy -- RAM the CPU leg reads at
                // the same time (h2d test, 2026-11-19). It pins the page-cache pages of the file (no second copy).
                // OPT-IN (IE_HY4F_REGISTER_PACK=1): measured at 400 decode tokens, twice interleaved, the same tokens -- 24.5 tok/s
                // off, 23.2 on: the faster uploads land while the CPU leg reads its experts and take RAM bandwidth from it.
                static const bool reg_pack = [] { const char* v = std::getenv("IE_HY4F_REGISTER_PACK"); return v && *v == '1'; }();
                if (reg_pack) {
                    if (!reg_ctx_) reg_ctx_ = std::make_unique<sycl::context>(q.get_context());
                    try { sycl::ext::oneapi::experimental::prepare_for_device_copy(mp, size_t(E_) * slot_bytes_, *reg_ctx_); registered_[l] = 1; }
                    catch (const sycl::exception& e) { if (l == 0) std::fprintf(stderr, "[hy4-flash] the expert pack could not be registered for direct copies (%s); copies stage through the CPU\n", e.what()); }
                }
                }   // (the mapped form)
            }
        }
        if (!mapped_[l]) {
        if (!lw.exp_gate_up.w || !lw.exp_down.w) return "layer " + std::to_string(L0_ + l) + " has no pack file and the full-precision experts are not in the source directory";
        if (lw.exp_gate_up.w->dtype_str != "BF16" || lw.exp_down.w->dtype_str != "BF16") return "the source's expert stacks are not BF16";
        ram_[l] = static_cast<uint8_t*>(std::aligned_alloc(64, size_t(E_) * slot_bytes_));
        if (!ram_[l]) return "out of memory for the 8-bit experts of layer " + std::to_string(L0_ + l);
        ram_bytes_ += uint64_t(E_) * slot_bytes_;
        }
        const uint8_t* gu = mapped_[l] ? nullptr : lw.exp_gate_up.w->data; const uint8_t* dn = mapped_[l] ? nullptr : lw.exp_down.w->data;
        uint8_t* base = ram_[l];
        #pragma omp parallel for schedule(dynamic, 4) if (!mapped_[l])
        for (uint32_t e = 0; e < (mapped_[l] ? 0u : E_); ++e) {
            uint8_t* s = base + size_t(e) * slot_bytes_;
            const uint8_t* g = gu + size_t(e) * 2 * src_g;                     // [2 * MI, LAT] BF16: gate rows, then up rows
            hy4f_qn_from_bf16(g, MI_, LAT_, asked.gu, container8, s, reinterpret_cast<uint16_t*>(s + off_gate_d_));
            hy4f_qn_from_bf16(g + src_g, MI_, LAT_, asked.gu, container8, s + off_up_, reinterpret_cast<uint16_t*>(s + off_up_d_));
            hy4f_qn_from_bf16(dn + size_t(e) * src_d, LAT_, MI_, asked.dn, container8, s + off_down_, reinterpret_cast<uint16_t*>(s + off_down_d_));
        }
        if (lend_bytes && l == 0) {                               // the lending layout (one granule rule for every layer)
            namespace ex = sycl::ext::oneapi::experimental;
            vm_ctx_ = std::make_unique<sycl::context>(q.get_context());
            // (the driver reports 64 KiB, but refused a 1.2 GB piece that was not a whole number of 2 MiB pages)
            const uint64_t g = std::max<uint64_t>(ex::get_mem_granularity(q.get_device(), q.get_context()), 2ull << 20);
            res_bytes_ = (uint64_t(n_slots_) * slot_bytes_ + g - 1) / g * g;
            uint64_t ctx_total = 0; std::vector<uint64_t> cb;
            for (uint64_t b : ctx_pieces) { cb.push_back((b + g - 1) / g * g); ctx_total += cb.back(); }
            if (ctx_total + lend_bytes + uint64_t(cfg.top_k) * slot_bytes_ >= res_bytes_) { ctx_total = 0; cb.clear(); }   // (no room: no pieces)
            const uint64_t ctx0 = res_bytes_ - ctx_total;
            tail_off_ = ctx0 > lend_bytes ? (ctx0 - lend_bytes) / g * g : 0;
            tail_bytes_ = ctx0 - tail_off_;
            lend_live_ = uint32_t(tail_off_ / slot_bytes_);
            if (lend_live_ >= cfg.top_k) { lend_on_ = true; ph_head_.reserve(nL_); ph_tail_.reserve(nL_); ctx_bytes_ = cb; ph_ctx_.resize(cb.size()); ctx_gone_.assign(cb.size(), 0); }
            else std::fprintf(stderr, "[hy4-flash] lending %.2f GiB a layer would leave %u expert slots: not lending\n", double(lend_bytes) / double(1ull << 30), lend_live_);
            set_slot_bounds();
        }
        if (lend_on_) {
            namespace ex = sycl::ext::oneapi::experimental;
            try {
                const uintptr_t va = ex::reserve_virtual_mem(res_bytes_, *vm_ctx_);
                if (tail_off_) { ph_head_.emplace_back(q.get_device(), *vm_ctx_, tail_off_); ph_head_.back().map(va, tail_off_, ex::address_access_mode::read_write); }
                else ph_head_.emplace_back(q.get_device(), *vm_ctx_, size_t(2ull << 20));   // (a placeholder, never mapped)
                ph_tail_.emplace_back(q.get_device(), *vm_ctx_, tail_bytes_); ph_tail_.back().map(va + tail_off_, tail_bytes_, ex::address_access_mode::read_write);
                for (uint32_t i = 0; i < ctx_bytes_.size(); ++i) { ph_ctx_[i].emplace_back(q.get_device(), *vm_ctx_, ctx_bytes_[i]); ph_ctx_[i].back().map(va + ctx_off(i), ctx_bytes_[i], ex::address_access_mode::read_write); }
                dev_[l] = reinterpret_cast<uint8_t*>(va);
            } catch (const sycl::exception& x) { return std::string("device allocation failed (8-bit expert slots of layer ") + std::to_string(L0_ + l) + ", lending layout: " + x.what() + ")"; }
        } else dev_[l] = sycl::malloc_device<uint8_t>(size_t(n_slots_) * slot_bytes_, q);
        if (!dev_[l]) return "device allocation failed (8-bit expert slots of layer " + std::to_string(L0_ + l) + ")";
        vram_bytes_ += uint64_t(n_slots_) * slot_bytes_;
        const std::vector<uint32_t>* rk = (L0_ + l) < ranking.size() && ranking[L0_ + l].size() == E_ ? &ranking[L0_ + l] : nullptr;
        for (uint32_t sidx = 0; sidx < n_slots_; ++sidx) {
            const uint32_t e = rk ? (*rk)[sidx] : sidx;
            q.memcpy(dev_[l] + size_t(sidx) * slot_bytes_, base + size_t(e) * slot_bytes_, slot_bytes_);
            slot_of_[l][e] = int32_t(sidx); in_slot_[l][sidx] = int32_t(e);
        }
        q.wait();
    }
    st_.build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    {   // page the mapped layers in behind the caller's back: one byte a page, in file order
        const char* v = std::getenv("IE_HY4F_PRELOAD");
        bool any = false; for (uint32_t l = 0; l < nL_; ++l) any = any || mapped_[l];
        warm_stop_.store(false); warm_done_.store(!any);
        if (any && !(v && *v == '0')) {
            const size_t layer_bytes = size_t(E_) * slot_bytes_;
            warm_ = std::thread([this, layer_bytes] {
                unsigned sink = 0;
                for (uint32_t l = 0; l < nL_ && !warm_stop_.load(std::memory_order_relaxed); ++l) {
                    if (!mapped_[l]) continue;
                    const uint8_t* p = ram_[l];
                    for (size_t off = 0; off < layer_bytes; off += 4096) {
                        sink += p[off];
                        if ((off & ((size_t(64) << 20) - 1)) == 0) { warm_bytes_.fetch_add(size_t(64) << 20, std::memory_order_relaxed); if (warm_stop_.load(std::memory_order_relaxed)) break; }
                    }
                }
                static volatile unsigned keep; keep = sink;
                if (!warm_stop_.load()) warm_done_.store(true);
            });
        }
    }
    return {};
}

Hy4fQ8Store::Planes Hy4fQ8Store::ram_planes(uint32_t L, uint32_t e) const {
    const uint8_t* b = ram_[L - L0_] + size_t(e) * slot_bytes_;
    Planes p;
    p.gate_qs = reinterpret_cast<const int8_t*>(b); p.up_qs = reinterpret_cast<const int8_t*>(b + off_up_); p.down_qs = reinterpret_cast<const int8_t*>(b + off_down_);
    p.gate_d = reinterpret_cast<const uint16_t*>(b + off_gate_d_); p.up_d = reinterpret_cast<const uint16_t*>(b + off_up_d_); p.down_d = reinterpret_cast<const uint16_t*>(b + off_down_d_);
    return p;
}

float Hy4fQ8Store::eff(uint32_t l, uint32_t e, uint64_t token) const {
    if (score_.size() != nL_) return 0.f;
    const uint64_t t0 = score_tok_[l][e];
    return token > t0 ? score_[l][e] * std::exp2(-float(token - t0) / kHalfLife) : score_[l][e];
}

void Hy4fQ8Store::note_pick(uint32_t L, uint32_t e, uint64_t token) {
    const uint32_t l = L - L0_;
    if (score_.size() != nL_) {             // first use: what the ranking put in VRAM starts with a small credit
        score_.assign(nL_, std::vector<float>(E_, 0.f)); score_tok_.assign(nL_, std::vector<uint64_t>(E_, token));
        for (uint32_t ll = 0; ll < nL_; ++ll) for (uint32_t x = 0; x < E_; ++x) if (slot_of_[ll][x] >= 0) score_[ll][x] = kSeedScore;
    }
    score_[l][e] = eff(l, e, token) + 1.f; score_tok_[l][e] = std::max(score_tok_[l][e], token);
}

bool Hy4fQ8Store::promote(sycl::queue& q, uint32_t L, uint32_t e, uint64_t token, uint64_t mark, Planes& out) {
    const uint32_t l = L - L0_;
    if (score_.size() != nL_) return false;
    int32_t victim = -1; float vmin = 0.f;
    for (uint32_t i = 0; i < n_live_; ++i) {
        if (stamp_[l][i] > mark) continue;                       // in this call's kernels
        const int32_t ve = in_slot_[l][i];
        const float sv = ve < 0 ? -1.f : eff(l, uint32_t(ve), token);
        if (victim < 0 || sv < vmin) { vmin = sv; victim = int32_t(i); }
    }
    if (victim < 0 || eff(l, e, token) < std::max(vmin, 0.f) + kPromoteGain) return false;
    if (in_slot_[l][size_t(victim)] >= 0) slot_of_[l][size_t(in_slot_[l][size_t(victim)])] = -1;
    q.memcpy(dev_[l] + size_t(victim) * slot_bytes_, ram_[l] + size_t(e) * slot_bytes_, slot_bytes_);
    slot_of_[l][e] = victim; in_slot_[l][size_t(victim)] = int32_t(e);
    stamp_[l][size_t(victim)] = ++clock_; ++st_.promotions;
    const uint8_t* b = dev_[l] + size_t(victim) * slot_bytes_;
    out.gate_qs = reinterpret_cast<const int8_t*>(b); out.up_qs = reinterpret_cast<const int8_t*>(b + off_up_); out.down_qs = reinterpret_cast<const int8_t*>(b + off_down_);
    out.gate_d = reinterpret_cast<const uint16_t*>(b + off_gate_d_); out.up_d = reinterpret_cast<const uint16_t*>(b + off_up_d_); out.down_d = reinterpret_cast<const uint16_t*>(b + off_down_d_);
    return true;
}

std::vector<std::pair<uint32_t, int32_t>> Hy4fQ8Store::plan_promotions(uint32_t L, uint64_t token, uint32_t budget) {
    std::vector<std::pair<uint32_t, int32_t>> out;
    const uint32_t l = L - L0_;
    if (score_.size() != nL_ || !budget) return out;
    std::vector<std::pair<float, uint32_t>> cand, vict;
    for (uint32_t e = 0; e < E_; ++e) {
        const float sc = eff(l, e, token);
        if (slot_of_[l][e] < 0) { if (sc >= kPromoteGain && !pending(l, e)) cand.emplace_back(sc, e); }
        else vict.emplace_back(sc, e);
    }
    std::sort(cand.begin(), cand.end(), [](const auto& x, const auto& y) { return x.first > y.first || (x.first == y.first && x.second < y.second); });
    std::sort(vict.begin(), vict.end(), [](const auto& x, const auto& y) { return x.first < y.first || (x.first == y.first && x.second < y.second); });
    for (size_t i = 0; i < cand.size() && i < vict.size() && out.size() < budget; ++i) {
        if (cand[i].first < vict[i].first + kPromoteGain) break;            // (sorted: no later pair can pass either)
        const uint32_t in = cand[i].second, out_e = vict[i].second;
        const int32_t slot = slot_of_[l][out_e];
        slot_of_[l][out_e] = -1;
        slot_of_[l][in] = slot; in_slot_[l][size_t(slot)] = int32_t(in); stamp_[l][size_t(slot)] = ++clock_;
        ++st_.promotions; out.emplace_back(in, slot);
    }
    return out;
}

uint32_t Hy4fQ8Store::rebalance(sycl::queue& q, uint32_t L, uint64_t token, uint32_t budget) {
    const uint32_t l = L - L0_;
    const auto plan = plan_promotions(L, token, budget);
    for (const auto& [in, slot] : plan) q.memcpy(dev_[l] + size_t(slot) * slot_bytes_, ram_[l] + size_t(in) * slot_bytes_, slot_bytes_);
    return uint32_t(plan.size());
}

std::string Hy4fQ8Store::acquire(sycl::queue& q, uint32_t L, uint32_t e, Planes& out) {
    if (L < L0_ || L >= L0_ + nL_ || e >= E_) return "8-bit expert store: layer or expert out of range";
    const uint32_t l = L - L0_;
    int32_t s = slot_of_[l][e];
    if (s < 0) {
        uint32_t victim = 0;
        for (uint32_t i = 1; i < n_live_; ++i) if (stamp_[l][i] < stamp_[l][victim]) victim = i;
        if (in_slot_[l][victim] >= 0) slot_of_[l][size_t(in_slot_[l][victim])] = -1;
        q.memcpy(dev_[l] + size_t(victim) * slot_bytes_, ram_[l] + size_t(e) * slot_bytes_, slot_bytes_);
        slot_of_[l][e] = int32_t(victim); in_slot_[l][victim] = int32_t(e); s = int32_t(victim);
        ++st_.uploads;
    } else ++st_.hits;
    stamp_[l][size_t(s)] = ++clock_;
    const uint8_t* b = dev_[l] + size_t(s) * slot_bytes_;
    out.gate_qs = reinterpret_cast<const int8_t*>(b); out.up_qs = reinterpret_cast<const int8_t*>(b + off_up_); out.down_qs = reinterpret_cast<const int8_t*>(b + off_down_);
    out.gate_d = reinterpret_cast<const uint16_t*>(b + off_gate_d_); out.up_d = reinterpret_cast<const uint16_t*>(b + off_up_d_); out.down_d = reinterpret_cast<const uint16_t*>(b + off_down_d_);
    return {};
}

uint8_t* Hy4fQ8Store::lend(sycl::queue& q) {
    namespace ex = sycl::ext::oneapi::experimental;
    if (!lend_on_) return nullptr;
    if (lent_) return reinterpret_cast<uint8_t*>(lend_va_);
    activate_pending();                                          // (no upload may still be writing a slot being lent)
    q.wait();
    const uint64_t tb = tail_bytes_;
    if (!lend_va_) lend_va_ = ex::reserve_virtual_mem(uint64_t(nL_) * tb, *vm_ctx_);
    for (uint32_t l = 0; l < nL_; ++l) {
        for (uint32_t s = lend_live_; s < n_slots_; ++s) {
            if (in_slot_[l][s] >= 0) slot_of_[l][size_t(in_slot_[l][s])] = -1;
            if (s < tail_slot_end_) in_slot_[l][s] = -1;          // (a context piece's slot keeps its expert, parked until take_back)
        }
        ex::unmap(dev_[l] + tail_off_, tb, *vm_ctx_);
        ph_tail_[l].map(lend_va_ + l * tb, tb, ex::address_access_mode::read_write);
    }
    n_live_ = lend_live_; lent_ = true;
    return reinterpret_cast<uint8_t*>(lend_va_);
}

void Hy4fQ8Store::take_back(sycl::queue& q, uint64_t token) {
    namespace ex = sycl::ext::oneapi::experimental;
    if (!lent_) return;
    q.wait();
    const uint64_t tb = tail_bytes_;
    for (uint32_t l = 0; l < nL_; ++l) {
        ex::unmap(reinterpret_cast<void*>(lend_va_ + l * tb), tb, *vm_ctx_);
        ph_tail_[l].map(reinterpret_cast<uintptr_t>(dev_[l]) + tail_off_, tb, ex::address_access_mode::read_write);
    }
    n_live_ = n_slots_; lent_ = false;
    for (uint32_t l = 0; l < nL_; ++l)                           // the parked experts of the context pieces serve again
        for (uint32_t sl = tail_slot_end_; sl < n_slots_; ++sl) {
            const int32_t e = in_slot_[l][sl];
            if (e < 0) continue;
            if (slot_of_[l][size_t(e)] < 0) slot_of_[l][size_t(e)] = int32_t(sl); else in_slot_[l][sl] = -1;   // (it took another slot meanwhile)
        }
    fill_empty(q, token, lend_live_);
}

// the slots at or above s0 that hold no expert: the best-scoring experts not resident (copies on q)
void Hy4fQ8Store::fill_empty(sycl::queue& q, uint64_t token, uint32_t s0) {
    for (uint32_t l = 0; l < nL_; ++l) {
        std::vector<std::pair<float, uint32_t>> cand;
        for (uint32_t e = 0; e < E_; ++e) if (slot_of_[l][e] < 0 && !pending(l, e)) cand.emplace_back(score_.size() == nL_ ? eff(l, e, token) : 0.f, e);
        std::sort(cand.begin(), cand.end(), [](const auto& x, const auto& y) { return x.first > y.first || (x.first == y.first && x.second < y.second); });
        size_t k = 0;
        for (uint32_t s = s0; s < n_slots_ && k < cand.size(); ++s) {
            if (in_slot_[l][s] >= 0 || stamp_[l][s] == kDead) continue;
            const uint32_t e = cand[k++].second;
            q.memcpy(dev_[l] + size_t(s) * slot_bytes_, ram_[l] + size_t(e) * slot_bytes_, slot_bytes_);
            slot_of_[l][e] = int32_t(s); in_slot_[l][s] = int32_t(e); stamp_[l][s] = ++clock_;
        }
    }
}

// a slot any part of which lies in a released piece is dead (stamp kDead); tail_slot_end_ = the first slot wholly inside the pieces
void Hy4fQ8Store::set_slot_bounds() {
    n_usable_ = 0;
    for (uint32_t sl = 0; sl < n_slots_; ++sl) {
        const uint64_t a = uint64_t(sl) * slot_bytes_, b = a + slot_bytes_;
        bool dead = b > res_bytes_;
        for (uint32_t i = 0; i < ctx_bytes_.size() && !dead; ++i) dead = ctx_gone_[i] && a < ctx_off(i) + ctx_bytes_[i] && b > ctx_off(i);
        for (uint32_t l = 0; l < nL_ && l < stamp_.size(); ++l) {
            if (dead) stamp_[l][sl] = kDead; else if (stamp_[l][sl] == kDead) stamp_[l][sl] = 0;
        }
        n_usable_ += !dead;
    }
    const uint64_t ctx0 = tail_off_ + tail_bytes_;
    tail_slot_end_ = std::min<uint32_t>(n_slots_, uint32_t((ctx0 + slot_bytes_ - 1) / slot_bytes_));
}

uint64_t Hy4fQ8Store::release_ctx(sycl::queue& q, uint32_t i) {
    namespace ex = sycl::ext::oneapi::experimental;
    if (i >= ctx_bytes_.size() || ctx_gone_[i]) return 0;
    activate_pending();                                          // (no upload may still be writing a slot being dropped)
    q.wait();
    ctx_gone_[i] = 1; set_slot_bounds();
    for (uint32_t l = 0; l < nL_; ++l) {
        for (uint32_t s = 0; s < n_slots_; ++s) {
            if (stamp_[l][s] != kDead) continue;
            const int32_t e = in_slot_[l][s];
            if (e >= 0 && slot_of_[l][size_t(e)] == int32_t(s)) slot_of_[l][size_t(e)] = -1;
            in_slot_[l][s] = -1;
        }
        ex::unmap(dev_[l] + ctx_off(i), ctx_bytes_[i], *vm_ctx_);
    }
    ph_ctx_[i].clear();                                          // the physical memory goes back to the driver
    vram_bytes_ -= uint64_t(nL_) * ctx_bytes_[i];
    return uint64_t(nL_) * ctx_bytes_[i];
}

std::string Hy4fQ8Store::restore_ctx(sycl::queue& q, uint64_t token, uint32_t i) {
    namespace ex = sycl::ext::oneapi::experimental;
    if (i >= ctx_bytes_.size() || !ctx_gone_[i]) return {};
    if (lent_) return "internal: a context piece cannot come back while the slots are lent";
    activate_pending();
    q.wait();
    try {
        ph_ctx_[i].reserve(nL_);
        for (uint32_t l = 0; l < nL_; ++l) {
            ph_ctx_[i].emplace_back(q.get_device(), *vm_ctx_, ctx_bytes_[i]);
            ph_ctx_[i].back().map(reinterpret_cast<uintptr_t>(dev_[l]) + ctx_off(i), ctx_bytes_[i], ex::address_access_mode::read_write);
        }
    } catch (const sycl::exception& x) {
        for (uint32_t l = 0; l < ph_ctx_[i].size(); ++l) { try { ex::unmap(dev_[l] + ctx_off(i), ctx_bytes_[i], *vm_ctx_); } catch (...) {} }
        ph_ctx_[i].clear();
        return std::string("a context piece of the expert slots could not be taken back: ") + x.what();
    }
    ctx_gone_[i] = 0; set_slot_bounds();
    vram_bytes_ += uint64_t(nL_) * ctx_bytes_[i];
    fill_empty(q, token, 0);
    return {};
}

}  // namespace ie
