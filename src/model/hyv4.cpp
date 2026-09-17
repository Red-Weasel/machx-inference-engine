// src/model/hyv4.cpp — GLM-5.3-Flash (`hyv4`) loader (v0).
//
// Placement contract in include/ie/hyv4.hpp; port plan in
// docs/glm53/PORT_PLAN.md. This translation unit is LOAD ONLY — the forward
// lands on top of it (port P2) so the placement can gate first, exactly the
// qwen4exp bring-up sequence.

#include "ie/hyv4.hpp"

#include "ie/cpu_moe_gemv.hpp"
#include "ie/deepseek4.hpp"       // ds4_hc_mix (the DecoderLayer residual mix)
#include "ie/deepseek4_ops.hpp"
#include "ie/dequant.hpp"
#include "ie/dequant_ref.hpp"
#include "ie/dtype.hpp"
#include "ie/kernel_profiler.hpp"
#include "ie/ops.hpp"
#include "ie/quant_blocks.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <pthread.h>
#include <sched.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <queue>
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
// plus the K-quants so a differently-quantized hyv4 GGUF still loads.
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
            return std::string("hyv4 load: unsupported device-tensor dtype ") +
                   std::string(type_name(t->dtype));
    }
}

}  // namespace

static void pin_copy_thread();

void Hyv4Model::CopyPool::start(uint32_t nw) {
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
void Hyv4Model::CopyPool::copy(uint8_t* d, const void* s, uint64_t bytes) {
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
Hyv4Model::CopyPool::~CopyPool() {
    { std::lock_guard<std::mutex> lk(mu); stop = true; }
    cv.notify_all();
    for (auto& t : th) t.join();
}

// IE_HY4_PIN_CPUS=1: pin copy-path threads (fill worker, pread pools) to the
// P-core set 0-7. The unpinned scheduler migrates them across P/E cores and
// the H2D submission path collapses (measured 2026-08-29: card0 10.6 GB/s
// unpinned vs 21 pinned, card1 20 vs 26, standalone). Compute threads stay
// free — a full-process 8-core pin cost decode 5.95 -> 4.24.
static void pin_copy_thread() {
    static const bool on = [] {
        const char* v = std::getenv("IE_HY4_PIN_CPUS");
        return v && v[0] == '1';
    }();
    if (!on) return;
    cpu_set_t s;
    CPU_ZERO(&s);
    for (int c = 0; c < 8; ++c) CPU_SET(c, &s);
    pthread_setaffinity_np(pthread_self(), sizeof(s), &s);
}

void Hyv4Model::FillThread::start(Hyv4Model* m) {
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
                    std::fprintf(stderr, "[hyv4] pf pread failed (%s) — memcpy fallback\n",
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
                // IE_HY4_PP_STREAM whole-layer load: three [E x slice] regions
                // chunked through the pin ring — pread into pinned, H2D on
                // the in-order copyq_. Striped mmap memcpy was slower
                // (pool_copy 14 s vs pread 10 s). First piece carries the
                // buffer-reuse fence; copyq_ in-order covers the rest.
                static const bool use_pread = [] {
                    const char* v = std::getenv("IE_HY4_PREAD");
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
                                std::fprintf(stderr, "[hyv4] pp-stream pread "
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
                    std::fprintf(stderr, "[hyv4] %s\n", e2.c_str());
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
std::shared_future<sycl::event> Hyv4Model::FillThread::push(FillReq&& r) {
    auto fut = r.done.get_future().share();
    {
        std::lock_guard<std::mutex> lk(mu);
        reqs.push_back(std::move(r));
    }
    cv.notify_one();
    return fut;
}
Hyv4Model::FillThread::~FillThread() {
    { std::lock_guard<std::mutex> lk(mu); stop = true; }
    cv.notify_all();
    if (th.joinable()) th.join();
}

Hyv4Model::~Hyv4Model() { free_all(); }

void Hyv4Model::shutdown() noexcept {
    if (shutting_down_) return;
    shutting_down_ = true;
    {
        std::lock_guard<std::mutex> lk(fill_.mu);
        fill_.stop = true;
    }
    fill_.cv.notify_all();
    if (fill_.th.joinable()) fill_.th.join();
    try { if (copyq_) copyq_->wait(); } catch (...) {}
    try { if (jobsq_) jobsq_->wait(); } catch (...) {}
    try { if (alloc_ && alloc_->ready()) alloc_->queue().wait(); } catch (...) {}
}

void Hyv4Model::free_all() {
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

std::string Hyv4Model::load(DeviceAllocator& alloc, const GgufReader& g,
                                const Hyv4Config& cfg, uint64_t vram_budget_bytes,
                                uint32_t layer_lo, uint32_t layer_hi) {
    alloc_ = &alloc;
    cfg_   = cfg;
    layers_.assign(cfg.n_layers, {});
    // The uploaded range never includes the MTP block: it only serves
    // speculative decode (P3) and its tensors are bound as host views below.
    const uint32_t n_tf = cfg.n_transformer_layers();
    layer_lo_ = std::min(layer_lo, n_tf);
    layer_hi_ = std::min(layer_hi, n_tf);
    if (layer_lo_ >= layer_hi_) return "hyv4 load: empty layer range";
    if (vram_budget_bytes == 0) vram_budget_bytes = 28ull << 30;

    char buf[96];
    auto Tl = [&](uint32_t L, const char* n) -> const GgufTensorInfo* {
        std::snprintf(buf, sizeof(buf), "blk.%u.%s", L, n);
        return g.find_tensor(buf);
    };
    std::string err;

    // ---- pass 1: bind every tensor + project device residency BEFORE any
    // upload, so the VRAM guard fires with zero bytes allocated. -------------
    // Layout classes (see the header comment on Hyv4Layer):
    //  kNative   — GGUF element order as-is (token_embd, k_b/v_b, 1-D, conv)
    //  kTransKN  — transposed to [K, N] row-major for gemv_fp16/gemm_fp16
    enum class Lay { kNative, kTransKN, kQ8Soa, kRawKq };
    struct Job {
        const GgufTensorInfo* t;
        sycl::half** dst_h;    // exactly one of dst_h / dst_f / dst_dw is set
        float**      dst_f;
        Hyv4DW*      dst_dw = nullptr;
        Lay          lay = Lay::kNative;
    };
    std::vector<Job> jobs;
    uint64_t projected = 0;

    auto want = [&](const GgufTensorInfo* t, const char* name,
                    sycl::half** h, float** f, Lay lay = Lay::kNative) -> bool {
        if (!t) { err = std::string("hyv4 load: missing tensor ") + name; return false; }
        uint64_t n = 1;
        for (uint32_t d = 0; d < t->n_dims; ++d) n *= t->shape[d];
        const uint64_t bytes = n * (h ? sizeof(sycl::half) : sizeof(float));
        projected += bytes;
        placement_.push_back({std::string(t->name), "device", h ? "F16" : "F32", bytes});
        jobs.push_back({t, h, f, nullptr, lay});
        return true;
    };
    // Dense 2-D projection: Q8_0-SoA when the file stores Q8_0 (halves the
    // resident reads — W8A16, quality-neutral; IE_HY4_DENSE_Q8=0 opts out),
    // F16 [K,N]-transposed otherwise.
    const bool q8_dense = [] {
        const char* v = std::getenv("IE_HY4_DENSE_Q8");
        return !(v && v[0] == '0');
    }();
    // P4: keep Q4_K/Q5_K/Q6_K dense mats PACKED on device (raw bytes, no
    // dequant at load). IE_HY4_DENSE_RAW=0 restores the f16 expansion.
    const bool raw_dense = [] {
        const char* v = std::getenv("IE_HY4_DENSE_RAW");
        return !(v && v[0] == '0');
    }();
    auto want_dw = [&](const GgufTensorInfo* t, const char* name,
                       Hyv4DW* dw) -> bool {
        if (!t) { err = std::string("hyv4 load: missing tensor ") + name; return false; }
        if (raw_dense && t->n_dims == 2 &&
            (t->dtype == DType::kQ4_K || t->dtype == DType::kQ5_K ||
             t->dtype == DType::kQ6_K)) {
            projected += t->nbytes;
            placement_.push_back({std::string(t->name), "device",
                                  type_name(t->dtype).data(), t->nbytes});
            jobs.push_back({t, nullptr, nullptr, dw, Lay::kRawKq});
            return true;
        }
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
        if (!t) { err = std::string("hyv4 load: missing expert bank blk.") +
                        std::to_string(L) + "." + n; return false; }
        *dst = t;
        host_bytes_ += t->nbytes;
        placement_.push_back({std::string(t->name), "host", type_name(t->dtype).data(), t->nbytes});
        return true;
    };
    // Globals by role. The tail stage also loads token_embd, but only when
    // MTP spec decode is requested (IE_HY4_MTP=1, set by the runner for
    // --spec): the fusion embeds the drafted token there. Default skips the
    // whole MTP kit — 1.18 GiB embd + blk.45 decoder + its ecache slot cut.
    const bool mtp_want = [] {
        const char* v = std::getenv("IE_HY4_MTP");
        return v && v[0] == '1';
    }();
    if (layer_lo_ == 0 || (mtp_want && layer_hi_ == n_tf)) {
        if (!want(g.find_tensor("token_embd.weight"), "token_embd.weight",
                  &token_embd, nullptr)) return err;
    }
    if (layer_hi_ == n_tf) {
        if (!want_dw(g.find_tensor("output.weight"), "output.weight", &lm_head)) return err;
        if (!want(g.find_tensor("output_norm.weight"), "output_norm.weight",
                  nullptr, &output_norm)) return err;
        if (!want(g.find_tensor("output_hc_fn.weight"), "output_hc_fn.weight",
                  nullptr, &hh_fn_)) return err;
        if (!want(g.find_tensor("output_hc_base.weight"), "output_hc_base.weight",
                  nullptr, &hh_base_)) return err;
        if (!want(g.find_tensor("output_hc_scale.weight"), "output_hc_scale.weight",
                  nullptr, &hh_scale_)) return err;
    }

    for (uint32_t L = layer_lo_; L < layer_hi_; ++L) {
        Hyv4Layer& w = layers_[L];
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
            if (!want_dw(Tl(L, "attn_gate.weight"), "attn_gate", &w.attn_gate)) return err;
            if (!want(Tl(L, "attn_sinks.weight"), "attn_sinks", nullptr, &w.attn_sinks)) return err;
            // Lightning indexer — only is_full layers ship weights; the
            // following layers reuse the previous full layer's top-k.
            if (L < cfg.indexer_is_full.size() && cfg.indexer_is_full[L]) {
            if (!want(Tl(L, "indexer.attn_k.weight"),   "indexer.attn_k",   &w.idx_k, nullptr, Lay::kTransKN)) return err;
            if (!want(Tl(L, "indexer.k_norm.weight"),   "indexer.k_norm",   nullptr, &w.idx_k_norm_w)) return err;
            if (!want(Tl(L, "indexer.k_norm.bias"),     "indexer.k_norm.bias", nullptr, &w.idx_k_norm_b)) return err;
            if (!want(Tl(L, "indexer.attn_q_b.weight"), "indexer.attn_q_b", &w.idx_q_b, nullptr, Lay::kTransKN)) return err;
            if (!want(Tl(L, "indexer.proj.weight"),     "indexer.proj",     nullptr, &w.idx_proj)) return err;
            }
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
    if (mtp_want && cfg.nextn_predict_layers == 1 && layer_hi_ == n_tf) {
        const uint32_t M = n_tf;   // blk.45 on the real file
        Hyv4Layer& w = layers_[M];
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

    // ---- VRAM guard: projected weights only (KV/workspaces come later and
    // small); refuse before the first byte moves. ----------------------------
    if (projected > vram_budget_bytes && std::getenv("IE_ALLOW_OOM") == nullptr) {
        std::snprintf(buf, sizeof(buf), "%.2f GiB > budget %.2f GiB",
                      projected / 1073741824.0, vram_budget_bytes / 1073741824.0);
        return std::string("hyv4 load: projected device weights ") + buf +
               " (IE_ALLOW_OOM=1 to attempt anyway)";
    }

    // ---- pass 2: dequant + upload. -----------------------------------------
    std::vector<float>      f32;
    std::vector<sycl::half> h16;
    sycl::queue& q = alloc.queue();
    for (const Job& j : jobs) {
        if (j.lay == Lay::kRawKq) {
            // packed bytes verbatim; the GEMV/Bt kernels read the native
            // [N][K/256] block layout straight off the device copy.
            void* d2 = alloc.malloc(j.t->nbytes);
            if (!d2) return "hyv4 load: raw dense alloc failed";
            q.memcpy(d2, j.t->data, j.t->nbytes).wait();
            owned_.push_back(d2);
            j.dst_dw->raw = static_cast<const uint8_t*>(d2);
            j.dst_dw->rdt = j.t->dtype;
            dev_bytes_ += j.t->nbytes;
            continue;
        }
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
            auto* dqs = static_cast<int8_t*>(alloc.malloc(qs.size()));
            auto* ddd = static_cast<uint16_t*>(alloc.malloc(dd.size() * 2));
            if (!dqs || !ddd) return "hyv4 load: q8 soa alloc failed";
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
            d = alloc.malloc(n * sizeof(float));
            if (!d) return "hyv4 load: device malloc failed (f32)";
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
            d = alloc.malloc(n * sizeof(sycl::half));
            if (!d) return "hyv4 load: device malloc failed (f16)";
            q.memcpy(d, h16.data(), n * sizeof(sycl::half)).wait();
            *j.dst_h = static_cast<sycl::half*>(d);
            dev_bytes_ += n * sizeof(sycl::half);
        } else {                             // kNative F16 (embd, k_b/v_b, conv, 1-D)
            h16.resize(n);
            for (uint64_t i = 0; i < n; ++i) h16[i] = sycl::half(f32[i]);
            d = alloc.malloc(n * sizeof(sycl::half));
            if (!d) return "hyv4 load: device malloc failed (f16)";
            q.memcpy(d, h16.data(), n * sizeof(sycl::half)).wait();
            *j.dst_h = static_cast<sycl::half*>(d);
            dev_bytes_ += n * sizeof(sycl::half);
        }
        owned_.push_back(d);
    }

    // ---- pass 3: bank fill sources — pin the owned banks into host USM ----
    // (IE_HY4_PIN_BANKS=0 opts out). Copies the mmap bytes into pinned
    // allocations layer by layer, releasing the page cache behind itself
    // (madvise DONTNEED) so RAM holds ONE copy, not two. A miss fill then
    // needs no CPU bounce at all. Stops pinning (gracefully, per layer) if
    // MemAvailable would drop under the 40 GiB floor — the 2026-08-27
    // pinned-livelock lesson.
    {
        for (uint32_t L = layer_lo_; L < cfg.n_layers; ++L) {
            Hyv4Layer& w2 = layers_[L];
            if (!w2.gate_exps) continue;
            w2.gate_src = static_cast<const uint8_t*>(w2.gate_exps->data);
            w2.up_src   = static_cast<const uint8_t*>(w2.up_exps->data);
            w2.down_src = static_cast<const uint8_t*>(w2.down_exps->data);
            g.locate(w2.gate_src, w2.gate_fd, w2.gate_foff);
            g.locate(w2.up_src,   w2.up_fd,   w2.up_foff);
            g.locate(w2.down_src, w2.down_fd, w2.down_foff);
        }
        // Founder 2026-08-29: banks stay on mmap (page cache — reclaimable,
        // "used" RAM stays ~20GB) by default. IE_HY4_PIN_BANKS=1 opts back in
        // to pinned host USM (26 vs ~8-12 GB/s fills, +160GB used RAM).
        const char* pv = std::getenv("IE_HY4_PIN_BANKS");
        if (pv && pv[0] == '1') {
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
            // floor tunable (IE_HY4_PIN_FLOOR_GIB, default 40): the last stage-B
            // layers land on mmap+bounce otherwise — ~1.5ms/miss slower at decode
            double floor_gib = 40.0;
            if (const char* fv = std::getenv("IE_HY4_PIN_FLOOR_GIB"))
                floor_gib = std::max(8.0, std::atof(fv));
            // Per-stage pinned-bytes cap (IE_HY4_PIN_MAX_GIB): bounds TOTAL used
            // RAM regardless of MemAvailable — the founder's 200+GB "big no"
            // (2026-08-29). Every stage gets its own cap, so EP always has
            // pinned layers on BOTH cards.
            double cap_gib = 1e9;
            if (const char* cv = std::getenv("IE_HY4_PIN_MAX_GIB"))
                cap_gib = std::max(4.0, std::atof(cv));
            uint32_t n_pinned = 0, n_skipped = 0;
            for (uint32_t L = layer_lo_; L < cfg.n_layers; ++L) {
                Hyv4Layer& w2 = layers_[L];
                if (!w2.gate_exps) continue;
                const GgufTensorInfo* ts[3] = {w2.gate_exps, w2.up_exps, w2.down_exps};
                const uint8_t** dsts[3] = {&w2.gate_src, &w2.up_src, &w2.down_src};
                const double need_gib =
                    (ts[0]->nbytes + ts[1]->nbytes + ts[2]->nbytes) / 1073741824.0;
                if (avail_gib() < need_gib + floor_gib) { ++n_skipped; continue; }
                // hard per-stage cap (Codex finding: parsed but unenforced)
                if (double(pinned_bytes_) / 1073741824.0 + need_gib > cap_gib) { ++n_skipped; continue; }
                (void)dsts;
                const uint64_t E2 = cfg.n_experts;
                const uint64_t gsl2 = ts[0]->nbytes / E2, usl2 = ts[1]->nbytes / E2,
                               dsl2 = ts[2]->nbytes / E2;
                const uint64_t stride = gsl2 + usl2 + dsl2;
                uint8_t* p = static_cast<uint8_t*>(
                    sycl::malloc_host(stride * E2, alloc.queue()));
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
            }
            std::fprintf(stderr,
                         "[hyv4] pinned banks: %u layers (%.1f GiB), %u left on mmap\n",
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
    return ie::ps(q, "hy4_embed", [&](sycl::handler& h) {
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
    return ie::ps(q, "hy4_sigmoid", [&](sycl::handler& h) {
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
    return ie::ps(q, "hy4_router", [&](sycl::handler& h) {
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
            for (uint32_t k = l; k < H; k += SG) acc = sycl::fma(xr[k], wr[k], acc);
            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (l == 0) rl[uint64_t(t) * E + e] = acc;
        });
    });
}

// Sigmoid + selection-bias top-k (glm52_run's k_router_topk, at 288 experts).
// Selection is by sigmoid(logit) + exp_probs_b; the returned weights are the
// UNBIASED sigmoids, normalised over the top-k (weights_norm) and scaled
// (weights_scale 2.5). Ties break to the lower expert id.
sycl::event k_router_topk(sycl::queue& q, const float* rl, const float* bias,
                          int32_t* top, float* topw, uint32_t NE, uint32_t TOPK,
                          float wscale, uint32_t T,
                          const std::vector<sycl::event>& deps) {
    constexpr int SG = 32;
    const uint32_t per = NE / SG;   // 9 at 288
    return ie::ps(q, "hy4_topk", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<1>(uint64_t(T) * SG, SG),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t t = uint32_t(it.get_group(0));
            auto sg = it.get_sub_group();
            const uint32_t lane = uint32_t(sg.get_local_id()[0]);
            const float* rlt = rl + uint64_t(t) * NE;
            float sc[16];
            for (uint32_t i = 0; i < per; ++i) {
                const float v = rlt[lane * per + i];
                sc[i] = 1.f / (1.f + sycl::exp(-v)) + bias[lane * per + i];
            }
            float wsum = 0.f;
            for (uint32_t k = 0; k < TOPK; ++k) {
                float bv = -INFINITY; uint32_t bi = 0;
                for (uint32_t i = 0; i < per; ++i) if (sc[i] > bv) { bv = sc[i]; bi = i; }
                const float m = sycl::reduce_over_group(sg, bv, sycl::maximum<float>());
                const uint32_t cand = (bv == m) ? (lane * per + bi) : NE;
                const uint32_t win  = sycl::reduce_over_group(sg, cand, sycl::minimum<uint32_t>());
                if (win / per == lane) sc[win - lane * per] = -INFINITY;
                const float wp = 1.f / (1.f + sycl::exp(-rlt[win]));
                wsum += wp;
                if (lane == 0) { top[uint64_t(t) * TOPK + k] = int32_t(win);
                                 topw[uint64_t(t) * TOPK + k] = wp; }
            }
            if (lane == 0)
                for (uint32_t k = 0; k < TOPK; ++k)
                    topw[uint64_t(t) * TOPK + k] = topw[uint64_t(t) * TOPK + k] / wsum * wscale;
        });
    });
}

sycl::event k_gather_rows(sycl::queue& q, const sycl::half* x, const int32_t* idx,
                          sycl::half* out, uint32_t n_rows, uint32_t H,
                          const std::vector<sycl::event>& deps) {
    return ie::ps(q, "hy4_gather", [&](sycl::handler& h) {
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
    return ie::ps(q, "hy4_scatter", [&](sycl::handler& h) {
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
    return ie::ps(q, "hy4_scatter_h", [&](sycl::handler& h) {
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
    return ie::ps(q, "hy4_scatter_hr", [&](sycl::handler& h) {
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
    constexpr int SG = 16;
    return ie::ps(q, "hy4_kabsorb", [&](sycl::handler& h) {
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
            for (uint32_t d = lane; d < HD; d += SG)
                acc = sycl::fma(float(kb[d]), qh[d], acc);
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
    constexpr int SG = 16;
    return ie::ps(q, "hy4_vabsorb", [&](sycl::handler& h) {
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
            for (uint32_t l = lane; l < LAT; l += SG)
                acc = sycl::fma(float(vb[l]), ar[l], acc);
            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (lane == 0) o16[uint64_t(t) * NH * HD + hh * HD + o] = sycl::half(acc);
        });
    });
}


// Interleaved (NORM) rope on the trailing RD dims of each HD-wide row —
// rotate ADJACENT pairs (v[2i], v[2i+1]), matching the deployed vLLM
// hunyuan_v4 is_neox_style=False (see docs/hy4/01_arch_delta_vs_ds4.md §0).
sycl::event k_hy4_rope_tail(sycl::queue& q, float* v, uint32_t T, uint32_t NH,
                            uint32_t HD, uint32_t RD, uint32_t pos0, float theta,
                            const std::vector<sycl::event>& deps) {
    return ie::ps(q, "hy4_rope_q", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<2>(uint64_t(T) * NH, RD / 2), [=](sycl::id<2> id) {
            const uint32_t th = uint32_t(id[0]), k = uint32_t(id[1]);
            const uint32_t t = th / NH;
            const float inv = sycl::pow(theta, -float(2 * k) / float(RD));
            const float ang = float(pos0 + t) * inv;
            const float c = sycl::cos(ang), s = sycl::sin(ang);
            float* p = v + uint64_t(th) * HD + (HD - RD) + 2 * k;
            const float a = p[0], b = p[1];
            p[0] = a * c - b * s;
            p[1] = a * s + b * c;
        });
    });
}

// q_lat concat tail: qcat[th, LAT + i] = roped q tail (last RD of the HD head).
sycl::event k_hy4_copy_qpe(sycl::queue& q, const float* qb, float* qcat,
                           uint32_t T, uint32_t NH, uint32_t HD, uint32_t RD,
                           uint32_t LAT, uint32_t CW,
                           const std::vector<sycl::event>& deps) {
    return ie::ps(q, "hy4_copy_qpe", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<2>(uint64_t(T) * NH, RD), [=](sycl::id<2> id) {
            const uint32_t th = uint32_t(id[0]), i = uint32_t(id[1]);
            qcat[uint64_t(th) * CW + LAT + i] = qb[uint64_t(th) * HD + (HD - RD) + i];
        });
    });
}

// Absorb form with strides: q_lat[t,h,l] = sum_d k_b[h][l][d] * q[t, h*qstride + d],
// d over the HD-wide NOPE prefix; output row stride `ostride` (the [T,NH,CW]
// attend query, whose roped tail k_hy4_copy_qpe fills).
sycl::event k_hy4_kabsorb(sycl::queue& q, const float* q_head, const sycl::half* k_b,
                          float* q_lat, uint32_t T, uint32_t NH, uint32_t LAT,
                          uint32_t HD, uint32_t qstride, uint32_t ostride,
                          const std::vector<sycl::event>& deps) {
    constexpr int SG = 16;
    return ie::ps(q, "hy4_kabsorb", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({uint64_t(T) * NH, uint64_t(LAT) * SG},
                                         {1, SG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const uint32_t th = uint32_t(it.get_group(0));
            const uint32_t l = uint32_t(it.get_group(1));
            const uint32_t lane = uint32_t(it.get_local_id(1));
            auto sg = it.get_sub_group();
            const uint32_t hh = th % NH;
            const float* qh = q_head + uint64_t(th) * qstride;
            const sycl::half* kb = k_b + (uint64_t(hh) * LAT + l) * HD;
            float acc = 0.f;
            for (uint32_t d = lane; d < HD; d += SG)
                acc = sycl::fma(float(kb[d]), qh[d], acc);
            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (lane == 0) q_lat[uint64_t(th) * ostride + l] = acc;
        });
    });
}

// kv row prep: rms-norm the LAT latent prefix (weighted), interleaved-rope the
// RD tail UNNORMED (the reference norms kv_cmpr only), store fp16 [T, CW].
sycl::event k_hy4_kv_prep(sycl::queue& q, const float* kv32, const float* norm_w,
                          sycl::half* out16, uint32_t T, uint32_t LAT, uint32_t RD,
                          uint32_t pos0, float theta, float eps,
                          const std::vector<sycl::event>& deps) {
    constexpr uint32_t WG = 256;
    const uint32_t CW = LAT + RD;
    return ie::ps(q, "hy4_kv_prep", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({T, WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t t = uint32_t(it.get_group(0));
            const uint32_t lane = uint32_t(it.get_local_id(1));
            const float* x = kv32 + uint64_t(t) * CW;
            sycl::half* y = out16 + uint64_t(t) * CW;
            float ss = 0.f;
            for (uint32_t d = lane; d < LAT; d += WG) ss += x[d] * x[d];
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float r = sycl::rsqrt(ss / float(LAT) + eps);
            for (uint32_t d = lane; d < LAT; d += WG)
                y[d] = sycl::half(x[d] * r * norm_w[d]);
            for (uint32_t k = lane; k < RD / 2; k += WG) {
                const float inv = sycl::pow(theta, -float(2 * k) / float(RD));
                const float ang = float(pos0 + t) * inv;
                const float c = sycl::cos(ang), s = sycl::sin(ang);
                const float a = x[LAT + 2 * k], b = x[LAT + 2 * k + 1];
                y[LAT + 2 * k]     = sycl::half(a * c - b * s);
                y[LAT + 2 * k + 1] = sycl::half(a * s + b * c);
            }
        });
    });
}

// Dense causal MLA attention over the CW-wide cache rows: scores contract the
// full CW (latent 512 + roped pe 64), the value accumulation reads the first
// LAT lanes only, and each head carries a learnable SINK logit that joins the
// softmax denominator without contributing value mass (gpt-oss semantics).
sycl::event k_hy4_attend(sycl::queue& q, const float* q_cat, const sycl::half* lat,
                         const float* sinks, float* att, uint32_t T, uint32_t NH,
                         uint32_t LAT, uint32_t CW, uint32_t pos0, float scale,
                         uint32_t slm_cap, const std::vector<sycl::event>& deps) {
    constexpr uint32_t WG = 128;
    return ie::ps(q, "hy4_attend", [&](sycl::handler& h) {
        h.depends_on(deps);
        sycl::local_accessor<float, 1> p_slm(slm_cap, h);
        h.parallel_for(sycl::nd_range<2>({uint64_t(T) * NH, WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t th = uint32_t(it.get_group(0));
            const uint32_t t = th / NH, hh = th % NH;
            const uint32_t lane = uint32_t(it.get_local_id(1));
            const uint32_t n = pos0 + t + 1;          // causal: j in [0, n)
            const float* qr = q_cat + uint64_t(th) * CW;

            float mx = -1e30f;
            for (uint32_t j = lane; j < n; j += WG) {
                const sycl::half* lj = lat + uint64_t(j) * CW;
                float s = 0.f;
                for (uint32_t d = 0; d < CW; ++d) s = sycl::fma(float(lj[d]), qr[d], s);
                s *= scale;
                p_slm[j] = s;
                mx = sycl::fmax(mx, s);
            }
            mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
            const float snk = sinks ? sinks[hh] : -1e30f;
            mx = sycl::fmax(mx, snk);
            float sum = 0.f;
            for (uint32_t j = lane; j < n; j += WG) {
                const float e = sycl::exp(p_slm[j] - mx);
                p_slm[j] = e;
                sum += e;
            }
            sum = sycl::reduce_over_group(it.get_group(), sum, sycl::plus<float>());
            if (sinks) sum += sycl::exp(snk - mx);
            const float inv = 1.f / sum;
            sycl::group_barrier(it.get_group());

            float* ar = att + uint64_t(th) * LAT;
            for (uint32_t d = lane; d < LAT; d += WG) {
                float acc = 0.f;
                for (uint32_t j = 0; j < n; ++j)
                    acc = sycl::fma(p_slm[j], float(lat[uint64_t(j) * CW + d]), acc);
                ar[d] = acc * inv;
            }
        });
    });
}

// y[i] *= sigmoid(g[i]) — the hyv4 attention gate on the decompressed output.
sycl::event k_hy4_gate_sigmul(sycl::queue& q, sycl::half* y, const float* g,
                              uint64_t n, const std::vector<sycl::event>& deps) {
    return ie::ps(q, "hy4_attn_gate", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            y[i] = sycl::half(float(y[i]) / (1.f + sycl::exp(-g[i])));
        });
    });
}

// ---- iHC (hyv4 hyper-connections: pre/post only, no comb/Sinkhorn) --------
sycl::event k_ihc_flatnorm(sycl::queue& q, const float* streams, float* flatN,
                           uint32_t T, uint32_t HW, float eps,
                           const std::vector<sycl::event>& deps) {
    constexpr uint32_t WG = 256;
    return ie::ps(q, "hy4_ihc_norm", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({T, WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t t = uint32_t(it.get_group(0));
            const uint32_t lane = uint32_t(it.get_local_id(1));
            const float* x = streams + uint64_t(t) * HW;
            float* y = flatN + uint64_t(t) * HW;
            float ss = 0.f;
            for (uint32_t d = lane; d < HW; d += WG) ss += x[d] * x[d];
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float r = sycl::rsqrt(ss / float(HW) + eps);
            for (uint32_t d = lane; d < HW; d += WG) y[d] = x[d] * r;
        });
    });
}

// Split-K projection: work-group (t, m, kc) of 256 lanes reduces its
// HW/KC slice; partials land in mix[(t*M+m)*KC + kc] and k_ihc_gates sums
// them in FIXED order (no atomics — cross-process determinism is certified).
// Profiled 2026-09-01: the one-subgroup-per-row version cost 0.26 ms/call
// (786 KB at 3 GB/s, 18% of decode GPU time).
constexpr uint32_t kIhcKC = 8;
sycl::event k_ihc_project(sycl::queue& q, const float* flatN, const float* fn,
                          float* mix, uint32_t T, uint32_t HW, uint32_t M,
                          const std::vector<sycl::event>& deps) {
    constexpr uint32_t WG = 256;
    const uint32_t chunk = HW / kIhcKC;
    return ie::ps(q, "hy4_ihc_proj", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({uint64_t(T) * M * kIhcKC, WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t g = uint32_t(it.get_group(0));
            const uint32_t kc = g % kIhcKC, tm = g / kIhcKC;
            const uint32_t t = tm / M, m = tm % M;
            const uint32_t lane = uint32_t(it.get_local_id(1));
            const float* x = flatN + uint64_t(t) * HW + uint64_t(kc) * chunk;
            const float* w = fn + uint64_t(m) * HW + uint64_t(kc) * chunk;
            float acc = 0.f;
            for (uint32_t d = lane; d < chunk; d += WG)
                acc = sycl::fma(w[d], x[d], acc);
            acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
            if (lane == 0) mix[(uint64_t(t) * M + m) * kIhcKC + kc] = acc;
        });
    });
}

sycl::event k_ihc_gates(sycl::queue& q, const float* mix, const float* scale2,
                        const float* base, float* pre, float* post,
                        uint32_t T, uint32_t hc, float eps, float mag,
                        const std::vector<sycl::event>& deps) {
    return ie::ps(q, "hy4_ihc_gates", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<2>(T, hc), [=](sycl::id<2> id) {
            const uint32_t t = uint32_t(id[0]), j = uint32_t(id[1]);
            const float* mp = mix + uint64_t(t) * 2 * hc * kIhcKC;
            float mj = 0.f, mpj = 0.f;
            for (uint32_t c = 0; c < kIhcKC; ++c) {           // fixed order
                mj  += mp[uint64_t(j) * kIhcKC + c];
                mpj += mp[uint64_t(hc + j) * kIhcKC + c];
            }
            pre[uint64_t(t) * hc + j] =
                1.f / (1.f + sycl::exp(-(mj * scale2[0] + base[j]))) + eps;
            post[uint64_t(t) * hc + j] =
                mag / (1.f + sycl::exp(-(mpj * scale2[1] + base[hc + j]))) + eps;
        });
    });
}

sycl::event k_ihc_collapse(sycl::queue& q, const float* streams, const float* pre,
                           float* coll, uint32_t T, uint32_t H, uint32_t hc,
                           const std::vector<sycl::event>& deps) {
    return ie::ps(q, "hy4_ihc_coll", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<2>(T, H), [=](sycl::id<2> id) {
            const uint32_t t = uint32_t(id[0]), d = uint32_t(id[1]);
            float s = 0.f;
            for (uint32_t j = 0; j < hc; ++j)
                s += pre[uint64_t(t) * hc + j] * streams[(uint64_t(t) * hc + j) * H + d];
            coll[uint64_t(t) * H + d] = s;
        });
    });
}

sycl::event k_hy4_rmsnorm_f32(sycl::queue& q, const float* x, const float* w,
                              float* y, uint32_t T, uint32_t H, float eps,
                              const std::vector<sycl::event>& deps) {
    constexpr uint32_t WG = 256;
    return ie::ps(q, "hy4_rms_f32", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::nd_range<2>({T, WG}, {1, WG}),
                       [=](sycl::nd_item<2> it) {
            const uint32_t t = uint32_t(it.get_group(0));
            const uint32_t lane = uint32_t(it.get_local_id(1));
            const float* xr = x + uint64_t(t) * H;
            float* yr = y + uint64_t(t) * H;
            float ss = 0.f;
            for (uint32_t d = lane; d < H; d += WG) ss += xr[d] * xr[d];
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float r = sycl::rsqrt(ss / float(H) + eps);
            for (uint32_t d = lane; d < H; d += WG) yr[d] = xr[d] * r * w[d];
        });
    });
}

// iHC post-mix: out[t,j,d] = streams[t,j,d] + post[t,j] * sub[t,d] (comb == I).
sycl::event k_ihc_mix(sycl::queue& q, const float* streams, const float* post,
                      const float* sub, float* out, uint32_t T, uint32_t H,
                      uint32_t hc, const std::vector<sycl::event>& deps) {
    return ie::ps(q, "hy4_ihc_mix", [&](sycl::handler& h) {
        h.depends_on(deps);
        h.parallel_for(sycl::range<2>(uint64_t(T) * hc, H), [=](sycl::id<2> id) {
            const uint32_t tj = uint32_t(id[0]), d = uint32_t(id[1]);
            const uint32_t t = tj / hc;
            out[uint64_t(tj) * H + d] =
                streams[uint64_t(tj) * H + d] + post[tj] * sub[uint64_t(t) * H + d];
        });
    });
}

// Dense causal MLA attention over the latent cache (absorbed MQA: the key AND
// value of position j are the same latent row). One work-group per (t, h);
// scores staged in SLM (n_ctx <= 2048 on hyv4 (kpool 1) by the caller's refusal), two-pass
// softmax in fp32.  att[t, h, :] = Σ_j p_j · lat[j, :].
sycl::event k_mla_attend(sycl::queue& q, const float* q_lat, const sycl::half* lat,
                         float* att, uint32_t T, uint32_t NH, uint32_t LAT,
                         uint32_t pos0, float scale, uint32_t slm_cap,
                         const std::vector<sycl::event>& deps) {
    constexpr uint32_t WG = 128;
    return ie::ps(q, "hy4_attend", [&](sycl::handler& h) {
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
sycl::event k_stream_mean(sycl::queue& q, const float* streams, float* x,
                          uint32_t T, uint32_t H, uint32_t hc,
                          const std::vector<sycl::event>& deps) {
    return ie::ps(q, "hy4_mean", [&](sycl::handler& h) {
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

sycl::event hyv4_router_logits(sycl::queue& q, const float* x, const float* w,
                               float* rl, uint32_t T, uint32_t H, uint32_t E,
                               const std::vector<sycl::event>& deps) {
    return k_router_logits(q, x, w, rl, T, H, E, deps);
}

std::string Hyv4Model::init_runtime(uint32_t max_ctx, uint32_t max_chunk) {
    max_ctx_ = max_ctx;
    max_chunk_ = max_chunk;
    const uint32_t H = cfg_.hidden, hc = cfg_.hc_count;
    // gemm_fp16 stores full 8x16 output tiles; every [MT, N] workspace is
    // padded to the next row-tile so a ragged T's tail tile lands in-buffer.
    const uint32_t MT = (max_chunk + 7) & ~7u;
    const uint32_t n_tf = cfg_.n_transformer_layers();
    const uint32_t DI = cfg_.n_q_heads * cfg_.kda_head_dim;   // 8192
    const uint32_t LAT = cfg_.kv_lora_rank;                   // 512
    const uint32_t CW  = LAT + cfg_.rope_dim;                 // 576 cache row
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
    lat_cache_ = static_cast<sycl::half*>(dev(uint64_t(n_full) * max_ctx * CW * 2));
    ok = ok && lat_cache_;
    d_tokens_ = static_cast<int32_t*>(dev(uint64_t(MT) * 4));
    ok = ok && d_tokens_;
    ok = ok && f32buf(wide_a_, uint64_t(MT) * hc * H) && f32buf(wide_b_, uint64_t(MT) * hc * H);
    ok = ok && f32buf(cpu_acc_dev_[0], H) && f32buf(cpu_acc_dev_[1], H);
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
    // T==1 f16 scratch must cover the WIDEST dense output: hyv4's blk.0 FFN is
    // 18432 (> the donor's 16384 assumption) and NH*HDM = 16384.
    const uint32_t scr_n = std::max<uint32_t>({16384u, cfg_.ffn, cfg_.n_q_heads * cfg_.value_len_mla});
    ok = ok && h16buf(mmscr16_, scr_n) && f32buf(mmscr32_, uint64_t(MT) * scr_n);
    ok = ok && h16buf(dense_bt_, uint64_t(6144) * 18432);   // raw-dense T>1 Bt
    ok = ok && h16buf(mla_qa16_, uint64_t(MT) * cfg_.q_lora_rank);
    ok = ok && f32buf(mla_qb_, uint64_t(MT) * cfg_.n_q_heads * cfg_.key_len_mla);
    ok = ok && f32buf(mla_qlat_, uint64_t(MT) * cfg_.n_q_heads * LAT);
    ok = ok && h16buf(mla_kv16_, uint64_t(MT) * CW);
    ok = ok && f32buf(mla_kv32_, uint64_t(MT) * CW);
    ok = ok && f32buf(mla_qcat_, uint64_t(MT) * cfg_.n_q_heads * CW);
    ok = ok && f32buf(gate32_, uint64_t(MT) * cfg_.n_q_heads * cfg_.value_len_mla);
    ok = ok && f32buf(ihc_pre_, uint64_t(MT) * hc);
    ok = ok && f32buf(ihc_flat_, uint64_t(MT) * hc * H);
    ok = ok && f32buf(ihc_mix_, uint64_t(MT) * 2 * hc * kIhcKC);   // split-K partials
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
    if (!ok) return "hyv4 init_runtime: device alloc failed";

    // Per-layer LRU expert slot cache. Budget from IE_HY4_ECACHE_MB (default
    // 10 GiB); slots split evenly across the MoE layers, floored at top_k so
    // one token's routing always fits, disabled below that.
    {
        uint64_t budget = 10240ull << 20;
        if (const char* v = std::getenv("IE_HY4_ECACHE_MB")) budget = uint64_t(std::atoll(v)) << 20;
        // EP decode: half the VRAM budget goes to this card's half-caches for
        // the PEER's layers (allocated later by ep_enable) — shrink ours now.
        if (std::getenv("IE_HY4_EP_DECODE")) budget /= 2;
        // The tail stage's ecache also serves the MTP block (blk.45).
        const uint32_t eco_hi = (mtp_loaded() && layer_hi_ == n_tf) ? n_tf + 1 : layer_hi_;
        uint32_t n_moe = 0;
        uint64_t slot_bytes_max = 0;
        std::vector<uint64_t> lslot(cfg_.n_layers, 0);
        for (uint32_t L = layer_lo_; L < eco_hi; ++L)
            if (!cfg_.is_dense_layer(L)) {
                ++n_moe;
                const Hyv4Layer& w = layers_[L];
                lslot[L] = w.gate_exps->nbytes / cfg_.n_experts +
                           w.up_exps->nbytes / cfg_.n_experts +
                           w.down_exps->nbytes / cfg_.n_experts;
                slot_bytes_max = std::max(slot_bytes_max, lslot[L]);
            }
        uint32_t slots = uint32_t(budget / (uint64_t(n_moe) * slot_bytes_max));
        slots = std::min(slots, cfg_.n_experts);
        if (slots >= cfg_.n_experts_used) {
            pin_sz_ = slot_bytes_max;
            for (uint32_t r = 0; r < kPinRing; ++r) {
                pin_ring_[r] = static_cast<uint8_t*>(
                    sycl::malloc_host(pin_sz_, alloc_->queue()));
                if (!pin_ring_[r]) return "hyv4 init_runtime: pinned ring alloc failed";
            }
            cpool_.start(3);
            copyq_ = std::make_unique<sycl::queue>(
                alloc_->queue().get_context(), alloc_->queue().get_device(),
                sycl::property::queue::in_order{});
            jobsq_ = std::make_unique<sycl::queue>(
                alloc_->queue().get_context(), alloc_->queue().get_device(),
                sycl::property::queue::in_order{});
            fill_.start(this);
            if (std::getenv("IE_HY4_PREFETCH")) {
                // Sized for ANY layer's slice, not just this stage's max —
                // peer-side prefetch stages PEER-stage slices (blk 11's
                // 17.56 MiB overran a local-sbm buffer: EFAULT'd preads).
                const uint64_t pf_sz = std::max<uint64_t>(pin_sz_, 20ull << 20);
                for (uint32_t r = 0; r < 2; ++r) {
                    pf_pin_[r] = static_cast<uint8_t*>(
                        sycl::malloc_host(pf_sz, alloc_->queue()));
                    if (!pf_pin_[r]) return "hyv4 init_runtime: pf ring alloc failed";
                }
            }
            if (std::getenv("IE_HY4_PP_TILES") || std::getenv("IE_HY4_PP_STREAM")) {
                // 16 device slices × 24 KB. Each tiles batch H2Ds its slice
                // on jobsq_ (main-thread, not copyq_) WITHOUT waiting.
                // Compute kernels depend on that event.
                pp_jobs_ = static_cast<int32_t*>(dev(16ull * 2048ull * 3 * 4));
                if (!pp_jobs_) return "hyv4 init_runtime: pp_jobs alloc failed";
                pp_hjobs_.reserve(2048 * 3);
            }
            if (std::getenv("IE_HY4_PP_STREAM")) {
                // biggest layer bank (gate+up+down) across owned MoE layers
                uint64_t mx = 0;
                for (uint32_t L = layer_lo_; L < eco_hi; ++L)
                    if (!cfg_.is_dense_layer(L))
                        mx = std::max(mx, layers_[L].gate_exps->nbytes +
                                          layers_[L].up_exps->nbytes +
                                          layers_[L].down_exps->nbytes);
                pp_stream_sz_ = mx;
                for (int b2 = 0; b2 < 2; ++b2) {
                    pp_stream_[b2] = static_cast<uint8_t*>(dev(mx));
                    if (!pp_stream_[b2])
                        return "hyv4 init_runtime: pp_stream alloc failed "
                               "(shrink IE_HY4_ECACHE_MB — stream needs 2 layer banks)";
                }
                const uint32_t npin = std::max(8u,
                    uint32_t((mx + kPpPinSz - 1) / kPpPinSz));
                pp_pin_.assign(npin, nullptr);
                pp_pin_ev_.assign(npin, {});
                for (uint32_t b2 = 0; b2 < npin; ++b2) {
                    pp_pin_[b2] = static_cast<uint8_t*>(
                        sycl::malloc_host(kPpPinSz, alloc_->queue()));
                    if (!pp_pin_[b2])
                        return "hyv4 init_runtime: pp_pin alloc failed";
                }
                std::fprintf(stderr,
                             "[hyv4] pp-stream: 2 x %.2f GiB layer buffers, "
                             "%u x %.0f MiB pin\n",
                             double(mx) / 1073741824.0, npin,
                             double(kPpPinSz) / (1024.0 * 1024.0));
            }
            ecache_slots_ = slots;
            // Per-layer slot budget (IE_HY4_SLOT_PROFILE = u64[n_layers x E]
            // pick counts, the IE_HY4_EPROFILE_OUT dump): greedy allocation by
            // marginal hits per byte — the k-th slot at layer L is worth that
            // layer's k-th-hottest expert count. Flat-routing layers earn more
            // slots than skewed ones. Uniform split without a profile.
            std::vector<uint32_t> slots_l(cfg_.n_layers, slots);
            // IE_HY4_UNIFORM_SLOTS=1: the pre-greedy uniform division (bisect
            // aid — 2026-08-29 divergence hunt: output text changed with slot
            // GEOMETRY, which correct caching never does).
            if (!std::getenv("IE_HY4_UNIFORM_SLOTS")) {
                // Budget-exact balanced fill by ACTUAL per-layer slot bytes.
                // The uniform slot_bytes_MAX division strands ~17% of the
                // budget (Opus 5 report, 2026-08-29): stage A 72 -> 87 slots
                // at identical VRAM.
                uint64_t left = budget;
                for (uint32_t L = layer_lo_; L < eco_hi; ++L)
                    if (lslot[L]) { slots_l[L] = 0; }
                bool grew = true;
                while (grew) {
                    grew = false;
                    uint32_t pick = UINT32_MAX, mn = UINT32_MAX;
                    for (uint32_t L = layer_lo_; L < eco_hi; ++L)
                        if (lslot[L] && slots_l[L] < cfg_.n_experts &&
                            lslot[L] <= left && slots_l[L] < mn) { mn = slots_l[L]; pick = L; }
                    if (pick != UINT32_MAX) { left -= lslot[pick]; ++slots_l[pick]; grew = true; }
                }
                uint32_t mnv = UINT32_MAX, mxv = 0;
                for (uint32_t L = layer_lo_; L < eco_hi; ++L)
                    if (lslot[L]) { mnv = std::min(mnv, slots_l[L]); mxv = std::max(mxv, slots_l[L]); }
                if (mnv != UINT32_MAX && mnv < cfg_.n_experts_used)
                    return "hyv4 init_runtime: ecache budget too small";
                slots = mnv == UINT32_MAX ? slots : mnv;   // report + eprio floor
            }
            if (const char* pf = std::getenv("IE_HY4_SLOT_PROFILE")) {
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
                        if (cfg_.is_dense_layer(L)) continue;
                        const Hyv4Layer& w = layers_[L];
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
                        if (!cfg_.is_dense_layer(L)) slots_l[L] = fl;
                    uint64_t left = budget - floor_cost;
                    // (value, layer) max-heap; value = count/byte of the next slot
                    using Cand = std::pair<double, uint32_t>;
                    std::priority_queue<Cand> pq;
                    for (uint32_t L = layer_lo_; L < eco_hi; ++L)
                        if (!cfg_.is_dense_layer(L) && fl < cfg_.n_experts)
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
                        if (!cfg_.is_dense_layer(L)) {
                            mn = std::min(mn, slots_l[L]);
                            mx = std::max(mx, slots_l[L]);
                        }
                    std::fprintf(stderr,
                                 "[hyv4] slot profile %s: per-layer slots %u..%u\n",
                                 pf, mn, mx);
                } else if (pf[0]) {
                    std::fprintf(stderr, "[hyv4] slot profile %s unreadable — uniform slots\n", pf);
                }
            }
            ecache_.resize(cfg_.n_layers);
            uint64_t ec_bytes = 0;
            // IE_HY4_ECACHE_PAD=<MiB>: dead padding before each layer's cache
            // (2026-08-29 divergence probe — shifts every device address while
            // keeping all counts identical; a text flip under padding proves
            // ADDRESS-dependence, i.e. an OOB read or address-dependent rot).
            const uint64_t ec_pad = [] {
                const char* v = std::getenv("IE_HY4_ECACHE_PAD");
                return v ? uint64_t(std::atoll(v)) << 20 : 0ull;
            }();
            for (uint32_t L = layer_lo_; L < eco_hi; ++L) {
                if (cfg_.is_dense_layer(L)) continue;
                if (ec_pad) (void)dev(ec_pad);
                const Hyv4Layer& w = layers_[L];
                ECache& ec = ecache_[L];
                const uint64_t gsl = w.gate_exps->nbytes / cfg_.n_experts;
                const uint64_t usl = w.up_exps->nbytes / cfg_.n_experts;
                const uint64_t dsl = w.down_exps->nbytes / cfg_.n_experts;
                ec.gate_off = 0; ec.up_off = gsl; ec.down_off = gsl + usl;
                ec.slot_bytes = gsl + usl + dsl;
                ec.base = static_cast<uint8_t*>(dev(uint64_t(slots_l[L]) * ec.slot_bytes));
                if (!ec.base) return "hyv4 init_runtime: ecache alloc failed";
                ec.slot_of.assign(cfg_.n_experts, -1);
                ec.expert_in.assign(slots_l[L], -1);
                ec.last_use.assign(slots_l[L], 0);
                if (std::getenv("IE_HY4_PREFETCH")) {   // empty == prefetch off
                    ec.pf_ev.assign(slots_l[L], {});
                    ec.pf_pending.assign(slots_l[L], 0);
                }
                ec_bytes += uint64_t(slots_l[L]) * ec.slot_bytes;
            }
            std::fprintf(stderr, "[hyv4] expert cache: %u slots/layer, %.2f GiB\n",
                         slots, double(ec_bytes) / 1073741824.0);
            // Profile-guided residency (IE_HY4_EPRIO=<counts file>): preload
            // each owned MoE layer's hottest experts into its slots and make
            // the top fraction STICKY (last_use = UINT64_MAX — the LRU victim
            // scan can never pick them). Measured 2026-08-28: per-layer
            // top-55 covers 74.8% of picks vs LRU's achieved 62.6%.
            if (const char* pf = std::getenv("IE_HY4_EPRIO")) {
                std::vector<uint64_t> counts(uint64_t(cfg_.n_layers) * cfg_.n_experts);
                FILE* f = std::fopen(pf, "rb");
                if (f && std::fread(counts.data(), 8, counts.size(), f) == counts.size()) {
                    uint32_t sticky_pct = 60;
                    if (const char* sp = std::getenv("IE_HY4_STICKY_PCT"))
                        sticky_pct = uint32_t(std::atoi(sp));
                    uint32_t n_sticky = 0;
                    sycl::queue& q = alloc_->queue();
                    for (uint32_t L = layer_lo_; L < eco_hi; ++L) {
                        if (cfg_.is_dense_layer(L)) continue;
                        ECache& ec = ecache_[L];
                        const uint32_t lsl = uint32_t(ec.expert_in.size());
                        n_sticky = lsl * sticky_pct / 100;
                        const Hyv4Layer& w = layers_[L];
                        std::vector<uint32_t> order(cfg_.n_experts);
                        for (uint32_t e = 0; e < cfg_.n_experts; ++e) order[e] = e;
                        const uint64_t* row = counts.data() + uint64_t(L) * cfg_.n_experts;
                        std::sort(order.begin(), order.end(),
                                  [&](uint32_t a, uint32_t b) { return row[a] > row[b]; });
                        const uint64_t gsl = w.gate_exps->nbytes / cfg_.n_experts;
                        const uint64_t usl = w.up_exps->nbytes / cfg_.n_experts;
                        const uint64_t dsl = w.down_exps->nbytes / cfg_.n_experts;
                        for (uint32_t s = 0; s < lsl; ++s) {
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
                    std::fprintf(stderr, "[hyv4] eprio preload: %u/%u sticky per layer (%s)\n",
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

void Hyv4Model::reset_state() {
    if (!alloc_) return;
    sycl::queue& q = alloc_->queue();
    dn_.reset(q);
    kv_len_ = 0;
    q.wait();
}

void Hyv4Model::mm(const sycl::half* A, const Hyv4DW& W, uint32_t T,
                       uint32_t N, uint32_t K, sycl::half* y16, float* y32) {
    sycl::queue& q = alloc_->queue();
    if (W.raw) {   // P4 raw K-quant residency
        // The single-row K-quant GEMVs stage the whole K-length activation in
        // SLM; above K=12288 that crosses the per-WG SLM ceiling (measured
        // 2026-09-01: UR OUT_OF_RESOURCES at the first decode step — attn_out
        // K=16384 / blk.0 ffn_down K=18432). Large K: rows variants read A
        // from global (Q4_K/Q5_K); Q6_K (blk.0 ffn_down only) takes the
        // dequant+gemm path even at T==1 (~0.5 ms/token for that one matrix).
        const bool big_k = K > 12288;
        if (T == 1 && !(big_k && W.rdt == DType::kQ6_K)) {
            sycl::half* y = y16 ? y16 : mmscr16_;
            switch (W.rdt) {
                case DType::kQ4_K:
                    if (big_k) gemv_q4_K_rows(q, A, W.raw, y, K, N, 1, {});
                    else       gemv_q4_K(q, A, W.raw, y, K, N, {});
                    break;
                case DType::kQ5_K:
                    if (big_k) gemv_q5_K_rows(q, A, W.raw, y, K, N, 1, {});
                    else       gemv_q5_K(q, A, W.raw, y, K, N, {});
                    break;
                default:       gemv_q6_K(q, A, W.raw, y, K, N, {}); break;
            }
            if (y32) cast_fp16_to_fp32(q, y, y32, N, {});
        } else {
            switch (W.rdt) {
                case DType::kQ4_K: dequant_q4_K_to_Bt(q, W.raw, dense_bt_, K, N, {}); break;
                case DType::kQ5_K: dequant_q5_K_to_Bt(q, W.raw, dense_bt_, K, N, {}); break;
                default:           dequant_q6_K_to_Bt(q, W.raw, dense_bt_, K, N, {}); break;
            }
            float* c = y32 ? y32 : mmscr32_;
            gemm_fp16(q, A, dense_bt_, c, T, N, K, {});
            if (y16) cast_fp32_to_fp16(q, c, y16, uint64_t(T) * N, {});
        }
        return;
    }
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

void Hyv4Model::mm_dual(const sycl::half* A, const Hyv4DW& Wa, const Hyv4DW& Wb,
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

std::string Hyv4Model::run_moe(uint32_t L, uint32_t T) {
    sycl::queue& q = alloc_->queue();
    const Hyv4Layer& w = layers_[L];
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
        ie::ps(q, "hy4_shexp_add", [&](sycl::handler& h) {
            const float* dn = moe_dn_; float* acc = moe_acc_;
            h.parallel_for(sycl::range<2>(T, H), [=](sycl::id<2> id) {
                acc[uint64_t(id[0]) * H + id[1]] += dn[uint64_t(id[0]) * H + id[1]];
            });
        });
    }
    e_d2h.wait();   // routing landed; shexp may still be in flight

    // R9 prediction SUBMIT (consumed at end-of-call): the next MoE layer's
    // router on THIS layer's xn_, D2H into pinned staging with no wait — the
    // in-order queue completes it well before the waves finish, so the
    // end-of-call collect is free (the old per-layer .wait() was a full
    // queue drain x42/token). moe_rl_/moe_top_/moe_topw_ reuse is safe:
    // their real contents landed in h_top_/h_topw_ above.
    static const bool shadow_cnt = std::getenv("IE_HY4_SHADOW_COUNT") != nullptr;
    static const int pf_depth = [] {
        const char* v = std::getenv("IE_HY4_PREFETCH");
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
    // objective function; bytes moved is. Opt in with IE_HY4_PF_SPEC=1.
    static const bool pf_spec = std::getenv("IE_HY4_PF_SPEC") != nullptr;
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
    const bool epv = std::getenv("IE_HY4_EP_VERIFY") != nullptr;
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
            // IE_HY4_EP_PUSH: P2P D2D WRITES are certified on this stack (only
            // reads corrupt) — push x straight into the peer's ep_x16_ and
            // have the peer push its partial straight into our ep_ret_.
            // 4 PCIe hops/layer (D2H + wait + H2D each way) become 2 direct
            // pushes; each side waits only its OWN events (trusted pattern).
            static const bool ep_push = std::getenv("IE_HY4_EP_PUSH") != nullptr;
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
    if (!ec) return "hyv4 moe: expert cache disabled (IE_HY4_ECACHE_MB too small)";
    float* acc_dst = moe_acc_;   // IE_HY4_EP_VERIFY redirects compute_e here
    const uint64_t gsl = w.gate_exps->nbytes / E;
    const uint64_t usl = w.up_exps->nbytes / E;
    const uint64_t dsl = w.down_exps->nbytes / E;

    // IE_HY4_PP_STREAM (T>4): whole-layer bank double-buffer replaces the
    // per-miss cache stream — 3 big sequential H2Ds per layer, the NEXT MoE
    // layer's loading behind THIS layer's compute. Buffer reuse is fenced
    // behind all submitted compute (the old occupant's readers).
    static const bool pp_stream_env = std::getenv("IE_HY4_PP_STREAM") != nullptr;
    const bool pp_stream = pp_stream_env && pp_stream_[0] && T > 4;
    const auto pps_call_t0 = std::chrono::steady_clock::now();
    const uint8_t *str_g = nullptr, *str_u = nullptr, *str_d = nullptr;
    sycl::event stream_dep;
    if (pp_stream) {
        auto load_layer = [&](uint32_t Ls, int buf) {
            const Hyv4Layer& wl = layers_[Ls];
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
        static const bool pps_sync = std::getenv("IE_HY4_PPS_SYNC") != nullptr;
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
            if (pf_busy(s)) continue;
            if (slot_wave[s] >= int32_t(cur_wave) - 1) continue;   // in-flight reader
            if (ec->expert_in[s] >= 0 && pending[ec->expert_in[s]]) continue;
            if (ec->last_use[s] < oldest) { oldest = ec->last_use[s]; victim = s; }
        }
        if (victim == UINT32_MAX) {
            int32_t far = -1;
            for (uint32_t s = 0; s < ec->expert_in.size(); ++s) {
                if (pf_busy(s)) continue;
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
                if (pf_busy(s)) continue;
                if (ec->expert_in[s] >= 0 && pending[ec->expert_in[s]]) continue;
                if (ec->last_use[s] < oldest) { oldest = ec->last_use[s]; victim = s; }
            }
            if (victim == UINT32_MAX) {
                int32_t far = -1;
                for (uint32_t s = 0; s < ec->expert_in.size(); ++s) {
                    if (pf_busy(s)) continue;
                    if (act_pos[ec->expert_in[s]] > far) { far = act_pos[ec->expert_in[s]]; victim = s; }
                }
            }
            if (victim == UINT32_MAX) { fill_err = "hyv4 moe: no evictable slot (pf saturation)"; return; }
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
        if (T <= 4) {
            // Decode/verify trickle: INLINE bounce fill — the fill thread's
            // per-miss wake/promise handoff measured decode 6.43 -> 5.26.
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
    // Diagnostic (IE_HY4_DUMP_MOE=<layer>, needs IE_HY4_DUMP_WIDE): stage that
    // layer's per-expert x / gate / up / down tensors as the kernels see them
    // (async D2H on the in-order q; forward_range flushes). Record layout:
    // u32{L,e,nt,path,r0,nrows,kind,len} + i32 tok[nrows] + u16 data[len].
    static const int dmoe_layer = [] {
        const char* v = std::getenv("IE_HY4_DUMP_MOE");
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
                case DType::kIQ2_XXS: gemv_iq2_xxs_raw(q, A, raw, yv, K, N, deps); break;
                case DType::kSTQ1_0:  gemv_stq1_0(q, A, raw, yv, K, N, deps); break;
                default:
                    return std::string("hyv4 moe: unsupported bank dtype ") +
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
                    gemv_q4_K_dual(q, xr, g_raw, u_raw, yg, yu, H, EF, "hy4_moe_gu", dep);
                } else {
                    if (auto err = gemv_raw(xr, g_raw, w.gate_exps->dtype, yg, H, EF, dep); !err.empty()) return err;
                    if (auto err = gemv_raw(xr, u_raw, w.up_exps->dtype,   yu, H, EF, {}); !err.empty()) return err;
                }
                dmoe(0, xr, H, e, nt, 0, r, 1); dmoe(1, yg, EF, e, nt, 0, r, 1); dmoe(2, yu, EF, e, nt, 0, r, 1);
                {   // Diagnostic (IE_HY4_NAN_PROBE): per-expert stage check on the
                    // small-T f16 path — where does the non-finite value enter?
                    static const bool nan_probe = std::getenv("IE_HY4_NAN_PROBE") != nullptr;
                    static int left = 6;
                    if (nan_probe && left > 0) {
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
                case DType::kIQ2_XXS: dequant_iq2_xxs_to_Bt(q, raw, dst, K, N, deps); break;
                case DType::kSTQ1_0:  dequant_stq1_0_to_Bt(q, raw, dst, K, N, deps); break;
                default:
                    return std::string("hyv4 moe: unsupported bank dtype ") +
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
    // the sequential scatter's fp32 rounding. OPT-IN (IE_HY4_GROUPED=1) until
    // the post-reboot oracle clears it: on the 2026-08-29 rotten driver the
    // EP oracle flagged (L14, tok 7) in both grouped runs with VARYING value
    // (race signature — suspect: up to 8 cross-queue fill events on one
    // launch vs the certified single-event pattern) — unattributable there.
    // Eligibility: gate/up Q4_K, down Q5_K/Q6_K (blk 11 falls back).
    // IE_HY4_GROUPED: 0/unset off; 1 strict single-launch (bit-identical);
    // 2 ready/late split — ready experts (hit or fill COMPLETE) launch as one
    // grouped job NOW, stragglers run solo so each starts the moment ITS
    // fill lands (strict grouped waits for the SLOWEST fill, which killed
    // the prefetcher's overlap: 7.15 -> 5.14 measured 2026-08-29). Mode 2
    // reorders the fp32 accumulation (EP-noise class) — text/PPL gated.
    static const int grouped_mode = [] {
        const char* v = std::getenv("IE_HY4_GROUPED");
        return v ? std::atoi(v) : 0;
    }();
    const bool grouped_on = grouped_mode >= 1;
    static const bool no_q6_slm = std::getenv("IE_NO_Q6K_SLM") != nullptr;
    const bool hy4_gu = (w.gate_exps->dtype == DType::kSTQ1_0 || w.gate_exps->dtype == DType::kIQ2_XXS) &&
                        w.up_exps->dtype == w.gate_exps->dtype;
    const bool hy4_dn = w.down_exps->dtype == DType::kIQ3_XXS || w.down_exps->dtype == DType::kIQ4_XS;
    const bool grouped = grouped_on && T == 1 &&
        ((w.gate_exps->dtype == DType::kQ4_K && w.up_exps->dtype == DType::kQ4_K &&
          (w.down_exps->dtype == DType::kQ5_K ||
           (w.down_exps->dtype == DType::kQ6_K && !no_q6_slm))) ||
         (hy4_gu && hy4_dn));
    // IE_HY4_PP_TILES: prefill launch-storm fix — a wave's nt>4 experts run
    // as tile batches (3 launches + per-expert scatters vs ~5 x nE); nt<=4
    // experts keep the small-T solo fold, so routing == the old path and the
    // whole layer stays bit-identical. Q6_K-down / Q5_K-gate layers solo.
    static const bool pp_tiles_env = std::getenv("IE_HY4_PP_TILES") != nullptr;
    // Stream+tiles measured 39.9 tok/s (GPU1 ring_wait 16 s) vs ~44-48
    // without — overlapping tiles compute with layer H2D starves BCS.
    // Keep tiles opt-in (IE_HY4_PP_TILES), not implicit with PP_STREAM.
    const bool pp_tiles = pp_tiles_env && pp_jobs_ && T > 4 &&
        w.gate_exps->dtype == DType::kQ4_K && w.up_exps->dtype == DType::kQ4_K &&
        w.down_exps->dtype == DType::kQ5_K;
    auto compute_grouped = [&](const std::vector<uint32_t>& act_v,
                               uint32_t w0, uint32_t w1) -> std::string {
        static bool once = [] {
            std::fprintf(stderr, "[hyv4] grouped MoE decode active\n");
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
            if (w.gate_exps->dtype == DType::kSTQ1_0)
                moe_gemv_stq1_0_gu_grouped(q, x16_, ec->base, ec->slot_bytes,
                                           ec->gate_off, ec->up_off, slots + r0, len,
                                           yg + uint64_t(r0) * EF, yu + uint64_t(r0) * EF,
                                           H, EF, dep);
            else if (w.gate_exps->dtype == DType::kIQ2_XXS)
                moe_gemv_iq2_xxs_gu_grouped(q, x16_, ec->base, ec->slot_bytes,
                                            ec->gate_off, ec->up_off, slots + r0, len,
                                            yg + uint64_t(r0) * EF, yu + uint64_t(r0) * EF,
                                            H, EF, dep);
            else
                moe_gemv_q4_K_gu_grouped(q, x16_, ec->base, ec->slot_bytes,
                                         ec->gate_off, ec->up_off, slots + r0, len,
                                         yg + uint64_t(r0) * EF, yu + uint64_t(r0) * EF,
                                         H, EF, dep);
            ds4_swiglu_clamped_h(q, yg + uint64_t(r0) * EF, yu + uint64_t(r0) * EF,
                                 yg + uint64_t(r0) * EF, uint64_t(len) * EF, clamp, {});
            switch (w.down_exps->dtype) {
                case DType::kIQ3_XXS:
                    moe_gemv_iq3_xxs_down_grouped(q, yg + uint64_t(r0) * EF, ec->base,
                                                  ec->slot_bytes, ec->down_off, slots + r0,
                                                  len, yd + uint64_t(r0) * H, EF, H, {});
                    break;
                case DType::kIQ4_XS:
                    moe_gemv_iq4_xs_down_grouped(q, yg + uint64_t(r0) * EF, ec->base,
                                                 ec->slot_bytes, ec->down_off, slots + r0,
                                                 len, yd + uint64_t(r0) * H, EF, H, {});
                    break;
                case DType::kQ5_K:
                    moe_gemv_q5_K_down_grouped(q, yg + uint64_t(r0) * EF, ec->base,
                                               ec->slot_bytes, ec->down_off, slots + r0,
                                               len, yd + uint64_t(r0) * H, EF, H, {});
                    break;
                default:
                    moe_gemv_q6_K_down_grouped(q, yg + uint64_t(r0) * EF, ec->base,
                                               ec->slot_bytes, ec->down_off, slots + r0,
                                               len, yd + uint64_t(r0) * H, EF, H, {});
                    break;
            }
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
    static const bool cpu_miss_on = [] {
        const char* v = std::getenv("IE_HY4_CPU_MISS");
        return v && v[0] == '1';
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
    // -> 0.44. IE_HY4_QSTAR overrides (1.0 == all-fill, 0.0 == grok's all-CPU
    // mode, which measured 3.59 tok/s vs 6.65 baseline: serial CPU compute
    // after the waves PLUS a DMA fill of every CPU expert). The CPU branch
    // runs on a worker thread launched BEFORE the wave submits.
    static const float qstar = [] {
        const char* v = std::getenv("IE_HY4_QSTAR");
        return v ? std::strtof(v, nullptr) : 0.44f;
    }();
    static const bool qstar_install = std::getenv("IE_HY4_QSTAR_INSTALL") != nullptr;
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
            cpu_acc_ring_[cpu_acc_sel_].assign(H, 0.f);
            cpu_xev = q.memcpy(cpu_x16h_.data(), x16_, uint64_t(H) * 2);
            const auto* gbase = reinterpret_cast<const block_q4_K*>(w.gate_src);
            const auto* ubase = reinterpret_cast<const block_q4_K*>(w.up_src);
            const auto* dbase = reinterpret_cast<const block_q5_K*>(w.down_src);
            const uint64_t gblk = gsl / sizeof(block_q4_K);
            const uint64_t ublk = usl / sizeof(block_q4_K);
            const uint64_t dblk = dsl / sizeof(block_q5_K);
            float* cbuf = cpu_acc_ring_[cpu_acc_sel_].data();
            const std::vector<uint32_t>& cact = cpu_act;
            if (cpu_worker_.core_lo < 0) {
                const char* cr = std::getenv(layer_lo_ == 0 ? "IE_HY4_CPU_CORES_A"
                                                            : "IE_HY4_CPU_CORES_B");
                int lo = -1, hi = -1;
                if (cr && std::sscanf(cr, "%d-%d", &lo, &hi) == 2) {
                    cpu_worker_.core_lo = lo; cpu_worker_.core_hi = hi;
                }
            }
            cpu_worker_.submit(
                [this, cpu_xev, gbase, ubase, dbase, gblk, ublk, dblk, cbuf,
                 &cact, &e_off, H, EF, clamp]() mutable {
                    cpu_xev.wait();
                    const auto tcb0 = std::chrono::steady_clock::now();
                    for (uint32_t i = 0; i < H; ++i) cpu_xf_[i] = float(cpu_x16h_[i]);
                    for (uint32_t e : cact) {
                        cpu_moe_expert_q8(cpu_xf_.data(),
                                          gbase + e * gblk, ubase + e * ublk,
                                          dbase + e * dblk,
                                          H, EF, clamp, cpu_y_.data());
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
    // Half-slot waves: by wave w+1's fills, wave w's computes have cleared
    // their pending flags — those slots become the stream-through victims,
    // and the far-future retained set survives across chunks.
    const uint32_t Wsz = std::max<uint32_t>(8, uint32_t(ec->expert_in.size()) / 2);
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
        cpu_worker_.wait();   // CPU branch ran concurrently with the GPU waves
        t_cpu_join += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - tw0).count();
        auto& cbuf = cpu_acc_ring_[cpu_acc_sel_];
        // q* semantics: CPU-executed experts leave residency unchanged (the
        // q* fills are the cache's admissions). IE_HY4_QSTAR_INSTALL=1 restores
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
        q.memcpy(dacc, cbuf.data(), uint64_t(H) * 4);
        float* acc = acc_dst;
        ie::ps(q, "hy4_cpu_acc", [&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(H), [=](sycl::id<1> id) {
                acc[id[0]] += dacc[id[0]];
            });
        });
        cpu_acc_sel_ ^= 1u;
        static bool once = [] {
            std::fprintf(stderr, "[hyv4] CPU-miss MoE decode active (IE_HY4_CPU_MISS=1)\n");
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
    // ~free by now). IE_HY4_SHADOW_COUNT scores the proxy (measured 76.8%,
    // 2026-08-29); IE_HY4_PREFETCH streams the predicted misses into Ln's
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
            const Hyv4Layer& wn = layers_[Ln];
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
        static const bool ep_push2 = std::getenv("IE_HY4_EP_PUSH") != nullptr;
        ep_ret_ev = ep_ret_fut_.get();   // worker finished submitting the chain
        ep_ret_ev.wait();                // peer ret landed (host-side sync)
        if (!ep_push2)   // push mode: partial is ALREADY in ep_ret_ (P2P write)
            q.memcpy(ep_ret_, ep_rstage_, uint64_t(T) * H * 4);  // H2D (our link)
        float* acc = moe_acc_;
        const float* ret = ep_ret_;
        ep_add_ev_ = ie::ps(q, "hy4_ep_add", [&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(uint64_t(T) * H), [=](sycl::id<1> i) {
                acc[i[0]] += ret[i[0]];
            });
        });
        // IE_HY4_EP_VERIFY: recompute the peer half LOCALLY and diff against
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
            // IE_HY4_EPV_DUMP=<dir>: on the FIRST mismatch, dump everything a
            // CPU ground-truth arbiter needs — x, the peer expert list with
            // routing weights, pa (peer partial), pb (local recompute). The
            // arbiter (tools/hyv4_epv_ref) recomputes from the disk-
            // verified bank bytes and says WHICH side is wrong.
            static bool dumped = false;
            const char* dd = std::getenv("IE_HY4_EPV_DUMP");
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
        static const bool pps_sync2 = std::getenv("IE_HY4_PPS_SYNC") != nullptr;
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

// iHC pre-site: flat-RMS -> fn projection -> pre/post gates -> weighted
// collapse -> weighted RMS norm into xn_. Replaces the donor's Sinkhorn
// ds4_hyper_connection_norm (hyv4 has no comb term; post carries magnitude
// and +eps — docs/hy4/01_arch_delta_vs_ds4.md §2).
void Hyv4Model::ihc_pre_norm(sycl::queue& q, const float* fn, const float* base,
                             const float* scale, const float* norm_w, uint32_t T) {
    const uint32_t H = cfg_.hidden, hc = cfg_.hc_count;
    k_ihc_flatnorm(q, wide_a_, ihc_flat_, T, hc * H, cfg_.rms_eps, {});
    k_ihc_project(q, ihc_flat_, fn, ihc_mix_, T, hc * H, 2 * hc, {});
    k_ihc_gates(q, ihc_mix_, scale, base, ihc_pre_, post_, T, hc,
                cfg_.hc_eps, cfg_.hc_magnitude, {});
    k_ihc_collapse(q, wide_a_, ihc_pre_, coll_, T, H, hc, {});
    k_hy4_rmsnorm_f32(q, coll_, norm_w, xn_, T, H, cfg_.rms_eps, {});
}

std::string Hyv4Model::run_block(uint32_t L, uint32_t T, uint32_t start_pos) {
    sycl::queue& q = alloc_->queue();
    const Hyv4Layer& w = layers_[L];
    const uint32_t H = cfg_.hidden, hc = cfg_.hc_count;
    const uint32_t DI = cfg_.n_q_heads * cfg_.kda_head_dim;
    const uint32_t NH = cfg_.n_q_heads, HD = cfg_.kda_head_dim;
    const uint32_t LAT = cfg_.kv_lora_rank;

    // ---- attention site: HC pre + norm --------------------------------------
    ihc_pre_norm(q, w.hc_attn_fn, w.hc_attn_base, w.hc_attn_scale, w.attn_norm, T);
    cast_fp32_to_fp16(q, xn_, x16_, uint64_t(T) * H, {});

    if (cfg_.is_full_attn(L)) {
        // ---- MLA (absorbed; interleaved-roped trailing 64, sinks, gate) ----
        const uint32_t QL = cfg_.q_lora_rank, HDM = cfg_.key_len_mla;
        const uint32_t RD = cfg_.rope_dim, CW = LAT + RD;
        mm(x16_, w.q_a, T, QL, H, mla_qa16_, nullptr);
        rms_norm_f32w(q, mla_qa16_, w.q_a_norm, mla_qa16_, T, QL, cfg_.rms_eps, {});
        mm(mla_qa16_, w.q_b, T, NH * HDM, QL, nullptr, mla_qb_);
        k_hy4_rope_tail(q, mla_qb_, T, NH, HDM, RD, start_pos, cfg_.rope_theta, {});
        // absorb the (HDM-RD)-wide nope prefix into the latent; the roped q
        // tail lands in the last RD lanes of the [T, NH, CW] attend query.
        k_hy4_kabsorb(q, mla_qb_, w.k_b, mla_qcat_, T, NH, LAT, HDM - RD, HDM, CW, {});
        k_hy4_copy_qpe(q, mla_qb_, mla_qcat_, T, NH, HDM, RD, LAT, CW, {});

        mm(x16_, w.kv_a, T, CW, H, nullptr, mla_kv32_);
        k_hy4_kv_prep(q, mla_kv32_, w.kv_a_norm, mla_kv16_, T, LAT, RD, start_pos,
                      cfg_.rope_theta, cfg_.rms_eps, {});
        // append this chunk's CW-wide rows to the cache (D2D; in-order queue
        // sequences it before the attend below — no host sync needed)
        sycl::half* lay_cache = lat_cache_ +
            (uint64_t(full_idx_[L]) * max_ctx_ + start_pos) * CW;
        q.memcpy(lay_cache, mla_kv16_, uint64_t(T) * CW * 2);

        const float scale = 1.f / std::sqrt(float(HDM));
        const uint32_t n_sel = cfg_.indexer_top_k + cfg_.indexer_kpool - 1;
        k_hy4_attend(q, mla_qcat_, lat_cache_ + uint64_t(full_idx_[L]) * max_ctx_ * CW,
                     w.attn_sinks, mla_att_, T, NH, LAT, CW, start_pos, scale, n_sel, {});
        k_mla_vabsorb(q, mla_att_, w.v_b, mla_o16_, T, NH, LAT, cfg_.value_len_mla, {});
        // sigmoid attention gate, computed from the SAME normed input x16_
        mm(x16_, w.attn_gate, T, NH * cfg_.value_len_mla, H, nullptr, gate32_);
        k_hy4_gate_sigmul(q, mla_o16_, gate32_, uint64_t(T) * NH * cfg_.value_len_mla, {});
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
        // kda_y16_ serves as its out buffer serially — the queue is in-order.
        sycl::half* cs = dn_.conv_state_ptr() +
                         uint64_t(li) * dn_.conv_elems_per_layer();
        const uint32_t cs_per = (cfg_.conv_kernel - 1) * DI;
        depthwise_conv1d_causal(q, kda_h16a_, w.conv_q, cs,              kda_y16_, T, DI, cfg_.conv_kernel, {});
        cast_fp16_to_fp32(q, kda_y16_, kda_f32a_, uint64_t(T) * DI, {});
        depthwise_conv1d_causal(q, kda_h16b_, w.conv_k, cs + cs_per,     kda_y16_, T, DI, cfg_.conv_kernel, {});
        cast_fp16_to_fp32(q, kda_y16_, kda_f32b_, uint64_t(T) * DI, {});
        depthwise_conv1d_causal(q, kda_h16c_, w.conv_v, cs + 2 * cs_per, kda_y16_, T, DI, cfg_.conv_kernel, {});
        cast_fp16_to_fp32(q, kda_y16_, kda_f32c_, uint64_t(T) * DI, {});
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
    k_ihc_mix(q, wide_a_, post_, mix_out_, wide_b_, T, H, hc, {});
    std::swap(wide_a_, wide_b_);
    if (!dump_h_.empty()) {   // IE_HY4_DUMP_WIDE (forward_range): async D2H on the in-order q
        const uint64_t n = uint64_t(T) * hc * H;
        if (dump_cur_ + n <= dump_h_.size()) {
            q.memcpy(dump_h_.data() + dump_cur_, wide_a_, n * 4);
            dump_cur_ += n;
            dump_hdr_.insert(dump_hdr_.end(), {L, 0u, T, start_pos, uint32_t(n)});
        }
    }
    {   // Diagnostic (IE_HY4_NAN_PROBE): mixer-stage check, pairs with the
        // per-layer probe in forward_range to split mixer vs FFN.
        static const bool nan_probe = std::getenv("IE_HY4_NAN_PROBE") != nullptr;
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
    ihc_pre_norm(q, w.hc_ffn_fn, w.hc_ffn_base, w.hc_ffn_scale, w.ffn_norm, T);
    cast_fp32_to_fp16(q, xn_, x16_, uint64_t(T) * H, {});

    if (cfg_.is_dense_layer(L)) {
        const uint32_t FF = cfg_.ffn;
        mm(x16_, w.ffn_gate, T, FF, H, nullptr, moe_gu_[0]);
        mm(x16_, w.ffn_up,   T, FF, H, nullptr, moe_gu_[1]);
        // hyv4 dense FFN is plain SiLU — the GGUF's swiglu_clamp_exp array
        // covers ROUTED experts only (the reference build_ffn has no clamp).
        ds4_swiglu_clamped(q, moe_gu_[0], moe_gu_[1], moe_gu_[0],
                           uint64_t(T) * FF, std::numeric_limits<float>::infinity(), {});
        cast_fp32_to_fp16(q, moe_gu_[0], moe_h16_, uint64_t(T) * FF, {});
        mm(moe_h16_, w.ffn_down, T, H, FF, nullptr, moe_acc_);
    } else {
        if (auto e = run_moe(L, T); !e.empty()) return e;
    }

    k_ihc_mix(q, wide_a_, post_, moe_acc_, wide_b_, T, H, hc, {});
    std::swap(wide_a_, wide_b_);
    if (!dump_h_.empty()) {   // IE_HY4_DUMP_WIDE (forward_range): async D2H on the in-order q
        const uint64_t n = uint64_t(T) * hc * H;
        if (dump_cur_ + n <= dump_h_.size()) {
            q.memcpy(dump_h_.data() + dump_cur_, wide_a_, n * 4);
            dump_cur_ += n;
            dump_hdr_.insert(dump_hdr_.end(), {L, 1u, T, start_pos, uint32_t(n)});
        }
    }
    {
        static const bool pps_sync = std::getenv("IE_HY4_PPS_SYNC") != nullptr;
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

std::string Hyv4Model::mtp_step(const float* hidden_row, int32_t tok, uint32_t pos) {
    if (!mtp_loaded()) return "hyv4 mtp: not loaded on this stage";
    if (pos >= max_ctx_ || pos < mtp_base_) return "hyv4 mtp: bad pos";
    const uint32_t n_sel = cfg_.indexer_top_k + cfg_.indexer_kpool - 1;
    if (pos - mtp_base_ + 1 > n_sel) return "hyv4 mtp: history exceeds v0 dense cap";
    sycl::queue& q = alloc_->queue();
    const uint32_t H = cfg_.hidden, NH = cfg_.n_q_heads, LAT = cfg_.kv_lora_rank;
    const uint32_t HDM = cfg_.key_len_mla, QL = cfg_.q_lora_rank;
    const uint32_t n_tf = cfg_.n_transformer_layers();
    const Hyv4Layer& w = layers_[n_tf];

    // fusion: x = eh_proj([RMS(embed(tok), enorm) | RMS(hidden, hnorm)])
    {
        const sycl::half* emb = token_embd;
        sycl::half* e16 = mtp_e16_;
        const uint64_t row = uint64_t(tok) * H;
        ie::ps(q, "hy4_mtp_embed", [&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(H), [=](sycl::id<1> i) { e16[i] = emb[row + i]; });
        });
    }
    rms_norm_f32w(q, mtp_e16_, mtp_enorm_, mtp_cat_, 1, H, cfg_.rms_eps, {});
    cast_fp32_to_fp16(q, hidden_row, mtp_x16_, H, {});
    rms_norm_f32w(q, mtp_x16_, mtp_hnorm_, mtp_cat_ + H, 1, H, cfg_.rms_eps, {});
    mm(mtp_cat_, mtp_eh_, 1, H, 2 * H, mtp_x16_, mtp_x_);

    auto add_row = [&](const float* src) {
        float* dst = mtp_x_;
        ie::ps(q, "hy4_mtp_add", [&](sycl::handler& h) {
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
    k_mla_attend(q, mla_qlat_, mtp_lat_ + uint64_t(mtp_base_) * LAT, mla_att_,
                 1, NH, LAT, pos - mtp_base_, 1.f / std::sqrt(float(HDM)), n_sel, {});
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

std::string Hyv4Model::snapshot_state() {
    sycl::queue& q = alloc_->queue();
    const uint64_t sb = uint64_t(dn_.config().n_layers_linear) * dn_.state_elems_per_layer() * 4;
    const uint64_t cb = uint64_t(dn_.config().n_layers_linear) * dn_.conv_elems_per_layer() * 2;
    q.memcpy(snap_dn_, dn_.state_ptr(), sb);
    q.memcpy(snap_conv_, dn_.conv_state_ptr(), cb);
    q.wait();
    snap_kv_len_ = kv_len_;
    snap_mtp_len_ = mtp_len_;
    return {};
}

std::string Hyv4Model::commit_verify(uint32_t n) {
    if (!spec_verify_ || spec_T_ == 0) return "hyv4 commit: no spec-verify forward recorded";
    if (n == 0 || n > spec_T_) return "hyv4 commit: bad n";
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
    const uint32_t cs_per = (cfg_.conv_kernel - 1) * DI;
    for (uint32_t L = layer_lo_; L < layer_hi_; ++L) {
        const int32_t li = lin_idx_[L];
        if (li < 0) continue;
        const Hyv4Layer& w = layers_[L];
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

std::string Hyv4Model::restore_state() {
    sycl::queue& q = alloc_->queue();
    const uint64_t sb = uint64_t(dn_.config().n_layers_linear) * dn_.state_elems_per_layer() * 4;
    const uint64_t cb = uint64_t(dn_.config().n_layers_linear) * dn_.conv_elems_per_layer() * 2;
    q.memcpy(dn_.state_ptr(), snap_dn_, sb);
    q.memcpy(dn_.conv_state_ptr(), snap_conv_, cb);
    q.wait();
    kv_len_ = snap_kv_len_;
    mtp_len_ = snap_mtp_len_;
    return {};
}

void Hyv4Model::PreadPool::start(int n) {
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
bool Hyv4Model::PreadPool::run(const Job* j, size_t n) {
    if (!n) return true;
    std::unique_lock<std::mutex> lk(mu);
    jobs = j; njobs = n; next = 0; done = 0; fail = false;
    cv.notify_all();
    cvd.wait(lk, [&] { return done == njobs; });
    return !fail;
}
void Hyv4Model::PreadPool::shutdown() {
    {
        std::lock_guard<std::mutex> lk(mu);
        stop = true;
    }
    cv.notify_all();
    for (auto& w : ws) if (w.joinable()) w.join();
    ws.clear();
}

std::string Hyv4Model::pread_fill_(int gfd, uint64_t goff, const uint8_t* gs, uint64_t gsl,
                                       int ufd, uint64_t uoff, const uint8_t* us, uint64_t usl,
                                       int dfd, uint64_t doff, const uint8_t* ds2, uint64_t dsl,
                                       uint8_t* dst, PreadPool* pool, bool serial_cp) {
    static const bool use_pread = [] {
        const char* v = std::getenv("IE_HY4_PREAD");
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
        return "hyv4: pread fill failed (short read / IO error) — refusing DMA";
    return {};
}

// Peer-side prefetch (EpThread only — the sole ep_cache_ mutator): warm
// ep_cache_[L] with predicted experts of OUR parity through OUR fill
// thread's pf path (our PCIe link). Bank sources live on the layer's OWNER.
void Hyv4Model::ep_prefetch(uint32_t L, const int32_t* p8) {
    if (!ep_peer_ || L >= ep_cache_.size() || !ep_cache_[L].base) return;
    ECache& ec = ep_cache_[L];
    if (ec.pf_pending.empty()) return;
    const Hyv4Layer& w = ep_peer_->layers_[L];
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

void Hyv4Model::EpThread::start(Hyv4Model* m) {
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
            if (!err.empty()) std::fprintf(stderr, "[hyv4] ep worker: %s\n", err.c_str());
            r.done.set_value(done_ev);
        }
    });
}
std::shared_future<sycl::event> Hyv4Model::EpThread::push(EpReq&& r) {
    auto fut = r.done.get_future().share();
    {
        std::lock_guard<std::mutex> lk(mu);
        reqs.push_back(std::move(r));
    }
    cv.notify_one();
    return fut;
}
Hyv4Model::EpThread::~EpThread() {
    { std::lock_guard<std::mutex> lk(mu); stop = true; }
    cv.notify_all();
    if (th.joinable()) th.join();
}

std::string Hyv4Model::warm_banks() {
    if (!copyq_) return {};
    const uint64_t sz = 128ull << 20;
    uint8_t* scratch = static_cast<uint8_t*>(sycl::malloc_device(sz, *copyq_));
    if (!scratch) return "hyv4 warm_banks: scratch alloc failed";
    for (uint32_t L = layer_lo_; L < cfg_.n_layers; ++L) {
        const Hyv4Layer& w = layers_[L];
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

std::string Hyv4Model::warm_cache(const char* profile_path) {
    if (!copyq_ || ecache_.empty()) return {};
    const uint32_t E = cfg_.n_experts;
    std::vector<uint64_t> counts(uint64_t(cfg_.n_layers) * E);
    FILE* f = std::fopen(profile_path, "rb");
    if (!f) return std::string("hyv4 warm_cache: cannot open ") + profile_path;
    const bool ok = std::fread(counts.data(), 8, counts.size(), f) == counts.size();
    std::fclose(f);
    if (!ok) return "hyv4 warm_cache: short profile (want u64[n_layers x E])";
    std::vector<std::shared_future<sycl::event>> pend;
    uint64_t bytes = 0;
    for (uint32_t L = layer_lo_; L < uint32_t(ecache_.size()); ++L) {
        ECache& ec = ecache_[L];
        if (!ec.base || !layers_[L].gate_exps) continue;
        const Hyv4Layer& w = layers_[L];
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
    std::fprintf(stderr, "[hyv4] warm cache: %zu experts, %.2f GiB\n",
                 pend.size(), double(bytes) / 1073741824.0);
    return {};
}

std::string Hyv4Model::ep_enable(Hyv4Model& peer, uint32_t parity) {
    if (!copyq_) return "hyv4 ep_enable: expert cache disabled";
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
        return "hyv4 ep_enable: buffer alloc failed";
    // Half-caches for the PEER's pinned MoE layers — the second half of this
    // card's IE_HY4_ECACHE_MB (init_runtime kept only half for our own layers).
    uint64_t budget = 10240ull << 20;
    if (const char* v = std::getenv("IE_HY4_ECACHE_MB")) budget = uint64_t(std::atoll(v)) << 20;
    budget /= 2;
    uint32_t n_pl = 0;
    uint64_t sbm = 0;
    const uint32_t n_tf = cfg_.n_transformer_layers();
    for (uint32_t L = 0; L < n_tf; ++L) {
        if (L >= peer.ecache_.size() || !peer.ecache_[L].base) continue;
        ++n_pl;
        sbm = std::max(sbm, peer.ecache_[L].slot_bytes);
    }
    if (!n_pl) return "hyv4 ep_enable: peer has no MoE layers";
    uint32_t slots = uint32_t(budget / (uint64_t(n_pl) * sbm));
    slots = std::min(slots, E);
    // hyv4 has no MTP/spec windows: EP decode is strictly T==1, so a half-
    // parity call routes at most top_k picks (all 8 could share a parity).
    // (The donor's 4*top_k floor covered GLM's T<=4 verify windows.)
    if (slots < cfg_.n_experts_used)
        return "hyv4 ep_enable: budget too small (" + std::to_string(slots) + " slots)";
    ep_cache_.resize(cfg_.n_layers);
    uint64_t total = 0;
    for (uint32_t L = 0; L < n_tf; ++L) {
        if (L >= peer.ecache_.size() || !peer.ecache_[L].base) continue;
        const ECache& pec = peer.ecache_[L];
        ECache& ec = ep_cache_[L];
        ec.gate_off = pec.gate_off; ec.up_off = pec.up_off; ec.down_off = pec.down_off;
        ec.slot_bytes = pec.slot_bytes;
        ec.base = static_cast<uint8_t*>(dev(uint64_t(slots) * ec.slot_bytes));
        if (!ec.base) return "hyv4 ep_enable: ep cache alloc failed";
        ec.slot_of.assign(E, -1);
        ec.expert_in.assign(slots, -1);
        ec.last_use.assign(slots, 0);
        if (std::getenv("IE_HY4_PREFETCH")) {
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
        if (!ep_pin_[r]) return "hyv4 ep_enable: bounce ring alloc failed";
    }
    ep_xstage_ = static_cast<sycl::half*>(sycl::malloc_host(8ull * H * 2, alloc_->queue()));
    ep_rstage_ = static_cast<float*>(sycl::malloc_host(8ull * H * 4, alloc_->queue()));
    if (!ep_xstage_ || !ep_rstage_) return "hyv4 ep_enable: staging alloc failed";
    std::fprintf(stderr,
                 "[hyv4] EP decode: parity %u, %u peer-layer half-caches, %u slots (%.2f GiB)\n",
                 parity, n_pl, slots, double(total) / 1073741824.0);
    return {};
}

// Peer-side half of one MoE layer at T<=4: fill misses from the owner's
// pinned host bank on OUR PCIe link, run the small-T gemv chain out of OUR
// half-cache, and land the fp32 partial in the owner's ep_ret_. Runs on this
// card's queues; `x_ev` = the owner's activation P2P copy, `ret_dep` = the
// owner's previous partial-add (ret_dst reuse fence).
std::string Hyv4Model::ep_partial(const Hyv4Layer& w, uint32_t L, uint32_t T,
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
        if (v == UINT32_MAX) return "hyv4 ep: half-cache smaller than one call";
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
        const char* v = std::getenv("IE_HY4_GROUPED");
        return v && v[0] == '1';
    }();
    static const bool ep_no_q6_slm = std::getenv("IE_NO_Q6K_SLM") != nullptr;
    if (ep_grouped_on && T == 1 && n_exp <= 8 &&
        w.gate_exps->dtype == DType::kQ4_K && w.up_exps->dtype == DType::kQ4_K &&
        (w.down_exps->dtype == DType::kQ5_K ||
         (w.down_exps->dtype == DType::kQ6_K && !ep_no_q6_slm))) {
        static bool once = [] {
            std::fprintf(stderr, "[hyv4] grouped MoE decode active (EP peer)\n");
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
            case DType::kIQ2_XXS: gemv_iq2_xxs_raw(pq, A, raw, yv, K, N, deps); break;
            case DType::kSTQ1_0:  gemv_stq1_0(pq, A, raw, yv, K, N, deps); break;
            default:
                return std::string("hyv4 ep: unsupported bank dtype ") +
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
                gemv_q4_K_dual(pq, xr, g_raw, u_raw, yg, yu, H, EF, "hy4_moe_gu", dep);
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

std::string Hyv4Model::forward(const int32_t* tokens_host, uint32_t T,
                                   uint32_t start_pos, sycl::half* logits_out,
                                   float* all_logits) {
    return forward_range(tokens_host, T, start_pos, nullptr, nullptr,
                         logits_out, all_logits);
}


void Hyv4Model::CpuWorker::start() {
    if (th.joinable()) return;
    th = std::thread([this] {
        if (core_lo >= 0 && core_hi >= core_lo) {
            // IE_HY4_CPU_CORES=lo-hi per stage: pin the worker (and the OpenMP
            // team it spawns, which inherits this mask) so two stages' teams
            // never share cores. Probe 2026-08-31: both teams "close"-bound
            // on the same 8 P-cores -> 1.44 ms/expert vs 0.25 in isolation.
            cpu_set_t set; CPU_ZERO(&set);
            for (int c = core_lo; c <= core_hi; ++c) CPU_SET(c, &set);
            sched_setaffinity(0, sizeof(set), &set);
        }
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
            {
                std::lock_guard<std::mutex> lk(mu);
                done = true;
            }
            cv.notify_all();
        }
    });
}
void Hyv4Model::CpuWorker::submit(std::function<void()> j) {
    start();
    {
        std::lock_guard<std::mutex> lk(mu);
        job = std::move(j);
        has_job = true;
        done = false;
    }
    cv.notify_all();
}
void Hyv4Model::CpuWorker::wait() {
    std::unique_lock<std::mutex> lk(mu);
    cv.wait(lk, [this] { return done; });
}
Hyv4Model::CpuWorker::~CpuWorker() {
    {
        std::lock_guard<std::mutex> lk(mu);
        quit = true;
    }
    cv.notify_all();
    if (th.joinable()) th.join();
}

std::string Hyv4Model::forward_range(const int32_t* tokens_host, uint32_t T,
                                         uint32_t start_pos, const float* wide_in_host,
                                         float* wide_out_host, sycl::half* logits_out,
                                         float* all_logits) {
    if (!lat_cache_) return "hyv4 forward: init_runtime not called";
    if (T == 0 || T > max_chunk_) return "hyv4 forward: bad T";
    if (start_pos != kv_len_) return "hyv4 forward: non-contiguous position";
    // v0 is EXACT dense MLA, which equals the sparse path only while every
    // position is selected (n_select = top_k + kpool - 1 = 2048 on hyv4 (kpool 1) on the real
    // file). Refuse honestly beyond it — the indexer is a later phase.
    const uint32_t n_sel = cfg_.indexer_top_k + cfg_.indexer_kpool - 1;
    if (start_pos + T > n_sel)
        return "hyv4 forward: v0 dense==sparse only to " + std::to_string(n_sel) +
               " positions (indexer not yet implemented)";
    if (start_pos + T > max_ctx_) return "hyv4 forward: exceeds max_ctx";

    sycl::queue& q = alloc_->queue();
    const uint32_t H = cfg_.hidden, hc = cfg_.hc_count;
    const uint32_t n_tf = cfg_.n_transformer_layers();
    // IE_HY4_TRACE_FIRST=1: time every layer of THIS stage's first forward
    // (the ~50s stage-A first-forward anomaly hunt). The per-layer q.wait()
    // distorts overlap — diagnosis only.
    const bool trace = !traced_first_ && std::getenv("IE_HY4_TRACE_FIRST") != nullptr;
    const auto tf0 = std::chrono::steady_clock::now();
    auto since = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - tf0).count();
    };
    if (wide_in_host == nullptr) {
        if (layer_lo_ != 0) return "hyv4 forward_range: no wide input for a tail stage";
        q.memcpy(d_tokens_, tokens_host, uint64_t(T) * 4).wait();
        k_embed_streams(q, d_tokens_, token_embd, wide_a_, T, H, hc, {});
    } else {
        q.memcpy(wide_a_, wide_in_host, uint64_t(T) * hc * H * 4);
    }
    if (trace) { q.wait(); std::fprintf(stderr, "[trace s%u] embed/wide-in %.2fs\n", layer_lo_, since()); }

    if (spec_verify_) { spec_T_ = T; spec_pos_ = start_pos; }
    // Diagnostic (IE_HY4_DUMP_WIDE=<path>): cross-process determinism bisect —
    // stage this forward's residual after every block's attn mix and ffn mix
    // (run_block pushes async D2H copies on the in-order queue), then append
    // the records to <path>.<layer_lo_>: header {L, stage, T, start_pos, n}
    // + n floats each. The host write lands after the forward's final wait.
    static const char* dump_path = std::getenv("IE_HY4_DUMP_WIDE");
    if (dump_path) {
        dump_h_.assign(uint64_t(2) * (layer_hi_ - layer_lo_) * T * hc * H, 0.f);
        dump_cur_ = 0;
        dump_hdr_.clear();
    }
    for (uint32_t L = layer_lo_; L < layer_hi_; ++L) {
        if (auto e = run_block(L, T, start_pos); !e.empty()) return e;
        {   // Diagnostic: IE_HY4_NAN_PROBE=1 reports the first layer whose
            // residual stream goes non-finite (decode T=1 NaN hunt, 2026-09-01).
            static const bool nan_probe = std::getenv("IE_HY4_NAN_PROBE") != nullptr;
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
        if (!dmoe_hdr_.empty()) {   // IE_HY4_DUMP_MOE records (see run_moe)
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
        q.wait();
        return {};
    }
    if (layer_hi_ != n_tf) return "hyv4 forward_range: mid-pipe stage needs wide_out";

    // final merge: unweighted stream mean -> output_norm -> lm_head
    ds4_hyper_head(q, wide_a_, hh_fn_, hh_base_, hh_scale_, mean_, T, H, hc,
                   cfg_.rms_eps, cfg_.hc_eps, {});
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
    q.wait();
    return {};
}

}  // namespace ie
